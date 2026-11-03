"""Sample stage process trees and retain bounded, hardware-specific feedback."""

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import threading
import time
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

from shared.cache_lock import cache_lock


@lru_cache(maxsize=1)
def _implementation():
    root = Path(__file__).resolve().parents[1]
    sources = ('shared/cohort_profile.py', 'shared/cohort_planning.py', 'shared/cohort_schedule.py',
               'shared/cohort_policy.py', 'shared/cohort_resources.py', 'shared/compute_schedule.py',
               'shared/compute_tuning.py', 'shared/numeric_threads.py', 'shared/cohort_targets.py',
               'scripts/autotune_batch.py',
               'microstructure/pipeline.py', 'microstructure/dki.py', 'microstructure/mkcurve.py',
               'microstructure/mkcurve_tensor.py', 'microstructure/mkcurve_inputs.py', 'microstructure/chunk_dispatch.py',
               'microstructure/freewater_dispatch.py', 'microstructure/freewater_wls.py',
               'microstructure/freewater_protocol.py', 'tractography/wma.py',
               'tractography/wma_registration.py', 'tractography/wma_transform.py')
    result = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources}
    for name in ('numpy', 'scipy', 'dipy', 'torch', 'vtk', 'dmri-amico'):
        try:
            result['package:' + name] = version(name)
        except PackageNotFoundError:
            result['package:' + name] = None
    return result


def workload_key(stage, shape, arguments, cpu, gpu):
    # Match numerical settings, data dimensions, CPU reservation and implementation.
    payload = dict(stage=stage, shape=shape, arguments=arguments, cpu=cpu, gpu=gpu,
                   machine=platform.machine(), processor=platform.processor(), node=platform.node(),
                   python=platform.python_version(), code=_implementation())
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def read_history(path):
    if not path or not Path(path).is_file():
        return {}
    try:
        if Path(path).stat().st_size > 8 * 1024**2:
            return {}
        result = json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(result, dict) or result.get('version') != 1 or not isinstance(result.get('profiles'), dict):
            return {}
        result['profiles'] = {key: [row for row in rows if isinstance(row, dict)][-12:]
                              for key, rows in result['profiles'].items() if isinstance(rows, list)}
        return result
    except (OSError, ValueError):
        return {}


def history_estimate(history, key):
    rows = history.get('profiles', {}).get(key, [])[-12:]
    rows = [row for row in rows if row.get('outcome') == 'completed' and not row.get('cache_hit') and row.get('samples', 0) >= 2
            and all(isinstance(row.get(field), (int, float)) and math.isfinite(row[field]) and row[field] > 0
                    for field in ('peak_tree_rss_bytes', 'execution_seconds'))]
    if not rows:
        return None
    return dict(samples=len(rows), memory_gb=max(row['peak_tree_rss_bytes'] for row in rows) / 1024**3 * 1.3,
                duration_seconds=max(row['execution_seconds'] for row in rows) * 1.25)


def record_history(path, key, measurement):
    if not path or not key:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with cache_lock(path.with_suffix('.lock')):
        history = read_history(path) or dict(version=1, profiles={})
        profiles = history.setdefault('profiles', {})
        profiles[key] = (profiles.get(key, []) + [measurement])[-12:]
        # Preserve recent workload keys rather than growing indefinitely.
        while len(profiles) > 512:
            del profiles[next(iter(profiles))]
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(history, indent=2) + '\n', encoding='utf-8')
        tmp.replace(path)


class StageMonitor:
    """Sample RSS sums (shared pages may be counted twice), not OS peak enforcement."""
    def __init__(self, interval=.2):
        self.interval = interval
        self.stop_event = threading.Event()
        self.peak = 0
        self.samples = 0
        self.cpu = {}
        self.io = {}
        self.start_cpu = {}
        self.start_io = {}
        self.started = time.monotonic()
        self.thread = None
        self.latest = {}

    def sample(self):
        import psutil
        try:
            owner = psutil.Process(os.getpid())
            processes = [owner, *owner.children(recursive=True)]
        except psutil.Error:
            return
        rss = 0
        for process in processes:
            try:
                ident = (process.pid, process.create_time())
                rss += process.memory_info().rss
                times = process.cpu_times()
                current = times.user + times.system
                self.start_cpu.setdefault(ident, current if self.samples == 0 else 0.)
                self.cpu[ident] = current
                try:
                    counters = process.io_counters()
                    current_io = counters.read_bytes + counters.write_bytes
                    self.start_io.setdefault(ident, current_io if self.samples == 0 else 0)
                    self.io[ident] = current_io
                except (psutil.Error, AttributeError):
                    pass
            except psutil.Error:
                continue
        self.peak = max(self.peak, rss)
        self.samples += 1
        self.latest = dict(rss_bytes=rss, process_count=len(processes),
                           pids=[process.pid for process in processes],
                           cpu_seconds=sum(max(0, value-self.start_cpu[key]) for key, value in self.cpu.items()),
                           io_bytes=sum(max(0, value-self.start_io[key]) for key, value in self.io.items()))

    def start(self):
        self.sample()
        def run():
            while not self.stop_event.wait(self.interval):
                self.sample()
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        return self

    def finish(self, outcome):
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        self.sample()
        elapsed = time.monotonic() - self.started
        cpu = sum(max(0, value - self.start_cpu[key]) for key, value in self.cpu.items())
        return dict(outcome=outcome, samples=self.samples, execution_seconds=elapsed,
                    peak_tree_rss_bytes=self.peak, sampled_cpu_seconds=cpu,
                    mean_cpu_cores=cpu / max(elapsed, 1e-9),
                    sampled_io_bytes=sum(max(0, value - self.start_io[key]) for key, value in self.io.items()),
                    interval_seconds=self.interval,
                    scope='Sampled owner plus descendants; short-lived processes and between-sample peaks may be missed')
