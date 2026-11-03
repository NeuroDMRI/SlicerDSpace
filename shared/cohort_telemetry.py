"""Owned process-tree measurements and separately labelled whole-device context."""

import csv
import io
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time

from shared.cohort_profile import StageMonitor


def gpu_sample(pids, executable=None):
    executable = executable or shutil.which('nvidia-smi')
    if not executable:
        return dict(available=False, reason='nvidia-smi unavailable', owned_vram_mib=None, devices=[])
    def query(kind, fields):
        result = subprocess.run([executable, '--query-'+kind+'='+fields, '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=2, check=True)
        return list(csv.reader(io.StringIO(result.stdout), skipinitialspace=True))
    try:
        devices = [dict(uuid=row[0], utilization_percent=float(row[1]), used_mib=float(row[2]))
                   for row in query('gpu', 'uuid,utilization.gpu,memory.used')]
        apps = query('compute-apps', 'pid,used_gpu_memory,gpu_uuid')
        owned = sum(float(row[1]) for row in apps if int(row[0]) in pids)
        return dict(available=True, owned_vram_mib=owned, devices=devices,
                    utilization_scope='Whole device including unrelated users; not study-attributable GPU utilization')
    except (OSError, ValueError, IndexError, subprocess.SubprocessError) as exc:
        return dict(available=False, reason=str(exc), owned_vram_mib=None, devices=[])


class CohortTelemetry:
    def __init__(self, path, interval=1.):
        self.path = Path(path)
        self.interval = interval
        self.monitor = StageMonitor(interval)
        self.stop_event = threading.Event()
        self.thread = None
        self.errors = []
        self.peak_vram_mib = None
        self.started = time.monotonic()

    def sample(self):
        self.monitor.sample()
        row = dict(time=time.time(), elapsed_seconds=time.monotonic()-self.started, **self.monitor.latest)
        row['gpu'] = gpu_sample(set(row.pop('pids', [])))
        value = row['gpu']['owned_vram_mib']
        if value is not None:
            self.peak_vram_mib = max(self.peak_vram_mib or 0, value)
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')

    def start(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(self.path)
        def run():
            while not self.stop_event.is_set():
                try:
                    self.sample()
                except Exception as exc:
                    self.errors.append(str(exc))
                if self.stop_event.wait(self.interval):
                    break
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        return self

    def finish(self, outcome):
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        result = self.monitor.finish(outcome)
        result.update(trace=str(self.path), sampled_peak_owned_vram_mib=self.peak_vram_mib,
                      errors=self.errors, gpu_utilization_scope='Whole-device context only')
        return result
