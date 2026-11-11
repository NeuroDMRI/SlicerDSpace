"""Ops helpers for the T3 paired cohort benchmark: launch rule, samplers, arm arguments, metrics.

Ops code (ops-runner tier): it runs existing tools and adds no science decision. Nothing here deletes
or overwrites files; output folders are claimed with ``os.mkdir`` and an existing folder is refused.
"""

import csv
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time

T3_ROUTES = ('cpu', 'hybrid', 'anatomy', 'tractography', 'regional_ifw', 'end_to_end')
ARM_CPUS = 32
ARM_THREADS = 8
SUBJECT_JOBS = ARM_CPUS // ARM_THREADS          # S4: --jobs 4 = 32 // 8
BATCH_MEMORY_GB = 224.0
KILL_MEMORY_GB = 256.0
PHYSICAL_CORES = 16
EXCLUDED_CPUS = frozenset({0, 1, 2, 3, 56, 57, 58, 59})   # cores 0-3 and their siblings (S4)
MAX_BUSY_PERCENT = 25.0
MIN_FREE_GPU_MIB = 14 * 1024
# SPEC-GAP: S4 says "500 GB"; decimal gigabytes are the smaller reading.
MIN_FREE_DISK_BYTES = 500 * 10**9
MAX_ATTEMPTS = 8
RETRY_WAIT_SECONDS = 15 * 60
SAMPLE_SECONDS = 300
SAMPLE_INTERVAL = 10
GPU_SAMPLE_SECONDS = 5
HOST_SAMPLE_SECONDS = 10
THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', 'MRTRIX_NTHREADS')


class LaunchRefused(RuntimeError):
    def __init__(self, attempts):
        super().__init__(f'Launch rule failed {len(attempts)} times; stop and report')
        self.attempts = attempts


# ---------------------------------------------------------------- validation

def safe_component(value, what='name'):
    """One path component: not empty, not '.' or '..', no separator."""
    if (not isinstance(value, str) or not value or value in ('.', '..') or value != value.strip()
            or '/' in value or '\\' in value or '\0' in value):
        raise ValueError(f'{what} must be one path component, got {value!r}')
    return value


def parse_cpu_list(text, expected=None):
    """Parse '4-7,60' into a sorted unique list of non-negative ints."""
    cpus = []
    for part in str(text).split(','):
        part = part.strip()
        if not part:
            raise ValueError('Empty entry in CPU list')
        low, _, high = part.partition('-')
        first, last = int(low), int(high or low)
        if first < 0 or last < first:
            raise ValueError('Bad CPU range: ' + part)
        cpus.extend(range(first, last + 1))
    if len(cpus) != len(set(cpus)):
        raise ValueError('CPU list repeats a CPU')
    if expected is not None and len(cpus) != expected:
        raise ValueError(f'CPU list has {len(cpus)} CPUs; expected {expected}')
    return sorted(cpus)


def claim_folder(path):
    """Create a new folder atomically; refuse an existing one. Never deletes."""
    safe_component(Path(path).name, 'output folder name')
    os.mkdir(path)
    return Path(path)


# ---------------------------------------------------------------- launch rule

def parse_mpstat(text):
    """Mean busy % (100 - %idle) per logical CPU over the interval rows; 'Average:' rows are ignored."""
    sums, counts = {}, {}
    cpu_column = idle_column = None
    for line in text.splitlines():
        tokens = line.split()
        if not tokens:
            continue
        if 'CPU' in tokens and '%idle' in tokens:
            cpu_column, idle_column = tokens.index('CPU'), tokens.index('%idle')
            continue
        if tokens[0] == 'Average:' or cpu_column is None or len(tokens) <= max(cpu_column, idle_column):
            continue
        label = tokens[cpu_column]
        if not label.isdigit():
            continue
        try:
            idle = float(tokens[idle_column])
        except ValueError:
            continue
        cpu = int(label)
        sums[cpu] = sums.get(cpu, 0.) + 100. - idle
        counts[cpu] = counts.get(cpu, 0) + 1
    if not sums:
        raise ValueError('mpstat output has no per-CPU rows')
    return {cpu: sums[cpu] / counts[cpu] for cpu in sums}


def parse_cpulist(text):
    cpus = []
    for part in text.strip().split(','):
        if part:
            low, _, high = part.partition('-')
            cpus.extend(range(int(low), int(high or low) + 1))
    return cpus


def read_topology(root='/sys/devices/system'):
    """{node: [physical core as a tuple of its logical CPUs]} from sysfs."""
    root = Path(root)
    nodes = {}
    for directory in sorted((root / 'node').glob('node[0-9]*')):
        node = int(directory.name[4:])
        members = parse_cpulist((directory / 'cpulist').read_text())
        seen, cores = set(), []
        for cpu in members:
            if cpu in seen:
                continue
            siblings = parse_cpulist((root / f'cpu/cpu{cpu}/topology/thread_siblings_list').read_text())
            core = tuple(sorted(item for item in siblings if item in members)) or (cpu,)
            seen.update(core)
            cores.append(core)
        nodes[node] = cores
    return nodes


def choose_cores(busy, topology, *, excluded=EXCLUDED_CPUS, count=PHYSICAL_CORES):
    """Per node: the `count` physical cores with the lowest summed busy % of their siblings."""
    chosen = {}
    for node, cores in topology.items():
        eligible = [core for core in cores if not set(core) & set(excluded)]
        if len(eligible) < count:
            raise ValueError(f'node{node} has {len(eligible)} usable physical cores; need {count}')
        missing = [cpu for core in eligible for cpu in core if cpu not in busy]
        if missing:
            raise ValueError(f'No busy sample for CPUs {missing[:4]}')
        ranked = sorted(eligible, key=lambda core: (sum(busy[cpu] for cpu in core), core))[:count]
        cpus = sorted(cpu for core in ranked for cpu in core)
        chosen[node] = dict(cores=[list(core) for core in sorted(ranked)], cpus=cpus,
                            mean_busy_percent=sum(busy[cpu] for cpu in cpus) / len(cpus))
    return chosen


def choose_gpus(samples, *, count=2, min_free_mib=MIN_FREE_GPU_MIB):
    """GPUs with at least `min_free_mib` free in every sample; the `count` lowest by mean utilisation."""
    if not samples or any(not sample for sample in samples):
        return [], {}
    indices = set.intersection(*({gpu['index'] for gpu in sample} for sample in samples))
    stats = {}
    for index in sorted(indices):
        rows = [gpu for sample in samples for gpu in sample if gpu['index'] == index]
        stats[index] = dict(min_free_mib=min(row['total_mib'] - row['used_mib'] for row in rows),
                            mean_utilization_percent=sum(row['utilization_percent'] for row in rows) / len(rows),
                            samples=len(rows))
    eligible = sorted((index for index, row in stats.items() if row['min_free_mib'] >= min_free_mib),
                      key=lambda index: (stats[index]['mean_utilization_percent'], index))
    return eligible[:count], stats


def decide_launch(busy, topology, gpu_samples, disk_free_bytes, *, hybrid, max_busy_percent=MAX_BUSY_PERCENT):
    """Apply S4 launch-rule steps 2 and 3. Returns the decision and every reason for a refusal.
    `max_busy_percent`: the step-3 CPU limit; the design value is 25 (pair `--max-busy-percent`, scope change SC-M4a)."""
    reasons = []
    cores = choose_cores(busy, topology)
    for node in (0, 1):
        if node not in cores:
            reasons.append(f'node{node} missing from the topology')
        elif cores[node]['mean_busy_percent'] > max_busy_percent:
            reasons.append(f'node{node} chosen set busy {cores[node]["mean_busy_percent"]:.1f} % > {max_busy_percent:g} %')
    gpus, gpu_stats = choose_gpus(gpu_samples)
    if hybrid and len(gpus) < 2:
        reasons.append(f'only {len(gpus)} GPU(s) have >= 14 GiB free in every sample; need 2')
    if disk_free_bytes < MIN_FREE_DISK_BYTES:
        reasons.append(f'/data04 has {disk_free_bytes / 1e9:.0f} GB free < 500 GB')
    return dict(ok=not reasons, reasons=reasons, cores=cores, gpus=gpus if hybrid else [],
                gpu_candidates=gpus, gpu_stats=gpu_stats, disk_free_bytes=disk_free_bytes)


def assign_arms(decision, repeat, *, previous=None):
    """Repeat 1: Subject on node0, Adaptive on node1; repeat 2 swaps (and the GPUs when the same two are chosen)."""
    if repeat not in (1, 2):
        raise ValueError('repeat must be 1 or 2')
    subject_node, adaptive_node = (0, 1) if repeat == 1 else (1, 0)
    gpus = list(decision.get('gpus') or [])
    subject_gpu = adaptive_gpu = None
    if gpus:
        subject_gpu, adaptive_gpu = gpus[0], gpus[1]
        if previous and {previous.get('subject_gpu'), previous.get('adaptive_gpu')} == set(gpus):
            subject_gpu, adaptive_gpu = previous['adaptive_gpu'], previous['subject_gpu']
    return dict(subject=dict(node=subject_node, cpus=decision['cores'][subject_node]['cpus'], gpu=subject_gpu),
                adaptive=dict(node=adaptive_node, cpus=decision['cores'][adaptive_node]['cpus'], gpu=adaptive_gpu))


def run_launch_rule(attempt, *, max_attempts=MAX_ATTEMPTS, wait_seconds=RETRY_WAIT_SECONDS,
                    sleep=time.sleep, save=None):
    """Steps 1-4: up to `max_attempts` attempts, `wait_seconds` apart. `attempt(n)` returns a dict with `ok`."""
    attempts = []
    for number in range(1, max_attempts + 1):
        outcome = dict(attempt(number), attempt=number)
        attempts.append(outcome)
        if save:
            save(attempts)
        if outcome['ok']:
            return outcome, attempts
        if number < max_attempts:
            sleep(wait_seconds)
    raise LaunchRefused(attempts)


def kill_group(process, *, sig=getattr(signal, 'SIGKILL', signal.SIGTERM), wait=10):
    """Signal a process group started with start_new_session and reap its leader."""
    try:
        if hasattr(os, 'killpg'):
            os.killpg(process.pid, sig)
        else:
            process.kill()
    except (ProcessLookupError, PermissionError):
        pass
    try:
        process.wait(timeout=wait)
    except subprocess.TimeoutExpired:
        pass


def query_gpus(smi='nvidia-smi', run=subprocess.run):
    result = run([smi, '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                  '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15, check=True)
    return [dict(index=int(row[0]), used_mib=float(row[1]), total_mib=float(row[2]), utilization_percent=float(row[3]))
            for row in csv.reader(io.StringIO(result.stdout), skipinitialspace=True) if row]


def sample_window(folder, attempt, *, duration=SAMPLE_SECONDS, interval=SAMPLE_INTERVAL, mpstat='mpstat',
                  smi='nvidia-smi', popen=subprocess.Popen, run=subprocess.run, clock=time.monotonic,
                  sleep=time.sleep, grace=60):
    """S4 step 1: `mpstat -P ALL 10 30` plus nvidia-smi every 10 s. Bounded by a deadline; mpstat is killed on exit."""
    raw = Path(folder) / f'mpstat_{attempt}.txt'
    gpu_samples, gpu_errors = [], []
    with raw.open('x') as handle:                        # file, not a pipe: 300 KB would block a pipe
        process = popen([mpstat, '-P', 'ALL', str(interval), str(max(1, duration // interval))], stdout=handle,
                        stderr=subprocess.STDOUT, env={**os.environ, 'LC_ALL': 'C'}, start_new_session=True)
    deadline = clock() + duration + grace
    try:
        while process.poll() is None:
            if clock() >= deadline:
                raise TimeoutError('mpstat did not finish before its deadline')
            try:
                gpu_samples.append(query_gpus(smi, run))
            except Exception as exc:                    # nvidia-smi problems matter only for the hybrid workload
                gpu_errors.append(f'{type(exc).__name__}: {exc}')
            sleep(interval)
    finally:
        if process.poll() is None:
            kill_group(process)
    return parse_mpstat(raw.read_text()), gpu_samples, gpu_errors


def run_bounded(command, *, timeout, env=None, popen=subprocess.Popen):
    """Run a command in its own session with a deadline; kill the whole group on timeout or error."""
    process = popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
                    start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except BaseException:
        kill_group(process)
        raise
    return process.returncode, stdout, stderr


def parse_doctor_output(stdout):
    """`--doctor` prints one JSON document (possibly after launcher noise); return it."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r'^\{', stdout, flags=re.MULTILINE):
        try:
            document, _ = decoder.raw_decode(stdout[match.start():])
        except ValueError:
            continue
        if isinstance(document, dict) and 'environment' in document:
            return document
    raise ValueError('Doctor output has no environment document')


def doctor_blockers(document):
    return int(document['environment']['summary']['blocker_count'])


# ---------------------------------------------------------------- samplers

def _clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


class Sampler:
    """Calls `read()` every `interval` seconds on a thread and appends one JSON line per sample."""

    def __init__(self, path, interval, read):
        self.path, self.interval, self.read = Path(path), interval, read
        self.stop_event = threading.Event()
        self.thread = None
        self.errors = []
        self.samples = 0

    def _loop(self):
        with self.path.open('a', encoding='utf-8') as stream:
            while True:
                try:
                    row = self.read()
                    if row:
                        stream.write(json.dumps(_clean(row), allow_nan=False) + '\n')
                        stream.flush()
                        self.samples += 1
                except Exception as exc:
                    self.errors.append(f'{type(exc).__name__}: {exc}')
                if self.stop_event.wait(self.interval):
                    return

    def start(self):
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def finish(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=self.interval + 30)
        return dict(path=str(self.path), samples=self.samples, errors=self.errors[:20])


def _number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def make_gpu_reader(index, pids, *, smi='nvidia-smi', run=subprocess.run, clock=time.time):
    """H5: utilisation, total used memory and our processes' memory of GPU `index` (PCI order)."""
    def query(kind, fields):
        result = run([smi, f'--query-{kind}={fields}', '--format=csv,noheader,nounits'],
                     capture_output=True, text=True, timeout=10, check=True)
        return [row for row in csv.reader(io.StringIO(result.stdout), skipinitialspace=True) if row]

    def read():
        device = next(row for row in query('gpu', 'index,uuid,memory.used,memory.total,utilization.gpu')
                      if int(row[0]) == index)
        mine = {int(pid) for pid in pids()}
        owned = sum(_number(row[1]) or 0. for row in query('compute-apps', 'pid,used_gpu_memory,gpu_uuid')
                    if row[2] == device[1] and _number(row[0]) is not None and int(row[0]) in mine)
        used = _number(device[2])
        return dict(time=clock(), gpu_index=index, utilization_percent=_number(device[4]), used_mib=used,
                    total_mib=_number(device[3]), owned_mib=owned,
                    other_mib=None if used is None else max(0., used - owned))
    return read


def read_proc_stat(text):
    """{cpu: (busy ticks, total ticks)} from /proc/stat text."""
    result = {}
    for line in text.splitlines():
        tokens = line.split()
        if tokens and re.fullmatch(r'cpu\d+', tokens[0]):
            values = [int(item) for item in tokens[1:9]]
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            result[int(tokens[0][3:])] = (sum(values) - idle, sum(values))
    return result


def read_diskstats(text, major, minor):
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 10 and int(fields[0]) == major and int(fields[1]) == minor:
            return dict(device=fields[2], reads=int(fields[3]), read_sectors=int(fields[5]),
                        writes=int(fields[7]), write_sectors=int(fields[9]))
    return None


def read_nfs_bytes(text, path):
    """Byte counters of the NFS mount that holds `path` (longest matching mount point), from mountstats."""
    best, current = None, None
    for line in text.splitlines():
        match = re.match(r'device \S+ mounted on (\S+) with fstype (nfs\d?)', line)
        if match:
            current = match.group(1)
            continue
        if current is not None and line.strip().startswith('bytes:'):
            mount = current.rstrip('/') or '/'
            if (path == mount or path.startswith(mount + '/')) and (best is None or len(mount) > len(best[0])):
                names = ('normal_read', 'normal_write', 'direct_read', 'direct_write',
                         'server_read', 'server_write', 'read_pages', 'write_pages')
                best = (mount, dict(zip(names, (int(item) for item in line.split()[1:9]))))
    return None if best is None else dict(mount=best[0], **best[1])


class HostReader:
    """H6: per-CPU busy % of the arm's set, process-tree CPU time, /data04 disk and NFS counters."""

    def __init__(self, cpus, *, tree_cpu_seconds, data_path, nfs_path, read_text=None, device_numbers=None,
                 clock=time.time):
        self.cpus, self.tree_cpu_seconds, self.clock = list(cpus), tree_cpu_seconds, clock
        self.read_text = read_text or (lambda name: Path(name).read_text())
        self.nfs_path = str(nfs_path)
        if device_numbers is None:
            try:
                status = os.stat(data_path).st_dev
                device_numbers = (os.major(status), os.minor(status))
            except (OSError, AttributeError):
                device_numbers = None
        self.device_numbers = device_numbers
        self.previous = None

    def __call__(self):
        counters = read_proc_stat(self.read_text('/proc/stat'))
        row = dict(time=self.clock(), tree_cpu_seconds=self.tree_cpu_seconds())
        percent = {}
        if self.previous is not None:
            for cpu in self.cpus:
                if cpu in counters and cpu in self.previous:
                    busy = counters[cpu][0] - self.previous[cpu][0]
                    total = counters[cpu][1] - self.previous[cpu][1]
                    percent[cpu] = 100. * busy / total if total > 0 else None
        self.previous = counters
        values = [item for item in percent.values() if item is not None]
        row.update(cpu_busy_percent=percent, set_busy_percent=sum(values) / len(values) if values else None)
        try:
            row['disk'] = read_diskstats(self.read_text('/proc/diskstats'), *self.device_numbers) \
                if self.device_numbers else None
        except (OSError, ValueError):
            row['disk'] = None
        try:
            row['nfs'] = read_nfs_bytes(self.read_text('/proc/self/mountstats'), self.nfs_path)
        except (OSError, ValueError):
            row['nfs'] = None
        return row


# ---------------------------------------------------------------- metrics

def read_jsonl(path):
    rows = []
    path = Path(path)
    if path.is_file():
        for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def _mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def gpu_summary(rows):
    return dict(samples=len(rows),
                mean_device_utilization_percent=_mean(row.get('utilization_percent') for row in rows),
                max_device_utilization_percent=max((row['utilization_percent'] for row in rows
                                                    if row.get('utilization_percent') is not None), default=None),
                peak_total_used_mib=max((row['used_mib'] for row in rows if row.get('used_mib') is not None), default=None),
                peak_own_vram_mib=max((row['owned_mib'] for row in rows if row.get('owned_mib') is not None), default=None),
                peak_other_users_mib=max((row['other_mib'] for row in rows if row.get('other_mib') is not None), default=None),
                scope='Whole-device utilisation includes other users; own VRAM is from compute-apps')


def _delta(rows, key, fields):
    values = [row[key] for row in rows if row.get(key)]
    if len(values) < 2:
        return None
    return {field: values[-1][field] - values[0][field] for field in fields if field in values[0]}


def host_summary(rows, cpu_count):
    if not rows:
        return dict(samples=0)
    elapsed = rows[-1]['time'] - rows[0]['time']
    set_busy = _mean(row.get('set_busy_percent') for row in rows)
    own = None
    if elapsed > 0 and cpu_count and rows[-1].get('tree_cpu_seconds') is not None \
            and rows[0].get('tree_cpu_seconds') is not None:
        own = 100. * (rows[-1]['tree_cpu_seconds'] - rows[0]['tree_cpu_seconds']) / (elapsed * cpu_count)
    return dict(samples=len(rows), mean_set_busy_percent=set_busy, own_tree_busy_percent=own,
                # Estimate: the tree sampler misses exited children, so this is an upper bound.
                background_busy_percent_estimate=None if set_busy is None or own is None else max(0., set_busy - own),
                data04_disk=_delta(rows, 'disk', ('reads', 'read_sectors', 'writes', 'write_sectors')),
                nfs=_delta(rows, 'nfs', ('normal_read', 'normal_write', 'direct_read', 'direct_write',
                                        'server_read', 'server_write')))


def schedule_events(folder):
    rows = []
    for path in sorted(Path(folder).glob('schedule_*/events.jsonl')):
        rows.extend(read_jsonl(path))
    return rows


def adaptive_stage_metrics(folder):
    """Adaptive-only: per-stage queue wait, reserved RAM against peak RSS, reserved-but-unused CPU."""
    stages = {}
    started = {}
    for row in schedule_events(folder):
        if row.get('event') == 'started':
            started[row['id']] = row
            entry = stages.setdefault(row['stage'], dict(waits=[], memory_reserved_gb=[], peak_rss_gb=[],
                                                          cpu_reserved_seconds=0., cpu_sampled_seconds=0.))
            entry['waits'].append(row.get('wait_seconds', 0.))
            entry['memory_reserved_gb'].append(row.get('memory_gb'))
        elif row.get('event') in ('completed', 'failed') and row['id'] in started:
            entry = stages[row['stage']]
            measurement = row.get('measurement') or {}
            if measurement.get('peak_tree_rss_bytes') is not None:
                entry['peak_rss_gb'].append(measurement['peak_tree_rss_bytes'] / 1024**3)
            seconds = row.get('time', 0) - started[row['id']].get('time', 0)
            entry['cpu_reserved_seconds'] += started[row['id']].get('cpu', 0) * seconds
            entry['cpu_sampled_seconds'] += measurement.get('sampled_cpu_seconds') or 0.
    return {stage: dict(starts=len(entry['waits']), mean_head_wait_seconds=_mean(entry['waits']),
                        max_head_wait_seconds=max(entry['waits']),
                        mean_memory_reserved_gb=_mean(entry['memory_reserved_gb']),
                        max_peak_rss_gb=max(entry['peak_rss_gb'], default=None),
                        reserved_but_unused_cpu_seconds=max(0., entry['cpu_reserved_seconds'] - entry['cpu_sampled_seconds']))
            for stage, entry in stages.items()}


def gate_hold_seconds(folder):
    """Sum of GPU-gate hold times of the Subject arm (`gpu_guard/gpu_events.jsonl`)."""
    rows = read_jsonl(Path(folder) / 'gpu_guard/gpu_events.jsonl')
    return sum(max(0., row['ended_at'] - row['started_at']) for row in rows
               if row.get('admitted') and row.get('started_at') and row.get('ended_at'))


def trial_metrics(trial, record, passed_subjects):
    """Metric table of one arm (S4 'Metrics')."""
    folder = Path(trial['output'])
    measurement = trial.get('measurement') or {}
    makespan = measurement.get('execution_seconds')
    return dict(
        policy=trial['policy'], returncode=trial.get('returncode'), reason=trial.get('reason'),
        makespan_seconds=makespan, subjects_passed_rule1=passed_subjects,
        throughput_subjects_per_hour=None if not makespan else passed_subjects / makespan * 3600.,
        mean_cpu_cores=measurement.get('mean_cpu_cores'),
        ram_peak_gib=None if measurement.get('peak_tree_rss_bytes') is None else measurement['peak_tree_rss_bytes'] / 1024**3,
        gpu=gpu_summary(read_jsonl(folder / 'gpu_samples.jsonl')),
        gpu_gate_hold_seconds=gate_hold_seconds(folder) if trial['policy'] == 'subject' else None,
        host=host_summary(read_jsonl(folder / 'host_samples.jsonl'), len(record.get('cpu_affinity') or [])),
        adaptive_only=adaptive_stage_metrics(folder) if trial['policy'] == 'adaptive' else None)
