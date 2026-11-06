"""Cross-process stage admission for one explicitly resource-bounded cohort."""

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
import json
import os
from pathlib import Path
import sys
import time
from uuid import uuid4
import warnings

from shared.cache_lock import cache_lock

ENV = 'SLICERDSPACE_COHORT_SCHEDULE'
LEASE = 'SLICERDSPACE_COHORT_LEASE'
_inside = ContextVar('cohort_stage_active', default=False)
_reservation = ContextVar('cohort_stage_reservation', default=None)


def mark_reused(stage):
    row = _reservation.get()
    if row is not None and row['stage'] == stage:
        row['cache_hit'] = True


def create_schedule(directory, *, cpu_budget, memory_gb, stage_cpus, stage_memory_gb, gpu_slots=1,
                    subject_profiles=None, planning=None, policy='first_fit', history_path=None, io_slots=2,
                    ready_order=None, cpu_affinity=None):
    import psutil
    values = (cpu_budget, memory_gb, stage_cpus, stage_memory_gb, gpu_slots, io_slots)
    import math
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('Resource budgets and per-stage reservations must be finite and positive')
    if stage_cpus > cpu_budget or stage_memory_gb > memory_gb:
        raise ValueError('A stage reservation exceeds the cohort budget')
    if policy not in ('first_fit', 'fair_backfill'):
        raise ValueError('Unknown cohort admission policy')
    if any(type(value) is not int for value in (cpu_budget,stage_cpus,gpu_slots,io_slots)):
        raise ValueError('CPU and slot budgets must be integer counts')
    if any(isinstance(value,bool) for value in (memory_gb,stage_memory_gb)):
        raise ValueError('Memory budgets must be numeric, not boolean')
    ready_order=ready_order or ('mem_thread' if policy=='fair_backfill' else 'fifo')
    if ready_order not in ('fifo','mem_thread'):
        raise ValueError('Unknown ready-task ordering')
    from shared.cohort_resources import validate_request
    limits=dict(cpu=cpu_budget,memory_gb=memory_gb,gpu=gpu_slots,io=io_slots)
    selected=(planning or {}).get('selected_stages')
    for profile in (subject_profiles or {}).values():
        for stage,request in profile.items():
            validate_request({'cpu':stage_cpus,'memory_gb':stage_memory_gb,**request},limits,
                             check_budget=selected is None or stage in selected)
    from shared.cohort_affinity import cpu_allocation
    affinity = cpu_allocation(cpu_budget, cpu_affinity)
    folder = Path(directory).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    config = dict(kind='cohort_resource_schedule', version=1,
                  cpu_budget=cpu_budget, memory_gb=memory_gb, stage_cpus=stage_cpus,
                  stage_memory_gb=stage_memory_gb, gpu_slots=gpu_slots,
                  subject_profiles=subject_profiles or {}, planning=planning,
                  policy=policy, history_path=str(history_path) if history_path else None, io_slots=io_slots,
                  ready_order=ready_order,
                  cpu_affinity=affinity[:int(cpu_budget)],
                  scope='Linux stage CPU affinity; declared RAM/GPU reservations, not cgroup or per-device VRAM limits',
                  parent_pid=os.getpid(), parent_created=psutil.Process().create_time())
    (folder / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    (folder / 'state.json').write_text(json.dumps(dict(waiting=[], active=[])))
    return folder


def _alive(row):
    import psutil
    try:
        proc = psutil.Process(row['pid'])
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE and proc.create_time() == row['created']
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True


def _event(folder, row, event, **extra):
    with (folder / 'events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({**row, 'event': event, 'time': time.time(), **extra}) + '\n')


def _write(folder, state):
    tmp = folder / 'state.tmp'
    tmp.write_text(json.dumps(state))
    tmp.replace(folder / 'state.json')


def schedule_summary(folder):
    """Summarise declared reservations and sampled execution, separately from QC."""
    folder = Path(folder)
    path = folder / 'events.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.is_file() else []
    active, peaks = {}, dict(cpu=0, memory_gb=0, gpu=0, io=0)
    terminal = []
    waits = []
    for row in rows:
        if row['event'] == 'started':
            active[row['id']] = row
            waits.append(row.get('wait_seconds', 0))
            for key in peaks:
                peaks[key] = max(peaks[key], sum(item.get(key, 0) for item in active.values()))
        elif row['event'] in ('completed', 'failed', 'cancelled', 'abandoned'):
            active.pop(row['id'], None)
            terminal.append(row)
    measurements = [row['measurement'] for row in terminal if row.get('measurement')]
    return dict(peak_declared_reservations=peaks, stage_outcomes={
        name: sum(row['event'] == name for row in terminal) for name in ('completed', 'failed', 'cancelled', 'abandoned')},
        queue_wait_seconds=sum(waits), sampled_cpu_seconds=sum(item['sampled_cpu_seconds'] for item in measurements),
        max_sampled_single_stage_tree_rss_bytes=max((item['peak_tree_rss_bytes'] for item in measurements), default=0),
        measurement_scope='Per-stage process-tree samples; single-stage RSS maximum is not simultaneous cohort peak RAM')


def _inherited_lease(folder):
    """Only descendants of a currently admitted owner may reuse its reservation."""
    import psutil
    token = os.environ.get(LEASE)
    if not token:
        return False
    ancestors = {process.pid for process in psutil.Process().parents()}
    with cache_lock(folder / 'lock'):
        state = json.loads((folder / 'state.json').read_text())
        return any(row['id'] == token and row['pid'] in ancestors and _alive(row) for row in state['active'])


@contextmanager
def stage_admission(name, *, gpu=False):
    selected = os.environ.get(ENV)
    if not selected or _inside.get():
        yield
        return
    folder = Path(selected)
    import psutil
    config = json.loads((folder / 'config.json').read_text())
    parent=dict(pid=config['parent_pid'],created=config['parent_created'])
    if not _alive(parent):
        raise InterruptedError('Cohort coordinator exited; no new computation admitted')
    if _inherited_lease(folder):
        yield
        return
    subject = os.environ.get('SLICERDSPACE_COHORT_SUBJECT', '')
    from shared.cohort_resources import stage_request, validate_request
    request = stage_request(config,subject,name)
    row = dict(id=uuid4().hex, stage=name, subject=os.environ.get('SLICERDSPACE_COHORT_SUBJECT', ''),
               pid=os.getpid(), created=psutil.Process().create_time(), cpu=request.get('cpu', config['stage_cpus']),
               memory_gb=request.get('memory_gb', config['stage_memory_gb']), gpu=int(gpu), queued_at=time.time())
    row.update(io=request.get('io', 0), duration_seconds=request.get('duration_seconds', 0),
               profile_key=request.get('profile_key'), bypasses=0)
    row.update(estimate_source=request.get('estimate_source','header heuristic'),
               history_samples=request.get('history_samples',0),memory_floor_gb=request['memory_floor_gb'])
    if request.get('calibration'):
        row['calibration'] = request['calibration']
    validate_request(row,dict(cpu=config['cpu_budget'],memory_gb=config['memory_gb'],gpu=config['gpu_slots'],io=config['io_slots']))
    admitted, queued = False, False
    began = time.monotonic()
    token, reservation_token, previous = None, None, os.environ.get(LEASE)
    thread_keys = ('SLICERDSPACE_STAGE_CPUS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                   'MKL_NUM_THREADS', 'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', 'MRTRIX_NTHREADS')
    previous_threads = {key: os.environ.get(key) for key in thread_keys}
    outcome = 'failed'
    monitor = None
    try:
        while not admitted:
            if not _alive(parent):
                raise InterruptedError('Cohort coordinator exited while waiting for resources')
            with cache_lock(folder / 'lock'):
                state = json.loads((folder / 'state.json').read_text())
                for key in ('active', 'waiting'):
                    for dead in [item for item in state[key] if not _alive(item)]:
                        _event(folder, dead, 'abandoned')
                        state[key].remove(dead)
                if not queued:
                    state['waiting'].append(row)
                    _event(folder, row, 'queued')
                    queued = True
                used = {key: sum(item.get(key, 0) for item in state['active']) for key in ('cpu', 'memory_gb', 'gpu', 'io')}
                limits = dict(cpu=config['cpu_budget'], memory_gb=config['memory_gb'], gpu=config['gpu_slots'], io=config['io_slots'])
                # First fitting request: CPU work may pass a GPU waiter, but not an earlier fitting job.
                if config['policy'] == 'fair_backfill':
                    from shared.cohort_policy import choose_ready
                    candidate = choose_ready(state['waiting'], state['active'], limits, time.time(),
                                             ready_order=config.get('ready_order','fifo'))
                else:
                    candidate = next((item for item in state['waiting']
                                      if all(used[key] + item.get(key, 0) <= limits[key] for key in used)), None)
                if candidate and candidate['id'] == row['id']:
                    for earlier in state['waiting'][:state['waiting'].index(candidate)]:
                        earlier['bypasses'] = earlier.get('bypasses', 0) + 1
                    state['waiting'].remove(candidate)
                    row.update(candidate)
                    if config.get('cpu_affinity'):
                        used_cpus = {cpu for item in state['active'] for cpu in item.get('cpu_affinity', [])}
                        row['cpu_affinity'] = [cpu for cpu in config['cpu_affinity'] if cpu not in used_cpus][:int(row['cpu'])]
                    row['started_at'] = time.time()
                    state['active'].append(row)
                    admitted = True
                    _event(folder, row, 'started', wait_seconds=time.monotonic() - began)
                _write(folder, state)
            if not admitted:
                time.sleep(.05)
        token = _inside.set(True)
        reservation_token = _reservation.set(row)
        if config['policy'] == 'fair_backfill':
            from shared.cohort_profile import StageMonitor
            monitor = StageMonitor().start()
        os.environ[LEASE] = row['id']
        os.environ.update({key: str(row['cpu']) for key in thread_keys})
        try:
            from threadpoolctl import threadpool_limits
            thread_limit = threadpool_limits(limits=row['cpu'])
        except ImportError:
            thread_limit = nullcontext()
        from shared.cohort_affinity import cpu_placement
        with thread_limit, cpu_placement(row.get('cpu_affinity')):
            yield row
        outcome = 'completed'
    finally:
        measurement = None
        if monitor is not None:
            try:
                measurement = monitor.finish(outcome)
            except Exception as exc:
                warnings.warn('Resource measurement failed: ' + str(exc), RuntimeWarning)
        if measurement is not None:
            measurement['cache_hit'] = bool(row.get('cache_hit'))
        if token is not None:
            _inside.reset(token)
            _reservation.reset(reservation_token)
            for key, value in previous_threads.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            if previous is None:
                os.environ.pop(LEASE, None)
            else:
                os.environ[LEASE] = previous
        with cache_lock(folder / 'lock'):
            state = json.loads((folder / 'state.json').read_text())
            for key in ('active', 'waiting'):
                state[key] = [item for item in state[key] if item['id'] != row['id']]
            _event(folder, row, outcome if admitted else 'cancelled', elapsed_seconds=time.monotonic() - began,
                   measurement=measurement)
            _write(folder, state)
        if measurement is not None:
            from shared.cohort_profile import record_history
            try:
                record_history(config.get('history_path'), row['profile_key'], measurement)
            except (OSError, ValueError) as exc:
                warnings.warn('Resource history could not be saved: ' + str(exc), RuntimeWarning)


def scheduled_stage(name, *, gpu=False):
    """No-op outside the opt-in cohort mode; preserve arguments and return values."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            requested = gpu
            config = kwargs.get('config') or kwargs.get('cfg') or (args[0] if args else None)
            if str(getattr(config, 'device', 'auto')).lower() == 'cpu':
                requested = False
            stage = name(config) if callable(name) else name
            from shared.cohort_io import gpu_admission
            with stage_admission(stage, gpu=requested) as reservation, gpu_admission(stage, requested):
                try:
                    result = function(*args, **kwargs)
                    status = result.get('status') if isinstance(result, dict) else getattr(result, 'status', None)
                    if reservation is not None and status in ('cached', 'reused'):
                        reservation['cache_hit'] = True
                    return result
                finally:
                    # A finished in-process inference must not retain unused CUDA
                    # allocator blocks while another participant takes the slot.
                    torch = sys.modules.get('torch') if requested and os.environ.get(ENV) else None
                    cuda = getattr(torch, 'cuda', None)
                    if cuda is not None and cuda.is_initialized():
                        for device in range(cuda.device_count()):
                            if cuda.memory_reserved(device):
                                with cuda.device(device):
                                    cuda.synchronize()
                                    cuda.empty_cache()
        return wrapped
    return decorate
