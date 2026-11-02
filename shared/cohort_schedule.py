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

from shared.cache_lock import cache_lock

ENV = 'SLICERDSPACE_COHORT_SCHEDULE'
LEASE = 'SLICERDSPACE_COHORT_LEASE'
_inside = ContextVar('cohort_stage_active', default=False)


def create_schedule(directory, *, cpu_budget, memory_gb, stage_cpus, stage_memory_gb, gpu_slots=1,
                    subject_profiles=None, planning=None):
    import psutil
    values = (cpu_budget, memory_gb, stage_cpus, stage_memory_gb, gpu_slots)
    import math
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('Resource budgets and per-stage reservations must be finite and positive')
    if stage_cpus > cpu_budget or stage_memory_gb > memory_gb:
        raise ValueError('A stage reservation exceeds the cohort budget')
    folder = Path(directory).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    config = dict(kind='cohort_resource_schedule', version=1,
                  cpu_budget=cpu_budget, memory_gb=memory_gb, stage_cpus=stage_cpus,
                  stage_memory_gb=stage_memory_gb, gpu_slots=gpu_slots,
                  subject_profiles=subject_profiles or {}, planning=planning,
                  scope='Declared stage reservations, not OS-enforced CPU/RAM or per-device VRAM limits',
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


def _event(folder, row, event, **extra):
    with (folder / 'events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({**row, 'event': event, 'time': time.time(), **extra}) + '\n')


def _write(folder, state):
    tmp = folder / 'state.tmp'
    tmp.write_text(json.dumps(state))
    tmp.replace(folder / 'state.json')


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
    if _inherited_lease(folder):
        yield
        return
    import psutil
    config = json.loads((folder / 'config.json').read_text())
    subject = os.environ.get('SLICERDSPACE_COHORT_SUBJECT', '')
    request = config.get('subject_profiles', {}).get(subject, {}).get(name, {})
    row = dict(id=uuid4().hex, stage=name, subject=os.environ.get('SLICERDSPACE_COHORT_SUBJECT', ''),
               pid=os.getpid(), created=psutil.Process().create_time(), cpu=request.get('cpu', config['stage_cpus']),
               memory_gb=request.get('memory_gb', config['stage_memory_gb']), gpu=int(gpu), queued_at=time.time())
    if row['cpu'] > config['cpu_budget'] or row['memory_gb'] > config['memory_gb']:
        raise ValueError(f'{subject}/{name}: stage estimate exceeds the cohort resource budget')
    admitted, queued = False, False
    began = time.monotonic()
    token, previous = None, os.environ.get(LEASE)
    thread_keys = ('SLICERDSPACE_STAGE_CPUS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                   'MKL_NUM_THREADS', 'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', 'MRTRIX_NTHREADS')
    previous_threads = {key: os.environ.get(key) for key in thread_keys}
    outcome = 'failed'
    try:
        while not admitted:
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
                used = {key: sum(item[key] for item in state['active']) for key in ('cpu', 'memory_gb', 'gpu')}
                limits = dict(cpu=config['cpu_budget'], memory_gb=config['memory_gb'], gpu=config['gpu_slots'])
                # First fitting request: CPU work may pass a GPU waiter, but not an earlier fitting job.
                candidate = next((item for item in state['waiting']
                                  if all(used[key] + item[key] <= limits[key] for key in used)), None)
                if candidate and candidate['id'] == row['id']:
                    state['waiting'].remove(row)
                    state['active'].append(row)
                    admitted = True
                    _event(folder, row, 'started', wait_seconds=time.monotonic() - began)
                _write(folder, state)
            if not admitted:
                time.sleep(.05)
        token = _inside.set(True)
        os.environ[LEASE] = row['id']
        os.environ.update({key: str(row['cpu']) for key in thread_keys})
        try:
            from threadpoolctl import threadpool_limits
            thread_limit = threadpool_limits(limits=row['cpu'])
        except ImportError:
            thread_limit = nullcontext()
        with thread_limit:
            yield
        outcome = 'completed'
    finally:
        if token is not None:
            _inside.reset(token)
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
            _event(folder, row, outcome if admitted else 'cancelled', elapsed_seconds=time.monotonic() - began)
            _write(folder, state)


def scheduled_stage(name, *, gpu=False):
    """No-op outside the opt-in cohort mode; preserve arguments and return values."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            requested = gpu
            config = kwargs.get('config') or kwargs.get('cfg') or (args[0] if args else None)
            if str(getattr(config, 'device', 'auto')).lower() == 'cpu':
                requested = False
            with stage_admission(name, gpu=requested):
                try:
                    return function(*args, **kwargs)
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
