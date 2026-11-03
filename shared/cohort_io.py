"""Short I/O permits, independent of a model's longer CPU/RAM reservation."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
from uuid import uuid4

from shared.cache_lock import cache_lock
from shared.cohort_schedule import ENV, _alive


GPU_ENV = 'SLICERDSPACE_COHORT_GPU_GUARD'


@contextmanager
def _admission(phase, folder, resource):
    if not folder:
        yield
        return
    import psutil
    folder = Path(folder)
    config = json.loads((folder/'config.json').read_text())
    path = folder/(resource+'_state.json')
    row = dict(id=uuid4().hex, pid=os.getpid(), created=psutil.Process().create_time(),
               subject=os.environ.get('SLICERDSPACE_COHORT_SUBJECT', ''), phase=phase,
               queued_at=time.time())
    admitted = queued = False
    try:
        while not admitted:
            with cache_lock(folder/(resource+'_lock')):
                state = json.loads(path.read_text()) if path.exists() else dict(active=[], waiting=[])
                for key in state:
                    state[key] = [item for item in state[key] if _alive(item)]
                if not queued:
                    state['waiting'].append(row)
                    queued = True
                if len(state['active']) < config[resource+'_slots'] and state['waiting'][0]['id'] == row['id']:
                    state['waiting'].pop(0)
                    state['active'].append(row)
                    admitted = True
                    row['started_at'] = time.time()
                temporary = path.with_suffix('.tmp')
                temporary.write_text(json.dumps(state))
                temporary.replace(path)
            if not admitted:
                time.sleep(.05)
        yield
    finally:
        with cache_lock(folder/(resource+'_lock')):
            state = json.loads(path.read_text()) if path.exists() else dict(active=[], waiting=[])
            for key in state:
                state[key] = [item for item in state[key] if item['id'] != row['id']]
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(state))
            temporary.replace(path)
            with (folder/(resource+'_events.jsonl')).open('a') as stream:
                stream.write(json.dumps({**row, 'ended_at':time.time(), 'admitted':admitted})+'\n')


def io_admission(phase):
    return _admission(phase, os.environ.get(ENV), 'io')


def gpu_admission(phase, requested):
    # Optional independent gate also applies to subject-level benchmark arms.
    return _admission(phase, os.environ.get(GPU_ENV) if requested else None, 'gpu')
