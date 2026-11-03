"""Scope Linux CPU placement for native tools that ignore thread variables."""

from contextlib import contextmanager
import os


@contextmanager
def cpu_placement(cpus):
    if not cpus or not hasattr(os, 'sched_setaffinity'):
        yield
        return
    import psutil
    process = psutil.Process()
    original = os.sched_getaffinity(0)
    previous = {}
    try:
        for thread in process.threads():
            try:
                previous[thread.id] = os.sched_getaffinity(thread.id)
                os.sched_setaffinity(thread.id, cpus)
            except ProcessLookupError:
                pass
        yield
    finally:
        for thread in process.threads():
            try:
                os.sched_setaffinity(thread.id, previous.get(thread.id, original))
            except ProcessLookupError:
                pass
