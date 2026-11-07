"""Scoped native numerical threads, within an admitted stage's CPU budget."""

from contextlib import contextmanager
from functools import wraps
import os


def reserved_stage_cpus():
    """CPUs of the current stage reservation (``SLICERDSPACE_STAGE_CPUS``), or None.

    Thread precedence for every stage: explicit user value, then this reservation,
    then the stage's built-in default. A missing or invalid value gives None.
    """
    value = os.environ.get('SLICERDSPACE_STAGE_CPUS', '').strip()
    try:
        count = int(value)
    except ValueError:
        return None
    return count if count >= 1 else None


@contextmanager
def numerical_threads(threads, *, workers=1):
    if threads is None:
        yield
        return
    if type(threads) is not int or threads < 1 or type(workers) is not int or workers < 1:
        raise ValueError('Numerical thread and worker counts must be positive integers')
    reservation = os.environ.get('SLICERDSPACE_STAGE_CPUS')
    if reservation and threads * workers > int(reservation):
        raise ValueError('Numerical workers/threads exceed the admitted CPU reservation')
    from threadpoolctl import threadpool_limits
    keys = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')
    previous = {key: os.environ.get(key) for key in keys}
    try:
        os.environ.update({key: str(threads) for key in keys})
        with threadpool_limits(limits=threads):
            yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def dki_threads(function):
    @wraps(function)
    def wrapped(config, *args, **kwargs):
        workers = config.mkcurve_workers if config.fit_method.upper() == 'MKCURVE' else 1
        with numerical_threads(config.num_threads, workers=workers):
            return function(config, *args, **kwargs)
    return wrapped
