"""OS-owned, crash-released locks for shared preparation on all desktop platforms."""

from contextlib import contextmanager
import os
from pathlib import Path
import time


@contextmanager
def cache_lock(path, *, timeout=1800):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f'Refusing a symlinked cache lock: {path}')
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(descriptor, 'r+b', buffering=0) as stream:
        try:
            if os.fstat(stream.fileno()).st_size == 0:
                stream.write(b'0')
        except PermissionError:
            # On Windows mandatory byte-range locks, another holder has already written and locked byte 0.
            pass
        started = time.monotonic()
        wait_s = 0.01
        while True:
            try:
                if os.name == 'nt':
                    import msvcrt
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (BlockingIOError, PermissionError):
                if time.monotonic() - started >= timeout:
                    raise TimeoutError(f'Another process is preparing shared inputs: {path.parent}')
                time.sleep(wait_s)
                wait_s = min(0.05, wait_s * 1.5)
        try:
            yield
        finally:
            if os.name == 'nt':
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)
