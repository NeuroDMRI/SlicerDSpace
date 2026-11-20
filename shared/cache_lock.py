"""OS-owned, crash-released locks for shared preparation on all desktop platforms."""

from contextlib import contextmanager
import os
from pathlib import Path
import sys
import time

LOG_INTERVAL_SECONDS = 30.0


def _holder_pid(stream):
    """Best-effort pid of the lock holder: POSIX holders record their pid in the lock file; None when unknown."""
    try:
        text = os.pread(stream.fileno(), 32, 0).decode('ascii', 'ignore').strip()
        return int(text) if text.isdigit() and int(text) > 0 else None
    except (AttributeError, OSError, ValueError):
        return None


def _default_log(message):
    # stderr, so a CLI whose stdout carries a JSON result stays parseable; the GUI merges both streams.
    print(message, file=sys.stderr, flush=True)


@contextmanager
def cache_lock(path, *, timeout=1800, log=None, log_interval=LOG_INTERVAL_SECONDS):
    """Hold an OS lock on ``path``. While waiting, ``log`` (default: stderr) gets a line every ``log_interval`` s (H7)."""
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
        next_log = started + log_interval
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
                now = time.monotonic()
                if now >= next_log:
                    next_log = now + log_interval
                    holder = _holder_pid(stream)
                    (log or _default_log)(
                        f"waiting for {path} (held by pid {holder if holder else 'unknown'}) for {int(now - started)} s")
                time.sleep(wait_s)
                wait_s = min(0.05, wait_s * 1.5)
        if os.name != 'nt':
            try:
                # Fixed width, overwritten in place (no truncate): a waiter that sees an empty file writes b'0' first,
                # which equals the leading zero of this value, so the two writes cannot corrupt each other.
                os.pwrite(stream.fileno(), b'%020d' % os.getpid(), 0)
            except (AttributeError, OSError):
                pass
        try:
            yield
        finally:
            if os.name == 'nt':
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)
