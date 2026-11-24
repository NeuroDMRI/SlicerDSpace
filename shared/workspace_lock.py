"""Cross-platform per-subject workspace lock preventing concurrent output clobbering.

Uses atomic file creation (hard-link from a complete temp file with ``O_CREAT | O_EXCL``
fallback) plus JSON owner metadata (PID, ISO start time, run name, subject ID, hostname,
status, token). Supports:
- Clear user-facing contention errors showing the active owner's PID/start time/run name/status.
- Race-free stale-lock reclamation when the recorded same-host PID is no longer alive.
- Same-thread re-entrancy (CLI -> orchestrator -> multi-tracking child runs) without deadlocking,
  while still rejecting concurrent threads in the same process targeting the same workspace.
- Child subprocess inheritance via ``SLICERDSPACE_WORKSPACE_LOCKS`` so subprocesses spawned by
  the owning orchestrator never deadlock against their parent.
"""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
from typing import Any, Iterator, Optional
import uuid

LOCK_FILENAME = ".workspace.lock"
RECLAIM_FILENAME = ".workspace.lock.reclaim"
WORKSPACE_LOCKS_ENV = "SLICERDSPACE_WORKSPACE_LOCKS"

_IN_PROCESS_GUARD = threading.RLock()
_ACTIVE_LOCKS: dict[str, dict[str, Any]] = {}


class WorkspaceLockError(RuntimeError):
    """Raised when another active run holds the per-subject workspace lock."""

    def __init__(self, message: str, *, lock_path: Path, owner: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.lock_path = Path(lock_path)
        self.owner = dict(owner or {})


def _lock_key(lock_path: Path) -> str:
    return os.path.normcase(os.path.normpath(str(lock_path.expanduser().resolve())))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def is_pid_alive(pid: int) -> bool:
    """Return True if ``pid`` refers to a live OS process on the local host."""
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return False
    if pid_int <= 0:
        return False
    if pid_int == os.getpid():
        return True
    if sys.platform == "win32":
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        ERROR_ACCESS_DENIED = 5
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
            kernel32.OpenProcess.restype = ctypes.c_void_p
            kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
            kernel32.GetExitCodeProcess.restype = ctypes.c_int
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel32.CloseHandle.restype = ctypes.c_int
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid_int)
            if not handle:
                return ctypes.get_last_error() == ERROR_ACCESS_DENIED
            try:
                exit_code = ctypes.c_uint32(0)
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    return True
                return int(exit_code.value) == STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return False

    if sys.platform.startswith("linux"):
        stat_path = Path(f"/proc/{pid_int}/stat")
        try:
            stat_text = stat_path.read_text(encoding="utf-8", errors="ignore")
            fields = stat_text.rsplit(") ", 1)[1].split()
            if fields and fields[0] in {"Z", "X"}:
                return False
        except FileNotFoundError:
            return False
        except (OSError, IndexError):
            pass

    try:
        os.kill(pid_int, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _read_env_locks() -> dict[str, dict[str, Any]]:
    raw = (os.environ.get(WORKSPACE_LOCKS_ENV) or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return {str(k): v for k, v in parsed.items() if isinstance(v, dict)}
    except Exception:
        pass
    return {}


def _write_env_locks(entries: dict[str, dict[str, Any]]) -> None:
    if not entries:
        os.environ.pop(WORKSPACE_LOCKS_ENV, None)
    else:
        os.environ[WORKSPACE_LOCKS_ENV] = json.dumps(entries, sort_keys=True)


def _read_lock_metadata(lock_path: Path, *, retries: int = 5, delay: float = 0.03) -> Optional[dict[str, Any]]:
    for attempt in range(max(1, retries)):
        try:
            raw = lock_path.read_text(encoding="utf-8").strip()
            if raw:
                payload = json.loads(raw)
                if isinstance(payload, dict):
                    return payload
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            pass
        if attempt + 1 < retries:
            time.sleep(delay)
    return None


def _atomic_create_lock_file(lock_path: Path, payload_bytes: bytes) -> bool:
    """Atomically create ``lock_path`` with ``payload_bytes`` only if it does not exist."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".workspace.lock.", suffix=".tmp", dir=lock_path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload_bytes)
            fh.flush()
        try:
            os.link(str(tmp_path), str(lock_path))
            return True
        except FileExistsError:
            return False
        except (OSError, NotImplementedError):
            # Filesystem without hard links (e.g. FAT32/exFAT/some network mounts): use O_CREAT | O_EXCL.
            try:
                excl_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                return False
            try:
                os.write(excl_fd, payload_bytes)
            finally:
                os.close(excl_fd)
            return True
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _lock_is_stale(meta: Optional[dict[str, Any]], *, lock_path: Path, current_host: str) -> bool:
    if meta is None:
        try:
            stat_info = lock_path.stat()
        except FileNotFoundError:
            return True
        except OSError:
            return False
        # Empty/corrupt lock file older than 5 seconds is considered crashed mid-write.
        return (time.time() - float(stat_info.st_mtime)) > 5.0
    host = str(meta.get("hostname") or "").strip()
    if host and host != current_host:
        return False
    pid = meta.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return True
    return not is_pid_alive(pid)


def _format_lock_error(lock_path: Path, owner: Optional[dict[str, Any]]) -> str:
    info = owner or {}
    pid = info.get("pid", "unknown")
    started = info.get("started_at", "unknown")
    run_name = info.get("run_name", "unknown")
    subject_id = info.get("subject_id", "unknown")
    status = info.get("status", "running")
    host = info.get("hostname", "unknown")
    return (
        f"Subject workspace is currently locked by an active SlicerDSpace run "
        f"(PID={pid}, subject_id={subject_id}, run_name={run_name}, status={status}, "
        f"started_at={started}, host={host}, lock_file={lock_path}). "
        f"Wait for the active job to complete before starting another run on the same workspace."
    )


def _is_inherited_lock(lock_key: str, meta: Optional[dict[str, Any]], current_host: str) -> bool:
    if not meta:
        return False
    env_locks = _read_env_locks()
    inherited = env_locks.get(lock_key)
    if not inherited:
        return False
    if str(inherited.get("token") or "") != str(meta.get("token") or ""):
        return False
    if not meta.get("token"):
        return False
    owner_host = str(meta.get("hostname") or "")
    if owner_host and owner_host != current_host:
        return False
    owner_pid = meta.get("pid")
    if not isinstance(owner_pid, int) or owner_pid <= 0:
        return False
    return is_pid_alive(owner_pid)


def _acquire_reclaim_mutex(reclaim_path: Path, *, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(str(reclaim_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            try:
                os.write(fd, f"{os.getpid()}\n".encode("ascii"))
            finally:
                os.close(fd)
            return True
        except FileExistsError:
            try:
                age = time.time() - float(reclaim_path.stat().st_mtime)
                if age > 5.0:
                    reclaim_path.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            except OSError:
                pass
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.02)


@contextmanager
def workspace_lock(
    lock_dir: Path,
    *,
    subject_id: str,
    run_name: str,
    status: str = "running",
) -> Iterator[Path]:
    """Acquire an exclusive per-subject workspace lock under ``lock_dir``.

    Re-entrant on the same thread and inherited by child subprocesses via
    ``SLICERDSPACE_WORKSPACE_LOCKS``. Raises :class:`WorkspaceLockError` if another
    active process or thread holds the lock.
    """
    lock_root = Path(lock_dir).expanduser().resolve()
    lock_path = lock_root / LOCK_FILENAME
    reclaim_path = lock_root / RECLAIM_FILENAME
    key = _lock_key(lock_path)
    this_thread = threading.get_ident()
    current_host = socket.gethostname()

    # 1. Same-process check: re-entrant on the owning thread, conflict on another thread.
    with _IN_PROCESS_GUARD:
        existing = _ACTIVE_LOCKS.get(key)
        if existing is not None:
            if existing["thread_id"] == this_thread:
                existing["refcount"] += 1
                reentered = True
            else:
                owner = existing.get("metadata") or _read_lock_metadata(lock_path, retries=1)
                raise WorkspaceLockError(_format_lock_error(lock_path, owner), lock_path=lock_path, owner=owner)
        else:
            reentered = False

    if reentered:
        try:
            yield lock_path
        finally:
            with _IN_PROCESS_GUARD:
                entry = _ACTIVE_LOCKS.get(key)
                if entry is not None:
                    entry["refcount"] -= 1
        return

    # 2. Cross-process acquisition with inherited-env check and atomic stale-lock reclamation.
    token = uuid.uuid4().hex
    metadata: dict[str, Any] = {
        "pid": os.getpid(),
        "hostname": current_host,
        "started_at": _now_iso(),
        "run_name": str(run_name),
        "subject_id": str(subject_id),
        "status": str(status),
        "token": token,
    }
    payload_bytes = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8")

    acquired = False
    inherited = False
    for _attempt in range(6):
        if _atomic_create_lock_file(lock_path, payload_bytes):
            acquired = True
            break

        owner = _read_lock_metadata(lock_path)
        if _is_inherited_lock(key, owner, current_host):
            inherited = True
            break

        if not _lock_is_stale(owner, lock_path=lock_path, current_host=current_host):
            raise WorkspaceLockError(_format_lock_error(lock_path, owner), lock_path=lock_path, owner=owner)

        # Atomically reclaim confirmed-stale lock under a short-lived reclaim mutex.
        if not _acquire_reclaim_mutex(reclaim_path):
            continue
        try:
            rechecked = _read_lock_metadata(lock_path, retries=2, delay=0.01)
            if _is_inherited_lock(key, rechecked, current_host):
                inherited = True
                break
            if rechecked is not None and not _lock_is_stale(rechecked, lock_path=lock_path, current_host=current_host):
                raise WorkspaceLockError(_format_lock_error(lock_path, rechecked), lock_path=lock_path, owner=rechecked)
            try:
                lock_path.unlink(missing_ok=True)
            except OSError:
                pass
            if _atomic_create_lock_file(lock_path, payload_bytes):
                acquired = True
                break
        finally:
            try:
                reclaim_path.unlink(missing_ok=True)
            except OSError:
                pass

    if inherited:
        yield lock_path
        return

    if not acquired:
        owner = _read_lock_metadata(lock_path)
        raise WorkspaceLockError(_format_lock_error(lock_path, owner), lock_path=lock_path, owner=owner)

    with _IN_PROCESS_GUARD:
        _ACTIVE_LOCKS[key] = {
            "thread_id": this_thread,
            "token": token,
            "refcount": 1,
            "metadata": metadata,
        }
        env_locks = _read_env_locks()
        prev_entry = env_locks.get(key)
        env_locks[key] = {"token": token, "pid": os.getpid(), "hostname": current_host}
        _write_env_locks(env_locks)

    try:
        yield lock_path
    finally:
        with _IN_PROCESS_GUARD:
            entry = _ACTIVE_LOCKS.get(key)
            should_release = entry is not None and entry.get("token") == token and entry.get("refcount", 1) <= 1
            if should_release:
                _ACTIVE_LOCKS.pop(key, None)
                env_locks = _read_env_locks()
                if env_locks.get(key, {}).get("token") == token:
                    if prev_entry is not None:
                        env_locks[key] = prev_entry
                    else:
                        env_locks.pop(key, None)
                    _write_env_locks(env_locks)
            elif entry is not None:
                entry["refcount"] -= 1
        if should_release:
            try:
                on_disk = _read_lock_metadata(lock_path, retries=1)
                if on_disk is None or on_disk.get("token") == token:
                    lock_path.unlink(missing_ok=True)
            except OSError:
                pass
