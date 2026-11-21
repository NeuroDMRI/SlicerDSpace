"""Update journal and rollback for an interrupted code-overlay apply (H6, U1, U2).

SlicerDSpace.py loads this file by path before any ``shared`` import, so a half-written ``shared`` package
cannot stop the rollback. Keep it standard-library only; psutil is optional and imported inside a try.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

JOURNAL_DIRNAME = ".slicerdspace_update_journal"
JOURNAL_FILENAME = "journal.json"
# Outside the journal folder: apply and discard remove that folder while the lock is held.
LOCK_FILENAME = ".slicerdspace_update.lock"
# Without psutil a pid can be a reused one; a journal older than this counts as left by a dead worker.
OWNER_MAX_AGE_SECONDS = 2 * 3600
_DISCARD_RETRY_SECONDS = 5.0
_DISCARD_RETRY_INTERVAL = 0.1

_MODULE_ROOT = Path(__file__).resolve().parents[1]


def journal_dir(module_root: Path) -> Path:
    return Path(module_root) / JOURNAL_DIRNAME


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path_fs = win_fs_path(path)
    path_fs.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = win_fs_path(path.with_name(path.name + ".tmp"))
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, path_fs)


def clear_stale_pyc(root: Path, python_relatives: set) -> None:
    for rel in python_relatives:
        if not rel.endswith(".py"):
            continue
        py_path = root / rel
        cache_dir = win_fs_path(py_path.parent / "__pycache__")
        if not cache_dir.is_dir():
            continue
        for pyc in cache_dir.glob(f"{py_path.stem}.*.pyc"):
            try:
                win_fs_path(pyc).unlink()
            except OSError:
                pass


def current_process_identity() -> Dict[str, Any]:
    """pid, start time and wall clock of this worker, so a later start can tell a live apply from a killed one."""

    identity: Dict[str, Any] = {"pid": os.getpid(), "started": time.time()}
    try:
        import psutil

        identity["pid_created"] = psutil.Process().create_time()
    except Exception:
        pass
    return identity


_WIN_EXTENDED_PATH_THRESHOLD = 240


def win_fs_path(path: Path) -> Path:
    """Return a \\\\?\\ extended-length Path on Windows when the resolved path is near MAX_PATH."""

    if os.name != "nt":
        return path
    resolved = os.path.abspath(os.fspath(path))
    if len(resolved) < _WIN_EXTENDED_PATH_THRESHOLD or resolved.startswith("\\\\?\\"):
        return path
    if resolved.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + resolved[2:])
    return Path("\\\\?\\" + resolved)


def _pid_alive_without_psutil(pid: int) -> bool:
    """Liveness of a pid without psutil; any doubt counts as alive."""

    if os.name == "nt":
        # os.kill(pid, 0) terminates the process on Windows: ask the kernel instead.
        try:
            import ctypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
            kernel32.OpenProcess.restype = ctypes.c_void_p
            kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
            kernel32.GetExitCodeProcess.restype = ctypes.c_int
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel32.CloseHandle.restype = ctypes.c_int

            ctypes.set_last_error(0)
            handle = kernel32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
            if not handle:
                return ctypes.get_last_error() != 87  # ERROR_INVALID_PARAMETER: no such process
            try:
                code = ctypes.c_uint32()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return True
                return code.value == 259  # STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return True
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except Exception:  # PermissionError: the pid exists but belongs to another user
        return True
    return True


def _journal_age_seconds(journal: Dict[str, Any], journal_file: Optional[Path]) -> float:
    started = journal.get("started")
    if isinstance(started, (int, float)):
        return time.time() - float(started)
    try:
        return time.time() - journal_file.stat().st_mtime if journal_file is not None else 0.0
    except OSError:
        return 0.0


def journal_owner_is_running(journal: Dict[str, Any], journal_file: Optional[Path] = None) -> bool:
    """True when the worker that wrote the journal may still run (U2: a missing psutil never means dead)."""

    pid = journal.get("pid")
    created = journal.get("pid_created")
    if not isinstance(pid, int) or pid <= 0 or pid == os.getpid():
        return False
    try:
        import psutil
    except Exception:
        psutil = None
    if psutil is not None:
        try:
            process = psutil.Process(pid)
            if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
                return False
            if created is not None:
                return abs(process.create_time() - float(created)) < 1.0
        except Exception as exc:  # NoSuchProcess: dead; AccessDenied and others: assume alive
            if type(exc).__name__ in ("NoSuchProcess", "ZombieProcess"):
                return False
            if type(exc).__name__ != "AccessDenied":
                return True
    elif not _pid_alive_without_psutil(pid):
        return False
    # The pid runs but its identity is not proven (no psutil or no create time): trust it for a limited time.
    return _journal_age_seconds(journal, journal_file) < OWNER_MAX_AGE_SECONDS


def acquire_update_lock(module_root: Path) -> Tuple[str, Any]:
    """Take the interprocess update lock without waiting: ("acquired", handle), ("busy", None) or ("unavailable", None).

    The OS frees the lock when its process dies, so a killed worker never leaves a stale lock.
    """

    lock_path = Path(module_root) / LOCK_FILENAME
    try:
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    except OSError:
        return "unavailable", None
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, PermissionError):
        os.close(fd)
        return "busy", None
    except OSError as exc:
        os.close(fd)
        import errno

        if exc.errno in (errno.EAGAIN, errno.EACCES, getattr(errno, "EDEADLK", -1)):
            return "busy", None
        return "unavailable", None
    return "acquired", fd


def release_update_lock(handle: Any) -> None:
    if handle is None:
        return
    try:
        if os.name == "nt":
            import msvcrt

            os.lseek(handle, 0, os.SEEK_SET)
            msvcrt.locking(handle, msvcrt.LK_UNLCK, 1)
    except OSError:
        pass
    try:
        os.close(handle)
    except OSError:
        pass


def write_update_journal(
    journal_directory: Path,
    *,
    target_version: str,
    backed_up: List[str],
    created: List[str],
    backup_map: Optional[Dict[str, str]] = None,
) -> None:
    """Record what an apply is about to change, before the first file is replaced (H6)."""

    payload: Dict[str, Any] = {
        "kind": "slicerdspace_update_journal",
        "version": 1,
        **current_process_identity(),
        "target_version": target_version,
        "backed_up": list(backed_up),
        "created": list(created),
    }
    if backup_map:
        payload["backup_map"] = dict(backup_map)
    write_json_atomic(journal_directory / JOURNAL_FILENAME, payload)


def discard_journal(journal_directory: Path) -> None:
    """Remove the journal after a finished apply; a journal that stays must never roll back a good update."""

    journal_file = win_fs_path(journal_directory / JOURNAL_FILENAME)
    deadline = time.monotonic() + _DISCARD_RETRY_SECONDS
    while journal_file.exists():
        try:
            journal_file.unlink()
        except OSError:
            if time.monotonic() >= deadline:
                try:
                    write_json_atomic(journal_file, {"kind": "slicerdspace_update_journal", "completed": True})
                except OSError:
                    pass
                break
            time.sleep(_DISCARD_RETRY_INTERVAL)
    shutil.rmtree(win_fs_path(journal_directory), ignore_errors=True)


def rollback_apply(
    module_root: Path,
    backup_dir: Path,
    backed_up_relatives: List[str],
    created_relatives: List[str],
    backup_map: Optional[Dict[str, str]] = None,
) -> None:
    for rel in created_relatives:
        dst = win_fs_path(module_root / rel)
        try:
            dst.unlink(missing_ok=True)
        except OSError:
            pass
    for rel in backed_up_relatives:
        backup_rel = backup_map.get(rel, rel) if backup_map else rel
        src = win_fs_path(backup_dir / backup_rel)
        dst = win_fs_path(module_root / rel)
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def recover_interrupted_update(module_root: Optional[Path] = None, *, lock_held: bool = False) -> Optional[Dict[str, Any]]:
    """Roll back a half-applied update left by a killed worker; return None when there is nothing to do (H6).

    Without ``lock_held`` the call takes the update lock first; a busy lock means a live updater (U2).
    R3-B1: every journal writer holds the lock, so with the lock acquired or held the journal owner cannot be a live
    updater and the owner check (which a reused pid can fool without psutil) is skipped. Only when the lock cannot be
    opened ("unavailable") does the owner check still decide.
    """

    root = Path(module_root or _MODULE_ROOT).expanduser().resolve()
    journal_file = journal_dir(root) / JOURNAL_FILENAME
    if not journal_file.is_file():
        return None
    if lock_held:
        return _recover_locked(root, check_owner=False)
    status, handle = acquire_update_lock(root)
    if status == "busy":
        pid = None
        try:
            pid = json.loads(journal_file.read_text(encoding="utf-8")).get("pid")
        except Exception:
            pass
        return {"status": "update_in_progress", "pid": pid}
    try:
        return _recover_locked(root, check_owner=status != "acquired")
    finally:
        release_update_lock(handle)


def _recover_locked(root: Path, *, check_owner: bool) -> Optional[Dict[str, Any]]:
    directory = journal_dir(root)
    journal_file = directory / JOURNAL_FILENAME
    if not journal_file.is_file():
        return None
    try:
        journal = json.loads(win_fs_path(journal_file).read_text(encoding="utf-8"))
        if journal.get("completed"):
            shutil.rmtree(win_fs_path(directory), ignore_errors=True)
            return None
        if check_owner and journal_owner_is_running(journal, journal_file):
            return {"status": "update_in_progress", "pid": journal.get("pid")}
        backed_up = [str(item) for item in journal["backed_up"]]
        created = [str(item) for item in journal["created"]]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        print(
            f"SlicerDSpace: unreadable update journal {journal_file}; delete the folder {directory} "
            "and reinstall the SlicerDSpace package if the module fails to load.",
            flush=True,
        )
        return {"status": "journal_unreadable", "journal": str(journal_file)}
    raw_backup_map = journal.get("backup_map")
    backup_map = {str(k): str(v) for k, v in raw_backup_map.items()} if isinstance(raw_backup_map, dict) else None
    backup_dir = directory / "backup"
    missing = [
        rel
        for rel in backed_up
        if not win_fs_path(backup_dir / (backup_map.get(rel, rel) if backup_map else rel)).is_file()
    ]
    if missing:
        print(
            f"SlicerDSpace: cannot roll back the interrupted update to {journal.get('target_version')}: "
            f"{len(missing)} backup file(s) are missing (first: {missing[0]}). Reinstall the SlicerDSpace package.",
            flush=True,
        )
        return {"status": "rollback_incomplete", "missing": missing}
    rollback_apply(root, backup_dir, backed_up, created, backup_map=backup_map)
    clear_stale_pyc(root, set(backed_up) | set(created))
    discard_journal(directory)
    print(
        f"SlicerDSpace: rolled back an interrupted update to {journal.get('target_version')} "
        f"({len(backed_up)} files restored, {len(created)} removed). Restart Slicer if the module fails to load.",
        flush=True,
    )
    return {
        "status": "rolled_back",
        "target_version": journal.get("target_version"),
        "restored_files": len(backed_up),
        "removed_files": len(created),
    }
