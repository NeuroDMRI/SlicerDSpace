"""Update journal and rollback for an interrupted code-overlay apply (H6, U1, U2).

SlicerDSpace.py loads this file by path before any ``shared`` import, so a half-written ``shared`` package
cannot stop the rollback. Keep it standard-library only; psutil is optional and imported inside a try.
"""

from __future__ import annotations

import hashlib
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


_ROLLBACK_RETRIES = 3
_ROLLBACK_RETRY_DELAY = 0.05


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(262144)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _files_match(src: Path, dst: Path) -> bool:
    try:
        if not src.is_file() or not dst.is_file():
            return False
        if src.stat().st_size != dst.stat().st_size:
            return False
        return _sha256_file(src) == _sha256_file(dst)
    except OSError:
        return False


def _unlink_with_retry(dst: Path, *, retries: int, retry_delay: float) -> Optional[str]:
    attempts = max(1, int(retries))
    last_error: Optional[str] = None
    for attempt in range(attempts):
        try:
            if not os.path.lexists(dst):
                return None
            dst.unlink()
        except FileNotFoundError:
            return None
        except OSError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < attempts and retry_delay > 0:
                time.sleep(retry_delay)
            continue
        if not os.path.lexists(dst):
            return None
        last_error = f"file still exists after unlink: {dst}"
        if attempt + 1 < attempts and retry_delay > 0:
            time.sleep(retry_delay)
    if not os.path.lexists(dst):
        return None
    return last_error or f"failed to remove {dst}"


def _restore_file_with_retry(
    src: Path,
    dst_raw: Path,
    *,
    retries: int,
    retry_delay: float,
) -> Optional[str]:
    if not src.is_file():
        return f"backup file missing: {src}"
    dst = win_fs_path(dst_raw)
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return f"{type(exc).__name__}: {exc}"
    tmp_name = f".{dst_raw.name[:180]}.rbk.{os.getpid()}.tmp"
    tmp_fs = win_fs_path(dst_raw.with_name(tmp_name))
    attempts = max(1, int(retries))
    last_error: Optional[str] = None
    for attempt in range(attempts):
        try:
            shutil.copy2(src, tmp_fs)
            os.replace(tmp_fs, dst)
        except OSError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            try:
                tmp_fs.unlink(missing_ok=True)
            except OSError:
                pass
            if attempt + 1 < attempts and retry_delay > 0:
                time.sleep(retry_delay)
            continue
        if _files_match(src, dst):
            return None
        last_error = f"post-restore verification failed for {dst_raw}"
        if attempt + 1 < attempts and retry_delay > 0:
            time.sleep(retry_delay)
    try:
        tmp_fs.unlink(missing_ok=True)
    except OSError:
        pass
    return last_error or f"failed to restore {dst_raw}"


def rollback_apply(
    module_root: Path,
    backup_dir: Path,
    backed_up_relatives: List[str],
    created_relatives: List[str],
    backup_map: Optional[Dict[str, str]] = None,
    *,
    retries: int = _ROLLBACK_RETRIES,
    retry_delay: float = _ROLLBACK_RETRY_DELAY,
) -> Dict[str, Any]:
    """Roll back created and overwritten files, tracking and verifying every operation."""
    root = Path(module_root)
    backup_root = Path(backup_dir)
    removed: List[str] = []
    restored: List[str] = []
    failed_removals: List[Dict[str, str]] = []
    failed_restores: List[Dict[str, str]] = []
    errors: List[str] = []

    for rel in created_relatives:
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            err = f"unsafe relative path: {rel!r}"
            failed_removals.append({"path": str(rel), "error": err})
            errors.append(f"remove {rel}: {err}")
            continue
        dst = win_fs_path(root / rel)
        err = _unlink_with_retry(dst, retries=retries, retry_delay=retry_delay)
        if err is not None:
            failed_removals.append({"path": str(rel), "error": err})
            errors.append(f"remove {rel}: {err}")
        else:
            removed.append(str(rel))

    for rel in backed_up_relatives:
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            err = f"unsafe relative path: {rel!r}"
            failed_restores.append({"path": str(rel), "error": err})
            errors.append(f"restore {rel}: {err}")
            continue
        backup_rel = backup_map.get(rel, rel) if backup_map else rel
        src = win_fs_path(backup_root / backup_rel)
        err = _restore_file_with_retry(src, root / rel, retries=retries, retry_delay=retry_delay)
        if err is not None:
            failed_restores.append({"path": str(rel), "error": err})
            errors.append(f"restore {rel}: {err}")
        else:
            restored.append(str(rel))

    return {
        "ok": not failed_removals and not failed_restores,
        "restored": restored,
        "removed": removed,
        "failed_removals": failed_removals,
        "failed_restores": failed_restores,
        "errors": errors,
    }


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
        return {"status": "rollback_incomplete", "ok": False, "missing": missing, "journal": str(journal_file)}
    report = rollback_apply(root, backup_dir, backed_up, created, backup_map=backup_map)
    clear_stale_pyc(root, set(report["restored"]) | set(report["removed"]))
    if not report["ok"]:
        first_err = report["errors"][0] if report["errors"] else "unknown error"
        print(
            f"SlicerDSpace: rollback of the interrupted update to {journal.get('target_version')} is incomplete "
            f"({len(report['failed_restores'])} restore failure(s), {len(report['failed_removals'])} removal failure(s); "
            f"first: {first_err}). Journal and backups were kept at {directory}; close any process using those files "
            "and restart Slicer to retry, or reinstall the SlicerDSpace package.",
            flush=True,
        )
        return {
            "status": "rollback_incomplete",
            "ok": False,
            "target_version": journal.get("target_version"),
            "restored_files": len(report["restored"]),
            "removed_files": len(report["removed"]),
            "failed_removals": report["failed_removals"],
            "failed_restores": report["failed_restores"],
            "errors": report["errors"],
            "journal": str(journal_file),
        }
    discard_journal(directory)
    print(
        f"SlicerDSpace: rolled back an interrupted update to {journal.get('target_version')} "
        f"({len(report['restored'])} files restored, {len(report['removed'])} removed). Restart Slicer if the module fails to load.",
        flush=True,
    )
    return {
        "status": "rolled_back",
        "ok": True,
        "target_version": journal.get("target_version"),
        "restored_files": len(report["restored"]),
        "removed_files": len(report["removed"]),
    }
