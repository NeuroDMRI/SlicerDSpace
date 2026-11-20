"""Cross-platform incremental code-overlay updater (~2-3 MB) for SlicerDSpace."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tarfile
import tempfile
import time
import urllib.request
from typing import Any, Callable

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from scripts.extension_runtime.runtime_resources import runtime_files
from shared.network_config import (
    PRESET_MIRRORS,
    expand_candidate_urls,
    find_local_pack_archive,
    load_network_settings,
    open_url,
)
# U1: the journal code lives in a standard-library-only module that SlicerDSpace.py loads before `shared`.
from shared.update_journal import (
    JOURNAL_DIRNAME,
    JOURNAL_FILENAME as _JOURNAL_FILENAME,
    acquire_update_lock,
    clear_stale_pyc as _clear_stale_pyc,
    current_process_identity as _current_process_identity,
    discard_journal as _discard_journal,
    journal_dir as _journal_dir,
    journal_owner_is_running as _journal_owner_is_running,
    recover_interrupted_update,
    release_update_lock,
    rollback_apply as _rollback_apply,
    write_json_atomic as _write_json_atomic,
    write_update_journal as _write_update_journal,
)

REPO_ROOT = _PACKAGE_ROOT

EXTENSION_VERSION = "0.1.1"
RUNTIME_ABI_VERSION = 1
GITHUB_REPO = "NeuroDMRI/SlicerDSpace"
DEFAULT_UPDATE_MANIFEST_URL = (
    f"https://github.com/{GITHUB_REPO}/releases/latest/download/update_manifest.json"
)
UPDATE_MANIFEST_URL_ENV = "SLICERDSPACE_UPDATE_MANIFEST_URL"

_MAX_OVERLAY_FILES = 5000
_MAX_OVERLAY_UNPACKED_BYTES = 256 * 1024 * 1024

ProgressCallback = Callable[[int, int], None]
PhaseCallback = Callable[[str], None]

_REPLACE_RETRY_SECONDS = 5.0
_REPLACE_RETRY_INTERVAL = 0.1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_code_tree_sha256(root: Path | None = None) -> tuple[str, list[str]]:
    """Return deterministic SHA-256 fingerprint and sorted runtime file list."""

    resolved = (root or REPO_ROOT).expanduser().resolve()
    files = runtime_files(resolved)
    digest = hashlib.sha256()
    for relative in files:
        file_hash = _sha256_file(resolved / relative)
        digest.update(f"{relative}\0{file_hash}\n".encode("utf-8"))
    return digest.hexdigest(), files


def _is_directory_writable(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    probe = directory / f".slicerdspace_write_probe_{os.getpid()}"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def current_installation_info(module_root: Path | None = None) -> dict[str, Any]:
    """Inspect local SlicerDSpace installation root, writability, ABI, and tree hash."""

    resolved = (module_root or REPO_ROOT).expanduser().resolve()
    is_git_checkout = (resolved / ".git").exists()
    code_sha256 = ""
    file_count = 0
    try:
        code_sha256, files = compute_code_tree_sha256(resolved)
        file_count = len(files)
    except Exception:
        code_sha256 = ""
        file_count = 0
    return {
        "module_root": str(resolved),
        "version": EXTENSION_VERSION,
        "runtime_abi_version": RUNTIME_ABI_VERSION,
        "code_sha256": code_sha256,
        "file_count": file_count,
        "is_git_checkout": is_git_checkout,
        "writable": _is_directory_writable(resolved),
    }


def _candidate_urls_with_fallback(url: str, *, mirror_prefix: str | None = None) -> list[str]:
    """Expand GitHub URLs with configured mirror first, then preset fallback mirrors."""

    primary = expand_candidate_urls(url, mirror_prefix=mirror_prefix, allow_mirror=True)
    candidates = list(primary)
    for _preset_id, prefix, _en, _zh in PRESET_MIRRORS:
        if not prefix:
            continue
        for extra in expand_candidate_urls(url, mirror_prefix=prefix, allow_mirror=True):
            if extra not in candidates:
                candidates.append(extra)
    return candidates


def _validate_manifest_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Invalid update manifest: expected JSON object.")
    if int(payload.get("schema_version") or 0) != 1:
        raise ValueError(f"Unsupported update manifest schema_version: {payload.get('schema_version')!r}")
    code_sha = str(payload.get("code_sha256") or "").strip().lower()
    if len(code_sha) != 64:
        raise ValueError("Invalid update manifest: missing or malformed code_sha256.")
    try:
        runtime_abi = int(payload.get("runtime_abi_version"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid update manifest: missing runtime_abi_version.") from exc
    overlay = payload.get("overlay")
    if not isinstance(overlay, dict):
        raise ValueError("Invalid update manifest: missing overlay metadata.")
    filename = str(overlay.get("filename") or "").strip()
    url = str(overlay.get("url") or "").strip()
    overlay_sha = str(overlay.get("sha256") or "").strip().lower()
    try:
        size_bytes = int(overlay.get("size_bytes") or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid update manifest: invalid overlay size_bytes.") from exc
    if not filename or not url or len(overlay_sha) != 64 or size_bytes <= 0:
        raise ValueError("Invalid update manifest: incomplete overlay fields.")

    normalized = dict(payload)
    normalized["code_sha256"] = code_sha
    normalized["runtime_abi_version"] = runtime_abi
    normalized["version"] = str(payload.get("version") or EXTENSION_VERSION)
    normalized["release_tag"] = str(payload.get("release_tag") or f"v{normalized['version']}")
    normalized["git_commit"] = str(payload.get("git_commit") or "")
    normalized["git_short_commit"] = str(payload.get("git_short_commit") or normalized["git_commit"][:7])
    normalized["summary"] = str(payload.get("summary") or "")
    normalized["release_url"] = str(
        payload.get("release_url")
        or f"https://github.com/{GITHUB_REPO}/releases/tag/{normalized['release_tag']}"
    )
    normalized["overlay"] = {
        "filename": filename,
        "url": url,
        "sha256": overlay_sha,
        "size_bytes": size_bytes,
    }
    return normalized


def fetch_update_manifest(
    *,
    manifest_url: str | None = None,
    timeout: int = 12,
    offline_dir: Path | str | None = None,
    mirror_prefix: str | None = None,
    proxy_url: str | None = None,
    urlopen_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Fetch and validate remote update_manifest.json (supporting offline folder, mirror, proxy)."""

    settings = load_network_settings()
    resolved_offline = str(offline_dir if offline_dir is not None else settings.get("offline_assets_dir") or "").strip()
    if resolved_offline and manifest_url is None and urlopen_fn is None:
        local_manifest = Path(resolved_offline).expanduser() / "update_manifest.json"
        if local_manifest.is_file():
            payload = json.loads(local_manifest.read_text(encoding="utf-8"))
            return _validate_manifest_payload(payload)

    effective_url = (
        str(manifest_url or "").strip()
        or str(os.environ.get(UPDATE_MANIFEST_URL_ENV) or "").strip()
        or DEFAULT_UPDATE_MANIFEST_URL
    )
    candidates = _candidate_urls_with_fallback(effective_url, mirror_prefix=mirror_prefix)
    errors: list[str] = []
    for candidate in candidates:
        req = urllib.request.Request(
            candidate,
            headers={"User-Agent": f"SlicerDSpace-Updater/{EXTENSION_VERSION}"},
        )
        try:
            with open_url(req, timeout=timeout, proxy_url=proxy_url, urlopen_fn=urlopen_fn) as resp:
                raw = resp.read()
            payload = json.loads(raw.decode("utf-8"))
            return _validate_manifest_payload(payload)
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
    raise RuntimeError("Failed to fetch update manifest: " + "; ".join(errors[-3:]))


def check_for_code_update(
    *,
    module_root: Path | None = None,
    manifest_url: str | None = None,
    manifest: dict[str, Any] | None = None,
    timeout: int = 12,
    offline_dir: Path | str | None = None,
    mirror_prefix: str | None = None,
    proxy_url: str | None = None,
    urlopen_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Compare local installation against remote manifest and classify update status."""

    local = current_installation_info(module_root)
    remote = (
        _validate_manifest_payload(manifest)
        if manifest is not None
        else fetch_update_manifest(
            manifest_url=manifest_url,
            timeout=timeout,
            offline_dir=offline_dir,
            mirror_prefix=mirror_prefix,
            proxy_url=proxy_url,
            urlopen_fn=urlopen_fn,
        )
    )
    if remote["runtime_abi_version"] > int(local["runtime_abi_version"]):
        status = "runtime_upgrade_required"
    elif local["code_sha256"] and local["code_sha256"] == remote["code_sha256"]:
        status = "up_to_date"
    else:
        status = "update_available"

    overlay = remote["overlay"]
    return {
        "status": status,
        "local": local,
        "remote": remote,
        "is_git_checkout": bool(local["is_git_checkout"]),
        "writable": bool(local["writable"]),
        "overlay_size_bytes": int(overlay["size_bytes"]),
        "release_url": remote["release_url"],
        "summary": remote.get("summary", ""),
    }


def _safe_overlay_member_path(name: str) -> PurePosixPath:
    raw = str(name or "").replace("\\", "/")
    if raw.startswith("/"):
        raise ValueError(f"Unsafe overlay member path: {name!r}")
    while raw.startswith("./"):
        raw = raw[2:]
    posix = PurePosixPath(raw)
    if (
        not raw
        or ".." in posix.parts
        or any(":" in part for part in posix.parts)
    ):
        raise ValueError(f"Unsafe overlay member path: {name!r}")
    return posix


def _download_verified_overlay(
    overlay: dict[str, Any],
    dest: Path,
    *,
    timeout: int = 60,
    offline_dir: Path | str | None = None,
    mirror_prefix: str | None = None,
    proxy_url: str | None = None,
    urlopen_fn: Callable[..., Any] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    filename = str(overlay["filename"])
    expected_sha = str(overlay["sha256"]).lower()
    expected_size = int(overlay["size_bytes"])

    settings = load_network_settings()
    resolved_offline = str(offline_dir if offline_dir is not None else settings.get("offline_assets_dir") or "").strip()
    if resolved_offline and urlopen_fn is None:
        local_pack = find_local_pack_archive(resolved_offline, filename, expected_sha256=expected_sha)
        if local_pack is not None and local_pack.is_file():
            shutil.copy2(local_pack, dest)
            if dest.stat().st_size == expected_size and _sha256_file(dest) == expected_sha:
                if progress_callback is not None:
                    progress_callback(expected_size, expected_size)
                return dest
            dest.unlink(missing_ok=True)

    candidates = _candidate_urls_with_fallback(str(overlay["url"]), mirror_prefix=mirror_prefix)
    errors: list[str] = []
    part_path = dest.with_name(dest.name + ".part")
    for candidate in candidates:
        part_path.unlink(missing_ok=True)
        req = urllib.request.Request(
            candidate,
            headers={"User-Agent": f"SlicerDSpace-Updater/{EXTENSION_VERSION}"},
        )
        try:
            downloaded = 0
            digest = hashlib.sha256()
            with open_url(req, timeout=timeout, proxy_url=proxy_url, urlopen_fn=urlopen_fn) as resp:
                with part_path.open("wb") as out_fh:
                    while True:
                        chunk = resp.read(256 * 1024)
                        if not chunk:
                            break
                        out_fh.write(chunk)
                        digest.update(chunk)
                        downloaded += len(chunk)
                        if downloaded > _MAX_OVERLAY_UNPACKED_BYTES:
                            raise ValueError("Overlay download exceeded maximum allowed size.")
                        if progress_callback is not None:
                            progress_callback(downloaded, expected_size)
            if downloaded != expected_size:
                raise ValueError(f"Overlay size mismatch: {downloaded} != {expected_size}")
            actual_sha = digest.hexdigest()
            if actual_sha != expected_sha:
                raise ValueError(f"Overlay SHA-256 mismatch: {actual_sha} != {expected_sha}")
            os.replace(part_path, dest)
            return dest
        except Exception as exc:
            part_path.unlink(missing_ok=True)
            errors.append(f"{candidate}: {exc}")
    raise RuntimeError("Failed to download code overlay: " + "; ".join(errors[-3:]))


def _extract_overlay_safely(archive_path: Path, staging_dir: Path) -> dict[str, int]:
    """Extract overlay tar.gz into ``staging_dir`` with strict path/link/size validation."""

    file_modes: dict[str, int] = {}
    total_files = 0
    total_bytes = 0
    staging_resolved = staging_dir.resolve()
    with tarfile.open(archive_path, "r:*") as tf:
        for member in tf:
            if member.issym() or member.islnk() or member.isdev() or member.ischr() or member.isblk() or member.isfifo():
                raise ValueError(f"Unsupported link or special member in code overlay: {member.name!r}")
            raw_name = member.name.rstrip("/")
            if not raw_name or raw_name == ".":
                continue
            posix = _safe_overlay_member_path(raw_name)
            rel_str = posix.as_posix()
            target = (staging_dir / posix).resolve()
            if not target.is_relative_to(staging_resolved):
                raise ValueError(f"Overlay member escapes staging root: {member.name!r}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            total_files += 1
            total_bytes += int(member.size)
            if total_files > _MAX_OVERLAY_FILES:
                raise ValueError(f"Overlay exceeds maximum file limit ({_MAX_OVERLAY_FILES}).")
            if total_bytes > _MAX_OVERLAY_UNPACKED_BYTES:
                raise ValueError("Overlay exceeds maximum unpacked byte limit.")
            target.parent.mkdir(parents=True, exist_ok=True)
            src_fh = tf.extractfile(member)
            if src_fh is None:
                raise ValueError(f"Unable to read archive member: {member.name!r}")
            with src_fh, target.open("wb") as dst_fh:
                shutil.copyfileobj(src_fh, dst_fh, length=1024 * 1024)
            is_exec = bool((member.mode or 0) & stat.S_IXUSR)
            file_modes[rel_str] = 0o755 if is_exec else 0o644
    return file_modes


def _replace_with_retry(src: Path, dst: Path) -> None:
    """os.replace that retries PermissionError for a few seconds (X5: Windows scanners lock new files)."""

    deadline = time.monotonic() + _REPLACE_RETRY_SECONDS
    while True:
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(_REPLACE_RETRY_INTERVAL)


def download_and_apply_code_overlay(
    *,
    module_root: Path | None = None,
    manifest_url: str | None = None,
    manifest: dict[str, Any] | None = None,
    timeout: int = 60,
    offline_dir: Path | str | None = None,
    mirror_prefix: str | None = None,
    proxy_url: str | None = None,
    urlopen_fn: Callable[..., Any] | None = None,
    progress_callback: ProgressCallback | None = None,
    phase_callback: PhaseCallback | None = None,
    allow_git_checkout: bool = False,
    _apply_hook: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Download, verify, and atomically apply a code-overlay update with rollback."""

    resolved_root = (module_root or REPO_ROOT).expanduser().resolve()
    check = check_for_code_update(
        module_root=resolved_root,
        manifest_url=manifest_url,
        manifest=manifest,
        timeout=min(timeout, 30),
        offline_dir=offline_dir,
        mirror_prefix=mirror_prefix,
        proxy_url=proxy_url,
        urlopen_fn=urlopen_fn,
    )
    local = check["local"]
    remote = check["remote"]
    if check["status"] == "up_to_date":
        return {
            "status": "up_to_date",
            "version": remote["version"],
            "git_short_commit": remote["git_short_commit"],
            "code_sha256": remote["code_sha256"],
            "updated_files": 0,
            "removed_files": 0,
            "restart_required": False,
        }
    if check["status"] == "runtime_upgrade_required":
        raise RuntimeError(
            "This update requires a newer Python/native runtime package "
            f"(remote ABI {remote['runtime_abi_version']} > local ABI {local['runtime_abi_version']}). "
            f"Please update from {remote['release_url']}."
        )
    if local["is_git_checkout"] and not allow_git_checkout:
        raise RuntimeError(
            "Refusing to overwrite a Git development checkout; run `git pull` instead."
        )
    if not local["writable"]:
        raise PermissionError(
            f"Extension directory is not writable: {resolved_root}"
        )

    # U2: one updater at a time. The OS frees the lock when its process dies, so a killed worker never blocks.
    lock_status, update_lock = acquire_update_lock(resolved_root)
    if lock_status == "busy":
        raise RuntimeError(
            "Another SlicerDSpace update is running for this installation; wait until it finishes, then check again."
        )
    if lock_status != "acquired":
        # R3-U2: without the lock a second worker could interleave its apply or recovery with this one.
        raise RuntimeError(
            f"Cannot take the SlicerDSpace update lock in {resolved_root}; no file was changed. "
            "Check that the extension folder is writable, then check again."
        )
    try:
        # A leftover journal means an earlier apply was killed half way; restore the old files first.
        recovery = recover_interrupted_update(resolved_root, lock_held=True)
        if recovery is not None and recovery.get("status") != "rolled_back":
            # Going on would delete the journal and its backup, the only copy of the old files.
            raise RuntimeError(
                f"An earlier SlicerDSpace update is not finished ({recovery.get('status')}); "
                f"restart Slicer, or reinstall the SlicerDSpace package if this message stays."
            )
        tmp_root = Path(tempfile.mkdtemp(prefix="slicerdspace_code_update_"))
    except BaseException:
        release_update_lock(update_lock)
        raise
    journal_dir = _journal_dir(resolved_root)

    try:
        archive_path = tmp_root / remote["overlay"]["filename"]
        staging_dir = tmp_root / "staging"
        backup_dir = journal_dir / "backup"
        staging_dir.mkdir(parents=True, exist_ok=True)

        if phase_callback is not None:
            phase_callback("downloading")
        _download_verified_overlay(
            remote["overlay"],
            archive_path,
            timeout=timeout,
            offline_dir=offline_dir,
            mirror_prefix=mirror_prefix,
            proxy_url=proxy_url,
            urlopen_fn=urlopen_fn,
            progress_callback=progress_callback,
        )
        if phase_callback is not None:
            phase_callback("verifying")
        file_modes = _extract_overlay_safely(archive_path, staging_dir)
        if not (staging_dir / "SlicerDSpace.py").is_file():
            raise ValueError("Overlay archive does not contain SlicerDSpace.py at its root.")
        staged_sha, new_files_list = compute_code_tree_sha256(staging_dir)
        if staged_sha != remote["code_sha256"]:
            raise ValueError(
                f"Staged code tree SHA-256 mismatch: {staged_sha} != {remote['code_sha256']}"
            )

        try:
            old_files = set(runtime_files(resolved_root))
        except Exception:
            old_files = set()
        new_files = set(new_files_list)
        obsolete_files = sorted(old_files - new_files)

        # From here on files change in place: a kill must not stop the worker (H6/X5).
        if phase_callback is not None:
            phase_callback("applying")
        shutil.rmtree(journal_dir, ignore_errors=True)
        backup_dir.mkdir(parents=True, exist_ok=True)
        backed_up: list[str] = []
        created: list[str] = []
        for rel in sorted(old_files | new_files):
            existing = resolved_root / rel
            if existing.is_file():
                b_dst = backup_dir / rel
                b_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(existing, b_dst)
                backed_up.append(rel)
        created_planned = [rel for rel in sorted(new_files) if not (resolved_root / rel).exists()]
        _write_update_journal(
            journal_dir,
            target_version=str(remote["version"]),
            backed_up=backed_up,
            created=created_planned,
        )

        try:
            # R3-N1: SlicerDSpace.py goes last, after the obsolete files are removed, so a kill before that point
            # leaves the old entry file, which matches the tree that its startup recovery restores.
            for rel in [*(rel for rel in sorted(new_files) if rel != "SlicerDSpace.py"), "SlicerDSpace.py"]:
                if rel == "SlicerDSpace.py":
                    for obsolete in obsolete_files:
                        target = resolved_root / obsolete
                        if target.is_file():
                            target.unlink()
                if _apply_hook is not None:
                    _apply_hook(rel)
                src = staging_dir / rel
                dst = resolved_root / rel
                was_present = dst.exists()
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp_dst = dst.with_name(f".{dst.name}.upd.{os.getpid()}")
                shutil.copy2(src, tmp_dst)
                if os.name != "nt":
                    try:
                        tmp_dst.chmod(file_modes.get(rel, 0o644))
                    except OSError:
                        pass
                _replace_with_retry(tmp_dst, dst)
                if not was_present:
                    created.append(rel)

            _clear_stale_pyc(resolved_root, old_files | new_files)

            applied_sha, _ = compute_code_tree_sha256(resolved_root)
            if applied_sha != remote["code_sha256"]:
                raise RuntimeError(
                    f"Applied code tree verification failed: {applied_sha} != {remote['code_sha256']}"
                )
        except Exception:
            _rollback_apply(resolved_root, backup_dir, backed_up, created)
            _clear_stale_pyc(resolved_root, old_files | new_files)
            _discard_journal(journal_dir)
            raise

        _discard_journal(journal_dir)
        return {
            "status": "applied",
            "version": remote["version"],
            "release_tag": remote["release_tag"],
            "git_short_commit": remote["git_short_commit"],
            "code_sha256": remote["code_sha256"],
            "updated_files": len(new_files),
            "removed_files": len(obsolete_files),
            "restart_required": True,
        }
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
        release_update_lock(update_lock)


def main(argv: list[str] | None = None) -> int:
    """CLI worker entrypoint so Slicer UI runs update checks/applies out-of-process."""

    parser = argparse.ArgumentParser(description="SlicerDSpace incremental code updater worker")
    sub = parser.add_subparsers(dest="command", required=True)

    check_p = sub.add_parser("check", help="Check whether an incremental code update is available")
    check_p.add_argument("--module-root", required=True)
    check_p.add_argument("--result-file", required=True)
    check_p.add_argument("--manifest-url", default=None)
    check_p.add_argument("--timeout", type=int, default=12)

    apply_p = sub.add_parser("apply", help="Download and apply an incremental code overlay")
    apply_p.add_argument("--module-root", required=True)
    apply_p.add_argument("--manifest-file", required=True)
    apply_p.add_argument("--result-file", required=True)
    apply_p.add_argument("--timeout", type=int, default=60)

    args = parser.parse_args(argv)
    result_file = Path(args.result_file).expanduser()
    module_root = Path(args.module_root).expanduser()

    try:
        if args.command == "check":
            print("SLICERDSPACE_UPDATE_STAGE:checking", flush=True)
            res = check_for_code_update(
                module_root=module_root,
                manifest_url=args.manifest_url,
                timeout=int(args.timeout),
            )
        else:
            manifest_payload = json.loads(
                Path(args.manifest_file).expanduser().read_text(encoding="utf-8")
            )
            last_pct = [-1]

            def _on_progress(downloaded: int, total: int) -> None:
                if total <= 0:
                    return
                pct = max(0, min(100, int(downloaded * 100 / total)))
                if pct != last_pct[0]:
                    last_pct[0] = pct
                    print(f"SLICERDSPACE_UPDATE_PROGRESS:{pct}", flush=True)

            def _on_phase(phase: str) -> None:
                print(f"SLICERDSPACE_UPDATE_STAGE:{phase}", flush=True)

            res = download_and_apply_code_overlay(
                module_root=module_root,
                manifest=manifest_payload,
                timeout=int(args.timeout),
                progress_callback=_on_progress,
                phase_callback=_on_phase,
            )
        _write_json_atomic(result_file, {"ok": True, "result": res})
        return 0
    except Exception as exc:
        try:
            _write_json_atomic(result_file, {"ok": False, "error": str(exc)})
        except Exception:
            pass
        print(f"SLICERDSPACE_UPDATE_ERROR:{exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
