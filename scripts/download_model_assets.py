#!/usr/bin/env python3
"""Download and verify SlicerDSpace model assets from a manifest."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import http.client
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "models" / "assets_manifest.json"
LFS_POINTER_HEADER = b"version https://git-lfs.github.com/spec/v1"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.external_process import text_subprocess_kwargs
from shared.model_assets import MODEL_ASSETS_ENV, configured_model_assets_root, resolve_model_assets_root


@dataclass(frozen=True)
class AssetPlan:
    profile: str
    resolved_profiles: tuple[str, ...]
    all_lfs: bool
    include_patterns: tuple[str, ...]
    required_paths: tuple[str, ...]
    external_overrides: tuple[str, ...]
    asset_groups: tuple[str, ...]
    terms_notices: tuple[str, ...]
    source_ids: tuple[str, ...]
    asset_records: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "resolved_profiles": list(self.resolved_profiles),
            "all_lfs": self.all_lfs,
            "include_patterns": list(self.include_patterns),
            "required_paths": list(self.required_paths),
            "external_overrides": list(self.external_overrides),
            "asset_groups": list(self.asset_groups),
            "terms_notices": list(self.terms_notices),
            "source_ids": list(self.source_ids),
            "asset_records": self.asset_records,
        }


DownloadProgressCallback = Callable[[dict[str, Any]], None]
DownloadCancelCheck = Callable[[], bool]


class DownloadCancelledError(RuntimeError):
    """Raised when an interactive model-asset download is cancelled."""


class DownloadIntegrityError(RuntimeError):
    """Raised when a partial/resumed download's byte range or size is untrustworthy.

    Covers: a resumed (206) response whose Content-Range does not actually
    start where we asked it to, a short read that ended without the server
    reporting an error, and a connection dropped mid-transfer. In every
    case the caller should treat this like any other per-source download
    failure (try the next mirror / surface the error) rather than promoting
    a corrupt or incomplete .part file to the final asset path.
    """


def _content_range_starts_at(content_range: str, expected_start: int) -> bool:
    """Return True if an RFC 7233 ``Content-Range: bytes <start>-<end>/<total>`` value starts at ``expected_start``.

    Returns False for a missing or malformed header, since that means the
    actual starting byte cannot be confirmed at all.
    """

    prefix = "bytes "
    if not content_range.startswith(prefix):
        return False
    range_part = content_range[len(prefix):].split("/", 1)[0]
    start_text = range_part.split("-", 1)[0].strip()
    try:
        return int(start_text) == expected_start
    except ValueError:
        return False


def _emit_download_progress(
    callback: DownloadProgressCallback | None,
    *,
    phase: str,
    relative_path: str,
    downloaded_bytes: int = 0,
    total_bytes: int = 0,
    resumed: bool = False,
    source_id: str = "",
) -> None:
    if callback is None:
        return
    callback(
        {
            "phase": phase,
            "path": relative_path,
            "downloaded_bytes": max(0, int(downloaded_bytes)),
            "total_bytes": max(0, int(total_bytes)),
            "resumed": bool(resumed),
            "source_id": source_id,
        }
    )


def _raise_if_download_cancelled(cancel_check: DownloadCancelCheck | None) -> None:
    if cancel_check is not None and bool(cancel_check()):
        raise DownloadCancelledError("Model asset download cancelled; the partial file was kept for resume.")


def _dedupe(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def _manifest_asset_records(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = manifest.get("asset_records", {})
    if not isinstance(records, dict):
        return {}
    result: dict[str, dict[str, Any]] = {}
    default_source_ids = tuple(str(value) for value in manifest.get("default_asset_source_ids", ["github_lfs"]))
    for relative, payload in records.items():
        if not isinstance(payload, dict):
            continue
        record = dict(payload)
        record["path"] = str(relative)
        source_ids = record.get("source_ids")
        if not isinstance(source_ids, list) or not source_ids:
            record["source_ids"] = list(default_source_ids)
        else:
            record["source_ids"] = [str(value) for value in source_ids if str(value or "").strip()]
        result[str(relative)] = record
    return result


def _asset_record_for_path(
    path: str,
    records: dict[str, dict[str, Any]],
    *,
    default_source_ids: tuple[str, ...],
) -> dict[str, Any]:
    record = dict(records.get(path, {}))
    record["path"] = path
    source_ids = record.get("source_ids")
    if not isinstance(source_ids, list) or not source_ids:
        record["source_ids"] = list(default_source_ids)
    return record


def resolve_asset_plan(manifest: dict[str, Any], profile: str | None = None) -> AssetPlan:
    profiles = manifest.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError("Asset manifest must contain a profiles object.")
    selected = profile or str(manifest.get("default_profile") or "core")
    seen: set[str] = set()
    resolved_profiles: list[str] = []
    include_patterns: list[str] = []
    required_paths: list[str] = []
    external_overrides: list[str] = []
    asset_groups: list[str] = []
    terms_notices: list[str] = []
    all_lfs = False
    manifest_records = _manifest_asset_records(manifest)
    default_source_ids = tuple(str(value) for value in manifest.get("default_asset_source_ids", ["github_lfs"]))

    def visit(name: str) -> None:
        nonlocal all_lfs
        if name in seen:
            return
        profile_data = profiles.get(name)
        if not isinstance(profile_data, dict):
            raise ValueError(f"Unsupported asset profile: {name}")
        alias = profile_data.get("alias_for")
        if alias:
            visit(str(alias))
            resolved_profiles.append(name)
            seen.add(name)
            return
        seen.add(name)
        resolved_profiles.append(name)
        if profile_data.get("all_lfs"):
            all_lfs = True
        for child in profile_data.get("include_profiles", []):
            visit(str(child))
        include_patterns.extend(str(value) for value in profile_data.get("include_patterns", []))
        required_paths.extend(str(value) for value in profile_data.get("required_paths", []))
        external_overrides.extend(str(value) for value in profile_data.get("external_overrides", []))
        asset_groups.extend(str(value) for value in profile_data.get("asset_groups", []))
        terms_notices.extend(str(value) for value in profile_data.get("terms_notice", []))

    visit(selected)
    selected_paths: list[str] = []
    if all_lfs:
        selected_paths.extend(sorted(manifest_records))
    elif include_patterns:
        selected_paths.extend(path for path in sorted(manifest_records) if _matches_any(path, tuple(include_patterns)))
    selected_paths.extend(required_paths)
    selected_paths = list(_dedupe(selected_paths))
    selected_records = {
        path: _asset_record_for_path(path, manifest_records, default_source_ids=default_source_ids)
        for path in selected_paths
    }
    source_ids = _dedupe(
        [
            source_id
            for record in selected_records.values()
            for source_id in record.get("source_ids", [])
            if str(source_id or "").strip()
        ]
    )
    return AssetPlan(
        profile=selected,
        resolved_profiles=_dedupe(resolved_profiles),
        all_lfs=all_lfs,
        include_patterns=_dedupe(include_patterns),
        required_paths=tuple(selected_paths),
        external_overrides=_dedupe(external_overrides),
        asset_groups=_dedupe(asset_groups),
        terms_notices=_dedupe(terms_notices),
        source_ids=source_ids,
        asset_records=selected_records,
    )


def _run(command: list[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, cwd=cwd, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **text_subprocess_kwargs()
    )


def _check_git_checkout(repo_root: Path) -> None:
    result = _run(["git", "rev-parse", "--is-inside-work-tree"], cwd=repo_root, check=False)
    if result.returncode != 0:
        raise RuntimeError("This script must run inside a Git checkout.")


def _check_git_lfs(repo_root: Path) -> None:
    result = _run(["git", "lfs", "version"], cwd=repo_root, check=False)
    if result.returncode != 0:
        raise RuntimeError("Git LFS is required. Install Git LFS, then rerun this script.")


def _git_lfs_files(repo_root: Path) -> tuple[str, ...]:
    try:
        result = _run(["git", "lfs", "ls-files", "-n"], cwd=repo_root, check=False)
    except OSError:
        return ()
    if result.returncode != 0:
        return ()
    return tuple(line.strip() for line in result.stdout.splitlines() if line.strip())


def _matches_any(path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def _is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            return stream.read(256).startswith(LFS_POINTER_HEADER)
    except OSError:
        return False


def _sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _git_head(repo_root: Path) -> str:
    result = _run(["git", "rev-parse", "HEAD"], cwd=repo_root, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def verify_asset_plan(repo_root: Path, plan: AssetPlan, *, check_hashes: bool = True) -> dict[str, Any]:
    lfs_files = _git_lfs_files(repo_root)
    checked: list[str] = []
    if plan.all_lfs:
        checked.extend(lfs_files)
    elif plan.include_patterns:
        checked.extend(path for path in lfs_files if _matches_any(path, plan.include_patterns))
    checked.extend(plan.required_paths)
    checked = list(_dedupe(checked))

    missing: list[str] = []
    pointers: list[str] = []
    size_mismatches: list[dict[str, Any]] = []
    sha256_mismatches: list[dict[str, Any]] = []
    hashes_checked = 0
    for relative in checked:
        path = repo_root / relative
        record = plan.asset_records.get(relative, {})
        if not path.is_file():
            missing.append(relative)
            continue
        if _is_lfs_pointer(path):
            pointers.append(relative)
            continue
        expected_size = record.get("size_bytes")
        if expected_size is not None:
            actual_size = path.stat().st_size
            if actual_size != int(expected_size):
                size_mismatches.append(
                    {
                        "path": relative,
                        "expected": int(expected_size),
                        "actual": actual_size,
                    }
                )
        expected_sha256 = str(record.get("sha256") or "").strip()
        if check_hashes and expected_sha256:
            hashes_checked += 1
            actual_sha256 = _sha256(path)
            if actual_sha256 != expected_sha256:
                sha256_mismatches.append(
                    {
                        "path": relative,
                        "expected": expected_sha256,
                        "actual": actual_sha256,
                    }
                )

    return {
        "checked_count": len(checked),
        "hashes_checked": hashes_checked,
        "missing": missing,
        "lfs_pointers": pointers,
        "size_mismatches": size_mismatches,
        "sha256_mismatches": sha256_mismatches,
        "ok": bool(checked) and not missing and not pointers and not size_mismatches and not sha256_mismatches,
    }


def build_asset_lockfile(repo_root: Path, plan: AssetPlan) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for relative in plan.required_paths:
        path = repo_root / relative
        exists = path.is_file()
        is_pointer = _is_lfs_pointer(path) if exists else False
        plan_record = plan.asset_records.get(relative, {})
        records.append(
            {
                "path": relative,
                "exists": exists,
                "size_bytes": path.stat().st_size if exists else 0,
                "expected_size_bytes": plan_record.get("size_bytes"),
                "sha256": _sha256(path) if exists and not is_pointer else None,
                "expected_sha256": plan_record.get("sha256"),
                "lfs_pointer": is_pointer,
                "source_ids": list(plan_record.get("source_ids", plan.source_ids)),
            }
        )
    return {
        "kind": "slicerdsurfer_model_asset_lockfile",
        "version": 1,
        "schema": "slicerddsurfer.model_assets.lockfile.v1",
        "git_commit": _git_head(repo_root),
        "profile": plan.profile,
        "resolved_profiles": list(plan.resolved_profiles),
        "asset_groups": list(plan.asset_groups),
        "terms_notices": list(plan.terms_notices),
        "download_sources": list(plan.source_ids),
        "mirror_policy": "Optional mirrors must preserve path, size, SHA256, and terms notices from this lockfile.",
        "offline_package_policy": "Offline archives should be built from files verified by this lockfile and distributed only after asset terms are confirmed.",
        "records": records,
    }


def _asset_record_ok(path: Path, record: dict[str, Any]) -> bool:
    if not path.is_file() or _is_lfs_pointer(path):
        return False
    expected_size = record.get("size_bytes")
    if expected_size is not None and path.stat().st_size != int(expected_size):
        return False
    expected_sha256 = str(record.get("sha256") or "").strip()
    return not expected_sha256 or _sha256(path) == expected_sha256


def _format_url_template(template: str, *, relative: str, record: dict[str, Any]) -> str:
    encoded_path = urllib.parse.quote(relative, safe="/")
    return template.format(
        path=relative,
        path_urlencoded=encoded_path,
        filename=urllib.parse.quote(Path(relative).name),
        sha256=str(record.get("sha256") or ""),
    )


def _format_bytes(value: Any) -> str:
    try:
        size = float(value)
    except (TypeError, ValueError):
        return "unknown size"
    units = ("B", "KB", "MB", "GB", "TB")
    unit = 0
    while size >= 1024 and unit < len(units) - 1:
        size /= 1024
        unit += 1
    if unit == 0:
        return f"{int(size)} {units[unit]}"
    return f"{size:.1f} {units[unit]}"


def _active_url_source_ids(manifest: dict[str, Any], plan: AssetPlan) -> tuple[str, ...]:
    source_map = manifest.get("asset_sources", {})
    if not isinstance(source_map, dict):
        return ()
    active: list[str] = []
    for source_id in plan.source_ids:
        source = source_map.get(source_id)
        if not isinstance(source, dict):
            continue
        if source.get("status") != "active":
            continue
        if source.get("kind") in {"url-template", "url"}:
            active.append(source_id)
    return _dedupe(active)


def model_asset_setup_guidance(
    plan: AssetPlan,
    manifest: dict[str, Any],
    verification: dict[str, Any] | None = None,
    asset_root: Path | None = None,
) -> dict[str, Any]:
    """Return user-facing remediation guidance for missing model assets."""

    missing = list((verification or {}).get("missing", []))
    pointers = list((verification or {}).get("lfs_pointers", []))
    size_mismatches = list((verification or {}).get("size_mismatches", []))
    sha256_mismatches = list((verification or {}).get("sha256_mismatches", []))
    total_size = sum(int(record.get("size_bytes") or 0) for record in plan.asset_records.values())
    active_url_sources = _active_url_source_ids(manifest, plan)
    preferred_source = active_url_sources[0] if active_url_sources else "slicerdsurfer_models_git"
    default_root = configured_model_assets_root()
    target_root = asset_root.expanduser().resolve() if asset_root is not None else default_root
    command = (
        "PythonSlicer scripts/download_model_assets.py "
        f"--profile {plan.profile} --asset-root {target_root.as_posix()} "
        f"--source url --source-id {preferred_source}"
    )
    verify_command = (
        "PythonSlicer scripts/download_model_assets.py "
        f"--profile {plan.profile} --asset-root {target_root.as_posix()} --verify-only"
    )
    return {
        "profile": plan.profile,
        "resolved_profiles": list(plan.resolved_profiles),
        "required_file_count": len(plan.required_paths),
        "estimated_download_size": _format_bytes(total_size),
        "active_url_sources": list(active_url_sources),
        "preferred_source": preferred_source,
        "recommended_command": command,
        "verify_command": verify_command,
        "default_asset_root": default_root.as_posix(),
        "target_asset_root": target_root.as_posix(),
        "missing_count": len(missing),
        "lfs_pointer_count": len(pointers),
        "size_mismatch_count": len(size_mismatches),
        "sha256_mismatch_count": len(sha256_mismatches),
        "examples": {
            "missing": missing[:8],
            "lfs_pointers": pointers[:8],
            "size_mismatches": size_mismatches[:3],
            "sha256_mismatches": sha256_mismatches[:3],
        },
        "docs": "docs/assets/MODEL_ASSETS.md",
    }


def print_model_asset_setup_guidance(guidance: dict[str, Any], *, stream: Any = sys.stderr) -> None:
    print("[model-assets] Model asset setup guidance:", file=stream)
    print(
        f"[model-assets]   profile={guidance['profile']} "
        f"files={guidance['required_file_count']} estimated_size={guidance['estimated_download_size']}",
        file=stream,
    )
    if guidance["active_url_sources"]:
        print("[model-assets]   active URL sources: " + ", ".join(guidance["active_url_sources"]), file=stream)
    print("[model-assets]   download: " + guidance["recommended_command"], file=stream)
    print("[model-assets]   verify:   " + guidance["verify_command"], file=stream)
    print(f"[model-assets]   env override: export {MODEL_ASSETS_ENV}=/path/to/model-assets", file=stream)
    examples = guidance.get("examples", {})
    if examples.get("missing"):
        print("[model-assets]   missing examples: " + ", ".join(examples["missing"]), file=stream)
    if examples.get("lfs_pointers"):
        print("[model-assets]   LFS pointer examples: " + ", ".join(examples["lfs_pointers"]), file=stream)
    print("[model-assets]   docs: " + guidance["docs"], file=stream)


def _source_url_templates(
    manifest: dict[str, Any],
    source_ids: tuple[str, ...],
    *,
    allow_planned_sources: bool,
) -> list[tuple[str, str]]:
    source_map = manifest.get("asset_sources", {})
    if not isinstance(source_map, dict):
        return []
    templates: list[tuple[str, str]] = []
    for source_id in source_ids:
        source = source_map.get(source_id)
        if not isinstance(source, dict):
            continue
        if str(source.get("kind") or "") not in {"url-template", "url"}:
            continue
        status = str(source.get("status") or "active")
        if status != "active" and not allow_planned_sources:
            continue
        template = str(source.get("url_template") or "").strip()
        if not template:
            base_url = str(source.get("base_url") or "").rstrip("/")
            template = base_url + "/{path_urlencoded}" if base_url else ""
        if template:
            templates.append((source_id, template))
    return templates


def _url_candidates_for_asset(
    manifest: dict[str, Any],
    record: dict[str, Any],
    *,
    source_ids: tuple[str, ...],
    base_urls: tuple[str, ...],
    allow_planned_sources: bool,
) -> list[tuple[str, str]]:
    relative = str(record["path"])
    upstream_only = record.get("redistribution") == "upstream-only"
    candidates: list[tuple[str, str]] = []
    for base_url in (() if upstream_only else base_urls):
        clean_base = base_url.rstrip("/")
        if clean_base:
            candidates.append(("base-url", clean_base + "/" + urllib.parse.quote(relative, safe="/")))
    for key in (("urls",) if upstream_only else ("urls", "mirrors")):
        values = record.get(key, [])
        if isinstance(values, str):
            values = [values]
        if isinstance(values, list):
            for value in values:
                text = str(value or "").strip()
                if text:
                    candidates.append((key.rstrip("s"), _format_url_template(text, relative=relative, record=record)))
    if upstream_only:
        return candidates
    for source_id, template in _source_url_templates(
        manifest,
        source_ids,
        allow_planned_sources=allow_planned_sources,
    ):
        candidates.append((source_id, _format_url_template(template, relative=relative, record=record)))
    return candidates


def _download_url_to_path(
    url: str,
    target: Path,
    *,
    timeout: int = 60,
    relative_path: str = "",
    source_id: str = "",
    expected_size: int = 0,
    progress_callback: DownloadProgressCallback | None = None,
    cancel_check: DownloadCancelCheck | None = None,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")
    resume_from = tmp.stat().st_size if tmp.is_file() else 0
    if expected_size and resume_from > int(expected_size):
        # A .part larger than the asset's recorded size cannot be a valid
        # prefix of it (e.g. left over from a previous, different version of
        # the asset); asking the server to resume from an out-of-range byte
        # would otherwise just get a 416 on every future attempt.
        print(
            f"[model-assets] Existing .part for {target.name} is larger than the expected size; "
            "discarding it and restarting the download.",
            file=sys.stderr,
        )
        tmp.unlink()
        resume_from = 0
    display_path = relative_path or target.name
    _raise_if_download_cancelled(cancel_check)
    _emit_download_progress(
        progress_callback,
        phase="connecting",
        relative_path=display_path,
        downloaded_bytes=resume_from,
        total_bytes=expected_size,
        resumed=bool(resume_from),
        source_id=source_id,
    )
    headers = {"User-Agent": "SlicerDSpace-model-asset-downloader"}
    if resume_from:
        headers["Range"] = f"bytes={resume_from}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", None)
            append = bool(resume_from and status == 206)
            if append:
                content_range = response.headers.get("Content-Range", "")
                if not _content_range_starts_at(content_range, resume_from):
                    # The server's own Content-Range disagrees with the byte
                    # offset we asked it to resume from: the local .part is
                    # not a byte-for-byte prefix of what this URL is
                    # actually serving right now (e.g. the remote file
                    # changed), so keeping it around would just make the
                    # next resume attempt hit this same mismatch again.
                    # Discard it, mirroring the 416 handling below.
                    if tmp.is_file():
                        tmp.unlink()
                    raise DownloadIntegrityError(
                        f"Server returned Content-Range {content_range!r} for {target.name}, "
                        f"expected a range starting at byte {resume_from}; removed the stale "
                        ".part file so the next attempt restarts from scratch."
                    )
            if resume_from and not append:
                print("[model-assets] Server did not resume partial download; restarting " + target.name, file=sys.stderr)
                resume_from = 0
            total_header = response.headers.get("Content-Length")
            header_total = int(total_header) + resume_from if total_header and str(total_header).isdigit() else 0
            expected_total = header_total or max(0, int(expected_size or 0))
            if expected_total:
                print(f"[model-assets]   target size: {_format_bytes(expected_total)}")
            downloaded = resume_from
            _emit_download_progress(
                progress_callback,
                phase="downloading",
                relative_path=display_path,
                downloaded_bytes=downloaded,
                total_bytes=expected_total,
                resumed=bool(resume_from),
                source_id=source_id,
            )
            next_report = downloaded + 16 * 1024 * 1024
            mode = "ab" if append else "wb"
            with tmp.open(mode) as stream:
                while True:
                    _raise_if_download_cancelled(cancel_check)
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    stream.write(chunk)
                    downloaded += len(chunk)
                    _emit_download_progress(
                        progress_callback,
                        phase="downloading",
                        relative_path=display_path,
                        downloaded_bytes=downloaded,
                        total_bytes=expected_total,
                        resumed=bool(resume_from),
                        source_id=source_id,
                    )
                    if expected_total and downloaded >= next_report:
                        percent = 100.0 * downloaded / max(1, expected_total)
                        print(f"[model-assets]   progress: {_format_bytes(downloaded)} / {_format_bytes(expected_total)} ({percent:.1f}%)")
                        next_report = downloaded + 16 * 1024 * 1024
            if expected_total:
                print(f"[model-assets]   complete: {_format_bytes(downloaded)} / {_format_bytes(expected_total)}")
    except urllib.error.HTTPError as exc:
        if exc.code == 416:
            # Range Not Satisfiable: the local .part is stale/invalid for
            # this URL (e.g. the remote file changed). Discard it so the
            # next attempt restarts cleanly instead of repeating the same
            # failed Range request forever.
            if tmp.is_file():
                tmp.unlink()
            raise DownloadIntegrityError(
                f"Server rejected the resume range for {target.name} (416 Range Not Satisfiable); "
                "removed the stale .part file so the next attempt restarts from scratch."
            ) from exc
        raise
    except (http.client.IncompleteRead, ConnectionError) as exc:
        # The .part file on disk keeps whatever was already written (the
        # exception unwinds out of the `with tmp.open(...)` block, which
        # flushes and closes it); do not delete it, so the next attempt can
        # resume instead of restarting from byte zero.
        raise DownloadIntegrityError(
            f"Connection interrupted while downloading {target.name}: {exc}. "
            "Kept the partial .part file for the next resume attempt."
        ) from exc
    if expected_total and downloaded < expected_total:
        # The read loop exited (EOF/short read) without the server raising
        # an error we would otherwise catch above: never promote a short
        # .part file to be the final asset.
        raise DownloadIntegrityError(
            f"Downloaded {downloaded} of {expected_total} expected bytes for {target.name}; "
            "keeping the .part file for the next resume attempt."
        )
    os.replace(tmp, target)
    _emit_download_progress(
        progress_callback,
        phase="complete",
        relative_path=display_path,
        downloaded_bytes=target.stat().st_size,
        total_bytes=expected_total or target.stat().st_size,
        resumed=bool(resume_from),
        source_id=source_id,
    )


def download_asset_urls(
    repo_root: Path,
    plan: AssetPlan,
    manifest: dict[str, Any],
    *,
    source_ids: tuple[str, ...],
    base_urls: tuple[str, ...],
    allow_planned_sources: bool,
    force: bool,
    progress_callback: DownloadProgressCallback | None = None,
    cancel_check: DownloadCancelCheck | None = None,
) -> None:
    if not plan.required_paths:
        raise RuntimeError("No concrete asset paths were resolved for URL download.")
    file_count = len(plan.required_paths)
    for file_index, relative in enumerate(plan.required_paths, start=1):
        _raise_if_download_cancelled(cancel_check)
        record = plan.asset_records.get(relative, {"path": relative, "source_ids": list(plan.source_ids)})
        target = repo_root / relative
        if force:
            # --force means "redownload from scratch": also discard any
            # stale .part left over from a prior attempt, rather than
            # silently resuming it.
            stale_part = target.with_name(target.name + ".part")
            if stale_part.is_file():
                stale_part.unlink()
        if not force and _asset_record_ok(target, record):
            print(f"[model-assets] Existing verified asset: {relative}")
            if progress_callback is not None:
                progress_callback(
                    {
                        "phase": "verified",
                        "path": relative,
                        "downloaded_bytes": target.stat().st_size,
                        "total_bytes": target.stat().st_size,
                        "resumed": False,
                        "source_id": "existing",
                        "file_index": file_index,
                        "file_count": file_count,
                    }
                )
            continue
        source_ids_for_asset = tuple(source_ids or tuple(str(value) for value in record.get("source_ids", plan.source_ids)))
        candidates = _url_candidates_for_asset(
            manifest,
            record,
            source_ids=source_ids_for_asset,
            base_urls=base_urls,
            allow_planned_sources=allow_planned_sources,
        )
        if not candidates:
            raise RuntimeError(
                f"No URL or active mirror is configured for {relative}. "
                "Use --base-url, --source-id with an active manifest source, or keep --source git-lfs."
            )
        errors: list[str] = []
        for source_id, url in candidates:
            try:
                print(f"[model-assets] Downloading {relative} from {source_id}: {url}")
                def report_asset_progress(event: dict[str, Any]) -> None:
                    if progress_callback is None:
                        return
                    progress_callback(
                        {
                            **event,
                            "file_index": file_index,
                            "file_count": file_count,
                        }
                    )

                _download_url_to_path(
                    url,
                    target,
                    relative_path=relative,
                    source_id=source_id,
                    expected_size=int(record.get("size_bytes") or 0),
                    progress_callback=report_asset_progress,
                    cancel_check=cancel_check,
                )
                if not _asset_record_ok(target, record):
                    raise RuntimeError("Downloaded file failed size/SHA256 verification.")
                break
            except DownloadCancelledError:
                raise
            except (OSError, RuntimeError, urllib.error.URLError, urllib.error.HTTPError) as exc:
                errors.append(f"{source_id}: {exc}")
                try:
                    target.unlink()
                except OSError:
                    pass
        else:
            raise RuntimeError(f"Could not download {relative} from any configured source. Errors: {errors}")


def download_asset_plan(
    repo_root: Path,
    plan: AssetPlan,
    manifest: dict[str, Any],
    *,
    source: str = "auto",
    source_ids: tuple[str, ...] = (),
    base_urls: tuple[str, ...] = (),
    allow_planned_sources: bool = False,
    force: bool = False,
    progress_callback: DownloadProgressCallback | None = None,
    cancel_check: DownloadCancelCheck | None = None,
) -> None:
    if source not in {"auto", "git-lfs", "url"}:
        raise ValueError(f"Unsupported download source: {source}")
    if any(record.get("redistribution") == "upstream-only" for record in plan.asset_records.values()):
        if source == "git-lfs":
            raise RuntimeError("Upstream-only resources cannot be acquired from Git LFS mirrors")
        source = "url"
    if source in {"auto", "git-lfs"}:
        try:
            download_asset_plan_from_git_lfs(repo_root, plan)
            return
        except Exception:
            if source == "git-lfs":
                raise
            print("[model-assets] Git LFS download failed or is unavailable; trying URL/mirror sources.", file=sys.stderr)
    download_asset_urls(
        repo_root,
        plan,
        manifest,
        source_ids=source_ids,
        base_urls=base_urls,
        allow_planned_sources=allow_planned_sources,
        force=force,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )


def download_asset_plan_from_git_lfs(repo_root: Path, plan: AssetPlan) -> None:
    _check_git_checkout(repo_root)
    _check_git_lfs(repo_root)
    _run(["git", "lfs", "install", "--local"], cwd=repo_root)
    if plan.all_lfs:
        print("[model-assets] Downloading all Git LFS objects.")
        subprocess.run(["git", "lfs", "pull"], cwd=repo_root, check=True)
        return
    include_patterns = plan.include_patterns or plan.required_paths
    include_csv = ",".join(include_patterns)
    if not include_csv:
        raise RuntimeError(f"Profile {plan.profile} did not resolve any Git LFS include pattern or required path.")
    print(f"[model-assets] profile={plan.profile}")
    print(f"[model-assets] include={include_csv}")
    subprocess.run(["git", "lfs", "pull", f"--include={include_csv}", "--exclude="], cwd=repo_root, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download and verify SlicerDSpace model weights and templates.")
    parser.add_argument("--profile", "-p", help="Asset profile to download. Defaults to manifest default_profile.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Asset manifest JSON path.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Repository root. Defaults to this checkout.")
    parser.add_argument(
        "--asset-root",
        type=Path,
        help=(
            "Directory where model assets are downloaded and verified. Defaults to "
            f"${MODEL_ASSETS_ENV} or a per-user SlicerDSpace model-assets directory."
        ),
    )
    parser.add_argument("--source", choices=("auto", "git-lfs", "url"), default="auto", help="Download backend. Defaults to Git LFS with URL fallback when configured.")
    parser.add_argument("--source-id", action="append", help="Manifest asset source id to use for URL downloads. Can be repeated.")
    parser.add_argument("--base-url", action="append", help="Ad-hoc mirror base URL. Relative asset paths are appended to this URL.")
    parser.add_argument("--allow-planned-sources", action="store_true", help="Allow manifest URL sources marked as planned/TBD.")
    parser.add_argument("--force", action="store_true", help="Redownload URL assets even if an existing file already verifies.")
    parser.add_argument("--no-verify", action="store_true", help="Skip post-download pointer/missing-file verification.")
    parser.add_argument("--verify-only", action="store_true", help="Do not download; only verify selected assets already exist.")
    parser.add_argument("--print-plan", action="store_true", help="Print the resolved manifest plan as JSON and exit.")
    parser.add_argument("--write-lockfile", type=Path, help="Write a SHA256 lockfile for required assets in the resolved profile.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    try:
        manifest = load_manifest(args.manifest)
        plan = resolve_asset_plan(manifest, args.profile)
        asset_root = resolve_model_assets_root(
            explicit=args.asset_root,
            repo_root=repo_root,
            required_paths=tuple(plan.required_paths),
            prefer_existing_legacy=True,
        )
        if args.print_plan:
            payload = plan.to_dict()
            payload["asset_root"] = asset_root.as_posix()
            payload["asset_root_env"] = MODEL_ASSETS_ENV
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if not args.verify_only:
            download_root = asset_root
            source = args.source
            if source == "git-lfs" and download_root != repo_root:
                raise RuntimeError(
                    "Git LFS downloads materialize files in the code checkout. "
                    "Use --source url for custom --asset-root downloads, or omit --asset-root for legacy Git LFS mode."
                )
            if source == "auto" and download_root != repo_root:
                source = "url"
            download_asset_plan(
                repo_root if source == "git-lfs" else download_root,
                plan,
                manifest,
                source=source,
                source_ids=tuple(args.source_id or ()),
                base_urls=tuple(args.base_url or ()),
                allow_planned_sources=bool(args.allow_planned_sources),
                force=bool(args.force),
            )
        if not args.no_verify:
            print(f"[model-assets] Verifying checksums for profile={plan.profile}...", flush=True)
            result = verify_asset_plan(asset_root, plan)
            if not result["ok"]:
                print(json.dumps({"profile": plan.profile, "verification": result}, indent=2, sort_keys=True), file=sys.stderr)
                print_model_asset_setup_guidance(model_asset_setup_guidance(plan, manifest, result, asset_root))
                return 3
            print(f"[model-assets] Verified {result['checked_count']} file(s) for profile={plan.profile}.")
        if args.write_lockfile:
            lockfile = build_asset_lockfile(asset_root, plan)
            lockfile["asset_root"] = asset_root.as_posix()
            lockfile_path = args.write_lockfile.expanduser().resolve()
            lockfile_path.parent.mkdir(parents=True, exist_ok=True)
            lockfile_path.write_text(json.dumps(lockfile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"[model-assets] Wrote lockfile: {lockfile_path}")
        if plan.external_overrides:
            print("[model-assets] Optional external override env vars: " + ", ".join(plan.external_overrides))
        if plan.asset_groups:
            print("[model-assets] Asset groups: " + "; ".join(plan.asset_groups))
        for notice in plan.terms_notices:
            print("[model-assets] Terms notice: " + notice)
        print("[model-assets] Done.")
        return 0
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"[model-assets] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
