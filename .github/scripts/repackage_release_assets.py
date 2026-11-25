#!/usr/bin/env python3
"""GitHub Actions helper to build standalone WMA atlas packs and compact OS Base packages."""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_extension_package import audit_package
from scripts.extension_runtime.archive_layout import inspect_archive_layout
from scripts.extension_runtime.runtime_resources import runtime_files
from scripts.package_modular_assets import (
    ASSETS_RELEASE_BASE_URL,
    ASSETS_RELEASE_TAG,
    CORE_MODEL_PACK_ORDER,
    CORE_MODEL_PACK_SPECS,
    DETERMINISTIC_MTIME,
    RELEASE_BASE_URL,
    RELEASE_TAG,
    WMA_ATLAS_PACK_SPECS,
    WMA_ATLAS_RELEASE_BASE_URL,
    WMA_ATLAS_RELEASE_TAG,
    collect_model_pack_files,
    collect_wma_pack_files,
    manifest_model_pack_entry,
    manifest_pack_entry,
    sha256_file,
    split_file_if_needed,
    write_deterministic_tar_gz,
)
from scripts.package_slicer_extension import _should_mark_executable
from shared.code_updater import EXTENSION_VERSION, RUNTIME_ABI_VERSION, compute_code_tree_sha256

GITHUB_REPO = "NeuroDMRI/SlicerDSpace"
MODEL_ASSETS_TAG = ASSETS_RELEASE_TAG
IMMUTABLE_LEGACY_TAGS = frozenset({"v0.1.1", "model-assets-v0.1.1"})
MACOS_UKF_ARCHIVE = "ukf-standalone-darwin-arm64.tar.gz"
STANDALONE_UKF_ARCHIVES: dict[str, tuple[str, str]] = {
    "windows-x64": ("ukf-standalone-windows-x64.zip", "ukf-windows-x64"),
    "linux-x64": ("ukf-standalone-linux-x64.tar.gz", "ukf-linux-x64"),
    "macos-arm64": ("ukf-standalone-darwin-arm64.tar.gz", "ukf-darwin-arm64"),
}

# Verified immutable v0.1.1 OS Base archives used as the clean runtime bootstrap source when building v1.0.0+.
V011_BOOTSTRAP_BUNDLES: dict[str, dict[str, Any]] = {
    "windows-x64": {
        "filename": "SlicerDSpace-v0.1.1-Windows-x64.zip",
        "base_url": "https://github.com/NeuroDMRI/SlicerDSpace/releases/download/v0.1.1",
        "size_bytes": 1471482327,
        "sha256": "0cd4825374dc209a697c7976d6daceb7f358173ddcb7066fdf9a18be73c667ac",
        "parts": [],
    },
    "linux-x64": {
        "filename": "SlicerDSpace-v0.1.1-Linux-x64.tar.gz",
        "base_url": "https://github.com/NeuroDMRI/SlicerDSpace/releases/download/v0.1.1",
        "size_bytes": 1620043144,
        "sha256": "b96db0274fe8d22cab04f8ce87d441b105196115674b1028de98303e03d85dd8",
        "parts": [],
    },
    "macos-arm64": {
        "filename": "SlicerDSpace-v0.1.1-macOS-arm64.tar.gz",
        "base_url": "https://github.com/NeuroDMRI/SlicerDSpace/releases/download/v0.1.1",
        "size_bytes": 1468508163,
        "sha256": "3277f724e942f5dcd4e203e5d3da3a98dec404234b1871d429a027ac93fa2dab",
        "parts": [],
    },
}

SOURCE_BUNDLES: dict[str, dict[str, str]] = {
    "windows-x64": {
        "bundle_key": "windows_x64_base",
        "os": "windows",
        "arch": "x86_64",
        "out_name_template": "SlicerDSpace-{tag}-Windows-x64.zip",
        "out_name": f"SlicerDSpace-{RELEASE_TAG}-Windows-x64.zip",
        "ext_prefix": "SlicerDSpace",
        "slicer_subdir": "Slicer-5.10",
    },
    "linux-x64": {
        "bundle_key": "linux_x64_base",
        "os": "linux",
        "arch": "x86_64",
        "out_name_template": "SlicerDSpace-{tag}-Linux-x64.tar.gz",
        "out_name": f"SlicerDSpace-{RELEASE_TAG}-Linux-x64.tar.gz",
        "ext_prefix": "SlicerDSpace",
        "slicer_subdir": "Slicer-5.10",
    },
    "macos-arm64": {
        "bundle_key": "macos_arm64_base",
        "os": "macos",
        "arch": "arm64",
        "out_name_template": "SlicerDSpace-{tag}-macOS-arm64.tar.gz",
        "out_name": f"SlicerDSpace-{RELEASE_TAG}-macOS-arm64.tar.gz",
        "ext_prefix": "SlicerDSpace/Slicer.app/Contents/Extensions-34045/SlicerDSpace",
        "slicer_subdir": "Slicer-5.10",
    },
}


def _release_base_url(release_tag: str, repo: str = GITHUB_REPO) -> str:
    return f"https://github.com/{repo}/releases/download/{release_tag}"


def _assert_mutable_release_tag(release_tag: str, *, dry_run: bool = False) -> None:
    if not dry_run and str(release_tag or "").strip() in IMMUTABLE_LEGACY_TAGS:
        raise RuntimeError(
            f"Refusing to modify immutable legacy release tag {release_tag!r}; target v1.0.0 or a newer release tag."
        )


def _fetch_remote_release_assets(
    release_tag: str,
    repo: str,
) -> dict[str, dict[str, Any]]:
    try:
        proc = subprocess.run(
            ["gh", "api", f"repos/{repo}/releases/tags/{release_tag}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return {}
        data = json.loads(proc.stdout)
        return {
            str(a.get("name") or ""): a
            for a in (data.get("assets") or [])
            if isinstance(a, dict) and a.get("name")
        }
    except Exception:
        return {}


def _asset_matches_remote(local_path: Path, expected_size: int, expected_sha256: str, remote_assets: dict[str, dict[str, Any]]) -> bool:
    asset = remote_assets.get(local_path.name)
    if not isinstance(asset, dict):
        return False
    if str(asset.get("state") or "").lower() != "uploaded":
        return False
    if int(asset.get("size") or -1) != expected_size:
        return False
    remote_digest = str(asset.get("digest") or "").strip().lower()
    if remote_digest:
        return remote_digest == f"sha256:{expected_sha256.lower()}"
    return False


def _upload_release_files(
    files: list[Path],
    *,
    release_tag: str = RELEASE_TAG,
    repo: str = GITHUB_REPO,
) -> None:
    file_meta = [(f, f.stat().st_size, sha256_file(f)) for f in files]
    remote_assets = _fetch_remote_release_assets(release_tag, repo)
    pending: list[tuple[Path, int, str]] = []
    for f, sz, digest in file_meta:
        if _asset_matches_remote(f, sz, digest, remote_assets):
            print(f"[upload] Already uploaded with verified SHA-256 on {release_tag}: {f.name}", flush=True)
        else:
            pending.append((f, sz, digest))
    if not pending:
        return

    for attempt in range(1, 6):
        cmd = [
            "gh",
            "release",
            "upload",
            release_tag,
            "--repo",
            repo,
            "--clobber",
            *[str(f) for f, _, _ in pending],
        ]
        proc = subprocess.run(cmd)
        if proc.returncode == 0:
            return
        remote_assets = _fetch_remote_release_assets(release_tag, repo)
        still_pending = [
            (f, sz, digest)
            for f, sz, digest in pending
            if not _asset_matches_remote(f, sz, digest, remote_assets)
        ]
        if not still_pending:
            print(
                f"[upload] gh returned {proc.returncode} on attempt {attempt}/5, "
                f"but all {len(pending)} asset(s) verified uploaded with matching SHA-256 on {release_tag}.",
                flush=True,
            )
            return
        pending = still_pending
        print(
            f"[warn] gh release upload failed (attempt {attempt}/5, {len(pending)} file(s) remaining), retrying...",
            flush=True,
        )
        time.sleep(5 * attempt)
    raise RuntimeError(f"Failed to upload release assets after 5 attempts: {[f for f, _, _ in pending]}")


def _download_verified_part(
    filename: str,
    expected_size: int,
    expected_sha256: str,
    dest: Path,
    *,
    base_url: str = RELEASE_BASE_URL,
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and sha256_file(dest) == expected_sha256:
        return dest
    url = f"{base_url.rstrip('/')}/{filename}"
    for attempt in range(1, 4):
        try:
            print(f"[download] {filename} (attempt {attempt}/3)...", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "SlicerDSpace-repackager"})
            with urllib.request.urlopen(req, timeout=300) as resp, dest.open("wb") as out:
                shutil.copyfileobj(resp, out, length=8 * 1024 * 1024)
            if dest.stat().st_size != expected_size:
                raise RuntimeError(f"Size mismatch for {filename}: {dest.stat().st_size} != {expected_size}")
            actual_sha = sha256_file(dest)
            if actual_sha != expected_sha256:
                raise RuntimeError(f"SHA256 mismatch for {filename}: {actual_sha} != {expected_sha256}")
            return dest
        except Exception as exc:
            dest.unlink(missing_ok=True)
            print(f"[warn] Download retry for {filename}: {exc}", flush=True)
            time.sleep(3 * attempt)
    raise RuntimeError(f"Could not download verified release part: {filename}")


def _download_model_asset(
    filename: str,
    dest: Path,
    *,
    repo: str = GITHUB_REPO,
    tag: str = MODEL_ASSETS_TAG,
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    base_url = f"https://github.com/{repo}/releases/download/{tag}"
    sha_url = f"{base_url}/{filename}.sha256"
    expected_sha = ""
    for attempt in range(1, 4):
        try:
            req = urllib.request.Request(sha_url, headers={"User-Agent": "SlicerDSpace-repackager"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw_sha = resp.read().decode("utf-8", errors="replace").strip()
            expected_sha = raw_sha.split()[0].lower() if raw_sha else ""
            if len(expected_sha) != 64:
                raise RuntimeError(f"Invalid .sha256 contents for {filename}: {raw_sha!r}")
            break
        except Exception as exc:
            print(f"[warn] SHA256 fetch retry for {filename} ({attempt}/3): {exc}", flush=True)
            time.sleep(2 * attempt)
    if len(expected_sha) != 64:
        raise RuntimeError(f"Could not fetch valid SHA-256 sidecar for {filename} from {tag}")

    if dest.is_file() and sha256_file(dest) == expected_sha:
        return dest

    url = f"{base_url}/{filename}"
    for attempt in range(1, 4):
        try:
            print(f"[download] {tag}/{filename} (attempt {attempt}/3)...", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "SlicerDSpace-repackager"})
            with urllib.request.urlopen(req, timeout=300) as resp, dest.open("wb") as out:
                shutil.copyfileobj(resp, out, length=8 * 1024 * 1024)
            actual = sha256_file(dest)
            if actual != expected_sha:
                raise RuntimeError(f"SHA256 mismatch for {filename}: {actual} != {expected_sha}")
            return dest
        except Exception as exc:
            dest.unlink(missing_ok=True)
            print(f"[warn] Download retry for {tag}/{filename}: {exc}", flush=True)
            time.sleep(3 * attempt)
    raise RuntimeError(f"Could not download verified asset {filename} from {tag}")


def _download_verified_surface_runtime(target: str, work_dir: Path) -> tuple[Path, str] | None:
    """Download the catalog-pinned OpenMP macOS arm64 surface runtime archive for repacking."""
    if target != "macos-arm64":
        return None
    catalog = json.loads((REPO_ROOT / "models" / "runtime_bundles.json").read_text(encoding="utf-8"))
    record = next(
        (
            item
            for item in catalog.get("surface", ())
            if item.get("system") == "Darwin"
            and item.get("machine") == "arm64"
            and item.get("published")
        ),
        None,
    )
    if not record:
        return None
    expected_sha = str(record["sha256"]).lower()
    expected_size = int(record["bytes"])
    url = str(record["url"])
    filename = PurePosixPath(str(record["asset"])).name
    dest = work_dir / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size == expected_size and sha256_file(dest) == expected_sha:
        return dest, "surface-runtime-darwin-arm64"
    for attempt in range(1, 4):
        try:
            print(f"[download] surface runtime {filename} (attempt {attempt}/3)...", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "SlicerDSpace-repackager"})
            with urllib.request.urlopen(req, timeout=300) as resp, dest.open("wb") as out:
                shutil.copyfileobj(resp, out, length=8 * 1024 * 1024)
            if dest.stat().st_size != expected_size:
                raise RuntimeError(f"Size mismatch for {filename}: {dest.stat().st_size} != {expected_size}")
            actual_sha = sha256_file(dest)
            if actual_sha != expected_sha:
                raise RuntimeError(f"SHA256 mismatch for {filename}: {actual_sha} != {expected_sha}")
            return dest, "surface-runtime-darwin-arm64"
        except Exception as exc:
            dest.unlink(missing_ok=True)
            print(f"[warn] Surface runtime download retry for {filename}: {exc}", flush=True)
            time.sleep(3 * attempt)
    raise RuntimeError(f"Could not download verified surface runtime {filename} from {url}")


def _should_skip_member(
    norm_path: str,
    ext_prefix: str,
    qt_root: str,
    *,
    decouple_model_assets: bool = False,
    replace_macos_ukf: bool = False,
    replace_ukf_runtime_dir: str | None = None,
    replace_surface_runtime_dir: str | None = None,
    replace_wma_source: bool = False,
    slicer_subdir: str = "Slicer-5.10",
) -> bool:
    if decouple_model_assets:
        bma_root = f"{ext_prefix}/bundled-model-assets"
        if norm_path == bma_root or norm_path.startswith(bma_root + "/"):
            return True
    wma_root = f"{ext_prefix}/bundled-model-assets/bundles/wma"
    if norm_path == wma_root or norm_path.startswith(wma_root + "/"):
        return True
    if norm_path.endswith("/IITmean_b0_256.nii.gz"):
        return True
    if norm_path == qt_root or norm_path.startswith(qt_root + "/"):
        return True
    if replace_wma_source:
        wma_src_prefix = f"{ext_prefix}/bundled-sources/wma-"
        if norm_path.startswith(wma_src_prefix):
            return True
    if replace_ukf_runtime_dir:
        ukf_rt = f"{ext_prefix}/bundled-runtimes/{replace_ukf_runtime_dir}"
        if norm_path == ukf_rt or norm_path.startswith(ukf_rt + "/"):
            return True
    if replace_surface_runtime_dir:
        surf_rt = f"{ext_prefix}/bundled-runtimes/{replace_surface_runtime_dir}"
        if norm_path == surf_rt or norm_path.startswith(surf_rt + "/"):
            return True
    if replace_macos_ukf:
        cli_root = f"{ext_prefix}/lib/{slicer_subdir}/cli-modules"
        ukf_rt_root = f"{ext_prefix}/bundled-runtimes/ukf-darwin-arm64"
        base_lib = f"{ext_prefix}/lib/{slicer_subdir}/libUKFBase.dylib"
        if norm_path == cli_root or norm_path.startswith(cli_root + "/"):
            return True
        if norm_path == ukf_rt_root or norm_path.startswith(ukf_rt_root + "/"):
            return True
        if norm_path == base_lib or (
            norm_path.startswith(f"{ext_prefix}/lib/{slicer_subdir}/libUKFBase") and norm_path.endswith(".dylib")
        ):
            return True
    return False


def cmd_build_model_pack(args: argparse.Namespace) -> int:
    pack_id = args.pack
    assets_tag = str(getattr(args, "assets_tag", "") or getattr(args, "release_tag", "") or ASSETS_RELEASE_TAG)
    repo = str(args.repo or GITHUB_REPO)
    _assert_mutable_release_tag(assets_tag, dry_run=bool(args.dry_run))

    spec = CORE_MODEL_PACK_SPECS[pack_id]
    work_dir = args.work_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = (
        args.asset_root.expanduser().resolve()
        if getattr(args, "asset_root", None)
        else (work_dir / f"model-assets-{pack_id}")
    )
    assets_dir.mkdir(parents=True, exist_ok=True)

    from scripts.download_model_assets import download_asset_urls, load_manifest, resolve_asset_plan

    for manifest_name, profile_name in spec["manifest_sources"]:
        manifest_path = REPO_ROOT / "models" / manifest_name
        manifest = load_manifest(manifest_path)
        plan = resolve_asset_plan(manifest, profile_name)
        download_asset_urls(
            assets_dir,
            plan,
            manifest,
            source_ids=(),
            base_urls=(),
            allow_planned_sources=False,
            force=False,
        )

    files = collect_model_pack_files(pack_id, repo_root=REPO_ROOT)
    filename = spec["filename_template"].format(tag=assets_tag)
    archive_path = out_dir / filename
    size_bytes, digest = write_deterministic_tar_gz(assets_dir, files, archive_path)

    # Verify archive members match expected manifest files 1-to-1 and exclude upstream-only IIT.
    with tarfile.open(archive_path, "r:gz") as tf:
        members = sorted(m.name for m in tf.getmembers() if m.isfile())
    if members != sorted(files):
        raise RuntimeError(f"Pack archive member mismatch for {pack_id}: got {members}, expected {sorted(files)}")
    if any(m.endswith("/IITmean_b0_256.nii.gz") for m in members):
        raise RuntimeError(f"Refusing upstream-only IIT template in pack {pack_id}")

    if not getattr(args, "asset_root", None):
        shutil.rmtree(assets_dir, ignore_errors=True)

    assets_base = _release_base_url(assets_tag, repo)
    parts, upload_paths = split_file_if_needed(
        archive_path,
        release_base_url=assets_base,
        remove_unsplit_when_split=True,
    )
    if not args.dry_run:
        _upload_release_files(upload_paths, release_tag=assets_tag, repo=repo)
        for p in upload_paths:
            p.unlink(missing_ok=True)

    receipt = manifest_model_pack_entry(
        pack_id,
        size_bytes=size_bytes,
        sha256=digest,
        files=files,
        parts=parts,
        published=not args.dry_run,
        release_tag=assets_tag,
        release_base_url=assets_base,
    )
    receipt_path = out_dir / f"model-pack-receipt-{pack_id}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"[model-pack-receipt] Wrote {receipt_path} ({size_bytes:,} bytes, sha256={digest})", flush=True)
    return 0


def apply_model_pack_receipts(
    receipts_dir: Path,
    *,
    release_tag: str = RELEASE_TAG,
    repo: str = GITHUB_REPO,
) -> dict[str, dict[str, Any]]:
    """Merge model-pack-receipt-*.json files into models/release_packs_manifest.json and return core pack map."""
    receipt_files = sorted(receipts_dir.expanduser().resolve().rglob("model-pack-receipt-*.json"))
    packs_manifest_path = REPO_ROOT / "models" / "release_packs_manifest.json"
    packs_manifest = json.loads(packs_manifest_path.read_text(encoding="utf-8"))
    existing_packs = dict(packs_manifest.get("packs") or {})
    ingested: dict[str, dict[str, Any]] = {}
    for rf in receipt_files:
        rec = json.loads(rf.read_text(encoding="utf-8"))
        pid = str(rec.get("id") or "").strip()
        if pid:
            ingested[pid] = rec
            existing_packs[pid] = rec

    ordered_packs: dict[str, dict[str, Any]] = {}
    for pid in CORE_MODEL_PACK_ORDER:
        if pid in existing_packs:
            ordered_packs[pid] = existing_packs[pid]
    for pid in ("wma_org", "wma_naba", "wma_east_west"):
        if pid in existing_packs:
            ordered_packs[pid] = existing_packs[pid]
    for pid, rec in existing_packs.items():
        if pid not in ordered_packs:
            ordered_packs[pid] = rec

    rel_base = _release_base_url(release_tag, repo)
    packs_manifest["release_tag"] = release_tag
    packs_manifest["release_base_url"] = rel_base
    packs_manifest["packs"] = ordered_packs
    packs_manifest_path.write_text(json.dumps(packs_manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[model-pack-receipt] Synced {len(ingested)} core model pack(s) into {packs_manifest_path}", flush=True)

    _update_python_release_tag(REPO_ROOT / "shared" / "installer_script_builder.py", release_tag)
    return {pid: ordered_packs[pid] for pid in CORE_MODEL_PACK_ORDER if pid in ordered_packs}


def _update_python_release_tag(py_path: Path, release_tag: str) -> bool:
    if not py_path.is_file():
        return False
    src = py_path.read_text(encoding="utf-8")
    new_src, count = re.subn(
        r'^(RELEASE_TAG\s*=\s*)"[^"]+"',
        rf'\g<1>"{release_tag}"',
        src,
        count=1,
        flags=re.MULTILINE,
    )
    if count and new_src != src:
        py_path.write_text(new_src, encoding="utf-8")
        print(f"[sync-metadata] Updated RELEASE_TAG={release_tag} in {py_path}", flush=True)
        return True
    return False


def cmd_apply_model_receipts(args: argparse.Namespace) -> int:
    apply_model_pack_receipts(
        args.receipts_dir,
        release_tag=str(args.release_tag or RELEASE_TAG),
        repo=str(args.repo or GITHUB_REPO),
    )
    return 0


def cmd_build_atlas(args: argparse.Namespace) -> int:
    atlas = args.atlas
    spec = WMA_ATLAS_PACK_SPECS[atlas]
    work_dir = args.work_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    assets_dir = work_dir / f"assets-{atlas}"
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "setup_optional_assets.py"),
        "--profiles",
        spec["profile"],
        "--asset-root",
        str(assets_dir),
        "--execute",
        "--accept-terms",
    ]
    subprocess.run(setup_cmd, cwd=str(REPO_ROOT), check=True)

    # Drop raw mirror parts/zips so the redistributable archive holds verified extracted files only.
    for archives_dir in list((assets_dir / "bundles" / "wma").rglob("archives")):
        if archives_dir.is_dir():
            shutil.rmtree(archives_dir, ignore_errors=True)

    ext_manifest = json.loads((REPO_ROOT / "models" / "external_assets_manifest.json").read_text(encoding="utf-8"))
    files = collect_wma_pack_files(assets_dir, ext_manifest, spec["profile"])
    archive_path = out_dir / spec["filename"]
    size_bytes, digest = write_deterministic_tar_gz(assets_dir, files, archive_path)
    shutil.rmtree(assets_dir, ignore_errors=True)

    assets_tag = str(getattr(args, "assets_tag", "") or ASSETS_RELEASE_TAG)
    assets_base = _release_base_url(assets_tag, GITHUB_REPO)
    parts, upload_paths = split_file_if_needed(
        archive_path,
        release_base_url=assets_base,
        remove_unsplit_when_split=True,
    )
    if not args.dry_run:
        _upload_release_files(upload_paths, release_tag=assets_tag, repo=GITHUB_REPO)
    for p in upload_paths:
        p.unlink(missing_ok=True)

    meta_files = {
        k: v
        for k, v in files.items()
        if not k.endswith((".vtp", ".vtk", ".mrml"))
    }
    receipt = manifest_pack_entry(
        atlas,
        size_bytes=size_bytes,
        sha256=digest,
        parts=parts,
        published=not args.dry_run,
        files=meta_files,
    )
    receipt_path = out_dir / f"atlas-receipt-{atlas}.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"[atlas-receipt] Wrote {receipt_path}", flush=True)
    return 0


def cmd_repack_os_base(args: argparse.Namespace) -> int:
    target = args.target
    release_tag = str(getattr(args, "release_tag", "") or RELEASE_TAG)
    repo = str(getattr(args, "repo", "") or GITHUB_REPO)
    decouple_model_assets = not bool(getattr(args, "keep_bundled_model_assets", False))
    _assert_mutable_release_tag(release_tag, dry_run=bool(args.dry_run))

    spec = SOURCE_BUNDLES[target]
    out_name = spec["out_name_template"].format(tag=release_tag)
    rel_base = _release_base_url(release_tag, repo)
    work_dir = args.work_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    explicit_source = getattr(args, "source_archive", None)
    if explicit_source is not None:
        old_archive = Path(explicit_source).expanduser().resolve()
        if not old_archive.is_file():
            raise FileNotFoundError(f"Explicit source archive not found: {old_archive}")
        cleanup_source = False
    else:
        source_tag = str(getattr(args, "source_release_tag", "") or "").strip()
        artifacts_path = REPO_ROOT / "docs" / "website" / "installer_artifacts.json"
        artifacts = json.loads(artifacts_path.read_text(encoding="utf-8"))
        bundle_spec = (artifacts.get("bundles") or {}).get(spec["bundle_key"]) or {}
        if source_tag == "v0.1.1" or not bundle_spec.get("sha256"):
            src_info = V011_BOOTSTRAP_BUNDLES[target]
            source_filename = str(src_info["filename"])
            source_sha256 = str(src_info["sha256"])
            source_size = int(src_info["size_bytes"])
            source_parts = list(src_info.get("parts") or ())
            source_base_url = str(src_info["base_url"])
        else:
            source_filename = str(bundle_spec["filename"])
            source_sha256 = str(bundle_spec["sha256"])
            source_size = int(bundle_spec["size_bytes"])
            source_parts = list(bundle_spec.get("parts") or ())
            source_base_url = (
                _release_base_url(source_tag, repo)
                if source_tag
                else str(artifacts.get("release_base_url") or RELEASE_BASE_URL)
            )

        parts_dir = work_dir / f"parts-{target}"
        old_archive = work_dir / f"{source_filename}.source"
        cleanup_source = True
        if not (old_archive.is_file() and sha256_file(old_archive) == source_sha256):
            if source_parts:
                part_paths = []
                for pspec in source_parts:
                    pname = str(pspec["filename"])
                    psize = int(pspec["size_bytes"])
                    psha = str(pspec["sha256"])
                    part_paths.append(
                        _download_verified_part(pname, psize, psha, parts_dir / pname, base_url=source_base_url)
                    )
                merging = old_archive.with_name(old_archive.name + ".merging")
                h = hashlib.sha256()
                with merging.open("wb") as out_fh:
                    for pp in part_paths:
                        with pp.open("rb") as in_fh:
                            while True:
                                buf = in_fh.read(16 * 1024 * 1024)
                                if not buf:
                                    break
                                out_fh.write(buf)
                                h.update(buf)
                if h.hexdigest() != source_sha256:
                    merging.unlink(missing_ok=True)
                    raise RuntimeError(f"Merged source SHA-256 mismatch for {source_filename}")
                os.replace(merging, old_archive)
                shutil.rmtree(parts_dir, ignore_errors=True)
            else:
                _download_verified_part(
                    source_filename,
                    source_size,
                    source_sha256,
                    old_archive,
                    base_url=source_base_url,
                )

    ext_prefix = spec["ext_prefix"]
    slicer_subdir = spec["slicer_subdir"]
    qt_root = f"{ext_prefix}/lib/{slicer_subdir}/qt-scripted-modules"
    fresh_runtime_files = sorted(runtime_files(REPO_ROOT))
    fresh_wma_source_dir: Path | None = None
    try:
        from shared.backend_sources import prepare_sources

        wma_prep_root = work_dir / "fresh-wma-source"
        wma_res = prepare_sources(("wma",), root=wma_prep_root, execute=True, accept_terms=True)
        cand_wma = Path(wma_res["external_dependencies"]["wma_root"])
        if cand_wma.is_dir():
            fresh_wma_source_dir = cand_wma
            print(f"[repack] Prepared fresh WMA source: {fresh_wma_source_dir.name}", flush=True)
    except Exception as exc:
        print(f"[warn] Could not prepare fresh WMA source ({exc}); keeping existing bundled WMA.", flush=True)
    replace_macos_ukf = (target == "macos-arm64") and (not getattr(args, "skip_macos_ukf", False))
    standalone_ukf_spec = STANDALONE_UKF_ARCHIVES.get(target)
    standalone_ukf_archive: Path | None = None
    standalone_ukf_dir: str | None = None
    if standalone_ukf_spec is not None and not getattr(args, "skip_standalone_ukf", False):
        arch_name, rt_dir_name = standalone_ukf_spec
        standalone_ukf_dir = rt_dir_name
        explicit_ukf = getattr(args, "macos_ukf_archive", None) if target == "macos-arm64" else getattr(args, "standalone_ukf_archive", None)
        if explicit_ukf is not None:
            standalone_ukf_archive = Path(explicit_ukf).expanduser().resolve()
            if not standalone_ukf_archive.is_file():
                raise FileNotFoundError(f"Standalone UKF archive not found: {standalone_ukf_archive}")
        elif target == "macos-arm64" and not replace_macos_ukf:
            standalone_ukf_archive = None
            standalone_ukf_dir = None
        else:
            try:
                standalone_ukf_archive = _download_model_asset(arch_name, work_dir / arch_name)
            except Exception as exc:
                if target == "macos-arm64":
                    raise
                print(f"[warn] Standalone UKF asset {arch_name} unavailable ({exc}); keeping existing UKF.", flush=True)
                standalone_ukf_archive = None
                standalone_ukf_dir = None
    surface_rt_archive: Path | None = None
    surface_rt_dir: str | None = None
    surf_pair = _download_verified_surface_runtime(target, work_dir)
    if surf_pair is not None:
        surface_rt_archive, surface_rt_dir = surf_pair
    new_archive = out_dir / out_name
    tmp_new = new_archive.with_name(new_archive.name + ".tmp")
    saw_qt_entry = False

    if new_archive.name.lower().endswith(".zip"):
        with zipfile.ZipFile(old_archive, "r") as old_zf, zipfile.ZipFile(
            tmp_new, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True, compresslevel=6
        ) as new_zf:
            for info in old_zf.infolist():
                norm = info.filename.replace("\\", "/").strip("/")
                if norm == f"{qt_root}/SlicerDSpace.py":
                    saw_qt_entry = True
                if info.is_dir() or _should_skip_member(
                    norm,
                    ext_prefix,
                    qt_root,
                    decouple_model_assets=decouple_model_assets,
                    replace_macos_ukf=replace_macos_ukf,
                    replace_ukf_runtime_dir=standalone_ukf_dir if standalone_ukf_archive is not None else None,
                    replace_wma_source=fresh_wma_source_dir is not None,
                    slicer_subdir=slicer_subdir,
                ):
                    continue
                if "/bundles/wma" in norm or norm.endswith("/IITmean_b0_256.nii.gz"):
                    raise RuntimeError(f"Refusing forbidden asset in base package: {norm}")
                out_info = zipfile.ZipInfo(filename=norm, date_time=info.date_time)
                mode = (info.external_attr >> 16) & 0o777
                if not mode:
                    mode = 0o755 if _should_mark_executable(Path(norm), PurePosixPath(norm)) else 0o644
                out_info.external_attr = (stat.S_IFREG | mode) << 16
                out_info.compress_type = zipfile.ZIP_DEFLATED
                with old_zf.open(info, "r") as src_fh, new_zf.open(out_info, "w") as dst_fh:
                    shutil.copyfileobj(src_fh, dst_fh, length=1024 * 1024)

            for rel in fresh_runtime_files:
                src = REPO_ROOT / rel
                arcname = f"{qt_root}/{rel}"
                out_info = zipfile.ZipInfo.from_file(src, arcname=arcname, strict_timestamps=False)
                out_info.compress_type = zipfile.ZIP_DEFLATED
                mode = 0o755 if _should_mark_executable(src, Path(rel)) else 0o644
                out_info.external_attr = (stat.S_IFREG | mode) << 16
                with src.open("rb") as src_fh, new_zf.open(out_info, "w") as dst_fh:
                    shutil.copyfileobj(src_fh, dst_fh, length=1024 * 1024)

            if standalone_ukf_archive is not None and standalone_ukf_dir is not None:
                rt_prefix = f"{ext_prefix}/bundled-runtimes"
                with zipfile.ZipFile(standalone_ukf_archive, "r") as ukf_zf:
                    for ui in ukf_zf.infolist():
                        if ui.is_dir():
                            continue
                        unorm = ui.filename.replace("\\", "/").strip("/")
                        if not unorm or ".." in PurePosixPath(unorm).parts:
                            continue
                        if not (unorm == standalone_ukf_dir or unorm.startswith(standalone_ukf_dir + "/")):
                            raise RuntimeError(f"Unexpected member inside standalone UKF archive: {unorm}")
                        arcname = f"{rt_prefix}/{unorm}"
                        out_info = zipfile.ZipInfo(filename=arcname, date_time=ui.date_time)
                        mode = (ui.external_attr >> 16) & 0o777
                        if not mode:
                            mode = 0o755 if unorm.lower().endswith((".exe", ".dll")) else 0o644
                        out_info.external_attr = (stat.S_IFREG | mode) << 16
                        out_info.compress_type = zipfile.ZIP_DEFLATED
                        with ukf_zf.open(ui, "r") as src_fh, new_zf.open(out_info, "w") as dst_fh:
                            shutil.copyfileobj(src_fh, dst_fh, length=1024 * 1024)

            if fresh_wma_source_dir is not None:
                wma_arc_root = f"{ext_prefix}/bundled-sources/{fresh_wma_source_dir.name}"
                for src in sorted(fresh_wma_source_dir.rglob("*")):
                    if not src.is_file() or "__pycache__" in src.parts or src.suffix == ".pyc":
                        continue
                    rel_posix = src.relative_to(fresh_wma_source_dir).as_posix()
                    arcname = f"{wma_arc_root}/{rel_posix}"
                    out_info = zipfile.ZipInfo.from_file(src, arcname=arcname, strict_timestamps=False)
                    out_info.compress_type = zipfile.ZIP_DEFLATED
                    mode = 0o755 if (rel_posix.startswith("bin/") or os.access(src, os.X_OK)) else 0o644
                    out_info.external_attr = (stat.S_IFREG | mode) << 16
                    with src.open("rb") as src_fh, new_zf.open(out_info, "w") as dst_fh:
                        shutil.copyfileobj(src_fh, dst_fh, length=1024 * 1024)
    else:
        written_dirs: set[str] = set()
        with tarfile.open(old_archive, "r:*") as old_tf, tarfile.open(
            tmp_new, mode="w:gz", format=tarfile.PAX_FORMAT, compresslevel=6
        ) as new_tf:
            for member in old_tf:
                norm = member.name.replace("\\", "/").strip("/")
                if norm == f"{qt_root}/SlicerDSpace.py":
                    saw_qt_entry = True
                if member.issym() or member.islnk() or member.isdev():
                    raise RuntimeError(f"Unsupported link/dev member in source archive: {member.name}")
                if not norm or _should_skip_member(
                    norm,
                    ext_prefix,
                    qt_root,
                    decouple_model_assets=decouple_model_assets,
                    replace_macos_ukf=replace_macos_ukf,
                    replace_ukf_runtime_dir=standalone_ukf_dir if standalone_ukf_archive is not None else None,
                    replace_surface_runtime_dir=surface_rt_dir if surface_rt_archive is not None else None,
                    replace_wma_source=fresh_wma_source_dir is not None,
                    slicer_subdir=slicer_subdir,
                ):
                    continue
                if "/bundles/wma" in norm or norm.endswith("/IITmean_b0_256.nii.gz"):
                    raise RuntimeError(f"Refusing forbidden asset in base package: {norm}")
                out_info = tarfile.TarInfo(name=norm)
                out_info.mtime = member.mtime or DETERMINISTIC_MTIME
                out_info.uid = 0
                out_info.gid = 0
                out_info.uname = "root"
                out_info.gname = "root"
                if member.isdir():
                    out_info.type = tarfile.DIRTYPE
                    out_info.mode = member.mode or 0o755
                    if norm not in written_dirs:
                        written_dirs.add(norm)
                        new_tf.addfile(out_info)
                elif member.isfile():
                    out_info.type = tarfile.REGTYPE
                    out_info.size = member.size
                    out_info.mode = member.mode or 0o644
                    with old_tf.extractfile(member) as src_fh:
                        new_tf.addfile(out_info, src_fh)

            def _ensure_tar_dir(dir_posix: str) -> None:
                parts = PurePosixPath(dir_posix).parts
                for i in range(1, len(parts) + 1):
                    sub = "/".join(parts[:i])
                    if sub not in written_dirs:
                        written_dirs.add(sub)
                        d_info = tarfile.TarInfo(name=sub)
                        d_info.type = tarfile.DIRTYPE
                        d_info.mode = 0o755
                        d_info.mtime = DETERMINISTIC_MTIME
                        d_info.uid = 0
                        d_info.gid = 0
                        d_info.uname = "root"
                        d_info.gname = "root"
                        new_tf.addfile(d_info)

            _ensure_tar_dir(qt_root)
            for rel in fresh_runtime_files:
                src = REPO_ROOT / rel
                arcname = f"{qt_root}/{rel}"
                parent_posix = PurePosixPath(arcname).parent.as_posix()
                _ensure_tar_dir(parent_posix)
                f_info = new_tf.gettarinfo(src.as_posix(), arcname=arcname)
                f_info.mode = 0o755 if _should_mark_executable(src, Path(rel)) else 0o644
                f_info.uid = 0
                f_info.gid = 0
                f_info.uname = "root"
                f_info.gname = "root"
                with src.open("rb") as src_fh:
                    new_tf.addfile(f_info, src_fh)

            if standalone_ukf_archive is not None and standalone_ukf_dir is not None:
                rt_prefix = f"{ext_prefix}/bundled-runtimes"
                _ensure_tar_dir(rt_prefix)
                with tarfile.open(standalone_ukf_archive, "r:*") as ukf_tf:
                    for um in ukf_tf:
                        unorm = um.name.replace("\\", "/").strip("/")
                        if not unorm or ".." in PurePosixPath(unorm).parts:
                            continue
                        if not (unorm == standalone_ukf_dir or unorm.startswith(standalone_ukf_dir + "/")):
                            raise RuntimeError(f"Unexpected member inside standalone UKF archive: {unorm}")
                        arcname = f"{rt_prefix}/{unorm}"
                        if um.isdir():
                            _ensure_tar_dir(arcname)
                        elif um.isfile():
                            _ensure_tar_dir(PurePosixPath(arcname).parent.as_posix())
                            uinfo = tarfile.TarInfo(name=arcname)
                            uinfo.type = tarfile.REGTYPE
                            uinfo.size = um.size
                            uinfo.mode = (um.mode & 0o777) or (0o755 if unorm.endswith("/UKFTractography") else 0o644)
                            uinfo.mtime = um.mtime or DETERMINISTIC_MTIME
                            uinfo.uid = 0
                            uinfo.gid = 0
                            uinfo.uname = "root"
                            uinfo.gname = "root"
                            with ukf_tf.extractfile(um) as src_fh:
                                new_tf.addfile(uinfo, src_fh)

            if surface_rt_archive is not None and surface_rt_dir is not None:
                surf_prefix = f"{ext_prefix}/bundled-runtimes/{surface_rt_dir}"
                _ensure_tar_dir(f"{ext_prefix}/bundled-runtimes")
                _ensure_tar_dir(surf_prefix)
                saw_surface_manifest = False
                with tarfile.open(surface_rt_archive, "r:gz") as surf_tf:
                    for sm in surf_tf:
                        snorm = sm.name.replace("\\", "/").strip("/")
                        if not snorm or ".." in PurePosixPath(snorm).parts or "\\" in sm.name or ":" in sm.name:
                            raise RuntimeError(f"Unsafe surface runtime member: {sm.name}")
                        if sm.issym() or sm.islnk() or sm.isdev():
                            raise RuntimeError(f"Unsupported link/dev in surface runtime: {sm.name}")
                        if snorm == "runtime.json":
                            saw_surface_manifest = True
                        arcname = f"{surf_prefix}/{snorm}"
                        if sm.isdir():
                            _ensure_tar_dir(arcname)
                        elif sm.isfile():
                            _ensure_tar_dir(PurePosixPath(arcname).parent.as_posix())
                            sinfo = tarfile.TarInfo(name=arcname)
                            sinfo.type = tarfile.REGTYPE
                            sinfo.size = sm.size
                            sinfo.mode = 0o755 if (sm.mode & 0o111) else 0o644
                            sinfo.mtime = sm.mtime or DETERMINISTIC_MTIME
                            sinfo.uid = 0
                            sinfo.gid = 0
                            sinfo.uname = "root"
                            sinfo.gname = "root"
                            with surf_tf.extractfile(sm) as src_fh:
                                new_tf.addfile(sinfo, src_fh)
                if not saw_surface_manifest:
                    raise RuntimeError(f"Surface runtime archive missing runtime.json: {surface_rt_archive}")
                print(f"[repack] Replaced {surf_prefix} from {surface_rt_archive.name}", flush=True)

            if fresh_wma_source_dir is not None:
                wma_arc_root = f"{ext_prefix}/bundled-sources/{fresh_wma_source_dir.name}"
                _ensure_tar_dir(f"{ext_prefix}/bundled-sources")
                _ensure_tar_dir(wma_arc_root)
                for src in sorted(fresh_wma_source_dir.rglob("*")):
                    if not src.is_file() or "__pycache__" in src.parts or src.suffix == ".pyc":
                        continue
                    rel_posix = src.relative_to(fresh_wma_source_dir).as_posix()
                    arcname = f"{wma_arc_root}/{rel_posix}"
                    _ensure_tar_dir(PurePosixPath(arcname).parent.as_posix())
                    f_info = new_tf.gettarinfo(src.as_posix(), arcname=arcname)
                    f_info.mode = 0o755 if (rel_posix.startswith("bin/") or os.access(src, os.X_OK)) else 0o644
                    f_info.uid = 0
                    f_info.gid = 0
                    f_info.uname = "root"
                    f_info.gname = "root"
                    with src.open("rb") as src_fh:
                        new_tf.addfile(f_info, src_fh)

    if not saw_qt_entry:
        tmp_new.unlink(missing_ok=True)
        raise RuntimeError(f"Expected module root {qt_root}/SlicerDSpace.py was not found in source archive")

    os.replace(tmp_new, new_archive)
    if cleanup_source:
        old_archive.unlink(missing_ok=True)

    audit = audit_package(new_archive, REPO_ROOT)
    layout = inspect_archive_layout(new_archive)
    if audit["status"] != "pass":
        raise RuntimeError(f"audit_package failed for {new_archive.name}: {json.dumps(audit, indent=2)}")
    if layout["status"] != "pass":
        raise RuntimeError(f"inspect_archive_layout failed for {new_archive.name}: {json.dumps(layout, indent=2)}")
    if audit["bundled_wma_atlases"] != [] or layout["bundled_wma_atlases"] != []:
        raise RuntimeError(f"WMA atlases leaked into {new_archive.name}")
    if audit["bundled_python_file_count"] <= 0:
        raise RuntimeError(f"Missing bundled Python runtime in {new_archive.name}")
    if decouple_model_assets:
        if audit["bundled_model_asset_file_count"] != 0:
            raise RuntimeError(
                f"Expected decoupled base archive without bundled model weights, found {audit['bundled_model_asset_file_count']} files in {new_archive.name}"
            )
    elif audit["bundled_model_asset_file_count"] <= 0:
        raise RuntimeError(f"Missing bundled model assets in {new_archive.name}")

    size_bytes = new_archive.stat().st_size
    digest = sha256_file(new_archive)
    parts, upload_paths = split_file_if_needed(
        new_archive,
        release_base_url=rel_base,
        remove_unsplit_when_split=True,
    )
    if not args.dry_run:
        _upload_release_files(upload_paths, release_tag=release_tag, repo=repo)
        for p in upload_paths:
            p.unlink(missing_ok=True)

    receipt: dict[str, Any] = {
        "os": spec["os"],
        "arch": spec["arch"],
        "mode": "base",
        "filename": out_name,
        "url": "" if parts else f"{rel_base}/{out_name}",
        "size_bytes": size_bytes,
        "sha256": digest,
        "published": not args.dry_run,
        "parts": parts,
        "audit": audit,
    }
    receipt_path = out_dir / f"os-base-receipt-{target}.json"
    receipt_path.write_text(json.dumps({spec["bundle_key"]: receipt}, indent=2) + "\n", encoding="utf-8")
    print(f"[os-base-receipt] Wrote {receipt_path} ({size_bytes:,} bytes, parts={len(parts) or 1})", flush=True)
    return 0


_BUNDLE_ORDER = ("windows_x64_base", "linux_x64_base", "macos_arm64_base")
SETUP_OS_ORDER = (
    ("windows", "SlicerDSpaceSetup-Windows.zip", "windows_x64_base", "win"),
    ("linux", "SlicerDSpaceSetup-Linux.zip", "linux_x64_base", "linux"),
    ("macos", "SlicerDSpaceSetup-macOS.zip", "macos_arm64_base", "macosx"),
)


def _render_js_bundles_block(bundles: dict[str, dict[str, Any]]) -> str:
    keys = [k for k in _BUNDLE_ORDER if k in bundles] + [k for k in bundles if k not in _BUNDLE_ORDER]
    lines = ["    // __ARTIFACTS_BUNDLES_BEGIN__", "    bundles: {"]
    for idx, key in enumerate(keys):
        b = bundles[key]
        fn = str(b["filename"])
        parts = list(b.get("parts") or ())
        url_expr = "''" if parts else f"RELEASE_BASE + '/{fn}'"
        lines.append(f"      {key}: {{")
        lines.append(f"        filename: '{fn}',")
        lines.append(f"        url: {url_expr},")
        lines.append(f"        size_bytes: {int(b['size_bytes'])},")
        lines.append(f"        sha256: '{b['sha256']}',")
        if not parts:
            lines.append("        parts: []")
        else:
            lines.append("        parts: [")
            for p_idx, p in enumerate(parts):
                pfn = str(p["filename"])
                pcomma = "," if p_idx + 1 < len(parts) else ""
                lines.append("          {")
                lines.append(f"            filename: '{pfn}',")
                lines.append(f"            size_bytes: {int(p['size_bytes'])},")
                lines.append(f"            sha256: '{p['sha256']}',")
                lines.append(f"            url: RELEASE_BASE + '/{pfn}'")
                lines.append(f"          }}{pcomma}")
            lines.append("        ]")
        comma = "," if idx + 1 < len(keys) else ""
        lines.append(f"      }}{comma}")
    lines.append("    }")
    lines.append("    // __ARTIFACTS_BUNDLES_END__")
    return "\n".join(lines)


def _render_js_model_packs_block(model_packs: dict[str, dict[str, Any]]) -> str:
    keys = [k for k in CORE_MODEL_PACK_ORDER if k in model_packs] + [
        k for k in model_packs if k not in CORE_MODEL_PACK_ORDER
    ]
    lines = ["    // __ARTIFACTS_MODEL_PACKS_BEGIN__", "    model_packs: {"]
    for idx, key in enumerate(keys):
        p = model_packs[key]
        fn = str(p["filename"])
        parts = list(p.get("parts") or ())
        base_var = "ASSETS_RELEASE_BASE" if "/releases/download/assets/" in str(p.get("url") or "") or not p.get("url") else "RELEASE_BASE"
        url_expr = "''" if parts else f"{base_var} + '/{fn}'"
        lines.append(f"      {key}: {{")
        lines.append(f"        id: '{key}',")
        lines.append(f"        filename: '{fn}',")
        lines.append(f"        url: {url_expr},")
        lines.append(f"        size_bytes: {int(p['size_bytes'])},")
        lines.append(f"        sha256: '{p['sha256']}',")
        if not parts:
            lines.append("        parts: []")
        else:
            lines.append("        parts: [")
            for p_idx, sp in enumerate(parts):
                pfn = str(sp["filename"])
                pcomma = "," if p_idx + 1 < len(parts) else ""
                lines.append("          {")
                lines.append(f"            filename: '{pfn}',")
                lines.append(f"            size_bytes: {int(sp['size_bytes'])},")
                lines.append(f"            sha256: '{sp['sha256']}',")
                lines.append(f"            url: {base_var} + '/{pfn}'")
                lines.append(f"          }}{pcomma}")
            lines.append("        ]")
        comma = "," if idx + 1 < len(keys) else ""
        lines.append(f"      }}{comma}")
    lines.append("    },")
    lines.append("    // __ARTIFACTS_MODEL_PACKS_END__")
    return "\n".join(lines)


def _sync_customizer_js_bundles(
    bundles: dict[str, dict[str, Any]],
    *,
    model_packs: dict[str, dict[str, Any]] | None = None,
    release_tag: str | None = None,
) -> Path:
    js_path = REPO_ROOT / "docs" / "site" / "assets" / "js" / "installer-customizer.js"
    js_src = js_path.read_text(encoding="utf-8")
    new_js = js_src

    if release_tag:
        new_js, tag_count = re.subn(
            r"(var\s+RELEASE_TAG\s*=\s*')[^']+(';)",
            rf"\g<1>{release_tag}\g<2>",
            new_js,
            count=1,
        )
        if tag_count != 1:
            raise RuntimeError(f"Could not update RELEASE_TAG in {js_path}")

    if model_packs is not None:
        mp_pattern = re.compile(
            r"^[ \t]*// __ARTIFACTS_MODEL_PACKS_BEGIN__.*?^[ \t]*// __ARTIFACTS_MODEL_PACKS_END__",
            flags=re.MULTILINE | re.DOTALL,
        )
        if not mp_pattern.search(new_js):
            raise RuntimeError(
                f"Missing __ARTIFACTS_MODEL_PACKS_BEGIN__/__ARTIFACTS_MODEL_PACKS_END__ markers in {js_path}"
            )
        new_js = mp_pattern.sub(lambda _m: _render_js_model_packs_block(model_packs), new_js, count=1)

    pattern = re.compile(
        r"^[ \t]*// __ARTIFACTS_BUNDLES_BEGIN__.*?^[ \t]*// __ARTIFACTS_BUNDLES_END__",
        flags=re.MULTILINE | re.DOTALL,
    )
    if not pattern.search(new_js):
        raise RuntimeError(f"Missing __ARTIFACTS_BUNDLES_BEGIN__/__ARTIFACTS_BUNDLES_END__ markers in {js_path}")
    new_js = pattern.sub(lambda _m: _render_js_bundles_block(bundles), new_js, count=1)

    for os_key, _, bundle_key, _ in SETUP_OS_ORDER:
        if bundle_key in bundles:
            sz = int(bundles[bundle_key]["size_bytes"])
            est_pat = re.compile(
                rf"(bundleKey:\s*'{re.escape(bundle_key)}'.*?baseEstimateBytes:\s*)\d+",
                flags=re.DOTALL,
            )
            new_js, count = est_pat.subn(rf"\g<1>{sz}", new_js, count=1)
            if count != 1:
                raise RuntimeError(f"Could not update baseEstimateBytes for {os_key} ({bundle_key}) in {js_path}")

    if new_js != js_src:
        js_path.write_text(new_js, encoding="utf-8")
        print(f"[sync-metadata] Updated {js_path}", flush=True)
    return js_path


def _extract_embedded_setup_config(py_src: str) -> dict[str, Any]:
    tree = ast.parse(py_src)
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "CONFIG" for t in node.targets)
            and isinstance(node.value, ast.Call)
            and len(node.value.args) == 1
            and isinstance(node.value.args[0], ast.Constant)
            and isinstance(node.value.args[0].value, str)
        ):
            return json.loads(node.value.args[0].value)
    raise RuntimeError("Could not extract embedded CONFIG JSON literal from SlicerDSpaceSetup.py")


MACOS_SLICER_REVISIONS: tuple[tuple[str, str], ...] = (
    ("5.10", "34045"),
    ("5.12", "34645"),
    ("5.12", "34627"),
    ("5.12", "34625"),
    ("5.12", "34624"),
    ("5.12", "34621"),
)
GENERIC_SLICER_MINORS: tuple[str, ...] = ("5.10", "5.12", "5.11", "5.13", "5.14")


def _verify_setup_py(
    py_src: str,
    zip_path: Path,
    os_key: str,
    bundle_key: str,
    bundles: dict[str, dict[str, Any]],
    model_packs: dict[str, dict[str, Any]] | None = None,
) -> None:
    if not all(ord(ch) < 128 or ch == "·" for ch in py_src):
        raise RuntimeError(f"Non-ASCII wording detected in {zip_path}")
    cfg = _extract_embedded_setup_config(py_src)
    if cfg.get("auto_start") is not False:
        raise RuntimeError(f"Expected auto_start=False inside {zip_path}, got {cfg.get('auto_start')!r}")
    if cfg.get("os") != os_key or cfg.get("selected_packs") != []:
        raise RuntimeError(f"Unexpected OS/pack defaults inside {zip_path}")
    expected_bundle_sha = str((bundles.get(bundle_key) or {}).get("sha256") or "")
    actual_bundle_sha = str((((cfg.get("artifacts") or {}).get("bundles") or {}).get(bundle_key) or {}).get("sha256") or "")
    if expected_bundle_sha and actual_bundle_sha != expected_bundle_sha:
        raise RuntimeError(
            f"Embedded {bundle_key} sha256 mismatch in {zip_path}: {actual_bundle_sha} != {expected_bundle_sha}"
        )
    if model_packs:
        if cfg.get("selected_model_packs") != list(CORE_MODEL_PACK_ORDER):
            raise RuntimeError(
                f"Expected default selected_model_packs={list(CORE_MODEL_PACK_ORDER)} in {zip_path}, got {cfg.get('selected_model_packs')!r}"
            )
        emb_mp = ((cfg.get("artifacts") or {}).get("model_packs") or {})
        for pid, spec in model_packs.items():
            exp_sha = str(spec.get("sha256") or "")
            act_sha = str((emb_mp.get(pid) or {}).get("sha256") or "")
            if exp_sha and act_sha != exp_sha:
                raise RuntimeError(f"Embedded model pack {pid} sha256 mismatch in {zip_path}: {act_sha} != {exp_sha}")


def _verify_generated_setup_zip(
    zip_path: Path,
    os_key: str,
    bundle_key: str,
    expected_s4ext_os: str,
    bundles: dict[str, dict[str, Any]],
    model_packs: dict[str, dict[str, Any]] | None = None,
) -> None:
    if not zip_path.is_file():
        raise RuntimeError(f"Missing generated setup archive: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupt entry {bad} inside {zip_path}")
        names = set(zf.namelist())
        verified_py = False
        if os_key == "macos":
            for minor, rev in MACOS_SLICER_REVISIONS:
                prefix = f"SlicerDSpaceSetup/Slicer.app/Contents/Extensions-{rev}/SlicerDSpaceSetup"
                s4ext_rel = f"{prefix}/share/Slicer-{minor}/SlicerDSpaceSetup.s4ext"
                root_s4ext_rel = f"{prefix}/SlicerDSpaceSetup.s4ext"
                py_rel = f"{prefix}/lib/Slicer-{minor}/qt-scripted-modules/SlicerDSpaceSetup.py"
                for required in (s4ext_rel, root_s4ext_rel, py_rel):
                    if required not in names:
                        raise RuntimeError(f"Missing {required} inside {zip_path}")
                s4ext_text = zf.read(s4ext_rel).decode("utf-8")
                if f"os {expected_s4ext_os}" not in s4ext_text or f"slicer_revision {rev}" not in s4ext_text:
                    raise RuntimeError(f"Unexpected .s4ext metadata inside {zip_path} ({s4ext_rel})")
                if not verified_py:
                    _verify_setup_py(
                        zf.read(py_rel).decode("utf-8"), zip_path, os_key, bundle_key, bundles, model_packs=model_packs
                    )
                    verified_py = True
        else:
            if "SlicerDSpaceSetup/SlicerDSpaceSetup.s4ext" not in names:
                raise RuntimeError(f"Missing SlicerDSpaceSetup/SlicerDSpaceSetup.s4ext inside {zip_path}")
            for minor in GENERIC_SLICER_MINORS:
                s4ext_rel = f"SlicerDSpaceSetup/share/Slicer-{minor}/SlicerDSpaceSetup.s4ext"
                py_rel = f"SlicerDSpaceSetup/lib/Slicer-{minor}/qt-scripted-modules/SlicerDSpaceSetup.py"
                for required in (s4ext_rel, py_rel):
                    if required not in names:
                        raise RuntimeError(f"Missing {required} inside {zip_path}")
                s4ext_text = zf.read(s4ext_rel).decode("utf-8")
                if f"os {expected_s4ext_os}" not in s4ext_text:
                    raise RuntimeError(f"Unexpected .s4ext metadata inside {zip_path} ({s4ext_rel})")
                if not verified_py:
                    _verify_setup_py(
                        zf.read(py_rel).decode("utf-8"), zip_path, os_key, bundle_key, bundles, model_packs=model_packs
                    )
                    verified_py = True


def build_setup_zips(
    out_dir: Path,
    installer_artifacts: dict[str, Any],
    *,
    release_tag: str = RELEASE_TAG,
    published: bool = True,
) -> tuple[dict[str, dict[str, Any]], list[Path]]:
    node_bin = shutil.which("node")
    if node_bin is None:
        raise RuntimeError("node is required in PATH to build SlicerDSpaceSetup-<OS>.zip packages")

    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    js_path = REPO_ROOT / "docs" / "site" / "assets" / "js" / "installer-customizer.js"

    node_script = (
        "const fs = require('fs');\n"
        "const path = require('path');\n"
        "const ic = require(process.argv[1]);\n"
        "const outDir = process.argv[2];\n"
        "(async () => {\n"
        "  for (const osKey of ['windows', 'linux', 'macos']) {\n"
        "    const bytes = await ic.buildSetupZip({\n"
        "      os: osKey,\n"
        "      wmaEnabled: false,\n"
        "      wmaAtlases: [],\n"
        "      autoStart: false,\n"
        "    });\n"
        "    const outFile = path.join(outDir, ic.OS_DEFAULTS[osKey].setupZipName);\n"
        "    fs.writeFileSync(outFile, Buffer.from(bytes));\n"
        "  }\n"
        "})().catch((err) => { console.error(err); process.exit(1); });\n"
    )
    proc = subprocess.run([node_bin, "-e", node_script, str(js_path), str(out_dir)], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"Failed generating setup zips via node:\n{proc.stderr or proc.stdout}")

    base_url = str(
        installer_artifacts.get("release_base_url")
        or f"https://github.com/{GITHUB_REPO}/releases/download/{release_tag}"
    ).rstrip("/")
    bundles = installer_artifacts.get("bundles") or {}
    model_packs = installer_artifacts.get("model_packs") or {}
    setup_meta: dict[str, dict[str, Any]] = {}
    written_paths: list[Path] = []

    for os_key, filename, bundle_key, expected_s4ext_os in SETUP_OS_ORDER:
        zip_path = out_dir / filename
        _verify_generated_setup_zip(
            zip_path, os_key, bundle_key, expected_s4ext_os, bundles, model_packs=model_packs or None
        )
        size_bytes = zip_path.stat().st_size
        digest = sha256_file(zip_path)
        setup_meta[os_key] = {
            "os": os_key,
            "filename": filename,
            "url": f"{base_url}/{filename}",
            "size_bytes": size_bytes,
            "sha256": digest,
            "published": published,
        }
        written_paths.append(zip_path)
        print(f"[setup-zip] Verified {zip_path.name} ({size_bytes:,} bytes, sha256={digest})", flush=True)

    return setup_meta, written_paths


def build_code_overlay(
    out_dir: Path,
    *,
    release_tag: str = RELEASE_TAG,
    repo: str = GITHUB_REPO,
) -> tuple[dict[str, Any], list[Path]]:
    """Build deterministic SlicerDSpace-<tag>-CodeOverlay.tar.gz and update_manifest.json."""

    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    code_sha256, rel_files = compute_code_tree_sha256(REPO_ROOT)
    filename = f"SlicerDSpace-{release_tag}-CodeOverlay.tar.gz"
    overlay_path = out_dir / filename
    tmp_path = overlay_path.with_name(overlay_path.name + ".tmp")

    try:
        with gzip.GzipFile(filename=tmp_path, mode="wb", compresslevel=6, mtime=DETERMINISTIC_MTIME) as gz_fh:
            with tarfile.open(fileobj=gz_fh, mode="w", format=tarfile.PAX_FORMAT) as tf:
                for rel in rel_files:
                    src = REPO_ROOT / rel
                    info = tf.gettarinfo(src.as_posix(), arcname=rel)
                    info.mode = 0o755 if _should_mark_executable(src, Path(rel)) else 0o644
                    info.mtime = DETERMINISTIC_MTIME
                    info.uid = 0
                    info.gid = 0
                    info.uname = "root"
                    info.gname = "root"
                    with src.open("rb") as in_fh:
                        tf.addfile(info, in_fh)
        os.replace(tmp_path, overlay_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    overlay_size = overlay_path.stat().st_size
    overlay_sha = sha256_file(overlay_path)

    git_commit = ""
    summary = ""
    try:
        git_commit = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        summary = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "log", "-1", "--pretty=%s"],
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).strip()
    except Exception:
        pass

    base_download = f"https://github.com/{repo}/releases/download/{release_tag}"
    manifest = {
        "schema_version": 1,
        "version": EXTENSION_VERSION,
        "release_tag": release_tag,
        "runtime_abi_version": RUNTIME_ABI_VERSION,
        "git_commit": git_commit,
        "git_short_commit": git_commit[:7] if git_commit else "",
        "code_sha256": code_sha256,
        "file_count": len(rel_files),
        "published_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "summary": summary,
        "release_url": f"https://github.com/{repo}/releases/tag/{release_tag}",
        "overlay": {
            "filename": filename,
            "url": f"{base_download}/{filename}",
            "size_bytes": overlay_size,
            "sha256": overlay_sha,
        },
    }
    manifest_path = out_dir / "update_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"[code-overlay] Built {filename} ({overlay_size:,} bytes, {len(rel_files)} files, tree={code_sha256[:12]})",
        flush=True,
    )
    return manifest, [overlay_path, manifest_path]


def _format_sha256sums(
    artifacts: dict[str, Any],
    packs_manifest: dict[str, Any],
    *,
    release_tag: str | None = None,
) -> str:
    tag = str(release_tag or artifacts.get("release_tag") or RELEASE_TAG)
    entries: list[tuple[str, str]] = []
    merged_entries: list[tuple[str, str]] = []
    seen_names: set[str] = set()

    def _add(sha: str, name: str, *, merged: bool = False) -> None:
        if not sha or not name or name in seen_names:
            return
        seen_names.add(name)
        (merged_entries if merged else entries).append((sha, name))

    bundles = artifacts.get("bundles") or {}
    for key in [k for k in _BUNDLE_ORDER if k in bundles] + [k for k in bundles if k not in _BUNDLE_ORDER]:
        b = bundles[key]
        parts = list(b.get("parts") or ())
        if parts:
            for p in parts:
                _add(str(p["sha256"]), str(p["filename"]))
            if b.get("sha256") and b.get("filename"):
                _add(str(b["sha256"]), str(b["filename"]), merged=True)
        elif b.get("sha256"):
            _add(str(b["sha256"]), str(b["filename"]))

    tag_segment = f"/releases/download/{tag}/"

    model_packs = artifacts.get("model_packs") or {}
    for key in [k for k in CORE_MODEL_PACK_ORDER if k in model_packs] + [
        k for k in model_packs if k not in CORE_MODEL_PACK_ORDER
    ]:
        p = model_packs[key]
        parts = list(p.get("parts") or ())
        urls = [str(sp.get("url") or "") for sp in parts] if parts else [str(p.get("url") or "")]
        if urls and not any(tag_segment in u for u in urls):
            continue
        if parts:
            for sp in parts:
                _add(str(sp["sha256"]), str(sp["filename"]))
            if p.get("sha256") and p.get("filename"):
                _add(str(p["sha256"]), str(p["filename"]), merged=True)
        elif p.get("sha256"):
            _add(str(p["sha256"]), str(p["filename"]))

    code_overlay = artifacts.get("code_overlay") or {}
    if code_overlay.get("sha256") and code_overlay.get("filename"):
        _add(str(code_overlay["sha256"]), str(code_overlay["filename"]))
    setup_installers = artifacts.get("setup_installers") or {}
    for os_key, _, _, _ in SETUP_OS_ORDER:
        rec = setup_installers.get(os_key)
        if rec and rec.get("sha256") and rec.get("filename"):
            _add(str(rec["sha256"]), str(rec["filename"]))

    packs = packs_manifest.get("packs") or {}
    pack_order = [k for k in (*CORE_MODEL_PACK_ORDER, "wma_org", "wma_naba", "wma_east_west") if k in packs] + [
        k for k in packs if k not in (*CORE_MODEL_PACK_ORDER, "wma_org", "wma_naba", "wma_east_west")
    ]
    for key in pack_order:
        p = packs[key]
        parts = list(p.get("parts") or ())
        urls = [str(sp.get("url") or "") for sp in parts] if parts else [str(p.get("url") or "")]
        if urls and not any(tag_segment in u for u in urls):
            continue
        if parts:
            for sp in parts:
                _add(str(sp["sha256"]), str(sp["filename"]))
            if p.get("sha256") and p.get("filename"):
                _add(str(p["sha256"]), str(p["filename"]), merged=True)
        elif p.get("sha256"):
            _add(str(p["sha256"]), str(p["filename"]))

    out = "".join(f"{sha}  {name}\n" for sha, name in entries if sha and name)
    if merged_entries:
        out += "\n# Merged multi-part archive SHA-256 (after concatenating .001..N parts):\n"
        out += "".join(f"# {sha}  {name}\n" for sha, name in merged_entries)
    return out


def _sync_release_tag_references(release_tag: str) -> None:
    """Update CLI installer/CI/website documentation references when publishing a new release tag (e.g. v1.0.0)."""
    _update_python_release_tag(REPO_ROOT / "shared" / "installer_script_builder.py", release_tag)
    _update_python_release_tag(REPO_ROOT / ".github" / "scripts" / "ci_install_dspace.py", release_tag)
    if release_tag == WMA_ATLAS_RELEASE_TAG:
        return

    index_path = REPO_ROOT / "docs" / "site" / "index.html"
    if index_path.is_file():
        src = index_path.read_text(encoding="utf-8")
        new_src = src
        new_src = re.sub(
            r"(/releases/download/)v[0-9]+\.[0-9]+\.[0-9]+(/SlicerDSpace-v[0-9]+\.[0-9]+\.[0-9]+-(?:Linux-x64\.tar\.gz|Windows-x64\.zip|macOS-arm64\.tar\.gz))",
            rf"\g<1>{release_tag}\2",
            new_src,
        )
        new_src = re.sub(
            r"SlicerDSpace-v[0-9]+\.[0-9]+\.[0-9]+-(Linux-x64\.tar\.gz|Windows-x64\.zip|macOS-arm64\.tar\.gz)",
            rf"SlicerDSpace-{release_tag}-\1",
            new_src,
        )
        new_src = new_src.replace(
            " (with the v0.1.1 archive on Linux, UKF tractography requires this update):",
            ":",
        )
        old_opt_dd = (
            "Base archives bundle Python packages, core deep-learning weights and native runtimes; fetch the NITRC "
            "<code>IITmean_b0_256.nii.gz</code> template for PNL CNN masking or optional WMA atlases "
            "(<code>wma</code>, <code>wma-naba</code>, <code>wma-east-west</code>) by previewing with "
            '<code>"$SLICER_LAUNCHER" --launch PythonSlicer "$DSPACE_REPO/scripts/setup_optional_assets.py" '
            "--profiles pnl-cnn</code> and then adding <code>--execute --accept-terms</code> to install."
        )
        new_opt_dd = (
            "OS base archives bundle Python packages and native runtimes; model weights are packaged as separate "
            "modular packs (installed by default in Route A Setup). When unpacking a manual CLI archive, fetch core "
            'model weights via <code>"$SLICER_LAUNCHER" --launch PythonSlicer '
            '"$DSPACE_REPO/scripts/download_model_assets.py" --profile core</code>, and fetch PNL CNN weights, '
            "TractCloud, the NITRC <code>IITmean_b0_256.nii.gz</code> template, or optional WMA atlases "
            "(<code>wma</code>, <code>wma-naba</code>, <code>wma-east-west</code>) by previewing with "
            '<code>"$SLICER_LAUNCHER" --launch PythonSlicer "$DSPACE_REPO/scripts/setup_optional_assets.py" '
            "--profiles pnl-cnn tractcloud surface-postprocess</code> and then adding "
            "<code>--execute --accept-terms</code> to install."
        )
        new_src = new_src.replace(old_opt_dd, new_opt_dd)
        if new_src != src:
            index_path.write_text(new_src, encoding="utf-8")
            print(f"[sync-metadata] Updated CLI release references in {index_path}", flush=True)

    editorial_test = REPO_ROOT / "Testing" / "Python" / "test_website_editorial_copy.py"
    if editorial_test.is_file():
        tsrc = editorial_test.read_text(encoding="utf-8")
        new_tsrc = re.sub(
            r"SlicerDSpace-v[0-9]+\.[0-9]+\.[0-9]+-Linux-x64\.tar\.gz",
            f"SlicerDSpace-{release_tag}-Linux-x64.tar.gz",
            tsrc,
        )
        if new_tsrc != tsrc:
            editorial_test.write_text(new_tsrc, encoding="utf-8")
            print(f"[sync-metadata] Updated CLI marker in {editorial_test}", flush=True)


def cmd_sync_metadata(args: argparse.Namespace) -> int:
    release_tag = str(args.release_tag or RELEASE_TAG)
    repo = str(args.repo or GITHUB_REPO)
    _assert_mutable_release_tag(release_tag, dry_run=bool(args.dry_run))
    rel_base = _release_base_url(release_tag, repo)

    receipts_dir = args.receipts_dir.expanduser().resolve()
    receipt_files = sorted(receipts_dir.rglob("os-base-receipt-*.json"))
    model_receipt_files = sorted(receipts_dir.rglob("model-pack-receipt-*.json"))
    if not receipt_files and not model_receipt_files:
        raise RuntimeError(f"No os-base-receipt-*.json or model-pack-receipt-*.json files found under {receipts_dir}")

    core_model_packs: dict[str, dict[str, Any]] = {}
    if model_receipt_files:
        core_model_packs = apply_model_pack_receipts(receipts_dir, release_tag=release_tag, repo=repo)
    else:
        packs_manifest_existing = json.loads((REPO_ROOT / "models" / "release_packs_manifest.json").read_text(encoding="utf-8"))
        for pid in CORE_MODEL_PACK_ORDER:
            rec = (packs_manifest_existing.get("packs") or {}).get(pid)
            if isinstance(rec, dict) and rec.get("sha256"):
                core_model_packs[pid] = rec

    artifacts_path = REPO_ROOT / "docs" / "website" / "installer_artifacts.json"
    artifacts = json.loads(artifacts_path.read_text(encoding="utf-8"))
    artifacts["release_tag"] = release_tag
    artifacts["release_base_url"] = rel_base
    bundles = artifacts.setdefault("bundles", {})
    allowed_keys = ("os", "arch", "mode", "filename", "url", "size_bytes", "sha256", "published", "parts")
    for rf in receipt_files:
        data = json.loads(rf.read_text(encoding="utf-8"))
        for key, rec in data.items():
            rec_copy = {k: rec[k] for k in allowed_keys if k in rec}
            bundles[key] = rec_copy

    compact_model_packs: dict[str, dict[str, Any]] = {}
    pack_keys = ("id", "tier", "default_install", "filename", "url", "size_bytes", "sha256", "published", "parts")
    for pid in CORE_MODEL_PACK_ORDER:
        rec = core_model_packs.get(pid)
        if isinstance(rec, dict) and rec.get("sha256"):
            entry = {k: rec[k] for k in pack_keys if k in rec}
            entry.setdefault("parts", [])
            compact_model_packs[pid] = entry
    if compact_model_packs:
        artifacts["model_packs"] = compact_model_packs

    # Keep models/release_packs_manifest.json bundles in sync with installer_artifacts.json BEFORE CodeOverlay!
    packs_manifest_path = REPO_ROOT / "models" / "release_packs_manifest.json"
    packs_manifest = json.loads(packs_manifest_path.read_text(encoding="utf-8"))
    packs_manifest["release_tag"] = release_tag
    packs_manifest["release_base_url"] = rel_base
    packs_manifest["bundles"] = bundles
    packs_manifest_path.write_text(json.dumps(packs_manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[sync-metadata] Updated {packs_manifest_path}", flush=True)

    _sync_release_tag_references(release_tag)
    _sync_customizer_js_bundles(
        bundles,
        model_packs=compact_model_packs if compact_model_packs else None,
        release_tag=release_tag,
    )

    setup_dir = (args.setup_out_dir or (receipts_dir / "setup-zips")).expanduser().resolve()
    setup_meta, setup_paths = build_setup_zips(
        setup_dir,
        artifacts,
        release_tag=release_tag,
        published=not args.dry_run,
    )
    artifacts["setup_installers"] = setup_meta
    artifacts.pop("cross_version_installers", None)
    artifacts.pop("universal_installers", None)

    overlay_manifest, overlay_paths = build_code_overlay(
        setup_dir / "overlay",
        release_tag=release_tag,
        repo=repo,
    )
    artifacts["code_overlay"] = {
        **overlay_manifest["overlay"],
        "code_sha256": overlay_manifest["code_sha256"],
        "runtime_abi_version": overlay_manifest["runtime_abi_version"],
        "version": overlay_manifest["version"],
    }

    artifacts_path.write_text(json.dumps(artifacts, indent=2) + "\n", encoding="utf-8")
    print(f"[sync-metadata] Updated {artifacts_path}", flush=True)

    sha_out = (args.sha256sums_out or (receipts_dir / "SHA256SUMS.txt")).expanduser().resolve()
    sha_out.parent.mkdir(parents=True, exist_ok=True)
    sha_out.write_text(_format_sha256sums(artifacts, packs_manifest, release_tag=release_tag), encoding="utf-8")
    print(f"[sync-metadata] Wrote {sha_out}", flush=True)

    if args.upload_sha256sums and not args.dry_run:
        _upload_release_files(
            [*overlay_paths, *setup_paths, sha_out],
            release_tag=release_tag,
            repo=repo,
        )
        # Keep software releases clean: once all 5 core model packs are hosted on the permanent
        # 'assets' release, prune any legacy SlicerDSpace-<tag>-Model-*.tar.gz off the software release.
        if release_tag != ASSETS_RELEASE_TAG:
            assets_remote = _fetch_remote_release_assets(ASSETS_RELEASE_TAG, repo)
            all_on_assets = all(
                _asset_matches_remote(
                    Path(str((compact_model_packs.get(pid) or {}).get("filename") or "")),
                    int((compact_model_packs.get(pid) or {}).get("size_bytes") or 0),
                    str((compact_model_packs.get(pid) or {}).get("sha256") or ""),
                    assets_remote,
                )
                for pid in CORE_MODEL_PACK_ORDER
                if pid in compact_model_packs
            )
            if all_on_assets:
                rel_remote = _fetch_remote_release_assets(release_tag, repo)
                legacy_prefix = f"SlicerDSpace-{release_tag}-Model-"
                for name in sorted(rel_remote):
                    if name.startswith(legacy_prefix):
                        print(f"[cleanup] Removing migrated model pack from {release_tag}: {name}", flush=True)
                        subprocess.run(
                            ["gh", "release", "delete-asset", release_tag, name, "--repo", repo, "--yes"],
                            check=False,
                        )
    return 0


def cmd_sync_code_overlay(args: argparse.Namespace) -> int:
    _assert_mutable_release_tag(args.release_tag, dry_run=bool(args.dry_run))
    artifacts_path = REPO_ROOT / "docs" / "website" / "installer_artifacts.json"
    artifacts = json.loads(artifacts_path.read_text(encoding="utf-8"))
    out_dir = args.out_dir.expanduser().resolve()
    overlay_manifest, overlay_paths = build_code_overlay(
        out_dir,
        release_tag=args.release_tag,
        repo=args.repo,
    )
    artifacts["code_overlay"] = {
        **overlay_manifest["overlay"],
        "code_sha256": overlay_manifest["code_sha256"],
        "runtime_abi_version": overlay_manifest["runtime_abi_version"],
        "version": overlay_manifest["version"],
    }
    artifacts_path.write_text(json.dumps(artifacts, indent=2) + "\n", encoding="utf-8")
    print(f"[sync-code-overlay] Updated {artifacts_path}", flush=True)

    packs_manifest_path = REPO_ROOT / "models" / "release_packs_manifest.json"
    packs_manifest = json.loads(packs_manifest_path.read_text(encoding="utf-8"))
    sha_out = (args.sha256sums_out or (out_dir / "SHA256SUMS.txt")).expanduser().resolve()
    sha_out.parent.mkdir(parents=True, exist_ok=True)
    sha_out.write_text(_format_sha256sums(artifacts, packs_manifest, release_tag=args.release_tag), encoding="utf-8")
    print(f"[sync-code-overlay] Wrote {sha_out}", flush=True)

    if not args.dry_run:
        _upload_release_files(
            [*overlay_paths, sha_out],
            release_tag=args.release_tag,
            repo=args.repo,
        )
    return 0


def cmd_sync_setup_zips(args: argparse.Namespace) -> int:
    _assert_mutable_release_tag(args.release_tag, dry_run=bool(args.dry_run))
    artifacts_path = REPO_ROOT / "docs" / "website" / "installer_artifacts.json"
    artifacts = json.loads(artifacts_path.read_text(encoding="utf-8"))
    bundles = artifacts.get("bundles") or {}
    model_packs = artifacts.get("model_packs") or None
    _sync_customizer_js_bundles(bundles, model_packs=model_packs, release_tag=args.release_tag)

    out_dir = args.out_dir.expanduser().resolve()
    setup_meta, setup_paths = build_setup_zips(
        out_dir,
        artifacts,
        release_tag=args.release_tag,
        published=not args.dry_run,
    )
    artifacts["setup_installers"] = setup_meta
    artifacts.pop("cross_version_installers", None)
    artifacts.pop("universal_installers", None)
    artifacts_path.write_text(json.dumps(artifacts, indent=2) + "\n", encoding="utf-8")
    print(f"[sync-setup-zips] Updated {artifacts_path}", flush=True)

    packs_manifest_path = REPO_ROOT / "models" / "release_packs_manifest.json"
    packs_manifest = json.loads(packs_manifest_path.read_text(encoding="utf-8"))
    sha_out = (args.sha256sums_out or (out_dir / "SHA256SUMS.txt")).expanduser().resolve()
    sha_out.parent.mkdir(parents=True, exist_ok=True)
    sha_out.write_text(_format_sha256sums(artifacts, packs_manifest, release_tag=args.release_tag), encoding="utf-8")
    print(f"[sync-setup-zips] Wrote {sha_out}", flush=True)

    if not args.dry_run:
        _upload_release_files(
            [*setup_paths, sha_out],
            release_tag=args.release_tag,
            repo=args.repo,
        )
    return 0


def _download_exact_url(
    url: str,
    dest: Path,
    *,
    expected_sha256: str = "",
    expected_size: int = 0,
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and expected_sha256 and sha256_file(dest).lower() == expected_sha256.lower():
        return dest
    for attempt in range(1, 5):
        try:
            print(f"[migrate-download] {dest.name} <- {url} (attempt {attempt}/4)...", flush=True)
            req = urllib.request.Request(url, headers={"User-Agent": "SlicerDSpace-repackager"})
            with urllib.request.urlopen(req, timeout=600) as resp, dest.open("wb") as out:
                shutil.copyfileobj(resp, out, length=8 * 1024 * 1024)
            actual_size = dest.stat().st_size
            if expected_size and actual_size != expected_size:
                raise RuntimeError(f"Size mismatch for {dest.name}: {actual_size} != {expected_size}")
            actual_sha = sha256_file(dest).lower()
            if expected_sha256 and actual_sha != expected_sha256.lower():
                raise RuntimeError(f"SHA256 mismatch for {dest.name}: {actual_sha} != {expected_sha256}")
            return dest
        except Exception as exc:
            dest.unlink(missing_ok=True)
            print(f"[warn] Download retry for {dest.name}: {exc}", flush=True)
            time.sleep(4 * attempt)
    raise RuntimeError(f"Failed downloading {dest.name} from {url}")


_SURFACE_RUNTIME_REPACK_SPECS: tuple[dict[str, Any], ...] = (
    {
        "dest_name": "surface-runtime-linux-x86_64.tar.gz",
        "base_filename": "SlicerDSpace-v1.0.0-Linux-x64.tar.gz",
        "base_release_tag": "v1.0.0",
        "runtime_dirname": "surface-runtime-linux-x86_64",
        "expected_recipe_sha256": "0ca82110b74fc3898b2a96ee4b4749369a208e5fc65553335008d1c4c05284c6",
        "expected_files": 67,
        "expected_unpacked_bytes": 89406503,
    },
    {
        "dest_name": "surface-runtime-windows-x86_64.tar.gz",
        "base_filename": "SlicerDSpace-v1.0.0-Windows-x64.zip",
        "base_release_tag": "v1.0.0",
        "runtime_dirname": "surface-runtime-windows-x86_64",
        "expected_recipe_sha256": "0ca82110b74fc3898b2a96ee4b4749369a208e5fc65553335008d1c4c05284c6",
        "expected_files": 72,
        "expected_unpacked_bytes": 154982057,
    },
)


def _repack_surface_runtime_from_base(
    spec: dict[str, Any],
    *,
    work_dir: Path,
    assets_tag: str,
    repo: str,
    remote_assets: dict[str, dict[str, Any]],
    dry_run: bool = False,
) -> None:
    dest_name = str(spec["dest_name"])
    sha_name = f"{dest_name}.sha256"
    existing_tgz = remote_assets.get(dest_name)
    existing_sha = remote_assets.get(sha_name)
    if (
        isinstance(existing_tgz, dict)
        and str(existing_tgz.get("state") or "").lower() == "uploaded"
        and int(existing_tgz.get("size") or 0) > 1_000_000
        and isinstance(existing_sha, dict)
        and str(existing_sha.get("state") or "").lower() == "uploaded"
        and int(existing_sha.get("size") or 0) > 64
    ):
        print(f"[migrate-assets] Already uploaded on {assets_tag}: {dest_name} (+ .sha256)", flush=True)
        return
    base_url = f"https://github.com/{repo}/releases/download/{spec['base_release_tag']}/{spec['base_filename']}"
    if dry_run:
        print(f"[migrate-assets] [dry-run] Would extract {spec['runtime_dirname']} from {base_url} -> {dest_name}", flush=True)
        return

    base_path = work_dir / str(spec["base_filename"])
    rt_dir = work_dir / str(spec["runtime_dirname"])
    out_tgz = work_dir / dest_name
    out_sha = work_dir / sha_name
    prefix = f"SlicerDSpace/bundled-runtimes/{spec['runtime_dirname']}/"
    try:
        _download_exact_url(base_url, base_path)
        shutil.rmtree(rt_dir, ignore_errors=True)
        rt_dir.mkdir(parents=True, exist_ok=True)
        if base_path.suffix == ".zip":
            with zipfile.ZipFile(base_path, "r") as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    norm = info.filename.replace("\\", "/").lstrip("/")
                    if not norm.startswith(prefix):
                        continue
                    rel = norm[len(prefix):]
                    if not rel or ".." in PurePosixPath(rel).parts:
                        raise RuntimeError(f"Unsafe runtime member: {norm}")
                    dst = rt_dir.joinpath(*PurePosixPath(rel).parts)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info, "r") as src_fh, dst.open("wb") as dst_fh:
                        shutil.copyfileobj(src_fh, dst_fh, length=1024 * 1024)
        else:
            with tarfile.open(base_path, "r:*") as tf:
                for member in tf:
                    if not member.isfile():
                        continue
                    norm = member.name.replace("\\", "/").lstrip("/")
                    if not norm.startswith(prefix):
                        continue
                    rel = norm[len(prefix):]
                    if not rel or ".." in PurePosixPath(rel).parts:
                        raise RuntimeError(f"Unsafe runtime member: {norm}")
                    dst = rt_dir.joinpath(*PurePosixPath(rel).parts)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    with tf.extractfile(member) as src_fh, dst.open("wb") as dst_fh:
                        shutil.copyfileobj(src_fh, dst_fh, length=1024 * 1024)

        rj = rt_dir / "runtime.json"
        if not rj.is_file():
            raise RuntimeError(f"Missing runtime.json after extracting {prefix} from {base_path.name}")
        manifest = json.loads(rj.read_text(encoding="utf-8"))
        if manifest.get("activation_key_included") is not False:
            raise RuntimeError(f"Refusing surface runtime with activation_key_included != false: {dest_name}")
        if manifest.get("recipe_sha256") != spec["expected_recipe_sha256"]:
            raise RuntimeError(
                f"Surface runtime recipe mismatch for {dest_name}: {manifest.get('recipe_sha256')} != {spec['expected_recipe_sha256']}"
            )
        files_map = manifest.get("files") or {}
        if len(files_map) != int(spec["expected_files"]):
            raise RuntimeError(f"Unexpected file count in {dest_name}: {len(files_map)} != {spec['expected_files']}")
        total_bytes = 0
        for rel, meta in sorted(files_map.items()):
            fp = rt_dir.joinpath(*PurePosixPath(rel).parts)
            if not fp.is_file():
                raise RuntimeError(f"Missing declared file in {dest_name}: {rel}")
            actual_sz = fp.stat().st_size
            if actual_sz != int(meta["bytes"]):
                raise RuntimeError(f"Size mismatch for {rel} in {dest_name}: {actual_sz} != {meta['bytes']}")
            actual_sha = sha256_file(fp)
            if actual_sha != str(meta["sha256"]):
                raise RuntimeError(f"SHA256 mismatch for {rel} in {dest_name}: {actual_sha} != {meta['sha256']}")
            total_bytes += actual_sz
        if total_bytes != int(spec["expected_unpacked_bytes"]):
            raise RuntimeError(f"Unpacked size mismatch for {dest_name}: {total_bytes} != {spec['expected_unpacked_bytes']}")

        with out_tgz.open("wb") as output, gzip.GzipFile(fileobj=output, mode="wb", mtime=0, filename="") as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                for name in sorted([*files_map.keys(), "runtime.json"]):
                    fp = rt_dir.joinpath(*PurePosixPath(name).parts)
                    info = archive.gettarinfo(str(fp), arcname=name)
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ""
                    info.mode = 0o755 if (name.startswith("bin/") and not name.lower().endswith(".dll")) else 0o644
                    with fp.open("rb") as stream:
                        archive.addfile(info, stream)

        tgz_sha = sha256_file(out_tgz)
        tgz_sz = out_tgz.stat().st_size
        out_sha.write_text(f"{tgz_sha}  {dest_name}\n", encoding="utf-8")
        print(f"[surface-runtime-repacked] {dest_name} bytes={tgz_sz} sha256={tgz_sha}", flush=True)
        _upload_release_files([out_tgz, out_sha], release_tag=assets_tag, repo=repo)
    finally:
        base_path.unlink(missing_ok=True)
        out_tgz.unlink(missing_ok=True)
        out_sha.unlink(missing_ok=True)
        shutil.rmtree(rt_dir, ignore_errors=True)


def _migration_items_for_group(group: str, *, repo: str = GITHUB_REPO) -> list[tuple[str, str, str, int]]:
    """Return list of ``(dest_name, source_url, expected_sha256, expected_size)`` tuples."""
    dl = f"https://github.com/{repo}/releases/download"
    if group == "models":
        return [
            (
                "SlicerDSpace-Model-DDSurfer-v1.tar.gz",
                f"{dl}/v1.0.0/SlicerDSpace-v1.0.0-Model-DDSurfer.tar.gz",
                "b7efeea47aa736301f569622c461fd8d032d4e8734c6174b7aadcd86f58f8277",
                220436307,
            ),
            (
                "SlicerDSpace-Model-DDSeg-v1.tar.gz",
                f"{dl}/v1.0.0/SlicerDSpace-v1.0.0-Model-DDSeg.tar.gz",
                "f0146b958ac8a9ba35bd4fb7f4e2b334b7f7f3d324639428fa7d0f3b8adc19d4",
                280945320,
            ),
            (
                "SlicerDSpace-Model-DDParcel-v1.tar.gz",
                f"{dl}/v1.0.0/SlicerDSpace-v1.0.0-Model-DDParcel.tar.gz",
                "4f6e265f282c210647ed3c287fcfb7c2473b061c1802aac3c88db4c3d9ff50a6",
                391935821,
            ),
            (
                "SlicerDSpace-Model-PNL-CNN-v1.tar.gz",
                f"{dl}/v1.0.0/SlicerDSpace-v1.0.0-Model-PNL-CNN.tar.gz",
                "cf8b677106604ae10fc14f83639aa95d3709b585c81e2f39ca7fac581b3cbd65",
                200140589,
            ),
            (
                "SlicerDSpace-Model-TractCloud-v1.tar.gz",
                f"{dl}/v1.0.0/SlicerDSpace-v1.0.0-Model-TractCloud.tar.gz",
                "3f3a3a5e17c7e022736e7eb5ec20bc5c43247bc3135a858096eb9717583a3640",
                8225793,
            ),
        ]
    if group == "wma-core":
        return [
            (
                "SlicerDSpace-WMA-ORG-v1.tar.gz",
                f"{dl}/v0.1.1/SlicerDSpace-v0.1.1-WMA-ORG.tar.gz",
                "e1bb6f42218c80705746bc677ba7962468801585af31f9ce712b2164c298e6cd",
                1804334817,
            ),
            (
                "SlicerDSpace-WMA-NABA-v1.tar.gz.001",
                f"{dl}/v0.1.1/SlicerDSpace-v0.1.1-WMA-NABA.tar.gz.001",
                "a948e9d82eedcb3daec6ed7a0c5e5ec866d10d4070373aa7cbf4baeb2e3cf0c1",
                1900000000,
            ),
            (
                "SlicerDSpace-WMA-NABA-v1.tar.gz.002",
                f"{dl}/v0.1.1/SlicerDSpace-v0.1.1-WMA-NABA.tar.gz.002",
                "bdc60b267630cd737a6a9e8cf22db771147a5daba6c9e623998d0b5f4c5985c1",
                1465003883,
            ),
            (
                "SlicerDSpace-Example-HCP-100307-NIfTI.zip",
                f"{dl}/v0.1.1/SlicerDSpace-Example-HCP-100307-NIfTI.zip",
                "fada1b1f8c3f0ec52817b4db8b0cde7249490682d396f08dfab19a9693d71db0",
                1258033642,
            ),
        ]
    if group == "wma-east-west":
        return [
            (
                "SlicerDSpace-WMA-EastWest-v1.tar.gz.001",
                f"{dl}/v0.1.1/SlicerDSpace-v0.1.1-WMA-EastWest.tar.gz.001",
                "6daf10aba6b3bc530dc136c67e4da2d8b45f7b89d8cd4203de28bdf71b703d8f",
                1900000000,
            ),
            (
                "SlicerDSpace-WMA-EastWest-v1.tar.gz.002",
                f"{dl}/v0.1.1/SlicerDSpace-v0.1.1-WMA-EastWest.tar.gz.002",
                "805e74a323dac63573759c362c3de81296ac1712482858d021dff8d986b2b670",
                1900000000,
            ),
            (
                "SlicerDSpace-WMA-EastWest-v1.tar.gz.003",
                f"{dl}/v0.1.1/SlicerDSpace-v0.1.1-WMA-EastWest.tar.gz.003",
                "24c27591a46c54f387fd86045a564ee4241abf9a51d4cd8f30066a7b712e0071",
                1900000000,
            ),
            (
                "SlicerDSpace-WMA-EastWest-v1.tar.gz.004",
                f"{dl}/v0.1.1/SlicerDSpace-v0.1.1-WMA-EastWest.tar.gz.004",
                "4a060647f641c12309953bb3ba887a23ba51ef4dac9d911ac42428fdda1f3932",
                1900000000,
            ),
            (
                "SlicerDSpace-WMA-EastWest-v1.tar.gz.005",
                f"{dl}/v0.1.1/SlicerDSpace-v0.1.1-WMA-EastWest.tar.gz.005",
                "4fd4650b67bc51d49869e10363dba988fcad2e071ffce5fea6e1c3254680fb95",
                1900000000,
            ),
            (
                "SlicerDSpace-WMA-EastWest-v1.tar.gz.006",
                f"{dl}/v0.1.1/SlicerDSpace-v0.1.1-WMA-EastWest.tar.gz.006",
                "5ad69210366044801eb3be30ad26a1744c3bf658f8b1388ec3dfe0a607039a44",
                1900000000,
            ),
            (
                "SlicerDSpace-WMA-EastWest-v1.tar.gz.007",
                f"{dl}/v0.1.1/SlicerDSpace-v0.1.1-WMA-EastWest.tar.gz.007",
                "2663479e6cd9fddaa0df5360646c86e7b2a31643c78c5a0eae3af715370cb455",
                5353584,
            ),
        ]
    if group == "runtimes-and-mirror":
        items: list[tuple[str, str, str, int]] = [
            (
                "surface-runtime-darwin-arm64-macos13.tar.gz",
                f"{dl}/v0.1.1/surface-runtime-darwin-arm64-macos13.tar.gz",
                "cfd1d7b85c595479b5378fc91dde6ab0a781fc2b6c6e2a04de9feabc414faff9",
                39339026,
            ),
            (
                "surface-runtime-darwin-arm64-macos13.tar.gz.sha256",
                f"{dl}/v0.1.1/surface-runtime-darwin-arm64-macos13.tar.gz.sha256",
                "5d26d27dfdc9c28955d4e825eaaac1895ba5856eb47f14e63f7e8bc95c0fd008",
                110,
            ),
            (
                "ukf-standalone-darwin-arm64.tar.gz",
                f"{dl}/model-assets-v0.1.1/ukf-standalone-darwin-arm64.tar.gz",
                "e68e8190f8d3e5ae70e5dbc68d55d99f6c6551abbebc3dd1a1696c43ada51f8f",
                3699789,
            ),
            (
                "ukf-standalone-darwin-arm64.tar.gz.sha256",
                f"{dl}/model-assets-v0.1.1/ukf-standalone-darwin-arm64.tar.gz.sha256",
                "f6d4240f47101c4ee92b484a8011c13d00c2c6f746daf412ce9c52bdd5695a50",
                101,
            ),
            (
                "ukf-standalone-linux-x64.tar.gz",
                f"{dl}/model-assets-v0.1.1/ukf-standalone-linux-x64.tar.gz",
                "8631887f22ed6d79bf692f29280a824d9276dec3eb41702241f93822401dcffd",
                6031472,
            ),
            (
                "ukf-standalone-linux-x64.tar.gz.sha256",
                f"{dl}/model-assets-v0.1.1/ukf-standalone-linux-x64.tar.gz.sha256",
                "4576d7d6d4073ce5d981cd80550b276afe74569d81c9d083415c5dd1c97caf58",
                98,
            ),
            (
                "ukf-standalone-windows-x64.zip",
                f"{dl}/model-assets-v0.1.1/ukf-standalone-windows-x64.zip",
                "a9359316af30163f4a2abd19a282d0595e311d0c2c0122e2c29cc70bf23ce47c",
                2365153,
            ),
            (
                "ukf-standalone-windows-x64.zip.sha256",
                f"{dl}/model-assets-v0.1.1/ukf-standalone-windows-x64.zip.sha256",
                "54f0753a2f7b5c50badb117a34e46a21df2e06b55b8385f32277cb6ef30d10cf",
                98,
            ),
        ]
        # Mirror all active individual files from assets_manifest.json + external_assets_manifest.json
        # (excluding upstream-only NITRC IIT and large split WMA zip parts that are superseded by WMA-*-v1.tar.gz).
        seen_names: set[str] = {name for name, _, _, _ in items}
        for mname in ("assets_manifest.json", "external_assets_manifest.json"):
            mdata = json.loads((REPO_ROOT / "models" / mname).read_text(encoding="utf-8"))
            for rel_path, rec in (mdata.get("asset_records") or {}).items():
                if not isinstance(rec, dict):
                    continue
                if rec.get("redistribution") == "upstream-only":
                    continue
                if rel_path.startswith("bundles/wma/") and ("archives/" in rel_path or "ATTRIBUTION.md" in rel_path or "naba/" in rel_path or "east-west/" in rel_path):
                    continue
                if rel_path.startswith("brain_mask/pnl_cnn/"):
                    continue
                sha = str(rec.get("sha256") or "").strip().lower()
                sz = int(rec.get("size_bytes") or 0)
                if len(sha) != 64:
                    continue
                fname = PurePosixPath(rel_path).name
                dest_name = f"{sha}-{fname}"
                if dest_name in seen_names:
                    continue
                seen_names.add(dest_name)
                urls = rec.get("urls") or []
                src_url = str(urls[0]) if urls else f"{dl}/model-assets-v0.1.1/{dest_name}"
                items.append((dest_name, src_url, sha, sz))
        return items
    raise ValueError(f"Unknown migrate-assets group: {group}")


def cmd_migrate_assets(args: argparse.Namespace) -> int:
    assets_tag = str(args.assets_tag or "assets").strip()
    repo = str(args.repo or GITHUB_REPO)
    _assert_mutable_release_tag(assets_tag, dry_run=bool(args.dry_run))
    work_dir = args.work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    items = _migration_items_for_group(args.group, repo=repo)
    remote_assets = {} if args.dry_run else _fetch_remote_release_assets(assets_tag, repo)
    print(f"[migrate-assets] Group={args.group} ({len(items)} item(s)) -> release {assets_tag}", flush=True)

    if args.group == "runtimes-and-mirror":
        for spec in _SURFACE_RUNTIME_REPACK_SPECS:
            _repack_surface_runtime_from_base(
                spec,
                work_dir=work_dir,
                assets_tag=assets_tag,
                repo=repo,
                remote_assets=remote_assets,
                dry_run=bool(args.dry_run),
            )

    for dest_name, src_url, expected_sha, expected_size in items:
        existing = remote_assets.get(dest_name)
        if (
            isinstance(existing, dict)
            and str(existing.get("state") or "").lower() == "uploaded"
            and int(existing.get("size") or -1) == expected_size
            and str(existing.get("digest") or "").strip().lower() == f"sha256:{expected_sha.lower()}"
        ):
            print(f"[migrate-assets] Already verified on {assets_tag}: {dest_name}", flush=True)
            continue
        if args.dry_run:
            print(f"[migrate-assets] [dry-run] Would copy {dest_name} ({expected_size:,} B) from {src_url}", flush=True)
            continue
        local_path = work_dir / dest_name
        try:
            _download_exact_url(src_url, local_path, expected_sha256=expected_sha, expected_size=expected_size)
            _upload_release_files([local_path], release_tag=assets_tag, repo=repo)
        finally:
            local_path.unlink(missing_ok=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_model = sub.add_parser(
        "build-model-pack",
        help="Download/verify one modular core model weight pack and upload tar.gz to release.",
    )
    p_model.add_argument("--pack", required=True, choices=CORE_MODEL_PACK_ORDER)
    p_model.add_argument("--work-dir", required=True, type=Path)
    p_model.add_argument("--out-dir", required=True, type=Path)
    p_model.add_argument("--asset-root", default=None, type=Path, help="Optional pre-populated local verified asset root.")
    p_model.add_argument("--release-tag", default=ASSETS_RELEASE_TAG)
    p_model.add_argument("--assets-tag", default="")
    p_model.add_argument("--repo", default=GITHUB_REPO)
    p_model.add_argument("--dry-run", action="store_true")

    p_apply_models = sub.add_parser(
        "apply-model-receipts",
        help="Merge model-pack-receipt-*.json files into models/release_packs_manifest.json before OS base repacking.",
    )
    p_apply_models.add_argument("--receipts-dir", required=True, type=Path)
    p_apply_models.add_argument("--release-tag", default=RELEASE_TAG)
    p_apply_models.add_argument("--repo", default=GITHUB_REPO)

    p_atlas = sub.add_parser("build-atlas", help="Download/verify one WMA atlas and upload tar.gz/split volumes.")
    p_atlas.add_argument("--atlas", required=True, choices=tuple(WMA_ATLAS_PACK_SPECS.keys()))
    p_atlas.add_argument("--work-dir", required=True, type=Path)
    p_atlas.add_argument("--out-dir", required=True, type=Path)
    p_atlas.add_argument("--dry-run", action="store_true")

    p_base = sub.add_parser("repack-os-base", help="Repack slimmed OS Base archive without bundled model weights and with fresh HEAD code.")
    p_base.add_argument("--target", required=True, choices=tuple(SOURCE_BUNDLES.keys()))
    p_base.add_argument("--work-dir", required=True, type=Path)
    p_base.add_argument("--out-dir", required=True, type=Path)
    p_base.add_argument("--release-tag", default=RELEASE_TAG)
    p_base.add_argument("--source-release-tag", default="", help="Optional source release tag (e.g. v0.1.1 when bootstrapping v1.0.0).")
    p_base.add_argument("--source-archive", default=None, type=Path, help="Optional local source base archive path.")
    p_base.add_argument("--repo", default=GITHUB_REPO)
    p_base.add_argument("--keep-bundled-model-assets", action="store_true", help="Keep bundled-model-assets inside OS Base archive (legacy monolithic layout).")
    p_base.add_argument("--macos-ukf-archive", default=None, type=Path, help="Optional local ukf-standalone-darwin-arm64.tar.gz path.")
    p_base.add_argument("--standalone-ukf-archive", default=None, type=Path, help="Optional local standalone UKF archive path for target OS.")
    p_base.add_argument("--skip-macos-ukf", action="store_true", help="Do not replace macOS UKF binary during repack.")
    p_base.add_argument("--skip-standalone-ukf", action="store_true", help="Do not inject standalone UKF binary during repack.")
    p_base.add_argument("--dry-run", action="store_true")

    p_sync = sub.add_parser(
        "sync-metadata",
        help="Sync model-pack & os-base receipt checksums into release_packs_manifest.json, installer_artifacts.json, website JS, setup zips, and SHA256SUMS.txt.",
    )
    p_sync.add_argument("--receipts-dir", required=True, type=Path)
    p_sync.add_argument("--setup-out-dir", default=None, type=Path)
    p_sync.add_argument("--sha256sums-out", default=None, type=Path)
    p_sync.add_argument("--release-tag", default=RELEASE_TAG)
    p_sync.add_argument("--repo", default=GITHUB_REPO)
    p_sync.add_argument("--upload-sha256sums", action="store_true")
    p_sync.add_argument("--dry-run", action="store_true")

    p_setup = sub.add_parser(
        "sync-setup-zips",
        help="Build and upload universal SlicerDSpaceSetup-<OS>.zip packages plus updated SHA256SUMS.txt.",
    )
    p_setup.add_argument("--out-dir", required=True, type=Path)
    p_setup.add_argument("--sha256sums-out", default=None, type=Path)
    p_setup.add_argument("--release-tag", default=RELEASE_TAG)
    p_setup.add_argument("--repo", default=GITHUB_REPO)
    p_setup.add_argument("--dry-run", action="store_true")

    p_overlay = sub.add_parser(
        "sync-code-overlay",
        help="Build and upload lightweight SlicerDSpace-<tag>-CodeOverlay.tar.gz and update_manifest.json.",
    )
    p_overlay.add_argument("--out-dir", required=True, type=Path)
    p_overlay.add_argument("--sha256sums-out", default=None, type=Path)
    p_overlay.add_argument("--release-tag", default=RELEASE_TAG)
    p_overlay.add_argument("--repo", default=GITHUB_REPO)
    p_overlay.add_argument("--dry-run", action="store_true")

    p_migrate = sub.add_parser(
        "migrate-assets",
        help="Copy and SHA-256 verify shared model packs, WMA atlases, example data and runtimes into the permanent assets release.",
    )
    p_migrate.add_argument("--group", required=True, choices=("models", "wma-core", "wma-east-west", "runtimes-and-mirror"))
    p_migrate.add_argument("--work-dir", required=True, type=Path)
    p_migrate.add_argument("--assets-tag", default="assets")
    p_migrate.add_argument("--repo", default=GITHUB_REPO)
    p_migrate.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "migrate-assets":
        return cmd_migrate_assets(args)
    if args.command == "build-model-pack":
        return cmd_build_model_pack(args)
    if args.command == "apply-model-receipts":
        return cmd_apply_model_receipts(args)
    if args.command == "build-atlas":
        return cmd_build_atlas(args)
    if args.command == "sync-metadata":
        return cmd_sync_metadata(args)
    if args.command == "sync-setup-zips":
        return cmd_sync_setup_zips(args)
    if args.command == "sync-code-overlay":
        return cmd_sync_code_overlay(args)
    return cmd_repack_os_base(args)


if __name__ == "__main__":
    raise SystemExit(main())
