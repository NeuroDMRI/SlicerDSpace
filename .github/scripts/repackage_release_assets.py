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
    DETERMINISTIC_MTIME,
    RELEASE_BASE_URL,
    RELEASE_TAG,
    WMA_ATLAS_PACK_SPECS,
    collect_wma_pack_files,
    manifest_pack_entry,
    sha256_file,
    split_file_if_needed,
    write_deterministic_tar_gz,
)
from scripts.package_slicer_extension import _should_mark_executable
from shared.code_updater import EXTENSION_VERSION, RUNTIME_ABI_VERSION, compute_code_tree_sha256

GITHUB_REPO = "NeuroDMRI/SlicerDSpace"
MODEL_ASSETS_TAG = "model-assets-v0.1.1"
MACOS_UKF_ARCHIVE = "ukf-standalone-darwin-arm64.tar.gz"

SOURCE_BUNDLES: dict[str, dict[str, str]] = {
    "windows-x64": {
        "bundle_key": "windows_x64_base",
        "os": "windows",
        "arch": "x86_64",
        "out_name": f"SlicerDSpace-{RELEASE_TAG}-Windows-x64.zip",
        "ext_prefix": "SlicerDSpace",
        "slicer_subdir": "Slicer-5.10",
    },
    "linux-x64": {
        "bundle_key": "linux_x64_base",
        "os": "linux",
        "arch": "x86_64",
        "out_name": f"SlicerDSpace-{RELEASE_TAG}-Linux-x64.tar.gz",
        "ext_prefix": "SlicerDSpace",
        "slicer_subdir": "Slicer-5.10",
    },
    "macos-arm64": {
        "bundle_key": "macos_arm64_base",
        "os": "macos",
        "arch": "arm64",
        "out_name": f"SlicerDSpace-{RELEASE_TAG}-macOS-arm64.tar.gz",
        "ext_prefix": "SlicerDSpace/Slicer.app/Contents/Extensions-34045/SlicerDSpace",
        "slicer_subdir": "Slicer-5.10",
    },
}


def _upload_release_files(
    files: list[Path],
    *,
    release_tag: str = RELEASE_TAG,
    repo: str = GITHUB_REPO,
) -> None:
    cmd = [
        "gh",
        "release",
        "upload",
        release_tag,
        "--repo",
        repo,
        "--clobber",
        *[str(f) for f in files],
    ]
    for attempt in range(1, 4):
        proc = subprocess.run(cmd)
        if proc.returncode == 0:
            return
        print(f"[warn] gh release upload failed (attempt {attempt}/3), retrying...", flush=True)
        time.sleep(5 * attempt)
    raise RuntimeError(f"Failed to upload release assets after 3 attempts: {files}")


def _download_verified_part(filename: str, expected_size: int, expected_sha256: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and sha256_file(dest) == expected_sha256:
        return dest
    url = f"{RELEASE_BASE_URL}/{filename}"
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


def _should_skip_member(
    norm_path: str,
    ext_prefix: str,
    qt_root: str,
    *,
    replace_macos_ukf: bool = False,
    slicer_subdir: str = "Slicer-5.10",
) -> bool:
    wma_root = f"{ext_prefix}/bundled-model-assets/bundles/wma"
    if norm_path == wma_root or norm_path.startswith(wma_root + "/"):
        return True
    if norm_path.endswith("/IITmean_b0_256.nii.gz"):
        return True
    if norm_path == qt_root or norm_path.startswith(qt_root + "/"):
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

    parts, upload_paths = split_file_if_needed(archive_path, remove_unsplit_when_split=True)
    if not args.dry_run:
        _upload_release_files(upload_paths)
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
    spec = SOURCE_BUNDLES[target]
    work_dir = args.work_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts_path = REPO_ROOT / "docs" / "website" / "installer_artifacts.json"
    artifacts = json.loads(artifacts_path.read_text(encoding="utf-8"))
    bundle_spec = artifacts["bundles"][spec["bundle_key"]]
    source_filename = str(bundle_spec["filename"])
    source_sha256 = str(bundle_spec["sha256"])
    source_size = int(bundle_spec["size_bytes"])
    source_parts = list(bundle_spec.get("parts") or ())

    parts_dir = work_dir / f"parts-{target}"
    old_archive = work_dir / f"{source_filename}.source"
    if not (old_archive.is_file() and sha256_file(old_archive) == source_sha256):
        if source_parts:
            part_paths = []
            for pspec in source_parts:
                pname = str(pspec["filename"])
                psize = int(pspec["size_bytes"])
                psha = str(pspec["sha256"])
                part_paths.append(_download_verified_part(pname, psize, psha, parts_dir / pname))
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
            _download_verified_part(source_filename, source_size, source_sha256, old_archive)

    ext_prefix = spec["ext_prefix"]
    slicer_subdir = spec["slicer_subdir"]
    qt_root = f"{ext_prefix}/lib/{slicer_subdir}/qt-scripted-modules"
    fresh_runtime_files = sorted(runtime_files(REPO_ROOT))
    replace_macos_ukf = (target == "macos-arm64") and (not getattr(args, "skip_macos_ukf", False))
    macos_ukf_tar: Path | None = None
    if replace_macos_ukf:
        explicit_ukf = getattr(args, "macos_ukf_archive", None)
        if explicit_ukf is not None:
            macos_ukf_tar = Path(explicit_ukf).expanduser().resolve()
            if not macos_ukf_tar.is_file():
                raise FileNotFoundError(f"--macos-ukf-archive not found: {macos_ukf_tar}")
        else:
            macos_ukf_tar = _download_model_asset(MACOS_UKF_ARCHIVE, work_dir / MACOS_UKF_ARCHIVE)
    new_archive = out_dir / spec["out_name"]
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
                    replace_macos_ukf=replace_macos_ukf,
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
                    replace_macos_ukf=replace_macos_ukf,
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

            if macos_ukf_tar is not None:
                rt_prefix = f"{ext_prefix}/bundled-runtimes"
                _ensure_tar_dir(rt_prefix)
                with tarfile.open(macos_ukf_tar, "r:*") as ukf_tf:
                    for um in ukf_tf:
                        unorm = um.name.replace("\\", "/").strip("/")
                        if not unorm or ".." in PurePosixPath(unorm).parts:
                            continue
                        if not (unorm == "ukf-darwin-arm64" or unorm.startswith("ukf-darwin-arm64/")):
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

    if not saw_qt_entry:
        tmp_new.unlink(missing_ok=True)
        raise RuntimeError(f"Expected module root {qt_root}/SlicerDSpace.py was not found in source archive")

    os.replace(tmp_new, new_archive)
    old_archive.unlink(missing_ok=True)

    audit = audit_package(new_archive, REPO_ROOT)
    layout = inspect_archive_layout(new_archive)
    if audit["status"] != "pass":
        raise RuntimeError(f"audit_package failed for {new_archive.name}: {json.dumps(audit, indent=2)}")
    if layout["status"] != "pass":
        raise RuntimeError(f"inspect_archive_layout failed for {new_archive.name}: {json.dumps(layout, indent=2)}")
    if audit["bundled_wma_atlases"] != [] or layout["bundled_wma_atlases"] != []:
        raise RuntimeError(f"WMA atlases leaked into {new_archive.name}")
    if audit["bundled_python_file_count"] <= 0 or audit["bundled_model_asset_file_count"] <= 0:
        raise RuntimeError(f"Missing bundled Python/model assets in {new_archive.name}")

    size_bytes = new_archive.stat().st_size
    digest = sha256_file(new_archive)
    parts, upload_paths = split_file_if_needed(new_archive, remove_unsplit_when_split=True)
    if not args.dry_run:
        _upload_release_files(upload_paths)
    for p in upload_paths:
        p.unlink(missing_ok=True)

    receipt: dict[str, Any] = {
        "os": spec["os"],
        "arch": spec["arch"],
        "mode": "base",
        "filename": spec["out_name"],
        "url": "" if parts else f"{RELEASE_BASE_URL}/{spec['out_name']}",
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


def _sync_customizer_js_bundles(bundles: dict[str, dict[str, Any]]) -> Path:
    js_path = REPO_ROOT / "docs" / "site" / "assets" / "js" / "installer-customizer.js"
    js_src = js_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"^[ \t]*// __ARTIFACTS_BUNDLES_BEGIN__.*?^[ \t]*// __ARTIFACTS_BUNDLES_END__",
        flags=re.MULTILINE | re.DOTALL,
    )
    if not pattern.search(js_src):
        raise RuntimeError(f"Missing __ARTIFACTS_BUNDLES_BEGIN__/__ARTIFACTS_BUNDLES_END__ markers in {js_path}")
    new_js = pattern.sub(lambda _m: _render_js_bundles_block(bundles), js_src, count=1)

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
) -> None:
    if not all(ord(ch) < 128 or ch == "·" for ch in py_src):
        raise RuntimeError(f"Non-ASCII wording detected in {zip_path}")
    cfg = _extract_embedded_setup_config(py_src)
    if cfg.get("auto_start") is not False:
        raise RuntimeError(f"Expected auto_start=False inside {zip_path}, got {cfg.get('auto_start')!r}")
    if cfg.get("os") != os_key or cfg.get("selected_packs") != ["wma_org"]:
        raise RuntimeError(f"Unexpected OS/pack defaults inside {zip_path}")
    expected_bundle_sha = str((bundles.get(bundle_key) or {}).get("sha256") or "")
    actual_bundle_sha = str((((cfg.get("artifacts") or {}).get("bundles") or {}).get(bundle_key) or {}).get("sha256") or "")
    if expected_bundle_sha and actual_bundle_sha != expected_bundle_sha:
        raise RuntimeError(
            f"Embedded {bundle_key} sha256 mismatch in {zip_path}: {actual_bundle_sha} != {expected_bundle_sha}"
        )


def _verify_generated_setup_zip(
    zip_path: Path,
    os_key: str,
    bundle_key: str,
    expected_s4ext_os: str,
    bundles: dict[str, dict[str, Any]],
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
                    _verify_setup_py(zf.read(py_rel).decode("utf-8"), zip_path, os_key, bundle_key, bundles)
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
                    _verify_setup_py(zf.read(py_rel).decode("utf-8"), zip_path, os_key, bundle_key, bundles)
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
        "      wmaEnabled: true,\n"
        "      wmaAtlases: ['wma_org'],\n"
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
    setup_meta: dict[str, dict[str, Any]] = {}
    written_paths: list[Path] = []

    for os_key, filename, bundle_key, expected_s4ext_os in SETUP_OS_ORDER:
        zip_path = out_dir / filename
        _verify_generated_setup_zip(zip_path, os_key, bundle_key, expected_s4ext_os, bundles)
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


def _format_sha256sums(artifacts: dict[str, Any], packs_manifest: dict[str, Any]) -> str:
    entries: list[tuple[str, str]] = []
    merged_entries: list[tuple[str, str]] = []
    bundles = artifacts.get("bundles") or {}
    for key in [k for k in _BUNDLE_ORDER if k in bundles] + [k for k in bundles if k not in _BUNDLE_ORDER]:
        b = bundles[key]
        parts = list(b.get("parts") or ())
        if parts:
            for p in parts:
                entries.append((str(p["sha256"]), str(p["filename"])))
            if b.get("sha256") and b.get("filename"):
                merged_entries.append((str(b["sha256"]), str(b["filename"])))
        elif b.get("sha256"):
            entries.append((str(b["sha256"]), str(b["filename"])))
    code_overlay = artifacts.get("code_overlay") or {}
    if code_overlay.get("sha256") and code_overlay.get("filename"):
        entries.append((str(code_overlay["sha256"]), str(code_overlay["filename"])))
    setup_installers = artifacts.get("setup_installers") or {}
    for os_key, _, _, _ in SETUP_OS_ORDER:
        rec = setup_installers.get(os_key)
        if rec and rec.get("sha256") and rec.get("filename"):
            entries.append((str(rec["sha256"]), str(rec["filename"])))
    packs = packs_manifest.get("packs") or {}
    pack_order = [k for k in ("wma_org", "wma_naba", "wma_east_west") if k in packs] + [
        k for k in packs if k not in ("wma_org", "wma_naba", "wma_east_west")
    ]
    for key in pack_order:
        p = packs[key]
        parts = list(p.get("parts") or ())
        if parts:
            for sp in parts:
                entries.append((str(sp["sha256"]), str(sp["filename"])))
            if p.get("sha256") and p.get("filename"):
                merged_entries.append((str(p["sha256"]), str(p["filename"])))
        elif p.get("sha256"):
            entries.append((str(p["sha256"]), str(p["filename"])))
    out = "".join(f"{sha}  {name}\n" for sha, name in entries if sha and name)
    if merged_entries:
        out += "\n# Merged multi-part archive SHA-256 (after concatenating .001..N parts):\n"
        out += "".join(f"# {sha}  {name}\n" for sha, name in merged_entries)
    return out


def cmd_sync_metadata(args: argparse.Namespace) -> int:
    receipts_dir = args.receipts_dir.expanduser().resolve()
    receipt_files = sorted(receipts_dir.rglob("os-base-receipt-*.json"))
    if not receipt_files:
        raise RuntimeError(f"No os-base-receipt-*.json files found under {receipts_dir}")

    artifacts_path = REPO_ROOT / "docs" / "website" / "installer_artifacts.json"
    artifacts = json.loads(artifacts_path.read_text(encoding="utf-8"))
    bundles = artifacts.setdefault("bundles", {})
    allowed_keys = ("os", "arch", "mode", "filename", "url", "size_bytes", "sha256", "published", "parts")
    for rf in receipt_files:
        data = json.loads(rf.read_text(encoding="utf-8"))
        for key, rec in data.items():
            rec_copy = {k: rec[k] for k in allowed_keys if k in rec}
            bundles[key] = rec_copy

    _sync_customizer_js_bundles(bundles)

    setup_dir = (args.setup_out_dir or (receipts_dir / "setup-zips")).expanduser().resolve()
    setup_meta, setup_paths = build_setup_zips(
        setup_dir,
        artifacts,
        release_tag=args.release_tag,
        published=not args.dry_run,
    )
    artifacts["setup_installers"] = setup_meta
    artifacts.pop("cross_version_installers", None)
    artifacts.pop("universal_installers", None)

    overlay_manifest, overlay_paths = build_code_overlay(
        setup_dir / "overlay",
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
    print(f"[sync-metadata] Updated {artifacts_path}", flush=True)

    packs_manifest_path = REPO_ROOT / "models" / "release_packs_manifest.json"
    packs_manifest = json.loads(packs_manifest_path.read_text(encoding="utf-8"))
    sha_out = (args.sha256sums_out or (receipts_dir / "SHA256SUMS.txt")).expanduser().resolve()
    sha_out.parent.mkdir(parents=True, exist_ok=True)
    sha_out.write_text(_format_sha256sums(artifacts, packs_manifest), encoding="utf-8")
    print(f"[sync-metadata] Wrote {sha_out}", flush=True)

    if args.upload_sha256sums and not args.dry_run:
        _upload_release_files(
            [*overlay_paths, *setup_paths, sha_out],
            release_tag=args.release_tag,
            repo=args.repo,
        )
    return 0


def cmd_sync_code_overlay(args: argparse.Namespace) -> int:
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
    sha_out.write_text(_format_sha256sums(artifacts, packs_manifest), encoding="utf-8")
    print(f"[sync-code-overlay] Wrote {sha_out}", flush=True)

    if not args.dry_run:
        _upload_release_files(
            [*overlay_paths, sha_out],
            release_tag=args.release_tag,
            repo=args.repo,
        )
    return 0


def cmd_sync_setup_zips(args: argparse.Namespace) -> int:
    artifacts_path = REPO_ROOT / "docs" / "website" / "installer_artifacts.json"
    artifacts = json.loads(artifacts_path.read_text(encoding="utf-8"))
    bundles = artifacts.get("bundles") or {}
    _sync_customizer_js_bundles(bundles)

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
    sha_out.write_text(_format_sha256sums(artifacts, packs_manifest), encoding="utf-8")
    print(f"[sync-setup-zips] Wrote {sha_out}", flush=True)

    if not args.dry_run:
        _upload_release_files(
            [*setup_paths, sha_out],
            release_tag=args.release_tag,
            repo=args.repo,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_atlas = sub.add_parser("build-atlas", help="Download/verify one WMA atlas and upload tar.gz/split volumes.")
    p_atlas.add_argument("--atlas", required=True, choices=tuple(WMA_ATLAS_PACK_SPECS.keys()))
    p_atlas.add_argument("--work-dir", required=True, type=Path)
    p_atlas.add_argument("--out-dir", required=True, type=Path)
    p_atlas.add_argument("--dry-run", action="store_true")

    p_base = sub.add_parser("repack-os-base", help="Repack v0.1.1 archive without WMA/IIT and with fresh HEAD code.")
    p_base.add_argument("--target", required=True, choices=tuple(SOURCE_BUNDLES.keys()))
    p_base.add_argument("--work-dir", required=True, type=Path)
    p_base.add_argument("--out-dir", required=True, type=Path)
    p_base.add_argument("--macos-ukf-archive", default=None, type=Path, help="Optional local ukf-standalone-darwin-arm64.tar.gz path.")
    p_base.add_argument("--skip-macos-ukf", action="store_true", help="Do not replace macOS UKF binary during repack.")
    p_base.add_argument("--dry-run", action="store_true")

    p_sync = sub.add_parser(
        "sync-metadata",
        help="Sync os-base-receipt-*.json checksums into installer_artifacts.json, website JS, setup zips, and SHA256SUMS.txt.",
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

    args = parser.parse_args(argv)
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
