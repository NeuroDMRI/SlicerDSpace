#!/usr/bin/env python3
"""GitHub Actions helper to build standalone WMA atlas packs and compact OS Base packages."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
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

GITHUB_REPO = "NeuroDMRI/SlicerDSpace"

SOURCE_BUNDLES: dict[str, dict[str, Any]] = {
    "windows-x64": {
        "bundle_key": "windows_x64_base",
        "os": "windows",
        "arch": "x86_64",
        "out_name": f"SlicerDSpace-{RELEASE_TAG}-Windows-x64.zip",
        "ext_prefix": "SlicerDSpace",
        "slicer_subdir": "Slicer-5.10",
        "source_filename": f"SlicerDSpace-{RELEASE_TAG}-Windows-x64.zip",
        "source_size_bytes": 18124374419,
        "source_sha256": "b38d612ad000c6a86b78cc1db10c58c7422c3cd1ca506a65285d7d7598f2feae",
        "parts": [
            ("SlicerDSpace-v0.1.1-Windows-x64.zip.001", 1900000000, "816a9e730f75d7673dc2f1f4f2b9fc6caf13867939e8ab93feb94cd57838f7ee"),
            ("SlicerDSpace-v0.1.1-Windows-x64.zip.002", 1900000000, "514f8cfe930ef54b3e7d63171eb2ffbf0ce4efc8ef0f888d5e2abe44685931e4"),
            ("SlicerDSpace-v0.1.1-Windows-x64.zip.003", 1900000000, "ccf65e829384e13ba1925bff1f87860030f406ed96b640d024ed22dc742df337"),
            ("SlicerDSpace-v0.1.1-Windows-x64.zip.004", 1900000000, "20300c3bef02e9f4a79fafb76c8f4d46994a18d5ee9ce5745be8db2e90ed3a13"),
            ("SlicerDSpace-v0.1.1-Windows-x64.zip.005", 1900000000, "b14d457459979741583ba7139317394333dc2fb0cfc1c4c5e921d73a7da25d7b"),
            ("SlicerDSpace-v0.1.1-Windows-x64.zip.006", 1900000000, "4021369d1e9433c02e9eb51efe7d417721781d6545672b059511d51e99e34600"),
            ("SlicerDSpace-v0.1.1-Windows-x64.zip.007", 1900000000, "ab317ff156cc9cd444f1c5f88e91ee886dfbef2f3b172973ff498ed93c2e5692"),
            ("SlicerDSpace-v0.1.1-Windows-x64.zip.008", 1900000000, "9002830250981f09aecd9936142f6924e5dfb265daa82bc360ead30f297a8368"),
            ("SlicerDSpace-v0.1.1-Windows-x64.zip.009", 1900000000, "5a7e8c8ce83c96b89f247a375c24fa0e0d23a10ee96c3da7c194cb31a1cd037c"),
            ("SlicerDSpace-v0.1.1-Windows-x64.zip.010", 1024374419, "97ef58d4da44ab62d28d597c17b8f3e7c0173df6f835852905206aca1050a7e2"),
        ],
    },
    "linux-x64": {
        "bundle_key": "linux_x64_base",
        "os": "linux",
        "arch": "x86_64",
        "out_name": f"SlicerDSpace-{RELEASE_TAG}-Linux-x64.tar.gz",
        "ext_prefix": "SlicerDSpace",
        "slicer_subdir": "Slicer-5.10",
        "source_filename": f"SlicerDSpace-{RELEASE_TAG}-Linux-x64.tar.gz",
        "source_size_bytes": 14416513720,
        "source_sha256": "86cc9cb5ef34dbede12c9abdbe110b972ff0fc873d9cef34df3bc8cbad36d403",
        "parts": [
            ("SlicerDSpace-v0.1.1-Linux-x64.tar.gz.001", 1900000000, "01b7fd7c39941e8c3757042fbdce0a14e3ced752cd0672cbc847f7fbeb04f4f8"),
            ("SlicerDSpace-v0.1.1-Linux-x64.tar.gz.002", 1900000000, "476aefe2f555f629241f62b10636c8431272581ea2fb91c2b3606a036c5e3ee8"),
            ("SlicerDSpace-v0.1.1-Linux-x64.tar.gz.003", 1900000000, "0b0b8a9bd3a955b44cf9b3016f5e84dd5c3a2fdf45464fcbe9e872379b701639"),
            ("SlicerDSpace-v0.1.1-Linux-x64.tar.gz.004", 1900000000, "358c0b66de302e7b64fb63d3ada4a22bf690374852d8874e49552d813e562515"),
            ("SlicerDSpace-v0.1.1-Linux-x64.tar.gz.005", 1900000000, "3284c0b811ccf10e5c907fc152e1e49a98640a1a13be693fb3d94b7d35d5b436"),
            ("SlicerDSpace-v0.1.1-Linux-x64.tar.gz.006", 1900000000, "5a65b97b8f16bc88fef9826fc1e7f68804545d18ddf966a0bc5126e9a2ab9a50"),
            ("SlicerDSpace-v0.1.1-Linux-x64.tar.gz.007", 1900000000, "c27bca4aafb1ccd4e5b6c50b78a374e5d4c49483d33bef03c401792678c45dd8"),
            ("SlicerDSpace-v0.1.1-Linux-x64.tar.gz.008", 1116513720, "d6488d61641cc5ea6c85db778b00266851378e77b012990ad28ddc095effe5c0"),
        ],
    },
    "macos-arm64": {
        "bundle_key": "macos_arm64_base",
        "os": "macos",
        "arch": "arm64",
        "out_name": f"SlicerDSpace-{RELEASE_TAG}-macOS-arm64.tar.gz",
        "ext_prefix": "SlicerDSpace/Slicer.app/Contents/Extensions-34045/SlicerDSpace",
        "slicer_subdir": "Slicer-5.10",
        "source_filename": f"SlicerDSpace-{RELEASE_TAG}-macOS-arm64.tar.gz",
        "source_size_bytes": 14326384768,
        "source_sha256": "ca27a69813c9f988ad13dad0d01244ca907cfdfc981db5a445bf4f1ec1aa5d9a",
        "parts": [
            ("SlicerDSpace-v0.1.1-macOS-arm64.tar.gz.001", 1900000000, "719e3e6150a70291f44b20386c02f023e04edb9893160d7b2ff4935862fdb620"),
            ("SlicerDSpace-v0.1.1-macOS-arm64.tar.gz.002", 1900000000, "51d69f43c3ed140d56ab7516d3791bc213b4a80d037045bdec107d400748b4c7"),
            ("SlicerDSpace-v0.1.1-macOS-arm64.tar.gz.003", 1900000000, "2b3a00198601c9c6d7b6088388e508f788a4bf968a13ecf9f7bb969bc54e6cbc"),
            ("SlicerDSpace-v0.1.1-macOS-arm64.tar.gz.004", 1900000000, "1aeed976ad2a100c5c98f35087c2ff14e3f1ee8851721187b89935c73e574079"),
            ("SlicerDSpace-v0.1.1-macOS-arm64.tar.gz.005", 1900000000, "1d31624e6637ba2b6a2b8f6fcd9d7477c98c55c28320921cbc3a8e1d3d4044a9"),
            ("SlicerDSpace-v0.1.1-macOS-arm64.tar.gz.006", 1900000000, "698e7c963a6fbd5004f14a5e30ceebd9deb665a4c2d702bd37ceaf45c643eec7"),
            ("SlicerDSpace-v0.1.1-macOS-arm64.tar.gz.007", 1900000000, "82a779a2d9c86281bf1fb51207d0da6a2d0c630682bb06164491f63f67a938de"),
            ("SlicerDSpace-v0.1.1-macOS-arm64.tar.gz.008", 1026384768, "e78c500f2eab4102ab9d45d4d7f1a7506cd6cc3ed50841f0bab639a9c20f691b"),
        ],
    },
}


def _upload_release_files(files: list[Path]) -> None:
    cmd = [
        "gh",
        "release",
        "upload",
        RELEASE_TAG,
        "--repo",
        GITHUB_REPO,
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


def _should_skip_member(norm_path: str, ext_prefix: str, qt_root: str) -> bool:
    wma_root = f"{ext_prefix}/bundled-model-assets/bundles/wma"
    if norm_path == wma_root or norm_path.startswith(wma_root + "/"):
        return True
    if norm_path.endswith("/IITmean_b0_256.nii.gz"):
        return True
    if norm_path == qt_root or norm_path.startswith(qt_root + "/"):
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

    parts_dir = work_dir / f"parts-{target}"
    old_archive = work_dir / spec["source_filename"]
    if not (old_archive.is_file() and sha256_file(old_archive) == spec["source_sha256"]):
        part_paths = []
        for pname, psize, psha in spec["parts"]:
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
        if h.hexdigest() != spec["source_sha256"]:
            merging.unlink(missing_ok=True)
            raise RuntimeError(f"Merged source SHA-256 mismatch for {spec['source_filename']}")
        os.replace(merging, old_archive)
        shutil.rmtree(parts_dir, ignore_errors=True)

    ext_prefix = spec["ext_prefix"]
    qt_root = f"{ext_prefix}/lib/{spec['slicer_subdir']}/qt-scripted-modules"
    fresh_runtime_files = sorted(runtime_files(REPO_ROOT))
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
                if info.is_dir() or _should_skip_member(norm, ext_prefix, qt_root):
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
        with tarfile.open(old_archive, "r:gz") as old_tf, tarfile.open(
            tmp_new, mode="w:gz", format=tarfile.PAX_FORMAT, compresslevel=6
        ) as new_tf:
            for member in old_tf:
                norm = member.name.replace("\\", "/").strip("/")
                if norm == f"{qt_root}/SlicerDSpace.py":
                    saw_qt_entry = True
                if member.issym() or member.islnk() or member.isdev():
                    raise RuntimeError(f"Unsupported link/dev member in source archive: {member.name}")
                if not norm or _should_skip_member(norm, ext_prefix, qt_root):
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
    p_base.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "build-atlas":
        return cmd_build_atlas(args)
    return cmd_repack_os_base(args)


if __name__ == "__main__":
    raise SystemExit(main())
