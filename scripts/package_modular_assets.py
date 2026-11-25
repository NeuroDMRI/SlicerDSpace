#!/usr/bin/env python3
"""Build SHA256-pinned modular model-weight & WMA atlas archives and maintain models/release_packs_manifest.json."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RELEASE_TAG = "v1.0.0"
RELEASE_BASE_URL = f"https://github.com/NeuroDMRI/SlicerDSpace/releases/download/{RELEASE_TAG}"
ASSETS_RELEASE_TAG = "assets"
ASSETS_RELEASE_BASE_URL = f"https://github.com/NeuroDMRI/SlicerDSpace/releases/download/{ASSETS_RELEASE_TAG}"
WMA_ATLAS_RELEASE_TAG = ASSETS_RELEASE_TAG
WMA_ATLAS_RELEASE_BASE_URL = ASSETS_RELEASE_BASE_URL
SINGLE_FILE_LIMIT_BYTES = 1_950_000_000  # stay safely under GitHub Releases' 2.0 GiB ceiling
SPLIT_CHUNK_BYTES = 1_900_000_000
DETERMINISTIC_MTIME = 1767225600  # 2026-01-01T00:00:00Z

CORE_MODEL_PACK_ORDER: tuple[str, ...] = (
    "ddsurfer",
    "ddseg",
    "ddparcel",
    "pnl_cnn",
    "tractcloud",
)

CORE_MODEL_PACK_SPECS: dict[str, dict[str, Any]] = {
    "ddsurfer": {
        "id": "ddsurfer",
        "filename_template": "SlicerDSpace-Model-DDSurfer-v1.tar.gz",
        "filename": "SlicerDSpace-Model-DDSurfer-v1.tar.gz",
        "profiles": ["surface", "surface-postprocess"],
        "manifest_sources": [
            ("assets_manifest.json", "surface"),
            ("external_assets_manifest.json", "surface-postprocess"),
        ],
        "tier": "core",
        "default_install": True,
        "title_en": "DDSurfer Cortical Surface Weights, Templates & Surface Atlas",
        "title_zh": "DDSurfer 皮层表面重建模型权重、模板与表面图谱",
    },
    "ddseg": {
        "id": "ddseg",
        "filename_template": "SlicerDSpace-Model-DDSeg-v1.tar.gz",
        "filename": "SlicerDSpace-Model-DDSeg-v1.tar.gz",
        "profiles": ["segmentation"],
        "manifest_sources": [
            ("assets_manifest.json", "segmentation"),
        ],
        "tier": "core",
        "default_install": True,
        "title_en": "DDSeg Tissue-Segmentation Weights",
        "title_zh": "DDSeg 组织分割模型权重",
    },
    "ddparcel": {
        "id": "ddparcel",
        "filename_template": "SlicerDSpace-Model-DDParcel-v1.tar.gz",
        "filename": "SlicerDSpace-Model-DDParcel-v1.tar.gz",
        "profiles": ["parcellation"],
        "manifest_sources": [
            ("assets_manifest.json", "parcellation"),
        ],
        "tier": "core",
        "default_install": True,
        "title_en": "DDParcel 101-ROI Parcellation Weights & T2 Template",
        "title_zh": "DDParcel 101 脑区分区模型权重与 T2 模板",
    },
    "pnl_cnn": {
        "id": "pnl_cnn",
        "filename_template": "SlicerDSpace-Model-PNL-CNN-v1.tar.gz",
        "filename": "SlicerDSpace-Model-PNL-CNN-v1.tar.gz",
        "profiles": ["pnl-cnn-weights", "pnl-cnn"],
        "manifest_sources": [
            ("external_assets_manifest.json", "pnl-cnn-weights"),
        ],
        "tier": "core",
        "default_install": True,
        "title_en": "PNL CNN Brain-Mask Weights (PyTorch)",
        "title_zh": "PNL CNN 自动脑掩膜权重（PyTorch 版）",
    },
    "tractcloud": {
        "id": "tractcloud",
        "filename_template": "SlicerDSpace-Model-TractCloud-v1.tar.gz",
        "filename": "SlicerDSpace-Model-TractCloud-v1.tar.gz",
        "profiles": ["tractcloud"],
        "manifest_sources": [
            ("external_assets_manifest.json", "tractcloud"),
        ],
        "tier": "core",
        "default_install": True,
        "title_en": "TractCloud Deep-Learning Bundle Weights",
        "title_zh": "TractCloud 白质束深度学习权重",
    },
}

WMA_ATLAS_PACK_SPECS: dict[str, dict[str, Any]] = {
    "org": {
        "id": "wma_org",
        "profile": "wma",
        "filename": "SlicerDSpace-WMA-ORG-v1.tar.gz",
        "tier": "optional",
        "default_install": False,
        "title_en": "WMA ORG 800-Cluster White-Matter Atlas (74 Tracts)",
        "title_zh": "WMA ORG 800 簇标准白质纤维图谱（74 束）",
    },
    "naba": {
        "id": "wma_naba",
        "profile": "wma-naba",
        "filename": "SlicerDSpace-WMA-NABA-v1.tar.gz",
        "tier": "optional",
        "default_install": False,
        "title_en": "WMA NABA Lifespan White-Matter Atlas",
        "title_zh": "WMA NABA 全年龄段白质纤维图谱",
    },
    "east-west": {
        "id": "wma_east_west",
        "profile": "wma-east-west",
        "filename": "SlicerDSpace-WMA-EastWest-v1.tar.gz",
        "tier": "optional",
        "default_install": False,
        "title_en": "WMA East-West Cross-Population White-Matter Atlas",
        "title_zh": "WMA East-West 东西方人群白质图谱",
    },
}

DEFAULT_BASE_BUNDLES: dict[str, dict[str, Any]] = {
    "windows_x64_base": {
        "os": "windows",
        "arch": "x86_64",
        "mode": "base",
        "filename": f"SlicerDSpace-{RELEASE_TAG}-Windows-x64.zip",
        "url": f"{RELEASE_BASE_URL}/SlicerDSpace-{RELEASE_TAG}-Windows-x64.zip",
        "size_bytes": 0,
        "sha256": "",
        "published": False,
        "parts": [],
    },
    "linux_x64_base": {
        "os": "linux",
        "arch": "x86_64",
        "mode": "base",
        "filename": f"SlicerDSpace-{RELEASE_TAG}-Linux-x64.tar.gz",
        "url": f"{RELEASE_BASE_URL}/SlicerDSpace-{RELEASE_TAG}-Linux-x64.tar.gz",
        "size_bytes": 0,
        "sha256": "",
        "published": False,
        "parts": [],
    },
    "macos_arm64_base": {
        "os": "macos",
        "arch": "arm64",
        "mode": "base",
        "filename": f"SlicerDSpace-{RELEASE_TAG}-macOS-arm64.tar.gz",
        "url": f"{RELEASE_BASE_URL}/SlicerDSpace-{RELEASE_TAG}-macOS-arm64.tar.gz",
        "size_bytes": 0,
        "sha256": "",
        "published": False,
        "parts": [],
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_model_pack_files(pack_id: str, *, repo_root: Path = REPO_ROOT) -> dict[str, str]:
    """Return verified `{relative_path: sha256}` entries for one modular core model-weight pack."""
    if pack_id not in CORE_MODEL_PACK_SPECS:
        raise KeyError(f"Unknown core model pack: {pack_id}")
    spec = CORE_MODEL_PACK_SPECS[pack_id]
    expected: dict[str, str] = {}
    for manifest_name, profile_name in spec["manifest_sources"]:
        manifest_path = repo_root / "models" / manifest_name
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = manifest.get("asset_records") or {}
        profiles = manifest.get("profiles") or {}

        def _collect_profile(pname: str, seen: set[str]) -> None:
            if pname in seen:
                return
            seen.add(pname)
            pdef = profiles[pname]
            for child in pdef.get("include_profiles") or ():
                _collect_profile(str(child), seen)
            for rel in pdef.get("required_paths") or ():
                rel_s = str(rel)
                rec = records.get(rel_s) or {}
                if rec.get("redistribution") == "upstream-only" or rel_s.endswith("/IITmean_b0_256.nii.gz"):
                    raise ValueError(f"Refusing upstream-only asset in redistributable pack {pack_id}: {rel_s}")
                sha = str(rec.get("sha256") or "")
                if len(sha) != 64:
                    raise ValueError(f"Missing/invalid SHA-256 for {rel_s} in {manifest_name}")
                expected[rel_s] = sha

        _collect_profile(profile_name, set())
    if not expected:
        raise ValueError(f"Core model pack {pack_id} resolved zero asset files")
    return expected


def collect_wma_pack_files(asset_root: Path, ext_manifest: dict[str, Any], profile: str) -> dict[str, str]:
    """Return verified `{relative_path: sha256}` entries for an extracted WMA atlas profile."""
    prof_def = ext_manifest["profiles"][profile]
    prof_dir = prof_def["directory"]
    index_rel = prof_def.get("extracted_index", "bundles/wma/extracted_resource_index.json")
    expected: dict[str, str] = {}

    for rel in prof_def.get("required_paths", ()):
        if "/archives/" in rel:
            continue
        expected[rel] = ext_manifest["asset_records"][rel]["sha256"]

    index_path = asset_root / index_rel
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing WMA extracted resource index: {index_path}")
    actual_idx_sha = sha256_file(index_path)
    expected_idx_sha = ext_manifest["asset_records"][index_rel]["sha256"]
    if actual_idx_sha != expected_idx_sha:
        raise ValueError(f"Index SHA-256 mismatch for {index_rel}: expected {expected_idx_sha}, got {actual_idx_sha}")
    expected[index_rel] = expected_idx_sha

    index_data = json.loads(index_path.read_text(encoding="utf-8"))
    for entry in index_data.get("files") or ():
        rel = f"{prof_dir}/{entry['path']}"
        expected[rel] = entry["sha256"]
    return expected


def write_deterministic_tar_gz(
    asset_root: Path,
    files: dict[str, str],
    output_tar_gz: Path,
    *,
    compresslevel: int = 6,
) -> tuple[int, str]:
    """Stream sorted regular files into a deterministic .tar.gz and verify source hashes."""
    output_tar_gz.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_tar_gz.with_name(output_tar_gz.name + ".tmp")
    try:
        with gzip.GzipFile(filename=tmp_path, mode="wb", compresslevel=compresslevel, mtime=DETERMINISTIC_MTIME) as gz_fh:
            with tarfile.open(fileobj=gz_fh, mode="w", format=tarfile.PAX_FORMAT) as tf:
                for rel in sorted(files):
                    src = asset_root / rel
                    if not src.is_file():
                        raise FileNotFoundError(f"Missing asset file for pack {output_tar_gz.name}: {src}")
                    expected_sha = files[rel]
                    if expected_sha:
                        actual_sha = sha256_file(src)
                        if actual_sha != expected_sha:
                            raise ValueError(
                                f"Asset SHA-256 mismatch for {rel}: expected {expected_sha}, got {actual_sha}"
                            )
                    info = tf.gettarinfo(src.as_posix(), arcname=rel)
                    info.mode = 0o644
                    info.mtime = DETERMINISTIC_MTIME
                    info.uid = 0
                    info.gid = 0
                    info.uname = "root"
                    info.gname = "root"
                    with src.open("rb") as in_fh:
                        tf.addfile(info, in_fh)
        os.replace(tmp_path, output_tar_gz)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return output_tar_gz.stat().st_size, sha256_file(output_tar_gz)


def split_file_if_needed(
    archive_path: Path,
    *,
    release_base_url: str = RELEASE_BASE_URL,
    single_file_limit: int = SINGLE_FILE_LIMIT_BYTES,
    chunk_bytes: int = SPLIT_CHUNK_BYTES,
    remove_unsplit_when_split: bool = False,
) -> tuple[list[dict[str, Any]], list[Path]]:
    """Split ``archive_path`` into `.001`, `.002`, ... volumes only when above ``single_file_limit``."""
    total_size = archive_path.stat().st_size
    if total_size <= single_file_limit:
        return [], [archive_path]

    parts_meta: list[dict[str, Any]] = []
    part_paths: list[Path] = []
    index = 1
    with archive_path.open("rb") as src_fh:
        while True:
            part_name = f"{archive_path.name}.{index:03d}"
            part_path = archive_path.with_name(part_name)
            written = 0
            digest = hashlib.sha256()
            with part_path.open("wb") as out_fh:
                while written < chunk_bytes:
                    to_read = min(16 * 1024 * 1024, chunk_bytes - written)
                    buf = src_fh.read(to_read)
                    if not buf:
                        break
                    out_fh.write(buf)
                    digest.update(buf)
                    written += len(buf)
            if written == 0:
                part_path.unlink(missing_ok=True)
                break
            parts_meta.append(
                {
                    "filename": part_name,
                    "size_bytes": written,
                    "sha256": digest.hexdigest(),
                    "url": f"{release_base_url}/{part_name}",
                }
            )
            part_paths.append(part_path)
            index += 1
    if remove_unsplit_when_split:
        archive_path.unlink(missing_ok=True)
    return parts_meta, part_paths


def manifest_model_pack_entry(
    pack_id: str,
    *,
    size_bytes: int,
    sha256: str,
    files: dict[str, str],
    parts: list[dict[str, Any]] | None = None,
    published: bool = True,
    release_tag: str = ASSETS_RELEASE_TAG,
    release_base_url: str | None = None,
) -> dict[str, Any]:
    spec = CORE_MODEL_PACK_SPECS[pack_id]
    filename = spec["filename_template"].format(tag=release_tag)
    base_url = (release_base_url or f"https://github.com/NeuroDMRI/SlicerDSpace/releases/download/{release_tag}").rstrip("/")
    part_list = list(parts or ())
    entry: dict[str, Any] = {
        "id": spec["id"],
        "tier": spec.get("tier", "core"),
        "default_install": bool(spec.get("default_install", True)),
        "filename": filename,
        "url": "" if part_list else f"{base_url}/{filename}",
        "size_bytes": int(size_bytes),
        "sha256": sha256,
        "published": bool(published),
        "profiles": list(spec["profiles"]),
        "title_en": spec["title_en"],
        "title_zh": spec["title_zh"],
        "files": dict(files),
        "upstream_files": {},
    }
    if part_list:
        entry["parts"] = part_list
    return entry


def manifest_pack_entry(
    atlas_family: str,
    *,
    size_bytes: int,
    sha256: str,
    parts: list[dict[str, Any]] | None = None,
    published: bool = True,
    files: dict[str, str] | None = None,
    release_base_url: str = WMA_ATLAS_RELEASE_BASE_URL,
) -> dict[str, Any]:
    spec = WMA_ATLAS_PACK_SPECS[atlas_family]
    part_list = list(parts or ())
    entry: dict[str, Any] = {
        "id": spec["id"],
        "tier": spec.get("tier", "optional"),
        "default_install": bool(spec.get("default_install", False)),
        "filename": spec["filename"],
        "url": "" if part_list else f"{release_base_url}/{spec['filename']}",
        "size_bytes": int(size_bytes),
        "sha256": sha256,
        "published": bool(published),
        "profiles": [spec["profile"]],
        "title_en": spec["title_en"],
        "title_zh": spec["title_zh"],
        "files": dict(files or {}),
        "upstream_files": {},
    }
    if part_list:
        entry["parts"] = part_list
    return entry


def build_model_packs(
    *,
    asset_root: Path,
    output_dir: Path,
    manifest_out: Path,
    pack_ids: Iterable[str] = CORE_MODEL_PACK_ORDER,
    repo_root: Path = REPO_ROOT,
    release_tag: str = RELEASE_TAG,
    assets_release_tag: str = ASSETS_RELEASE_TAG,
    published: bool = True,
) -> dict[str, Any]:
    existing: dict[str, Any] = {}
    if manifest_out.is_file():
        try:
            existing = json.loads(manifest_out.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    base_url = f"https://github.com/NeuroDMRI/SlicerDSpace/releases/download/{release_tag}"
    assets_base_url = f"https://github.com/NeuroDMRI/SlicerDSpace/releases/download/{assets_release_tag}"
    bundles = {
        k: (existing.get("bundles") or {}).get(k, v)
        for k, v in DEFAULT_BASE_BUNDLES.items()
    }
    existing_packs = dict(existing.get("packs") or {})
    packs_payload: dict[str, dict[str, Any]] = {}
    selected = set(pack_ids)
    for pid in CORE_MODEL_PACK_ORDER:
        if pid in selected:
            files = collect_model_pack_files(pid, repo_root=repo_root)
            filename = CORE_MODEL_PACK_SPECS[pid]["filename_template"].format(tag=assets_release_tag)
            out_archive = output_dir / filename
            size_bytes, digest = write_deterministic_tar_gz(asset_root, files, out_archive)
            parts, _ = split_file_if_needed(out_archive, release_base_url=assets_base_url, remove_unsplit_when_split=True)
            packs_payload[pid] = manifest_model_pack_entry(
                pid,
                size_bytes=size_bytes,
                sha256=digest,
                files=files,
                parts=parts,
                published=published,
                release_tag=assets_release_tag,
                release_base_url=assets_base_url,
            )
            print(f"[model-pack] Built {filename} ({size_bytes:,} bytes, parts={len(parts) or 1}, sha256={digest})")
        elif pid in existing_packs:
            packs_payload[pid] = existing_packs[pid]
    for pid, entry in existing_packs.items():
        if pid not in packs_payload:
            packs_payload[pid] = entry

    payload = {
        "kind": "slicerdspace_release_packs_manifest",
        "version": 1,
        "release_tag": release_tag,
        "release_base_url": base_url,
        "assets_release_tag": assets_release_tag,
        "assets_release_base_url": assets_base_url,
        "packs": packs_payload,
        "bundles": bundles,
    }
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def build_wma_packs(
    *,
    asset_root: Path,
    output_dir: Path,
    manifest_out: Path,
    atlases: Iterable[str] = ("org", "naba", "east-west"),
    repo_root: Path = REPO_ROOT,
    published: bool = True,
) -> dict[str, Any]:
    ext_manifest = json.loads((repo_root / "models" / "external_assets_manifest.json").read_text(encoding="utf-8"))
    existing = {}
    if manifest_out.is_file():
        try:
            existing = json.loads(manifest_out.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    bundles = {
        k: (existing.get("bundles") or {}).get(k, v)
        for k, v in DEFAULT_BASE_BUNDLES.items()
    }
    existing_packs = existing.get("packs") or {}
    packs_payload: dict[str, dict[str, Any]] = {}
    for pid in CORE_MODEL_PACK_ORDER:
        if pid in existing_packs:
            packs_payload[pid] = existing_packs[pid]
    for family, spec in WMA_ATLAS_PACK_SPECS.items():
        if family in set(atlases):
            files = collect_wma_pack_files(asset_root, ext_manifest, spec["profile"])
            out_archive = output_dir / spec["filename"]
            size_bytes, digest = write_deterministic_tar_gz(asset_root, files, out_archive)
            parts, _ = split_file_if_needed(
                out_archive,
                release_base_url=WMA_ATLAS_RELEASE_BASE_URL,
                remove_unsplit_when_split=True,
            )
            key_files = {k: v for k, v in files.items() if not k.endswith(".vtp") and not k.endswith(".vtk") and not k.endswith(".mrml")}
            packs_payload[spec["id"]] = manifest_pack_entry(
                family,
                size_bytes=size_bytes,
                sha256=digest,
                parts=parts,
                published=published,
                files=key_files,
            )
            print(f"[wma-pack] Built {spec['filename']} ({size_bytes:,} bytes, parts={len(parts) or 1}, sha256={digest})")
        elif spec["id"] in existing_packs:
            packs_payload[spec["id"]] = existing_packs[spec["id"]]

    payload = {
        "kind": "slicerdspace_release_packs_manifest",
        "version": 1,
        "release_tag": existing.get("release_tag") or RELEASE_TAG,
        "release_base_url": existing.get("release_base_url") or RELEASE_BASE_URL,
        "packs": packs_payload,
        "bundles": bundles,
    }
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path, help="Verified model-assets directory.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Destination directory for modular archives.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=CORE_MODEL_PACK_ORDER,
        help="Core model weight packs to package (ddsurfer, ddseg, ddparcel, pnl_cnn, tractcloud).",
    )
    parser.add_argument(
        "--atlases",
        nargs="+",
        default=None,
        choices=tuple(WMA_ATLAS_PACK_SPECS.keys()),
        help="WMA atlas families to package.",
    )
    parser.add_argument(
        "--release-tag",
        default=RELEASE_TAG,
        help=f"Release tag for generated model weight packs (default: {RELEASE_TAG}).",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=REPO_ROOT / "models" / "release_packs_manifest.json",
        help="Output JSON path for models/release_packs_manifest.json.",
    )
    parser.add_argument("--unpublished", action="store_true", help="Mark packs as published=false in manifest.")
    args = parser.parse_args(argv)
    if args.models:
        build_model_packs(
            asset_root=args.asset_root.expanduser().resolve(),
            output_dir=args.output_dir.expanduser().resolve(),
            manifest_out=args.manifest_out.expanduser().resolve(),
            pack_ids=args.models,
            release_tag=args.release_tag,
            published=not args.unpublished,
        )
    if args.atlases or not args.models:
        build_wma_packs(
            asset_root=args.asset_root.expanduser().resolve(),
            output_dir=args.output_dir.expanduser().resolve(),
            manifest_out=args.manifest_out.expanduser().resolve(),
            atlases=args.atlases or ("org", "naba", "east-west"),
            published=not args.unpublished,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
