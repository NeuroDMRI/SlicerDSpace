#!/usr/bin/env python3
"""Build SHA256-pinned WMA atlas archives and maintain models/release_packs_manifest.json."""

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

RELEASE_TAG = "v0.1.1"
RELEASE_BASE_URL = f"https://github.com/NeuroDMRI/SlicerDSpace/releases/download/{RELEASE_TAG}"
SINGLE_FILE_LIMIT_BYTES = 1_950_000_000  # stay safely under GitHub Releases' 2.0 GiB ceiling
SPLIT_CHUNK_BYTES = 1_900_000_000
DETERMINISTIC_MTIME = 1767225600  # 2026-01-01T00:00:00Z

WMA_ATLAS_PACK_SPECS: dict[str, dict[str, Any]] = {
    "org": {
        "id": "wma_org",
        "profile": "wma",
        "filename": f"SlicerDSpace-{RELEASE_TAG}-WMA-ORG.tar.gz",
        "title_en": "WMA ORG 800-Cluster White-Matter Atlas (74 Tracts)",
        "title_zh": "WMA ORG 800 簇标准白质纤维图谱（74 束）",
    },
    "naba": {
        "id": "wma_naba",
        "profile": "wma-naba",
        "filename": f"SlicerDSpace-{RELEASE_TAG}-WMA-NABA.tar.gz",
        "title_en": "WMA NABA Lifespan White-Matter Atlas",
        "title_zh": "WMA NABA 全年龄段白质纤维图谱",
    },
    "east-west": {
        "id": "wma_east_west",
        "profile": "wma-east-west",
        "filename": f"SlicerDSpace-{RELEASE_TAG}-WMA-EastWest.tar.gz",
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
                        raise FileNotFoundError(f"Missing WMA file for pack {output_tar_gz.name}: {src}")
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


def manifest_pack_entry(
    atlas_family: str,
    *,
    size_bytes: int,
    sha256: str,
    parts: list[dict[str, Any]] | None = None,
    published: bool = True,
    files: dict[str, str] | None = None,
    release_base_url: str = RELEASE_BASE_URL,
) -> dict[str, Any]:
    spec = WMA_ATLAS_PACK_SPECS[atlas_family]
    part_list = list(parts or ())
    entry: dict[str, Any] = {
        "id": spec["id"],
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
    packs_payload: dict[str, dict[str, Any]] = {}
    existing_packs = existing.get("packs") or {}
    for family, spec in WMA_ATLAS_PACK_SPECS.items():
        if family in set(atlases):
            files = collect_wma_pack_files(asset_root, ext_manifest, spec["profile"])
            out_archive = output_dir / spec["filename"]
            size_bytes, digest = write_deterministic_tar_gz(asset_root, files, out_archive)
            parts, _ = split_file_if_needed(out_archive, remove_unsplit_when_split=True)
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
        "release_tag": RELEASE_TAG,
        "release_base_url": RELEASE_BASE_URL,
        "packs": packs_payload,
        "bundles": bundles,
    }
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path, help="Verified model-assets directory.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Destination directory for WMA archives.")
    parser.add_argument(
        "--atlases",
        nargs="+",
        default=["org", "naba", "east-west"],
        choices=tuple(WMA_ATLAS_PACK_SPECS.keys()),
        help="WMA atlas families to package.",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=REPO_ROOT / "models" / "release_packs_manifest.json",
        help="Output JSON path for models/release_packs_manifest.json.",
    )
    parser.add_argument("--unpublished", action="store_true", help="Mark packs as published=false in manifest.")
    args = parser.parse_args(argv)
    build_wma_packs(
        asset_root=args.asset_root.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        manifest_out=args.manifest_out.expanduser().resolve(),
        atlases=args.atlases,
        published=not args.unpublished,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
