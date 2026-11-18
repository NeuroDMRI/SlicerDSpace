#!/usr/bin/env python3
"""Build SHA256-pinned modular model/atlas packs for on-demand SlicerDSpace installation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RELEASE_TAG = "v0.1.1"
RELEASE_BASE_URL = f"https://github.com/NeuroDMRI/SlicerDSpace/releases/download/{RELEASE_TAG}"
GITHUB_MAX_SINGLE_FILE_BYTES = 2_000_000_000  # stay safely below 2.0 GiB (2,147,483,648)
DEFAULT_FIXED_ZIP_TIME = (2026, 1, 1, 0, 0, 0)

# Exact verified Windows & Linux v0.1.1 split All-in-One metadata (preserved untouched on Release v0.1.1).
PRESERVED_RELEASE_BUNDLES: dict[str, dict[str, Any]] = {
    "windows_x64_all_in_one": {
        "os": "windows",
        "arch": "x86_64",
        "mode": "all_in_one",
        "filename": "SlicerDSpace-v0.1.1-Windows-x64.zip",
        "size_bytes": 3309216987,
        "sha256": "74a720cd3029c536e9ab2911455619759fe1f09d4a528c56ff0db637eaaf55c9",
        "published": True,
        "parts": [
            {
                "filename": "SlicerDSpace-v0.1.1-Windows-x64.zip.001",
                "size_bytes": 1900000000,
                "sha256": "5fc4f4acff25e90e340d12aa05744343a8717e35aa4b3bb8544e09b941e27357",
                "url": f"{RELEASE_BASE_URL}/SlicerDSpace-v0.1.1-Windows-x64.zip.001",
            },
            {
                "filename": "SlicerDSpace-v0.1.1-Windows-x64.zip.002",
                "size_bytes": 1409216987,
                "sha256": "b6ff5c1ddc7cf0c22dfbe825342a1dc273a789d9438232c236df78f36919cf7e",
                "url": f"{RELEASE_BASE_URL}/SlicerDSpace-v0.1.1-Windows-x64.zip.002",
            },
        ],
    },
    "linux_x64_all_in_one": {
        "os": "linux",
        "arch": "x86_64",
        "mode": "all_in_one",
        "filename": "SlicerDSpace-v0.1.1-Linux-x64.tar.gz",
        "size_bytes": 2813478952,
        "sha256": "76010c010a29a765fbdc338f23905284d68a0044d9323eb72a9e11c061cd8a1c",
        "published": True,
        "parts": [
            {
                "filename": "SlicerDSpace-v0.1.1-Linux-x64.tar.gz.001",
                "size_bytes": 1900000000,
                "sha256": "bd8585df009d85e96300b5715e8d2609c76b8a3706a7628eea71fb8497ebc987",
                "url": f"{RELEASE_BASE_URL}/SlicerDSpace-v0.1.1-Linux-x64.tar.gz.001",
            },
            {
                "filename": "SlicerDSpace-v0.1.1-Linux-x64.tar.gz.002",
                "size_bytes": 913478952,
                "sha256": "acad1221601fe86a022f9dc128f8a5873cf913efe1d15a666bcfb73efbe5c8c5",
                "url": f"{RELEASE_BASE_URL}/SlicerDSpace-v0.1.1-Linux-x64.tar.gz.002",
            },
        ],
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifests(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    core = json.loads((repo_root / "models" / "assets_manifest.json").read_text(encoding="utf-8"))
    ext = json.loads((repo_root / "models" / "external_assets_manifest.json").read_text(encoding="utf-8"))
    return core, ext


def _pack_definitions(core: dict[str, Any], ext: dict[str, Any]) -> list[dict[str, Any]]:
    core_records = core["asset_records"]
    ext_records = ext["asset_records"]

    def pick_core(*profiles: str) -> dict[str, str]:
        files: dict[str, str] = {}
        for prof in profiles:
            for rel in core["profiles"][prof]["required_paths"]:
                files[rel] = core_records[rel]["sha256"]
        return files

    def pick_ext(*profiles: str) -> dict[str, str]:
        files: dict[str, str] = {}
        for prof in profiles:
            for rel in ext["profiles"][prof]["required_paths"]:
                rec = ext_records[rel]
                if rec.get("redistribution") == "upstream-only":
                    continue
                files[rel] = rec["sha256"]
        return files

    surface_files = pick_core("surface")
    surface_files.update(pick_ext("surface-postprocess"))

    iit_rel = "brain_mask/pnl_cnn_torch/IITmean_b0_256.nii.gz"
    iit_rec = ext_records[iit_rel]

    return [
        {
            "id": "brain_mask_pnl",
            "filename": f"SlicerDSpace-{RELEASE_TAG}-Pack-BrainMask-PNL.zip",
            "profiles": ["pnl-cnn"],
            "title_en": "PNL CNN Brain Mask Weights",
            "title_zh": "PNL CNN 自动脑掩膜权重",
            "files": pick_ext("pnl-cnn-weights"),
            "upstream_files": {
                iit_rel: {
                    "sha256": iit_rec["sha256"],
                    "size_bytes": int(iit_rec["size_bytes"]),
                    "urls": list(iit_rec["urls"]),
                    "redistribution": iit_rec.get("redistribution", "upstream-only"),
                }
            },
        },
        {
            "id": "anatomy",
            "filename": f"SlicerDSpace-{RELEASE_TAG}-Pack-Anatomy.zip",
            "profiles": ["segmentation", "parcellation"],
            "title_en": "DDSeg Segmentation + DDParcel 101-ROI Parcellation Weights",
            "title_zh": "DDSeg 组织分割与 DDParcel 101 脑区分区权重",
            "files": pick_core("segmentation", "parcellation"),
            "upstream_files": {},
        },
        {
            "id": "surface",
            "filename": f"SlicerDSpace-{RELEASE_TAG}-Pack-Surface.zip",
            "profiles": ["surface", "surface-postprocess"],
            "title_en": "DDSurfer Cortical Reconstruction + Surface Atlas",
            "title_zh": "DDSurfer 皮层表面重建与 FreeSurfer 图谱",
            "files": surface_files,
            "upstream_files": {},
        },
        {
            "id": "tractcloud",
            "filename": f"SlicerDSpace-{RELEASE_TAG}-Pack-TractCloud.zip",
            "profiles": ["tractcloud"],
            "title_en": "TractCloud Deep-Learning Bundle Weights",
            "title_zh": "TractCloud 深度学习白质束模型",
            "files": pick_ext("tractcloud"),
            "upstream_files": {},
        },
        {
            "id": "wma_org",
            "filename": f"SlicerDSpace-{RELEASE_TAG}-Pack-WMA-ORG.zip",
            "profiles": ["wma"],
            "title_en": "WMA ORG 800-Cluster White-Matter Atlas",
            "title_zh": "WMA ORG 800 簇白质纤维图谱",
            "files": pick_ext("wma"),
            "upstream_files": {},
        },
    ]


def _write_pack_zip(
    asset_root: Path,
    files: dict[str, str],
    output_zip: Path,
) -> tuple[int, str]:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    tmp_zip = output_zip.with_name(output_zip.name + ".tmp")
    with zipfile.ZipFile(tmp_zip, "w", allowZip64=True) as zf:
        for rel in sorted(files):
            expected_sha = files[rel]
            src = asset_root / rel
            if not src.is_file():
                raise FileNotFoundError(f"Missing model asset for pack {output_zip.name}: {src}")
            actual_sha = _sha256(src)
            if actual_sha != expected_sha:
                raise ValueError(f"Asset SHA256 mismatch for {rel}: expected {expected_sha}, got {actual_sha}")
            info = zipfile.ZipInfo(filename=rel, date_time=DEFAULT_FIXED_ZIP_TIME)
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            already_compressed = rel.lower().endswith((".zip", ".tar.gz", ".tgz", ".gz"))
            info.compress_type = zipfile.ZIP_STORED if already_compressed else zipfile.ZIP_DEFLATED
            with src.open("rb") as in_fh, zf.open(info, "w") as out_fh:
                while True:
                    chunk = in_fh.read(1024 * 1024)
                    if not chunk:
                        break
                    out_fh.write(chunk)
    size_bytes = tmp_zip.stat().st_size
    if size_bytes >= GITHUB_MAX_SINGLE_FILE_BYTES:
        tmp_zip.unlink(missing_ok=True)
        raise RuntimeError(
            f"Pack {output_zip.name} size {size_bytes} exceeds single-file GitHub Release threshold"
        )
    os.replace(tmp_zip, output_zip)
    return size_bytes, _sha256(output_zip)


def build_modular_packs(
    *,
    asset_root: Path,
    output_dir: Path,
    manifest_out: Path,
    repo_root: Path = REPO_ROOT,
    published: bool = True,
) -> dict[str, Any]:
    core, ext = _load_manifests(repo_root)
    existing = {}
    if manifest_out.is_file():
        try:
            existing = json.loads(manifest_out.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    bundles = dict(existing.get("bundles") or {})
    for key, val in PRESERVED_RELEASE_BUNDLES.items():
        bundles[key] = val

    packs_payload: dict[str, dict[str, Any]] = {}
    for spec in _pack_definitions(core, ext):
        out_zip = output_dir / spec["filename"]
        size_bytes, digest = _write_pack_zip(asset_root, spec["files"], out_zip)
        packs_payload[spec["id"]] = {
            "id": spec["id"],
            "filename": spec["filename"],
            "url": f"{RELEASE_BASE_URL}/{spec['filename']}",
            "size_bytes": size_bytes,
            "sha256": digest,
            "published": bool(published),
            "profiles": spec["profiles"],
            "title_en": spec["title_en"],
            "title_zh": spec["title_zh"],
            "files": spec["files"],
            "upstream_files": spec["upstream_files"],
        }
        print(f"[modular-pack] Built {spec['filename']} ({size_bytes:,} bytes, sha256={digest})")

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
    parser.add_argument("--output-dir", required=True, type=Path, help="Destination directory for Pack-*.zip archives.")
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=REPO_ROOT / "models" / "release_packs_manifest.json",
        help="Output JSON path for models/release_packs_manifest.json.",
    )
    parser.add_argument("--unpublished", action="store_true", help="Mark packs as published=false in manifest.")
    args = parser.parse_args(argv)
    build_modular_packs(
        asset_root=args.asset_root.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        manifest_out=args.manifest_out.expanduser().resolve(),
        published=not args.unpublished,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
