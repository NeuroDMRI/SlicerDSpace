#!/usr/bin/env python3
"""Maintainer: package byte-preserved WMA atlas resources into opt-in mirror parts."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tractography.wma import inspect_wma_org_atlas
from tractography.wma_atlases import atlas_definition

SOURCES = {
    "east-west": [("atlas.tar.gz", 11501574963, "abda9fd02ede00c525f4313455e4ab09")],
    "naba": [("NABA-800FC.zip", 963335626, "0342a3b9a0dbeb95e52882c1066e929c"),
             ("NABA-RegAtlas.zip", 2401732103, "aa8f880b6122273c20e9c93fd35f62ec")],
}


def digest(path, algorithm="sha256"):
    hasher = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def select_member(name, definition):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError("Unsafe upstream archive member: " + name)
    reg, cluster = definition["directories"]
    for index, part in enumerate(path.parts):
        if part not in (reg, cluster):
            continue
        relative = PurePosixPath(*path.parts[index:])
        if len(relative.parts) != 2:
            return None
        filename = relative.name
        selected = (part == reg and filename == "registration_atlas.vtk") or (
            part == cluster and (filename in ("atlas.p", "atlas.vtp", "cluster_hemisphere_location.txt")
                                 or filename.startswith("cluster_") and filename.endswith(".vtp")
                                 or filename.startswith("T") and filename.endswith(".mrml")))
        return relative.as_posix() if selected else None
    return None


def materialize_archive(path, destination, definition):
    def save(name, stream, regular):
        relative = select_member(name, definition)
        if relative is None:
            return
        if not regular:
            raise ValueError("Selected atlas member is not a regular file: " + name)
        target = destination / relative
        if any(p.is_symlink() for p in (target, *target.parents)):
            raise ValueError("Refusing a symlink atlas destination")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".atlas-copy-", dir=target.parent) as temp:
            staged = Path(temp) / "member"
            with staged.open("wb") as output:
                shutil.copyfileobj(stream, output)
            if target.exists():
                if digest(target) != digest(staged):
                    raise FileExistsError("Existing atlas member differs: " + relative)
            else:
                staged.replace(target)
    if path.name.endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist():
                if select_member(item.filename, definition):
                    with archive.open(item) as stream:
                        save(item.filename, stream, not item.is_dir() and not stat.S_ISLNK(item.external_attr >> 16))
    else:
        with tarfile.open(path, "r|gz") as archive:
            for item in archive:
                if select_member(item.name, definition):
                    if not item.isfile():
                        raise ValueError("Selected atlas member is not a regular file")
                    with archive.extractfile(item) as stream:
                        save(item.name, stream, True)


def package(family, downloads, mirror, manifest_path):
    definition = atlas_definition(family)
    destination = mirror / definition["directory"]
    destination.mkdir(parents=True, exist_ok=True)
    upstream = []
    for name, size, md5 in SOURCES[family]:
        path = downloads / name
        if not path.is_file() or path.stat().st_size != size or digest(path, "md5") != md5:
            raise ValueError("Complete official archive / published MD5 required: " + name)
        upstream.append({"filename": name, "size_bytes": size, "md5": md5, "sha256": digest(path)})
        print("[Atlas] Verified upstream " + name, flush=True)
        materialize_archive(path, destination, definition)
    inspected = inspect_wma_org_atlas(destination, family)
    if not inspected["available"]:
        raise ValueError("Atlas runtime is incomplete: " + json.dumps(inspected))
    metadata = {"atlas_family": family, "source": definition["source"], "paper": definition["paper"],
                "license": "CC-BY-4.0", "upstream_archives": upstream,
                "runtime_subset": "Original registration atlas, spectral atlas, all 800 outlier-reference clusters, hemisphere assignments and tract MRMLs.",
                "output_contract": inspected["output_contract"],
                "modifications": "Data bytes unchanged; runtime subset and archive packaging only."}
    (destination / "ATLAS.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (destination / "ATTRIBUTION.md").write_text(
        "# " + definition["label"] + "\n\nOriginal atlas data by the authors of " + definition["paper"]
        + ". Source: " + definition["source"] + ".\n\nCreative Commons Attribution 4.0 International: "
        + "https://creativecommons.org/licenses/by/4.0/ . Preserve attribution and cite the source dataset and paper.\n\n"
        + metadata["modifications"] + " See ATLAS.json for upstream checksums and retained contents.\n")
    runtime_files = sorted([path for directory in definition["directories"] for path in (destination / directory).iterdir()
                            if path.is_file() and not path.is_symlink()
                            and select_member(path.relative_to(destination).as_posix(), definition)]
                           + [destination / "ATLAS.json", destination / "ATTRIBUTION.md"])
    files = [{"path": path.relative_to(destination).as_posix(), "size_bytes": path.stat().st_size, "sha256": digest(path)}
             for path in runtime_files]
    index = destination / "extracted_resource_index.json"
    index.write_text(json.dumps({"kind": "wma_runtime_extraction_index", "atlas_family": family, "files": files}, indent=2) + "\n")
    archive_dir = destination / "archives"
    archive_dir.mkdir(exist_ok=True)
    parts = []
    with tempfile.TemporaryDirectory(prefix=".atlas-package-", dir=destination) as temp:
        archive = Path(temp) / "runtime.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as zipped:
            for path in runtime_files:
                info = zipfile.ZipInfo(path.relative_to(destination).as_posix(), (2026, 10, 30, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info._compresslevel = 1
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                with path.open("rb") as source, zipped.open(info, "w", force_zip64=True) as output:
                    shutil.copyfileobj(source, output)
        archive_sha = digest(archive)
        with archive.open("rb") as stream:
            for number in range((archive.stat().st_size + 512 * 1024**2 - 1) // (512 * 1024**2)):
                part = archive_dir / f"runtime.zip.part-{number:03d}"
                block = stream.read(512 * 1024**2)
                if part.exists() and digest(part) != hashlib.sha256(block).hexdigest():
                    raise FileExistsError("Existing mirror part differs; use a clean atlas destination")
                part.write_bytes(block)
                parts.append(part)
    manifest = json.loads(manifest_path.read_text())
    relative = lambda path: path.relative_to(mirror).as_posix()
    records = {relative(path): {"sha256": digest(path), "size_bytes": path.stat().st_size,
                               "source_ids": ["wma_atlases_mirror"],
                               "redistribution": "with-notice", "license": "CC-BY-4.0",
                               "terms_url": definition["source"], "attribution": definition["paper"]}
               for path in [*parts, index, destination / "ATTRIBUTION.md"]}
    manifest["asset_records"].update(records)
    manifest["profiles"][definition["profile"]] = {
        "directory": definition["directory"], "atlas_family": family,
        "required_paths": list(records), "extracted_index": relative(index),
        "archive_parts": list(map(relative, parts)), "archive_sha256": archive_sha,
        "expanded_size_bytes": sum(item["size_bytes"] for item in files),
        "terms_notice": [definition["label"] + ": original atlas data under CC BY 4.0. Retain attribution and cite " + definition["paper"] + ". Download only for the selected atlas route."]}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    receipt_dir = mirror / "release-manifests"
    receipt_dir.mkdir(exist_ok=True)
    receipt = {"kind": "wma_atlas_resource_mirror", "atlas_family": family, "records": records,
               "upstream": upstream, "runtime_files": len(files), "archive_sha256": archive_sha}
    (receipt_dir / ("wma-" + family + "-20261030.json")).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"atlas": family, "files": len(files), "parts": len(parts),
                      "download_bytes": sum(path.stat().st_size for path in parts)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=tuple(SOURCES), required=True)
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument("--mirror", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "models/external_assets_manifest.json")
    args = parser.parse_args()
    package(args.family, args.downloads, args.mirror.resolve(), args.manifest)


if __name__ == "__main__":
    main()
