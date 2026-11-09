#!/usr/bin/env python3
"""Create a clean, rooted archive for local Slicer Extension Manager tests."""

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
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


EXCLUDED_DIR_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache"}
EXCLUDED_FILE_NAMES = {"extension_staging_manifest.json"}
EXCLUDED_FILE_SUFFIXES = {".pyc", ".pyo"}


def _is_excluded(path: Path, extension_root: Path) -> bool:
    relative = path.relative_to(extension_root)
    return (
        any(part in EXCLUDED_DIR_NAMES for part in relative.parts)
        or path.name in EXCLUDED_FILE_NAMES
        or path.suffix.lower() in EXCLUDED_FILE_SUFFIXES
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stage_surface_runtime(surface_archive: Path, bundled_root: Path) -> dict[str, object]:
    catalog = json.loads((REPO_ROOT / "models" / "runtime_bundles.json").read_text(encoding="utf-8"))
    digest = _sha256(surface_archive)
    record = next((item for item in catalog.get("surface", ()) if item.get("sha256") == digest), None)
    if record is None:
        raise ValueError(f"Surface runtime archive SHA256 {digest} does not match models/runtime_bundles.json")
    system = str(record["system"]).lower()
    machine = str(record["machine"]).lower()
    dest = bundled_root / f"surface-runtime-{system}-{machine}"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(surface_archive, "r:gz") as bundle:
        for member in bundle:
            rel = PurePosixPath(member.name)
            if rel.is_absolute() or ".." in rel.parts or "\\" in member.name or ":" in member.name or member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"Unsafe surface runtime archive member: {member.name}")
            if not member.isfile():
                continue
            target = dest.joinpath(*rel.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.extractfile(member) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
            target.chmod(0o755 if (member.mode & 0o111) else 0o644)
    manifest_file = dest / "runtime.json"
    if not manifest_file.is_file():
        raise ValueError("Bundled surface runtime is missing runtime.json")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("activation_key_included") is not False:
        raise ValueError("Bundled surface runtime must have activation_key_included=false")
    return {
        "name": "surface",
        "system": record["system"],
        "machine": record["machine"],
        "archive_sha256": digest,
        "directory": f"bundled-runtimes/{dest.name}",
        "activation_key_included": False,
    }


def _stage_bundled_runtimes(
    extension_root: Path,
    *,
    mrtrix_archive: Path | None = None,
    fsl_archive: Path | None = None,
    surface_archive: Path | None = None,
) -> list[dict[str, object]]:
    if not any((mrtrix_archive, fsl_archive, surface_archive)):
        return []
    from shared.native_runtimes import install_bundle, load_catalog

    bundled_root = extension_root / "bundled-runtimes"
    if bundled_root.exists():
        shutil.rmtree(bundled_root)
    bundled_root.mkdir(parents=True, exist_ok=True)
    catalog = load_catalog()
    staged: list[dict[str, object]] = []
    for tool_name, archive_path in (("mrtrix", mrtrix_archive), ("fsl", fsl_archive)):
        if archive_path is None:
            continue
        resolved_archive = archive_path.expanduser().resolve()
        digest = _sha256(resolved_archive)
        record = next(
            (r for r in catalog.get("bundles", ()) if r.get("name") == tool_name and r.get("sha256") == digest),
            None,
        )
        if record is None:
            raise ValueError(f"{tool_name} archive SHA256 {digest} does not match models/native_runtime_bundles.json")
        dest = install_bundle(record, archive=resolved_archive, root=bundled_root, accept_terms=True)
        staged.append({
            "name": tool_name,
            "version": record["version"],
            "system": record["system"],
            "machine": record["machine"],
            "archive_sha256": digest,
            "directory": f"bundled-runtimes/{dest.name}",
            "command_count": len(record["commands"]),
        })
    if surface_archive is not None:
        staged.append(_stage_surface_runtime(surface_archive.expanduser().resolve(), bundled_root))
    return staged


def package_extension(
    extension_root: Path,
    output_path: Path,
    *,
    mrtrix_archive: Path | None = None,
    fsl_archive: Path | None = None,
    surface_archive: Path | None = None,
) -> dict[str, object]:
    """Package a staged extension with the top-level directory Slicer expects."""

    root = extension_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Staged extension root does not exist: {root}")
    module_files = sorted(root.glob("lib/Slicer-*/qt-scripted-modules/SlicerDSpace.py"))
    if len(module_files) != 1:
        raise ValueError(f"Expected exactly one staged SlicerDSpace.py, found {len(module_files)}")
    if not (root / "SlicerDSpace.s4ext").is_file():
        raise FileNotFoundError(f"Extension metadata is missing: {root / 'SlicerDSpace.s4ext'}")

    bundled_runtimes = _stage_bundled_runtimes(
        root,
        mrtrix_archive=mrtrix_archive,
        fsl_archive=fsl_archive,
        surface_archive=surface_archive,
    )

    output = output_path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    members = [root]
    members.extend(path for path in sorted(root.rglob("*")) if not _is_excluded(path, root))
    if output.name.lower().endswith(".zip"):
        with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in members:
                if not path.is_file():
                    continue
                relative = path.relative_to(root)
                archive_name = (Path(root.name) / relative).as_posix()
                info = zipfile.ZipInfo.from_file(path, arcname=archive_name)
                info.compress_type = zipfile.ZIP_DEFLATED
                mode = 0o755 if (os.access(path, os.X_OK) or path.suffix.lower() in {".exe", ".dll", ".so", ".dylib"}) else 0o644
                info.external_attr = (stat.S_IFREG | mode) << 16
                with path.open("rb") as src, archive.open(info, "w") as dst:
                    shutil.copyfileobj(src, dst)
    else:
        with tarfile.open(output, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
            for path in members:
                relative = path.relative_to(root)
                archive_name = Path(root.name) / relative
                info = archive.gettarinfo(path.as_posix(), arcname=archive_name.as_posix())
                info.uid = 0
                info.gid = 0
                info.uname = "root"
                info.gname = "root"
                if path.is_file():
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)
                else:
                    archive.addfile(info)

    payload = {
        "kind": "slicerdsurfer_extension_package",
        "version": 1,
        "extension_name": root.name,
        "extension_root": root.as_posix(),
        "archive": output.as_posix(),
        "archive_size_bytes": output.stat().st_size,
        "archive_sha256": _sha256(output),
        "member_count": len(members),
        "module_file": module_files[0].relative_to(root).as_posix(),
        "bundled_runtimes": bundled_runtimes,
        "excluded_directory_names": sorted(EXCLUDED_DIR_NAMES),
        "excluded_file_names": sorted(EXCLUDED_FILE_NAMES),
        "excluded_file_suffixes": sorted(EXCLUDED_FILE_SUFFIXES),
    }
    manifest_path = output.with_name(output.name + ".manifest.json")
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload["manifest_path"] = manifest_path.as_posix()
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mrtrix-archive", type=Path, help="Optional modular MRtrix3 runtime archive to bundle.")
    parser.add_argument("--fsl-archive", type=Path, help="Optional modular FSL runtime archive to bundle.")
    parser.add_argument("--surface-archive", type=Path, help="Optional modular FreeSurfer surface runtime archive to bundle.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = package_extension(
        args.extension_root,
        args.output,
        mrtrix_archive=args.mrtrix_archive,
        fsl_archive=args.fsl_archive,
        surface_archive=args.surface_archive,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
