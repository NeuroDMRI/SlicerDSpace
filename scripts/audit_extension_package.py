#!/usr/bin/env python3
"""Compare a TGZ/ZIP extension package with the runtime source manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import tarfile
import zipfile

try:
    from scripts.extension_runtime.runtime_resources import runtime_files
except ModuleNotFoundError:
    from extension_runtime.runtime_resources import runtime_files


def _stream_sha256(stream) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return _stream_sha256(stream)


def audit_package(archive: Path, source_root: Path) -> dict:
    expected = runtime_files(source_root)
    hashes = {}
    unexpected = []
    unsafe = []
    module_roots = set()
    bundled_runtimes = set()
    bundled_sources = set()
    bundled_cli_modules = set()
    bundled_python_files = 0
    bundled_model_asset_files = 0

    def inspect_member(name: str, open_stream):
        nonlocal bundled_python_files, bundled_model_asset_files
        parts = PurePosixPath(name.replace("\\", "/")).parts
        if ".." in parts or name.startswith(("/", "\\")) or (parts and ":" in parts[0]):
            unsafe.append(name)
            return
        if parts and (parts[-1] == ".license" or parts[-1].lower() == "freesurfer_license.txt"):
            unsafe.append(f"Bundled user activation key: {name}")
            return
        if "bundled-runtimes" in parts:
            idx = parts.index("bundled-runtimes")
            if len(parts) > idx + 1:
                bundled_runtimes.add(parts[idx + 1])
            if len(parts) == idx + 3 and parts[-1] == "runtime.json":
                try:
                    with open_stream() as handle:
                        rt_meta = json.loads(handle.read().decode("utf-8"))
                    if rt_meta.get("activation_key_included") is not False:
                        unsafe.append(f"Bundled surface runtime has activation_key_included != false: {name}")
                except Exception as exc:
                    unsafe.append(f"Invalid runtime.json in {name}: {exc}")
            return
        if "bundled-sources" in parts:
            idx = parts.index("bundled-sources")
            if len(parts) > idx + 1:
                bundled_sources.add(parts[idx + 1])
            return
        if "bundled-python" in parts:
            bundled_python_files += 1
            return
        if "bundled-model-assets" in parts:
            bundled_model_asset_files += 1
            return
        if "cli-modules" in parts:
            idx = parts.index("cli-modules")
            if len(parts) > idx + 1:
                bundled_cli_modules.add(parts[idx + 1])
            return
        if "qt-scripted-modules" not in parts:
            return
        index = parts.index("qt-scripted-modules")
        module_roots.add("/".join(parts[:index + 1]))
        relative = "/".join(parts[index + 1:])
        if relative in hashes:
            unsafe.append(f"Duplicate module resource: {name}")
        with open_stream() as handle:
            hashes[relative] = _stream_sha256(handle)
        if relative not in expected and not relative.endswith((".pyc", ".pyo")):
            unexpected.append(relative)

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as stream:
            for item in stream.infolist():
                if stat.S_ISLNK(item.external_attr >> 16):
                    unsafe.append(f"Archive link: {item.filename}")
                elif not item.is_dir():
                    inspect_member(item.filename, lambda member=item: stream.open(member, "r"))
    else:
        with tarfile.open(archive, "r:*") as stream:
            for item in stream:
                if item.issym() or item.islnk():
                    unsafe.append(f"Archive link: {item.name}")
                elif item.isfile():
                    inspect_member(item.name, lambda member=item: stream.extractfile(member))

    missing = [name for name in expected if name not in hashes]
    changed = [
        name
        for name in expected
        if name in hashes and hashes[name] != _file_sha256(source_root / name)
    ]
    return {
        "kind": "slicerdsurfer_package_resource_audit",
        "status": "pass" if not (missing or changed or unexpected or unsafe) and len(module_roots) == 1 else "fail",
        "archive": str(archive.resolve()),
        "source_root": str(source_root.resolve()),
        "archive_sha256": _file_sha256(archive),
        "expected_file_count": len(expected),
        "module_roots": sorted(module_roots),
        "bundled_runtimes": sorted(bundled_runtimes),
        "bundled_sources": sorted(bundled_sources),
        "bundled_cli_modules": sorted(bundled_cli_modules),
        "bundled_python_file_count": bundled_python_files,
        "bundled_model_asset_file_count": bundled_model_asset_files,
        "missing": missing,
        "changed": changed,
        "unexpected": sorted(unexpected),
        "unsafe": unsafe,
        "scope": "Package contents only; not SDK provenance or scientific correctness",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    result = audit_package(args.archive, args.source_root)
    text = json.dumps(result, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
