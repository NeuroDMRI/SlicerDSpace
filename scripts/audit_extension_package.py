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


def audit_package(archive: Path, source_root: Path) -> dict:
    expected = runtime_files(source_root)
    hashes = {}
    unexpected = []
    unsafe = []
    module_roots = set()
    bundled_runtimes = set()

    def record(name, payload):
        parts = PurePosixPath(name.replace("\\", "/")).parts
        if ".." in parts or name.startswith(("/", "\\")) or (parts and ":" in parts[0]):
            unsafe.append(name)
            return
        if "bundled-runtimes" in parts:
            idx = parts.index("bundled-runtimes")
            if len(parts) > idx + 1:
                bundled_runtimes.add(parts[idx + 1])
            if parts[-1] == ".license" or parts[-1].lower() == "freesurfer_license.txt":
                unsafe.append(f"Bundled user activation key: {name}")
            if len(parts) == idx + 3 and parts[-1] == "runtime.json":
                try:
                    rt_meta = json.loads(payload.decode("utf-8"))
                    if rt_meta.get("activation_key_included") is not False:
                        unsafe.append(f"Bundled surface runtime has activation_key_included != false: {name}")
                except Exception as exc:
                    unsafe.append(f"Invalid runtime.json in {name}: {exc}")
            return
        if "qt-scripted-modules" not in parts:
            return
        index = parts.index("qt-scripted-modules")
        module_roots.add("/".join(parts[:index + 1]))
        relative = "/".join(parts[index + 1:])
        if relative in hashes:
            unsafe.append(f"Duplicate module resource: {name}")
        hashes[relative] = hashlib.sha256(payload).hexdigest()
        if relative not in expected and not relative.endswith((".pyc", ".pyo")):
            unexpected.append(relative)

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as stream:
            for item in stream.infolist():
                if stat.S_ISLNK(item.external_attr >> 16):
                    unsafe.append(f"Archive link: {item.filename}")
                elif not item.is_dir():
                    record(item.filename, stream.read(item))
    else:
        with tarfile.open(archive, "r:*") as stream:
            for item in stream:
                if item.issym() or item.islnk():
                    unsafe.append(f"Archive link: {item.name}")
                elif item.isfile():
                    record(item.name, stream.extractfile(item).read())
    missing = [name for name in expected if name not in hashes]
    changed = [name for name in expected if name in hashes and hashes[name] != hashlib.sha256((source_root / name).read_bytes()).hexdigest()]
    return {
        "kind": "slicerdsurfer_package_resource_audit", "status": "pass" if not (missing or changed or unexpected or unsafe) and len(module_roots) == 1 else "fail",
        "archive": str(archive.resolve()), "source_root": str(source_root.resolve()),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "expected_file_count": len(expected), "module_roots": sorted(module_roots),
        "bundled_runtimes": sorted(bundled_runtimes),
        "missing": missing, "changed": changed, "unexpected": sorted(unexpected), "unsafe": unsafe,
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
