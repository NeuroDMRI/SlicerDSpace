"""Inspect installable Slicer archive layout without extracting any members."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import stat
import tarfile
import zipfile


def inspect_archive_layout(archive: Path):
    members = []
    unsafe = []
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as stream:
            for item in stream.infolist():
                if stat.S_ISLNK(item.external_attr >> 16):
                    unsafe.append(item.filename)
                elif not item.is_dir():
                    members.append(item.filename)
    else:
        with tarfile.open(archive, "r:*") as stream:
            for item in stream:
                if item.issym() or item.islnk():
                    unsafe.append(item.name)
                elif item.isfile():
                    members.append(item.name)
    module_roots = set()
    bundled_runtimes = set()
    development = []
    seen = set()
    for name in members:
        normalized = name.replace("\\", "/")
        path = PurePosixPath(normalized)
        if path.is_absolute() or ".." in path.parts or (path.parts and ":" in path.parts[0]):
            unsafe.append(name)
        if normalized in seen:
            unsafe.append(name)
        seen.add(normalized)
        if set(path.parts) & {".git", ".pytest_cache", ".mypy_cache"}:
            development.append(name)
        elif "bundled-runtimes" not in path.parts and (set(path.parts) & {"Testing", "tests"}):
            development.append(name)
        if "bundled-runtimes" in path.parts:
            idx = path.parts.index("bundled-runtimes")
            if len(path.parts) > idx + 1:
                bundled_runtimes.add(path.parts[idx + 1])
        if "qt-scripted-modules" in path.parts:
            index = path.parts.index("qt-scripted-modules")
            if path.parts[index + 1:] == ("SlicerDSpace.py",):
                module_roots.add("/".join(path.parts[:index + 1]))
    # CPack chooses a revision/platform-prefixed top directory; compiled Python
    # bytecode is a normal Slicer build product, not proof of a dirty checkout.
    return {"status": "pass" if len(module_roots) == 1 and not unsafe and not development else "fail",
            "module_roots": sorted(module_roots),
            "bundled_runtimes": sorted(bundled_runtimes),
            "unsafe": unsafe,
            "development_artifacts": development,
            "bytecode_count": sum(name.endswith((".pyc", ".pyo")) for name in members)}
