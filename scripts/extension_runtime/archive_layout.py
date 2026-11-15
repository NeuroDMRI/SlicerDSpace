"""Inspect installable Slicer archive layout without extracting any members."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import stat
import tarfile
import zipfile

BUNDLED_TOP_DIRS = frozenset({
    "bundled-runtimes",
    "bundled-python",
    "bundled-sources",
    "bundled-model-assets",
})


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
    bundled_sources = set()
    bundled_cli_modules = set()
    has_bundled_python = False
    has_bundled_model_assets = False
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
        in_bundled = bool(set(path.parts) & BUNDLED_TOP_DIRS)
        if set(path.parts) & {".git", ".pytest_cache", ".mypy_cache"}:
            development.append(name)
        elif not in_bundled and (set(path.parts) & {"Testing", "tests"}):
            development.append(name)
        if "bundled-runtimes" in path.parts:
            idx = path.parts.index("bundled-runtimes")
            if len(path.parts) > idx + 1:
                bundled_runtimes.add(path.parts[idx + 1])
        if "bundled-sources" in path.parts:
            idx = path.parts.index("bundled-sources")
            if len(path.parts) > idx + 1:
                bundled_sources.add(path.parts[idx + 1])
        if "bundled-python" in path.parts:
            has_bundled_python = True
        if "bundled-model-assets" in path.parts:
            has_bundled_model_assets = True
        if "cli-modules" in path.parts:
            idx = path.parts.index("cli-modules")
            if len(path.parts) > idx + 1:
                bundled_cli_modules.add(path.parts[idx + 1])
        if "qt-scripted-modules" in path.parts:
            index = path.parts.index("qt-scripted-modules")
            if path.parts[index + 1:] == ("SlicerDSpace.py",):
                module_roots.add("/".join(path.parts[:index + 1]))
    # CPack chooses a revision/platform-prefixed top directory; compiled Python
    # bytecode is a normal Slicer build product, not proof of a dirty checkout.
    return {
        "status": "pass" if len(module_roots) == 1 and not unsafe and not development else "fail",
        "module_roots": sorted(module_roots),
        "bundled_runtimes": sorted(bundled_runtimes),
        "bundled_sources": sorted(bundled_sources),
        "bundled_cli_modules": sorted(bundled_cli_modules),
        "bundled_python": has_bundled_python,
        "bundled_model_assets": has_bundled_model_assets,
        "unsafe": unsafe,
        "development_artifacts": development,
        "bytecode_count": sum(name.endswith((".pyc", ".pyo")) for name in members),
    }
