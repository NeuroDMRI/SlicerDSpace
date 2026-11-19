"""Discover and activate Python packages bundled inside an installed SlicerDSpace extension."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import site
import subprocess
import sys
from typing import MutableMapping

BUNDLED_PYTHON_ENV = "SLICERDSPACE_BUNDLED_PYTHON_ROOT"
_DEQUARANTINED_ROOTS: set[Path] = set()


def _clear_macos_quarantine_once(module_root: Path | None = None) -> bool:
    """Clear macOS quarantine xattrs once per process so manual extension installs avoid Gatekeeper stalls."""
    if sys.platform != "darwin" or os.environ.get("SLICERDSPACE_SKIP_DEQUARANTINE") == "1":
        return False
    xattr_bin = shutil.which("xattr")
    if not xattr_bin:
        return False
    cleared = False
    for ext_root in _default_extension_roots(module_root):
        if ext_root in _DEQUARANTINED_ROOTS:
            continue
        if not ((ext_root / "bundled-python").is_dir() or (ext_root / "bundled-runtimes").is_dir()):
            continue
        _DEQUARANTINED_ROOTS.add(ext_root)
        probe_targets = [
            cand
            for cand in (
                ext_root,
                ext_root / "bundled-python",
                ext_root / "bundled-python" / "site-packages",
                ext_root / "bundled-runtimes",
            )
            if cand.exists()
        ]
        has_quarantine = False
        for target in probe_targets:
            try:
                probe = subprocess.run(
                    [xattr_bin, "-p", "com.apple.quarantine", str(target)],
                    capture_output=True,
                    timeout=5,
                )
                if probe.returncode == 0:
                    has_quarantine = True
                    break
            except Exception:
                continue
        if not has_quarantine:
            continue
        try:
            proc = subprocess.run(
                [xattr_bin, "-dr", "com.apple.quarantine", str(ext_root)],
                capture_output=True,
                timeout=60,
            )
            if proc.returncode == 0:
                cleared = True
        except Exception:
            pass
    return cleared


def _default_extension_roots(module_root: Path | None = None) -> tuple[Path, ...]:
    resolved = (module_root or Path(__file__).resolve().parents[1]).expanduser().resolve()
    roots: list[Path] = []
    if resolved.name == "qt-scripted-modules" and len(resolved.parents) >= 3:
        roots.append(resolved.parents[2])
    roots.append(resolved)
    deduped: list[Path] = []
    for item in roots:
        if item not in deduped:
            deduped.append(item)
    return tuple(deduped)


def _runtime_compatible(bundled_root: Path) -> bool:
    meta_file = bundled_root / "python-runtime.json"
    if not meta_file.is_file():
        return True
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    expected_py = meta.get("python_version")
    current_py = f"{sys.version_info[0]}.{sys.version_info[1]}"
    if expected_py and str(expected_py) != current_py:
        return False
    return True


def bundled_python_roots(module_root: Path | None = None) -> tuple[Path, ...]:
    """Return existing ``bundled-python`` directories for the current extension."""
    env_override = os.environ.get(BUNDLED_PYTHON_ENV)
    candidates: list[Path] = []
    if env_override:
        override = Path(env_override).expanduser().resolve()
        if override.name == "site-packages" and override.is_dir():
            candidates.append(override.parent)
        else:
            candidates.append(override)
    else:
        for ext_root in _default_extension_roots(module_root):
            candidates.append(ext_root / "bundled-python")

    valid: list[Path] = []
    for cand in candidates:
        if cand.is_dir() and (cand / "site-packages").is_dir() and _runtime_compatible(cand):
            resolved = cand.resolve()
            if resolved not in valid:
                valid.append(resolved)
    return tuple(valid)


def bundled_site_packages_dirs(module_root: Path | None = None) -> tuple[Path, ...]:
    """Return existing ``bundled-python/site-packages`` directories."""
    return tuple((root / "site-packages").resolve() for root in bundled_python_roots(module_root))


def bundled_script_dirs(module_root: Path | None = None) -> tuple[Path, ...]:
    """Return existing script/binary directories inside ``bundled-python``."""
    dirs: list[Path] = []
    for root in bundled_python_roots(module_root):
        for rel in (
            "site-packages/bin",
            "site-packages/Scripts",
            "site-packages/pywin32_system32",
            "bin",
            "Scripts",
        ):
            cand = root / rel
            if cand.is_dir():
                resolved = cand.resolve()
                if resolved not in dirs:
                    dirs.append(resolved)
    return tuple(dirs)


def _append_env_paths(env: MutableMapping[str, str], key: str, additions: tuple[Path, ...]) -> None:
    if not additions:
        return
    existing = [part for part in env.get(key, "").split(os.pathsep) if part]
    norm_existing = {os.path.normcase(os.path.normpath(part)) for part in existing}
    for path in additions:
        rendered = str(path)
        norm_rendered = os.path.normcase(os.path.normpath(rendered))
        if norm_rendered not in norm_existing:
            existing.append(rendered)
            norm_existing.add(norm_rendered)
    env[key] = os.pathsep.join(existing)


def inject_bundled_python_env(
    env: MutableMapping[str, str],
    *,
    module_root: Path | None = None,
    include_pythonpath: bool = True,
) -> MutableMapping[str, str]:
    """Append bundled ``site-packages`` and script directories to ``env`` for Slicer Python subprocesses."""
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if include_pythonpath:
        py_dirs: list[Path] = []
        for site_dir in bundled_site_packages_dirs(module_root):
            py_dirs.append(site_dir)
            for sub in ("win32", "win32/lib", "Pythonwin"):
                cand = site_dir / sub
                if cand.is_dir():
                    py_dirs.append(cand.resolve())
        _append_env_paths(env, "PYTHONPATH", tuple(py_dirs))
        if any((site_dir / "numpy").is_dir() for site_dir in py_dirs):
            env["SLICERDSPACE_BUNDLED_SITE_PACKAGES"] = os.pathsep.join(str(p) for p in py_dirs)
    _append_env_paths(env, "PATH", bundled_script_dirs(module_root))
    return env


def activate_bundled_python(module_root: Path | None = None) -> tuple[Path, ...]:
    """Activate bundled ``site-packages`` in the running Slicer Python process and its child env."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    _clear_macos_quarantine_once(module_root)
    site_dirs = bundled_site_packages_dirs(module_root)
    if not site_dirs:
        return ()
    norm_sys_path = {os.path.normcase(os.path.normpath(entry)) for entry in sys.path if entry}
    prepend: list[str] = []
    for site_dir in site_dirs:
        rendered = str(site_dir)
        norm_rendered = os.path.normcase(os.path.normpath(rendered))
        if norm_rendered not in norm_sys_path:
            site.addsitedir(rendered)
            norm_sys_path.add(norm_rendered)
        if (site_dir / "numpy").is_dir() and "numpy" not in sys.modules:
            prepend.append(rendered)
    if prepend:
        prepend_norms = {os.path.normcase(os.path.normpath(p)) for p in prepend}
        sys.path[:] = [
            *prepend,
            *(entry for entry in sys.path if os.path.normcase(os.path.normpath(entry)) not in prepend_norms),
        ]
    inject_bundled_python_env(os.environ, module_root=module_root, include_pythonpath=True)
    return site_dirs
