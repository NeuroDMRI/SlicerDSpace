"""Discover and activate Python packages bundled inside an installed SlicerDSpace extension."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import re
import shutil
import site
import struct
import subprocess
import sys
from typing import MutableMapping

BUNDLED_PYTHON_ENV = "SLICERDSPACE_BUNDLED_PYTHON_ROOT"
WINDOWS_CUDA_ALLOC_CONF = "garbage_collection_threshold:0.8,max_split_size_mb:128"
_TORCH_DIST_INFO_RE = re.compile(r"^torch-\d", re.IGNORECASE)
_TORCH_VERSION_CUDA_RE = re.compile(r"(?m)^\s*cuda\s*=\s*['\"][0-9]+(?:\.[0-9]+)*['\"]")
_DEQUARANTINED_ROOTS: set[Path] = set()
_WINDOWS_DLL_DIR_HANDLES: list[object] = []
_WINDOWS_REGISTERED_DLL_DIRS: set[str] = set()


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
        for payload_dir in (ext_root / "bundled-python", ext_root / "bundled-runtimes"):
            if not payload_dir.is_dir():
                continue
            added = 0
            for suffix in ("*.so", "*.dylib"):
                for binary in payload_dir.rglob(suffix):
                    if binary.is_file():
                        probe_targets.append(binary)
                        added += 1
                        if added >= 3:
                            break
                if added >= 3:
                    break
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


def _default_extension_roots(module_root: Path | str | None = None) -> tuple[Path, ...]:
    base_root = Path(module_root) if module_root is not None else Path(__file__).resolve().parents[1]
    resolved = base_root.expanduser().resolve()
    roots: list[Path] = []
    if resolved.name == "qt-scripted-modules" and len(resolved.parents) >= 3:
        roots.append(resolved.parents[2])
    roots.append(resolved)
    deduped: list[Path] = []
    for item in roots:
        if item not in deduped:
            deduped.append(item)
    return tuple(deduped)


def _normalize_arch(raw: str | None = None) -> str:
    value = str(raw or platform.machine() or "").strip().lower()
    return {
        "amd64": "x86_64",
        "x86_64": "x86_64",
        "arm64": "arm64",
        "aarch64": "arm64",
    }.get(value, value)


def _macho_arches(path: Path) -> set[str]:
    """Return Mach-O architectures supported by a binary or shared library."""
    arches: set[str] = set()
    cpu_map = {0x01000007: "x86_64", 0x0100000C: "arm64"}
    try:
        with path.open("rb") as fh:
            head = fh.read(8)
            if len(head) < 8:
                return arches
            magic_be = struct.unpack(">I", head[:4])[0]
            if magic_be == 0xCFFAEDFE:  # MH_CIGAM_64 (little-endian 64-bit Mach-O)
                cpu = struct.unpack("<I", head[4:8])[0]
                if cpu in cpu_map:
                    arches.add(cpu_map[cpu])
            elif magic_be == 0xFEEDFACF:  # MH_MAGIC_64 (big-endian 64-bit Mach-O)
                cpu = struct.unpack(">I", head[4:8])[0]
                if cpu in cpu_map:
                    arches.add(cpu_map[cpu])
            elif magic_be in (0xCAFEBABE, 0xCAFEBABF):  # FAT_MAGIC / FAT_MAGIC_64
                nfat = struct.unpack(">I", head[4:8])[0]
                if 1 <= nfat <= 16:
                    stride = 20 if magic_be == 0xCAFEBABE else 32
                    for _ in range(nfat):
                        entry = fh.read(stride)
                        if len(entry) < 4:
                            break
                        cpu = struct.unpack(">I", entry[:4])[0]
                        if cpu in cpu_map:
                            arches.add(cpu_map[cpu])
    except OSError:
        pass
    return arches


_PROBLEMS: dict[str, str] = {}
_LOGGED_PROBLEMS: set[str] = set()


def _incompatibility_reason(bundled_root: Path) -> str | None:
    """Return why the bundle cannot run in this Slicer Python, or None when it is compatible (F3)."""
    meta_file = bundled_root / "python-runtime.json"
    meta: dict[str, object] = {}
    fix = "install the SlicerDSpace package built for this Slicer"
    if meta_file.is_file():
        try:
            loaded = json.loads(meta_file.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                return "bundled Python metadata (python-runtime.json) is not a JSON object; reinstall the SlicerDSpace package"
            meta = loaded
        except (OSError, ValueError, TypeError):
            return "bundled Python metadata (python-runtime.json) cannot be read; reinstall the SlicerDSpace package"
    expected_py = meta.get("python_version")
    current_py = f"{sys.version_info[0]}.{sys.version_info[1]}"
    host_arch = _normalize_arch()
    expected_arch = meta.get("arch")
    expected_arch_text = _normalize_arch(str(expected_arch)) if expected_arch else host_arch
    if (expected_py and str(expected_py) != current_py) or expected_arch_text != host_arch:
        return (
            f"bundled Python stack is for Python {expected_py or current_py} {expected_arch_text}; "
            f"this Slicer runs Python {current_py} on {host_arch}; {fix}"
        )
    if sys.platform == "darwin":
        sp = bundled_root / "site-packages"
        if sp.is_dir():
            sampled = 0
            for ext in sp.rglob("*.so"):
                if not ext.is_file():
                    continue
                arches = _macho_arches(ext)
                if not arches:
                    continue
                if host_arch not in arches:
                    return (
                        f"bundled Python stack is for {'/'.join(sorted(arches))} "
                        f"(for example {ext.name}); this Slicer runs Python {current_py} on {host_arch}; {fix}"
                    )
                sampled += 1
                if sampled >= 10:
                    break
    return None


def _runtime_compatible(bundled_root: Path) -> bool:
    reason = _incompatibility_reason(bundled_root)
    key = str(bundled_root)
    if reason is None:
        _PROBLEMS.pop(key, None)
        return True
    _PROBLEMS[key] = reason
    return False


def activation_problem() -> str | None:
    """Return the reason a bundled Python stack was skipped, or None; the doctor shows it as a blocking row.

    None when a valid bundled root is active, even if another candidate root was skipped (P-g).
    """
    if not _PROBLEMS:
        return None
    if bundled_python_roots():  # P-g: another bundled root is valid and active; a skipped stale one is not a problem
        return None
    return next(iter(_PROBLEMS.values()), None)


def _log_activation_problem() -> None:
    for key, reason in _PROBLEMS.items():
        if key in _LOGGED_PROBLEMS:
            continue
        _LOGGED_PROBLEMS.add(key)
        import logging

        logging.getLogger(__name__).warning("SlicerDSpace: bundled Python at %s is not used: %s", key, reason)


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


def _slicer_site_packages_dirs(python_executable: Path | str | None = None) -> tuple[Path, ...]:
    """Return Slicer's own ``site-packages`` directories (never including ``bundled-python``)."""
    candidates: list[Path] = []
    if python_executable is not None:
        exe = Path(python_executable).expanduser().absolute()
        for base in (exe.parent, exe.parent.parent, exe.parent.parent.parent):
            for rel in (
                Path("lib") / "Python" / "Lib" / "site-packages",
                Path("Lib") / "site-packages",
                Path("lib") / "Python" / "lib" / f"python{sys.version_info[0]}.{sys.version_info[1]}" / "site-packages",
            ):
                candidates.append(base / rel)
    try:
        for sp in site.getsitepackages():
            candidates.append(Path(sp))
    except Exception:
        pass
    slicer_home = os.environ.get("SLICER_HOME") or os.environ.get("Slicer_HOME")
    if slicer_home:
        home = Path(slicer_home).expanduser()
        candidates.append(home / "lib" / "Python" / "Lib" / "site-packages")
        candidates.append(
            home / "lib" / "Python" / "lib" / f"python{sys.version_info[0]}.{sys.version_info[1]}" / "site-packages"
        )
    valid: list[Path] = []
    seen: set[str] = set()
    for cand in candidates:
        try:
            if not cand.is_dir():
                continue
            resolved = cand.resolve()
        except OSError:
            continue
        if any(part.lower() == "bundled-python" for part in resolved.parts):
            continue
        norm = os.path.normcase(os.path.normpath(str(resolved)))
        if norm not in seen:
            seen.add(norm)
            valid.append(resolved)
    return tuple(valid)


def register_windows_dll_directories(
    extra_site_dirs: tuple[Path | str, ...] | list[Path | str] = (),
    *,
    python_executable: Path | str | None = None,
    module_root: Path | None = None,
) -> tuple[str, ...]:
    """Register PyTorch/CUDA/pywin32 DLL directories via ``os.add_dll_directory`` on Windows Python 3.8+.

    Python 3.8+ on Windows does not search ``PATH`` when loading ``.pyd`` extension DLL
    dependencies. Handles returned by ``os.add_dll_directory`` are retained for the life
    of the process so garbage collection never removes them.
    """
    if sys.platform != "win32":
        return ()
    adder = getattr(os, "add_dll_directory", None)
    if not callable(adder):
        return ()

    site_roots: list[Path] = [
        *_slicer_site_packages_dirs(python_executable),
        *bundled_site_packages_dirs(module_root),
    ]
    for extra in extra_site_dirs:
        if extra:
            try:
                p = Path(extra).expanduser().resolve()
                if p.is_dir():
                    site_roots.append(p)
            except OSError:
                continue

    dll_candidates: list[Path] = []
    for site_dir in site_roots:
        for rel in (Path("torch") / "lib", Path("pywin32_system32")):
            cand = site_dir / rel
            if cand.is_dir():
                dll_candidates.append(cand)
        nvidia_dir = site_dir / "nvidia"
        if nvidia_dir.is_dir():
            try:
                for sub in sorted(nvidia_dir.iterdir()):
                    if not sub.is_dir():
                        continue
                    for leaf in ("bin", "lib"):
                        cand = sub / leaf
                        if cand.is_dir():
                            dll_candidates.append(cand)
            except OSError:
                pass

    for env_key in ("CUDA_PATH", "CUDA_HOME"):
        val = os.environ.get(env_key, "").strip()
        if val:
            cand = Path(val) / "bin"
            if cand.is_dir():
                dll_candidates.append(cand)

    registered: list[str] = []
    for cand in dll_candidates:
        try:
            rendered = str(cand.resolve())
        except OSError:
            rendered = str(cand)
        norm = os.path.normcase(os.path.normpath(rendered))
        if norm in _WINDOWS_REGISTERED_DLL_DIRS:
            continue
        try:
            handle = adder(rendered)
        except OSError:
            continue
        _WINDOWS_DLL_DIR_HANDLES.append(handle)
        _WINDOWS_REGISTERED_DLL_DIRS.add(norm)
        registered.append(rendered)
    return tuple(registered)


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


def site_dir_has_cuda_torch(site_dir: Path | str) -> bool:
    """Return True when ``site_dir`` contains a CUDA-enabled PyTorch installation (zero-import check)."""
    try:
        root = Path(site_dir)
        if not root.is_dir():
            return False
    except OSError:
        return False

    saw_cpu_only = False
    try:
        for entry in root.iterdir():
            name = entry.name
            if not name.lower().endswith(".dist-info") or not _TORCH_DIST_INFO_RE.match(name):
                continue
            low = name.lower()
            if "+cu" in low:
                return True
            if "+cpu" in low:
                saw_cpu_only = True
    except OSError:
        pass
    if saw_cpu_only:
        return False

    torch_lib = root / "torch" / "lib"
    try:
        if (torch_lib / "libtorch_cuda.so").is_file() or (torch_lib / "torch_cuda.dll").is_file():
            return True
    except OSError:
        pass

    version_py = root / "torch" / "version.py"
    try:
        if version_py.is_file():
            head = version_py.read_text(encoding="utf-8", errors="ignore")[:4096]
            if _TORCH_VERSION_CUDA_RE.search(head):
                return True
    except OSError:
        pass
    return False


_site_dir_has_cuda_torch = site_dir_has_cuda_torch


def inject_bundled_python_env(
    env: MutableMapping[str, str],
    *,
    module_root: Path | None = None,
    include_pythonpath: bool = True,
    python_executable: Path | str | None = None,
) -> MutableMapping[str, str]:
    """Append bundled ``site-packages`` and script directories to ``env`` for Slicer Python subprocesses."""
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if sys.platform == "win32":
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", WINDOWS_CUDA_ALLOC_CONF)
    if include_pythonpath:
        py_dirs: list[Path] = []
        for site_dir in bundled_site_packages_dirs(module_root):
            py_dirs.append(site_dir)
            for sub in ("win32", "win32/lib", "Pythonwin"):
                cand = site_dir / sub
                if cand.is_dir():
                    py_dirs.append(cand.resolve())
        bundle_has_numpy = any((site_dir / "numpy").is_dir() for site_dir in py_dirs)
        slicer_sp = _slicer_site_packages_dirs(python_executable) if sys.platform != "darwin" else ()
        slicer_has_cuda_torch = bool(slicer_sp) and any(site_dir_has_cuda_torch(sp) for sp in slicer_sp)
        prefer_slicer_sp = bool(py_dirs) and (
            (sys.platform == "win32" and not bundle_has_numpy) or slicer_has_cuda_torch
        )
        if prefer_slicer_sp and slicer_sp:
            # Release packages ship CPU-only torch in bundled-python while GPU users install
            # CUDA torch into Slicer's own site-packages (on Windows or Linux).
            # CPython inserts PYTHONPATH ahead of site-packages, so place Slicer's site-packages
            # ahead of bundled-python in PYTHONPATH to prevent shadowing CUDA torch.
            _append_env_paths(env, "PYTHONPATH", (*slicer_sp, *py_dirs))
        else:
            _append_env_paths(env, "PYTHONPATH", tuple(py_dirs))
        if bundle_has_numpy and not slicer_has_cuda_torch:
            env["SLICERDSPACE_BUNDLED_SITE_PACKAGES"] = os.pathsep.join(str(p) for p in py_dirs)
        elif slicer_has_cuda_torch:
            env.pop("SLICERDSPACE_BUNDLED_SITE_PACKAGES", None)
    _append_env_paths(env, "PATH", bundled_script_dirs(module_root))
    return env


def prepend_env_bundled_site_packages() -> tuple[str, ...]:
    """Prepend ``SLICERDSPACE_BUNDLED_SITE_PACKAGES`` entries ahead of ``sys.path`` before ``numpy`` loads."""
    if sys.platform == "win32":
        register_windows_dll_directories()
    if "numpy" in sys.modules:
        return ()
    if sys.platform != "darwin" and any(site_dir_has_cuda_torch(sp) for sp in _slicer_site_packages_dirs()):
        return ()
    raw = os.environ.get("SLICERDSPACE_BUNDLED_SITE_PACKAGES", "")
    dirs = [entry for entry in raw.split(os.pathsep) if entry and os.path.isdir(entry)]
    if not dirs:
        return ()
    if sys.platform == "win32" and not any((Path(entry) / "numpy").is_dir() for entry in dirs):
        return ()
    norms = {os.path.normcase(os.path.normpath(entry)) for entry in dirs}
    sys.path[:] = [
        *dirs,
        *(entry for entry in sys.path if os.path.normcase(os.path.normpath(entry)) not in norms),
    ]
    return tuple(dirs)


def activate_bundled_python(module_root: Path | None = None) -> tuple[Path, ...]:
    """Activate bundled ``site-packages`` in the running Slicer Python process and its child env."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if sys.platform == "win32":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", WINDOWS_CUDA_ALLOC_CONF)
        register_windows_dll_directories(module_root=module_root)
    _clear_macos_quarantine_once(module_root)
    site_dirs = bundled_site_packages_dirs(module_root)
    _log_activation_problem()
    if not site_dirs:
        return ()
    slicer_sp = _slicer_site_packages_dirs() if sys.platform != "darwin" else ()
    slicer_has_cuda_torch = bool(slicer_sp) and any(site_dir_has_cuda_torch(sp) for sp in slicer_sp)
    norm_sys_path = {os.path.normcase(os.path.normpath(entry)) for entry in sys.path if entry}
    prepend: list[str] = []
    demote: list[str] = []
    for site_dir in site_dirs:
        rendered = str(site_dir)
        norm_rendered = os.path.normcase(os.path.normpath(rendered))
        if norm_rendered not in norm_sys_path:
            site.addsitedir(rendered)
            norm_sys_path.add(norm_rendered)
        if (site_dir / "numpy").is_dir() and "numpy" not in sys.modules and not slicer_has_cuda_torch:
            prepend.append(rendered)
        elif (
            (sys.platform == "win32" and not (site_dir / "numpy").is_dir())
            or slicer_has_cuda_torch
        ) and "torch" not in sys.modules:
            demote.append(rendered)
    if prepend:
        prepend_norms = {os.path.normcase(os.path.normpath(p)) for p in prepend}
        sys.path[:] = [
            *prepend,
            *(entry for entry in sys.path if os.path.normcase(os.path.normpath(entry)) not in prepend_norms),
        ]
    elif demote:
        # Ensure Slicer's own site-packages (with any user-installed CUDA torch) stays ahead of
        # bundled-python even if inherited via PYTHONPATH.
        demote_norms = {os.path.normcase(os.path.normpath(p)) for p in demote}
        slicer_norms = {os.path.normcase(os.path.normpath(str(p))) for p in slicer_sp}
        if slicer_norms:
            kept = [entry for entry in sys.path if os.path.normcase(os.path.normpath(entry)) not in demote_norms]
            sys.path[:] = [*kept, *demote]
    if sys.platform == "win32":
        register_windows_dll_directories(site_dirs, module_root=module_root)
    inject_bundled_python_env(os.environ, module_root=module_root, include_pythonpath=True)
    return site_dirs
