"""Resolve optional surface components without importing the backend into Slicer."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess

from anatomy.surface.backend_bridge import BACKEND_ROOT, backend_environment, python_executable, sha256
from shared.external_process import subprocess_no_window_kwargs, text_subprocess_kwargs


def default_surface_runtime():
    path = BACKEND_ROOT / 'postprocessing/core/config.py'
    spec = importlib.util.spec_from_file_location('_ddsurfer_surface_runtime_config', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.default_prefix()


def is_valid_surface_runtime_dir(cand):
    """Return True if cand has a valid runtime.json and all listed files exist."""
    cand = Path(cand)
    manifest_file = cand / "runtime.json"
    bin_dir = cand / "bin"
    if not cand.is_dir() or not manifest_file.is_file() or not bin_dir.is_dir():
        return False
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest.get("activation_key_included") is not False:
            return False
        files = manifest.get("files")
        if files is not None:
            if not isinstance(files, dict) or not files:
                return False
            return all((cand / rel).is_file() for rel in files)
        return any(p.is_file() for p in bin_dir.iterdir())
    except (OSError, ValueError, TypeError):
        return False


class SurfaceRuntimeMacosBlocked(RuntimeError):
    """G4: the runtime holds binaries that need a newer macOS than this Mac runs."""


def _macos_block_message(home, label="surface runtime"):
    """G4: 'The <label> needs macOS 15.0 or later (this Mac: 14.2)' when ``home`` needs a newer macOS, else None.

    The minimum is read from the Mach-O binaries (G1). An unknown minimum, or an unknown macOS version
    (not a Mac), is not a block: the old behaviour stays.
    """
    from shared.macho_min_os import current_macos_version, format_version, min_macos_for_dir
    current = current_macos_version()
    if current is None:
        return None
    required = min_macos_for_dir(home)
    if required is None or current >= required:
        return None
    return "The %s needs macOS %s or later (this Mac: %s)" % (label, format_version(required), format_version(current))


def _bundled_surface_candidates(system=None, machine=None):
    from shared.native_runtimes import bundled_runtime_roots, host_platform
    sys_name, arch = host_platform(system, machine)
    folder_names = [
        f"surface-runtime-{sys_name.lower()}-{arch}",
        f"{sys_name.lower()}-{arch}",
    ]
    if sys_name == "Darwin" and arch == "arm64":
        folder_names.extend((
            "surface-runtime-darwin-x86_64",
            "darwin-x86_64",
        ))
    for root in bundled_runtime_roots():
        for name in folder_names:
            cand = root / name
            if is_valid_surface_runtime_dir(cand):
                yield sys_name, cand.resolve()


def bundled_surface_runtime(system=None, machine=None):
    """Return a bundled surface-runtime directory inside the installed extension if present and complete.

    G4: on macOS a bundle whose binaries need a newer macOS than this Mac runs is skipped, so it is never loaded
    (a dyld failure). ``surface_runtime_macos_block`` gives the reason.
    """
    for sys_name, cand in _bundled_surface_candidates(system, machine):
        if sys_name == "Darwin" and _macos_block_message(cand, "bundled surface runtime"):
            continue
        return cand
    return None


def surface_runtime_macos_block(system=None, machine=None):
    """G4: the message for a complete bundled surface runtime that this macOS cannot load, else None."""
    if bundled_surface_runtime(system, machine) is not None:
        return None
    for sys_name, cand in _bundled_surface_candidates(system, machine):
        message = _macos_block_message(cand, "bundled surface runtime") if sys_name == "Darwin" else None
        if message:
            return message
    return None


def _is_bundled(home):
    from shared.native_runtimes import bundled_runtime_roots
    try:
        parent = Path(home).resolve().parent
        return any(parent == root for root in bundled_runtime_roots())
    except OSError:
        return False


def runtime_dir_macos_block(home):
    """G4: the macOS block message for the runtime directory ``home`` (a valid native runtime), else None."""
    if not is_valid_surface_runtime_dir(home):
        return None
    return _macos_block_message(home, 'bundled surface runtime' if _is_bundled(home) else 'surface runtime')


def surface_runtime_block_reason(backend, home):
    """G4: why a resolved native runtime cannot start on this macOS (a message), else None.

    A valid runtime directory is checked on its own binaries. When ``home`` is not a valid runtime (for example the
    default prefix, because the bundled runtime was skipped), the bundled runtime's block message applies.
    """
    if backend != 'native':
        return None
    if is_valid_surface_runtime_dir(home):
        return runtime_dir_macos_block(home)
    return surface_runtime_macos_block()


def resolve_surface_runtime(runtime=None, freesurfer_home=None):
    if runtime is not None:
        return 'native', Path(runtime).expanduser().resolve()
    if freesurfer_home is not None:
        home = Path(freesurfer_home).expanduser().resolve()
        return ('native' if (home / 'runtime.json').is_file() else 'freesurfer'), home
    selected = os.environ.get('SLICERDSPACE_SURFACE_RUNTIME')
    if selected:
        return 'native', Path(selected).expanduser().resolve()
    from shared.environment import resolve_freesurfer_home
    home = resolve_freesurfer_home()
    if home is not None:
        return ('native' if (home / 'runtime.json').is_file() else 'freesurfer'), home
    bundled = bundled_surface_runtime()
    if bundled is not None:
        return 'native', bundled
    return 'native', default_surface_runtime()


def surface_runtime_identity(runtime=None, freesurfer_home=None):
    backend, home = resolve_surface_runtime(runtime, freesurfer_home)
    record = home / ('runtime.json' if backend == 'native' else 'build-stamp.txt')
    return dict(backend=backend, home=str(home), record_sha256=sha256(record) if record.is_file() else None)


def inspect_surface_runtime(runtime=None, freesurfer_home=None, license_file=None, *, python_bin=None,
                            hemisphere='both', atlases='aparc,aparc.a2009s'):
    """Verify in a fresh worker; never compile or download from a Check action."""
    script = Path(__file__).resolve().parents[1] / 'scripts/probe_surface_runtime.py'
    command = [python_executable(python_bin), str(script), '--hemi', hemisphere, '--atlases', atlases]
    try:
        backend, home = resolve_surface_runtime(runtime, freesurfer_home)
        # O5: resolve the licence before the macOS check, so a blocked Mac does not also report the licence missing.
        from shared.surface_license import resolve_surface_license
        license_file = resolve_surface_license(
            license_file, freesurfer_home=home if backend == 'freesurfer' else None,
        )
        # G4: a macOS too old for the runtime is a clear message here, not a dyld failure in the probe.
        blocked = surface_runtime_block_reason(backend, home)
        if blocked:
            raise SurfaceRuntimeMacosBlocked(blocked)
        command += ['--backend', backend, '--home', str(home)]
        if license_file is not None:
            command += ['--license-file', str(license_file)]
        result = subprocess.run(
            command,
            env=backend_environment(python_bin=python_bin),
            capture_output=True,
            timeout=60,
            **subprocess_no_window_kwargs(**text_subprocess_kwargs()),
        )
        if result.returncode:
            raise RuntimeError(result.stderr[-2000:] or result.stdout[-2000:])
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        activation = license_file or os.environ.get('FS_LICENSE')
        try:
            license_available = bool(activation and Path(activation).expanduser().is_file()
                                     and Path(activation).expanduser().stat().st_size)
        except OSError:
            license_available = False
        record = dict(components_available=False, license_available=license_available, available=False,
                      backend='native', build_required=False, error=str(exc))
        if isinstance(exc, SurfaceRuntimeMacosBlocked):
            record.update(macos_blocked=True, notes=str(exc))
        return record
