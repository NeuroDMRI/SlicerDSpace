"""Resolve optional surface components without importing the backend into Slicer."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess

from anatomy.surface.backend_bridge import BACKEND_ROOT, backend_environment, python_executable, sha256


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


def bundled_surface_runtime(system=None, machine=None):
    """Return a bundled surface-runtime directory inside the installed extension if present and complete."""
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
                return cand.resolve()
    return None


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
        command += ['--backend', backend, '--home', str(home)]
        from shared.surface_license import resolve_surface_license
        license_file = resolve_surface_license(
            license_file, freesurfer_home=home if backend == 'freesurfer' else None,
        )
        if license_file is not None:
            command += ['--license-file', str(license_file)]
        result = subprocess.run(command, env=backend_environment(), capture_output=True, text=True, timeout=60)
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
        return dict(components_available=False, license_available=license_available, available=False,
                    backend='native', build_required=False, error=str(exc))
