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


class SurfaceRuntimeMissing(RuntimeError):
    """b2-os: no runtime is installed, and Prepare can download one from the catalog."""


def _macos_block_message(home, label="surface runtime"):
    """G4: 'The <label> needs macOS 15.0 or later (this Mac: 14.2)' when ``home`` needs a newer macOS, else None.

    The minimum is read from the Mach-O binaries (G1). An unknown minimum, or an unknown macOS version
    (not a Mac), is not a block: the old behaviour stays.
    """
    versions = _macos_block_versions(home)
    if versions is None:
        return None
    return "The %s needs macOS %s or later (this Mac: %s)" % (label, versions[0], versions[1])


def _macos_block_versions(home):
    """b2-os: ``(required, current)`` as version text when ``home`` needs a newer macOS than this Mac, else None."""
    from shared.macho_min_os import current_macos_version, format_version, min_macos_for_dir
    current = current_macos_version()
    if current is None:
        return None
    required = min_macos_for_dir(home)
    if required is None or current >= required:
        return None
    return format_version(required), format_version(current)


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


def _resolve_surface_runtime_source(runtime=None, freesurfer_home=None):
    """``(backend, home, source)``; ``source`` is 'explicit', 'environment', 'freesurfer_home', 'bundled' or 'default'."""
    if runtime is not None:
        return 'native', Path(runtime).expanduser().resolve(), 'explicit'
    if freesurfer_home is not None:
        home = Path(freesurfer_home).expanduser().resolve()
        return ('native' if (home / 'runtime.json').is_file() else 'freesurfer'), home, 'explicit'
    selected = os.environ.get('SLICERDSPACE_SURFACE_RUNTIME')
    if selected:
        return 'native', Path(selected).expanduser().resolve(), 'environment'
    from shared.environment import resolve_freesurfer_home
    home = resolve_freesurfer_home()
    if home is not None:
        return ('native' if (home / 'runtime.json').is_file() else 'freesurfer'), home, 'freesurfer_home'
    bundled = bundled_surface_runtime()
    if bundled is not None:
        return 'native', bundled, 'bundled'
    return 'native', default_surface_runtime(), 'default'


def resolve_surface_runtime(runtime=None, freesurfer_home=None):
    backend, home, _ = _resolve_surface_runtime_source(runtime, freesurfer_home)
    return backend, home


# b2-os: the user-facing texts of an OS-blocked surface runtime. The setting is named as the UI names it
# (Environment > More options > External Runtime Paths > FreeSurfer home); no FreeSurfer version is named.
_OS_BLOCK_HINT = {
    'en': ('To run Surface postprocess on this Mac: upgrade macOS to %s or later; or set the folder of a compatible '
           'FreeSurfer installation in More options (offline, repair, paths) > External Runtime Paths > '
           'FreeSurfer home (optional); or untick Surface postprocess (the other steps still run).'),
    'zh': ('要在本机运行表面后处理：将 macOS 升级到 %s 或更高版本；或在“更多选项（离线、修复、路径）> 外部运行环境路径 > '
           'FreeSurfer 主目录（可选）”中设置一个兼容的 FreeSurfer 安装文件夹；或取消勾选“表面后处理”（其他步骤照常运行）。'),
}
_OS_BLOCK_REASON_ZH = {True: '随包的表面运行环境需要 macOS %s 或更高版本（本机：%s）',
                       False: '表面运行环境需要 macOS %s 或更高版本（本机：%s）'}
_DOWNLOAD_HINT = {
    'en': 'Use Environment > Prepare to download the surface runtime built for this macOS (%.1f MB).',
    'zh': '请使用“环境 > 准备”下载适用于本机 macOS 的表面运行环境（%.1f MB）。',
}


def os_block_texts(required, current, *, bundled=True, remedy='none', download_bytes=0):
    """b2-os: ``{'reason', 'reason_zh', 'hint', 'hint_zh'}`` for a runtime that needs macOS ``required``.

    The reason names both versions; the hint names actions that can fix it, never the reason again.
    """
    label = 'bundled surface runtime' if bundled else 'surface runtime'
    texts = dict(reason='The %s needs macOS %s or later (this Mac: %s)' % (label, required, current),
                 reason_zh=_OS_BLOCK_REASON_ZH[bool(bundled)] % (required, current))
    if remedy == 'download':
        size = float(download_bytes or 0) / 1e6
        texts.update(hint=_DOWNLOAD_HINT['en'] % size, hint_zh=_DOWNLOAD_HINT['zh'] % size)
    else:
        texts.update(hint=_OS_BLOCK_HINT['en'] % required, hint_zh=_OS_BLOCK_HINT['zh'] % required)
    return texts


def _catalog_download(asset_root=None):
    """b2-os, decision 2: ``(True, bytes)`` when the catalog has a runtime this platform can download and run."""
    try:
        from shared.prebuilt_surface import bundle_for_platform
        record = bundle_for_platform()  # None when this Mac is below the record's min_macos
        if not record or not record.get('url'):
            return False, 0
        available = bool(record.get('published', False))
        if not available and record.get('asset'):
            from shared.model_assets import configured_model_assets_root
            available = (configured_model_assets_root(asset_root) / record['asset']).is_file()
        return (True, int(record.get('bytes') or 0)) if available else (False, 0)
    except Exception:
        return False, 0


def _catalog_macos_block_versions():
    """b2-os: ``(required, current)`` when the catalog runtime of this Mac needs a newer macOS, else None.

    The case of a Mac with no installed runtime: the catalog's ``min_macos`` is then the true cause, not "missing".
    """
    try:
        import platform
        from shared.macho_min_os import current_macos_version, format_version, parse_version_text
        from shared.prebuilt_surface import CATALOG
        current = current_macos_version()
        if current is None or platform.system() != 'Darwin':
            return None
        machine = platform.machine().lower()
        if machine in ('x86_64', 'amd64'):
            from shared.platform_info import host_machine
            machine = 'arm64' if host_machine() == 'arm64' else 'x86_64'
        machine = {'aarch64': 'arm64', 'amd64': 'x86_64'}.get(machine, machine)
        payload = json.loads(CATALOG.read_text(encoding='utf-8'))
        record = next((item for item in payload.get('surface', ())
                       if item.get('system') == 'Darwin' and item.get('machine') == machine), None)
        required = parse_version_text(record.get('min_macos')) if record and record.get('min_macos') else None
        if required is None or current >= required:
            return None
        return format_version(required), format_version(current)
    except Exception:
        return None


def surface_runtime_state(runtime=None, freesurfer_home=None, *, asset_root=None):
    """b2-os, decision 1: the one structured state of the surface runtime that the run would use.

    ``state`` is 'ready', 'missing' or 'os_blocked'. ``remedy`` is 'download' when a catalog runtime that this
    platform can run would be used after Prepare installs it, else 'none' (also for a path the user set: the
    resolver keeps it, so a download cannot replace it). ``macos_required`` / ``macos_current`` are set for
    'os_blocked'. ``reason`` / ``hint`` (and ``_zh``) are the texts; the hint is empty when there is no action.
    It reads files only (no probe, no network).
    """
    backend, home, source = _resolve_surface_runtime_source(runtime, freesurfer_home)
    explicit = source in ('explicit', 'environment', 'freesurfer_home')
    state = dict(state='missing', remedy='none', macos_required='', macos_current='', download_bytes=0,
                 backend=backend, home=str(home), explicit=explicit, reason='', reason_zh='', hint='', hint_zh='')
    versions, bundled = None, False
    if source == 'bundled':  # bundled_surface_runtime returns only a complete runtime that this OS can load
        state['state'] = 'ready'
        return state
    if backend == 'freesurfer':
        if (home / 'build-stamp.txt').is_file():
            state['state'] = 'ready'
            return state
    elif is_valid_surface_runtime_dir(home):
        versions, bundled = _macos_block_versions(home), _is_bundled(home)
        if versions is None:
            state['state'] = 'ready'
            return state
    elif not explicit:
        for sys_name, cand in _bundled_surface_candidates():
            versions = _macos_block_versions(cand) if sys_name == 'Darwin' else None
            if versions:
                bundled = True
                break
        if versions is None:
            versions = _catalog_macos_block_versions()  # no runtime here, and the catalog's needs a newer macOS
    # A path the user set is kept by the resolver, so a download cannot replace it. b2-os F5: a blocked copy in the
    # default prefix keeps the download remedy: prebuilt_surface.prepare() validates that prefix and replaces it.
    downloadable, size = (False, 0) if explicit else _catalog_download(asset_root)
    if downloadable:
        state.update(remedy='download', download_bytes=size)
    if versions:
        state.update(state='os_blocked', macos_required=versions[0], macos_current=versions[1])
        state.update(os_block_texts(versions[0], versions[1], bundled=bundled, remedy=state['remedy'],
                                    download_bytes=size))
    return state


def surface_postprocess_os_skip(backend_paths=None):
    """b2-os, decision 6: the state when Surface postprocess cannot run on this OS and nothing can fix it, else None.

    ``backend_paths`` are the saved external paths (``postprocess_runtime``, ``postprocess_freesurfer_home``).
    """
    paths = dict(backend_paths or {})
    try:
        state = surface_runtime_state(paths.get('postprocess_runtime') or None,
                                      paths.get('postprocess_freesurfer_home') or None,
                                      asset_root=paths.get('model_assets_dir') or None)
    except Exception:
        return None
    return state if state['state'] == 'os_blocked' and state['remedy'] == 'none' else None


def surface_runtime_identity(runtime=None, freesurfer_home=None):
    backend, home = resolve_surface_runtime(runtime, freesurfer_home)
    record = home / ('runtime.json' if backend == 'native' else 'build-stamp.txt')
    return dict(backend=backend, home=str(home), record_sha256=sha256(record) if record.is_file() else None)


def inspect_surface_runtime(runtime=None, freesurfer_home=None, license_file=None, *, python_bin=None,
                            hemisphere='both', atlases='aparc,aparc.a2009s', asset_root=None):
    """Verify in a fresh worker; never compile or download from a Check action.

    ``asset_root`` is the configured model asset folder (b2-os F4): an imported offline archive there makes the
    catalog runtime a download remedy, as for the on-demand check and the Run.
    """
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
        if backend == 'native' and not is_valid_surface_runtime_dir(home):
            state = surface_runtime_state(runtime, freesurfer_home, asset_root=asset_root)
            if state['remedy'] == 'download':
                # b2-os: nothing to probe; Prepare installs the catalog runtime (selected_setup needs build_required).
                raise SurfaceRuntimeMissing('No surface runtime is installed at %s' % home)
            if state['state'] == 'os_blocked':
                # b2-os: no runtime installed, and the catalog runtime needs a newer macOS (the probe would say "build").
                raise SurfaceRuntimeMacosBlocked(state['reason'])
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
        record = json.loads(result.stdout.strip().splitlines()[-1])
        if isinstance(record, dict):  # b2-os, decision 1: every record names its state
            record.setdefault('state', 'ready' if record.get('components_available') else 'missing')
        return record
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        activation = license_file or os.environ.get('FS_LICENSE')
        try:
            license_available = bool(activation and Path(activation).expanduser().is_file()
                                     and Path(activation).expanduser().stat().st_size)
        except OSError:
            license_available = False
        record = dict(components_available=False, license_available=license_available, available=False,
                      backend='native', build_required=False, error=str(exc))
        if isinstance(exc, (SurfaceRuntimeMacosBlocked, SurfaceRuntimeMissing)):
            # b2-os, decision 1: the structured state goes with the record to every user interface.
            try:
                state = surface_runtime_state(runtime, freesurfer_home, asset_root=asset_root)
            except Exception:
                state = None
            if state is not None:
                record.update({key: state[key] for key in ('state', 'remedy', 'macos_required', 'macos_current',
                                                           'download_bytes')})
                record['build_required'] = state['remedy'] == 'download'
                if state['state'] == 'os_blocked':
                    record.update(notes_zh=state['reason_zh'], install_hint=state['hint'],
                                  install_hint_zh=state['hint_zh'])
        if isinstance(exc, SurfaceRuntimeMacosBlocked):
            record.update(macos_blocked=True, notes=str(exc))
        record.setdefault('state', 'missing')
        return record
