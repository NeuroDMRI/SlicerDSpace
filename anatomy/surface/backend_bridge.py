"""Launch unchanged upstream code without polluting Slicer's import namespace."""
from __future__ import annotations

from collections import deque
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys

from shared.external_process import (
    external_python_environment,
    is_slicer_python,
    subprocess_no_window_kwargs,
    text_subprocess_kwargs,
)

BACKEND_ROOT = Path(__file__).resolve().parent / 'ddsurfer_backend'


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def upstream_record():
    return json.loads((BACKEND_ROOT / 'UPSTREAM.json').read_text(encoding='utf-8'))


def native_surface_hashes(native_dir, mode='whole'):
    hemis = ('lh',) if mode == 'left' else ('rh',) if mode == 'right' else ('lh', 'rh')
    return {f'{hemi}.{kind}.obj': sha256(Path(native_dir) / f'{hemi}.{kind}.obj')
            for hemi in hemis for kind in ('white', 'pial')}


def python_executable(explicit=None):
    configured = explicit or os.environ.get('DDSURFER_PYTHON_BIN')
    if configured:
        selected = Path(configured).expanduser()
        # The raw embedded interpreter needs the launcher to reconstruct its
        # Python home after Slicer's process environment has been isolated.
        for name in ('PythonSlicer', 'PythonSlicer.exe'):
            launcher = selected.parent / name
            if selected.stem.lower() == 'python-real' and launcher.is_file():
                return str(launcher)
        return str(selected)
    try:
        import slicer
        if hasattr(slicer, 'app'):
            from shared.pipeline import resolve_pythonslicer_executable
            return resolve_pythonslicer_executable()
    except ImportError:
        pass
    current = Path(sys.executable)
    for name in ('PythonSlicer', 'PythonSlicer.exe'):
        launcher = current.parent / name
        if current.stem.lower() == 'python-real' and launcher.is_file():
            return str(launcher)
    return str(current)


def _bundle_first_pythonpath(existing, bundled):
    norm = lambda part: os.path.normcase(os.path.normpath(part))
    first = {norm(part) for part in bundled}
    rest = [part for part in existing.split(os.pathsep) if part and norm(part) not in first]
    return os.pathsep.join([*bundled, *rest])


def backend_environment(extra=None, *, python_bin=None):
    environment = external_python_environment()
    # Slicer ITK factories can crash SimpleITK from a different Python runtime.
    for key in ('ITK_AUTOLOAD_PATH', 'QT_PLUGIN_PATH'):
        environment.pop(key, None)
    if extra:
        environment.update({str(k): str(v) for k, v in extra.items() if v is not None})
    resolved_py = python_executable(python_bin)
    if is_slicer_python(resolved_py):
        from shared.bundled_python import inject_bundled_python_env

        inject_bundled_python_env(environment, python_executable=resolved_py)
        # D1: inject_bundled_python_env appends. Put the bundle (it holds the NumPy that matches its torch) first.
        bundled = _bundled_numpy_site_dirs(python_bin)
        if bundled:
            environment['PYTHONPATH'] = _bundle_first_pythonpath(environment.get('PYTHONPATH', ''), bundled)
            environment['SLICERDSPACE_BUNDLED_SITE_PACKAGES'] = os.pathsep.join(bundled)
    if os.name == 'nt' and not any(environment.get(k) for k in ('LOGNAME', 'USER', 'LNAME', 'USERNAME')):
        environment['USERNAME'] = 'slicer'
    return environment


def _bundled_numpy_site_dirs(python_bin=None) -> list[str]:
    if not is_slicer_python(python_executable(python_bin)):
        return []
    from shared.bundled_python import bundled_site_packages_dirs

    return [str(d) for d in bundled_site_packages_dirs() if (d / 'numpy').is_dir()]


def run_backend(arguments, *, python_bin=None, env=None, log=None, log_path=None):
    sp_dirs = _bundled_numpy_site_dirs(python_bin)
    args_list = [str(item) for item in arguments]
    if sp_dirs and args_list and args_list[0].endswith('.py'):
        script_dir = str((BACKEND_ROOT / args_list[0]).resolve().parent)
        prepend = list(dict.fromkeys([str(BACKEND_ROOT), script_dir, *sp_dirs]))
        command = [
            python_executable(python_bin),
            '-c',
            # D1: the PythonSlicer launcher puts its own paths before PYTHONPATH, so the children that the
            # upstream script starts with sys.executable would get Slicer's NumPy first. Rewrite PYTHONPATH
            # here, after the launcher ran, so every child puts the bundle first.
            "import os, runpy, sys; "
            f"bundle = {sp_dirs!r}; "
            f"sys.path[:0] = {prepend!r}; "
            "os.environ['PYTHONPATH'] = os.pathsep.join(bundle + [p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p and p not in bundle]); "
            "os.environ['SLICERDSPACE_BUNDLED_SITE_PACKAGES'] = os.pathsep.join(bundle); "
            f"sys.argv = {args_list!r}; runpy.run_path(sys.argv[0], run_name='__main__')",
        ]
    else:
        command = [python_executable(python_bin), *args_list]
    # PythonSlicer rebuilds its own runtime; external Python must not inherit it.
    environment = backend_environment(env, python_bin=python_bin)
    environment['PYTHONUNBUFFERED'] = '1'
    if log:
        log('[DDSurfer] ' + ' '.join(command))
    tail = deque(maxlen=30)
    stream = None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stream = log_path.open('w', encoding='utf-8')
    # Own the backend group; upstream owns separate groups for its stage processes.
    try:
        process = subprocess.Popen(command, cwd=str(BACKEND_ROOT), env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   start_new_session=os.name == 'posix',
                                   **subprocess_no_window_kwargs(**text_subprocess_kwargs(bufsize=1)))
    except BaseException:
        if stream:
            stream.close()
        raise
    def on_line(line):
        tail.append(line.rstrip())
        if stream:
            stream.write(line)
            stream.flush()
        if log:
            log(line.rstrip())

    try:
        # H8: a surviving grandchild must not keep the stage open after the direct child has exited.
        from shared.line_stream import stream_process_lines
        code = stream_process_lines(process, on_line,
                                    notice=(lambda message: log('[DDSurfer] ' + message)) if log else None)
    except BaseException:
        if os.name == 'posix':
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        elif process.poll() is None:
            try:
                subprocess.run(
                    ['taskkill', '/F', '/T', '/PID', str(process.pid)],
                    check=False,
                    capture_output=True,
                    **subprocess_no_window_kwargs(),
                )
            except OSError:
                process.terminate()
        try:
            # Upstream needs up to 10 seconds to reap an interrupted stage tree.
            process.wait(timeout=12)
        except subprocess.TimeoutExpired:
            if os.name == 'posix':
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                try:
                    subprocess.run(
                        ['taskkill', '/F', '/T', '/PID', str(process.pid)],
                        check=False,
                        capture_output=True,
                        **subprocess_no_window_kwargs(),
                    )
                except OSError:
                    pass
                if process.poll() is None:
                    process.kill()
            process.wait()
        raise
    finally:
        # stream_process_lines closes the pipe itself once its reader has finished.
        if stream:
            stream.close()
    if code:
        raise RuntimeError(f'DDSurfer backend failed ({code}):\n' + '\n'.join(tail))


def runtime_info(python_bin=None):
    sp_dirs = _bundled_numpy_site_dirs(python_bin)
    prefix = f"sys.path[:0] = {sp_dirs!r}; " if sp_dirs else ""
    command = [python_executable(python_bin), '-c',
        "import json,sys; "
        + prefix
        + "import torch,numpy,nibabel,SimpleITK,trimesh; "
        "print(json.dumps(dict(python=sys.version,executable=sys.executable,"
        "torch=torch.__version__,cuda=torch.version.cuda,numpy=numpy.__version__,"
        "nibabel=nibabel.__version__,SimpleITK=SimpleITK.Version_VersionString(),trimesh=trimesh.__version__)))"]
    completed = subprocess.run(command, cwd=str(BACKEND_ROOT), env=backend_environment(python_bin=python_bin),
                               capture_output=True, timeout=60, **subprocess_no_window_kwargs(**text_subprocess_kwargs()))
    if completed.returncode:
        raise RuntimeError('DDSurfer Python runtime probe failed: ' + completed.stderr)
    return json.loads(completed.stdout.strip().splitlines()[-1])


def native_products(native_dir, subject, predict_mode, *, export_stl: bool = True):
    """Keep indexed RAS OBJ masters and generate historical Slicer LPS STL."""
    import numpy as np
    import trimesh
    products, checks = [], {}
    for hemi in ('lh', 'rh'):
        white_faces = None
        for kind in ('white', 'pial') if predict_mode == 'all' else ('white',):
            path = native_dir / f'{hemi}.{kind}.obj'
            mesh = trimesh.load(str(path), process=False)
            vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
            if not len(vertices) or not len(faces) or not np.isfinite(vertices).all():
                raise ValueError(f'Invalid native surface: {path}')
            if kind == 'white':
                white_faces = faces.copy()
            elif not np.array_equal(white_faces, faces):
                raise ValueError(f'White/pial vertex correspondence lost: {path}')
            token = 'wm' if kind == 'white' else 'pial'
            stem = f'{subject}_predicted_{token}_{hemi}'
            ras_stl = native_dir / f'{stem}_orig.stl'
            lps_stl = native_dir / f'{stem}.stl'
            if export_stl or not ras_stl.is_file():
                mesh.export(str(ras_stl))
            if export_stl or not lps_stl.is_file():
                trimesh.Trimesh(vertices=vertices * [-1., -1., 1.], faces=faces,
                                process=False).export(str(lps_stl))
            products.append(lps_stl)
            checks[f'{hemi}.{kind}'] = dict(vertices=len(vertices), faces=len(faces),
                finite=True, watertight=bool(mesh.is_watertight), euler_number=int(mesh.euler_number),
                bounds_RAS_mm=mesh.bounds.tolist(), sha256=sha256(path),
                coordinate_space='native_scanner_RAS_mm', indexed_master=str(path))
    return products, checks
