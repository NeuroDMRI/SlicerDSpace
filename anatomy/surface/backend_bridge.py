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

from shared.external_process import external_python_environment, text_subprocess_kwargs

BACKEND_ROOT = Path(__file__).resolve().parent / 'ddsurfer_backend'


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def upstream_record():
    return json.loads((BACKEND_ROOT / 'UPSTREAM.json').read_text())


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


def backend_environment(extra=None):
    environment = external_python_environment()
    # Slicer ITK factories can crash SimpleITK from a different Python runtime.
    for key in ('ITK_AUTOLOAD_PATH', 'QT_PLUGIN_PATH'):
        environment.pop(key, None)
    if extra:
        environment.update({str(k): str(v) for k, v in extra.items() if v is not None})
    return environment


def run_backend(arguments, *, python_bin=None, env=None, log=None, log_path=None):
    command = [python_executable(python_bin), *map(str, arguments)]
    # PythonSlicer rebuilds its own runtime; external Python must not inherit it.
    environment = backend_environment(env)
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
                                   **text_subprocess_kwargs(bufsize=1))
    except BaseException:
        if stream:
            stream.close()
        raise
    try:
        for line in process.stdout:
            tail.append(line.rstrip())
            if stream:
                stream.write(line)
                stream.flush()
            if log:
                log(line.rstrip())
        code = process.wait()
    except BaseException:
        if os.name == 'posix':
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        elif process.poll() is None:
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
                process.kill()
            process.wait()
        raise
    finally:
        process.stdout.close()
        if stream:
            stream.close()
    if code:
        raise RuntimeError(f'DDSurfer backend failed ({code}):\n' + '\n'.join(tail))


def runtime_info(python_bin=None):
    command = [python_executable(python_bin), '-c',
        "import json,sys,torch,numpy,nibabel,SimpleITK,trimesh; "
        "print(json.dumps(dict(python=sys.version,executable=sys.executable,"
        "torch=torch.__version__,cuda=torch.version.cuda,numpy=numpy.__version__,"
        "nibabel=nibabel.__version__,SimpleITK=SimpleITK.Version_VersionString(),trimesh=trimesh.__version__)))"]
    completed = subprocess.run(command, cwd=str(BACKEND_ROOT), env=backend_environment(),
                               capture_output=True, timeout=60, **text_subprocess_kwargs())
    if completed.returncode:
        raise RuntimeError('DDSurfer Python runtime probe failed: ' + completed.stderr)
    return json.loads(completed.stdout.strip().splitlines()[-1])


def native_products(native_dir, subject, predict_mode):
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
            mesh.export(str(native_dir / f'{stem}_orig.stl'))
            lps_stl = native_dir / f'{stem}.stl'
            trimesh.Trimesh(vertices=vertices * [-1., -1., 1.], faces=faces,
                            process=False).export(str(lps_stl))
            products.append(lps_stl)
            checks[f'{hemi}.{kind}'] = dict(vertices=len(vertices), faces=len(faces),
                finite=True, watertight=bool(mesh.is_watertight), euler_number=int(mesh.euler_number),
                bounds_RAS_mm=mesh.bounds.tolist(), sha256=sha256(path),
                coordinate_space='native_scanner_RAS_mm', indexed_master=str(path))
    return products, checks
