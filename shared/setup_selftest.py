"""Small installation checks in the same interpreters used by the selected algorithms."""

import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time

MRTRIX_VERSION = '3.0.8'
TOOL_TIMEOUT_SECONDS = 120


def receipt_warnings(receipt_paths):
    """A2: the warnings of passed self-check receipts (GUI and CLI show them; they do not block readiness)."""
    found = []
    for path in receipt_paths:
        try:
            found.extend(str(text) for text in json.loads(Path(path).read_text()).get('warnings') or ())
        except (OSError, ValueError, AttributeError):
            continue
    return found


def requests_for_report(report, *, asset_root=None, repo_root=None, backend_paths=None, device_choices=None):
    groups = {}
    device_choices = device_choices or {}
    target_devices = {'segmentation': device_choices.get('ddseg_device', 'cpu'),
                      'parcellation': device_choices.get('ddseg_device', 'cpu'),
                      'surface': device_choices.get('surface_device', 'cpu'),
                      'tractcloud': device_choices.get('tractcloud_device', 'cpu')}
    for item in getattr(report, 'statuses', ()):
        if item.kind == 'device' and item.required and item.status == 'available':
            for target in getattr(item, 'targets', ()):
                target_devices[target] = (item.details or {}).get('resolved_device') or (item.details or {}).get('requested_device', 'cpu')
    for item in getattr(report, 'statuses', ()):
        if not item.required or item.status not in {'available', 'ok', 'passed'} or item.kind != 'python':
            continue
        details = item.details or {}
        name = details.get('import')
        if not name or name.startswith('slicer'):
            continue
        python = details.get('python_executable')
        group = groups.setdefault(python, dict(python=python, imports=[], pythonpath=[], models=[]))
        group['imports'].append(name)
        group['pythonpath'].extend(details.get('pythonpath', ()))
    for item in getattr(report, 'statuses', ()):
        # Teem is Slicer's library, so only the interpreter that loaded it gets the Teem self-check.
        if getattr(item, 'key', '') == 'native_tensor_fit' and item.required and item.status == 'available':
            python = (item.details or {}).get('python_executable')
            group = groups.setdefault(python, dict(python=python, imports=[], pythonpath=[], models=[]))
            group['teem_self_check'] = True
    backend_paths = backend_paths or {}
    if 'wma' in getattr(report, 'targets', ()) and backend_paths.get('wma_root'):
        wma_python = next(((item.details or {}).get('python_executable') for item in report.statuses
                           if getattr(item, 'key', '').startswith('python:wma:')), None)
        if wma_python in groups:
            groups[wma_python]['imports'].append('whitematteranalysis')
            groups[wma_python]['pythonpath'].append(str(backend_paths['wma_root']))
    for group in groups.values():
        group['imports'] = sorted(set(group['imports']))
        group['pythonpath'] = sorted(set(group['pythonpath']))
        group['devices'] = sorted({(item.details or {}).get('resolved_device') or
                                   (item.details or {}).get('requested_device', 'cpu')
                                   for item in getattr(report, 'statuses', ())
                                   if item.required and item.status == 'available' and item.kind == 'device'
                                   and (item.details or {}).get('python_executable') == group['python']})
        if not group['devices']:
            group['devices'] = ['cpu']
    if groups and repo_root is not None:
        from shared.model_assets import model_asset_path_or_none
        manifest = json.loads((Path(repo_root) / 'models/assets_manifest.json').read_text())
        torch_groups = [g for g in groups.values() if 'torch' in g['imports']]
        if torch_groups:
            for profile in ('segmentation', 'parcellation', 'surface'):
                if profile not in report.targets:
                    continue
                owner = next(((item.details or {}).get('python_executable') for item in report.statuses
                              if profile in getattr(item, 'targets', ()) and item.kind == 'python'
                              and (item.details or {}).get('import') == 'torch'), None)
                torch_group = groups.get(owner) if owner else torch_groups[0] if len(torch_groups) == 1 else None
                if torch_group is None:
                    raise ValueError('Cannot identify the selected model interpreter: ' + profile)
                for name in manifest['profiles'][profile]['required_paths']:
                    if Path(name).suffix not in {'.pt', '.pkl', '.pth'}:
                        continue
                    path, error = model_asset_path_or_none(name, repo_root=Path(repo_root), explicit_root=asset_root)
                    if path is None:
                        raise FileNotFoundError(error or 'Selected model is unavailable: ' + name)
                    torch_group['models'].append(dict(path=str(path), sha256=manifest['asset_records'][name]['sha256'],
                                                      torchscript=profile == 'segmentation', device=target_devices[profile]))
        tract_group = next((g for g in groups.values() if 'tractcloud' in g['imports']), None)
        if tract_group is not None:
            from shared.optional_assets import resolve_tractcloud_data_directory
            data = resolve_tractcloud_data_directory(backend_paths.get('tractcloud_data_dir'), asset_root=asset_root)
            if data is None:
                raise FileNotFoundError('Selected TractCloud resources are unavailable')
            from shared.backend_sources import file_hash
            paths = [data / 'TrainedModel/best_tract_f1_model.pth', data / 'TrainedModel/cli_args.txt',
                     data / 'TrainData_800clu800ol/HCP_mass_center.npy']
            tract_group['tractcloud_resources'] = [{'path': str(path), 'sha256': file_hash(path)} for path in paths]
            tract_group['tractcloud_device'] = target_devices['tractcloud']
    _add_tool_requests(report, groups, backend_paths)
    return list(groups.values())


_AVAILABLE = {'available', 'ok', 'passed'}


def _tool_bin_dir(report, install_group):
    """Directory of the first required, available command of an install group ('mrtrix' or 'fsl'), or None."""
    for item in getattr(report, 'statuses', ()):
        if item.kind == 'command' and item.required and item.status in _AVAILABLE \
                and getattr(item, 'install_group', '') == install_group:
            resolved = (item.details or {}).get('resolved')
            if resolved:
                return str(Path(resolved).parent)
    return None


def _dcm2niix_executable(report, backend_paths):
    """The dcm2niix that DICOM import would use (G1).

    An explicit report field or path setting comes first. Otherwise the pip package in this Python
    (`input_import.packaged_dcm2niix()`), then PATH (`find_dcm2niix()` does both). None: no check (DWIConvert is the fallback).
    """
    for item in getattr(report, 'statuses', ()):
        if getattr(item, 'key', '') == 'command:dcm2niix' and item.required and item.status in _AVAILABLE:
            resolved = (item.details or {}).get('resolved')
            if resolved:
                return str(resolved)
    explicit = next((str(backend_paths[name]) for name in ('packaged_dcm2niix', 'dcm2niix_executable')
                     if backend_paths.get(name)), None)
    if explicit:
        return explicit
    from shared import input_import
    return input_import.find_dcm2niix()


def _same_executable(first, second):
    if not first or not second:
        return False
    try:
        return os.path.samefile(first, second)
    except OSError:
        return os.path.abspath(str(first)) == os.path.abspath(str(second))


def _plugin_dcm2niix(executable, backend_paths):
    """A2: True when ``executable`` is the pinned wheel that the plugin installed in a Python (not one found on PATH)."""
    from shared import input_import
    if backend_paths.get('packaged_dcm2niix') and _same_executable(executable, backend_paths['packaged_dcm2niix']):
        return True
    return _same_executable(executable, input_import.packaged_dcm2niix())


def _add_tool_requests(report, groups, backend_paths):
    """R-B3: MRtrix, FSL, dcm2niix and the PNL native mask checks for what the report selects and has a path for.

    They go into the request of Slicer's Python (the managed `python` shim and the PNL check need it).
    """
    from shared.external_process import is_slicer_python
    wanted = {}
    mrtrix, fsl = _tool_bin_dir(report, 'mrtrix'), _tool_bin_dir(report, 'fsl')
    if not mrtrix:
        # G3: MRtrix used only as the brain-mask backend has no command entry; the setting carries the dwi2mask path.
        for item in getattr(report, 'statuses', ()):
            details = item.details or {}
            dwi2mask = (details.get('tool_paths') or {}).get('dwi2mask')
            if getattr(item, 'key', '') == 'brain_mask_backend' and item.required and item.status in _AVAILABLE \
                    and details.get('backend') == 'mrtrix' and dwi2mask:
                mrtrix = str(Path(dwi2mask).parent)
    if mrtrix:
        wanted['mrtrix_bin_dir'] = mrtrix
    if fsl:
        wanted['fsl_bin_dir'] = fsl
    dcm2niix = _dcm2niix_executable(report, backend_paths)
    if dcm2niix:
        wanted['dcm2niix'] = dcm2niix
        wanted['dcm2niix_installed_by_plugin'] = _plugin_dcm2niix(dcm2niix, backend_paths)
    slicer_hint = None
    for item in getattr(report, 'statuses', ()):
        details = item.details or {}
        if getattr(item, 'key', '') == 'brain_mask_backend' and item.required and item.status in _AVAILABLE \
                and details.get('backend') == 'pnl_cnn' and (details.get('tool_paths') or {}).get('model_dir') \
                and (details.get('settings') or {}).get('pnl_route') == 'native-sitk':
            # SC-1d: the native check runs only for the native route; the managed route keeps its runtime check.
            wanted['pnl_native_mask'] = {'model_dir': details['tool_paths']['model_dir'],
                                         'device': (details.get('settings') or {}).get('device', 'auto')}
            slicer_hint = (details.get('tool_paths') or {}).get('pythonslicer')
    if not wanted:
        return
    group = next((g for python, g in groups.items() if python and is_slicer_python(python)), None)
    if group is None:
        # python=None: the interpreter that ran the environment check (Slicer's Python in the GUI).
        group = groups.setdefault(slicer_hint, dict(python=slicer_hint, imports=[], pythonpath=[], models=[], devices=['cpu']))
    group.update(wanted)


def _run_tool(argv, env):
    try:
        return subprocess.run([str(item) for item in argv], capture_output=True, text=True,
                              timeout=TOOL_TIMEOUT_SECONDS, env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f'{Path(str(argv[0])).name} could not run: {type(exc).__name__}: {exc}') from exc


def _output(result):
    return (result.stdout or '') + (result.stderr or '')


def is_managed_prefix(bin_dir):
    """True when ``bin_dir`` lies inside the plugin runtime root (a prefix this plugin installed)."""
    from shared.managed_runtimes import runtime_root
    root, path = runtime_root(), Path(bin_dir).expanduser().resolve()
    return path == root or root in path.parents


def check_mrtrix(bin_dir, fsl_bin_dir=None, *, shim_root=None):
    """MRtrix3 through the B3 environment: the version, and two scripts that need the managed `python` shim.

    R-B4: the plugin-managed prefix must report exactly MRTRIX_VERSION. A path the user set may be any MRtrix3 release.
    R-B1: MRtrix scripts take a single dash (`-help`); measured 2026-11-07, `dwi2response --help` exits 1.
    """
    from shared.external_process import managed_tool_environment
    env = managed_tool_environment(mrtrix_bin_dir=bin_dir, fsl_bin_dir=fsl_bin_dir, shim_root=shim_root)
    checks = []
    version = _run_tool([Path(bin_dir) / 'mrconvert', '--version'], env)
    text = _output(version)
    if is_managed_prefix(bin_dir):
        reports = re.search(r'(?<![\d.])' + re.escape(MRTRIX_VERSION) + r'(?![\d.])', text)
        expected = MRTRIX_VERSION
    else:
        reports = re.search(r'mrconvert\s+3\.\d', text)
        expected = '3.x'
    if version.returncode != 0 or not reports:
        raise RuntimeError(f'mrconvert --version does not report {expected}: {text.strip()[:200]}')
    checks.append('mrtrix:mrconvert_version')
    for name in ('dwi2response', 'dwifslpreproc'):
        result = _run_tool([Path(bin_dir) / name, '-help'], env)
        if result.returncode != 0 or name not in _output(result):
            raise RuntimeError(f'{name} -help failed (exit {result.returncode}): {_output(result).strip()[:200]}')
        checks.append(f'mrtrix:{name}_help')
    return checks


def check_fsl(fsl_bin_dir, mrtrix_bin_dir=None, *, shim_root=None):
    """FSL eddy and topup (and eddy_cpu / eddy_openmp) with FSLDIR, FSLOUTPUTTYPE and the FSL bin on PATH.

    G2: a plugin-managed prefix must have eddy, eddy_cpu and topup. Any other FSL needs eddy and topup, what the
    doctor requires; eddy_cpu (FSL 6.0.7+) and eddy_openmp (older FSL) are checked only when present.
    The tools print their usage and exit with 1 for --help; exit 127 means the FSL Python was not found.
    """
    from shared.external_process import managed_tool_environment
    env = managed_tool_environment(mrtrix_bin_dir=mrtrix_bin_dir, fsl_bin_dir=fsl_bin_dir, shim_root=shim_root)
    checks = []
    managed = is_managed_prefix(fsl_bin_dir)
    required = {'eddy', 'topup'} | ({'eddy_cpu'} if managed else set())
    for name in ('eddy', 'eddy_cpu', 'eddy_openmp', 'topup'):
        if name not in required and (managed or not (Path(fsl_bin_dir) / name).exists()):
            continue
        result = _run_tool([Path(fsl_bin_dir) / name, '--help'], env)
        if result.returncode != 1 or 'usage' not in _output(result).lower():
            raise RuntimeError(f'{name} --help should print its usage and exit with 1; got exit {result.returncode}: '
                               f'{_output(result).strip()[:200]}')
        checks.append(f'fsl:{name}_help')
    return checks


def check_dcm2niix(executable):
    # R-B2: measured 2026-11-07 (wheel v1.0.20261029): `dcm2niix -v` prints the banner and exits 3
    # (main_console.cpp kEXIT_REPORT_VERSION). Exit 0 is accepted as well.
    result = _run_tool([executable, '-v'], None)
    text = _output(result).lower()
    if result.returncode not in (0, 3) or 'dcm2niix' not in text or 'version' not in text:
        raise RuntimeError(f'dcm2niix -v did not report a version (exit {result.returncode}): {_output(result).strip()[:200]}')
    return ['dcm2niix:version']


def check_torch_convolution(torch, device):
    """One small convolution on the selected device: all-ones 4x4 input and 3x3 kernel give four values of 9."""
    out = torch.nn.functional.conv2d(torch.ones((1, 1, 4, 4), device=device), torch.ones((1, 1, 3, 3), device=device))
    if tuple(out.shape) != (1, 1, 2, 2) or not torch.equal(out, torch.full_like(out, 9)):
        raise RuntimeError('PyTorch convolution returned a wrong result on ' + device)
    return 'torch:convolution:' + device


def native_mask_module_present():
    return Path(__file__).with_name('pnl_mask_native.py').is_file()


def check_native_mask(spec, *, torch=None, inference=None, weights=None):
    """PNL native mask: the three views load and one 256x256 slice per view runs on the selected device.

    Returns the check names. R-B5: a requested check with `shared/pnl_mask_native.py` absent is a failure.
    It uses the W-A API `shared.pnl_cnn.weights.load_model(dir, view, device)` and `inference.select_device`.
    """
    if not native_mask_module_present():
        raise RuntimeError('PNL native mask is selected but shared/pnl_mask_native.py is not present in this install')
    if torch is None:
        import torch
    if inference is None or weights is None:
        from shared.pnl_cnn import inference, weights
    device = inference.select_device(spec.get('device', 'auto'))['resolved']
    checks = []
    with torch.inference_mode():
        for view in inference.VIEWS:
            model = weights.load_model(spec['model_dir'], view, device)
            output = model(torch.zeros((1, 1, 256, 256), dtype=torch.float32, device=device), segmentation_only=True)
            if tuple(output.shape[-2:]) != (256, 256) or not bool(torch.isfinite(output).all()):
                raise RuntimeError(f'PNL {view} view gave a wrong or nonfinite slice on {device}')
            checks.append(f'pnl_native_mask:{view}:{device}')
    return checks


def run_checks(request):
    import sys
    for path in request.get('pythonpath', ()):
        sys.path.insert(0, path)
    checks, started = [], time.monotonic()
    for name in request.get('imports', ()):
        importlib.import_module(name)
        checks.append('import:' + name)
    teem_report = None
    if request.get('teem_self_check'):
        from shared import teem_dti
        teem_report = teem_dti.self_check()  # raises TeemError (reported as a failed check) when Teem is unusable
        checks.append('teem:self_check')
    names = set(request.get('imports', ()))
    with tempfile.TemporaryDirectory(prefix='slicerdspace-selftest-') as temporary:
        folder = Path(temporary)
        if 'numpy' in names:
            import numpy as np
            result = np.linalg.solve(np.array([[2., 0.], [0., 4.]]), [4., 12.])
            np.testing.assert_array_equal(result, [2., 3.])
            checks.append('numpy:linear_solve')
        if 'nibabel' in names:
            import nibabel as nib
            import numpy as np
            array = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
            target = folder / 'volume.nii.gz'
            nib.save(nib.Nifti1Image(array, np.eye(4)), target)
            image = nib.load(target)
            np.testing.assert_array_equal(np.asarray(image.dataobj), array)
            np.testing.assert_array_equal(image.affine, np.eye(4))
            checks.append('nifti:write_read')
        if 'vtk' in names:
            import vtk
            points = vtk.vtkPoints()
            points.InsertNextPoint(0, 0, 0)
            points.InsertNextPoint(1, 2, 3)
            lines = vtk.vtkCellArray()
            lines.InsertNextCell(2)
            lines.InsertCellPoint(0)
            lines.InsertCellPoint(1)
            poly = vtk.vtkPolyData()
            poly.SetPoints(points)
            poly.SetLines(lines)
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetFileName(str(folder / 'tracks.vtp'))
            writer.SetInputData(poly)
            if writer.Write() != 1:
                raise RuntimeError('VTK output could not be written')
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(writer.GetFileName())
            reader.Update()
            if reader.GetOutput().GetNumberOfLines() != 1 or reader.GetOutput().GetPoint(1) != (1., 2., 3.):
                raise RuntimeError('VTK streamline round-trip changed its geometry')
            checks.append('vtk:streamline_write_read')
        if 'torch' in names:
            import torch
            torch.set_num_threads(2)
            devices = request.get('devices') or [request.get('device', 'auto')]
            for device in devices:
                if device == 'auto':
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                tensor = torch.ones((2, 2), device=device)
                if not torch.equal(tensor @ tensor, tensor * 2):
                    raise RuntimeError('PyTorch tensor operation failed')
                checks.append('torch:tensor_operation:' + device)
                checks.append(check_torch_convolution(torch, device))
            from shared.backend_sources import file_hash
            for model in request.get('models', ()):
                path = Path(model['path'])
                if file_hash(path) != model['sha256']:
                    raise ValueError('Model checksum changed before loading: ' + path.name)
                if model['torchscript']:
                    model_device = model.get('device', device)
                    if model_device == 'auto':
                        model_device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    module = torch.jit.load(str(path), map_location=model_device).eval()
                    with torch.inference_mode():
                        output = module(torch.zeros((1, 5, 256, 256), device=model_device))
                    if not torch.isfinite(output).all():
                        raise RuntimeError('Model produced nonfinite synthetic output: ' + path.name)
                    del module, output
                else:
                    state = torch.load(str(path), map_location='cpu', weights_only=True)
                    if not isinstance(state, dict) or not state:
                        raise ValueError('Model checkpoint has no state: ' + path.name)
                    del state
                checks.append('model:load:' + path.name)
            if request.get('tractcloud_resources'):
                import numpy as np
                from tractcloud.inference import load_model
                paths = [Path(item['path']) for item in request['tractcloud_resources']]
                for path, item in zip(paths, request['tractcloud_resources']):
                    if file_hash(path) != item['sha256']:
                        raise ValueError('TractCloud resource changed before loading: ' + path.name)
                device = request.get('tractcloud_device', device)
                if device == 'auto':
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                model, options = load_model(str(paths[0]), str(paths[1]), torch.device(device))
                center = np.load(paths[2], allow_pickle=False)
                if center.size == 0 or not np.isfinite(center).all():
                    raise ValueError('TractCloud reference has no finite coordinates')
                count = int(options.get('num_point_per_fiber', 15))
                neighbours = int(options.get('k', 20)) + int(options.get('k_global', 80))
                with torch.inference_mode():
                    output = model(torch.zeros((1, 3, count), device=device),
                                   torch.zeros((1, 3, count, neighbours), device=device))
                if not torch.isfinite(output).all():
                    raise RuntimeError('TractCloud produced nonfinite synthetic output')
                checks.append('tractcloud:load_and_synthetic_forward')
    if request.get('mrtrix_bin_dir'):
        checks.extend(check_mrtrix(request['mrtrix_bin_dir'], request.get('fsl_bin_dir')))
    if request.get('fsl_bin_dir'):
        checks.extend(check_fsl(request['fsl_bin_dir'], request.get('mrtrix_bin_dir')))
    warnings = []
    if request.get('dcm2niix'):
        # A2: only the plugin's own wheel blocks "ready". A dcm2niix found on PATH is a warning: NIfTI input never needs it
        # and DICOM import tries the wheel first.
        try:
            checks.extend(check_dcm2niix(request['dcm2niix']))
        except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
            from shared import input_import
            if request.get('dcm2niix_installed_by_plugin') or _same_executable(request['dcm2niix'], input_import.packaged_dcm2niix()):
                raise
            warnings.append(f"dcm2niix at {request['dcm2niix']} was not installed by this plugin and failed its check ({exc}). "
                            'NIfTI input does not need it; DICOM import tries the plugin wheel first.')
    if request.get('pnl_native_mask'):
        checks.extend(check_native_mask(request['pnl_native_mask']))
    extra = {} if teem_report is None else dict(teem_self_check=teem_report)
    if warnings:
        extra['warnings'] = warnings
    return dict(kind='slicerdspace_setup_selftest', status='passed', checks=checks, **extra,
                elapsed_seconds=time.monotonic() - started,
                scope='Installation functionality using synthetic inputs; no patient processing or anatomical QC')
