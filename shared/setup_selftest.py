"""Small installation checks in the same interpreters used by the selected algorithms."""

import importlib
import json
from pathlib import Path
import tempfile
import time


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
    return list(groups.values())


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
    extra = {} if teem_report is None else dict(teem_self_check=teem_report)
    return dict(kind='slicerdspace_setup_selftest', status='passed', checks=checks, **extra,
                elapsed_seconds=time.monotonic() - started,
                scope='Installation functionality using synthetic inputs; no patient processing or anatomical QC')
