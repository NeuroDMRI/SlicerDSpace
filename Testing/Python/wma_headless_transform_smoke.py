"""Recover archived WMA clusters in a new directory and verify native geometry.

This exercises the real Slicer/WMA hardening and bundle append steps. Reused
registration/clustering products make it a recovery check, not a timing sample.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run(args):
    import numpy as np
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    from shared.external_process import external_python_environment
    from tractography.wma import (
        _command_path, _resolve_slicer_runtime, _resolve_slicer_launcher,
        _slicerdmri_module_paths, _slicer_runtime_library_paths, _use_virtual_x,
        summarize_wma_outputs, _wma_output_contract_error,
    )

    if os.environ.get('DISPLAY') or os.environ.get('SLICERDSPACE_DISPLAY'):
        raise ValueError('Run with no DISPLAY to test automatic headless execution')
    assert _use_virtual_x()
    case = args.archived_case.resolve()
    name = case.name
    clusters = case / 'FiberClustering/OutlierRemovedClusters' / (name + '_reg_outlier_removed')
    sources = sorted(clusters.glob('*.vtp'))
    if len(sources) != 800:
        raise ValueError('Expected all 800 archived clusters')
    transform = case / 'TractRegistration' / name / 'output_tractography' / ('itk_txform_' + name + '.tfm')
    before = {str(p): digest(p) for p in [*sources, transform]}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.verify_existing)
    shim_dir = output / 'launchers'
    shim_dir.mkdir(exist_ok=args.verify_existing)
    wma = args.wma_root.resolve()
    runtime = _resolve_slicer_runtime(args.slicer)
    launcher = _resolve_slicer_launcher(args.slicer.resolve())
    from shared.external_process import slicer_runtime_roots
    python = next(path for root in slicer_runtime_roots(runtime.parent)
                  for path in (root / 'PythonSlicer', root / 'bin/PythonSlicer') if path.is_file())
    if not (shim_dir / 'python').exists():
        (shim_dir / 'python').symlink_to(python)
    harden = shim_dir / 'harden_transform_with_slicer.py'
    harden.write_text('import sys\n' + f'sys.path.insert(0, {str(ROOT / "scripts")!r})\n'
                      'import run_wma_harden_transform\n' +
                      f'run_wma_harden_transform.run({str(wma / "bin/harden_transform_with_slicer.py")!r})\n')
    harden.chmod(0o755)
    module_args = ' '.join('--additional-module-paths ' + shlex.quote(str(p))
                           for p in _slicerdmri_module_paths(runtime))
    slicer = shim_dir / 'Slicer'
    slicer.write_text(f'#!/bin/sh\nexec {shlex.quote(str(launcher))} {module_args} "$@"\n')
    slicer.chmod(0o755)
    xvfb = _command_path('xvfb-run', wma)
    if xvfb is None:
        raise RuntimeError('xvfb-run is required for this Linux headless check')
    env = external_python_environment()
    env['PATH'] = os.pathsep.join([str(shim_dir), str(wma / 'bin'), str(wma / 'utilities'),
                                  str(xvfb.parent), env.get('PATH', os.defpath)])
    env['PYTHONPATH'] = str(wma)
    env['LD_LIBRARY_PATH'] = os.pathsep.join([*(str(p) for p in _slicer_runtime_library_paths(runtime)),
                                            env.get('LD_LIBRARY_PATH', '')])
    env.update({key: '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                    'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS')})
    transformed = output / name / 'FiberClustering/TransformedClusters' / name
    separated = output / name / 'FiberClustering/SeparatedClusters'
    anatomical = output / name / 'AnatomicalTracts'
    commands = [
        [str(xvfb), '-a', str(wma / 'bin/wm_harden_transform.py'), '-i', '-t', str(transform),
         str(clusters), str(transformed), str(slicer)],
        [str(wma / 'bin/wm_separate_clusters_by_hemisphere.py'), str(transformed), str(separated)],
        [str(wma / 'bin/wm_append_clusters_to_anatomical_tracts.py'), str(separated),
         str(args.atlas_dir.resolve() / 'ORG-800FC-100HCP'), str(anatomical)],
    ]
    started = time.monotonic()
    receipt = dict(status='running', scope='Archived real-subject WMA recovery; not cohort timing',
                   source_case=str(case), commands=commands, source_sha256=before)
    def save():
        (output / 'verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if args.verify_existing:
        previous = json.loads((output / 'verification.json').read_text())
        if previous['source_sha256'] != before:
            raise ValueError('Sources differ from the recorded recovery attempt')
        receipt['previous_verification'] = previous
    save()
    try:
        for index, command in enumerate([] if args.verify_existing else commands):
            with (output / f'step_{index}.log').open('w') as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True,
                               timeout=1800)
            if index == 0 and len(list(transformed.glob('*.vtp'))) != 800:
                raise ValueError('Slicer did not write all 800 transformed clusters; inspect its log.txt')
        # Slicer loads ITK's from-parent transform; WMA then inverts the node.
        # Together these give the file's forward affine, converted LPS -> RAS.
        fields = dict(line.split(':', 1) for line in transform.read_text().splitlines() if ':' in line)
        if fields['Transform'].strip() != 'AffineTransform_double_3_3':
            raise ValueError('This geometry check requires an ITK affine transform')
        parameters = np.fromstring(fields['Parameters'], sep=' ')
        centre = np.fromstring(fields['FixedParameters'], sep=' ')
        affine = np.eye(4)
        affine[:3, :3] = parameters[:9].reshape(3, 3)
        affine[:3, 3] = parameters[9:] + centre - affine[:3, :3] @ centre
        lps_ras = np.diag([-1., -1., 1., 1.])
        expected = lps_ras @ affine @ lps_ras
        if len(list(transformed.glob('*.vtp'))) != 800:
            raise ValueError('Expected all 800 transformed clusters')
        max_error = 0.
        total_lines = 0
        for source in sources:
            readers = [vtk.vtkXMLPolyDataReader(), vtk.vtkXMLPolyDataReader()]
            for reader, path in zip(readers, (source, transformed / source.name)):
                reader.SetFileName(str(path)); reader.Update()
            src, dst = [reader.GetOutput() for reader in readers]
            assert src.GetNumberOfPoints() == dst.GetNumberOfPoints(), source.name
            assert src.GetNumberOfLines() == dst.GetNumberOfLines(), source.name
            total_lines += src.GetNumberOfLines()
            assert np.array_equal(vtk_to_numpy(src.GetLines().GetData()),
                                  vtk_to_numpy(dst.GetLines().GetData())), source.name
            if src.GetNumberOfPoints():
                xyz = vtk_to_numpy(src.GetPoints().GetData()).astype(np.float64)
                actual = vtk_to_numpy(dst.GetPoints().GetData()).astype(np.float64)
                error = float(np.max(np.abs(xyz @ expected[:3, :3].T + expected[:3, 3] - actual)))
                max_error = max(max_error, error)
                assert error < 5e-4, (source.name, error)
        summary = summarize_wma_outputs(output, Path(name + '.vtk'))
        contract_error = _wma_output_contract_error(summary)
        assert contract_error is None, contract_error
        assert all(digest(Path(p)) == value for p, value in before.items()), 'Source products changed'
        receipt.update(status='passed', virtual_x_automatic=True, transformed_clusters=800,
                       source_streamlines=total_lines, maximum_coordinate_error_mm=max_error,
                       sources_unchanged=True, output_summary=summary,
                       implementation_sha256=digest(ROOT / 'tractography/wma.py'))
    except Exception as error:
        receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        elapsed = time.monotonic() - started
        if args.verify_existing:
            receipt['verification_seconds'] = elapsed
            receipt['recovery_runtime_seconds'] = previous['recovery_runtime_seconds']
        else:
            receipt['recovery_runtime_seconds'] = elapsed
        save()
    print(json.dumps({key: receipt[key] for key in ('status', 'transformed_clusters',
          'maximum_coordinate_error_mm', 'sources_unchanged', 'recovery_runtime_seconds')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archived-case', type=Path, required=True)
    parser.add_argument('--wma-root', type=Path, required=True)
    parser.add_argument('--atlas-dir', type=Path, required=True)
    parser.add_argument('--slicer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-existing', action='store_true',
                        help='Recheck recorded recovery products without rerunning scientific commands')
    run(parser.parse_args())
