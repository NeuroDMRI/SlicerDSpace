#!/usr/bin/env python3
"""Validate shared native preparation against unchanged author CLI calls on real ROIs.

This is workflow reuse, not an optimization of tensor estimation. Never writes
to source subjects or extrapolates cropped-input timings to full-brain speed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

import nibabel as nib
import numpy as np
import nrrd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared import pipeline
from shared.external_process import merged_startup_environment


def compare(first, second):
    a, ha = nrrd.read(str(first))
    b, hb = nrrd.read(str(second))
    return dict(exact_values=bool(np.array_equal(a, b, equal_nan=True)),
                max_abs_difference=float(np.nanmax(np.abs(a.astype(float) - b.astype(float)))),
                geometry_equal=all(np.array_equal(np.asarray(ha[key]), np.asarray(hb[key]), equal_nan=True)
                    for key in ('space directions', 'space origin')))


def run_case(args, source):
    source = Path(source).resolve()
    subject = source.name
    output = args.output / subject
    output.mkdir(parents=True, exist_ok=False)
    folder = output / 'input'
    folder.mkdir()
    raw = source / 'input'
    original = {name: pipeline._cache_file_signature(raw / filename) for name, filename in
                (('dwi', 'data.nii.gz'), ('bval', 'bvals'), ('bvec', 'bvecs'), ('mask', 'nodif_brain_mask.nii.gz'))}
    image, mask = nib.load(raw / 'data.nii.gz'), nib.load(raw / 'nodif_brain_mask.nii.gz')
    mask_data = np.asarray(mask.dataobj)
    center = np.median(np.argwhere(mask_data > 0), axis=0).astype(int)
    start = np.clip(center - args.size // 2, 0, np.array(mask.shape) - args.size)
    slices = tuple(slice(int(i), int(i + args.size)) for i in start)
    affine = image.affine.copy()
    affine[:3, 3] = nib.affines.apply_affine(affine, start)
    data = np.asanyarray(image.dataobj[slices + (slice(None),)])
    nib.save(nib.Nifti1Image(data, affine), folder / 'data.nii.gz')
    nib.save(nib.Nifti1Image(mask_data[slices].astype(np.uint8), affine), folder / 'nodif_brain_mask.nii.gz')
    for name in ('bvals', 'bvecs'):
        shutil.copy2(raw / name, folder / name)
    bundle = pipeline.RawInputBundle(output, subject, folder, folder / 'data.nii.gz', folder / 'bvals',
                                    folder / 'bvecs', folder / 'nodif_brain_mask.nii.gz')
    cli = args.slicer_home / 'slicer.org'
    tensor_cli = next(cli.rglob('DWIToDTIEstimation'))
    scalar_cli = tensor_cli.with_name('DiffusionTensorScalarMeasurements')
    launcher = args.slicer_home / 'Slicer'
    python_slicer = next(path for path in (args.slicer_home / 'bin/PythonSlicer', args.slicer_home / 'PythonSlicer')
                         if path.is_file())
    author = ROOT / 'anatomy/surface/ddsurfer_backend/preprocessing/conversion/nhdr_write.py'
    references = []
    started = time.perf_counter()
    for index in range(2):
        directory = output / f'author_consumer_{index}'
        directory.mkdir()
        header = directory / 'DWI.nhdr'
        with (directory / 'commands.log').open('wb') as log:
            commands = [[sys.executable, str(author), '--nifti', str(bundle.dwi_path),
                         '--bval', str(bundle.bval_path), '--bvec', str(bundle.bvec_path), '--nhdr', str(header)]]
            for command in commands:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            environment = merged_startup_environment(pipeline.build_cli_environment(str(tensor_cli)))
            pipeline._rewrite_nhdr_data_file(header, bundle.dwi_path)
            subprocess.run([str(launcher), '--launch', str(tensor_cli), '--enumeration', 'WLS',
                            str(header), str(directory / 'DTI.nrrd'), str(directory / 'b0.nrrd')],
                           check=True, stdout=log, stderr=subprocess.STDOUT, env=environment)
            outputs = {'tensor': directory / 'DTI.nrrd', 'b0': directory / 'b0.nrrd'}
            for enum in ('FractionalAnisotropy', 'Trace', 'MeanDiffusivity', 'MinEigenvalue', 'MidEigenvalue', 'MaxEigenvalue'):
                target = directory / f'{enum}.nrrd'
                subprocess.run([str(launcher), '--launch', str(scalar_cli), '--enumeration', enum,
                                str(directory / 'DTI.nrrd'), str(target)],
                               check=True, stdout=log, stderr=subprocess.STDOUT, env=environment)
                outputs[enum] = target
        references.append(outputs)
    reference_seconds = time.perf_counter() - started
    commands = []
    def execute(command, log=None):
        commands.append(command)
        # The shared tensor fit is one native PythonSlicer command (shared/teem_dti.py), run without a GUI.
        with (output / 'shared_commands.log').open('ab') as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT,
                env=merged_startup_environment(pipeline.build_cli_environment(command[0])))
    with patch.object(pipeline, '_require_slicer'), \
            patch.object(pipeline, 'resolve_pythonslicer_executable', return_value=str(python_slicer)), \
            patch.object(pipeline, '_run_cli', side_effect=execute):
        stamp = time.perf_counter()
        shared = pipeline.ensure_native_dti_bundle(bundle, 'wls')
        first_seconds = time.perf_counter() - stamp
        before = {name: getattr(shared, name).stat().st_mtime_ns for name in shared.__dataclass_fields__
                  if name.endswith('_path')}
        stamp = time.perf_counter()
        pipeline.ensure_native_dti_bundle(bundle, 'wls')
        reuse_seconds = time.perf_counter() - stamp
        unchanged = before == {name: getattr(shared, name).stat().st_mtime_ns for name in before}
    mapping = dict(tensor=shared.dti_nrrd_path, b0=shared.b0_nrrd_path,
        FractionalAnisotropy=shared.fa_nrrd_path, Trace=shared.trace_nrrd_path,
        MeanDiffusivity=shared.mean_diffusivity_nrrd_path, MinEigenvalue=shared.min_eig_nrrd_path,
        MidEigenvalue=shared.mid_eig_nrrd_path, MaxEigenvalue=shared.max_eig_nrrd_path)
    comparisons = [{key: compare(paths[key], target) for key, target in mapping.items()} for paths in references]
    sources_unchanged = all(pipeline._cache_file_signature(Path(record['path'])) == record for record in original.values())
    passed = (sources_unchanged and unchanged and len(commands) == 1 and
              all(item['exact_values'] and item['geometry_equal'] for row in comparisons for item in row.values()))
    return dict(subject=subject, passed=passed, crop_shape=list(data.shape), crop_origin=start.tolist(),
        reference_seconds=reference_seconds, shared_build_seconds=first_seconds, checked_reuse_seconds=reuse_seconds,
        reference_tensor_calls=2, reference_scalar_calls=12, shared_tensor_calls=1, shared_scalar_calls=0,
        cache_hit_wrote_no_files=unchanged, source_inputs_unchanged=sources_unchanged, comparisons=comparisons,
        limitations=['Native ROI preparation, not whole brain or trained surface inference',
                     'Shared build is one native Teem process plus six NIfTI exports; references are the SlicerDMRI tensor/scalar CLI calls',
                     'One ordered pair on a shared host; no global speed ratio claimed'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subjects', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--slicer-home', type=Path, required=True)
    parser.add_argument('--size', type=int, default=12)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.slicer_home = args.slicer_home.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for source in args.subjects:
        record = run_case(args, source)
        records.append(record)
        (args.output / 'summary.json').write_text(json.dumps(dict(kind='shared_preparation_validation',
            passed=all(item['passed'] for item in records), records=records,
            code_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                (Path(__file__), ROOT / 'shared/pipeline.py', ROOT / 'anatomy/surface/pipeline.py')}), indent=2) + '\n')
        print(json.dumps(record), flush=True)
    return 0 if all(item['passed'] for item in records) else 1


if __name__ == '__main__':
    raise SystemExit(main())
