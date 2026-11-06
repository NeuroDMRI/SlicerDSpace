#!/usr/bin/env python3
"""Fresh author surface route versus a consumer of already-prepared shared WLS.

Runs outside any live Slicer GUI. All new files stay in a fresh evidence root.
The second timing intentionally has shared native preparation ready, as after
DDParcel; its preparation time is recorded separately, not hidden in a speedup.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import nibabel as nib
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from anatomy.surface import SurfaceConfig, run_surface
from shared import pipeline
from shared.external_process import merged_startup_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject-root', type=Path, required=True)
    parser.add_argument('--subject', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--slicer-home', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    subject = root / 'subject'
    subject.mkdir()
    (subject / 'input').symlink_to(args.subject_root.resolve() / 'input', target_is_directory=True)
    raw = pipeline.RawInputBundle(subject, args.subject, subject / 'input', subject / 'input/data.nii.gz',
        subject / 'input/bvals', subject / 'input/bvecs', subject / 'input/nodif_brain_mask.nii.gz')
    config = SurfaceConfig(args.subject, subject, root / 'author', input_bundle=raw, execute=True,
        model_assets_dir=args.assets.resolve(), slicer_home=args.slicer_home.resolve(), device='auto')
    binaries = {name: next(args.slicer_home.rglob(filename)) for name, filename in (
        ('brainsfit', 'BRAINSFit'), ('resamplescalarvectordwivolume', 'ResampleScalarVectorDWIVolume'))}
    python_slicer = next(path for path in (args.slicer_home / 'bin/PythonSlicer', args.slicer_home / 'PythonSlicer')
                         if path.is_file())
    def execute(command, log=None):
        # The native tensor fit is a PythonSlicer command; Slicer CLI modules still start through the launcher.
        launch = [] if Path(command[0]) == python_slicer else [str(args.slicer_home / 'Slicer'), '--launch']
        subprocess.run([*launch, *command], check=True,
            env=merged_startup_environment(pipeline.build_cli_environment(command[0])))
    started = time.perf_counter()
    author = run_surface(config, log=lambda value: print(value, flush=True))
    author_seconds = time.perf_counter() - started
    with patch.object(pipeline, '_require_slicer'), \
            patch.object(pipeline, 'resolve_pythonslicer_executable', return_value=str(python_slicer)), \
            patch.object(pipeline, '_get_cli_executable', side_effect=lambda name: str(binaries[name])), \
            patch.object(pipeline, '_run_cli', side_effect=execute):
        started = time.perf_counter()
        pipeline.ensure_native_dti_bundle(raw, 'wls')
        preparation_seconds = time.perf_counter() - started
        started = time.perf_counter()
        shared = run_surface(replace(config, output_dir=root / 'shared', reuse_shared_preprocessing=True),
                             log=lambda value: print(value, flush=True))
        shared_seconds = time.perf_counter() - started
    maps = {}
    for suffix in ('FA', 'MinEigenvalue', 'MidEigenvalue', 'MaxEigenvalue', 'MD', 'brainmask_resampled'):
        a = nib.load(root / 'author/upstream' / args.subject / '.cache/volumes' / args.subject / f'{args.subject}-{suffix}.nii.gz')
        b = nib.load(root / 'shared/preprocessed' / args.subject / f'{args.subject}-{suffix}.nii.gz')
        aa, bb = np.asanyarray(a.dataobj), np.asanyarray(b.dataobj)
        maps[suffix] = dict(exact_values=bool(np.array_equal(aa, bb)),
                            max_abs_difference=float(np.max(np.abs(aa.astype(float) - bb))),
                            affine_exact=bool(np.array_equal(a.affine, b.affine)))
    surfaces = {}
    for hemi in ('lh', 'rh'):
        for kind in ('white', 'pial'):
            a = trimesh.load(root / 'author/pred_results_SlicerDSpace/native' / args.subject / f'{hemi}.{kind}.obj', process=False)
            b = trimesh.load(root / 'shared/pred_results_SlicerDSpace/native' / args.subject / f'{hemi}.{kind}.obj', process=False)
            surfaces[f'{hemi}.{kind}'] = dict(topology_exact=bool(np.array_equal(a.faces, b.faces)),
                vertices_exact=bool(np.array_equal(a.vertices, b.vertices)),
                max_vertex_distance_mm=float(np.max(np.linalg.norm(a.vertices - b.vertices, axis=1))))
    qcs = [json.loads(result.qc_path.read_text()) for result in (author, shared)]
    record = dict(kind='shared_surface_full_pair', subject=args.subject, author_seconds=author_seconds,
        shared_native_preparation_seconds=preparation_seconds, shared_consumer_seconds=shared_seconds,
        maps=maps, surfaces=surfaces, automatic_qc=[qc['automatic_qc_passed'] for qc in qcs],
        code_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in
            (Path(__file__), ROOT / 'shared/pipeline.py', ROOT / 'anatomy/surface/pipeline.py')},
        limitations=['Single subject, shared host, author first', 'Shared WLS preparation is separately timed',
                     'Workflow reuse, not a faster tensor estimator or a whole-workflow ratio',
                     'Automatic geometric QC is not manual anatomical review'])
    record['passed'] = (all(record['automatic_qc']) and all(row['exact_values'] and row['affine_exact'] for row in maps.values())
                        and all(row['topology_exact'] and row['vertices_exact'] for row in surfaces.values()))
    (root / 'summary.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)
    return 0 if record['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
