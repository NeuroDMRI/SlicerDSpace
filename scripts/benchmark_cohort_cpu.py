#!/usr/bin/env python3
"""Rotating, paired CPU scheduling pilot on supplied real diffusion ROI inputs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.cohort_schedule import ENV, create_schedule, stage_admission, schedule_summary
from shared.cohort_profile import StageMonitor


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(source, output, mode):
    import nibabel as nib
    import numpy as np
    from dipy.core.gradients import gradient_table
    from dipy.reconst.dti import TensorModel, fractional_anisotropy, mean_diffusivity
    outer = stage_admission('whole_subject') if mode == 'subjects' else nullcontext()
    with outer:
        with stage_admission('read'):
            image = nib.load(str(source / 'data.nii.gz'))
            data = image.get_fdata(dtype=np.float64)
            bvals, bvecs = np.loadtxt(source / 'bvals'), np.loadtxt(source / 'bvecs')
            if bvecs.shape[0] == 3:
                bvecs = bvecs.T
            mask = np.isfinite(data).all(axis=-1) & (data[..., bvals < 50].mean(axis=-1) > 0)
            gtab = gradient_table(bvals, bvecs=bvecs)
        with stage_admission('fit'):
            model = TensorModel(gtab, fit_method='WLS').fit(data, mask=mask)
            result = dict(parameters=model.model_params, fa=fractional_anisotropy(model.evals),
                          md=mean_diffusivity(model.evals), affine=image.affine, mask=mask)
            if not all(np.isfinite(value).all() for value in result.values()):
                raise ValueError('Nonfinite numerical output')
            qc = dict(finite=True, voxels=int(mask.sum()),
                      fa_in_range=bool(((result['fa'][mask] >= 0) & (result['fa'][mask] <= 1)).all()))
            if not qc['fa_in_range'] or not qc['voxels']:
                raise ValueError('Tensor output QC failed')
        with stage_admission('export'):
            output.mkdir()
            for key, value in result.items():
                np.save(output / (key + '.npy'), value, allow_pickle=False)
            (output / 'qc.json').write_text(json.dumps(qc, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--worker', choices=('subjects', 'resources', 'adaptive'))
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.source, args.output, args.worker)
        return
    if not args.inputs or args.repeats < 1:
        parser.error('Supply input directories and positive repeats')
    args.output.mkdir(parents=True, exist_ok=False)
    baseline, runs = {}, []
    modes = ('subjects', 'resources', 'adaptive')
    for repeat in range(args.repeats):
        for mode in modes[repeat % 3:] + modes[:repeat % 3]:
            folder = args.output / f'{repeat}-{mode}'
            profiles = {str(i): dict(read=dict(cpu=1, memory_gb=.5, io=1),
                                    fit=dict(cpu=1, memory_gb=.5), export=dict(cpu=1, memory_gb=.5, io=1))
                        for i in range(len(args.inputs))}
            schedule = create_schedule(folder, cpu_budget=2, memory_gb=4, stage_cpus=1,
                stage_memory_gb=1, subject_profiles=profiles, io_slots=1,
                policy='fair_backfill' if mode == 'adaptive' else 'first_fit')
            children = []
            started = time.monotonic()
            monitor = StageMonitor(.05).start()
            def run(index):
                env = {**os.environ, ENV: str(schedule), 'SLICERDSPACE_COHORT_SUBJECT': str(index),
                       'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
                command = [sys.executable, __file__, '--worker', mode, '--source', str(args.inputs[index]),
                           '--output', str(folder / f'output-{index}')]
                child = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                children.append(child)
                _, error = child.communicate(timeout=300)
                if child.returncode:
                    raise RuntimeError(error.decode(errors='replace'))
            try:
                with ThreadPoolExecutor(max_workers=1 if mode == 'subjects' else 2 if mode == 'resources' else 8) as pool:
                    list(pool.map(run, range(len(args.inputs))))
            finally:
                for child in children:
                    if child.poll() is None:
                        child.kill()
                        child.wait()
                measurement = monitor.finish('completed')
            elapsed = time.monotonic() - started
            for index in range(len(args.inputs)):
                hashes = {path.name: sha(path) for path in sorted((folder / f'output-{index}').iterdir())}
                if index in baseline and baseline[index] != hashes:
                    raise RuntimeError('Numerical/affine/mask/QC outputs changed')
                baseline[index] = hashes
            runs.append(dict(repeat=repeat, mode=mode, seconds=elapsed,
                qc_passing_inputs=len(args.inputs), inputs_per_minute=len(args.inputs) * 60 / elapsed,
                measurement=measurement, schedule=schedule_summary(schedule)))
    report = dict(status='PASS', scope='Real-data ROI tensor-fitting pilot, not full-brain or full-pipeline cohort throughput',
        input_count=len(args.inputs), repeats=args.repeats, modes=list(modes), runs=runs,
        median_seconds={mode: statistics.median(row['seconds'] for row in runs if row['mode'] == mode) for mode in modes},
        outputs_identical=True, checked_outputs=['13 tensor parameters', 'FA', 'MD', 'affine', 'mask', 'QC'],
        input_hashes=[{name: sha(root / name) for name in ('data.nii.gz', 'bvals', 'bvecs')} for root in args.inputs],
        output_hashes=baseline, budget=dict(cpu=2, memory_gb=4, io_slots=1, numerical_threads=1),
        environment=dict(python=sys.version, platform=platform.platform(), load_average=os.getloadavg() if hasattr(os, 'getloadavg') else None),
        cache_scope='Fresh output folders; OS cache is not cleared; policies rotate across repeats',
        source_hashes={name: sha(ROOT / name) for name in ('shared/cohort_schedule.py', 'shared/cohort_policy.py',
                      'shared/cohort_profile.py', 'scripts/benchmark_cohort_cpu.py')})
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'scope', 'median_seconds', 'outputs_identical')}, indent=2))


if __name__ == '__main__':
    main()
