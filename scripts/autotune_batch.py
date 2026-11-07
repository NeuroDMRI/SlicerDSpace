#!/usr/bin/env python3
"""Calibrate only requested CPU stages; retain raw trials and exact comparisons."""

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.compute_tuning import candidates, config_id, tuning_key, select_configuration, read_profile, calibration_options
_active_record = None


def save(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def prepare_sample(root, destination, stage, sample_voxels, protocol=None):
    import nibabel as nib
    import numpy as np
    from shared.pipeline import RawInputBundle
    from microstructure.dki import DkiConfig, _load_protocol, _signature
    protocol = protocol or calibration_options(stage)
    directory = root / 'input' if (root / 'input/data.nii.gz').is_file() else root
    paths = [directory / name for name in ('data.nii.gz', 'bvals', 'bvecs', 'nodif_brain_mask.nii.gz')]
    if not all(path.is_file() for path in paths):
        raise ValueError('Calibration needs final DWI, gradients and a supplied mask: ' + str(root))
    source = {name: _signature(path) for name, path in zip(('dwi', 'bval', 'bvec', 'mask'), paths)}
    shape = list(nib.load(paths[0]).shape)
    destination.mkdir(parents=True, exist_ok=False)
    grid = None
    if stage == 'microstructure_mkcurve':
        from microstructure.mkcurve_inputs import _prepare_inputs
        from microstructure.mkcurve import repair_nonpositive, synthetic_b0_grid
        with tempfile.TemporaryDirectory(prefix='.prepare-', dir=destination) as temporary:
            prepared, _, _ = _prepare_inputs(*paths, Path(temporary), protocol['b0_threshold'],
                                              protocol['max_bvalue'], allow_nonfinite=True)
            image = nib.load(prepared[0], mmap='c')
            data = image.dataobj.get_unscaled()
            try:
                mask = np.asarray(nib.load(prepared[3]).dataobj) > 0
                repair = repair_nonpositive(data, mask)
                coords = np.unravel_index(np.flatnonzero(mask.ravel(order='F')), mask.shape, order='F')
                grid = synthetic_b0_grid(data[..., 0][coords])
                indices = np.linspace(0, len(coords[0])-1, min(sample_voxels, len(coords[0])), dtype=int)
                values = np.array(data[tuple(axis[indices] for axis in coords)], copy=True)
                bvals, bvecs = np.loadtxt(prepared[1]).reshape(-1), np.loadtxt(prepared[2])
            finally:
                if isinstance(data, np.memmap):
                    data._mmap.close()
    else:
        bundle = RawInputBundle(root, root.name, directory, *paths)
        config = DkiConfig(root.name, bundle, destination, **protocol)
        image, mask, gtab, selected, _ = _load_protocol(config)
        indices = np.flatnonzero(mask.ravel())
        positions = indices[np.linspace(0, len(indices)-1, min(sample_voxels, len(indices)), dtype=int)]
        coords = np.unravel_index(positions, mask.shape)
        from shared.diffusion_io import diffusion_image
        with diffusion_image(paths[0], destination) as staged:
            # Read the selected spatial samples, preserving the image's scaling.
            values = np.stack([np.asarray(staged.dataobj[..., int(volume)], dtype=np.float64)[coords]
                               for volume in selected], axis=1)
        bvals, bvecs, repair = gtab.bvals, gtab.bvecs.T, None
    valid = np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
    values = values[valid]
    if len(values) < 8:
        raise ValueError('Too few positive finite calibration signals')
    nib.save(nib.Nifti1Image(values[:, None, None, :], np.eye(4)), destination / 'data.nii')
    nib.save(nib.Nifti1Image(np.ones((len(values), 1, 1), np.uint8), np.eye(4)), destination / 'mask.nii.gz')
    np.savetxt(destination / 'bvals', bvals)
    np.savetxt(destination / 'bvecs', bvecs)
    if grid is not None:
        np.savetxt(destination / 'grid.txt', grid)
    if any(source[key] != _signature(path) for key, path in zip(source, paths)):
        raise RuntimeError('Calibration source changed during preparation')
    record = dict(source=str(root), shape=shape, sampled_voxels=len(values), excluded_samples=int((~valid).sum()),
                  inputs=source, repair=repair, grid_samples=len(grid) if grid is not None else None,
                  protocol=protocol, selected_measurements=len(bvals),
                  scope='Sampled real signals; full acquisition table and full MK-Curve grid; not a full-volume timing')
    save(destination / 'source.json', record)
    return record


def probe(request):
    from shared.cohort_profile import StageMonitor
    from scripts.benchmark_cohort_compare import product_hash
    stage, config = request['stage'], request['settings']
    protocol = request.get('protocol') or calibration_options(stage)
    slots = config['workers'] * config['threads']
    if hasattr(os, 'sched_getaffinity'):
        allocation = sorted(os.sched_getaffinity(0))
        if slots > len(allocation):
            raise ValueError('Calibration candidate exceeds available CPU affinity')
        os.sched_setaffinity(0, allocation[:slots])
    os.environ['SLICERDSPACE_STAGE_CPUS'] = str(slots)
    folder, source = Path(request['output']), Path(request['input'])
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    monitor = StageMonitor(interval=.05).start()
    try:
        if stage == 'wma':
            from tractography.wma import WMAOrgAtlasConfig, run_wma_org_atlas
            result = run_wma_org_atlas(WMAOrgAtlasConfig(subject_id='calibration', input_tractography=source,
                atlas_dir=Path(request['atlas']), output_dir=folder,
                slicer_executable=Path(request['launcher']), wma_root=Path(request['wma_root']),
                num_threads=config['threads'], execute=True))
            if result.status != 'completed':
                raise ValueError('Full WMA calibration did not complete')
        elif stage == 'microstructure_mkcurve':
            import numpy as np
            from microstructure.mkcurve import run_mkcurve
            run_mkcurve(source/'data.nii', source/'bvals', source/'bvecs', source/'mask.nii.gz', folder,
                        grid=np.loadtxt(source/'grid.txt'), voxels_per_chunk=config['chunk'],
                        num_workers=config['workers'], worker_threads=config['threads'],
                        b0_threshold=protocol['b0_threshold'], max_bvalue=protocol['max_bvalue'],
                        chunk_schedule=request.get('chunk_schedule', 'dynamic'))
        else:
            from shared.pipeline import RawInputBundle
            from microstructure.dki import DkiConfig, run_dki
            bundle = RawInputBundle(source, 'calibration', source, source/'data.nii', source/'bvals',
                                    source/'bvecs', source/'mask.nii.gz')
            run_dki(DkiConfig('calibration', bundle, folder, **protocol,
                              num_threads=config['threads'], voxels_per_chunk=config['chunk']))
        seconds = time.monotonic() - started
        products = {str(path.relative_to(folder)): product_hash(path) for path in sorted(folder.rglob('*'))
                    if path.is_file() and path.name.endswith(('.nii', '.nii.gz', '.npy', '.vtk', '.vtp', '.tfm'))}
        if not products:
            raise ValueError('Calibration produced no scientific products')
        measurement = monitor.finish('completed')
        return dict(status='completed', seconds=seconds, rss_bytes=measurement['peak_tree_rss_bytes'],
                    fingerprints=products, measurement=measurement)
    except BaseException:
        monitor.finish('failed')
        raise


def main(argv=None):
    global _active_record
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject-root', type=Path, action='append', default=[])
    parser.add_argument('--stages', default='microstructure_dki')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max-cpus', type=int, default=4)
    parser.add_argument('--cpu-budget', type=int, default=4)
    parser.add_argument('--memory-gb', type=float, default=32.)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--sample-voxels', type=int, default=1024)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--objective', choices=('throughput', 'latency'), default='throughput')
    parser.add_argument('--wma-input', type=Path)
    parser.add_argument('--probe-request', type=Path, help=argparse.SUPPRESS)
    args, forwarded = parser.parse_known_args(argv)
    if args.probe_request:
        request = json.loads(args.probe_request.read_text())
        save(args.probe_request.with_suffix('.result.json'), probe(request))
        return 0
    if (args.output is None or args.repeats < 3 or args.sample_voxels < 8 or args.max_cpus < 1
            or args.cpu_budget < args.max_cpus or args.jobs < 1 or not math.isfinite(args.memory_gb) or args.memory_gb <= 0):
        parser.error('Provide output, >=3 repeats, >=8 sampled voxels and valid CPU/RAM limits')
    forwarded = [value for value in forwarded if value != '--']
    from shared.cohort_planning import _option
    profile = read_profile(args.output) or dict(version=1, status='passed', profiles={})
    work = args.output.parent / ('calibration_' + uuid4().hex)
    work.mkdir(parents=True)
    _active_record = work/'campaign.json'
    record = dict(status='running', stages=args.stages.split(','), trials=[], deferred=[],
                  preparation_seconds=0., objective=args.objective, max_cpus=args.max_cpus,
                  cpu_budget=args.cpu_budget, memory_gb=args.memory_gb, jobs=args.jobs,
                  scope='Exact-checked bounded configuration search; sample times are not cohort speedups')
    import platform
    from shared.cohort_profile import _implementation
    record['environment'] = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                                 node=platform.node(), implementation=_implementation(),
                                 cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None)
    save(work/'campaign.json', record)
    began = time.monotonic()
    for stage in record['stages']:
        configs = candidates(stage, args.max_cpus)
        if not configs:
            record['deferred'].append(dict(stage=stage, reason='No calibrated adapter; retain declared defaults'))
            continue
        prepared = []
        for index, root in enumerate(args.subject_root[:3]):
            import nibabel as nib
            from shared.cohort_planning import image_size
            shape = image_size(root)['shape']
            key = tuning_key(stage, shape, forwarded)
            if key in profile['profiles']:
                continue
            if stage == 'wma':
                source = args.wma_input or Path(_option(forwarded, '--wma-input-tractography', '/__missing__'))
                atlas = _option(forwarded, '--wma-atlas-dir')
                wma_root = _option(forwarded, '--wma-root')
                launcher = _option(forwarded, '--slicer-executable')
                if not source.is_file() or not all((atlas, wma_root, launcher)):
                    record['deferred'].append(dict(stage=stage, reason='WMA tuning requires an existing tractogram, atlas and launcher; no upstream stages are run'))
                    break
                prepared.append(dict(input=str(source), key=key, shape=shape,
                                     atlas=atlas, wma_root=wma_root, launcher=launcher))
                break
            sample = work / f'{stage}_{index}_input'
            if not any((directory/'data.nii.gz').is_file() for directory in (root/'input',root)):
                record['deferred'].append(dict(stage=stage,input=str(root),
                    reason='Calibration requires final NIfTI inputs; normalisation/execution remain available without calibration'))
                continue
            started = time.monotonic()
            source_record = prepare_sample(root, sample, stage, args.sample_voxels, calibration_options(stage, forwarded))
            record['preparation_seconds'] += time.monotonic() - started
            prepared.append(dict(input=str(sample), key=key, shape=shape, source=source_record))
        if not prepared:
            continue
        baselines, rows = {}, []
        for repeat in range(-1, args.repeats):
            # One warm run per configuration, then rotating repeated measurements.
            order = configs if repeat < 0 else configs[repeat % len(configs):] + configs[:repeat % len(configs)]
            for index, sample in enumerate(prepared):
                for config in order:
                    ident = config_id(config)
                    trial = work / f'{stage}-i{index}-{ident}-r{repeat}'
                    request = dict(stage=stage, settings=config, output=str(trial),
                                   protocol=calibration_options(stage, forwarded),
                                   fit_method=_option(forwarded, '--dki-fit-method', 'MKCURVE'), **sample)
                    request_path = trial.with_suffix('.request.json')
                    save(request_path, request)
                    environment = {key:value for key,value in os.environ.items()
                                   if not key.startswith('SLICERDSPACE_COHORT_') and key != 'SLICERDSPACE_STAGE_CPUS'}
                    environment.update({key:str(config['threads']) for key in
                                        ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')})
                    wall = time.monotonic()
                    with trial.with_suffix('.stdout.log').open('w') as stdout, trial.with_suffix('.stderr.log').open('w') as stderr:
                        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--probe-request', str(request_path)],
                            env=environment, stdout=stdout, stderr=stderr)
                    result_path = request_path.with_suffix('.result.json')
                    row = json.loads(result_path.read_text()) if result.returncode == 0 and result_path.is_file() else dict(status='failed')
                    row.update(config=ident, input=sample['input'], repeat=repeat, key=sample['key'],
                               launch_seconds=time.monotonic()-wall, output=str(trial))
                    if repeat == -1 and config == configs[0] and row['status'] == 'completed':
                        baselines[sample['input']] = row['fingerprints']
                    row['exact'] = bool(row.get('fingerprints')) and row.get('fingerprints') == baselines.get(sample['input'])
                    record['trials'].append(row)
                    if repeat >= 0:
                        rows.append(row)
                    save(work/'campaign.json', record)
                    print(f'[Autotune] {stage} {ident} repeat={repeat}: {row["status"]}; exact={row["exact"]}', flush=True)
        grouped = defaultdict(list)
        for row in rows:
            grouped[row['key']].append(row)
        for key, measurements in grouped.items():
            selected = select_configuration(measurements, configs, cpu_budget=args.cpu_budget,
                memory_gb=args.memory_gb, jobs=args.jobs, objective=args.objective)
            compact = [{name:row[name] for name in ('config','input','repeat','seconds','rss_bytes','status','exact')
                        if name in row} for row in measurements]
            profile['profiles'][key] = dict(stage=stage, configs=configs, measurements=compact,
                objective=args.objective, selection=selected, source=str(work/'campaign.json'))
    record.update(status='passed', total_seconds=time.monotonic()-began)
    save(work/'campaign.json', record)
    profile.update(last_calibration=str(work/'campaign.json'))
    save(args.output, profile)
    print(json.dumps(dict(status='passed', profile=str(args.output), profiles=len(profile['profiles']),
                          deferred=record['deferred'], elapsed_seconds=record['total_seconds']), indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        if _active_record and _active_record.is_file():
            record = json.loads(_active_record.read_text())
            record.update(status='failed',error=f'{type(exc).__name__}: {exc}')
            save(_active_record,record)
        raise
