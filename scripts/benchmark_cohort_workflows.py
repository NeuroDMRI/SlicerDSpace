#!/usr/bin/env python3
"""Frozen-source full-volume cohort campaign; never substitutes ROI or duplicate subjects.

Actions: prepare, run, select (legacy campaigns); pair, qc, compare, link-subjects (T3 paired benchmark).
"""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from importlib.metadata import PackageNotFoundError, version

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.cohort_telemetry import CohortTelemetry
from scripts.benchmark_cohort_harness import (
    ARM_CPUS, ARM_THREADS, BATCH_MEMORY_GB, KILL_MEMORY_GB, MAX_ATTEMPTS, MAX_BUSY_PERCENT, RETRY_WAIT_SECONDS,
    SUBJECT_JOBS,
    GPU_SAMPLE_SECONDS, HOST_SAMPLE_SECONDS, THREAD_ENV, T3_ROUTES, LaunchRefused, HostReader, Sampler,
    assign_arms, claim_folder, decide_launch, doctor_blockers, kill_group, make_gpu_reader, parse_cpu_list,
    parse_doctor_output, read_topology, run_bounded, run_launch_rule, safe_component, sample_window)

POLICIES = ('sequential', 'subject', 'stage', 'adaptive')
ALL_POLICIES = (*POLICIES, 'adaptive_tuned')
INPUTS = ('data.nii.gz', 'bvals', 'bvecs', 'nodif_brain_mask.nii.gz')


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def signature(path):
    stat = path.stat()
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def inspect_cohort(root, count, names=None):
    import nibabel as nib
    import numpy as np
    rows, skipped, real_sources = [], [], set()
    # `names` (T3): an explicit, ordered ID list; each ID becomes a path component and is validated first.
    directories = (sorted(root.iterdir()) if names is None
                   else [root/safe_component(name, 'subject id') for name in names])
    for directory in directories:
        source = directory / 'T1w/Diffusion'
        if not all((source/name).is_file() for name in INPUTS):
            continue
        try:
            image, mask = (nib.load(str(source/name)) for name in ('data.nii.gz', 'nodif_brain_mask.nii.gz'))
            bvals, bvecs = np.loadtxt(source/'bvals'), np.loadtxt(source/'bvecs')
            if len(image.shape) != 4 or bvals.size != image.shape[3] or bvecs.shape not in ((3,bvals.size),(bvals.size,3)):
                raise ValueError('DWI/gradient count mismatch')
            if mask.shape != image.shape[:3] or not np.allclose(image.affine, mask.affine, atol=1e-5):
                raise ValueError('Mask geometry mismatch')
            if not np.isfinite(bvals).all() or not np.isfinite(bvecs).all():
                raise ValueError('Nonfinite gradients')
            if len(set(np.round(bvals[bvals > 50]/100))) < 2:
                raise ValueError('Multi-shell acquisition required')
            resolved = str((source/'data.nii.gz').resolve())
            if resolved in real_sources:
                raise ValueError('Duplicate acquisition path')
            real_sources.add(resolved)
            rows.append(dict(subject=directory.name, source=str(source.resolve()), shape=list(image.shape),
                             signatures={name: signature(source/name) for name in INPUTS}))
        except Exception as exc:
            skipped.append(dict(subject=directory.name, reason=str(exc)))
        if len(rows) == count:
            break
    if len(rows) < count:
        raise ValueError(f'Only {len(rows)} eligible distinct acquisitions; requested {count}')
    return rows, skipped


def select_campaign(source, output, *, policies, routes, reference_policy='subject'):
    """Select unstarted arms or reuse completed evidence without editing its snapshot."""
    original = json.loads(source.read_text())
    if reference_policy not in policies or 'adaptive' not in policies:
        raise ValueError('Keep the comparison reference and adaptive arms')
    selected = [deepcopy(row) for row in original['trials']
                if row['policy'] in policies and row['route'] in routes]
    if not selected or any(row['status'] not in ('pending','completed') for row in selected):
        raise ValueError('Only pending or completed trials can be selected; finish active work first')
    for route in routes:
        if not any(row['route']==route for row in selected):
            raise ValueError('Selected route is absent: '+route)
        for threads in {row['threads'] for row in selected if row['route']==route}:
            observed = {row['policy'] for row in selected if row['route']==route and row['threads']==threads}
            if not set(policies) <= observed:
                raise ValueError('Selected route lacks a required comparison arm')
    record = deepcopy(original)
    for key in ('driver_pid','started_at','inputs_unchanged'):
        record.pop(key, None)
    record.update(status='prepared', trials=selected, reference_policy=reference_policy,
                  selected_from=dict(path=str(source.resolve()), sha256=sha(source)),
                  omitted_trials=[row['id'] for row in original['trials'] if row['id'] not in {x['id'] for x in selected}])
    record['notes'].append('Priority comparison selected before starting omitted arms; completed measurements and scientific snapshot unchanged')
    for row in selected:
        if row['status']=='completed':
            evidence = Path(row['output'])
            required = ('independent_qc.json','scientific_fingerprints.json')
            if any(not (evidence/name).is_file() for name in required):
                raise ValueError('Completed trial has missing evidence: '+row['id'])
            row['inherited_evidence'] = {name:sha(evidence/name) for name in required}
    output.mkdir(parents=True, exist_ok=False)
    save(output/'campaign.json', record)
    return output/'campaign.json'


def comparison_reference(record, trial):
    policy = record.get('reference_policy', 'sequential')
    return next((row for row in record['trials'] if row['route']==trial['route'] and
                 row['threads']==trial['threads'] and row['policy']==policy and row['status']=='completed'), None)


def timing_issues(record, trial):
    """Do not accept speedups when the recorded CPU-isolation contract failed."""
    folder = Path(trial['output'])
    issues = []
    expected = record.get('cpu_affinity')
    if expected:
        paths = [folder/'subject_guard/config.json'] if trial['policy'] in ('subject','sequential') else list(folder.glob('schedule_*/config.json'))
        if not paths or any(not path.is_file() for path in paths):
            issues.append('Missing CPU-placement guard record')
        for path in paths:
            if path.is_file() and set(json.loads(path.read_text()).get('cpu_affinity', [])) != set(expected):
                issues.append('Guard CPU set differs from campaign allocation: '+str(path))
    if trial.get('measurement', {}).get('cpu_affinity_violations'):
        issues.append('Observed descendant thread escaped the CPU allocation')
    condition = folder/'timing_conditions.json'
    if condition.is_file():
        context = json.loads(condition.read_text())
        if context.get('performance_valid') is False:
            issues.extend(context.get('reasons') or ['Recorded timing intervention or resource contention'])
    return issues


def update_comparisons(record, folder):
    from scripts.benchmark_cohort_compare import compare_inventories
    reports = []
    for trial in record['trials']:
        reference = comparison_reference(record, trial)
        if trial['status']!='completed' or not reference:
            continue
        left = Path(reference.get('scientific_fingerprints') or Path(reference['output'])/'scientific_fingerprints.json')
        right = Path(trial.get('scientific_fingerprints') or Path(trial['output'])/'scientific_fingerprints.json')
        if not (left.is_file() and right.is_file()):
            continue
        baseline, candidate = json.loads(left.read_text()), json.loads(right.read_text())
        expected = {row['subject'] for row in record['subjects']}
        subjects_match = set(baseline)==set(candidate)==expected
        comparisons = {key:compare_inventories(baseline[key], candidate[key]) for key in expected
                       if key in baseline and key in candidate}
        trial['comparison_status'] = 'passed' if subjects_match and all(
            row['status']=='passed' for row in comparisons.values()) else 'failed'
        # Reused receipts stay immutable; derivative comparisons live in the new campaign.
        destination = folder/'comparisons'/trial['id']
        destination.mkdir(parents=True,exist_ok=True)
        save(destination/'comparison.json',dict(status=trial['comparison_status'],subjects=comparisons,
             reference=reference['id'],subjects_match=subjects_match))
        if trial['policy'] not in ('adaptive','adaptive_tuned'):
            continue
        isolation_issues = timing_issues(record, reference) + timing_issues(record, trial)
        trial['timing_issues'] = isolation_issues
        accepted = (trial.get('qc_status')==reference.get('qc_status')=='passed'
                    and trial['comparison_status']=='passed' and not isolation_issues)
        before = reference['measurement']['execution_seconds']
        after = trial['measurement']['execution_seconds']
        reports.append(dict(route=trial['route'],reference=reference['id'],candidate=trial['id'],
            status='invalid_resource_isolation' if isolation_issues else 'passed' if accepted else 'pending_acceptance',
            timing_issues=isolation_issues,reference_minutes=before/60,candidate_minutes=after/60,
            speed_ratio=before/after if accepted else None,
            reduction_percent=100*(1-after/before) if accepted else None,
            cpu_budget=record['cpus'],memory_gb=record['memory_gb'],jobs=record['jobs'],
            effect='scheduling plus runtime selection' if trial['policy']=='adaptive_tuned' else 'matched-configuration scheduling',
            threads=trial['threads'],subjects=len(expected)))
    save(folder/'primary_comparison.json',dict(comparisons=reports,
         scope='Single paired cohort measurement; no confidence interval or generalised speedup claim'))
    incremental = []
    for tuned in (row for row in record['trials'] if row['policy']=='adaptive_tuned' and row.get('comparison_status')=='passed'):
        fixed = next((row for row in record['trials'] if row['policy']=='adaptive'
                      and row['route']==tuned['route'] and row['threads']==tuned['threads']
                      and row.get('comparison_status')=='passed' and row.get('qc_status')=='passed'),None)
        if (fixed and tuned.get('qc_status')=='passed' and not timing_issues(record, fixed)
                and not timing_issues(record, tuned)):
            before, after = (row['measurement']['execution_seconds'] for row in (fixed,tuned))
            incremental.append(dict(reference=fixed['id'],candidate=tuned['id'],route=tuned['route'],
                speed_ratio=before/after,reduction_percent=100*(1-after/before),
                effect='Runtime configuration selection with the same latest scheduler and cohort budgets'))
    save(folder/'configuration_comparison.json',dict(comparisons=incremental))

SCHEDULERS = dict(sequential='subjects', subject='subjects', stage='resources', adaptive='adaptive',
                  adaptive_tuned='adaptive')
ROUTES = ('dki', 'mkcurve', 'tracts', *T3_ROUTES)
MODEL_DIRECTORIES = ('segmentation', 'parcellation', 'surface', 'Resources')


def common_arguments(args, *, print_manifest_only=True):
    """Arguments shared by all arms. H1: no --strict-environment; the per-subject environment check is off
    (H7 runs --doctor once before launch)."""
    common = ['--no-user-config', '--brain-mask-backend', 'provided', '--skip-environment-check',
              '--model-assets-dir', str(args.models.resolve()), '--mrtrix-bin-dir', str(args.mrtrix.resolve()),
              '--max-parallel-models', '1']
    return common + ['--print-manifest-only'] if print_manifest_only else common


def route_options(args, routes):
    options = {}
    if 'dki' in routes:
        options['dki'] = ['--targets', 'validation,microstructure', '--microstructure-stages', 'dti,dki',
                          '--dki-fit-method', 'WLS']
    if 'mkcurve' in routes:
        options['mkcurve'] = ['--targets', 'validation,microstructure', '--microstructure-stages', 'dti,dki',
                              '--dki-fit-method', 'MKCURVE']
    if 'tracts' in routes:
        options['tracts'] = ['--targets', 'validation,segmentation,parcellation,microstructure,tractography,bundles',
                '--microstructure-stages', 'dti', '--tractography-backend', 'csd_act,ukf',
                '--bundle-backend', 'wma', '--bundle-tractography-backend', 'ukf',
                '--select-streamlines', '1000000', '--mrtrix-seed', '20261103',
                '--wma-root', str(args.wma.resolve()), '--wma-atlas-dir', str(args.atlas.resolve()),
                '--slicer-executable', str(args.launcher.resolve()),
                '--ukf-executable', str(args.ukf_executable.resolve()) if args.ukf_executable else '']
    if 'cpu' in routes:       # S4 CPU workload
        options['cpu'] = ['--targets', 'validation,microstructure', '--microstructure-stages', 'dti,freewater,noddi,dki',
                          '--dki-fit-method', 'WLS', '--microstructure-free-water-n-jobs', '1']
    if 'hybrid' in routes:    # S4 hybrid workload; no --mrtrix-seed (tractography is unseeded)
        # SPEC-GAP: DDParcel has no device flag; it follows CUDA_VISIBLE_DEVICES like the other GPU stages.
        options['hybrid'] = ['--targets', 'validation,segmentation,parcellation,surface,microstructure,tractography,connectome',
                '--microstructure-stages', 'dti,noddi', '--tractography-backend', 'csd_act',
                '--select-streamlines', '1000000', '--connectome-backend', 'mrtrix',
                '--ddseg-device', 'cuda:0', '--surface-device', 'cuda:0', '--surface-precision', 'fp32']
    if 'anatomy' in routes:
        options['anatomy'] = ['--targets', 'validation,segmentation,parcellation,surface,microstructure',
                '--microstructure-stages', 'dti', '--ddseg-device', 'cuda:0',
                '--surface-device', 'cuda:0', '--surface-precision', 'fp32']
    if 'tractography' in routes:
        options['tractography'] = ['--targets', 'validation,segmentation,parcellation,microstructure,tractography,bundles,connectome',
                '--microstructure-stages', 'dti', '--tractography-backend', 'dti_det',
                '--bundle-backend', 'wmql,tractcloud', '--bundle-tractography-backend', 'dti_det',
                '--connectome-backend', 'python', '--wmql-query-preset', 'official',
                '--tract-querier-root', '/data04/chengjin/opt/backend-sources/wmql-5c69c594d489',
                '--tractcloud-root', '/data04/chengjin/opt/backend-sources/tractcloud-94de6275d07d',
                '--ddseg-device', 'cuda:0', '--tractcloud-device', 'cuda']
    if 'regional_ifw' in routes:
        options['regional_ifw'] = ['--targets', 'validation,segmentation,parcellation,microstructure',
                '--microstructure-stages', 'dti,freewater,noddi,ifw', '--ifw-echo-time-ms', '89.5',
                '--microstructure-free-water-n-jobs', '1', '--ddseg-device', 'cuda:0']
    if 'end_to_end' in routes:
        options['end_to_end'] = ['--targets', 'validation,segmentation,parcellation,surface,microstructure,tractography,bundles,connectome',
                '--microstructure-stages', 'dti,freewater,noddi,ifw', '--ifw-echo-time-ms', '89.5',
                '--microstructure-free-water-n-jobs', '1', '--tractography-backend', 'dti_det',
                '--bundle-backend', 'wmql,tractcloud', '--bundle-tractography-backend', 'dti_det',
                '--connectome-backend', 'python', '--wmql-query-preset', 'official',
                '--tract-querier-root', '/data04/chengjin/opt/backend-sources/wmql-5c69c594d489',
                '--tractcloud-root', '/data04/chengjin/opt/backend-sources/tractcloud-94de6275d07d',
                '--ddseg-device', 'cuda:0', '--surface-device', 'cuda:0', '--surface-precision', 'fp32',
                '--tractcloud-device', 'cuda']
    return options


def validate_prepare(args):
    """All checks that need no side effect; runs before anything is created."""
    if args.count < 10 or args.repeats < 1 or args.jobs < 1 or args.cpus < 1 or args.memory_gb <= 0:
        raise ValueError('Use at least 10 subjects, positive repeats and resource budgets')
    if args.jobs*max(args.threads) > args.cpus or min(args.threads) < 1:
        raise ValueError('Subject-level jobs x threads must fit the same CPU budget')
    safe_component(args.output.name, 'campaign name')
    if args.cpu_list is not None:
        parse_cpu_list(args.cpu_list, args.cpus)
    if args.gpu is not None and args.gpu < 0:
        raise ValueError('--gpu must be a non-negative device index')
    if args.kill_memory_gb is not None and args.kill_memory_gb < args.memory_gb:
        raise ValueError('The RSS kill limit must not be below the declared batch memory')
    if 'tracts' in args.routes and (not args.ukf_executable or not args.ukf_executable.is_file()):
        raise ValueError('Tract campaign requires an explicit installed --ukf-executable')


def stage_snapshot(install):
    """Stage the extension once and copy the benchmark scripts into it."""
    import shutil
    from scripts.stage_slicer_extension import stage_extension
    staged = stage_extension(install, slicer_version='Slicer-5.10', clean=False, link_mode='copy')
    snapshot = Path(staged['module_dir'])
    shutil.copy2(__file__, snapshot/'scripts/benchmark_cohort_workflows.py')
    for name in ('benchmark_cohort_compare.py', 'benchmark_cohort_harness.py', 'benchmark_subject_worker.py'):
        shutil.copy2(ROOT/'scripts'/name, snapshot/'scripts'/name)
    return snapshot


def external_assets(args, routes):
    assets = {}
    for root in (args.models/name for name in MODEL_DIRECTORIES) if set(routes) & {'hybrid', 'anatomy', 'tractography', 'regional_ifw', 'end_to_end'} else ():
        if not root.is_dir():
            raise ValueError('Required model directory absent: '+str(root))
        for resource in sorted(root.rglob('*')):
            if resource.is_file():
                assets[str(resource.resolve())] = sha(resource)
    if 'tracts' in routes:
        from tractography.ukf import executable_identity
        identity = executable_identity(args.ukf_executable)
        assets[identity['path']] = identity['sha256']
        assets.update(identity['companion_libraries'])
        for root in (args.models/'segmentation', args.models/'parcellation', args.models/'Resources', args.atlas):
            if not root.is_dir():
                raise ValueError('Required model/atlas directory absent: '+str(root))
            for resource in sorted(root.rglob('*')):
                if resource.is_file():
                    assets[str(resource.resolve())] = sha(resource)
        for resource in sorted(args.wma.rglob('*.py')):
            if '__pycache__' not in resource.parts:
                assets[str(resource.resolve())] = sha(resource)
    return assets


def make_trials(routes, threads, repeats, policies=None):
    trials = []
    for route in routes:
        for thread_count in threads:
            for repeat in range(repeats):
                order = policies or POLICIES[repeat % 4:] + POLICIES[:repeat % 4]
                for policy in order:
                    trials.append(dict(id=f'{route}-t{thread_count}-r{repeat}-{policy}', route=route, policy=policy,
                                       threads=thread_count, repeat=repeat, status='pending'))
    return trials


def build_record(args, *, rows, skipped, snapshot, routes, trials, common, options, assets,
                 cpu_list=None, gpu=None, kill_memory_gb=None, cpus=None, memory_gb=None, jobs=None, repeats=None,
                 extra=None):
    hashes = {str(path.relative_to(snapshot)): sha(path) for path in snapshot.rglob('*') if path.is_file()}
    packages = {}
    for name in ('numpy','scipy','nibabel','dipy','torch','dmri-amico','vtk'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    cpus = cpus or args.cpus
    memory_gb = memory_gb or args.memory_gb
    affinity = (cpu_list if cpu_list is not None else
                (sorted(os.sched_getaffinity(0))[:cpus] if hasattr(os, 'sched_getaffinity') else None))
    record = dict(version=1, status='prepared', subjects=rows, skipped=skipped, snapshot=str(snapshot),
                  source_hashes=hashes, launcher=str(args.launcher.resolve()), common=common, routes=options,
                  external_asset_hashes=assets, launcher_sha256=sha(args.launcher),
                  environment=dict(python=sys.version, executable=sys.executable, packages=packages,
                                   platform=sys.platform, load_average=list(os.getloadavg()) if hasattr(os,'getloadavg') else None),
                  cpus=cpus, memory_gb=memory_gb, kill_memory_gb=kill_memory_gb or memory_gb,
                  gpu=gpu, jobs=jobs or args.jobs, repeats=repeats or args.repeats, trials=trials,
                  cpu_affinity=affinity,
                  # H1: set for the T3 routes; the Subject arm's --jobs and lease memory are then fixed by H3.
                  automatic_jobs=bool(set(routes) <= set(T3_ROUTES)),
                  notes=['Full input grids; fresh per-trial outputs; shared OS cache is not cleared',
                         'Same active window for subject/stage/adaptive; sequential uses one by definition',
                         'All policies use one model per subject in the primary comparison',
                         'DKI and MK-Curve are separate routes, not equivalent numerical methods',
                         'External atlas/model/backend paths remain read-only; their terms still apply'])
    record.update(extra or {})
    return record


def prepare(args):
    validate_prepare(args)
    rows, skipped = inspect_cohort(args.dataset_root.resolve(), args.count)
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = stage_snapshot(args.output/'install')
    driver = snapshot/'scripts/benchmark_cohort_workflows.py'
    acquisitions = set()
    for row in rows:
        row['sha256'] = {name:sha(Path(row['source'])/name) for name in INPUTS}
        if row['sha256']['data.nii.gz'] in acquisitions:
            raise ValueError('Duplicate DWI content, not a distinct participant')
        acquisitions.add(row['sha256']['data.nii.gz'])
    assets = external_assets(args, args.routes)
    record = build_record(args, rows=rows, skipped=skipped, snapshot=snapshot, routes=args.routes,
                          trials=make_trials(args.routes, args.threads, args.repeats),
                          common=common_arguments(args), options=route_options(args, args.routes), assets=assets,
                          cpu_list=parse_cpu_list(args.cpu_list, args.cpus) if args.cpu_list is not None else None,
                          gpu=args.gpu, kill_memory_gb=args.kill_memory_gb)
    save(args.output/'campaign.json', record)
    print(json.dumps(dict(status=record['status'], subjects=len(rows), trials=len(record['trials']),
                          driver=str(driver), campaign=str(args.output/'campaign.json')), indent=2))


def lease(record, trial):
    """Whole-subject lease and `--jobs` of a trial. H3 fixes both for the T3 routes: Subject = 8 CPUs,
    `--jobs` = cpus // threads (4), lease memory = batch memory / jobs (56 GiB); Adaptive has none."""
    if trial['route'] in T3_ROUTES:
        if trial['policy'] == 'subject':
            jobs = record['cpus'] // trial['threads']
            return trial['threads'], record['memory_gb'] / jobs, jobs
        return None, None, None
    jobs = 1 if trial['policy'] == 'sequential' else record['jobs']
    return trial['threads'], record['memory_gb'] / (record['jobs'] or 1), jobs


def automatic_admission(record, trial, destination):
    """Legacy automatic block for the non-T3 routes (unchanged behaviour)."""
    from types import SimpleNamespace
    from shared.cohort_planning import plan_cohort, _option
    arguments = [*record['common'], *record['routes'][trial['route']]]
    subjects = [SimpleNamespace(subject_id=s['subject'], subject_root=destination/'inputs'/s['subject'])
                for s in record['subjects']]
    plan = plan_cohort(subjects, arguments, adaptive=True, cpu_budget=record['cpus'],
                       memory_gb=record['memory_gb'], stage_cpus=trial['threads'])
    models = int(_option(arguments, '--max-parallel-models', '1'))
    eligible_models = {'microstructure_freewater','microstructure_noddi',
                       'microstructure_dki','microstructure_mkcurve'} & set(plan['selected_stages'])
    models = min(models, max(1, len(eligible_models)))
    peaks = []
    for profile in plan['subject_profiles'].values():
        selected = [profile[n]['memory_gb'] for n in plan['selected_stages'] if n in profile]
        parallel = sorted((profile[n]['memory_gb'] for n in plan['selected_stages']
                           if n in profile and n in {'microstructure_freewater','microstructure_noddi',
                                                     'microstructure_dki','microstructure_mkcurve'}), reverse=True)
        peaks.append(max([*selected, sum(parallel[:models])]))
    whole_cpu = min(record['cpus'], trial['threads'] * models)
    whole_memory = max(peaks)
    jobs = (max(1, min(len(subjects), record['cpus']//whole_cpu, int(record['memory_gb']//whole_memory)))
            if trial['policy']=='subject' else None)
    if trial['policy']=='sequential':
        jobs = 1
    trial['automatic_admission'] = dict(whole_subject_cpu=whole_cpu, whole_subject_memory_gb=whole_memory,
                                        jobs_argument=jobs, manual_subject_limit=False)
    return whole_cpu, whole_memory, jobs


def build_command(record, trial, destination, snapshot, jobs):
    """Batch command of one arm (H3). Adaptive T3 arm: no --jobs and no thread flag; Subject T3 arm: threads 8."""
    t3 = trial['route'] in T3_ROUTES
    threads = str(trial['threads'])
    command = [record['launcher'],'--launch','PythonSlicer',str(snapshot/'scripts/run_batch_pipeline.py'),
               '--batch-root',str(destination/'inputs'),'--output-root',str(destination/'outputs'),
               '--manifest',str(destination/'batch.json'),'--run-name','benchmark',
               '--scheduler',SCHEDULERS[trial['policy']],
               '--batch-cpus',str(record['cpus']),'--batch-memory-gb',str(record['memory_gb'])]
    if not t3:
        command += ['--surface-cpu-threads',threads,'--mrtrix-nthreads',threads]
    command += ['--resource-history',str(destination/'history.json'),
                '--execute',*record['common'],*record['routes'][trial['route']]]
    if t3:
        command.append('--continue-on-error')
    if jobs is not None:
        command += ['--jobs', str(jobs)]
    if trial['policy'] == 'adaptive_tuned':
        profile = Path(trial['compute_profile'])
        if not profile.is_file() or sha(profile) != trial['compute_profile_sha256']:
            raise ValueError('Frozen runtime calibration changed or is missing')
        command += ['--compute-profile', str(profile)]
    elif not (t3 and trial['policy'] == 'adaptive'):
        if t3:
            command += ['--surface-cpu-threads',threads,'--mrtrix-nthreads',threads]
        command += ['--batch-stage-cpus', threads]
    if trial['route'] == 'tracts':
        command += ['--ukf-parameters',json.dumps(dict(numThreads=trial['threads']))]
    return command


def trial_environment(record, trial, base=None):
    environment = dict(os.environ if base is None else base)
    t3 = trial['route'] in T3_ROUTES
    if not (t3 and trial['policy'] == 'adaptive'):
        environment.update({key: str(trial['threads']) for key in THREAD_ENV})
    else:
        # SPEC-GAP: S4 gives the Adaptive arm no thread flags; inherited thread variables are removed so the
        # arm does not depend on the shell that started it.
        for key in THREAD_ENV:
            environment.pop(key, None)
    if t3:
        gpu = record.get('gpu')
        environment['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        # SPEC-GAP: S4 sets the variable per arm GPU; the CPU workload has none, so no device is visible.
        environment['CUDA_VISIBLE_DEVICES'] = '' if gpu is None else str(gpu)
    return environment


def process_start_ticks(pid):
    """Start time of a process in clock ticks (/proc/<pid>/stat field 22); None where /proc is absent."""
    try:
        text = Path(f'/proc/{pid}/stat').read_text()
        return int(text[text.rindex(')')+2:].split()[19])     # fields after the command name start at field 3
    except (OSError, ValueError, IndexError):
        return None


def reap_group(pid, start_ticks=None, require_match=False):
    """Kill whatever is left in a finished arm's process group. Returns True when something was left.

    H-F3: when the stored start time differs from the live process, the id belongs to another process; do nothing.
    H-F3b: run() and pair() differ on a missing leader. In run() the leader is normally gone and the group must
    still be reaped, so an unreadable start time keeps the old behaviour (require_match=False). In pair() the arm
    process already reaped its own trial; the fallback only runs after an abnormal exit, and a missing leader
    there gives no proof that the stored group is still ours. With require_match=True, send no signal unless the
    start time was read and equals start_ticks."""
    if require_match:
        if start_ticks is None or process_start_ticks(pid) != start_ticks:
            return False
    elif start_ticks is not None:
        current = process_start_ticks(pid)
        if current is not None and current != start_ticks:
            return False
    if not hasattr(os, 'killpg'):
        res = subprocess.run(
            ['taskkill', '/F', '/T', '/PID', str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return res.returncode == 0
    try:
        os.killpg(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    try:
        os.killpg(pid, getattr(signal, 'SIGKILL', signal.SIGTERM))
    except (ProcessLookupError, PermissionError):
        pass
    return True


def wait_for_go(ready_file, go_file, cancelled, timeout):
    """Barrier for paired arms: announce readiness, then wait for the pair driver to release both arms."""
    with open(ready_file, 'x'):
        pass
    end = time.monotonic()+timeout
    while not Path(go_file).exists():
        if cancelled.is_set() or time.monotonic() >= end:
            return False
        time.sleep(.2)
    return True


def stop(child):
    try:
        if os.name != 'nt':
            os.killpg(child.pid, signal.SIGTERM)
        else:
            subprocess.run(
                ['taskkill', '/F', '/T', '/PID', str(child.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            child.terminate()
        child.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if child.poll() is None:
            if os.name != 'nt':
                os.killpg(child.pid, signal.SIGKILL)
            else:
                child.kill()
            child.wait()
    if os.name != 'nt':
        # A launcher may exit before descendants finish handling SIGTERM.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def tail_text(path, limit=64*1024):
    try:
        with open(path, 'rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell()-limit))
            return stream.read().decode('utf-8', 'replace')
    except OSError:
        return ''


def subject_outcomes(record, trial, audit, batch, gpu_rows):
    """H4: one outcome per subject: QC rule 1 result, batch result and failure cause."""
    audited = {row['subject_id']: row for row in (audit or {}).get('subjects', [])}
    results = {row['subject_id']: row for row in (batch or {}).get('results', [])}
    not_launched = set((batch or {}).get('not_launched', []))
    other = max((row['other_mib'] for row in gpu_rows if row.get('other_mib') is not None), default=None)
    outcomes = {}
    for row in record['subjects']:
        name = row['subject']
        found, result = audited.get(name), results.get(name)
        rule1 = ('audit_error' if audit is None else 'no_manifest' if found is None else found['status'])
        outcome = dict(subject=name, rule1=rule1, batch_status=(result or {}).get('status'),
                       returncode=(result or {}).get('returncode'),
                       failed_checks=[c['name'] for c in (found or {}).get('checks', []) if not c['passed']][:20],
                       cause=None)
        if rule1 != 'passed':
            text = ''.join(tail_text(result[key]) for key in ('stderr_log', 'stdout_log') if result and result.get(key))
            if re.search(r'CUDA out of memory|OutOfMemoryError', text):
                # External when other users held GPU memory in the samples (evidence, not a threshold).
                outcome.update(cause='gpu_out_of_memory', external_candidate=bool(other), other_users_peak_mib=other)
            elif trial.get('reason'):
                outcome['cause'] = 'harness_stopped_' + str(trial['reason'])
            elif name in not_launched:
                outcome['cause'] = 'not_launched'
            elif result and result.get('returncode'):
                outcome['cause'] = f"subject_exit_{result['returncode']}"
            else:
                outcome['cause'] = 'qc_rule_1_failed'
        outcomes[name] = outcome
    return outcomes


def audit_trial(record, trial):
    """H4: audit and fingerprint every subject once the batch has exited, whatever its return code."""
    from scripts.audit_scientific_outputs import audit_output_root
    from scripts.benchmark_cohort_compare import scientific_inventory
    from scripts.benchmark_cohort_harness import read_jsonl
    destination = Path(trial['output'])
    names = [row['subject'] for row in record['subjects']]
    options = dict(require_dti=True, require_tractography=trial['route'] in ('tracts', 'hybrid'), run_name='benchmark')
    audit = None
    try:
        audit = audit_output_root(destination/'outputs', min_subjects=len(names), **options)
    except Exception as exc:
        trial['audit_error'] = f'{type(exc).__name__}: {exc}'
        try:     # one unreadable subject must not hide the others
            rows = []
            for name in names:
                try:
                    rows.extend(audit_output_root(destination/'outputs'/name, min_subjects=1, **options)['subjects'])
                except Exception:
                    pass
            failed = [row for row in rows if row['status'] != 'passed']
            audit = dict(status='passed' if len(rows) >= len(names) and not failed else 'failed',
                         summary=dict(passed_subjects=len(rows)-len(failed)), subjects=rows)
        except Exception:
            audit = None
    if audit is not None:
        save(destination/'independent_qc.json', audit)
        trial.update(qc_report=str(destination/'independent_qc.json'), qc_status=audit['status'],
                     qc_passed_subjects=audit['summary']['passed_subjects'])
    else:
        trial.update(qc_status='audit_error')
    inventories, fingerprint_errors = {}, {}
    for name in names:
        try:
            inventories[name] = scientific_inventory(destination/'outputs'/name)
        except Exception as exc:
            fingerprint_errors[name] = f'{type(exc).__name__}: {exc}'
    save(destination/'scientific_fingerprints.json', inventories)
    batch = None
    try:
        batch = json.loads((destination/'batch.json').read_text())
    except (OSError, ValueError):
        pass
    outcomes = subject_outcomes(record, trial, audit, batch, read_jsonl(destination/'gpu_samples.jsonl'))
    for name, message in fingerprint_errors.items():
        outcomes[name]['fingerprint_error'] = message
    save(destination/'subject_outcomes.json', outcomes)
    trial.update(subject_outcomes=str(destination/'subject_outcomes.json'),
                 rule1_passed_subjects=sum(row['rule1'] == 'passed' for row in outcomes.values()))
    return outcomes


def run(args):
    folder = args.output.resolve()
    path = folder/'campaign.json'
    record = json.loads(path.read_text())
    if record['status'] != 'prepared':
        raise ValueError('Campaign already started; existing evidence must not be overwritten')
    from shared.cohort_affinity import cpu_allocation, isolated_command, affinity_violations
    allocation = cpu_allocation(record['cpus'], record.get('cpu_affinity'))
    isolated_command([sys.executable, '-V'], allocation)  # Check taskset before any trial starts.
    snapshot = Path(record['snapshot'])
    if any(sha(snapshot/name) != value for name,value in record['source_hashes'].items()):
        raise ValueError('Frozen implementation changed')
    for trial in record['trials']:
        if any(sha(Path(trial['output'])/name)!=value for name,value in trial.get('inherited_evidence',{}).items()):
            raise ValueError('Inherited completed evidence changed')
        if 'snapshot' in trial and any(sha(Path(trial['snapshot'])/name)!=value
                                      for name,value in trial['snapshot_hashes'].items()):
            raise ValueError('Candidate execution snapshot changed')
    for row in record['subjects']:
        if any(signature(Path(row['source'])/name) != value for name,value in row['signatures'].items()):
            raise ValueError('Source inputs changed')
    if sha(record['launcher']) != record['launcher_sha256'] or any(sha(name) != value
            for name,value in record['external_asset_hashes'].items()):
        raise ValueError('External launcher/model/atlas/backend changed')
    cancelled = threading.Event()
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, lambda *_: cancelled.set())   # the pair driver stops an arm this way
    if args.ready_file:
        if not wait_for_go(args.ready_file, args.go_file, cancelled, args.go_timeout):
            return 2
    record.update(status='running', driver_pid=os.getpid(), started_at=time.time())
    record['execution_harness'] = dict(path=str(Path(__file__).resolve()),sha256=sha(__file__))
    save(path,record)
    deadline = time.monotonic()+args.max_hours*3600
    for trial in record['trials']:
        if trial['status'] == 'completed':
            continue
        if trial['status'] != 'pending':
            raise ValueError('Incomplete trial retained; create a separate campaign rather than overwriting results')
        if time.monotonic() >= deadline:
            record['status'] = 'time_budget_exhausted'
            save(path,record)
            return 2
        destination = folder/'trials'/trial['id']
        destination.mkdir(parents=True,exist_ok=False)
        for subject in record['subjects']:
            inputs = destination/'inputs'/subject['subject']/'input'
            inputs.mkdir(parents=True)
            for name in INPUTS:
                try:
                    (inputs/name).symlink_to(Path(subject['source'])/name)
                except OSError:
                    shutil.copy2(Path(subject['source'])/name, inputs/name)
        execution_snapshot=Path(trial.get('snapshot',snapshot))
        t3 = trial['route'] in T3_ROUTES
        if record.get('automatic_jobs', False) and not t3:
            whole_cpu, whole_memory, jobs = automatic_admission(record, trial, destination)
        else:
            whole_cpu, whole_memory, jobs = lease(record, trial)
        command = build_command(record, trial, destination, execution_snapshot, jobs)
        environment = trial_environment(record, trial)
        from shared.cohort_schedule import create_schedule
        from shared.cohort_io import GPU_ENV
        if not (t3 and trial['policy'] == 'adaptive'):
            # SPEC-GAP: S4 lists the GPU FIFO gate for the Subject arm only; the Adaptive T3 arm uses its own GPU slot.
            guard = create_schedule(destination/'gpu_guard', cpu_budget=record['cpus'], memory_gb=record['memory_gb'],
                                    stage_cpus=trial['threads'], stage_memory_gb=1, gpu_slots=1, cpu_affinity=allocation)
            environment[GPU_ENV] = str(guard)
        if trial['policy'] in ('sequential','subject'):
            from shared.cohort_schedule import ENV
            environment[ENV] = str(create_schedule(destination/'subject_guard', cpu_budget=record['cpus'],
                memory_gb=record['memory_gb'], stage_cpus=whole_cpu,
                stage_memory_gb=whole_memory, cpu_affinity=allocation))
            environment['SLICERDSPACE_STAGE_CPUS'] = str(whole_cpu)
            command += ['--pipeline-script',str(execution_snapshot/'scripts/benchmark_subject_worker.py')]
        execution_command = isolated_command(command, allocation)
        trial.update(status='running', command=command, execution_command=execution_command,
                     started_at=time.time(), output=str(destination))
        save(path,record)
        kill_limit = record.get('kill_memory_gb', record['memory_gb'])*1024**3     # H2: separate from the batch budget
        with (destination/'stdout.log').open('w') as stdout, (destination/'stderr.log').open('w') as stderr:
            child = subprocess.Popen(execution_command,env=environment,stdout=stdout,stderr=stderr,start_new_session=True)
            telemetry, samplers = None, []
            reason = None
            affinity_checks, placement_errors, last_check = 0, [], 0.
            try:
                # H-F4: everything after Popen sits inside the cleanup, so a failed save cannot leave an unwatched child.
                trial['child_pid'] = child.pid
                trial['child_start_ticks'] = process_start_ticks(child.pid)
                save(path,record)
                telemetry = CohortTelemetry(destination/'resources.jsonl').start()
                if t3:
                    samplers.append(Sampler(destination/'host_samples.jsonl', HOST_SAMPLE_SECONDS, HostReader(
                        allocation, tree_cpu_seconds=lambda: telemetry.monitor.latest.get('cpu_seconds'),
                        data_path=destination, nfs_path=Path(record['subjects'][0]['source']).resolve())).start())
                    if record.get('gpu') is not None:
                        samplers.append(Sampler(destination/'gpu_samples.jsonl', GPU_SAMPLE_SECONDS, make_gpu_reader(
                            record['gpu'], lambda: telemetry.monitor.latest.get('pids', []))).start())
                while child.poll() is None:
                    if time.monotonic()-last_check >= 10:
                        last_check = time.monotonic()
                        placement_errors = affinity_violations(child.pid, allocation)
                        affinity_checks += 1
                        if placement_errors:
                            reason = 'cpu_affinity_escape'
                    if time.monotonic() >= deadline:
                        reason = 'time_budget_exhausted'
                    if telemetry.monitor.peak > kill_limit:
                        reason = 'memory_budget_exceeded'
                    if cancelled.is_set():
                        reason = 'cancelled'
                    if reason:
                        stop(child)
                        break
                    time.sleep(1)
            except (KeyboardInterrupt, InterruptedError):
                reason = 'cancelled'
                stop(child)
            finally:
                if child.poll() is None:
                    stop(child)
                trial['leftover_processes_killed'] = reap_group(child.pid, trial.get('child_start_ticks'))
                trial['samplers'] = [sampler.finish() for sampler in samplers]
                measured = telemetry.finish('completed' if child.returncode == 0 else 'failed') if telemetry else {}
                measured.update(cpu_affinity_checks=affinity_checks, cpu_affinity_violations=placement_errors)
            trial.update(status='completed' if child.returncode == 0 and not reason else 'failed',
                         returncode=child.returncode, reason=reason, ended_at=time.time(), measurement=measured)
        # Process completion is not scientific acceptance; independent audit follows (H4: also after a failure).
        if not args.defer_qc:
            audit_trial(record, trial)
            if trial['status']=='completed' and not record.get('paired'):
                trial['comparison_status'] = 'pending_reference'
                update_comparisons(record, folder)
        save(path,record)
        if trial['status'] != 'completed':
            record['status']='failed'
            save(path,record)
            return 2
    if args.defer_qc:
        record['status'] = 'completed_qc_deferred'
        save(path,record)
        return 0
    if not record.get('paired'):
        update_comparisons(record, folder)
    record['inputs_unchanged'] = all(sha(Path(row['source'])/name)==value for row in record['subjects']
                                   for name,value in row['sha256'].items())
    record['status']='passed' if record['inputs_unchanged'] and all(row.get('qc_status')=='passed' and row.get('comparison_status')=='passed'
                                    and not timing_issues(record, row)
                                    for row in record['trials']) else 'completed_pending_acceptance'
    save(path,record)
    return 0


def qc(args):
    """Run the deferred QC (H4) of a campaign whose timed process tree has stopped."""
    path = args.output.resolve()/'campaign.json'
    record = json.loads(path.read_text())
    if record['status'] not in ('completed_qc_deferred', 'failed', 'time_budget_exhausted'):
        raise ValueError('QC runs only on a finished campaign: ' + str(record['status']))
    folder = args.output.resolve()
    for trial in record['trials']:
        if trial['status'] in ('completed', 'failed') and 'output' in trial and 'subject_outcomes' not in trial:
            try:
                Path(trial['output']).resolve().relative_to(folder)
            except ValueError:
                trial['qc_skipped'] = 'inherited'      # H-F2: never write into the folder of another campaign
                continue
            audit_trial(record, trial)
    record['qc_done'] = True
    save(path, record)
    return 0


# ---------------------------------------------------------------- H7: pair

def link_subjects(args):
    """Build the folder of symlinks the harness reads: the first `count` eligible IDs after `skip` IDs."""
    id_dir, dataset, count = args.id_list_dir, args.dataset_root.resolve(), args.count
    if not id_dir.is_dir() or not dataset.is_dir() or count < 1 or args.skip < 0:
        raise ValueError('Check --id-list-dir, --dataset-root, --count and --skip')
    names = []
    for entry in sorted(os.listdir(id_dir)):
        try:
            names.append(safe_component(entry, 'subject id'))
        except ValueError:
            continue
    rows, skipped = inspect_cohort(dataset, count, names=names[args.skip:])
    os.mkdir(args.output)
    for row in rows:
        os.symlink(dataset/row['subject'], Path(args.output)/row['subject'])
    print(json.dumps(dict(linked=[row['subject'] for row in rows], skipped=skipped), indent=2))
    return 0


def validate_pair(args):
    """Every argument check, before anything is created or started."""
    safe_component(args.output.name, 'pair name')
    if not args.output.parent.is_dir() or args.output.exists():
        raise ValueError('The pair folder must be new, inside an existing folder')
    if args.workload not in T3_ROUTES or args.repeat not in (1, 2) or args.attempts < 1 or args.retry_wait_seconds < 0:
        raise ValueError('Bad --workload, --repeat, --attempts or --retry-wait-seconds')
    if args.count < 10 or args.max_hours <= 0:
        raise ValueError('Use at least 10 subjects and a positive time budget')
    if not 0 < getattr(args, 'max_busy_percent', MAX_BUSY_PERCENT) <= 100:
        raise ValueError('--max-busy-percent must be above 0 and at most 100')
    for label, item, kind in (('--dataset-root', args.dataset_root, 'dir'), ('--models', args.models, 'dir'),
                              ('--mrtrix', args.mrtrix, 'dir'), ('--launcher', args.launcher, 'file')):
        if not (item.is_dir() if kind == 'dir' else item.is_file()):
            raise ValueError(f'{label} does not exist: {item}')
    if args.workload != 'cpu':          # fail at time zero, not after the 5-minute launch sampling
        for name in MODEL_DIRECTORIES:
            if not (args.models/name).is_dir():
                raise ValueError(f'Required model directory absent: {args.models/name}')
    previous = None
    if args.previous_pair is not None:
        assignment = json.loads((args.previous_pair/'pair.json').read_text()).get('assignment')
        if assignment is None or args.repeat != 2:
            raise ValueError('--previous-pair needs a pair.json with an assignment and --repeat 2')
        previous = dict(subject_gpu=assignment['subject']['gpu'], adaptive_gpu=assignment['adaptive']['gpu'])
    return previous


def doctor_command(args, options, subject_root, subject_id, report):
    return [str(args.launcher.resolve()), '--launch', 'PythonSlicer', str(ROOT/'shared/run_subject_pipeline_cli.py'),
            '--subject-root', str(subject_root), '--subject-id', subject_id, '--doctor',
            '--doctor-output-dir', str(report), *common_arguments(args, print_manifest_only=False), *options]


def pair(args, *, topology=None, sampler=sample_window, runner=run_bounded, sleep=time.sleep):
    previous = validate_pair(args)
    topology = topology or read_topology()
    rows, skipped = inspect_cohort(args.dataset_root.resolve(), args.count)
    options = route_options(args, [args.workload])[args.workload]
    claim_folder(args.output)
    pair_dir = args.output.resolve()
    for name in ('launch', 'doctor'):
        os.mkdir(pair_dir/name)
    first = rows[0]
    doctor_root = pair_dir/'doctor'/'subject'/first['subject']
    (doctor_root/'input').mkdir(parents=True)
    for name in INPUTS:
        try:
            (doctor_root/'input'/name).symlink_to(Path(first['source'])/name)
        except OSError:
            shutil.copy2(Path(first['source'])/name, doctor_root/'input'/name)
    hybrid = args.workload != 'cpu'
    max_busy = getattr(args, 'max_busy_percent', MAX_BUSY_PERCENT)
    state = dict(workload=args.workload, repeat=args.repeat, started_at=time.time(), attempts=[],
                 max_busy_percent=max_busy)

    def attempt(number):
        busy, gpu_samples, gpu_errors = sampler(pair_dir/'launch', number)
        decision = decide_launch(busy, topology, gpu_samples, shutil.disk_usage(pair_dir).free, hybrid=hybrid,
                                 max_busy_percent=max_busy)
        outcome = dict(ok=decision['ok'], reasons=list(decision['reasons']), decision=decision, gpu_errors=gpu_errors[:5])
        if hybrid and not gpu_samples:
            outcome.update(ok=False, reasons=outcome['reasons'] + ['nvidia-smi gave no sample'])
        if outcome['ok']:
            outcome['assignment'] = assign_arms(decision, args.repeat, previous=previous)
            environment = {**os.environ, 'CUDA_DEVICE_ORDER': 'PCI_BUS_ID'}
            if hybrid:
                environment['CUDA_VISIBLE_DEVICES'] = str(outcome['assignment']['subject']['gpu'])
            try:     # S4 "Both": --doctor once; a blocker counts as a failed attempt
                code, stdout, stderr = runner(doctor_command(args, options, doctor_root, first['subject'],
                                              pair_dir/'doctor'/f'report_{number}'), timeout=900, env=environment)
                with (pair_dir/'doctor'/f'doctor_{number}.json').open('x') as stream:
                    stream.write(stdout)
                document = parse_doctor_output(stdout)
                blockers = doctor_blockers(document)
                outcome['doctor'] = dict(returncode=code, blocker_count=blockers, document=f'doctor/doctor_{number}.json')
                if code != 0 or blockers > 0:
                    outcome.update(ok=False, reasons=outcome['reasons'] + [f'doctor: rc {code}, {blockers} blocker(s)'])
            except Exception as exc:
                outcome.update(ok=False, reasons=outcome['reasons'] + [f'doctor failed: {type(exc).__name__}: {exc}'])
        return outcome

    def persist(attempts):
        state['attempts'] = [{key: value for key, value in item.items() if key != 'decision'} |
                             dict(decision=item['decision']) for item in attempts]
        save(pair_dir/'pair.json', state)

    try:
        chosen, _ = run_launch_rule(attempt, max_attempts=args.attempts, wait_seconds=args.retry_wait_seconds,
                                    sleep=sleep, save=persist)
    except LaunchRefused as refusal:
        persist(refusal.attempts)
        state['status'] = 'launch_refused'
        save(pair_dir/'pair.json', state)
        print(json.dumps(dict(status='launch_refused', attempts=len(refusal.attempts),
                              last_reasons=refusal.attempts[-1]['reasons'])))
        return 2
    state.update(assignment=chosen['assignment'], doctor=chosen['doctor'], launch_attempt=chosen['attempt'])
    save(pair_dir/'pair.json', state)
    # Two one-trial campaigns from one staged snapshot (same source_hashes).
    snapshot = stage_snapshot(pair_dir/'install')
    for row in rows:
        row['sha256'] = {name: sha(Path(row['source'])/name) for name in INPUTS}
    assets = external_assets(args, [args.workload])
    common = common_arguments(args)
    campaigns = {}
    for policy in ('subject', 'adaptive'):
        arm = chosen['assignment'][policy]
        campaign = pair_dir/policy
        os.mkdir(campaign)
        record = build_record(args, rows=rows, skipped=skipped, snapshot=snapshot, routes=[args.workload],
                              trials=make_trials([args.workload], [ARM_THREADS], 1, [policy]), common=common,
                              options={args.workload: options}, assets=assets, cpu_list=arm['cpus'], gpu=arm['gpu'],
                              kill_memory_gb=KILL_MEMORY_GB, cpus=ARM_CPUS, memory_gb=BATCH_MEMORY_GB,
                              jobs=SUBJECT_JOBS, repeats=1, extra=dict(paired=True, pair=str(pair_dir)))
        save(campaign/'campaign.json', record)
        campaigns[policy] = campaign
    driver = snapshot/'scripts/benchmark_cohort_workflows.py'
    go_file = pair_dir/'go'
    processes = {}
    status = 2
    try:
        for policy, campaign in campaigns.items():
            with (campaign/'run.log').open('x') as log:
                processes[policy] = subprocess.Popen(
                    [sys.executable, str(driver), 'run', '--output', str(campaign), '--defer-qc',
                     '--ready-file', str(campaign/'ready'), '--go-file', str(go_file),
                     '--go-timeout', '3600', '--max-hours', str(args.max_hours)],
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        end = time.monotonic()+7200      # both arms verify the frozen snapshot, then wait at the barrier
        while not all((campaigns[p]/'ready').exists() for p in campaigns):
            if time.monotonic() >= end or any(proc.poll() is not None for proc in processes.values()):
                raise RuntimeError('An arm did not reach the start barrier')
            sleep(1)
        with go_file.open('x'):
            pass
        state['go_at'] = time.time()
        end = time.monotonic()+args.max_hours*3600+3600
        while any(proc.poll() is None for proc in processes.values()):
            if time.monotonic() >= end:
                raise TimeoutError('Arms exceeded the time budget')
            sleep(5)
        state['run_returncodes'] = {p: proc.returncode for p, proc in processes.items()}
    except BaseException as exc:
        state.update(status='aborted', error=f'{type(exc).__name__}: {exc}')
        save(pair_dir/'pair.json', state)
        raise
    finally:
        for proc in processes.values():       # abnormal exit: stop each arm, then reap its batch group
            if proc.poll() is None:
                kill_group(proc, sig=signal.SIGTERM, wait=120)
                if proc.poll() is None:
                    kill_group(proc)
        for campaign in campaigns.values():
            try:
                for trial in json.loads((campaign/'campaign.json').read_text())['trials']:
                    if trial.get('status') == 'running' and trial.get('child_pid'):
                        reap_group(trial['child_pid'], trial.get('child_start_ticks'), require_match=True)
            except (OSError, ValueError, KeyError):
                pass
    # Both timed trees have stopped; only now run QC and the comparison.
    from scripts.benchmark_cohort_compare import compare_pair
    state['qc_errors'] = {}
    for policy, campaign in campaigns.items():
        try:
            qc(argparse.Namespace(output=campaign))
        except Exception as exc:          # one arm's QC problem must not hide the other arm's evidence
            state['qc_errors'][policy] = f'{type(exc).__name__}: {exc}'
    try:
        report = compare_pair(campaigns['subject'], campaigns['adaptive'], pair_dir/'pair_report.json')
    except Exception as exc:
        state.update(status='compare_failed', error=f'{type(exc).__name__}: {exc}')
        save(pair_dir/'pair.json', state)
        print(json.dumps(dict(status='compare_failed', pair=str(pair_dir), error=state['error'])))
        return 2
    state.update(status='done', report='pair_report.json', flagged=report['flagged'], ratio=report['throughput_ratio'])
    save(pair_dir/'pair.json', state)
    print(json.dumps(dict(status='done', pair=str(pair_dir), ratio=report['throughput_ratio'], flagged=report['flagged'])))
    return 0 if all(code == 0 for code in state['run_returncodes'].values()) else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','select','pair','qc','compare','link-subjects'))
    parser.add_argument('--output',type=Path)
    parser.add_argument('--dataset-root',type=Path,default=Path('/dataset/nfs_share01/HCP/HCP-YA/3T/Diffusion_Preprocessed'))
    parser.add_argument('--count',type=int,default=12)
    parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--cpus',type=int,default=32)
    parser.add_argument('--memory-gb',type=float,default=96)
    parser.add_argument('--kill-memory-gb',type=float,help='RSS kill limit; default is --memory-gb')
    parser.add_argument('--cpu-list',help='Explicit logical CPUs, for example 4-19,60-75; must match --cpus')
    parser.add_argument('--gpu',type=int,help='GPU index in PCI bus order for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--threads',type=int,nargs='+',default=[4])
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--routes',nargs='+',choices=ROUTES,default=['dki','mkcurve','tracts'])
    parser.add_argument('--max-hours',type=float,default=72)
    parser.add_argument('--launcher',type=Path,default=Path('/data/chengjin/software/run-slicer.sh'))
    parser.add_argument('--models',type=Path,default=ROOT.parent/'SlicerDDSurfer-models')
    parser.add_argument('--mrtrix',type=Path,default=ROOT.parent/'software/mrtrix3_env/bin')
    parser.add_argument('--wma',type=Path,default=ROOT/'third_party/whitematteranalysis')
    parser.add_argument('--atlas',type=Path,default=ROOT.parent/'SlicerDDSurfer-test/resources/ORG-Atlases')
    parser.add_argument('--ukf-executable',type=Path)
    parser.add_argument('--from-campaign',type=Path)
    parser.add_argument('--policies',nargs='+',choices=ALL_POLICIES,default=['subject','adaptive'])
    parser.add_argument('--reference-policy',choices=POLICIES,default='subject')
    parser.add_argument('--defer-qc',action='store_true',help='run: skip QC; the pair driver runs it after both arms stop')
    parser.add_argument('--ready-file',type=Path,help='run: announce readiness here, then wait for --go-file')
    parser.add_argument('--go-file',type=Path)
    parser.add_argument('--go-timeout',type=float,default=3600)
    parser.add_argument('--workload',choices=T3_ROUTES,help='pair: cpu or hybrid')
    parser.add_argument('--repeat',type=int,default=1,help='pair: 1 (Subject on node0) or 2 (swapped)')
    parser.add_argument('--previous-pair',type=Path,help='pair repeat 2: repeat 1 folder, to swap GPUs')
    parser.add_argument('--attempts',type=int,default=MAX_ATTEMPTS)
    parser.add_argument('--retry-wait-seconds',type=float,default=RETRY_WAIT_SECONDS)
    parser.add_argument('--max-busy-percent',type=float,default=MAX_BUSY_PERCENT,
                        help='pair: launch-rule CPU limit on each chosen set (design value 25)')
    parser.add_argument('--subject-campaign',type=Path,help='compare: Subject-arm campaign folder')
    parser.add_argument('--adaptive-campaign',type=Path,help='compare: Adaptive-arm campaign folder')
    parser.add_argument('--id-list-dir',type=Path,default=Path('/dataset/nfs_share01/100HCP'))
    parser.add_argument('--skip',type=int,default=1,help='link-subjects: leading IDs to leave out (the M1 subject)')
    args=parser.parse_args()
    if args.action != 'compare' and args.output is None:
        parser.error('--output is required')
    if args.action=='select':
        if not args.from_campaign:
            parser.error('select requires --from-campaign')
        return select_campaign(args.from_campaign,args.output,policies=args.policies,routes=args.routes,
                               reference_policy=args.reference_policy) and 0
    if args.action=='pair':
        if args.workload is None:
            parser.error('pair requires --workload')

        def terminate(*_):          # a SIGTERM must stop both arms through the cleanup in `pair`
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, terminate)
        try:
            return pair(args)
        except LaunchRefused:
            return 2
    if args.action=='link-subjects':
        return link_subjects(args)
    if args.action=='qc':
        return qc(args)
    if args.action=='compare':
        if not (args.subject_campaign and args.adaptive_campaign and args.output):
            parser.error('compare requires --subject-campaign, --adaptive-campaign and --output')
        from scripts.benchmark_cohort_compare import compare_pair
        compare_pair(args.subject_campaign, args.adaptive_campaign, args.output)
        return 0
    return prepare(args) if args.action=='prepare' else run(args)


if __name__=='__main__':
    raise SystemExit(main())
