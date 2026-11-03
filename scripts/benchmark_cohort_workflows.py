#!/usr/bin/env python3
"""Frozen-source full-volume cohort campaign; never substitutes ROI or duplicate subjects."""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.cohort_telemetry import CohortTelemetry

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


def inspect_cohort(root, count):
    import nibabel as nib
    import numpy as np
    rows, skipped, real_sources = [], [], set()
    for directory in sorted(root.iterdir()):
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


def update_comparisons(record, folder):
    from scripts.benchmark_cohort_compare import compare_inventories
    reports = []
    for trial in record['trials']:
        reference = comparison_reference(record, trial)
        if trial['status']!='completed' or not reference:
            continue
        left = Path(reference['output'])/'scientific_fingerprints.json'
        right = Path(trial['output'])/'scientific_fingerprints.json'
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
        accepted = trial.get('qc_status')==reference.get('qc_status')=='passed' and trial['comparison_status']=='passed'
        before = reference['measurement']['execution_seconds']
        after = trial['measurement']['execution_seconds']
        reports.append(dict(route=trial['route'],reference=reference['id'],candidate=trial['id'],
            status='passed' if accepted else 'pending_acceptance',reference_minutes=before/60,candidate_minutes=after/60,
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
        if fixed and tuned.get('qc_status')=='passed':
            before, after = (row['measurement']['execution_seconds'] for row in (fixed,tuned))
            incremental.append(dict(reference=fixed['id'],candidate=tuned['id'],route=tuned['route'],
                speed_ratio=before/after,reduction_percent=100*(1-after/before),
                effect='Runtime configuration selection with the same latest scheduler and cohort budgets'))
    save(folder/'configuration_comparison.json',dict(comparisons=incremental))


def prepare(args):
    if args.count < 10 or args.repeats < 1 or args.jobs < 1 or args.cpus < 1 or args.memory_gb <= 0:
        raise ValueError('Use at least 10 subjects, positive repeats and resource budgets')
    if args.jobs*max(args.threads) > args.cpus or min(args.threads) < 1:
        raise ValueError('Subject-level jobs x threads must fit the same CPU budget')
    rows, skipped = inspect_cohort(args.dataset_root.resolve(), args.count)
    if 'tracts' in args.routes and (not args.ukf_executable or not args.ukf_executable.is_file()):
        raise ValueError('Tract campaign requires an explicit installed --ukf-executable')
    args.output.mkdir(parents=True, exist_ok=False)
    from scripts.stage_slicer_extension import stage_extension
    staged = stage_extension(args.output/'install', slicer_version='Slicer-5.10', clean=False, link_mode='copy')
    snapshot = Path(staged['module_dir'])
    import shutil
    driver = snapshot/'scripts/benchmark_cohort_workflows.py'
    shutil.copy2(__file__, driver)
    shutil.copy2(ROOT/'scripts/benchmark_cohort_compare.py', snapshot/'scripts/benchmark_cohort_compare.py')
    shutil.copy2(ROOT/'scripts/benchmark_subject_worker.py', snapshot/'scripts/benchmark_subject_worker.py')
    acquisitions = set()
    for row in rows:
        row['sha256'] = {name:sha(Path(row['source'])/name) for name in INPUTS}
        if row['sha256']['data.nii.gz'] in acquisitions:
            raise ValueError('Duplicate DWI content, not a distinct participant')
        acquisitions.add(row['sha256']['data.nii.gz'])
    assets = {}
    if 'tracts' in args.routes:
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
    hashes = {str(path.relative_to(snapshot)): sha(path) for path in snapshot.rglob('*') if path.is_file()}
    common = ['--no-user-config', '--brain-mask-backend', 'provided', '--strict-environment',
              '--model-assets-dir', str(args.models.resolve()), '--mrtrix-bin-dir', str(args.mrtrix.resolve()),
              '--max-parallel-models', '1', '--print-manifest-only']
    route_options = dict(
        dki=['--targets','validation,microstructure','--microstructure-stages','dti,dki','--dki-fit-method','WLS'],
        mkcurve=['--targets','validation,microstructure','--microstructure-stages','dti,dki','--dki-fit-method','MKCURVE'],
        tracts=['--targets','validation,segmentation,parcellation,microstructure,tractography,bundles',
                '--microstructure-stages','dti','--tractography-backend','csd_act,ukf',
                '--bundle-backend','wma','--bundle-tractography-backend','ukf',
                '--select-streamlines','1000000','--mrtrix-seed','20261103',
                '--wma-root', str(args.wma.resolve()), '--wma-atlas-dir', str(args.atlas.resolve()),
                '--slicer-executable', str(args.launcher.resolve()),
                '--ukf-executable', str(args.ukf_executable.resolve()) if args.ukf_executable else ''])
    trials = []
    for route in args.routes:
        for threads in args.threads:
            for repeat in range(args.repeats):
                order = POLICIES[repeat % 4:] + POLICIES[:repeat % 4]
                for policy in order:
                    trials.append(dict(id=f'{route}-t{threads}-r{repeat}-{policy}', route=route, policy=policy,
                                       threads=threads, repeat=repeat, status='pending'))
    packages = {}
    for name in ('numpy','scipy','nibabel','dipy','torch','dmri-amico','vtk'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    record = dict(version=1, status='prepared', subjects=rows, skipped=skipped, snapshot=str(snapshot),
                  source_hashes=hashes, launcher=str(args.launcher.resolve()), common=common, routes=route_options,
                  external_asset_hashes=assets, launcher_sha256=sha(args.launcher),
                  environment=dict(python=sys.version, executable=sys.executable, packages=packages,
                                   platform=sys.platform, load_average=list(os.getloadavg()) if hasattr(os,'getloadavg') else None),
                  cpus=args.cpus, memory_gb=args.memory_gb, jobs=args.jobs, repeats=args.repeats, trials=trials,
                  cpu_affinity=sorted(os.sched_getaffinity(0))[:args.cpus] if hasattr(os,'sched_getaffinity') else None,
                  notes=['Full input grids; fresh per-trial outputs; shared OS cache is not cleared',
                         'Same active window for subject/stage/adaptive; sequential uses one by definition',
                         'All policies use one model per subject in the primary comparison',
                         'DKI and MK-Curve are separate routes, not equivalent numerical methods',
                         'External atlas/model/backend paths remain read-only; their terms still apply'])
    save(args.output/'campaign.json', record)
    print(json.dumps(dict(status=record['status'], subjects=len(rows), trials=len(trials),
                          driver=str(driver), campaign=str(args.output/'campaign.json')), indent=2))


def stop(child):
    try:
        if os.name != 'nt':
            os.killpg(child.pid, signal.SIGTERM)
        else:
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


def run(args):
    folder = args.output.resolve()
    path = folder/'campaign.json'
    record = json.loads(path.read_text())
    if record['status'] != 'prepared':
        raise ValueError('Campaign already started; existing evidence must not be overwritten')
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
                (inputs/name).symlink_to(Path(subject['source'])/name)
        scheduler = dict(sequential='subjects', subject='subjects', stage='resources', adaptive='adaptive', adaptive_tuned='adaptive')[trial['policy']]
        execution_snapshot=Path(trial.get('snapshot',snapshot))
        automatic = record.get('automatic_jobs', False)
        whole_cpu, whole_memory = trial['threads'], record['memory_gb'] / (record['jobs'] or 1)
        jobs = 1 if trial['policy']=='sequential' else record['jobs']
        if automatic:
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
            jobs = (max(1, min(len(subjects), record['cpus']//whole_cpu,
                              int(record['memory_gb']//whole_memory)))
                    if trial['policy']=='subject' else None)
            if trial['policy']=='sequential':
                jobs = 1
            trial['automatic_admission'] = dict(whole_subject_cpu=whole_cpu, whole_subject_memory_gb=whole_memory,
                                                jobs_argument=jobs, manual_subject_limit=False)
        command = [record['launcher'],'--launch','PythonSlicer',str(execution_snapshot/'scripts/run_batch_pipeline.py'),
                   '--batch-root',str(destination/'inputs'),'--output-root',str(destination/'outputs'),
                   '--manifest',str(destination/'batch.json'),'--run-name','benchmark',
                   '--scheduler',scheduler,
                   '--batch-cpus',str(record['cpus']),'--batch-memory-gb',str(record['memory_gb']),
                   '--surface-cpu-threads',str(trial['threads']),
                   '--mrtrix-nthreads',str(trial['threads']), '--resource-history',str(destination/'history.json'),
                   '--execute',*record['common'],*record['routes'][trial['route']]]
        if jobs is not None:
            command += ['--jobs', str(jobs)]
        if trial['policy'] == 'adaptive_tuned':
            profile = Path(trial['compute_profile'])
            if not profile.is_file() or sha(profile) != trial['compute_profile_sha256']:
                raise ValueError('Frozen runtime calibration changed or is missing')
            command += ['--compute-profile', str(profile)]
        else:
            command += ['--batch-stage-cpus', str(trial['threads'])]
        if trial['route'] == 'tracts':
            command += ['--ukf-parameters',json.dumps(dict(numThreads=trial['threads']))]
        environment = {**os.environ, **{key:str(trial['threads']) for key in
                       ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS','MRTRIX_NTHREADS')}}
        from shared.cohort_schedule import create_schedule
        from shared.cohort_io import GPU_ENV
        guard = create_schedule(destination/'gpu_guard', cpu_budget=record['cpus'], memory_gb=record['memory_gb'],
                                stage_cpus=trial['threads'], stage_memory_gb=1, gpu_slots=1)
        environment[GPU_ENV] = str(guard)
        if trial['policy'] in ('sequential','subject'):
            from shared.cohort_schedule import ENV
            environment[ENV] = str(create_schedule(destination/'subject_guard', cpu_budget=record['cpus'],
                memory_gb=record['memory_gb'], stage_cpus=whole_cpu,
                stage_memory_gb=whole_memory))
            command += ['--pipeline-script',str(execution_snapshot/'scripts/benchmark_subject_worker.py')]
        trial.update(status='running', command=command, started_at=time.time(), output=str(destination))
        save(path,record)
        with (destination/'stdout.log').open('w') as stdout, (destination/'stderr.log').open('w') as stderr:
            child = subprocess.Popen(command,env=environment,stdout=stdout,stderr=stderr,start_new_session=True)
            if record['cpu_affinity'] and hasattr(os,'sched_setaffinity'):
                os.sched_setaffinity(child.pid,record['cpu_affinity'])
            telemetry = CohortTelemetry(destination/'resources.jsonl').start()
            reason = None
            try:
                while child.poll() is None:
                    if time.monotonic() >= deadline:
                        reason = 'time_budget_exhausted'
                    if telemetry.monitor.peak > record['memory_gb']*1024**3:
                        reason = 'memory_budget_exceeded'
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
                measured = telemetry.finish('completed' if child.returncode == 0 else 'failed')
            trial.update(status='completed' if child.returncode == 0 and not reason else 'failed',
                         returncode=child.returncode, reason=reason, ended_at=time.time(), measurement=measured)
        # Process completion is not scientific acceptance; independent audit follows.
        if trial['status']=='completed':
            from scripts.audit_scientific_outputs import audit_output_root
            from scripts.benchmark_cohort_compare import scientific_inventory
            try:
                audit = audit_output_root(destination/'outputs', require_dti=True,
                    require_tractography=trial['route']=='tracts', min_subjects=len(record['subjects']), run_name='benchmark')
                save(destination/'independent_qc.json',audit)
                trial.update(qc_report=str(destination/'independent_qc.json'), qc_status=audit['status'],
                             qc_passed_subjects=audit['summary']['passed_subjects'])
                inventories = {row['subject']:scientific_inventory(destination/'outputs'/row['subject'])
                               for row in record['subjects']}
                save(destination/'scientific_fingerprints.json', inventories)
                trial['comparison_status'] = 'pending_reference'
                update_comparisons(record, folder)
            except Exception as exc:
                trial.update(qc_status='audit_error', audit_error=f'{type(exc).__name__}: {exc}')
        save(path,record)
        if trial['status'] != 'completed':
            record['status']='failed'
            save(path,record)
            return 2
    update_comparisons(record, folder)
    record['inputs_unchanged'] = all(sha(Path(row['source'])/name)==value for row in record['subjects']
                                   for name,value in row['sha256'].items())
    record['status']='passed' if record['inputs_unchanged'] and all(row.get('qc_status')=='passed' and row.get('comparison_status')=='passed'
                                    for row in record['trials']) else 'completed_pending_acceptance'
    save(path,record)
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','select'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dataset-root',type=Path,default=Path('/dataset/nfs_share01/HCP/HCP-YA/3T/Diffusion_Preprocessed'))
    parser.add_argument('--count',type=int,default=12)
    parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--cpus',type=int,default=32)
    parser.add_argument('--memory-gb',type=float,default=96)
    parser.add_argument('--threads',type=int,nargs='+',default=[4])
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--routes',nargs='+',choices=('dki','mkcurve','tracts'),default=['dki','mkcurve','tracts'])
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
    args=parser.parse_args()
    if args.action=='select':
        if not args.from_campaign:
            parser.error('select requires --from-campaign')
        return select_campaign(args.from_campaign,args.output,policies=args.policies,routes=args.routes,
                               reference_policy=args.reference_policy) and 0
    return prepare(args) if args.action=='prepare' else run(args)


if __name__=='__main__':
    raise SystemExit(main())
