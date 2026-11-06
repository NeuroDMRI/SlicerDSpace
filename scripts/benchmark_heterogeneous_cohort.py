#!/usr/bin/env python3
"""Prepare paired CPU tract/WMA and GPU/CPU complete-workflow experiments."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_cohort_workflows import save, sha


def options(record, hybrid):
    original = record['routes']['tracts']
    def value(name):
        return original[original.index(name)+1]
    if hybrid:
        # WMA/UKF has its own full-volume comparison. Keep this distinct hybrid
        # workload focused on models, CSD/ACT and its matching weighted connectome.
        return ['--targets', 'validation,segmentation,parcellation,surface,microstructure,tractography,connectome',
                '--microstructure-stages', 'dti,freewater,noddi', '--tractography-backend', 'csd_act',
                '--ddseg-device', 'cuda:0', '--surface-device', 'cuda:0', '--surface-precision', 'fp32',
                '--select-streamlines', '1000000', '--mrtrix-seed', '20261104',
                '--connectome-backend', 'mrtrix', '--microstructure-free-water-n-jobs', '1']
    arguments = ['--targets', 'validation,microstructure,tractography,bundles',
                 '--microstructure-stages', 'dti', '--tractography-backend', 'ukf',
                 '--bundle-backend', 'wma', '--wma-root', value('--wma-root'),
                 '--wma-atlas-dir', value('--wma-atlas-dir'),
                 '--slicer-executable', value('--slicer-executable'),
                 '--ukf-executable', value('--ukf-executable')]
    arguments += ['--bundle-tractography-backend', 'ukf']
    return arguments


def build_record(base, snapshot, hashes, hybrid, affinity, memory_gb):
    record = deepcopy(base)
    for key in ('driver_pid', 'started_at', 'ended_at', 'inputs_unchanged',
                'execution_harness', 'candidate_provenance', 'continuation', 'selected_from'):
        record.pop(key, None)
    common = list(record['common'])
    common[common.index('--max-parallel-models')+1] = '1'
    policies = ('subject', 'adaptive') if hybrid else ('adaptive', 'subject')
    record.update(status='prepared', snapshot=str(snapshot), source_hashes=hashes,
                  common=common, routes={'tracts': options(base, hybrid)},
                  trials=[dict(id=('hybrid' if hybrid else 'ukf-wma')+'-'+policy,
                               route='tracts', policy=policy, threads=4, repeat=0, status='pending')
                          for policy in policies],
                  jobs=None, automatic_jobs=True, reference_policy='subject',
                  cpus=len(affinity), cpu_affinity=list(affinity), memory_gb=memory_gb,
                  repeats=1, requires_idle_gpu=hybrid,
                  workload_scope=('GPU anatomical reconstruction, CPU microstructure and CSD/ACT/SIFT2 connectomes; '
                                  'WMA is evaluated separately in the CPU UKF/WMA campaign' if hybrid else
                                  'DTI, two-tensor UKF and ORG WMA with unchanged all-tract sampling QC'),
                  notes=['12 distinct full-volume acquisitions; no cropped or duplicate participants',
                         'One frozen implementation, fresh outputs and matched algorithm/thread/seed settings',
                         'Automatic subject frontier in both arms; no arbitrary four-subject ceiling',
                         'Single within-subject model worker in both arms; only cross-subject scheduling differs',
                         'Subject vs Adaptive only; Stage pilot rejected nested workers and is not reused',
                         'CPU and hybrid campaigns use disjoint physical CPU sets; existing MK-Curve unchanged',
                         'Single paired pilot; shared host and OS caches; report observed concurrency with throughput'])
    return record


def idle_devices(rows, minimum_free_mib=30000):
    """Require a substantially idle device, not merely spare VRAM on a busy GPU."""
    return [int(row[0]) for row in rows if float(row[1]) <= 5 and
            float(row[2]) <= 1024 and float(row[3])-float(row[2]) >= minimum_free_mib]


def preflight(record, folder, environment):
    if record.get('requires_idle_gpu'):
        # Device discovery only: no model is loaded or inference submitted before admission.
        environment = {**environment, 'CUDA_VISIBLE_DEVICES': '0'}
    inputs = folder/'preflight-input'/'input'
    inputs.mkdir(parents=True)
    source = Path(record['subjects'][0]['source'])
    for name in ('data.nii.gz', 'bvals', 'bvecs', 'nodif_brain_mask.nii.gz'):
        (inputs/name).symlink_to(source/name)
    snapshot = Path(record['snapshot'])
    command = [record['launcher'], '--launch', 'PythonSlicer',
               str(snapshot/'shared/run_subject_pipeline_cli.py'),
               '--subject-root', str(inputs.parent), '--subject-id', record['subjects'][0]['subject'],
               '--doctor', '--doctor-output-dir', str(folder/'preflight'),
               *record['common'], *record['routes']['tracts']]
    for attempt in range(3):
        name = 'preflight' if attempt == 0 else f'preflight-retry-{attempt+1}'
        command[command.index('--doctor-output-dir')+1] = str(folder/name)
        with (folder/f'{name}.log').open('w') as log:
            subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        # Doctor exits successfully when it writes a report, even for blocked targets.
        report = json.loads((folder/name/'environment_doctor.json').read_text())
        summary = report['summary']
        if not summary['blocked_targets']:
            break
        failed = [row for row in report.get('dependencies', [])
                  if row.get('required') and row.get('status') != 'available'
                  and set(row.get('targets', [])) & set(summary['blocked_targets'])]
        if attempt < 2 and failed and all('probe timed out' in str(row.get('details', {}).get('error', ''))
                                       for row in failed):
            time.sleep(10)
            continue
        raise RuntimeError('Route dependency checks blocked: '+', '.join(summary['blocked_targets']))
    command = [record['launcher'], '--launch', 'PythonSlicer',
               str(snapshot/'shared/run_subject_pipeline_cli.py'),
               '--subject-root', str(inputs.parent), '--subject-id', record['subjects'][0]['subject'],
               '--plan', '--skip-environment-check', '--run-name', 'benchmark',
               '--output-root', str(folder/'preflight-output'),
               *record['common'], *record['routes']['tracts']]
    with (folder/'preflight-plan.log').open('w') as log:
        subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    plan = json.loads((folder/'preflight-output/00_run_metadata/benchmark_run_plan.json').read_text())
    check_plan(plan, summary['runnable_targets'])
    summary['task_plan'] = plan['summary']
    from shared.cohort_targets import selected_stages
    if 'ukf' in selected_stages(record['routes']['tracts']):
        command = [record['launcher'], '--launch', 'PythonSlicer',
                   str(snapshot/'scripts/benchmark_heterogeneous_cohort.py'),
                   'validate-inputs', '--output', str(folder)]
        with (folder/'preflight-inputs.log').open('w') as log:
            subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        summary['input_checks'] = str(folder/'ukf_input_checks.json')
    return summary


def check_plan(plan, expected):
    items = plan['items']
    missing = set(expected) - {item['target'] for item in items}
    blocked = [f"{item['target']}: {item.get('reason', '')}" for item in items
               if item['status'] not in ('will_run', 'skipped')]
    if not items or missing or blocked:
        raise RuntimeError('Unrunnable cohort plan: '+ '; '.join([*sorted(missing), *blocked]))


def validate_campaign_inputs(folder):
    from shared.pipeline import RawInputBundle
    from tractography.ukf import validate_inputs

    record = json.loads((folder/'campaign.json').read_text())
    report = dict(status='running', subjects=[])
    path = folder/'ukf_input_checks.json'
    for row in record['subjects']:
        source = Path(row['source'])
        raw = RawInputBundle(source, row['subject'], source, source/'data.nii.gz',
                             source/'bvals', source/'bvecs', source/'nodif_brain_mask.nii.gz')
        try:
            checks = validate_inputs(raw)
        except Exception as exc:
            report['subjects'].append(dict(subject=row['subject'], status='failed', error=str(exc)))
            report['status'] = 'failed'
            save(path, report)
            raise
        report['subjects'].append(dict(subject=row['subject'], status='passed', checks=checks))
        save(path, report)
        print(json.dumps(report['subjects'][-1]), flush=True)
    report['status'] = 'passed'
    save(path, report)


def verify(args):
    """Apply the corrected output auditor without changing frozen timing records."""
    from scripts.audit_scientific_outputs import audit_output_root
    from scripts.benchmark_cohort_compare import scientific_inventory, scientific_paths
    from scripts.benchmark_cohort_workflows import update_comparisons, timing_issues

    name = getattr(args, 'verification_name', 'post_validation')
    if Path(name).name != name or name in ('.', '..'):
        raise ValueError('Verification name must be a single directory name')
    destination = args.output/name
    destination.mkdir(exist_ok=False)
    state_path = destination/'verification.json'
    sources = [Path(__file__), ROOT/'scripts/benchmark_cohort_workflows.py', ROOT/'scripts/audit_scientific_outputs.py',
               ROOT/'scripts/benchmark_cohort_compare.py', ROOT/'shared/orchestrator.py',
               ROOT/'tractography/ukf.py']
    state = dict(status='waiting', started_at=time.time(), pid=os.getpid(),
                 validation_source_hashes={str(p): sha(p) for p in sources},
                 scope='Fresh output validation only; no scientific computation or timing replacement')
    save(state_path, state)
    try:
        deadline = time.monotonic()+args.max_hours*3600
        while True:
            record = json.loads((args.output/'campaign.json').read_text())
            if record['status'] not in ('prepared', 'running'):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('Cohort computation has not completed within the waiting budget')
            time.sleep(30)
        if not record['trials'] or not all(trial['status'] == 'completed' for trial in record['trials']):
            raise RuntimeError('Scientific execution incomplete; no accepted comparison can be produced')
        if record.get('cpu_affinity') and hasattr(os, 'sched_setaffinity'):
            os.sched_setaffinity(0, record['cpu_affinity'][:1])
        if any(sha(p) != value for p, value in state['validation_source_hashes'].items()):
            raise RuntimeError('Validation code changed while waiting; retain the old record and start a new verification')
        state.update(status='verifying', timing_record_sha256=sha(args.output/'campaign.json'))
        save(state_path, state)
        derived = deepcopy(record)
        for trial in derived['trials']:
            folder = Path(trial['output'])
            original_qc_status = trial.get('qc_status')
            audit = audit_output_root(folder/'outputs', require_dti=True, require_tractography=True,
                                      min_subjects=len(record['subjects']), run_name='benchmark')
            report = destination/f"{trial['id']}_qc.json"
            save(report, audit)
            trial.update(qc_report=str(report), qc_status=audit['status'],
                         qc_passed_subjects=audit['summary']['passed_subjects'])
            inventory_path = folder/'scientific_fingerprints.json'
            missing = not inventory_path.is_file()
            evidence = None
            if missing:
                # This repair is for a failed comparison reader, never failed computation/QC.
                previous_qc = folder/'independent_qc.json'
                error = trial.get('audit_error', '')
                if (not getattr(args, 'rebuild_missing_fingerprints', False)
                        or original_qc_status != 'audit_error'
                        or 'Empty or unreadable polydata:' not in error or 'FiberClustering/' not in error
                        or not previous_qc.is_file()
                        or json.loads(previous_qc.read_text()).get('status') != 'passed'):
                    raise RuntimeError('Missing original product inventory requires a documented reader-only repair')
                paths = [p for row in record['subjects']
                         for p in scientific_paths(folder/'outputs'/row['subject'])]
                evidence = {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in paths}
                ended = trial.get('ended_at', 0)
                if not evidence or any(mtime > (ended+2)*1e9 for _, mtime in evidence.values()):
                    raise RuntimeError('Retained scientific products were written after the timed trial')
            fresh = {row['subject']: scientific_inventory(folder/'outputs'/row['subject'])
                     for row in record['subjects']}
            if missing:
                current = {str(p): (p.stat().st_size, p.stat().st_mtime_ns)
                           for row in record['subjects'] for p in scientific_paths(folder/'outputs'/row['subject'])}
                if current != evidence:
                    raise RuntimeError('Scientific products changed during inventory reconstruction')
                trial['inventory_provenance'] = dict(
                    mode='reconstructed_after_documented_reader_error', original_error=error,
                    original_qc=str(previous_qc), original_qc_sha256=sha(previous_qc),
                    file_count=len(evidence), retained_file_metadata=evidence,
                    limitation='First successful content inventory is post-run; timestamps are not a prior content hash')
            elif fresh != json.loads(inventory_path.read_text()):
                raise RuntimeError('Scientific products changed after the timed trial: '+trial['id'])
            fresh_path = destination/f"{trial['id']}_fingerprints.json"
            save(fresh_path, fresh)
            trial['scientific_fingerprints'] = str(fresh_path)
        update_comparisons(derived, destination)
        derived['inputs_unchanged'] = all(sha(Path(row['source'])/name) == value
            for row in record['subjects'] for name, value in row['sha256'].items())
        derived['status'] = 'passed' if derived['inputs_unchanged'] and all(
            trial.get('qc_status') == trial.get('comparison_status') == 'passed' and not timing_issues(record, trial)
            for trial in derived['trials']) else 'completed_pending_acceptance'
        if any(sha(p) != value for p, value in state['validation_source_hashes'].items()) or \
                sha(args.output/'campaign.json') != state['timing_record_sha256']:
            raise RuntimeError('Validation code or original timing record changed during verification')
        save(destination/'campaign.json', derived)
        state.update(status=derived['status'], ended_at=time.time())
        save(state_path, state)
        return 0 if state['status'] == 'passed' else 2
    except Exception as exc:
        state.update(status='failed', ended_at=time.time(), error=f'{type(exc).__name__}: {exc}')
        save(state_path, state)
        raise


def prepare(args):
    base = json.loads(args.base_campaign.read_text())
    if len(base['subjects']) < 10:
        raise ValueError('At least 10 distinct full-volume subjects are required')
    affinity = list(range(args.first_cpu, args.first_cpu+args.cpus))
    if not set(affinity) <= os.sched_getaffinity(0):
        raise ValueError('Requested CPU set is outside the current allocation')
    args.output.mkdir(parents=True, exist_ok=False)
    from scripts.stage_slicer_extension import stage_extension
    staged = stage_extension(args.output/'install', slicer_version='Slicer-5.10', clean=False, link_mode='copy')
    snapshot = Path(staged['module_dir'])
    for name in ('benchmark_cohort_workflows.py', 'benchmark_cohort_compare.py',
                 'benchmark_subject_worker.py', 'benchmark_heterogeneous_cohort.py'):
        shutil.copy2(ROOT/'scripts'/name, snapshot/'scripts'/name)
    hashes = {str(p.relative_to(snapshot)): sha(p) for p in snapshot.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts}
    record = build_record(base, snapshot, hashes, args.hybrid, affinity, args.memory_gb)
    # Surface inference is a new consumer; freeze its model/atlas inputs too.
    if args.hybrid:
        models = Path(record['common'][record['common'].index('--model-assets-dir')+1])
        for folder in ('surface', 'Resources'):
            for p in sorted((models/folder).rglob('*')):
                if p.is_file():
                    record['external_asset_hashes'][str(p.resolve())] = sha(p)
    save(args.output/'campaign.json', record)
    print(json.dumps(dict(campaign=str(args.output/'campaign.json'), subjects=len(record['subjects']),
                          hybrid=args.hybrid, cpu_affinity=affinity, routes=record['routes']), indent=2))


def run(args):
    path = args.output/'campaign.json'
    record = json.loads(path.read_text())
    if record['status'] != 'prepared':
        raise ValueError('Existing results must not be overwritten')
    state_path = args.output/'controller.json'
    if state_path.exists():
        raise FileExistsError('Controller already started')
    state = dict(status='preflight', pid=os.getpid(), started_at=time.time())
    save(state_path, state)
    try:
        environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '',
                       **{key: '4' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                             'MKL_NUM_THREADS', 'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS')}}
        state['preflight_summary'] = preflight(record, args.output, environment)
        save(state_path, state)
        if record['requires_idle_gpu']:
            deadline = time.monotonic()+args.max_hours*3600
            previous, consecutive = None, 0
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError('No idle GPU became available within the waiting budget')
                result = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
                                         '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
                rows = [line.split(',') for line in result.stdout.splitlines() if line.strip()]
                choices = idle_devices(rows)
                selected = choices[0] if choices else None
                consecutive = consecutive+1 if selected is not None and selected == previous else int(selected is not None)
                previous = selected
                state.update(status='waiting_for_idle_gpu', gpu_observations=rows,
                             updated_at=time.time(), consecutive_idle_checks=consecutive)
                save(state_path, state)
                if consecutive >= 3:
                    environment['CUDA_VISIBLE_DEVICES'] = str(selected)
                    break
                time.sleep(30)
        snapshot = Path(record['snapshot'])
        state.update(status='running', cuda_visible_devices=environment['CUDA_VISIBLE_DEVICES'])
        save(state_path, state)
        command = [record['launcher'], '--launch', 'PythonSlicer',
                   str(snapshot/'scripts/benchmark_cohort_workflows.py'), 'run',
                   '--output', str(args.output), '--max-hours', str(args.max_hours)]
        with (args.output/'driver.log').open('w') as log:
            result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        state.update(status=json.loads(path.read_text())['status'], returncode=result.returncode, ended_at=time.time())
        save(state_path, state)
        return result.returncode
    except Exception as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}', ended_at=time.time())
        save(state_path, state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'validate-inputs', 'verify'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-campaign', type=Path)
    parser.add_argument('--hybrid', action='store_true')
    parser.add_argument('--first-cpu', type=int, default=32)
    parser.add_argument('--cpus', type=int, default=24)
    parser.add_argument('--memory-gb', type=float, default=160)
    parser.add_argument('--max-hours', type=float, default=168)
    parser.add_argument('--verification-name', default='post_validation')
    parser.add_argument('--rebuild-missing-fingerprints', action='store_true',
                        help='Reconstruct inventories only for retained outputs with passed timed QC and a documented empty-cluster reader error')
    args = parser.parse_args()
    if args.cpus < 4 or args.memory_gb < 32 or args.max_hours <= 0:
        parser.error('Use at least four CPUs, 32 GiB and a positive time budget')
    if args.action == 'prepare':
        if args.base_campaign is None:
            parser.error('prepare requires --base-campaign')
        prepare(args)
        return 0
    if args.action == 'validate-inputs':
        validate_campaign_inputs(args.output)
        return 0
    if args.action == 'verify':
        return verify(args)
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
