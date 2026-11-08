#!/usr/bin/env python3
"""Preregistered, snapshot-isolated, no-resume cohort acceptance (no retries)."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path, record):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n')
    temporary.replace(path)


def audit_input(directory):
    import nibabel as nib
    import numpy as np
    directory = Path(directory).resolve()
    files = {name: directory / name for name in ('data.nii.gz', 'bvals', 'bvecs', 'nodif_brain_mask.nii.gz')}
    for path in files.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'Missing real acquisition file: {path}')
    dwi, mask = (nib.load(str(files[name])) for name in ('data.nii.gz', 'nodif_brain_mask.nii.gz'))
    bvals, bvecs = np.loadtxt(files['bvals']), np.loadtxt(files['bvecs'])
    if len(dwi.shape) != 4 or bvals.shape != (dwi.shape[3],) or bvecs.shape not in ((3, dwi.shape[3]), (dwi.shape[3], 3)):
        raise ValueError(f'Inconsistent diffusion dimensions: {directory}')
    values = np.asarray(mask.dataobj)
    if (mask.shape != dwi.shape[:3] or not np.allclose(mask.affine, dwi.affine, rtol=0, atol=1e-4)
            or not np.isfinite(dwi.affine).all() or not np.isin(values, [0, 1]).all() or not values.any()
            or not np.isfinite(bvals).all() or not np.isfinite(bvecs).all()):
        raise ValueError(f'Invalid gradients/native brain mask: {directory}')
    return {'directory': str(directory), 'shape': list(dwi.shape), 'brain_voxels': int(values.sum()),
            'files': {name: {'path': str(path), 'size': path.stat().st_size, 'sha256': digest(path)}
                      for name, path in files.items()}}


def validate_protocol(protocol):
    subjects = protocol['subjects']
    ids = [str(item['subject_id']) for item in subjects]
    if len(ids) < 2 or len(set(ids)) != len(ids) or any(not item.isalnum() for item in ids):
        raise ValueError('At least two distinct safe subject IDs are required')
    protected = {'--subject-root', '--subject-id', '--output-root', '--run-name', '--no-resume',
                 '--user-config', '--no-user-config', '--plan', '--doctor', '--execute',
                 '--ignore-environment-blockers', '--skip-environment-check'}
    arguments = protocol['common_cli_args']
    if any(str(item).split('=')[0] in protected for item in arguments):
        raise ValueError('Cohort CLI must not override isolation, execution or preflight policy')
    if not protocol.get('expected_tasks'):
        raise ValueError('Preregister expected tasks; an empty workflow is not acceptance')
    return subjects


def evaluate_attempt(manifest, expected_tasks):
    entries = manifest.get('tasks', [])
    tasks = {item['name']: item for item in entries}
    duplicates = sorted(name for name in tasks if sum(item['name'] == name for item in entries) > 1)
    missing = sorted(set(expected_tasks) - set(tasks))
    # Reuse/skip/blocked are not evidence that a fresh selected task ran once.
    unacceptable = {name: item.get('status') for name, item in tasks.items()
                    if item.get('status') not in {'completed', 'passed'}}
    return {'missing_tasks': missing, 'unacceptable_tasks': unacceptable, 'duplicate_tasks': duplicates,
            'manifest_completed': manifest.get('status') == 'completed',
            'passed': not missing and not unacceptable and not duplicates and manifest.get('status') == 'completed'}


def run_isolated(command, *, environment, cwd, log_path, timeout):
    """Own the complete process group, including CLI grandchildren on timeout."""
    options = ({'start_new_session': True} if os.name != 'nt' else
               {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP})
    with log_path.open('w') as stream:
        process = subprocess.Popen(command, env=environment, cwd=cwd, stdout=stream,
                                   stderr=subprocess.STDOUT, **options)
        try:
            return process.wait(timeout=timeout)
        except BaseException:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            else:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                # A launcher may exit before its children; never use the
                # launcher's exit alone as evidence the owned group ended.
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                try:
                    os.killpg(process.pid, getattr(signal, 'SIGKILL', signal.SIGTERM))
                except ProcessLookupError:
                    pass
            process.wait()
            raise


def run(protocol, report, slicer):
    from scripts.stage_slicer_extension import stage_extension
    subjects = validate_protocol(protocol)
    # Never overwrite or resume a failed cohort in the same report directory.
    report.mkdir(parents=True, exist_ok=False)
    record = {'kind': 'one_pass_cohort_acceptance', 'status': 'auditing_inputs',
              'automatic_qc_passed': False, 'human_review': 'pending', 'attempts_per_subject': 1,
              'scope': protocol.get('scope', ''), 'protocol': protocol, 'subjects': [],
              'started_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    result_path = report / 'acceptance.json'
    write_json(result_path, record)
    try:
        acquisitions = [audit_input(item['diffusion_dir']) for item in subjects]
        if len({item['files']['data.nii.gz']['sha256'] for item in acquisitions}) != len(subjects):
            raise ValueError('Different subject IDs point to duplicate acquisitions')
        record['input_audit'] = acquisitions
        staged = stage_extension(report / 'install', slicer_version='Slicer-5.10', clean=False, link_mode='copy')
        module = Path(staged['module_dir'])
        frozen = {str(path.relative_to(module)): digest(path) for path in module.rglob('*') if path.is_file()}
        backend = Path(protocol['tract_querier_root']).resolve()
        backend_copy = report / 'tract_querier'
        shutil.copytree(backend, backend_copy, ignore=shutil.ignore_patterns('.git', '__pycache__', '*.pyc'))
        frozen_backend = {str(path.relative_to(backend_copy)): digest(path)
                          for path in backend_copy.rglob('*') if path.is_file()}
        record.update(status='running', runtime_hashes=frozen, backend_hashes=frozen_backend,
                      runtime_module=str(module), git_head=subprocess.check_output(
                          ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
        write_json(report / 'protocol.json', protocol)
        write_json(result_path, record)
        for subject, acquisition in zip(subjects, acquisitions):
            subject_id = str(subject['subject_id'])
            case = report / 'subjects' / subject_id
            (case / 'input').mkdir(parents=True)
            for name, item in acquisition['files'].items():
                (case / 'input' / name).symlink_to(item['path'])
            command = [str(slicer), '--launch', 'PythonSlicer', str(module / 'shared/run_subject_pipeline_cli.py'),
                       '--subject-root', str(case), '--subject-id', subject_id, '--output-root', str(case / 'outputs'),
                       '--run-name', 'one_pass', '--no-resume', '--no-user-config', '--execute',
                       '--tract-querier-root', str(backend_copy), *protocol['common_cli_args']]
            attempt = {'subject_id': subject_id, 'status': 'running', 'command': command,
                       'attempt': 1, 'log': str(case / 'pipeline.log')}
            record['subjects'].append(attempt)
            write_json(result_path, record)
            started = time.monotonic()
            environment = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
                               PYTHONUNBUFFERED='1', XDG_CONFIG_HOME=str(case / 'settings'), XDG_CACHE_HOME=str(case / 'cache'))
            code = run_isolated(command, environment=environment, cwd=case,
                                log_path=case / 'pipeline.log', timeout=protocol.get('timeout_seconds', 21600))
            attempt.update(return_code=code, elapsed_seconds=time.monotonic() - started)
            manifest_path = case / 'outputs/00_run_metadata/one_pass_orchestration_manifest.json'
            attempt['manifest'] = str(manifest_path)
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text())
                attempt.update(evaluate_attempt(manifest, protocol['expected_tasks']))
                attempt['task_statuses'] = {item['name']: item['status'] for item in manifest.get('tasks', [])}
            else:
                attempt.update(passed=False, error='No terminal orchestration manifest')
            audit_path = case / 'independent_audit.json'
            audit_command = [sys.executable, str(module / 'scripts/audit_scientific_outputs.py'),
                             '--output-root', str(case / 'outputs'), '--report', str(audit_path),
                             '--run-name', 'one_pass', *protocol.get('audit_flags', [])]
            audit_code = run_isolated(audit_command, cwd=case, environment=environment,
                                      log_path=case / 'independent_audit.log', timeout=1800)
            attempt['independent_audit'] = str(audit_path)
            attempt['independent_audit_passed'] = (audit_code == 0 and audit_path.is_file()
                and json.loads(audit_path.read_text()).get('status') == 'passed')
            attempt['inputs_unchanged'] = all(digest(item['path']) == item['sha256']
                                               for item in acquisition['files'].values())
            attempt['status'] = 'passed' if (attempt.get('passed') and code == 0
                                             and attempt['inputs_unchanged'] and attempt['independent_audit_passed']) else 'failed'
            write_json(result_path, record)
        record['runtime_unchanged'] = all((module / path).is_file() and digest(module / path) == value for path, value in frozen.items())
        record['backend_unchanged'] = all((backend_copy / path).is_file() and digest(backend_copy / path) == value for path, value in frozen_backend.items())
        record['automatic_qc_passed'] = (len(record['subjects']) == len(subjects)
            and all(item['status'] == 'passed' for item in record['subjects'])
            and record['runtime_unchanged'] and record['backend_unchanged'])
        record['status'] = 'passed' if record['automatic_qc_passed'] else 'failed'
    except Exception as exc:
        record.update(status='failed', error=f'{type(exc).__name__}: {exc}', automatic_qc_passed=False)
        if record['subjects'] and record['subjects'][-1].get('status') == 'running':
            record['subjects'][-1].update(status='failed', passed=False, error=record['error'])
    finally:
        record['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        write_json(result_path, record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--slicer', required=True, type=Path)
    args = parser.parse_args()
    outcome = run(json.loads(args.protocol.read_text()), args.report.resolve(), args.slicer.resolve())
    print(json.dumps({'status': outcome['status'], 'report': str(args.report / 'acceptance.json')}), flush=True)
    raise SystemExit(0 if outcome['automatic_qc_passed'] else 1)
