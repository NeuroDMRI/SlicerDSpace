#!/usr/bin/env python3
"""Freeze current execution changes, calibrate, and compare scheduling and tuning separately."""

import argparse
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.benchmark_cohort_workflows import save, sha


def function_ast(path, name):
    tree = ast.parse(path.read_text())
    return ast.dump(next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==name),include_attributes=False)


def freeze_current(base, folder):
    from scripts.stage_slicer_extension import stage_extension
    for relative, names in {'microstructure/mkcurve.py':('_fit_chunk','detect_abnormal','repair_nonpositive'),
                            'microstructure/dki.py':('_compute_maps',)}.items():
        if any(function_ast(base/relative,name)!=function_ast(ROOT/relative,name) for name in names):
            raise ValueError('Numerical fitting kernel changed: '+relative)
    for relative in ('microstructure/mkcurve_tensor.py','microstructure/mkcurve_inputs.py',
                     'tractography/wma_registration.py','tractography/wma_transform.py','tractography/wma_fibers.py'):
        if sha(base/relative)!=sha(ROOT/relative):
            raise ValueError('Numerical kernel changed: '+relative)
    staged = stage_extension(folder/'install',slicer_version='Slicer-5.10',clean=False,link_mode='copy')
    snapshot = Path(staged['module_dir'])
    for name in ('benchmark_cohort_workflows.py','benchmark_cohort_compare.py','benchmark_subject_worker.py','benchmark_cohort_latest.py'):
        shutil.copy2(ROOT/'scripts'/name,snapshot/'scripts'/name)
    hashes = {str(path.relative_to(snapshot)):sha(path) for path in snapshot.rglob('*') if path.is_file() and '__pycache__' not in path.parts}
    changed = {name:dict(previous=sha(base/name) if (base/name).is_file() else None,current=digest)
               for name,digest in hashes.items() if not (base/name).is_file() or sha(base/name)!=digest}
    record = dict(snapshot=str(snapshot),hashes=hashes,changed_files=changed,
                  numerical_kernel_ast_checks=True,base_snapshot=str(base),
                  scope='Execution changes include indexed chunk dispatch and bounded runtime selection; full-product equivalence still required')
    save(folder/'candidate_provenance.json',record)
    return record


def prepare_campaign(source, destination, runtime, route, profile=None):
    record = deepcopy(json.loads(source.read_text()))
    original = next(row for row in record['trials'] if row['route']==route and row['policy']=='subject')
    trials = []
    if original['status']=='completed' and original.get('qc_status')=='passed':
        reference = deepcopy(original)
        reference['inherited_evidence'] = {name:sha(Path(reference['output'])/name)
            for name in ('independent_qc.json','scientific_fingerprints.json')}
        trials.append(reference)
    else:
        trials.append(dict(id=route+'-latest-subject-fixed',route=route,policy='subject',threads=4,repeat=0,
                           status='pending',snapshot=runtime['snapshot'],snapshot_hashes=runtime['hashes']))
    policies = ['adaptive'] + (['adaptive_tuned'] if profile else [])
    for policy in policies:
        trial = dict(id=route+'-latest-'+policy,route=route,policy=policy,threads=4,repeat=0,status='pending',
                     snapshot=runtime['snapshot'],snapshot_hashes=runtime['hashes'])
        if policy == 'adaptive_tuned':
            trial.update(compute_profile=str(profile),compute_profile_sha256=sha(profile),
                         effect='Scheduling plus exact-checked thread/worker selection')
        else:
            trial['effect']='Scheduling with matched fixed execution configuration'
        trials.append(trial)
    for key in ('driver_pid','started_at','ended_at','inputs_unchanged','execution_harness'):
        record.pop(key,None)
    record.update(status='prepared',trials=trials,reference_policy='subject',
                  selected_from=dict(path=str(source),sha256=sha(source)),candidate_provenance=runtime)
    record['notes'].append('Latest fixed adaptive arm isolates scheduling; tuned adaptive arm adds configuration selection at the same CPU/RAM and active-subject ceiling')
    destination.mkdir(parents=True,exist_ok=False)
    save(destination/'campaign.json',record)
    return destination/'campaign.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','run'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--base-campaign',type=Path,required=True)
    parser.add_argument('--long-campaign',type=Path,required=True)
    parser.add_argument('--wait-for',type=Path)
    parser.add_argument('--plotter',type=Path)
    args = parser.parse_args()
    if args.action == 'freeze':
        args.output.mkdir(parents=True,exist_ok=False)
        source = json.loads(args.base_campaign.read_text())
        runtime = freeze_current(Path(source['snapshot']),args.output)
        print(json.dumps(dict(snapshot=runtime['snapshot'],changed_files=len(runtime['changed_files'])),indent=2))
        return 0
    runtime = json.loads((args.output/'candidate_provenance.json').read_text())
    snapshot = Path(runtime['snapshot'])
    if any(sha(snapshot/name)!=digest for name,digest in runtime['hashes'].items()):
        raise ValueError('Frozen latest runtime changed')
    state = dict(status='waiting',campaigns=[],calibrations=[],controller_pid=os.getpid())
    state_path = args.output/'queue.json'
    if state_path.exists():
        raise FileExistsError('Latest queue already started')
    save(state_path,state)
    if args.wait_for:
        while json.loads(args.wait_for.read_text())['status']=='running':
            import psutil
            waiting = json.loads(args.wait_for.read_text())
            process = psutil.Process(waiting['driver_pid'])
            if str(args.wait_for.parent) not in ' '.join(process.cmdline()):
                raise RuntimeError('Previous scientific driver exited without a terminal record')
            time.sleep(30)
        previous = json.loads(args.wait_for.read_text())
        if previous['status'] not in ('passed','completed_pending_acceptance','execution_complete_pending_comparison'):
            raise ValueError('Previous scientific experiment requires inspection')
    base = json.loads(args.base_campaign.read_text())
    # Finish the primary DKI result before calibration for the longer MK-Curve route.
    for route in ('dki','mkcurve','tracts'):
        profile = None
        if route != 'tracts':
            profile = args.output/(route+'_compute_profile.json')
            command = [base['launcher'],'--launch','PythonSlicer',str(snapshot/'scripts/autotune_batch.py'),
                '--output',str(profile),'--stages','microstructure_'+route,'--max-cpus','4',
                '--cpu-budget',str(base['cpus']),'--memory-gb',str(base['memory_gb']),'--jobs',str(base['jobs'])]
            for row in base['subjects'][:3]:
                command += ['--subject-root',row['source']]
            method = 'MKCURVE' if route=='mkcurve' else 'WLS'
            command += ['--','--dki-fit-method',method]
            state.update(status='calibrating',route=route,command=command)
            save(state_path,state)
            with (args.output/(route+'_calibration.log')).open('w') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            state['calibrations'].append(dict(route=route,path=str(profile),sha256=sha(profile)))
        source = args.base_campaign if route=='dki' else args.long_campaign
        campaign = prepare_campaign(source,args.output/(route+'-latest'),runtime,route,profile)
        state.update(status='running',current=str(campaign))
        save(state_path,state)
        command = [base['launcher'],'--launch','PythonSlicer',str(snapshot/'scripts/benchmark_cohort_workflows.py'),
                   'run','--output',str(campaign.parent),'--max-hours','168']
        with (campaign.parent/'driver.log').open('w') as log:
            result = subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        outcome = json.loads(campaign.read_text())
        state['campaigns'].append(dict(path=str(campaign),status=outcome['status'],returncode=result.returncode))
        save(state_path,state)
        if result.returncode or outcome['status']!='passed':
            state.update(status='blocked',reason='Full-volume QC or equivalence requires inspection')
            save(state_path,state)
            return 2
        if args.plotter:
            with (campaign.parent/'figures.log').open('w') as log:
                subprocess.run([sys.executable,str(args.plotter),'--campaign',str(campaign),
                                '--output',str(campaign.parent/'figures')],stdout=log,stderr=subprocess.STDOUT,check=True)
    state.update(status='completed')
    save(state_path,state)
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        if '--output' in sys.argv:
            path = Path(sys.argv[sys.argv.index('--output')+1])/'queue.json'
            if path.is_file():
                state = json.loads(path.read_text())
                state.update(status='failed',error=f'{type(exc).__name__}: {exc}')
                save(path,state)
        raise
