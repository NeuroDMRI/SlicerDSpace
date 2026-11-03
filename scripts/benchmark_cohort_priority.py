#!/usr/bin/env python3
"""Switch a legacy frozen campaign at an unstarted boundary, preserving active work."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_cohort_workflows import save, sha, select_campaign


def arm(args):
    import psutil
    source = json.loads(args.source.read_text())
    active = [row for row in source['trials'] if row['status']=='running']
    pending = [row for row in source['trials'] if row['status']=='pending']
    if source['status']!='running' or len(active)!=1 or active[0]['policy']!='subject':
        raise ValueError('Wait for an active subject-level arm; no scientific process will be interrupted')
    if not pending or pending[0]['policy']!='stage':
        raise ValueError('The next unstarted boundary must be the stage-level ablation')
    owner = psutil.Process(source['driver_pid'])
    if str(args.source.parent) not in ' '.join(owner.cmdline()):
        raise ValueError('Source driver identity mismatch')
    queue = psutil.Process(args.old_queue_pid)
    old_queue = json.loads(args.old_queue.read_text())
    if old_queue['status']!='waiting' or queue.children() or str(args.old_queue) not in ' '.join(queue.cmdline()):
        raise ValueError('Only an idle matching continuation queue can be replaced')
    args.control.mkdir(parents=True,exist_ok=False)
    from scripts.stage_slicer_extension import stage_extension
    frozen = Path(stage_extension(args.control/'tools',slicer_version='Slicer-5.10',clean=False,
                                   link_mode='copy')['module_dir'])
    for name in ('benchmark_cohort_workflows.py','benchmark_cohort_compare.py','benchmark_subject_worker.py',
                 'benchmark_cohort_priority.py'):
        shutil.copy2(ROOT/'scripts'/name, frozen/'scripts'/name)
    plotter = args.control/'plot_cohort.py'
    shutil.copy2(args.plotter,plotter)
    save(args.control/'source_at_arm.json',source)
    save(args.control/'followup_at_arm.json',json.loads(args.followup.read_text()))
    boundary = args.source.parent/'trials'/pending[0]['id']
    boundary.mkdir(parents=True,exist_ok=False)
    save(boundary/'priority_boundary_hold.json',dict(reason='Defer unstarted stage-level ablation after current subject arm',
        control=str(args.control),scientific_work_started=False))
    record = dict(status='armed',source=str(args.source),followup=str(args.followup),
        source_pid=owner.pid,source_created=owner.create_time(),boundary=str(boundary),
        control_snapshot=str(frozen),plotter=str(plotter),plotter_sha256=sha(plotter),figure_root=str(args.figure_root),
        snapshot_hashes={str(p.relative_to(frozen)):sha(p) for p in frozen.rglob('*') if p.is_file()},
        followup_sha256=sha(args.followup),max_hours=args.max_hours,created_at=time.time(),campaigns=[],
        notes=['Current subject arm drains unchanged; filesystem boundary prevents the legacy driver launching its next trial',
               'Expected legacy FileExistsError is a planned handover, not an algorithm or scientific failure',
               'No input cropping, no participant reduction, no numerical parameter or scientific snapshot change'])
    save(args.control/'priority.json',record)
    queue.terminate()
    queue.wait(timeout=15)
    old_queue.update(status='superseded',continuation=str(args.control/'priority.json'),ended_at=time.time())
    save(args.old_queue,old_queue)
    return record


def original_alive(record):
    import psutil
    try:
        p=psutil.Process(record['source_pid'])
        return p.create_time()==record['source_created'] and p.status()!=psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def validate_handover(record):
    source = json.loads(Path(record['source']).read_text())
    hold = Path(record['boundary'])
    if {p.name for p in hold.iterdir()} != {'priority_boundary_hold.json'}:
        raise ValueError('Unexpected work at held boundary; do not start another budget')
    selected = [row for row in source['trials'] if row['policy'] in ('sequential','subject')]
    if len(selected)!=2 or any(row['status']!='completed' for row in selected):
        raise ValueError('Existing comparison arms did not finish; inspect retained logs')
    if any(row['status']!='pending' for row in source['trials'] if row['policy'] in ('stage','adaptive')):
        raise ValueError('A supposedly deferred arm has already started')
    log=Path(record['source']).parent/'driver.log'
    if 'FileExistsError' not in log.read_text() or str(hold) not in log.read_text():
        raise ValueError('Legacy driver exit does not match the planned boundary hold')
    return source


def run(args):
    path=args.control/'priority.json'
    record=json.loads(path.read_text())
    if record['status']!='armed':
        raise ValueError('Priority controller already started')
    snapshot=Path(record['control_snapshot'])
    if any(sha(snapshot/name)!=value for name,value in record['snapshot_hashes'].items()):
        raise ValueError('Control snapshot changed')
    record.update(status='waiting_existing_subject',controller_pid=__import__('os').getpid())
    save(path,record)
    deadline=time.monotonic()+record['max_hours']*3600
    try:
        while original_alive(record):
            if time.monotonic()>=deadline:
                raise TimeoutError('Existing arm exceeded controller wait budget')
            time.sleep(5)
        original=validate_handover(record)
        original.update(status='superseded_after_completed_subject',continuation=str(path),ended_at=time.time())
        save(Path(record['source']),original)
        if sha(record['followup'])!=record['followup_sha256']:
            raise ValueError('Pending long-model campaign changed')
        plans=[('dki-primary',Path(record['source']),['dki'],['sequential','subject','adaptive']),
               ('tracts-primary',Path(record['followup']),['tracts'],['subject','adaptive']),
               ('mkcurve-primary',Path(record['followup']),['mkcurve'],['subject','adaptive'])]
        for name,source,routes,policies in plans:
            folder=args.control/name
            campaign=select_campaign(source,folder,policies=policies,routes=routes,reference_policy='subject')
            spec=json.loads(campaign.read_text())
            command=[spec['launcher'],'--launch','PythonSlicer',str(snapshot/'scripts/benchmark_cohort_workflows.py'),
                     'run','--output',str(folder),'--max-hours',str(record['max_hours'])]
            row=dict(name=name,path=str(campaign),status='running',command=command,started_at=time.time())
            record['campaigns'].append(row)
            record.update(status='running',current=name)
            save(path,record)
            with (folder/'driver.log').open('ab') as stream:
                child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
                row['pid']=child.pid
                save(path,record)
                code=child.wait()
            outcome=json.loads(campaign.read_text())
            row.update(status=outcome['status'],returncode=code,ended_at=time.time())
            save(path,record)
            if code or outcome['status'] not in ('passed','completed_pending_acceptance'):
                raise RuntimeError(name+' stopped; inspect retained outputs before continuing')
            if outcome['status']=='passed':
                if sha(record['plotter'])!=record['plotter_sha256']:
                    raise ValueError('Frozen plotter changed')
                figure=Path(record['figure_root'])/name
                plotted=subprocess.run([sys.executable,record['plotter'],'--campaign',str(campaign),
                                        '--output',str(figure)],capture_output=True,text=True)
                row['figures']=dict(status='exported' if not plotted.returncode else 'failed',
                                    path=str(figure),error=plotted.stderr)
                save(path,record)
        record['status']='completed' if all(row['status']=='passed' for row in record['campaigns']) else 'completed_pending_acceptance'
    except Exception as exc:
        record.update(status='blocked',error=f'{type(exc).__name__}: {exc}')
        save(path,record)
        return 2
    save(path,record)
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('arm','run'))
    parser.add_argument('--control',type=Path,required=True)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--followup',type=Path)
    parser.add_argument('--old-queue',type=Path)
    parser.add_argument('--old-queue-pid',type=int)
    parser.add_argument('--plotter',type=Path)
    parser.add_argument('--figure-root',type=Path)
    parser.add_argument('--max-hours',type=float,default=168)
    args=parser.parse_args()
    if args.action=='arm':
        if not all((args.source,args.followup,args.old_queue,args.old_queue_pid,args.plotter,args.figure_root)):
            parser.error('arm requires the current campaigns, idle queue, plotter and figure root')
        record=arm(args)
        print(json.dumps(dict(status=record['status'],driver=str(Path(record['control_snapshot'])/'scripts/benchmark_cohort_priority.py'))))
        return 0
    return run(args)


if __name__=='__main__':
    raise SystemExit(main())
