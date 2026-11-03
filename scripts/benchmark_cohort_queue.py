#!/usr/bin/env python3
"""Continue frozen campaigns serially and draw only accepted completed results."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def adopt_controller(pid, state_path, campaign_path, continuation):
    """Transfer the supervisor only; the scientific driver and its group keep running."""
    import psutil
    old=json.loads(state_path.read_text())
    controller=psutil.Process(pid)
    current=json.loads(campaign_path.read_text())
    if old.get('controller_pid')!=pid or old.get('status')!='running' or current.get('status')!='running':
        raise ValueError('Only an active matching supervisor can be adopted')
    if old.get('campaigns',[])[-1].get('path')!=str(campaign_path):
        raise ValueError('Supervisor is not waiting on the selected campaign')
    driver=psutil.Process(current['driver_pid'])
    if controller.pid not in {parent.pid for parent in driver.parents()}:
        raise ValueError('Scientific driver is not owned by this supervisor')
    # Reserve the unstarted next route in case the old driver finishes during transfer.
    boundary=state_path.parent/'tracts-primary'
    boundary.mkdir(exist_ok=False)
    (boundary/'supervisor_handover.json').write_text(json.dumps(dict(continuation=str(continuation),started=False)))
    controller.terminate()
    controller.wait(timeout=15)
    if not driver.is_running() or driver.status()==psutil.STATUS_ZOMBIE:
        if json.loads(campaign_path.read_text())['status']=='running':
            raise RuntimeError('Scientific driver exited without a terminal record during handover')
    old.update(status='superseded_with_active_campaign',continuation=str(continuation),ended_at=time.time())
    state_path.write_text(json.dumps(old,indent=2)+'\n')
    return dict(previous_controller=pid,scientific_driver=current['driver_pid'],
                driver_creation_time=driver.create_time(),signal_scope='Supervisor PID only; no process-group signal')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-for',type=Path,required=True)
    parser.add_argument('--then',type=Path,nargs='+',required=True)
    parser.add_argument('--record',type=Path,required=True)
    parser.add_argument('--plotter',type=Path,required=True)
    parser.add_argument('--figure-root',type=Path,required=True)
    parser.add_argument('--max-hours',type=float,default=168)
    parser.add_argument('--harness-root',type=Path,help='Frozen control tools; per-trial scientific snapshots stay in each campaign')
    parser.add_argument('--adopt-controller-pid',type=int)
    parser.add_argument('--adopt-controller-record',type=Path)
    args = parser.parse_args()
    if args.record.exists():
        raise FileExistsError('Queue record already exists')
    record = dict(status='waiting', waiting_for=str(args.wait_for), campaigns=[],
                  plotter=str(args.plotter),plotter_sha256=hashlib.sha256(args.plotter.read_bytes()).hexdigest(),figures=[])
    def save():
        temporary = args.record.with_suffix('.tmp')
        temporary.write_text(json.dumps(record,indent=2)+'\n')
        temporary.replace(args.record)
    save()
    if args.adopt_controller_pid:
        if not args.adopt_controller_record:
            parser.error('Controller adoption requires its state record')
        record['adopted_controller']=adopt_controller(args.adopt_controller_pid,args.adopt_controller_record,
                                                     args.wait_for,args.record)
        save()
    def plot(path):
        outcome=json.loads(path.read_text())
        if outcome['status']!='passed':
            record['figures'].append(dict(campaign=str(path),status='withheld_pending_acceptance'))
        else:
            if hashlib.sha256(args.plotter.read_bytes()).hexdigest()!=record['plotter_sha256']:
                raise ValueError('Frozen plotter changed')
            command=[sys.executable,str(args.plotter),'--campaign',str(path),
                     '--output',str(args.figure_root/path.parent.name)]
            result=subprocess.run(command,capture_output=True,text=True)
            record['figures'].append(dict(campaign=str(path),returncode=result.returncode,
                status='exported' if not result.returncode else 'failed',error=result.stderr))
        save()
    deadline = time.monotonic()+args.max_hours*3600
    while True:
        previous = json.loads(args.wait_for.read_text())
        if previous['status'] != 'running':
            break
        import psutil
        try:
            process = psutil.Process(previous['driver_pid'])
            alive = str(args.wait_for.parent) in ' '.join(process.cmdline())
        except psutil.Error:
            alive = False
        if not alive or time.monotonic() >= deadline:
            record.update(status='blocked', reason='Waiting campaign exited without a terminal record or exceeded time budget')
            save()
            return 2
        time.sleep(30)
    complete = ('passed','completed_pending_acceptance','execution_complete_pending_comparison')
    if previous['status'] not in complete:
        record.update(status='blocked',reason='Previous campaign did not complete execution', previous_status=previous['status'])
        save()
        return 2
    plot(args.wait_for)
    for path in args.then:
        campaign = json.loads(path.read_text())
        harness=args.harness_root or Path(campaign['snapshot'])
        remaining=(deadline-time.monotonic())/3600
        if remaining<=0:
            record.update(status='blocked',reason='Queue time budget exhausted before starting another campaign')
            save()
            return 2
        command = [campaign['launcher'],'--launch','PythonSlicer',
                   str(harness/'scripts/benchmark_cohort_workflows.py'),
                   'run','--output',str(path.parent),'--max-hours',str(remaining)]
        record.update(status='running', current=str(path))
        save()
        with (path.parent/'driver.log').open('ab') as log:
            result = subprocess.run(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        outcome = json.loads(path.read_text())
        record['campaigns'].append(dict(path=str(path),returncode=result.returncode,status=outcome['status']))
        save()
        if result.returncode or outcome['status'] not in complete:
            record.update(status='blocked',reason='Campaign requires inspection; failure records retained')
            save()
            return 2
        plot(path)
    record['status'] = 'completed' if all(row.get('status')=='exported' for row in record['figures']) else 'completed_pending_acceptance'
    save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
