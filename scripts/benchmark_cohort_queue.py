#!/usr/bin/env python3
"""Continue frozen campaigns serially and draw only accepted completed results."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-for',type=Path,required=True)
    parser.add_argument('--then',type=Path,nargs='+',required=True)
    parser.add_argument('--record',type=Path,required=True)
    parser.add_argument('--plotter',type=Path,required=True)
    parser.add_argument('--figure-root',type=Path,required=True)
    parser.add_argument('--max-hours',type=float,default=168)
    args = parser.parse_args()
    if args.record.exists():
        raise FileExistsError('Queue record already exists')
    record = dict(status='waiting', waiting_for=str(args.wait_for), campaigns=[],
                  plotter=str(args.plotter),plotter_sha256=hashlib.sha256(args.plotter.read_bytes()).hexdigest())
    def save():
        temporary = args.record.with_suffix('.tmp')
        temporary.write_text(json.dumps(record,indent=2)+'\n')
        temporary.replace(args.record)
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
    campaigns = [args.wait_for]
    for path in args.then:
        campaign = json.loads(path.read_text())
        command = [campaign['launcher'],'--launch','PythonSlicer',
                   str(Path(campaign['snapshot'])/'scripts/benchmark_cohort_workflows.py'),
                   'run','--output',str(path.parent),'--max-hours',str(args.max_hours)]
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
        campaigns.append(path)
    record['figures'] = []
    if hashlib.sha256(args.plotter.read_bytes()).hexdigest() != record['plotter_sha256']:
        record.update(status='blocked',reason='Plotter changed after queue creation')
        save()
        return 2
    for path in campaigns:
        outcome = json.loads(path.read_text())
        if outcome['status'] != 'passed':
            record['figures'].append(dict(campaign=str(path),status='withheld_pending_acceptance'))
            continue
        command = [sys.executable,str(args.plotter),'--campaign',str(path),
                   '--output',str(args.figure_root/path.parent.name)]
        result = subprocess.run(command,capture_output=True,text=True)
        record['figures'].append(dict(campaign=str(path),returncode=result.returncode,
                                     status='exported' if not result.returncode else 'failed',error=result.stderr))
    record['status'] = 'completed' if all(row.get('status')=='exported' for row in record['figures']) else 'completed_pending_acceptance'
    save()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
