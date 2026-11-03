#!/usr/bin/env python3
"""Prepare a fresh, budget-only, multi-route cohort comparison on one snapshot."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_cohort_workflows import save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-campaign', type=Path, required=True)
    parser.add_argument('--long-campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpus', type=int, default=40)
    parser.add_argument('--memory-gb', type=float, default=192)
    args = parser.parse_args()
    if args.cpus < 8 or args.memory_gb < 32:
        parser.error('This full-volume multi-model campaign needs at least 8 CPUs and 32 GiB')
    record = deepcopy(json.loads(args.base_campaign.read_text()))
    longer = json.loads(args.long_campaign.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    from scripts.stage_slicer_extension import stage_extension
    staged = stage_extension(args.output/'install', slicer_version='Slicer-5.10', clean=False, link_mode='copy')
    snapshot = Path(staged['module_dir'])
    for name in ('benchmark_cohort_workflows.py', 'benchmark_cohort_compare.py',
                 'benchmark_subject_worker.py', 'benchmark_budget_cohort.py'):
        shutil.copy2(ROOT/'scripts'/name, snapshot/'scripts'/name)
    hashes = {str(p.relative_to(snapshot)): sha(p) for p in snapshot.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts}
    common = list(record['common'])
    common[common.index('--max-parallel-models')+1] = '2'
    common += ['--microstructure-free-water-n-jobs', '2', '--dki-threads', '4']
    mixed = ['--targets','validation,microstructure','--microstructure-stages','dti,freewater,noddi']
    routes = dict(mixed=mixed, mixed_dki=[*mixed[:-1], 'dti,freewater,noddi,dki', '--dki-fit-method','WLS'],
                  tracts=list(longer['routes']['tracts']), mkcurve=list(longer['routes']['mkcurve']))
    # No GPU-heavy anatomical inference while other users occupy both devices.
    routes['tracts'][routes['tracts'].index('--targets')+1] = 'validation,microstructure,tractography,bundles'
    routes['tracts'][routes['tracts'].index('--tractography-backend')+1] = 'ukf'
    trials = []
    for i, route in enumerate(routes):
        policies = ('adaptive','subject','stage') if i % 2 == 0 else ('subject','stage','adaptive')
        for policy in policies:
            trials.append(dict(id=f'{route}-auto-{policy}', route=route, policy=policy,
                               threads=4, repeat=0, status='pending'))
    for key in ('driver_pid','started_at','ended_at','inputs_unchanged','execution_harness',
                'candidate_provenance','continuation','selected_from'):
        record.pop(key, None)
    record.update(status='prepared', snapshot=str(snapshot), source_hashes=hashes,
                  common=common, routes=routes, trials=trials, jobs=None, automatic_jobs=True,
                  reference_policy='subject', cpus=args.cpus, memory_gb=args.memory_gb,
                  cpu_affinity=sorted(os.sched_getaffinity(0))[:args.cpus],
                  external_asset_hashes=longer['external_asset_hashes'],
                  notes=['Fresh full-volume outputs, 12 distinct participants, one identical snapshot for all arms',
                         'No manually selected subject concurrency: baseline uses conservative whole-subject capacity; stage/adaptive use automatic planning',
                         'Two independent model workers per participant in all arms; identical numerical thread arguments',
                         'Report observed concurrency separately: automatic admission gains include utilisation and frontier effects',
                         'CPU-only routes; GPUs are occupied by unrelated jobs; OS cache not cleared',
                         'Single pilot repeat with alternating policy order across routes; no confidence interval'])
    save(args.output/'campaign.json', record)
    print(json.dumps(dict(campaign=str(args.output/'campaign.json'), snapshot=str(snapshot),
                         subjects=len(record['subjects']), trials=len(trials), cpu_affinity=record['cpu_affinity']), indent=2))


if __name__ == '__main__':
    main()
