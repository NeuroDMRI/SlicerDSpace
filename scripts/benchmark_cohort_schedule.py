#!/usr/bin/env python3
"""Controlled synthetic admission benchmark, not MRI algorithm timing."""

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
import platform
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.cohort_schedule import create_schedule, stage_admission, ENV


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--worker', choices=('subjects', 'stages'))
    parser.add_argument('--subject', default='fixture')
    args = parser.parse_args()
    if args.worker:
        outer = stage_admission('subject_reservation', gpu=True) if args.worker == 'subjects' else nullcontext()
        with outer:
            for name, gpu, duration in (('inference_fixture', True, .20), ('fit_fixture', False, .40)):
                with stage_admission(name, gpu=gpu):
                    time.sleep(duration)
            # Identical deterministic output regardless of admission order.
            args.output.write_text(json.dumps(dict(subject=args.subject, values=[i * i for i in range(100)])))
        return
    if args.repeats < 1:
        parser.error('repeats must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    runs, hashes = [], {}
    for repeat in range(args.repeats):
        for mode in (('subjects', 'stages') if repeat % 2 == 0 else ('stages', 'subjects')):
            folder = create_schedule(args.output / f'{repeat}-{mode}', cpu_budget=2, memory_gb=2,
                                     stage_cpus=1, stage_memory_gb=1)
            children = []
            started = time.monotonic()
            try:
                for index in range(3):
                    env = {**os.environ, ENV: str(folder), 'OPENBLAS_NUM_THREADS': '1',
                           'SLICERDSPACE_COHORT_SUBJECT': str(index)}
                    children.append(subprocess.Popen([sys.executable, __file__, '--worker', mode,
                        '--subject', str(index), '--output', str(folder / f'{index}.json')], env=env))
                for child in children:
                    if child.wait(timeout=30):
                        raise RuntimeError('Synthetic worker failed')
            finally:
                for child in children:
                    if child.poll() is None:
                        child.kill()
                        child.wait()
            runs.append(dict(repeat=repeat, mode=mode, seconds=time.monotonic() - started))
            for index in range(3):
                digest = hashlib.sha256((folder / f'{index}.json').read_bytes()).hexdigest()
                if index in hashes and hashes[index] != digest:
                    raise RuntimeError('Synthetic output changed')
                hashes[index] = digest
    medians = {mode: statistics.median(row['seconds'] for row in runs if row['mode'] == mode)
               for mode in ('subjects', 'stages')}
    report = dict(status='PASS', scope='Synthetic sleep-based scheduling fixture; no MRI throughput or algorithm speedup claim',
                  comparison='Same declared CPU/RAM/GPU-slot limits; whole-subject versus per-stage reservations',
                  subjects=3, cpu_budget=2, memory_gb=2, gpu_slots=1, runs=runs, median_seconds=medians,
                  output_sha256=hashes, outputs_identical=True,
                  environment=dict(python=sys.version, platform=platform.platform()),
                  source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                 for name in ('shared/cohort_schedule.py', 'scripts/benchmark_cohort_schedule.py')})
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
