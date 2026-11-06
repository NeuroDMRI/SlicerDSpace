"""CPU placement must survive launchers and cannot escape a declared cohort set."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from shared.cohort_affinity import cpu_allocation, cpu_placement, isolated_command, affinity_violations
from shared.cohort_schedule import create_schedule
from scripts.benchmark_cohort_workflows import update_comparisons


@unittest.skipUnless(hasattr(os, 'sched_getaffinity'), 'Linux placement required')
class CohortIsolationTests(unittest.TestCase):
    def test_explicit_nonzero_allocation_survives_guard_creation(self):
        with tempfile.TemporaryDirectory() as directory, patch('os.sched_getaffinity', return_value=set(range(64))):
            cpus = list(range(32, 56))
            folder = create_schedule(Path(directory)/'guard', cpu_budget=24, memory_gb=64,
                                     stage_cpus=4, stage_memory_gb=8, cpu_affinity=cpus)
            self.assertEqual(json.loads((folder/'config.json').read_text())['cpu_affinity'], cpus)
            for bad in ([32,32], [64,65], [True,2], [-1,2], [32]):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    cpu_allocation(2, bad)

    def test_placement_never_widens_a_childs_inherited_allocation(self):
        with patch('os.sched_getaffinity', return_value={32,33}), patch('os.sched_setaffinity') as setter:
            with self.assertRaisesRegex(ValueError, 'escape'):
                with cpu_placement([0,1]):
                    self.fail('Escaped placement was admitted')
            setter.assert_not_called()

    def test_taskset_is_applied_before_exec_and_reaches_threads_and_descendants(self):
        cpus = sorted(os.sched_getaffinity(0))[-2:]
        code = """import json,os,subprocess,sys,threading
rows=[sorted(os.sched_getaffinity(0))]
t=threading.Thread(target=lambda:rows.append(sorted(os.sched_getaffinity(0))))
t.start();t.join()
rows.append(json.loads(subprocess.check_output([sys.executable,'-c','import os,json; print(json.dumps(sorted(os.sched_getaffinity(0))))'])))
print(json.dumps(rows))
"""
        result = subprocess.run(isolated_command([sys.executable,'-c',code],cpus),
                                capture_output=True,text=True,check=True,timeout=15)
        self.assertEqual(json.loads(result.stdout), [cpus]*3)
        with patch('shared.cohort_affinity.shutil.which', return_value=None), self.assertRaises(RuntimeError):
            isolated_command([sys.executable,'-V'],cpus)

    def test_live_guard_detects_an_escape_without_moving_the_process(self):
        current = sorted(os.sched_getaffinity(0))
        self.assertEqual(affinity_violations(os.getpid(), current), [])
        if len(current)>1:
            self.assertTrue(affinity_violations(os.getpid(), current[:1]))
        self.assertEqual(sorted(os.sched_getaffinity(0)), current)

    def test_scientific_equivalence_does_not_accept_bad_cpu_isolation_as_a_speedup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trials=[]
            for policy,actual in [('subject',[0,1]),('adaptive',[32,33])]:
                folder=root/policy
                guard=folder/('subject_guard' if policy=='subject' else 'schedule_test')
                guard.mkdir(parents=True)
                (guard/'config.json').write_text(json.dumps({'cpu_affinity':actual}))
                (folder/'scientific_fingerprints.json').write_text(json.dumps({'A':{'array':'same'}}))
                trials.append(dict(id=policy,policy=policy,route='dki',threads=1,status='completed',
                    output=str(folder),qc_status='passed',measurement={'execution_seconds':100}))
            record=dict(trials=trials,subjects=[{'subject':'A'}],reference_policy='subject',
                        cpu_affinity=[32,33],cpus=2,memory_gb=32,jobs=2)
            update_comparisons(record,root)
            result=json.loads((root/'primary_comparison.json').read_text())['comparisons'][0]
            self.assertEqual(result['status'],'invalid_resource_isolation')
            self.assertIsNone(result['speed_ratio'])
            (root/'subject/subject_guard/config.json').write_text(json.dumps({'cpu_affinity':[32,33]}))
            (root/'adaptive/timing_conditions.json').write_text(json.dumps(dict(performance_valid=False,
                reasons=['Interference by another campaign on the same CPU set'])))
            update_comparisons(record,root)
            self.assertIsNone(json.loads((root/'primary_comparison.json').read_text())['comparisons'][0]['speed_ratio'])
