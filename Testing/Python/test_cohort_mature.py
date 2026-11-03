"""Established DAG/futures execution and strict resource-admission contracts."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import Mock

from shared.compute_schedule import ComputeJob, run_independent_jobs
from shared.cohort_policy import choose_ready
from shared.cohort_profile import record_history
from shared.cohort_resources import validate_request
from shared.cohort_schedule import ENV, create_schedule, stage_admission


class MatureCohortTests(unittest.TestCase):
    def test_dag_releases_a_ready_successor_without_a_wave_barrier(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            code=lambda delay: (sys.executable,'-c',f'import time; time.sleep({delay})')
            jobs=[ComputeJob('slow',code(.6),memory_bytes=1),
                  ComputeJob('fast',code(.08),memory_bytes=1),
                  ComputeJob('successor',code(.05),memory_bytes=1,dependencies=('fast',))]
            record=run_independent_jobs(jobs,directory=root/'run',max_workers=2,cpu_budget=2,memory_budget=2)
            rows={row['name']:row for row in record['jobs']}
            self.assertEqual(record['readiness'],'graphlib.TopologicalSorter')
            self.assertEqual(record['completion'],'concurrent.futures.FIRST_COMPLETED')
            self.assertGreaterEqual(rows['successor']['start_offset_seconds'],rows['fast']['start_offset_seconds']+rows['fast']['runtime_seconds']-.02)
            self.assertLess(rows['successor']['start_offset_seconds'],rows['slow']['runtime_seconds'])

    def test_cycles_and_missing_dependencies_fail_before_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for dependencies in (('absent',),('self',)):
                with self.assertRaises(ValueError):
                    run_independent_jobs([ComputeJob('self',(sys.executable,'-c',''),dependencies=dependencies)],
                        directory=root/'invalid',max_workers=1,cpu_budget=1,memory_budget=1)
                self.assertFalse((root/'invalid').exists())

    def test_failed_dependency_does_not_launch_its_successor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            marker=root/'should_not_exist'
            jobs=[ComputeJob('failed',(sys.executable,'-c','raise SystemExit(3)')),
                  ComputeJob('dependent',(sys.executable,'-c',f'from pathlib import Path;Path({str(marker)!r}).touch()'),dependencies=('failed',))]
            with self.assertRaisesRegex(RuntimeError,'exit code 3'):
                run_independent_jobs(jobs,directory=root/'run',max_workers=2,cpu_budget=2,memory_budget=2)
            record=json.loads((root/'run/schedule.json').read_text())
            self.assertEqual(record['jobs'][1]['status'],'not_run')
            self.assertFalse(marker.exists())

    def test_mem_thread_orders_only_safe_backfill_and_retains_ageing(self):
        limits=dict(cpu=4,memory_gb=16,gpu=1,io=2)
        active=[dict(id='running',cpu=1,memory_gb=1,started_at=90,duration_seconds=30)]
        head=dict(id='head',cpu=4,memory_gb=1,queued_at=90)
        large=dict(id='large',cpu=1,memory_gb=4,queued_at=91,duration_seconds=5)
        small=dict(id='small',cpu=1,memory_gb=1,queued_at=92,duration_seconds=5)
        self.assertEqual(choose_ready([head,large,small],active,limits,100)['id'],'large')
        self.assertEqual(choose_ready([head,large,small],active,limits,100,ready_order='mem_thread')['id'],'small')
        self.assertIsNone(choose_ready([dict(head,bypasses=8),large,small],active,limits,100,ready_order='mem_thread'))
        self.assertEqual(choose_ready([head,small],[],limits,100,ready_order='mem_thread')['id'],'head')

    def test_invalid_resource_contracts_cannot_create_a_queue(self):
        limits=dict(cpu=4,memory_gb=8,gpu=1,io=2)
        normal=dict(cpu=1,memory_gb=1,gpu=0,io=0,duration_seconds=1)
        for extra in (dict(cpu=.5),dict(cpu=True),dict(cpu=0),dict(memory_gb=float('nan')),
                      dict(gpu=2),dict(io=-1),dict(duration_seconds=float('inf'))):
            with self.assertRaises(ValueError):
                validate_request({**normal,**extra},limits)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'invalid'
            with self.assertRaises(ValueError):
                create_schedule(root,cpu_budget=4,memory_gb=8,stage_cpus=1,stage_memory_gb=1,
                                subject_profiles={'A':{'fit':dict(cpu=5,memory_gb=1)}})
            self.assertFalse(root.exists())

    def test_unused_models_do_not_prevent_a_small_selected_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=create_schedule(Path(tmp)/'queue',cpu_budget=2,memory_gb=2,stage_cpus=1,stage_memory_gb=1,
                subject_profiles={'A':{'dti':dict(cpu=1,memory_gb=1),'unused':dict(cpu=1,memory_gb=16)}},
                planning=dict(selected_stages=['dti']))
            self.assertTrue(folder.is_dir())

    def test_new_admission_uses_completed_feedback_from_the_same_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            history=Path(tmp)/'history.json'
            folder=create_schedule(Path(tmp)/'queue',cpu_budget=2,memory_gb=4,stage_cpus=1,stage_memory_gb=1,
                policy='fair_backfill',history_path=history,
                subject_profiles={'A':{'fit':dict(cpu=1,memory_gb=1,profile_key='key')}})
            record_history(history,'key',dict(outcome='completed',samples=3,peak_tree_rss_bytes=1024**3,execution_seconds=10))
            with patch.dict(os.environ,{ENV:str(folder),'SLICERDSPACE_COHORT_SUBJECT':'A'}):
                with stage_admission('fit') as request:
                    self.assertEqual(request['cpu'],1)
                    self.assertEqual(request['memory_gb'],1.5)
                    self.assertEqual(request['memory_floor_gb'],1)
                    self.assertEqual(request['duration_seconds'],12.5)
                    self.assertEqual(request['history_samples'],1)

    def test_waiters_reject_a_dead_coordinator(self):
        import psutil
        with tempfile.TemporaryDirectory() as tmp:
            folder=create_schedule(Path(tmp)/'queue',cpu_budget=2,memory_gb=2,stage_cpus=1,stage_memory_gb=1)
            owner=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
            try:
                config=json.loads((folder/'config.json').read_text())
                config.update(parent_pid=owner.pid,parent_created=psutil.Process(owner.pid).create_time())
                (folder/'config.json').write_text(json.dumps(config))
            finally:
                owner.terminate()
                owner.wait(timeout=5)
            with patch.dict(os.environ,{ENV:str(folder)}):
                with self.assertRaisesRegex(InterruptedError,'coordinator exited'):
                    with stage_admission('fit'):
                        self.fail('Orphan computation was admitted')
            self.assertEqual(json.loads((folder/'state.json').read_text()),dict(active=[],waiting=[]))

    def test_supervisor_transfer_does_not_signal_the_scientific_driver(self):
        from scripts.benchmark_cohort_queue import adopt_controller
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            campaign=root/'dki/campaign.json'
            campaign.parent.mkdir()
            campaign.write_text(json.dumps(dict(status='running',driver_pid=20)))
            state=root/'priority.json'
            state.write_text(json.dumps(dict(status='running',controller_pid=10,
                campaigns=[dict(path=str(campaign))])))
            owner,driver=Mock(pid=10),Mock(pid=20)
            driver.parents.return_value=[owner]
            driver.is_running.return_value=True
            driver.status.return_value='running'
            driver.create_time.return_value=100
            with patch('psutil.Process',side_effect=lambda pid:{10:owner,20:driver}[pid]):
                result=adopt_controller(10,state,campaign,root/'queue.json')
            owner.terminate.assert_called_once_with()
            driver.terminate.assert_not_called()
            driver.kill.assert_not_called()
            self.assertEqual(result['scientific_driver'],20)
            self.assertEqual(json.loads(state.read_text())['status'],'superseded_with_active_campaign')

    def test_candidate_freeze_rejects_math_changes_but_allows_scoped_cache_markers(self):
        from scripts.benchmark_cohort_candidate import cache_normalised_ast
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            old,new=root/'old.py',root/'new.py'
            old.write_text('def fit(x):\n    mark_reused()\n    return x * 2\n')
            new.write_text('def fit(x):\n    mark_reused("native_dti")\n    return x * 2\n')
            self.assertEqual(cache_normalised_ast(old),cache_normalised_ast(new))
            new.write_text('def fit(x):\n    mark_reused("native_dti")\n    return x * 3\n')
            self.assertNotEqual(cache_normalised_ast(old),cache_normalised_ast(new))


if __name__=='__main__':
    unittest.main()
