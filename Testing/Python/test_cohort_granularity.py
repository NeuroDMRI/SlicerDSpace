"""Model-level leases, cached-fit feedback and bounded resource telemetry."""

import json
import os
import sys
from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from shared.cohort_policy import reservation_time
from shared.cohort_profile import history_estimate
from shared.cohort_schedule import ENV, create_schedule, scheduled_stage
from shared.cohort_telemetry import CohortTelemetry, gpu_sample


class ModelGranularityTests(unittest.TestCase):
    def test_microstructure_controller_has_no_parent_lease_and_leaves_are_wrapped(self):
        from microstructure import pipeline as p
        from microstructure.dki import run_dki
        self.assertFalse(hasattr(p.run_microstructure, '__wrapped__'))
        for function in (p._export_dti_scalar_maps, p._compute_free_water_maps, p._compute_noddi_maps, run_dki):
            self.assertTrue(hasattr(function, '__wrapped__'))

    def test_method_names_and_cached_results_are_recorded_without_learning(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = create_schedule(Path(tmp)/'queue', cpu_budget=2, memory_gb=2,
                                     stage_cpus=1, stage_memory_gb=1, policy='fair_backfill')
            @scheduled_stage(lambda config: 'fit_' + config.method)
            def fit(config):
                return SimpleNamespace(status='cached')
            with patch.dict(os.environ, {ENV: str(folder)}):
                fit(SimpleNamespace(method='dki', device='cpu'))
            rows = [json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
            self.assertEqual(rows[-1]['stage'], 'fit_dki')
            measurement = rows[-1]['measurement']
            self.assertTrue(measurement['cache_hit'])
            self.assertIsNone(history_estimate({'profiles': {'key': [measurement]}}, 'key'))

    def test_overdue_running_task_is_not_assumed_to_have_released_resources(self):
        import math
        limits = dict(cpu=4, memory_gb=4, gpu=1, io=2)
        head = dict(cpu=4, memory_gb=1)
        active = [dict(cpu=2, memory_gb=1, duration_seconds=2, started_at=0)]
        self.assertTrue(math.isinf(reservation_time(head, active, limits, 10)))

    def test_failed_monitor_cannot_leak_a_reservation(self):
        monitors = []

        def fail(monitor, outcome):
            monitors.append(monitor)
            raise RuntimeError('probe failed')

        def stop_samplers():
            # The mocked finish never stops the sampler thread; stop and join it here (Opus P2).
            for monitor in monitors:
                monitor.stop_event.set()
                thread = getattr(monitor, 'thread', None)
                if thread:
                    thread.join(timeout=5)
                    self.assertFalse(thread.is_alive())
        self.addCleanup(stop_samplers)
        with tempfile.TemporaryDirectory() as tmp:
            folder = create_schedule(Path(tmp)/'queue', cpu_budget=2, memory_gb=2,
                                     stage_cpus=1, stage_memory_gb=1, policy='fair_backfill')
            @scheduled_stage('fit')
            def fit(config):
                return 1
            with patch.dict(os.environ, {ENV: str(folder)}), \
                 patch('shared.cohort_profile.StageMonitor.finish', autospec=True, side_effect=fail), \
                 self.assertWarns(RuntimeWarning):
                self.assertEqual(fit(SimpleNamespace(device='cpu')), 1)
            self.assertEqual(json.loads((folder/'state.json').read_text()), dict(active=[], waiting=[]))

    def test_model_selection_and_explicit_workers_have_matching_reservations(self):
        from shared.cohort_planning import plan_cohort
        subjects = [SimpleNamespace(subject_id='A', subject_root=Path('/absent'))]
        plan = plan_cohort(subjects, ['--targets','microstructure','--microstructure-stages','dti',
            '--microstructure-stages','freewater,dki','--dki-fit-method','MKCURVE',
            '--microstructure-free-water-n-jobs','3'], adaptive=True, available=(16, 128*1024**3))
        self.assertIn('microstructure_mkcurve', plan['selected_stages'])
        self.assertNotIn('microstructure', plan['selected_stages'])
        self.assertEqual(plan['subject_profiles']['A']['microstructure_freewater']['cpu'], 3)

    def test_ukf_automatic_threads_obey_the_cohort_reservation(self):
        from tractography.ukf import parameters
        with patch.dict(os.environ, {ENV:'/unused', 'SLICERDSPACE_STAGE_CPUS':'2'}):
            self.assertEqual(parameters().numThreads, 2)
            with self.assertRaisesRegex(ValueError, 'CPU reservation'):
                parameters({'numThreads':4})

    def test_reusing_a_prerequisite_does_not_mark_the_outer_fit_cached(self):
        from shared.cohort_schedule import mark_reused
        with tempfile.TemporaryDirectory() as tmp:
            folder = create_schedule(Path(tmp)/'queue', cpu_budget=2, memory_gb=2,
                                     stage_cpus=1, stage_memory_gb=1, policy='fair_backfill')
            @scheduled_stage('registered_dti')
            def fit():
                mark_reused('native_dti')
            with patch.dict(os.environ, {ENV:str(folder)}):
                fit()
            row = json.loads((folder/'events.jsonl').read_text().splitlines()[-1])
            self.assertFalse(row['measurement']['cache_hit'])

    @unittest.skipUnless(hasattr(os,'sched_setaffinity'), 'Linux affinity required')
    def test_cpu_placement_limits_all_current_threads_and_restores_after_error(self):
        from shared.cohort_affinity import cpu_placement
        import psutil
        previous = os.sched_getaffinity(0)
        with self.assertRaises(ValueError):
            with cpu_placement([min(previous)]):
                self.assertTrue(all(os.sched_getaffinity(t.id)=={min(previous)} for t in psutil.Process().threads()))
                raise ValueError('fixture')
        self.assertEqual(os.sched_getaffinity(0), previous)

    def test_gpu_context_and_owned_vram_are_distinguished(self):
        from subprocess import CompletedProcess
        responses = [CompletedProcess([], 0, 'GPU-test, 99, 4000\n'),
                     CompletedProcess([], 0, '123, 300, GPU-test\n999, 3700, GPU-test\n')]
        with patch('shared.cohort_telemetry.subprocess.run', side_effect=responses):
            row = gpu_sample({123}, executable='nvidia-smi')
        self.assertEqual(row['owned_vram_mib'], 300)
        self.assertEqual(row['devices'][0]['utilization_percent'], 99)
        self.assertIn('unrelated users', row['utilization_scope'])

    def test_io_permit_is_released_on_model_error(self):
        from shared.cohort_io import io_admission
        with tempfile.TemporaryDirectory() as tmp:
            folder = create_schedule(Path(tmp)/'queue', cpu_budget=2, memory_gb=2,
                                     stage_cpus=1, stage_memory_gb=1, io_slots=1)
            with patch.dict(os.environ, {ENV:str(folder)}):
                with self.assertRaises(ValueError):
                    with io_admission('decompression'):
                        raise ValueError('bad input')
            self.assertEqual(json.loads((folder/'io_state.json').read_text()), dict(active=[], waiting=[]))

    def test_unavailable_gpu_is_missing_data_not_zero_utilization(self):
        with patch('shared.cohort_telemetry.shutil.which', return_value=None):
            row = gpu_sample(set())
        self.assertFalse(row['available'])
        self.assertIsNone(row['owned_vram_mib'])

    def test_trace_reports_owned_process_tree_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'resources.jsonl'
            missing = dict(available=False, owned_vram_mib=None, devices=[])
            with patch('shared.cohort_telemetry.gpu_sample', return_value=missing):
                monitor = CohortTelemetry(path, .01).start()
                import time
                time.sleep(.03)
                result = monitor.finish('completed')
            self.assertGreater(result['peak_tree_rss_bytes'], 0)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertTrue(rows)
            self.assertIn('cpu_seconds', rows[0])
            self.assertIn('rss_bytes', rows[0])
            with self.assertRaises(FileExistsError):
                CohortTelemetry(path).start()


def _sampler_threads():
    """Alive threads whose target function lives in shared.cohort_profile (StageMonitor sampler)."""
    return {t for t in threading.enumerate()
            if t.is_alive() and getattr(getattr(t, '_target', None), '__module__', None) == 'shared.cohort_profile'}


class MonitorThreadCleanupTests(unittest.TestCase):
    def test_failed_monitor_test_leaves_no_cohort_profile_sampler_thread(self):
        try:
            import psutil  # noqa: F401
        except ImportError:
            self.skipTest("psutil is not installed: the inner test cannot create a schedule "
                          "(shared/cohort_schedule.py:37), so no sampler thread starts")
        # Finding: test_failed_monitor_cannot_leak_a_reservation mocked StageMonitor.finish, so the
        # daemon sampler thread was never stopped and ran for the rest of the process (Opus P2).
        # Only threads created during this run count, so a thread leaked by another module cannot fail it.
        before = _sampler_threads()
        case = ModelGranularityTests('test_failed_monitor_cannot_leak_a_reservation')
        result = unittest.TestResult()
        case.run(result)
        leaked = _sampler_threads() - before
        # If this fails, the leaked threads keep running: nothing reachable from a thread object stops them.
        self.assertEqual(result.testsRun, 1)
        self.assertTrue(result.wasSuccessful(), f'errors={result.errors} failures={result.failures}')
        self.assertFalse(leaked, f'leaked StageMonitor sampler threads: {leaked}')

    def test_monitor_cleanup_check_skips_without_psutil(self):
        # Finding (CI run 45743369148): without psutil the inner test errors at create_schedule, so the
        # cleanup check reported a duplicate failure for a missing dependency instead of skipping.
        with patch.dict(sys.modules, {'psutil': None}):
            result = unittest.TestResult()
            MonitorThreadCleanupTests(
                'test_failed_monitor_test_leaves_no_cohort_profile_sampler_thread').run(result)
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(len(result.skipped), 1)
        self.assertEqual(result.failures, [])
        self.assertEqual(result.errors, [])


if __name__ == '__main__':
    unittest.main()
