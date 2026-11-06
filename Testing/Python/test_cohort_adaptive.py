"""CPU-first admission, reservation fairness and measured feedback contracts."""

import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from shared.cohort_policy import choose_ready
from shared.cohort_profile import StageMonitor, history_estimate, read_history, record_history
from shared.cohort_planning import plan_cohort
from shared.cohort_schedule import ENV, create_schedule, stage_admission
from scripts.run_batch_pipeline import build_parser


class AdaptiveCohortTests(unittest.TestCase):
    limits = dict(cpu=8, memory_gb=16, gpu=1, io=2)

    def task(self, name, cpu, duration=0, **extra):
        return dict(id=name, cpu=cpu, memory_gb=1, gpu=0, io=0,
                    duration_seconds=duration, queued_at=0, **extra)

    def test_short_backfill_preserves_head_reservation(self):
        active = [self.task('running', 4, 10, started_at=0)]
        head = self.task('head', 8)
        short = self.task('short', 2, 3)
        long = self.task('long', 2, 30)
        self.assertEqual(choose_ready([head, long, short], active, self.limits, 1)['id'], 'short')
        self.assertIsNone(choose_ready([head, long], active, self.limits, 1))

    def test_reservation_includes_tasks_still_running_at_head_start(self):
        active = [self.task('early', 2, 10, started_at=0), self.task('late', 4, 100, started_at=0)]
        head = self.task('head', 4)
        self.assertIsNone(choose_ready([head, self.task('unsafe', 2, 200)], active, self.limits, 1))

    def test_aged_or_overtaken_head_has_no_barrier_but_keeps_its_reservation(self):
        # T3 S3: the 120 s / 8-bypass ageing barrier is removed. The head reservation stays.
        active = [self.task('running', 4)]
        for head, now in ((self.task('head', 8), 121), (self.task('head', 8, bypasses=8), 1)):
            self.assertIsNone(choose_ready([head, self.task('small', 1)], active, self.limits, now))
            self.assertEqual(choose_ready([head], [], self.limits, now)['id'], 'head')

    def test_gpu_waiter_does_not_prevent_compatible_cpu_work(self):
        active = [dict(self.task('inference', 1), gpu=1)]
        head = dict(self.task('next-inference', 1), gpu=1)
        cpu = self.task('fit', 2)
        self.assertEqual(choose_ready([head, cpu], active, self.limits, 1)['id'], 'fit')

    def test_io_and_memory_budgets_are_not_ignored(self):
        for field, used in (('io', 2), ('memory_gb', 16)):
            active = [dict(self.task('active', 1), **{field: used})]
            waiting = [dict(self.task('head', 1), **{field: 1})]
            self.assertIsNone(choose_ready(waiting, active, self.limits, 1))

    def test_feedback_ignores_failed_and_nonfinite_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'history.json'
            sample = dict(outcome='completed', samples=3, peak_tree_rss_bytes=1024**3, execution_seconds=10)
            record_history(path, 'key', sample)
            record_history(path, 'key', dict(sample, outcome='failed', peak_tree_rss_bytes=999 * 1024**3))
            record_history(path, 'key', dict(sample, execution_seconds=float('nan')))
            estimate = history_estimate(read_history(path), 'key')
            self.assertAlmostEqual(estimate['memory_gb'], 1.3)
            self.assertEqual(estimate['duration_seconds'], 12.5)
            self.assertEqual(estimate['samples'], 1)

    def test_adaptive_frontier_is_not_whole_subject_peak_capacity(self):
        subjects = [SimpleNamespace(subject_id=str(i), subject_root=Path('/absent') / str(i)) for i in range(10)]
        args = ['--targets', 'microstructure', '--microstructure-stages', 'dti']
        plan = plan_cohort(subjects, args, available=(16, 32 * 1024**3), adaptive=True)
        self.assertEqual(plan['cpu_budget'], 12)
        self.assertEqual(plan['stage_cpus'], 4)
        self.assertGreater(plan['jobs'], plan['cpu_budget'] // plan['stage_cpus'])
        self.assertEqual(plan['subject_profiles']['0']['wma']['cpu'], 1)
        self.assertEqual(plan['subject_profiles']['0']['microstructure']['io'], 0)
        self.assertEqual(plan['subject_profiles']['0']['wma']['io'], 0)
        self.assertEqual(plan['policy'], 'fair_backfill')

    def test_measured_history_can_raise_but_not_lower_memory(self):
        subjects = [SimpleNamespace(subject_id='A', subject_root=Path('/absent/A'))]
        args = ['--targets', 'connectome']
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'history.json'
            plan = plan_cohort(subjects, args, adaptive=True, available=(8, 64 * 1024**3))
            key = plan['subject_profiles']['A']['connectome']['profile_key']
            for gib in (.1, 10):
                record_history(path, key, dict(outcome='completed', samples=3,
                    peak_tree_rss_bytes=gib*1024**3, execution_seconds=10))
                updated = plan_cohort(subjects, args, adaptive=True, history_path=path,
                                      available=(8, 64 * 1024**3))
                self.assertGreaterEqual(updated['subject_profiles']['A']['connectome']['memory_gb'], 4)
            self.assertEqual(updated['subject_profiles']['A']['connectome']['memory_gb'], 13)

    def test_automatic_frontier_is_budget_bound_not_fixed_at_eight(self):
        subjects = [SimpleNamespace(subject_id=str(i), subject_root=Path('/absent') / str(i)) for i in range(20)]
        args = ['--targets', 'microstructure', '--microstructure-stages', 'dti,noddi,dki']
        plan = plan_cohort(subjects, args, adaptive=True, cpu_budget=40, memory_gb=192)
        self.assertEqual(plan['jobs'], 20)
        limited = plan_cohort(subjects, args, adaptive=True, cpu_budget=6, memory_gb=192)
        self.assertEqual(limited['jobs'], 6)
        explicit = plan_cohort(subjects, args, adaptive=True, jobs=4, cpu_budget=40, memory_gb=192)
        self.assertEqual(explicit['jobs'], 4)

    def test_malformed_history_is_optional_not_a_workflow_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'history.json'
            for payload in ({'version': 1, 'profiles': []}, {'version': 1, 'profiles': {'bad': 1, 'key': ['bad']}}):
                path.write_text(json.dumps(payload))
                self.assertIsNone(history_estimate(read_history(path), 'key'))

    def test_new_cli_options_are_real_and_invalid_budgets_fail(self):
        parser = build_parser()
        args = parser.parse_args(['--batch-root', '/tmp', '--scheduler', 'adaptive',
                                  '--resource-budget', '80%', '--subject-limit', '20'])
        self.assertEqual(args.resource_budget, .8)
        self.assertEqual(args.subject_limit, 20)
        with patch('sys.stderr'):
            for value in ('0', '101%', 'NaN', '-1'):
                with self.assertRaises(SystemExit):
                    parser.parse_args(['--batch-root', '/tmp', '--resource-budget', value])

    def test_adaptive_stage_records_usage_and_releases_after_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = Path(tmp) / 'profiles.json'
            folder = create_schedule(Path(tmp) / 'schedule', cpu_budget=2, memory_gb=2,
                                     stage_cpus=1, stage_memory_gb=1, policy='fair_backfill',
                                     history_path=history,
                                     subject_profiles={'A': {'fit': dict(cpu=1, memory_gb=1, profile_key='fit')}})
            with patch.dict(os.environ, {ENV: str(folder), 'SLICERDSPACE_COHORT_SUBJECT': 'A'}):
                with self.assertRaises(ValueError):
                    with stage_admission('fit'):
                        raise ValueError('scientific error')
            self.assertEqual(json.loads((folder / 'state.json').read_text()), dict(active=[], waiting=[]))
            row = read_history(history)['profiles']['fit'][0]
            self.assertEqual(row['outcome'], 'failed')
            self.assertGreater(row['peak_tree_rss_bytes'], 0)

    def test_monitor_reports_cpu_and_memory_without_changing_computation(self):
        monitor = StageMonitor(.01).start()
        result = sum(i*i for i in range(100000))
        time.sleep(.03)
        measured = monitor.finish('completed')
        self.assertEqual(result, 333328333350000)
        self.assertGreaterEqual(measured['samples'], 2)
        self.assertGreater(measured['peak_tree_rss_bytes'], 0)


if __name__ == '__main__':
    unittest.main()
