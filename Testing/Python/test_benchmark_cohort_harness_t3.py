"""T3 paired-benchmark harness (H1-H8): launch rule, arm arguments, audit, pair order, QC classes.

Mocks and tiny synthetic files only; no server, no GPU, no Slicer.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import time
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from scripts import benchmark_cohort_compare as compare
from scripts import benchmark_cohort_harness as harness
from scripts import benchmark_cohort_workflows as workflows


def topology():
    """2 nodes, sibling CPU c and c+56 (fan93 shape)."""
    return {0: [(c, c + 56) for c in range(0, 28)], 1: [(c, c + 56) for c in range(28, 56)]}


def idle_busy(overrides=None, default=2.0):
    busy = {cpu: default for cpu in range(112)}
    busy.update(overrides or {})
    return busy


def gpu_sample(free_mib, util=10.0, total=81920.0):
    return [dict(index=i, used_mib=total - free, total_mib=total, utilization_percent=util + i)
            for i, free in enumerate(free_mib)]


class LaunchRuleTests(unittest.TestCase):
    def test_core_choice_takes_lowest_busy_physical_cores_and_both_siblings(self):
        busy = idle_busy({c: 90.0 for c in range(4, 12)})       # node0 cores 4-11 busy
        chosen = harness.choose_cores(busy, topology())
        node0 = chosen[0]['cpus']
        self.assertEqual(len(node0), 32)
        self.assertFalse(set(node0) & harness.EXCLUDED_CPUS)
        self.assertFalse(set(node0) & set(range(4, 12)))
        for cpu in node0:
            sibling = cpu + 56 if cpu < 56 else cpu - 56
            self.assertIn(sibling, node0)
        self.assertTrue(set(chosen[1]['cpus']) <= set(range(28, 56)) | set(range(84, 112)))

    def test_busy_threshold_is_25_percent_of_chosen_set(self):
        topo = topology()
        ok = harness.decide_launch(idle_busy(default=25.0), topo, [], 10**12, hybrid=False)
        self.assertTrue(ok['ok'])
        bad = harness.decide_launch(idle_busy(default=25.5), topo, [], 10**12, hybrid=False)
        self.assertFalse(bad['ok'])
        self.assertEqual(len(bad['reasons']), 2)                 # both nodes fail

    def test_gpu_rule_needs_two_gpus_with_14_gib_free_in_every_sample(self):
        topo = topology()
        gib = 1024.0
        good = [gpu_sample([60 * gib, 50 * gib, 5 * gib, 40 * gib]) for _ in range(3)]
        decision = harness.decide_launch(idle_busy(), topo, good, 10**12, hybrid=True)
        self.assertTrue(decision['ok'])
        self.assertEqual(decision['gpus'], [0, 1])               # lowest mean utilisation first
        dip = [gpu_sample([60 * gib, 50 * gib, 5 * gib, 40 * gib]),
               gpu_sample([60 * gib, 13.9 * gib, 5 * gib, 13.9 * gib])]       # GPUs 1 and 3 dip under 14 GiB once
        decision = harness.decide_launch(idle_busy(), topo, dip, 10**12, hybrid=True)
        self.assertFalse(decision['ok'])
        self.assertIn('need 2', ' '.join(decision['reasons']))
        # the CPU workload ignores GPUs
        self.assertTrue(harness.decide_launch(idle_busy(), topo, dip, 10**12, hybrid=False)['ok'])

    def test_gpu_choice_prefers_lowest_mean_utilisation(self):
        gib = 1024.0
        samples = [[dict(index=i, used_mib=1000., total_mib=81920., utilization_percent=u)
                    for i, u in enumerate((90., 5., 40., 10.))] for _ in range(2)]
        self.assertEqual(harness.choose_gpus(samples)[0], [1, 3])
        self.assertEqual(harness.MIN_FREE_GPU_MIB, 14 * gib)

    def test_disk_rule_needs_500_gb(self):
        topo = topology()
        self.assertFalse(harness.decide_launch(idle_busy(), topo, [], 499 * 10**9, hybrid=False)['ok'])
        self.assertTrue(harness.decide_launch(idle_busy(), topo, [], 500 * 10**9, hybrid=False)['ok'])

    def test_eight_attempts_then_stop_with_15_minute_waits(self):
        sleeps, calls = [], []

        def attempt(number):
            calls.append(number)
            return dict(ok=False, reasons=['busy'])
        with self.assertRaises(harness.LaunchRefused) as caught:
            harness.run_launch_rule(attempt, sleep=sleeps.append)
        self.assertEqual(calls, list(range(1, 9)))
        self.assertEqual(sleeps, [900] * 7)
        self.assertEqual(len(caught.exception.attempts), 8)

    def test_success_on_third_attempt_stops_retrying(self):
        outcomes = iter([dict(ok=False), dict(ok=False), dict(ok=True)])
        chosen, attempts = harness.run_launch_rule(lambda n: next(outcomes), sleep=lambda s: None)
        self.assertEqual(chosen['attempt'], 3)
        self.assertEqual(len(attempts), 3)

    def test_mpstat_parser_averages_interval_rows_and_skips_average(self):
        text = textwrap.dedent("""\
            12:00:00     CPU    %usr   %nice    %sys %iowait    %irq   %soft  %steal  %guest  %gnice   %idle
            12:00:10     all    1.0     0.0     1.0     0.0     0.0     0.0     0.0     0.0     0.0    98.0
            12:00:10       0   40.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0    60.0
            12:00:20       0   20.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0    80.0
            Average:       0   99.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     0.0     1.0
            """)
        self.assertEqual(harness.parse_mpstat(text), {0: 30.0})

    def test_repeat_2_swaps_nodes_and_gpus(self):
        decision = harness.decide_launch(idle_busy(), topology(), [gpu_sample([60000., 60000., 1.])] * 2, 10**12,
                                         hybrid=True)
        first = harness.assign_arms(decision, 1)
        self.assertEqual((first['subject']['node'], first['adaptive']['node']), (0, 1))
        second = harness.assign_arms(decision, 2, previous=dict(subject_gpu=first['subject']['gpu'],
                                                                adaptive_gpu=first['adaptive']['gpu']))
        self.assertEqual((second['subject']['node'], second['adaptive']['node']), (1, 0))
        self.assertEqual(second['subject']['gpu'], first['adaptive']['gpu'])


class ArgumentTests(unittest.TestCase):
    def record(self, route='cpu', gpu=None):
        args = types.SimpleNamespace(models=Path('/m'), mrtrix=Path('/x'), wma=Path('/w'), atlas=Path('/a'),
                                     launcher=Path('/l'), ukf_executable=None)
        return dict(launcher='/l', cpus=32, memory_gb=224.0, kill_memory_gb=256.0, jobs=4, gpu=gpu,
                    common=workflows.common_arguments(args), routes=workflows.route_options(args, [route]),
                    subjects=[])

    def trial(self, policy, route='cpu'):
        return dict(id=f'{route}-{policy}', route=route, policy=policy, threads=8, repeat=0, status='pending')

    def command(self, policy, route='cpu'):
        record = self.record(route)
        trial = self.trial(policy, route)
        _, _, jobs = workflows.lease(record, trial)
        return workflows.build_command(record, trial, Path('/d'), Path('/s'), jobs), record, trial

    def test_adaptive_has_no_jobs_and_no_thread_flags(self):
        command, record, trial = self.command('adaptive')
        for flag in ('--jobs', '--surface-cpu-threads', '--mrtrix-nthreads', '--batch-stage-cpus'):
            self.assertNotIn(flag, command)
        self.assertEqual(command[command.index('--scheduler') + 1], 'adaptive')
        env = workflows.trial_environment(record, trial, base={})
        for name in harness.THREAD_ENV:
            self.assertNotIn(name, env)

    def test_subject_has_jobs_4_and_threads_8(self):
        command, record, trial = self.command('subject')
        self.assertEqual(command[command.index('--jobs') + 1], '4')
        self.assertEqual(command[command.index('--surface-cpu-threads') + 1], '8')
        self.assertEqual(command[command.index('--mrtrix-nthreads') + 1], '8')
        self.assertEqual(workflows.lease(record, trial), (8, 56.0, 4))
        env = workflows.trial_environment(record, trial, base={})
        self.assertTrue(all(env[name] == '8' for name in harness.THREAD_ENV))

    def test_both_arms_share_budget_and_skip_environment_check(self):
        for policy in ('subject', 'adaptive'):
            for route in ('cpu', 'hybrid'):
                command, _, _ = self.command(policy, route)
                self.assertIn('--skip-environment-check', command)
                self.assertNotIn('--strict-environment', command)
                self.assertEqual(command[command.index('--batch-memory-gb') + 1], '224.0')
                self.assertEqual(command[command.index('--batch-cpus') + 1], '32')

    def test_kill_limit_and_gpu_environment(self):
        record = self.record('hybrid', gpu=3)
        env = workflows.trial_environment(record, self.trial('subject', 'hybrid'), base={})
        self.assertEqual((env['CUDA_DEVICE_ORDER'], env['CUDA_VISIBLE_DEVICES']), ('PCI_BUS_ID', '3'))
        self.assertEqual(record['kill_memory_gb'], 256.0)
        self.assertEqual(harness.KILL_MEMORY_GB, 256.0)
        self.assertEqual(harness.BATCH_MEMORY_GB, 224.0)
        self.assertEqual(workflows.trial_environment(self.record('cpu'), self.trial('adaptive'), base={})
                         ['CUDA_VISIBLE_DEVICES'], '')

    def test_hybrid_has_no_mrtrix_seed(self):
        command, _, _ = self.command('subject', 'hybrid')
        self.assertNotIn('--mrtrix-seed', command)
        self.assertIn('connectome', command[command.index('--targets') + 1])

    def test_ids_and_names_must_be_one_path_component(self):
        for bad in ('', '.', '..', 'a/b', 'a\\b', ' a', 'a\0'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                harness.safe_component(bad)
        self.assertEqual(harness.safe_component('100307'), '100307')
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            workflows.inspect_cohort(Path(directory), 1, names=['../etc'])

    def test_claim_folder_refuses_an_existing_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'pair1'
            harness.claim_folder(target)
            (target / 'keep').write_text('x')
            with self.assertRaises(FileExistsError):
                harness.claim_folder(target)
            self.assertEqual((target / 'keep').read_text(), 'x')

    def test_cpu_list_parsing(self):
        self.assertEqual(harness.parse_cpu_list('4-5,60', 3), [4, 5, 60])
        for bad in ('4-5,5', '', '7-3'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                harness.parse_cpu_list(bad)
        with self.assertRaises(ValueError):
            harness.parse_cpu_list('1-3', 4)


class AuditTests(unittest.TestCase):
    def test_every_subject_is_audited_after_return_code_2(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            names = ['s1', 's2', 's3']
            (destination / 'outputs').mkdir()
            (destination / 'batch.json').write_text(json.dumps(dict(
                results=[dict(subject_id='s1', status='completed', returncode=0),
                         dict(subject_id='s2', status='failed', returncode=2),
                         dict(subject_id='s3', status='failed', returncode=2)], not_launched=[])))
            record = dict(subjects=[dict(subject=n) for n in names])
            trial = dict(route='hybrid', policy='subject', output=str(destination), status='failed', returncode=2)
            audit = dict(status='failed', summary=dict(passed_subjects=1), subjects=[
                dict(subject_id='s1', status='passed', checks=[]),
                dict(subject_id='s2', status='failed', checks=[dict(name='dti.fa.exists', passed=False)])])
            seen = {}

            def fake_audit(root, **options):
                seen.update(options, root=root)
                return audit
            with patch('scripts.audit_scientific_outputs.audit_output_root', fake_audit), \
                    patch('scripts.benchmark_cohort_compare.scientific_inventory',
                          side_effect=ValueError('no products')):
                outcomes = workflows.audit_trial(record, trial)
            self.assertTrue(seen['require_tractography'])
            self.assertEqual(seen['min_subjects'], 3)
            self.assertEqual(set(outcomes), set(names))
            self.assertEqual(outcomes['s1']['rule1'], 'passed')
            self.assertEqual(outcomes['s2']['rule1'], 'failed')
            self.assertEqual(outcomes['s2']['failed_checks'], ['dti.fa.exists'])
            self.assertEqual(outcomes['s3']['rule1'], 'no_manifest')
            self.assertEqual(outcomes['s3']['cause'], 'subject_exit_2')
            self.assertEqual(trial['rule1_passed_subjects'], 1)
            self.assertTrue((destination / 'subject_outcomes.json').is_file())
            self.assertTrue((destination / 'scientific_fingerprints.json').is_file())

    def test_audit_error_still_writes_an_outcome_per_subject(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            record = dict(subjects=[dict(subject='s1')])
            trial = dict(route='cpu', policy='adaptive', output=str(destination), status='failed')
            with patch('scripts.audit_scientific_outputs.audit_output_root', side_effect=RuntimeError('boom')), \
                    patch('scripts.benchmark_cohort_compare.scientific_inventory', side_effect=ValueError('x')):
                outcomes = workflows.audit_trial(record, trial)
            self.assertEqual(outcomes['s1']['rule1'], 'no_manifest')
            self.assertIn('boom', trial['audit_error'])

    def test_gpu_out_of_memory_is_labelled_with_other_users_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            log = destination / 'err.log'
            log.write_text('torch.OutOfMemoryError: CUDA out of memory')
            record = dict(subjects=[dict(subject='s1')])
            batch = dict(results=[dict(subject_id='s1', status='failed', returncode=1, stderr_log=str(log))])
            outcomes = workflows.subject_outcomes(record, dict(), dict(subjects=[]), batch,
                                                  [dict(other_mib=40000.)])
            self.assertEqual(outcomes['s1']['cause'], 'gpu_out_of_memory')
            self.assertTrue(outcomes['s1']['external_candidate'])


FAKE_DRIVER = textwrap.dedent("""\
    import sys, time
    from pathlib import Path
    args = sys.argv
    campaign = Path(args[args.index('--output') + 1])
    ready, go = Path(args[args.index('--ready-file') + 1]), Path(args[args.index('--go-file') + 1])
    assert '--defer-qc' in args
    ready.touch()
    while not go.exists():
        time.sleep(0.02)
    (campaign / 'started').write_text(str(time.time()))
    time.sleep(0.3)
    (campaign / 'stopped').write_text(str(time.time()))
    """)


class PairTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        for name in ('models', 'mrtrix', 'dataset'):
            (self.base / name).mkdir()
        (self.base / 'launcher.sh').write_text('#!/bin/sh\n')
        self.rows = [dict(subject=f's{i}', source=str(self.base / f'src{i}'), signatures={}, shape=[1, 1, 1, 1])
                     for i in range(10)]
        for row in self.rows:
            Path(row['source']).mkdir()
            for name in workflows.INPUTS:
                (Path(row['source']) / name).write_text('x')
        snapshot = self.base / 'snapshot'
        (snapshot / 'scripts').mkdir(parents=True)
        (snapshot / 'scripts/benchmark_cohort_workflows.py').write_text(FAKE_DRIVER)
        self.snapshot = snapshot

    def args(self, **kw):
        values = dict(output=self.base / 'pair1', workload='cpu', repeat=1, attempts=3, retry_wait_seconds=0,
                      count=10, max_hours=1.0, dataset_root=self.base / 'dataset', models=self.base / 'models',
                      mrtrix=self.base / 'mrtrix', launcher=self.base / 'launcher.sh', previous_pair=None,
                      ukf_executable=None, wma=self.base, atlas=self.base)
        values.update(kw)
        return types.SimpleNamespace(**values)

    def run_pair(self, args, runner, qc_hook=None, busy=None):
        events = []

        def sampler(folder, number):
            events.append(('sample', number))
            return busy or idle_busy(), [], []

        def stage(install):
            events.append(('stage',))
            return self.snapshot

        def fake_qc(arguments):
            events.append(('qc', Path(arguments.output).name))
            qc_hook and qc_hook(arguments)
            return 0

        def fake_compare(subject, adaptive, output):
            events.append(('compare',))
            return dict(flagged=False, throughput_ratio=1.0)

        def wrapped(command, *, timeout, env=None):
            events.append(('doctor', env.get('CUDA_DEVICE_ORDER')))
            return runner(command)
        with patch.object(workflows, 'inspect_cohort', return_value=(self.rows, [])), \
                patch.object(workflows, 'stage_snapshot', stage), \
                patch.object(workflows, 'qc', fake_qc), \
                patch.object(workflows.shutil, 'disk_usage', return_value=types.SimpleNamespace(free=10**13)), \
                patch('scripts.benchmark_cohort_compare.compare_pair', fake_compare):
            code = workflows.pair(args, topology=topology(), sampler=sampler, runner=wrapped,
                                  sleep=lambda s: time.sleep(0.02))
        return code, events

    @staticmethod
    def doctor_ok(blockers=0):
        document = json.dumps(dict(environment=dict(summary=dict(blocker_count=blockers))))
        return lambda command: (0, 'launcher noise\n' + document, '')

    def test_order_doctor_once_then_both_arms_together_then_qc_after_both_stop(self):
        qc_checks = []

        def hook(arguments):
            campaign = Path(arguments.output)
            qc_checks.append(all((self.args().output / p / 'stopped').exists() for p in ('subject', 'adaptive')))
        code, events = self.run_pair(self.args(), self.doctor_ok(), qc_hook=hook)
        self.assertEqual(code, 0)
        names = [e[0] for e in events]
        self.assertEqual(names.count('doctor'), 1)
        self.assertLess(names.index('doctor'), names.index('stage'))
        self.assertLess(names.index('stage'), names.index('qc'))
        self.assertEqual(names[-1], 'compare')
        self.assertEqual(qc_checks, [True, True])           # QC waited for both process trees
        out = self.args().output
        starts = [float((out / p / 'started').read_text()) for p in ('subject', 'adaptive')]
        self.assertLess(abs(starts[0] - starts[1]), 0.5)
        state = json.loads((out / 'pair.json').read_text())
        self.assertEqual(state['assignment']['subject']['node'], 0)
        self.assertEqual(len(state['assignment']['subject']['cpus']), 32)
        # one-trial campaigns from one snapshot
        records = [json.loads((out / p / 'campaign.json').read_text()) for p in ('subject', 'adaptive')]
        self.assertEqual(records[0]['source_hashes'], records[1]['source_hashes'])
        self.assertEqual([len(r['trials']) for r in records], [1, 1])
        self.assertEqual([r['trials'][0]['policy'] for r in records], ['subject', 'adaptive'])

    def test_hf3b_pair_fallback_reaps_with_require_match(self):
        """H-F3b: fails before the fix (the fallback called reap_group without require_match)."""
        driver = FAKE_DRIVER + textwrap.dedent("""\
            import json
            record = json.loads((campaign / 'campaign.json').read_text())
            record['trials'][0].update(status='running', child_pid=4242, child_start_ticks=111)
            (campaign / 'campaign.json').write_text(json.dumps(record))
            """)
        (self.snapshot / 'scripts/benchmark_cohort_workflows.py').write_text(driver)
        reap = MagicMock(return_value=False)
        with patch.object(workflows, 'reap_group', reap):
            code, _ = self.run_pair(self.args(), self.doctor_ok())
        self.assertEqual(code, 0)
        self.assertEqual(reap.call_count, 2)
        for call in reap.call_args_list:
            self.assertEqual(call.args, (4242, 111))
            self.assertIs(call.kwargs.get('require_match'), True)

    def test_doctor_blocker_counts_as_failed_attempt_and_stops_launch(self):
        code, events = self.run_pair(self.args(attempts=2), self.doctor_ok(blockers=1))
        self.assertEqual(code, 2)
        names = [e[0] for e in events]
        self.assertEqual(names.count('doctor'), 2)
        self.assertNotIn('stage', names)
        self.assertFalse((self.args().output / 'subject').exists())
        state = json.loads((self.args().output / 'pair.json').read_text())
        self.assertEqual(state['status'], 'launch_refused')
        self.assertEqual(len(state['attempts']), 2)

    def test_busy_host_skips_doctor_and_retries(self):
        code, events = self.run_pair(self.args(attempts=3), self.doctor_ok(), busy=idle_busy(default=60.0))
        self.assertEqual(code, 2)
        names = [e[0] for e in events]
        self.assertEqual(names.count('sample'), 3)
        self.assertNotIn('doctor', names)

    def test_arguments_are_validated_before_anything_is_created(self):
        for bad in (dict(workload='dki'), dict(repeat=3), dict(attempts=0), dict(count=3),
                    dict(output=self.base / 'a' / 'b'), dict(launcher=self.base / 'missing.sh')):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                workflows.validate_pair(self.args(**bad))
        self.assertEqual(sorted(p.name for p in self.base.iterdir() if p.name.startswith('pair')), [])
        existing = self.base / 'pair9'
        existing.mkdir()
        with self.assertRaises(ValueError):
            workflows.validate_pair(self.args(output=existing))

    def test_hybrid_pair_checks_model_directories_before_launch(self):
        with self.assertRaises(ValueError):
            workflows.validate_pair(self.args(workload='hybrid'))
        for name in workflows.MODEL_DIRECTORIES:
            (self.base / 'models' / name).mkdir()
        workflows.validate_pair(self.args(workload='hybrid'))

    def test_pair_refuses_an_existing_folder_without_touching_it(self):
        existing = self.base / 'pair1'
        existing.mkdir()
        (existing / 'keep').write_text('x')
        with self.assertRaises(ValueError):
            self.run_pair(self.args(), self.doctor_ok())
        self.assertEqual((existing / 'keep').read_text(), 'x')

    def test_run_barrier_waits_for_go_and_honours_cancel(self):
        import threading
        with tempfile.TemporaryDirectory() as directory:
            ready, go = Path(directory) / 'ready', Path(directory) / 'go'
            cancelled = threading.Event()
            threading.Timer(0.2, cancelled.set).start()
            self.assertFalse(workflows.wait_for_go(ready, go, cancelled, 5))
            self.assertTrue(ready.exists())
            ready2, go2 = Path(directory) / 'r2', Path(directory) / 'g2'
            go2.touch()
            self.assertTrue(workflows.wait_for_go(ready2, go2, threading.Event(), 5))

    def test_leftover_process_group_is_killed(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
        try:
            self.assertTrue(workflows.reap_group(child.pid))
            child.wait(timeout=10)
            self.assertEqual(child.returncode, -9)
        finally:
            if child.poll() is None:
                child.kill()
        self.assertFalse(workflows.reap_group(child.pid))

    def test_hf3_reap_skips_a_process_with_another_start_time(self):
        with patch.object(workflows, 'process_start_ticks', return_value=222), patch('os.killpg') as killpg:
            self.assertFalse(workflows.reap_group(4242, start_ticks=111))
        killpg.assert_not_called()

    def test_hf3_reap_signals_the_group_when_the_start_time_matches(self):
        with patch.object(workflows, 'process_start_ticks', return_value=111), patch('os.killpg') as killpg:
            self.assertTrue(workflows.reap_group(4242, start_ticks=111))
        self.assertEqual(killpg.call_count, 2)           # existence probe, then SIGKILL

    def test_hf3_reap_without_proc_keeps_the_old_behaviour(self):
        with patch.object(workflows, 'process_start_ticks', return_value=None), patch('os.killpg') as killpg:
            self.assertTrue(workflows.reap_group(4242, start_ticks=111))
        self.assertEqual(killpg.call_count, 2)

    def test_hf3b_require_match_skips_when_the_start_time_is_unreadable(self):
        with patch.object(workflows, 'process_start_ticks', return_value=None), patch('os.killpg') as killpg:
            self.assertFalse(workflows.reap_group(4242, start_ticks=111, require_match=True))
        killpg.assert_not_called()

    def test_hf3b_require_match_skips_when_the_start_time_differs(self):
        with patch.object(workflows, 'process_start_ticks', return_value=222), patch('os.killpg') as killpg:
            self.assertFalse(workflows.reap_group(4242, start_ticks=111, require_match=True))
        killpg.assert_not_called()

    def test_hf3b_require_match_kills_when_the_start_time_matches(self):
        with patch.object(workflows, 'process_start_ticks', return_value=111), patch('os.killpg') as killpg:
            self.assertTrue(workflows.reap_group(4242, start_ticks=111, require_match=True))
        self.assertEqual(killpg.call_count, 2)

    def test_hf3_start_ticks_is_field_22_of_proc_stat(self):
        fields = ['S'] + [str(n) for n in range(4, 22)] + ['987654'] + ['0'] * 5      # fields 3..22 then more
        stat = '4242 (my (odd) name) ' + ' '.join(fields)
        with patch.object(Path, 'read_text', return_value=stat):
            self.assertEqual(workflows.process_start_ticks(4242), 987654)
        with patch.object(Path, 'read_text', side_effect=FileNotFoundError):
            self.assertIsNone(workflows.process_start_ticks(4242))

    def test_doctor_output_parsing(self):
        text = 'noise {not json\n' + json.dumps(dict(environment=dict(summary=dict(blocker_count=2))))
        self.assertEqual(harness.doctor_blockers(harness.parse_doctor_output(text)), 2)
        with self.assertRaises(ValueError):
            harness.parse_doctor_output('nothing here')


def nifti(path, array, affine=None):
    import nibabel as nib
    nib.save(nib.Nifti1Image(np.asarray(array), np.eye(4) if affine is None else affine), str(path))
    return path


class ClassRuleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def load(self, array, affine=None):
        import nibabel as nib
        return nib.Nifti1Image(np.asarray(array), np.eye(4) if affine is None else affine)

    def test_class_t_tolerance_is_1e4_times_p99_inside_the_mask(self):
        first = np.linspace(1, 100, 1000).reshape(10, 10, 10)
        mask = np.ones((10, 10, 10), dtype=bool)
        p99 = np.percentile(first, 99)
        near = first.copy()
        near[0, 0, 0] += 0.9e-4 * p99
        self.assertEqual(compare.compare_float(self.load(first), self.load(near), mask)['status'], 'passed')
        far = first.copy()
        far[0, 0, 0] += 1.1e-4 * p99
        self.assertEqual(compare.compare_float(self.load(first), self.load(far), mask)['status'], 'mismatch')

    def test_class_t_difference_outside_the_domain_mask_is_ignored(self):
        first = np.ones((4, 4, 4))
        second = first.copy()
        second[3, 3, 3] = 50.
        mask = np.ones((4, 4, 4), dtype=bool)
        mask[3, 3, 3] = False
        self.assertEqual(compare.compare_float(self.load(first), self.load(second), mask)['status'], 'passed')

    def test_class_t_nonfinite_positions_must_match_over_the_whole_array(self):
        first = np.ones((4, 4, 4))
        second = first.copy()
        first[0, 0, 0] = np.nan
        second[0, 0, 0] = np.nan
        mask = np.ones((4, 4, 4), dtype=bool)
        self.assertEqual(compare.compare_float(self.load(first), self.load(second), mask)['status'], 'passed')
        second[1, 1, 1] = np.nan        # a different position, outside the mask
        mask[1, 1, 1] = False
        result = compare.compare_float(self.load(first), self.load(second), mask)
        self.assertEqual(result['status'], 'mismatch')
        self.assertIn('non-finite', result['reason'])

    def test_class_t_p99_zero_needs_equal_values(self):
        mask = np.ones((3, 3, 3), dtype=bool)
        zeros = np.zeros((3, 3, 3))
        tiny = zeros.copy()
        tiny[0, 0, 0] = 1e-12
        self.assertEqual(compare.compare_float(self.load(zeros), self.load(zeros.copy()), mask)['status'], 'passed')
        result = compare.compare_float(self.load(zeros), self.load(tiny), mask)
        self.assertEqual(result['status'], 'mismatch')
        self.assertEqual(result['p99'], 0.0)

    def test_class_t_no_finite_voxel_in_domain_is_not_compared(self):
        first = np.full((3, 3, 3), np.nan)
        mask = np.ones((3, 3, 3), dtype=bool)
        self.assertEqual(compare.compare_float(self.load(first), self.load(first.copy()), mask)['status'],
                         'not_compared')
        self.assertEqual(compare.compare_float(self.load(np.ones((3, 3, 3))), self.load(np.ones((3, 3, 3))),
                                               np.zeros((3, 3, 3), dtype=bool))['status'], 'not_compared')

    def test_class_t_four_dimensional_output_uses_its_spatial_mask(self):
        first = np.ones((4, 4, 4, 5))
        second = first.copy()
        second[0, 0, 0, 4] += 10.             # inside the mask, in the last channel
        mask = np.ones((4, 4, 4), dtype=bool)
        self.assertEqual(compare.compare_float(self.load(first), self.load(second), mask)['status'], 'mismatch')
        mask[0, 0, 0] = False                 # the same voxel in all five channels leaves the domain
        self.assertEqual(compare.compare_float(self.load(first), self.load(second), mask)['status'], 'passed')

    def test_class_l_label_fraction_limit(self):
        mask = np.ones((20, 20, 30), dtype=bool)          # 12000 voxels: 1 voxel = 8.3e-5
        first = np.zeros((20, 20, 30), dtype=np.int16)
        one = first.copy()
        one[0, 0, 0] = 1
        two = one.copy()
        two[1, 0, 0] = 1
        self.assertEqual(compare.compare_label(self.load(first), self.load(one), mask)['status'], 'passed')
        self.assertEqual(compare.compare_label(self.load(first), self.load(two), mask)['status'], 'mismatch')

    def test_domain_mask_grid_match_within_1e4_mm_and_fallbacks(self):
        affine = np.diag([2., 2., 2., 1.])
        native = nifti(self.base / 'native.nii.gz', np.ones((4, 4, 4), dtype=np.uint8), affine)
        registered = nifti(self.base / 'reg.nii.gz', np.ones((5, 5, 5), dtype=np.uint8), np.eye(4))
        moved = affine.copy()
        moved[0, 3] = 0.9e-4
        image = self.load(np.zeros((4, 4, 4)), moved)
        mask, source = compare.domain_mask(image, native, [registered])
        self.assertEqual(source, str(native))
        moved[0, 3] = 1.1e-4
        image = self.load(np.zeros((4, 4, 4)), moved)
        mask, source = compare.domain_mask(image, native, [registered])
        self.assertEqual(source, 'all voxels')
        self.assertTrue(mask.all())
        # 4-D output on the registered grid uses its spatial part
        image = self.load(np.zeros((5, 5, 5, 5)), np.eye(4))
        self.assertEqual(compare.domain_mask(image, native, [registered])[1], str(registered))

    def test_class_e_exact_nii_and_nrrd_hash(self):
        import nrrd
        first, second = self.base / 'a.nii.gz', self.base / 'b.nii.gz'
        nifti(first, np.arange(8, dtype=np.float32).reshape(2, 2, 2))
        nifti(second, np.arange(8, dtype=np.float32).reshape(2, 2, 2))
        self.assertEqual(compare.compare_entry('E', first, second, None)['status'], 'passed')
        nifti(second, np.arange(8, dtype=np.float32).reshape(2, 2, 2) + 1e-6)
        self.assertEqual(compare.compare_entry('E', first, second, None)['status'], 'mismatch')
        data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        nrrd.write(str(self.base / 'a.nrrd'), data)
        nrrd.write(str(self.base / 'b.nrrd'), data.copy())
        self.assertEqual(compare.compare_entry('E', self.base / 'a.nrrd', self.base / 'b.nrrd', None)['status'],
                         'passed')
        nrrd.write(str(self.base / 'c.nrrd'), data + 1)
        self.assertEqual(compare.compare_entry('E', self.base / 'a.nrrd', self.base / 'c.nrrd', None)['status'],
                         'mismatch')

    def test_geometry_difference_is_a_mismatch(self):
        first, second = self.base / 'a.nii.gz', self.base / 'b.nii.gz'
        nifti(first, np.ones((3, 3, 3)))
        nifti(second, np.ones((3, 3, 3)), np.diag([1., 1., 1.001, 1.]))
        mask = nifti(self.base / 'm.nii.gz', np.ones((3, 3, 3), dtype=np.uint8))
        self.assertEqual(compare.compare_entry('T', first, second, (mask, []))['status'], 'mismatch')

    def write_surface(self, path, shift=0.0, faces=None):
        import trimesh
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        vertices[0, 0] += shift
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces if faces is not None else [[0, 1, 2], [0, 1, 3], [1, 2, 3]],
                               process=False)
        mesh.export(str(path))

    def test_class_s_surface_displacement_limit_and_topology(self):
        self.write_surface(self.base / 'a.stl')
        self.write_surface(self.base / 'near.stl', shift=0.009)
        self.write_surface(self.base / 'far.stl', shift=0.011)
        self.write_surface(self.base / 'faces.stl', faces=[[0, 1, 2], [0, 1, 3]])
        self.assertEqual(compare.compare_surface(self.base / 'a.stl', self.base / 'near.stl')['status'], 'passed')
        self.assertEqual(compare.compare_surface(self.base / 'a.stl', self.base / 'far.stl')['status'], 'mismatch')
        result = compare.compare_surface(self.base / 'a.stl', self.base / 'faces.stl')
        self.assertEqual(result['status'], 'mismatch')
        self.assertIn('count', result['reason'])

    def edge_list(self, path, weights):
        lines = ['source_node,target_node,metric'] + [f'{i},{j},{w}' for (i, j), w in weights.items()]
        path.write_text('\n'.join(lines) + '\n')
        return path

    def test_rule3_spearman_and_fewer_than_10_edges_is_undefined(self):
        few = {(i, i + 1): float(i + 1) for i in range(9)}
        a, b = self.edge_list(self.base / 'a.csv', few), self.edge_list(self.base / 'b.csv', few)
        self.assertEqual(compare.rule3(a, b)['status'], 'undefined')
        many = {(i, i + 1): float(i + 1) for i in range(12)}
        reverse = {key: 100. - value for key, value in many.items()}
        a, b = self.edge_list(self.base / 'c.csv', many), self.edge_list(self.base / 'd.csv', reverse)
        result = compare.rule3(a, b)
        self.assertEqual(result['status'], 'computed')
        self.assertAlmostEqual(result['spearman'], -1.0)
        same = compare.rule3(a, self.edge_list(self.base / 'e.csv', many))
        self.assertAlmostEqual(same['spearman'], 1.0)
        # an edge that is zero in both arms is not counted
        padded = dict(many)
        padded[(30, 31)] = 0.0
        result = compare.rule3(self.edge_list(self.base / 'f.csv', padded), a)
        self.assertEqual(result['edges'], 12)

    def test_rule3_dense_matrix(self):
        matrix = np.arange(25, dtype=float).reshape(5, 5)
        np.savetxt(self.base / 'd.csv', matrix, delimiter=',')
        np.savetxt(self.base / 'e.csv', matrix * 2, delimiter=',')
        result = compare.rule3(self.base / 'd.csv', self.base / 'e.csv')
        self.assertEqual(result, dict(status='computed', spearman=result['spearman'], edges=10))
        self.assertAlmostEqual(result['spearman'], 1.0)


def build_subject(root, subject, workload='cpu', include=None, perturb=None):
    """One arm's synthetic tree for compare_subject: outputs/<id>/..., inputs/<id>/..."""
    import nrrd
    perturb = perturb or {}
    output = root / 'outputs' / subject
    source = root / 'inputs' / subject
    (output / '00_run_metadata').mkdir(parents=True)
    (source / 'input').mkdir(parents=True)
    nifti(source / 'input/nodif_brain_mask.nii.gz', np.ones((4, 4, 4), dtype=np.uint8))
    native = source / 'cache/native_dti/LS'
    native.mkdir(parents=True)
    for name in ('DTI.nrrd', 'b0.nrrd'):
        nrrd.write(str(native / name), np.ones((4, 4, 4), dtype=np.float32))
    tasks = []
    wanted = compare.expected_entries(workload) if include is None else include
    for task, table in compare.TASK_CLASSES.items():
        if task not in compare.EXPECTED_TASKS[workload]:
            continue
        outputs = {}
        for key in table:
            if f'{task}:{key}' not in wanted or key.endswith('_raw'):
                continue
            path = output / f'{task}_{key}.nii.gz'
            if task == 'surface.ddsurfer':
                path = output / f'{task}_{key}.stl'
                import trimesh
                trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]], process=False
                                ).export(str(path))
            else:
                value = np.full((4, 4, 4), 2.0 if table[key] == 'T' else 1, dtype=np.float32 if table[key] == 'T'
                                else np.int16)
                value = value + perturb.get(f'{task}:{key}', 0)
                nifti(path, value)
            outputs[key] = str(path)
        tasks.append(dict(name=task, status='completed', outputs=outputs))
    outputs = {}
    if workload == 'hybrid':
        edge = output / 'sift2.csv'
        edge.write_text('source_node,target_node,metric\n' + ''.join(f'{i},{i + 1},{i + 1}\n' for i in range(12)))
        tasks.append(dict(name='connectome.sift2_weighted', status='completed', outputs=dict(matrix_path=str(edge))))
        five = output / 'five_tt.nii.gz'
        nifti(five, np.ones((4, 4, 4, 5), dtype=np.float32))
        (output / '04_tractography').mkdir()
        (output / '04_tractography/csd_act_manifest.json').write_text(
            json.dumps(dict(dependencies=dict(five_tt=dict(path=str(five))))))
    (output / '00_run_metadata/benchmark_orchestration_manifest.json').write_text(json.dumps(dict(tasks=tasks)))


class SubjectCompareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)

    def trees(self, workload='cpu', first=None, second=None):
        for arm, kw in (('subject', first or {}), ('adaptive', second or {})):
            build_subject(self.base / arm, 's1', workload, **kw)
        return self.base / 'subject', self.base / 'adaptive'

    def test_identical_arms_pass_every_class(self):
        a, b = self.trees()
        result = compare.compare_subject('s1', a, b, 'cpu')
        self.assertEqual(result['rule2']['status'], 'passed', result['rule2']['mismatches'])
        classes = {row['class'] for row in result['rule2']['entries'].values()}
        self.assertEqual(classes, {'E', 'T'})
        self.assertEqual(result['rule3']['status'], 'undefined')

    def test_hybrid_covers_labels_surfaces_5tt_and_rule3(self):
        a, b = self.trees('hybrid')
        result = compare.compare_subject('s1', a, b, 'hybrid')
        self.assertEqual(result['rule2']['status'], 'passed', result['rule2']['mismatches'])
        classes = {row['class'] for row in result['rule2']['entries'].values()}
        self.assertEqual(classes, {'E', 'L', 'T', 'S'})
        self.assertIn('csd:five_tt', result['rule2']['entries'])
        self.assertEqual(result['rule3']['status'], 'computed')

    def test_expected_entry_missing_in_one_arm_is_a_mismatch_not_skipped(self):
        full = compare.expected_entries('cpu')
        partial = full - {'microstructure.noddi:ndi'}
        a, b = self.trees(second=dict(include=partial))
        result = compare.compare_subject('s1', a, b, 'cpu')
        self.assertEqual(result['rule2']['status'], 'mismatch')
        self.assertIn('microstructure.noddi:ndi', result['rule2']['mismatches'])
        self.assertEqual(result['rule2']['entries']['microstructure.noddi:ndi']['status'], 'missing')

    def test_missing_native_bundle_in_one_arm_is_a_mismatch(self):
        a, b = self.trees()
        import shutil
        shutil.rmtree(b / 'inputs/s1/cache/native_dti/LS')
        result = compare.compare_subject('s1', a, b, 'cpu')
        self.assertIn('cache/native_dti/LS/DTI.nrrd', result['rule2']['mismatches'])
        self.assertEqual(result['rule2']['entries']['cache/native_dti/LS/b0.nrrd']['status'], 'missing')

    def test_float_mismatch_is_reported_per_entry(self):
        a, b = self.trees(second=dict(perturb={'microstructure.dti:fa': 1.0}))
        result = compare.compare_subject('s1', a, b, 'cpu')
        self.assertEqual(result['rule2']['mismatches'], ['microstructure.dti:fa'])

    def test_pair_report_checks_source_hashes_and_flags_mismatch(self):
        for arm, policy in (('subject', 'subject'), ('adaptive', 'adaptive')):
            root = self.base / arm
            build_subject(root, 's1', 'cpu', perturb={'microstructure.dti:fa': 1.0} if arm == 'adaptive' else None)
            outcomes = dict(s1=dict(subject='s1', rule1='passed'))
            (root / 'subject_outcomes.json').write_text(json.dumps(outcomes))
            trial = dict(id=f'cpu-{policy}', route='cpu', policy=policy, status='completed', output=str(root),
                         measurement=dict(execution_seconds=3600.0), returncode=0)
            record = dict(subjects=[dict(subject='s1')], trials=[trial], source_hashes=dict(a='1'), cpu_affinity=[0, 1])
            (root / 'campaign.json').write_text(json.dumps(record))
        report = compare.compare_pair(self.base / 'subject', self.base / 'adaptive', self.base / 'pair_report.json')
        self.assertTrue(report['source_hashes_equal'])
        self.assertTrue(report['flagged'])
        self.assertEqual(report['rule2_mismatch_subjects'], ['s1'])
        self.assertEqual(report['throughput_ratio'], 1.0)
        self.assertTrue((self.base / 'pair_report.json').is_file())
        # a second run never overwrites the temporary file silently
        record = json.loads((self.base / 'adaptive/campaign.json').read_text())
        record['source_hashes'] = dict(a='2')
        (self.base / 'adaptive/campaign.json').write_text(json.dumps(record))
        report = compare.compare_pair(self.base / 'subject', self.base / 'adaptive', self.base / 'pair_report2.json')
        self.assertFalse(report['source_hashes_equal'])
        self.assertTrue(report['flagged'])
        self.assertEqual(report['subjects'], {})

    def test_rule1_failure_skips_rule2_for_that_subject(self):
        for arm, policy in (('subject', 'subject'), ('adaptive', 'adaptive')):
            root = self.base / arm
            build_subject(root, 's1', 'cpu')
            rule1 = 'passed' if arm == 'subject' else 'failed'
            (root / 'subject_outcomes.json').write_text(json.dumps(dict(s1=dict(subject='s1', rule1=rule1))))
            trial = dict(id=f'cpu-{policy}', route='cpu', policy=policy, status='completed', output=str(root),
                         measurement=dict(execution_seconds=100.0), returncode=0)
            (root / 'campaign.json').write_text(json.dumps(dict(
                subjects=[dict(subject='s1')], trials=[trial], source_hashes={}, cpu_affinity=[0])))
        report = compare.compare_pair(self.base / 'subject', self.base / 'adaptive', self.base / 'r.json')
        self.assertEqual(report['subjects']['s1']['rule2']['status'], 'skipped_rule1')
        self.assertFalse(report['flagged'])
        self.assertIsNone(report['throughput_ratio'])        # the Adaptive arm passed no subject


    def test_hf1_existing_report_is_never_replaced(self):
        """H-F1: fails before the fix (the old code replaced the file)."""
        for arm in ('subject', 'adaptive'):
            root = self.base / arm
            build_subject(root, 's1', 'cpu')
            (root / 'subject_outcomes.json').write_text(json.dumps(dict(s1=dict(subject='s1', rule1='passed'))))
            trial = dict(id=f'cpu-{arm}', route='cpu', policy=arm, status='completed', output=str(root),
                         measurement=dict(execution_seconds=100.0), returncode=0)
            (root / 'campaign.json').write_text(json.dumps(dict(
                subjects=[dict(subject='s1')], trials=[trial], source_hashes={}, cpu_affinity=[0])))
        report = self.base / 'pair_report.json'
        report.write_text('{"evidence": "keep"}\n')
        before = report.read_bytes()
        with self.assertRaises(FileExistsError):
            compare.compare_pair(self.base / 'subject', self.base / 'adaptive', report)
        self.assertEqual(report.read_bytes(), before)
        self.assertFalse((self.base / 'pair_report.json.tmp').exists())


FAKE_LAUNCHER = textwrap.dedent("""\
    #!/bin/sh
    echo "$@" > "$FAKE_ARGV"
    echo "$CUDA_VISIBLE_DEVICES|$OMP_NUM_THREADS" >> "$FAKE_ARGV"
    while [ $# -gt 0 ]; do if [ "$1" = "--manifest" ]; then m="$2"; fi; shift; done
    echo '{"results": [], "not_launched": []}' > "$m"
    exit 2
    """)


class RunTests(unittest.TestCase):
    """`run` with a fake launcher: QC is deferred, a return code 2 is still audited later, the group is reaped."""

    def tmpdir(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return directory.name

    def make_campaign(self, base, policy):
        launcher = base / 'launcher.sh'
        launcher.write_text(FAKE_LAUNCHER)
        launcher.chmod(0o755)
        snapshot = base / 'snapshot'
        (snapshot / 'scripts').mkdir(parents=True)
        (snapshot / 'scripts/benchmark_cohort_workflows.py').write_text('x')
        source = base / 'src'
        source.mkdir()
        for name in workflows.INPUTS:
            (source / name).write_text('x')
        args = types.SimpleNamespace(models=Path('/m'), mrtrix=Path('/x'), wma=Path('/w'), atlas=Path('/a'),
                                     launcher=launcher, ukf_executable=None, cpus=32, memory_gb=224.0, jobs=4)
        rows = [dict(subject='s1', source=str(source), signatures={n: workflows.signature(source / n)
                                                                   for n in workflows.INPUTS},
                     sha256={n: workflows.sha(source / n) for n in workflows.INPUTS})]
        record = workflows.build_record(
            args, rows=rows, skipped=[], snapshot=snapshot, routes=['hybrid'],
            trials=workflows.make_trials(['hybrid'], [8], 1, [policy]), common=workflows.common_arguments(args),
            options={'hybrid': workflows.route_options(args, ['hybrid'])['hybrid']}, assets={},
            cpu_list=list(range(32)), gpu=2, kill_memory_gb=256.0, cpus=32, memory_gb=224.0, jobs=4, repeats=1,
            extra=dict(paired=True))
        folder = base / policy
        folder.mkdir()
        workflows.save(folder / 'campaign.json', record)
        return folder

    def run_arm(self, policy):
        base = Path(self.tmpdir())
        folder = self.make_campaign(base, policy)
        env = {'FAKE_ARGV': str(base / 'argv.txt')}
        guard = base / 'guard'
        guard.mkdir()
        args = types.SimpleNamespace(output=folder, max_hours=1.0, defer_qc=True, ready_file=base / 'ready',
                                     go_file=base / 'go', go_timeout=5)
        (base / 'go').touch()
        with patch.dict(os.environ, env), \
                patch('shared.cohort_affinity.cpu_allocation', return_value=list(range(32))), \
                patch('shared.cohort_affinity.isolated_command', side_effect=lambda command, allocation: command), \
                patch('shared.cohort_affinity.affinity_violations', return_value=[]), \
                patch('shared.cohort_schedule.create_schedule', return_value=guard):
            code = workflows.run(args)
        return code, folder, base

    def test_subject_arm_defers_qc_keeps_the_failure_and_qc_audits_afterwards(self):
        code, folder, base = self.run_arm('subject')
        self.assertEqual(code, 2)
        record = json.loads((folder / 'campaign.json').read_text())
        trial = record['trials'][0]
        self.assertEqual((record['status'], trial['status'], trial['returncode']), ('failed', 'failed', 2))
        self.assertIn('child_pid', trial)
        self.assertNotIn('qc_status', trial)                      # QC was deferred
        self.assertTrue((base / 'ready').exists())
        argv = (base / 'argv.txt').read_text().splitlines()
        self.assertIn('--jobs 4', argv[0])
        self.assertEqual(argv[1], '2|8')                          # GPU 2, thread variables 8
        with patch('scripts.audit_scientific_outputs.audit_output_root', side_effect=RuntimeError('x')), \
                patch('scripts.benchmark_cohort_compare.scientific_inventory', side_effect=ValueError('x')):
            workflows.qc(types.SimpleNamespace(output=folder))
        record = json.loads((folder / 'campaign.json').read_text())
        self.assertTrue(record['qc_done'])
        self.assertTrue(Path(record['trials'][0]['output'], 'subject_outcomes.json').is_file())

    def test_adaptive_arm_gets_no_thread_variables_and_no_jobs(self):
        code, folder, base = self.run_arm('adaptive')
        argv = (base / 'argv.txt').read_text().splitlines()
        self.assertNotIn('--jobs', argv[0])
        self.assertEqual(argv[1], '2|')                           # GPU 2, thread variables unset

    def test_qc_refuses_a_campaign_that_has_not_finished(self):
        base = Path(self.tmpdir())
        folder = self.make_campaign(base, 'subject')
        with self.assertRaises(ValueError):
            workflows.qc(types.SimpleNamespace(output=folder))


    def test_hf2_qc_skips_an_inherited_trial_and_writes_nothing_outside_the_campaign(self):
        """H-F2: fails before the fix (qc audited the earlier campaign and wrote into its folder)."""
        base = Path(self.tmpdir())
        folder = self.make_campaign(base, 'subject')
        earlier = base / 'earlier'
        (earlier / 'outputs/s1').mkdir(parents=True)
        (earlier / 'independent_qc.json').write_text('{}')
        (earlier / 'scientific_fingerprints.json').write_text('{}')

        def listing():
            return {str(p.relative_to(earlier)): hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
                    for p in sorted(earlier.rglob('*'))}
        before = listing()
        record = json.loads((folder / 'campaign.json').read_text())
        record.update(status='completed_qc_deferred')
        record['trials'][0].update(status='completed', output=str(earlier), inherited_evidence=dict(a='1'))
        workflows.save(folder / 'campaign.json', record)
        workflows.qc(types.SimpleNamespace(output=folder))
        self.assertEqual(listing(), before)
        record = json.loads((folder / 'campaign.json').read_text())
        self.assertEqual(record['trials'][0]['qc_skipped'], 'inherited')
        self.assertTrue(record['qc_done'])

    def test_hf2_every_audit_write_stays_under_the_trial_output(self):
        base = Path(self.tmpdir())
        output = base / 'trial'
        (output / 'outputs/s1').mkdir(parents=True)
        written = []
        real = workflows.save

        def spy(path, value):
            written.append(Path(path))
            return real(path, value)
        with patch.object(workflows, 'save', side_effect=spy), \
                patch('scripts.audit_scientific_outputs.audit_output_root', side_effect=RuntimeError('x')), \
                patch('scripts.benchmark_cohort_compare.scientific_inventory', side_effect=ValueError('x')):
            workflows.audit_trial(dict(subjects=[dict(subject='s1')]),
                                  dict(output=str(output), route='hybrid', policy='subject'))
        self.assertTrue(written)
        self.assertTrue(all(path.resolve().is_relative_to(output.resolve()) for path in written))

    def test_hf4_post_launch_save_failure_stops_the_child_group(self):
        """H-F4: fails before the fix (the cleanup never ran when the post-launch save raised)."""
        real = workflows.save

        def failing_save(path, value):
            if isinstance(value, dict) and any('child_pid' in row for row in value.get('trials', [])):
                raise OSError('disk full')
            return real(path, value)
        reap = MagicMock(return_value=False)
        with patch.object(workflows, 'save', side_effect=failing_save), patch.object(workflows, 'reap_group', reap):
            with self.assertRaises(OSError):
                self.run_arm('subject')
        reap.assert_called_once()
        self.assertIsInstance(reap.call_args[0][0], int)

    def test_hf3_run_stores_the_child_start_ticks(self):
        with patch.object(workflows, 'process_start_ticks', return_value=12345):
            code, folder, base = self.run_arm('subject')
        trial = json.loads((folder / 'campaign.json').read_text())['trials'][0]
        self.assertEqual(trial['child_start_ticks'], 12345)


class MetricTests(unittest.TestCase):
    def test_throughput_counts_only_rule1_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            trial = dict(policy='subject', output=directory, measurement=dict(execution_seconds=7200.0, mean_cpu_cores=20.),
                         returncode=2)
            metrics = harness.trial_metrics(trial, dict(cpu_affinity=[0, 1]), 5)
            self.assertAlmostEqual(metrics['throughput_subjects_per_hour'], 2.5)
            self.assertIsNone(metrics['adaptive_only'])

    def test_sampler_stops_in_finally_and_writes_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 's.jsonl'
            counter = iter(range(1000))
            sampler = harness.Sampler(path, 0.01, lambda: dict(n=next(counter), bad=float('nan'))).start()
            time.sleep(0.1)
            summary = sampler.finish()
            self.assertGreater(summary['samples'], 2)
            lines = path.read_text().splitlines()
            self.assertEqual(json.loads(lines[0])['bad'], None)
            self.assertFalse(sampler.thread.is_alive())


if __name__ == '__main__':
    unittest.main()
