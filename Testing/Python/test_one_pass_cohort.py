"""No-resume cohort acceptance must not accept cached, missing or blocked tasks."""

import unittest
from pathlib import Path
import signal
import subprocess
import tempfile
from unittest import mock
from scripts.validate_pipeline_cohort import evaluate_attempt, validate_protocol, run_isolated


class CohortContractTests(unittest.TestCase):
    def test_timeout_reaps_owned_group_even_when_launcher_already_exited(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / 'log'
            process = mock.Mock(pid=43210)
            process.wait.side_effect = [subprocess.TimeoutExpired(['fixture'], 10), 0, 0]
            with mock.patch('scripts.validate_pipeline_cohort.os.name', 'posix'), mock.patch(
                    'scripts.validate_pipeline_cohort.subprocess.Popen', return_value=process) as popen, mock.patch(
                    'scripts.validate_pipeline_cohort.os.killpg', create=True) as kill:
                with self.assertRaises(subprocess.TimeoutExpired):
                    run_isolated(['fixture'], environment={}, cwd=tmp, log_path=log_path, timeout=10)
            self.assertTrue(popen.call_args.kwargs['start_new_session'])
            sigkill = getattr(signal, 'SIGKILL', signal.SIGTERM)
            self.assertEqual(kill.call_args_list, [mock.call(43210, signal.SIGTERM), mock.call(43210, sigkill)])

    def test_a_retry_cannot_hide_a_first_attempt_failure(self):
        result = evaluate_attempt({'status': 'completed', 'tasks': [
            {'name': 'bundles.wmql', 'status': 'failed'},
            {'name': 'bundles.wmql', 'status': 'completed'},
        ]}, ['bundles.wmql'])
        self.assertFalse(result['passed'])
        self.assertEqual(result['duplicate_tasks'], ['bundles.wmql'])

    def test_acceptance_requires_selected_tasks_to_have_executed(self):
        for status in ('skipped', 'failed', 'blocked', 'planned'):
            result = evaluate_attempt({'status': 'completed', 'tasks': [{'name': 'bundles.wmql', 'status': status}]}, ['bundles.wmql'])
            self.assertFalse(result['passed'])
        self.assertFalse(evaluate_attempt({'status': 'completed', 'tasks': []}, ['bundles.wmql'])['passed'])
        self.assertTrue(evaluate_attempt({'status': 'completed', 'tasks': [{'name': 'bundles.wmql', 'status': 'completed'}]}, ['bundles.wmql'])['passed'])

    def test_no_subject_specific_or_resume_overrides(self):
        protocol = {'subjects': [{'subject_id': '100307'}, {'subject_id': '149741'}],
                    'common_cli_args': [], 'expected_tasks': ['microstructure.dti']}
        self.assertEqual(len(validate_protocol(protocol)), 2)
        for argument in ('--no-resume', '--subject-id=100307', '--ignore-environment-blockers', '--skip-environment-check'):
            with self.assertRaises(ValueError):
                validate_protocol(dict(protocol, common_cli_args=[argument]))
        with self.assertRaises(ValueError):
            validate_protocol(dict(protocol, subjects=[{'subject_id': 'same'}, {'subject_id': 'same'}]))


if __name__ == '__main__':
    unittest.main()
