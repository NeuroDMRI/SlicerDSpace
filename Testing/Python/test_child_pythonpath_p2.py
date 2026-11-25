"""Opus P2 (child PYTHONPATH): test child processes keep the inherited PYTHONPATH behind ROOT.

The repo-only CI job puts its test-dependency directory first on PYTHONPATH. A child process that
replaces PYTHONPATH with ROOT loses that directory and fails on import psutil or nibabel.
"""
import os
import subprocess
import unittest
from unittest.mock import patch

import test_cohort_schedule
import test_ddsurfer_sync

SENTINEL_DIR = os.path.join(os.sep, 'sentinel', 'test-site')


class ChildStarted(Exception):
    """Raised by the fake launcher so the real child work never runs."""


def _capture(seen):
    def fake(*args, **kwargs):
        env = kwargs.get('env')
        seen.append(dict(os.environ if env is None else env))
        raise ChildStarted
    return fake


class ChildPythonpathInheritsTests(unittest.TestCase):
    expected = None

    def setUp(self):
        patcher = patch.dict(os.environ, {'PYTHONPATH': SENTINEL_DIR})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.expected = os.pathsep.join([str(test_cohort_schedule.ROOT), SENTINEL_DIR])

    def run_site(self, name, **targets):
        seen = []
        case = test_cohort_schedule.CohortScheduleTests
        with patch.multiple(subprocess, **{key: _capture(seen) for key in targets}):
            with self.assertRaises(ChildStarted):
                getattr(case(name), name)()
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].get('PYTHONPATH'), self.expected)

    def test_cohort_popen_site(self):
        self.run_site('test_gpu_serializes_but_cpu_fits_overlap_across_subjects', Popen=True)

    def test_cohort_run_site(self):
        self.run_site('test_dead_owner_is_reclaimed', run=True)

    def test_cohort_descendant_site(self):
        self.run_site('test_descendant_reuses_parent_reservation_without_deadlock', run=True)

    @unittest.skipUnless(os.name == 'posix', 'POSIX stage-group cleanup')
    def test_cancel_backend_site(self):
        seen = []

        def fake_run_backend(arguments, **kwargs):
            seen.append(kwargs.get('env'))
            raise ChildStarted
        case = test_ddsurfer_sync.DDSurferSyncTests
        name = 'test_cancel_reaps_upstream_stage_in_its_separate_process_group'
        with patch.object(test_ddsurfer_sync, 'run_backend', fake_run_backend):
            with self.assertRaises(ChildStarted):
                getattr(case(name), name)()
        self.assertEqual(seen, [{'PYTHONPATH': self.expected}])


if __name__ == '__main__':
    unittest.main()
