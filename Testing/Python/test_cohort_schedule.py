"""Process-level resource accounting, cleanup and unchanged task results."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from unittest.mock import Mock
from contextlib import nullcontext
from types import SimpleNamespace

from shared.cohort_schedule import ENV, create_schedule, stage_admission, scheduled_stage

ROOT = Path(__file__).resolve().parents[2]
WORKER = '''
import os, time, json
from pathlib import Path
from shared.cohort_schedule import stage_admission
for name, gpu in [('inference',True),('fit',False)]:
    with stage_admission(name, gpu=gpu):
        time.sleep(.18)
Path(os.environ['RESULT']).write_text(json.dumps({'value': sum(i*i for i in range(100))}))
'''


class CohortScheduleTests(unittest.TestCase):
    def create(self, root, **kwargs):
        return create_schedule(Path(root) / 'schedule', cpu_budget=2, memory_gb=2,
                               stage_cpus=1, stage_memory_gb=1, **kwargs)

    def test_disabled_preserves_return_and_exception(self):
        @scheduled_stage('disabled')
        def function(value):
            if value < 0:
                raise ValueError('fixture')
            return value * 2
        with patch.dict(os.environ, {ENV: ''}):
            self.assertEqual(function(4), 8)
            with self.assertRaises(ValueError):
                function(-1)

    def test_invalid_or_oversized_budgets_fail_before_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            for memory in (0, -1, float('nan'), float('inf'), .5):
                with self.assertRaises(ValueError):
                    create_schedule(Path(tmp) / 'unused', cpu_budget=2, memory_gb=memory,
                                    stage_cpus=1, stage_memory_gb=1)
            self.assertFalse((Path(tmp) / 'unused').exists())

    def test_nested_and_failed_stages_release_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = self.create(tmp)
            with patch.dict(os.environ, {ENV: str(folder)}):
                with self.assertRaisesRegex(ValueError, 'fixture'):
                    with stage_admission('outer', gpu=True):
                        with stage_admission('inner', gpu=True):
                            raise ValueError('fixture')
            state = json.loads((folder / 'state.json').read_text())
            self.assertEqual(state, dict(active=[], waiting=[]))
            rows = [json.loads(line) for line in (folder / 'events.jsonl').read_text().splitlines()]
            self.assertEqual([row['event'] for row in rows], ['queued', 'started', 'failed'])

    def test_gpu_stage_releases_unused_allocator_blocks_only_on_used_devices(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = self.create(tmp)
            cuda = Mock()
            cuda.is_initialized.return_value = True
            cuda.device_count.return_value = 2
            cuda.memory_reserved.side_effect = [128, 0]
            cuda.device.return_value = nullcontext()
            @scheduled_stage('inference', gpu=True)
            def inference():
                return 'unchanged'
            with patch.dict(os.environ, {ENV: str(folder)}), \
                 patch.dict(sys.modules, {'torch': SimpleNamespace(cuda=cuda)}):
                self.assertEqual(inference(), 'unchanged')
            cuda.device.assert_called_once_with(0)
            cuda.synchronize.assert_called_once_with()
            cuda.empty_cache.assert_called_once_with()
            self.assertEqual(json.loads((folder / 'state.json').read_text()), dict(active=[], waiting=[]))

    def test_gpu_serializes_but_cpu_fits_overlap_across_subjects(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = self.create(tmp)
            children = []
            try:
                for index in range(3):
                    env = {**os.environ, ENV: str(folder), 'PYTHONPATH': str(ROOT),
                           'SLICERDSPACE_COHORT_SUBJECT': str(index), 'RESULT': str(Path(tmp) / f'{index}.json'),
                           'OPENBLAS_NUM_THREADS': '1'}
                    children.append(subprocess.Popen([sys.executable, '-c', WORKER], env=env,
                                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE))
                for child in children:
                    out, error = child.communicate(timeout=30)
                    self.assertEqual(child.returncode, 0, error.decode())
            finally:
                for child in children:
                    if child.poll() is None:
                        child.kill()
                        child.communicate()
            outputs = [(Path(tmp) / f'{index}.json').read_bytes() for index in range(3)]
            self.assertEqual(len(set(outputs)), 1)
            active, mixed = {}, False
            rows = [json.loads(line) for line in (folder / 'events.jsonl').read_text().splitlines()]
            for row in rows:
                if row['event'] == 'started':
                    active[row['id']] = row
                    self.assertLessEqual(sum(item['cpu'] for item in active.values()), 2)
                    self.assertLessEqual(sum(item['memory_gb'] for item in active.values()), 2)
                    self.assertLessEqual(sum(item['gpu'] for item in active.values()), 1)
                    mixed |= {item['stage'] for item in active.values()} == {'inference', 'fit'}
                elif row['event'] in ('completed', 'failed', 'abandoned'):
                    active.pop(row['id'], None)
            self.assertTrue(mixed, 'A CPU fit should overlap a different subject inference')
            self.assertEqual(active, {})

    def test_dead_owner_is_reclaimed(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = self.create(tmp)
            code = "from shared.cohort_schedule import stage_admission; import os\nwith stage_admission('crash',gpu=True): os._exit(7)"
            env = {**os.environ, ENV: str(folder), 'PYTHONPATH': str(ROOT)}
            child = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, timeout=30)
            self.assertEqual(child.returncode, 7)
            with patch.dict(os.environ, {ENV: str(folder)}):
                with stage_admission('next', gpu=True):
                    pass
            events = (folder / 'events.jsonl').read_text()
            self.assertIn('abandoned', events)
            self.assertEqual(json.loads((folder / 'state.json').read_text()), dict(active=[], waiting=[]))

    def test_subject_stage_estimate_and_thread_defaults_are_applied_then_restored(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = create_schedule(Path(tmp) / 'schedule', cpu_budget=4, memory_gb=4,
                stage_cpus=4, stage_memory_gb=4,
                subject_profiles={'A': {'small': {'cpu': 1, 'memory_gb': .5}}})
            with patch.dict(os.environ, {ENV: str(folder), 'SLICERDSPACE_COHORT_SUBJECT': 'A', 'OMP_NUM_THREADS': '9'}):
                with stage_admission('small'):
                    state = json.loads((folder / 'state.json').read_text())
                    self.assertEqual(state['active'][0]['cpu'], 1)
                    self.assertEqual(state['active'][0]['memory_gb'], .5)
                    self.assertEqual(os.environ['OMP_NUM_THREADS'], '1')
                self.assertEqual(os.environ['OMP_NUM_THREADS'], '9')

    def test_descendant_reuses_parent_reservation_without_deadlock(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = self.create(tmp)
            env = {ENV: str(folder), 'PYTHONPATH': str(ROOT)}
            with patch.dict(os.environ, env):
                with stage_admission('parent', gpu=True):
                    child = subprocess.run([sys.executable, '-c',
                        "from shared.cohort_schedule import stage_admission\nwith stage_admission('child',gpu=True): print('ok')"],
                        capture_output=True, timeout=30)
            self.assertEqual(child.returncode, 0, child.stderr.decode())
            events = [json.loads(line) for line in (folder / 'events.jsonl').read_text().splitlines()]
            self.assertEqual([row['stage'] for row in events], ['parent'] * 3)


if __name__ == '__main__':
    unittest.main()
