"""Automatic cohort defaults, data-dependent estimates and manual overrides."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from shared.cohort_planning import image_size, plan_cohort, available_cohort_resources

GIB = 1024**3


class CohortPlanningTests(unittest.TestCase):
    def subjects(self, root, count=4):
        return [SimpleNamespace(subject_id=f'sub-{i}', subject_root=Path(root) / f'sub-{i}') for i in range(count)]

    def test_default_jobs_is_conservative_and_explicit_jobs_is_retained(self):
        subjects = self.subjects('/unused')
        args = ['--targets', 'microstructure', '--microstructure-stages', 'dti']
        auto = plan_cohort(subjects, args, available=(16, 64 * GIB))
        self.assertEqual(auto['jobs'], 2)
        self.assertTrue(auto['jobs_automatic'])
        explicit = plan_cohort(subjects, args, jobs=3, available=(16, 64 * GIB))
        self.assertEqual(explicit['jobs'], 3)
        self.assertFalse(explicit['jobs_automatic'])
        self.assertTrue(auto['input_headers']['sub-0']['estimated'])

    def test_single_core_can_plan_without_eight_thread_default(self):
        plan = plan_cohort(self.subjects('/unused'), ['--targets', 'connectome'], available=(1, 32 * GIB))
        self.assertEqual(plan['stage_cpus'], 1)
        self.assertEqual(plan['jobs'], 1)

    def test_small_and_large_headers_change_memory_without_loading_arrays(self):
        with TemporaryDirectory() as tmp:
            subjects = self.subjects(tmp, 2)
            for subject in subjects:
                (subject.subject_root / 'input').mkdir(parents=True)
                (subject.subject_root / 'input/data.nii.gz').write_bytes(b'header-only fixture')
            sizes = [SimpleNamespace(shape=(32, 32, 32, 8)), SimpleNamespace(shape=(144, 174, 144, 288))]
            with patch('nibabel.load', side_effect=sizes):
                plan = plan_cohort(subjects, ['--targets', 'microstructure'], available=(16, 128 * GIB))
            small = plan['subject_profiles']['sub-0']['microstructure']['memory_gb']
            large = plan['subject_profiles']['sub-1']['microstructure']['memory_gb']
            self.assertGreater(large, small)
            self.assertGreater(plan['subject_profiles']['sub-0']['parcellation']['memory_gb'], small)
            self.assertEqual((subjects[0].subject_root / 'input/data.nii.gz').read_bytes(), b'header-only fixture')

    def test_manual_stage_overrides_take_precedence(self):
        plan = plan_cohort(self.subjects('/unused'), [], jobs=3, cpu_budget=2, memory_gb=2,
                          stage_cpus=1, stage_memory_gb=1, available=(100, 100 * GIB))
        self.assertEqual(plan['jobs'], 3)
        self.assertEqual(plan['stage_memory_gb'], 1)
        self.assertTrue(all(row == dict(cpu=1, memory_gb=1)
                            for profile in plan['subject_profiles'].values() for row in profile.values()))

    def test_explicit_algorithm_threads_are_reserved_not_silently_reduced(self):
        plan = plan_cohort(self.subjects('/unused'), ['--targets', 'surface', '--surface-cpu-threads', '12'],
                          available=(16, 128 * GIB))
        self.assertEqual(plan['stage_cpus'], 12)
        with self.assertRaises(ValueError):
            plan_cohort(self.subjects('/unused'), ['--surface-cpu-threads', '12'],
                        stage_cpus=4, available=(16, 128 * GIB))

    def test_insufficient_memory_and_unbounded_workers_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'estimated stage RAM'):
            plan_cohort(self.subjects('/unused'), ['--targets', 'parcellation'], available=(16, 4 * GIB))
        with self.assertRaises(ValueError):
            plan_cohort(self.subjects('/unused'), ['--microstructure-free-water-n-jobs', '-1'],
                        available=(16, 128 * GIB))
        with self.assertRaises(ValueError):
            plan_cohort(self.subjects('/unused'), [], stage_memory_gb=float('nan'), available=(16, 128 * GIB))

    def test_unknown_headers_do_not_abort_planning(self):
        with TemporaryDirectory() as tmp:
            (Path(tmp) / 'input').mkdir()
            (Path(tmp) / 'input/data.nii.gz').write_text('truncated fixture')
            self.assertTrue(image_size(tmp)['estimated'])

    def test_nrrd_measurement_axis_is_normalized_from_header(self):
        with TemporaryDirectory() as tmp:
            (Path(tmp) / 'data.nhdr').touch()
            with patch('nrrd.read_header', return_value=dict(sizes=[90, 64, 64, 48],
                       kinds=['list', 'space', 'space', 'space'])):
                record = image_size(tmp)
            self.assertEqual(record['shape'], [64, 64, 48, 90])
            self.assertFalse(record['estimated'])

    def test_container_cpu_and_remaining_memory_limits_are_honoured(self):
        files = {'/sys/fs/cgroup/cpu.max': '200000 100000',
                 '/sys/fs/cgroup/memory.max': str(8 * GIB),
                 '/sys/fs/cgroup/memory.current': str(2 * GIB)}
        def read(path, *args, **kwargs):
            if str(path) not in files:
                raise OSError('absent')
            return files[str(path)]
        with patch('shared.compute_schedule.available_resources', return_value=(64, 100 * GIB)), \
             patch.object(Path, 'read_text', read):
            cpu, memory = available_cohort_resources()
        self.assertEqual(cpu, 2)
        self.assertEqual(memory, int(6 * GIB * .7))


if __name__ == '__main__':
    unittest.main()
