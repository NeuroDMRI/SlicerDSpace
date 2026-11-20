"""Cache invalidation and original-author dispatch, without optional WMA."""

import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tractography.wma_registration import RegistrationPreprocessCache, registration_cache_support, default_cache_bytes
from scripts.run_wma_registration import main


class PolyData:
    def __init__(self, value=0):
        self.value = value
        self.mtime = 1

    def GetMTime(self):
        return self.mtime

    def GetActualMemorySize(self):
        return 1


def preprocess(inpd, min_length_mm, remove_u=False, remove_u_endpoint_dist=40,
               remove_brainstem=False, return_indices=False, return_lengths=False,
               preserve_point_data=False, preserve_cell_data=False, verbose=True, max_length_mm=None):
    return PolyData(inpd.value + min_length_mm)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        def original(*args, **kwargs):
            self.calls.append((args, kwargs))
            return preprocess(*args, **kwargs)
        import functools
        self.cache = RegistrationPreprocessCache(functools.wraps(preprocess)(original))
        self.source = PolyData(2)

    def call(self, source=None, **kwargs):
        return self.cache(self.source if source is None else source, 40, verbose=False, **kwargs)

    def test_repeated_filter_is_identical_object_and_does_not_reexecute(self):
        first = self.call(max_length_mm=260)
        self.assertIs(first, self.call(max_length_mm=260))
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.cache.summary()['hits'], 1)

    def test_atlas_subject_cache_is_bounded(self):
        atlas = PolyData(10)
        self.call()
        first = self.call(atlas)
        self.call()
        self.assertIs(first, self.call(atlas))
        self.call(PolyData(20))
        self.assertEqual(len(self.cache.entries), 2)
        self.assertEqual(self.cache.hits, 2)

    def test_memory_limit_evicts_instead_of_growing_without_bound(self):
        cache = RegistrationPreprocessCache(preprocess, max_bytes=1024)
        first = cache(self.source, 40, verbose=False)
        cache(PolyData(12), 40, verbose=False)
        self.assertEqual(cache.retained_bytes, 1024)
        self.assertEqual(len(cache.entries), 1)
        self.assertIsNot(first, cache(self.source, 40, verbose=False))

    def test_oversized_or_disabled_memory_cache_returns_original_output(self):
        for size in (0, 512):
            cache = RegistrationPreprocessCache(preprocess, max_bytes=size)
            first = cache(self.source, 40, verbose=False)
            second = cache(self.source, 40, verbose=False)
            self.assertEqual(first.value, second.value)
            self.assertIsNot(first, second)
            self.assertEqual(cache.oversized, 2)
            self.assertEqual(cache.retained_bytes, 0)

    def test_default_memory_budget_is_conservative_and_has_no_new_dependency(self):
        with mock.patch.dict(sys.modules, {'psutil': None}):
            self.assertEqual(default_cache_bytes(), 256 * 1024 ** 2)
        psutil = types.SimpleNamespace(virtual_memory=lambda: types.SimpleNamespace(available=8 * 1024 ** 3))
        # M1/F2: pin the Rosetta page ratio so the result does not depend on the host running this test
        with mock.patch.dict(sys.modules, {'psutil': psutil}), \
                mock.patch('shared.platform_info.rosetta_page_ratio', return_value=1):
            self.assertEqual(default_cache_bytes(), 2 * 1024 ** 3)

    def test_input_modification_invalidates(self):
        first = self.call()
        self.source.mtime += 1
        self.assertIsNot(first, self.call())

    def test_filtered_output_modification_invalidates(self):
        first = self.call()
        first.mtime += 1
        self.assertIsNot(first, self.call())

    def test_length_arguments_invalidate(self):
        self.call(max_length_mm=260)
        self.call(max_length_mm=200)
        self.cache(self.source, 60, verbose=False, max_length_mm=200)
        self.assertEqual(len(self.calls), 3)

    def test_requested_attributes_indices_and_other_filters_bypass(self):
        for option in ('remove_u', 'remove_brainstem', 'return_indices', 'return_lengths',
                       'preserve_point_data', 'preserve_cell_data'):
            self.call(**{option: True})
            self.call(**{option: True})
        self.cache(self.source, 40)
        self.assertEqual(self.cache.bypassed, 13)
        self.assertEqual(len(self.cache.entries), 0)

    def test_unknown_author_signature_runs_without_cache(self):
        calls = []
        def new_version(inpd, min_length_mm, new_behavior=True):
            calls.append(inpd)
            return PolyData()
        cache = RegistrationPreprocessCache(new_version)
        cache(self.source, 40)
        cache(self.source, 40)
        self.assertEqual(len(calls), 2)
        self.assertEqual(cache.bypassed, 2)

    def test_missing_author_sources_not_optimized(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(registration_cache_support(Path(tmp))['supported'])

    def test_unverified_conversion_does_not_import_optional_vtk(self):
        from tractography.wma_fibers import fiber_conversion_support
        import builtins
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name == 'vtk' or name.startswith('vtkmodules'):
                raise AssertionError('Unverified author source must not load VTK')
            return original_import(name, *args, **kwargs)
        with tempfile.TemporaryDirectory() as tmp, mock.patch('builtins.__import__', side_effect=guarded_import):
            record = fiber_conversion_support(Path(tmp))
        self.assertFalse(record['supported'])
        self.assertIsNone(record['source_sha256'])
        self.assertIsNone(record['vtk_version'])

    def test_vtk_child_point_modification_invalidates(self):
        try:
            import vtk
        except ImportError:
            self.skipTest('Optional VTK unavailable')
        points = vtk.vtkPoints()
        points.InsertNextPoint(0, 0, 0)
        source = vtk.vtkPolyData()
        source.SetPoints(points)
        def original(inpd, min_length_mm, remove_u=False, remove_u_endpoint_dist=40,
                     remove_brainstem=False, return_indices=False, return_lengths=False,
                     preserve_point_data=False, preserve_cell_data=False, verbose=True, max_length_mm=None):
            output = vtk.vtkPolyData()
            output.DeepCopy(inpd)
            return output
        cache = RegistrationPreprocessCache(original)
        first = cache(source, 40, verbose=False)
        self.assertIs(first, cache(source, 40, verbose=False))
        points.SetPoint(0, 1, 2, 3)
        points.Modified()
        updated = cache(source, 40, verbose=False)
        self.assertIsNot(first, updated)
        self.assertEqual(updated.GetPoint(0), (1, 2, 3))

    def test_author_error_preserved_and_hook_argv_restored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / 'author.py'
            script.write_text('import sys\nassert sys.argv[1:] == ["-mode", "affine"]\nraise SystemExit(7)\n')
            filters = types.SimpleNamespace(preprocess=preprocess)
            wma = types.SimpleNamespace(__file__=str(root / '__init__.py'), filter=filters)
            previous_argv = sys.argv
            with mock.patch.dict(sys.modules, {'whitematteranalysis': wma}), mock.patch(
                    'scripts.run_wma_registration.registration_cache_support', return_value={'supported': True}):
                with self.assertRaises(SystemExit) as caught:
                    main(['--author-script', str(script), '--report-dir', str(root), '--', '-mode', 'affine'])
            self.assertEqual(caught.exception.code, 7)
            self.assertIs(filters.preprocess, preprocess)
            self.assertIs(sys.argv, previous_argv)
            report = json.loads(next(root.glob('registration_cache_*.json')).read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['exit_code'], 7)
            self.assertTrue(report['enabled'])

    def test_unsupported_source_dispatches_unmodified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / 'author.py'
            script.write_text('import whitematteranalysis as wma\nassert wma.filter.preprocess.__name__ == "preprocess"\n')
            filters = types.SimpleNamespace(preprocess=preprocess)
            wma = types.SimpleNamespace(__file__=str(root / '__init__.py'), filter=filters)
            with mock.patch.dict(sys.modules, {'whitematteranalysis': wma}):
                main(['--author-script', str(script), '--report-dir', str(root)])
            report = json.loads(next(root.glob('registration_cache_*.json')).read_text())
            self.assertEqual(report['status'], 'completed')
            self.assertFalse(report['enabled'])

    def test_transform_hook_restored_after_author_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root/'author.py'
            script.write_text('import whitematteranalysis as wma\n'
                              'assert wma.register_two_subjects.transform_fiber_array_numpy.__wrapped__ is wma.original_transform\n'
                              'raise SystemExit(7)\n')
            def transform(*args, **kwargs):
                return 'author'
            module = types.SimpleNamespace(transform_fiber_array_numpy=transform, convert_transform_to_vtk=object())
            wma = types.SimpleNamespace(__file__=str(root/'__init__.py'),
                                        filter=types.SimpleNamespace(preprocess=preprocess),
                                        register_two_subjects=module, original_transform=transform)
            with mock.patch.dict(sys.modules, {'whitematteranalysis': wma}), mock.patch(
                    'scripts.run_wma_registration.registration_transform_support', return_value={'supported': True}):
                with self.assertRaises(SystemExit):
                    main(['--author-script', str(script), '--report-dir', str(root)])
            self.assertIs(module.transform_fiber_array_numpy, transform)
            report = json.loads(next(root.glob('registration_cache_*.json')).read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertIn('fiber_transform', report)

    def test_fiber_converter_restored_after_author_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root/'author.py'
            script.write_text('import whitematteranalysis as wma\n'
                              'assert wma.fibers.FiberArray.convert_from_polydata.__wrapped__ is wma.original\n'
                              'raise SystemExit(7)\n')
            class FiberArray:
                def convert_from_polydata(self, source, points_per_fiber=None):
                    return 'original'
                def _calculate_line_indices(self, *args):
                    return []
            original = FiberArray.convert_from_polydata
            wma = types.SimpleNamespace(__file__=str(root/'__init__.py'),
                                        fibers=types.SimpleNamespace(FiberArray=FiberArray), original=original)
            with mock.patch.dict(sys.modules, {'whitematteranalysis': wma}), mock.patch(
                    'scripts.run_wma_registration.fiber_conversion_support', return_value={'supported': True}):
                with self.assertRaises(SystemExit):
                    main(['--author-script', str(script), '--report-dir', str(root)])
            self.assertIs(FiberArray.convert_from_polydata, original)
            report = json.loads(next(root.glob('registration_cache_*.json')).read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertIn('fiber_array_conversion', report)


if __name__ == '__main__':
    unittest.main()
