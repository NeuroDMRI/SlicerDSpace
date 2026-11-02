"""Direct region arguments reuse the request backend without manual JSON files."""

import contextlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts import analyze_regions


class RegionCliTests(unittest.TestCase):
    def call(self, *args):
        with patch.object(analyze_regions.signal, 'signal'), contextlib.redirect_stdout(io.StringIO()):
            return analyze_regions.main(list(args))

    def test_names_ids_commas_spaces_and_repeated_flags(self):
        with patch.object(analyze_regions, 'run_analysis', return_value=Path('/results/manifest.json')) as run:
            self.call('--source-manifest', '/study/run.json', '--regions', '1024,2024',
                      '--regions', 'Left Hippocampus', '--metrics', 'fa,md', '--metrics', 'fa', 'noddi_ndi',
                      '--output-dir', '/results/new', '--atlas', '/atlas.nii.gz', '--lut', '/atlas.txt')
        request = run.call_args.args[0]
        self.assertEqual(request['selected_regions'], ['1024', '2024', 'Left Hippocampus'])
        self.assertEqual(request['metrics'], ['fa', 'md', 'noddi_ndi'])
        self.assertEqual(request['atlas_override'], str(Path('/atlas.nii.gz').resolve()))
        self.assertEqual(request['lut_path'], str(Path('/atlas.txt').resolve()))
        self.assertEqual(request['source_manifest'], str(Path('/study/run.json').resolve()))

    def test_default_fa_does_not_leak_into_other_requested_analyses(self):
        for flags, metrics in (([], ['fa']), (['--tissue'], []), (['--surface'], []),
                               (['--connectome', 'connectome.streamline_count'], [])):
            with self.subTest(flags=flags), patch.object(analyze_regions, 'run_analysis') as run:
                self.call('--source-manifest', '/run.json', '--regions', '1024', '--output-dir', '/results', *flags)
                self.assertEqual(run.call_args.args[0]['metrics'], metrics)

    def test_tract_rules_and_waypoints_pass_through(self):
        with patch.object(analyze_regions, 'run_analysis') as run:
            self.call('--source-manifest', '/run.json', '--regions', '1024', '--output-dir', '/results',
                      '--tractogram', 'tractography.ukf', '--tract-rule', 'endpoint_pair',
                      '--target-regions', '2024', '--include-regions', '16', '--exclude-regions', '8,47')
        request = run.call_args.args[0]
        self.assertEqual(request['tract_rule'], 'endpoint_pair')
        self.assertEqual(request['target_regions'], ['2024'])
        self.assertEqual(request['include_regions'], ['16'])
        self.assertEqual(request['exclude_regions'], ['8', '47'])
        self.assertEqual(request['metrics'], [])

    def test_multiple_manifests_create_a_shared_cohort_request(self):
        with patch('shared.region_batch.run_region_batch', return_value=Path('/batch/manifest.json')) as run, \
                patch.object(analyze_regions, 'read_json', return_value={'status': 'completed'}):
            self.call('--source-manifest', '/a.json', '/b.json', '--source-manifest', '/c.json',
                      '--regions', '1024', '2024', '--metrics', 'fa', 'md', '--output-dir', '/batch', '--tissue')
        config = run.call_args.args[0]
        self.assertEqual([item['source_manifest'] for item in config['subjects']],
                         [str(Path(path).resolve()) for path in ('/a.json', '/b.json', '/c.json')])
        self.assertNotIn('source_manifest', config['request'])
        self.assertNotIn('output_dir', config['request'])
        self.assertTrue(config['request']['tissue'])

    def test_partial_cohort_keeps_failure_exit_status(self):
        with patch('shared.region_batch.run_region_batch', return_value=Path('/batch/manifest.json')), \
                patch.object(analyze_regions, 'read_json', return_value={'status': 'partial_failed'}), \
                self.assertRaises(SystemExit) as raised:
            self.call('--source-manifest', '/a.json', '/b.json', '--regions', '1024', '--output-dir', '/batch')
        self.assertEqual(raised.exception.code, 1)

    def test_invalid_or_mixed_arguments_do_not_start_computation(self):
        base = ['--source-manifest', '/run.json', '--regions', '1024', '--output-dir', '/results']
        cases = [base + ['--metrics', ','], base + ['--target-regions', '2024'],
                 base + ['--tract-rule', 'through_all'], base + ['--tractogram', 'tractography.ukf',
                 '--tract-rule', 'endpoint_pair'], ['--request', '/request.json', '--regions', '1024'],
                 ['--request', '/request.json', '--atlas', '/atlas.nii.gz'],
                 ['--source-manifest', '/run.json', '/run.json', '--regions', '1024', '--output-dir', '/results'],
                 ['--source-manifest', '/run.json', '--regions', '1024'],
                 ['--source-manifest', '/run.json', '--output-dir', '/results']]
        for args in cases:
            with self.subTest(args=args), patch.object(analyze_regions, 'run_analysis') as run, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                self.call(*args)
            self.assertEqual(raised.exception.code, 2)
            run.assert_not_called()

    def test_inspect_can_print_without_creating_a_catalog_file(self):
        with patch.object(analyze_regions, 'inspect_run', return_value={'regions': []}), \
                patch.object(analyze_regions, 'write_json') as write:
            self.call('--inspect', '/run.json')
        write.assert_not_called()

    def test_json_request_mode_remains_unchanged(self):
        request = dict(source_manifest='/run.json', selected_labels=[1024], metrics=['fa'], output_dir='/results')
        with patch.object(analyze_regions, 'read_json', return_value=request), \
                patch.object(analyze_regions, 'run_analysis') as run:
            self.call('--request', '/request.json')
        run.assert_called_once_with(request)


if __name__ == '__main__':
    unittest.main()
