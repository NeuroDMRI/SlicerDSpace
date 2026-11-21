"""Explicit CLI bundle routing and current-run safety contracts."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from shared.multi_tracking import variants
from shared.orchestrator import ProductOrchestratorConfig, _csd_bundle_parameters_current, _dti_bundle_qc_acceptable, _dti_bundle_source_current, _validate_bundle_tractography_source, plan_subject_products, run_subject_products
from shared.run_subject_pipeline_cli import main


class BundleSourceTests(unittest.TestCase):
    def test_only_requested_variant_receives_bundle_targets(self):
        config = ProductOrchestratorConfig('sub', Path('/tmp/sub'),
            ('segmentation', 'tractography', 'bundles', 'wma', 'tractcloud', 'connectome'),
            tractography_backend='csd_act,ukf', bundle_tractography_backend='ukf')
        first, second = variants(config)
        self.assertEqual(first.targets, ('segmentation', 'tractography', 'connectome'))
        self.assertIsNone(first.bundle_tractography_backend)
        self.assertEqual(second.targets, ('tractography', 'bundles', 'wma', 'tractcloud', 'connectome'))
        self.assertEqual(second.bundle_tractography_backend, 'ukf')
        _validate_bundle_tractography_source(first)
        _validate_bundle_tractography_source(second)

    def test_reject_saf_missing_tracking_and_conflicting_inputs(self):
        common = dict(subject_id='s', subject_root=Path('/tmp/s'), targets=('tractography', 'bundles'))
        for values in (dict(tractography_backend='saf', bundle_tractography_backend='saf'),
                       dict(tractography_backend='ukf', bundle_tractography_backend='dti_det'),
                       dict(tractography_backend='ukf', bundle_tractography_backend='ukf', wma_input_tractography=Path('/old.vtk'))):
            with self.subTest(values=values), self.assertRaises(ValueError):
                _validate_bundle_tractography_source(ProductOrchestratorConfig(**common, **values))

    def test_cli_selects_source_and_adds_tractography_to_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            captured = []
            def fake_plan(config, write=True):
                captured.append(config)
                return SimpleNamespace(run_plan_path=Path(tmp) / 'plan.json', to_dict=lambda: {})
            with patch('shared.run_subject_pipeline_cli.plan_subject_products', side_effect=fake_plan), contextlib.redirect_stdout(io.StringIO()):
                rc = main(['--no-user-config', '--subject-root', tmp, '--targets', 'bundles',
                           '--bundle-backend', 'wmql,tractcloud', '--bundle-tractography-backend', 'ukf',
                           '--plan', '--skip-environment-check'])
            self.assertEqual(rc, 0)
            self.assertEqual(captured[0].tractography_backend, 'ukf')
            self.assertEqual(captured[0].bundle_tractography_backend, 'ukf')
            self.assertIn('tractography', captured[0].targets)
            self.assertIn('tractcloud', captured[0].targets)

    def test_cli_adds_source_without_removing_explicit_tracking(self):
        with tempfile.TemporaryDirectory() as tmp:
            captured = []
            def fake_plan(config, write=True):
                captured.append(config)
                return SimpleNamespace(run_plan_path=Path(tmp) / 'plan.json', to_dict=lambda: {})
            with patch('shared.run_subject_pipeline_cli.plan_subject_products', side_effect=fake_plan), contextlib.redirect_stdout(io.StringIO()):
                main(['--no-user-config', '--subject-root', tmp, '--targets', 'bundles',
                      '--tractography-backend', 'csd_act', '--bundle-tractography-backend', 'ukf',
                      '--plan', '--skip-environment-check'])
            self.assertEqual(captured[0].tractography_backend, 'csd_act,ukf')

    def test_doctor_checks_added_bundle_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            captured = []
            def fake_doctor(args, *, targets, output_root):
                captured.append((args.tractography_backend, targets))
                return SimpleNamespace(to_dict=lambda: {}), {}
            with patch('shared.run_subject_pipeline_cli._run_environment_doctor_from_args', side_effect=fake_doctor), contextlib.redirect_stdout(io.StringIO()):
                main(['--no-user-config', '--subject-root', tmp, '--targets', 'bundles',
                      '--bundle-tractography-backend', 'dti_det', '--doctor'])
            self.assertEqual(captured[0][0], 'dti_det')
            self.assertIn('tractography', captured[0][1])

    def test_cli_rejects_conflicts_and_skipped_tracking(self):
        for extra in (['--skip-targets', 'tractography'], ['--wma-input-tractography', '/old.vtk']):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(['--no-user-config', '--subject-root', tmp, '--targets', 'bundles',
                      '--bundle-tractography-backend', 'ukf', *extra])

    def test_dti_cache_requires_matching_inputs_and_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = {key: root / key for key in ('dwi', 'bval', 'bvec', 'mask')}
            for file in inputs.values():
                file.write_text('data')
            config = ProductOrchestratorConfig('s', root, ('tractography', 'bundles'),
                tractography_backend='dti_det', bundle_tractography_backend='dti_det')
            records = {key: {'path': path.resolve().as_posix(), 'exists': True,
                             'size': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
                       for key, path in inputs.items()}
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'status': 'completed', 'raw_inputs': records,
                'parameters': {'fit_method': config.microstructure_fit_method, 'tracking_method': 'deterministic',
                    'output_format': 'trk', 'max_seed_voxels': config.select_streamlines,
                    'min_length_mm': config.dti_min_length_mm, 'max_length_mm': config.dti_max_length_mm}}))
            self.assertTrue(_dti_bundle_source_current(config, manifest, inputs))
            inputs['bval'].write_text('changed')
            self.assertFalse(_dti_bundle_source_current(config, manifest, inputs))

    def test_real_plan_uses_selected_dti_for_all_bundle_backends(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / 'input'
            inputs.mkdir()
            for name in ('data.nii.gz', 'bvals', 'bvecs', 'nodif_brain_mask.nii.gz'):
                (inputs / name).write_text('fixture')
            parcel = root / 'parcel.nii.gz'
            parcel.write_text('fixture')
            config = ProductOrchestratorConfig('s', root, ('tractography', 'bundles'),
                tractography_backend='dti_det', bundle_tractography_backend='dti_det',
                bundle_backend='wmql,wma,tractcloud', wma_atlas_dir=root, resume=False)
            with patch('shared.orchestrator._resolve_parcellation', return_value=parcel), \
                 patch('shared.orchestrator._resolve_slicer_executable', return_value=root / 'Slicer'):
                plan = plan_subject_products(config, write=False)
            items = {item.target: item for item in plan.items}
            self.assertEqual(items['tractography'].status, 'will_run')
            self.assertEqual(items['bundles'].status, 'will_run')
            self.assertEqual(items['wma'].status, 'will_run')
            self.assertEqual(items['tractcloud'].status, 'will_run')
            self.assertEqual(items['wma'].inputs['input_tractography'], items['tractography'].outputs['tractogram_vtk'])
            self.assertEqual(items['tractcloud'].inputs['input_tractography'], items['tractography'].outputs['tractogram_vtk'])
            self.assertEqual(items['bundles'].inputs['tractogram_trk'], items['tractography'].outputs['tractogram_trk'])

    def test_dti_bundle_qc_rejects_empty_or_out_of_range_streamlines(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = ProductOrchestratorConfig('s', root, ('tractography', 'bundles'))
            tractogram, qc = root / 'tract.trk', root / 'qc.json'
            tractogram.write_bytes(b'tracks')
            payload = dict(status='completed', streamlines_exists=True, streamline_count=10,
                           length_min_mm=config.dti_min_length_mm, length_max_mm=config.dti_max_length_mm)
            qc.write_text(json.dumps(payload))
            self.assertTrue(_dti_bundle_qc_acceptable(config, qc, tractogram))
            for key, value in [('streamline_count', 0), ('length_max_mm', config.dti_max_length_mm + 1),
                               ('length_min_mm', float('nan'))]:
                altered = dict(payload, **{key: value})
                qc.write_text(json.dumps(altered))
                self.assertFalse(_dti_bundle_qc_acceptable(config, qc, tractogram))

    def test_csd_reuse_requires_selected_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = ProductOrchestratorConfig('s', root, ('tractography', 'bundles'))
            params = dict(select_streamlines=config.select_streamlines, min_length_mm=config.dti_min_length_mm,
                          max_length_mm=config.dti_max_length_mm, random_seed=config.mrtrix_random_seed,
                          seeding=config.tractography_seeding, nthreads=config.mrtrix_nthreads)
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'parameters': params}))
            self.assertTrue(_csd_bundle_parameters_current(config, manifest))
            params['select_streamlines'] += 1
            manifest.write_text(json.dumps({'parameters': params}))
            self.assertFalse(_csd_bundle_parameters_current(config, manifest))

    def test_missing_current_tracking_blocks_even_if_old_bundle_cache_claims_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = ProductOrchestratorConfig('s', root, ('tractography', 'bundles'),
                tractography_backend='dti_det', bundle_tractography_backend='dti_det',
                bundle_backend='wmql,wma,tractcloud', wma_atlas_dir=root, execute=True)
            with patch('shared.orchestrator._wma_complete', return_value=True), \
                 patch('shared.orchestrator._wmql_complete', return_value=True), \
                 patch('shared.orchestrator._tractcloud_complete', return_value=True):
                result = run_subject_products(config)
            tasks = {task.name: task for task in result.tasks}
            self.assertEqual(tasks['tractography.dti_det'].status, 'blocked')
            for name in ('bundles.wmql', 'bundles.wma', 'bundles.tractcloud'):
                self.assertEqual(tasks[name].status, 'blocked')
                self.assertIn('current-run QC', tasks[name].reason)

    def test_dti_runtime_hands_same_converted_vtk_to_wma_and_tractcloud(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / 'input'
            inputs.mkdir()
            for name in ('data.nii.gz', 'bvals', 'bvecs', 'nodif_brain_mask.nii.gz'):
                (inputs / name).write_text('fixture')
            from test_product_orchestrator import _write_valid_raw_subject
            _write_valid_raw_subject(root)
            config = ProductOrchestratorConfig('s', root, ('tractography', 'bundles'),
                tractography_backend='dti_det', bundle_tractography_backend='dti_det',
                bundle_backend='wma,tractcloud', wma_atlas_dir=root, execute=True, resume=False)
            received = {}

            def fake_tracking(options, log=None):
                options.output_dir.mkdir(parents=True, exist_ok=True)
                track = options.output_dir / 's-dti-deterministic.trk'
                track.write_bytes(b'tracks')
                manifest = options.output_dir / 's-dti-deterministic_manifest.json'
                qc = options.output_dir / 's-dti-deterministic_qc.json'
                bundle = options.input_bundle
                paths = {'dwi': bundle.dwi_path, 'bval': bundle.bval_path, 'bvec': bundle.bvec_path, 'mask': bundle.mask_path}
                records = {key: {'path': path.resolve().as_posix(), 'size': path.stat().st_size,
                                 'mtime_ns': path.stat().st_mtime_ns} for key, path in paths.items()}
                # Gradient frame (batch 2): the fixture DWI has affine eye(4) (det > 0); the real run writes this marker.
                from tractography.dti import DIPY_GRADIENT_FRAME
                manifest.write_text(json.dumps({'status': 'completed', 'raw_inputs': records,
                    'gradient_frame': DIPY_GRADIENT_FRAME,
                    'parameters': {'fit_method': options.fit_method, 'tracking_method': 'deterministic',
                        'output_format': 'trk', 'max_seed_voxels': options.max_seed_voxels,
                        'min_length_mm': options.min_length_mm, 'max_length_mm': options.max_length_mm}}))
                qc.write_text(json.dumps({'status': 'completed', 'streamline_count': 10,
                    'streamlines_exists': True, 'length_min_mm': options.min_length_mm,
                    'length_max_mm': options.max_length_mm}))
                return SimpleNamespace(streamlines_path=track, manifest_path=manifest, qc_path=qc)

            def fake_convert(source, destination):
                destination.write_bytes(b'vtk')
                return destination

            def fake_bundle(name):
                def run(options, log=None):
                    received[name] = options.input_tractography
                    return SimpleNamespace(status='completed')
                return run

            with patch('tractography.dti.run_dti_tracking', side_effect=fake_tracking), \
                 patch('tractography.conversion.ensure_vtk_tractogram', side_effect=fake_convert), \
                 patch('tractography.run_wma_org_atlas', side_effect=fake_bundle('wma')), \
                 patch('tractography.run_tractcloud_parcellation', side_effect=fake_bundle('tractcloud')), \
                 patch('shared.orchestrator._wma_complete', return_value=True), \
                 patch('shared.orchestrator._resolve_slicer_executable', return_value=root / 'Slicer'):
                result = run_subject_products(config)
            tasks = {task.name: task for task in result.tasks}
            self.assertEqual(tasks['tractography.dti_det'].status, 'completed')
            self.assertEqual(tasks['bundles.wma'].status, 'completed')
            self.assertEqual(tasks['bundles.tractcloud'].status, 'completed')
            self.assertEqual(received['wma'], received['tractcloud'])
            self.assertTrue(received['wma'].name.endswith('dti-deterministic.vtk'))


if __name__ == '__main__':
    unittest.main()
