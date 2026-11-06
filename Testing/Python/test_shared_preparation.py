"""Reuse identical prerequisites, not similar-looking scientific protocols."""

from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np
import nrrd

from shared import pipeline
from shared.cache_lock import cache_lock

ROOT = Path(__file__).resolve().parents[2]


def inputs(root, *, scaled=False):
    folder = root / 'input'
    folder.mkdir()
    data = np.arange(4 * 5 * 6 * 7, dtype=np.int16).reshape(4, 5, 6, 7)
    image = nib.Nifti1Image(data, np.diag([-2., 2., 2., 1.]))
    if scaled:
        image.header.set_slope_inter(2., 3.)
    nib.save(image, folder / 'data.nii.gz')
    nib.save(nib.Nifti1Image(np.ones(data.shape[:3], np.uint8), image.affine), folder / 'mask.nii.gz')
    np.savetxt(folder / 'bvals', [[0, 1000, 1000, 1000, 2000, 2000, 2000]])
    np.savetxt(folder / 'bvecs', np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                                        [1, 1, 0], [1, 0, 1], [0, 1, 1]], float).T)
    return pipeline.RawInputBundle(root, 'x', folder, folder / 'data.nii.gz', folder / 'bvals',
                                   folder / 'bvecs', folder / 'mask.nii.gz')


class SharedPreparationTests(unittest.TestCase):
    def test_headers_reused_across_fits_and_corrupt_payload_is_rebuilt(self):
        from anatomy.parcellation.conversion.nhdr_write import nhdr_write
        with tempfile.TemporaryDirectory() as tmp:
            raw = inputs(Path(tmp), scaled=True)
            with patch('anatomy.parcellation.conversion.nhdr_write.nhdr_write', wraps=nhdr_write) as convert:
                headers = pipeline.ensure_diffusion_headers(raw)
                self.assertEqual(pipeline.ensure_diffusion_headers(raw), headers)
                self.assertEqual(convert.call_count, 2)
                values, header = nrrd.read(str(headers[0]))
                np.testing.assert_array_equal(values, nib.load(raw.dwi_path).get_fdata(dtype=np.float32))
                self.assertEqual(header['DWMRI_gradient_0000'], '0.0   0.0   0.0')
                payload = Path(header['data file'])
                previous = payload.read_bytes()
                payload.write_bytes(b'corrupt')
                pipeline.ensure_diffusion_headers(raw)
                self.assertEqual(convert.call_count, 4)
                self.assertEqual(nrrd.read(str(headers[0]))[0].shape, values.shape)
                self.assertNotEqual(payload.read_bytes(), b'corrupt')
                self.assertTrue(previous)

    def test_changed_gradients_and_mask_invalidate_shared_headers(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = inputs(Path(tmp))
            header, _ = pipeline.ensure_diffusion_headers(raw)
            before = header.read_bytes()
            raw.bval_path.write_text('0 500 1000 1000 2000 2000 2000')
            pipeline.ensure_diffusion_headers(raw)
            self.assertNotEqual(before, header.read_bytes())
            manifest = header.parent / 'manifest.json'
            previous = json.loads(manifest.read_text())
            mask = nib.load(raw.mask_path)
            nib.save(nib.Nifti1Image(np.zeros(mask.shape, np.uint8), mask.affine), raw.mask_path)
            pipeline.ensure_diffusion_headers(raw)
            self.assertNotEqual(previous, json.loads(manifest.read_text()))

    def test_shared_header_matches_current_author_converter(self):
        # Load only this isolated helper, never the backend's generic preprocessing package.
        converter = ROOT / 'anatomy/surface/ddsurfer_backend/preprocessing/conversion'
        for scaled in (False, True):
            with self.subTest(scaled=scaled), tempfile.TemporaryDirectory() as tmp:
                raw = inputs(Path(tmp), scaled=scaled)
                shared, _ = pipeline.ensure_diffusion_headers(raw)
                spec = importlib.util.spec_from_file_location('author_gradient_fixture', converter / 'bval_bvec_io.py')
                gradient = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(gradient)
                spec = importlib.util.spec_from_file_location('author_header_fixture', converter / 'nhdr_write.py')
                author = importlib.util.module_from_spec(spec)
                with patch.dict('sys.modules', {'bval_bvec_io': gradient}):
                    spec.loader.exec_module(author)
                output = raw.input_dir / 'author.nhdr'
                author.write_nhdr(str(raw.dwi_path), str(raw.bval_path), str(raw.bvec_path), str(output))
                actual, actual_header = nrrd.read(str(shared))
                expected, expected_header = nrrd.read(str(output))
                np.testing.assert_array_equal(actual, expected)
                for key in expected_header:
                    if key == 'data file':
                        continue
                    if isinstance(expected_header[key], np.ndarray):
                        np.testing.assert_array_equal(actual_header[key], expected_header[key])
                    else:
                        self.assertEqual(actual_header[key], expected_header[key], key)

    def test_native_tensor_once_per_fit_not_once_per_consumer(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = inputs(Path(tmp))
            commands = []
            def execute(command, log=None):
                # W1: one teem_dti.py process per fit writes the tensor, baseline and six scalar maps.
                commands.append(command)
                arguments = command[2:]
                Path(arguments[arguments.index('--receipt') + 1]).write_text('{"kind": "teem_tensor_estimation"}')
                for target in [*arguments[-2:], *(item.partition('=')[2] for index, item in enumerate(arguments)
                                                  if arguments[index - 1] == '--scalar'),
                               *(arguments[index + 1] for index, item in enumerate(arguments) if item == '--dec')]:
                    Path(target).write_bytes(b'fixture output')
            def convert(source, target):
                target.write_bytes(source.read_bytes())
            with patch.object(pipeline, '_require_slicer'), \
                    patch.object(pipeline, 'resolve_pythonslicer_executable', return_value='PythonSlicer'), \
                    patch.object(pipeline, '_run_cli', side_effect=execute), \
                    patch.object(pipeline, '_convert_nrrd_to_nifti', side_effect=convert):
                ls = pipeline.ensure_native_dti_bundle(raw, 'ls')
                wls = pipeline.ensure_native_dti_bundle(raw, 'wls')
                self.assertNotEqual(ls.bundle_dir, wls.bundle_dir)
                pipeline.ensure_native_dti_bundle(raw, 'wls')
                self.assertEqual(len(commands), 2)
                self.assertEqual([cmd[4] for cmd in commands], ['LS', 'WLS'])
                wls.fa_nrrd_path.write_bytes(b'corrupt output')
                pipeline.ensure_native_dti_bundle(raw, 'wls')
                self.assertEqual(len(commands), 3)
                self.assertTrue(pipeline._native_outputs_complete(wls))

    def test_registration_separates_atlas_grid_and_resampling_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = inputs(Path(tmp))
            a, b = Path(tmp) / 'atlas1.nii.gz', Path(tmp) / 'atlas2.nii.gz'
            nii = pipeline._build_registered_bundle(raw, 'wls', a)
            nrrd_bundle = pipeline._build_registered_bundle(raw, 'wls', a, 'nrrd')
            other = pipeline._build_registered_bundle(raw, 'wls', b, 'nrrd')
            self.assertEqual(len({nii.bundle_dir, nrrd_bundle.bundle_dir, other.bundle_dir}), 3)
            self.assertEqual(nrrd_bundle.scalar_format, 'nrrd')

    def test_forcing_optional_models_does_not_rewrite_verified_dti_prerequisites(self):
        from types import SimpleNamespace
        from microstructure import pipeline as micro
        with patch.object(micro, '_dti_outputs_current', return_value=True), \
                patch.object(micro, '_copy_if_needed') as copy:
            micro._export_dti_scalar_maps(SimpleNamespace(force_recompute=True), {})
            copy.assert_not_called()

    def test_anatomy_source_receipt_rejects_changed_originals_and_preprocessing_switch(self):
        from shared.orchestrator import _write_anatomy_source, _anatomy_source_current
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = inputs(root)
            receipt = root / 'anatomy_source.json'
            sources = {key: pipeline._cache_file_signature(getattr(raw, key + '_path'))
                       for key in ('dwi', 'bval', 'bvec', 'mask')}
            _write_anatomy_source(receipt, raw, sources)
            current = {key: getattr(raw, key + '_path') for key in sources}
            with patch('shared.orchestrator._resolve_analysis_inputs', return_value=current):
                self.assertTrue(_anatomy_source_current(receipt, root, None))
                raw.bval_path.write_text('changed gradient values')
                self.assertFalse(_anatomy_source_current(receipt, root, None))
                with self.assertRaisesRegex(RuntimeError, 'changed after shared preparation'):
                    _write_anatomy_source(receipt, raw, sources)
            with patch('shared.orchestrator._resolve_analysis_inputs', return_value={
                    **current, 'dwi': root / 'different-preprocessed-dwi.nii.gz'}):
                self.assertFalse(_anatomy_source_current(receipt, root, None))

    def test_surface_facade_uses_shared_wls_but_not_the_parcellation_atlas(self):
        from anatomy.surface.pipeline import SurfaceConfig, _run_shared_raw
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()  # F4: macOS /var is a symlink to /private/var
            raw = inputs(root)
            atlas = root / 'trained-surface-atlas.nii.gz'
            bundle = pipeline._build_registered_bundle(raw, 'wls', atlas, 'nrrd')
            bundle.bundle_dir.mkdir(parents=True)
            for name in bundle.__dataclass_fields__:
                if name.endswith('_path'):
                    getattr(bundle, name).write_bytes(b'fixture')
            bundle.manifest_path.write_text(json.dumps({'native_bundle': {'dir': str(root / 'cache/native_dti/wls')}}))
            weights, templates = root / 'weights', root / 'templates'
            weights.mkdir()
            templates.mkdir()
            (weights / 'manifest.json').write_text('{}')
            checkpoints, shapes = {}, {}
            for hemi in ('left', 'right'):
                checkpoints[hemi] = {}
                for kind in ('white', 'pial'):
                    filename = f'{hemi}.{kind}.pt'
                    checkpoints[hemi][kind] = filename
                    (weights / filename).write_bytes(b'weights')
                shapes[hemi] = {'file': hemi + '.obj'}
                (templates / (hemi + '.obj')).write_bytes(b'template')
            resources = dict(registration_atlas=str(atlas), checkpoint_root=str(weights),
                template_dir=str(templates), model_config=dict(checkpoint_files=checkpoints, templates=shapes))
            config = SurfaceConfig('x', root, root / 'out', input_bundle=raw, resume=True)
            manifest = dict(status='running', subject_id='x')
            with patch.object(pipeline, 'ensure_registered_dti_bundle', return_value=bundle) as prepare, \
                    patch('anatomy.surface.pipeline._run_registered') as predict:
                _run_shared_raw(config, resources, config.output_dir, manifest, {}, None)
                self.assertEqual(prepare.call_args.args[1], atlas)
                self.assertEqual(prepare.call_args.kwargs['fit_method'], 'wls')
                self.assertEqual(prepare.call_args.kwargs['scalar_format'], 'nrrd')
                self.assertTrue(prepare.call_args.kwargs['reuse'])
                predict.assert_called_once_with(config, resources, config.output_dir, bundle.bundle_dir)
                native = config.output_dir / 'pred_results_SlicerDSpace/native/x'
                native.mkdir(parents=True)
                outputs = []
                for hemi in ('lh', 'rh'):
                    for kind in ('white', 'pial'):
                        token = 'wm' if kind == 'white' else 'pial'
                        outputs += [native / f'{hemi}.{kind}.obj', native / f'x_predicted_{token}_{hemi}.stl']
                for surface in outputs:
                    surface.write_text('surface fixture')
                previous = {**manifest, 'status': 'completed',
                            'output_sha256': {str(path): __import__('hashlib').sha256(path.read_bytes()).hexdigest()
                                              for path in outputs}}
                predict.reset_mock()
                _run_shared_raw(config, resources, config.output_dir, manifest, previous, None)
                predict.assert_not_called()
                surface.write_text('corrupted surface')
                _run_shared_raw(config, resources, config.output_dir, manifest, previous, None)
                predict.assert_called_once()

    def test_parcellation_tail_reuses_only_matching_inputs_settings_and_outputs(self):
        from anatomy.parcellation.pipeline import DDParcelRegisteredConfig, run_ddparcel_from_registered_bundle
        from shared.orchestrator import _parcellation_complete
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = inputs(root)
            bundle = pipeline._build_registered_bundle(raw, 'wls', root / 'atlas.nii.gz')
            bundle.bundle_dir.mkdir(parents=True)
            for name in bundle.__dataclass_fields__:
                if name.endswith('_path'):
                    getattr(bundle, name).write_bytes(b'input fixture')
            bundle.manifest_path.write_text(json.dumps({'native_bundle': {'manifest': {
                'source_files': {key: pipeline._cache_file_signature(getattr(raw, key + '_path'))
                                 for key in ('dwi', 'bval', 'bvec', 'mask')}}}}))
            weights = root / 'weights'
            weights.mkdir()
            (weights / 'model.pkl').write_bytes(b'weights fixture')
            out = root / 'out'
            config = DDParcelRegisteredConfig('x', bundle, out, raw.mask_path, weights, device='cpu')
            def normalize(source, mask, output, flip):
                output.write_bytes(source.read_bytes() + str(flip).encode())
            def predict(folder, subject, weights, log, **options):
                (folder / f'{subject}-DDSurfer-wmparc-Reg.mgz').write_bytes(b'prediction')
            def resample(command, log=None):
                Path(command[-1]).write_bytes(b'native prediction')
            with patch('anatomy.parcellation.pipeline._run_normalize', side_effect=normalize) as norm, \
                    patch('anatomy.parcellation.pipeline._run_ddsurfer_pred', side_effect=predict) as pred, \
                    patch('anatomy.parcellation.pipeline.resolve_cli_executable', return_value='resample'), \
                    patch('anatomy.parcellation.pipeline._run_cli', side_effect=resample), \
                    patch('anatomy.parcellation.qc.validate_outputs'):
                run_ddparcel_from_registered_bundle(config)
                run_ddparcel_from_registered_bundle(config)
                self.assertEqual(norm.call_count, 4)
                self.assertEqual(pred.call_count, 1)
                bundle.fa_reg_path.write_bytes(b'changed registered input')
                run_ddparcel_from_registered_bundle(config)
                self.assertEqual(pred.call_count, 2)
                (out / 'x-dti-MidEigenvalue-Reg-NormMasked.nii.gz').write_bytes(b'corrupt')
                run_ddparcel_from_registered_bundle(config)
                self.assertEqual(pred.call_count, 3)
                run_ddparcel_from_registered_bundle(replace(config, flip=0))
                self.assertEqual(pred.call_count, 4)
                run_ddparcel_from_registered_bundle(replace(config, force_recompute=True))
                self.assertEqual(pred.call_count, 5)
            outputs = {'wmparc': out / 'x-DDSurfer-wmparc.nii.gz'}
            with patch('shared.orchestrator._parcellation_outputs', return_value=outputs), \
                    patch('shared.orchestrator._resolve_analysis_inputs', return_value={
                        key: getattr(raw, key + '_path') for key in ('dwi', 'bval', 'bvec', 'mask')}), \
                    patch('shared.scalar_qc.scalar_qc_current', return_value=True):
                self.assertTrue(_parcellation_complete(root, 'x'))
                raw.bval_path.write_text('changed original gradients')
                self.assertFalse(_parcellation_complete(root, 'x'))

    @unittest.skipUnless(os.name == 'posix', 'real kernel file locks')
    def test_concurrent_cache_writer_cannot_enter_and_failure_releases_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / 'cache.lock'
            with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                with cache_lock(file):
                    with self.assertRaises(TimeoutError):
                        with cache_lock(file, timeout=0):
                            self.fail('A second writer entered')
                    raise RuntimeError('fixture failure')
            with cache_lock(file, timeout=0):
                pass


if __name__ == '__main__':
    unittest.main()
