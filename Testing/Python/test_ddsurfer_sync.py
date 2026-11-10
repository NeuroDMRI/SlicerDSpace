"""Release, coordinate and cache contracts for the current DDSurfer backend."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import trimesh

from anatomy.surface.backend_bridge import BACKEND_ROOT, native_products, run_backend, runtime_info, sha256, upstream_record
from anatomy.surface.pipeline import SurfaceConfig, _raw_command, inspect_surface_resources, run_surface

ROOT = Path(__file__).resolve().parents[2]


class DDSurferSyncTests(unittest.TestCase):
    def test_identical_assets_are_not_overwritten_on_source_sync(self):
        from scripts.sync_ddsurfer_upstream import copy_if_changed
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / 'source', Path(directory) / 'target'
            source.write_bytes(b'unchanged checkpoint fixture')
            self.assertTrue(copy_if_changed(source, target))
            previous = target.stat().st_mtime_ns
            self.assertFalse(copy_if_changed(source, target))
            self.assertEqual(target.stat().st_mtime_ns, previous)
            source.write_bytes(b'changed checkpoint fixture')
            self.assertTrue(copy_if_changed(source, target))
            self.assertEqual(target.read_bytes(), source.read_bytes())

    def test_c4_sync_refuses_to_overwrite_recorded_local_patches(self):
        import scripts.sync_ddsurfer_upstream as sync
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = root / 'anatomy/surface/ddsurfer_backend/UPSTREAM.json'
            record.parent.mkdir(parents=True)
            record.write_text(json.dumps({'files': {}, 'local_patches': {'preprocessing/dti.sh': 'x',
                                                                       'run_ddsurfer_pipeline.sh': 'y'}}))
            with patch.object(sync, 'ROOT', root):
                with self.assertRaises(ValueError) as caught:
                    sync.synchronize(root / 'upstream', root / 'assets')
            self.assertIn('preprocessing/dti.sh', str(caught.exception))
            self.assertIn('run_ddsurfer_pipeline.sh', str(caught.exception))

    def test_c4_local_bash_patches_stay_recorded_and_guarded(self):
        record = json.loads((BACKEND_ROOT / 'UPSTREAM.json').read_text())
        # R5 (W4-W6): the native tensor fit also patches dti.py, preflight.py and the tests that model the old CLIs.
        self.assertEqual(set(record['local_patches']), {
            'postprocessing/core/archive.py', 'postprocessing/core/build.py',
            'postprocessing/core/config.py', 'postprocessing/core/runtime.py',
            'postprocessing/environment.py', 'postprocessing/pipeline.py',
            'preprocessing/dti.sh', 'run_ddsurfer_pipeline.sh', 'preprocessing/dti.py',
            'utils/preflight.py', 'utils/stages.py',
            'tests/check_environment.py', 'tests/test_environment_check.py', 'tests/test_inference.py',
            'tests/test_native_build.py', 'tests/test_pipeline_reliability.py', 'tests/test_preprocessing.py'})
        wrapper = (BACKEND_ROOT / 'run_ddsurfer_pipeline.sh').read_text()
        self.assertIn('${ARGS[@]+"${ARGS[@]}"}', wrapper)
        self.assertNotIn(' "${ARGS[@]}"', wrapper)

    def test_optional_training_loss_has_packaged_upstream_notice(self):
        from scripts.extension_runtime.runtime_resources import runtime_files
        self.assertEqual(sha256(BACKEND_ROOT / 'utils/LICENSE.pytorch3d'), sha256(ROOT / 'licenses/PyTorch3D.txt'))
        selected = set(runtime_files(ROOT))
        self.assertIn('anatomy/surface/ddsurfer_backend/utils/LICENSE.pytorch3d', selected)
        self.assertIn('anatomy/surface/ddsurfer_backend/CITATION.cff', selected)

    def test_surface_postprocess_dependencies_match_modern_and_fallback_backend(self):
        from shared.environment import EnvironmentDoctorConfig, build_dependency_specs, _is_freesurfer_home
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            (home / 'bin').mkdir()
            (home / 'subjects/fsaverage').mkdir(parents=True)
            (home / 'bin/mri_surf2surf').touch()
            self.assertTrue(_is_freesurfer_home(home))  # Surface-only installations need no recon-all.
            for modern in (False, True):
                if modern:
                    path = home / 'bin/mris_place_surface'
                    path.write_text('#!/bin/sh\n')
                    path.chmod(0o755)
                specs = build_dependency_specs(EnvironmentDoctorConfig(
                    targets=('surface_postprocess',), repo_root=ROOT, postprocess_freesurfer_home=home))
                commands = {spec.command for spec in specs if spec.kind == 'command'}
                self.assertFalse(commands & {'recon-all', 'mri_annotation2label', 'mri_segstats'})
                self.assertEqual(commands, {'mris_sphere', 'mris_curvature', 'mris_inflate', 'mris_register',
                                            'mri_label2label', 'mri_surf2surf',
                                            'mris_place_surface' if modern else 'mris_thickness'})

    def test_external_runtime_does_not_inherit_slicer_python_home(self):
        completed = SimpleNamespace(returncode=0, stdout='{"python": "external"}\n')
        with patch.dict(os.environ, {'PYTHONHOME': '/Slicer/lib/Python', 'PYTHONPATH': '/Slicer/lib',
                                    'ITK_AUTOLOAD_PATH': '/Slicer/ITKFactories'}):
            with patch('anatomy.surface.backend_bridge.subprocess.run', return_value=completed) as run:
                self.assertEqual(runtime_info('/independent/bin/python')['python'], 'external')
        environment = run.call_args.kwargs['env']
        self.assertNotIn('PYTHONHOME', environment)
        self.assertNotIn('PYTHONPATH', environment)
        self.assertNotIn('ITK_AUTOLOAD_PATH', environment)

    def test_vendored_sources_match_recorded_snapshot(self):
        record = upstream_record()
        self.assertEqual(len(record['commit']), 40)
        patched = record.get('local_patches', {})
        for filename, digest in record['files'].items():
            self.assertEqual(sha256(BACKEND_ROOT / filename), patched.get(filename, digest), filename)

    def test_snapshot_matches_asset_release(self):
        assets = json.loads((ROOT / 'models/assets_manifest.json').read_text())
        self.assertEqual(assets['ddsurfer_upstream_commit'], upstream_record()['commit'])
        self.assertEqual(len(assets['profiles']['surface']['required_paths']), 9)
        self.assertFalse(any('ckpts/hcp' in p for p in assets['profiles']['surface']['required_paths']))

    def test_prediction_compatibility_accepts_subject_selection(self):
        from anatomy.surface.prediction import PredictionStageConfig, SurfacePredictor, discover_subject_directories
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'x').mkdir()
            self.assertEqual(discover_subject_directories(root, ['x', 'missing']), {'x': root / 'x'})
            config = PredictionStageConfig('other', 'left', 'all', 'cpu', 0.02, 2, 3,
                root, root, root, root)
            with patch('anatomy.surface.prediction.run_backend') as run:
                result = SurfacePredictor(config).predict(root / 'x')
            self.assertEqual(result.subject_id, 'x')
            self.assertEqual(run.call_args.args[0][2], 'x')
            command = run.call_args.args[0]
            self.assertEqual(command[command.index('--precision') + 1], 'fp32')
            self.assertEqual(command[command.index('--cpu-threads') + 1], '4')
            self.assertEqual(command[command.index('--max-gpu-gib') + 1], '24.0')

    def test_resource_budgets_are_validated_before_writing(self):
        for options in (dict(cpu_threads=0), dict(mask_threads=0), dict(max_gpu_gib=0),
                        dict(max_gpu_gib=float('nan')), dict(max_gpu_gib=float('inf'))):
            with self.subTest(options=options), self.assertRaises(ValueError):
                run_surface(SurfaceConfig('x', ROOT, ROOT, **options))

    def test_invalid_surface_settings_fail_before_planning(self):
        from shared.orchestrator import ProductOrchestratorConfig, plan_subject_products
        for options in (dict(surface_cpu_threads=0), dict(surface_max_gpu_gib=float('nan')),
                        dict(surface_max_gpu_gib=-1), dict(surface_precision='fp16')):
            with self.subTest(options=options), self.assertRaises(ValueError):
                plan_subject_products(ProductOrchestratorConfig(subject_id='x', subject_root=ROOT,
                    targets=('surface',), **options), write=False)

    def test_cancelled_surface_has_terminal_manifest_and_qc(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            resources = dict(available=True, upstream_commit=upstream_record()['commit'])
            with patch('anatomy.surface.pipeline.inspect_surface_resources', return_value=resources), \
                    patch('anatomy.surface.pipeline.runtime_info', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    run_surface(SurfaceConfig('x', root, root, execute=True, device='cpu'))
            self.assertEqual(json.loads((root / 'surface_manifest.json').read_text())['status'], 'cancelled')
            self.assertEqual(json.loads((root / 'surface_qc.json').read_text())['status'], 'cancelled')

    @unittest.skipUnless(os.name == 'posix', 'POSIX stage-group cleanup')
    def test_cancel_reaps_upstream_stage_in_its_separate_process_group(self):
        with tempfile.TemporaryDirectory() as temp:
            pid_file = Path(temp) / 'stage.pid'
            fixture = ROOT / 'Testing/Python/fixtures/ddsurfer_cancel_backend.py'

            def cancel_on_ready(line):
                if line == 'STAGE_READY':
                    raise KeyboardInterrupt

            with self.assertRaises(KeyboardInterrupt):
                run_backend([fixture, BACKEND_ROOT, pid_file], log=cancel_on_ready)
            pid = int(pid_file.read_text())
            # PythonSlicer is a launcher; its child may still be reaping the
            # interrupted stage when the launcher itself exits.
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.05)
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

    def test_failure_always_writes_qc_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            config = SurfaceConfig('x', output, output, checkpoint_root=output / 'missing', execute=True)
            with self.assertRaises(RuntimeError):
                run_surface(config)
            self.assertEqual(json.loads((output / 'surface_manifest.json').read_text())['status'], 'failed')
            qc = json.loads((output / 'surface_qc.json').read_text())
            self.assertFalse(qc['automatic_qc_passed'])

    def test_subject_path_and_precision_are_validated(self):
        for subject, precision in (('../escape', 'auto'), ('x', 'fp16')):
            with self.assertRaises(ValueError):
                run_surface(SurfaceConfig(subject, ROOT, ROOT, precision=precision))

    def test_raw_command_preserves_selected_input_and_assets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = SimpleNamespace(**{name: root / name for name in ('dwi_path', 'bval_path', 'bvec_path', 'mask_path')})
            for path in vars(bundle).values():
                path.touch()
            resources = dict(checkpoint_root='/weights', template_dir='/templates', registration_atlas='/atlas')
            command, env = _raw_command(SurfaceConfig('x', root, root, input_bundle=bundle,
                device='cuda:1', precision='bf16', slicer_home=root, resume=True,
                cpu_threads=2, max_gpu_gib=18.), resources, root)
            self.assertIn('--keep-cache', command)
            self.assertEqual(command[command.index('--dwi') + 1], bundle.dwi_path)
            self.assertEqual(command[command.index('--precision') + 1], 'bf16')
            self.assertEqual(env['DTI_REFERENCE_IMAGE'], '/atlas')
            # N9: the backend learns where shared/teem_dti.py is, even when deployed away from this repository.
            self.assertTrue(Path(env['SLICERDSPACE_TEEM_DTI_SCRIPT']).is_file())
            self.assertEqual(Path(env['SLICERDSPACE_TEEM_DTI_SCRIPT']), ROOT / 'shared/teem_dti.py')
            self.assertIn('--resume', command)
            self.assertEqual(command[command.index('--cpu-threads') + 1], '2')
            self.assertEqual(command[command.index('--max-gpu-gib') + 1], '18.0')

    def test_t4_a_preset_teem_script_is_not_overwritten(self):
        from anatomy.surface.backend_bridge import backend_environment
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = SimpleNamespace(**{name: root / name for name in ('dwi_path', 'bval_path', 'bvec_path', 'mask_path')})
            for path in vars(bundle).values():
                path.touch()
            resources = dict(checkpoint_root='/weights', template_dir='/templates', registration_atlas='/atlas')
            config = SurfaceConfig('x', root, root, input_bundle=bundle, slicer_home=root)
            with patch.dict(os.environ, {'SLICERDSPACE_TEEM_DTI_SCRIPT': '/user/teem_dti.py'}):
                _, env = _raw_command(config, resources, root)
                self.assertEqual(backend_environment(env)['SLICERDSPACE_TEEM_DTI_SCRIPT'], '/user/teem_dti.py')
            with patch.dict(os.environ, clear=False):
                os.environ.pop('SLICERDSPACE_TEEM_DTI_SCRIPT', None)
                _, env = _raw_command(config, resources, root)
                self.assertEqual(Path(env['SLICERDSPACE_TEEM_DTI_SCRIPT']), ROOT / 'shared/teem_dti.py')

    @unittest.skipUnless(os.name == 'posix', 'POSIX bash script test')
    def test_t3_dti_sh_resolves_runtime_root_and_launcher_for_every_slicer_layout(self):
        script = ROOT / 'anatomy/surface/ddsurfer_backend/preprocessing/dti.sh'

        def resolve(path):
            run = subprocess.run(['bash', str(script), '--subject', 'x', '--slicer-path', str(path)],
                                 env={**os.environ, 'DTI_RESOLVE_SLICER_ONLY': '1'},
                                 capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            values = dict(line.split('=', 1) for line in run.stdout.splitlines() if '=' in line)
            return Path(values['root']).resolve(), Path(values['launcher']).resolve()

        def touch(path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            path.chmod(0o755)

        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            linux = base / 'linux'
            touch(linux / 'Slicer')
            touch(linux / 'bin/PythonSlicer')
            (linux / 'lib').mkdir()
            self.assertEqual(resolve(linux), (linux.resolve(), (linux / 'Slicer').resolve()))
            windows = base / 'windows'
            touch(windows / 'Slicer.exe')
            touch(windows / 'bin/PythonSlicer.exe')
            (windows / 'lib').mkdir()
            self.assertEqual(resolve(windows), (windows.resolve(), (windows / 'Slicer.exe').resolve()))
            app = base / 'Slicer.app'
            touch(app / 'Contents/MacOS/Slicer')
            touch(app / 'Contents/bin/PythonSlicer')
            (app / 'Contents/lib').mkdir()
            expected = ((app / 'Contents').resolve(), (app / 'Contents/MacOS/Slicer').resolve())
            for given in (app, app / 'Contents', app / 'Contents/MacOS'):
                with self.subTest(given=str(given.relative_to(base))):
                    self.assertEqual(resolve(given), expected)
            (base / 'empty').mkdir()
            missing = subprocess.run(['bash', str(script), '--subject', 'x', '--slicer-path', str(base / 'empty')],
                                     env={**os.environ, 'DTI_RESOLVE_SLICER_ONLY': '1'},
                                     capture_output=True, text=True, timeout=60)
            self.assertNotEqual(missing.returncode, 0)

    def test_missing_mask_requires_explicit_synthstrip_opt_in(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = SimpleNamespace(dwi_path=root / 'dwi', bval_path=root / 'bval',
                                     bvec_path=root / 'bvec', mask_path=None)
            for key in ('dwi_path', 'bval_path', 'bvec_path'):
                getattr(bundle, key).touch()
            resources = dict(checkpoint_root='/weights', template_dir='/templates', registration_atlas='/atlas')
            with self.assertRaisesRegex(FileNotFoundError, 'mask'):
                _raw_command(SurfaceConfig('x', root, root, input_bundle=bundle), resources, root)
            command, _ = _raw_command(SurfaceConfig('x', root, root, input_bundle=bundle,
                auto_mask=True, freesurfer_home=root, slicer_home=root), resources, root)
            self.assertIn('--auto-mask', command)
            self.assertNotIn('--mask', command)
            bundle.mask_path = root / 'mask'
            bundle.mask_path.touch()
            command, _ = _raw_command(SurfaceConfig('x', root, root, input_bundle=bundle,
                auto_mask=True, slicer_home=root), resources, root)
            self.assertEqual(command[command.index('--mask') + 1], bundle.mask_path)

    def test_raw_resume_cannot_bypass_upstream_receipt_validation(self):
        from shared.io_layout import subject_layout
        from shared.orchestrator import _surface_complete
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'x'
            surface = subject_layout(root).ddsurfer
            surface.mkdir(parents=True)
            (surface / 'surface_manifest.json').write_text(json.dumps(dict(version=3,
                status='completed', input_mode='raw', upstream_commit=upstream_record()['commit'])))
            (surface / 'surface_qc.json').write_text(json.dumps(dict(automatic_qc_passed=True)))
            self.assertFalse(_surface_complete(root, 'x'))

    def test_registered_resume_rejects_precision_input_and_output_changes(self):
        from shared.io_layout import subject_layout
        from shared.orchestrator import ProductOrchestratorConfig, _surface_complete
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'x'
            surface = subject_layout(root).ddsurfer
            native = surface / 'pred_results_SlicerDSpace/native/x'
            native.mkdir(parents=True)
            for hemi in ('lh', 'rh'):
                for kind in ('white', 'pial'):
                    trimesh.creation.icosphere(subdivisions=1).export(str(native / f'{hemi}.{kind}.obj'))
            products, checks = native_products(native, 'x', 'all')
            inputs = []
            for index in range(15):
                path = root / f'input-{index}'
                path.write_text('registered-input')
                inputs.append(path)
            outputs = [*products, *(Path(c['indexed_master']) for c in checks.values())]
            manifest = dict(version=3, status='completed', input_mode='registered', precision='fp32',
                subject_id='x', subject_root=str(root.resolve()),
                cpu_threads=4, max_gpu_gib=24., upstream_commit=upstream_record()['commit'],
                input_sha256={str(p): sha256(p) for p in inputs}, output_sha256={str(p): sha256(p) for p in outputs})
            (surface / 'surface_manifest.json').write_text(json.dumps(manifest))
            (surface / 'surface_qc.json').write_text(json.dumps(dict(automatic_qc_passed=True)))
            self.assertTrue(_surface_complete(root, 'x'))
            self.assertFalse(_surface_complete(root, 'x', config=ProductOrchestratorConfig(
                subject_id='x', subject_root=root, targets=('surface',), surface_precision='bf16')))
            inputs[0].write_text('changed-input')
            self.assertFalse(_surface_complete(root, 'x'))
            inputs[0].write_text('registered-input')
            outputs[0].write_bytes(b'broken-mesh')
            self.assertFalse(_surface_complete(root, 'x'))

    def test_native_obj_and_lps_stl_have_consistent_coordinates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mesh = trimesh.creation.icosphere(subdivisions=1)
            mesh.vertices += [10, 20, 30]
            for hemi in ('lh', 'rh'):
                for kind in ('white', 'pial'):
                    mesh.export(str(root / f'{hemi}.{kind}.obj'))
            products, qc = native_products(root, 'x', 'all')
            self.assertEqual(len(products), 4)
            self.assertTrue(all(v['watertight'] and v['euler_number'] == 2 for v in qc.values()))
            lps = trimesh.load(str(products[0]), process=False)
            np.testing.assert_allclose(lps.bounds, [[-11, -21, 29], [-9, -19, 31]])
            ras = trimesh.load(str(root / 'x_predicted_wm_lh_orig.stl'), process=False)
            np.testing.assert_allclose(ras.bounds, mesh.bounds)

    def test_white_only_does_not_require_pial(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for hemi in ('lh', 'rh'):
                trimesh.creation.icosphere(subdivisions=1).export(str(root / f'{hemi}.white.obj'))
            products, _ = native_products(root, 'x', 'wm')
            self.assertEqual(len(products), 2)

    def test_topology_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for kind, subdivisions in (('white', 1), ('pial', 2)):
                trimesh.creation.icosphere(subdivisions=subdivisions).export(str(root / f'lh.{kind}.obj'))
            with self.assertRaisesRegex(ValueError, 'correspondence'):
                native_products(root, 'x', 'all')

    def test_postprocess_cache_rejects_legacy_and_modified_surfaces(self):
        from shared.io_layout import subject_layout
        from shared.orchestrator import _surface_postprocess_complete
        from anatomy.surface.postprocess_release import DDSurferPostprocessConfig, run_ddsurfer_postprocess
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'x'
            layout = subject_layout(root)
            native = layout.ddsurfer / 'pred_results_SlicerDSpace/native/x'
            native.mkdir(parents=True)
            for hemi in ('lh', 'rh'):
                for kind in ('white', 'pial'):
                    trimesh.creation.icosphere(subdivisions=1).export(str(native / f'{hemi}.{kind}.obj'))

            def fake_backend(command, **kwargs):
                from Testing.Python.test_surface_postprocess_qc import write_postprocess_fixture
                output = Path(command[command.index('--output-root') + 1]) / 'x'
                write_postprocess_fixture(output)

            config = DDSurferPostprocessConfig('x', layout.outputs, subject_root=root, native_dir=native)
            with patch('anatomy.surface.backend_bridge.run_backend', side_effect=fake_backend), \
                    patch('shared.surface_runtime.resolve_surface_runtime', return_value=('native', root / 'runtime')):
                result = run_ddsurfer_postprocess(config)
                original = json.loads(result.manifest_path.read_text())
                self.assertFalse(original['requires_ddparcel'])
                self.assertTrue(_surface_postprocess_complete(root, 'x'))
                self.assertFalse(_surface_postprocess_complete(root, 'x', mode='left'))
                self.assertFalse(_surface_postprocess_complete(root, 'x', atlases=('another_atlas',)))
                self.assertFalse(_surface_postprocess_complete(root, 'x', native_dir=root / 'another_run'))
                legacy = dict(original, upstream_commit='old-version')
                result.manifest_path.write_text(json.dumps(legacy))
                self.assertFalse(_surface_postprocess_complete(root, 'x'))
                result.manifest_path.write_text(json.dumps(original))
                with (native / 'lh.white.obj').open('a') as stream:
                    stream.write('\n# changed surface\n')
                self.assertFalse(_surface_postprocess_complete(root, 'x'))
                rerun = run_ddsurfer_postprocess(config)
                rerun_key = json.loads(rerun.manifest_path.read_text())['cache_key']
                self.assertNotEqual(rerun_key, original['cache_key'])
                self.assertTrue(_surface_postprocess_complete(root, 'x'))
                forced = run_ddsurfer_postprocess(DDSurferPostprocessConfig('x', layout.outputs,
                    subject_root=root, native_dir=native, force=True))
                self.assertNotEqual(json.loads(forced.manifest_path.read_text())['cache_key'],
                                    rerun_key)
                (forced.structural_root / 'surf/lh.thickness').write_bytes(b'corrupt morphometry')
                self.assertFalse(_surface_postprocess_complete(root, 'x'))

                mri = root / 'reference'
                mri.mkdir()
                import nibabel as nib
                nib.save(nib.MGHImage(np.zeros((3, 3, 3), dtype=np.float32), np.eye(4)), mri / 'brain.mgz')
                run_ddsurfer_postprocess(DDSurferPostprocessConfig('x', layout.outputs,
                    subject_root=root, native_dir=native, mri_dir=mri))
                self.assertTrue(_surface_postprocess_complete(root, 'x'))
                nib.save(nib.MGHImage(np.ones((3, 3, 3), dtype=np.float32), np.eye(4)), mri / 'brain.mgz')
                self.assertFalse(_surface_postprocess_complete(root, 'x'))


if __name__ == '__main__':
    unittest.main()
