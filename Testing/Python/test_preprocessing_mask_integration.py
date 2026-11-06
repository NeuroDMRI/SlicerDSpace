"""Selected-mask integration using small NIfTI images and external test tools."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing import DwiPreprocessingConfig, run_dwi_preprocessing
from shared.brain_mask import validate_brain_mask
from shared.validation import InputValidationConfig, validate_subject_inputs
from test_preprocessing_pipeline import _write_fake_backend, _write_fake_command, _write_nifti_subject


# The pnl_cnn backend runs [PythonSlicer, shared/pnl_mask_native.py, ...] (T2 A1/A7). The tests put a fake
# PythonSlicer in place of Slicer's interpreter; it ignores the module path and writes a fixed mask.
_FAKE_PYTHONSLICER: list[Path] = []
_RESOLVER = mock.patch("shared.brain_mask._pnl_native_python",
                       side_effect=lambda: _FAKE_PYTHONSLICER[-1] if _FAKE_PYTHONSLICER else None)


def setUpModule():
    _RESOLVER.start()


def tearDownModule():
    _RESOLVER.stop()


def _pnl_options(root: Path) -> dict:
    models = root / "pnl models"
    models.mkdir()
    (models / "weights.json").write_text("{}")
    nib.save(nib.Nifti1Image(np.ones((5, 5, 5), np.float32), np.eye(4)), models / "IITmean_b0_256.nii.gz")
    for view in ("sagittal", "coronal", "axial"):
        (models / f"{view}.pt").write_text("test weights")
    python = root / "slicer bin" / "PythonSlicer"
    python.parent.mkdir()
    _write_fake_command(
        python,
        "import json, pathlib, sys\n"
        "import nibabel as nib\n"
        "import numpy as np\n"
        "assert pathlib.Path(sys.argv[1]).name == 'pnl_mask_native.py'\n"
        "baseline = pathlib.Path(sys.argv[sys.argv.index('--input') + 1])\n"
        "image = nib.load(baseline)\n"
        "mask = np.zeros(image.shape, dtype=np.uint8)\n"
        "mask[1:4, 1:4, 1:4] = 1\n"
        "output = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
        "nib.save(nib.Nifti1Image(mask, image.affine), output)\n"
        "record = pathlib.Path(sys.argv[sys.argv.index('--record') + 1])\n"
        "record.write_text(json.dumps({'requested': 'cpu', 'resolved': 'cpu', 'route': 'native-sitk'}))\n"
        f"with pathlib.Path({str(root / 'pnl_calls.jsonl')!r}).open('a') as stream:\n"
        "    stream.write(json.dumps({'b0_mean': float(image.get_fdata().mean()), 'affine': image.affine.tolist()}) + '\\n')\n"
        "print('test PNL inference completed')\n",
    )
    _FAKE_PYTHONSLICER[:] = [python]
    return {
        "brain_mask_backend": "pnl_cnn",
        "brain_mask_device": "cpu",
        "pnl_mask_model_dir": models,
    }


def _motion_correction(image, gtab, **kwargs):
    affine = image.affine.copy()
    affine[0, 3] += 2.0
    corrected = nib.Nifti1Image(image.get_fdata(dtype=np.float32) + 4, affine)
    affines = np.repeat(np.eye(4)[..., None], image.shape[3], axis=2)
    return corrected, affines


def _motion_patch(backend: str):
    return mock.patch("dipy.align.motion_correction", side_effect=_motion_correction) if backend == "dipy" else nullcontext()


def _config(subject: Path, bin_dir: Path, backend: str, **options) -> DwiPreprocessingConfig:
    return DwiPreprocessingConfig(
        subject_id="100307", input_root=subject, backend=backend, execute=True,
        mrtrix_bin_dir=bin_dir, fsl_bin_dir=bin_dir,
        dipy_denoise_method="none", dipy_run_gibbs=False,
        **options,
    )


class PreprocessingMaskIntegrationTest(unittest.TestCase):
    def test_planning_and_skip_never_generate_a_missing_mask(self):
        for backend in ("dipy", "mrtrix_fsl"):
            for skip in (False, True):
                with self.subTest(backend=backend, skip=skip), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    subject = _write_nifti_subject(root)
                    canonical = subject / "input/nodif_brain_mask.nii.gz"
                    canonical.unlink()
                    bin_dir = root / "bin"
                    _write_fake_backend(bin_dir)
                    config = replace(_config(subject, bin_dir, backend), execute=False, skip=skip)
                    with mock.patch("shared.pipeline.generate_brain_mask") as raw_generator, \
                            mock.patch("preprocessing.pipeline.generate_brain_mask") as final_generator, \
                            mock.patch("preprocessing.pipeline._execute_preprocessing_backend") as executor:
                        result = run_dwi_preprocessing(config)
                    raw_generator.assert_not_called()
                    final_generator.assert_not_called()
                    executor.assert_not_called()
                    self.assertFalse(canonical.exists())
                    self.assertEqual(result.status, "skipped" if skip else "planned")
                    manifest = json.loads(result.manifest_path.read_text())
                    self.assertEqual(manifest["brain_mask_backend"]["backend"], "pnl_cnn")
                    self.assertFalse(manifest["brain_mask_backend"]["available"])

    def test_pnl_runs_once_on_corrected_dwi_for_both_routes_and_refreshes_only_mask(self):
        for backend in ("dipy", "mrtrix_fsl"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                subject = _write_nifti_subject(root)
                bin_dir = root / "bin"
                _write_fake_backend(bin_dir)
                canonical = subject / "input/nodif_brain_mask.nii.gz"
                original_mask = canonical.read_bytes()
                original_dwi = (subject / "input/data.nii.gz").read_bytes()
                config = _config(subject, bin_dir, backend, **_pnl_options(root))
                log = []
                with _motion_patch(backend):
                    result = run_dwi_preprocessing(config, log=log.append)
                self.assertTrue(any("test PNL inference completed" in line for line in log))
                calls = root / "pnl_calls.jsonl"
                call_records = [json.loads(line) for line in calls.read_text().splitlines()]
                self.assertEqual(len(call_records), 1)
                self.assertEqual(call_records[0]["b0_mean"], 5 if backend == "dipy" else 1)
                final_image = nib.load(result.final_dwi_path)
                np.testing.assert_allclose(call_records[0]["affine"], final_image.affine)
                qc = json.loads(result.qc_path.read_text())
                self.assertEqual(qc["brain_mask_backend"]["backend"], "pnl_cnn")
                self.assertTrue(qc["brain_mask_qc"]["geometry_pass"])
                self.assertEqual(qc["brain_mask_qc"]["brain_voxels"], 27)
                self.assertEqual(qc["mask_generation"]["source"], result.final_dwi_path.as_posix())
                signature = qc["mask_generation"]["signature"]
                self.assertEqual([record["path"] for record in signature["inputs"]], [
                    str(result.final_dwi_path.resolve()), str(result.final_bval_path.resolve()),
                    str(result.final_bvec_path.resolve()),
                ])
                manifest = json.loads(result.manifest_path.read_text())
                self.assertFalse(any(Path(record["argv"][0]).name == "dwi2mask" for record in manifest["commands"]))
                self.assertFalse(result.final_dwi_path.with_name("mask_calls.txt").exists())
                core_stat = result.final_dwi_path.stat().st_mtime_ns
                with mock.patch("preprocessing.pipeline._execute_preprocessing_backend", side_effect=AssertionError("core rerun")):
                    self.assertEqual(run_dwi_preprocessing(config).status, "cached")
                    self.assertEqual(len(calls.read_text().splitlines()), 1)
                    result.final_mask_path.unlink()
                    refreshed = run_dwi_preprocessing(config)
                self.assertEqual(refreshed.status, "cached")
                self.assertEqual(len(calls.read_text().splitlines()), 2)
                self.assertEqual(refreshed.final_dwi_path.stat().st_mtime_ns, core_stat)
                refreshed_qc = json.loads(refreshed.qc_path.read_text())
                self.assertEqual(refreshed_qc["mask_refresh_status"], "regenerated_from_corrected_dwi")
                self.assertEqual(canonical.read_bytes(), original_mask)
                self.assertEqual((subject / "input/data.nii.gz").read_bytes(), original_dwi)

    def test_missing_raw_mask_is_generated_only_in_final_space(self):
        for backend in ("dipy", "mrtrix_fsl"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                subject = _write_nifti_subject(root)
                canonical = subject / "input/nodif_brain_mask.nii.gz"
                canonical.unlink()
                bin_dir = root / "bin"
                _write_fake_backend(bin_dir)
                config = _config(subject, bin_dir, backend, **_pnl_options(root))
                with _motion_patch(backend):
                    result = run_dwi_preprocessing(config)
                self.assertFalse(canonical.exists())
                self.assertTrue(validate_brain_mask(result.final_dwi_path, result.final_mask_path)["geometry_pass"])
                self.assertEqual(len((root / "pnl_calls.jsonl").read_text().splitlines()), 1)

    def test_provided_mask_is_copied_after_motion_with_review_warning_and_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject = _write_nifti_subject(root)
            bin_dir = root / "bin"
            _write_fake_backend(bin_dir)
            canonical = subject / "input/nodif_brain_mask.nii.gz"
            original = canonical.read_bytes()
            config = _config(subject, bin_dir, "mrtrix_fsl", brain_mask_backend="provided")
            with mock.patch("preprocessing.pipeline.generate_brain_mask") as generator:
                result = run_dwi_preprocessing(config)
                self.assertEqual(run_dwi_preprocessing(config).status, "cached")
            generator.assert_not_called()
            self.assertFalse(result.final_mask_path.is_symlink())
            self.assertEqual(result.final_mask_path.read_bytes(), original)
            qc = json.loads(result.qc_path.read_text())
            self.assertTrue(qc["mask_generation"]["motion_applied"])
            self.assertTrue(qc["mask_generation"]["human_review_required"])
            self.assertTrue(any("human review" in warning and "motion correction" in warning for warning in qc["warnings"]))
            self.assertTrue(qc["brain_mask_qc"]["geometry_pass"])
            self.assertEqual(qc["mask_refresh_status"], "not_needed")
            result.final_mask_path.unlink()
            with mock.patch("preprocessing.pipeline._execute_preprocessing_backend", side_effect=AssertionError("core rerun")):
                self.assertEqual(run_dwi_preprocessing(config).status, "cached")
            self.assertEqual(canonical.read_bytes(), original)
            self.assertEqual(json.loads(result.qc_path.read_text())["mask_refresh_status"], "copied_from_input")

    def test_provided_mask_mismatching_corrected_geometry_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject = _write_nifti_subject(root)
            canonical = subject / "input/nodif_brain_mask.nii.gz"
            original = canonical.read_bytes()
            config = _config(subject, root, "dipy", brain_mask_backend="provided")
            with _motion_patch("dipy"), self.assertRaisesRegex(ValueError, "native 3D voxel grid"):
                run_dwi_preprocessing(config)
            self.assertEqual(canonical.read_bytes(), original)
            qc = json.loads((subject / "outputs/01_preproc/05_qc/100307_preproc_qc.json").read_text())
            self.assertEqual(qc["status"], "failed")

    def test_provided_missing_mask_fails_without_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject = _write_nifti_subject(root)
            canonical = subject / "input/nodif_brain_mask.nii.gz"
            canonical.unlink()
            config = _config(subject, root, "dipy", brain_mask_backend="provided", dipy_run_motion_correction=False)
            with mock.patch("preprocessing.pipeline.generate_brain_mask") as generator, \
                    self.assertRaisesRegex(FileNotFoundError, "provided.*supplied mask"):
                run_dwi_preprocessing(config)
            generator.assert_not_called()
            self.assertFalse(canonical.exists())

    def test_unavailable_selected_backend_never_falls_back_to_mrtrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject = _write_nifti_subject(root)
            bin_dir = root / "bin"
            _write_fake_backend(bin_dir)
            config = _config(subject, bin_dir, "mrtrix_fsl")
            with self.assertRaisesRegex(FileNotFoundError, "pnl_cnn unavailable"):
                run_dwi_preprocessing(config)
            self.assertFalse((subject / "outputs/01_preproc/mask_calls.txt").exists())
            qc = json.loads((subject / "outputs/01_preproc/05_qc/100307_preproc_qc.json").read_text())
            self.assertEqual(qc["status"], "failed")
            self.assertEqual(qc["brain_mask_backend"]["backend"], "pnl_cnn")

    def test_fast_dipy_preserves_existing_mask_without_mask_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject = _write_nifti_subject(root)
            config = _config(subject, root, "dipy", dipy_run_motion_correction=False)
            with mock.patch("preprocessing.pipeline.generate_brain_mask") as generator:
                result = run_dwi_preprocessing(config)
            generator.assert_not_called()
            qc = json.loads(result.qc_path.read_text())
            self.assertFalse(qc["brain_mask_backend"]["available"])
            self.assertEqual(qc["mask_generation"]["method"], "input_mask")
            self.assertEqual(qc["mask_generation"]["backend"], "provided")
            self.assertEqual(qc["mask_generation"]["selected_backend"], "pnl_cnn")
            self.assertTrue(qc["mask_generation"]["human_review_required"])
            self.assertTrue(qc["brain_mask_qc"]["geometry_pass"])

    def test_force_after_fast_links_does_not_overwrite_canonical_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject = _write_nifti_subject(root)
            canonical_paths = list((subject / "input").iterdir())
            originals = {path: path.read_bytes() for path in canonical_paths}
            config = _config(subject, root, "dipy", brain_mask_backend="provided", dipy_run_motion_correction=False)
            result = run_dwi_preprocessing(config)
            result.final_mask_path.unlink()
            result.final_mask_path.symlink_to(subject / "input/nodif_brain_mask.nii.gz")
            changed = replace(config, force=True, dipy_run_motion_correction=True, **_pnl_options(root))
            with _motion_patch("dipy"):
                result = run_dwi_preprocessing(changed)
            self.assertEqual(nib.load(result.final_dwi_path).affine[0, 3], 2)
            self.assertFalse(result.final_mask_path.is_symlink())
            for path, original in originals.items():
                self.assertEqual(path.read_bytes(), original, path.name)

    def test_mask_backend_tools_resources_and_options_invalidate_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject = _write_nifti_subject(root)
            bin_dir = root / "bin"
            _write_fake_backend(bin_dir)
            config = _config(subject, bin_dir, "mrtrix_fsl", **_pnl_options(root))
            result = run_dwi_preprocessing(config)
            initial_key = json.loads(result.manifest_path.read_text())["cache_key"]["brain_mask"]
            changes = (
                replace(config, b0_threshold=25),
                replace(config, eddy_threads=2),
                replace(config, brain_mask_backend="provided"),
            )
            for settings in changes:
                with self.subTest(settings=settings):
                    result = run_dwi_preprocessing(settings)
                    self.assertEqual(result.status, "completed")
                    self.assertNotEqual(json.loads(result.manifest_path.read_text())["cache_key"]["brain_mask"], initial_key)
            result = run_dwi_preprocessing(config)
            python = _FAKE_PYTHONSLICER[-1]
            python.write_text(python.read_text() + "\n# interpreter changed\n")
            self.assertEqual(run_dwi_preprocessing(config).status, "completed")
            (config.pnl_mask_model_dir / "axial.pt").write_text("different test weights")
            self.assertEqual(run_dwi_preprocessing(config).status, "completed")

    def test_cached_invalid_mask_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject = _write_nifti_subject(root)
            config = _config(subject, root, "dipy", brain_mask_backend="provided", dipy_run_motion_correction=False)
            result = run_dwi_preprocessing(config)
            nib.save(nib.Nifti1Image(np.zeros((5, 5, 5), np.uint8), np.eye(4)), result.final_mask_path)
            with self.assertRaisesRegex(ValueError, "binary and nonempty"):
                run_dwi_preprocessing(config)


class ValidationMaskIntegrationTest(unittest.TestCase):
    def test_validation_generates_selected_pnl_mask_with_selection_and_qc_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject = _write_nifti_subject(root)
            (subject / "input/nodif_brain_mask.nii.gz").unlink()
            result = validate_subject_inputs(InputValidationConfig(
                subject_id="100307", input_root=subject, dependency_groups=(), **_pnl_options(root),
            ))
            self.assertTrue(result.passed, result.messages)
            self.assertEqual(result.metrics["brain_mask_selection"]["backend"], "pnl_cnn")
            self.assertTrue(result.metrics["brain_mask_qc"]["geometry_pass"])
            self.assertEqual(result.metrics["brain_mask_qc"]["brain_voxels"], 27)
            self.assertEqual(len((root / "pnl_calls.jsonl").read_text().splitlines()), 1)

    def test_validation_selected_backend_failure_does_not_mention_other_backend(self):
        for backend in ("pnl_cnn", "provided"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                subject = _write_nifti_subject(root)
                (subject / "input/nodif_brain_mask.nii.gz").unlink()
                result = validate_subject_inputs(InputValidationConfig(
                    subject_id="100307", input_root=subject, dependency_groups=(),
                    brain_mask_backend=backend, write_report=False,
                ))
                self.assertFalse(result.passed)
                errors = [issue.message for issue in result.messages if "selected backend" in issue.message]
                self.assertEqual(len(errors), 1)
                self.assertIn(backend, errors[0])
                self.assertNotIn("MRtrix", errors[0])
                self.assertEqual(result.metrics["brain_mask_qc"]["status"], "not_run")

    def test_validation_retains_supplied_mask_without_requiring_selected_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject = _write_nifti_subject(root)
            mask = subject / "input/nodif_brain_mask.nii.gz"
            original = mask.read_bytes()
            with mock.patch("shared.pipeline.generate_brain_mask") as generator:
                result = validate_subject_inputs(InputValidationConfig(
                    subject_id="100307", input_root=subject, dependency_groups=(), write_report=False,
                ))
            generator.assert_not_called()
            self.assertTrue(result.passed, result.messages)
            self.assertFalse(result.metrics["brain_mask_backend"]["available"])
            self.assertTrue(result.metrics["brain_mask_qc"]["geometry_pass"])
            self.assertEqual(mask.read_bytes(), original)

    def test_validation_rejects_invalid_supplied_mask_geometry_and_content(self):
        invalid_masks = (
            (np.ones((4, 5, 5), np.uint8), np.eye(4)),
            (np.ones((5, 5, 5), np.uint8), np.diag([2, 2, 2, 1])),
            (np.zeros((5, 5, 5), np.uint8), np.eye(4)),
            (np.full((5, 5, 5), 0.5, np.float32), np.eye(4)),
            (np.full((5, 5, 5), np.nan, np.float32), np.eye(4)),
        )
        for data, affine in invalid_masks:
            with self.subTest(shape=data.shape, value=data.flat[0]), tempfile.TemporaryDirectory() as tmp:
                subject = _write_nifti_subject(Path(tmp))
                nib.save(nib.Nifti1Image(data, affine), subject / "input/nodif_brain_mask.nii.gz")
                result = validate_subject_inputs(InputValidationConfig(
                    subject_id="100307", input_root=subject, dependency_groups=(),
                    brain_mask_backend="provided", write_report=False,
                ))
                self.assertFalse(result.passed)
                self.assertFalse(result.metrics["brain_mask_qc"]["valid"])
                self.assertTrue(any("provided" in issue.message and issue.category == "brain_mask" for issue in result.messages))

    def test_module_logic_forwards_plain_brain_mask_settings(self):
        from preprocessing.module_logic import PreprocessingLogicMixin

        logic = PreprocessingLogicMixin()
        logic.log = mock.Mock()
        expected = mock.Mock()
        expected.final_dwi_path.is_file.return_value = False
        with mock.patch("preprocessing.module_logic.run_dwi_preprocessing", return_value=expected) as runner:
            actual = logic.runDwiPreprocessingPipeline(
                subject_id="100307", input_root=Path("/subject"), brain_mask_backend="synthstrip",
                pnl_mask_python=Path("/python"), pnl_mask_script=Path("/script"),
                pnl_mask_model_dir=Path("/models"), pnl_mask_ants_bin_dir=Path("/ants"),
                postprocess_freesurfer_home=Path("/freesurfer"),
            )
        self.assertIs(actual, expected)
        config = runner.call_args.args[0]
        self.assertEqual(config.brain_mask_backend, "synthstrip")
        self.assertEqual(config.pnl_mask_python, Path("/python"))
        self.assertEqual(config.pnl_mask_script, Path("/script"))
        self.assertEqual(config.pnl_mask_model_dir, Path("/models"))
        self.assertEqual(config.pnl_mask_ants_bin_dir, Path("/ants"))
        self.assertEqual(config.postprocess_freesurfer_home, Path("/freesurfer"))


if __name__ == "__main__":
    unittest.main()
