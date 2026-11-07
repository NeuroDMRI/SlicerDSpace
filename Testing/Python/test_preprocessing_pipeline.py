#!/usr/bin/env python3
"""Tests for optional DWI preprocessing wrapper."""

from __future__ import annotations

import json
import stat
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _write_subject(root: Path, subject_id: str = "100307") -> Path:
    return _write_nifti_subject(root, subject_id)


def _write_nifti_subject(root: Path, subject_id: str = "100307") -> Path:
    import nibabel as nib
    import numpy as np

    subject = root / subject_id
    input_dir = subject / "input"
    input_dir.mkdir(parents=True)
    data = np.zeros((5, 5, 5, 3), dtype=np.float32)
    data[..., 0] = 1.0
    data[2, 2, 2, 1] = 2.0
    data[2, 2, 3, 2] = 3.0
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[1:4, 1:4, 1:4] = 1
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(input_dir / "data.nii.gz"))
    nib.save(nib.Nifti1Image(mask, np.eye(4)), str(input_dir / "nodif_brain_mask.nii.gz"))
    (input_dir / "bvals").write_text("0 1000 1000\n", encoding="utf-8")
    (input_dir / "bvecs").write_text("0 1 0\n0 0 1\n0 0 0\n", encoding="utf-8")
    return subject


def _write_fake_command(path: Path, body: str) -> None:
    path.write_text(f"#!{sys.executable}\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _write_fake_backend(bin_dir: Path) -> None:
    bin_dir.mkdir(parents=True)
    _write_fake_command(
        bin_dir / "dwidenoise",
        """
import pathlib, shutil, sys
if pathlib.Path(sys.argv[2]).exists() and '-force' not in sys.argv:
    raise SystemExit('existing denoised output needs -force')
shutil.copy2(sys.argv[1], sys.argv[2])
if '-noise' in sys.argv:
    pathlib.Path(sys.argv[sys.argv.index('-noise') + 1]).write_text('noise')
""",
    )
    _write_fake_command(
        bin_dir / "mrdegibbs",
        """
import shutil, sys
shutil.copy2(sys.argv[1], sys.argv[2])
""",
    )
    _write_fake_command(
        bin_dir / "dwifslpreproc",
        """
import os, pathlib, shutil, sys
shutil.copy2(sys.argv[1], sys.argv[2])
pathlib.Path(sys.argv[2]).with_name('fsl_environment.txt').write_text(
    os.environ.get('FSLDIR', '') + '\\n' + os.environ.get('FSLOUTPUTTYPE', '') + '\\n'
)
scratch = pathlib.Path(sys.argv[sys.argv.index('-scratch') + 1])
scratch.mkdir(parents=True, exist_ok=True)
prefix = scratch / 'dwi_post_eddy'
for suffix in (
    '.eddy_parameters', '.eddy_movement_rms', '.eddy_outlier_map',
    '.eddy_outlier_report', '.eddy_cnr_maps.nii.gz', '.eddy_residuals.nii.gz',
):
    pathlib.Path(str(prefix) + suffix).write_text('eddy artifact: ' + suffix)
if '-export_grad_fsl' in sys.argv:
    idx = sys.argv.index('-export_grad_fsl')
    pathlib.Path(sys.argv[idx + 1]).write_text('0 1 0\\n0 0 1\\n0 0 0\\n')
    pathlib.Path(sys.argv[idx + 2]).write_text('0 1000 1000\\n')
if '-scratch' in sys.argv:
    scratch = pathlib.Path(sys.argv[sys.argv.index('-scratch') + 1])
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / 'dwi_post_eddy.eddy_parameters').write_text('0 0 0 0 0 0\\n')
    (scratch / 'dwi_post_eddy.eddy_movement_rms').write_text('0 0\\n')
    (scratch / 'dwi_post_eddy.eddy_outlier_map').write_text('0 0 0\\n')
    (scratch / 'field_fieldcoef.nii.gz').write_text('field')
    (scratch / 'field_map.nii.gz').write_text('field-map')
    (scratch / 'field_movpar.txt').write_text('0 0 0\\n')
    (scratch / 'topup_in.topup_log').write_text('topup complete\\n')
""",
    )
    _write_fake_command(
        bin_dir / "dwi2mask",
        """
import pathlib, sys
import nibabel as nib
import numpy as np
image = nib.load(sys.argv[1])
mask = np.zeros(image.shape[:3], dtype=np.uint8)
mask[1:4, 1:4, 1:4] = 1
nib.save(nib.Nifti1Image(mask, image.affine), sys.argv[2])
with pathlib.Path(sys.argv[1]).with_name('mask_calls.txt').open('a') as stream:
    stream.write(sys.argv[1] + '\\n')
""",
    )
    _write_fake_command(
        bin_dir / "mrconvert",
        """
import pathlib, sys
pathlib.Path(sys.argv[2]).write_text('converted')
""",
    )
    _write_fake_command(
        bin_dir / "dwiextract",
        """
import pathlib, sys
pathlib.Path(sys.argv[2]).write_text('forward-b0s')
""",
    )
    _write_fake_command(
        bin_dir / "mrcat",
        """
import pathlib, sys
pathlib.Path(sys.argv[3]).write_text('se-epi-pair')
""",
    )


def _fake_pnl_mask(dwi, bval, bvec, output, *, config, force=False, log=None):
    """W5 (D1/D2): the final mask comes from PNL CNN. This fixture writes the mask a fake PNL CNN would write."""
    import nibabel as nib
    import numpy as np

    assert config.backend == "pnl_cnn", config.backend
    image = nib.load(str(dwi))
    mask = np.zeros(image.shape[:3], dtype=np.uint8)
    mask[1:4, 1:4, 1:4] = 1
    nib.save(nib.Nifti1Image(mask, image.affine), str(output))
    with Path(dwi).with_name("mask_calls.txt").open("a") as stream:
        stream.write(str(dwi) + "\n")
    return {"status": "completed"}


class DwiPreprocessingPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        from unittest.mock import patch

        patcher = patch("preprocessing.pipeline.generate_brain_mask", side_effect=_fake_pnl_mask)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_motion_cache_requires_current_gradient_rotation_policy(self):
        from preprocessing.pipeline import DIPY_BVEC_ROTATION_POLICY, preprocessing_gradients_current
        manifest = {"cache_key": {"backend": "dipy", "parameters": {"dipy_run_motion_correction": True}}}
        self.assertFalse(preprocessing_gradients_current(manifest))
        manifest["cache_key"]["parameters"]["dipy_bvec_rotation_policy"] = DIPY_BVEC_ROTATION_POLICY
        self.assertTrue(preprocessing_gradients_current(manifest))
        manifest["cache_key"]["parameters"] = {"dipy_run_motion_correction": False}
        self.assertTrue(preprocessing_gradients_current(manifest))
        manifest["cache_key"]["backend"] = "mrtrix_fsl"
        self.assertTrue(preprocessing_gradients_current(manifest))

    def test_rotate_bvecs_accepts_n_by_4_by_4_affines(self) -> None:
        import numpy as np

        from preprocessing.pipeline import _rotate_bvecs_best_effort

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.bvec"
            output = root / "rotated.bvec"
            source.write_text("1 0 0\n0 1 0\n0 0 1\n", encoding="utf-8")
            rotation = np.array(
                [
                    [0.0, -1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            affines = np.repeat(np.eye(4)[None, ...], 3, axis=0)
            affines[1] = rotation

            status = _rotate_bvecs_best_effort(source, output, affines)

            rotated = np.loadtxt(output)
            self.assertEqual(status["status"], "rotated")
            self.assertEqual(status["affine_layout"], "Nx4x4")
            np.testing.assert_allclose(rotated[:, 0], [1.0, 0.0, 0.0], atol=1e-8)
            np.testing.assert_allclose(rotated[:, 1], [1.0, 0.0, 0.0], atol=1e-8)

    def test_rotate_bvecs_accepts_dipy_4_by_4_by_n_affines(self) -> None:
        import numpy as np

        from preprocessing.pipeline import _rotate_bvecs_best_effort

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.bvec"
            output = root / "rotated.bvec"
            source.write_text("1 0 0\n0 1 0\n0 0 1\n", encoding="utf-8")
            rotation = np.array(
                [
                    [0.0, -1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            affines = np.repeat(np.eye(4)[None, ...], 3, axis=0)
            affines[1] = rotation

            status = _rotate_bvecs_best_effort(source, output, np.moveaxis(affines, 0, -1))

            rotated = np.loadtxt(output)
            self.assertEqual(status["status"], "rotated")
            self.assertEqual(status["affine_layout"], "4x4xN")
            np.testing.assert_allclose(rotated[:, 0], [1.0, 0.0, 0.0], atol=1e-8)
            np.testing.assert_allclose(rotated[:, 1], [1.0, 0.0, 0.0], atol=1e-8)

    def test_gradient_reorientation_matches_dipy_with_rotation_and_shear(self):
        import numpy as np
        from dipy.core.gradients import gradient_table, reorient_bvecs
        from scipy.spatial.transform import Rotation
        from preprocessing.pipeline import _rotate_bvecs_best_effort

        rng = np.random.default_rng(42)
        vectors = rng.normal(size=(7, 3))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors[0] = 0
        bvals = np.array([0] + [1000] * 6)
        affines = np.repeat(np.eye(4)[..., None], len(vectors), axis=-1)
        for index in range(1, len(vectors)):
            stretch = np.array([[1.1, .08, 0], [.08, .9, .04], [0, .04, 1.03]])
            affines[:3, :3, index] = Rotation.from_euler("xyz", [index * 7, -11, 18], degrees=True).as_matrix() @ stretch
        reference = reorient_bvecs(gradient_table(bvals, bvecs=vectors.copy()), affines[..., 1:]).bvecs
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "source.bvec", Path(tmp) / "rotated.bvec"
            np.savetxt(source, vectors.T)
            for layout in (affines, np.moveaxis(affines, -1, 0)):
                result = _rotate_bvecs_best_effort(source, output, layout)
                np.testing.assert_allclose(np.loadtxt(output).T, reference, atol=1e-9, rtol=1e-9)
                self.assertEqual(result["status"], "rotated")

    def test_invalid_motion_affines_do_not_silently_copy_gradients(self):
        import numpy as np
        from preprocessing.pipeline import _rotate_bvecs_best_effort

        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "source.bvec", Path(tmp) / "rotated.bvec"
            source.write_text("1 0\n0 1\n0 0\n")
            for affines in (np.eye(4)[..., None], np.full((4, 4, 2), np.nan), np.zeros((4, 4, 2))):
                with self.assertRaises(ValueError):
                    _rotate_bvecs_best_effort(source, output, affines)
                self.assertFalse(output.exists())

    def test_skip_writes_optional_skipped_manifest(self) -> None:
        from preprocessing import DwiPreprocessingConfig, run_dwi_preprocessing

        with tempfile.TemporaryDirectory() as tmp:
            subject = _write_subject(Path(tmp))
            result = run_dwi_preprocessing(
                DwiPreprocessingConfig(subject_id="100307", input_root=subject, skip=True)
            )

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(result.status, "skipped")
            self.assertTrue(manifest["optional"])
            self.assertTrue(manifest["skip"])

    def test_dry_run_writes_plan_without_requiring_commands(self) -> None:
        from preprocessing import DwiPreprocessingConfig, run_dwi_preprocessing

        with tempfile.TemporaryDirectory() as tmp:
            subject = _write_subject(Path(tmp))
            result = run_dwi_preprocessing(
                DwiPreprocessingConfig(subject_id="100307", input_root=subject, execute=False)
            )

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(result.status, "planned")
            self.assertEqual(manifest["status"], "planned")
            self.assertEqual(manifest["backend"]["backend"], "dipy")
            self.assertIn("motion_correction", " ".join(manifest["commands"][2]["argv"]))

    def test_execute_with_fake_backend_generates_final_outputs(self) -> None:
        from preprocessing import DwiPreprocessingConfig, inspect_preprocessing_backend, run_dwi_preprocessing

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            subject = _write_subject(tmp_path)
            bin_dir = tmp_path / "bin"
            _write_fake_backend(bin_dir)

            config = DwiPreprocessingConfig(
                subject_id="100307",
                input_root=subject,
                backend="mrtrix_fsl",
                execute=True,
                mrtrix_bin_dir=bin_dir,
                fsl_bin_dir=bin_dir,
            )
            backend = inspect_preprocessing_backend(config)
            self.assertTrue(backend["available"])

            result = run_dwi_preprocessing(config)

            self.assertEqual(result.status, "completed")
            self.assertTrue(result.final_dwi_path.is_file())
            self.assertTrue(result.final_bval_path.is_file())
            self.assertTrue(result.final_bvec_path.is_file())
            self.assertTrue(result.final_mask_path.is_file())
            fsl_environment = (result.final_dwi_path.parent / "fsl_environment.txt").read_text(
                encoding="utf-8"
            ).splitlines()
            self.assertEqual(fsl_environment, [bin_dir.parent.resolve().as_posix(), "NIFTI_GZ"])
            qc = json.loads(result.qc_path.read_text(encoding="utf-8"))
            self.assertEqual(backend["capabilities"]["denoise"], "mppca")
            self.assertFalse(backend["capabilities"]["susceptibility_distortion_correction"])
            self.assertFalse(any("Denoising" in warning for warning in qc["warnings"]))
            self.assertIn("Susceptibility-distortion correction was not applied.", qc["warnings"])
            self.assertIn("Bias-field correction was not applied by the selected backend.", qc["warnings"])
            self.assertEqual(qc["backend_details"]["denoise"]["status"], "completed")
            expected_suffixes = {
                "eddy_parameters": "_eddy_parameters.txt",
                "eddy_movement_rms": "_eddy_movement_rms.txt",
                "eddy_outlier_map": "_eddy_outlier_map.txt",
                "eddy_outlier_report": "_eddy_outlier_report.txt",
                "eddy_cnr_maps": "_eddy_cnr_maps.nii.gz",
                "eddy_residuals": "_eddy_residuals.nii.gz",
            }
            for key, suffix in expected_suffixes.items():
                self.assertTrue(qc["files"][key]["exists"], key)
                self.assertTrue(qc["files"][key]["path"].endswith(suffix), key)
                self.assertEqual(
                    qc["backend_details"]["eddy_qc_collection"]["artifacts"][key]["status"],
                    "copied",
                )

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            executable_commands = [record["argv"] for record in manifest["commands"][:3]]
            self.assertTrue(all("-force" not in command for command in executable_commands))
            self.assertTrue(qc["backend_details"]["diagnostics"]["topup_field_map"]["exists"])
            self.assertEqual(qc["mask_generation"]["method"], "pnl_cnn_corrected_dwi")
            import nibabel as nib
            import numpy as np

            mask = nib.load(result.final_mask_path).get_fdata()
            self.assertEqual(int(np.count_nonzero(mask)), 27)
            calls = result.final_mask_path.with_name("mask_calls.txt")
            self.assertEqual(calls.read_text().splitlines(), [result.final_dwi_path.as_posix()])

            cached = run_dwi_preprocessing(config)
            cached_qc = json.loads(cached.qc_path.read_text(encoding="utf-8"))
            self.assertEqual(cached.status, "cached")
            self.assertEqual(cached_qc["status"], "completed")
            self.assertEqual(cached_qc["reuse_status"], "cached")
            self.assertEqual(cached_qc["mask_refresh_status"], "not_needed")
            self.assertEqual(cached_qc["warnings"], qc["warnings"])
            self.assertEqual(cached_qc["backend_details"], qc["backend_details"])
            self.assertEqual(cached_qc["mask_generation"], qc["mask_generation"])

            result.final_mask_path.unlink()
            refreshed = run_dwi_preprocessing(config)
            refreshed_qc = json.loads(refreshed.qc_path.read_text(encoding="utf-8"))
            self.assertEqual(refreshed.status, "cached")
            self.assertEqual(
                refreshed_qc["mask_refresh_status"],
                "regenerated_from_corrected_dwi",
            )
            np.testing.assert_array_equal(nib.load(refreshed.final_mask_path).get_fdata(), mask)
            self.assertEqual(calls.read_text().splitlines(), [result.final_dwi_path.as_posix()] * 2)

            changed_config = DwiPreprocessingConfig(
                subject_id="100307",
                input_root=subject,
                backend="mrtrix_fsl",
                execute=True,
                mrtrix_bin_dir=bin_dir,
                fsl_bin_dir=bin_dir,
                eddy_second_level_model="linear",
            )
            changed = run_dwi_preprocessing(changed_config)
            changed_manifest = json.loads(changed.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(changed.status, "completed")
            self.assertTrue(changed_manifest["overwrite_existing"])
            self.assertIn("-force", changed_manifest["commands"][0]["argv"])
            self.assertEqual(
                changed_manifest["cache_key"]["parameters"]["eddy_second_level_model"],
                "linear",
            )

    def test_unknown_capabilities_do_not_claim_denoising_was_skipped(self) -> None:
        from preprocessing.pipeline import _preprocessing_warnings

        self.assertEqual(_preprocessing_warnings({}, {}), [])

    def test_failed_or_running_attempt_cannot_reuse_leftover_final_files(self) -> None:
        from preprocessing import DwiPreprocessingConfig, run_dwi_preprocessing

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject = _write_subject(root)
            bin_dir = root / "bin"
            _write_fake_backend(bin_dir)
            config = DwiPreprocessingConfig(
                subject_id="100307", input_root=subject, backend="mrtrix_fsl",
                execute=True, mrtrix_bin_dir=bin_dir, fsl_bin_dir=bin_dir,
            )
            result = run_dwi_preprocessing(config)
            for status in ("failed", "running"):
                manifest = json.loads(result.manifest_path.read_text())
                manifest["status"] = status
                result.manifest_path.write_text(json.dumps(manifest))
                self.assertEqual(run_dwi_preprocessing(config).status, "completed")

    def test_mrtrix_reverse_pair_capabilities_report_applied_topup(self) -> None:
        from preprocessing import DwiPreprocessingConfig, inspect_preprocessing_backend
        from preprocessing.pipeline import _preprocessing_warnings

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = DwiPreprocessingConfig(
                subject_id="100307", input_root=root, backend="mrtrix_fsl",
                reverse_phase_b0=root / "reverse.nii.gz",
            )
            backend = inspect_preprocessing_backend(config)
            self.assertTrue(backend["capabilities"]["susceptibility_distortion_correction"])
            self.assertFalse(any("Denoising" in warning or "Susceptibility" in warning
                                 for warning in _preprocessing_warnings(backend, {})))

    def test_mrtrix_rpe_none_forwards_phase_encoding_and_readout(self) -> None:
        from preprocessing import DwiPreprocessingConfig, run_dwi_preprocessing

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            subject = _write_subject(tmp_path)
            bin_dir = tmp_path / "fsl" / "bin"
            _write_fake_backend(bin_dir)

            result = run_dwi_preprocessing(
                DwiPreprocessingConfig(
                    subject_id="100307",
                    input_root=subject,
                    backend="mrtrix_fsl",
                        execute=False,
                    mrtrix_bin_dir=bin_dir,
                    fsl_bin_dir=bin_dir,
                    phase_encoding_direction="AP",
                    readout_time=0.1,
                )
            )

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            command = manifest["commands"][2]["argv"]
            self.assertIn("-rpe_none", command)
            self.assertEqual(command[command.index("-pe_dir") + 1], "AP")
            self.assertEqual(command[command.index("-readout_time") + 1], "0.1")

    def test_force_is_forwarded_to_mrtrix_commands(self) -> None:
        from preprocessing import DwiPreprocessingConfig, run_dwi_preprocessing

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            subject = _write_subject(tmp_path)
            bin_dir = tmp_path / "bin"
            _write_fake_backend(bin_dir)

            result = run_dwi_preprocessing(
                DwiPreprocessingConfig(
                    subject_id="100307",
                    input_root=subject,
                    backend="mrtrix_fsl",
                        execute=True,
                    force=True,
                    mrtrix_bin_dir=bin_dir,
                    fsl_bin_dir=bin_dir,
                )
            )

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            executable_commands = [record["argv"] for record in manifest["commands"][:3]]
            self.assertTrue(all("-force" in command for command in executable_commands))

    def test_external_command_failure_is_recorded_in_manifest_and_qc(self) -> None:
        from preprocessing import DwiPreprocessingConfig, run_dwi_preprocessing

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            subject = _write_subject(tmp_path)
            bin_dir = tmp_path / "bin"
            _write_fake_backend(bin_dir)
            _write_fake_command(
                bin_dir / "mrdegibbs",
                """
import sys
sys.exit(9)
""",
            )
            output_dir = subject / "outputs" / "01_preproc"

            with self.assertRaisesRegex(RuntimeError, "exit code 9"):
                run_dwi_preprocessing(
                    DwiPreprocessingConfig(
                        subject_id="100307",
                        input_root=subject,
                        output_dir=output_dir,
                        backend="mrtrix_fsl",
                                execute=True,
                        mrtrix_bin_dir=bin_dir,
                        fsl_bin_dir=bin_dir,
                    )
                )

            manifest = json.loads(
                (output_dir / "100307_preproc_manifest.json").read_text(encoding="utf-8")
            )
            qc = json.loads(
                (output_dir / "05_qc" / "100307_preproc_qc.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(qc["status"], "failed")
            self.assertIn("exit code 9", qc["error"])

    def test_single_volume_reverse_phase_image_is_normalized_to_4d(self) -> None:
        import nibabel as nib
        import numpy as np

        from preprocessing import DwiPreprocessingConfig, run_dwi_preprocessing

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            subject = _write_subject(tmp_path)
            bin_dir = tmp_path / "bin"
            _write_fake_backend(bin_dir)
            reverse_b0 = tmp_path / "reverse_b0.nii.gz"
            nib.save(nib.Nifti1Image(np.zeros((5, 5, 5), dtype=np.float32), np.eye(4)), str(reverse_b0))

            result = run_dwi_preprocessing(
                DwiPreprocessingConfig(
                    subject_id="100307",
                    input_root=subject,
                    backend="mrtrix_fsl",
                        execute=True,
                    force=True,
                    mrtrix_bin_dir=bin_dir,
                    fsl_bin_dir=bin_dir,
                    reverse_phase_b0=reverse_b0,
                    phase_encoding_direction="AP",
                    readout_time=0.05,
                    eddy_threads=3,
                    eddy_second_level_model="linear",
                )
            )

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            commands = [record["argv"] for record in manifest["commands"]]
            normalize = next(command for command in commands if command[0].endswith("mrconvert"))
            pair = next(command for command in commands if command[0].endswith("mrcat"))
            preprocess = next(command for command in commands if command[0].endswith("dwifslpreproc"))
            self.assertIn("0,1,2,-1", normalize)
            self.assertTrue(pair[3].endswith("_se_epi_pair.mif"))
            self.assertTrue(preprocess[preprocess.index("-se_epi") + 1].endswith("_se_epi_pair.mif"))
            eddy_options = preprocess[preprocess.index("-eddy_options") + 1]
            self.assertIn("--nthr=3", eddy_options)
            self.assertIn("--slm=linear", eddy_options)
            self.assertEqual(manifest["parameters"]["eddy_threads"], 3)
            self.assertEqual(manifest["parameters"]["eddy_second_level_model"], "linear")

    def test_execute_dipy_basic_generates_final_outputs(self) -> None:
        from preprocessing import DwiPreprocessingConfig, run_dwi_preprocessing

        with tempfile.TemporaryDirectory() as tmp:
            subject = _write_nifti_subject(Path(tmp))
            result = run_dwi_preprocessing(
                DwiPreprocessingConfig(
                    subject_id="100307",
                    input_root=subject,
                    backend="dipy",
                    brain_mask_backend="provided",
                    execute=True,
                    dipy_denoise_method="none",
                    dipy_run_gibbs=False,
                    dipy_run_motion_correction=False,
                )
            )

            qc = json.loads(result.qc_path.read_text(encoding="utf-8"))
            self.assertEqual(result.status, "completed")
            self.assertTrue(result.final_dwi_path.is_file())
            self.assertEqual(qc["backend_details"]["denoise"]["status"], "skipped")
            self.assertEqual(qc["backend_details"]["motion"]["status"], "skipped")

    def test_execute_dipy_fast_path_links_raw_outputs(self) -> None:
        from preprocessing import DwiPreprocessingConfig, run_dwi_preprocessing

        with tempfile.TemporaryDirectory() as tmp:
            subject = _write_subject(Path(tmp))
            stale_final = subject / "outputs" / "01_preproc" / "100307_dwi_final.nii.gz"
            stale_final.parent.mkdir(parents=True)
            stale_final.write_text("stale preprocessed data", encoding="utf-8")
            result = run_dwi_preprocessing(
                DwiPreprocessingConfig(
                    subject_id="100307",
                    input_root=subject,
                    backend="dipy",
                    brain_mask_backend="provided",
                    execute=True,
                    dipy_denoise_method="none",
                    dipy_run_gibbs=False,
                    dipy_run_motion_correction=False,
                    force=True,
                )
            )

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            qc = json.loads(result.qc_path.read_text(encoding="utf-8"))
            self.assertEqual(result.status, "completed")
            self.assertTrue(result.final_dwi_path.exists())
            self.assertEqual(
                result.final_dwi_path.resolve(),
                (subject / "input" / "data.nii.gz").resolve(),
            )
            self.assertEqual(manifest["status"], "completed")
            self.assertTrue(qc["backend_details"]["fast_path"])
            self.assertEqual(len(qc["warnings"]), 5)
            self.assertTrue(any("Bias-field" in warning for warning in qc["warnings"]))


if __name__ == "__main__":
    unittest.main()
