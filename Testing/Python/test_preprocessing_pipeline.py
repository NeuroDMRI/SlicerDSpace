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
    # W2b (T1 D1): preprocessing needs a phase-encoding direction. The sidecar gives it (j- on an identity affine = AP).
    (input_dir / "dwi.json").write_text(
        json.dumps({"PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.05}), encoding="utf-8")
    return subject


def _write_fake_command(path: Path, body: str) -> None:
    from shared.external_process import _default_slicer_python
    python_exec = _default_slicer_python() or Path(sys.executable)
    if sys.platform != "win32":
        py_file = path.with_name(f"._{path.name}_fake.py")
        py_file.write_text(body, encoding="utf-8")
        path.write_text(f'#!/bin/sh\nexec "{python_exec}" "{py_file}" "$@"\n', encoding="utf-8")
    else:
        path.write_text(f"#!{python_exec}\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _write_fake_backend(bin_dir: Path) -> None:
    bin_dir.mkdir(parents=True)
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


def _external_commands(manifest):
    """Argv of the subprocess commands of a plan: the in-process DIPY steps and the output marker are left out."""
    return [record["argv"] for record in manifest["commands"]
            if record["argv"][0] != "dipy" and record["argv"][0] != "optional-output"]


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
            self.assertEqual(manifest["backend"]["backend"], "dipy_mppca+gibbs+fsl_eddy")
            self.assertEqual([record["argv"][1] for record in manifest["commands"][:2]], ["mppca", "gibbs_removal"])
            self.assertEqual(Path(manifest["commands"][2]["argv"][0]).name, "dwifslpreproc")

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
            executable_commands = _external_commands(manifest)
            self.assertEqual(len(executable_commands), 1)
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
                execute=True,
                mrtrix_bin_dir=bin_dir,
                fsl_bin_dir=bin_dir,
                eddy_second_level_model="linear",
            )
            changed = run_dwi_preprocessing(changed_config)
            changed_manifest = json.loads(changed.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(changed.status, "completed")
            self.assertTrue(changed_manifest["overwrite_existing"])
            self.assertIn("-force", _external_commands(changed_manifest)[0])
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
                subject_id="100307", input_root=subject,
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
                subject_id="100307", input_root=root,
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
                        execute=False,
                    mrtrix_bin_dir=bin_dir,
                    fsl_bin_dir=bin_dir,
                    phase_encoding_direction="AP",
                    readout_time=0.1,
                )
            )

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            command = _external_commands(manifest)[0]
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
                        execute=True,
                    force=True,
                    mrtrix_bin_dir=bin_dir,
                    fsl_bin_dir=bin_dir,
                )
            )

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            executable_commands = _external_commands(manifest)
            self.assertEqual(len(executable_commands), 1)
            self.assertTrue(all("-force" in command for command in executable_commands))

    def test_external_command_failure_is_recorded_in_manifest_and_qc(self) -> None:
        from preprocessing import DwiPreprocessingConfig, run_dwi_preprocessing

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            subject = _write_subject(tmp_path)
            bin_dir = tmp_path / "bin"
            _write_fake_backend(bin_dir)
            _write_fake_command(
                bin_dir / "dwifslpreproc",
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


    def _w2a_plan(self, root, **kwargs):
        from preprocessing import DwiPreprocessingConfig, run_dwi_preprocessing

        subject = _write_subject(root)
        bin_dir = root / "bin"
        _write_fake_backend(bin_dir)
        result = run_dwi_preprocessing(DwiPreprocessingConfig(
            subject_id="100307", input_root=subject, execute=False, mrtrix_bin_dir=bin_dir, fsl_bin_dir=bin_dir, **kwargs))
        return json.loads(result.manifest_path.read_text(encoding="utf-8"))

    def test_w2a_plan_is_mppca_then_gibbs_then_dwifslpreproc_without_rpe_pair(self) -> None:
        """W2a (D1): one route. MP-PCA, then Gibbs removal, then dwifslpreproc on the Gibbs-corrected image."""
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self._w2a_plan(Path(tmp))
            argv = [record["argv"] for record in manifest["commands"]]
            names = [Path(item[0]).name if item[0] != "dipy" else "dipy:" + item[1] for item in argv]
            self.assertEqual(names, ["dipy:mppca", "dipy:gibbs_removal", "dwifslpreproc", "optional-output"])
            flat = " ".join(" ".join(item) for item in argv)
            self.assertNotIn("dwidenoise", flat)
            self.assertNotIn("mrdegibbs", flat)
            self.assertNotIn("-rpe_pair", argv[2])
            self.assertIn("-rpe_none", argv[2])
            # The dwifslpreproc input is the Gibbs-corrected image, with the original FSL gradients.
            self.assertEqual(argv[2][1], argv[1][3])
            self.assertEqual(argv[2][argv[2].index("-fslgrad") + 1:argv[2].index("-fslgrad") + 3],
                             [str(Path(tmp).resolve() / "100307" / "input" / "bvecs"), str(Path(tmp).resolve() / "100307" / "input" / "bvals")])
            self.assertNotIn("-eddy_mask", argv[2])
            self.assertEqual(manifest["backend"]["missing_required"], [])

    def test_w2a_skip_gibbs_keeps_the_degibbs_path_for_dwifslpreproc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self._w2a_plan(Path(tmp), dipy_run_gibbs=False)
            argv = [record["argv"] for record in manifest["commands"]]
            self.assertEqual([item[1] for item in argv[:2]], ["mppca", "skip_gibbs_removal"])
            self.assertEqual(argv[2][1], argv[1][3])

    def test_w2a_rpe_pair_only_when_a_reverse_b0_is_given(self) -> None:
        import nibabel as nib
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reverse = root / "reverse_b0.nii.gz"
            nib.save(nib.Nifti1Image(np.zeros((5, 5, 5, 1), dtype=np.float32), np.eye(4)), str(reverse))
            manifest = self._w2a_plan(root, reverse_phase_b0=reverse, phase_encoding_direction="AP", readout_time=0.05)
            argv = [record["argv"] for record in manifest["commands"]]
            names = [Path(item[0]).name if item[0] != "dipy" else "dipy:" + item[1] for item in argv]
            self.assertEqual(names, ["dipy:mppca", "dipy:gibbs_removal", "dwiextract", "mrconvert", "mrcat",
                                     "dwifslpreproc", "optional-output"])
            self.assertIn("-rpe_pair", argv[5])
            self.assertNotIn("-rpe_none", argv[5])
            self.assertEqual(argv[2][1], argv[1][3])

    def test_w2a_dipy_steps_run_in_process_and_save_float32_with_the_source_affine(self) -> None:
        import nibabel as nib
        import numpy as np

        from preprocessing import DwiPreprocessingConfig, run_dwi_preprocessing

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject = _write_subject(root)
            bin_dir = root / "bin"
            _write_fake_backend(bin_dir)
            result = run_dwi_preprocessing(DwiPreprocessingConfig(
                subject_id="100307", input_root=subject, execute=True, mrtrix_bin_dir=bin_dir, fsl_bin_dir=bin_dir))
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            argv = [record["argv"] for record in manifest["commands"]]
            for step in argv[:2]:
                image = nib.load(step[3])
                self.assertEqual(image.get_data_dtype(), np.float32)
                np.testing.assert_array_equal(image.affine, np.eye(4))
                self.assertEqual(image.shape, (5, 5, 5, 3))
            qc = json.loads(result.qc_path.read_text(encoding="utf-8"))
            self.assertEqual(qc["backend_details"]["denoise"]["method"], "dipy_mppca")
            self.assertEqual(qc["backend_details"]["gibbs"]["method"], "dipy_gibbs_removal")

    def test_w2a_the_route_needs_dwifslpreproc_and_dwi2mask_not_the_mrtrix_denoise_tools(self) -> None:
        from preprocessing.pipeline import REQUIRED_COMMANDS

        self.assertEqual(REQUIRED_COMMANDS, ("dwifslpreproc", "dwi2mask"))

    def test_w2a_removed_config_fields_are_gone(self) -> None:
        from preprocessing import DwiPreprocessingConfig

        for name in ("backend", "dipy_denoise_method", "dipy_run_motion_correction"):
            with self.subTest(name=name), self.assertRaises(TypeError):
                DwiPreprocessingConfig(subject_id="100307", input_root=Path("."), **{name: "x"})


class NativePreprocessingIntegrationTest(unittest.TestCase):
    def test_native_dry_run_has_own_backend_and_cache_identity(self):
        from preprocessing import DwiPreprocessingConfig, run_dwi_preprocessing
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subject = _write_nifti_subject(root)
            common = dict(subject_id="100307", input_root=subject, brain_mask_backend="provided",
                          mrtrix_bin_dir=root / "mrtrix", fsl_bin_dir=root / "fsl")
            with patch.dict("os.environ", {"DSP_PREPROCESSING_EXECUTION_BACKEND": "native"}):
                native = run_dwi_preprocessing(DwiPreprocessingConfig(output_dir=root / "native", **common))
                legacy = run_dwi_preprocessing(DwiPreprocessingConfig(
                    output_dir=root / "legacy", execution_backend="legacy", **common))
            native_record = json.loads(native.manifest_path.read_text())
            legacy_record = json.loads(legacy.manifest_path.read_text())
            self.assertEqual(native_record["backend"]["execution_backend"], "native")
            self.assertNotIn("environment", native_record["backend"])
            self.assertTrue(any(row["argv"][0] == "native-fsl-correction" for row in native_record["commands"]))
            self.assertNotEqual(native_record["cache_key"], legacy_record["cache_key"])
            self.assertNotIn("execution_backend", legacy_record["cache_key"])


if __name__ == "__main__":
    unittest.main()
