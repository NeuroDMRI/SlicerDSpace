#!/usr/bin/env python3
"""Tests for the external MRtrix CSD/FOD/ACT backend planning layer."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from tractography.csd import CSDActConfig, _timed_step, inspect_mrtrix_csd_backend, run_csd_act


class CSDActBackendTest(unittest.TestCase):
    def test_windows_native_response_uses_explicit_python_and_runtime_library(self):
        from unittest.mock import patch
        from tractography.csd import _mrtrix_csd_invocation

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            script = root / "runtime with spaces" / "bin" / "dwi2response"
            script.parent.mkdir(parents=True)
            script.touch()
            library = script.parent.parent / "lib"
            (library / "mrtrix3").mkdir(parents=True)
            (library / "mrtrix3" / "__init__.py").touch()
            python = root / "Slicer with spaces" / "PythonSlicer.exe"
            with patch("tractography.csd.sys.platform", "win32"), patch(
                "tractography.csd.find_pythonslicer", return_value=python,
            ):
                command, environment = _mrtrix_csd_invocation((str(script), "dhollander", "a b.mif"), {})
                legacy, legacy_env = _mrtrix_csd_invocation((str(script) + ".exe", "dhollander"), {})
            self.assertEqual(command, (str(python), "-B", "-S", str(script), "dhollander", "a b.mif"))
            self.assertEqual(environment["PYTHONPATH"], str(library))
            self.assertEqual(legacy, (str(script) + ".exe", "dhollander"))
            self.assertNotIn("PYTHONPATH", legacy_env)

    def test_phase_timing_preserves_running_success_and_failed_subprocess(self):
        import subprocess

        records, snapshots = [], []
        def persist():
            snapshots.append(json.loads(json.dumps(records)))
        with _timed_step(records, 'tckgen', persist, ['tckgen', 'input', 'output']) as phase:
            phase['return_code'] = 0
        self.assertEqual(snapshots[0][0]['status'], 'running')
        self.assertEqual(snapshots[1][0]['status'], 'completed')
        self.assertEqual(records[0]['command'], ['tckgen', 'input', 'output'])
        self.assertGreaterEqual(records[0]['elapsed_seconds'], 0)
        with self.assertRaises(subprocess.CalledProcessError):
            with _timed_step(records, 'tcksift2', persist):
                raise subprocess.CalledProcessError(7, ['tcksift2'])
        self.assertEqual(snapshots[-1][-1]['status'], 'failed')
        self.assertEqual(records[-1]['return_code'], 7)
        self.assertEqual(records[-1]['index'], 1)

    def test_dry_run_writes_manifest_qc_and_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            dwi = root / "data.nii.gz"
            mask = root / "mask.nii.gz"
            five_tt = root / "5tt.nii.gz"
            gmwmi = root / "gmwmi.nii.gz"
            bval = root / "bvals"
            bvec = root / "bvecs"
            nib.save(nib.Nifti1Image(np.ones((2, 2, 2, 2), dtype=np.float32), np.eye(4)), str(dwi))
            nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4)), str(mask))
            nib.save(nib.Nifti1Image(np.ones((2, 2, 2, 5), dtype=np.float32), np.eye(4)), str(five_tt))
            nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4)), str(gmwmi))
            bval.write_text("0 1000\n", encoding="utf-8")
            bvec.write_text("0 1\n0 0\n0 0\n", encoding="utf-8")
            prep_qc = root / "act_prep_qc.json"
            prior_warning = "Explicit DDSeg-only fallback: subcortical GM is unavailable."
            prep_qc.write_text(json.dumps({"status": "completed", "anatomical_prior_status": "ddseg_only",
                "sgm_from_ddparcel": False, "warnings": [prior_warning],
                "pathology_channel": {"index": 4, "modelled": False, "value": 0.0},
                "anatomical_review": "required"}))

            result = run_csd_act(
                CSDActConfig(
                    subject_id="100307",
                    dwi_path=dwi,
                    bval_path=bval,
                    bvec_path=bvec,
                    mask_path=mask,
                    five_tt_path=five_tt,
                    gmwmi_path=gmwmi,
                    output_dir=root / "outputs" / "04_tractography" / "csd",
                    tractogram_dir=root / "outputs" / "04_tractography" / "tractograms" / "csd_act_test",
                    select_streamlines=10,
                    export_trk=True,
                    export_vtk=True,
                    mrtrix_bin_dir=root / "fake_mrtrix_bin",
                    act_prep_qc_path=prep_qc,
                )
            )

            self.assertFalse(result.executed)
            self.assertTrue(result.manifest_path.is_file())
            self.assertTrue(result.qc_path.is_file())
            self.assertGreaterEqual(len(result.commands), 7)
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            qc = json.loads(result.qc_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["kind"], "mrtrix_csd_act")
            self.assertEqual(qc["status"], "planned")
            self.assertEqual(qc["act_preparation"], manifest["act_preparation"])
            self.assertEqual(qc["act_preparation"]["anatomical_prior_status"], "ddseg_only")
            self.assertEqual(qc["act_preparation"]["warnings"], [prior_warning])
            self.assertEqual(manifest["dependencies"]["act_prep_qc"]["path"], prep_qc.as_posix())
            self.assertEqual(manifest["parameters"]["mrtrix_bin_dir"], (root / "fake_mrtrix_bin").as_posix())
            self.assertEqual(
                manifest["parameters"]["tractogram_dir"],
                (root / "outputs" / "04_tractography" / "tractograms" / "csd_act_test").as_posix(),
            )
            self.assertTrue(manifest["parameters"]["export_trk"])
            self.assertTrue(manifest["parameters"]["export_vtk"])
            self.assertIsNotNone(manifest["outputs"]["tractogram_trk"])
            self.assertIsNotNone(manifest["outputs"]["tractogram_vtk"])
            self.assertFalse(qc["tractogram_trk_exists"])
            self.assertFalse(qc["tractogram_vtk_exists"])
            self.assertIsNotNone(result.tractogram_trk_path)
            self.assertIsNotNone(result.tractogram_vtk_path)
            self.assertIn("dwi2response", " ".join(" ".join(command) for command in result.commands))
            self.assertIn("tckgen", " ".join(" ".join(command) for command in result.commands))
            self.assertIn("tckconvert", " ".join(" ".join(command) for command in result.commands))
            dwi2response = next(command for command in result.commands if "dwi2response" in command[0])
            tckgen = next(command for command in result.commands if "tckgen" in command[0])
            # MRtrix3 ignores "-config RandomSeed <n>" on the command line;
            # the seed is recorded in the manifest and applied via the
            # MRTRIX_RNG_SEED environment variable at execution time instead.
            self.assertNotIn("RandomSeed", dwi2response)
            self.assertNotIn("RandomSeed", tckgen)
            # Unseeded by default: no MRTRIX_RNG_SEED, no pinned threads.
            self.assertIsNone(manifest["parameters"]["random_seed"])
            self.assertIsNone(manifest["parameters"]["nthreads"])
            self.assertNotIn("-nthreads", tckgen)
            self.assertFalse(manifest["reproducibility"]["tckgen_bit_reproducible"])
            self.assertEqual(manifest['phase_timings'], [])
            self.assertIn("reproducibility_note", manifest)
            # Dynamic seeding is the default, on the normalised WM FOD.
            self.assertEqual(manifest["parameters"]["seeding"], "dynamic")
            self.assertIn("-seed_dynamic", tckgen)
            self.assertNotIn("-seed_gmwmi", tckgen)
            self.assertEqual(tckgen[tckgen.index("-seed_dynamic") + 1], manifest["outputs"]["wm_fod_norm"])
            # Explicit length limits are always passed (never MRtrix defaults).
            self.assertEqual(tckgen[tckgen.index("-minlength") + 1], "10.0")
            self.assertEqual(tckgen[tckgen.index("-maxlength") + 1], "250.0")
            self.assertEqual(manifest["version"], 3)

    def test_pinned_nthreads_is_passed_to_tckgen_and_tcksift2_and_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dwi = root / "data.nii.gz"
            mask = root / "mask.nii.gz"
            five_tt = root / "5tt.nii.gz"
            gmwmi = root / "gmwmi.nii.gz"
            bval = root / "bvals"
            bvec = root / "bvecs"
            nib.save(nib.Nifti1Image(np.ones((2, 2, 2, 2), dtype=np.float32), np.eye(4)), str(dwi))
            nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4)), str(mask))
            nib.save(nib.Nifti1Image(np.ones((2, 2, 2, 5), dtype=np.float32), np.eye(4)), str(five_tt))
            nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4)), str(gmwmi))
            bval.write_text("0 1000\n", encoding="utf-8")
            bvec.write_text("0 1\n0 0\n0 0\n", encoding="utf-8")

            result = run_csd_act(
                CSDActConfig(
                    subject_id="100307",
                    dwi_path=dwi,
                    bval_path=bval,
                    bvec_path=bvec,
                    mask_path=mask,
                    five_tt_path=five_tt,
                    gmwmi_path=gmwmi,
                    output_dir=root / "outputs" / "04_tractography" / "csd",
                    select_streamlines=10,
                    mrtrix_bin_dir=root / "fake_mrtrix_bin",
                    use_sift2=True,
                    nthreads=4,
                )
            )

            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["parameters"]["nthreads"], 4)
            tckgen = next(command for command in result.commands if "tckgen" in command[0])
            tcksift2 = next(command for command in result.commands if "tcksift2" in command[0])
            self.assertIn("-nthreads", tckgen)
            self.assertEqual(tckgen[tckgen.index("-nthreads") + 1], "4")
            self.assertIn("-nthreads", tcksift2)
            self.assertEqual(tcksift2[tcksift2.index("-nthreads") + 1], "4")

    def test_negative_nthreads_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = CSDActConfig(
                subject_id="100307",
                dwi_path=root / "data.nii.gz",
                bval_path=root / "bvals",
                bvec_path=root / "bvecs",
                mask_path=root / "mask.nii.gz",
                five_tt_path=root / "5tt.nii.gz",
                gmwmi_path=root / "gmwmi.nii.gz",
                output_dir=root / "outputs",
                nthreads=-1,
            )
            with self.assertRaises(ValueError):
                run_csd_act(config)

    def test_execute_sets_mrtrix_rng_seed_env_for_every_command(self) -> None:
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dwi = root / "data.nii.gz"
            mask = root / "mask.nii.gz"
            five_tt = root / "5tt.nii.gz"
            gmwmi = root / "gmwmi.nii.gz"
            bval = root / "bvals"
            bvec = root / "bvecs"
            nib.save(nib.Nifti1Image(np.ones((2, 2, 2, 2), dtype=np.float32), np.eye(4)), str(dwi))
            nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4)), str(mask))
            nib.save(nib.Nifti1Image(np.ones((2, 2, 2, 5), dtype=np.float32), np.eye(4)), str(five_tt))
            nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4)), str(gmwmi))
            bval.write_text("0 1000\n", encoding="utf-8")
            bvec.write_text("0 1\n0 0\n0 0\n", encoding="utf-8")

            bin_dir = root / "mrtrix_bin"
            bin_dir.mkdir()
            (root / "lib" / "mrtrix3").mkdir(parents=True, exist_ok=True)
            (root / "lib" / "mrtrix3" / "__init__.py").write_text("", encoding="utf-8")
            for command in ("mrconvert", "dwi2response", "dwi2fod", "mtnormalise", "tckgen", "tcksift2"):
                path = bin_dir / command
                path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                path.chmod(0o755)

            config = CSDActConfig(
                subject_id="100307",
                dwi_path=dwi,
                bval_path=bval,
                bvec_path=bvec,
                mask_path=mask,
                five_tt_path=five_tt,
                gmwmi_path=gmwmi,
                output_dir=root / "outputs" / "04_tractography" / "csd",
                select_streamlines=10,
                mrtrix_bin_dir=bin_dir,
                use_sift2=True,
                random_seed=999,
                execute=True,
            )

            import subprocess

            with mock.patch("tractography.csd.subprocess.run") as mock_run:
                mock_run.return_value = subprocess.CompletedProcess(args=(), returncode=0)
                # Validation of the (never-created) outputs fails after the
                # planned commands run; that is expected here.
                with self.assertRaises(RuntimeError):
                    run_csd_act(config)

            self.assertTrue(mock_run.call_args_list)
            seen_commands = set()
            for call_args in mock_run.call_args_list:
                command = call_args.args[0]
                self.assertNotIn("RandomSeed", command)
                self.assertEqual(call_args.kwargs["env"].get("MRTRIX_RNG_SEED"), "999")
                seen_commands.add(command[0])
            tckgen_command = next(cmd for cmd in seen_commands if "tckgen" in cmd)
            tcksift2_command = next(cmd for cmd in seen_commands if "tcksift2" in cmd)
            self.assertTrue(tckgen_command)
            self.assertTrue(tcksift2_command)
            manifest = json.loads((config.output_dir / 'csd_act_manifest.json').read_text())
            qc = json.loads((config.output_dir / 'csd_act_qc.json').read_text())
            self.assertEqual(manifest['status'], 'failed')
            self.assertEqual(len(manifest['phase_timings']), mock_run.call_count + 1)
            self.assertTrue(all(p['status'] == 'completed' for p in manifest['phase_timings'][:-1]))
            self.assertEqual(manifest['phase_timings'][-1]['step'], 'scientific_output_validation')
            self.assertEqual(manifest['phase_timings'][-1]['status'], 'failed')
            self.assertEqual(qc['phase_timings'], manifest['phase_timings'])
            self.assertEqual(qc['elapsed_seconds'], manifest['runtime_seconds'])

    def test_backend_inspection_honors_explicit_bin_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp)
            for command in ("mrconvert", "dwi2response", "dwi2fod", "mtnormalise", "tckgen", "tcksift2"):
                path = bin_dir / command
                path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                path.chmod(0o755)

            status = inspect_mrtrix_csd_backend(bin_dir)

            self.assertTrue(status["available"])
            self.assertEqual(status["missing_commands"], [])
            self.assertEqual(status["commands"]["tckgen"], (bin_dir / "tckgen").resolve().as_posix())

    def test_tractogram_summary_fast_path_matches_reference(self) -> None:
        from tractography.csd import _tractogram_summary

        rng = np.random.default_rng(42)
        streamlines = [
            rng.standard_normal((int(rng.integers(5, 30)), 3)).astype(np.float32) * 5.0
            for _ in range(120)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tck_path = Path(tmp) / "sample.tck"
            tractogram = nib.streamlines.Tractogram(streamlines, affine_to_rasmm=np.eye(4))
            nib.streamlines.save(nib.streamlines.TckFile(tractogram), str(tck_path))
            summary = _tractogram_summary(tck_path)

            ref_lengths = [
                float(np.linalg.norm(np.diff(np.asarray(sl, dtype=np.float64), axis=0), axis=1).sum())
                for sl in nib.streamlines.load(str(tck_path), lazy_load=False).streamlines
            ]
            ref_values = np.asarray(ref_lengths, dtype=np.float64)
            self.assertEqual(summary["streamline_count"], len(streamlines))
            self.assertTrue(summary["all_points_finite"])
            self.assertEqual(summary["fraction_shorter_than_20mm"], float(np.mean(ref_values < 20.0)))
            for key, expected in (
                ("min", float(np.min(ref_values))),
                ("p05", float(np.percentile(ref_values, 5))),
                ("median", float(np.median(ref_values))),
                ("mean", float(np.mean(ref_values))),
                ("p95", float(np.percentile(ref_values, 95))),
                ("max", float(np.max(ref_values))),
            ):
                self.assertEqual(summary["length_mm"][key], expected)

    def test_fod_preparation_receipt_reuse_and_invalidation(self) -> None:
        from tractography.csd import (
            _fod_preparation_paths,
            _prepared_fod_reusable,
            _write_fod_preparation_receipt,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            dwi = root / "data.nii.gz"
            mask = root / "mask.nii.gz"
            five_tt = root / "5tt.nii.gz"
            gmwmi = root / "gmwmi.nii.gz"
            bval = root / "bvals"
            bvec = root / "bvecs"
            nib.save(nib.Nifti1Image(np.ones((2, 2, 2, 2), dtype=np.float32), np.eye(4)), str(dwi))
            nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4)), str(mask))
            nib.save(nib.Nifti1Image(np.ones((2, 2, 2, 5), dtype=np.float32), np.eye(4)), str(five_tt))
            nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4)), str(gmwmi))
            bval.write_text("0 1000\n", encoding="utf-8")
            bvec.write_text("0 1\n0 0\n0 0\n", encoding="utf-8")

            out_dir = root / "csd_out"
            out_dir.mkdir()
            config = CSDActConfig(
                subject_id="100307",
                dwi_path=dwi,
                bval_path=bval,
                bvec_path=bvec,
                mask_path=mask,
                five_tt_path=five_tt,
                gmwmi_path=gmwmi,
                output_dir=out_dir,
                select_streamlines=25,
                nthreads=8,
            )
            self.assertFalse(_prepared_fod_reusable(config, out_dir))
            for path in _fod_preparation_paths("100307", out_dir).values():
                path.write_bytes(b"mrtrix-artifact")
            self.assertTrue(_write_fod_preparation_receipt(config, out_dir))
            self.assertTrue(_prepared_fod_reusable(config, out_dir))

            # Changing downstream tracking parameters (e.g., select_streamlines or seeding)
            # preserves FOD preparation reuse:
            config_more_streamlines = CSDActConfig(
                subject_id="100307",
                dwi_path=dwi,
                bval_path=bval,
                bvec_path=bvec,
                mask_path=mask,
                five_tt_path=five_tt,
                gmwmi_path=gmwmi,
                output_dir=out_dir,
                select_streamlines=500,
                seeding="gmwmi",
                nthreads=8,
            )
            self.assertTrue(_prepared_fod_reusable(config_more_streamlines, out_dir))

            # Modifying an input file invalidates the receipt:
            bval.write_text("0 2000\n", encoding="utf-8")
            self.assertFalse(_prepared_fod_reusable(config_more_streamlines, out_dir))


if __name__ == "__main__":
    unittest.main()
