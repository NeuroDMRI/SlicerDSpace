#!/usr/bin/env python3
"""tckgen seeding, seeded thread pinning, SIFT2 mu and anatomy-space checks."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from tractography.csd import (  # noqa: E402
    CSDActConfig,
    build_csd_act_commands,
    reproducibility_record,
    run_csd_act,
    tckgen_thread_count,
)


def _inputs(root: Path) -> dict[str, Path]:
    paths = {
        "dwi_path": root / "data.nii.gz",
        "mask_path": root / "mask.nii.gz",
        "five_tt_path": root / "5tt.nii.gz",
        "gmwmi_path": root / "gmwmi.nii.gz",
        "bval_path": root / "bvals",
        "bvec_path": root / "bvecs",
    }
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2, 2), dtype=np.float32), np.eye(4)), str(paths["dwi_path"]))
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4)), str(paths["mask_path"]))
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2, 5), dtype=np.float32), np.eye(4)), str(paths["five_tt_path"]))
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4)), str(paths["gmwmi_path"]))
    paths["bval_path"].write_text("0 1000\n", encoding="utf-8")
    paths["bvec_path"].write_text("0 1\n0 0\n0 0\n", encoding="utf-8")
    return paths


def _config(root: Path, **overrides) -> CSDActConfig:
    values = {
        "subject_id": "100307",
        **_inputs(root),
        "output_dir": root / "out" / "csd",
        "select_streamlines": 10,
        "mrtrix_bin_dir": root / "fake_bin",
    }
    values.update(overrides)
    return CSDActConfig(**values)


def _command(commands, name: str) -> tuple[str, ...]:
    return next(
        command
        for command in commands
        if Path(command[0]).stem == name or Path(command[0]).name == name or command[0] == name
    )


class TckgenSeedingTest(unittest.TestCase):
    def test_gmwmi_seeding_option_uses_interface_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(Path(tmp), seeding="gmwmi")
            tckgen = _command(build_csd_act_commands(config), "tckgen")
            self.assertIn("-seed_gmwmi", tckgen)
            self.assertNotIn("-seed_dynamic", tckgen)
            self.assertEqual(tckgen[tckgen.index("-seed_gmwmi") + 1], config.gmwmi_path.resolve().as_posix())
            self.assertIn("-backtrack", tckgen)
            self.assertIn("-crop_at_gmwmi", tckgen)

    def test_unknown_seeding_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "seeding"):
                run_csd_act(_config(Path(tmp), seeding="random"))

    def test_seed_pins_tckgen_to_single_thread_and_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(Path(tmp), random_seed=20260926, nthreads=16)
            commands = build_csd_act_commands(config)
            tckgen = _command(commands, "tckgen")
            tcksift2 = _command(commands, "tcksift2")
            self.assertEqual(tckgen[tckgen.index("-nthreads") + 1], "0")
            # tcksift2 keeps the requested thread count (no RNG involved).
            self.assertEqual(tcksift2[tcksift2.index("-nthreads") + 1], "16")
            self.assertEqual(tckgen_thread_count(config), 0)
            record = reproducibility_record(config)
            self.assertTrue(record["tckgen_bit_reproducible"])
            self.assertTrue(record["tckgen_nthreads_pinned_for_seed"])
            self.assertEqual(record["random_seed"], 20260926)
            result = run_csd_act(config)
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(manifest["reproducibility"]["tckgen_bit_reproducible"])
            self.assertEqual(manifest["parameters"]["tckgen_nthreads"], 0)
            for command in manifest["commands"]:
                self.assertNotIn("RandomSeed", command)
                self.assertNotIn("-config", command)

    def test_unseeded_run_keeps_requested_threads_and_is_not_bit_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(Path(tmp), nthreads=8)
            tckgen = _command(build_csd_act_commands(config), "tckgen")
            self.assertEqual(tckgen[tckgen.index("-nthreads") + 1], "8")
            self.assertFalse(reproducibility_record(config)["tckgen_bit_reproducible"])

    def test_every_mrtrix_stage_respects_the_requested_thread_budget(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for seed, threads in ((None, 8), (17, 8), (None, 0)):
                with self.subTest(seed=seed, threads=threads):
                    commands = build_csd_act_commands(_config(Path(tmp), random_seed=seed, nthreads=threads,
                                                              export_vtk=True))
                    for command in commands:
                        self.assertEqual(command.count('-nthreads'), 1)
                        expected = 0 if Path(command[0]).stem == 'tckgen' and seed is not None else threads
                        self.assertEqual(command[command.index('-nthreads') + 1], str(expected))

    def test_tcksift2_writes_mu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _config(Path(tmp))
            result = run_csd_act(config)
            tcksift2 = _command(result.commands, "tcksift2")
            self.assertIn("-out_mu", tcksift2)
            mu_path = tcksift2[tcksift2.index("-out_mu") + 1]
            self.assertTrue(mu_path.endswith("_sift2_mu.txt"))
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["outputs"]["sift2_mu"], mu_path)

    def test_execute_without_seed_does_not_export_rng_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            for name in ("mrconvert", "dwi2response", "dwi2fod", "mtnormalise", "tckgen", "tcksift2"):
                (bin_dir / name).write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                (bin_dir / name).chmod(0o755)
            config = _config(root, mrtrix_bin_dir=bin_dir, execute=True)
            with mock.patch("tractography.csd.subprocess.run") as mock_run, mock.patch.dict(
                "os.environ", {}, clear=False
            ) as environ:
                environ.pop("MRTRIX_RNG_SEED", None)
                mock_run.return_value = subprocess.CompletedProcess(args=(), returncode=0)
                with self.assertRaises(RuntimeError):
                    run_csd_act(config)
            for call in mock_run.call_args_list:
                self.assertNotIn("MRTRIX_RNG_SEED", call.kwargs["env"])
                self.assertEqual(call.kwargs['cwd'], config.output_dir.resolve())


class AnatomySpaceTest(unittest.TestCase):
    def test_mismatched_anatomy_dwi_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            other = root / "raw_data.nii.gz"
            other.write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different DWI"):
                run_csd_act(_config(root, anatomy_source_dwi=other))

    def test_mismatch_can_be_explicitly_allowed_and_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            other = root / "raw_data.nii.gz"
            other.write_text("x", encoding="utf-8")
            result = run_csd_act(_config(root, anatomy_source_dwi=other, allow_anatomy_space_mismatch=True))
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertFalse(manifest["anatomy_space"]["same_dwi"])
            self.assertTrue(manifest["anatomy_space"]["mismatch_allowed"])

    def test_matching_anatomy_dwi_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _config(root)
            config = _config(root, anatomy_source_dwi=config.dwi_path)
            manifest = json.loads(run_csd_act(config).manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(manifest["anatomy_space"]["same_dwi"])


class OutputValidationTest(unittest.TestCase):
    def test_sift2_mu_must_be_finite_and_positive(self) -> None:
        from tractography.csd import _validate_generated_outputs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = _config(root, select_streamlines=1)
            tck = root / "t.tck"
            streamlines = nib.streamlines.Tractogram([np.array([[0, 0, 0], [20, 0, 0]], dtype=np.float32)], affine_to_rasmm=np.eye(4))
            nib.streamlines.save(streamlines, str(tck))
            weights = root / "t_sift2_weights.txt"
            weights.write_text("1.0\n", encoding="utf-8")
            mu = root / "t_sift2_mu.txt"
            mu.write_text("0.5\n", encoding="utf-8")
            summary, error = _validate_generated_outputs(config, tck, None, None, weights)
            self.assertIsNone(error)
            self.assertAlmostEqual(summary["sift2_mu"]["value"], 0.5)
            self.assertAlmostEqual(summary["tck"]["length_mm"]["mean"], 20.0, places=4)
            mu.write_text("nan\n", encoding="utf-8")
            _, error = _validate_generated_outputs(config, tck, None, None, weights)
            self.assertIsNotNone(error)


if __name__ == "__main__":
    unittest.main()
