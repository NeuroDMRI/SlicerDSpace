#!/usr/bin/env python3
"""Tests for automatic brain-mask estimation."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import brain_mask
from shared.brain_mask import BrainMaskConfig
from Testing.Python.test_brain_mask_backends import (
    ISOLATED_ENV, command_record, fake_pnl_config, fake_pythonslicer, make_subject, mask_data,
)


class BrainMaskTest(unittest.TestCase):
    def setUp(self) -> None:
        environment = patch.dict(os.environ, ISOLATED_ENV)
        environment.start()
        self.addCleanup(environment.stop)

    def fake_pnl(self, root: Path) -> BrainMaskConfig:
        """W5: the one generating backend, on the native route with a fake PythonSlicer."""
        config = fake_pnl_config(root)
        tool = fake_pythonslicer(root)
        resolver = patch.object(brain_mask, "_pnl_native_python", return_value=tool)
        resolver.start()
        self.addCleanup(resolver.stop)
        return config

    def test_missing_subject_mask_is_generated_with_pnl_cnn(self) -> None:
        from shared.pipeline import find_subject_root

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            config = self.fake_pnl(root)
            subject_root = make_subject(root)
            input_dir = subject_root / "input"

            raw = find_subject_root(subject_root, "sub-01", brain_mask_config=config)

            self.assertTrue(raw.mask_path.is_file())
            self.assertTrue((input_dir / "nodif_brain_mask.manifest.json").is_file())
            np.testing.assert_array_equal(nib.load(str(raw.mask_path)).dataobj, mask_data())

    def test_missing_subject_mask_without_pnl_files_fails_with_setup_message(self) -> None:
        from shared.pipeline import find_subject_root

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = make_subject(Path(tmp))
            with self.assertRaisesRegex(FileNotFoundError, "pnl_cnn unavailable"):
                find_subject_root(subject_root, "sub-01")

    def test_validation_accepts_subject_without_supplied_mask_when_pnl_is_available(self) -> None:
        from shared import InputValidationConfig, validate_subject_inputs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            config = self.fake_pnl(root)
            subject_root = make_subject(root)
            input_dir = subject_root / "input"

            result = validate_subject_inputs(
                InputValidationConfig(
                    subject_id="sub-01", input_root=subject_root, dependency_groups=(),
                    brain_mask_backend="pnl_cnn", brain_mask_device="cpu",
                    pnl_mask_model_dir=config.pnl_mask_model_dir, pnl_mask_route="native",
                )
            )

            self.assertTrue(result.passed, result.messages)
            self.assertTrue((input_dir / "nodif_brain_mask.nii.gz").is_file())
            self.assertTrue(result.metrics["brain_mask_qc"]["geometry_pass"])

    def test_default_pnl_missing_dependencies_fails_without_a_fallback(self) -> None:
        from shared import InputValidationConfig, validate_subject_inputs
        from shared.pipeline import find_subject_root

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject = make_subject(root)
            with self.assertRaisesRegex(FileNotFoundError, "pnl_cnn unavailable"):
                find_subject_root(subject, "sub-01")
            result = validate_subject_inputs(InputValidationConfig(
                subject_id="sub-01", input_root=subject, dependency_groups=(), write_report=False,
            ))
            self.assertFalse(result.passed)
            self.assertEqual(result.metrics["brain_mask_selection"]["backend"], "pnl_cnn")
            self.assertTrue(any("pnl_cnn" in issue.message for issue in result.messages))
            self.assertFalse((subject / "input/nodif_brain_mask.nii.gz").exists())

    def test_valid_supplied_mask_is_preserved_and_passes_qc_without_pnl_dependencies(self) -> None:
        from shared import InputValidationConfig, validate_subject_inputs
        from shared.pipeline import find_subject_root

        with tempfile.TemporaryDirectory() as tmp:
            subject = make_subject(Path(tmp).resolve(), supplied_mask=True)
            mask = subject / "input/nodif_brain_mask.nii.gz"
            original = mask.read_bytes()
            with patch("shared.brain_mask._run") as run:
                raw = find_subject_root(subject, "sub-01")
                result = validate_subject_inputs(InputValidationConfig(
                    subject_id="sub-01", input_root=subject, dependency_groups=(), write_report=False,
                ))
            run.assert_not_called()
            self.assertEqual(raw.mask_path, mask)
            self.assertEqual(mask.read_bytes(), original)
            self.assertTrue(result.passed, result.messages)
            self.assertFalse(result.metrics["brain_mask_backend"]["available"])
            self.assertEqual(result.metrics["brain_mask_qc"]["brain_voxels"], 12)
            self.assertFalse(mask.with_name("nodif_brain_mask.manifest.json").exists())

    def test_find_subject_root_invalidates_generated_mask_after_dwi_changes(self) -> None:
        from shared.pipeline import find_subject_root
        from Testing.Python.test_brain_mask_backends import save_nifti

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            config = self.fake_pnl(root)
            tool = fake_pythonslicer(root)
            subject = make_subject(root)
            raw = find_subject_root(subject, "sub-01", brain_mask_config=config)
            record = raw.mask_path.with_name("nodif_brain_mask.manifest.json")
            first = json.loads(record.read_text(encoding="utf-8"))
            data = nib.load(str(raw.dwi_path)).get_fdata().astype(np.float32)
            save_nifti(raw.dwi_path, data + 1)
            find_subject_root(subject, "sub-01", brain_mask_config=config)
            second = json.loads(record.read_text(encoding="utf-8"))
            self.assertNotEqual(first["signature"]["inputs"], second["signature"]["inputs"])
            self.assertEqual(len(command_record(tool).read_text(encoding="utf-8").splitlines()), 2)
            with patch("shared.brain_mask._run") as run:
                find_subject_root(subject, "sub-01", brain_mask_config=config)
            run.assert_not_called()

    def test_find_subject_root_does_not_ignore_invalid_generated_record_json(self) -> None:
        from shared.pipeline import find_subject_root

        with tempfile.TemporaryDirectory() as tmp:
            subject = make_subject(Path(tmp), supplied_mask=True)
            record = subject / "input/nodif_brain_mask.manifest.json"
            record.write_text('{"version": 2, broken json\n', encoding="ascii")
            mask = subject / "input/nodif_brain_mask.nii.gz"
            original = mask.read_bytes()
            with patch("shared.brain_mask._run") as run:
                with self.assertRaises(json.JSONDecodeError):
                    find_subject_root(subject, "sub-01")
            run.assert_not_called()
            self.assertEqual(mask.read_bytes(), original)

    def test_invalid_supplied_mask_fails_validation_without_inference_or_replacement(self) -> None:
        from shared import InputValidationConfig, validate_subject_inputs
        from Testing.Python.test_brain_mask_backends import AFFINE, SHAPE, save_nifti

        with tempfile.TemporaryDirectory() as tmp:
            subject = make_subject(Path(tmp))
            mask = subject / "input/nodif_brain_mask.nii.gz"
            shifted = AFFINE.copy()
            shifted[0, 3] += 1
            for data, affine in (
                (np.zeros(SHAPE, dtype=np.uint8), AFFINE),
                (np.full(SHAPE, 0.5, dtype=np.float32), AFFINE),
                (mask_data(), shifted),
            ):
                with self.subTest(affine=affine.tolist(), value=float(data.max())):
                    save_nifti(mask, data, affine)
                    original = mask.read_bytes()
                    with patch("shared.brain_mask._run") as run:
                        result = validate_subject_inputs(InputValidationConfig(
                            subject_id="sub-01", input_root=subject, dependency_groups=(),
                            brain_mask_backend="provided", write_report=False,
                        ))
                    run.assert_not_called()
                    self.assertFalse(result.passed)
                    self.assertTrue(any(issue.category == "brain_mask" for issue in result.messages))
                    self.assertEqual(mask.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
