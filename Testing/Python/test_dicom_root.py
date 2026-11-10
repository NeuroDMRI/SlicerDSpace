"""DICOM selection defers conversion to the worker and publishes only valid input."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

from shared.dicom_input import converter_path, is_dicom_root, prepare_dicom_root
from shared.pipeline import inspect_subject_directory
from shared.orchestrator import ProductOrchestratorConfig, run_subject_products


class DicomRootTests(unittest.TestCase):
    def test_headless_converter_discovery_and_missing_backend(self):
        with patch.dict("os.environ", {"SLICERDSPACE_DWICONVERT_EXECUTABLE": ""}), \
                patch("shared.input_import.packaged_dcm2niix", return_value=None), \
                patch("shared.input_import.shutil.which", return_value=None), \
                patch("shared.dicom_input.shutil.which", return_value=None), \
                patch("tractography.ukf.resolve_executable", return_value=Path("/slicer/DWIConvert")) as discover:
            self.assertEqual(converter_path(), ("dwiconvert", Path("/slicer/DWIConvert")))
            discover.assert_called_once_with(name="DWIConvert")
            discover.side_effect = RuntimeError("not installed")
            with self.assertRaisesRegex(FileNotFoundError, "no conversion or analysis was run"):
                converter_path()

    def source(self, base):
        root = base / "dicom"
        root.mkdir()
        file = root / "image.dcm"
        file.write_bytes(b"\0" * 128 + b"DICM" + b"synthetic header fixture")
        return root

    def converter(self, command, **kwargs):
        folder = Path(command[command.index("-o") + 1])
        nib.save(nib.Nifti1Image(np.ones((2, 3, 4, 3), dtype=np.float32), np.eye(4)), folder / "data.nii.gz")
        (folder / "data.bval").write_text("0 1000 1000\n")
        (folder / "data.bvec").write_text("0 1 0\n0 0 1\n0 0 0\n")
        (folder / "data.json").write_text('{"PhaseEncodingDirection":"j"}')
        return {"returncode": 0, "command": command}

    def test_only_positive_dicom_evidence_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self.assertFalse(is_dicom_root(base))
            (base / "README").write_text("not DICOM")
            (base / "wrong.dcm").write_text("not DICOM")
            self.assertFalse(is_dicom_root(base))
            root = self.source(base)
            self.assertTrue(is_dicom_root(root))
            self.assertEqual(inspect_subject_directory(root, "subject")["mode"], "dicom")
            self.assertFalse((root / "input").exists())

    def test_conversion_is_transactional_unchanged_source_and_verified_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.source(Path(tmp).resolve())
            original = (root / "image.dcm").read_bytes()
            with patch("shared.dicom_input.converter_path", return_value=("dcm2niix", Path("/converter"))), \
                    patch("shared.input_import._run", side_effect=self.converter) as run:
                record = prepare_dicom_root(root, "subject")
                self.assertEqual(record["status"], "completed")
                self.assertEqual(prepare_dicom_root(root, "subject"), record)
                self.assertEqual(run.call_count, 1)
            self.assertEqual((root / "image.dcm").read_bytes(), original)
            self.assertEqual(inspect_subject_directory(root, "subject")["mode"], "subject_root")
            receipt = json.loads((root / "input/input_import_manifest.json").read_text())
            self.assertEqual(receipt["outputs"]["dwi"], str(root / "input/data.nii.gz"))
            self.assertTrue((root / "input/bvals").is_file())
            (root / "input/bvals").write_text("changed")
            with self.assertRaisesRegex(ValueError, "converted DICOM input changed"):
                prepare_dicom_root(root, "subject")

    def test_conversion_failures_publish_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.source(Path(tmp))
            with patch("shared.dicom_input.converter_path", return_value=("dcm2niix", Path("/converter"))), \
                    patch("shared.input_import._run", side_effect=RuntimeError("conversion failed")):
                with self.assertRaisesRegex(RuntimeError, "conversion failed"):
                    prepare_dicom_root(root, "subject")
            self.assertFalse((root / "input").exists())
            self.assertFalse(list(root.parent.glob(".dds-dicom-*")))

    def test_ambiguous_series_and_mismatched_gradients_are_rejected(self):
        for bad in ("multiple", "gradients", "nonfinite", "image_nonfinite"):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as tmp:
                root = self.source(Path(tmp))
                def convert(command, **kwargs):
                    result = self.converter(command, **kwargs)
                    folder = Path(command[command.index("-o") + 1])
                    if bad == "multiple":
                        nib.save(nib.Nifti1Image(np.ones((2, 3, 4, 3)), np.eye(4)), folder / "other.nii.gz")
                        (folder / "other.bval").write_text("0 1000 1000")
                        (folder / "other.bvec").write_text("0 1 0\n0 0 1\n0 0 0")
                    elif bad == "image_nonfinite":
                        nib.save(nib.Nifti1Image(np.full((2, 3, 4, 3), np.nan), np.eye(4)), folder / "data.nii.gz")
                    else:
                        (folder / "data.bval").write_text("0 1000" if bad == "gradients" else "0 nan 1000")
                    return result
                with patch("shared.dicom_input.converter_path", return_value=("dcm2niix", Path("/converter"))), \
                        patch("shared.input_import._run", side_effect=convert), self.assertRaises(ValueError):
                    prepare_dicom_root(root, "subject")
                self.assertFalse((root / "input").exists())

    def test_existing_input_and_changed_source_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.source(Path(tmp))
            (root / "input").mkdir()
            with self.assertRaises(FileExistsError):
                prepare_dicom_root(root, "subject")
            (root / "input").rmdir()
            with patch("shared.dicom_input.converter_path", return_value=("dcm2niix", Path("/converter"))), \
                    patch("shared.input_import._run", side_effect=self.converter):
                prepare_dicom_root(root, "subject")
            (root / "image.dcm").write_bytes(b"\0" * 128 + b"DICMchanged")
            with self.assertRaisesRegex(ValueError, "source changed"):
                prepare_dicom_root(root, "subject")

    def test_only_execution_entry_converts_before_any_product_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = ProductOrchestratorConfig("subject", Path(tmp), ("validation",), execute=True)
            calls = []
            with patch("shared.dicom_input.prepare_dicom_root", side_effect=lambda *a, **k: calls.append("import")), \
                    patch("shared.orchestrator._run_subject_products_impl", side_effect=lambda *a, **k: calls.append("run") or "done"):
                self.assertEqual(run_subject_products(config), "done")
            self.assertEqual(calls, ["import", "run"])

    def test_invalid_subject_never_starts_conversion(self):
        with tempfile.TemporaryDirectory() as tmp, patch("shared.dicom_input.prepare_dicom_root") as convert:
            config = ProductOrchestratorConfig("", Path(tmp), ("validation",), execute=True)
            with self.assertRaises(ValueError):
                run_subject_products(config)
            convert.assert_not_called()

    def test_plan_only_never_converts_or_creates_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self.source(Path(tmp))
            config = ProductOrchestratorConfig("subject", root, ("validation",), execute=False)
            with patch("shared.dicom_input.prepare_dicom_root") as convert, \
                    patch("shared.dicom_input.converter_path", side_effect=FileNotFoundError("converter absent")):
                run_subject_products(config)
            convert.assert_not_called()
            self.assertFalse((root / "input").exists())


if __name__ == "__main__":
    unittest.main()
