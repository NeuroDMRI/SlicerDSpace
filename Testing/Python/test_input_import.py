#!/usr/bin/env python3
"""Tests for external input import planning."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

from shared.input_import import InputImportConfig, detect_import_source_kind, prepare_inputs, _lossless_signed_nrrd


class InputImportTest(unittest.TestCase):
    def test_detects_canonical_subject_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "sub-01"
            input_dir = subject_root / "input"
            input_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs"):
                (input_dir / name).write_text("x\n", encoding="utf-8")

            self.assertEqual(detect_import_source_kind(subject_root), "canonical")

    def test_nifti_source_is_copied_to_canonical_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "dwi.nii.gz"
            source_data = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)
            nib.save(nib.Nifti1Image(source_data, np.eye(4)), str(source))
            (root / "dwi.bval").write_text("0 1000\n", encoding="utf-8")
            (root / "dwi.bvec").write_text("0 1\n0 0\n0 0\n", encoding="utf-8")
            (root / "dwi.json").write_text('{"PhaseEncodingDirection": "j"}\n', encoding="utf-8")
            subject_root = root / "prepared" / "sub-01"

            manifest = prepare_inputs(
                InputImportConfig(
                    source=source,
                    subject_root=subject_root,
                    subject_id="sub-01",
                    source_kind="nifti",
                )
            )

            self.assertEqual(manifest["status"], "completed")
            imported = nib.load(str(subject_root / "input" / "data.nii.gz"))
            np.testing.assert_array_equal(np.asarray(imported.dataobj), source_data)
            self.assertTrue((subject_root / "input" / "bvals").is_file())
            self.assertTrue((subject_root / "input" / "bvecs").is_file())
            self.assertEqual(json.loads((subject_root / "input" / "dwi.json").read_text())["PhaseEncodingDirection"], "j")
            self.assertTrue((subject_root / "input" / "input_import_manifest.json").is_file())

    def test_dicom_import_prefers_dcm2niix_for_canonical_nifti(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dicom_dir = root / "dicom"
            dicom_dir.mkdir()
            (dicom_dir / "IM0001.dcm").write_text("dicom\n", encoding="utf-8")

            with patch("shared.input_import.shutil.which", return_value="/tools/dcm2niix"):
                manifest = prepare_inputs(
                    InputImportConfig(
                        source=dicom_dir,
                        subject_root=root / "sub-01",
                        subject_id="sub-01",
                        source_kind="dicom",
                        dry_run=True,
                    )
                )

            plan = manifest["plan"]
            self.assertEqual(manifest["status"], "planned")
            self.assertEqual(plan["backend"], "dcm2niix")
            self.assertEqual(plan["command"][plan["command"].index("-ba") + 1], "y")
            self.assertIn("-f", plan["command"])
            self.assertIn("data", plan["command"])

    def test_dicom_import_falls_back_to_slicer_dwiconvert(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dicom_dir = root / "dicom"
            dicom_dir.mkdir()
            (dicom_dir / "IM0001.dcm").write_text("dicom\n", encoding="utf-8")

            def which(name: str) -> str | None:
                return "/slicer/DWIConvert" if name == "DWIConvert" else None

            with patch("shared.input_import.packaged_dcm2niix", return_value=None), \
                    patch("shared.input_import.shutil.which", side_effect=which):
                manifest = prepare_inputs(
                    InputImportConfig(
                        source=dicom_dir,
                        subject_root=root / "sub-01",
                        subject_id="sub-01",
                        source_kind="dicom",
                        dry_run=True,
                    )
                )

            plan = manifest["plan"]
            self.assertEqual(plan["backend"], "dwiconvert")
            self.assertIn("DicomToFSL", plan["command"])
            self.assertIn("--outputNiftiFile", plan["command"])

    def test_nrrd_import_uses_dwiconvert_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nrrd = root / "data.nhdr"
            nrrd.write_text("NRRD0005\n", encoding="utf-8")

            manifest = prepare_inputs(
                InputImportConfig(
                    source=nrrd,
                    subject_root=root / "sub-01",
                    subject_id="sub-01",
                    source_kind="nrrd",
                    dry_run=True,
                )
            )

            plan = manifest["plan"]
            self.assertEqual(plan["backend"], "dwiconvert")
            self.assertIn("NrrdToFSL", plan["command"])
            self.assertIn("--outputNiftiFile", plan["command"])
            self.assertTrue(plan["canonical_layout"]["dwi"].endswith("input/data.nii.gz"))
            self.assertFalse((root / 'sub-01').exists(), 'Import preview must not create a subject directory')

    def test_nrrd_import_can_force_mrconvert(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nrrd = root / "data.nhdr"
            nrrd.write_text("NRRD0005\n", encoding="utf-8")

            manifest = prepare_inputs(
                InputImportConfig(
                    source=nrrd,
                    subject_root=root / "sub-01",
                    subject_id="sub-01",
                    source_kind="nrrd",
                    backend="mrconvert",
                    dry_run=True,
                )
            )

            plan = manifest["plan"]
            self.assertEqual(plan["backend"], "mrconvert")
            self.assertIn("-export_grad_fsl", plan["command"])
            self.assertNotIn("-force", plan["command"])

    def test_converter_never_overwrites_existing_inputs_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'dwi.nhdr'
            source.write_text('NRRD0005\n')
            input_dir = root / 'subject' / 'input'
            input_dir.mkdir(parents=True)
            existing = input_dir / 'bvals'
            existing.write_text('original')
            config = InputImportConfig(source=source, subject_root=root / 'subject', subject_id='s',
                                       source_kind='nrrd', backend='mrconvert')
            with patch('shared.input_import._run') as execute, self.assertRaises(FileExistsError):
                prepare_inputs(config)
            execute.assert_not_called()
            self.assertEqual(existing.read_text(), 'original')

    def test_unsigned_nrrd_bridge_preserves_values_and_metadata(self) -> None:
        import nrrd
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = np.arange(24, dtype=np.uint16).reshape(3, 2, 2, 2)
            header = {'space': 'right-anterior-superior', 'kinds': ['list', 'domain', 'domain', 'domain'],
                      'space directions': np.array([[np.nan] * 3, [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                      'space origin': np.zeros(3), 'measurement frame': np.eye(3),
                      'DWMRI_b-value': '1000',
                      **{f'DWMRI_gradient_{i:04d}': value for i, value in enumerate(['0 0 0', '1 0 0', '0 1 0'])}}
            source, signed = root / 'source.nrrd', root / 'signed.nrrd'
            nrrd.write(str(source), data, header=header)
            receipt = _lossless_signed_nrrd(source, signed)
            converted, info = nrrd.read(str(signed))
            np.testing.assert_array_equal(data, converted)
            self.assertEqual(converted.dtype, np.int16)
            self.assertEqual(info['DWMRI_gradient_0001'], '1 0 0')
            self.assertFalse(receipt['lossy_conversion'])
            data[0, 0, 0, 0] = 65535
            nrrd.write(str(source), data, header=header)
            with self.assertRaisesRegex(ValueError, 'refusing lossy'):
                _lossless_signed_nrrd(source, root / 'must-not-exist.nrrd')
            self.assertFalse((root / 'must-not-exist.nrrd').exists())

    def test_dicom_to_compat_nrrd_uses_dwiconvert(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dicom_dir = root / "dicom"
            dicom_dir.mkdir()
            (dicom_dir / "IM0001").write_text("dicom\n", encoding="utf-8")

            manifest = prepare_inputs(
                InputImportConfig(
                    source=dicom_dir,
                    subject_root=root / "sub-01",
                    subject_id="sub-01",
                    source_kind="dicom",
                    output_kind="compat-nrrd",
                    dry_run=True,
                )
            )

            plan = manifest["plan"]
            self.assertEqual(plan["backend"], "dwiconvert")
            self.assertIn("DicomToNrrd", plan["command"])
            self.assertTrue(plan["canonical_layout"]["compat_nrrd"].endswith("cache/imported_nrrd/data.nhdr"))


if __name__ == "__main__":
    unittest.main()
