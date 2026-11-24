#!/usr/bin/env python3
"""Tests for external input import planning."""

from __future__ import annotations

import json
import shutil
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

            import os
            from shared.diffusion_io import get_cached_uncompressed_nifti

            with patch.dict(os.environ, {"SLICERDSPACE_UNCOMPRESSED_CACHE_MIN_BYTES": "1"}):
                manifest = prepare_inputs(
                    InputImportConfig(
                        source=source,
                        subject_root=subject_root,
                        subject_id="sub-01",
                        source_kind="nifti",
                    )
                )
                cached_nii = get_cached_uncompressed_nifti(subject_root / "input" / "data.nii.gz")
                self.assertIsNotNone(cached_nii)
                self.assertTrue(cached_nii.is_file())
                np.testing.assert_array_equal(np.asarray(nib.load(str(cached_nii), mmap="r").dataobj), source_data)

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

    def test_force_import_reuses_unchanged_inputs_and_overwrites_on_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "dwi.nii.gz"
            data1 = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)
            nib.save(nib.Nifti1Image(data1, np.eye(4)), str(source))
            (root / "dwi.bval").write_text("0 1000\n", encoding="utf-8")
            (root / "dwi.bvec").write_text("0 1\n0 0\n0 0\n", encoding="utf-8")
            subject_root = root / "out" / "sub-01"

            cfg = InputImportConfig(
                source=source,
                subject_root=subject_root,
                subject_id="sub-01",
                source_kind="nifti",
                force=True,
            )
            first = prepare_inputs(cfg)
            self.assertEqual(first["status"], "completed")
            self.assertFalse(first.get("reused", False))
            target_dwi = subject_root / "input" / "data.nii.gz"
            first_mtime = target_dwi.stat().st_mtime_ns

            # An auto-generated brain mask + manifest inside input/ must not invalidate reuse.
            (subject_root / "input" / "nodif_brain_mask.nii.gz").write_text("mask", encoding="utf-8")
            (subject_root / "input" / "nodif_brain_mask.manifest.json").write_text('{"version": 2}\n', encoding="utf-8")

            second = prepare_inputs(cfg)
            self.assertTrue(second.get("reused"))
            self.assertEqual(target_dwi.stat().st_mtime_ns, first_mtime)

            # Updating source overwrites cleanly even when target already exists.
            data2 = data1 + 10.0
            nib.save(nib.Nifti1Image(data2, np.eye(4)), str(source))
            third = prepare_inputs(cfg)
            self.assertFalse(third.get("reused", False))
            np.testing.assert_array_equal(np.asarray(nib.load(str(target_dwi)).dataobj), data2)

    def test_nifti_gz_prefers_cow_reflink_and_preserves_archive_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "dwi.nii.gz"
            data1 = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)
            data2 = np.full((2, 2, 2, 2), 99.0, dtype=np.float32)
            nib.save(nib.Nifti1Image(data1, np.eye(4)), str(source))
            (root / "dwi.bval").write_text("0 1000\n", encoding="utf-8")
            (root / "dwi.bvec").write_text("0 1\n0 0\n0 0\n", encoding="utf-8")

            # Fallback copy path must isolate staged input from later in-place edits to source.
            copied_root = root / "copied" / "sub-01"
            with patch("shared.input_import._try_cow_clone", return_value=False):
                prepare_inputs(InputImportConfig(source=source, subject_root=copied_root, subject_id="sub-01", source_kind="nifti"))
            nib.save(nib.Nifti1Image(data2, np.eye(4)), str(source))
            np.testing.assert_array_equal(np.asarray(nib.load(str(copied_root / "input" / "data.nii.gz")).dataobj), data1)

            # When CoW reflink succeeds, shutil.copyfile must not be called for data.nii.gz.
            def fake_clone(src: Path, dst: Path) -> bool:
                dst.write_bytes(src.read_bytes())
                return True

            cloned_root = root / "cloned" / "sub-01"
            with (
                patch("shared.input_import._try_cow_clone", side_effect=fake_clone) as mock_clone,
                patch("shared.input_import.shutil.copyfile", wraps=shutil.copyfile) as mock_copyfile,
            ):
                prepare_inputs(InputImportConfig(source=source, subject_root=cloned_root, subject_id="sub-01", source_kind="nifti"))
            self.assertEqual(mock_clone.call_count, 1)
            copied_sources = [Path(call.args[0]).resolve() for call in mock_copyfile.call_args_list]
            self.assertNotIn(source.resolve(), copied_sources)
            np.testing.assert_array_equal(np.asarray(nib.load(str(cloned_root / "input" / "data.nii.gz")).dataobj), data2)


if __name__ == "__main__":
    unittest.main()
