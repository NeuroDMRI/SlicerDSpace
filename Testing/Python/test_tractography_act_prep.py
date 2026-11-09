#!/usr/bin/env python3
"""Tests for ACT 5TT/GMWMI preparation (partial volume + DDParcel sGM + MRtrix GMWMI)."""

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

from tractography.act_prep import (  # noqa: E402
    ACT_PREP_VERSION,
    DEFAULT_SGM_LABELS,
    DEFAULT_WM_LABELS,
    build_partial_volume_5tt,
)

RAS = np.diag([2.0, 2.0, 2.0, 1.0])
RAS[:3, 3] = (-5.0, -5.0, -5.0)


def _las_affine_for_same_grid(shape) -> np.ndarray:
    """Affine of the same world grid stored with the first axis reversed (LAS)."""

    las = RAS.copy()
    las[0, 0] = -2.0
    las[0, 3] = RAS[0, 3] + 2.0 * (shape[0] - 1)
    return las


def _fake_mrtrix(bin_dir: Path) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name in ("5ttcheck", "5tt2gmwmi"):
        (bin_dir / name).write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        (bin_dir / name).chmod(0o755)


def _fake_run(command, capture_output=True, text=True, **_kwargs):
    """Stand-in for MRtrix: 5tt2gmwmi writes a GMWMI copied from the 5TT grid."""

    if Path(command[0]).name == "5tt2gmwmi":
        img = nib.load(command[1])
        data = np.asarray(img.dataobj)
        gmwmi = ((data[..., 0] + data[..., 1]) > 0.1) & (data[..., 2] > 0.1)
        nib.save(nib.Nifti1Image(gmwmi.astype(np.float32), img.affine), command[2])
    return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")


class PartialVolumeChannelMappingTest(unittest.TestCase):
    def test_probabilities_are_renormalised_into_cgm_wm_csf(self) -> None:
        labels = np.array([[[1, 2, 3, 0]]], dtype=np.int16)
        probabilities = {
            "wm": np.array([[[0.6, 0.2, 0.0, 0.9]]]),
            "gm": np.array([[[0.2, 0.6, 0.1, 0.0]]]),
            "csf": np.array([[[0.1, 0.1, 0.8, 0.0]]]),
        }
        five_tt, stats = build_partial_volume_5tt(labels, probabilities)
        np.testing.assert_allclose(five_tt[0, 0, 0], [0.2 / 0.9, 0, 0.6 / 0.9, 0.1 / 0.9, 0], atol=1e-6)
        np.testing.assert_allclose(five_tt[0, 0, 2], [0.1 / 0.9, 0, 0, 0.8 / 0.9, 0], atol=1e-6)
        # Background stays empty even if the network produced mass there.
        np.testing.assert_array_equal(five_tt[0, 0, 3], np.zeros(5))
        np.testing.assert_allclose(five_tt[labels > 0].sum(axis=-1), 1.0, atol=1e-6)
        self.assertEqual(stats["background_nonzero_voxels"], 0)

    def test_ddparcel_deep_gm_goes_to_sgm_and_ventricles_to_csf(self) -> None:
        labels = np.array([[[1, 2, 2, 1, 3]]], dtype=np.int16)
        probabilities = {
            "wm": np.array([[[0.7, 0.3, 0.2, 0.8, 0.1]]]),
            "gm": np.array([[[0.2, 0.6, 0.7, 0.1, 0.1]]]),
            "csf": np.array([[[0.1, 0.1, 0.1, 0.1, 0.8]]]),
        }
        # thalamus (10), hippocampus (17), ventral DC (60), cortex (1030), lateral ventricle (4)
        parcellation = np.array([[[10, 17, 1030, 60, 4]]], dtype=np.int32)
        five_tt, stats = build_partial_volume_5tt(labels, probabilities, parcellation)
        for index in (0, 1):
            self.assertAlmostEqual(float(five_tt[0, 0, index, 1]), 0.9, places=6)  # sGM = 1 - P(CSF)
            self.assertEqual(float(five_tt[0, 0, index, 0]), 0.0)
            self.assertEqual(float(five_tt[0, 0, index, 2]), 0.0)
            self.assertAlmostEqual(float(five_tt[0, 0, index, 3]), 0.1, places=6)
        # Cortex keeps DDSeg partial volume.
        self.assertAlmostEqual(float(five_tt[0, 0, 2, 0]), 0.7, places=6)
        np.testing.assert_array_equal(five_tt[0, 0, 4], [0, 0, 0, 1, 0])
        np.testing.assert_allclose(five_tt[0, 0, 3], [0, 0, 0.9, 0.1, 0], atol=1e-6)
        np.testing.assert_allclose(five_tt.sum(axis=-1), 1.0, atol=1e-6)
        self.assertEqual(stats["sgm_override_voxels"], 2)
        self.assertEqual(stats["wm_prior_override_voxels"], 1)
        self.assertEqual(stats["ventricle_override_voxels"], 1)

    def test_default_sgm_labels_cover_requested_structures(self) -> None:
        for label in (10, 11, 12, 13, 17, 18, 26, 49, 50, 51, 52, 53, 54, 58):
            self.assertIn(label, DEFAULT_SGM_LABELS)
        self.assertEqual(set(DEFAULT_WM_LABELS), {7, 16, 28, 46, 60})
        self.assertFalse(set(DEFAULT_WM_LABELS) & set(DEFAULT_SGM_LABELS))

    def test_non_cortical_wm_prior_preserves_csf_and_foreground(self) -> None:
        parc = np.array([[[7, 16, 28, 46, 60, 16]]])
        labels = np.array([[[2, 2, 2, 2, 2, 0]]])
        probabilities = {"wm": np.full(labels.shape, 0.1),
                         "gm": np.full(labels.shape, 0.7),
                         "csf": np.full(labels.shape, 0.2)}
        five_tt, stats = build_partial_volume_5tt(labels, probabilities, parc)
        np.testing.assert_allclose(five_tt[0, 0, :5], np.tile([0, 0, 0.8, 0.2, 0], (5, 1)), atol=1e-6)
        np.testing.assert_array_equal(five_tt[0, 0, 5], np.zeros(5))
        self.assertEqual(stats["wm_prior_override_voxels"], 5)

    def test_conflicting_tissue_priors_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "disjoint"):
            build_partial_volume_5tt(np.ones((1, 1, 1)), None, sgm_labels=(28,))

    def test_hard_label_fallback_without_probabilities(self) -> None:
        labels = np.array([[[1, 2, 3, 0]]], dtype=np.int16)
        five_tt, _ = build_partial_volume_5tt(labels, None)
        np.testing.assert_array_equal(five_tt[0, 0, :, 0], [0, 1, 0, 0])
        np.testing.assert_array_equal(five_tt[0, 0, :, 2], [1, 0, 0, 0])
        np.testing.assert_array_equal(five_tt[0, 0, :, 3], [0, 0, 1, 0])


class ACTPrepRunTest(unittest.TestCase):
    def _write_ddseg(self, root: Path, shape=(5, 4, 3)):
        labels = np.zeros(shape, dtype=np.float32)
        labels[1:4, 1:3, :] = 1
        labels[0, 1:3, :] = 2
        labels[4, 1:3, :] = 2
        labels[2, 0, 0] = 3
        wm = (labels == 1) * 0.8 + (labels == 2) * 0.1 + (labels == 3) * 0.05
        gm = (labels == 1) * 0.15 + (labels == 2) * 0.8 + (labels == 3) * 0.05
        csf = (labels == 1) * 0.05 + (labels == 2) * 0.1 + (labels == 3) * 0.9
        seg_dir = root / "ddseg"
        seg_dir.mkdir()
        for name, data in (
            ("SegmentationMap_GMWMCSF", labels),
            ("ProbabilisticMap_WM", wm),
            ("ProbabilisticMap_GM", gm),
            ("ProbabilisticMap_CSF", csf),
        ):
            nib.save(nib.Nifti1Image(data.astype(np.float32), RAS), str(seg_dir / f"{name}.nii.gz"))
        return seg_dir / "SegmentationMap_GMWMCSF.nii.gz", labels

    def test_execute_builds_pv_5tt_runs_5ttcheck_and_5tt2gmwmi(self) -> None:
        from tractography import ACTPrepConfig, run_act_prep

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            segmentation, labels = self._write_ddseg(root)
            _fake_mrtrix(root / "bin")
            with mock.patch("tractography.act_prep.subprocess.run", side_effect=_fake_run) as mock_run:
                result = run_act_prep(
                    ACTPrepConfig(
                        subject_id="100307",
                        segmentation_path=segmentation,
                        output_dir=root / "act_prep",
                        execute=True,
                        mrtrix_bin_dir=root / "bin",
                        anatomy_source={"ddseg_source_dwi": "/x/dwi_final.nii.gz"},
                        allow_ddseg_only=True,
                    )
                )
            invoked = [Path(call.args[0][0]).name for call in mock_run.call_args_list]
            self.assertEqual(invoked, ["5ttcheck", "5tt2gmwmi"])
            five_tt_img = nib.load(result.five_tt_path.as_posix())
            self.assertEqual(five_tt_img.shape, (*labels.shape, 5))
            np.testing.assert_allclose(five_tt_img.affine, RAS)
            data = np.asarray(five_tt_img.dataobj)
            np.testing.assert_allclose(data[labels > 0].sum(axis=-1), 1.0, atol=1e-5)
            self.assertTrue(((data > 0) & (data < 1)).any(), "expected partial-volume values")
            qc = json.loads(result.qc_path.read_text(encoding="utf-8"))
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(qc["status"], "completed")
            self.assertEqual(qc["version"], ACT_PREP_VERSION)
            self.assertEqual(qc["mode"], "partial_volume")
            self.assertFalse(qc["sgm_from_ddparcel"])
            self.assertGreater(qc["gmwmi_voxels"], 0)
            self.assertEqual(manifest["anatomy_source"]["ddseg_source_dwi"], "/x/dwi_final.nii.gz")
            self.assertTrue(any("DDSeg-only" in warning for warning in manifest["warnings"]))
            self.assertEqual(qc["anatomical_prior_status"], "ddseg_only")
            self.assertEqual(qc["warnings"], manifest["warnings"])
            self.assertTrue(qc["format_validated"])
            self.assertFalse(qc["pathology_channel"]["modelled"])

    def test_ddparcel_is_resampled_through_affines_not_array_indices(self) -> None:
        from tractography import ACTPrepConfig, run_act_prep

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            segmentation, labels = self._write_ddseg(root)
            # DDParcel on the same world grid but LAS-stored: thalamus at world voxel x=1.
            parcellation_ras = np.zeros(labels.shape, dtype=np.int32)
            parcellation_ras[1, 1:3, :] = 10
            parcellation_las = parcellation_ras[::-1, :, :]
            parcellation_path = root / "wmparc.nii.gz"
            nib.save(nib.Nifti1Image(parcellation_las, _las_affine_for_same_grid(labels.shape)), str(parcellation_path))
            _fake_mrtrix(root / "bin")
            with mock.patch("tractography.act_prep.subprocess.run", side_effect=_fake_run):
                result = run_act_prep(
                    ACTPrepConfig(
                        subject_id="100307",
                        segmentation_path=segmentation,
                        output_dir=root / "act_prep",
                        execute=True,
                        parcellation_path=parcellation_path,
                        mrtrix_bin_dir=root / "bin",
                    )
                )
            data = np.asarray(nib.load(result.five_tt_path.as_posix()).dataobj)
            sgm = data[..., 1] > 0.5
            np.testing.assert_array_equal(sgm, parcellation_ras == 10)
            # Raw indexing would have put the thalamus at x=3 instead.
            self.assertFalse(sgm[3].any())
            qc = json.loads(result.qc_path.read_text(encoding="utf-8"))
            self.assertTrue(qc["sgm_from_ddparcel"])
            self.assertEqual(qc["anatomical_prior_status"], "complete")

    def test_missing_ddparcel_is_rejected_before_mrtrix_and_clears_old_completion(self):
        from tractography import ACTPrepConfig, run_act_prep

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            segmentation, _ = self._write_ddseg(root)
            output = root / "act_prep"
            output.mkdir()
            (output / "act_prep_qc.json").write_text('{"status":"completed"}')
            with mock.patch("tractography.act_prep.subprocess.run") as run:
                with self.assertRaisesRegex(ValueError, "requires DDParcel"):
                    run_act_prep(ACTPrepConfig("subject", segmentation, output, execute=True))
            run.assert_not_called()
            for name in ("act_prep_qc.json", "act_prep_manifest.json"):
                self.assertEqual(json.loads((output / name).read_text())["status"], "failed")

    def test_supplied_parcellation_without_sgm_is_not_accepted_as_complete(self):
        from tractography import ACTPrepConfig, run_act_prep

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            segmentation, labels = self._write_ddseg(root)
            parcellation = root / "cortex_only.nii.gz"
            nib.save(nib.Nifti1Image(np.where(labels > 0, 1001, 0).astype(np.int16), RAS), parcellation)
            _fake_mrtrix(root / "bin")
            with mock.patch("tractography.act_prep.subprocess.run") as run:
                with self.assertRaisesRegex(ValueError, "no subcortical GM"):
                    run_act_prep(ACTPrepConfig("subject", segmentation, root / "act_prep",
                        parcellation_path=parcellation, mrtrix_bin_dir=root / "bin", execute=True))
            run.assert_not_called()

    def test_missing_mrtrix_fails_loudly(self) -> None:
        from tractography import ACTPrepConfig, run_act_prep

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            segmentation, _ = self._write_ddseg(root)
            with mock.patch("shared.external_process.shutil.which", return_value=None):
                with self.assertRaisesRegex(RuntimeError, "5ttcheck"):
                    run_act_prep(
                        ACTPrepConfig(
                            subject_id="100307",
                            segmentation_path=segmentation,
                            output_dir=root / "act_prep",
                            execute=True,
                            mrtrix_bin_dir=root / "no_bin",
                            allow_ddseg_only=True,
                        )
                    )

    def test_failed_5ttcheck_does_not_leave_a_completed_record(self) -> None:
        from tractography import ACTPrepConfig, run_act_prep

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            segmentation, _ = self._write_ddseg(root)
            _fake_mrtrix(root / "bin")
            output_dir = root / "act_prep"
            output_dir.mkdir()
            (output_dir / "act_prep_qc.json").write_text('{"status": "completed", "version": 2}', encoding="utf-8")
            failing = subprocess.CompletedProcess(["5ttcheck"], 1, stdout="", stderr="bad 5TT")
            with mock.patch("tractography.act_prep.subprocess.run", return_value=failing):
                with self.assertRaisesRegex(RuntimeError, "5ttcheck"):
                    run_act_prep(
                        ACTPrepConfig(
                            subject_id="100307",
                            segmentation_path=segmentation,
                            output_dir=output_dir,
                            execute=True,
                            mrtrix_bin_dir=root / "bin",
                            allow_ddseg_only=True,
                        )
                    )
            qc = json.loads((output_dir / "act_prep_qc.json").read_text(encoding="utf-8"))
            self.assertNotEqual(qc["status"], "completed")


if __name__ == "__main__":
    unittest.main()
