#!/usr/bin/env python3
"""Tests for the bundled DDSurfer postprocess wrapper."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import nibabel as nib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class SurfacePostprocessReleaseTest(unittest.TestCase):
    def test_windows_prefers_configured_slicer_bash_bridge(self) -> None:
        from anatomy.surface.postprocess_release import _resolve_bash_executable

        with tempfile.TemporaryDirectory() as tmp:
            slicer_root = Path(tmp) / "Slicer 5.6.2"
            python_bin = slicer_root / "bin" / "PythonSlicer.exe"
            python_bin.parent.mkdir(parents=True)
            python_bin.write_bytes(b"python")
            (slicer_root / "bash.exe").write_bytes(b"bridge")
            (slicer_root / "bash.exe.conf").write_text("target=/bin/bash\n", encoding="utf-8")

            self.assertEqual(
                _resolve_bash_executable(python_bin, platform="nt"),
                str(slicer_root / "bash.exe"),
            )

    def test_extract_ctab_preserves_annotation_ids_and_rgb(self) -> None:
        from anatomy.surface.ddsurfer_postprocess_release.src.extract_ctab import extract_and_save_ctab

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annot_path = root / "lh.synthetic.annot"
            ctab_path = root / "lh.synthetic.ctab"
            input_ctab = np.asarray(
                [
                    [12, 34, 56, 0],
                    [78, 90, 123, 17],
                ],
                dtype=np.int32,
            )
            names = [b"region_one", b"region_two"]
            nib.freesurfer.io.write_annot(
                str(annot_path),
                np.asarray([0, 1, 0, 1], dtype=np.int32),
                input_ctab,
                names,
            )
            _, expected_ctab, expected_names = nib.freesurfer.io.read_annot(str(annot_path))

            self.assertTrue(extract_and_save_ctab(str(annot_path), str(ctab_path)))
            rows = {}
            for line in ctab_path.read_text(encoding="utf-8").splitlines():
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                rows[parts[1]] = tuple(int(value) for value in (parts[0], *parts[-4:]))

            for index, raw_name in enumerate(expected_names):
                name = raw_name.decode("utf-8")
                expected = expected_ctab[index]
                self.assertEqual(rows[name][0], int(expected[4]))
                self.assertEqual(rows[name][1:4], tuple(int(value) for value in expected[:3]))
                self.assertEqual(rows[name][4], 255 - int(expected[3]))

    def test_regional_stats_accepts_legacy_duplicate_annotation_ids(self) -> None:
        from anatomy.surface.ddsurfer_postprocess_release.src.compute_regional_stats import main as compute_stats
        from anatomy.surface.ddsurfer_postprocess_release.src.extract_ctab import extract_and_save_ctab

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annot_path = root / "lh.legacy.annot"
            ctab_path = root / "lh.legacy.ctab"
            stats_path = root / "lh.legacy.stats"
            nib.freesurfer.io.write_annot(
                str(annot_path),
                np.asarray([0, 1, 2, 1, 2, 0, 3, 4], dtype=np.int32),
                np.asarray(
                    [
                        [10, 20, 30, 0],
                        [200, 210, 220, 0],
                        [200, 210, 220, 0],
                        [1, 2, 3, 0],
                        [4, 5, 6, 0],
                    ],
                    dtype=np.int32,
                ),
                [
                    b"unique_region",
                    b"legacy_alias_one",
                    b"legacy_alias_two",
                    b"???",
                    b"FreeSurfer_Defined_Medial_Wall",
                ],
            )
            self.assertTrue(extract_and_save_ctab(str(annot_path), str(ctab_path)))

            argv = [
                "compute_regional_stats.py",
                "--annot_file",
                str(annot_path),
                "--ctab_file",
                str(ctab_path),
                "--hemi",
                "lh",
                "--out_stats_file",
                str(stats_path),
            ]
            with mock.patch.object(sys, "argv", argv):
                compute_stats()

            rows = [
                line
                for line in stats_path.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")
            ]
            self.assertEqual(len(rows), 2)
            self.assertFalse(any(line.startswith("??? ") for line in rows))
            self.assertFalse(any(line.startswith("FreeSurfer_Defined_Medial_Wall ") for line in rows))

    def test_stage_inputs_respects_left_mode(self) -> None:
        from anatomy.surface.postprocess_release import stage_postprocess_inputs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native_dir = root / "native"
            mri_dir = root / "mri"
            native_dir.mkdir()
            mri_dir.mkdir()
            for name in ("brain.mgz", "DDSurfer-wmparc.nii.gz"):
                (mri_dir / name).write_text(name, encoding="utf-8")
            for name in (
                "877168_predicted_wm_lh_orig.stl",
                "877168_predicted_pial_lh_orig.stl",
                "877168_predicted_wm_rh_orig.stl",
                "877168_predicted_pial_rh_orig.stl",
            ):
                (native_dir / name).write_text(name, encoding="utf-8")

            staged = stage_postprocess_inputs(
                subject_id="877168",
                native_dir=native_dir,
                wmparc_path=mri_dir / "DDSurfer-wmparc.nii.gz",
                brain_mgz_path=mri_dir / "brain.mgz",
                staging_inputs_root=root / "staging",
                mode="left",
                force=False,
            )

            surf_names = sorted(path.name for path in (staged / "surf").iterdir())
            self.assertEqual(
                surf_names,
                ["877168_predicted_pial_lh_orig.stl", "877168_predicted_wm_lh_orig.stl"],
            )
            for staged_file in (*((staged / "surf").iterdir()), *((staged / "mri").iterdir())):
                self.assertTrue(staged_file.is_file())
                self.assertFalse(staged_file.is_symlink())

    def test_run_returns_canonical_paths(self) -> None:
        from anatomy.surface.postprocess_release import DDSurferPostprocessConfig, run_ddsurfer_postprocess

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            native_dir = root / "native"
            mri_dir = root / "mri"
            backend_root = root / "backend"
            output_root = root / "outputs"
            native_dir.mkdir()
            mri_dir.mkdir()
            (backend_root / "config").mkdir(parents=True)

            for name in ("brain.mgz", "DDSurfer-wmparc.nii.gz"):
                (mri_dir / name).write_text(name, encoding="utf-8")
            for name in (
                "877168_predicted_wm_lh_orig.stl",
                "877168_predicted_pial_lh_orig.stl",
            ):
                (native_dir / name).write_text(name, encoding="utf-8")

            run_sh = backend_root / "run.sh"
            run_sh.write_bytes(
                """#!/usr/bin/env bash
set -euo pipefail
subject_id=""
outputs_dir=""
nested="${NEST_SUBJECT_OUTPUTS:-1}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --subject-id) subject_id="$2"; shift 2 ;;
    --outputs-dir) outputs_dir="$2"; shift 2 ;;
    --nest-subject-outputs) nested="$2"; shift 2 ;;
    *) shift ;;
  esac
done
if [[ "$nested" = "0" ]]; then
  structural_root="${outputs_dir}/${SLICERDSPACE_ANATOMY_DIRNAME:-03_structural}/ddsurfer/postprocess"
  stats_root="${outputs_dir}/05_statistics/ddsurfer/postprocess/stats"
else
  structural_root="${outputs_dir}/${subject_id}/${SLICERDSPACE_ANATOMY_DIRNAME:-03_structural}/ddsurfer/postprocess"
  stats_root="${outputs_dir}/${subject_id}/05_statistics/ddsurfer/postprocess/stats"
fi
mkdir -p "${structural_root}/surf" "${structural_root}/label" "${stats_root}"
printf '{"kind":"ddsurfer_postprocess","stage_status":{"sync_outputs":"completed"}}' > "${structural_root}/manifest.json"
printf '{"status":"completed"}' > "${structural_root}/qc.json"
""".encode("utf-8")
            )
            run_sh.chmod(0o755)

            result = run_ddsurfer_postprocess(
                DDSurferPostprocessConfig(
                    subject_id="877168",
                    native_dir=native_dir,
                    mri_dir=mri_dir,
                    output_root=output_root,
                    backend_root=backend_root,
                    mode="left",
                )
            )

            self.assertEqual(
                result.structural_root,
                output_root / "03_anatomy" / "ddsurfer" / "postprocess",
            )
            self.assertEqual(
                result.stats_root,
                output_root / "05_statistics" / "ddsurfer" / "postprocess",
            )
            self.assertTrue(result.manifest_path.is_file())
            self.assertTrue(result.qc_path.is_file())
            self.assertTrue((result.staged_input_dir / "surf" / "877168_predicted_wm_lh_orig.stl").exists())
            self.assertTrue((output_root / "03_anatomy").is_dir())
            self.assertFalse((output_root / "03_structural").exists())
            self.assertTrue((output_root / "05_statistics").is_dir())
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            qc = json.loads(result.qc_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["kind"], "ddsurfer_postprocess")
            self.assertEqual(qc["status"], "completed")

    def test_run_generates_brain_mgz_from_subject_fa_when_missing(self) -> None:
        from anatomy.surface.postprocess_release import DDSurferPostprocessConfig, run_ddsurfer_postprocess

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject_root = root / "100307"
            output_root = subject_root / "outputs"
            native_dir = output_root / "03_structural" / "ddsurfer" / "pred_results_SlicerDSpace" / "native" / "100307"
            mri_dir = output_root / "03_structural" / "ddparcel"
            dti_dir = output_root / "02_microstructure" / "dti"
            backend_root = root / "backend"
            native_dir.mkdir(parents=True)
            mri_dir.mkdir(parents=True)
            dti_dir.mkdir(parents=True)
            (backend_root / "config").mkdir(parents=True)

            for name in (
                "100307_predicted_wm_lh_orig.stl",
                "100307_predicted_pial_lh_orig.stl",
            ):
                (native_dir / name).write_text(name, encoding="utf-8")
            (mri_dir / "100307-DDSurfer-wmparc.nii.gz").write_text("wmparc", encoding="utf-8")
            fa = nib.Nifti1Image(np.ones((8, 9, 10), dtype=np.float32), np.diag([-1.25, 1.25, 1.25, 1.0]))
            nib.save(fa, dti_dir / "100307_dti_FA.nii.gz")

            run_sh = backend_root / "run.sh"
            run_sh.write_bytes(
                """#!/usr/bin/env bash
set -euo pipefail
subject_id=""
outputs_dir=""
nested="1"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --subject-id) subject_id="$2"; shift 2 ;;
    --outputs-dir) outputs_dir="$2"; shift 2 ;;
    --nest-subject-outputs) nested="$2"; shift 2 ;;
    *) shift ;;
  esac
done
if [[ "$nested" = "0" ]]; then
  structural_root="${outputs_dir}/03_structural/ddsurfer/postprocess"
  stats_root="${outputs_dir}/05_statistics/ddsurfer/postprocess/stats"
else
  structural_root="${outputs_dir}/${subject_id}/03_structural/ddsurfer/postprocess"
  stats_root="${outputs_dir}/${subject_id}/05_statistics/ddsurfer/postprocess/stats"
fi
mkdir -p "${structural_root}/surf" "${structural_root}/label" "${stats_root}"
printf '{"kind":"ddsurfer_postprocess","stage_status":{"sync_outputs":"completed"}}' > "${structural_root}/manifest.json"
printf '{"status":"completed"}' > "${structural_root}/qc.json"
""".encode("utf-8")
            )
            run_sh.chmod(0o755)

            with mock.patch("anatomy.surface.postprocess_release.FS_BRAIN_TEMPLATE_SHAPE", (32, 32, 32)), mock.patch(
                "anatomy.surface.postprocess_release.FS_BRAIN_TEMPLATE_AFFINE",
                np.eye(4, dtype=np.float32),
            ):
                result = run_ddsurfer_postprocess(
                    DDSurferPostprocessConfig(
                        subject_id="100307",
                        subject_root=subject_root,
                        output_root=output_root,
                        backend_root=backend_root,
                        mode="left",
                    )
                )

            generated = output_root / "03_anatomy" / "cache" / "ddsurfer_postprocess_runtime" / "generated_mri" / "100307" / "brain.mgz"
            staged = result.staged_input_dir / "mri" / "brain.mgz"
            self.assertTrue(generated.is_file())
            self.assertTrue(staged.exists())
            generated_img = nib.load(str(generated))
            self.assertEqual(generated_img.shape, (32, 32, 32))


if __name__ == "__main__":
    unittest.main()
