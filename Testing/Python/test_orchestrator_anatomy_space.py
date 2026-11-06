#!/usr/bin/env python3
"""Orchestrator: one space for anatomy and tractography, QC gate, resume invalidation."""

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


def _record(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": path.resolve().as_posix(), "exists": True, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _subject(tmp: str) -> Path:
    subject_root = Path(tmp).resolve() / "100307"
    input_dir = subject_root / "input"
    input_dir.mkdir(parents=True)
    for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
        (input_dir / name).write_text("input", encoding="utf-8")
    return subject_root


def _complete_preprocessing(subject_root: Path) -> Path:
    preproc = subject_root / "outputs" / "01_preproc"
    preproc.mkdir(parents=True)
    for name in ("dwi_final.nii.gz", "dwi_final.bval", "dwi_final.bvec", "brain_mask_final.nii.gz"):
        (preproc / f"100307_{name}").write_text("preproc", encoding="utf-8")
    (preproc / "100307_preproc_manifest.json").write_text('{"status": "completed"}', encoding="utf-8")
    return preproc


def _ddseg_qc(schema_version: int, convention: str) -> dict[str, object]:
    return {
        "status": "completed",
        "schema_version": schema_version,
        "orientation_convention": convention,
        "probability_semantics": "four_class_softmax_tissue_channels_v1",
        "probability_qc": {"finite_and_bounded": True, "tissue_mass_valid_fraction": 1.0, "label_agreement": 1.0},
    }


class AnatomyInputBundleTest(unittest.TestCase):
    def test_anatomy_rejects_preprocessing_from_changed_raw_input(self) -> None:
        from shared.orchestrator import _anatomy_input_bundle

        with tempfile.TemporaryDirectory() as tmp:
            subject = _subject(tmp)
            preproc = _complete_preprocessing(subject)
            manifest = preproc / "100307_preproc_manifest.json"
            inputs = {key: _record(subject / "input" / name) for key, name in
                      {"dwi": "data.nii.gz", "bval": "bvals", "bvec": "bvecs", "mask": "nodif_brain_mask.nii.gz"}.items()}
            manifest.write_text(json.dumps({"status": "completed", "cache_key": {"inputs": inputs}}))
            self.assertEqual(_anatomy_input_bundle(subject, "100307", None, None).input_dir, preproc)
            (subject / "input/bvals").write_text("changed gradients")
            self.assertEqual(_anatomy_input_bundle(subject, "100307", None, None).input_dir, subject / "input")

    def test_anatomy_rejects_malformed_preprocessing_cache_fingerprint(self) -> None:
        from shared.orchestrator import _anatomy_input_bundle

        with tempfile.TemporaryDirectory() as tmp:
            subject = _subject(tmp)
            preproc = _complete_preprocessing(subject)
            (preproc / "100307_preproc_manifest.json").write_text(json.dumps({"status": "completed", "cache_key": None}))
            self.assertEqual(_anatomy_input_bundle(subject, "100307", None, None).input_dir, subject / "input")

    def test_anatomy_uses_preprocessed_dwi_when_preprocessing_completed(self) -> None:
        from shared.orchestrator import _anatomy_input_bundle

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = _subject(tmp)
            raw = _anatomy_input_bundle(subject_root, "100307", None, None)
            self.assertEqual(raw.dwi_path, subject_root / "input" / "data.nii.gz")
            preproc = _complete_preprocessing(subject_root)
            bundle = _anatomy_input_bundle(subject_root, "100307", None, None)
            self.assertEqual(bundle.dwi_path, preproc / "100307_dwi_final.nii.gz")
            self.assertEqual(bundle.mask_path, preproc / "100307_brain_mask_final.nii.gz")

    def test_preprocessed_features_get_their_own_cache_directory(self) -> None:
        from shared.orchestrator import _anatomy_input_bundle
        from shared.pipeline import _build_native_bundle, _build_registered_bundle, input_source_cache_prefix

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = _subject(tmp)
            raw = _anatomy_input_bundle(subject_root, "100307", None, None)
            _complete_preprocessing(subject_root)
            pre = _anatomy_input_bundle(subject_root, "100307", None, None)
            self.assertEqual(input_source_cache_prefix(raw), "")
            self.assertEqual(input_source_cache_prefix(pre), "preproc_")
            self.assertEqual(_build_native_bundle(raw, "ls").bundle_dir, subject_root / "cache" / "native_dti" / "ls")
            self.assertEqual(
                _build_native_bundle(pre, "ls").bundle_dir, subject_root / "cache" / "native_dti" / "preproc_ls"
            )
            atlas = Path(tmp) / "atlas.nii.gz"
            self.assertNotEqual(
                _build_registered_bundle(raw, "wls", atlas).bundle_dir,
                _build_registered_bundle(pre, "wls", atlas).bundle_dir,
            )

    def test_anatomy_dwi_for_check_prefers_a_mismatching_source(self) -> None:
        from shared.orchestrator import _anatomy_dwi_for_check

        sources = {"ddseg_source_dwi": "/a/dwi_final.nii.gz", "ddparcel_source_dwi": "/raw/data.nii.gz", "csd_dwi": "/a/dwi_final.nii.gz"}
        self.assertEqual(_anatomy_dwi_for_check(sources), Path("/raw/data.nii.gz"))
        sources["ddparcel_source_dwi"] = "/a/dwi_final.nii.gz"
        self.assertEqual(_anatomy_dwi_for_check(sources), Path("/a/dwi_final.nii.gz"))
        self.assertIsNone(_anatomy_dwi_for_check({"csd_dwi": "/a/dwi.nii.gz", "ddseg_source_dwi": None}))


class ResumeInvalidationTest(unittest.TestCase):
    def _write_ddseg(self, subject_root: Path, qc: dict[str, object]) -> Path:
        from shared import subject_layout

        layout = subject_layout(subject_root)
        layout.ddseg.mkdir(parents=True, exist_ok=True)
        for name in ("SegmentationMap_GMWMCSF", "ProbabilisticMap_GM", "ProbabilisticMap_WM", "ProbabilisticMap_CSF"):
            (layout.ddseg / f"{name}.nii.gz").write_text(name, encoding="utf-8")
        from shared.orchestrator import _segmentation_outputs
        from shared.pipeline import _artifact_record
        qc['output_artifacts'] = {key: _artifact_record(path, verify_content=True)
                                  for key, path in _segmentation_outputs(subject_root).items() if key != 'qc'}
        (layout.ddseg / "ddseg_qc.json").write_text(json.dumps(qc), encoding="utf-8")
        return layout.ddseg / "SegmentationMap_GMWMCSF.nii.gz"

    def test_schema_v1_ddseg_outputs_are_not_reused(self) -> None:
        from shared.orchestrator import _segmentation_complete
        from anatomy.segmentation.orientation import ORIENTATION_CONVENTION

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = _subject(tmp)
            legacy = _ddseg_qc(1, ORIENTATION_CONVENTION)
            legacy.pop("orientation_convention")
            self._write_ddseg(subject_root, legacy)
            self.assertFalse(_segmentation_complete(subject_root))
            self._write_ddseg(subject_root, _ddseg_qc(2, "some_other_convention"))
            self.assertFalse(_segmentation_complete(subject_root))
            self._write_ddseg(subject_root, _ddseg_qc(2, ORIENTATION_CONVENTION))
            self.assertTrue(_segmentation_complete(subject_root))

    def _act_prep(self, subject_root: Path, segmentation: Path, version: int) -> tuple[Path, Path]:
        act_dir = subject_root / "outputs" / "04_tractography" / "act_prep"
        act_dir.mkdir(parents=True, exist_ok=True)
        five_tt = act_dir / "100307_5tt_from_labels.nii.gz"
        gmwmi = act_dir / "100307_gmwmi_from_labels.nii.gz"
        five_tt.write_text("5tt", encoding="utf-8")
        gmwmi.write_text("gmwmi", encoding="utf-8")
        parcellation = act_dir / "source_parcellation.nii.gz"
        parcellation.write_text("parcellation", encoding="utf-8")
        (act_dir / "act_prep_manifest.json").write_text(
            json.dumps({"version": version, "inputs": {"segmentation": _record(segmentation),
                "parcellation": {**_record(parcellation), "exists": True}}}), encoding="utf-8"
        )
        (act_dir / "act_prep_qc.json").write_text(json.dumps({"version": version, "status": "completed",
            "sgm_from_ddparcel": True, "five_tt_stats": {"channel_volume_voxel_units": {"sgm": 1.0}}}), encoding="utf-8")
        return five_tt, gmwmi

    def _csd_manifest(self, root: Path, five_tt: Path, gmwmi: Path) -> Path:
        manifest = root / "csd_act_manifest.json"
        manifest.write_text(
            json.dumps({"version": 3, "dependencies": {"five_tt": _record(five_tt), "gmwmi": _record(gmwmi)}}),
            encoding="utf-8",
        )
        return manifest

    def test_tractogram_from_outdated_act_prep_is_not_reused(self) -> None:
        from shared.orchestrator import _act_inputs_current, _act_prep_complete
        from tractography.act_prep import ACT_PREP_VERSION

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = _subject(tmp)
            segmentation = self._write_ddseg(subject_root, _ddseg_qc(2, "x"))
            five_tt, gmwmi = self._act_prep(subject_root, segmentation, version=1)
            manifest = self._csd_manifest(Path(tmp), five_tt, gmwmi)
            self.assertFalse(_act_inputs_current(manifest))
            self.assertFalse(_act_prep_complete(subject_root, "100307"))

            five_tt, gmwmi = self._act_prep(subject_root, segmentation, version=2)
            manifest = self._csd_manifest(Path(tmp), five_tt, gmwmi)
            self.assertFalse(_act_inputs_current(manifest))
            self.assertFalse(_act_prep_complete(subject_root, "100307"))

            five_tt, gmwmi = self._act_prep(subject_root, segmentation, version=ACT_PREP_VERSION)
            manifest = self._csd_manifest(Path(tmp), five_tt, gmwmi)
            self.assertTrue(_act_inputs_current(manifest))
            self.assertTrue(_act_prep_complete(subject_root, "100307"))

            # DDSeg re-ran after ACT prep: the recorded segmentation no longer matches.
            segmentation.write_text("new segmentation with different size", encoding="utf-8")
            self.assertFalse(_act_inputs_current(manifest))

    def test_rewritten_5tt_invalidates_the_tractogram(self) -> None:
        from shared.orchestrator import _act_inputs_current
        from tractography.act_prep import ACT_PREP_VERSION

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = _subject(tmp)
            segmentation = self._write_ddseg(subject_root, _ddseg_qc(2, "x"))
            five_tt, gmwmi = self._act_prep(subject_root, segmentation, version=ACT_PREP_VERSION)
            manifest = self._csd_manifest(Path(tmp), five_tt, gmwmi)
            self.assertTrue(_act_inputs_current(manifest))
            five_tt.write_text("a different 5tt", encoding="utf-8")
            self.assertFalse(_act_inputs_current(manifest))


    def test_ddseg_only_cache_requires_explicit_permission_on_each_run(self):
        from shared.orchestrator import _act_inputs_current, _act_prep_complete
        from tractography.act_prep import ACT_PREP_VERSION

        with tempfile.TemporaryDirectory() as tmp:
            subject = _subject(tmp)
            segmentation = self._write_ddseg(subject, _ddseg_qc(2, "x"))
            five_tt, gmwmi = self._act_prep(subject, segmentation, ACT_PREP_VERSION)
            manifest = self._csd_manifest(Path(tmp), five_tt, gmwmi)
            prep_path = five_tt.parent / "act_prep_manifest.json"
            prep = json.loads(prep_path.read_text())
            prep["inputs"]["parcellation"] = None
            prep["allow_ddseg_only"] = True
            prep_path.write_text(json.dumps(prep))
            qc_path = five_tt.parent / "act_prep_qc.json"
            qc = json.loads(qc_path.read_text())
            qc.update(sgm_from_ddparcel=False, anatomical_prior_status="ddseg_only")
            qc_path.write_text(json.dumps(qc))
            self.assertFalse(_act_inputs_current(manifest))
            self.assertFalse(_act_prep_complete(subject, "100307"))
            self.assertTrue(_act_inputs_current(manifest, allow_ddseg_only=True))
            self.assertTrue(_act_prep_complete(subject, "100307", allow_ddseg_only=True))

    def test_changed_probability_map_invalidates_act_cache(self):
        from shared.orchestrator import _act_inputs_current
        from tractography.act_prep import ACT_PREP_VERSION

        with tempfile.TemporaryDirectory() as tmp:
            subject = _subject(tmp)
            segmentation = self._write_ddseg(subject, _ddseg_qc(2, "x"))
            five_tt, gmwmi = self._act_prep(subject, segmentation, ACT_PREP_VERSION)
            manifest = self._csd_manifest(Path(tmp), five_tt, gmwmi)
            probability = segmentation.parent / "ProbabilisticMap_WM.nii.gz"
            prep_path = five_tt.parent / "act_prep_manifest.json"
            prep = json.loads(prep_path.read_text())
            prep["inputs"]["probabilities"] = {"wm": {**_record(probability), "exists": True}}
            prep_path.write_text(json.dumps(prep))
            self.assertTrue(_act_inputs_current(manifest))
            probability.write_text("changed probability data")
            self.assertFalse(_act_inputs_current(manifest))


class WorldQCGateTest(unittest.TestCase):
    def test_missing_ddparcel_blocks_default_act_in_plan_and_execution(self):
        from dataclasses import replace
        from unittest.mock import patch
        from shared import ProductOrchestratorConfig, plan_subject_products, run_subject_products, subject_layout
        from test_product_orchestrator import _write_valid_raw_subject

        with tempfile.TemporaryDirectory() as tmp:
            subject = _subject(tmp)
            _write_valid_raw_subject(subject)
            layout = subject_layout(subject)
            layout.ddseg.mkdir(parents=True)
            (layout.ddseg / "SegmentationMap_GMWMCSF.nii.gz").write_text("segmentation")
            config = ProductOrchestratorConfig("100307", subject, ("tractography",),
                execute=True, resume=False, brain_mask_backend="provided",
                excluded_targets=("segmentation", "parcellation"))
            plan = plan_subject_products(config, write=False)
            task = next(t for t in plan.items if t.target == "tractography")
            self.assertEqual(task.status, "blocked")
            self.assertIn("DDParcel", task.reason)
            fallback = plan_subject_products(replace(config, allow_act_ddseg_only=True), write=False)
            self.assertEqual(next(t for t in fallback.items if t.target == "tractography").status, "will_run")
            with patch("tractography.act_prep.run_act_prep") as prepare, patch("tractography.csd.run_csd_act") as track:
                result = run_subject_products(config)
            prepare.assert_not_called()
            track.assert_not_called()
            task = next(t for t in result.tasks if t.name == "tractography.csd_act")
            self.assertEqual(task.status, "blocked")
            self.assertIn("DDParcel", task.reason)

    def test_mirrored_ddseg_blocks_csd_act(self) -> None:
        from shared import ProductOrchestratorConfig, run_subject_products, subject_layout

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = _subject(tmp)
            from test_product_orchestrator import _write_valid_raw_subject
            _write_valid_raw_subject(subject_root)
            layout = subject_layout(subject_root)
            layout.ddseg.mkdir(parents=True)
            layout.ddparcel.mkdir(parents=True)
            affine = np.diag([2.0, 2.0, 2.0, 1.0])
            ddseg = np.zeros((8, 10, 6), dtype=np.float32)
            ddseg[1:-1, 1:-1, 1:-1] = 2
            ddseg[2:-2, 6:-1, 2:-2] = 1
            wmparc = np.where(ddseg == 1, 3001, np.where(ddseg == 2, 1001, 0)).astype(np.int32)
            nib.save(nib.Nifti1Image(ddseg[:, ::-1, :], affine), str(layout.ddseg / "SegmentationMap_GMWMCSF.nii.gz"))
            nib.save(nib.Nifti1Image(wmparc, affine), str(layout.ddparcel / "100307-DDSurfer-wmparc.nii.gz"))

            result = run_subject_products(
                ProductOrchestratorConfig(
                    subject_id="100307",
                    subject_root=subject_root,
                    targets=("tractography",),
                    execute=True,
                    resume=True,
                    excluded_targets=("segmentation", "parcellation"),
                )
            )
            statuses = {task.name: task.status for task in result.tasks}
            self.assertEqual(statuses.get("segmentation.ddseg_world_qc"), "failed")
            self.assertEqual(statuses.get("tractography.csd_act"), "blocked")
            reason = next(task.reason for task in result.tasks if task.name == "tractography.csd_act")
            self.assertIn("world-space QC gate failed", reason)
            record = json.loads((layout.ddseg / "ddseg_world_qc.json").read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "failed")

    def test_settings_are_validated(self) -> None:
        from shared import ProductOrchestratorConfig, plan_subject_products

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = _subject(tmp)
            for overrides in ({"tractography_seeding": "random"}, {"mrtrix_random_seed": -1}, {"ddseg_min_wm_dice": 2.0}):
                with self.subTest(**overrides), self.assertRaises(ValueError):
                    plan_subject_products(
                        ProductOrchestratorConfig(
                            subject_id="100307", subject_root=subject_root, targets=("tractography",), **overrides
                        ),
                        write=False,
                    )


class CliFlagsTest(unittest.TestCase):
    def test_new_tractography_flags_are_parsed(self) -> None:
        from shared.run_subject_pipeline_cli import build_parser

        args = build_parser().parse_args([])
        self.assertIsNone(args.mrtrix_seed)
        self.assertEqual(args.tckgen_seeding, "dynamic")
        self.assertAlmostEqual(args.ddseg_min_wm_dice, 0.70)
        self.assertFalse(args.allow_anatomy_space_mismatch)
        args = build_parser().parse_args(
            ["--mrtrix-seed", "20260926", "--tckgen-seeding", "gmwmi", "--ddseg-min-wm-dice", "0.8", "--allow-anatomy-space-mismatch"]
        )
        self.assertEqual(args.mrtrix_seed, 20260926)
        self.assertEqual(args.tckgen_seeding, "gmwmi")
        self.assertAlmostEqual(args.ddseg_min_wm_dice, 0.8)
        self.assertTrue(args.allow_anatomy_space_mismatch)


if __name__ == "__main__":
    unittest.main()
