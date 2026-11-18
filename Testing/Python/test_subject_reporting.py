#!/usr/bin/env python3
"""Tests for subject report/provenance generation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class SubjectReportingTest(unittest.TestCase):
    def test_write_subject_reports_from_manifest(self) -> None:
        from shared import subject_layout, write_subject_reports

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            layout = subject_layout(subject_root)
            layout.run_metadata.mkdir(parents=True)
            (layout.microstructure_dti).mkdir(parents=True)
            (layout.microstructure_dti / "100307_dti_FA.nii.gz").write_text("fa", encoding="utf-8")
            qc_path = layout.microstructure_dti / "dti_qc.json"
            qc_path.write_text('{"kind": "dti_qc", "status": "completed"}\n', encoding="utf-8")
            manifest_path = layout.run_metadata / "csd_act_orchestration_manifest.json"
            manifest = {
                "kind": "slicerddsurfer_product_orchestration",
                "subject_id": "100307",
                "subject_root": subject_root.as_posix(),
                "output_root": layout.outputs.as_posix(),
                "run_name": "csd_act",
                "targets": ["microstructure"],
                "run_targets": ["microstructure"],
                "run_plan_summary": {"skipped": 1},
                "tasks": [
                    {
                        "name": "microstructure.dti",
                        "status": "skipped",
                        "reason": "completed outputs found",
                        "qc_path": qc_path.as_posix(),
                    }
                ],
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            paths = write_subject_reports(subject_root=subject_root, manifest_path=manifest_path, run_name="csd_act")

            for path in (
                paths.report_html,
                paths.provenance_json,
                paths.qc_summary_json,
                paths.summary_md,
                paths.report_html_latest,
                paths.provenance_json_latest,
                paths.qc_summary_json_latest,
                paths.summary_md_latest,
            ):
                self.assertTrue(path.is_file(), path)
            provenance = json.loads(paths.provenance_json.read_text(encoding="utf-8"))
            qc_summary = json.loads(paths.qc_summary_json.read_text(encoding="utf-8"))
            self.assertEqual(provenance["subject_id"], "100307")
            self.assertEqual(provenance["task_summary"]["skipped"], 1)
            self.assertEqual(qc_summary["summary"]["status"], "completed")
            self.assertEqual(qc_summary["qc_files"][0]["status"], "completed")
            report_html = paths.report_html.read_text(encoding="utf-8")
            self.assertIn("SlicerDSpace Report", report_html)
            self.assertIn("Task Status", report_html)
            self.assertIn("Key Outputs", report_html)
            self.assertIn("SlicerDSpace Run Summary", paths.summary_md.read_text(encoding="utf-8"))

    def test_planned_manifest_reports_planned_overall_status(self) -> None:
        from shared import subject_layout, write_subject_reports

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            layout = subject_layout(subject_root)
            layout.run_metadata.mkdir(parents=True)
            manifest_path = layout.run_metadata / "preview_orchestration_manifest.json"
            manifest = {
                "kind": "slicerddsurfer_product_orchestration",
                "subject_id": "100307",
                "subject_root": subject_root.as_posix(),
                "output_root": layout.outputs.as_posix(),
                "run_name": "preview",
                "targets": ["segmentation", "surface"],
                "run_targets": ["segmentation", "surface"],
                "run_plan_summary": {"will_run": 2, "skipped": 0, "blocked": 0},
                "tasks": [
                    {"name": "segmentation.ddseg", "status": "planned", "reason": "execute=False"},
                    {"name": "surface.ddsurfer", "status": "planned", "reason": "execute=False"},
                ],
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            paths = write_subject_reports(subject_root=subject_root, manifest_path=manifest_path, run_name="preview")

            qc_summary = json.loads(paths.qc_summary_json.read_text(encoding="utf-8"))
            self.assertEqual(qc_summary["summary"]["status"], "planned")
            self.assertIn("planned", qc_summary["summary"]["interpretation"])
            self.assertIn("badge-planned", paths.report_html.read_text(encoding="utf-8"))

    def test_write_output_tree_report_counts_files_symlinks_and_postprocess_qc(self) -> None:
        from shared import subject_layout, write_output_tree_report

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            layout = subject_layout(subject_root)
            layout.microstructure_dti.mkdir(parents=True)
            layout.preproc.mkdir(parents=True)
            (layout.preproc / "04_bias_normalize").mkdir(parents=True)
            (layout.microstructure_dti / "100307_dti_FA.nii.gz").write_text("fa", encoding="utf-8")
            layout.ddsurfer_postprocess.mkdir(parents=True)
            (layout.ddsurfer_postprocess / "manifest.json").write_text(
                json.dumps({"stage_status": {"validate_inputs": "completed", "sync_outputs": "completed"}}),
                encoding="utf-8",
            )
            (layout.ddsurfer_postprocess / "qc.json").write_text(
                json.dumps({"status": "completed", "core_outputs": {"annot_count": 2, "fsaverage_overlay_count": 1, "stats_count": 3}}),
                encoding="utf-8",
            )
            (layout.outputs / "linked_fa.nii.gz").symlink_to(layout.microstructure_dti / "100307_dti_FA.nii.gz")

            paths = write_output_tree_report(
                output_root=layout.outputs,
                destination=layout.slicer_scene_report / "unit_output_tree.html",
                latest_destination=layout.slicer_scene_report / "output_tree.html",
                subject_id="100307",
                run_name="unit",
            )

            self.assertTrue(paths.report_html.is_file())
            self.assertTrue(paths.report_html_latest and paths.report_html_latest.is_file())
            self.assertEqual(paths.regular_file_count, 3)
            self.assertEqual(paths.symlink_count, 1)
            html = paths.report_html.read_text(encoding="utf-8")
            self.assertIn("DTI scalar map", html)
            self.assertIn("01_anatomical_priors/", html)
            self.assertIn("surface_morphometry/", html)
            self.assertIn("02_microstructure/", html)
            self.assertIn("diffusion_tensor/", html)
            self.assertIn("symlink", html)
            self.assertIn("annot=2", html)


if __name__ == "__main__":
    unittest.main()
