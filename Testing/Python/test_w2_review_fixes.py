#!/usr/bin/env python3
"""W2 review fixes F1..F5 (T1 D1 part 2): one test class per fix."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_w2b_reuse_predicate import RAW_NAMES, make_subject, reuse_points, write_preprocessing  # noqa: E402


def rewrite_input_records(subject: Path, make_record) -> None:
    """Replace the raw-input records of the manifest with ``make_record(path)``."""
    manifest_path = subject / "outputs" / "01_preproc" / "100307_preproc_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cache_key"]["inputs"] = {key: make_record(subject / "input" / name) for key, name in RAW_NAMES.items()}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


class F1InputRecordPathTest(unittest.TestCase):
    def test_f1_a_manifest_written_with_the_producer_record_function_is_usable(self) -> None:
        from preprocessing.pipeline import _file_record

        with tempfile.TemporaryDirectory() as tmp:
            subject = make_subject(Path(tmp))
            write_preprocessing(subject, cache_key="v6")
            rewrite_input_records(subject, _file_record)
            self.assertEqual(set(reuse_points(subject).values()), {True})

    def test_f1_b_a_stored_path_string_that_differs_but_names_the_same_file_is_usable(self) -> None:
        """The Windows case in miniature: the stored string is not `str(resolve())`, the file is the same."""
        from preprocessing.pipeline import _file_record

        def other_spelling(path: Path) -> dict:
            record = _file_record(path)
            record["path"] = (path.parent / ".." / path.parent.name / path.name).as_posix()
            self.assertNotEqual(record["path"], str(path.resolve()))
            return record

        with tempfile.TemporaryDirectory() as tmp:
            subject = make_subject(Path(tmp))
            write_preprocessing(subject, cache_key="v6")
            rewrite_input_records(subject, other_spelling)
            self.assertEqual(set(reuse_points(subject).values()), {True})

    def test_f1_c_a_record_of_another_file_or_a_changed_size_or_mtime_is_not_usable(self) -> None:
        from preprocessing.pipeline import _file_record

        def other_file(path: Path) -> dict:
            record = _file_record(path)
            record["path"] = (path.parent / "elsewhere.nii.gz").as_posix()
            return record

        with tempfile.TemporaryDirectory() as tmp:
            subject = make_subject(Path(tmp))
            write_preprocessing(subject, cache_key="v6")
            rewrite_input_records(subject, _file_record)
            self.assertEqual(set(reuse_points(subject).values()), {True})
            rewrite_input_records(subject, other_file)
            self.assertEqual(set(reuse_points(subject).values()), {False})
            rewrite_input_records(subject, _file_record)
            path = subject / "input" / RAW_NAMES["bval"]
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
            self.assertEqual(set(reuse_points(subject).values()), {False})


class F2UnsupportedPlatformTest(unittest.TestCase):
    """F2: a platform with no FSL build does not offer preprocessing, and it is not left selected."""

    def fake_checkbox(self, checked=True):
        state = {"checked": checked, "enabled": True, "calls": []}

        def set_checked(value):
            state["checked"] = value
            state["calls"].append(("checked", value))

        box = types.SimpleNamespace(
            setEnabled=lambda value: state.update(enabled=value), setToolTip=lambda text: None,
            blockSignals=lambda value: state["calls"].append(("signals", value)), setChecked=set_checked,
            isChecked=lambda: state["checked"])
        return box, state

    def test_f2_apply_unchecks_the_box_and_clears_the_parameter_node_where_there_is_no_fsl_build(self) -> None:
        from unittest.mock import patch

        from shared.ui_parameter_sync import apply_preprocessing_availability

        box, state = self.fake_checkbox(True)
        node = types.SimpleNamespace(targetPreprocessing=True)
        with patch("platform.mac_ver", return_value=("15.1", ("", "", ""), "")):
            self.assertFalse(apply_preprocessing_availability(box, "Linux", "aarch64", node=node))
        self.assertFalse(state["enabled"])
        self.assertFalse(state["checked"])
        self.assertFalse(node.targetPreprocessing)

    def test_f2_apply_keeps_the_selection_where_preprocessing_is_offered(self) -> None:
        from shared.ui_parameter_sync import apply_preprocessing_availability

        box, state = self.fake_checkbox(True)
        node = types.SimpleNamespace(targetPreprocessing=True)
        self.assertTrue(apply_preprocessing_availability(box, "Linux", "x86_64", node=node))
        self.assertTrue(state["checked"])
        self.assertTrue(node.targetPreprocessing)

    def test_f2_sync_parameter_node_to_ui_clears_a_saved_selection_where_there_is_no_fsl_build(self) -> None:
        from unittest.mock import MagicMock, patch

        from shared import ui_parameter_sync

        box, state = self.fake_checkbox(True)
        node = MagicMock()
        node.targetPreprocessing = True
        node.orchestrationTractographyBackend = "csd_act"
        node.orchestrationBundleBackend = "wmql"
        widget = MagicMock()
        widget._parameterNode = node
        del widget._ukfControls
        widget.ui = types.SimpleNamespace(targetPreprocessingCheckBox=box)
        widget._microstructure_stages_text.return_value = "dti"
        widget._sync_checkbox.side_effect = lambda w, desired: w.setChecked(desired) if w is not None else None
        with patch("shared.selected_setup.preprocessing_offered", return_value=False):
            ui_parameter_sync.sync_parameter_node_to_ui(widget)
        self.assertFalse(node.targetPreprocessing)
        self.assertFalse(state["checked"])


class F3PeBlockDecidedOnceTest(unittest.TestCase):
    """F3: rule 4 is decided once at the entry point and blocks every DWI target of every tracking variant."""

    TARGETS = ("preprocessing", "microstructure", "tractography")

    def config(self, subject: Path, **kwargs):
        from shared import ProductOrchestratorConfig

        return ProductOrchestratorConfig(
            subject_id="100307", subject_root=subject, targets=self.TARGETS,
            tractography_backend="csd_act,dti_det", **kwargs)

    def test_f3_a_multi_variant_run_without_a_pe_direction_runs_no_stage_of_any_variant(self) -> None:
        from shared import run_subject_products
        from shared.pe_resolution import RULE4_MESSAGE
        from test_w2b_pe_resolution import write_subject
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            subject = write_subject(Path(tmp))
            lines: list[str] = []
            with patch("preprocessing.run_dwi_preprocessing") as preprocessing:
                result = run_subject_products(self.config(subject, execute=True), log=lines.append)
            preprocessing.assert_not_called()
            names = sorted(task.name for task in result.tasks)
            self.assertEqual({task.status for task in result.tasks}, {"blocked"}, [(t.name, t.status) for t in result.tasks])
            # Both variants have a blocked tractography task: the second one did not run on the raw DWI.
            self.assertEqual(len([name for name in names if name.startswith("tractography")]), 2, names)
            self.assertFalse([name for name in names if name.startswith("execution.failure")], names)
            self.assertTrue(all(RULE4_MESSAGE in task.reason for task in result.tasks if task.name != "preprocessing.dwi"))
            self.assertFalse((subject / "outputs" / "02_microstructure").exists())
            self.assertFalse((subject / "outputs" / "01_preproc" / "100307_preproc_manifest.json").exists())

    def test_f3_the_plan_marks_the_same_dwi_targets_blocked_as_execute(self) -> None:
        from shared import plan_subject_products, run_subject_products
        from shared.pe_resolution import RULE4_MESSAGE
        from test_w2b_pe_resolution import write_subject

        with tempfile.TemporaryDirectory() as tmp:
            subject = write_subject(Path(tmp))
            plan = plan_subject_products(self.config(subject), write=False)
            planned = {item.task_name for item in plan.items if item.status == "blocked"
                       and RULE4_MESSAGE in item.reason}
            result = run_subject_products(self.config(subject, execute=True))
            executed = {task.name for task in result.tasks if task.status == "blocked" and RULE4_MESSAGE in task.reason}
            self.assertIn("microstructure.dti", planned)
            self.assertIn("preprocessing.dwi", planned)
            self.assertEqual(planned, executed)
            self.assertFalse([item for item in plan.items if item.status == "will_run"
                              and item.target in {"microstructure", "tractography", "segmentation"}])

    def test_f3_a_single_variant_plan_blocks_the_dwi_targets_with_preprocessing(self) -> None:
        from shared import ProductOrchestratorConfig, plan_subject_products
        from test_w2b_pe_resolution import write_subject

        with tempfile.TemporaryDirectory() as tmp:
            subject = write_subject(Path(tmp))
            plan = plan_subject_products(ProductOrchestratorConfig(
                subject_id="100307", subject_root=subject, targets=("preprocessing", "microstructure")), write=False)
            status = {item.target: item.status for item in plan.items}
            self.assertEqual((status["preprocessing"], status["microstructure"]), ("blocked", "blocked"))


class F4NoQcRequirementTest(unittest.TestCase):
    """F4: the config-less predicate is true for a completed manifest with its files and raw signatures; the QC JSON
    is not part of it (D1)."""

    def test_f4_a_qc_json_that_says_running_or_is_absent_or_failed_still_gives_usable_outputs_at_both_points(self) -> None:
        for version in ("v5", "v6"):
            for qc in ("running", "absent", "failed"):
                with self.subTest(version=version, qc=qc), tempfile.TemporaryDirectory() as tmp:
                    subject = make_subject(Path(tmp))
                    preproc = write_preprocessing(subject, cache_key=version)
                    qc_path = preproc / "05_qc" / "100307_preproc_qc.json"
                    if qc == "absent":
                        qc_path.unlink()
                    else:
                        qc_path.write_text(json.dumps({"status": qc}), encoding="utf-8")
                    self.assertEqual(set(reuse_points(subject).values()), {True})

    def test_f4_a_missing_cache_key_or_missing_output_or_changed_raw_input_is_still_unusable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subject = make_subject(Path(tmp))
            write_preprocessing(subject, cache_key=None, qc="running")
            self.assertEqual(set(reuse_points(subject).values()), {False})
            write_preprocessing(subject, cache_key="v5", qc="running", missing="dwi_final.bval")
            self.assertEqual(set(reuse_points(subject).values()), {False})


if __name__ == "__main__":
    unittest.main()
