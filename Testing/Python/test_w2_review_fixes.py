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


if __name__ == "__main__":
    unittest.main()
