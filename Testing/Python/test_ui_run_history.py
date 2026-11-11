#!/usr/bin/env python3
"""Widget-order and wording contracts for the Results panel, run against a fake ``qt`` module."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from shared import ui_algorithm_picker, ui_run_history
from shared.run_history import RunDiscovery, SUBJECT_MANIFEST_KIND, parse_run_manifest, run_record_details
from shared.result_scene import review_state_text

WIDGET_KINDS = (
    "QLabel", "QPushButton", "QComboBox", "QToolButton", "QTableWidget", "QWidget",
    "QHBoxLayout", "QVBoxLayout", "QFormLayout", "QGridLayout", "QDialog", "QDialogButtonBox", "QCheckBox",
)


def fake_qt() -> mock.MagicMock:
    """A qt stand-in whose widgets and layouts are distinct mocks that remember how they were built."""

    qt = mock.MagicMock(name="qt")
    created: list[mock.MagicMock] = []

    def factory(kind: str):
        def make(*args, **kwargs):
            widget = mock.MagicMock(name=kind)
            widget.kind = kind
            widget.ctor_args = args
            if kind == "QTableWidget":
                widget.fontMetrics.return_value.height.return_value = 14
                widget.horizontalHeader.return_value.sizeHint.return_value.height.return_value = 24
                widget.verticalHeader.return_value.defaultSectionSize.return_value = 22
                widget.frameWidth.return_value = 1
                widget.rowCount.return_value = 0
            created.append(widget)
            return widget

        return make

    for kind in WIDGET_KINDS:
        getattr(qt, kind).side_effect = factory(kind)
    qt.created = created
    return qt


def object_name(widget) -> str:
    call = widget.setObjectName.call_args
    return call[0][0] if call else ""


def by_name(qt, name: str):
    return next(widget for widget in qt.created if object_name(widget) == name)


def widget_names(layout) -> list[str]:
    return [
        object_name(call[1][0])
        for call in layout.method_calls
        if call[0] == "addWidget"
    ]


class ResultsPanelTest(unittest.TestCase):
    def build(self):
        qt = fake_qt()
        layout = mock.MagicMock(name="resultsLayout")
        section = mock.MagicMock(name="resultsCollapsibleButton")
        self.refreshes = []
        owner = SimpleNamespace(
            ui=SimpleNamespace(resultsLayout=layout, resultsCollapsibleButton=section),
            _on_run_status_table_double_clicked=lambda *a: None,
            _manage_result_scene=lambda *a: None,
            _on_open_latest_report_clicked=lambda *a: None,
            _on_open_output_tree_report_clicked=lambda *a: None,
            _on_run_history_selection_changed=lambda *a: None,
            _refresh_run_history=lambda *a: self.refreshes.append(a),
            _on_browse_run_directory_clicked=lambda *a: None,
        )
        with mock.patch.dict(sys.modules, {"qt": qt}):
            ui_run_history.install_results_panel(owner)
        return qt, layout, owner

    def test_run_row_holds_only_the_label_and_the_stretching_combo(self) -> None:
        qt, layout, owner = self.build()
        row = next(
            widget for widget in qt.created
            if widget.kind == "QHBoxLayout" and "runHistoryComboBox" in widget_names(widget)
        )
        self.assertEqual(widget_names(row), ["runHistoryLabel", "runHistoryComboBox"])
        combo_call = next(call for call in row.method_calls if call[0] == "addWidget"
                          and object_name(call[1][0]) == "runHistoryComboBox")
        self.assertEqual(combo_call[1][1], 1)  # the combo stretches
        by_name(qt, "runHistoryLabel").setBuddy.assert_called_once_with(by_name(qt, "runHistoryComboBox"))

    def test_there_are_no_refresh_or_browse_buttons(self) -> None:
        qt, layout, owner = self.build()
        names = {object_name(widget) for widget in qt.created}
        self.assertNotIn("runHistoryRefreshButton", names)
        self.assertNotIn("runHistoryBrowseButton", names)
        self.assertEqual([widget for widget in qt.created if widget.kind == "QPushButton"
                          and object_name(widget).startswith("runHistory")], [])
        self.assertFalse(hasattr(owner, "_runHistoryRefreshButton"))
        self.assertFalse(hasattr(owner, "_runHistoryBrowseButton"))

    def test_expanding_the_results_section_rescans_once(self) -> None:
        qt, layout, owner = self.build()
        calls = [call for call in owner.ui.resultsCollapsibleButton.connect.call_args_list
                 if call[0][0] == "contentsCollapsed(bool)"]
        self.assertEqual(len(calls), 1)
        slot = calls[0][0][1]
        slot(True)  # collapsing does not rescan
        self.assertEqual(self.refreshes, [])
        slot(False)
        self.assertEqual(self.refreshes, [()])

    def test_summary_card_comes_first_then_details_review_and_table(self) -> None:
        qt, layout, owner = self.build()
        self.assertEqual(
            widget_names(layout)[:4],
            ["runSummaryLabel", "runDetailsLabel", "reviewStateLabel", "runStatusTable"],
        )
        first = layout.method_calls[0]
        self.assertEqual(first[0], "addLayout")  # the run row is above the card

    def test_placeholders_use_the_new_wording(self) -> None:
        qt, layout, owner = self.build()
        self.assertEqual(
            by_name(qt, "reviewStateLabel").ctor_args,
            ("Recorded QC: consult report\nVisual review: pending",),
        )

    def test_stage_table_is_one_line_per_row_with_only_the_prose_columns_visible(self) -> None:
        qt, layout, owner = self.build()
        table = by_name(qt, "runStatusTable")
        table.setWordWrap.assert_called_once_with(False)
        table.setTextElideMode.assert_called_once_with(qt.Qt.ElideRight)
        table.setHorizontalScrollBarPolicy.assert_called_once_with(qt.Qt.ScrollBarAlwaysOff)
        self.assertCountEqual([call[0][0] for call in table.setColumnHidden.call_args_list if call[0][1]], [3, 4, 5])
        header = table.verticalHeader.return_value
        header.setVisible.assert_called_with(False)
        header.setDefaultSectionSize.assert_called_once()
        self.assertGreaterEqual(header.setDefaultSectionSize.call_args[0][0], 14)
        columns = table.horizontalHeader.return_value
        columns.setSectionResizeMode.assert_any_call(2, qt.QHeaderView.Stretch)  # Reason takes the width left over
        table.setToolTip.assert_called_with(
            "Per-stage outcome of the selected run. Double-click a row to open its QC file or logs."
        )

    def test_stage_table_height_fits_the_rows_instead_of_a_fixed_minimum(self) -> None:
        qt, layout, owner = self.build()
        table = by_name(qt, "runStatusTable")
        table.setMinimumHeight.assert_not_called()
        # header 24 + one row (empty table counts as 1) 22 + frame 2 * 1
        table.setFixedHeight.assert_called_with(24 + 22 + 2)

    def test_scene_actions_are_one_row_of_four_and_reports_one_row_of_two(self) -> None:
        qt, layout, owner = self.build()

        def labels(row) -> list[str]:
            return [call[1][0].ctor_args[0] if call[1][0].ctor_args else "" for call in row.method_calls if call[0] == "addWidget"]

        rows = [widget for widget in qt.created if widget.kind == "QHBoxLayout"]
        scene_row = next(row for row in rows if labels(row) == ["Load", "Replace", "Append", "Clear"])
        report_row = next(row for row in rows if labels(row) == ["Open report", "Open output tree"])
        tips = [call[1][0].setToolTip.call_args[0][0] for call in scene_row.method_calls if call[0] == "addWidget"]
        self.assertIn("reuse", tips[0].lower())
        self.assertIn("remove", tips[1].lower())
        self.assertIn("another copy", tips[2].lower())
        self.assertIn("source files are never deleted", tips[3].lower())
        self.assertEqual(len(owner._resultSceneButtons), 4)
        self.assertEqual([call[0] for call in layout.method_calls if call[0] == "addLayout"], ["addLayout"] * 3)

    def test_scene_buttons_show_their_meaning_when_enabled_and_the_reason_when_disabled(self) -> None:
        buttons = [mock.MagicMock() for _ in range(4)]
        owner = SimpleNamespace(_resultSceneButtons=buttons)
        ui_run_history._set_scene_actions(owner, has_subject_run=True, is_batch=False)
        self.assertTrue(all(button.enabled for button in buttons))
        tips = [button.setToolTip.call_args[0][0] for button in buttons]
        self.assertEqual(len(set(tips)), 4)  # one meaning per button, not the shared sentence
        ui_run_history._set_scene_actions(owner, has_subject_run=False, is_batch=False)
        self.assertFalse(any(button.enabled for button in buttons))
        reasons = {button.setToolTip.call_args[0][0] for button in buttons}
        self.assertEqual(len(reasons), 1)
        self.assertIn("Add a run folder", reasons.pop())


class SelectionTest(unittest.TestCase):
    def test_selection_shows_brief_with_full_details_as_tooltip_and_shared_review_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r_orchestration_manifest.json"
            payload = {
                "kind": SUBJECT_MANIFEST_KIND, "version": 1, "status": "completed", "subject_id": "sub-01",
                "run_name": "ms_micro", "targets": ["microstructure", "microstructure_noddi"], "tasks": [],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            details, review = mock.MagicMock(), mock.MagicMock()
            # The list is: the run, a separator, and the "Add a run folder…" item.
            combo = SimpleNamespace(count=3, currentIndex=0, itemData=lambda index: path.as_posix())
            owner = SimpleNamespace(
                _runDetailsLabel=details, _reviewStateLabel=review, _runHistoryComboBox=combo,
                _runHistoryNotes=("one note",), _tableShowsLiveRun=False,
                _set_run_status_table_rows=lambda rows: None,
                _set_latest_report_path=lambda value: None,
                _set_latest_output_tree_path=lambda value: None,
                _set_run_summary_message=lambda message, state="": None,
                _run_summary_from_manifest=lambda manifest: ("", ""),
            )
            with mock.patch.object(ui_run_history, "_set_scene_actions"):
                ui_run_history.on_run_selection_changed(owner)

            record = parse_run_manifest(path)
            details.setText.assert_called_once_with("Products: DTI, NODDI\none note")
            details.setToolTip.assert_called_once_with(run_record_details(record))
            review.setText.assert_called_once_with(
                review_state_text({"passed": 0, "issues": 0, "unavailable": 0}, issues_found=False)
            )


class FakeCombo:
    """PythonQt-style combo box: ``count`` and ``currentIndex`` are properties, not calls."""

    def __init__(self) -> None:
        self.items: list[dict] = []
        self.currentIndex = 0
        self.enabled = True

    @property
    def count(self) -> int:
        return len(self.items)

    def clear(self) -> None:
        self.items = []

    def addItem(self, text, data=None) -> None:
        self.items.append({"text": text, "data": data, "separator": False})

    def insertSeparator(self, index) -> None:
        self.items.insert(index, {"text": "", "data": None, "separator": True})

    def setItemData(self, index, value, role=0) -> None:
        self.items[index].setdefault("roles", {})[role] = value

    def itemData(self, index):
        return self.items[index]["data"] if 0 <= index < len(self.items) else None

    def setCurrentIndex(self, index) -> None:
        self.currentIndex = index

    def blockSignals(self, flag) -> bool:
        return False


def make_record(root: Path, name: str):
    path = root / f"{name}_orchestration_manifest.json"
    path.write_text(json.dumps({
        "kind": SUBJECT_MANIFEST_KIND, "version": 1, "status": "completed", "subject_id": "sub-01",
        "run_name": name, "targets": ["segmentation"], "tasks": [],
    }), encoding="utf-8")
    return parse_run_manifest(path)


class AddRunFolderItemTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.combo = FakeCombo()
        self.browsed: list[int] = []
        self.owner = SimpleNamespace(
            _runHistoryComboBox=self.combo, _runHistoryNotes=(), _tableShowsLiveRun=False,
            _runDetailsLabel=mock.MagicMock(), _reviewStateLabel=mock.MagicMock(),
            _set_run_status_table_rows=lambda rows: None,
            _set_latest_report_path=lambda value: None,
            _set_latest_output_tree_path=lambda value: None,
            _set_run_summary_message=lambda message, state="": None,
            _run_summary_from_manifest=lambda manifest: ("", ""),
            _on_browse_run_directory_clicked=lambda: self.browsed.append(self.combo.currentIndex),
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def refresh(self, *names: str) -> None:
        records = tuple(make_record(self.root, name) for name in names)
        discovery = RunDiscovery(records=records)
        with mock.patch.object(ui_run_history, "discover_runs", return_value=discovery), \
                mock.patch.object(ui_run_history, "history_roots", return_value=[]), \
                mock.patch.object(ui_run_history, "_set_scene_actions"):
            ui_run_history.refresh_run_history(self.owner)

    def test_last_item_is_add_a_run_folder_after_a_separator_and_carries_no_manifest(self) -> None:
        self.refresh("a", "b")
        items = self.combo.items
        self.assertEqual(len(items), 4)
        self.assertTrue(items[2]["separator"])
        self.assertEqual(items[3]["text"], "Add a run folder…")
        self.assertFalse(items[3]["data"])
        self.assertTrue(all(item["data"] for item in items[:2]))

    def test_empty_list_keeps_the_add_item_and_stays_enabled(self) -> None:
        self.refresh()
        texts = [item["text"] for item in self.combo.items if not item["separator"]]
        self.assertEqual(texts, ["No runs found", "Add a run folder…"])
        self.assertTrue(self.combo.enabled)

    def test_selected_manifest_path_ignores_the_add_item(self) -> None:
        self.refresh("a")
        self.assertIsNotNone(ui_run_history.selected_manifest_path(self.owner))
        self.combo.currentIndex = self.combo.count - 1
        self.assertIsNone(ui_run_history.selected_manifest_path(self.owner))

    def test_choosing_the_add_item_browses_and_restores_the_previous_run(self) -> None:
        self.refresh("a", "b")
        self.combo.currentIndex = 1
        with mock.patch.object(ui_run_history, "_set_scene_actions") as scene_actions:
            ui_run_history.on_run_combo_changed(self.owner)  # a real run: remembered
            self.combo.currentIndex = self.combo.count - 1  # the user picks "Add a run folder…"
            self.owner._runDetailsLabel.reset_mock()
            scene_actions.reset_mock()
            ui_run_history.on_run_combo_changed(self.owner)
        self.assertEqual(self.browsed, [1])  # the selection was already restored when the dialog opened
        self.assertEqual(self.combo.currentIndex, 1)
        scene_actions.assert_not_called()  # no scene action was enabled or disabled by the special item
        self.owner._runDetailsLabel.setText.assert_not_called()

    def test_browse_that_adds_a_run_keeps_the_new_selection(self) -> None:
        self.refresh("a")
        self.owner._on_browse_run_directory_clicked = lambda: self.refresh("a", "new") or self.combo.setCurrentIndex(1)
        self.combo.currentIndex = self.combo.count - 1
        with mock.patch.object(ui_run_history, "_set_scene_actions"):
            ui_run_history.on_run_combo_changed(self.owner)
        self.assertEqual(self.combo.currentIndex, 1)

    def test_rescan_keeps_the_current_selection(self) -> None:
        self.refresh("a", "b")
        self.combo.currentIndex = 1
        self.refresh("a", "b")
        self.assertEqual(self.combo.currentIndex, 1)


class ModuleWidgetWordingTest(unittest.TestCase):
    def test_load_error_dialog_points_to_the_run_list_not_to_removed_buttons(self) -> None:
        source = (Path(__file__).resolve().parents[2] / "shared" / "module_widget.py").read_text(encoding="utf-8")
        self.assertNotIn("Click Refresh", source)
        self.assertIn("Open Results and QC again to rescan run folders", source)
        self.assertIn("(\"Rescan\", self._refresh_run_history)", source)
        self.assertIn("(\"Add a run folder…\", self._on_browse_run_directory_clicked)", source)


class AlgorithmDialogTest(unittest.TestCase):
    def open_dialog(self, language: str):
        qt = fake_qt()
        slicer = mock.MagicMock()
        owner = SimpleNamespace(_onboardingLanguage=language, _parameterNode=SimpleNamespace())
        with mock.patch.dict(sys.modules, {"qt": qt, "slicer": slicer}):
            ui_algorithm_picker.choose_algorithms(owner, "tractography")
        dialog_layout = next(widget for widget in qt.created if widget.kind == "QVBoxLayout")
        return qt, dialog_layout

    def test_tractography_note_is_short_and_the_buttons_have_room(self) -> None:
        for language, text in (
            ("en", "Each method writes its own tractogram. Fiber bundles and connectomes are computed for each method separately."),
            ("zh", "每种方法分别输出流线；束分析和连接组按方法分别计算。"),
        ):
            with self.subTest(language=language):
                qt, layout = self.open_dialog(language)
                notes = [widget for widget in qt.created if widget.kind == "QLabel" and widget.ctor_args == (text,)]
                self.assertEqual(len(notes), 1)
                calls = [(call[0], call[1]) for call in layout.method_calls if call[0] in {"addWidget", "addSpacing"}]
                note_index = next(i for i, call in enumerate(calls) if call[1] and call[1][0] is notes[0])
                self.assertEqual(calls[note_index + 1], ("addSpacing", (8,)))
                self.assertEqual(calls[note_index + 2][0], "addWidget")  # the Apply/Cancel box


class AlgorithmRowTest(unittest.TestCase):
    def test_summary_text_aligns_with_a_plain_checkbox_and_rows_are_vertically_centred(self) -> None:
        qt = fake_qt()
        indicator, gap = 19, 6  # deliberately unlike the old 20, while >=18px unclipped floor
        style = qt.QApplication.style.return_value
        style.pixelMetric.side_effect = lambda metric: {
            qt.QStyle.PM_IndicatorWidth: indicator, qt.QStyle.PM_CheckBoxLabelSpacing: gap,
        }[metric]
        boxes = {name: mock.MagicMock(name=name) for _, (name, _, _) in ui_algorithm_picker.FAMILIES.items()}
        form = mock.MagicMock()
        owner = SimpleNamespace(
            ui=SimpleNamespace(targetProductsFormLayout=form, **boxes), _parameterNode=SimpleNamespace(),
        )
        with mock.patch.dict(sys.modules, {"qt": qt}):
            ui_algorithm_picker.install_algorithm_pickers(owner)

        rows = [widget for widget in qt.created if widget.kind == "QHBoxLayout"]
        self.assertEqual(len(rows), len(ui_algorithm_picker.FAMILIES))
        self.assertEqual(len(rows), 5)
        for box, row in zip(boxes.values(), rows):
            box.setFixedWidth.assert_called_once_with(indicator)
            calls = [(call[0], call[1]) for call in row.method_calls if call[0] in {"addWidget", "addSpacing"}]
            self.assertEqual([name for name, _ in calls], ["addWidget", "addSpacing", "addWidget", "addSpacing", "addWidget"])
            self.assertEqual(calls[0][1], (box, 0, qt.Qt.AlignVCenter))
            self.assertEqual(calls[1][1], (gap,))
            self.assertEqual(calls[4][1][1:], (0, qt.Qt.AlignVCenter))
            row.setSpacing.assert_called_once_with(0)


if __name__ == "__main__":
    unittest.main()
