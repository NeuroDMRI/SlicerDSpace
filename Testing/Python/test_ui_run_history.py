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
    "QLabel", "QPushButton", "QComboBox", "QToolButton", "QTableWidget", "QSplitter", "QTreeWidget", "QWidget",
    "QHBoxLayout", "QVBoxLayout", "QFormLayout", "QGridLayout", "QDialog", "QDialogButtonBox", "QCheckBox",
    "QSpinBox", "QDoubleSpinBox", "QScrollArea", "QLineEdit",
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
            if kind in {"QSpinBox", "QDoubleSpinBox"}:
                widget.value = 0
                widget.setValue.side_effect = lambda v, w=widget: setattr(w, "value", v)
            elif kind == "QCheckBox":
                widget.checked = False
                widget.setChecked.side_effect = lambda v, w=widget: setattr(w, "checked", bool(v))
            elif kind == "QTableWidget":
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

    def test_two_column_splitter_aligns_runs_and_outputs_and_hides_browse_and_review(self) -> None:
        qt, layout, owner = self.build()
        names = {object_name(widget) for widget in qt.created}
        self.assertIn("subjectRunHistory", names)
        self.assertIn("outputsHeaderLabel", names)
        self.assertIn("addRunFolderButton", names)
        self.assertIn("resultsReportBar", names)
        by_name(qt, "addRunFolderButton").hide.assert_called_once_with()
        by_name(qt, "reviewStateLabel").hide.assert_called_once_with()
        by_name(qt, "resultsReportBar").hide.assert_called_once_with()
        self.assertEqual(widget_names(layout), ["", "", "resultsReportBar"])
        top_names = [call[1][0].kind if call[1][0].kind != "QSplitter" else "QSplitter"
                     for call in layout.method_calls if call[0] == "addWidget"]
        self.assertEqual(len(top_names), 3)

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
        self.assertIsNone(owner._runStatusTable)
        names = {object_name(widget) for widget in qt.created}
        self.assertNotIn("runStatusTable", names)
        self.assertIn("runOutputFiles", names)
        self.assertIn("reviewStateLabel", names)

    def test_placeholders_use_the_new_wording(self) -> None:
        qt, layout, owner = self.build()
        self.assertEqual(
            by_name(qt, "reviewStateLabel").ctor_args,
            ("Recorded QC: consult report\nVisual review: pending",),
        )

    def test_stage_table_is_not_instantiated_in_results_panel(self) -> None:
        qt, layout, owner = self.build()
        self.assertIsNone(owner._runStatusTable)
        self.assertEqual([w for w in qt.created if w.kind == "QTableWidget"], [])
        self.assertNotIn("runStatusTable", {object_name(w) for w in qt.created})

    def test_report_bar_holds_open_report_and_open_output_tree(self) -> None:
        qt, layout, owner = self.build()

        def labels(row) -> list[str]:
            return [call[1][0].ctor_args[0] if call[1][0].ctor_args else "" for call in row.method_calls if call[0] == "addWidget"]

        rows = [widget for widget in qt.created if widget.kind == "QHBoxLayout"]
        report_row = next(row for row in rows if labels(row) == ["Open report", "Open output tree"])
        self.assertIsNotNone(report_row)
        self.assertEqual(owner._resultSceneButtons, [])

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
        self.assertEqual(texts, ["No runs in output folder yet", "Add a run folder…"])
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
        self.assertIn("Open Results again to rescan run folders", source)
        self.assertIn("(\"Rescan\", self._refresh_run_history)", source)
        self.assertIn("(\"Add a run folder…\", self._on_browse_run_directory_clicked)", source)


class AlgorithmDialogTest(unittest.TestCase):
    def open_dialog(self, language: str):
        qt = fake_qt()
        slicer = mock.MagicMock()
        owner = SimpleNamespace(
            _onboardingLanguage=language,
            _parameterNode=SimpleNamespace(ukfParameters="{}"),
        )
        with mock.patch.dict(sys.modules, {"qt": qt, "slicer": slicer}):
            ui_algorithm_picker.choose_algorithms(owner, "tractography")
        dialog_layout = next(widget for widget in qt.created if widget.kind == "QVBoxLayout")
        return qt, dialog_layout

    def test_picker_hides_note_until_error_and_aligns_cancel_apply_on_right(self) -> None:
        for language, cancel_text, apply_text in (("en", "Cancel", "Apply"), ("zh", "取消", "应用")):
            with self.subTest(language=language):
                qt, layout = self.open_dialog(language)
                note = by_name(qt, "algorithmPickerNote")
                note.hide.assert_called_once_with()
                cancel_btn = by_name(qt, "algorithmPickerCancelButton")
                apply_btn = by_name(qt, "algorithmPickerApplyButton")
                self.assertEqual(cancel_btn.ctor_args, (cancel_text,))
                self.assertEqual(apply_btn.ctor_args, (apply_text,))
                btn_row = next(
                    widget for widget in qt.created
                    if widget.kind == "QHBoxLayout" and "algorithmPickerApplyButton" in widget_names(widget)
                )
                self.assertEqual(
                    widget_names(btn_row),
                    ["algorithmPickerCancelButton", "algorithmPickerApplyButton"],
                )
                self.assertTrue(any(call[0] == "addStretch" for call in btn_row.method_calls))

    def test_picker_uses_scoped_scroll_transparency_and_styles_gear_comboboxes(self) -> None:
        source = (Path(__file__).resolve().parents[2] / "shared" / "ui_algorithm_picker.py").read_text(encoding="utf-8")
        self.assertNotIn('setStyleSheet("background: transparent;")', source)
        self.assertIn("QScrollArea#algorithmPickerScrollArea", source)
        self.assertIn("QWidget#algorithmPickerViewport", source)
        self.assertIn("QWidget#algorithmPickerBody", source)

        qt = fake_qt()
        slicer = mock.MagicMock(name="slicer")
        combo_factory = qt.QComboBox.side_effect
        spin_factory = qt.QDoubleSpinBox.side_effect

        class _FakeCombo:
            def __new__(cls, *args, **kwargs):
                return combo_factory(*args, **kwargs)

        class _FakeSpin:
            def __new__(cls, *args, **kwargs):
                return spin_factory(*args, **kwargs)

        qt.QComboBox = _FakeCombo
        qt.QDoubleSpinBox = _FakeSpin
        orig_combo = object.__new__(_FakeCombo)
        orig_combo.count = 2
        orig_combo.itemText = lambda idx: ("From sidecar (input/dwi.json)", "AP (anterior -> posterior)")[idx]
        orig_combo.itemData = lambda idx: ("auto", "AP")[idx]
        orig_combo.currentIndex = 0
        orig_combo.toolTip = "PE direction"
        orig_spin = object.__new__(_FakeSpin)
        orig_spin.minimum, orig_spin.maximum, orig_spin.decimals, orig_spin.singleStep, orig_spin.value = 0.0, 1.0, 4, 0.005, 0.05
        orig_spin.toolTip = "Readout time"
        orig_path = SimpleNamespace(currentPath="", filters=0, nameFilters=["*"], toolTip="Reverse PE")
        ui_ns = SimpleNamespace(
            preprocessingPhaseEncodingDirectionComboBox=orig_combo,
            preprocessingPhaseEncodingDirectionLabel=SimpleNamespace(text="Phase-encoding direction"),
            preprocessingReadoutTimeSpinBox=orig_spin,
            preprocessingReadoutTimeLabel=SimpleNamespace(text="Total readout time (s)"),
            preprocessingReversePhaseB0Path=orig_path,
            preprocessingReversePhaseB0Label=SimpleNamespace(text="Reverse-PE b=0 image"),
        )
        owner = SimpleNamespace(
            ui=ui_ns,
            _uiWidget=None,
            _pipelineActive=False,
            _selectedSetupActive=False,
            _onboardingLanguage="en",
            _parameterNode=SimpleNamespace(
                preprocessingPhaseEncodingDirection="auto",
                preprocessingReadoutTime=0.05,
                preprocessingReversePhaseB0="",
            ),
        )
        with (
            mock.patch.dict(sys.modules, {"qt": qt, "slicer": slicer, "ctk": mock.MagicMock()}),
            mock.patch("shared.ui_styles.style_combobox_popup") as style_popup_mock,
            mock.patch("shared.ui_styles.is_dark_mode", return_value=True),
        ):
            ui_algorithm_picker.choose_algorithms(owner, "preprocessing")

        style_popup_mock.assert_called()
        combo_editors = [w for w in qt.created if getattr(w, "kind", None) == "QComboBox"]
        self.assertEqual(len(combo_editors), 1)
        self.assertIn(mock.call(combo_editors[0], dark=True), style_popup_mock.call_args_list)
        scroll = by_name(qt, "algorithmPickerScrollArea")
        self.assertTrue(any(
            call[0] == "setStyleSheet" and call[1] == ("QScrollArea#algorithmPickerScrollArea { background: transparent; }",)
            for call in scroll.method_calls
        ))


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

        rows = [
            widget for widget in qt.created
            if widget.kind == "QHBoxLayout"
            and [call[0] for call in widget.method_calls if call[0] == "addWidget"] == ["addWidget", "addWidget", "addWidget"]
        ]
        self.assertEqual(len(rows), len(ui_algorithm_picker.FAMILIES))
        self.assertEqual(len(rows), 5)
        for box, row in zip(boxes.values(), rows):
            box.setFixedWidth.assert_called_once_with(indicator)
            calls = [(call[0], call[1]) for call in row.method_calls if call[0] == "addWidget"]
            self.assertEqual(len(calls), 3)
            self.assertEqual(calls[0][1], (box, 0, qt.Qt.AlignVCenter))
            self.assertEqual(calls[2][1][1:], (0, qt.Qt.AlignVCenter))
            row.setSpacing.assert_called_once_with(0)
        summary_insets = [
            call[1] for widget in qt.created if widget.kind == "QVBoxLayout"
            for call in widget.method_calls if call[0] == "setContentsMargins" and call[1][0] == gap
        ]
        self.assertEqual(len(summary_insets), 5)

    def _build_full_picker_owner(self, qt):
        qt.QDialog.Accepted = 1
        qt.Qt.Checked = 2
        qt.Qt.Unchecked = 0
        qt.Qt.PartiallyChecked = 1
        qt.QApplication.style.return_value.pixelMetric.side_effect = lambda _m: 18
        form = mock.MagicMock(name="targetProductsFormLayout")
        form.indexOf.return_value = 0
        form.labelForField.side_effect = lambda w: mock.MagicMock(name=f"label_for_{getattr(w, 'objectName', 'ctrl')}")

        def make_ui_checkbox(obj_name: str):
            cb = mock.MagicMock(name=obj_name)
            cb.objectName = obj_name
            cb.checked = False
            cb.isEnabled.return_value = True
            cb.setChecked.side_effect = lambda v, w=cb: setattr(w, "checked", bool(v))
            return cb

        ui_fields = {
            "targetProductsFormLayout": form,
            "targetSurfaceCheckBox": make_ui_checkbox("targetSurfaceCheckBox"),
            "targetSegmentationCheckBox": make_ui_checkbox("targetSegmentationCheckBox"),
            "targetParcellationCheckBox": make_ui_checkbox("targetParcellationCheckBox"),
            "targetSurfacePostprocessCheckBox": make_ui_checkbox("targetSurfacePostprocessCheckBox"),
        }
        for _, (box_name, _, _) in ui_algorithm_picker.FAMILIES.items():
            ui_fields[box_name] = make_ui_checkbox(box_name)

        node = SimpleNamespace(
            targetSurface=False,
            targetSegmentation=False,
            targetParcellation=False,
            targetSurfacePostprocess=False,
            targetPreprocessing=False,
            targetMicrostructure=False,
            targetTractography=False,
            targetBundles=False,
            targetConnectome=False,
            orchestrationPreprocessingStages="denoise,eddy,topup",
            orchestrationMicrostructureStages="dti",
            orchestrationTractographyBackend="csd_act",
            orchestrationBundleBackend="wmql",
            orchestrationConnectomeMetrics="sift2",
            ukfParameters="{}",
        )
        owner = SimpleNamespace(
            ui=SimpleNamespace(**ui_fields),
            _uiWidget=None,
            _parameterNode=node,
            _onboardingLanguage="en",
            _lastEnvironmentResult=None,
            _lastEnvironmentSignature=None,
            _refresh_advanced_run_settings=lambda: None,
            _schedule_common_user_settings_save=lambda: None,
            _sync_checkbox=lambda box, desired: box.setChecked(bool(desired)),
        )
        slicer = mock.MagicMock(name="slicer")
        with (
            mock.patch.dict(sys.modules, {"qt": qt, "slicer": slicer}),
            mock.patch.object(ui_algorithm_picker, "refresh_picker_icons"),
        ):
            ui_algorithm_picker.install_algorithm_pickers(owner)
        return owner, slicer

    def _run_picker_accept(self, qt, slicer, owner, family: str, chosen: set[str], *, cancel: bool = False):
        def fake_exec():
            if cancel:
                return 0
            for widget in qt.created:
                if widget.kind == "QCheckBox" and str(getattr(widget, "objectName", "")).startswith("algorithm_"):
                    key = widget.objectName.removeprefix("algorithm_")
                    widget.checked = key in chosen
            return qt.QDialog.Accepted

        orig_qdialog = qt.QDialog.side_effect

        def make_dialog(*args, **kwargs):
            dlg = orig_qdialog(*args, **kwargs)
            dlg.exec_.side_effect = fake_exec
            return dlg

        qt.QDialog.side_effect = make_dialog
        with mock.patch.dict(sys.modules, {"qt": qt, "slicer": slicer}):
            return ui_algorithm_picker.choose_algorithms(owner, family)

    def test_selecting_surfaces_only_checks_outer_box_and_preserves_subset_on_toggle(self) -> None:
        qt = fake_qt()
        owner, slicer = self._build_full_picker_owner(qt)
        outer_checkbox = owner._anatomicalPriorsRow[2]
        summary_label = owner._anatomicalPriorsRow[3]
        outer_checkbox.setTristate.assert_not_called()

        accepted = self._run_picker_accept(qt, slicer, owner, "anatomical_priors", {"targetSurface"})
        self.assertTrue(accepted)
        node = owner._parameterNode
        self.assertTrue(node.targetSurface)
        self.assertFalse(node.targetSegmentation)
        self.assertFalse(node.targetParcellation)
        self.assertTrue(outer_checkbox.checked)
        self.assertEqual(summary_label.setText.call_args[0][0], "Surfaces")

        # Clicking the checked box disables all 3 while remembering ("targetSurface",)
        on_clicked = outer_checkbox.clicked.connect.call_args.args[0]
        with mock.patch.dict(sys.modules, {"qt": qt, "slicer": slicer}):
            on_clicked(False)
        self.assertFalse(any((node.targetSurface, node.targetSegmentation, node.targetParcellation)))
        self.assertFalse(outer_checkbox.checked)
        self.assertEqual(summary_label.setText.call_args[0][0], "Surfaces")

        # Re-checking restores ONLY Surfaces, never Segmentation or Parcellation
        with mock.patch.dict(sys.modules, {"qt": qt, "slicer": slicer}):
            on_clicked(True)
        self.assertTrue(node.targetSurface)
        self.assertFalse(node.targetSegmentation)
        self.assertFalse(node.targetParcellation)
        self.assertTrue(outer_checkbox.checked)

        # Unchecking everything inside the popup and applying clears memory too
        accepted = self._run_picker_accept(qt, slicer, owner, "anatomical_priors", set())
        self.assertTrue(accepted)
        self.assertFalse(any((node.targetSurface, node.targetSegmentation, node.targetParcellation)))
        self.assertFalse(outer_checkbox.checked)
        self.assertEqual(summary_label.setText.call_args[0][0], "None selected")

        # Direct check with no remembered subset enables all 3 priors
        with mock.patch.dict(sys.modules, {"qt": qt, "slicer": slicer}):
            on_clicked(True)
        self.assertTrue(all((node.targetSurface, node.targetSegmentation, node.targetParcellation)))

    def test_accepting_family_picker_auto_checks_outer_checkbox_and_cancel_leaves_unchecked(self) -> None:
        qt = fake_qt()
        owner, slicer = self._build_full_picker_owner(qt)
        node = owner._parameterNode

        self.assertFalse(self._run_picker_accept(qt, slicer, owner, "tractography", {"ukf"}, cancel=True))
        self.assertFalse(node.targetTractography)
        self.assertFalse(owner.ui.targetTractographyCheckBox.checked)

        self.assertTrue(self._run_picker_accept(qt, slicer, owner, "tractography", {"ukf"}))
        self.assertEqual(node.orchestrationTractographyBackend, "ukf")
        self.assertTrue(node.targetTractography)
        self.assertTrue(owner.ui.targetTractographyCheckBox.checked)

        self.assertTrue(self._run_picker_accept(qt, slicer, owner, "bundles", {"wma"}))
        self.assertEqual(node.orchestrationBundleBackend, "wma")
        self.assertTrue(node.targetBundles)
        self.assertTrue(owner.ui.targetBundlesCheckBox.checked)


if __name__ == "__main__":
    unittest.main()
