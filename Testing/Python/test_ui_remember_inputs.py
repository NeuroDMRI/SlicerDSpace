#!/usr/bin/env python3
"""Package C: remember the input tiles across Slicer sessions (C1) and the Refresh button (C2).

The Prepare Inputs panel is built for real against a fake ``qt``; the widget methods are compiled from
``shared/module_widget.py`` (the module needs Slicer and Qt).
"""

from __future__ import annotations

import contextlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from shared import ui_prepare_inputs as prep
from shared.ui_onboarding import TRANSLATIONS, onboarding_text
from shared.ui_settings import SETTINGS_STATE_ATTRIBUTE
from shared.user_config import load_user_config, write_user_config
from test_b2_fix_round import widget_methods
from test_ui_run_history import by_name, fake_qt, object_name

TILES = ("nifti_source", "bval", "bvec", "nrrd_source", "raw", "dicom_source", "mask")
KINDS = ("nifti", "nrrd", "dicom")


class FakeEdit:
    """A ``QPlainTextEdit`` stand-in that keeps its text and properties and fires ``textChanged`` like Qt."""

    def __init__(self, *args) -> None:
        self.props: dict = {}
        self.text = ""
        self.slots: list = []
        self.textChanged = SimpleNamespace(connect=self.slots.append)
        self._rest = mock.MagicMock()

    def __getattr__(self, name):
        return getattr(self._rest, name)

    def property(self, name):
        return self.props.get(name)

    def setProperty(self, name, value) -> None:
        self.props[name] = value

    def toPlainText(self) -> str:
        return self.text

    def setPlainText(self, text) -> None:
        self.text = text
        for slot in list(self.slots):
            slot()

    def clear(self) -> None:
        self.setPlainText("")


class FakePathWidget:
    """``ctkPathLineEdit`` stand-in: an unblocked ``setCurrentPath`` fires ``currentPathChanged``."""

    def __init__(self) -> None:
        self.currentPath = ""
        self.blocked = False
        self.slots: list = []
        self.currentPathChanged = SimpleNamespace(connect=self.slots.append)

    def setCurrentPath(self, path) -> None:
        self.currentPath = path
        if not self.blocked:
            for slot in list(self.slots):
                slot(path)


class FakeBlocker:
    def __init__(self, widget) -> None:
        self.widget = widget
        widget.blocked = True

    def __del__(self) -> None:
        self.widget.blocked = False


BaseWidget = widget_methods("_prepare_inputs_widget_text", "_set_prepare_inputs_widget_text", "_sync_path_widget",
                            QSignalBlocker=FakeBlocker)


class Owner(BaseWidget):
    def __init__(self, ui) -> None:
        self.ui = ui
        self.saves = 0
        self.run_state_refreshes = 0
        self._onboardingLanguage = "en"
        self._pipelineActive = False
        self._parameterNode = SimpleNamespace(subjectId="sub-01", inputDirectory="/data/sub-01",
                                              orchestrationMicrostructureStages="dti,noddi")

    def _refresh_run_button_state(self, *args) -> None:
        self.run_state_refreshes += 1

    def _schedule_common_user_settings_save(self) -> None:
        self.saves += 1

    def _update_parameter_node(self, attribute, value) -> bool:
        return True

    def _onBatchRootPathChanged(self, *args) -> None:
        pass

    def _browse_prepare_input_source_path(self, *args) -> None:
        pass

    def _browse_prepare_input_file(self, *args) -> None:
        pass

    def appendLog(self, text) -> None:
        self.logs.append(text)

    logs: list = []


def make_ui() -> SimpleNamespace:
    return SimpleNamespace(
        configurationCollapsibleButton=mock.MagicMock(), configurationFormLayout=mock.MagicMock(),
        subjectIdLabel=mock.MagicMock(), subjectIdLineEdit=mock.MagicMock(), inputDirectoryPath=FakePathWidget(),
        outputDirectoryLabel=mock.MagicMock(), outputDirectoryPath=mock.MagicMock(),
    )


class PanelCase(unittest.TestCase):
    """A real Prepare Inputs panel on a fake ``qt``."""

    def setUp(self) -> None:
        self.qt = fake_qt()
        self.qt.QPlainTextEdit.side_effect = FakeEdit
        self.qt.QLabel.side_effect = lambda *args: mock.MagicMock(name="QLabel")
        self.qt.QTabBar.return_value.tabData.side_effect = lambda index: KINDS[index]
        modules = mock.patch.dict(sys.modules, {"qt": self.qt, "ctk": mock.MagicMock()})
        modules.start()
        self.addCleanup(modules.stop)
        compact = mock.patch("shared.ui_compact.refresh_compact_algorithms")
        self.compact = compact.start()
        self.addCleanup(compact.stop)
        dark = mock.patch("shared.ui_styles.is_dark_mode", return_value=False)
        dark.start()
        self.addCleanup(dark.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.owner = Owner(make_ui())
        self.owner.logs = []

    def install(self) -> dict:
        prep.install_prepare_inputs_panel(self.owner, mock.MagicMock())
        return self.owner._prepareInputWidgets

    def hint(self, key):
        return self.owner._prepareInputWidgets[key + "_hint"]

    def tile_texts(self) -> dict:
        return {key: self.owner._prepare_inputs_widget_text(key) for key in TILES}


# ----- C1: save -----------------------------------------------------------------------------------------------


def save_method(update_calls: list):
    return widget_methods(
        "_save_common_user_settings",
        slicer=SimpleNamespace(app=SimpleNamespace(userSettings=mock.Mock())),
        ENVIRONMENT_DEPENDENCY_PATHS=(),
        update_user_config=lambda payload: (update_calls.append(payload) or ({}, Path("/none/config.json"))),
    )


class SaveTest(PanelCase):
    def save(self, node_extra=None, owner_extra=None) -> dict:
        calls: list = []
        Save = save_method(calls)
        node = SimpleNamespace(
            orchestrationMicrostructureStages="dti", subjectId="s", inputDirectory="/d", outputDirectory="/o",
            orchestrationRunName="r", ddsegDevice="cpu", device="cpu", orchestrationTractographyBackend="csd_act",
            orchestrationBundleBackend="wmql", orchestrationSelectStreamlines=1000, autoInstallPythonDependencies=False,
            connectomeBackend="python", **(node_extra or {}))
        owner = self.owner
        owner._parameterNode = node
        owner._common_user_setting_specs = lambda: ()
        owner._user_settings_key = lambda name: name
        owner._requested_product_targets = lambda: ("microstructure",)
        owner._selected_product_targets = lambda: ("microstructure",)
        for name, value in (owner_extra or {}).items():
            setattr(owner, name, value)
        Save._save_common_user_settings(owner)
        self.assertEqual(len(calls), 1)
        return calls[0]

    def test_c1_save_writes_the_kind_every_tile_text_and_the_three_preprocessing_values(self) -> None:
        self.install()
        for key, value in (("nifti_source", "/data/s1/dwi.nii.gz"), ("bval", "/data/s1/dwi.bval"),
                           ("bvec", "/data/s1/dwi.bvec"), ("mask", "/data/s1/mask.nii.gz")):
            self.owner._set_prepare_inputs_widget_text(key, value)
        recent = self.save(dict(preprocessingReversePhaseB0="/data/rev.nii.gz",
                                preprocessingPhaseEncodingDirection="AP", preprocessingReadoutTime=0.05))["recent"]
        self.assertEqual(recent["input_sources"], {
            "kind": "nifti", "nifti_source": "/data/s1/dwi.nii.gz", "bval": "/data/s1/dwi.bval",
            "bvec": "/data/s1/dwi.bvec", "nrrd_source": "", "raw": "", "dicom_source": "", "mask": "/data/s1/mask.nii.gz"})
        self.assertEqual(recent["reverse_pe_b0"], "/data/rev.nii.gz")
        self.assertEqual(recent["pe_direction"], "AP")
        self.assertEqual(recent["readout_time"], 0.05)
        self.assertEqual(recent["subject_id"], "s")  # the existing values are still written

    def test_c1_save_writes_the_full_path_not_the_file_name_shown_in_the_tile(self) -> None:
        fields = self.install()
        self.owner._set_prepare_inputs_widget_text("dicom_source", "/data/long/dicom")
        self.assertEqual(fields["dicom_source"].toPlainText(), "dicom")
        prep.show_input_format(self.owner, "dicom")
        sources = self.save()["recent"]["input_sources"]
        self.assertEqual((sources["kind"], sources["dicom_source"]), ("dicom", "/data/long/dicom"))

    def test_c1_a_cleared_tile_is_written_as_empty_so_a_stale_path_is_replaced(self) -> None:
        self.install()
        self.owner._set_prepare_inputs_widget_text("bval", "/data/old.bval")
        self.owner._prepareInputWidgets["bval"].clear()
        self.assertEqual(self.save()["recent"]["input_sources"]["bval"], "")

    def test_c1_save_leaves_input_sources_out_without_a_panel(self) -> None:
        recent = self.save()["recent"]  # no panel: the K4 owner shape
        self.assertNotIn("input_sources", recent)
        self.assertEqual((recent["reverse_pe_b0"], recent["pe_direction"], recent["readout_time"]), ("", "", 0.0))

    def test_c1_save_keeps_stored_tiles_while_a_restore_is_waiting_for_the_panel(self) -> None:
        self.install()
        self.owner._pendingInputSources = {"kind": "nifti", "nifti_source": "/data/waiting.nii.gz"}
        self.assertNotIn("input_sources", self.save()["recent"])


# ----- C1: restore --------------------------------------------------------------------------------------------


class FakeRawNode:
    def __init__(self, state) -> None:
        self.attrs = {SETTINGS_STATE_ATTRIBUTE: state} if state else {}

    def GetAttribute(self, name):
        return self.attrs.get(name)

    def SetAttribute(self, name, value) -> None:
        self.attrs[name] = value


def make_node(state="pending", **extra):
    values = dict(
        subjectId="", inputDirectory="", outputDirectory="", orchestrationRunName="", brainMaskBackend="",
        orchestrationTractographyBackend="csd_act", orchestrationBundleBackend="wmql", connectomeBackend="auto",
        orchestrationSelectStreamlines=1_000_000, autoInstallPythonDependencies=True, preprocessingReversePhaseB0="",
        preprocessingPhaseEncodingDirection="", preprocessingReadoutTime=0.0, orchestrationMicrostructureStages="dti")
    values.update(extra)
    return SimpleNamespace(parameterNode=FakeRawNode(state), allParameters=(), **values)


class RestoreTest(PanelCase):
    SOURCES = {"kind": "dicom", "dicom_source": "/data/s1/dicom", "nifti_source": "/data/s1/dwi.nii.gz",
               "bval": "/data/s1/dwi.bval", "bvec": "/data/s1/dwi.bvec", "nrrd_source": "", "raw": "", "mask": ""}

    def restore(self, recent, node=None, *, panel=True):
        config_path = self.root / "user_config.json"
        write_user_config({"recent": {"subject_id": "sub-01", "subject_root": "/data/sub-01",
                                      "output_root": "/data/out", **recent}}, config_path)
        Restore = widget_methods(
            "_restore_common_user_settings_to_node",
            __file__="module_widget.py",
            slicer=SimpleNamespace(
                app=SimpleNamespace(userSettings=lambda: SimpleNamespace(contains=lambda key: False, remove=mock.Mock(),
                                                                         setValue=mock.Mock())),
                util=SimpleNamespace(NodeModify=lambda node: contextlib.nullcontext())),
            default_user_config_path=lambda: config_path, load_user_config=load_user_config,
            write_user_config=write_user_config, extension_root_for_module=lambda path: self.root,
            prune_stale_bundled_dependencies=lambda deps, **kw: (dict(deps), []), ENVIRONMENT_DEPENDENCY_PATHS=(),
            REMOVED_BRAIN_MASK_BACKENDS=(), selected_microstructure_stages=lambda stages: tuple(stages),
            consume_user_defaults_request=__import__("shared.ui_settings", fromlist=["x"]).consume_user_defaults_request)
        owner = self.owner
        owner.__class__ = type("RestoreOwner", (Owner, Restore), {})
        owner.logs = []
        owner._parameterNode = node or make_node()
        owner._environment_settings_key = lambda dependency_id: dependency_id
        owner._user_settings_key = lambda name: name
        owner._common_user_setting_specs = lambda: ()
        if panel:
            self.install()
        owner._restore_common_user_settings_to_node()
        self.assertEqual(owner.logs, [])
        return owner

    def test_c1_restore_on_a_new_empty_node_sets_the_tiles_the_tab_and_the_three_attributes(self) -> None:
        owner = self.restore({"input_sources": self.SOURCES, "reverse_pe_b0": "/data/rev.nii.gz",
                              "pe_direction": "PA", "readout_time": 0.04})
        self.assertEqual(owner._prepareInputWidgets["source_kind"], "dicom")
        self.assertEqual(self.tile_texts(), {**{key: "" for key in TILES}, **{
            key: value for key, value in self.SOURCES.items() if key in TILES and value}})
        self.assertEqual(owner._prepareInputWidgets["dicom_source"].toPlainText(), "dicom")  # the tile shows the name
        node = owner._parameterNode
        self.assertEqual(node.preprocessingReversePhaseB0, "/data/rev.nii.gz")
        self.assertEqual(node.preprocessingPhaseEncodingDirection, "PA")
        self.assertEqual(node.preprocessingReadoutTime, 0.04)
        self.assertEqual(node.subjectId, "sub-01")  # the existing restore still runs

    def test_c1_empty_saved_preprocessing_values_leave_the_node_defaults(self) -> None:
        owner = self.restore({"input_sources": {"kind": "nifti"}},
                             make_node(preprocessingPhaseEncodingDirection="LR", preprocessingReadoutTime=0.07))
        node = owner._parameterNode
        self.assertEqual((node.preprocessingReversePhaseB0, node.preprocessingPhaseEncodingDirection,
                          node.preprocessingReadoutTime), ("", "LR", 0.07))

    def test_c1_restore_does_nothing_for_a_node_that_is_not_pending(self) -> None:
        for state in ("applied", None):
            with self.subTest(state=state):
                owner = self.restore({"input_sources": self.SOURCES, "pe_direction": "PA"}, make_node(state))
                self.assertEqual(self.tile_texts(), {key: "" for key in TILES})
                self.assertEqual(owner._prepareInputWidgets["source_kind"], "nifti")
                self.assertEqual(owner._parameterNode.preprocessingPhaseEncodingDirection, "")
                self.assertIsNone(getattr(owner, "_pendingInputSources", None))
                self.setUp()  # a fresh panel for the next state

    def test_c1_restore_before_the_panel_exists_applies_the_values_when_the_panel_is_installed_once(self) -> None:
        owner = self.restore({"input_sources": self.SOURCES}, panel=False)
        self.assertIsNone(getattr(owner, "_prepareInputWidgets", None))
        self.assertEqual(owner._pendingInputSources["dicom_source"], "/data/s1/dicom")
        self.install()
        self.assertEqual(owner._prepareInputWidgets["source_kind"], "dicom")
        self.assertEqual(self.tile_texts()["dicom_source"], "/data/s1/dicom")
        self.assertIsNone(owner._pendingInputSources)
        owner._prepareInputWidgets["dicom_source"].clear()  # the user edits; a later call must not bring it back
        self.assertFalse(prep.apply_pending_input_sources(owner))
        self.assertEqual(self.tile_texts()["dicom_source"], "")

    def test_c1_after_a_restore_the_prepared_selection_is_unset_and_run_goes_to_the_import_gate(self) -> None:
        owner = self.restore({"input_sources": {"kind": "nifti", "nifti_source": "/data/s1/dwi.nii.gz"}})
        self.assertIsNone(getattr(owner, "_preparedInputSelection", None))
        Run = widget_methods("_onRunSelectedClicked")
        gate = []
        owner._input_import_selection = lambda: ("nifti", "/data/s1/dwi.nii.gz")
        owner._on_prepare_inputs_clicked = lambda **kwargs: gate.append(kwargs)
        owner.logic = object()
        owner._selectedSetupActive = False
        Run._onRunSelectedClicked(owner)
        self.assertEqual(gate, [{"run_after": True}])

    def test_c1_restore_does_not_clear_the_tiles_through_the_hidden_input_folder_widget(self) -> None:
        owner = self.restore({"input_sources": self.SOURCES})
        folder = owner.ui.inputDirectoryPath
        self.assertEqual(len(folder.slots), 1)  # use_folder is connected: an unblocked set would clear every tile
        before = self.tile_texts()
        owner._sync_path_widget(folder, owner._parameterNode.inputDirectory)  # what _checkCanApply does next
        self.assertEqual(folder.currentPath, "/data/sub-01")
        self.assertEqual(self.tile_texts(), before)
        self.assertEqual(before["dicom_source"], "/data/s1/dicom")

    def test_c1_one_failing_tile_does_not_stop_the_other_tiles_and_is_logged(self) -> None:
        owner = self.restore({"input_sources": {"kind": "nifti", "nifti_source": "/data/a.nii.gz", "bval": "/d/a.bval"}},
                             panel=False)
        self.install_with_failing_tile(owner, "nifti_source")
        self.assertEqual(owner._prepare_inputs_widget_text("bval"), "/d/a.bval")
        self.assertTrue(any("nifti_source" in line for line in owner.logs), owner.logs)

    def install_with_failing_tile(self, owner, failing):
        original = owner._set_prepare_inputs_widget_text

        def set_text(key, value):
            if key == failing:
                raise RuntimeError("tile broke")
            original(key, value)

        owner._set_prepare_inputs_widget_text = set_text
        self.install()

    def fill_subject_a(self) -> None:
        """The reused panel after a scene close still shows subject A (``onSceneEndClose`` keeps the panel)."""
        self.install()
        for key, value in (("nifti_source", "/data/A/dwi.nii.gz"), ("bval", "/data/A/dwi.bval"),
                           ("bvec", "/data/A/dwi.bvec"), ("mask", "/data/A/mask.nii.gz")):
            self.owner._set_prepare_inputs_widget_text(key, value)

    def test_fc1_restore_clears_the_tiles_that_the_saved_selection_leaves_empty_or_out(self) -> None:
        """FC1: B's DWI with empty bval/bvec must not keep A's gradient files (or A's mask, a missing key)."""
        self.fill_subject_a()
        self.restore({"input_sources": {"kind": "nifti", "nifti_source": "/data/B/dwi.nii.gz", "bval": "", "bvec": ""}},
                     panel=False)
        self.assertEqual(self.tile_texts(), {**{key: "" for key in TILES}, "nifti_source": "/data/B/dwi.nii.gz"})

    def test_fc1_a_saved_kind_alone_clears_every_tile(self) -> None:
        """FC1: a saved selection with only the format tab reproduces exactly that: no tile path."""
        self.fill_subject_a()
        owner = self.restore({"input_sources": {"kind": "nrrd"}}, panel=False)
        self.assertEqual(self.tile_texts(), {key: "" for key in TILES})
        self.assertEqual(owner._prepareInputWidgets["source_kind"], "nrrd")

    def test_fc1_a_waiting_restore_also_clears_the_empty_tiles_when_the_panel_is_installed(self) -> None:
        """FC1: the deferred path (restore before the panel exists) follows the same exact-selection rule."""
        owner = self.restore({"input_sources": {"kind": "nifti", "nifti_source": "/data/B/dwi.nii.gz"}}, panel=False)
        original = owner._set_prepare_inputs_widget_text
        calls = []
        owner._set_prepare_inputs_widget_text = lambda key, value: (calls.append((key, value)), original(key, value))
        self.install()
        self.assertEqual(sorted(key for key, _ in calls), sorted(TILES))  # every tile key is set
        self.assertEqual(self.tile_texts(), {**{key: "" for key in TILES}, "nifti_source": "/data/B/dwi.nii.gz"})

    def test_fc1_guard_an_old_config_without_input_sources_leaves_the_tiles_as_they_are(self) -> None:
        """FC1 guard (passes before and after the fix): no saved ``input_sources`` changes no tile."""
        self.fill_subject_a()
        before = self.tile_texts()
        self.restore({}, panel=False)
        self.assertEqual(self.tile_texts(), before)
        self.assertEqual(before["bval"], "/data/A/dwi.bval")

    def test_c1_a_bad_saved_kind_is_ignored_and_the_tiles_still_load(self) -> None:
        owner = self.restore({"input_sources": {"kind": "bogus", "bval": "/data/s1/dwi.bval"}})
        self.assertEqual(owner._prepareInputWidgets["source_kind"], "nifti")
        self.assertEqual(self.tile_texts()["bval"], "/data/s1/dwi.bval")


# ----- C1: save triggers ----------------------------------------------------------------------------------------


class SaveTriggerTest(PanelCase):
    def test_c1_a_tile_edit_schedules_the_debounced_save(self) -> None:
        self.install()
        self.assertEqual(self.owner.saves, 0)
        for index, key in enumerate(TILES, start=1):
            before = self.owner.saves
            self.owner._set_prepare_inputs_widget_text(key, f"/data/{key}")
            self.assertGreater(self.owner.saves, before, key)

    def test_c1_a_format_switch_schedules_the_debounced_save(self) -> None:
        fields = self.install()
        self.assertEqual(self.owner.saves, 0)  # building the panel saves nothing
        prep.show_input_format(self.owner, "nifti")
        self.assertEqual(self.owner.saves, 0)  # an unchanged kind saves nothing
        update_format = fields["format_tabs"].currentChanged.connect.call_args[0][0]
        update_format(2)  # the user clicks the DICOM tab
        self.assertEqual(fields["source_kind"], "dicom")
        self.assertEqual(self.owner.saves, 1)

    def test_c1_thin_owners_without_the_save_hook_still_work(self) -> None:
        self.install()
        with mock.patch.object(Owner, "_schedule_common_user_settings_save", None):
            self.owner._set_prepare_inputs_widget_text("bval", "/data/x.bval")
            prep.show_input_format(self.owner, "nrrd")
        self.assertEqual(self.tile_texts()["bval"], "/data/x.bval")


# ----- C1.5: missing-file hint --------------------------------------------------------------------------------------


class MissingFileHintTest(PanelCase):
    def test_c1_5_a_saved_path_that_no_longer_exists_keeps_its_text_and_shows_file_not_found(self) -> None:
        fields = self.install()
        missing = self.root / "gone" / "dwi.nii.gz"
        self.owner._set_prepare_inputs_widget_text("nifti_source", str(missing))
        self.assertEqual(self.owner._prepare_inputs_widget_text("nifti_source"), str(missing))
        self.assertEqual(fields["nifti_source"].toPlainText(), "dwi.nii.gz")
        hint = self.hint("nifti_source")
        self.assertEqual(hint.setText.call_args[0][0], "File not found")
        self.assertEqual(hint.setVisible.call_args[0][0], True)
        fields["nifti_source_tile"].setProperty.assert_any_call("missing", True)

    def test_c1_5_the_children_that_the_missing_rules_style_are_repolished_with_the_tile(self) -> None:
        fields = self.install()
        self.owner._set_prepare_inputs_widget_text("bval", str(self.root / "gone.bval"))
        for widget in (fields["bval_tile"], fields["bval_hint"], fields["bval"]):
            widget.style.return_value.unpolish.assert_called_with(widget)
            widget.style.return_value.polish.assert_called_with(widget)
        self.assertIn('QWidget#dspaceInputTile[missing="true"] QLabel#dspaceInputHint',
                      self.owner.ui.configurationCollapsibleButton.setStyleSheet.call_args[0][0])

    def test_c1_5_an_existing_file_or_folder_shows_no_hint(self) -> None:
        self.install()
        present = self.root / "dwi.nii.gz"
        present.write_text("x")
        folder = self.root / "dicom"
        folder.mkdir()
        for key, path in (("nifti_source", present), ("dicom_source", folder)):
            self.owner._set_prepare_inputs_widget_text(key, str(path))
            self.assertEqual(self.hint(key).setVisible.call_args[0][0], False, key)
            self.owner._prepareInputWidgets[key + "_tile"].setProperty.assert_any_call("missing", False)

    def test_c1_5_clearing_the_tile_brings_the_normal_hint_back(self) -> None:
        self.install()
        self.owner._set_prepare_inputs_widget_text("bval", str(self.root / "gone.bval"))
        self.owner._prepareInputWidgets["bval"].clear()
        hint = self.hint("bval")
        self.assertEqual(hint.setText.call_args[0][0], ".bval / bvals")
        self.assertEqual(hint.setVisible.call_args[0][0], True)

    def test_c1_5_the_text_follows_the_language(self) -> None:
        self.install()
        self.owner._set_prepare_inputs_widget_text("bval", str(self.root / "gone.bval"))
        self.owner._onboardingLanguage = "zh"
        prep.apply_input_language(self.owner)
        self.assertEqual(self.hint("bval").setText.call_args[0][0], "文件不存在")
        self.assertEqual(onboarding_text("zh", "input_file_missing"), "文件不存在")

    def test_fc2_a_path_whose_home_cannot_be_resolved_is_shown_as_missing_and_the_slot_finishes(self) -> None:
        """FC2: ``Path("~nouser/...").expanduser()`` raises RuntimeError; the hint must not raise.

        The error is forced with a patch: on Windows ``~otheruser`` can expand without an error (CI runs all OSes).
        """
        fields = self.install()
        text = "~nouser_fc2_zz/dwi.nii.gz"
        self.compact.reset_mock()
        saves, refreshes = self.owner.saves, self.owner.run_state_refreshes
        with mock.patch.object(Path, "expanduser", side_effect=RuntimeError("Could not determine home directory.")):
            self.owner._set_prepare_inputs_widget_text("nifti_source", text)
        self.assertEqual(self.owner._prepare_inputs_widget_text("nifti_source"), text)  # the text stays
        self.assertEqual(self.hint("nifti_source").setText.call_args[0][0], "File not found")
        fields["nifti_source_tile"].setProperty.assert_any_call("missing", True)
        self.assertGreater(self.owner.saves, saves)
        self.assertGreater(self.owner.run_state_refreshes, refreshes)
        self.compact.assert_called()

    def test_fc2_a_failing_hint_update_does_not_skip_the_rest_of_the_tile_slot(self) -> None:
        """FC2: status hide, run-button refresh, algorithm refresh and the save trigger still run."""
        fields = self.install()
        status = fields["status"]
        status.hide.reset_mock()
        self.compact.reset_mock()
        saves = self.owner.saves
        with mock.patch.object(prep, "update_input_hint", side_effect=RuntimeError("hint broke")):
            self.owner._set_prepare_inputs_widget_text("bval", "/data/x.bval")
        status.hide.assert_called()
        self.compact.assert_called()
        self.assertGreater(self.owner.saves, saves)
        self.assertEqual(self.owner._prepare_inputs_widget_text("bval"), "/data/x.bval")

    def test_c1_5_recheck_notices_a_file_that_appears_or_disappears(self) -> None:
        self.install()
        late = self.root / "late.bval"
        self.owner._set_prepare_inputs_widget_text("bval", str(late))
        self.assertEqual(self.hint("bval").setText.call_args[0][0], "File not found")
        late.write_text("0 1000")
        prep.refresh_input_hints(self.owner)
        self.assertEqual(self.hint("bval").setVisible.call_args[0][0], False)
        late.unlink()
        prep.refresh_input_hints(self.owner)
        self.assertEqual(self.hint("bval").setText.call_args[0][0], "File not found")
        self.assertEqual(self.owner._prepare_inputs_widget_text("bval"), str(late))  # the text is never changed


# ----- C2: Refresh button -----------------------------------------------------------------------------------------


class RefreshButtonTest(PanelCase):
    def test_c2_the_button_exists_with_name_tooltip_in_both_languages_and_an_icon(self) -> None:
        fields = self.install()
        button = by_name(self.qt, "inputsRefreshButton")
        self.assertEqual(button.kind, "QToolButton")
        self.assertIs(fields["refresh_button"], button)
        self.assertEqual(onboarding_text("en", "refresh_inputs_tooltip"),
                         "Refresh: re-check the input files and rescan the results")
        self.assertEqual(onboarding_text("zh", "refresh_inputs_tooltip"), "刷新：重新检查输入文件并重新扫描结果")
        self.assertEqual(button.setToolTip.call_args[0][0], onboarding_text("en", "refresh_inputs_tooltip"))
        self.assertEqual(button.setAccessibleName.call_args[0][0], onboarding_text("en", "refresh_inputs_tooltip"))
        self.assertIn((button, "tooltip", "refresh_inputs_tooltip"), self.owner._uiLanguageBindings)
        self.owner._onboardingLanguage = "zh"
        prep.apply_input_language(self.owner)
        self.assertEqual(button.setAccessibleName.call_args[0][0], onboarding_text("zh", "refresh_inputs_tooltip"))
        button.setIcon.assert_called()
        button.setToolButtonStyle.assert_called_with(self.qt.Qt.ToolButtonIconOnly)
        button.setIconSize.assert_called_once()
        button.setFixedSize.assert_called_once_with(28, 22)

    def test_c2_no_run_history_refresh_button_appears(self) -> None:
        self.install()
        self.assertEqual([w for w in self.qt.created
                          if w.kind == "QPushButton" and object_name(w).startswith("runHistory")], [])
        self.assertFalse(hasattr(self.owner, "_runHistoryRefreshButton"))

    def test_c2_the_button_sits_after_the_output_folder_widget_in_its_row_and_is_not_its_child(self) -> None:
        self.install()
        button = by_name(self.qt, "inputsRefreshButton")
        output = self.owner.ui.outputDirectoryPath
        rows = [w for w in self.qt.created if w.kind == "QHBoxLayout"
                and [c[1][0] for c in w.method_calls if c[0] == "addWidget"][:2] == [output, button]]
        self.assertEqual(len(rows), 1)
        self.assertNotIn(output, button.ctor_args)
        field_calls = [c for c in self.owner.ui.configurationFormLayout.setWidget.call_args_list
                       if c[0][1] == self.qt.QFormLayout.FieldRole and c[0][2] is not self.owner.ui.subjectIdLineEdit]
        self.assertTrue(any(c[0][0] == 2 for c in field_calls), field_calls)  # the Output folder row

    def test_c2_the_icon_is_redrawn_in_the_ink_of_each_theme(self) -> None:
        self.install()
        button = by_name(self.qt, "inputsRefreshButton")
        inks = []
        for dark in (True, False, True):
            before = button.setIcon.call_count
            self.qt.QByteArray.reset_mock()
            prep.refresh_prepare_inputs_styles(self.owner, dark=dark)
            self.assertEqual(button.setIcon.call_count, before + 1)
            inks.append(self.qt.QByteArray.call_args[0][0].decode())
        self.assertIn("#8CD2F5", inks[0])  # the ink of the dspaceFolderBrowse button beside it
        self.assertIn("#326A8A", inks[1])
        self.assertEqual(inks[0], inks[2])
        self.assertIn("inputsRefreshButton", self.owner.ui.configurationCollapsibleButton.setStyleSheet.call_args[0][0])


class RefreshClickTest(unittest.TestCase):
    """``_on_refresh_inputs_clicked``: the order of the steps, one failing step, and the busy case."""

    def make_owner(self, calls, **extra):
        owner = SimpleNamespace(
            _onboardingLanguage="en", _pipelineActive=False, _setupJob=None, _selectedSetupActive=False,
            _inputImportPending=False, _inputInspectionCache=object(), _algorithmGateCache=("key", 1.0, {}),
            _refresh_input_preview_controls=lambda: calls.append("preview"),
            _refresh_run_button_state=lambda *args: calls.append("run_state"),
            _refresh_run_history=lambda *args, **kwargs: calls.append("history"),
            appendLog=lambda text: calls.append(("log", text)))
        for name, value in extra.items():
            setattr(owner, name, value)
        return owner

    @contextlib.contextmanager
    def patched(self, calls, fail=()):
        def step(name):
            def run(*args, **kwargs):
                calls.append(name)
                if name in fail:
                    raise RuntimeError(f"{name} broke")
            return run

        with mock.patch("shared.algorithm_availability.clear_cache", step("clear_cache")), \
                mock.patch("shared.ui_prepare_inputs.refresh_input_hints", step("hints")), \
                mock.patch("shared.ui_compact.refresh_compact_algorithms", step("compact")):
            yield

    def test_c2_a_click_runs_the_steps_in_order_and_logs_one_line(self) -> None:
        calls: list = []
        owner = self.make_owner(calls)
        with self.patched(calls):
            prep.on_refresh_inputs_clicked(owner)
        self.assertEqual(calls, ["clear_cache", "hints", "preview", "run_state", "compact", "history",
                                 ("log", "Refreshed inputs and results.")])
        self.assertIsNone(owner._inputInspectionCache)
        self.assertIsNone(owner._algorithmGateCache)

    def test_c2_the_log_line_follows_the_language(self) -> None:
        calls: list = []
        owner = self.make_owner(calls, _onboardingLanguage="zh")
        with self.patched(calls):
            prep.on_refresh_inputs_clicked(owner)
        self.assertEqual(calls[-1], ("log", "已刷新输入和结果。"))

    def test_c2_a_failing_step_does_not_stop_the_later_steps(self) -> None:
        expected = ["clear_cache", "hints", "preview", "run_state", "compact", "history"]
        for failing in expected:
            with self.subTest(failing=failing):
                calls: list = []
                owner = self.make_owner(calls)
                for attribute, name in (("_refresh_input_preview_controls", "preview"),
                                        ("_refresh_run_button_state", "run_state"),
                                        ("_refresh_run_history", "history")):
                    if name == failing:
                        def broken(*args, _name=name, **kwargs):
                            calls.append(_name)
                            raise RuntimeError("broke")
                        setattr(owner, attribute, broken)
                with self.patched(calls, fail={failing}):
                    prep.on_refresh_inputs_clicked(owner)
                self.assertEqual([c for c in calls if isinstance(c, str)], expected)
                logs = [c[1] for c in calls if isinstance(c, tuple)]
                self.assertEqual(logs[-1], "Refreshed inputs and results.")
                self.assertTrue(any("broke" in line for line in logs[:-1]), logs)

    def test_c2_the_caches_are_dropped_even_when_clear_cache_fails(self) -> None:
        calls: list = []
        owner = self.make_owner(calls)
        with self.patched(calls, fail={"clear_cache"}):
            prep.on_refresh_inputs_clicked(owner)
        self.assertIsNone(owner._inputInspectionCache)
        self.assertIsNone(owner._algorithmGateCache)

    def test_c2_during_a_run_a_setup_job_or_an_import_only_steps_1_5_and_6_run(self) -> None:
        for busy in ({"_pipelineActive": True}, {"_setupJob": object()}, {"_selectedSetupActive": True},
                     {"_inputImportPending": True}):
            with self.subTest(busy=list(busy)):
                calls: list = []
                owner = self.make_owner(calls, **busy)
                with self.patched(calls):
                    prep.on_refresh_inputs_clicked(owner)
                self.assertEqual(calls, ["clear_cache", "history", ("log", "Refreshed inputs and results.")])
                self.assertIsNone(owner._inputInspectionCache)
                self.assertIsNone(owner._algorithmGateCache)

    def test_c2_the_owner_method_hands_the_click_to_the_handler(self) -> None:
        Widget = widget_methods("_on_refresh_inputs_clicked")
        owner = Widget()
        with mock.patch("shared.ui_prepare_inputs.on_refresh_inputs_clicked") as handler:
            owner._on_refresh_inputs_clicked()
        handler.assert_called_once_with(owner)


class RefreshKeepsTheInputsTest(PanelCase):
    def test_c2_a_click_changes_no_tile_no_tab_no_subject_id_no_selection_and_no_prepared_selection(self) -> None:
        fields = self.install()
        prep.show_input_format(self.owner, "dicom")
        for key, value in (("dicom_source", str(self.root / "gone")), ("bval", "/data/x.bval"), ("mask", "/m.nii.gz")):
            self.owner._set_prepare_inputs_widget_text(key, value)
        owner = self.owner
        owner._preparedInputSelection = ("dicom", "/data/prepared")
        owner._inputInspectionCache = object()
        owner._refresh_input_preview_controls = lambda: None
        owner._refresh_run_history = lambda *args, **kwargs: None
        node = owner._parameterNode
        before = (self.tile_texts(), fields["source_kind"], node.subjectId, node.orchestrationMicrostructureStages,
                  owner._preparedInputSelection, node.inputDirectory)
        for _ in range(2):
            prep.on_refresh_inputs_clicked(owner)
        after = (self.tile_texts(), fields["source_kind"], node.subjectId, node.orchestrationMicrostructureStages,
                 owner._preparedInputSelection, node.inputDirectory)
        self.assertEqual(before, after)
        self.assertEqual(self.hint("dicom_source").setText.call_args[0][0], "File not found")  # step 2 re-checked


class TranslationTest(unittest.TestCase):
    def test_new_texts_exist_in_english_and_chinese(self) -> None:
        for key in ("refresh_inputs_tooltip", "inputs_refreshed", "input_file_missing"):
            for language in ("en", "zh"):
                self.assertIn(key, TRANSLATIONS[language], (key, language))


if __name__ == "__main__":
    unittest.main()
