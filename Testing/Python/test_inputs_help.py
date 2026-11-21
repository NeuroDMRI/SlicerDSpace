#!/usr/bin/env python3
"""b2 inputs-help: the "?" buttons of Subject ID, Batch Folder and the batch Output Folder.

Three groups: the texts (en/zh, key phrases), the truth of the texts (checked against the code that defines the rules),
and the buttons on a Prepare Inputs panel built for real on a fake ``qt``.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import test_ui_remember_inputs as remember
from shared import ui_prepare_inputs as prep
from shared.ui_onboarding import TRANSLATIONS, onboarding_text

REPO_ROOT = Path(__file__).resolve().parents[2]
KEYS = ("subject_id", "batch_folder", "output_folder")


def plain(html: str) -> str:
    import re

    text = re.sub(r"<[^>]+>", "", html)
    return text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")


class HelpTextTest(unittest.TestCase):
    """b2 inputs-help items 1-2: English and Chinese texts through the TRANSLATIONS pattern."""

    def test_b2_inputs_each_key_exists_in_both_languages_and_differs(self):
        for key in KEYS:
            for suffix in ("", "_name"):
                name = f"help_{key}{suffix}"
                self.assertIn(name, TRANSLATIONS["en"])
                self.assertIn(name, TRANSLATIONS["zh"])
                self.assertNotEqual(onboarding_text("en", name), onboarding_text("zh", name))
                self.assertTrue(onboarding_text("en", name).strip() and onboarding_text("zh", name).strip())

    def test_b2_inputs_subject_id_text_names_the_subfolder_and_the_suffix_rule(self):
        for lang in ("en", "zh"):
            text = plain(onboarding_text(lang, "help_subject_id"))
            self.assertIn("<Output Folder>/<Subject ID>" if lang == "en" else "<输出文件夹>/<受试者 ID>", text)
            self.assertIn("<ID>_2", text)
            self.assertIn("_99", text)

    def test_b2_inputs_batch_texts_name_the_real_rules(self):
        for lang in ("en", "zh"):
            folder = plain(onboarding_text(lang, "help_batch_folder"))
            for phrase in ("input/data.nii.gz", "input/bvals", "input/bvecs", "name.bval", "name.bvec", "bvals",
                           "NRRD", "NHDR", "DICOM", "mask", "Batch Folder/", "sub-01/"):
                self.assertIn(phrase, folder, (lang, phrase))
            output = plain(onboarding_text(lang, "help_output_folder"))
            for phrase in ("<ID>/outputs", "00_batch_metadata", "batch_outputs", "logs", "Output Folder/",
                           "stdout", "stderr"):
                self.assertIn(phrase, output, (lang, phrase))

    def test_b2_inputs_the_card_is_rich_text_with_a_fixed_width_so_a_tooltip_wraps_like_the_popup(self):
        for lang in ("en", "zh"):
            for key in KEYS:
                self.assertTrue(onboarding_text(lang, "help_" + key).startswith('<table width="360"'))


class HelpTruthTest(unittest.TestCase):
    """The texts describe what the code does. Each test runs the code the text describes."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()

    def test_b2_inputs_subject_id_text_matches_the_workspace_rule(self):
        from shared.ui_run_controller import resolve_single_subject_workspace_root as resolve

        out = self.root / "out"
        self.assertEqual(resolve(str(out), "S1"), out / "S1")  # a subfolder named after the Subject ID
        self.assertEqual(resolve(str(out / "S1"), "S1"), out / "S1")  # the Output Folder already ends with the ID
        self.assertEqual(resolve(str(out / "S1" / "outputs"), "S1"), out / "S1")

    def test_b2_inputs_subject_id_text_matches_the_suffix_rule(self):
        from shared.input_import import SUBJECT_SUFFIX_LIMIT, choose_subject_for_source

        out = self.root / "out"
        (out / "S1" / "input").mkdir(parents=True)
        (out / "S1" / "input" / "input_import_manifest.json").write_text(json.dumps({
            "status": "completed", "source_signatures": {"dwi": {"size": 1}}}), encoding="utf-8")

        def root_for_id(subject_id):
            return out / subject_id

        same = choose_subject_for_source("S1", root_for_id, {"dwi": {"size": 1}})
        self.assertEqual((same.subject_id, same.decision), ("S1", "same_input"))
        other = choose_subject_for_source("S1", root_for_id, {"dwi": {"size": 2}})
        self.assertEqual((other.subject_id, other.subject_root), ("S1_2", out / "S1_2"))
        self.assertEqual(SUBJECT_SUFFIX_LIMIT, 99)  # the text says _2 to _99

    def make_subject(self, parent, name, kind):
        folder = parent / name
        folder.mkdir(parents=True)
        if kind == "canonical":
            (folder / "input").mkdir()
            for item in ("data.nii.gz", "bvals", "bvecs"):
                (folder / "input" / item).write_bytes(b"x")
        elif kind == "nifti":
            for item in ("dwi.nii.gz", "dwi.bval", "dwi.bvec"):
                (folder / item).write_bytes(b"x")
        elif kind == "nifti_bvals":
            for item in ("dwi.nii.gz", "bvals", "bvecs"):
                (folder / item).write_bytes(b"x")
        elif kind == "nrrd":
            (folder / "dwi.nhdr").write_bytes(b"NRRD")
        elif kind == "dicom_deeper":
            (folder / "series1").mkdir()
            (folder / "series1" / "IM0001").write_bytes(b"\0" * 128 + b"DICM" + b"\0" * 64)
        elif kind == "two_nifti":
            for item in ("a.nii.gz", "a.bval", "a.bvec", "b.nii.gz", "b.bval", "b.bvec"):
                (folder / item).write_bytes(b"x")
        return folder

    def load_batch_script(self):
        import importlib.util

        script = REPO_ROOT / "scripts" / "run_batch_pipeline.py"
        spec = importlib.util.spec_from_file_location("_inputs_help_batch", script)
        module = importlib.util.module_from_spec(spec)
        sys.modules["_inputs_help_batch"] = module
        self.addCleanup(sys.modules.pop, "_inputs_help_batch", None)
        spec.loader.exec_module(module)
        return module

    def test_b2_inputs_batch_text_matches_the_scan_rules(self):
        batch = self.root / "batch"
        for name, kind in (("sub-01", "canonical"), ("sub-02", "nifti"), ("sub-03", "nifti_bvals"),
                           ("sub-04", "nrrd"), ("sub-05", "empty"), ("sub-06", "two_nifti")):
            self.make_subject(batch, name, kind)
        (batch / "notes.txt").write_text("not a subject", encoding="utf-8")
        deeper = batch / "group" / "sub-07"
        self.make_subject(batch / "group", "sub-07", "canonical")
        self.assertTrue(deeper.is_dir())
        module = self.load_batch_script()
        subjects, skipped = module.discover_subjects(batch)
        ids = {subject.subject_id for subject in subjects}
        # Each direct subfolder is one subject, named after the folder; deeper folders and files are not searched.
        self.assertEqual(ids, {"sub-01", "sub-02", "sub-03", "sub-04"})
        self.assertNotIn("sub-07", ids)
        self.assertNotIn("group/sub-07", ids)
        # A subfolder with no readable input is skipped; so is one with two NIfTI files (the text says "one NIfTI").
        self.assertEqual({item["subject_id"] for item in skipped} - {"group"}, {"sub-05", "sub-06"})

    def test_b2fix_dwi_W1_batch_text_states_only_what_the_scan_does(self):
        """W1: the text claimed deeper folders are not searched and an empty subfolder is skipped; neither is true."""
        batch = self.root / "batch"
        self.make_subject(batch, "sub-a", "dicom_deeper")  # the DICOM file is one folder deeper
        self.make_subject(batch, "sub-b", "empty")
        module = self.load_batch_script()
        subjects, skipped = module.discover_subjects(batch)
        by_id = {subject.subject_id: subject for subject in subjects}
        # A DICOM subject may hold its files in a deeper folder, so "deeper folders are not searched" is false.
        self.assertEqual(by_id["sub-a"].mode, "dicom")
        # An empty subfolder is not skipped here (the scan falls back to the Batch Folder), so "is skipped" is false.
        self.assertNotIn("sub-b", {item["subject_id"] for item in skipped})
        for lang, false_claims in (("en", ("Files and deeper folders are not searched",
                                           "A subfolder with no readable input is skipped")),
                                   ("zh", ("文件和更深层的文件夹不会被搜索", "没有可读取输入的子文件夹会被跳过"))):
            for key in ("help_batch_folder", "help_output_folder"):
                text = plain(onboarding_text(lang, key))
                for claim in false_claims:
                    self.assertNotIn(claim, text, (lang, key))
                self.assertNotIn("skipped" if lang == "en" else "跳过", text, (lang, key))
            folder = plain(onboarding_text(lang, "help_batch_folder"))
            self.assertIn("input/", folder)
            self.assertIn("DICOM", folder)
            output = plain(onboarding_text(lang, "help_output_folder"))
            self.assertNotIn("<ID>/outputs</tt> inside the Batch Folder", output)
        # The empty Output Folder text names the real default: an outputs folder next to the subject's input.
        self.assertIn("next to that subject's input", plain(onboarding_text("en", "help_output_folder")))
        self.assertIn("旁边", plain(onboarding_text("zh", "help_output_folder")).split("<ID>/outputs")[0])

    def test_b2_inputs_output_text_matches_the_batch_output_layout(self):
        module = self.load_batch_script()
        out = self.root / "out"
        subject = SimpleNamespace(subject_id="sub-01", subject_root=self.root / "batch" / "sub-01")
        command = module.build_subject_command(
            subject, pipeline_script=Path("p.py"), python_executable=Path("py"), output_root=out, run_name="r",
            execute=True, forwarded_args=[])
        self.assertEqual(command[command.index("--output-root") + 1], (out / "sub-01" / "outputs").as_posix())
        from shared.io_layout import SubjectIOLayout

        layout = SubjectIOLayout(self.root / "batch" / "sub-01", output_root=out / "sub-01" / "outputs")
        self.assertEqual((layout.workspace_root, layout.cache, layout.preproc),
                         (out / "sub-01", out / "sub-01" / "cache", out / "sub-01" / "preprocessed"))
        source = (REPO_ROOT / "scripts" / "run_batch_pipeline.py").read_text(encoding="utf-8")
        self.assertIn('/ "00_batch_metadata" / f"batch_{time.strftime(\'%Y%m%d_%H%M%S\')}.json"', source)
        self.assertIn('args.batch_root.expanduser().resolve() / "batch_outputs"', source)
        self.assertIn('manifest_path.expanduser().resolve().parent / "logs"', source)
        self.assertIn('f"{safe_id}.stdout.log"', source)
        self.assertIn('f"{safe_id}.stderr.log"', source)


class HelpButtonCase(unittest.TestCase):
    """The shared helper on a fake ``qt`` (real PythonQt behaviour is checked in the Slicer probe)."""

    def setUp(self):
        from test_ui_run_history import fake_qt

        self.qt = fake_qt()
        self.frames = []

        def frame(*args):
            widget = mock.MagicMock(name="QFrame")
            widget.visible = False
            widget.show.side_effect = lambda w=widget: setattr(w, "visible", True)
            widget.hide.side_effect = lambda w=widget: setattr(w, "visible", False)
            widget.isVisible.side_effect = lambda w=widget: w.visible
            widget.ctor_args = args
            self.frames.append(widget)
            return widget

        self.qt.QFrame.side_effect = frame
        self.labels = []

        def label(*args):
            widget = mock.MagicMock(name="QLabel")
            self.labels.append(widget)
            return widget

        self.qt.QLabel.side_effect = label
        self.owner = SimpleNamespace(_onboardingLanguage="en", _uiWidget=None)
        for patcher in (mock.patch.dict(sys.modules, {"qt": self.qt}),
                        mock.patch("shared.ui_styles.is_dark_mode", return_value=False)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def button(self, key="subject_id"):
        from shared import ui_help_button

        return ui_help_button.make_help_button(self.owner, self.qt, None, key)


class HelpButtonTest(HelpButtonCase):
    def test_b2_inputs_button_is_a_flat_focusable_round_tool_button_with_an_accessible_name(self):
        from shared import ui_help_button

        button = self.button()
        self.assertEqual(button.kind, "QToolButton")
        button.setText.assert_called_with("?")
        button.setFixedSize.assert_called_with(18, 18)
        button.setFocusPolicy.assert_called_with(self.qt.Qt.StrongFocus)  # reachable with the keyboard
        button.setAutoRaise.assert_called_with(True)
        button.setAccessibleName.assert_called_with(onboarding_text("en", "help_subject_id_name"))
        sheet = button.setStyleSheet.call_args[0][0]
        self.assertIn("border-radius: 9px", sheet)
        self.assertIn(":focus", sheet)
        self.assertIs(self.owner._helpButtons["subject_id"], button)
        self.assertEqual(ui_help_button.help_text(self.owner, "subject_id"), onboarding_text("en", "help_subject_id"))

    def test_b2_inputs_hover_text_and_popup_text_are_the_same_rich_text(self):
        for key in KEYS:
            button = self.button(key)
            tip = button.setToolTip.call_args[0][0]
            self.assertEqual(tip, onboarding_text("en", "help_" + key))
            button.clicked.connect.call_args[0][0]()
            popup_label = self.labels[-1]
            self.assertEqual(popup_label.setText.call_args[0][0], tip)
            popup_label.setWordWrap.assert_called_with(True)

    def test_b2_inputs_click_opens_a_qt_popup_frame_and_no_modal_dialog(self):
        button = self.button()
        click = button.clicked.connect.call_args[0][0]
        click()
        self.assertEqual(len(self.frames), 1)
        popup = self.frames[0]
        self.assertEqual(popup.ctor_args[1], self.qt.Qt.Popup)
        popup.setAttribute.assert_any_call(self.qt.Qt.WA_NoMouseReplay, True)  # the closing click is not replayed
        popup.show.assert_called_once()
        self.qt.QDialog.assert_not_called()
        self.qt.QMessageBox.assert_not_called()
        click()  # a keyboard click on the open popup's button closes it
        popup.hide.assert_called()
        self.assertFalse(popup.visible)
        click()
        self.assertEqual(len(self.frames), 1)  # one shared popup
        self.assertTrue(popup.visible)

    def test_b2_inputs_the_popup_text_and_names_follow_a_language_switch(self):
        from shared import ui_help_button

        button = self.button("batch_folder")
        button.clicked.connect.call_args[0][0]()
        label = self.labels[-1]
        self.assertEqual(label.setText.call_args[0][0], onboarding_text("en", "help_batch_folder"))
        self.owner._onboardingLanguage = "zh"
        ui_help_button.apply_help_language(self.owner)
        self.assertEqual(button.setToolTip.call_args[0][0], onboarding_text("zh", "help_batch_folder"))
        self.assertEqual(button.setAccessibleName.call_args[0][0], onboarding_text("zh", "help_batch_folder_name"))
        self.assertEqual(label.setText.call_args[0][0], onboarding_text("zh", "help_batch_folder"))  # open popup
        button.clicked.connect.call_args[0][0]()  # hides it; a new show reads the language again
        button.clicked.connect.call_args[0][0]()
        self.assertEqual(label.setText.call_args[0][0], onboarding_text("zh", "help_batch_folder"))

    def test_b2_inputs_the_look_follows_the_theme_for_buttons_and_popup(self):
        from shared import ui_help_button

        button = self.button()
        button.clicked.connect.call_args[0][0]()
        popup = self.frames[0]
        ui_help_button.refresh_help_styles(self.owner, dark=True)
        dark_button, dark_popup = button.setStyleSheet.call_args[0][0], popup.setStyleSheet.call_args[0][0]
        ui_help_button.refresh_help_styles(self.owner, dark=False)
        light_button, light_popup = button.setStyleSheet.call_args[0][0], popup.setStyleSheet.call_args[0][0]
        self.assertNotEqual(dark_button, light_button)
        self.assertNotEqual(dark_popup, light_popup)
        self.assertIn("#7D8590", dark_button)
        self.assertIn("#64748B", light_button)
        self.assertIn("#1C2128", dark_popup)
        self.assertIn("#FFFFFF", light_popup)

    def test_b2_inputs_cleanup_closes_and_drops_the_popup(self):
        from shared import ui_help_button

        button = self.button()
        button.clicked.connect.call_args[0][0]()
        popup = self.frames[0]
        ui_help_button.close_help_popup(self.owner)
        self.assertFalse(popup.visible)
        ui_help_button.close_help_popup(self.owner, destroy=True)
        popup.deleteLater.assert_called()
        self.assertIsNone(getattr(self.owner, "_helpPopup", None))
        ui_help_button.close_help_popup(SimpleNamespace())  # an owner that never made one: no error


class HelpPanelTest(remember.PanelCase):
    """b2 inputs-help: where the three buttons sit and when they show."""

    def setUp(self):
        super().setUp()
        self.owner._onboardingLanguage = "en"
        self.owner._uiWidget = None
        self.owner._refresh_input_preview_stepper = lambda: None

    def buttons(self):
        return self.owner._helpButtons

    def test_b2_inputs_panel_makes_the_three_buttons_with_their_texts(self):
        self.install()
        self.assertEqual(set(self.buttons()), set(KEYS))
        for key, button in self.buttons().items():
            self.assertEqual(button.setToolTip.call_args[0][0], onboarding_text("en", "help_" + key))
            self.assertEqual(button.setAccessibleName.call_args[0][0], onboarding_text("en", "help_" + key + "_name"))

    def test_b2_inputs_subject_id_and_batch_folder_buttons_sit_at_the_right_of_their_fields(self):
        fields = self.install()
        stack = self.qt.QStackedWidget.return_value
        pages = [call[0][0] for call in stack.addWidget.call_args_list]
        self.assertNotIn(self.owner.ui.subjectIdLineEdit, pages)  # wrapped, not added bare
        for field, key in ((self.owner.ui.subjectIdLineEdit, "subject_id"), (fields["batch_root"], "batch_folder")):
            rows = [w for w in self.qt.created if w.kind == "QHBoxLayout"
                    and [c[1][0] for c in w.method_calls if c[0] == "addWidget"][:2] == [field, self.buttons()[key]]]
            self.assertEqual(len(rows), 1, key)
        self.assertNotIn(self.buttons()["subject_id"], self.owner.ui.subjectIdLineEdit.ctor_args
                         if hasattr(self.owner.ui.subjectIdLineEdit, "ctor_args") else ())

    def test_b2_inputs_output_button_sits_at_the_far_right_of_the_output_row_and_shows_only_in_batch(self):
        self.install()
        output = self.owner.ui.outputDirectoryPath
        refresh = remember.by_name(self.qt, "inputsRefreshButton")
        out_help = self.buttons()["output_folder"]
        rows = [w for w in self.qt.created if w.kind == "QHBoxLayout"
                and [c[1][0] for c in w.method_calls if c[0] == "addWidget"] == [output, refresh, out_help]]
        self.assertEqual(len(rows), 1)
        for batch in (True, False, True):
            self.owner._parameterNode.batchMode = batch
            prep.sync_input_mode(self.owner)
            self.assertEqual(out_help.setVisible.call_args[0][0], batch)

    def test_b2_inputs_a_language_switch_reaches_all_three_buttons(self):
        self.install()
        self.owner._onboardingLanguage = "zh"
        prep.apply_input_language(self.owner)
        for key, button in self.buttons().items():
            self.assertEqual(button.setToolTip.call_args[0][0], onboarding_text("zh", "help_" + key))
            self.assertEqual(button.setAccessibleName.call_args[0][0], onboarding_text("zh", "help_" + key + "_name"))

    def test_b2_inputs_a_theme_change_restyles_the_three_buttons(self):
        self.install()
        for dark, ink in ((True, "#7D8590"), (False, "#64748B")):
            prep.refresh_prepare_inputs_styles(self.owner, dark=dark)
            for button in self.buttons().values():
                self.assertIn(ink, button.setStyleSheet.call_args[0][0])

    def test_b2_inputs_the_wrapper_rows_do_not_widen_the_panel(self):
        self.install()
        wrappers = [w for w in self.qt.created if w.kind == "QWidget" and w.setSizePolicy.called
                    and w.setMinimumWidth.called]
        policies = [w.setSizePolicy.call_args[0] for w in wrappers]
        self.assertGreaterEqual(len([p for p in policies if p[0] == self.qt.QSizePolicy.Ignored]), 3)
        for key, button in self.buttons().items():
            button.setFixedSize.assert_called_with(18, 18)


if __name__ == "__main__":
    unittest.main()
