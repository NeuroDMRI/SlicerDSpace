#!/usr/bin/env python3
"""Package D3.3: the GUI import uses the subject folder that the input identity chooses.

``shared.module_widget`` is imported against mocked Slicer/Qt modules; the import jobs run for real (thread job, or
the child-process CLI run in this process for the QProcess path) on temporary folders.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from unittest.mock import MagicMock

from shared.ui_background_process import build_background_run_context
from shared.user_config import load_user_config, write_user_config
from test_b2_fix_round import widget_methods
from test_subject_suffix_choice import dicom_folder, fake_dcm2niix, write_nifti

SUFFIX_EN = "The input differs from the data stored for subject 3167. Results go to subject 3167_2."
SUFFIX_ZH = "输入与受试者 3167 已保存的数据不同。结果写入受试者 3167_2。"


class FakeLineEdit:
    def __init__(self, text: str = "") -> None:
        self._text = text

    def text(self) -> str:
        return self._text

    def setText(self, text: str) -> None:
        self._text = text

    def __getattr__(self, name):
        return MagicMock()


class FakeProcess:
    """``qt.QProcess`` stand-in: records the start; the test runs the child CLI and emits ``finished``."""

    NotRunning = 0
    Running = 2
    MergedChannels = 1
    created: list = []

    def __init__(self, parent=None) -> None:
        self.signals: dict = {}
        self.args = None
        FakeProcess.created.append(self)

    def connect(self, name, callback) -> None:
        self.signals[name] = callback

    def disconnect(self, name, callback) -> None:
        self.signals.pop(name, None)

    def start(self, program, args) -> None:
        self.args = list(args)

    def state(self) -> int:
        return 0

    def processId(self) -> int:
        return 0

    def readAllStandardOutput(self):
        return b""

    def __getattr__(self, name):
        return MagicMock()

    def run_child(self) -> None:
        from shared.prepare_inputs_cli import main

        args = self.args
        request = args[args.index("--request") + 1]
        result = args[args.index("--result") + 1]
        with open(os.devnull, "w", encoding="utf-8") as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            code = main(["--request", request, "--result", result])
        self.signals["finished(int,QProcess::ExitStatus)"](code, 0)


class GuiCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name).resolve()
        self.out = self.tmp / "out"
        self.out.mkdir()
        self.timers: list = []
        mods = {name: MagicMock() for name in ("qt", "slicer", "vtk", "slicer.ScriptedLoadableModule", "ctk")}
        mods["qt"].QTimer.singleShot = lambda _ms, fn: self.timers.append(fn)
        mods["qt"].QProcess = FakeProcess
        self.errors: list = []
        mods["slicer"].util.NodeModify = lambda node: contextlib.nullcontext()
        mods["slicer"].util.errorDisplay = lambda text, **kw: self.errors.append(text)
        patcher = mock.patch.dict(sys.modules, mods)
        patcher.start()
        self.addCleanup(patcher.stop)
        sys.modules.pop("shared.module_widget", None)
        self.addCleanup(sys.modules.pop, "shared.module_widget", None)
        import shared.module_widget as mw

        self.mw = mw
        for name in ("set_run_state", "set_live_stage_text"):
            patched = mock.patch.object(mw, name, lambda *a, **k: None)
            patched.start()
            self.addCleanup(patched.stop)
        config = mock.patch("shared.user_config.default_user_config_path",
                            return_value=self.tmp / "cfg" / "user_config.json")
        config.start()
        self.addCleanup(config.stop)
        FakeProcess.created = []

    def widget(self, subject_id: str = "3167", *, qprocess: bool = False, lang: str = "en"):
        widget = self.mw.SharedWidgetMixin()
        widget.parent = None
        widget._pipelineActive = False
        widget._cleanedUp = False
        widget._onboardingLanguage = lang
        widget._parameterNode = SimpleNamespace(brainMaskBackend="pnl_cnn", inputDirectory="", subjectId=subject_id,
                                                outputDirectory=str(self.out), batchMode=False)
        widget.ui = SimpleNamespace(subjectIdLineEdit=FakeLineEdit(subject_id))
        self.tiles = {}
        widget._prepareInputWidgets = {"source_kind": "nifti", "status": MagicMock(), "panel": MagicMock()}
        widget._prepare_inputs_widget_text = lambda key: self.tiles.get(key, "")
        widget._environment_dependency_path_kwargs = lambda: {}
        widget._refresh_run_button_state = lambda *a: None
        widget._apply_shared_inputs_to_workflow_fields = lambda: None
        widget._checkCanApply = lambda: None
        widget._capture_background_process_group_id = lambda *a: None
        self.status: list = []
        self.logs: list = []
        widget._setStatus = lambda kind, text: self.status.append((kind, text))
        widget.appendLog = lambda text: self.logs.append(text)
        widget._slicer_launcher_program = (lambda: "/Applications/Slicer.app/Contents/bin/PythonSlicer") if qprocess \
            else (lambda: "")
        return widget

    def set_nifti(self, source: Path) -> None:
        self.tiles.update(source=str(source), bval=str(source.with_name("dwi.bval")),
                          bvec=str(source.with_name("dwi.bvec")))

    def drive(self, timeout: float = 60.0) -> None:
        end = time.monotonic() + timeout
        while self.timers:
            if time.monotonic() > end:
                self.fail("the import job did not finish")
            callback = self.timers.pop(0)
            callback()
            if self.timers:
                time.sleep(0.005)

    def import_now(self, widget, **kwargs) -> None:
        widget._on_prepare_inputs_clicked(**kwargs)
        # vars(): FakeProcess.__getattr__ answers any missing name with a (truthy) MagicMock.
        if FakeProcess.created and FakeProcess.created[-1].args and not vars(FakeProcess.created[-1]).get("ran"):
            process = FakeProcess.created[-1]
            process.ran = True
            process.run_child()
        self.drive()
        self.assertFalse(widget._inputImportPending)

    def suffix_lines(self, text: str) -> list:
        return [line for line in self.logs if text in line]


class ThreadPathTest(GuiCase):
    def test_d3_3_the_job_writes_to_the_chosen_target_and_the_handler_shows_the_effective_id(self) -> None:
        widget = self.widget()
        a = write_nifti(self.tmp / "a")
        b = write_nifti(self.tmp / "b", fill=2.0)
        self.set_nifti(a)
        self.import_now(widget)
        node = widget._parameterNode
        self.assertEqual((node.inputDirectory, node.subjectId), ((self.out / "3167").as_posix(), "3167"))
        self.assertEqual(self.suffix_lines("Results go to subject"), [])
        self.assertFalse(any(text == SUFFIX_EN for _kind, text in self.status))

        self.set_nifti(b)
        self.import_now(widget)
        self.assertTrue((self.out / "3167_2" / "input" / "data.nii.gz").is_file())
        self.assertEqual((node.inputDirectory, node.subjectId), ((self.out / "3167_2").as_posix(), "3167_2"))
        self.assertEqual(widget.ui.subjectIdLineEdit.text(), "3167_2")
        self.assertEqual(len(self.suffix_lines(SUFFIX_EN)), 1, self.logs)
        self.assertEqual([text for _kind, text in self.status if text == SUFFIX_EN], [SUFFIX_EN])
        self.assertEqual(widget._preparedInputSelection, widget._input_import_selection())
        self.assertEqual(widget._subjectIdSuffixState,
                         {"base": "3167", "effective": "3167_2", "output": str(self.out)})

    def test_d3_3_the_original_input_goes_back_to_the_base_id_and_a_user_edit_starts_a_new_base(self) -> None:
        widget = self.widget()
        a = write_nifti(self.tmp / "a")
        b = write_nifti(self.tmp / "b", fill=2.0)
        self.set_nifti(a)
        self.import_now(widget)
        self.set_nifti(b)
        self.import_now(widget)
        self.assertEqual(widget.ui.subjectIdLineEdit.text(), "3167_2")
        self.set_nifti(a)  # the field still shows 3167_2
        self.logs.clear()
        self.import_now(widget)
        node = widget._parameterNode
        self.assertEqual((node.subjectId, node.inputDirectory), ("3167", (self.out / "3167").as_posix()))
        self.assertFalse((self.out / "3167_2_2").exists())
        self.assertEqual(self.suffix_lines("Results go to subject"), [])
        widget.ui.subjectIdLineEdit.setText("ABC")
        self.import_now(widget)
        self.assertEqual((node.subjectId, node.inputDirectory), ("ABC", (self.out / "ABC").as_posix()))
        self.assertEqual(widget._subjectIdSuffixState["base"], "ABC")

    def test_d3_3_a_changed_output_folder_drops_the_state(self) -> None:
        widget = self.widget()
        self.set_nifti(write_nifti(self.tmp / "a"))
        self.import_now(widget)
        self.set_nifti(write_nifti(self.tmp / "b", fill=2.0))
        self.import_now(widget)
        other = self.tmp / "other"
        widget._parameterNode.outputDirectory = str(other)
        self.import_now(widget)
        self.assertEqual(widget._parameterNode.inputDirectory, (other / "3167_2").as_posix())
        self.assertEqual(widget._subjectIdSuffixState["base"], "3167_2")

    def test_d3_3_the_suffix_message_follows_the_language(self) -> None:
        widget = self.widget(lang="zh")
        self.set_nifti(write_nifti(self.tmp / "a"))
        self.import_now(widget)
        self.set_nifti(write_nifti(self.tmp / "b", fill=2.0))
        self.import_now(widget)
        self.assertEqual(len(self.suffix_lines(SUFFIX_ZH)), 1, self.logs)
        self.assertIn(("warning", SUFFIX_ZH), self.status)


class BatchModeTest(GuiCase):
    def test_d3_3_batch_mode_sends_no_subject_choice_and_keeps_no_state(self) -> None:
        widget = self.widget(qprocess=True)
        widget._subjectIdSuffixState = {"base": "3167", "effective": "3167", "output": str(self.out)}
        widget._parameterNode.batchMode = True
        self.set_nifti(write_nifti(self.tmp / "a"))
        widget._on_prepare_inputs_clicked()
        process = FakeProcess.created[-1]
        request = json.loads(Path(process.args[process.args.index("--request") + 1]).read_text())
        self.assertNotIn("subject_choice", request)
        self.assertIsNone(widget._subjectIdSuffixState)


class QProcessPathTest(GuiCase):
    def test_d3_3_the_qprocess_path_and_the_thread_path_choose_the_same_target(self) -> None:
        a = write_nifti(self.tmp / "a")
        b = write_nifti(self.tmp / "b", fill=2.0)
        threaded = self.widget()
        self.set_nifti(a)
        self.import_now(threaded)
        self.set_nifti(b)
        self.import_now(threaded)
        self.assertEqual(threaded._parameterNode.subjectId, "3167_2")

        process_widget = self.widget(qprocess=True)  # a new session: no state, the field shows 3167
        self.set_nifti(b)
        self.import_now(process_widget)
        self.assertEqual(len(FakeProcess.created), 1)
        node = process_widget._parameterNode
        self.assertEqual((node.subjectId, node.inputDirectory), ("3167_2", (self.out / "3167_2").as_posix()))
        self.assertEqual(process_widget.ui.subjectIdLineEdit.text(), "3167_2")
        self.assertEqual(len(self.suffix_lines(SUFFIX_EN)), 1, self.logs)  # widget() starts a new log

    def test_d3_3_the_child_request_carries_the_base_id_and_the_root_rule(self) -> None:
        widget = self.widget(qprocess=True)
        self.set_nifti(write_nifti(self.tmp / "a"))
        widget._on_prepare_inputs_clicked()
        process = FakeProcess.created[-1]
        request = json.loads(Path(process.args[process.args.index("--request") + 1]).read_text())
        self.assertEqual(request["subject_choice"], {
            "base_id": "3167", "output_dir": str(self.out),
            "fallback_parent": str(self.tmp / "cfg" / "subjects")})
        process.run_child()
        self.drive()
        self.assertEqual(widget._parameterNode.inputDirectory, (self.out / "3167").as_posix())


class DicomPathTest(GuiCase):
    def test_d3_3_no_dicom_inventory_runs_on_the_gui_thread_and_the_series_is_reused(self) -> None:
        widget = self.widget()
        widget._prepareInputWidgets["source_kind"] = "dicom"
        folder = dicom_folder(self.tmp)
        self.tiles.update(source=str(folder))
        series = ({"uid": "1.2.3", "description": "dwi", "number": "1", "acquisition_time": "", "frames": 3,
                   "paths": sorted(folder.glob("*.dcm"))},)
        import shared.dicom_input as dicom_input
        import shared.input_import as input_import

        calls: list = []
        real_signature = input_import.dicom_source_signature
        real_inventory = dicom_input._inventory

        def signature(*args, **kwargs):
            calls.append(("signature", threading.current_thread() is threading.main_thread()))
            return real_signature(*args, **kwargs)

        def inventory(*args, **kwargs):
            calls.append(("inventory", threading.current_thread() is threading.main_thread()))
            return real_inventory(*args, **kwargs)

        with mock.patch("shared.dicom_input.scan_series", return_value=series), \
                mock.patch("shared.dicom_input.converter_path", return_value=("dcm2niix", Path("/converter"))), \
                mock.patch("shared.input_import._run", side_effect=fake_dcm2niix) as run, \
                mock.patch.object(input_import, "dicom_source_signature", signature), \
                mock.patch.object(dicom_input, "_inventory", inventory):
            widget._prepare_inputs_config()
            widget._on_prepare_inputs_clicked()
            self.assertEqual(calls, [])  # nothing listed by the click handler itself
            self.drive()
            self.assertTrue(calls)
            self.assertFalse(any(on_main for _name, on_main in calls), calls)
            self.assertEqual(widget._parameterNode.inputDirectory, (self.out / "3167").as_posix())
            self.import_now(widget)
            self.assertEqual(run.call_count, 1)  # the unchanged series is reused, not converted again
            self.assertEqual(widget._parameterNode.subjectId, "3167")


ENVIRONMENT_FILES = (
    "check-20261121T101500-request.json",
    "check-20261121T101500-result.json",
    "dependency_profile.json",
    "environment_doctor.json",
    "install_external_tools.md",
    "install_missing_python.ps1",
    "install_missing_python.sh",
    "setup-20261121T101500.log",
)


class FixRoundTest(GuiCase):
    """Round-1 findings FE1 (the environment check scaffold) and FE4 (the suffix notice stays visible)."""

    def write_environment_check(self, widget, subject_id: str) -> Path:
        """The files "Check environment dependencies" writes, at the folder the GUI itself computes."""
        from shared.ui_run_controller import resolve_single_subject_output_root

        output_root = resolve_single_subject_output_root(str(self.out), subject_id)
        # As _environment_check_kwargs gives them: no subject root yet, the resolved single-subject output root.
        folder = widget._environment_report_dir_from_kwargs({"subject_root": None, "output_root": Path(output_root)})
        self.assertIsNotNone(folder)
        folder.mkdir(parents=True)
        for name in ENVIRONMENT_FILES:
            (folder / name).write_text("{}\n" if name.endswith(".json") else "x\n", encoding="utf-8")
        return folder

    def test_d3_2_fe1_a_first_import_after_the_environment_check_keeps_the_base_id(self) -> None:
        widget = self.widget("P1")
        folder = self.write_environment_check(widget, "P1")
        self.assertEqual(sorted(p.relative_to(self.out).as_posix() for p in (self.out / "P1").rglob("*")
                                if p.is_file()),
                         sorted(f"P1/outputs/00_run_metadata/environment/{name}" for name in ENVIRONMENT_FILES))
        self.set_nifti(write_nifti(self.tmp / "a"))
        self.import_now(widget)
        node = widget._parameterNode
        self.assertEqual((node.subjectId, node.inputDirectory), ("P1", (self.out / "P1").as_posix()))
        self.assertFalse((self.out / "P1_2").exists())
        self.assertTrue((self.out / "P1" / "input" / "data.nii.gz").is_file())
        self.assertEqual(sorted(p.name for p in folder.iterdir()), sorted(ENVIRONMENT_FILES))
        self.assertEqual(self.suffix_lines("Results go to subject"), [])

    def test_d3_2_fe1_a_run_manifest_beside_the_environment_folder_still_counts_as_data(self) -> None:
        widget = self.widget("P1")
        folder = self.write_environment_check(widget, "P1")
        (folder.parent / "Test1_orchestration_manifest.json").write_text("{}\n", encoding="utf-8")
        self.set_nifti(write_nifti(self.tmp / "a"))
        self.import_now(widget)
        node = widget._parameterNode
        self.assertEqual((node.subjectId, node.inputDirectory), ("P1_2", (self.out / "P1_2").as_posix()))
        self.assertFalse((self.out / "P1" / "input").exists())

    def test_d3_3_fe4_the_suffix_notice_is_the_last_status_after_the_input_check(self) -> None:
        widget = self.widget()
        check_text = "Input layout found. Dependencies are checked when you run."
        # As the real _checkCanApply does (validateInputs(manual=False) sets the status).
        widget._checkCanApply = lambda: widget._setStatus("info", check_text)
        self.set_nifti(write_nifti(self.tmp / "a"))
        self.import_now(widget)
        self.assertEqual(self.status[-1], ("info", check_text))  # no suffix: no notice
        self.assertFalse(any(text == SUFFIX_EN for _kind, text in self.status))
        self.set_nifti(write_nifti(self.tmp / "b", fill=2.0))
        self.import_now(widget)
        self.assertEqual(self.status[-1], ("warning", SUFFIX_EN))
        self.assertEqual([text for _kind, text in self.status if text == SUFFIX_EN], [SUFFIX_EN])
        self.assertEqual(len(self.suffix_lines(SUFFIX_EN)), 1, self.logs)


class RunAfterImportTest(GuiCase):
    def test_d3_3_the_run_after_a_suffixed_import_carries_the_suffixed_root_and_id(self) -> None:
        from shared.topup_parameters import TopupParameters

        from test_ui_algorithm_availability import make_node

        widget = self.widget()
        node = make_node(None, "dti", subjectId="3167", outputDirectory=str(self.out), brainMaskBackend="pnl_cnn",
                         device="auto", ddsegDevice="auto", autoInstallPythonDependencies=False,
                         orchestrationRunName="Test1", orchestrationSelectStreamlines=1000, connectomeBackend="auto",
                         connectomeCompressMrtrixDenseState="unset", brainMaskDevice="auto", pnlMaskRoute="auto",
                         maxParallelModels=1)
        widget._parameterNode = node
        self.set_nifti(write_nifti(self.tmp / "a"))
        self.import_now(widget)
        self.assertEqual(node.subjectId, "3167")
        started: list = []
        widget.logic = SimpleNamespace(planSelectedProductOrchestration=lambda **kw: SimpleNamespace(
            run_plan_path=Path("/x/plan.json"), run_name="Test1", items=[], warnings=[]))
        stubs = dict(
            validateInputs=lambda manual=False: True,
            _input_preview_pending_source=lambda: None,
            _save_common_user_settings=lambda: None,
            _topup_validation=lambda batch_mode=False: SimpleNamespace(errors=[], warnings=[],
                                                                       parameters=TopupParameters()),
            _environment_check_kwargs=lambda **kw: {},
            _environment_check_signature=lambda kw: "sig",
            _cached_environment_result=lambda sig: object(),
            _environment_doctor_result_for_run=lambda kw, sig: SimpleNamespace(statuses=[]),
            _update_run_plan_table=lambda plan: None,
            _summarise_missing=lambda items: "missing " + ",".join(items),
            _startBackgroundPipeline=lambda kwargs, paths, selected: started.append((kwargs, selected)),
            _ifw_echo_time_ms=lambda: None,
            _data_filtered_run=lambda batch_mode: None,
            _selected_workflows=lambda: ["Microstructure"],
        )
        for name, fn in stubs.items():
            setattr(widget, name, fn)
        widget._setupRestartRequired = False
        widget._setupFunctionalFailure = False
        widget._selectedSetupActive = False
        self.set_nifti(write_nifti(self.tmp / "b", fill=2.0))
        mw = self.mw
        with mock.patch.object(mw, "preflight_blockers_for_targets", return_value={}), \
                mock.patch("shared.setup_resume.pending_setup", return_value=None), \
                mock.patch("shared.ui_input_inspection.inspect_input_directory", return_value={"mode": "subject_root"}), \
                mock.patch("shared.ui_compact.sync_detected_input_format"):
            widget._onRunSelectedClicked()  # the run gate imports first, then runs
            self.drive()
        self.assertEqual(len(started), 1, (self.status, self.errors))
        kwargs, selected = started[0]
        self.assertEqual(Path(kwargs["subject_root"]), self.out / "3167_2")
        self.assertEqual(kwargs["subject_id"], "3167_2")
        context = build_background_run_context(kwargs, selected)
        self.assertEqual((Path(context["subject_root"]), context["subject_id"]), (self.out / "3167_2", "3167_2"))
        self.assertEqual(node.inputDirectory, (self.out / "3167_2").as_posix())


# ----- Persistence of the base ID ------------------------------------------------------------------------------


class PersistenceTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def save(self, state) -> dict:
        calls: list = []
        Save = widget_methods(
            "_save_common_user_settings",
            slicer=SimpleNamespace(app=SimpleNamespace(userSettings=mock.Mock())),
            ENVIRONMENT_DEPENDENCY_PATHS=(),
            update_user_config=lambda payload: (calls.append(payload) or ({}, Path("/none/config.json"))),
        )
        owner = Save()
        owner._parameterNode = SimpleNamespace(
            orchestrationMicrostructureStages="dti", subjectId="3167_2", inputDirectory="/o/3167_2",
            outputDirectory="/o", orchestrationRunName="r", ddsegDevice="cpu", device="cpu",
            orchestrationTractographyBackend="csd_act", orchestrationBundleBackend="wmql",
            orchestrationSelectStreamlines=1000, autoInstallPythonDependencies=False, connectomeBackend="python")
        owner._common_user_setting_specs = lambda: ()
        owner._requested_product_targets = lambda: ()
        if state is not None:
            owner._subjectIdSuffixState = state
        Save._save_common_user_settings(owner)
        return calls[0]["recent"]

    def test_d3_3_save_writes_the_base_id_or_an_empty_text(self) -> None:
        recent = self.save({"base": "3167", "effective": "3167_2", "output": "/o"})
        self.assertEqual((recent["subject_id_base"], recent["subject_id"]), ("3167", "3167_2"))
        self.assertEqual(self.save(None)["subject_id_base"], "")

    def test_d3_3_save_writes_no_base_after_a_hand_edit_of_the_id_or_the_output(self) -> None:
        # The node shows 3167_2 in /o; a state for another ID or another Output Folder is stale.
        self.assertEqual(self.save({"base": "3167", "effective": "3167_5", "output": "/o"})["subject_id_base"], "")
        self.assertEqual(self.save({"base": "3167", "effective": "3167_2", "output": "/x"})["subject_id_base"], "")

    def restore(self, recent: dict, node_extra=None):
        config_path = self.root / "user_config.json"
        write_user_config({"recent": recent}, config_path)
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
            consume_user_defaults_request=lambda node: True)
        owner = Restore()
        owner.logs = []
        owner.appendLog = owner.logs.append
        owner._parameterNode = SimpleNamespace(
            subjectId="", inputDirectory="", outputDirectory="", orchestrationRunName="", brainMaskBackend="",
            orchestrationTractographyBackend="csd_act", orchestrationBundleBackend="wmql", connectomeBackend="auto",
            orchestrationSelectStreamlines=1_000_000, autoInstallPythonDependencies=True,
            preprocessingReversePhaseB0="", preprocessingPhaseEncodingDirection="", preprocessingReadoutTime=0.0,
            orchestrationMicrostructureStages="dti", **(node_extra or {}))
        owner._environment_settings_key = lambda dependency_id: dependency_id
        owner._user_settings_key = lambda name: name
        owner._common_user_setting_specs = lambda: ()
        owner._restore_common_user_settings_to_node()
        return owner

    def test_d3_3_restore_rebuilds_the_state_for_the_base_or_a_suffixed_id(self) -> None:
        for subject_id in ("3167", "3167_2", "3167_12"):
            with self.subTest(subject_id=subject_id):
                owner = self.restore({"subject_id": subject_id, "subject_id_base": "3167", "output_root": "/o",
                                      "subject_root": f"/o/{subject_id}"})
                self.assertEqual(owner._subjectIdSuffixState,
                                 {"base": "3167", "effective": subject_id, "output": "/o"})

    def test_d3_3_restore_builds_no_state_for_another_id_or_an_old_config(self) -> None:
        for recent in ({"subject_id": "ABC", "subject_id_base": "3167", "output_root": "/o"},
                       {"subject_id": "3167_x", "subject_id_base": "3167", "output_root": "/o"},
                       {"subject_id": "3167_2", "subject_id_base": "", "output_root": "/o"},
                       {"subject_id": "3167_2", "output_root": "/o", "subject_root": "/o/3167_2"}):
            with self.subTest(recent=recent):
                owner = self.restore(recent)
                self.assertIsNone(getattr(owner, "_subjectIdSuffixState", None))
                self.assertEqual(owner._parameterNode.subjectId, recent["subject_id"])  # an old config loads as today
                self.assertEqual(owner.logs, [])

    def test_d3_3_user_config_keeps_the_new_key_and_defaults_it_to_empty(self) -> None:
        from shared.user_config import empty_user_config, normalize_user_config

        self.assertEqual(empty_user_config()["recent"]["subject_id_base"], "")
        self.assertEqual(normalize_user_config({"recent": {"subject_id_base": None}})["recent"]["subject_id_base"], "")
        self.assertEqual(normalize_user_config({})["recent"]["subject_id_base"], "")


if __name__ == "__main__":
    unittest.main()
