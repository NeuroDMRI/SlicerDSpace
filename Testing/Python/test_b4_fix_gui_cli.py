#!/usr/bin/env python3
"""Batch 4 fix round, group gui-cli: U3, P-a, P-b, P-c, P-d, P-e.

Each test is named after its finding. The widget methods are loaded from ``shared/module_widget.py`` by AST, as the
other widget tests of this repo do, so no Slicer GUI is needed.
"""

from __future__ import annotations

import ast
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODULE_WIDGET_PATH = REPO_ROOT / "shared" / "module_widget.py"
BATCH_CLI_PATH = REPO_ROOT / "scripts" / "run_batch_pipeline.py"

from shared import run_heartbeat as hb  # noqa: E402
from shared.subject_id import effective_subject_id  # noqa: E402
from shared.ui_run_state import gui_brain_mask_required  # noqa: E402


def _load_batch_cli():
    spec = importlib.util.spec_from_file_location("run_batch_pipeline_b4_fix", BATCH_CLI_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _widget_functions(names, **namespace):
    tree = ast.parse(MODULE_WIDGET_PATH.read_text(encoding="utf-8"))
    found = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names}
    missing = set(names) - set(found)
    assert not missing, f"not in module_widget.py: {missing}"
    namespace.update(__file__=str(MODULE_WIDGET_PATH), Any=__import__("typing").Any, Optional=__import__("typing").Optional, Path=Path, os=os, sys=sys,
                     effective_subject_id=effective_subject_id, gui_brain_mask_required=gui_brain_mask_required)
    exec(compile(ast.Module(body=[found[name] for name in names], type_ignores=[]), str(MODULE_WIDGET_PATH), "exec"),
         namespace)
    return {name: namespace[name] for name in names}


# ------------------------------------------------------------------------------------------------------------- U3
def _make_subject(root: Path, name: str, *, mask: bool) -> Path:
    folder = root / name / "input"
    folder.mkdir(parents=True)
    for filename in ("data.nii.gz", "bvals", "bvecs"):
        (folder / filename).touch()
    if mask:
        (folder / "nodif_brain_mask.nii.gz").touch()
    return folder


class U3BatchDoctorMaskTests(unittest.TestCase):
    def _owner(self, batch_root: Path, *, input_directory: str = "", targets=("microstructure",), **node_values):
        functions = _widget_functions(("_environment_check_kwargs", "_batch_mask_state"),
                                      inspect_input_directory=lambda *args, **kwargs: {})
        node = SimpleNamespace(subjectId="", inputDirectory=input_directory, outputDirectory="",
                               brainMaskBackend="pnl_cnn", targetPreprocessing=False, connectomeBackend="python",
                               ddsegDevice="cpu", device="cpu", batchMode=True, batchRoot=str(batch_root),
                               **node_values)

        class Owner:
            _environment_check_kwargs = functions["_environment_check_kwargs"]
            _batch_mask_state = functions["_batch_mask_state"]

            def __init__(self):
                self._parameterNode = node
                self._selected_environment_targets = lambda: targets
                self._closure_tractography_backend_text = lambda: "dti_det"
                self._microstructure_stage_flags = lambda: (False, False)
                self._microstructure_dki_enabled = lambda: False
                self._effective_output_root_text = lambda text="": text

        return Owner(), node

    def test_u3_batch_with_masked_subjects_and_an_empty_subject_field_needs_no_mask_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            batch = Path(tmp)
            _make_subject(batch, "sub-01", mask=True)
            _make_subject(batch, "sub-02", mask=True)
            owner, _node = self._owner(batch)
            self.assertEqual(owner._batch_mask_state(str(batch)), (2, True))  # both subjects are really discovered
            kwargs = owner._environment_check_kwargs(external_paths={})
            self.assertFalse(kwargs["brain_mask_required"])
            self.assertIsNone(kwargs["subject_root"])

    def test_u3_batch_needs_the_mask_runtime_when_one_subject_lacks_a_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            batch = Path(tmp)
            _make_subject(batch, "sub-01", mask=True)
            _make_subject(batch, "sub-02", mask=False)
            owner, _node = self._owner(batch)
            self.assertTrue(owner._environment_check_kwargs(external_paths={})["brain_mask_required"])

    def test_u3_batch_with_no_discoverable_subject_leaves_the_mask_check_to_each_subject(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            owner, _node = self._owner(Path(tmp))
            self.assertFalse(owner._environment_check_kwargs(external_paths={})["brain_mask_required"])
            owner, _node = self._owner(Path(tmp) / "missing")
            self.assertFalse(owner._environment_check_kwargs(external_paths={})["brain_mask_required"])

    def test_u3_batch_still_requires_the_mask_when_preprocessing_regenerates_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            batch = Path(tmp)
            _make_subject(batch, "sub-01", mask=True)
            owner, node = self._owner(batch)
            node.targetPreprocessing = True
            self.assertTrue(owner._environment_check_kwargs(external_paths={})["brain_mask_required"])

    def test_u3_the_batch_kwargs_ignore_the_single_subject_field_so_the_check_signature_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            batch = Path(tmp) / "batch"
            _make_subject(batch, "sub-01", mask=True)
            owner, _node = self._owner(batch, input_directory=str(Path(tmp) / "stale_single_subject"))
            from_start = owner._environment_check_kwargs(external_paths={})
            from_run = owner._environment_check_kwargs(
                subject_root_text=str(Path(tmp) / "stale_single_subject"), output_root_text="", external_paths={})
            self.assertEqual(from_start, from_run)
            self.assertIsNone(from_run["subject_root"])

    def test_u3_batch_discovery_arranges_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            batch = Path(tmp)
            raw = batch / "sub-raw"
            raw.mkdir()
            for filename in ("data.nii.gz", "bvals", "bvecs"):
                (raw / filename).touch()
            owner, _node = self._owner(batch)
            owner._environment_check_kwargs(external_paths={})
            self.assertFalse((raw / "input").exists())

    def test_u3_run_environment_doctor_accepts_no_subject_root(self) -> None:
        source = (REPO_ROOT / "shared" / "module_logic.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        method = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.FunctionDef) and node.name == "runEnvironmentDoctor")
        calls = []
        namespace = {
            "Optional": __import__("typing").Optional, "Path": Path,
            "EnvironmentDoctorResult": object,
            "subject_layout": lambda *a, **k: self.fail("a layout needs a subject root"),
            "slicer": SimpleNamespace(app=SimpleNamespace(temporaryPath="/tmp/slicer"), modules=SimpleNamespace()),
            "EnvironmentDoctorConfig": lambda **kwargs: SimpleNamespace(**kwargs),
            "run_environment_doctor": lambda config: calls.append(config) or SimpleNamespace(),
            "write_environment_report": lambda result, folder: calls.append(folder) or {
                "json": "j", "python_sh": "p", "external_notes": "n"},
            "summarize_environment_doctor": lambda result: "summary",
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), "module_logic.py", "exec"), namespace)
        owner = SimpleNamespace(module_dir=Path("/repo"), log=lambda text: None)
        namespace["runEnvironmentDoctor"](owner, targets=("validation",), subject_root=None,
                                          output_root=Path("/tmp/out"))
        self.assertEqual(calls[0].output_dir, Path("/tmp/out").resolve() / "00_run_metadata" / "environment")


# ------------------------------------------------------------------------------------------------------------- P-a
class _Clock:
    now = 0.0


class PaHeartbeatOwnLogTests(unittest.TestCase):
    def _run_ticks(self, tmp: Path, write_own_line) -> list[str]:
        lines: list[str] = []

        def out(line: str) -> None:
            lines.append(line)
            write_own_line(line)

        _Clock.now = 0.0
        beat = hb.Heartbeat(interval=60, stall_minutes=10, out=out, log_root=tmp, clock=lambda: _Clock.now)
        beat._cpu.sample = lambda: (5.0, 1)  # frozen CPU
        beat.set_stage("UKF")
        for _ in range(12):
            _Clock.now += 60
            beat.tick()
        return lines

    def test_p_a_growth_of_the_heartbeats_own_stdout_file_is_not_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "sub-01.stdout.log"
            with log.open("w", encoding="utf-8") as stream, patch.object(sys, "stdout", stream):
                def write_own_line(line: str) -> None:
                    stream.write(line + "\n")
                    stream.flush()

                lines = self._run_ticks(Path(tmp), write_own_line)
            self.assertTrue([line for line in lines if line.startswith("[Stall warning]")], lines[-3:])
            self.assertFalse([line for line in lines if "log_bytes=" in line], lines[-3:])

    def test_p_a_a_file_named_in_the_exclude_variable_is_not_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "sub-02.stdout.log"
            log.write_text("start\n", encoding="utf-8")

            def write_own_line(line: str) -> None:
                with log.open("a", encoding="utf-8") as stream:
                    stream.write(line + "\n")

            with patch.dict(os.environ, {hb.EXCLUDE_ENV: str(log)}), patch.object(sys, "stdout", io.StringIO()):
                lines = self._run_ticks(Path(tmp), write_own_line)
            self.assertTrue([line for line in lines if line.startswith("[Stall warning]")], lines[-3:])

    def test_p_a_another_stage_log_still_counts_as_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            own = Path(tmp) / "sub-03.stdout.log"
            tool = Path(tmp) / "ukf_1.log"
            own.write_text("", encoding="utf-8")
            tool.write_text("x", encoding="utf-8")

            def write_own_line(line: str) -> None:
                with own.open("a", encoding="utf-8") as stream:
                    stream.write(line + "\n")
                with tool.open("a", encoding="utf-8") as stream:
                    stream.write("y")
                # FLAKY-P-A: the ticks run in a few ms, and Linux stamps the append with a coarse clock that can be
                # behind the stage start (time.time()); stamp it clearly after the start, as a real tool's later
                # write is.
                stamp = time.time() + 5.0
                os.utime(tool, (stamp, stamp))

            with patch.dict(os.environ, {hb.EXCLUDE_ENV: str(own)}), patch.object(sys, "stdout", io.StringIO()):
                lines = self._run_ticks(Path(tmp), write_own_line)
            self.assertFalse([line for line in lines if line.startswith("[Stall warning]")], lines[-3:])
            self.assertTrue(any("log_bytes=" in line for line in lines))

    def test_flaky_p_a_file_clock_behind_the_wall_clock(self) -> None:
        """FLAKY-P-A: Linux stamps a write with a coarse clock that can be a few ms behind time.time(). The ticks run
        in a few ms, so each append to the tool log gets an mtime before the stage start and is not progress. This
        runs the P-a test with the stage-start clock 1 s ahead of the file clock (larger than any run time of the
        ticks, so a slow runner cannot hide a removed fix; backlog P2, 27 November 2026)."""
        real_time = time.time
        ahead = SimpleNamespace(time=lambda: real_time() + 1.0, monotonic=time.monotonic)
        with patch.object(hb, "time", ahead):
            self.test_p_a_another_stage_log_still_counts_as_progress()

    def test_p_a_the_batch_script_names_the_subject_stdout_log_to_the_child(self) -> None:
        batch_cli = _load_batch_cli()
        captured = {}

        def fake_run(command, **options):
            captured.update(options)
            return subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryDirectory() as tmp:
            subject = batch_cli.BatchSubject("sub-01", Path(tmp) / "sub-01", "glob", "subject_root")
            log_dir = Path(tmp) / "logs"
            with patch.object(batch_cli.subprocess, "run", fake_run):
                result = batch_cli._run_subject(["python", "x.py"], subject, log_dir, relay=None)
            self.assertIn(str(result.stdout_log), captured["env"][hb.EXCLUDE_ENV].split(os.pathsep))
            self.assertEqual(captured["env"]["PATH"], os.environ["PATH"])


# ------------------------------------------------------------------------------------------------------------- P-b
class PbNoTorchOnImportTests(unittest.TestCase):
    def test_p_b_cli_parser_import_does_not_put_torch_in_the_process(self) -> None:
        probe = textwrap.dedent("""
            import json, sys
            sys.path.insert(0, %r)
            import shared.run_subject_pipeline_cli
            from shared.experiments import cli_parser
            cli_parser()
            print(json.dumps({"torch": "torch" in sys.modules, "dipy": "dipy" in sys.modules,
                              "sklearn": "sklearn" in sys.modules}))
        """ % str(REPO_ROOT))
        completed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=240,
                                   env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), cwd=tempfile.gettempdir())
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertFalse(result["torch"])
        self.assertFalse(result["dipy"] or result["sklearn"], "the preload must still come before any of these")

    def test_p_b_main_runs_the_preload_before_anything_else(self) -> None:
        # Reuse the loaded module: popping it from sys.modules would make later patch() calls in other test modules
        # target a new module object while they still hold the old main() (test_cli_bundle_source).
        import shared.run_subject_pipeline_cli as cli

        order = []
        with patch.object(cli, "_preload_torch_for_openmp", lambda: order.append("preload") or True), \
                patch.object(cli, "build_parser", lambda: order.append("parser") or (_ for _ in ()).throw(SystemExit(0))):
            with self.assertRaises(SystemExit):
                cli.main([])
        self.assertEqual(order, ["preload", "parser"])


# ------------------------------------------------------------------------------------------------------------- P-c
class PcSingleWriteTests(unittest.TestCase):
    class _Recorder:
        def __init__(self) -> None:
            self.writes: list[str] = []

        def write(self, text) -> int:
            self.writes.append(text)
            return len(text)

        def flush(self) -> None:
            pass

    def test_p_c_a_log_line_is_one_write_call_on_stdout(self) -> None:
        import shared.run_subject_pipeline_cli as cli

        recorder = self._Recorder()
        with patch.object(sys, "stdout", recorder):
            cli._cli_log("[Orchestrator] Running UKF")
        self.assertEqual(recorder.writes, ["[Orchestrator] Running UKF\n"])

    def test_p_c_a_stderr_notice_is_one_write_call_on_stderr(self) -> None:
        import shared.run_subject_pipeline_cli as cli

        out, err = self._Recorder(), self._Recorder()
        with patch.object(sys, "stdout", out), patch.object(sys, "stderr", err):
            cli._cli_log("[Targets] closure notice")
        self.assertEqual(err.writes, ["[Targets] closure notice\n"])
        self.assertEqual(out.writes, [])


# ------------------------------------------------------------------------------------------------------------- P-d
class PdHeartbeatDuringDoctorTests(unittest.TestCase):
    def _run_main_until_doctor(self, extra_args=()):
        import shared.run_subject_pipeline_cli as cli

        class Stop(Exception):
            pass

        seen = {}

        def fake_doctor(args, **kwargs):
            seen["threads"] = [thread.name for thread in threading.enumerate()]
            beat = [t for t in threading.enumerate() if t.name == "run-heartbeat"]
            seen["alive"] = bool(beat)
            raise Stop()

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(cli, "_preload_torch_for_openmp", lambda: True), \
                patch.object(cli, "_run_environment_doctor_from_args", fake_doctor):
            try:
                cli.main(["--subject-root", tmp, "--subject-id", "s1", "--no-user-config",
                          "--targets", "validation", *extra_args])
            except Stop:
                pass
        return seen

    def test_p_d_the_heartbeat_runs_while_the_doctor_runs_in_execute_mode(self) -> None:
        seen = self._run_main_until_doctor(("--execute",))
        self.assertTrue(seen.get("alive"), seen)
        deadline = __import__("time").monotonic() + 5
        while any(t.name == "run-heartbeat" for t in threading.enumerate()) and __import__("time").monotonic() < deadline:
            __import__("time").sleep(0.05)
        self.assertFalse(any(t.name == "run-heartbeat" for t in threading.enumerate()),
                         "the doctor heartbeat must stop when the doctor ends, also on an error")

    def test_p_d_a_plan_only_run_starts_no_heartbeat(self) -> None:
        seen = self._run_main_until_doctor(("--plan",))
        self.assertFalse(seen.get("alive"), seen)


# ------------------------------------------------------------------------------------------------------------- P-e
class PeFailureDialogAfterReleaseTests(unittest.TestCase):
    def test_p_e_the_failure_dialog_is_queued_and_shown_after_run_is_released(self) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import test_b4b_w1_gui_run_state as w1

        owner = w1._Owner()
        owner._handleBackgroundOutputLine("Traceback: boom")
        order = []
        original_show = owner._show_actionable_error
        owner._show_actionable_error = lambda **kwargs: (order.append(("dialog", owner._pipelineActive,
                                                                       owner._backgroundProcess)), original_show(**kwargs))
        owner._onBackgroundProcessFinished(1, w1.NORMAL)
        self.assertEqual(order, [], "the modal dialog must not open inside the finish callback")
        self.assertFalse(owner._pipelineActive)
        self.assertIsNone(owner._backgroundProcess)
        owner.run_scheduled()
        self.assertEqual(order, [("dialog", False, None)])
        self.assertEqual(len(owner.failures), 1)
        self.assertIn("Traceback: boom", owner.failures[0][1])

    def test_p_e_a_clean_run_queues_no_dialog(self) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import test_b4b_w1_gui_run_state as w1

        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "Test1_orchestration_manifest.json"
            manifest.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
            owner = w1._Owner()
            owner._backgroundManifestPath = manifest
            owner._onBackgroundProcessFinished(0, w1.NORMAL)
            self.assertEqual(w1._FakeTimer.scheduled, [])


if __name__ == "__main__":
    unittest.main()
