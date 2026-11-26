#!/usr/bin/env python3
"""B4b W1: GUI run state (H1), stall display (H2), batch progress (H3), worker kill (H4), pre-batch preflight (LE-5),
UTF-8 worker output (X1) and plan notices (LE hooks).

The widget methods are loaded from ``shared/module_widget.py`` by AST, as other tests of this repo do, so the tests
need no Slicer GUI.
"""

from __future__ import annotations

import ast
import contextlib
import importlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import ui_run_controller as controller
from shared.ui_background_process import Utf8StreamDecoder, background_process_env_overrides
from shared.ui_run_controller import (
    WORKER_ABNORMAL_EXIT_MARKER,
    WorkerLiveness,
    actionable_plan_blockers,
    build_batch_background_cli_args,
    classify_background_run_outcome,
    format_heartbeat_status,
    format_last_output,
    plan_notice_text,
    plan_warning_lines,
    rows_from_batch_progress,
    split_relayed_subject,
    worker_kill_info,
    worker_killed_message,
    worker_silence_message,
    worker_silence_seconds,
)
from shared.ui_run_history import set_live_stage_text

MODULE_WIDGET_PATH = REPO_ROOT / "shared" / "module_widget.py"
BATCH_CLI_PATH = REPO_ROOT / "scripts" / "run_batch_pipeline.py"

CRASH, NORMAL = 1, 0  # QProcess.ExitStatus values

_spec = importlib.util.spec_from_file_location("run_batch_pipeline_b4b", BATCH_CLI_PATH)
batch_cli = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = batch_cli
_spec.loader.exec_module(batch_cli)


# --------------------------------------------------------------------------------------------- widget harness
METHODS = (
    "_onBackgroundProcessReadyRead", "_qbytearray_to_text", "_begin_worker_tracking", "_end_worker_tracking",
    "_check_worker_silence", "_clear_worker_warning", "_show_worker_heartbeat", "_show_worker_stall",
    "_handleBackgroundOutputLine", "_extract_background_manifest_path", "_manifest_is_final",
    "_write_cancelled_manifest_safe", "_show_background_failure", "_onBackgroundProcessFinished",
    "_report_run_finish_error", "_discard_batch_progress_file",
)


class _Clock:
    now = 1000.0


class _FakeTimer:
    scheduled: list = []  # P-e: callbacks of QTimer.singleShot(0, ...); the tests run them by hand

    @staticmethod
    def singleShot(_msec, callback) -> None:
        _FakeTimer.scheduled.append(callback)

    def __init__(self) -> None:
        self.active, self.interval = False, 0

    def connect(self, *_args) -> None:
        pass

    def setInterval(self, value) -> None:
        self.interval = value

    def start(self) -> None:
        self.active = True

    def stop(self) -> None:
        self.active = False


class _FakeProcess:
    def __init__(self) -> None:
        self.running = True
        self.killed = False
        self.terminated = False
        self.pending = b""

    def readAllStandardOutput(self):
        data, self.pending = self.pending, b""
        return data

    def kill(self) -> None:
        self.killed = True

    def terminate(self) -> None:
        self.terminated = True


class _Label:
    def __init__(self) -> None:
        self.text = ""
        self.visible = False

    def setText(self, value) -> None:
        self.text = value

    def setVisible(self, value) -> None:
        self.visible = value


def _load_harness_class():
    tree = ast.parse(MODULE_WIDGET_PATH.read_text(encoding="utf-8"))
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in METHODS:
            found[node.name] = node
    missing = set(METHODS) - set(found)
    assert not missing, missing

    def set_run_state(owner, state):
        owner._runState = state

    namespace = {
        "Optional": __import__("typing").Optional,
        "Path": Path,
        "json": json,
        "deque": deque,
        "monotonic": lambda: _Clock.now,
        "qt": SimpleNamespace(QTimer=_FakeTimer),
        "qprocess_is_running": lambda process: bool(process is not None and process.running),
        "WorkerLiveness": WorkerLiveness,
        "Utf8StreamDecoder": Utf8StreamDecoder,
        "WORKER_ABNORMAL_EXIT_MARKER": WORKER_ABNORMAL_EXIT_MARKER,
        "classify_background_run_outcome": classify_background_run_outcome,
        "format_heartbeat_status": format_heartbeat_status,
        "format_last_output": format_last_output,
        "worker_kill_info": worker_kill_info,
        "worker_killed_message": worker_killed_message,
        "worker_silence_message": worker_silence_message,
        "extract_background_manifest_path": controller.extract_background_manifest_path,
        "parse_manifest_line": controller.parse_manifest_line,
        "parse_live_stage_from_log_line": lambda line, language: None,
        "qbytearray_to_text": lambda value: bytes(value).decode("utf-8", errors="replace"),
        "set_live_stage_text": set_live_stage_text,
        "set_run_state": set_run_state,
        "stop_progress_polling": lambda owner: setattr(owner, "_progressStopped", True),
        "stop_run_animation": lambda owner: setattr(owner, "_animationStopped", True),
    }
    body = []
    for name in METHODS:
        node = found[name]
        body.append(node)
    module = ast.Module(body=body, type_ignores=[])
    exec(compile(module, str(MODULE_WIDGET_PATH), "exec"), namespace)

    class Harness:
        pass

    for name in METHODS:
        setattr(Harness, name, namespace[name])
    return Harness


Harness = _load_harness_class()


class _Owner(Harness):
    def __init__(self) -> None:
        self.logs: list[str] = []
        self.statuses: list[tuple[str, str]] = []
        self.summaries: list[tuple[str, str]] = []
        self.failures: list[tuple[str, str]] = []
        self.active_calls: list[tuple] = []
        self._runState = "running"
        self._pipelineActive = True
        self._backgroundProcess = _FakeProcess()
        self._backgroundCancelRequested = False
        self._backgroundOutputBuffer = ""
        self._backgroundManifestPath = None
        self._backgroundBatchProgressPath = None
        self._runStageLabel = _Label()
        self._onboardingLanguage = "en"
        self.cancel_manifest_error: Exception | None = None
        self.rows_error: Exception | None = None
        _FakeTimer.scheduled = []
        self._begin_worker_tracking()

    def run_scheduled(self) -> None:
        """P-e: run the callbacks queued by QTimer.singleShot (the failure dialog is shown from there)."""
        while _FakeTimer.scheduled:
            _FakeTimer.scheduled.pop(0)()

    # recorders
    def appendLog(self, text) -> None:
        self.logs.append(str(text))

    def _setStatus(self, kind, message) -> None:
        self.statuses.append((kind, message))

    def _set_run_progress_message(self, text) -> None:
        pass

    def _set_run_summary_message(self, text, state) -> None:
        self.summaries.append((state, text))

    def _set_latest_report_path(self, path) -> None:
        pass

    def _set_latest_output_tree_path(self, path) -> None:
        pass

    def _refresh_run_history(self, select_path=None) -> None:
        pass

    def _poll_run_progress(self) -> None:
        pass

    def _task_keys_for_running_line(self, line):
        return ()

    def _update_run_status_table_item(self, *args) -> None:
        pass

    def _mark_unfinished_rows(self, status, reason) -> None:
        if self.rows_error is not None:
            raise self.rows_error

    def _write_cancelled_manifest(self):
        if self.cancel_manifest_error is not None:
            raise self.cancel_manifest_error
        return Path("/tmp/cancelled_manifest.json")

    def _load_background_manifest_result(self, path) -> None:
        self.loaded = path

    def _setPipelineActive(self, active, *, preserve_status=False) -> None:
        self.active_calls.append((active, preserve_status))
        self._pipelineActive = active

    def _show_actionable_error(self, **kwargs) -> None:
        self.failures.append((kwargs["what"], str(kwargs["details"])))

    def feed(self, raw: bytes) -> None:
        self._backgroundProcess.pending = raw
        self._onBackgroundProcessReadyRead()


def _heartbeat(stage="UKF", elapsed=125, cpu=40, stalled=None):
    text = f"[Heartbeat] stage={stage} elapsed={elapsed} cpu_s={cpu} children=2"
    return text + (f" stalled_min={stalled}" if stalled is not None else "")


# ------------------------------------------------------------------------------------------------------- H1
class H1FinishRobustnessTest(unittest.TestCase):
    def assertIdle(self, owner) -> None:
        self.assertIsNone(owner._backgroundProcess)
        self.assertFalse(owner._backgroundCancelRequested)
        self.assertEqual(owner.active_calls, [(False, True)])
        self.assertFalse(owner._pipelineActive)

    def test_h1_cancel_with_unwritable_manifest_leaves_the_ui_idle(self) -> None:
        owner = _Owner()
        owner._backgroundCancelRequested = True
        owner.cancel_manifest_error = OSError(28, "No space left on device")
        owner._onBackgroundProcessFinished(-15, CRASH)
        self.assertIdle(owner)
        self.assertEqual(owner._runState, "cancelled")
        self.assertTrue(any("Could not write the cancellation manifest" in line for line in owner.logs))
        self.assertTrue(any(kind == "warning" and "cancelled" in text for kind, text in owner.statuses))

    def test_h1_unexpected_exception_ends_the_run_as_failed(self) -> None:
        owner = _Owner()
        owner.rows_error = RuntimeError("table is gone")
        owner._onBackgroundProcessFinished(1, NORMAL)
        self.assertIdle(owner)
        self.assertEqual(owner._runState, "failed")
        self.assertTrue(any(kind == "error" and "table is gone" in text for kind, text in owner.statuses))
        self.assertTrue(any("table is gone" in line for line in owner.logs))

    def test_h1_exception_after_a_final_state_keeps_that_state(self) -> None:
        owner = _Owner()
        owner._backgroundCancelRequested = True
        owner._runState = "cancelled"
        owner.rows_error = RuntimeError("late failure")
        owner._onBackgroundProcessFinished(-15, CRASH)
        self.assertIdle(owner)
        self.assertEqual(owner._runState, "cancelled")

    def test_h1_the_single_release_call_is_in_a_finally_block(self) -> None:
        tree = ast.parse(MODULE_WIDGET_PATH.read_text(encoding="utf-8"))
        method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                      and node.name == "_onBackgroundProcessFinished")
        tries = [node for node in ast.walk(method) if isinstance(node, ast.Try)]
        self.assertTrue(tries and tries[0].finalbody)
        released = [node for node in ast.walk(ast.Module(body=tries[0].finalbody, type_ignores=[]))
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "_setPipelineActive"]
        self.assertEqual(len(released), 1)


# ------------------------------------------------------------------------------------------------------- H4
class H4WorkerKilledTest(unittest.TestCase):
    def test_h4_classify_uses_the_exit_status(self) -> None:
        kwargs = dict(cancelled=False, manifest_available=False)
        self.assertEqual(classify_background_run_outcome(exit_code=0, exit_status=CRASH, **kwargs), "worker_killed")
        self.assertEqual(classify_background_run_outcome(exit_code=137, exit_status=NORMAL, **kwargs), "worker_killed")
        self.assertEqual(classify_background_run_outcome(exit_code=-9, **kwargs), "worker_killed")
        self.assertEqual(classify_background_run_outcome(exit_code=1, abnormal_exit_seen=True, **kwargs), "worker_killed")
        self.assertEqual(classify_background_run_outcome(exit_code=1, exit_status=NORMAL, **kwargs), "process_failed")
        self.assertEqual(classify_background_run_outcome(exit_code=2, **kwargs), "process_failed")
        self.assertEqual(classify_background_run_outcome(exit_code=0, **kwargs), "no_manifest")

    def test_h4_cancel_and_a_final_manifest_win_over_a_crash_exit(self) -> None:
        self.assertEqual(classify_background_run_outcome(
            exit_code=-15, exit_status=CRASH, cancelled=True, manifest_available=False), "cancelled")
        self.assertEqual(classify_background_run_outcome(
            exit_code=139, exit_status=CRASH, cancelled=False, manifest_available=True), "manifest")

    def test_h4_a_running_checkpoint_manifest_does_not_hide_a_kill(self) -> None:
        self.assertEqual(classify_background_run_outcome(
            exit_code=137, exit_status=CRASH, cancelled=False, manifest_available=True, manifest_final=False),
            "worker_killed")
        # No kill sign: the manifest stays authoritative, as before.
        self.assertEqual(classify_background_run_outcome(
            exit_code=2, cancelled=False, manifest_available=True, manifest_final=False), "manifest")

    def test_h4_signal_numbers_and_windows_codes(self) -> None:
        self.assertEqual(worker_kill_info(137), {"signal": 9})
        self.assertEqual(worker_kill_info(-11), {"signal": 11})
        self.assertEqual(worker_kill_info(9, CRASH), {"signal": 9})
        self.assertEqual(worker_kill_info(0xC0000005, CRASH), {"signal": None})  # NTSTATUS: a crash, no signal
        self.assertIsNone(worker_kill_info(0xC0000005, NORMAL))
        self.assertIsNone(worker_kill_info(1))
        self.assertIsNone(worker_kill_info(2))
        self.assertIn("(signal 9)", worker_killed_message({"signal": 9}))
        message = worker_killed_message({"signal": None})
        self.assertNotIn("signal", message)
        self.assertIn("possibly out of memory", message)
        self.assertIn("Lower the number of parallel models or close other apps, then run again.", message)
        nt_msg = worker_killed_message({"signal": None}, exit_code=-1073741571)
        self.assertIn("stack overflow (0xC00000FD)", nt_msg)

    def test_h4_enum_like_exit_status_is_understood(self) -> None:
        class Crash:
            value = 1

            def __int__(self) -> int:
                raise TypeError

        self.assertEqual(worker_kill_info(0, Crash()), {"signal": None})
        self.assertEqual(worker_kill_info(0, "QProcess.CrashExit"), {"signal": None})

    def test_h4_widget_shows_the_message_and_the_last_20_lines(self) -> None:
        owner = _Owner()
        for index in range(25):
            owner._handleBackgroundOutputLine(f"line {index}")
        owner._onBackgroundProcessFinished(137, NORMAL)
        owner.run_scheduled()  # P-e: the dialog is queued, not shown inside the finish callback
        self.assertEqual(owner._runState, "failed")
        self.assertEqual(len(owner.failures), 1)
        what, details = owner.failures[0]
        self.assertIn("worker was killed (signal 9), possibly out of memory", what)
        self.assertIn("line 24", details)
        self.assertIn("line 5", details)
        self.assertNotIn("line 4\n", details + "\n")
        self.assertEqual(len(details.splitlines()), 20)
        self.assertEqual(owner.active_calls, [(False, True)])

    def test_h4_the_launcher_abnormal_exit_line_marks_a_kill(self) -> None:
        owner = _Owner()
        owner._handleBackgroundOutputLine(
            "error: [/Applications/Slicer.app/Contents/bin/./python-real] exit abnormally - Report the problem.")
        owner._onBackgroundProcessFinished(1, NORMAL)
        owner.run_scheduled()
        self.assertTrue(owner.failures and "worker was killed" in owner.failures[0][0])

    def test_h4_a_plain_failure_is_not_called_a_kill_but_shows_the_tail(self) -> None:
        owner = _Owner()
        owner._handleBackgroundOutputLine("Traceback: boom")
        owner._onBackgroundProcessFinished(1, NORMAL)
        owner.run_scheduled()
        self.assertEqual(owner._runState, "failed")
        self.assertNotIn("killed", owner.failures[0][0])
        self.assertIn("Traceback: boom", owner.failures[0][1])

    def test_h4_final_manifest_after_crash_exit_is_loaded_not_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "Test1_orchestration_manifest.json"
            manifest.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
            owner = _Owner()
            owner._backgroundManifestPath = manifest
            owner._onBackgroundProcessFinished(139, CRASH)
            self.assertEqual(getattr(owner, "loaded", None), manifest)
            self.assertEqual(owner.failures, [])


# ------------------------------------------------------------------------------------------------------- X1
class X1Utf8Test(unittest.TestCase):
    def test_x1_overrides_force_utf8_for_the_worker(self) -> None:
        overrides = background_process_env_overrides()
        self.assertEqual(overrides["PYTHONUNBUFFERED"], "1")
        self.assertEqual(overrides["PYTHONUTF8"], "1")
        self.assertEqual(overrides["PYTHONIOENCODING"], "utf-8")

    def test_x1_a_worker_can_print_a_non_ascii_path_with_the_overrides(self) -> None:
        code = "print('/data/\\u8def\\u5f84/sub-01')"
        legacy = {**os.environ, "PYTHONIOENCODING": "ascii"}  # a strict legacy codec, like cp1252 for these characters
        legacy.pop("PYTHONUTF8", None)
        failing = subprocess.run([sys.executable, "-c", code], env=legacy, capture_output=True)
        self.assertNotEqual(failing.returncode, 0)  # fails before the fix
        fixed = subprocess.run([sys.executable, "-c", code], env={**legacy, **background_process_env_overrides()},
                               capture_output=True)
        self.assertEqual(fixed.returncode, 0, fixed.stderr)
        self.assertEqual(fixed.stdout.decode("utf-8").strip(), "/data/路径/sub-01")

    def test_x1_a_character_split_across_two_reads_is_decoded_once(self) -> None:
        data = "[Batch] start 路径-é\n".encode("utf-8")
        for cut in range(1, len(data)):
            decoder = Utf8StreamDecoder()
            text = decoder.decode(data[:cut]) + decoder.decode(data[cut:]) + decoder.flush()
            self.assertEqual(text, data.decode("utf-8"), cut)
            self.assertNotIn("�", text)

    def test_x1_widget_reads_split_characters_without_corruption(self) -> None:
        owner = _Owner()
        data = "[Connectome] 路径\n".encode("utf-8")
        owner.feed(data[:14])
        owner.feed(data[14:])
        self.assertEqual(owner.logs, ["[Connectome] 路径"])

    def test_x1_invalid_bytes_are_replaced_not_raised(self) -> None:
        self.assertEqual(Utf8StreamDecoder().decode(b"ok \xff\xfe"), "ok ��")


# ------------------------------------------------------------------------------------------------------- H2
class H2StallDisplayTest(unittest.TestCase):
    def setUp(self) -> None:
        _Clock.now = 1000.0

    def test_h2_heartbeats_go_to_the_stage_label_not_the_log(self) -> None:
        owner = _Owner()
        _Clock.now += 60
        owner._handleBackgroundOutputLine(_heartbeat(elapsed=125, cpu=40))
        _Clock.now += 60
        owner._handleBackgroundOutputLine(_heartbeat(elapsed=185, cpu=90))
        self.assertEqual(owner.logs, [])
        # Two minutes with heartbeats only: the label also says how long ago the last real output was.
        self.assertEqual(owner._runStageLabel.text, "UKF - running for 3m, CPU active, last output 2m ago")
        self.assertTrue(owner._runStageLabel.visible)
        owner._handleBackgroundOutputLine("[UKF] fiber 1000")
        _Clock.now += 5
        owner._handleBackgroundOutputLine(_heartbeat(elapsed=190, cpu=95))
        self.assertEqual(owner._runStageLabel.text, "UKF - running for 3m, CPU active")

    def test_h2_idle_cpu_and_last_output_age_are_shown(self) -> None:
        owner = _Owner()
        owner._handleBackgroundOutputLine(_heartbeat(elapsed=60, cpu=40))
        _Clock.now += 300
        owner._handleBackgroundOutputLine(_heartbeat(elapsed=360, cpu=40))
        self.assertEqual(owner._runStageLabel.text, "UKF - running for 6m, CPU idle, last output 5m ago")

    def test_h2_stall_warning_is_visible_and_clears_when_progress_resumes(self) -> None:
        owner = _Owner()
        owner._handleBackgroundOutputLine(_heartbeat(stalled=10, cpu=40))
        stall = ("[Stall warning] UKF: no CPU progress and no output for 10 min. The step may be hung; "
                 "press Cancel to stop it, or wait if the tool is known to be quiet.")
        owner._handleBackgroundOutputLine(stall)
        self.assertEqual(owner.logs, [stall])
        kind, message = owner.statuses[-1]
        self.assertEqual(kind, "warning")
        self.assertTrue(message.startswith("UKF: no CPU progress and no output for 10 min."))
        self.assertIn("no progress for 10 min", owner._runStageLabel.text)
        _Clock.now += 60
        owner._handleBackgroundOutputLine(_heartbeat(elapsed=900, cpu=80))
        self.assertEqual(owner.statuses[-1], ("running", "Running pipeline..."))
        self.assertTrue(any("resumed" in line for line in owner.logs))

    def test_h2_a_normal_output_line_also_clears_the_stall(self) -> None:
        owner = _Owner()
        owner._handleBackgroundOutputLine("[Stall warning] WMA: no CPU progress and no output for 10 min. The step may be hung.")
        self.assertEqual(owner.statuses[-1][0], "warning")
        owner._handleBackgroundOutputLine("[Orchestrator] WMA completed in 1.0s")
        self.assertEqual(owner.statuses[-1], ("running", "Running pipeline..."))

    def test_h2_stall_never_kills_the_worker(self) -> None:
        owner = _Owner()
        owner._handleBackgroundOutputLine(_heartbeat(stalled=30))
        owner._handleBackgroundOutputLine("[Stall warning] UKF: no CPU progress and no output for 30 min. x")
        _Clock.now += 100000
        owner._check_worker_silence()
        process = owner._backgroundProcess
        self.assertFalse(process.killed or process.terminated)

    def test_h2_silence_is_a_pure_function_of_clock_and_interval(self) -> None:
        self.assertIsNone(worker_silence_seconds(now=1179, last_output_at=1000, interval_seconds=60))
        self.assertEqual(worker_silence_seconds(now=1180, last_output_at=1000, interval_seconds=60), 180)
        self.assertEqual(
            worker_silence_message(180),
            "No output from the worker for 3 min (no heartbeat). It may be hung; press Cancel to stop it.")

    def test_h2_widget_warns_after_three_silent_intervals_and_clears_on_output(self) -> None:
        owner = _Owner()
        _Clock.now += 120
        owner._check_worker_silence()
        self.assertEqual(owner.statuses, [])
        _Clock.now += 80
        owner._check_worker_silence()
        owner._check_worker_silence()
        self.assertEqual(owner.statuses[-1], (
            "warning", "No output from the worker for 3 min (no heartbeat). It may be hung; press Cancel to stop it."))
        self.assertEqual(len([line for line in owner.logs if "No output from the worker" in line]), 1)
        owner._handleBackgroundOutputLine("[Preproc] eddy started")
        self.assertEqual(owner.statuses[-1], ("running", "Running pipeline..."))

    def test_h2_a_heartbeat_counts_as_output_for_the_silence_check(self) -> None:
        owner = _Owner()
        for _ in range(5):
            _Clock.now += 100
            owner._handleBackgroundOutputLine(_heartbeat())
            owner._check_worker_silence()
        self.assertFalse(any("No output" in text for _kind, text in owner.statuses))

    def test_h2_silence_check_ignores_a_finished_process_and_a_cancel(self) -> None:
        owner = _Owner()
        _Clock.now += 1000
        owner._backgroundProcess.running = False
        owner._check_worker_silence()
        owner._backgroundProcess.running = True
        owner._backgroundCancelRequested = True
        owner._check_worker_silence()
        self.assertEqual(owner.statuses, [])

    def test_h2_the_timer_starts_with_the_run_and_stops_when_it_finishes(self) -> None:
        owner = _Owner()
        self.assertTrue(owner._workerLivenessTimer.active)
        owner._onBackgroundProcessFinished(0, NORMAL)
        self.assertFalse(owner._workerLivenessTimer.active)

    def test_h2_relayed_batch_lines_round_trip_through_the_prefix(self) -> None:
        line = batch_cli.relay_line("sub-01", _heartbeat(stage="WMA", elapsed=300, cpu=12))
        self.assertEqual(line, "[sub-01] [Heartbeat] stage=WMA elapsed=300 cpu_s=12 children=2")
        subject, bare = split_relayed_subject(line)
        self.assertEqual((subject, bare), ("sub-01", "[Heartbeat] stage=WMA elapsed=300 cpu_s=12 children=2"))
        self.assertEqual(split_relayed_subject("[Heartbeat] stage=x elapsed=1")[0], None)
        self.assertEqual(split_relayed_subject("[Batch] start sub-01 (1/3)")[0], None)
        owner = _Owner()
        owner._handleBackgroundOutputLine(line)
        self.assertEqual(owner.logs, [])
        self.assertEqual(owner._runStageLabel.text, "sub-01: WMA - running for 5m, CPU active")
        owner._handleBackgroundOutputLine("[sub-02] [Stall warning] UKF: no CPU progress and no output for 10 min. x")
        self.assertTrue(owner.statuses[-1][1].startswith("sub-02: UKF: no CPU progress"))

    def test_h2_liveness_tracks_cpu_per_subject(self) -> None:
        liveness = WorkerLiveness(0.0, 60.0)
        first = liveness.observe("[a] " + _heartbeat(cpu=10), 1.0)
        other = liveness.observe("[b] " + _heartbeat(cpu=10), 2.0)
        again = liveness.observe("[a] " + _heartbeat(cpu=10), 3.0)
        self.assertTrue(first["cpu_active"] and other["cpu_active"])
        self.assertFalse(again["cpu_active"])

    def test_h2_a_healthy_subject_does_not_clear_another_subjects_stall(self) -> None:
        owner = _Owner()
        owner._handleBackgroundOutputLine("[a] " + _heartbeat(stalled=10))
        owner._handleBackgroundOutputLine("[a] [Stall warning] UKF: no CPU progress and no output for 10 min. x")
        self.assertEqual(owner.statuses[-1][0], "warning")
        count = len(owner.statuses)
        owner._handleBackgroundOutputLine("[b] " + _heartbeat(elapsed=70, cpu=50))
        owner._handleBackgroundOutputLine("[b] " + _heartbeat(elapsed=130, cpu=90))
        owner._handleBackgroundOutputLine("[Batch] start c (3/3)")
        self.assertEqual(len(owner.statuses), count)
        self.assertFalse(any("resumed" in line for line in owner.logs))
        owner._handleBackgroundOutputLine("[a] " + _heartbeat(elapsed=900, cpu=99))
        self.assertEqual(owner.statuses[-1], ("running", "Running pipeline..."))
        self.assertTrue(any("resumed" in line for line in owner.logs))

    def test_h2_a_finished_subject_ends_its_own_stall(self) -> None:
        liveness = WorkerLiveness(0.0, 60.0)
        liveness.observe("[a] [Stall warning] UKF: no CPU progress and no output for 10 min. x", 1.0)
        self.assertTrue(liveness.stalled)
        event = liveness.observe("[Batch] done a status=failed elapsed=700s (1/2)", 2.0)
        self.assertFalse(liveness.stalled)
        self.assertTrue(event["resumed"])

    def test_h2_experiment_campaign_has_no_silence_timer(self) -> None:
        owner = _Owner()
        self.assertTrue(owner._workerLivenessTimer.active)
        owner._begin_worker_tracking(silence_check=False)
        self.assertFalse(owner._workerLivenessTimer.active)
        source = MODULE_WIDGET_PATH.read_text(encoding="utf-8")
        self.assertIn("self._begin_worker_tracking(silence_check=launch_args is None)", source)

    def test_h2_the_heartbeat_formatter_handles_a_missing_stage_and_chinese(self) -> None:
        self.assertEqual(format_heartbeat_status({"stage": "-", "elapsed": 30}), "Pipeline - running for 30s")
        text = format_heartbeat_status({"stage": "UKF", "elapsed": 120}, cpu_active=True, zh=True)
        self.assertIn("已运行 2m", text)


# ------------------------------------------------------------------------------------------------------- H3
FAKE_SUBJECT = '''import argparse, json, os, sys, time
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--subject-root")
p.add_argument("--subject-id")
p.add_argument("--output-root")
p.add_argument("--fail", default="")
a, _extra = p.parse_known_args()
print("[Heartbeat] stage=UKF elapsed=61 cpu_s=30", flush=True)
print("an ordinary line that is not relayed", flush=True)
time.sleep(1.3)
seen = os.environ.get("B4B_PROGRESS_FILE")
if seen and Path(seen).is_file():
    snapshots = Path(os.environ["B4B_SEEN_DIR"])
    snapshots.mkdir(exist_ok=True)
    (snapshots / (a.subject_id + ".json")).write_text(Path(seen).read_text())
sys.exit(3 if a.subject_id == a.fail else 0)
'''


class H3BatchProgressTest(unittest.TestCase):
    def run_batch(self, root: Path, *extra, jobs=1):
        for subject in ("sub-01", "sub-02", "sub-03"):
            folder = root / "subjects" / subject / "input"
            folder.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (folder / name).write_text("x", encoding="utf-8")
        script = root / "pipeline.py"
        script.write_text(FAKE_SUBJECT, encoding="utf-8")
        out = io.StringIO()
        progress = root / "progress" / "run.json"
        env = {"B4B_PROGRESS_FILE": str(progress), "B4B_SEEN_DIR": str(root / "seen")}
        with patch.dict(os.environ, env), contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            status = batch_cli.main([
                "--batch-root", str(root / "subjects"), "--output-root", str(root / "outputs"),
                "--pipeline-script", str(script), "--python-executable", sys.executable,
                "--manifest", str(root / "batch.json"), "--execute", "--jobs", str(jobs),
                "--progress-file", str(progress), *extra,
            ])
        return status, out.getvalue(), progress

    def test_h3_batch_prints_start_and_done_lines_and_relays_heartbeats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            status, stdout, _progress = self.run_batch(Path(tmp))
            self.assertEqual(status, 0)
            lines = stdout.splitlines()
            self.assertIn("[Batch] start sub-01 (1/3)", lines)
            self.assertIn("[Batch] start sub-03 (3/3)", lines)
            done = [line for line in lines if line.startswith("[Batch] done ")]
            self.assertEqual(len(done), 3)
            self.assertRegex(done[0], r"^\[Batch\] done sub-01 status=\w+ elapsed=\d+s \(1/3\)$")
            self.assertRegex(done[2], r"\(3/3\)$")
            relayed = [line for line in lines if line.startswith("[sub-0")]
            self.assertEqual(sorted(relayed), [f"[sub-0{i}] [Heartbeat] stage=UKF elapsed=61 cpu_s=30" for i in (1, 2, 3)])
            self.assertNotIn("an ordinary line that is not relayed", stdout)
            self.assertLess(lines.index("[Batch] start sub-01 (1/3)"), lines.index(relayed[0]))

    def test_h3_progress_file_is_written_after_each_subject(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            status, _stdout, progress = self.run_batch(root)
            self.assertEqual(status, 0)
            final = json.loads(progress.read_text())
            self.assertEqual(final["kind"], "slicerdsurfer_batch_progress")
            self.assertEqual((final["status"], final["total"], final["running"]), ("finished", 3, []))
            self.assertEqual([item["subject_id"] for item in final["results"]], ["sub-01", "sub-02", "sub-03"])
            # What a running subject saw: while sub-02 ran, sub-01 was finished and sub-02 was running.
            middle = json.loads((root / "seen" / "sub-02.json").read_text())
            self.assertEqual(middle["status"], "running")
            self.assertEqual(middle["running"], ["sub-02"])
            self.assertEqual([item["subject_id"] for item in middle["results"]], ["sub-01"])
            self.assertFalse(list(progress.parent.glob("*.tmp")))

    def test_h3_a_dead_subject_is_recorded_failed_at_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            status, stdout, progress = self.run_batch(Path(tmp), "--continue-on-error", "--fail", "sub-02")
            # --fail is forwarded to the subject script; the batch parser does not know it.
            self.assertEqual(status, 2)
            self.assertIn("[Batch] done sub-02 status=failed", stdout)
            record = json.loads(progress.read_text())
            by_id = {item["subject_id"]: item for item in record["results"]}
            self.assertEqual((by_id["sub-02"]["returncode"], by_id["sub-02"]["status"]), (3, "failed"))

    def test_h3_a_progress_write_error_never_fails_the_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            blocker = Path(tmp) / "not_a_folder"
            blocker.write_text("x")
            batch_cli.write_progress_file(blocker / "inside.json", {"kind": "x"})  # parent is a file: OSError inside
            batch_cli.write_progress_file(None, {"kind": "x"})

    def test_h3_no_progress_file_without_the_option(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = batch_cli.build_parser().parse_args(["--batch-root", str(root)])
            self.assertIsNone(args.progress_file)

    def test_h3_gui_passes_the_progress_path_to_the_batch_script(self) -> None:
        kwargs = dict(run_name="r", targets=("validation",), tractography_backend="csd_act", bundle_backend="wmql",
                      connectome_backend="auto", select_streamlines=1000, ddseg_device="auto", surface_device="auto")
        args = build_batch_background_cli_args(
            kwargs, {}, repo_root=REPO_ROOT, batch_root=Path("/b"), batch_size=2, arrange_inputs="none",
            progress_file=Path("/tmp/SlicerDSpace/batch_progress_1.json"))
        self.assertEqual(args[args.index("--progress-file") + 1], "/tmp/SlicerDSpace/batch_progress_1.json")
        without = build_batch_background_cli_args(
            kwargs, {}, repo_root=REPO_ROOT, batch_root=Path("/b"), batch_size=2, arrange_inputs="none")
        self.assertNotIn("--progress-file", without)

    def test_h3_widget_batch_start_polls_the_progress_file(self) -> None:
        source = MODULE_WIDGET_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                      and node.name == "_startBackgroundBatchPipeline")
        calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "start_progress_polling"]
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(calls[0].args[1], ast.Name)  # a path, no longer None
        self.assertTrue(any(keyword.arg == "batch_progress" for keyword in calls[0].keywords))

    def test_h3_progress_rows_and_counts(self) -> None:
        record = {"kind": "slicerdsurfer_batch_progress", "total": 4, "subjects": ["a", "b", "c", "d"], "running": ["c"],
                  "results": [{"subject_id": "a", "returncode": 0, "status": "completed"},
                              {"subject_id": "b", "returncode": 3, "status": "failed"}]}
        rows, done, total = rows_from_batch_progress(record)
        self.assertEqual((done, total), (2, 4))
        self.assertEqual([(row["target"], row["status"]) for row in rows],
                         [("a", "completed"), ("b", "failed"), ("c", "running"), ("d", "queued")])

    def test_h3_poll_applies_the_progress_file_to_the_widget(self) -> None:
        from shared import ui_run_history

        class Bar:
            def __init__(self) -> None:
                self.range, self.value, self.format = None, None, None

            def setRange(self, low, high) -> None:
                self.range = (low, high)

            def setValue(self, value) -> None:
                self.value = value

            def setFormat(self, text) -> None:
                self.format = text

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.json"
            path.write_text(json.dumps({
                "kind": "slicerdsurfer_batch_progress", "total": 3, "subjects": ["a", "b", "c"], "running": ["b"],
                "results": [{"subject_id": "a", "returncode": 0, "status": "completed"}]}), encoding="utf-8")
            rows_seen = []
            owner = SimpleNamespace(_runProgressBar=Bar(), _runState="running", _runProgressTimer=None,
                                    _set_run_status_table_rows=rows_seen.append)
            with patch.object(ui_run_history, "refresh_run_progress_styles", lambda _owner: None):
                ui_run_history.start_progress_polling(owner, path, batch_progress=True)
                owner._runState = "running"
                ui_run_history.poll_run_progress(owner)
            self.assertEqual(owner._runProgressCounts, (1, 3))
            self.assertEqual(owner._runProgressBar.range, (0, 3))
            self.assertEqual(owner._runProgressBar.value, 1)
            self.assertEqual([row["status"] for row in rows_seen[-1]], ["completed", "running", "queued"])
            # A later single-subject run polls an orchestration manifest again.
            ui_run_history.start_progress_polling(owner, None)
            self.assertFalse(owner._runProgressIsBatch)


# ------------------------------------------------------------------------------------------------------ LE-5
class LE5PreflightBeforeBatchTest(unittest.TestCase):
    def _method(self):
        tree = ast.parse(MODULE_WIDGET_PATH.read_text(encoding="utf-8"))
        return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                    and node.name == "_onRunSelectedClicked")

    def _lines(self, method, name):
        return sorted(node.lineno for node in ast.walk(method) if isinstance(node, ast.Call)
                      and ((isinstance(node.func, ast.Attribute) and node.func.attr == name)
                           or (isinstance(node.func, ast.Name) and node.func.id == name)))

    def test_le5_the_batch_starts_only_after_the_doctor_and_preflight(self) -> None:
        method = self._method()
        starts = self._lines(method, "_startBackgroundBatchPipeline")
        self.assertEqual(len(starts), 1)
        for name in ("_environment_doctor_result_for_run", "preflight_blockers_for_targets", "start_environment_check"):
            lines = self._lines(method, name)
            self.assertTrue(lines, name)
            self.assertLess(max(lines), starts[0], name)

    def test_le5_batch_and_single_share_one_doctor_signature_path(self) -> None:
        method = self._method()
        # One doctor block for both modes: a second call site would risk two signatures and a re-entry loop.
        self.assertEqual(len(self._lines(method, "_environment_doctor_result_for_run")), 1)
        self.assertEqual(len(self._lines(method, "start_environment_check")), 1)
        self.assertEqual(len(self._lines(method, "preflight_blockers_for_targets")), 1)


# --------------------------------------------------------------------------------------------- LE hooks
def _plan(items, warnings=()):
    from shared.orchestrator import RunPlanItem, RunPlanResult

    return RunPlanResult(
        subject_id="s", subject_root=Path("/s"), output_root=Path("/o"), run_name="r", targets=("noddi",),
        run_plan_path=Path("/p.json"),
        items=tuple(RunPlanItem(target=t, task_name=t, status=status, reason=reason, reason_category=category)
                    for t, status, reason, category in items),
        warnings=tuple(warnings))


class LeHooksTest(unittest.TestCase):
    def test_le_blocked_reasons_of_le1_le2_le3_reach_the_notice(self) -> None:
        plan = _plan([
            ("noddi", "blocked", "NODDI needs at least two non-zero b-shells", "acquisition_or_backend"),
            ("segmentation", "blocked", "model weights incomplete: missing 3 files", "data"),
            ("surface", "blocked", "needs 8 GiB GPU memory", "environment"),
            ("connectome", "blocked", "missing tractogram", "upstream"),
            ("microstructure", "will_run", "ready", "runnable"),
        ])
        blockers = actionable_plan_blockers(plan)
        self.assertEqual(set(blockers), {"noddi", "segmentation", "surface"})
        text = plan_notice_text(blockers, ("microstructure",), [])
        self.assertIn("- noddi: NODDI needs at least two non-zero b-shells", text)
        self.assertIn("- segmentation: model weights incomplete", text)
        self.assertIn("Runnable targets: microstructure", text)

    def test_le4_plan_warnings_are_shown(self) -> None:
        from shared.ui_run_controller import modal_plan_warnings

        plan = _plan([("microstructure", "will_run", "ready", "runnable")],
                     warnings=["Low disk space: 3.0 GiB free, about 12 GiB needed.", "  "])
        self.assertEqual(plan_warning_lines(plan), ["Low disk space: 3.0 GiB free, about 12 GiB needed."])
        self.assertEqual(modal_plan_warnings(plan_warning_lines(plan)), ["Low disk space: 3.0 GiB free, about 12 GiB needed."])
        text = plan_notice_text({}, ("microstructure",), plan_warning_lines(plan))
        self.assertIn("Warnings:\n- Low disk space", text)
        self.assertNotIn("Blocked:", text)

        # Soft recommended-RAM advisory should stay in logs without popping a blocking modal, while
        # critical host-memory warnings still surface as a modal notice.
        advisory = (
            "Available host memory for DDParcel is 4.8 GiB available RAM "
            "(below the recommended 5.5 GiB); close other memory-heavy applications."
        )
        critical = (
            "DDParcel host memory critically low (2.1 GiB available RAM; minimum working set ~3.2 GiB)."
        )
        self.assertEqual(modal_plan_warnings([advisory]), [])
        self.assertEqual(modal_plan_warnings([advisory, critical]), [critical])

    def test_le_widget_uses_the_actionable_blockers_and_warnings(self) -> None:
        source = MODULE_WIDGET_PATH.read_text(encoding="utf-8")
        run = source[source.index("    def _onRunSelectedClicked"):source.index("    def _show_topup_error")]
        self.assertIn("actionable_plan_blockers(plan)", run)
        self.assertIn("plan_warning_lines(plan)", run)
        self.assertIn("modal_plan_warnings(plan_warnings)", run)
        self.assertNotIn("environment_blockers_from_plan(plan)", run)
        notice_fn = source[source.index("    def _show_partial_run_notice"):source.index("    def _setStatus")]
        self.assertIn('"SlicerDSpace partial run" if blockers else "SlicerDSpace notice"', notice_fn)

    def test_le10_validation_receives_the_selected_targets(self) -> None:
        previous = sys.modules.get("slicer")
        try:
            sys.modules["slicer"] = types.ModuleType("slicer")
            logic_module = importlib.import_module("shared.module_logic")
        finally:
            if previous is None:
                sys.modules.pop("slicer", None)
            else:
                sys.modules["slicer"] = previous

        class Logic(logic_module.SharedLogicMixin):
            def __init__(self) -> None:
                self.module_dir = REPO_ROOT

            def log(self, _message) -> None:
                pass

        verdict = SimpleNamespace(report_path=None, status="passed", error_count=0, warning_count=0)
        with patch.object(logic_module, "validate_subject_inputs", return_value=verdict) as validate:
            Logic().runInputValidation(subject_id="s", input_root=Path("/s"), selected_targets=("noddi", "surface"))
            self.assertEqual(validate.call_args.args[0].selected_targets, ("noddi", "surface"))
            Logic().runInputValidation(subject_id="s", input_root=Path("/s"))
            self.assertEqual(validate.call_args.args[0].selected_targets, ())


# ------------------------------------------------------------------------------------------------------- FD4
class _RescanOwner(_Owner):
    """Records the Results rescans. A rescan while the user views another run shows that run's summary
    (``show_record``); the failure summary must still be the last one."""

    def __init__(self) -> None:
        super().__init__()
        self.rescans: list = []

    def _refresh_run_history(self, select_path=None) -> None:
        self.rescans.append((self._runState, select_path))
        self._set_run_summary_message("summary of the run the user views", "completed")


class FD4RunEndRescanTest(unittest.TestCase):
    def test_fd4_every_failed_run_end_rescans_the_results_after_the_state_change(self) -> None:
        """FD4(a): worker_killed, process_failed and no_manifest rescan, so a never-scanned failed run appears and
        the tree is not built from a scan taken while the run was running."""

        for exit_code, outcome in ((137, "worker_killed"), (1, "process_failed"), (0, "no_manifest")):
            with self.subTest(outcome=outcome):
                owner = _RescanOwner()
                owner._onBackgroundProcessFinished(exit_code, NORMAL)
                self.assertEqual(owner._runState, "failed")
                self.assertEqual(owner.rescans, [("failed", None)])
                self.assertEqual(owner.summaries[-1][0], "failed")  # the failure summary stays on top

    def test_fd4_a_finish_error_rescans_too(self) -> None:
        owner = _RescanOwner()
        owner.rows_error = RuntimeError("table is gone")
        owner._onBackgroundProcessFinished(1, NORMAL)
        self.assertEqual(owner._runState, "failed")
        self.assertEqual(owner.rescans[-1], ("failed", None))


if __name__ == "__main__":
    unittest.main()
