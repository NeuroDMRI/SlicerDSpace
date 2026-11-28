#!/usr/bin/env python3
"""Tests for Testing/Python/slicer_close_guard_smoke.py (the real-Slicer close-guard smoke script).

The script imports ``qt`` and ``slicer`` and calls ``main()`` at import. These tests load it with
minimal fakes in ``sys.modules`` and drive ``_run_checks`` directly, so no Slicer is needed.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT = Path(__file__).resolve().parent / "slicer_close_guard_smoke.py"

PRECONDITION_KEYS = {
    "helper_running_at_start",
    "guard_module_from_repo",
    "guard_installed",
    "window_visible_before_close",
    "synthetic_close_consumed",
}


class FakeProcess:
    def __init__(self):
        self._state = FakeQProcess.NotRunning

    def start(self, program, args=None):
        self._state = FakeQProcess.Running

    def waitForStarted(self, ms=0):
        return True

    def state(self):
        return self._state

    def kill(self):
        self._state = FakeQProcess.NotRunning

    def waitForFinished(self, ms=0):
        return True


class FakeQProcess(FakeProcess):
    NotRunning = 0
    Running = 2


class FakeCloseEvent:
    def __init__(self):
        self._accepted = True

    def isAccepted(self):
        return self._accepted

    def ignore(self):
        self._accepted = False

    def accept(self):
        self._accepted = True


class FakeAction:
    def __init__(self, window):
        self._window = window

    def trigger(self):
        self._window.exit_triggers += 1
        self._window.close()


class FakeWindow:
    def __init__(self):
        self.visible = True
        self.owner = None
        self.close_calls = 0
        self.exit_triggers = 0

    def isVisible(self):
        return self.visible

    def installEventFilter(self, event_filter):
        pass

    def removeEventFilter(self, event_filter):
        pass

    def close(self):
        # Like Qt: send a Close event to the filter; a consumed event keeps the window open.
        self.close_calls += 1
        event = FakeCloseEvent()
        consumed = self.owner._closeGuardFilter.eventFilter(self, event)
        return not consumed

    def findChild(self, cls, name):
        return FakeAction(self) if name == "FileExitAction" else None


class PassThroughFilter:
    """A filter that does not consume the Close event (the guard is not working)."""

    def eventFilter(self, obj, event):
        return False


class FakeFilter:
    """Stands in for the installed CloseGuardEventFilter: consume the event and record one ask."""

    def __init__(self, module):
        self._module = module

    def eventFilter(self, obj, event):
        self._module.ask_calls.append(([{"kind": "run"}], "quit"))
        event.ignore()
        return True


def _fake_modules():
    qt = types.ModuleType("qt")
    qt.QProcess = FakeQProcess
    qt.QTimer = types.SimpleNamespace(singleShot=lambda ms, fn: None)  # recorded by nobody; never called
    qt.qVersion = lambda: "5.15.2"
    qt.QCloseEvent = FakeCloseEvent
    qt.QAction = type("QAction", (), {})
    qt.QWidget = type("QWidget", (), {})
    qt.QEvent = types.SimpleNamespace(Close=19)
    window = FakeWindow()
    slicer = types.ModuleType("slicer")
    slicer.app = types.SimpleNamespace(
        applicationVersion="5.10.0",
        exit=mock.Mock(),
        closeAllWindows=lambda: None,
    )
    slicer.util = types.SimpleNamespace(mainWindow=lambda: window)
    return qt, slicer, window


class SmokeScriptTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.result_path = Path(tmp.name) / "result.json"
        qt, slicer, window = _fake_modules()
        self.window = window
        self.slicer = slicer
        env = mock.patch.dict(os.environ, {"SLICERDSPACE_SMOKE_RESULT": str(self.result_path)})
        mods = mock.patch.dict(sys.modules, {"qt": qt, "slicer": slicer})
        env.start()
        mods.start()
        self.addCleanup(env.stop)
        self.addCleanup(mods.stop)
        # Importing the script runs main(), which replaces the real guard's ask function; restore it.
        from shared import ui_close_guard

        ask = mock.patch.object(ui_close_guard, "_ask_with_message_box", ui_close_guard._ask_with_message_box)
        ask.start()
        self.addCleanup(ask.stop)
        spec = importlib.util.spec_from_file_location("slicer_close_guard_smoke_under_test", SCRIPT)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def run_checks(self, close_all_windows=None, *, helper_running=True, guard_installed=True,
                   window_visible=True, event_filter=None, module_from_repo=True):
        """Drive _run_checks with fakes. Return (result, handle_close_event mock, exit code)."""
        module = self.module
        process = FakeQProcess()
        if helper_running:
            process.start("sleep", ["60"])
        owner = types.SimpleNamespace(_backgroundProcess=process, _backgroundRunContext={"run_name": "t"})
        if guard_installed:
            owner._closeGuardFilter = event_filter if event_filter is not None else FakeFilter(module)
        self.window.owner = owner
        self.window.visible = window_visible
        self.close_all_calls = 0

        def closeAllWindows():
            self.close_all_calls += 1
            if close_all_windows is not None:
                close_all_windows(self.window, process, module)

        self.slicer.app.closeAllWindows = closeAllWindows
        result = {"checks": {"guard_module_from_repo": module_from_repo}, "passed": False}
        with mock.patch.object(module.ui_close_guard, "handle_close_event", return_value=False) as handle:
            with contextlib.redirect_stdout(io.StringIO()):  # _finish prints the result JSON
                module._run_checks(owner, process, result)
        self.slicer.app.exit.assert_called_once()
        return result, handle, self.slicer.app.exit.call_args[0][0]


def _one_ask_close_all(window, process, module):
    window.close()  # one refused close, window stays visible, job keeps running


def _hide_window_close_all(window, process, module):
    window.close()
    window.visible = False  # regression: the window disappears


def _stop_helper_close_all(window, process, module):
    window.close()
    process.kill()  # regression: the job stops


def _two_asks_close_all(window, process, module):
    window.close()
    window.close()


class EarlyStopAfterCloseAllWindowsTest(SmokeScriptTestCase):
    """Opus r2 P2-1: step (3) stops early like steps (0)-(2); step (4) must not run after a failure."""

    def _assert_stops_after_step_3(self, close_all):
        result, handle, code = self.run_checks(close_all)
        handle.assert_not_called()
        self.assertNotIn("finished_job_not_consumed", result["checks"])
        self.assertNotIn("ask_not_called_after_finish", result["checks"])
        self.assertIs(result["passed"], False)
        self.assertEqual(code, 2)
        self.assertEqual(json.loads(self.result_path.read_text(encoding="utf-8"))["passed"], False)

    def test_p2_1_window_hidden_by_close_all_windows_stops_before_step_4(self):
        self._assert_stops_after_step_3(_hide_window_close_all)

    def test_p2_1_helper_stopped_by_close_all_windows_stops_before_step_4(self):
        self._assert_stops_after_step_3(_stop_helper_close_all)

    def test_p2_1_docstring_says_passed_field_is_result_of_record(self):
        doc = " ".join(self.module.__doc__.split())
        self.assertRegex(doc, r"`?passed`?[^.]*result of record")


class SafetyOrderDocstringTest(SmokeScriptTestCase):
    """Opus r2 P2-2: the Safety order paragraph names every precondition the code checks."""

    def _safety_paragraph(self):
        paragraphs = [" ".join(p.split()) for p in self.module.__doc__.split("\n\n")]
        found = [p for p in paragraphs if p.startswith("Safety order")]
        self.assertEqual(len(found), 1)
        return found[0]

    def test_p2_2_safety_order_names_all_five_precondition_keys(self):
        paragraph = self._safety_paragraph()
        for key in sorted(PRECONDITION_KEYS):
            self.assertIn(f"`{key}`", paragraph)

    def test_p2_2_each_of_the_five_preconditions_stops_the_run_before_any_real_close(self):
        """Codex P1 (close guard fix): each precondition must gate the real close.

        Make one precondition false at a time (the other four true) and check that the script
        stops before window.close(), the File > Exit action and closeAllWindows(). A source regex
        cannot see a key that was dropped from the gate; this test can.
        """
        failing = {
            "guard_module_from_repo": {"module_from_repo": False},
            "helper_running_at_start": {"helper_running": False},
            "guard_installed": {"guard_installed": False},
            "window_visible_before_close": {"window_visible": False},
            "synthetic_close_consumed": {"event_filter": PassThroughFilter()},
        }
        self.assertEqual(set(failing), PRECONDITION_KEYS)
        for key, options in failing.items():
            with self.subTest(precondition=key):
                self.slicer.app.exit.reset_mock()
                self.module.state["finished"] = False  # _finish runs once per module load
                self.module.ask_calls.clear()
                self.window.close_calls = 0
                self.window.exit_triggers = 0
                result, handle, code = self.run_checks(_one_ask_close_all, **options)
                self.assertEqual(self.window.close_calls, 0)
                self.assertEqual(self.window.exit_triggers, 0)
                self.assertEqual(self.close_all_calls, 0)
                handle.assert_not_called()
                # A clean stop at the gate, not an exception caught later (a missing filter would raise).
                self.assertNotIn("error", result)
                self.assertIs(result["passed"], False)
                self.assertEqual(code, 2)


class CloseAllWindowsAskCountTest(SmokeScriptTestCase):
    """Opus r2 P2-3: close_all_windows_refused needs exactly one ask, like the other paths."""

    def test_p2_3_two_asks_from_close_all_windows_is_not_refused(self):
        result, _handle, _code = self.run_checks(_two_asks_close_all)
        self.assertIs(result["checks"]["close_all_windows_refused"], False)

    def test_p2_3_exactly_one_ask_with_window_visible_passes(self):
        result, handle, code = self.run_checks(_one_ask_close_all)
        checks = result["checks"]
        self.assertIs(checks["close_all_windows_refused"], True)
        for name in self.module.REQUIRED_CHECKS:
            self.assertIs(checks.get(name), True, name)
        handle.assert_called_once()
        self.assertIs(result["passed"], True)
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
