#!/usr/bin/env python3
"""Real-Slicer smoke test for shared/ui_close_guard.py.

Run by hand (the SlicerDSpace module does not need to be installed):

    Slicer --disable-settings --no-splash --disable-cli-modules \
        --python-script Testing/Python/slicer_close_guard_smoke.py

A real QProcess ("sleep 60") stands in for a running DSpace job. The default ask is replaced by a
recording function that answers "keep running", so no modal dialog opens. The result is written as
JSON to $SLICERDSPACE_SMOKE_RESULT (default: <tempdir>/slicer_close_guard_smoke.json).
Exit code: 0 all required checks passed, 2 a check failed or could not run, 3 watchdog timeout.
The JSON `passed` field is the result of record: the process exit code can be changed by a posted Quit.

Safety order: a real close is attempted only after (a) the helper job is running
(`helper_running_at_start`), (b) the guard module comes from this repository
(`guard_module_from_repo`), (c) the close guard is installed on the owner (`guard_installed`),
(d) the main window is visible (`window_visible_before_close`), and (e) a synthetic Close event sent
straight to the installed filter was consumed (`synthetic_close_consumed`). If one of these fails the
script stops with code 2 before any real close.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["SLICERDSPACE_CLOSE_GUARD"] = "1"

import qt
import slicer

from shared import ui_close_guard

WATCHDOG_MS = 30000
WINDOW_WAIT_MS = 250
RESULT_PATH = Path(os.environ.get("SLICERDSPACE_SMOKE_RESULT") or Path(tempfile.gettempdir()) / "slicer_close_guard_smoke.json")
REQUIRED_CHECKS = (
    "guard_module_from_repo",
    "helper_running_at_start",
    "guard_installed",
    "window_visible_before_close",
    "synthetic_close_consumed",
    "close_refused",
    "window_visible_after_close",
    "ask_called_once_for_close",
    "helper_running_after_close",
    "file_exit_action",
    "helper_running_after_file_exit",
    "close_all_windows_refused",
    "helper_running_after_close_all_windows",
    "finished_job_not_consumed",
    "ask_not_called_after_finish",
)

ask_calls: list[tuple[list, str]] = []
state = {"finished": False}


def _recording_ask(owner, jobs, action) -> bool:
    ask_calls.append((list(jobs), action))
    return True  # keep running; never opens a dialog


def _running(process) -> bool:
    try:
        return process.state() != qt.QProcess.NotRunning
    except Exception:
        return False


def _start_helper_process() -> "qt.QProcess":
    process = qt.QProcess()
    if sys.platform == "win32":
        process.start("ping", ["-n", "60", "127.0.0.1"])
    else:
        process.start("sleep", ["60"])
    process.waitForStarted(5000)
    return process


def _kill(process) -> None:
    try:
        if process is not None and _running(process):
            process.kill()
            process.waitForFinished(5000)
    except Exception:
        pass


def _passed(result: dict) -> bool:
    checks = result["checks"]
    return "error" not in result and all(checks.get(name) is True for name in REQUIRED_CHECKS)


def _finish(owner, process, result: dict, code: int | None = None) -> None:
    if state["finished"]:
        return
    state["finished"] = True
    _kill(process)
    try:
        ui_close_guard.uninstall_close_guard(owner)
    except Exception:
        pass
    result["passed"] = _passed(result)
    if code is None:
        code = 0 if result["passed"] else 2
    result["exit_code"] = code
    try:
        RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    except Exception as exc:
        result["write_error"] = str(exc)
    print(json.dumps(result, indent=2))
    slicer.app.exit(code)


def _run_checks(owner, process, result: dict) -> None:
    window = slicer.util.mainWindow()
    checks = result["checks"]
    try:
        # Preconditions. Without them a real close could succeed, so stop before any close.
        checks["helper_running_at_start"] = _running(process)
        checks["guard_installed"] = getattr(owner, "_closeGuardFilter", None) is not None
        checks["window_visible_before_close"] = bool(window.isVisible())
        if not (checks["guard_module_from_repo"] and checks["helper_running_at_start"]
                and checks["guard_installed"] and checks["window_visible_before_close"]):
            _finish(owner, process, result)
            return

        # (0) A synthetic Close event sent straight to the installed filter is consumed.
        before = len(ask_calls)
        synthetic = qt.QCloseEvent()
        consumed = owner._closeGuardFilter.eventFilter(window, synthetic)
        checks["synthetic_close_consumed"] = (consumed is True and synthetic.isAccepted() is False
                                              and len(ask_calls) - before == 1)
        if not checks["synthetic_close_consumed"]:
            _finish(owner, process, result)
            return

        # (1) close() is refused, the window stays, the question is asked once.
        before = len(ask_calls)
        closed = window.close()
        checks["close_refused"] = closed is False
        checks["window_visible_after_close"] = bool(window.isVisible())
        checks["ask_called_once_for_close"] = len(ask_calls) - before == 1
        checks["helper_running_after_close"] = _running(process)
        if not (checks["close_refused"] and checks["window_visible_after_close"] and checks["helper_running_after_close"]):
            _finish(owner, process, result)
            return

        # (2) File > Exit goes through the main-window close and is refused the same way.
        action = window.findChild(qt.QAction, "FileExitAction")
        if action is None:
            checks["file_exit_action"] = "missing"
        else:
            before = len(ask_calls)
            action.trigger()
            checks["file_exit_action"] = bool(window.isVisible()) and len(ask_calls) - before == 1
        checks["helper_running_after_file_exit"] = _running(process)
        if checks["file_exit_action"] is False or not checks["helper_running_after_file_exit"]:
            _finish(owner, process, result)
            return

        # (3) The application-level path (what a Dock quit or a logout uses in Qt 5) is refused too.
        before = len(ask_calls)
        # PythonQt exposes closeAllWindows() as a slot of the application instance; it returns nothing.
        slicer.app.closeAllWindows()
        checks["close_all_windows_refused"] = bool(window.isVisible()) and len(ask_calls) - before == 1
        checks["helper_running_after_close_all_windows"] = _running(process)
        if checks["close_all_windows_refused"] is False or not checks["helper_running_after_close_all_windows"]:
            _finish(owner, process, result)
            return

        # (4) After the job ended the guard lets the event through and does not ask.
        process.kill()
        process.waitForFinished(5000)
        before = len(ask_calls)
        calls = {"ignore": 0, "accept": 0}
        event = types.SimpleNamespace(
            ignore=lambda: calls.__setitem__("ignore", calls["ignore"] + 1),
            accept=lambda: calls.__setitem__("accept", calls["accept"] + 1),
        )
        consumed = ui_close_guard.handle_close_event(owner, event)
        checks["finished_job_not_consumed"] = consumed is False and calls == {"ignore": 0, "accept": 0}
        checks["ask_not_called_after_finish"] = len(ask_calls) == before
    except Exception as exc:
        result["error"] = repr(exc)
    _finish(owner, process, result)


def main() -> None:
    guard_file = Path(ui_close_guard.__file__).resolve()
    result: dict = {
        "slicer_version": slicer.app.applicationVersion,
        "qt_version": qt.qVersion(),
        "platform": sys.platform,
        "guard_module_file": guard_file.as_posix(),
        "checks": {"guard_module_from_repo": REPO_ROOT in guard_file.parents},
        "passed": False,
    }
    process = _start_helper_process()
    owner = types.SimpleNamespace(_backgroundProcess=process, _backgroundRunContext={"run_name": "close_guard_smoke"})
    ui_close_guard._ask_with_message_box = _recording_ask
    result["install_returned"] = bool(ui_close_guard.install_close_guard(owner))

    def watchdog() -> None:
        if state["finished"]:
            return
        result["error"] = "watchdog timeout"
        _finish(owner, process, result, 3)

    def start_when_window_is_visible() -> None:
        if state["finished"]:
            return
        window = slicer.util.mainWindow()
        if window is not None and window.isVisible():
            _run_checks(owner, process, result)
        else:
            qt.QTimer.singleShot(WINDOW_WAIT_MS, start_when_window_is_visible)

    qt.QTimer.singleShot(WATCHDOG_MS, watchdog)
    qt.QTimer.singleShot(WINDOW_WAIT_MS, start_when_window_is_visible)


main()
