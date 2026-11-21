#!/usr/bin/env python3
"""Tests for shared/ui_close_guard.py (ask before closing Slicer while a job runs)."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import ui_close_guard
from shared.ui_close_guard import (
    close_guard_enabled,
    guard_message,
    handle_close_event,
    install_close_guard,
    may_close,
    running_jobs,
    uninstall_close_guard,
)

RUNNING = "shared.ui_close_guard.qprocess_is_running"


def _owner(**kwargs):
    return types.SimpleNamespace(**kwargs)


def _kinds(jobs):
    return [job["kind"] for job in jobs]


class RunningJobsTest(unittest.TestCase):
    def test_bare_owner_has_no_jobs(self):
        self.assertEqual(running_jobs(object()), [])
        self.assertEqual(running_jobs(_owner()), [])

    def test_run_job_with_name(self):
        owner = _owner(_backgroundProcess=object(), _backgroundRunContext={"run_name": "r1"})
        with mock.patch(RUNNING, return_value=True):
            jobs = running_jobs(owner)
        self.assertEqual(jobs, [{"kind": "run", "run_name": "r1"}])

    def test_run_job_without_name(self):
        owner = _owner(_backgroundProcess=object())
        with mock.patch(RUNNING, return_value=True):
            jobs = running_jobs(owner)
        self.assertEqual(_kinds(jobs), ["run"])
        self.assertNotIn("run_name", jobs[0])

    def test_finished_process_is_ignored(self):
        owner = _owner(_backgroundProcess=object(), _backgroundRunContext={"run_name": "r1"})
        with mock.patch(RUNNING, return_value=False):
            self.assertEqual(running_jobs(owner), [])

    def test_setup_job(self):
        self.assertEqual(_kinds(running_jobs(_owner(_setupJob=object()))), ["setup"])
        self.assertEqual(running_jobs(_owner(_setupJob=None)), [])

    def test_import_job_both_triggers(self):
        self.assertEqual(_kinds(running_jobs(_owner(_inputImportPending=True))), ["import"])
        self.assertEqual(_kinds(running_jobs(_owner(_importProcess=object()))), ["import"])
        self.assertEqual(_kinds(running_jobs(_owner(_inputImportPending=True, _importProcess=object()))), ["import"])
        self.assertEqual(running_jobs(_owner(_inputImportPending=False, _importProcess=None)), [])

    def test_update_job_only_when_not_completed(self):
        self.assertEqual(_kinds(running_jobs(_owner(_codeUpdateJob={"completed": False}))), ["update"])
        self.assertEqual(_kinds(running_jobs(_owner(_codeUpdateJob={}))), ["update"])
        self.assertEqual(running_jobs(_owner(_codeUpdateJob={"completed": True})), [])
        self.assertEqual(running_jobs(_owner(_codeUpdateJob=None)), [])


class CloseGuardEnabledTest(unittest.TestCase):
    def test_env_off_values(self):
        for value in ("0", "false", "off", "no", "FALSE", "Off", "No"):
            with self.subTest(value=value):
                self.assertFalse(close_guard_enabled({"SLICERDSPACE_CLOSE_GUARD": value}, _owner(testingEnabled=lambda: False)))

    def test_env_on_values_win_over_testing(self):
        app = _owner(testingEnabled=lambda: True)
        for value in ("1", "true", "on", "yes", "TRUE", "On"):
            with self.subTest(value=value):
                self.assertTrue(close_guard_enabled({"SLICERDSPACE_CLOSE_GUARD": value}, app))

    def test_unset_and_testing_is_off(self):
        self.assertFalse(close_guard_enabled({}, _owner(testingEnabled=lambda: True)))

    def test_unset_and_not_testing_is_on(self):
        self.assertTrue(close_guard_enabled({}, _owner(testingEnabled=lambda: False)))

    def test_other_value_falls_back_to_default(self):
        self.assertTrue(close_guard_enabled({"SLICERDSPACE_CLOSE_GUARD": "maybe"}, _owner(testingEnabled=lambda: False)))
        self.assertFalse(close_guard_enabled({"SLICERDSPACE_CLOSE_GUARD": "maybe"}, _owner(testingEnabled=lambda: True)))

    def test_testing_check_that_raises_counts_as_not_testing(self):
        def boom():
            raise RuntimeError("no app")

        self.assertTrue(close_guard_enabled({}, _owner(testingEnabled=boom)))


class MayCloseTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch("shared.ui_close_guard.close_guard_enabled", return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_no_job_allows_without_asking(self):
        ask = mock.Mock()
        self.assertTrue(may_close(_owner(), ask=ask))
        ask.assert_not_called()

    def test_disabled_guard_allows_without_asking(self):
        ask = mock.Mock()
        with mock.patch("shared.ui_close_guard.close_guard_enabled", return_value=False):
            self.assertTrue(may_close(_owner(_setupJob=object()), ask=ask))
        ask.assert_not_called()

    def test_keep_running_blocks_close(self):
        ask = mock.Mock(return_value=True)
        self.assertFalse(may_close(_owner(_setupJob=object()), ask=ask))
        ask.assert_called_once()

    def test_leave_allows_close(self):
        ask = mock.Mock(return_value=False)
        self.assertTrue(may_close(_owner(_setupJob=object()), ask=ask))

    def test_ask_exception_fails_open(self):
        ask = mock.Mock(side_effect=RuntimeError("dialog failed"))
        owner = _owner(_setupJob=object())
        self.assertTrue(may_close(owner, ask=ask))
        self.assertFalse(owner._closeGuardAsking)

    def test_reentrant_call_returns_false_without_second_dialog(self):
        owner = _owner(_setupJob=object())
        inner_results = []

        def ask(o, jobs, action):
            inner_results.append(may_close(o, ask=ask_inner))
            return False

        ask_inner = mock.Mock(return_value=False)
        outer = may_close(owner, ask=ask)
        self.assertEqual(inner_results, [False])
        ask_inner.assert_not_called()
        self.assertTrue(outer)
        self.assertFalse(owner._closeGuardAsking)

    def test_flag_reset_allows_a_later_question(self):
        owner = _owner(_setupJob=object())
        ask = mock.Mock(return_value=True)
        self.assertFalse(may_close(owner, ask=ask))
        self.assertFalse(may_close(owner, ask=ask))
        self.assertEqual(ask.call_count, 2)

    def test_action_and_jobs_are_passed_to_ask(self):
        ask = mock.Mock(return_value=True)
        owner = _owner(_setupJob=object())
        may_close(owner, action="restart", ask=ask)
        ask.assert_called_once_with(owner, [{"kind": "setup"}], "restart")
        ask.reset_mock()
        may_close(owner, ask=ask)
        self.assertEqual(ask.call_args[0][2], "quit")


class HandleCloseEventTest(unittest.TestCase):
    def test_consumed_event_is_ignored(self):
        event = mock.Mock()
        with mock.patch("shared.ui_close_guard.may_close", return_value=False):
            self.assertTrue(handle_close_event(_owner(), event))
        event.ignore.assert_called_once_with()

    def test_allowed_event_is_untouched(self):
        event = mock.Mock()
        with mock.patch("shared.ui_close_guard.may_close", return_value=True):
            self.assertFalse(handle_close_event(_owner(), event))
        event.ignore.assert_not_called()
        event.accept.assert_not_called()

    def test_exception_returns_false(self):
        event = mock.Mock()
        with mock.patch("shared.ui_close_guard.may_close", side_effect=RuntimeError("x")):
            self.assertFalse(handle_close_event(_owner(), event))
        event.ignore.assert_not_called()


class GuardMessageTest(unittest.TestCase):
    def test_english_quit_with_run(self):
        msg = guard_message([{"kind": "run", "run_name": "r1"}], "quit", False)
        self.assertEqual(msg["title"], "SlicerDSpace is still working")
        self.assertIn("A DSpace job is running: run 'r1'.", msg["text"])
        self.assertIn("If Slicer closes now, the job stops.", msg["text"])
        self.assertIn("Finished stages are kept. The stage in progress starts again from its beginning next time.", msg["text"])
        self.assertEqual(msg["keep"], "Keep Running")
        self.assertEqual(msg["leave"], "Stop and Quit")

    def test_english_restart_without_run(self):
        msg = guard_message([{"kind": "setup"}, {"kind": "import"}, {"kind": "update"}], "restart", False)
        self.assertIn("environment setup, input import, code update", msg["text"])
        self.assertIn("If Slicer restarts now, the job stops.", msg["text"])
        self.assertNotIn("Finished stages", msg["text"])
        self.assertEqual(msg["leave"], "Stop and Restart")

    def test_english_run_without_name(self):
        msg = guard_message([{"kind": "run"}], "quit", False)
        self.assertIn("pipeline run", msg["text"])

    def test_chinese_quit_with_run(self):
        msg = guard_message([{"kind": "run", "run_name": "r1"}], "quit", True)
        self.assertEqual(msg["title"], "SlicerDSpace 仍在运行")
        self.assertIn("DSpace 任务正在运行：运行“r1”。", msg["text"])
        self.assertIn("现在关闭 Slicer 会停止该任务。", msg["text"])
        self.assertIn("已完成的阶段会保留，正在进行的阶段下次要从头重跑。", msg["text"])
        self.assertEqual(msg["keep"], "继续运行")
        self.assertEqual(msg["leave"], "停止并退出")

    def test_chinese_restart_without_run(self):
        msg = guard_message([{"kind": "setup"}, {"kind": "import"}, {"kind": "update"}, {"kind": "run"}], "restart", True)
        self.assertIn("环境准备、输入导入、代码更新、流程运行", msg["text"])
        self.assertIn("现在重启 Slicer 会停止该任务。", msg["text"])
        self.assertEqual(msg["leave"], "停止并重启")
        msg = guard_message([{"kind": "setup"}], "restart", True)
        self.assertNotIn("已完成的阶段", msg["text"])


class _FakeQt(types.ModuleType):
    """Minimal qt stand-in: QWidget base class, QEvent.Close and a window double."""

    def __init__(self):
        super().__init__("qt")

        class QWidget:
            def __init__(self, *args):
                self.deleted = False

            def deleteLater(self):
                self.deleted = True

        class QEvent:
            Close = 19
            Show = 17

        self.QWidget = QWidget
        self.QEvent = QEvent


class _FakeWindow:
    def __init__(self):
        self.filters = []

    def installEventFilter(self, flt):
        self.filters.append(flt)

    def removeEventFilter(self, flt):
        self.filters.remove(flt)


class _FakeEvent:
    def __init__(self, kind):
        self._kind = kind
        self.ignored = 0

    def type(self):
        return self._kind

    def ignore(self):
        self.ignored += 1


class InstallCloseGuardTest(unittest.TestCase):
    def _modules(self, window):
        fake_qt = _FakeQt()
        fake_slicer = types.ModuleType("slicer")
        fake_slicer.util = types.SimpleNamespace(mainWindow=lambda: window)
        return {"qt": fake_qt, "slicer": fake_slicer}, fake_qt

    def test_install_keeps_strong_reference_and_is_idempotent(self):
        window = _FakeWindow()
        modules, _ = self._modules(window)
        owner = _owner()
        with mock.patch.dict(sys.modules, modules):
            self.assertTrue(install_close_guard(owner))
            first = owner._closeGuardFilter
            self.assertTrue(install_close_guard(owner))
        self.assertIs(owner._closeGuardFilter, first)
        self.assertIs(owner._closeGuardWindow, window)
        self.assertEqual(window.filters, [first])

    def test_filter_consumes_close_event_when_user_keeps_running(self):
        window = _FakeWindow()
        modules, fake_qt = self._modules(window)
        owner = _owner(_setupJob=object())
        with mock.patch.dict(sys.modules, modules), mock.patch(
                "shared.ui_close_guard.close_guard_enabled", return_value=True), mock.patch(
                "shared.ui_close_guard._ask_with_message_box", return_value=True) as ask:
            install_close_guard(owner)
            close_event = _FakeEvent(fake_qt.QEvent.Close)
            self.assertTrue(owner._closeGuardFilter.eventFilter(window, close_event))
            self.assertEqual(close_event.ignored, 1)
            ask.assert_called_once()
            other_event = _FakeEvent(fake_qt.QEvent.Show)
            self.assertFalse(owner._closeGuardFilter.eventFilter(window, other_event))
            self.assertEqual(other_event.ignored, 0)
            ask.assert_called_once()

    def test_filter_lets_close_through_without_a_job(self):
        window = _FakeWindow()
        modules, fake_qt = self._modules(window)
        owner = _owner()
        with mock.patch.dict(sys.modules, modules), mock.patch(
                "shared.ui_close_guard.close_guard_enabled", return_value=True):
            install_close_guard(owner)
            close_event = _FakeEvent(fake_qt.QEvent.Close)
            self.assertFalse(owner._closeGuardFilter.eventFilter(window, close_event))
        self.assertEqual(close_event.ignored, 0)

    def test_no_main_window_returns_false(self):
        modules, _ = self._modules(None)
        owner = _owner()
        with mock.patch.dict(sys.modules, modules):
            self.assertFalse(install_close_guard(owner))
        self.assertIsNone(getattr(owner, "_closeGuardFilter", None))

    def test_install_failure_returns_false(self):
        window = mock.Mock()
        window.installEventFilter.side_effect = RuntimeError("boom")
        modules, _ = self._modules(window)
        with mock.patch.dict(sys.modules, modules):
            self.assertFalse(install_close_guard(_owner()))

    def test_uninstall_removes_filter_and_clears_attributes(self):
        window = _FakeWindow()
        modules, _ = self._modules(window)
        owner = _owner()
        with mock.patch.dict(sys.modules, modules):
            install_close_guard(owner)
            flt = owner._closeGuardFilter
            uninstall_close_guard(owner)
        self.assertEqual(window.filters, [])
        self.assertTrue(flt.deleted)
        self.assertIsNone(owner._closeGuardFilter)
        self.assertIsNone(owner._closeGuardWindow)

    def test_uninstall_never_raises(self):
        uninstall_close_guard(_owner())
        window = mock.Mock()
        window.removeEventFilter.side_effect = RuntimeError("gone")
        owner = _owner(_closeGuardFilter=object(), _closeGuardWindow=window)
        uninstall_close_guard(owner)
        self.assertIsNone(owner._closeGuardFilter)
        self.assertIsNone(owner._closeGuardWindow)


class UpdatePanelRestartTest(unittest.TestCase):
    def test_restart_blocked_when_user_keeps_running(self):
        from shared import ui_update_panel

        with mock.patch("shared.ui_close_guard.may_close", return_value=False) as guard, mock.patch.object(
                ui_update_panel, "_restart_slicer", return_value=True) as restart:
            owner = _owner()
            self.assertFalse(ui_update_panel._restart_slicer_if_allowed(owner))
        restart.assert_not_called()
        guard.assert_called_once_with(owner, action="restart")

    def test_restart_runs_when_allowed(self):
        from shared import ui_update_panel

        with mock.patch("shared.ui_close_guard.may_close", return_value=True), mock.patch.object(
                ui_update_panel, "_restart_slicer", return_value=True) as restart:
            self.assertTrue(ui_update_panel._restart_slicer_if_allowed(_owner()))
        restart.assert_called_once_with()

    def test_restart_runs_when_guard_raises(self):
        from shared import ui_update_panel

        with mock.patch("shared.ui_close_guard.may_close", side_effect=RuntimeError("x")), mock.patch.object(
                ui_update_panel, "_restart_slicer", return_value=True) as restart:
            self.assertTrue(ui_update_panel._restart_slicer_if_allowed(_owner()))
        restart.assert_called_once_with()

    def test_both_call_sites_use_the_helper(self):
        from shared import ui_update_panel

        source = Path(ui_update_panel.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count("_restart_slicer()"), 2)  # the def line and the helper's own call
        self.assertEqual(source.count("        _restart_slicer_if_allowed(owner)"), 2)  # the two call sites


if __name__ == "__main__":
    unittest.main()
