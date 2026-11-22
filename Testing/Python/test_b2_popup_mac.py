"""Batch 2 popup-mac: on macOS a click outside the algorithm picker applies it (the picker no longer runs a modal session).

On macOS ``QDialog.exec_()`` runs a native modal session, and Cocoa then drops every click on the Slicer main window, so
Qt never saw the outside click that should apply the popup. On macOS the picker is therefore shown non-modal and a local
``QEventLoop`` waits for it. Windows and Linux keep ``exec_()``. These tests play the user inside that wait with a fake
loop; the real-Slicer check posts NSEvents (see the lane report).
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path
from unittest import mock

from shared import ui_algorithm_picker
from test_ui_run_history import AlgorithmRowTest, by_name, fake_qt

CLOSE = 19
ACCEPTED = 1
NO_REPLAY = 54
REPO = Path(__file__).resolve().parents[2]


class _FakeQObject:
    """A real base class: ``fake_qt()`` is a MagicMock and cannot be subclassed."""


class _Event:
    def __init__(self, kind=CLOSE):
        self._kind = kind
        self.ignored = 0

    def type(self):
        return self._kind

    def ignore(self):
        self.ignored += 1


class _Signal:
    def __init__(self):
        self.slots = []
        self.disconnected = []

    def connect(self, slot):
        self.slots.append(slot)

    def disconnect(self, slot):
        self.disconnected.append(slot)
        self.slots = [s for s in self.slots if s is not slot]

    def emit(self, *args):
        for slot in list(self.slots):
            slot(*args)


class _FakeTimer:
    instances: list = []

    def __init__(self):
        self.interval = None
        self.running = False
        self.timeout = _Signal()
        _FakeTimer.instances.append(self)

    def setInterval(self, ms):
        self.interval = ms

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    @staticmethod
    def singleShot(_ms, _fn):
        return None


class PopupMacTest(unittest.TestCase):
    def setUp(self):
        self.qt = fake_qt()
        self.qt.QObject = _FakeQObject
        self.qt.QEvent.Close = CLOSE
        self.qt.Qt.WA_NoMouseReplay = NO_REPLAY
        self.qt.QDialog.Rejected = 0
        self.qt.QTimer = _FakeTimer
        _FakeTimer.instances = []
        self.app = mock.MagicMock(name="app")
        self.app.aboutToQuit = _Signal()
        self.qt.QApplication.instance.return_value = self.app
        helper = AlgorithmRowTest("test_summary_text_aligns_with_a_plain_checkbox_and_rows_are_vertically_centred")
        self.owner, self.slicer = helper._build_full_picker_owner(self.qt)
        self.node = self.owner._parameterNode
        self.dialogs = []
        self.loops = []
        test = self

        class _FakeLoop:
            def __init__(self):
                self.exec_calls = 0
                self.quit_calls = 0
                test.loops.append(self)

            def exec_(self):
                self.exec_calls += 1
                test.on_wait(test.dialogs[-1])
                return 0

            def quit(self):
                self.quit_calls += 1

        self.qt.QEventLoop = _FakeLoop
        self.on_wait = lambda dialog: None
        self.on_show = None

    # ---- helpers -------------------------------------------------------------------------------------------------
    def _wire(self, dialog):
        """A dialog whose accept/reject emit ``finished`` and set ``result()``, as QDialog does."""

        state = {"result": 0, "visible": False}
        dialog.finished = _Signal()
        dialog.destroyed = _Signal()

        def done(code):
            state["result"] = code
            state["visible"] = False
            dialog.finished.emit(code)

        def show():
            state["visible"] = True
            if self.on_show is not None:
                self.on_show(dialog)

        dialog.accept.side_effect = lambda: done(ACCEPTED)
        dialog.reject.side_effect = lambda: done(0)
        dialog.result.side_effect = lambda: state["result"]
        dialog.show.side_effect = show
        dialog.isVisible.side_effect = lambda: state["visible"]
        dialog.exec_.side_effect = lambda: (show(), self.on_wait(dialog), state["result"])[-1]
        dialog.state = state
        return dialog

    def _choose(self, family="tractography", *, platform="darwin"):
        if not getattr(self, "_dialog_factory_wired", False):
            original = self.qt.QDialog.side_effect

            def make_dialog(*args, **kwargs):
                dialog = self._wire(original(*args, **kwargs))
                self.dialogs.append(dialog)
                return dialog

            self.qt.QDialog.side_effect = make_dialog
            self._dialog_factory_wired = True
        with (
            mock.patch.dict(sys.modules, {"qt": self.qt, "slicer": self.slicer}),
            mock.patch.object(ui_algorithm_picker, "_host_platform", lambda: platform, create=True),
        ):
            return ui_algorithm_picker.choose_algorithms(self.owner, family)

    def _tick(self, chosen):
        for widget in self.qt.created:
            if widget.kind == "QCheckBox" and str(getattr(widget, "objectName", "")).startswith("algorithm_"):
                widget.checked = widget.objectName.removeprefix("algorithm_") in chosen

    @staticmethod
    def _filter(dialog):
        return dialog.installEventFilter.call_args.args[0]

    def _spy_commit(self):
        return mock.patch.object(ui_algorithm_picker, "_apply_family_selection_to_node",
                                 wraps=ui_algorithm_picker._apply_family_selection_to_node)

    # ---- the macOS wait -------------------------------------------------------------------------------------------
    def test_b2_popup_mac_darwin_waits_non_modally_and_returns_the_dialog_result(self):
        """An outside click (the Close event) applies inside the non-modal wait; the result is the dialog's."""

        def outside_click(dialog):
            self._tick(["ukf"])
            self._filter(dialog).eventFilter(dialog, _Event())

        self.on_wait = outside_click
        with self._spy_commit() as commit:
            accepted = self._choose("tractography")
        dialog = self.dialogs[0]
        self.assertTrue(accepted)
        dialog.exec_.assert_not_called()
        dialog.show.assert_called_once_with()
        self.assertEqual([loop.exec_calls for loop in self.loops], [1])
        self.assertGreaterEqual(self.loops[0].quit_calls, 1)
        commit.assert_called_once()
        self.assertEqual(self.node.orchestrationTractographyBackend, "ukf")
        # Every connection made for the wait is released after it.
        self.assertEqual(dialog.destroyed.slots, [])
        self.assertEqual(self.app.aboutToQuit.slots, [])

    def test_b2_popup_mac_cancel_inside_the_wait_discards(self):
        def cancel(dialog):
            self._tick(["ukf"])
            next(c.args[1] for c in by_name(self.qt, "algorithmPickerCancelButton").connect.call_args_list
                 if c.args[0] == "clicked()")()

        self.on_wait = cancel
        with self._spy_commit() as commit:
            self.assertFalse(self._choose("tractography"))
        commit.assert_not_called()
        self.assertEqual(self.node.orchestrationTractographyBackend, "csd_act")

    def test_b2_popup_mac_other_platforms_keep_exec(self):
        for platform in ("linux", "win32"):
            with self.subTest(platform=platform):
                self.dialogs.clear()
                self.loops.clear()
                self.node.orchestrationTractographyBackend = "csd_act"
                self.on_wait = lambda dialog: (self._tick(["ukf"]), self._filter(dialog).eventFilter(dialog, _Event()))
                self.assertTrue(self._choose("tractography", platform=platform))
                dialog = self.dialogs[0]
                dialog.exec_.assert_called_once_with()
                self.assertEqual(self.loops, [])

    def test_b2_popup_mac_a_dialog_finished_during_show_never_enters_the_loop(self):
        """``QEventLoop.quit()`` before ``exec_()`` is lost; a dialog that finished while it was shown must not wait."""

        self.on_show = lambda dialog: dialog.reject()
        self.on_wait = lambda dialog: self.fail("the loop ran after the dialog had finished")
        self.assertFalse(self._choose("tractography"))
        self.assertEqual([loop.exec_calls for loop in self.loops], [0])

    def test_b2_popup_mac_a_dialog_not_visible_after_show_is_rejected_without_waiting(self):
        def never_visible(dialog):
            dialog.state["visible"] = False

        self.on_show = never_visible
        self.on_wait = lambda dialog: self.fail("the loop ran for an invisible popup")
        self.assertFalse(self._choose("tractography"))
        self.dialogs[0].reject.assert_called_with()
        self.assertEqual([loop.exec_calls for loop in self.loops], [0])

    def test_b2_popup_mac_the_wait_ends_when_the_dialog_is_destroyed_or_the_app_quits(self):
        for source in ("destroyed", "aboutToQuit"):
            with self.subTest(source=source):
                self.dialogs.clear()
                self.loops.clear()

                def vanish(dialog, source=source):
                    self._tick(["ukf"])
                    if source == "destroyed":
                        dialog.result.side_effect = RuntimeError("Trying to call 'result' on a destroyed QDialog")
                        dialog.deleteLater.side_effect = RuntimeError("destroyed")
                        dialog.destroyed.emit(dialog)
                    else:
                        self.app.aboutToQuit.emit()

                self.on_wait = vanish
                with self._spy_commit() as commit:
                    self.assertFalse(self._choose("tractography"))
                commit.assert_not_called()
                self.assertGreaterEqual(self.loops[0].quit_calls, 1)
                self.assertEqual(self.node.orchestrationTractographyBackend, "csd_act")

    def test_b2_popup_mac_a_lost_quit_is_sent_again_from_the_loop(self):
        """Qt's Cocoa dispatcher drops a quit() made inside a nested processEvents(); a timer repeats it."""

        seen = {}

        def accept_then_tick_timer(dialog):
            requit = [t for t in _FakeTimer.instances if t.interval == 100]
            seen["running_during_wait"] = [t.running for t in requit]
            quits = self.loops[0].quit_calls
            requit[0].timeout.emit()  # not finished yet: no quit
            seen["quit_before_finish"] = self.loops[0].quit_calls - quits
            self._tick(["ukf"])
            self._filter(dialog).eventFilter(dialog, _Event())
            quits = self.loops[0].quit_calls
            requit[0].timeout.emit()
            seen["quit_after_finish"] = self.loops[0].quit_calls - quits

        self.on_wait = accept_then_tick_timer
        self.assertTrue(self._choose("tractography"))
        self.assertEqual(seen, {"running_during_wait": [True], "quit_before_finish": 0, "quit_after_finish": 1})
        self.assertEqual([t.running for t in _FakeTimer.instances if t.interval == 100], [False])

    # ---- re-entry, owner cleanup ---------------------------------------------------------------------------------
    def test_b2_popup_mac_a_second_picker_does_not_open_while_one_is_open(self):
        seen = {}

        def reenter(dialog):
            with mock.patch.dict(sys.modules, {"qt": self.qt, "slicer": self.slicer}):
                seen["second"] = ui_algorithm_picker.choose_algorithms(self.owner, "bundles")
            seen["dialogs"] = len(self.dialogs)
            dialog.reject()

        self.on_wait = reenter
        self.assertFalse(self._choose("tractography"))
        self.assertEqual(seen, {"second": False, "dialogs": 1})
        # After the first picker has closed, a picker opens again.
        self.on_wait = lambda dialog: dialog.reject()
        self._choose("bundles")
        self.assertEqual(len(self.dialogs), 2)

    def test_b2_popup_mac_owner_cleanup_closes_the_open_picker_and_discards(self):
        def cleanup(dialog):
            self._tick(["ukf"])
            ui_algorithm_picker.close_algorithm_picker(self.owner)

        self.on_wait = cleanup
        with self._spy_commit() as commit:
            self.assertFalse(self._choose("tractography"))
        commit.assert_not_called()
        self.dialogs[0].reject.assert_called_once_with()
        self.assertEqual(self.node.orchestrationTractographyBackend, "csd_act")
        # With no picker open, the call does nothing.
        ui_algorithm_picker.close_algorithm_picker(self.owner)
        self.dialogs[0].reject.assert_called_once_with()

    def test_b2_popup_mac_module_cleanup_and_exit_close_the_picker(self):
        tree = ast.parse((REPO / "shared" / "module_widget.py").read_text(encoding="utf-8"))
        calls = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in ("cleanup", "exit"):
                calls[node.name] = any(isinstance(c, ast.Call) and getattr(c.func, "id", "") == "_close_algorithm_picker"
                                       for c in ast.walk(node))
        self.assertEqual(calls, {"cleanup": True, "exit": True})

    # ---- refused apply: no click replay to the main window --------------------------------------------------------
    def test_b2_popup_mac_every_close_event_turns_off_the_click_replay(self):
        """After a refused apply, Qt's widget popup code closes the popup on the next outside click and would replay
        that press to the control under the mouse (measured: Run got "pressed"). The Close filter sets
        WA_NoMouseReplay each time, because QApplication clears it before every mouse event."""

        problems = {"raise": True}

        def staged_ukf(layout, node, zh):
            def read():
                if problems["raise"]:
                    raise ValueError("UKF value out of range")
                return "{}"
            return read

        seen = {}

        def refused_then_fixed(dialog):
            self._tick(["ukf"])
            flt = self._filter(dialog)
            event = _Event()
            seen["refused"] = (flt.eventFilter(dialog, event), event.ignored, dialog.accept.call_count)
            seen["attr_after_refused"] = dialog.setAttribute.call_args_list.count(mock.call(NO_REPLAY, True))
            problems["raise"] = False
            flt.eventFilter(dialog, _Event())
            seen["attr_after_fixed"] = dialog.setAttribute.call_args_list.count(mock.call(NO_REPLAY, True))

        self.on_wait = refused_then_fixed
        with mock.patch("shared.ui_ukf.add_staged_ukf_fields", staged_ukf), self._spy_commit() as commit:
            self.assertTrue(self._choose("tractography"))
        self.assertEqual(seen, {"refused": (True, 1, 0), "attr_after_refused": 1, "attr_after_fixed": 2})
        commit.assert_called_once()

    def test_b2_popup_mac_click_replay_is_unchanged_on_other_platforms(self):
        self.on_wait = lambda dialog: (self._tick(["ukf"]), self._filter(dialog).eventFilter(dialog, _Event()))
        self.assertTrue(self._choose("tractography", platform="win32"))
        self.assertNotIn(mock.call(NO_REPLAY, True), self.dialogs[0].setAttribute.call_args_list)


if __name__ == "__main__":
    unittest.main()
