"""Batch 2 add-on popup-apply: a click outside the algorithm picker popup applies, like the Apply button.

The picker (with the settings behind each "..." button) is one ``Qt.Popup`` dialog with Apply, Cancel and an X. Qt closes
a popup on an outside click by calling ``close()``, and ``QDialog::closeEvent`` turns that into ``reject()``. The code
therefore filters the dialog's Close event: it runs the same validate-and-accept function as the Apply button, and it
always ignores the close event (so ``closeEvent`` cannot reject after an accept, and a refused apply keeps the popup open
with the error visible). Cancel, the X and Esc call ``reject()``, which sends no Close event, so they still discard.
"""

from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from shared import ui_algorithm_picker
from test_ui_run_history import AlgorithmRowTest, by_name, fake_qt

CLOSE = 19
SHOW = 17


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


class PopupApplyTest(unittest.TestCase):
    def setUp(self):
        self.qt = fake_qt()
        self.qt.QObject = _FakeQObject
        self.qt.QEvent.Close = CLOSE
        helper = AlgorithmRowTest("test_summary_text_aligns_with_a_plain_checkbox_and_rows_are_vertically_centred")
        self.owner, self.slicer = helper._build_full_picker_owner(self.qt)
        self.node = self.owner._parameterNode
        self.dialog = None

    def _tick(self, chosen):
        for widget in self.qt.created:
            if widget.kind == "QCheckBox" and str(getattr(widget, "objectName", "")).startswith("algorithm_"):
                widget.checked = widget.objectName.removeprefix("algorithm_") in chosen

    def _callback(self, name, signal):
        return next(c.args[1] for c in by_name(self.qt, name).connect.call_args_list if c.args[0] == signal)

    def _run(self, family, scenario, *, chosen=("ukf",)):
        """Open the picker; ``scenario(filter, dialog)`` plays what the user does while ``exec_`` runs."""

        original = self.qt.QDialog.side_effect

        def make_dialog(*args, **kwargs):
            dialog = original(*args, **kwargs)
            self.dialog = dialog

            def fake_exec():
                self._tick(chosen)
                flt = dialog.installEventFilter.call_args.args[0]
                scenario(flt, dialog)
                return self.qt.QDialog.Accepted if dialog.accept.called else 0

            dialog.exec_.side_effect = fake_exec
            return dialog

        self.qt.QDialog.side_effect = make_dialog
        with mock.patch.dict(sys.modules, {"qt": self.qt, "slicer": self.slicer}):
            return ui_algorithm_picker.choose_algorithms(self.owner, family)

    def _spy_commit(self):
        return mock.patch.object(ui_algorithm_picker, "_apply_family_selection_to_node",
                                 wraps=ui_algorithm_picker._apply_family_selection_to_node)

    def test_b2_outside_click_commits_staged_values_once(self):
        events = []

        def scenario(flt, dialog):
            event = _Event()
            events.append((flt.eventFilter(dialog, event), event))
            # A second close event after the commit must not commit or accept again.
            second = _Event()
            events.append((flt.eventFilter(dialog, second), second))

        with self._spy_commit() as commit:
            accepted = self._run("tractography", scenario)
        self.assertTrue(accepted)
        commit.assert_called_once()
        self.assertEqual(self.node.orchestrationTractographyBackend, "ukf")
        self.assertTrue(self.node.targetTractography)
        self.dialog.accept.assert_called_once_with()
        self.dialog.reject.assert_not_called()
        # The first close event is consumed and ignored, so closeEvent cannot reject after the accept.
        self.assertEqual((events[0][0], events[0][1].ignored), (True, 1))

    def test_b2_outside_click_runs_the_same_function_as_apply(self):
        """Both routes end in one accept; the only commit is the code after ``exec_`` returns."""

        def by_apply(flt, dialog):
            self._callback("algorithmPickerApplyButton", "clicked()")()

        def by_outside(flt, dialog):
            flt.eventFilter(dialog, _Event())

        results = []
        for scenario in (by_apply, by_outside):
            self.node.orchestrationTractographyBackend = "csd_act"
            with self._spy_commit() as commit:
                accepted = self._run("tractography", scenario)
            results.append((accepted, commit.call_count, self.node.orchestrationTractographyBackend,
                            self.dialog.accept.call_count, self.dialog.reject.call_count))
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0], (True, 1, "ukf", 1, 0))

    def test_b2_cancel_then_close_event_does_not_commit(self):
        seen = []

        def scenario(flt, dialog):
            self._callback("algorithmPickerCancelButton", "clicked()")()
            event = _Event()
            seen.append((flt.eventFilter(dialog, event), event))

        with self._spy_commit() as commit:
            accepted = self._run("tractography", scenario)
        self.assertFalse(accepted)
        commit.assert_not_called()
        self.assertEqual(self.node.orchestrationTractographyBackend, "csd_act")
        self.dialog.accept.assert_not_called()
        self.dialog.reject.assert_called_once_with()
        self.assertEqual((seen[0][0], seen[0][1].ignored), (False, 0))

    def test_b2_x_button_discards(self):
        def scenario(flt, dialog):
            self._callback("algorithmPickerCloseButton", "clicked(bool)")(False)
            flt.eventFilter(dialog, _Event())

        with self._spy_commit() as commit:
            accepted = self._run("tractography", scenario)
        self.assertFalse(accepted)
        commit.assert_not_called()
        self.dialog.accept.assert_not_called()
        self.assertEqual(self.node.orchestrationTractographyBackend, "csd_act")

    def test_b2_esc_discards(self):
        """Esc makes ``QDialog`` call ``reject()``: it emits ``finished(0)`` and sends no Close event."""

        def scenario(flt, dialog):
            dialog.finished.connect.call_args.args[0](0)
            flt.eventFilter(dialog, _Event())  # a close event that follows must not commit

        with self._spy_commit() as commit:
            accepted = self._run("tractography", scenario)
        self.assertFalse(accepted)
        commit.assert_not_called()
        self.dialog.accept.assert_not_called()
        self.assertEqual(self.node.orchestrationTractographyBackend, "csd_act")

    def test_b2_refused_apply_and_refused_outside_click_behave_alike(self):
        """A staged value that Apply refuses is refused for an outside click too, and the popup stays open."""

        problems = {"raise": True}

        def staged_ukf(layout, node, zh):
            def read():
                if problems["raise"]:
                    raise ValueError("UKF value out of range")
                return "{}"
            return read

        seen = {}

        def scenario(flt, dialog):
            note = by_name(self.qt, "algorithmPickerNote")
            event = _Event()
            seen["outside"] = (flt.eventFilter(dialog, event), event.ignored, dialog.accept.call_count)
            seen["outside_note"] = (note.setText.call_args.args[0], note.show.called)
            note.reset_mock()
            self._callback("algorithmPickerApplyButton", "clicked()")()
            seen["apply"] = (dialog.accept.call_count, note.setText.call_args.args[0], note.show.called)
            problems["raise"] = False
            event = _Event()
            seen["fixed"] = (flt.eventFilter(dialog, event), event.ignored, dialog.accept.call_count)

        with mock.patch("shared.ui_ukf.add_staged_ukf_fields", staged_ukf), self._spy_commit() as commit:
            accepted = self._run("tractography", scenario)
        # Refused: close consumed and ignored (the popup stays open), nothing accepted, same message as Apply.
        self.assertEqual(seen["outside"], (True, 1, 0))
        self.assertEqual(seen["outside_note"], ("UKF value out of range", True))
        self.assertEqual(seen["apply"], (0, "UKF value out of range", True))
        # After the value is fixed, the outside click accepts, once.
        self.assertEqual(seen["fixed"], (True, 1, 1))
        self.assertTrue(accepted)
        commit.assert_called_once()

    def test_b2_refused_outside_click_commits_nothing(self):
        def staged_ukf(layout, node, zh):
            def read():
                raise ValueError("bad")
            return read

        def scenario(flt, dialog):
            flt.eventFilter(dialog, _Event())

        with mock.patch("shared.ui_ukf.add_staged_ukf_fields", staged_ukf), self._spy_commit() as commit:
            accepted = self._run("tractography", scenario)
        self.assertFalse(accepted)
        commit.assert_not_called()
        self.assertEqual(self.node.orchestrationTractographyBackend, "csd_act")

    def test_b2_filter_ignores_other_events_and_is_removed_afterwards(self):
        seen = []

        def scenario(flt, dialog):
            event = _Event(SHOW)
            seen.append((flt.eventFilter(dialog, event), event.ignored))

        self._run("tractography", scenario)
        self.assertEqual(seen, [(False, 0)])
        self.dialog.installEventFilter.assert_called_once()
        self.dialog.removeEventFilter.assert_called_once_with(self.dialog.installEventFilter.call_args.args[0])

    def test_b2_every_family_gets_the_outside_click_rule(self):
        """The anatomical-priors popup and the family pickers share this function and the same rule."""

        for family in ("preprocessing", "microstructure", "tractography", "bundles", "connectome"):
            with self.subTest(family=family):
                def scenario(flt, dialog):
                    flt.eventFilter(dialog, _Event())

                self.assertTrue(self._run(family, scenario, chosen=()))
                self.dialog.accept.assert_called_once_with()

        def anatomical(flt, dialog):
            flt.eventFilter(dialog, _Event())

        self.assertTrue(self._run("anatomical_priors", anatomical, chosen=("targetSurface",)))
        self.assertTrue(self.node.targetSurface)


if __name__ == "__main__":
    unittest.main()
