#!/usr/bin/env python3
"""Package A: a locked selection must still show what is ticked (fake ``qt``, no Slicer application)."""

from __future__ import annotations

import re
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from shared import ui_algorithm_picker
from shared.ui_styles import SLICERDSPACE_DARK_UI_STYLE, SLICERDSPACE_UI_STYLE
from test_b2_fix_round import widget_methods
from test_ui_run_history import AlgorithmRowTest, fake_qt

SELECTOR = "QCheckBox::indicator:checked:disabled"
DISABLED = "QCheckBox::indicator:disabled"
CHECKED = "QCheckBox::indicator:checked"


def _rules(style: str) -> list[tuple[list[str], str]]:
    return [
        ([part.strip() for part in selectors.split(",")], body)
        for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", style)
    ]


def _index(rules, selector: str) -> int:
    for position, (selectors, _body) in enumerate(rules):
        if selector in selectors:
            return position
    return -1


class LockedTickStyleTest(unittest.TestCase):
    def test_checked_and_disabled_checkbox_keeps_a_visible_tick_in_both_themes(self) -> None:
        for name, style in (("light", SLICERDSPACE_UI_STYLE), ("dark", SLICERDSPACE_DARK_UI_STYLE)):
            with self.subTest(theme=name):
                rules = _rules(style)
                locked = _index(rules, SELECTOR)
                self.assertNotEqual(locked, -1, "no rule for a checked and disabled box")
                self.assertGreater(locked, _index(rules, DISABLED))
                self.assertGreater(locked, _index(rules, CHECKED))
                body = rules[locked][1]
                self.assertRegex(body, r'image:\s*url\("[^"]+"\)')
                self.assertNotIn("__CHECKMARK_ICON_URL__", body)
                self.assertRegex(body, r"background:\s*#[0-9A-Fa-f]{6}")
                self.assertRegex(body, r"border:\s*1px solid #[0-9A-Fa-f]{6}")

    def test_enabled_checkbox_look_is_unchanged(self) -> None:
        light = dict((tuple(s), b) for s, b in _rules(SLICERDSPACE_UI_STYLE))
        self.assertIn("background: #335B70;", light[(CHECKED,)])
        dark = dict((tuple(s), b) for s, b in _rules(SLICERDSPACE_DARK_UI_STYLE))
        self.assertIn("background: #3B7390;", dark[(CHECKED,)])


class LockedSelectionControlsTest(unittest.TestCase):
    def _owner(self):
        qt = fake_qt()
        owner, slicer = AlgorithmRowTest._build_full_picker_owner(object(), qt)
        owner._algorithmButtons = {}
        owner._pipelineActive = False
        owner._selectedSetupActive = False
        owner.ui.unifiedTorchDeviceComboBox = mock.MagicMock(name="device")
        cls = widget_methods("_set_selection_controls_enabled")
        owner.set_enabled = cls._set_selection_controls_enabled.__get__(owner)
        return qt, slicer, owner

    def test_run_lock_disables_anatomical_priors_row_and_never_unticks_targets(self) -> None:
        _qt, _slicer, owner = self._owner()
        _label, _label_button, checkbox, _summary, arrow = owner._anatomicalPriorsRow
        target_boxes = [
            getattr(owner.ui, name) for name in dir(owner.ui)
            if name.startswith("target") and name.endswith("CheckBox")
        ]
        self.assertTrue(target_boxes)
        for box in target_boxes:
            box.setChecked.reset_mock()
        checkbox.setChecked.reset_mock()

        owner.set_enabled(False)

        self.assertIs(checkbox.enabled, False)
        self.assertIs(arrow.enabled, False)
        for box in target_boxes:
            self.assertIs(box.enabled, False)
            box.setChecked.assert_not_called()
        checkbox.setChecked.assert_not_called()

        owner.set_enabled(True)

        self.assertIs(checkbox.enabled, True)
        self.assertIs(arrow.enabled, True)

    def test_run_lock_tolerates_a_panel_without_the_anatomical_row(self) -> None:
        _qt, _slicer, owner = self._owner()
        del owner._anatomicalPriorsRow
        owner.set_enabled(False)
        owner._anatomicalPriorsRow = None
        owner.set_enabled(True)

    def test_anatomical_click_while_a_job_runs_changes_nothing_and_restores_the_box(self) -> None:
        qt, slicer, owner = self._owner()
        checkbox = owner._anatomicalPriorsRow[2]
        node = owner._parameterNode
        on_clicked = checkbox.clicked.connect.call_args.args[0]
        owner._pipelineActive = True
        checkbox.setChecked.reset_mock()
        with mock.patch.dict(sys.modules, {"qt": qt, "slicer": slicer}):
            on_clicked(True)
        self.assertFalse(any((node.targetSurface, node.targetSegmentation, node.targetParcellation)))
        self.assertFalse(checkbox.checked)
        slicer.util.NodeModify.assert_not_called()

        owner._pipelineActive = False
        owner._selectedSetupActive = True
        with mock.patch.dict(sys.modules, {"qt": qt, "slicer": slicer}):
            on_clicked(True)
        self.assertFalse(any((node.targetSurface, node.targetSegmentation, node.targetParcellation)))

        owner._selectedSetupActive = False
        with mock.patch.dict(sys.modules, {"qt": qt, "slicer": slicer}):
            on_clicked(True)
        self.assertTrue(node.targetSurface)


if __name__ == "__main__":
    unittest.main()
