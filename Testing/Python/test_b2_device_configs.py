#!/usr/bin/env python3
"""B2 device-configuration fixes D1..D6 (host RAM warning, TLS trust, small screens, DICOM space, NHDR path, mirrors)."""

from __future__ import annotations

import contextlib
import errno
import io
import os
import re
import shutil
import ssl
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

GIB = 1024 ** 3


class D3SmallScreenTest(unittest.TestCase):
    """D3 (audit F3): the first-run wizard and the environment dialog fit a 1092x574 logical screen."""

    SMALL = (1092, 574)  # 1366x768 at 125 %
    LARGE = (1920, 1080)
    CHROME_ESTIMATE = 180  # header, progress line, footer buttons and margins around the wizard's scroll area (audit estimate)

    @staticmethod
    def _qt(screen_size):
        qt = mock.MagicMock(name="qt")
        geometry = mock.MagicMock(name="availableGeometry")
        geometry.width.return_value = screen_size[0]
        geometry.height.return_value = screen_size[1]
        geometry.left.return_value = 0
        geometry.top.return_value = 0
        screen = mock.MagicMock(name="screen")
        screen.availableGeometry.return_value = geometry
        qt.QApplication.screenAt.return_value = screen
        qt.QApplication.primaryScreen.return_value = screen
        return qt

    def _run_wizard(self, screen_size):
        from shared import ui_onboarding

        qt = self._qt(screen_size)
        slicer = mock.MagicMock(name="slicer")
        owner = mock.MagicMock(name="owner")
        owner._onboardingLanguage = "en"
        owner._read_environment_path_setting.return_value = ""
        with mock.patch.dict(sys.modules, {"qt": qt, "slicer": slicer}):
            ui_onboarding.show_first_run_wizard(owner)
        dialog = qt.QDialog.return_value
        scroll = qt.QScrollArea.return_value
        return dialog, scroll

    @staticmethod
    def _last_arg(widget, method):
        calls = getattr(widget, method).call_args_list
        return calls[-1][0] if calls else None

    def _run_environment_dialog(self, screen_size):
        from shared import ui_environment_dialog as module

        qt = self._qt(screen_size)
        slicer = mock.MagicMock(name="slicer")
        owner = mock.MagicMock(name="owner")
        owner._environmentDialog = None
        owner._onboardingLanguage = "en"
        state = mock.MagicMock(name="state")
        state.missing_items = ()
        with mock.patch.dict(sys.modules, {"qt": qt, "slicer": slicer}), \
                mock.patch.object(module, "environment_dialog_state", return_value=state):
            module.show_environment_results_dialog(
                owner=owner, result=mock.MagicMock(), python_install_target="x",
                on_install_all=lambda: None, on_show_notes=lambda: None, on_install_item=lambda item: None)
        return qt.QDialog.return_value

    def test_b2_d3_wizard_minimum_sizes_fit_a_small_screen(self):
        dialog, scroll = self._run_wizard(self.SMALL)
        (min_width,) = self._last_arg(dialog, "setMinimumWidth")
        (scroll_min,) = self._last_arg(scroll, "setMinimumHeight")
        self.assertLessEqual(min_width, self.SMALL[0])
        self.assertLessEqual(scroll_min + self.CHROME_ESTIMATE, self.SMALL[1])
        self.assertGreater(scroll_min, 0)

    def test_b2_d3_wizard_keeps_today_s_minimum_sizes_on_a_large_screen(self):
        dialog, scroll = self._run_wizard(self.LARGE)
        self.assertEqual(self._last_arg(dialog, "setMinimumWidth"), (760,))
        self.assertEqual(self._last_arg(scroll, "setMinimumHeight"), (430,))

    def test_b2_d3_environment_dialog_size_is_clamped_to_a_small_screen(self):
        dialog = self._run_environment_dialog(self.SMALL)
        width, height = self._last_arg(dialog, "resize")
        self.assertLessEqual(width, self.SMALL[0])
        self.assertLessEqual(height, self.SMALL[1])

    def test_b2_d3_environment_dialog_keeps_today_s_size_on_a_large_screen(self):
        dialog = self._run_environment_dialog(self.LARGE)
        self.assertEqual(self._last_arg(dialog, "resize"), (900, 640))

    def test_b2_d3_unknown_screen_keeps_today_s_sizes(self):
        from shared.ui_compact import fit_size_to_screen

        qt = mock.MagicMock(name="qt")
        qt.QApplication.screenAt.return_value = None
        qt.QApplication.primaryScreen.return_value = None
        self.assertEqual(fit_size_to_screen(qt, 900, 640), (900, 640))




if __name__ == "__main__":
    unittest.main()
