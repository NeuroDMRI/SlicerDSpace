#!/usr/bin/env python3
"""Tests for widget settings/status helper modules."""

from __future__ import annotations

import tempfile
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from shared.ui_settings import (coerce_user_setting_value, common_user_setting_specs,
                                mark_new_parameter_node, consume_user_defaults_request, SETTINGS_STATE_ATTRIBUTE)
from shared.ui_status_log import set_existing_report_path, append_log, repaint_log_widget


class ParameterNode:
    def __init__(self, parameters=()):
        self.parameters = parameters
        self.attributes = {}

    def GetParameterNames(self):
        return self.parameters

    def GetAttribute(self, name):
        return self.attributes.get(name)

    def SetAttribute(self, name, value):
        self.attributes[name] = value


LONG_LINE_CHARS = 1000
HORIZONTAL_MAXIMUM = 40000


class LogWidget:
    """Fake QPlainTextEdit that models the Qt behaviour measured in real Qt 5.15.

    appendHtml moves the text cursor to the end of the chunk and leaves the horizontal bar alone.
    ensureCursorVisible scrolls the horizontal bar to the far right when the cursor line is long,
    and follows a clicked line vertically. The first long line makes the horizontal bar appear,
    which adds one line to the vertical range.
    """

    def __init__(self, position=100):
        self.bar = SimpleNamespace(value=position, maximum=100)
        self.bar.setValue = lambda value: setattr(self.bar, 'value', min(int(value), self.bar.maximum))
        self.hbar = SimpleNamespace(value=0, maximum=0)
        self.hbar.setValue = lambda value: setattr(self.hbar, 'value', min(int(value), self.hbar.maximum))
        self.insertions = []
        self.followed = False
        self.cursor_calls = []
        self.clicked_line = None
        self._last_line_long = False
        self.plain_lines = []
        self.cleared = False
        self.layout_pending = False

    def verticalScrollBar(self):
        return self.bar

    def horizontalScrollBar(self):
        return self.hbar

    def appendHtml(self, text):
        self.insertions.append(text)
        self.plain_lines.append(text)
        long_line = ('x' * LONG_LINE_CHARS) in text
        self._last_line_long = long_line
        self.bar.maximum += max(1, text.count('<p '))
        self.bar.value = self.bar.maximum
        if long_line and self.hbar.maximum == 0:
            self.hbar.maximum = HORIZONTAL_MAXIMUM
            self.layout_pending = True

    def run_layout(self):
        """The event loop lays out the widget: the new horizontal bar costs one line of viewport."""
        if self.layout_pending:
            self.layout_pending = False
            self.bar.maximum += 1

    def ensureCursorVisible(self):
        self.followed = True
        self.cursor_calls.append('ensureCursorVisible')
        if self._last_line_long:
            self.hbar.value = self.hbar.maximum
        if self.clicked_line is not None:
            self.bar.value = self.clicked_line
        else:
            self.bar.value = self.bar.maximum

    def moveCursor(self, *args):
        self.cursor_calls.append('moveCursor')

    def setTextCursor(self, *args):
        self.cursor_calls.append('setTextCursor')

    def toPlainText(self):
        return '\n'.join(self.plain_lines)

    def clear(self):
        self.cleared = True
        self.plain_lines = []
        self.insertions = []
        self.hbar.value = 0
        self.bar.value = 0
        self.bar.maximum = 0


class UISettingsStatusLogTest(unittest.TestCase):
    def test_user_defaults_only_apply_once_to_new_nodes(self):
        raw = ParameterNode()
        mark_new_parameter_node(raw)
        self.assertEqual(raw.GetAttribute(SETTINGS_STATE_ATTRIBUTE), 'pending')
        wrapped = SimpleNamespace(parameterNode=raw)
        self.assertTrue(consume_user_defaults_request(wrapped))
        raw.parameters = ('subjectId',)
        mark_new_parameter_node(raw)
        self.assertFalse(consume_user_defaults_request(wrapped))
        pending = ParameterNode()
        mark_new_parameter_node(pending)
        configured = SimpleNamespace(parameterNode=pending, allParameters=('subjectId',),
                                     getValue=lambda name: 'configured-before-gui',
                                     default=lambda name: SimpleNamespace(value=''))
        self.assertFalse(consume_user_defaults_request(configured))

    def test_existing_legacy_scene_keeps_parameters_without_new_marker(self):
        raw = ParameterNode(('subjectId', 'orchestrationTractographyBackend'))
        mark_new_parameter_node(raw)
        self.assertFalse(consume_user_defaults_request(SimpleNamespace(parameterNode=raw)))
        self.assertEqual(raw.attributes, {})

    def test_log_chunk_is_one_escaped_insertion_and_preserves_reading_position(self):
        for position in (0, 40, 100):
            log = LogWidget(position)
            owner = SimpleNamespace(ui=SimpleNamespace(logPlainTextEdit=log),
                                    _lastProcessEventsAt=0, _trim_log_widget=lambda widget: None)
            with patch.dict(sys.modules, {'slicer': SimpleNamespace(app=SimpleNamespace(processEvents=lambda: None))}):
                append_log(owner, 'first <line>\nWARNING: second line\nlast line')
            self.assertEqual(len(log.insertions), 1)
            self.assertEqual(log.insertions[0].count('<p '), 3)
            self.assertIn('&lt;line&gt;', log.insertions[0])
            # b2 log: the end position is checked, not the ensureCursorVisible call.
            if position == 100:
                self.assertEqual(log.bar.value, log.bar.maximum)
            else:
                self.assertEqual(log.bar.value, position)

    # ---- batch 2, lane log: the log view must not scroll horizontally by itself ----

    def _owner(self, log):
        return SimpleNamespace(ui=SimpleNamespace(logPlainTextEdit=log),
                               _lastProcessEventsAt=0, _trim_log_widget=lambda widget: None)

    def _append(self, owner, text, timers=None):
        queue = timers if timers is not None else []
        fake_qt = SimpleNamespace(QTimer=SimpleNamespace(singleShot=lambda ms, fn: queue.append(fn)))
        with patch.dict(sys.modules, {'slicer': SimpleNamespace(app=SimpleNamespace(processEvents=lambda: None)),
                                      'qt': fake_qt}):
            append_log(owner, text)
        return queue

    def test_b2_log_long_last_line_keeps_horizontal_value_at_zero(self):
        log = LogWidget(100)
        owner = self._owner(log)
        self._append(owner, 'short line')
        self._append(owner, 'x' * 5000)
        self.assertEqual(log.hbar.value, 0)
        self.assertGreater(log.hbar.maximum, 0)
        self.assertEqual(log.bar.value, log.bar.maximum)

    def test_b2_log_chunk_ending_in_long_line_keeps_horizontal_value_at_zero(self):
        log = LogWidget(100)
        owner = self._owner(log)
        self._append(owner, 'a\nb\n' + 'x' * 5000)
        self.assertEqual(log.hbar.value, 0)

    def test_b2_log_user_chosen_horizontal_value_survives_append(self):
        log = LogWidget(100)
        owner = self._owner(log)
        self._append(owner, 'x' * 5000)
        log.hbar.setValue(300)
        self.assertEqual(log.hbar.value, 300)
        self._append(owner, 'short line')
        self.assertEqual(log.hbar.value, 300)
        self._append(owner, 'y' * 5000 + ' ' + 'x' * 5000)
        self.assertEqual(log.hbar.value, 300)

    def test_b2_log_user_chosen_horizontal_value_survives_theme_repaint(self):
        log = LogWidget(100)
        owner = self._owner(log)
        self._append(owner, 'first\n' + 'x' * 5000)
        log.hbar.setValue(300)
        repaint_log_widget(log, dark=True)
        self.assertTrue(log.cleared)
        self.assertEqual(log.hbar.value, 300)
        self.assertEqual(len(log.insertions), 1)
        self.assertIn('first', log.insertions[0])

    def test_b2_log_theme_repaint_keeps_vertical_rules(self):
        log = LogWidget(100)
        owner = self._owner(log)
        for index in range(30):
            self._append(owner, f'line {index}')
        log.bar.setValue(10)
        repaint_log_widget(log, dark=True)
        self.assertEqual(log.bar.value, 10)

        log = LogWidget(100)
        owner = self._owner(log)
        for index in range(30):
            self._append(owner, f'line {index}')
        self.assertEqual(log.bar.value, log.bar.maximum)
        repaint_log_widget(log, dark=False)
        self.assertEqual(log.bar.value, log.bar.maximum)

    def test_b2_log_follow_at_bottom_and_scrolled_up_keep_vertical_rule(self):
        log = LogWidget(100)
        owner = self._owner(log)
        self._append(owner, 'one\ntwo')
        self.assertEqual(log.bar.value, log.bar.maximum)
        scrolled = LogWidget(40)
        owner = self._owner(scrolled)
        self._append(owner, 'one\ntwo')
        self.assertEqual(scrolled.bar.value, 40)
        self._append(owner, 'x' * 5000)
        self.assertEqual(scrolled.bar.value, 40)
        self.assertEqual(scrolled.hbar.value, 0)

    def test_b2_log_deferred_pin_restores_bottom_after_horizontal_bar_appears(self):
        log = LogWidget(100)
        owner = self._owner(log)
        timers = self._append(owner, 'x' * 5000)
        # The event loop lays out the widget: the horizontal bar that appeared adds one line.
        log.run_layout()
        self.assertEqual(log.bar.value, log.bar.maximum - 1)
        self.assertEqual(len(timers), 1)
        for callback in timers:
            callback()
        self.assertEqual(log.bar.value, log.bar.maximum)

    def test_b2_log_deferred_pin_keeps_position_of_reader_who_scrolled_up(self):
        log = LogWidget(40)
        owner = self._owner(log)
        timers = self._append(owner, 'x' * 5000)
        for callback in timers:
            callback()
        self.assertEqual(log.bar.value, 40)

    def test_b2_log_deferred_pin_ignores_a_deleted_widget(self):
        log = LogWidget(100)
        owner = self._owner(log)
        timers = self._append(owner, 'x' * 5000)

        def broken():
            raise RuntimeError('Internal C++ object already deleted')
        log.bar.setValue = lambda value: broken()
        for callback in timers:
            callback()

    def test_b2_log_works_without_qt_event_loop(self):
        log = LogWidget(100)
        owner = self._owner(log)
        with patch.dict(sys.modules, {'slicer': SimpleNamespace(app=SimpleNamespace(processEvents=lambda: None)),
                                      'qt': None}):
            append_log(owner, 'x' * 5000)
        self.assertEqual(log.hbar.value, 0)
        self.assertEqual(log.bar.value, log.bar.maximum)

        class BrokenTimer:
            @staticmethod
            def singleShot(ms, fn):
                raise RuntimeError('no event loop')
        log = LogWidget(100)
        owner = self._owner(log)
        with patch.dict(sys.modules, {'slicer': SimpleNamespace(app=SimpleNamespace(processEvents=lambda: None)),
                                      'qt': SimpleNamespace(QTimer=BrokenTimer)}):
            append_log(owner, 'x' * 5000)
        self.assertEqual(log.bar.value, log.bar.maximum)

    def test_b2_log_path_makes_no_cursor_call(self):
        log = LogWidget(100)
        owner = self._owner(log)
        self._append(owner, 'a\nb')
        self._append(owner, 'x' * 5000)
        repaint_log_widget(log, dark=True)
        self.assertEqual(log.cursor_calls, [])

    def test_b2_log_follow_survives_clicked_text_in_the_log(self):
        log = LogWidget(100)
        log.clicked_line = 5
        owner = self._owner(log)
        self._append(owner, 'next line')
        self.assertEqual(log.bar.value, log.bar.maximum)

    def test_blank_lines_keep_the_original_insertion_path(self):
        log = LogWidget()
        owner = SimpleNamespace(ui=SimpleNamespace(logPlainTextEdit=log),
                                _lastProcessEventsAt=0, _trim_log_widget=lambda widget: None)
        with patch.dict(sys.modules, {'slicer': SimpleNamespace(app=SimpleNamespace(processEvents=lambda: None))}):
            append_log(owner, 'first\n\nlast')
        self.assertEqual(len(log.insertions), 3)
        self.assertNotIn('<p ', ''.join(log.insertions))

    def test_common_user_setting_specs_include_run_and_target_state(self) -> None:
        names = [name for name, _value_type in common_user_setting_specs()]

        self.assertIn("orchestrationRunName", names)
        self.assertIn("batchMode", names)
        self.assertIn("targetTractography", names)

    def test_coerce_user_setting_values(self) -> None:
        self.assertTrue(coerce_user_setting_value("true", bool))
        self.assertFalse(coerce_user_setting_value("0", bool))
        self.assertEqual(coerce_user_setting_value("7", int), 7)
        self.assertEqual(coerce_user_setting_value(None, str), "")

    def test_set_existing_report_path_updates_button_state(self) -> None:
        owner = SimpleNamespace(_button=SimpleNamespace(enabled=False))
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.html"
            report.write_text("report", encoding="utf-8")

            set_existing_report_path(owner, "_report", "_button", report)

        self.assertEqual(owner._report, report)
        self.assertTrue(owner._button.enabled)

    def test_set_existing_report_path_clears_missing_path(self) -> None:
        owner = SimpleNamespace(_button=SimpleNamespace(enabled=True))

        set_existing_report_path(owner, "_report", "_button", "/missing/report.html")

        self.assertIsNone(owner._report)
        self.assertFalse(owner._button.enabled)


if __name__ == "__main__":
    unittest.main()
