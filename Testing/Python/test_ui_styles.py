#!/usr/bin/env python3
"""Tests for shared UI stylesheet helpers."""

from __future__ import annotations

import re
import unittest

from shared.ui_styles import (
    DARK_ENVIRONMENT_READY_LABEL_STYLE,
    DARK_ENVIRONMENT_RESULTS_DIALOG_STYLE,
    DARK_RUN_STATUS_TABLE_STYLE,
    ENVIRONMENT_READY_LABEL_STYLE,
    ENVIRONMENT_RESULTS_DIALOG_STYLE,
    RUN_STATUS_TABLE_STYLE,
    SLICERDSPACE_DARK_UI_STYLE,
    SLICERDSPACE_UI_STYLE,
    environment_ready_label_style,
    environment_results_dialog_style,
    run_status_table_style,
    run_summary_label_style,
    ui_stylesheet,
)


class UIStylesTest(unittest.TestCase):
    def test_main_stylesheet_contains_key_panels(self) -> None:
        self.assertIn("QFrame#headerFrame", SLICERDSPACE_UI_STYLE)
        self.assertIn("QWidget#externalBackendsGroup", SLICERDSPACE_UI_STYLE)
        self.assertIn("QPushButton#runSelectedButton", SLICERDSPACE_UI_STYLE)
        self.assertIn("QPlainTextEdit#logPlainTextEdit", SLICERDSPACE_UI_STYLE)
        self.assertNotIn("border-left", SLICERDSPACE_UI_STYLE)
        self.assertNotIn("qlineargradient", SLICERDSPACE_UI_STYLE)
        self.assertNotIn("font-size: 18px", SLICERDSPACE_UI_STYLE)

    def test_run_status_table_style_contains_header_and_selection_rules(self) -> None:
        self.assertIn("QTableWidget", RUN_STATUS_TABLE_STYLE)
        self.assertIn("QHeaderView::section", RUN_STATUS_TABLE_STYLE)
        self.assertIn("selection-background-color", RUN_STATUS_TABLE_STYLE)

    def test_environment_dialog_style_contains_summary_and_item_rules(self) -> None:
        self.assertIn("QLabel#environmentSummaryLabel", ENVIRONMENT_RESULTS_DIALOG_STYLE)
        self.assertIn("QFrame#environmentItemFrame", ENVIRONMENT_RESULTS_DIALOG_STYLE)
        self.assertIn("QScrollArea", ENVIRONMENT_RESULTS_DIALOG_STYLE)
        self.assertIn("#2E7D32", ENVIRONMENT_READY_LABEL_STYLE)

    def test_run_summary_label_style_uses_state_palette_and_unknown_fallback(self) -> None:
        failed = run_summary_label_style("failed")
        unknown = run_summary_label_style("not-a-state")

        self.assertIn("#FDECEC", failed)
        self.assertIn("#6B1A1A", failed)
        self.assertIn("#F3F7F4", unknown)
        self.assertIn("QLabel#runSummaryLabel", unknown)
        self.assertNotIn("border-left", failed)


    def _rule_for(self, selector: str) -> str:
        """Return the declaration block of the rule whose selector list names ``selector``."""

        for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", SLICERDSPACE_UI_STYLE):
            if selector in [part.strip() for part in selectors.split(",")]:
                return body
        self.fail(f"no rule for {selector}")

    def test_review_and_details_labels_have_no_box(self) -> None:
        for selector in ("QLabel#reviewStateLabel", "QLabel#runDetailsLabel"):
            with self.subTest(selector=selector):
                body = self._rule_for(selector)
                self.assertIn("background: transparent", body)
                self.assertIn("border: none", body)
                self.assertIn("color: #63727D", body)
                self.assertRegex(body, r"padding: 0 \d+px")

    def test_run_summary_card_keeps_its_box(self) -> None:
        body = self._rule_for("QLabel#runSummaryLabel")
        self.assertIn("border: 1px solid", body)

    def test_dark_mode_stylesheets_provide_high_contrast_inputs_and_labels(self) -> None:
        self.assertEqual(ui_stylesheet(dark=False), SLICERDSPACE_UI_STYLE)
        self.assertEqual(ui_stylesheet(dark=True), SLICERDSPACE_DARK_UI_STYLE)
        self.assertEqual(run_status_table_style(dark=True), DARK_RUN_STATUS_TABLE_STYLE)
        self.assertEqual(environment_results_dialog_style(dark=True), DARK_ENVIRONMENT_RESULTS_DIALOG_STYLE)
        self.assertEqual(environment_ready_label_style(dark=True), DARK_ENVIRONMENT_READY_LABEL_STYLE)
        self.assertIn("background: #22272E;", SLICERDSPACE_DARK_UI_STYLE)
        self.assertIn("color: #F0F6FC;", SLICERDSPACE_DARK_UI_STYLE)
        self.assertIn("color: #7CC4EA;", SLICERDSPACE_DARK_UI_STYLE)
        self.assertIn("color: #1E293B;", SLICERDSPACE_UI_STYLE)
        dark_failed = run_summary_label_style("failed", dark=True)
        self.assertIn("#3B2124", dark_failed)
        self.assertIn("#FFA8A8", dark_failed)


if __name__ == "__main__":
    unittest.main()
