#!/usr/bin/env python3
"""Tests for run-state presentation, button enablement and log colouring rules."""

from __future__ import annotations

import unittest

from shared.ui_run_state import (
    LOG_LEVEL_COLORS,
    RUN_BUTTON_READY_TOOLTIP,
    log_line_html,
    log_line_level,
    progress_bar_format,
    result_actions_state,
    run_button_state,
    run_state_from_manifest_status,
    run_state_presentation,
)


class RunStateTest(unittest.TestCase):
    def test_states_mirror_slicer_cli_progress_labels(self) -> None:
        self.assertEqual(run_state_presentation("idle").label, "Idle")
        self.assertEqual(run_state_presentation("running").label, "Running")
        self.assertTrue(run_state_presentation("running").busy)
        self.assertEqual(run_state_presentation("failed").label, "Completed with errors")
        self.assertEqual(run_state_presentation("cancelled").label, "Cancelled")
        self.assertEqual(run_state_presentation("nonsense").label, "Idle")

    def test_manifest_status_mapping(self) -> None:
        self.assertEqual(run_state_from_manifest_status("completed"), "completed")
        self.assertEqual(run_state_from_manifest_status("completed_with_blocks"), "completed_with_blocks")
        self.assertEqual(run_state_from_manifest_status("partial_failed"), "failed")
        self.assertEqual(run_state_from_manifest_status("planned"), "idle")
        self.assertEqual(run_state_from_manifest_status(""), "idle")

    def test_progress_format(self) -> None:
        self.assertEqual(progress_bar_format(2, 5), "2 / 5 stages")
        self.assertEqual(progress_bar_format(1, 1), "1 / 1 stage")
        self.assertIn("Waiting", progress_bar_format(0, 0))


class RunButtonTest(unittest.TestCase):
    BASE = dict(
        pipeline_active=False,
        selected_targets=("validation", "segmentation"),
        batch_mode=False,
        subject_id="sub-01",
        subject_root_text="/data/sub-01",
    )

    def test_ready(self) -> None:
        state = run_button_state(**self.BASE)
        self.assertTrue(state.enabled)
        self.assertEqual(state.tooltip, RUN_BUTTON_READY_TOOLTIP)

    def test_pending_inputs_keep_button_clickable_with_actionable_tooltips(self) -> None:
        cases = {
            "selected_targets": (("validation",), "2. Analysis"),
            "subject_root_text": ("", "Subject root"),
        }
        for key, (value, expected) in cases.items():
            state = run_button_state(**{**self.BASE, key: value})
            self.assertTrue(state.enabled, key)
            self.assertIn(expected, state.tooltip, key)

    def test_c6_empty_subject_id_does_not_block_the_run_button(self) -> None:
        # C6: the ID is derived from the input path, so an empty field is ready, not pending.
        state = run_button_state(**{**self.BASE, "subject_id": ""})
        self.assertTrue(state.enabled)
        self.assertEqual(state.tooltip, RUN_BUTTON_READY_TOOLTIP)

    def test_active_run_is_disabled_even_with_missing_inputs(self) -> None:
        state = run_button_state(**{**self.BASE, "pipeline_active": True, "subject_id": ""})
        self.assertFalse(state.enabled)
        self.assertIn("Cancel running job", state.tooltip)

    def test_batch_mode_needs_a_root_but_not_a_subject_id(self) -> None:
        self.assertTrue(run_button_state(**{**self.BASE, "batch_mode": True, "subject_id": ""}).enabled)
        state = run_button_state(**{**self.BASE, "batch_mode": True, "subject_root_text": "", "batch_root_text": ""})
        self.assertTrue(state.enabled)
        self.assertIn("Batch folder", state.tooltip)

    def test_topup_errors_keep_button_clickable_with_first_error(self) -> None:
        state = run_button_state(**self.BASE, topup_errors=("Reverse-PE b=0 image not found: /x", "other"))
        self.assertTrue(state.enabled)
        self.assertIn("not found: /x", state.tooltip)
        self.assertIn("3. Route and parameters", state.tooltip)


class ResultActionsTest(unittest.TestCase):
    def test_result_actions(self) -> None:
        self.assertTrue(result_actions_state(has_subject_run=True, is_batch=False).enabled)
        batch = result_actions_state(has_subject_run=False, is_batch=True)
        self.assertFalse(batch.enabled)
        self.assertIn("subject's run", batch.tooltip)
        none = result_actions_state(has_subject_run=False, is_batch=False)
        self.assertIn("Add a run folder…", none.tooltip)
        error = result_actions_state(has_subject_run=True, is_batch=False, error="version 9 unsupported")
        self.assertFalse(error.enabled)
        self.assertEqual(error.tooltip, "version 9 unsupported")


class LogStylingTest(unittest.TestCase):
    def test_levels(self) -> None:
        self.assertEqual(log_line_level("[Orchestrator] Running DDSeg segmentation"), "stage")
        self.assertEqual(log_line_level("ERROR: model missing"), "error")
        self.assertEqual(log_line_level("Traceback (most recent call last):"), "error")
        self.assertEqual(log_line_level("[Orchestrator] tractography.csd_act: failed (tckgen)"), "error")
        self.assertEqual(log_line_level("WARNING: low b-value shell count"), "warning")
        self.assertEqual(log_line_level("[Orchestrator] bundles.wmql: blocked (missing atlas)"), "warning")
        self.assertEqual(log_line_level("[Orchestrator] segmentation.ddseg: completed"), "success")
        self.assertEqual(log_line_level("plain text"), "info")
        self.assertEqual(log_line_level(""), "info")

    def test_html_is_escaped_and_coloured(self) -> None:
        html = log_line_html("ERROR: <script>alert(1)</script> & more")
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("&amp; more", html)
        self.assertIn(LOG_LEVEL_COLORS["error"], html)
        self.assertIn("white-space:pre", html)
        self.assertNotIn("<script>", html)


if __name__ == "__main__":
    unittest.main()
