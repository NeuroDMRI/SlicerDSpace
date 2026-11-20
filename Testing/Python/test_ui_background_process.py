#!/usr/bin/env python3
"""Tests for background-process helper state."""

from __future__ import annotations

import unittest
from pathlib import Path

from shared.ui_background_process import (
    background_process_env_overrides,
    build_background_run_context,
    qprocess_is_running,
)


class UIBackgroundProcessTest(unittest.TestCase):
    def test_missing_process_is_not_running(self) -> None:
        self.assertFalse(qprocess_is_running(None))

    def test_background_process_env_overrides_forces_unbuffered_stdout(self) -> None:
        overrides = background_process_env_overrides()
        # X1: workers also get UTF-8 stdout so a non-ASCII path cannot crash a cp1252 console.
        self.assertEqual(overrides, {"PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})

    def test_build_single_subject_background_context(self) -> None:
        context = build_background_run_context(
            {
                "subject_id": "sub-01",
                "subject_root": Path("/data/sub-01"),
                "output_root": Path("/out"),
                "run_name": "Test1",
                "targets": ("validation",),
            },
            ["validation"],
        )

        self.assertEqual(context["subject_id"], "sub-01")
        self.assertEqual(context["subject_root"], Path("/data/sub-01"))
        self.assertEqual(context["selected_workflows"], ["validation"])
        self.assertIn("started_at", context)

    def test_build_batch_background_context(self) -> None:
        context = build_background_run_context(
            {
                "subject_id": "sub-01",
                "subject_root": Path("/data/sub-01"),
                "output_root": Path("/out"),
                "run_name": "Test1",
                "targets": ("validation",),
            },
            ["validation"],
            batch_root=Path("/data/batch"),
            batch_size=4,
        )

        self.assertEqual(context["subject_id"], "batch")
        self.assertEqual(context["subject_root"], Path("/data/batch"))
        self.assertEqual(context["batch_size"], 4)


if __name__ == "__main__":
    unittest.main()
