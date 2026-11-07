"""Batch-backlog round, part B: B7 (resolver), B9, B10, B11 checks. Each test names its item."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (REPO_ROOT, REPO_ROOT / "Testing" / "Python"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import user_data_isolation  # noqa: E402

user_data_isolation.install()

_SLICER_VARIABLES = ("SLICER_HOME", "Slicer_HOME", "SLICERDSPACE_SLICER_HOME", "SLICERDDSURFER_SLICER_HOME",
                     "SLICERDSPACE_SLICER_EXECUTABLE", "SLICERDDSURFER_SLICER_EXECUTABLE", "DTI_SLICER_PATH", "SLICER_PATH")


def _clean_slicer_environment(**extra):
    environment = {key: value for key, value in os.environ.items() if key not in _SLICER_VARIABLES}
    environment.update(extra)
    return patch.dict(os.environ, environment, clear=True)


class B7ResolverTest(unittest.TestCase):
    def test_b7_a_c1_2_the_resolver_message_reaches_the_mask_user(self) -> None:
        from shared import brain_mask
        from shared.brain_mask import BrainMaskConfig, inspect_brain_mask_backend

        message = "Failed to resolve the PythonSlicer executable outside Slicer. Set SLICER_HOME to the Slicer install folder."
        with patch("shared.pipeline.resolve_pythonslicer_executable", side_effect=RuntimeError(message)):
            found = inspect_brain_mask_backend(BrainMaskConfig(pnl_mask_route="native", brain_mask_device="cpu"))
        problems = [item for item in found["missing_required"] if str(item).startswith(brain_mask.PNL_NATIVE_PYTHON_MISSING)]
        self.assertEqual(len(problems), 1)
        self.assertIn("Set SLICER_HOME", problems[0])

    def _slicer_install(self, tmp: str) -> Path:
        home = Path(tmp) / "Slicer-5.8"
        (home / "bin").mkdir(parents=True)
        python = home / "bin" / "PythonSlicer"
        python.write_text("#!/bin/sh\n")
        python.chmod(0o755)
        return home

    def test_b7_a_c1_3_the_pythonslicer_resolver_reads_the_slicerdspace_home_variable(self) -> None:
        from shared.pipeline import resolve_pythonslicer_executable

        with tempfile.TemporaryDirectory() as tmp:
            home = self._slicer_install(tmp)
            with _clean_slicer_environment(SLICERDSPACE_SLICER_HOME=str(home)):
                self.assertEqual(Path(resolve_pythonslicer_executable()).resolve(),
                                 (home / "bin" / "PythonSlicer").resolve())

    def test_b7_a_c1_3_the_pythonslicer_resolver_reads_the_slicerdspace_executable_variable(self) -> None:
        from shared.pipeline import resolve_pythonslicer_executable

        with tempfile.TemporaryDirectory() as tmp:
            home = self._slicer_install(tmp)
            launcher = home / "Slicer"
            launcher.write_text("#!/bin/sh\n")
            with _clean_slicer_environment(SLICERDSPACE_SLICER_EXECUTABLE=str(launcher)):
                self.assertEqual(Path(resolve_pythonslicer_executable()).resolve(),
                                 (home / "bin" / "PythonSlicer").resolve())

    def test_b7_a_c1_3_macos_app_launcher_gives_the_contents_python(self) -> None:
        from shared.pipeline import resolve_pythonslicer_executable

        with tempfile.TemporaryDirectory() as tmp:
            app = Path(tmp) / "Slicer.app" / "Contents"
            (app / "bin").mkdir(parents=True)
            (app / "MacOS").mkdir()
            (app / "bin" / "PythonSlicer").write_text("#!/bin/sh\n")
            (app / "MacOS" / "Slicer").write_text("#!/bin/sh\n")
            with _clean_slicer_environment(SLICERDSPACE_SLICER_EXECUTABLE=str(app / "MacOS" / "Slicer")):
                self.assertEqual(Path(resolve_pythonslicer_executable()).resolve(),
                                 (app / "bin" / "PythonSlicer").resolve())


class B10SmallItemsTest(unittest.TestCase):
    def test_b10_multi_line_hint_stays_one_markdown_bullet(self) -> None:
        from dataclasses import replace
        from shared.environment import EnvironmentDoctorConfig, run_environment_doctor, write_environment_report

        with tempfile.TemporaryDirectory() as tmp:
            result = run_environment_doctor(EnvironmentDoctorConfig(targets=("tractography",), repo_root=REPO_ROOT,
                                                                    tractography_backend="csd_act"))
            external = next(item for item in result.statuses if item.kind != "python")
            hint = "Teem library not found.\nSearched:\n  /a/lib\n  /b/lib\n\nSet SLICERDSPACE_TEEM_LIBRARY."
            statuses = tuple(replace(item, status="missing", install_hint=hint) if item is external else item
                             for item in result.statuses)
            written = write_environment_report(replace(result, statuses=statuses), Path(tmp) / "report")
            lines = written["external_notes"].read_text(encoding="utf-8").splitlines()
            bullet = next(index for index, line in enumerate(lines) if line.startswith("- Suggested fix:"))
            self.assertEqual(lines[bullet], "- Suggested fix: Teem library not found. Searched: /a/lib /b/lib Set SLICERDSPACE_TEEM_LIBRARY.")
            self.assertEqual(lines[bullet + 1], "")

    def test_b10_benchmark_drops_the_cohort_environment(self) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location("benchmark_shared_surface_b10", REPO_ROOT / "scripts" / "benchmark_shared_surface.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with patch.dict(os.environ, {"SLICERDSPACE_COHORT_SCHEDULE": "/x", "SLICERDSPACE_COHORT_LEASE": "y",
                                     "SLICERDSPACE_OTHER": "keep"}):
            module.leave_cohort_environment()
            self.assertNotIn("SLICERDSPACE_COHORT_SCHEDULE", os.environ)
            self.assertNotIn("SLICERDSPACE_COHORT_LEASE", os.environ)
            self.assertEqual(os.environ["SLICERDSPACE_OTHER"], "keep")


if __name__ == "__main__":
    unittest.main()
