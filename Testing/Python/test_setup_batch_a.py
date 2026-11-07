"""Batch-backlog part A: setup file fixes (A1 W-B-CARRY-1, A2, A3, A4, A5, A6, A7)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import setup_selected  # noqa: E402


class SlicerPythonIdentityTests(unittest.TestCase):
    def test_a1_venv_symlink_to_python_real_is_external(self):
        """A1 W-B-CARRY-1: a venv bin/python linked to python-real is not Slicer's Python. Fails before: realpath matched."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve() / "Slicer"
            real = root / "lib/Python/bin/python-real"
            real.parent.mkdir(parents=True)
            real.write_text("")
            venv_python = Path(folder).resolve() / "venv/bin/python"
            venv_python.parent.mkdir(parents=True)
            venv_python.symlink_to(real)
            with patch("sys.executable", str(real)):
                self.assertTrue(setup_selected.is_running_slicer_python(real))
                self.assertFalse(setup_selected.is_running_slicer_python(venv_python))

    def test_a1_real_bin_layout_matches_python_real_and_pythonslicer(self):
        """A1 (round-3 NB6): the layout bin/python-real + bin/PythonSlicer; an unrelated symlink to either is external."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve() / "Slicer"
            launcher, real = root / "bin/PythonSlicer", root / "bin/python-real"
            real.parent.mkdir(parents=True)
            launcher.write_text("")
            real.write_text("")
            link = Path(folder).resolve() / "venv/bin/python"
            link.parent.mkdir(parents=True)
            link.symlink_to(real)
            with patch("sys.executable", str(real)):
                self.assertTrue(setup_selected.is_running_slicer_python(launcher))
                self.assertTrue(setup_selected.is_running_slicer_python(real))
                self.assertFalse(setup_selected.is_running_slicer_python(link))

    def test_a1_nb5_running_pythons_computed_once_per_call(self):
        """A1 (NB5): is_running_slicer_python calls running_slicer_pythons once."""
        with patch.object(setup_selected, "running_slicer_pythons", return_value=set()) as found:
            setup_selected.is_running_slicer_python("/x/python")
        self.assertEqual(found.call_count, 1)


class Dcm2niixWarningTests(unittest.TestCase):
    """A2: a failing dcm2niix blocks readiness only when the plugin installed it."""

    @staticmethod
    def failing_run(argv, **kwargs):
        from types import SimpleNamespace
        return SimpleNamespace(returncode=1, stdout="", stderr="dcm2niix: bad news")

    def run_checks(self, request, *, packaged=None):
        from shared import setup_selftest
        with patch.object(setup_selftest.subprocess, "run", side_effect=self.failing_run), \
                patch("shared.input_import.packaged_dcm2niix", return_value=packaged):
            return setup_selftest.run_checks(request)

    def test_a2_a_failing_path_dcm2niix_is_a_warning_with_path_and_output(self):
        report = self.run_checks({"dcm2niix": "/usr/local/bin/dcm2niix"})
        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(report["warnings"]), 1)
        self.assertIn("/usr/local/bin/dcm2niix", report["warnings"][0])
        self.assertIn("bad news", report["warnings"][0])
        self.assertNotIn("dcm2niix:version", report["checks"])

    def test_a2_a_failing_wheel_dcm2niix_is_not_ready(self):
        from shared import setup_selftest
        with self.assertRaises(RuntimeError):
            self.run_checks({"dcm2niix": "/slicer/bin/dcm2niix", "dcm2niix_installed_by_plugin": True})
        with tempfile.TemporaryDirectory() as folder:
            wheel = Path(folder) / "dcm2niix"
            wheel.write_text("")
            with self.assertRaises(RuntimeError):  # the request flag is missing, the running Python's wheel is the same file
                self.run_checks({"dcm2niix": str(wheel)}, packaged=wheel)
            report = SimpleStatuses()
            request = setup_selftest.requests_for_report(report, backend_paths={"packaged_dcm2niix": str(wheel)})[0]
            self.assertTrue(request["dcm2niix_installed_by_plugin"])

    def test_a2_the_request_for_a_path_dcm2niix_is_not_plugin_installed(self):
        from shared import setup_selftest
        with patch("shared.input_import.packaged_dcm2niix", return_value=None):
            request = setup_selftest.requests_for_report(SimpleStatuses(), backend_paths={"dcm2niix_executable": "/usr/bin/dcm2niix"})[0]
        self.assertFalse(request["dcm2niix_installed_by_plugin"])

    def test_a2_the_cli_logs_the_warning_and_stays_completed(self):
        from shared.selected_setup import run_setup_steps
        with tempfile.TemporaryDirectory() as folder:
            receipt = Path(folder) / "result.json"
            receipt.write_text('{"status": "passed", "warnings": ["dcm2niix at /x failed its check"]}')
            lines = []
            records = run_setup_steps([dict(id="self_check", title="Test", commands=[])], self_check=lambda step: ([], [receipt]),
                                      log=lines.append)
        self.assertEqual(records[0]["status"], "completed")
        self.assertTrue(any("Warning" in line and "/x" in line for line in lines))


def SimpleStatuses():
    from types import SimpleNamespace
    return SimpleNamespace(statuses=[], targets=("validation",))


if __name__ == "__main__":
    unittest.main()
