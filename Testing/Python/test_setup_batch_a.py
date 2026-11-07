"""Batch-backlog part A: setup file fixes (A1 W-B-CARRY-1, A2, A3, A4, A5, A6, A7)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import setup_selected  # noqa: E402
from shared.setup_components import LicenceNotAccepted  # noqa: E402


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
            self.run_checks({"dcm2niix": "/slicer/bin/dcm2niix", "dcm2niix_configured": True})
        with tempfile.TemporaryDirectory() as folder:
            wheel = Path(folder) / "dcm2niix"
            wheel.write_text("")
            with self.assertRaises(RuntimeError):  # the request flag is missing, the running Python's wheel is the same file
                self.run_checks({"dcm2niix": str(wheel)}, packaged=wheel)
            report = SimpleStatuses()
            request = setup_selftest.requests_for_report(report, backend_paths={"packaged_dcm2niix": str(wheel)})[0]
            self.assertTrue(request["dcm2niix_configured"])

    def test_a2_the_request_for_a_path_found_dcm2niix_is_not_configured(self):
        from shared import setup_selftest
        with patch("shared.input_import.packaged_dcm2niix", return_value=None), \
                patch("shared.input_import.find_dcm2niix", return_value="/usr/bin/dcm2niix"):
            request = setup_selftest.requests_for_report(SimpleStatuses())[0]
        self.assertFalse(request["dcm2niix_configured"])

    def test_a2b_an_explicit_user_set_failing_dcm2niix_is_not_ready(self):
        """A2b: an explicit dcm2niix_executable blocks ready when it fails. Fails at 2d121ad: it was a warning."""
        from shared import setup_selftest
        with patch("shared.input_import.packaged_dcm2niix", return_value=None):
            request = setup_selftest.requests_for_report(SimpleStatuses(), backend_paths={"dcm2niix_executable": "/opt/d2n/dcm2niix"})[0]
            self.assertTrue(request["dcm2niix_configured"])
            with self.assertRaises(RuntimeError):
                self.run_checks(request)

    def _main_with_failing_dcm2niix(self, argv_extra, folder):
        """Run scripts/check_setup_functionality.main() with a failing dcm2niix; return (exit code, result file text)."""
        from scripts import check_setup_functionality
        from shared import setup_selftest
        result = Path(folder) / "result.json"
        with patch.object(setup_selftest.subprocess, "run", side_effect=self.failing_run), \
                patch("shared.input_import.packaged_dcm2niix", return_value=None), \
                patch("sys.stdout", new=__import__("io").StringIO()):
            code = check_setup_functionality.main(["--result", str(result)] + argv_extra)
        return code, json.loads(result.read_text())

    def test_c1_an_explicit_dcm2niix_option_that_fails_is_not_ready(self):
        """C1: --dcm2niix is an explicit user choice. Fails at 0064d04: the script gave a passed result with a warning."""
        with tempfile.TemporaryDirectory() as folder:
            code, record = self._main_with_failing_dcm2niix(["--dcm2niix", "/opt/d2n/dcm2niix"], folder)
        self.assertEqual(record["status"], "failed")
        self.assertNotEqual(code, 0)
        self.assertIn("dcm2niix", record["error"])

    def test_c1_a_path_discovered_dcm2niix_in_a_request_file_still_gives_the_warning(self):
        """C1: a request with a PATH-found dcm2niix only (no flag, no setting) keeps the warning."""
        with tempfile.TemporaryDirectory() as folder:
            request = Path(folder) / "request.json"
            request.write_text(json.dumps({"dcm2niix": "/usr/local/bin/dcm2niix"}))
            code, record = self._main_with_failing_dcm2niix(["--request", str(request)], folder)
        self.assertEqual(code, 0)
        self.assertEqual(record["status"], "passed")
        self.assertEqual(len(record["warnings"]), 1)

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


class LicenceDialogRowTests(unittest.TestCase):
    def test_a3_nb1_the_gui_licence_dialog_shows_each_packages_local_licence_file(self):
        """A3 NB1: the GUI rows name the local file of every Python package, as the CLI does. Fails before: no such row part."""
        from types import SimpleNamespace
        from shared.ui_selected_setup import component_rows

        with tempfile.TemporaryDirectory() as folder:
            record = Path(folder) / "numpy-LICENSE.txt"
            record.write_text("x")
            packages = [dict(name="numpy", license="BSD-3-Clause", url="https://pypi.org/project/numpy/", file=str(record)),
                        dict(name="nibabel", license="MIT", url="https://pypi.org/project/nibabel/", file=None)]
            licence = dict(id="pypi_packages", name="Python packages", url="https://pypi.org", file=None, packages=packages)
            components = [dict(title="Python packages", size_bytes=0, licence=licence)]
            rows = component_rows(SimpleNamespace(_onboardingLanguage="en"), components, {"pypi_packages"})
        self.assertIn(record.as_uri(), rows[0])
        self.assertIn("no local file", rows[0])


def bridge_receipt(root, distribution):
    """A ps1 receipt: the three WSL bridge folders with an owner marker and a .conf for the distribution."""
    folders = {}
    for name, key in (("mrtrix", "mrtrix_bin_dir"), ("mrtrix2", "preprocessing_mrtrix_bin_dir"),
                      ("fsl", "preprocessing_fsl_bin_dir")):
        folder = Path(root) / name
        folder.mkdir()
        exe = folder / "tool.exe"
        Path(str(exe) + ".slicerdsurfer-owner").write_text("owner")
        Path(str(exe) + ".conf").write_text(f"distribution={distribution}\ntarget=/usr/bin/env\n")
        folders[key] = str(folder)
    return {"kind": "slicerdsurfer_user_config", "external_dependencies": folders}


class WindowsReceiptSettingsTests(unittest.TestCase):
    def test_a4_1_the_ps1_model_assets_dir_comes_back_to_the_gui(self):
        """A4 (1): WINDOWS_WRITTEN_SETTINGS includes model_assets_dir. Fails before: the key was dropped from the settings."""
        import json
        from types import SimpleNamespace
        from shared.selected_setup import WINDOWS_WRITTEN_SETTINGS, complete_step

        self.assertIn("model_assets_dir", WINDOWS_WRITTEN_SETTINGS)
        with tempfile.TemporaryDirectory() as root:
            config = Path(root) / "user_config.json"
            written = bridge_receipt(root, "Ubuntu")
            written["external_dependencies"]["model_assets_dir"] = str(Path(root) / "assets")
            config.write_text(json.dumps(written))
            with patch.dict("os.environ", {"SLICERDSPACE_USER_CONFIG": str(config)}):
                settings = complete_step({"id": "windows_runtime", "distribution": "Ubuntu", "receipt": config})
        self.assertEqual(settings["model_assets_dir"], str(Path(root) / "assets"))


class CliLauncherAndLicenceTests(unittest.TestCase):
    def test_a4_3_the_cli_passes_the_running_slicer_launcher_not_the_saved_one(self):
        """A4 (3): same launcher as the GUI (running Slicer first, saved path only as the fallback). Fails before: saved path."""
        from types import SimpleNamespace
        from shared.user_config import empty_user_config
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve() / "Slicer-5.10"
            real = root / "lib/Python/bin/python-real"
            launcher = root / "Slicer"
            real.parent.mkdir(parents=True)
            real.write_text("")
            launcher.write_text("")
            config = empty_user_config()
            config["external_dependencies"]["slicer_executable"] = "/old/Slicer"
            seen = []

            def plan(report, **kwargs):
                seen.append(kwargs["slicer_executable"])
                return {}

            def build(sys_executable):
                with patch("sys.executable", sys_executable), patch("shared.selected_setup.build_setup_plan", side_effect=plan), \
                        patch("scripts.setup_python_packages.dcm2niix_pending", return_value=False):
                    setup_selected.build_cli_plan(["validation"], config,
                                                  doctor=lambda cfg: SimpleNamespace(statuses=[], python_executable=None))
                return seen[-1]

            self.assertEqual(build(str(real)), str(launcher))
            self.assertEqual(build("/elsewhere/bin/python"), "/old/Slicer")

    def test_a4_2_the_cli_reports_a_licence_problem_with_the_names(self):
        """A4 (2): a PermissionError from setup_steps reaches the log with the licence names. Fails before: a traceback."""
        import os
        from types import SimpleNamespace
        closure = SimpleNamespace(targets=("validation",), tractography_algorithms=("csd_act",))
        plan = {"python_packages": [], "extensions": [], "restart_extensions": [], "manual_items": []}
        lines = []
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(os.environ, {"SLICERDSPACE_USER_CONFIG": str(Path(folder) / "user_config.json")}), \
                patch.object(setup_selected, "build_cli_plan", return_value=(closure, SimpleNamespace(), plan)), \
                patch("shared.setup_components.plan_components", return_value=[]), \
                patch("shared.selected_setup.setup_steps", side_effect=LicenceNotAccepted("No licence is listed for: fsl_terms")):
            code = setup_selected.main(["--targets", "validation", "--execute"], log=lines.append)
        self.assertEqual(code, 2)
        self.assertTrue(any("fsl_terms" in line for line in lines))


class LicenceTitleTests(unittest.TestCase):
    def test_ob3_the_cli_does_not_call_an_os_permission_error_a_licence_problem(self):
        """OB3: an OS PermissionError from the setup steps is not logged as a licence problem. Fails at 0064d04."""
        import os
        from types import SimpleNamespace
        closure = SimpleNamespace(targets=("validation",), tractography_algorithms=("csd_act",))
        plan = {"python_packages": [], "extensions": [], "restart_extensions": [], "manual_items": []}
        lines = []
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(os.environ, {"SLICERDSPACE_USER_CONFIG": str(Path(folder) / "user_config.json")}), \
                patch.object(setup_selected, "build_cli_plan", return_value=(closure, SimpleNamespace(), plan)), \
                patch("shared.setup_components.plan_components", return_value=[]), \
                patch("shared.selected_setup.setup_steps", side_effect=PermissionError(13, "Permission denied", "/x/log.txt")):
            with self.assertRaises(PermissionError):  # it keeps its own path: no exit code 2, no licence text
                setup_selected.main(["--targets", "validation", "--execute"], log=lines.append)
        self.assertFalse(any("Licence problem" in line for line in lines))

    def test_ob3_the_real_licence_checks_raise_the_licence_error(self):
        from shared.selected_setup import setup_steps
        from shared.setup_components import LicenceNotAccepted
        self.assertTrue(issubclass(LicenceNotAccepted, PermissionError))
        plan = {"python_packages": ["numpy"], "extensions": [], "restart_extensions": [], "manual_items": []}
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(LicenceNotAccepted):
            setup_steps(plan, {}, folder=folder)


class MigrationEmptiesSettingsTests(unittest.TestCase):
    def test_a5_n4_a_migration_that_keeps_every_user_path_is_not_a_failed_step(self):
        """A5 N4: completed receipt, no settings left, a migration note: nothing to save, no failure. Fails before: ValueError."""
        from shared.selected_setup import validate_setup_receipt
        receipt = {"kind": "slicerdsurfer_runtime_setup", "status": "completed", "external_dependencies": {},
                   "migration": {"old_prefix": "/old", "repointed": [], "kept_user_paths": ["mrtrix_bin_dir"], "note": "n"}}
        self.assertEqual(validate_setup_receipt({"id": "managed_runtime"}, receipt), {})
        del receipt["migration"]  # an empty receipt without a migration is still wrong
        with self.assertRaises(ValueError):
            validate_setup_receipt({"id": "managed_runtime"}, receipt)


if __name__ == "__main__":
    unittest.main()
