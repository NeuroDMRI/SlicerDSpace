"""The environment doctor checks the native tensor fit (Slicer core Teem), not SlicerDMRI (D6, D12)."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from shared import run_subject_pipeline_cli as cli
from shared.environment import DependencySpec, EnvironmentDoctorConfig, run_environment_doctor
from shared.teem_dti import TeemError


def _native_spec(targets=("microstructure",)) -> DependencySpec:
    return DependencySpec(key="python:slicer.util", kind="python", label="slicer.util", required=True, targets=targets)


def _available(spec: DependencySpec):
    return SimpleNamespace(
        key=spec.key, kind=spec.kind, label=spec.label, status="available", required=True,
        targets=spec.targets, details={}, install_group="slicer-runtime", install_hint="")


class _FakeLibrary:
    _slicerdspace_path = "/slicer/lib/Teem-1.12.0/libteem.so"


def _doctor(config: EnvironmentDoctorConfig, *, load=None, version="1.12.0"):
    """Run the doctor with Teem loading faked: ``load`` is a callable that returns a library or raises."""
    spec = _native_spec()
    load = load or (lambda: _FakeLibrary())
    with patch("shared.environment.build_dependency_specs", return_value=(spec,)), \
         patch("shared.environment._check_spec", return_value=_available(spec)), \
         patch("shared.teem_dti.load_teem", side_effect=load), \
         patch("shared.teem_dti.teem_version", return_value=version):
        return run_environment_doctor(config)


def _missing_library():
    raise TeemError("Slicer's Teem library was not found or not usable.\n  Searched roots: /fake/Slicer\n"
                    "  File names: libteem.so, libteem.so.1")


class NativeTensorFitDoctorTests(unittest.TestCase):
    config = EnvironmentDoctorConfig(
        targets=("microstructure",), repo_root=Path("."), check_slicer_cli_modules=True)
    old_keys = ("slicer_cli:dwitodtiestimation", "slicer_cli:diffusiontensorscalarmeasurements",
                "slicer_module:dwitodtiestimation")

    def test_d6_missing_teem_is_one_blocker_with_the_pythonslicer_hint(self) -> None:
        report = _doctor(self.config, load=_missing_library)
        statuses = {item.key: item for item in report.statuses}
        item = statuses["native_tensor_fit"]
        self.assertEqual(item.label, "Native tensor fit (Slicer core Teem)")
        self.assertEqual((item.kind, item.status, item.required, item.targets), ("setting", "missing", True, ("microstructure",)))
        self.assertEqual(item.install_hint, "Run with Slicer's PythonSlicer")
        self.assertIn("Searched roots: /fake/Slicer", item.details["error"])  # D12: the message reaches the report
        self.assertIn("File names: libteem.so", item.details["error"])
        for key in self.old_keys:
            self.assertNotIn(key, statuses)
        self.assertEqual(report.blocker_count, 1)

    def test_d6_found_teem_records_version_and_library(self) -> None:
        report = _doctor(self.config)
        item = {item.key: item for item in report.statuses}["native_tensor_fit"]
        self.assertEqual(item.status, "available")
        self.assertEqual(item.details["teem_version"], "1.12.0")
        self.assertEqual(item.details["library"], _FakeLibrary._slicerdspace_path)
        self.assertEqual(report.blocker_count, 0)

    def test_d6_check_runs_in_the_selected_interpreter(self) -> None:
        selected = Path(tempfile.gettempdir()) / "selected-slicer" / "bin" / "PythonSlicer"
        payload = {"teem_version": "1.12.0", "library": "/selected/libteem.so", "error": ""}
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

        def caller_runtime():
            raise AssertionError("the caller's runtime must not load Teem")

        config = EnvironmentDoctorConfig(targets=("microstructure",), repo_root=Path("."),
                                         check_slicer_cli_modules=True, python_executable=selected)
        with patch("shared.environment.subprocess.run", side_effect=fake_run):
            report = _doctor(config, load=caller_runtime)
        item = {item.key: item for item in report.statuses}["native_tensor_fit"]
        self.assertEqual(calls[0][0], selected.as_posix())
        self.assertEqual((item.status, item.details["library"]), ("available", "/selected/libteem.so"))

    def test_d6_selected_interpreter_without_teem_reports_its_message(self) -> None:
        selected = Path(tempfile.gettempdir()) / "plain-venv" / "bin" / "python"
        payload = {"teem_version": "", "library": "", "error": "Slicer's Teem library was not found\n  Searched roots: (none)"}
        config = EnvironmentDoctorConfig(targets=("microstructure",), repo_root=Path("."),
                                         check_slicer_cli_modules=True, python_executable=selected)
        with patch("shared.environment.subprocess.run",
                   return_value=subprocess.CompletedProcess([], 0, stdout=json.dumps(payload), stderr="")):
            report = _doctor(config)
        item = {item.key: item for item in report.statuses}["native_tensor_fit"]
        self.assertEqual(item.status, "missing")
        self.assertIn("Searched roots: (none)", item.details["error"])
        self.assertEqual(item.install_hint, "Run with Slicer's PythonSlicer")

    def test_d6_probe_failure_is_a_status_not_a_traceback(self) -> None:
        selected = Path(tempfile.gettempdir()) / "broken-venv" / "bin" / "python"
        config = EnvironmentDoctorConfig(targets=("microstructure",), repo_root=Path("."),
                                         check_slicer_cli_modules=True, python_executable=selected)
        with patch("shared.environment.subprocess.run", side_effect=OSError("cannot execute")):
            report = _doctor(config)
        item = {item.key: item for item in report.statuses}["native_tensor_fit"]
        self.assertEqual(item.status, "missing")
        self.assertIn("cannot execute", item.details["error"])

    def test_d6_same_interpreter_keeps_the_in_process_check(self) -> None:
        config = EnvironmentDoctorConfig(targets=("microstructure",), repo_root=Path("."),
                                         check_slicer_cli_modules=True, python_executable=Path(sys.executable))
        with patch("shared.environment.subprocess.run", side_effect=AssertionError("no subprocess")):
            report = _doctor(config)
        self.assertEqual({item.key: item for item in report.statuses}["native_tensor_fit"].status, "available")

    def test_d6_gui_session_checks_teem_in_process_and_drops_the_module_requirement(self) -> None:
        gui = EnvironmentDoctorConfig(targets=("microstructure",), repo_root=Path("."), loaded_slicer_modules=())
        with patch("shared.environment.subprocess.run", side_effect=AssertionError("no subprocess")):
            report = _doctor(gui)
        keys = [item.key for item in report.statuses]
        self.assertEqual(keys[-1], "native_tensor_fit")
        for key in self.old_keys:
            self.assertNotIn(key, keys)

    def test_d6_default_config_does_not_check_teem(self) -> None:
        plain = EnvironmentDoctorConfig(targets=("microstructure",), repo_root=Path("."))
        report = _doctor(plain, load=lambda: (_ for _ in ()).throw(AssertionError("Teem check must not run")))
        self.assertNotIn("native_tensor_fit", [item.key for item in report.statuses])

    def test_cli_doctor_config_enables_the_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            args = cli.build_parser().parse_args(["--subject-root", str(Path(tmp) / "sub-01"), "--no-user-config"])
            config = cli._environment_config_from_args(args, targets=("microstructure",), output_root=None)
        self.assertTrue(config.check_slicer_cli_modules)


if __name__ == "__main__":
    unittest.main()
