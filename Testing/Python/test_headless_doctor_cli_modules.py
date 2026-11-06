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


def _doctor(config: EnvironmentDoctorConfig, *, load=None, version="1.12.0", slicer_process=True):
    """Run the doctor with Teem loading faked: ``load`` is a callable that returns a library or raises.

    ``slicer_process`` says whether the running interpreter counts as Slicer's (F1): only then does the doctor
    load Teem in process. Otherwise it probes the PythonSlicer that it resolves from the settings.
    """
    spec = _native_spec()
    load = load or (lambda: _FakeLibrary())
    with patch("shared.environment.build_dependency_specs", return_value=(spec,)), \
         patch("shared.environment._check_spec", return_value=_available(spec)), \
         patch("shared.environment.sys.executable", "/slicer/bin/PythonSlicer" if slicer_process else sys.executable), \
         patch("shared.teem_dti.load_teem", side_effect=load), \
         patch("shared.teem_dti.teem_version", return_value=version):
        return run_environment_doctor(config)


def _missing_library():
    raise TeemError("Slicer's Teem library was not found or not usable.\n  Searched roots: /fake/Slicer\n"
                    "  File names: libteem.so, libteem.so.1")


class NativeTensorFitDoctorTests(unittest.TestCase):
    config = EnvironmentDoctorConfig(
        targets=("microstructure",), repo_root=Path("."), check_slicer_cli_modules=True)
    def setUp(self) -> None:
        # T5: a machine with Slicer installed must not start a real PythonSlicer from the common install paths.
        common = patch("shared.environment._common_slicer_executables", return_value=())
        common.start()
        self.addCleanup(common.stop)

    old_keys = ("slicer_cli:dwitodtiestimation", "slicer_cli:diffusiontensorscalarmeasurements",
                "slicer_module:dwitodtiestimation")

    def test_d6_missing_teem_is_one_blocker_with_the_pythonslicer_hint(self) -> None:
        report = _doctor(self.config, load=_missing_library)
        statuses = {item.key: item for item in report.statuses}
        item = statuses["native_tensor_fit"]
        self.assertEqual(item.label, "Native tensor fit (Slicer core Teem)")
        self.assertEqual((item.kind, item.status, item.required, item.targets), ("setting", "missing", True, ("microstructure",)))
        self.assertIn("Searched roots: /fake/Slicer", item.details["error"])  # D12: the message reaches the report
        self.assertIn("File names: libteem.so", item.details["error"])
        # N1: the reason is in the panel text (notes) and in the install notes (install_hint), not a generic hint.
        self.assertIn("Searched roots: /fake/Slicer", item.details["notes"])
        self.assertIn("Searched roots: /fake/Slicer", item.install_hint)
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

    @staticmethod
    def _slicer_tree(root: Path) -> Path:
        pythonslicer = root / "Slicer-5.10.0-linux-amd64" / "bin" / "PythonSlicer"
        pythonslicer.parent.mkdir(parents=True)
        pythonslicer.write_text("#!/bin/sh\n")
        (pythonslicer.parent.parent / "Slicer").write_text("#!/bin/sh\n")
        return pythonslicer

    def test_d6_check_runs_in_the_selected_pythonslicer(self) -> None:
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
            report = _doctor(config, load=caller_runtime, slicer_process=False)
        item = {item.key: item for item in report.statuses}["native_tensor_fit"]
        self.assertEqual(calls[0][0], selected.as_posix())
        self.assertEqual((item.status, item.details["library"]), ("available", "/selected/libteem.so"))
        self.assertEqual(item.details["python_executable"], selected.as_posix())

    def test_n2_plain_python_option_with_a_resolvable_pythonslicer_is_not_a_blocker(self) -> None:
        """N2: --python-executable = plain python; the fit runs in PythonSlicer, so the PythonSlicer probe decides."""
        payload = {"teem_version": "1.12.0", "library": "/slicer/libteem.so", "error": ""}
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

        with tempfile.TemporaryDirectory() as tmp:
            pythonslicer = self._slicer_tree(Path(tmp).resolve())
            plain = Path(tmp) / "venv" / "bin" / "python"
            config = EnvironmentDoctorConfig(
                targets=("microstructure",), repo_root=Path("."), check_slicer_cli_modules=True,
                python_executable=plain, slicer_executable=pythonslicer.parent.parent / "Slicer")
            with patch("shared.environment.subprocess.run", side_effect=fake_run):
                report = _doctor(config, slicer_process=False,
                                 load=lambda: (_ for _ in ()).throw(AssertionError("no in-process probe")))
        item = {item.key: item for item in report.statuses}["native_tensor_fit"]
        self.assertEqual(report.blocker_count, 0)
        self.assertEqual(calls[0][0], pythonslicer.as_posix())
        self.assertEqual(item.details["python_executable"], pythonslicer.as_posix())
        self.assertNotEqual(item.details["python_executable"], plain.as_posix())

    def test_f3_plain_python_never_reports_available_from_its_own_teem_load(self) -> None:
        """F3: SLICER_HOME / a Teem override lets a plain Python load Teem; only the PythonSlicer probe counts."""
        failure = {"teem_version": "", "library": "", "error": "Slicer's Teem library was not found\n  Searched roots: /s"}
        with tempfile.TemporaryDirectory() as tmp:
            pythonslicer = self._slicer_tree(Path(tmp).resolve())
            home = pythonslicer.parent.parent
            config = EnvironmentDoctorConfig(
                targets=("microstructure",), repo_root=Path("."), check_slicer_cli_modules=True,
                python_executable=Path(tmp) / "venv" / "bin" / "python")
            with patch.dict("os.environ", {"SLICER_HOME": str(home)}), \
                 patch("shared.environment.subprocess.run",
                       return_value=subprocess.CompletedProcess([], 0, stdout=json.dumps(failure), stderr="")):
                # The plain Python could load a stub Teem here; the doctor must not ask it.
                report = _doctor(config, slicer_process=False, load=_FakeLibrary)
        item = {item.key: item for item in report.statuses}["native_tensor_fit"]
        self.assertEqual(item.status, "missing")
        self.assertEqual(item.details["python_executable"], pythonslicer.as_posix())
        self.assertIn("Searched roots: /s", item.details["notes"])

    def test_n1_probe_failure_in_pythonslicer_shows_the_reason_not_the_generic_hint(self) -> None:
        payload = {"teem_version": "", "library": "", "error": "Slicer's Teem library was not found\n  Searched roots: /x/Slicer"}
        selected = Path(tempfile.gettempdir()) / "ps" / "bin" / "PythonSlicer"
        config = EnvironmentDoctorConfig(targets=("microstructure",), repo_root=Path("."),
                                         check_slicer_cli_modules=True, python_executable=selected)
        with patch("shared.environment.subprocess.run",
                   return_value=subprocess.CompletedProcess([], 0, stdout=json.dumps(payload), stderr="")):
            report = _doctor(config, slicer_process=False)
        item = {item.key: item for item in report.statuses}["native_tensor_fit"]
        self.assertIn("Searched roots: /x/Slicer", item.details["notes"])
        self.assertIn("Searched roots: /x/Slicer", item.install_hint)
        self.assertNotIn("Run with Slicer's PythonSlicer", item.install_hint)

    def test_n1_without_any_pythonslicer_the_hint_asks_for_one(self) -> None:
        config = EnvironmentDoctorConfig(targets=("microstructure",), repo_root=Path("."),
                                         check_slicer_cli_modules=True,
                                         python_executable=Path(tempfile.gettempdir()) / "plain" / "bin" / "python")
        with patch.dict("os.environ", {}, clear=True), \
             patch("shared.environment._common_slicer_executables", return_value=()), \
             patch("shared.environment.subprocess.run", side_effect=AssertionError("nothing to probe")):
            report = _doctor(config, slicer_process=False)
        item = {item.key: item for item in report.statuses}["native_tensor_fit"]
        self.assertEqual(item.status, "missing")
        self.assertEqual(item.install_hint, "Run with Slicer's PythonSlicer")
        self.assertIn("PythonSlicer was not found", item.details["notes"])
        self.assertEqual(item.details["python_executable"], "")

    def test_d6_probe_failure_is_a_status_not_a_traceback(self) -> None:
        selected = Path(tempfile.gettempdir()) / "broken-slicer" / "bin" / "PythonSlicer"
        config = EnvironmentDoctorConfig(targets=("microstructure",), repo_root=Path("."),
                                         check_slicer_cli_modules=True, python_executable=selected)
        with patch("shared.environment.subprocess.run", side_effect=OSError("cannot execute")):
            report = _doctor(config, slicer_process=False)
        item = {item.key: item for item in report.statuses}["native_tensor_fit"]
        self.assertEqual(item.status, "missing")
        self.assertIn("cannot execute", item.details["error"])

    def test_d6_slicer_process_keeps_the_in_process_check(self) -> None:
        config = EnvironmentDoctorConfig(targets=("microstructure",), repo_root=Path("."),
                                         check_slicer_cli_modules=True, python_executable=Path(sys.executable))
        with patch("shared.environment.subprocess.run", side_effect=AssertionError("no subprocess")):
            report = _doctor(config)
        self.assertEqual({item.key: item for item in report.statuses}["native_tensor_fit"].status, "available")

    def test_f1_configured_venv_does_not_move_the_teem_self_check_to_the_venv(self) -> None:
        """F1 (D10): the recorded interpreter is the running Slicer one, so requests_for_report targets only it."""
        from shared.external_process import interpreter_path
        from shared.setup_selftest import requests_for_report

        venv_python = Path(tempfile.gettempdir()) / "user-venv" / "bin" / "python"
        config = EnvironmentDoctorConfig(targets=("microstructure",), repo_root=Path("."),
                                         check_slicer_cli_modules=True, python_executable=venv_python)
        with patch("shared.environment.subprocess.run", side_effect=AssertionError("no subprocess")):
            report = _doctor(config, slicer_process=True)
        item = {item.key: item for item in report.statuses}["native_tensor_fit"]
        slicer_python = interpreter_path("/slicer/bin/PythonSlicer").as_posix()  # the fake running Slicer interpreter
        self.assertEqual(item.details["python_executable"], slicer_python)
        numpy = SimpleNamespace(key="python:numpy", kind="python", required=True, status="available",
                                targets=("microstructure",),
                                details={"import": "numpy", "python_executable": venv_python.as_posix()})
        flags = {request["python"]: request.get("teem_self_check", False)
                 for request in requests_for_report(SimpleNamespace(statuses=[item, numpy], targets=("microstructure",)))}
        self.assertEqual(flags, {slicer_python: True, venv_python.as_posix(): False})

    def test_d6_gui_session_checks_teem_in_process_and_drops_the_module_requirement(self) -> None:
        gui = EnvironmentDoctorConfig(targets=("microstructure",), repo_root=Path("."), loaded_slicer_modules=())
        with patch("shared.environment.subprocess.run", side_effect=AssertionError("no subprocess")):
            report = _doctor(gui)
        keys = [item.key for item in report.statuses]
        self.assertEqual(keys[-1], "native_tensor_fit")
        for key in self.old_keys:
            self.assertNotIn(key, keys)

    def test_t1_a_conda_env_named_slicer_never_loads_teem_in_process(self) -> None:
        from shared.environment import _native_tensor_status

        conda_python = Path(tempfile.gettempdir()) / "envs" / "slicer-dspace" / "bin" / "python"
        config = EnvironmentDoctorConfig(targets=("microstructure",), repo_root=Path("."),
                                         check_slicer_cli_modules=True)
        with patch("shared.environment.sys.executable", conda_python.as_posix()), \
             patch("shared.environment._teem_info_in_process", side_effect=AssertionError("no in-process Teem")), \
             patch("shared.environment._doctor_pythonslicer", return_value=None):
            status = _native_tensor_status(("microstructure",), config)
        self.assertEqual(status.status, "missing")
        self.assertIn("PythonSlicer was not found", status.details["notes"])

    def test_t1_slicer_interpreter_names_include_windows_and_ignore_case(self) -> None:
        from shared.external_process import is_slicer_python

        self.assertTrue(is_slicer_python("C:/Program Files/Slicer 5.10.0/bin/PythonSlicer.exe"))
        self.assertTrue(is_slicer_python("C:/Program Files/Slicer 5.10.0/bin/python-real.exe"))
        self.assertTrue(is_slicer_python("/opt/slicer/bin/PYTHONSLICER"))
        self.assertFalse(is_slicer_python("/opt/tools/bin/python"))

    def test_t1_a_slicer_interpreter_name_takes_the_in_process_branch(self) -> None:
        from shared.environment import _native_tensor_status

        for name in ("PythonSlicer", "pythonslicer.exe", "python-real", "Python-Real.exe"):
            with self.subTest(name=name), \
                 patch("shared.environment.sys.executable", f"/slicer/bin/{name}"), \
                 patch("shared.environment._teem_info_in_process",
                       return_value={"teem_version": "1.12.0", "library": "/lib/libteem.so", "error": ""}) as probe:
                self.assertEqual(_native_tensor_status(("microstructure",)).status, "available")
                probe.assert_called_once()

    def test_t2_the_doctor_probes_the_pythonslicer_that_the_fit_resolves(self) -> None:
        from shared.environment import _doctor_pythonslicer

        with tempfile.TemporaryDirectory() as tmp:
            trees = {}
            for name in ("A", "B"):
                root = Path(tmp).resolve() / name
                (root / "bin").mkdir(parents=True)
                (root / "lib").mkdir()
                (root / "bin" / "PythonSlicer").touch()
                (root / "Slicer").touch()
                trees[name] = root
            # The old doctor search follows the configured launcher (tree B); the fit resolver follows SLICER_HOME (tree A).
            config = EnvironmentDoctorConfig(targets=("microstructure",), repo_root=Path("."),
                                             check_slicer_cli_modules=True,
                                             slicer_executable=trees["B"] / "Slicer")
            with patch.dict("os.environ", {"SLICER_HOME": trees["A"].as_posix()}), \
                 patch("shared.pipeline._require_slicer"), \
                 patch("shared.pipeline.slicer", None):
                found = _doctor_pythonslicer(config)
        self.assertEqual(found.resolve(), (trees["A"] / "bin" / "PythonSlicer").resolve())

    def test_r3_the_call_site_resolver_comes_before_a_configured_pythonslicer(self) -> None:
        from shared.environment import _doctor_pythonslicer

        with tempfile.TemporaryDirectory() as tmp:
            trees = {}
            for name in ("A", "B"):
                root = Path(tmp).resolve() / name
                (root / "bin").mkdir(parents=True)
                (root / "lib").mkdir()
                (root / "bin" / "PythonSlicer").touch()
                trees[name] = root
            config = EnvironmentDoctorConfig(targets=("microstructure",), repo_root=Path("."),
                                             check_slicer_cli_modules=True,
                                             python_executable=trees["B"] / "bin" / "PythonSlicer")
            with patch.dict("os.environ", {"SLICER_HOME": trees["A"].as_posix()}), \
                 patch("shared.pipeline._require_slicer"), \
                 patch("shared.pipeline.slicer", None):
                found = _doctor_pythonslicer(config)
        self.assertEqual(found.resolve(), (trees["A"] / "bin" / "PythonSlicer").resolve())

    def test_t2_without_a_resolver_result_the_doctor_search_still_applies(self) -> None:
        from shared.environment import _doctor_pythonslicer

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve() / "B"
            (root / "bin").mkdir(parents=True)
            (root / "bin" / "PythonSlicer").touch()
            (root / "Slicer").touch()
            config = EnvironmentDoctorConfig(targets=("microstructure",), repo_root=Path("."),
                                             check_slicer_cli_modules=True, slicer_executable=root / "Slicer")
            with patch("shared.pipeline.resolve_pythonslicer_executable", side_effect=RuntimeError("no Slicer")):
                found = _doctor_pythonslicer(config)
        self.assertEqual(found.resolve(), (root / "bin" / "PythonSlicer").resolve())

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
