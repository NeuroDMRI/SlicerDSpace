"""Batch 4b task W6: dependency conflicts and remaining hooks (D3, D1, D4, D5, D8, F3 hook, H8 sites, plan warnings)."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
POSIX = os.name == "posix"


REAL_POPEN = subprocess.Popen


def _make_leaver(directory: Path) -> Path:
    """A command whose direct child exits at once and leaves a grandchild that holds the output pipe."""
    script = directory / "leaver.py"
    script.write_text(textwrap.dedent("""
        import subprocess, sys, time
        subprocess.Popen([sys.executable, "-c", "import time; time.sleep(6)"])
        print("hello from the child", flush=True)
    """))
    return script


# ---------------------------------------------------------------- D3
class D3OpenMpOrderTests(unittest.TestCase):
    def test_d3_main_calls_the_torch_preload_first_and_the_import_does_not(self) -> None:
        # P-b: the preload moved from import time into main(), so the GUI process (which imports this module
        # through shared.experiments) does not load torch. main() runs it before any DIPY/scikit-learn import.
        tree = ast.parse((REPO_ROOT / "shared" / "run_subject_pipeline_cli.py").read_text(encoding="utf-8"))
        for node in tree.body:
            self.assertFalse(
                isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and getattr(node.value.func, "id", "") == "_preload_torch_for_openmp",
                "P-b: the preload must not run at import time")
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        body = [stmt for stmt in main.body if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant))]
        first = body[0]
        self.assertTrue(
            isinstance(first, ast.Expr) and isinstance(first.value, ast.Call)
            and getattr(first.value.func, "id", "") == "_preload_torch_for_openmp",
            "D3: main() must call _preload_torch_for_openmp() as its first statement")

    def test_d3_preload_runs_only_on_darwin_and_never_sets_the_duplicate_lib_flag(self) -> None:
        import shared.run_subject_pipeline_cli as cli

        fake_torch = SimpleNamespace(__name__="torch")
        with patch.dict(os.environ, {"SLICERDSPACE_SKIP_DEQUARANTINE": "1"}), \
                patch.dict(sys.modules, {"torch": fake_torch}):
            os.environ.pop("KMP_DUPLICATE_LIB_OK", None)
            with patch.object(cli.sys, "platform", "linux"):
                self.assertFalse(cli._preload_torch_for_openmp())
            with patch.object(cli.sys, "platform", "darwin"):
                self.assertTrue(cli._preload_torch_for_openmp())
            self.assertNotIn("KMP_DUPLICATE_LIB_OK", os.environ)

    def test_d3_preload_survives_a_missing_torch(self) -> None:
        import shared.run_subject_pipeline_cli as cli

        with patch.dict(os.environ, {"SLICERDSPACE_SKIP_DEQUARANTINE": "1"}), \
                patch.dict(sys.modules, {"torch": None}), patch.object(cli.sys, "platform", "darwin"):
            self.assertFalse(cli._preload_torch_for_openmp())

    @unittest.skipUnless(sys.platform == "darwin", "the OpenMP clash is a macOS problem")
    def test_d3_running_main_loads_torch_before_dipy_or_sklearn(self) -> None:
        probe = textwrap.dedent("""
            import importlib.abc, json, sys
            order = []
            class Recorder(importlib.abc.MetaPathFinder):
                def find_spec(self, name, path=None, target=None):
                    if name in ("torch", "dipy", "sklearn") and name not in order:
                        order.append(name)
                    return None
            sys.meta_path.insert(0, Recorder())
            sys.path.insert(0, %r)
            import shared.run_subject_pipeline_cli as cli
            loaded_by_import = "torch" in sys.modules
            cli.main(["--print-user-config", "--no-user-config"])
            print(json.dumps({"order": order, "torch_loaded": "torch" in sys.modules,
                              "loaded_by_import": loaded_by_import}))
        """ % str(REPO_ROOT))
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        try:
            completed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                                       timeout=240, env=env, cwd=tempfile.gettempdir())
        except subprocess.TimeoutExpired:
            self.fail("importing the CLI hung")
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertFalse(result["loaded_by_import"], "P-b: importing the CLI module must not load torch")
        try:
            import importlib.util
            torch_importable = importlib.util.find_spec("torch") is not None
        except Exception:
            torch_importable = False
        if torch_importable:
            self.assertTrue(result["torch_loaded"], completed.stderr[-500:])
            order = result["order"]
            for later in ("dipy", "sklearn"):
                if later in order:
                    self.assertLess(order.index("torch"), order.index(later))

    def test_d3_dipy_tracking_keeps_one_thread_on_macos(self) -> None:
        from tractography import dti as tract_dti

        tree = ast.parse((REPO_ROOT / "tractography" / "dti.py").read_text(encoding="utf-8"))
        found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "nbr_threads":
                found.append(node.value)
        self.assertTrue(found, "D3: the eudx_tracking call must pass nbr_threads")
        self.assertEqual(tract_dti._resolve_eudx_threads(platform_name="darwin"), 1)
        self.assertEqual(tract_dti._resolve_eudx_threads(platform_name="darwin", stage_cpus=8), 1)
        self.assertEqual(tract_dti._resolve_eudx_threads(platform_name="linux", stage_cpus=6), 6)
        self.assertEqual(tract_dti._resolve_eudx_threads(platform_name="linux", host_cpus=144), 32)
        self.assertEqual(tract_dti._resolve_eudx_threads(platform_name="win32", host_cpus=24), 0)

    def test_d3_no_dipy_nlmeans_call_remains_in_the_run_path(self) -> None:
        offenders = []
        for path in REPO_ROOT.rglob("*.py"):
            parts = set(path.relative_to(REPO_ROOT).parts)
            if parts & {"Testing", "ddsurfer_backend", ".git", "docs", "build"}:
                continue
            if "nlmeans" in path.read_text(encoding="utf-8", errors="ignore").lower():
                offenders.append(str(path.relative_to(REPO_ROOT)))
        self.assertEqual(offenders, [], "a DIPY nlmeans call needs a darwin num_threads=1 pin")


# ---------------------------------------------------------------- D1
class D1BackendBundleFirstTests(unittest.TestCase):
    def _bundle(self, root: Path) -> Path:
        site = root / "bundled-python" / "site-packages"
        (site / "numpy").mkdir(parents=True)
        return site

    def test_d1_backend_environment_puts_the_bundle_first_in_pythonpath(self) -> None:
        from anatomy.surface import backend_bridge

        with tempfile.TemporaryDirectory() as td:
            site = self._bundle(Path(td))
            with patch.dict(os.environ, {"SLICERDSPACE_BUNDLED_PYTHON_ROOT": str(site.parent)}), \
                    patch.object(backend_bridge, "is_slicer_python", return_value=True), \
                    patch.object(backend_bridge, "external_python_environment",
                                 return_value={"PYTHONPATH": os.pathsep.join(["/slicer/site-packages", "/other"])}):
                environment = backend_environment = backend_bridge.backend_environment(python_bin="/x/PythonSlicer")
        parts = environment["PYTHONPATH"].split(os.pathsep)
        self.assertEqual(parts[0], str(site.resolve()))
        self.assertEqual(parts.count(str(site.resolve())), 1)
        self.assertIn("/slicer/site-packages", parts)
        self.assertEqual(environment["SLICERDSPACE_BUNDLED_SITE_PACKAGES"].split(os.pathsep)[0], str(site.resolve()))

    def test_d1_wrapper_command_rewrites_pythonpath_after_the_launcher(self) -> None:
        from anatomy.surface import backend_bridge

        with tempfile.TemporaryDirectory() as td:
            site = self._bundle(Path(td))
            captured = {}

            def fake_popen(command, **kwargs):
                captured["command"] = command
                raise OSError("stop here")

            with patch.dict(os.environ, {"SLICERDSPACE_BUNDLED_PYTHON_ROOT": str(site.parent)}), \
                    patch.object(backend_bridge, "is_slicer_python", return_value=True), \
                    patch.object(backend_bridge.subprocess, "Popen", side_effect=fake_popen):
                with self.assertRaises(OSError):
                    backend_bridge.run_backend(["run_ddsurfer_pipeline.py"], python_bin="/x/PythonSlicer")
        code = captured["command"][2]
        self.assertIn("os.environ['PYTHONPATH']", code)
        self.assertIn("SLICERDSPACE_BUNDLED_SITE_PACKAGES", code)
        self.assertIn(repr(str(site.resolve())), code)

    @unittest.skipUnless(POSIX and (Path("/Applications/Slicer.app/Contents/bin/PythonSlicer")).is_file(),
                         "needs the PythonSlicer launcher")
    def test_d1_a_child_of_the_backend_script_sees_the_bundle_before_slicer(self) -> None:
        from anatomy.surface import backend_bridge

        launcher = "/Applications/Slicer.app/Contents/bin/PythonSlicer"
        with tempfile.TemporaryDirectory() as td:
            site = self._bundle(Path(td))
            script = Path(td) / "spawn_child.py"
            script.write_text(textwrap.dedent("""
                import json, subprocess, sys
                out = subprocess.run([sys.executable, "-c", "import json,sys; print(json.dumps(sys.path))"],
                                     capture_output=True, text=True, check=True).stdout
                print("CHILD_PATH=" + out.strip())
            """))
            lines = []
            with patch.dict(os.environ, {"SLICERDSPACE_BUNDLED_PYTHON_ROOT": str(site.parent)}):
                backend_bridge.run_backend([str(script)], python_bin=launcher, log=lines.append)
            child = next(line for line in lines if line.startswith("CHILD_PATH="))
            path = json.loads(child.split("=", 1)[1])
        bundle = str(site.resolve())
        slicer_sites = [i for i, entry in enumerate(path) if entry.endswith("site-packages") and entry != bundle]
        self.assertIn(bundle, path)
        if slicer_sites:
            self.assertLess(path.index(bundle), min(slicer_sites))


# ---------------------------------------------------------------- D4 / D8 / F3 hook
class D4TorchNumpyProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        from shared import environment

        self.env = environment
        self.env._TORCH_ABI_CACHE.clear()
        self.addCleanup(self.env._TORCH_ABI_CACHE.clear)

    def _fake_packages(self, root: Path, *, from_numpy_raises: bool, numpy_version: str = "2.4.6") -> Path:
        (root / "numpy").mkdir(parents=True)
        (root / "numpy" / "__init__.py").write_text(f"__version__ = {numpy_version!r}\ndef zeros(n):\n    return [0.0] * n\n")
        (root / "torch").mkdir(parents=True)
        body = ("def from_numpy(array):\n    raise RuntimeError('Numpy is not available')\n" if from_numpy_raises
                else "def from_numpy(array):\n    return array\n")
        (root / "torch" / "__init__.py").write_text("__version__ = '2.2.2'\n" + body)
        return root

    def test_d4_probe_reports_an_abi_mismatch_without_importing_torch_in_process(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            packages = self._fake_packages(Path(td) / "pk", from_numpy_raises=True)
            with patch.object(self.env, "_bundled_prepend_paths", return_value=[]):
                result = self.env.torch_numpy_abi_probe(Path(sys.executable), (packages,))
        self.assertFalse(result["ok"])
        self.assertTrue(result["abi_mismatch"])
        self.assertEqual(result["numpy_version"], "2.4.6")
        self.assertIn("Numpy is not available", result["error"])

    def test_d4_probe_passes_when_arrays_are_exchanged_and_caches_the_result(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            packages = self._fake_packages(Path(td) / "pk", from_numpy_raises=False)
            with patch.object(self.env, "_bundled_prepend_paths", return_value=[]):
                first = self.env.torch_numpy_abi_probe(Path(sys.executable), (packages,))
                with patch.object(self.env.subprocess, "run", side_effect=AssertionError("cached")):
                    second = self.env.torch_numpy_abi_probe(Path(sys.executable), (packages,))
        self.assertTrue(first["ok"])
        self.assertFalse(first["abi_mismatch"])
        self.assertEqual(first, second)

    def test_d4_probe_timeout_is_bounded_and_not_cached(self) -> None:
        timeout = subprocess.TimeoutExpired(["python"], self.env.TORCH_ABI_PROBE_TIMEOUT_SECONDS)
        with patch.object(self.env.subprocess, "run", side_effect=timeout) as run, \
                patch.object(self.env, "_bundled_prepend_paths", return_value=[]):
            result = self.env.torch_numpy_abi_probe(Path(sys.executable))
        self.assertEqual(run.call_args.kwargs["timeout"], 120)
        self.assertFalse(result["abi_mismatch"])
        self.assertIn("timed out", result["error"])
        self.assertEqual(self.env._TORCH_ABI_CACHE, {})

    def _statuses_for_torch(self):
        spec = self.env.DependencySpec(key="python:torch", kind="python", label="torch", required=True,
                                       targets=("segmentation",), python_import="torch", python_package="torch>=2.0.0")
        status = self.env.DependencyStatus(key="python:torch", kind="python", label="torch", status="available",
                                           required=True, targets=("segmentation",))
        return (spec,), (status,)

    def test_d4_doctor_adds_a_blocking_row_with_the_reinstall_message(self) -> None:
        specs, statuses = self._statuses_for_torch()
        probe = {"numpy_version": "2.4.6", "torch_version": "2.2.2", "ok": False, "abi_mismatch": True,
                 "error": "Numpy is not available"}
        with patch.object(self.env, "torch_numpy_abi_probe", return_value=probe), \
                patch("shared.bundled_python.activation_problem", return_value=None):
            rows = self.env._runtime_compatibility_statuses(specs, statuses)
        row = next(item for item in rows if item.key == "torch_numpy_abi")
        self.assertEqual(row.status, "missing")
        self.assertTrue(row.required)
        self.assertEqual(row.targets, ("segmentation",))
        # b2-os, decision 4: the reason and the fix are two sentences, so the row does not repeat the reason.
        self.assertEqual(row.details["notes"], "PyTorch cannot exchange arrays with NumPy 2.4.6 in this Slicer.")
        self.assertEqual(row.install_hint, "Reinstall the SlicerDSpace package for this Slicer.")

    def test_d4_doctor_adds_no_row_when_torch_and_numpy_agree(self) -> None:
        specs, statuses = self._statuses_for_torch()
        probe = {"numpy_version": "2.0.2", "torch_version": "2.2.2", "ok": True, "abi_mismatch": False, "error": None}
        with patch.object(self.env, "torch_numpy_abi_probe", return_value=probe), \
                patch("shared.bundled_python.activation_problem", return_value=None), \
                patch.object(self.env, "_bundle_runtime_numpy_floor", return_value=None):
            self.assertEqual(self.env._runtime_compatibility_statuses(specs, statuses), ())

    def test_d4_a_missing_torch_row_starts_no_probe(self) -> None:
        specs, statuses = self._statuses_for_torch()
        missing = (self.env.DependencyStatus(key="python:torch", kind="python", label="torch", status="missing",
                                             required=True, targets=("segmentation",)),)
        with patch.object(self.env, "torch_numpy_abi_probe", side_effect=AssertionError("no probe")), \
                patch("shared.bundled_python.activation_problem", return_value=None):
            self.assertEqual(self.env._runtime_compatibility_statuses(specs, missing), ())

    def test_d8_numpy_below_the_bundle_floor_blocks(self) -> None:
        specs, statuses = self._statuses_for_torch()
        probe = {"numpy_version": "1.26.4", "torch_version": "2.2.2", "ok": True, "abi_mismatch": False, "error": None}
        with patch.object(self.env, "torch_numpy_abi_probe", return_value=probe), \
                patch("shared.bundled_python.activation_problem", return_value=None), \
                patch.object(self.env, "_bundle_runtime_numpy_floor", return_value="2.0.2"):
            rows = self.env._runtime_compatibility_statuses(specs, statuses)
        row = next(item for item in rows if item.key == "slicer_numpy_floor")
        self.assertTrue(row.required)
        self.assertIn("1.26.4", row.details["notes"])  # b2-os: the reason holds the found version, the fix the floor
        self.assertIn("2.0.2", row.details["notes"])
        self.assertIn("2.0.2", row.install_hint)

    def test_d8_floor_is_read_from_python_runtime_json_and_skipped_when_absent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "bundled-python"
            (root / "site-packages").mkdir(parents=True)
            with patch.dict(os.environ, {"SLICERDSPACE_BUNDLED_PYTHON_ROOT": str(root)}):
                self.assertIsNone(self.env._bundle_runtime_numpy_floor())
                (root / "python-runtime.json").write_text(json.dumps({"slicer_numpy_minimum": "2.0.2"}))
                self.assertEqual(self.env._bundle_runtime_numpy_floor(), "2.0.2")

    def test_d8_windows_device_row_says_gpu_needs_the_pytorch_extension(self) -> None:
        spec = self.env.DependencySpec(key="device:surface:cuda:0", kind="device", label="surface device cuda:0",
                                       required=True, targets=("surface",), device="cuda:0", install_group="device")
        probe = lambda *args, **kwargs: (False, {"requested_device": "cuda:0"})
        with patch.object(self.env, "_check_cuda_device", side_effect=probe):
            status = self.env._check_spec(spec, "Windows")
        self.assertIn("PyTorch extension", status.details["windows_gpu_note"])
        self.assertIn("CUDA build of torch", status.install_hint)
        with patch.object(self.env, "_check_cuda_device", side_effect=probe):
            linux = self.env._check_spec(spec, "Linux")
        self.assertNotIn("windows_gpu_note", linux.details)

    def test_f3_hook_blocks_with_the_activation_problem_text(self) -> None:
        specs, statuses = self._statuses_for_torch()
        reason = "bundled Python stack is for Python 3.99 x86_64; install the SlicerDSpace package built for this Slicer"
        with patch("shared.bundled_python.activation_problem", return_value=reason):
            rows = self.env._runtime_compatibility_statuses(specs, ())
        row = next(item for item in rows if item.key == "bundled_python_activation")
        self.assertEqual((row.status, row.required), ("missing", True))
        # b2-os, decision 4: reason and fix are split at the last "; ".
        self.assertEqual(row.details["notes"], "bundled Python stack is for Python 3.99 x86_64.")
        self.assertEqual(row.install_hint, "Install the SlicerDSpace package built for this Slicer.")
        self.assertEqual(row.details["error"], reason)
        self.assertEqual(row.targets, ("segmentation",))

    def test_f3_hook_is_silent_without_a_problem(self) -> None:
        specs, _ = self._statuses_for_torch()
        with patch("shared.bundled_python.activation_problem", return_value=None):
            self.assertEqual(self.env._runtime_compatibility_statuses(specs, ()), ())

    def test_d4_f3_run_environment_doctor_includes_the_rows(self) -> None:
        from shared.environment import EnvironmentDoctorConfig, run_environment_doctor

        reason = "bundled Python stack is for Python 3.99 x86_64; install the SlicerDSpace package built for this Slicer"
        config = EnvironmentDoctorConfig(repo_root=REPO_ROOT, targets=("validation",))
        with patch("shared.bundled_python.activation_problem", return_value=reason), \
                patch.object(self.env, "torch_numpy_abi_probe",
                             return_value={"numpy_version": "2.4.6", "ok": False, "abi_mismatch": True}):
            result = run_environment_doctor(config)
        keys = {item.key: item for item in result.statuses}
        self.assertEqual(keys["bundled_python_activation"].status, "missing")
        self.assertGreaterEqual(result.blocker_count, 1)
        self.assertIn("validation", [t.target for t in result.target_statuses if t.status == "blocked"])


# ---------------------------------------------------------------- D5
class D5BundledShortfallTests(unittest.TestCase):
    def _bundle(self, root: Path, name: str, version: str) -> Path:
        site = root / "bundled-python" / "site-packages"
        (site / f"{name}-{version}.dist-info").mkdir(parents=True)
        (site / f"{name}-{version}.dist-info" / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
        return site

    def test_d5_request_above_the_bundled_version_does_not_run_pip(self) -> None:
        from shared import python_setup as setup

        with tempfile.TemporaryDirectory() as td:
            site = self._bundle(Path(td), "nibabel", "5.3.2")
            with patch.dict(os.environ, {"SLICERDSPACE_BUNDLED_PYTHON_ROOT": str(site.parent)}), \
                    patch.object(sys, "path", [str(site), *sys.path]), \
                    patch.object(setup, "requirement_satisfied", return_value=False), \
                    patch.object(setup.subprocess, "run") as run, \
                    patch.object(setup.importlib, "import_module", side_effect=AssertionError("no pip_ensure")):
                with self.assertRaises(RuntimeError) as caught:
                    setup.install_requirements(("nibabel>=6",))
        run.assert_not_called()
        self.assertIn("bundled nibabel 5.3.2 is older than required", str(caught.exception))
        self.assertIn("reinstall the SlicerDSpace package", str(caught.exception))

    def test_d5_request_the_bundle_satisfies_still_reaches_pip(self) -> None:
        from shared import python_setup as setup

        with tempfile.TemporaryDirectory() as td:
            site = self._bundle(Path(td), "nibabel", "5.3.2")
            with patch.dict(os.environ, {"SLICERDSPACE_BUNDLED_PYTHON_ROOT": str(site.parent)}):
                self.assertEqual(setup.bundled_shortfalls(["nibabel>=5"]), [])
                self.assertEqual(setup.bundled_shortfalls(["other-package>=1"]), [])

    def test_d5_no_bundle_means_no_shortfall(self) -> None:
        from shared import python_setup as setup

        with tempfile.TemporaryDirectory() as td, \
                patch.dict(os.environ, {"SLICERDSPACE_BUNDLED_PYTHON_ROOT": str(Path(td) / "missing")}):
            self.assertEqual(setup.bundled_shortfalls(["nibabel>=99"]), [])


# ---------------------------------------------------------------- H8 sites
@unittest.skipUnless(POSIX, "the descendant probe uses a POSIX pipe")
class H8RemainingSiteTests(unittest.TestCase):
    """A grandchild that keeps the pipe open must not hold the stage; callbacks keep their format."""

    def _popen(self, td: str):
        script = _make_leaver(Path(td))
        return lambda *args, **kwargs: REAL_POPEN(
            [sys.executable, str(script)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    def _run_fast(self, call, td):
        import time

        started = time.monotonic()
        with patch("subprocess.Popen", side_effect=self._popen(td)):
            outcome = call()
        self.assertLess(time.monotonic() - started, 5.0, "the stage waited for the grandchild")
        return outcome

    def _no_default_grace(self):
        # stream_process_lines bound its default at definition time; pass a short grace through a wrapper.
        import shared.line_stream as line_stream

        original = line_stream.stream_process_lines

        def quick(process, on_line, *, notice=None, grace_seconds=0.3):
            return original(process, on_line, notice=notice, grace_seconds=grace_seconds)

        return patch.object(line_stream, "stream_process_lines", quick)

    def test_h8_postprocess_docker_returns_after_the_child_exits(self) -> None:
        from anatomy.surface import postprocess_docker as module

        with tempfile.TemporaryDirectory() as td, self._no_default_grace():
            repo = Path(td) / "repo"
            (repo / "docker").mkdir(parents=True)
            script = repo / "docker" / "run_container.sh"
            script.write_text("#!/bin/bash\n")
            script.chmod(0o755)
            lines: list[str] = []
            self._run_fast(lambda: module.run_docker(
                "S1", "whole", repo, Path(td) / "lic", Path(td) / "in", Path(td) / "out", None,
                log_callback=lines.append), td)
        self.assertIn("hello from the child", lines)
        self.assertTrue(any("descendants still hold the output pipe" in line for line in lines))

    def test_h8_parcellation_cli_returns_after_the_child_exits(self) -> None:
        from anatomy.parcellation import pipeline as module

        lines: list[str] = []
        with tempfile.TemporaryDirectory() as td, self._no_default_grace(), \
                patch.object(module, "build_cli_environment", return_value=dict(os.environ)):
            self._run_fast(lambda: module._run_cli(["unused"], lines.append), td)
        self.assertIn("[DDParcel] hello from the child", lines)
        self.assertTrue(any(line.startswith("[DDParcel] child exited") for line in lines))

    def test_h8_segmentation_subprocess_logs_every_line_with_its_prefix(self) -> None:
        import inspect
        from anatomy.segmentation import module_logic

        source = inspect.getsource(module_logic)
        self.assertIn("stream_process_lines", source)
        self.assertNotIn("for line in process.stdout", source)

    def test_h8_mkcurve_matlab_loop_is_replaced(self) -> None:
        import inspect
        from microstructure import mkcurve_matlab

        source = inspect.getsource(mkcurve_matlab)
        self.assertIn("stream_process_lines", source)
        self.assertNotIn("for line in process.stdout", source)

    def test_h8_no_remaining_plain_stdout_loop_in_the_four_sites(self) -> None:
        for relative in ("anatomy/surface/postprocess_docker.py", "anatomy/parcellation/pipeline.py",
                         "anatomy/segmentation/module_logic.py", "microstructure/mkcurve_matlab.py"):
            text = (REPO_ROOT / relative).read_text(encoding="utf-8")
            self.assertNotIn("for line in process.stdout", text, relative)
            self.assertIn("stream_process_lines", text, relative)


# ---------------------------------------------------------------- multi_tracking plan warnings
class PlanWarningForwardingTests(unittest.TestCase):
    def test_multi_tracking_plan_forwards_child_warnings_once(self) -> None:
        from dataclasses import replace
        from shared import multi_tracking
        from shared.orchestrator import ProductOrchestratorConfig, RunPlanResult

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = ProductOrchestratorConfig(
                subject_id="S1", subject_root=root, output_root=root / "out", targets=("tractography",),
                tractography_backend="dti_det,csd_act")

            def fake_plan(child, write=True):
                texts = ("Free disk is low: 4 GiB", f"[{child.tractography_backend}] only here")
                return RunPlanResult("S1", root, root / "out", "run", child.targets, root / "plan.json", (), warnings=texts[:2])

            with patch("shared.orchestrator.plan_subject_products", side_effect=fake_plan):
                result = multi_tracking.plan(replace(config), write=False)
        self.assertIn("Free disk is low: 4 GiB", result.warnings)
        self.assertEqual(result.warnings.count("Free disk is low: 4 GiB"), 1)
        self.assertIn("[csd_act] only here", result.warnings)


if __name__ == "__main__":
    unittest.main()
