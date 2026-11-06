from __future__ import annotations

import os
import tempfile
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import shared.pipeline as pipeline


class SharedPipelineWindowsCompatibilityTest(unittest.TestCase):
    def test_itk_respects_explicit_cpu_budget_without_disabling_cuda(self):
        with patch.dict(os.environ, {'OMP_NUM_THREADS': '4', 'CUDA_VISIBLE_DEVICES': '1'}, clear=True):
            env = pipeline.build_cli_environment('DWIToDTIEstimation')
        self.assertEqual(env['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'], '4')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '1')

    def test_explicit_itk_user_override_is_preserved(self):
        with patch.dict(os.environ, {'OMP_NUM_THREADS': '4', 'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS': '16'}, clear=True):
            self.assertEqual(pipeline.build_cli_environment('CLI')['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'], '16')

    def test_no_unrequested_thread_cap_or_invalid_budget_is_invented(self):
        for value in ('', '0', '-1', 'not-an-integer'):
            with patch.dict(os.environ, {'OMP_NUM_THREADS': value}, clear=True):
                self.assertNotIn('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', pipeline.build_cli_environment('CLI'))

    def test_macos_app_root_and_launcher_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = Path(tmp).resolve() / "Slicer.app"
            home = app / "Contents"
            (home / "lib/Slicer-5.10").mkdir(parents=True)
            (home / "MacOS").mkdir()
            (home / "bin").mkdir()
            (home / "MacOS/Slicer").touch()
            (home / "bin/PythonSlicer").touch()
            fake = SimpleNamespace(app=SimpleNamespace(slicerHome=str(app)))
            with patch.object(pipeline, "SLICER_IMPORT_ERROR", None), patch.object(pipeline, "slicer", fake):
                self.assertEqual(pipeline._resolve_slicer_home(), home)
                self.assertEqual(Path(pipeline.resolve_launcher_executable()), home / "MacOS/Slicer")
                self.assertEqual(Path(pipeline.resolve_pythonslicer_executable()), home / "bin/PythonSlicer")

    def test_pythonslicer_resolves_outside_slicer_from_the_interpreter_or_the_environment(self):
        # A-C1: the resolver no longer needs an imported ``slicer`` module.
        error = ImportError("no slicer")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            interpreter = root / "tool" / "PythonSlicer"
            interpreter.parent.mkdir()
            interpreter.touch()
            home = root / "Slicer-5.10"
            (home / "bin").mkdir(parents=True)
            (home / "bin/PythonSlicer").touch()
            configured = root / "Configured"
            (configured / "bin").mkdir(parents=True)
            (configured / "bin/PythonSlicer").touch()
            other = root / "venv" / "bin" / "python"
            other.parent.mkdir(parents=True)
            other.touch()
            outside = (patch.object(pipeline, "SLICER_IMPORT_ERROR", error), patch.object(pipeline, "slicer", None))
            with outside[0], outside[1]:
                # 1. sys.executable named PythonSlicer wins, even when SLICER_HOME names another install.
                with patch.object(sys, "executable", str(interpreter)), \
                        patch.dict(os.environ, {"SLICER_HOME": str(home)}, clear=True):
                    self.assertEqual(Path(pipeline.resolve_pythonslicer_executable()), interpreter)
                # 2. SLICER_HOME, then the configured Slicer path.
                with patch.object(sys, "executable", str(other)), \
                        patch.dict(os.environ, {"SLICER_HOME": str(home), "SLICER_PATH": str(configured)}, clear=True):
                    self.assertEqual(Path(pipeline.resolve_pythonslicer_executable()), home / "bin/PythonSlicer")
                with patch.object(sys, "executable", str(other)), \
                        patch.dict(os.environ, {"DTI_SLICER_PATH": str(configured)}, clear=True):
                    self.assertEqual(Path(pipeline.resolve_pythonslicer_executable()), configured / "bin/PythonSlicer")
                # 3. Nothing found: the error names what to set.
                with patch.object(sys, "executable", str(other)), patch.dict(os.environ, {}, clear=True):
                    with self.assertRaisesRegex(RuntimeError, "Set SLICER_HOME"):
                        pipeline.resolve_pythonslicer_executable()

    def test_pythonslicer_inside_slicer_keeps_the_slicer_home_order(self):
        # A-C1: inside Slicer the interpreter and environment are not consulted before slicerHome.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            running, environment = root / "running", root / "environment"
            for home in (running, environment):
                (home / "bin").mkdir(parents=True)
                (home / "bin/PythonSlicer").touch()
            fake = SimpleNamespace(app=SimpleNamespace(slicerHome=str(running)))
            with patch.object(pipeline, "SLICER_IMPORT_ERROR", None), patch.object(pipeline, "slicer", fake), \
                    patch.dict(os.environ, {"SLICER_HOME": str(environment)}, clear=True):
                self.assertEqual(Path(pipeline.resolve_pythonslicer_executable()), running / "bin/PythonSlicer")

    def test_cli_library_variable_matches_platform(self):
        with tempfile.TemporaryDirectory() as tmp:
            executable = Path(tmp) / "cli-modules/CLI"
            executable.parent.mkdir()
            for platform, variable in (("win32", "PATH"), ("darwin", "DYLD_LIBRARY_PATH"), ("linux", "LD_LIBRARY_PATH")):
                with self.subTest(platform=platform), patch.object(sys, "platform", platform):
                    env = pipeline.build_cli_environment(str(executable))
                    self.assertIn(str(executable.parent), env[variable])

    def test_macos_extension_cli_finds_slicer_core_libraries(self):
        # UKFTractography 5.12 on macOS links libITK*-5.4.1.dylib by build-machine path.
        with tempfile.TemporaryDirectory() as tmp:
            contents = Path(tmp).resolve() / "Slicer.app/Contents"
            core = contents / "lib/Slicer-5.12"
            executable = contents / "Extensions-34645/UKFTractography/lib/Slicer-5.12/cli-modules/UKFTractography"
            core.mkdir(parents=True)
            executable.parent.mkdir(parents=True)
            executable.touch()
            for platform, variable, expected in (("darwin", "DYLD_LIBRARY_PATH", True), ("linux", "LD_LIBRARY_PATH", False)):
                with self.subTest(platform=platform), patch.object(sys, "platform", platform), \
                        patch.dict(os.environ, {}, clear=True):
                    entries = pipeline.build_cli_environment(str(executable))[variable].split(os.pathsep)
                    self.assertEqual(str(core) in entries, expected)
                    self.assertEqual(entries[0], str(executable.parent))

    def test_c5_dyld_fallback_only_for_clis_that_belong_to_the_running_slicer(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            contents = base / "Slicer.app/Contents"
            core = contents / "lib/Slicer-5.12"
            core.mkdir(parents=True)
            (contents / "bin").mkdir()
            python = contents / "bin/PythonSlicer"
            python.touch()
            standalone = base / "tools/ukf/bin/UKFTractography"
            standalone.parent.mkdir(parents=True)
            standalone.touch()
            (base / "tools/ukf/lib/Slicer-5.12").mkdir(parents=True)  # the tool's own copy
            inside = contents / "Extensions-34645/UKF/lib/Slicer-5.12/cli-modules/UKFTractography"
            inside.parent.mkdir(parents=True)
            inside.touch()
            user = base / "user/slicer.org/Extensions-34645/UKF/lib/Slicer-5.12/cli-modules/UKFTractography"
            user.parent.mkdir(parents=True)
            user.touch()
            # A folder above the .app that also holds lib/Slicer-*: the walk must stop at Contents.
            (base / "lib/Slicer-9.9").mkdir(parents=True)
            with patch.object(sys, "platform", "darwin"), patch.object(sys, "executable", str(python)), \
                    patch.dict(os.environ, {}, clear=True):
                def entries(cli: Path) -> list[str]:
                    return pipeline.build_cli_environment(str(cli))["DYLD_LIBRARY_PATH"].split(os.pathsep)

                self.assertNotIn(str(core), entries(standalone))
                self.assertIn(str(core), entries(inside))
                self.assertIn(str(core), entries(user))
                self.assertNotIn(str(base / "lib/Slicer-9.9"), entries(inside))

    def test_cli_in_an_additional_factory_search_path_is_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp).resolve() / 'home'
            extra = Path(tmp).resolve() / 'external-cli'
            home.mkdir()
            extra.mkdir()
            executable = extra / 'DWIToDTIEstimation'
            executable.touch()
            factory = SimpleNamespace(searchPaths=[str(extra)])
            app = SimpleNamespace(moduleManager=lambda: SimpleNamespace(factoryManager=lambda: factory))
            pipeline._cli_search_dirs.cache_clear()
            try:
                with patch.object(pipeline, 'SLICER_IMPORT_ERROR', None), \
                     patch.object(pipeline, 'slicer', SimpleNamespace(modules=SimpleNamespace(), app=app)), \
                     patch.object(pipeline, '_resolve_slicer_home', return_value=home):
                    self.assertEqual(Path(pipeline.resolve_cli_executable('dwitodtiestimation')), executable)
            finally:
                pipeline._cli_search_dirs.cache_clear()

    def test_cli_fallback_selects_exe_and_not_xml_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cli_dir = Path(tmp)
            (cli_dir / "DWIToDTIEstimation.xml").write_text("<executable />", encoding="utf-8")
            executable = cli_dir / "DWIToDTIEstimation.exe"
            executable.write_bytes(b"")
            with (
                patch.object(pipeline, "SLICER_IMPORT_ERROR", None),
                patch.object(pipeline, "slicer", SimpleNamespace(modules=SimpleNamespace())),
                patch.object(pipeline, "_cli_search_dirs", return_value=(cli_dir,)),
            ):
                self.assertEqual(Path(pipeline.resolve_cli_executable("DWIToDTIEstimation")), executable)

    def test_windows_launcher_fallbacks_accept_exe_suffix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            slicer_home = Path(tmp)
            slicer_launcher = slicer_home / "Slicer.exe"
            python_slicer = slicer_home / "bin" / "PythonSlicer.exe"
            python_slicer.parent.mkdir(parents=True)
            slicer_launcher.write_bytes(b"")
            python_slicer.write_bytes(b"")
            with (
                patch.object(pipeline, "SLICER_IMPORT_ERROR", None),
                patch.object(pipeline, "slicer", SimpleNamespace()),
                patch.object(pipeline, "_resolve_slicer_home", return_value=slicer_home),
            ):
                self.assertEqual(Path(pipeline.resolve_launcher_executable()), slicer_launcher)
                self.assertEqual(Path(pipeline.resolve_pythonslicer_executable()), python_slicer)


if __name__ == "__main__":
    unittest.main()
