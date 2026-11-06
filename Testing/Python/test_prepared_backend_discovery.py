"""Prepared-source discovery without installs, network access or real algorithms."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import backend_sources
from tractography import tractcloud, wma, wmql


class PreparedBackendDiscoveryTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.cache = self.root / "prepared"
        self.start_patch(mock.patch.dict(os.environ, {}, clear=True))
        self.start_patch(mock.patch("shared.backend_sources.source_root", return_value=self.cache))
        self.which = self.start_patch(mock.patch("shutil.which", return_value=None))
        self.available = self.start_patch(mock.patch(
            "shared.backend_sources.available_source", wraps=backend_sources.available_source))
        self.packages = self.start_patch(mock.patch(
            "tractography.wmql._probe_python_package",
            return_value={"tract_querier": True, "numpy": True, "nibabel": True}))

    def start_patch(self, patcher):
        result = patcher.start()
        self.addCleanup(patcher.stop)
        return result

    def source(self, profile, directory=None):
        """Only cache fixtures have receipts; custom checkouts stay unmanaged."""
        spec = backend_sources.SOURCES[profile]
        prepared = directory is None
        directory = directory or backend_sources.profile_path(profile, self.cache)
        files = {spec["entry"]: "# synthetic entry point\n"}
        if profile == "tractcloud":
            files["src/tractcloud/__init__.py"] = ""
        elif profile == "wmql":
            files["tract_querier/__init__.py"] = ""
            files["tract_querier/data/prepared.qry"] = "prepared |= 16\n"
        else:
            files.update({"bin/" + name: "# synthetic command\n" for name in wma.REQUIRED_WMA_COMMANDS})
        for relative, contents in files.items():
            path = directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents, encoding="utf-8")
            path.chmod(0o755)
        if prepared:
            receipt = dict(profile=profile, revision=spec["revision"], archive_sha256=spec["sha256"],
                           files={relative: backend_sources.file_hash(directory / relative) for relative in files})
            (directory / ".slicerdspace-source.json").write_text(json.dumps(receipt), encoding="utf-8")
        return directory

    def wmql_config(self, **kwargs):
        query = self.root / "custom.qry"
        query.write_text("import prepared.qry\n", encoding="utf-8")
        tracks, atlas = self.root / "input.vtk", self.root / "atlas.nii.gz"
        tracks.touch()
        atlas.touch()
        return wmql.WMQLConfig("synthetic", tracks, atlas, self.root / "wmql-output",
                               query_path=query, python_executable=Path(sys.executable), **kwargs)

    def test_tractcloud_uses_verified_cache(self):
        prepared = self.source("tractcloud")
        self.assertEqual(tractcloud.resolve_tractcloud_root(), prepared)
        self.available.assert_called_once_with("tractcloud")

    def test_tractcloud_explicit_root_precedes_environment_and_cache(self):
        self.source("tractcloud")
        custom = self.source("tractcloud", self.root / "custom")
        with mock.patch.dict(os.environ, {"SLICERDSPACE_TRACTCLOUD_ROOT": str(self.root / "missing")}):
            self.assertEqual(tractcloud.resolve_tractcloud_root(custom), custom)
        self.available.assert_not_called()

    def test_tractcloud_environment_precedence(self):
        self.source("tractcloud")
        canonical = self.source("tractcloud", self.root / "canonical")
        legacy = self.source("tractcloud", self.root / "legacy")
        for environment, expected in (({"TRACTCLOUD_ROOT": str(legacy)}, legacy),
                ({"TRACTCLOUD_ROOT": str(legacy), "SLICERDSPACE_TRACTCLOUD_ROOT": str(canonical)}, canonical)):
            with self.subTest(environment=environment), mock.patch.dict(os.environ, environment):
                self.assertEqual(tractcloud.resolve_tractcloud_root(), expected)
        self.available.assert_not_called()

    def test_tractcloud_invalid_custom_root_does_not_fall_back(self):
        self.source("tractcloud")
        missing = self.root / "missing"
        with self.assertRaises(FileNotFoundError):
            tractcloud.resolve_tractcloud_root(missing)
        with mock.patch.dict(os.environ, {"TRACTCLOUD_ROOT": str(missing)}):
            with self.assertRaises(FileNotFoundError):
                tractcloud.resolve_tractcloud_root()
        self.available.assert_not_called()

    def test_tractcloud_probe_execution_and_manifest_use_cached_source(self):
        prepared = self.source("tractcloud")
        tracks, data = self.root / "input.vtp", self.root / "data"
        tracks.touch()
        data.mkdir()

        def environment(_python, extra_env=None):
            return {"PYTHONPATH": "retained", **(extra_env or {})}

        with mock.patch("tractography.tractcloud.python_tool_environment", side_effect=environment), \
                mock.patch("tractography.tractcloud.subprocess.run", return_value=subprocess.CompletedProcess(
                    [], 0, stdout='{"tractcloud": true, "torch": true}')) as run:
            result = tractcloud.run_tractcloud_parcellation(tractcloud.TractCloudConfig(
                "synthetic", tracks, self.root / "tractcloud-output", python_executable=Path(sys.executable),
                data_dir=data, execute=True))
        source_path = (prepared / "src").as_posix()
        self.assertEqual(result.backend_status["source"]["root"], prepared.as_posix())
        self.assertEqual(len(run.call_args_list), 3)
        for call in run.call_args_list[:2]:
            self.assertEqual(call.kwargs["env"]["PYTHONPATH"], source_path + os.pathsep + "retained")
        self.assertEqual(run.call_args.kwargs["env"]["PYTHONPATH"], source_path)
        manifest = json.loads(result.manifest_path.read_text())
        self.assertEqual(manifest["tractcloud_root"], prepared.as_posix())

    def test_wmql_cache_drives_probe_command_and_query_includes(self):
        prepared = self.source("wmql")
        with mock.patch("tractography.wmql._requires_windows_query_staging", return_value=False):
            command = wmql.build_wmql_command(self.wmql_config())
        self.assertEqual(command[:2], (Path(sys.executable).absolute().as_posix(),
                                      (prepared / "scripts/cli_tract_querier.py").as_posix()))
        self.packages.assert_called_once_with(Path(sys.executable).absolute(), prepared)
        self.assertIn((prepared / "tract_querier/data").as_posix(), command[command.index("-I") + 1])
        self.available.assert_called_once_with("wmql")

    def test_wmql_cached_root_reaches_execution_and_query_snapshot(self):
        prepared = self.source("wmql")
        with mock.patch("tractography.wmql.python_tool_environment", return_value={"PYTHONPATH": "retained"}), \
                mock.patch("tractography.wmql.subprocess.run", return_value=subprocess.CompletedProcess([], 0)) as run:
            result = wmql.run_wmql_bundle_segmentation(self.wmql_config(execute=True))
        self.assertEqual(result.backend_status["root"], prepared.as_posix())
        self.assertEqual(run.call_args.args[0][1], (prepared / "scripts/cli_tract_querier.py").as_posix())
        self.assertEqual(run.call_args.kwargs["env"]["PYTHONPATH"], prepared.as_posix() + os.pathsep + "retained")
        manifest = json.loads(result.manifest_path.read_text())
        self.assertIn((prepared / "tract_querier/data/prepared.qry").as_posix(), manifest["query_dependencies"])
        self.assertTrue((Path(manifest["staged_query"]).parent / "prepared.qry").is_file())
        self.available.assert_called_once_with("wmql")

    def test_wmql_cached_root_supports_windows_query_staging(self):
        prepared = self.source("wmql")
        with mock.patch("tractography.wmql._requires_windows_query_staging", return_value=True), \
                mock.patch("tractography.wmql.python_tool_environment", return_value={}), \
                mock.patch("tractography.wmql.subprocess.run", return_value=subprocess.CompletedProcess([], 0)) as run:
            result = wmql.run_wmql_bundle_segmentation(self.wmql_config(execute=True))
        manifest = json.loads(result.manifest_path.read_text())
        query_directory = Path(manifest["staged_query"]).parent
        self.assertNotIn("-I", result.command)
        self.assertEqual(run.call_args.kwargs["cwd"], query_directory)
        self.assertTrue((query_directory / "prepared.qry").is_file())
        self.assertEqual(run.call_args.kwargs["env"]["PYTHONPATH"].split(os.pathsep)[0], prepared.as_posix())
        self.available.assert_called_once_with("wmql")

    def test_wmql_explicit_root_precedes_environment_and_cache(self):
        self.source("wmql")
        custom = self.source("wmql", self.root / "custom")
        with mock.patch.dict(os.environ, {"TRACT_QUERIER_ROOT": str(self.root / "missing")}):
            status = wmql.inspect_wmql_backend(tract_querier_root=custom)
        self.assertEqual(status["root"], custom.as_posix())
        self.packages.assert_called_once_with(Path(sys.executable).absolute(), custom)
        self.available.assert_not_called()

    def test_wmql_environment_root_reaches_command_and_execution(self):
        self.source("wmql")
        custom = self.source("wmql", self.root / "custom")
        for key in ("SLICERDSPACE_TRACT_QUERIER_ROOT", "TRACT_QUERIER_ROOT"):
            with self.subTest(key=key), mock.patch.dict(os.environ, {key: str(custom)}), \
                    mock.patch("tractography.wmql.python_tool_environment", return_value={}), \
                    mock.patch("tractography.wmql.subprocess.run", return_value=subprocess.CompletedProcess([], 0)) as run:
                result = wmql.run_wmql_bundle_segmentation(self.wmql_config(execute=True))
                self.assertEqual(result.backend_status["root"], custom.as_posix())
                self.assertEqual(result.command[1], (custom / "scripts/cli_tract_querier.py").as_posix())
                self.assertEqual(run.call_args.kwargs["env"]["PYTHONPATH"].split(os.pathsep)[0], custom.as_posix())
        self.available.assert_not_called()

    def test_wmql_canonical_environment_root_precedes_legacy(self):
        canonical = self.source("wmql", self.root / "canonical")
        legacy = self.source("wmql", self.root / "legacy")
        with mock.patch.dict(os.environ, {"SLICERDSPACE_TRACT_QUERIER_ROOT": str(canonical),
                                          "TRACT_QUERIER_ROOT": str(legacy)}):
            self.assertEqual(wmql.inspect_wmql_backend()["root"], canonical.as_posix())
        self.available.assert_not_called()

    def test_wmql_configured_and_path_executables_precede_cache(self):
        self.source("wmql")
        executable = self.root / "tract_querier"
        executable.touch()
        for selection in ("explicit", "SLICERDSPACE_WMQL_EXECUTABLE", "TRACT_QUERIER_EXECUTABLE", "PATH"):
            with self.subTest(selection=selection), mock.patch.dict(os.environ, {}, clear=True):
                self.which.return_value = str(executable) if selection == "PATH" else None
                kwargs = {"tract_querier_executable": executable} if selection == "explicit" else {}
                if selection not in {"explicit", "PATH"}:
                    os.environ[selection] = str(executable)
                status = wmql.inspect_wmql_backend(**kwargs)
                self.assertIsNone(status["root"])
                self.assertEqual(status["commands"]["tract_querier_executable"], executable.as_posix())
                self.assertIsNone(self.packages.call_args.args[1])
        self.available.assert_not_called()

    def test_wmql_invalid_explicit_selections_do_not_fall_back(self):
        self.source("wmql")
        missing = self.root / "missing"
        for kwargs, environment in (({"tract_querier_root": missing}, {}),
                ({}, {"TRACT_QUERIER_ROOT": str(missing)}),
                ({"tract_querier_executable": missing}, {}),
                ({}, {"TRACT_QUERIER_EXECUTABLE": str(missing)})):
            with self.subTest(kwargs=kwargs, environment=environment), mock.patch.dict(os.environ, environment):
                self.assertFalse(wmql.inspect_wmql_backend(**kwargs)["available"])
        self.available.assert_not_called()

    def test_wma_posix_cache_is_used_for_inspection_and_command(self):
        prepared = self.source("wma")
        tools = self.root / 'display-tools'
        tools.mkdir()
        for name in ('xvfb-run', 'Xvfb', 'xauth'):
            (tools / name).touch()
        self.which.side_effect = lambda name: str(tools / name) if (tools / name).is_file() else None
        for system in ("linux", "darwin"):
            with self.subTest(system=system), mock.patch("tractography.wma.sys.platform", system):
                status = wma.inspect_wma_backend()
                self.assertTrue(status["available"])
                self.assertEqual(status["root"], prepared.as_posix())
                command = wma.build_wma_org_atlas_command(wma.WMAOrgAtlasConfig(
                    "synthetic", self.root / "input.vtk", self.root / "atlas", self.root / "out",
                    self.root / "Slicer"))
                self.assertEqual(command[0], (prepared / wma.WMA_APPLY_ORG_SCRIPT).as_posix())
                self.assertEqual(command[command.index('-x') + 1], '1' if system == 'linux' else '0')

    def test_wma_explicit_root_precedes_environment_path_and_cache(self):
        self.source("wma")
        custom = self.source("wma", self.root / "custom")
        with mock.patch.dict(os.environ, {"WMA_ROOT": str(self.root / "missing")}):
            self.assertEqual(wma.resolve_wma_root(custom), custom)
        self.which.assert_not_called()
        self.available.assert_not_called()

    def test_wma_environment_precedes_path_and_cache(self):
        self.source("wma")
        canonical = self.source("wma", self.root / "canonical")
        legacy = self.source("wma", self.root / "legacy")
        for environment, expected in (({"WMA_ROOT": str(legacy)}, legacy),
                ({"WMA_ROOT": str(legacy), "SLICERDSPACE_WMA_ROOT": str(canonical)}, canonical)):
            with self.subTest(environment=environment), mock.patch.dict(os.environ, environment):
                self.assertEqual(wma.resolve_wma_root(), expected)
        self.which.assert_not_called()
        self.available.assert_not_called()

    def test_wma_path_precedes_cache(self):
        self.source("wma")
        custom = self.source("wma", self.root / "path-install")
        self.which.return_value = str(custom / wma.WMA_APPLY_ORG_SCRIPT)
        self.assertEqual(wma.resolve_wma_root(), custom)
        self.available.assert_not_called()

    def test_wma_invalid_custom_root_does_not_fall_back(self):
        self.source("wma")
        missing = self.root / "missing"
        with self.assertRaises(FileNotFoundError):
            wma.resolve_wma_root(missing)
        with mock.patch.dict(os.environ, {"WMA_ROOT": str(missing)}):
            with self.assertRaises(FileNotFoundError):
                wma.resolve_wma_root()
        self.available.assert_not_called()

    def test_native_windows_wma_never_discovers_bash_only_cache(self):
        self.source("wma")
        with mock.patch("tractography.wma.sys.platform", "win32"):
            with self.assertRaises(FileNotFoundError):
                wma.resolve_wma_root()
        self.available.assert_not_called()

    def test_native_windows_wma_keeps_explicit_environment_and_path_bridges(self):
        self.source("wma")
        custom = self.source("wma", self.root / "bridge")
        bridge = custom / (str(wma.WMA_APPLY_ORG_SCRIPT) + ".exe")
        bridge.touch()
        with mock.patch("tractography.wma.sys.platform", "win32"):
            self.assertEqual(wma.resolve_wma_root(custom), custom)
            with mock.patch.dict(os.environ, {"WMA_ROOT": str(custom)}):
                self.assertEqual(wma.resolve_wma_root(), custom)
            self.which.return_value = str(bridge)
            self.assertEqual(wma.resolve_wma_root(), custom)
            self.assertEqual(wma._wma_script_path(custom), bridge)
        self.available.assert_not_called()

    def test_absent_and_tampered_caches_are_not_consumed(self):
        with mock.patch("tractography.wma.sys.platform", "linux"):
            self.assertIsNone(tractcloud.resolve_tractcloud_root())
            self.assertIsNone(wmql.inspect_wmql_backend()["root"])
            with self.assertRaises(FileNotFoundError):
                wma.resolve_wma_root()
            for profile in ("tractcloud", "wmql", "wma"):
                prepared = self.source(profile)
                (prepared / backend_sources.SOURCES[profile]["entry"]).write_text("tampered\n", encoding="utf-8")
            self.assertIsNone(tractcloud.resolve_tractcloud_root())
            self.assertIsNone(wmql.inspect_wmql_backend()["root"])
            with self.assertRaises(FileNotFoundError):
                wma.resolve_wma_root()


if __name__ == "__main__":
    unittest.main()
