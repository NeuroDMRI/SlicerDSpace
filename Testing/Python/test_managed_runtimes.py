"""Managed setup contracts without network access, Qt, CUDA or real installs."""

from contextlib import nullcontext
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from shared import managed_runtimes as runtime
from shared.ui_managed_runtimes import apply_runtime_result, asset_profiles_for_items


ORIGINAL_CONDA_SUBDIR = runtime.conda_subdir


class ManagedRuntimeTests(unittest.TestCase):
    def setUp(self):
        # The suite runs on any host; these contracts are the Linux x86_64 route.
        for patcher in (patch.object(runtime.platform, 'system', return_value='Linux'),
                        patch.object(runtime, 'host_machine', return_value='x86_64')):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_pnl_native_image_test_must_succeed_before_a_ready_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp)
            (prefix / 'bin').mkdir()
            (prefix / 'conda-meta').mkdir()
            for name in runtime.PROFILES['pnl_cnn']['commands']:
                command = prefix / 'bin' / name
                command.write_text('# synthetic executable fixture\n')
                command.chmod(0o755)
            for package in runtime.PROFILES['pnl_cnn']['packages']:
                name, version = package.split('=')[:2]
                (prefix / 'conda-meta' / (name + '.json')).write_text(json.dumps(dict(name=name, version=version, subdir='linux-64')))
            with patch.object(runtime.subprocess, 'run') as run:
                result = runtime.verify_prefix('pnl_cnn', prefix)
                self.assertEqual(result['functional_checks'], ['ants:synthetic_image_roundtrip'])
                self.assertEqual(Path(run.call_args_list[1].args[0][0]).name, 'ImageMath')
                self.assertEqual(len(run.call_args_list), 3)
            import subprocess
            with patch.object(runtime.subprocess, 'run', side_effect=[None, subprocess.CalledProcessError(1, 'ImageMath')]):
                with self.assertRaises(subprocess.CalledProcessError):
                    runtime.verify_prefix('pnl_cnn', prefix)

    def test_plan_has_no_filesystem_side_effect_and_only_selected_profiles(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "uncreated"
            plan = runtime.install_plan(("mrtrix",), root)
            self.assertFalse(root.exists())
            self.assertEqual(plan["profiles"], ["mrtrix"])
            self.assertFalse(plan["fsl_license_required"])
            self.assertFalse(plan["changes_slicer_python"])
            self.assertFalse(plan["changes_shell_or_driver"])
            self.assertEqual(len(plan["bootstrap_sha256"]), 64)

    def test_profile_validation_and_platform(self):
        for profiles in ((), ("unknown",), ("fsl_cpu", "fsl_cuda")):
            with self.assertRaises(ValueError):
                runtime.install_plan(profiles)
        self.assertTrue(runtime.supported_platform("Linux", "x86_64"))
        for system, arch in (("Windows", "AMD64"), ("Darwin", "arm64"), ("Linux", "aarch64")):
            self.assertFalse(runtime.supported_platform(system, arch))

    def test_fsl_permission_gate_precedes_all_download_and_directory_changes(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(runtime, "supported_platform", return_value=True), \
                patch.object(runtime, "ensure_conda") as bootstrap:
            root = Path(temp) / "uncreated"
            with self.assertRaises(PermissionError):
                runtime.provision(("fsl_cuda",), root=root)
            bootstrap.assert_not_called()
            self.assertFalse(root.exists())

    def test_only_selected_gui_routes_install_external_tools(self):
        node = SimpleNamespace(targetPreprocessing=True, brainMaskBackend="provided")
        self.assertEqual(runtime.selected_profiles(node, True), ("mrtrix", "fsl_cuda"))
        self.assertEqual(runtime.selected_profiles(node, False), ("mrtrix", "fsl_cpu"))
        node.targetPreprocessing = False
        node.targetTractography, node.orchestrationTractographyBackend = True, "csd_act"
        self.assertEqual(runtime.selected_profiles(node, True), ("mrtrix",))

    def test_gpu_visibility_respects_user_restriction(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "-1"}), patch("subprocess.run") as probe:
            self.assertFalse(runtime.cuda_visible())
            probe.assert_not_called()

    def test_unmanaged_environment_is_never_modified(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(runtime, "supported_platform", return_value=True), \
                patch.object(runtime, "setup_lock", return_value=nullcontext()), patch.object(runtime, "ensure_conda") as bootstrap:
            root = Path(temp)
            prefix = root / runtime.PROFILES["mrtrix"]["directory"]
            prefix.mkdir()
            sentinel = prefix / "untouched.txt"
            sentinel.write_text("original")
            with self.assertRaisesRegex(RuntimeError, "unmanaged environment"):
                runtime.provision(("mrtrix",), root=root)
            bootstrap.assert_not_called()
            self.assertEqual(sentinel.read_text(), "original")

    def test_reuse_is_verified_without_installer_or_network(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(runtime, "supported_platform", return_value=True), \
                patch.object(runtime, "setup_lock", return_value=nullcontext()), patch.object(runtime, "ensure_conda") as bootstrap:
            root = Path(temp)
            prefix = root / runtime.PROFILES["mrtrix"]["directory"]
            prefix.mkdir()
            record = {"external_dependencies": {"mrtrix_bin_dir": str(prefix / "bin")}, "command_sha256": {"mrconvert": "hash"}}
            receipt = root / (prefix.name + ".json")
            receipt.write_text(json.dumps({"kind": "slicerdsurfer_managed_runtime", "profile": "mrtrix", "status": "ready", **record}))
            with patch.object(runtime, "verify_prefix", return_value=record):
                result = runtime.provision(("mrtrix",), root=root)
            self.assertEqual(result["status"], "completed")
            bootstrap.assert_not_called()
            with patch.object(runtime, "verify_prefix", return_value=dict(record, command_sha256={"mrconvert": "changed"})):
                with self.assertRaisesRegex(RuntimeError, "checksum changed"):
                    runtime.provision(("mrtrix",), root=root)

    def test_repair_reinstalls_owned_prefix_and_preserves_previous_receipt(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(runtime, "supported_platform", return_value=True), \
                patch.object(runtime, "setup_lock", return_value=nullcontext()), \
                patch.object(runtime, "ensure_conda", return_value="conda"), patch.object(runtime, "_run") as run, \
                patch.object(runtime.subprocess, "run", return_value=SimpleNamespace(stdout="lock")):
            root = Path(temp)
            prefix = root / runtime.PROFILES["mrtrix"]["directory"]
            (prefix / "conda-meta").mkdir(parents=True)
            (prefix / "conda-meta/history").write_text("owned fixture")
            old = {"kind": "slicerdsurfer_managed_runtime", "profile": "mrtrix", "status": "ready"}
            receipt = root / (prefix.name + ".json")
            receipt.write_text(json.dumps(old))
            record = {"external_dependencies": {"mrtrix_bin_dir": str(prefix / "bin")}, "command_sha256": {}}
            with patch.object(runtime, "verify_prefix", return_value=record):
                runtime.provision(("mrtrix",), root=root, repair=True)
            self.assertIn("--force-reinstall", run.call_args.args[0])
            self.assertEqual(json.loads(next(root.glob("*.before-repair-*.json")).read_text()), old)
            self.assertTrue(json.loads(receipt.read_text())["repair"])

    def test_configured_profiles_never_adopt_unmanaged_environments(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            prefix = root / runtime.PROFILES["mrtrix"]["directory"]
            config = {"mrtrix_bin_dir": str(prefix / "bin")}
            self.assertEqual(runtime.configured_profiles(config, ("mrtrix",), root), ())
            (root / (prefix.name + ".json")).write_text("{}")
            self.assertEqual(runtime.configured_profiles(config, ("mrtrix",), root), ("mrtrix",))
            self.assertEqual(runtime.configured_profiles({"mrtrix_bin_dir": "/other/bin"}, ("mrtrix",), root), ())

    def test_gui_does_not_apply_failed_partial_or_unknown_settings(self):
        owner = Mock()
        for result in ({"status": "failed"}, {"kind": "slicerdsurfer_runtime_setup", "status": "completed", "external_dependencies": {"python_executable": "/wrong"}}):
            with self.assertRaises(ValueError):
                apply_runtime_result(owner, result)
        owner._save_environment_path_setting.assert_not_called()

    def test_model_downloads_only_include_requested_product_profiles(self):
        self.assertEqual(asset_profiles_for_items((SimpleNamespace(targets=("segmentation",)),)), ("segmentation",))
        self.assertEqual(asset_profiles_for_items((SimpleNamespace(targets=("surface", "parcellation")),)), ("surface", "parcellation"))

    def test_gui_saves_verified_paths_and_invalidates_environment_cache(self):
        with tempfile.TemporaryDirectory() as temp, \
                patch.dict("os.environ", {"SLICERDSPACE_USER_CONFIG": str(Path(temp) / "user_config.json")}):
            owner = Mock(_environmentPathLineEdits={"mrtrix_bin_dir": Mock()})
            apply_runtime_result(owner, {"kind": "slicerdsurfer_runtime_setup", "status": "completed",
                                         "external_dependencies": {"mrtrix_bin_dir": temp}})
            owner._save_environment_path_setting.assert_called_once_with("mrtrix_bin_dir", temp)
            owner._environmentPathLineEdits["mrtrix_bin_dir"].setText.assert_called_once_with(temp)
            self.assertIsNone(owner._lastEnvironmentResult)

    def test_bootstrap_retry_repairs_only_plugin_owned_prefix(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(runtime.shutil, "which", return_value=None), \
                patch.object(runtime, "file_hash", return_value=runtime.MINIFORGE_SHA256):
            root = Path(temp)
            bootstrap = root / "miniforge-26.7.2-0-linux-64"
            bootstrap.mkdir()
            (root / "Miniforge3-26.7.2-0-Linux-x86_64.sh").write_text("test fixture, not executed")
            with self.assertRaisesRegex(RuntimeError, "unmanaged bootstrap"):
                runtime.ensure_conda(root)
            (root / "miniforge-26.7.2-0-linux-64.bootstrap.json").write_text(json.dumps(
                {"kind": "slicerdsurfer_miniforge_bootstrap", "prefix": str(bootstrap), "subdir": "linux-64"}))

            def installed(argv):
                self.assertIn("-u", argv)
                (bootstrap / "bin").mkdir()
                (bootstrap / "bin/conda").write_text("test fixture")

            with patch.object(runtime, "_run", side_effect=installed):
                self.assertEqual(runtime.ensure_conda(root), str(bootstrap / "bin/conda"))


if __name__ == "__main__":
    unittest.main()
