"""GPU preference, runtime isolation and explicit CPU choices, without GPU inference."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import Mock, patch

from shared.brain_mask import BrainMaskConfig, brain_mask_config_from_options
from shared.environment import resolve_auto_torch_device
from shared.pnl_mask_runner import select_device as pnl_device, main as pnl_main


class GPUExecutionPolicyTests(unittest.TestCase):
    def fake_torch(self, available):
        torch = Mock()
        torch.__version__ = "test-double"
        torch.version.cuda = "12.1" if available else None
        torch.cuda.is_available.return_value = available
        torch.cuda.get_device_capability.return_value = (9, 0)
        torch.cuda.get_arch_list.return_value = ["sm_90"]
        return torch

    def test_pnl_auto_prefers_gpu_and_preserves_caller_visibility(self):
        torch = self.fake_torch(True)
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3,5"}):
            result = pnl_device("auto", torch)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "3,5")
        self.assertEqual(result["resolved"], "cuda")
        self.assertEqual(result["framework"], "pytorch")
        torch.empty.assert_called_once_with(1, device="cuda:0")

    def test_pnl_explicit_cpu_does_not_probe_gpu(self):
        torch = self.fake_torch(True)
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}):
            self.assertEqual(pnl_device("cpu", torch)["resolved"], "cpu")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "-1")
        torch.cuda.is_available.assert_not_called()

    def test_pnl_no_gpu_auto_uses_cpu_but_explicit_cuda_fails(self):
        self.assertEqual(pnl_device("auto", self.fake_torch(False))["resolved"], "cpu")
        with self.assertRaisesRegex(RuntimeError, "no visible CUDA GPU"):
            pnl_device("cuda", self.fake_torch(False))

    def test_pnl_gpu_configuration_errors_do_not_fall_back(self):
        torch = self.fake_torch(True)
        torch.empty.side_effect = RuntimeError("CUDA initialization failure")
        with self.assertRaisesRegex(RuntimeError, "CUDA initialization"):
            pnl_device("auto", torch)

    def test_pnl_launcher_executes_standalone_mask_without_source_rewriting(self):
        import json
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            script, record = root / "pnl-brain-mask", root / "record.json"
            script.write_text("# console script fixture; no model inference")
            with patch("shared.pnl_mask_runner.runpy.run_path") as execute, patch.object(sys, "argv", ["pnl-runner"]), patch.dict(os.environ):
                pnl_main(["--device", "cpu", "--record", str(record), "--script", str(script),
                          "--", "mask", "--input", "b0.nii.gz"])
                self.assertEqual(sys.argv, [str(script), "mask", "--input", "b0.nii.gz",
                                           "--device", "cpu", "--record", str(record)])
            execute.assert_called_once_with(str(script), run_name="__main__")
            self.assertEqual(json.loads(record.read_text())["framework"], "pytorch")
            self.assertEqual(script.read_text(), "# console script fixture; no model inference")

    def test_torch_auto_gpu_cpu_and_probe_failure(self):
        for available, count, expected in ((True, 2, "cuda:0"), (False, 0, "cpu")):
            with patch("shared.environment._check_cuda_device", return_value=(available, {"device_count": count})) as probe:
                python = Path("/external/runtime/python")
                self.assertEqual(resolve_auto_torch_device("auto", python), expected)
                probe.assert_called_once_with("cuda:0", python, ())
        with patch("shared.environment._check_cuda_device") as probe:
            self.assertEqual(resolve_auto_torch_device("cpu"), "cpu")
            probe.assert_not_called()
        with patch("shared.environment._check_cuda_device", return_value=(False, {"error": "probe timed out"})):
            with self.assertRaisesRegex(RuntimeError, "no silent CPU fallback"):
                resolve_auto_torch_device("auto")

    def test_public_inference_defaults_prefer_gpu(self):
        from anatomy.segmentation.config import DDSegConfig
        from anatomy.surface.pipeline import SurfaceConfig
        with patch.dict(sys.modules, {"nrrd": ModuleType("nrrd")}):
            from anatomy.parcellation.pipeline import DDParcelConfig
        from tractography.tractcloud import TractCloudConfig
        for config in (DDSegConfig, SurfaceConfig, DDParcelConfig, TractCloudConfig):
            self.assertEqual(config.__dataclass_fields__["device"].default, "auto")
        self.assertEqual(BrainMaskConfig().brain_mask_device, "auto")
        self.assertEqual(brain_mask_config_from_options(SimpleNamespace(brain_mask_device="cuda")).brain_mask_device, "cuda")

    def test_tractcloud_auto_is_not_resolved_in_the_callers_python(self):
        from shared.orchestrator import _resolved_tractcloud_device_for_execution
        with patch("shared.orchestrator.resolve_auto_torch_device", side_effect=AssertionError("Wrong runtime")):
            self.assertEqual(_resolved_tractcloud_device_for_execution("auto", execute=True, target_will_run=True), "auto")

    def test_tractcloud_rejects_gpu_failure_cpu_retry(self):
        from tractography.tractcloud import TRACTCLOUD_COMPAT_BOOTSTRAP
        package = ModuleType("tractcloud")
        pipeline = ModuleType("tractcloud.pipeline")
        pipeline.run_inference = Mock(return_value="prediction")
        original = pipeline.run_inference
        package.pipeline = pipeline
        torch = self.fake_torch(True)
        with patch.dict(sys.modules, {"tractcloud": package, "tractcloud.pipeline": pipeline, "torch": torch}), \
                patch("runpy.run_module"), patch.object(sys, "argv", ["tractcloud", "--device", "auto"]), \
                patch.dict(os.environ, {"SLICERDSPACE_TRACTCLOUD_DEVICE_RECORD": ""}):
            exec(TRACTCLOUD_COMPAT_BOOTSTRAP, {})
            self.assertEqual(sys.argv[-1], "cuda")
            self.assertEqual(pipeline.run_inference(None, None, None, 3, "cuda:0"), "prediction")
            with self.assertRaisesRegex(RuntimeError, "automatic CPU retry disabled"):
                pipeline.run_inference(None, None, None, 3, "cpu")
        self.assertEqual(original.call_count, 1)

    def test_invalid_devices_are_rejected(self):
        for select, runtime in ((pnl_device, self.fake_torch(False)),):
            with self.assertRaises(ValueError):
                select("invalid", runtime)

    def test_cli_defaults_and_explicit_devices(self):
        from shared.run_subject_pipeline_cli import build_parser
        parser = build_parser()
        defaults = parser.parse_args([])
        for key in ("brain_mask_device", "ddseg_device", "surface_device", "tractcloud_device"):
            self.assertEqual(getattr(defaults, key), "auto")
        for value in ("auto", "cpu", "cuda"):
            self.assertEqual(parser.parse_args(["--brain-mask-device", value]).brain_mask_device, value)

    def test_eddy_gpu_success_cannot_fall_back_to_cpu_on_failure(self):
        from shared.mrtrix_preproc_runner import configure_eddy
        for gpu_visible, cuda_binary, expected in ((True, "eddy_cuda12.1", "cuda"), (False, "eddy_cuda12.1", "cpu"), (True, "", "cpu")):
            with self.subTest(gpu_visible=gpu_visible, cuda_binary=cuda_binary):
                fsl = SimpleNamespace(eddy_binary=lambda cuda: cuda_binary if cuda else "eddy_cpu")
                record = configure_eddy(fsl, gpu_visible)
                self.assertEqual(record["resolved"], expected)
                self.assertFalse(record["cuda_error_cpu_retry"])
                self.assertEqual(fsl.eddy_binary(False), "" if expected == "cuda" else "eddy_cpu")

    def test_eddy_cuda_threads_are_one_but_cpu_keeps_requested_threads(self):
        from shared.mrtrix_preproc_runner import normalize_eddy_threads
        for options in ("--cnr_maps --nthr=8 --slm=linear", "--nthr 4 --residuals"):
            argv = ["in.nii.gz", "out.nii.gz", "-eddy_options", options]
            self.assertEqual(normalize_eddy_threads(argv, "cpu"), argv)
            effective = normalize_eddy_threads(argv, "cuda")
            self.assertIn("--nthr=1", effective[-1])
            self.assertNotIn("--nthr=8", effective[-1])
            self.assertEqual(argv[-1], options)

    def test_installing_cuda_eddy_invalidates_cpu_preprocessing_cache(self):
        from preprocessing.pipeline import DwiPreprocessingConfig, _preprocessing_cache_key
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            eddy = root / "eddy"
            eddy.write_text("same wrapper before and after CUDA installation")
            raw = SimpleNamespace(dwi_path=root / "data.nii.gz", bval_path=root / "bvals",
                                  bvec_path=root / "bvecs", mask_path=root / "mask.nii.gz")
            config = DwiPreprocessingConfig("subject", root, brain_mask_backend="provided")
            backend = {"backend": "mrtrix_fsl", "commands": {"eddy": str(eddy)}}
            before = _preprocessing_cache_key(raw, config, backend)
            (root / "eddy_cuda").write_text("new official CUDA executable")
            after = _preprocessing_cache_key(raw, config, backend)
            self.assertNotEqual(before, after)
            self.assertIn("eddy_cuda", after["eddy_binaries"])

    def test_surface_supplied_mask_bypasses_the_cpu_only_vendor_helper(self):
        from shared.pipeline import RawInputBundle
        from anatomy.surface.pipeline import SurfaceConfig, _raw_command
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("dwi", "bval", "bvec", "mask"):
                (root / name).touch()
            raw = RawInputBundle(root, "x", root, root / "dwi", root / "bval", root / "bvec", root / "mask")
            command, _ = _raw_command(SurfaceConfig("x", root, root, input_bundle=raw,
                auto_mask=True, device="cuda:0", slicer_home=root),
                {"checkpoint_root": root, "template_dir": root, "registration_atlas": root}, root)
            self.assertEqual(command[command.index("--mask") + 1], root / "mask")
            self.assertEqual(command[command.index("--device") + 1], "cuda:0")


if __name__ == "__main__":
    unittest.main()
