"""Regression tests for Windows CUDA hardening and performance optimizations (CUDA-1 through CUDA-7)."""

from __future__ import annotations

import ctypes
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class WindowsCudaOptimizationTests(unittest.TestCase):
    # ------------------------------------------------------------------ CUDA-1: nvidia-smi discovery, WDDM [N/A], alloc conf, TDR cap
    def test_find_nvidia_smi_windows_fallback_and_alloc_conf(self) -> None:
        from shared import hardware_adapt
        from shared.bundled_python import WINDOWS_CUDA_ALLOC_CONF as BP_ALLOC_CONF, inject_bundled_python_env
        from shared.external_process import startup_environment

        self.assertEqual(hardware_adapt.WINDOWS_CUDA_ALLOC_CONF, BP_ALLOC_CONF)
        self.assertIn("garbage_collection_threshold:0.8", BP_ALLOC_CONF)
        self.assertIn("max_split_size_mb:128", BP_ALLOC_CONF)
        self.assertNotIn("expandable_segments", BP_ALLOC_CONF)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            win_dir = root / "Windows"
            sys32 = win_dir / "System32"
            prog_dir = root / "ProgramFiles"
            nvsmi_dir = prog_dir / "NVIDIA Corporation" / "NVSMI"
            sys32.mkdir(parents=True)
            nvsmi_dir.mkdir(parents=True)
            nvsmi_exe = nvsmi_dir / "nvidia-smi.exe"
            nvsmi_exe.write_bytes(b"")

            env_patch = {
                "SystemRoot": str(win_dir),
                "ProgramFiles": str(prog_dir),
                "ProgramW6432": str(prog_dir),
            }
            with (
                patch.object(hardware_adapt.sys, "platform", "win32"),
                patch.object(hardware_adapt.shutil, "which", return_value=None),
                patch.dict(os.environ, env_patch, clear=False),
            ):
                self.assertEqual(hardware_adapt.find_nvidia_smi(), str(nvsmi_exe))
                sys32_exe = sys32 / "nvidia-smi.exe"
                sys32_exe.write_bytes(b"")
                self.assertEqual(hardware_adapt.find_nvidia_smi(), str(sys32_exe))

        child_env: dict[str, str] = {}
        with patch("shared.bundled_python.sys.platform", "win32"):
            inject_bundled_python_env(child_env, include_pythonpath=False)
        self.assertEqual(child_env.get("PYTORCH_CUDA_ALLOC_CONF"), BP_ALLOC_CONF)

        with (
            patch("shared.external_process.sys.platform", "win32"),
            patch.dict(os.environ, {"PYTORCH_CUDA_ALLOC_CONF": ""}, clear=False),
        ):
            # startup_environment preserves caller env unless key is absent
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
                senv = startup_environment()
                self.assertEqual(senv.get("PYTORCH_CUDA_ALLOC_CONF"), BP_ALLOC_CONF)

        self.assertTrue(hardware_adapt.is_cuda_oom(RuntimeError("CUBLAS_STATUS_ALLOC_FAILED")))
        self.assertTrue(hardware_adapt.is_cuda_oom(RuntimeError("cuDNN error: CUDNN_STATUS_ALLOC_FAILED")))

        with (
            patch.object(hardware_adapt.os, "name", "nt"),
            patch.object(
                hardware_adapt,
                "query_cuda_free_total_bytes",
                return_value=(40 * hardware_adapt.GIB, 48 * hardware_adapt.GIB),
            ),
        ):
            capped = hardware_adapt.adaptive_cuda_scaled_batch(
                2048,
                65536,
                min_batch=512,
                max_batch=65536,
                windows_tdr_cap=8192,
            )
            self.assertEqual(capped, 8192)

        with (
            patch.object(hardware_adapt.os, "name", "posix"),
            patch.object(
                hardware_adapt,
                "query_cuda_free_total_bytes",
                return_value=(40 * hardware_adapt.GIB, 48 * hardware_adapt.GIB),
            ),
        ):
            uncapped = hardware_adapt.adaptive_cuda_scaled_batch(
                2048,
                65536,
                min_batch=512,
                max_batch=65536,
                windows_tdr_cap=8192,
            )
            self.assertEqual(uncapped, 65536)

    def test_cohort_telemetry_and_runtime_discovery_handle_wddm_na(self) -> None:
        from shared.cohort_telemetry import gpu_sample
        from shared import managed_runtimes

        wddm_responses = [
            subprocess.CompletedProcess([], 0, "GPU-wddm, [N/A], 1234\n"),
            subprocess.CompletedProcess([], 0, "4321, [N/A], GPU-wddm\n"),
        ]
        with patch("shared.cohort_telemetry.subprocess.run", side_effect=wddm_responses):
            sample = gpu_sample({4321}, executable="nvidia-smi")
        self.assertTrue(sample["available"])
        self.assertIsNone(sample["owned_vram_mib"])
        self.assertIn("WDDM", sample["owned_vram_note"])
        self.assertIsNone(sample["devices"][0]["utilization_percent"])
        self.assertEqual(sample["devices"][0]["used_mib"], 1234.0)

        with (
            patch.object(managed_runtimes, "_resolve_nvidia_smi", return_value=r"C:\Windows\System32\nvidia-smi.exe"),
            patch.object(
                managed_runtimes.subprocess,
                "run",
                return_value=subprocess.CompletedProcess([], 0, "GPU 0: NVIDIA RTX 4090\n"),
            ) as run_mock,
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=False),
        ):
            self.assertTrue(managed_runtimes.cuda_candidate())
            self.assertTrue(managed_runtimes.cuda_visible())
            self.assertEqual(run_mock.call_args.args[0][0], r"C:\Windows\System32\nvidia-smi.exe")

    # ------------------------------------------------------------------ CUDA-2: UKF Windows nvcuda.dll + CUDA_PATH\bin
    def test_ukf_gpu_env_and_windows_nvcuda_detection(self) -> None:
        from tractography import ukf

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cuda_root = root / "CUDA_v12.6"
            cuda_bin = cuda_root / "bin"
            cuda_bin.mkdir(parents=True)
            ukf_exe = root / "bundled-runtimes" / "ukf" / "bin" / "UKFTractography_gpu.exe"
            ukf_exe.parent.mkdir(parents=True)
            ukf_exe.write_bytes(b"")

            with patch.object(ukf.sys, "platform", "win32"):
                env = ukf.ukf_gpu_process_environment(
                    {"SYSTEMROOT": r"C:\Windows", "CUDA_PATH": str(cuda_root), "PATH": r"C:\Slicer"},
                    ukf_exe,
                )
                self.assertIn(str(cuda_bin), env["PATH"].split(";"))

            # Test nvcuda.dll fallback when cudart is not found
            class _FakeFunc:
                def __init__(self, fn) -> None:
                    self._fn = fn
                    self.argtypes = None
                    self.restype = None

                def __call__(self, *args):
                    return self._fn(*args)

            class _FakeDriver:
                def __init__(self) -> None:
                    self.cuInit = _FakeFunc(lambda _flags: 0)
                    self.cuDeviceGetCount = _FakeFunc(self._get_count)

                def _get_count(self, count_ptr):
                    obj = ctypes.cast(count_ptr, ctypes.POINTER(ctypes.c_int))
                    obj[0] = 2
                    return 0

            with (
                patch.dict(sys.modules, {"torch": None}),
                patch.object(ukf.sys, "platform", "win32"),
                patch("ctypes.util.find_library", return_value=None),
                patch("ctypes.WinDLL", return_value=_FakeDriver(), create=True),
            ):
                self.assertTrue(ukf._cuda_runtime_available_for_ukf(ukf_exe))

    # ------------------------------------------------------------------ CUDA-3: Silent CREATE_NO_WINDOW & torch_cuda_build
    def test_environment_cuda_probe_records_build_and_uses_no_window_on_nt(self) -> None:
        from shared import environment

        seen_kwargs: list[dict] = []

        def fake_run(cmd, **kwargs):
            seen_kwargs.append(dict(kwargs))
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=json.dumps(
                    {
                        "torch_importable": True,
                        "cuda_available": True,
                        "device_count": 1,
                        "mps_available": False,
                        "torch_version": "2.6.0+cu124",
                        "torch_cuda_build": "12.4",
                        "device_name": "NVIDIA RTX 4080",
                        "compute_capability": [8, 9],
                    }
                ),
            )

        with (
            patch("shared.external_process.os.name", "nt"),
            patch.object(environment.subprocess, "CREATE_NO_WINDOW", 0x08000000, create=True),
            patch.object(environment.subprocess, "run", side_effect=fake_run),
        ):
            ok, details = environment._check_cuda_device("cuda:0", Path(sys.executable))

        self.assertTrue(ok)
        self.assertEqual(details["torch_cuda_build"], "12.4")
        self.assertEqual(details["device_name"], "NVIDIA RTX 4080")
        self.assertEqual(details["compute_capability"], [8, 9])
        self.assertEqual(seen_kwargs[0].get("creationflags"), 0x08000000)

    # ------------------------------------------------------------------ CUDA-4 & CUDA-5: Multi-GPU cuda:1 + NODDI fallback
    def test_multigpu_cuda1_preservation_and_noddi_fallback(self) -> None:
        from microstructure import dki, freewater_wls, mkcurve, noddi_accel
        from preprocessing import dipy_accel
        from shared import orchestrator
        from tractography import dti

        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 2)
        )
        with patch.dict(sys.modules, {"torch": fake_torch}), patch.object(noddi_accel, "torch", fake_torch):
            self.assertEqual(noddi_accel.resolve_noddi_device("cuda:1"), "cuda:1")
            self.assertEqual(noddi_accel.resolve_noddi_device("cuda:0"), "cuda")
            self.assertEqual(dipy_accel.resolve_dipy_accel_device("cuda:1"), "cuda:1")
            self.assertEqual(dipy_accel.resolve_dipy_accel_device("cuda"), "cuda")
            self.assertEqual(dki._resolve_dki_device("cuda:1"), "cuda:1")
            self.assertEqual(dki._resolve_dki_device("cuda:0"), "cuda")
            self.assertEqual(freewater_wls._resolve_fw_device("cuda:1"), "cuda:1")
            self.assertEqual(mkcurve._resolve_mkcurve_device("cuda:1"), "cuda:1")
            self.assertEqual(dti._resolve_dti_device("cuda:1"), "cuda:1")

        cfg = orchestrator.ProductOrchestratorConfig(
            subject_id="sub01",
            subject_root=Path("."),
            targets=("segmentation",),
            ddseg_device="cuda:1",
        )
        self.assertEqual(orchestrator._stage_compute_device(cfg), "cuda:1")

        # Verify mkcurve_tensor cache separates cuda:0 vs cuda:1 keys
        src_mk = (REPO_ROOT / "microstructure/mkcurve_tensor.py").read_text(encoding="utf-8")
        self.assertIn("def _cached_cuda_operators(scaled_shape, scaled_bytes, grid_shape, grid_bytes, device_str=\"cuda\")", src_mk)

        # Verify tractography/wma propagates CUDA_VISIBLE_DEVICES for cuda:N
        src_wma = (REPO_ROOT / "tractography/wma.py").read_text(encoding="utf-8")
        self.assertIn('env["CUDA_VISIBLE_DEVICES"] = gpu_idx', src_wma)

        # Verify noddi_accel passes windows_tdr_cap=8192 and uses CPU fallback on OOM when explicit_cuda=False
        src_noddi = (REPO_ROOT / "microstructure/noddi_accel.py").read_text(encoding="utf-8")
        self.assertIn("windows_tdr_cap=8192", src_noddi)
        self.assertIn("cpu_fallback_fn=None if explicit_cuda else _cpu_chunk_fallback", src_noddi)

    # ------------------------------------------------------------------ CUDA-6 & CUDA-7: CNN hardening & lazy surface
    def test_cnn_hardening_and_ddsurfer_lazy_cuda_models(self) -> None:
        from anatomy.segmentation import inference as seg_inf
        from shared import pnl_mask_native

        # Segmentation inference clears GPU cache in finally even on error
        class _RaisingModel:
            device = "cuda:0"
            input_layout = "NCHW"
            output_layout = "NCHW"

            def forward(self, _x):
                raise RuntimeError("boom")

        with patch("shared.hardware_adapt.empty_gpu_cache_safely") as empty_mock:
            with self.assertRaisesRegex(RuntimeError, "boom"):
                seg_inf.predict_view(_RaisingModel(), [np.ones((4, 4, 3), dtype=np.float32)], False)
            self.assertGreaterEqual(empty_mock.call_count, 1)

        # DDSurfer_Pred torchvision Compose fallback check via source contract
        dd_pred_src = (REPO_ROOT / "anatomy/parcellation/DDSurfer_Pred.py").read_text(encoding="utf-8")
        self.assertIn("class _ComposeFallback", dd_pred_src)
        self.assertIn("torch.backends.cudnn.benchmark = True", dd_pred_src)
        self.assertIn("is_pinned", dd_pred_src)

        # PNL native predict: no pin_memory on Windows, OOM halving retry, cache cleanup
        if importlib.util.find_spec("torch") is not None:
            import torch

            call_batches: list[int] = []
            requested_devs: list[str] = []
            pin_called = {"n": 0}
            orig_pin = torch.Tensor.pin_memory
            orig_zeros = torch.zeros
            orig_to = torch.Tensor.to

            def spy_pin(self_t, *a, **kw):
                pin_called["n"] += 1
                return orig_pin(self_t, *a, **kw)

            def fake_zeros(*args, **kwargs):
                dev = kwargs.get("device")
                if isinstance(dev, str) and dev.startswith("cuda"):
                    requested_devs.append(dev)
                    kwargs["device"] = "cpu"
                return orig_zeros(*args, **kwargs)

            def fake_to(self_t, *args, **kwargs):
                if args and isinstance(args[0], str) and args[0].startswith("cuda"):
                    requested_devs.append(args[0])
                    args = ("cpu", *args[1:])
                elif isinstance(kwargs.get("device"), str) and kwargs["device"].startswith("cuda"):
                    requested_devs.append(kwargs["device"])
                    kwargs["device"] = "cpu"
                return orig_to(self_t, *args, **kwargs)

            class _FakePnlModel:
                def load_state_dict(self, _st, strict=True):
                    pass

                def __call__(self, t_in, segmentation_only=True):
                    bs = int(t_in.shape[0])
                    call_batches.append(bs)
                    if bs > 4:
                        raise RuntimeError("CUDA out of memory")
                    return orig_zeros((bs, 1, 256, 256), dtype=torch.float32)

            fake_states = {"sagittal": {}, "coronal": {}, "axial": {}}
            fake_inf = types.SimpleNamespace(
                VIEWS=("sagittal", "coronal", "axial"),
                orient=lambda vol, view: np.swapaxes(vol, 0, ("sagittal", "coronal", "axial").index(view)),
                aggregate=lambda probs: (probs["axial"], (probs["axial"] > 0.45).astype(np.float32)),
            )
            vol = np.zeros((256, 256, 256), dtype=np.float32)
            vol[100:110, 100:110, 100:110] = 0.5
            with (
                patch.object(pnl_mask_native.sys, "platform", "win32"),
                patch.object(torch.Tensor, "pin_memory", spy_pin),
                patch.object(torch, "zeros", fake_zeros),
                patch.object(torch.Tensor, "to", fake_to),
                patch("shared.hardware_adapt.adaptive_cuda_scaled_batch", return_value=8),
                patch("shared.hardware_adapt.empty_cuda_cache_safely") as cache_mock,
            ):
                mask = pnl_mask_native._fast_vendored_predict(
                    vol,
                    "cuda:0",
                    32,
                    (fake_states, _FakePnlModel()),
                    fake_inf,
                )
            self.assertEqual(mask.shape, (256, 256, 256))
            self.assertEqual(pin_called["n"], 0)
            self.assertTrue(any(d == "cuda:0" for d in requested_devs))
            self.assertIn(8, call_batches)
            self.assertIn(4, call_batches)
            self.assertGreaterEqual(cache_mock.call_count, 1)

            # CUDA-7: ddsurfer_backend/inference/predict.py lazy loading on < 18 GiB CUDA vs >= 18 GiB
            dd_root = str(REPO_ROOT / "anatomy/surface/ddsurfer_backend")
            sys_path_added = dd_root not in sys.path
            if sys_path_added:
                sys.path.insert(0, dd_root)
            try:
                from inference import predict as dd_predict

                with tempfile.TemporaryDirectory() as td:
                    root = Path(td)
                    ckpt_dir = root / "ckpts"
                    tmpl_dir = root / "tmpl"
                    out_dir = root / "out"
                    ckpt_dir.mkdir()
                    tmpl_dir.mkdir()
                    (tmpl_dir / "left.obj").write_text("", encoding="utf-8")
                    manifest = {
                        "architecture": {"step_size": 0.1, "M": 4, "R": 2},
                        "geometry": {
                            "affine": np.eye(4).tolist(),
                            "full_shape": [176, 224, 176],
                        },
                        "templates": {"left": {"file": "left.obj", "sha256": "a" * 64}},
                        "checkpoint_files": {"left": {"white": "wm.pt", "pial": "pial.pt"}},
                        "postprocess_iters": {"white": 0, "pial": 0},
                        "precision": "fp32",
                    }
                    (ckpt_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                    fake_mesh = types.SimpleNamespace(
                        vertices=np.array([[10.0, 20.0, 30.0]], dtype=np.float32),
                        faces=np.array([[0, 0, 0]], dtype=np.int64),
                    )
                    args = types.SimpleNamespace(
                        device="cuda:0",
                        checkpoint_root=ckpt_dir,
                        surf_hemi="left",
                        predict_mode="all",
                        step_size=None,
                        n_svf=None,
                        n_res=None,
                        template_dir=tmpl_dir,
                        max_gpu_gib=24.0,
                        precision="fp32",
                        output_dir=out_dir,
                        save_debug=False,
                    )
                    fake_cuda_dev = types.SimpleNamespace(type="cuda", index=0)
                    for free_gib, expect_lazy, expect_eager_loads in ((12, True, 0), (22, False, 2)):
                        with (
                            patch.object(dd_predict.torch, "device", return_value=fake_cuda_dev),
                            patch.object(dd_predict.torch.cuda, "set_device"),
                            patch.object(
                                dd_predict.torch.cuda,
                                "mem_get_info",
                                return_value=(free_gib * 1024**3, 24 * 1024**3),
                            ),
                            patch.object(dd_predict.torch.cuda, "set_per_process_memory_fraction"),
                            patch.object(dd_predict, "load_checksums", return_value={"wm.pt": "b" * 64, "pial.pt": "c" * 64}),
                            patch.object(dd_predict, "sha256_file", return_value="a" * 64),
                            patch.object(dd_predict, "load_mesh", return_value=fake_mesh),
                            patch.object(dd_predict, "load_model", return_value=object()) as load_mock,
                        ):
                            predictor = dd_predict.SurfacePredictor(args)
                        self.assertIs(predictor._lazy_models, expect_lazy)
                        self.assertIs(predictor._lazy_mps_models, expect_lazy)
                        self.assertEqual(load_mock.call_count, expect_eager_loads)
            finally:
                if sys_path_added and dd_root in sys.path:
                    sys.path.remove(dd_root)


if __name__ == "__main__":
    unittest.main()
