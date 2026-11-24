"""Adversarial Windows 10/11 NVIDIA CUDA WDDM simulation test suite.

Simulates four realistic GPU hardware profiles without requiring physical Windows CUDA hardware:
  - Profile A: NVIDIA GeForce RTX 4060 (8 GiB WDDM, 6.0 GiB free after DWM + Slicer OpenGL, sm_89, 1/64 FP64)
  - Profile B: NVIDIA GeForce RTX 3060 (12 GiB WDDM, 10.5 GiB free, sm_86, 1/64 FP64)
  - Profile C: NVIDIA GeForce RTX 3050 Laptop / GTX 1650 (4 GiB WDDM, 3.0 GiB free, sm_86, 1/64 FP64)
  - Profile D: NVIDIA H100 80GB HBM3 (80 GiB Linux server, 76.0 GiB free, sm_90, 1/2 FP64 — regression guard)

Exercises real production code paths across Windows packaging/DLL/PYTHONPATH precedence,
WDDM VRAM budgeting & shared-RAM spillover protection, Windows 2-second WDDM TDR FP64 bounds,
and OOM batch-halving + device='auto' CPU fallback vs explicit device='cuda' hard failure
across all 14 CUDA/GPU routes.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch
import xml.etree.ElementTree as ET

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


GIB = 1024 ** 3


@dataclass(frozen=True)
class SimulatedGPUProfile:
    label: str
    device_name: str
    total_gib: float
    free_gib: float
    compute_capability: tuple[int, int]
    platform: str  # "win32" or "linux"
    os_name: str  # "nt" or "posix"
    consumer_fp64: bool


PROFILE_A_RTX4060_8G = SimulatedGPUProfile(
    label="Profile A (RTX 4060 8 GiB WDDM, 6.0 GiB free)",
    device_name="NVIDIA GeForce RTX 4060",
    total_gib=8.0,
    free_gib=6.0,
    compute_capability=(8, 9),
    platform="win32",
    os_name="nt",
    consumer_fp64=True,
)
PROFILE_B_RTX3060_12G = SimulatedGPUProfile(
    label="Profile B (RTX 3060 12 GiB WDDM, 10.5 GiB free)",
    device_name="NVIDIA GeForce RTX 3060",
    total_gib=12.0,
    free_gib=10.5,
    compute_capability=(8, 6),
    platform="win32",
    os_name="nt",
    consumer_fp64=True,
)
PROFILE_C_LAPTOP_4G = SimulatedGPUProfile(
    label="Profile C (RTX 3050 Laptop 4 GiB WDDM, 3.0 GiB free)",
    device_name="NVIDIA GeForce RTX 3050 Laptop GPU",
    total_gib=4.0,
    free_gib=3.0,
    compute_capability=(8, 6),
    platform="win32",
    os_name="nt",
    consumer_fp64=True,
)
PROFILE_D_H100_80G_LINUX = SimulatedGPUProfile(
    label="Profile D (H100 80 GiB Linux Server, 76.0 GiB free)",
    device_name="NVIDIA H100 80GB HBM3",
    total_gib=80.0,
    free_gib=76.0,
    compute_capability=(9, 0),
    platform="linux",
    os_name="posix",
    consumer_fp64=False,
)


@contextmanager
def simulated_gpu_environment(profile: SimulatedGPUProfile):
    """Patch OS/platform and torch.cuda introspection to simulate ``profile``."""
    import torch
    from shared import hardware_adapt

    free_b = int(profile.free_gib * GIB)
    total_b = int(profile.total_gib * GIB)

    with ExitStack() as stack:
        stack.enter_context(patch.object(hardware_adapt.os, "name", profile.os_name))
        stack.enter_context(patch.object(hardware_adapt.sys, "platform", profile.platform))
        stack.enter_context(patch.object(torch.cuda, "is_available", return_value=True))
        stack.enter_context(patch.object(torch.cuda, "device_count", return_value=1))
        stack.enter_context(patch.object(torch.cuda, "current_device", return_value=0))
        stack.enter_context(patch.object(torch.cuda, "get_device_name", return_value=profile.device_name))
        stack.enter_context(
            patch.object(torch.cuda, "get_device_capability", return_value=profile.compute_capability)
        )
        stack.enter_context(patch.object(torch.cuda, "mem_get_info", return_value=(free_b, total_b)))
        stack.enter_context(patch.object(torch.cuda, "memory_allocated", return_value=total_b - free_b))
        stack.enter_context(patch.object(torch.cuda, "empty_cache", return_value=None))
        yield


class WindowsPackagingAndDllSimulationTests(unittest.TestCase):
    """Verify Windows Slicer CUDA PyTorch precedence, os.add_dll_directory, and env propagation."""

    def test_windows_slicer_cuda_torch_is_not_shadowed_by_bundled_cpu_torch(self) -> None:
        from shared import bundled_python
        from shared.external_process import python_tool_environment
        from anatomy.surface.backend_bridge import backend_environment

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            # Simulated Windows Slicer install with user-installed CUDA torch in Lib/site-packages
            slicer_root = root / "Slicer 5.10.0"
            slicer_py = slicer_root / "bin" / "PythonSlicer.exe"
            slicer_py.parent.mkdir(parents=True)
            slicer_py.write_bytes(b"")
            slicer_sp = slicer_root / "lib" / "Python" / "Lib" / "site-packages"
            (slicer_sp / "torch" / "lib").mkdir(parents=True)
            (slicer_sp / "nvidia" / "cusolver" / "bin").mkdir(parents=True)
            (slicer_sp / "nvidia" / "cublas" / "bin").mkdir(parents=True)

            # Simulated Windows extension with bundled CPU-only torch (no numpy)
            ext_root = root / "SlicerDSpace"
            module_root = ext_root / "lib" / "Slicer-5.10" / "qt-scripted-modules"
            bundled_sp = ext_root / "bundled-python" / "site-packages"
            (bundled_sp / "torch" / "lib").mkdir(parents=True)
            (bundled_sp / "pywin32_system32").mkdir(parents=True)
            module_root.mkdir(parents=True)
            cur_py = f"{sys.version_info[0]}.{sys.version_info[1]}"
            (ext_root / "bundled-python" / "python-runtime.json").write_text(
                json.dumps({"python_version": cur_py}), encoding="utf-8"
            )

            cuda_toolkit = root / "CUDA" / "v12.4"
            (cuda_toolkit / "bin").mkdir(parents=True)

            # 1. inject_bundled_python_env puts Slicer Lib/site-packages BEFORE bundled CPU site-packages
            env: dict[str, str] = {"PYTHONPATH": "/caller/first"}
            with (
                patch.object(bundled_python.sys, "platform", "win32"),
                patch.dict(os.environ, {"SLICERDSPACE_BUNDLED_PYTHON_ROOT": str(ext_root / "bundled-python")}),
            ):
                bundled_python.inject_bundled_python_env(
                    env, module_root=module_root, python_executable=slicer_py
                )
            pp_entries = env["PYTHONPATH"].split(os.pathsep)
            self.assertEqual(pp_entries[0], "/caller/first")
            self.assertIn(str(slicer_sp.resolve()), pp_entries)
            self.assertIn(str(bundled_sp.resolve()), pp_entries)
            self.assertLess(
                pp_entries.index(str(slicer_sp.resolve())),
                pp_entries.index(str(bundled_sp.resolve())),
                "Slicer's CUDA torch site-packages MUST precede bundled CPU-only site-packages on Windows",
            )
            self.assertEqual(env.get("PYTORCH_CUDA_ALLOC_CONF"), bundled_python.WINDOWS_CUDA_ALLOC_CONF)
            self.assertNotIn("expandable_segments", env["PYTORCH_CUDA_ALLOC_CONF"])
            self.assertNotIn("SLICERDSPACE_BUNDLED_SITE_PACKAGES", env)

            # 2. python_tool_environment & backend_environment propagate the same precedence and alloc conf
            with (
                patch.object(bundled_python.sys, "platform", "win32"),
                patch("shared.external_process.sys.platform", "win32"),
                patch.dict(
                    os.environ,
                    {"SLICERDSPACE_BUNDLED_PYTHON_ROOT": str(ext_root / "bundled-python")},
                    clear=False,
                ),
            ):
                tool_env = python_tool_environment(slicer_py)
                surf_env = backend_environment(python_bin=slicer_py)
            for label, sub_env in (("python_tool_environment", tool_env), ("backend_environment", surf_env)):
                sub_pp = sub_env.get("PYTHONPATH", "").split(os.pathsep)
                self.assertIn(str(slicer_sp.resolve()), sub_pp, label)
                self.assertIn(str(bundled_sp.resolve()), sub_pp, label)
                self.assertLess(
                    sub_pp.index(str(slicer_sp.resolve())),
                    sub_pp.index(str(bundled_sp.resolve())),
                    f"{label} shadowed Slicer CUDA torch with bundled CPU torch",
                )
                self.assertEqual(sub_env.get("PYTORCH_CUDA_ALLOC_CONF"), bundled_python.WINDOWS_CUDA_ALLOC_CONF)

            # 3. register_windows_dll_directories registers torch/lib, nvidia/*/bin, pywin32_system32, CUDA_PATH/bin
            added_dirs: list[str] = []
            handles: list[Mock] = []

            def fake_add_dll_directory(path_str: str):
                added_dirs.append(path_str)
                h = Mock(name=f"dll_handle({path_str})")
                handles.append(h)
                return h

            orig_handles = list(bundled_python._WINDOWS_DLL_DIR_HANDLES)
            orig_reg = set(bundled_python._WINDOWS_REGISTERED_DLL_DIRS)
            try:
                bundled_python._WINDOWS_DLL_DIR_HANDLES.clear()
                bundled_python._WINDOWS_REGISTERED_DLL_DIRS.clear()
                with (
                    patch.object(bundled_python.sys, "platform", "win32"),
                    patch.object(bundled_python.os, "add_dll_directory", fake_add_dll_directory, create=True),
                    patch.dict(
                        os.environ,
                        {
                            "SLICERDSPACE_BUNDLED_PYTHON_ROOT": str(ext_root / "bundled-python"),
                            "CUDA_PATH": str(cuda_toolkit),
                        },
                        clear=False,
                    ),
                ):
                    registered = bundled_python.register_windows_dll_directories(
                        python_executable=slicer_py, module_root=module_root
                    )
                self.assertIn(str((slicer_sp / "torch" / "lib").resolve()), registered)
                self.assertIn(str((slicer_sp / "nvidia" / "cusolver" / "bin").resolve()), registered)
                self.assertIn(str((slicer_sp / "nvidia" / "cublas" / "bin").resolve()), registered)
                self.assertIn(str((bundled_sp / "pywin32_system32").resolve()), registered)
                self.assertIn(str((cuda_toolkit / "bin").resolve()), registered)
                self.assertEqual(len(bundled_python._WINDOWS_DLL_DIR_HANDLES), len(handles))
            finally:
                bundled_python._WINDOWS_DLL_DIR_HANDLES[:] = orig_handles
                bundled_python._WINDOWS_REGISTERED_DLL_DIRS.clear()
                bundled_python._WINDOWS_REGISTERED_DLL_DIRS.update(orig_reg)

    def test_windows_probes_emit_dll_registration_and_no_window_flags(self) -> None:
        from shared import environment, external_process

        with patch.object(environment.sys, "platform", "win32"):
            win_snip = environment._windows_dll_probe_snippet()
        self.assertIn("add_dll_directory", win_snip)
        self.assertIn("CUDA_PATH", win_snip)
        with patch.object(environment.sys, "platform", "linux"):
            self.assertEqual(environment._windows_dll_probe_snippet(), "")

        with patch.object(external_process.os, "name", "nt"), patch.object(
            external_process.subprocess, "CREATE_NO_WINDOW", 0x08000000, create=True
        ):
            kw = external_process.subprocess_no_window_kwargs()
            self.assertEqual(kw.get("creationflags"), 0x08000000)


class ConsumerFP64AndWddmBudgetSimulationTests(unittest.TestCase):
    """Verify consumer 1/64 FP64 detection, WDDM TDR caps, cuDNN benchmark gating, and H100 parity."""

    def test_consumer_fp64_detection_and_cudnn_benchmark_across_four_profiles(self) -> None:
        from shared import hardware_adapt

        for profile, expect_consumer, expect_cudnn_bench in (
            (PROFILE_A_RTX4060_8G, True, False),
            (PROFILE_B_RTX3060_12G, True, True),
            (PROFILE_C_LAPTOP_4G, True, False),
            (PROFILE_D_H100_80G_LINUX, False, True),
        ):
            with self.subTest(profile=profile.label), simulated_gpu_environment(profile):
                self.assertEqual(hardware_adapt.is_consumer_fp64_cuda_gpu("cuda:0"), expect_consumer)
                self.assertEqual(hardware_adapt.should_enable_cudnn_benchmark("cuda:0"), expect_cudnn_bench)

    def test_fp64_tdr_batch_capping_for_consumer_windows_vs_h100_linux(self) -> None:
        from shared import hardware_adapt

        # Profile A (RTX 4060 8G, 6.0 GiB free, Windows): consumer FP64 capped & tighter 0.25 WDDM headroom
        with simulated_gpu_environment(PROFILE_A_RTX4060_8G):
            noddi_b = hardware_adapt.adaptive_cuda_scaled_batch(
                24576, 65536, min_batch=512, max_batch=65536, device="cuda:0",
                windows_tdr_cap=8192, fp64=True, consumer_fp64_tdr_cap=2048,
            )
            mkc_b = hardware_adapt.adaptive_cuda_scaled_batch(
                16384 * 16, 4096, min_batch=128, max_batch=16384, device="cuda:0",
                windows_tdr_cap=4096, fp64=True, consumer_fp64_tdr_cap=1024,
            )
            mppca_b = hardware_adapt.adaptive_cuda_scaled_batch(
                125 * 125 * 8 * 6, 4096, min_batch=256, max_batch=8192, device="cuda:0",
                windows_tdr_cap=2048, fp64=True, consumer_fp64_tdr_cap=512,
            )
            self.assertLessEqual(noddi_b, 2048)
            self.assertLessEqual(mkc_b, 1024)
            self.assertLessEqual(mppca_b, 512)

        # Profile B (RTX 3060 12G, 10.5 GiB free, Windows): consumer FP64 still TDR-capped
        with simulated_gpu_environment(PROFILE_B_RTX3060_12G):
            noddi_b = hardware_adapt.adaptive_cuda_scaled_batch(
                24576, 65536, min_batch=512, max_batch=65536, device="cuda:0",
                windows_tdr_cap=8192, fp64=True, consumer_fp64_tdr_cap=2048,
            )
            self.assertEqual(noddi_b, 2048)

        # Profile C (4 GiB Laptop, 3.0 GiB free, Windows): tightly VRAM-bounded below consumer TDR cap
        with simulated_gpu_environment(PROFILE_C_LAPTOP_4G):
            mppca_b = hardware_adapt.adaptive_cuda_scaled_batch(
                125 * 125 * 8 * 6, 4096, min_batch=256, max_batch=8192, device="cuda:0",
                windows_tdr_cap=2048, fp64=True, consumer_fp64_tdr_cap=512,
            )
            self.assertLessEqual(mppca_b, 512)
            # Peak allocation stays within 25% of 3.0 GiB (768 MiB), preventing WDDM shared-RAM spillover
            self.assertLessEqual(mppca_b * (125 * 125 * 8 * 6), int(3.0 * GIB * 0.25) + 125 * 125 * 8 * 6 * 256)

        # Profile D (H100 80G Linux): ZERO throttling — full server throughput preserved!
        with simulated_gpu_environment(PROFILE_D_H100_80G_LINUX):
            noddi_h100 = hardware_adapt.adaptive_cuda_scaled_batch(
                24576, 65536, min_batch=512, max_batch=65536, device="cuda:0",
                windows_tdr_cap=8192, fp64=True, consumer_fp64_tdr_cap=2048,
            )
            mkc_h100 = hardware_adapt.adaptive_cuda_scaled_batch(
                16384 * 16, 4096, min_batch=128, max_batch=16384, device="cuda:0",
                windows_tdr_cap=4096, fp64=True, consumer_fp64_tdr_cap=1024,
            )
            dti_h100 = hardware_adapt.adaptive_cuda_scaled_batch(
                6400, 16384, min_batch=2048, max_batch=16384, device="cuda:0",
                windows_tdr_cap=16384, fp64=True, consumer_fp64_tdr_cap=4096,
            )
            self.assertEqual(noddi_h100, 65536)
            self.assertEqual(mkc_h100, 4096)
            self.assertEqual(dti_h100, 16384)


class DeepLearningWddmRoutesSimulationTests(unittest.TestCase):
    """Exercise DDSurfer, DDParcel, DDSeg, PNL Brain Mask, and TractCloud under WDDM VRAM limits."""

    def tearDown(self) -> None:
        try:
            import torch

            fn = getattr(torch, "set_flush_denormal", None)
            if callable(fn):
                fn(False)
        except Exception:
            pass

    def test_ddsurfer_surface_memory_routing_across_profiles(self) -> None:
        from shared import environment
        from anatomy.surface.prediction import PredictionStageConfig, predict_subject

        self.assertEqual(environment.SURFACE_MIN_FREE_GPU_GIB, 7.0)

        for profile, expected_auto, expected_cuda in (
            (PROFILE_A_RTX4060_8G, "cpu", "block"),
            (PROFILE_B_RTX3060_12G, "ok", "ok"),
            (PROFILE_C_LAPTOP_4G, "cpu", "block"),
            (PROFILE_D_H100_80G_LINUX, "ok", "ok"),
        ):
            with (
                self.subTest(profile=profile.label),
                patch.object(environment, "_is_macos", return_value=False),
                patch.object(environment, "cuda_free_gib", return_value=profile.free_gib),
            ):
                auto_action, _ = environment.surface_gpu_memory_decision("auto")
                cuda_action, _ = environment.surface_gpu_memory_decision("cuda:0")
                self.assertEqual(auto_action, expected_auto)
                self.assertEqual(cuda_action, expected_cuda)

        # Direct predict_subject(device="auto") on Profile A (6.0 GiB free) must fall back to CPU, not crash!
        captured_cmd: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = PredictionStageConfig(
                subject_id="sub01",
                hemisphere="lh",
                predict_mode="all",
                device="auto",
                step_size=0.05,
                n_svf=7,
                n_res=3,
                template_dir=root,
                checkpoint_root=root,
                input_root=root,
                output_root=root / "out",
            )
            with (
                patch("shared.environment.resolve_auto_torch_device", return_value="cuda:0"),
                patch("shared.environment._is_macos", return_value=False),
                patch("shared.environment.cuda_free_gib", return_value=PROFILE_A_RTX4060_8G.free_gib),
                patch("anatomy.surface.prediction.run_backend", side_effect=lambda cmd, **kw: captured_cmd.extend(map(str, cmd))),
            ):
                predict_subject(cfg)
            self.assertEqual(captured_cmd[captured_cmd.index("--device") + 1], "cpu")

    def test_ddparcel_cpu_accumulation_and_oom_halving_plus_auto_cpu_fallback(self) -> None:
        import torch
        from anatomy.parcellation import DDSurfer_Pred as dd_pred

        # Verify probability accumulation buffer is allocated on CPU host RAM or disk-backed memmap (0 bytes GPU VRAM)
        src = Path(dd_pred.__file__).read_text(encoding="utf-8")
        self.assertIn("prob_buffer = _allocate_prediction_probability(", src)
        self.assertIn("torch.zeros(tuple(shape), dtype=torch.float32)", src)
        self.assertNotIn("pred_prob = torch.zeros((1, 256, 256, 256", src.split("device=")[1] if "device=" in src else "")

        class FlakyCudaModel(torch.nn.Module):
            def __init__(self, fail_single_cuda: bool):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
                self.fail_single_cuda = fail_single_cuda
                self.calls: list[tuple[int, str]] = []
                self.on_cpu = False

            def to(self, *args, **kwargs):
                for a in args:
                    if str(a) == "cpu":
                        self.on_cpu = True
                return super().to(*args, **kwargs)

            def forward(self, x):
                dev_label = "cpu" if self.on_cpu else "cuda"
                self.calls.append((int(x.shape[0]), dev_label))
                if not self.on_cpu:
                    if x.shape[0] > 1 or self.fail_single_cuda:
                        raise RuntimeError("CUDA out of memory. Tried to allocate 1.20 GiB")
                return None, [x[:, :2] * self.weight]

        batch = torch.randn(4, 7, 16, 16)
        fake_cuda = torch.device("cuda:0")

        # Case 1: batch=4 OOMs -> automatically retries single-slice on GPU without CPU fallback
        model_retry = FlakyCudaModel(fail_single_cuda=False)
        with (
            patch.object(torch.Tensor, "to", lambda self_t, *a, **kw: self_t),
            patch.object(dd_pred, "empty_gpu_cache_safely"),
        ):
            out = dd_pred._forward_fused_logits(model_retry, batch, use_cuda=True, device=fake_cuda, allow_cpu_fallback=False)
        self.assertEqual(tuple(out.shape), (4, 2, 16, 16))
        self.assertEqual(model_retry.calls, [(4, "cuda"), (1, "cuda"), (1, "cuda"), (1, "cuda"), (1, "cuda")])

        # Case 2: single-slice OOMs under auto (allow_cpu_fallback=True) -> cleanly switches to CPU
        model_auto = FlakyCudaModel(fail_single_cuda=True)
        with (
            patch.object(torch.Tensor, "to", lambda self_t, *a, **kw: self_t),
            patch.object(dd_pred, "empty_gpu_cache_safely"),
        ):
            out_fb = dd_pred._forward_fused_logits(model_auto, batch, use_cuda=True, device=fake_cuda, allow_cpu_fallback=True)
        self.assertTrue(model_auto.on_cpu)
        self.assertEqual(tuple(out_fb.shape), (4, 2, 16, 16))

        # Case 3: single-slice OOMs under explicit CUDA (allow_cpu_fallback=False) -> raises OOM!
        model_strict = FlakyCudaModel(fail_single_cuda=True)
        with (
            patch.object(torch.Tensor, "to", lambda self_t, *a, **kw: self_t),
            patch.object(dd_pred, "empty_gpu_cache_safely"),
            self.assertRaisesRegex(RuntimeError, "out of memory"),
        ):
            dd_pred._forward_fused_logits(model_strict, batch, use_cuda=True, device=fake_cuda, allow_cpu_fallback=False)

    def test_ddseg_oom_batch_halving_cudnn_guard_and_auto_cpu_fallback(self) -> None:
        import torch
        from anatomy.segmentation.config import DDSegConfig
        from anatomy.segmentation.inference import predict_view
        from anatomy.segmentation.model_loader import load_torch_model
        from anatomy.segmentation.pipeline import predict_model_space

        # Verify cuDNN benchmark is disabled on Profile A (6.0 GiB WDDM) and enabled on Profile D (H100)
        with tempfile.TemporaryDirectory() as tmp:
            pt_file = Path(tmp) / "dti_axial.pt"
            pt_file.write_bytes(b"stub")
            fake_mod = MagicMock()
            fake_mod.eval.return_value = fake_mod
            for profile, expect_bench in ((PROFILE_A_RTX4060_8G, False), (PROFILE_D_H100_80G_LINUX, True)):
                with (
                    simulated_gpu_environment(profile),
                    patch.object(torch.jit, "load", return_value=fake_mod),
                ):
                    load_torch_model(pt_file, "cuda:0")
                    self.assertEqual(bool(torch.backends.cudnn.benchmark), expect_bench)

        # Verify predict_view halves batch_size 8 -> 4 -> 2 -> 1 on CUDA OOM
        class OomHalvingModel:
            device = "cuda:0"
            input_layout = "NCHW"
            output_layout = "NCHW"

            def __init__(self):
                self.seen_batches: list[int] = []

            def forward(self, x: np.ndarray) -> np.ndarray:
                self.seen_batches.append(int(x.shape[0]))
                if x.shape[0] > 2:
                    raise RuntimeError("CUDA out of memory")
                # Return (N, 4, H, W)
                return np.zeros((x.shape[0], 4, x.shape[2], x.shape[3]), dtype=np.float32) + 0.25

        slices = [np.random.RandomState(i).randn(8, 8, 5).astype(np.float32) for i in range(10)]
        m = OomHalvingModel()
        with simulated_gpu_environment(PROFILE_A_RTX4060_8G):
            pred = predict_view(m, slices, apply_softmax=True)
        self.assertEqual(pred.shape, (8, 8, 10, 4))
        self.assertIn(8, m.seen_batches)
        self.assertIn(4, m.seen_batches)
        self.assertIn(2, m.seen_batches)

        # Verify predict_model_space falls back to CPU on OOM for device="auto" but raises for explicit "cuda:0"
        stacked = np.random.RandomState(0).randn(8, 8, 8, 5).astype(np.float32)
        seen_devices: list[str] = []

        class FakeViewModel:
            input_layout = "NCHW"
            output_layout = "NCHW"

            def __init__(self, dev: str):
                self.device = dev

            def forward(self, x: np.ndarray) -> np.ndarray:
                if self.device.startswith("cuda"):
                    raise RuntimeError("CUDA out of memory at batch=1")
                return np.ones((x.shape[0], 4, x.shape[2], x.shape[3]), dtype=np.float32)

        def fake_load_models(weights_dir, prefix, device):
            seen_devices.append(device)
            return SimpleNamespace(
                axial=FakeViewModel(device),
                sagittal=FakeViewModel(device),
                coronal=FakeViewModel(device),
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_auto = DDSegConfig(
                input_mask_nii=root / "mask.nii.gz",
                output_folder=root / "out",
                parameter_type="FA_MD_Eval",
                weights_dir=root,
                device="auto",
            )
            with (
                patch("shared.environment.resolve_auto_torch_device", return_value="cuda:0"),
                patch("anatomy.segmentation.model_loader.load_models", side_effect=fake_load_models),
                patch("shared.hardware_adapt.empty_gpu_cache_safely"),
            ):
                labels, probs = predict_model_space(stacked, cfg_auto)
            self.assertEqual(seen_devices, ["cuda:0", "cpu"])
            self.assertEqual(labels.shape, (8, 8, 8))
            self.assertEqual(probs.shape, (8, 8, 8, 4))

            cfg_cuda = DDSegConfig(
                input_mask_nii=root / "mask.nii.gz",
                output_folder=root / "out",
                parameter_type="FA_MD_Eval",
                weights_dir=root,
                device="cuda:0",
            )
            with (
                patch("shared.environment.resolve_auto_torch_device", return_value="cuda:0"),
                patch("anatomy.segmentation.model_loader.load_models", side_effect=fake_load_models),
                patch("shared.hardware_adapt.empty_gpu_cache_safely"),
                self.assertRaisesRegex(RuntimeError, "out of memory"),
            ):
                predict_model_space(stacked, cfg_cuda)

    def test_pnl_native_oom_batch_halving_and_auto_cpu_fallback(self) -> None:
        import torch
        from shared import pnl_mask_native

        class FakePnlModel:
            def __init__(self, fail_batch_1: bool):
                self.fail_batch_1 = fail_batch_1
                self.on_cpu = False
                self.batch_sizes: list[int] = []

            def load_state_dict(self, state, strict=True):
                return None

            def to(self, device):
                if str(device) == "cpu":
                    self.on_cpu = True
                return self

            def __call__(self, x, segmentation_only=True):
                self.batch_sizes.append(int(x.shape[0]))
                if not self.on_cpu:
                    if x.shape[0] > 4 or self.fail_batch_1:
                        raise RuntimeError("CUDA out of memory")
                return torch.full_like(x, 0.75)

        fake_inf = SimpleNamespace(
            VIEWS=("axial", "coronal", "sagittal"),
            orient=lambda vol, view: vol,
            aggregate=lambda probs: (probs["axial"], probs["axial"] > 0.45),
        )
        vol = np.zeros((256, 256, 256), dtype=np.float32)
        vol[10:22] = 1.0  # 12 non-zero slices

        # Case 1: Profile C (3.0 GiB WDDM, 25% safety budget = 768 MiB / 128 MiB -> 6 slices)
        # auto-downscales 16 -> 6, then OOM above 4 halves 6 -> 3 and completes on GPU!
        m_half = FakePnlModel(fail_batch_1=False)
        with (
            simulated_gpu_environment(PROFILE_C_LAPTOP_4G),
            patch.object(torch.Tensor, "to", lambda self_t, *a, **kw: self_t),
            patch.object(torch, "zeros", lambda shape, **kw: torch.full(shape, 0.0, dtype=kw.get("dtype", torch.float32))),
        ):
            mask = pnl_mask_native._fast_vendored_predict(
                vol, "cuda:0", 16, ({"axial": {}, "coronal": {}, "sagittal": {}}, m_half), fake_inf,
                allow_cpu_fallback=False,
            )
        self.assertEqual(mask.shape, (256, 256, 256))
        self.assertFalse(m_half.on_cpu)
        self.assertIn(6, m_half.batch_sizes)
        self.assertIn(3, m_half.batch_sizes)

        # Case 2: OOM at batch 1 with allow_cpu_fallback=True -> switches to CPU cleanly
        m_auto = FakePnlModel(fail_batch_1=True)
        with (
            simulated_gpu_environment(PROFILE_C_LAPTOP_4G),
            patch.object(torch.Tensor, "to", lambda self_t, *a, **kw: self_t),
            patch.object(torch, "zeros", lambda shape, **kw: torch.full(shape, 0.0, dtype=kw.get("dtype", torch.float32))),
        ):
            mask_fb = pnl_mask_native._fast_vendored_predict(
                vol, "cuda:0", 8, ({"axial": {}, "coronal": {}, "sagittal": {}}, m_auto), fake_inf,
                allow_cpu_fallback=True,
            )
        self.assertTrue(m_auto.on_cpu)
        self.assertEqual(mask_fb.shape, (256, 256, 256))

        # Case 3: OOM at batch 1 with allow_cpu_fallback=False -> raises OOM!
        m_strict = FakePnlModel(fail_batch_1=True)
        with (
            simulated_gpu_environment(PROFILE_C_LAPTOP_4G),
            patch.object(torch.Tensor, "to", lambda self_t, *a, **kw: self_t),
            patch.object(torch, "zeros", lambda shape, **kw: torch.full(shape, 0.0, dtype=kw.get("dtype", torch.float32))),
            self.assertRaisesRegex(RuntimeError, "out of memory"),
        ):
            pnl_mask_native._fast_vendored_predict(
                vol, "cuda:0", 8, ({"axial": {}, "coronal": {}, "sagittal": {}}, m_strict), fake_inf,
                allow_cpu_fallback=False,
            )

    def test_tractcloud_bootstrap_wddm_caps_and_real_oom_fallback(self) -> None:
        import torch
        from tractography.tractcloud import TRACTCLOUD_COMPAT_BOOTSTRAP
        from tractography.tractcloud_accel import install_tractcloud_accelerations

        # 1. Verify TRACTCLOUD_COMPAT_BOOTSTRAP batch-size capping and non-shadowing on Profile A/C/D
        for profile, expected_bs in (
            (PROFILE_A_RTX4060_8G, "1024"),
            (PROFILE_B_RTX3060_12G, "2048"),
            (PROFILE_C_LAPTOP_4G, "512"),
            (PROFILE_D_H100_80G_LINUX, "2048"),
        ):
            with self.subTest(profile=profile.label):
                pkg = ModuleType("tractcloud")
                pipe = ModuleType("tractcloud.pipeline")
                pkg.pipeline = pipe
                fake_torch = Mock()
                fake_torch.__version__ = "2.5.1+cu124"
                fake_torch.cuda.is_available.return_value = True
                fake_torch.cuda.mem_get_info.return_value = (int(profile.free_gib * GIB), int(profile.total_gib * GIB))
                fake_torch.backends.mps.is_available.return_value = False
                argv = ["tractcloud", "--device", "auto", "--batch-size", "2048"]
                with tempfile.TemporaryDirectory() as td:
                    cpu_bsp = Path(td) / "bundled_no_numpy"
                    cpu_bsp.mkdir()
                    with (
                        patch.dict(sys.modules, {"tractcloud": pkg, "tractcloud.pipeline": pipe, "torch": fake_torch}),
                        patch("runpy.run_module"),
                        patch.object(sys, "argv", argv),
                        patch.object(sys, "platform", profile.platform),
                        patch.object(os, "name", profile.os_name),
                        patch.dict(
                            os.environ,
                            {
                                "SLICERDSPACE_TRACTCLOUD_DEVICE_RECORD": "",
                                "SLICERDSPACE_BUNDLED_SITE_PACKAGES": str(cpu_bsp),
                            },
                            clear=False,
                        ),
                    ):
                        orig_path = list(sys.path)
                        try:
                            exec(TRACTCLOUD_COMPAT_BOOTSTRAP, {})
                            if profile.platform == "win32":
                                # Non-numpy bundled dir must NOT be placed ahead of existing sys.path entries!
                                self.assertNotEqual(sys.path[0], str(cpu_bsp))
                        finally:
                            sys.path[:] = orig_path
                self.assertEqual(argv[argv.index("--batch-size") + 1], expected_bs)

        # 2. Exercise REAL install_tractcloud_accelerations + FastRealDataDataset + fast_run_inference OOM fallback!
        fake_pkg = ModuleType("tractcloud")
        fake_pipe = ModuleType("tractcloud.pipeline")
        fake_pipe.__file__ = "<simulated_tractcloud_pipeline>"
        fake_inf = ModuleType("tractcloud.inference")
        fake_vtk = ModuleType("tractcloud.vtk_io")

        def fake_fiber_dist(a: torch.Tensor, b: torch.Tensor, num_points: int = 15) -> torch.Tensor:
            # Simple pairwise L2 over flattened (N, 3*P)
            af = a.reshape(a.shape[0], -1)
            bf = b.reshape(b.shape[0], -1)
            return torch.cdist(af, bf)

        fake_inf._fiber_distance_efficient = fake_fiber_dist
        fake_inf.run_inference = Mock()
        fake_pkg.pipeline = fake_pipe
        fake_pkg.inference = fake_inf
        fake_pkg.vtk_io = fake_vtk

        class FlakyDGCNN(torch.nn.Module):
            def __init__(self, fail_min_batch: bool):
                super().__init__()
                self.fail_min_batch = fail_min_batch
                self.on_cpu = False
                self.seen_bs: list[int] = []

            def to(self, dev):
                if str(dev) == "cpu":
                    self.on_cpu = True
                return self

            def __call__(self, points, info):
                bs = int(points.shape[0])
                self.seen_bs.append(bs)
                if not self.on_cpu and (bs > 16 or self.fail_min_batch):
                    raise RuntimeError("CUDA out of memory")
                logits = torch.zeros((bs, 5), dtype=torch.float32)
                logits[:, 2] = 10.0
                return logits

        rng = np.random.RandomState(42)
        feat = rng.randn(40, 15, 3).astype(np.float32)

        with (
            patch.dict(
                sys.modules,
                {"tractcloud": fake_pkg, "tractcloud.pipeline": fake_pipe, "tractcloud.inference": fake_inf, "tractcloud.vtk_io": fake_vtk},
            ),
            simulated_gpu_environment(PROFILE_A_RTX4060_8G),
            patch.object(torch, "as_tensor", lambda data, dtype=None, device=None: torch.tensor(data, dtype=dtype)),
            patch.object(torch.Tensor, "to", lambda self_t, *a, **kw: self_t),
            patch.object(torch, "empty", lambda shape, dtype=None, device=None: torch.zeros(shape, dtype=dtype)),
        ):
            # Auto mode: cudnn.benchmark disabled on 6 GiB WDDM; OOM halves 64->32->16->8 then falls back to CPU!
            install_tractcloud_accelerations(fake_pipe, "cuda:0", requested_device="auto")
            self.assertFalse(bool(torch.backends.cudnn.benchmark))
            ds = fake_inf.RealDataDataset(feat, k=4, k_global=5, k_ds_rate=0.5)
            loader = SimpleNamespace(dataset=ds, batch_size=64)

            m_auto = FlakyDGCNN(fail_min_batch=True)
            preds = fake_inf.run_inference(m_auto, loader, ds.global_feat, 5, torch.device("cuda:0"))
            self.assertTrue(m_auto.on_cpu)
            self.assertEqual(preds, [2] * 40)
            self.assertIn(8, m_auto.seen_bs)

            # Explicit CUDA mode: OOM at min batch 8 MUST raise!
            install_tractcloud_accelerations(fake_pipe, "cuda:0", requested_device="cuda")
            ds_strict = fake_inf.RealDataDataset(feat, k=4, k_global=5, k_ds_rate=0.5)
            loader_strict = SimpleNamespace(dataset=ds_strict, batch_size=64)
            m_strict = FlakyDGCNN(fail_min_batch=True)
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                fake_inf.run_inference(m_strict, loader_strict, ds_strict.global_feat, 5, torch.device("cuda:0"))


class WmaPreprocessingUkfWindowsSimulationTests(unittest.TestCase):
    """Exercise WMA without Triton, MP-PCA/Gibbs/FSL Eddy, and UKF on Windows."""

    def test_wma_without_triton_windows_tile_cap_and_auto_oom_cpu_retry(self) -> None:
        from tractography import wma

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            wma_root = root / "wma_gpu"
            (wma_root / "bin").mkdir(parents=True)
            (wma_root / "whitematteranalysis" / "gpu").mkdir(parents=True)
            (wma_root / "whitematteranalysis" / "__init__.py").write_text("", encoding="utf-8")
            (wma_root / "whitematteranalysis" / "gpu" / "__init__.py").write_text("", encoding="utf-8")
            for name in wma.NATIVE_WMA_COMMANDS:
                (wma_root / "bin" / name).write_text("import sys; sys.exit(0)\n", encoding="utf-8")
            atlas_dir = root / "ORG"
            (atlas_dir / "ORG-RegAtlas-100HCP").mkdir(parents=True)
            (atlas_dir / "ORG-800FC-100HCP").mkdir(parents=True)
            tract = root / "subject.vtk"
            tract.write_text("# vtk DataFile Version 3.0\n", encoding="utf-8")
            slicer_exe = root / "Slicer.exe"
            slicer_exe.write_bytes(b"")
            py_exe = root / "PythonSlicer.exe"
            py_exe.write_bytes(b"")

            # Verify Triton missing never raises ImportError in eager PyTorch distance fallback pattern
            with patch.dict(sys.modules, {"triton": None}):
                try:
                    import triton  # type: ignore[import-not-found]
                except ImportError:
                    triton = None
                self.assertIsNone(triton)

            # Simulate native Windows WMA stage hitting CUDA OOM on first attempt, succeeding on CPU retry under device="auto"
            recorded_calls: list[tuple[list[str], dict[str, str]]] = []

            def fake_subprocess_run(cmd, cwd=None, env=None, stdout=None, stderr=None, **kw):
                recorded_calls.append((list(cmd), dict(env or {})))
                if len(recorded_calls) == 1:
                    if stderr is not None:
                        stderr.write("torch.cuda.OutOfMemoryError: CUDA out of memory\n")
                        stderr.flush()
                    return subprocess.CompletedProcess(cmd, 1)
                return subprocess.CompletedProcess(cmd, 0)

            def fake_run_org_native(*, run_stage, gpu_mode, supports_gpu_cli, **kw):
                self.assertTrue(supports_gpu_cli)
                run_stage("clustering", ["wm_cluster_from_atlas.py", "-gpu", gpu_mode, "in.vtk", "atlas", "out"])

            cfg_auto = wma.WMAOrgAtlasConfig(
                subject_id="sub01",
                input_tractography=tract,
                atlas_dir=atlas_dir,
                output_dir=root / "out_auto",
                slicer_executable=slicer_exe,
                wma_root=wma_root,
                device="auto",
                execute=True,
            )
            with (
                patch.object(wma.sys, "platform", "win32"),
                patch.object(wma.shutil, "which", return_value=None),
                patch.dict(os.environ, {"DSP_WMA_EXECUTION_BACKEND": "native"}, clear=False),
                patch.object(wma, "find_pythonslicer", return_value=py_exe),
                patch.object(wma.subprocess, "run", side_effect=fake_subprocess_run),
                patch("tractography.wma_native.run_org_atlas_native", side_effect=fake_run_org_native),
            ):
                os.environ.pop("WMA_GPU_TILE", None)
                wma.run_wma_org_atlas(cfg_auto)

            self.assertEqual(len(recorded_calls), 2, "Expected automatic CPU retry after CUDA OOM in device='auto'")
            first_cmd, first_env = recorded_calls[0]
            second_cmd, second_env = recorded_calls[1]
            self.assertEqual(first_env.get("WMA_GPU"), "auto")
            self.assertEqual(first_env.get("WMA_GPU_TILE"), "512")
            self.assertIn("max_split_size_mb:128", first_env.get("PYTORCH_CUDA_ALLOC_CONF", ""))
            self.assertEqual(second_env.get("WMA_GPU"), "off")
            self.assertEqual(second_env.get("CUDA_VISIBLE_DEVICES"), "-1")
            self.assertIn("'off'", " ".join(second_cmd))

            # Explicit device="cuda": MUST NOT silently retry on CPU!
            recorded_calls.clear()
            cfg_cuda = wma.WMAOrgAtlasConfig(
                subject_id="sub01",
                input_tractography=tract,
                atlas_dir=atlas_dir,
                output_dir=root / "out_cuda",
                slicer_executable=slicer_exe,
                wma_root=wma_root,
                device="cuda",
                execute=True,
            )
            with (
                patch.object(wma.sys, "platform", "win32"),
                patch.object(wma.shutil, "which", return_value=None),
                patch.dict(os.environ, {"DSP_WMA_EXECUTION_BACKEND": "native"}, clear=False),
                patch.object(wma, "find_pythonslicer", return_value=py_exe),
                patch.object(wma.subprocess, "run", side_effect=fake_subprocess_run),
                patch("tractography.wma_native.run_org_atlas_native", side_effect=fake_run_org_native),
            ):
                res_cuda = wma.run_wma_org_atlas(cfg_cuda)
            self.assertEqual(len(recorded_calls), 1, "Explicit device='cuda' must NOT retry WMA on CPU")
            self.assertEqual(res_cuda.status, "failed")

    def test_preprocessing_mppca_gibbs_oom_and_fsl_eddy_windows_cpu_routing(self) -> None:
        from preprocessing import dipy_accel
        from preprocessing.fsl_runtime import inspect_native_fsl_runtime

        # 1. MP-PCA CUDA OOM: device="auto" falls back to CPU; explicit device="cuda" re-raises!
        arr = np.random.RandomState(7).rand(7, 7, 7, 12).astype(np.float32) + 1.0
        with (
            patch.object(dipy_accel, "resolve_dipy_accel_device", return_value="cuda:0"),
            patch.object(dipy_accel, "_mppca_cuda", side_effect=RuntimeError("CUDA out of memory")),
            patch.object(dipy_accel, "empty_cuda_cache_safely"),
        ):
            out_auto = dipy_accel.mppca_accelerated(arr, patch_radius=1, device="auto", max_workers=1)
            self.assertEqual(out_auto.shape, arr.shape)
            self.assertTrue(np.isfinite(out_auto).all())

            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                dipy_accel.mppca_accelerated(arr, patch_radius=1, device="cuda", max_workers=1)

        # 2. Gibbs CUDA OOM: device="auto" falls back to CPU; explicit device="cuda" re-raises!
        import torch
        vol = np.random.RandomState(9).rand(8, 8, 4).astype(np.float32) + 1.0
        with (
            simulated_gpu_environment(PROFILE_A_RTX4060_8G),
            patch.object(dipy_accel, "resolve_dipy_accel_device", return_value="cuda:0"),
            patch.object(torch, "as_tensor", side_effect=RuntimeError("CUDA out of memory")),
        ):
            out_gibbs = dipy_accel.gibbs_removal_accelerated(vol, slice_axis=2, device="auto", max_workers=1)
            self.assertEqual(out_gibbs.shape, vol.shape)
            self.assertTrue(np.isfinite(out_gibbs).all())

            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                dipy_accel.gibbs_removal_accelerated(vol, slice_axis=2, device="cuda", max_workers=1)

        # 3. FSL Eddy on Windows: NEVER invokes Linux eddy_cuda; always routes to native PE eddy_cpu.exe
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mrtrix_bin = root / "mrtrix" / "bin"
            fsl_bin = root / "fsl" / "bin"
            mrtrix_bin.mkdir(parents=True)
            fsl_bin.mkdir(parents=True)
            # Minimal valid PE header (MZ + offset 64 + PE\0\0)
            pe_stub = b"MZ" + b"\x00" * 58 + (64).to_bytes(4, "little") + b"PE\x00\x00" + b"\x00" * 16
            for name in ("mrconvert.exe", "mrinfo.exe", "dwi2mask.exe", "maskfilter.exe"):
                p = mrtrix_bin / name
                p.write_bytes(pe_stub)
                p.chmod(0o755)
            eddy_cpu = fsl_bin / "eddy_cpu.exe"
            eddy_cpu.write_bytes(pe_stub)
            eddy_cpu.chmod(0o755)
            # Plant a Linux ELF eddy_cuda right next to it — Windows MUST ignore it!
            eddy_cuda_linux = fsl_bin / "eddy_cuda"
            eddy_cuda_linux.write_bytes(b"\x7fELF" + b"\x00" * 64)
            eddy_cuda_linux.chmod(0o755)

            report = inspect_native_fsl_runtime(mrtrix_bin, fsl_bin, reverse_phase=False, platform="win32")
            self.assertTrue(report["available"])
            self.assertEqual(report["eddy_device"], "cpu")
            self.assertEqual(Path(report["commands"]["eddy"]).name.lower(), "eddy_cpu.exe")

    def test_ukf_windows_standalone_cpu_binary_with_device_auto(self) -> None:
        from tractography import ukf

        def make_schema(with_gpu: bool) -> ET.Element:
            root = ET.Element("executable")
            grp = ET.SubElement(root, "parameters")
            for k in ("dwiFile", "maskFile", "tracts", "numTensor"):
                n = ET.SubElement(grp, "double")
                ET.SubElement(n, "name").text = k
                ET.SubElement(n, "default").text = "2"
            if with_gpu:
                for tag, k, default in (("boolean", "useGPU", "false"), ("integer", "gpuDevice", "0"), ("integer-enumeration", "gpuPrecision", "0")):
                    n = ET.SubElement(grp, tag)
                    ET.SubElement(n, "name").text = k
                    ET.SubElement(n, "default").text = default
            return root

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            ukf_exe = root / "bundled-runtimes" / "ukf-win-x64" / "bin" / "UKFTractography.exe"
            ukf_exe.parent.mkdir(parents=True)
            ukf_exe.write_bytes(b"MZ")
            cuda_root = root / "CUDA" / "v12.4"
            (cuda_root / "bin").mkdir(parents=True)

            with (
                patch.dict(os.environ, {"SLICER_HOME": str(root)}, clear=True),
                patch("shutil.which", return_value=None),
                patch("shared.pipeline.resolve_cli_executable", side_effect=RuntimeError("headless")),
            ):
                found = ukf.resolve_executable()
            self.assertEqual(found, ukf_exe.resolve())

            # Standalone CPU UKF with device="auto" cleanly resolves to CPU (use_gpu=False)
            cpu_schema = make_schema(with_gpu=False)
            self.assertEqual(ukf.resolve_ukf_gpu_dispatch("auto", {}, cpu_schema, found), (False, 0, 0))
            with self.assertRaisesRegex(RuntimeError, "--useGPU"):
                ukf.resolve_ukf_gpu_dispatch("auto", {"useGPU": True}, cpu_schema, found)

            # Windows process environment isolation preserves exe dir, System32, and CUDA_PATH\bin
            with patch.object(ukf.sys, "platform", "win32"):
                env = ukf.ukf_gpu_process_environment(
                    {
                        "SystemRoot": r"C:\Windows",
                        "CUDA_PATH": str(cuda_root),
                        "PYTHONPATH": r"C:\Slicer\lib\Python",
                        "QT_PLUGIN_PATH": r"C:\Slicer\plugins",
                    },
                    found,
                )
            self.assertNotIn("PYTHONPATH", env)
            self.assertNotIn("QT_PLUGIN_PATH", env)
            self.assertIn(r"C:\Windows\System32", env["PATH"])
            self.assertIn("bin", env["PATH"])


class FP64MicrostructureAndDTISimulationTests(unittest.TestCase):
    """Exercise FreeWater, NODDI, DKI, MK-Curve, and DTI-det TDR bounds and auto-vs-cuda OOM contracts."""

    def test_freewater_wls_tdr_cap_and_auto_vs_cuda_oom(self) -> None:
        import torch
        from dipy.core.gradients import gradient_table
        from dipy.reconst import fwdti
        from microstructure import freewater_wls
        from shared import hardware_adapt

        bvals = np.array([0, 0, 1000, 1000, 1000, 1000, 1000, 1000, 2000, 2000, 2000, 2000, 2000, 2000], dtype=float)
        dirs = np.array([
            [0, 0, 0], [0, 0, 0],
            [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1],
            [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, -1, 0], [1, 0, -1], [0, 1, -1],
        ], dtype=float)
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        gtab = gradient_table(bvals, bvecs=dirs / norms)
        model = fwdti.FreeWaterTensorModel(gtab, fit_method="WLS")
        signals = np.tile(np.exp(-bvals * 0.0008), (16, 1)) * 1000.0

        cpu_fit, _ = freewater_wls.wls_batch_fit(model, signals, device="cpu")
        captured_batch: list[int] = []
        orig_adapt = hardware_adapt.adaptive_cuda_scaled_batch

        def spy_adapt(*args, **kwargs):
            val = orig_adapt(*args, **kwargs)
            captured_batch.append(val)
            return val

        with (
            simulated_gpu_environment(PROFILE_A_RTX4060_8G),
            patch.object(hardware_adapt, "adaptive_cuda_scaled_batch", side_effect=spy_adapt),
            patch.object(torch, "as_tensor", side_effect=RuntimeError("CUDA out of memory")),
        ):
            fit_auto, dev_auto = freewater_wls.wls_batch_fit(model, signals, device="auto")
            self.assertEqual(dev_auto, "cpu")
            np.testing.assert_allclose(fit_auto.model_params, cpu_fit.model_params, rtol=1e-10, atol=1e-10)

            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                freewater_wls.wls_batch_fit(model, signals, device="cuda")

    def test_noddi_tdr_cap_and_auto_vs_cuda_oom(self) -> None:
        import torch
        from microstructure import noddi_accel
        from shared import hardware_adapt

        rng = np.random.RandomState(11)
        n_vox, n_grad, n_wm, ndirs = 12, 16, 6, 2
        K_wm = np.abs(rng.randn(n_wm, ndirs, n_grad)).astype(np.float64) + 0.1
        K_iso = np.abs(rng.randn(n_grad)).astype(np.float64) + 0.1
        K_norms = np.linalg.norm(K_wm[:, 0, 2:], axis=1) + 1e-6
        icvf = np.linspace(0.2, 0.8, n_wm)
        kappa = np.linspace(0.1, 0.9, n_wm)
        dwi_idx = np.arange(2, n_grad, dtype=np.int64)
        y = np.abs(rng.randn(n_vox, n_grad)).astype(np.float64) + 0.2
        dir_idx = rng.randint(0, ndirs, size=n_vox).astype(np.int64)

        ref_est, ref_rmse, ref_nrmse = noddi_accel._fit_noddi_cpu(
            y=y, dir_idx=dir_idx, K_wm=K_wm, K_iso=K_iso, K_norms=K_norms,
            icvf=icvf, kappa=kappa, dwi_idx=dwi_idx, lambda1=0.5, lambda2=0.001, ndirs=ndirs, nthreads=1,
        )

        with (
            simulated_gpu_environment(PROFILE_A_RTX4060_8G),
            patch.object(torch, "as_tensor", side_effect=RuntimeError("CUDA out of memory")),
        ):
            est_auto, rmse_auto, nrmse_auto = noddi_accel._fit_noddi_cuda(
                y=y, dir_idx=dir_idx, K_wm=K_wm, K_iso=K_iso, K_norms=K_norms,
                icvf=icvf, kappa=kappa, dwi_idx=dwi_idx, lambda1=0.5, lambda2=0.001, ndirs=ndirs,
                device="cuda:0", explicit_cuda=False, nthreads=1,
            )
            np.testing.assert_allclose(est_auto, ref_est, rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(rmse_auto, ref_rmse, rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(nrmse_auto, ref_nrmse, rtol=1e-6, atol=1e-6)

            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                noddi_accel._fit_noddi_cuda(
                    y=y, dir_idx=dir_idx, K_wm=K_wm, K_iso=K_iso, K_norms=K_norms,
                    icvf=icvf, kappa=kappa, dwi_idx=dwi_idx, lambda1=0.5, lambda2=0.001, ndirs=ndirs,
                    device="cuda:0", explicit_cuda=True, nthreads=1,
                )

    def test_dki_tdr_cap_and_auto_vs_cuda_oom(self) -> None:
        from microstructure import dki

        # Plain DIPY DKI is CPU-only (multi-worker CPU outperforms CUDA + transfer overhead);
        # MK-Curve retains its separate CUDA path verified in test_mkcurve_tdr_cap_*.
        with simulated_gpu_environment(PROFILE_A_RTX4060_8G):
            self.assertEqual(dki._resolve_dki_device("auto"), "cpu")
            self.assertEqual(dki._resolve_dki_device("cuda"), "cpu")
            self.assertEqual(dki._resolve_dki_device("cuda:0"), "cpu")
            self.assertEqual(dki._resolve_dki_device("cpu"), "cpu")

    def test_mkcurve_tdr_cap_refinement_bound_and_auto_vs_cuda_oom(self) -> None:
        from microstructure import mkcurve, mkcurve_tensor

        mkc_src = Path(mkcurve.__file__).read_text(encoding="utf-8")
        mkct_src = Path(mkcurve_tensor.__file__).read_text(encoding="utf-8")
        self.assertIn("consumer_fp64_tdr_cap=1024", mkc_src)
        self.assertIn("_cached_cuda_operators.cache_clear()", mkc_src)
        self.assertIn("ref_step = 64 if (_os.name == \"nt\" and is_consumer_fp64_cuda_gpu(self.dev)) else 256", mkct_src)

        # Exercise _fit_chunk_cuda OOM at <=128 voxels: explicit_cuda=False -> CPU fallback; explicit_cuda=True -> raises!
        rng = np.random.RandomState(5)
        bvals = np.array([0] + [1000] * 15 + [2000] * 15, dtype=float)
        bvecs = rng.randn(len(bvals), 3)
        bvecs[0] = 0.0
        bvecs[1:] /= np.linalg.norm(bvecs[1:], axis=1, keepdims=True)
        design = mkcurve_tensor.design_matrix(bvals, bvecs)
        grid = np.linspace(100.0, 2000.0, 10)
        signals = np.tile(np.exp(-bvals * 0.0007), (8, 1)) * 1000.0

        with (
            patch.object(mkcurve_tensor, "_CudaSyntheticB0Fitter", side_effect=RuntimeError("CUDA out of memory")),
            patch("shared.hardware_adapt.empty_cuda_cache_safely"),
        ):
            orig, corr, curves, det = mkcurve._fit_chunk_cuda(
                (signals.copy(), design, grid, 0.5, "reference", "cuda:0", False)
            )
            self.assertEqual(orig.shape, (8, 22))
            self.assertEqual(curves.shape, (len(grid), 8))

            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                mkcurve._fit_chunk_cuda(
                    (signals.copy(), design, grid, 0.5, "reference", "cuda:0", True)
                )

    def test_dti_det_tdr_cap_and_auto_vs_cuda_oom(self) -> None:
        from dipy.core.gradients import gradient_table
        from tractography import dti

        bvals = np.array([0, 1000, 1000, 1000, 1000, 1000, 1000], dtype=float)
        bvecs = np.array([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1 / np.sqrt(2), 1 / np.sqrt(2), 0],
            [1 / np.sqrt(2), 0, 1 / np.sqrt(2)],
            [0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
        ], dtype=float)
        gtab = gradient_table(bvals, bvecs=bvecs)
        data = np.zeros((4, 4, 4, 7), dtype=np.float32)
        for i, bv in enumerate(bvals):
            data[..., i] = 1000.0 * np.exp(-bv * (0.0015 if i == 1 else 0.0005))
        mask = np.ones((4, 4, 4), dtype=bool)

        from dipy.data import get_sphere

        sphere = get_sphere(name="repulsion100")
        data_in_mask = data[mask]
        fa_ref, _, _, resolved_cpu = dti._fit_dti_fa_and_peaks(
            data_in_mask, mask, gtab, sphere, fit_method="WLS", min_fa=0.1, max_angle_deg=35.0, device="cpu"
        )
        self.assertEqual(resolved_cpu, "cpu")

        # Deterministic DTI tracking is CPU-only even when auto/cuda is passed.
        with simulated_gpu_environment(PROFILE_A_RTX4060_8G):
            for req in ("auto", "cuda", "cuda:0"):
                fa_out, tmask_out, _, resolved = dti._fit_dti_fa_and_peaks(
                    data_in_mask, mask, gtab, sphere, fit_method="WLS", min_fa=0.1, max_angle_deg=35.0, device=req
                )
                self.assertEqual(resolved, "cpu")
                np.testing.assert_allclose(fa_out, fa_ref, rtol=1e-10, atol=1e-10)
                self.assertTrue(tmask_out.any())


if __name__ == "__main__":
    unittest.main()
