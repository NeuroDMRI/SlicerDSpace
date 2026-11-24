"""Round-5 surgical regression tests covering all 5 review findings:
1. P0: MK-Curve PORT_VERSION=4 invalidates pre-07fafcc v3 cached manifests while v4 caches reuse cleanly.
2. P1: Gibbs CUDA timeout/error fallback under inplace=True skips completed slices (no double-processing).
3. P1: DDParcel keeps exact float32 precision in both ram_fp32 and memmap_fp32 modes (zero near-tie flips).
4. P1: Windows CUDA Setup wheel replacement when PyTorch extension already loaded + multi-GPU DDSurfer preflight + global VRAM cap in hardware_adapt.
5. P2: External CUDA UKF cohort GPU slot reservation + isolated WDDM compute-apps telemetry failure handling.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_DIPY = importlib.util.find_spec("dipy") is not None
HAS_SCIPY = importlib.util.find_spec("scipy") is not None
HAS_NIBABEL = importlib.util.find_spec("nibabel") is not None


class MkCurveV4CacheInvalidationTests(unittest.TestCase):
    """Finding 1 (P0): Stale pre-07fafcc MK-Curve v3 cache must miss and recompute; v4 must hit."""

    def test_mkcurve_port_version_4_rejects_v3_and_accepts_v4_backend(self) -> None:
        from microstructure.dki import _mkcurve_backend_matches
        from microstructure.mkcurve import PORT_VERSION, inspect_mkcurve_backend

        self.assertGreaterEqual(PORT_VERSION, 4)
        current = inspect_mkcurve_backend()
        self.assertEqual(current["port_version"], PORT_VERSION)

        # Pre-07fafcc v3 (and older v2) must be rejected even if data files match
        self.assertFalse(_mkcurve_backend_matches(dict(current, port_version=3), current))
        self.assertFalse(_mkcurve_backend_matches(dict(current, port_version=2), current))
        self.assertFalse(_mkcurve_backend_matches(dict(current, port_version=None), current))

        # Fresh v4 hits cache cleanly even across pure .py edits, while data-asset changes still miss
        diff_py = dict(current, source_sha256={**current["source_sha256"], "mkcurve.py": "f" * 64})
        self.assertTrue(_mkcurve_backend_matches(diff_py, current))
        diff_dirs = dict(current, source_sha256={**current["source_sha256"], "mkcurve_directions.json": "0" * 64})
        self.assertFalse(_mkcurve_backend_matches(diff_dirs, current))

    @unittest.skipUnless(HAS_NIBABEL and HAS_SCIPY and HAS_DIPY, "nibabel/scipy/dipy unavailable")
    def test_dki_outputs_current_and_run_dki_recompute_stale_v3_corrupted_mkcurve_cache(self) -> None:
        import nibabel as nib
        from microstructure.dki import DkiConfig, dki_output_paths, dki_outputs_current, run_dki
        from microstructure.mkcurve import PORT_VERSION
        from shared.pipeline import RawInputBundle

        # Build a tiny valid multi-shell DKI synthetic DWI dataset (3 shells: b=0, 1000, 2000; 21 dirs/shell)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(20261124)
            n_b0 = 3
            n_dir = 21
            raw_dirs = rng.standard_normal((n_dir, 3))
            raw_dirs /= np.linalg.norm(raw_dirs, axis=1, keepdims=True)
            bvals = np.concatenate([np.zeros(n_b0), np.full(n_dir, 1000.0), np.full(n_dir, 2000.0)])
            bvecs = np.vstack([np.zeros((n_b0, 3)), raw_dirs, raw_dirs])

            # Physically realistic multi-shell signal: S(b, g) = S0 * exp(-b*D + (1/6)*b^2*D^2*K)
            shape = (3, 3, 3)
            s0 = 1000.0
            d_iso = 0.0009
            k_iso = 0.75
            sig_1d = s0 * np.exp(-bvals * d_iso + (1.0 / 6.0) * (bvals ** 2) * (d_iso ** 2) * k_iso)
            dwi = np.broadcast_to(sig_1d.astype(np.float32), shape + (len(bvals),)).copy()
            dwi += rng.normal(0.0, 2.0, size=dwi.shape).astype(np.float32)
            dwi = np.clip(dwi, 1.0, None)
            mask = np.zeros(shape, dtype=np.uint8)
            mask[1, 1, 1] = 1
            mask[1, 1, 2] = 1

            affine = np.diag([-2.0, -2.0, 2.0, 1.0])
            dwi_p = root / "dwi.nii.gz"
            bval_p = root / "dwi.bval"
            bvec_p = root / "dwi.bvec"
            mask_p = root / "mask.nii.gz"
            nib.save(nib.Nifti1Image(dwi, affine), str(dwi_p))
            nib.save(nib.Nifti1Image(mask, affine), str(mask_p))
            np.savetxt(bval_p, bvals[None, :], fmt="%.1f")
            np.savetxt(bvec_p, bvecs.T, fmt="%.6f")

            bundle = RawInputBundle(
                subject_root=root,
                subject_id="sub_r5",
                input_dir=root,
                dwi_path=dwi_p,
                bval_path=bval_p,
                bvec_path=bvec_p,
                mask_path=mask_p,
            )
            out_dir = root / "dki_out"
            cfg = DkiConfig(
                subject_id="sub_r5",
                raw_input_bundle=bundle,
                output_dir=out_dir,
                fit_method="MKCURVE",
                num_threads=1,
                mkcurve_workers=1,
            )

            with patch("microstructure.mkcurve.synthetic_b0_grid", return_value=np.arange(100.0, 2200.0, 100.0)):
                # 1. Fresh run produces v4 cache
                first = run_dki(cfg)
                self.assertEqual(first.status, "completed")
                paths = dki_output_paths("sub_r5", out_dir)
                manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
                self.assertEqual(manifest["signature"]["mkcurve_backend"]["port_version"], PORT_VERSION)
                self.assertEqual(manifest["mkcurve"]["backend"]["port_version"], PORT_VERSION)
                self.assertTrue(dki_outputs_current(cfg))
                self.assertEqual(run_dki(cfg).status, "cached")

                # 2. Downgrade manifest & record to stale pre-07fafcc v3 (simulating corrupted v3 cache)
                manifest["signature"]["mkcurve_backend"]["port_version"] = 3
                manifest["mkcurve"]["backend"]["port_version"] = 3
                paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")

                self.assertFalse(dki_outputs_current(cfg))

                # 3. Re-running MUST miss cache and recompute fresh v4 manifest
                recomputed = run_dki(cfg)
                self.assertEqual(recomputed.status, "completed")
                refreshed = json.loads(paths["manifest"].read_text(encoding="utf-8"))
                self.assertEqual(refreshed["signature"]["mkcurve_backend"]["port_version"], PORT_VERSION)
                self.assertTrue(dki_outputs_current(cfg))


@unittest.skipUnless(HAS_TORCH and HAS_SCIPY, "torch/scipy unavailable")
class GibbsInplaceTimeoutFallbackTests(unittest.TestCase):
    """Finding 2 (P1): Gibbs GPU timeout/error fallback under inplace=True must never double-process slices."""

    def test_inplace_late_cuda_timeout_fallback_matches_clean_cpu_baseline(self) -> None:
        import torch
        from preprocessing import dipy_accel as da

        rng = np.random.default_rng(5150)
        # 0-1000 intensity scale with sharp edges where double-Gibbs causes huge distortion (~200-500+)
        vol = np.zeros((16, 16, 10, 6), dtype=np.float32)
        vol[3:13, 3:13, :, :] = 800.0 + 120.0 * rng.standard_normal((10, 10, 10, 6)).astype(np.float32)
        vol = np.clip(vol, 0.0, 1000.0)

        baseline = da.gibbs_removal_accelerated(
            vol.copy(), slice_axis=2, n_points=3, inplace=False, device="cpu", max_workers=1
        )
        # Sanity check: double-processing WOULD corrupt the volume significantly
        double_processed = da.gibbs_removal_accelerated(
            baseline.copy(), slice_axis=2, n_points=3, inplace=False, device="cpu", max_workers=1
        )
        self.assertGreater(float(np.max(np.abs(double_processed - baseline))), 5.0)

        # Simulate GPU chunked execution (3 chunks of 20 slices = 60 slices total):
        # Chunk 1 [0:20) succeeds and scatters into `vol_inplace`, then Chunk 2 [20:40) hits WDDM TDR timeout!
        real_gibbs_1d = da._gibbs_1d_batch_torch
        real_torch_device = torch.device
        call_counter = {"n": 0}

        def _flaky_gibbs_1d(x, *, axis: int, n_points: int = 3):
            call_counter["n"] += 1
            # Each _gpu_step calls _gibbs_1d_batch_torch twice (axis=1, axis=0).
            # Fail on call 3 (start of 2nd chunk, AFTER 1st chunk was scattered into vol_inplace).
            if call_counter["n"] >= 3:
                raise RuntimeError("CUDA error: the launch timed out and was terminated")
            return real_gibbs_1d(x, axis=axis, n_points=n_points)

        vol_inplace = vol.copy()
        with (
            patch.object(da, "resolve_dipy_accel_device", return_value="cuda:0"),
            patch.object(da, "adaptive_cuda_scaled_batch", return_value=20),
            patch.object(da, "_gibbs_1d_batch_torch", side_effect=_flaky_gibbs_1d),
            patch.object(da, "empty_gpu_cache_safely"),
            patch.object(torch, "device", side_effect=lambda spec: real_torch_device("cpu")),
        ):
            out = da.gibbs_removal_accelerated(
                vol_inplace, slice_axis=2, n_points=3, inplace=True, device="auto", max_workers=1
            )

        self.assertIs(out, vol_inplace)
        self.assertGreaterEqual(call_counter["n"], 3)
        np.testing.assert_allclose(out, baseline, rtol=1e-5, atol=2e-3)

        # Verify explicit CUDA mode still raises on TDR timeout without silent CPU fallback
        call_counter["n"] = 0
        with (
            patch.object(da, "resolve_dipy_accel_device", return_value="cuda:0"),
            patch.object(da, "adaptive_cuda_scaled_batch", return_value=20),
            patch.object(da, "_gibbs_1d_batch_torch", side_effect=_flaky_gibbs_1d),
            patch.object(da, "empty_gpu_cache_safely"),
            patch.object(torch, "device", side_effect=lambda spec: real_torch_device("cpu")),
            self.assertRaisesRegex(RuntimeError, "launch timed out"),
        ):
            da.gibbs_removal_accelerated(
                vol.copy(), slice_axis=2, n_points=3, inplace=True, device="cuda:0", max_workers=1
            )


@unittest.skipUnless(HAS_TORCH, "torch unavailable")
class DDParcelFp32ExactNearTieTests(unittest.TestCase):
    """Finding 3 (P1): DDParcel probability buffer must keep full float32 precision in RAM and memmap modes."""

    def _import_ddsurfer_pred(self):
        try:
            from anatomy.parcellation import DDSurfer_Pred as dd_pred
            return dd_pred
        except ImportError as exc:
            self.skipTest(f"DDSurfer_Pred optional dependencies unavailable: {exc}")

    def test_ddparcel_allocates_ram_fp32_or_memmap_fp32_and_eliminates_near_tie_flips(self) -> None:
        import torch
        from shared.hardware_adapt import GIB

        dd_pred = self._import_ddsurfer_pred()

        # 1. Verify RAM vs memmap selection threshold (avail >= 8 GiB and fp32 <= 0.60 * avail -> ram_fp32)
        small_shape = (1, 16, 16, 16, 82)
        fp32_bytes = int(np.prod(small_shape)) * 4

        buf_ram = dd_pred._allocate_prediction_probability(small_shape, avail_ram_bytes=32 * GIB)
        try:
            self.assertEqual(buf_ram.mode, "ram_fp32")
            self.assertEqual(buf_ram.tensor.dtype, torch.float32)
            self.assertEqual(buf_ram.nbytes, fp32_bytes)
        finally:
            buf_ram.close()

        # Tight RAM (< 8 GiB) -> memmap_fp32
        with tempfile.TemporaryDirectory() as td:
            buf_mm = dd_pred._allocate_prediction_probability(small_shape, avail_ram_bytes=6 * GIB, temp_root=td)
            backing_dir = buf_mm._backing_dir
            try:
                self.assertEqual(buf_mm.mode, "memmap_fp32")
                self.assertEqual(buf_mm.tensor.dtype, torch.float32)
                self.assertEqual(buf_mm.nbytes, fp32_bytes)
                self.assertTrue(Path(backing_dir, "pred_prob_fp32.npy").is_file())
            finally:
                buf_mm.close()
            self.assertFalse(Path(backing_dir).exists())

        # High-ish RAM (8.2 GiB >= 8 GiB) where full 256^3x82 fp32 (~5.125 GiB) > 60% of 8.2 GiB (4.92 GiB) -> memmap_fp32
        fake_mm = np.zeros((1, 2, 2, 2, 4), dtype=np.float32)
        with (
            tempfile.TemporaryDirectory() as td,
            patch.object(np.lib.format, "open_memmap", return_value=fake_mm) as mock_open_mm,
        ):
            buf_ratio = dd_pred._allocate_prediction_probability(
                (1, 256, 256, 256, 82), avail_ram_bytes=int(8.2 * GIB), temp_root=td
            )
            try:
                self.assertEqual(buf_ratio.mode, "memmap_fp32")
                mock_open_mm.assert_called_once()
                self.assertEqual(mock_open_mm.call_args.kwargs["dtype"], np.float32)
            finally:
                buf_ratio.close()

        # 2. Construct near-tie voxels where FP16 rounding flips labels, and verify 0 flips in ram_fp32 & memmap_fp32
        nx, ny, nz, nc = 8, 8, 8, 12
        rng = np.random.default_rng(4242)
        ax = np.full((nx, ny, nz, nc), -5.0, dtype=np.float32)
        cor = np.full((nx, ny, nz, nc), -5.0, dtype=np.float32)
        sag = np.full((nx, ny, nz, nc), -5.0, dtype=np.float32)

        # Two competing classes (c1, c2) with magnitude ~12.0 and FP32 difference ~0.001 (smaller than FP16 ULP 0.0078)
        # Design Axial + Coronal so FP16 rounds c1 down and c2 up, flipping their ordering!
        ax[..., 1] = 15.0040
        cor[..., 1] = 15.0040  # 0.4*15.004 + 0.4*15.004 = 12.0032
        sag[..., 1] = 0.0

        ax[..., 2] = 15.0000
        cor[..., 2] = 15.0000  # 0.4*15.000 + 0.4*15.000 = 12.0000
        sag[..., 2] = 0.0080   # + 0.2*0.008 = 12.0016 -> FP32 winner is class 1 (12.0032 > 12.0016)!
        # In FP16: 0.4*15.004 = 6.0016 -> rounds to 6.000 in FP16! 6.000 + 6.000 = 12.000 -> +0.0 = 12.000
        # while class 2 gets 12.000 + 0.0016 = 12.0016 -> FP16 FLIPS winner to class 2!

        # Add randomized near-tie noise across voxels
        jitter = rng.uniform(-0.0003, 0.0003, size=(nx, ny, nz)).astype(np.float32)
        ax[..., 1] += jitter

        ref_prob = torch.from_numpy(ax) * 0.4 + torch.from_numpy(cor) * 0.4 + torch.from_numpy(sag) * 0.2
        _, ref_idx = torch.max(ref_prob, 3)
        ref_labels = ref_idx.numpy().astype(np.int16)

        # Prove legacy FP16 accumulation flips labels on these exact near-tie voxels
        fp16_buf = torch.zeros((1, nx, ny, nz, nc), dtype=torch.float16)
        dd_pred._accumulate_prob_slice(fp16_buf[0], torch.from_numpy(ax) * 0.4, overwrite=True)
        dd_pred._accumulate_prob_slice(fp16_buf[0], torch.from_numpy(cor) * 0.4)
        _, fp16_idx = torch.max(fp16_buf[0].float() + torch.from_numpy(sag) * 0.2, 3)
        fp16_flips = int(np.count_nonzero(fp16_idx.numpy() != ref_labels))
        self.assertGreater(fp16_flips, 0, "Test fixture must trigger FP16 near-tie flips")

        # Verify BOTH ram_fp32 and memmap_fp32 produce 0 flips (100% exact FP32 parity)
        with tempfile.TemporaryDirectory() as td:
            for avail_ram, expected_mode in ((32 * GIB, "ram_fp32"), (4 * GIB, "memmap_fp32")):
                buf = dd_pred._allocate_prediction_probability(
                    (1, nx, ny, nz, nc), avail_ram_bytes=avail_ram, temp_root=td
                )
                try:
                    self.assertEqual(buf.mode, expected_mode)
                    prob = buf.tensor
                    dd_pred._accumulate_prob_slice(prob[0], torch.from_numpy(ax) * 0.4, overwrite=True)
                    dd_pred._accumulate_prob_slice(prob[0], torch.from_numpy(cor) * 0.4)
                    fused_labels = np.empty((nx, ny, nz), dtype=np.int16)
                    for x0 in range(0, nx, 2):
                        x1 = min(nx, x0 + 2)
                        combined = prob[0, x0:x1].float() + torch.from_numpy(sag[x0:x1]) * 0.2
                        _, idx = torch.max(combined, 3)
                        fused_labels[x0:x1] = idx.numpy()
                    np.testing.assert_array_equal(fused_labels, ref_labels)
                finally:
                    buf.close()


class WindowsCudaSetupMultiGpuAndVramCapTests(unittest.TestCase):
    """Finding 4 (P1): Setup CUDA wheel replacement + DDSurfer multi-GPU preflight + global VRAM cap."""

    def test_4a_setup_emits_and_executes_cuda_wheel_replacement_when_pytorch_ext_already_loaded(self) -> None:
        from shared import python_setup
        from shared.selected_setup import build_setup_plan, setup_steps
        from shared.setup_components import plan_components, required_licences
        from shared.ui_selected_setup import setup_plan_pending_actions

        torch_cpu_ok = SimpleNamespace(
            key="python:torch",
            kind="python",
            label="torch",
            status="available",
            required=True,
            targets=("segmentation",),
            install_group="python",
            install_hint="",
            details={
                "import": "torch",
                "package": "torch>=2.3.1",
                "installed_version": "2.15.0+cpu",
                "python_executable": None,
            },
        )
        cuda_missing = SimpleNamespace(
            key="device:segmentation:cuda",
            kind="device",
            label="segmentation requested device cuda",
            status="missing",
            required=True,
            targets=("segmentation",),
            install_group="device",
            install_hint="CPU-only PyTorch installed",
            details={
                "requested_device": "cuda",
                "cpu_only_torch": True,
                "torch_importable": True,
                "torch_version": "2.15.0+cpu",
                "torch_cuda_build": None,
            },
        )
        report = SimpleNamespace(statuses=(torch_cpu_ok, cuda_missing), targets=("segmentation",), blocker_count=1)

        # PyTorch extension IS already installed AND loaded on Windows
        plan = build_setup_plan(
            report,
            installed_extensions=("PyTorch",),
            loaded_extensions=["PyTorch"],
            need_torch=True,
            system="Windows",
            machine="AMD64",
        )
        self.assertEqual(plan["extensions"], [])
        self.assertEqual(plan["restart_extensions"], [])
        self.assertFalse(plan["cuda_torch_ready"])
        self.assertTrue(plan["cuda_torch_cpu_only"])
        self.assertTrue(plan["install_pytorch_cuda"])
        self.assertIn("torch>=2.3.1", plan["python_packages"])
        self.assertTrue(setup_plan_pending_actions(plan))
        self.assertTrue(
            any(
                req.get("requirement") == "torch>=2.3.1"
                and req.get("require_cuda") is True
                and req.get("replace_cpu_torch") is True
                for req in plan["python_requests"]
            ),
            plan["python_requests"],
        )

        # setup_steps MUST emit a python_packages step with CUDA wheel replacement flags
        accepted = {lic_id: {"version": rec["version"]} for lic_id, rec in required_licences(plan_components(plan)).items()}
        with tempfile.TemporaryDirectory() as td:
            steps = setup_steps(
                plan,
                accepted,
                folder=Path(td),
                launcher=("Slicer.exe", ["--launch", "PythonSlicer"]),
                python_command=lambda _ov: ("Slicer.exe", ["--launch", "PythonSlicer"]),
            )
            pkg_steps = [s for s in steps if s["id"] == "python_packages"]
            self.assertEqual(len(pkg_steps), 1)
            cmd_args = pkg_steps[0]["commands"][0][1]
            req_path = Path(cmd_args[cmd_args.index("--request") + 1])
            req_payload = json.loads(req_path.read_text(encoding="utf-8"))
            self.assertEqual(req_payload["torch_minimum"], "2.3.1")
            self.assertTrue(req_payload["require_cuda_torch"])
            self.assertTrue(req_payload["replace_cpu_torch"])

        # Verify install_requirements replaces CPU-only torch even when torch>=2.3.1 is version-satisfied
        state = {"torch_ver": "2.15.0+cpu"}
        install_calls = []

        class _FakeTorchLogic:
            def getCompatibleComputationBackend(self):
                return "cu124"

            def installTorch(self, askConfirmation=False, forceComputationBackend=None, torchVersionRequirement=None):
                install_calls.append((askConfirmation, forceComputationBackend, torchVersionRequirement))
                state["torch_ver"] = "2.5.1+cu124"
                return SimpleNamespace(__version__=state["torch_ver"])

        helper = SimpleNamespace(PyTorchUtilsLogic=lambda: _FakeTorchLogic())
        with (
            patch.dict("sys.modules", {"PyTorchUtils": helper}),
            patch.object(python_setup, "requirement_satisfied", return_value=True),
            patch.object(python_setup.importlib.metadata, "version", side_effect=lambda name: state["torch_ver"] if name == "torch" else "1.0"),
            patch.object(python_setup, "installed_versions", side_effect=lambda: {"torch": state["torch_ver"]}),
            patch.object(python_setup, "bundled_shortfalls", return_value=[]),
        ):
            rec = python_setup.install_requirements(
                (),
                loaded=("torch",),
                torch_minimum="2.3.1",
                require_cuda_torch=True,
                replace_cpu_torch=True,
            )
        self.assertEqual(rec["backend"], "PyTorch_extension")
        self.assertEqual(install_calls, [(False, "cu124", ">=2.3.1")])
        self.assertTrue(rec["restart_required"])
        self.assertEqual(rec["restart_packages"], ["torch"])

        # Guard: if CUDA torch IS already installed (e.g. +cu124) and device failed for driver/no-GPU, do not reinstall
        torch_cuda_ok = SimpleNamespace(
            key="python:torch",
            kind="python",
            label="torch",
            status="available",
            required=True,
            targets=("segmentation",),
            install_group="python",
            install_hint="",
            details={"import": "torch", "package": "torch>=2.3.1", "installed_version": "2.5.1+cu124"},
        )
        cuda_driver_fail = SimpleNamespace(
            key="device:segmentation:cuda",
            kind="device",
            label="segmentation requested device cuda",
            status="missing",
            required=True,
            targets=("segmentation",),
            install_group="device",
            install_hint="No NVIDIA GPU driver",
            details={"requested_device": "cuda", "torch_importable": True, "torch_version": "2.5.1+cu124", "torch_cuda_build": "12.4"},
        )
        plan_driver_fail = build_setup_plan(
            SimpleNamespace(statuses=(torch_cuda_ok, cuda_driver_fail), targets=("segmentation",), blocker_count=1),
            installed_extensions=("PyTorch",),
            loaded_extensions=["PyTorch"],
            need_torch=True,
            system="Windows",
            machine="AMD64",
        )
        self.assertFalse(plan_driver_fail["cuda_torch_cpu_only"])
        self.assertFalse(plan_driver_fail["install_pytorch_cuda"])
        self.assertEqual(plan_driver_fail["python_packages"], [])

    def test_4b_ddsurfer_multi_gpu_preflight_probes_resolved_cuda1_not_cuda0(self) -> None:
        from anatomy.surface.prediction import PredictionStageConfig, predict_subject
        from shared import environment

        environment._CUDA_FREE_CACHE.clear()
        environment._AUTO_TORCH_DEVICE_CACHE.clear()
        self.addCleanup(environment._CUDA_FREE_CACHE.clear)
        self.addCleanup(environment._AUTO_TORCH_DEVICE_CACHE.clear)

        probed_devices: list[str] = []

        def _fake_cuda_free(device="cuda:0", python_executable=None, *, cached_only=False):
            probed_devices.append(device)
            return 12.0 if device == "cuda:1" else 4.0

        with (
            patch.dict(os.environ, {}, clear=False),
            patch.object(environment, "_is_macos", return_value=False),
            patch.object(environment, "resolve_auto_torch_device", return_value="cuda:1"),
            patch.object(environment, "cuda_free_gib", side_effect=_fake_cuda_free),
        ):
            os.environ.pop("SLICERDSPACE_MAX_VRAM_GIB", None)
            os.environ.pop("SLICERDSPACE_MAX_VRAM_GB", None)
            action, note = environment.surface_gpu_memory_decision("auto", Path("/fake/python"))
            self.assertEqual((action, note), ("ok", ""))
            self.assertEqual(probed_devices, ["cuda:1"])

        # Inverse: if cuda:1 has only 4.0 GiB while cuda:0 has 12.0 GiB, Auto resolving to cuda:1 must fall back to CPU!
        probed_devices.clear()

        def _fake_cuda_free_inv(device="cuda:0", python_executable=None, *, cached_only=False):
            probed_devices.append(device)
            return 4.0 if device == "cuda:1" else 12.0

        with (
            patch.dict(os.environ, {}, clear=False),
            patch.object(environment, "_is_macos", return_value=False),
            patch.object(environment, "resolve_auto_torch_device", return_value="cuda:1"),
            patch.object(environment, "cuda_free_gib", side_effect=_fake_cuda_free_inv),
        ):
            os.environ.pop("SLICERDSPACE_MAX_VRAM_GIB", None)
            os.environ.pop("SLICERDSPACE_MAX_VRAM_GB", None)
            action_low, _ = environment.surface_gpu_memory_decision("auto", Path("/fake/python"))
            self.assertEqual(action_low, "cpu")
            self.assertEqual(probed_devices, ["cuda:1"])

        # End-to-end predict_subject(device="auto"): cuda:0 has 4 GiB, cuda:1 has 12 GiB -> must launch --device cuda:1!
        captured_cmd: list[str] = []
        probed_devices.clear()
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
                patch.dict(os.environ, {}, clear=False),
                patch("shared.environment._is_macos", return_value=False),
                patch("shared.environment.resolve_auto_torch_device", return_value="cuda:1"),
                patch("shared.environment.cuda_free_gib", side_effect=_fake_cuda_free),
                patch("anatomy.surface.prediction.run_backend", side_effect=lambda cmd, **kw: captured_cmd.extend(map(str, cmd))),
            ):
                os.environ.pop("SLICERDSPACE_MAX_VRAM_GIB", None)
                os.environ.pop("SLICERDSPACE_MAX_VRAM_GB", None)
                predict_subject(cfg)
        self.assertEqual(captured_cmd[captured_cmd.index("--device") + 1], "cuda:1")
        self.assertEqual(probed_devices, ["cuda:1"])

    @unittest.skipUnless(HAS_TORCH, "torch unavailable")
    def test_4c_hardware_adapt_vram_queries_and_batch_allocators_obey_global_cap_and_gb_alias(self) -> None:
        import torch
        from shared import hardware_adapt as ha

        gib = ha.GIB
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(torch.cuda, "mem_get_info", return_value=(20 * gib, 24 * gib)),
            patch.object(torch.cuda, "memory_allocated", return_value=0),
        ):
            # SLICERDSPACE_MAX_VRAM_GIB clamps both free and total
            with patch.dict(os.environ, {"SLICERDSPACE_MAX_VRAM_GIB": "3.5"}, clear=False):
                os.environ.pop("SLICERDSPACE_MAX_VRAM_GB", None)
                free_b, total_b = ha.query_cuda_free_total_bytes("cuda:0")
                self.assertEqual((free_b, total_b), (int(3.5 * gib), int(3.5 * gib)))
                self.assertEqual(ha.query_gpu_free_total_bytes("cuda:0"), (int(3.5 * gib), int(3.5 * gib)))

            # SLICERDSPACE_MAX_VRAM_GB alias also clamps both free and total
            with patch.dict(os.environ, {"SLICERDSPACE_MAX_VRAM_GB": "2.5"}, clear=False):
                os.environ.pop("SLICERDSPACE_MAX_VRAM_GIB", None)
                free_b, total_b = ha.query_cuda_free_total_bytes("cuda:0")
                self.assertEqual((free_b, total_b), (int(2.5 * gib), int(2.5 * gib)))
                self.assertEqual(ha.query_gpu_free_total_bytes("cuda:0"), (int(2.5 * gib), int(2.5 * gib)))

            # Very small cap (< 64 MiB) never overshoots cap and preserves free <= total
            with patch.dict(os.environ, {"SLICERDSPACE_MAX_VRAM_GIB": "0.03"}, clear=False):
                os.environ.pop("SLICERDSPACE_MAX_VRAM_GB", None)
                free_b, total_b = ha.query_cuda_free_total_bytes("cuda:0")
                cap_b = int(0.03 * gib)
                self.assertLessEqual(free_b, total_b)
                self.assertLessEqual(total_b, cap_b)

            # Inference batch allocators shrink automatically under global cap
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("SLICERDSPACE_MAX_VRAM_GIB", None)
                os.environ.pop("SLICERDSPACE_MAX_VRAM_GB", None)
                uncapped = ha.adaptive_gpu_scaled_batch(
                    per_item_bytes=16 * ha.MIB, default_batch=256, min_batch=4, max_batch=256, device="cuda:0"
                )
            with patch.dict(os.environ, {"SLICERDSPACE_MAX_VRAM_GB": "1.0"}, clear=False):
                os.environ.pop("SLICERDSPACE_MAX_VRAM_GIB", None)
                capped = ha.adaptive_gpu_scaled_batch(
                    per_item_bytes=16 * ha.MIB, default_batch=256, min_batch=4, max_batch=256, device="cuda:0"
                )
            self.assertEqual(uncapped, 256)
            self.assertLess(capped, 32)

        # With active memory_allocated > 0, remaining free stays <= total <= cap
        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(torch.cuda, "mem_get_info", return_value=(20 * gib, 24 * gib)),
            patch.object(torch.cuda, "memory_allocated", return_value=1 * gib),
            patch.dict(os.environ, {"SLICERDSPACE_MAX_VRAM_GIB": "3.5"}, clear=False),
        ):
            free_b, total_b = ha.query_cuda_free_total_bytes("cuda:0")
            self.assertEqual(total_b, int(3.5 * gib))
            self.assertEqual(free_b, int(2.5 * gib))
            self.assertLessEqual(free_b, total_b)


class UkfGpuSlotAndWddmTelemetryTests(unittest.TestCase):
    """Finding 5 (P2): External CUDA UKF reserves 1 GPU slot + WDDM compute-apps failure preserves whole-GPU stats."""

    def test_5a_ukf_uses_gpu_predicate_and_cohort_gpu_slot_reservation(self) -> None:
        from shared import cohort_schedule
        from tractography import ukf

        gpu_schema = ET.fromstring(
            "<executable><parameters>"
            "<boolean><name>useGPU</name><longflag>useGPU</longflag></boolean>"
            "<integer><name>gpuDevice</name><longflag>gpuDevice</longflag></integer>"
            "<integer><name>gpuPrecision</name><longflag>gpuPrecision</longflag></integer>"
            "</parameters></executable>"
        )
        cpu_schema = ET.fromstring("<executable><parameters></parameters></executable>")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cuda_exe = root / "ukf-gpu"
            cuda_exe.write_bytes(b"#!/bin/sh\n")
            cpu_exe = root / "UKFTractography.exe"
            cpu_exe.write_bytes(
                b"MZ\x90\x00" + b"\x00" * 32 + b"[GPU] this build has no CUDA support (rebuild with -DUKF_USE_CUDA=ON)\x00"
            )

            # 1. External CUDA UKF binary resolving use_gpu=True -> True
            with (
                patch.object(ukf, "resolve_executable", return_value=cuda_exe),
                patch.object(ukf, "inspect_schema", return_value=(gpu_schema, "<executable/>", 0)),
                patch.object(ukf, "_cuda_runtime_available_for_ukf", return_value=True),
                patch("shared.pipeline.build_cli_environment", return_value={}),
            ):
                self.assertTrue(ukf._ukf_uses_gpu(device="auto"))
                self.assertTrue(ukf._ukf_uses_gpu(device="cuda:0"))
                self.assertTrue(ukf._ukf_uses_gpu(device="auto", options={"useGPU": True}))
                self.assertFalse(ukf._ukf_uses_gpu(device="cpu"))
                self.assertFalse(ukf._ukf_uses_gpu(device="mps"))
                self.assertFalse(ukf._ukf_uses_gpu(device="auto", options={"useGPU": False}))

            # 2. Bundled CPU-only UKF binary -> False (even under device="auto" or "cuda")
            with (
                patch.object(ukf, "resolve_executable", return_value=cpu_exe),
                patch.object(ukf, "_cuda_runtime_available_for_ukf", return_value=True),
            ):
                self.assertFalse(ukf._ukf_uses_gpu(device="auto"))
                self.assertFalse(ukf._ukf_uses_gpu(device="cuda"))

            # 3. Legacy schema without --useGPU -> False
            with (
                patch.object(ukf, "resolve_executable", return_value=cuda_exe),
                patch.object(ukf, "inspect_schema", return_value=(cpu_schema, "<executable/>", 0)),
                patch.object(ukf, "_cuda_runtime_available_for_ukf", return_value=True),
                patch("shared.pipeline.build_cli_environment", return_value={}),
            ):
                self.assertFalse(ukf._ukf_uses_gpu(device="auto"))

            # 4. End-to-end cohort schedule admission: CUDA UKF reserves gpu=1; CPU UKF reserves gpu=0
            sched_gpu = cohort_schedule.create_schedule(
                root / "sched_gpu", cpu_budget=4, memory_gb=8, stage_cpus=2, stage_memory_gb=2, gpu_slots=1
            )
            sched_cpu = cohort_schedule.create_schedule(
                root / "sched_cpu", cpu_budget=4, memory_gb=8, stage_cpus=2, stage_memory_gb=2, gpu_slots=1
            )
            raw = SimpleNamespace(subject_id="sub01")
            with (
                patch.dict(os.environ, {cohort_schedule.ENV: str(sched_gpu)}, clear=False),
                patch.object(ukf, "resolve_executable", return_value=cuda_exe),
                patch.object(ukf, "inspect_schema", return_value=(gpu_schema, "<executable/>", 0)),
                patch.object(ukf, "_cuda_runtime_available_for_ukf", return_value=True),
                patch("shared.pipeline.build_cli_environment", return_value={}),
                patch.object(ukf, "signature", return_value={"v": 1}),
                patch.object(ukf, "current", return_value=True),
            ):
                res = ukf.run_ukf(raw, root / "out_gpu", device="cuda:0")
                self.assertEqual(res["status"], "cached")
            self.assertEqual(cohort_schedule.schedule_summary(sched_gpu)["peak_declared_reservations"]["gpu"], 1)

            with (
                patch.dict(os.environ, {cohort_schedule.ENV: str(sched_cpu)}, clear=False),
                patch.object(ukf, "resolve_executable", return_value=cpu_exe),
                patch.object(ukf, "signature", return_value={"v": 1}),
                patch.object(ukf, "current", return_value=True),
            ):
                res = ukf.run_ukf(raw, root / "out_cpu", device="cpu")
                self.assertEqual(res["status"], "cached")
            self.assertEqual(cohort_schedule.schedule_summary(sched_cpu)["peak_declared_reservations"]["gpu"], 0)

    def test_5b_wddm_compute_apps_failure_preserves_whole_gpu_utilization_and_vram(self) -> None:
        from shared.cohort_telemetry import gpu_sample

        gpu_ok = subprocess.CompletedProcess([], 0, "GPU-win-0, 67, 5120\n")

        # 1. --query-compute-apps returns non-zero (CalledProcessError)
        with patch(
            "shared.cohort_telemetry.subprocess.run",
            side_effect=[gpu_ok, subprocess.CalledProcessError(6, ["nvidia-smi"], stderr="Not Supported")],
        ):
            sample = gpu_sample({4321}, executable="nvidia-smi")
        self.assertTrue(sample["available"])
        self.assertIsNone(sample["owned_vram_mib"])
        self.assertIn("whole-GPU telemetry preserved", sample["owned_vram_note"])
        self.assertEqual(len(sample["devices"]), 1)
        self.assertEqual(sample["devices"][0]["uuid"], "GPU-win-0")
        self.assertEqual(sample["devices"][0]["utilization_percent"], 67.0)
        self.assertEqual(sample["devices"][0]["used_mib"], 5120.0)

        # 2. --query-compute-apps times out
        with patch(
            "shared.cohort_telemetry.subprocess.run",
            side_effect=[gpu_ok, subprocess.TimeoutExpired(["nvidia-smi"], 2)],
        ):
            sample_to = gpu_sample({4321}, executable="nvidia-smi")
        self.assertTrue(sample_to["available"])
        self.assertIsNone(sample_to["owned_vram_mib"])
        self.assertEqual(sample_to["devices"][0]["utilization_percent"], 67.0)
        self.assertEqual(sample_to["devices"][0]["used_mib"], 5120.0)

        # 3. --query-compute-apps exits 0 with WDDM "[Not Supported]" non-PID text
        with patch(
            "shared.cohort_telemetry.subprocess.run",
            side_effect=[gpu_ok, subprocess.CompletedProcess([], 0, "[Not Supported]\n")],
        ):
            sample_ns = gpu_sample({4321}, executable="nvidia-smi")
        self.assertTrue(sample_ns["available"])
        self.assertIsNone(sample_ns["owned_vram_mib"])
        self.assertEqual(sample_ns["devices"][0]["utilization_percent"], 67.0)
        self.assertEqual(sample_ns["devices"][0]["used_mib"], 5120.0)

        # 4. --query-compute-apps returns matching PID with [N/A]
        with patch(
            "shared.cohort_telemetry.subprocess.run",
            side_effect=[gpu_ok, subprocess.CompletedProcess([], 0, "4321, [N/A], GPU-win-0\n")],
        ):
            sample_na = gpu_sample({4321}, executable="nvidia-smi")
        self.assertTrue(sample_na["available"])
        self.assertIsNone(sample_na["owned_vram_mib"])
        self.assertEqual(sample_na["devices"][0]["utilization_percent"], 67.0)
        self.assertEqual(sample_na["devices"][0]["used_mib"], 5120.0)

        # 5. If --query-gpu itself fails, available is still False
        with patch(
            "shared.cohort_telemetry.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, ["nvidia-smi"]),
        ):
            sample_fail = gpu_sample({4321}, executable="nvidia-smi")
        self.assertFalse(sample_fail["available"])
        self.assertEqual(sample_fail["devices"], [])
        self.assertIsNone(sample_fail["owned_vram_mib"])


class FreeWaterDeviceAttrTests(unittest.TestCase):
    """Regression: MicrostructureConfig exposes free_water_device, not device."""

    def test_free_water_uses_free_water_device_attribute(self) -> None:
        from microstructure import pipeline as mp

        cfg = mp.MicrostructureConfig(
            subject_id="sub-01",
            raw_input_bundle=SimpleNamespace(),
            native_bundle=SimpleNamespace(),
            output_dir=Path("/tmp/sub-01"),
            free_water_device="cuda",
        )
        self.assertFalse(hasattr(cfg, "device"))
        self.assertEqual(cfg.free_water_device, "cuda")
        src = Path(mp.__file__).read_text(encoding="utf-8")
        self.assertNotIn("config.device", src)
        self.assertIn('getattr(config, "free_water_device", "auto")', src)


if __name__ == "__main__":
    unittest.main()
