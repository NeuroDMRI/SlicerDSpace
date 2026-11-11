"""Unit tests for hardware-adaptive scaling, OOM fallback, Windows cache locks, and CPU/CUDA kernels."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from shared.hardware_adapt import (
    GIB,
    MIB,
    adaptive_cpu_slab_z,
    adaptive_cuda_batch_size,
    adaptive_cuda_scaled_batch,
    is_cuda_oom,
    optimal_cpu_torch_threads,
    run_cuda_chunked_with_oom_fallback,
)
from scripts.run_batch_pipeline import build_parser as build_batch_parser

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_DIPY = importlib.util.find_spec("dipy") is not None
HAS_SCIPY = importlib.util.find_spec("scipy") is not None


class HardwareAdaptUnitTests(unittest.TestCase):
    def test_cuda_oom_detection_and_chunk_halving_to_cpu_fallback(self):
        class FakeCudaOutOfMemoryError(RuntimeError):
            pass

        self.assertTrue(is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")))
        self.assertTrue(is_cuda_oom(FakeCudaOutOfMemoryError("alloc failed")))
        self.assertFalse(is_cuda_oom(ValueError("invalid shape")))

        out = np.zeros(100, dtype=np.int32)
        calls: list[tuple[str, int, int]] = []

        def cuda_step(s: int, e: int) -> None:
            calls.append(("cuda", s, e))
            if (e - s) > 16:
                raise RuntimeError("CUDA out of memory")
            if s >= 64:
                # Force floor OOM on final segment to exercise seamless CPU fallback
                raise FakeCudaOutOfMemoryError("cuBLAS out of memory")
            out[s:e] = np.arange(s, e, dtype=np.int32) + 1

        def cpu_fallback(s: int, e: int) -> None:
            calls.append(("cpu", s, e))
            out[s:e] = np.arange(s, e, dtype=np.int32) + 1

        stats = run_cuda_chunked_with_oom_fallback(
            100,
            64,
            cuda_step,
            min_chunk=16,
            cpu_fallback_fn=cpu_fallback,
        )
        np.testing.assert_array_equal(out, np.arange(1, 101, dtype=np.int32))
        self.assertGreaterEqual(stats["oom_retries"], 3)
        self.assertEqual(stats["cpu_fallback_items"], 36)
        self.assertEqual(stats["final_chunk"], 16)

    def test_modest_vram_and_ram_scaling_clamps_safely(self):
        # Simulate a modest 4 GiB laptop GPU with 1.2 GiB free VRAM
        with patch("shared.hardware_adapt.query_cuda_free_total_bytes", return_value=(int(1.2 * GIB), 4 * GIB)):
            batch_low = adaptive_cuda_scaled_batch(262144, 16384, min_batch=128, max_batch=32768)
            self.assertGreaterEqual(batch_low, 128)
            self.assertLess(batch_low, 2048)

        # Simulate a high-end 80 GiB server GPU
        with patch("shared.hardware_adapt.query_cuda_free_total_bytes", return_value=(int(76 * GIB), 80 * GIB)):
            batch_high = adaptive_cuda_scaled_batch(262144, 16384, min_batch=128, max_batch=32768)
            self.assertEqual(batch_high, 16384)
            batch_auto = adaptive_cuda_batch_size(262144, 16384, min_batch=128, max_batch=32768)
            self.assertEqual(batch_auto, 16384)

        # Simulate an 8 GiB RAM laptop vs 128 GiB workstation for z-slab streaming
        with patch("shared.hardware_adapt.query_host_available_ram_bytes", return_value=4 * GIB):
            slab_low = adaptive_cpu_slab_z((145, 174, 145, 288), default_step_z=16, min_step_z=2, bytes_per_element=32)
            self.assertGreaterEqual(slab_low, 2)
            self.assertLessEqual(slab_low, 4)

        with patch("shared.hardware_adapt.query_host_available_ram_bytes", return_value=128 * GIB):
            slab_high = adaptive_cpu_slab_z((145, 174, 145, 288), default_step_z=16, min_step_z=2, bytes_per_element=32)
            self.assertEqual(slab_high, 16)

    def test_optimal_cpu_torch_threads_and_batch_low_memory_flag(self):
        with patch.dict(os.environ, {"SLICERDSPACE_STAGE_CPUS": "4"}, clear=False):
            self.assertEqual(optimal_cpu_torch_threads(None, max_conv_threads=8), 4)
        self.assertEqual(optimal_cpu_torch_threads(32, max_conv_threads=8), 8)

        args = build_batch_parser().parse_args(["--batch-root", ".", "--scheduler", "adaptive", "--low-memory"])
        self.assertTrue(args.low_memory)
        self.assertEqual(args.scheduler, "adaptive")


@unittest.skipUnless(HAS_TORCH and HAS_DIPY and HAS_SCIPY, "Requires PyTorch, DIPY and SciPy (PythonSlicer)")
class AcceleratedNumericsParityTests(unittest.TestCase):
    def test_mppca_and_gibbs_match_dipy_reference(self):
        from dipy.denoise.gibbs import gibbs_removal
        from dipy.denoise.localpca import mppca
        from preprocessing.dipy_accel import gibbs_removal_accelerated, mppca_accelerated

        rng = np.random.default_rng(20261111)
        vol = (100.0 + 15.0 * rng.standard_normal((10, 10, 8, 12))).astype(np.float32)

        ref_den = mppca(vol.copy(), patch_radius=1, return_sigma=False)
        opt_den = mppca_accelerated(vol.copy(), patch_radius=1, device="cpu", max_workers=2)
        self.assertEqual(opt_den.shape, ref_den.shape)
        np.testing.assert_allclose(opt_den, ref_den, rtol=1e-4, atol=2e-3)

        opt_den2, opt_sig = mppca_accelerated(vol.copy(), patch_radius=1, return_sigma=True, device="cpu", max_workers=2)
        ref_den2, ref_sig = mppca(vol.copy(), patch_radius=1, return_sigma=True)
        np.testing.assert_allclose(opt_den2, ref_den2, rtol=1e-4, atol=2e-3)
        np.testing.assert_allclose(opt_sig, ref_sig, rtol=1e-4, atol=2e-3)

        ref_gibbs = gibbs_removal(vol[..., :4].copy(), slice_axis=2, n_points=3, inplace=False)
        opt_gibbs = gibbs_removal_accelerated(vol[..., :4].copy(), slice_axis=2, n_points=3, device="cpu", max_workers=2)
        np.testing.assert_allclose(opt_gibbs, ref_gibbs, rtol=1e-5, atol=1e-4)

        # Dual covariance path (dim > num_samples, e.g. 5x5x5=125 < 140) + constant/zero-patch short-circuit
        vol_dual_64 = (80.0 + 12.0 * rng.standard_normal((8, 8, 7, 140))).astype(np.float64)
        ref_dual_64, ref_dual_sig_64 = mppca(vol_dual_64.copy(), patch_radius=2, return_sigma=True, suppress_warning=True)
        opt_dual_64, opt_dual_sig_64 = mppca_accelerated(vol_dual_64.copy(), patch_radius=2, return_sigma=True, device="cpu", max_workers=2)
        np.testing.assert_allclose(opt_dual_64, ref_dual_64, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(opt_dual_sig_64, ref_dual_sig_64, rtol=1e-11, atol=1e-11)

        vol_dual_air = np.zeros((9, 9, 11, 140), dtype=np.float32)
        vol_dual_air[2:8, 2:8, 5:10, :] = (80.0 + 12.0 * rng.standard_normal((6, 6, 5, 140))).astype(np.float32)
        ref_air = mppca(vol_dual_air.copy(), patch_radius=2, return_sigma=False, suppress_warning=True)
        opt_air, opt_air_sig = mppca_accelerated(vol_dual_air.copy(), patch_radius=2, return_sigma=True, device="cpu", max_workers=2)
        np.testing.assert_allclose(opt_air, ref_air, rtol=1e-4, atol=2e-3)
        self.assertTrue(np.all(np.isfinite(opt_air_sig)))
        np.testing.assert_array_equal(opt_air[:, :, :1, :], 0.0)

        import torch

        if torch.cuda.is_available():
            cuda_dual_64, cuda_dual_sig_64 = mppca_accelerated(vol_dual_64.copy(), patch_radius=2, return_sigma=True, device="cuda")
            np.testing.assert_allclose(cuda_dual_64, ref_dual_64, rtol=1e-10, atol=1e-10)
            np.testing.assert_allclose(cuda_dual_sig_64, ref_dual_sig_64, rtol=1e-10, atol=1e-10)
            cuda_air = mppca_accelerated(vol_dual_air.copy(), patch_radius=2, return_sigma=False, device="cuda")
            np.testing.assert_allclose(cuda_air, opt_air, rtol=1e-5, atol=1e-4)
            cuda_gibbs = gibbs_removal_accelerated(vol[..., :4].copy(), slice_axis=2, n_points=3, device="cuda")
            np.testing.assert_allclose(cuda_gibbs, ref_gibbs, rtol=5e-4, atol=5e-2)
            self.assertLess(float(np.mean(np.abs(cuda_gibbs - ref_gibbs)) / np.max(np.abs(ref_gibbs))), 1e-5)
            vol64_g = vol[..., :2].astype(np.float64)
            np.testing.assert_allclose(
                gibbs_removal_accelerated(vol64_g.copy(), slice_axis=2, n_points=3, device="cuda"),
                gibbs_removal(vol64_g.copy(), slice_axis=2, n_points=3, inplace=False),
                rtol=1e-11,
                atol=1e-11,
            )

    def test_batched_dki_wls_matches_per_voxel_dipy(self):
        from dipy.core.gradients import gradient_table
        from dipy.reconst.dki import DiffusionKurtosisModel
        from microstructure.dki import _fit_dki_chunk

        rng = np.random.default_rng(42)
        dirs = rng.standard_normal((60, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        bvals = np.array([0.0] * 6 + [1000.0] * 30 + [2000.0] * 30)
        bvecs = np.vstack([np.zeros((6, 3)), dirs[:30], dirs[30:]])
        gtab = gradient_table(bvals=bvals, bvecs=bvecs)
        model = DiffusionKurtosisModel(gtab, fit_method="WLS")

        # Simulate 80 realistic multi-shell voxels (> 64 triggers vectorized Cholesky path)
        d_true = 0.0008 + 0.0003 * rng.random((80, 1))
        k_true = 0.7 + 0.3 * rng.random((80, 1))
        b_col = bvals[None, :]
        signals = 500.0 * np.exp(-b_col * d_true + (1.0 / 6.0) * (b_col ** 2) * (d_true ** 2) * k_true)
        signals *= 1.0 + 0.01 * rng.standard_normal(signals.shape)
        signals = np.clip(signals, 1.0, None)

        ref_p, ref_s0 = _fit_dki_chunk(model, signals[:40], "WLS")
        ref_p2, ref_s02 = _fit_dki_chunk(model, signals[40:], "WLS")
        ref_params = np.vstack([ref_p, ref_p2])
        ref_s0_all = np.concatenate([ref_s0, ref_s02])

        opt_params, opt_s0 = _fit_dki_chunk(model, signals, "WLS")
        np.testing.assert_allclose(opt_params, ref_params, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(opt_s0, ref_s0_all, rtol=1e-6, atol=1e-7)

    def test_dti_tracking_multislab_ram_cap_matches_single_pass(self):
        import nibabel as nib
        from shared.pipeline import RawInputBundle
        from tractography.dti import DTITrackingConfig, run_dti_tracking

        rng = np.random.default_rng(7)
        dirs = rng.standard_normal((16, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        bvals = np.array([0.0, 0.0] + [1000.0] * 16)
        bvecs = np.vstack([np.zeros((2, 3)), dirs])
        # z=22 > default_step_z=16 so low-RAM cap splits into multiple z-slabs
        shape = (8, 8, 22)
        mask = np.zeros(shape, dtype=np.uint8)
        mask[1:7, 1:7, 1:21] = 1
        d_tensor = np.array([[1.6e-3, 0.0, 0.0], [0.0, 0.35e-3, 0.0], [0.0, 0.0, 0.35e-3]])
        atten = np.exp(-np.einsum("gi,ij,gj->g", bvecs, d_tensor, bvecs) * bvals)
        dwi = np.zeros(shape + (18,), dtype=np.float32)
        for z in range(1, 21):
            # Vary scale by z so any z-slab order mismatch would scramble voxels
            dwi[1:7, 1:7, z, :] = float(500.0 + 25.0 * z) * atten[None, None, :]

        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            affine = np.diag([2.0, 2.0, 2.0, 1.0])
            dwi_p = tdir / "data.nii.gz"
            mask_p = tdir / "nodif_brain_mask.nii.gz"
            bval_p = tdir / "bvals"
            bvec_p = tdir / "bvecs"
            nib.save(nib.Nifti1Image(dwi, affine), str(dwi_p))
            nib.save(nib.Nifti1Image(mask, affine), str(mask_p))
            np.savetxt(bval_p, bvals[None, :], fmt="%.1f")
            np.savetxt(bvec_p, bvecs.T, fmt="%.6f")
            raw = RawInputBundle(tdir, "sub01", tdir, dwi_p, bval_p, bvec_p, mask_p)

            res_ram = run_dti_tracking(
                DTITrackingConfig(
                    subject_id="sub01", input_root=tdir, input_bundle=raw, output_dir=tdir / "out_ram",
                    min_length_mm=4.0, max_length_mm=80.0, device="cpu", force_recompute=True, execute=True,
                )
            )
            with patch.dict(os.environ, {"SLICERDSPACE_MAX_RAM_GIB": "0.0001"}, clear=False):
                res_slab = run_dti_tracking(
                    DTITrackingConfig(
                        subject_id="sub01", input_root=tdir, input_bundle=raw, output_dir=tdir / "out_slab",
                        min_length_mm=4.0, max_length_mm=80.0, device="cpu", force_recompute=True, execute=True,
                    )
                )
            sl_ram = nib.streamlines.load(str(res_ram.streamlines_path)).streamlines
            sl_slab = nib.streamlines.load(str(res_slab.streamlines_path)).streamlines
            self.assertEqual(len(sl_ram), len(sl_slab))
            self.assertGreater(len(sl_ram), 0)
            np.testing.assert_allclose(sl_ram._data, sl_slab._data, rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
