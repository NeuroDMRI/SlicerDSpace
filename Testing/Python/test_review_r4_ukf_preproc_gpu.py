"""Targeted regression tests for R4 UKF CPU-build/cache isolation, preprocessing GPU slot gating, and DIPY CUDA fallback classification."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

import numpy as np


class UkfCpuBuildAndCacheIsolationTests(unittest.TestCase):
    def test_cpu_only_native_ukf_binary_detected_despite_gpu_xml(self) -> None:
        from tractography import ukf

        schema = ET.fromstring(
            "<executable><parameters>"
            "<boolean><name>useGPU</name><longflag>useGPU</longflag></boolean>"
            "<integer><name>gpuDevice</name><longflag>gpuDevice</longflag></integer>"
            "<integer><name>gpuPrecision</name><longflag>gpuPrecision</longflag></integer>"
            "</parameters></executable>"
        )
        with tempfile.TemporaryDirectory() as tmp:
            exe = Path(tmp) / "UKFTractography.exe"
            exe.write_bytes(
                b"MZ\x90\x00"
                + b"\x00" * 64
                + b"[GPU] this build has no CUDA support (rebuild with -DUKF_USE_CUDA=ON), using CPU\x00"
            )
            self.assertFalse(ukf._ukf_binary_has_compiled_cuda(exe))
            with mock.patch.object(ukf, "_cuda_runtime_available_for_ukf", return_value=True):
                # Auto mode must NOT generate a GPU request for a CPU-only binary.
                self.assertEqual(ukf.resolve_ukf_gpu_dispatch("auto", {}, schema, exe), (False, 0, 0))
                # Explicit CUDA request must fail loudly instead of silently running CPU.
                with self.assertRaisesRegex(RuntimeError, "CPU-only build without CUDA support"):
                    ukf.resolve_ukf_gpu_dispatch("cuda", {}, schema, exe)
                with self.assertRaisesRegex(RuntimeError, "CPU-only build without CUDA support"):
                    ukf.resolve_ukf_gpu_dispatch("auto", {"useGPU": True}, schema, exe)

    def test_explicit_gpu_request_does_not_hit_stale_cpu_cache(self) -> None:
        from tractography import ukf

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dwi = root / "dwi.nii.gz"
            bval = root / "bvals"
            bvec = root / "bvecs"
            mask = root / "mask.nii.gz"
            exe = root / "ukf_bin"
            conv = root / "dwiconvert_bin"
            for p in (dwi, bval, bvec, mask, exe, conv):
                p.write_bytes(b"stub")

            raw = types.SimpleNamespace(
                subject_id="sub-01",
                dwi_path=dwi,
                bval_path=bval,
                bvec_path=bvec,
                mask_path=mask,
            )
            out_dir = root / "ukf_out"
            out_dir.mkdir()
            prods = ukf.paths("sub-01", out_dir)
            for key, path in prods.items():
                if key == "manifest":
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                if key == "qc":
                    path.write_text(json.dumps({"automatic_qc_passed": True}), encoding="utf-8")
                else:
                    path.write_bytes(b"artifact")

            with mock.patch.object(ukf, "_input_affine", return_value=np.eye(4)):
                cpu_sig = ukf.signature(raw, None, exe, conv, device="auto")
                manifest = {
                    "status": "completed",
                    "signature": cpu_sig,
                    "gpu": {"enabled": False, "device": None, "precision": None},
                    "artifacts": {
                        k: ukf.digest(p) for k, p in prods.items() if k != "manifest"
                    },
                }
                prods["manifest"].write_text(json.dumps(manifest), encoding="utf-8")

                # Auto/CPU still reuses the matching CPU cache
                self.assertTrue(ukf.current(raw, out_dir, None, exe, conv, device="auto"))
                # Explicit GPU request MUST NOT reuse the CPU cache
                self.assertFalse(ukf.current(raw, out_dir, None, exe, conv, device="cuda"))
                self.assertFalse(ukf.current(raw, out_dir, {"useGPU": True}, exe, conv, device="auto"))

                # Even if an old manifest had a matching signature, GPU request rejects gpu.enabled=False
                gpu_sig = ukf.signature(raw, None, exe, conv, device="cuda")
                manifest["signature"] = gpu_sig
                prods["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
                self.assertFalse(ukf.current(raw, out_dir, None, exe, conv, device="cuda"))


class PreprocessingGpuSlotTests(unittest.TestCase):
    def test_cpu_only_eddy_with_denoise_off_reserves_zero_gpu_slots(self) -> None:
        from preprocessing.pipeline import DwiPreprocessingConfig, _preprocessing_uses_gpu

        cfg_eddy_only = DwiPreprocessingConfig(
            subject_id="sub-01",
            input_root=Path("."),
            execute=True,
            run_denoise=False,
            run_eddy=True,
            run_topup=False,
            execution_backend="native",
        )
        with mock.patch("preprocessing.pipeline.inspect_preprocessing_backend", return_value={"execution_backend": "native"}):
            self.assertFalse(_preprocessing_uses_gpu(cfg_eddy_only))

        # When denoise IS enabled and DIPY resolves CUDA, 1 GPU slot is reserved.
        cfg_denoise_cuda = DwiPreprocessingConfig(
            subject_id="sub-01",
            input_root=Path("."),
            execute=True,
            run_denoise=True,
            run_eddy=True,
            execution_backend="native",
            dipy_device="auto",
        )
        with mock.patch("preprocessing.dipy_accel.resolve_dipy_accel_device", return_value="cuda:0"):
            self.assertTrue(_preprocessing_uses_gpu(cfg_denoise_cuda))

        # When denoise is enabled but DIPY runs CPU on native Eddy, 0 GPU slots are reserved.
        with (
            mock.patch("preprocessing.dipy_accel.resolve_dipy_accel_device", return_value="cpu"),
            mock.patch("preprocessing.pipeline.inspect_preprocessing_backend", return_value={"execution_backend": "native"}),
        ):
            self.assertFalse(_preprocessing_uses_gpu(cfg_denoise_cuda))

    def test_cohort_planner_drops_preprocessing_gpu_when_denoise_excluded(self) -> None:
        from shared.cohort_planning import _stage_uses_gpu_in_plan

        self.assertTrue(_stage_uses_gpu_in_plan("preprocessing", []))
        self.assertFalse(_stage_uses_gpu_in_plan("preprocessing", ["--preprocessing-stages", "eddy"]))
        self.assertFalse(_stage_uses_gpu_in_plan("preprocessing", ["--preprocessing-stages", "eddy,topup"]))
        self.assertTrue(_stage_uses_gpu_in_plan("preprocessing", ["--preprocessing-stages", "denoise,eddy"]))


class MppcaCudaFallbackClassificationTests(unittest.TestCase):
    def test_auto_mode_logs_tdr_and_oom_fallbacks_but_propagates_non_cuda_errors(self) -> None:
        from preprocessing import dipy_accel as da

        arr = np.ones((6, 6, 6, 12), dtype=np.float32)

        # 1. WDDM TDR timeout in auto mode -> logs clear TDR diagnostic and falls back to CPU
        buf_tdr = io.StringIO()
        with (
            mock.patch.object(da, "resolve_dipy_accel_device", return_value="cuda:0"),
            mock.patch.object(da, "_mppca_cuda", side_effect=RuntimeError("CUDA error: the launch timed out and was terminated")),
            mock.patch.object(da, "_mppca_cpu", return_value=arr.copy()) as cpu_mock,
            contextlib.redirect_stderr(buf_tdr),
        ):
            out = da.mppca_accelerated(arr, patch_radius=1, device="auto")
            self.assertEqual(out.shape, arr.shape)
            cpu_mock.assert_called_once()
        self.assertIn("WDDM TDR", buf_tdr.getvalue())
        self.assertIn("falling back to CPU", buf_tdr.getvalue())

        # 2. Non-CUDA ValueError in auto mode -> MUST propagate, NOT silently restart on CPU
        with (
            mock.patch.object(da, "resolve_dipy_accel_device", return_value="cuda:0"),
            mock.patch.object(da, "_mppca_cuda", side_effect=ValueError("bad internal shape")),
            mock.patch.object(da, "_mppca_cpu") as cpu_mock2,
        ):
            with self.assertRaisesRegex(ValueError, "bad internal shape"):
                da.mppca_accelerated(arr, patch_radius=1, device="auto")
            cpu_mock2.assert_not_called()

        # 3. Explicit CUDA mode on WDDM TDR -> logs actionable guidance and raises
        buf_explicit = io.StringIO()
        with (
            mock.patch.object(da, "resolve_dipy_accel_device", return_value="cuda:0"),
            mock.patch.object(da, "_mppca_cuda", side_effect=RuntimeError("cudaErrorLaunchTimeout")),
            contextlib.redirect_stderr(buf_explicit),
        ):
            with self.assertRaisesRegex(RuntimeError, "cudaErrorLaunchTimeout"):
                da.mppca_accelerated(arr, patch_radius=1, device="cuda:0")
        self.assertIn("Explicit CUDA mode does not fall back", buf_explicit.getvalue())


if __name__ == "__main__":
    unittest.main()
