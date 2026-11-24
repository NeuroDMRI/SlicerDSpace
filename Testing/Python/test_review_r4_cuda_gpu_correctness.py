"""Review R4 regression tests: Windows NVIDIA/CUDA correctness, MK-Curve OOM immutability,
CPU-only Torch detection, non-ASCII Windows probes, multi-GPU & CUDA_VISIBLE_DEVICES parity,
NODDI/DDSeg fallback device provenance, WDDM TDR handling, DDSurfer VRAM cap, WMA GPU slot,
and WSL Blackwell (RTX 50) cu128 + kernel smoke verification.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _write_mkcurve_fixture_case(root: Path, fixture: dict, *, repeat: int = 1) -> tuple[Path, Path, Path, Path, np.ndarray]:
    """Write the 4-voxel author Octave fixture (optionally tiled) as NIfTI/bval/bvec/mask."""
    gradients = np.asarray(fixture["gradients"], dtype=np.float64)
    signals = np.asarray(fixture["signals"], dtype=np.float64)
    grid = np.asarray(fixture["grid"], dtype=np.float64)
    if repeat > 1:
        signals = np.tile(signals, (repeat, 1))
    n_vox, n_vol = signals.shape
    dwi_data = signals.reshape((n_vox, 1, 1, n_vol)).astype(np.float64)
    mask_data = np.ones((n_vox, 1, 1), dtype=np.uint8)
    affine = np.eye(4, dtype=np.float64)
    dwi_path = root / "dwi.nii.gz"
    mask_path = root / "mask.nii.gz"
    bval_path = root / "dwi.bval"
    bvec_path = root / "dwi.bvec"
    nib.save(nib.Nifti1Image(dwi_data, affine), str(dwi_path))
    nib.save(nib.Nifti1Image(mask_data, affine), str(mask_path))
    np.savetxt(bval_path, gradients[:, 3][None, :], fmt="%.6f")
    np.savetxt(bvec_path, gradients[:, :3].T, fmt="%.8f")
    return dwi_path, bval_path, bvec_path, mask_path, grid


class MKCurveLateOomImmutabilityAndProvenanceTests(unittest.TestCase):
    """Item 1 (P0): Late CUDA OOM must never mutate original b0 or falsely record device='cuda'."""

    def _load_fixture(self) -> dict:
        fixture_path = REPO_ROOT / "Testing" / "Python" / "fixtures" / "mkcurve_author_octave.json"
        return json.loads(fixture_path.read_text(encoding="utf-8"))

    def _make_fake_cuda_fitter_cls(self, *, fail_on_fit_call: int | None = None):
        """CPU-backed stand-in for _CudaSyntheticB0Fitter that exercises the exact _fit_chunk_cuda body."""
        import torch
        from microstructure.mkcurve_tensor import _SyntheticB0Fitter

        fit_calls = {"count": 0}

        class FakeCudaFitter:
            def __init__(self, signals, design, grid, *, solver="reference", device="cuda"):
                self.cpu_fitter = _SyntheticB0Fitter(signals, design, solver=solver)
                self.grid_np = np.ascontiguousarray(grid, dtype=np.float64)
                self.dev = torch.device("cpu")
                # Non-None ops keeps _fit_chunk_cuda on the CUDA branch rather than early CPU return.
                self.ops = {
                    "diff_basis": torch.zeros((1, 6), dtype=torch.float64),
                    "kurt_basis": torch.zeros((1, 15), dtype=torch.float64),
                }

            def fit(self, b0):
                fit_calls["count"] += 1
                if fail_on_fit_call is not None and fit_calls["count"] == fail_on_fit_call:
                    raise RuntimeError("CUDA out of memory. Tried to allocate 1.50 GiB")
                return self.cpu_fitter.fit_fast(b0)

            def fit_grid(self):
                return self.cpu_fitter.fit_grid(self.grid_np)

        return FakeCudaFitter, fit_calls

    def test_late_cuda_oom_preserves_original_mk_and_reports_cpu_device(self) -> None:
        from microstructure import mkcurve, mkcurve_tensor

        fixture = self._load_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dwi, bval, bvec, mask, grid = _write_mkcurve_fixture_case(root, fixture)

            # 1. Clean CPU reference on the 4-voxel author fixture
            ref_record = mkcurve.run_mkcurve(dwi, bval, bvec, mask, root / "cpu_ref", grid=grid, device="cpu")
            ref_orig_mk = nib.load(ref_record["maps"]["original_mk"]).get_fdata().ravel()
            ref_corr_mk = nib.load(ref_record["maps"]["corrected_mk"]).get_fdata().ravel()
            # Sanity check: voxel 0 original MK is ~0.479 and corrected MK is ~0.685
            self.assertAlmostEqual(float(ref_orig_mk[0]), float(fixture["original_maps"]["mk"][0]), places=5)
            self.assertAlmostEqual(float(ref_corr_mk[0]), float(fixture["corrected_maps"]["mk"][0]), places=5)
            self.assertGreater(abs(float(ref_corr_mk[0]) - float(ref_orig_mk[0])), 0.15)

            # 2. Inject late CUDA OOM during _parameter_maps_cuda (after corrected_b0 fit)
            fake_fitter_cls, _ = self._make_fake_cuda_fitter_cls()
            logs: list[str] = []
            with (
                patch("torch.cuda.is_available", return_value=True),
                patch.object(mkcurve_tensor, "_CudaSyntheticB0Fitter", fake_fitter_cls),
                patch.object(
                    mkcurve_tensor,
                    "_mean_kurtosis_cuda",
                    side_effect=lambda coeff_t, *_a, **_kw: __import__("torch").as_tensor(
                        mkcurve_tensor.mean_kurtosis(coeff_t.detach().cpu().numpy()), dtype=__import__("torch").float64
                    ),
                ),
                patch.object(
                    mkcurve_tensor,
                    "_parameter_maps_cuda",
                    side_effect=RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"),
                ),
                patch("shared.hardware_adapt.empty_cuda_cache_safely"),
            ):
                oom_record = mkcurve.run_mkcurve(
                    dwi, bval, bvec, mask, root / "oom_late_maps", grid=grid, device="auto", log=logs.append
                )

            # Original MK and all maps MUST match the clean CPU reference (NO 0.479 -> 0.685 corruption!)
            for key in ("original_mk", "corrected_mk", "original_fa", "corrected_fa", "abnormal_mask"):
                ref_arr = nib.load(ref_record["maps"][key]).get_fdata()
                oom_arr = nib.load(oom_record["maps"][key]).get_fdata()
                np.testing.assert_allclose(oom_arr, ref_arr, rtol=1e-6, atol=1e-6, err_msg=f"Mismatch in {key}")

            oom_orig_mk = nib.load(oom_record["maps"]["original_mk"]).get_fdata().ravel()
            self.assertAlmostEqual(float(oom_orig_mk[0]), float(fixture["original_maps"]["mk"][0]), places=5)

            # Provenance MUST report CPU, never falsely claiming CUDA
            self.assertEqual(oom_record["device"], "cpu")
            self.assertEqual(oom_record["settings"]["device"], "cpu")
            self.assertEqual(oom_record["settings"]["resolved_device"], "cuda")
            self.assertTrue(oom_record["settings"]["device_fallback"])
            self.assertEqual(oom_record["settings"]["fallback_from"], "cuda")
            self.assertIn("out of memory", oom_record["settings"]["fallback_reason"].lower())
            self.assertEqual(oom_record["settings"]["cpu_fallback_voxels"], 4)
            progress = json.loads((root / "oom_late_maps" / "mkcurve_progress.json").read_text(encoding="utf-8"))
            self.assertEqual(progress["device"], "cpu")
            self.assertTrue(any("cpu" in msg.lower() and "fallback" in msg.lower() for msg in logs), logs)

            # 3. Also inject OOM directly on the corrected_b0 fit call (3rd fit call: original, max_mk_b0, corrected_b0)
            fake_fitter_cls2, _ = self._make_fake_cuda_fitter_cls(fail_on_fit_call=3)
            with (
                patch("torch.cuda.is_available", return_value=True),
                patch.object(mkcurve_tensor, "_CudaSyntheticB0Fitter", fake_fitter_cls2),
                patch.object(
                    mkcurve_tensor,
                    "_mean_kurtosis_cuda",
                    side_effect=lambda coeff_t, *_a, **_kw: __import__("torch").as_tensor(
                        mkcurve_tensor.mean_kurtosis(coeff_t.detach().cpu().numpy()), dtype=__import__("torch").float64
                    ),
                ),
                patch("shared.hardware_adapt.empty_cuda_cache_safely"),
            ):
                oom_rec2 = mkcurve.run_mkcurve(
                    dwi, bval, bvec, mask, root / "oom_corrected_fit", grid=grid, device="auto"
                )
            np.testing.assert_allclose(
                nib.load(oom_rec2["maps"]["original_mk"]).get_fdata(),
                nib.load(ref_record["maps"]["original_mk"]).get_fdata(),
                rtol=1e-6,
                atol=1e-6,
            )
            self.assertEqual(oom_rec2["settings"]["device"], "cpu")

    def test_late_cuda_oom_chunk_halving_preserves_original_b0_and_caller_buffer(self) -> None:
        import torch
        from microstructure import mkcurve, mkcurve_tensor

        fixture = self._load_fixture()
        gradients = np.asarray(fixture["gradients"], dtype=np.float64)
        base_signals = np.asarray(fixture["signals"], dtype=np.float64)
        grid = np.asarray(fixture["grid"], dtype=np.float64)
        design = mkcurve_tensor.design_matrix(gradients[:, 3], gradients[:, :3])

        # 136 voxels (> 128) forces recursive chunk halving into 68 + 68 on late OOM
        tiled_signals = np.tile(base_signals, (34, 1))
        snapshot = tiled_signals.copy()

        ref_orig, ref_corr, _, ref_det = mkcurve._fit_chunk((tiled_signals.copy(), design, grid, 0.5, "reference"))

        fake_fitter_cls, _ = self._make_fake_cuda_fitter_cls()
        map_calls = {"count": 0}

        def flaky_parameter_maps_cuda(coefficients, _ops, *, device_str="cuda"):
            map_calls["count"] += 1
            if len(coefficients) > 128:
                raise RuntimeError("CUDA out of memory during large chunk parameter maps")
            return mkcurve_tensor.parameter_maps(coefficients)

        tracker = {"cuda_voxels": 0, "cpu_fallback_voxels": 0, "oom_retries": 0, "fallback_reasons": []}
        with (
            patch.object(mkcurve_tensor, "_CudaSyntheticB0Fitter", fake_fitter_cls),
            patch.object(
                mkcurve_tensor,
                "_mean_kurtosis_cuda",
                side_effect=lambda coeff_t, *_a, **_kw: torch.as_tensor(
                    mkcurve_tensor.mean_kurtosis(coeff_t.detach().cpu().numpy()), dtype=torch.float64
                ),
            ),
            patch.object(mkcurve_tensor, "_parameter_maps_cuda", side_effect=flaky_parameter_maps_cuda),
            patch("shared.hardware_adapt.empty_cuda_cache_safely"),
        ):
            orig, corr, _, det = mkcurve._fit_chunk_cuda(
                (tiled_signals, design, grid, 0.5, "reference", "cuda", False, tracker)
            )

        # Caller's input buffer must NEVER have been mutated in-place
        np.testing.assert_array_equal(tiled_signals, snapshot)
        # Both halved chunks must preserve original_mk (~0.479) and corrected_mk (~0.685)
        np.testing.assert_allclose(det["original_mk"], ref_det["original_mk"], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(det["corrected_mk"], ref_det["corrected_mk"], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(orig, ref_orig, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(corr, ref_corr, rtol=1e-6, atol=1e-6)
        self.assertEqual(tracker["oom_retries"], 1)
        self.assertEqual(tracker["cuda_voxels"], 136)
        self.assertEqual(tracker["cpu_fallback_voxels"], 0)

    def test_fit_chunk_cpu_short_grid_does_not_mutate_input_signals(self) -> None:
        from microstructure import mkcurve, mkcurve_tensor

        fixture = self._load_fixture()
        gradients = np.asarray(fixture["gradients"], dtype=np.float64)
        signals = np.asarray(fixture["signals"], dtype=np.float64).copy()
        snapshot = signals.copy()
        short_grid = np.linspace(100.0, 2000.0, 12)
        design = mkcurve_tensor.design_matrix(gradients[:, 3], gradients[:, :3])

        mkcurve._fit_chunk((signals, design, short_grid, 0.5, "reference"))
        np.testing.assert_array_equal(signals, snapshot)


class SetupCpuTorchAndNonAsciiCudaProbeTests(unittest.TestCase):
    """Item 2 (P1): CPU-only Torch must not satisfy CUDA setup; non-ASCII Windows paths must probe cleanly."""

    def test_setup_does_not_mark_cuda_satisfied_with_cpu_only_torch(self) -> None:
        from shared.environment import DependencySpec, _check_spec
        from shared.selected_setup import build_setup_plan

        # 1. Doctor _check_spec reports CPU-only PyTorch clearly when CUDA is requested
        cpu_probe_details = {
            "torch_importable": True,
            "cuda_available": False,
            "device_count": 0,
            "mps_available": False,
            "torch_version": "2.15.0+cpu",
            "torch_cuda_build": None,
            "error": None,
            "requested_device": "cuda",
            "requested_index": None,
            "python_executable": sys.executable,
            "pythonpath": [],
        }
        spec = DependencySpec(
            key="device:segmentation:cuda",
            kind="device",
            label="segmentation requested device cuda",
            required=True,
            targets=("segmentation",),
            device="cuda",
            install_group="device",
        )
        with patch("shared.environment._check_cuda_device", return_value=(False, dict(cpu_probe_details))):
            status = _check_spec(spec, "Windows")

        self.assertEqual(status.status, "missing")
        self.assertTrue(status.details.get("cpu_only_torch"))
        self.assertEqual(status.details.get("detected_source"), "CPU-only PyTorch")
        self.assertIn("CPU-only", status.install_hint)
        self.assertIn("torch.version.cuda is None", status.install_hint)

        # 2. build_setup_plan must NOT treat python:torch (+cpu) as ready when CUDA is requested
        torch_ok_cpu = SimpleNamespace(
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
            },
        )
        report_cuda = SimpleNamespace(
            statuses=(torch_ok_cpu, status),
            targets=("segmentation",),
            blocker_count=1,
        )
        plan = build_setup_plan(
            report_cuda,
            installed_extensions=(),
            loaded_extensions=[],
            need_torch=True,
            system="Windows",
            machine="AMD64",
        )
        self.assertIn("PyTorch", plan["extensions"])
        self.assertFalse(plan["cuda_torch_ready"])
        self.assertTrue(plan["cuda_torch_cpu_only"])
        self.assertTrue(
            any("cpu-only" in entry["hint"].lower() and "cuda" in entry["hint"].lower() for entry in plan["manual_items"]),
            plan["manual_items"],
        )

        # Even if PyTorch extension is already installed, CUDA setup must still NOT be marked ready
        plan_ext_installed = build_setup_plan(
            report_cuda,
            installed_extensions=("PyTorch",),
            loaded_extensions=["PyTorch"],
            need_torch=True,
            system="Windows",
            machine="AMD64",
        )
        self.assertFalse(plan_ext_installed["cuda_torch_ready"])
        self.assertTrue(plan_ext_installed["cuda_torch_cpu_only"])
        self.assertTrue(plan_ext_installed["manual_items"])

        # 3. Guard: when CUDA is NOT requested (auto/cpu), bundled +cpu torch stays ready
        auto_device = SimpleNamespace(
            key="device:segmentation:auto",
            kind="device",
            label="segmentation requested device auto",
            status="available",
            required=True,
            targets=("segmentation",),
            install_group="device",
            install_hint="",
            details={"requested_device": "auto", "resolved_device": "cpu"},
        )
        report_auto = SimpleNamespace(
            statuses=(torch_ok_cpu, auto_device),
            targets=("segmentation",),
            blocker_count=0,
        )
        plan_auto = build_setup_plan(
            report_auto,
            installed_extensions=(),
            loaded_extensions=[],
            need_torch=True,
            system="Windows",
            machine="AMD64",
        )
        self.assertEqual(plan_auto["extensions"], [])
        self.assertTrue(plan_auto["cuda_torch_ready"])
        self.assertFalse(plan_auto["cuda_torch_cpu_only"])

    def test_real_pythonslicer_cuda_probe_under_chinese_non_ascii_path(self) -> None:
        from shared.environment import _check_cuda_device
        from shared.external_process import find_pythonslicer

        # Prefer the real PythonSlicer.exe next to sys.executable on Windows, falling back to sys.executable
        py_exe = find_pythonslicer(Path(sys.executable).parent.parent) or Path(sys.executable)

        with tempfile.TemporaryDirectory() as tmp:
            chinese_sp = Path(tmp) / "中文目录_CUDA测试" / "site-packages"
            fake_torch_dir = chinese_sp / "torch"
            fake_torch_dir.mkdir(parents=True)
            (fake_torch_dir / "__init__.py").write_text(
                "__version__ = '2.11.0+cu128'\n"
                "class version:\n"
                "    cuda = '12.8'\n"
                "class _FakeTensor:\n"
                "    def __init__(self, val): self._v = val\n"
                "    def __add__(self, other): return _FakeTensor(self._v + other)\n"
                "    def item(self): return float(self._v)\n"
                "def zeros(n, device=None):\n"
                "    assert device in ('cuda:0', 'cuda:1'), f'unexpected {device}'\n"
                "    return _FakeTensor(0.0)\n"
                "class cuda:\n"
                "    @staticmethod\n"
                "    def is_available(): return True\n"
                "    @staticmethod\n"
                "    def device_count(): return 2\n"
                "    @staticmethod\n"
                "    def get_device_name(i): return f'Simulated Chinese-Path GPU {i}'\n"
                "    @staticmethod\n"
                "    def get_device_capability(i): return (8, 9)\n"
                "    @staticmethod\n"
                "    def synchronize(i=None): return None\n"
                "class backends:\n"
                "    class mps:\n"
                "        @staticmethod\n"
                "        def is_available(): return False\n",
                encoding="utf-8",
            )

            ok, details = _check_cuda_device("cuda:0", py_exe, python_path_entries=(chinese_sp,))

        self.assertTrue(ok, details)
        self.assertEqual(details.get("torch_version"), "2.11.0+cu128")
        self.assertEqual(details.get("torch_cuda_build"), "12.8")
        self.assertEqual(details.get("device_count"), 2)
        self.assertEqual(details.get("per_device_ok"), {"0": True, "1": True})
        self.assertEqual(details.get("usable_indices"), [0, 1])
        self.assertEqual(details.get("device_name"), "Simulated Chinese-Path GPU 0")


class MultiGpuProbingAndVisibleDevicesParityTests(unittest.TestCase):
    """Item 3 (P1): Multi-GPU per-device validation and CUDA_VISIBLE_DEVICES=1,0 DDParcel/DDSeg parity."""

    def _write_multigpu_fake_torch(self, sp_dir: Path, *, fail_indices: set[int]) -> None:
        torch_dir = sp_dir / "torch"
        torch_dir.mkdir(parents=True, exist_ok=True)
        (torch_dir / "__init__.py").write_text(
            "__version__ = '2.11.0+cu126'\n"
            f"FAIL_INDICES = {sorted(fail_indices)!r}\n"
            "class version:\n"
            "    cuda = '12.6'\n"
            "class _FakeTensor:\n"
            "    def __init__(self, val, idx):\n"
            "        self._v = val\n"
            "        self._idx = idx\n"
            "    def __add__(self, other):\n"
            "        if self._idx in FAIL_INDICES:\n"
            "            raise RuntimeError(f'CUDA error: no kernel image is available for execution on cuda:{self._idx}')\n"
            "        return _FakeTensor(self._v + other, self._idx)\n"
            "    def item(self): return float(self._v)\n"
            "def zeros(n, device='cuda:0'):\n"
            "    idx = int(str(device).split(':', 1)[1]) if ':' in str(device) else 0\n"
            "    return _FakeTensor(0.0, idx)\n"
            "class cuda:\n"
            "    @staticmethod\n"
            "    def is_available(): return True\n"
            "    @staticmethod\n"
            "    def device_count(): return 2\n"
            "    @staticmethod\n"
            "    def get_device_name(i): return f'Simulated GPU {i}'\n"
            "    @staticmethod\n"
            "    def get_device_capability(i): return (12, 0) if i in FAIL_INDICES else (8, 9)\n"
            "    @staticmethod\n"
            "    def synchronize(i=None): return None\n"
            "class backends:\n"
            "    class mps:\n"
            "        @staticmethod\n"
            "        def is_available(): return False\n",
            encoding="utf-8",
        )

    def test_multigpu_probe_rejects_broken_cuda1_and_accepts_healthy_cuda1_when_cuda0_fails(self) -> None:
        from shared.environment import _check_cuda_device, resolve_auto_torch_device
        from shared.external_process import find_pythonslicer

        py_exe = find_pythonslicer(Path(sys.executable).parent.parent) or Path(sys.executable)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Case A: cuda:0 healthy, cuda:1 fails with no kernel image
            sp_a = root / "case_a"
            self._write_multigpu_fake_torch(sp_a, fail_indices={1})
            ok0, det0 = _check_cuda_device("cuda:0", py_exe, python_path_entries=(sp_a,))
            ok1, det1 = _check_cuda_device("cuda:1", py_exe, python_path_entries=(sp_a,))
            auto_a = resolve_auto_torch_device("auto", py_exe, python_path_entries=(sp_a,))

            self.assertTrue(ok0, det0)
            self.assertFalse(ok1, det1)
            self.assertEqual(det1["per_device_ok"], {"0": True, "1": False})
            self.assertEqual(det1["usable_indices"], [0])
            self.assertIn("no kernel image", det1["per_device_error"]["1"].lower())
            self.assertIn("no kernel image", det1["cuda_kernel_error"].lower())
            self.assertEqual(auto_a, "cuda:0")

            # Case B: cuda:0 fails with no kernel image, cuda:1 healthy
            sp_b = root / "case_b"
            self._write_multigpu_fake_torch(sp_b, fail_indices={0})
            ok0_b, det0_b = _check_cuda_device("cuda:0", py_exe, python_path_entries=(sp_b,))
            ok1_b, det1_b = _check_cuda_device("cuda:1", py_exe, python_path_entries=(sp_b,))
            auto_b = resolve_auto_torch_device("auto", py_exe, python_path_entries=(sp_b,))

            self.assertFalse(ok0_b, det0_b)
            self.assertTrue(ok1_b, det1_b)
            self.assertEqual(det1_b["per_device_ok"], {"0": False, "1": True})
            self.assertEqual(det1_b["usable_indices"], [1])
            self.assertEqual(det1_b["device_name"], "Simulated GPU 1")
            self.assertIn("no kernel image", det0_b["per_device_error"]["0"].lower())
            self.assertEqual(auto_b, "cuda:1")

    def test_cuda_visible_devices_1_0_ddparcel_matches_ddseg_exactly(self) -> None:
        from anatomy.parcellation.device import logical_cuda_index
        from shared.environment import resolve_auto_torch_device

        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1,0"}):
            for requested in ("cuda:0", "cuda:1"):
                expected_idx = int(requested.split(":")[1])
                # DDSeg resolves via resolve_auto_torch_device and passes directly to PyTorch
                ddseg_device = resolve_auto_torch_device(requested)
                # DDParcel extracts the GPU arg and maps via logical_cuda_index
                gpu_arg = requested.split(":", 1)[1]
                ddparcel_idx = logical_cuda_index(gpu_arg)
                ddparcel_device = f"cuda:{ddparcel_idx}"

                self.assertEqual(ddseg_device, requested)
                self.assertEqual(ddparcel_idx, expected_idx)
                self.assertEqual(ddparcel_device, ddseg_device)
                self.assertEqual(logical_cuda_index(requested), expected_idx)

        # Preserve legacy physical-ID fallback when requested ID is outside visible count
        self.assertEqual(logical_cuda_index("1", "1"), 0)
        self.assertEqual(logical_cuda_index("3", "1,3"), 1)
        self.assertEqual(logical_cuda_index("GPU-bbb", "GPU-aaa,GPU-bbb"), 1)


class NoddiAndDdsegFallbackProvenanceAndTdrTests(unittest.TestCase):
    """Item 4 (P1): Accurate CPU device provenance after fallback in NODDI/DDSeg + WDDM TDR handling."""

    def _noddi_problem(self):
        rng = np.random.RandomState(23)
        n_vox, n_grad, n_wm, ndirs = 10, 16, 6, 2
        K_wm = np.abs(rng.randn(n_wm, ndirs, n_grad)).astype(np.float64) + 0.1
        K_iso = np.abs(rng.randn(n_grad)).astype(np.float64) + 0.1
        K_norms = np.linalg.norm(K_wm[:, 0, 2:], axis=1) + 1e-6
        icvf = np.linspace(0.2, 0.8, n_wm)
        kappa = np.linspace(0.1, 0.9, n_wm)
        dwi_idx = np.arange(2, n_grad, dtype=np.int64)
        y = np.abs(rng.randn(n_vox, n_grad)).astype(np.float64) + 0.2
        dir_idx = rng.randint(0, ndirs, size=n_vox).astype(np.int64)
        return dict(
            y=y, dir_idx=dir_idx, K_wm=K_wm, K_iso=K_iso, K_norms=K_norms,
            icvf=icvf, kappa=kappa, dwi_idx=dwi_idx, lambda1=0.5, lambda2=0.001, ndirs=ndirs,
        )

    def test_noddi_init_oom_records_cpu_device_in_exec_meta_and_config_pickle(self) -> None:
        import torch
        from microstructure import noddi_accel

        prob = self._noddi_problem()
        ref_est, ref_rmse, ref_nrmse = noddi_accel._fit_noddi_cpu(**prob, nthreads=1)

        # 1. Direct _fit_noddi_cuda init-time OOM provenance
        exec_info: dict[str, object] = {}
        logs: list[str] = []
        with (
            patch.object(torch, "as_tensor", side_effect=RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")),
            patch("shared.hardware_adapt.empty_cuda_cache_safely"),
        ):
            est, rmse, nrmse = noddi_accel._fit_noddi_cuda(
                **prob,
                device="cuda:0",
                explicit_cuda=False,
                nthreads=1,
                execution_info=exec_info,
                log=logs.append,
            )
        np.testing.assert_allclose(est, ref_est, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(rmse, ref_rmse, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(nrmse, ref_nrmse, rtol=1e-6, atol=1e-6)
        self.assertEqual(exec_info["device"], "cpu")
        self.assertTrue(exec_info["device_fallback"])
        self.assertEqual(exec_info["fallback_from"], "cuda:0")
        self.assertEqual(exec_info["fallback_kind"], "cuda_oom")
        self.assertTrue(any("falling back to cpu" in m.lower() for m in logs), logs)

        # 2. Full run_accelerated_noddi_fit provenance (exec_meta AND config.pickle MUST record cpu!)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vol_shape = (5, 2, 1)
            mask_arr = np.ones(vol_shape, dtype=np.uint8)
            dwi_img = nib.Nifti1Image(np.ones((*vol_shape, 16), dtype=np.float32), np.eye(4))
            mask_img = nib.Nifti1Image(mask_arr, np.eye(4))
            scheme_path = root / "dwi.scheme"
            scheme_path.write_text("VERSION: BVECTOR\n", encoding="utf-8")
            fake_scheme = SimpleNamespace(dwi_idx=prob["dwi_idx"], b0_idx=np.array([0, 1]), b=np.zeros(16))
            fake_amico = SimpleNamespace(
                __version__="2.0.3",
                scheme=SimpleNamespace(Scheme=lambda path, thr: fake_scheme),
                lut=object(),
            )
            fake_kernels = {
                "model": "NODDI",
                "wm": prob["K_wm"],
                "iso": prob["K_iso"],
                "norms": prob["K_norms"],
                "icvf": prob["icvf"],
                "kappa": prob["kappa"],
            }
            fake_config = SimpleNamespace(
                subject_id="sub01",
                noddi_b0_threshold=50,
                noddi_lmax=12,
                noddi_ndirs=2,
                noddi_solver_lambda1=0.5,
                noddi_solver_lambda2=0.001,
                noddi_device="auto",
            )
            out_dir = root / "amico_out"
            with (
                patch.object(noddi_accel, "_load_masked_normalized_dwi", return_value=(dwi_img, mask_img, mask_arr, prob["y"])),
                patch.object(noddi_accel, "_load_or_build_noddi_kernels", return_value=(fake_scheme, fake_kernels, np.zeros((2, 3)))),
                patch.object(noddi_accel, "_estimate_principal_directions_ols", return_value=np.tile([1.0, 0.0, 0.0], (10, 1))),
                patch.object(noddi_accel, "_dirs_to_lut_indices", return_value=prob["dir_idx"]),
                patch.object(torch, "as_tensor", side_effect=RuntimeError("CUDA out of memory")),
                patch("shared.hardware_adapt.empty_cuda_cache_safely"),
            ):
                _, _, _, _, exec_meta = noddi_accel.run_accelerated_noddi_fit(
                    amico_mod=fake_amico,
                    config=fake_config,
                    cache_root=root / "cache",
                    dwi_path=root / "dwi.nii.gz",
                    mask_path=root / "mask.nii.gz",
                    scheme_path=scheme_path,
                    amico_output_dir=out_dir,
                    nthreads=1,
                    resolved_device="cuda:0",
                    ensure_rotations_fn=lambda *a, **kw: None,
                )

            self.assertEqual(exec_meta["device"], "cpu")
            self.assertEqual(exec_meta["engine"], "batched_lawson_hanson_fp64_cpu")
            self.assertEqual(exec_meta["resolved_device"], "cuda:0")
            self.assertTrue(exec_meta["device_fallback"])
            self.assertEqual(exec_meta["fallback_from"], "cuda:0")
            with (out_dir / "config.pickle").open("rb") as fh:
                saved_cfg = pickle.load(fh)
            self.assertEqual(saved_cfg["device"], "cpu")
            self.assertEqual(saved_cfg["fallback_from"], "cuda:0")

    def test_noddi_wddm_tdr_timeout_auto_fallback_vs_explicit_cuda_error(self) -> None:
        import torch
        from microstructure import noddi_accel

        prob = self._noddi_problem()
        ref_est, ref_rmse, ref_nrmse = noddi_accel._fit_noddi_cpu(**prob, nthreads=1)

        tdr_messages = (
            "CUDA error: the launch timed out and was terminated",
            "RuntimeError: cudaErrorLaunchTimeout on device 0",
            "CUDA error: unspecified launch failure",
        )
        for msg in tdr_messages:
            with self.subTest(tdr_error=msg):
                exec_info: dict[str, object] = {}
                logs: list[str] = []
                with (
                    patch.object(torch, "as_tensor", side_effect=RuntimeError(msg)),
                    patch("shared.hardware_adapt.empty_cuda_cache_safely"),
                ):
                    est, rmse, nrmse = noddi_accel._fit_noddi_cuda(
                        **prob,
                        device="cuda:0",
                        explicit_cuda=False,
                        nthreads=1,
                        execution_info=exec_info,
                        log=logs.append,
                    )
                np.testing.assert_allclose(est, ref_est, rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(rmse, ref_rmse, rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(nrmse, ref_nrmse, rtol=1e-6, atol=1e-6)
                self.assertEqual(exec_info["device"], "cpu")
                self.assertTrue(exec_info["device_fallback"])
                self.assertEqual(exec_info["fallback_from"], "cuda:0")
                self.assertEqual(exec_info["fallback_kind"], "wddm_tdr")
                self.assertTrue(any("tdr" in line.lower() and "not oom" in line.lower() for line in logs), logs)

                # Explicit CUDA mode MUST fail loudly with an actionable WDDM TDR explanation
                with (
                    patch.object(torch, "as_tensor", side_effect=RuntimeError(msg)),
                    patch("shared.hardware_adapt.empty_cuda_cache_safely"),
                    self.assertRaisesRegex(RuntimeError, r"(?i)wddm\s+tdr.*tdrdelay"),
                ):
                    noddi_accel._fit_noddi_cuda(
                        **prob,
                        device="cuda:0",
                        explicit_cuda=True,
                        nthreads=1,
                    )

    def test_noddi_chunk_level_oom_fallback_records_cpu_device(self) -> None:
        import torch
        from microstructure import noddi_accel

        prob = self._noddi_problem()
        ref_est, ref_rmse, ref_nrmse = noddi_accel._fit_noddi_cpu(**prob, nthreads=1)
        orig_as_tensor = torch.as_tensor
        orig_empty = torch.empty
        orig_eye = torch.eye
        orig_zeros = torch.zeros

        def cpu_as_tensor(data, *args, **kwargs):
            kwargs.pop("device", None)
            return orig_as_tensor(data, *args, **kwargs)

        def cpu_empty(*args, **kwargs):
            kwargs.pop("device", None)
            return orig_empty(*args, **kwargs)

        def cpu_eye(*args, **kwargs):
            kwargs.pop("device", None)
            return orig_eye(*args, **kwargs)

        def cpu_zeros(*args, **kwargs):
            kwargs.pop("device", None)
            return orig_zeros(*args, **kwargs)

        exec_info: dict[str, object] = {}
        logs: list[str] = []
        with (
            patch.object(torch, "as_tensor", side_effect=cpu_as_tensor),
            patch.object(torch, "empty", side_effect=cpu_empty),
            patch.object(torch, "eye", side_effect=cpu_eye),
            patch.object(torch, "zeros", side_effect=cpu_zeros),
            patch.object(noddi_accel, "_solve_qp_batched_torch", side_effect=RuntimeError("CUDA out of memory in chunk QP")),
            patch("shared.hardware_adapt.empty_cuda_cache_safely"),
        ):
            est, rmse, nrmse = noddi_accel._fit_noddi_cuda(
                **prob,
                device="cuda:0",
                explicit_cuda=False,
                nthreads=1,
                execution_info=exec_info,
                log=logs.append,
            )
        np.testing.assert_allclose(est, ref_est, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(rmse, ref_rmse, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(nrmse, ref_nrmse, rtol=1e-6, atol=1e-6)
        self.assertEqual(exec_info["device"], "cpu")
        self.assertTrue(exec_info["device_fallback"])
        self.assertEqual(exec_info["fallback_from"], "cuda:0")
        self.assertEqual(exec_info["cpu_fallback_voxels"], len(prob["y"]))
        self.assertTrue(any("falling back to cpu" in m.lower() for m in logs), logs)

    def test_ddseg_cpu_fallback_records_cpu_device_in_ddseg_qc_json(self) -> None:
        from anatomy.segmentation.config import DDSegConfig
        from anatomy.segmentation import pipeline as ddseg_pipeline

        shape = (8, 8, 8)
        affine = np.eye(4, dtype=np.float64)

        class FlakyCudaViewModel:
            input_layout = "NCHW"
            output_layout = "NCHW"

            def __init__(self, dev: str):
                self.device = dev

            def forward(self, x: np.ndarray) -> np.ndarray:
                if self.device.startswith("cuda"):
                    raise RuntimeError("CUDA out of memory. Tried to allocate 1.80 GiB")
                # Deterministic class logits favoring class 1 (WM)
                out = np.zeros((x.shape[0], 4, x.shape[2], x.shape[3]), dtype=np.float32)
                out[:, 1, :, :] = 4.0
                return out

        def fake_load_models(_weights_dir, _prefix, device):
            return SimpleNamespace(
                axial=FlakyCudaViewModel(device),
                sagittal=FlakyCudaViewModel(device),
                coronal=FlakyCudaViewModel(device),
            )

        import nrrd

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mask_path = root / "mask.nii.gz"
            mask_data = np.ones(shape, dtype=np.uint8)
            nib.save(nib.Nifti1Image(mask_data, affine), str(mask_path))
            feature_dir = root / "features"
            feature_dir.mkdir()
            nrrd_hdr = {
                "space": "right-anterior-superior",
                "space directions": np.eye(3, dtype=np.float64),
                "space origin": np.zeros(3, dtype=np.float64),
            }
            for name, val in (("DTI_FA", 0.5), ("DTI_MD", 0.001), ("DTI_E1", 0.0015), ("DTI_E2", 0.0008), ("DTI_E3", 0.0007)):
                nrrd.write(str(feature_dir / f"{name}.nrrd"), np.full(shape, val, dtype=np.float32), header=dict(nrrd_hdr))

            cfg = DDSegConfig(
                input_mask_nii=mask_path,
                parameter_type="DTI",
                output_folder=root / "ddseg_out",
                weights_dir=root / "weights",
                dti_feature_dir=feature_dir,
                device="auto",
            )
            with (
                patch("shared.environment.resolve_auto_torch_device", return_value="cuda:0"),
                patch("anatomy.segmentation.model_loader.load_models", side_effect=fake_load_models),
                patch("shared.hardware_adapt.empty_gpu_cache_safely"),
            ):
                ddseg_pipeline.run_ddseg(cfg)

            qc = json.loads((cfg.output_folder / "ddseg_qc.json").read_text(encoding="utf-8"))
            self.assertEqual(qc["device"], "cpu")
            self.assertEqual(qc["requested_device"], "auto")
            self.assertEqual(qc["resolved_device"], "cuda:0")
            self.assertTrue(qc["device_fallback"])
            self.assertEqual(qc["fallback_from"], "cuda:0")
            self.assertIn("out of memory", qc["fallback_reason"].lower())


class DdsurferGlobalVramCapTests(unittest.TestCase):
    """Item 5 (P1): DDSurfer VRAM probe must strictly honor global SLICERDSPACE_MAX_VRAM_GIB cap."""

    def test_vram_probe_and_surface_decision_respect_3_5_gib_global_cap_on_8_gib_gpu(self) -> None:
        from shared import environment
        from anatomy.surface.prediction import PredictionStageConfig, predict_subject

        environment._CUDA_FREE_CACHE.clear()
        self.addCleanup(environment._CUDA_FREE_CACHE.clear)

        completed_8g = SimpleNamespace(stdout='{"free_gib": 8.0}')
        with (
            patch.object(environment, "_is_macos", return_value=False),
            patch.object(environment.subprocess, "run", return_value=completed_8g) as mock_run,
        ):
            # Without global cap: 8.0 GiB free -> CUDA allowed
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("SLICERDSPACE_MAX_VRAM_GIB", None)
                os.environ.pop("SLICERDSPACE_MAX_VRAM_GB", None)
                self.assertAlmostEqual(environment.cuda_free_gib("cuda:0", Path("/fake/python")), 8.0)
                self.assertEqual(environment.surface_gpu_memory_decision("auto", Path("/fake/python")), ("ok", ""))
                self.assertEqual(environment.surface_gpu_memory_decision("cuda:0", Path("/fake/python")), ("ok", ""))

            # Now set global cap = 3.5 GiB (even while 8.0 GiB is cached!): MUST clamp to 3.5 GiB!
            with patch.dict(os.environ, {"SLICERDSPACE_MAX_VRAM_GIB": "3.5"}, clear=False):
                self.assertAlmostEqual(environment.cuda_free_gib("cuda:0", Path("/fake/python")), 3.5)
                self.assertAlmostEqual(environment.cuda_free_gib("cuda:0", Path("/fake/python"), cached_only=True), 3.5)
                auto_action, auto_note = environment.surface_gpu_memory_decision("auto", Path("/fake/python"))
                cuda_action, cuda_reason = environment.surface_gpu_memory_decision("cuda:0", Path("/fake/python"))
                self.assertEqual(auto_action, "cpu")
                self.assertIn("3.5 GiB", auto_note)
                self.assertIn("SLICERDSPACE_MAX_VRAM_GIB", auto_note)
                self.assertEqual(cuda_action, "block")
                self.assertIn("3.5 GiB", cuda_reason)
                self.assertIn("SLICERDSPACE_MAX_VRAM_GIB", cuda_reason)

            # Only one subprocess probe was needed; dynamic cap clamping worked immediately on cache
            self.assertEqual(mock_run.call_count, 1)

        # End-to-end predict_subject(device="auto") with 8 GiB free + 3.5 GiB cap MUST launch CPU, not CUDA!
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
                patch.dict(os.environ, {"SLICERDSPACE_MAX_VRAM_GIB": "3.5"}, clear=False),
                patch("shared.environment.resolve_auto_torch_device", return_value="cuda:0"),
                patch("shared.environment._is_macos", return_value=False),
                patch("shared.environment.cuda_free_gib", return_value=8.0),
                patch("anatomy.surface.prediction.run_backend", side_effect=lambda cmd, **kw: captured_cmd.extend(map(str, cmd))),
            ):
                predict_subject(cfg)
        self.assertEqual(captured_cmd[captured_cmd.index("--device") + 1], "cpu")


class WmaCohortGpuSlotAccountingTests(unittest.TestCase):
    """Item 6 (P1): WMA executing with CUDA must reserve a GPU slot in cohort scheduling."""

    def test_wma_gpu_predicate_and_cohort_slot_reservation(self) -> None:
        from shared import cohort_schedule
        from tractography import wma

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gpu_wma = root / "wma_gpu"
            (gpu_wma / "whitematteranalysis" / "gpu").mkdir(parents=True)
            (gpu_wma / "whitematteranalysis" / "gpu" / "__init__.py").write_text("", encoding="utf-8")
            cpu_wma = root / "wma_legacy"
            (cpu_wma / "whitematteranalysis").mkdir(parents=True)

            base = dict(
                subject_id="sub01",
                input_tractography=root / "in.vtk",
                atlas_dir=root / "atlas",
                output_dir=root / "out",
                slicer_executable=root / "Slicer.exe",
            )
            cfg_cuda = wma.WMAOrgAtlasConfig(**base, wma_root=gpu_wma, device="cuda", execute=True)
            cfg_auto = wma.WMAOrgAtlasConfig(**base, wma_root=gpu_wma, device="auto", execute=True)
            cfg_cpu = wma.WMAOrgAtlasConfig(**base, wma_root=gpu_wma, device="cpu", execute=True)
            cfg_dry = wma.WMAOrgAtlasConfig(**base, wma_root=gpu_wma, device="cuda", execute=False)
            cfg_legacy = wma.WMAOrgAtlasConfig(**base, wma_root=cpu_wma, device="cuda", execute=True)

            self.assertTrue(wma._wma_uses_gpu(cfg_cuda))
            self.assertFalse(wma._wma_uses_gpu(cfg_cpu))
            self.assertFalse(wma._wma_uses_gpu(cfg_dry))
            self.assertFalse(wma._wma_uses_gpu(cfg_legacy))

            fake_torch_cuda = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
            fake_torch_cpu = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
            with patch.dict(sys.modules, {"torch": fake_torch_cuda}):
                self.assertTrue(wma._wma_uses_gpu(cfg_auto))
            with patch.dict(sys.modules, {"torch": fake_torch_cpu}):
                self.assertFalse(wma._wma_uses_gpu(cfg_auto))

            # Verify live cohort schedule admission reserves gpu=1 for CUDA WMA and gpu=0 for CPU WMA
            sched_dir = cohort_schedule.create_schedule(
                root / "sched",
                cpu_budget=4,
                memory_gb=8.0,
                stage_cpus=2,
                stage_memory_gb=2.0,
                gpu_slots=1,
            )
            with patch.dict(os.environ, {cohort_schedule.ENV: str(sched_dir)}):
                with patch.object(wma, "_inspect_wma_run", side_effect=RuntimeError("stop-after-admission")):
                    with self.assertRaisesRegex(RuntimeError, "stop-after-admission"):
                        wma.run_wma_org_atlas(cfg_cuda)
                    with self.assertRaisesRegex(RuntimeError, "stop-after-admission"):
                        wma.run_wma_org_atlas(cfg_cpu)

            events = [
                json.loads(line)
                for line in (sched_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            started = [row for row in events if row.get("event") == "started" and row.get("stage") == "wma"]
            self.assertEqual([row["gpu"] for row in started], [1, 0])


class WslBlackwellRtx50AndKernelSmokeTests(unittest.TestCase):
    """Item 7 (P2): WSL setup supports RTX 50 / Blackwell (cu128) and runs real GPU kernel smoke ops."""

    def test_blackwell_rtx50_selects_cu128_while_older_gpus_stay_on_cu126(self) -> None:
        from scripts.wsl import configure_device as wsl_dev

        # RTX 5090 detected via nvidia-smi --query-gpu (compute_cap 12.0)
        rtx5090_query = SimpleNamespace(returncode=0, stdout="NVIDIA GeForce RTX 5090, 12.0, GPU-5090\n", stderr="")
        rtx5090_list = SimpleNamespace(returncode=0, stdout="GPU 0: NVIDIA GeForce RTX 5090 (UUID: GPU-5090)\n", stderr="")
        with (
            patch.object(wsl_dev, "_find_nvidia_smi", return_value="nvidia-smi"),
            patch.object(wsl_dev.subprocess, "run", side_effect=[rtx5090_list, rtx5090_query]),
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ.pop("SLICERDSPACE_WSL_CUDA_FLAVOR", None)
            plan_5090 = wsl_dev.device_plan("auto")
        self.assertEqual(plan_5090["resolved"], "cuda")
        self.assertEqual(plan_5090["cuda_flavor"], "cu128")
        self.assertEqual(plan_5090["torch_requirement"], f"torch=={wsl_dev.TORCH_VERSION}+cu128")
        self.assertEqual(plan_5090["torch_index"], "https://download.pytorch.org/whl/cu128")

        # RTX 5070 passed via compute_capabilities / gpu_names
        plan_5070 = wsl_dev.device_plan("auto", True, compute_capabilities=[(12, 0)])
        self.assertEqual(plan_5070["cuda_flavor"], "cu128")
        self.assertTrue(plan_5070["torch_index"].endswith("/cu128"))

        plan_b200 = wsl_dev.device_plan("auto", True, compute_capabilities=[(10, 0)])
        self.assertEqual(plan_b200["cuda_flavor"], "cu128")

        # Older Turing / Ampere / Ada GPUs keep cu126 cleanly
        for cap, name in (((7, 5), "RTX 2080 Ti"), ((8, 6), "RTX 3060"), ((8, 9), "RTX 4090")):
            plan_old = wsl_dev.device_plan("auto", True, compute_capabilities=[cap], gpu_names=[name])
            self.assertEqual(plan_old["cuda_flavor"], "cu126")
            self.assertTrue(plan_old["torch_index"].endswith("/cu126"))

        # Multi-GPU heterogeneous system respects CUDA_VISIBLE_DEVICES
        het_query = SimpleNamespace(
            returncode=0,
            stdout="NVIDIA GeForce RTX 4090, 8.9, GPU-4090\nNVIDIA GeForce RTX 5090, 12.0, GPU-5090\n",
            stderr="",
        )
        het_list = SimpleNamespace(returncode=0, stdout="GPU 0: RTX 4090\nGPU 1: RTX 5090\n", stderr="")
        with (
            patch.object(wsl_dev, "_find_nvidia_smi", return_value="nvidia-smi"),
            patch.object(wsl_dev.subprocess, "run", side_effect=[het_list, het_query]),
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}, clear=False),
        ):
            self.assertEqual(wsl_dev.device_plan("auto")["cuda_flavor"], "cu128")
        with (
            patch.object(wsl_dev, "_find_nvidia_smi", return_value="nvidia-smi"),
            patch.object(wsl_dev.subprocess, "run", side_effect=[het_list, het_query]),
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=False),
        ):
            self.assertEqual(wsl_dev.device_plan("auto")["cuda_flavor"], "cu126")

    def test_wsl_smoke_probe_executes_real_gpu_kernels_and_catches_no_kernel_image(self) -> None:
        from scripts.wsl import configure_device as wsl_dev

        # Verify smoke test includes real compute ops + synchronization (not just allocation)
        for token in ("torch.empty", "torch.matmul", "F.conv2d", "torch.relu", "torch.cuda.synchronize"):
            self.assertIn(token, wsl_dev._CUDA_SMOKE_PROBE)

        # Execute the real _CUDA_SMOKE_PROBE script in a child Python where torch.empty succeeds
        # (simulating cu126 on RTX 5090!) but torch.matmul raises 'no kernel image'
        with tempfile.TemporaryDirectory() as tmp:
            sp = Path(tmp) / "fake_site"
            tdir = sp / "torch"
            (tdir / "nn").mkdir(parents=True)
            (tdir / "__init__.py").write_text(
                "__version__ = '2.11.0+cu126'\n"
                "float32 = 'float32'\n"
                "class version:\n"
                "    cuda = '12.6'\n"
                "def empty(*a, **kw): return object()\n"
                "def ones(*a, **kw): return object()\n"
                "def matmul(a, b):\n"
                "    raise RuntimeError('CUDA error: no kernel image is available for execution on the device')\n"
                "def relu(x): return x\n"
                "class cuda:\n"
                "    @staticmethod\n"
                "    def is_available(): return True\n"
                "    @staticmethod\n"
                "    def device_count(): return 1\n"
                "    @staticmethod\n"
                "    def get_device_name(i): return 'NVIDIA GeForce RTX 5090'\n"
                "    @staticmethod\n"
                "    def get_device_capability(i): return (12, 0)\n"
                "    @staticmethod\n"
                "    def synchronize(i=None): return None\n",
                encoding="utf-8",
            )
            (tdir / "nn" / "__init__.py").write_text("", encoding="utf-8")
            (tdir / "nn" / "functional.py").write_text("def conv2d(x, w): return x\n", encoding="utf-8")

            env = dict(os.environ)
            env["PYTHONPATH"] = str(sp) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            completed = subprocess.run(
                [sys.executable, "-c", wsl_dev._CUDA_SMOKE_PROBE, "cuda"],
                capture_output=True,
                text=True,
                env=env,
                check=True,
                timeout=30,
            )
            payload = json.loads(completed.stdout)

        self.assertTrue(payload["cuda_available"])
        self.assertFalse(payload["kernel_ok"])
        self.assertIn("no kernel image", payload["kernel_error"].lower())

        # configure() MUST reject this payload instead of reporting status='ready'
        plan = wsl_dev.device_plan("cuda", True)
        with patch.object(
            wsl_dev.subprocess,
            "run",
            side_effect=[SimpleNamespace(returncode=0), SimpleNamespace(stdout=json.dumps(payload))],
        ):
            with self.assertRaisesRegex(RuntimeError, r"(?i)smoke test failed.*no kernel image"):
                wsl_dev.configure(plan)


if __name__ == "__main__":
    unittest.main()
