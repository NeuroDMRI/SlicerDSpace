"""Unit tests for cross-platform microstructure, surface, segmentation, and parcellation performance optimizations."""

from __future__ import annotations

from contextlib import nullcontext
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class FreeWaterCpuThreadingTests(unittest.TestCase):
    def _make_fw_case(self, n_voxels: int = 1280):
        from dipy.core.gradients import gradient_table
        from dipy.reconst import fwdti
        from microstructure.freewater_dispatch import configure_freewater_dispatch

        rng = np.random.default_rng(1234)
        bvals = np.r_[np.zeros(6), np.full(30, 1000.0), np.full(30, 2000.0)]
        bvecs = rng.normal(size=(len(bvals), 3))
        bvecs /= np.linalg.norm(bvecs, axis=1, keepdims=True)
        bvecs[:6] = 0.0
        gtab = gradient_table(bvals, bvecs=bvecs)
        model = fwdti.FreeWaterTensorModel(gtab, fit_method="WLS")
        configure_freewater_dispatch(model, "WLS", "serial")
        tissue = np.exp(-bvals * np.sum(bvecs ** 2 * np.array([0.0016, 0.0006, 0.0003]), axis=1))
        f_vals = rng.uniform(0.05, 0.95, size=(n_voxels, 1))
        signals = 1000.0 * ((1.0 - f_vals) * tissue[None, :] + f_vals * np.exp(-bvals[None, :] * 0.003))
        signals = (signals + rng.normal(0.0, 2.0, size=signals.shape)).astype(np.float32)
        # Include an edge-case zero voxel and high-MD free-water-only voxel
        signals[0] = 0.0
        signals[1] = (1000.0 * np.exp(-bvals * 0.003)).astype(np.float32)
        return model, signals

    def test_multithreaded_cpu_bit_for_bit_parity_with_single_threaded(self) -> None:
        from microstructure.freewater_wls import wls_batch_fit

        model, signals = self._make_fw_case(n_voxels=1280)
        with patch.dict(os.environ, {"SLICERDSPACE_STAGE_CPUS": "1"}, clear=False):
            fit_single, dev_single = wls_batch_fit(model, signals, device="cpu")
        with patch.dict(os.environ, {"SLICERDSPACE_STAGE_CPUS": "4"}, clear=False):
            fit_multi, dev_multi = wls_batch_fit(model, signals, device="cpu")

        self.assertEqual(dev_single, "cpu")
        self.assertEqual(dev_multi, "cpu")
        np.testing.assert_array_equal(fit_multi.model_params, fit_single.model_params)
        self.assertEqual(float(np.max(np.abs(fit_multi.model_params - fit_single.model_params))), 0.0)

    def test_small_input_or_single_worker_avoids_thread_pool(self) -> None:
        from microstructure import freewater_wls

        model, signals = self._make_fw_case(n_voxels=256)
        with patch.object(freewater_wls, "ThreadPoolExecutor", side_effect=AssertionError("pool not expected")):
            fit_small, _ = freewater_wls.wls_batch_fit(model, signals, device="cpu")
        self.assertEqual(fit_small.model_params.shape, (256, 13))

        _, larger = self._make_fw_case(n_voxels=1024)
        with (
            patch.dict(os.environ, {"SLICERDSPACE_STAGE_CPUS": "1"}, clear=False),
            patch.object(freewater_wls, "ThreadPoolExecutor", side_effect=AssertionError("pool not expected")),
        ):
            fit_1cpu, _ = freewater_wls.wls_batch_fit(model, larger, device="cpu")
        self.assertEqual(fit_1cpu.model_params.shape, (1024, 13))

    def test_worker_count_policy_across_server_pc_mac_low_ram_and_stage_cap(self) -> None:
        from microstructure.freewater_wls import _resolve_fw_cpu_workers

        # High-RAM Linux server (>48 GiB) allows up to 16 workers
        with (
            patch.dict(os.environ, {}, clear=False),
            patch("microstructure.freewater_wls.sys.platform", "linux"),
            patch("shared.hardware_adapt.host_total_ram_gib", return_value=256.0),
            patch("shared.hardware_adapt.is_apple_silicon_host", return_value=False),
            patch("shared.numeric_threads.os.cpu_count", return_value=144),
        ):
            os.environ.pop("SLICERDSPACE_STAGE_CPUS", None)
            self.assertEqual(_resolve_fw_cpu_workers(20000), 16)

        # Stage CPU reservation strictly caps server worker count
        with (
            patch.dict(os.environ, {"SLICERDSPACE_STAGE_CPUS": "3"}, clear=False),
            patch("microstructure.freewater_wls.sys.platform", "linux"),
            patch("shared.hardware_adapt.host_total_ram_gib", return_value=256.0),
            patch("shared.hardware_adapt.is_apple_silicon_host", return_value=False),
        ):
            self.assertEqual(_resolve_fw_cpu_workers(20000), 3)

        # Low-RAM host (<= 12 GiB) caps at 4
        with (
            patch.dict(os.environ, {}, clear=False),
            patch("microstructure.freewater_wls.sys.platform", "win32"),
            patch("shared.hardware_adapt.host_total_ram_gib", return_value=8.0),
            patch("shared.hardware_adapt.is_apple_silicon_host", return_value=False),
            patch("shared.numeric_threads.os.cpu_count", return_value=16),
        ):
            os.environ.pop("SLICERDSPACE_STAGE_CPUS", None)
            self.assertEqual(_resolve_fw_cpu_workers(20000), 4)

        # 32 GiB Windows PC caps at 8
        with (
            patch.dict(os.environ, {}, clear=False),
            patch("microstructure.freewater_wls.sys.platform", "win32"),
            patch("shared.hardware_adapt.host_total_ram_gib", return_value=32.0),
            patch("shared.hardware_adapt.is_apple_silicon_host", return_value=False),
            patch("shared.compute_schedule.available_resources", return_value=(16, 32 * 1024**3)),
            patch("shared.numeric_threads.os.cpu_count", return_value=16),
        ):
            os.environ.pop("SLICERDSPACE_STAGE_CPUS", None)
            self.assertEqual(_resolve_fw_cpu_workers(20000), 8)

        # Apple Silicon 8-core Mac leaves E-core headroom (6 workers)
        with (
            patch.dict(os.environ, {}, clear=False),
            patch("microstructure.freewater_wls.sys.platform", "darwin"),
            patch("shared.hardware_adapt.sys.platform", "darwin"),
            patch("shared.hardware_adapt.host_total_ram_gib", return_value=64.0),
            patch("shared.hardware_adapt.is_apple_silicon_host", return_value=True),
            patch("shared.compute_schedule.available_resources", return_value=(8, 64 * 1024**3)),
            patch("shared.numeric_threads.os.cpu_count", return_value=8),
        ):
            os.environ.pop("SLICERDSPACE_STAGE_CPUS", None)
            self.assertEqual(_resolve_fw_cpu_workers(20000), 6)


class MKCurvePerfAndRamSafetyTests(unittest.TestCase):
    def test_initialize_worker_pins_all_five_blas_omp_env_vars(self) -> None:
        from microstructure import mkcurve

        keys = (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
        previous = {k: os.environ.get(k) for k in keys}
        old_limit = getattr(mkcurve, "_worker_thread_limit", None)
        try:
            mkcurve._initialize_worker(1)
            for key in keys:
                self.assertEqual(os.environ.get(key), "1")
        finally:
            cur_limit = getattr(mkcurve, "_worker_thread_limit", None)
            if cur_limit is not None and cur_limit is not old_limit and hasattr(cur_limit, "unregister"):
                cur_limit.unregister()
            mkcurve._worker_thread_limit = old_limit
            for key, val in previous.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val

    def test_mkcurve_ram_safe_spawn_clamping_and_low_ram_thread_fallback(self) -> None:
        import concurrent.futures
        from microstructure.mkcurve import run_mkcurve
        from Testing.Python.test_microstructure_dki import synthetic_case, write_case

        # Build a 300-voxel synthetic volume (>= 256 so spawn path is exercised)
        case = synthetic_case(noisy=True)
        # Tile (2,3,2) -> (6,10,5) = 300 voxels (299 masked)
        data = np.tile(case["data"], (3, 4, 3, 1))[:6, :10, :5, :]
        mask = np.ones((6, 10, 5), dtype=np.uint8)
        mask[0, 0, 0] = 0
        big_case = {**case, "data": data, "mask": mask}
        grid = np.arange(100.0, 2200.0, 300.0)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = write_case(root, big_case)
            inputs = [paths[k] for k in ("dwi", "bval", "bvec", "mask")]

            # 1) Simulate 32 GiB Windows PC requesting 8 workers -> ProcessPool clamped to 4
            seen_proc_workers: list[int] = []
            logs_clamp: list[str] = []

            def fake_proc_pool(*, max_workers, mp_context=None, initializer=None, initargs=()):
                seen_proc_workers.append(int(max_workers))
                return concurrent.futures.ThreadPoolExecutor(max_workers=int(max_workers))

            fake_main = types.SimpleNamespace(__file__=__file__)
            with (
                patch.dict(os.environ, {}, clear=False),
                patch.dict(sys.modules, {"__main__": fake_main}),
                patch("shared.hardware_adapt.adaptive_spawn_workers", return_value=4) as adapt_mock,
                patch("concurrent.futures.ProcessPoolExecutor", side_effect=fake_proc_pool),
            ):
                os.environ.pop("SLICERDSPACE_STAGE_CPUS", None)
                rec = run_mkcurve(
                    *inputs,
                    root / "clamped",
                    grid=grid,
                    voxels_per_chunk=64,
                    num_workers=8,
                    worker_threads=1,
                    log=logs_clamp.append,
                )
            adapt_mock.assert_called_once()
            self.assertEqual(seen_proc_workers, [4])
            self.assertEqual(rec["settings"]["num_workers"], 8)
            self.assertEqual(rec["settings"]["effective_workers"], 4)
            self.assertEqual(rec["settings"]["max_inflight_chunks"], 4)
            self.assertTrue(any("Clamping spawned workers from 8 to 4" in m for m in logs_clamp))

            # 2) Simulate 8 GiB low-RAM host (adaptive_spawn_workers -> 1) -> falls back to ThreadPool (<=4)
            seen_thread_workers: list[int] = []
            logs_low: list[str] = []
            real_thread_pool = concurrent.futures.ThreadPoolExecutor

            def spy_thread_pool(*, max_workers, initializer=None, initargs=()):
                seen_thread_workers.append(int(max_workers))
                return real_thread_pool(max_workers=int(max_workers))

            with (
                patch.dict(os.environ, {}, clear=False),
                patch.dict(sys.modules, {"__main__": fake_main}),
                patch("shared.hardware_adapt.adaptive_spawn_workers", return_value=1),
                patch("concurrent.futures.ProcessPoolExecutor", side_effect=AssertionError("must not spawn on low RAM")),
                patch("concurrent.futures.ThreadPoolExecutor", side_effect=spy_thread_pool),
            ):
                os.environ.pop("SLICERDSPACE_STAGE_CPUS", None)
                rec_low = run_mkcurve(
                    *inputs,
                    root / "lowram",
                    grid=grid,
                    voxels_per_chunk=64,
                    num_workers=8,
                    worker_threads=1,
                    log=logs_low.append,
                )
            self.assertEqual(seen_thread_workers, [4])
            self.assertEqual(rec_low["settings"]["num_workers"], 8)
            self.assertEqual(rec_low["settings"]["effective_workers"], 4)
            self.assertTrue(any("Low host RAM" in m for m in logs_low))


class NoddiAccelPerfTests(unittest.TestCase):
    def _make_noddi_problem(self):
        rng = np.random.default_rng(99)
        n_vox, n_grad, n_wm, ndirs = 64, 30, 12, 8
        K_wm = rng.uniform(0.1, 1.0, size=(n_wm, ndirs, n_grad)).astype(np.float32)
        K_iso = np.linspace(1.0, 0.2, n_grad, dtype=np.float32)
        dwi_idx = np.arange(2, n_grad, dtype=np.int64)
        K_norms = np.ones(n_wm, dtype=np.float64)
        icvf = np.linspace(0.1, 0.9, n_wm, dtype=np.float64)
        kappa = np.linspace(0.5, 8.0, n_wm, dtype=np.float64)
        # Skewed direction counts to exercise descending LPT bucket sorting
        dir_idx = np.array(([0] * 30) + ([3] * 20) + ([5] * 10) + ([7] * 4), dtype=np.int64)
        y = rng.uniform(0.05, 1.0, size=(n_vox, n_grad)).astype(np.float64)
        return dict(
            y=y,
            dir_idx=dir_idx,
            K_wm=K_wm,
            K_iso=K_iso,
            K_norms=K_norms,
            icvf=icvf,
            kappa=kappa,
            dwi_idx=dwi_idx,
            lambda1=0.5,
            lambda2=0.001,
            ndirs=ndirs,
        )

    def test_fit_noddi_cpu_lpt_multithreaded_matches_serial_exact(self) -> None:
        from microstructure.noddi_accel import _fit_noddi_cpu

        prob = self._make_noddi_problem()
        est1, rmse1, nrmse1 = _fit_noddi_cpu(**prob, nthreads=1)
        est4, rmse4, nrmse4 = _fit_noddi_cpu(**prob, nthreads=4)
        np.testing.assert_array_equal(est4, est1)
        np.testing.assert_array_equal(rmse4, rmse1)
        np.testing.assert_array_equal(nrmse4, nrmse1)

    def test_fit_noddi_cuda_setup_oom_falls_back_when_auto_and_raises_when_explicit(self) -> None:
        from microstructure import noddi_accel

        if noddi_accel.torch is None:
            self.skipTest("torch unavailable")
        prob = self._make_noddi_problem()
        expected = noddi_accel._fit_noddi_cpu(**prob, nthreads=1)

        with (
            patch.object(noddi_accel.torch, "as_tensor", side_effect=RuntimeError("CUDA out of memory")),
            patch("shared.hardware_adapt.empty_cuda_cache_safely") as empty_mock,
        ):
            est, rmse, nrmse = noddi_accel._fit_noddi_cuda(
                **prob, device="cuda:0", explicit_cuda=False, nthreads=1
            )
            np.testing.assert_array_equal(est, expected[0])
            np.testing.assert_array_equal(rmse, expected[1])
            np.testing.assert_array_equal(nrmse, expected[2])
            self.assertGreaterEqual(empty_mock.call_count, 1)

            with self.assertRaisesRegex(RuntimeError, "CUDA out of memory"):
                noddi_accel._fit_noddi_cuda(
                    **prob, device="cuda:0", explicit_cuda=True, nthreads=1
                )

    def test_load_masked_normalized_dwi_streaming_matches_full_and_closes_memmap(self) -> None:
        from microstructure import noddi_accel

        rng = np.random.default_rng(7)
        vol = rng.uniform(50.0, 1000.0, size=(4, 5, 3, 6)).astype(np.float32)
        mask = np.zeros((4, 5, 3), dtype=np.uint8)
        mask[1:3, 1:4, 0:2] = 1
        scheme = types.SimpleNamespace(nS=6, b0_count=2, b0_idx=[0, 1])

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dwi_nii = root / "dwi.nii"
            dwi_gz = root / "dwi.nii.gz"
            mask_gz = root / "mask.nii.gz"
            nib.save(nib.Nifti1Image(vol, np.eye(4)), str(dwi_nii))
            nib.save(nib.Nifti1Image(vol, np.eye(4)), str(dwi_gz))
            nib.save(nib.Nifti1Image(mask, np.eye(4)), str(mask_gz))

            # Reference via non-fast nibabel path
            with patch.object(noddi_accel, "_try_fast_gunzip_nifti1_4d", return_value=None):
                _, _, _, y_ref = noddi_accel._load_masked_normalized_dwi(
                    dwi_path=dwi_gz, mask_path=mask_gz, scheme=scheme
                )

            # Fast memmap streaming path
            hdr_img = nib.load(str(dwi_nii), mmap="r", keep_file_open=False)
            vox_off = int(hdr_img.header.get_data_offset()) or 352
            raw_mm = np.memmap(dwi_nii, dtype="<f4", mode="r", offset=vox_off, shape=vol.shape, order="F")
            with patch.object(noddi_accel, "_try_fast_gunzip_nifti1_4d", return_value=(hdr_img, raw_mm)):
                _, _, _, y_fast = noddi_accel._load_masked_normalized_dwi(
                    dwi_path=dwi_gz, mask_path=mask_gz, scheme=scheme
                )
            np.testing.assert_array_equal(y_fast, y_ref)
            self.assertTrue(raw_mm._mmap.closed)


class DDSurferPerfAndConsumerGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._dd_root = str(REPO_ROOT / "anatomy/surface/ddsurfer_backend")
        cls._saved_modules = {
            k: v
            for k, v in sys.modules.items()
            if k in ("preprocessing", "inference", "utils", "model")
            or k.startswith(("preprocessing.", "inference.", "utils.", "model."))
        }
        for k in list(cls._saved_modules):
            sys.modules.pop(k, None)
        cls._added = cls._dd_root not in sys.path
        if cls._added:
            sys.path.insert(0, cls._dd_root)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._added and cls._dd_root in sys.path:
            sys.path.remove(cls._dd_root)
        for k in list(sys.modules):
            if k in ("preprocessing", "inference", "utils", "model") or k.startswith(
                ("preprocessing.", "inference.", "utils.", "model.")
            ):
                sys.modules.pop(k, None)
        sys.modules.update(cls._saved_modules)

    def _make_predictor_fixture(self, root: Path):
        ckpt_dir = root / "ckpts"
        tmpl_dir = root / "tmpl"
        out_dir = root / "out"
        ckpt_dir.mkdir()
        tmpl_dir.mkdir()
        (tmpl_dir / "left.obj").write_text("", encoding="utf-8")
        manifest = {
            "architecture": {"step_size": 0.1, "M": 4, "R": 2},
            "geometry": {"affine": np.eye(4).tolist(), "full_shape": [176, 224, 176]},
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
        return args, fake_mesh

    def test_ddsurfer_consumer_8_12gib_cuda_lazy_vs_insufficient_vs_high_vram(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch unavailable")
        from inference import predict as dd_predict
        from utils.preflight import check_device

        with tempfile.TemporaryDirectory() as td:
            args, fake_mesh = self._make_predictor_fixture(Path(td))
            fake_cuda_dev = types.SimpleNamespace(type="cuda", index=0)

            # 8 GiB & 11 GiB & 12 GiB free -> accepted with lazy single-surface loading
            for free_gib, total_gib in ((7.5, 8.0), (8.0, 8.0), (11.0, 12.0), (12.0, 12.0)):
                with (
                    patch.object(dd_predict.torch, "device", return_value=fake_cuda_dev),
                    patch.object(dd_predict.torch.cuda, "set_device"),
                    patch.object(
                        dd_predict.torch.cuda,
                        "mem_get_info",
                        return_value=(int(free_gib * 1024**3), int(total_gib * 1024**3)),
                    ),
                    patch.object(dd_predict.torch.cuda, "set_per_process_memory_fraction"),
                    patch.object(dd_predict, "load_checksums", return_value={"wm.pt": "b" * 64, "pial.pt": "c" * 64}),
                    patch.object(dd_predict, "sha256_file", return_value="a" * 64),
                    patch.object(dd_predict, "load_mesh", return_value=fake_mesh),
                    patch.object(dd_predict, "load_model", return_value=object()) as load_mock,
                ):
                    predictor = dd_predict.SurfacePredictor(args)
                self.assertTrue(predictor._lazy_models, f"expected lazy on {free_gib} GiB free")
                self.assertEqual(load_mock.call_count, 0)

                # Preflight agrees
                pf_args = types.SimpleNamespace(device="cuda:0", precision="fp32", max_gpu_gib=24.0)
                with (
                    patch("torch.cuda.is_available", return_value=True),
                    patch("torch.cuda.device_count", return_value=1),
                    patch("torch.cuda.device", return_value=nullcontext()),
                    patch("torch.cuda.mem_get_info", return_value=(int(free_gib * 1024**3), int(total_gib * 1024**3))),
                    patch("torch.empty"),
                ):
                    self.assertEqual(check_device(pf_args, {"precision": "fp32"}), "fp32")

            # Truly insufficient VRAM (< 7 GiB free, e.g. 5 GiB or 6 GiB, or max_gpu_gib=5) -> rejected
            for free_gib, max_gib in ((5.0, 24.0), (6.0, 24.0), (24.0, 5.0)):
                bad_args = types.SimpleNamespace(**vars(args))
                bad_args.max_gpu_gib = max_gib
                with (
                    patch.object(dd_predict.torch, "device", return_value=fake_cuda_dev),
                    patch.object(dd_predict.torch.cuda, "set_device"),
                    patch.object(
                        dd_predict.torch.cuda,
                        "mem_get_info",
                        return_value=(int(free_gib * 1024**3), 24 * 1024**3),
                    ),
                    patch.object(dd_predict, "load_checksums", return_value={"wm.pt": "b" * 64, "pial.pt": "c" * 64}),
                    patch.object(dd_predict, "sha256_file", return_value="a" * 64),
                    patch.object(dd_predict, "load_mesh", return_value=fake_mesh),
                ):
                    with self.assertRaisesRegex(RuntimeError, "5.5-GiB"):
                        dd_predict.SurfacePredictor(bad_args)

            # High-VRAM (>= 18 GiB free) preserves eager dual-surface loading
            with (
                patch.object(dd_predict.torch, "device", return_value=fake_cuda_dev),
                patch.object(dd_predict.torch.cuda, "set_device"),
                patch.object(dd_predict.torch.cuda, "mem_get_info", return_value=(22 * 1024**3, 24 * 1024**3)),
                patch.object(dd_predict.torch.cuda, "set_per_process_memory_fraction"),
                patch.object(dd_predict, "load_checksums", return_value={"wm.pt": "b" * 64, "pial.pt": "c" * 64}),
                patch.object(dd_predict, "sha256_file", return_value="a" * 64),
                patch.object(dd_predict, "load_mesh", return_value=fake_mesh),
                patch.object(dd_predict, "load_model", return_value=object()) as load_mock,
            ):
                predictor = dd_predict.SurfacePredictor(args)
            self.assertFalse(predictor._lazy_models)
            self.assertEqual(load_mock.call_count, 2)

    def test_ddsurfer_cpu_flush_denormal_and_thread_resolution(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch unavailable")
        from inference import predict as dd_predict

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "sub01").mkdir()
            with (
                patch.dict(os.environ, {}, clear=False),
                patch.object(dd_predict, "SurfacePredictor"),
                patch.object(dd_predict.torch, "set_flush_denormal", return_value=True) as flush_mock,
                patch.object(dd_predict.torch, "set_num_threads") as threads_mock,
                patch("shared.hardware_adapt.optimal_cpu_torch_threads", return_value=8),
            ):
                os.environ.pop("SLICERDSPACE_STAGE_CPUS", None)
                dd_predict.main("left", ["--input_root", str(root), "--subjects", "sub01", "--device", "cpu"])
            flush_mock.assert_called_with(True)
            threads_mock.assert_called_with(8)

            # Respects SLICERDSPACE_STAGE_CPUS and explicit non-default --cpu-threads
            with patch.dict(os.environ, {"SLICERDSPACE_STAGE_CPUS": "3"}, clear=False):
                self.assertEqual(dd_predict._resolve_optimal_cpu_threads(3, "cpu"), 3)
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("SLICERDSPACE_STAGE_CPUS", None)
                self.assertEqual(dd_predict._resolve_optimal_cpu_threads(2, "cpu"), 2)
                self.assertEqual(dd_predict._resolve_optimal_cpu_threads(12, "cpu"), 12)


class SegmentationAndParcellationPerfTests(unittest.TestCase):
    def test_segmentation_predict_view_caps_and_restores_cpu_threads_and_halves_gpu_oom_batch(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch unavailable")
        import torch
        from anatomy.segmentation.inference import predict_view

        class _CpuModel:
            device = "cpu"
            input_layout = "NCHW"
            output_layout = "NCHW"

            def __init__(self) -> None:
                self.seen_threads: list[int] = []

            def forward(self, x: np.ndarray) -> np.ndarray:
                self.seen_threads.append(int(torch.get_num_threads()))
                return np.zeros((x.shape[0], 4, x.shape[2], x.shape[3]), dtype=np.float32)

        slices = [np.full((4, 4, 3), float(i + 1), dtype=np.float32) for i in range(4)]
        # Make slices non-uniform so uniform-slice cache does not bypass forward
        for idx, slc in enumerate(slices):
            slc[0, 0, 0] = float(idx + 100)

        cpu_model = _CpuModel()
        orig_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(max(2, orig_threads))
            with (
                patch.dict(os.environ, {"SLICERDSPACE_STAGE_CPUS": "2"}, clear=False),
                patch.object(torch, "set_flush_denormal", return_value=True) as flush_mock,
            ):
                if torch.get_num_threads() <= 2:
                    # Force higher initial thread count if host has <=2 cores
                    with (
                        patch.object(torch, "get_num_threads", return_value=16),
                        patch.object(torch, "set_num_threads") as set_mock,
                    ):
                        predict_view(cpu_model, slices, apply_softmax=False)
                        set_mock.assert_any_call(2)
                        set_mock.assert_any_call(16)
                else:
                    before = torch.get_num_threads()
                    predict_view(cpu_model, slices, apply_softmax=False)
                    self.assertEqual(cpu_model.seen_threads, [2])
                    self.assertEqual(torch.get_num_threads(), before)
            flush_mock.assert_called_with(True)
        finally:
            torch.set_num_threads(orig_threads)

        # GPU OOM halves batch 8 -> 4 instead of jumping directly to 1
        class _OomGpuModel:
            device = "cuda:0"
            input_layout = "NCHW"
            output_layout = "NCHW"

            def __init__(self) -> None:
                self.batch_sizes: list[int] = []

            def forward(self, x: np.ndarray) -> np.ndarray:
                bs = int(x.shape[0])
                self.batch_sizes.append(bs)
                if bs > 4:
                    raise RuntimeError("CUDA out of memory")
                return np.zeros((bs, 4, x.shape[2], x.shape[3]), dtype=np.float32)

        gpu_slices = [np.full((4, 4, 3), float(i), dtype=np.float32) for i in range(8)]
        for i, slc in enumerate(gpu_slices):
            slc[0, 0, 0] = float(i + 50)
        gpu_model = _OomGpuModel()
        with (
            patch("shared.hardware_adapt.query_gpu_free_total_bytes", return_value=(8 * 1024**3, 8 * 1024**3)),
            patch("shared.hardware_adapt.empty_gpu_cache_safely") as empty_mock,
        ):
            out = predict_view(gpu_model, gpu_slices, apply_softmax=False)
        self.assertEqual(out.shape, (4, 4, 8, 4))
        self.assertEqual(gpu_model.batch_sizes, [8, 4, 4])
        self.assertGreaterEqual(empty_mock.call_count, 2)

    def test_parcellation_forward_fused_logits_oom_retry_and_thread_policy_contract(self) -> None:
        if importlib.util.find_spec("torch") is None:
            self.skipTest("torch unavailable")
        import torch

        dd_pred_src = (REPO_ROOT / "anatomy/parcellation/DDSurfer_Pred.py").read_text(encoding="utf-8")
        # Ensure SLICERDSPACE_STAGE_CPUS no longer skips optimal_cpu_torch_threads or flush_denormal
        self.assertIn("enable_cpu_flush_denormal(torch)", dd_pred_src)
        self.assertNotIn('if device.type == "cpu" and not os.environ.get("SLICERDSPACE_STAGE_CPUS"):', dd_pred_src)
        self.assertIn("finally:", dd_pred_src)

        # Test _forward_fused_logits MPS OOM single-slice fallback (preserve sys.modules isolation)
        had_dd_pred = "anatomy.parcellation.DDSurfer_Pred" in sys.modules
        orig_dd_pred = sys.modules.get("anatomy.parcellation.DDSurfer_Pred")
        try:
            from anatomy.parcellation.DDSurfer_Pred import _forward_fused_logits
        except ImportError:
            self.skipTest("DDSurfer_Pred optional dependencies unavailable")

        seen_bs: list[int] = []

        class _FakeMpsNet:
            def __call__(self, batch):
                bs = int(batch.shape[0])
                seen_bs.append(bs)
                if bs > 1:
                    raise RuntimeError("MPS backend out of memory")
                return None, [torch.zeros((bs, 4, 4, 4), dtype=torch.float32)]

        sub_batch = torch.zeros((3, 7, 4, 4), dtype=torch.float32)
        orig_to = torch.Tensor.to

        def fake_to(self_t, *a, **kw):
            if a and isinstance(a[0], torch.device) and a[0].type == "mps":
                a = (torch.device("cpu"), *a[1:])
            return orig_to(self_t, *a, **kw)

        try:
            with (
                patch.object(torch.Tensor, "to", fake_to),
                patch("anatomy.parcellation.DDSurfer_Pred.empty_gpu_cache_safely") as empty_mock,
            ):
                res = _forward_fused_logits(_FakeMpsNet(), sub_batch, use_cuda=False, device=torch.device("mps"))
            self.assertEqual(tuple(res.shape), (3, 4, 4, 4))
            self.assertEqual(seen_bs, [3, 1, 1, 1])
            self.assertGreaterEqual(empty_mock.call_count, 2)
        finally:
            if not had_dd_pred:
                sys.modules.pop("anatomy.parcellation.DDSurfer_Pred", None)
                pkg = sys.modules.get("anatomy.parcellation")
                if pkg is not None and hasattr(pkg, "DDSurfer_Pred"):
                    delattr(pkg, "DDSurfer_Pred")
            elif orig_dd_pred is not None:
                sys.modules["anatomy.parcellation.DDSurfer_Pred"] = orig_dd_pred


if __name__ == "__main__":
    unittest.main()
