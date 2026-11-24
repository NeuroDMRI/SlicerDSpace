"""Round-3 regression tests for Group B:
1. MP-PCA low-RAM worker cap + CPU Gibbs single-buffer memory footprint & parity.
2. DDParcel float16 / tight-RAM memmap host probability buffer & exact argmax parity.
3. NRRD/NHDR preview available-RAM threshold & streaming mean-b0 decode (raw/gzip, axis-first/last).
4. MK-Curve sequential volume-major + bounded block-transpose staging (zero column thrashing).
5. TractCloud multi-GPU `cuda:<index>` preservation end-to-end from UI controller to bootstrap.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import tracemalloc
import types
import unittest
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_DIPY = importlib.util.find_spec("dipy") is not None
HAS_SCIPY = importlib.util.find_spec("scipy") is not None
HAS_NIBABEL = importlib.util.find_spec("nibabel") is not None
HAS_NRRD = importlib.util.find_spec("nrrd") is not None


class MppcaLowRamCapAndGibbsMemoryTests(unittest.TestCase):
    def test_mppca_low_ram_cap_applies_to_explicit_workers_and_pipeline(self) -> None:
        from preprocessing import dipy_accel as da, pipeline as preproc_pipeline
        from shared.hardware_adapt import GIB, MIB

        # Simulating 32 cores + 1 GiB available RAM must cap down below 4 even when explicit
        # thread count (16) is passed directly or via DwiPreprocessingConfig!
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SLICERDSPACE_STAGE_CPUS", None)
            with (
                mock.patch.object(da, "is_apple_silicon_host", return_value=False),
                mock.patch("os.cpu_count", return_value=32),
                mock.patch.object(da, "effective_cpu_threads", return_value=16),
                mock.patch.object(da, "query_host_available_ram_bytes", return_value=1 * GIB),
            ):
                self.assertEqual(da._resolve_dipy_cpu_workers(None, apply_low_ram_cap=True), 1)
                self.assertEqual(da._resolve_dipy_cpu_workers(16, apply_low_ram_cap=True), 1)
                self.assertEqual(da._resolve_dipy_cpu_workers(8, avail_ram_bytes=2 * GIB, apply_low_ram_cap=True), 2)
                self.assertEqual(da._resolve_dipy_cpu_workers(16, avail_ram_bytes=4 * GIB, apply_low_ram_cap=True), 4)
                self.assertEqual(da._resolve_dipy_cpu_workers(16, avail_ram_bytes=8 * GIB, apply_low_ram_cap=True), 6)
                # Explicit unsafe override remains available only when requested explicitly.
                self.assertEqual(
                    da._resolve_dipy_cpu_workers(
                        16, avail_ram_bytes=1 * GIB, apply_low_ram_cap=True, allow_unsafe_ram_override=True
                    ),
                    16,
                )
                # High-RAM 32-core server keeps full 16 workers (zero server regression).
                self.assertEqual(da._resolve_dipy_cpu_workers(None, avail_ram_bytes=64 * GIB, apply_low_ram_cap=True), 16)
                self.assertEqual(da._mppca_cpu_batch_bytes(1 * GIB), 32 * MIB)

                # Pipeline helper also respects available RAM for both default and explicit eddy_threads.
                cfg_default = preproc_pipeline.DwiPreprocessingConfig("sub", Path("."))
                cfg_explicit = preproc_pipeline.DwiPreprocessingConfig("sub", Path("."), eddy_threads=16)
                self.assertEqual(preproc_pipeline._resolved_dipy_workers(cfg_default), 1)
                self.assertEqual(preproc_pipeline._resolved_dipy_workers(cfg_explicit), 1)
                self.assertEqual(preproc_pipeline._resolved_dipy_workers(cfg_default, avail_ram_bytes=64 * GIB), 16)

    @unittest.skipUnless(HAS_SCIPY, "scipy unavailable")
    def test_mppca_cpu_does_not_spawn_thread_pool_under_1gib_even_with_explicit_16_workers(self) -> None:
        from preprocessing import dipy_accel as da
        from shared.hardware_adapt import GIB

        rng = np.random.default_rng(101)
        vol = (90.0 + 6.0 * rng.standard_normal((10, 10, 8, 12))).astype(np.float32)

        with (
            mock.patch.object(da, "query_host_available_ram_bytes", return_value=1 * GIB),
            mock.patch.object(da, "ThreadPoolExecutor", side_effect=AssertionError("must cap to 1 worker under 1 GiB")),
        ):
            out_low = da.mppca_accelerated(vol, patch_radius=1, device="cpu", max_workers=16)

        with mock.patch.object(da, "query_host_available_ram_bytes", return_value=64 * GIB):
            out_high = da.mppca_accelerated(vol, patch_radius=1, device="cpu", max_workers=2)

        np.testing.assert_allclose(out_low, out_high, rtol=1e-5, atol=1e-4)

    @unittest.skipUnless(HAS_SCIPY and HAS_DIPY, "scipy/dipy unavailable")
    def test_gibbs_cpu_numerical_parity_and_bounded_memory_footprint(self) -> None:
        from dipy.denoise.gibbs import gibbs_removal
        from preprocessing import dipy_accel as da

        rng = np.random.default_rng(20261124)

        # 1. Numerical parity across all slice axes, float32 & float64, inplace=False/True, 2D/3D/4D.
        vol4 = (100.0 + 12.0 * rng.standard_normal((12, 14, 10, 6))).astype(np.float32)
        for slice_axis in (0, 1, 2):
            ref = gibbs_removal(vol4.copy(), slice_axis=slice_axis, n_points=3, inplace=False)
            out_copy = da.gibbs_removal_accelerated(
                vol4, slice_axis=slice_axis, n_points=3, inplace=False, device="cpu", max_workers=2
            )
            vol_in = vol4.copy()
            out_inp = da.gibbs_removal_accelerated(
                vol_in, slice_axis=slice_axis, n_points=3, inplace=True, device="cpu", max_workers=2
            )
            self.assertIs(out_inp, vol_in)
            np.testing.assert_array_equal(out_copy, out_inp)
            np.testing.assert_allclose(out_copy, ref, rtol=5e-4, atol=5e-2)
            self.assertGreaterEqual(float(np.mean(np.isclose(out_copy, ref, rtol=1e-5, atol=1e-4))), 0.99)

        vol4_64 = vol4[..., :3].astype(np.float64)
        ref64 = gibbs_removal(vol4_64.copy(), slice_axis=2, n_points=3, inplace=False)
        out64 = da.gibbs_removal_accelerated(vol4_64, slice_axis=2, n_points=3, inplace=False, device="cpu", max_workers=2)
        np.testing.assert_allclose(out64, ref64, rtol=1e-12, atol=1e-12)

        vol3 = vol4[..., 0].copy()
        np.testing.assert_allclose(
            da.gibbs_removal_accelerated(vol3, slice_axis=1, n_points=3, device="cpu", max_workers=2),
            gibbs_removal(vol3.copy(), slice_axis=1, n_points=3, inplace=False),
            rtol=5e-4,
            atol=5e-2,
        )
        vol2 = vol4[:, :, 0, 0].copy()
        np.testing.assert_allclose(
            da.gibbs_removal_accelerated(vol2, n_points=3, device="cpu"),
            gibbs_removal(vol2.copy(), n_points=3, inplace=False),
            rtol=5e-4,
            atol=5e-2,
        )

        # 2. Verify peak heap allocation is ~1x output (+ small bounded FFT scratch) instead of ~5x full 4D copies.
        # (48, 48, 24, 32) float32 = 7.08 MiB per 4D copy; old path peaked at >4x (~30+ MiB).
        big = (80.0 + 8.0 * rng.standard_normal((48, 48, 24, 32))).astype(np.float32)
        vol_bytes = big.nbytes

        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            res_copy = da.gibbs_removal_accelerated(big, slice_axis=2, n_points=3, inplace=False, device="cpu", max_workers=1)
            _, peak_copy = tracemalloc.get_traced_memory()
            del res_copy
        finally:
            tracemalloc.stop()

        big_inp = big.copy()
        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            da.gibbs_removal_accelerated(big_inp, slice_axis=2, n_points=3, inplace=True, device="cpu", max_workers=1)
            _, peak_inplace = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        # Single output buffer + bounded 32-slice FFT scratch (~1.5 MiB) stays well under 2.2x vol_bytes.
        self.assertLess(peak_copy, int(vol_bytes * 2.2), f"peak_copy={peak_copy} vs vol_bytes={vol_bytes}")
        # In-place uses zero extra full 4D arrays: peak scratch is strictly < 1.0x vol_bytes!
        self.assertLess(peak_inplace, int(vol_bytes * 1.0), f"peak_inplace={peak_inplace} vs vol_bytes={vol_bytes}")


@unittest.skipUnless(HAS_TORCH, "torch unavailable")
class DDParcelHostProbabilityBufferTests(unittest.TestCase):
    def _import_ddsurfer_pred(self):
        had_dd_pred = "anatomy.parcellation.DDSurfer_Pred" in sys.modules
        orig_dd_pred = sys.modules.get("anatomy.parcellation.DDSurfer_Pred")
        pkg = sys.modules.get("anatomy.parcellation")
        had_attr = pkg is not None and hasattr(pkg, "DDSurfer_Pred")
        orig_attr = getattr(pkg, "DDSurfer_Pred", None) if had_attr else None

        def _restore() -> None:
            if not had_dd_pred:
                sys.modules.pop("anatomy.parcellation.DDSurfer_Pred", None)
            elif orig_dd_pred is not None:
                sys.modules["anatomy.parcellation.DDSurfer_Pred"] = orig_dd_pred
            cur_pkg = sys.modules.get("anatomy.parcellation")
            if cur_pkg is not None:
                if not had_attr and hasattr(cur_pkg, "DDSurfer_Pred"):
                    delattr(cur_pkg, "DDSurfer_Pred")
                elif had_attr:
                    setattr(cur_pkg, "DDSurfer_Pred", orig_attr)

        self.addCleanup(_restore)
        try:
            from anatomy.parcellation import DDSurfer_Pred as dd_pred
            return dd_pred
        except ImportError as exc:
            self.skipTest(f"DDSurfer_Pred optional dependencies unavailable: {exc}")

    def test_ddparcel_probability_buffer_uses_fp16_ram_or_memmap_under_tight_ram(self) -> None:
        import torch
        from shared.hardware_adapt import GIB

        dd_pred = self._import_ddsurfer_pred()
        shape = (1, 32, 32, 32, 82)
        fp32_bytes = int(np.prod(shape)) * 4
        fp16_bytes = int(np.prod(shape)) * 2

        # 1. Roomy host (>= 8 GiB): in-RAM float16 cuts host RAM by exactly 50%.
        buf_ram = dd_pred._allocate_prediction_probability(shape, avail_ram_bytes=32 * GIB)
        try:
            self.assertEqual(buf_ram.mode, "ram_fp16")
            self.assertEqual(buf_ram.tensor.dtype, torch.float16)
            self.assertEqual(buf_ram.nbytes, fp16_bytes)
            self.assertEqual(buf_ram.nbytes * 2, fp32_bytes)
        finally:
            buf_ram.close()

        # 2. Tight host RAM (< 8 GiB): automatically switches to disk-backed float16 memmap & cleans up cleanly.
        with tempfile.TemporaryDirectory() as td:
            buf_mm = dd_pred._allocate_prediction_probability(shape, avail_ram_bytes=4 * GIB, temp_root=td)
            backing_dir = buf_mm._backing_dir
            self.assertEqual(buf_mm.mode, "memmap_fp16")
            self.assertEqual(buf_mm.tensor.dtype, torch.float16)
            self.assertEqual(buf_mm.nbytes, fp16_bytes)
            self.assertIsNotNone(backing_dir)
            self.assertTrue(Path(backing_dir).is_dir())
            buf_mm.close()
            self.assertFalse(Path(backing_dir).exists())

    def test_ddparcel_multiview_fp16_and_memmap_match_fp32_argmax_exactly(self) -> None:
        import torch
        from shared.hardware_adapt import GIB

        dd_pred = self._import_ddsurfer_pred()
        rng = np.random.default_rng(77)
        nx, ny, nz, nc = 16, 16, 16, 82

        # Synthetic multi-view CNN logits with realistic class margin (winning class ~+6 over background noise).
        ax_logits = rng.normal(0.0, 1.0, size=(nx, ny, nz, nc)).astype(np.float32)
        cor_logits = rng.normal(0.0, 1.0, size=(nx, ny, nz, nc)).astype(np.float32)
        sag_logits = rng.normal(0.0, 1.0, size=(nx, ny, nz, nc)).astype(np.float32)
        true_cls = rng.integers(0, nc, size=(nx, ny, nz))
        ix, iy, iz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
        ax_logits[ix, iy, iz, true_cls] += 6.0
        cor_logits[ix, iy, iz, true_cls] += 5.5
        sag_logits[ix, iy, iz, true_cls] += 5.0

        # Exact float32 reference aggregation: 0.4 * axial + 0.4 * coronal + 0.2 * sagittal
        ref_prob = (
            torch.from_numpy(ax_logits) * 0.4
            + torch.from_numpy(cor_logits) * 0.4
            + torch.from_numpy(sag_logits) * 0.2
        )
        _, ref_labels = torch.max(ref_prob, 3)
        ref_labels_np = ref_labels.numpy().astype(np.int16)

        with tempfile.TemporaryDirectory() as td:
            for avail_ram, expected_mode in ((32 * GIB, "ram_fp16"), (4 * GIB, "memmap_fp16")):
                buf = dd_pred._allocate_prediction_probability(
                    (1, nx, ny, nz, nc), avail_ram_bytes=avail_ram, temp_root=td
                )
                try:
                    self.assertEqual(buf.mode, expected_mode)
                    prob = buf.tensor
                    # Axial pass (overwrite=True) in batches of 4 along axis 1
                    for y0 in range(0, ny, 4):
                        y1 = min(ny, y0 + 4)
                        dd_pred._accumulate_prob_slice(
                            prob[0, :, y0:y1, :, :],
                            torch.from_numpy(ax_logits[:, y0:y1, :, :]) * 0.4,
                            overwrite=True,
                        )
                    # Coronal pass in batches of 4 along axis 2
                    for z0 in range(0, nz, 4):
                        z1 = min(nz, z0 + 4)
                        dd_pred._accumulate_prob_slice(
                            prob[0, :, :, z0:z1, :],
                            torch.from_numpy(cor_logits[:, :, z0:z1, :]) * 0.4,
                        )
                    # Sagittal pass with fused FP32 argmax in batches of 4 along axis 0
                    fused_labels = np.empty((nx, ny, nz), dtype=np.int16)
                    for x0 in range(0, nx, 4):
                        x1 = min(nx, x0 + 4)
                        combined = prob[0, x0:x1, :, :, :].float() + torch.from_numpy(sag_logits[x0:x1, :, :, :]) * 0.2
                        _, idx = torch.max(combined, 3)
                        fused_labels[x0:x1] = idx.numpy()

                    np.testing.assert_array_equal(fused_labels, ref_labels_np)

                    # Also verify chunked _argmax_prediction_probability when Sagittal writes back into buffer.
                    for x0 in range(0, nx, 4):
                        x1 = min(nx, x0 + 4)
                        dd_pred._accumulate_prob_slice(
                            prob[0, x0:x1, :, :, :],
                            torch.from_numpy(sag_logits[x0:x1, :, :, :]) * 0.2,
                        )
                    chunked_labels = dd_pred._argmax_prediction_probability(prob, chunk_slices=5)
                    np.testing.assert_array_equal(chunked_labels, ref_labels_np)
                finally:
                    buf.close()

        # Verify extreme logit clamping avoids FP16 inf/nan overflow.
        dst = torch.zeros((2, 2, 2, 4), dtype=torch.float16)
        huge = torch.full((2, 2, 2, 4), 1e6, dtype=torch.float32)
        dd_pred._accumulate_prob_slice(dst, huge, overwrite=True)
        self.assertTrue(torch.isfinite(dst).all())


@unittest.skipUnless(HAS_NIBABEL and HAS_NRRD, "nibabel/pynrrd unavailable")
class NrrdPreviewAvailableRamAndStreamingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.cache = self.root / "cache"
        self.addCleanup(self._tmp.cleanup)

    def _write_test_nrrd(
        self,
        *,
        name: str,
        volumes: np.ndarray,
        axis_last: bool,
        encoding: str,
        detached: bool,
        space: str = "left-posterior-superior",
        endian: str = "little",
    ) -> Path:
        import nrrd

        # volumes is (X, Y, Z, G)
        data = volumes if axis_last else np.moveaxis(volumes, -1, 0)
        if endian == "big":
            data = data.astype(data.dtype.newbyteorder(">"))
        else:
            data = data.astype(data.dtype.newbyteorder("<"))
        spatial = [[2.0, 0.0, 0.0], [0.0, 2.5, 0.0], [0.0, 0.0, 3.0]]
        none = [float("nan")] * 3
        directions = spatial + [none] if axis_last else [none] + spatial
        kinds = ["domain", "domain", "domain", "list"] if axis_last else ["list"] + ["domain", "domain", "domain"]
        ng = volumes.shape[-1]
        header = {
            "space": space,
            "space directions": np.array(directions),
            "kinds": kinds,
            "space origin": np.array([10.0, -5.0, 7.5]),
            "DWMRI_b-value": "1000",
            "encoding": encoding,
        }
        # Put b0 volumes at indices 0, ng//2, ng-1
        b0_set = {0, ng // 2, ng - 1}
        for idx in range(ng):
            header["DWMRI_gradient_%04d" % idx] = "0 0 0" if idx in b0_set else "1 0 0"
        ext = ".nhdr" if detached else ".nrrd"
        path = self.root / f"{name}{ext}"
        nrrd.write(str(path), data, header=header, detached_header=detached)
        return path

    def test_full_dwi_limit_uses_available_ram_not_only_installed_ram(self) -> None:
        from shared import input_preview as ip

        # 64 GiB installed RAM, but Slicer scene already occupies 62 GiB -> only 2 GiB available!
        with (
            mock.patch.object(ip, "physical_memory_bytes", return_value=64 << 30),
            mock.patch.object(ip, "available_memory_bytes", return_value=2 << 30),
        ):
            self.assertEqual(ip.full_dwi_limit_bytes(), 512 << 20)

        # Roomy host (64 GiB installed, 32 GiB available) retains 4 GiB cap.
        with (
            mock.patch.object(ip, "physical_memory_bytes", return_value=64 << 30),
            mock.patch.object(ip, "available_memory_bytes", return_value=32 << 30),
        ):
            self.assertEqual(ip.full_dwi_limit_bytes(), 4 << 30)

        # Explicit numeric/None overrides still keep exact API behavior.
        self.assertEqual(ip.full_dwi_limit_bytes(8 << 30), 2 << 30)
        self.assertEqual(ip.full_dwi_limit_bytes(None), 4 << 30)

    def test_nrrd_and_nhdr_streaming_mean_b0_matches_full_decode_without_calling_nrrd_read(self) -> None:
        import nibabel as nib
        import nrrd
        from shared import input_preview as ip

        rng = np.random.default_rng(42)
        volumes = rng.integers(10, 4000, size=(7, 8, 6, 9), dtype=np.int16)
        expected_mean = (
            volumes[..., 0].astype(np.float32)
            + volumes[..., 4].astype(np.float32)
            + volumes[..., 8].astype(np.float32)
        ) / np.float32(3)
        expected_affine = np.array(
            [[-2.0, 0.0, 0.0, -10.0], [0.0, -2.5, 0.0, 5.0], [0.0, 0.0, 3.0, 7.5], [0.0, 0.0, 0.0, 1.0]]
        )

        cases = [
            (False, "raw", True, "little"),
            (False, "raw", False, "little"),
            (False, "gzip", True, "little"),
            (False, "gzip", False, "little"),
            (True, "raw", True, "little"),
            (True, "raw", False, "little"),
            (True, "gzip", True, "little"),
            (True, "gzip", False, "little"),
            (False, "raw", True, "big"),
            (False, "gzip", False, "big"),
        ]
        for idx, (detached, encoding, axis_last, endian) in enumerate(cases):
            with self.subTest(detached=detached, encoding=encoding, axis_last=axis_last, endian=endian):
                path = self._write_test_nrrd(
                    name=f"case_{idx}",
                    volumes=volumes,
                    axis_last=axis_last,
                    encoding=encoding,
                    detached=detached,
                    endian=endian,
                )
                source = ip.PreviewSource("nrrd", path, None, None, f"sub-{idx}")
                with mock.patch.object(nrrd, "read", side_effect=AssertionError("must stream b0 without full nrrd.read")):
                    out_path = ip.mean_b0_image(source, self.cache)
                img = nib.load(str(out_path))
                self.assertEqual(img.shape, (7, 8, 6))
                np.testing.assert_allclose(np.asarray(img.dataobj), expected_mean, rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(img.affine, expected_affine)

    def test_nrrd_streaming_mean_b0_never_allocates_full_4d_array_and_handles_large_total_nrrd(self) -> None:
        import nrrd
        from shared import input_preview as ip

        rng = np.random.default_rng(99)
        # 40 volumes of (40, 40, 24) float32 -> full 4D array = 6.14 MiB, single 3D volume = 153.6 KiB.
        volumes = rng.uniform(10.0, 500.0, size=(40, 40, 24, 40)).astype(np.float32)
        full_4d_bytes = volumes.nbytes
        expected_mean = (
            volumes[..., 0] + volumes[..., 20] + volumes[..., 39]
        ) / np.float32(3)

        for axis_last in (True, False):
            for encoding in ("raw", "gzip"):
                with self.subTest(axis_last=axis_last, encoding=encoding):
                    path = self._write_test_nrrd(
                        name=f"mem_{axis_last}_{encoding}",
                        volumes=volumes,
                        axis_last=axis_last,
                        encoding=encoding,
                        detached=False,
                    )
                    source = ip.PreviewSource("nrrd", path, None, None, f"mem-{axis_last}-{encoding}")
                    # Also cap stream chunk to 512 KiB and MAX_NRRD_BYTES to 1 MiB (< 6.14 MiB full 4D,
                    # > 153.6 KiB single 3D volume) to verify streaming works on multi-GiB 4D NRRDs!
                    with (
                        mock.patch.object(ip, "_NRRD_STREAM_CHUNK_BYTES", 512 << 10),
                        mock.patch.object(ip, "MAX_NRRD_BYTES", 1 << 20),
                        mock.patch.object(nrrd, "read", side_effect=AssertionError("must not call nrrd.read")),
                    ):
                        tracemalloc.start()
                        try:
                            tracemalloc.reset_peak()
                            img, count = ip._mean_b0_nrrd(source, ip.B0_THRESHOLD)
                            _, peak = tracemalloc.get_traced_memory()
                        finally:
                            tracemalloc.stop()
                    self.assertEqual(count, 3)
                    np.testing.assert_allclose(np.asarray(img.dataobj), expected_mean, rtol=1e-5, atol=1e-4)
                    # Peak allocation stays a small fraction of the 6.14 MiB full 4D array!
                    self.assertLess(peak, int(full_4d_bytes * 0.45), f"peak={peak} vs full_4d={full_4d_bytes}")


@unittest.skipUnless(HAS_NIBABEL, "nibabel unavailable")
class MkCurveSequentialBlockTransposeTests(unittest.TestCase):
    def test_mkcurve_staging_writes_sequential_rows_never_column_by_column_and_is_bit_exact(self) -> None:
        import nibabel as nib
        from microstructure import mkcurve, mkcurve_inputs
        from Testing.Python.test_microstructure_dki import synthetic_case, write_case

        case = synthetic_case(noisy=True)
        case["data"][1, 1, 1, 0] = 0.0
        case["data"][0, 1, 1, 5] = -3.0
        case["data"][1, 0, 1, 7] = np.nan

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = write_case(root, case)
            inputs = [paths[k] for k in ("dwi", "bval", "bvec", "mask")]

            # Reference two-step staging + 26-neighbor repair
            ref_dir = root / "ref"
            ref_dir.mkdir()
            prepared, _, _ = mkcurve_inputs._prepare_inputs(*inputs, ref_dir, 50.0, 3000.0, allow_nonfinite=True)
            img = nib.load(str(prepared[0]), mmap="c")
            data = img.dataobj.get_unscaled()
            mask = nib.load(str(prepared[3])).get_fdata() > 0
            rep_ref = mkcurve.repair_nonpositive(data, mask)
            coords_ref = np.unravel_index(np.flatnonzero(mask.ravel(order="F")), mask.shape, order="F")
            ref_masked = np.column_stack([data[..., v][coords_ref] for v in range(data.shape[-1])])
            if isinstance(data, np.memmap):
                data._mmap.close()
            for holder in img.file_map.values():
                if getattr(holder, "fileobj", None) is not None:
                    holder.fileobj.close()

            # Force disk-backed volume-major memmap + multi-block transpose by tightening thresholds!
            str_dir = root / "stream_disk"
            str_dir.mkdir()
            real_open_memmap = np.lib.format.open_memmap
            write_patterns: dict[str, list[object]] = {}

            class _WriteSpyMemmap(np.memmap):
                def __setitem__(self, key, value):
                    tag = getattr(self, "_spy_tag", "unknown")
                    write_patterns.setdefault(tag, []).append(key)
                    return super().__setitem__(key, value)

            def _spy_open_memmap(filename, *args, **kwargs):
                mm = real_open_memmap(filename, *args, **kwargs)
                wrapped = mm.view(_WriteSpyMemmap)
                wrapped._spy_tag = Path(filename).name
                return wrapped

            with (
                mock.patch.object(mkcurve_inputs, "_IN_RAM_STAGE_MAX_BYTES", 1),
                mock.patch.object(mkcurve_inputs, "_TRANSPOSE_BLOCK_BYTES", 2048),
                mock.patch.object(np.lib.format, "open_memmap", side_effect=_spy_open_memmap),
            ):
                masked, mask_s, coords_s, bvals_s, bvecs_s, _ref, _prep, rep_s = mkcurve_inputs._prepare_masked_signals(
                    *inputs, str_dir, 50.0, 3000.0, allow_nonfinite=True
                )
            try:
                # 1. Bit-exact numerical parity + exact repair counts + C-contiguous memmap contract.
                self.assertIsInstance(masked, np.memmap)
                self.assertTrue(masked.flags["C_CONTIGUOUS"])
                np.testing.assert_array_equal(np.asarray(masked), ref_masked)
                self.assertEqual(rep_s["repaired_measurements"], rep_ref["repaired_measurements"])
                self.assertEqual(rep_s["unresolved_measurements"], rep_ref["unresolved_measurements"])
                np.testing.assert_array_equal(mask_s, mask)
                self.assertEqual(len(bvals_s), ref_masked.shape[1])
                self.assertEqual(bvecs_s.shape, (ref_masked.shape[1], 3))

                # 2. Scratch volume-major file was unlinked immediately after block transpose.
                self.assertFalse((str_dir / "_masked_signals_vol_major.npy").exists())

                # 3. Verify ZERO column-by-column writes occurred on masked_signals.npy!
                final_writes = write_patterns.get("masked_signals.npy", [])
                self.assertGreater(len(final_writes), 1)  # multiple bounded row blocks exercised
                covered_rows = []
                for key in final_writes:
                    self.assertIsInstance(key, tuple)
                    self.assertEqual(len(key), 2)
                    row_sel, col_sel = key
                    # Row selector MUST be a bounded contiguous slice, col selector MUST be full slice(:)!
                    self.assertIsInstance(row_sel, slice)
                    self.assertEqual(col_sel, slice(None))
                    self.assertIsNotNone(row_sel.start)
                    self.assertIsNotNone(row_sel.stop)
                    covered_rows.append((row_sel.start, row_sel.stop))
                # Strict ascending contiguous non-overlapping coverage of [0, count)
                self.assertEqual(covered_rows[0][0], 0)
                self.assertEqual(covered_rows[-1][1], ref_masked.shape[0])
                for (_, prev_stop), (next_start, _) in zip(covered_rows, covered_rows[1:]):
                    self.assertEqual(prev_stop, next_start)

                # 4. Volume-major scratch writes were strictly full contiguous rows `(vol_idx, :)`.
                vol_writes = write_patterns.get("_masked_signals_vol_major.npy", [])
                self.assertEqual(len(vol_writes), ref_masked.shape[1])
                for expected_v, key in enumerate(vol_writes):
                    self.assertEqual(key, (expected_v, slice(None)))
            finally:
                masked._mmap.close()


class TractCloudMultiGpuDeviceTests(unittest.TestCase):
    def test_ui_run_controller_and_cli_preserve_cuda_index_for_tractcloud(self) -> None:
        from shared.run_subject_pipeline_cli import build_parser
        from shared.ui_run_controller import (
            RunSelection,
            build_background_cli_args,
            build_batch_background_cli_args,
            build_batch_run_kwargs,
            build_single_subject_run_kwargs,
        )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sel = RunSelection(
                subject_root_text=str(root),
                subject_id="sub-01",
                output_root_text=str(root / "out"),
                run_name="Test1",
                selected_targets=("bundles",),
                requested_targets=("bundles",),
                tractography_backend="csd_act",
                bundle_backend="tractcloud",
                microstructure_freewater=False,
                microstructure_noddi=False,
                select_streamlines=10000,
                connectome_backend="auto",
                connectome_compress_mrtrix_dense=None,
                ddseg_device="cuda:1",
                surface_device="cuda:1",
            )
            single = build_single_subject_run_kwargs(sel, {}, preflight_blocked_targets={})
            batch = build_batch_run_kwargs(sel, batch_root=root)
            self.assertEqual(single["tractcloud_device"], "cuda:1")
            self.assertEqual(batch["tractcloud_device"], "cuda:1")

            single_cli = build_background_cli_args(single, {}, repo_root=REPO_ROOT)
            batch_cli = build_batch_background_cli_args(
                batch, {}, repo_root=REPO_ROOT, batch_root=root, batch_size=1, arrange_inputs="none"
            )
            self.assertEqual(single_cli[single_cli.index("--tractcloud-device") + 1], "cuda:1")
            self.assertEqual(batch_cli[batch_cli.index("--tractcloud-device") + 1], "cuda:1")

            parser = build_parser()
            self.assertEqual(parser.parse_args(["--tractcloud-device", "cuda:1"]).tractcloud_device, "cuda:1")
            self.assertEqual(parser.parse_args(["--tractcloud-device", "cuda:0"]).tractcloud_device, "cuda:0")
            self.assertEqual(parser.parse_args(["--tractcloud-device", "cuda"]).tractcloud_device, "cuda")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(["--tractcloud-device", "tpu:0"])

    def test_tractcloud_runner_and_bootstrap_honor_cuda_1_on_multi_gpu_host(self) -> None:
        from tractography.tractcloud import (
            TRACTCLOUD_COMPAT_BOOTSTRAP,
            TractCloudConfig,
            _tractcloud_env,
            build_tractcloud_command,
            is_valid_tractcloud_device,
            validate_tractcloud_inputs,
        )

        self.assertTrue(is_valid_tractcloud_device("cuda:1"))
        self.assertTrue(is_valid_tractcloud_device("cuda:0"))
        self.assertTrue(is_valid_tractcloud_device("cuda"))
        self.assertTrue(is_valid_tractcloud_device("mps"))
        self.assertTrue(is_valid_tractcloud_device("cpu"))
        self.assertTrue(is_valid_tractcloud_device("auto"))
        self.assertFalse(is_valid_tractcloud_device("cuda:-1"))
        self.assertFalse(is_valid_tractcloud_device("cuda:abc"))
        self.assertFalse(is_valid_tractcloud_device("rocm:0"))

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            vtp = root / "tracks.vtp"
            vtp.write_text("<VTKFile/>", encoding="utf-8")
            py_exe = Path(sys.executable)
            cfg = TractCloudConfig(
                subject_id="sub-01",
                input_tractography=vtp,
                output_dir=root / "out",
                python_executable=py_exe,
                device="cuda:1",
                execute=False,
            )
            with mock.patch("tractography.tractcloud.inspect_tractcloud_backend", return_value={"available": True, "packages": {"tractcloud": True, "torch": True}}):
                validate_tractcloud_inputs(cfg)
            cmd = build_tractcloud_command(cfg)
            self.assertEqual(cmd[cmd.index("--device") + 1], "cuda:1")

            with mock.patch.object(sys, "platform", "linux"):
                env = _tractcloud_env(cfg)
                self.assertNotIn("OMP_PROC_BIND", env)
                self.assertNotIn("OMP_PLACES", env)

            # Execute TRACTCLOUD_COMPAT_BOOTSTRAP with a mocked 2-GPU PyTorch targeting cuda:1!
            init_devices: list[object] = []
            package = types.ModuleType("tractcloud")
            pipeline = types.ModuleType("tractcloud.pipeline")

            class FakePipeline:
                def __init__(self, reporter=None, device=None, *args, **kwargs):
                    init_devices.append(device)

            pipeline.TractCloudPipeline = FakePipeline
            pipeline.run_inference = mock.Mock(return_value="ok")
            package.pipeline = pipeline

            fake_torch = mock.Mock()
            fake_torch.__version__ = "2.5.1+cu124"
            fake_torch.cuda.is_available.return_value = True
            fake_torch.cuda.device_count.return_value = 2
            # GPU 0 has only 1 GiB free; GPU 1 has 24 GiB free -> mem_get_info(1) must keep batch-size 2048!
            fake_torch.cuda.mem_get_info.side_effect = lambda idx: (
                (1 * 1024**3, 8 * 1024**3) if idx == 0 else (24 * 1024**3, 24 * 1024**3)
            )
            fake_torch.backends.mps.is_available.return_value = False

            record_path = root / "tractcloud_device.json"
            argv = ["tractcloud", "--device", "cuda:1", "--batch-size", "2048"]

            def _fake_run_module(mod_name, run_name=None):
                # Upstream tractcloud.cli converts `--device cuda` -> `device="cuda:0"` before constructing pipeline!
                self.assertEqual(argv[argv.index("--device") + 1], "cuda")
                pipeline.TractCloudPipeline(device="cuda:0")

            with (
                mock.patch.dict(sys.modules, {"tractcloud": package, "tractcloud.pipeline": pipeline, "torch": fake_torch}),
                mock.patch("runpy.run_module", side_effect=_fake_run_module),
                mock.patch.object(sys, "argv", argv),
                mock.patch.dict(os.environ, {"SLICERDSPACE_TRACTCLOUD_DEVICE_RECORD": str(record_path)}),
            ):
                exec(TRACTCLOUD_COMPAT_BOOTSTRAP, {})
                # Verify strict GPU guard remains active on cuda:1 (refuses silent CPU fallback).
                self.assertEqual(pipeline.run_inference(None, None, None, 3, "cuda:1"), "ok")
                with self.assertRaisesRegex(RuntimeError, "automatic CPU retry disabled"):
                    pipeline.run_inference(None, None, None, 3, "cpu")

            fake_torch.cuda.set_device.assert_called_once_with(1)
            fake_torch.empty.assert_called_once_with(1, device="cuda:1")
            fake_torch.cuda.mem_get_info.assert_called_once_with(1)
            self.assertEqual(argv[argv.index("--batch-size") + 1], "2048")
            self.assertEqual(init_devices, ["cuda:1"])
            rec = json.loads(record_path.read_text(encoding="utf-8"))
            self.assertEqual(rec["requested"], "cuda:1")
            self.assertEqual(rec["resolved"], "cuda:1")

            # Out-of-range GPU index (cuda:3 on 2-GPU host) must raise a clear RuntimeError.
            argv_bad = ["tractcloud", "--device", "cuda:3", "--batch-size", "2048"]
            with (
                mock.patch.dict(sys.modules, {"tractcloud": package, "tractcloud.pipeline": pipeline, "torch": fake_torch}),
                mock.patch("runpy.run_module"),
                mock.patch.object(sys, "argv", argv_bad),
                mock.patch.dict(os.environ, {"SLICERDSPACE_TRACTCLOUD_DEVICE_RECORD": ""}),
            ):
                with self.assertRaisesRegex(RuntimeError, "only 2 CUDA device"):
                    exec(TRACTCLOUD_COMPAT_BOOTSTRAP, {})


if __name__ == "__main__":
    unittest.main()
