#!/usr/bin/env python3
"""Tests for shared/teem_dti.py: the parts that run without Slicer, plus opt-in Slicer checks (D3, D10, D12)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import teem_dti

# Element list of SlicerDMRI DiffusionTensorScalarMeasurements.xml (revision a40200b).
CLI_SCALAR_ENUMERATION = (
    "FractionalAnisotropy", "Trace", "Determinant", "MeanDiffusivity", "RelativeAnisotropy", "Mode",
    "LinearMeasure", "PlanarMeasure", "SphericalMeasure", "MinEigenvalue", "MidEigenvalue", "MaxEigenvalue",
    "ParallelDiffusivity", "PerpendicularDiffusivity",
)
SLICER_5_12_PYTHON = Path("/Applications/Slicer.app/Contents/bin/PythonSlicer")


class ScalarOperationTests(unittest.TestCase):
    def test_every_cli_enumeration_name_is_supported(self) -> None:
        for name in CLI_SCALAR_ENUMERATION:
            self.assertIn(name, teem_dti.SCALAR_OPERATIONS, name)

    def test_pipeline_scalar_names_map_to_vtk_operations(self) -> None:
        self.assertEqual(teem_dti.SCALAR_OPERATIONS["MidEigenvalue"], "MiddleEigenvalue")
        for name in ("FractionalAnisotropy", "Trace", "MeanDiffusivity", "MinEigenvalue", "MaxEigenvalue"):
            self.assertEqual(teem_dti.SCALAR_OPERATIONS[name], name)


class CastTests(unittest.TestCase):
    def test_float_target_is_a_plain_cast(self) -> None:
        result, bad = teem_dti._cast_like_c(np.array([1.5, -2.25]), np.float32)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(bad, 0)
        np.testing.assert_array_equal(result, [1.5, -2.25])

    def test_integer_target_truncates_toward_zero_and_counts_out_of_range(self) -> None:
        result, bad = teem_dti._cast_like_c(np.array([1.9, -1.9, 70000.0, np.nan]), np.int16)
        np.testing.assert_array_equal(result, [1, -1, 0, 0])
        self.assertEqual(bad, 2)


class ThreadTests(unittest.TestCase):
    def test_stage_cpus_variable_sets_the_thread_count(self) -> None:
        with mock.patch.dict(os.environ, {"SLICERDSPACE_STAGE_CPUS": "3"}):
            self.assertEqual(teem_dti.default_threads(), 3)

    def test_invalid_stage_cpus_falls_back_to_process_affinity(self) -> None:
        for value in ("", "0", "-2", "many"):
            with self.subTest(value=value), mock.patch.dict(os.environ, {"SLICERDSPACE_STAGE_CPUS": value}):
                self.assertGreaterEqual(teem_dti.default_threads(), 1)

    def test_slab_plan_covers_every_voxel_once(self) -> None:
        for voxels, threads, slab in ((10, 1, None), (1000, 4, None), (1000003, 8, None), (7, 2, 1), (5, 3, 100)):
            with self.subTest(voxels=voxels, threads=threads, slab=slab):
                bounds = teem_dti.plan_slabs(voxels, threads, slab)
                self.assertEqual(bounds[0][0], 0)
                self.assertEqual(bounds[-1][1], voxels)
                for (_, stop), (start, _) in zip(bounds, bounds[1:]):
                    self.assertEqual(stop, start)
                self.assertTrue(all(stop > start for start, stop in bounds))

    def test_default_slab_is_capped_and_aims_for_four_slabs_per_thread(self) -> None:
        self.assertEqual(teem_dti.plan_slabs(1000, 4)[0], (0, 63))
        self.assertEqual(teem_dti.plan_slabs(10**7, 1)[0], (0, 65536))


class CommandBuilderTests(unittest.TestCase):
    def test_estimate_command_shape(self) -> None:
        command = teem_dti.estimate_command(
            "/slicer/bin/PythonSlicer", "dwi.nhdr", "dti.nhdr", "b0.nhdr", "WLS",
            [("FractionalAnisotropy", "fa.nhdr"), ("Trace", "trace.nhdr")], receipt="receipt.json")
        self.assertEqual(command[0], "/slicer/bin/PythonSlicer")
        self.assertEqual(Path(command[1]), Path(teem_dti.__file__).resolve())
        self.assertEqual(command[2:6], ["estimate", "--enumeration", "WLS", "--receipt"])
        self.assertIn("FractionalAnisotropy=fa.nhdr", command)
        self.assertEqual(command[-3:], ["dwi.nhdr", "dti.nhdr", "b0.nhdr"])
        self.assertNotIn("--threads", command)

    def test_the_command_parses_with_the_script_parser(self) -> None:
        # F4: the receipt goes to a temporary folder, not the working directory.
        # T5: run in an empty working directory, so an old r.json elsewhere cannot fail the check.
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as work:
            receipt = Path(tmp) / "r.json"
            command = teem_dti.estimate_command("py", "dwi.nhdr", "dti.nhdr", "b0.nhdr", "LS",
                                                [("Trace", "t.nhdr")], receipt=str(receipt))
            os.chdir(work)
            try:
                with mock.patch.object(teem_dti, "estimate_file", return_value={}) as estimate, \
                        mock.patch.object(teem_dti, "scalar_files", return_value={}) as scalars:
                    self.assertEqual(teem_dti.main(command[2:]), 0)
                self.assertTrue(receipt.is_file())
                self.assertFalse((Path.cwd() / "r.json").exists(),
                                 "the parser test must not write into the working directory")
            finally:
                os.chdir(previous)
        self.assertEqual(estimate.call_args.args[:4], ("dwi.nhdr", "dti.nhdr", "b0.nhdr", "LS"))
        self.assertEqual(scalars.call_args.args[1], [("Trace", "t.nhdr")])


class LibraryDiscoveryTests(unittest.TestCase):
    def test_d12_missing_library_error_lists_roots_and_file_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            empty_root = Path(tmp) / "FakeSlicer"
            empty_root.mkdir()
            env = {key: value for key, value in os.environ.items() if key != "SLICERDSPACE_TEEM_LIBRARY"}
            with mock.patch.dict(os.environ, env, clear=True), \
                    mock.patch.object(teem_dti, "_LIBRARY", None), \
                    mock.patch.object(teem_dti, "_slicer_roots", return_value=[empty_root]), \
                    mock.patch.object(teem_dti.ctypes, "CDLL", side_effect=OSError("not found")):
                with self.assertRaises(teem_dti.TeemError) as caught:
                    teem_dti.load_teem()
            message = str(caught.exception)
            self.assertIn(str(empty_root), message)
            for name in teem_dti._library_file_names():
                self.assertIn(name, message)

    def test_windows_library_name_is_left_as_is(self) -> None:
        """D12: the Windows name stays unverified; this records it so a change is deliberate."""
        with mock.patch.object(teem_dti.sys, "platform", "win32"):
            self.assertEqual(teem_dti._library_file_names(), ["teem.dll", "libteem.dll"])


class TeemLibraryTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            self.lib = teem_dti.load_teem()
        except teem_dti.TeemError as error:
            self.skipTest(f"Slicer's Teem library is not available here: {error}")

    def test_self_check_recovers_a_known_tensor(self) -> None:
        report = teem_dti.self_check()
        self.assertEqual(report["status"], "ok")


def _usable_slicer_python() -> Path | None:
    from shared.external_process import _default_slicer_python

    active = _default_slicer_python()
    if active is not None and active.is_file():
        return active
    real = SLICER_5_12_PYTHON.parent / "python-real"
    if SLICER_5_12_PYTHON.is_file() and real.is_file() and os.access(real, os.X_OK):
        return SLICER_5_12_PYTHON
    return None


class SlicerPythonSelfCheckTests(unittest.TestCase):
    """D10: opt-in run of the script with a local Slicer's own Python (skipped when Slicer is absent)."""

    @unittest.skipUnless(_usable_slicer_python() is not None, "no usable local Slicer PythonSlicer")
    def test_selfcheck_command_passes_in_slicer_python(self) -> None:
        slicer_python = _usable_slicer_python()
        assert slicer_python is not None
        completed = subprocess.run(
            [str(slicer_python), str(REPO_ROOT / "shared/teem_dti.py"), "selfcheck"],
            capture_output=True, text=True, timeout=300)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["status"], "ok")


class NiftiBackedNhdrParityTests(unittest.TestCase):
    """Verify streaming NIfTI-backed .nhdr reader matches vtkTeemNRRDReader bit-for-bit and rejects all-zero reads."""

    def setUp(self) -> None:
        try:
            self.vtk, self.vtkTeem, self.ns, _ = teem_dti._vtk_modules()
            teem_dti.load_teem()
            import nibabel  # noqa: F401
        except Exception as error:
            self.skipTest(f"Slicer vtkTeem / nibabel unavailable in this interpreter: {error}")

    def test_streamed_nifti_nhdr_matches_vtk_teem_estimate_file_bit_for_bit(self) -> None:
        import nibabel as nib
        from anatomy.parcellation.conversion.nhdr_write import nhdr_write

        rng = np.random.default_rng(20261112)
        nx, ny, nz, ng = 5, 6, 7, 10
        affine = np.array([
            [-1.25, 0.0, 0.0, 90.0],
            [0.0, 1.25, 0.0, -126.0],
            [0.0, 0.0, 1.25, -72.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        data = rng.uniform(80.0, 1200.0, size=(nx, ny, nz, ng)).astype(np.float32)
        dirs = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
            [0.70710678, 0.70710678, 0.0], [0.70710678, 0.0, 0.70710678], [0.0, 0.70710678, 0.70710678],
            [0.70710678, -0.70710678, 0.0], [0.70710678, 0.0, -0.70710678], [0.0, 0.70710678, -0.70710678],
        ])
        bvals = [0] + [1000] * (ng - 1)

        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            nii_path = tdir / "dwi.nii.gz"
            bval_path = tdir / "dwi.bval"
            bvec_path = tdir / "dwi.bvec"
            nhdr_path = tdir / "DWI.nhdr"
            img = nib.Nifti1Image(data, affine)
            img.header["scl_slope"] = 1.0
            img.header["scl_inter"] = 0.0
            nib.save(img, str(nii_path))
            bval_path.write_text(" ".join(map(str, bvals)) + "\n", encoding="utf-8")
            bvec_path.write_text(
                "\n".join(" ".join(f"{dirs[g, r]:.8f}" for g in range(ng)) for r in range(3)) + "\n",
                encoding="utf-8",
            )
            nhdr_write(str(nii_path), str(bval_path), str(bvec_path), str(nhdr_path))
            self.assertEqual(teem_dti._nifti_payload_from_nhdr(nhdr_path), nii_path.resolve())

            ten_stream = tdir / "dti_stream.nrrd"
            b0_stream = tdir / "b0_stream.nrrd"
            teem_dti.estimate_file(nhdr_path, ten_stream, b0_stream, method="WLS", threads=2, log=None)

            ten_vtk = tdir / "dti_vtk.nrrd"
            b0_vtk = tdir / "b0_vtk.nrrd"
            with mock.patch.object(teem_dti, "_nifti_payload_from_nhdr", return_value=None):
                teem_dti.estimate_file(nhdr_path, ten_vtk, b0_vtk, method="WLS", threads=2, log=None)

            for p_a, p_b, is_tensor in ((ten_stream, ten_vtk, True), (b0_stream, b0_vtk, False)):
                ra = self.vtkTeem.vtkTeemNRRDReader()
                ra.SetFileName(str(p_a))
                ra.Update()
                rb = self.vtkTeem.vtkTeemNRRDReader()
                rb.SetFileName(str(p_b))
                rb.Update()
                arr_a = self.ns.vtk_to_numpy(
                    ra.GetOutput().GetPointData().GetTensors() if is_tensor else ra.GetOutput().GetPointData().GetScalars()
                )
                arr_b = self.ns.vtk_to_numpy(
                    rb.GetOutput().GetPointData().GetTensors() if is_tensor else rb.GetOutput().GetPointData().GetScalars()
                )
                np.testing.assert_array_equal(arr_a, arr_b)
                np.testing.assert_allclose(
                    teem_dti._matrix_to_numpy(ra.GetRasToIjkMatrix()),
                    teem_dti._matrix_to_numpy(rb.GetRasToIjkMatrix()),
                    atol=0, rtol=0,
                )

    def test_all_zero_diffusion_volume_raises_teem_error(self) -> None:
        import nibabel as nib
        from anatomy.parcellation.conversion.nhdr_write import nhdr_write

        with tempfile.TemporaryDirectory() as tmp:
            tdir = Path(tmp)
            nii_path = tdir / "zeros.nii.gz"
            bval_path = tdir / "zeros.bval"
            bvec_path = tdir / "zeros.bvec"
            nhdr_path = tdir / "DWI.nhdr"
            img = nib.Nifti1Image(np.zeros((4, 4, 4, 7), dtype=np.float32), np.eye(4))
            img.header["scl_slope"] = 1.0
            img.header["scl_inter"] = 0.0
            nib.save(img, str(nii_path))
            bval_path.write_text("0 1000 1000 1000 1000 1000 1000\n", encoding="utf-8")
            bvec_path.write_text("0 1 0 0 0.7 0.7 0\n0 0 1 0 0.7 0 0.7\n0 0 0 1 0 0.7 0.7\n", encoding="utf-8")
            nhdr_write(str(nii_path), str(bval_path), str(bvec_path), str(nhdr_path))
            with self.assertRaisesRegex(teem_dti.TeemError, "all zeros"):
                teem_dti.estimate_file(nhdr_path, tdir / "dti.nrrd", tdir / "b0.nrrd", method="WLS", threads=1, log=None)


if __name__ == "__main__":
    unittest.main()
