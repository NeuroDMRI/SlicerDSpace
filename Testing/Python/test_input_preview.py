"""Tests for the 'View input' preview (item H1, b2-dwi): DWI resolution, the full-DWI build, the
size fallback to the mean-b0 file, and the volume stepper.

The numeric tests need numpy, nibabel, pynrrd and vtk (PythonSlicer has them; the host python3
may not) and are skipped without them.
"""

from __future__ import annotations

import ast
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from shared.input_preview import (
    PREVIEW_ATTRIBUTE,
    PreviewError,
    PreviewSource,
    mean_b0_image,
    resolve_pending_source,
    resolve_preview_for_ui,
    resolve_preview_source,
    show_preview_in_scene,
)

try:
    import nibabel as nib
    import numpy as np
except ImportError:  # pragma: no cover - host python without the Slicer stack
    nib = None
    np = None
try:
    import nrrd
except ImportError:  # pragma: no cover
    nrrd = None

REPO_ROOT = Path(__file__).resolve().parents[2]
AFFINE = np.array([[2.0, 0, 0, -10.0], [0, 2.5, 0, 5.0], [0, 0, 3.0, 7.5], [0, 0, 0, 1]]) if np is not None else None
BVALS = [0, 1000, 5, 1000, 0]


def write_dwi(folder: Path, name: str = "data.nii.gz", bvals=BVALS, shape=(4, 5, 6)) -> Path:
    volumes = [np.full(shape, 10.0 * (i + 1), dtype=np.float32) for i in range(len(bvals))]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    nib.save(nib.Nifti1Image(np.stack(volumes, axis=-1), AFFINE), str(path))
    return path


def write_text(path: Path, text: str = "x\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@unittest.skipIf(nib is None, "numpy/nibabel unavailable")
class MeanB0Test(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.cache = self.root / "cache"

    def nifti_source(self, bvals=BVALS) -> PreviewSource:
        subject = self.root / "subject" / "input"
        subject.mkdir(parents=True)
        dwi = write_dwi(subject, bvals=bvals)
        bval = write_text(subject / "bvals", " ".join(str(b) for b in bvals))
        return PreviewSource("nifti", dwi, bval, None, "sub-01")

    def test_h1_nifti_mean_b0_uses_only_low_b_volumes_and_keeps_geometry(self):
        source = self.nifti_source()
        out = mean_b0_image(source, self.cache)
        image = nib.load(str(out))
        self.assertEqual(image.shape, (4, 5, 6))
        # volumes 0, 2, 4 (b <= 50) hold 10, 30 and 50.
        np.testing.assert_allclose(np.asarray(image.dataobj), 30.0)
        np.testing.assert_allclose(image.affine, AFFINE)
        self.assertEqual(image.get_data_dtype(), np.dtype("float32"))
        self.assertEqual(out.parent, self.cache)
        self.assertEqual(sorted(p.name for p in source.dwi.parent.iterdir()), ["bvals", "data.nii.gz"])

    def test_h1_mean_b0_is_reused_and_rebuilt_when_the_dwi_changes(self):
        source = self.nifti_source()
        first = mean_b0_image(source, self.cache)
        self.assertEqual(mean_b0_image(source, self.cache), first)
        write_dwi(source.dwi.parent, bvals=BVALS, shape=(4, 5, 7))
        second = mean_b0_image(source, self.cache)
        self.assertNotEqual(second, first)
        self.assertEqual(nib.load(str(second)).shape, (4, 5, 7))

    def test_h1_nifti_errors_are_preview_errors(self):
        source = self.nifti_source()
        write_text(source.bval, "0 1000")
        with self.assertRaises(PreviewError) as caught:
            mean_b0_image(source, self.cache)
        self.assertEqual(caught.exception.code, "bad_data")
        self.assertIn("2 entries", caught.exception.detail)
        write_text(source.bval, "1000 1000 1000 1000 1000")
        with self.assertRaises(PreviewError) as caught:
            mean_b0_image(source, self.cache)
        self.assertEqual(caught.exception.code, "no_b0")
        write_text(source.bval, "0 a 0 0 0")
        with self.assertRaises(PreviewError) as caught:
            mean_b0_image(source, self.cache)
        self.assertEqual(caught.exception.code, "bad_data")
        with self.assertRaises(PreviewError) as caught:
            mean_b0_image(PreviewSource("nifti", source.dwi, None, None, "s"), self.cache)
        self.assertEqual(caught.exception.code, "no_bval")
        with self.assertRaises(PreviewError) as caught:
            mean_b0_image(PreviewSource("nifti", self.root / "none.nii.gz", source.bval, None, "s"), self.cache)
        self.assertEqual(caught.exception.code, "missing")

    def test_h1_three_dimensional_dwi_is_rejected(self):
        source = self.nifti_source()
        nib.save(nib.Nifti1Image(np.zeros((3, 3, 3), dtype=np.float32), AFFINE), str(source.dwi))
        with self.assertRaises(PreviewError) as caught:
            mean_b0_image(source, self.cache)
        self.assertEqual(caught.exception.code, "bad_data")

    def test_h1_cache_keeps_only_the_newest_previews(self):
        for index in range(8):
            source = self.nifti_source() if index == 0 else source
            source = PreviewSource(source.kind, source.dwi, source.bval, None, f"sub-{index}")
            mean_b0_image(source, self.cache)
        self.assertEqual(len(list(self.cache.glob("preview_*_meanb0.nii.gz"))), 6)

    def write_nrrd(self, axis_last: bool, space: str = "left-posterior-superior") -> Path:
        shape = (4, 5, 6)
        volumes = [np.full(shape, 10.0 * (i + 1), dtype=np.float32) for i in range(5)]
        data = np.stack(volumes, axis=-1 if axis_last else 0).astype(np.int16)
        spatial = [[2.0, 0, 0], [0, 2.5, 0], [0, 0, 3.0]]
        none = [float("nan")] * 3
        directions = spatial + [none] if axis_last else [none] + spatial
        kinds = ["domain"] * 3 + ["list"] if axis_last else ["list"] + ["domain"] * 3
        header = {
            "space": space,
            "space directions": np.array(directions),
            "kinds": kinds,
            "space origin": np.array([10.0, -5.0, 7.5]),
            "DWMRI_b-value": "1000",
        }
        for index, vector in enumerate(["0 0 0", "1 0 0", "0 0 0", "0 1 0", "0 0 0"]):
            header["DWMRI_gradient_%04d" % index] = vector
        path = self.root / ("dwi_last.nrrd" if axis_last else "dwi_first.nrrd")
        nrrd.write(str(path), data, header=header)
        return path

    @unittest.skipIf(nrrd is None, "pynrrd unavailable")
    def test_h1_nrrd_mean_b0_from_gradient_header_with_ras_affine(self):
        for axis_last in (True, False):
            path = self.write_nrrd(axis_last)
            out = mean_b0_image(PreviewSource("nrrd", path, None, None, f"nrrd-{axis_last}"), self.cache)
            image = nib.load(str(out))
            self.assertEqual(image.shape, (4, 5, 6))
            np.testing.assert_allclose(np.asarray(image.dataobj), 30.0)
            # LPS -> RAS flips x and y: voxel axes become -2, -2.5, +3 and the origin (-10, 5, 7.5).
            expected = np.array([[-2.0, 0, 0, -10.0], [0, -2.5, 0, 5.0], [0, 0, 3.0, 7.5], [0, 0, 0, 1]])
            np.testing.assert_allclose(image.affine, expected)

    @unittest.skipIf(nrrd is None, "pynrrd unavailable")
    def test_h1_nrrd_ras_space_is_not_flipped_and_missing_gradients_fail(self):
        path = self.write_nrrd(True, space="right-anterior-superior")
        out = mean_b0_image(PreviewSource("nrrd", path, None, None, "ras"), self.cache)
        np.testing.assert_allclose(nib.load(str(out)).affine[:3, :3], np.diag([2.0, 2.5, 3.0]))
        data, header = nrrd.read(str(path))
        del header["DWMRI_gradient_0003"]
        broken = self.root / "broken.nrrd"
        nrrd.write(str(broken), data, header=header)
        with self.assertRaises(PreviewError) as caught:
            mean_b0_image(PreviewSource("nrrd", broken, None, None, "broken"), self.cache)
        self.assertEqual(caught.exception.code, "bad_data")


@unittest.skipIf(nib is None, "numpy/nibabel unavailable")
class ResolveTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)

    def canonical_subject(self, parent: Path, name: str, mask: bool = False) -> Path:
        subject = parent / name
        write_dwi(write_text(subject / "input" / "keep").parent)
        for item in ("bvals", "bvecs"):
            write_text(subject / "input" / item)
        if mask:
            write_text(subject / "input" / "nodif_brain_mask.nii.gz")
        return subject

    def test_h1_subject_root_uses_the_validated_canonical_files(self):
        subject = self.canonical_subject(self.root, "sub-01", mask=True)
        source = resolve_preview_for_ui(input_dir=str(subject), subject_id="sub-01")
        self.assertEqual(source.kind, "nifti")
        self.assertEqual(source.dwi, subject / "input" / "data.nii.gz")
        self.assertEqual(source.bval, subject / "input" / "bvals")
        self.assertEqual(source.mask, subject / "input" / "nodif_brain_mask.nii.gz")
        # A parent folder plus subject id resolves to the same subject.
        again = resolve_preview_for_ui(input_dir=str(self.root), subject_id="sub-01")
        self.assertEqual(again.dwi, source.dwi)

    def test_h1_generated_brain_mask_is_not_a_user_mask(self):
        subject = self.canonical_subject(self.root, "sub-01", mask=True)
        write_text(subject / "input" / "nodif_brain_mask.manifest.json", "{}")
        source = resolve_preview_for_ui(input_dir=str(subject), subject_id="sub-01")
        self.assertIsNone(source.mask)

    def test_h1_raw_nifti_file_with_gradients_is_previewed_in_place(self):
        folder = self.root / "raw"
        dwi = write_dwi(folder, "scan.nii.gz")
        write_text(folder / "scan.bval", "0 1000 5 1000 0")
        write_text(folder / "scan.bvec")
        write_text(folder / "scan_mask.nii.gz")
        for target in (folder, dwi):
            source = resolve_preview_for_ui(input_dir=str(target), subject_id="raw")
            self.assertEqual((source.kind, source.dwi.name, source.bval.name, source.mask.name),
                             ("nifti", "scan.nii.gz", "scan.bval", "scan_mask.nii.gz"))

    @unittest.skipIf(nrrd is None, "pynrrd unavailable")
    def test_h1_nrrd_folder_is_detected_as_nrrd(self):
        folder = self.root / "n"
        folder.mkdir()
        path = folder / "dwi.nrrd"
        nrrd.write(str(path), np.zeros((2, 2, 2, 1), dtype=np.int16))
        source = resolve_preview_for_ui(input_dir=str(folder), subject_id="n")
        self.assertEqual((source.kind, source.dwi, source.bval), ("nrrd", path.resolve(), None))

    def test_h1_dicom_and_other_layouts_are_not_previewable(self):
        for mode, code in (("dicom", "dicom_unconverted"), ("legacy_registered", "no_dwi"), ("missing", "missing")):
            with self.assertRaises(PreviewError) as caught:
                resolve_preview_source({"mode": mode, "subject_root": None}, self.root, "s")
            self.assertEqual(caught.exception.code, code)
        self.assertEqual(PreviewError("dicom_unconverted").text(), "Convert inputs first (Prepare Inputs)")
        self.assertIn("准备输入", PreviewError("dicom_unconverted").text(zh=True))

    def test_h1_empty_or_missing_input_reports_no_input(self):
        for value, code in (("", "no_input"), (str(self.root / "absent"), "missing")):
            with self.assertRaises(PreviewError) as caught:
                resolve_preview_for_ui(input_dir=value, subject_id="s")
            self.assertEqual(caught.exception.code, code)

    def test_b2_inputs_batch_has_no_preview_api_left(self):
        import inspect

        from shared import input_preview as ip

        self.assertFalse(hasattr(ip, "first_batch_source"))
        self.assertEqual(set(inspect.signature(ip.resolve_preview_for_ui).parameters),
                         {"input_dir", "subject_id", "pending", "inspect"})

    def test_b2_dwi_bvec_is_resolved_for_subject_root_raw_file_and_pending_tile(self):
        subject = self.canonical_subject(self.root, "sub-01")
        source = resolve_preview_for_ui(input_dir=str(subject), subject_id="sub-01")
        self.assertEqual(source.bvec, subject / "input" / "bvecs")
        folder = self.root / "raw"
        dwi = write_dwi(folder, "scan.nii.gz")
        write_text(folder / "scan.bval", "0 1000 5 1000 0")
        write_text(folder / "scan.bvec")
        for target in (folder, dwi):
            found = resolve_preview_for_ui(input_dir=str(target), subject_id="raw")
            self.assertEqual(found.bvec.name, "scan.bvec")
        tile_bvec = write_text(self.root / "tiles" / "other.bvec")
        tile_bval = write_text(self.root / "tiles" / "other.bval", "0 1000 5 1000 0")
        tile_dwi = write_dwi(self.root / "tiles", "x.nii.gz")
        pending = {"kind": "nifti", "source": str(tile_dwi), "bval": str(tile_bval), "bvec": str(tile_bvec)}
        self.assertEqual(resolve_preview_for_ui(input_dir="", subject_id="s",
                                                pending=pending).bvec, tile_bvec)
        # An empty bvec tile falls back to the matching sidecar, like the bval tile does.
        found = resolve_pending_source("nifti", dwi, bval=str(folder / "scan.bval"), bvec="")
        self.assertEqual(found.bvec, (folder / "scan.bvec").resolve())
        # No bvec anywhere is not a resolve error: the size fallback (mean b0) does not need one.
        self.assertIsNone(resolve_pending_source("nifti", tile_dwi, bval=str(tile_bval)).bvec)
        self.assertIsNone(resolve_pending_source("nrrd", dwi, subject_id="s").bvec)

    def test_b2_dwi_dicom_input_is_still_refused_with_convert_first(self):
        pending = {"kind": "dicom", "source": str(self.root)}
        with self.assertRaises(PreviewError) as caught:
            resolve_preview_for_ui(input_dir="", subject_id="s", pending=pending)
        self.assertEqual(caught.exception.text(), "Convert inputs first (Prepare Inputs)")

    def test_h1_pending_tile_sources_resolve_without_an_import(self):
        folder = self.root / "tiles"
        dwi = write_dwi(folder, "x.nii.gz")
        bval = write_text(folder / "other.bval", "0 1000 5 1000 0")
        pending = {"kind": "nifti", "source": str(dwi), "bval": str(bval), "mask": ""}
        source = resolve_preview_for_ui(input_dir="stale", subject_id="s", pending=pending)
        self.assertEqual((source.kind, source.bval, source.mask), ("nifti", bval, None))
        with self.assertRaises(PreviewError) as caught:
            resolve_pending_source("nifti", dwi, bval="")
        self.assertEqual(caught.exception.code, "no_bval")
        with self.assertRaises(PreviewError) as caught:
            resolve_pending_source("dicom", folder)
        self.assertEqual(caught.exception.code, "dicom_unconverted")
        self.assertEqual(resolve_pending_source("nrrd", dwi, subject_id="s").kind, "nrrd")


try:
    import vtk
    from vtk.util import numpy_support
except ImportError:  # pragma: no cover - host python without the Slicer stack
    vtk = None
    numpy_support = None

# b2-dwi gradient frame: affines, FSL bvec rows and the RAS gradients that MRtrix 3.0.8 `mrconvert -fslgrad
# -export_grad_mrtrix` wrote for the investigation's synthetic oblique files (scratchpad b2inv/view_input_dwi/ref).
OBLIQUE_POS = [[1.720178723335, -1.019072532654, -0.062204446644, -90.0],
               [0.939737915993, 1.620478391647, -0.875841140747, -100.0],
               [0.397338658571, 0.579258978367, 2.340733528137, -60.0],
               [0.0, 0.0, 0.0, 1.0]]  # det(A[:3, :3]) = +10
OBLIQUE_NEG = [[-1.720178723335, -1.019072532654, -0.062204446644, -90.0],
               [-0.939737915993, 1.620478391647, -0.875841140747, -100.0],
               [-0.397338658571, 0.579258978367, 2.340733528137, -60.0],
               [0.0, 0.0, 0.0, 1.0]]  # det(A[:3, :3]) = -10
FSL_POS = [[-0.35982316, -0.87237628, -0.33088808], [-0.04306522, 0.92408052, 0.37976386],
           [-0.60121242, 0.32504197, -0.72999408]]
MRTRIX_POS = [[0.7622204998, -0.4218415734, -0.4909883875], [-0.4432617946, 0.6359161383, 0.6317670825],
              [0.3696392712, 0.8015963158, -0.4699044112]]
FSL_NEG = [[-0.04902684, 0.98970584, 0.13445716], [0.95690755, 0.09360975, -0.27489116],
           [0.11460117, -0.9917759, -0.05698373]]
MRTRIX_NEG = [[-0.4654690847, 0.7778294033, 0.4222794697], [-0.8638837683, -0.2774704641, -0.4203748046],
              [0.408196397, -0.8374597536, -0.3633687696]]


def write_bvec(path: Path, rows, transpose: bool = True) -> Path:
    """FSL layout (3 rows x N columns) unless ``transpose`` is False (N rows x 3 columns)."""

    matrix = np.asarray(rows, dtype=float)
    matrix = matrix.T if transpose else matrix
    return write_text(path, "\n".join(" ".join(repr(float(v)) for v in line) for line in matrix) + "\n")


def vtk_voxels(image):
    """The vtkImageData scalars as (i, j, k, component), the NIfTI index order."""

    nx, ny, nz = image.GetDimensions()
    scalars = image.GetPointData().GetScalars()
    count = scalars.GetNumberOfComponents()
    return numpy_support.vtk_to_numpy(scalars).reshape(nz, ny, nx, count).transpose(2, 1, 0, 3)


def tree_snapshot(root: Path) -> dict:
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(root.rglob("*"))}


@unittest.skipIf(nib is None or vtk is None, "numpy/nibabel/vtk unavailable")
class FullDwiTest(unittest.TestCase):
    """b2-dwi: View input builds the whole DWI (not the mean b0) in memory."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)
        self.cache = self.root / "cache"

    def nifti(self, data, bvals, affine=None, bvecs=None, name="sub", slope_inter=None, transpose=True):
        folder = self.root / name / "input"
        folder.mkdir(parents=True)
        image = nib.Nifti1Image(data, AFFINE if affine is None else np.asarray(affine))
        if slope_inter is not None:
            image.header.set_slope_inter(*slope_inter)
        dwi = folder / "data.nii.gz"
        nib.save(image, str(dwi))
        bval = write_text(folder / "bvals", " ".join(str(b) for b in bvals))
        count = data.shape[3]
        rows = bvecs if bvecs is not None else [[0.0, 0.0, 0.0] if b <= 50 else [1.0, 0.0, 0.0] for b in bvals]
        bvec = write_bvec(folder / "bvecs", rows, transpose=transpose)
        self.assertEqual(len(rows), count)
        return PreviewSource("nifti", dwi, bval, None, name, bvec=bvec)

    def test_b2_dwi_int16_block_fill_keeps_dtype_and_component_order(self):
        from shared import input_preview as ip

        rng = np.random.default_rng(7)
        data = rng.integers(-3000, 3000, size=(5, 4, 3, 23)).astype(np.int16)  # 23 volumes: blocks 10 + 10 + 3
        bvals = [0, 1000, 2000] * 7 + [5, 1000]
        dwi = ip.read_full_dwi(self.nifti(data, bvals))
        self.assertEqual(dwi.image.GetDimensions(), (5, 4, 3))
        self.assertEqual(dwi.image.GetScalarTypeAsString(), "short")
        self.assertEqual(dwi.image.GetNumberOfScalarComponents(), 23)
        np.testing.assert_array_equal(vtk_voxels(dwi.image), data)
        np.testing.assert_allclose(dwi.ijk_to_ras, AFFINE)
        self.assertEqual(list(dwi.bvalues), [float(b) for b in bvals])  # two shells pass through
        np.testing.assert_allclose(dwi.measurement_frame, np.eye(3))
        self.assertEqual(dwi.voxel_bytes, data.nbytes)

    def test_b2_dwi_scaled_nifti_becomes_float32_with_the_scale_applied(self):
        from shared import input_preview as ip

        data = np.arange(4 * 3 * 2 * 12, dtype=np.int16).reshape(4, 3, 2, 12)
        source = self.nifti(data, [0] + [1000] * 11, slope_inter=(0.5, 3.0))
        self.assertEqual(ip.estimate_full_dwi_bytes(source), data.size * 4)
        dwi = ip.read_full_dwi(source)
        self.assertEqual(dwi.image.GetScalarTypeAsString(), "float")
        np.testing.assert_allclose(vtk_voxels(dwi.image), data.astype(np.float32) * 0.5 + 3.0)

    def test_b2_dwi_unscaled_float_and_unsigned_types_are_kept(self):
        from shared import input_preview as ip

        for dtype, vtk_name in ((np.float32, "float"), (np.uint16, "unsigned short")):
            data = (np.arange(3 * 3 * 2 * 4).reshape(3, 3, 2, 4) * 7).astype(dtype)
            dwi = ip.read_full_dwi(self.nifti(data, [0, 1000, 0, 1000], name=f"t-{vtk_name.replace(' ', '_')}"))
            self.assertEqual(dwi.image.GetScalarTypeAsString(), vtk_name)
            np.testing.assert_array_equal(vtk_voxels(dwi.image), data)

    def test_b2_dwi_gradients_ras_det_positive_matches_mrtrix(self):
        from shared import input_preview as ip

        np.testing.assert_allclose(ip.gradients_ras(OBLIQUE_POS, FSL_POS), MRTRIX_POS, atol=1e-6)

    def test_b2_dwi_gradients_ras_det_negative_matches_mrtrix(self):
        from shared import input_preview as ip

        np.testing.assert_allclose(ip.gradients_ras(OBLIQUE_NEG, FSL_NEG), MRTRIX_NEG, atol=1e-6)

    def test_b2_dwi_gradients_ras_keeps_b0_rows_zero(self):
        from shared import input_preview as ip

        out = ip.gradients_ras(OBLIQUE_POS, [[0.0, 0.0, 0.0]] + FSL_POS)
        np.testing.assert_array_equal(out[0], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(out[1:], MRTRIX_POS, atol=1e-6)

    def test_b2_dwi_nifti_gradients_use_the_ras_mapping_and_both_bvec_layouts(self):
        from shared import input_preview as ip

        data = np.zeros((3, 3, 2, 4), dtype=np.int16)
        rows = [[0.0, 0.0, 0.0]] + FSL_POS
        for transpose in (True, False):
            source = self.nifti(data, [0, 1000, 2000, 1000], affine=OBLIQUE_POS, bvecs=rows,
                                name=f"layout-{transpose}", transpose=transpose)
            dwi = ip.read_full_dwi(source)
            np.testing.assert_allclose(dwi.gradients, [[0.0, 0.0, 0.0]] + MRTRIX_POS, atol=1e-6)
            np.testing.assert_allclose(dwi.ijk_to_ras, OBLIQUE_POS)
            np.testing.assert_allclose(dwi.measurement_frame, np.eye(3))

    def test_b2_dwi_missing_bvec_and_count_mismatch_are_preview_errors(self):
        from shared import input_preview as ip

        data = np.zeros((3, 3, 2, 4), dtype=np.int16)
        source = self.nifti(data, [0, 1000, 0, 1000])
        with self.assertRaises(PreviewError) as caught:
            ip.read_full_dwi(PreviewSource("nifti", source.dwi, source.bval, None, "s"))
        self.assertEqual(caught.exception.code, "no_bvec")
        self.assertIn("bvec", caught.exception.text())
        self.assertIn("bvec", caught.exception.text(zh=True))
        write_bvec(source.bvec, [[1.0, 0.0, 0.0]] * 3)
        with self.assertRaises(PreviewError) as caught:
            ip.read_full_dwi(source)
        self.assertEqual(caught.exception.code, "bad_data")
        write_text(source.bvec, "0 1 x 0\n0 0 0 0\n0 0 0 0\n")
        with self.assertRaises(PreviewError) as caught:
            ip.read_full_dwi(source)
        self.assertEqual(caught.exception.code, "bad_data")

    def write_nrrd(self, axis_last: bool, space="left-posterior-superior", frame=None, name="dwi"):
        rng = np.random.default_rng(3)
        volumes = rng.integers(0, 1000, size=(4, 5, 6, 5)).astype(np.int16)  # (i, j, k, volume)
        data = volumes if axis_last else np.moveaxis(volumes, -1, 0)
        spatial = [[2.0, 0, 0], [0, 2.5, 0], [0, 0, 3.0]]
        none = [float("nan")] * 3
        header = {
            "space": space,
            "space directions": np.array(spatial + [none] if axis_last else [none] + spatial),
            "kinds": ["domain"] * 3 + ["list"] if axis_last else ["list"] + ["domain"] * 3,
            "space origin": np.array([10.0, -5.0, 7.5]),
            "DWMRI_b-value": "1000",
        }
        if frame is not None:
            header["measurement frame"] = np.array(frame, dtype=float)  # pynrrd: one row per header vector
        for index, vector in enumerate(["0 0 0", "1 0 0", "0 0 0", "0 0.6 0.8", "0 0 0.70710678"]):
            header["DWMRI_gradient_%04d" % index] = vector
        path = self.root / f"{name}.nrrd"
        nrrd.write(str(path), data, header=header)
        return path, volumes

    @unittest.skipIf(nrrd is None, "pynrrd unavailable")
    def test_b2_dwi_nrrd_lps_list_last_passes_gradients_and_measurement_frame(self):
        from shared import input_preview as ip

        # Header vectors (0,1,0) (-1,0,0) (0,0,1) are the matrix columns: M = [[0,-1,0],[1,0,0],[0,0,1]].
        path, volumes = self.write_nrrd(True, frame=[[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        dwi = ip.read_full_dwi(PreviewSource("nrrd", path, None, None, "n"))
        np.testing.assert_allclose(dwi.measurement_frame, [[0, 1, 0], [-1, 0, 0], [0, 0, 1]])  # diag(-1,-1,1) @ M
        np.testing.assert_allclose(dwi.gradients, [[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0.6, 0.8], [0, 0, 1]],
                                   atol=1e-7)
        np.testing.assert_allclose(dwi.bvalues, [0, 1000, 0, 1000, 500], atol=1e-4)
        np.testing.assert_allclose(dwi.ijk_to_ras, [[-2.0, 0, 0, -10.0], [0, -2.5, 0, 5.0], [0, 0, 3.0, 7.5],
                                                    [0, 0, 0, 1]])
        self.assertEqual(dwi.image.GetScalarTypeAsString(), "short")
        np.testing.assert_array_equal(vtk_voxels(dwi.image), volumes)

    @unittest.skipIf(nrrd is None, "pynrrd unavailable")
    def test_b2_dwi_nrrd_ras_list_first_without_frame_uses_identity(self):
        from shared import input_preview as ip

        path, volumes = self.write_nrrd(False, space="right-anterior-superior", name="ras")
        dwi = ip.read_full_dwi(PreviewSource("nrrd", path, None, None, "n"))
        np.testing.assert_allclose(dwi.measurement_frame, np.eye(3))
        np.testing.assert_allclose(dwi.ijk_to_ras[:3, :3], np.diag([2.0, 2.5, 3.0]))
        np.testing.assert_array_equal(vtk_voxels(dwi.image), volumes)

    @unittest.skipIf(nrrd is None, "pynrrd unavailable")
    def test_b2_dwi_c1_detached_nhdr_with_a_data_tile_builds_from_the_staged_header(self):
        from shared import input_preview as ip

        folder = self.root / "det"
        folder.mkdir()
        volumes = np.stack([np.full((4, 5, 6), 10 * (i + 1), dtype=np.int16) for i in range(5)], axis=-1)
        none = [float("nan")] * 3
        header = {"space": "right-anterior-superior", "kinds": ["domain"] * 3 + ["list"],
                  "space directions": np.array([[2.0, 0, 0], [0, 2.5, 0], [0, 0, 3.0], none]),
                  "space origin": np.zeros(3), "DWMRI_b-value": "1000", "encoding": "raw"}
        for index, vector in enumerate(["0 0 0", "1 0 0", "0 0 0", "0 1 0", "0 0 0"]):
            header["DWMRI_gradient_%04d" % index] = vector
        nrrd.write(str(folder / "dwi.nhdr"), volumes, header=header, detached_header=True)
        other = folder / "other.raw"
        (volumes * 3).astype(np.int16).ravel(order="F").tofile(str(other))  # same layout, other values
        source = resolve_pending_source("nrrd", folder / "dwi.nhdr", raw=str(other), subject_id="s")
        dwi = ip.read_full_dwi(source)
        np.testing.assert_array_equal(vtk_voxels(dwi.image), volumes * 3)
        self.assertEqual(sorted(p.name for p in folder.iterdir()), ["dwi.nhdr", "dwi.raw", "other.raw"])

    def test_b2_dwi_default_component_is_the_first_low_b_volume(self):
        from shared import input_preview as ip

        self.assertEqual(ip.default_component([1000, 2000, 5, 0]), 2)
        self.assertEqual(ip.default_component([0, 1000]), 0)
        self.assertEqual(ip.default_component([1000, 50.0, 0]), 1)
        self.assertEqual(ip.default_component([1000, 2000]), 0)  # no b0: the first volume

    def test_b2_dwi_size_limit_is_four_gib_or_a_quarter_of_ram(self):
        from shared import input_preview as ip

        self.assertEqual(ip.full_dwi_limit_bytes(8 << 30), 2 << 30)
        self.assertEqual(ip.full_dwi_limit_bytes(64 << 30), 4 << 30)
        self.assertEqual(ip.full_dwi_limit_bytes(None), 4 << 30)
        physical = ip.physical_memory_bytes()
        self.assertTrue(physical is None or physical > (256 << 20))

    def test_b2_dwi_size_guard_falls_back_to_the_mean_b0_with_a_notice(self):
        from shared import input_preview as ip

        data = np.stack([np.full((4, 5, 6), 10 * (i + 1), dtype=np.int16) for i in range(5)], axis=-1)
        source = self.nifti(data, BVALS)
        with mock.patch.object(ip, "read_full_dwi", side_effect=AssertionError("full read above the limit")):
            prepared = ip.prepare_input_preview(source, self.cache, limit=data.nbytes - 1)
        self.assertEqual(prepared.kind, "meanb0")
        self.assertEqual(prepared.estimated_bytes, data.nbytes)
        self.assertEqual(prepared.limit_bytes, data.nbytes - 1)
        self.assertEqual(prepared.mean_b0.parent, self.cache)
        np.testing.assert_allclose(np.asarray(nib.load(str(prepared.mean_b0)).dataobj), 30.0)
        prepared = ip.prepare_input_preview(source, self.cache, limit=data.nbytes)  # at the limit: full DWI
        self.assertEqual(prepared.kind, "dwi")
        self.assertIsNone(prepared.mean_b0)
        notice = ip.size_notice(5 * (1 << 30) + (1 << 29), 4 << 30)
        self.assertIn("5.5 GiB", notice)
        self.assertIn("4.0 GiB", notice)
        self.assertIn("mean b0", notice)
        zh = ip.size_notice(5 * (1 << 30) + (1 << 29), 4 << 30, zh=True)
        self.assertIn("5.5 GiB", zh)
        self.assertIn("平均 b0", zh)

    @unittest.skipIf(nrrd is None, "pynrrd unavailable")
    def test_b2_dwi_size_guard_reads_only_the_nrrd_header(self):
        from shared import input_preview as ip

        path, volumes = self.write_nrrd(True)
        source = PreviewSource("nrrd", path, None, None, "n")
        self.assertEqual(ip.estimate_full_dwi_bytes(source), volumes.nbytes)
        with mock.patch.object(ip, "read_full_dwi", side_effect=AssertionError("full read above the limit")):
            self.assertEqual(ip.prepare_input_preview(source, self.cache, limit=1).kind, "meanb0")

    def test_b2_dwi_full_view_writes_no_file_anywhere(self):
        from shared import input_preview as ip

        data = np.stack([np.full((4, 5, 6), 10 * (i + 1), dtype=np.int16) for i in range(5)], axis=-1)
        source = self.nifti(data, BVALS)
        (self.root / "output").mkdir()
        before = tree_snapshot(self.root)
        prepared = ip.prepare_input_preview(source, self.cache)
        self.assertEqual(prepared.kind, "dwi")
        self.assertEqual(tree_snapshot(self.root), before)
        self.assertFalse(self.cache.exists())

    def mean_b0_source(self, name="nb"):
        data = np.stack([np.full((4, 5, 6), 10 * (i + 1), dtype=np.int16) for i in range(5)], axis=-1)
        return data, self.nifti(data, BVALS, name=name)

    def test_b2_addon_missing_bvec_falls_back_to_the_mean_b0_with_a_gradient_notice(self):
        from shared import input_preview as ip

        data, source = self.mean_b0_source()
        for label, bvec in (("no bvec path", None), ("bvec file absent", source.bvec.with_name("gone.bvec"))):
            with self.subTest(label), mock.patch.object(ip, "read_full_dwi", side_effect=AssertionError("full read")):
                prepared = ip.prepare_input_preview(
                    PreviewSource("nifti", source.dwi, source.bval, None, "s", bvec=bvec), self.cache)
            self.assertEqual(prepared.kind, "meanb0")
            self.assertEqual(prepared.reason, "no_gradients")
            np.testing.assert_allclose(np.asarray(nib.load(str(prepared.mean_b0)).dataobj), 30.0)
        self.assertIn("gradient", ip.gradient_notice(False))
        self.assertIn("mean b0", ip.gradient_notice(False))
        self.assertIn("梯度", ip.gradient_notice(True))
        self.assertIn("平均 b0", ip.gradient_notice(True))
        self.assertNotIn("GiB", ip.gradient_notice(False))

    def test_b2_addon_unreadable_bvec_falls_back_but_other_bad_data_stays_an_error(self):
        from shared import input_preview as ip

        data, source = self.mean_b0_source("unreadable")
        write_text(source.bvec, "0 1 x 0 0\n0 0 0 0 0\n0 0 0 0 0\n")  # a non-numeric entry
        self.assertEqual(ip.prepare_input_preview(source, self.cache).reason, "no_gradients")
        write_bvec(source.bvec, [[1.0, 0.0, 0.0]] * 3)  # 3 entries for 5 volumes
        self.assertEqual(ip.prepare_input_preview(source, self.cache).kind, "meanb0")
        write_text(source.bval, "0 1000")  # the bval count does not match: still an error, not a fallback
        write_bvec(source.bvec, [[0.0, 0.0, 0.0]] * 5)
        with self.assertRaises(PreviewError) as caught:
            ip.prepare_input_preview(source, self.cache)
        self.assertEqual(caught.exception.code, "bad_data")

    def test_b2_addon_missing_bval_is_still_an_error_because_the_mean_b0_needs_it(self):
        from shared import input_preview as ip

        data, source = self.mean_b0_source("nobval")
        with self.assertRaises(PreviewError) as caught:
            ip.prepare_input_preview(PreviewSource("nifti", source.dwi, None, None, "s", bvec=None), self.cache)
        self.assertEqual(caught.exception.code, "no_bval")

    def test_b2_addon_full_dwi_path_is_unchanged_when_the_gradient_files_exist(self):
        from shared import input_preview as ip

        data, source = self.mean_b0_source("intact")
        prepared = ip.prepare_input_preview(source, self.cache)
        self.assertEqual((prepared.kind, prepared.reason), ("dwi", ""))
        self.assertIsNone(prepared.mean_b0)
        self.assertFalse(self.cache.exists())
        nrrd_like = PreviewSource("nrrd", source.dwi, None, None, "n")  # NRRD has no bvec file: never a gradient fallback
        self.assertFalse(ip.gradient_table_unusable(nrrd_like))


class FakeNode:
    def __init__(self, name):
        self.name = name
        self.attributes = {}

    def SetAttribute(self, key, value):
        self.attributes[key] = value

    def GetAttribute(self, key):
        return self.attributes.get(key)

    def GetID(self):
        return "id-" + self.name


class FakeSlicer:
    def __init__(self, fail_label=False):
        self.nodes = []
        self.removed = []
        self.layers = None
        self.fail_label = fail_label
        scene = SimpleNamespace(RemoveNode=self.removed.append, GetNodeByID=lambda _i: None)
        self.mrmlScene = scene
        self.app = SimpleNamespace(layoutManager=lambda: None)
        self.util = SimpleNamespace(
            getNodesByClass=lambda _cls: list(self.nodes),
            loadVolume=self.load_volume,
            loadLabelVolume=self.load_label,
            setSliceViewerLayers=self.set_layers,
            resetSliceViews=lambda: None,
        )

    # O10: slicer.util.loadVolume returns the node and raises on failure (no returnNode argument).
    def load_volume(self, path, properties=None):
        node = FakeNode(properties["name"])
        self.nodes.append(node)
        return node

    def load_label(self, path, properties=None):
        if self.fail_label:
            raise RuntimeError("Failed to load node from file")
        node = FakeNode(properties["name"])
        self.nodes.append(node)
        return node

    def set_layers(self, **kwargs):
        self.layers = kwargs


class SceneTest(unittest.TestCase):
    def test_h1_scene_shows_mean_b0_with_mask_overlay_and_replaces_old_preview(self):
        fake = FakeSlicer()
        old = FakeNode("old")
        old.SetAttribute(PREVIEW_ATTRIBUTE, "meanb0")
        unrelated = FakeNode("user-volume")
        fake.nodes = [old, unrelated]
        info = show_preview_in_scene(fake, "/tmp/m.nii.gz", "/tmp/mask.nii.gz", subject_id="sub-01")
        self.assertEqual(fake.removed, [old])
        self.assertEqual(info, {"volume_id": "id-sub-01_input_meanb0", "mask_id": "id-sub-01_input_mask"})
        self.assertEqual(fake.layers["background"], info["volume_id"])
        self.assertEqual(fake.layers["label"], info["mask_id"])
        self.assertTrue(fake.layers["fit"])
        self.assertEqual(fake.nodes[-2].GetAttribute(PREVIEW_ATTRIBUTE), "meanb0")

    def test_h1_scene_without_mask_or_with_unloadable_mask_still_shows_the_volume(self):
        fake = FakeSlicer(fail_label=True)
        messages = []
        info = show_preview_in_scene(fake, "/tmp/m.nii.gz", "/tmp/mask.nii.gz", log=messages.append)
        self.assertIsNone(info["mask_id"])
        self.assertIsNone(fake.layers["label"])
        self.assertTrue(messages and "mask" in messages[0])
        self.assertIsNone(show_preview_in_scene(FakeSlicer(), "/tmp/m.nii.gz")["mask_id"])


class FakeDisplay:
    def __init__(self):
        self.component = 0
        self.auto = None

    def SetDiffusionComponent(self, value):
        self.component = int(value)

    def GetDiffusionComponent(self):
        return self.component

    def SetAutoWindowLevel(self, value):
        self.auto = value


class FakeDwiNode(FakeNode):
    def __init__(self, name, cls):
        super().__init__(name)
        self.cls, self.gradients, self.bvalues = cls, {}, {}
        self.count = self.ijk = self.frame = self.image = self.display = None

    def SetIJKToRASMatrix(self, matrix):
        self.ijk = matrix

    def SetAndObserveImageData(self, image):
        self.image = image

    def SetNumberOfGradients(self, count):
        self.count = count

    def GetNumberOfGradients(self):
        return self.count

    def SetDiffusionGradient(self, index, vector):
        self.gradients[index] = list(vector)

    def SetBValue(self, index, value):
        self.bvalues[index] = value

    def GetBValue(self, index):
        return self.bvalues[index]

    def SetMeasurementFrameMatrix(self, matrix):
        self.frame = matrix

    def CreateDefaultDisplayNodes(self):
        self.display = FakeDisplay()

    def GetDisplayNode(self):
        return self.display


class FakeDwiSlicer(FakeSlicer):
    """FakeSlicer plus AddNewNodeByClass; RemoveNode takes the node out of the scene."""

    def __init__(self, fail_label=False):
        super().__init__(fail_label)

        def remove(node):
            self.removed.append(node)
            if node in self.nodes:
                self.nodes.remove(node)

        def add(cls, name):
            node = FakeDwiNode(name, cls)
            self.nodes.append(node)
            return node

        self.mrmlScene.RemoveNode = remove
        self.mrmlScene.AddNewNodeByClass = add
        self.util.vtkMatrixFromArray = lambda array: [list(map(float, row)) for row in array]


def fake_full_dwi(bvalues=(1000, 0, 2000, 5), image="IMAGE"):
    from shared import input_preview as ip

    count = len(bvalues)
    gradients = [[0.0, 0.0, 0.0] if b <= 50 else [0.0, 0.6, 0.8] for b in bvalues]
    return ip.FullDwi(image=image, ijk_to_ras=[[2.0, 0, 0, 1.0], [0, 2.5, 0, 2.0], [0, 0, 3.0, 3.0], [0, 0, 0, 1]],
                      gradients=gradients, bvalues=[float(b) for b in bvalues],
                      measurement_frame=[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                      voxel_bytes=count * 100, scalar_type="short")


class DwiSceneTest(unittest.TestCase):
    """b2-dwi: the GUI thread only creates the core DWI node, sets gradients and b-values, and shows it."""

    def test_b2_dwi_scene_builds_a_core_dwi_node_with_gradients_and_default_component(self):
        from shared import input_preview as ip

        fake = FakeDwiSlicer()
        info = ip.show_dwi_in_scene(fake, fake_full_dwi(), "/tmp/mask.nii.gz", subject_id="sub-01")
        node = next(n for n in fake.nodes if isinstance(n, FakeDwiNode))
        self.assertEqual(node.cls, "vtkMRMLDiffusionWeightedVolumeNode")
        self.assertEqual(node.GetAttribute(PREVIEW_ATTRIBUTE), "dwi")
        self.assertEqual(node.image, "IMAGE")
        self.assertEqual(node.ijk, [[2.0, 0, 0, 1.0], [0, 2.5, 0, 2.0], [0, 0, 3.0, 3.0], [0, 0, 0, 1]])
        self.assertEqual(node.count, 4)
        self.assertEqual(node.gradients[2], [0.0, 0.6, 0.8])
        self.assertEqual(node.gradients[1], [0.0, 0.0, 0.0])
        self.assertEqual([node.bvalues[i] for i in range(4)], [1000.0, 0.0, 2000.0, 5.0])
        self.assertEqual(node.frame, [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.assertEqual(node.display.component, 1)  # the first volume with b <= 50
        self.assertEqual(node.display.auto, 1)
        self.assertEqual(info["volume_id"], node.GetID())
        self.assertEqual((info["component"], info["count"], info["b_value"]), (1, 4, 0.0))
        # The brain-mask overlay stays.
        self.assertEqual(info["mask_id"], "id-sub-01_input_mask")
        self.assertEqual(fake.layers["background"], node.GetID())
        self.assertEqual(fake.layers["label"], info["mask_id"])
        self.assertIsNone(fake.layers["foreground"])

    def test_b2_dwi_second_view_replaces_the_preview_instead_of_adding_one(self):
        from shared import input_preview as ip

        fake = FakeDwiSlicer()
        old_mean = FakeNode("old-mean")
        old_mean.SetAttribute(PREVIEW_ATTRIBUTE, "meanb0")
        user = FakeNode("user-volume")
        fake.nodes = [old_mean, user]
        first = ip.show_dwi_in_scene(fake, fake_full_dwi(), None, subject_id="s")
        second = ip.show_dwi_in_scene(fake, fake_full_dwi(), None, subject_id="s")
        dwi_nodes = [n for n in fake.nodes if n.GetAttribute(PREVIEW_ATTRIBUTE) == "dwi"]
        self.assertEqual(len(dwi_nodes), 1)
        self.assertIn(old_mean, fake.removed)
        self.assertIn(user, fake.nodes)
        self.assertIsNone(first["mask_id"])
        self.assertIsNotNone(second["volume_id"])
        ip.remove_preview_nodes(fake)
        self.assertEqual(fake.nodes, [user])

    def test_b2_dwi_a_failing_node_build_becomes_a_preview_error_and_leaves_no_node(self):
        from shared import input_preview as ip

        fake = FakeDwiSlicer()
        broken = fake_full_dwi()
        broken.gradients = broken.gradients[:2]  # fewer gradients than volumes
        with self.assertRaises(PreviewError) as caught:
            ip.show_dwi_in_scene(fake, broken, None, subject_id="s")
        self.assertEqual(caught.exception.code, "bad_data")
        self.assertEqual([n for n in fake.nodes if isinstance(n, FakeDwiNode)], [])


class FakeSpin:
    def __init__(self):
        self.value, self.minimum, self.maximum = 1, 1, 1
        self.enabled, self.blocked, self.emitted = False, False, []

    def setRange(self, low, high):
        self.minimum, self.maximum = low, high
        self.value = min(max(self.value, low), high)

    def setValue(self, value):
        value = min(max(int(value), self.minimum), self.maximum)
        if value != self.value:
            self.value = value
            if not self.blocked:
                self.emitted.append(value)

    def blockSignals(self, flag):
        old, self.blocked = self.blocked, bool(flag)
        return old

    def setEnabled(self, flag):
        self.enabled = bool(flag)


class FakeLabel:
    def __init__(self):
        self.text, self.visible, self.tooltip = "", True, ""

    def setText(self, value):
        self.text = value

    def setVisible(self, flag):
        self.visible = bool(flag)

    def setToolTip(self, value):
        self.tooltip = value


class StepperTest(unittest.TestCase):
    """b2-dwi item 6: the 'Volume i / N (b = ...)' stepper in the View input row."""

    def setUp(self):
        from Testing.Python.test_b2_fix_round import widget_methods

        self.scene = {}
        self.fake_slicer = SimpleNamespace(mrmlScene=SimpleNamespace(GetNodeByID=lambda i: self.scene.get(i)))
        fake_vtk = SimpleNamespace(vtkCommand=SimpleNamespace(ModifiedEvent=33))
        Widget = widget_methods(
            "_input_preview_dwi_node", "_refresh_input_preview_stepper", "_on_input_preview_component_changed",
            "_on_input_preview_display_modified", slicer=self.fake_slicer, vtk=fake_vtk,
        )
        me = Widget()
        me.observers = []
        me.addObserver = lambda obj, event, method: me.observers.append((obj, event, method.__name__))
        me.removeObserver = lambda obj, event, method: me.observers.remove((obj, event, method.__name__))
        me._onboardingLanguage = "en"
        me._inputPreviewWidgets = {"stepper": FakeLabel(), "spin": FakeSpin(), "info": FakeLabel(),
                                   "stepper_label": FakeLabel()}
        self.me = me

    def add_node(self, bvalues=(1000, 0, 2000, 5)):
        from shared import input_preview as ip

        fake = FakeDwiSlicer()
        info = ip.show_dwi_in_scene(fake, fake_full_dwi(bvalues), None, subject_id="s")
        node = fake.nodes[-1]
        self.scene[info["volume_id"]] = node
        self.me._inputPreviewDwiNodeId = info["volume_id"]
        return node

    def test_b2_dwi_stepper_is_disabled_and_hidden_without_a_dwi_preview(self):
        self.me._refresh_input_preview_stepper()
        widgets = self.me._inputPreviewWidgets
        self.assertFalse(widgets["spin"].enabled)
        self.assertFalse(widgets["stepper"].visible)
        self.assertEqual(self.me.observers, [])

    def test_b2_dwi_stepper_shows_volume_count_and_b_value_and_drives_the_display_node(self):
        node = self.add_node()
        self.me._refresh_input_preview_stepper()
        widgets = self.me._inputPreviewWidgets
        spin = widgets["spin"]
        self.assertTrue(spin.enabled)
        self.assertTrue(widgets["stepper"].visible)
        self.assertEqual((spin.minimum, spin.maximum, spin.value), (1, 4, 2))  # component 1, counted from 1
        self.assertEqual(widgets["info"].text, "/ 4  (b = 0)")
        self.assertEqual(spin.emitted, [])  # the refresh does not echo into the node
        self.assertEqual(len(self.me.observers), 1)
        self.me._on_input_preview_component_changed(3)
        self.assertEqual(node.display.component, 2)
        self.me._on_input_preview_display_modified(node.display, "ModifiedEvent")
        self.assertEqual(widgets["info"].text, "/ 4  (b = 2000)")
        self.assertEqual(len(self.me.observers), 1)  # the observer is added once

    def test_b2_dwi_stepper_follows_a_change_made_in_the_volumes_module(self):
        node = self.add_node()
        self.me._refresh_input_preview_stepper()
        node.display.SetDiffusionComponent(3)  # Volumes > DWI Component
        self.me._on_input_preview_display_modified(node.display, "ModifiedEvent")
        spin = self.me._inputPreviewWidgets["spin"]
        self.assertEqual(spin.value, 4)
        self.assertEqual(spin.emitted, [])
        self.assertEqual(self.me._inputPreviewWidgets["info"].text, "/ 4  (b = 5)")

    def test_b2_dwi_stepper_resets_when_the_node_is_removed_or_replaced(self):
        first = self.add_node()
        self.me._refresh_input_preview_stepper()
        self.scene.clear()  # node removed or scene closed
        self.me._refresh_input_preview_stepper()
        self.assertFalse(self.me._inputPreviewWidgets["spin"].enabled)
        self.assertFalse(self.me._inputPreviewWidgets["stepper"].visible)
        self.assertEqual(self.me.observers, [])
        self.assertIsNone(self.me._inputPreviewDwiNodeId)
        self.me._on_input_preview_component_changed(2)  # no node: nothing to drive, no error
        second = self.add_node((0, 1000))
        self.me._refresh_input_preview_stepper()
        self.assertEqual(self.me._inputPreviewWidgets["spin"].maximum, 2)
        self.assertEqual([o[0] for o in self.me.observers], [second.display])
        self.assertIsNot(first.display, second.display)

    def test_b2_inputs_stepper_is_hidden_in_batch_mode_and_comes_back_in_single_mode(self):
        node = self.add_node()
        self.me._refresh_input_preview_stepper()
        widgets = self.me._inputPreviewWidgets
        self.assertTrue(widgets["stepper"].visible and widgets["spin"].enabled)
        self.me._parameterNode = SimpleNamespace(batchMode=True)
        self.me._refresh_input_preview_stepper()  # every refresh keeps it hidden, even though the node exists
        self.me._refresh_input_preview_stepper()
        self.assertFalse(widgets["stepper"].visible)
        self.assertFalse(widgets["spin"].enabled)
        self.assertIsNotNone(self.me._input_preview_dwi_node())  # the preview stays in the scene
        node.display.SetDiffusionComponent(3)
        self.me._parameterNode = SimpleNamespace(batchMode=False)
        self.me._refresh_input_preview_stepper()
        self.assertTrue(widgets["stepper"].visible)
        self.assertTrue(widgets["spin"].enabled)
        self.assertEqual(widgets["spin"].value, 4)  # the state that was kept while hidden
        self.assertEqual(widgets["info"].text, "/ 4  (b = 5)")

    def test_b2_dwi_stepper_ignores_a_preview_that_is_not_a_full_dwi(self):
        mean = FakeNode("mean")
        mean.SetAttribute(PREVIEW_ATTRIBUTE, "meanb0")
        self.scene["id-mean"] = mean
        self.me._inputPreviewDwiNodeId = "id-mean"
        self.me._refresh_input_preview_stepper()
        self.assertFalse(self.me._inputPreviewWidgets["stepper"].visible)

    def test_b2_dwi_stepper_widgets_are_installed_hidden_in_the_preview_row(self):
        from Testing.Python.test_ui_run_history import by_name, fake_qt
        from shared import ui_compact

        qt = fake_qt()
        layout = mock.MagicMock(name="configurationLayout")
        calls = []
        owner = SimpleNamespace(ui=SimpleNamespace(configurationLayout=layout), _onboardingLanguage="en",
                                _on_input_preview_clicked=lambda: None,
                                _on_input_preview_component_changed=calls.append,
                                _refresh_input_preview_controls=lambda: None)
        with mock.patch.dict(sys.modules, {"qt": qt}), \
                mock.patch("shared.ui_prepare_inputs.place_input_preview_button", lambda _o: None):
            ui_compact.install_input_preview_controls(owner)
        widgets = owner._inputPreviewWidgets
        stepper = by_name(qt, "inputPreviewStepper")
        spin = by_name(qt, "inputPreviewComponentSpinBox")
        self.assertIs(widgets["stepper"], stepper)
        self.assertIs(widgets["spin"], spin)
        self.assertIs(widgets["info"], by_name(qt, "inputPreviewStepperInfo"))
        stepper.hide.assert_called()
        spin.setEnabled.assert_called_with(False)
        signal, callback = spin.connect.call_args[0]
        self.assertEqual(signal, "valueChanged(int)")
        callback(5)
        self.assertEqual(calls, [5])
        row_layout = widgets["layout"]
        added = [call[1][0] for call in row_layout.method_calls if call[0] == "addWidget"]
        self.assertEqual(added, [stepper, widgets["status"]])  # stepper first, then the status line

    def test_b2_dwi_stepper_texts_are_bilingual(self):
        from shared import ui_compact

        button, label, spin = FakeLabel(), FakeLabel(), FakeLabel()
        owner = SimpleNamespace(_inputPreviewWidgets={"button": button, "stepper_label": label, "spin": spin},
                                _onboardingLanguage="en")
        button.setAccessibleName = lambda _v: None
        ui_compact.refresh_input_preview_controls(owner)
        self.assertEqual(label.text, "Volume")
        self.assertIn("Volumes", spin.tooltip)
        self.assertIn("full DWI", button.tooltip)  # no longer "the mean b0 image"
        ui_compact.refresh_input_preview_controls(owner, "zh")
        self.assertEqual(label.text, "体积")
        self.assertIn("DWI", spin.tooltip)


class ClickFlowTest(unittest.TestCase):
    """b2-dwi: the click handler builds on the worker and shows the full DWI or the size fallback."""

    def run_click(self, prepared, zh=False, batch=False):
        from Testing.Python.test_b2_fix_round import widget_methods
        from shared import input_preview as ip

        queue = []
        fake_qt = SimpleNamespace(QTimer=SimpleNamespace(singleShot=lambda _ms, fn: queue.append(fn)))
        fake_slicer = SimpleNamespace(name="slicer")
        Widget = widget_methods("_on_input_preview_clicked", qt=fake_qt, slicer=fake_slicer)
        me = Widget()
        source = PreviewSource("nifti", Path("/x/data.nii.gz"), Path("/x/bvals"), Path("/x/mask.nii.gz"), "sub",
                               bvec=Path("/x/bvecs"))
        statuses, logs, events = [], [], []
        me._onboardingLanguage = "zh" if zh else "en"
        me._inputPreviewPending = me._inputImportPending = me._cleanedUp = False
        me._resolve_input_preview_source = lambda: source
        me._input_preview_key = lambda s=None: ("source", str(getattr(s, "dwi", "")))
        me._input_preview_is_batch = lambda: False
        me._parameterNode = SimpleNamespace(batchMode=batch)
        me._input_preview_failed = lambda message, key=None: statuses.append(("error", message))
        me._set_input_preview_status = lambda text, error=False, key=None: statuses.append(("ok", text))
        me._refresh_input_preview_controls = lambda: None
        me._refresh_input_preview_stepper = lambda: events.append("stepper")
        me._input_preview_cache_dir = lambda: Path("/x/cache")
        me.appendLog = logs.append
        shown = {}

        def show_dwi(slicer, dwi, mask, subject_id="", log=None):
            shown.update(kind="dwi", dwi=dwi, mask=mask)
            return {"volume_id": "vtkMRMLDiffusionWeightedVolumeNode1", "mask_id": "m", "component": 1,
                    "count": 4, "b_value": 0.0}

        def show_mean(slicer, path, mask, subject_id="", log=None):
            shown.update(kind="meanb0", path=path)
            return {"volume_id": "vtkMRMLScalarVolumeNode1", "mask_id": None}

        with mock.patch.object(ip, "check_preview_dependencies", lambda s=None: None), \
                mock.patch.object(ip, "prepare_input_preview", lambda s, cache: prepared), \
                mock.patch.object(ip, "show_dwi_in_scene", show_dwi), \
                mock.patch.object(ip, "show_preview_in_scene", show_mean), \
                mock.patch.object(ip, "remove_preview_nodes", lambda s: events.append("removed")):
            me._on_input_preview_clicked()
            for _ in range(2000):
                if not queue:
                    break
                import time

                time.sleep(0.001)
                queue.pop(0)()
        return me, shown, statuses, logs, events

    def test_b2_dwi_click_shows_the_full_dwi_and_enables_the_stepper(self):
        from shared import input_preview as ip

        dwi = fake_full_dwi()
        prepared = ip.PreparedPreview("dwi", dwi=dwi, estimated_bytes=400, limit_bytes=4 << 30)
        me, shown, statuses, logs, events = self.run_click(prepared)
        self.assertEqual(shown["kind"], "dwi")
        self.assertIs(shown["dwi"], dwi)
        self.assertEqual(shown["mask"], Path("/x/mask.nii.gz"))
        self.assertEqual(me._inputPreviewDwiNodeId, "vtkMRMLDiffusionWeightedVolumeNode1")
        self.assertEqual(events[0], "removed")  # the old preview goes before the new one is read
        self.assertEqual(events[-1], "stepper")
        self.assertEqual(statuses[0], ("ok", "Loading the full DWI…"))
        self.assertIn("full DWI of sub", statuses[-1][1])
        self.assertIn("4 volumes", statuses[-1][1])
        self.assertNotIn("mean b0", statuses[-1][1])

    def test_b2_inputs_click_does_nothing_in_batch_mode_even_when_called(self):
        from shared import input_preview as ip

        prepared = ip.PreparedPreview("dwi", dwi=fake_full_dwi(), estimated_bytes=400, limit_bytes=4 << 30)
        me, shown, statuses, logs, events = self.run_click(prepared, batch=True)
        self.assertEqual((shown, statuses, logs, events), ({}, [], [], []))
        self.assertFalse(me._inputPreviewPending)
        me, shown, statuses, logs, events = self.run_click(prepared, batch=False)  # single mode still works
        self.assertEqual(shown["kind"], "dwi")

    def test_b2_dwi_click_above_the_size_limit_shows_the_mean_b0_with_the_notice(self):
        from shared import input_preview as ip

        prepared = ip.PreparedPreview("meanb0", mean_b0=Path("/x/cache/m.nii.gz"),
                                      estimated_bytes=6 << 30, limit_bytes=4 << 30)
        me, shown, statuses, logs, events = self.run_click(prepared)
        self.assertEqual(shown, {"kind": "meanb0", "path": Path("/x/cache/m.nii.gz")})
        self.assertIsNone(me._inputPreviewDwiNodeId)
        self.assertEqual(statuses[-1][0], "ok")
        self.assertIn(ip.size_notice(6 << 30, 4 << 30), statuses[-1][1])
        _me, _shown, zh_statuses, _logs, _events = self.run_click(prepared, zh=True)
        self.assertIn(ip.size_notice(6 << 30, 4 << 30, zh=True), zh_statuses[-1][1])

    def test_b2_addon_gradient_fallback_shows_its_own_notice_not_the_size_notice(self):
        from shared import input_preview as ip

        prepared = ip.PreparedPreview("meanb0", mean_b0=Path("/x/cache/m.nii.gz"), estimated_bytes=300,
                                      limit_bytes=4 << 30, reason="no_gradients")
        me, shown, statuses, logs, events = self.run_click(prepared)
        self.assertEqual(shown["kind"], "meanb0")
        self.assertEqual(statuses[-1][0], "ok")
        self.assertIn(ip.gradient_notice(False), statuses[-1][1])
        self.assertNotIn("GiB", statuses[-1][1])
        self.assertIn("mean b0 image of sub", statuses[-1][1])
        _me, _shown, zh_statuses, _logs, _events = self.run_click(prepared, zh=True)
        self.assertIn(ip.gradient_notice(True), zh_statuses[-1][1])

    def test_b2_dwi_click_failure_is_reported_and_leaves_no_stepper(self):
        from shared import input_preview as ip

        class Broken:  # a result without a DWI makes finish() fail
            kind = "dwi"

        self.assertTrue(hasattr(ip, "PreparedPreview"))
        me, shown, statuses, logs, events = self.run_click(Broken())
        self.assertEqual(statuses[-1][0], "error")
        self.assertIsNone(me._inputPreviewDwiNodeId)
        self.assertEqual(events[-1], "stepper")

    def test_b2_dwi_scene_close_and_node_removal_refresh_the_stepper(self):
        widget = (REPO_ROOT / "shared" / "module_widget.py").read_text(encoding="utf-8")
        tree = ast.parse(widget)
        end_close = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "onSceneEndClose")
        self.assertIn("_queue_input_preview_stepper_refresh", ast.unparse(end_close))
        self.assertIn("NodeRemovedEvent, self._queue_input_preview_stepper_refresh", widget)


class WiringTest(unittest.TestCase):
    def test_h1_module_is_python39_syntax_clean(self):
        source = (REPO_ROOT / "shared" / "input_preview.py").read_text(encoding="utf-8")
        ast.parse(source, feature_version=(3, 9))

    def test_h1_compact_language_refresh_sets_bilingual_label_and_tooltip(self):
        from shared.ui_compact import refresh_input_preview_controls

        class Button:
            def setText(self, value):
                self.text = value

            def setToolTip(self, value):
                self.tooltip = value

            def setAccessibleName(self, value):
                self.accessible = value

        button = Button()
        owner = SimpleNamespace(_inputPreviewWidgets={"button": button}, _onboardingLanguage="en")
        refresh_input_preview_controls(owner)
        self.assertEqual(button.text, "View input")
        refresh_input_preview_controls(owner, "zh")
        self.assertEqual(button.text, "预览输入")
        self.assertIn("b0", button.tooltip)
        refresh_input_preview_controls(SimpleNamespace())  # no widgets: no-op

    def test_h1_widget_refresh_disables_button_with_reason_and_follows_pending_import(self):
        source_path = REPO_ROOT / "shared" / "module_widget.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        wanted = {"_refresh_input_preview_controls"}
        methods = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in wanted]
        self.assertEqual(len(methods), 1)
        namespace = {"Path": Path}
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(source_path), "exec"), namespace)
        refresh = namespace["_refresh_input_preview_controls"]

        class Button:
            enabled = True
            tooltip = ""

            def setEnabled(self, value):
                self.enabled = value

            def setToolTip(self, value):
                self.tooltip = value

        def owner(resolve, **flags):
            button = Button()
            return SimpleNamespace(
                _inputPreviewWidgets={"button": button}, _onboardingLanguage="en", _parameterNode=object(),
                _set_input_preview_status=lambda text, **kw: None,
                _resolve_input_preview_source=resolve, **flags), button

        def dicom():
            raise PreviewError("dicom_unconverted")

        me, button = owner(dicom)
        refresh(me)
        self.assertFalse(button.enabled)
        self.assertEqual(button.tooltip, "Convert inputs first (Prepare Inputs)")
        me, button = owner(lambda: None)
        with mock.patch("importlib.util.find_spec", return_value=object()):  # O11: the host may lack nibabel
            refresh(me)
        self.assertTrue(button.enabled)
        me, button = owner(lambda: None, _inputImportPending=True)
        refresh(me)
        self.assertFalse(button.enabled)
        calls = []
        me, button = owner(lambda: calls.append("resolved"))
        me._parameterNode = SimpleNamespace(batchMode=True)  # b2 inputs-help item 3: Batch mode has no preview
        with mock.patch("importlib.util.find_spec", return_value=object()):
            refresh(me)
        self.assertFalse(button.enabled)
        self.assertEqual(calls, [])  # nothing is resolved or scanned in Batch mode

    def test_h1_button_is_installed_in_inputs_section_and_refreshed_from_run_state(self):
        compact = (REPO_ROOT / "shared" / "ui_compact.py").read_text(encoding="utf-8")
        self.assertIn("install_input_preview_controls(owner)", compact)
        self.assertIn("configurationLayout", compact)
        widget = (REPO_ROOT / "shared" / "module_widget.py").read_text(encoding="utf-8")
        self.assertIn("self._refresh_input_preview_controls()", widget)
        self.assertIn("def _on_input_preview_clicked", widget)

    def test_input_drop_routing_tile_border_and_preview_placement(self):
        from Testing.Python.test_ui_run_history import fake_qt
        from shared import ui_prepare_inputs as prep

        assigned: dict[str, str] = {}
        shown: list[str] = []

        class Owner:
            _parameterNode = SimpleNamespace(batchMode=False)

            def _prepare_inputs_widget_text(self, key):
                return assigned.get(key, "")

            def _set_prepare_inputs_widget_text(self, key, value):
                assigned[key] = value

        owner = Owner()
        with mock.patch.object(prep, "show_input_format", side_effect=lambda _o, k: shown.append(k)):
            ok = prep._assign_dropped_file_paths(
                owner, "nifti_source",
                [Path("/tmp/sub/dwi.nii.gz"), Path("/tmp/sub/dwi.bval"), Path("/tmp/sub/dwi.bvec")],
            )
            self.assertTrue(ok)
            self.assertEqual(shown, ["nifti"])
            self.assertEqual(assigned["source"], str(Path("/tmp/sub/dwi.nii.gz")))
            self.assertEqual(assigned["bval"], str(Path("/tmp/sub/dwi.bval")))
            self.assertEqual(assigned["bvec"], str(Path("/tmp/sub/dwi.bvec")))

            assigned.clear()
            ok_mask = prep._assign_dropped_file_paths(owner, "mask", [Path("/tmp/sub/mask.nii.gz")])
            self.assertTrue(ok_mask)
            self.assertEqual(assigned, {"mask": str(Path("/tmp/sub/mask.nii.gz"))})

            assigned.clear()
            ok_raw = prep._assign_dropped_file_paths(owner, "raw", [Path("/tmp/sub/volume.img")])
            self.assertTrue(ok_raw)
            self.assertEqual(assigned, {"raw": str(Path("/tmp/sub/volume.img"))})

        class FakeLayout:
            def __init__(self, initial=()):
                self.items = list(initial)

            def indexOf(self, w):
                return self.items.index(w) if w in self.items else -1

            def addWidget(self, w, *_a):
                self.items.append(w)

            def insertWidget(self, idx, w, *_a):
                self.items.insert(idx, w)

            def removeWidget(self, w):
                if w in self.items:
                    self.items.remove(w)

        qt = fake_qt()
        class Shown:
            visible = True

            def setVisible(self, flag):
                self.visible = bool(flag)

        btn, row = Shown(), Shown()
        header = FakeLayout()
        row_layout = FakeLayout()
        me = SimpleNamespace(
            _parameterNode=SimpleNamespace(batchMode=False),
            _prepareInputWidgets={"format_header": header},
            _inputPreviewWidgets={"button": btn, "layout": row_layout, "row": row},
        )
        with mock.patch.dict(sys.modules, {"qt": qt}):
            prep.place_input_preview_button(me)
            self.assertEqual(header.items, [btn])
            self.assertEqual(row_layout.items, [])
            self.assertTrue(btn.visible and row.visible)
            # b2 inputs-help item 3: Batch mode hides the button and its row; it no longer moves below Batch Folder.
            prep.place_input_preview_button(me, batch=True)
            self.assertEqual(header.items, [btn])
            self.assertEqual(row_layout.items, [])
            self.assertFalse(btn.visible or row.visible)
            prep.place_input_preview_button(me, batch=False)
            self.assertTrue(btn.visible and row.visible)

        prepare_source = (REPO_ROOT / "shared" / "ui_prepare_inputs.py").read_text(encoding="utf-8")
        self.assertIn("border: 2px dashed", prepare_source)
        self.assertNotIn("border: 1px dashed", prepare_source)

        class DummyQFrame:
            NoFrame = 0

            def __init__(self):
                self._props = {}
                self.frame_shape = None
                self.updated = False

            def property(self, name):
                return self._props.get(name)

            def setProperty(self, name, value):
                self._props[name] = value

            def setFrameShape(self, shape):
                self.frame_shape = shape

            def update(self):
                self.updated = True

        class DummyEvent:
            def __init__(self, urls=()):
                self._urls = list(urls)
                self.accepted = False
                self.ignored = False

            def mimeData(self):
                return SimpleNamespace(
                    hasUrls=lambda: bool(self._urls),
                    urls=lambda: list(self._urls),
                )

            def acceptProposedAction(self):
                self.accepted = True

            def ignore(self):
                self.ignored = True

        local_url = SimpleNamespace(isLocalFile=lambda: True, toLocalFile=lambda: "/tmp/sub/drop_dwi.nii.gz")
        assigned.clear()
        shown.clear()
        with mock.patch.object(prep, "show_input_format", side_effect=lambda _o, k: shown.append(k)):
            tile, handles_drops = prep._make_input_drop_tile(
                SimpleNamespace(QFrame=DummyQFrame), owner, "nifti_source"
            )
            self.assertTrue(handles_drops)
            self.assertEqual(tile.frame_shape, DummyQFrame.NoFrame)

            enter_ev = DummyEvent([local_url])
            tile.dragEnterEvent(enter_ev)
            self.assertTrue(enter_ev.accepted)
            self.assertTrue(tile.property("dragActive"))

            tile.dragLeaveEvent(DummyEvent())
            self.assertFalse(tile.property("dragActive"))

            drop_ev = DummyEvent([local_url])
            tile.dropEvent(drop_ev)
            self.assertTrue(drop_ev.accepted)
            self.assertFalse(tile.property("dragActive"))
            self.assertEqual(shown, ["nifti"])
            self.assertEqual(assigned["source"], str(Path("/tmp/sub/drop_dwi.nii.gz")))


if __name__ == "__main__":
    unittest.main()
