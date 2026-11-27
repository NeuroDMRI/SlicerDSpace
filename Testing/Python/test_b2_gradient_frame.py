#!/usr/bin/env python3
"""Gradient frame (batch 2): the world (RAS) frame of diffusion gradient directions on every route.

Rule: for a NIfTI with affine 3 x 3 part A and FSL bvecs b, R = A with each column normalised, F = diag(-1, 1, 1)
when det(A) > 0 (else the identity), and the world direction is g = R F b (MRtrix ``-fslgrad``). An NRRD header
stores the measurement frame vectors as columns: world = S MF g_file (S maps the header space to RAS).

Each route test reads back the WORLD directions and compares them with the rule (angle below 0.01 degree,
g and -g are the same axis). The "unchanged" tests compare with outputs of base commit 3c912ad2: the literals
were captured with that commit's code on the same fixtures.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

C20, S20 = math.cos(math.radians(20.0)), math.sin(math.radians(20.0))
U20 = np.array([[C20, -S20, 0.0], [S20, C20, 0.0], [0.0, 0.0, 1.0]])
H = np.diag([-1.0, 1.0, 1.0])
LPS = np.diag([-1.0, -1.0, 1.0])
PERM = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # det -1: i -> P, j -> S, k -> R
VOX = np.diag([2.0, 2.5, 3.0])  # anisotropic voxels

GEOMETRIES = {
    "axial_det_neg_LAS": H @ VOX,
    "axial_det_pos_RAS": VOX.copy(),
    "oblique20_det_neg": U20 @ H @ VOX,
    "oblique20_det_pos": U20 @ VOX,
    "permuted_det_neg": PERM @ VOX,
    "permuted_det_pos": PERM @ H @ VOX,
}
CHANGED = [name for name in GEOMETRIES if name != "axial_det_neg_LAS"]
MAX_ANGLE_DEG = 0.01
UKF_POLICY_V1 = "official-gradient-geometry-probe-with-lossless-native-payload-v1"
UKF_TEXT_V1 = "NIfTI RAS+, gradients converted by official DWIConvert"


# ----------------------------------------------------------------------------- fixtures (base-commit safe)
def reference_world(A3, bvecs):
    """The rule, written here independently of shared/gradient_frame.py."""
    A3 = np.asarray(A3, float)
    R = A3 / np.linalg.norm(A3, axis=0)
    F = H if np.linalg.det(A3) > 0 else np.eye(3)
    return (R @ F @ np.asarray(bvecs, float).T).T


def axis_angles(first, second):
    first, second = np.asarray(first, float), np.asarray(second, float)
    cosine = np.abs(np.sum(first * second, axis=1)) / (np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1))
    return np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0)))


def gradient_set(seed=11, b0=2, per_shell=12, shells=(1000,)):
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(per_shell * len(shells), 3))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    bvecs = np.vstack([np.zeros((b0, 3)), vectors])
    bvals = np.array([0.0] * b0 + [float(b) for b in shells for _ in range(per_shell)])
    return bvals, bvecs


def affine4(A3, origin=(-10.0, -12.0, 5.0)):
    A = np.eye(4)
    A[:3, :3] = A3
    A[:3, 3] = origin
    return A


def write_fsl_case(folder, A3, bvals, bvecs, shape=(4, 3, 5), seed=3):
    import nibabel as nib

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    data = rng.uniform(100.0, 1000.0, size=shape + (len(bvals),)).astype(np.float32)
    A = affine4(A3)
    image = nib.Nifti1Image(data, A)
    image.set_qform(A, 1)
    image.set_sform(A, 1)
    nib.save(image, str(folder / "data.nii.gz"))
    nib.save(nib.Nifti1Image(np.ones(shape, np.uint8), A), str(folder / "mask.nii.gz"))
    (folder / "bvals").write_text(" ".join("%g" % value for value in bvals) + "\n", encoding="utf-8")
    (folder / "bvecs").write_text(
        "\n".join(" ".join("%.10f" % value for value in bvecs[:, axis]) for axis in range(3)) + "\n",
        encoding="utf-8")
    return types.SimpleNamespace(
        subject_id="gradframe", dwi_path=folder / "data.nii.gz", mask_path=folder / "mask.nii.gz",
        bval_path=folder / "bvals", bvec_path=folder / "bvecs", affine=A)


def dwiconvert_fsl_to_nrrd_values(A3, bvals, bvecs):
    """DWIConvert 5.8 FSLToNrrd gradient values (measured on Slicer 5.12.4; BRAINSTools FSLDWIConverter).

    Space LPS, measurement frame identity, value = L P b * sqrt(b / max b) with P_ii = -1 when R_ii < -0.5
    (fitted on LAS, RPS, RAI, LPI, RAS and LPS storages). The rotation and the axis permutation of R are ignored,
    and no det-dependent flip is applied.
    """
    A3 = np.asarray(A3, float)
    R = A3 / np.linalg.norm(A3, axis=0)
    P = np.diag([-1.0 if R[i, i] < -0.5 else 1.0 for i in range(3)])
    values = []
    for b, vector in zip(bvals, bvecs):
        scaled = LPS @ P @ vector * math.sqrt(b / max(bvals))
        values.append("0   0   0" if not np.any(scaled) else "   ".join(repr(float(x)) for x in scaled))
    return values


def write_dwiconvert_probe_output(raw, probe, target, A3, bvals, bvecs):
    """The converted geometry probe as DWIConvert writes it (no axis change, LPS header)."""
    import nrrd

    A = raw.affine
    header = {
        "space": "left-posterior-superior",
        "space directions": np.vstack([(LPS @ A[:3, :3]).T, [np.nan] * 3]),
        "space origin": LPS @ A[:3, 3],
        "kinds": ["domain", "domain", "domain", "list"],
        "DWMRI_b-value": "%g" % max(bvals),
        "measurement frame": np.eye(3),
        "modality": "DWMRI",
    }
    for index, value in enumerate(dwiconvert_fsl_to_nrrd_values(A3, bvals, bvecs)):
        header["DWMRI_gradient_%04d" % index] = value
    nrrd.write(str(target), probe, header)


def run_ukf_restore(folder, A3, bvals, bvecs):
    from tractography import ukf

    raw = write_fsl_case(folder, A3, bvals, bvecs)
    probe = ukf.write_conversion_probe(raw, Path(folder) / "geometry_probe.nii.gz")
    converted = Path(folder) / "geometry_probe.nrrd"
    write_dwiconvert_probe_output(raw, probe, converted, A3, bvals, bvecs)
    dwi, mask = Path(folder) / "DWI.nrrd", Path(folder) / "Mask.nrrd"
    report = ukf.restore_lossless_payload(raw, probe, converted, dwi, mask)
    return raw, dwi, mask, report


def nrrd_world(path):
    """World (RAS) directions of every gradient of an NRRD header (pynrrd rows are the written vectors)."""
    import nrrd

    header = nrrd.read_header(str(path))
    frame = np.asarray(header["measurement frame"], float).T
    space = {"left-posterior-superior": LPS, "right-anterior-superior": np.eye(3)}[header["space"]]
    count = sum(1 for key in header if key.startswith("DWMRI_gradient_"))
    values = np.array([[float(x) for x in header["DWMRI_gradient_%04d" % i].split()] for i in range(count)])
    return (space @ frame @ values.T).T, values


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nrrd_digest(path):
    """sha256 of an attached NRRD file without pynrrd's write-time comment line and gzip time stamp.

    pynrrd writes "# on <date time>(GMT)." and gzip stores a time, so two runs of the same code differ there only.
    """
    import gzip

    content = Path(path).read_bytes()
    head, _, payload = content.partition(b"\n\n")
    lines = [line for line in head.split(b"\n") if not line.startswith(b"# on ")]
    if b"encoding: gzip" in lines:
        payload = gzip.decompress(payload)
    return hashlib.sha256(b"\n".join(lines) + b"\n\n" + payload).hexdigest()


# Base commit 3c912ad2 outputs on the fixtures above (captured with that commit's code).
BASE_UKF_LAS_MULTISHELL = {"DWI.nrrd": "23b9ace2b175fd0046aba52070a25eda6c069b1c1e734fea9b12e42caf0e7700", "Mask.nrrd": "2770a52bdd6f7733f65bcc32e300f6b51e18fed9f2941c91ebc359a0c8fb9457"}
BASE_NHDR = {"axial_det_neg_LAS": "8b820ad7271769dbd5881eef6d094c3485ee8aa2fa63b1fa39af260576819b30", "oblique20_det_neg": "e84b5e6d728420c82878b64fb3377ca9488d58a45267d20fbae073e23419db38"}


# ----------------------------------------------------------------------------- the rule itself
class GradientFrameRuleTests(unittest.TestCase):
    """Literal values of the rule (gradient frame) for each geometry class."""

    def setUp(self):
        from shared import gradient_frame

        self.gf = gradient_frame

    def assertDirections(self, actual, expected):
        np.testing.assert_allclose(np.asarray(actual, float), np.asarray(expected, float), atol=1e-12, rtol=0)

    def test_b2_gradient_frame_literal_world_directions(self):
        b = np.array([[1.0, 0.0, 0.0], [0.0, 0.6, 0.8], [0.6, 0.8, 0.0]])
        expected = {
            "axial_det_neg_LAS": [[-1.0, 0.0, 0.0], [0.0, 0.6, 0.8], [-0.6, 0.8, 0.0]],
            "axial_det_pos_RAS": [[-1.0, 0.0, 0.0], [0.0, 0.6, 0.8], [-0.6, 0.8, 0.0]],
            "oblique20_det_neg": [[-0.9396926207859084, -0.3420201433256687, 0.0],
                                  [-0.20521208599540122, 0.563815572471545, 0.8],
                                  [-0.8374316871320799, 0.5465420106333256, 0.0]],
            "oblique20_det_pos": [[-0.9396926207859084, -0.3420201433256687, 0.0],
                                  [-0.20521208599540122, 0.563815572471545, 0.8],
                                  [-0.8374316871320799, 0.5465420106333256, 0.0]],
            "permuted_det_neg": [[0.0, -1.0, 0.0], [0.8, 0.0, 0.6], [0.0, -0.6, 0.8]],
            "permuted_det_pos": [[0.0, -1.0, 0.0], [0.8, 0.0, 0.6], [0.0, -0.6, 0.8]],
        }
        for name, A3 in GEOMETRIES.items():
            with self.subTest(geometry=name):
                world = self.gf.fsl_to_world(b, affine4(A3))
                np.testing.assert_allclose(world, expected[name], atol=1e-9, rtol=0)
                self.assertTrue(np.all(axis_angles(world, reference_world(A3, b)) < 1e-9))

    def test_b2_gradient_frame_flip_only_for_positive_determinant(self):
        self.assertDirections(self.gf.fsl_flip(GEOMETRIES["axial_det_neg_LAS"]), np.eye(3))
        self.assertDirections(self.gf.fsl_flip(GEOMETRIES["axial_det_pos_RAS"]), H)
        self.assertDirections(self.gf.affine_rotation(GEOMETRIES["oblique20_det_neg"]), U20 @ H)
        b = np.array([[0.0, 0.0, 0.0], [0.6, 0.8, 0.0]])
        same = self.gf.voxel_axis_bvecs(b, GEOMETRIES["oblique20_det_neg"])
        self.assertEqual(same.tobytes(), b.tobytes())
        self.assertDirections(self.gf.voxel_axis_bvecs(b, GEOMETRIES["permuted_det_pos"]), [[0.0, 0.0, 0.0], [-0.6, 0.8, 0.0]])

    def test_b2_gradient_frame_inverse_round_trip(self):
        _, b = gradient_set(seed=5)
        for name, A3 in GEOMETRIES.items():
            with self.subTest(geometry=name):
                back = self.gf.world_to_fsl(self.gf.fsl_to_world(b, A3), A3)
                np.testing.assert_allclose(back, b, atol=1e-12, rtol=0)

    def test_b2_gradient_frame_nrrd_columns_and_space(self):
        self.assertDirections(self.gf.world_to_nrrd([[-0.6, 0.8, 0.0]], "left-posterior-superior", np.eye(3)),
                              [[0.6, -0.8, 0.0]])
        self.assertDirections(self.gf.nrrd_to_world([[0.6, -0.8, 0.0]], "left-posterior-superior", np.eye(3)),
                              [[-0.6, 0.8, 0.0]])
        columns = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])  # x -> y, y -> x, z -> -z
        self.assertDirections(self.gf.nrrd_to_world([[1.0, 0.0, 0.0]], "right-anterior-superior", columns),
                              [[0.0, 1.0, 0.0]])
        for name, A3 in GEOMETRIES.items():  # nhdr_write style: RAS space, MF = R, value = F b
            R = A3 / np.linalg.norm(A3, axis=0)
            _, b = gradient_set(seed=8)
            values = self.gf.world_to_nrrd(reference_world(A3, b), "right-anterior-superior", R)
            F = H if np.linalg.det(A3) > 0 else np.eye(3)
            np.testing.assert_allclose(values, (F @ b.T).T, atol=1e-12, rtol=0)

    def test_b2_gradient_frame_value_rewrite_keeps_norms_and_zero_lines(self):
        bvals, bvecs = gradient_set(seed=2, shells=(1000, 2000), per_shell=4)
        A3 = GEOMETRIES["oblique20_det_pos"]
        old = dwiconvert_fsl_to_nrrd_values(A3, bvals, bvecs)
        self.gf.check_nrrd_gradients_match_fsl(old, bvals, bvecs)
        new = self.gf.nrrd_gradient_values_for_world(old, self.gf.fsl_to_world(bvecs, A3),
                                                     "left-posterior-superior", np.eye(3))
        self.assertEqual(new[:2], ["0   0   0", "0   0   0"])
        old_norms = [np.linalg.norm([float(x) for x in v.split()]) for v in old]
        new_norms = [np.linalg.norm([float(x) for x in v.split()]) for v in new]
        np.testing.assert_allclose(new_norms, old_norms, atol=1e-12, rtol=0)
        np.testing.assert_allclose(new_norms[2:], np.sqrt(bvals[2:] / 2000.0), atol=1e-9, rtol=0)
        swapped = list(old)
        swapped[2], swapped[-1] = swapped[-1], swapped[2]  # a b = 1000 and a b = 2000 entry
        with self.assertRaisesRegex(ValueError, "order"):
            self.gf.check_nrrd_gradients_match_fsl(swapped, bvals, bvecs)
        with self.assertRaisesRegex(ValueError, "count"):
            self.gf.check_nrrd_gradients_match_fsl(old[:-1], bvals, bvecs)


# ----------------------------------------------------------------------------- UKF input header
class UkfGradientFrameTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def test_b2_gradient_frame_ukf_header_world_directions(self):
        bvals, bvecs = gradient_set(shells=(1000, 2000), per_shell=10)
        for name in CHANGED:
            with self.subTest(geometry=name):
                A3 = GEOMETRIES[name]
                _, dwi, _, _ = run_ukf_restore(self.root / name, A3, bvals, bvecs)
                world, values = nrrd_world(dwi)
                moving = bvals > 0
                self.assertLess(axis_angles(world[moving], reference_world(A3, bvecs)[moving]).max(), MAX_ANGLE_DEG)
                # The b-value scaling DWIConvert wrote (norm sqrt(b / max b)) is kept; b = 0 lines stay zero.
                np.testing.assert_allclose(np.linalg.norm(values, axis=1), np.sqrt(bvals / bvals.max()), atol=1e-7)

    def test_b2_gradient_frame_ukf_unchanged_axial_det_neg_is_byte_identical_to_base(self):
        bvals, bvecs = gradient_set(shells=(1000, 2000), per_shell=10)
        _, dwi, mask, _ = run_ukf_restore(self.root / "las", GEOMETRIES["axial_det_neg_LAS"], bvals, bvecs)
        self.assertEqual({"DWI.nrrd": nrrd_digest(dwi), "Mask.nrrd": nrrd_digest(mask)}, BASE_UKF_LAS_MULTISHELL)

    def test_b2_gradient_frame_ukf_gradient_order_mismatch_fails_clearly(self):
        from tractography import ukf

        bvals, bvecs = gradient_set(shells=(1000, 2000), per_shell=4)
        A3 = GEOMETRIES["oblique20_det_neg"]
        raw = write_fsl_case(self.root / "order", A3, bvals, bvecs)
        probe = ukf.write_conversion_probe(raw, self.root / "order" / "probe.nii.gz")
        converted = self.root / "order" / "probe.nrrd"
        write_dwiconvert_probe_output(raw, probe, converted, A3, bvals, bvecs)
        swapped_vals = bvals.copy()
        swapped_vals[[2, -1]] = swapped_vals[[-1, 2]]
        (self.root / "order" / "bvals").write_text(" ".join("%g" % v for v in swapped_vals) + "\n")
        with self.assertRaisesRegex(ValueError, "order"):
            ukf.restore_lossless_payload(raw, probe, converted, self.root / "order" / "DWI.nrrd",
                                         self.root / "order" / "Mask.nrrd")

    def _signature(self, A3, folder):
        from shared.pipeline import RawInputBundle
        from tractography import ukf

        bvals, bvecs = gradient_set()
        case = write_fsl_case(self.root / folder, A3, bvals, bvecs)
        raw = RawInputBundle(subject_root=self.root, subject_id="gradframe", input_dir=self.root / folder,
                             dwi_path=case.dwi_path, bval_path=case.bval_path, bvec_path=case.bvec_path,
                             mask_path=case.mask_path)
        binary, converter = self.root / "UKFTractography", self.root / "DWIConvert"
        binary.write_text("mock")
        converter.write_text("mock")
        return ukf.signature(raw, {}, binary, converter), raw

    def test_b2_gradient_frame_ukf_policy_changes_only_for_affected_geometry(self):
        base_keys = ["version", "subject_id", "parameters", "inputs", "executable", "converter", "protocol",
                     "shell_policy", "conversion_policy", "mask_policy", "max_excluded_mask_fraction"]
        tiny_tilt = np.array([[1.0, -1e-6, 0.0], [1e-6, 1.0, 0.0], [0.0, 0.0, 1.0]]) @ GEOMETRIES["axial_det_neg_LAS"]
        for name, A3 in list(GEOMETRIES.items()) + [("axial_det_neg_tiny_tilt", tiny_tilt)]:
            with self.subTest(geometry=name):
                signature, _ = self._signature(A3, name)
                self.assertEqual(list(signature), base_keys)
                if name in ("axial_det_neg_LAS", "axial_det_neg_tiny_tilt"):
                    self.assertEqual(signature["conversion_policy"], UKF_POLICY_V1)
                else:
                    self.assertNotEqual(signature["conversion_policy"], UKF_POLICY_V1)

    def test_b2_gradient_frame_ukf_manifest_text(self):
        from tractography import ukf

        _, raw = self._signature(GEOMETRIES["axial_det_neg_LAS"], "text_las")
        self.assertEqual(ukf.validate_inputs(raw)["input_coordinate_system"], UKF_TEXT_V1)
        _, raw = self._signature(GEOMETRIES["oblique20_det_neg"], "text_obl")
        text = ukf.validate_inputs(raw)["input_coordinate_system"]
        self.assertNotEqual(text, UKF_TEXT_V1)
        self.assertIn("R F b", text)


# ----------------------------------------------------------------------------- native tensor header (nhdr_write)
class NhdrWriteGradientFrameTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def write(self, name, shells=(1000,)):
        from anatomy.parcellation.conversion.nhdr_write import nhdr_write

        bvals, bvecs = gradient_set(shells=shells, per_shell=8)
        case = write_fsl_case(self.root / name, GEOMETRIES[name], bvals, bvecs)
        target = self.root / name / "DWI.nhdr"
        nhdr_write(str(case.dwi_path), str(case.bval_path), str(case.bvec_path), str(target))
        return target, bvals, bvecs

    def test_b2_gradient_frame_nhdr_write_world_directions(self):
        for name in GEOMETRIES:
            with self.subTest(geometry=name):
                target, bvals, bvecs = self.write(name, shells=(1000, 2000))
                world, _ = nrrd_world(target)
                moving = bvals > 0
                self.assertLess(axis_angles(world[moving], reference_world(GEOMETRIES[name], bvecs)[moving]).max(),
                                MAX_ANGLE_DEG)

    def test_b2_gradient_frame_nhdr_write_unchanged_det_neg_equals_base(self):
        for name in BASE_NHDR:
            with self.subTest(geometry=name):
                target, _, _ = self.write(name)
                self.assertEqual(sha256(target), BASE_NHDR[name])


# ----------------------------------------------------------------------------- DDSeg explicit-file route (dti_slicer)
class DtiSlicerGradientFrameTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def run_route(self, name, bvals, bvecs):
        import nrrd
        from anatomy.segmentation import dti_slicer

        A3 = GEOMETRIES[name]
        case = write_fsl_case(self.root / name, A3, bvals, bvecs)
        output = self.root / name / "DTI"
        written = {}

        def fake_run(cmd, **_kwargs):
            if "--conversionMode" in cmd:
                target = Path(cmd[cmd.index("--outputVolume") + 1])
                header = {
                    "space": "left-posterior-superior",
                    "space directions": np.vstack([(LPS @ case.affine[:3, :3]).T, [np.nan] * 3]),
                    "space origin": LPS @ case.affine[:3, 3],
                    "kinds": ["domain", "domain", "domain", "list"],
                    "DWMRI_b-value": "%g" % bvals.max(), "measurement frame": np.eye(3), "modality": "DWMRI"}
                for index, value in enumerate(dwiconvert_fsl_to_nrrd_values(A3, bvals, bvecs)):
                    header["DWMRI_gradient_%04d" % index] = value
                data = np.arange(4 * 3 * 5 * len(bvals), dtype=np.int16).reshape(4, 3, 5, len(bvals))
                nrrd.write(str(target), data, header)
                written["bytes"] = target.read_bytes()
                written["data"] = data
            else:
                (output / "DTI_E3.nrrd").write_bytes(b"x")
            return mock.Mock(returncode=0)

        with mock.patch.object(dti_slicer, "_find_slicer_lib_dir", return_value=self.root), \
                mock.patch.object(dti_slicer, "find_pythonslicer", return_value=self.root / "PythonSlicer"), \
                mock.patch.object(dti_slicer, "slicer_runtime_roots", return_value=[]), \
                mock.patch.object(dti_slicer.subprocess, "run", side_effect=fake_run):
            dti_slicer.run_dti_feature_extraction(case.dwi_path, case.bval_path, case.bvec_path, output, self.root)
        return output / "DWI.nrrd", written

    def test_b2_gradient_frame_dti_slicer_world_directions(self):
        import nrrd

        bvals, bvecs = gradient_set(shells=(1000, 2000), per_shell=6)
        for name in CHANGED:
            with self.subTest(geometry=name):
                path, written = self.run_route(name, bvals, bvecs)
                world, values = nrrd_world(path)
                moving = bvals > 0
                self.assertLess(axis_angles(world[moving], reference_world(GEOMETRIES[name], bvecs)[moving]).max(),
                                MAX_ANGLE_DEG)
                np.testing.assert_allclose(np.linalg.norm(values, axis=1), np.sqrt(bvals / bvals.max()), atol=1e-7)
                data, _ = nrrd.read(str(path))
                self.assertTrue(np.array_equal(data, written["data"]))  # payload untouched

    def test_b2_gradient_frame_dti_slicer_unchanged_axial_det_neg_file_untouched(self):
        bvals, bvecs = gradient_set(shells=(1000, 2000), per_shell=6)
        path, written = self.run_route("axial_det_neg_LAS", bvals, bvecs)
        self.assertEqual(path.read_bytes(), written["bytes"])


# ----------------------------------------------------------------------------- DIPY tracking input
class _Captured(Exception):
    pass


class DipyTrackingGradientFrameTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def captured_bvecs(self, name, bvals, bvecs):
        import dipy.core.gradients
        from tractography import dti

        case = write_fsl_case(self.root / name, GEOMETRIES[name], bvals, bvecs)
        config = types.SimpleNamespace(input_bundle=case, input_root=self.root, subject_id=case.subject_id)
        seen = {}

        def capture(bvals_in, bvecs=None, **_kwargs):
            seen["bvecs"] = np.array(bvecs, copy=True)
            raise _Captured()

        with mock.patch.object(dipy.core.gradients, "gradient_table", side_effect=capture):
            with self.assertRaises(_Captured):
                dti._run_dipy_dti_tracking(config, streamlines_path=self.root / name / "out.trk", log=None)
        return seen["bvecs"], case

    def test_b2_gradient_frame_dipy_tracking_world_directions(self):
        bvals, bvecs = gradient_set()
        for name in GEOMETRIES:
            with self.subTest(geometry=name):
                given, _ = self.captured_bvecs(name, bvals, bvecs)
                A3 = GEOMETRIES[name]
                # DIPY maps a voxel-axis direction d to the world through the affine: A V^-1 d = R d.
                world = (A3 / np.linalg.norm(A3, axis=0) @ given.T).T
                moving = bvals > 0
                self.assertLess(axis_angles(world[moving], reference_world(A3, bvecs)[moving]).max(), MAX_ANGLE_DEG)

    def test_b2_gradient_frame_dipy_tracking_unchanged_det_neg_is_bit_identical(self):
        bvals, bvecs = gradient_set()
        for name in ("axial_det_neg_LAS", "oblique20_det_neg", "permuted_det_neg"):
            with self.subTest(geometry=name):
                given, case = self.captured_bvecs(name, bvals, bvecs)
                self.assertEqual(given.tobytes(), np.loadtxt(case.bvec_path.as_posix()).T.tobytes())


class DipyConsumerGradientFrameTests(unittest.TestCase):
    """Resume predicate of DTI tracking, and the other class (a) DIPY consumers (DKI parameters, NODDI fit_dir)."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def test_b2_gradient_frame_dti_tracking_resume_needs_marker_only_for_det_pos(self):
        from tractography import dti

        bvals, bvecs = gradient_set()
        for name, expected_without in (("axial_det_neg_LAS", True), ("oblique20_det_neg", True),
                                       ("axial_det_pos_RAS", False), ("oblique20_det_pos", False)):
            with self.subTest(geometry=name):
                case = write_fsl_case(self.root / name, GEOMETRIES[name], bvals, bvecs)
                manifest = self.root / name / "manifest.json"
                manifest.write_text(json.dumps({"status": "completed"}))
                self.assertIs(dti.tracking_gradient_frame_current(manifest, case.dwi_path), expected_without)
                manifest.write_text(json.dumps({"status": "completed", "gradient_frame": dti.DIPY_GRADIENT_FRAME}))
                self.assertTrue(dti.tracking_gradient_frame_current(manifest, case.dwi_path))

    def test_b2_gradient_frame_dki_vectors_and_signature(self):
        from microstructure import dki
        from shared.pipeline import RawInputBundle

        bvals, bvecs = gradient_set(shells=(1000, 2000), per_shell=16)
        signatures = {}
        for name in ("axial_det_neg_LAS", "oblique20_det_pos"):
            with self.subTest(geometry=name):
                case = write_fsl_case(self.root / name, GEOMETRIES[name], bvals, bvecs)
                raw = RawInputBundle(subject_root=self.root, subject_id="gradframe", input_dir=self.root / name,
                                     dwi_path=case.dwi_path, bval_path=case.bval_path, bvec_path=case.bvec_path,
                                     mask_path=case.mask_path)
                config = dki.DkiConfig(subject_id="gradframe", raw_input_bundle=raw, output_dir=self.root / "dki",
                                       fit_method="WLS")
                seen = {}

                def capture(shape, bvals_in, vectors, limits):
                    seen["vectors"] = np.array(vectors, copy=True)
                    raise _Captured()

                with mock.patch.object(dki, "_check_acquisition", side_effect=capture), self.assertRaises(_Captured):
                    dki._load_protocol(config)
                A3 = GEOMETRIES[name]
                world = (A3 / np.linalg.norm(A3, axis=0) @ seen["vectors"].T).T
                moving = bvals > 0
                self.assertLess(axis_angles(world[moving], reference_world(A3, bvecs)[moving]).max(), MAX_ANGLE_DEG)
                signatures[name] = dki.expected_dki_signature(config)
        self.assertNotIn("gradient_frame", signatures["axial_det_neg_LAS"])
        self.assertEqual(signatures["oblique20_det_pos"]["gradient_frame"], dki.DKI_GRADIENT_FRAME)

    def test_b2_gradient_frame_noddi_amico_bvecs_and_manifest(self):
        from microstructure import pipeline as micro

        bvals, bvecs = gradient_set()
        for name in ("axial_det_neg_LAS", "permuted_det_pos"):
            with self.subTest(geometry=name):
                case = write_fsl_case(self.root / name, GEOMETRIES[name], bvals, bvecs)
                raw = types.SimpleNamespace(dwi_path=case.dwi_path, bval_path=case.bval_path,
                                            bvec_path=case.bvec_path, mask_path=case.mask_path)
                config = types.SimpleNamespace(
                    raw_input_bundle=raw, subject_id="gradframe", noddi_directional_average=False,
                    noddi_b0_threshold=0, noddi_lmax=12, noddi_ndirs=500, noddi_solver_lambda1=0.0,
                    noddi_solver_lambda2=0.001, noddi_max_isotropic_fraction_for_tissue=0.9)
                seen = {}

                def scheme(bval, bvec, schemeFilename):
                    seen["bvec"] = Path(bvec)

                with mock.patch.object(micro, "fsl2scheme", side_effect=scheme), \
                        mock.patch.object(micro, "_amico_subject_dir", return_value=self.root / name / "amico"):
                    micro._prepare_amico_inputs(config=config)
                given = np.loadtxt(seen["bvec"]).T
                A3 = GEOMETRIES[name]
                world = (A3 / np.linalg.norm(A3, axis=0) @ given.T).T
                moving = bvals > 0
                self.assertLess(axis_angles(world[moving], reference_world(A3, bvecs)[moving]).max(), MAX_ANGLE_DEG)
                with mock.patch.object(micro, "_file_signature", return_value={}), \
                        mock.patch.object(micro, "_noddi_threads", return_value=1):
                    manifest = micro._expected_noddi_manifest(config)
                if name == "axial_det_neg_LAS":
                    if seen["bvec"].is_symlink():
                        self.assertEqual(seen["bvec"].resolve(), case.bvec_path.resolve())  # the user's file, linked
                    else:
                        self.assertEqual(seen["bvec"].read_bytes(), case.bvec_path.read_bytes())  # Windows copy fallback
                    self.assertNotIn("gradient_frame", manifest)
                else:
                    self.assertEqual(manifest["gradient_frame"], micro.NODDI_GRADIENT_FRAME)


# ----------------------------------------------------------------------------- NRRD import (DWIConvert NrrdToFSL)
def write_source_nrrd(path, ras_axes, world_gradients, frame=np.eye(3), b_max=1000.0):
    """A diffusion NRRD in LPS with measurement frame ``frame`` (columns) whose gradients are ``world_gradients``."""
    import nrrd

    lps_axes = LPS @ ras_axes
    header = {"space": "left-posterior-superior", "space directions": np.vstack([lps_axes.T, [np.nan] * 3]),
              "space origin": [1.0, 2.0, 3.0], "kinds": ["space", "space", "space", "list"],
              "measurement frame": frame.T, "modality": "DWMRI", "DWMRI_b-value": "%g" % b_max}
    for index, g in enumerate(world_gradients):
        value = np.linalg.inv(frame) @ LPS @ g
        header["DWMRI_gradient_%04d" % index] = " ".join(repr(float(x)) for x in value)
    nrrd.write(str(path), np.full((3, 4, 2, len(world_gradients)), 500, np.int16), header)


class NrrdImportGradientFrameTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        rng = np.random.default_rng(5)
        v = rng.normal(size=(12, 3))
        self.world = np.vstack([np.zeros((1, 3)), v / np.linalg.norm(v, axis=1, keepdims=True)])
        self.converter = self.root / "DWIConvert"
        self.converter.write_text("#!/bin/sh\n")
        self.converter.chmod(0o755)

    def fake_nrrd_to_fsl(self, source_ras_axes, out_axes=None):
        """DWIConvert NrrdToFSL as measured on two LPS sources (gradcheck n2f, Slicer 5.12.4): the output reverses j
        (an LAS-like, det < 0 grid) and the bvec file holds H g_world (the rotation of the grid is ignored).
        ``out_axes`` replaces the output grid to exercise the det > 0 branch of the correction (not a measured case).
        """
        import nibabel as nib

        out_axes = source_ras_axes @ np.diag([1.0, -1.0, 1.0]) if out_axes is None else out_axes

        def run(command, **_kwargs):
            nifti = Path(command[command.index("--outputNiftiFile") + 1])
            nib.save(nib.Nifti1Image(np.full((3, 4, 2, len(self.world)), 500, np.int16), affine4(out_axes)), str(nifti))
            Path(command[command.index("--outputBValues") + 1]).write_text(
                " ".join("0" if not np.any(g) else "1000" for g in self.world) + "\n")
            bvec = (H @ self.world.T)
            Path(command[command.index("--outputBVectors") + 1]).write_text(
                "\n".join(" ".join("0" if x == 0 else repr(float(x)) for x in row) for row in bvec) + "\n")
            return {"returncode": 0}

        return run, out_axes

    def import_case(self, name, ras_axes, frame=np.eye(3), force=False, subject_root=None, out_axes=None):
        from shared.input_import import InputImportConfig, prepare_inputs

        source = self.root / f"{name}.nrrd"
        if not source.exists():
            write_source_nrrd(source, ras_axes, self.world, frame)
        run, out_axes = self.fake_nrrd_to_fsl(ras_axes, out_axes)
        subject_root = subject_root or self.root / name
        with mock.patch("shared.input_import._run", side_effect=run) as called:
            manifest = prepare_inputs(InputImportConfig(source=source, subject_root=subject_root, subject_id=name,
                                                        source_kind="nrrd", backend="dwiconvert",
                                                        dwiconvert_executable=self.converter, force=force))
        return manifest, subject_root / "input", out_axes, called

    def test_b2_gradient_frame_nrrd_import_world_directions(self):
        oblique_axes = U20 @ np.diag([-2.0, -2.0, 2.0])
        frame = U20.T @ np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])  # non-identity frame
        permuted_pos = PERM @ H @ np.diag([2.0, 2.0, 2.0])  # det > 0 output grid
        for name, axes, mf, out in (("oblique", oblique_axes, np.eye(3), None),
                                    ("oblique_frame", oblique_axes, frame, None),
                                    ("axial_det_pos_output", np.diag([-2.0, -2.0, 2.0]), np.eye(3), np.diag([2.0, 2.0, 2.0])),
                                    ("permuted_det_pos_output", np.diag([-2.0, -2.0, 2.0]), frame, permuted_pos)):
            with self.subTest(case=name):
                manifest, input_dir, out_axes, _ = self.import_case(name, axes, mf, out_axes=out)
                b = np.loadtxt(input_dir / "bvecs").T
                world = reference_world(out_axes, b)
                self.assertLess(axis_angles(world[1:], self.world[1:]).max(), MAX_ANGLE_DEG)
                if name != "axial_det_pos_output":  # H g is already right for an RAS output grid
                    self.assertIn("gradient_frame", manifest)
                self.assertEqual((input_dir / "bvals").read_text(), "0" + " 1000" * 12 + "\n")

    def test_b2_gradient_frame_nrrd_import_unchanged_axial_keeps_dwiconvert_bvecs(self):
        manifest, input_dir, _, _ = self.import_case("axial", np.diag([-2.0, -2.0, 2.0]))
        expected = "\n".join(" ".join("0" if x == 0 else repr(float(x)) for x in row) for row in (H @ self.world.T)) + "\n"
        self.assertEqual((input_dir / "bvecs").read_text(), expected)
        self.assertNotIn("gradient_frame", manifest)

    def test_b2_gradient_frame_nrrd_import_old_wrong_import_is_not_reused(self):
        axes = U20 @ np.diag([-2.0, -2.0, 2.0])
        first, input_dir, _, _ = self.import_case("reuse_oblique", axes)
        # An import made before the fix: same signatures, but DWIConvert's uncorrected bvecs.
        manifest_path = input_dir / "input_import_manifest.json"
        saved = json.loads(manifest_path.read_text())
        saved.pop("gradient_frame")
        (input_dir / "bvecs").write_text(
            "\n".join(" ".join("0" if x == 0 else repr(float(x)) for x in row) for row in (H @ self.world.T)) + "\n")
        from shared.input_import import _output_signatures, _canonical_targets
        saved["output_signatures"] = _output_signatures(_canonical_targets(self.root / "reuse_oblique"))
        manifest_path.write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n")
        again, _, out_axes, called = self.import_case("reuse_oblique", axes, force=True)
        self.assertNotIn("reused", again)
        self.assertEqual(called.call_count, 1)
        b = np.loadtxt(input_dir / "bvecs").T
        self.assertLess(axis_angles(reference_world(out_axes, b)[1:], self.world[1:]).max(), MAX_ANGLE_DEG)
        # A current import of the same source is reused.
        third, _, _, called = self.import_case("reuse_oblique", axes, force=True)
        self.assertTrue(third.get("reused"))
        self.assertEqual(called.call_count, 0)

    def auto_import(self, folder, ras_axes):
        """Run on a folder that holds one raw NRRD: the orchestrator's route (prepare_convertible_root)."""
        from shared.convertible_input import prepare_convertible_root

        run, out_axes = self.fake_nrrd_to_fsl(ras_axes)
        with mock.patch("shared.input_import._run", side_effect=run) as called:
            prepare_convertible_root(folder, "auto", dwiconvert=self.converter)
        return out_axes, called.call_count

    def test_b2fix_grad_G1_auto_import_receipt_does_not_keep_old_wrong_bvecs(self):
        """G1 (Opus B1): the auto-import receipt must not keep the bvecs that base code imported from an oblique NRRD."""
        from shared.input_import import _canonical_targets, _output_signatures

        axes = U20 @ np.diag([-2.0, -2.0, 2.0])
        folder = self.root / "auto_oblique"
        folder.mkdir()
        write_source_nrrd(folder / "dwi.nrrd", axes, self.world)
        _, count = self.auto_import(folder, axes)
        self.assertEqual(count, 1)
        # The state base code left: the same receipt, DWIConvert's uncorrected bvecs, no gradient_frame entry.
        input_dir = folder / "input"
        (input_dir / "bvecs").write_text(
            "\n".join(" ".join("0" if x == 0 else repr(float(x)) for x in row) for row in (H @ self.world.T)) + "\n")
        manifest_path = input_dir / "input_import_manifest.json"
        saved = json.loads(manifest_path.read_text())
        saved.pop("gradient_frame", None)
        saved["output_signatures"] = _output_signatures(_canonical_targets(folder))
        manifest_path.write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n")
        out_axes, count = self.auto_import(folder, axes)
        self.assertEqual(count, 1)  # imported again
        b = np.loadtxt(input_dir / "bvecs").T
        self.assertLess(axis_angles(reference_world(out_axes, b)[1:], self.world[1:]).max(), MAX_ANGLE_DEG)
        _, count = self.auto_import(folder, axes)
        self.assertEqual(count, 0)  # a current import is not repeated on the next resume

    def test_b2fix_grad_G1_auto_import_receipt_keeps_a_right_import(self):
        """G1 (Opus B1): an axis-aligned source whose bvecs are right keeps its receipt; no file is touched."""
        from shared.convertible_input import RECEIPT

        axes = np.diag([-2.0, -2.0, 2.0])
        folder = self.root / "auto_axial"
        folder.mkdir()
        write_source_nrrd(folder / "dwi.nrrd", axes, self.world)
        _, count = self.auto_import(folder, axes)
        self.assertEqual(count, 1)
        names = ("data.nii.gz", "bvals", "bvecs", "input_import_manifest.json", RECEIPT)

        def snapshot():
            return {name: ((folder / "input" / name).read_bytes(), (folder / "input" / name).stat().st_mtime_ns)
                    for name in names}

        before = snapshot()
        _, count = self.auto_import(folder, axes)
        self.assertEqual(count, 0)
        self.assertEqual(snapshot(), before)


# ----------------------------------------------------------------------------- tensor fit matrix (teem_dti)
def base_prepare_gradients(reader, vtk, storage_node_class):
    """prepare_gradients of base commit 3c912ad2 (column normalisation), copied for the bit-identity check."""
    from shared.teem_dti import GRADIENT_NORM_EPSILON

    grads, bvalues = vtk.vtkDoubleArray(), vtk.vtkDoubleArray()
    storage_node_class().ParseDiffusionInformation(reader, grads, bvalues)
    measurement_frame = vtk.vtkMatrix4x4()
    measurement_frame.DeepCopy(reader.GetMeasurementFrameMatrix())
    rotation = vtk.vtkMatrix4x4()
    rotation.DeepCopy(reader.GetRasToIjkMatrix())
    for row in range(3):
        rotation.SetElement(row, 3, 0)
    for column in range(3):
        values = [rotation.GetElement(row, column) for row in range(3)]
        vtk.vtkMath.Normalize(values)
        for row in range(3):
            rotation.SetElement(row, column, values[row])
    transform = vtk.vtkTransform()
    transform.PostMultiply()
    transform.SetMatrix(measurement_frame)
    transform.Concatenate(rotation)
    transform.Update()
    count, max_b = grads.GetNumberOfTuples(), bvalues.GetRange()[1]
    gradients = np.empty((count, 3))
    for index in range(count):
        point = transform.TransformPoint(grads.GetTuple3(index))
        norm = math.sqrt(point[0] * point[0] + point[1] * point[1] + point[2] * point[2])
        if norm > GRADIENT_NORM_EPSILON:
            point = (point[0] / norm, point[1] / norm, point[2] / norm)
        factor = math.sqrt(bvalues.GetValue(index) / max_b)
        gradients[index] = (point[0] * factor, point[1] * factor, point[2] * factor)
    return gradients, max_b, rotation


class TensorFitGradientFrameTests(unittest.TestCase):
    def setUp(self):
        from shared import teem_dti

        try:
            self.vtk, self.vtkTeem, _, self.storage = teem_dti._vtk_modules()
        except Exception as error:  # pragma: no cover - plain python3
            self.skipTest(f"Slicer vtkTeem unavailable in this interpreter: {error}")
        self.teem = teem_dti
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def control_reader(self, name, A3):
        """A correct header (RAS space, MF = R, values = F b): isolates the fit-side matrix."""
        bvals, bvecs = gradient_set(shells=(1000, 2000), per_shell=6)
        case = write_fsl_case(self.root / name, A3, bvals, bvecs)
        R = A3 / np.linalg.norm(A3, axis=0)
        F = H if np.linalg.det(A3) > 0 else np.eye(3)
        lines = ["NRRD0005", "type: float", "dimension: 4", "space: right-anterior-superior", "sizes: 4 3 5 %d" % len(bvals),
                 "space directions: " + " ".join("(%r,%r,%r)" % tuple(float(v) for v in c) for c in A3.T) + " none",
                 "kinds: space space space list", "endian: little", "encoding: gzip", "space origin: (-10,-12,5)",
                 "byteskip: -1", "data file: %s" % case.dwi_path.name,
                 "measurement frame: " + " ".join("(%r,%r,%r)" % tuple(float(v) for v in c) for c in R.T),
                 "modality:=DWMRI", "DWMRI_b-value:=%g" % bvals.max()]
        for index, (b, v) in enumerate(zip(bvals, bvecs)):
            lines.append("DWMRI_gradient_%04d:=%s" % (index, " ".join(repr(float(x)) for x in F @ v * math.sqrt(b / bvals.max()))))
        target = self.root / name / "control.nhdr"
        target.write_text("\n".join(lines) + "\n")
        reader = self.vtkTeem.vtkTeemNRRDReader()
        reader.SetFileName(str(target))
        reader.UpdateInformation()
        return reader, bvals, bvecs

    def test_b2_gradient_frame_tensor_fit_matrix_is_the_rotation(self):
        for name in ("oblique20_det_neg", "axial_det_pos_RAS", "oblique20_det_pos", "permuted_det_neg"):
            with self.subTest(geometry=name):
                A3 = GEOMETRIES[name]
                reader, bvals, bvecs = self.control_reader(name, A3)
                gradients, _, rotation = self.teem.prepare_gradients(reader, self.vtk, self.storage)
                N = self.teem._matrix_to_numpy(rotation)[:3, :3]
                self.assertLess(np.abs(N @ N.T - np.eye(3)).max(), 1e-9)  # the frame written is a rotation
                world = (np.linalg.inv(N) @ gradients.T).T
                moving = bvals > 0
                self.assertLess(axis_angles(world[moving], reference_world(A3, bvecs)[moving]).max(), MAX_ANGLE_DEG)

    def test_b2_gradient_frame_tensor_fit_unchanged_grids_are_bit_identical_to_base(self):
        iso_oblique = U20 @ H @ np.diag([2.0, 2.0, 2.0])
        hcp_like = np.diag([-1.25, 1.25, 1.25])
        for name, A3 in (("axial_det_neg_LAS", GEOMETRIES["axial_det_neg_LAS"]), ("hcp_like", hcp_like),
                         ("axial_det_pos_RAS", GEOMETRIES["axial_det_pos_RAS"]), ("iso_oblique", iso_oblique)):
            with self.subTest(geometry=name):
                reader, _, _ = self.control_reader(name, A3)
                new, new_b, new_rot = self.teem.prepare_gradients(reader, self.vtk, self.storage)
                old, old_b, old_rot = base_prepare_gradients(reader, self.vtk, self.storage)
                self.assertEqual(new.tobytes(), old.tobytes())
                self.assertEqual(new_b, old_b)
                self.assertEqual(self.teem._matrix_to_numpy(new_rot).tobytes(), self.teem._matrix_to_numpy(old_rot).tobytes())


# Cache-key fields that base commit 3c912ad2 computes on these fixtures (captured by running that commit's
# shared/pipeline.py: ensure_diffusion_headers, _native_manifest, _registered_manifest). Same on all 4 geometries.
BASE_CONVERSION = {"bval_bvec_io.py": "4ff4b5c17933d638f7913a330c910c780772de136b901f462e454a8a9400f2ff",
                   "nhdr_write.py": "14a19586994a7587e8dde1a986b879ceebfca57e950d7f0d3393b85afdca506f"}
BASE_PIPELINE = "2427fa6d0ee34c427c643e66e1ea75a6b093ce09cccc20ca08a2288d05792cbb"
BASE_KEY_FIELDS = {
    "headers": {"kind": "diffusion_headers", "version": 2, "implementation": BASE_CONVERSION},
    "native": {"kind": "native_dti_bundle", "version": 4, "conversion_implementation": BASE_CONVERSION,
               "preparation_implementation": BASE_PIPELINE,
               "tensor_implementation": "7a3058c95d33e3a446cec695e4d1edf2da980abf6dd5ce618531246b230dfa71"},
    "registered": {"kind": "registered_dti_bundle", "version": 2, "implementation": BASE_PIPELINE},
}
PINNED_FILES = ("anatomy/parcellation/conversion/nhdr_write.py", "shared/teem_dti.py", "shared/pipeline.py")


def cache_key_fields(folder, dwi_path=None, A3=None):
    """The implementation fields of the diffusion-headers, native-DTI and registered-DTI cache keys."""
    from shared import pipeline

    bvals, bvecs = gradient_set()
    case = write_fsl_case(folder, A3, bvals, bvecs)
    raw = pipeline.RawInputBundle(folder, "gradframe", folder, dwi_path or case.dwi_path, case.bval_path,
                                  case.bvec_path, case.mask_path)
    if dwi_path is None:
        pipeline.ensure_diffusion_headers(raw)
        headers = json.loads(next((raw.effective_cache_root / "diffusion_inputs").glob("*/manifest.json")).read_text())
    else:
        headers = None
    native = pipeline._build_native_bundle(raw, "wls")
    native_manifest = pipeline._native_manifest(raw, native)
    atlas = Path(folder) / "atlas.nii.gz"
    atlas.write_bytes(b"atlas")
    registered = pipeline._registered_manifest(raw, native, pipeline._build_registered_bundle(raw, "wls", atlas))
    return {
        "headers": headers and {key: headers[key] for key in BASE_KEY_FIELDS["headers"]},
        "native": {key: native_manifest[key] for key in BASE_KEY_FIELDS["native"]},
        "registered": {key: registered[key] for key in BASE_KEY_FIELDS["registered"]},
    }


class CacheKeyPinTests(unittest.TestCase):
    """Item 8: an input that neither change affects keeps the cache keys of base 3c912ad2 (pin by equivalence)."""

    def live_pins_or_skip(self):
        """Skip when a pinned file changed after the gradient-frame fix.

        Such an edit changes the cache keys of every input, as it did before the pin existed, and the stale table
        entry does nothing. This is not a failure: delete the entry in shared/cache_key_pins.py at leisure.
        """
        from shared import cache_key_pins

        retired = [name for name in PINNED_FILES
                   if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() not in cache_key_pins.BASE_HASH_PINS]
        if retired:
            self.skipTest("pin retired (file changed after the gradient-frame fix): %s" % ", ".join(retired))

    def test_b2_gradient_frame_cache_keys_equal_base_for_axis_aligned_det_neg(self):
        self.live_pins_or_skip()
        # bval_bvec_io.py is not pinned and the gradient-frame fix did not edit it: take its hash as it is now.
        expected = json.loads(json.dumps(BASE_KEY_FIELDS))
        io_now = hashlib.sha256((ROOT / "anatomy/parcellation/conversion/bval_bvec_io.py").read_bytes()).hexdigest()
        expected["headers"]["implementation"]["bval_bvec_io.py"] = io_now
        expected["native"]["conversion_implementation"]["bval_bvec_io.py"] = io_now
        if b"\r\n" in (ROOT / PINNED_FILES[0]).read_bytes():
            # A CRLF checkout (Windows, core.autocrlf): base computed the hashes of its CRLF files there.
            from shared import cache_key_pins

            def base(name):
                digest = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                return cache_key_pins.BASE_HASH_PINS.get(digest, digest)

            conversion = {Path(name).name: base(name) for name in (
                "anatomy/parcellation/conversion/bval_bvec_io.py", PINNED_FILES[0])}
            expected = json.loads(json.dumps(BASE_KEY_FIELDS))
            expected["headers"]["implementation"] = expected["native"]["conversion_implementation"] = conversion
            expected["native"]["tensor_implementation"] = base(PINNED_FILES[1])
            expected["native"]["preparation_implementation"] = base(PINNED_FILES[2])
            expected["registered"]["implementation"] = base(PINNED_FILES[2])
        for name in ("axial_det_neg_LAS", "permuted_det_neg"):
            with self.subTest(geometry=name), tempfile.TemporaryDirectory() as tmp:
                self.assertEqual(cache_key_fields(Path(tmp) / name, A3=GEOMETRIES[name]), expected)

    def test_b2_gradient_frame_cache_keys_change_for_affected_geometry(self):
        current = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in PINNED_FILES}
        for name in ("axial_det_pos_RAS", "oblique20_det_neg", "oblique20_det_pos", "permuted_det_pos"):
            with self.subTest(geometry=name), tempfile.TemporaryDirectory() as tmp:
                fields = cache_key_fields(Path(tmp) / name, A3=GEOMETRIES[name])
                for kind in ("headers", "native", "registered"):
                    self.assertNotEqual(fields[kind], BASE_KEY_FIELDS[kind])
                self.assertEqual(fields["headers"]["implementation"]["nhdr_write.py"], current[PINNED_FILES[0]])
                self.assertEqual(fields["native"]["tensor_implementation"], current[PINNED_FILES[1]])
                self.assertEqual(fields["native"]["preparation_implementation"], current[PINNED_FILES[2]])
                self.assertEqual(fields["registered"]["implementation"], current[PINNED_FILES[2]])

    def test_b2_gradient_frame_unreadable_dwi_uses_the_real_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "fake.nii.gz"
            fake.write_bytes(b"not a nifti")
            fields = cache_key_fields(Path(tmp) / "case", dwi_path=fake, A3=GEOMETRIES["axial_det_neg_LAS"])
        self.assertEqual(fields["native"]["tensor_implementation"],
                         hashlib.sha256((ROOT / PINNED_FILES[1]).read_bytes()).hexdigest())
        self.assertEqual(fields["registered"]["implementation"],
                         hashlib.sha256((ROOT / PINNED_FILES[2]).read_bytes()).hexdigest())

    def test_b2_gradient_frame_pin_predicate_is_exact(self):
        from shared import cache_key_pins

        tiny = GEOMETRIES["axial_det_neg_LAS"].copy()
        tiny[0, 1] = 1e-12
        cases = {"axial_det_neg_LAS": True, "permuted_det_neg": True, "axial_det_pos_RAS": False,
                 "oblique20_det_neg": False, "oblique20_det_pos": False, "permuted_det_pos": False}
        with tempfile.TemporaryDirectory() as tmp:
            bvals, bvecs = gradient_set()
            for name, expected in list(cases.items()) + [("tiny_off_axis", False)]:
                A3 = tiny if name == "tiny_off_axis" else GEOMETRIES[name]
                with self.subTest(geometry=name):
                    case = write_fsl_case(Path(tmp) / name, A3, bvals, bvecs)
                    self.assertIs(cache_key_pins.gradient_frame_fix_is_no_op(case.dwi_path), expected)
            self.assertIs(cache_key_pins.gradient_frame_fix_is_no_op(Path(tmp) / "missing.nii.gz"), False)


# ----------------------------------------------------------------------------- G3: product-level reuse on resume
BASE_LITERALS = {PINNED_FILES[0]: BASE_CONVERSION["nhdr_write.py"], PINNED_FILES[1]: BASE_KEY_FIELDS["native"]["tensor_implementation"],
                 PINNED_FILES[2]: BASE_PIPELINE}


def pinned_base(name):
    """The base 3c912ad2 sha256 of a pinned file in this checkout (LF or CRLF bytes), through the pin table."""
    from shared import cache_key_pins

    return cache_key_pins.BASE_HASH_PINS.get(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), BASE_LITERALS[name])


def subject_inputs(root, A3):
    """A subject root with canonical raw inputs (input/data.nii.gz, bvals, bvecs, nodif_brain_mask.nii.gz)."""
    from shared.pipeline import RawInputBundle

    bvals, bvecs = gradient_set()
    case = write_fsl_case(Path(root) / "input", A3, bvals, bvecs)
    mask = case.mask_path.with_name("nodif_brain_mask.nii.gz")
    case.mask_path.replace(mask)
    return RawInputBundle(Path(root), "gradframe", Path(root) / "input", case.dwi_path, case.bval_path, case.bvec_path, mask)


def native_manifest_as_written(raw, fit_method, base):
    """The native DTI bundle manifest that base 3c912ad2 (base=True) or the fixed code (base=False) writes."""
    from shared import pipeline

    manifest = pipeline._native_manifest(raw, types.SimpleNamespace(fit_method=fit_method))
    if base:
        manifest["conversion_implementation"]["nhdr_write.py"] = pinned_base(PINNED_FILES[0])
        manifest["tensor_implementation"] = pinned_base(PINNED_FILES[1])
        manifest["preparation_implementation"] = pinned_base(PINNED_FILES[2])
    return manifest


def scalar_qc_record(maps):
    from shared.scalar_qc import _signature

    return json.dumps({"status": "completed", "automatic_qc_passed": True, "checks": {"finite": True},
                       "output_artifacts": {str(Path(path).resolve()): _signature(path) for path in maps}})


def finish_microstructure_dti(raw, base):
    """A finished Microstructure DTI stage (records as the stage writes them; maps are placeholders)."""
    from shared import orchestrator
    from shared.pipeline import _file_signature

    outputs = orchestrator._microstructure_dti_outputs(raw.subject_root, raw.subject_id)
    maps = [outputs[key] for key in ("fa", "md", "ad", "rd", "l1", "l2", "l3", "trace")]
    for path in maps + [outputs["dec"]]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"map " + path.name.encode())
    outputs["qc"].write_text(scalar_qc_record(maps))
    outputs["manifest"].write_text(json.dumps({
        "kind": "microstructure_phase3", "version": 6, "subject_id": raw.subject_id, "fit_method": "wls",
        "native_bundle": {"dir": "native", "manifest": native_manifest_as_written(raw, "wls", base)},
        "raw_input_bundle": {key: _file_signature(getattr(raw, key + "_path")) for key in ("dwi", "bval", "bvec", "mask")}}))


def finish_segmentation(raw, base):
    """A finished DDSeg stage on the native LS bundle (orchestrator.py: ensure_native_dti_bundle, run_ddseg)."""
    import nibabel as nib
    from anatomy.segmentation.config import DDSegConfig
    from anatomy.segmentation.qc import write_ddseg_qc_record
    from shared import orchestrator, pipeline

    native = native_manifest_as_written(raw, "ls", base)
    bundle = pipeline._build_native_bundle(raw, "ls")
    bundle.bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle.manifest_path.write_text(json.dumps(native))
    outputs = {key: path for key, path in orchestrator._segmentation_outputs(raw.subject_root).items() if key != "qc"}
    directory = outputs["segmentation"].parent
    directory.mkdir(parents=True, exist_ok=True)
    labels = np.ones((3, 3, 3), dtype=np.int16)
    probabilities = np.zeros((*labels.shape, 4), dtype=np.float32)
    probabilities[..., 1] = 1
    for key, path in outputs.items():
        data = labels if key == "segmentation" else (probabilities[..., 1] if key == "wm_probability" else probabilities[..., 0])
        nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    write_ddseg_qc_record(DDSegConfig(input_mask_nii=raw.mask_path, parameter_type="DTI", output_folder=directory,
                                      weights_dir=raw.subject_root / "weights", dti_feature_dir=bundle.bundle_dir,
                                      source_dwi=raw.dwi_path),
                          labels=labels, probability_maps=probabilities, outputs=outputs,
                          geometry={"feature_dir": bundle.bundle_dir.resolve().as_posix()})
    # Base wrote no bundle key into ddseg_source.json; the fixed orchestrator records the one the run read (G3b).
    orchestrator._write_anatomy_source(directory / "ddseg_source.json", raw, native["source_files"],
                                       **({} if base else {"tensor_key": native}))
    return bundle


def finish_parcellation(raw, base):
    """A finished DDParcel stage on the registered WLS bundle (orchestrator.py: run_ddparcel_from_registered_bundle)."""
    from anatomy.parcellation.pipeline import DDParcelRegisteredConfig, _registered_task_signature
    from shared import orchestrator, pipeline

    atlas = raw.subject_root / "atlas.nii.gz"
    atlas.write_bytes(b"atlas")
    registered = pipeline._build_registered_bundle(raw, "wls", atlas)
    registered.bundle_dir.mkdir(parents=True, exist_ok=True)
    for name in registered.__dataclass_fields__:
        if name.endswith("_path") and name != "manifest_path":
            getattr(registered, name).write_bytes(b"registered " + name.encode())
    implementation = (pinned_base(PINNED_FILES[2]) if base else
                      pipeline._gradient_frame_key_pin(raw)(hashlib.sha256((ROOT / PINNED_FILES[2]).read_bytes()).hexdigest()))
    registered.manifest_path.write_text(json.dumps({
        "kind": "registered_dti_bundle", "version": 2, "subject_id": raw.subject_id, "fit_method": "wls",
        "scalar_format": "nifti", "implementation": implementation,
        "native_bundle": {"dir": "native", "manifest": native_manifest_as_written(raw, "wls", base), "artifacts": {}}}))
    weights = raw.subject_root / "weights"
    weights.mkdir(exist_ok=True)
    (weights / "model.pkl").write_bytes(b"weights")
    outputs = orchestrator._parcellation_outputs(raw.subject_root, raw.subject_id)
    directory = outputs["wmparc"].parent
    directory.mkdir(parents=True, exist_ok=True)
    for path in outputs.values():
        path.write_bytes(b"parcel " + path.name.encode())
    config = DDParcelRegisteredConfig(raw.subject_id, registered, directory, raw.mask_path, weights, device="cpu")
    pipeline._write_manifest(directory / "ddparcel_preparation.json", _registered_task_signature(config, weights))
    (directory / "ddparcel_qc.json").write_text(scalar_qc_record(list(outputs.values())))
    orchestrator._write_anatomy_source(directory / "ddparcel_source.json", raw,
                                       native_manifest_as_written(raw, "wls", base)["source_files"])
    return directory


class ProductReuseGradientFrameTests(unittest.TestCase):
    """G3 (Codex P1 "product-level reuse"): on resume, a product made from a tensor bundle of base 3c912ad2 is
    computed again for an affected geometry and stays reusable for an unaffected one (item 8, through the pin).

    The checks called here are the ones the resume path calls (plan_subject_products and the execution path).
    """

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def complete(self, product, name, base):
        from shared import orchestrator

        raw = subject_inputs(self.root / f"{product}_{name}_{base}", GEOMETRIES[name])
        if product == "microstructure_dti":
            finish_microstructure_dti(raw, base)
            return orchestrator._microstructure_dti_complete(raw.subject_root, raw.subject_id, fit_method="wls")
        if product == "segmentation":
            finish_segmentation(raw, base)
            return orchestrator._segmentation_complete(raw.subject_root)
        finish_parcellation(raw, base)
        return orchestrator._parcellation_complete(raw.subject_root, raw.subject_id)

    def test_b2fix_grad_G3_products_of_base_tensors_are_recomputed_for_affected_geometry(self):
        for name in ("oblique20_det_neg", "axial_det_pos_RAS"):
            for product in ("microstructure_dti", "segmentation", "parcellation"):
                with self.subTest(geometry=name, product=product):
                    self.assertFalse(self.complete(product, name, base=True))

    def test_b2fix_grad_G3_products_of_base_tensors_stay_reusable_for_unaffected_geometry(self):
        CacheKeyPinTests.live_pins_or_skip(self)
        for name in ("axial_det_neg_LAS", "permuted_det_neg"):
            for product in ("microstructure_dti", "segmentation", "parcellation"):
                with self.subTest(geometry=name, product=product):
                    self.assertTrue(self.complete(product, name, base=True))

    def test_b2fix_grad_G3_unaffected_products_with_older_bundle_records_stay_reusable(self):
        # A record written before base (other bundle keys) was reused on HEAD for an input the fix leaves unchanged.
        from unittest import mock

        module = sys.modules[__name__]
        written = module.native_manifest_as_written

        def older(raw, fit_method, base):
            return {**written(raw, fit_method, base), "tensor_implementation": "older"}

        with mock.patch.object(module, "native_manifest_as_written", older):
            for name in ("axial_det_neg_LAS", "permuted_det_neg"):
                for product in ("microstructure_dti", "segmentation", "parcellation"):
                    with self.subTest(geometry=name, product=product):
                        self.assertTrue(self.complete(product, name, base=True))

    def test_b2fix_grad_G3b_affected_old_segmentation_is_recomputed_after_its_bundle_was_rebuilt(self):
        from shared import orchestrator

        for name in ("oblique20_det_neg", "axial_det_pos_RAS"):
            with self.subTest(geometry=name):
                raw = subject_inputs(self.root / f"seg_rebuilt_{name}", GEOMETRIES[name])
                bundle = finish_segmentation(raw, base=True)
                # Another consumer rebuilt the LS bundle in the recorded folder with the fixed code.
                bundle.manifest_path.write_text(json.dumps(native_manifest_as_written(raw, "ls", base=False)))
                self.assertFalse(orchestrator._segmentation_complete(raw.subject_root))

    def test_b2fix_grad_G3b_orchestrator_ddseg_run_records_the_bundle_key(self):
        from unittest import mock
        from shared import orchestrator, pipeline

        raw = subject_inputs(self.root / "seg_run", GEOMETRIES["oblique20_det_neg"])
        bundle = pipeline._build_native_bundle(raw, "ls")
        native = native_manifest_as_written(raw, "ls", base=False)

        def ensure(bundle_raw, **_):
            bundle.bundle_dir.mkdir(parents=True, exist_ok=True)
            bundle.manifest_path.write_text(json.dumps(native))
            return bundle

        def run_ddseg(config):  # only the DDSeg outputs and QC record (finish_segmentation); the bundle stays as built
            finish_segmentation(raw, base=True)
            (config.output_folder / "ddseg_source.json").unlink()
            bundle.manifest_path.write_text(json.dumps(native))

        def fake_microstructure(micro_cfg, log=None):
            dti_dir = micro_cfg.output_dir / "dti"
            dti_dir.mkdir(parents=True, exist_ok=True)
            (micro_cfg.output_dir / "manifest.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
            (dti_dir / "dti_qc.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
            return types.SimpleNamespace(
                fa_path=dti_dir / f"{raw.subject_id}_dti_FA.nii.gz",
                md_path=dti_dir / f"{raw.subject_id}_dti_MD.nii.gz",
                ad_path=dti_dir / f"{raw.subject_id}_dti_AD.nii.gz",
                rd_path=dti_dir / f"{raw.subject_id}_dti_RD.nii.gz",
                l1_path=None, l2_path=None, l3_path=None, trace_path=None, dec_path=None,
                roi_stats_csv_path=None, roi_stats_json_path=None,
            )

        config = orchestrator.ProductOrchestratorConfig(subject_id=raw.subject_id, subject_root=raw.subject_root,
                                                        targets=("segmentation",), execute=True)
        with mock.patch("shared.pipeline.ensure_native_dti_bundle", side_effect=ensure), \
                mock.patch("microstructure.run_microstructure", side_effect=fake_microstructure), \
                mock.patch("anatomy.segmentation.pipeline.run_ddseg", side_effect=run_ddseg), \
                mock.patch.object(orchestrator, "_run_ddseg_world_qc_gate", return_value=None):
            result = orchestrator.run_subject_products(config)
        statuses = {task.name: task.status for task in result.tasks}
        self.assertEqual(statuses.get("segmentation.ddseg"), "completed", statuses)
        record = json.loads((orchestrator._segmentation_outputs(raw.subject_root)["qc"].parent / "ddseg_source.json").read_text())
        self.assertEqual({key: record.get(key) for key in orchestrator._TENSOR_KEY_FIELDS},
                         {key: native[key] for key in orchestrator._TENSOR_KEY_FIELDS})
        self.assertTrue(orchestrator._segmentation_complete(raw.subject_root))

    def test_b2fix_grad_G3b_standalone_ddseg_result_follows_the_pin(self):
        # The standalone DDSeg module writes no ddseg_source.json and no source_dwi in its QC record.
        from shared import orchestrator

        for name, expected in (("oblique20_det_neg", False), ("axial_det_neg_LAS", True)):
            with self.subTest(geometry=name):
                raw = subject_inputs(self.root / f"seg_standalone_{name}", GEOMETRIES[name])
                finish_segmentation(raw, base=False)
                outputs = orchestrator._segmentation_outputs(raw.subject_root)
                (outputs["qc"].parent / "ddseg_source.json").unlink()
                qc = json.loads(outputs["qc"].read_text())
                outputs["qc"].write_text(json.dumps({**qc, "source_dwi": None}))
                self.assertIs(orchestrator._segmentation_complete(raw.subject_root), expected)

    def test_b2fix_grad_G3b_unaffected_segmentation_with_a_base_record_stays_reusable(self):
        from shared import orchestrator

        for name in ("axial_det_neg_LAS", "permuted_det_neg"):
            with self.subTest(geometry=name):
                raw = subject_inputs(self.root / f"seg_base_{name}", GEOMETRIES[name])
                bundle = finish_segmentation(raw, base=True)
                bundle.manifest_path.write_text(json.dumps(native_manifest_as_written(raw, "ls", base=False)))
                self.assertTrue(orchestrator._segmentation_complete(raw.subject_root))

    def test_b2fix_grad_G3_unaffected_segmentation_stays_reusable_without_its_cache_folder(self):
        import shutil
        from shared import orchestrator

        raw = subject_inputs(self.root / "seg_no_cache", GEOMETRIES["axial_det_neg_LAS"])
        bundle = finish_segmentation(raw, base=True)
        shutil.rmtree(bundle.bundle_dir)
        self.assertTrue(orchestrator._segmentation_complete(raw.subject_root))

    def test_b2fix_grad_G3_parcellation_without_receipt_follows_the_pin(self):
        from shared import orchestrator

        for name, expected in (("oblique20_det_neg", False), ("axial_det_neg_LAS", True)):
            with self.subTest(geometry=name):
                raw = subject_inputs(self.root / f"parcel_no_receipt_{name}", GEOMETRIES[name])
                (finish_parcellation(raw, base=True) / "ddparcel_preparation.json").unlink()
                self.assertIs(orchestrator._parcellation_complete(raw.subject_root, raw.subject_id), expected)

    def test_b2fix_grad_G3_products_of_fixed_tensors_are_reused_for_affected_geometry(self):
        for name in ("oblique20_det_neg", "axial_det_pos_RAS"):
            for product in ("microstructure_dti", "segmentation", "parcellation"):
                with self.subTest(geometry=name, product=product):
                    self.assertTrue(self.complete(product, name, base=False))


# ----------------------------------------------------------------------------- G4: unreadable DWI
class UnreadableDwiGradientFrameTests(unittest.TestCase):
    """G4 (Opus N3 + N4, Codex P1 "unreadable affine"): a DWI that nibabel cannot read counts as AFFECTED (as in the
    pin); a currency check then needs the new key or marker and never raises."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        bvals, bvecs = gradient_set()
        self.case = write_fsl_case(self.root / "case", GEOMETRIES["axial_det_neg_LAS"], bvals, bvecs)
        self.case.dwi_path.write_bytes(b"not a nifti" * 40)  # nibabel: ImageFileError (not an OSError)

    def raw(self):
        from shared.pipeline import RawInputBundle

        return RawInputBundle(subject_root=self.root, subject_id="gradframe", input_dir=self.root / "case",
                              dwi_path=self.case.dwi_path, bval_path=self.case.bval_path,
                              bvec_path=self.case.bvec_path, mask_path=self.case.mask_path)

    def test_b2fix_grad_G4_ukf_unreadable_dwi_is_affected_and_never_raises(self):
        from tractography import ukf

        raw = self.raw()
        binary, converter = self.root / "UKFTractography", self.root / "DWIConvert"
        binary.write_text("mock")
        converter.write_text("mock")
        directory = self.root / "ukf"
        directory.mkdir()
        # A historical (unmarked) manifest: the v1 policy that base code recorded while the file was readable.
        with mock.patch.object(ukf, "_input_affine", return_value=affine4(GEOMETRIES["axial_det_neg_LAS"])):
            historical = ukf.signature(raw, {}, binary, converter)
        self.assertEqual(historical["conversion_policy"], UKF_POLICY_V1)
        ukf.paths(raw.subject_id, directory)["manifest"].write_text(
            json.dumps({"status": "completed", "signature": historical}))
        self.assertNotEqual(ukf.signature(raw, {}, binary, converter)["conversion_policy"], UKF_POLICY_V1)
        self.assertFalse(ukf.current(raw, directory, {}, binary, converter))

    def test_b2fix_grad_G4_dti_tracking_unreadable_dwi_needs_the_marker(self):
        from tractography import dti

        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"status": "completed"}))
        self.assertFalse(dti.tracking_gradient_frame_current(manifest, self.case.dwi_path))
        manifest.write_text(json.dumps({"status": "completed", "gradient_frame": dti.DIPY_GRADIENT_FRAME}))
        self.assertTrue(dti.tracking_gradient_frame_current(manifest, self.case.dwi_path))

    def test_b2fix_grad_G4_dki_unreadable_dwi_needs_the_new_key(self):
        from microstructure import dki

        config = dki.DkiConfig(subject_id="gradframe", raw_input_bundle=self.raw(), output_dir=self.root / "dki",
                               fit_method="WLS")
        signature = dki.expected_dki_signature(config)
        self.assertEqual(signature.get("gradient_frame"), dki.DKI_GRADIENT_FRAME)
        historical = {key: value for key, value in signature.items() if key != "gradient_frame"}
        paths = dki.dki_output_paths(config.subject_id, config.output_dir)
        paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
        paths["manifest"].write_text(json.dumps({"status": "completed", "signature": historical}))
        paths["qc"].write_text(json.dumps({"status": "completed", "valid_voxels": 10}))
        self.assertFalse(dki.dki_outputs_current(config))

    def test_b2fix_grad_G4_noddi_unreadable_dwi_needs_the_marker(self):
        from microstructure import pipeline as micro

        raw = types.SimpleNamespace(dwi_path=self.case.dwi_path, bval_path=self.case.bval_path,
                                    bvec_path=self.case.bvec_path, mask_path=self.case.mask_path)
        config = types.SimpleNamespace(
            raw_input_bundle=raw, subject_id="gradframe", noddi_directional_average=False,
            noddi_b0_threshold=0, noddi_lmax=12, noddi_ndirs=500, noddi_solver_lambda1=0.0,
            noddi_solver_lambda2=0.001, noddi_max_isotropic_fraction_for_tissue=0.9)
        with mock.patch.object(micro, "_file_signature", return_value={}), \
                mock.patch.object(micro, "_noddi_threads", return_value=1):
            manifest = micro._expected_noddi_manifest(config)
        self.assertEqual(manifest.get("gradient_frame"), micro.NODDI_GRADIENT_FRAME)
        historical = {key: value for key, value in manifest.items() if key != "gradient_frame"}
        self.assertNotEqual(historical, manifest)


if __name__ == "__main__":
    unittest.main()
