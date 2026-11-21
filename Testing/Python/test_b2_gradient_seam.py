"""Merge seam of batch 2: the View input module and the pipeline hold the gradient frame rule in two places.

``shared/input_preview.py`` (lane ``dwi``) maps FSL bvecs to RAS directions for the display.
``shared/gradient_frame.py`` (lane ``grad``) holds the same rule for every pipeline route. The two were written in
separate worktrees. This test fails when one of them changes and the other does not.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import gradient_frame, input_preview  # noqa: E402


def _rotation(axis, degrees):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    angle = np.deg2rad(degrees)
    cross = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def _affine(matrix3, voxel=(2.0, 2.0, 2.0)):
    affine = np.eye(4)
    affine[:3, :3] = np.asarray(matrix3, dtype=float) @ np.diag(voxel)
    affine[:3, 3] = (-90.0, 110.0, -70.0)
    return affine


LAS = np.diag([-1.0, 1.0, 1.0])
PERMUTED = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # det < 0
OBLIQUE = _rotation((1.0, 2.0, 0.5), 20.0)

GEOMETRIES = {
    "axial_det_neg_LAS": _affine(LAS),
    "axial_det_pos_RAS": _affine(np.eye(3)),
    "oblique20_det_neg": _affine(OBLIQUE @ LAS),
    "oblique20_det_pos": _affine(OBLIQUE),
    "permuted_det_neg": _affine(PERMUTED),
    "permuted_det_pos": _affine(PERMUTED @ LAS),
    "oblique20_det_neg_anisotropic": _affine(OBLIQUE @ LAS, voxel=(2.0, 2.0, 2.5)),
    "oblique20_det_pos_anisotropic": _affine(OBLIQUE, voxel=(1.25, 2.0, 2.5)),
}

BVECS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.6, -0.48, 0.64],
        [-0.36, 0.8, 0.48],
        [0.3, 0.3, -0.9],  # not unit length: the display normalises, the pipeline keeps the norm
    ]
)


class GradientFrameSeamTests(unittest.TestCase):
    def test_b2_seam_view_input_and_pipeline_use_one_gradient_frame_rule(self):
        for name, affine in GEOMETRIES.items():
            with self.subTest(geometry=name):
                self.assertEqual(
                    np.linalg.det(affine[:3, :3]) > 0,
                    "det_pos" in name,
                    "the fixture geometry has the determinant sign its name states",
                )
                display = np.asarray(input_preview.gradients_ras(affine, BVECS))
                pipeline = np.asarray(input_preview._unit_rows(gradient_frame.fsl_to_world(BVECS, affine)))
                self.assertEqual(display.shape, pipeline.shape)
                np.testing.assert_allclose(display, pipeline, rtol=0.0, atol=1e-12)
                self.assertTrue(np.all(display[0] == 0.0), "a zero bvec stays zero")

    def test_b2_seam_the_two_rules_differ_from_no_flip_for_positive_determinant(self):
        # Guard against a test that passes because both sides lost the flip.
        affine = GEOMETRIES["axial_det_pos_RAS"]
        display = np.asarray(input_preview.gradients_ras(affine, BVECS))
        np.testing.assert_allclose(display[1], [-1.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(gradient_frame.fsl_to_world(BVECS, affine)[1], [-1.0, 0.0, 0.0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
