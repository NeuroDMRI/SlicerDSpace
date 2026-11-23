#!/usr/bin/env python3
"""DDSeg orientation: affine-based reorientation to the model's RAS input and exact inverse.

The CNN itself is replaced by an oracle, so these tests are fast and need
neither torch nor pynrrd (a stub ``nrrd`` module serves the feature maps with
Slicer-style LPS headers).
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anatomy.segmentation.orientation import (  # noqa: E402
    MODEL_INPUT_AXCODES,
    ORIENTATION_CONVENTION,
    axcodes,
    from_model_orientation,
    nrrd_header_affine,
    to_model_orientation,
)

SHAPE_RAS = (6, 7, 5)  # x (L->R), y (P->A), z (I->S)
ZOOM = 2.0
ORIGIN = np.array([-5.0, -6.0, -4.0])


def _ras_affine() -> np.ndarray:
    affine = np.diag([ZOOM, ZOOM, ZOOM, 1.0])
    affine[:3, 3] = ORIGIN
    return affine


def _restore(data_ras: np.ndarray, codes: str) -> tuple[np.ndarray, np.ndarray]:
    """Re-store a RAS array/affine in another voxel order (world geometry unchanged)."""

    from nibabel.orientations import apply_orientation, axcodes2ornt, inv_ornt_aff, ornt_transform

    transform = ornt_transform(axcodes2ornt(("R", "A", "S")), axcodes2ornt(tuple(codes)))
    data = apply_orientation(data_ras, transform)
    affine = _ras_affine() @ inv_ornt_aff(transform, data_ras.shape)
    return np.ascontiguousarray(data), affine


def _world_grids():
    ijk = np.indices(SHAPE_RAS).astype(np.float64)
    return [ORIGIN[axis] + ZOOM * ijk[axis] for axis in range(3)]


STORAGES = ("RAS", "LAS", "LPS", "RPI", "ASR", "PLS")


class OrientationPrimitiveTest(unittest.TestCase):
    def test_round_trip_is_exact_for_every_storage(self) -> None:
        rng = np.random.default_rng(0)
        volume = rng.random(SHAPE_RAS)
        for codes in STORAGES:
            with self.subTest(storage=codes):
                data, affine = _restore(volume, codes)
                self.assertEqual(axcodes(affine), codes)
                model = to_model_orientation(data, affine)
                np.testing.assert_array_equal(from_model_orientation(model, affine), data)

    def test_model_array_is_identical_regardless_of_storage(self) -> None:
        rng = np.random.default_rng(1)
        volume = rng.random(SHAPE_RAS)
        for codes in STORAGES:
            with self.subTest(storage=codes):
                data, affine = _restore(volume, codes)
                np.testing.assert_array_equal(to_model_orientation(data, affine), volume)

    def test_trailing_channel_axes_are_preserved(self) -> None:
        volume = np.random.default_rng(2).random((*SHAPE_RAS, 4))
        data, affine = _restore(volume, "LPS")
        model = to_model_orientation(data, affine)
        self.assertEqual(model.shape, (*SHAPE_RAS, 4))
        np.testing.assert_array_equal(from_model_orientation(model, affine), data)

    def test_model_axes_are_ras(self) -> None:
        self.assertEqual(MODEL_INPUT_AXCODES, ("R", "A", "S"))

    def test_nrrd_header_affine_matches_nifti_affine(self) -> None:
        for codes in ("LAS", "LPS", "RAS"):
            with self.subTest(storage=codes):
                _, affine = _restore(np.zeros(SHAPE_RAS), codes)
                lps = np.array([-1.0, -1.0, 1.0])
                header_lps = {
                    "space": "left-posterior-superior",
                    "space directions": (affine[:3, :3] * lps[:, None]).T,
                    "space origin": affine[:3, 3] * lps,
                }
                np.testing.assert_allclose(nrrd_header_affine(header_lps), affine)
                header_ras = {
                    "space": "right-anterior-superior",
                    "space directions": affine[:3, :3].T,
                    "space origin": affine[:3, 3],
                }
                np.testing.assert_allclose(nrrd_header_affine(header_ras), affine)

    def test_nrrd_without_space_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "NRRD space"):
            nrrd_header_affine({"space directions": np.eye(3)})


def _fake_nrrd_module(store: dict[str, tuple[np.ndarray, np.ndarray]]) -> types.ModuleType:
    module = types.ModuleType("nrrd")

    def read(path: str):
        data, affine = store[Path(path).name]
        lps = np.array([-1.0, -1.0, 1.0])
        header = {
            "space": "left-posterior-superior",
            "space directions": (affine[:3, :3] * lps[:, None]).T,
            "space origin": affine[:3, 3] * lps,
        }
        return data, header

    module.read = read  # type: ignore[attr-defined]
    return module


class DDSegPipelineOrientationTest(unittest.TestCase):
    """Run the real preprocessing/unpadding/saving code with an oracle network."""

    def _run(self, codes: str, root: Path):
        from anatomy.segmentation.config import DDSegConfig
        from anatomy.segmentation import pipeline

        x, y, z = _world_grids()
        brain = np.ones(SHAPE_RAS, dtype=bool)
        brain[0, :, :] = False  # asymmetric mask: left-most slab outside the brain
        features_ras = {
            "DTI_FA": np.clip((y - y.min()) / (np.ptp(y) + 1.0), 0.0, 1.0),  # increases anteriorly
            "DTI_MD": 1e-3 + 1e-4 * x,  # increases to the right
            "DTI_E1": 1e-3 + 1e-4 * z,  # increases superiorly
            "DTI_E2": np.full(SHAPE_RAS, 1e-3),
            "DTI_E3": np.full(SHAPE_RAS, 5e-4),
        }
        store = {}
        for name, volume in features_ras.items():
            data, affine = _restore(volume, codes)
            store[f"{name}.nrrd"] = (data, affine)
        mask_data, mask_affine = _restore(brain.astype(np.uint8), codes)
        mask_path = root / "mask.nii.gz"
        mask_img = nib.Nifti1Image(mask_data, mask_affine)
        mask_img.set_qform(mask_affine, code=1)
        mask_img.set_sform(mask_affine, code=1)
        nib.save(mask_img, str(mask_path))
        feature_dir = root / "features"
        feature_dir.mkdir()
        for name in store:
            (feature_dir / name).write_bytes(b"")  # content is served by the stub nrrd module
        seen: dict[str, np.ndarray] = {}

        def oracle(stacked, cfg, **_kwargs):
            seen["stacked"] = stacked
            fa = stacked[..., 0]
            # WM (label 1) where the input says "anterior half", GM (2) elsewhere.
            anterior = np.zeros(stacked.shape[:3], dtype=bool)
            anterior[:, SHAPE_RAS[1] // 2 :, :] = True
            labels = np.where(anterior, 1, 2).astype(np.int16)
            labels[fa <= -99] = 0
            probs = np.zeros((*stacked.shape[:3], 4), dtype=np.float32)
            for k in range(4):
                probs[..., k] = (labels == k).astype(np.float32) * 0.97 + 0.01
            return labels, probs

        cfg = DDSegConfig(
            input_mask_nii=mask_path,
            parameter_type="DTI",
            output_folder=root / "out",
            weights_dir=root / "weights",
            dti_feature_dir=feature_dir,
            source_dwi=root / "dwi.nii.gz",
            device="cpu",  # This numerical oracle fixture does not load PyTorch.
        )
        with mock.patch.dict(sys.modules, {"nrrd": _fake_nrrd_module(store)}), mock.patch.object(
            pipeline, "predict_model_space", side_effect=oracle
        ):
            pipeline.run_ddseg(cfg)
        return cfg, seen, mask_affine

    def test_input_reaches_the_network_in_ras_order_for_any_storage(self) -> None:
        for codes in STORAGES:
            with self.subTest(storage=codes), tempfile.TemporaryDirectory() as tmp:
                _, seen, _ = self._run(codes, Path(tmp))
                stacked = seen["stacked"][: SHAPE_RAS[0], : SHAPE_RAS[1], : SHAPE_RAS[2], :]
                inside = stacked[1:, :, :, :]  # skip the masked-out left slab
                # FA increases along axis 1 (anterior), MD along axis 0 (right), E1 along axis 2 (superior).
                self.assertTrue(np.all(np.diff(inside[..., 0], axis=1) > 0))
                self.assertTrue(np.all(np.diff(inside[..., 1], axis=0) > 0))
                self.assertTrue(np.all(np.diff(inside[..., 2], axis=2) > 0))
                # The mask was reoriented identically: the empty slab is at model x=0 (left).
                self.assertTrue(np.all(stacked[0] == -100.0))

    def test_saved_prediction_maps_to_the_same_world_coordinates(self) -> None:
        for codes in STORAGES:
            with self.subTest(storage=codes), tempfile.TemporaryDirectory() as tmp:
                cfg, _, mask_affine = self._run(codes, Path(tmp))
                seg_img = nib.load(str(cfg.output_folder / "SegmentationMap_GMWMCSF.nii.gz"))
                np.testing.assert_allclose(seg_img.affine, mask_affine)
                np.testing.assert_allclose(seg_img.get_qform(), seg_img.get_sform())
                seg_ras = to_model_orientation(np.asarray(seg_img.dataobj), seg_img.affine)
                _, y, _ = _world_grids()
                anterior_world = np.zeros(SHAPE_RAS, dtype=bool)
                anterior_world[:, SHAPE_RAS[1] // 2 :, :] = True
                inside = np.ones(SHAPE_RAS, dtype=bool)
                inside[0] = False
                np.testing.assert_array_equal(seg_ras[inside & anterior_world], 1)
                np.testing.assert_array_equal(seg_ras[inside & ~anterior_world], 2)
                np.testing.assert_array_equal(seg_ras[~inside], 0)
                for name in ("WM", "GM", "CSF"):
                    prob_img = nib.load(str(cfg.output_folder / f"ProbabilisticMap_{name}.nii.gz"))
                    np.testing.assert_allclose(prob_img.affine, mask_affine)
                    self.assertEqual(prob_img.shape, seg_img.shape)

    def test_qc_records_schema_v2_orientation_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg, _, _ = self._run("LPS", Path(tmp))
            qc = json.loads((cfg.output_folder / "ddseg_qc.json").read_text(encoding="utf-8"))
            self.assertEqual(qc["schema_version"], 2)
            self.assertEqual(qc["orientation_convention"], ORIENTATION_CONVENTION)
            self.assertEqual(qc["model_input_axcodes"], "RAS")
            self.assertEqual(qc["geometry"]["mask_axcodes"], "LPS")
            self.assertLess(qc["geometry"]["feature_affine_max_abs_difference_mm"], 1e-6)
            self.assertTrue(qc["source_dwi"].endswith("dwi.nii.gz"))

    def test_feature_stored_in_a_different_orientation_is_rejected(self) -> None:
        from anatomy.segmentation.preprocess import load_mask, load_parameter_map

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mask_data, mask_affine = _restore(np.ones(SHAPE_RAS, dtype=np.uint8), "LAS")
            nib.save(nib.Nifti1Image(mask_data, mask_affine), str(root / "mask.nii.gz"))
            feat_data, feat_affine = _restore(np.ones(SHAPE_RAS), "LPS")
            nib.save(nib.Nifti1Image(feat_data, feat_affine), str(root / "DTI_FA.nii.gz"))
            mask = load_mask(root / "mask.nii.gz")
            with self.assertRaisesRegex(ValueError, "stored as LPS"):
                load_parameter_map(root / "DTI_FA.nii.gz", mask)


if __name__ == "__main__":
    unittest.main()
