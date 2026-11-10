#!/usr/bin/env python3
"""DKI verification. All generated signals here are SYNTHETIC ONLY, not clinical evidence."""

from __future__ import annotations

import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np
if sys.platform == "darwin":
    try:
        import torch as _darwin_preload_torch  # noqa: F401
    except Exception:
        pass
from dipy.core.gradients import gradient_table
from dipy.reconst.dki import DiffusionKurtosisModel, design_matrix
from dipy.sims.voxel import dki_signal
from dataclasses import replace
from unittest.mock import patch
from shared.pipeline import RawInputBundle
from microstructure.dki import DkiConfig, inspect_dki_inputs, run_dki, dki_outputs_current, DKI_METRICS


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


AFFINE = np.array([[-1.25, 0, 0, 8], [0, 1.5, 0, -11], [0, 0, 2, 4], [0, 0, 0, 1]], dtype=float)
SPATIAL_SHAPE = (2, 3, 2)
METRICS = ("fa", "md", "ad", "rd", "mk", "ak", "rk", "mkt", "rtk", "kfa")


def synthetic_gradients() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(80123)
    directions = rng.normal(size=(30, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    bvals = np.r_[np.zeros(4), np.repeat([1000.0, 2000.0, 3000.0], 30)]
    bvecs = np.vstack((np.zeros((4, 3)), np.tile(directions, (3, 1))))
    return bvals, bvecs


def synthetic_case(*, noisy: bool = False) -> dict[str, np.ndarray]:
    """Known constant directional kurtosis; no noise unless explicitly requested."""
    bvals, bvecs = synthetic_gradients()
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=50)
    data = np.empty((*SPATIAL_SHAPE, len(bvals)), dtype=np.float64)
    eigenvalues = np.empty((*SPATIAL_SHAPE, 3), dtype=np.float64)
    kurtosis = np.empty(SPATIAL_SHAPE, dtype=np.float64)
    for index, coordinate in enumerate(np.ndindex(SPATIAL_SHAPE)):
        evals = np.array([0.0012, 0.0008, 0.0005]) * (1 + index / 50)
        k = 0.4 + index / 40
        dt = np.array([evals[0], 0, evals[1], 0, 0, evals[2]])
        kt = np.zeros(15)
        kt[:3] = k * evals**2 / evals.mean() ** 2
        kt[9:12] = k * np.array([evals[0] * evals[1], evals[0] * evals[2], evals[1] * evals[2]])
        kt[9:12] /= 3 * evals.mean() ** 2
        data[coordinate] = dki_signal(gtab, dt, kt, S0=1000 + 17 * index)
        eigenvalues[coordinate] = evals
        kurtosis[coordinate] = k
    if noisy:
        # Deterministic measurement noise exposes smoothing and OLS/WLS mixups.
        data *= np.exp(np.random.default_rng(3021).normal(0, 0.025, size=data.shape))
    mask = np.ones(SPATIAL_SHAPE, dtype=np.uint8)
    mask[0, 0, 0] = 0
    return {"data": data, "bvals": bvals, "bvecs": bvecs, "mask": mask,
            "eigenvalues": eigenvalues, "kurtosis": kurtosis}


def direct_dipy_maps(data, bvals, bvecs, mask, method: str) -> dict[str, np.ndarray]:
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=50)
    fit = DiffusionKurtosisModel(gtab, fit_method=method).fit(data, mask=np.asarray(mask, dtype=bool))
    return {name: np.asarray(getattr(fit, name)() if name in ("mk", "ak", "rk", "mkt", "rtk")
                             else getattr(fit, name)) for name in METRICS}


def write_case(root: Path, case: dict[str, np.ndarray]) -> dict[str, Path]:
    inputs = root / "input"
    inputs.mkdir(parents=True, exist_ok=True)
    paths = {"dwi": inputs / "data.nii.gz", "bval": inputs / "bvals", "bvec": inputs / "bvecs",
             "mask": inputs / "nodif_brain_mask.nii.gz"}
    nib.save(nib.Nifti1Image(case["data"], AFFINE), paths["dwi"])
    nib.save(nib.Nifti1Image(case["mask"], AFFINE), paths["mask"])
    np.savetxt(paths["bval"], case["bvals"][None, :], fmt="%.12g")
    np.savetxt(paths["bvec"], case["bvecs"].T, fmt="%.12g")
    return paths


class DkiSyntheticReferenceTest(unittest.TestCase):
    def test_synthetic_design_has_full_22_parameter_rank(self) -> None:
        bvals, bvecs = synthetic_gradients()
        matrix = design_matrix(gradient_table(bvals, bvecs=bvecs, b0_threshold=50))
        self.assertEqual(matrix.shape, (94, 22))
        self.assertEqual(np.linalg.matrix_rank(matrix), 22)

    def test_actual_dipy_ols_and_wls_recover_synthetic_ground_truth(self) -> None:
        case = synthetic_case()
        selected = case["mask"].astype(bool)
        eigenvalues = case["eigenvalues"][selected]
        for method in ("OLS", "WLS"):
            with self.subTest(method=method):
                maps = direct_dipy_maps(case["data"], case["bvals"], case["bvecs"], case["mask"], method)
                np.testing.assert_allclose(maps["ad"][selected], eigenvalues[:, 0], rtol=1e-7, atol=1e-10)
                np.testing.assert_allclose(maps["rd"][selected], eigenvalues[:, 1:].mean(axis=-1), rtol=1e-7, atol=1e-10)
                np.testing.assert_allclose(maps["md"][selected], eigenvalues.mean(axis=-1), rtol=1e-7, atol=1e-10)
                for metric in ("mk", "ak", "rk"):
                    # DIPY's analytical MK approximation has ~1e-6 integration error.
                    np.testing.assert_allclose(maps[metric][selected], case["kurtosis"][selected], rtol=5e-6, atol=2e-6)


class DkiRoiExtractionTest(unittest.TestCase):
    def test_streamed_crop_matches_nibabel_scaling_for_compressed_and_plain_nifti(self) -> None:
        from Testing.Python.dki_backend_acceptance import stream_roi

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = np.arange(5 * 6 * 4 * 6, dtype=np.int16).reshape(5, 6, 4, 6)
            for extension in (".nii", ".nii.gz"):
                with self.subTest(extension=extension):
                    path = root / ("scaled" + extension)
                    image = nib.Nifti1Image(data, AFFINE)
                    image.header.set_slope_inter(2.5, 7)
                    nib.save(image, path)
                    original = path.read_bytes()
                    loaded = nib.load(path)
                    crop = stream_roi(loaded, path, np.array([1, 2, 1]), 2)
                    np.testing.assert_array_equal(crop, loaded.get_fdata()[1:3, 2:4, 1:3, :])
                    self.assertEqual(path.read_bytes(), original)

    def test_roi_preserves_supplied_mask_cropped_world_geometry_and_original_fingerprints(self) -> None:
        from Testing.Python.dki_backend_acceptance import fingerprint, prepare_roi

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            inputs = source / "input"
            inputs.mkdir(parents=True)
            data = np.arange(5 * 6 * 4 * 6, dtype=np.float32).reshape(5, 6, 4, 6)
            mask = np.ones(data.shape[:3], dtype=np.uint8)
            mask[1, 2, 1] = 0
            nib.save(nib.Nifti1Image(data, AFFINE), inputs / "data.nii.gz")
            nib.save(nib.Nifti1Image(mask, AFFINE), inputs / "nodif_brain_mask.nii.gz")
            (inputs / "bvals").write_text("0 1000 1000 2000 2000 3000\n", encoding="ascii")
            (inputs / "bvecs").write_text("0 1 0 1 0 1\n0 0 1 0 1 0\n0 0 0 0 0 0\n", encoding="ascii")
            before = {p: fingerprint(p) for p in inputs.iterdir()}
            with redirect_stdout(StringIO()):
                result = prepare_roi(inputs, root / "evidence", side=2, center=[2, 3, 2])
            cropped = Path(result["subject_root"]) / "input"
            output = nib.load(cropped / "data.nii.gz")
            np.testing.assert_array_equal(output.dataobj, data[1:3, 2:4, 1:3, :])
            np.testing.assert_array_equal(nib.load(cropped / "nodif_brain_mask.nii.gz").dataobj, mask[1:3, 2:4, 1:3])
            np.testing.assert_allclose(output.affine @ [0, 0, 0, 1], AFFINE @ [1, 2, 1, 1])
            np.testing.assert_allclose(output.affine @ [1, 1, 1, 1], AFFINE @ [2, 3, 2, 1])
            self.assertTrue(result["source_inputs_unchanged"])
            self.assertEqual(result["smoothing"], "none")
            self.assertEqual(result["supplied_mask_voxels"], 7)
            for path, original in before.items():
                self.assertEqual(fingerprint(path), original)
            for name in ("bvals", "bvecs"):
                self.assertEqual((cropped / name).read_bytes(), (inputs / name).read_bytes())

    def test_evidence_directory_must_not_overlap_original_inputs(self) -> None:
        from Testing.Python.dki_backend_acceptance import prepare_roi

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "source/input"
            for output in (inputs, inputs / "outputs", root):
                with self.subTest(output=output), self.assertRaisesRegex(ValueError, "separate from source"):
                    prepare_roi(inputs, output)


class DkiBackendContractTest(unittest.TestCase):
    def test_zero_valid_voxels_are_failure_and_cannot_reuse_old_green_qc(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = write_case(root, synthetic_case())
            raw = RawInputBundle(root, "synthetic", root / "input", paths["dwi"], paths["bval"], paths["bvec"], paths["mask"])
            config = DkiConfig("synthetic", raw, root / "outputs", display_max_kurtosis=.01)
            with self.assertRaisesRegex(ValueError, "no valid quantitative voxels"):
                run_dki(config)
            self.assertEqual(json.loads((config.output_dir / "dki_qc.json").read_text())["status"], "failed")
            self.assertFalse(dki_outputs_current(config))
            config = replace(config, display_max_kurtosis=3)
            result = run_dki(config)
            qc = json.loads(result.qc_path.read_text())
            qc["valid_voxels"] = 0
            result.qc_path.write_text(json.dumps(qc))
            self.assertFalse(dki_outputs_current(config))

    """Synthetic inputs exercise real backend fit/caching and acquisition rejection."""

    def config(self, root, case=None, **options):
        paths = write_case(root, case if case is not None else synthetic_case(noisy=True))
        bundle = RawInputBundle(root, "synthetic", root / "input",
                                **{f"{key}_path": value for key, value in paths.items()})
        return DkiConfig("synthetic", bundle, root / "outputs/dki", **options)

    def test_default_dti_does_not_enable_optional_models(self):
        from microstructure.pipeline import MicrostructureConfig
        for key in ("run_free_water", "run_noddi", "run_dki"):
            self.assertFalse(MicrostructureConfig.__dataclass_fields__[key].default)

    def test_backend_matches_dipy_selected_observations_and_reuses_exact_outputs(self):
        from microstructure.dki import _load_protocol
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.config(root, voxels_per_chunk=3)
            source = config.raw_input_bundle.dwi_path.read_bytes()
            image, mask, gtab, selected, info = _load_protocol(config)
            data = np.asarray(image.dataobj)[mask][:, selected]
            direct = DiffusionKurtosisModel(gtab, fit_method="WLS", return_S0_hat=True, min_signal=1e-6).fit(data)
            result = run_dki(config)
            for key in DKI_METRICS:
                expected = np.asarray(direct.kfa if key == "kfa" else getattr(direct, key)(min_kurtosis=None, max_kurtosis=None))
                actual = nib.load(result.outputs[f"{key}_raw"]).get_fdata()[mask]
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-6)
            self.assertTrue(dki_outputs_current(config))
            self.assertEqual(run_dki(config).status, "cached")
            self.assertEqual(source, config.raw_input_bundle.dwi_path.read_bytes())
            self.assertEqual(info["design_rank"], 22)
            self.assertEqual(info["excluded_high_b_volumes"], 30)

    def test_corrupted_artifact_and_changed_gradient_invalidate_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            result = run_dki(config)
            result.mk_path.write_bytes(b"corrupted")
            self.assertFalse(dki_outputs_current(config))
            run_dki(config)
            config.raw_input_bundle.bval_path.write_text("0\n")
            self.assertFalse(dki_outputs_current(config))

    def test_single_shell_no_baseline_repeated_directions_and_nan_gradients_block_before_model(self):
        for defect in ("single", "no_b0", "rank", "nan", "nonunit"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as directory:
                case = synthetic_case()
                if defect == "single":
                    case["bvals"][case["bvals"] > 0] = 1000
                elif defect == "no_b0":
                    case["bvals"][:4] = 1000
                    case["bvecs"][:4] = [1, 0, 0]
                elif defect == "rank":
                    case["bvecs"][4:] = [1, 0, 0]
                elif defect == "nan":
                    case["bvecs"][6, 0] = np.nan
                else:
                    case["bvecs"][6] *= 2
                config = self.config(Path(directory), case)
                with patch("dipy.reconst.dki.DiffusionKurtosisModel") as model:
                    self.assertFalse(inspect_dki_inputs(config)["eligible"])
                    with self.assertRaisesRegex(ValueError, "ineligible"):
                        run_dki(config)
                    model.assert_not_called()

    def test_geometry_and_empty_mask_block(self):
        for defect in ("affine", "empty"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as directory:
                config = self.config(Path(directory))
                mask = nib.load(config.raw_input_bundle.mask_path)
                values = mask.get_fdata()
                affine = mask.affine.copy()
                if defect == "affine":
                    affine[0, 3] += 2
                else:
                    values[:] = 0
                nib.save(nib.Nifti1Image(values, affine), config.raw_input_bundle.mask_path)
                self.assertFalse(inspect_dki_inputs(config)["eligible"])

    def test_nominal_baseline_directions_cleared_without_modifying_gradient_file(self):
        with tempfile.TemporaryDirectory() as directory:
            case = synthetic_case()
            case["bvecs"][:4] = [1, 0, 0]
            config = self.config(Path(directory), case)
            original = config.raw_input_bundle.bvec_path.read_bytes()
            info = inspect_dki_inputs(config)
            self.assertTrue(info["eligible"])
            self.assertEqual(info["baseline_bvectors_cleared_in_working_table"], 4)
            self.assertEqual(original, config.raw_input_bundle.bvec_path.read_bytes())

    def test_nonfinite_and_nonpositive_signal_are_qc_not_filled(self):
        with tempfile.TemporaryDirectory() as directory:
            case = synthetic_case()
            case["data"][0, 0, 1, 6] = np.nan
            case["data"][0, 1, 0, 6] = 0
            config = self.config(Path(directory), case)
            result = run_dki(config)
            qc = json.loads(result.qc_path.read_text())
            self.assertEqual(qc["nonfinite_signal_voxels"], 1)
            self.assertEqual(qc["nonpositive_signal_voxels"], 1)
            self.assertTrue(np.isnan(nib.load(result.outputs["mk_raw"]).get_fdata()[0, 0, 1]))
            self.assertFalse(nib.load(result.valid_mask_path).get_fdata()[0, 0, 1])

    def test_mkcurve_missing_runtime_is_not_wls_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory), fit_method="MKCURVE_MATLAB", matlab_executable=Path(directory)/"absent")
            with patch("dipy.reconst.dki.DiffusionKurtosisModel") as model:
                self.assertFalse(inspect_dki_inputs(config)["eligible"])
                with self.assertRaisesRegex(ValueError, "MKCURVE backend unavailable"):
                    run_dki(config)
                model.assert_not_called()

    def test_run_plan_blocks_single_shell_but_preserves_dti_selection(self):
        from shared.orchestrator import ProductOrchestratorConfig, plan_subject_products, normalize_orchestrator_targets
        with tempfile.TemporaryDirectory() as directory:
            case = synthetic_case()
            case["bvals"][case["bvals"] > 0] = 1000
            root = Path(directory)
            self.config(root, case)
            config = ProductOrchestratorConfig("synthetic", root, ("microstructure",),
                                                microstructure_run_dki=True, brain_mask_backend="provided")
            plan = plan_subject_products(config, write=False)
            self.assertEqual(next(item for item in plan.items if item.target == "microstructure_dki").status, "blocked")
            self.assertEqual(next(item for item in plan.items if item.target == "microstructure").status, "will_run")
            self.assertEqual(normalize_orchestrator_targets(("microstructure",)), ("microstructure",))

    def test_independent_dki_audit_rejects_corrupted_display_not_raw_values(self):
        from scripts.audit_scientific_outputs import _audit_dki
        with tempfile.TemporaryDirectory() as directory:
            result = run_dki(self.config(Path(directory)))
            manifest = {"subject_root": directory, "tasks": [{"name": "microstructure.dki", "status": "completed",
                                    "qc_path": str(result.qc_path),
                                    "outputs": {key: str(path) for key, path in result.outputs.items()}}]}
            checks = []
            self.assertTrue(_audit_dki(manifest, checks))
            self.assertTrue(all(check["passed"] for check in checks), checks)
            image = nib.load(result.mk_path)
            values = image.get_fdata()
            values[0, 0, 1] = 2.9
            nib.save(nib.Nifti1Image(values, image.affine), result.mk_path)
            checks = []
            _audit_dki(manifest, checks)
            self.assertIn("dki.mk.display_derivation", {check["name"] for check in checks if not check["passed"]})


if __name__ == "__main__":
    unittest.main()
