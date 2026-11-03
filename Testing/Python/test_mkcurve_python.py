"""Scientific and integration contracts; synthetic data are not clinical proof."""

from pathlib import Path
import json
import importlib.metadata
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

from microstructure.mkcurve import detect_abnormal, inspect_mkcurve_backend, repair_nonpositive, run_mkcurve, synthetic_b0_grid
from microstructure.mkcurve_tensor import design_matrix, fit_wlls, integration_directions, parameter_maps
from microstructure.dki import run_dki, dki_outputs_current, inspect_dki_inputs
from Testing.Python import test_microstructure_dki as fixtures
from Testing.Python.test_microstructure_dki import synthetic_case, write_case


class MKCurvePythonTests(unittest.TestCase):
    def test_dynamic_and_ordered_workers_preserve_all_raw_products_exactly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = write_case(root, synthetic_case(noisy=True))
            records = [run_mkcurve(*[paths[key] for key in ('dwi', 'bval', 'bvec', 'mask')], root / mode,
                                   grid=np.arange(100,2200,100.), voxels_per_chunk=2, num_workers=2,
                                   worker_threads=1, chunk_schedule=mode) for mode in ('ordered','dynamic')]
            for key in records[0]['maps']:
                left, right = [nib.load(record['maps'][key]) for record in records]
                self.assertEqual(left.header.get_data_dtype(), right.header.get_data_dtype())
                np.testing.assert_array_equal(left.affine, right.affine)
                np.testing.assert_array_equal(np.asanyarray(left.dataobj), np.asanyarray(right.dataobj))
            for key in ('curve','original_coefficients','corrected_coefficients'):
                np.testing.assert_array_equal(*[np.load(record['auxiliary_files'][key]) for record in records])
            self.assertEqual(records[1]['settings']['max_inflight_chunks'],2)

    def test_batched_solver_matches_pivoted_qr_and_falls_back_safely(self):
        case = synthetic_case(noisy=True)
        design = design_matrix(case['bvals'], case['bvecs'])
        signals = case['data'].reshape(-1, len(design))
        reference = fit_wlls(signals, design)
        np.testing.assert_allclose(fit_wlls(signals, design, solver='batched'), reference, rtol=1e-8, atol=1e-10)
        with patch('microstructure.mkcurve_tensor.np.linalg.eigvalsh', side_effect=lambda matrix: -np.ones(matrix.shape[:-1])):
            np.testing.assert_equal(fit_wlls(signals, design, solver='batched'), reference)

    def test_parallel_batched_maps_match_serial_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = write_case(root, synthetic_case(noisy=True))
            inputs = [paths[key] for key in ('dwi', 'bval', 'bvec', 'mask')]
            grid = np.arange(100, 2200, 100.)
            reference = run_mkcurve(*inputs, root / 'reference', grid=grid)
            parallel = run_mkcurve(*inputs, root / 'parallel', grid=grid, solver='batched', num_workers=2, voxels_per_chunk=2)
            for key in reference['maps']:
                np.testing.assert_allclose(nib.load(reference['maps'][key]).get_fdata(),
                                           nib.load(parallel['maps'][key]).get_fdata(), rtol=2e-6, atol=2e-6)
            np.testing.assert_allclose(np.load(reference['auxiliary_files']['curve']),
                                       np.load(parallel['auxiliary_files']['curve']), rtol=2e-7, atol=2e-7)

    def test_zero_valid_corrected_maps_cannot_finish_green(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = write_case(root, synthetic_case(noisy=True))
            with patch("microstructure.mkcurve_tensor.parameter_maps", side_effect=lambda coefficients: {
                "mk": np.full(len(coefficients), np.nan)
            }), self.assertRaisesRegex(ValueError, "no valid corrected"):
                run_mkcurve(*[paths[key] for key in ("dwi", "bval", "bvec", "mask")], root / "fit",
                            grid=np.arange(100, 2200, 100.))
            self.assertEqual(json.loads((root / "fit/mkcurve_record.json").read_text())["status"], "failed")

    def test_backend_is_python_without_external_paths(self):
        record = inspect_mkcurve_backend()
        self.assertTrue(record["available"])
        self.assertEqual(record["implementation"], "python_port")
        self.assertEqual(integration_directions().shape, (256, 3))

    def test_unconstrained_wlls_recovers_known_synthetic_maps(self):
        case = synthetic_case()
        design = design_matrix(case["bvals"], case["bvecs"])
        maps = parameter_maps(fit_wlls(case["data"].reshape(-1, len(design)), design))
        for key in ("mk", "ak", "rk"):
            np.testing.assert_allclose(maps[key], case["kurtosis"].ravel(), atol=1e-9, rtol=1e-8)
        np.testing.assert_allclose(maps["md"], case["eigenvalues"].mean(axis=-1).ravel(), atol=1e-10)

    def test_dependency_discovery_does_not_crash_missing_package(self):
        real_version = importlib.metadata.version
        def version(key):
            if key == "scipy":
                raise importlib.metadata.PackageNotFoundError(key)
            return real_version(key)
        with patch("microstructure.mkcurve.importlib.metadata.version", side_effect=version):
            record = inspect_mkcurve_backend()
        self.assertFalse(record["available"])
        self.assertIn("Python package scipy", record["missing_required"])

    def test_zero_md_is_not_made_valid_by_algebraic_cancellation(self):
        coefficients = np.zeros((1, 22))
        coefficients[0, 1] = .001
        coefficients[0, 4] = -.001
        maps = parameter_maps(coefficients)
        for key in ("mk", "ak", "rk"):
            self.assertTrue(np.isnan(maps[key][0]))

    def test_predicted_not_observed_signal_weights(self):
        case = synthetic_case(noisy=True)
        design = design_matrix(case["bvals"], case["bvecs"])
        signals = case["data"].reshape(-1, len(design))
        actual = fit_wlls(signals, design)
        initial = np.linalg.lstsq(design, np.log(signals).T, rcond=None)[0]
        weights = np.exp(design @ initial).T
        reference = np.stack([np.linalg.lstsq(design * w[:, None], np.log(s) * w, rcond=None)[0]
                              for s, w in zip(signals, weights)])
        np.testing.assert_allclose(actual, reference, rtol=2e-6, atol=2e-8)

    def test_detection_matlab_indices_and_nonfinite_guard(self):
        grid = np.arange(100, 800, 100.)
        curves = np.array([[-1, 1, np.nan], [-.5, 2, 0], [0, 3, 0], [1, 2, 0], [2, 1, 0], [1, .5, 0], [.5, .1, 0]])
        result = detect_abnormal(curves, [250, 1000, 250], grid)
        np.testing.assert_equal(result["zero_mk_b0"][:2], [300, 200])
        np.testing.assert_equal(result["max_mk_b0"][:2], [500, 300])
        np.testing.assert_equal(result["abnormal_mask"], [True, False, False])
        self.assertFalse(result["curve_valid_mask"][2])
        self.assertEqual(result["corrected_b0"][0], 400)

    def test_nonpositive_working_signal_repair_is_explicit(self):
        data = np.full((3, 3, 3, 2), 10.)
        data[1, 1, 1, 0] = 0
        record = repair_nonpositive(data, np.ones((3, 3, 3), dtype=bool))
        self.assertEqual(data[1, 1, 1, 0], 10)
        self.assertEqual(record["repaired_measurements"], 1)
        data[:] = 0
        record = repair_nonpositive(data, np.ones((3, 3, 3), dtype=bool))
        self.assertEqual(record["unresolved_measurements"], 54)

    def test_scale_grid_requires_explicit_usable_signal_units(self):
        np.testing.assert_equal(synthetic_b0_grid([1000]), np.arange(100, 2001, 10))
        with self.assertRaisesRegex(ValueError, "Signal scale"):
            synthetic_b0_grid([1])

    def test_raw_outputs_geometry_input_immutability_and_chunk_equivalence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = synthetic_case(noisy=True)
            paths = write_case(root, case)
            originals = {key: path.read_bytes() for key, path in paths.items()}
            records = [run_mkcurve(*[paths[k] for k in ("dwi", "bval", "bvec", "mask")], root / str(chunk),
                                   grid=np.arange(100, 2200, 100.), voxels_per_chunk=chunk) for chunk in (2, 64)]
            for key in records[0]["maps"]:
                first = nib.load(records[0]["maps"][key])
                np.testing.assert_equal(first.affine, nib.load(paths["dwi"]).affine)
                np.testing.assert_equal(first.get_fdata(), nib.load(records[1]["maps"][key]).get_fdata())
            self.assertFalse(records[0]["raw_metric_clipping"])
            for key, path in paths.items():
                self.assertEqual(originals[key], path.read_bytes())

    def test_dki_python_integration_and_curve_cache_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = fixtures.DkiBackendContractTest().config(Path(tmp), fit_method="MKCURVE")
            self.assertTrue(inspect_dki_inputs(config)["eligible"])
            with patch("microstructure.mkcurve.synthetic_b0_grid", return_value=np.arange(100, 2200, 100.)):
                result = run_dki(config)
                self.assertIn("corrected_mk", result.mkcurve_outputs)
                self.assertTrue(dki_outputs_current(config))
                self.assertEqual(run_dki(config).status, "cached")
                result.mkcurve_outputs["corrected_mk"].write_bytes(b"corrupt")
                self.assertFalse(dki_outputs_current(config))
                result = run_dki(config)
                from scripts.audit_scientific_outputs import _audit_dki
                manifest = {"subject_root": tmp, "tasks": [{"name": "microstructure.dki", "status": "completed",
                            "qc_path": str(result.qc_path), "outputs": {key: str(path) for key, path in result.outputs.items()}}]}
                checks = []
                _audit_dki(manifest, checks)
                self.assertTrue(all(check["passed"] for check in checks), checks)
                curves = config.output_dir / "mkcurve/mk_vs_synthetic_b0.npy"
                curves.write_bytes(b"corrupt")
                self.assertFalse(dki_outputs_current(config))
                checks = []
                _audit_dki(manifest, checks)
                self.assertIn("mkcurve.curve.hash", {check["name"] for check in checks if not check["passed"]})

    def test_actual_author_octave_reference_fixture(self):
        fixture = json.loads(Path(__file__).with_name("fixtures").joinpath("mkcurve_author_octave.json").read_text())
        gradients = np.asarray(fixture["gradients"])
        signals = np.asarray(fixture["signals"])
        grid = np.asarray(fixture["grid"])
        design = design_matrix(gradients[:, 3], gradients[:, :3])
        original = fit_wlls(signals, design)
        maps = parameter_maps(original)
        for key, values in fixture["original_maps"].items():
            np.testing.assert_allclose(maps[key], values, rtol=1e-6, atol=1e-7)
        from microstructure.mkcurve_tensor import mean_kurtosis
        curves = []
        modified = signals.copy()
        for value in grid:
            modified[:, 0] = value
            curves.append(mean_kurtosis(fit_wlls(modified, design)))
        curves = np.asarray(curves)
        np.testing.assert_allclose(curves, fixture["curves"], rtol=1e-6, atol=1e-6)
        detection = detect_abnormal(curves, signals[:, 0], grid)
        for key, values in fixture["detection"].items():
            np.testing.assert_allclose(detection[key], values, rtol=1e-6, atol=1e-7)
        modified[:, 0] = detection["corrected_b0"]
        maps = parameter_maps(fit_wlls(modified, design))
        for key, values in fixture["corrected_maps"].items():
            np.testing.assert_allclose(maps[key], values, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
