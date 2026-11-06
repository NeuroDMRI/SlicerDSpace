"""Native PNL mask route (T2 design A3-A6): parameters, winsorise, arrays, FillHoles, threads, end to end."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from shared import pnl_mask_native as native


HAS_SITK = importlib.util.find_spec("SimpleITK") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_SCIPY = importlib.util.find_spec("scipy") is not None
SCRIPT = Path(native.__file__).resolve()


class Recorder:
    REGULAR = "REGULAR"
    EachIteration = "EachIteration"

    def __init__(self):
        self.calls = {}

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls[name] = (args, kwargs)
        return call


class ParameterTest(unittest.TestCase):
    def test_parameter_object_holds_the_ants_rigid_stage(self):
        p = native.PARAMETERS
        self.assertEqual((p.histogram_bins, p.sampling_percentage, p.sampling_seed), (32, 0.25, 1))
        self.assertEqual((p.learning_rate, p.maximum_step_physical), (0.1, 0.1))
        self.assertEqual((p.line_search_lower_limit, p.line_search_upper_limit, p.line_search_epsilon,
                          p.line_search_iterations), (0.0, 2.0, 0.2, 20))
        self.assertEqual((p.convergence_minimum, p.convergence_window), (1e-6, 10))
        self.assertEqual(p.shrink_factors, (8, 4, 2))
        self.assertEqual(p.smoothing_sigmas_voxels, (3.0, 2.0, 1.0))
        self.assertEqual(p.iterations, (1000, 500, 250))
        self.assertEqual(p.winsorize_lower, float(np.float32(0.005)))
        self.assertEqual(p.winsorize_upper, float(np.float32(0.995)))
        self.assertNotEqual(p.winsorize_lower, 0.005)
        self.assertEqual((p.winsorize_bins, p.winsorize_marginal_scale), (256, 10.0))

    @unittest.skipUnless(HAS_SITK, "SimpleITK is not installed")
    def test_each_level_is_configured_from_the_parameter_object(self):
        import SimpleITK as sitk

        for level, (shrink, sigma, iterations) in enumerate(((8, 3.0, 1000), (4, 2.0, 500), (2, 1.0, 250))):
            method = Recorder()
            native.configure_level(method, native.PARAMETERS, level)
            calls = method.calls
            self.assertEqual(calls["SetMetricAsMattesMutualInformation"][0], (32,))
            self.assertEqual(calls["SetMetricSamplingStrategy"][0], ("REGULAR",))
            self.assertEqual(calls["SetMetricSamplingPercentage"][0], (0.25, 1))
            self.assertIn("MetricUseFixedImageGradientFilterOff", calls)
            self.assertIn("MetricUseMovingImageGradientFilterOff", calls)
            self.assertIn("SetOptimizerScalesFromPhysicalShift", calls)
            self.assertIn("SmoothingSigmasAreSpecifiedInPhysicalUnitsOff", calls)
            self.assertEqual(calls["SetInterpolator"][0], (sitk.sitkLinear,))
            self.assertEqual(calls["SetShrinkFactorsPerLevel"][0], ([shrink],))
            self.assertEqual(calls["SetSmoothingSigmasPerLevel"][0], ([sigma],))
            self.assertEqual(calls["SetOptimizerAsConjugateGradientLineSearch"][1], dict(
                learningRate=0.1, numberOfIterations=iterations, convergenceMinimumValue=1e-6,
                convergenceWindowSize=10, lineSearchLowerLimit=0.0, lineSearchUpperLimit=2.0,
                lineSearchEpsilon=0.2, lineSearchMaximumIterations=20, estimateLearningRate="EachIteration",
                maximumStepSizeInPhysicalUnits=0.1))


class WinsorizeTest(unittest.TestCase):
    """Hand-computed ITK histogram quantiles (itkHistogram.hxx Initialize/GetIndex/Quantile)."""

    def volume(self):
        # 1000 voxels. min 0, max 2560 -> upper = 2560 + (2560/256)/10 = 2561;
        # interval = float32(2561/256) = 10.00390625, so bin j = [10.00390625 j, 10.00390625 (j+1)).
        data = np.full(1000, 1000.0)
        data[:4] = 0.0              # bin 0: 4 voxels
        data[4:6] = 15.0            # bin 1: 2 voxels
        data[6] = 2560.0            # the maximum, bin 255
        data[7] = 2550.99609375     # exactly the lower edge of bin 255 (= 255 * interval): half-open -> bin 255
        data[8:12] = 2545.0         # bin 254: 4 voxels
        return data.reshape(10, 10, 10)

    def test_bins_follow_itk_initialize(self):
        mins, maxs = native.histogram_bins(0.0, 2561.0)
        self.assertEqual(mins[1], 10.00390625)
        self.assertEqual(maxs[0], mins[1])
        self.assertEqual(mins[255], 2550.99609375)
        self.assertEqual(maxs[255], 2561.0)
        index = native.histogram_index(np.array([0.0, 10.00390625, 2550.99609375, 2560.0, 2561.0]), mins, maxs)
        self.assertEqual(index.tolist(), [0, 1, 255, 255, 255])

    def test_quantiles_cover_both_branches_with_float32_probabilities(self):
        lower, upper = native.winsorize_window(self.volume())
        # Lower branch (p < 0.5), p = float32(0.005) = 0.004999999888241291: cumulative 0.004 after bin 0,
        # 0.006 after bin 1 -> bin 1, p_n_prev 0.004, proportion 0.002:
        # 10.00390625 + ((p - 0.004) / 0.002) * 10.00390625 = 15.0058588160 -> float32 15.005858421325684.
        # (p = 0.005 exactly would give float32 15.005859375.)
        self.assertEqual(lower, 15.005858421325684)
        # Upper branch, p = float32(0.995) = 0.9950000047683716: walking down, bin 255 holds the maximum and
        # the edge voxel (p_n = 0.998), bin 254 four voxels (p_n = 0.994 <= p) -> bin 254, p_n_prev 0.998:
        # 2550.99609375 - ((0.998 - p) / 0.004) * 10.00390625 = 2543.4931759881 -> float32 2543.4931640625.
        # (A closed-right bin edge would move the edge voxel to bin 254 and give 2542.992919921875.)
        self.assertEqual(upper, 2543.4931640625)

    def test_constant_image_is_refused(self):
        with self.assertRaisesRegex(ValueError, "constant"):
            native.winsorize_window(np.ones((3, 3, 3)))

    @unittest.skipUnless(HAS_SITK, "SimpleITK is not installed")
    def test_winsorised_image_is_windowed_to_unit_range(self):
        import SimpleITK as sitk

        reference = sitk.Image(10, 10, 10, sitk.sitkFloat64)
        image, window = native.winsorize(native.to_sitk(self.volume(), reference))
        values = native.to_numpy(image)
        self.assertEqual(window, (15.005858421325684, 2543.4931640625))
        self.assertEqual(values[0, 0, 0], 0.0)
        self.assertEqual(values.max(), 1.0)
        self.assertAlmostEqual(values.flat[500], (1000.0 - window[0]) / (window[1] - window[0]), places=12)


@unittest.skipUnless(HAS_SITK, "SimpleITK is not installed")
class ArrayOrderTest(unittest.TestCase):
    def test_transpose_matches_itk_index_and_nibabel_on_an_asymmetric_volume(self):
        import nibabel as nib
        import SimpleITK as sitk

        array = np.arange(3 * 4 * 5, dtype=np.float64).reshape(3, 4, 5)
        reference = sitk.Image(3, 4, 5, sitk.sitkFloat64)
        reference.SetSpacing((1.5, 2.0, 2.5))
        image = native.to_sitk(array, reference)
        self.assertEqual(image.GetSize(), (3, 4, 5))
        self.assertEqual(image.GetSpacing(), (1.5, 2.0, 2.5))
        for index in ((0, 0, 0), (2, 0, 0), (0, 3, 0), (0, 0, 4), (2, 3, 4), (1, 2, 3)):
            self.assertEqual(image.GetPixel(*index), array[index])
        np.testing.assert_array_equal(native.to_numpy(image), array)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "asymmetric.nii.gz"
            affine = np.array([[-1.5, 0, 0, 10], [0, 2.0, 0, -20], [0, 0, 2.5, 5], [0, 0, 0, 1]])
            nib.save(nib.Nifti1Image(array.astype(np.float32), affine), str(path))
            read = sitk.ReadImage(str(path), sitk.sitkFloat64)
            np.testing.assert_array_equal(native.to_numpy(read), nib.load(str(path)).get_fdata())


@unittest.skipUnless(HAS_SCIPY, "scipy is not installed")
class FillHolesTest(unittest.TestCase):
    def test_hole_inside_is_filled_and_outside_background_is_kept(self):
        values = np.zeros((7, 7, 7), dtype=np.float32)
        values[1:6, 1:6, 1:6] = 1
        values[2:5, 2:5, 2:5] = 0
        filled = native.fill_holes(values)
        self.assertEqual(filled.dtype, np.float32)
        np.testing.assert_array_equal(filled[2:5, 2:5, 2:5], 1)
        self.assertEqual(filled[0, 0, 0], 0)
        self.assertEqual(int(filled.sum()), 125)

    def test_two_equal_background_components_keep_the_first_in_itk_raster_order(self):
        # (i, k) grid, j = 0; O = object. Component A = {(1,0), (2,0), (2,1)}, B = {(0,1), (0,2), (1,2)}.
        # ITK raster order (i fastest): A starts at index 1, B at 3 -> keep A, fill B.
        # (C order on the [i, j, k] array would start B first and keep B.)
        values = np.zeros((3, 1, 3), dtype=np.float32)
        for i, k in ((0, 0), (1, 1), (2, 2)):
            values[i, 0, k] = 1
        filled = native.fill_holes(values)
        for i, k in ((1, 0), (2, 0), (2, 1)):
            self.assertEqual(filled[i, 0, k], 0)
        for i, k in ((0, 1), (0, 2), (1, 2)):
            self.assertEqual(filled[i, 0, k], 1)

    def test_fractional_values_and_uint8_truncation(self):
        values = np.full((7, 7, 7), 0.2, dtype=np.float32)
        values[1:6, 1:6, 1:6] = 1
        values[1, 3, 3] = 0.7       # object (>= 0.5) but < 1: kept as 0.7, then truncated to 0
        values[5, 3, 3] = 0.5       # object at the threshold
        values[2:5, 2:5, 2:5] = 0.3  # hole (< 0.5): set to 1
        filled = native.fill_holes(values)
        self.assertEqual(filled[3, 3, 3], 1.0)
        self.assertAlmostEqual(float(filled[1, 3, 3]), 0.7, places=6)
        self.assertEqual(filled[5, 3, 3], np.float32(0.5))
        self.assertAlmostEqual(float(filled[0, 0, 0]), 0.2, places=6)
        mask = native.finish_mask(filled)
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual((mask[3, 3, 3], mask[1, 3, 3], mask[5, 3, 3], mask[0, 0, 0]), (1, 0, 0, 0))
        self.assertEqual(native.finish_mask(np.array([[[0.999, 1.0]]], dtype=np.float32)).tolist(), [[[0, 1]]])
        with self.assertRaisesRegex(RuntimeError, "nonempty"):
            native.finish_mask(np.full((2, 2, 2), 0.9, dtype=np.float32))


class ThreadTest(unittest.TestCase):
    def test_stage_cpus_then_affinity_then_cpu_count(self):
        with mock.patch.dict(os.environ, {"SLICERDSPACE_STAGE_CPUS": "3"}):
            self.assertEqual(native.default_threads(), 3)
        with mock.patch.dict(os.environ, {"SLICERDSPACE_STAGE_CPUS": "0"}), \
                mock.patch.object(native.os, "sched_getaffinity", create=True, return_value={0, 1}):
            self.assertEqual(native.default_threads(), 2)
        with mock.patch.dict(os.environ, {}, clear=False), mock.patch.object(native.os, "cpu_count", return_value=7):
            os.environ.pop("SLICERDSPACE_STAGE_CPUS", None)
            with mock.patch.object(native.os, "sched_getaffinity", create=True, side_effect=AttributeError):
                self.assertEqual(native.default_threads(), 7)

    @unittest.skipUnless(HAS_SITK and HAS_TORCH, "SimpleITK or torch is not installed")
    def test_threads_reach_torch_and_simpleitk(self):
        import SimpleITK as sitk
        import torch

        before = (torch.get_num_threads(), sitk.ProcessObject.GetGlobalDefaultNumberOfThreads())
        try:
            native.apply_threads(2)
            self.assertEqual(torch.get_num_threads(), 2)
            self.assertEqual(sitk.ProcessObject.GetGlobalDefaultNumberOfThreads(), 2)
        finally:
            torch.set_num_threads(before[0])
            sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(before[1])


@unittest.skipUnless(HAS_SITK and HAS_TORCH and HAS_SCIPY, "SimpleITK, torch or scipy is not installed")
class EndToEndTest(unittest.TestCase):
    """Tiny synthetic run: real registration, normalisation, predict/orient/aggregate, inverse, FillHoles."""

    def write_inputs(self, root):
        import SimpleITK as sitk

        template = sitk.GaussianSource(sitk.sitkFloat32, [256] * 3, [30, 38, 34], [128, 124, 130], 1000.0)
        template.SetOrigin((-128.0, -124.0, -130.0))
        template_path = root / "template.nii.gz"
        sitk.WriteImage(template, str(template_path))
        grid = sitk.Image(48, 52, 40, sitk.sitkFloat32)
        grid.SetSpacing((3.0, 3.0, 3.5))
        grid.SetOrigin((-70.0, -76.0, -68.0))
        truth = sitk.Euler3DTransform((0.0, 0.0, 0.0), 0.02, -0.03, 0.05, (4.0, -3.0, 2.0))
        baseline = sitk.Resample(template, grid, truth, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        baseline_path = root / "mean_b0.nii.gz"
        sitk.WriteImage(baseline, str(baseline_path))
        models = root / "models"
        models.mkdir()
        (models / "weights.json").write_text(json.dumps({"format": 1, "note": "test stand-in"}))
        return baseline_path, template_path, models

    def test_synthetic_volume_end_to_end(self):
        import nibabel as nib
        import torch

        class StandIn(torch.nn.Module):
            def forward(self, x, segmentation_only=True):
                return torch.sigmoid((x - 0.2) * 40)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, template, models = self.write_inputs(root)
            output = root / "out" / "mask.nii.gz"
            keep = root / "keep"
            fast = replace(native.PARAMETERS, iterations=(5, 3, 2))
            with mock.patch("shared.pnl_cnn.weights.check"), \
                    mock.patch("shared.pnl_cnn.inference.load_model", return_value=StandIn()):
                receipt = native.run(baseline, output, models, template, device="cpu", threads=2,
                                     keep_dir=keep, parameters=fast, log=lambda message: None)
            image = nib.load(str(output))
            source = nib.load(str(baseline))
            self.assertEqual(image.get_data_dtype(), np.int16)
            self.assertEqual(image.shape, source.shape)
            np.testing.assert_allclose(image.affine, source.affine, atol=1e-6)
            mask = np.asarray(image.dataobj)
            self.assertTrue(set(np.unique(mask).tolist()) <= {0, 1})
            self.assertEqual(mask[24, 26, 20], 1)
            self.assertEqual(mask[0, 0, 0], 0)
            self.assertEqual(receipt["route"], "native-sitk")
            self.assertEqual((receipt["device"], receipt["threads"]), ("cpu", 2))
            self.assertEqual(len(receipt["registration"]["levels"]), 3)
            self.assertEqual(set(receipt["phase_seconds"]),
                             {"read_inputs", "registration", "inference", "inverse_and_fill", "write"})
            self.assertTrue(receipt["simpleitk_version"] and receipt["torch_version"])
            for name in ("transform.tfm", "Warped.nii.gz", "template_mask.nii.gz", "inverse.nii.gz", "filled.nii.gz"):
                self.assertTrue((keep / name).is_file(), name)

    def test_script_mode_imports_vendored_code_and_records_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = root / "record.json"
            result = subprocess.run([sys.executable, str(SCRIPT), "--input", str(root / "b0.nii.gz"),
                                     "--output", str(root / "mask.nii.gz"), "--model-dir", str(root),
                                     "--device", "cpu", "--record", str(record)],
                                    capture_output=True, text=True, cwd=root)
            self.assertNotEqual(result.returncode, 0)
            payload = json.loads(record.read_text())
            self.assertEqual((payload["route"], payload["status"]), ("native-sitk", "failed"))
            self.assertIn("IITmean_b0_256.nii.gz", payload["error"])


if __name__ == "__main__":
    unittest.main()
