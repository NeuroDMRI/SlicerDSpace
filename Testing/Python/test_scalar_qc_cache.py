"""Adversarial scalar-QC/cache regressions using synthetic, not clinical, data."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

from microstructure.pipeline import (
    DTI_KEYS, NODDI_KEYS, _output_paths, _export_dti_scalar_maps, _write_noddi_qc,
    _noddi_outputs_complete, _free_water_outputs_complete, scalar_qc_current,
)


class ScalarQcCacheTest(unittest.TestCase):
    def fixture(self, root):
        mask = root / "mask.nii.gz"
        self.save(mask, 1)
        native = SimpleNamespace()
        for attr, value in (("fa_nii_path", .5), ("mean_diffusivity_nii_path", .002),
                            ("max_eig_nii_path", .003), ("mid_eig_nii_path", .002),
                            ("min_eig_nii_path", .001), ("trace_nii_path", .006)):
            path = root / (attr + ".nii.gz")
            self.save(path, value)
            setattr(native, attr, path)
        native.dec_nrrd_path = root / "dec.nrrd"  # D6
        native.dec_nrrd_path.write_bytes(b"dec")
        native.manifest_path = root / "native_manifest.json"
        native.manifest_path.write_text("{}")
        config = SimpleNamespace(subject_id="test", native_bundle=native, raw_input_bundle=SimpleNamespace(mask_path=mask),
                                 force_recompute=False, noddi_max_isotropic_fraction_for_tissue=.7)
        return config, _output_paths("test", root / "outputs")

    @staticmethod
    def save(path, value, affine=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(np.full((3, 3, 3), value, dtype=np.float32),
                               np.eye(4) if affine is None else affine), path)

    def test_dti_cache_avoids_copy_and_recomputes_rd_after_source_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, paths = self.fixture(Path(tmp))
            _export_dti_scalar_maps(config, paths)
            with patch("microstructure.pipeline._copy_if_needed", side_effect=AssertionError("cache should not copy")):
                _export_dti_scalar_maps(config, paths)
            self.save(config.native_bundle.mid_eig_nii_path, .0015)
            self.save(config.native_bundle.trace_nii_path, .0055)
            self.save(config.native_bundle.mean_diffusivity_nii_path, .0055 / 3)
            _export_dti_scalar_maps(config, paths)
            np.testing.assert_allclose(nib.load(paths["rd"]).get_fdata(), .00125)
            self.assertTrue(scalar_qc_current(paths["dti_qc"], [paths[key] for key in DTI_KEYS]))
            self.save(paths["fa"], 2)
            self.assertFalse(scalar_qc_current(paths["dti_qc"], [paths[key] for key in DTI_KEYS]))

    def test_noddi_empty_validity_nan_residual_and_wrong_grid_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, paths = self.fixture(Path(tmp))
            for key in NODDI_KEYS:
                self.save(paths[key], 1 if "mask" in key else .2)
            _write_noddi_qc(config, paths)
            self.assertTrue(_noddi_outputs_complete(paths))
            for key, value in (("noddi_tissue_valid_mask", 0), ("noddi_nrmse", np.nan), ("noddi_rmse", -.1)):
                self.save(paths[key], value)
                with self.assertRaises(RuntimeError):
                    _write_noddi_qc(config, paths)
                self.assertFalse(_noddi_outputs_complete(paths))
                self.save(paths[key], 1 if "mask" in key else .2)
            self.save(paths["noddi_ndi"], .2, np.diag([2, 1, 1, 1]))
            with self.assertRaises(ValueError):
                _write_noddi_qc(config, paths)
            self.assertEqual(json.loads(paths["noddi_qc"].read_text())["status"], "failed")

    def test_slicer_clamp_semantics_do_not_hide_wrong_md_or_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, paths = self.fixture(Path(tmp))
            self.save(config.native_bundle.min_eig_nii_path, 0)
            self.save(config.native_bundle.mean_diffusivity_nii_path, .005 / 3)
            self.save(config.native_bundle.trace_nii_path, .0045)
            _export_dti_scalar_maps(config, paths)
            record = json.loads(paths["dti_qc"].read_text())
            self.assertEqual(record["trace_clamp_boundary_voxels"], 27)
            self.save(config.native_bundle.trace_nii_path, .006)
            with self.assertRaises(RuntimeError):
                _export_dti_scalar_maps(config, paths)
            self.save(config.native_bundle.trace_nii_path, .0045)
            self.save(config.native_bundle.mean_diffusivity_nii_path, .0001)
            with self.assertRaises(RuntimeError):
                _export_dti_scalar_maps(config, paths)

    def test_failed_legacy_or_corrupt_qc_never_reuses_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, paths = self.fixture(Path(tmp))
            for key in ("free_water_qc", "noddi_qc"):
                paths[key].parent.mkdir(parents=True, exist_ok=True)
                for value in ('{"status":"completed"}', '{"status":"failed"}', 'broken'):
                    paths[key].write_text(value)
                    self.assertFalse(_noddi_outputs_complete(paths))
                    self.assertFalse(_free_water_outputs_complete(paths))

    def test_amico_links_refresh_when_source_changes_including_windows_copy_fallback(self):
        from microstructure.pipeline import _link_or_copy

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source1, source2, dest = root / "one", root / "two", root / "cache" / "input"
            source1.write_text("old")
            source2.write_text("new-input")
            _link_or_copy(source1, dest)
            _link_or_copy(source2, dest)
            self.assertEqual(dest.read_text(), "new-input")
            if dest.is_symlink():
                dest.unlink()
            with patch.object(Path, "symlink_to", side_effect=OSError("Windows privilege unavailable")):
                _link_or_copy(source1, dest)
                _link_or_copy(source2, dest)
                self.assertEqual(dest.read_text(), "new-input")
                with patch("microstructure.pipeline.shutil.copy2", side_effect=AssertionError("verified copy reused")):
                    _link_or_copy(source2, dest)

    def test_noddi_auto_threads_bounded_and_manual_cpu_setting_preserved(self):
        from microstructure.pipeline import _noddi_threads

        with patch("microstructure.pipeline.os.sched_getaffinity", return_value=set(range(144)), create=True):
            self.assertEqual(_noddi_threads(SimpleNamespace(noddi_nthreads=None)), 8)
            self.assertEqual(_noddi_threads(SimpleNamespace(noddi_nthreads=1)), 1)
            with self.assertRaises(ValueError):
                _noddi_threads(SimpleNamespace(noddi_nthreads=0))


if __name__ == "__main__":
    unittest.main()
