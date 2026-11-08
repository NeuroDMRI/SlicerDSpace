"""Adapter contract tests; mocked MATLAB is not scientific MK-Curve validation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

from microstructure.mkcurve_matlab import _matlab_literal, _prepare_inputs, inspect_mkcurve_backend, run_mkcurve


class MKCurveAdapterTests(unittest.TestCase):
    def test_missing_backend_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            record = inspect_mkcurve_backend(Path(directory) / "matlab", Path(directory) / "source")
            self.assertFalse(record["available"])
            self.assertTrue(record["missing_required"])

    def test_matlab_literal_escapes_quotes_not_shell(self):
        self.assertEqual(_matlab_literal("/tmp/it's a; path"), "'/tmp/it''s a; path'")
        with self.assertRaises(ValueError):
            _matlab_literal("/tmp/path\nexit")

    def _inputs(self, directory):
        paths = tuple(directory / name for name in ("data.nii.gz", "bvals", "bvecs", "mask.nii.gz"))
        data = np.full((2, 3, 2, 4), 5, dtype=np.float32)
        data[..., 0] = 10
        data[..., 1] = 14
        mask = np.ones((2, 3, 2), dtype=np.uint8)
        nib.save(nib.Nifti1Image(data, np.diag([1.25, 1.25, 1.25, 1])), str(paths[0]))
        np.savetxt(paths[1], [5, 45, 1000, 2000])
        np.savetxt(paths[2], np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]]).T)
        nib.save(nib.Nifti1Image(mask, np.diag([1.25, 1.25, 1.25, 1])), str(paths[3]))
        return paths

    def test_baseline_conversion_is_explicit_without_source_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._inputs(root)
            original = [path.read_bytes() for path in paths]
            work = root / "work"
            work.mkdir()
            prepared, reference, record = _prepare_inputs(*paths, work, 50)
            loaded = nib.load(str(prepared[0]), mmap=False)
            data = loaded.get_fdata()
            self.assertEqual(data.shape, (2, 3, 2, 3))
            np.testing.assert_array_equal(data[..., 0], 12)
            np.testing.assert_array_equal(np.loadtxt(prepared[1]), [0, 1000, 2000])
            self.assertEqual(record["baseline_count"], 2)
            np.testing.assert_array_equal(loaded.affine, reference.affine)
            self.assertEqual([path.read_bytes() for path in paths], original)

    def test_mask_geometry_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._inputs(root)
            nib.save(nib.Nifti1Image(np.ones((2, 3, 2)), np.eye(4)), str(paths[3]))
            with self.assertRaisesRegex(ValueError, "geometry"):
                _prepare_inputs(*paths, root, 50)

    def test_unavailable_method_does_not_run_subprocess(self):
        with tempfile.TemporaryDirectory() as directory, patch("microstructure.mkcurve_matlab.subprocess.Popen") as process:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "MK-Curve unavailable"):
                run_mkcurve(*self._inputs(root), root / "out", matlab_executable=root / "missing",
                            mkcurve_root=root / "missing_source")
            process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
