"""Chunk I/O parity with real NIfTI compression/scaling, no fitting changes."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

from shared.diffusion_io import diffusion_image


class DiffusionChunkIoTest(unittest.TestCase):
    def test_compressed_slabs_equal_original_scaled_values_and_clean_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "data.nii.gz"
            data = np.arange(3 * 4 * 5 * 6, dtype=np.int16).reshape(3, 4, 5, 6)
            image = nib.Nifti1Image(data, np.diag([2, 2, 2, 1]))
            image.header.set_slope_inter(2, 3)
            nib.save(image, source)
            before = source.read_bytes()
            expected = nib.load(source).get_fdata(dtype=np.float32)
            with diffusion_image(source, root / "work") as opened:
                for index in range(5):
                    np.testing.assert_array_equal(np.asarray(opened.dataobj[:, :, index:index+1, :]), expected[:, :, index:index+1, :])
                np.testing.assert_array_equal(opened.affine, image.affine)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(list((root / "work").iterdir()), [])

    def test_disk_space_failure_is_actionable_and_plain_nifti_needs_no_copy(self):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = nib.Nifti1Image(np.ones((2, 2, 2, 4), np.float32), np.eye(4))
            nib.save(image, root / "data.nii.gz")
            with patch("shared.diffusion_io.shutil.disk_usage", return_value=SimpleNamespace(free=0)):
                with self.assertRaises(RuntimeError):
                    with diffusion_image(root / "data.nii.gz", root / "work"):
                        pass
            nib.save(image, root / "data.nii")
            with patch("shared.diffusion_io.shutil.copyfileobj", side_effect=AssertionError("uncompressed input must not copy")):
                with diffusion_image(root / "data.nii", root / "work") as opened:
                    self.assertEqual(opened.shape, (2, 2, 2, 4))

    def test_open_fast_gzip_reader_threaded_fallback_and_corruption(self):
        import gzip
        import os
        from shared.diffusion_io import open_fast_gzip_reader

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "data.nii.gz"
            arr = np.arange(3 * 4 * 5 * 6, dtype=np.float32).reshape(3, 4, 5, 6)
            nib.save(nib.Nifti1Image(arr, np.eye(4)), source)
            expected_raw = gzip.decompress(source.read_bytes())

            with patch("shared.diffusion_io.shutil.which", return_value=None):
                with open_fast_gzip_reader(source) as stream:
                    chunks = []
                    while True:
                        part = stream.read(128)
                        if not part:
                            break
                        chunks.append(part)
                self.assertEqual(b"".join(chunks), expected_raw)

            truncated = root / "truncated.nii.gz"
            raw_gz = source.read_bytes()
            truncated.write_bytes(raw_gz[: len(raw_gz) // 2])
            with self.assertRaises((EOFError, gzip.BadGzipFile, OSError)):
                with open_fast_gzip_reader(truncated) as stream:
                    while stream.read(256):
                        pass

    def test_uncompressed_cache_lifecycle_and_invalidation(self):
        import os
        from shared.diffusion_io import ensure_cached_uncompressed_nifti

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"SLICERDSPACE_UNCOMPRESSED_CACHE_MIN_BYTES": "1"}
        ):
            subject = Path(tmp) / "sub-01"
            source = subject / "input" / "data.nii.gz"
            source.parent.mkdir(parents=True)
            arr1 = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
            nib.save(nib.Nifti1Image(arr1, np.eye(4)), source)

            cached1 = ensure_cached_uncompressed_nifti(source)
            self.assertIsNotNone(cached1)
            self.assertTrue(cached1.is_file())
            self.assertEqual(cached1.parent.name, "uncompressed_nifti")

            work = subject / "work"
            with diffusion_image(source, work) as opened:
                np.testing.assert_array_equal(np.asarray(opened.dataobj), arr1)
            self.assertEqual(list(work.iterdir()), [])

            # Modify source and verify cache invalidates & updates bit-for-bit.
            arr2 = arr1 + 10.0
            nib.save(nib.Nifti1Image(arr2, np.eye(4)), source)
            with diffusion_image(source, work) as opened2:
                np.testing.assert_array_equal(np.asarray(opened2.dataobj), arr2)


if __name__ == "__main__":
    unittest.main()
