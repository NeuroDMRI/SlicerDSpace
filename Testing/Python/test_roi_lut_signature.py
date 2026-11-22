"""fan93 pass 2 finding L1: a code overlay rewrites FreeSurferColorLUT.txt with the same bytes; ROI tables must stay current."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from microstructure.pipeline import _compute_roi_stats, roi_statistics_current  # noqa: E402


class RoiLutSignatureTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.roi = self.volume("roi", [1, 1, 2, 2])
        self.fa = self.volume("fa", [0.2, 0.4, 0.6, 0.8])
        self.lut = self.root / "lut.txt"
        self.lut.write_text("1 RegionOne 1 2 3 0\n2 RegionTwo 4 5 6 0\n", encoding="utf-8")
        self.json_path, self.csv_path = self.root / "stats.json", self.root / "stats.csv"
        _compute_roi_stats(metric_paths={"fa": self.fa}, roi_label_map=self.roi, lut_path=self.lut,
                           csv_path=self.csv_path, json_path=self.json_path)

    def volume(self, name, values):
        path = self.root / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(np.asarray(values, dtype=np.float32).reshape(4, 1, 1), np.eye(4)), path)
        return path

    def current(self):
        return roi_statistics_current(self.json_path, self.csv_path, roi_label_map=self.roi, required_metrics=("fa",))

    def test_rewrite_with_same_bytes_keeps_table_current(self):
        self.assertTrue(self.current())
        stat = self.lut.stat()
        os.utime(self.lut, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**10))
        self.assertTrue(self.current())

    def test_changed_lut_content_makes_table_stale(self):
        self.lut.write_text("1 RegionOne 1 2 3 0\n2 RegionTwo 9 9 9 0\n", encoding="utf-8")
        self.assertFalse(self.current())


if __name__ == "__main__":
    unittest.main()
