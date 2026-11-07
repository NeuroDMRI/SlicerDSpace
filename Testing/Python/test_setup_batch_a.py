"""Batch-backlog part A: setup file fixes (A1 W-B-CARRY-1, A2, A3, A4, A5, A6, A7)."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import setup_selected  # noqa: E402


class SlicerPythonIdentityTests(unittest.TestCase):
    def test_a1_venv_symlink_to_python_real_is_external(self):
        """A1 W-B-CARRY-1: a venv bin/python linked to python-real is not Slicer's Python. Fails before: realpath matched."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve() / "Slicer"
            real = root / "lib/Python/bin/python-real"
            real.parent.mkdir(parents=True)
            real.write_text("")
            venv_python = Path(folder).resolve() / "venv/bin/python"
            venv_python.parent.mkdir(parents=True)
            venv_python.symlink_to(real)
            with patch("sys.executable", str(real)):
                self.assertTrue(setup_selected.is_running_slicer_python(real))
                self.assertFalse(setup_selected.is_running_slicer_python(venv_python))

    def test_a1_real_bin_layout_matches_python_real_and_pythonslicer(self):
        """A1 (round-3 NB6): the layout bin/python-real + bin/PythonSlicer; an unrelated symlink to either is external."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve() / "Slicer"
            launcher, real = root / "bin/PythonSlicer", root / "bin/python-real"
            real.parent.mkdir(parents=True)
            launcher.write_text("")
            real.write_text("")
            link = Path(folder).resolve() / "venv/bin/python"
            link.parent.mkdir(parents=True)
            link.symlink_to(real)
            with patch("sys.executable", str(real)):
                self.assertTrue(setup_selected.is_running_slicer_python(launcher))
                self.assertTrue(setup_selected.is_running_slicer_python(real))
                self.assertFalse(setup_selected.is_running_slicer_python(link))

    def test_a1_nb5_running_pythons_computed_once_per_call(self):
        """A1 (NB5): is_running_slicer_python calls running_slicer_pythons once."""
        with patch.object(setup_selected, "running_slicer_pythons", return_value=set()) as found:
            setup_selected.is_running_slicer_python("/x/python")
        self.assertEqual(found.call_count, 1)


if __name__ == "__main__":
    unittest.main()
