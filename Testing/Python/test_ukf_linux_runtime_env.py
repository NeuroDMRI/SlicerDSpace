"""Fan93 smoke 2026-11-22, finding F1: the Linux bundled UKF runtime could not load its own libraries.

The binary has no RUNPATH and ukf_gpu_process_environment removed LD_LIBRARY_PATH, so
``UKFTractography --xml`` failed with "libITKFactoryRegistration.so: cannot open shared object file".
"""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tractography import ukf


def _runtime(tmp):
    root = Path(tmp) / "SlicerDSpace" / "bundled-runtimes" / "ukf-2d2b661-linux-x86_64"
    binary = root / "bin" / "UKFTractography"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    (root / "lib").mkdir()
    return binary, root / "lib"


class LinuxBundledUkfEnvironmentTests(unittest.TestCase):
    def test_linux_standalone_runtime_uses_only_its_own_lib_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary, lib_dir = _runtime(tmp)
            with mock.patch.object(ukf.sys, "platform", "linux"):
                env = ukf.ukf_gpu_process_environment(
                    {"LD_LIBRARY_PATH": "/slicer/lib/Slicer-5.10", "PATH": "/usr/bin", "PYTHONHOME": "/slicer"},
                    binary,
                )
            self.assertEqual(env["LD_LIBRARY_PATH"], str(lib_dir))
            self.assertNotIn("PYTHONHOME", env)
            self.assertEqual(env["PATH"], "/usr/bin")

    def test_macos_standalone_runtime_keeps_loader_paths_removed(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary, _lib_dir = _runtime(tmp)
            with mock.patch.object(ukf.sys, "platform", "darwin"):
                env = ukf.ukf_gpu_process_environment({"LD_LIBRARY_PATH": "/x", "DYLD_LIBRARY_PATH": "/y"}, binary)
            self.assertNotIn("LD_LIBRARY_PATH", env)
            self.assertNotIn("DYLD_LIBRARY_PATH", env)


if __name__ == "__main__":
    unittest.main()
