"""CP-1: a UKF CLI linked to another Slicer's VTK must not be selected (pure-Python ABI guard)."""

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tractography import ukf


def make_cli(root, slicer="Slicer-5.12", exe_bytes=b"MZ\x00vtkcommon-9.5.dll\x00vtksys-9.5.dll\x00", lib_bytes=None,
             exe_name="UKFTractography.exe", lib_name="UKFBase.dll"):
    cli_dir = Path(root) / "lib" / slicer / "cli-modules"
    cli_dir.mkdir(parents=True, exist_ok=True)
    exe = cli_dir / exe_name
    exe.write_bytes(exe_bytes)
    if lib_bytes is not None:
        (cli_dir / lib_name).write_bytes(lib_bytes)
    return exe


class UkfVtkAbiTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)
        ukf._VTK_SCAN_CACHE.clear()

    def running(self, version):
        return mock.patch.object(ukf, "_running_vtk_version", return_value=version)

    def resolve(self, version, rejected_cli=None):
        env = {k: v for k, v in os.environ.items()
               if k not in ("SLICERDSPACE_UKF_EXECUTABLE", "UKF_GPU_EXECUTABLE")}
        with self.running(version), mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(ukf, "_extension_search_roots", return_value=[self.root]), \
                mock.patch.object(ukf.shutil, "which", return_value=None), \
                mock.patch("shared.pipeline.resolve_cli_executable",
                           side_effect=(lambda _n: str(rejected_cli)) if rejected_cli else RuntimeError("no registry")):
            return ukf.resolve_executable()

    def test_mismatch_in_executable_is_detected(self):
        exe = make_cli(self.root)
        with self.running((9, 6)):
            self.assertEqual(ukf.ukf_vtk_mismatch(exe), ("9.5", "9.6"))

    def test_mismatch_in_sibling_ukfbase_library_is_detected(self):
        exe = make_cli(self.root, exe_bytes=b"MZ no vtk names", lib_bytes=b"xx vtkio-9.5.dll xx")
        with self.running((9, 6)):
            self.assertEqual(ukf.ukf_vtk_mismatch(exe), ("9.5", "9.6"))

    def test_linux_so_names_are_detected(self):
        exe = make_cli(self.root, exe_bytes=b"\x7fELF libvtkCommonCore-9.5.so.1 libvtkIOLegacy-9.5.so.1",
                       exe_name="UKFTractography", lib_bytes=b"libvtkCommonCore-9.5.so.1", lib_name="libUKFBase.so")
        with self.running((9, 6)):
            self.assertEqual(ukf.ukf_vtk_mismatch(exe), ("9.5", "9.6"))

    def test_matching_version_does_not_block(self):
        exe = make_cli(self.root)
        with self.running((9, 5)):
            self.assertIsNone(ukf.ukf_vtk_mismatch(exe))

    def test_stray_extra_vtk_version_does_not_block_when_running_version_is_linked(self):
        """Review P2-4 (CP-1 fix round): one stray vtk*-X.Y string must not reject a CLI that links the running VTK."""
        exe = make_cli(self.root, exe_bytes=b"vtkcommon-9.6.dll vtksys-9.6.dll docs mention vtkio-9.5.dll")
        with self.running((9, 6)):
            self.assertIsNone(ukf.ukf_vtk_mismatch(exe))

    def test_error_message_names_extension_manager_remedy(self):
        """Review P2-6 (CP-1 fix round): the error tells the user to install UKF for the running Slicer."""
        make_cli(self.root)
        with self.assertRaises(RuntimeError) as ctx:
            self.resolve((9, 6))
        self.assertIn("Extension Manager", str(ctx.exception))

    def test_no_versioned_vtk_name_does_not_block(self):
        exe = make_cli(self.root, exe_bytes=b"standalone ukf, statically linked")
        with self.running((9, 6)):
            self.assertIsNone(ukf.ukf_vtk_mismatch(exe))

    def test_unknown_running_vtk_does_not_block(self):
        exe = make_cli(self.root)
        with self.running(None):
            self.assertIsNone(ukf.ukf_vtk_mismatch(exe))

    def test_standalone_runtime_outside_cli_modules_is_not_checked(self):
        standalone = self.root / "bundled-runtimes" / "ukf-darwin-arm64" / "bin" / "UKFTractography"
        standalone.parent.mkdir(parents=True)
        standalone.write_bytes(b"vtkcommon-9.5.dll")
        with self.running((9, 6)):
            self.assertIsNone(ukf.ukf_vtk_mismatch(standalone))

    def test_scan_is_cached_per_path(self):
        exe = make_cli(self.root)
        with self.running((9, 6)), mock.patch.object(ukf, "_scan_vtk_versions", wraps=ukf._scan_vtk_versions) as scan:
            ukf.ukf_vtk_mismatch(exe)
            ukf.ukf_vtk_mismatch(exe)
        self.assertEqual(scan.call_count, 1)

    def test_resolve_raises_clear_error_for_mismatched_cli(self):
        make_cli(self.root)
        with self.assertRaises(RuntimeError) as ctx:
            self.resolve((9, 6))
        message = str(ctx.exception)
        self.assertIn("9.5", message)
        self.assertIn("9.6", message)
        self.assertIn("different 3D Slicer version", message)
        self.assertIn("5.10", message)

    def test_resolve_error_when_slicer_registry_returns_mismatched_cli(self):
        exe = make_cli(self.root)
        with self.assertRaises(RuntimeError) as ctx:
            self.resolve((9, 6), rejected_cli=exe)
        self.assertIn("different 3D Slicer version", str(ctx.exception))

    def test_resolve_returns_matching_cli(self):
        exe = make_cli(self.root)
        self.assertEqual(self.resolve((9, 5)), exe.resolve())

    def test_resolve_falls_back_to_standalone_runtime(self):
        make_cli(self.root)
        standalone = self.root / "bundled-runtimes" / "ukf-win-x64" / "bin" / "UKFTractography.exe"
        standalone.parent.mkdir(parents=True)
        standalone.write_bytes(b"standalone")
        self.assertEqual(self.resolve((9, 6)), standalone.resolve())

    def test_resolve_skips_mismatched_cli_for_compatible_one(self):
        make_cli(self.root, slicer="Slicer-5.10")
        good = make_cli(self.root, slicer="Slicer-5.12", exe_bytes=b"MZ vtkcommon-9.6.dll")
        # the Slicer registry offers the mismatched one first; the search must still find the good one
        bad = self.root / "lib" / "Slicer-5.10" / "cli-modules" / "UKFTractography.exe"
        self.assertEqual(self.resolve((9, 6), rejected_cli=bad), good.resolve())

    def test_explicit_executable_is_never_checked(self):
        exe = make_cli(self.root)
        with self.running((9, 6)):
            self.assertEqual(ukf.resolve_executable(str(exe)), exe.resolve())

    def test_dwiconvert_is_never_checked(self):
        make_cli(self.root, exe_name="DWIConvert.exe", exe_bytes=b"vtkcommon-9.5.dll")
        with mock.patch.object(ukf, "_extension_search_roots", return_value=[self.root]), \
                mock.patch.object(ukf.shutil, "which", return_value=None), \
                mock.patch("shared.pipeline.resolve_cli_executable", side_effect=RuntimeError("no registry")), \
                mock.patch.dict(os.environ, {}, clear=False), self.running((9, 6)):
            found = ukf.resolve_executable(name="DWIConvert")
        self.assertEqual(found.name, "DWIConvert.exe")


if __name__ == "__main__":
    unittest.main()
