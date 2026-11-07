#!/usr/bin/env python3
"""Tests for segmentation/dti_slicer.py's external command construction."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anatomy.segmentation.dti_slicer import run_dti_feature_extraction


class DtiSlicerCommandConstructionTest(unittest.TestCase):
    def test_windows_executables_and_macos_app_layout(self):
        """W3: PythonSlicer is found in the Linux, macOS .app and Windows layouts; slicer_ext is not needed."""
        for layout in ("linux", "macos", "windows"):
            with self.subTest(layout=layout), tempfile.TemporaryDirectory() as tmp:
                windows = layout == "windows"
                root = Path(tmp) / ("Slicer.app" if layout == "macos" else "Slicer")
                runtime = root / "Contents" if layout == "macos" else root
                cli = runtime / "lib/Slicer-5.10/cli-modules"
                cli.mkdir(parents=True)
                suffix = ".exe" if windows else ""
                (cli / ("DWIConvert" + suffix)).touch()
                python = runtime / "bin" / ("PythonSlicer" + suffix)
                python.parent.mkdir(parents=True)
                python.touch()
                launcher = runtime / ("Slicer.exe" if windows else "MacOS/Slicer" if layout == "macos" else "Slicer")
                launcher.parent.mkdir(parents=True, exist_ok=True)
                launcher.touch()
                with mock.patch("anatomy.segmentation.dti_slicer.subprocess.run") as run:
                    with self.assertRaises(RuntimeError):
                        run_dti_feature_extraction(root / "data.nii.gz", root / "bvals", root / "bvecs",
                                                   root / "output", root, None)
                self.assertEqual(run.call_count, 2)
                self.assertEqual(run.call_args_list[0].args[0][:3],
                                 [str(launcher), "--launch", str(cli / ("DWIConvert" + suffix))])
                teem = run.call_args_list[1].args[0]
                self.assertEqual(teem[0], str(python))
                self.assertEqual(Path(teem[1]), REPO_ROOT / "shared/teem_dti.py")
                self.assertEqual(teem[2:5], ["estimate", "--enumeration", "LS"])
                scalars = dict(item.split("=", 1) for index, item in enumerate(teem) if teem[index - 1] == "--scalar")
                output = root / "output"
                self.assertEqual(scalars, {
                    "FractionalAnisotropy": str(output / "DTI_FA.nrrd"), "Trace": str(output / "DTI_MD.nrrd"),
                    "MaxEigenvalue": str(output / "DTI_E1.nrrd"), "MidEigenvalue": str(output / "DTI_E2.nrrd"),
                    "MinEigenvalue": str(output / "DTI_E3.nrrd")})
                self.assertEqual(teem[-3:], [str(output / "DWI.nrrd"), str(output / "DTI.nrrd"), str(output / "b0.nrrd")])

    def test_w3_missing_pythonslicer_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "Slicer"
            (root / "lib/Slicer-5.10/cli-modules").mkdir(parents=True)
            with mock.patch("anatomy.segmentation.dti_slicer.subprocess.run") as run:
                with self.assertRaisesRegex(FileNotFoundError, "PythonSlicer"):
                    run_dti_feature_extraction(root / "d.nii.gz", root / "bvals", root / "bvecs", root / "out", root)
            run.assert_not_called()

    def test_w3_run_ddseg_still_accepts_the_deprecated_slicer_ext_option(self):
        import importlib
        run_ddseg = importlib.import_module("anatomy.segmentation.run_ddseg")
        with mock.patch.object(sys, "argv", ["run_ddseg.py", "--mask", "m", "--output", "o", "--weights-dir", "w",
                                             "--slicer-ext", "/old/SlicerDMRI"]), \
                mock.patch.object(run_ddseg, "run_ddseg") as run:
            run_ddseg.main()
        self.assertEqual(run.call_args.args[0].slicer_ext, Path("/old/SlicerDMRI").resolve())
        self.assertIn("deprecated", run_ddseg.build_parser().format_help())

    def _make_slicer_base(self, root: Path) -> Path:
        slicer_base = root / "SlicerBase"
        (slicer_base / "lib" / "Slicer-5.10").mkdir(parents=True)
        (slicer_base / "bin").mkdir()
        (slicer_base / "bin" / "PythonSlicer").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        (slicer_base / "Slicer").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        return slicer_base

    def test_paths_with_spaces_and_semicolons_are_passed_literally(self) -> None:
        # Commands used to be built as shell=True strings with a naive
        # f'"{path}"' quote; a path containing a shell metacharacter such as
        # ";" could break out of that quoting. Argument lists passed without
        # shell=True make this impossible: each path is one argv element,
        # unmodified, with no shell parsing involved at all.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            slicer_base = self._make_slicer_base(root)

            input_root = root / "input dir; rm -rf ~"
            input_root.mkdir()
            nii_file = input_root / "data.nii.gz"
            bval_file = input_root / "bvals"
            bvec_file = input_root / "bvecs"
            for path in (nii_file, bval_file, bvec_file):
                path.write_text("stub", encoding="utf-8")
            output_folder = root / "out put"

            with mock.patch("anatomy.segmentation.dti_slicer.subprocess.run") as mock_run:
                mock_run.return_value = mock.Mock(returncode=0)
                with self.assertRaises(RuntimeError):
                    # The mocked subprocess never creates DTI_E3.nrrd, so the
                    # function raises at the end; that is expected here and
                    # orthogonal to the argv assertions below.
                    run_dti_feature_extraction(
                        nii_file=nii_file,
                        bval_file=bval_file,
                        bvec_file=bvec_file,
                        output_folder=output_folder,
                        slicer_base=slicer_base,
                        slicer_ext=slicer_base,
                    )

            self.assertTrue(mock_run.call_args_list)
            first_call = mock_run.call_args_list[0]
            (argv,) = first_call.args
            self.assertIsInstance(argv, list)
            self.assertNotIn(True, [first_call.kwargs.get("shell", False)])
            self.assertIn(str(bvec_file), argv)
            self.assertIn(str(bval_file), argv)
            self.assertIn(str(nii_file), argv)
            self.assertIn(str(output_folder / "DWI.nrrd"), argv)

            for call_args in mock_run.call_args_list:
                self.assertFalse(call_args.kwargs.get("shell", False))
                self.assertIsInstance(call_args.args[0], list)

    def test_launcher_wraps_cli_with_slicer_executable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            slicer_base = self._make_slicer_base(root)
            nii_file = root / "data.nii.gz"
            bval_file = root / "bvals"
            bvec_file = root / "bvecs"
            for path in (nii_file, bval_file, bvec_file):
                path.write_text("stub", encoding="utf-8")
            output_folder = root / "out"

            with mock.patch("anatomy.segmentation.dti_slicer.subprocess.run") as mock_run:
                mock_run.return_value = mock.Mock(returncode=0)
                with self.assertRaises(RuntimeError):
                    run_dti_feature_extraction(
                        nii_file=nii_file,
                        bval_file=bval_file,
                        bvec_file=bvec_file,
                        output_folder=output_folder,
                        slicer_base=slicer_base,
                        slicer_ext=slicer_base,
                    )

            self.assertEqual(mock_run.call_count, 2)
            first_argv = mock_run.call_args_list[0].args[0]
            self.assertEqual(first_argv[0], str(slicer_base / "Slicer"))
            self.assertEqual(first_argv[1], "--launch")
            self.assertIn("DWIConvert", first_argv[2])


if __name__ == "__main__":
    unittest.main()
