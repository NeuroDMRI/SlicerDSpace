"""W-B6: dcm2niix comes from the pinned pip wheel; the importer looks there before PATH."""

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.setup_python_packages import DCM2NIIX_REQUIREMENT, dcm2niix_pending
from shared.dicom_input import converter_path
from shared.input_import import find_dcm2niix, packaged_dcm2niix


def fake_module(folder, *, name="dcm2niix", executable=True):
    path = Path(folder) / name
    if executable:
        path.write_text("#!/bin/sh\n")
        path.chmod(0o755)
    return types.SimpleNamespace(bin_path=path, bin=str(path))


class Dcm2niixLookupTests(unittest.TestCase):
    def test_the_pin_is_the_spec_version(self):
        self.assertEqual(DCM2NIIX_REQUIREMENT, "dcm2niix==1.0.20261029")
        self.assertFalse(dcm2niix_pending("1.0.20261029"))
        self.assertTrue(dcm2niix_pending("1.0.20250506"))

    def test_package_module_is_used_before_path(self):
        with tempfile.TemporaryDirectory() as temp:
            module = fake_module(temp)
            with patch.dict(sys.modules, {"dcm2niix": module}), \
                    patch("shared.input_import.shutil.which", return_value="/usr/bin/dcm2niix") as which:
                self.assertEqual(packaged_dcm2niix(), module.bin_path)
                self.assertEqual(find_dcm2niix(), module.bin_path.as_posix())
                which.assert_not_called()

    def test_path_is_used_when_the_package_is_absent_or_has_no_executable(self):
        with tempfile.TemporaryDirectory() as temp:
            for modules in ({"dcm2niix": None}, {"dcm2niix": fake_module(temp, executable=False)}):
                with patch.dict(sys.modules, modules), \
                        patch("shared.input_import.shutil.which", return_value="/usr/bin/dcm2niix"):
                    self.assertIsNone(packaged_dcm2niix())
                    self.assertEqual(find_dcm2niix(), "/usr/bin/dcm2niix")

    def test_windows_exe_next_to_the_module_path_is_found(self):
        with tempfile.TemporaryDirectory() as temp:
            exe = Path(temp) / "dcm2niix.exe"
            exe.write_text("MZ")
            exe.chmod(0o755)
            module = types.SimpleNamespace(bin_path=Path(temp) / "dcm2niix", bin=str(Path(temp) / "dcm2niix"))
            with patch.dict(sys.modules, {"dcm2niix": module}):
                self.assertEqual(packaged_dcm2niix(), exe)

    def test_s_n4_odd_attributes_fall_back_and_the_wheel_layout_site_bin_is_found(self):
        with tempfile.TemporaryDirectory() as temp:
            site = Path(temp) / "site-packages"
            (site / "dcm2niix").mkdir(parents=True)
            (site / "bin").mkdir()
            exe = site / "bin" / "dcm2niix"
            exe.write_text("#!/bin/sh\n")
            exe.chmod(0o755)
            odd = types.SimpleNamespace(bin_path=None, bin=5, __file__=str(site / "dcm2niix" / "__init__.py"))
            with patch.dict(sys.modules, {"dcm2niix": odd}):
                self.assertEqual(packaged_dcm2niix(), exe)  # fails before: Path(5) raised TypeError
            nothing = types.SimpleNamespace(bin_path=["x"], bin=object())
            with patch.dict(sys.modules, {"dcm2niix": nothing}), \
                    patch("shared.input_import.shutil.which", return_value="/usr/bin/dcm2niix"):
                self.assertIsNone(packaged_dcm2niix())
                self.assertEqual(find_dcm2niix(), "/usr/bin/dcm2niix")

    def test_dicom_converter_order_is_package_then_path_then_dwiconvert(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"SLICERDSPACE_DWICONVERT_EXECUTABLE": ""}):
            module = fake_module(temp)
            with patch.dict(sys.modules, {"dcm2niix": module}), \
                    patch("shared.dicom_input.shutil.which", return_value="/usr/bin/DWIConvert"):
                self.assertEqual(converter_path(), ("dcm2niix", module.bin_path))
            which = {"dcm2niix": "/usr/bin/dcm2niix", "DWIConvert": "/usr/bin/DWIConvert"}
            with patch.dict(sys.modules, {"dcm2niix": None}), \
                    patch("shared.input_import.shutil.which", side_effect=which.get), \
                    patch("shared.dicom_input.shutil.which", side_effect=which.get):
                self.assertEqual(converter_path(), ("dcm2niix", Path("/usr/bin/dcm2niix")))
            which.pop("dcm2niix")
            with patch.dict(sys.modules, {"dcm2niix": None}), \
                    patch("shared.input_import.shutil.which", side_effect=which.get), \
                    patch("shared.dicom_input.shutil.which", side_effect=which.get):
                self.assertEqual(converter_path(), ("dwiconvert", Path("/usr/bin/DWIConvert")))

    def test_import_plan_command_uses_the_package_executable(self):
        from shared.input_import import InputImportConfig, build_import_plan
        with tempfile.TemporaryDirectory() as temp:
            module = fake_module(temp)
            source = Path(temp) / "dicom"
            source.mkdir()
            (source / "IM0001.dcm").write_text("dicom\n")
            config = InputImportConfig(source=source, subject_root=Path(temp) / "sub", subject_id="sub",
                                       source_kind="dicom", dry_run=True)
            with patch.dict(sys.modules, {"dcm2niix": module}), \
                    patch("shared.input_import.shutil.which", return_value="/usr/bin/dcm2niix"):
                plan = build_import_plan(config)
            self.assertEqual(plan["backend"], "dcm2niix")
            self.assertEqual(plan["command"][0], module.bin_path.as_posix())


if __name__ == "__main__":
    unittest.main()
