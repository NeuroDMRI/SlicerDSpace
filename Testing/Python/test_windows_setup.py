"""W-B5: WSL distribution detection and the Windows ps1 command line (mocked; Windows status: implemented, not run)."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from shared.selected_setup import build_setup_plan, windows_install_command, windows_runtime_plan
from shared.wsl_bridge import (check_bridge_distribution, list_distributions, parse_distributions,
                               saved_distribution, select_distribution)

UBUNTU = [{"name": "Ubuntu", "state": "Running", "version": 2, "default": True}]


def wsl_output(*rows):
    lines = ["  NAME            STATE           VERSION"]
    for name, state, version, default in rows:
        lines.append(("* " if default else "  ") + f"{name:<15} {state:<15} {version}")
    return ("\r\n".join(lines) + "\r\n").encode("utf-16")  # UTF-16LE with BOM, as wsl.exe prints it


def runner(output=b"", code=0):
    return lambda *args, **kwargs: subprocess.CompletedProcess(args[0], code, stdout=output, stderr=b"")


class WslDetectionTests(unittest.TestCase):
    def test_wsl_missing_or_failing_gives_no_distribution(self):
        def missing(*args, **kwargs):
            raise FileNotFoundError("wsl.exe")
        self.assertEqual(list_distributions(missing), [])
        self.assertEqual(list_distributions(runner(b"", 1)), [])
        self.assertIsNone(select_distribution([]))

    def test_version_1_ubuntu_is_not_selected(self):
        found = list_distributions(runner(wsl_output(("Ubuntu", "Running", 1, True), ("Debian", "Stopped", 2, False))))
        self.assertEqual([item["version"] for item in found], [1, 2])
        self.assertIsNone(select_distribution(found))

    def test_default_ubuntu_version_2_is_selected_from_utf16_output(self):
        found = list_distributions(runner(wsl_output(("Debian", "Stopped", 2, False), ("Ubuntu-22.04", "Running", 2, True))))
        self.assertEqual(found[1], {"name": "Ubuntu-22.04", "state": "Running", "version": 2, "default": True})
        self.assertEqual(select_distribution(found), "Ubuntu-22.04")

    def test_first_ubuntu_version_2_is_used_when_the_default_is_not_usable(self):
        found = parse_distributions(wsl_output(("Debian", "Running", 2, True), ("Ubuntu-20.04", "Stopped", 1, False),
                                               ("Ubuntu-24.04", "Stopped", 2, False)))
        self.assertEqual(select_distribution(found), "Ubuntu-24.04")


class Ps1CommandTests(unittest.TestCase):
    def build(self, **kwargs):
        defaults = dict(slicer_executable=r"C:\Slicer\Slicer.exe")
        return windows_install_command("Ubuntu-22.04", **{**defaults, **kwargs})

    def test_fixed_switches_and_the_selected_distribution(self):
        command = self.build(install_freesurfer=False)
        self.assertEqual(command[:5], ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File"])
        self.assertTrue(command[5].endswith("install_deps_windows.ps1"))
        self.assertEqual(command[command.index("-Distro") + 1], "Ubuntu-22.04")
        self.assertEqual(command[command.index("-SlicerExecutable") + 1], r"C:\Slicer\Slicer.exe")
        for switch in ("-AcceptFslLicense", "-SkipModelAssets", "-SkipSlicerPythonInstall"):
            self.assertIn(switch, command)

    def test_freesurfer_with_a_licence_file_passes_the_file_and_no_skip_switch(self):
        command = self.build(install_freesurfer=True, license_file=r"C:\fs\license.txt")
        self.assertEqual(command[command.index("-FreeSurferLicensePath") + 1], r"C:\fs\license.txt")
        self.assertNotIn("-SkipFreeSurfer", command)
        self.assertNotIn("-SkipLargeDownloads", command)

    def test_freesurfer_not_installed_skips_it(self):
        self.assertIn("-SkipFreeSurfer", self.build(install_freesurfer=False))
        self.assertNotIn("-FreeSurferLicensePath", self.build(install_freesurfer=False))

    def test_runtime_plan_covers_the_freesurfer_cases(self):
        with tempfile.TemporaryDirectory() as folder:
            licence = Path(folder) / "license.txt"
            licence.write_text("user@example.org\n12345\n")
            with patch.dict("os.environ", {"FS_LICENSE": "", "FREESURFER_LICENSE": ""}):
                # needed + file found: the file is passed, no skip switches
                found = windows_runtime_plan({"mrtrix", "freesurfer"}, ("tractography",), wsl_distributions=UBUNTU,
                                             license_file=licence)
                command = found["command"]
                self.assertEqual(command[command.index("-FreeSurferLicensePath") + 1], str(licence.resolve()))
                self.assertNotIn("-SkipLargeDownloads", command)
                self.assertNotIn("-SkipFreeSurfer", command)
                self.assertIn("freesurfer", found["covers"])
                # needed + no file: skipped this run, and the pending item stays (freesurfer is not covered)
                missing = windows_runtime_plan({"mrtrix", "freesurfer"}, ("tractography",), wsl_distributions=UBUNTU,
                                               license_file=Path(folder) / "absent.txt")
                self.assertIn("-SkipFreeSurfer", missing["command"])
                self.assertIn("-SkipLargeDownloads", missing["command"])
                self.assertNotIn("freesurfer", missing["covers"])
                # not needed: skipped
                other = windows_runtime_plan({"mrtrix"}, ("tractography",), wsl_distributions=UBUNTU)
                self.assertIn("-SkipFreeSurfer", other["command"])

    def test_skip_large_downloads_only_without_freesurfer_install_and_without_wma(self):
        with_wma = windows_runtime_plan({"mrtrix"}, ("wma",), wsl_distributions=UBUNTU)
        self.assertNotIn("-SkipLargeDownloads", with_wma["command"])
        without = windows_runtime_plan({"fsl"}, ("bundles",), wsl_distributions=UBUNTU)
        self.assertIn("-SkipLargeDownloads", without["command"])

    def test_no_distribution_means_no_command_and_a_pending_item(self):
        runtime = windows_runtime_plan({"mrtrix"}, ("tractography",), wsl_distributions=[])
        self.assertIsNone(runtime["command"])
        mrtrix = SimpleNamespace(key="command:dwifslpreproc", label="MRtrix", install_group="mrtrix", kind="command",
                                 targets=("preprocessing",), required=True, status="missing", install_hint="Install MRtrix3",
                                 details={})
        result = SimpleNamespace(statuses=(mrtrix,), targets=("preprocessing",), blocker_count=1)
        plan = build_setup_plan(result, system="Windows", machine="AMD64", wsl_distributions=[],
                                closure_targets=("preprocessing",))
        self.assertIn("wsl --install -d Ubuntu", [item["hint"] for item in plan["manual_items"]])
        self.assertIsNone(plan["windows_runtime"]["command"])
        ready = build_setup_plan(result, system="Windows", machine="AMD64", closure_targets=("preprocessing",),
                                 wsl_distributions=UBUNTU, slicer_executable=r"C:\Slicer\Slicer.exe")
        self.assertEqual(ready["manual_items"], [])
        self.assertTrue(ready["fsl_license_required"])
        self.assertEqual(ready["windows_runtime"]["distribution"], "Ubuntu")


class SavedDistributionTests(unittest.TestCase):
    def test_saved_distribution_and_bridge_mismatch(self):
        self.assertEqual(saved_distribution({"wsl_distribution": " Ubuntu-22.04 "}), "Ubuntu-22.04")
        self.assertIsNone(saved_distribution({}))
        with tempfile.TemporaryDirectory() as folder:
            bridge = Path(folder) / "dwifslpreproc.exe"
            Path(str(bridge) + ".slicerdsurfer-owner").write_text("owner")
            Path(str(bridge) + ".conf").write_text("distribution=Ubuntu\ntarget=/usr/bin/env\narg=/x/dwifslpreproc\n")
            self.assertEqual(check_bridge_distribution(bridge, "Ubuntu")["distribution"], "Ubuntu")
            with self.assertRaisesRegex(ValueError, "saved distribution is 'Ubuntu-22.04'"):
                check_bridge_distribution(bridge, "Ubuntu-22.04")


class WindowsReceiptTests(unittest.TestCase):
    def bridges(self, root, distribution):
        folders = {}
        for name, key in (("mrtrix", "mrtrix_bin_dir"), ("mrtrix2", "preprocessing_mrtrix_bin_dir"),
                          ("fsl", "preprocessing_fsl_bin_dir")):
            folder = Path(root) / name
            folder.mkdir()
            exe = folder / "tool.exe"
            Path(str(exe) + ".slicerdsurfer-owner").write_text("owner")
            Path(str(exe) + ".conf").write_text(f"distribution={distribution}\ntarget=/usr/bin/env\n")
            folders[key] = str(folder)
        return {"kind": "slicerdsurfer_user_config", "external_dependencies": folders}

    def test_receipt_saves_bridge_folders_and_the_distribution(self):
        from shared.selected_setup import apply_setup_receipt
        from shared.user_config import load_user_config
        with tempfile.TemporaryDirectory() as root:
            config = Path(root) / "user_config.json"
            receipt = self.bridges(root, "Ubuntu-22.04")
            with patch.dict("os.environ", {"SLICERDSPACE_USER_CONFIG": str(config)}):
                step = {"id": "windows_runtime", "distribution": "Ubuntu-22.04"}
                settings = apply_setup_receipt(step, receipt)
                self.assertEqual(settings, receipt["external_dependencies"])
                saved = load_user_config()
                self.assertEqual(saved["wsl_distribution"], "Ubuntu-22.04")
                self.assertEqual(saved["external_dependencies"]["preprocessing_fsl_bin_dir"], settings["preprocessing_fsl_bin_dir"])
                self.assertEqual(saved_distribution(saved), "Ubuntu-22.04")
                with self.assertRaises(ValueError):
                    apply_setup_receipt({"id": "windows_runtime", "distribution": "Debian"}, receipt)
                with self.assertRaises(ValueError):
                    apply_setup_receipt(step, {"external_dependencies": {}})


if __name__ == "__main__":
    unittest.main()
