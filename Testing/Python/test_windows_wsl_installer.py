from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
POWERSHELL_INSTALLER = REPO_ROOT / "scripts" / "install_deps_windows.ps1"
WSL_INSTALLER = REPO_ROOT / "scripts" / "wsl" / "install_runtime.sh"
BRIDGE_SOURCE = REPO_ROOT / "scripts" / "wsl" / "WslCommandBridge.cs"
RUNTIME_REQUIREMENTS = REPO_ROOT / "scripts" / "wsl" / "requirements-runtime.txt"


def _powershell_executable() -> str | None:
    configured = os.environ.get("POWERSHELL_EXECUTABLE", "").strip()
    if configured and Path(configured).is_file():
        return configured
    return shutil.which("powershell") or shutil.which("pwsh")


def _powershell_environment(temporary: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.setdefault("LOCALAPPDATA", str(temporary / "local-app-data"))
    environment.setdefault("APPDATA", str(temporary / "app-data"))
    return environment


class WindowsWslInstallerContractTest(unittest.TestCase):
    def test_reproducibility_pins_and_verified_resources_are_literal(self) -> None:
        powershell = POWERSHELL_INSTALLER.read_text(encoding="utf-8")
        bash = WSL_INSTALLER.read_text(encoding="utf-8")
        combined = powershell + "\n" + bash

        expected_values = (
            "26.3.2-3",
            "848194851a98903134187fbb4ab50efe87b003e0c0f808f97644b7524a62bf2c",
            "3.0.8",
            "2602.0",
            "hb35a746_0",
            "2203.5",
            "ha905fa9_6",
            "7.4.1",
            "5c69c594d489859e08485110678dca7f6f4cba40",
            "4532b911386d2d1bc7ebba63fdd13e65b1660623",
            "94de6275d07d7b5eeb50e4cd4d6e85fde29d5fb0",
            "c2b024980999b925ea896c49183995bfc97aeed8cbb648946dcbfa9e3ea6a5f1",
            "8ab81aed998ff417f91284a33a3c9233afaaeb314caa32d520f6123c2fd9e741",
            "72d70d4cf7a465a824ff421f7df29ef9116ab2c890ad428d312f020704cdfac7",
            "973911c073349f8b121ef67a4d378af17e7f7252883bed37588cec8ffa7c3a7d",
        )
        for expected in expected_values:
            self.assertIn(expected, combined)

        self.assertIn('"mrtrix3=${MRTRIX_VERSION}"', bash)
        self.assertIn('"fsl-eddy=${FSL_EDDY_VERSION}=${FSL_EDDY_BUILD}"', bash)
        self.assertIn('"fsl-topup=${FSL_TOPUP_VERSION}=${FSL_TOPUP_BUILD}"', bash)
        self.assertIn("https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/", bash)
        self.assertIn('export FSLDIR=%q', bash)
        self.assertIn('export FSLOUTPUTTYPE=NIFTI_GZ', bash)
        self.assertIn("download_sha256", bash)
        self.assertIn("! -path '*/__MACOSX/*'", bash)
        self.assertRegex(powershell, r"--user root --exec bash -lc")
        self.assertNotRegex(bash, r"(?m)^\s*sudo(?:\s|$)")

    def test_installer_has_plan_and_large_download_skip_paths(self) -> None:
        powershell = POWERSHELL_INSTALLER.read_text(encoding="utf-8")
        bash = WSL_INSTALLER.read_text(encoding="utf-8")

        self.assertIn("[switch]$PlanOnly", powershell)
        self.assertIn("[switch]$SkipLargeDownloads", powershell)
        self.assertIn("[switch]$SkipModelAssets", powershell)
        self.assertIn("download_model_assets.py", powershell)
        self.assertIn("slicerdsurfer_models_git", powershell)
        self.assertIn("if ($PlanOnly)", powershell)
        self.assertIn("--skip-large-downloads", powershell)
        self.assertIn("SKIP_LARGE_DOWNLOADS", bash)
        self.assertIn("--freesurfer-license", bash)
        self.assertIn("slicerdsurfer_windows_wsl_install_state", bash)
        self.assertIn("SlicerDSpace\\user_config.json", powershell)
        self.assertIn("[switch]$SkipUserConfigUpdate", powershell)
        self.assertIn('(Join-Path $mrtrixBridge "$command.exe")', powershell)
        self.assertIn("preprocessing_fsl_bin_dir", powershell)
        self.assertIn('(Join-Path $fslBridge "$command.exe")', powershell)
        self.assertNotIn("--preprocessing-backend", powershell)  # W2a (D1): preprocessing has one route
        self.assertIn('"--only-binary=:all:"', powershell)
        self.assertIn('"--platform", "win_amd64"', powershell)
        self.assertIn('"--pip-no-index"', powershell)
        self.assertIn('"--strict-exit-code"', powershell)
        self.assertIn("wheelhouse-manifest.json", powershell)
        self.assertIn('"pywin32>=306"', powershell)
        self.assertIn('"importlib-metadata>=6.0"', powershell)
        self.assertIn("windows_marker_requirements", powershell)

    def test_bridge_converts_drive_and_wsl_unc_paths(self) -> None:
        source = BRIDGE_SOURCE.read_text(encoding="utf-8")

        self.assertIn("DrivePath", source)
        self.assertIn("wslpath", source)
        self.assertIn("\\\\wsl.localhost\\", source)
        self.assertIn("\\\\wsl$\\", source)
        self.assertIn('key.Equals("map"', source)
        self.assertIn('key.Equals("arg"', source)
        self.assertIn("return process.ExitCode", source)

    def test_bridge_supervision_is_installed_for_owned_tasks(self):
        source = BRIDGE_SOURCE.read_text(encoding='utf-8')
        powershell = POWERSHELL_INSTALLER.read_text(encoding='utf-8')
        self.assertIn('supervisor_python', powershell)
        self.assertIn('run-command.py', powershell)
        self.assertIn('heartbeatTimer', source)
        self.assertIn('File.SetLastWriteTimeUtc', source)
        self.assertIn('command.Add("--heartbeat")', source)
        self.assertIn('cp "$REPO_ROOT/scripts/wsl/run_command.py"', WSL_INSTALLER.read_text())

    def test_runtime_requirements_cover_all_external_python_backends(self) -> None:
        requirements = RUNTIME_REQUIREMENTS.read_text(encoding="utf-8")
        for requirement in (
            "numpy==1.26.4",
            "dipy==1.11.0",
            "scikit-learn==1.6.1",
            "vtk==9.3.1",
            "dmri-amico==2.1.1",
            "editables==0.5",
        ):
            self.assertIn(requirement, requirements)
        self.assertNotIn("torch==", requirements)
        self.assertNotIn("--extra-index-url", requirements)
        self.assertIn("configure_device.py", WSL_INSTALLER.read_text())

    def test_gpu_selection_and_license_gate_are_explicit_and_bridges_complete(self):
        powershell = POWERSHELL_INSTALLER.read_text()
        bash = WSL_INSTALLER.read_text()
        bridge = BRIDGE_SOURCE.read_text()
        self.assertIn('[string]$Device = "auto"', powershell)
        self.assertIn('[switch]$AcceptFslLicense', powershell)
        self.assertIn('"--device", $Device, "--accept-fsl-license"', powershell)
        self.assertIn('"dwiextract", "mrcat", "python"', powershell)
        self.assertIn('"eddy_cuda.exe"', powershell)
        self.assertIn('fsl_packages+=("$FSL_CUDA_PACKAGE")', bash)
        self.assertIn('fsl-${FSL_EDDY_VERSION}-${DEVICE_RESOLVED}', bash)
        self.assertIn('if [[ "$DEVICE" == cpu ]]', bash)
        self.assertIn('CUDA_ENV="export CUDA_VISIBLE_DEVICES=-1"', bash)
        self.assertIn('Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES")', bridge)
        self.assertIn('command.Add("/usr/bin/env")', bridge)
        self.assertLess(powershell.index("if ($PlanOnly)"), powershell.index("if (-not $AcceptFslLicense)"))

    def test_plan_only_runs_without_contacting_wsl(self) -> None:
        shell = _powershell_executable()
        if not shell:
            self.skipTest("PowerShell is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            install_root = temporary_root / "windows-runtime"
            completed = subprocess.run(
                [
                    shell,
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(POWERSHELL_INSTALLER),
                    "-PlanOnly",
                    "-Distro",
                    "DefinitelyNotInstalledForPlanTest",
                    "-WindowsInstallRoot",
                    str(install_root),
                ],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
                env=_powershell_environment(temporary_root),
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            plan = json.loads((install_root / "install-plan.json").read_text(encoding="utf-8"))
            self.assertFalse(plan["execute"])
            self.assertEqual(plan["distribution"], "DefinitelyNotInstalledForPlanTest")
            self.assertEqual(plan["pinned"]["mrtrix_version"], "3.0.8")
            self.assertEqual(plan["pinned"]["fsl_eddy_version"], "2602.0")
            self.assertEqual(plan["pinned"]["fsl_eddy_build"], "hb35a746_0")
            self.assertEqual(plan["pinned"]["fsl_topup_version"], "2203.5")
            self.assertEqual(plan["pinned"]["fsl_topup_build"], "ha905fa9_6")
            self.assertEqual(plan["device"]["requested"], "auto")
            self.assertIn("Deferred", plan["device"]["resolved"])
            self.assertFalse(plan["fsl_license"]["accepted"])

    def test_url_override_without_digest_is_rejected(self) -> None:
        shell = _powershell_executable()
        if not shell:
            self.skipTest("PowerShell is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            completed = subprocess.run(
                [
                    shell,
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(POWERSHELL_INSTALLER),
                    "-PlanOnly",
                    "-WindowsInstallRoot",
                    str(temporary_root),
                    "-OrgRegUrl",
                    "https://mirror.invalid/ORG-RegAtlas-100HCP.zip",
                ],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
                env=_powershell_environment(temporary_root),
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("provide -OrgRegSha256", completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
