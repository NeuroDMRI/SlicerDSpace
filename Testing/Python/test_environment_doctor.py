from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from shared.environment import (
    DEPENDENCY_PROBE_TIMEOUT_SECONDS,
    DependencySpec,
    EnvironmentDoctorConfig,
    _apply_default_python,
    _check_cuda_device,
    _check_spec,
    _common_command_dirs,
    _common_freesurfer_homes,
    _common_slicer_executables,
    _install_hint,
    _model_asset_spec_path,
    _parse_json_probe_output,
    _python_requirement_details,
    build_dependency_specs,
    run_environment_doctor,
    write_environment_report,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


class EnvironmentDoctorDiscoveryTest(unittest.TestCase):
    def test_wmql_doctor_checks_dependencies_in_the_selected_worker(self) -> None:
        worker = Path(sys.executable).absolute().with_name("wmql-worker.exe")
        config = EnvironmentDoctorConfig(repo_root=REPO_ROOT, targets=("bundles",), wmql_python_executable=worker)

        specs = {spec.key: spec for spec in build_dependency_specs(config)}

        self.assertEqual(specs["python:tract_querier"].python_executable, worker)
        for name in ("nibabel", "numpy", "vtk"):
            self.assertEqual(specs[f"python:wmql:{name}"].python_executable, worker)
        self.assertNotEqual(specs["python:nibabel"].python_executable, worker)

    def test_default_pythonpath_does_not_leak_into_an_external_python(self) -> None:
        default_python = Path(sys.executable).resolve()
        external_python = default_python.with_name("external-python.exe")
        default_packages = default_python.parent / "slicer-packages"
        external_packages = external_python.parent / "tractcloud-src"
        specs = (
            DependencySpec(
                key="python:default",
                kind="python",
                label="default",
                required=True,
                targets=("validation",),
                python_import="default_package",
            ),
            DependencySpec(
                key="python:external",
                kind="python",
                label="external",
                required=True,
                targets=("tractcloud",),
                python_import="tractcloud",
                python_executable=external_python,
                python_path_entries=(external_packages,),
            ),
        )

        default_spec, external_spec = _apply_default_python(
            specs,
            default_python,
            (default_packages,),
        )

        self.assertEqual(default_spec.python_executable, default_python)
        self.assertEqual(default_spec.python_path_entries, (default_packages.resolve(),))
        self.assertEqual(external_spec.python_executable, external_python.resolve())
        self.assertEqual(external_spec.python_path_entries, (external_packages,))

    def test_python_probe_prioritizes_configured_pythonpath(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package_root = Path(tmp)
            (package_root / "doctor_external_package.py").write_text(
                "VALUE = 'external'\n",
                encoding="utf-8",
            )
            status = _check_spec(
                DependencySpec(
                    key="python:doctor_external_package",
                    kind="python",
                    label="doctor_external_package",
                    required=True,
                    targets=("validation",),
                    python_import="doctor_external_package",
                    python_package="",
                    python_executable=Path(sys.executable),
                    python_path_entries=(package_root,),
                ),
                sys.platform,
            )

        self.assertEqual(status.status, "available")
        self.assertEqual(status.details["pythonpath"], [package_root.as_posix()])

    def test_external_python_probe_drops_slicer_pythonhome(self) -> None:
        # PythonSlicer exports PYTHONHOME; an external venv probed with it fails ("SRE module mismatch").
        slicer_env = {"PYTHONHOME": "/Applications/Slicer.app/Contents/lib/Python",
                      "PYTHONPATH": "/Applications/Slicer.app/Contents/lib/Slicer-5.12"}
        with mock.patch.dict(os.environ, slicer_env):
            status = _check_spec(
                DependencySpec(
                    key="python:json",
                    kind="python",
                    label="json",
                    required=True,
                    targets=("tractcloud",),
                    python_import="json",
                    python_package="",
                    python_executable=Path(sys.executable),
                ),
                sys.platform,
            )

        self.assertEqual(status.status, "available", status.details.get("error"))

    def test_python_version_requirement_rejects_old_torch(self) -> None:
        specifier, satisfied = _python_requirement_details("torch>=2.0.0", "1.8.1+cpu")
        self.assertEqual(specifier, ">=2.0.0")
        self.assertFalse(satisfied)
        self.assertTrue(_python_requirement_details("torch>=2.0.0", "2.2.2+cpu")[1])

    def test_python_probe_ignores_launcher_startup_notice(self) -> None:
        self.assertEqual(
            _parse_json_probe_output('No module named \'logic\'\n{"available": true}\n'),
            {"available": True},
        )

    def _make_mrtrix_bin(self, root: Path) -> Path:
        bin_dir = root / "mrtrix_bin"
        bin_dir.mkdir(parents=True)
        for command in ("mrconvert", "dwi2response", "dwi2fod", "mtnormalise", "tckgen", "tcksift2"):
            command_path = bin_dir / command
            command_path.write_text("#!/usr/bin/env sh\n", encoding="utf-8")
            command_path.chmod(0o755)
        return bin_dir

    def _make_windows_mrtrix_bin(self, root: Path) -> Path:
        bin_dir = root / "MRtrix3" / "bin"
        bin_dir.mkdir(parents=True)
        for command in ("mrconvert", "dwi2response", "dwi2fod", "mtnormalise", "tckgen", "tcksift2"):
            command_path = bin_dir / f"{command}.exe"
            command_path.write_text("", encoding="utf-8")
        return bin_dir

    def _make_freesurfer_home(self, root: Path) -> Path:
        freesurfer_home = root / "software" / "freesurfer"
        bin_dir = freesurfer_home / "bin"
        bin_dir.mkdir(parents=True)
        (freesurfer_home / "subjects" / "fsaverage").mkdir(parents=True)
        for command in (
            "recon-all",
            "mris_sphere",
            "mris_curvature",
            "mris_inflate",
            "mris_register",
            "mri_label2label",
            "mri_surf2surf",
            "mris_thickness",
            "mri_annotation2label",
            "mri_segstats",
        ):
            command_path = bin_dir / command
            command_path.write_text("#!/usr/bin/env sh\n", encoding="utf-8")
            command_path.chmod(0o755)
        return freesurfer_home

    def test_fsl_preprocessing_requires_corrected_dwi_mask_command(self) -> None:
        specs = build_dependency_specs(
            EnvironmentDoctorConfig(
                targets=("preprocessing",),
                repo_root=REPO_ROOT,
            )
        )
        self.assertIn("command:dwi2mask", {spec.key for spec in specs})

    def test_w2a_preprocessing_target_needs_the_one_route_tools_on_every_os(self) -> None:
        """W2a (D1): DIPY, MRtrix3 (dwifslpreproc, dwi2mask) and FSL (eddy, topup); no dwidenoise or mrdegibbs."""
        for platform_name in ("Linux", "Darwin", "Windows"):
            with self.subTest(platform_name=platform_name):
                specs = build_dependency_specs(EnvironmentDoctorConfig(
                    targets=("preprocessing",), repo_root=REPO_ROOT, platform_name=platform_name))
                keys = {spec.key: spec for spec in specs if "preprocessing" in spec.targets}
                for command, group in (("dwifslpreproc", "mrtrix"), ("dwi2mask", "mrtrix"), ("eddy", "fsl"), ("topup", "fsl")):
                    self.assertTrue(keys[f"command:{command}"].required, command)
                    self.assertEqual(keys[f"command:{command}"].install_group, group, command)
                self.assertTrue(keys["python:dipy"].required)
                self.assertNotIn("command:dwidenoise", keys)
                self.assertNotIn("command:mrdegibbs", keys)

    def test_dti_runtime_requires_real_slicer_python(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake_python = Path(tmp) / "python-without-slicer"
            fake_python.write_text(
                f"#!{sys.executable}\nprint('{{\"available\": false}}')\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            result = run_environment_doctor(
                EnvironmentDoctorConfig(
                    targets=("microstructure", "tractography"),
                    repo_root=REPO_ROOT,
                    python_executable=fake_python,
                    tractography_backend="dti_det",
                )
            )

        runtime = next(item for item in result.statuses if item.key == "python:slicer.util")
        self.assertEqual(runtime.status, "missing")
        self.assertEqual(runtime.details["package"], "")
        self.assertIn("slicer-runtime", result.missing_external_groups)
        self.assertNotIn("slicer.util", result.missing_python_packages)
        self.assertEqual(set(result.blocked_targets), {"microstructure", "tractography"})

    def test_user_path_source_overrides_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp).resolve()
            user_bin = self._make_mrtrix_bin(tmp_path / "user")
            env_bin = self._make_mrtrix_bin(tmp_path / "env")
            old_env = os.environ.get("SLICERDSPACE_MRTRIX_BIN_DIR")
            os.environ["SLICERDSPACE_MRTRIX_BIN_DIR"] = env_bin.as_posix()
            try:
                result = run_environment_doctor(
                    EnvironmentDoctorConfig(
                        targets=("tractography",),
                        repo_root=REPO_ROOT,
                        tractography_backend="csd_act",
                        mrtrix_bin_dir=user_bin,
                    )
                )
            finally:
                if old_env is None:
                    os.environ.pop("SLICERDSPACE_MRTRIX_BIN_DIR", None)
                else:
                    os.environ["SLICERDSPACE_MRTRIX_BIN_DIR"] = old_env

            mrconvert = next(item for item in result.statuses if item.key == "command:mrconvert")
            self.assertEqual(mrconvert.status, "available")
            self.assertEqual(mrconvert.details["detected_source"], "user setting")
            self.assertTrue(mrconvert.details["resolved"].startswith(user_bin.as_posix()))

    def test_explicit_wma_paths_do_not_leave_environment_false_alarms(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wma_root = root / "whitematteranalysis"
            wma_bin = wma_root / "bin"
            wma_bin.mkdir(parents=True)
            wma_command = wma_bin / "wm_apply_ORG_atlas_to_subject.sh"
            wma_command.write_text("#!/usr/bin/env sh\n", encoding="utf-8")
            wma_command.chmod(0o755)
            atlas_root = root / "ORG-Atlases"
            for child in ("ORG-RegAtlas-100HCP", "ORG-800FC-100HCP"):
                (atlas_root / child).mkdir(parents=True)
            slicer_executable = root / "Slicer"
            slicer_executable.write_text("#!/usr/bin/env sh\n", encoding="utf-8")
            slicer_executable.chmod(0o755)

            specs = build_dependency_specs(
                EnvironmentDoctorConfig(
                    targets=("wma",),
                    repo_root=REPO_ROOT,
                    wma_root=wma_root,
                    wma_atlas_dir=atlas_root,
                    slicer_executable=slicer_executable,
                )
            )
            statuses = {
                spec.key: _check_spec(spec, "Linux")
                for spec in specs
                if spec.key in {"env:wma_root", "env:slicer_executable"}
            }

            self.assertEqual(statuses["env:wma_root"].status, "available")
            self.assertEqual(statuses["env:wma_root"].details["detected_source"], "user setting")
            self.assertEqual(statuses["env:slicer_executable"].status, "available")
            self.assertEqual(statuses["env:slicer_executable"].details["detected_source"], "user setting")

    def test_wma_checks_the_python_packages_the_author_package_imports(self) -> None:
        # Real-run finding (macOS EDDEN run, 2026-10-23): the doctor reported WMA ready, then
        # `import whitematteranalysis` failed in Slicer's Python with "No module named 'xlrd'".
        specs = build_dependency_specs(EnvironmentDoctorConfig(targets=("wma",), repo_root=REPO_ROOT))
        keys = {spec.key for spec in specs}
        for module in ("joblib", "scipy", "xlrd"):
            self.assertIn(f"python:wma:{module}", keys)

    def test_c3_wma_package_checks_use_the_configured_slicers_pythonslicer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            contents = Path(tmp).resolve() / "Slicer.app/Contents"
            (contents / "MacOS").mkdir(parents=True)
            (contents / "bin").mkdir()
            launcher = contents / "MacOS/Slicer"
            launcher.touch()
            pythonslicer = contents / "bin/PythonSlicer"
            pythonslicer.touch()
            with mock.patch.object(sys, "platform", "darwin"):
                specs = {spec.key: spec for spec in build_dependency_specs(EnvironmentDoctorConfig(
                    targets=("wma",), repo_root=REPO_ROOT, slicer_executable=launcher))}
                for module in ("joblib", "scipy", "xlrd"):
                    self.assertEqual(specs[f"python:wma:{module}"].python_executable, pythonslicer, module)
                # Fallback: a launcher with no PythonSlicer keeps the doctor's own Python.
                bare = Path(tmp).resolve() / "bare/Slicer"
                bare.parent.mkdir()
                bare.touch()
                specs = {spec.key: spec for spec in build_dependency_specs(EnvironmentDoctorConfig(
                    targets=("wma",), repo_root=REPO_ROOT, slicer_executable=bare))}
                for module in ("joblib", "scipy", "xlrd"):
                    self.assertNotEqual(getattr(specs[f"python:wma:{module}"].python_executable, "name", ""),
                                        "PythonSlicer", module)

    def test_n1_wma_scipy_check_stays_separate_from_the_host_scipy_check(self) -> None:
        # N1: WMA probes PythonSlicer; segmentation/saf probe the host interpreter. They must not merge.
        host_python = Path(sys.executable).resolve()
        with tempfile.TemporaryDirectory() as tmp:
            contents = Path(tmp).resolve() / "Slicer.app/Contents"
            (contents / "MacOS").mkdir(parents=True)
            (contents / "bin").mkdir()
            launcher = contents / "MacOS/Slicer"
            launcher.touch()
            pythonslicer = contents / "bin/PythonSlicer"
            pythonslicer.touch()
            extra = Path(tmp).resolve() / "extra_site"
            extra.mkdir()
            for host_target in ("segmentation", "saf"):
                with self.subTest(host_target=host_target), mock.patch.object(sys, "platform", "darwin"):
                    specs = {spec.key: spec for spec in build_dependency_specs(EnvironmentDoctorConfig(
                        targets=(host_target, "wma"), repo_root=REPO_ROOT, slicer_executable=launcher,
                        python_executable=host_python, python_path_entries=(extra,)))}
                    host = specs["python:scipy"]
                    wma = specs["python:wma:scipy"]
                    self.assertEqual(host.targets, (host_target,))
                    self.assertEqual(wma.targets, ("wma",))
                    self.assertEqual(wma.python_executable, pythonslicer)
                    self.assertNotEqual(host.python_executable, pythonslicer)
                    self.assertIn(extra, host.python_path_entries)
                    self.assertNotIn("WMA", host.label)
                    self.assertIn("WMA", wma.label)
                    self.assertIn(pythonslicer.as_posix(), _install_hint(wma, "Darwin"))

    def test_environment_path_source_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_bin = self._make_mrtrix_bin(Path(tmp))
            old_env = os.environ.get("SLICERDSPACE_MRTRIX_BIN_DIR")
            os.environ["SLICERDSPACE_MRTRIX_BIN_DIR"] = env_bin.as_posix()
            try:
                result = run_environment_doctor(
                    EnvironmentDoctorConfig(
                        targets=("tractography",),
                        repo_root=REPO_ROOT,
                        tractography_backend="csd_act",
                    )
                )
            finally:
                if old_env is None:
                    os.environ.pop("SLICERDSPACE_MRTRIX_BIN_DIR", None)
                else:
                    os.environ["SLICERDSPACE_MRTRIX_BIN_DIR"] = old_env

            mrconvert = next(item for item in result.statuses if item.key == "command:mrconvert")
            self.assertEqual(mrconvert.status, "available")
            self.assertEqual(mrconvert.details["detected_source"], "env:SLICERDSPACE_MRTRIX_BIN_DIR")

    def test_path_source_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path_bin = self._make_mrtrix_bin(Path(tmp))
            old_path = os.environ.get("PATH", "")
            old_env = os.environ.get("SLICERDSPACE_MRTRIX_BIN_DIR")
            os.environ.pop("SLICERDSPACE_MRTRIX_BIN_DIR", None)
            os.environ["PATH"] = path_bin.as_posix() + os.pathsep + old_path
            try:
                result = run_environment_doctor(
                    EnvironmentDoctorConfig(
                        targets=("tractography",),
                        repo_root=REPO_ROOT,
                        tractography_backend="csd_act",
                    )
                )
            finally:
                os.environ["PATH"] = old_path
                if old_env is not None:
                    os.environ["SLICERDSPACE_MRTRIX_BIN_DIR"] = old_env

            mrconvert = next(item for item in result.statuses if item.key == "command:mrconvert")
            self.assertEqual(mrconvert.status, "available")
            self.assertEqual(mrconvert.details["detected_source"], "PATH")

    def test_common_path_source_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            common_bin = Path(tmp) / "common"
            common_bin.mkdir()
            fake_tool = common_bin / "fake_tool"
            fake_tool.write_text("#!/usr/bin/env sh\n", encoding="utf-8")
            fake_tool.chmod(0o755)
            status = _check_spec(
                DependencySpec(
                    key="command:fake_tool",
                    kind="command",
                    label="fake_tool",
                    required=True,
                    targets=("fake",),
                    command="fake_tool",
                    install_group="fake",
                    common_paths=(common_bin,),
                ),
                "Linux",
            )
            self.assertEqual(status.status, "available")
            self.assertEqual(status.details["detected_source"], "common path")

    def test_windows_common_path_checks_executable_suffixes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            common_bin = Path(tmp) / "common"
            common_bin.mkdir()
            fake_tool = common_bin / "fake_tool.exe"
            fake_tool.write_text("", encoding="utf-8")
            status = _check_spec(
                DependencySpec(
                    key="command:fake_tool",
                    kind="command",
                    label="fake_tool",
                    required=True,
                    targets=("fake",),
                    command="fake_tool",
                    install_group="fake",
                    common_paths=(common_bin,),
                ),
                "Windows",
            )
            self.assertEqual(status.status, "available")
            self.assertEqual(status.details["detected_source"], "common path")
            self.assertEqual(status.details["resolved"], fake_tool.as_posix())
            self.assertIn(fake_tool.as_posix(), status.details["checked_locations"])

    def test_platform_override_is_reported_and_used_for_windows_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp).resolve()
            user_bin = self._make_windows_mrtrix_bin(tmp_path)
            result = run_environment_doctor(
                EnvironmentDoctorConfig(
                    targets=("tractography",),
                    repo_root=REPO_ROOT,
                    platform_name="Windows",
                    tractography_backend="csd_act",
                    mrtrix_bin_dir=user_bin,
                )
            )
            self.assertEqual(result.platform, "Windows")
            self.assertTrue(result.host_platform)
            mrconvert = next(item for item in result.statuses if item.key == "command:mrconvert")
            self.assertEqual(mrconvert.status, "available")
            self.assertEqual(mrconvert.details["detected_source"], "user setting")
            self.assertEqual(mrconvert.details["resolved"], (user_bin / "mrconvert.exe").as_posix())
            self.assertIn((user_bin / "mrconvert.exe").as_posix(), mrconvert.details["checked_locations"])

            written = write_environment_report(result, tmp_path / "report")
            profile = json.loads(written["profile"].read_text(encoding="utf-8"))
            self.assertEqual(profile["platform"], "Windows")
            self.assertEqual(profile["host_platform"], result.host_platform)
            self.assertTrue(profile["portability_policy"]["revalidate_after_import"])

    def test_platform_override_is_reported_and_used_for_macos_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp).resolve()
            user_bin = self._make_mrtrix_bin(tmp_path)
            result = run_environment_doctor(
                EnvironmentDoctorConfig(
                    targets=("tractography",),
                    repo_root=REPO_ROOT,
                    platform_name="Darwin",
                    tractography_backend="csd_act",
                    mrtrix_bin_dir=user_bin,
                )
            )

            self.assertEqual(result.platform, "Darwin")
            mrconvert = next(item for item in result.statuses if item.key == "command:mrconvert")
            self.assertEqual(mrconvert.status, "available")
            self.assertEqual(mrconvert.details["detected_source"], "user setting")
            self.assertEqual(mrconvert.details["resolved"], (user_bin / "mrconvert").as_posix())

            written = write_environment_report(result, tmp_path / "report")
            profile = json.loads(written["profile"].read_text(encoding="utf-8"))
            self.assertEqual(profile["platform"], "Darwin")
            # The override is honored regardless of the host: when the test
            # itself runs on a Darwin host, the override and host platform
            # legitimately coincide, so only assert divergence when the host
            # is something else (the common CI case).
            if platform.system() != "Darwin":
                self.assertNotEqual(profile["platform"], profile["host_platform"])
            else:
                self.assertEqual(profile["platform"], profile["host_platform"])
            self.assertTrue(profile["portability_policy"]["revalidate_after_import"])

    def test_common_platform_paths_include_macos_and_windows_runtime_locations(self) -> None:
        mac_slicers = {path.as_posix() for path in _common_slicer_executables("Darwin")}
        win_slicers = {path.as_posix() for path in _common_slicer_executables("Windows")}
        mac_mrtrix = {path.as_posix() for path in _common_command_dirs("mrtrix", "Darwin")}
        win_mrtrix = {path.as_posix() for path in _common_command_dirs("mrtrix", "Windows")}
        win_fsl = {path.as_posix() for path in _common_command_dirs("fsl", "Windows")}
        mac_freesurfer = {path.as_posix() for path in _common_freesurfer_homes("Darwin")}
        win_freesurfer = {path.as_posix() for path in _common_freesurfer_homes("Windows")}

        self.assertIn("/Applications/Slicer.app/Contents/MacOS/Slicer", mac_slicers)
        self.assertIn("C:/Program Files/Slicer 5.10.0/Slicer.exe", win_slicers)
        self.assertIn("/Applications/MRtrix3.app/Contents/bin", mac_mrtrix)
        self.assertIn("C:/Program Files/MRtrix3/bin", win_mrtrix)
        self.assertIn("C:/Program Files/FSL/bin", win_fsl)
        self.assertIn("/Applications/FreeSurfer", mac_freesurfer)
        self.assertIn("C:/Program Files/FreeSurfer", win_freesurfer)

    def test_python_install_requirement_survives_default_python_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing_python = Path(tmp) / "missing-python"
            result = run_environment_doctor(
                EnvironmentDoctorConfig(
                    targets=("bundles",),
                    repo_root=REPO_ROOT,
                    python_executable=missing_python,
                )
            )
            tract_querier = next(item for item in result.statuses if item.key == "python:tract_querier")
            self.assertEqual(
                tract_querier.details["install_requirement"],
                "git+https://github.com/demianw/tract_querier.git",
            )
            self.assertIn("git+https://github.com/demianw/tract_querier.git", result.missing_python_packages)

    def test_b10_wmql_git_pip_line_is_not_listed_when_the_checkout_passes(self) -> None:
        url = "git+https://github.com/demianw/tract_querier.git"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "tract_querier"
            (root / "scripts").mkdir(parents=True)
            (root / "scripts" / "cli_tract_querier.py").write_text("# fixture\n")
            missing_python = Path(tmp) / "missing-python"
            with_root = run_environment_doctor(EnvironmentDoctorConfig(
                targets=("bundles",), repo_root=REPO_ROOT, python_executable=missing_python, tract_querier_root=root))
            by_key = {item.key: item for item in with_root.statuses}
            self.assertEqual(by_key["path:tract_querier_root"].status, "available")
            self.assertEqual(by_key["python:tract_querier"].status, "missing")
            self.assertNotIn(url, with_root.missing_python_packages)
            without_root = run_environment_doctor(EnvironmentDoctorConfig(
                targets=("bundles",), repo_root=REPO_ROOT, python_executable=missing_python,
                tract_querier_root=Path(tmp) / "absent"))
            self.assertEqual({item.key: item for item in without_root.statuses}["path:tract_querier_root"].status, "missing")
            self.assertIn(url, without_root.missing_python_packages)

    def test_freesurfer_home_is_discovered_from_local_software_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Resolve up front: on macOS `tempfile` returns a path under the
            # `/tmp` -> `/private/tmp` symlink, while the code under test
            # resolves user-supplied paths before reporting them.
            tmp_path = Path(tmp).resolve()
            freesurfer_home = self._make_freesurfer_home(tmp_path)
            old_home = os.environ.get("HOME")
            old_user = os.environ.get("USER")
            old_fs_home = os.environ.get("FREESURFER_HOME")
            old_slicer_fs_home = os.environ.get("SLICERDSPACE_FREESURFER_HOME")
            os.environ["HOME"] = tmp_path.as_posix()
            os.environ["USER"] = "slicerddsurfer-test-user"
            os.environ.pop("FREESURFER_HOME", None)
            os.environ.pop("SLICERDSPACE_FREESURFER_HOME", None)
            try:
                # Verify local discovery independently of /opt or PATH on this host.
                self.assertIn(freesurfer_home, _common_freesurfer_homes())
                from shutil import which
                def fixture_which(command, *args, **kwargs):
                    return None if command in {'recon-all', 'mri_surf2surf'} else which(command, *args, **kwargs)
                with mock.patch('shared.environment._common_freesurfer_homes', return_value=(freesurfer_home,)), \
                        mock.patch('shared.environment.shutil.which', side_effect=fixture_which):
                    result = run_environment_doctor(
                        EnvironmentDoctorConfig(
                            targets=("surface_postprocess",),
                            repo_root=REPO_ROOT,
                        )
                    )
            finally:
                if old_home is None:
                    os.environ.pop("HOME", None)
                else:
                    os.environ["HOME"] = old_home
                if old_user is None:
                    os.environ.pop("USER", None)
                else:
                    os.environ["USER"] = old_user
                if old_fs_home is None:
                    os.environ.pop("FREESURFER_HOME", None)
                else:
                    os.environ["FREESURFER_HOME"] = old_fs_home
                if old_slicer_fs_home is None:
                    os.environ.pop("SLICERDSPACE_FREESURFER_HOME", None)
                else:
                    os.environ["SLICERDSPACE_FREESURFER_HOME"] = old_slicer_fs_home

            freesurfer_env = next(item for item in result.statuses if item.key == "env:freesurfer_home")
            surface_mapping = next(item for item in result.statuses if item.key == "command:mri_surf2surf")
            rtree = next(item for item in result.statuses if item.key == "python:rtree")
            self.assertEqual(freesurfer_env.status, "available")
            self.assertEqual(freesurfer_env.details["detected_source"], "common path")
            self.assertEqual(freesurfer_env.details["resolved"], freesurfer_home.as_posix())
            self.assertEqual(surface_mapping.status, "available")
            self.assertEqual(surface_mapping.details["detected_source"], "common path")
            self.assertTrue(surface_mapping.details["resolved"].startswith((freesurfer_home / "bin").as_posix()))
            self.assertTrue(rtree.required)
            self.assertEqual(rtree.details["package"], "rtree")

    def test_cli_dependency_config_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            subject_root = tmp_path / "subject"
            subject_root.mkdir()
            user_bin = self._make_mrtrix_bin(tmp_path)
            config_path = tmp_path / "deps.json"
            first = subprocess.run(
                [
                    sys.executable,
                    (REPO_ROOT / "run_subject_pipeline.py").as_posix(),
                    "--subject-root",
                    subject_root.as_posix(),
                    "--no-user-config",
                    "--targets",
                    "tractography",
                    "--doctor",
                    "--mrtrix-bin-dir",
                    user_bin.as_posix(),
                    "--save-dependency-config",
                    config_path.as_posix(),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("Dependency config written", first.stderr)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(config["external_dependencies"]["mrtrix_bin_dir"], user_bin.as_posix())

            second = subprocess.run(
                [
                    sys.executable,
                    (REPO_ROOT / "run_subject_pipeline.py").as_posix(),
                    "--subject-root",
                    subject_root.as_posix(),
                    "--no-user-config",
                    "--targets",
                    "tractography",
                    "--doctor",
                    "--dependency-config",
                    config_path.as_posix(),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(second.stdout)
            mrconvert = next(
                item for item in payload["environment"]["dependencies"] if item["key"] == "command:mrconvert"
            )
            self.assertEqual(mrconvert["details"]["detected_source"], "user setting")

    def test_environment_doctor_resolves_conda_environment_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected_python = root / "envs" / "slicerdsurfer" / "bin" / "python"
            expected_python.parent.mkdir(parents=True)
            expected_python.write_text("", encoding="utf-8")
            conda = root / "conda"
            conda.write_text(
                "#!/usr/bin/env python3\n"
                "import sys\n"
                "assert sys.argv[1:4] == ['run', '-n', 'slicerdsurfer']\n"
                f"print({expected_python.as_posix()!r})\n",
                encoding="utf-8",
            )
            conda.chmod(0o755)

            result = run_environment_doctor(
                EnvironmentDoctorConfig(
                    targets=("validation",),
                    repo_root=REPO_ROOT,
                    python_environment="slicerdsurfer",
                    conda_executable=conda,
                )
            )

            self.assertEqual(result.python_executable, expected_python.as_posix())

    def test_cuda_probe_reports_timeout_status_instead_of_hanging(self) -> None:
        timeout_exc = subprocess.TimeoutExpired(cmd=["python", "-c", "..."], timeout=DEPENDENCY_PROBE_TIMEOUT_SECONDS)
        with mock.patch("shared.environment.subprocess.run", side_effect=timeout_exc) as mock_run:
            available, details = _check_cuda_device("cuda:0", None)

        self.assertFalse(available)
        self.assertIn("timed out", details["error"])
        self.assertEqual(mock_run.call_args.kwargs["timeout"], DEPENDENCY_PROBE_TIMEOUT_SECONDS)

    def test_python_package_probe_reports_timeout_status_instead_of_hanging(self) -> None:
        timeout_exc = subprocess.TimeoutExpired(cmd=["python", "-c", "..."], timeout=DEPENDENCY_PROBE_TIMEOUT_SECONDS)
        spec = DependencySpec(
            key="python:slow_package",
            kind="python",
            label="slow_package",
            required=True,
            targets=("validation",),
            python_import="slow_package",
            python_package="slow_package",
            python_executable=Path(sys.executable).with_name("external-python.exe"),
        )

        with mock.patch("shared.environment.subprocess.run", side_effect=timeout_exc) as mock_run:
            status = _check_spec(spec, "Linux")

        self.assertEqual(status.status, "missing")
        self.assertIn("timed out", status.details["error"])
        self.assertEqual(mock_run.call_args.kwargs["timeout"], DEPENDENCY_PROBE_TIMEOUT_SECONDS)


class ModelAssetIntegrityDoesNotAbortTheDoctorScanTest(unittest.TestCase):
    """A tampered model checkpoint must become one "missing" dependency item
    with an actionable note, not an exception that aborts build_dependency_specs()
    (and therefore run_environment_doctor()) before every other target's
    checks even run.
    """

    def setUp(self) -> None:
        import shared.model_assets as model_assets_module

        self.model_assets_module = model_assets_module
        self._cache_tmp = tempfile.TemporaryDirectory()
        cache_path = Path(self._cache_tmp.name) / "model_asset_sha256_cache.json"
        model_assets_module._SHA256_CACHE = None
        self._cache_path_patch = mock.patch.object(
            model_assets_module, "_sha256_cache_path", return_value=cache_path
        )
        self._cache_path_patch.start()
        self.addCleanup(self._cache_path_patch.stop)
        self.addCleanup(setattr, model_assets_module, "_SHA256_CACHE", None)
        self.addCleanup(self._cache_tmp.cleanup)

    def test_model_asset_spec_path_reports_failure_without_raising(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            user_root = tmp_path / "user-assets"
            repo_root = tmp_path / "repo"
            planted = user_root / "segmentation" / "weights" / "dti_axial.pt"
            planted.parent.mkdir(parents=True)
            planted.write_bytes(b"a planted, untrusted checkpoint")
            manifest_dir = repo_root / "models"
            manifest_dir.mkdir(parents=True)
            (manifest_dir / "assets_manifest.json").write_text(
                json.dumps({"asset_records": {"segmentation/weights/dti_axial.pt": {"sha256": "0" * 64}}}),
                encoding="utf-8",
            )
            config = EnvironmentDoctorConfig(targets=("segmentation",), repo_root=repo_root)

            with mock.patch.object(
                self.model_assets_module, "default_user_model_assets_root", return_value=user_root
            ):
                path, candidates, notes = _model_asset_spec_path(config, "segmentation/weights")

            self.assertFalse(path.exists())
            self.assertIn("Integrity check failed", notes)
            self.assertIn("re-download", notes)

    def test_build_dependency_specs_completes_and_reports_the_blocker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            user_root = tmp_path / "user-assets"
            repo_root = tmp_path / "repo"
            planted = user_root / "segmentation" / "weights" / "dti_axial.pt"
            planted.parent.mkdir(parents=True)
            planted.write_bytes(b"a planted, untrusted checkpoint")
            manifest_dir = repo_root / "models"
            manifest_dir.mkdir(parents=True)
            (manifest_dir / "assets_manifest.json").write_text(
                json.dumps({"asset_records": {"segmentation/weights/dti_axial.pt": {"sha256": "0" * 64}}}),
                encoding="utf-8",
            )
            config = EnvironmentDoctorConfig(targets=("segmentation",), repo_root=repo_root)

            with mock.patch.object(
                self.model_assets_module, "default_user_model_assets_root", return_value=user_root
            ):
                # Must not raise ModelAssetIntegrityError.
                specs = build_dependency_specs(config)

            ddseg_spec = next(spec for spec in specs if spec.key == "path:ddseg_weights")
            self.assertIn("Integrity check failed", ddseg_spec.notes)
            status = _check_spec(ddseg_spec, "Linux")
            self.assertEqual(status.status, "missing")
            self.assertIn("Integrity check failed", status.details["notes"])


if __name__ == "__main__":
    unittest.main()
