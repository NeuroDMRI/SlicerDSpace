"""Selected-route planning, phase orchestration and honest readiness states."""

from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import Mock, patch

from shared.selected_setup import build_setup_plan
from shared.target_closure import expand_closure
from shared.ui_selected_setup import extension_states, prepare_selected_environment
from shared.environment import DependencySpec, EnvironmentDoctorConfig, run_environment_doctor


def item(key, group="python", kind="python", targets=("validation",), package="numpy>=1", required=True):
    return SimpleNamespace(key=key, label=key, install_group=group, kind=kind, targets=targets,
                           required=required, status="missing", install_hint="Review " + key,
                           details={"package": package} if package else {})


def result(*items, targets=("validation",), blockers=None):
    return SimpleNamespace(statuses=items, targets=targets,
                           blocker_count=len(items) if blockers is None else blockers)


def fake_jobs(started):
    """B7: stand-in for start_setup_job. It records the step and writes the receipt the script would write."""
    def start(widget, title, commands, done):
        started.append((title, commands))
        for _program, args, _env in commands:
            if "--result" in args:
                Path(args[args.index("--result") + 1]).write_text(json.dumps({"status": "completed"}))
        done("completed", "", None)
        return True
    return start


class SelectedSetupTests(unittest.TestCase):
    def test_d6_gui_doctor_needs_the_native_tensor_fit_not_the_dmri_module(self):
        spec = DependencySpec(key='python:slicer.util', kind='python', label='slicer.util', required=True,
                              targets=('microstructure',))
        checked = SimpleNamespace(key=spec.key, kind='python', label=spec.label, status='available', required=True,
                                  targets=spec.targets, details={}, install_group='slicer-runtime', install_hint='')
        from shared.teem_dti import TeemError
        config = EnvironmentDoctorConfig(targets=('microstructure',), repo_root=Path('.'), loaded_slicer_modules=())
        with patch('shared.environment.build_dependency_specs', return_value=(spec,)), \
                patch('shared.environment._check_spec', return_value=checked):
            with patch('shared.teem_dti.load_teem', side_effect=TeemError('no Teem')):
                report = run_environment_doctor(config)
            self.assertEqual(report.blocker_count, 1)
            self.assertEqual(report.statuses[-1].key, 'native_tensor_fit')
            library = SimpleNamespace(_slicerdspace_path='/slicer/libteem.so')
            with patch('shared.teem_dti.load_teem', return_value=library), \
                    patch('shared.teem_dti.teem_version', return_value='1.12.0'):
                report = run_environment_doctor(config)
            self.assertEqual(report.blocker_count, 0)
    def test_slicer_510_installed_is_not_loaded_and_nonexposed_api_is_not_called(self):
        manager = Mock(spec=('isExtensionInstalled', 'isExtensionEnabled'))
        manager.isExtensionInstalled.side_effect = lambda name: name in ('SlicerDMRI', 'PyTorch')
        manager.isExtensionEnabled.return_value = True
        with patch('importlib.util.find_spec', return_value=None):
            installed, loaded = extension_states(manager, SimpleNamespace())
        self.assertEqual(installed, ['SlicerDMRI', 'PyTorch'])
        self.assertEqual(loaded, [])
        plan = build_setup_plan(result(), need_slicerdmri=True, installed_extensions=installed, loaded_extensions=loaded)
        self.assertEqual(plan['restart_extensions'], ['SlicerDMRI'])
        with patch('importlib.util.find_spec', return_value=object()):
            _, loaded = extension_states(manager, SimpleNamespace(dwitodtiestimation=object()))
        self.assertEqual(loaded, ['SlicerDMRI', 'PyTorch'])

    def test_d8_slicerdmri_is_needed_only_for_wma(self):
        missing = result()
        self.assertEqual(build_setup_plan(missing, need_dti=True)["extensions"], [])
        self.assertEqual(build_setup_plan(missing, need_dti=True, need_slicerdmri=True)["extensions"], ["SlicerDMRI"])
        self.assertEqual(build_setup_plan(missing, need_ukf=True)["extensions"], ["UKFTractography"])
        both = build_setup_plan(missing, need_slicerdmri=True, need_torch=True, need_ukf=True)
        self.assertEqual(both["extensions"], ["SlicerDMRI", "PyTorch", "UKFTractography"])
        installed = build_setup_plan(missing, need_dti=True, installed_extensions=("SlicerDMRI",), loaded_extensions=[])
        self.assertEqual((installed["extensions"], installed["restart_extensions"]), ([], []))

    def test_empty_or_ready_routes_do_not_install_unselected_backends(self):
        plan = build_setup_plan(result(), installed_extensions=("PyTorch",), need_dti=True, need_torch=True,
                                system="Linux", machine="x86_64")
        for key in ("extensions", "python_packages", "asset_profiles", "managed_profiles", "manual_items"):
            self.assertEqual(plan[key], [], key)

    def test_only_missing_required_selected_items_are_included(self):
        plan = build_setup_plan(result(item("mrconvert", "mrtrix", "command", package=None),
                                       item("eddy", "fsl", "command", package=None),
                                       item("brain_mask_backend", "brain_mask", "setting", package=None),
                                       item("weights", "model-resources", "path", ("segmentation",), None),
                                       item("unused", "wma", "command", package=None, required=False)),
                                gpu_available=True, need_dti=True, need_torch=True,
                                system="Linux", machine="x86_64")
        self.assertEqual(plan["managed_profiles"], ["mrtrix", "fsl_cuda", "pnl_cnn"])
        self.assertEqual(plan["asset_profiles"], ["segmentation"])
        self.assertEqual(plan["extensions"], ["PyTorch"])
        self.assertTrue(plan["fsl_license_required"] and plan["pnl_terms_required"])
        self.assertEqual(plan["manual_items"], [])

    def test_ukf_is_optional_and_platform_limits_remain_visible(self):
        missing = result(item("brain_mask_backend", "brain_mask", "setting", package=None))
        plan = build_setup_plan(missing, need_ukf=True, system="Windows", machine="AMD64")
        self.assertEqual(plan["extensions"], ["UKFTractography"])  # D8: UKF alone needs no SlicerDMRI
        self.assertEqual(plan["managed_profiles"], [])
        self.assertEqual(plan["unsupported_profiles"], ["pnl_cnn"])
        self.assertEqual(plan["manual_items"][0]["key"], "brain_mask_backend")

    def test_existing_managed_tools_are_verified_even_without_missing_items(self):
        plan = build_setup_plan(result(), existing_managed_profiles=("mrtrix",), repair=True,
                                system="Linux", machine="x86_64")
        self.assertEqual(plan["managed_profiles"], ["mrtrix"])
        self.assertTrue(plan["repair"])

    def owner(self, report, closure=None):
        owner = Mock()
        # K1: the widget hands out the one closure the doctor and the run use.
        owner._selected_closure.return_value = closure or expand_closure(report.targets)
        owner._setupJob = owner._pipelineActive = owner._selectedSetupActive = owner._setupRestartRequired = False
        owner._parameterNode = SimpleNamespace(brainMaskBackend="provided")
        owner._onboardingLanguage = "en"
        owner._managedRuntimeSetupButton = None
        owner._uiLanguageBindings = []
        owner._environment_dependency_path_kwargs.return_value = {}
        owner._setup_python_command.return_value = ('python', [])
        owner._run_environment_dependency_check.return_value = report
        owner._tractography_backend_text.return_value = "csd_act"
        return owner

    def run_preparation(self, owner, report, *, manager=None, confirm=True):
        manager = manager or Mock()
        manager.isExtensionInstalled.return_value = True
        slicer = SimpleNamespace(app=SimpleNamespace(extensionsManagerModel=lambda: manager),
                                 modules=SimpleNamespace(), util=SimpleNamespace(infoDisplay=Mock()))
        return patch.dict("sys.modules", {"slicer": slicer}), manager

    def test_d8_widget_plan_asks_for_slicerdmri_only_with_the_wma_target(self):
        import shared.ui_selected_setup as ui
        for ticked, doctor_targets, expected in (
                (("microstructure", "tractography", "surface"), ("microstructure", "tractography", "surface"), []),
                # C2: WMA needs a UKF tractogram, so its closure also needs the UKF extension.
                # K1: the doctor result holds the real closed shape (validation, tractography, wma).
                (("wma",), ("validation", "tractography", "wma"), ["SlicerDMRI", "UKFTractography"])):
            with self.subTest(targets=ticked):
                owner = self.owner(result(targets=doctor_targets), closure=expand_closure(ticked))
                context, manager = self.run_preparation(owner, owner._run_environment_dependency_check.return_value)
                manager.isExtensionInstalled.return_value = False
                with context:
                    plan = ui.selected_setup_plan(owner, owner._run_environment_dependency_check.return_value)
                self.assertEqual([name for name in plan["extensions"] if name != "PyTorch"], expected)

    def test_cancelled_confirmation_has_no_install_or_settings_side_effect(self):
        owner = self.owner(result(item("missing")))
        context, manager = self.run_preparation(owner, owner._run_environment_dependency_check.return_value)
        with context, patch("shared.ui_selected_setup._confirm", return_value=False):
            self.assertFalse(prepare_selected_environment(owner))
        manager.installExtensionFromServer.assert_not_called()
        owner._save_environment_path_setting.assert_not_called()

    def test_python_models_and_runtime_phases_finish_with_a_fresh_check(self):
        # B7: the GUI runs the steps from setup_steps, one async job each, then the fresh check.
        initial = result(item("numpy"), item("weights", "model-resources", "path", ("segmentation",), None))
        owner = self.owner(result())
        started = []
        context, _ = self.run_preparation(owner, initial)
        with tempfile.TemporaryDirectory() as temp, context, patch("shared.ui_selected_setup._confirm", return_value=True), \
                patch("shared.ui_environment_check.start_environment_check", side_effect=lambda widget, callback, **kwargs: callback(owner._run_environment_dependency_check()) or True), \
                patch("shared.ui_setup_jobs.start_setup_job", side_effect=fake_jobs(started)), \
                patch("scripts.setup_python_packages.dcm2niix_pending", return_value=False), \
                patch("shared.user_config.default_user_config_path", return_value=Path(temp) / "config.json"), \
                patch.dict("os.environ", {"SLICERDSPACE_USER_CONFIG": str(Path(temp) / "config.json")}):
            owner._environment_dependency_path_kwargs.return_value = {"model_assets_dir": Path(temp)}
            self.assertTrue(prepare_selected_environment(owner, initial))
            receipt = next(Path(temp).glob("setup-history/*/result.json"))
            self.assertEqual(json.loads(receipt.read_text())["status"], "ready")
        self.assertEqual([title for title, _ in started], ["Install Python packages", "Download and verify model assets"])
        self.assertFalse(owner._selectedSetupActive)
        owner._run_environment_dependency_check.assert_called_once()

    def test_ready_environment_skips_confirmation_installation_and_redundant_check(self):
        report = result()
        owner = self.owner(report)
        context, manager = self.run_preparation(owner, report)
        with tempfile.TemporaryDirectory() as temp, context, patch("shared.ui_selected_setup._confirm") as confirm, \
                patch("shared.user_config.default_user_config_path", return_value=Path(temp) / "config.json"):
            self.assertTrue(prepare_selected_environment(owner, report))
            receipt = next(Path(temp).glob("setup-history/*/result.json"))
            self.assertEqual(json.loads(receipt.read_text())["status"], "ready")
            confirm.assert_not_called()
        manager.installExtensionFromServer.assert_not_called()
        owner._run_environment_dependency_check.assert_not_called()

    def test_importable_environment_must_pass_functional_checks_before_ready(self):
        report = result()
        owner = self.owner(report)
        context, _ = self.run_preparation(owner, report)
        with tempfile.TemporaryDirectory() as temp, context, \
                patch('shared.user_config.default_user_config_path', return_value=Path(temp) / 'config.json'), \
                patch('shared.setup_selftest.requests_for_report', return_value=[{'python': 'python'}]), \
                patch('shared.ui_setup_jobs.start_setup_job', side_effect=lambda widget, title, commands, done:
                      done('failed', 'Synthetic model test failed', None)):
            prepare_selected_environment(owner, report)
            receipt = next(Path(temp).glob('setup-history/*/result.json'))
            self.assertEqual(json.loads(receipt.read_text())['status'], 'failed')
        self.assertTrue(owner._setupFunctionalFailure)
        self.assertFalse(owner._selectedSetupActive)

    def test_completed_restart_preparation_clears_previous_pending_request(self):
        from shared.setup_resume import remember_setup, pending_setup
        report = result()
        owner = self.owner(report)
        context, manager = self.run_preparation(owner, report)
        with tempfile.TemporaryDirectory() as temp, context, \
                patch('shared.user_config.default_user_config_path', return_value=Path(temp) / 'config.json'):
            remember_setup('before-restart')
            self.assertTrue(prepare_selected_environment(owner, report))
            self.assertIsNone(pending_setup())
        manager.installExtensionFromServer.assert_not_called()

    def test_failure_stops_the_chain_and_releases_preparation_lock(self):
        initial = result(item("numpy"), item("weights", "model-resources", "path", ("segmentation",), None))
        owner = self.owner(initial)
        started = []

        def failing(widget, title, commands, done):
            started.append(title)
            done("failed", "pip failed", None)
            return True

        context, _ = self.run_preparation(owner, initial)
        with tempfile.TemporaryDirectory() as temp, context, patch("shared.ui_selected_setup._confirm", return_value=True), \
                patch("shared.ui_setup_jobs.start_setup_job", side_effect=failing), \
                patch("scripts.setup_python_packages.dcm2niix_pending", return_value=False), \
                patch("shared.user_config.default_user_config_path", return_value=Path(temp) / "config.json"):
            prepare_selected_environment(owner, initial)
        self.assertFalse(owner._selectedSetupActive)
        self.assertEqual(started, ["Install Python packages"])  # the model-asset step never started
        self.assertEqual(owner._setStatus.call_args.args[0], "warning")

    def test_extension_install_requires_restart_not_false_readiness(self):
        owner = self.owner(result(targets=("wma",)))
        context, manager = self.run_preparation(owner, owner._run_environment_dependency_check.return_value)
        manager.isExtensionInstalled.return_value = False
        manager.installExtensionFromServer.return_value = True
        with tempfile.TemporaryDirectory() as temp, context, patch("shared.ui_selected_setup._confirm", return_value=True), \
                patch("shared.user_config.default_user_config_path", return_value=Path(temp) / "config.json"):
            prepare_selected_environment(owner)
        self.assertEqual([call.args for call in manager.installExtensionFromServer.call_args_list],
                         [("SlicerDMRI", False), ("UKFTractography", False)])
        self.assertTrue(owner._setupRestartRequired)
        self.assertFalse(owner._selectedSetupActive)

    # ---- fix round 1: Windows GUI route (mocks only; Windows stays "implemented, not run") ----
    WSL_UBUNTU = [{"name": "Ubuntu", "state": "Running", "version": 2, "default": True}]

    def windows_plan(self, report, **extra):
        plan = build_setup_plan(report, system="Windows", machine="AMD64", closure_targets=("validation",),
                                wsl_distributions=self.WSL_UBUNTU, slicer_executable="C:/Slicer/SlicerApp.exe")
        plan.update(also_runs="", closure_targets=["validation"], **extra)
        return plan

    def write_ps1_config(self, root, extra=None):
        """What install_deps_windows.ps1 leaves in the user config: bridge folders and the other paths."""
        written = {}
        for name, key in (("mrtrix", "mrtrix_bin_dir"), ("mrtrix2", "preprocessing_mrtrix_bin_dir"),
                          ("fsl", "preprocessing_fsl_bin_dir")):
            folder = Path(root) / name
            folder.mkdir(exist_ok=True)
            exe = folder / "tool.exe"
            Path(str(exe) + ".slicerdsurfer-owner").write_text("owner")
            Path(str(exe) + ".conf").write_text("distribution=Ubuntu\ntarget=/usr/bin/env\n")
            written[key] = str(folder)
        written.update(extra or {})
        (Path(root) / "config.json").write_text(json.dumps(
            {"kind": "slicerdsurfer_user_config", "version": 1, "external_dependencies": written}))
        return written

    def run_windows_preparation(self, root, *, started, extra_written=None, owner=None, plan_extra=None):
        from shared.setup_components import plan_components, required_licences
        pre = result(item("mrtrix", "mrtrix", "path", package=None))
        owner = owner or self.owner(result())
        plan = self.windows_plan(pre, **(plan_extra or {}))
        accepted = {key: {"id": key, "version": record["version"], "time": "t"}
                    for key, record in required_licences(plan_components(plan)).items()}

        def jobs(widget, title, commands, done):
            started.append(title)
            if title == "Install the WSL runtime":
                self.write_ps1_config(root, extra_written)
            done("completed", "", None)
            return True

        context, _ = self.run_preparation(owner, pre)
        environment_check = Mock(side_effect=lambda widget, callback, **kwargs:
                                 callback(owner._run_environment_dependency_check()) or True)
        confirm = Mock(return_value=True)
        with context, patch("shared.ui_selected_setup.selected_setup_plan", return_value=plan), \
                patch("shared.ui_selected_setup._confirm", confirm), \
                patch("shared.setup_resume.accepted_licences", return_value=accepted), \
                patch("shared.ui_environment_check.start_environment_check", environment_check), \
                patch("shared.ui_setup_jobs.start_setup_job", side_effect=jobs), \
                patch("scripts.setup_python_packages.dcm2niix_pending", return_value=False), \
                patch("shared.user_config.default_user_config_path", return_value=Path(root) / "config.json"), \
                patch.dict("os.environ", {"SLICERDSPACE_USER_CONFIG": str(Path(root) / "config.json")}):
            ok = prepare_selected_environment(owner, pre)
        return ok, owner, confirm, environment_check

    def test_s_b1_only_the_windows_step_pending_still_asks_and_runs_a_fresh_final_check(self):
        with tempfile.TemporaryDirectory() as root:
            started = []
            ok, owner, confirm, environment_check = self.run_windows_preparation(root, started=started)
            self.assertTrue(ok)
            confirm.assert_called_once()  # fails before S-B1: the dialog was skipped and setup_steps raised PermissionError
            self.assertEqual(started, ["Install the WSL runtime"])
            environment_check.assert_called_once()  # a fresh doctor report, not the pre-install one
            saved = json.loads((Path(root) / "config.json").read_text())
            self.assertEqual(saved["wsl_distribution"], "Ubuntu")


if __name__ == "__main__":
    unittest.main()
