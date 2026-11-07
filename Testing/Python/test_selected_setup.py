"""Selected-route planning, phase orchestration and honest readiness states."""

from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import Mock, patch

from shared.selected_setup import build_setup_plan, need_flags
from shared.setup_components import LicenceNotAccepted
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


def accepting_confirm(owner, plan):
    """Stand-in for the dialog: the user presses "Accept licences and install" (S-B8: the steps need the acceptance)."""
    from shared.setup_components import accept_licences, plan_components, required_licences
    accept_licences(required_licences(plan_components(plan)))
    return True


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
    def setUp(self):
        # S-B3: a missing dcm2niix wheel is pending work; these tests are about other things, so the wheel is "installed".
        patcher = patch("scripts.setup_python_packages.dcm2niix_pending", return_value=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        # G1-T: requests_for_report asks DICOM import for dcm2niix; keep this test independent of the machine's PATH and wheel.
        patcher = patch("shared.input_import.find_dcm2niix", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)

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
        manager.isExtensionInstalled.side_effect = lambda name: name == 'PyTorch'
        manager.isExtensionEnabled.return_value = True
        with patch('importlib.util.find_spec', return_value=None):
            installed, loaded = extension_states(manager, SimpleNamespace())
        self.assertEqual(installed, ['PyTorch'])
        self.assertEqual(loaded, [])
        plan = build_setup_plan(result(), need_slicerdmri=True, installed_extensions=installed, loaded_extensions=loaded)
        self.assertEqual(plan['restart_extensions'], [])
        with patch('importlib.util.find_spec', return_value=object()):
            _, loaded = extension_states(manager, SimpleNamespace(dwitodtiestimation=object()))
        self.assertEqual(loaded, ['PyTorch'])

    def test_d8_wma_does_not_require_slicerdmri(self):
        missing = result()
        self.assertEqual(build_setup_plan(missing, need_dti=True)["extensions"], [])
        self.assertEqual(build_setup_plan(missing, need_dti=True, need_slicerdmri=True)["extensions"], [])
        self.assertEqual(build_setup_plan(missing, need_ukf=True)["extensions"], ["UKFTractography"])
        both = build_setup_plan(missing, need_slicerdmri=True, need_torch=True, need_ukf=True)
        self.assertEqual(both["extensions"], ["PyTorch", "UKFTractography"])
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

    def test_sc1c_managed_profile_and_licence_only_for_the_managed_route(self):
        missing = result(item("brain_mask_backend", "brain_mask", "setting", package=None))
        cases = (("auto", "Linux", True), ("auto", "Darwin", False), ("managed", "Linux", True),
                 ("native", "Linux", False), ("native", "Darwin", False), ("auto", "Windows", False))
        for route, system, managed in cases:
            with self.subTest(route=route, system=system):
                plan = build_setup_plan(missing, system=system, machine="x86_64" if system != "Darwin" else "arm64",
                                        pnl_mask_route=route)
                self.assertEqual("pnl_cnn" in plan["managed_profiles"], managed)
                self.assertEqual(plan["pnl_terms_required"], managed)
                from shared.setup_components import plan_components, required_licences
                licences = list(required_licences(plan_components(plan)))
                # F3: with no pnl-cnn asset planned, only the managed runtime brings the PNL licence.
                self.assertEqual("pnl" in licences, managed)

    def test_sc1c_native_route_installs_the_brain_mask_assets_when_only_the_models_are_missing(self):
        models = item("brain_mask_backend", "brain_mask", "setting", package=None)
        models.details = {"backend": "pnl_cnn", "missing_required": [
            "pnl_mask_model_dir"]}
        weights = item("brain_mask_backend", "brain_mask", "setting", package=None)
        weights.details = {"backend": "pnl_cnn", "missing_required": [
            "sagittal.pt (convert/download PNLBrainMask-Torch weights; legacy HDF5 is not a Torch runtime)", "weights.json"]}
        torch_gap = item("brain_mask_backend", "brain_mask", "setting", package=None)
        torch_gap.details = {"backend": "pnl_cnn", "missing_required": [
            "pnl_mask_model_dir", "PNL native file /x/UPSTREAM.json"]}
        for label, found, assets in (("models", models, ["pnl-cnn"]), ("weights", weights, ["pnl-cnn"]),
                                     ("other gap", torch_gap, [])):
            with self.subTest(case=label):
                plan = build_setup_plan(result(found), system="Darwin", machine="arm64", pnl_mask_route="native")
                self.assertEqual(plan["optional_asset_profiles"], assets)
                self.assertEqual(plan["managed_profiles"], [])
                self.assertEqual([entry["key"] for entry in plan["manual_items"]],
                                 [] if assets else ["brain_mask_backend"])
        # The managed route never asks for the optional pnl-cnn resources through this path.
        plan = build_setup_plan(result(models), system="Linux", machine="x86_64", pnl_mask_route="managed")
        self.assertEqual(plan["optional_asset_profiles"], [])
        self.assertEqual(plan["managed_profiles"], ["pnl_cnn"])

    def test_f3_native_route_with_planned_pnl_assets_requires_the_pnl_licence_but_not_the_managed_runtime(self):
        """F3: the native assets need the PNL licence (preface and shared/pnl_cnn/LICENSE.txt); no managed runtime."""
        from shared.setup_components import plan_components, required_licences
        models = item("brain_mask_backend", "brain_mask", "setting", package=None)
        models.details = {"backend": "pnl_cnn", "missing_required": ["pnl_mask_model_dir"]}
        for system, machine in (("Darwin", "arm64"), ("Windows", "AMD64")):
            with self.subTest(system=system):
                plan = build_setup_plan(result(models), system=system, machine=machine)
                self.assertEqual(plan["optional_asset_profiles"], ["pnl-cnn"])
                self.assertEqual(plan["managed_profiles"], [])
                self.assertFalse(plan["pnl_terms_required"])
                licences = required_licences(plan_components(plan))
                self.assertIn("pnl", licences)
                self.assertIn("optional_resources", licences)
                self.assertTrue(licences["pnl"]["file"].endswith("shared/pnl_cnn/LICENSE.txt"))
        # The managed route plans no pnl-cnn resources, so its PNL licence comes from the managed profile only.
        linux = build_setup_plan(result(models), system="Linux", machine="x86_64")
        self.assertEqual(linux["optional_asset_profiles"], [])
        self.assertIn("pnl", required_licences(plan_components(linux)))

    def test_f1_native_mask_only_closure_plans_pytorch_and_the_managed_route_does_not(self):
        """F1: a required native pnl_cnn mask needs torch even when no torch target is selected."""
        mask = item("brain_mask_backend", "brain_mask", "setting", package=None, targets=("microstructure",))
        for system, machine in (("Darwin", "arm64"), ("Windows", "AMD64")):
            with self.subTest(system=system):
                plan = build_setup_plan(result(mask, targets=("microstructure",)), system=system, machine=machine,
                                        closure_targets=("microstructure",), **need_flags(("microstructure",)))
                self.assertIn("PyTorch", plan["extensions"])
                installed = build_setup_plan(result(mask, targets=("microstructure",)), system=system, machine=machine,
                                             installed_extensions=("PyTorch",), loaded_extensions=["PyTorch"])
                self.assertEqual(installed["extensions"], [])
        # Guards: the managed route (Linux auto), another backend, and a mask that is not required add no PyTorch.
        linux = build_setup_plan(result(mask, targets=("microstructure",)), system="Linux", machine="x86_64",
                                 **need_flags(("microstructure",)))
        self.assertNotIn("PyTorch", linux["extensions"])
        provided = build_setup_plan(result(mask), system="Darwin", machine="arm64", brain_mask_backend="provided")
        self.assertNotIn("PyTorch", provided["extensions"])
        optional = item("brain_mask_backend", "brain_mask", "setting", package=None, required=False)
        self.assertNotIn("PyTorch", build_setup_plan(result(optional), system="Darwin", machine="arm64")["extensions"])

    def _w2a_missing_preprocessing_items(self):
        """The doctor's own list for the preprocessing target, every MRtrix3 and FSL item missing."""
        from shared.environment import build_dependency_specs

        repo = Path(__file__).resolve().parents[2]
        specs = build_dependency_specs(EnvironmentDoctorConfig(targets=("preprocessing",), repo_root=repo))
        items = [SimpleNamespace(key=spec.key, label=spec.label, install_group=spec.install_group, kind=spec.kind,
                                 targets=spec.targets, required=spec.required, status="missing",
                                 install_hint="Review " + spec.key, details={})
                 for spec in specs if spec.install_group in {"mrtrix", "fsl"}]
        self.assertEqual({item.key for item in items},
                         {"command:dwifslpreproc", "command:dwi2mask", "command:eddy", "command:topup"})
        return items

    def test_w2a_setup_plan_for_preprocessing_holds_mrtrix3_fsl_and_the_fsl_licence_on_every_os(self):
        """W2a (D1): preprocessing needs MRtrix3, FSL (eddy, topup) and the FSL licence on Linux, macOS and Windows."""
        from shared.setup_components import plan_components, required_licences

        ubuntu = [{"name": "Ubuntu", "state": "Running", "version": 2, "default": True}]
        items = self._w2a_missing_preprocessing_items()
        with patch("platform.mac_ver", return_value=("15.1", (), "")):
            for system, machine in (("Linux", "x86_64"), ("Darwin", "arm64"), ("Darwin", "x86_64"), ("Windows", "AMD64")):
                with self.subTest(system=system, machine=machine):
                    plan = build_setup_plan(result(*items, targets=("preprocessing",)), system=system, machine=machine,
                                            closure_targets=("preprocessing",), wsl_distributions=ubuntu,
                                            slicer_executable="C:/Slicer/Slicer.exe")
                    self.assertEqual(plan["unsupported_profiles"], [])
                    self.assertEqual(plan["manual_items"], [])
                    self.assertTrue(plan["fsl_license_required"])
                    licences = required_licences(plan_components(plan))
                    self.assertIn("fsl", licences)
                    self.assertIn("mrtrix", licences)
                    if system == "Windows":
                        self.assertEqual(plan["managed_profiles"], [])
                        self.assertIn("-AcceptFslLicense", plan["windows_runtime"]["command"])
                    else:
                        self.assertEqual(plan["managed_profiles"], ["mrtrix", "fsl_cpu"])

    def test_w2a_a_platform_without_an_fsl_build_does_not_offer_preprocessing(self):
        from shared.selected_setup import PREPROCESSING_UNSUPPORTED_TEXT, preprocessing_offered

        items = self._w2a_missing_preprocessing_items()
        with patch("platform.mac_ver", return_value=("15.1", (), "")):
            plan = build_setup_plan(result(*items, targets=("preprocessing",)), system="Linux", machine="aarch64",
                                    closure_targets=("preprocessing",))
            self.assertFalse(preprocessing_offered("Linux", "aarch64"))
            for system, machine in (("Linux", "x86_64"), ("Darwin", "arm64"), ("Darwin", "x86_64"), ("Windows", "AMD64")):
                self.assertTrue(preprocessing_offered(system, machine), (system, machine))
        self.assertEqual(plan["managed_profiles"], [])
        self.assertFalse(plan["fsl_license_required"])
        self.assertIn(PREPROCESSING_UNSUPPORTED_TEXT, plan["unsupported_notes"].values())
        self.assertEqual(PREPROCESSING_UNSUPPORTED_TEXT, "Preprocessing needs FSL, which has no build for this platform; "
                                                         "import preprocessed data instead")
        self.assertTrue(plan["manual_items"])
        self.assertEqual({item["hint"] for item in plan["manual_items"]}, {PREPROCESSING_UNSUPPORTED_TEXT})

    def test_ukf_is_optional_and_platform_limits_remain_visible(self):
        missing = result(item("brain_mask_backend", "brain_mask", "setting", package=None))
        plan = build_setup_plan(missing, need_ukf=True, system="Windows", machine="AMD64", pnl_mask_route="managed")
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

    def test_d8_widget_plan_uses_ukf_without_slicerdmri_for_wma(self):
        import shared.ui_selected_setup as ui
        for ticked, doctor_targets, expected in (
                (("microstructure", "tractography", "surface"), ("microstructure", "tractography", "surface"), []),
                # C2: WMA needs a UKF tractogram, so its closure also needs the UKF extension.
                # K1: the doctor result holds the real closed shape (validation, tractography, wma).
                (("wma",), ("validation", "tractography", "wma"), ["UKFTractography"])):
            with self.subTest(targets=ticked):
                owner = self.owner(result(targets=doctor_targets), closure=expand_closure(ticked))
                context, manager = self.run_preparation(owner, owner._run_environment_dependency_check.return_value)
                manager.isExtensionInstalled.return_value = False
                with context:
                    plan = ui.selected_setup_plan(owner, owner._run_environment_dependency_check.return_value)
                self.assertEqual([name for name in plan["extensions"] if name != "PyTorch"], expected)

    def test_a6_gui_closure_equals_cli_closure_for_a_non_default_tractography_selection(self):
        """A6 (closure Opus N3): UKF selected. The GUI closure, the GUI plan and the CLI plan name the same targets.

        The no-node fallback of selected_setup_plan builds the Closure from the doctor result without a second expansion.
        """
        from scripts import setup_selected
        from shared.ui_run_controller import ProductTargetFlags, selected_closure
        from shared.user_config import empty_user_config
        import shared.ui_selected_setup as ui

        gui = selected_closure(ProductTargetFlags(tractography=True), tractography_backend="ukf", bundle_backend="wmql",
                               microstructure_freewater=False, microstructure_noddi=False)
        config = empty_user_config()
        config["run_defaults"]["tractography_backend"] = "ukf"
        with patch("scripts.setup_python_packages.dcm2niix_pending", return_value=False):
            cli, _report, cli_plan = setup_selected.build_cli_plan(
                ["validation", "tractography"], config, doctor=lambda cfg: SimpleNamespace(statuses=[], targets=cfg.targets, blocker_count=0,
                                                                              python_executable=None),
                system="Linux", machine="x86_64")
        self.assertEqual((gui.targets, gui.tractography_algorithms), (cli.targets, cli.tractography_algorithms))
        self.assertIn("ukf", gui.tractography_algorithms)
        report = result(targets=gui.targets)
        owner = self.owner(report, closure=gui)
        context, _ = self.run_preparation(owner, report)
        with context, patch("scripts.setup_python_packages.dcm2niix_pending", return_value=False):
            gui_plan = ui.selected_setup_plan(owner, report)
            owner._selected_closure.return_value = None  # no parameter node: the closed doctor targets are taken as they are
            fallback = ui.selected_setup_plan(owner, report)
        self.assertEqual(gui_plan["closure_targets"], cli_plan["closure_targets"])
        self.assertEqual(fallback["closure_targets"], list(gui.targets))

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
        with tempfile.TemporaryDirectory() as temp, context, patch("shared.ui_selected_setup._confirm", side_effect=accepting_confirm), \
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

    def test_a2_a_warning_receipt_keeps_the_gui_ready_and_shows_the_warning(self):
        """A2: a passed self-check receipt with a warning (a failing PATH dcm2niix) is ready, not failed."""
        report = result()
        owner = self.owner(report)
        context, _ = self.run_preparation(owner, report)

        def jobs(widget, title, commands, done):
            for _program, args, _env in commands:
                Path(args[args.index('--result') + 1]).write_text(json.dumps(
                    {'status': 'passed', 'warnings': ['dcm2niix at /usr/bin/dcm2niix failed its check (garbled)']}))
            done('completed', '', None)
            return True

        with tempfile.TemporaryDirectory() as temp, context, \
                patch('shared.user_config.default_user_config_path', return_value=Path(temp) / 'config.json'), \
                patch('shared.setup_selftest.requests_for_report', return_value=[{'python': 'python'}]), \
                patch('shared.ui_setup_jobs.start_setup_job', side_effect=jobs):
            prepare_selected_environment(owner, report)
            record = json.loads(next(Path(temp).glob('setup-history/*/result.json')).read_text())
        self.assertEqual(record['status'], 'ready')
        self.assertIn('/usr/bin/dcm2niix', record['warnings'][0])
        self.assertFalse(owner._setupFunctionalFailure)
        self.assertIn('/usr/bin/dcm2niix', owner._setStatus.call_args.args[1])

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
        with tempfile.TemporaryDirectory() as temp, context, patch("shared.ui_selected_setup._confirm", side_effect=accepting_confirm), \
                patch("shared.ui_setup_jobs.start_setup_job", side_effect=failing), \
                patch("scripts.setup_python_packages.dcm2niix_pending", return_value=False), \
                patch("shared.user_config.default_user_config_path", return_value=Path(temp) / "config.json"):
            prepare_selected_environment(owner, initial)
        self.assertFalse(owner._selectedSetupActive)
        self.assertEqual(started, ["Install Python packages"])  # the model-asset step never started
        self.assertEqual(owner._setStatus.call_args.args[0], "warning")

    def test_a4_2_a_licence_permission_error_reaches_the_user_with_the_licence_names(self):
        """A4 (2): the status line and the error dialog carry the licence names. Fails before: generic "stopped" text."""
        initial = result(item("numpy"))
        owner = self.owner(initial)
        context, _ = self.run_preparation(owner, initial)
        with tempfile.TemporaryDirectory() as temp, context, patch("shared.ui_selected_setup._confirm", side_effect=accepting_confirm), \
                patch("shared.selected_setup.setup_steps", side_effect=LicenceNotAccepted("No licence is listed for: fsl_terms")), \
                patch("shared.user_config.default_user_config_path", return_value=Path(temp) / "config.json"):
            self.assertFalse(prepare_selected_environment(owner, initial))
        self.assertIn("fsl_terms", owner._setStatus.call_args.args[1])
        self.assertIn("fsl_terms", owner._show_actionable_error.call_args.kwargs["why"])
        self.assertFalse(owner._selectedSetupActive)

    def test_ob3_an_os_permission_error_does_not_get_the_licence_title(self):
        """OB3: only LicenceNotAccepted gets the licence dialog. Fails at 0064d04: any PermissionError did."""
        initial = result(item("numpy"))
        owner = self.owner(initial)
        context, _ = self.run_preparation(owner, initial)
        denied = PermissionError(13, "Permission denied", "/read-only/setup-history/log.txt")
        with tempfile.TemporaryDirectory() as temp, context, patch("shared.ui_selected_setup._confirm", side_effect=accepting_confirm), \
                patch("shared.selected_setup.setup_steps", side_effect=denied), \
                patch("shared.user_config.default_user_config_path", return_value=Path(temp) / "config.json"):
            self.assertFalse(prepare_selected_environment(owner, initial))
            details = [json.loads(path.read_text())["details"] for path in Path(temp).glob("setup-history/*/result.json")]
        owner._show_actionable_error.assert_not_called()
        self.assertEqual(len(details), 1)
        self.assertIn("Permission denied", details[0])  # the generic path keeps the real text
        self.assertFalse(owner._selectedSetupActive)

    def test_extension_install_requires_restart_not_false_readiness(self):
        owner = self.owner(result(targets=("wma",)))
        context, manager = self.run_preparation(owner, owner._run_environment_dependency_check.return_value)
        manager.isExtensionInstalled.return_value = False
        manager.installExtensionFromServer.return_value = True
        with tempfile.TemporaryDirectory() as temp, context, patch("shared.ui_selected_setup._confirm", return_value=True), \
                patch("shared.user_config.default_user_config_path", return_value=Path(temp) / "config.json"):
            prepare_selected_environment(owner)
        self.assertEqual([call.args for call in manager.installExtensionFromServer.call_args_list],
                         [("UKFTractography", False)])
        self.assertTrue(owner._setupRestartRequired)
        self.assertFalse(owner._selectedSetupActive)

    def test_s_n1_python_and_model_steps_keep_the_status_writes_and_error_dialogs_of_the_old_chain(self):
        initial = result(item("numpy"), item("weights", "model-resources", "path", ("segmentation",), None))
        for fail_title in ("Install Python packages", "Download and verify model assets", None):
            owner = self.owner(result())
            owner._environment_dependency_path_kwargs.return_value = {}
            started = []

            def jobs(widget, title, commands, done):
                started.append(title)
                if title == fail_title:
                    done("failed", "boom", Path("/log/path"))
                    return True
                return fake_jobs([])(widget, title, commands, lambda status, details, _log: done(status, details, Path("/log/path")))

            context, _ = self.run_preparation(owner, initial)
            with tempfile.TemporaryDirectory() as temp, context, patch("shared.ui_selected_setup._confirm", side_effect=accepting_confirm), \
                    patch("shared.ui_environment_check.start_environment_check", side_effect=lambda widget, callback, **kwargs: callback(owner._run_environment_dependency_check()) or True), \
                    patch("shared.ui_setup_jobs.start_setup_job", side_effect=jobs), \
                    patch("shared.user_config.default_user_config_path", return_value=Path(temp) / "config.json"), \
                    patch.dict("os.environ", {"SLICERDSPACE_USER_CONFIG": str(Path(temp) / "config.json")}):
                owner._environment_dependency_path_kwargs.return_value = {"model_assets_dir": Path(temp)}
                prepare_selected_environment(owner, initial)
            python_calls = owner._save_python_install_status.call_args_list
            self.assertEqual(len(python_calls), 1, fail_title)  # guard for S-N1: the call was lost in B7
            self.assertEqual(python_calls[0].args[1], "failed" if fail_title == "Install Python packages" else "completed")
            titles = [call.kwargs["title"] for call in owner._show_actionable_error.call_args_list]
            if fail_title == "Install Python packages":
                self.assertEqual(titles, ["Python dependency installation failed"])
                owner._set_model_assets_status_text.assert_not_called()
            elif fail_title == "Download and verify model assets":
                self.assertEqual(titles, ["Model asset setup failed"])
                self.assertIn("failed", owner._set_model_assets_status_text.call_args.args[0])
            else:
                self.assertEqual(titles, [])
                self.assertIn("checksum-verified", owner._set_model_assets_status_text.call_args.args[0])

    def test_s_b9_slicer_executable_is_the_launcher_then_the_application_path_then_the_configured_one(self):
        import shared.ui_selected_setup as ui
        report = result(item("mrtrix", "mrtrix", "path", package=None))
        app = SimpleNamespace(extensionsManagerModel=lambda: Mock(isExtensionInstalled=Mock(return_value=True)),
                              applicationFilePath=lambda: "C:/Slicer/bin/SlicerApp-real.exe")
        slicer = SimpleNamespace(app=app, modules=SimpleNamespace(), util=SimpleNamespace(infoDisplay=Mock()))

        def command_for(launcher, configured):
            owner = self.owner(report)
            owner._slicer_launcher_program.return_value = launcher
            owner._environment_dependency_path_kwargs.return_value = {"slicer_executable": configured} if configured else {}
            with patch.dict("sys.modules", {"slicer": slicer}), patch("platform.system", return_value="Windows"), \
                    patch("shared.selected_setup.list_distributions", return_value=self.WSL_UBUNTU):
                command = ui.selected_setup_plan(owner, report)["windows_runtime"]["command"]
            return command[command.index("-SlicerExecutable") + 1]

        # fails before S-B9: the configured path (or the app path SlicerApp-real.exe) was used
        self.assertEqual(command_for("C:/Slicer/Slicer.exe", "D:/old/Slicer.exe"), "C:/Slicer/Slicer.exe")
        self.assertEqual(command_for("", "D:/old/Slicer.exe"), "C:/Slicer/bin/SlicerApp-real.exe")
        slicer.app.applicationFilePath = lambda: ""
        self.assertEqual(command_for("", "D:/old/Slicer.exe"), "D:/old/Slicer.exe")

    def test_s_b3_a_missing_dcm2niix_wheel_alone_is_pending_work_in_the_gui(self):
        report = result()
        owner = self.owner(report)
        context, _ = self.run_preparation(owner, report)
        started = []
        with tempfile.TemporaryDirectory() as temp, context, patch("shared.ui_selected_setup._confirm", side_effect=accepting_confirm) as confirm, \
                patch("scripts.setup_python_packages.dcm2niix_pending", return_value=True), \
                patch("shared.ui_environment_check.start_environment_check", side_effect=lambda widget, callback, **kwargs: callback(report) or True), \
                patch("shared.ui_setup_jobs.start_setup_job", side_effect=fake_jobs(started)), \
                patch("shared.user_config.default_user_config_path", return_value=Path(temp) / "config.json"), \
                patch.dict("os.environ", {"SLICERDSPACE_USER_CONFIG": str(Path(temp) / "config.json")}):
            self.assertTrue(prepare_selected_environment(owner, report))
        confirm.assert_called_once()  # fails before S-B3: nothing else was missing, so no dialog and no install
        self.assertEqual([title for title, _ in started], ["Install Python packages"])

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

    def test_s_b2_settings_the_ps1_wrote_are_loaded_and_survive_the_next_common_save(self):
        from shared.dependency_config import UI_DEPENDENCY_PATHS
        from shared.user_config import load_user_config, update_user_config
        written = {"tract_querier_root": "\\\\wsl$\\Ubuntu\\tq", "wma_root": "\\\\wsl$\\Ubuntu\\wma",
                   "wma_atlas_dir": "\\\\wsl$\\Ubuntu\\atlas", "tractcloud_root": "\\\\wsl$\\Ubuntu\\tc",
                   "tractcloud_python_executable": "\\\\wsl$\\Ubuntu\\tc\\python", "tractcloud_data_dir": "\\\\wsl$\\Ubuntu\\tcdata",
                   "slicer_executable": "C:\\Slicer\\Slicer.exe", "python_executable": "C:\\Slicer\\bin\\PythonSlicer.exe",
                   "postprocess_freesurfer_home": "\\\\wsl$\\Ubuntu\\fs"}
        with tempfile.TemporaryDirectory() as root:
            owner, store = self.owner(result()), {}
            owner._save_environment_path_setting.side_effect = lambda key, value: store.__setitem__(key, value)
            ok, owner, _confirm, _check = self.run_windows_preparation(root, started=[], extra_written=written, owner=owner)
            self.assertTrue(ok)
            for key, value in written.items():  # fails before S-B2: only the three bridge keys were reloaded
                self.assertEqual(store.get(key), value, key)
            # What _save_common_user_settings does next: every dependency id is read back from the settings.
            with patch.dict("os.environ", {"SLICERDSPACE_USER_CONFIG": str(Path(root) / "config.json")}):
                update_user_config({"external_dependencies": {item["id"]: store.get(item["id"]) or None
                                                              for item in UI_DEPENDENCY_PATHS}})
                kept = load_user_config()["external_dependencies"]
            for key, value in written.items():
                self.assertEqual(kept.get(key), value, key)


if __name__ == "__main__":
    unittest.main()
