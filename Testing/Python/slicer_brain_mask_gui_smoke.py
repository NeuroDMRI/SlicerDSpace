"""Real Slicer selector/persistence smoke; run via run_brain_mask_gui_smoke.py."""

from __future__ import annotations

import json
import os
from pathlib import Path
import traceback

import qt
import slicer


ROOT = Path(os.environ["SLICERDSPACE_SMOKE_ROOT"]).resolve()
PHASE = os.environ.get("SLICERDSPACE_BRAIN_MASK_SMOKE_PHASE", "select")
BACKENDS = ("pnl_cnn", "provided")
CHECKS = []


def require(condition, message):
    if not condition:
        raise AssertionError(message)
    CHECKS.append(message)


def isolated_settings():
    config = Path(os.environ["SLICERDSPACE_USER_CONFIG"]).resolve()
    private_home = Path(os.environ["SLICERDSPACE_PRIVATE_SLICER_HOME"]).resolve()
    require(config.is_relative_to(ROOT), "Private shared user JSON")
    for name, settings, parent in (
        ("user", slicer.app.userSettings(), ROOT / "settings"),
        ("revision", slicer.app.revisionUserSettings(), private_home / "slicer.org"),
    ):
        settings.sync()
        require(Path(settings.fileName()).resolve().is_relative_to(parent), f"Private {name} settings")
        require(int(settings.status()) == 0, f"Writable {name} settings")


def verify_saved(owner, backend):
    from shared.user_config import load_user_config, user_config_cli_values

    saved = load_user_config()
    require(saved["run_defaults"]["brain_mask_backend"] == backend, f"JSON saved {backend}")
    require(user_config_cli_values(saved)["brain_mask_backend"] == backend, f"CLI mapping saved {backend}")
    require(saved["run_defaults"]["brain_mask_device"] == owner._parameterNode.brainMaskDevice, "Device selection saved in JSON")
    require(user_config_cli_values(saved)["brain_mask_device"] == owner._parameterNode.brainMaskDevice, "Device reaches CLI config")
    settings = slicer.app.userSettings()
    settings.sync()
    require(str(settings.value(owner._user_settings_key("brainMaskBackend"))) == backend,
            f"Slicer user settings saved {backend}")


def verify_gui(owner):
    ui = owner.ui
    combo = ui.brainMaskBackendComboBox
    advanced = owner._advancedSettingsPanel
    require(advanced.collapsed, "Advanced initially collapsed")
    require(ui.parametersCollapsibleButton.collapsed, "Analysis parameters initially collapsed")
    require(advanced.isAncestorOf(combo), "Selector under Advanced")
    require(owner._advancedCategoryStack.widget(0).isAncestorOf(combo), "Selector under preprocessing category")
    for section in (ui.configurationCollapsibleButton, ui.targetsCollapsibleButton):
        require(not section.isAncestorOf(combo), "Selector absent from " + section.objectName)
    require(not combo.isVisible(), "Selector hidden on main UI")
    setup = owner._managedRuntimeSetupButton
    require(ui.environmentCollapsibleButton.isAncestorOf(setup), "Managed setup under independent environment section")
    require(not advanced.isAncestorOf(setup), "Managed setup separate from Advanced")
    require(not setup.isVisible(), "Managed setup absent from compact main UI")
    require(tuple(str(combo.itemData(index)) for index in range(combo.count)) == BACKENDS,
            "Four canonical item-data values")
    require(combo.itemText(combo.findData("pnl_cnn")) == "PNL CNN (dMRI)", "PNL CNN default display")
    for name in ("pnl_mask_python", "pnl_mask_script", "pnl_mask_model_dir", "pnl_mask_ants_bin_dir"):
        row = owner._environmentPathLineEdits[name]
        require(owner._environmentAdvancedPathsWidget.isAncestorOf(row), name + " under external paths")
        require(ui.environmentCollapsibleButton.isAncestorOf(row), name + " under environment section")
        require(not row.isVisible(), name + " hidden on main UI")
    advanced.collapsed = False
    owner._advancedCategorySelector.setCurrentIndex(0)
    slicer.app.processEvents()
    slicer.app.processEvents()
    require(combo.isVisible(), "Selector visible when Analysis parameters is expanded")
    return combo


def verify_languages(owner):
    from shared.ui_onboarding import apply_onboarding_language, onboarding_text

    combo = owner.ui.brainMaskBackendComboBox
    index = combo.currentIndex
    for language in ("zh", "en"):
        apply_onboarding_language(owner, language)
        require(combo.currentIndex == index, language + " preserves index")
        require(str(combo.itemData(index)) == "provided", language + " preserves canonical value")
        require(combo.currentText == onboarding_text(language, "brain_mask_provided"), language + " display")
        require(owner._parameterNode.brainMaskBackend == "provided", language + " preserves node")
        slicer.app.processEvents()
        owner._uiWidget.grab().save(str(ROOT / f"brain_mask_{PHASE}_{language}.png"))


def finish_select():
    try:
        owner = slicer.modules.slicerdspace.widgetRepresentation().self()
        verify_saved(owner, "provided")
        from shared.user_config import load_user_config
        saved = load_user_config()["run_defaults"]
        require(saved["microstructure_stages"] == ["dti", "dki"], "Independent DKI saved")
        require(saved["dki_fit_method"] == "MKCURVE", "Python MK-Curve method saved")
        require(not owner._pipelineActive, "No pipeline started")
        verify_languages(owner)
        if os.environ.get("SLICERDSPACE_RUNTIME_ROOT"):
            verify_managed_runtime_setup(owner)
        else:
            finish()
    except Exception:
        fail()


def verify_managed_runtime_setup(owner):
    from shared.ui_managed_runtimes import install_selected_runtimes

    def fsl_decline():
        try:
            dialog = slicer.util.findChild(slicer.util.mainWindow(), "managedRuntimeConfirmationDialog")
            accepted = slicer.util.findChild(dialog, "managedRuntimeFslAcceptance")
            buttons = slicer.util.findChild(dialog, "managedRuntimeDialogButtons")
            require(not buttons.button(qt.QDialogButtonBox.Ok).enabled, "FSL install disabled without acceptance")
            accepted.checked = True
            require(buttons.button(qt.QDialogButtonBox.Ok).enabled, "Explicit FSL license acceptance enables install")
            dialog.reject()
        except Exception:
            fail()

    qt.QTimer.singleShot(300, fsl_decline)
    require(install_selected_runtimes(owner, ("fsl_cuda",)) is False, "Declined FSL setup launches no worker")
    require(not owner._pipelineActive, "License decline changes no pipeline state")

    def mrtrix_accept():
        try:
            dialogs = [widget for widget in qt.QApplication.topLevelWidgets()
                       if widget.objectName == "managedRuntimeConfirmationDialog" and widget.isVisible()]
            require(len(dialogs) == 1, "One active managed setup confirmation")
            dialogs[0].accept()
        except Exception:
            fail()

    qt.QTimer.singleShot(300, mrtrix_accept)
    require(install_selected_runtimes(owner, ("mrtrix",)) is True, "Real background runtime setup starts")
    require(owner._pipelineActive and owner._setupJob is not None, "Setup worker owned and cancellable")
    state = {"ticks": 0}

    def wait():
        try:
            state["ticks"] += 1
            if owner._setupJob is not None:
                require(state["ticks"] < 100, "Setup completes within smoke timeout")
                qt.QTimer.singleShot(200, wait)
                return
            from shared.user_config import load_user_config
            saved = load_user_config()["external_dependencies"]
            expected = str(Path(os.environ["SLICERDSPACE_RUNTIME_ROOT"]) / "mrtrix-3.0.8/bin")
            require(saved.get("mrtrix_bin_dir") == expected, "Verified MRtrix path saved automatically")
            require(saved.get("preprocessing_mrtrix_bin_dir") == expected, "Preprocessing path saved automatically")
            require(owner._environmentPathLineEdits["mrtrix_bin_dir"].text == expected, "GUI path populated automatically")
            require(not owner._pipelineActive, "Background setup finished without scientific inference")
            finish()
        except Exception:
            fail()

    qt.QTimer.singleShot(200, wait)


def check():
    try:
        isolated_settings()
        slicer.util.selectModule("SlicerDSpace")
        owner = slicer.modules.slicerdspace.widgetRepresentation().self()
        from shared.user_config import load_user_config

        node = owner._parameterNode
        combo = verify_gui(owner)
        if PHASE == "restore":
            require(node.brainMaskBackend == "provided", "New process restores node from saved config")
            require(node.brainMaskDevice == "cpu", "New process restores explicit CPU device")
            require(owner.ui.brainMaskDeviceComboBox.currentText == "cpu", "Device selector restored")
            require(node.device == "cpu" and node.ddsegDevice == "cpu", "Explicit unified PyTorch CPU choice restored")
            require(str(combo.itemData(combo.currentIndex)) == "provided", "New process restores selector")
            verify_saved(owner, "provided")
            require(node.orchestrationMicrostructureStages == "dti,dki", "New process restores DKI only")
            require(owner.ui.orchestrationDkiCheckBox.isChecked(), "DKI checkbox restored")
            require(not owner.ui.orchestrationFreeWaterCheckBox.isChecked(), "FW remains off after restore")
            require(not owner.ui.orchestrationNoddiCheckBox.isChecked(), "NODDI remains off after restore")
            require(node.orchestrationDkiFitMethod == "MKCURVE", "Python MK-Curve method restored")
            external = load_user_config()["external_dependencies"]
            for key in ("pnl_mask_python", "pnl_mask_script", "pnl_mask_model_dir", "pnl_mask_ants_bin_dir", "dki_matlab_executable", "dki_mkcurve_root"):
                require(owner._environmentPathLineEdits[key].text == external[key], key + " restored UI")
            verify_languages(owner)
            finish()
            return
        require(PHASE == "select", "Known smoke phase")
        require(node.brainMaskBackend == "pnl_cnn", "Fresh node defaults to pnl_cnn")
        require(node.brainMaskDevice == "auto", "Fresh node prefers GPU automatically")
        require(node.device == "auto" and node.ddsegDevice == "auto", "Fresh PyTorch defaults prefer GPU")
        owner.ui.unifiedTorchDeviceComboBox.setCurrentIndex(owner.ui.unifiedTorchDeviceComboBox.findText("cpu"))
        require(node.device == "cpu" and node.ddsegDevice == "cpu", "Unified CPU choice reaches both nodes")
        device_combo = owner.ui.brainMaskDeviceComboBox
        require(owner._advancedSettingsPanel.isAncestorOf(device_combo), "Mask device under Advanced")
        for device in ("auto", "cuda", "cpu"):
            device_combo.setCurrentIndex(device_combo.findText(device))
            require(node.brainMaskDevice == device, device + " device reaches MRML")
        require(combo.currentText == "PNL CNN (dMRI)", "Fresh selector defaults to PNL CNN (dMRI)")
        require(load_user_config()["run_defaults"]["brain_mask_backend"] == "pnl_cnn", "Fresh JSON defaults to pnl_cnn")
        require(node.orchestrationMicrostructureStages == "dti", "Fresh node defaults to DTI only")
        require(not owner.ui.orchestrationMicrostructureStagesComboBox.isVisible(), "Canonical legacy selector hidden")
        for name in ("orchestrationFreeWaterCheckBox", "orchestrationNoddiCheckBox", "orchestrationDkiCheckBox"):
            checkbox = getattr(owner.ui, name)
            require(not checkbox.isChecked(), name + " defaults off")
            require(owner.ui.parametersCollapsibleButton.isAncestorOf(checkbox), name + " in Advanced analysis")
        owner.ui.orchestrationDkiCheckBox.setChecked(True)
        require(node.orchestrationMicrostructureStages == "dti,dki", "DKI independent of FW and NODDI")
        require("microstructure_dki" in owner._selected_product_targets(), "DKI reaches product target expansion")
        owner.ui.orchestrationFreeWaterCheckBox.setChecked(True)
        require(node.orchestrationMicrostructureStages == "dti,freewater,dki", "FW plus DKI without NODDI")
        owner.ui.orchestrationFreeWaterCheckBox.setChecked(False)
        node.orchestrationMicrostructureStages = "dti,noddi"
        require(owner.ui.orchestrationNoddiCheckBox.isChecked(), "Node syncs independent NODDI")
        require(not owner.ui.orchestrationDkiCheckBox.isChecked(), "Node syncs DKI off")
        node.orchestrationMicrostructureStages = "dti,dki"
        owner.ui.orchestrationDkiFitMethodComboBox.setCurrentIndex(1)
        require(node.orchestrationDkiFitMethod == "OLS", "DKI method reaches parameter node")
        method_combo = owner.ui.orchestrationDkiFitMethodComboBox
        require(tuple(method_combo.itemText(i) for i in range(method_combo.count)) ==
                ("WLS", "OLS", "MKCURVE", "MKCURVE_MATLAB"), "All DKI method choices, Python and reference distinct")
        method_combo.setCurrentIndex(method_combo.findText("MKCURVE"))
        require(node.orchestrationDkiFitMethod == "MKCURVE", "Python MK-Curve reaches parameter node")
        require("Python" in method_combo.toolTip, "MK-Curve Python help visible")
        owner._save_common_user_settings()
        doctor_kwargs = owner._environment_check_kwargs()
        require(doctor_kwargs["brain_mask_backend"] == "pnl_cnn", "Doctor sees selected PNL backend")
        require(doctor_kwargs["brain_mask_required"] == bool(node.targetPreprocessing),
                "Doctor requires generator for selected default preprocessing")
        for backend in BACKENDS:
            combo.setCurrentIndex(combo.findData(backend))
            require(node.brainMaskBackend == backend, "UI switch updates node: " + backend)
            owner._save_common_user_settings()
            verify_saved(owner, backend)
        node.brainMaskBackend = "pnl_cnn"
        require(str(combo.itemData(combo.currentIndex)) == "pnl_cnn", "Node change syncs selector by data")
        owner._save_common_user_settings()
        for key in ("pnl_mask_python", "pnl_mask_script", "pnl_mask_model_dir", "pnl_mask_ants_bin_dir", "dki_matlab_executable", "dki_mkcurve_root"):
            path = str(ROOT / "external-config-only" / key)
            row = owner._environmentPathLineEdits[key]
            row.setText(path)
            owner._on_environment_dependency_path_edited(key, row)
            require(load_user_config()["external_dependencies"][key] == path, key + " saved externally")
        combo.setCurrentIndex(combo.findData("provided"))
        require(node.brainMaskBackend == "provided", "Final UI switch selects provided")
        require(not owner._environment_check_kwargs()["brain_mask_required"],
                "Provided mode with unknown inputs does not require generator dependencies")
        # Let the existing 350 ms debounce save the last switch without calling Save.
        qt.QTimer.singleShot(1000, finish_select)
    except Exception:
        fail()


def finish():
    record = {"status": "PASS", "phase": PHASE, "checks": CHECKS,
              "slicer_version": slicer.app.applicationVersion}
    (ROOT / f"brain_mask_{PHASE}.json").write_text(json.dumps(record, indent=2) + "\n")
    print("BRAIN_MASK_GUI_SMOKE_PASS " + PHASE, flush=True)
    slicer.app.exit(0)


def fail():
    details = traceback.format_exc()
    print(details, flush=True)
    (ROOT / f"brain_mask_{PHASE}.json").write_text(
        json.dumps({"status": "FAIL", "phase": PHASE, "checks": CHECKS, "error": details}, indent=2) + "\n")
    slicer.app.exit(2)


qt.QTimer.singleShot(5000, check)
qt.QTimer.singleShot(60000, lambda: slicer.app.exit(3))
