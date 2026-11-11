"""Native Qt/MRML checks for the live product algorithm summaries."""

from pathlib import Path
import json
import os
import traceback

import qt
import slicer


def run():
    output = Path(os.environ.get("SLICERDSPACE_UI_EVIDENCE") or os.environ["SLICERDSPACE_SMOKE_ROOT"])
    output.mkdir(parents=True, exist_ok=True)
    checks = []
    try:
        slicer.util.selectModule("SlicerDSpace")
        widget = slicer.modules.slicerdspace.widgetRepresentation().self()
        node = widget._parameterNode
        ui = widget.ui
        language = widget._onboardingWidgets["language_combo"]
        language.setCurrentIndex(language.findData("en"))
        with slicer.util.NodeModify(node):
            node.orchestrationMicrostructureStages = "dti"
            node.orchestrationRunFreeWater = False
            node.orchestrationRunNoddi = False
            node.orchestrationRunDki = False
            node.orchestrationTractographyBackend = "csd_act"
            node.orchestrationBundleBackend = "wmql"
        slicer.app.processEvents()

        def expect(control, text):
            actual = control.accessibleName if control.objectName in widget._algorithmSummaryLabels else control.text
            assert actual == text, (control.objectName, actual, text)
            if control.objectName in widget._algorithmSummaryLabels:
                assert control.text == ''
                assert widget._algorithmSummaryLabels[control.objectName].text == text.split(': ', 1)[-1].removeprefix('当前：')
            checks.append(text)

        expect(ui.targetMicrostructureCheckBox, "Selected: DTI")
        expect(ui.targetTractographyCheckBox, "Selected: MRtrix3 CSD / ACT")
        expect(ui.targetBundlesCheckBox, "Selected: WMQL")
        assert widget._advancedSettingsPanel.collapsed
        assert "Advanced settings" in widget._advancedSettingsPanel.text
        assert "Select" in widget._advancedSettingsHintLabel.text
        slicer.util.mainWindow().resize(1100, 780)
        slicer.app.processEvents()
        widget._uiWidget.grab().save(str(output / "default_algorithms.png"))

        node.targetMicrostructure = True
        node.targetBundles = True
        widget._advancedSettingsPanel.collapsed = False
        ui.parametersCollapsibleButton.collapsed = False
        ui.orchestrationFreeWaterCheckBox.setChecked(True)
        ui.orchestrationNoddiCheckBox.setChecked(True)
        ui.orchestrationDkiCheckBox.setChecked(True)
        slicer.app.processEvents()
        expect(ui.targetMicrostructureCheckBox, "Selected: DTI + FW + NODDI + DKI (WLS)")
        for method, label in (("OLS", "DKI (OLS)"), ("MKCURVE", "MK-Curve"),
                              ("MKCURVE_MATLAB", "MK-Curve (MATLAB)")):
            ui.orchestrationDkiFitMethodComboBox.setCurrentText(method)
            slicer.app.processEvents()
            assert node.orchestrationDkiFitMethod == method
            expect(ui.targetMicrostructureCheckBox, "Selected: DTI + FW + NODDI + " + label)
        for backend, label in (("ukf", "UKF"), ("dti_det", "DTI"), ("saf", "CSD + SAF")):
            ui.orchestrationTractographyBackendComboBox.setCurrentText(backend)
            slicer.app.processEvents()
            assert node.orchestrationTractographyBackend == backend
            expect(ui.targetTractographyCheckBox, "Selected: " + label)
        for backend, label in (("wma", "WMA"), ("tractcloud", "TractCloud"),
                                ("all", "WMQL + WMA + TractCloud")):
            ui.orchestrationBundleBackendComboBox.setCurrentText(backend)
            slicer.app.processEvents()
            assert node.orchestrationBundleBackend == backend
            expect(ui.targetBundlesCheckBox, "Selected: " + label)
        language.setCurrentIndex(language.findData("zh"))
        slicer.app.processEvents()
        expect(ui.targetBundlesCheckBox, "当前：WMQL + WMA + TractCloud")
        assert "选择" in widget._advancedSettingsHintLabel.text
        node.orchestrationTractographyBackend = "ukf"
        slicer.app.processEvents()
        expect(ui.targetTractographyCheckBox, "当前：UKF")
        with slicer.util.NodeModify(node):
            node.orchestrationMicrostructureStages = "dti,noddi"
            node.orchestrationBundleBackend = "wmql"
        slicer.app.processEvents()
        expect(ui.targetMicrostructureCheckBox, "当前：DTI + NODDI")
        expect(ui.targetBundlesCheckBox, "当前：WMQL")
        widget._advancedSettingsPanel.collapsed = True
        slicer.app.processEvents()
        widget._uiWidget.grab().save(str(output / "changed_algorithms_zh.png"))
        payload = {"status": "passed", "checks": checks,
                   "module_path": str(Path(__import__("shared.ui_compact", fromlist=[""]).__file__).resolve())}
        (output / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print("ALGORITHM_LABELS_GUI_PASS", flush=True)
        slicer.app.exit(0)
    except Exception:
        (output / "result.json").write_text(json.dumps({"status": "failed", "checks": checks,
                                                        "error": traceback.format_exc()}, indent=2), encoding="utf-8")
        traceback.print_exc()
        slicer.app.exit(1)


qt.QTimer.singleShot(4000, run)
