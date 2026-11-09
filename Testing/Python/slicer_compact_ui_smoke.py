"""Check compact layout, localization and connected advanced controls in Slicer."""

import json
import os
from pathlib import Path
import traceback

import qt
import slicer

ROOT = Path(os.environ["SLICERDSPACE_SMOKE_ROOT"])


def check():
    try:
        slicer.util.selectModule("SlicerDSpace")
        owner = slicer.modules.slicerdspace.widgetRepresentation().self()
        ui = owner.ui
        advanced = owner._advancedSettingsPanel
        assert ui.gettingStartedCollapsibleButton.isHidden()
        assert advanced.collapsed
        assert owner._cancelRunButton.isHidden()
        assert owner._runProgressBar.isHidden()
        prepare = owner._prepareInputWidgets["panel"]
        assert advanced.isAncestorOf(prepare) and prepare.collapsed
        assert not ui.configurationCollapsibleButton.isAncestorOf(prepare)
        for name in ("environmentCollapsibleButton", "resultsCollapsibleButton"):
            section = getattr(ui, name)
            assert not advanced.isAncestorOf(section) and section.collapsed, name
        assert ui.headerFrame.isAncestorOf(owner._onboardingWidgets["language_combo"])
        assert not owner._onboardingWidgets["language_combo"].isHidden()

        # Invalid clicks must show inline feedback without preflight or a job.
        node = owner._parameterNode
        with slicer.util.NodeModify(node):
            node.batchMode = False
            node.subjectId = ""
            node.inputDirectory = ""
            node.outputDirectory = ""
        assert ui.runSelectedButton.enabled
        assert any(c in ui.statusLabel.styleSheet for c in ("#63727D", "#9BA7B4")), ui.statusLabel.styleSheet
        doctor = owner._environment_doctor_result_for_run

        def unexpected_preflight(*args, **kwargs):
            raise AssertionError("Invalid Run click reached dependency checks")

        owner._environment_doctor_result_for_run = unexpected_preflight
        ui.runSelectedButton.click()
        assert ui.runSelectedButton.enabled
        assert any(c in ui.statusLabel.styleSheet for c in ("#C62828", "#FF7B72")), ui.statusLabel.styleSheet
        assert not owner._pipelineActive
        with slicer.util.NodeModify(node):
            node.subjectId = "sub-test"
        assert any(c in ui.statusLabel.styleSheet for c in ("#C62828", "#FF7B72")), ui.statusLabel.styleSheet
        node.subjectId = ""

        window = slicer.util.mainWindow()
        window.showNormal()
        window.resize(1280, 900)
        slicer.util.findChild(window, "DataProbeCollapsibleWidget").collapsed = True
        panel = slicer.util.findChild(window, "PanelDockWidget")
        screenshots = []
        from shared.ui_onboarding import apply_onboarding_language
        for lang in ("en", "zh"):
            combo = owner._onboardingWidgets["language_combo"]
            combo.setCurrentIndex(combo.findData(lang))
            assert owner._onboardingLanguage == lang
            assert ui.runSelectedButton.text == ("Run SlicerDSpace" if lang == "en" else "运行 SlicerDSpace")
            ui.runSelectedButton.click()
            assert ui.runSelectedButton.enabled and any(c in ui.statusLabel.styleSheet for c in ("#C62828", "#FF7B72"))
            assert "Waiting:" in ui.statusLabel.text if lang == "en" else "待填写：" in ui.statusLabel.text
            assert "advanced" in str(advanced.text).lower() if lang == "en" else "高级" in advanced.text
            for width in (440, 540):
                panel.setFixedWidth(width)
                slicer.app.processEvents()
                # Every primary control must remain reachable without horizontal scrolling.
                assert ui.runSelectedButton.visible and ui.logPlainTextEdit.visible
                assert owner._uiWidget.width <= width, (owner._uiWidget.width, width)
                path = ROOT / f"compact_{lang}_{width}.png"
                owner._uiWidget.grab().save(str(path))
                screenshots.append(str(path))

        with slicer.util.NodeModify(node):
            node.batchMode = True
            node.batchRoot = ""
        ui.runSelectedButton.click()
        assert ui.runSelectedButton.enabled and any(c in ui.statusLabel.styleSheet for c in ("#C62828", "#FF7B72"))
        assert not node.batchMode and node.batchSize == 1
        assert "待填写：" in ui.statusLabel.text, ui.statusLabel.text
        assert "批处理根目录" not in ui.statusLabel.text
        with slicer.util.NodeModify(node):
            node.batchMode = False
            node.subjectId = "100307"
            node.inputDirectory = os.environ["SLICERDSPACE_SMOKE_SUBJECT_ROOT"]
        assert owner.validateInputs()
        assert any(c in ui.statusLabel.styleSheet for c in ("#63727D", "#9BA7B4")), ui.statusLabel.styleSheet
        assert "点击运行时将检查依赖" in ui.statusLabel.text
        owner._environment_doctor_result_for_run = doctor

        advanced.collapsed = False
        owner._advancedCategorySelector.setCurrentIndex(3)
        prepare.collapsed = False
        slicer.app.processEvents()
        slicer.app.processEvents()
        owner._uiWidget.grab().save(str(ROOT / "advanced_import.png"))
        print("IMPORT_VISIBILITY", {key: {"visible": value.visible, "hidden": value.isHidden(),
              "parent": value.parentWidget().objectName} for key, value in owner._prepareInputWidgets.items()
              if key in {"panel", "source", "preview_button", "prepare_button"}}, flush=True)
        assert owner._prepareInputWidgets["preview_button"].visible
        assert owner._prepareInputWidgets["prepare_button"].visible
        prepare.collapsed = True
        owner._advancedCategorySelector.setCurrentIndex(1)
        slicer.app.processEvents()
        assert not ui.orchestrationTractographyBackendComboBox.visible
        combo = ui.orchestrationTractographyBackendComboBox
        original = str(combo.currentText)
        index = combo.findText("dti_det")
        assert index >= 0
        combo.setCurrentIndex(index)
        assert owner._parameterNode.orchestrationTractographyBackend == "dti_det"
        combo.setCurrentIndex(combo.findText(original))
        ui.advancedRunCollapsibleButton.collapsed = False
        original_count = owner._parameterNode.orchestrationSelectStreamlines
        ui.orchestrationSelectStreamlinesSpinBox.setValue(1234)
        assert owner._parameterNode.orchestrationSelectStreamlines == 1234
        ui.orchestrationSelectStreamlinesSpinBox.setValue(original_count)
        ui.parametersCollapsibleButton.collapsed = True
        panel.setFixedWidth(440)
        for name in ("environmentCollapsibleButton", "resultsCollapsibleButton"):
            section = getattr(ui, name)
            section.collapsed = False
            slicer.app.processEvents()
            slicer.app.processEvents()
            assert owner._uiWidget.width <= 440, (name, owner._uiWidget.width)
            owner._uiWidget.grab().save(str(ROOT / f"advanced_{name}.png"))
            section.collapsed = True
        panel.setFixedWidth(540)
        owner._setPipelineActive(True)
        assert not ui.runSelectedButton.enabled
        assert owner._cancelRunButton.visible and owner._cancelRunButton.enabled
        owner._setPipelineActive(False)
        assert ui.runSelectedButton.enabled
        assert owner._cancelRunButton.isHidden()
        owner.appendLog("[Background] Starting layout verification")
        owner.appendLog("warning: test message; no scientific computation")
        assert "layout verification" in ui.logPlainTextEdit.toPlainText()
        ui.clearLogButton.click()
        assert not ui.logPlainTextEdit.toPlainText()
        ui.advancedRunCollapsibleButton.collapsed = True
        ui.parametersCollapsibleButton.collapsed = True
        advanced.collapsed = True
        apply_onboarding_language(owner, "en")
        panel.setFixedWidth(390)
        slicer.app.processEvents()
        panel.setMinimumWidth(0)
        panel.setMaximumWidth(16777215)
        window.resize(1280, 900)
        slicer.app.processEvents()
        slicer.app.processEvents()
        jitter_steps = [
            ("advanced_open", advanced, False),
            ("prepare_open", prepare, False),
            ("regions_open", owner._regionWidgets["panel"], False),
            ("region_options_open", owner._regionWidgets["options"], False),
            ("env_open", ui.environmentCollapsibleButton, False),
            ("env_more_open", owner._environmentMoreOptions, False),
            ("env_paths_open", owner._environmentAdvancedPathsWidget, False),
            ("results_open", ui.resultsCollapsibleButton, False),
            ("results_close", ui.resultsCollapsibleButton, True),
            ("env_paths_close", owner._environmentAdvancedPathsWidget, True),
            ("env_more_close", owner._environmentMoreOptions, True),
            ("env_close", ui.environmentCollapsibleButton, True),
            ("region_options_close", owner._regionWidgets["options"], True),
            ("regions_close", owner._regionWidgets["panel"], True),
            ("prepare_close", prepare, True),
            ("advanced_close", advanced, True),
        ]
        owner._advancedCategorySelector.setCurrentIndex(3)
        slicer.app.processEvents()
        jitter_trace = []
        for step_name, section_widget, collapsed_state in jitter_steps:
            section_widget.collapsed = collapsed_state
            ui_w_sync = owner._uiWidget.width
            ui_h_sync = owner._uiWidget.height
            slicer.app.processEvents()
            slicer.app.processEvents()
            ui_w = owner._uiWidget.width
            ui_h = owner._uiWidget.height
            jitter_trace.append({
                "step": step_name,
                "win": [window.width, window.height],
                "panel_w": panel.width,
                "ui_w_sync": ui_w_sync,
                "ui_w": ui_w,
                "ui_h_sync": ui_h_sync,
                "ui_h": ui_h,
                "w_diff": abs(ui_w - ui_w_sync),
                "h_diff": abs(ui_h - ui_h_sync),
            })
        (ROOT / "jitter_trace.json").write_text(json.dumps(jitter_trace, indent=2) + "\n")
        for item in jitter_trace:
            assert item["w_diff"] == 0, item
            assert item["win"] == [1280, 900], item
        (ROOT / "layout_height.json").write_text(json.dumps({"height": owner._uiWidget.height,
            "minimum_height": owner._uiWidget.minimumSizeHint.height(),
            "size_hint": owner._uiWidget.sizeHint.height()}, indent=2))
        assert owner._uiWidget.height < 800, owner._uiWidget.height
        assert not window.isMaximized() and not window.isFullScreen()
        assert window.width == 1280 and window.height == 900, (window.width, window.height)
        window.grab().save(str(ROOT / "compact_window.png"))
        sizes = {name: {"height": getattr(ui, name).height, "size_hint": getattr(ui, name).sizeHint.height()}
                 for name in ("headerFrame", "configurationCollapsibleButton", "targetsCollapsibleButton",
                              "runCollapsibleButton", "logCollapsibleButton")}
        (ROOT / "compact_ui.json").write_text(json.dumps({"status": "PASS", "screenshots": screenshots,
            "sizes": sizes, "root_height": owner._uiWidget.height,
            "checks": ["hidden beginner", "collapsed advanced", "header language", "440/540 px EN/ZH",
                       "backend binding", "streamline binding", "440 px advanced panels",
                       "active cancel", "log and clear", "collapsed height restored",
                       "import under advanced", "1280x900 normal window", "clickable empty Run",
                       "inline red Waiting without preflight", "batch empty Run", "valid input resets status",
                       "localized Run label", "active Run guarded"]}, indent=2) + "\n")
        print("COMPACT_UI_PASS", flush=True)
        slicer.app.exit(0)
    except Exception:
        err = traceback.format_exc()
        (ROOT / "smoke_error.log").write_text(err, encoding="utf-8")
        traceback.print_exc()
        slicer.app.exit(2)


qt.QTimer.singleShot(5000, check)
qt.QTimer.singleShot(90000, lambda: slicer.app.exit(3))
