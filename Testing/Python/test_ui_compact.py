"""Pure contracts for concise labels and pending-input presentation."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

from shared.ui_compact import apply_compact_language, compact_algorithm_labels, refresh_compact_algorithms
from shared.ui_onboarding import translated_ui_message
from shared.user_config import RUN_DEFAULTS


class Text:
    def setText(self, value):
        self.text = value

    def setToolTip(self, value):
        self.tooltip = value


class CompactUiTest(unittest.TestCase):
    def test_language_changes_presentation_not_analysis_state(self):
        names = ("headlineLabel", "configurationCollapsibleButton", "targetsCollapsibleButton",
                 "parametersCollapsibleButton", "environmentCollapsibleButton", "resultsCollapsibleButton",
                 "runSelectedButton", "logHintLabel", "targetPreprocessingCheckBox", "targetSurfaceCheckBox",
                 "targetSurfacePostprocessCheckBox", "targetParcellationCheckBox", "targetSegmentationCheckBox",
                 "targetMicrostructureCheckBox", "targetTractographyCheckBox", "targetBundlesCheckBox",
                 "targetConnectomeCheckBox")
        ui = SimpleNamespace(**{name: Text() for name in names})
        ui.targetSurfaceCheckBox.checked = True
        node = SimpleNamespace(orchestrationSelectStreamlines=1000000)
        owner = SimpleNamespace(ui=ui, _advancedSettingsPanel=Text(), _parameterNode=node,
                                _advancedSettingsHintLabel=Text(),
                                _onboardingWidgets={"language_combo": Text()},
                                _prepareInputWidgets={"panel": Text()})
        for language, advanced, run, log in (("en", "Advanced settings", "Run SlicerDSpace", "Log"),
                                           ("zh", "高级设置", "运行 SlicerDSpace", "日志")):
            apply_compact_language(owner, language)
            self.assertEqual(owner._advancedSettingsPanel.text, advanced)
            self.assertEqual(ui.runSelectedButton.text, run)
            self.assertEqual(ui.logHintLabel.text, log)
            self.assertEqual(ui.headlineLabel.text, "SlicerDSpace")
            self.assertIn("import" if language == "en" else "导入", owner._prepareInputWidgets["panel"].text)
            self.assertTrue(ui.targetSurfaceCheckBox.checked)
            self.assertEqual(node.orchestrationSelectStreamlines, 1000000)
            self.assertEqual(ui.targetMicrostructureCheckBox.text,
                             "Selected: DTI" if language == "en" else "当前：DTI")
            self.assertIn("...", owner._advancedSettingsHintLabel.text)
            self.assertEqual(owner._advancedSettingsHintLabel.text, owner._advancedSettingsPanel.tooltip)

    def test_default_algorithms_match_run_defaults(self):
        node = SimpleNamespace(
            orchestrationMicrostructureStages=",".join(RUN_DEFAULTS["microstructure_stages"]),
            orchestrationTractographyBackend=RUN_DEFAULTS["tractography_backend"],
            orchestrationBundleBackend=RUN_DEFAULTS["bundle_backend"],
        )
        expected = {"targetMicrostructureCheckBox": "DTI",
                    "targetTractographyCheckBox": "MRtrix3 CSD / ACT",
                    "targetBundlesCheckBox": "WMQL"}
        self.assertEqual(compact_algorithm_labels(node), expected)
        self.assertEqual(compact_algorithm_labels(None), expected)

    def test_optional_models_and_dki_method_are_independent(self):
        for flags in range(8):
            stages = ["dti"] + [stage for bit, stage in enumerate(("freewater", "noddi", "dki"))
                                if flags & (1 << bit)]
            for method, dki in (("WLS", "DKI (WLS)"), ("OLS", "DKI (OLS)"),
                                ("MKCURVE", "MK-Curve"), ("MKCURVE_MATLAB", "MK-Curve (MATLAB)")):
                with self.subTest(stages=stages, method=method):
                    node = SimpleNamespace(orchestrationMicrostructureStages=",".join(reversed(stages)),
                                           orchestrationDkiFitMethod=method,
                                           orchestrationRunFreeWater="freewater" not in stages,
                                           orchestrationRunNoddi="noddi" not in stages)
                    expected = ["DTI"]
                    expected.extend(label for stage, label in (("freewater", "FW"), ("noddi", "NODDI"), ("dki", dki))
                                    if stage in stages)
                    self.assertEqual(compact_algorithm_labels(node)["targetMicrostructureCheckBox"],
                                     " + ".join(expected))
        node = SimpleNamespace(orchestrationMicrostructureStages="", orchestrationRunFreeWater=True,
                               orchestrationRunNoddi=True, orchestrationRunDki=True)
        self.assertEqual(compact_algorithm_labels(node)["targetMicrostructureCheckBox"],
                         "DTI + FW + NODDI + MK-Curve")

    def test_backend_changes_update_labels_without_mutating_state(self):
        names = ("targetMicrostructureCheckBox", "targetTractographyCheckBox", "targetBundlesCheckBox")
        ui = SimpleNamespace(**{name: Text() for name in names})
        for control in vars(ui).values():
            control.checked = False
            control.enabled = True
        node = SimpleNamespace(orchestrationMicrostructureStages="dti,noddi",
                               orchestrationTractographyBackend="ukf", orchestrationBundleBackend="wma")
        owner = SimpleNamespace(ui=ui, _parameterNode=node, _onboardingLanguage="en")
        for backend, label in (("csd_act", "MRtrix3 CSD / ACT"), ("ukf", "UKF"),
                               ("dti_det", "DTI deterministic"), ("saf", "CSD + SAF")):
            for bundle, bundle_label in (("wmql", "WMQL"), ("wma", "WMA"), ("tractcloud", "TractCloud"),
                                         ("all", "WMQL + WMA + TractCloud")):
                node.orchestrationTractographyBackend = backend
                node.orchestrationBundleBackend = bundle
                before = vars(node).copy()
                refresh_compact_algorithms(owner)
                self.assertEqual(ui.targetTractographyCheckBox.text, "Selected: " + label)
                self.assertEqual(ui.targetBundlesCheckBox.text, "Selected: " + bundle_label)
                self.assertEqual(ui.targetMicrostructureCheckBox.text, "Selected: DTI + NODDI")
                self.assertIn("Use the ... button", ui.targetBundlesCheckBox.tooltip)
                owner._onboardingLanguage = "zh"
                refresh_compact_algorithms(owner)
                self.assertEqual(ui.targetTractographyCheckBox.text, "当前：" + label)
                self.assertIn("... 按钮", ui.targetBundlesCheckBox.tooltip)
                owner._onboardingLanguage = "en"
                self.assertEqual(vars(node), before)
                self.assertTrue(all(not control.checked and control.enabled for control in vars(ui).values()))

    def test_pending_input_status_switches_both_directions(self):
        english = "Waiting: Subject ID; Subject directory"
        chinese = "待填写：受试者 ID; 受试者目录"
        self.assertEqual(translated_ui_message("zh", english), chinese)
        self.assertEqual(translated_ui_message("en", chinese), english)

    def test_empty_inputs_are_neutral_until_manual_validation(self):
        path = Path(__file__).resolve().parents[2] / "anatomy/surface/module_widget.py"
        method = next(node for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))) if
                      isinstance(node, ast.FunctionDef) and node.name == "validateInputs")
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        updates = []
        statuses = []
        owner = SimpleNamespace(_evaluateInputs=lambda **kwargs: (False, ["Subject ID", "Subject directory"]),
            _pipelineActive=False, _parameterNode=None, _lastStatusMessage="pending",
            ui=SimpleNamespace(subjectIdLineEdit="id", inputDirectoryPath="input", outputDirectoryPath="output"),
            _setStatus=lambda *args: statuses.append(args), appendLog=lambda *args: None,
            _summarise_missing=lambda *args: "pending", _setWidgetError=lambda *args: updates.append(args))
        validate = namespace["validateInputs"]
        self.assertFalse(validate(owner))
        self.assertFalse(any(error for widget, error in updates))
        self.assertEqual(statuses[-1][0], "waiting")
        updates.clear()
        self.assertFalse(validate(owner, manual=True))
        self.assertIn(("id", True), updates)
        self.assertIn(("input", True), updates)
        self.assertEqual(statuses[-1][0], "error")
        self.assertFalse(validate(owner))
        self.assertEqual(statuses[-1][0], "error")
        owner._evaluateInputs = lambda **kwargs: (True, [])
        self.assertTrue(validate(owner))
        self.assertEqual(statuses[-1][0], "waiting")
        self.assertIn("Dependencies are checked", statuses[-1][1])
        self.assertFalse(owner._inputValidationAttempted)

    def test_invalid_run_click_returns_before_setup_or_launch(self):
        path = Path(__file__).resolve().parents[2] / "shared/module_widget.py"
        method = next(node for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))) if
                      isinstance(node, ast.FunctionDef) and node.name == "_onRunSelectedClicked")
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        statuses = []
        validations = []
        owner = SimpleNamespace(_pipelineActive=False, logic=object(),
            _parameterNode=SimpleNamespace(batchMode=False, inputDirectory=""),
            validateInputs=lambda **kwargs: validations.append(kwargs) or False,
            _setStatus=lambda *args: statuses.append(args),
            _summarise_missing=lambda entries: "Waiting: " + "; ".join(entries))
        click = namespace["_onRunSelectedClicked"]
        click(owner)
        self.assertEqual(validations, [{"manual": True}])
        owner._parameterNode.batchMode = True
        click(owner)
        self.assertFalse(owner._parameterNode.batchMode)
        self.assertEqual(validations, [{"manual": True}, {"manual": True}])
        owner._parameterNode.batchMode = False
        owner.validateInputs = lambda **kwargs: True
        owner._selected_workflows = lambda: []
        click(owner)
        self.assertEqual(statuses[-1], ("error", "Waiting: Target products"))
        owner._pipelineActive = True
        owner.validateInputs = lambda **kwargs: self.fail("Active run must not revalidate or start")
        click(owner)

    def test_advanced_layout_fit_never_calls_adjust_size_to_narrow_panel(self):
        root = Path(__file__).resolve().parents[2]
        for rel in ("shared/ui_advanced_layout.py", "shared/ui_compact.py", "shared/ui_regions.py"):
            source = (root / rel).read_text(encoding="utf-8")
            self.assertNotIn("adjustSize(", source, rel)
            self.assertNotIn("contentsRect().width()", source, rel)
        compact_source = (root / "shared/ui_compact.py").read_text(encoding="utf-8")
        self.assertIn("root.resize(root.width, target_height)", compact_source)


if __name__ == "__main__":
    unittest.main()
