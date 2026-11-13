from __future__ import annotations

import ast
import unittest
from types import SimpleNamespace
import xml.etree.ElementTree as ET
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
UI_PATH = REPO_ROOT / "Resources" / "UI" / "SlicerDSpace.ui"
MODULE_WIDGET_PATH = REPO_ROOT / "shared" / "module_widget.py"


def _load_ui() -> ET.Element:
    return ET.parse(UI_PATH).getroot()


def _widget(root: ET.Element, name: str) -> ET.Element:
    for element in root.iter("widget"):
        if element.attrib.get("name") == name:
            return element
    raise AssertionError(f"UI widget not found: {name}")


def _layout(root: ET.Element, name: str) -> ET.Element:
    for element in root.iter("layout"):
        if element.attrib.get("name") == name:
            return element
    raise AssertionError(f"UI layout not found: {name}")


def _property_text(element: ET.Element, property_name: str) -> str:
    for prop in element.findall("property"):
        if prop.attrib.get("name") != property_name:
            continue
        string = prop.find("string")
        if string is not None and string.text is not None:
            return string.text
        boolean = prop.find("bool")
        if boolean is not None and boolean.text is not None:
            return boolean.text
    return ""


class SlicerUiLayoutContractTest(unittest.TestCase):
    def test_running_another_backend_does_not_archive_previous_tractograms(self) -> None:
        source = REPO_ROOT / "shared/orchestrator.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call) and
                    isinstance(node.func, ast.Name) and node.func.id == "archive_development_outputs")
        flag = next(arg.value for arg in call.keywords if arg.arg == "archive_alternate_tractograms")
        self.assertIsInstance(flag, ast.Constant)
        self.assertIs(flag.value, False)

    def test_completion_keeps_run_disabled_until_result_loading_finishes(self) -> None:
        tree = ast.parse(MODULE_WIDGET_PATH.read_text(encoding="utf-8"))
        method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                      and node.name == "_onBackgroundProcessFinished")
        calls = [node for node in ast.walk(method) if isinstance(node, ast.Call) and
                 isinstance(node.func, ast.Attribute)]
        release = [node.lineno for node in calls if node.func.attr == "_setPipelineActive"]
        loads = [node.lineno for node in calls if node.func.attr == "_load_background_manifest_result"]
        self.assertEqual(len(release), 1)
        self.assertTrue(loads and release[0] > max(loads))

    def test_completion_preserves_terminal_status_while_reenabling_buttons(self) -> None:
        tree = ast.parse(MODULE_WIDGET_PATH.read_text(encoding="utf-8"))
        method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                      and node.name == "_onBackgroundProcessFinished")
        call = next(node for node in ast.walk(method) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute) and node.func.attr == "_setPipelineActive")
        flag = next(arg.value for arg in call.keywords if arg.arg == "preserve_status")
        self.assertIs(flag.value, True)
        source = REPO_ROOT / "anatomy/surface/module_widget.py"
        method = next(node for node in ast.walk(ast.parse(source.read_text())) if
                      isinstance(node, ast.FunctionDef) and node.name == "validateInputs")
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        updates = []
        owner = SimpleNamespace(_evaluateInputs=lambda **kwargs: (True, []), _pipelineActive=False,
                                _parameterNode=None, ui=SimpleNamespace(validateButton=SimpleNamespace(enabled=False)),
                                _lastStatusMessage="Pipeline cancelled", _summarise_missing=lambda missing: "Waiting",
                                _setStatus=lambda *args: updates.append(args), _setWidgetError=lambda *args: None)
        self.assertTrue(namespace["validateInputs"](owner, update_status=False))
        self.assertTrue(owner.ui.validateButton.enabled)
        self.assertEqual(updates, [])

    def test_widget_routes_the_postprocess_checkbox_to_selected_targets(self) -> None:
        from shared.ui_run_controller import ProductTargetFlags, selected_orchestrator_targets
        tree = ast.parse(MODULE_WIDGET_PATH.read_text(encoding="utf-8"))
        method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                      and node.name == "_product_target_flags")
        namespace = {"ProductTargetFlags": ProductTargetFlags}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(MODULE_WIDGET_PATH), "exec"), namespace)
        node = SimpleNamespace(**{name: False for name in (
            "targetPreprocessing", "targetSurface", "targetSurfacePostprocess", "targetParcellation",
            "targetSegmentation", "targetMicrostructure", "targetTractography", "targetBundles", "targetConnectome")})
        owner = SimpleNamespace(_parameterNode=node)
        node.targetSurfacePostprocess = True
        flags = namespace["_product_target_flags"](owner)
        self.assertTrue(flags.surface_postprocess)
        self.assertIn("surface_postprocess", selected_orchestrator_targets(flags))
        node.targetSurfacePostprocess = False
        self.assertFalse(namespace["_product_target_flags"](owner).surface_postprocess)

    def test_main_workflow_sections_follow_workflow_order(self) -> None:
        root = _load_ui()
        expected = [
            ("gettingStartedCollapsibleButton", "Getting started"),
            ("configurationCollapsibleButton", "1. Inputs"),
            ("targetsCollapsibleButton", "2. Analysis"),
            ("parametersCollapsibleButton", "3. Route and parameters"),
            ("resultsCollapsibleButton", "5. Results"),
            ("environmentCollapsibleButton", "Advanced: Dependency profile"),
        ]
        top_level = [
            element.attrib.get("name")
            for element in _layout(root, "verticalLayout").findall("item/widget")
            if element.attrib.get("class") == "ctkCollapsibleButton"
        ]
        ordered = [name for name in top_level if name in {name for name, _ in expected}]
        self.assertEqual(ordered, [name for name, _ in expected])
        for name, text in expected:
            self.assertEqual(_property_text(_widget(root, name), "text"), text)
        self.assertEqual(_property_text(_widget(root, "runSelectedButton"), "text"), "Run SlicerDSpace")
        self.assertEqual(_property_text(_widget(root, "runSelectedButton"), "styleSheet"), "")

    def test_run_and_log_are_plain_frames_without_extra_section_headers(self) -> None:
        root = _load_ui()
        for name in ("runCollapsibleButton", "logCollapsibleButton"):
            self.assertEqual(_widget(root, name).attrib["class"], "QFrame")
            self.assertEqual(_property_text(_widget(root, name), "text"), "")

    def test_compact_layout_is_installed_after_all_dynamic_controls(self) -> None:
        tree = ast.parse(MODULE_WIDGET_PATH.read_text(encoding="utf-8"))
        method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                      and node.name == "_initialize_common_ui_state")
        calls = [node for node in ast.walk(method) if isinstance(node, ast.Call)]
        compact = next(node for node in calls if isinstance(node.func, ast.Name)
                       and node.func.id == "install_compact_layout")
        controls = [node for node in calls if isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"_initialize_onboarding_panel", "_initialize_run_status_panel",
                                           "_initialize_background_run_controls"}]
        self.assertTrue(all(node.lineno < compact.lineno for node in controls))
        self.assertFalse(any(isinstance(node.func, ast.Name) and
                             node.func.id == "maybe_show_first_run_wizard" for node in calls))

    def test_route_section_holds_backends_and_topup_group(self) -> None:
        root = _load_ui()
        parameters = _widget(root, "parametersCollapsibleButton")
        names = {element.attrib.get("name") for element in parameters.iter("widget")}
        for name in (
            "topupCollapsibleGroupBox",
            "preprocessingPhaseEncodingDirectionComboBox",
            "preprocessingReadoutTimeSpinBox",
            "preprocessingReversePhaseB0Path",
            "orchestrationTractographyBackendComboBox",
            "orchestrationBundleBackendComboBox",
            "orchestrationMicrostructureStagesComboBox",
            "advancedRunCollapsibleButton",
        ):
            self.assertIn(name, names)
        self.assertEqual(_property_text(_widget(root, "topupCollapsibleGroupBox"), "title"), "Distortion correction (topup)")
        self.assertEqual(_property_text(_widget(root, "preprocessingReadoutTimeLabel"), "text"), "Total readout time (s)")
        self.assertEqual(_property_text(_widget(root, "preprocessingPhaseEncodingDirectionLabel"), "text"), "Phase-encoding direction")
        self.assertEqual(_property_text(_widget(root, "preprocessingReversePhaseB0Label"), "text"), "Reverse-PE b=0 image")
        spin = _widget(root, "preprocessingReadoutTimeSpinBox")
        self.assertEqual(_property_text(spin, "specialValueText"), "Not set")
        # ctkPathLineEdit stops emitting currentPathChanged for .nii.gz when nameFilters are set,
        # which would silently drop the reverse-PE path; suffixes are validated in Python instead.
        reverse = _widget(root, "preprocessingReversePhaseB0Path")
        self.assertNotIn("nameFilters", {prop.attrib.get("name") for prop in reverse.findall("property")})
        # The topup widgets are bound by hand (combo data != label), not via SlicerParameterName.
        for name in ("preprocessingPhaseEncodingDirectionComboBox", "preprocessingReadoutTimeSpinBox", "preprocessingReversePhaseB0Path"):
            self.assertEqual(_property_text(_widget(root, name), "SlicerParameterName"), "")

    def test_w2a_gui_offers_no_preprocessing_route_or_denoise_choice(self) -> None:
        """W2a (D1): preprocessing has one route, so the GUI has no route combo, no denoise combo and no motion switch."""
        root = _load_ui()
        names = {element.attrib.get("name") for element in root.iter("widget")}
        for name in ("preprocessingBackendLabel", "preprocessingBackendComboBox"):
            self.assertNotIn(name, names)
        self.assertFalse([name for name in names if name and ("Denoise" in name or ("Motion" in name and "preprocessing" in name))])
        texts = {element.text for element in root.iter("string")}
        self.assertNotIn("mrtrix_fsl", texts)
        self.assertNotIn("preprocessingBackendComboBox", {element.text for element in root.findall("tabstops/tabstop")})
        parameter_names = {element.text for element in root.iter("string")
                           if element.text and element.text.startswith("preprocessing")}
        self.assertNotIn("preprocessingBackend", parameter_names)
        source = (REPO_ROOT / "SlicerDSpace.py").read_text(encoding="utf-8")
        self.assertNotIn("preprocessingBackend", source)
        self.assertNotIn("preprocessingBackend", MODULE_WIDGET_PATH.read_text(encoding="utf-8"))

    def test_w2a_a_platform_without_an_fsl_build_does_not_offer_preprocessing(self) -> None:
        from shared.selected_setup import PREPROCESSING_UNSUPPORTED_TEXT
        from shared.ui_parameter_sync import apply_preprocessing_availability

        from unittest.mock import patch

        calls = []
        checkbox = SimpleNamespace(setEnabled=lambda value: calls.append(("enabled", value)),
                                   setToolTip=lambda text: calls.append(("tip", text)),
                                   blockSignals=lambda value: calls.append(("signals", value)),
                                   setChecked=lambda value: calls.append(("checked", value)))
        patcher = patch("platform.mac_ver", return_value=("15.1", ("", "", ""), ""))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.assertFalse(apply_preprocessing_availability(checkbox, "Linux", "aarch64"))
        self.assertEqual(calls, [("enabled", False), ("tip", PREPROCESSING_UNSUPPORTED_TEXT),
                                 ("signals", True), ("checked", False), ("signals", False)])
        self.assertEqual(PREPROCESSING_UNSUPPORTED_TEXT, "Preprocessing needs FSL, which has no build for this platform; "
                                                         "import preprocessed data instead")
        for system, machine in (("Linux", "x86_64"), ("Darwin", "arm64"), ("Darwin", "x86_64"), ("Windows", "AMD64")):
            calls.clear()
            self.assertTrue(apply_preprocessing_availability(checkbox, system, machine), (system, machine))
            self.assertEqual(calls, [])

    def test_tab_order_covers_the_main_workflow(self) -> None:
        root = _load_ui()
        stops = [element.text for element in root.findall("tabstops/tabstop")]
        for earlier, later in (
            ("subjectIdLineEdit", "inputDirectoryPath"),
            ("inputDirectoryPath", "targetPreprocessingCheckBox"),
            ("targetConnectomeCheckBox", "preprocessingPhaseEncodingDirectionComboBox"),
            ("preprocessingReversePhaseB0Path", "runSelectedButton"),
        ):
            self.assertLess(stops.index(earlier), stops.index(later))
        widget_names = {element.attrib.get("name") for element in root.iter("widget")}
        self.assertTrue(set(stops) <= widget_names)

    def test_forms_wrap_long_rows_for_narrow_panels(self) -> None:
        root = _load_ui()
        for name in ("configurationFormLayout", "targetProductsFormLayout", "parametersFormLayout", "topupFormLayout", "advancedRunFormLayout"):
            layout = _layout(root, name)
            policies = {prop.attrib.get("name"): prop.findtext("enum") for prop in layout.findall("property")}
            self.assertEqual(policies.get("rowWrapPolicy"), "QFormLayout::WrapLongRows", name)

    def test_advanced_sections_are_collapsed_by_default(self) -> None:
        root = _load_ui()
        environment = _widget(root, "environmentCollapsibleButton")
        advanced = _widget(root, "advancedRunCollapsibleButton")
        self.assertEqual(_property_text(environment, "text"), "Advanced: Dependency profile")
        self.assertEqual(_property_text(environment, "collapsed"), "true")
        self.assertEqual(
            _property_text(advanced, "text"),
            "Analysis settings",
        )
        self.assertEqual(_property_text(advanced, "collapsed"), "true")
        self.assertIsNotNone(_layout(root, "advancedRunFormLayout"))

    def test_native_surface_and_freesurfer_postprocess_are_distinct_targets(self) -> None:
        root = _load_ui()
        native_surface = _widget(root, "targetSurfaceCheckBox")
        postprocess = _widget(root, "targetSurfacePostprocessCheckBox")

        self.assertIn("native", _property_text(native_surface, "text").lower())
        self.assertIn("does not create a complete FreeSurfer subject", _property_text(native_surface, "toolTip"))
        self.assertEqual(_property_text(postprocess, "checked"), "false")
        self.assertEqual(_property_text(postprocess, "SlicerParameterName"), "targetSurfacePostprocess")
        tooltip = _property_text(postprocess, "toolTip")
        self.assertIn("standalone native component runtime", tooltip)
        self.assertIn("FreeSurfer license", tooltip)
        self.assertIn("compatible native meshes", tooltip)
        self.assertNotIn("complete FreeSurfer installation", tooltip)

    def test_surface_postprocess_target_is_restored_from_json_defaults(self) -> None:
        source = MODULE_WIDGET_PATH.read_text(encoding="utf-8")
        self.assertIn('targets_configured = "targets" in defaults', source)
        self.assertIn("if targets_configured:", source)
        self.assertIn(
            'node.targetSurfacePostprocess = "surface_postprocess" in targets',
            source,
        )


if __name__ == "__main__":
    unittest.main()
