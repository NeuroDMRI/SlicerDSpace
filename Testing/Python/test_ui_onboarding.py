"""Tests for beginner onboarding helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

from shared.demo_data import create_synthetic_demo_subject
from shared.ui_onboarding import (
    apply_onboarding_language,
    capability_dashboard_text,
    dependency_ui_text,
    diagnostic_report_text,
    environment_section_text,
    normalized_language,
    onboarding_text,
    platform_path_examples,
    prepare_demo_for_owner,
    register_ui_language_binding,
    translated_ui_message,
)


@dataclass
class FakeItem:
    key: str
    label: str
    status: str
    targets: tuple[str, ...]


@dataclass
class FakeResult:
    runnable_targets: tuple[str, ...]
    blocked_targets: tuple[str, ...]
    statuses: tuple[FakeItem, ...]


class FakeOwner:
    _parameterNode = None
    _lastEnvironmentResult = None
    ui = None

    def _read_environment_path_setting(self, key: str) -> str:
        return {"model_assets_dir": "/tmp/models"}.get(key, "")


class FakeTextWidget:
    def __init__(self, text: str = "") -> None:
        self.value = text
        self.tooltip = ""
        self.placeholder = ""

    def setText(self, text: str) -> None:
        self.value = text

    def text(self) -> str:
        return self.value

    def setToolTip(self, text: str) -> None:
        self.tooltip = text

    def setPlaceholderText(self, text: str) -> None:
        self.placeholder = text


class FakeCombo:
    def __init__(self) -> None:
        self.currentIndex = 0
        self.signals_blocked = False

    def findData(self, language: str) -> int:
        return {"en": 0, "zh": 1}.get(language, -1)

    def setCurrentIndex(self, index: int) -> None:
        self.currentIndex = index

    def blockSignals(self, blocked: bool) -> None:
        self.signals_blocked = blocked


class FakeTable:
    def __init__(self) -> None:
        self.headers: list[str] = []

    def setHorizontalHeaderLabels(self, headers: list[str]) -> None:
        self.headers = list(headers)


class FakeLanguageUi:
    def __init__(self) -> None:
        for name in (
            "headlineLabel",
            "subtitleLabel",
            "environmentCollapsibleButton",
            "environmentResultLabel",
            "configurationCollapsibleButton",
            "targetsCollapsibleButton",
            "parametersCollapsibleButton",
            "runCollapsibleButton",
            "resultsCollapsibleButton",
            "preprocessingReadoutTimeLabel",
            "subjectIdLabel",
            "inputDirectoryLabel",
            "outputDirectoryLabel",
            "targetSurfaceLabel",
            "targetSurfaceCheckBox",
            "targetSurfacePostprocessLabel",
            "targetSurfacePostprocessCheckBox",
            "targetParcellationLabel",
            "targetSegmentationLabel",
            "targetBundlesLabel",
            "targetBundlesCheckBox",
            "advancedRunCollapsibleButton",
            "orchestrationBundleBackendLabel",
            "validateButton",
            "runSelectedButton",
            "logCollapsibleButton",
            "logHintLabel",
            "clearLogButton",
            "statusLabel",
        ):
            setattr(self, name, FakeTextWidget())


class FakeLanguageOwner:
    def __init__(self) -> None:
        self.ui = FakeLanguageUi()
        self._lastEnvironmentResult = None
        self._statusMessageSource = "Waiting for input"
        self._onboardingWidgets = {
            "title": FakeTextWidget(),
            "language_label": FakeTextWidget(),
            "body": FakeTextWidget(),
            "wizard_button": FakeTextWidget(),
            "demo_button": FakeTextWidget(),
            "diagnostic_button": FakeTextWidget(),
            "config_button": FakeTextWidget(),
            "dashboard": FakeTextWidget(),
            "examples": FakeTextWidget(),
            "language_combo": FakeCombo(),
        }
        self._runStatusPanel = FakeTextWidget()
        self._runStatusHintLabel = FakeTextWidget("Run plan has not been generated yet.")
        self._runSummaryLabel = FakeTextWidget("Run summary will appear after a plan or run completes.")
        self._runReportButton = FakeTextWidget()
        self._runOutputTreeButton = FakeTextWidget()
        self._runStatusTable = FakeTable()


class OnboardingHelpersTest(unittest.TestCase):
    def test_normalized_language_defaults_to_english(self) -> None:
        self.assertEqual(normalized_language(""), "en")
        self.assertEqual(normalized_language("zh-CN"), "zh")

    def test_main_ui_switches_between_chinese_and_english(self) -> None:
        owner = FakeLanguageOwner()

        apply_onboarding_language(owner, "zh")

        self.assertEqual(owner._onboardingLanguage, "zh")
        self.assertEqual(owner.ui.headlineLabel.value, "SlicerDSpace 运行工作台")
        self.assertEqual(owner.ui.subjectIdLabel.value, "受试者 ID")
        self.assertEqual(owner.ui.targetSurfaceLabel.value, "皮层表面")
        self.assertEqual(owner.ui.targetSurfacePostprocessLabel.value, "表面后处理")
        self.assertEqual(owner.ui.targetParcellationLabel.value, "脑区分割")
        self.assertEqual(owner.ui.targetSegmentationLabel.value, "组织分割")
        self.assertEqual(owner.ui.targetSurfacePostprocessCheckBox.value, "FreeSurfer 风格后处理")
        self.assertIn("DDSurfer", owner.ui.targetSurfaceCheckBox.tooltip)
        self.assertIn("双侧白质与软脑膜皮层表面", owner.ui.targetSurfaceCheckBox.tooltip)
        self.assertEqual(owner.ui.targetSurfaceLabel.tooltip, owner.ui.targetSurfaceCheckBox.tooltip)
        self.assertIn("不强制安装完整 FreeSurfer", owner.ui.targetSurfacePostprocessCheckBox.tooltip)
        self.assertIn("需许可证", owner.ui.targetSurfacePostprocessCheckBox.tooltip)
        self.assertEqual(owner.ui.targetSurfacePostprocessLabel.tooltip, owner.ui.targetSurfacePostprocessCheckBox.tooltip)
        self.assertEqual(owner.ui.targetBundlesLabel.value, "纤维束")
        self.assertEqual(owner.ui.runSelectedButton.value, "运行 SlicerDSpace")
        self.assertEqual(owner.ui.statusLabel.value, "等待输入")
        self.assertEqual(owner.ui.configurationCollapsibleButton.value, "1. 输入")
        self.assertEqual(owner.ui.resultsCollapsibleButton.value, "5. 结果")
        self.assertEqual(owner.ui.preprocessingReadoutTimeLabel.value, "总读出时间（秒）")
        self.assertEqual(owner._runStatusTable.headers, ["目标", "状态", "原因", "清单", "质控", "日志"])
        self.assertEqual(owner._onboardingWidgets["language_combo"].currentIndex, 1)

        apply_onboarding_language(owner, "en")

        self.assertEqual(owner._onboardingLanguage, "en")
        self.assertEqual(owner.ui.headlineLabel.value, "SlicerDSpace")
        self.assertEqual(owner.ui.subjectIdLabel.value, "Subject ID")
        self.assertEqual(owner.ui.targetSurfaceLabel.value, "Cortical surfaces")
        self.assertEqual(owner.ui.targetSurfacePostprocessLabel.value, "Surface postprocess")
        self.assertEqual(owner.ui.targetParcellationLabel.value, "Brain parcellation")
        self.assertEqual(owner.ui.targetSegmentationLabel.value, "Tissue segmentation")
        self.assertEqual(owner.ui.targetSurfacePostprocessCheckBox.value, "FreeSurfer-style postprocess")
        self.assertIn("DDSurfer", owner.ui.targetSurfaceCheckBox.tooltip)
        self.assertIn("white and pial cortical surfaces", owner.ui.targetSurfaceCheckBox.tooltip)
        self.assertEqual(owner.ui.targetSurfaceLabel.tooltip, owner.ui.targetSurfaceCheckBox.tooltip)
        self.assertIn("full FreeSurfer installation is optional", owner.ui.targetSurfacePostprocessCheckBox.tooltip)
        self.assertIn("license required", owner.ui.targetSurfacePostprocessCheckBox.tooltip)
        self.assertEqual(owner.ui.targetSurfacePostprocessLabel.tooltip, owner.ui.targetSurfacePostprocessCheckBox.tooltip)
        self.assertEqual(owner.ui.targetBundlesLabel.value, "Fiber bundles")
        self.assertEqual(owner.ui.runSelectedButton.value, "Run SlicerDSpace")
        self.assertEqual(owner.ui.statusLabel.value, "Waiting for input")
        self.assertEqual(owner.ui.configurationCollapsibleButton.value, "1. Inputs")
        self.assertEqual(owner.ui.targetsCollapsibleButton.value, "2. Analysis")
        self.assertEqual(owner.ui.parametersCollapsibleButton.value, "3. Route and parameters")
        self.assertEqual(owner.ui.resultsCollapsibleButton.value, "5. Results")
        self.assertEqual(owner.ui.preprocessingReadoutTimeLabel.value, "Total readout time (s)")
        self.assertEqual(owner._runStatusTable.headers, ["Target", "State", "Reason", "Manifest", "QC", "Logs"])
        self.assertEqual(owner._onboardingWidgets["language_combo"].currentIndex, 0)

    def test_every_workflow_text_key_has_english_and_chinese(self) -> None:
        from shared.ui_onboarding import TRANSLATIONS
        from shared.ui_workflow_text import WORKFLOW_TEXT, WORKFLOW_TEXT_BINDINGS, WORKFLOW_TOOLTIP_BINDINGS

        self.assertEqual(set(WORKFLOW_TEXT["en"]), set(WORKFLOW_TEXT["zh"]))
        for key in list(WORKFLOW_TEXT_BINDINGS.values()) + list(WORKFLOW_TOOLTIP_BINDINGS.values()):
            self.assertIn(key, TRANSLATIONS["en"])
            self.assertIn(key, TRANSLATIONS["zh"])
        self.assertEqual(onboarding_text("en", "browse_runs"), "Browse…")
        self.assertEqual(onboarding_text("en", "run_selector_label"), "Run")
        self.assertEqual(onboarding_text("en", "preprocessing_backend"), "preprocessing_backend")  # W2a: the route choice is gone

    def test_known_status_messages_follow_selected_language(self) -> None:
        self.assertEqual(translated_ui_message("zh", "Environment ready"), "环境已就绪")
        self.assertEqual(translated_ui_message("en", "环境已就绪"), "Environment ready")
        self.assertEqual(translated_ui_message("zh", "Runtime detail: 3 tasks"), "Runtime detail: 3 tasks")
        self.assertEqual(onboarding_text("unknown", "subject_root"), "Subject root")

    def test_dynamic_status_templates_switch_in_both_directions(self) -> None:
        cases = (
            ("Partial results: 3 blocked task(s)", "部分结果：3 个任务被阻塞"),
            ("Pipeline failed: exit code 17", "Pipeline 失败：退出码 17"),
            ("Completed with 2 failed task(s)", "完成，但有 2 个任务失败"),
            ("Completed with 4 blocked task(s)", "完成，但有 4 个任务被阻塞"),
            ("Run finished. Manifest: D:/out/run.json", "运行完成。清单：D:/out/run.json"),
        )
        for english, chinese in cases:
            with self.subTest(english=english):
                self.assertEqual(translated_ui_message("zh", english), chinese)
                self.assertEqual(translated_ui_message("en", chinese), english)

    def test_result_buttons_use_short_labels_in_both_languages(self) -> None:
        owner = FakeLanguageOwner()
        owner._resultSceneButtons = [FakeTextWidget() for _ in range(4)]

        apply_onboarding_language(owner, "zh")
        self.assertEqual([button.value for button in owner._resultSceneButtons], ["加载", "替换", "追加", "清除"])
        self.assertEqual(onboarding_text("zh", "open_latest_report"), "打开报告")

        apply_onboarding_language(owner, "en")
        self.assertEqual([button.value for button in owner._resultSceneButtons], ["Load", "Replace", "Append", "Clear"])
        self.assertEqual(onboarding_text("en", "open_latest_report"), "Open report")

    def test_wma_input_path_shows_what_empty_means_in_both_languages(self) -> None:
        owner = FakeLanguageOwner()
        line_edit = FakeTextWidget()
        path_edit = FakeTextWidget()
        path_edit.findChildren = lambda _kind: [line_edit]
        owner.ui.wmaInputTractographyPath = path_edit

        with patch.dict(sys.modules, {"qt": MagicMock()}):  # findChildren(qt.QLineEdit) needs qt
            apply_onboarding_language(owner, "en")
            # short enough to show in full in the 430 px panel (the long form was cut off in Slicer)
            self.assertEqual(line_edit.placeholder, "Automatic (UKF tractogram)")
            apply_onboarding_language(owner, "zh")
            self.assertEqual(line_edit.placeholder, "自动（UKF 流线）")

    def test_prepare_and_environment_dynamic_controls_switch_language(self) -> None:
        owner = FakeLanguageOwner()
        prepare_label = FakeTextWidget()
        prepare_edit = FakeTextWidget()
        prepare_button = FakeTextWidget()
        register_ui_language_binding(owner, prepare_label, "text", "source_type")
        register_ui_language_binding(owner, prepare_edit, "placeholder", "source_placeholder")
        register_ui_language_binding(owner, prepare_edit, "tooltip", "source_placeholder")
        register_ui_language_binding(owner, prepare_button, "text", "browse")

        section_title = FakeTextWidget()
        section_description = FakeTextWidget()
        dependency_label = FakeTextWidget()
        dependency_edit = FakeTextWidget()
        dependency_browse = FakeTextWidget()
        dependency_clear = FakeTextWidget()
        dependency_status = FakeTextWidget()
        owner._environmentSectionLanguageWidgets = [
            ("core-runtime", section_title, section_description),
        ]
        owner._environmentPathLanguageWidgets = {
            "postprocess_freesurfer_home": {
                "label": dependency_label,
                "line_edit": dependency_edit,
                "browse": dependency_browse,
                "clear": dependency_clear,
            }
        }
        owner._environmentPathStatusLabels = {
            "postprocess_freesurfer_home": dependency_status,
        }
        owner._environmentPathStatusSources = {
            "postprocess_freesurfer_home": "check pending",
        }

        apply_onboarding_language(owner, "zh")

        self.assertEqual(prepare_label.value, "来源类型")
        self.assertIn("DICOM 文件夹", prepare_edit.placeholder)
        self.assertEqual(prepare_edit.tooltip, prepare_edit.placeholder)
        self.assertEqual(prepare_button.value, "浏览…")
        self.assertEqual(section_title.value, "核心 Python 与运行环境路径")
        self.assertIn("环境变量", section_description.value)
        self.assertEqual(dependency_label.value, "FreeSurfer 主目录（可选）")
        self.assertEqual(dependency_edit.placeholder, "留空时自动检测")
        self.assertIn("FreeSurfer 安装根目录", dependency_edit.tooltip)
        self.assertEqual(dependency_browse.value, "浏览…")
        self.assertEqual(dependency_clear.value, "清空")
        self.assertEqual(dependency_status.value, "等待检查")

        apply_onboarding_language(owner, "en")

        self.assertEqual(prepare_label.value, "Source type")
        self.assertIn("DICOM folder", prepare_edit.placeholder)
        self.assertEqual(section_title.value, "Core Python & Runtime Paths")
        self.assertEqual(dependency_label.value, "FreeSurfer home (optional)")
        self.assertEqual(dependency_edit.placeholder, "Auto-detect if empty")
        self.assertEqual(dependency_status.value, "check pending")
        self.assertEqual(environment_section_text("zh", "wma")[0], "WMA 路径")
        self.assertEqual(dependency_ui_text("zh", "mrtrix_bin_dir", "label"), "MRtrix3 bin 目录")

    def test_path_examples_match_the_selected_platform(self) -> None:
        windows = platform_path_examples("en", system="Windows")
        macos = platform_path_examples("en", system="Darwin")
        linux = platform_path_examples("zh", system="Linux")

        self.assertIn(r"C:\Users\name\data\sub-01", windows)
        self.assertIn("/Users/name/data/sub-01", macos)
        self.assertIn("/opt/homebrew/bin", macos)
        self.assertIn("本机路径示例", linux)
        self.assertIn("/home/name/data/sub-01", linux)

    def test_capability_dashboard_explains_runnable_and_blocked_targets(self) -> None:
        result = FakeResult(
            runnable_targets=("validation", "microstructure"),
            blocked_targets=("tractography",),
            statuses=(FakeItem("command:tckgen", "tckgen", "missing", ("tractography",)),),
        )

        english = capability_dashboard_text(result, language="en")
        chinese = capability_dashboard_text(result, language="zh")

        self.assertIn("Runnable now: validation, microstructure", english)
        self.assertIn("tractography: tckgen", english)
        self.assertIn("你现在可以运行：validation, microstructure", chinese)
        self.assertIn("tractography: tckgen", chinese)

    def test_diagnostic_report_is_json(self) -> None:
        report = json.loads(diagnostic_report_text(FakeOwner()))

        self.assertEqual(report["external_paths"]["model_assets_dir"], "/tmp/models")
        self.assertIn("python", report)

    def test_create_synthetic_demo_subject(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subject_root = create_synthetic_demo_subject(Path(tmp))

            self.assertTrue((subject_root / "input" / "data.nii.gz").is_file())
            self.assertTrue((subject_root / "input" / "bvals").is_file())
            self.assertTrue((subject_root / "input" / "bvecs").is_file())
            self.assertTrue((subject_root / "input" / "nodif_brain_mask.nii.gz").is_file())

    def test_prepare_demo_resets_every_non_demo_target(self) -> None:
        class DemoOwner:
            def __init__(self) -> None:
                self._parameterNode = types.SimpleNamespace(
                    targetPreprocessing=True,
                    targetSurface=True,
                    targetSurfacePostprocess=True,
                    targetParcellation=True,
                    targetSegmentation=True,
                    targetMicrostructure=False,
                    targetTractography=False,
                    targetBundles=True,
                    targetConnectome=True,
                )
                self.checked = False
                self.logs: list[str] = []

            def _checkCanApply(self) -> None:
                self.checked = True

            def appendLog(self, message: str) -> None:
                self.logs.append(message)

        owner = DemoOwner()
        fake_slicer = types.SimpleNamespace(
            app=types.SimpleNamespace(temporaryPath="/tmp"),
            util=types.SimpleNamespace(NodeModify=lambda _node: nullcontext()),
        )
        subject_root = Path("/tmp/SlicerDSpace/demo/slicerdsurfer-demo")
        with (
            patch.dict("sys.modules", {"slicer": fake_slicer}),
            patch("shared.ui_onboarding.create_synthetic_demo_subject", return_value=subject_root),
        ):
            returned = prepare_demo_for_owner(owner)

        node = owner._parameterNode
        self.assertEqual(returned, subject_root)
        self.assertFalse(node.targetPreprocessing)
        self.assertFalse(node.targetSurface)
        self.assertFalse(node.targetSurfacePostprocess)
        self.assertFalse(node.targetParcellation)
        self.assertFalse(node.targetSegmentation)
        self.assertTrue(node.targetMicrostructure)
        self.assertTrue(node.targetTractography)
        self.assertFalse(node.targetBundles)
        self.assertFalse(node.targetConnectome)
        self.assertEqual(node.orchestrationTractographyBackend, "dti_det")
        self.assertEqual(node.orchestrationMicrostructureStages, "dti")
        self.assertEqual(node.orchestrationRunName, "Demo")
        self.assertEqual(node.orchestrationSelectStreamlines, 512)
        self.assertTrue(owner.checked)


if __name__ == "__main__":
    unittest.main()
