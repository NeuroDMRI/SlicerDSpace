#!/usr/bin/env python3
"""Tests for pure environment-panel helpers."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from shared.ui_environment_panel import (
    dependency_backend_group,
    dependency_items_for_section,
    dependency_profile_payload,
    dependency_section_metadata,
    dependency_subsection_key,
    dependency_subsection_metadata,
    environment_detail_lines,
    environment_dialog_state,
    environment_item_button_text,
    environment_item_button_tooltip,
    environment_item_text,
    environment_missing_items,
    environment_path_entries,
    environment_path_status_from_dependencies,
    environment_path_status_style,
    environment_summary_text,
    external_dependencies_from_profile,
    format_detected_source,
    model_assets_summary_text,
    model_assets_provenance_text,
    optional_backend_label_styles,
    optional_backend_widget_style,
    path_status_dependencies,
    python_install_target_text,
)
from shared.environment import DependencyStatus, EnvironmentDoctorResult, TargetEnvironmentStatus
from shared.dependency_config import UI_DEPENDENCY_PATHS


class UIEnvironmentPanelTest(unittest.TestCase):
    def _sample_result(self) -> EnvironmentDoctorResult:
        return EnvironmentDoctorResult(
            platform="Linux",
            host_platform="Linux",
            python_executable="/opt/Slicer/bin/PythonSlicer",
            targets=("tractography", "bundles", "segmentation"),
            statuses=(
                DependencyStatus(
                    key="python:tract_querier",
                    kind="python",
                    label="tract_querier",
                    status="missing",
                    required=True,
                    targets=("bundles",),
                    details={
                        "package": "tract_querier",
                        "python_executable": "/opt/Slicer/bin/PythonSlicer",
                        "install_requirement": "tract_querier",
                    },
                    install_hint="Install in Slicer Python.",
                ),
                DependencyStatus(
                    key="command:dwi2fod",
                    kind="command",
                    label="dwi2fod",
                    status="missing",
                    required=True,
                    targets=("tractography",),
                    details={
                        "command": "dwi2fod",
                        "checked_locations": ["/opt/mrtrix3/bin/dwi2fod", "/usr/local/bin/dwi2fod"],
                    },
                    install_hint="Install MRtrix3.",
                    install_group="mrtrix",
                ),
                DependencyStatus(
                    key="path:wma_atlas",
                    kind="path",
                    label="WMA atlas",
                    status="missing",
                    required=False,
                    targets=("bundles",),
                    details={"path": "/data/atlas", "path_kind": "directory"},
                    install_hint="Download ORG atlas.",
                    install_group="wma",
                ),
                DependencyStatus(
                    key="path:ddseg_weights",
                    kind="path",
                    label="DDSeg weights directory",
                    status="missing",
                    required=True,
                    targets=("segmentation",),
                    details={"path": "/models/segmentation/weights", "path_kind": "nonempty_dir"},
                    install_hint="Choose a model asset root.",
                    install_group="model-resources",
                ),
                DependencyStatus(
                    key="python:numpy",
                    kind="python",
                    label="numpy",
                    status="available",
                    required=True,
                    targets=("validation",),
                ),
            ),
            target_statuses=(
                TargetEnvironmentStatus("tractography", "blocked", ("command:dwi2fod",), ()),
                TargetEnvironmentStatus("bundles", "blocked", ("python:tract_querier",), ("path:wma_atlas",)),
                TargetEnvironmentStatus("segmentation", "blocked", ("path:ddseg_weights",), ()),
            ),
            blocker_count=3,
            missing_python_packages=("tract_querier",),
            missing_external_groups=("mrtrix", "wma"),
        )

    def test_python_install_target_text_names_external_tool_policy(self) -> None:
        text = python_install_target_text("/opt/Slicer/bin/PythonSlicer")

        self.assertIn("/opt/Slicer/bin/PythonSlicer", text)
        self.assertIn("External tools are never installed silently", text)

    def test_model_assets_provenance_text_names_terms_and_lockfile(self) -> None:
        text = model_assets_provenance_text(Path("/repo"))

        self.assertIn("/repo/models/assets_manifest.json", text)
        self.assertIn("/repo/docs/assets/MODEL_ASSET_TERMS.md", text)
        self.assertIn("SLICERDSPACE_MODEL_ASSETS_DIR", text)
        self.assertIn("--write-lockfile", text)
        self.assertIn("DOI-archive rights", text)

    def test_dependency_schema_grouping_drives_environment_panel_sections(self) -> None:
        core = dependency_items_for_section(UI_DEPENDENCY_PATHS, "core-runtime")
        optional = dependency_items_for_section(UI_DEPENDENCY_PATHS, "bundle-backends")

        self.assertIn("Python", dependency_section_metadata("core-runtime")["title"])
        self.assertIn("Optional", dependency_section_metadata("bundle-backends")["title"])
        self.assertEqual(core[0]["id"], "python_executable")
        self.assertEqual(optional[0]["id"], "wma_root")
        self.assertEqual(dependency_subsection_key("wma_root"), "wma")
        self.assertEqual(dependency_subsection_key("tractcloud_root"), "tractcloud")
        self.assertEqual(dependency_subsection_metadata("wma")["title"], "WMA Paths")
        self.assertEqual(dependency_backend_group("slicer_executable"), "wma")
        self.assertEqual(dependency_backend_group("tractcloud_data_dir"), "tractcloud")
        self.assertIsNone(dependency_backend_group("mrtrix_bin_dir"))

    def test_environment_path_entries_preserve_panel_order(self) -> None:
        entries = environment_path_entries(UI_DEPENDENCY_PATHS)
        kinds = [entry.kind for entry in entries]
        row_ids = [entry.item["id"] for entry in entries if entry.kind == "row" and entry.item]
        model_index = next(index for index, entry in enumerate(entries) if entry.kind == "model-assets")
        wma_index = next(
            index
            for index, entry in enumerate(entries)
            if entry.kind == "row" and entry.item and entry.item["id"] == "wma_root"
        )

        self.assertEqual(entries[0].kind, "section")
        self.assertEqual(entries[0].title, "Model Asset Paths")
        self.assertIn("model-assets", kinds)
        self.assertLess(model_index, wma_index)
        self.assertEqual(row_ids[0], "model_assets_dir")
        self.assertIn("python_executable", row_ids)
        self.assertIn("wma_root", row_ids)
        self.assertIn("tractcloud_root", row_ids)
        self.assertEqual(
            [(entry.subsection, entry.title) for entry in entries if entry.kind == "subsection"],
            [("wma", "WMA Paths"), ("tractcloud", "TractCloud Paths")],
        )

    def test_dependency_profile_payload_filters_empty_values(self) -> None:
        payload = dependency_profile_payload(
            {"mrtrix_bin_dir": "/opt/mrtrix3/bin", "wma_root": ""},
            python_executable="/slicer/PythonSlicer",
            platform_text="Linux-test",
            created_at="2026-06-23T00:00:00Z",
        )

        self.assertEqual(payload["kind"], "slicerdsurfer_dependency_profile")
        self.assertEqual(payload["version"], 2)
        self.assertEqual(payload["schema"], "slicerddsurfer.dependency_profile.v2")
        self.assertEqual(payload["platform"], "Linux-test")
        self.assertEqual(payload["python_install_target"], "/slicer/PythonSlicer")
        self.assertEqual(payload["external_dependencies"], {"mrtrix_bin_dir": "/opt/mrtrix3/bin"})

    def test_external_dependencies_from_profile_accepts_nested_or_flat_payload(self) -> None:
        nested = {"external_dependencies": {"mrtrix_bin_dir": "/opt/mrtrix3/bin", "empty": ""}}
        flat = {"mrtrix_bin_dir": "/opt/mrtrix3/bin"}

        self.assertEqual(external_dependencies_from_profile(nested), {"mrtrix_bin_dir": "/opt/mrtrix3/bin"})
        self.assertEqual(external_dependencies_from_profile(flat), {"mrtrix_bin_dir": "/opt/mrtrix3/bin"})

    def test_external_dependencies_from_profile_rejects_invalid_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "external_dependencies"):
            external_dependencies_from_profile({"external_dependencies": []})

    def test_environment_summary_and_details_prioritize_required_items(self) -> None:
        result = self._sample_result()

        missing = environment_missing_items(result)
        summary = environment_summary_text(result)
        details = environment_detail_lines(result)

        self.assertEqual(
            [item.label for item in missing],
            ["dwi2fod", "DDSeg weights directory", "tract_querier", "WMA atlas"],
        )
        self.assertIn("0 runnable, 3 need setup, 3 required missing, 1 optional missing", summary)
        self.assertIn("Python packages available for auto-install: 1", summary)
        self.assertIn("Needs attention: dwi2fod, DDSeg weights directory, tract_querier, +1 more", summary)
        self.assertIn("Checked targets: tractography, bundles, segmentation", details)
        self.assertIn("Blocked targets: tractography, bundles, segmentation", details)
        self.assertIn("Auto-installable Python packages: tract_querier", details)
        self.assertIn("Manual setup groups: mrtrix, wma", details)

    def test_environment_item_text_includes_actionable_context(self) -> None:
        result = self._sample_result()
        command_item = next(item for item in result.statuses if item.key == "command:dwi2fod")
        python_item = next(item for item in result.statuses if item.key == "python:tract_querier")
        path_item = next(item for item in result.statuses if item.key == "path:wma_atlas")

        command_text = environment_item_text(command_item)
        python_text = environment_item_text(python_item)
        path_text = environment_item_text(path_item)

        self.assertIn("dwi2fod (required)", command_text)
        self.assertIn("Command: dwi2fod", command_text)
        self.assertIn("Checked locations: /opt/mrtrix3/bin/dwi2fod", command_text)
        self.assertIn("Package: tract_querier", python_text)
        self.assertIn("Python: /opt/Slicer/bin/PythonSlicer", python_text)
        self.assertIn("Path: /data/atlas", path_text)
        self.assertIn("Expected type: directory", path_text)

    def test_environment_dialog_state_collects_render_decisions(self) -> None:
        result = self._sample_result()
        state = environment_dialog_state(result, python_install_target="Python target.")
        python_item = next(item for item in result.statuses if item.key == "python:tract_querier")
        command_item = next(item for item in result.statuses if item.key == "command:dwi2fod")
        model_item = next(item for item in result.statuses if item.key == "path:ddseg_weights")

        self.assertIn("required missing", state.summary_text)
        self.assertIn("Checked targets: tractography, bundles, segmentation", state.detail_text)
        self.assertEqual(
            [item.label for item in state.missing_items],
            ["dwi2fod", "DDSeg weights directory", "tract_querier", "WMA atlas"],
        )
        self.assertTrue(state.can_install_python)
        self.assertTrue(state.has_manual_setup_notes)
        self.assertIn("Python target.", state.python_install_tooltip)
        self.assertEqual(environment_item_button_text(python_item), "Install")
        from shared.managed_runtimes import supported_platform
        self.assertEqual(environment_item_button_text(command_item), "Install and configure" if supported_platform() else "Choose path / setup")
        self.assertEqual(environment_item_button_text(model_item), "Download / choose path")
        self.assertEqual(environment_item_button_tooltip(command_item), "Install MRtrix3.")

    def test_environment_path_status_maps_dependencies_to_label_text(self) -> None:
        result = self._sample_result()

        mrtrix_dependencies = path_status_dependencies(result, "mrtrix_bin_dir")
        text, state = environment_path_status_from_dependencies(mrtrix_dependencies)

        self.assertEqual([item.key for item in mrtrix_dependencies], ["command:dwi2fod"])
        self.assertEqual((text, state), ("not found", "missing"))
        self.assertEqual(environment_path_status_from_dependencies([]), ("not used by current selection", "inactive"))
        self.assertIn("#C62828", environment_path_status_style("missing"))
        self.assertIn("#78909C", environment_path_status_style("unknown-state"))

    def test_non_installable_slicer_runtime_is_shown_as_setup(self) -> None:
        runtime = DependencyStatus(
            key="python:slicer.util",
            kind="python",
            label="slicer.util",
            status="missing",
            required=True,
            targets=("microstructure",),
            details={
                "import": "slicer.util",
                "package": "",
                "python_executable": "/usr/bin/python3",
            },
            install_hint="Run with PythonSlicer; do not pip install slicer.",
            install_group="slicer-runtime",
        )

        self.assertEqual(environment_item_button_text(runtime), "Setup")
        self.assertIn("Runtime module: slicer.util", environment_item_text(runtime))
        self.assertIn("do not pip install", environment_item_button_tooltip(runtime))

    def test_detected_source_text_is_compact(self) -> None:
        self.assertEqual(format_detected_source(""), "not found")
        self.assertEqual(format_detected_source("user setting"), "using saved path")
        self.assertEqual(format_detected_source("env:MRTRIX_BIN_DIR"), "detected from env:MRTRIX_BIN_DIR")
        self.assertEqual(format_detected_source("PATH"), "detected from PATH")
        self.assertEqual(format_detected_source("common path"), "detected from common path")

    def test_optional_backend_styles_are_schema_helpers(self) -> None:
        active_title, active_description = optional_backend_label_styles(True)
        inactive_title, inactive_description = optional_backend_label_styles(False)

        self.assertNotIn("border-left", active_title)
        self.assertNotEqual(active_description, inactive_description)
        self.assertIn("QLabel", inactive_title)
        self.assertIn("QPushButton", optional_backend_widget_style("QPushButton", True))
        self.assertEqual(optional_backend_widget_style("QPushButton", False), "")
        self.assertIsNone(optional_backend_widget_style("QComboBox", True))

    def test_model_assets_summary_uses_manifest_presence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            missing_text = model_assets_summary_text(repo_root)
            (repo_root / "models").mkdir()
            (repo_root / "models" / "assets_manifest.json").write_text("{}", encoding="utf-8")
            present_text = model_assets_summary_text(repo_root)

        self.assertIn("Unified source expected", missing_text)
        self.assertIn("Default root:", present_text)
        self.assertIn("SLICERDSPACE_MODEL_ASSETS_DIR", present_text)

    def test_environment_item_display_and_summary_parts_format_cleanly(self) -> None:
        from shared.ui_environment_panel import environment_item_display, environment_summary_parts

        result = self._sample_result()
        severity, headline, subtitle = environment_summary_parts(result, zh=False)
        self.assertEqual(severity, "blocked")
        self.assertIn("3 required items", headline)
        self.assertIn("0 runnable", subtitle)
        self.assertIn("1 optional missing", subtitle)

        command_item = next(item for item in result.statuses if item.key == "command:dwi2fod")
        disp = environment_item_display(command_item, zh=False)
        self.assertEqual(disp.title, "Command: dwi2fod")
        self.assertEqual(disp.badge_text, "Required")
        self.assertEqual(disp.badge_severity, "required")
        self.assertIn("Tractography", disp.meta_text)
        self.assertIn("Install MRtrix3.", disp.guidance_text)
        self.assertIn("Key: command:dwi2fod", disp.technical_details)


if __name__ == "__main__":
    unittest.main()
