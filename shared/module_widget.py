"""Common scripted-module widget glue for SlicerDSpace."""

from __future__ import annotations

import importlib.util
import os
import json
import platform
import sys
from pathlib import Path
from datetime import datetime
from time import monotonic
from typing import Any, List, Optional

import qt
import slicer
import vtk
from qt import QPixmap, QSignalBlocker

from shared.dependency_config import UI_DEPENDENCY_PATHS
from shared.docs import doc_path
from shared.input_import import InputImportConfig, detect_import_source_kind, prepare_inputs
from shared.model_assets import model_asset_path_or_none
from shared.ui_error_dialogs import show_actionable_error_dialog
from shared.ui_compact import close_input_format_help_popovers, install_compact_layout
from shared.ui_environment_dialog import show_environment_results_dialog
from shared.ui_environment_paths import (
    add_environment_path_row,
    add_environment_section_header,
    add_environment_subsection_label,
    add_model_assets_row,
    install_environment_dependency_path_controls,
)
from shared.ui_environment_panel import (
    dependency_profile_payload,
    environment_detail_lines,
    environment_item_text,
    environment_missing_items,
    environment_path_status_from_dependencies,
    environment_path_status_style,
    environment_summary_text,
    external_dependencies_from_profile,
    format_detected_source,
    model_assets_provenance_text,
    optional_backend_label_styles,
    optional_backend_widget_style,
    path_status_dependencies,
    python_install_target_text,
)
from shared.ui_prepare_inputs import install_prepare_inputs_panel
from shared.run_history import manifest_path_for_plan
from shared.topup_parameters import (
    PHASE_ENCODING_CHOICES,
    TopupValidation,
    route_uses_topup,
    topup_summary_text,
    validate_topup_parameters,
)
from shared.ui_run_history import (
    browse_run_directory,
    on_run_combo_changed,
    poll_run_progress,
    refresh_run_history,
    set_run_state,
    start_progress_polling,
    stop_progress_polling,
    validated_selected_manifest,
)
from shared.subject_id import effective_subject_id
from shared.ui_run_state import gui_brain_mask_required, run_button_state, run_state_from_manifest_status
from shared.ui_output_loading import qbytearray_to_text
from shared.ui_input_inspection import inspect_input_directory
from shared.ui_parameter_sync import sync_parameter_node_to_ui
from shared.brain_mask import REMOVED_BRAIN_MASK_BACKENDS
from shared.ui_workflow_text import BRAIN_MASK_BACKEND_TEXT_KEYS, PNL_MASK_ROUTE_TEXT_KEYS
from shared.ui_run_controller import (
    BackendSelection,
    ProductTargetFlags,
    RunSelection,
    active_bundle_backend_sections,
    build_background_cli_args,
    build_batch_run_kwargs,
    build_batch_background_cli_args,
    build_single_subject_run_kwargs,
    cancelled_manifest_path_and_payload,
    classify_background_run_outcome,
    default_run_name_for_backend,
    extract_background_manifest_path,
    is_auto_generated_run_name,
    compress_mrtrix_dense_value,
    microstructure_dti_enabled,
    microstructure_ifw_enabled,
    microstructure_stage_flags,
    microstructure_optional_stage_flags,
    parse_ifw_echo_time_ms,
    parse_manifest_line,
    blocked_plan_items,
    blocked_plan_log_text,
    environment_blockers_from_plan,
    partial_run_notice_text,
    preflight_blockers_for_targets,
    runnable_targets_from_plan,
    rows_from_batch_manifest,
    rows_from_plan,
    rows_from_result,
    rows_from_single_subject_manifest,
    run_summary_from_manifest,
    selected_closure,
    selected_environment_targets,
    selected_orchestrator_targets,
    selected_product_targets,
    selected_products_summary,
    selected_workflows,
    streamline_control_state,
    task_keys_for_running_line,
    validate_batch_run_options,
    validate_single_subject_inputs,
)
from shared.ui_settings import coerce_user_setting_value, common_user_setting_specs, consume_user_defaults_request
from shared.ui_status_log import (
    append_log,
    apply_widget_styles,
    configure_log_widget,
    install_theme_change_listener,
    set_existing_report_path,
    set_run_progress_message,
    set_run_summary_message,
    set_status,
    uninstall_theme_change_listener,
)
from shared.ui_background_process import (
    background_process_env_overrides,
    build_background_run_context,
    qprocess_is_running,
)
from shared.process_group import capture_process_group_id, finish_process_group, grouped_launch_command, kill_process_tree
from shared.ui_run_status_panel import (
    install_run_status_panel,
    mark_unfinished_rows,
    row_open_path,
    set_run_status_table_rows,
    table_item,
    update_run_status_table_item,
)
from shared.ui_onboarding import (
    LANGUAGE_SETTING,
    apply_onboarding_language,
    capability_dashboard_text,
    dependency_ui_text,
    diagnostic_report_text,
    install_onboarding_panel,
    normalized_language,
    onboarding_text,
    prepare_demo_for_owner,
    show_first_run_wizard,
    translated_ui_message,
)
from shared.user_config import (
    canonical_microstructure_stages,
    default_user_config_path,
    load_user_config,
    selected_microstructure_stages,
    update_user_config,
)
from shared.ui_environment_check import start_environment_check


ENVIRONMENT_DEPENDENCY_PATHS = UI_DEPENDENCY_PATHS


class SharedWidgetMixin:
    """Shared UI helpers used across the unified SlicerDSpace widget."""

    def _connect_common_ui(self) -> None:
        slicer.app.connect("aboutToQuit()", self.cleanup)
        if hasattr(self.ui, "runSelectedButton"):
            self.ui.runSelectedButton.connect("clicked()", self._onRunSelectedClicked)
        if hasattr(self.ui, "validateButton"):
            self.ui.validateButton.connect("clicked()", self._onValidateClicked)
        if hasattr(self.ui, "clearLogButton"):
            self.ui.clearLogButton.connect("clicked()", self._onClearLogClicked)
        if hasattr(self.ui, "unifiedTorchDeviceComboBox"):
            self.ui.unifiedTorchDeviceComboBox.currentIndexChanged.connect(self._onUnifiedTorchDeviceChanged)
        if hasattr(self.ui, "orchestrationTractographyBackendComboBox"):
            self.ui.orchestrationTractographyBackendComboBox.currentTextChanged.connect(
                self._onTractographyBackendChanged
            )
        if hasattr(self.ui, "orchestrationBundleBackendComboBox"):
            self.ui.orchestrationBundleBackendComboBox.currentTextChanged.connect(
                self._onBundleBackendChanged
            )
        if hasattr(self.ui, "batchRootPath"):
            try:
                self.ui.batchRootPath.currentPathChanged.connect(self._onBatchRootPathChanged)
            except Exception:
                self.ui.batchRootPath.connect("currentPathChanged(QString)", self._onBatchRootPathChanged)
        if hasattr(self.ui, "wmaInputTractographyPath"):
            try:
                self.ui.wmaInputTractographyPath.currentPathChanged.connect(
                    self._onWmaInputTractographyPathChanged
                )
            except Exception:
                self.ui.wmaInputTractographyPath.connect(
                    "currentPathChanged(QString)", self._onWmaInputTractographyPathChanged
                )
        self._connect_brain_mask_ui()
        self._connect_microstructure_stage_ui()
        self._connect_topup_ui()

    def _connect_microstructure_stage_ui(self) -> None:
        for name in ("orchestrationFreeWaterCheckBox", "orchestrationNoddiCheckBox", "orchestrationDkiCheckBox",
                     "orchestrationIfwCheckBox"):
            checkbox = getattr(self.ui, name, None)
            if checkbox is not None:
                checkbox.toggled.connect(self._onOptionalMicrostructureStageChanged)
        # T-C2: the T2 values are fixed in code (Cho et al.); the GUI only shows them.
        from microstructure.ifw import IFW_T2_CSF_MS, IFW_T2_GM_MS
        for name, value in (("orchestrationIfwT2TissueLineEdit", IFW_T2_GM_MS),
                            ("orchestrationIfwT2CsfLineEdit", IFW_T2_CSF_MS)):
            field = getattr(self.ui, name, None)
            if field is not None:
                field.setText(f"{value:g}")
                field.setReadOnly(True)
        echo_time = getattr(self.ui, "orchestrationIfwEchoTimeLineEdit", None)
        if echo_time is not None:
            echo_time.textChanged.connect(self._onIfwEchoTimeChanged)

    def _onIfwEchoTimeChanged(self, text: str) -> None:
        self._update_parameter_node("orchestrationIfwEchoTimeMs", str(text or "").strip())
        self._schedule_common_user_settings_save()

    def _onOptionalMicrostructureStageChanged(self, _checked: bool) -> None:
        node = self._parameterNode
        if not node:
            return
        current_selected = set(selected_microstructure_stages(node.orchestrationMicrostructureStages or "dti"))
        stages = ["dti"] if "dti" in current_selected else []
        for name, stage in (
            ("orchestrationFreeWaterCheckBox", "freewater"),
            ("orchestrationNoddiCheckBox", "noddi"),
            ("orchestrationDkiCheckBox", "dki"),
            ("orchestrationIfwCheckBox", "ifw"),
        ):
            checkbox = getattr(self.ui, name, None)
            if checkbox is not None and checkbox.isChecked():
                stages.append(stage)
        if not stages:
            stages = ["dti"]
        with slicer.util.NodeModify(node):
            node.orchestrationMicrostructureStages = ",".join(stages)
            node.orchestrationRunFreeWater = "freewater" in stages
            node.orchestrationRunNoddi = "noddi" in stages
            node.orchestrationRunDki = "dki" in stages
            node.orchestrationRunIfw = "ifw" in stages

    def _connect_brain_mask_ui(self) -> None:
        """Keep readable masking labels separate from canonical backend values."""

        route_combo = getattr(self.ui, "pnlMaskRouteComboBox", None)
        if route_combo is not None:
            # SC-1b: the route is an advanced choice with its own label; "native (experimental)" is only a label.
            blocker = QSignalBlocker(route_combo)
            route_combo.clear()
            for route, key in PNL_MASK_ROUTE_TEXT_KEYS.items():
                route_combo.addItem(onboarding_text(getattr(self, "_onboardingLanguage", "en"), key), route)
            del blocker
            route_combo.currentIndexChanged.connect(self._onPnlMaskRouteChanged)
        combo = getattr(self.ui, "brainMaskBackendComboBox", None)
        if combo is None:
            return
        blocker = QSignalBlocker(combo)
        combo.clear()
        for backend, key in BRAIN_MASK_BACKEND_TEXT_KEYS.items():
            combo.addItem(onboarding_text(getattr(self, "_onboardingLanguage", "en"), key), backend)
        combo.setSizeAdjustPolicy(qt.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(8)
        del blocker
        combo.currentIndexChanged.connect(self._onBrainMaskBackendChanged)
        self._update_pnl_route_visibility("pnl_cnn")

    def _update_pnl_route_visibility(self, backend: str) -> None:
        """SC-1b: the PNL route choice is shown only for the pnl_cnn backend."""
        target_visible = backend == "pnl_cnn"
        changed = False
        for name in ("pnlMaskRouteLabel", "pnlMaskRouteComboBox"):
            widget = getattr(self.ui, name, None)
            if widget is not None and widget.visible != target_visible:
                widget.setVisible(target_visible)
                changed = True
        if changed and hasattr(self, "_fitAdvancedPage"):
            self._fitAdvancedPage()

    def _onPnlMaskRouteChanged(self, index: int) -> None:
        combo = getattr(self.ui, "pnlMaskRouteComboBox", None)
        if combo is None or index < 0:
            return
        route = str(combo.itemData(index) or "")
        if route not in PNL_MASK_ROUTE_TEXT_KEYS:
            return
        self._lastEnvironmentResult = None
        self._lastEnvironmentSignature = None
        self._update_parameter_node("pnlMaskRoute", route)

    def _onBrainMaskBackendChanged(self, index: int) -> None:
        combo = getattr(self.ui, "brainMaskBackendComboBox", None)
        if combo is None or index < 0:
            return
        backend = str(combo.itemData(index) or "")
        if backend not in BRAIN_MASK_BACKEND_TEXT_KEYS:
            return
        self._lastEnvironmentResult = None
        self._lastEnvironmentSignature = None
        self._update_pnl_route_visibility(backend)
        self._update_parameter_node("brainMaskBackend", backend)

    def _connect_topup_ui(self) -> None:
        """Bind the distortion-correction widgets to the parameter node by hand.

        The phase-encoding combo stores canonical values as item data while
        showing readable labels, so it is synchronized explicitly instead of
        through SlicerParameterName.
        """

        combo = getattr(self.ui, "preprocessingPhaseEncodingDirectionComboBox", None)
        if combo is not None and combo.count == 0:
            for value, label in PHASE_ENCODING_CHOICES:
                combo.addItem(label, value)
            # Do not let the longest item label dictate the panel's minimum width.
            combo.setSizeAdjustPolicy(qt.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(8)
            combo.currentIndexChanged.connect(self._onPhaseEncodingDirectionChanged)
        spin = getattr(self.ui, "preprocessingReadoutTimeSpinBox", None)
        if spin is not None:
            spin.valueChanged.connect(self._onReadoutTimeChanged)
        path = getattr(self.ui, "preprocessingReversePhaseB0Path", None)
        if path is not None:
            try:
                path.currentPathChanged.connect(self._onReversePhaseB0Changed)
            except Exception:
                path.connect("currentPathChanged(QString)", self._onReversePhaseB0Changed)

    def _onPhaseEncodingDirectionChanged(self, index: int) -> None:
        combo = getattr(self.ui, "preprocessingPhaseEncodingDirectionComboBox", None)
        value = str(combo.itemData(index) or "") if combo is not None and index >= 0 else ""
        self._update_parameter_node("preprocessingPhaseEncodingDirection", value)

    def _onReadoutTimeChanged(self, value: float) -> None:
        self._update_parameter_node("preprocessingReadoutTime", float(value or 0.0))

    def _onReversePhaseB0Changed(self, value: str) -> None:
        self._update_parameter_node("preprocessingReversePhaseB0", str(value or "").strip())

    def _sync_combo_data(self, widget, desired: str) -> None:
        if not widget:
            return
        index = widget.findData(desired)
        if index < 0 or widget.currentIndex == index:
            return
        blocker = QSignalBlocker(widget)
        widget.setCurrentIndex(index)
        del blocker

    def _sync_double_spinbox(self, widget, desired: float) -> None:
        if not widget or abs(float(widget.value) - float(desired)) < 1e-12:
            return
        blocker = QSignalBlocker(widget)
        widget.value = float(desired)
        del blocker

    def _topup_validation(self, *, batch_mode: Optional[bool] = None) -> TopupValidation:
        node = self._parameterNode
        if not node:
            return validate_topup_parameters(preprocessing_selected=False)
        return validate_topup_parameters(
            preprocessing_selected=bool(node.targetPreprocessing),
            preprocessing_stages=getattr(node, "orchestrationPreprocessingStages", "") or "",
            phase_encoding_direction=getattr(node, "preprocessingPhaseEncodingDirection", "") or "",
            readout_time=float(getattr(node, "preprocessingReadoutTime", 0.0) or 0.0),
            reverse_phase_b0=getattr(node, "preprocessingReversePhaseB0", "") or "",
            batch_mode=bool(getattr(node, "batchMode", False)) if batch_mode is None else batch_mode,
            # W2b rule 4: the sidecar of the chosen subject decides whether a direction exists.
            subject_root=(getattr(node, "inputDirectory", "") or "").strip() or None,
            output_root=(getattr(node, "outputDirectory", "") or "").strip() or None,
        )

    def _refresh_topup_controls(self) -> TopupValidation:
        """Show the topup group only for the MRtrix3/FSL route and flag bad values."""

        node = self._parameterNode
        validation = self._topup_validation()
        preprocessing_on = bool(node.targetPreprocessing) if node else False
        preprocessing_stages = (getattr(node, "orchestrationPreprocessingStages", "") or "") if node else ""
        group = getattr(self.ui, "topupCollapsibleGroupBox", None)
        if group is not None:
            group.visible = bool(node) and route_uses_topup(preprocessing_on, preprocessing_stages=preprocessing_stages)
        hint = getattr(self.ui, "topupHintLabel", None)
        if hint is not None:
            full_text = topup_summary_text(validation)
            if validation.errors or validation.warnings:
                text = full_text
            else:
                zh = str(getattr(self, "_onboardingLanguage", "en") or "en").startswith("zh")
                text = (
                    "默认从 input/dwi.json 读取参数；提供反向相位 b=0 图像时启用 topup。"
                    if zh
                    else "Leave on 'From sidecar' to read input/dwi.json; set Reverse-PE b=0 to enable topup."
                )
            if hint.text != text:
                hint.setText(text)
            if hasattr(hint, "setToolTip") and hint.toolTip != full_text:
                hint.setToolTip(full_text)
            from shared.ui_styles import is_dark_mode, set_stylesheet_if_changed

            dark = is_dark_mode(hint)
            if dark:
                color = "#FF7B72" if validation.errors else ("#FFB74D" if validation.warnings else "#B8C5D0")
            else:
                color = "#C62828" if validation.errors else ("#E65100" if validation.warnings else "#607D8B")
            set_stylesheet_if_changed(hint, f"QLabel#topupHintLabel {{ color: {color}; }}")
        errors = " ".join(validation.errors).lower()
        self._setWidgetError(getattr(self.ui, "preprocessingPhaseEncodingDirectionComboBox", None), "phase-encoding" in errors)
        self._setWidgetError(getattr(self.ui, "preprocessingReadoutTimeSpinBox", None), "readout" in errors)
        self._setWidgetError(getattr(self.ui, "preprocessingReversePhaseB0Path", None), "reverse-pe" in errors)
        return validation

    def _refresh_run_button_state(self, validation: Optional[TopupValidation] = None) -> None:
        button = getattr(self.ui, "runSelectedButton", None)
        node = self._parameterNode
        if button is None:
            return
        if not node:
            button.enabled = False
            return
        validation = validation or self._topup_validation()
        self._refresh_subject_id_placeholder()
        cohort_experiments = bool(getattr(node, 'experimentsEnabled', False) and
                                  getattr(node, 'experimentScope', 'subject') == 'cohort')
        state = run_button_state(
            pipeline_active=bool(getattr(self, "_pipelineActive", False)),
            selected_targets=self._selected_product_targets(),
            batch_mode=cohort_experiments or bool(getattr(node, "batchMode", False)),
            subject_id=node.subjectId or "",
            subject_root_text=node.inputDirectory or "",
            batch_root_text=(getattr(node, 'experimentRoot', '') if cohort_experiments else getattr(node, "batchRoot", "")) or "",
            topup_errors=validation.errors,
        )
        has_source = bool(self._prepare_inputs_widget_text("source"))
        can_import = has_source and bool(self._selected_product_targets()) and not self._pipelineActive
        button.enabled = (state.enabled or can_import) and not getattr(self, "_inputImportPending", False)
        button.setToolTip(state.tooltip)

    def _schedule_run_history_refresh(self) -> None:
        node = self._parameterNode
        if not node:
            return
        key = (node.inputDirectory or "", node.outputDirectory or "", getattr(node, "batchRoot", "") or "")
        if key == getattr(self, "_runHistoryRootsKey", None):
            return
        self._runHistoryRootsKey = key
        timer = getattr(self, "_runHistoryRefreshTimer", None)
        if timer is None:
            timer = qt.QTimer()
            timer.setSingleShot(True)
            timer.setInterval(600)
            timer.connect("timeout()", self._refresh_run_history)
            self._runHistoryRefreshTimer = timer
        timer.start()

    def _refresh_run_history(self, select_path=None) -> None:
        try:
            refresh_run_history(self, select_path=select_path)
        except Exception as exc:
            self.appendLog(f"[Results] Could not scan for previous runs: {exc}")

    def _on_run_history_selection_changed(self) -> None:
        on_run_combo_changed(self)

    def _on_browse_run_directory_clicked(self) -> None:
        browse_run_directory(self)

    def _poll_run_progress(self) -> None:
        poll_run_progress(self)

    def _initialize_common_ui_state(self) -> None:
        self._populateDeviceCombo()
        self._lastProcessEventsAt = 0.0
        self._logLineLimit = 4000
        self._logAppendCount = 0
        self._environmentCacheTtlSeconds = 300.0
        self._lastEnvironmentSignature = None
        self._lastEnvironmentCheckedAt = 0.0
        if hasattr(self.ui, "subjectIdLineEdit"):
            self.ui.subjectIdLineEdit.setClearButtonEnabled(True)
        environment_hint = getattr(self.ui, "environmentHintLabel", None)
        if environment_hint:
            environment_hint.visible = False
        if hasattr(self.ui, "applyButton"):
            self.ui.applyButton.setAutoDefault(True)
            self.ui.applyButton.visible = False

        self._configure_workflow_shell()
        self._hide_legacy_product_panels()
        self._initialize_onboarding_panel()
        self._initialize_prepare_inputs_panel()
        self._initialize_environment_dependency_path_controls()
        self._initialize_product_selection_summary()
        self._initialize_run_status_panel()
        self._initialize_background_run_controls()
        install_compact_layout(self)
        from shared.ui_regions import install_region_panel
        install_region_panel(self)
        from shared.ui_advanced_layout import install_advanced_categories
        install_advanced_categories(self)
        self._apply_slicerddsurfer_ui_style()
        install_theme_change_listener(self)
        self._configure_log_widget()
        self._refresh_advanced_run_settings()
        apply_onboarding_language(self, getattr(self, "_onboardingLanguage", "en"))

        for widget_name in (
            "surfaceOptionsSectionLabel",
            "predictModeLabel",
            "predictModeComboBox",
            "deviceLabel",
            "deviceComboBox",
            "cleanupLabel",
            "cleanupCheckBox",
        ):
            widget = getattr(self.ui, widget_name, None)
            if widget:
                widget.visible = False

        headline = getattr(self.ui, "headlineLabel", None)
        if headline is not None:
            # Let the header shrink with narrow module panels instead of forcing a scroll bar.
            headline.setMinimumWidth(80)
        logo_label = getattr(self.ui, "logoLabel", None)
        if logo_label:
            logo_path = self.resourcePath("Icons/SlicerDSpace.png")
            if os.path.exists(logo_path):
                logo_label.setPixmap(QPixmap(logo_path))

        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.NodeRemovedEvent, self._queue_output_visibility_refresh)

        progress = getattr(self.ui, "statusProgressBar", None)
        if progress:
            progress.setRange(0, 100)
            progress.setValue(0)

        self._setStatus("waiting", "Waiting for input")
        self.initializeParameterNode()

    def _configure_workflow_shell(self) -> None:
        """Keep the top-level UI focused on input, products, and execution."""

        text_updates = {
            "environmentCollapsibleButton": "Advanced: Dependency profile",
            "advancedRunCollapsibleButton": "Analysis settings",
            "runSelectedButton": "Run SlicerDSpace",
        }
        for widget_name, text in text_updates.items():
            widget = getattr(self.ui, widget_name, None)
            if widget:
                try:
                    widget.text = text
                except Exception:
                    try:
                        widget.setText(text)
                    except Exception:
                        pass
        for widget_name in ("environmentCollapsibleButton", "advancedRunCollapsibleButton"):
            widget = getattr(self.ui, widget_name, None)
            if widget:
                try:
                    widget.collapsed = True
                except Exception:
                    pass
        for widget_name in (
            "parametersCollapsibleButton",
            "runCollapsibleButton",
        ):
            section = getattr(self.ui, widget_name, None)
            if section:
                try:
                    section.collapsed = False
                except Exception:
                    pass

    def _initialize_background_run_controls(self) -> None:
        if getattr(self, "_cancelRunButton", None) is not None:
            return
        actions_layout = getattr(self.ui, "selectedActionsLayout", None)
        if not actions_layout:
            return
        cancel_button = qt.QPushButton("Cancel running job")
        cancel_button.setObjectName("cancelRunButton")
        cancel_button.enabled = False
        cancel_button.visible = False
        cancel_button.setToolTip("Stop the background SlicerDSpace pipeline process and write a cancellation manifest.")
        cancel_button.connect("clicked()", self._onCancelRunClicked)
        actions_layout.addWidget(cancel_button)
        try:
            qt.QWidget.setTabOrder(self.ui.runSelectedButton, cancel_button)
        except Exception:
            pass
        self._cancelRunButton = cancel_button
        self._backgroundProcess = None
        self._backgroundOutputBuffer = ""
        self._backgroundManifestPath = None
        self._backgroundCancelRequested = False
        self._backgroundRunContext = {}

    def _hide_legacy_product_panels(self) -> None:
        """Hide old per-product forms; unified subject parameters now drive all runs."""
        for widget_name in (
            "dockerPostprocessCollapsibleButton",
            "ddparcelCollapsibleButton",
            "microstructureCollapsibleButton",
            "ddsegCollapsibleButton",
            "infoCollapsibleButton",
        ):
            widget = getattr(self.ui, widget_name, None)
            if widget:
                widget.visible = False

    def _environment_settings_key(self, dependency_id: str) -> str:
        return f"SlicerDSpace/ExternalDependencies/{dependency_id}"

    def _user_settings_key(self, name: str) -> str:
        return f"SlicerDSpace/UserSettings/{name}"

    def _common_user_setting_specs(self) -> tuple[tuple[str, type], ...]:
        return common_user_setting_specs() + (("brainMaskBackend", str), ("brainMaskDevice", str), ("pnlMaskRoute", str), ("device", str),
                                             ("ddsegDevice", str), ("orchestrationDkiFitMethod", str))

    def _coerce_user_setting_value(self, value: Any, value_type: type) -> Any:
        return coerce_user_setting_value(value, value_type)

    def _restore_common_user_settings_to_node(self) -> None:
        node = self._parameterNode
        if not node:
            return
        settings = slicer.app.userSettings()
        config = None
        try:
            config_path = default_user_config_path().expanduser()
            if config_path.is_file():
                config = load_user_config(config_path)
                for note in config.get("notes", ()):  # W5 / SC-2: a saved removed mask backend loaded as pnl_cnn
                    self.appendLog(f"[Settings] {note}")
                configured_dependencies = config.get("external_dependencies", {})
                for item in ENVIRONMENT_DEPENDENCY_PATHS:
                    dependency_id = item["id"]
                    value = configured_dependencies.get(dependency_id)
                    if value:
                        settings.setValue(self._environment_settings_key(dependency_id), value)
                    else:
                        settings.remove(self._environment_settings_key(dependency_id))
        except Exception as exc:
            self.appendLog(f"[Settings] Shared user config could not be restored: {exc}")
        if not consume_user_defaults_request(node):
            return
        with slicer.util.NodeModify(node):
            for attribute, value_type in self._common_user_setting_specs():
                key = self._user_settings_key(attribute)
                try:
                    has_key = bool(settings.contains(key))
                except Exception:
                    has_key = settings.value(key, None) is not None
                if not has_key:
                    continue
                try:
                    value = self._coerce_user_setting_value(settings.value(key), value_type)
                    if attribute == 'orchestrationWmqlPreset' and value != 'official':
                        value = 'official'  # Migrate saved defaults, not loaded scientific run records.
                    if attribute == 'brainMaskBackend' and value in REMOVED_BRAIN_MASK_BACKENDS:
                        # W5 / SC-2: a saved removed backend loads as pnl_cnn with one note; the clinician is not blocked.
                        self.appendLog(f"[Settings] Saved brain mask backend {value!r} was removed. Brain masking now uses pnl_cnn.")
                        value = 'pnl_cnn'
                    setattr(node, attribute, value)
                except Exception:
                    continue
        if config is not None:
            recent = config.get("recent", {})
            defaults = config.get("run_defaults", {})
            targets = set(defaults.get("targets", ()) or ())
            targets_configured = "targets" in defaults
            with slicer.util.NodeModify(node):
                node.subjectId = str(recent.get("subject_id") or node.subjectId or "")
                node.inputDirectory = str(recent.get("subject_root") or node.inputDirectory or "")
                node.outputDirectory = str(recent.get("output_root") or node.outputDirectory or "")
                node.orchestrationRunName = str(defaults.get("run_name") or node.orchestrationRunName or "Test1")
                node.brainMaskBackend = str(defaults.get("brain_mask_backend") or "pnl_cnn")
                node.brainMaskDevice = str(defaults.get("brain_mask_device") or "auto")
                node.pnlMaskRoute = str(defaults.get("pnl_mask_route") or "auto")
                node.device = str(defaults.get("surface_device") or "auto")
                node.ddsegDevice = str(defaults.get("ddseg_device") or "auto")
                from shared.algorithm_selection import choice_value, BUNDLES
                node.orchestrationTractographyBackend = choice_value(
                    defaults.get("tractography_backend") or node.orchestrationTractographyBackend or "csd_act"
                )
                node.orchestrationBundleBackend = choice_value(
                    defaults.get("bundle_backend") or node.orchestrationBundleBackend or "wmql", BUNDLES, "wmql"
                )
                stages = defaults.get("microstructure_stages") or ["dti"]
                node.wmaAtlas = str(defaults.get("wma_atlas") or "org")
                node.orchestrationMicrostructureStages = ",".join(selected_microstructure_stages(stages))
                node.orchestrationDkiFitMethod = str(defaults.get("dki_fit_method") or "MKCURVE").upper()
                node.connectomeBackend = str(defaults.get("connectome_backend") or node.connectomeBackend or "auto")
                node.orchestrationSelectStreamlines = int(
                    defaults.get("select_streamlines") or node.orchestrationSelectStreamlines or 1_000_000
                )
                node.autoInstallPythonDependencies = bool(
                    defaults.get("auto_install_python_dependencies", node.autoInstallPythonDependencies)
                )
                if targets_configured:
                    node.targetPreprocessing = "preprocessing" in targets
                    node.targetSurface = "surface" in targets
                    node.targetSurfacePostprocess = "surface_postprocess" in targets
                    node.targetParcellation = "parcellation" in targets
                    node.targetSegmentation = "segmentation" in targets
                    node.targetMicrostructure = bool(
                        targets & {"microstructure", "microstructure_freewater", "microstructure_noddi", "microstructure_dki",
                                "microstructure_ifw"}
                    )
                    node.targetTractography = bool(targets & {"tractography", "saf"})
                    node.targetBundles = bool(targets & {"bundles", "wma", "tractcloud"})
                    node.targetConnectome = "connectome" in targets
        else:
            self._save_common_user_settings()

    def _save_common_user_settings(self) -> None:
        node = self._parameterNode
        if not node:
            return
        settings = slicer.app.userSettings()
        for attribute, _value_type in self._common_user_setting_specs():
            try:
                value = getattr(node, attribute)
            except Exception:
                continue
            settings.setValue(self._user_settings_key(attribute), value)
        from shared.user_config import selected_microstructure_stages

        targets = list(self._requested_product_targets())  # K4: the user's ticks, not the closed set
        stages = list(selected_microstructure_stages(node.orchestrationMicrostructureStages or "dti"))
        external_dependencies = {
            item["id"]: self._read_environment_path_setting(item["id"]) or None
            for item in ENVIRONMENT_DEPENDENCY_PATHS
        }
        try:
            _, path = update_user_config(
                {
                    "recent": {
                        "subject_id": node.subjectId or "",
                        "subject_root": node.inputDirectory or "",
                        "output_root": node.outputDirectory or "",
                    },
                    "run_defaults": {
                        "run_name": node.orchestrationRunName or "Test1",
                        "targets": targets,
                        "brain_mask_backend": getattr(node, "brainMaskBackend", "pnl_cnn") or "pnl_cnn",
                        "brain_mask_device": getattr(node, "brainMaskDevice", "auto") or "auto",
                        "pnl_mask_route": getattr(node, "pnlMaskRoute", "auto") or "auto",
                        "ddseg_device": node.ddsegDevice or "auto",
                        "surface_device": node.device or "auto",
                        "tractcloud_device": "cuda" if (node.ddsegDevice or "auto").startswith("cuda") else node.ddsegDevice or "auto",
                        "tractography_backend": node.orchestrationTractographyBackend or "csd_act",
                        "bundle_backend": node.orchestrationBundleBackend or "wmql",
                        "wma_atlas": getattr(node, "wmaAtlas", "org") or "org",
                        "microstructure_stages": stages or ["dti"],
                        "dki_fit_method": getattr(node, "orchestrationDkiFitMethod", "MKCURVE") or "MKCURVE",
                        "connectome_backend": node.connectomeBackend or "auto",
                        "select_streamlines": int(node.orchestrationSelectStreamlines),
                        "auto_install_python_dependencies": bool(node.autoInstallPythonDependencies),
                    },
                    "external_dependencies": external_dependencies,
                }
            )
            self._sharedUserConfigPath = path
        except Exception as exc:
            self.appendLog(f"[Settings] Shared user config could not be saved: {exc}")

    def _schedule_common_user_settings_save(self) -> None:
        if not self._parameterNode:
            return
        timer = getattr(self, "_commonUserSettingsSaveTimer", None)
        if timer is None:
            timer = qt.QTimer(self.parent)
            timer.setSingleShot(True)
            timer.setInterval(350)
            timer.connect("timeout()", self._save_common_user_settings)
            self._commonUserSettingsSaveTimer = timer
        timer.start()

    def _save_python_install_status(self, packages: tuple[str, ...], status: str, detail: str = "") -> None:
        try:
            payload = {
                "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "packages": list(packages),
                "status": status,
                "detail": detail,
            }
            slicer.app.userSettings().setValue(
                self._user_settings_key("lastPythonInstall"),
                json.dumps(payload, ensure_ascii=False),
            )
        except Exception:
            pass

    @staticmethod
    def _is_other_slicer_installation_path(raw_value: str) -> bool:
        """Return True when ``raw_value`` points inside a different 3D Slicer installation than the running one."""

        from shared.ui_environment_panel import is_other_slicer_installation_path

        slicer_home_raw = getattr(getattr(slicer, "app", None), "slicerHome", "")
        slicer_home_val = slicer_home_raw() if callable(slicer_home_raw) else slicer_home_raw
        return is_other_slicer_installation_path(raw_value, slicer_home_val)

    def _read_environment_path_setting(self, dependency_id: str) -> str:
        settings = slicer.app.userSettings()
        value = str(settings.value(self._environment_settings_key(dependency_id), "") or "").strip()
        if not value:
            try:
                ext = load_user_config().get("external_dependencies") or {}
                value = str(ext.get(dependency_id) or "").strip()
            except Exception:
                value = ""
        if value and dependency_id in {
            "python_executable",
            "slicer_executable",
            "wmql_python_executable",
            "tractcloud_python_executable",
            "fiber_tract_measurements_cli",
        } and self._is_other_slicer_installation_path(value):
            value = ""
            try:
                settings.setValue(self._environment_settings_key(dependency_id), "")
                update_user_config({"external_dependencies": {dependency_id: ""}})
            except Exception:
                pass
        if value and dependency_id in {
            "mrtrix_bin_dir",
            "preprocessing_mrtrix_bin_dir",
            "preprocessing_fsl_bin_dir",
            "postprocess_runtime",
        }:
            if not Path(value).expanduser().exists():
                value = ""
        if value and dependency_id in {"mrtrix_bin_dir", "preprocessing_mrtrix_bin_dir"}:
            tckedit_name = "tckedit.exe" if sys.platform == "win32" else "tckedit"
            if not (Path(value).expanduser() / tckedit_name).is_file():
                try:
                    from shared.native_runtimes import installed_bundle_bin_dirs
                    for cand_bin in installed_bundle_bin_dirs("mrtrix"):
                        if (cand_bin / tckedit_name).is_file():
                            value = cand_bin.as_posix()
                            settings.setValue(self._environment_settings_key(dependency_id), value)
                            update_user_config({"external_dependencies": {dependency_id: value}})
                            break
                except Exception:
                    pass
        if not value:
            try:
                if dependency_id in {"mrtrix_bin_dir", "preprocessing_mrtrix_bin_dir"}:
                    from shared.native_runtimes import installed_bundle_bin_dirs
                    bins = installed_bundle_bin_dirs("mrtrix")
                    if bins:
                        value = bins[0].as_posix()
                elif dependency_id == "preprocessing_fsl_bin_dir":
                    from shared.native_runtimes import installed_bundle_bin_dirs
                    bins = installed_bundle_bin_dirs("fsl")
                    if bins:
                        value = bins[0].as_posix()
                elif dependency_id == "postprocess_runtime":
                    from shared.surface_runtime import bundled_surface_runtime
                    bundled = bundled_surface_runtime()
                    if bundled is not None:
                        value = bundled.as_posix()
                if value:
                    settings.setValue(self._environment_settings_key(dependency_id), value)
                    update_user_config({"external_dependencies": {dependency_id: value}})
            except Exception:
                pass
        return value

    def _save_environment_path_setting(self, dependency_id: str, value: str) -> None:
        self._lastEnvironmentResult = None
        self._lastEnvironmentSignature = None
        settings = slicer.app.userSettings()
        clean_value = (value or "").strip()
        if clean_value:
            settings.setValue(self._environment_settings_key(dependency_id), clean_value)
        else:
            settings.remove(self._environment_settings_key(dependency_id))
        try:
            _, path = update_user_config(
                {"external_dependencies": {dependency_id: clean_value or None}}
            )
            self._sharedUserConfigPath = path
        except Exception as exc:
            self.appendLog(f"[Settings] Dependency path JSON could not be saved: {exc}")

    def _line_edit_text(self, line_edit) -> str:
        text_attr = getattr(line_edit, "text", "")
        return str(text_attr() if callable(text_attr) else text_attr).strip()

    def _show_actionable_error(
        self,
        *,
        title: str,
        what: str,
        why: str,
        next_step: str,
        auto_fix: str,
        details: object,
    ) -> None:
        actions = [
            ("Re-check environment", self._onValidateClicked),
            ("Open dependency guide", self._on_open_model_assets_guide_clicked),
        ]
        lower_title = title.lower()
        if "model asset" in lower_title or "model" in lower_title:
            actions.insert(0, ("Choose/download model assets", lambda: self._install_model_assets_from_environment_dialog(profile="core")))
            actions.append(("Open model asset folder", self._on_open_model_assets_root_clicked))
        show_actionable_error_dialog(
            parent=self.parent,
            title=title,
            what=what,
            why=why,
            next_step=next_step,
            auto_fix=auto_fix,
            details=details,
            actions=tuple(actions),
        )

    def _on_open_model_assets_root_clicked(self) -> None:
        try:
            module = self._load_model_assets_module()
            configured = self._environment_dependency_path_kwargs().get("model_assets_dir")
            asset_root = configured if isinstance(configured, Path) and str(configured) else module.configured_model_assets_root()
            self._open_local_path(Path(asset_root), missing_message="The configured model asset root does not exist yet.")
        except Exception as exc:
            slicer.util.infoDisplay(f"Model asset root is unavailable:\n{exc}", parent=self.parent)

    def _initialize_environment_dependency_path_controls(self) -> None:
        if getattr(self, "_environmentPathLineEdits", None) is not None:
            return
        environment_layout = getattr(self.ui, "environmentLayout", None)
        if not environment_layout:
            return
        install_environment_dependency_path_controls(self, environment_layout)
        for key, name in (
            ("dki_matlab_executable", "dkiMatlabExecutablePathLineEdit"),
            ("dki_mkcurve_root", "dkiMkcurveRootPathLineEdit"),
        ):
            line_edit = self._environmentPathLineEdits.get(key)
            if line_edit is not None:
                line_edit.setObjectName(name)

    def _initialize_prepare_inputs_panel(self) -> None:
        if getattr(self, "_prepareInputWidgets", None) is not None:
            return
        configuration_layout = getattr(self.ui, "configurationLayout", None)
        if not configuration_layout:
            return
        install_prepare_inputs_panel(self, configuration_layout)

    def _initialize_onboarding_panel(self) -> None:
        if getattr(self, "_onboardingWidgets", None) is not None:
            return
        from shared.ui_run_history import section_layout

        parent_layout = section_layout(self, "gettingStartedCollapsibleButton", "gettingStartedLayout")
        if parent_layout is None:
            return
        install_onboarding_panel(self, parent_layout)
        section = getattr(self.ui, "gettingStartedCollapsibleButton", None)
        if section is not None:
            # Expanded for newcomers; collapsed once the first-run wizard was seen.
            try:
                from shared.ui_onboarding import FIRST_RUN_SETTING

                seen = str(slicer.app.userSettings().value(FIRST_RUN_SETTING, "false")).lower() in {"1", "true", "yes"}
            except Exception:
                seen = False
            section.collapsed = seen
        settings = slicer.app.userSettings()
        try:
            config = load_user_config()
            self._sharedUserConfigPath = default_user_config_path()
            configured_language = config.get("ui", {}).get("language")
        except Exception:
            configured_language = None
        language = normalized_language(str(configured_language or settings.value(LANGUAGE_SETTING, "en") or "en"))
        self._onboardingLanguage = language
        apply_onboarding_language(self, language)

    def _on_onboarding_language_changed(self, *args) -> None:
        combo = (getattr(self, "_onboardingWidgets", {}) or {}).get("language_combo")
        language = "en"
        if combo:
            try:
                language = str(combo.itemData(combo.currentIndex) or "en")
            except Exception:
                language = str(getattr(combo, "currentText", "English") or "English")
        language = normalized_language(language)
        self._onboardingLanguage = language
        slicer.app.userSettings().setValue(LANGUAGE_SETTING, language)
        try:
            _, path = update_user_config({"ui": {"language": language}})
            self._sharedUserConfigPath = path
        except Exception as exc:
            self.appendLog(f"[Settings] Language JSON could not be saved: {exc}")
        apply_onboarding_language(self, language)

    def _update_onboarding_dashboard(self, result=None) -> None:
        dashboard = (getattr(self, "_onboardingWidgets", {}) or {}).get("dashboard")
        if not dashboard:
            return
        language = getattr(self, "_onboardingLanguage", "en")
        dashboard.setText(capability_dashboard_text(result, language=language))

    def _on_show_first_run_wizard_clicked(self) -> None:
        show_first_run_wizard(self)

    def _on_try_demo_clicked(self) -> None:
        try:
            subject_root = prepare_demo_for_owner(self)
            result = self._run_environment_dependency_check()
            if result is not None:
                self._update_environment_result_label(result)
                self._update_environment_path_status_labels(result)
                self._update_environment_status(result)
                self._update_onboarding_dashboard(result)
            language = getattr(self, "_onboardingLanguage", "en")
            if language == "zh":
                message = (
                    "合成示例数据已准备完成。\n\n"
                    f"Subject root: {subject_root}\n\n"
                    "将使用 DTI microstructure + dti_det tractography 轻量流程。关闭此提示后，"
                    "SlicerDSpace 会自动执行 preflight 并开始运行。"
                )
            else:
                message = (
                    "Synthetic demo subject is ready.\n\n"
                    f"Subject root: {subject_root}\n\n"
                    "SlicerDSpace will run the lightweight DTI microstructure + dti_det tractography workflow. "
                    "Close this message to start preflight and execution automatically."
                )
            slicer.util.infoDisplay(
                message,
                parent=self.parent,
            )
            qt.QTimer.singleShot(0, self._onRunSelectedClicked)
        except Exception as exc:
            self._show_actionable_error(
                title="Demo Setup Failed",
                what="SlicerDSpace could not prepare the synthetic demo subject.",
                why="The temporary directory may be unavailable, or required Python packages such as nibabel/numpy are missing.",
                next_step="Run Check environment dependencies, install missing Python packages, then try Demo again.",
                auto_fix="No external software or model files were modified.",
                details=exc,
            )

    def _on_copy_diagnostic_report_clicked(self) -> None:
        report = diagnostic_report_text(self)
        try:
            qt.QApplication.clipboard().setText(report)
            slicer.util.infoDisplay("Diagnostic report copied to clipboard.", parent=self.parent)
        except Exception:
            slicer.util.infoDisplay(report, windowTitle="SlicerDSpace diagnostic report", parent=self.parent)

    def _on_open_user_config_clicked(self) -> None:
        path = default_user_config_path().expanduser()
        try:
            if not path.is_file():
                _, path = update_user_config({})
            self._sharedUserConfigPath = path
            self._open_local_path(path, missing_message="The shared user configuration JSON is not available.")
        except Exception as exc:
            slicer.util.infoDisplay(f"Could not open shared user configuration:\n{exc}", parent=self.parent)

    def _add_environment_path_row(self, form_layout, item: dict[str, str]) -> None:
        add_environment_path_row(self, form_layout, item)

    def _add_environment_section_header(self, form_layout, title: str, description: str) -> None:
        add_environment_section_header(form_layout, title, description)

    def _add_environment_subsection_label(self, form_layout, title: str, description: str):
        return add_environment_subsection_label(form_layout, title, description)

    def _add_model_assets_row(self, form_layout) -> None:
        add_model_assets_row(self, form_layout)

    def _open_local_path(self, path: Path, *, missing_message: str) -> None:
        resolved = path.expanduser().resolve()
        if not resolved.exists():
            slicer.util.infoDisplay(missing_message, parent=self.parent)
            return
        try:
            opened = qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(resolved.as_posix()))
        except Exception as exc:
            slicer.util.errorDisplay(f"Failed to open path:\n{exc}", parent=self.parent)
            return
        if not opened:
            slicer.util.infoDisplay(f"Path: {resolved}", parent=self.parent)

    def _on_open_model_assets_guide_clicked(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        self._open_local_path(
            doc_path(repo_root, "model_assets"),
            missing_message="docs/assets/MODEL_ASSETS.md is not available in the current repository checkout.",
        )

    def _on_open_model_assets_manifest_clicked(self) -> None:
        self._open_local_path(
            Path(__file__).resolve().parent.parent / "models" / "assets_manifest.json",
            missing_message="models/assets_manifest.json is not available in the current repository checkout.",
        )

    def _on_open_model_asset_terms_clicked(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        self._open_local_path(
            doc_path(repo_root, "model_asset_terms"),
            missing_message="docs/assets/MODEL_ASSET_TERMS.md is not available in the current repository checkout.",
        )

    def _on_open_release_notes_clicked(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        self._open_local_path(
            doc_path(repo_root, "release_notes"),
            missing_message="docs/release/RELEASE_NOTES_v0.1.0.md is not available in the current repository checkout.",
        )

    def _on_show_model_asset_provenance_clicked(self) -> None:
        slicer.util.infoDisplay(
            model_assets_provenance_text(Path(__file__).resolve().parent.parent),
            windowTitle="SlicerDSpace model asset provenance",
            parent=self.parent,
        )

    def _load_model_assets_module(self):
        module_name = "slicerdsurfer_download_model_assets"
        cached = sys.modules.get(module_name)
        if cached is not None:
            return cached
        script_path = Path(__file__).resolve().parent.parent / "scripts" / "download_model_assets.py"
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load model asset helper: {script_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _model_asset_verification(self, profile: str, *, check_hashes: bool = True) -> tuple[object, object, dict[str, Any]]:
        module = self._load_model_assets_module()
        repo_root = Path(__file__).resolve().parent.parent
        manifest = module.load_manifest(repo_root / "models" / "assets_manifest.json")
        plan = module.resolve_asset_plan(manifest, profile)
        configured = self._environment_dependency_path_kwargs().get("model_assets_dir")
        asset_root = module.resolve_model_assets_root(
            explicit=configured if isinstance(configured, Path) else None,
            repo_root=repo_root,
            required_paths=tuple(plan.required_paths),
            prefer_existing_legacy=True,
        )
        verification = module.verify_asset_plan(asset_root, plan, check_hashes=check_hashes)
        verification["asset_root"] = asset_root.as_posix()
        return module, plan, verification

    def _format_model_asset_status_line(self, profile: str, verification: dict[str, Any]) -> str:
        checked = int(verification.get("checked_count") or 0)
        missing = list(verification.get("missing") or [])
        pointers = list(verification.get("lfs_pointers") or [])
        if verification.get("ok"):
            if int(verification.get("hashes_checked") or 0) > 0:
                return f"{profile}=verified ({checked} file(s))"
            return f"{profile}=present ({checked} file(s))"
        problems = []
        if missing:
            problems.append(f"{len(missing)} missing")
        if pointers:
            problems.append(f"{len(pointers)} LFS pointer")
        problem_text = ", ".join(problems) if problems else "not verified"
        return f"{profile}=needs download ({checked} checked; {problem_text})"

    def _set_model_assets_status_text(self, text: str) -> None:
        label = getattr(self, "_modelAssetsSummaryLabel", None)
        if label:
            from shared.ui_environment_paths import compact_model_assets_summary

            label.setToolTip(text)
            label.setText(compact_model_assets_summary(text))
        self.appendLog("[Model assets] " + text)

    def _on_check_model_assets_status_clicked(self) -> None:
        try:
            _, _, core = self._model_asset_verification("core", check_hashes=False)
            _, _, all_assets = self._model_asset_verification("all", check_hashes=False)
        except Exception as exc:
            self._show_actionable_error(
                title="Model Asset Status Failed",
                what="SlicerDSpace could not verify the model asset manifest.",
                why="The checkout is missing the manifest, Git LFS metadata is unavailable, or the manifest profile is invalid.",
                next_step="Set Model assets root or run scripts/download_model_assets.sh --profile core --asset-root <asset-root>, then check status again.",
                auto_fix="Use the Verify core button for local verification; large downloads are not started by Check status.",
                details=exc,
            )
            return
        text = (
            f"Unified source: models/assets_manifest.json. Asset root: {core.get('asset_root')}. Current status: "
            + self._format_model_asset_status_line("core", core)
            + "; "
            + self._format_model_asset_status_line("all", all_assets)
            + "."
        )
        self._set_model_assets_status_text(text)

    def _on_verify_core_model_assets_clicked(self) -> None:
        self._start_model_asset_verification()

    def _start_model_asset_verification(self, *, lockfile: Optional[Path] = None) -> bool:
        from shared.ui_setup_jobs import start_setup_job
        try:
            # Resolve paths without hashing large files on the GUI thread.
            _, plan, verification = self._model_asset_verification("core", check_hashes=False)
            asset_root = Path(str(verification["asset_root"]))
        except Exception as exc:
            self._show_actionable_error(
                title="Core Model Asset Verification Failed",
                what="SlicerDSpace could not verify the core model assets.",
                why="The manifest or Git LFS file list could not be read.",
                next_step="Set Model assets root or run scripts/download_model_assets.sh --profile core --asset-root <asset-root> and retry.",
                auto_fix="This button verifies existing files only; it does not download large Git LFS objects.",
                details=exc,
            )
            return False
        script = Path(__file__).resolve().parents[1] / "scripts/download_model_assets.py"
        args = ["--launch", "PythonSlicer", "-u", str(script), "--profile", "core",
                "--asset-root", str(asset_root), "--verify-only"]
        if lockfile is not None:
            args.extend(["--write-lockfile", str(lockfile)])

        def done(status, details, log_path):
            self._lastEnvironmentResult = self._lastEnvironmentSignature = None
            if status == "completed":
                text = f"Core model assets verified ({len(plan.required_paths)} files)."
                if lockfile is not None:
                    text += f" Lockfile: {lockfile}"
            else:
                text = "Model asset verification cancelled." if status == "cancelled" else "Model asset verification failed."
            self._set_model_assets_status_text(text)
            self._setStatus("ready" if status == "completed" else "warning", text)
            if status == "failed":
                self._show_actionable_error(
                    title="Core Model Asset Verification Failed", what=text,
                    why="A required file is missing, is an LFS pointer, has a checksum mismatch, or the lockfile destination is not writable.",
                    next_step="Inspect the setup log, select the correct model asset root, and retry.",
                    auto_fix="Verification does not download or modify model files.",
                    details=details + "\nLog: " + str(log_path))

        return start_setup_job(self, "Verify core model assets", [(self._slicer_launcher_program(), args, {})], done)

    def _on_write_model_asset_lockfile_clicked(self) -> None:
        selected = qt.QFileDialog.getSaveFileName(
            slicer.util.mainWindow(),
            "Write SlicerDSpace model asset lockfile",
            str(Path.home() / "model_assets_core_lock.json"),
            "JSON files (*.json)",
        )
        path_text = selected[0] if isinstance(selected, tuple) else selected
        path_text = str(path_text or "").strip()
        if not path_text:
            return
        self._start_model_asset_verification(lockfile=Path(path_text).expanduser().resolve())

    def _on_open_environment_report_dir_clicked(self) -> None:
        report_dir = getattr(self, "_latestEnvironmentReportDir", None)
        if not report_dir:
            slicer.util.infoDisplay("Run Check environment dependencies first.", parent=self.parent)
            return
        self._open_local_path(Path(report_dir), missing_message="The latest environment report directory is unavailable.")

    def _on_open_install_logs_dir_clicked(self) -> None:
        report_dir = getattr(self, "_latestEnvironmentReportDir", None)
        if not report_dir:
            slicer.util.infoDisplay("Run Check environment dependencies or install Python packages first.", parent=self.parent)
            return
        self._open_local_path(Path(report_dir), missing_message="The latest install log directory is unavailable.")

    def _python_install_target_text(self) -> str:
        return python_install_target_text(sys.executable or "unknown")

    def _environment_dependency_config_payload(self) -> dict[str, object]:
        widgets = getattr(self, "_environmentPathLineEdits", {}) or {}
        external_dependencies = {}
        for item in ENVIRONMENT_DEPENDENCY_PATHS:
            dependency_id = item["id"]
            widget = widgets.get(dependency_id)
            value = self._line_edit_text(widget) if widget else self._read_environment_path_setting(dependency_id)
            if value:
                external_dependencies[dependency_id] = value
        return dependency_profile_payload(
            external_dependencies,
            python_executable=sys.executable or "",
            platform_text=platform.platform(),
        )

    def _remember_environment_report_dir(self, subject_root_text: str, output_root_text: str) -> None:
        try:
            subject_root = Path(subject_root_text).expanduser().resolve() if subject_root_text else None
            output_root = Path(output_root_text).expanduser().resolve() if output_root_text else None
            if subject_root is not None and subject_root.exists():
                from shared.io_layout import subject_layout

                report_dir = subject_layout(subject_root, output_root=output_root).run_metadata / "environment"
            elif output_root is not None:
                report_dir = output_root / "00_run_metadata" / "environment"
            else:
                report_dir = Path(str(slicer.app.temporaryPath)) / "SlicerDSpace" / "00_run_metadata" / "environment"
            self._latestEnvironmentReportDir = report_dir
        except Exception:
            self._latestEnvironmentReportDir = None

    def _apply_environment_dependency_config(self, payload: dict[str, object]) -> int:
        data = external_dependencies_from_profile(payload)
        widgets = getattr(self, "_environmentPathLineEdits", {}) or {}
        applied = 0
        for item in ENVIRONMENT_DEPENDENCY_PATHS:
            dependency_id = item["id"]
            value = data.get(dependency_id, "")
            widget = widgets.get(dependency_id)
            if widget:
                widget.setText(value)
            self._save_environment_path_setting(dependency_id, value)
            status_label = (getattr(self, "_environmentPathStatusLabels", {}) or {}).get(dependency_id)
            if status_label:
                self._set_environment_path_status_label(status_label, "check pending", "pending" if value else "unchecked")
            if value:
                applied += 1
        return applied

    def _on_export_dependency_config_clicked(self) -> None:
        selected = qt.QFileDialog.getSaveFileName(
            slicer.util.mainWindow(),
            "Export SlicerDSpace dependency config",
            str(Path.home() / "slicerdsurfer-deps.json"),
            "JSON files (*.json)",
        )
        path_text = selected[0] if isinstance(selected, tuple) else selected
        path_text = str(path_text or "").strip()
        if not path_text:
            return
        path = Path(path_text).expanduser()
        payload = self._environment_dependency_config_payload()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.appendLog(f"[Environment] Exported dependency config: {path}")
        slicer.util.infoDisplay(f"Dependency config exported:\n{path}", parent=self.parent)

    def _on_import_dependency_config_clicked(self) -> None:
        selected = qt.QFileDialog.getOpenFileName(
            slicer.util.mainWindow(),
            "Import SlicerDSpace dependency config",
            str(Path.home()),
            "JSON files (*.json)",
        )
        path_text = selected[0] if isinstance(selected, tuple) else selected
        path_text = str(path_text or "").strip()
        if not path_text:
            return
        path = Path(path_text).expanduser()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            applied = self._apply_environment_dependency_config(payload)
        except Exception as exc:
            self._show_actionable_error(
                title="Dependency Config Import Failed",
                what="SlicerDSpace could not read or apply the selected dependency profile.",
                why="The file is not valid JSON, does not contain external_dependencies, or contains incompatible values.",
                next_step="Select a SlicerDSpace dependency profile JSON exported by this module, then rerun the environment scan.",
                auto_fix="Not available for malformed config files.",
                details=exc,
            )
            return
        self.appendLog(f"[Environment] Imported dependency config: {path} ({applied} path(s)); re-running environment scan.")
        slicer.util.infoDisplay(
            f"Imported {applied} dependency path(s). A fresh environment scan will verify this machine.",
            parent=self.parent,
        )
        self._onValidateClicked()

    def _prepare_inputs_widget_text(self, key: str) -> str:
        widget = (getattr(self, "_prepareInputWidgets", {}) or {}).get(key)
        if not widget:
            return ""
        return self._line_edit_text(widget)

    def _prepare_inputs_combo_text(self, key: str, default: str) -> str:
        widget = (getattr(self, "_prepareInputWidgets", {}) or {}).get(key)
        if not widget:
            return default
        if hasattr(widget, "tabData"):
            return str(widget.tabData(widget.currentIndex) or default)
        value = getattr(widget, "currentText", "")
        return str(value() if callable(value) else value or default).strip() or default

    def _set_prepare_inputs_widget_text(self, key: str, value: str) -> None:
        widget = (getattr(self, "_prepareInputWidgets", {}) or {}).get(key)
        if widget:
            widget.setText(value)

    def _shared_input_directory_text(self) -> str:
        widget = getattr(self.ui, "inputDirectoryPath", None)
        if widget:
            current_path = getattr(widget, "currentPath", "")
            return str(current_path() if callable(current_path) else current_path).strip()
        node = self._parameterNode
        return str((node.inputDirectory if node else "") or "").strip()

    def _effective_subject_id(self) -> str:
        """Typed subject ID, else the ID derived from the input path (C6)."""
        node = self._parameterNode
        if not node:
            return ""
        from shared.subject_id import effective_subject_id
        return effective_subject_id(node.subjectId, node.inputDirectory)

    def _refresh_subject_id_placeholder(self) -> None:
        widget = getattr(self.ui, "subjectIdLineEdit", None)
        node = self._parameterNode
        if widget is None or not node:
            return
        from shared.subject_id import derive_subject_id
        derived = derive_subject_id(node.inputDirectory or "")
        lang = getattr(self, "_onboardingLanguage", "en")
        placeholder = derived or onboarding_text(lang, "subject_id_placeholder")
        try:
            widget.setPlaceholderText(placeholder)
        except Exception:
            pass

    def _shared_subject_id_text(self) -> str:
        widget = getattr(self.ui, "subjectIdLineEdit", None)
        if widget:
            text = self._line_edit_text(widget)
            if text:
                return text
        node = self._parameterNode
        return str((node.subjectId if node else "") or "").strip()

    def _browse_prepare_input_source_path(self, key: str) -> None:
        source_kind = self._prepare_inputs_combo_text("source_kind", "auto")
        start_dir = self._prepare_inputs_widget_text(key) or self._shared_input_directory_text() or os.path.expanduser("~")
        if source_kind in {"dicom", "canonical"}:
            selected = qt.QFileDialog.getExistingDirectory(
                slicer.util.mainWindow(),
                "Select DWI source folder",
                start_dir,
            )
            selected_text = str(selected or "").strip()
        else:
            selected = qt.QFileDialog.getOpenFileName(
                slicer.util.mainWindow(),
                "Select DWI source file",
                start_dir,
                "DWI files (*.nii *.nii.gz *.nrrd *.nhdr);;All files (*)",
            )
            selected_text = selected[0] if isinstance(selected, tuple) else selected
            selected_text = str(selected_text or "").strip()
        if selected_text:
            self._set_prepare_inputs_widget_text(key, selected_text)

    def _browse_prepare_input_file(self, key: str) -> None:
        start_dir = self._prepare_inputs_widget_text(key) or self._prepare_inputs_widget_text("source") or os.path.expanduser("~")
        selected = qt.QFileDialog.getOpenFileName(
            slicer.util.mainWindow(),
            "Select sidecar file",
            start_dir,
            "DWI sidecars (*.bval *.bvec *.json *.nii *.nii.gz);;All files (*)",
        )
        selected_text = selected[0] if isinstance(selected, tuple) else selected
        selected_text = str(selected_text or "").strip()
        if selected_text:
            self._set_prepare_inputs_widget_text(key, selected_text)

    def _prepare_inputs_config(self, *, force_dry_run: bool = False) -> InputImportConfig:
        source_text = self._prepare_inputs_widget_text("source")
        if not source_text:
            raise ValueError("Select a DICOM folder, NIfTI file, NRRD/NHDR file, or canonical subject root.")
        from uuid import uuid4
        from shared.subject_id import derive_subject_id
        from shared.user_config import default_user_config_path

        source = Path(source_text).expanduser().resolve()
        subject_id = self._shared_subject_id_text() or derive_subject_id(str(source))
        safe_id = derive_subject_id(subject_id)
        if not safe_id:
            raise ValueError("Enter a valid subject ID.")
        workspace = default_user_config_path().parent / "subjects"
        subject_root = workspace / f"{safe_id}-{uuid4().hex[:12]}"
        source_kind_text = self._prepare_inputs_combo_text("source_kind", "auto")
        source_kind = detect_import_source_kind(source) if source_kind_text == "auto" else source_kind_text
        backend = self._prepare_inputs_combo_text("backend", "auto")
        output_kind = self._prepare_inputs_combo_text("output_kind", "canonical-nifti")

        def optional_path(key: str) -> Optional[Path]:
            value = self._prepare_inputs_widget_text(key)
            return Path(value).expanduser() if value else None

        external_paths = self._environment_dependency_path_kwargs()
        mrtrix_bin_dir = external_paths.get("mrtrix_bin_dir") or external_paths.get("preprocessing_mrtrix_bin_dir")
        mrconvert = (Path(mrtrix_bin_dir) / "mrconvert") if mrtrix_bin_dir else None

        return InputImportConfig(
            source=source,
            subject_root=subject_root,
            subject_id=subject_id,
            source_kind=source_kind,
            backend=backend,
            output_kind=output_kind,
            bval_path=optional_path("bval") if source_kind == "nifti" else None,
            bvec_path=optional_path("bvec") if source_kind == "nifti" else None,
            metadata_path=optional_path("metadata"),
            mask_path=optional_path("mask"),
            mrconvert_executable=mrconvert,
            dry_run=force_dry_run,
        )

    def _on_prepare_inputs_preview_clicked(self) -> None:
        try:
            manifest = prepare_inputs(self._prepare_inputs_config(force_dry_run=True))
        except Exception as exc:
            self._show_actionable_error(
                title="Input Import Preview Failed",
                what="SlicerDSpace could not build an input import plan.",
                why="The selected source path, sidecars, source type, backend, or destination subject root is invalid.",
                next_step="Check the source path and sidecar paths, or set Source type explicitly instead of auto.",
                auto_fix="Not available. The preview does not mutate files.",
                details=exc,
            )
            return
        self.appendLog("[Prepare Inputs] Dry-run plan:\n" + json.dumps(manifest, indent=2, sort_keys=True))
        slicer.util.infoDisplay("Input import plan created. See the SlicerDSpace log for details.", parent=self.parent)

    def _input_import_selection(self) -> tuple:
        source_kind = self._prepare_inputs_combo_text("source_kind", "nifti")
        file_fields = ["source", "mask", "metadata"]
        if source_kind == "nifti":
            file_fields += ["bval", "bvec"]
        elif source_kind == "nrrd":
            file_fields.append("raw")
        return (source_kind, self._shared_subject_id_text(), self._prepare_inputs_combo_text("backend", "auto"),
                self._prepare_inputs_combo_text("output_kind", "canonical-nifti"),
                *(self._prepare_inputs_widget_text(key) for key in file_fields))

    def _on_prepare_inputs_clicked(self, *, run_after: bool = False) -> None:
        from concurrent.futures import Future
        from threading import Thread

        if getattr(self, "_inputImportPending", False) or self._pipelineActive:
            return
        try:
            config = self._prepare_inputs_config()
        except ValueError as exc:
            slicer.util.errorDisplay(str(exc), parent=self.parent)
            return
        selection = self._input_import_selection()
        self._inputImportPending = True
        zh = getattr(self, "_onboardingLanguage", "en").startswith("zh")
        status_label = self._prepareInputWidgets["status"]
        status_label.setText("正在检查并导入影像…" if zh else "Checking and importing images…")
        button = self._prepareInputWidgets["prepare_button"]
        button.enabled = False
        self._refresh_run_button_state()
        def start(job, callback):
            future = Future()

            def work():
                try:
                    future.set_result(job())
                except Exception as exc:
                    future.set_exception(exc)

            def poll():
                if getattr(self, "_cleanedUp", False):
                    return
                if future.done():
                    callback(future)
                else:
                    qt.QTimer.singleShot(100, poll)

            Thread(target=work, daemon=True).start()
            qt.QTimer.singleShot(100, poll)

        def finish(future):
            self._inputImportPending = False
            if getattr(self, "_cleanedUp", False):
                return
            button.enabled = True
            self._refresh_run_button_state()
            try:
                manifest = future.result()
            except Exception as exc:
                status_label.setText(str(exc))
                return
            self.appendLog("[Prepare Inputs] Result:\n" + json.dumps(manifest, indent=2, sort_keys=True))
            if (manifest.get("status") == "completed" and self._parameterNode is not None
                    and selection == self._input_import_selection()):
                status_label.setText("影像已导入" if zh else "Inputs ready")
                with slicer.util.NodeModify(self._parameterNode):
                    self._parameterNode.inputDirectory = config.subject_root.as_posix()
                    self._parameterNode.subjectId = config.subject_id
                self._preparedInputSelection = self._input_import_selection()
                self._apply_shared_inputs_to_workflow_fields()
                self._checkCanApply()
                if run_after:
                    self._onRunSelectedClicked()
            else:
                status_label.clear()

        if config.source_kind != "dicom":
            raw = self._prepare_inputs_widget_text("raw")
            if config.source_kind == "nrrd" and raw:
                from shared.ui_prepare_inputs import import_detached_nrrd
                start(lambda: import_detached_nrrd(config, Path(raw).expanduser().resolve()), finish)
            else:
                start(lambda: prepare_inputs(config), finish)
            return

        from shared.dicom_input import scan_series
        from shared.ui_prepare_inputs import import_dicom_series

        def choose_series(future):
            try:
                series = future.result()
                if not series:
                    raise ValueError("No DICOM series found in the selected folder.")
            except Exception:
                finish(future if future.exception() else _failed_import(ValueError(
                    "No DICOM series found in the selected folder.")))
                return
            if selection != self._input_import_selection():
                self._inputImportPending = False
                button.enabled = True
                status_label.clear()
                self._refresh_run_button_state()
                return
            index = 0
            if len(series) > 1:
                zh = getattr(self, "_onboardingLanguage", "en").startswith("zh")
                labels = [
                    f"{index + 1}. {entry['description']} · {entry['frames']} "
                    + ("幅影像" if zh else "images")
                    + (f" · {entry['acquisition_time']}" if entry["acquisition_time"] else "")
                    for index, entry in enumerate(series)
                ]
                picked = qt.QInputDialog.getItem(
                    self.parent, "DICOM", "选择序列" if zh else "Select series", labels, 0, False,
                )
                label = picked[0] if isinstance(picked, tuple) else picked
                if not label:
                    self._inputImportPending = False
                    button.enabled = True
                    status_label.clear()
                    self._refresh_run_button_state()
                    return
                index = labels.index(str(label))
            start(lambda: import_dicom_series(config, series[index]), finish)

        def _failed_import(exc):
            failed = Future()
            failed.set_exception(exc)
            return failed

        start(lambda: scan_series(config.source), choose_series)

    def _on_open_prepare_input_manifest_clicked(self) -> None:
        subject_root_text = self._shared_input_directory_text()
        if not subject_root_text:
            slicer.util.infoDisplay("Select a subject root first.", parent=self.parent)
            return
        manifest_path = Path(subject_root_text).expanduser() / "input" / "input_import_manifest.json"
        self._open_local_path(
            manifest_path,
            missing_message="input/input_import_manifest.json is not available for the selected subject root.",
        )

    def _initialize_run_status_panel(self) -> None:
        install_run_status_panel(self)

    def _initialize_product_selection_summary(self) -> None:
        if getattr(self, "_selectedProductsSummaryLabel", None) is not None:
            return
        targets_layout = getattr(self.ui, "targetsLayout", None)
        configuration_layout = getattr(self.ui, "configurationLayout", None)
        if targets_layout is not None:
            configuration_layout = None
        elif configuration_layout is None:
            return
        summary = qt.QLabel("Selected products will appear here after the parameter node is ready.")
        summary.wordWrap = True
        summary.setObjectName("selectedProductsSummaryLabel")
        try:
            summary.setTextInteractionFlags(qt.Qt.TextSelectableByMouse | qt.Qt.TextSelectableByKeyboard)
        except Exception:
            pass
        summary.visible = False
        self._selectedProductsSummaryLabel = summary
        (targets_layout if targets_layout is not None else configuration_layout).addWidget(summary)
        # C4: one line under the target checkboxes naming the prerequisites that run automatically.
        from shared.ui_styles import is_dark_mode

        also_runs = qt.QLabel("")
        also_runs.wordWrap = True
        also_runs.setObjectName("alsoRunsAutomaticallyLabel")
        also_color = "#7CC4EA" if is_dark_mode(getattr(self, "_uiWidget", None)) else "#335B70"
        also_runs.setStyleSheet(f"QLabel#alsoRunsAutomaticallyLabel {{ color: {also_color}; }}")
        also_runs.visible = False
        self._alsoRunsLabel = also_runs
        (targets_layout if targets_layout is not None else configuration_layout).addWidget(also_runs)

    def _refresh_selected_products_summary(self) -> None:
        label = getattr(self, "_selectedProductsSummaryLabel", None)
        if not label:
            return
        if not self._parameterNode:
            label.setText("Selected products will appear here after the parameter node is ready.")
            return
        notice = getattr(self, "_alsoRunsLabel", None)
        if notice is not None:
            notice_text = self._closure_notice_text()
            if notice.text != notice_text:
                notice.setText(notice_text)
            target_visible = bool(notice_text)
            if notice.visible != target_visible:
                notice.visible = target_visible
        text, target_text = selected_products_summary(
            self._selected_product_targets(),
            BackendSelection(
                tractography=self._tractography_backend_text(),
                bundles=self._bundle_backend_text(),
                microstructure_stages=self._microstructure_stages_text(),
                connectome=(self._parameterNode.connectomeBackend or "auto").strip() or "auto",
            ),
        )
        if label.text != text:
            label.setText(text)
        run_button = getattr(self.ui, "runSelectedButton", None)
        if run_button:
            tip = "Run selected products after environment preflight. Current selection: " + target_text
            if run_button.toolTip != tip:
                run_button.setToolTip(tip)

    def _configure_log_widget(self) -> None:
        configure_log_widget(getattr(self.ui, "logPlainTextEdit", None), line_limit=int(self._logLineLimit))
        clear_button = getattr(self.ui, "clearLogButton", None)
        if clear_button is not None:
            clear_button.setSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Fixed)

    def _apply_slicerddsurfer_ui_style(self) -> None:
        apply_widget_styles(self)

    def _set_run_progress_message(self, message: str) -> None:
        self._runProgressMessageSource = str(message or "")
        set_run_progress_message(
            self,
            translated_ui_message(getattr(self, "_onboardingLanguage", "en"), self._runProgressMessageSource),
        )

    def _set_run_summary_message(self, message: str, state: str = "") -> None:
        self._runSummaryMessageSource = str(message or "")
        set_run_summary_message(
            self,
            translated_ui_message(getattr(self, "_onboardingLanguage", "en"), self._runSummaryMessageSource),
            state,
        )

    def _run_summary_from_manifest(self, manifest: dict[str, Any]) -> tuple[str, str]:
        return run_summary_from_manifest(manifest)

    def _set_run_summary_from_manifest(self, manifest: dict[str, Any], fallback: str = "") -> None:
        message, status = self._run_summary_from_manifest(manifest)
        if message:
            self._set_run_summary_message(message, status)
        elif fallback:
            self._set_run_summary_message(fallback, "unknown")

    def _set_latest_report_path(self, report_path: str | Path | None) -> None:
        set_existing_report_path(self, "_latestReportPath", "_runReportButton", report_path)

    def _set_latest_output_tree_path(self, report_path: str | Path | None) -> None:
        set_existing_report_path(self, "_latestOutputTreePath", "_runOutputTreeButton", report_path)

    def _on_open_latest_report_clicked(self) -> None:
        report_path = getattr(self, "_latestReportPath", None)
        if not report_path:
            slicer.util.infoDisplay("No generated report is available yet.", parent=self.parent)
            return
        try:
            opened = qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(Path(report_path).as_posix()))
        except Exception as exc:
            slicer.util.errorDisplay(f"Failed to open report:\n{exc}", parent=self.parent)
            return
        if not opened:
            slicer.util.infoDisplay(f"Report path: {report_path}", parent=self.parent)

    def _on_open_output_tree_report_clicked(self) -> None:
        report_path = getattr(self, "_latestOutputTreePath", None)
        if not report_path:
            slicer.util.infoDisplay("No generated output-tree report is available yet.", parent=self.parent)
            return
        try:
            opened = qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(Path(report_path).as_posix()))
        except Exception as exc:
            slicer.util.errorDisplay(f"Failed to open output-tree report:\n{exc}", parent=self.parent)
            return
        if not opened:
            slicer.util.infoDisplay(f"Output-tree report path: {report_path}", parent=self.parent)

    def _table_item(self, text: str, state: str = ""):
        return table_item(text, state)

    def _set_run_status_table_rows(self, rows: list[dict[str, str]]) -> None:
        set_run_status_table_rows(self, rows)

    def _update_run_status_table_item(self, key_candidates: tuple[str, ...], status: str, reason: str = "") -> None:
        update_run_status_table_item(self, key_candidates, status, reason)

    def _task_keys_for_running_line(self, line: str) -> tuple[str, ...]:
        return task_keys_for_running_line(line)

    def _update_run_plan_table(self, plan) -> None:
        rows, summary, plan_state = rows_from_plan(plan)
        self._set_run_status_table_rows(rows)
        self._set_run_summary_message(
            f"Plan: {summary['will_run']} will run, {summary['skipped']} skipped/reused, {summary['blocked']} blocked.",
            plan_state,
        )
        self._set_run_progress_message(
            f"Run plan ready: {summary['will_run']} will run, "
            f"{summary['skipped']} skipped, {summary['blocked']} blocked."
        )

    def _update_run_result_table(self, result) -> None:
        manifest_data = {}
        try:
            manifest_data = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest_data = {}
        rows = rows_from_result(result, manifest_data)
        self._set_run_status_table_rows(rows)
        if manifest_data:
            self._set_run_summary_from_manifest(manifest_data, fallback=f"Run finished. Manifest: {result.run_manifest_path}")
        self._set_run_progress_message(f"Run finished. Manifest: {result.run_manifest_path}")
        self._refresh_run_history(select_path=Path(result.run_manifest_path))

    def _on_run_status_table_double_clicked(self, row: int, column: int) -> None:
        table = getattr(self, "_runStatusTable", None)
        if not table:
            return
        path = row_open_path(table, row)
        if path is not None:
            qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(path.as_posix()))

    def _browse_environment_dependency_path(self, item: dict[str, str], line_edit) -> None:
        current = self._line_edit_text(line_edit)
        start_dir = current if current else os.path.expanduser("~")
        language = getattr(self, "_onboardingLanguage", "en")
        localized_label = dependency_ui_text(language, item["id"], "label") or item["label"]
        dialog_title = f"{onboarding_text(language, 'select_path')} {localized_label}"
        if item.get("mode") == "file":
            selected = qt.QFileDialog.getOpenFileName(
                slicer.util.mainWindow(),
                dialog_title,
                start_dir,
            )
        else:
            selected = qt.QFileDialog.getExistingDirectory(
                slicer.util.mainWindow(),
                dialog_title,
                start_dir,
            )
        selected_text = selected[0] if isinstance(selected, tuple) else selected
        selected_text = str(selected_text or "").strip()
        if not selected_text:
            return
        line_edit.setText(selected_text)
        self._save_environment_path_setting(item["id"], selected_text)
        status_label = (getattr(self, "_environmentPathStatusLabels", {}) or {}).get(item["id"])
        if status_label:
            self._set_environment_path_status_label(status_label, "check pending", "pending")

    def _clear_environment_dependency_path(self, dependency_id: str, line_edit) -> None:
        line_edit.setText("")
        self._save_environment_path_setting(dependency_id, "")
        status_label = (getattr(self, "_environmentPathStatusLabels", {}) or {}).get(dependency_id)
        if status_label:
            self._set_environment_path_status_label(status_label, "check pending", "pending")

    def _on_environment_dependency_path_edited(self, dependency_id: str, line_edit) -> None:
        self._save_environment_path_setting(dependency_id, self._line_edit_text(line_edit))
        status_label = (getattr(self, "_environmentPathStatusLabels", {}) or {}).get(dependency_id)
        if status_label:
            self._set_environment_path_status_label(status_label, "check pending", "pending")

    def _environment_dependency_path_kwargs(self) -> dict[str, Any]:
        widgets = getattr(self, "_environmentPathLineEdits", {}) or {}
        kwargs: dict[str, Any] = {}
        for item in ENVIRONMENT_DEPENDENCY_PATHS:
            dependency_id = item["id"]
            widget = widgets.get(dependency_id)
            value = self._line_edit_text(widget) if widget else self._read_environment_path_setting(dependency_id)
            kwargs[dependency_id] = value if item.get("value_type") == "text" else (Path(value).expanduser() if value else None)
        selected_atlas = getattr(getattr(self, "_parameterNode", None), "wmaAtlas", "org") or "org"
        from shared.optional_assets import inherited_wma_atlas_directory
        kwargs["wma_atlas_dir"] = inherited_wma_atlas_directory(
            kwargs.get("wma_atlas_dir"), selected_atlas, kwargs.get("model_assets_dir"))
        return kwargs

    def _environment_check_kwargs(
        self,
        *,
        subject_root_text: str = "",
        output_root_text: str = "",
        external_paths: Optional[dict[str, Optional[Path]]] = None,
    ) -> dict[str, Any]:
        node = self._parameterNode
        if not node:
            return {}
        external_paths = external_paths if external_paths is not None else self._environment_dependency_path_kwargs()
        subject_root_text = (subject_root_text or (node.inputDirectory or "")).strip()
        output_root_text = (output_root_text or (node.outputDirectory or "")).strip()
        targets = self._selected_environment_targets()
        mask_backend = (getattr(node, "brainMaskBackend", "pnl_cnn") or "pnl_cnn").strip()
        preprocessing_stages = (getattr(node, "orchestrationPreprocessingStages", "") or "").strip()
        inspected = inspect_input_directory(self, subject_root_text, effective_subject_id(node.subjectId, subject_root_text)) if subject_root_text else {}
        raw_inputs_known = inspected.get("mode") in {"subject_root", "dicom", "nifti", "nrrd"}
        input_root = inspected.get("subject_root") or subject_root_text
        input_dir = Path(input_root).expanduser() / "input" if input_root else None
        mask_file = input_dir / "nodif_brain_mask.nii.gz" if input_dir else None
        user_mask_exists = bool(
            (mask_file and mask_file.is_file() and not mask_file.with_name("nodif_brain_mask.manifest.json").is_file())
            or inspected.get("has_mask")
        )
        brain_mask_required = gui_brain_mask_required(
            targets,
            input_known=raw_inputs_known,
            input_mask_exists=user_mask_exists,
            preprocessing_on=bool(node.targetPreprocessing),
            mask_backend=mask_backend,
            preprocessing_stages=preprocessing_stages,
        )
        configured_py = external_paths.get("python_executable")
        same_interpreter = True
        if configured_py:
            try:
                from shared.external_process import interpreter_path

                same_interpreter = interpreter_path(Path(configured_py)).resolve() == interpreter_path(Path(sys.executable)).resolve()
            except Exception:
                same_interpreter = False
        python_path_entries = (
            tuple(
                Path(entry).expanduser().resolve()
                for entry in sys.path
                if entry and Path(entry).expanduser().exists()
            )
            if same_interpreter
            else (Path(__file__).resolve().parents[1],)
        )
        return {
            "targets": targets,
            "subject_root": Path(subject_root_text).expanduser() if subject_root_text else None,
            "output_root": Path(output_root_text).expanduser() if output_root_text else None,
            "brain_mask_backend": mask_backend,
            "wma_atlas": getattr(node, "wmaAtlas", "org") or "org",
            "brain_mask_device": getattr(node, "brainMaskDevice", "auto") or "auto",
            "pnl_mask_route": getattr(node, "pnlMaskRoute", "auto") or "auto",
            "tractcloud_device": "cuda" if (node.ddsegDevice or "auto").startswith("cuda") else node.ddsegDevice or "auto",
            "brain_mask_required": bool(brain_mask_required),
            "tractography_backend": self._closure_tractography_backend_text(),
            "connectome_backend": (node.connectomeBackend or "auto").strip(),
            "ddseg_device": (node.ddsegDevice or "auto").strip() or "auto",
            "surface_device": (node.device or "auto").strip() or "auto",
            "python_path_entries": python_path_entries,
            "microstructure_run_free_water": self._microstructure_stage_flags()[0],
            "microstructure_run_noddi": self._microstructure_stage_flags()[1],
            "microstructure_run_dki": self._microstructure_dki_enabled(),
            "dki_fit_method": getattr(node, "orchestrationDkiFitMethod", "MKCURVE") or "MKCURVE",
            "preprocessing_execution_backend": os.environ.get("SLICERDSPACE_PREPROCESSING_EXECUTION_BACKEND")
            or ("native" if sys.platform == "win32" else None),
            "preprocessing_stages": preprocessing_stages or None,
            **external_paths,
        }

    def _environment_check_signature(self, kwargs: dict[str, Any]) -> tuple:
        normalized = []
        for key, value in sorted(kwargs.items()):
            if isinstance(value, Path):
                normalized.append((key, value.expanduser().as_posix()))
            elif isinstance(value, tuple):
                normalized.append((key, tuple(str(item) for item in value)))
            else:
                normalized.append((key, value))
        return tuple(normalized)

    def _cache_environment_result(self, result, signature: tuple) -> None:
        self._lastEnvironmentResult = result
        self._lastEnvironmentSignature = signature
        self._lastEnvironmentCheckedAt = monotonic()
        try:
            payload = {
                "checked_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "signature": signature,
                "summary": result.to_dict() if hasattr(result, "to_dict") else {},
            }
            slicer.app.userSettings().setValue(
                self._user_settings_key("lastEnvironmentCheck"),
                json.dumps(payload, ensure_ascii=False),
            )
        except Exception:
            pass

    def _cached_environment_result(self, signature: tuple):
        result = getattr(self, "_lastEnvironmentResult", None)
        if result is None or getattr(self, "_lastEnvironmentSignature", None) != signature:
            return None
        age = monotonic() - float(getattr(self, "_lastEnvironmentCheckedAt", 0.0) or 0.0)
        if age > float(getattr(self, "_environmentCacheTtlSeconds", 300.0) or 300.0):
            return None
        return result

    def _set_environment_path_status_label(self, label, text: str, state: str) -> None:
        from shared.ui_styles import is_dark_mode

        for dependency_id, candidate in (getattr(self, "_environmentPathStatusLabels", {}) or {}).items():
            if candidate is label:
                sources = getattr(self, "_environmentPathStatusSources", None)
                if sources is not None:
                    sources[dependency_id] = str(text or "")
                states = getattr(self, "_environmentPathStatusStates", None)
                if states is not None:
                    states[dependency_id] = str(state or "")
                break
        label.setText(translated_ui_message(getattr(self, "_onboardingLanguage", "en"), text))
        label.setStyleSheet(environment_path_status_style(state, dark=is_dark_mode(getattr(self, "_uiWidget", None))))

    def _path_status_dependencies(self, result, dependency_id: str) -> list:
        return path_status_dependencies(result, dependency_id)

    def _format_detected_source(self, source: str) -> str:
        return format_detected_source(source)

    def _update_environment_path_status_labels(self, result=None) -> None:
        labels = getattr(self, "_environmentPathStatusLabels", {}) or {}
        if not labels:
            return
        for item in ENVIRONMENT_DEPENDENCY_PATHS:
            dependency_id = item["id"]
            label = labels.get(dependency_id)
            if not label:
                continue
            if result is None:
                self._set_environment_path_status_label(label, "not checked", "unchecked")
                continue
            dependencies = self._path_status_dependencies(result, dependency_id)
            text, state = environment_path_status_from_dependencies(dependencies)
            self._set_environment_path_status_label(label, text, state)

    def _find_slicer_module_panel(self):
        try:
            mw = slicer.util.mainWindow() if hasattr(slicer, "util") else None
            if mw is not None:
                if hasattr(mw, "modulePanel"):
                    panel = mw.modulePanel() if callable(mw.modulePanel) else mw.modulePanel
                    if panel is not None:
                        return panel
                if hasattr(slicer.util, "findChild"):
                    panel = slicer.util.findChild(mw, "ModulePanel")
                    if panel is not None:
                        return panel
        except Exception:
            pass
        curr = getattr(self, "parent", None)
        while curr is not None:
            if hasattr(curr, "setHelpAndAcknowledgmentVisible") or (
                hasattr(curr, "inherits") and curr.inherits("qSlicerModulePanel")
            ):
                return curr
            parent_fn = getattr(curr, "parentWidget", None)
            curr = parent_fn() if callable(parent_fn) else None
        return None

    def _set_module_help_section_hidden(self, hidden: bool, *, _retry: bool = True) -> None:
        """Hide Slicer's outer Help & Acknowledgement collapsible bar while SlicerDSpace is active."""

        panel = self._find_slicer_module_panel()
        if panel is None:
            if hidden and _retry:
                try:
                    qt.QTimer.singleShot(
                        0,
                        lambda: (
                            not getattr(self, "_cleanedUp", False)
                            and self._set_module_help_section_hidden(True, _retry=False)
                        ),
                    )
                except Exception:
                    pass
            return
        try:
            if hidden:
                if getattr(self, "_savedModulePanelHelpVisible", None) is None:
                    if hasattr(panel, "isHelpAndAcknowledgmentVisible"):
                        vis_attr = panel.isHelpAndAcknowledgmentVisible
                        prev = bool(vis_attr() if callable(vis_attr) else vis_attr)
                    elif hasattr(panel, "helpAndAcknowledgmentVisible"):
                        prev = bool(panel.helpAndAcknowledgmentVisible)
                    else:
                        prev = True
                    self._savedModulePanelHelpVisible = prev
                if hasattr(panel, "setHelpAndAcknowledgmentVisible"):
                    panel.setHelpAndAcknowledgmentVisible(False)
                elif hasattr(panel, "helpAndAcknowledgmentVisible"):
                    panel.helpAndAcknowledgmentVisible = False
            else:
                saved = getattr(self, "_savedModulePanelHelpVisible", None)
                if saved is not None:
                    if hasattr(panel, "setHelpAndAcknowledgmentVisible"):
                        panel.setHelpAndAcknowledgmentVisible(bool(saved))
                    elif hasattr(panel, "helpAndAcknowledgmentVisible"):
                        panel.helpAndAcknowledgmentVisible = bool(saved)
                    self._savedModulePanelHelpVisible = None
        except Exception:
            pass

    def cleanup(self) -> None:
        if getattr(self, "_cleanedUp", False):
            return
        self._cleanedUp = True
        close_input_format_help_popovers(self, destroy=True)
        uninstall_theme_change_listener(self)
        self._set_module_help_section_hidden(False, _retry=False)
        try:
            slicer.app.disconnect("aboutToQuit()", self.cleanup)
        except Exception:
            pass
        job = getattr(self, "_setupJob", None)
        if job:
            job.shutdown()
        process = getattr(self, "_backgroundProcess", None)
        if process is not None and qprocess_is_running(process):
            # Teardown cannot rely on timers or callbacks into deleted widgets.
            for signal_name, callback in (
                ("readyReadStandardOutput()", self._onBackgroundProcessReadyRead),
                ("finished(int,QProcess::ExitStatus)", self._onBackgroundProcessFinished),
                ("errorOccurred(QProcess::ProcessError)", self._onBackgroundProcessError),
            ):
                try:
                    process.disconnect(signal_name, callback)
                except Exception:
                    pass
            self._backgroundCancelRequested = True
            self._signal_background_process(process, force=False)
            if not process.waitForFinished(15000):
                self._signal_background_process(process, force=True)
                process.waitForFinished(3000)
            pgid = getattr(self, "_backgroundProcessGroupId", None)
            if getattr(self, "_backgroundProcessGrouped", False) and pgid:
                finish_process_group(pgid, grace_seconds=15)
            self._write_cancelled_manifest()
        # Cancelled launchers can exit before their children. The event loop
        # may stop before the delayed force-kill callback gets its turn.
        for pgid in list(getattr(self, "_pendingCancelledProcessGroups", set())):
            finish_process_group(pgid, grace_seconds=15)
        self._pendingCancelledProcessGroups = set()
        stop_progress_polling(self)
        timer = getattr(self, "_runHistoryRefreshTimer", None)
        if timer is not None:
            timer.stop()
        timer = getattr(self, "_commonUserSettingsSaveTimer", None)
        if timer is not None:
            timer.stop()
        self.removeObservers()

    def enter(self) -> None:
        self._set_module_help_section_hidden(True)
        try:
            from shared.ui_styles import theme_signature

            if theme_signature(getattr(self, "_uiWidget", None)) != getattr(self, "_activeAppearanceSignature", None):
                self._apply_slicerddsurfer_ui_style()
        except Exception:
            pass
        self.initializeParameterNode()
        from shared.setup_resume import refresh_setup_button
        refresh_setup_button(self)
        # Runs may have been produced from the command line or another session.
        self._refresh_run_history()

    def exit(self) -> None:
        close_input_format_help_popovers(self, destroy=False)
        self._set_module_help_section_hidden(False, _retry=False)
        if self._parameterNode:
            if self._parameterNodeGuiTag is not None:
                self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            if self.hasObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply):
                self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        self.setParameterNode(None)

    def _queue_output_visibility_refresh(self, caller=None, event=None) -> None:
        if getattr(self, "_outputVisibilityRefreshPending", False):
            return
        self._outputVisibilityRefreshPending = True

        def refresh():
            self._outputVisibilityRefreshPending = False
            if not getattr(self, "_cleanedUp", False):
                from shared.ui_run_history import _refresh_output_visibility
                _refresh_output_visibility(self)

        qt.QTimer.singleShot(0, refresh)

    def onSceneEndClose(self, caller, event) -> None:
        self._queue_output_visibility_refresh()
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        self.setParameterNode(self.logic.getParameterNode())
        if not self._parameterNode:
            return

        if not (self._parameterNode.ddparcelAtlasPath or "").strip():
            module_root = Path(__file__).resolve().parent.parent
            atlas_path, atlas_error = model_asset_path_or_none(
                "Resources/Data/100HCP-population-mean-T2.nii.gz", repo_root=module_root
            )
            if atlas_error is not None:
                # A tampered/corrupt asset must not abort module
                # initialization (that would leave the user unable to reach
                # the environment-check/redownload UI at all); log it and
                # leave the field for the user to fill in after redownloading.
                self.appendLog(f"[Model assets] DDParcel atlas failed integrity check: {atlas_error}")
            elif atlas_path is not None and atlas_path.is_file():
                self._parameterNode.ddparcelAtlasPath = atlas_path.as_posix()

        ddseg_weights_dir, ddseg_weights_error = model_asset_path_or_none(
            "segmentation/weights", repo_root=Path(__file__).resolve().parent.parent
        )
        if ddseg_weights_error is not None:
            self.appendLog(f"[Model assets] DDSeg weights failed integrity check: {ddseg_weights_error}")
        elif (
            not (self._parameterNode.ddsegWeightsDir or "").strip()
            and ddseg_weights_dir is not None
            and ddseg_weights_dir.is_dir()
        ):
            self._parameterNode.ddsegWeightsDir = ddseg_weights_dir.as_posix()

        slicer_base = self._default_slicer_base_dir()
        if not (self._parameterNode.ddsegSlicerBase or "").strip() and slicer_base is not None:
            self._parameterNode.ddsegSlicerBase = slicer_base.as_posix()

        slicer_dmri = self._default_slicerdmri_dir()
        if not (self._parameterNode.ddsegSlicerExt or "").strip() and slicer_dmri is not None:
            self._parameterNode.ddsegSlicerExt = slicer_dmri.as_posix()

    def _default_slicer_base_dir(self) -> Optional[Path]:
        app = getattr(slicer, "app", None)
        if app is None:
            return None
        try:
            return Path(str(app.slicerHome)).resolve()
        except Exception:
            return None

    def _default_slicerdmri_dir(self) -> Optional[Path]:
        slicer_base = self._default_slicer_base_dir()
        if slicer_base is None:
            return None
        extensions_root = slicer_base / "slicer.org"
        if not extensions_root.is_dir():
            return None
        for candidate in sorted(extensions_root.glob("Extensions-*/*")):
            if candidate.is_dir() and candidate.name == "SlicerDMRI":
                return candidate.resolve()
        return None

    def _jump_to_module(self, target: str) -> None:
        """Handle navigation button clicks."""
        if target == "DDSeg":
            try:
                slicer.util.selectModule("SlicerDSpace")
                panel = getattr(self.ui, "ddsegCollapsibleButton", None)
                if panel:
                    panel.collapsed = False
                    panel.raise_()
            except Exception:
                pass
            slicer.util.showStatusMessage("DDSeg pipeline is available in this module", 3000)
            return
        if target == "LOCAL_DDSURFER":
            try:
                slicer.util.selectModule("SlicerDSpace")
            except Exception:
                pass
            slicer.util.showStatusMessage("Jumped to DDSurfer configuration", 3000)
            return

        if target == "LOCAL_MICROSTRUCTURE":
            try:
                slicer.util.selectModule("SlicerDSpace")
                panel = getattr(self.ui, "microstructureCollapsibleButton", None)
                if panel:
                    panel.collapsed = False
                    panel.raise_()
            except Exception:
                pass
            slicer.util.showStatusMessage("Jumped to microstructure workflow", 3000)
            return

        try:
            slicer.util.selectModule(target)
            slicer.util.showStatusMessage(f"Jumped to {target}", 3000)
        except Exception:
            slicer.util.infoDisplay(f"Module {target} not installed or not yet implemented. Coming soon.")

    def setParameterNode(self, inputParameterNode) -> None:
        if self._parameterNode:
            if self._parameterNodeGuiTag is not None:
                self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            if self.hasObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply):
                self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        self._parameterNodeGuiTag = None
        if self._parameterNode:
            self._restore_common_user_settings_to_node()
            from shared.ui_advanced_layout import enforce_single_subject_gui
            enforce_single_subject_gui(self._parameterNode)
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _sync_path_widget(self, widget, desired: str) -> None:
        if not widget or widget.currentPath == desired:
            return
        blocker = QSignalBlocker(widget)
        widget.setCurrentPath(desired)
        del blocker

    def _sync_line_edit(self, widget, desired: str) -> None:
        if not widget:
            return
        current_text = widget.text() if callable(getattr(widget, "text", None)) else widget.text
        if current_text == desired:
            return
        blocker = QSignalBlocker(widget)
        widget.setText(desired)
        del blocker

    def _sync_combo_text(self, widget, desired: str, *, via_index: bool = True) -> None:
        if not widget or widget.currentText == desired:
            return
        blocker = QSignalBlocker(widget)
        if via_index:
            index = widget.findText(desired)
            if index >= 0:
                widget.setCurrentIndex(index)
        else:
            widget.setCurrentText(desired)
        del blocker

    def _sync_checkbox(self, widget, desired: bool) -> None:
        if not widget or widget.isChecked() == desired:
            return
        blocker = QSignalBlocker(widget)
        widget.setChecked(desired)
        del blocker

    def _sync_spinbox(self, widget, desired: int) -> None:
        if not widget or widget.value == desired:
            return
        blocker = QSignalBlocker(widget)
        widget.value = desired
        del blocker

    def _checkCanApply(self, caller=None, event=None) -> None:
        if getattr(self, "_syncingDetectedInputFormat", False):
            return
        sync_parameter_node_to_ui(self)
        self._refresh_selected_products_summary()
        self._refresh_advanced_run_settings()
        self.validateInputs(manual=False)
        self._refresh_run_button_state(self._refresh_topup_controls())
        self._schedule_run_history_refresh()
        self._schedule_common_user_settings_save()

    def _update_parameter_node(self, attribute: str, value: Any) -> bool:
        if not self._parameterNode:
            return False
        with slicer.util.NodeModify(self._parameterNode):
            setattr(self._parameterNode, attribute, value)
        return True

    def _onUnifiedTorchDeviceChanged(self, value_or_index: Any) -> None:
        if not self._parameterNode:
            return
        combo = getattr(self.ui, "unifiedTorchDeviceComboBox", None)
        if isinstance(value_or_index, int):
            if combo is None or value_or_index < 0:
                return
            raw = combo.itemData(value_or_index) if hasattr(combo, "itemData") else None
            if not raw:
                raw = getattr(combo, "currentText", "auto")
        else:
            raw = value_or_index
        device = str(raw or "auto").strip().lower() or "auto"
        if device.startswith("auto"):
            device = "auto"
        elif device.startswith("cuda"):
            device = device.split()[0]
        elif device != "cpu":
            device = "auto"
        with slicer.util.NodeModify(self._parameterNode):
            self._parameterNode.device = device
            self._parameterNode.ddsegDevice = device
            self._parameterNode.brainMaskDevice = "cuda" if device.startswith("cuda") else device
        self._lastEnvironmentResult = None
        self._lastEnvironmentSignature = None
        self._checkCanApply()

    def _onTractographyBackendChanged(self, value: str) -> None:
        self._refresh_advanced_run_settings(backend=(value or "").strip() or None)

    def _onBundleBackendChanged(self, value: str) -> None:
        self._refresh_bundle_backend_highlight(bundle_backend=(value or "").strip() or None)

    def _onBatchRootPathChanged(self, value: str) -> None:
        self._update_parameter_node("batchRoot", str(value or "").strip())

    def _onWmaInputTractographyPathChanged(self, value: str) -> None:
        self._update_parameter_node("wmaInputTractography", str(value or "").strip())

    def _default_run_name_for_backend(self, backend: str) -> str:
        return default_run_name_for_backend(backend)

    def _is_auto_generated_run_name(self, run_name: str) -> bool:
        return is_auto_generated_run_name(run_name)

    def _refresh_advanced_run_settings(self, *, backend: Optional[str] = None) -> None:
        node = self._parameterNode
        has_compact_categories = hasattr(self, "_advancedCategorySelector")
        if node:
            groups = (
                (not has_compact_categories and bool(node.targetMicrostructure), ("orchestrationMicrostructureStagesLabel", "orchestrationMicrostructureStageOptions")),
                (bool(node.targetMicrostructure), ('parallelModelsLabel','parallelModelsSpinBox')),
                (bool(node.targetMicrostructure) and self._microstructure_dki_enabled(),
                 ("orchestrationDkiFitMethodLabel", "orchestrationDkiFitMethodComboBox")),
                (bool(node.targetMicrostructure) and self._microstructure_ifw_enabled(),
                 ("orchestrationIfwEchoTimeLabel", "orchestrationIfwEchoTimeLineEdit",
                  "orchestrationIfwT2TissueLabel", "orchestrationIfwT2TissueLineEdit",
                  "orchestrationIfwT2CsfLabel", "orchestrationIfwT2CsfLineEdit")),
                (not has_compact_categories and bool(node.targetBundles), ("orchestrationBundleBackendLabel", "orchestrationBundleBackendComboBox")),
                (not has_compact_categories and bool(node.targetTractography or node.targetBundles or node.targetConnectome), ("orchestrationTractographyBackendLabel", "orchestrationTractographyBackendComboBox")),
                (False, ("batchModeLabel", "batchModeCheckBox", "batchRootLabel", "batchRootPath", "batchSizeLabel", "batchSizeSpinBox", "batchArrangeInputsLabel", "batchArrangeInputsComboBox")),
            )
            for visible, names in groups:
                for name in names:
                    control = getattr(self.ui, name, None)
                    if control and control.visible != visible:
                        control.visible = visible
            legacy_combo = getattr(self.ui, "orchestrationMicrostructureStagesComboBox", None)
            if legacy_combo is not None and legacy_combo.visible:
                legacy_combo.visible = False
        backend = (backend or self._tractography_backend_text() or "csd_act").strip() or "csd_act"
        run_name_edit = getattr(self.ui, "orchestrationRunNameLineEdit", None)
        current_run_name = self._line_edit_text(run_name_edit) if run_name_edit else ""
        desired_default = self._default_run_name_for_backend(backend)
        if run_name_edit and self._is_auto_generated_run_name(current_run_name) and current_run_name != desired_default:
            run_name_edit.setText(desired_default)
            self._update_parameter_node("orchestrationRunName", desired_default)

        streamline_label = getattr(self.ui, "orchestrationSelectStreamlinesLabel", None)
        streamline_spinbox = getattr(self.ui, "orchestrationSelectStreamlinesSpinBox", None)
        if not streamline_label or not streamline_spinbox:
            self._refresh_bundle_backend_highlight()
            return

        control_state = streamline_control_state(backend)
        if streamline_spinbox.enabled != control_state.enabled:
            streamline_spinbox.enabled = control_state.enabled
        streamlines_visible = control_state.enabled and (not node or bool(node.targetTractography or node.targetConnectome))
        if streamline_spinbox.visible != streamlines_visible:
            streamline_spinbox.visible = streamlines_visible
        if streamline_label.visible != streamlines_visible:
            streamline_label.visible = streamlines_visible
        if streamline_label.text != control_state.label:
            streamline_label.text = control_state.label
        if streamline_label.toolTip != control_state.tooltip:
            streamline_label.toolTip = control_state.tooltip
        if streamline_spinbox.toolTip != control_state.tooltip:
            streamline_spinbox.toolTip = control_state.tooltip
        self._refresh_bundle_backend_highlight()
        from shared.ui_advanced_layout import refresh_advanced_categories
        refresh_advanced_categories(self)

    def _refresh_bundle_backend_highlight(self, *, bundle_backend: Optional[str] = None) -> None:
        from shared.ui_styles import is_dark_mode, set_stylesheet_if_changed

        dark = is_dark_mode(getattr(self, "_uiWidget", None))
        active_sections = active_bundle_backend_sections(bundle_backend or self._bundle_backend_text())
        node = self._parameterNode
        for name in ("wmqlQueryPresetLabel", "wmqlQueryPresetComboBox", 'wmqlQuerySelectionLabel', 'wmqlQuerySelectionLineEdit'):
            control = getattr(self.ui, name, None)
            if control is not None:
                from shared.algorithm_selection import algorithms, BUNDLES
                target_vis = bool(node and node.targetBundles and "wmql" in algorithms(bundle_backend or self._bundle_backend_text(), BUNDLES, "wmql"))
                if control.visible != target_vis:
                    control.visible = target_vis

        wma_active = "wma" in active_sections
        wma_input_widgets = (
            getattr(self.ui, "wmaAtlasLabel", None),
            getattr(self.ui, "wmaAtlasComboBox", None),
            getattr(self.ui, "wmaInputTractographyLabel", None),
            getattr(self.ui, "wmaInputTractographyPath", None),
        )
        for widget in wma_input_widgets:
            if widget is not None:
                if widget.enabled != wma_active:
                    widget.enabled = wma_active
                if widget.visible != wma_active:
                    widget.visible = wma_active

        label_groups = getattr(self, "_environmentBackendSectionLabels", {}) or {}
        row_groups = getattr(self, "_environmentBackendSectionRows", {}) or {}
        for key in ("wma", "tractcloud"):
            title_label, description_label = label_groups.get(key, (None, None))
            active = key in active_sections
            title_style, description_style = optional_backend_label_styles(active, dark=dark)
            if title_label:
                set_stylesheet_if_changed(title_label, title_style)
            if description_label:
                set_stylesheet_if_changed(description_label, description_style)

            for widgets in row_groups.get(key, []):
                for widget in widgets:
                    if widget is None:
                        continue
                    if hasattr(widget, "setStyleSheet"):
                        style = optional_backend_widget_style(widget.__class__.__name__, active, dark=dark)
                        if style is not None:
                            set_stylesheet_if_changed(widget, style)

    def appendLog(self, text: str) -> None:
        append_log(self, text)

    def _trim_log_widget(self, log_widget) -> None:
        limit = int(getattr(self, "_logLineLimit", 4000) or 4000)
        if limit <= 0:
            return
        block_count = 0
        try:
            block_count = int(log_widget.document().blockCount())
        except Exception:
            pass
        if block_count and block_count <= limit + 200:
            return
        try:
            lines = str(log_widget.toPlainText()).splitlines()
        except Exception:
            return
        if len(lines) <= limit + 200:
            return
        log_widget.setPlainText("\n".join(lines[-limit:]))

    def _onClearLogClicked(self) -> None:
        if hasattr(self.ui, "logPlainTextEdit"):
            self.ui.logPlainTextEdit.clear()

    def _selected_workflows(self) -> list[str]:
        node = self._parameterNode
        if not node:
            return []
        return selected_workflows(self._product_target_flags())

    def _selected_orchestrator_targets(self) -> tuple[str, ...]:
        node = self._parameterNode
        if not node:
            return ()
        return selected_orchestrator_targets(self._product_target_flags())

    def _selected_closure(self):
        """Ticked targets plus every prerequisite (C1); ``None`` without a parameter node."""
        node = self._parameterNode
        if not node:
            return None
        freewater, noddi = self._microstructure_stage_flags()
        return selected_closure(
            self._product_target_flags(),
            tractography_backend=self._tractography_backend_text(),
            bundle_backend=self._bundle_backend_text(),
            microstructure_dti=self._microstructure_dti_enabled() if hasattr(self, "_microstructure_dti_enabled") else True,
            microstructure_freewater=freewater,
            microstructure_noddi=noddi,
            microstructure_dki=self._microstructure_dki_enabled(),
            microstructure_ifw=self._microstructure_ifw_enabled(),
            provided_tractography=self._provided_tractography(),
        )

    def _provided_tractography(self) -> tuple[str, ...]:
        """K2: targets whose tractogram file the user supplied (same set the CLI builds).

        The GUI has one input-tractogram parameter, ``wmaInputTractography``; there is none for
        TractCloud or SAF.
        """
        node = self._parameterNode
        if node and str(getattr(node, "wmaInputTractography", "") or "").strip():
            return ("wma",)
        return ()

    def _selected_product_targets(self) -> tuple[str, ...]:
        closure = self._selected_closure()
        return closure.targets if closure is not None else ()

    def _closure_tractography_backend_text(self) -> str:
        """Tractography algorithms after the closure (WMA adds UKF); the node text if no tracking runs."""
        closure = self._selected_closure()
        if closure is None or "tractography" not in closure.targets:
            return self._tractography_backend_text()
        from shared.algorithm_selection import choice_value
        return choice_value(closure.algorithms_text)

    def _closure_notice_text(self) -> str:
        """C4 one-line notice naming the targets that run automatically."""
        closure = self._selected_closure()
        if closure is None:
            return ""
        from shared.target_closure import closure_notice
        return closure_notice(closure)

    def _requested_product_targets(self) -> tuple[str, ...]:
        closure = self._selected_closure()
        return closure.requested if closure is not None else ()

    def _product_target_flags(self) -> ProductTargetFlags:
        node = self._parameterNode
        if not node:
            return ProductTargetFlags()
        return ProductTargetFlags(
            preprocessing=bool(node.targetPreprocessing),
            surface=bool(node.targetSurface),
            surface_postprocess=bool(node.targetSurfacePostprocess),
            parcellation=bool(node.targetParcellation),
            segmentation=bool(node.targetSegmentation),
            microstructure=bool(node.targetMicrostructure),
            tractography=bool(node.targetTractography),
            bundles=bool(node.targetBundles),
            connectome=bool(node.targetConnectome),
        )

    def _tractography_backend_text(self) -> str:
        node = self._parameterNode
        return ((node.orchestrationTractographyBackend if node else "") or "csd_act").strip() or "csd_act"

    def _bundle_backend_text(self) -> str:
        node = self._parameterNode
        return ((getattr(node, "orchestrationBundleBackend", "") if node else "") or "wmql").strip() or "wmql"

    def _microstructure_stages_text(self) -> str:
        node = self._parameterNode
        if not node:
            return "dti"
        text = (getattr(node, "orchestrationMicrostructureStages", "") or "").strip().lower()
        if text:
            return ",".join(selected_microstructure_stages(text))
        stages = ["dti"]
        if bool(getattr(node, "orchestrationRunFreeWater", False)):
            stages.append("freewater")
        if bool(getattr(node, "orchestrationRunNoddi", False)):
            stages.append("noddi")
        if bool(getattr(node, "orchestrationRunDki", False)):
            stages.append("dki")
        if bool(getattr(node, "orchestrationRunIfw", False)):
            stages.append("ifw")
        return ",".join(stages)

    def _microstructure_dti_enabled(self) -> bool:
        node = self._parameterNode
        if not node:
            return True
        text = (getattr(node, "orchestrationMicrostructureStages", "") or "").strip().lower()
        return microstructure_dti_enabled(text if text else "dti")

    def _microstructure_stage_flags(self) -> tuple[bool, bool]:
        node = self._parameterNode
        return microstructure_stage_flags(
            self._microstructure_stages_text(),
            legacy_freewater=bool(getattr(node, "orchestrationRunFreeWater", False)) if node else False,
            legacy_noddi=bool(getattr(node, "orchestrationRunNoddi", False)) if node else False,
        )

    def _selected_environment_targets(self) -> tuple[str, ...]:
        node = self._parameterNode
        if not node:
            return ()
        return selected_environment_targets(self._selected_product_targets())

    def _microstructure_dki_enabled(self) -> bool:
        return microstructure_optional_stage_flags(self._microstructure_stages_text())[2]

    def _microstructure_ifw_enabled(self) -> bool:
        node = self._parameterNode
        return microstructure_ifw_enabled(
            self._microstructure_stages_text(),
            legacy_ifw=bool(getattr(node, "orchestrationRunIfw", False)) if node else False,
        )

    def _ifw_echo_time_ms(self) -> Optional[float]:
        """Echo time (ms) typed by the user; None when iFW is off or the field is empty. Raises ValueError if invalid."""
        node = self._parameterNode
        if not node or not self._microstructure_ifw_enabled():
            return None
        return parse_ifw_echo_time_ms(getattr(node, "orchestrationIfwEchoTimeMs", ""))

    def _apply_shared_inputs_to_workflow_fields(self) -> None:
        """Populate empty product-specific fields from the shared subject setup."""
        node = self._parameterNode
        if not node:
            return
        subject_id = self._effective_subject_id()
        subject_root = (node.inputDirectory or "").strip()
        with slicer.util.NodeModify(node):
            if subject_id:
                if not (node.ddparcelSubjectId or "").strip():
                    node.ddparcelSubjectId = subject_id
                if not (node.microstructureSubjectId or "").strip():
                    node.microstructureSubjectId = subject_id
            if subject_root:
                if not (node.ddparcelInputRoot or "").strip():
                    node.ddparcelInputRoot = subject_root
                if not (node.microstructureInputRoot or "").strip():
                    node.microstructureInputRoot = subject_root
                if not (node.ddsegDwiPath or "").strip():
                    node.ddsegDwiPath = subject_root

    def _slicer_launcher_program(self) -> str:
        launcher_path = getattr(slicer.app, "launcherExecutableFilePath", "")
        launcher_path = launcher_path() if callable(launcher_path) else launcher_path
        launcher_path = str(launcher_path or "").strip()
        if launcher_path and Path(launcher_path).name != "SlicerApp-real":
            return launcher_path
        app_path = getattr(slicer.app, "applicationFilePath", "")
        app_path = app_path() if callable(app_path) else app_path
        app_path = str(app_path or "").strip()
        if app_path:
            app_candidate = Path(app_path)
            if app_candidate.name == "SlicerApp-real":
                for candidate in (
                    app_candidate.parent.parent / "Slicer",
                    app_candidate.parent / "Slicer",
                ):
                    if candidate.is_file():
                        return candidate.as_posix()
            return app_path
        slicer_home = str(getattr(slicer.app, "slicerHome", "") or "").strip()
        if slicer_home:
            for candidate in (Path(slicer_home) / "Slicer", Path(slicer_home).parent / "Slicer"):
                if candidate.is_file():
                    return candidate.as_posix()
        return "Slicer"

    def _build_background_cli_args(self, common_run_kwargs: dict[str, Any], external_paths: dict[str, Optional[Path]]) -> list[str]:
        node = self._parameterNode
        repo_root = Path(__file__).resolve().parent.parent
        return build_background_cli_args(
            common_run_kwargs,
            external_paths,
            repo_root=repo_root,
        )

    def _build_batch_background_cli_args(
        self,
        common_run_kwargs: dict[str, Any],
        external_paths: dict[str, Optional[Path]],
        *,
        batch_root: Path,
        batch_size: int,
        arrange_inputs: str,
    ) -> list[str]:
        node = self._parameterNode
        repo_root = Path(__file__).resolve().parent.parent
        return build_batch_background_cli_args(
            common_run_kwargs,
            external_paths,
            repo_root=repo_root,
            batch_root=batch_root,
            batch_size=batch_size,
            arrange_inputs=arrange_inputs,
        )

    def _reset_result_scene_actions(self) -> None:
        self._latestResultManifestPath = None
        files = getattr(self, "_runOutputFiles", None)
        if files is not None:
            files.clear()
        for button in getattr(self, "_resultSceneButtons", []):
            button.enabled = False
        review = getattr(self, "_reviewStateLabel", None)
        if review:
            review.text = "Recorded QC: not available yet\nVisual review: pending"

    def _startBackgroundPipeline(
        self,
        common_run_kwargs: dict[str, Any],
        external_paths: dict[str, Optional[Path]],
        selected_workflows: list[str],
        *,
        launch_args: Optional[list[str]] = None,
        run_context: Optional[dict] = None,
    ) -> None:
        if getattr(self, "_backgroundProcess", None) is not None:
            process = self._backgroundProcess
            if qprocess_is_running(process):
                slicer.util.infoDisplay("A SlicerDSpace job is already running.", parent=self.parent)
                return

        process = qt.QProcess(self.parent)
        process.setWorkingDirectory(Path(__file__).resolve().parent.parent.as_posix())
        process.setProcessChannelMode(qt.QProcess.MergedChannels)
        process_environment = qt.QProcessEnvironment.systemEnvironment()
        for key, value in background_process_env_overrides().items():
            process_environment.insert(key, value)
        process.setProcessEnvironment(process_environment)
        process.connect("readyReadStandardOutput()", self._onBackgroundProcessReadyRead)
        process.connect("finished(int,QProcess::ExitStatus)", self._onBackgroundProcessFinished)
        process.connect("errorOccurred(QProcess::ProcessError)", self._onBackgroundProcessError)

        program = self._slicer_launcher_program()
        args = launch_args if launch_args is not None else self._build_background_cli_args(common_run_kwargs, external_paths)
        # Run the CLI in its own process group so Cancel can kill the whole
        # descendant tree (MRtrix, FreeSurfer, dcm2niix, ...), not just the
        # launcher process. See shared/process_group.py.
        program, args, grouped = grouped_launch_command(program, args)
        self._reset_result_scene_actions()
        self._backgroundProcess = process
        self._backgroundProcessGrouped = grouped
        self._backgroundProcessGroupId = None
        self._backgroundOutputBuffer = ""
        self._backgroundManifestPath = None
        self._backgroundCancelRequested = False
        self._backgroundRunContext = build_background_run_context(common_run_kwargs, selected_workflows)
        self._backgroundRunContext.update(run_context or {})
        self.appendLog("[Background] Starting: " + program + " " + " ".join(args))
        self._setPipelineActive(True)
        self._set_run_progress_message("Background execution submitted.")
        manifest_path = getattr(self, "_pendingRunManifestPath", None)
        self._pendingRunManifestPath = None
        start_progress_polling(self, manifest_path)
        set_run_state(self, "running")
        process.start(program, args)
        if not process.waitForStarted(5000):
            stop_progress_polling(self)
            self._setPipelineActive(False)
            set_run_state(self, "failed")
            self._setStatus("error", "Failed to start background job")
            self.appendLog("[Background] Failed to start process.")
        else:
            self._backgroundProcessGroupId = self._capture_background_process_group_id(process, grouped)

    def _startBackgroundBatchPipeline(
        self,
        common_run_kwargs: dict[str, Any],
        external_paths: dict[str, Optional[Path]],
        selected_workflows: list[str],
        *,
        batch_root: Path,
        batch_size: int,
        arrange_inputs: str,
    ) -> None:
        if getattr(self, "_backgroundProcess", None) is not None:
            process = self._backgroundProcess
            if qprocess_is_running(process):
                slicer.util.infoDisplay("A SlicerDSpace job is already running.", parent=self.parent)
                return

        process = qt.QProcess(self.parent)
        process.setWorkingDirectory(Path(__file__).resolve().parent.parent.as_posix())
        process.setProcessChannelMode(qt.QProcess.MergedChannels)
        process_environment = qt.QProcessEnvironment.systemEnvironment()
        for key, value in background_process_env_overrides().items():
            process_environment.insert(key, value)
        process.setProcessEnvironment(process_environment)
        process.connect("readyReadStandardOutput()", self._onBackgroundProcessReadyRead)
        process.connect("finished(int,QProcess::ExitStatus)", self._onBackgroundProcessFinished)
        process.connect("errorOccurred(QProcess::ProcessError)", self._onBackgroundProcessError)

        program = self._slicer_launcher_program()
        args = self._build_batch_background_cli_args(
            common_run_kwargs,
            external_paths,
            batch_root=batch_root,
            batch_size=batch_size,
            arrange_inputs=arrange_inputs,
        )
        # Run the CLI in its own process group so Cancel can kill the whole
        # descendant tree, not just the launcher process.
        program, args, grouped = grouped_launch_command(program, args)
        self._reset_result_scene_actions()
        self._backgroundProcess = process
        self._backgroundProcessGrouped = grouped
        self._backgroundProcessGroupId = None
        self._backgroundOutputBuffer = ""
        self._backgroundManifestPath = None
        self._backgroundCancelRequested = False
        self._backgroundRunContext = build_background_run_context(
            common_run_kwargs,
            selected_workflows,
            batch_root=batch_root,
            batch_size=batch_size,
        )
        self.appendLog("[Background] Starting batch: " + program + " " + " ".join(args))
        self._setPipelineActive(True)
        self._set_run_progress_message(f"Batch execution submitted for {batch_root}.")
        # Batch runs write their manifest only at the end: show a busy bar.
        start_progress_polling(self, None)
        set_run_state(self, "running")
        process.start(program, args)
        if not process.waitForStarted(5000):
            self._setPipelineActive(False)
            set_run_state(self, "failed")
            self._setStatus("error", "Failed to start batch job")
            self.appendLog("[Background] Failed to start batch process.")
        else:
            self._backgroundProcessGroupId = self._capture_background_process_group_id(process, grouped)

    def _onBackgroundProcessReadyRead(self) -> None:
        process = getattr(self, "_backgroundProcess", None)
        if process is None:
            return
        data = self._qbytearray_to_text(process.readAllStandardOutput())
        self._backgroundOutputBuffer += data
        while "\n" in self._backgroundOutputBuffer:
            line, self._backgroundOutputBuffer = self._backgroundOutputBuffer.split("\n", 1)
            self._handleBackgroundOutputLine(line.rstrip())

    def _qbytearray_to_text(self, value) -> str:
        return qbytearray_to_text(value)

    def _handleBackgroundOutputLine(self, line: str) -> None:
        if not line:
            return
        self.appendLog(line)
        if line.startswith("[Orchestrator] Running "):
            keys = self._task_keys_for_running_line(line)
            self._update_run_status_table_item(keys, "running", line.replace("[Orchestrator] ", "", 1))
        path_text = parse_manifest_line(line)
        if path_text:
            path = Path(path_text)
            if path.is_file():
                self._backgroundManifestPath = path

    def _extract_background_manifest_path(self) -> Optional[Path]:
        return extract_background_manifest_path(
            getattr(self, "_backgroundOutputBuffer", "") or "",
            explicit_path=getattr(self, "_backgroundManifestPath", None),
        )

    def _onBackgroundProcessFinished(self, exit_code: int, exit_status) -> None:
        if getattr(self, "_backgroundOutputBuffer", ""):
            tail = self._backgroundOutputBuffer
            self._backgroundOutputBuffer = ""
            for line in tail.splitlines():
                self._handleBackgroundOutputLine(line.rstrip())
        cancelled = bool(getattr(self, "_backgroundCancelRequested", False))
        self._poll_run_progress()
        stop_progress_polling(self)
        manifest_path = self._extract_background_manifest_path()
        manifest_available = bool(manifest_path and manifest_path.is_file())
        outcome = classify_background_run_outcome(
            exit_code=exit_code,
            cancelled=cancelled,
            manifest_available=manifest_available,
        )
        if outcome == "cancelled":
            cancel_manifest = self._write_cancelled_manifest()
            self._mark_unfinished_rows("cancelled", "User cancelled the background job.")
            self.appendLog(f"[Background] Cancelled. Manifest: {cancel_manifest}")
            self._setStatus("warning", "Pipeline cancelled")
            self._set_latest_report_path(None)
            self._set_latest_output_tree_path(None)
            self._set_run_progress_message(f"Pipeline cancelled. Manifest: {cancel_manifest}")
            self._set_run_summary_message("Status: cancelled | User cancelled the background job.", "cancelled")
            set_run_state(self, "cancelled")
            # Keep the live rows (completed stages + cancelled ones) visible.
            self._refresh_run_history(select_path=cancel_manifest)
            self._tableShowsLiveRun = False
        elif outcome == "manifest":
            # A manifest was written even though the process may have exited
            # non-zero (e.g. exit code 2 for "some tasks failed"): it is
            # authoritative, so load per-task status from it rather than
            # treating a non-zero exit as a blanket process failure.
            self._load_background_manifest_result(manifest_path)
        elif outcome == "process_failed":
            self._mark_unfinished_rows("failed", f"Background process exited with code {exit_code}")
            self._setStatus("error", f"Pipeline failed: exit code {exit_code}")
            self._set_latest_report_path(None)
            self._set_latest_output_tree_path(None)
            self._set_run_progress_message(f"Pipeline failed: exit code {exit_code}")
            self._set_run_summary_message(f"Status: failed | Background process exited with code {exit_code}.", "failed")
            set_run_state(self, "failed")
        else:
            self._mark_unfinished_rows("failed", "Background process finished without a manifest path.")
            self._setStatus("error", "Pipeline failed: no manifest")
            self._set_latest_report_path(None)
            self._set_latest_output_tree_path(None)
            self._set_run_progress_message("Pipeline failed: no manifest path was reported.")
            self._set_run_summary_message("Status: failed | Background process finished without a manifest path.", "failed")
            set_run_state(self, "failed")
        self._backgroundProcess = None
        self._backgroundCancelRequested = False
        # Loading volumes/fibers can process Qt events. Keep Run disabled until
        # the old callback has released its handle, so it cannot erase a new job.
        self._setPipelineActive(False, preserve_status=True)

    def _load_background_manifest_result(self, manifest_path: Path) -> None:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self._mark_unfinished_rows("failed", f"Unable to read manifest: {exc}")
            self._setStatus("error", "Pipeline failed: invalid manifest")
            return
        if manifest.get("kind") == "slicerdsurfer_batch_run":
            self._load_background_batch_manifest_result(manifest_path, manifest)
            self._tableShowsLiveRun = False
            set_run_state(self, "failed" if rows_from_batch_manifest(manifest_path, manifest)[1]["failed"] else "completed")
            self._refresh_run_history(select_path=manifest_path)
            return
        finished_state = run_state_from_manifest_status(str(manifest.get("status") or ""))
        set_run_state(self, finished_state if finished_state not in {"idle", "running"} else "completed")
        rows, counts = rows_from_single_subject_manifest(manifest)
        blocked = counts["blocked"]
        failed = counts["failed"]
        completed_like = counts["completed_like"]
        self._set_run_status_table_rows(rows)
        if failed:
            self._setStatus("error", f"Completed with {failed} failed task(s)")
        elif blocked and completed_like:
            self._setStatus("warning", f"Partial results: {blocked} blocked task(s)")
        elif blocked:
            self._setStatus("error", f"Completed with {blocked} blocked task(s)")
        else:
            self._setStatus("warning", "Execution completed; review QC and images")
        self._latestResultManifestPath = manifest_path
        for button in getattr(self, "_resultSceneButtons", []):
            button.enabled = True
        review = getattr(self, "_reviewStateLabel", None)
        if review:
            from shared.result_scene import recorded_qc_summary, review_state_text
            review.text = review_state_text(recorded_qc_summary(manifest), issues_found=bool(failed or blocked))
        report_path = ""
        report_paths = manifest.get("report_paths") or {}
        if isinstance(report_paths, dict):
            report_path = str(report_paths.get("report_html_latest") or report_paths.get("report_html") or "")
        self._set_latest_report_path(report_path)
        output_tree_path = ""
        output_tree_report_paths = manifest.get("output_tree_report_paths") or {}
        if isinstance(output_tree_report_paths, dict):
            output_tree_path = str(output_tree_report_paths.get("report_html_latest") or output_tree_report_paths.get("report_html") or "")
        self._set_latest_output_tree_path(output_tree_path)
        self._set_run_summary_from_manifest(manifest, fallback=f"Run finished. Manifest: {manifest_path}")
        message = f"Run finished. Manifest: {manifest_path}"
        if report_path:
            message += f" Report: {report_path}"
        if output_tree_path:
            message += f" Output tree: {output_tree_path}"
        self._set_run_progress_message(message)
        self._tableShowsLiveRun = False
        self._refresh_run_history(select_path=manifest_path)

    def _manage_result_scene(self, mode: str) -> None:
        if not self.logic:
            return
        manifest, error = validated_selected_manifest(self)
        if error or manifest is None:
            show_actionable_error_dialog(
                parent=self.parent,
                title="Cannot load this run",
                what="The selected run could not be loaded into the scene.",
                why=error,
                next_step=(
                    "Open Results and QC again to rescan run folders, or choose “Add a run folder…” "
                    "in the Run list."
                ),
                auto_fix="Not available; no scene nodes were changed.",
                details=str(getattr(self, "_latestResultManifestPath", "") or "no run selected"),
                actions=(("Rescan", self._refresh_run_history), ("Add a run folder…", self._on_browse_run_directory_clicked)),
            )
            return
        if mode in {"replace", "clear"} and not slicer.util.confirmYesNoDisplay(
                "Remove scene nodes loaded for this run? Source files will be kept.", parent=self.parent):
            return
        if mode == "edit_tissue":
            from shared.result_scene import owned_nodes
            from shared.segmentation_review import editable_tissue_copy
            sources = [node for node in owned_nodes(slicer, str(Path(manifest).resolve()))
                       if node.IsA("vtkMRMLLabelMapVolumeNode") and
                       str(node.GetAttribute("SlicerDDSurfer.Product") or "").startswith("segmentation.")]
            if not sources:
                slicer.util.infoDisplay("Load this run's DDSeg tissue labels first.", parent=self.parent)
                return
            try:
                copy = editable_tissue_copy(slicer, sources[-1])
                slicer.util.selectModule("SegmentEditor")
                editor = slicer.modules.segmenteditor.widgetRepresentation().self().editor
                editor.setSegmentationNode(copy)
                self.appendLog("[Review] Opened a separate tissue edit copy; original files and QC are unchanged.")
            except Exception as exc:
                slicer.util.errorDisplay(str(exc), parent=self.parent)
            return
        if mode == "clear":
            count = self.logic.clearResultsFromManifest(Path(manifest))
            self.appendLog(f"[Scene] Removed {count} owned node(s); files unchanged.")
            return
        result = self.logic.loadKeyResultsFromManifest(Path(manifest), mode=mode)
        self._set_run_progress_message(
            f"Loaded {len(result['loaded'])}; reused {len(result['reused'])}; "
            f"not loaded {len(result['skipped'])}; deferred {len(result['deferred'])}. See log for details.")
        for item in result["skipped"]:
            self.appendLog("[Scene] " + str(item))

    def _load_background_batch_manifest_result(self, manifest_path: Path, manifest: dict[str, Any]) -> None:
        rows, counts = rows_from_batch_manifest(manifest_path, manifest)
        failed = counts["failed"]
        completed = counts["completed"]
        skipped = counts["skipped"]
        total = counts["total"]
        unit = 'experiment job(s)' if manifest.get('experiment') else 'subject(s)'
        self._set_run_status_table_rows(rows)
        if failed:
            self._setStatus("error", f"Batch failed: {failed} {unit}")
        else:
            self._setStatus("warning", "Execution completed; review individual runs and QC")
        self._set_latest_report_path(None)
        self._set_latest_output_tree_path(None)
        self._set_run_progress_message(f"Batch finished. Manifest: {manifest_path}")
        self._set_run_summary_message(
            f"Status: batch completed | Subjects: {total} | Completed: {completed} | "
            f"Failed: {failed} | Skipped: {skipped} | Manifest: {manifest_path}",
            "failed" if failed else "completed",
        )

    def _onBackgroundProcessError(self, error) -> None:
        self.appendLog(f"[Background] QProcess error: {error}")

    def _mark_unfinished_rows(self, status: str, reason: str) -> None:
        mark_unfinished_rows(self, status, reason)

    def _write_cancelled_manifest(self) -> Path:
        context = getattr(self, "_backgroundRunContext", {}) or {}
        experiment_path = context.get('experiment_manifest_path')
        if experiment_path and Path(experiment_path).is_file():
            try:
                path = Path(experiment_path)
                payload = json.loads(path.read_text(encoding='utf-8'))
                if payload.get('experiment') and payload.get('kind') == 'slicerdsurfer_batch_run':
                    from scripts.run_experiments import write_record
                    payload.update(status='cancelled', ended_at=datetime.now(timezone.utc).isoformat())
                    write_record(path, payload)
                    return path
            except (OSError, ValueError) as exc:
                self.appendLog('[Experiments] Could not update cancellation record: ' + str(exc))
        manifest_path, payload = cancelled_manifest_path_and_payload(
            context,
            cancelled_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return manifest_path

    def _onCancelRunClicked(self) -> None:
        job = getattr(self, "_setupJob", None)
        if job:
            job.cancel()
            return
        process = getattr(self, "_backgroundProcess", None)
        if process is None:
            return
        try:
            if process.state() == qt.QProcess.NotRunning:
                return
        except Exception:
            return
        self._backgroundCancelRequested = True
        grouped = bool(getattr(self, "_backgroundProcessGrouped", False))
        pgid = getattr(self, "_backgroundProcessGroupId", None)
        if grouped and pgid:
            groups = getattr(self, "_pendingCancelledProcessGroups", set())
            groups.add(pgid)
            self._pendingCancelledProcessGroups = groups
        self.appendLog("[Background] Cancellation requested by user.")
        self._set_run_progress_message("Cancelling background job...")
        self._signal_background_process(process, force=False)
        # Capture grouped/pgid now, not inside the lambda: if a new
        # background run were started during the grace period (replacing
        # self._backgroundProcessGrouped/self._backgroundProcessGroupId),
        # reading them fresh when the timer fires would kill the *new*
        # run's process group instead of the one actually being cancelled.
        grouped = bool(getattr(self, "_backgroundProcessGrouped", False))
        pgid = getattr(self, "_backgroundProcessGroupId", None)
        # Allow DDSurfer's 10-second stage-tree cleanup to finish before SIGKILL.
        qt.QTimer.singleShot(
            15000, lambda: self._kill_background_process_if_running(process, grouped=grouped, pgid=pgid)
        )

    def _capture_background_process_group_id(self, process, grouped: bool) -> Optional[int]:
        # Capture the pgid once, right here, at launch time, and reuse it for
        # every later kill attempt -- see capture_process_group_id()'s own
        # docstring in shared/process_group.py for why it deliberately does
        # not query os.getpgid() at kill time (or here: getpgid would race
        # the launched process's own setsid()/setpgrp() call).
        if not grouped:
            return None
        try:
            pid = int(process.processId())
        except Exception:
            return None
        return capture_process_group_id(pid)

    def _signal_background_process(self, process, *, force: bool) -> None:
        # Signal the whole process group/tree (see shared/process_group.py),
        # not just the top-level QProcess: that process is typically a
        # launcher whose actual pipeline work happens in child subprocesses
        # (MRtrix, FreeSurfer, dcm2niix, ...) that QProcess.terminate()/.kill()
        # never reach on their own.
        try:
            pid = int(process.processId())
        except Exception:
            pid = 0
        grouped = bool(getattr(self, "_backgroundProcessGrouped", False))
        pgid = getattr(self, "_backgroundProcessGroupId", None)
        if pid > 0 or (grouped and pgid):
            try:
                kill_process_tree(pid, grouped=grouped, force=force, pgid=pgid)
            except OSError as exc:
                # A Qt slot must not propagate e.g. PermissionError; the
                # forced follow-up timer still gets its chance to kill.
                self.appendLog(f"[Run] Could not signal the background process group: {exc}")
        elif process:
            process.kill() if force else process.terminate()

    def _kill_background_process_if_running(self, process, *, grouped: bool, pgid: Optional[int]) -> None:
        if grouped and pgid:
            # The launcher process itself may have already exited by the
            # time this forced follow-up fires -- routine, since launchers
            # exit well before the descendants they spawned (MRtrix,
            # FreeSurfer, ...) finish -- so this must not gate on
            # process.state(): killpg() targets the recorded process group
            # directly (which also reaches the launcher itself, if it is
            # still alive) and shared/process_group.py already ignores
            # ESRCH on its own. Still guard the call itself: a Qt timer
            # callback must not propagate an unexpected exception (e.g.
            # PermissionError, if the pgid were somehow reused by another
            # user's process in the meantime).
            try:
                kill_process_tree(0, grouped=True, force=True, pgid=pgid)
            except Exception:
                pass
            getattr(self, "_pendingCancelledProcessGroups", set()).discard(pgid)
            return
        try:
            if process and process.state() != qt.QProcess.NotRunning:
                self._signal_background_process(process, force=True)
        except Exception:
            pass

    def _environment_doctor_result_for_run(self, doctor_kwargs: dict, doctor_signature: tuple):
        doctor_result = self._cached_environment_result(doctor_signature)
        try:
            if doctor_result is None:
                doctor_result = self.logic.runEnvironmentDoctor(**doctor_kwargs)
                self._cache_environment_result(doctor_result, doctor_signature)
            else:
                self.appendLog("[Environment] Reusing recent dependency check; inputs and selected targets are unchanged.")
        except Exception as exc:
            self.appendLog(f"Environment check failed: {exc}")
            self._show_actionable_error(
                title="Environment Check Failed",
                what="SlicerDSpace could not complete the dependency preflight scan.",
                why="A configured path, Python executable, environment variable, or external probe raised an exception.",
                next_step="Check the External Runtime Paths values, clear stale paths, then run Check environment dependencies again.",
                auto_fix="Python packages can be installed on demand after a successful scan; external tools are never changed silently.",
                details=exc,
            )
            self._setStatus("error", "Environment check failed")
            return None
        self._update_environment_result_label(doctor_result)
        self._update_environment_path_status_labels(doctor_result)
        self._latestEnvironmentReportDir = self._environment_report_dir_from_kwargs(doctor_kwargs)
        return doctor_result

    def _environment_report_dir_from_kwargs(self, doctor_kwargs: dict) -> Optional[Path]:
        try:
            subject_root = doctor_kwargs.get("subject_root")
            output_root = doctor_kwargs.get("output_root")
            if subject_root is not None and Path(subject_root).expanduser().exists():
                from shared.io_layout import subject_layout

                return subject_layout(Path(subject_root).expanduser().resolve(), output_root=output_root).run_metadata / "environment"
            if output_root is not None:
                return Path(output_root).expanduser().resolve() / "00_run_metadata" / "environment"
            return Path(str(slicer.app.temporaryPath)) / "SlicerDSpace" / "00_run_metadata" / "environment"
        except Exception:
            return None

    def _install_missing_python_before_run(self, doctor_result, doctor_kwargs: dict, doctor_signature: tuple):
        self._install_python_packages_from_environment_dialog(tuple(doctor_result.missing_python_packages))
        self.appendLog("[Setup] After installation, review dependencies and click Run SlicerDSpace again.")
        return None

    def _onRunSelectedClicked(self) -> None:
        if (not self._parameterNode or not self.logic or self._pipelineActive
                or getattr(self, "_selectedSetupActive", False)
                or getattr(self, "_inputImportPending", False)):
            return
        if self._prepare_inputs_widget_text("source"):
            prepared = getattr(self, "_preparedInputSelection", None)
            if prepared != self._input_import_selection():
                self._on_prepare_inputs_clicked(run_after=True)
                return
        if getattr(self, "_setupRestartRequired", False):
            self._setStatus("warning", onboarding_text(getattr(self, "_onboardingLanguage", "en"), "setup_restart_required"))
            return
        from shared.setup_resume import functional_failure_matches, pending_setup
        previous = pending_setup()
        if (getattr(self, '_setupFunctionalFailure', False) or
                (previous and previous.get('status') == 'functional_failed' and functional_failure_matches(
                    self._environment_check_signature(self._environment_check_kwargs())))):
            from shared.ui_environment_check import request_prepare_environment
            request_prepare_environment(self)
            return
        if getattr(self._parameterNode, 'experimentsEnabled', False):
            from shared.ui_experiments import submit
            submit(self)
            return
        # Validate before saving settings, checking dependencies or starting a job.
        from shared.ui_advanced_layout import enforce_single_subject_gui
        enforce_single_subject_gui(self._parameterNode)
        batch_mode = False
        if not self.validateInputs(manual=True):
            return
        raw_input_text = (self._parameterNode.inputDirectory or "").strip()
        if raw_input_text:
            from shared.ui_compact import sync_detected_input_format
            from shared.ui_input_inspection import inspect_input_directory

            eff_sid = self._effective_subject_id()
            inspected = inspect_input_directory(self, Path(raw_input_text).expanduser(), eff_sid, refresh=False)
            sync_detected_input_format(self, inspected.get("mode"), signature=(raw_input_text, eff_sid), force=True)
        selected = self._selected_workflows()
        if not selected:
            self._setStatus("error", self._summarise_missing(["Target products"]))
            return
        try:
            from tractography.wmql_protocol import canonical_wmql_selection
            canonical_wmql_selection(getattr(self._parameterNode, 'orchestrationWmqlSelection', ''),
                                     getattr(self._parameterNode, 'orchestrationWmqlPreset', 'official'))
        except ValueError as exc:
            self._setStatus('error', str(exc))
            return
        try:
            ifw_echo_time_ms = self._ifw_echo_time_ms()
        except ValueError as exc:
            self._setStatus("error", str(exc))
            return
        self._save_common_user_settings()

        self._apply_shared_inputs_to_workflow_fields()
        topup_validation = self._topup_validation(batch_mode=batch_mode)
        if topup_validation.errors:
            self._show_topup_error(topup_validation)
            return
        for warning in topup_validation.warnings:
            self.appendLog("[Preprocessing] WARNING: " + warning)
        subject_root_text = (self._parameterNode.inputDirectory or "").strip()
        subject_id = self._effective_subject_id()
        output_root_text = (self._parameterNode.outputDirectory or "").strip()
        surface_device = (self._parameterNode.device or "auto").strip() or "auto"
        ddseg_device = (self._parameterNode.ddsegDevice or "auto").strip() or "auto"
        run_free_water, run_noddi = self._microstructure_stage_flags()
        selected_targets = self._selected_product_targets()
        auto_install_python = bool(self._parameterNode.autoInstallPythonDependencies)
        external_paths = self._environment_dependency_path_kwargs()
        run_name = (self._parameterNode.orchestrationRunName or "Test1").strip() or "Test1"
        selection = RunSelection(
            subject_id=subject_id,
            subject_root_text=subject_root_text,
            output_root_text=output_root_text,
            selected_targets=selected_targets,
            run_name=run_name,
            tractography_backend=self._tractography_backend_text(),
            closure_tractography_backend=self._closure_tractography_backend_text(),
            requested_targets=self._requested_product_targets(),
            bundle_backend=self._bundle_backend_text(),
            microstructure_freewater=run_free_water,
            wmql_query_preset=getattr(self._parameterNode, "orchestrationWmqlPreset", "official") or "official",
            wma_atlas=getattr(self._parameterNode, "wmaAtlas", "org") or "org",
            wmql_query_selection=getattr(self._parameterNode, 'orchestrationWmqlSelection', '') or '',
            max_parallel_models=int(getattr(self._parameterNode, 'maxParallelModels', 1)),
            ukf_parameters_text=getattr(self._parameterNode, 'ukfParameters', '{}'),
            microstructure_noddi=run_noddi,
            microstructure_dki=self._microstructure_dki_enabled(),
            microstructure_ifw=self._microstructure_ifw_enabled(),
            ifw_echo_time_ms=ifw_echo_time_ms,
            dki_fit_method=getattr(self._parameterNode, "orchestrationDkiFitMethod", "MKCURVE") or "MKCURVE",
            select_streamlines=max(1, int(self._parameterNode.orchestrationSelectStreamlines)),
            ddseg_device=ddseg_device,
            surface_device=surface_device,
            connectome_backend=(self._parameterNode.connectomeBackend or "auto").strip(),
            connectome_compress_mrtrix_dense=compress_mrtrix_dense_value(
                getattr(self._parameterNode, "connectomeCompressMrtrixDenseState", "unset")),
            brain_mask_backend=(getattr(self._parameterNode, "brainMaskBackend", "pnl_cnn") or "pnl_cnn").strip(),
            brain_mask_device=(getattr(self._parameterNode, "brainMaskDevice", "auto") or "auto").strip(),
            pnl_mask_route=(getattr(self._parameterNode, "pnlMaskRoute", "auto") or "auto").strip(),
            wma_input_tractography_text=(
                getattr(self._parameterNode, "wmaInputTractography", "") or ""
            ).strip(),
            topup=topup_validation.parameters,
            preprocessing_stages=(getattr(self._parameterNode, "orchestrationPreprocessingStages", "") or "").strip(),
            connectome_metrics=(getattr(self._parameterNode, "orchestrationConnectomeMetrics", "") or "").strip(),
        )

        if batch_mode:
            batch_root_text = (getattr(self._parameterNode, "batchRoot", "") or subject_root_text).strip()
            batch_options, error = validate_batch_run_options(
                batch_root_text,
                fallback_subject_root_text=subject_root_text,
                batch_size=int(getattr(self._parameterNode, "batchSize", 1) or 1),
                arrange_inputs=getattr(self._parameterNode, "batchArrangeInputs", "none") or "none",
            )
            if error or batch_options is None:
                slicer.util.errorDisplay(error, parent=self.parent)
                return
            common_run_kwargs = build_batch_run_kwargs(selection, batch_root=batch_options.batch_root)
            self.appendLog(
                f"Submitting batch products to background runner: root={batch_options.batch_root}, "
                f"batch_size={batch_options.batch_size}, targets={', '.join(selected)}"
            )
            self._startBackgroundBatchPipeline(
                common_run_kwargs,
                external_paths,
                selected,
                batch_root=batch_options.batch_root,
                batch_size=batch_options.batch_size,
                arrange_inputs=batch_options.arrange_inputs,
            )
            return

        input_error = validate_single_subject_inputs(subject_root_text, subject_id)
        if input_error:
            slicer.util.errorDisplay(input_error, parent=self.parent)
            return
        doctor_kwargs = self._environment_check_kwargs(
            subject_root_text=subject_root_text,
            output_root_text=output_root_text,
            external_paths=external_paths,
        )
        doctor_signature = self._environment_check_signature(doctor_kwargs)
        if self._cached_environment_result(doctor_signature) is None:
            from shared.ui_environment_check import start_environment_check
            start_environment_check(self, lambda _report: self._onRunSelectedClicked())
            return
        doctor_result = self._environment_doctor_result_for_run(doctor_kwargs, doctor_signature)
        if doctor_result is None:
            return
        preflight_blockers = preflight_blockers_for_targets(doctor_result, selected_targets)
        if preflight_blockers:
            if auto_install_python:
                from shared.ui_selected_setup import prepare_selected_environment
                prepare_selected_environment(self, doctor_result)
                return
            self.ui.environmentCollapsibleButton.collapsed = False
            self._update_environment_result_label(doctor_result)
            self._update_environment_status(doctor_result)
            self.appendLog("Selected analysis needs dependencies. Use Environment > Prepare selected analysis environment.")
            self._show_environment_results_dialog(doctor_result)
            return
        common_run_kwargs = build_single_subject_run_kwargs(
            selection,
            external_paths,
            preflight_blocked_targets=preflight_blockers,
        )
        try:
            plan = self.logic.planSelectedProductOrchestration(**common_run_kwargs)
            self._update_run_plan_table(plan)
            blocked_items = blocked_plan_items(plan)
            if blocked_items:
                self.appendLog("Run plan blocked target(s): " + blocked_plan_log_text(blocked_items))
                environment_blockers = environment_blockers_from_plan(plan)
                if environment_blockers:
                    self._show_partial_run_notice(environment_blockers, runnable_targets_from_plan(plan))
        except Exception as exc:
            self.appendLog(f"Run planning failed: {exc}")
            self._show_actionable_error(
                title="Run Planning Failed",
                what="SlicerDSpace could not build the selected product run plan.",
                why="The selected targets, backend choices, or input/output paths are inconsistent.",
                next_step="Review Subject Setup, selected products, backend choices, and the output directory, then run the plan again.",
                auto_fix="Not available for invalid run configuration.",
                details=exc,
            )
            self._setStatus("error", "Run planning failed")
            return

        self.appendLog("Submitting selected products to background runner: " + ", ".join(selected))
        self._pendingRunManifestPath = manifest_path_for_plan(plan.run_plan_path, plan.run_name)
        self._startBackgroundPipeline(common_run_kwargs, external_paths, selected)

    def _show_topup_error(self, validation: TopupValidation) -> None:
        tools = getattr(self, "_workflowToolsDialog", None)
        if tools is not None:
            tools.show()
        section = getattr(self, "_advancedSettingsPanel", None)
        if section is not None:
            section.collapsed = False
        selector = getattr(self, "_advancedCategorySelector", None)
        if selector is not None:
            selector.setCurrentIndex(0)
        group = getattr(self.ui, "topupCollapsibleGroupBox", None)
        if group is not None:
            try:
                group.collapsed = False
            except Exception:
                pass
        show_actionable_error_dialog(
            parent=self.parent,
            title="Distortion-correction settings need attention",
            what="The run was not started because a topup setting is invalid.",
            why="\n".join(validation.errors),
            next_step=(
                "Open Products > Preprocessing > … > topup > Settings, "
                "and correct or clear the invalid value."
            ),
            auto_fix="Not available; nothing was written.",
            details="dwifslpreproc options: -pe_dir, -readout_time, -rpe_pair -se_epi",
        )

    def _show_partial_run_notice(self, blockers: dict[str, str], runnable_targets: tuple[str, ...]) -> None:
        message = partial_run_notice_text(blockers, runnable_targets)
        slicer.util.infoDisplay(message, windowTitle="SlicerDSpace partial run", parent=self.parent)

    def _setStatus(self, kind: str, message: str) -> None:
        self._statusMessageSource = str(message or "")
        set_status(
            self,
            kind,
            translated_ui_message(getattr(self, "_onboardingLanguage", "en"), self._statusMessageSource),
        )

    def _summarise_missing(self, missing: List[str]) -> str:
        if not missing:
            return "Ready to run."
        preview = "; ".join(missing[:3])
        if len(missing) > 3:
            preview += " ..."
        return "Waiting: " + preview

    def _setWidgetError(self, widget, has_error: bool) -> None:
        if not widget:
            return
        from shared.ui_styles import is_dark_mode, set_stylesheet_if_changed

        if not has_error:
            set_stylesheet_if_changed(widget, "")
            return
        border_color = "#FF7B72" if is_dark_mode(widget) else "#C62828"
        border_style = f"border: 1px solid {border_color};"
        class_name = widget.metaObject().className() if hasattr(widget, "metaObject") else ""
        if class_name == "ctkPathLineEdit":
            set_stylesheet_if_changed(widget, f"ctkPathLineEdit {{ {border_style} }}")
        else:
            set_stylesheet_if_changed(widget, border_style)

    def _populateDeviceCombo(self) -> None:
        combo = getattr(self.ui, "deviceComboBox", None)
        if not combo:
            return
        current = combo.currentText if combo.count else "auto"
        combo.clear()
        combo.addItem("auto")
        combo.addItem("cpu")
        combo.addItem("cuda:0")
        index = combo.findText(current)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _onValidateClicked(self):
        self._lastProcessEventsAt = monotonic()
        self._save_common_user_settings()
        return start_environment_check(self)

    def _run_environment_dependency_check(self):
        node = self._parameterNode
        if not node or not self.logic:
            self.appendLog("Environment check failed. Parameter node or logic is unavailable.")
            self._setStatus("error", "Environment check failed")
            self._update_environment_result_label(None, error="Parameter node or logic is unavailable.")
            return None

        output_dir = (node.outputDirectory or "").strip()
        subject_root_text = (node.inputDirectory or "").strip()
        subject_root = Path(subject_root_text).expanduser() if subject_root_text else None
        if subject_root is not None and not subject_root.exists():
            self.appendLog(
                "[Environment] Subject root is not available yet; dependency report will be written outside the subject tree."
            )

        self.appendLog("Starting environment dependency check...")
        external_paths = self._environment_dependency_path_kwargs()
        kwargs = self._environment_check_kwargs(
            subject_root_text=subject_root_text,
            output_root_text=output_dir,
            external_paths=external_paths,
        )
        signature = self._environment_check_signature(kwargs)
        try:
            result = self.logic.runEnvironmentDependencyCheck(**kwargs)
        except Exception as exc:
            self.appendLog(f"Environment check failed: {exc}")
            self._setStatus("error", "Environment check failed")
            self._update_environment_result_label(None, error=str(exc))
            return None

        self._cache_environment_result(result, signature)
        self._remember_environment_report_dir(subject_root_text, output_dir)
        return result

    def _update_environment_status(self, result) -> None:
        self._update_onboarding_dashboard(result)
        if result.blocker_count:
            self._setStatus("error", f"Environment: {result.blocker_count} missing required item(s)")
        else:
            ok, missing = self._evaluateInputs()
            if not ok:
                kind = "error" if getattr(self, '_inputValidationAttempted', False) else "waiting"
                self._setStatus(kind, self._summarise_missing(missing))
            elif not self._selected_workflows():
                self._setStatus("waiting", self._summarise_missing(["Target products"]))
            else:
                self._setStatus("ready", "Environment ready")

    def _missing_environment_items(self, result) -> list:
        return environment_missing_items(result)

    def _environment_summary_text(self, result) -> str:
        return environment_summary_text(result)

    def _environment_detail_lines(self, result) -> list[str]:
        return environment_detail_lines(result)

    def _update_environment_result_label(self, result, *, error: Optional[str] = None) -> None:
        self._lastEnvironmentError = error
        label = getattr(self.ui, "environmentResultLabel", None)
        if not label:
            return
        from shared.ui_styles import is_dark_mode

        dark = is_dark_mode(label)
        lang = getattr(self, "_onboardingLanguage", "en")
        card_bg = "#20262D" if dark else "#F4F7FA"
        card_border = "#343E48" if dark else "#D6DFE7"
        err_color = "#FF7B72" if dark else "#C62828"
        neutral_color = "#B8C5D0" if dark else "#546E7A"
        ok_color = "#66BB6A" if dark else "#2E7D32"
        base = (
            f"background: {card_bg}; border: 1px solid {card_border}; "
            "border-radius: 4px; padding: 5px 8px;"
        )
        if error:
            label.setText(f"Environment check failed: {error}")
            label.setStyleSheet(f"QLabel#environmentResultLabel {{ {base} color: {err_color}; }}")
            return
        if result is None:
            label.setText(onboarding_text(lang, "environment_not_checked"))
            label.setStyleSheet(f"QLabel#environmentResultLabel {{ {base} color: {neutral_color}; }}")
            return
        color = err_color if result.blocker_count else ok_color
        label.setText(self._environment_summary_text(result))
        label.setStyleSheet(f"QLabel#environmentResultLabel {{ {base} color: {color}; }}")

    def _show_environment_results_dialog(self, result) -> None:
        show_environment_results_dialog(
            owner=self,
            result=result,
            python_install_target=self._python_install_target_text(),
            on_install_all=self._onInstallAllEnvironmentItemsClicked,
            on_show_notes=lambda: self._show_environment_install_notes(result),
            on_install_item=self._onInstallEnvironmentItemClicked,
        )

    def _build_environment_item_row(self, item):
        from shared.ui_environment_dialog import build_environment_item_row

        return build_environment_item_row(item, on_install_item=self._onInstallEnvironmentItemClicked)

    def _environment_item_text(self, item) -> str:
        return environment_item_text(item)

    def _onInstallAllEnvironmentItemsClicked(self) -> None:
        result = getattr(self, "_lastEnvironmentResult", None)
        if result is None:
            slicer.util.infoDisplay("Run the environment check first.", parent=self.parent)
            return
        from shared.ui_selected_setup import prepare_selected_environment
        prepare_selected_environment(self, result)

    def _onInstallEnvironmentItemClicked(self, item) -> None:
        group = str(getattr(item, "install_group", "") or "")
        if group in {"mrtrix", "fsl"}:
            from shared.managed_runtimes import cuda_candidate, supported_platform
            if supported_platform():
                from shared.ui_managed_runtimes import install_selected_runtimes
                install_selected_runtimes(self, ("mrtrix",) if group == "mrtrix" else
                                          ("fsl_cuda" if cuda_candidate() else "fsl_cpu",))
                return
        package = (item.details or {}).get("package")
        if item.kind == "python" and package:
            self._install_python_packages_from_environment_dialog((package,))
            return
        if self._is_model_resource_item(item):
            from shared.ui_managed_runtimes import asset_profiles_for_items
            self._install_model_assets_from_environment_dialog(profile=asset_profiles_for_items((item,)))
            return
        if self._configure_path_for_environment_item(item):
            return
        self._show_environment_install_notes_for_item(item)

    def _setup_python_command(self, python_override=None):
        result = getattr(self, "_lastEnvironmentResult", None)
        python = str(python_override or getattr(result, "python_executable", "") or "")
        from shared.python_environment import setup_python_command
        return setup_python_command(python, slicer_home=str(slicer.app.slicerHome),
                                    launcher=self._slicer_launcher_program())

    def _install_python_packages_from_environment_dialog(self, packages: tuple[str, ...], on_complete=None, *, requests=None) -> None:
        from shared.ui_setup_jobs import start_setup_job
        from shared.ui_selected_setup import python_setup_commands
        packages = tuple(dict.fromkeys(p.strip() for p in packages if p.strip()))
        if not packages:
            return
        commands, receipts = python_setup_commands(self, packages, requests=requests)

        def done(status, details, log_path):
            if status == "completed":
                for receipt in receipts:
                    try:
                        record = json.loads(receipt.read_text())
                    except (OSError, ValueError):
                        record = {}
                    if record.get("status") != "completed":
                        status = "failed"
                    if record.get("restart_required"):
                        self._setupRestartRequired = True
            self._save_python_install_status(packages, status, str(log_path))
            self._lastEnvironmentResult = None
            self._lastEnvironmentSignature = None
            self._setStatus("warning" if status != "completed" or getattr(self, "_setupRestartRequired", False) else "ready",
                            "Setup " + status + "; check dependencies before running")
            if status == "failed":
                self._show_actionable_error(
                    title="Python dependency installation failed",
                    what="The background installer stopped.",
                    why="See the recorded pip output and exact command.",
                    next_step="Open install logs, correct the reported problem, then check dependencies again.",
                    auto_fix="No external software was installed.", details=details + "\\nLog: " + str(log_path))
            elif status == "cancelled":
                slicer.util.infoDisplay(
                    "Installation stopped. Packages already installed were not rolled back. "
                    "Check dependencies before retrying.", parent=self.parent)
            else:
                if not on_complete:
                    slicer.util.infoDisplay(
                        "Installation finished. Check dependencies again; a Slicer restart may be needed "
                        "for packages already imported.", parent=self.parent)
            if on_complete:
                on_complete(status == "completed")
        start_setup_job(self, "Install Python dependencies", commands, done)

    def _show_environment_install_notes(self, result) -> None:
        missing = [item for item in self._missing_environment_items(result) if item.kind != "python"]
        if not missing:
            slicer.util.infoDisplay("No manual setup items are missing.", parent=self.parent)
            return
        message = "\n\n".join(self._environment_item_text(item) for item in missing)
        slicer.util.infoDisplay(message, windowTitle="SlicerDSpace manual setup notes", parent=self.parent)

    def _show_environment_install_notes_for_item(self, item) -> None:
        slicer.util.infoDisplay(
            self._environment_item_text(item),
            windowTitle=f"Install / setup: {item.label}",
            parent=self.parent,
        )

    def _is_model_resource_item(self, item) -> bool:
        return str(getattr(item, "install_group", "") or "") == "model-resources" or str(getattr(item, "key", "")).startswith(
            "path:dd"
        )

    def _missing_model_resource_items(self, result) -> list:
        return [item for item in self._missing_environment_items(result) if self._is_model_resource_item(item)]

    def _environment_setting_for_item(self, item) -> tuple[str | None, str]:
        key = str(getattr(item, "key", "") or "")
        group = str(getattr(item, "install_group", "") or "")
        if self._is_model_resource_item(item):
            return "model_assets_dir", "dir"
        if key == "env:freesurfer_home" or group == "freesurfer":
            return "postprocess_freesurfer_home", "dir"
        if group == "mrtrix":
            return "mrtrix_bin_dir", "dir"
        if group == "fsl":
            return "preprocessing_fsl_bin_dir", "dir"
        if group == "wma":
            return "wma_root", "dir"
        if group == "tractcloud":
            if "data" in key:
                return "tractcloud_data_dir", "dir"
            if "python" in key:
                return "tractcloud_python_executable", "file"
            return "tractcloud_root", "dir"
        if group == "slicer":
            return "slicer_executable", "file"
        if group == "slicer-dmri":
            return "fiber_tract_measurements_cli", "file"
        if key == "path:wmql_python":
            return "wmql_python_executable", "file"
        if "tract_querier" in key:
            return "tract_querier_root", "dir"
        return None, "dir"

    def _configure_path_for_environment_item(self, item) -> bool:
        setting_key, mode = self._environment_setting_for_item(item)
        if not setting_key:
            return False
        start = self._read_environment_path_setting(setting_key) or str(Path.home())
        if mode == "file":
            selected = qt.QFileDialog.getOpenFileName(
                slicer.util.mainWindow(),
                f"Choose {item.label}",
                start,
            )
            selected_text = selected[0] if isinstance(selected, tuple) else selected
        else:
            selected_text = qt.QFileDialog.getExistingDirectory(
                slicer.util.mainWindow(),
                f"Choose {item.label}",
                start,
            )
        selected_text = str(selected_text or "").strip()
        if not selected_text:
            return True
        self._save_environment_path_setting(setting_key, selected_text)
        line_edit = (getattr(self, "_environmentPathLineEdits", {}) or {}).get(setting_key)
        if line_edit:
            line_edit.setText(selected_text)
        self.appendLog(f"[Environment] Saved {setting_key}: {selected_text}")
        result = self._run_environment_dependency_check()
        if result is not None:
            self._show_environment_results_dialog(result)
        return True

    def _choose_model_assets_root(self) -> Optional[Path]:
        module = self._load_model_assets_module()
        configured = self._environment_dependency_path_kwargs().get("model_assets_dir")
        default_root = configured if isinstance(configured, Path) and str(configured) else module.configured_model_assets_root()
        selected = qt.QFileDialog.getExistingDirectory(
            slicer.util.mainWindow(),
            "Choose SlicerDSpace model asset root",
            default_root.expanduser().as_posix(),
        )
        selected_text = str(selected or "").strip()
        if not selected_text:
            return None
        asset_root = Path(selected_text).expanduser().resolve()
        self._save_environment_path_setting("model_assets_dir", asset_root.as_posix())
        line_edit = (getattr(self, "_environmentPathLineEdits", {}) or {}).get("model_assets_dir")
        if line_edit:
            line_edit.setText(asset_root.as_posix())
        self._set_model_assets_status_text(f"Model asset root selected: {asset_root}")
        return asset_root

    def _install_model_assets_from_environment_dialog(
        self, *, profile: str | tuple[str, ...] = "core", prompt_for_root: bool = False, on_complete=None,
    ) -> bool:
        from shared.ui_setup_jobs import start_setup_job
        if getattr(self, "_setupJob", None) or getattr(self, "_pipelineActive", False):
            slicer.util.infoDisplay("Wait for the current job or cancel it before changing setup.", parent=self.parent)
            return False
        if prompt_for_root:
            asset_root = self._choose_model_assets_root()
            if asset_root is None:
                return False
        else:
            module = self._load_model_assets_module()
            configured = self._environment_dependency_path_kwargs().get("model_assets_dir")
            asset_root = Path(configured or module.configured_model_assets_root()).expanduser().resolve()
        script = Path(__file__).resolve().parents[1] / "scripts" / "download_model_assets.py"
        program, prefix = self._slicer_launcher_program(), ["--launch", "PythonSlicer"]
        profiles = (profile,) if isinstance(profile, str) else tuple(dict.fromkeys(profile))
        commands = [(program, [*prefix, "-u", str(script), "--profile", selected,
                              "--asset-root", str(asset_root), "--source", "url",
                              "--source-id", "slicerdsurfer_models_git"], {}) for selected in profiles]

        def done(status, details, log_path):
            if status == "completed":
                self._save_environment_path_setting("model_assets_dir", str(asset_root))
                edit = (getattr(self, "_environmentPathLineEdits", {}) or {}).get("model_assets_dir")
                if edit is not None:
                    edit.setText(str(asset_root))
            message = ("Model assets downloaded and checksum-verified." if status == "completed" else
                       "Download stopped; partial files are retained for retry." if status == "cancelled" else
                       "Download or checksum verification failed.")
            self._set_model_assets_status_text(message + " " + str(asset_root))
            self._setStatus("ready" if status == "completed" else "warning", message)
            self._lastEnvironmentResult = None
            self._lastEnvironmentSignature = None
            if status == "failed":
                self._show_actionable_error(
                    title="Model asset setup failed", what=message,
                    why="Network, file permissions or checksum validation may have failed.",
                    next_step="Inspect the install log and retry using the same asset directory.",
                    auto_fix="Partial downloads are retained; no scientific outputs were modified.",
                    details=details + "\\nLog: " + str(log_path))
            if on_complete:
                on_complete(status == "completed", message)
        return start_setup_job(self, "Download and verify model assets", commands, done)

    def _setPipelineActive(self, active: bool, *, preserve_status: bool = False) -> None:
        self._pipelineActive = active
        if getattr(self, '_experimentWidgets', None):
            for key in ('enabled', 'scope', 'root', 'jobs', 'options', 'variants', 'preview', 'export'):
                self._experimentWidgets[key].enabled = not active
        for button in getattr(self, '_algorithmButtons', {}).values():
            button.enabled = not active
        self._refresh_run_button_state()
        if active:
            if hasattr(self.ui, "applyButton"):
                self.ui.applyButton.enabled = False
            if hasattr(self.ui, "runSelectedButton"):
                self.ui.runSelectedButton.enabled = False
            if hasattr(self.ui, "validateButton"):
                self.ui.validateButton.enabled = False
            if getattr(self, "_cancelRunButton", None):
                self._cancelRunButton.enabled = True
                self._cancelRunButton.visible = True
            self._setStatus("running", "Running pipeline...")
            self._lastStatusMessage = "Running pipeline..."
        else:
            self._refresh_run_button_state()
            self.validateInputs(manual=False, update_status=not preserve_status)
            if getattr(self, "_cancelRunButton", None):
                self._cancelRunButton.enabled = False
                self._cancelRunButton.visible = False
