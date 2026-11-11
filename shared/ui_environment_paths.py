"""Qt builders for the SlicerDSpace environment-path settings panel."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shared.dependency_config import UI_DEPENDENCY_PATHS
from shared.ui_environment_panel import environment_path_entries, dependency_backend_group, model_assets_summary_text
from shared.ui_onboarding import register_ui_language_binding


_SECTION_KEYS_BY_TITLE = {
    "Model Asset Paths": "model-assets",
    "Core Python & Runtime Paths": "core-runtime",
    "Optional Backend Paths": "bundle-backends",
}


def install_environment_dependency_path_controls(owner: Any, environment_layout: Any) -> None:
    """Install the schema-driven environment dependency path controls."""

    import qt

    owner._environmentPathLineEdits = {}
    owner._environmentPathStatusLabels = {}
    owner._environmentBackendSectionLabels = {}
    owner._environmentBackendSectionRows = {"wma": [], "tractcloud": []}
    owner._environmentSectionLanguageWidgets = []
    owner._environmentPathLanguageWidgets = {}
    owner._environmentPathStatusSources = {}
    owner._environmentPathStatusStates = {}

    from shared.ui_selected_setup import use_offline_assets
    from shared.ui_environment_check import request_prepare_environment, start_environment_check
    setup_button = qt.QPushButton("Prepare selected analysis environment")
    setup_button.setObjectName("managedRuntimeSetupButton")
    setup_button.connect("clicked()", lambda: request_prepare_environment(owner))
    register_ui_language_binding(owner, setup_button, "text", "managed_runtime_setup")
    register_ui_language_binding(owner, setup_button, "tooltip", "managed_runtime_setup_tooltip")
    owner._managedRuntimeSetupButton = setup_button
    from shared.setup_resume import refresh_setup_button
    refresh_setup_button(owner)
    environment_layout.addWidget(setup_button)
    import ctk
    more = ctk.ctkCollapsibleButton()
    more.setObjectName("environmentMoreOptions")
    more.collapsed = True
    more.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)
    more.setMinimumWidth(0)
    register_ui_language_binding(owner, more, "text", "environment_more_options")
    more_layout = qt.QVBoxLayout(more)
    more_layout.setContentsMargins(8, 8, 8, 8)
    owner._environmentMoreOptions = more
    details_button = qt.QPushButton("View dependency details")
    details_button.setObjectName("environmentDetailsButton")

    def show_details():
        report = owner._cached_environment_result(owner._environment_check_signature(owner._environment_check_kwargs()))
        if report is not None:
            owner._show_environment_results_dialog(report)
        else:
            start_environment_check(owner, owner._show_environment_results_dialog)

    details_button.connect("clicked()", show_details)
    register_ui_language_binding(owner, details_button, "text", "environment_details")
    owner._environmentDetailsButton = details_button
    details_button.setMinimumWidth(0)
    details_button.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
    more_layout.addWidget(details_button)
    offline_button = qt.QPushButton("Use offline model resources")
    offline_button.setObjectName("offlineModelResourcesButton")
    offline_button.setMinimumWidth(0)
    offline_button.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
    offline_button.connect("clicked()", lambda: use_offline_assets(owner))
    register_ui_language_binding(owner, offline_button, "text", "offline_model_resources")
    owner._offlineModelResourcesButton = offline_button
    more_layout.addWidget(offline_button)
    repair_button = qt.QPushButton("Repair selected managed tools")
    repair_button.setObjectName("repairManagedToolsButton")
    repair_button.setMinimumWidth(0)
    repair_button.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
    repair_button.connect("clicked()", lambda: request_prepare_environment(owner, repair=True))
    register_ui_language_binding(owner, repair_button, "text", "repair_managed_tools")
    register_ui_language_binding(owner, repair_button, "tooltip", "repair_managed_tools_tooltip")
    owner._repairManagedToolsButton = repair_button
    more_layout.addWidget(repair_button)

    from shared.ui_styles import is_dark_mode

    dark = is_dark_mode(getattr(owner, "_uiWidget", None))
    python_target_label = qt.QLabel(owner._python_install_target_text())
    python_target_label.setObjectName("environmentPythonTargetLabel")
    python_target_label.wordWrap = True
    python_target_label.setStyleSheet(
        "QLabel { background: #22272E; color: #B8C5D0; padding: 6px; border-radius: 4px; }"
        if dark
        else "QLabel { background: #F5F7F9; color: #455A64; padding: 6px; border-radius: 4px; }"
    )
    python_target_label.visible = False
    owner._environmentPythonTargetLabel = python_target_label

    group_box, form_layout = _build_environment_group(owner)
    owner._environmentAdvancedPathsWidget = group_box

    for entry in environment_path_entries(UI_DEPENDENCY_PATHS):
        if entry.kind == "section":
            section = _SECTION_KEYS_BY_TITLE.get(entry.title, entry.title)
            title_label, description_label = add_environment_section_header(form_layout, entry.title, entry.description)
            owner._environmentSectionLanguageWidgets.append((section, title_label, description_label))
        elif entry.kind == "subsection":
            owner._environmentBackendSectionLabels[entry.subsection] = add_environment_subsection_label(
                form_layout,
                entry.title,
                entry.description,
            )
            title_label, description_label = owner._environmentBackendSectionLabels[entry.subsection]
            owner._environmentSectionLanguageWidgets.append(
                (entry.subsection, title_label, description_label)
            )
        elif entry.kind == "row" and entry.item:
            add_environment_path_row(owner, form_layout, entry.item)
        elif entry.kind == "model-assets":
            add_model_assets_row(owner, form_layout)

    add_dependency_config_buttons(owner, form_layout)

    more_layout.addWidget(python_target_label)
    more_layout.addWidget(group_box)
    environment_layout.addWidget(more)


def _build_environment_group(owner: Any):
    import qt

    tooltip = (
        "Discovery order: saved path, environment variables, PATH, then common install paths. "
        "Leave fields empty to rely on automatic discovery."
    )
    try:
        import ctk

        group_box = ctk.ctkCollapsibleButton()
        group_box.setObjectName("externalBackendsGroup")
        group_box.text = "External Runtime Paths"
        group_box.collapsed = True
        group_box.setToolTip(tooltip)
        form_layout = qt.QFormLayout(group_box)
    except Exception:
        group_box = qt.QGroupBox("External Runtime Paths")
        group_box.setObjectName("externalBackendsGroup")
        group_box.setToolTip(tooltip)
        form_layout = qt.QFormLayout(group_box)

    form_layout.setSpacing(6)
    form_layout.setContentsMargins(10, 10, 10, 10)
    form_layout.setRowWrapPolicy(qt.QFormLayout.WrapAllRows)
    form_layout.setFieldGrowthPolicy(qt.QFormLayout.AllNonFixedFieldsGrow)
    group_box.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)
    group_box.setMinimumWidth(0)
    group_box.setToolTip(tooltip)
    register_ui_language_binding(owner, group_box, "text", "external_runtime_paths")
    register_ui_language_binding(owner, group_box, "tooltip", "external_runtime_paths_tooltip")
    return group_box, form_layout


def add_dependency_config_buttons(owner: Any, form_layout: Any) -> None:
    import qt

    config_container = qt.QWidget()
    config_container.setMinimumWidth(0)
    config_buttons = qt.QGridLayout(config_container)
    config_buttons.setContentsMargins(0, 0, 0, 0)
    config_buttons.setSpacing(6)
    import_button = qt.QPushButton("Import dependency config")
    import_button.setMinimumWidth(0)
    import_button.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
    import_button.setToolTip("Load a SlicerDSpace dependency config JSON and save the external paths into Slicer settings.")
    import_button.connect("clicked()", owner._on_import_dependency_config_clicked)
    export_button = qt.QPushButton("Export dependency config")
    export_button.setMinimumWidth(0)
    export_button.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
    export_button.setToolTip("Save the currently configured external paths as a CLI-compatible JSON config.")
    export_button.connect("clicked()", owner._on_export_dependency_config_clicked)
    owner._environmentImportConfigButton = import_button
    owner._environmentExportConfigButton = export_button
    register_ui_language_binding(owner, import_button, "text", "import_dependency_config")
    register_ui_language_binding(owner, import_button, "tooltip", "import_dependency_config_tooltip")
    register_ui_language_binding(owner, export_button, "text", "export_dependency_config")
    register_ui_language_binding(owner, export_button, "tooltip", "export_dependency_config_tooltip")
    config_buttons.addWidget(import_button, 0, 0)
    config_buttons.addWidget(export_button, 1, 0)

    report_button = qt.QPushButton("Open environment report")
    report_button.setMinimumWidth(0)
    report_button.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
    report_button.setToolTip("Open the latest environment_doctor.json directory after a dependency check.")
    report_button.connect("clicked()", owner._on_open_environment_report_dir_clicked)
    register_ui_language_binding(owner, report_button, "text", "open_environment_report")
    register_ui_language_binding(owner, report_button, "tooltip", "open_environment_report_tooltip")
    config_buttons.addWidget(report_button, 2, 0)

    install_logs_button = qt.QPushButton("Open install logs")
    install_logs_button.setMinimumWidth(0)
    install_logs_button.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
    install_logs_button.setToolTip("Open the latest Python dependency install logs/results directory.")
    install_logs_button.connect("clicked()", owner._on_open_install_logs_dir_clicked)
    register_ui_language_binding(owner, install_logs_button, "text", "open_install_logs")
    register_ui_language_binding(owner, install_logs_button, "tooltip", "open_install_logs_tooltip")
    config_buttons.addWidget(install_logs_button, 3, 0)
    config_buttons.setColumnStretch(0, 1)

    config_label = qt.QLabel("Config")
    register_ui_language_binding(owner, config_label, "text", "config")
    form_layout.addRow(config_label, config_container)


def add_environment_path_row(owner: Any, form_layout: Any, item: dict[str, str]) -> None:
    import qt

    dependency_id = item["id"]
    row = qt.QWidget()
    row.setMinimumWidth(0)
    row_layout = qt.QHBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)

    line_edit = qt.QLineEdit()
    line_edit.setMinimumWidth(0)
    line_edit.setText(owner._read_environment_path_setting(dependency_id))
    line_edit.setPlaceholderText("Auto-detect if empty")
    line_edit.setToolTip(item["tooltip"])
    line_edit.editingFinished.connect(
        lambda dependency_id=dependency_id, line_edit=line_edit: owner._on_environment_dependency_path_edited(
            dependency_id,
            line_edit,
        )
    )
    row_layout.addWidget(line_edit, 1)

    browse_button = qt.QPushButton("Browse...")
    browse_button.setMinimumWidth(0)
    browse_button.setToolTip(item["tooltip"])
    browse_button.connect(
        "clicked()",
        lambda item=item, line_edit=line_edit: owner._browse_environment_dependency_path(item, line_edit),
    )
    row_layout.addWidget(browse_button)

    clear_button = qt.QPushButton("Clear")
    clear_button.setMinimumWidth(0)
    clear_button.connect(
        "clicked()",
        lambda dependency_id=dependency_id, line_edit=line_edit: owner._clear_environment_dependency_path(
            dependency_id,
            line_edit,
        ),
    )
    row_layout.addWidget(clear_button)

    from shared.ui_styles import is_dark_mode

    dark = is_dark_mode(getattr(owner, "_uiWidget", None))
    status_label = qt.QLabel("not checked")
    status_label.setMinimumWidth(72)
    status_label.setStyleSheet("QLabel { color: #9BA7B4; }" if dark else "QLabel { color: #78909C; }")
    row_layout.addWidget(status_label)

    row_label = qt.QLabel(item["label"])
    form_layout.addRow(row_label, row)
    owner._environmentPathLineEdits[dependency_id] = line_edit
    owner._environmentPathStatusLabels[dependency_id] = status_label
    owner._environmentPathStatusSources[dependency_id] = "not checked"
    owner._environmentPathLanguageWidgets[dependency_id] = {
        "label": row_label,
        "line_edit": line_edit,
        "browse": browse_button,
        "clear": clear_button,
    }
    backend_group = dependency_backend_group(dependency_id)
    if backend_group:
        owner._environmentBackendSectionRows[backend_group].append(
            (line_edit, browse_button, clear_button, status_label)
        )


def add_environment_section_header(form_layout: Any, title: str, description: str):
    import qt
    from shared.ui_styles import is_dark_mode

    dark = is_dark_mode()
    title_label = qt.QLabel(title)
    title_label.setObjectName("externalSectionTitleLabel")
    title_label.setStyleSheet(
        "QLabel { background: #1E2732; border: 1px solid #354658; border-radius: 6px; "
        "color: #CBE4F5; font-weight: 700; padding: 6px 10px; }"
        if dark
        else "QLabel { background: #EEF3F8; border: 1px solid #CFDCE8; border-radius: 6px; "
        "color: #1D3549; font-weight: 700; padding: 6px 10px; }"
    )
    description_label = qt.QLabel(description)
    description_label.setObjectName("externalSectionDescriptionLabel")
    description_label.wordWrap = True
    description_label.setStyleSheet(
        "QLabel { color: #9BA7B4; padding: 1px 4px 6px 4px; }"
        if dark
        else "QLabel { color: #546E7A; padding: 1px 4px 6px 4px; }"
    )
    form_layout.addRow(title_label)
    form_layout.addRow(description_label)
    return (title_label, description_label)


def add_environment_subsection_label(form_layout: Any, title: str, description: str):
    import qt
    from shared.ui_styles import is_dark_mode

    dark = is_dark_mode()
    title_label = qt.QLabel(title)
    title_label.setObjectName("externalSubsectionTitleLabel")
    title_label.setStyleSheet(
        "QLabel { background: #232B34; border: 1px solid #374350; border-radius: 5px; "
        "color: #BDD5E7; font-weight: 700; padding: 5px 9px; }"
        if dark
        else "QLabel { background: #F4F7FA; border: 1px solid #DAE3EC; border-radius: 5px; "
        "color: #2A4359; font-weight: 700; padding: 5px 9px; }"
    )
    description_label = qt.QLabel(description)
    description_label.setObjectName("externalSubsectionDescriptionLabel")
    description_label.wordWrap = True
    description_label.setStyleSheet(
        "QLabel { color: #9BA7B4; padding: 1px 4px 4px 4px; }"
        if dark
        else "QLabel { color: #607D8B; padding: 1px 4px 4px 4px; }"
    )
    form_layout.addRow(title_label)
    form_layout.addRow(description_label)
    return (title_label, description_label)


def compact_model_assets_summary(full_text: str) -> str:
    """Format model-asset summary concisely on two lines while keeping full details for tooltips."""

    raw = str(full_text or "").strip()
    if not raw:
        return ""
    root_part = ""
    for marker in ("Default root: ", "Asset root: "):
        if marker in raw:
            tail = raw.split(marker, 1)[1]
            root_val = tail.split(". ", 1)[0].rstrip(".")
            label = "Default root" if marker.startswith("Default") else "Asset root"
            root_part = f"{label}: {root_val}"
            break
    status_part = ""
    if "Current status: " in raw:
        status_val = raw.split("Current status: ", 1)[1].rstrip(".")
        status_part = f"Status: {status_val}"
    if root_part and status_part:
        return f"{root_part}\n{status_part}"
    if root_part:
        return f"{root_part}\nClick Check status to verify local assets."
    return raw


def add_model_assets_row(owner: Any, form_layout: Any) -> None:
    import qt
    from shared.ui_styles import is_dark_mode

    dark = is_dark_mode(getattr(owner, "_uiWidget", None))
    row = qt.QWidget()
    row.setMinimumWidth(0)
    row_layout = qt.QVBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(6)

    full_summary = model_assets_summary_text(Path(__file__).resolve().parent.parent)
    summary = qt.QLabel(compact_model_assets_summary(full_summary))
    summary.setObjectName("modelAssetsSummaryLabel")
    summary.wordWrap = True
    summary.setToolTip(full_summary)
    summary.setStyleSheet(
        "QLabel { background: #20262E; border: 1px solid #35414E; border-radius: 5px; color: #C5D4E0; padding: 6px 8px; }"
        if dark
        else "QLabel { background: #F5F8FB; border: 1px solid #D6E0EA; border-radius: 5px; color: #34495E; padding: 6px 8px; }"
    )
    row_layout.addWidget(summary)
    owner._modelAssetsSummaryLabel = summary

    buttons_grid = qt.QGridLayout()
    buttons_grid.setSpacing(6)

    status_button = qt.QPushButton("Check status")
    status_button.setMinimumWidth(0)
    status_button.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
    status_button.setToolTip("Check whether core and all model asset profiles are present and not Git LFS pointer files.")
    status_button.connect("clicked()", owner._on_check_model_assets_status_clicked)
    buttons_grid.addWidget(status_button, 0, 0)
    owner._modelAssetsCheckStatusButton = status_button
    register_ui_language_binding(owner, status_button, "text", "check_status")
    register_ui_language_binding(owner, status_button, "tooltip", "check_status_tooltip")

    verify_button = qt.QPushButton("Verify core")
    verify_button.setMinimumWidth(0)
    verify_button.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
    verify_button.setToolTip("Verify the core model asset profile required by DDSurfer, DDSeg, and DDParcel.")
    verify_button.connect("clicked()", owner._on_verify_core_model_assets_clicked)
    buttons_grid.addWidget(verify_button, 0, 1)
    owner._modelAssetsVerifyCoreButton = verify_button
    register_ui_language_binding(owner, verify_button, "text", "verify_core")
    register_ui_language_binding(owner, verify_button, "tooltip", "verify_core_tooltip")

    lockfile_button = qt.QPushButton("Write lockfile")
    lockfile_button.setMinimumWidth(0)
    lockfile_button.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
    lockfile_button.setToolTip("Write a SHA256 lockfile for the core model asset profile.")
    lockfile_button.connect("clicked()", owner._on_write_model_asset_lockfile_clicked)
    buttons_grid.addWidget(lockfile_button, 1, 0)
    owner._modelAssetsWriteLockfileButton = lockfile_button
    register_ui_language_binding(owner, lockfile_button, "text", "write_lockfile")
    register_ui_language_binding(owner, lockfile_button, "tooltip", "write_lockfile_tooltip")

    guide_button = qt.QPushButton("Open guide")
    guide_button.setMinimumWidth(0)
    guide_button.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
    guide_button.setToolTip("Open docs/assets/MODEL_ASSETS.md for unified model weights/template download and verification.")
    guide_button.connect("clicked()", owner._on_open_model_assets_guide_clicked)
    register_ui_language_binding(owner, guide_button, "text", "open_guide")
    register_ui_language_binding(owner, guide_button, "tooltip", "open_guide_tooltip")
    buttons_grid.addWidget(guide_button, 1, 1)

    terms_button = qt.QPushButton("Terms")
    terms_button.setMinimumWidth(0)
    terms_button.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
    terms_button.setToolTip("Open docs/assets/MODEL_ASSET_TERMS.md.")
    terms_button.connect("clicked()", owner._on_open_model_asset_terms_clicked)
    register_ui_language_binding(owner, terms_button, "text", "terms")
    register_ui_language_binding(owner, terms_button, "tooltip", "terms_tooltip")
    buttons_grid.addWidget(terms_button, 2, 0)

    manifest_button = qt.QPushButton("Open manifest")
    manifest_button.setMinimumWidth(0)
    manifest_button.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
    manifest_button.setToolTip("Open models/assets_manifest.json.")
    manifest_button.connect("clicked()", owner._on_open_model_assets_manifest_clicked)
    register_ui_language_binding(owner, manifest_button, "text", "open_manifest")
    register_ui_language_binding(owner, manifest_button, "tooltip", "model_manifest_tooltip")
    buttons_grid.addWidget(manifest_button, 2, 1)

    provenance_button = qt.QPushButton("Provenance")
    provenance_button.setMinimumWidth(0)
    provenance_button.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
    provenance_button.setToolTip("Show model asset profile, verification, lockfile, and release-boundary notes.")
    provenance_button.connect("clicked()", owner._on_show_model_asset_provenance_clicked)
    register_ui_language_binding(owner, provenance_button, "text", "provenance")
    register_ui_language_binding(owner, provenance_button, "tooltip", "provenance_tooltip")
    buttons_grid.addWidget(provenance_button, 3, 0)

    release_button = qt.QPushButton("Release notes")
    release_button.setMinimumWidth(0)
    release_button.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
    release_button.setToolTip("Open docs/release/RELEASE_NOTES_v0.1.0.md.")
    release_button.connect("clicked()", owner._on_open_release_notes_clicked)
    register_ui_language_binding(owner, release_button, "text", "release_notes")
    register_ui_language_binding(owner, release_button, "tooltip", "release_notes_tooltip")
    buttons_grid.addWidget(release_button, 3, 1)
    buttons_grid.setColumnStretch(0, 1)
    buttons_grid.setColumnStretch(1, 1)

    row_layout.addLayout(buttons_grid)

    row_label = qt.QLabel("Model assets bundle")
    register_ui_language_binding(owner, row_label, "text", "model_assets_bundle")
    form_layout.addRow(row_label, row)


def refresh_environment_paths_styles(owner: Any, *, dark: bool | None = None) -> None:
    """Refresh inline stylesheets inside the Environment & Installation panel for light/dark themes."""

    from shared.ui_environment_panel import environment_path_status_style
    from shared.ui_styles import is_dark_mode

    use_dark = is_dark_mode(getattr(owner, "_uiWidget", None)) if dark is None else bool(dark)
    python_target = getattr(owner, "_environmentPythonTargetLabel", None)
    if python_target is not None:
        python_target.setStyleSheet(
            "QLabel { background: #22272E; color: #B8C5D0; padding: 6px; border-radius: 4px; }"
            if use_dark
            else "QLabel { background: #F5F7F9; color: #455A64; padding: 6px; border-radius: 4px; }"
        )
    model_summary = getattr(owner, "_modelAssetsSummaryLabel", None)
    if model_summary is not None:
        model_summary.setStyleSheet(
            "QLabel { background: #20262E; border: 1px solid #35414E; border-radius: 5px; color: #C5D4E0; padding: 6px 8px; }"
            if use_dark
            else "QLabel { background: #F5F8FB; border: 1px solid #D6E0EA; border-radius: 5px; color: #34495E; padding: 6px 8px; }"
        )
    for section, title_label, description_label in tuple(
        getattr(owner, "_environmentSectionLanguageWidgets", ()) or ()
    ):
        if section in {"wma", "tractcloud"}:
            if title_label is not None:
                title_label.setStyleSheet(
                    "QLabel { background: #232B34; border: 1px solid #374350; border-radius: 5px; "
                    "color: #BDD5E7; font-weight: 700; padding: 5px 9px; }"
                    if use_dark
                    else "QLabel { background: #F4F7FA; border: 1px solid #DAE3EC; border-radius: 5px; "
                    "color: #2A4359; font-weight: 700; padding: 5px 9px; }"
                )
            if description_label is not None:
                description_label.setStyleSheet(
                    "QLabel { color: #9BA7B4; padding: 1px 4px 4px 4px; }"
                    if use_dark
                    else "QLabel { color: #607D8B; padding: 1px 4px 4px 4px; }"
                )
        else:
            if title_label is not None:
                title_label.setStyleSheet(
                    "QLabel { background: #1E2732; border: 1px solid #354658; border-radius: 6px; "
                    "color: #CBE4F5; font-weight: 700; padding: 6px 10px; }"
                    if use_dark
                    else "QLabel { background: #EEF3F8; border: 1px solid #CFDCE8; border-radius: 6px; "
                    "color: #1D3549; font-weight: 700; padding: 6px 10px; }"
                )
            if description_label is not None:
                description_label.setStyleSheet(
                    "QLabel { color: #9BA7B4; padding: 1px 4px 6px 4px; }"
                    if use_dark
                    else "QLabel { color: #546E7A; padding: 1px 4px 6px 4px; }"
                )
    states = getattr(owner, "_environmentPathStatusStates", {}) or {}
    for dependency_id, status_label in (getattr(owner, "_environmentPathStatusLabels", {}) or {}).items():
        if status_label is None:
            continue
        state = states.get(dependency_id)
        if state:
            status_label.setStyleSheet(environment_path_status_style(state, dark=use_dark))
        else:
            status_label.setStyleSheet("QLabel { color: #9BA7B4; }" if use_dark else "QLabel { color: #78909C; }")
