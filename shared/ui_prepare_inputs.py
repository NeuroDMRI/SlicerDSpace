"""Qt builders for preparing external DWI inputs from the Slicer UI."""

from __future__ import annotations

from typing import Any

from shared.ui_onboarding import register_ui_language_binding


def install_prepare_inputs_panel(owner: Any, configuration_layout: Any) -> None:
    """Install a compact Prepare Inputs panel into the main workflow section."""

    import qt

    owner._prepareInputWidgets = {}

    try:
        import ctk

        panel = ctk.ctkCollapsibleButton()
        panel.text = "Prepare Inputs"
        panel.collapsed = True
    except Exception:
        panel = qt.QGroupBox("Prepare Inputs")
    panel.setObjectName("prepareInputsPanel")
    panel.setToolTip("Normalize DICOM, NRRD/NHDR, or NIfTI+bval/bvec inputs into a SlicerDSpace subject root.")
    register_ui_language_binding(owner, panel, "text", "prepare_inputs")
    register_ui_language_binding(owner, panel, "tooltip", "prepare_inputs_tooltip")
    from shared.ui_styles import is_dark_mode

    dark = is_dark_mode(getattr(owner, "_uiWidget", None))
    panel.setStyleSheet(
        "QWidget#prepareInputsPanel { background: #222830; border: 1px solid #394450; border-radius: 6px; }"
        if dark
        else "QWidget#prepareInputsPanel { background: #F6F8FB; border: 1px solid #D6DFE8; border-radius: 6px; }"
    )
    owner._prepareInputWidgets["panel"] = panel

    form_layout = qt.QFormLayout(panel)
    form_layout.setSpacing(6)
    form_layout.setContentsMargins(8, 8, 8, 8)
    form_layout.setRowWrapPolicy(qt.QFormLayout.WrapLongRows)
    form_layout.setFieldGrowthPolicy(qt.QFormLayout.AllNonFixedFieldsGrow)
    panel.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)
    panel.setMinimumWidth(0)

    summary = qt.QLabel(
        "Organize external DICOM, NRRD, or NIfTI + bval/bvec files into <subject root>/input/."
    )
    summary.wordWrap = True
    summary.setStyleSheet(
        "QLabel { background: #1E2732; border: 1px solid #354658; border-radius: 5px; color: #C5D8E8; padding: 6px 8px; }"
        if dark
        else "QLabel { background: #EEF3F8; border: 1px solid #CFDCE8; border-radius: 5px; color: #243B53; padding: 6px 8px; }"
    )
    owner._prepareInputWidgets["summary"] = summary
    register_ui_language_binding(owner, summary, "text", "prepare_summary")
    form_layout.addRow(summary)

    source_kind = qt.QComboBox()
    for label in ("auto", "dicom", "nifti", "nrrd", "canonical"):
        source_kind.addItem(label)
    source_kind.setToolTip("Use auto unless you need to force a specific source type.")
    owner._prepareInputWidgets["source_kind"] = source_kind
    register_ui_language_binding(owner, source_kind, "tooltip", "source_type_tooltip")
    _add_labeled_row(owner, form_layout, "source_type", source_kind)

    source_row, source_edit = _path_row(
        owner,
        key="source",
        placeholder="DICOM folder, DWI NIfTI, NRRD/NHDR, or canonical subject root",
        browse_callback=owner._browse_prepare_input_source_path,
    )
    _add_labeled_row(owner, form_layout, "source", source_row)
    owner._prepareInputWidgets["source"] = source_edit

    subject_root_row, subject_root_edit = _path_row(
        owner,
        key="subject_root",
        placeholder="Destination subject root, e.g. /data/study/sub-01",
        browse_callback=owner._browse_prepare_input_subject_root,
    )
    _add_labeled_row(owner, form_layout, "subject_root", subject_root_row)
    owner._prepareInputWidgets["subject_root"] = subject_root_edit

    subject_id = qt.QLineEdit()
    subject_id.setPlaceholderText("sub-01")
    subject_id.setToolTip("Subject identifier recorded in input/input_import_manifest.json.")
    owner._prepareInputWidgets["subject_id"] = subject_id
    register_ui_language_binding(owner, subject_id, "tooltip", "prepare_subject_id_tooltip")
    _add_labeled_row(owner, form_layout, "subject_id", subject_id)

    sidecar_summary = qt.QLabel("Optional sidecars for NIfTI inputs or externally generated masks.")
    sidecar_summary.setStyleSheet(
        "QLabel { color: #B8C5D0; padding-top: 4px; }"
        if dark
        else "QLabel { color: #5B7167; padding-top: 4px; }"
    )
    owner._prepareInputWidgets["sidecar_summary"] = sidecar_summary
    register_ui_language_binding(owner, sidecar_summary, "text", "sidecar_summary")
    form_layout.addRow(sidecar_summary)

    for key, label_key, placeholder in (
        ("bval", "bval", "Optional .bval sidecar"),
        ("bvec", "bvec", "Optional .bvec sidecar"),
        ("metadata", "metadata_json", "Optional BIDS-style .json sidecar"),
        ("mask", "brain_mask", "Optional nodif_brain_mask.nii.gz"),
    ):
        row, edit = _path_row(owner, key=key, placeholder=placeholder, browse_callback=owner._browse_prepare_input_file)
        _add_labeled_row(owner, form_layout, label_key, row, storage_key=f"{key}_label")
        owner._prepareInputWidgets[key] = edit

    backend = qt.QComboBox()
    for label in ("auto", "copy", "dcm2niix", "mrconvert", "dwiconvert"):
        backend.addItem(label)
    backend.setToolTip("Use auto unless you need to force dcm2niix, MRtrix3 mrconvert, SlicerDMRI DWIConvert, or copy.")
    owner._prepareInputWidgets["backend"] = backend
    register_ui_language_binding(owner, backend, "tooltip", "backend_tooltip")
    _add_labeled_row(owner, form_layout, "backend", backend)

    output_kind = qt.QComboBox()
    for label in ("canonical-nifti", "compat-nrrd"):
        output_kind.addItem(label)
    output_kind.setToolTip("canonical-nifti is preferred. compat-nrrd is available for legacy DWIConvert routes.")
    owner._prepareInputWidgets["output_kind"] = output_kind
    register_ui_language_binding(owner, output_kind, "tooltip", "output_tooltip")
    _add_labeled_row(owner, form_layout, "output", output_kind)

    options_row = qt.QWidget()
    options_layout = qt.QHBoxLayout(options_row)
    options_layout.setContentsMargins(0, 0, 0, 0)
    dry_run = qt.QCheckBox("Dry run")
    dry_run.setToolTip("Plan the import without writing files.")
    force = qt.QCheckBox("Overwrite existing files")
    force.setToolTip("Allow replacing existing files in the destination input folder.")
    owner._prepareInputWidgets["dry_run"] = dry_run
    owner._prepareInputWidgets["force"] = force
    register_ui_language_binding(owner, dry_run, "text", "dry_run")
    register_ui_language_binding(owner, dry_run, "tooltip", "dry_run_tooltip")
    register_ui_language_binding(owner, force, "text", "overwrite_files")
    register_ui_language_binding(owner, force, "tooltip", "overwrite_files_tooltip")
    options_layout.addWidget(dry_run)
    options_layout.addWidget(force)
    options_layout.addStretch(1)
    _add_labeled_row(owner, form_layout, "options", options_row)

    buttons = qt.QHBoxLayout()
    preview_button = qt.QPushButton("Preview plan")
    preview_button.setToolTip("Build and show the input import plan without writing files.")
    preview_button.connect("clicked()", owner._on_prepare_inputs_preview_clicked)
    prepare_button = qt.QPushButton("Prepare inputs")
    prepare_button.setToolTip("Create the canonical subject-root input folder and write input_import_manifest.json.")
    prepare_button.connect("clicked()", owner._on_prepare_inputs_clicked)
    manifest_button = qt.QPushButton("Open manifest")
    manifest_button.setToolTip("Open input/input_import_manifest.json for the selected subject root.")
    manifest_button.connect("clicked()", owner._on_open_prepare_input_manifest_clicked)
    owner._prepareInputWidgets["preview_button"] = preview_button
    owner._prepareInputWidgets["prepare_button"] = prepare_button
    owner._prepareInputWidgets["manifest_button"] = manifest_button
    register_ui_language_binding(owner, preview_button, "text", "preview_plan")
    register_ui_language_binding(owner, preview_button, "tooltip", "preview_plan_tooltip")
    register_ui_language_binding(owner, prepare_button, "text", "prepare_inputs_action")
    register_ui_language_binding(owner, prepare_button, "tooltip", "prepare_inputs_action_tooltip")
    register_ui_language_binding(owner, manifest_button, "text", "open_manifest")
    register_ui_language_binding(owner, manifest_button, "tooltip", "open_prepare_manifest_tooltip")
    buttons.addWidget(preview_button)
    buttons.addWidget(prepare_button)
    buttons.addWidget(manifest_button)
    buttons.addStretch(1)
    _add_labeled_row(owner, form_layout, "actions", buttons)

    # Optional pre-step: keep it collapsed below the primary input fields.
    configuration_layout.addWidget(panel)
    refresh_prepare_inputs_styles(owner, dark=dark)


def refresh_prepare_inputs_styles(owner: Any, *, dark: bool | None = None) -> None:
    """Refresh inline stylesheets inside the Prepare Inputs panel for light/dark themes."""

    widgets = getattr(owner, "_prepareInputWidgets", None) or {}
    if not widgets:
        return
    from shared.ui_styles import is_dark_mode

    use_dark = is_dark_mode(getattr(owner, "_uiWidget", None)) if dark is None else bool(dark)
    panel = widgets.get("panel")
    if panel is not None:
        if getattr(owner, "_advancedSettingsPanel", None) is not None:
            panel.setStyleSheet("")
        else:
            panel.setStyleSheet(
                "QWidget#prepareInputsPanel { background: #222830; border: 1px solid #394450; border-radius: 6px; }"
                if use_dark
                else "QWidget#prepareInputsPanel { background: #F6F8FB; border: 1px solid #D6DFE8; border-radius: 6px; }"
            )
    summary = widgets.get("summary")
    if summary is not None:
        summary.setStyleSheet(
            "QLabel { background: #1E2732; border: 1px solid #354658; border-radius: 5px; color: #C5D8E8; padding: 6px 8px; }"
            if use_dark
            else "QLabel { background: #EEF3F8; border: 1px solid #CFDCE8; border-radius: 5px; color: #243B53; padding: 6px 8px; }"
        )
    sidecar_summary = widgets.get("sidecar_summary")
    if sidecar_summary is not None:
        sidecar_summary.setStyleSheet(
            "QLabel { color: #9BA7B4; padding-top: 4px; }"
            if use_dark
            else "QLabel { color: #546E7A; padding-top: 4px; }"
        )


def _path_row(owner: Any, *, key: str, placeholder: str, browse_callback):
    import qt

    row = qt.QWidget()
    row_layout = qt.QHBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)
    edit = qt.QLineEdit()
    edit.setPlaceholderText(placeholder)
    edit.setToolTip(placeholder)
    placeholder_key = {
        "source": "source_placeholder",
        "subject_root": "prepare_subject_root_placeholder",
        "bval": "bval_placeholder",
        "bvec": "bvec_placeholder",
        "metadata": "metadata_placeholder",
        "mask": "brain_mask_placeholder",
    }[key]
    register_ui_language_binding(owner, edit, "placeholder", placeholder_key)
    register_ui_language_binding(owner, edit, "tooltip", placeholder_key)
    row_layout.addWidget(edit, 1)
    browse = qt.QPushButton("Browse...")
    browse.setToolTip(placeholder)
    register_ui_language_binding(owner, browse, "text", "browse")
    register_ui_language_binding(owner, browse, "tooltip", placeholder_key)
    browse.connect("clicked()", lambda key=key: browse_callback(key))
    row_layout.addWidget(browse)
    clear = qt.QPushButton("Clear")
    register_ui_language_binding(owner, clear, "text", "clear")
    clear.connect("clicked()", lambda edit=edit: edit.setText(""))
    row_layout.addWidget(clear)
    owner._prepareInputWidgets[f"{key}_browse"] = browse
    owner._prepareInputWidgets[f"{key}_clear"] = clear
    return row, edit


def _add_labeled_row(
    owner: Any,
    form_layout: Any,
    translation_key: str,
    field: Any,
    *,
    storage_key: str | None = None,
) -> Any:
    """Add an explicit form label so its text can change at runtime."""

    import qt

    label = qt.QLabel()
    register_ui_language_binding(owner, label, "text", translation_key)
    owner._prepareInputWidgets[storage_key or f"{translation_key}_label"] = label
    form_layout.addRow(label, field)
    return label
