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

    source_kind = qt.QTabBar()
    source_kind.setExpanding(False)
    source_kind.setUsesScrollButtons(True)
    for kind, label in (("nifti", "NIfTI"), ("nrrd", "NRRD / NHDR"),
                        ("dicom", "DICOM"), ("canonical", "Prepared folder")):
        index = source_kind.addTab(label)
        source_kind.setTabData(index, kind)
    owner._prepareInputWidgets["source_kind"] = source_kind
    form_layout.addRow(source_kind)

    source_row, source_edit = _path_row(
        owner,
        key="source",
        placeholder="DICOM folder, DWI NIfTI, NRRD/NHDR, or canonical subject root",
        browse_callback=owner._browse_prepare_input_source_path,
    )
    _add_labeled_row(owner, form_layout, "source", source_row)
    owner._prepareInputWidgets["source"] = source_edit
    source_edit.textChanged.connect(lambda *_: owner._refresh_run_button_state())
    raw_row, raw_edit = _path_row(
        owner, key="raw", placeholder="Detached NHDR data (.raw, .raw.gz)",
        browse_callback=owner._browse_prepare_input_file,
    )
    raw_label = qt.QLabel()
    raw_label.setText("Detached data (.raw, .raw.gz)")
    form_layout.addRow(raw_label, raw_row)
    owner._prepareInputWidgets["raw"] = raw_edit
    owner._prepareInputWidgets["raw_label"] = raw_label
    raw_row.hide()
    raw_label.hide()

    sidecar_rows = []
    for key, label_key, placeholder in (
        ("bval", "bval", "Optional .bval sidecar"),
        ("bvec", "bvec", "Optional .bvec sidecar"),
        ("metadata", "metadata_json", "Optional BIDS-style .json sidecar"),
        ("mask", "brain_mask", "Optional nodif_brain_mask.nii.gz"),
    ):
        row, edit = _path_row(owner, key=key, placeholder=placeholder, browse_callback=owner._browse_prepare_input_file)
        label = _add_labeled_row(owner, form_layout, label_key, row, storage_key=f"{key}_label")
        if key in {"bval", "bvec"}:
            sidecar_rows.extend((label, row))
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
    status = qt.QLabel()
    status.setWordWrap(True)
    status.setTextFormat(qt.Qt.PlainText)
    owner._prepareInputWidgets["status"] = status
    form_layout.addRow(status)
    for key in ("source", "bval", "bvec", "raw", "metadata", "mask"):
        owner._prepareInputWidgets[key].textChanged.connect(lambda *_: status.clear())

    def update_format(index):
        raw_row.setVisible(index == 1)
        raw_label.setVisible(index == 1)
        for widget in sidecar_rows:
            widget.setVisible(index == 0)
        fit = getattr(owner, "_queueCompactFit", None)
        if callable(fit):
            fit()

    source_kind.currentChanged.connect(update_format)
    for key in ("source", "raw", "bval", "bvec"):
        edit = owner._prepareInputWidgets[key]
        edit.textChanged.connect(lambda value, key=key: _assign_dropped_paths(owner, key, value))
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
        "raw": None,
        "bval": "bval_placeholder",
        "bvec": "bvec_placeholder",
        "metadata": "metadata_placeholder",
        "mask": "brain_mask_placeholder",
    }[key]
    if placeholder_key:
        register_ui_language_binding(owner, edit, "placeholder", placeholder_key)
        register_ui_language_binding(owner, edit, "tooltip", placeholder_key)
    row_layout.addWidget(edit, 1)
    browse = qt.QPushButton("Browse...")
    browse.setToolTip(placeholder)
    register_ui_language_binding(owner, browse, "text", "browse")
    if placeholder_key:
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


def import_dicom_series(config, series):
    """Convert only the chosen series using the standard input importer."""
    from dataclasses import replace
    import json
    import os
    from pathlib import Path
    import shutil
    from tempfile import TemporaryDirectory

    from shared.dicom_input import _validate_converted, converter_path
    from shared.input_import import prepare_inputs

    if config.backend == "auto":
        backend, executable = converter_path()
        config = replace(config, backend=backend,
                         dcm2niix_executable=executable if backend == "dcm2niix" else None,
                         dwiconvert_executable=executable if backend == "dwiconvert" else None)
    with TemporaryDirectory(prefix="slicerdspace-series-") as temporary:
        staged = Path(temporary)
        for index, original in enumerate(series["paths"]):
            target = staged / f"{index:08d}.dcm"
            try:
                os.link(original, target)
            except OSError:
                shutil.copyfile(original, target)
        manifest = prepare_inputs(replace(config, source=staged))
    manifest["plan"]["dicom_series"] = {
        "source": str(config.source),
        "series_instance_uid": series["uid"],
        "instances": [str(path.relative_to(config.source)) for path in series["paths"]],
    }
    if not config.dry_run:
        input_dir = config.subject_root / "input"
        receipt = input_dir / "input_import_manifest.json"
        try:
            if config.output_kind == "canonical-nifti":
                _validate_converted(input_dir)
        except Exception:
            receipt.unlink(missing_ok=True)
            raise
        receipt.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
    return manifest


def _assign_dropped_paths(owner, field, text):
    """Route local-file URLs supplied by Qt's native text drop support."""
    import re
    from pathlib import Path
    import qt

    urls = re.findall(r"file:/+[^\s]+?(?=file:/+|\s|$)", str(text))
    if not urls:
        return
    paths = [Path(qt.QUrl(url).toLocalFile()) for url in urls]
    assigned = {}
    detected_kind = None
    for path in paths:
        name = path.name.lower()
        if name.endswith(".bval") or name == "bvals":
            assigned["bval"] = path
        elif name.endswith(".bvec") or name == "bvecs":
            assigned["bvec"] = path
        elif name.endswith((".raw", ".raw.gz")):
            assigned["raw"] = path
        elif path.is_dir() or name.endswith((".nii", ".nii.gz", ".nrrd", ".nhdr")):
            assigned["source"] = path
            detected_kind = "dicom" if path.is_dir() else "nrrd" if name.endswith((".nrrd", ".nhdr")) else "nifti"
    owner._set_prepare_inputs_widget_text(field, "")
    if detected_kind:
        source_kind = owner._prepareInputWidgets["source_kind"]
        for index in range(source_kind.count):
            if source_kind.tabData(index) == detected_kind:
                source_kind.setCurrentIndex(index)
                break
    for key, path in assigned.items():
        owner._set_prepare_inputs_widget_text(key, str(path))


def import_detached_nrrd(config, raw):
    """Use the selected detached data with a temporary copy of the NHDR."""
    from dataclasses import replace
    import json
    from pathlib import Path
    import re
    from tempfile import TemporaryDirectory

    from shared.input_import import prepare_inputs

    if config.source.suffix.lower() != ".nhdr" or not raw.is_file():
        raise ValueError("Select an NHDR header and an existing detached data file.")
    header = config.source.read_text(encoding="ascii")
    pattern = r"^data\s*file\s*:\s*(.+)$"
    match = re.search(pattern, header, re.IGNORECASE | re.MULTILINE)
    if match is None or match.group(1).upper().startswith("LIST") or "%" in match.group(1):
        raise ValueError("A single detached data file is required for this header override.")
    with TemporaryDirectory(prefix="slicerdspace-nhdr-") as temporary:
        staged = Path(temporary) / config.source.name
        header = re.sub(pattern, lambda _: f"data file: {raw.as_posix()}", header,
                        count=1, flags=re.IGNORECASE | re.MULTILINE)
        staged.write_text(header, encoding="ascii")
        manifest = prepare_inputs(replace(config, source=staged))
    manifest["plan"]["detached_nrrd"] = {"header": str(config.source), "data": str(raw)}
    if not config.dry_run:
        (config.subject_root / "input" / "input_import_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
    return manifest
