"""Qt builders for preparing external DWI inputs from the Slicer UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from shared.ui_onboarding import register_ui_language_binding


def install_prepare_inputs_panel(owner: Any, configuration_layout: Any) -> None:
    import ctk
    import qt

    panel = qt.QWidget()
    panel.setObjectName("prepareInputsPanel")
    layout = qt.QVBoxLayout(panel)
    layout.setContentsMargins(0, 5, 0, 0)
    layout.setSpacing(5)
    fields = {"panel": panel}
    owner._prepareInputWidgets = fields

    section = owner.ui.configurationCollapsibleButton
    header = qt.QWidget(section)
    header_layout = qt.QHBoxLayout(header)
    header_layout.setContentsMargins(2, 0, 0, 0)
    title = qt.QLabel(header)
    title.setObjectName("dspaceInputsTitle")
    header_layout.addWidget(title, 1)
    fields["title"] = title

    tabs = qt.QTabBar(panel)
    tabs.setObjectName("dspaceFormatTabs")
    tabs.setDrawBase(False)
    tabs.setExpanding(False)
    tabs.setUsesScrollButtons(True)
    formats = (("nifti", "NIfTI"), ("nrrd", "NRRD"), ("dicom", "DICOM"))
    for kind, caption in formats:
        tabs.setTabData(tabs.addTab(caption), kind)
    fields["format_tabs"] = tabs

    format_header = qt.QHBoxLayout()
    format_header.setSpacing(8)
    format_header.addWidget(tabs, 0, qt.Qt.AlignVCenter)
    drop_hint = qt.QLabel(panel)
    drop_hint.setObjectName("dspaceDropHint")
    drop_hint.setWordWrap(True)
    fields["drop_hint"] = drop_hint
    drop_hint.setMinimumWidth(0)
    drop_hint.setAlignment(qt.Qt.AlignRight | qt.Qt.AlignVCenter)
    format_header.addWidget(drop_hint, 1)
    layout.addLayout(format_header)
    mode_switch = qt.QWidget(header)
    mode_switch.setObjectName("dspaceInputMode")
    mode_layout = qt.QHBoxLayout(mode_switch)
    mode_layout.setContentsMargins(2, 2, 2, 2)
    mode_layout.setSpacing(0)
    mode_group = qt.QButtonGroup(mode_switch)
    for batch in (False, True):
        mode = qt.QToolButton(mode_switch)
        mode.setObjectName("dspaceModeOption")
        mode.setCheckable(True)
        mode.setChecked(not batch)
        mode.setCursor(qt.Qt.PointingHandCursor)
        mode.clicked.connect(lambda _checked=False, batch=batch: owner._update_parameter_node("batchMode", batch))
        mode_group.addButton(mode)
        mode_layout.addWidget(mode)
        fields["batch_mode" if batch else "single_mode"] = mode
    header_layout.addWidget(mode_switch, 0, qt.Qt.AlignVCenter)
    configuration_layout.insertWidget(0, header)

    pages = qt.QStackedWidget(panel)
    pages.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
    layout.addWidget(pages)
    fields["format_pages"] = pages
    batch_label = qt.QLabel(section)
    batch_root = ctk.ctkPathLineEdit(section)
    batch_root.setObjectName("dspaceBatchFolder")
    batch_root.filters = ctk.ctkPathLineEdit.Dirs
    batch_root.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)
    batch_root.currentPathChanged.connect(owner._onBatchRootPathChanged)
    form = owner.ui.configurationFormLayout
    for control in (owner.ui.subjectIdLabel, owner.ui.subjectIdLineEdit):
        form.takeAt(form.indexOf(control))
    subject_labels = qt.QStackedWidget(section)
    subject_fields = qt.QStackedWidget(section)
    subject_labels.addWidget(owner.ui.subjectIdLabel)
    subject_labels.addWidget(batch_label)
    subject_fields.addWidget(owner.ui.subjectIdLineEdit)
    subject_fields.addWidget(batch_root)
    subject_fields.setMinimumWidth(0)
    subject_fields.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)
    form.setWidget(0, qt.QFormLayout.LabelRole, subject_labels)
    form.setWidget(0, qt.QFormLayout.FieldRole, subject_fields)
    fields.update(batch_root=batch_root, batch_label=batch_label,
                  subject_labels=subject_labels, subject_fields=subject_fields)
    status = qt.QLabel()
    status.setWordWrap(True)
    status.setTextFormat(qt.Qt.PlainText)
    status.hide()
    fields["status"] = status

    def add_field(parent_layout, key, title, callback):
        tile = qt.QWidget()
        tile.setObjectName("dspaceInputTile")
        tile.setProperty("standalone", key == "mask")
        tile.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
        tile_layout = qt.QVBoxLayout(tile)
        tile_layout.setContentsMargins(12, 9, 11, 8)
        tile_layout.setSpacing(1)

        heading = qt.QHBoxLayout()
        heading.setSpacing(1)
        label = qt.QLabel(title, tile)
        label.setObjectName("dspaceInputName")
        label.setMinimumWidth(0)
        label.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
        heading.addWidget(label, 1)

        edit = qt.QPlainTextEdit(tile)
        edit.setObjectName("dspaceInput_" + key)
        edit.setAcceptDrops(True)
        edit.viewport().setAcceptDrops(True)
        edit.setFrameShape(qt.QFrame.NoFrame)
        edit.setLineWrapMode(qt.QPlainTextEdit.NoWrap)
        edit.setTabChangesFocus(True)
        edit.setVerticalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOff)
        edit.setHorizontalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOff)
        edit.document().setDocumentMargin(0)
        edit.setMinimumWidth(0)
        edit.setFixedHeight(20)

        clear = qt.QToolButton(tile)
        clear.setObjectName("dspaceInputAction")
        clear.setText("×")
        clear.setAutoRaise(True)
        clear.setFixedSize(15, 18)
        register_ui_language_binding(owner, clear, "tooltip", "clear")
        clear.clicked.connect(lambda: edit.clear())
        clear.hide()
        heading.addWidget(clear)

        browse = qt.QToolButton(tile)
        browse.setObjectName("dspaceInputBrowse")
        browse.setText("…")
        browse.setToolButtonStyle(qt.Qt.ToolButtonIconOnly)
        browse.setIconSize(qt.QSize(16, 16))
        browse.setAutoRaise(True)
        browse.setCursor(qt.Qt.PointingHandCursor)
        browse.setFixedSize(22, 20)
        register_ui_language_binding(owner, browse, "tooltip", "browse")
        browse.clicked.connect(lambda: callback(key))
        heading.addWidget(browse)
        file_row = qt.QWidget(tile)
        file_row.setFixedHeight(20)
        file_layout = qt.QStackedLayout(file_row)
        file_layout.setContentsMargins(0, 0, 0, 0)
        file_layout.setStackingMode(qt.QStackedLayout.StackAll)
        file_layout.addWidget(edit)
        hint = qt.QLabel(file_row)
        hint.setObjectName("dspaceInputHint")
        hint.setAttribute(qt.Qt.WA_TransparentForMouseEvents)
        hint.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)
        file_layout.addWidget(hint)
        file_layout.setCurrentWidget(hint)
        tile_layout.addLayout(heading)
        tile_layout.addWidget(file_row)
        parent_layout.addWidget(tile, 1)

        fields[key] = edit
        fields[key + "_tile"] = tile
        fields[key + "_label"] = label
        fields[key + "_hint"] = hint

        def text_changed():
            text = edit.toPlainText()
            if _assign_dropped_paths(owner, key, text):
                return
            path = edit.property("dspaceSelectedPath") or ""
            if not path or text != Path(path).name:
                path = text.strip()
                edit.setProperty("dspaceSelectedPath", path)
            tooltip = path or (hint.toolTip if key == "raw" else "")
            tile.setToolTip(tooltip)
            edit.setToolTip(tooltip)
            clear.setVisible(bool(path))
            hint.setVisible(not path)
            status.hide()
            owner._refresh_run_button_state()

        edit.textChanged.connect(text_changed)

    def format_row(kind):
        page = qt.QWidget()
        page.setObjectName("dspaceFileStrip")
        row = qt.QHBoxLayout(page)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(5)
        fields[kind + "_page"] = pages.addWidget(page)
        return row

    nifti = format_row("nifti")
    add_field(nifti, "nifti_source", "DWI", owner._browse_prepare_input_source_path)
    add_field(nifti, "bval", "bval", owner._browse_prepare_input_file)
    add_field(nifti, "bvec", "bvec", owner._browse_prepare_input_file)
    nrrd = format_row("nrrd")
    add_field(nrrd, "nrrd_source", "NRRD / NHDR", owner._browse_prepare_input_source_path)
    add_field(nrrd, "raw", "Detached data", owner._browse_prepare_input_file)
    dicom = format_row("dicom")
    add_field(dicom, "dicom_source", "DICOM folder", owner._browse_prepare_input_source_path)
    optional = qt.QHBoxLayout()
    add_field(optional, "mask", "Brain mask", owner._browse_prepare_input_file)
    layout.addLayout(optional)
    fields["mask_tile"].hide()
    layout.addWidget(status)

    def update_format(index):
        show_input_format(owner, tabs.tabData(index))
        owner._refresh_run_button_state()

    tabs.currentChanged.connect(update_format)
    show_input_format(owner, "nifti")

    def use_folder(_path):
        for key in ("nifti_source", "nrrd_source", "dicom_source", "bval", "bvec", "raw"):
            fields[key].clear()

    owner.ui.inputDirectoryPath.currentPathChanged.connect(use_folder)
    configuration_layout.addWidget(panel)
    apply_input_language(owner)
    refresh_prepare_inputs_styles(owner)


def show_input_format(owner, kind):
    fields = owner._prepareInputWidgets
    fields["source_kind"] = kind
    fields["source"] = fields[kind + "_source"]
    page = fields[kind + "_page"]
    fields["format_pages"].setCurrentIndex(page)
    tabs = fields["format_tabs"]
    if tabs.currentIndex != page:
        blocked = tabs.blockSignals(True)
        tabs.setCurrentIndex(page)
        tabs.blockSignals(blocked)
    _update_drop_hint(owner)


def sync_input_mode(owner):
    fields = getattr(owner, "_prepareInputWidgets", {})
    if "batch_root" not in fields:
        return
    node = owner._parameterNode
    batch = bool(getattr(node, "batchMode", False))
    fields["batch_mode" if batch else "single_mode"].setChecked(True)
    fields["panel"].setVisible(not batch)
    fields["subject_labels"].setCurrentIndex(int(batch))
    fields["subject_fields"].setCurrentIndex(int(batch))
    fields["mask_tile"].setVisible(not batch and getattr(node, "brainMaskBackend", "") == "provided")
    owner._sync_path_widget(fields["batch_root"], getattr(node, "batchRoot", "") or "")
    fields["single_mode"].setEnabled(not owner._pipelineActive)
    fields["batch_mode"].setEnabled(not owner._pipelineActive)
    fields["batch_root"].setEnabled(not owner._pipelineActive)
    if fields.get("active_batch_mode") != batch:
        fields["active_batch_mode"] = batch
        fit = getattr(owner, "_queueCompactFit", None)
        if callable(fit):
            fit()


def _update_drop_hint(owner):
    fields = owner._prepareInputWidgets
    zh = str(getattr(owner, "_onboardingLanguage", "en")).startswith("zh")
    dicom = fields["source_kind"] == "dicom"
    if zh:
        text = "拖入文件夹，或点 …" if dicom else "拖入文件，或点 …"
    else:
        text = "Drop a folder below or click …" if dicom else "Drop files below or click …"
    fields["drop_hint"].setText(text)


def apply_input_language(owner):
    _update_drop_hint(owner)
    fields = owner._prepareInputWidgets
    zh = str(getattr(owner, "_onboardingLanguage", "en")).startswith("zh")
    if "batch_root" in fields:
        from shared.ui_onboarding import onboarding_text

        fields["title"].setText(onboarding_text("zh" if zh else "en", "section_inputs"))
        owner.ui.configurationCollapsibleButton.setTitle("")
        fields["single_mode"].setText("单个" if zh else "Single")
        fields["batch_mode"].setText("批量" if zh else "Batch")
        fields["batch_label"].setText("批量输入目录" if zh else "Batch Folder")
        fields["batch_root"].setToolTip("选择包含多个受试者文件夹的目录。" if zh else
                                        "Select the folder containing the subject folders.")
    owner.ui.outputDirectoryLabel.setText("输出文件夹" if zh else "Output folder")
    owner.ui.outputDirectoryPath.setToolTip(
        "选择结果保存位置。留空则使用默认位置。" if zh else
        "Choose where to save results. Leave blank to use the default location."
    )
    for key, en, cn in (
        ("raw", "Detached data", "分离数据"),
        ("dicom_source", "DICOM folder", "DICOM 文件夹"),
        ("mask", "Brain mask", "脑掩膜"),
    ):
        fields[key + "_label"].setText(cn if zh else en)
    for key, en, cn in (
        ("nifti_source", ".nii / .nii.gz", ".nii / .nii.gz"),
        ("bval", ".bval / bvals", ".bval / bvals"),
        ("bvec", ".bvec / bvecs", ".bvec / bvecs"),
        ("nrrd_source", ".nrrd / .nhdr", ".nrrd / .nhdr"),
        ("raw", "Required for NHDR", "NHDR 必需"),
        ("dicom_source", "DICOM series", "DICOM 序列"),
        ("mask", "Drop or browse", "拖入或选择"),
    ):
        fields[key + "_hint"].setText(cn if zh else en)
        fields[key].setAccessibleName(fields[key + "_label"].text)
    raw_help = (
        "可按 NHDR 头文件中的路径自动定位；找不到时再选择数据文件。" if zh else
        "The NHDR header points to its data file. Select it here if the referenced file cannot be found."
    )
    fields["raw_hint"].setToolTip(raw_help)
    if not fields["raw"].toPlainText():
        fields["raw"].setToolTip(raw_help)
        fields["raw_tile"].setToolTip(raw_help)


def refresh_prepare_inputs_styles(owner: Any, *, dark: bool | None = None) -> None:
    import qt
    from shared.ui_styles import is_dark_mode

    fields = getattr(owner, "_prepareInputWidgets", {})
    panel = fields.get("panel")
    if panel is None:
        return
    dark = is_dark_mode(panel) if dark is None else dark
    if dark:
        base, line, muted, ink, hover, action = (
            "#292a2b", "#444547", "#b7b8ba", "#f2f2f2", "#363739", "#393a3b"
        )
    else:
        base, line, muted, ink, hover, action = (
            "#fcfcfc", "#dedfdf", "#606367", "#222426", "#f2f3f3", "#f0f1f1"
        )
    drop_line = "#85898d" if dark else "#969b9f"
    section = owner.ui.configurationCollapsibleButton
    from shared.ui_algorithm_picker import ellipsis_icon

    icon = ellipsis_icon("#f2f2f2" if dark else "#334155")
    for key in ("nifti_source", "bval", "bvec", "nrrd_source", "raw", "dicom_source", "mask"):
        for button in fields[key + "_tile"].findChildren(qt.QToolButton):
            if button.objectName == "dspaceInputBrowse":
                button.setIcon(icon)
    output_icon = ellipsis_icon("#8CD2F5" if dark else "#326A8A")
    for path_widget in (owner.ui.outputDirectoryPath, fields["batch_root"]):
        for button in path_widget.findChildren(qt.QToolButton):
            button.setObjectName("dspaceFolderBrowse")
            button.setToolButtonStyle(qt.Qt.ToolButtonIconOnly)
            button.setIconSize(qt.QSize(16, 16))
            button.setIcon(output_icon)
            button.setFixedSize(28, 22)
            button.setCursor(qt.Qt.PointingHandCursor)
    section.setStyleSheet(
        f"QWidget#dspaceInputMode {{ background: {action}; border: 1px solid {line}; border-radius: 12px; }}"
        f"QToolButton#dspaceModeOption {{ color: {muted}; background: transparent; border: 1px solid transparent;"
        " border-radius: 9px; padding: 1px 7px; min-height: 16px; font-size: 11px; }"
        f"QToolButton#dspaceModeOption:checked {{ background: {base}; color: {ink}; border-color: {line}; }}"
        f"QToolButton#dspaceModeOption:hover:!disabled {{ color: {ink}; }}"
        f"QLabel#subjectIdLabel, QLabel#inputDirectoryLabel {{ color: {ink}; font-weight: 400; }}"
        f"QLineEdit#subjectIdLineEdit, ctkPathLineEdit#outputDirectoryPath QComboBox {{ border-color: {line}; }}"
        f"QTabBar#dspaceFormatTabs::tab {{ background: transparent; color: {muted};"
        " border: 0; border-bottom: 2px solid transparent; padding: 5px 9px 4px; }"
        f"QTabBar#dspaceFormatTabs::tab:hover {{ background: {hover}; }}"
        f"QTabBar#dspaceFormatTabs::tab:selected {{ color: {ink}; border-bottom-color: {ink}; font-weight: 600; }}"
        f"QLabel#dspaceInputsTitle {{ color: {ink}; font-size: 16px; font-weight: 700; }}"
        "QWidget#dspaceFileStrip { background: transparent; border: 0; }"
        f"QWidget#dspaceInputTile {{ background: {base}; border: 2px dashed {drop_line}; border-radius: 10px; }}"
        f"QLabel#dspaceInputName {{ background: transparent; color: {ink}; border: 0; font-weight: 600; }}"
        f"QLabel#dspaceInputHint, QLabel#dspaceDropHint {{ background: transparent; color: {muted}; border: 0; font-size: 11px; }}"
        f"QWidget#dspaceInputTile QPlainTextEdit {{ background: transparent; color: {ink};"
        " border: 0; padding: 0; margin: 0; font-size: 11px; }"
        f"QToolButton#dspaceInputAction, QToolButton#dspaceInputBrowse, QToolButton#dspaceFolderBrowse {{ background: {action}; color: {ink}; border: 1px solid transparent;"
        " border-radius: 4px; padding: 0; }"
        f"QToolButton#dspaceInputAction:hover, QToolButton#dspaceInputAction:focus,"
        " QToolButton#dspaceInputBrowse:hover, QToolButton#dspaceInputBrowse:focus,"
        " QToolButton#dspaceFolderBrowse:hover, QToolButton#dspaceFolderBrowse:focus"
        f" {{ background: {hover}; border-color: #478CB3; }}"
    )


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
    import qt

    urls = re.findall(r"file:/+[^\r\n]+?(?=file:/+|[\r\n]|$)", str(text))
    if not urls:
        return False
    paths = [Path(qt.QUrl(url.strip()).toLocalFile()) for url in urls]
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
    previous = owner._prepare_inputs_widget_text(field)
    owner._set_prepare_inputs_widget_text(field, previous)
    if detected_kind:
        show_input_format(owner, detected_kind)
    for key, path in assigned.items():
        owner._set_prepare_inputs_widget_text(key, str(path))
    return True


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
