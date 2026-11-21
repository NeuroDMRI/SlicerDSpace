"""Qt builders for preparing external DWI inputs from the Slicer UI."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from shared.ui_onboarding import onboarding_text, register_ui_language_binding

# Package C1: every Prepare Inputs tile, in the order they are saved, and the format tabs.
INPUT_TILE_KEYS = ("nifti_source", "bval", "bvec", "nrrd_source", "raw", "dicom_source", "mask")
INPUT_FORMAT_KINDS = ("nifti", "nrrd", "dicom")
# The hint under an empty tile: (tile key, English, Chinese).
INPUT_TILE_HINTS = (
    ("nifti_source", ".nii / .nii.gz", ".nii / .nii.gz"),
    ("bval", ".bval / bvals", ".bval / bvals"),
    ("bvec", ".bvec / bvecs", ".bvec / bvecs"),
    ("nrrd_source", ".nrrd / .nhdr", ".nrrd / .nhdr"),
    ("raw", "Required for NHDR", "NHDR 必需"),
    ("dicom_source", "DICOM series", "DICOM 序列"),
    ("mask", "Drop or browse", "拖入或选择"),
)


def _extract_drop_paths(qt: Any, mime_data: Any) -> list[Path]:
    if mime_data is None:
        return []
    paths: list[Path] = []
    seen: set[Path] = set()
    urls_fn = getattr(mime_data, "urls", None)
    has_urls_fn = getattr(mime_data, "hasUrls", None)
    if callable(urls_fn) and (not callable(has_urls_fn) or has_urls_fn()):
        for url in urls_fn() or ():
            is_local = getattr(url, "isLocalFile", None)
            if callable(is_local) and not is_local():
                continue
            local = str(url.toLocalFile() or "").strip() if hasattr(url, "toLocalFile") else ""
            if local:
                cand = Path(local)
                if cand not in seen:
                    seen.add(cand)
                    paths.append(cand)
    if not paths:
        has_text_fn = getattr(mime_data, "hasText", None)
        text_fn = getattr(mime_data, "text", None)
        if callable(text_fn) and (not callable(has_text_fn) or has_text_fn()):
            raw_text = str(text_fn() or "")
            for url_text in re.findall(r"file:/+[^\r\n]+?(?=file:/+|[\r\n]|$)", raw_text):
                local = str(qt.QUrl(url_text.strip()).toLocalFile() or "").strip()
                if local:
                    cand = Path(local)
                    if cand not in seen:
                        seen.add(cand)
                        paths.append(cand)
    return paths


def _set_tile_drag_active(tile: Any, active: bool) -> None:
    if bool(tile.property("dragActive")) == bool(active):
        return
    tile.setProperty("dragActive", bool(active))
    style_fn = getattr(tile, "style", None)
    style = style_fn() if callable(style_fn) else style_fn
    if style is not None:
        if hasattr(style, "unpolish"):
            style.unpolish(tile)
        if hasattr(style, "polish"):
            style.polish(tile)
    if hasattr(tile, "update"):
        tile.update()


_DROP_TILE_CLASSES: dict[type, type] = {}


def _make_input_drop_tile(qt: Any, owner: Any, key: str) -> tuple[Any, bool]:
    """Return an input tile widget; subclass QFrame so the entire tile handles drops without an eventFilter."""
    qframe_cls = getattr(qt, "QFrame", None)
    if isinstance(qframe_cls, type):
        tile_cls = _DROP_TILE_CLASSES.get(qframe_cls)
        if tile_cls is None:
            class _InputDropTile(qframe_cls):
                def dragEnterEvent(self, event):
                    qt_mod, drop_owner, _drop_key = getattr(self, "_dspaceDropCtx", (None, None, ""))
                    mime = event.mimeData() if hasattr(event, "mimeData") else None
                    if qt_mod is not None and _extract_drop_paths(qt_mod, mime):
                        _set_tile_drag_active(self, True)
                        if hasattr(event, "acceptProposedAction"):
                            event.acceptProposedAction()
                        return
                    if hasattr(event, "ignore"):
                        event.ignore()

                def dragMoveEvent(self, event):
                    self.dragEnterEvent(event)

                def dragLeaveEvent(self, event):
                    _set_tile_drag_active(self, False)

                def dropEvent(self, event):
                    _set_tile_drag_active(self, False)
                    qt_mod, drop_owner, drop_key = getattr(self, "_dspaceDropCtx", (None, None, ""))
                    mime = event.mimeData() if hasattr(event, "mimeData") else None
                    paths = _extract_drop_paths(qt_mod, mime) if qt_mod is not None else []
                    if paths and _assign_dropped_file_paths(drop_owner, drop_key, paths):
                        if hasattr(event, "acceptProposedAction"):
                            event.acceptProposedAction()
                        return
                    if hasattr(event, "ignore"):
                        event.ignore()

            tile_cls = _InputDropTile
            _DROP_TILE_CLASSES[qframe_cls] = tile_cls

        tile = tile_cls()
        tile._dspaceDropCtx = (qt, owner, key)
        no_frame = getattr(qframe_cls, "NoFrame", None)
        if no_frame is not None and hasattr(tile, "setFrameShape"):
            tile.setFrameShape(no_frame)
        return tile, True
    return qt.QWidget(), False


def place_input_preview_button(owner: Any, batch: Any = None) -> None:
    """Keep 'View input' aligned next to NIfTI/NRRD/DICOM in Single mode, and below Batch Folder in Batch mode."""
    import qt

    preview = getattr(owner, "_inputPreviewWidgets", None)
    fields = getattr(owner, "_prepareInputWidgets", None)
    if not isinstance(preview, dict) or "button" not in preview:
        return
    button = preview["button"]
    preview_layout = preview.get("layout")
    format_header = fields.get("format_header") if isinstance(fields, dict) else None
    if batch is None:
        node = getattr(owner, "_parameterNode", None)
        is_batch = bool(getattr(node, "batchMode", False)) if node is not None else False
    else:
        is_batch = bool(batch)
    if not is_batch and format_header is not None:
        if preview_layout is not None and preview_layout.indexOf(button) >= 0:
            preview_layout.removeWidget(button)
        if format_header.indexOf(button) < 0:
            format_header.addWidget(button, 0, qt.Qt.AlignRight | qt.Qt.AlignVCenter)
    else:
        if format_header is not None and format_header.indexOf(button) >= 0:
            format_header.removeWidget(button)
        if preview_layout is not None and preview_layout.indexOf(button) < 0:
            preview_layout.insertWidget(0, button, 0, qt.Qt.AlignLeft)
    if hasattr(button, "show"):
        button.show()


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
    format_header.addStretch(1)
    fields["format_header"] = format_header
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
        tile, handles_drops = _make_input_drop_tile(qt, owner, key)
        tile.setObjectName("dspaceInputTile")
        tile.setProperty("standalone", key == "mask")
        tile.setAcceptDrops(True)
        styled_bg = getattr(qt.Qt, "WA_StyledBackground", None)
        if styled_bg is not None and hasattr(tile, "setAttribute"):
            tile.setAttribute(styled_bg, True)
        tile.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
        tile_layout = qt.QVBoxLayout(tile)
        tile_layout.setContentsMargins(12, 9, 11, 8)
        tile_layout.setSpacing(1)

        heading = qt.QHBoxLayout()
        heading.setSpacing(1)
        label = qt.QLabel(title, tile)
        label.setObjectName("dspaceInputName")
        label.setAttribute(qt.Qt.WA_TransparentForMouseEvents)
        label.setMinimumWidth(0)
        label.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
        heading.addWidget(label, 1)

        edit = qt.QPlainTextEdit(tile)
        edit.setObjectName("dspaceInput_" + key)
        edit.setAcceptDrops(not handles_drops)
        edit.viewport().setAcceptDrops(not handles_drops)
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
            try:
                update_input_hint(owner, key)
            except Exception:  # FC2: the hint must never skip the status, run-button, algorithm and save steps
                pass
            status.hide()
            owner._refresh_run_button_state()
            if key in {"nifti_source", "nrrd_source", "dicom_source", "bval", "bvec"}:
                # B3: a new input can change which algorithms its data can run.
                from shared.ui_compact import refresh_compact_algorithms

                refresh_compact_algorithms(owner)
            _schedule_settings_save(owner)  # C1: a tile edit does not modify the parameter node

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
    drop_hint = qt.QLabel(panel)
    drop_hint.setObjectName("dspaceDropHint")
    drop_hint.setWordWrap(True)
    drop_hint.setMinimumWidth(0)
    drop_hint.setAlignment(qt.Qt.AlignLeft | qt.Qt.AlignVCenter)
    drop_hint.setContentsMargins(2, 1, 0, 0)
    fields["drop_hint"] = drop_hint
    layout.addWidget(drop_hint)
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
    _install_refresh_button(owner, qt, section, fields)
    apply_input_language(owner)
    refresh_prepare_inputs_styles(owner)
    apply_pending_input_sources(owner)  # C1: a restore that ran before this panel existed


def show_input_format(owner, kind):
    fields = owner._prepareInputWidgets
    previous_kind = fields.get("source_kind")
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
    if previous_kind is not None and previous_kind != kind:
        # F4: another tab is another source, so the family summary and its tooltip change with it.
        from shared.ui_compact import refresh_compact_algorithms

        refresh_compact_algorithms(owner)
        _schedule_settings_save(owner)  # C1: the format tab is remembered


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
    place_input_preview_button(owner, batch=batch)
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
        text = "Drop a folder above or click …" if dicom else "Drop files above or click …"
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
    for key, en, cn in INPUT_TILE_HINTS:
        fields[key + "_hint"].setText(cn if zh else en)
        fields[key].setAccessibleName(fields[key + "_label"].text)
    if callable(getattr(owner, "_prepare_inputs_widget_text", None)):
        refresh_input_hints(owner)  # C1.5: a missing file keeps its warning in the new language
    refresh_button = fields.get("refresh_button")
    if refresh_button is not None:
        text = onboarding_text("zh" if zh else "en", "refresh_inputs_tooltip")
        refresh_button.setToolTip(text)
        refresh_button.setAccessibleName(text)
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
    missing_ink = "#ff8f8a" if dark else "#b3261e"
    section = owner.ui.configurationCollapsibleButton
    from shared.ui_algorithm_picker import ellipsis_icon

    icon = ellipsis_icon("#f2f2f2" if dark else "#334155")
    for key in ("nifti_source", "bval", "bvec", "nrrd_source", "raw", "dicom_source", "mask"):
        for button in fields[key + "_tile"].findChildren(qt.QToolButton):
            if button.objectName == "dspaceInputBrowse":
                button.setIcon(icon)
    output_icon = ellipsis_icon("#8CD2F5" if dark else "#326A8A")
    refresh_button = fields.get("refresh_button")
    if refresh_button is not None:
        # The same ink as the dspaceFolderBrowse button beside it.
        refresh_icon = inputs_refresh_icon("#8CD2F5" if dark else "#326A8A")
        if refresh_icon is not None:
            refresh_button.setIcon(refresh_icon)
        else:  # Qt could not load the artwork: show a text glyph instead of an empty button
            refresh_button.setToolButtonStyle(qt.Qt.ToolButtonTextOnly)
            refresh_button.setText("\u21bb")
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
        f"QWidget#dspaceInputTile[dragActive=\"true\"] {{ border: 2px dashed {'#7CC4EA' if dark else '#335B70'}; background: {hover}; }}"
        f"QPushButton#inputPreviewButton {{ background: {action}; color: {ink}; border: 1px solid {line}; border-radius: 5px; padding: 2px 10px; font-size: 11px; min-height: 20px; }}"
        f"QPushButton#inputPreviewButton:hover:!disabled {{ background: {hover}; border-color: #478CB3; }}"
        f"QPushButton#inputPreviewButton:disabled {{ color: {muted}; background: transparent; border-color: {line}; }}"
        f"QLabel#dspaceInputName {{ background: transparent; color: {ink}; border: 0; font-weight: 600; }}"
        f"QLabel#dspaceInputHint, QLabel#dspaceDropHint {{ background: transparent; color: {muted}; border: 0; font-size: 11px; }}"
        f"QWidget#dspaceInputTile QPlainTextEdit {{ background: transparent; color: {ink};"
        " border: 0; padding: 0; margin: 0; font-size: 11px; }"
        f"QToolButton#dspaceInputAction, QToolButton#dspaceInputBrowse, QToolButton#dspaceFolderBrowse,"
        f" QToolButton#inputsRefreshButton {{ background: {action}; color: {ink}; border: 1px solid transparent;"
        " border-radius: 4px; padding: 0; }"
        f"QToolButton#dspaceInputAction:hover, QToolButton#dspaceInputAction:focus,"
        " QToolButton#dspaceInputBrowse:hover, QToolButton#dspaceInputBrowse:focus,"
        " QToolButton#dspaceFolderBrowse:hover, QToolButton#dspaceFolderBrowse:focus,"
        " QToolButton#inputsRefreshButton:hover, QToolButton#inputsRefreshButton:focus"
        f" {{ background: {hover}; border-color: #478CB3; }}"
        # C1.5: a saved file that no longer exists keeps its name and shows "File not found" on the right.
        f"QWidget#dspaceInputTile[missing=\"true\"] QLabel#dspaceInputHint {{ color: {missing_ink}; }}"
        "QWidget#dspaceInputTile[missing=\"true\"] QPlainTextEdit { padding-right: 96px; }"
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


def _assign_dropped_file_paths(owner, field, paths, *, restore_field=False):
    """Route dropped local file/directory Paths across the input tiles."""
    assigned = {}
    detected_kind = None
    clean_paths = [Path(p) for p in paths if str(p or "").strip()]
    if not clean_paths:
        return False
    if field == "mask" and len(clean_paths) == 1 and not clean_paths[0].is_dir():
        assigned["mask"] = clean_paths[0]
    else:
        for path in clean_paths:
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
        if not assigned and field == "raw" and len(clean_paths) == 1 and not clean_paths[0].is_dir():
            assigned["raw"] = clean_paths[0]
    if not assigned:
        return False
    if restore_field:
        previous = owner._prepare_inputs_widget_text(field)
        owner._set_prepare_inputs_widget_text(field, previous)
    if detected_kind:
        show_input_format(owner, detected_kind)
    for key, path in assigned.items():
        owner._set_prepare_inputs_widget_text(key, str(path))
    return True


def _assign_dropped_paths(owner, field, text):
    """Route local-file URLs supplied by Qt's native text drop support."""
    import qt

    urls = re.findall(r"file:/+[^\r\n]+?(?=file:/+|[\r\n]|$)", str(text))
    if not urls:
        return False
    paths = [Path(qt.QUrl(url.strip()).toLocalFile()) for url in urls]
    if not _assign_dropped_file_paths(owner, field, paths, restore_field=True):
        previous = owner._prepare_inputs_widget_text(field)
        owner._set_prepare_inputs_widget_text(field, previous)
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


# ----- Package C1: remember the input tiles ---------------------------------------------------------------------


def _schedule_settings_save(owner: Any) -> None:
    """A tile edit or a format switch does not modify the parameter node, so it asks for the debounced save itself."""
    schedule = getattr(owner, "_schedule_common_user_settings_save", None)
    if callable(schedule):
        schedule()


def collect_input_sources(owner: Any) -> dict[str, str] | None:
    """``recent.input_sources``: the format tab and the full path of every tile, or ``None`` when it must not be saved.

    ``None`` means: no panel yet, or a restore still waits for the panel. Saving then would replace the stored
    paths with empty ones. Every tile key is returned (empty when unset) because the config merge keeps omitted keys.
    """
    fields = getattr(owner, "_prepareInputWidgets", None)
    read = getattr(owner, "_prepare_inputs_widget_text", None)
    if not isinstance(fields, dict) or "source_kind" not in fields or not callable(read):
        return None
    if getattr(owner, "_pendingInputSources", None):
        return None
    sources = {"kind": str(fields.get("source_kind") or "")}
    for key in INPUT_TILE_KEYS:
        sources[key] = read(key)
    return sources


def restore_input_sources(owner: Any, recent: Any) -> None:
    """Put the saved tiles and format tab back. Call it only for a new, empty parameter node."""
    sources = recent.get("input_sources") if hasattr(recent, "get") else None
    if not isinstance(sources, dict) or not any(key in sources for key in ("kind",) + INPUT_TILE_KEYS):
        return  # an old config file (no saved selection): the tiles stay as they are
    owner._pendingInputSources = dict(sources)
    apply_pending_input_sources(owner)


def apply_pending_input_sources(owner: Any) -> bool:
    """Apply the waiting restore once, as soon as the panel exists. Return whether it ran."""
    pending = getattr(owner, "_pendingInputSources", None)
    fields = getattr(owner, "_prepareInputWidgets", None)
    if not pending or not isinstance(fields, dict) or "source_kind" not in fields:
        return False
    owner._pendingInputSources = None
    log = getattr(owner, "appendLog", None)

    def guarded(step, label):
        try:
            step()
        except Exception as exc:  # one failing tile must not stop the others, or the module load
            if callable(log):
                log(f"[Settings] Saved input {label} could not be restored: {exc}")

    kind = str(pending.get("kind") or "")
    if kind in INPUT_FORMAT_KINDS:
        guarded(lambda: show_input_format(owner, kind), "format")
    for key in INPUT_TILE_KEYS:
        # FC1: reproduce the saved selection exactly. A missing or empty key clears the tile, so a panel reused
        # after a scene close cannot keep another subject's bval/bvec next to the restored DWI.
        # A missing file stays as text: the user sees which one is gone (see update_input_hint).
        value = str(pending.get(key) or "").strip()
        guarded(lambda key=key, value=value: owner._set_prepare_inputs_widget_text(key, value), key)
    return True


def update_input_hint(owner: Any, key: str) -> None:
    """Show the normal hint of an empty tile, or "File not found" when the saved path no longer exists.

    The text of the tile is never changed here. The path is tested with ``exists`` because the DICOM tile holds a folder.
    """
    fields = getattr(owner, "_prepareInputWidgets", None) or {}
    hint = fields.get(key + "_hint")
    if hint is None:
        return
    read = getattr(owner, "_prepare_inputs_widget_text", None)
    path = read(key) if callable(read) else ""
    try:
        missing = bool(path) and not Path(path).expanduser().exists()
    except OSError:
        missing = False
    except Exception:  # FC2: e.g. "~nouser/dwi.nii.gz" (RuntimeError from expanduser) names no reachable file
        missing = bool(path)
    zh = str(getattr(owner, "_onboardingLanguage", "en")).startswith("zh")
    if missing:
        text = onboarding_text("zh" if zh else "en", "input_file_missing")
    else:
        text = next((cn if zh else en for name, en, cn in INPUT_TILE_HINTS if name == key), "")
    hint.setText(text)
    hint.setVisible(bool(missing or not path))
    try:
        import qt

        side = qt.Qt.AlignRight if missing else qt.Qt.AlignLeft
        hint.setAlignment(side | qt.Qt.AlignVCenter)
    except Exception:
        pass
    tile = fields.get(key + "_tile")
    if tile is not None:
        was_missing = bool(tile.property("missing"))
        tile.setProperty("missing", bool(missing))
        if was_missing != bool(missing):
            # Qt does not restyle the children of a widget whose dynamic property changed, and the rules for the
            # hint and the edit are selected by the tile's property.
            for widget in (tile, hint, fields.get(key)):
                _repolish(widget)


def _repolish(widget: Any) -> None:
    if widget is None:
        return
    style_fn = getattr(widget, "style", None)
    style = style_fn() if callable(style_fn) else style_fn
    if style is not None:
        getattr(style, "unpolish", lambda target: None)(widget)
        getattr(style, "polish", lambda target: None)(widget)
    if hasattr(widget, "update"):
        widget.update()


def refresh_input_hints(owner: Any) -> None:
    """Re-check every tile path (the Refresh button and a language change)."""
    for key in INPUT_TILE_KEYS:
        try:
            update_input_hint(owner, key)
        except Exception:
            continue


# ----- Package C2: Refresh button --------------------------------------------------------------------------------


def inputs_refresh_icon(ink: str) -> Any:
    """A circular arrow drawn in code in the ink of the theme, or ``None`` when Qt cannot load the artwork."""
    import qt

    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24">'
        f'<g fill="none" stroke="{ink}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M20 12a8 8 0 1 1-2.34-5.66L19.8 8.5"/><polyline points="19.8 4 19.8 8.5 15.3 8.5"/></g></svg>'
    )
    pixmap = qt.QPixmap()
    if not pixmap.loadFromData(qt.QByteArray(svg.encode("utf-8"))):
        return None
    pixmap.setDevicePixelRatio(2)
    return qt.QIcon(pixmap)


def _form_field_row(form: Any, qt: Any, widget: Any, default: int) -> int:
    """The form row whose field is ``widget`` (the Output folder row is row 2 of ``configurationFormLayout``)."""
    try:
        for row in range(int(form.rowCount())):
            item = form.itemAt(row, qt.QFormLayout.FieldRole)
            if item is not None and item.widget() == widget:
                return row
    except Exception:
        pass
    return default


def _install_refresh_button(owner: Any, qt: Any, section: Any, fields: dict) -> None:
    """Put the Refresh button at the right end of the Output folder row, beside ``outputDirectoryPath``."""
    form = owner.ui.configurationFormLayout
    path = owner.ui.outputDirectoryPath
    row = _form_field_row(form, qt, path, 2)
    form.takeAt(form.indexOf(path))
    container = qt.QWidget(section)
    container.setMinimumWidth(0)
    container.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)
    layout = qt.QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    button = qt.QToolButton(container)
    button.setObjectName("inputsRefreshButton")
    button.setToolButtonStyle(qt.Qt.ToolButtonIconOnly)
    button.setIconSize(qt.QSize(16, 16))
    button.setFixedSize(28, 22)
    button.setCursor(qt.Qt.PointingHandCursor)
    register_ui_language_binding(owner, button, "tooltip", "refresh_inputs_tooltip")
    button.clicked.connect(lambda _checked=False: owner._on_refresh_inputs_clicked())
    layout.addWidget(path, 1)
    layout.addWidget(button, 0)
    form.setWidget(row, qt.QFormLayout.FieldRole, container)
    fields["refresh_button"] = button
    fields["output_row"] = container


def _inputs_busy(owner: Any) -> bool:
    return bool(
        getattr(owner, "_pipelineActive", False) or getattr(owner, "_setupJob", None)
        or getattr(owner, "_selectedSetupActive", False) or getattr(owner, "_inputImportPending", False)
    )


def _drop_input_caches(owner: Any) -> None:
    owner._inputInspectionCache = None
    owner._algorithmGateCache = None  # package B: the short-lived gate-input cache
    from shared import algorithm_availability

    algorithm_availability.clear_cache()


def _refresh_run_state(owner: Any) -> None:
    topup = getattr(owner, "_refresh_topup_controls", None)
    owner._refresh_run_button_state(topup() if callable(topup) else None)


def _refresh_algorithms(owner: Any) -> None:
    from shared.ui_compact import refresh_compact_algorithms

    refresh_compact_algorithms(owner)


def on_refresh_inputs_clicked(owner: Any) -> str:
    """Re-check the input files and rescan the results. Never changes a tile, the format tab, the Subject ID or a selection.

    Each step is guarded, so one failure does not stop the rest. While a pipeline, a setup job or an input import
    runs, only the caches (1), the Results rescan (5) and the log line (6) run. Return the log line.
    """
    steps = [("caches", lambda: _drop_input_caches(owner))]
    if not _inputs_busy(owner):
        steps += [
            ("input files", lambda: refresh_input_hints(owner)),
            ("input preview", lambda: owner._refresh_input_preview_controls()),
            ("run button", lambda: _refresh_run_state(owner)),
            ("algorithms", lambda: _refresh_algorithms(owner)),
        ]
    steps.append(("results", lambda: owner._refresh_run_history()))
    log = getattr(owner, "appendLog", None)
    for label, step in steps:
        try:
            step()
        except Exception as exc:
            if callable(log):
                log(f"[Refresh] Could not refresh the {label}: {exc}")
    zh = str(getattr(owner, "_onboardingLanguage", "en")).startswith("zh")
    message = onboarding_text("zh" if zh else "en", "inputs_refreshed")
    if callable(log):
        log(message)
    return message
