"""Multi-select picker buttons beside the product checkboxes."""

from shared.algorithm_selection import (
    BUNDLES,
    CONNECTOME,
    LABELS,
    MICROSTRUCTURE,
    PREPROCESSING,
    TRACKING,
    algorithms,
)

DEFAULT_CHOICES = {
    "preprocessing": "denoise,eddy,topup",
    "microstructure": "dti",
    "tractography": "csd_act",
    "bundles": "wmql",
    "connectome": "sift2",
}

ROW_INDICES = {
    "preprocessing": 0,
    "microstructure": 5,
    "tractography": 6,
    "bundles": 7,
    "connectome": 8,
}

FAMILIES = {
    "preprocessing": ("targetPreprocessingCheckBox", PREPROCESSING, "orchestrationPreprocessingStages"),
    "microstructure": ("targetMicrostructureCheckBox", MICROSTRUCTURE, "orchestrationMicrostructureStages"),
    "tractography": ("targetTractographyCheckBox", TRACKING, "orchestrationTractographyBackend"),
    "bundles": ("targetBundlesCheckBox", BUNDLES, "orchestrationBundleBackend"),
    "connectome": ("targetConnectomeCheckBox", CONNECTOME, "orchestrationConnectomeMetrics"),
}


ANATOMICAL_PRIORS = {
    "targetSurface": ("Surfaces", "皮层表面"),
    "targetSegmentation": ("Segmentation", "组织分割"),
    "targetParcellation": ("Parcellation", "大脑分区"),
}


def _set_anatomical_priors(owner, selected):
    import slicer

    node = owner._parameterNode
    if node is None:
        return
    with slicer.util.NodeModify(node):
        for field in ANATOMICAL_PRIORS:
            setattr(node, field, field in selected)
    owner._lastEnvironmentResult = owner._lastEnvironmentSignature = None
    owner._refresh_advanced_run_settings()
    owner._schedule_common_user_settings_save()


def _refresh_anatomical_priors(owner, zh):
    row = getattr(owner, "_anatomicalPriorsRow", None)
    if row is None:
        return
    import qt

    label, label_button, checkbox, summary, arrow = row
    title = "解剖先验" if zh else "Anatomical priors"
    label.setText(title)
    label_button.setMinimumWidth(label.sizeHint.width())
    checkbox.setAccessibleName(title)
    selected = [names[int(zh)] for field, names in ANATOMICAL_PRIORS.items()
                if getattr(owner._parameterNode, field, False)]
    checkbox.setCheckState({
        0: qt.Qt.Unchecked,
        len(ANATOMICAL_PRIORS): qt.Qt.Checked,
    }.get(len(selected), qt.Qt.PartiallyChecked))
    summary.setText(" · ".join(selected) if selected else ("未选择" if zh else "None selected"))
    tip = "选择解剖先验（可多选）" if zh else "Select anatomical priors (multiple allowed)"
    arrow.setAccessibleName(tip)
    arrow.setToolTip(tip)


def selected_algorithms(node, family):
    _, choices, attribute = FAMILIES[family]
    return algorithms(getattr(node, attribute, ""), choices, DEFAULT_CHOICES[family])


def set_algorithms(owner, family, values):
    _, choices, attribute = FAMILIES[family]
    selected = algorithms(",".join(values), choices, DEFAULT_CHOICES[family]) if values else ()
    if not selected:
        raise ValueError("Select at least one algorithm")
    import slicer

    selected = tuple(value for value in choices if value in selected)
    with slicer.util.NodeModify(owner._parameterNode):
        setattr(owner._parameterNode, attribute, ",".join(selected))
        if family == "microstructure":
            for stage, field in (
                ("freewater", "orchestrationRunFreeWater"),
                ("noddi", "orchestrationRunNoddi"),
                ("dki", "orchestrationRunDki"),
                ("ifw", "orchestrationRunIfw"),
            ):
                setattr(owner._parameterNode, field, stage in selected)
    owner._lastEnvironmentResult = owner._lastEnvironmentSignature = None
    owner._refresh_advanced_run_settings()
    owner._schedule_common_user_settings_save()


# Reuse the translated labels and allowed values of the established controls.
ALGORITHM_SETTINGS = {
    ("preprocessing", "brain_mask"): (
        ("brainMaskBackend", "brainMaskBackendComboBox", "brainMaskBackendLabel"),
        ("pnlMaskRoute", "pnlMaskRouteComboBox", "pnlMaskRouteLabel"),
    ),
    ("preprocessing", "topup"): (
        ("preprocessingPhaseEncodingDirection", "preprocessingPhaseEncodingDirectionComboBox", "preprocessingPhaseEncodingDirectionLabel"),
        ("preprocessingReadoutTime", "preprocessingReadoutTimeSpinBox", "preprocessingReadoutTimeLabel"),
        ("preprocessingReversePhaseB0", "preprocessingReversePhaseB0Path", "preprocessingReversePhaseB0Label"),
    ),
    ("microstructure", "dki"): (
        ("orchestrationDkiFitMethod", "orchestrationDkiFitMethodComboBox", "orchestrationDkiFitMethodLabel"),
    ),
    ("microstructure", "ifw"): (
        ("orchestrationIfwEchoTimeMs", "orchestrationIfwEchoTimeLineEdit", "orchestrationIfwEchoTimeLabel"),
    ),
    ("bundles", "wmql"): (
        ("orchestrationWmqlPreset", "wmqlQueryPresetComboBox", "wmqlQueryPresetLabel"),
        ("orchestrationWmqlSelection", "wmqlQuerySelectionLineEdit", "wmqlQuerySelectionLabel"),
    ),
    ("bundles", "wma"): (
        ("wmaAtlas", "wmaAtlasComboBox", "wmaAtlasLabel"),
    ),
}


def _add_algorithm_row(qt, owner, layout, box, family, algorithm, staged, zh, resize_popup):
    specs = ALGORITHM_SETTINGS.get((family, algorithm), ())
    fields = [(attribute, getattr(owner.ui, control, None), getattr(owner.ui, label, None))
              for attribute, control, label in specs]
    fields = [(attribute, control, label) for attribute, control, label in fields
              if control is not None and label is not None]
    ukf = (family, algorithm) == ("tractography", "ukf")
    if not fields and not ukf:
        layout.addWidget(box)
        return
    row = qt.QHBoxLayout()
    row.addWidget(box, 1)
    button = qt.QToolButton()
    button.setObjectName("algorithmSettings_" + algorithm)
    caption = "设置" if zh else "Settings"
    button.setAccessibleName(caption)
    button.setToolTip(caption)
    from shared.ui_styles import is_dark_mode

    ink = "#E6EDF3" if is_dark_mode(owner._uiWidget) else "#334155"
    artwork = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20">'
        f'<g fill="none" stroke="{ink}" stroke-width="1.7" stroke-linejoin="round">'
        '<path d="m10 2-.6 2.6-1.9.8-2.3-1.4-2.2 2.2 1.4 2.3-.8 1.9-2.6.6v3l2.6.6.8 1.9'
        '-1.4 2.3 2.2 2.2 2.3-1.4 1.9.8.6 2.6h3l.6-2.6 1.9-.8 2.3 1.4 2.2-2.2-1.4-2.3.8-1.9'
        ' 2.6-.6v-3l-2.6-.6-.8-1.9 1.4-2.3-2.2-2.2-2.3 1.4-1.9-.8-.6-2.6z"/>'
        '<circle cx="11.5" cy="12.5" r="3.1"/></g></svg>'
    )
    pixmap = qt.QPixmap()
    pixmap.loadFromData(qt.QByteArray(artwork.encode("ascii")), "SVG")
    button.setIcon(qt.QIcon(pixmap))
    button.setIconSize(qt.QSize(15, 15))
    button.setFixedSize(28, 24)
    button.setCheckable(True)
    row.addWidget(button)
    layout.addLayout(row)
    panel = qt.QWidget()
    form = qt.QFormLayout(panel)
    form.setRowWrapPolicy(qt.QFormLayout.WrapLongRows)
    for attribute, original, label in fields:
        if isinstance(original, qt.QComboBox):
            editor = qt.QComboBox()
            for index in range(original.count):
                editor.addItem(original.itemText(index), original.itemData(index))
            editor.setCurrentIndex(original.currentIndex)
            def choice(editor=editor):
                value = editor.itemData(editor.currentIndex)
                return str(editor.currentText if value is None else value)
            staged[attribute] = choice
        elif isinstance(original, qt.QDoubleSpinBox):
            editor = qt.QDoubleSpinBox()
            editor.setRange(original.minimum, original.maximum)
            editor.setDecimals(original.decimals)
            editor.setSingleStep(original.singleStep)
            editor.setValue(original.value)
            staged[attribute] = lambda editor=editor: float(editor.value)
        elif hasattr(original, "currentPath"):
            import ctk
            editor = ctk.ctkPathLineEdit()
            editor.filters = original.filters
            editor.nameFilters = original.nameFilters
            editor.setCurrentPath(original.currentPath)
            staged[attribute] = lambda editor=editor: str(editor.currentPath)
        else:
            editor = qt.QLineEdit()
            editor.setText(str(getattr(owner._parameterNode, attribute)))
            staged[attribute] = lambda editor=editor: str(editor.text).strip()
        editor.setToolTip(original.toolTip)
        form.addRow(str(label.text), editor)
    if ukf:
        from shared.ui_ukf import add_staged_ukf_fields
        staged["ukfParameters"] = add_staged_ukf_fields(form, owner._parameterNode, zh)
    layout.addWidget(panel)
    panel.hide()
    def toggle_settings(visible):
        panel.setVisible(visible)
        resize_popup()

    button.toggled.connect(toggle_settings)


def choose_algorithms(owner, family):
    import qt
    import slicer

    if getattr(owner, "_pipelineActive", False) or getattr(owner, "_selectedSetupActive", False):
        return False
    zh = getattr(owner, "_onboardingLanguage", "en").startswith("zh")
    is_anatomical = family == "anatomical_priors"
    if is_anatomical:
        choices = tuple(ANATOMICAL_PRIORS)
        title_text = "解剖先验" if zh else "Anatomical priors"
        current = tuple(field for field in choices if getattr(owner._parameterNode, field, False))
    else:
        _, choices, _ = FAMILIES[family]
        title_text = "选择算法（可多选）" if zh else "Select algorithms (multiple allowed)"
        current = selected_algorithms(owner._parameterNode, family)
    dialog = qt.QDialog(slicer.util.mainWindow())
    dialog.objectName = "algorithmPicker_" + family
    dialog.windowTitle = title_text
    popup_flag = getattr(qt.Qt, "Popup", 0)
    frameless_flag = getattr(qt.Qt, "FramelessWindowHint", 0)
    if popup_flag and hasattr(dialog, "setWindowFlags"):
        try:
            dialog_flags = popup_flag | frameless_flag
        except TypeError:
            dialog_flags = popup_flag
        dialog.setWindowFlags(dialog_flags)
    if hasattr(dialog, "setProperty"):
        dialog.setProperty("popupCard", "true")
    try:
        from shared.ui_styles import is_dark_mode, ui_stylesheet

        dark = is_dark_mode(getattr(owner, "_uiWidget", None))
        dialog.setStyleSheet(ui_stylesheet(dark=dark, widget=dialog))
    except Exception:
        pass

    layout = qt.QVBoxLayout(dialog)
    if hasattr(layout, "setContentsMargins"):
        layout.setContentsMargins(12, 10, 12, 12)
    if hasattr(layout, "setSpacing"):
        layout.setSpacing(6)

    if hasattr(qt, "QHBoxLayout") and hasattr(qt, "QToolButton") and hasattr(layout, "addLayout"):
        header_row = qt.QHBoxLayout()
        if hasattr(header_row, "setContentsMargins"):
            header_row.setContentsMargins(0, 0, 0, 2)
        title_label = qt.QLabel(title_text)
        if hasattr(title_label, "setObjectName"):
            title_label.setObjectName("algorithmPickerTitle")
        close_btn = qt.QToolButton()
        if hasattr(close_btn, "setObjectName"):
            close_btn.setObjectName("algorithmPickerCloseButton")
        if hasattr(close_btn, "setText"):
            close_btn.setText("×")
        else:
            close_btn.text = "×"
        if hasattr(close_btn, "setFixedSize"):
            close_btn.setFixedSize(20, 20)
        if hasattr(close_btn, "setFocusPolicy") and hasattr(qt.Qt, "NoFocus"):
            close_btn.setFocusPolicy(qt.Qt.NoFocus)
        if hasattr(close_btn, "setToolTip"):
            close_btn.setToolTip("关闭" if zh else "Close")
        if hasattr(close_btn, "clicked") and hasattr(close_btn.clicked, "connect"):
            close_btn.clicked.connect(dialog.reject)
        elif hasattr(close_btn, "connect"):
            close_btn.connect("clicked()", dialog.reject)
        header_row.addWidget(title_label)
        if hasattr(header_row, "addStretch"):
            header_row.addStretch(1)
        header_row.addWidget(close_btn)
        layout.addLayout(header_row)

    anchor = (owner._anatomicalPriorsRow[-1] if is_anatomical else
              (getattr(owner, "_algorithmButtons", None) or {}).get(family))
    scroll = qt.QScrollArea(dialog)
    scroll.setObjectName("algorithmPickerScrollArea")
    scroll.setFrameShape(qt.QFrame.NoFrame)
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOff)
    body = qt.QWidget()
    body_layout = qt.QVBoxLayout(body)
    body_layout.setContentsMargins(0, 0, 4, 0)
    body_layout.setSpacing(6)
    scroll.setWidget(body)
    # QScrollArea uses QPalette.Base for its viewport, unlike the popup.
    # Let the popup paint the background behind all of the choices.
    for surface in (scroll, scroll.viewport(), body):
        surface.setAutoFillBackground(False)
        surface.setStyleSheet("background: transparent;")
    layout.addWidget(scroll)

    from shared.ui_compact import _position_input_help_card, _qt_int, _qt_val

    def resize_popup():
        anchor_visible = anchor is not None and anchor.isVisible()
        screen = qt.QApplication.screenAt(anchor.mapToGlobal(qt.QPoint(0, 0))) if anchor_visible else None
        if screen is None:
            screen = qt.QApplication.primaryScreen()
        available = _qt_val(screen, "availableGeometry")
        available_height = _qt_int(available, "height", 800)
        body_layout.invalidate()
        content_size = body_layout.sizeHint()
        content_height = _qt_int(content_size, "height", 0)
        max_body_height = min(480, max(80, available_height - 108))
        scroll.setFixedHeight(min(content_height, max_body_height))
        scroll.setMinimumWidth(min(_qt_int(content_size, "width", 0) + 20,
                                   max(100, _qt_int(available, "width", 800) - 40)))
        layout.activate()
        dialog.adjustSize()
        if anchor_visible:
            _position_input_help_card(anchor, dialog, qt)

    boxes = {}
    staged = {}
    pointing_cursor = getattr(qt.Qt, "PointingHandCursor", None)
    for value in choices:
        label = ANATOMICAL_PRIORS[value][int(zh)] if is_anatomical else LABELS[value]
        box = qt.QCheckBox(label)
        box.objectName = "algorithm_" + value
        box.checked = value in current
        if pointing_cursor is not None and hasattr(box, "setCursor"):
            box.setCursor(pointing_cursor)
        boxes[value] = box
        _add_algorithm_row(qt, owner, body_layout, box, family, value, staged, zh, resize_popup)

    if is_anatomical:
        note_text = ""
    elif family == "tractography":
        note_text = (
            "每种方法分别输出流线；束分析和连接组按方法分别计算。"
            if zh
            else "Each method writes its own tractogram. Fiber bundles and connectomes are computed for each method separately."
        )
    elif family == "preprocessing":
        note_text = (
            "Brain mask 默认不勾选；未勾选且无现成掩膜时仍会自动计算。"
            if zh
            else "Brain mask is off by default; if unchecked and no mask is provided, it runs automatically."
        )
    elif family == "microstructure":
        note_text = (
            "若未勾选 DTI 但所选模型需要其预处理结果，运行时会自动计算 DTI。"
            if zh
            else "If DTI is unchecked, it still runs automatically whenever required by selected models."
        )
    elif family == "connectome":
        note_text = (
            "默认 SIFT2 加权连接组；也可切换或同时勾选 Fiber count。"
            if zh
            else "Defaults to SIFT2-weighted connectome; Fiber count can also be selected."
        )
    else:
        note_text = (
            "可展开各算法的设置。"
            if zh
            else "Open an algorithm’s settings to adjust its parameters."
        )
    note = qt.QLabel(note_text)
    if hasattr(note, "setObjectName"):
        note.setObjectName("algorithmPickerNote")
    note.wordWrap = True
    note.setMaximumWidth(360)
    if note_text:
        body_layout.addWidget(note)
    layout.addSpacing(4)
    buttons = qt.QDialogButtonBox(dialog)
    buttons.setStandardButtons(qt.QDialogButtonBox.Apply | qt.QDialogButtonBox.Cancel)
    apply = buttons.button(qt.QDialogButtonBox.Apply)
    apply.text = "应用" if zh else "Apply"
    cancel_btn = buttons.button(qt.QDialogButtonBox.Cancel) if hasattr(buttons, "button") else None
    if cancel_btn is not None and zh:
        cancel_btn.text = "取消"

    def update(*_):
        apply.enabled = is_anatomical or any(box.checked for box in boxes.values())

    for box in boxes.values():
        box.toggled.connect(update)
    update()
    def validate_and_accept():
        try:
            if "orchestrationIfwEchoTimeMs" in staged:
                echo_time = staged["orchestrationIfwEchoTimeMs"]()
                if echo_time:
                    from math import isfinite
                    number = float(echo_time)
                    if not isfinite(number) or number <= 0:
                        raise ValueError("Echo time must be a positive number in milliseconds.")
            if "ukfParameters" in staged:
                staged["ukfParameters"]()
            if "orchestrationWmqlSelection" in staged:
                from tractography.wmql_protocol import canonical_wmql_selection
                canonical_wmql_selection(staged["orchestrationWmqlSelection"](), staged["orchestrationWmqlPreset"]())
        except ValueError as exc:
            note.setText(str(exc))
            resize_popup()
            scroll.ensureWidgetVisible(note)
            return
        dialog.accept()

    apply.connect("clicked()", validate_and_accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)

    resize_popup()

    popup_watch = None
    if hasattr(qt, "QTimer") and popup_flag:
        try:
            seen_visible = {"flag": False}
            popup_watch = qt.QTimer()
            popup_watch.setInterval(50)

            def _watch_popup() -> None:
                vis = dialog.isVisible() if callable(getattr(dialog, "isVisible", None)) else bool(getattr(dialog, "visible", True))
                if vis:
                    seen_visible["flag"] = True
                elif seen_visible["flag"]:
                    popup_watch.stop()
                    dialog.reject()

            if hasattr(popup_watch, "timeout") and hasattr(popup_watch.timeout, "connect"):
                popup_watch.timeout.connect(_watch_popup)
            elif hasattr(popup_watch, "connect"):
                popup_watch.connect("timeout()", _watch_popup)
            popup_watch.start()
        except Exception:
            popup_watch = None

    try:
        accepted = dialog.exec_() == qt.QDialog.Accepted
    finally:
        if popup_watch is not None:
            try:
                popup_watch.stop()
            except Exception:
                pass
    if accepted:
        selection = [value for value, box in boxes.items() if box.checked]
        if is_anatomical:
            _set_anatomical_priors(owner, selection)
        else:
            with slicer.util.NodeModify(owner._parameterNode):
                for attribute, value in staged.items():
                    setattr(owner._parameterNode, attribute, value())
                set_algorithms(owner, family, selection)
    dialog.deleteLater()
    return accepted


def install_algorithm_pickers(owner):
    import qt

    owner._algorithmButtons = {}
    owner._algorithmSummaryLabels = {}
    owner._productLabelButtons = []
    form = owner.ui.targetProductsFormLayout
    form.setVerticalSpacing(4)
    # Match a plain QCheckBox row without clipping the 16px styled indicator box on any DPI scale.
    style = qt.QApplication.style()
    indicator_width = max(18, style.pixelMetric(qt.QStyle.PM_IndicatorWidth))
    label_gap = max(5, style.pixelMetric(qt.QStyle.PM_CheckBoxLabelSpacing))
    pointing_cursor = getattr(qt.Qt, "PointingHandCursor", None)

    def picker_button(name, family):
        button = qt.QToolButton()
        button.setObjectName(name)
        button.setProperty("algorithmPickerButton", True)
        button.setText("...")
        button.setToolButtonStyle(qt.Qt.ToolButtonIconOnly)
        button.setIconSize(qt.QSize(16, 16))
        button.setFixedSize(28, 22)
        button.setCursor(qt.Qt.PointingHandCursor)
        button.clicked.connect(lambda: choose_algorithms(owner, family))
        return button

    def click_area(content, checkbox, *, expand=False, inset=0):
        button = qt.QPushButton()
        button.setFlat(True)
        button.setFocusPolicy(qt.Qt.NoFocus)
        button.setStyleSheet("QPushButton { border: none; background: transparent; }")
        button.setSizePolicy(qt.QSizePolicy.Ignored if expand else qt.QSizePolicy.Preferred,
                             qt.QSizePolicy.Preferred)
        if pointing_cursor is not None:
            button.setCursor(pointing_cursor)
        button.clicked.connect(lambda _checked=False: checkbox.click())
        layout = qt.QVBoxLayout(button)
        layout.setContentsMargins(inset, 0, 0, 0)
        content.setAttribute(qt.Qt.WA_TransparentForMouseEvents)
        layout.addWidget(content)
        return button

    def make_label_clickable(row, checkbox):
        label = form.labelForField(checkbox)
        if label is not None:
            form.takeAt(form.indexOf(label))
            button = click_area(label, checkbox)
            owner._productLabelButtons.append((label, button))
            form.setWidget(row, qt.QFormLayout.LabelRole, button)

    old_anatomical_rows = ("targetSurfaceCheckBox", "targetSegmentationCheckBox", "targetParcellationCheckBox")
    if all(getattr(owner.ui, name, None) is not None for name in old_anatomical_rows):
        for name in old_anatomical_rows:
            control = getattr(owner.ui, name)
            for widget in (form.labelForField(control), control):
                if widget is not None:
                    form.takeAt(form.indexOf(widget))
                    widget.hide()

        checkbox = qt.QCheckBox()
        checkbox.setObjectName("anatomicalPriorsCheckBox")
        checkbox.setTristate(True)
        checkbox.setFixedWidth(indicator_width)
        checkbox.setMinimumHeight(18)
        checkbox.clicked.connect(lambda checked: _set_anatomical_priors(
            owner, ANATOMICAL_PRIORS if checked else (),
        ))
        label = qt.QLabel()
        label_button = click_area(label, checkbox)
        form.setWidget(1, qt.QFormLayout.LabelRole, label_button)
        container = qt.QWidget()
        row_layout = qt.QHBoxLayout(container)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(0)
        row_layout.addWidget(checkbox, 0, qt.Qt.AlignVCenter)
        summary = qt.QLabel()
        summary.setWordWrap(True)
        summary.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)
        row_layout.addWidget(click_area(summary, checkbox, expand=True, inset=label_gap), 1)
        arrow = picker_button("anatomicalPriorsSelectButton", "anatomical_priors")
        row_layout.addWidget(arrow, 0, qt.Qt.AlignVCenter)
        form.setWidget(1, qt.QFormLayout.FieldRole, container)
        owner._anatomicalPriorsRow = (label, label_button, checkbox, summary, arrow)

    checkbox = getattr(owner.ui, "targetSurfacePostprocessCheckBox", None)
    if checkbox is not None:
        make_label_clickable(2, checkbox)
        form.takeAt(form.indexOf(checkbox))
        checkbox.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
        checkbox.setMinimumWidth(0)
        form.setWidget(2, qt.QFormLayout.FieldRole, click_area(checkbox, checkbox, expand=True))

    for family, (name, _, _) in FAMILIES.items():
        row = ROW_INDICES[family]
        checkbox = getattr(owner.ui, name)
        role = qt.QFormLayout.FieldRole
        make_label_clickable(row, checkbox)
        form.takeAt(form.indexOf(checkbox))
        container = qt.QWidget()
        layout = qt.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        checkbox.setSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Fixed)
        # Keep the checkbox's full accessible text, but render the summary in a
        # wrapping label so multi-selection never forces a wider Slicer dock.
        checkbox.setFixedWidth(indicator_width)
        if hasattr(checkbox, "setMinimumHeight"):
            checkbox.setMinimumHeight(18)
        if pointing_cursor is not None and hasattr(checkbox, "setCursor"):
            checkbox.setCursor(pointing_cursor)
        layout.addWidget(checkbox, 0, qt.Qt.AlignVCenter)
        summary = qt.QLabel()
        summary.wordWrap = True
        summary.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)
        summary.setMinimumWidth(0)
        summary_button = click_area(summary, checkbox, expand=True, inset=label_gap)
        summary_button.objectName = family + "AlgorithmRow"
        layout.addWidget(summary_button, 1)
        owner._algorithmSummaryLabels[name] = summary
        button = picker_button(family + "AlgorithmSelectButton", family)
        layout.addWidget(button, 0, qt.Qt.AlignVCenter)
        form.setWidget(row, role, container)
        owner._algorithmButtons[family] = button

    refresh_picker_language(owner)
    refresh_picker_icons(owner)


def refresh_picker_icons(owner, *, dark=None):
    import qt
    from shared.ui_styles import is_dark_mode

    dark = is_dark_mode(getattr(owner, "_uiWidget", None)) if dark is None else dark
    ink = "#8CD2F5" if dark else "#326A8A"
    pixmap = qt.QPixmap(32, 32)
    pixmap.fill(qt.Qt.transparent)
    painter = qt.QPainter(pixmap)
    painter.setRenderHint(qt.QPainter.Antialiasing)
    painter.setPen(qt.Qt.NoPen)
    painter.setBrush(qt.QColor(ink))
    for x in (8, 16, 24):
        painter.drawEllipse(qt.QPointF(x, 16), 2.2, 2.2)
    painter.end()
    pixmap.setDevicePixelRatio(2)
    icon = qt.QIcon(pixmap)
    for button in getattr(owner, "_algorithmButtons", {}).values():
        button.setIcon(icon)
    row = getattr(owner, "_anatomicalPriorsRow", None)
    if row is not None:
        row[-1].setIcon(icon)


def refresh_picker_language(owner):
    for label, button in getattr(owner, "_productLabelButtons", ()):
        button.setMinimumWidth(label.sizeHint.width())
    zh = getattr(owner, "_onboardingLanguage", "en").startswith("zh")
    _refresh_anatomical_priors(owner, zh)
    for family, button in getattr(owner, "_algorithmButtons", {}).items():
        selected = selected_algorithms(getattr(owner, "_parameterNode", None), family)
        name = {
            "preprocessing": ("预处理", "preprocessing"),
            "microstructure": ("微结构分析", "microstructure"),
            "tractography": ("纤维追踪", "tractography"),
            "bundles": ("纤维束分析", "bundle analysis"),
            "connectome": ("连接组", "connectome"),
        }[family][0 if zh else 1]
        action = "选择" + name + "算法" if zh else "Select " + name + " algorithms"
        tip = action + ("（可多选）。当前：" if zh else " (multiple allowed). Current: ") + ", ".join(LABELS[value] for value in selected)
        if getattr(button, "text", None) != "...":
            button.text = "..."
        if getattr(button, "accessibleName", None) != action:
            button.setAccessibleName(action)
        if getattr(button, "toolTip", None) != tip:
            button.toolTip = tip
