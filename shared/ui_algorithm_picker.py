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


def _add_algorithm_row(qt, owner, layout, box, family, algorithm, staged, zh):
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
    button.setText("设置" if zh else "Settings")
    button.setCheckable(True)
    row.addWidget(button)
    layout.addLayout(row)
    panel = qt.QWidget()
    form = qt.QFormLayout(panel)
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
    button.toggled.connect(panel.setVisible)


def choose_algorithms(owner, family):
    import qt
    import slicer

    if getattr(owner, "_pipelineActive", False) or getattr(owner, "_selectedSetupActive", False):
        return False
    zh = getattr(owner, "_onboardingLanguage", "en").startswith("zh")
    _, choices, _ = FAMILIES[family]
    title_text = "选择算法（可多选）" if zh else "Select algorithms (multiple allowed)"
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

    boxes = {}
    staged = {}
    current = selected_algorithms(owner._parameterNode, family)
    pointing_cursor = getattr(qt.Qt, "PointingHandCursor", None)
    for value in choices:
        box = qt.QCheckBox(LABELS[value])
        box.objectName = "algorithm_" + value
        box.checked = value in current
        if pointing_cursor is not None and hasattr(box, "setCursor"):
            box.setCursor(pointing_cursor)
        boxes[value] = box
        _add_algorithm_row(qt, owner, layout, box, family, value, staged, zh)

    if family == "tractography":
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
    layout.addWidget(note)
    layout.addSpacing(8)
    buttons = qt.QDialogButtonBox(dialog)
    buttons.setStandardButtons(qt.QDialogButtonBox.Apply | qt.QDialogButtonBox.Cancel)
    apply = buttons.button(qt.QDialogButtonBox.Apply)
    apply.text = "应用" if zh else "Apply"
    cancel_btn = buttons.button(qt.QDialogButtonBox.Cancel) if hasattr(buttons, "button") else None
    if cancel_btn is not None and zh:
        cancel_btn.text = "取消"

    def update(*_):
        apply.enabled = any(box.checked for box in boxes.values())

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
            return
        dialog.accept()

    apply.connect("clicked()", validate_and_accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)

    anchor = (getattr(owner, "_algorithmButtons", None) or {}).get(family)
    if anchor is not None and hasattr(anchor, "mapToGlobal"):
        try:
            vis = anchor.isVisible() if callable(getattr(anchor, "isVisible", None)) else bool(getattr(anchor, "visible", False))
            if vis:
                from shared.ui_compact import _position_input_help_card

                if hasattr(dialog, "adjustSize"):
                    dialog.adjustSize()
                _position_input_help_card(anchor, dialog, qt)
        except Exception:
            pass

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
        with slicer.util.NodeModify(owner._parameterNode):
            for attribute, value in staged.items():
                setattr(owner._parameterNode, attribute, value())
            set_algorithms(owner, family, [value for value, box in boxes.items() if box.checked])
    dialog.deleteLater()
    return accepted


def install_algorithm_pickers(owner):
    import qt

    owner._algorithmButtons = {}
    owner._algorithmSummaryLabels = {}
    form = owner.ui.targetProductsFormLayout
    form.setVerticalSpacing(4)
    # Match a plain QCheckBox row without clipping the 16px styled indicator box on any DPI scale.
    style = qt.QApplication.style()
    indicator_width = max(18, style.pixelMetric(qt.QStyle.PM_IndicatorWidth))
    label_gap = max(5, style.pixelMetric(qt.QStyle.PM_CheckBoxLabelSpacing))
    pointing_cursor = getattr(qt.Qt, "PointingHandCursor", None)

    for plain_name in (
        "targetSurfaceCheckBox",
        "targetSurfacePostprocessCheckBox",
        "targetParcellationCheckBox",
        "targetSegmentationCheckBox",
    ):
        plain_cb = getattr(owner.ui, plain_name, None)
        if plain_cb is not None:
            if hasattr(plain_cb, "setSizePolicy"):
                plain_cb.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
            if hasattr(plain_cb, "setMinimumWidth"):
                plain_cb.setMinimumWidth(0)
            if pointing_cursor is not None and hasattr(plain_cb, "setCursor"):
                plain_cb.setCursor(pointing_cursor)

    for family, (name, _, _) in FAMILIES.items():
        row = ROW_INDICES[family]
        checkbox = getattr(owner.ui, name)
        role = qt.QFormLayout.FieldRole
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
        layout.addSpacing(label_gap)
        summary = qt.QLabel()
        summary.wordWrap = True
        summary.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)
        summary.setMinimumWidth(0)
        layout.addWidget(summary, 1)
        owner._algorithmSummaryLabels[name] = summary
        button = qt.QToolButton()
        button.objectName = family + "AlgorithmSelectButton"
        button.setFixedSize(28, 22)
        if pointing_cursor is not None and hasattr(button, "setCursor"):
            button.setCursor(pointing_cursor)
        button.clicked.connect(lambda _checked=False, family=family: choose_algorithms(owner, family))
        layout.addSpacing(6)
        layout.addWidget(button, 0, qt.Qt.AlignVCenter)
        form.setWidget(row, role, container)
        owner._algorithmButtons[family] = button

    refresh_picker_language(owner)


def refresh_picker_language(owner):
    zh = getattr(owner, "_onboardingLanguage", "en").startswith("zh")
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
