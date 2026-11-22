"""Multi-select picker buttons beside the product checkboxes."""

import sys

from shared.algorithm_selection import (
    BUNDLES,
    CONNECTOME,
    FSL_FREE_PREPROCESSING_DEFAULT,
    FSL_STAGES,
    LABELS,
    MICROSTRUCTURE,
    PREPROCESSING,
    TRACKING,
    algorithms,
    without_fsl_stages,
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

FAMILY_TARGET_ATTRS = {
    "preprocessing": "targetPreprocessing",
    "microstructure": "targetMicrostructure",
    "tractography": "targetTractography",
    "bundles": "targetBundles",
    "connectome": "targetConnectome",
}


ANATOMICAL_PRIORS = {
    "targetSurface": ("Surfaces", "皮层表面"),
    "targetSegmentation": ("Segmentation", "组织分割"),
    "targetParcellation": ("Parcellation", "大脑分区"),
}


def _remembered_anatomical_priors(owner) -> tuple[str, ...]:
    raw = getattr(owner, "_lastAnatomicalPriorsSelection", None) or ()
    return tuple(field for field in ANATOMICAL_PRIORS if field in raw)


def _set_anatomical_priors(owner, selected, *, clear_memory: bool = False):
    import slicer

    node = owner._parameterNode
    if node is None:
        return
    ordered = tuple(field for field in ANATOMICAL_PRIORS if field in selected)
    if ordered:
        owner._lastAnatomicalPriorsSelection = ordered
    elif clear_memory:
        owner._lastAnatomicalPriorsSelection = ()
    with slicer.util.NodeModify(node):
        for field in ANATOMICAL_PRIORS:
            setattr(node, field, field in ordered)
    owner._lastEnvironmentResult = owner._lastEnvironmentSignature = None
    refresh_adv = getattr(owner, "_refresh_advanced_run_settings", None)
    if callable(refresh_adv):
        refresh_adv()
    lang = getattr(owner, "_onboardingLanguage", "en")
    zh = isinstance(lang, str) and lang.startswith("zh")
    _refresh_anatomical_priors(owner, zh)
    schedule_save = getattr(owner, "_schedule_common_user_settings_save", None)
    if callable(schedule_save):
        schedule_save()


def _refresh_anatomical_priors(owner, zh):
    row = getattr(owner, "_anatomicalPriorsRow", None)
    if not isinstance(row, tuple) or len(row) != 5:
        return
    import qt

    label, label_button, checkbox, summary, arrow = row
    title = "解剖先验" if zh else "Anatomical priors"
    label.setText(title)
    label_button.setMinimumWidth(label.sizeHint.width())
    checkbox.setAccessibleName(title)
    active = tuple(
        field for field in ANATOMICAL_PRIORS
        if getattr(owner._parameterNode, field, False)
    )
    if active:
        owner._lastAnatomicalPriorsSelection = active
    display_fields = active or _remembered_anatomical_priors(owner)
    display_labels = [ANATOMICAL_PRIORS[field][int(zh)] for field in display_fields]
    checked = bool(active)
    block = getattr(checkbox, "blockSignals", None)
    was_blocked = block(True) if callable(block) else False
    try:
        if hasattr(checkbox, "setChecked"):
            checkbox.setChecked(checked)
        elif hasattr(checkbox, "setCheckState"):
            checkbox.setCheckState(qt.Qt.Checked if checked else qt.Qt.Unchecked)
    finally:
        if callable(block):
            block(was_blocked if isinstance(was_blocked, bool) else False)
    summary.setText(" · ".join(display_labels) if display_labels else ("未选择" if zh else "None selected"))
    tip = "选择解剖先验（可多选）" if zh else "Select anatomical priors (multiple allowed)"
    arrow.setAccessibleName(tip)
    arrow.setToolTip(tip)


def _cleared_algorithm_families(owner) -> set[str]:
    cleared = getattr(owner, "_clearedAlgorithmFamilies", None)
    if not isinstance(cleared, set):
        cleared = set()
        owner._clearedAlgorithmFamilies = cleared
    return cleared


def _fsl_stages_offered() -> bool:
    """G3: False where Eddy and Topup cannot run (no FSL). A failed lookup counts as offered (the old behaviour)."""
    try:
        from shared import selected_setup

        return bool(selected_setup.fsl_preprocessing_offered())
    except Exception:
        return True


def _available_selection(family, selected):
    """G3: drop the FSL stages from a Preprocessing selection where FSL is not offered."""
    if family != "preprocessing" or _fsl_stages_offered():
        return tuple(selected)
    return tuple(value for value in selected if value not in FSL_STAGES)


def selected_algorithms(node, family):
    _, choices, attribute = FAMILIES[family]
    value, default = getattr(node, attribute, ""), DEFAULT_CHOICES[family]
    if family == "preprocessing" and not _fsl_stages_offered():
        value, default = without_fsl_stages(value or default), FSL_FREE_PREPROCESSING_DEFAULT
    return algorithms(value, choices, default)


def _apply_family_selection_to_node(node, family, values):
    _, choices, attribute = FAMILIES[family]
    selected = algorithms(",".join(values), choices, DEFAULT_CHOICES[family]) if values else ()
    selected = _available_selection(family, selected)
    if not selected:
        raise ValueError("Select at least one algorithm")
    selected = tuple(value for value in choices if value in selected)
    setattr(node, attribute, ",".join(selected))
    if family == "microstructure":
        for stage, field in (
            ("freewater", "orchestrationRunFreeWater"),
            ("noddi", "orchestrationRunNoddi"),
            ("dki", "orchestrationRunDki"),
            ("ifw", "orchestrationRunIfw"),
        ):
            setattr(node, field, stage in selected)
    return selected


def set_algorithms(owner, family, values):
    _, choices, _ = FAMILIES[family]
    selected = algorithms(",".join(values), choices, DEFAULT_CHOICES[family]) if values else ()
    if not _available_selection(family, selected):
        raise ValueError("Select at least one algorithm")
    import slicer

    _cleared_algorithm_families(owner).discard(family)
    with slicer.util.NodeModify(owner._parameterNode):
        _apply_family_selection_to_node(owner._parameterNode, family, values)
    owner._lastEnvironmentResult = owner._lastEnvironmentSignature = None
    from shared.ui_compact import refresh_compact_algorithms

    refresh_compact_algorithms(owner)
    refresh_adv = getattr(owner, "_refresh_advanced_run_settings", None)
    if callable(refresh_adv):
        refresh_adv()
    schedule_save = getattr(owner, "_schedule_common_user_settings_save", None)
    if callable(schedule_save):
        schedule_save()


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


def _unavailable_marker(qt, owner, algorithm, reason, dark=None):
    """B2: a small "?" at the right end of a row whose algorithm this data or system cannot run.

    b2 ui2: a real button with the look of the Prepare Inputs "?" (``help_marker_style``). The keyboard reaches it,
    a hover shows the reason as a tooltip, and a click shows the same reason under the row (``_unavailable_reason``).
    It is a sibling of the row's check box, so it works while that box is disabled.
    """
    from shared.ui_help_button import BUTTON_SIZE, help_marker_style
    from shared.ui_styles import is_dark_mode

    use_dark = is_dark_mode(getattr(owner, "_uiWidget", None)) if dark is None else bool(dark)
    name = "algorithmUnavailable_" + algorithm
    marker = qt.QToolButton()
    marker.setObjectName(name)
    marker.setText("?")
    if hasattr(marker, "setAutoRaise"):
        marker.setAutoRaise(True)
    marker.setToolTip(reason)
    marker.setAccessibleName(reason)
    marker.setFixedSize(BUTTON_SIZE, BUTTON_SIZE)
    pointing = getattr(qt.Qt, "PointingHandCursor", None)
    if pointing is not None and hasattr(marker, "setCursor"):
        marker.setCursor(pointing)
    strong = getattr(qt.Qt, "StrongFocus", None)
    if strong is not None:
        marker.setFocusPolicy(strong)
    marker.setStyleSheet(help_marker_style(use_dark, name=name))
    return marker


def _unavailable_reason(qt, owner, algorithm, reason, marker, resize_popup, dark=None):
    """b2 ui2: the reason of a blocked row as a word-wrapped line under the row, hidden until the "?" is clicked.

    A click on the "?" shows it, a second click hides it, and the popup resizes each time. It is not a second
    ``Qt.Popup``: a nested popup would close the picker's own popup. The reason is not a staged value, so reading it
    does not count as a change and a look-only outside click still commits nothing.
    """
    from shared.ui_styles import is_dark_mode

    use_dark = is_dark_mode(getattr(owner, "_uiWidget", None)) if dark is None else bool(dark)
    ink = "#9AA4AF" if use_dark else "#55636F"
    name = "algorithmUnavailableReason_" + algorithm
    label = qt.QLabel(reason)
    label.setObjectName(name)
    label.wordWrap = True
    if hasattr(label, "setTextFormat") and hasattr(qt.Qt, "PlainText"):
        label.setTextFormat(qt.Qt.PlainText)
    if hasattr(label, "setMaximumWidth"):
        label.setMaximumWidth(360)
    label.setStyleSheet(f"QLabel#{name} {{ color: {ink}; font-size: 11px; padding: 0px 0px 2px 22px; }}")
    label.setVisible(False)
    shown = {"value": False}

    def toggle(*_args) -> None:
        shown["value"] = not shown["value"]
        label.setVisible(shown["value"])
        resize_popup()

    marker.clicked.connect(toggle)
    return label


def _data_block_texts(owner, choices, zh) -> dict:
    """Algorithm id -> reason text for the algorithms of ``choices`` that the current data cannot run."""
    try:
        from shared import algorithm_availability

        blocks = algorithm_availability.current_data_blocks(owner)
        return {value: blocks[value].text("zh" if zh else "en") for value in choices if value in blocks}
    except Exception:
        return {}


def _add_algorithm_row(qt, owner, layout, box, family, algorithm, staged, zh, resize_popup, dark=None,
                       disabled=False, disabled_tooltip=""):
    specs = ALGORITHM_SETTINGS.get((family, algorithm), ())
    marker = _unavailable_marker(qt, owner, algorithm, disabled_tooltip, dark) if disabled else None
    ui_root = getattr(owner, "ui", None)
    fields = [(attribute, getattr(ui_root, control, None), getattr(ui_root, label, None))
              for attribute, control, label in specs] if ui_root is not None else []
    fields = [(attribute, control, label) for attribute, control, label in fields
              if control is not None and label is not None]
    ukf = (family, algorithm) == ("tractography", "ukf")
    # Batch 2 item 5: every bundle algorithm gets a "Tractography source" combo built in code.
    bundle_source = family == "bundles" and algorithm in ("wmql", "wma", "tractcloud")
    if not fields and not ukf and not bundle_source:
        if marker is None:
            layout.addWidget(box)
            return
        row = qt.QHBoxLayout()
        row.addWidget(box, 1)
        row.addWidget(marker)
        layout.addLayout(row)
        layout.addWidget(_unavailable_reason(qt, owner, algorithm, disabled_tooltip, marker, resize_popup, dark))
        return
    row = qt.QHBoxLayout()
    row.addWidget(box, 1)
    button = qt.QToolButton()
    button.setObjectName("algorithmSettings_" + algorithm)
    caption = "设置" if zh else "Settings"
    button.setAccessibleName(caption)
    if disabled:
        button.setEnabled(False)
        button.setToolTip(disabled_tooltip or caption)
    else:
        button.setToolTip(caption)
    from shared.ui_styles import is_dark_mode, style_combobox_popup

    use_dark = is_dark_mode(getattr(owner, "_uiWidget", None)) if dark is None else bool(dark)
    ink = ("#7D8590" if use_dark else "#94A3B8") if disabled else ("#E6EDF3" if use_dark else "#334155")
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
    if marker is not None:
        row.addWidget(marker)
    layout.addLayout(row)
    if marker is not None:
        layout.addWidget(_unavailable_reason(qt, owner, algorithm, disabled_tooltip, marker, resize_popup, dark))
    panel = qt.QWidget()
    form = qt.QFormLayout(panel)
    form.setRowWrapPolicy(qt.QFormLayout.WrapLongRows)
    for attribute, original, label in fields:
        if isinstance(original, qt.QComboBox):
            editor = qt.QComboBox()
            for index in range(original.count):
                editor.addItem(original.itemText(index), original.itemData(index))
            editor.setCurrentIndex(original.currentIndex)
            style_combobox_popup(editor, dark=use_dark)
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
            path_combo = getattr(editor, "comboBox", None)
            if callable(path_combo):
                try:
                    path_combo = path_combo()
                except Exception:
                    path_combo = None
            if path_combo is not None:
                style_combobox_popup(path_combo, dark=use_dark)
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
    if bundle_source:
        from shared.ui_bundle_source import add_staged_source_field
        source_attribute, source_value = add_staged_source_field(
            qt, form, owner._parameterNode, algorithm, zh, selected_algorithms(owner._parameterNode, "tractography"),
            dark=use_dark)
        staged[source_attribute] = source_value
    layout.addWidget(panel)
    panel.hide()
    def toggle_settings(visible):
        panel.setVisible(visible)
        resize_popup()

    button.toggled.connect(toggle_settings)


def _install_outside_close_apply(qt, dialog, popup_state, apply_staged, changed=lambda: True, no_replay=False):
    """Make a click outside the popup act like the Apply button (batch 2, popup-apply).

    Qt closes a ``Qt.Popup`` on an outside click with ``close()``, and ``QDialog::closeEvent`` turns that into
    ``reject()``. This filter sees the Close event first. When ``changed()`` says that no tick box and no staged value
    differs from the popup's state at open (fix S8), the event passes through to that reject: looking at a popup
    commits nothing. Otherwise it calls ``apply_staged``, the function behind the Apply button: it validates, and on
    success it accepts the dialog (the one commit runs after ``exec_`` returns). That Close event is ignored, so
    ``closeEvent`` cannot reject after the accept, and a refused apply leaves the popup open with the error shown, as
    the Apply button does. Cancel, the X and Esc call ``reject()``, which sends no Close event: they still discard.
    After the popup has finished, a late Close event passes through and commits nothing.

    ``no_replay`` (macOS, popup-mac): after a refused apply the popup stays open, but Qt's Cocoa layer has already
    taken it off its popup stack, so the next outside click goes to Qt's widget popup code. That closes the popup and
    then replays the press to the control under the mouse (measured: the Run button got "pressed"). Every Close event
    therefore sets ``Qt.WA_NoMouseReplay`` first; it must be set here, because ``QApplication::notify`` clears it before
    each mouse event that reaches the dialog.
    Returns the filter; the caller must keep it alive. Returns None if the filter cannot be built.
    """

    close_type = getattr(getattr(qt, "QEvent", None), "Close", None)
    if close_type is None or not hasattr(dialog, "installEventFilter"):
        return None
    replay_attribute = getattr(getattr(qt, "Qt", None), "WA_NoMouseReplay", None) if no_replay else None
    try:
        base = getattr(qt, "QObject")

        class OutsideCloseApplyFilter(base):
            def eventFilter(self, _watched, event):
                try:
                    if event.type() != close_type:
                        return False
                    if replay_attribute is not None:
                        dialog.setAttribute(replay_attribute, True)
                    if popup_state["finished"] or not changed():
                        return False
                    apply_staged()
                    event.ignore()
                    return True
                except Exception:
                    return False

        flt = OutsideCloseApplyFilter()
        dialog.installEventFilter(flt)
        return flt
    except Exception:
        return None


def _host_platform() -> str:
    return sys.platform


def _popup_waits_non_modal(qt, is_popup) -> bool:
    """macOS only: the picker popup is shown non-modal and a local event loop waits for it (batch 2, popup-mac)."""

    return bool(is_popup) and _host_platform() == "darwin" and getattr(qt, "QEventLoop", None) is not None


def _rejected(qt) -> int:
    value = getattr(getattr(qt, "QDialog", None), "Rejected", 0)
    return value if isinstance(value, int) else 0


def _run_popup(qt, dialog, is_popup):
    """Show the picker and wait for its result; returns the ``QDialog`` result code.

    Windows and Linux: ``exec_()``, as before. macOS: ``exec_()`` runs a native modal session, and Cocoa then drops
    every click on the main window (Qt's ``-[QNSPanel worksWhenModal]`` is NO for a parent of the modal window), so Qt
    never sees an outside click and the popup can neither apply nor close. There the popup is shown non-modal and a
    local ``QEventLoop`` waits. The loop ends when the dialog finishes (Apply, Cancel, X, Esc, an outside click, the
    watch timer, ``close_algorithm_picker``), when the dialog is destroyed, or when the application quits
    (``QCoreApplication::exit`` also ends every nested loop). A dialog that is already finished, or that is not
    visible after ``show()``, never enters the loop. A result that cannot be read counts as Rejected.
    """

    if not _popup_waits_non_modal(qt, is_popup):
        return dialog.exec_()
    loop = qt.QEventLoop()
    state = {"done": False}

    def _end(*_args) -> None:
        state["done"] = True
        try:
            loop.quit()
        except Exception:
            pass

    connected = []
    app_getter = getattr(getattr(qt, "QApplication", None), "instance", None)
    app = app_getter() if callable(app_getter) else None
    for sender, name in ((dialog, "finished"), (dialog, "destroyed"), (app, "aboutToQuit")):
        signal = getattr(sender, name, None) if sender is not None else None
        if signal is None or not hasattr(signal, "connect"):
            continue
        try:
            signal.connect(_end)
            connected.append(signal)
        except Exception:
            pass
    # Qt's Cocoa dispatcher loses a quit() that runs inside a nested manual processEvents() (its interrupt flag is
    # reset when that call returns; measured with the ui2 probe), and the loop would then wait forever. A
    # repeating timer therefore re-sends quit() from the loop's own level once the wait is over.
    requit = None
    if hasattr(qt, "QTimer"):
        try:
            requit = qt.QTimer()
            requit.setInterval(100)
            requit.timeout.connect(lambda: _end() if state["done"] else None)
        except Exception:
            requit = None
    try:
        dialog.show()
        visible = dialog.isVisible() if callable(getattr(dialog, "isVisible", None)) else True
        if not visible and not state["done"]:
            dialog.reject()
        if not state["done"]:
            if requit is not None:
                requit.start()
            loop.exec_()
    finally:
        if requit is not None:
            try:
                requit.stop()
            except Exception:
                pass
        for signal in connected:
            try:
                signal.disconnect(_end)
            except Exception:
                pass
    try:
        result = dialog.result()
    except Exception:
        return _rejected(qt)
    return result if isinstance(result, int) else _rejected(qt)


def close_algorithm_picker(owner) -> None:
    """Discard and close an open algorithm picker (module cleanup or exit); this also ends the macOS wait."""

    closer = getattr(owner, "_algorithmPickerCloser", None)
    if callable(closer):
        try:
            closer()
        except Exception:
            pass


def choose_algorithms(owner, family):
    import qt
    import slicer

    if getattr(owner, "_pipelineActive", False) or getattr(owner, "_selectedSetupActive", False):
        return False
    # popup-mac: on macOS the picker no longer runs a modal session, so a second picker must not open while one is open.
    if getattr(owner, "_algorithmPickerOpen", False) is True:
        return False
    zh = getattr(owner, "_onboardingLanguage", "en").startswith("zh")
    is_anatomical = family == "anatomical_priors"
    if is_anatomical:
        choices = tuple(ANATOMICAL_PRIORS)
        title_text = "解剖先验" if zh else "Anatomical priors"
        active = tuple(field for field in choices if getattr(owner._parameterNode, field, False))
        current = active or _remembered_anatomical_priors(owner)
    else:
        _, choices, _ = FAMILIES[family]
        title_text = "选择算法（可多选）" if zh else "Select algorithms (multiple allowed)"
        target_attr = FAMILY_TARGET_ATTRS.get(family)
        if family in _cleared_algorithm_families(owner) and target_attr and not getattr(owner._parameterNode, target_attr, False):
            current = ()
        else:
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
    always_tips = getattr(qt.Qt, "WA_AlwaysShowToolTips", None)
    if always_tips is not None and hasattr(dialog, "setAttribute"):
        dialog.setAttribute(always_tips, True)
    try:
        from shared.ui_styles import is_dark_mode, ui_stylesheet

        dark = is_dark_mode(getattr(owner, "_uiWidget", None))
        dialog.setStyleSheet(ui_stylesheet(dark=dark, widget=dialog))
    except Exception:
        dark = None

    popup_state = {"seen_visible": False, "finished": False}
    popup_watch_ref = [None]

    def _stop_popup_watch() -> None:
        watch = popup_watch_ref[0]
        if watch is not None:
            try:
                watch.stop()
            except Exception:
                pass

    def _finish_popup(accept: bool) -> None:
        popup_state["finished"] = True
        _stop_popup_watch()
        if accept:
            dialog.accept()
        else:
            dialog.reject()

    finished_sig = getattr(dialog, "finished", None)
    if finished_sig is not None and hasattr(finished_sig, "connect"):
        try:
            finished_sig.connect(lambda _result=0: (popup_state.__setitem__("finished", True), _stop_popup_watch()))
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
        close_btn.connect("clicked(bool)", lambda _checked=False: _finish_popup(False))
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
    viewport = scroll.viewport()
    if viewport is not None and hasattr(viewport, "setObjectName"):
        viewport.setObjectName("algorithmPickerViewport")
    body = qt.QWidget()
    if hasattr(body, "setObjectName"):
        body.setObjectName("algorithmPickerBody")
    body_layout = qt.QVBoxLayout(body)
    body_layout.setContentsMargins(0, 0, 4, 0)
    body_layout.setSpacing(6)
    scroll.setWidget(body)
    # QScrollArea uses QPalette.Base for its viewport, unlike the popup.
    # Use ID-scoped selectors so background transparency never cascades to QComboBox popups.
    for surface, selector in (
        (scroll, "QScrollArea#algorithmPickerScrollArea"),
        (viewport, "QWidget#algorithmPickerViewport"),
        (body, "QWidget#algorithmPickerBody"),
    ):
        if surface is not None:
            surface.setAutoFillBackground(False)
            surface.setStyleSheet(f"{selector} {{ background: transparent; }}")
    layout.addWidget(scroll)

    from shared.ui_compact import _position_input_help_card, _qt_int, _qt_val

    def resize_popup():
        anchor_visible = anchor is not None and anchor.isVisible()
        screen = qt.QApplication.screenAt(anchor.mapToGlobal(qt.QPoint(0, 0))) if anchor_visible else None
        if screen is None:
            screen = _qt_val(qt.QApplication, "primaryScreen")
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
    # G3: where FSL is not offered, Eddy and Topup are disabled and unchecked; Brain mask and DWI denoising stay.
    fsl_missing = family == "preprocessing" and not _fsl_stages_offered()
    fsl_tip = ""
    if fsl_missing:
        from shared.selected_setup import fsl_unavailable_text

        fsl_tip = fsl_unavailable_text(zh=zh)
    # B2: algorithms that the chosen DATA cannot run. They stay in the stored selection (see Apply below).
    data_tips = {} if is_anatomical else _data_block_texts(owner, choices, zh)
    for value in choices:
        label = ANATOMICAL_PRIORS[value][int(zh)] if is_anatomical else LABELS[value]
        box = qt.QCheckBox(label)
        box.objectName = "algorithm_" + value
        fsl_blocked = fsl_missing and value in FSL_STAGES
        blocked_tip = fsl_tip if fsl_blocked else data_tips.get(value, "")
        blocked = fsl_blocked or value in data_tips
        box.checked = value in current and not blocked
        if blocked:
            box.setEnabled(False)
            box.setToolTip(blocked_tip)
        if pointing_cursor is not None and hasattr(box, "setCursor"):
            box.setCursor(pointing_cursor)
        boxes[value] = box
        _add_algorithm_row(
            qt, owner, body_layout, box, family, value, staged, zh, resize_popup,
            dark=dark, disabled=blocked, disabled_tooltip=blocked_tip,
        )

    if fsl_missing and fsl_tip:
        avail_note = qt.QLabel(fsl_tip)
        if hasattr(avail_note, "setObjectName"):
            avail_note.setObjectName("algorithmPickerAvailabilityNote")
        avail_note.wordWrap = True
        if hasattr(avail_note, "setMaximumWidth"):
            avail_note.setMaximumWidth(360)
        if hasattr(avail_note, "setTextFormat") and hasattr(qt.Qt, "PlainText"):
            avail_note.setTextFormat(qt.Qt.PlainText)
        body_layout.addWidget(avail_note)

    note = qt.QLabel("")
    if hasattr(note, "setObjectName"):
        note.setObjectName("algorithmPickerNote")
    note.wordWrap = True
    note.setMaximumWidth(360)
    note.hide()
    body_layout.addWidget(note)
    btn_row = qt.QHBoxLayout()
    if hasattr(btn_row, "setContentsMargins"):
        btn_row.setContentsMargins(0, 4, 0, 0)
    if hasattr(btn_row, "setSpacing"):
        btn_row.setSpacing(8)
    if hasattr(btn_row, "addStretch"):
        btn_row.addStretch(1)
    cancel_btn = qt.QPushButton("取消" if zh else "Cancel")
    cancel_btn.setObjectName("algorithmPickerCancelButton")
    apply = qt.QPushButton("应用" if zh else "Apply")
    apply.setObjectName("algorithmPickerApplyButton")
    for btn in (cancel_btn, apply):
        if pointing_cursor is not None and hasattr(btn, "setCursor"):
            btn.setCursor(pointing_cursor)

    def update(*_):
        apply.enabled = True

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
            note.show()
            resize_popup()
            scroll.ensureWidgetVisible(note)
            return
        _finish_popup(True)

    cancel_btn.connect("clicked()", lambda: _finish_popup(False))
    apply.connect("clicked()", validate_and_accept)
    btn_row.addWidget(cancel_btn)
    btn_row.addWidget(apply)
    layout.addLayout(btn_row)

    resize_popup()

    # Fix S8: the state at open (tick boxes and staged values); an outside click commits only when it changed.
    unreadable = object()

    def popup_state_now(fail):
        values = []
        for attribute, getter in staged.items():
            try:
                values.append((attribute, getter()))
            except Exception:
                values.append((attribute, fail))
        return ({value: bool(box.checked) for value, box in boxes.items()}, values)

    opened_state = popup_state_now(unreadable)

    def popup_changed():
        changed_marker = object()  # a value that cannot be read now counts as changed (Apply shows its error)
        return popup_state_now(changed_marker) != opened_state

    outside_filter = (_install_outside_close_apply(qt, dialog, popup_state, validate_and_accept, popup_changed,
                                                   no_replay=_popup_waits_non_modal(qt, popup_flag))
                      if popup_flag else None)

    popup_watch = None
    if hasattr(qt, "QTimer") and popup_flag:
        try:
            popup_watch = qt.QTimer()
            popup_watch_ref[0] = popup_watch
            popup_watch.setInterval(50)

            def _watch_popup() -> None:
                if popup_state["finished"]:
                    _stop_popup_watch()
                    return
                res_fn = getattr(dialog, "result", None)
                res = res_fn() if callable(res_fn) else 0
                if isinstance(res, int) and res != 0:
                    popup_state["finished"] = True
                    _stop_popup_watch()
                    return
                vis = dialog.isVisible() if callable(getattr(dialog, "isVisible", None)) else bool(getattr(dialog, "visible", True))
                if vis:
                    popup_state["seen_visible"] = True
                elif popup_state["seen_visible"]:
                    popup_state["finished"] = True
                    _stop_popup_watch()
                    dialog.reject()

            if hasattr(popup_watch, "timeout") and hasattr(popup_watch.timeout, "connect"):
                popup_watch.timeout.connect(_watch_popup)
            elif hasattr(popup_watch, "connect"):
                popup_watch.connect("timeout()", _watch_popup)
            popup_watch.start()
        except Exception:
            popup_watch = None

    owner._algorithmPickerOpen = True
    owner._algorithmPickerCloser = lambda: None if popup_state["finished"] else _finish_popup(False)
    try:
        accepted = _run_popup(qt, dialog, bool(popup_flag)) == qt.QDialog.Accepted
    finally:
        owner._algorithmPickerOpen = False
        owner._algorithmPickerCloser = None
        popup_state["finished"] = True
        _stop_popup_watch()
        if outside_filter is not None:
            try:
                dialog.removeEventFilter(outside_filter)
            except Exception:
                pass
    if accepted:
        # A data-blocked algorithm that was stored before stays stored (it is skipped at Run, not removed here).
        selection = [value for value, box in boxes.items()
                     if box.checked or (value in data_tips and value in current)]
        if is_anatomical:
            _set_anatomical_priors(owner, selection, clear_memory=not selection)
        else:
            target_attr = FAMILY_TARGET_ATTRS.get(family)
            box_name, _, _ = FAMILIES[family]
            ui_box = getattr(getattr(owner, "ui", None), box_name, None)
            can_enable = True
            if ui_box is not None:
                enabled_attr = getattr(ui_box, "isEnabled", None)
                enabled_val = enabled_attr() if callable(enabled_attr) else getattr(ui_box, "enabled", True)
                if isinstance(enabled_val, bool):
                    can_enable = enabled_val
            with slicer.util.NodeModify(owner._parameterNode):
                for attribute, value in staged.items():
                    setattr(owner._parameterNode, attribute, value())
                if selection:
                    _cleared_algorithm_families(owner).discard(family)
                    if target_attr and can_enable:
                        setattr(owner._parameterNode, target_attr, True)
                    _apply_family_selection_to_node(owner._parameterNode, family, selection)
                else:
                    _cleared_algorithm_families(owner).add(family)
                    if target_attr:
                        setattr(owner._parameterNode, target_attr, False)
            owner._lastEnvironmentResult = owner._lastEnvironmentSignature = None
            desired_checked = bool(selection and target_attr and can_enable)
            if ui_box is not None and (not selection or can_enable):
                sync_checkbox = getattr(owner, "_sync_checkbox", None)
                if callable(sync_checkbox):
                    sync_checkbox(ui_box, desired_checked)
                elif hasattr(ui_box, "setChecked"):
                    ui_box.setChecked(desired_checked)
            from shared.ui_compact import refresh_compact_algorithms

            refresh_compact_algorithms(owner)
            refresh_adv = getattr(owner, "_refresh_advanced_run_settings", None)
            if callable(refresh_adv):
                refresh_adv()
            schedule_save = getattr(owner, "_schedule_common_user_settings_save", None)
            if callable(schedule_save):
                schedule_save()
        if hasattr(qt, "QTimer"):
            from shared.ui_on_demand_resources import maybe_prompt_on_demand_install

            qt.QTimer.singleShot(0, lambda: maybe_prompt_on_demand_install(owner))
    try:
        dialog.deleteLater()
    except Exception:  # popup-mac: the dialog may already be destroyed when the wait ended
        pass
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
        checkbox.setFixedWidth(indicator_width)
        checkbox.setMinimumHeight(18)
        def _on_anatomical_clicked(checked: bool) -> None:
            if getattr(owner, "_pipelineActive", False) or getattr(owner, "_selectedSetupActive", False):
                lang = getattr(owner, "_onboardingLanguage", "en")
                _refresh_anatomical_priors(owner, isinstance(lang, str) and lang.startswith("zh"))
                return
            if checked:
                fields = _remembered_anatomical_priors(owner) or tuple(ANATOMICAL_PRIORS)
                _set_anatomical_priors(owner, fields)
            else:
                _set_anatomical_priors(owner, (), clear_memory=False)
            if checked and hasattr(qt, "QTimer"):
                from shared.ui_on_demand_resources import maybe_prompt_on_demand_install

                qt.QTimer.singleShot(0, lambda: maybe_prompt_on_demand_install(owner))

        checkbox.clicked.connect(_on_anatomical_clicked)
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
        labels_before = len(owner._productLabelButtons)
        make_label_clickable(2, checkbox)
        if len(owner._productLabelButtons) > labels_before:  # b2-os: ui_compact greys this label on an OS block
            owner._surfacePostprocessLabel = owner._productLabelButtons[-1][0]
        form.takeAt(form.indexOf(checkbox))
        checkbox.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
        checkbox.setMinimumWidth(0)
        form.setWidget(2, qt.QFormLayout.FieldRole, click_area(checkbox, checkbox, expand=True))
        if hasattr(checkbox, "clicked"):
            def _on_surface_post_clicked(checked: bool) -> None:
                if checked and hasattr(qt, "QTimer"):
                    from shared.ui_on_demand_resources import maybe_prompt_on_demand_install

                    qt.QTimer.singleShot(0, lambda: maybe_prompt_on_demand_install(owner))

            checkbox.clicked.connect(_on_surface_post_clicked)

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
        if hasattr(checkbox, "clicked"):
            def _on_family_clicked(checked: bool, _family=family) -> None:
                if checked:
                    _cleared_algorithm_families(owner).discard(_family)
                    from shared.ui_compact import refresh_compact_algorithms

                    refresh_compact_algorithms(owner)
                if checked and hasattr(qt, "QTimer"):
                    from shared.ui_on_demand_resources import maybe_prompt_on_demand_install

                    qt.QTimer.singleShot(0, lambda: maybe_prompt_on_demand_install(owner))

            checkbox.clicked.connect(_on_family_clicked)
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


def ellipsis_icon(ink):
    import qt

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
    return qt.QIcon(pixmap)


def refresh_picker_icons(owner, *, dark=None):
    from shared.ui_styles import is_dark_mode

    dark = is_dark_mode(getattr(owner, "_uiWidget", None)) if dark is None else dark
    icon = ellipsis_icon("#8CD2F5" if dark else "#326A8A")
    buttons = getattr(owner, "_algorithmButtons", None)
    if isinstance(buttons, dict):
        for button in buttons.values():
            button.setIcon(icon)
    row = getattr(owner, "_anatomicalPriorsRow", None)
    if isinstance(row, tuple) and len(row) == 5:
        row[-1].setIcon(icon)


def refresh_picker_language(owner):
    label_buttons = getattr(owner, "_productLabelButtons", None)
    if isinstance(label_buttons, (list, tuple)):
        for label, button in label_buttons:
            button.setMinimumWidth(label.sizeHint.width())
    raw_lang = getattr(owner, "_onboardingLanguage", "en")
    zh = isinstance(raw_lang, str) and raw_lang.startswith("zh")
    _refresh_anatomical_priors(owner, zh)
    raw_cleared = getattr(owner, "_clearedAlgorithmFamilies", None)
    cleared = raw_cleared if isinstance(raw_cleared, (set, frozenset, list, tuple)) else ()
    node = getattr(owner, "_parameterNode", None)
    buttons = getattr(owner, "_algorithmButtons", None)
    if not isinstance(buttons, dict):
        return
    for family, button in buttons.items():
        selected = selected_algorithms(node, family)
        name = {
            "preprocessing": ("预处理", "preprocessing"),
            "microstructure": ("微结构分析", "microstructure"),
            "tractography": ("纤维追踪", "tractography"),
            "bundles": ("纤维束分析", "bundle analysis"),
            "connectome": ("连接组", "connectome"),
        }[family][0 if zh else 1]
        action = "选择" + name + "算法" if zh else "Select " + name + " algorithms"
        target_attr = FAMILY_TARGET_ATTRS.get(family)
        is_cleared = family in cleared and not (node is not None and target_attr and getattr(node, target_attr, False))
        current_text = ("未选择" if zh else "None selected") if is_cleared else ", ".join(LABELS[value] for value in selected)
        tip = action + ("（可多选）。当前：" if zh else " (multiple allowed). Current: ") + current_text
        if getattr(button, "text", None) != "...":
            button.text = "..."
        if getattr(button, "accessibleName", None) != action:
            button.setAccessibleName(action)
        if getattr(button, "toolTip", None) != tip:
            button.toolTip = tip
