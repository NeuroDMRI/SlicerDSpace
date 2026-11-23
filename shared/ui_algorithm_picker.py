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
    "preprocessing": "denoise,eddy",
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
    ("preprocessing", "eddy"): (
        ("preprocessingPhaseEncodingDirection", "preprocessingPhaseEncodingDirectionComboBox", "preprocessingPhaseEncodingDirectionLabel"),
        ("preprocessingReadoutTime", "preprocessingReadoutTimeSpinBox", "preprocessingReadoutTimeLabel"),
    ),
    ("preprocessing", "topup"): (
        ("preprocessingPhaseEncodingDirection", "preprocessingPhaseEncodingDirectionComboBox", "preprocessingPhaseEncodingDirectionLabel"),
        ("preprocessingReadoutTime", "preprocessingReadoutTimeSpinBox", "preprocessingReadoutTimeLabel"),
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


def _unavailable_reason(qt, owner, algorithm, reason, marker, resize_popup, dark=None, *, state=None):
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

    def collapse() -> None:
        if shown["value"]:
            shown["value"] = False
            label.setVisible(False)

    if isinstance(state, dict):
        state["collapse"] = collapse
    marker.clicked.connect(toggle)
    return label


def _gear_icon(qt, *, use_dark: bool, disabled: bool):
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
    return qt.QIcon(pixmap)


def _is_batch_mode(owner) -> bool:
    check = getattr(owner, "_input_preview_is_batch", None)
    if callable(check):
        verdict = check()
        if isinstance(verdict, bool):
            return verdict
    node = getattr(owner, "_parameterNode", None)
    cohort = bool(
        node is not None
        and getattr(node, "experimentsEnabled", False) is True
        and getattr(node, "experimentScope", "") == "cohort"
    )
    return cohort or bool(node is not None and getattr(node, "batchMode", False) is True)


def _data_block_texts(owner, choices, zh) -> dict:
    """Algorithm id -> reason text for the algorithms of ``choices`` that the current data cannot run."""
    try:
        from shared import algorithm_availability

        blocks = algorithm_availability.current_data_blocks(owner)
        return {value: blocks[value].text("zh" if zh else "en") for value in choices if value in blocks}
    except Exception:
        return {}


def _sync_shared_editor(attribute, editor, shared_editors, kind):
    peers = shared_editors.setdefault(attribute, [])
    peers.append(editor)
    if len(peers) < 2:
        return

    def _connect_peer(source):
        if kind == "combo":
            def on_changed(index, src=source):
                for peer in peers:
                    if peer is src:
                        continue
                    try:
                        was_blocked = peer.blockSignals(True)
                        try:
                            peer.setCurrentIndex(index)
                        finally:
                            peer.blockSignals(bool(was_blocked))
                    except Exception:
                        pass
            sig = getattr(source, "currentIndexChanged", None)
            if sig is not None and hasattr(sig, "connect"):
                try:
                    sig.connect(on_changed)
                except Exception:
                    pass
        elif kind == "double_spin":
            def on_changed(value, src=source):
                for peer in peers:
                    if peer is src:
                        continue
                    try:
                        was_blocked = peer.blockSignals(True)
                        try:
                            peer.setValue(float(value))
                        finally:
                            peer.blockSignals(bool(was_blocked))
                    except Exception:
                        pass
            sig = getattr(source, "valueChanged", None)
            if sig is not None and hasattr(sig, "connect"):
                try:
                    sig.connect(on_changed)
                except Exception:
                    pass

    if len(peers) == 2:
        _connect_peer(peers[0])
    _connect_peer(editor)


def _qt_supports_panel_widgets(qt, *, needs_spin: bool = False, needs_double_spin: bool = False,
                               needs_combo: bool = False) -> bool:
    """True when ``qt`` provides callable form and control factories (real Qt or full fake_qt doubles)."""

    if not callable(getattr(qt, "QFormLayout", None)) or not callable(getattr(qt, "QPixmap", None)):
        return False
    for needed, name, method in (
        (needs_spin, "QSpinBox", "setRange"),
        (needs_double_spin, "QDoubleSpinBox", "setRange"),
        (needs_combo, "QComboBox", "addItem"),
    ):
        if not needed:
            continue
        cls = getattr(qt, name, None)
        if not callable(cls) or (isinstance(cls, type) and not hasattr(cls, method)):
            return False
    return True


def _add_algorithm_row(qt, owner, layout, box, family, algorithm, staged, zh, resize_popup, dark=None,
                       disabled=False, disabled_tooltip="", create_marker=False, shared_editors=None):
    specs = ALGORITHM_SETTINGS.get((family, algorithm), ())
    need_marker = bool(disabled or create_marker)
    marker = _unavailable_marker(qt, owner, algorithm, disabled_tooltip, dark) if need_marker else None
    if marker is not None and not disabled:
        marker.setVisible(False)
    ui_root = getattr(owner, "ui", None)
    fields = [(attribute, getattr(ui_root, control, None), getattr(ui_root, label, None))
              for attribute, control, label in specs] if ui_root is not None else []
    fields = [(attribute, control, label) for attribute, control, label in fields
              if control is not None and label is not None]
    ukf = (family, algorithm) == ("tractography", "ukf")
    csd_act = (family, algorithm) == ("tractography", "csd_act") and _qt_supports_panel_widgets(
        qt, needs_spin=True, needs_double_spin=True, needs_combo=True)
    dti_det = (family, algorithm) == ("tractography", "dti_det") and _qt_supports_panel_widgets(
        qt, needs_spin=True, needs_double_spin=True, needs_combo=True)
    saf = (family, algorithm) == ("tractography", "saf") and _qt_supports_panel_widgets(
        qt, needs_spin=True, needs_double_spin=True)
    denoise_panel = (family, algorithm) == ("preprocessing", "denoise") and _qt_supports_panel_widgets(qt)
    micro_dti_panel = (family, algorithm) == ("microstructure", "dti") and _qt_supports_panel_widgets(
        qt, needs_combo=True)
    freewater_panel = (family, algorithm) == ("microstructure", "freewater") and _qt_supports_panel_widgets(
        qt, needs_double_spin=True)
    ifw_t2_panel = (family, algorithm) == ("microstructure", "ifw") and _qt_supports_panel_widgets(
        qt, needs_double_spin=True)
    tractcloud_panel = (family, algorithm) == ("bundles", "tractcloud") and _qt_supports_panel_widgets(
        qt, needs_spin=True)
    # Batch 2 item 5: every bundle algorithm gets a "Tractography source" combo built in code.
    bundle_source = family == "bundles" and algorithm in ("wmql", "wma", "tractcloud")
    has_code_panel = any((
        ukf, csd_act, dti_det, saf,
        denoise_panel, micro_dti_panel, freewater_panel, ifw_t2_panel, tractcloud_panel,
        bundle_source,
    ))
    if not fields and not has_code_panel:
        if marker is None:
            layout.addWidget(box)
            return {"gear": None, "panel": None, "marker": None, "reason_label": None, "collapse_reason": None}
        row = qt.QHBoxLayout()
        row.addWidget(box, 1)
        row.addWidget(marker)
        layout.addLayout(row)
        reason_state = {}
        reason_label = _unavailable_reason(
            qt, owner, algorithm, disabled_tooltip, marker, resize_popup, dark, state=reason_state
        )
        layout.addWidget(reason_label)
        return {
            "gear": None,
            "panel": None,
            "marker": marker,
            "reason_label": reason_label,
            "collapse_reason": reason_state.get("collapse"),
        }
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
    button.setIcon(_gear_icon(qt, use_dark=use_dark, disabled=disabled))
    button.setIconSize(qt.QSize(15, 15))
    button.setFixedSize(28, 24)
    button.setCheckable(True)
    row.addWidget(button)
    if marker is not None:
        row.addWidget(marker)
    layout.addLayout(row)
    reason_label = None
    reason_state = {}
    if marker is not None:
        reason_label = _unavailable_reason(
            qt, owner, algorithm, disabled_tooltip, marker, resize_popup, dark, state=reason_state
        )
        layout.addWidget(reason_label)
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
            if isinstance(shared_editors, dict):
                _sync_shared_editor(attribute, editor, shared_editors, "combo")
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
            if isinstance(shared_editors, dict):
                _sync_shared_editor(attribute, editor, shared_editors, "double_spin")
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
    if csd_act:
        from shared.ui_tractography_params import add_staged_csd_act_fields
        csd_staged, streamlines_staged = add_staged_csd_act_fields(form, owner._parameterNode, zh, dark=use_dark)
        staged["csdActParameters"] = csd_staged
        if streamlines_staged is not None:
            staged["orchestrationSelectStreamlines"] = streamlines_staged
    if dti_det:
        from shared.ui_tractography_params import add_staged_dti_det_fields
        staged["dtiParameters"] = add_staged_dti_det_fields(form, owner._parameterNode, zh, dark=use_dark)
    if saf:
        from shared.ui_tractography_params import add_staged_saf_fields
        staged["safParameters"] = add_staged_saf_fields(form, owner._parameterNode, zh)
    if denoise_panel:
        from shared.ui_tractography_params import add_staged_denoise_fields
        staged["preprocessingRunGibbs"] = add_staged_denoise_fields(form, owner._parameterNode, zh)
    if micro_dti_panel:
        from shared.ui_tractography_params import add_staged_microstructure_dti_fields
        staged["microstructureFitMethod"] = add_staged_microstructure_dti_fields(
            form, owner._parameterNode, zh, dark=use_dark)
    if freewater_panel:
        from shared.ui_tractography_params import add_staged_freewater_fields
        staged["freeWaterParameters"] = add_staged_freewater_fields(form, owner._parameterNode, zh)
    if ifw_t2_panel:
        from shared.ui_tractography_params import add_staged_ifw_t2_fields
        staged["ifwParameters"] = add_staged_ifw_t2_fields(form, owner._parameterNode, zh)
    if bundle_source:
        from shared.ui_bundle_source import add_staged_source_field
        source_attribute, source_value = add_staged_source_field(
            qt, form, owner._parameterNode, algorithm, zh, selected_algorithms(owner._parameterNode, "tractography"),
            dark=use_dark)
        staged[source_attribute] = source_value
    if tractcloud_panel:
        from shared.ui_tractography_params import add_staged_tractcloud_fields
        staged["tractcloudParameters"] = add_staged_tractcloud_fields(form, owner._parameterNode, zh)
    layout.addWidget(panel)
    panel.hide()
    def toggle_settings(visible):
        panel.setVisible(visible)
        resize_popup()

    button.toggled.connect(toggle_settings)
    return {
        "gear": button,
        "panel": panel,
        "marker": marker,
        "reason_label": reason_label,
        "collapse_reason": reason_state.get("collapse"),
    }


_TOPUP_DROP_BOX_CLASSES: dict[type, type] = {}


def _set_drop_box_properties(widget, **props) -> None:
    if widget is None or not hasattr(widget, "setProperty"):
        return
    changed = False
    for name, value in props.items():
        if bool(widget.property(name)) != bool(value):
            widget.setProperty(name, bool(value))
            changed = True
    if not changed:
        return
    style_fn = getattr(widget, "style", None)
    style = style_fn() if callable(style_fn) else style_fn
    if style is not None:
        if hasattr(style, "unpolish"):
            style.unpolish(widget)
        if hasattr(style, "polish"):
            style.polish(widget)
    if hasattr(widget, "update"):
        widget.update()


def _make_topup_drop_box(qt, on_drop):
    """Return a QFrame drop zone for reverse-PE b=0 files (reuses Prepare Inputs path extraction)."""
    from shared.ui_prepare_inputs import _extract_drop_paths, _set_tile_drag_active

    qframe_cls = getattr(qt, "QFrame", None)
    if isinstance(qframe_cls, type):
        box_cls = _TOPUP_DROP_BOX_CLASSES.get(qframe_cls)
        if box_cls is None:
            class _TopupReverseB0DropBox(qframe_cls):
                def dragEnterEvent(self, event):
                    qt_mod, _callback = getattr(self, "_topupDropCtx", (None, None))
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
                    qt_mod, callback = getattr(self, "_topupDropCtx", (None, None))
                    mime = event.mimeData() if hasattr(event, "mimeData") else None
                    paths = _extract_drop_paths(qt_mod, mime) if qt_mod is not None else []
                    if paths and callable(callback):
                        callback(str(paths[0]))
                        if hasattr(event, "acceptProposedAction"):
                            event.acceptProposedAction()
                        return
                    if hasattr(event, "ignore"):
                        event.ignore()

            box_cls = _TopupReverseB0DropBox
            _TOPUP_DROP_BOX_CLASSES[qframe_cls] = box_cls

        frame = box_cls()
        frame._topupDropCtx = (qt, on_drop)
        no_frame = getattr(qframe_cls, "NoFrame", None)
        if no_frame is not None and hasattr(frame, "setFrameShape"):
            frame.setFrameShape(no_frame)
        return frame, True
    frame = qt.QFrame() if hasattr(qt, "QFrame") else qt.QWidget()
    return frame, False


def _add_topup_reverse_b0_controls(
    qt, owner, dialog, layout, box, row_controls, staged, popup_state, zh, resize_popup, *, dark=None,
):
    """Inline reverse-PE b=0 drop zone (left) + Browse button (right) under Topup; gates Topup live."""
    from pathlib import Path
    from shared.topup_parameters import topup_unavailable_reason
    from shared.ui_styles import is_dark_mode

    use_dark = is_dark_mode(getattr(owner, "_uiWidget", None)) if dark is None else bool(dark)
    pointing = getattr(qt.Qt, "PointingHandCursor", None)
    initial_path = str(getattr(getattr(owner, "_parameterNode", None), "preprocessingReversePhaseB0", "") or "").strip()

    container = qt.QWidget()
    container.setObjectName("topupReversePhaseB0Container")
    container_layout = qt.QVBoxLayout(container)
    if hasattr(container_layout, "setContentsMargins"):
        container_layout.setContentsMargins(22, 0, 0, 2)
    if hasattr(container_layout, "setSpacing"):
        container_layout.setSpacing(4)

    muted_ink = "#9AA4AF" if use_dark else "#55636F"
    ink = "#E6EDF3" if use_dark else "#1F2937"
    drop_bg = "#161B22" if use_dark else "#F8FAFC"
    filled_bg = "#1C232B" if use_dark else "#FFFFFF"
    hover_bg = "#1F2D3D" if use_dark else "#EBF3F7"
    drop_line = "#4B5563" if use_dark else "#94A3B8"
    drag_accent = "#7CC4EA" if use_dark else "#335B70"
    err_color = "#FF7B72" if use_dark else "#C62828"

    label_text = "反向相位 b=0 图像" if zh else "Reverse-PE b=0 image"
    field_label = qt.QLabel(label_text)
    field_label.setObjectName("topupReversePhaseB0Label")
    container_layout.addWidget(field_label)

    input_row = qt.QHBoxLayout()
    if hasattr(input_row, "setContentsMargins"):
        input_row.setContentsMargins(0, 0, 0, 0)
    if hasattr(input_row, "setSpacing"):
        input_row.setSpacing(6)

    dropped_handler = [None]
    drop_box, handles_drops = _make_topup_drop_box(
        qt, lambda dropped_path: dropped_handler[0](dropped_path) if callable(dropped_handler[0]) else None
    )
    drop_box.setObjectName("topupReversePhaseB0DropBox")
    if hasattr(drop_box, "setAcceptDrops"):
        drop_box.setAcceptDrops(True)
    styled_bg = getattr(qt.Qt, "WA_StyledBackground", None)
    if styled_bg is not None and hasattr(drop_box, "setAttribute"):
        drop_box.setAttribute(styled_bg, True)
    if hasattr(drop_box, "setMinimumHeight"):
        drop_box.setMinimumHeight(34)
    if hasattr(drop_box, "setMinimumWidth"):
        drop_box.setMinimumWidth(195)

    drop_layout = qt.QHBoxLayout(drop_box)
    if hasattr(drop_layout, "setContentsMargins"):
        drop_layout.setContentsMargins(8, 4, 6, 4)
    if hasattr(drop_layout, "setSpacing"):
        drop_layout.setSpacing(4)

    path_edit = qt.QLineEdit(drop_box)
    path_edit.setObjectName("topupReversePhaseB0LineEdit")
    if hasattr(path_edit, "setAcceptDrops"):
        path_edit.setAcceptDrops(not handles_drops)
    path_edit.setText(initial_path)
    if hasattr(path_edit, "setPlaceholderText"):
        path_edit.setPlaceholderText(
            "拖入反向 b=0 文件到此…" if zh else "Drop reverse-PE b=0 file here…"
        )
    path_tip = (
        "将与主 DWI 相位编码方向相反的 b=0 图像拖入此处，或点击右侧「浏览…」（例如主 DWI 为 AP 时提供 PA b=0）。" if zh
        else "Drop a reverse phase-encoding b=0 image here or click Browse… (e.g. PA b=0 for an AP main DWI)."
    )
    path_edit.setToolTip(initial_path or path_tip)
    if hasattr(drop_box, "setToolTip"):
        drop_box.setToolTip(initial_path or path_tip)

    clear_btn = qt.QToolButton(drop_box)
    clear_btn.setObjectName("topupReversePhaseB0ClearButton")
    clear_btn.setText("×")
    if hasattr(clear_btn, "setAutoRaise"):
        clear_btn.setAutoRaise(True)
    if hasattr(clear_btn, "setFixedSize"):
        clear_btn.setFixedSize(18, 18)
    if hasattr(clear_btn, "setToolTip"):
        clear_btn.setToolTip("清除反向 b=0 图像" if zh else "Clear reverse-PE b=0 image")
    if pointing is not None and hasattr(clear_btn, "setCursor"):
        clear_btn.setCursor(pointing)
    clear_btn.setVisible(bool(initial_path))

    drop_layout.addWidget(path_edit, 1)
    drop_layout.addWidget(clear_btn)

    browse_btn = qt.QPushButton("浏览…" if zh else "Browse…")
    browse_btn.setObjectName("topupReversePhaseB0BrowseButton")
    if hasattr(browse_btn, "setMinimumHeight"):
        browse_btn.setMinimumHeight(34)
    if hasattr(browse_btn, "setToolTip"):
        browse_btn.setToolTip("选择反向相位编码 b=0 图像文件" if zh else "Choose a reverse phase-encoding b=0 image file")
    if pointing is not None and hasattr(browse_btn, "setCursor"):
        browse_btn.setCursor(pointing)

    input_row.addWidget(drop_box, 1)
    input_row.addWidget(browse_btn)
    container_layout.addLayout(input_row)
    container.setStyleSheet(
        f"QLabel#topupReversePhaseB0Label {{ color: {muted_ink}; font-size: 11px; }}"
        f"QFrame#topupReversePhaseB0DropBox {{ background: {drop_bg}; border: 2px dashed {drop_line}; border-radius: 8px; }}"
        f"QFrame#topupReversePhaseB0DropBox[hasFile=\"true\"] {{ background: {filled_bg}; border: 1.5px solid #478CB3; }}"
        f"QFrame#topupReversePhaseB0DropBox[invalid=\"true\"] {{ background: {filled_bg}; border: 1.5px solid {err_color}; }}"
        f"QFrame#topupReversePhaseB0DropBox[dragActive=\"true\"] {{ background: {hover_bg}; border: 2px dashed {drag_accent}; }}"
        f"QLineEdit#topupReversePhaseB0LineEdit {{ background: transparent; border: 0; padding: 0; font-size: 11px; color: {ink}; }}"
        f"QToolButton#topupReversePhaseB0ClearButton {{ background: transparent; border: 0; border-radius: 4px; color: {muted_ink}; }}"
        f"QToolButton#topupReversePhaseB0ClearButton:hover {{ background: {hover_bg}; color: {ink}; }}"
    )
    layout.addWidget(container)

    state = {
        "text": initial_path,
        "enabled": not bool(topup_unavailable_reason(initial_path, batch_mode=False, zh=zh)),
    }
    _set_drop_box_properties(
        drop_box,
        hasFile=bool(initial_path and state["enabled"]),
        invalid=bool(initial_path and not state["enabled"]),
        dragActive=False,
    )

    def _read_text() -> str:
        raw = getattr(path_edit, "text", None)
        if callable(raw):
            raw = raw()
        if isinstance(raw, str):
            return raw.strip()
        return state["text"]

    gear = row_controls.get("gear")
    panel = row_controls.get("panel")
    marker = row_controls.get("marker")
    reason_label = row_controls.get("reason_label")
    collapse_reason = row_controls.get("collapse_reason")
    caption = "设置" if zh else "Settings"

    def _apply_topup_gate(new_text=None, *, auto_check: bool = True) -> None:
        if isinstance(new_text, str):
            state["text"] = new_text.strip()
        else:
            state["text"] = _read_text()
        raw_path = state["text"]
        clear_btn.setVisible(bool(raw_path))
        path_edit.setToolTip(raw_path or path_tip)
        if hasattr(drop_box, "setToolTip"):
            drop_box.setToolTip(raw_path or path_tip)
        reason = topup_unavailable_reason(raw_path, batch_mode=False, zh=zh)
        was_enabled = state["enabled"]
        now_enabled = not bool(reason)
        state["enabled"] = now_enabled
        if now_enabled:
            box.setEnabled(True)
            box.setToolTip("")
            if auto_check and not was_enabled:
                box.setChecked(True)
            if gear is not None:
                gear.setEnabled(True)
                gear.setToolTip(caption)
                gear.setIcon(_gear_icon(qt, use_dark=use_dark, disabled=False))
            if marker is not None:
                marker.setVisible(False)
            if reason_label is not None:
                if callable(collapse_reason):
                    collapse_reason()
                else:
                    reason_label.setVisible(False)
            _set_drop_box_properties(drop_box, hasFile=True, invalid=False)
        else:
            box.setChecked(False)
            box.setEnabled(False)
            box.setToolTip(reason)
            if gear is not None:
                if hasattr(gear, "setChecked"):
                    gear.setChecked(False)
                gear.setEnabled(False)
                gear.setToolTip(reason)
                gear.setIcon(_gear_icon(qt, use_dark=use_dark, disabled=True))
            if panel is not None:
                panel.hide()
            if marker is not None:
                marker.setVisible(True)
                marker.setToolTip(reason)
                marker.setAccessibleName(reason)
            if reason_label is not None:
                reason_label.setText(reason)
            _set_drop_box_properties(drop_box, hasFile=False, invalid=bool(raw_path))
        resize_popup()

    def _on_dropped_path(dropped_path: str) -> None:
        cleaned = str(dropped_path or "").strip()
        if not cleaned:
            return
        path_edit.setText(cleaned)
        if state["text"] != cleaned:
            _apply_topup_gate(cleaned, auto_check=True)

    dropped_handler[0] = _on_dropped_path

    def _on_browse(*_args) -> None:
        file_dialog_cls = getattr(qt, "QFileDialog", None)
        if file_dialog_cls is None or not hasattr(file_dialog_cls, "getOpenFileName"):
            return
        start_dir = ""
        current_text = _read_text()
        if current_text:
            candidate = Path(current_text).expanduser()
            parent = candidate if candidate.is_dir() else candidate.parent
            if parent.is_dir():
                start_dir = str(parent)
        if not start_dir:
            input_dir = str(getattr(getattr(owner, "_parameterNode", None), "inputDirectory", "") or "").strip()
            if input_dir:
                cand = Path(input_dir).expanduser()
                parent = cand if cand.is_dir() else cand.parent
                if parent.is_dir():
                    start_dir = str(parent)
        browse_title = "选择反向相位编码 b=0 图像" if zh else "Select reverse-PE b=0 image"
        file_filter = (
            "MRI / Diffusion images (*.nii *.nii.gz *.nrrd *.nhdr *.mif *.mif.gz *.mgh *.mgz);;All files (*)"
        )
        file_dialog_parent = None
        try:
            import slicer

            main_win_fn = getattr(getattr(slicer, "util", None), "mainWindow", None)
            if callable(main_win_fn):
                file_dialog_parent = main_win_fn()
        except Exception:
            file_dialog_parent = None
        if file_dialog_parent is None:
            parent_getter = getattr(dialog, "parentWidget", None)
            if callable(parent_getter):
                try:
                    file_dialog_parent = parent_getter()
                except Exception:
                    file_dialog_parent = None
        popup_state["browsing_file"] = True
        hid_for_browse = False
        chosen = ""
        try:
            # Never parent a native QFileDialog to a Qt.Popup: on macOS Cocoa a popup window is
            # auto-dismissed (QEvent.Close -> reject()) the moment NSOpenPanel activates, which
            # immediately aborts the file dialog and closes the picker. On macOS (non-modal
            # QEventLoop wait) hide the popup cleanly before opening NSOpenPanel and re-show it
            # afterwards; on Windows/Linux (dialog.exec_()) keep it shown because hide() exits exec_().
            if popup_state.get("waits_non_modal") and hasattr(dialog, "hide"):
                try:
                    dialog.hide()
                    hid_for_browse = True
                except Exception:
                    hid_for_browse = False
            result = file_dialog_cls.getOpenFileName(file_dialog_parent, browse_title, start_dir, file_filter)
            if isinstance(result, (tuple, list)):
                chosen = str(result[0] or "")
            else:
                chosen = str(result or "")
        finally:
            if not popup_state.get("finished"):
                try:
                    if hid_for_browse or (hasattr(dialog, "isVisible") and not dialog.isVisible()):
                        dialog.show()
                        resize_popup()
                    if hasattr(dialog, "raise_"):
                        dialog.raise_()
                    if hasattr(dialog, "activateWindow"):
                        dialog.activateWindow()
                except Exception:
                    pass
            popup_state["browsing_file"] = False
        if chosen:
            path_edit.setText(chosen)
            if state["text"] != chosen.strip():
                _apply_topup_gate(chosen, auto_check=True)

    def _on_clear(*_args) -> None:
        path_edit.setText("")
        if state["text"] != "":
            _apply_topup_gate("", auto_check=True)

    text_sig = getattr(path_edit, "textChanged", None)
    if text_sig is not None and hasattr(text_sig, "connect"):
        text_sig.connect(lambda text: _apply_topup_gate(text, auto_check=True))
    elif hasattr(path_edit, "connect"):
        path_edit.connect("textChanged(QString)", lambda text: _apply_topup_gate(text, auto_check=True))

    browse_sig = getattr(browse_btn, "clicked", None)
    if browse_sig is not None and hasattr(browse_sig, "connect"):
        browse_sig.connect(_on_browse)
    elif hasattr(browse_btn, "connect"):
        browse_btn.connect("clicked()", _on_browse)

    clear_sig = getattr(clear_btn, "clicked", None)
    if clear_sig is not None and hasattr(clear_sig, "connect"):
        clear_sig.connect(_on_clear)
    elif hasattr(clear_btn, "connect"):
        clear_btn.connect("clicked()", _on_clear)

    staged["preprocessingReversePhaseB0"] = _read_text
    return container


def _add_batch_clear_reverse_b0_row(qt, owner, layout, initial_path, staged, zh, resize_popup, *, dark=None):
    """When a stale single-subject reverse-PE path lingers in Batch mode, offer a one-click Clear button."""
    from pathlib import Path
    from shared.ui_styles import is_dark_mode

    use_dark = is_dark_mode(getattr(owner, "_uiWidget", None)) if dark is None else bool(dark)
    state = {"path": str(initial_path or "").strip()}
    container = qt.QWidget()
    container.setObjectName("topupBatchClearReverseContainer")
    row = qt.QHBoxLayout(container)
    if hasattr(row, "setContentsMargins"):
        row.setContentsMargins(22, 0, 0, 2)
    if hasattr(row, "setSpacing"):
        row.setSpacing(6)
    ink = "#FF7B72" if use_dark else "#C62828"
    label = qt.QLabel(
        f"残留单受试者反向 b=0：{Path(state['path']).name}" if zh
        else f"Stale single-subject reverse-PE b=0: {Path(state['path']).name}"
    )
    label.setObjectName("topupBatchClearReverseLabel")
    label.setToolTip(state["path"])
    label.setStyleSheet(f"QLabel#topupBatchClearReverseLabel {{ color: {ink}; font-size: 11px; }}")
    clear_btn = qt.QPushButton("清除" if zh else "Clear")
    clear_btn.setObjectName("topupBatchClearReverseButton")
    pointing = getattr(qt.Qt, "PointingHandCursor", None)
    if pointing is not None and hasattr(clear_btn, "setCursor"):
        clear_btn.setCursor(pointing)

    def _clear(*_args) -> None:
        state["path"] = ""
        container.setVisible(False)
        resize_popup()

    sig = getattr(clear_btn, "clicked", None)
    if sig is not None and hasattr(sig, "connect"):
        sig.connect(_clear)
    elif hasattr(clear_btn, "connect"):
        clear_btn.connect("clicked()", _clear)
    row.addWidget(label, 1)
    row.addWidget(clear_btn)
    layout.addWidget(container)
    staged["preprocessingReversePhaseB0"] = lambda: state["path"]
    return container


def _install_outside_close_apply(qt, dialog, popup_state, apply_staged, changed=lambda: True, no_replay=False,
                                 on_hidden=None, dismiss_on_deactivate=True):
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
    ``on_hidden`` (macOS, review U1): called on a Hide event after which the dialog is no longer visible (a spontaneous
    Hide from the window system leaves it visible). The event is not consumed.
    Returns the filter; the caller must keep it alive. Returns None if the filter cannot be built.
    """

    close_type = getattr(getattr(qt, "QEvent", None), "Close", None)
    if close_type is None or not hasattr(dialog, "installEventFilter"):
        return None
    hide_type = getattr(getattr(qt, "QEvent", None), "Hide", None) if on_hidden is not None else None
    deactivate_type = getattr(getattr(qt, "QEvent", None), "ApplicationDeactivate", None)
    replay_attribute = getattr(getattr(qt, "Qt", None), "WA_NoMouseReplay", None) if no_replay else None
    try:
        base = getattr(qt, "QObject")

        class OutsideCloseApplyFilter(base):
            def eventFilter(self, _watched, event):
                try:
                    kind = event.type()
                    if popup_state.get("browsing_file"):
                        if kind == close_type:
                            if hasattr(event, "ignore"):
                                event.ignore()
                            return True
                        return False
                    if hide_type is not None and kind == hide_type:
                        if not dialog.isVisible():
                            on_hidden()
                        return False
                    if dismiss_on_deactivate and deactivate_type is not None and kind == deactivate_type:
                        if not popup_state["finished"]:
                            dialog.reject()
                        return False
                    if kind != close_type:
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


def _install_app_outside_click_close(qt, dialog, popup_state):
    """For non-Popup frameless pickers (e.g. Preprocessing with file drop), close on clicks elsewhere in Slicer.

    External app clicks (like switching to Finder/Explorer to drag a file) never send MouseButtonPress to Slicer,
    so the card stays open for drag-and-drop while clicks on Slicer's main window still trigger OutsideCloseApplyFilter.
    """
    press_type = getattr(getattr(qt, "QEvent", None), "MouseButtonPress", None)
    app_getter = getattr(getattr(qt, "QApplication", None), "instance", None)
    app = app_getter() if callable(app_getter) else None
    if press_type is None or app is None or not hasattr(app, "installEventFilter"):
        return None, None
    popup_flag = int(getattr(getattr(qt, "Qt", None), "Popup", 0) or 0)
    type_mask = int(getattr(getattr(qt, "Qt", None), "WindowType_Mask", 0xFF) or 0xFF)
    try:
        base = getattr(qt, "QObject")

        qwidget_cls = getattr(qt, "QWidget", None)
        check_widget_type = isinstance(qwidget_cls, type)

        class AppOutsideClickFilter(base):
            def eventFilter(self, watched, event):
                try:
                    if event.type() != press_type:
                        return False
                    if popup_state.get("finished") or popup_state.get("browsing_file"):
                        return False
                    if hasattr(dialog, "isVisible") and not dialog.isVisible():
                        return False
                    if check_widget_type and not isinstance(watched, qwidget_cls):
                        return False
                    if watched is dialog:
                        return False
                    if hasattr(dialog, "isAncestorOf"):
                        try:
                            if dialog.isAncestorOf(watched):
                                return False
                        except Exception:
                            pass
                    if hasattr(event, "globalPos") and hasattr(dialog, "frameGeometry"):
                        try:
                            fg = dialog.frameGeometry
                            fg_rect = fg() if callable(fg) else fg
                            if fg_rect is not None and fg_rect.contains(event.globalPos()):
                                return False
                        except Exception:
                            pass
                    active_popup_fn = getattr(getattr(qt, "QApplication", None), "activePopupWidget", None)
                    if callable(active_popup_fn) and active_popup_fn() is not None:
                        return False
                    win_fn = getattr(watched, "window", None)
                    if callable(win_fn):
                        win = win_fn()
                        if win is dialog:
                            return False
                        if win is not None and popup_flag and hasattr(win, "windowFlags"):
                            try:
                                if (int(win.windowFlags()) & type_mask) == popup_flag:
                                    return False
                            except Exception:
                                pass
                    dialog.close()
                    return True
                except Exception:
                    return False

        flt = AppOutsideClickFilter()
        app.installEventFilter(flt)
        return app, flt
    except Exception:
        return None, None


def _host_platform() -> str:
    return sys.platform


def _popup_waits_non_modal(qt, is_popup) -> bool:
    """macOS only: the picker popup is shown non-modal and a local event loop waits for it (batch 2, popup-mac)."""

    return bool(is_popup) and _host_platform() == "darwin" and getattr(qt, "QEventLoop", None) is not None


def _rejected(qt) -> int:
    value = getattr(getattr(qt, "QDialog", None), "Rejected", 0)
    return value if isinstance(value, int) else 0


def _dialog_hidden(dialog) -> bool:
    try:
        return not dialog.isVisible()
    except Exception:  # a destroyed dialog
        return True


def _run_popup(qt, dialog, is_popup, enders=None, popup_state=None):
    """Show the picker and wait for its result; returns the ``QDialog`` result code.

    Windows and Linux: ``exec_()``, as before. macOS: ``exec_()`` runs a native modal session, and Cocoa then drops
    every click on the main window (Qt's ``-[QNSPanel worksWhenModal]`` is NO for a parent of the modal window), so Qt
    never sees an outside click and the popup can neither apply nor close. There the popup is shown non-modal and a
    local ``QEventLoop`` waits. The loop ends when the dialog finishes (Apply, Cancel, X, Esc, an outside click, the
    watch timer, ``close_algorithm_picker``), when the dialog is destroyed, or when the application quits
    (``QCoreApplication::exit`` also ends every nested loop). A dialog that is already finished, or that is not
    visible after ``show()``, never enters the loop. A result that cannot be read counts as Rejected.

    Review U1: a hide that is not an accept emits no ``finished`` (``exec_()`` returned Rejected on it). The wait
    therefore also ends when the shown dialog is no longer visible, whatever the order of events: at once through
    ``enders`` (a list; the function that ends the wait is put in it while the loop runs, and the caller's Hide event
    filter calls it), and at the latest at the next tick of the re-quit timer. The result is read after the loop, so
    a hide inside ``accept()`` still returns Accepted and a bare hide returns Rejected.
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

    if isinstance(enders, list):
        enders.append(_end)

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
    # repeating timer therefore re-sends quit() from the loop's own level once the wait is over. It also ends the
    # wait once the shown dialog is no longer visible (review U1).
    requit = None
    if hasattr(qt, "QTimer"):
        try:
            requit = qt.QTimer()
            requit.setInterval(100)
            requit.timeout.connect(
                lambda: _end()
                if state["done"] or (not (popup_state and popup_state.get("browsing_file")) and _dialog_hidden(dialog))
                else None
            )
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
        if isinstance(enders, list) and _end in enders:
            enders.remove(_end)
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


def _owner_node_live(owner) -> bool:
    """Review U2: the owner is not torn down and its parameter node exists and is still in a scene.

    On macOS the picker waits non-modally, so File > Close Scene can run while it is open (``onSceneStartClose`` sets
    the node to None). A commit then must not run.
    """

    if getattr(owner, "_cleanedUp", False) is True:
        return False
    node = getattr(owner, "_parameterNode", None)
    if node is None:
        return False
    get_scene = getattr(getattr(node, "parameterNode", None), "GetScene", None)
    if callable(get_scene):
        try:
            return get_scene() is not None
        except Exception:
            return False
    return True


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
    # G3: where FSL is not offered, Eddy and Topup are disabled and unchecked; Brain mask and DWI denoising stay.
    fsl_missing = family == "preprocessing" and not _fsl_stages_offered()
    is_batch = family == "preprocessing" and _is_batch_mode(owner)
    supports_file_drop = family == "preprocessing" and not fsl_missing and not is_batch

    dialog = qt.QDialog(slicer.util.mainWindow())
    dialog.objectName = "algorithmPicker_" + family
    dialog.windowTitle = title_text
    popup_flag = getattr(qt.Qt, "Popup", 0)
    dialog_flag = getattr(qt.Qt, "Dialog", 0)
    frameless_flag = getattr(qt.Qt, "FramelessWindowHint", 0)
    if popup_flag and hasattr(dialog, "setWindowFlags"):
        # When the card embeds an external file drop zone (Preprocessing single-subject), use a frameless
        # Qt.Dialog instead of Qt.Popup so clicking Finder/Explorer to drag a file in (or opening a native
        # QFileDialog via Browse) never auto-dismisses the card. Clicks elsewhere in Slicer still dismiss/apply
        # via the app-level MouseButtonPress filter installed below.
        base_flag = (dialog_flag or popup_flag) if supports_file_drop else popup_flag
        try:
            dialog_flags = base_flag | frameless_flag
        except TypeError:
            dialog_flags = base_flag
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

    waits_non_modal = _popup_waits_non_modal(qt, popup_flag)
    popup_state = {
        "seen_visible": False,
        "finished": False,
        "browsing_file": False,
        "waits_non_modal": waits_non_modal,
    }
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
        available = None
        try:
            screen = qt.QApplication.screenAt(anchor.mapToGlobal(qt.QPoint(0, 0))) if anchor_visible else None
            if screen is None:
                screen = _qt_val(qt.QApplication, "primaryScreen")
            available = _qt_val(screen, "availableGeometry") if screen is not None else None
        except Exception:
            available = None
        available_height = _qt_int(available, "height", 800) if available is not None else 800
        available_width = _qt_int(available, "width", 800) if available is not None else 800
        body_layout.invalidate()
        content_size = body_layout.sizeHint()
        content_height = _qt_int(content_size, "height", 0)
        max_body_height = min(480, max(80, available_height - 108))
        scroll.setFixedHeight(min(content_height, max_body_height))
        scroll.setMinimumWidth(min(_qt_int(content_size, "width", 0) + 20,
                                   max(100, available_width - 40)))
        layout.activate()
        dialog.adjustSize()
        if anchor_visible:
            _position_input_help_card(anchor, dialog, qt)

    boxes = {}
    staged = {}
    shared_editors = {}
    pointing_cursor = getattr(qt.Qt, "PointingHandCursor", None)
    fsl_tip = ""
    if fsl_missing:
        from shared.selected_setup import fsl_unavailable_text

        fsl_tip = fsl_unavailable_text(zh=zh)
    topup_reverse_tip = ""
    if family == "preprocessing" and not fsl_missing:
        from shared.topup_parameters import topup_unavailable_reason

        initial_reverse = str(getattr(owner._parameterNode, "preprocessingReversePhaseB0", "") or "").strip()
        topup_reverse_tip = topup_unavailable_reason(initial_reverse, batch_mode=is_batch, zh=zh)
    # B2: algorithms that the chosen DATA cannot run. They stay in the stored selection (see Apply below).
    data_tips = {} if is_anatomical else _data_block_texts(owner, choices, zh)
    for value in choices:
        label = ANATOMICAL_PRIORS[value][int(zh)] if is_anatomical else LABELS[value]
        box = qt.QCheckBox(label)
        box.objectName = "algorithm_" + value
        fsl_blocked = fsl_missing and value in FSL_STAGES
        topup_blocked = family == "preprocessing" and value == "topup" and not fsl_blocked and bool(topup_reverse_tip)
        blocked_tip = fsl_tip if fsl_blocked else (topup_reverse_tip if topup_blocked else data_tips.get(value, ""))
        blocked = fsl_blocked or topup_blocked or value in data_tips
        box.checked = value in current and not blocked
        if blocked:
            box.setEnabled(False)
            box.setToolTip(blocked_tip)
        if pointing_cursor is not None and hasattr(box, "setCursor"):
            box.setCursor(pointing_cursor)
        boxes[value] = box
        inline_topup_picker = family == "preprocessing" and value == "topup" and not fsl_missing and not is_batch
        row_controls = _add_algorithm_row(
            qt, owner, body_layout, box, family, value, staged, zh, resize_popup,
            dark=dark, disabled=blocked, disabled_tooltip=blocked_tip,
            create_marker=inline_topup_picker, shared_editors=shared_editors,
        )
        if inline_topup_picker:
            _add_topup_reverse_b0_controls(
                qt, owner, dialog, body_layout, box, row_controls, staged, popup_state, zh, resize_popup, dark=dark,
            )
        elif family == "preprocessing" and value == "topup" and not fsl_missing and is_batch and initial_reverse:
            _add_batch_clear_reverse_b0_row(
                qt, owner, body_layout, initial_reverse, staged, zh, resize_popup, dark=dark,
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
            if "preprocessingReversePhaseB0" in staged:
                reverse_value = staged["preprocessingReversePhaseB0"]()
                if reverse_value:
                    from shared.topup_parameters import validate_reverse_phase_b0_path

                    _, reverse_error = validate_reverse_phase_b0_path(reverse_value, batch_mode=is_batch, zh=zh)
                    if reverse_error:
                        raise ValueError(reverse_error)
            if "orchestrationIfwEchoTimeMs" in staged:
                echo_time = staged["orchestrationIfwEchoTimeMs"]()
                if echo_time:
                    from math import isfinite
                    number = float(echo_time)
                    if not isfinite(number) or number <= 0:
                        raise ValueError("Echo time must be a positive number in milliseconds.")
            for param_attr in (
                "ukfParameters",
                "csdActParameters",
                "dtiParameters",
                "safParameters",
                "microstructureFitMethod",
                "freeWaterParameters",
                "ifwParameters",
                "tractcloudParameters",
            ):
                if param_attr in staged:
                    staged[param_attr]()
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

    # Review U1: on macOS a Hide event after which the popup is not visible ends the non-modal wait (see _run_popup).
    wait_enders = []

    def _popup_hidden() -> None:
        for end in list(wait_enders):
            end()

    outside_filter = (
        _install_outside_close_apply(
            qt,
            dialog,
            popup_state,
            validate_and_accept,
            popup_changed,
            no_replay=waits_non_modal,
            on_hidden=_popup_hidden if waits_non_modal else None,
            dismiss_on_deactivate=not supports_file_drop,
        )
        if popup_flag
        else None
    )
    app_click_target, app_click_filter = (
        _install_app_outside_click_close(qt, dialog, popup_state)
        if (popup_flag and supports_file_drop)
        else (None, None)
    )

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
                if popup_state.get("browsing_file"):
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
        accepted = _run_popup(qt, dialog, bool(popup_flag), wait_enders, popup_state=popup_state) == qt.QDialog.Accepted
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
        if app_click_target is not None and app_click_filter is not None:
            try:
                app_click_target.removeEventFilter(app_click_filter)
            except Exception:
                pass
    if accepted and not _owner_node_live(owner):
        accepted = False  # review U2: the scene closed (or the module was torn down) while the popup was open
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
            from shared.ui_tractography_params import set_node_parameter

            with slicer.util.NodeModify(owner._parameterNode):
                for attribute, value in staged.items():
                    set_node_parameter(owner._parameterNode, attribute, value())
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
