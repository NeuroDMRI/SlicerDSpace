"""Arrange existing, connected controls into a compact Slicer workflow."""

from __future__ import annotations

from time import monotonic
from typing import Any


_COMPACT_PRODUCT_TOOLTIP_KEYS = {
    "targetPreprocessingCheckBox": "preprocessing_tooltip",
    "targetMicrostructureCheckBox": "microstructure_tooltip",
    "targetTractographyCheckBox": "tractography_tooltip",
    "targetBundlesCheckBox": "bundles_tooltip",
    "targetConnectomeCheckBox": "connectome_tooltip",
}


def _qt_val(obj: Any, attr: str, default: Any = None) -> Any:
    val = getattr(obj, attr, default)
    return val() if callable(val) else val


def _qt_int(obj: Any, attr: str, default: int = 0) -> int:
    try:
        return int(_qt_val(obj, attr, default))
    except Exception:
        return default


def _connect_clicked(button: Any, callback: Any) -> None:
    clicked = getattr(button, "clicked", None)
    if clicked is not None and hasattr(clicked, "connect"):
        clicked.connect(callback)
        return
    if hasattr(button, "connect"):
        button.connect("clicked()", callback)


def compact_algorithm_labels(node: Any) -> dict[str, str]:
    """Describe the configured algorithms, not execution or QC status."""

    from shared.algorithm_selection import BUNDLES, CONNECTOME, LABELS, PREPROCESSING, algorithms
    from shared.ui_run_controller import microstructure_ifw_enabled, microstructure_optional_stage_flags
    from shared.user_config import RUN_DEFAULTS, selected_microstructure_stages

    def _str_attr(attr_name: str) -> str:
        val = getattr(node, attr_name, "")
        return val.strip() if isinstance(val, str) else ""

    preprocessing = (_str_attr("orchestrationPreprocessingStages") or "denoise,eddy,topup").lower()
    preproc_labels = " + ".join(LABELS[name] for name in algorithms(preprocessing, PREPROCESSING, "denoise,eddy,topup"))

    raw_micro = _str_attr("orchestrationMicrostructureStages")
    method = (_str_attr("orchestrationDkiFitMethod") or RUN_DEFAULTS["dki_fit_method"]).upper()
    dki_label = {"MKCURVE": "MK-Curve", "MKCURVE_MATLAB": "MK-Curve (MATLAB)"}.get(method, f"DKI ({method})")
    if raw_micro:
        chosen = set(selected_microstructure_stages(raw_micro))
        models = []
        if "dti" in chosen:
            models.append("DTI")
        if "freewater" in chosen:
            models.append("FW")
        if "noddi" in chosen:
            models.append("NODDI")
        if "dki" in chosen:
            models.append(dki_label)
        if "ifw" in chosen:
            models.append("iFW")
        if not models:
            models.append("DTI")
    else:
        freewater, noddi, dki = microstructure_optional_stage_flags(
            "",
            legacy_freewater=bool(getattr(node, "orchestrationRunFreeWater", False)),
            legacy_noddi=bool(getattr(node, "orchestrationRunNoddi", False)),
            legacy_dki=bool(getattr(node, "orchestrationRunDki", False)),
        )
        models = ["DTI"]
        if freewater:
            models.append("FW")
        if noddi:
            models.append("NODDI")
        if dki:
            models.append(dki_label)
        if microstructure_ifw_enabled("", legacy_ifw=bool(getattr(node, "orchestrationRunIfw", False))):
            models.append("iFW")
    tractography = (_str_attr("orchestrationTractographyBackend")
                    or RUN_DEFAULTS["tractography_backend"]).lower()
    bundles = (_str_attr("orchestrationBundleBackend")
               or RUN_DEFAULTS["bundle_backend"]).lower()
    connectome = (_str_attr("orchestrationConnectomeMetrics") or "sift2").lower()
    connectome_labels = " + ".join(LABELS[name] for name in algorithms(connectome, CONNECTOME, "sift2"))
    return {
        "targetPreprocessingCheckBox": preproc_labels,
        "targetMicrostructureCheckBox": " + ".join(models),
        "targetTractographyCheckBox": {
            "csd_act": "MRtrix3 CSD / ACT", "ukf": "UKF",
            "dti_det": "DTI", "saf": "CSD + SAF",
        }.get(tractography, " + ".join(LABELS[name] for name in algorithms(tractography))),
        "targetBundlesCheckBox": {
            "wmql": "WMQL", "wma": "WMA", "tractcloud": "TractCloud",
            "all": "WMQL + WMA + TractCloud",
        }.get(bundles, " + ".join(LABELS[name] for name in algorithms(bundles, BUNDLES, "wmql"))),
        "targetConnectomeCheckBox": connectome_labels,
    }


def refresh_compact_algorithms(owner: Any, language: str | None = None) -> None:
    """Refresh the product rows after parameter or language changes."""

    from shared.ui_onboarding import onboarding_text

    ui = getattr(owner, "ui", None)
    if ui is None:
        return
    raw_lang = language or getattr(owner, "_onboardingLanguage", "en")
    language = raw_lang if isinstance(raw_lang, str) else "en"
    summary_map = getattr(owner, "_algorithmSummaryLabels", None)
    if not isinstance(summary_map, dict):
        summary_map = {}
    for name, algorithms in compact_algorithm_labels(getattr(owner, "_parameterNode", None)).items():
        control = getattr(ui, name, None)
        if control is not None:
            text = onboarding_text(language, "compact_algorithm_selection").format(algorithms=algorithms)
            selection_line = onboarding_text(language, "compact_algorithm_tooltip").format(algorithms=algorithms)
            desc_key = _COMPACT_PRODUCT_TOOLTIP_KEYS.get(name)
            desc = onboarding_text(language, desc_key) if desc_key else ""
            tooltip = f"{desc}\n{selection_line}" if desc else selection_line
            summary = summary_map.get(name)
            if summary is None:
                if hasattr(control, "setText") and getattr(control, "text", None) != text:
                    control.setText(text)
            else:
                if hasattr(control, "setText") and getattr(control, "text", None) != "":
                    control.setText("")
                if hasattr(control, "setAccessibleName") and getattr(control, "accessibleName", None) != text:
                    control.setAccessibleName(text)
            if hasattr(control, "setToolTip") and getattr(control, "toolTip", None) != tooltip:
                control.setToolTip(tooltip)
            if summary is not None:
                if getattr(summary, "text", None) != algorithms:
                    summary.setText(algorithms)
                if getattr(summary, "toolTip", None) != tooltip:
                    summary.setToolTip(tooltip)
    from shared.ui_algorithm_picker import refresh_picker_language
    refresh_picker_language(owner)


def _apply_input_path_mode(owner: Any, fmt: str, language: str | None = None) -> None:
    """Adjust the input path placeholder and browser filter for the chosen format."""

    from shared.ui_onboarding import onboarding_text

    ui = getattr(owner, "ui", None)
    if ui is None:
        return
    lang = language or getattr(owner, "_onboardingLanguage", "en")
    normalized = (fmt or "nifti").strip().lower()
    if normalized not in {"nifti", "dicom", "nrrd"}:
        normalized = "nifti"
    placeholder = onboarding_text(lang, f"input_format_placeholder_{normalized}")
    path_widget = getattr(ui, "inputDirectoryPath", None)
    if path_widget is None:
        return
    try:
        import ctk
        import qt

        if normalized == "nrrd" and hasattr(ctk, "ctkPathLineEdit"):
            files_flag = getattr(ctk.ctkPathLineEdit, "Files", 0)
            dirs_flag = getattr(ctk.ctkPathLineEdit, "Dirs", 0)
            path_widget.filters = (
                (files_flag | dirs_flag)
                if isinstance(files_flag, int) and isinstance(dirs_flag, int)
                else files_flag
            )
            path_widget.nameFilters = []
        elif hasattr(ctk, "ctkPathLineEdit"):
            path_widget.filters = ctk.ctkPathLineEdit.Dirs
            path_widget.nameFilters = []
        if hasattr(path_widget, "findChildren"):
            for edit in path_widget.findChildren(qt.QLineEdit):
                edit.setPlaceholderText(placeholder)
    except Exception:
        pass
    if hasattr(path_widget, "setPlaceholderText"):
        path_widget.setPlaceholderText(placeholder)


def sync_detected_input_format(
    owner: Any,
    mode: str | None,
    *,
    signature: tuple[str, str] | None = None,
    force: bool = False,
) -> str | None:
    """Sync ``Input format`` to the detected layout while preserving manual browse changes."""

    from shared.pipeline import detected_input_format_from_mode

    detected = detected_input_format_from_mode(mode)
    if detected is None:
        return None
    if not force and signature is not None:
        if getattr(owner, "_lastAutoDetectedInputSignature", None) == signature:
            return detected
    if signature is not None:
        owner._lastAutoDetectedInputSignature = signature

    node = getattr(owner, "_parameterNode", None)
    current = (str(getattr(node, "inputFormat", "") or "") if node is not None else "").strip().lower()
    owner._syncingDetectedInputFormat = True
    try:
        if node is not None and current != detected:
            update_fn = getattr(owner, "_update_parameter_node", None)
            if callable(update_fn):
                update_fn("inputFormat", detected)
            else:
                node.inputFormat = detected
        combo = getattr(owner, "_inputFormatComboBox", None) or getattr(
            getattr(owner, "ui", None), "inputFormatComboBox", None
        )
        if combo is not None and hasattr(combo, "findData") and hasattr(combo, "setCurrentIndex"):
            target_idx = combo.findData(detected)
            if isinstance(target_idx, int) and target_idx >= 0:
                was_blocked = combo.blockSignals(True) if hasattr(combo, "blockSignals") else False
                try:
                    combo.setCurrentIndex(target_idx)
                finally:
                    if hasattr(combo, "blockSignals"):
                        combo.blockSignals(bool(was_blocked))
        _apply_input_path_mode(owner, detected)
    finally:
        owner._syncingDetectedInputFormat = False
    return detected


def _position_input_help_card(anchor: Any, card: Any, qt_mod: Any) -> None:
    """Position ``card`` directly below ``anchor`` (right-aligned inside the module dock, screen-clamped)."""

    try:
        card_layout = _qt_val(card, "layout")
        if card_layout is not None and hasattr(card_layout, "activate"):
            card_layout.activate()
    except Exception:
        pass
    btn_w = _qt_int(anchor, "width", 22)
    btn_h = _qt_int(anchor, "height", 22)
    hint = _qt_val(card, "sizeHint")
    card_w = _qt_int(hint, "width", 0) or _qt_int(card, "width", 364) or 364
    card_h = _qt_int(hint, "height", 0) or _qt_int(card, "height", 180) or 180
    try:
        if hasattr(card, "resize") and _qt_int(hint, "width", 0) > 0 and _qt_int(hint, "height", 0) > 0:
            card.resize(card_w, card_h)
    except Exception:
        pass
    anchor_br = anchor.mapToGlobal(qt_mod.QPoint(btn_w, btn_h + 6))
    bx = _qt_int(anchor_br, "x", anchor_br[0] if isinstance(anchor_br, tuple) else 0)
    by = _qt_int(anchor_br, "y", anchor_br[1] if isinstance(anchor_br, tuple) else 0)
    x = bx - card_w
    y = by
    try:
        app_cls = getattr(qt_mod, "QApplication", None)
        screen = app_cls.screenAt(anchor_br) if app_cls is not None and hasattr(app_cls, "screenAt") else None
        if screen is None and app_cls is not None and hasattr(app_cls, "primaryScreen"):
            screen = app_cls.primaryScreen()
        if screen is not None:
            avail = _qt_val(screen, "availableGeometry")
            left = _qt_int(avail, "left", 0)
            top = _qt_int(avail, "top", 0)
            right = _qt_int(avail, "right", left + 1920)
            bottom = _qt_int(avail, "bottom", top + 1080)
            x = max(left + 8, min(x, right - card_w - 8))
            if y + card_h > bottom - 8:
                above_pt = anchor.mapToGlobal(qt_mod.QPoint(btn_w, -6))
                above_y = _qt_int(above_pt, "y", by) - card_h
                if above_y >= top + 8:
                    y = above_y
    except Exception:
        pass
    if hasattr(card, "move"):
        card.move(x, y)


def refresh_input_format_help_popover_style(owner: Any, *, dark: bool | None = None) -> None:
    """Apply active Light/Dark stylesheet to the Input format '?' hover and pinned popovers."""

    from shared.ui_styles import input_format_popover_style, set_stylesheet_if_changed

    root = getattr(owner, "_uiWidget", None)
    style = input_format_popover_style(dark=dark, widget=root)
    for attr in ("_inputFormatHoverPopover", "_inputFormatPinnedPopover"):
        card = getattr(owner, attr, None)
        if card is not None:
            set_stylesheet_if_changed(card, style)


def close_input_format_help_popovers(owner: Any, *, destroy: bool = False) -> None:
    """Hide (and optionally tear down) the Input format '?' popovers."""

    if destroy:
        timer = getattr(owner, "_inputFormatHelpPollTimer", None)
        if timer is not None:
            try:
                if hasattr(timer, "stop"):
                    timer.stop()
                poll_cb = getattr(owner, "_inputFormatHelpPollCallback", None)
                if poll_cb is not None:
                    timeout_sig = getattr(timer, "timeout", None)
                    if timeout_sig is not None and hasattr(timeout_sig, "disconnect"):
                        timeout_sig.disconnect(poll_cb)
                    elif hasattr(timer, "disconnect"):
                        timer.disconnect("timeout()", poll_cb)
            except Exception:
                pass
            owner._inputFormatHelpPollTimer = None
            owner._inputFormatHelpPollCallback = None
    state = getattr(owner, "_inputFormatHelpState", None)
    if isinstance(state, dict):
        state["pinned"] = False
    for attr in ("_inputFormatHoverPopover", "_inputFormatPinnedPopover"):
        card = getattr(owner, attr, None)
        if card is None:
            continue
        try:
            card.hide()
        except Exception:
            pass
        if destroy:
            try:
                if hasattr(card, "close"):
                    card.close()
            except Exception:
                pass
            setattr(owner, attr, None)


def _sync_input_help_card_texts(owner: Any, language: str | None = None) -> None:
    from shared.ui_onboarding import onboarding_text

    lang = language or getattr(owner, "_onboardingLanguage", "en")
    title_text = onboarding_text(lang, "input_format_help_title")
    body_text = onboarding_text(lang, "input_format_help")
    close_tip = onboarding_text(lang, "input_format_help_close")
    for attr, value in (
        ("_inputFormatHoverBodyLabel", body_text),
        ("_inputFormatPinnedTitleLabel", title_text),
        ("_inputFormatPinnedBodyLabel", body_text),
    ):
        lbl = getattr(owner, attr, None)
        if lbl is not None and hasattr(lbl, "setText"):
            lbl.setText(value)
    close_btn = getattr(owner, "_inputFormatHelpCloseButton", None)
    if close_btn is not None:
        if hasattr(close_btn, "setToolTip"):
            close_btn.setToolTip(close_tip)
        if hasattr(close_btn, "setAccessibleName"):
            close_btn.setAccessibleName(close_tip)


def install_input_format_selector(owner: Any) -> None:
    """Insert the input-format combo box and hover + click-to-pin '?' button at the top of 1. Inputs."""

    if getattr(owner, "_inputFormatComboBox", None) is not None:
        return
    import qt

    ui = owner.ui
    form = ui.configurationFormLayout
    label = qt.QLabel()
    label.setObjectName("inputFormatLabel")
    container = qt.QWidget()
    row_layout = qt.QHBoxLayout(container)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(6)
    combo = qt.QComboBox()
    combo.setObjectName("inputFormatComboBox")
    for text, value in (("NIfTI", "nifti"), ("DICOM", "dicom"), ("NRRD", "nrrd")):
        combo.addItem(text, value)
    combo.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed)
    help_btn = qt.QToolButton()
    help_btn.setObjectName("inputFormatHelpButton")
    help_btn.setText("?")
    help_btn.setFixedSize(22, 22)
    help_btn.setCursor(getattr(qt.Qt, "PointingHandCursor", qt.Qt.WhatsThisCursor))
    help_btn.setFocusPolicy(qt.Qt.NoFocus)

    state: dict[str, Any] = {"pinned": False, "closed_at": 0.0}
    owner._inputFormatHelpState = state
    hover_card = None
    pinned_card = None

    def _set_pinned_state(is_pinned: bool) -> None:
        state["pinned"] = bool(is_pinned)
        try:
            if hasattr(help_btn, "setProperty"):
                help_btn.setProperty("pinned", "true" if is_pinned else "false")
            btn_style = _qt_val(help_btn, "style")
            if btn_style is not None:
                btn_style.unpolish(help_btn)
                btn_style.polish(help_btn)
        except Exception:
            pass

    if hasattr(qt, "QFrame") and hasattr(qt, "QVBoxLayout"):
        frameless = getattr(qt.Qt, "FramelessWindowHint", 0)
        tooltip_flag = getattr(qt.Qt, "ToolTip", 0)
        popup_flag = getattr(qt.Qt, "Popup", 0)

        hover_card = qt.QFrame()
        hover_card.setObjectName("inputFormatHelpPopover")
        if hasattr(hover_card, "setWindowFlags"):
            hover_card.setWindowFlags(tooltip_flag | frameless)
        for attr_name in ("WA_ShowWithoutActivating", "WA_TransparentForMouseEvents"):
            attr_flag = getattr(qt.Qt, attr_name, None)
            if attr_flag is not None and hasattr(hover_card, "setAttribute"):
                hover_card.setAttribute(attr_flag, True)
        hover_layout = qt.QVBoxLayout(hover_card)
        hover_layout.setContentsMargins(12, 10, 12, 10)
        hover_layout.setSpacing(4)
        hover_body = qt.QLabel()
        hover_body.setObjectName("inputFormatHelpBody")
        if hasattr(hover_body, "setWordWrap"):
            hover_body.setWordWrap(True)
        else:
            hover_body.wordWrap = True
        if hasattr(hover_body, "setFixedWidth"):
            hover_body.setFixedWidth(340)
        hover_layout.addWidget(hover_body)

        pinned_card = qt.QFrame()
        pinned_card.setObjectName("inputFormatHelpPopover")
        if hasattr(pinned_card, "setWindowFlags"):
            pinned_card.setWindowFlags(popup_flag | frameless)
        pinned_layout = qt.QVBoxLayout(pinned_card)
        pinned_layout.setContentsMargins(12, 8, 10, 12)
        pinned_layout.setSpacing(6)
        header_row = qt.QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(6)
        pinned_title = qt.QLabel()
        pinned_title.setObjectName("inputFormatHelpTitle")
        close_btn = qt.QToolButton()
        close_btn.setObjectName("inputFormatHelpCloseButton")
        close_btn.setText("×")
        close_btn.setFixedSize(20, 20)
        close_btn.setCursor(getattr(qt.Qt, "PointingHandCursor", qt.Qt.WhatsThisCursor))
        close_btn.setFocusPolicy(qt.Qt.NoFocus)
        header_row.addWidget(pinned_title)
        header_row.addStretch(1)
        header_row.addWidget(close_btn)
        if hasattr(pinned_layout, "addLayout"):
            pinned_layout.addLayout(header_row)
        else:
            pinned_layout.addWidget(pinned_title)
            pinned_layout.addWidget(close_btn)
        pinned_body = qt.QLabel()
        pinned_body.setObjectName("inputFormatHelpBody")
        if hasattr(pinned_body, "setWordWrap"):
            pinned_body.setWordWrap(True)
        else:
            pinned_body.wordWrap = True
        if hasattr(pinned_body, "setFixedWidth"):
            pinned_body.setFixedWidth(340)
        selectable = getattr(qt.Qt, "TextSelectableByMouse", None)
        if selectable is not None and hasattr(pinned_body, "setTextInteractionFlags"):
            pinned_body.setTextInteractionFlags(selectable)
        pinned_layout.addWidget(pinned_body)

        def _close_pinned_popover(*_args: Any) -> None:
            if hasattr(pinned_card, "hide"):
                pinned_card.hide()
            _set_pinned_state(False)
            state["closed_at"] = 0.0

        _connect_clicked(close_btn, _close_pinned_popover)

        hide_event_type = getattr(qt.QEvent, "Hide", 18)
        close_event_type = getattr(qt.QEvent, "Close", 19)

        class _PinnedCardController:
            def eventFilter(self, _watched, event):
                if event.type() in (hide_event_type, close_event_type):
                    if state["pinned"]:
                        state["closed_at"] = monotonic()
                        _set_pinned_state(False)
                return False

        pinned_filter = _PinnedCardController()
        if not hasattr(qt, "QTimer") and hasattr(pinned_card, "installEventFilter"):
            pinned_card.installEventFilter(pinned_filter)
        owner._inputFormatPinnedFilter = pinned_filter
        owner._inputFormatHoverPopover = hover_card
        owner._inputFormatHoverBodyLabel = hover_body
        owner._inputFormatPinnedPopover = pinned_card
        owner._inputFormatPinnedTitleLabel = pinned_title
        owner._inputFormatPinnedBodyLabel = pinned_body
        owner._inputFormatHelpCloseButton = close_btn
        ui.inputFormatHelpCloseButton = close_btn
        refresh_input_format_help_popover_style(owner)

    def _show_hover_popover(anchor_widget: Any = None) -> None:
        target_btn = anchor_widget or help_btn
        if hover_card is not None:
            if state["pinned"]:
                return
            if hasattr(qt, "QToolTip") and hasattr(qt.QToolTip, "hideText"):
                qt.QToolTip.hideText()
            refresh_input_format_help_popover_style(owner)
            _sync_input_help_card_texts(owner)
            _position_input_help_card(target_btn, hover_card, qt)
            hover_card.show()
            return
        tip = _qt_val(target_btn, "toolTip", "")
        if tip and hasattr(qt, "QToolTip") and hasattr(qt.QToolTip, "showText"):
            pos = target_btn.mapToGlobal(qt.QPoint(0, _qt_int(target_btn, "height", 22) + 4))
            qt.QToolTip.showText(pos, str(tip), target_btn)

    def _hide_hover_popover() -> None:
        if hover_card is not None:
            hover_card.hide()
            return
        if hasattr(qt, "QToolTip") and hasattr(qt.QToolTip, "hideText"):
            qt.QToolTip.hideText()

    def _on_help_button_clicked(*_args: Any) -> None:
        if hover_card is not None and hasattr(hover_card, "hide"):
            hover_card.hide()
        if hasattr(qt, "QToolTip") and hasattr(qt.QToolTip, "hideText"):
            qt.QToolTip.hideText()
        if pinned_card is None:
            return
        if state["pinned"] or (monotonic() - float(state.get("closed_at") or 0.0) < 0.22):
            pinned_card.hide()
            _set_pinned_state(False)
            state["closed_at"] = 0.0
            return
        refresh_input_format_help_popover_style(owner)
        _sync_input_help_card_texts(owner)
        _set_pinned_state(True)
        _position_input_help_card(help_btn, pinned_card, qt)
        pinned_card.show()
        if hasattr(pinned_card, "raise_"):
            pinned_card.raise_()

    _connect_clicked(help_btn, _on_help_button_clicked)

    enter_type = getattr(qt.QEvent, "Enter", 10)
    leave_type = getattr(qt.QEvent, "Leave", 11)
    hide_type = getattr(qt.QEvent, "Hide", 18)
    tooltip_event_type = getattr(qt.QEvent, "ToolTip", 110)

    class _HoverHelpController:
        def eventFilter(self, watched, event):
            etype = event.type()
            if hover_card is not None and etype == tooltip_event_type:
                return True
            if etype == enter_type:
                state["manual_hover"] = True
                _show_hover_popover(watched)
            elif etype in (leave_type, hide_type):
                state["manual_hover"] = False
                _hide_hover_popover()
            return False

    hover_filter = _HoverHelpController()
    if not hasattr(qt, "QTimer") and hasattr(help_btn, "installEventFilter"):
        help_btn.installEventFilter(hover_filter)
    owner._inputFormatHelpFilter = hover_filter
    owner._showInputFormatHoverHelp = _show_hover_popover
    owner._hideInputFormatHoverHelp = _hide_hover_popover

    if hasattr(qt, "QTimer") and hover_card is not None:
        try:
            poll_timer = qt.QTimer()
            if hasattr(poll_timer, "setInterval"):
                poll_timer.setInterval(80)

            def _poll_input_help() -> None:
                if getattr(owner, "_cleanedUp", False):
                    return
                if state["pinned"] and pinned_card is not None and not bool(_qt_val(pinned_card, "visible", False)):
                    state["closed_at"] = monotonic()
                    _set_pinned_state(False)
                btn_vis = bool(_qt_val(help_btn, "visible", False))
                hovered = btn_vis and bool(_qt_val(help_btn, "underMouse", False))
                if hovered:
                    state["mouse_hovered"] = True
                    if not state["pinned"] and not bool(_qt_val(hover_card, "visible", False)):
                        _show_hover_popover(help_btn)
                    elif hasattr(qt, "QToolTip") and hasattr(qt.QToolTip, "hideText"):
                        qt.QToolTip.hideText()
                elif state.get("mouse_hovered"):
                    state["mouse_hovered"] = False
                    if not state.get("manual_hover") and bool(_qt_val(hover_card, "visible", False)):
                        _hide_hover_popover()

            timeout_sig = getattr(poll_timer, "timeout", None)
            if timeout_sig is not None and hasattr(timeout_sig, "connect"):
                timeout_sig.connect(_poll_input_help)
            elif hasattr(poll_timer, "connect"):
                poll_timer.connect("timeout()", _poll_input_help)
            if hasattr(poll_timer, "start"):
                poll_timer.start()
            owner._inputFormatHelpPollTimer = poll_timer
            owner._inputFormatHelpPollCallback = _poll_input_help
        except Exception:
            pass

    def on_format_changed(index: int) -> None:
        value = str(combo.itemData(index) or "nifti")
        update_fn = getattr(owner, "_update_parameter_node", None)
        if callable(update_fn):
            update_fn("inputFormat", value)
        elif getattr(owner, "_parameterNode", None) is not None:
            owner._parameterNode.inputFormat = value
        _apply_input_path_mode(owner, value)
        save_fn = getattr(owner, "_schedule_common_user_settings_save", None)
        if callable(save_fn):
            save_fn()
        check_fn = getattr(owner, "_checkCanApply", None)
        if callable(check_fn):
            check_fn()

    combo.currentIndexChanged.connect(on_format_changed)
    row_layout.addWidget(combo, 1)
    row_layout.addWidget(help_btn, 0, qt.Qt.AlignVCenter)
    form.insertRow(0, label, container)
    ui.inputFormatLabel = label
    ui.inputFormatComboBox = combo
    ui.inputFormatHelpButton = help_btn
    owner._inputFormatLabel = label
    owner._inputFormatComboBox = combo
    owner._inputFormatHelpButton = help_btn
    refresh_input_format_selector(owner)


def refresh_input_format_selector(owner: Any, language: str | None = None) -> None:
    """Refresh labels and hover/pinned popover text for the input format selector."""

    from shared.ui_onboarding import onboarding_text

    lang = language or getattr(owner, "_onboardingLanguage", "en")
    label = getattr(owner, "_inputFormatLabel", None) or getattr(getattr(owner, "ui", None), "inputFormatLabel", None)
    combo = getattr(owner, "_inputFormatComboBox", None) or getattr(getattr(owner, "ui", None), "inputFormatComboBox", None)
    help_btn = getattr(owner, "_inputFormatHelpButton", None) or getattr(getattr(owner, "ui", None), "inputFormatHelpButton", None)
    if label is not None:
        label.setText(onboarding_text(lang, "input_format_label"))
    help_text = onboarding_text(lang, "input_format_help")
    if help_btn is not None:
        has_custom_popover = getattr(owner, "_inputFormatHoverPopover", None) is not None
        help_btn.setToolTip("" if has_custom_popover else help_text)
        if hasattr(help_btn, "setAccessibleName"):
            help_btn.setAccessibleName(onboarding_text(lang, "input_format_label"))
    _sync_input_help_card_texts(owner, lang)
    fmt = "nifti"
    node = getattr(owner, "_parameterNode", None)
    if node is not None and getattr(node, "inputFormat", ""):
        fmt = str(node.inputFormat)
    elif combo is not None and hasattr(combo, "currentIndex") and hasattr(combo, "itemData"):
        idx = combo.currentIndex() if callable(combo.currentIndex) else combo.currentIndex
        fmt = str(combo.itemData(idx) or "nifti")
    if combo is not None and hasattr(combo, "setItemText"):
        for index, key in enumerate(("input_format_nifti", "input_format_dicom", "input_format_nrrd")):
            combo.setItemText(index, onboarding_text(lang, key))
    if combo is not None and hasattr(combo, "findData") and hasattr(combo, "setCurrentIndex"):
        target_idx = combo.findData(fmt)
        if target_idx >= 0:
            was_blocked = combo.blockSignals(True) if hasattr(combo, "blockSignals") else False
            try:
                combo.setCurrentIndex(target_idx)
            finally:
                if hasattr(combo, "blockSignals"):
                    combo.blockSignals(bool(was_blocked))
    _apply_input_path_mode(owner, fmt, lang)


_DETECTED_CUDA_DEVICES_CACHE: tuple[tuple[str, str], ...] | None = None


def _detect_cuda_devices() -> tuple[tuple[str, str], ...]:
    """Return ``(canonical_device, display_label)`` entries for CUDA GPUs without slow imports."""

    global _DETECTED_CUDA_DEVICES_CACHE
    if _DETECTED_CUDA_DEVICES_CACHE is not None:
        return _DETECTED_CUDA_DEVICES_CACHE

    import os
    import shutil
    import subprocess
    import sys

    entries: list[tuple[str, str]] = []
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None:
        try:
            cuda_mod = getattr(torch_mod, "cuda", None)
            if cuda_mod is not None and bool(cuda_mod.is_available()):
                count = min(8, int(cuda_mod.device_count()))
                for idx in range(count):
                    gpu_name = str(cuda_mod.get_device_name(idx) or "").strip()
                    short_name = gpu_name.removeprefix("NVIDIA ").removeprefix("GeForce ").strip()
                    label = f"CUDA:{idx} ({short_name})" if short_name else f"CUDA:{idx}"
                    entries.append((f"cuda:{idx}", label))
        except Exception:
            entries.clear()

    if not entries:
        smi = shutil.which("nvidia-smi")
        if not smi and sys.platform == "win32":
            cand = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe")
            if os.path.isfile(cand):
                smi = cand
        if smi:
            try:
                kwargs: dict[str, Any] = {"capture_output": True, "text": True, "timeout": 1.5}
                if sys.platform == "win32":
                    kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                proc = subprocess.run([smi, "--query-gpu=index,name", "--format=csv,noheader"], **kwargs)
                if proc.returncode == 0:
                    for raw_line in (proc.stdout or "").splitlines():
                        parts = [p.strip() for p in raw_line.split(",", 1)]
                        if not parts or not parts[0].isdigit():
                            continue
                        idx = int(parts[0])
                        if 0 <= idx < 8:
                            gpu_name = parts[1] if len(parts) > 1 else ""
                            short_name = gpu_name.removeprefix("NVIDIA ").removeprefix("GeForce ").strip()
                            label = f"CUDA:{idx} ({short_name})" if short_name else f"CUDA:{idx}"
                            entries.append((f"cuda:{idx}", label))
            except Exception:
                entries.clear()

    if not entries:
        entries = [("cuda:0", "CUDA:0"), ("cuda:1", "CUDA:1")]
    _DETECTED_CUDA_DEVICES_CACHE = tuple(entries)
    return _DETECTED_CUDA_DEVICES_CACHE


def install_compute_device_selector(owner: Any) -> None:
    """Place the Compute device (Auto / CPU / CUDA:N) selector in 2. Products above Run SlicerDSpace."""

    if getattr(owner, "_computeDeviceRow", None) is not None:
        return
    import qt

    ui = getattr(owner, "ui", None)
    if ui is None:
        return
    label = getattr(ui, "unifiedTorchDeviceLabel", None)
    combo = getattr(ui, "unifiedTorchDeviceComboBox", None)
    targets_layout = getattr(ui, "targetsLayout", None)
    if label is None or combo is None or targets_layout is None:
        return

    env_form = getattr(ui, "environmentFormLayout", None)
    if env_form is not None:
        for item in (label, combo):
            idx = env_form.indexOf(item)
            if idx >= 0:
                env_form.takeAt(idx)

    row = qt.QWidget()
    row.setObjectName("computeDeviceRow")
    row_layout = qt.QHBoxLayout(row)
    row_layout.setContentsMargins(0, 3, 0, 0)
    row_layout.setSpacing(10)
    combo.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed)
    if hasattr(qt, "QComboBox") and hasattr(combo, "setSizeAdjustPolicy"):
        combo.setSizeAdjustPolicy(qt.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        combo.setMinimumContentsLength(10)
    pointing = getattr(qt.Qt, "PointingHandCursor", None)
    if pointing is not None and hasattr(combo, "setCursor"):
        combo.setCursor(pointing)
    label.show()
    combo.show()
    row_layout.addWidget(label, 0, qt.Qt.AlignVCenter)
    row_layout.addWidget(combo, 1)
    targets_layout.addWidget(row)
    owner._computeDeviceRow = row
    refresh_compute_device_selector(owner)


def refresh_compute_device_selector(owner: Any, language: str | None = None) -> None:
    """Populate and localize the Compute device combo box while preserving canonical device data."""

    from shared.ui_onboarding import onboarding_text

    ui = getattr(owner, "ui", None)
    if ui is None:
        return
    label = getattr(ui, "unifiedTorchDeviceLabel", None)
    combo = getattr(ui, "unifiedTorchDeviceComboBox", None)
    if combo is None:
        return
    lang = language or getattr(owner, "_onboardingLanguage", "en")
    tip = onboarding_text(lang, "compute_device_tooltip")
    if label is not None:
        label.setText(onboarding_text(lang, "torch_device"))
        if hasattr(label, "setToolTip"):
            label.setToolTip(tip)
    if hasattr(combo, "setToolTip"):
        combo.setToolTip(tip)
    if not (hasattr(combo, "clear") and hasattr(combo, "addItem")):
        return

    node = getattr(owner, "_parameterNode", None)
    current = "auto"
    if node is not None:
        current = str(getattr(node, "device", "") or getattr(node, "ddsegDevice", "") or "auto").strip().lower() or "auto"
    elif hasattr(combo, "currentIndex") and hasattr(combo, "itemData"):
        idx = _qt_int(combo, "currentIndex", 0)
        current = str(combo.itemData(idx) or _qt_val(combo, "currentText", "auto") or "auto").strip().lower() or "auto"

    items: list[tuple[str, str]] = [
        ("auto", onboarding_text(lang, "compute_device_auto")),
        ("cpu", "CPU"),
        *_detect_cuda_devices(),
    ]
    seen_values = {val for val, _ in items}
    if current not in seen_values and current.startswith("cuda:"):
        items.append((current, current.upper()))

    was_blocked = combo.blockSignals(True) if hasattr(combo, "blockSignals") else False
    try:
        combo.clear()
        for value, display in items:
            combo.addItem(display, value)
        target_idx = combo.findData(current) if hasattr(combo, "findData") else -1
        if target_idx < 0:
            target_idx = 0
        if hasattr(combo, "setCurrentIndex"):
            combo.setCurrentIndex(target_idx)
    finally:
        if hasattr(combo, "blockSignals"):
            combo.blockSignals(bool(was_blocked))


def organize_environment_section(owner: Any) -> None:
    """Arrange Environment and installation cleanly without empty QFormLayout columns."""

    if getattr(owner, "_environmentSectionOrganized", False):
        return
    import qt

    ui = getattr(owner, "ui", None)
    env_layout = getattr(ui, "environmentLayout", None) if ui is not None else None
    env_form = getattr(ui, "environmentFormLayout", None) if ui is not None else None
    if env_layout is None:
        return

    for hidden_name in ("environmentHintLabel", "autoInstallPythonDependenciesLabel"):
        lbl = getattr(ui, hidden_name, None)
        if lbl is not None:
            lbl.hide()
            if env_form is not None and env_form.indexOf(lbl) >= 0:
                env_form.takeAt(env_form.indexOf(lbl))

    result_label = getattr(ui, "environmentResultLabel", None)
    setup_btn = getattr(owner, "_managedRuntimeSetupButton", None)
    validate_btn = getattr(ui, "validateButton", None)
    lite_btn = getattr(owner, "_nativeDependencyRouteButton", None)
    auto_cb = getattr(ui, "autoInstallPythonDependenciesCheckBox", None)
    more_panel = getattr(owner, "_environmentMoreOptions", None)

    for widget in (validate_btn, lite_btn, auto_cb):
        if widget is not None and env_form is not None and env_form.indexOf(widget) >= 0:
            env_form.takeAt(env_form.indexOf(widget))
    if env_form is not None:
        env_form.setContentsMargins(0, 0, 0, 0)
        env_form.setSpacing(0)

    for widget in (result_label, setup_btn, validate_btn, lite_btn, auto_cb, more_panel):
        if widget is not None and env_layout.indexOf(widget) >= 0:
            env_layout.removeWidget(widget)

    env_layout.setContentsMargins(8, 6, 8, 8)
    env_layout.setSpacing(6)
    pointing = getattr(qt.Qt, "PointingHandCursor", None)

    if result_label is not None:
        result_label.show()
        env_layout.addWidget(result_label)
    if setup_btn is not None:
        setup_btn.setMinimumWidth(0)
        if pointing is not None and hasattr(setup_btn, "setCursor"):
            setup_btn.setCursor(pointing)
        setup_btn.show()
        env_layout.addWidget(setup_btn)

    for btn in (validate_btn, lite_btn):
        if btn is not None:
            btn.setMinimumWidth(0)
            btn.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
            if pointing is not None and hasattr(btn, "setCursor"):
                btn.setCursor(pointing)
            btn.show()
            env_layout.addWidget(btn)

    if auto_cb is not None:
        if pointing is not None and hasattr(auto_cb, "setCursor"):
            auto_cb.setCursor(pointing)
        auto_cb.show()
        env_layout.addWidget(auto_cb)

    if more_panel is not None:
        more_panel.show()
        env_layout.addWidget(more_panel)

    owner._environmentSectionOrganized = True


def install_compact_layout(owner: Any) -> None:
    """Move secondary controls, not their state or signals, into Advanced."""

    if getattr(owner, "_advancedSettingsPanel", None) is not None:
        return
    import ctk
    import qt

    for attr in ("reloadCollapsibleButton", "helpCollapsibleButton"):
        section = getattr(owner, attr, None)
        if section is not None:
            section.hide()

    ui = owner.ui
    root = owner._uiWidget
    root.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Maximum)
    root.setMinimumWidth(0)
    parent_widget = getattr(owner, "parent", None)
    if parent_widget is not None and hasattr(parent_widget, "setMinimumWidth"):
        parent_widget.setMinimumWidth(0)
    # Content stays at the top through trailing stretches, not setAlignment(AlignTop): an aligned
    # layout keeps its size hint as height, which leaves out wrapped lines (height-for-width), so
    # rows were clipped in narrow docks.
    owner.layout.addStretch(1)
    owner.layout.setSizeConstraint(qt.QLayout.SetMinAndMaxSize)
    layout = root.layout()
    layout.setContentsMargins(4, 4, 4, 6)
    layout.setSpacing(5)
    layout.setSizeConstraint(qt.QLayout.SetMinAndMaxSize)
    ui.headerFrame.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Maximum)
    ui.headerFrame.setMinimumWidth(0)
    ui.headerLayout.setContentsMargins(0, 2, 0, 5)
    ui.headerTextLayout.setContentsMargins(0, 0, 0, 0)
    from shared.ui_branding import install_header_wordmark
    install_header_wordmark(owner)
    ui.subtitleLabel.hide()

    # Reuse the selector so persistence and all translations still work.
    language = owner._onboardingWidgets["language_combo"]
    language.setObjectName("interfaceLanguageComboBox")
    language.setSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Fixed)
    ui.headerLayout.addWidget(language)
    language.show()
    ui.gettingStartedCollapsibleButton.hide()

    advanced = ctk.ctkCollapsibleButton(root)
    advanced.setObjectName("advancedSettingsPanel")
    advanced.collapsed = True
    advanced.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)
    advanced.setMinimumWidth(0)
    advanced_layout = qt.QVBoxLayout(advanced)
    advanced_layout.setContentsMargins(8, 8, 8, 8)
    advanced_layout.setSpacing(8)
    advanced_hint = qt.QLabel(advanced)
    advanced_hint.setObjectName("advancedSettingsHintLabel")
    advanced_hint.setWordWrap(True)
    advanced_layout.addWidget(advanced_hint)
    owner._advancedSettingsHintLabel = advanced_hint
    prepare_inputs = owner._prepareInputWidgets["panel"]
    # CTK's old parent keeps a visibility event filter after reparenting.
    prepare_inputs.removeEventFilter(ui.configurationCollapsibleButton)
    ui.configurationLayout.removeWidget(prepare_inputs)
    advanced_layout.addWidget(prepare_inputs)
    prepare_inputs.show()
    prepare_inputs.collapsed = True
    prepare_inputs.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)
    prepare_inputs.setMinimumWidth(0)
    for name in ("parametersCollapsibleButton", "resultsCollapsibleButton"):
        section = getattr(ui, name)
        layout.removeWidget(section)
        advanced_layout.addWidget(section)
        section.collapsed = True
        section.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)
        section.setMinimumWidth(0)

    utilities = qt.QHBoxLayout()
    for name in ("diagnostic_button", "config_button"):
        button = owner._onboardingWidgets[name]
        button.setMinimumWidth(0)
        utilities.addWidget(button)
        button.show()
    advanced_layout.addLayout(utilities)
    layout.insertWidget(layout.indexOf(ui.logCollapsibleButton), advanced)
    owner._advancedSettingsPanel = advanced
    # Dependency preparation is an independent, first-class setup section.
    environment = ui.environmentCollapsibleButton
    layout.removeWidget(environment)
    layout.insertWidget(layout.indexOf(advanced) + 1, environment)
    environment.collapsed = True
    environment.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)
    environment.setMinimumWidth(0)
    from shared.ui_algorithm_picker import install_algorithm_pickers
    install_input_format_selector(owner)
    install_algorithm_pickers(owner)
    install_compute_device_selector(owner)

    layout.removeWidget(ui.statusFrame)
    ui.statusLayout.setContentsMargins(0, 0, 0, 0)
    ui.runLayout.addWidget(ui.statusFrame)
    ui.runLayout.setContentsMargins(0, 0, 0, 0)
    ui.runLayout.setSpacing(4)
    ui.runCollapsibleButton.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)
    ui.runCollapsibleButton.setMinimumWidth(0)
    ui.logLayout.setContentsMargins(0, 0, 0, 0)
    ui.logLayout.setSpacing(4)
    ui.logPlainTextEdit.setMinimumHeight(64)
    ui.logPlainTextEdit.setMaximumHeight(84)
    ui.logCollapsibleButton.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Maximum)
    ui.logCollapsibleButton.setMinimumWidth(0)
    owner._prepareInputWidgets["panel"].setStyleSheet("")
    for section in (ui.configurationCollapsibleButton, ui.targetsCollapsibleButton):
        # Preferred, not Maximum: Maximum caps a section at its size hint, which leaves out wrapped
        # form rows; the trailing stretch below keeps sections from growing beyond their content.
        section.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)
        section.setMinimumWidth(0)
        section.layout().setContentsMargins(4, 6, 4, 6)
        section.layout().setSpacing(4)
    ui.configurationFormLayout.setVerticalSpacing(2)
    ui.configurationFormLayout.setHorizontalSpacing(8)
    ui.targetProductsFormLayout.setVerticalSpacing(3)
    ui.targetProductsFormLayout.setHorizontalSpacing(6)
    owner._runStateLabel.hide()
    owner._runStatusHintLabel.hide()
    owner._runProgressBar.hide()
    if getattr(owner, "_runProgressRowWidget", None) is not None:
        owner._runProgressRowWidget.hide()
    owner._cancelRunButton.hide()
    owner._selectedProductsSummaryLabel.hide()
    layout.addStretch(1)
    owner._fitHeightQueued = False

    def _configure_parent_scroll_area() -> None:
        curr = getattr(owner, "parent", None) or root.parentWidget()
        while curr is not None:
            if curr.inherits("QScrollArea"):
                curr.setVerticalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOn)
                curr.setHorizontalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOff)
                break
            curr = curr.parentWidget()

    _configure_parent_scroll_area()

    def fit_height(*_args: Any) -> None:
        # Slicer's scroll area keeps an old expanded height unless resized explicitly.
        # Only adjust height (never width) and include height-for-width so root.resize matches
        # owner.layout's exact height calculation on the first pass without a 2-step vertical bounce.
        owner._fitHeightQueued = False
        try:
            _configure_parent_scroll_area()
            if root.width <= 0:
                return
            children = [c for c in root.findChildren(qt.QWidget) if c.isVisibleTo(root) and c.layout() is not None]
            sync_stack = getattr(owner, "_syncAdvancedStackHeight", None)
            for _ in range(3):
                # Top-down: propagate container widths/geometries down to newly shown child widgets.
                layout.invalidate()
                layout.activate()
                for child in children:
                    child.layout().invalidate()
                    child.layout().activate()
                if callable(sync_stack):
                    sync_stack()
                # Bottom-up: propagate updated height-for-width from leaf layouts up to root.
                for child in reversed(children):
                    child.layout().invalidate()
                    child.layout().activate()
                if callable(sync_stack):
                    sync_stack()
                layout.invalidate()
                layout.activate()
                hfw = layout.totalHeightForWidth(root.width) if layout.hasHeightForWidth() else 0
                target_height = max(root.minimumSizeHint.height(), hfw)
                if root.height != target_height:
                    root.resize(root.width, target_height)
            if getattr(owner, "layout", None) is not None:
                owner.layout.invalidate()
                owner.layout.activate()
        except ValueError:
            return

    def queue_fit(*_args: Any) -> None:
        if getattr(owner, "_fitHeightQueued", False):
            return
        owner._fitHeightQueued = True
        qt.QTimer.singleShot(0, fit_height)

    def on_section_toggled(*_args: Any) -> None:
        fit_height()
        queue_fit()

    owner._fitCompactHeight = fit_height
    owner._queueCompactFit = queue_fit

    sections = [
        advanced,
        ui.configurationCollapsibleButton,
        ui.targetsCollapsibleButton,
        ui.parametersCollapsibleButton,
        ui.environmentCollapsibleButton,
        ui.resultsCollapsibleButton,
        ui.advancedRunCollapsibleButton,
        owner._prepareInputWidgets["panel"],
        getattr(owner, "_environmentMoreOptions", None),
        getattr(owner, "_environmentAdvancedPathsWidget", None),
    ]
    for section in sections:
        if section is not None and hasattr(section, "connect"):
            section.connect("contentsCollapsed(bool)", on_section_toggled)
    queue_fit()


def apply_compact_language(owner: Any, language: str) -> None:
    """Keep concise main-screen labels in sync with the language system."""

    if getattr(owner, "_advancedSettingsPanel", None) is None:
        return
    from shared.ui_onboarding import onboarding_text

    owner._advancedSettingsPanel.setText(onboarding_text(language, "compact_advanced"))
    owner._advancedSettingsPanel.setToolTip(onboarding_text(language, "compact_advanced_hint"))
    advanced_hint = getattr(owner, "_advancedSettingsHintLabel", None)
    if advanced_hint is not None:
        advanced_hint.setText(onboarding_text(language, "compact_advanced_hint"))
    owner.ui.headlineLabel.setText("SlicerDSpace")
    owner._onboardingWidgets["language_combo"].setToolTip(onboarding_text(language, "language"))
    prepare_inputs = (getattr(owner, "_prepareInputWidgets", {}) or {}).get("panel")
    if prepare_inputs is not None:
        prepare_inputs.setText(onboarding_text(language, "compact_import"))
    for name, key in {
        "configurationCollapsibleButton": "section_inputs",
        "targetsCollapsibleButton": "section_targets",
        "parametersCollapsibleButton": "compact_parameters",
        "environmentCollapsibleButton": "compact_environment",
        "resultsCollapsibleButton": "compact_results",
        "runSelectedButton": "compact_run",
        "logHintLabel": "compact_log",
        "targetPreprocessingCheckBox": "compact_preprocessing",
        "targetSurfaceCheckBox": "compact_surface",
        "targetSurfacePostprocessCheckBox": "compact_postprocess",
        "targetParcellationCheckBox": "compact_parcellation",
        "targetSegmentationCheckBox": "compact_segmentation",
        "targetMicrostructureCheckBox": "compact_microstructure",
        "targetTractographyCheckBox": "compact_tractography",
        "targetBundlesCheckBox": "compact_bundles",
        "targetConnectomeCheckBox": "compact_connectome",
    }.items():
        getattr(owner.ui, name).setText(onboarding_text(language, key))
    refresh_input_format_selector(owner, language)
    refresh_compute_device_selector(owner, language)
    refresh_compact_algorithms(owner, language)
    from shared.ui_run_history import refresh_run_progress_styles
    refresh_run_progress_styles(owner)
    from shared.ui_advanced_layout import apply_advanced_language
    apply_advanced_language(owner)
