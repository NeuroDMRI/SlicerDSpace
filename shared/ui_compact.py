"""Arrange existing, connected controls into a compact Slicer workflow."""

from __future__ import annotations

import logging
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


def microstructure_models(node: Any) -> list[tuple[str, str]]:
    """The stored Microstructure selection as ``(algorithm id, label)`` pairs, in display order."""

    from shared.ui_run_controller import microstructure_ifw_enabled, microstructure_optional_stage_flags
    from shared.user_config import RUN_DEFAULTS, selected_microstructure_stages

    def _str_attr(attr_name: str) -> str:
        val = getattr(node, attr_name, "")
        return val.strip() if isinstance(val, str) else ""

    raw_micro = _str_attr("orchestrationMicrostructureStages")
    method = (_str_attr("orchestrationDkiFitMethod") or RUN_DEFAULTS["dki_fit_method"]).upper()
    dki_label = {"MKCURVE": "MK-Curve", "MKCURVE_MATLAB": "MK-Curve (MATLAB)"}.get(method, f"DKI ({method})")
    if raw_micro:
        chosen = set(selected_microstructure_stages(raw_micro))
        models = []
        if "dti" in chosen:
            models.append(("dti", "DTI"))
        if "freewater" in chosen:
            models.append(("freewater", "FW"))
        if "noddi" in chosen:
            models.append(("noddi", "NODDI"))
        if "dki" in chosen:
            models.append(("dki", dki_label))
        if "ifw" in chosen:
            models.append(("ifw", "iFW"))
        if not models:
            models.append(("dti", "DTI"))
    else:
        freewater, noddi, dki = microstructure_optional_stage_flags(
            "",
            legacy_freewater=bool(getattr(node, "orchestrationRunFreeWater", False)),
            legacy_noddi=bool(getattr(node, "orchestrationRunNoddi", False)),
            legacy_dki=bool(getattr(node, "orchestrationRunDki", False)),
        )
        models = [("dti", "DTI")]
        if freewater:
            models.append(("freewater", "FW"))
        if noddi:
            models.append(("noddi", "NODDI"))
        if dki:
            models.append(("dki", dki_label))
        if microstructure_ifw_enabled("", legacy_ifw=bool(getattr(node, "orchestrationRunIfw", False))):
            models.append(("ifw", "iFW"))
    return models


def compact_algorithm_labels(node: Any, blocked: Any = ()) -> dict[str, str]:
    """Describe the configured algorithms, not execution or QC status.

    ``blocked`` names algorithms that the data cannot run (B3). They are left out of the Microstructure text, unless
    that would leave nothing: then the stored text stays and the caller adds the "not available" note.
    """

    from shared.algorithm_selection import BUNDLES, CONNECTOME, LABELS, PREPROCESSING, algorithms
    from shared.user_config import RUN_DEFAULTS

    def _str_attr(attr_name: str) -> str:
        val = getattr(node, attr_name, "")
        return val.strip() if isinstance(val, str) else ""

    preprocessing = (_str_attr("orchestrationPreprocessingStages") or "denoise,eddy").lower()
    preproc_selected = algorithms(preprocessing, PREPROCESSING, "denoise,eddy")
    preproc_visible = [name for name in preproc_selected if name not in blocked] or list(preproc_selected)
    preproc_labels = " + ".join(LABELS[name] for name in preproc_visible)

    stored = microstructure_models(node)
    models = [label for algorithm, label in stored if algorithm not in blocked] or [label for _, label in stored]
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


def _summary_data_blocks(owner: Any) -> dict:
    """Algorithms that the current data cannot run (B3); ``{}`` when unknown or on any failure."""
    try:
        from shared.algorithm_availability import current_data_blocks

        return current_data_blocks(owner)
    except Exception:
        return {}


def _mute_summary(owner: Any, name: str, summary: Any, muted: bool) -> None:
    """Grey a summary label only while every stored algorithm of its family is blocked; undo it afterwards."""
    muted_names = getattr(owner, "_algorithmSummaryMuted", None)
    if not isinstance(muted_names, set):
        muted_names = set()
        try:
            owner._algorithmSummaryMuted = muted_names
        except Exception:
            return
    if muted == (name in muted_names) or not hasattr(summary, "setEnabled"):
        return
    summary.setEnabled(not muted)
    muted_names.add(name) if muted else muted_names.discard(name)


def surface_postprocess_block_tooltip(owner: Any, zh: bool = False) -> str:
    """b2-os, decision 6: the reason and the actions when this OS cannot run Surface postprocess, else "".

    The stored tick stays (the Run skips the step and logs it); the tooltip says why and what fixes it.
    """
    try:
        from shared.surface_runtime import surface_postprocess_os_skip

        getter = getattr(owner, "_environment_dependency_path_kwargs", None)
        state = surface_postprocess_os_skip(getter() if callable(getter) else {})
    except Exception:
        return ""
    if not state:
        return ""
    reason = state.get("reason_zh" if zh else "reason") or state.get("reason") or ""
    hint = state.get("hint_zh" if zh else "hint") or ""
    return "\n".join(text for text in (reason, hint) if text)


def _refresh_surface_postprocess_tooltip(owner: Any, ui: Any, zh: bool) -> None:
    control = getattr(ui, "targetSurfacePostprocessCheckBox", None)
    if control is None or not hasattr(control, "setToolTip"):
        return
    from shared.ui_onboarding import onboarding_text

    block = surface_postprocess_block_tooltip(owner, zh=zh)
    # b2-os F3: no stored tooltip; without a block the row shows the onboarding text of the UI language, as the
    # language binding (MAIN_UI_TOOLTIP_BINDINGS) does, so a language switch keeps the translation.
    tooltip = block or onboarding_text("zh" if zh else "en", "surface_postprocess_tooltip")
    if getattr(control, "toolTip", None) != tooltip:
        control.setToolTip(tooltip)
    # Decision 6: grey the row label (not the checkbox, so the stored tick can still be removed); undo it afterwards.
    label = getattr(owner, "_surfacePostprocessLabel", None)
    if label is not None:
        _mute_summary(owner, "targetSurfacePostprocessCheckBox", label, bool(block))


def refresh_compact_algorithms(owner: Any, language: str | None = None) -> None:
    """Refresh the product rows after parameter or language changes."""

    from shared.ui_algorithm_picker import FAMILIES, FAMILY_TARGET_ATTRS
    from shared.ui_onboarding import onboarding_text

    ui = getattr(owner, "ui", None)
    if ui is None:
        return
    raw_lang = language or getattr(owner, "_onboardingLanguage", "en")
    language = raw_lang if isinstance(raw_lang, str) else "en"
    zh = language.startswith("zh")
    summary_map = getattr(owner, "_algorithmSummaryLabels", None)
    if not isinstance(summary_map, dict):
        summary_map = {}
    node = getattr(owner, "_parameterNode", None)
    cleared = getattr(owner, "_clearedAlgorithmFamilies", None)
    box_to_family = {box_name: family for family, (box_name, _, _) in FAMILIES.items()}
    blocks = _summary_data_blocks(owner)
    from shared.topup_parameters import topup_unavailable_reason

    batch_check = getattr(owner, "_input_preview_is_batch", None)
    if callable(batch_check):
        is_batch = bool(batch_check())
    else:
        cohort = bool(
            node is not None
            and getattr(node, "experimentsEnabled", False) is True
            and getattr(node, "experimentScope", "") == "cohort"
        )
        is_batch = cohort or bool(node is not None and getattr(node, "batchMode", False) is True)
    reverse_b0_raw = getattr(node, "preprocessingReversePhaseB0", "") if node is not None else ""
    topup_reason = topup_unavailable_reason(
        reverse_b0_raw if isinstance(reverse_b0_raw, str) else "",
        batch_mode=is_batch,
        zh=zh,
    )
    effective_blocked = set(blocks)
    if topup_reason:
        effective_blocked.add("topup")
    for name, algorithms in compact_algorithm_labels(node, blocked=effective_blocked).items():
        family = box_to_family.get(name)
        none_selected = False
        if family and isinstance(cleared, set) and family in cleared:
            if node is not None and bool(getattr(node, FAMILY_TARGET_ATTRS[family], False)):
                cleared.discard(family)
            else:
                algorithms = "未选择" if zh else "None selected"
                none_selected = True
        # B3: what the data cannot run is skipped at Run; say so here without touching the stored selection.
        skipped_lines, muted = [], False
        if family == "preprocessing" and topup_reason and not none_selected:
            from shared.algorithm_selection import PREPROCESSING, algorithms as _parse_preproc

            raw_preproc = getattr(node, "orchestrationPreprocessingStages", "") if node is not None else ""
            stored_preproc = _parse_preproc(
                raw_preproc if isinstance(raw_preproc, str) else "", PREPROCESSING, "denoise,eddy"
            )
            if stored_preproc == ("topup",):
                muted = True
                algorithms = "Topup · 需要反向相位 b=0 图像" if zh else "Topup · needs reverse-PE b=0"
                skipped_lines = [f"Topup: {topup_reason}"]
        if family == "microstructure" and blocks and not none_selected:
            stored = microstructure_models(node)
            skipped = [(algorithm, label) for algorithm, label in stored if algorithm in blocks]
            skipped_lines = [f"{label}: {blocks[algorithm].text(language)}" for algorithm, label in skipped]
            if skipped and len(skipped) == len(stored):
                muted = True
                algorithms = " + ".join(label for _, label in skipped) + (
                    " · 当前数据不可用" if zh else " · not available for this data")
        control = getattr(ui, name, None)
        if control is not None:
            text = onboarding_text(language, "compact_algorithm_selection").format(algorithms=algorithms)
            selection_line = onboarding_text(language, "compact_algorithm_tooltip").format(algorithms=algorithms)
            desc_key = _COMPACT_PRODUCT_TOOLTIP_KEYS.get(name)
            desc = onboarding_text(language, desc_key) if desc_key else ""
            if name == "targetPreprocessingCheckBox":
                # G-H1: where Eddy and Topup are unavailable, the reason stays in the tooltip after every refresh.
                from shared.selected_setup import fsl_unavailable_text
                from shared.ui_parameter_sync import fsl_stages_offered

                if not fsl_stages_offered():
                    reason = fsl_unavailable_text(zh=zh)
                    desc = f"{desc}\n{reason}" if desc else reason
            tooltip = f"{desc}\n{selection_line}" if desc else selection_line
            if skipped_lines:
                tooltip = tooltip + "\n" + "\n".join(skipped_lines)
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
                _mute_summary(owner, name, summary, muted)
    _refresh_surface_postprocess_tooltip(owner, ui, zh)
    from shared.ui_algorithm_picker import refresh_picker_language
    refresh_picker_language(owner)
    # FC3: the "Also runs automatically" note follows the same data as the summary (a tile edit, the Refresh click).
    refresh_note = getattr(owner, "_refresh_also_runs_note", None)
    if callable(refresh_note):
        try:
            refresh_note()
        except Exception:
            pass


def _apply_input_path_mode(owner: Any, fmt: str, language: str | None = None) -> None:
    """Adjust the input path placeholder and browser filter for the chosen format."""

    from shared.ui_onboarding import onboarding_text

    # The compact panel gives each file format its own controls. The shared
    # folder field stays a directory picker even when a folder contains NRRD.
    if "nifti_source" in getattr(owner, "_prepareInputWidgets", {}):
        return
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


SCREEN_FRAME_MARGIN = 40  # logical px kept free for a dialog's title bar and window frame


def available_screen_size(qt_mod: Any, anchor: Any = None) -> "tuple[int, int] | None":
    """Width and height of the available geometry of the screen under ``anchor`` (else the primary screen).

    The same screen lookup as the help popups. ``None`` when the screen is unknown.
    """

    try:
        app_cls = getattr(qt_mod, "QApplication", None)
        screen = None
        if anchor is not None and app_cls is not None and hasattr(app_cls, "screenAt"):
            screen = app_cls.screenAt(anchor.mapToGlobal(qt_mod.QPoint(0, 0)))
        if screen is None and app_cls is not None and hasattr(app_cls, "primaryScreen"):
            screen = app_cls.primaryScreen()
        if screen is None:
            return None
        avail = _qt_val(screen, "availableGeometry")
        width = _qt_int(avail, "width", 0)
        height = _qt_int(avail, "height", 0)
        return (width, height) if width > 0 and height > 0 else None
    except Exception:
        return None


def fit_size_to_screen(qt_mod: Any, width: int, height: int, anchor: Any = None) -> "tuple[int, int]":
    """``width`` x ``height`` reduced to fit the available screen minus the window frame; unchanged on a big screen."""

    size = available_screen_size(qt_mod, anchor)
    if size is None:
        return width, height
    return min(width, max(1, size[0] - SCREEN_FRAME_MARGIN)), min(height, max(1, size[1] - SCREEN_FRAME_MARGIN))


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
            y = max(top + 8, min(y, bottom - card_h - 8))
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


_DETECTED_MPS_DEVICES_CACHE: tuple[tuple[str, str], ...] | None = None
_DETECTED_CUDA_DEVICES_CACHE: tuple[tuple[str, str], ...] | None = None


def _detect_mps_devices() -> tuple[tuple[str, str], ...]:
    """Return ``(('mps', 'MPS (...)'),)`` on Apple Silicon Macs without importing torch."""

    global _DETECTED_MPS_DEVICES_CACHE
    if _DETECTED_MPS_DEVICES_CACHE is not None:
        return _DETECTED_MPS_DEVICES_CACHE

    import platform
    import subprocess
    import sys

    if sys.platform != "darwin":
        _DETECTED_MPS_DEVICES_CACHE = ()
        return _DETECTED_MPS_DEVICES_CACHE

    available = False
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None:
        try:
            mps_mod = getattr(getattr(torch_mod, "backends", None), "mps", None)
            fn = getattr(mps_mod, "is_available", None)
            if callable(fn):
                available = bool(fn() is True)
        except Exception:
            available = False
    if not available:
        if platform.machine() == "arm64":
            available = True
        else:
            try:
                proc = subprocess.run(
                    ["sysctl", "-n", "hw.optional.arm64"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=1.5,
                )
                available = proc.returncode == 0 and proc.stdout.strip() == "1"
            except Exception:
                available = False

    if not available:
        _DETECTED_MPS_DEVICES_CACHE = ()
        return _DETECTED_MPS_DEVICES_CACHE

    chip = ""
    try:
        proc = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1.5,
        )
        if proc.returncode == 0:
            chip = proc.stdout.strip()
    except Exception:
        chip = ""
    label = f"MPS ({chip})" if chip.startswith("Apple") else "MPS (Apple Metal)"
    _DETECTED_MPS_DEVICES_CACHE = (("mps", label),)
    return _DETECTED_MPS_DEVICES_CACHE


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
        if not smi and not hasattr(shutil.which, "assert_called"):
            from shared.hardware_adapt import find_nvidia_smi

            smi = find_nvidia_smi()
        if smi:
            try:
                kwargs: dict[str, Any] = {
                    "capture_output": True,
                    "text": True,
                    "encoding": "utf-8",
                    "errors": "replace",
                    "timeout": 1.5,
                }
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

    _DETECTED_CUDA_DEVICES_CACHE = tuple(entries)
    return _DETECTED_CUDA_DEVICES_CACHE


def _cuda_pytorch_runtime_available() -> bool:
    """Zero-import check whether Slicer's active Python has a CUDA-enabled PyTorch build."""
    try:
        from shared.ui_cuda_setup import cuda_pytorch_runtime_available

        return bool(cuda_pytorch_runtime_available())
    except Exception:
        return False


def _cuda_help_button_style(dark: bool, *, restart_pending: bool = False) -> str:
    """Style for the circular ``?`` CUDA setup button next to Compute device."""
    if restart_pending:
        ink, hover = ("#66BB6A", "#81C784") if dark else ("#2E7D32", "#1B5E20")
    else:
        ink, hover = ("#F59E0B", "#FBBF24") if dark else ("#C27803", "#92400E")
    name = "computeDeviceCudaHelpButton"
    return (
        f"QToolButton#{name} {{ color: {ink}; border: 1px solid {ink}; border-radius: 9px; "
        "font-size: 11px; font-weight: 700; background: transparent; padding: 0px; }"
        f"QToolButton#{name}:hover {{ color: {hover}; border-color: {hover}; }}"
        f"QToolButton#{name}:focus {{ color: {hover}; border: 2px solid {hover}; }}"
    )


def refresh_cuda_help_button_style(owner: Any, *, dark: bool | None = None) -> None:
    """Refresh the light/dark stylesheet of the Compute device CUDA ``?`` button."""
    button = getattr(owner, "_computeDeviceCudaHelpButton", None)
    if button is None or not hasattr(button, "setStyleSheet"):
        return
    if dark is None:
        try:
            from shared.ui_styles import is_dark_mode

            use_dark = bool(is_dark_mode(getattr(owner, "_uiWidget", None)))
        except Exception:
            use_dark = False
    else:
        use_dark = bool(dark)
    restart_pending = bool(getattr(owner, "_cudaInstallRestartPending", False))
    button.setStyleSheet(_cuda_help_button_style(use_dark, restart_pending=restart_pending))


def _on_compute_device_cuda_help_clicked(owner: Any) -> None:
    from shared.ui_cuda_setup import show_cuda_pytorch_setup_dialog

    show_cuda_pytorch_setup_dialog(owner)


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
    row_layout.setSpacing(6)
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

    if hasattr(qt, "QToolButton"):
        cuda_help_btn = qt.QToolButton(row)
        cuda_help_btn.setObjectName("computeDeviceCudaHelpButton")
        cuda_help_btn.setText("?")
        if hasattr(cuda_help_btn, "setAutoRaise"):
            cuda_help_btn.setAutoRaise(True)
        if hasattr(cuda_help_btn, "setFixedSize"):
            cuda_help_btn.setFixedSize(18, 18)
        if pointing is not None and hasattr(cuda_help_btn, "setCursor"):
            cuda_help_btn.setCursor(pointing)
        strong_focus = getattr(qt.Qt, "StrongFocus", None)
        if strong_focus is not None and hasattr(cuda_help_btn, "setFocusPolicy"):
            cuda_help_btn.setFocusPolicy(strong_focus)
        _connect_clicked(cuda_help_btn, lambda _checked=False: _on_compute_device_cuda_help_clicked(owner))
        if hasattr(cuda_help_btn, "hide"):
            cuda_help_btn.hide()
        row_layout.addWidget(cuda_help_btn, 0, qt.Qt.AlignVCenter)
        owner._computeDeviceCudaHelpButton = cuda_help_btn
        try:
            setattr(ui, "computeDeviceCudaHelpButton", cuda_help_btn)
        except Exception:
            pass
        refresh_cuda_help_button_style(owner)

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

    cuda_devices = _detect_cuda_devices()
    restart_pending = bool(getattr(owner, "_cudaInstallRestartPending", False))
    cuda_ready = _cuda_pytorch_runtime_available() if cuda_devices else True
    cuda_needs_setup = bool(cuda_devices) and (not cuda_ready or restart_pending)

    suffix = ""
    if cuda_needs_setup:
        suffix_key = "cuda_device_restart_pending_suffix" if restart_pending else "cuda_device_not_installed_suffix"
        suffix = onboarding_text(lang, suffix_key)

    cuda_items = [(val, f"{disp}{suffix}") for val, disp in cuda_devices]

    items: list[tuple[str, str]] = [
        ("auto", onboarding_text(lang, "compute_device_auto")),
        ("cpu", "CPU"),
        *_detect_mps_devices(),
        *cuda_items,
    ]
    seen_values = {val for val, _ in items}
    if current == "mps" and current not in seen_values:
        items.append(("mps", "MPS"))
        seen_values.add("mps")
    if current not in seen_values and current.startswith("cuda:"):
        items.append((current, f"{current.upper()}{suffix}" if cuda_needs_setup else current.upper()))

    was_blocked = combo.blockSignals(True) if hasattr(combo, "blockSignals") else False
    try:
        combo.clear()
        for value, display in items:
            combo.addItem(display, value)

        disabled_indices: set[int] = set()
        if cuda_needs_setup and hasattr(combo, "model"):
            model = combo.model()
            item_getter = getattr(model, "item", None) if model is not None else None
            if callable(item_getter):
                for idx, (value, _display) in enumerate(items):
                    if value.startswith("cuda:"):
                        qitem = item_getter(idx)
                        if qitem is not None and hasattr(qitem, "setEnabled"):
                            qitem.setEnabled(False)
                            disabled_indices.add(idx)

        target_idx = combo.findData(current) if hasattr(combo, "findData") else -1
        if target_idx < 0 or target_idx in disabled_indices:
            if target_idx in disabled_indices and node is not None:
                if str(getattr(node, "device", "") or "").strip().lower().startswith("cuda"):
                    node.device = "auto"
                if str(getattr(node, "ddsegDevice", "") or "").strip().lower().startswith("cuda"):
                    node.ddsegDevice = "auto"
            target_idx = 0
        if hasattr(combo, "setCurrentIndex"):
            combo.setCurrentIndex(target_idx)
    finally:
        if hasattr(combo, "blockSignals"):
            combo.blockSignals(bool(was_blocked))

    cuda_help_btn = getattr(owner, "_computeDeviceCudaHelpButton", None)
    if cuda_help_btn is not None:
        if cuda_needs_setup:
            help_tip_key = "cuda_setup_restart_tooltip" if restart_pending else "cuda_setup_help_tooltip"
            if hasattr(cuda_help_btn, "setToolTip"):
                cuda_help_btn.setToolTip(onboarding_text(lang, help_tip_key))
            if hasattr(cuda_help_btn, "setAccessibleName"):
                cuda_help_btn.setAccessibleName(onboarding_text(lang, "cuda_setup_help_name"))
            refresh_cuda_help_button_style(owner)
            if hasattr(cuda_help_btn, "show"):
                cuda_help_btn.show()
        elif hasattr(cuda_help_btn, "hide"):
            cuda_help_btn.hide()


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


def _shimmer_gradient_stops(phase: float, *, dark: bool) -> list[tuple[float, tuple[int, int, int, int]]]:
    """Compute smooth left-to-right text-shimmer RGBA gradient stops for phase in [0, 1)."""
    pos = -0.35 + (float(phase) % 1.0) * 1.70
    half_width = 0.28
    if dark:
        base = (132, 138, 146, 255)   # calm muted gray matching dark tab bar labels
        mid = (196, 203, 212, 255)
        peak = (248, 250, 252, 255)   # crisp bright highlight sweeping left -> right
    else:
        base = (146, 152, 160, 255)   # calm muted gray matching light tab bar labels
        mid = (78, 84, 92, 255)
        peak = (24, 28, 34, 255)      # crisp dark sheen sweeping left -> right
    raw = [
        (0.0, base),
        (pos - half_width, base),
        (pos - half_width * 0.45, mid),
        (pos, peak),
        (pos + half_width * 0.45, mid),
        (pos + half_width, base),
        (1.0, base),
    ]
    stops: list[tuple[float, tuple[int, int, int, int]]] = []
    for p, rgba in raw:
        c = round(max(0.0, min(1.0, p)), 4)
        if stops and c <= stops[-1][0]:
            if c == stops[-1][0]:
                stops[-1] = (c, rgba)
            continue
        stops.append((c, rgba))
    if stops[-1][0] < 1.0:
        stops.append((1.0, base))
    return stops


def _qt_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read a Qt attribute whether PythonQt exposes it as a property or PyQt exposes it as a method."""
    if not hasattr(obj, name):
        return default
    val = getattr(obj, name)
    if callable(val):
        try:
            return val()
        except TypeError:
            return val
    return val


def _render_shimmer_text_icon(qt: Any, button: Any, text: str, phase: float, *, dark: bool) -> Optional[tuple[Any, Any]]:
    """Render borderless left-to-right shimmering text into a high-DPI QIcon (Normal + Disabled)."""
    required = ("QImage", "QPainter", "QPixmap", "QIcon", "QLinearGradient", "QColor", "QBrush", "QFontMetrics", "QRectF", "QSize")
    if not all(hasattr(qt, attr) for attr in required):
        return None
    try:
        font = _qt_attr(button, "font")
        if font is None:
            return None
        fm = qt.QFontMetrics(font)
        text_w = int(fm.horizontalAdvance(text) if hasattr(fm, "horizontalAdvance") else fm.width(text))
        text_h = int(fm.height() if hasattr(fm, "height") else 16)
        w = max(text_w + 4, 48)
        h = max(text_h + 4, 20)
        try:
            dpr = max(1.0, float(_qt_attr(button, "devicePixelRatioF", 1.0) or 1.0))
        except Exception:
            dpr = 1.0
        img = qt.QImage(
            max(1, int(round(w * dpr))),
            max(1, int(round(h * dpr))),
            qt.QImage.Format_ARGB32_Premultiplied,
        )
        if hasattr(img, "setDevicePixelRatio"):
            img.setDevicePixelRatio(dpr)
        img.fill(0)

        left = max(0.0, (float(w) - float(text_w)) * 0.5)
        right = min(float(w), left + max(1.0, float(text_w)))
        painter = qt.QPainter(img)
        try:
            painter.setRenderHint(qt.QPainter.Antialiasing, True)
            painter.setRenderHint(qt.QPainter.TextAntialiasing, True)
            painter.setFont(font)
            painter.setPen(qt.QColor(255, 255, 255, 255))
            painter.drawText(qt.QRectF(0.0, 0.0, float(w), float(h)), qt.Qt.AlignCenter, text)
            grad = qt.QLinearGradient(left, 0.0, right, 0.0)
            for stop_pos, (r, g, b, a) in _shimmer_gradient_stops(phase, dark=dark):
                grad.setColorAt(stop_pos, qt.QColor(r, g, b, a))
            painter.setCompositionMode(qt.QPainter.CompositionMode_SourceIn)
            painter.fillRect(qt.QRectF(0.0, 0.0, float(w), float(h)), qt.QBrush(grad))
        finally:
            painter.end()

        pix = qt.QPixmap.fromImage(img)
        if hasattr(pix, "setDevicePixelRatio"):
            pix.setDevicePixelRatio(dpr)
        icon = qt.QIcon()
        icon.addPixmap(pix, qt.QIcon.Normal)
        icon.addPixmap(pix, qt.QIcon.Disabled)
        return icon, qt.QSize(w, h)
    except Exception:
        return None


def install_input_preview_controls(owner: Any) -> None:
    """Add the 'View input' button, its volume stepper and its status label to the end of 1. Inputs (H1).

    The row lives in ``configurationLayout`` outside the Prepare Inputs panel. b2-dwi: the 'Volume i / N (b = ...)'
    stepper stays hidden until a full-DWI preview exists (the owner's ``_refresh_input_preview_stepper``).
    b2 inputs-help: Batch mode hides the button and this row (``place_input_preview_button``).
    """

    if getattr(owner, "_inputPreviewWidgets", None) is not None:
        return
    import qt

    layout = getattr(owner.ui, "configurationLayout", None)
    if layout is None:
        return
    row = qt.QWidget()
    row.setObjectName("dspaceInputPreviewRow")
    row_layout = qt.QVBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(3)
    button = qt.QPushButton()
    button.setObjectName("inputPreviewButton")
    button.setSizePolicy(qt.QSizePolicy.Maximum, qt.QSizePolicy.Fixed)
    pointing = getattr(qt.Qt, "PointingHandCursor", None)
    if pointing is not None and hasattr(button, "setCursor"):
        button.setCursor(pointing)
    stepper = qt.QWidget()
    stepper.setObjectName("inputPreviewStepper")
    stepper_layout = qt.QHBoxLayout(stepper)
    stepper_layout.setContentsMargins(0, 0, 0, 0)
    stepper_layout.setSpacing(6)
    stepper_label = qt.QLabel()
    stepper_label.setObjectName("inputPreviewStepperLabel")
    slider = qt.QSlider(qt.Qt.Horizontal)
    slider.setObjectName("inputPreviewComponentSlider")
    slider.setRange(1, 1)
    if hasattr(slider, "setSingleStep"):
        slider.setSingleStep(1)
    if hasattr(slider, "setPageStep"):
        slider.setPageStep(1)
    if hasattr(slider, "setTracking"):
        slider.setTracking(True)
    slider.setMinimumWidth(60)
    slider.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed)
    if pointing is not None and hasattr(slider, "setCursor"):
        slider.setCursor(pointing)
    slider.setEnabled(False)
    spin = qt.QSpinBox()
    spin.setObjectName("inputPreviewComponentSpinBox")
    spin.setRange(1, 1)
    spin.setSizePolicy(qt.QSizePolicy.Maximum, qt.QSizePolicy.Fixed)
    spin.setEnabled(False)
    info = qt.QLabel()
    info.setObjectName("inputPreviewStepperInfo")
    info.setTextFormat(qt.Qt.PlainText)
    info.setMinimumWidth(0)
    # The info text never sets a minimum width, so the panel-fit logic is not widened by it.
    info.setSizePolicy(qt.QSizePolicy.Maximum, qt.QSizePolicy.Preferred)
    stepper_layout.addWidget(stepper_label)
    stepper_layout.addWidget(slider, 1)
    stepper_layout.addWidget(spin)
    stepper_layout.addWidget(info)
    stepper.hide()
    row_layout.addWidget(stepper)
    status = qt.QLabel()
    status.setObjectName("inputPreviewStatusLabel")
    status.setWordWrap(True)
    status.setTextFormat(qt.Qt.PlainText)
    status.setMinimumWidth(0)
    status.hide()
    row_layout.addWidget(status)
    button.clicked.connect(lambda *_args: owner._on_input_preview_clicked())
    slider.connect("valueChanged(int)", lambda value: owner._on_input_preview_component_changed(value))
    spin.connect("valueChanged(int)", lambda value: owner._on_input_preview_component_changed(value))
    layout.addWidget(row)
    owner.ui.inputPreviewButton = button
    owner._inputPreviewWidgets = {
        "row": row, "layout": row_layout, "button": button, "status": status,
        "stepper": stepper, "stepper_label": stepper_label, "slider": slider, "spin": spin, "info": info,
    }
    from shared.ui_prepare_inputs import place_input_preview_button
    place_input_preview_button(owner)
    refresh_input_preview_controls(owner)


def input_preview_loading_stylesheet(*, dark: bool = False) -> str:
    """Return a borderless transparent stylesheet for QPushButton#inputPreviewButton while loading."""
    ink = "#9AA0A6" if dark else "#606367"
    rule = (
        f"background: transparent; color: {ink}; border: 1px solid transparent;"
        " border-radius: 5px; padding: 2px 10px; font-size: 11px; min-height: 20px;"
    )
    return (
        f"QPushButton#inputPreviewButton {{ {rule} }}"
        f"QPushButton#inputPreviewButton:disabled {{ {rule} }}"
    )


def _input_preview_dark_mode(owner: Any, button: Any) -> bool:
    cached = getattr(owner, "_lastAppliedDarkMode", None)
    if cached is not None:
        return bool(cached)
    try:
        from shared.ui_styles import is_dark_mode

        return bool(is_dark_mode(button))
    except Exception:
        return False


def stop_input_preview_loading_animation(owner: Any) -> None:
    """Stop the View input button's left-to-right text-shimmer timer if active."""
    timer = getattr(owner, "_inputPreviewAnimTimer", None)
    if timer is not None:
        stop_fn = getattr(timer, "stop", None)
        if callable(stop_fn):
            try:
                stop_fn()
            except Exception:
                pass


def set_input_preview_button_loading(owner: Any, loading: bool, *, language: str | None = None) -> None:
    """Replace 'View input' in place with borderless shimmering 'Loading…' text while loading."""
    widgets = getattr(owner, "_inputPreviewWidgets", None)
    if not isinstance(widgets, dict) or "button" not in widgets:
        stop_input_preview_loading_animation(owner)
        return
    button = widgets["button"]
    lang = language or getattr(owner, "_onboardingLanguage", "en")
    zh = str(lang).startswith("zh")
    try:
        import qt
    except Exception:
        qt = None

    def _apply_shimmer_frame(text: str, phase: float, *, dark: bool) -> None:
        rendered = _render_shimmer_text_icon(qt, button, text, phase, dark=dark) if qt is not None else None
        if rendered is not None and hasattr(button, "setIcon") and hasattr(button, "setIconSize"):
            icon, size = rendered
            if hasattr(button, "setText"):
                button.setText("")
            button.setIcon(icon)
            button.setIconSize(size)
        elif hasattr(button, "setText"):
            button.setText(text)

    if loading:
        # Preserve at least the idle button width so the tab header row does not shift while loading.
        if getattr(owner, "_inputPreviewIdleMinWidth", None) is None:
            width = 0
            try:
                raw_w = _qt_attr(button, "width", 0)
                width = int(raw_w or 0)
            except Exception:
                width = 0
            if width <= 0:
                try:
                    hint = _qt_attr(button, "sizeHint", None)
                    width = int(_qt_attr(hint, "width", 0) or 0) if hint is not None else 0
                except Exception:
                    width = 0
            owner._inputPreviewIdleMinWidth = width
            if width > 0 and hasattr(button, "setMinimumWidth"):
                try:
                    button.setMinimumWidth(width)
                except Exception:
                    pass
        loading_text = "加载中…" if zh else "Loading…"
        loading_tip = "正在载入完整 DWI…" if zh else "Loading the full DWI…"
        owner._inputPreviewLoadingText = loading_text
        if hasattr(button, "setAccessibleName"):
            button.setAccessibleName(loading_tip)
        if hasattr(button, "setToolTip"):
            button.setToolTip(loading_tip)
        owner._inputPreviewAnimPhase = 0.0
        dark = _input_preview_dark_mode(owner, button)
        button._dspaceLoading = True
        if hasattr(button, "setStyleSheet"):
            try:
                button.setStyleSheet(input_preview_loading_stylesheet(dark=dark))
            except Exception:
                pass
        _apply_shimmer_frame(loading_text, 0.0, dark=dark)
        timer_cls = getattr(qt, "QTimer", None) if qt is not None else None
        if timer_cls is not None and callable(timer_cls):
            timer = getattr(owner, "_inputPreviewAnimTimer", None)
            if timer is None:
                try:
                    timer = timer_cls(button)
                except TypeError:
                    timer = timer_cls()
                if hasattr(timer, "setInterval"):
                    timer.setInterval(33)

                def _tick() -> None:
                    if getattr(owner, "_cleanedUp", False) or not getattr(owner, "_inputPreviewPending", False):
                        stop_input_preview_loading_animation(owner)
                        return
                    phase = (float(getattr(owner, "_inputPreviewAnimPhase", 0.0)) + 0.025) % 1.0
                    owner._inputPreviewAnimPhase = phase
                    cur_text = str(getattr(owner, "_inputPreviewLoadingText", loading_text) or loading_text)
                    _apply_shimmer_frame(cur_text, phase, dark=_input_preview_dark_mode(owner, button))

                if hasattr(timer, "timeout") and hasattr(timer.timeout, "connect"):
                    timer.timeout.connect(_tick)
                elif hasattr(timer, "connect"):
                    timer.connect("timeout()", _tick)
                owner._inputPreviewAnimTimer = timer
            if hasattr(timer, "start"):
                try:
                    timer.start()
                except Exception:
                    pass
        return

    stop_input_preview_loading_animation(owner)
    owner._inputPreviewAnimPhase = 0.0
    owner._inputPreviewLoadingText = ""
    button._dspaceLoading = False
    if qt is not None and hasattr(qt, "QIcon") and hasattr(button, "setIcon"):
        try:
            button.setIcon(qt.QIcon())
        except Exception:
            pass
    if getattr(owner, "_inputPreviewIdleMinWidth", None) is not None:
        owner._inputPreviewIdleMinWidth = None
        if hasattr(button, "setMinimumWidth"):
            try:
                button.setMinimumWidth(0)
            except Exception:
                pass
    if hasattr(button, "setStyleSheet"):
        try:
            button.setStyleSheet("")
        except Exception:
            pass
    if hasattr(button, "setText"):
        button.setText("预览输入" if zh else "View input")
    if hasattr(button, "setAccessibleName"):
        button.setAccessibleName("预览输入" if zh else "View input")


def refresh_input_preview_controls(owner: Any, language: str | None = None) -> None:
    """Translate the 'View input' button; the owner sets its enabled state and tooltip."""

    widgets = getattr(owner, "_inputPreviewWidgets", None)
    if not widgets:
        return
    lang = language or getattr(owner, "_onboardingLanguage", "en")
    zh = str(lang).startswith("zh")
    button = widgets["button"]
    if getattr(owner, "_inputPreviewPending", False):
        set_input_preview_button_loading(owner, True, language=lang)
    else:
        button.setText("预览输入" if zh else "View input")
        if hasattr(button, "setAccessibleName"):
            button.setAccessibleName("预览输入" if zh else "View input")
    stepper_label, slider, spin = widgets.get("stepper_label"), widgets.get("slider"), widgets.get("spin")
    if stepper_label is not None:
        stepper_label.setText("体积" if zh else "Volume")
    stepper_tip = (
        "拖动滑块或调整编号切换 DWI 体积（从 1 开始计数）。Volumes 模块中的 DWI Component 显示同一体积，但从 0 开始计数。" if zh
        else "Drag the slider or step through DWI volumes (counted from 1). The Volumes module's DWI Component shows "
             "the same volume, counted from 0."
    )
    if slider is not None:
        slider.setToolTip(stepper_tip)
    if spin is not None:
        spin.setToolTip(stepper_tip)
    refresh = getattr(owner, "_refresh_input_preview_controls", None)
    if callable(refresh):
        refresh()
    else:
        button.setToolTip(
            "在切片视图中显示当前输入的完整 DWI（先显示第一个 b0）。" if zh
            else "Show the full DWI of the current input in the slice views, starting at the first b0."
        )
    stepper_refresh = getattr(owner, "_refresh_input_preview_stepper", None)
    if callable(stepper_refresh):
        stepper_refresh()


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
    prepare_inputs.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)
    prepare_inputs.setMinimumWidth(0)
    for name in ("inputDirectoryLabel", "inputDirectoryPath"):
        getattr(ui, name).hide()
    try:
        install_input_preview_controls(owner)
    except Exception as exc:  # R3-4: a preview-row failure must never break the module load
        logging.warning("Input preview controls could not be installed: %s", exc)
    for name in ("parametersCollapsibleButton",):
        section = getattr(ui, name)
        layout.removeWidget(section)
        advanced_layout.addWidget(section)
        section.collapsed = True
        section.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)
        section.setMinimumWidth(0)

    advanced.hide()
    owner._advancedSettingsPanel = advanced
    ui.environmentCollapsibleButton.hide()
    from shared.ui_algorithm_picker import install_algorithm_pickers
    install_algorithm_pickers(owner)
    install_compute_device_selector(owner)

    layout.removeWidget(ui.runCollapsibleButton)
    ui.targetsLayout.addWidget(ui.runCollapsibleButton)
    layout.removeWidget(ui.statusFrame)
    if ui.runLayout.indexOf(ui.statusFrame) >= 0:
        ui.runLayout.removeWidget(ui.statusFrame)
    ui.statusLayout.setContentsMargins(0, 0, 0, 0)
    # Keep statusFrame below _runProgressRowWidget and _runStageLabel so live progress
    # sits directly beneath Run SlicerDSpace.
    ui.runLayout.addWidget(ui.statusFrame)
    ui.runLayout.setContentsMargins(0, 0, 0, 0)
    ui.runLayout.setSpacing(4)
    ui.runCollapsibleButton.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)
    ui.runCollapsibleButton.setMinimumWidth(0)
    ui.logLayout.setContentsMargins(0, 0, 0, 0)
    ui.logLayout.setSpacing(4)
    ui.logPlainTextEdit.setMinimumHeight(86)
    ui.logPlainTextEdit.setMaximumHeight(140)
    ui.clearLogButton.setSizePolicy(qt.QSizePolicy.Maximum, qt.QSizePolicy.Fixed)
    ui.logCollapsibleButton.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Expanding)
    ui.logCollapsibleButton.setMinimumWidth(0)
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
    from shared.ui_update_panel import install_update_panel
    install_update_panel(owner)
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
        ui.parametersCollapsibleButton,
        ui.environmentCollapsibleButton,
        ui.advancedRunCollapsibleButton,
        getattr(owner, "_environmentMoreOptions", None),
        getattr(owner, "_environmentAdvancedPathsWidget", None),
    ]
    for section in sections:
        if section is not None and hasattr(section, "connect"):
            section.connect("contentsCollapsed(bool)", on_section_toggled)
    queue_fit()


def apply_compact_language(owner: Any, language: str) -> None:
    """Keep concise main-screen labels in sync with the language system."""

    import qt

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
    from shared.ui_prepare_inputs import apply_input_language
    apply_input_language(owner)
    for name, key in {
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
        widget = getattr(owner.ui, name)
        text = onboarding_text(language, key)
        qgroupbox_cls = getattr(qt, "QGroupBox", None)
        if isinstance(qgroupbox_cls, type) and isinstance(widget, qgroupbox_cls):
            widget.setTitle(text)
        else:
            widget.setText(text)
    refresh_input_format_selector(owner, language)
    try:
        refresh_input_preview_controls(owner, language)
    except Exception as exc:  # R3-4: a preview-row failure must never break language switching
        logging.warning("Input preview controls could not be refreshed: %s", exc)
    refresh_compute_device_selector(owner, language)
    refresh_compact_algorithms(owner, language)
    from shared.ui_run_history import refresh_run_progress_styles
    refresh_run_progress_styles(owner)
    from shared.ui_update_panel import refresh_update_panel_language
    refresh_update_panel_language(owner, language)
    from shared.ui_advanced_layout import apply_advanced_language
    apply_advanced_language(owner)
