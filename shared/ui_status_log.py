"""Status, log and report-path helpers for the SlicerDSpace widget."""

from __future__ import annotations

from pathlib import Path
from time import monotonic
from typing import Any

from shared.ui_run_state import log_line_html
from shared.ui_styles import (
    RUN_STATUS_TABLE_STYLE,
    SLICERDSPACE_UI_STYLE,
    apply_placeholder_palette,
    is_dark_mode,
    run_status_table_style,
    run_summary_label_style,
    set_stylesheet_if_changed,
    theme_signature,
    ui_stylesheet,
)


STATUS_COLORS = {
    "ready": "#2E7D32",
    "running": "#1565C0",
    "waiting": "#63727D",
    "warning": "#865F24",
    "error": "#C62828",
}

DARK_STATUS_COLORS = {
    "ready": "#66BB6A",
    "running": "#64B5F6",
    "waiting": "#B8C5D0",
    "warning": "#FFB74D",
    "error": "#FF7B72",
}


def configure_log_widget(widget: Any, *, line_limit: int) -> None:
    """Configure the Slicer log widget for long background runs."""

    import qt

    if not widget:
        return
    widget.setToolTip(f"Live background log. The UI keeps the latest {line_limit} lines to avoid slowing long runs.")
    try:
        widget.maximumBlockCount = int(line_limit)
    except Exception:
        pass
    try:
        widget.setMaximumBlockCount(int(line_limit))
    except Exception:
        try:
            widget.document().setMaximumBlockCount(int(line_limit))
        except Exception:
            pass
    try:
        widget.setLineWrapMode(qt.QPlainTextEdit.NoWrap)
    except Exception:
        pass


def _repolish_widget_tree(root_widget: Any) -> None:
    """Force Qt/CTK child widgets to re-evaluate stylesheet rules after a Day/Night switch."""

    if root_widget is None:
        return
    try:
        import qt

        style_attr = getattr(root_widget, "style", None)
        st = style_attr() if callable(style_attr) else style_attr
        if st is not None:
            st.unpolish(root_widget)
            st.polish(root_widget)
        if hasattr(root_widget, "findChildren"):
            for child in root_widget.findChildren(qt.QWidget):
                cst_attr = getattr(child, "style", None)
                cst = cst_attr() if callable(cst_attr) else cst_attr
                if cst is not None:
                    cst.unpolish(child)
                    cst.polish(child)
                child.update()
        root_widget.update()
    except Exception:
        pass


def apply_widget_styles(owner: Any) -> None:
    """Apply Light/Dark SlicerDSpace styles across the main widget and all sub-panels."""

    root_widget = getattr(owner, "_uiWidget", None)
    sig = theme_signature(root_widget)
    dark = sig[0] == "dark" if sig else is_dark_mode(root_widget)
    prev_dark = getattr(owner, "_lastAppliedDarkMode", None)
    owner._activeAppearanceSignature = sig
    owner._lastAppliedDarkMode = dark

    style = ui_stylesheet(dark=dark, widget=root_widget)
    try:
        if root_widget:
            set_stylesheet_if_changed(root_widget, style)
            apply_placeholder_palette(root_widget, dark=dark)
            if prev_dark is not None and prev_dark != dark:
                _repolish_widget_tree(root_widget)
    except Exception:
        try:
            set_stylesheet_if_changed(owner.parent, style)
        except Exception:
            pass

    ui = getattr(owner, "ui", None)
    run_btn = getattr(ui, "runSelectedButton", None) if ui is not None else None
    if run_btn is not None:
        try:
            font_attr = getattr(run_btn, "font", None)
            font = font_attr() if callable(font_attr) else font_attr
            if font is not None:
                if hasattr(font, "setBold"):
                    font.setBold(True)
                if hasattr(font, "setPixelSize"):
                    font.setPixelSize(15)
                run_btn.setFont(font)
        except Exception:
            pass

    also_runs = getattr(owner, "_alsoRunsLabel", None)
    if also_runs is not None:
        try:
            color = "#7CC4EA" if dark else "#335B70"
            set_stylesheet_if_changed(also_runs, f"QLabel#alsoRunsAutomaticallyLabel {{ color: {color}; }}")
        except Exception:
            pass

    if getattr(owner, "_headerWordmark", None) is not None:
        try:
            from shared.ui_branding import install_header_wordmark

            install_header_wordmark(owner, dark=dark)
        except Exception:
            pass

    table = getattr(owner, "_runStatusTable", None)
    if table:
        set_stylesheet_if_changed(table, run_status_table_style(dark=dark, widget=table))
        try:
            table.verticalHeader().setVisible(False)
        except Exception:
            pass

    status_label = getattr(ui, "statusLabel", None) if ui is not None else None
    last_status_kind = getattr(owner, "_lastStatusKind", None)
    if status_label is not None and last_status_kind:
        light_color = STATUS_COLORS.get(last_status_kind, STATUS_COLORS["error"])
        palette = DARK_STATUS_COLORS if dark else STATUS_COLORS
        color = palette.get(last_status_kind, palette["error"])
        token_comment = f" /* {light_color} */" if dark and color != light_color else ""
        set_stylesheet_if_changed(
            status_label,
            f"background: transparent; color: {color};{token_comment} padding: 2px 0; border: none;",
        )

    summary_label = getattr(owner, "_runSummaryLabel", None)
    last_summary_state = getattr(owner, "_lastRunSummaryState", None)
    if summary_label is not None and last_summary_state is not None:
        set_stylesheet_if_changed(
            summary_label,
            run_summary_label_style(last_summary_state, dark=dark, widget=summary_label),
        )

    for module_name, fn_name in (
        ("shared.ui_prepare_inputs", "refresh_prepare_inputs_styles"),
        ("shared.ui_algorithm_picker", "refresh_picker_icons"),
        ("shared.ui_environment_paths", "refresh_environment_paths_styles"),
        ("shared.ui_onboarding", "refresh_onboarding_styles"),
        ("shared.ui_regions", "refresh_region_styles"),
        ("shared.ui_compact", "refresh_input_format_help_popover_style"),
        ("shared.ui_run_history", "refresh_run_progress_styles"),
    ):
        try:
            mod = __import__(module_name, fromlist=[fn_name])
            getattr(mod, fn_name)(owner, dark=dark)
        except Exception:
            pass

    if prev_dark is not None and prev_dark != dark:
        for method_name in ("_refresh_topup_controls", "_refresh_bundle_backend_highlight"):
            fn = getattr(owner, method_name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
        log_widget = getattr(ui, "logPlainTextEdit", None) if ui is not None else None
        if log_widget is not None:
            try:
                existing_text = str(log_widget.toPlainText() or "")
                if existing_text:
                    scrollbar = log_widget.verticalScrollBar()
                    position = scrollbar.value if scrollbar is not None else 0
                    lines = existing_text.splitlines()
                    log_widget.clear()
                    if all(lines):
                        log_widget.appendHtml(
                            "".join(f'<p style="margin:0">{log_line_html(line, dark=dark)}</p>' for line in lines)
                        )
                    else:
                        for line in lines:
                            log_widget.appendHtml(log_line_html(line, dark=dark))
                    if scrollbar is not None:
                        scrollbar.setValue(position)
            except Exception:
                pass
    env_fn = getattr(owner, "_update_environment_result_label", None)
    if callable(env_fn):
        try:
            env_fn(
                getattr(owner, "_lastEnvironmentResult", None),
                error=getattr(owner, "_lastEnvironmentError", None),
            )
        except Exception:
            pass


def install_theme_change_listener(owner: Any) -> None:
    """Listen for Slicer/OS Day-Night Appearance changes and refresh widget styles live."""

    if getattr(owner, "_appearanceListenerInstalled", False):
        return
    try:
        import qt
    except Exception:
        return

    def _flush_theme_refresh() -> None:
        owner._themeRefreshQueued = False
        if getattr(owner, "_cleanedUp", False):
            return
        root_widget = getattr(owner, "_uiWidget", None)
        if theme_signature(root_widget) != getattr(owner, "_activeAppearanceSignature", None):
            apply_widget_styles(owner)

    def _queue_theme_refresh(*_args: Any) -> None:
        if getattr(owner, "_cleanedUp", False) or getattr(owner, "_themeRefreshQueued", False):
            return
        owner._themeRefreshQueued = True
        try:
            qt.QTimer.singleShot(0, _flush_theme_refresh)
        except Exception:
            _flush_theme_refresh()

    watch_timer = None
    if hasattr(qt, "QTimer"):
        try:
            watch_timer = qt.QTimer()
            if hasattr(watch_timer, "setInterval"):
                watch_timer.setInterval(400)
            timeout_sig = getattr(watch_timer, "timeout", None)
            if timeout_sig is not None and hasattr(timeout_sig, "connect"):
                timeout_sig.connect(_flush_theme_refresh)
            elif hasattr(watch_timer, "connect"):
                watch_timer.connect("timeout()", _flush_theme_refresh)
            if hasattr(watch_timer, "start"):
                watch_timer.start()
        except Exception:
            watch_timer = None

    app_cls = getattr(qt, "QApplication", None)
    app_instance = app_cls.instance() if app_cls is not None and hasattr(app_cls, "instance") else None
    palette_connected = False
    if app_instance is not None:
        try:
            sig_obj = getattr(app_instance, "paletteChanged", None)
            if sig_obj is not None and hasattr(sig_obj, "connect"):
                sig_obj.connect(_queue_theme_refresh)
                palette_connected = True
            elif hasattr(app_instance, "connect"):
                app_instance.connect("paletteChanged(QPalette)", _queue_theme_refresh)
                palette_connected = True
        except Exception:
            pass

    owner._appearanceListenerInstalled = True
    owner._appearanceWatchTimer = watch_timer
    owner._appearanceFlushCallback = _flush_theme_refresh
    owner._appearanceRefreshCallback = _queue_theme_refresh if palette_connected else None


def uninstall_theme_change_listener(owner: Any) -> None:
    """Detach the live Appearance listener on widget teardown."""

    watch_timer = getattr(owner, "_appearanceWatchTimer", None)
    if watch_timer is not None:
        try:
            if hasattr(watch_timer, "stop"):
                watch_timer.stop()
            flush_cb = getattr(owner, "_appearanceFlushCallback", None)
            if flush_cb is not None:
                timeout_sig = getattr(watch_timer, "timeout", None)
                if timeout_sig is not None and hasattr(timeout_sig, "disconnect"):
                    timeout_sig.disconnect(flush_cb)
                elif hasattr(watch_timer, "disconnect"):
                    watch_timer.disconnect("timeout()", flush_cb)
        except Exception:
            pass
    cb = getattr(owner, "_appearanceRefreshCallback", None)
    if cb is not None:
        try:
            import qt

            app_cls = getattr(qt, "QApplication", None)
            app_instance = app_cls.instance() if app_cls is not None and hasattr(app_cls, "instance") else None
            if app_instance is not None:
                sig_obj = getattr(app_instance, "paletteChanged", None)
                if sig_obj is not None and hasattr(sig_obj, "disconnect"):
                    sig_obj.disconnect(cb)
                elif hasattr(app_instance, "disconnect"):
                    app_instance.disconnect("paletteChanged(QPalette)", cb)
        except Exception:
            pass
    owner._appearanceListenerInstalled = False
    owner._appearanceWatchTimer = None
    owner._appearanceFlushCallback = None
    owner._appearanceRefreshCallback = None


def set_status(owner: Any, kind: str, message: str) -> None:
    """Update the status label and progress bar."""

    owner._lastStatusKind = kind
    label = getattr(owner.ui, "statusLabel", None)
    if not label:
        return
    progress = getattr(owner.ui, "statusProgressBar", None)
    dark = is_dark_mode(label)
    light_color = STATUS_COLORS.get(kind, STATUS_COLORS["error"])
    palette = DARK_STATUS_COLORS if dark else STATUS_COLORS
    color = palette.get(kind, palette["error"])
    token_comment = f" /* {light_color} */" if dark and color != light_color else ""
    try:
        current_text = label.text() if callable(getattr(label, "text", None)) else getattr(label, "text", None)
    except Exception:
        current_text = None
    if current_text != message:
        label.setText(message)
    set_stylesheet_if_changed(
        label,
        f"background: transparent; color: {color};{token_comment} padding: 2px 0; border: none;",
    )
    if not progress:
        return
    if kind == "running":
        progress.setRange(0, 0)
    elif kind == "ready":
        progress.setRange(0, 100)
        progress.setValue(100)
    else:
        progress.setRange(0, 100)
        progress.setValue(0)


def set_run_progress_message(owner: Any, message: str) -> None:
    """Update the run-progress label and transient Slicer status bar."""

    import slicer

    label = getattr(owner, "_runStatusHintLabel", None)
    if label:
        label.setText(message)
    slicer.util.showStatusMessage(message, 3000)


def set_run_summary_message(owner: Any, message: str, state: str = "") -> None:
    """Update the run summary label."""

    owner._lastRunSummaryState = state
    label = getattr(owner, "_runSummaryLabel", None)
    if not label:
        return
    target_text = message or "Run summary is not available yet."
    try:
        current_text = label.text() if callable(getattr(label, "text", None)) else getattr(label, "text", None)
    except Exception:
        current_text = None
    if current_text != target_text:
        label.setText(target_text)
    set_stylesheet_if_changed(label, run_summary_label_style(state, widget=label))


def set_existing_report_path(owner: Any, attribute: str, button_attribute: str, report_path: str | Path | None) -> None:
    """Store a report path only if it exists and update its button state."""

    path = Path(str(report_path)).expanduser() if report_path else None
    if path is not None and not path.is_file():
        path = None
    setattr(owner, attribute, path)
    button = getattr(owner, button_attribute, None)
    if button:
        button.enabled = bool(path)


def append_log(owner: Any, text: str) -> None:
    """Append text to the log widget while throttling event processing."""

    import slicer

    if hasattr(owner.ui, "logPlainTextEdit"):
        log_widget = owner.ui.logPlainTextEdit
        scrollbar = log_widget.verticalScrollBar()
        position = scrollbar.value
        following = position >= scrollbar.maximum - 1
        lines = str(text).splitlines() or [""]
        use_dark = is_dark_mode(log_widget)
        try:
            if all(lines):
                # Empty HTML paragraphs collapse in Qt; batch only nonempty lines.
                log_widget.appendHtml(
                    "".join('<p style="margin:0">' + log_line_html(line, dark=use_dark) + "</p>" for line in lines)
                )
            else:
                for line in lines:
                    log_widget.appendHtml(log_line_html(line, dark=use_dark))
        except Exception:
            log_widget.appendPlainText('\n'.join(lines))
        owner._logAppendCount = int(getattr(owner, "_logAppendCount", 0) or 0) + 1
        if owner._logAppendCount % 100 == 0:
            owner._trim_log_widget(owner.ui.logPlainTextEdit)
        if following:
            log_widget.ensureCursorVisible()
        else:
            scrollbar.setValue(position)
    if text.startswith("[Orchestrator] Running "):
        owner._set_run_progress_message(text.replace("[Orchestrator] ", "", 1))
    now = monotonic()
    if now - float(getattr(owner, "_lastProcessEventsAt", 0.0) or 0.0) < 0.15:
        return
    owner._lastProcessEventsAt = now
    try:
        slicer.app.processEvents()
    except Exception:
        pass
