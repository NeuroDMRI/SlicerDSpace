"""Qt builders for the SlicerDSpace environment-results dialog."""

from __future__ import annotations

import html
from typing import Any, Callable

from shared.ui_environment_panel import (
    environment_dialog_state,
    environment_item_button_text,
    environment_item_button_tooltip,
    environment_item_display,
    environment_summary_parts,
    should_route_prepare_all_to_cuda_setup,
)
from shared.ui_styles import (
    ENVIRONMENT_READY_LABEL_STYLE,
    ENVIRONMENT_RESULTS_DIALOG_STYLE,
    apply_placeholder_palette,
    environment_ready_label_style,
    environment_results_dialog_style,
    is_dark_mode,
)


def _set_selectable_text(label: Any, qt_mod: Any) -> None:
    try:
        label.setTextInteractionFlags(qt_mod.Qt.TextSelectableByMouse | qt_mod.Qt.TextSelectableByKeyboard)
    except Exception:
        pass


def build_environment_item_row(
    item: Any,
    *,
    on_install_item: Callable[[Any], None],
    zh: bool = False,
    last_row: bool = False,
):
    """Build one clean, scannable dependency row with collapsible technical diagnostics."""

    import qt

    disp = environment_item_display(item, zh=zh)

    frame = qt.QFrame()
    frame.setObjectName("environmentItemFrame")
    try:
        frame.setFrameShape(qt.QFrame.NoFrame)
    except Exception:
        pass
    try:
        frame.setProperty("lastRow", bool(last_row))
    except Exception:
        pass

    outer = qt.QHBoxLayout(frame)
    outer.setContentsMargins(16, 13, 16, 13)
    outer.setSpacing(16)

    left = qt.QVBoxLayout()
    left.setContentsMargins(0, 0, 0, 0)
    left.setSpacing(4)

    header_row = qt.QHBoxLayout()
    header_row.setContentsMargins(0, 0, 0, 0)
    header_row.setSpacing(8)

    title_lbl = qt.QLabel(disp.title)
    title_lbl.setObjectName("environmentItemTitle")
    title_lbl.wordWrap = False
    _set_selectable_text(title_lbl, qt)
    header_row.addWidget(title_lbl)

    badge = qt.QLabel(disp.badge_text)
    badge.setObjectName("environmentItemBadge")
    try:
        badge.setProperty("severity", disp.badge_severity)
        badge.setSizePolicy(qt.QSizePolicy.Fixed, qt.QSizePolicy.Fixed)
    except Exception:
        pass
    try:
        header_row.addWidget(badge, 0, qt.Qt.AlignVCenter)
    except Exception:
        header_row.addWidget(badge)
    header_row.addStretch(1)
    left.addLayout(header_row)

    if disp.meta_text:
        meta_lbl = qt.QLabel(disp.meta_text)
        meta_lbl.setObjectName("environmentItemMeta")
        meta_lbl.wordWrap = True
        _set_selectable_text(meta_lbl, qt)
        left.addWidget(meta_lbl)

    guidance_lbl = qt.QLabel(disp.guidance_text)
    guidance_lbl.setObjectName("environmentItemText")
    guidance_lbl.wordWrap = True
    _set_selectable_text(guidance_lbl, qt)
    left.addWidget(guidance_lbl)

    if disp.technical_details:
        toggle_label_closed = "▸ 查看诊断详情" if zh else "▸ Technical details"
        toggle_label_open = "▾ 收起诊断详情" if zh else "▾ Hide details"
        details_toggle = qt.QToolButton()
        details_toggle.setObjectName("environmentItemDetailsToggle")
        details_toggle.setText(toggle_label_closed)
        try:
            details_toggle.setCheckable(True)
            details_toggle.setChecked(False)
            if hasattr(qt.Qt, "PointingHandCursor"):
                details_toggle.setCursor(qt.Qt.PointingHandCursor)
        except Exception:
            pass

        tech_lbl = qt.QLabel(disp.technical_details)
        tech_lbl.setObjectName("environmentItemTechnicalDetails")
        tech_lbl.wordWrap = True
        tech_lbl.visible = False
        _set_selectable_text(tech_lbl, qt)

        def _toggle_details(checked: bool, btn=details_toggle, box=tech_lbl) -> None:
            box.setVisible(bool(checked))
            btn.setText(toggle_label_open if checked else toggle_label_closed)

        try:
            if hasattr(details_toggle, "toggled") and hasattr(details_toggle.toggled, "connect"):
                details_toggle.toggled.connect(_toggle_details)
            else:
                details_toggle.connect("toggled(bool)", _toggle_details)
        except Exception:
            try:
                details_toggle.connect("toggled(bool)", _toggle_details)
            except Exception:
                pass

        left.addWidget(details_toggle)
        left.addWidget(tech_lbl)

    outer.addLayout(left, 1)

    right = qt.QVBoxLayout()
    right.setContentsMargins(0, 0, 0, 0)
    right.setSpacing(0)
    button = qt.QPushButton(environment_item_button_text(item, zh=zh))
    button.setObjectName("environmentItemActionButton")
    button.toolTip = environment_item_button_tooltip(item, zh=zh)
    try:
        if hasattr(qt.Qt, "PointingHandCursor"):
            button.setCursor(qt.Qt.PointingHandCursor)
    except Exception:
        pass
    button.connect("clicked()", lambda: on_install_item(item))
    right.addStretch(1)
    right.addWidget(button)
    right.addStretch(1)
    outer.addLayout(right)
    return frame


def show_environment_results_dialog(
    *,
    owner: Any,
    result: Any,
    python_install_target: str,
    on_install_all: Callable[[], None],
    on_show_notes: Callable[[], None],
    on_install_item: Callable[[Any], None],
):
    """Create, show and return the environment-results dialog."""

    import qt
    import slicer

    previous = getattr(owner, "_environmentDialog", None)
    if previous:
        try:
            previous.close()
        except Exception:
            pass

    zh = str(getattr(owner, "_onboardingLanguage", "en") or "en").startswith("zh")
    dialog = qt.QDialog(slicer.util.mainWindow())
    dialog.setWindowTitle("SlicerDSpace 环境依赖检查" if zh else "SlicerDSpace Environment Dependencies")
    from shared.ui_compact import fit_size_to_screen

    dialog.resize(*fit_size_to_screen(qt, 900, 640, slicer.util.mainWindow()))  # small screens: stay inside it
    dark = is_dark_mode(dialog)
    dialog.setStyleSheet(environment_results_dialog_style(dark=dark, widget=dialog))
    apply_placeholder_palette(dialog, dark=dark)
    owner._environmentDialog = dialog
    state = environment_dialog_state(result, python_install_target=python_install_target)

    layout = qt.QVBoxLayout(dialog)
    layout.setContentsMargins(18, 16, 18, 16)
    layout.setSpacing(12)

    severity, headline, subtitle = environment_summary_parts(result, zh=zh)
    if headline:
        dot_color = {
            "blocked": "#EF4444" if dark else "#DC2626",
            "warning": "#F59E0B" if dark else "#D97706",
            "ready": "#22C55E" if dark else "#16A34A",
        }.get(severity, "#64748B")
        sub_color = "#B8C5D0" if dark else "#475569"
        summary_html = (
            f'<div style="line-height:1.4;">'
            f'<span style="color:{dot_color}; font-size:13px;">&#9679;</span>&nbsp;&nbsp;'
            f'<b>{html.escape(headline)}</b>'
            + (
                f'<div style="margin-top:3px; font-size:12px; font-weight:400; color:{sub_color};">'
                f'{html.escape(subtitle)}</div>'
                if subtitle
                else ""
            )
            + "</div>"
        )
        summary_label = qt.QLabel(summary_html)
        try:
            if hasattr(qt.Qt, "RichText"):
                summary_label.setTextFormat(qt.Qt.RichText)
        except Exception:
            pass
    else:
        summary_label = qt.QLabel(state.summary_text)
    summary_label.setObjectName("environmentSummaryLabel")
    try:
        summary_label.setProperty("severity", severity)
    except Exception:
        pass
    summary_label.wordWrap = True
    layout.addWidget(summary_label)

    if state.missing_items:
        list_container = qt.QFrame()
        list_container.setObjectName("environmentListContainer")
        container_layout = qt.QVBoxLayout(list_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        scroll = qt.QScrollArea()
        scroll.setWidgetResizable(True)
        try:
            scroll.setFrameShape(qt.QFrame.NoFrame)
        except Exception:
            pass
        scroll_widget = qt.QWidget()
        scroll_widget.setObjectName("environmentScrollContents")
        scroll_layout = qt.QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(0)
        items_list = list(state.missing_items)
        for idx, item in enumerate(items_list):
            scroll_layout.addWidget(
                build_environment_item_row(
                    item,
                    on_install_item=on_install_item,
                    zh=zh,
                    last_row=(idx == len(items_list) - 1),
                )
            )
        scroll_layout.addStretch(1)
        scroll.setWidget(scroll_widget)
        container_layout.addWidget(scroll)
        layout.addWidget(list_container, 1)
    else:
        ready_text = (
            "所选分析目标均已就绪，可直接开始运行。"
            if zh
            else state.ready_text
        )
        ready_label = qt.QLabel(ready_text)
        ready_label.wordWrap = True
        ready_label.setStyleSheet(environment_ready_label_style(dark=dark, widget=dialog))
        layout.addWidget(ready_label)
        layout.addStretch(1)

    details = qt.QPlainTextEdit()
    details.setReadOnly(True)
    details.visible = False
    details.setMaximumHeight(130)
    details.setPlainText(state.detail_text)
    layout.addWidget(details)

    sep = qt.QFrame()
    sep.setObjectName("dialogFooterSeparator")
    try:
        sep.setFrameShape(qt.QFrame.HLine)
    except Exception:
        pass
    layout.addWidget(sep)

    button_row = qt.QHBoxLayout()
    button_row.setContentsMargins(0, 2, 0, 0)
    button_row.setSpacing(8)

    notes_button = qt.QPushButton("手动配置说明" if zh else "Manual setup notes")
    notes_button.setObjectName("environmentManualNotesButton")
    notes_button.enabled = state.has_manual_setup_notes
    notes_button.connect("clicked()", on_show_notes)
    button_row.addWidget(notes_button)

    detail_closed = "▸ 摘要详情" if zh else "▸ Summary details"
    detail_open = "▾ 收起摘要" if zh else "▾ Hide summary"
    detail_button = qt.QPushButton(detail_closed)
    detail_button.setObjectName("environmentSummaryDetailsButton")
    detail_button.checkable = True
    detail_button.checked = False
    detail_button.connect("toggled(bool)", details.setVisible)
    detail_button.connect(
        "toggled(bool)",
        lambda checked: detail_button.setText(detail_open if checked else detail_closed),
    )
    button_row.addWidget(detail_button)
    button_row.addStretch(1)

    close_button = qt.QPushButton("关闭" if zh else "Close")
    close_button.setObjectName("environmentCloseButton")
    close_button.connect("clicked()", dialog.close)
    button_row.addWidget(close_button)

    try:
        all_cuda = should_route_prepare_all_to_cuda_setup(result)
    except Exception:
        all_cuda = False
    if all_cuda:
        install_all_label = "安装 CUDA 版 PyTorch" if zh else "Install CUDA PyTorch"
    else:
        install_all_label = "准备所选环境" if zh else "Prepare selected environment"
    install_all_button = qt.QPushButton(install_all_label)
    install_all_button.setObjectName("environmentInstallAllButton")
    try:
        install_all_button.setProperty("primaryAction", True)
        install_all_button.setDefault(bool(state.can_install_python))
    except Exception:
        pass
    install_all_button.toolTip = state.python_install_tooltip
    install_all_button.enabled = state.can_install_python
    install_all_button.connect("clicked()", on_install_all)
    button_row.addWidget(install_all_button)

    layout.addLayout(button_row)

    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    return dialog
