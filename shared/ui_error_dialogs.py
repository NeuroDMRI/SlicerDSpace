"""Small UI error-dialog helpers for SlicerDSpace widgets."""

from __future__ import annotations


def format_actionable_error_message(
    *,
    title: str,
    what: str,
    why: str,
    next_step: str,
    auto_fix: str,
    details: object,
) -> str:
    """Return the full user-facing actionable error message."""

    body = "\n\n".join(
        [
            f"What happened:\n{what}",
            f"Why:\n{why}",
            f"What to do next:\n{next_step}",
            f"Automatic fix:\n{auto_fix}",
            f"Details:\n{details}",
        ]
    )
    return f"{title}\n\n{body}"


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in str(text or ""))


def build_actionable_error_dialog(
    *,
    parent,
    title: str,
    what: str,
    why: str,
    next_step: str,
    auto_fix: str,
    details: object,
    actions: tuple[tuple[str, object], ...] = (),
):
    """Build (without executing) a clean, structured actionable error dialog."""

    import qt
    import slicer
    from shared.ui_styles import (
        actionable_error_dialog_style,
        apply_placeholder_palette,
        is_dark_mode,
    )

    message = format_actionable_error_message(
        title=title,
        what=what,
        why=why,
        next_step=next_step,
        auto_fix=auto_fix,
        details=details,
    )
    zh = _contains_cjk(title) or _contains_cjk(what)
    dialog = qt.QDialog(parent or slicer.util.mainWindow())
    dialog.setObjectName("actionableErrorDialog")
    dialog.setWindowTitle(title)
    dialog.setMinimumWidth(560)
    try:
        dialog.setMaximumWidth(760)
    except Exception:
        pass
    dark = is_dark_mode(dialog)
    dialog.setStyleSheet(actionable_error_dialog_style(dark=dark, widget=dialog))
    apply_placeholder_palette(dialog, dark=dark)

    layout = qt.QVBoxLayout(dialog)
    layout.setContentsMargins(20, 18, 20, 16)
    layout.setSpacing(14)

    header = qt.QHBoxLayout()
    header.setContentsMargins(0, 0, 0, 0)
    header.setSpacing(10)
    badge = qt.QLabel("!")
    badge.setObjectName("errorDialogBadge")
    try:
        badge.setAlignment(qt.Qt.AlignCenter)
    except Exception:
        pass
    header.addWidget(badge)

    title_lbl = qt.QLabel(title)
    title_lbl.setObjectName("errorDialogTitle")
    title_lbl.setWordWrap(True)
    header.addWidget(title_lbl, 1)
    layout.addLayout(header)

    if str(what or "").strip():
        lead = qt.QLabel(str(what).strip())
        lead.setObjectName("errorDialogLead")
        lead.setWordWrap(True)
        try:
            lead.setTextInteractionFlags(qt.Qt.TextSelectableByMouse | qt.Qt.TextSelectableByKeyboard)
        except Exception:
            pass
        layout.addWidget(lead)

    sections = [
        ("原因" if zh else "WHY", str(why or "").strip()),
        ("建议操作" if zh else "NEXT STEP", str(next_step or "").strip()),
        ("当前状态" if zh else "SAFETY NOTE", str(auto_fix or "").strip()),
    ]
    nonempty_sections = [(k, v) for k, v in sections if v]
    if nonempty_sections:
        card = qt.QFrame()
        card.setObjectName("errorDialogInfoCard")
        card_layout = qt.QVBoxLayout(card)
        card_layout.setContentsMargins(14, 12, 14, 12)
        card_layout.setSpacing(10)
        for heading_text, body_text in nonempty_sections:
            sec_box = qt.QVBoxLayout()
            sec_box.setContentsMargins(0, 0, 0, 0)
            sec_box.setSpacing(2)
            h_lbl = qt.QLabel(heading_text)
            h_lbl.setObjectName("errorSectionHeading")
            b_lbl = qt.QLabel(body_text)
            b_lbl.setObjectName("errorSectionBody")
            b_lbl.setWordWrap(True)
            try:
                b_lbl.setTextInteractionFlags(qt.Qt.TextSelectableByMouse | qt.Qt.TextSelectableByKeyboard)
            except Exception:
                pass
            sec_box.addWidget(h_lbl)
            sec_box.addWidget(b_lbl)
            card_layout.addLayout(sec_box)
        layout.addWidget(card)

    details_str = str(details or "").strip()
    tech_box = qt.QPlainTextEdit()
    tech_box.setObjectName("errorTechnicalDetails")
    tech_box.setReadOnly(True)
    tech_box.setPlainText(details_str or message)
    tech_box.setMinimumHeight(110)
    tech_box.setMaximumHeight(200)
    tech_box.setVisible(False)
    layout.addWidget(tech_box)

    divider = qt.QFrame()
    divider.setObjectName("errorDialogDivider")
    try:
        divider.setFrameShape(qt.QFrame.HLine)
    except Exception:
        pass
    layout.addWidget(divider)

    buttons = qt.QHBoxLayout()
    buttons.setContentsMargins(0, 2, 0, 0)
    buttons.setSpacing(8)

    detail_closed = "▸ 查看技术细节" if zh else "▸ Technical details"
    detail_open = "▾ 收起技术细节" if zh else "▾ Hide details"
    detail_toggle = qt.QPushButton(detail_closed)
    detail_toggle.setObjectName("errorToggleDetailsButton")
    try:
        detail_toggle.setProperty("ghostButton", True)
        detail_toggle.setCheckable(True)
        detail_toggle.setChecked(False)
    except Exception:
        pass

    def _on_toggle(checked: bool) -> None:
        tech_box.setVisible(bool(checked))
        detail_toggle.setText(detail_open if checked else detail_closed)
        try:
            dialog.adjustSize()
        except Exception:
            pass

    detail_toggle.connect("toggled(bool)", _on_toggle)
    buttons.addWidget(detail_toggle)

    copy_label = "复制完整报告" if zh else "Copy details"
    copied_label = "已复制" if zh else "Copied"
    copy_button = qt.QPushButton(copy_label)
    copy_button.setObjectName("errorCopyDetailsButton")
    try:
        copy_button.setProperty("ghostButton", True)
    except Exception:
        pass

    def _copy() -> None:
        try:
            qt.QApplication.clipboard().setText(message)
            copy_button.setText(copied_label)
            qt.QTimer.singleShot(1500, lambda: copy_button.setText(copy_label))
        except Exception:
            pass

    copy_button.connect("clicked()", _copy)
    buttons.addWidget(copy_button)

    buttons.addStretch(1)

    close_button = qt.QPushButton("关闭" if zh else "Close")
    close_button.setObjectName("errorCloseButton")
    if not actions:
        try:
            close_button.setProperty("primaryAction", True)
            close_button.setDefault(True)
        except Exception:
            pass
    close_button.connect("clicked()", dialog.accept)
    buttons.addWidget(close_button)

    for idx, (label, callback) in enumerate(actions):
        button = qt.QPushButton(label)
        if idx == len(actions) - 1:
            try:
                button.setProperty("primaryAction", True)
                button.setDefault(True)
            except Exception:
                pass
        button.connect("clicked()", callback)
        buttons.addWidget(button)
    layout.addLayout(buttons)
    return dialog


def show_actionable_error_dialog(
    *,
    parent,
    title: str,
    what: str,
    why: str,
    next_step: str,
    auto_fix: str,
    details: object,
    actions: tuple[tuple[str, object], ...] = (),
) -> None:
    """Show an actionable error dialog with clean layout, copyable report and optional buttons."""

    dialog = build_actionable_error_dialog(
        parent=parent,
        title=title,
        what=what,
        why=why,
        next_step=next_step,
        auto_fix=auto_fix,
        details=details,
        actions=actions,
    )
    dialog.exec_()
