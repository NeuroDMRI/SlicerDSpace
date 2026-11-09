"""Stylesheets shared by the SlicerDSpace Qt UI."""

from __future__ import annotations

from typing import Any


def is_dark_mode(widget: Any = None) -> bool:
    """Return True when the active Qt/Slicer palette uses a dark window background."""

    try:
        import qt

        if widget is not None:
            palette_attr = getattr(widget, "palette", None)
            palette = palette_attr() if callable(palette_attr) else palette_attr
            if palette is not None:
                win = palette.color(qt.QPalette.Window)
                txt = palette.color(qt.QPalette.WindowText)
                if int(txt.lightness()) > int(win.lightness()) and int(txt.lightness()) > 128:
                    return True
                if int(win.lightness()) > int(txt.lightness()) and int(win.lightness()) >= 128:
                    return False
        app_instance = qt.QApplication.instance() if hasattr(qt.QApplication, "instance") else None
        if app_instance is not None or hasattr(qt.QApplication, "palette"):
            app_palette = qt.QApplication.palette()
            return int(app_palette.color(qt.QPalette.Window).lightness()) < 128
    except Exception:
        pass
    return False


def apply_placeholder_palette(root_widget: Any, *, dark: bool | None = None) -> None:
    """Ensure QLineEdit placeholder text has high contrast in both light and dark themes."""

    if root_widget is None:
        return
    try:
        import qt

        use_dark = is_dark_mode(root_widget) if dark is None else bool(dark)
        placeholder_color = qt.QColor("#9BA7B4" if use_dark else "#6E7B85")
        text_color = qt.QColor("#F0F6FC" if use_dark else "#1E293B")
        base_color = qt.QColor("#22272E" if use_dark else "#FFFFFF")
        role_placeholder = getattr(qt.QPalette, "PlaceholderText", None)
        edits = [root_widget] if isinstance(root_widget, qt.QLineEdit) else []
        if hasattr(root_widget, "findChildren"):
            edits.extend(root_widget.findChildren(qt.QLineEdit))
        for edit in edits:
            pal = qt.QPalette(edit.palette)
            pal.setColor(qt.QPalette.Text, text_color)
            pal.setColor(qt.QPalette.Base, base_color)
            if role_placeholder is not None:
                pal.setColor(role_placeholder, placeholder_color)
            edit.setPalette(pal)
    except Exception:
        pass


SLICERDSPACE_UI_STYLE = """
QFrame#headerFrame {
  background: transparent;
  border: none;
  border-bottom: 1px solid #D8DEE3;
  border-radius: 0;
}
QLabel#headlineLabel {
  color: #24343D;
  font-size: 15px;
  font-weight: 600;
}
ctkCollapsibleButton {
  background: transparent;
  border: none;
}
QWidget#externalBackendsGroup,
ctkCollapsibleButton#externalBackendsGroup,
QGroupBox#externalBackendsGroup {
  background: #F8FAFB;
  border: 1px solid #D8DEE3;
  border-radius: 4px;
  padding: 4px;
}
QLabel#subtitleLabel, QLabel#environmentHintLabel,
QLabel#environmentResultLabel, QLabel#externalSectionDescriptionLabel,
QLabel#externalSubsectionDescriptionLabel, QLabel#advancedRunSectionLabel,
QLabel#runMessageLabel, QLabel#runStageLabel,
QLabel#advancedSettingsHintLabel,
QLabel#dockerHintLabel, QLabel#ddparcelHintLabel,
QLabel#microstructureHintLabel, QLabel#ddsegHintLabel {
  color: #63727D;
  background: transparent;
  border: none;
}
QLabel#alsoRunsAutomaticallyLabel {
  color: #335B70;
  background: transparent;
  border: none;
}
QLabel#logHintLabel {
  color: #334550;
  font-weight: 600;
}
QLabel#externalSectionTitleLabel, QLabel#externalSubsectionTitleLabel {
  background: transparent;
  border: none;
  color: #334550;
  font-weight: 600;
  padding: 4px 0;
}
QLabel#modelAssetsSummaryLabel, QLabel#selectedProductsSummaryLabel,
QLabel#runSummaryLabel {
  background: #F6F8FA;
  border: 1px solid #DFE5E9;
  border-radius: 4px;
  color: #52616D;
  padding: 6px;
}
QLabel#runDetailsLabel, QLabel#reviewStateLabel {
  color: #63727D;
  background: transparent;
  border: none;
  padding: 0 10px;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
  background: #FFFFFF;
  color: #1E293B;
  border: 1px solid #CCD5DC;
  border-radius: 3px;
  padding: 2px 5px;
  min-height: 18px;
  selection-background-color: #D4E2EA;
  selection-color: #1E293B;
}
QComboBox QAbstractItemView {
  background: #FFFFFF;
  color: #1E293B;
  border: 1px solid #CCD5DC;
  selection-background-color: #D4E2EA;
  selection-color: #1E293B;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
  border: 1px solid #476E83;
}
QPushButton {
  background: #F8FAFB;
  border: 1px solid #CCD5DC;
  border-radius: 4px;
  color: #334550;
  padding: 4px 8px;
}
QPushButton:hover:!disabled {
  background: #EEF2F5;
  border: 1px solid #9BAEBB;
}
QToolButton#microstructureAlgorithmSelectButton,
QToolButton#tractographyAlgorithmSelectButton,
QToolButton#bundlesAlgorithmSelectButton {
  color: #335B70;
  background: #FFFFFF;
  border: 1px solid #CCD5DC;
  border-radius: 3px;
  padding: 0;
  min-height: 18px;
}
QToolButton:hover { border-color: #476E83; }
QStackedWidget#advancedCategoryStack { background: transparent; border: none; }
QPushButton:disabled {
  color: #8A98A3;
  background: #F2F4F6;
}
QPushButton#runSelectedButton {
  background: #335B70;
  border: 1px solid #335B70;
  border-radius: 4px;
  color: #FFFFFF;
  font-weight: 600;
  padding: 8px 12px;
}
QPushButton#runSelectedButton:hover:!disabled {
  background: #28495C;
}
QPushButton#runSelectedButton:disabled {
  background: #E3E9ED;
  border-color: #D4DDE4;
  color: #81919C;
}
QPushButton#cancelRunButton {
  background: #FFFFFF;
  color: #A7403E;
  border: 1px solid #D8BAB9;
  padding: 8px;
}
QPushButton#clearLogButton {
  background: transparent;
  border: none;
  color: #63727D;
  padding: 3px 6px;
}
QProgressBar {
  background: #E8EDF1;
  border: none;
  border-radius: 3px;
  color: #334550;
  max-height: 16px;
  text-align: center;
}
QProgressBar::chunk {
  background: #9EB8C8;
  border-radius: 3px;
}
QPlainTextEdit#logPlainTextEdit {
  background: #F7F9FB;
  border: 1px solid #D8DEE3;
  border-radius: 4px;
  color: #425360;
  font-family: "DejaVu Sans Mono", "Consolas", monospace;
  font-size: 11px;
  selection-background-color: #D4E2EA;
  selection-color: #223642;
}
"""

SLICERDSPACE_DARK_UI_STYLE = """
QFrame#headerFrame {
  background: transparent;
  border: none;
  border-bottom: 1px solid #444C56;
  border-radius: 0;
}
QLabel#headlineLabel {
  color: #F0F6FC;
  font-size: 15px;
  font-weight: 600;
}
ctkCollapsibleButton {
  background: transparent;
  border: none;
  color: #E6EDF3;
}
QLabel {
  color: #E6EDF3;
  background: transparent;
}
QCheckBox, QRadioButton {
  color: #E6EDF3;
}
QLabel:disabled, QCheckBox:disabled, QRadioButton:disabled {
  color: #7D8590;
}
QWidget#externalBackendsGroup,
ctkCollapsibleButton#externalBackendsGroup,
QGroupBox#externalBackendsGroup {
  background: #252B32;
  border: 1px solid #444C56;
  border-radius: 4px;
  color: #E6EDF3;
  padding: 4px;
}
QLabel#subtitleLabel, QLabel#environmentHintLabel,
QLabel#environmentResultLabel, QLabel#externalSectionDescriptionLabel,
QLabel#externalSubsectionDescriptionLabel, QLabel#advancedRunSectionLabel,
QLabel#runMessageLabel, QLabel#runStageLabel,
QLabel#advancedSettingsHintLabel,
QLabel#dockerHintLabel, QLabel#ddparcelHintLabel,
QLabel#microstructureHintLabel, QLabel#ddsegHintLabel {
  color: #B8C5D0;
  background: transparent;
  border: none;
}
QLabel#alsoRunsAutomaticallyLabel {
  color: #7CC4EA;
  background: transparent;
  border: none;
}
QLabel#logHintLabel {
  color: #D2DEE8;
  font-weight: 600;
}
QLabel#externalSectionTitleLabel, QLabel#externalSubsectionTitleLabel {
  background: transparent;
  border: none;
  color: #E6EDF3;
  font-weight: 600;
  padding: 4px 0;
}
QLabel#modelAssetsSummaryLabel, QLabel#selectedProductsSummaryLabel,
QLabel#runSummaryLabel {
  background: #252B32;
  border: 1px solid #444C56;
  border-radius: 4px;
  color: #D2DEE8;
  padding: 6px;
}
QLabel#runDetailsLabel, QLabel#reviewStateLabel {
  color: #B8C5D0;
  background: transparent;
  border: none;
  padding: 0 10px;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
  background: #22272E;
  color: #F0F6FC;
  border: 1px solid #525C68;
  border-radius: 3px;
  padding: 2px 5px;
  min-height: 18px;
  selection-background-color: #2E5C7A;
  selection-color: #FFFFFF;
}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
  background: #1C2026;
  color: #7D8590;
  border: 1px solid #3B424B;
}
QLineEdit:read-only {
  background: #1F242A;
  color: #B8C5D0;
}
QComboBox QAbstractItemView {
  background: #22272E;
  color: #F0F6FC;
  border: 1px solid #525C68;
  selection-background-color: #335B70;
  selection-color: #FFFFFF;
}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
  border: 1px solid #6CB6FF;
}
QListWidget, QPlainTextEdit {
  background: #1E2228;
  color: #E6EDF3;
  border: 1px solid #444C56;
  border-radius: 4px;
  selection-background-color: #2E5C7A;
  selection-color: #FFFFFF;
}
QPushButton {
  background: #323842;
  border: 1px solid #525C68;
  border-radius: 4px;
  color: #E6EDF3;
  padding: 4px 8px;
}
QPushButton:hover:!disabled {
  background: #3C4450;
  border: 1px solid #768390;
}
QToolButton#microstructureAlgorithmSelectButton,
QToolButton#tractographyAlgorithmSelectButton,
QToolButton#bundlesAlgorithmSelectButton {
  color: #8CD2F5;
  background: #323842;
  border: 1px solid #525C68;
  border-radius: 3px;
  font-weight: 600;
  padding: 0;
  min-height: 18px;
}
QToolButton:hover {
  background: #3C4450;
  border-color: #7CC4EA;
}
QStackedWidget#advancedCategoryStack { background: transparent; border: none; }
QPushButton:disabled {
  color: #7D8590;
  background: #252A31;
  border-color: #3B424B;
}
QPushButton#runSelectedButton {
  background: #2F6B8A;
  border: 1px solid #4889AB;
  border-radius: 4px;
  color: #FFFFFF;
  font-weight: 600;
  padding: 8px 12px;
}
QPushButton#runSelectedButton:hover:!disabled {
  background: #397CA0;
}
QPushButton#runSelectedButton:disabled {
  background: #262C33;
  border-color: #3B424B;
  color: #7D8590;
}
QPushButton#cancelRunButton {
  background: #3A2326;
  color: #FF8A80;
  border: 1px solid #8C4347;
  padding: 8px;
}
QPushButton#clearLogButton {
  background: transparent;
  border: none;
  color: #9BA7B4;
  padding: 3px 6px;
}
QProgressBar {
  background: #22272E;
  border: 1px solid #3B424B;
  border-radius: 3px;
  color: #E6EDF3;
  max-height: 16px;
  text-align: center;
}
QProgressBar::chunk {
  background: #3B7EA1;
  border-radius: 3px;
}
QPlainTextEdit#logPlainTextEdit {
  background: #1E2228;
  border: 1px solid #444C56;
  border-radius: 4px;
  color: #D2DEE8;
  font-family: "DejaVu Sans Mono", "Consolas", monospace;
  font-size: 11px;
  selection-background-color: #2E5C7A;
  selection-color: #FFFFFF;
}
"""

RUN_STATUS_TABLE_STYLE = (
    "QTableWidget { background: #FBFCFA; alternate-background-color: #F2F6F3; color: #1E293B; "
    "gridline-color: #D7E0DA; selection-background-color: #DCEBE2; selection-color: #163526; } "
    "QHeaderView::section { background: #E6EFE9; color: #294B3B; font-weight: 700; padding: 5px; border: 0; }"
)

DARK_RUN_STATUS_TABLE_STYLE = (
    "QTableWidget { background: #1E2228; alternate-background-color: #252A32; color: #E6EDF3; "
    "gridline-color: #3B424B; selection-background-color: #2E5C7A; selection-color: #FFFFFF; } "
    "QHeaderView::section { background: #2D333B; color: #E6EDF3; font-weight: 700; padding: 5px; border: 0; }"
)

ENVIRONMENT_RESULTS_DIALOG_STYLE = """
QDialog {
  background: #F4F7F6;
}
QLabel#environmentSummaryLabel {
  background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #EEF6F1, stop:1 #F7FBF9);
  color: #173126;
  padding: 14px 16px;
  border: 1px solid #CADCD1;
  border-radius: 10px;
  font-weight: 600;
}
QFrame#environmentItemFrame {
  border: 1px solid #D5DEDC;
  border-radius: 10px;
  background: #FFFFFF;
}
QLabel#environmentItemText {
  color: #223238;
  background: transparent;
}
QPushButton {
  min-height: 30px;
  padding: 6px 14px;
  border: 1px solid #C8D4D1;
  border-radius: 8px;
  background: #FFFFFF;
  color: #173126;
}
QPushButton:hover {
  background: #F1F6F4;
}
QPushButton:pressed {
  background: #E3EEE8;
}
QPlainTextEdit {
  background: #FFFFFF;
  color: #4A5C5F;
  border: 1px solid #D7E0DE;
  border-radius: 8px;
  padding: 6px;
}
QScrollArea {
  border: none;
  background: transparent;
}
"""

DARK_ENVIRONMENT_RESULTS_DIALOG_STYLE = """
QDialog {
  background: #22272E;
  color: #E6EDF3;
}
QLabel#environmentSummaryLabel {
  background: #283430;
  color: #D2F4E0;
  padding: 14px 16px;
  border: 1px solid #3B5E4C;
  border-radius: 10px;
  font-weight: 600;
}
QFrame#environmentItemFrame {
  border: 1px solid #444C56;
  border-radius: 10px;
  background: #2A3038;
}
QLabel#environmentItemText {
  color: #E6EDF3;
  background: transparent;
}
QPushButton {
  min-height: 30px;
  padding: 6px 14px;
  border: 1px solid #525C68;
  border-radius: 8px;
  background: #323842;
  color: #E6EDF3;
}
QPushButton:hover {
  background: #3C4450;
}
QPushButton:pressed {
  background: #464F5D;
}
QPlainTextEdit {
  background: #1E2228;
  color: #D2DEE8;
  border: 1px solid #444C56;
  border-radius: 8px;
  padding: 6px;
}
QScrollArea {
  border: none;
  background: transparent;
}
"""

ENVIRONMENT_READY_LABEL_STYLE = "QLabel { color: #2E7D32; padding: 12px; }"
DARK_ENVIRONMENT_READY_LABEL_STYLE = "QLabel { color: #66BB6A; padding: 12px; }"

RUN_SUMMARY_PALETTE = {
    "completed": ("#F0FAF3", "#174D32"),
    "passed": ("#F0FAF3", "#174D32"),
    "skipped": ("#F3F7F4", "#2F4038"),
    "planned": ("#EEF6FF", "#163E50"),
    "running": ("#EEF6FF", "#163E50"),
    "partial": ("#FFF8EA", "#6B3F00"),
    "blocked": ("#FFF3E8", "#6B3F00"),
    "failed": ("#FDECEC", "#6B1A1A"),
    "cancelled": ("#FFF3E8", "#6B3F00"),
    "unknown": ("#F3F7F4", "#2F4038"),
}

DARK_RUN_SUMMARY_PALETTE = {
    "completed": ("#1D3326", "#8CE99A"),
    "passed": ("#1D3326", "#8CE99A"),
    "skipped": ("#252B32", "#C9D1D9"),
    "planned": ("#1C2F42", "#8CD2F5"),
    "running": ("#1C2F42", "#8CD2F5"),
    "partial": ("#382C1B", "#FFD8A8"),
    "blocked": ("#382C1B", "#FFD8A8"),
    "failed": ("#3B2124", "#FFA8A8"),
    "cancelled": ("#382C1B", "#FFD8A8"),
    "unknown": ("#252B32", "#C9D1D9"),
}


def set_stylesheet_if_changed(widget: Any, style: str) -> None:
    """Apply ``style`` only when it differs from ``widget``'s current stylesheet to avoid repaint flicker."""

    if widget is None:
        return
    try:
        attr = getattr(widget, "styleSheet", None)
        current = attr() if callable(attr) else attr
    except Exception:
        current = None
    if current == style:
        return
    try:
        widget.setStyleSheet(style)
    except Exception:
        pass


def ui_stylesheet(*, dark: bool | None = None, widget: Any = None) -> str:
    """Return the main SlicerDSpace stylesheet for the active light or dark theme."""

    use_dark = is_dark_mode(widget) if dark is None else bool(dark)
    return SLICERDSPACE_DARK_UI_STYLE if use_dark else SLICERDSPACE_UI_STYLE


def run_status_table_style(*, dark: bool | None = None, widget: Any = None) -> str:
    """Return the run-status table stylesheet for the active light or dark theme."""

    use_dark = is_dark_mode(widget) if dark is None else bool(dark)
    return DARK_RUN_STATUS_TABLE_STYLE if use_dark else RUN_STATUS_TABLE_STYLE


def environment_results_dialog_style(*, dark: bool | None = None, widget: Any = None) -> str:
    """Return the environment-results dialog stylesheet for the active light or dark theme."""

    use_dark = is_dark_mode(widget) if dark is None else bool(dark)
    return DARK_ENVIRONMENT_RESULTS_DIALOG_STYLE if use_dark else ENVIRONMENT_RESULTS_DIALOG_STYLE


def environment_ready_label_style(*, dark: bool | None = None, widget: Any = None) -> str:
    """Return the ready-label stylesheet for the environment dialog."""

    use_dark = is_dark_mode(widget) if dark is None else bool(dark)
    return DARK_ENVIRONMENT_READY_LABEL_STYLE if use_dark else ENVIRONMENT_READY_LABEL_STYLE


def run_summary_label_style(state: str, *, dark: bool | None = None, widget: Any = None) -> str:
    """Return the stylesheet for the run-summary label state."""

    use_dark = is_dark_mode(widget) if dark is None else bool(dark)
    palette = DARK_RUN_SUMMARY_PALETTE if use_dark else RUN_SUMMARY_PALETTE
    border = "#444C56" if use_dark else "#D8E4DD"
    background, text = palette.get(state, palette["unknown"])
    light_bg, light_text = RUN_SUMMARY_PALETTE.get(state, RUN_SUMMARY_PALETTE["unknown"])
    comment = f" /* {light_bg} {light_text} */" if use_dark else ""
    return (
        "QLabel#runSummaryLabel { "
        f"background-color: {background}; border: 1px solid {border}; "
        f"border-radius: 7px; color: {text}; "
        f"padding: 7px 9px; font-size: 11px;{comment} }}"
    )

