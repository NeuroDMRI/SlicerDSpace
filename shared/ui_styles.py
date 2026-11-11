"""Stylesheets shared by the SlicerDSpace Qt UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

_CHECKMARK_ICON_URL = (Path(__file__).resolve().parents[1] / "Resources" / "Icons" / "checkmark-white.svg").as_posix()


def _palette_looks_dark(palette: Any, qt_mod: Any) -> bool | None:
    """Inspect a QPalette and return True for dark, False for light, or None if indeterminate."""

    if palette is None:
        return None
    try:
        win_l = int(palette.color(qt_mod.QPalette.Window).lightness())
        txt_l = int(palette.color(qt_mod.QPalette.WindowText).lightness())
    except Exception:
        return None
    base_l = win_l
    try:
        base_l = int(palette.color(qt_mod.QPalette.Base).lightness())
    except Exception:
        pass
    # Transparent container frames sometimes report Window=(0,0,0,0) with dark WindowText (0,0,0);
    # only treat equal lightness as dark when both Window and Base are dark and non-transparent.
    if win_l == txt_l:
        try:
            win_alpha = int(palette.color(qt_mod.QPalette.Window).alpha())
        except Exception:
            win_alpha = 255
        if win_alpha == 0:
            return None
        return win_l < 128 and base_l < 128
    if win_l < txt_l and win_l < 140:
        return True
    if win_l > txt_l and win_l >= 128:
        return False
    return win_l < 128


def detect_appearance_mode(widget: Any = None) -> str:
    """Return ``'dark'`` or ``'light'`` from the active Slicer/OS Appearance.

    Application-level style and palette take priority over ``widget.palette`` when a
    ``QApplication`` is running because ``widget.setStyleSheet(...)`` mutates the widget's
    local palette and would otherwise lock the UI into the previous theme.
    """

    try:
        import qt

        app_cls = getattr(qt, "QApplication", None)
        app_instance = app_cls.instance() if app_cls is not None and hasattr(app_cls, "instance") else None

        style_obj = None
        if app_instance is not None and hasattr(app_instance, "style"):
            style_attr = app_instance.style
            style_obj = style_attr() if callable(style_attr) else style_attr
        elif app_cls is not None and hasattr(app_cls, "style"):
            style_attr = app_cls.style
            style_obj = style_attr() if callable(style_attr) else style_attr

        if style_obj is not None and hasattr(style_obj, "objectName"):
            name_attr = style_obj.objectName
            style_name = str(name_attr() if callable(name_attr) else name_attr or "").strip().lower()
            if "dark" in style_name:
                return "dark"
            if "light" in style_name:
                return "light"

        if app_instance is not None and hasattr(app_instance, "styleHints"):
            try:
                hints = app_instance.styleHints()
                scheme_fn = getattr(hints, "colorScheme", None)
                scheme = scheme_fn() if callable(scheme_fn) else scheme_fn
                qt_ns = getattr(qt, "Qt", None)
                color_scheme_enum = getattr(qt_ns, "ColorScheme", None)
                if color_scheme_enum is not None:
                    if scheme == getattr(color_scheme_enum, "Dark", None):
                        return "dark"
                    if scheme == getattr(color_scheme_enum, "Light", None):
                        return "light"
            except Exception:
                pass

        if app_instance is not None and hasattr(app_instance, "palette"):
            pal_attr = app_instance.palette
            app_palette = pal_attr() if callable(pal_attr) else pal_attr
            verdict = _palette_looks_dark(app_palette, qt)
            if verdict is not None:
                return "dark" if verdict else "light"

        if widget is not None:
            palette_attr = getattr(widget, "palette", None)
            palette = palette_attr() if callable(palette_attr) else palette_attr
            verdict = _palette_looks_dark(palette, qt)
            if verdict is not None:
                return "dark" if verdict else "light"

        if app_cls is not None and hasattr(app_cls, "palette"):
            pal_attr = app_cls.palette
            app_palette = pal_attr() if callable(pal_attr) else pal_attr
            verdict = _palette_looks_dark(app_palette, qt)
            if verdict is not None:
                return "dark" if verdict else "light"
    except Exception:
        pass
    return "light"


def is_dark_mode(widget: Any = None) -> bool:
    """Return True when the active Slicer/OS Appearance uses dark mode."""

    return detect_appearance_mode(widget) == "dark"


def theme_signature(widget: Any = None) -> tuple[Any, ...]:
    """Return a hashable signature of the active Qt/Slicer style and palette."""

    mode = detect_appearance_mode(widget)
    style_name = ""
    win_name = ""
    txt_name = ""
    try:
        import qt

        app_cls = getattr(qt, "QApplication", None)
        app_instance = app_cls.instance() if app_cls is not None and hasattr(app_cls, "instance") else None
        style_obj = app_instance.style() if app_instance is not None and hasattr(app_instance, "style") else None
        if style_obj is not None and hasattr(style_obj, "objectName"):
            oname = style_obj.objectName
            style_name = str(oname() if callable(oname) else oname or "")
        pal_source = app_instance if app_instance is not None and hasattr(app_instance, "palette") else app_cls
        if pal_source is not None and hasattr(pal_source, "palette"):
            pal = pal_source.palette() if callable(pal_source.palette) else pal_source.palette
            win_name = str(pal.color(qt.QPalette.Window).name())
            txt_name = str(pal.color(qt.QPalette.WindowText).name())
    except Exception:
        pass
    return (mode, style_name, win_name, txt_name)


def apply_placeholder_palette(root_widget: Any, *, dark: bool | None = None) -> None:
    """Ensure root background, checkboxes, and QLineEdit placeholders have crisp contrast in Light and Dark themes."""

    if root_widget is None:
        return
    try:
        import qt

        use_dark = is_dark_mode(root_widget) if dark is None else bool(dark)
        core_app = getattr(qt, "QCoreApplication", None) or getattr(qt, "QApplication", None)
        hi_dpi_attr = getattr(getattr(qt, "Qt", None), "AA_UseHighDpiPixmaps", None)
        if core_app is not None and hi_dpi_attr is not None and hasattr(core_app, "setAttribute"):
            core_app.setAttribute(hi_dpi_attr, True)
        styled_bg = getattr(qt.Qt, "WA_StyledBackground", None)
        if styled_bg is not None and hasattr(root_widget, "setAttribute"):
            root_widget.setAttribute(styled_bg, True)

        win_color = qt.QColor("#303030" if use_dark else "#FFFFFF")
        win_text_color = qt.QColor("#E6EDF3" if use_dark else "#1E293B")
        placeholder_color = qt.QColor("#9BA7B4" if use_dark else "#6E7B85")
        text_color = qt.QColor("#F0F6FC" if use_dark else "#1E293B")
        base_color = qt.QColor("#22272E" if use_dark else "#FFFFFF")
        btn_color = qt.QColor("#2D333B" if use_dark else "#F8FAFB")
        role_placeholder = getattr(qt.QPalette, "PlaceholderText", None)

        if hasattr(root_widget, "setPalette") and hasattr(root_widget, "palette"):
            rpal_src = root_widget.palette() if callable(root_widget.palette) else root_widget.palette
            rpal = qt.QPalette(rpal_src)
            rpal.setColor(qt.QPalette.Window, win_color)
            rpal.setColor(qt.QPalette.WindowText, win_text_color)
            rpal.setColor(qt.QPalette.Base, base_color)
            rpal.setColor(qt.QPalette.Text, text_color)
            rpal.setColor(qt.QPalette.Button, btn_color)
            rpal.setColor(qt.QPalette.ButtonText, win_text_color)
            root_widget.setPalette(rpal)

        edits = [root_widget] if isinstance(root_widget, qt.QLineEdit) else []
        if hasattr(root_widget, "findChildren"):
            edits.extend(root_widget.findChildren(qt.QLineEdit))
        for edit in edits:
            epal_src = edit.palette() if callable(edit.palette) else edit.palette
            pal = qt.QPalette(epal_src)
            pal.setColor(qt.QPalette.Text, text_color)
            pal.setColor(qt.QPalette.Base, base_color)
            if role_placeholder is not None:
                pal.setColor(role_placeholder, placeholder_color)
            edit.setPalette(pal)
    except Exception:
        pass


SLICERDSPACE_UI_STYLE = """
qMRMLWidget#DDSurfer {
  background-color: #FFFFFF;
  color: #1E293B;
}
QDialog {
  background: #FFFFFF;
  color: #1E293B;
}
QFrame#headerFrame {
  background: transparent;
  border: none;
  border-bottom: 1px solid #D8DEE3;
  border-radius: 0;
}
QLabel#headlineLabel {
  color: #24343D;
  font-size: 17px;
  font-weight: 600;
}
ctkCollapsibleButton {
  background: transparent;
  border: none;
  color: #1E293B;
}
ctkCollapsibleGroupBox, QGroupBox {
  color: #1E293B;
}
QLabel {
  color: #1E293B;
  background: transparent;
}
QCheckBox, QRadioButton {
  color: #1E293B;
  background: transparent;
  spacing: 7px;
}
QCheckBox::indicator {
  width: 14px;
  height: 14px;
  border: 1px solid #94A3B8;
  border-radius: 3px;
  background: #FFFFFF;
}
QCheckBox::indicator:hover:!disabled {
  border-color: #335B70;
}
QCheckBox::indicator:checked {
  background: #335B70;
  border: 1px solid #335B70;
  image: url("__CHECKMARK_ICON_URL__");
}
QCheckBox::indicator:disabled {
  background: #F1F5F9;
  border-color: #CBD5E1;
}
QLabel:disabled, QCheckBox:disabled, QRadioButton:disabled {
  color: #8A98A3;
}
QWidget#externalBackendsGroup,
ctkCollapsibleButton#externalBackendsGroup,
QGroupBox#externalBackendsGroup {
  background: #F8FAFB;
  border: 1px solid #D8DEE3;
  border-radius: 4px;
  color: #1E293B;
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
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
  background: #F2F5F7;
  color: #8A98A3;
  border: 1px solid #DCE3E8;
}
QLineEdit:read-only {
  background: #F6F8FA;
  color: #52616D;
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
QListWidget, QPlainTextEdit {
  background: #FFFFFF;
  color: #1E293B;
  border: 1px solid #D8DEE3;
  border-radius: 4px;
  selection-background-color: #D4E2EA;
  selection-color: #1E293B;
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
QToolButton {
  background: #F8FAFB;
  border: 1px solid #CCD5DC;
  border-radius: 3px;
  color: #334550;
  padding: 2px 6px;
  min-height: 18px;
}
QToolButton:hover:!disabled {
  background: #EEF2F5;
  border-color: #476E83;
}
QToolButton:disabled {
  color: #8A98A3;
  background: #F2F4F6;
  border-color: #DCE3E8;
}
QLineEdit QToolButton,
QLineEdit QToolButton:hover,
QLineEdit QToolButton:pressed,
QLineEdit QToolButton:disabled {
  background: transparent;
  border: none;
  border-radius: 0;
  padding: 0 0 2px 0;
  margin: 0;
  min-width: 0;
  min-height: 0;
}
QDialog[popupCard="true"] {
  background: #FFFFFF;
  color: #1E293B;
  border: 1px solid #9BAEBB;
  border-radius: 6px;
}
QLabel#algorithmPickerTitle {
  color: #1E293B;
  font-size: 12px;
  font-weight: 700;
  background: transparent;
  border: none;
}
QLabel#algorithmPickerNote {
  color: #63727D;
  font-size: 11px;
  background: transparent;
  border: none;
}
QToolButton#algorithmPickerCloseButton {
  color: #63727D;
  background: transparent;
  border: 1px solid transparent;
  border-radius: 4px;
  font-size: 14px;
  font-weight: 700;
  padding: 0;
  min-width: 20px;
  max-width: 20px;
  min-height: 20px;
  max-height: 20px;
}
QToolButton#algorithmPickerCloseButton:hover {
  color: #1E293B;
  background: #EEF2F5;
  border-color: #CCD5DC;
}
QToolButton#preprocessingAlgorithmSelectButton,
QToolButton#microstructureAlgorithmSelectButton,
QToolButton#tractographyAlgorithmSelectButton,
QToolButton#bundlesAlgorithmSelectButton,
QToolButton#connectomeAlgorithmSelectButton {
  color: #335B70;
  background: #FFFFFF;
  border: 1px solid #CCD5DC;
  border-radius: 3px;
  font-weight: 600;
  padding: 0;
  min-height: 18px;
}
QToolButton#inputFormatHelpButton {
  color: #335B70;
  background: #F1F5F8;
  border: 1px solid #C2D0DA;
  border-radius: 11px;
  font-weight: 700;
  font-size: 12px;
  padding: 0;
  min-width: 22px;
  max-width: 22px;
  min-height: 22px;
  max-height: 22px;
}
QToolButton#inputFormatHelpButton:hover,
QToolButton#inputFormatHelpButton[pinned="true"] {
  background: #E2ECF2;
  border-color: #476E83;
}
QToolTip {
  background-color: #FFFFFF;
  color: #1E293B;
  border: 1px solid #9BAEBB;
  padding: 6px 8px;
}
QStackedWidget#advancedCategoryStack { background: transparent; border: none; }
QPushButton:disabled {
  color: #8A98A3;
  background: #F2F4F6;
  border-color: #DCE3E8;
}
QPushButton#runSelectedButton {
  background: #335B70;
  border: 1px solid #28495C;
  border-radius: 5px;
  color: #FFFFFF;
  font-family: "Segoe UI", "Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", sans-serif;
  font-size: 15px;
  font-weight: 700;
  padding: 4px 14px;
  min-height: 22px;
}
QPushButton#managedRuntimeSetupButton {
  background: #EEF5F9;
  border: 1px solid #8EB1C7;
  border-radius: 4px;
  color: #1F4458;
  font-weight: 600;
  padding: 5px 10px;
}
QPushButton#managedRuntimeSetupButton:hover:!disabled {
  background: #E0EDF5;
  border-color: #476E83;
}
QPushButton#runSelectedButton:hover:!disabled {
  background: #28495C;
  border-color: #1F3A4A;
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
  border: 1px solid #D8DEE3;
  border-radius: 3px;
  color: #334550;
  max-height: 16px;
  text-align: center;
}
QProgressBar::chunk {
  background: #9EB8C8;
  border-radius: 3px;
}
QWidget#runStageStrip {
  background: transparent;
  border: none;
}
QProgressBar#runProgressBar {
  background: #DCE4EC;
  border: none;
  border-radius: 3px;
  min-height: 6px;
  max-height: 6px;
}
QProgressBar#runProgressBar::chunk {
  background: #0D9488;
  border-radius: 3px;
}
QLabel#runStageCountLabel {
  background: transparent;
  border: none;
  color: #0F766E;
  font-size: 11px;
  font-weight: 600;
  padding: 0 2px;
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
qMRMLWidget#DDSurfer {
  background-color: #303030;
  color: #E6EDF3;
}
QDialog {
  background: #22272E;
  color: #E6EDF3;
}
QFrame#headerFrame {
  background: transparent;
  border: none;
  border-bottom: 1px solid #444C56;
  border-radius: 0;
}
QLabel#headlineLabel {
  color: #F0F6FC;
  font-size: 17px;
  font-weight: 600;
}
ctkCollapsibleButton {
  background: transparent;
  border: none;
  color: #E6EDF3;
}
ctkCollapsibleGroupBox, QGroupBox {
  color: #E6EDF3;
}
QLabel {
  color: #E6EDF3;
  background: transparent;
}
QCheckBox, QRadioButton {
  color: #E6EDF3;
  background: transparent;
  spacing: 7px;
}
QCheckBox::indicator {
  width: 14px;
  height: 14px;
  border: 1px solid #6E7B88;
  border-radius: 3px;
  background: #22272E;
}
QCheckBox::indicator:hover:!disabled {
  border-color: #7CC4EA;
}
QCheckBox::indicator:checked {
  background: #3B7390;
  border: 1px solid #4E8BAE;
  image: url("__CHECKMARK_ICON_URL__");
}
QCheckBox::indicator:disabled {
  background: #1C2026;
  border-color: #3B424B;
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
QToolButton {
  background: #323842;
  border: 1px solid #525C68;
  border-radius: 3px;
  color: #E6EDF3;
  padding: 2px 6px;
  min-height: 18px;
}
QToolButton:hover:!disabled {
  background: #3C4450;
  border-color: #7CC4EA;
}
QToolButton:disabled {
  color: #7D8590;
  background: #252A31;
  border-color: #3B424B;
}
QLineEdit QToolButton,
QLineEdit QToolButton:hover,
QLineEdit QToolButton:pressed,
QLineEdit QToolButton:disabled {
  background: transparent;
  border: none;
  border-radius: 0;
  padding: 0 0 2px 0;
  margin: 0;
  min-width: 0;
  min-height: 0;
}
QDialog[popupCard="true"] {
  background: #1C2128;
  color: #E6EDF3;
  border: 1px solid #525C68;
  border-radius: 6px;
}
QLabel#algorithmPickerTitle {
  color: #F0F6FC;
  font-size: 12px;
  font-weight: 700;
  background: transparent;
  border: none;
}
QLabel#algorithmPickerNote {
  color: #B8C5D0;
  font-size: 11px;
  background: transparent;
  border: none;
}
QToolButton#algorithmPickerCloseButton {
  color: #9BA7B4;
  background: transparent;
  border: 1px solid transparent;
  border-radius: 4px;
  font-size: 14px;
  font-weight: 700;
  padding: 0;
  min-width: 20px;
  max-width: 20px;
  min-height: 20px;
  max-height: 20px;
}
QToolButton#algorithmPickerCloseButton:hover {
  color: #FFFFFF;
  background: #323842;
  border-color: #636E7B;
}
QToolButton#preprocessingAlgorithmSelectButton,
QToolButton#microstructureAlgorithmSelectButton,
QToolButton#tractographyAlgorithmSelectButton,
QToolButton#bundlesAlgorithmSelectButton,
QToolButton#connectomeAlgorithmSelectButton {
  color: #8CD2F5;
  background: #323842;
  border: 1px solid #525C68;
  border-radius: 3px;
  font-weight: 600;
  padding: 0;
  min-height: 18px;
}
QToolButton#inputFormatHelpButton {
  color: #8CD2F5;
  background: #2C333C;
  border: 1px solid #525C68;
  border-radius: 11px;
  font-weight: 700;
  font-size: 12px;
  padding: 0;
  min-width: 22px;
  max-width: 22px;
  min-height: 22px;
  max-height: 22px;
}
QToolButton#inputFormatHelpButton:hover,
QToolButton#inputFormatHelpButton[pinned="true"] {
  background: #3C4450;
  border-color: #7CC4EA;
}
QToolTip {
  background-color: #22272E;
  color: #F0F6FC;
  border: 1px solid #525C68;
  padding: 6px 8px;
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
  border-radius: 5px;
  color: #FFFFFF;
  font-family: "Segoe UI", "Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", sans-serif;
  font-size: 15px;
  font-weight: 700;
  padding: 4px 14px;
  min-height: 22px;
}
QPushButton#managedRuntimeSetupButton {
  background: #223544;
  border: 1px solid #4889AB;
  border-radius: 4px;
  color: #E6EDF3;
  font-weight: 600;
  padding: 5px 10px;
}
QPushButton#managedRuntimeSetupButton:hover:!disabled {
  background: #2B4558;
  border-color: #7CC4EA;
}
QPushButton#runSelectedButton:hover:!disabled {
  background: #397CA0;
  border-color: #63A4C7;
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
QWidget#runStageStrip {
  background: transparent;
  border: none;
}
QProgressBar#runProgressBar {
  background: #1C2128;
  border: none;
  border-radius: 3px;
  min-height: 6px;
  max-height: 6px;
}
QProgressBar#runProgressBar::chunk {
  background: #14B8A6;
  border-radius: 3px;
}
QLabel#runStageCountLabel {
  background: transparent;
  border: none;
  color: #5EEAD4;
  font-size: 11px;
  font-weight: 600;
  padding: 0 2px;
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

SLICERDSPACE_UI_STYLE = SLICERDSPACE_UI_STYLE.replace("__CHECKMARK_ICON_URL__", _CHECKMARK_ICON_URL)
SLICERDSPACE_DARK_UI_STYLE = SLICERDSPACE_DARK_UI_STYLE.replace("__CHECKMARK_ICON_URL__", _CHECKMARK_ICON_URL)

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

INPUT_FORMAT_POPOVER_STYLE = """
QFrame#inputFormatHelpPopover {
  background-color: #FFFFFF;
  border: 1px solid #9BAEBB;
  border-radius: 6px;
}
QLabel#inputFormatHelpTitle {
  color: #1E293B;
  background: transparent;
  border: none;
  font-size: 12px;
  font-weight: 700;
}
QLabel#inputFormatHelpBody {
  color: #334550;
  background: transparent;
  border: none;
  font-size: 12px;
}
QToolButton#inputFormatHelpCloseButton {
  color: #52616D;
  background: transparent;
  border: 1px solid transparent;
  border-radius: 4px;
  font-size: 14px;
  font-weight: 700;
  padding: 0;
  min-width: 20px;
  max-width: 20px;
  min-height: 20px;
  max-height: 20px;
}
QToolButton#inputFormatHelpCloseButton:hover {
  color: #1E293B;
  background: #EEF2F5;
  border-color: #CCD5DC;
}
"""

DARK_INPUT_FORMAT_POPOVER_STYLE = """
QFrame#inputFormatHelpPopover {
  background-color: #1C2128;
  border: 1px solid #525C68;
  border-radius: 6px;
}
QLabel#inputFormatHelpTitle {
  color: #F0F6FC;
  background: transparent;
  border: none;
  font-size: 12px;
  font-weight: 700;
}
QLabel#inputFormatHelpBody {
  color: #D2DEE8;
  background: transparent;
  border: none;
  font-size: 12px;
}
QToolButton#inputFormatHelpCloseButton {
  color: #9BA7B4;
  background: transparent;
  border: 1px solid transparent;
  border-radius: 4px;
  font-size: 14px;
  font-weight: 700;
  padding: 0;
  min-width: 20px;
  max-width: 20px;
  min-height: 20px;
  max-height: 20px;
}
QToolButton#inputFormatHelpCloseButton:hover {
  color: #FFFFFF;
  background: #323842;
  border-color: #636E7B;
}
"""

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


def input_format_popover_style(*, dark: bool | None = None, widget: Any = None) -> str:
    """Return the stylesheet for the Input format '?' hover/pinned popover card."""

    use_dark = is_dark_mode(widget) if dark is None else bool(dark)
    return DARK_INPUT_FORMAT_POPOVER_STYLE if use_dark else INPUT_FORMAT_POPOVER_STYLE


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


RUN_STAGE_INDICATOR_PALETTE: dict[str, dict[str, str]] = {
    "running": {"icon_bg": "#0D9488", "chunk": "#0D9488", "groove": "#DCE4EC", "text": "#0F766E", "glyph": "run"},
    "preflight": {"icon_bg": "#0D9488", "chunk": "#0D9488", "groove": "#DCE4EC", "text": "#0F766E", "glyph": "run"},
    "completed": {"icon_bg": "#16A34A", "chunk": "#16A34A", "groove": "#DCE4EC", "text": "#15803D", "glyph": "check"},
    "completed_with_blocks": {
        "icon_bg": "#D97706",
        "chunk": "#D97706",
        "groove": "#DCE4EC",
        "text": "#B45309",
        "glyph": "warn",
    },
    "cancelled": {"icon_bg": "#D97706", "chunk": "#D97706", "groove": "#DCE4EC", "text": "#B45309", "glyph": "warn"},
    "failed": {"icon_bg": "#DC2626", "chunk": "#DC2626", "groove": "#DCE4EC", "text": "#B91C1C", "glyph": "error"},
}

DARK_RUN_STAGE_INDICATOR_PALETTE: dict[str, dict[str, str]] = {
    "running": {"icon_bg": "#14B8A6", "chunk": "#14B8A6", "groove": "#1C2128", "text": "#5EEAD4", "glyph": "run"},
    "preflight": {"icon_bg": "#14B8A6", "chunk": "#14B8A6", "groove": "#1C2128", "text": "#5EEAD4", "glyph": "run"},
    "completed": {"icon_bg": "#2EA043", "chunk": "#2EA043", "groove": "#1C2128", "text": "#56D364", "glyph": "check"},
    "completed_with_blocks": {
        "icon_bg": "#D9822B",
        "chunk": "#E09132",
        "groove": "#1C2128",
        "text": "#F5B74F",
        "glyph": "warn",
    },
    "cancelled": {"icon_bg": "#D9822B", "chunk": "#E09132", "groove": "#1C2128", "text": "#F5B74F", "glyph": "warn"},
    "failed": {"icon_bg": "#E5534B", "chunk": "#E5534B", "groove": "#1C2128", "text": "#FF7B72", "glyph": "error"},
}


def run_stage_indicator_palette(state: str, *, dark: bool | None = None, widget: Any = None) -> dict[str, str]:
    """Return semantic colours and glyph type for the stage indicator strip."""

    use_dark = is_dark_mode(widget) if dark is None else bool(dark)
    source = DARK_RUN_STAGE_INDICATOR_PALETTE if use_dark else RUN_STAGE_INDICATOR_PALETTE
    key = (state or "").strip()
    return dict(source.get(key, source["running"]))


def run_progress_bar_style(state: str, *, dark: bool | None = None, widget: Any = None) -> str:
    """Return the slim pill-track stylesheet for ``QProgressBar#runProgressBar``."""

    pal = run_stage_indicator_palette(state, dark=dark, widget=widget)
    return (
        "QProgressBar#runProgressBar { "
        f"background: {pal['groove']}; border: none; border-radius: 3px; min-height: 6px; max-height: 6px; "
        "} "
        "QProgressBar#runProgressBar::chunk { "
        f"background: {pal['chunk']}; border-radius: 3px; "
        "}"
    )


def run_stage_count_label_style(state: str, *, dark: bool | None = None, widget: Any = None) -> str:
    """Return the semantic text stylesheet for ``QLabel#runStageCountLabel``."""

    pal = run_stage_indicator_palette(state, dark=dark, widget=widget)
    return (
        "QLabel#runStageCountLabel { "
        f"background: transparent; border: none; color: {pal['text']}; "
        "font-size: 11px; font-weight: 600; padding: 0 2px; "
        "}"
    )

