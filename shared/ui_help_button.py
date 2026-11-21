"""b2 inputs-help: one look and one behaviour for the small "?" help buttons of the Prepare Inputs panel.

A "?" is a flat round tool button that the keyboard can reach and that has an accessible name. A hover shows the help
text as a tooltip. A click shows the same text in one shared small popup (a ``Qt.Popup`` frame, no modal dialog) that
closes on a click outside or Esc. The texts live in ``shared/ui_onboarding.py`` (``help_<key>`` and ``help_<key>_name``,
English and Chinese); the buttons and the open popup follow the language and the theme.

Real PythonQt could differ from the fake ``qt`` of the unit tests: every optional Qt name is read with ``getattr``.
"""

from __future__ import annotations

from typing import Any

from shared.ui_onboarding import onboarding_text

BUTTON_SIZE = 18
BUTTON_NAME = "dspaceHelpMarker"
POPUP_NAME = "dspaceHelpPopup"
# The ink of the "?" is the one of the algorithm picker's unavailable marker (shared/ui_algorithm_picker.py).
_INK = {True: "#7D8590", False: "#64748B"}
_INK_HOVER = {True: "#B4BCC6", False: "#334550"}
_POPUP = {
    True: ("#1C2128", "#525C68", "#D2DEE8"),
    False: ("#FFFFFF", "#9BAEBB", "#334550"),
}


def _language(owner: Any) -> str:
    return "zh" if str(getattr(owner, "_onboardingLanguage", "en")).startswith("zh") else "en"


def _is_dark(owner: Any, dark: bool | None) -> bool:
    if dark is not None:
        return bool(dark)
    from shared.ui_styles import is_dark_mode

    return bool(is_dark_mode(getattr(owner, "_uiWidget", None)))


def help_text(owner: Any, key: str) -> str:
    """The rich text of one "?" in the owner's language (the tooltip and the popup show this same string)."""

    return onboarding_text(_language(owner), "help_" + key)


def help_marker_style(dark: bool) -> str:
    ink, hover = _INK[dark], _INK_HOVER[dark]
    return (
        f"QToolButton#{BUTTON_NAME} {{ color: {ink}; border: 1px solid {ink}; border-radius: 9px; "
        "font-size: 11px; font-weight: 600; background: transparent; padding: 0px; }"
        f"QToolButton#{BUTTON_NAME}:hover {{ color: {hover}; border-color: {hover}; }}"
        f"QToolButton#{BUTTON_NAME}:focus {{ color: {hover}; border: 2px solid {hover}; }}"
    )


def help_popup_style(dark: bool) -> str:
    background, border, ink = _POPUP[dark]
    return (
        f"QFrame#{POPUP_NAME} {{ background: {background}; border: 1px solid {border}; border-radius: 6px; }}"
        f"QFrame#{POPUP_NAME} QLabel {{ color: {ink}; background: transparent; border: none; }}"
    )


def make_help_button(owner: Any, qt: Any, parent: Any, key: str) -> Any:
    """One "?" button for help ``key`` (``subject_id``, ``batch_folder`` or ``output_folder``); registered on the owner."""

    button = qt.QToolButton(parent) if parent is not None else qt.QToolButton()
    button.setObjectName(BUTTON_NAME)
    button.setText("?")
    button.setAutoRaise(True)
    button.setFixedSize(BUTTON_SIZE, BUTTON_SIZE)
    button.setCursor(qt.Qt.PointingHandCursor)
    button.setFocusPolicy(qt.Qt.StrongFocus)
    button.setStyleSheet(help_marker_style(_is_dark(owner, None)))
    button.clicked.connect(lambda _checked=False, key=key: show_help_popup(owner, key))
    buttons = getattr(owner, "_helpButtons", None)
    if not isinstance(buttons, dict):
        buttons = {}
        owner._helpButtons = buttons
    buttons[key] = button
    _apply_button_text(owner, key, button)
    return button


def _apply_button_text(owner: Any, key: str, button: Any) -> None:
    lang = _language(owner)
    button.setToolTip(onboarding_text(lang, "help_" + key))
    button.setAccessibleName(onboarding_text(lang, "help_" + key + "_name"))


def apply_help_language(owner: Any) -> None:
    """Tooltips and accessible names in the current language; an open popup shows the new language at once."""

    for key, button in (getattr(owner, "_helpButtons", None) or {}).items():
        _apply_button_text(owner, key, button)
    popup = getattr(owner, "_helpPopup", None)
    label = getattr(owner, "_helpPopupLabel", None)
    key = getattr(owner, "_helpPopupKey", None)
    if popup is not None and label is not None and key:
        label.setText(help_text(owner, key))


def refresh_help_styles(owner: Any, *, dark: bool | None = None) -> None:
    use_dark = _is_dark(owner, dark)
    for button in (getattr(owner, "_helpButtons", None) or {}).values():
        button.setStyleSheet(help_marker_style(use_dark))
    popup = getattr(owner, "_helpPopup", None)
    if popup is not None:
        popup.setStyleSheet(help_popup_style(use_dark))


def _make_popup(owner: Any, qt: Any, anchor: Any) -> tuple[Any, Any]:
    window = anchor.window() if hasattr(anchor, "window") else None
    popup = qt.QFrame(window, qt.Qt.Popup)
    popup.setObjectName(POPUP_NAME)
    no_replay = getattr(qt.Qt, "WA_NoMouseReplay", None)
    if no_replay is not None and hasattr(popup, "setAttribute"):
        popup.setAttribute(no_replay, True)  # the click that closes the popup must not reach the "?" and reopen it
    layout = qt.QVBoxLayout(popup)
    layout.setContentsMargins(10, 8, 10, 8)
    label = qt.QLabel(popup)
    label.setTextFormat(qt.Qt.RichText)
    label.setWordWrap(True)
    layout.addWidget(label)
    owner._helpPopup = popup
    owner._helpPopupLabel = label
    popup.setStyleSheet(help_popup_style(_is_dark(owner, None)))
    return popup, label


def show_help_popup(owner: Any, key: str) -> None:
    """Show help ``key`` in the shared popup under its button; a second click on the same button closes it."""

    import qt
    from shared.ui_compact import _position_input_help_card

    button = (getattr(owner, "_helpButtons", None) or {}).get(key)
    if button is None:
        return
    popup = getattr(owner, "_helpPopup", None)
    if popup is not None and getattr(owner, "_helpPopupKey", None) == key and popup.isVisible():
        popup.hide()
        return
    if popup is None:
        popup, label = _make_popup(owner, qt, button)
    else:
        label = owner._helpPopupLabel
        popup.hide()
    if hasattr(qt, "QToolTip") and hasattr(qt.QToolTip, "hideText"):
        qt.QToolTip.hideText()
    label.setText(help_text(owner, key))
    owner._helpPopupKey = key
    popup.setStyleSheet(help_popup_style(_is_dark(owner, None)))
    _position_input_help_card(button, popup, qt)
    popup.show()
    if hasattr(popup, "raise_"):
        popup.raise_()


def close_help_popup(owner: Any, *, destroy: bool = False) -> None:
    popup = getattr(owner, "_helpPopup", None)
    if popup is None:
        return
    try:
        popup.hide()
    except Exception:
        pass
    if destroy:
        try:
            popup.deleteLater()
        except Exception:
            pass
        owner._helpPopup = None
        owner._helpPopupLabel = None
        owner._helpPopupKey = None
