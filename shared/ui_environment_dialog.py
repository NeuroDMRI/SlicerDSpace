"""Qt builders for the SlicerDSpace environment-results dialog."""

from __future__ import annotations

from typing import Any, Callable

from shared.ui_environment_panel import (
    environment_dialog_state,
    environment_item_button_text,
    environment_item_button_tooltip,
    environment_item_text,
)
from shared.ui_styles import (
    ENVIRONMENT_READY_LABEL_STYLE,
    ENVIRONMENT_RESULTS_DIALOG_STYLE,
    environment_ready_label_style,
    environment_results_dialog_style,
)


def build_environment_item_row(
    item: Any,
    *,
    on_install_item: Callable[[Any], None],
):
    """Build one missing-dependency row for the environment dialog."""

    import qt

    frame = qt.QFrame()
    frame.setObjectName("environmentItemFrame")
    frame.setFrameShape(qt.QFrame.StyledPanel)
    layout = qt.QHBoxLayout(frame)
    layout.setContentsMargins(14, 12, 14, 12)
    layout.setSpacing(14)

    text = qt.QLabel(environment_item_text(item))
    text.setObjectName("environmentItemText")
    text.wordWrap = True
    try:
        text.setTextInteractionFlags(qt.Qt.TextSelectableByMouse | qt.Qt.TextSelectableByKeyboard)
    except Exception:
        pass
    layout.addWidget(text, 1)

    button = qt.QPushButton(environment_item_button_text(item))
    button.setObjectName("environmentItemActionButton")
    button.toolTip = environment_item_button_tooltip(item)
    button.connect("clicked()", lambda: on_install_item(item))
    layout.addWidget(button)
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

    dialog = qt.QDialog(slicer.util.mainWindow())
    dialog.setWindowTitle("SlicerDSpace Environment Dependencies")
    dialog.resize(900, 640)
    dialog.setStyleSheet(environment_results_dialog_style(widget=dialog))
    owner._environmentDialog = dialog
    state = environment_dialog_state(result, python_install_target=python_install_target)

    layout = qt.QVBoxLayout(dialog)
    layout.setContentsMargins(16, 16, 16, 16)
    layout.setSpacing(12)
    summary_label = qt.QLabel(state.summary_text)
    summary_label.setObjectName("environmentSummaryLabel")
    summary_label.wordWrap = True
    layout.addWidget(summary_label)

    button_row = qt.QHBoxLayout()
    install_all_button = qt.QPushButton("Prepare selected environment")
    install_all_button.setObjectName("environmentInstallAllButton")
    install_all_button.toolTip = state.python_install_tooltip
    install_all_button.enabled = state.can_install_python
    install_all_button.connect("clicked()", on_install_all)
    button_row.addWidget(install_all_button)

    notes_button = qt.QPushButton("Manual setup notes")
    notes_button.setObjectName("environmentManualNotesButton")
    notes_button.enabled = state.has_manual_setup_notes
    notes_button.connect("clicked()", on_show_notes)
    button_row.addWidget(notes_button)

    detail_button = qt.QPushButton("Show summary details")
    detail_button.setObjectName("environmentSummaryDetailsButton")
    detail_button.checkable = True
    detail_button.checked = False
    button_row.addWidget(detail_button)
    button_row.addStretch(1)

    close_button = qt.QPushButton("Close")
    close_button.setObjectName("environmentCloseButton")
    close_button.connect("clicked()", dialog.close)
    button_row.addWidget(close_button)
    layout.addLayout(button_row)

    details = qt.QPlainTextEdit()
    details.setReadOnly(True)
    details.visible = False
    details.setMaximumHeight(120)
    details.setPlainText(state.detail_text)
    detail_button.connect("toggled(bool)", details.setVisible)
    detail_button.connect(
        "toggled(bool)",
        lambda checked: detail_button.setText("Hide summary details" if checked else "Show summary details"),
    )
    layout.addWidget(details)

    if state.missing_items:
        scroll = qt.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = qt.QWidget()
        scroll_layout = qt.QVBoxLayout(scroll_widget)
        for item in state.missing_items:
            scroll_layout.addWidget(build_environment_item_row(item, on_install_item=on_install_item))
        scroll_layout.addStretch(1)
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
    else:
        ready_label = qt.QLabel(state.ready_text)
        ready_label.wordWrap = True
        ready_label.setStyleSheet(environment_ready_label_style(widget=dialog))
        layout.addWidget(ready_label)

    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    return dialog
