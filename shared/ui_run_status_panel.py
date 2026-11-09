"""Qt helpers for the SlicerDSpace run-status panel."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

MAX_VISIBLE_ROWS = 8
# Manifest, QC and Logs paths stay in the table as hidden columns 3 to 5.
MANIFEST_COLUMN, QC_COLUMN, LOGS_COLUMN = 3, 4, 5


def _value(attribute):
    """PythonQt exposes many Qt getters as properties; plain Qt exposes them as methods."""

    return attribute() if callable(attribute) else attribute


def row_path_tooltip(manifest: str, qc: str, logs: str) -> str:
    """Tooltip for a row's Target cell: the paths the row has, one per line."""

    lines = [f"{label}: {value}" for label, value in (("Manifest", manifest), ("QC", qc), ("Logs", logs)) if value]
    return "\n".join(lines)


def first_existing_path(qc: str, logs: str, manifest: str) -> Optional[Path]:
    """The path a double-click on a row opens: QC file, else logs, else manifest, whichever exists first."""

    for text in (qc, logs, manifest):
        text = str(text or "").strip()
        if text and Path(text).expanduser().exists():
            return Path(text).expanduser()
    return None


def _cell_text(table, row: int, column: int) -> str:
    item = table.item(row, column)
    return str(_value(item.text) if item else "").strip()


def row_open_path(table, row: int) -> Optional[Path]:
    return first_existing_path(
        _cell_text(table, row, QC_COLUMN), _cell_text(table, row, LOGS_COLUMN), _cell_text(table, row, MANIFEST_COLUMN)
    )


def table_height(rows: int, header_height: int, row_height: int, frame_width: int) -> int:
    """Height that shows 1 to MAX_VISIBLE_ROWS rows; more rows scroll vertically."""

    shown = max(1, min(int(rows), MAX_VISIBLE_ROWS))
    return header_height + shown * row_height + 2 * frame_width


def fit_table_height(table) -> None:
    """Fix the table's height to its rows, so no empty rows show below a short result."""

    table.setFixedHeight(table_height(
        _value(table.rowCount),
        _value(table.horizontalHeader().sizeHint).height(),
        _value(table.verticalHeader().defaultSectionSize),
        _value(table.frameWidth),
    ))


def install_run_status_panel(widget) -> None:
    """Create the run progress area (4. Run) and the results panel (5. Results).

    Kept as the historical entry point; the widgets are built by
    ``shared.ui_run_history`` so the run selector, stage table and scene
    actions live together under 5. Results.
    """

    if getattr(widget, "_runStatusTable", None) is not None:
        return
    from shared.ui_run_history import install_results_panel, install_run_progress_panel

    install_run_progress_panel(widget)
    install_results_panel(widget)


def table_item(text: str, state: str = ""):
    """Return a styled run-status table item."""

    import qt
    from shared.ui_styles import is_dark_mode

    item = qt.QTableWidgetItem(text or "")
    if text:
        item.setToolTip(text)
    if state:
        dark = is_dark_mode()
        colors = (
            {
                "queued": qt.QColor("#9BA7B4"),
                "running": qt.QColor("#64B5F6"),
                "will_run": qt.QColor("#64B5F6"),
                "completed": qt.QColor("#66BB6A"),
                "passed": qt.QColor("#66BB6A"),
                "skipped": qt.QColor("#9BA7B4"),
                "blocked": qt.QColor("#FF7B72"),
                "failed": qt.QColor("#FF7B72"),
                "cancelled": qt.QColor("#FFB74D"),
            }
            if dark
            else {
                "queued": qt.QColor("#546E7A"),
                "running": qt.QColor("#1565C0"),
                "will_run": qt.QColor("#1565C0"),
                "completed": qt.QColor("#2E7D32"),
                "passed": qt.QColor("#2E7D32"),
                "skipped": qt.QColor("#546E7A"),
                "blocked": qt.QColor("#C62828"),
                "failed": qt.QColor("#C62828"),
                "cancelled": qt.QColor("#EF6C00"),
            }
        )
        color = colors.get(state)
        if color:
            item.setForeground(color)
    return item


def set_run_status_table_rows(widget, rows: list[dict[str, str]]) -> None:
    """Replace all run-status table rows."""

    table = getattr(widget, "_runStatusTable", None)
    if not table:
        return
    try:
        table.setUpdatesEnabled(False)
    except Exception:
        pass
    try:
        table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            state = row.get("status", "")
            values = [
                row.get("target") or row.get("name") or "",
                state,
                row.get("reason", ""),
                row.get("manifest_path", ""),
                row.get("qc_path", ""),
                row.get("logs_dir", ""),
            ]
            paths = row_path_tooltip(*(str(values[col] or "") for col in (MANIFEST_COLUMN, QC_COLUMN, LOGS_COLUMN)))
            for col, value in enumerate(values):
                item = table_item(str(value or ""), state)
                if col == 0 and paths:
                    item.setToolTip(paths)
                table.setItem(row_index, col, item)
        fit_table_height(table)
    finally:
        try:
            table.setUpdatesEnabled(True)
        except Exception:
            pass


def update_run_status_table_item(widget, key_candidates: tuple[str, ...], status: str, reason: str = "") -> None:
    """Update one existing run-status table row matching a target key."""

    table = getattr(widget, "_runStatusTable", None)
    if not table:
        return
    keys = {key for key in key_candidates if key}
    for row in range(table.rowCount):
        target_item = table.item(row, 0)
        if not target_item:
            continue
        target = str(target_item.text() if callable(getattr(target_item, "text", None)) else target_item.text)
        if target not in keys:
            continue
        try:
            table.setUpdatesEnabled(False)
        except Exception:
            pass
        try:
            table.setItem(row, 1, table_item(status, status))
            if reason:
                table.setItem(row, 2, table_item(reason, status))
        finally:
            try:
                table.setUpdatesEnabled(True)
            except Exception:
                pass
        return


def mark_unfinished_rows(widget, status: str, reason: str) -> None:
    """Mark queued/running table rows as failed or cancelled."""

    table = getattr(widget, "_runStatusTable", None)
    if not table:
        return
    try:
        table.setUpdatesEnabled(False)
    except Exception:
        pass
    try:
        for row in range(table.rowCount):
            status_item = table.item(row, 1)
            current = str(status_item.text() if status_item else "")
            if current in {"completed", "passed", "skipped", "blocked", "failed", "cancelled"}:
                continue
            table.setItem(row, 1, table_item(status, status))
            table.setItem(row, 2, table_item(reason, status))
    finally:
        try:
            table.setUpdatesEnabled(True)
        except Exception:
            pass
