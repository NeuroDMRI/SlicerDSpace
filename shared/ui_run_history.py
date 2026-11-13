"""Qt glue for the "4. Run" progress area and the "5. Results" run selector.

The pure logic (manifest discovery/validation, progress maths, button rules)
lives in ``shared/run_history.py`` and ``shared/ui_run_state.py``; this module
only builds widgets and moves data between them and the owner widget.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from shared.run_history import (
    ManifestError,
    RunRecord,
    candidate_run_roots,
    discover_runs,
    parse_run_manifest,
    progress_from_manifest,
    remember_directory,
    run_record_brief,
    run_record_details,
    run_record_label,
    status_label,
)
from shared.ui_run_state import (
    SCENE_ACTION_TOOLTIPS,
    progress_bar_format,
    result_actions_state,
    run_state_from_manifest_status,
    run_state_presentation,
)
from shared.ui_run_status_panel import fit_table_height


RUN_HISTORY_SETTING = "SlicerDSpace/UserSettings/runHistoryDirectories"
PROGRESS_POLL_INTERVAL_MS = 2000
ADD_RUN_FOLDER_TEXT = "Add a run folder…"
ADD_RUN_FOLDER_TOOLTIP = (
    "Add any run or output folder (GUI, command-line or batch). The folder is searched three levels deep "
    "for 00_run_metadata/<run>_orchestration_manifest.json and remembered for later sessions."
)
SCENE_ACTIONS = (("Load", "reuse"), ("Replace", "replace"), ("Append", "append"), ("Clear", "clear"))


def _text(owner: Any, key: str, fallback: str) -> str:
    try:
        from shared.ui_onboarding import onboarding_text

        value = onboarding_text(getattr(owner, "_onboardingLanguage", "en"), key)
        return value if value and value != key else fallback
    except Exception:
        return fallback


def _bind(owner: Any, widget: Any, prop: str, key: str) -> None:
    try:
        from shared.ui_onboarding import register_ui_language_binding

        register_ui_language_binding(owner, widget, prop, key)
    except Exception:
        pass


def section_layout(owner: Any, section_name: str, layout_name: str):
    """Return the layout of a workflow section, creating it if the loader dropped it.

    Qt's UI loader may not materialize an empty ``<layout>``, and an empty
    QLayout is falsy in PythonQt, so never test layouts for truthiness.
    """

    layout = getattr(owner.ui, layout_name, None)
    if layout is not None:
        return layout
    section = getattr(owner.ui, section_name, None)
    if section is not None:
        import qt

        layout = section.layout()
        if layout is None:
            layout = qt.QVBoxLayout(section)
        return layout
    return getattr(owner.ui, "configurationLayout", None)


def _layout_or_fallback(owner: Any, name: str):
    sections = {"runLayout": "runCollapsibleButton", "resultsLayout": "resultsCollapsibleButton"}
    return section_layout(owner, sections.get(name, ""), name)


# --------------------------------------------------------------------------- run

_STAGE_ICON_PIXMAP_CACHE: dict[tuple[str, bool], Any] = {}


def _stage_count_text(owner: Any, done: int, total: int, *, state: str, has_data: bool) -> str:
    lang = str(getattr(owner, "_onboardingLanguage", "en") or "en").lower()
    is_zh = lang.startswith("zh")
    if has_data and total > 0:
        if is_zh:
            return f"{done} / {total} 阶段"
        return progress_bar_format(done, total)
    if state == "idle":
        return ""
    if state == "preflight":
        return "环境检查中…" if is_zh else "Checking…"
    if state == "running":
        return "准备运行…" if is_zh else "Preparing…"
    fallback_zh = {
        "completed": "已完成",
        "completed_with_blocks": "部分受阻",
        "cancelled": "已取消",
        "failed": "运行出错",
    }
    if is_zh and state in fallback_zh:
        return fallback_zh[state]
    return run_state_presentation(state).label


def _stage_icon_svg(glyph: str, accent: str, *, dark: bool) -> bytes:
    if glyph == "run":
        fill_hex = "#2DD4BF" if dark else "#0D9488"
        fill_op = "0.18" if dark else "0.13"
        body = (
            f'<circle cx="8" cy="8" r="6.6" fill="{fill_hex}" fill-opacity="{fill_op}" '
            f'stroke="{accent}" stroke-width="1.6"/>'
            f'<polygon points="6.4,5.0 11.2,8.0 6.4,11.0" fill="{fill_hex}"/>'
        )
    elif glyph == "check":
        body = (
            f'<circle cx="8" cy="8" r="7" fill="{accent}"/>'
            '<path d="M4.6 8.3 L7.0 10.7 L11.5 5.5" fill="none" stroke="#FFFFFF" '
            'stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/>'
        )
    elif glyph == "warn":
        body = (
            f'<circle cx="8" cy="8" r="7" fill="{accent}"/>'
            '<line x1="8" y1="4.4" x2="8" y2="8.5" stroke="#FFFFFF" '
            'stroke-width="1.9" stroke-linecap="round"/>'
            '<circle cx="8" cy="11.4" r="1.05" fill="#FFFFFF"/>'
        )
    else:
        body = (
            f'<circle cx="8" cy="8" r="7" fill="{accent}"/>'
            '<path d="M5.4 5.4 L10.6 10.6 M10.6 5.4 L5.4 10.6" fill="none" '
            'stroke="#FFFFFF" stroke-width="1.85" stroke-linecap="round"/>'
        )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 16 16">'
        f"{body}</svg>"
    )
    return svg.encode("utf-8")


def _stage_icon_pixmap(state: str, *, dark: bool) -> Any:
    from shared.ui_styles import run_stage_indicator_palette

    pal = run_stage_indicator_palette(state, dark=dark)
    glyph = pal["glyph"]
    cache_key = (f"{glyph}:{pal['icon_bg']}", bool(dark))
    cached = _STAGE_ICON_PIXMAP_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        import qt

        if not hasattr(qt, "QPixmap") or not hasattr(qt, "QByteArray"):
            return None
        artwork = _stage_icon_svg(glyph, pal["icon_bg"], dark=dark)
        pix = qt.QPixmap()
        if not pix.loadFromData(qt.QByteArray(artwork), "SVG"):
            return None
        pix = pix.scaled(64, 64, qt.Qt.KeepAspectRatio, qt.Qt.SmoothTransformation)
        if hasattr(pix, "setDevicePixelRatio"):
            pix.setDevicePixelRatio(4.0)
        _STAGE_ICON_PIXMAP_CACHE[cache_key] = pix
        return pix
    except Exception:
        return None


def refresh_run_progress_styles(owner: Any) -> None:
    """Apply Light/Dark semantic styling and High-DPI vector icon to the stage indicator strip."""

    progress = getattr(owner, "_runProgressBar", None)
    if progress is None:
        return
    from shared.ui_styles import (
        is_dark_mode,
        run_progress_bar_style,
        run_stage_count_label_style,
        set_stylesheet_if_changed,
    )

    root = getattr(owner, "_uiWidget", None)
    dark = getattr(owner, "_lastAppliedDarkMode", None)
    if dark is None:
        dark = is_dark_mode(root)
    state = str(getattr(owner, "_runState", "idle") or "idle")
    set_stylesheet_if_changed(progress, run_progress_bar_style(state, dark=dark, widget=root))
    count_label = getattr(owner, "_runStageCountLabel", None)
    if count_label is not None:
        set_stylesheet_if_changed(count_label, run_stage_count_label_style(state, dark=dark, widget=root))
        done, total = getattr(owner, "_runProgressCounts", (0, 0))
        has_data = bool(getattr(owner, "_runProgressHasData", False))
        count_label.setText(_stage_count_text(owner, done, total, state=state, has_data=has_data))
    icon_label = getattr(owner, "_runStageIconLabel", None)
    if icon_label is not None:
        pix = _stage_icon_pixmap(state, dark=bool(dark))
        if pix is not None and hasattr(icon_label, "setPixmap"):
            icon_label.setPixmap(pix)


def install_run_progress_panel(owner: Any) -> None:
    """Add run state, per-stage progress strip and stage/message lines to 4. Run."""

    if getattr(owner, "_runProgressBar", None) is not None:
        return
    layout = _layout_or_fallback(owner, "runLayout")
    if layout is None:
        return
    import qt

    state_row = qt.QHBoxLayout()
    state_label = qt.QLabel("Idle")
    state_label.setObjectName("runStateLabel")
    state_label.setToolTip("Run state: Idle, Checking environment, Running, Completed, Completed with errors or Cancelled.")
    state_row.addWidget(state_label)

    stage_strip = qt.QWidget()
    stage_strip.setObjectName("runStageStrip")
    strip_layout = qt.QHBoxLayout(stage_strip)
    strip_layout.setContentsMargins(2, 3, 2, 2)
    strip_layout.setSpacing(8)

    stage_icon = qt.QLabel()
    stage_icon.setObjectName("runStageIconLabel")
    stage_icon.setFixedSize(16, 16)
    strip_layout.addWidget(stage_icon, 0, qt.Qt.AlignVCenter)

    progress = qt.QProgressBar()
    progress.setObjectName("runProgressBar")
    progress.setRange(0, 1)
    progress.setValue(0)
    progress.setTextVisible(False)
    progress.setFormat("No run in progress")
    progress.setToolTip("Finished stages out of the stages this run plans to execute (from the run manifest).")
    progress.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed)
    progress.setFixedHeight(6)
    progress.setMinimumWidth(48)
    strip_layout.addWidget(progress, 1, qt.Qt.AlignVCenter)

    count_label = qt.QLabel("")
    count_label.setObjectName("runStageCountLabel")
    count_label.setToolTip("Finished stages out of the stages this run plans to execute (from the run manifest).")
    count_label.setSizePolicy(qt.QSizePolicy.Maximum, qt.QSizePolicy.Fixed)
    strip_layout.addWidget(count_label, 0, qt.Qt.AlignVCenter)

    state_row.addWidget(stage_strip, 1)
    layout.addLayout(state_row)

    stage_label = qt.QLabel("")
    stage_label.setObjectName("runStageLabel")
    stage_label.wordWrap = True
    stage_label.visible = False
    layout.addWidget(stage_label)

    hint = qt.QLabel("Run plan has not been generated yet.")
    hint.setObjectName("runMessageLabel")
    hint.wordWrap = True
    try:
        hint.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
    except Exception:
        pass
    layout.addWidget(hint)

    timer = qt.QTimer()
    timer.setInterval(PROGRESS_POLL_INTERVAL_MS)
    timer.connect("timeout()", owner._poll_run_progress)

    owner._runStateLabel = state_label
    owner._runProgressRowWidget = stage_strip
    owner._runStageIconLabel = stage_icon
    owner._runProgressBar = progress
    owner._runStageCountLabel = count_label
    owner._runStageLabel = stage_label
    owner._runStatusHintLabel = hint
    owner._runProgressTimer = timer
    owner._runProgressManifestPath = None
    owner._runProgressStartedAt = None
    owner._runProgressHasData = False
    owner._runProgressCounts = (0, 0)
    set_run_state(owner, "idle")


def set_run_state(owner: Any, state: str) -> None:
    """Update the state chip and the stage indicator strip for ``state``."""

    presentation = run_state_presentation(state)
    owner._runState = state or "idle"
    label = getattr(owner, "_runStateLabel", None)
    if label is not None:
        label.setText(_text(owner, f"run_state_{owner._runState}", presentation.label))
        label.setStyleSheet(
            f"QLabel#runStateLabel {{ background-color: {presentation.color}; color: white; "
            "border-radius: 4px; padding: 3px 8px; font-weight: 600; }"
        )
    progress = getattr(owner, "_runProgressBar", None)
    if progress is None:
        return
    has_data = bool(getattr(owner, "_runProgressHasData", False))
    visible = owner._runState != "idle"
    stage_strip = getattr(owner, "_runProgressRowWidget", None)
    was_visible = bool(getattr(stage_strip, "visible", False)) if stage_strip is not None else False
    progress.visible = visible
    if stage_strip is not None:
        stage_strip.visible = visible
    if presentation.busy:
        if not has_data:
            progress.setRange(0, 0)  # busy indicator until the first manifest checkpoint
        refresh_run_progress_styles(owner)
        if stage_strip is not None and was_visible != visible:
            queue_fit = getattr(owner, "_queueCompactFit", None)
            if callable(queue_fit):
                queue_fit()
        return
    stage = getattr(owner, "_runStageLabel", None)
    if stage is not None:
        stage.visible = False
    if owner._runState == "idle":
        owner._runProgressCounts = (0, 0)
        progress.setRange(0, 1)
        progress.setValue(0)
        progress.setFormat(_text(owner, "no_run_in_progress", "No run in progress"))
    elif not has_data:
        progress.setRange(0, 1)
        progress.setValue(1 if owner._runState in {"completed", "completed_with_blocks", "failed"} else 0)
        progress.setFormat(presentation.label)
    refresh_run_progress_styles(owner)
    if stage_strip is not None and was_visible != visible:
        queue_fit = getattr(owner, "_queueCompactFit", None)
        if callable(queue_fit):
            queue_fit()


def reset_run_progress(owner: Any) -> None:
    owner._runProgressHasData = False
    owner._runProgressCounts = (0, 0)
    progress = getattr(owner, "_runProgressBar", None)
    if progress is not None:
        progress.setRange(0, 1)
        progress.setValue(0)
        progress.setFormat(progress_bar_format(0, 0))
    stage = getattr(owner, "_runStageLabel", None)
    if stage is not None:
        stage.setText("")
        stage.visible = False
    refresh_run_progress_styles(owner)


def start_progress_polling(owner: Any, manifest_path: Optional[Path], *, started_at: Optional[float] = None) -> None:
    """Poll the orchestration checkpoint manifest while the background run is alive."""

    owner._runProgressManifestPath = Path(manifest_path) if manifest_path else None
    owner._runProgressStartedAt = started_at if started_at is not None else time.time()
    owner._tableShowsLiveRun = True
    reset_run_progress(owner)
    timer = getattr(owner, "_runProgressTimer", None)
    if timer is not None and owner._runProgressManifestPath is not None:
        timer.start()


def stop_progress_polling(owner: Any) -> None:
    timer = getattr(owner, "_runProgressTimer", None)
    if timer is not None:
        timer.stop()


def poll_run_progress(owner: Any) -> None:
    path = getattr(owner, "_runProgressManifestPath", None)
    if path is None:
        return
    record = {}
    if (getattr(owner, '_backgroundRunContext', {}) or {}).get('experiment_manifest_path'):
        try:
            record = json.loads(Path(path).read_text(encoding='utf-8'))
        except (OSError, ValueError):
            pass
    if record.get('kind') == 'slicerdsurfer_batch_run' and record.get('experiment'):
        from shared.ui_run_controller import rows_from_batch_manifest
        total, done = len(record.get('subjects', [])), len(record.get('results', []))
        progress = getattr(owner, '_runProgressBar', None)
        if progress is not None and total:
            owner._runProgressHasData = True
            owner._runProgressCounts = (done, total)
            progress.setRange(0, total)
            progress.setValue(done)
            progress.setFormat(progress_bar_format(done, total))
            refresh_run_progress_styles(owner)
        owner._set_run_status_table_rows(rows_from_batch_manifest(Path(path), record)[0])
        return
    snapshot = progress_from_manifest(path, not_before=getattr(owner, "_runProgressStartedAt", None))
    if snapshot is not None:
        apply_progress_snapshot(owner, snapshot)


def apply_progress_snapshot(owner: Any, snapshot) -> None:
    """Show done/total stages, the current stage and finished task states."""

    progress = getattr(owner, "_runProgressBar", None)
    if progress is not None and snapshot.total > 0:
        owner._runProgressHasData = True
        owner._runProgressCounts = (snapshot.done, snapshot.total)
        progress.setRange(0, snapshot.total)
        progress.setValue(snapshot.done)
        progress.setFormat(progress_bar_format(snapshot.done, snapshot.total))
        refresh_run_progress_styles(owner)
    stage = getattr(owner, "_runStageLabel", None)
    if stage is not None:
        if snapshot.current_stage:
            stage.setText(_text(owner, "current_stage", "Current stage") + f": {snapshot.current_stage}")
            stage.visible = True
        else:
            stage.visible = False
    if getattr(owner, "_tableShowsLiveRun", False):
        for name, status in snapshot.finished_tasks:
            prefix = name.split(".", 1)[0]
            owner._update_run_status_table_item((name, prefix), status)


# ----------------------------------------------------------------------- results


def install_results_panel(owner: Any) -> None:
    """Build the run selector, stage table, scene actions and report buttons."""

    if getattr(owner, "_runHistoryComboBox", None) is not None:
        return
    layout = _layout_or_fallback(owner, "resultsLayout")
    if layout is None:
        return
    import qt

    empty_row = qt.QWidget()
    empty_layout = qt.QHBoxLayout(empty_row)
    empty_layout.setContentsMargins(0, 0, 0, 0)
    empty_label = qt.QLabel("No runs yet")
    owner._resultsEmptyLabel = empty_label
    _bind(owner, empty_label, "text", "no_runs_found")
    empty_layout.addWidget(empty_label, 1)
    layout.addWidget(empty_row)

    columns = qt.QSplitter(qt.Qt.Horizontal)
    columns.setChildrenCollapsible(False)
    columns.hide()
    layout.addWidget(columns)
    owner._resultsEmptyRow = empty_row
    owner._resultsColumns = columns
    runs = qt.QWidget()
    outputs = qt.QWidget()
    columns.addWidget(runs)
    columns.addWidget(outputs)
    columns.setStretchFactor(1, 1)
    run_layout = qt.QVBoxLayout(runs)
    layout = qt.QVBoxLayout(outputs)
    run_layout.setContentsMargins(0, 0, 3, 0)
    layout.setContentsMargins(3, 0, 0, 0)

    run_row = qt.QHBoxLayout()
    run_label = qt.QLabel("Run")
    run_label.setObjectName("runHistoryLabel")
    combo = qt.QComboBox(runs)
    combo.setObjectName("runHistoryComboBox")
    combo.setSizeAdjustPolicy(qt.QComboBox.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(14)
    combo.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed)
    combo.hide()
    history = qt.QTreeWidget()
    history.setObjectName("subjectRunHistory")
    history.setHeaderHidden(True)
    history.setMinimumHeight(85)
    history.setMaximumHeight(160)
    history.setFrameShape(qt.QFrame.NoFrame)
    history.setMinimumWidth(0)
    owner._runHistoryTree = history
    add_folder = qt.QPushButton("Add a run folder…")
    add_folder.clicked.connect(owner._on_browse_run_directory_clicked)
    _bind(owner, add_folder, "text", "add_run_folder")

    def select_run(item, _column):
        index = item.data(0, qt.Qt.UserRole)
        if index is not None:
            combo.setCurrentIndex(int(index))

    history.itemClicked.connect(select_run)
    combo.setToolTip(
        "Runs found under the output directory, the subject root, the batch root and folders added with "
        "“Add a run folder…”, newest first. GUI, command-line and batch runs all appear here, "
        "also from earlier Slicer sessions."
    )
    run_row.addWidget(run_label)
    run_layout.addLayout(run_row)
    run_layout.addWidget(history)
    empty_layout.addWidget(add_folder)

    summary = qt.QLabel("Run summary will appear after a plan or run completes.")
    summary.wordWrap = True
    summary.setObjectName("runSummaryLabel")
    summary.hide()
    layout.addWidget(summary)

    details = qt.QLabel("No run selected.")
    details.setObjectName("runDetailsLabel")
    details.wordWrap = True
    try:
        details.setTextInteractionFlags(qt.Qt.TextSelectableByMouse | qt.Qt.TextSelectableByKeyboard)
    except Exception:
        pass
    details.hide()
    run_layout.addWidget(details)
    run_layout.addStretch(1)

    review = qt.QLabel("Recorded QC: consult report\nVisual review: pending")
    review.wordWrap = True
    review.setObjectName("reviewStateLabel")
    layout.addWidget(review)

    files = qt.QTreeWidget()
    files.setObjectName("runOutputFiles")
    files.setHeaderHidden(True)
    files.setColumnCount(2)
    files.setMinimumHeight(85)
    files.setMaximumHeight(190)
    files.setFrameShape(qt.QFrame.NoFrame)
    files.header().setSectionResizeMode(0, qt.QHeaderView.Stretch)
    files.header().setSectionResizeMode(1, qt.QHeaderView.ResizeToContents)
    owner._runOutputFiles = files
    layout.addWidget(files)
    files.itemClicked.connect(lambda item, column: _toggle_output(owner, item) if column == 1 else None)
    files.setContextMenuPolicy(qt.Qt.CustomContextMenu)
    files.customContextMenuRequested.connect(lambda position: _output_menu(owner, position))

    table = qt.QTableWidget()
    table.setObjectName("runStatusTable")
    table.setColumnCount(6)
    table.setHorizontalHeaderLabels(["Target", "State", "Reason", "Manifest", "QC", "Logs"])
    table.setAlternatingRowColors(True)
    table.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
    table.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
    table.setHorizontalScrollMode(qt.QAbstractItemView.ScrollPerPixel)
    # One text line per row. Prose cells elide on the right; the manifest, QC and log paths sit in hidden
    # columns and show in the Target cell's tooltip. The height follows the rows (see fit_table_height).
    table.setWordWrap(False)
    table.setTextElideMode(qt.Qt.ElideRight)
    table.setHorizontalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOff)
    table.verticalHeader().setVisible(False)
    # Some supported PythonQt builds cannot wrap QWidget.fontMetrics().
    # Keep the style's row height as a fallback so panel setup can complete.
    try:
        row_height = table.fontMetrics().height() + 8
    except (TypeError, ValueError, AttributeError):
        row_height = max(24, table.verticalHeader().defaultSectionSize)
    table.verticalHeader().setDefaultSectionSize(row_height)
    for column in (3, 4, 5):
        table.setColumnHidden(column, True)
    table.setToolTip("Per-stage outcome of the selected run. Double-click a row to open its QC file or logs.")
    try:
        header = table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, qt.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, qt.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, qt.QHeaderView.Stretch)
    except Exception:
        pass
    fit_table_height(table)
    table.cellDoubleClicked.connect(owner._on_run_status_table_double_clicked)
    table.setContextMenuPolicy(qt.Qt.CustomContextMenu)

    def result_menu(position):
        index = table.indexAt(position)
        target = table.item(index.row(), 0) if index.isValid() else None
        if target is None or not target.text().startswith("segmentation."):
            return
        caption = ("编辑组织分割副本" if str(getattr(owner, "_onboardingLanguage", "en")).startswith("zh")
                   else "Edit tissue copy in Segment Editor")
        menu = qt.QMenu(table)
        action = menu.addAction(caption)
        action.connect("triggered()", lambda: owner._manage_result_scene("edit_tissue"))
        menu.exec_(table.viewport().mapToGlobal(position))

    table.customContextMenuRequested.connect(result_menu)
    layout.addWidget(table)

    owner._resultSceneButtons = []

    report_row = qt.QHBoxLayout()
    report_button = qt.QPushButton("Open report")
    report_button.enabled = False
    report_button.setToolTip("Open the HTML report of the selected run.")
    report_button.connect("clicked()", owner._on_open_latest_report_clicked)
    output_tree_button = qt.QPushButton("Open output tree")
    output_tree_button.enabled = False
    output_tree_button.setToolTip("Open the annotated output-tree HTML report of the selected run.")
    output_tree_button.connect("clicked()", owner._on_open_output_tree_report_clicked)
    report_row.addWidget(report_button)
    report_row.addWidget(output_tree_button)
    layout.addLayout(report_row)

    combo.currentIndexChanged.connect(lambda _index: owner._on_run_history_selection_changed())

    _bind(owner, run_label, "text", "run_selector_label")

    owner._runStatusPanel = getattr(owner.ui, "resultsCollapsibleButton", None) or table
    owner._runHistoryComboBox = combo
    owner._runHistoryRecords = ()
    owner._runDetailsLabel = details
    owner._runSummaryLabel = summary
    owner._reviewStateLabel = review
    owner._runStatusTable = table
    owner._runReportButton = report_button
    owner._runOutputTreeButton = output_tree_button
    owner._latestReportPath = None
    owner._latestOutputTreePath = None
    owner._latestResultManifestPath = None
    owner._tableShowsLiveRun = False


def _populate_run_tree(owner, records, current_index):
    tree = getattr(owner, "_runHistoryTree", None)
    if tree is None:
        return
    import qt

    columns = owner._resultsColumns
    columns.setVisible(bool(records))
    owner._resultsEmptyLabel.setVisible(not records)
    tree.clear()
    subjects = {}
    for index, record in enumerate(records):
        subject = record.subject_id or _text(owner, "run_selector_label", "Runs")
        if subject not in subjects:
            subjects[subject] = qt.QTreeWidgetItem(tree, [subject])
        group = subjects[subject]
        name = record.run_name or record.manifest_path.parent.name
        item = qt.QTreeWidgetItem(group, [f"{name} · {status_label(record.status)}"])
        item.setData(0, qt.Qt.UserRole, index)
        item.setToolTip(0, run_record_details(record))
        group.setExpanded(True)
        if index == current_index:
            tree.setCurrentItem(item)


def _output_nodes(slicer, manifest_path, path):
    from shared.result_scene import owned_nodes

    return [node for node in owned_nodes(slicer, str(manifest_path.resolve()))
            if node.GetAttribute("SlicerDDSurfer.SourcePath") == str(path)]


def _slice_connections(slicer, node):
    layer = "Label" if node.IsA("vtkMRMLLabelMapVolumeNode") else "Background"
    return [(composite, layer) for composite in slicer.util.getNodesByClass("vtkMRMLSliceCompositeNode")
            if getattr(composite, f"Get{layer}VolumeID")() == node.GetID()]


def _output_visible(slicer, nodes):
    for node in nodes:
        if node.IsA("vtkMRMLVolumeNode"):
            if _slice_connections(slicer, node):
                return True
        elif node.GetDisplayVisibility():
            return True
    return False


def _toggle_output(owner, item):
    import qt
    import slicer

    raw_path = item.data(0, qt.Qt.UserRole)
    manifest_path = getattr(owner, "_latestResultManifestPath", None)
    if not raw_path or manifest_path is None:
        return
    path = Path(raw_path)
    if item.text(1) == "↗":
        owner._open_local_path(path, missing_message="The result file is missing.")
        return
    try:
        from shared.result_scene import source_stamp

        nodes = _output_nodes(slicer, manifest_path, path)
        visible = _output_visible(slicer, nodes)
        if not visible and nodes and path.is_file():
            stamp = source_stamp(path)
            if any(node.GetAttribute("SlicerDDSurfer.SourceStamp") != stamp for node in nodes):
                for node in nodes:
                    slicer.mrmlScene.RemoveNode(node)
                nodes = []
        if not nodes:
            result = owner.logic.loadKeyResultsFromManifest(manifest_path, source_path=path)
            nodes = _output_nodes(slicer, manifest_path, path)
            if not nodes:
                error = next((entry["reason"] for entry in result["skipped"] if entry.get("path") == str(path)),
                             "This output is missing or no longer available in the run.")
                raise RuntimeError(error)
        for node in nodes:
            if node.IsA("vtkMRMLVolumeNode"):
                if visible:
                    for composite, layer in _slice_connections(slicer, node):
                        getattr(composite, f"Set{layer}VolumeID")(None)
                else:
                    layer = "label" if node.IsA("vtkMRMLLabelMapVolumeNode") else "background"
                    slicer.util.setSliceViewerLayers(**{layer: node})
            else:
                node.SetDisplayVisibility(not visible)
        _refresh_output_visibility(owner)
    except Exception as exc:
        slicer.util.errorDisplay(str(exc), parent=owner.parent)


def _output_menu(owner, position):
    import qt
    import slicer

    tree = owner._runOutputFiles
    item = tree.itemAt(position)
    raw_path = item.data(0, qt.Qt.UserRole) if item is not None else None
    manifest_path = owner._latestResultManifestPath
    if not raw_path or manifest_path is None:
        return
    nodes = _output_nodes(slicer, manifest_path, Path(raw_path))
    if not nodes:
        return
    menu = qt.QMenu(tree)
    zh = str(getattr(owner, "_onboardingLanguage", "en")).startswith("zh")
    unload = menu.addAction("从场景卸载" if zh else "Unload from scene")
    if menu.exec_(tree.viewport().mapToGlobal(position)) == unload:
        for node in nodes:
            slicer.mrmlScene.RemoveNode(node)
        _refresh_output_visibility(owner)


def _eye_icon(qt, visible):
    key = bool(visible)
    if key not in _EYE_ICONS:
        color = "#27824d" if visible else "#728195"
        slash = '' if visible else '<path d="M3 3l18 18"/>'
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24">'
               f'<g fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round">'
               '<path d="M2 12s3.6-6 10-6 10 6 10 6-3.6 6-10 6-10-6-10-6z"/>'
               f'<circle cx="12" cy="12" r="2.5"/>{slash}</g></svg>')
        pixmap = qt.QPixmap()
        pixmap.loadFromData(qt.QByteArray(svg.encode("utf-8")))
        _EYE_ICONS[key] = qt.QIcon(pixmap)
    return _EYE_ICONS[key]


_EYE_ICONS = {}


def _refresh_output_visibility(owner):
    import slicer

    tree = getattr(owner, "_runOutputFiles", None)
    manifest_path = getattr(owner, "_latestResultManifestPath", None)
    if tree is None or manifest_path is None:
        return
    import qt
    from shared.result_scene import owned_nodes

    by_path = {}
    for node in owned_nodes(slicer, str(manifest_path.resolve())):
        by_path.setdefault(node.GetAttribute("SlicerDDSurfer.SourcePath"), []).append(node)
    zh = str(getattr(owner, "_onboardingLanguage", "en")).startswith("zh")
    for index in range(tree.topLevelItemCount):
        group = tree.topLevelItem(index)
        for child_index in range(group.childCount()):
            item = group.child(child_index)
            if item.text(1) == "↗":
                continue
            path = str(item.data(0, qt.Qt.UserRole))
            visible = _output_visible(slicer, by_path.get(path, ()))
            item.setIcon(1, _eye_icon(qt, visible))
            item.setText(1, "")
            item.setToolTip(1, ("隐藏" if zh else "Hide") if visible else ("显示" if zh else "Show"))


def _show_output_files(owner, manifest):
    tree = getattr(owner, "_runOutputFiles", None)
    if tree is None:
        return
    import qt
    from shared.result_scene import result_candidates

    tree.clear()
    candidates, _ = result_candidates(manifest, allow_bundle_directory_scan=False)
    visual_paths = {item["path"] for item in candidates
                    if Path(item["path"]).suffix.lower() not in {".csv", ".tsv"}}
    groups = {}
    for candidate in candidates:
        path = candidate["path"]
        groups.setdefault(candidate["product"], {})[path] = path in visual_paths
    for task in manifest.get("tasks") or []:
        if not isinstance(task, dict) or not isinstance(task.get("outputs"), dict):
            continue
        for value in task["outputs"].values():
            if not isinstance(value, str):
                continue
            path = Path(value)
            if path.suffix.lower() == ".json" and path.is_absolute() and path.is_file():
                groups.setdefault(str(task.get("name") or "QC"), {})[str(path.resolve())] = False
    zh = str(getattr(owner, "_onboardingLanguage", "en")).startswith("zh")
    titles = {
        "anatomy": ("Anatomical priors", "解剖先验"),
        "preprocessing": ("Preprocessing", "预处理"),
        "microstructure": ("Microstructure", "微结构"),
        "tractography": ("Tractography", "纤维追踪"),
        "bundles": ("Fibre bundles", "纤维束"),
        "connectome": ("Connectome", "连接组"),
    }
    families = {}
    for product, paths in groups.items():
        family = product.split(".", 1)[0]
        if family in {"surface", "segmentation", "parcellation", "postprocess"}:
            family = "anatomy"
        families.setdefault(family, {}).update(paths)
    for family, paths in families.items():
        caption = titles.get(family, (family, family))[int(zh)]
        group = qt.QTreeWidgetItem(tree, [f"{caption} ({len(paths)})"])
        for path, visual in paths.items():
            item = qt.QTreeWidgetItem(group, [Path(path).name, "○" if visual else "↗"])
            item.setData(0, qt.Qt.UserRole, path)
            item.setToolTip(0, path)
    _refresh_output_visibility(owner)


def load_history_directories() -> tuple[str, ...]:
    try:
        import slicer

        value = slicer.app.userSettings().value(RUN_HISTORY_SETTING, [])
    except Exception:
        return ()
    if isinstance(value, str):
        value = [value] if value else []
    try:
        return tuple(str(item) for item in value if str(item or "").strip())
    except TypeError:
        return ()


def save_history_directory(directory: str) -> tuple[str, ...]:
    updated = remember_directory(load_history_directories(), directory)
    try:
        import slicer

        slicer.app.userSettings().setValue(RUN_HISTORY_SETTING, list(updated))
    except Exception:
        pass
    return updated


def _node_text(owner: Any, attribute: str) -> str:
    node = getattr(owner, "_parameterNode", None)
    return str(getattr(node, attribute, "") or "").strip() if node else ""


def history_roots(owner: Any):
    return candidate_run_roots(
        subject_root_text=_node_text(owner, "inputDirectory"),
        output_root_text=_node_text(owner, "outputDirectory"),
        batch_root_text=_node_text(owner, "batchRoot"),
        extra_dirs=load_history_directories(),
    )


def refresh_run_history(owner: Any, select_path: Optional[Path] = None) -> None:
    """Rescan disk for runs and repopulate the selector, keeping the selection."""

    combo = getattr(owner, "_runHistoryComboBox", None)
    if combo is None:
        return
    previous = select_path or selected_manifest_path(owner)
    discovery = discover_runs(history_roots(owner))
    owner._runHistoryRecords = discovery.records
    combo.blockSignals(True)
    try:
        combo.clear()
        for record in discovery.records:
            combo.addItem(run_record_label(record), record.manifest_path.as_posix())
            combo.setItemData(combo.count - 1, run_record_details(record), 3)  # Qt::ToolTipRole
        if not discovery.records:
            combo.addItem(_text(owner, "no_runs_found", "No runs found"), "")
        # The last item adds a folder; it carries no manifest, so a selection never reads it as a run.
        try:
            combo.insertSeparator(combo.count)
        except Exception:
            pass
        combo.addItem(_text(owner, "add_run_folder", ADD_RUN_FOLDER_TEXT), "")
        combo.setItemData(combo.count - 1, ADD_RUN_FOLDER_TOOLTIP, 3)
        combo.enabled = True
        target_index = 0
        if previous is not None:
            wanted = Path(previous).as_posix()
            for index, record in enumerate(discovery.records):
                if record.manifest_path.as_posix() == wanted:
                    target_index = index
                    break
        combo.setCurrentIndex(target_index)
        owner._runHistoryPreviousIndex = target_index
    finally:
        combo.blockSignals(False)
    _populate_run_tree(owner, discovery.records, target_index)
    notes = []
    if discovery.skipped:
        notes.append(f"{len(discovery.skipped)} unreadable manifest(s) skipped; first: {discovery.skipped[0][1]}")
    if discovery.truncated:
        notes.append("Search stopped early in a very large folder; choose “Add a run folder…” and pick the run folder itself.")
    owner._runHistoryNotes = tuple(notes)
    on_run_selection_changed(owner, from_refresh=True)


def selected_manifest_path(owner: Any) -> Optional[Path]:
    combo = getattr(owner, "_runHistoryComboBox", None)
    if combo is None or combo.count == 0:
        return None
    if is_add_run_folder_item(combo):
        return None
    data = combo.itemData(combo.currentIndex)
    return Path(str(data)) if data else None


def is_add_run_folder_item(combo: Any) -> bool:
    """True when the combo shows its last item, "Add a run folder…", which is an action and not a run."""

    return combo.count > 0 and combo.currentIndex == combo.count - 1


def on_run_combo_changed(owner: Any) -> None:
    """Combo slot: show the chosen run, or open the folder dialog for the "Add a run folder…" item."""

    combo = getattr(owner, "_runHistoryComboBox", None)
    if combo is not None and is_add_run_folder_item(combo):
        # Put the previous run back first, so a cancelled dialog leaves the selection unchanged.
        # A folder that adds runs re-selects through refresh_run_history.
        previous = getattr(owner, "_runHistoryPreviousIndex", 0)
        combo.blockSignals(True)
        try:
            combo.setCurrentIndex(previous if 0 <= previous < combo.count - 1 else 0)
        finally:
            combo.blockSignals(False)
        owner._on_browse_run_directory_clicked()
        return
    if combo is not None:
        owner._runHistoryPreviousIndex = combo.currentIndex
    on_run_selection_changed(owner)


def _set_scene_actions(owner: Any, *, has_subject_run: bool, is_batch: bool, error: str = "") -> None:
    state = result_actions_state(has_subject_run=has_subject_run, is_batch=is_batch, error=error)
    # An enabled button explains its own action; a disabled one gives the reason for all four.
    for button, (_label, mode) in zip(getattr(owner, "_resultSceneButtons", []), SCENE_ACTIONS):
        button.enabled = state.enabled
        button.setToolTip(SCENE_ACTION_TOOLTIPS[mode] if state.enabled else state.tooltip)


def on_run_selection_changed(owner: Any, *, from_refresh: bool = False) -> None:
    """Show the selected run's details, stage table, reports and scene actions."""

    details = getattr(owner, "_runDetailsLabel", None)
    path = selected_manifest_path(owner)
    notes = "\n".join(getattr(owner, "_runHistoryNotes", ()) or ())
    _show_output_files(owner, {})
    if path is None:
        owner._latestResultManifestPath = None
        _set_scene_actions(owner, has_subject_run=False, is_batch=False)
        if details is not None:
            details.setText(_text(owner, "no_run_selected", "No run selected.") + (f"\n{notes}" if notes else ""))
            details.setToolTip("")
        return
    # A programmatic refresh keeps the live run's table (running or just cancelled);
    # picking a run in the list always shows that run.
    live = from_refresh and getattr(owner, "_tableShowsLiveRun", False)
    try:
        record = parse_run_manifest(path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (ManifestError, OSError, ValueError) as exc:
        owner._latestResultManifestPath = None
        _set_scene_actions(owner, has_subject_run=False, is_batch=False, error=str(exc))
        if details is not None:
            details.setText(f"Cannot use this run: {exc}")
            details.setToolTip("")
        return
    if details is not None:
        details.setText(run_record_brief(record) + (f"\n{notes}" if notes else ""))
        details.setToolTip(run_record_details(record))
    if live:
        # Keep showing the live run's stage table while it executes.
        return
    owner._tableShowsLiveRun = False
    show_record(owner, record, manifest)


def show_record(owner: Any, record: RunRecord, manifest: dict[str, Any]) -> None:
    from shared.result_scene import recorded_qc_summary, review_state_text
    from shared.ui_run_controller import rows_from_batch_manifest, rows_from_single_subject_manifest

    review = getattr(owner, "_reviewStateLabel", None)
    if record.is_batch:
        rows, counts = rows_from_batch_manifest(record.manifest_path, manifest)
        owner._set_run_status_table_rows(rows)
        owner._latestResultManifestPath = None
        _set_scene_actions(owner, has_subject_run=False, is_batch=True)
        owner._set_latest_report_path(None)
        owner._set_latest_output_tree_path(None)
        owner._set_run_summary_message(
            f"{status_label(record.status).capitalize()} · {counts['total']} {'experiment jobs' if manifest.get('experiment') else 'subjects'}\n"
            f"{counts['completed']} completed · {counts['failed']} failed · {counts['skipped']} skipped",
            "failed" if counts["failed"] else "completed",
        )
        if review is not None:
            review.setText("Batch run: select one subject's run to review QC and images.")
        return
    rows, counts = rows_from_single_subject_manifest(manifest)
    owner._set_run_status_table_rows(rows)
    owner._latestResultManifestPath = record.manifest_path
    _show_output_files(owner, manifest)
    _set_scene_actions(owner, has_subject_run=True, is_batch=False)
    report_paths = manifest.get("report_paths") or {}
    tree_paths = manifest.get("output_tree_report_paths") or {}
    owner._set_latest_report_path(
        str(report_paths.get("report_html_latest") or report_paths.get("report_html") or "")
        if isinstance(report_paths, dict) else None
    )
    owner._set_latest_output_tree_path(
        str(tree_paths.get("report_html_latest") or tree_paths.get("report_html") or "")
        if isinstance(tree_paths, dict) else None
    )
    state = run_state_from_manifest_status(record.status)
    fallback = (
        f"{status_label(record.status).capitalize()}\n"
        f"{len(rows)} tasks · {counts['failed']} failed · {counts['blocked']} blocked"
    )
    message, summary_state = owner._run_summary_from_manifest(manifest)
    owner._set_run_summary_message(message or fallback, summary_state if message else state)
    if review is not None:
        qc = recorded_qc_summary(manifest)
        issues = counts["failed"] or counts["blocked"]
        review.setText(review_state_text(qc, issues_found=bool(issues)))


def validated_selected_manifest(owner: Any) -> tuple[Optional[Path], str]:
    """Re-read and re-validate the selected manifest right before loading."""

    path = getattr(owner, "_latestResultManifestPath", None) or selected_manifest_path(owner)
    if path is None:
        return None, "No run is selected. Choose one in the Run list or choose “Add a run folder…” there."
    try:
        record = parse_run_manifest(Path(path))
    except ManifestError as exc:
        return None, str(exc)
    if record.is_batch:
        return None, "This is a batch run. Choose one subject's run from the Run list to load its images."
    return record.manifest_path, ""


def browse_run_directory(owner: Any) -> None:
    """Ask for any run/output folder, validate what it contains and select it."""

    import qt
    import slicer

    start = _node_text(owner, "outputDirectory") or _node_text(owner, "inputDirectory")
    if not start:
        remembered = load_history_directories()
        start = remembered[0] if remembered else str(Path.home())
    directory = qt.QFileDialog.getExistingDirectory(
        owner.parent, _text(owner, "browse_runs_title", "Choose a run or output folder"), start
    )
    if not directory:
        return
    discovery = discover_runs([directory])
    if not discovery.records:
        from shared.ui_error_dialogs import show_actionable_error_dialog

        why = (
            discovery.skipped[0][1]
            if discovery.skipped
            else "No run manifest was found within three folder levels of the selected folder."
        )
        show_actionable_error_dialog(
            parent=owner.parent,
            title="No SlicerDSpace run found",
            what=f"{directory} does not contain a loadable SlicerDSpace run.",
            why=why,
            next_step=(
                "Choose the subject's output folder (it contains 00_run_metadata/<run>_orchestration_manifest.json), "
                "the 00_run_metadata folder itself, or a batch output folder with 00_batch_metadata/batch_*.json."
            ),
            auto_fix="Not available; the folder is only read, never modified.",
            details="\n".join(f"{path}: {reason}" for path, reason in discovery.skipped) or directory,
        )
        return
    save_history_directory(directory)
    refresh_run_history(owner, select_path=discovery.records[0].manifest_path)
    slicer.util.showStatusMessage(
        f"Found {len(discovery.records)} run(s) under {directory}; selected the newest.", 4000
    )
