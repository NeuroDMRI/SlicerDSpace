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


def install_run_progress_panel(owner: Any) -> None:
    """Add run state, per-stage progress bar and stage/message lines to 4. Run."""

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
    progress = qt.QProgressBar()
    progress.setObjectName("runProgressBar")
    progress.setRange(0, 1)
    progress.setValue(0)
    progress.setTextVisible(True)
    progress.setFormat("No run in progress")
    progress.setToolTip("Finished stages out of the stages this run plans to execute (from the run manifest).")
    progress.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed)
    progress.setMinimumWidth(60)  # its text would otherwise set a wide minimum
    state_row.addWidget(progress, 1)
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
    owner._runProgressBar = progress
    owner._runStageLabel = stage_label
    owner._runStatusHintLabel = hint
    owner._runProgressTimer = timer
    owner._runProgressManifestPath = None
    owner._runProgressStartedAt = None
    owner._runProgressHasData = False
    set_run_state(owner, "idle")


def set_run_state(owner: Any, state: str) -> None:
    """Update the state chip and the progress bar mode for ``state``."""

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
    progress.visible = owner._runState != "idle"
    if presentation.busy:
        if not has_data:
            progress.setRange(0, 0)  # busy indicator until the first manifest checkpoint
        return
    stage = getattr(owner, "_runStageLabel", None)
    if stage is not None:
        stage.visible = False
    if owner._runState == "idle":
        progress.setRange(0, 1)
        progress.setValue(0)
        progress.setFormat(_text(owner, "no_run_in_progress", "No run in progress"))
    elif not has_data:
        progress.setRange(0, 1)
        progress.setValue(1 if owner._runState in {"completed", "completed_with_blocks", "failed"} else 0)
        progress.setFormat(presentation.label)


def reset_run_progress(owner: Any) -> None:
    owner._runProgressHasData = False
    progress = getattr(owner, "_runProgressBar", None)
    if progress is not None:
        progress.setRange(0, 1)
        progress.setValue(0)
        progress.setFormat(progress_bar_format(0, 0))
    stage = getattr(owner, "_runStageLabel", None)
    if stage is not None:
        stage.setText("")
        stage.visible = False


def start_progress_polling(owner: Any, manifest_path: Optional[Path], *, started_at: Optional[float] = None) -> None:
    """Poll the orchestration checkpoint manifest while the background run is alive."""

    owner._runProgressManifestPath = Path(manifest_path) if manifest_path else None
    owner._runProgressStartedAt = started_at if started_at is not None else time.time()
    owner._tableShowsLiveRun = True
    owner._autoResultManifestSignatures = {}
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
            progress.setRange(0, total)
            progress.setValue(done)
            progress.setFormat(progress_bar_format(done, total))
        owner._set_run_status_table_rows(rows_from_batch_manifest(Path(path), record)[0])
        return
    snapshot = progress_from_manifest(path, not_before=getattr(owner, "_runProgressStartedAt", None))
    if snapshot is not None:
        apply_progress_snapshot(owner, snapshot)
        loader = getattr(owner, "_auto_load_key_results", None)
        if callable(loader):
            loader(Path(path))


def apply_progress_snapshot(owner: Any, snapshot) -> None:
    """Show done/total stages, the current stage and finished task states."""

    progress = getattr(owner, "_runProgressBar", None)
    if progress is not None and snapshot.total > 0:
        owner._runProgressHasData = True
        progress.setRange(0, snapshot.total)
        progress.setValue(snapshot.done)
        progress.setFormat(progress_bar_format(snapshot.done, snapshot.total))
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

    run_row = qt.QHBoxLayout()
    run_label = qt.QLabel("Run")
    run_label.setObjectName("runHistoryLabel")
    combo = qt.QComboBox()
    combo.setObjectName("runHistoryComboBox")
    combo.setSizeAdjustPolicy(qt.QComboBox.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(14)
    combo.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed)
    combo.setToolTip(
        "Runs found under the output directory, the subject root, the batch root and folders added with "
        "“Add a run folder…”, newest first. GUI, command-line and batch runs all appear here, "
        "also from earlier Slicer sessions. The list is rescanned whenever this section is opened."
    )
    run_label.setBuddy(combo)
    run_row.addWidget(run_label)
    run_row.addWidget(combo, 1)
    layout.addLayout(run_row)

    summary = qt.QLabel("Run summary will appear after a plan or run completes.")
    summary.wordWrap = True
    summary.setObjectName("runSummaryLabel")
    layout.addWidget(summary)

    details = qt.QLabel("No run selected.")
    details.setObjectName("runDetailsLabel")
    details.wordWrap = True
    try:
        details.setTextInteractionFlags(qt.Qt.TextSelectableByMouse | qt.Qt.TextSelectableByKeyboard)
    except Exception:
        pass
    layout.addWidget(details)

    review = qt.QLabel("Recorded QC: consult report\nVisual review: pending")
    review.wordWrap = True
    review.setObjectName("reviewStateLabel")
    layout.addWidget(review)

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

    scene_row = qt.QHBoxLayout()
    owner._resultSceneButtons = []
    for label, mode in SCENE_ACTIONS:
        button = qt.QPushButton(label)
        button.enabled = False
        button.setToolTip(SCENE_ACTION_TOOLTIPS[mode])
        button.connect("clicked()", lambda mode=mode: owner._manage_result_scene(mode))
        scene_row.addWidget(button)
        owner._resultSceneButtons.append(button)
    layout.addLayout(scene_row)

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
    section = getattr(owner.ui, "resultsCollapsibleButton", None)
    if section is not None:
        section.connect("contentsCollapsed(bool)", lambda collapsed: on_results_section_toggled(owner, collapsed))

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


def on_results_section_toggled(owner: Any, collapsed: bool) -> None:
    """Rescan for runs each time the Results and QC section is opened, keeping the selection."""

    if not collapsed:
        owner._refresh_run_history()


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
