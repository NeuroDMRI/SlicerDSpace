"""Qt glue for the "4. Run" progress area and the "5. Results" run selector.

The pure logic (manifest discovery/validation, progress maths, button rules)
lives in ``shared/run_history.py`` and ``shared/ui_run_state.py``; this module
only builds widgets and moves data between them and the owner widget.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
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
    format_elapsed_mmss,
    progress_bar_format,
    result_actions_state,
    run_state_from_manifest_status,
    run_state_presentation,
)


RUN_HISTORY_SETTING = "SlicerDSpace/UserSettings/runHistoryDirectories"
PROGRESS_POLL_INTERVAL_MS = 2000
ADD_RUN_FOLDER_TEXT = "Add a run folder…"
ADD_RUN_FOLDER_TOOLTIP = (
    "Add any run or output folder (GUI, command-line or batch). The folder is searched three levels deep "
    "for 00_run_metadata/<run>_orchestration_manifest.json and remembered for later sessions."
)
SCENE_ACTIONS = (("Load", "reuse"), ("Replace", "replace"), ("Append", "append"), ("Clear", "clear"))
RESULTS_TREE_MAX_HEIGHT = 280
RESULTS_TREE_NO_LIMIT = 16777215  # QWIDGETSIZE_MAX: the tree may grow with the Results window
RESULTS_WINDOW_SIZE = (900, 640)
RESULTS_HEADER_BUTTON_SIZE = (28, 22)  # the module's "..." buttons
RESULTS_WINDOW_NAME = "resultsWindow"
MODULE_ROOT_RULE = "qMRMLWidget#DDSurfer {"  # the root rule of ui_styles.ui_stylesheet, in both themes
RESULT_FAMILY_ORDER = ("preprocessing", "anatomy", "microstructure", "tractography", "bundles", "connectome", "qc")
# Item data roles of the Results tree, as offsets from Qt.UserRole.
ROLE_PATH, ROLE_KIND, ROLE_MANIFEST, ROLE_RUN, ROLE_KEY = 0, 1, 2, 3, 4


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

_SPINNER_FRAME_COUNT = 12
_RUN_ANIMATION_INTERVAL_MS = 150
_STAGE_ICON_SVG_CACHE: dict[tuple[str, str, int, bool], bytes] = {}
_STAGE_ICON_PIXMAP_CACHE: dict[tuple[str, int, bool], Any] = {}


def _set_widget_visible(widget: Any, visible: bool) -> None:
    if widget is None:
        return
    set_visible = getattr(widget, "setVisible", None)
    if callable(set_visible):
        set_visible(bool(visible))
        if not isinstance(getattr(widget, "visible", None), bool):
            try:
                widget.visible = bool(visible)
            except Exception:
                pass
    else:
        try:
            widget.visible = bool(visible)
        except Exception:
            pass


def _state_label_text(owner: Any, state: str, fallback: str) -> str:
    zh = str(getattr(owner, "_onboardingLanguage", "en") or "en").lower().startswith("zh")
    zh_fallbacks = {
        "idle": "空闲",
        "importing": "正在导入影像",
        "preflight": "正在检查环境",
        "setup": "正在准备环境",
        "running": "运行中",
        "completed": "已完成",
        "completed_with_blocks": "部分受阻",
        "cancelled": "已取消",
        "failed": "运行失败",
    }
    default = zh_fallbacks.get(state, fallback) if zh else fallback
    return _text(owner, f"run_state_{state}", default)


def _elapsed_text(owner: Any) -> str:
    started_at = getattr(owner, "_runBusyStartedAt", None)
    if started_at is None:
        started_at = getattr(owner, "_runProgressStartedAt", None)
    if started_at is None:
        return ""
    return format_elapsed_mmss(max(0.0, time.time() - float(started_at)))


def _stage_count_text(owner: Any, done: int, total: int, *, state: str, has_data: bool) -> str:
    lang = str(getattr(owner, "_onboardingLanguage", "en") or "en").lower()
    is_zh = lang.startswith("zh")
    presentation = run_state_presentation(state)
    if has_data and total > 0 and (done > 0 or not presentation.busy):
        if presentation.busy and done > 0:
            base = f"第 {done} / {total} 阶段" if is_zh else f"Stage {done} / {total}"
        elif is_zh:
            base = f"{done} / {total} 阶段"
        else:
            base = progress_bar_format(done, total)
    elif state == "idle":
        return ""
    elif state == "importing":
        base = "导入影像中…" if is_zh else "Importing…"
    elif state == "preflight":
        base = "环境检查中…" if is_zh else "Checking…"
    elif state == "setup":
        base = "配置环境中…" if is_zh else "Setting up…"
    elif state == "running":
        base = "准备运行…" if is_zh else "Preparing…"
    else:
        fallback_zh = {
            "completed": "已完成",
            "completed_with_blocks": "部分受阻",
            "cancelled": "已取消",
            "failed": "运行出错",
        }
        base = fallback_zh[state] if (is_zh and state in fallback_zh) else presentation.label

    if presentation.busy and base:
        elapsed = _elapsed_text(owner)
        if elapsed:
            return f"{base} · {elapsed}"
    return base


def _stage_icon_svg(glyph: str, accent: str, *, dark: bool, angle: int = 0) -> bytes:
    rot = int(angle) % 360 if glyph == "run" else 0
    cache_key = (glyph, accent, rot, bool(dark))
    cached = _STAGE_ICON_SVG_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if glyph == "run":
        track_op = "0.26" if dark else "0.20"
        body = (
            f'<circle cx="8" cy="8" r="5.5" fill="none" stroke="{accent}" '
            f'stroke-opacity="{track_op}" stroke-width="2.0"/>'
            f'<path d="M8 2.5 A5.5 5.5 0 0 1 13.5 8" fill="none" stroke="{accent}" '
            f'stroke-width="2.1" stroke-linecap="round" transform="rotate({rot} 8 8)"/>'
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
    ).encode("utf-8")
    _STAGE_ICON_SVG_CACHE[cache_key] = svg
    return svg


def _spinner_svg_frames(accent: str, *, dark: bool) -> tuple[bytes, ...]:
    return tuple(
        _stage_icon_svg("run", accent, dark=dark, angle=frame * 30)
        for frame in range(_SPINNER_FRAME_COUNT)
    )


def _stage_icon_pixmap(state: str, *, dark: bool, frame: int = 0) -> Any:
    from shared.ui_styles import run_stage_indicator_palette

    pal = run_stage_indicator_palette(state, dark=dark)
    glyph = pal["glyph"]
    angle = (int(frame) % _SPINNER_FRAME_COUNT) * 30 if glyph == "run" else 0
    cache_key = (f"{glyph}:{pal['icon_bg']}", angle, bool(dark))
    cached = _STAGE_ICON_PIXMAP_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        import qt

        if not hasattr(qt, "QPixmap") or not hasattr(qt, "QByteArray"):
            return None
        artwork = _stage_icon_svg(glyph, pal["icon_bg"], dark=dark, angle=angle)
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


def _resolve_dark_mode(owner: Any, dark: Optional[bool] = None) -> bool:
    if dark is not None:
        return bool(dark)
    from shared.ui_styles import is_dark_mode

    cached = getattr(owner, "_lastAppliedDarkMode", None)
    if cached is not None:
        return bool(cached)
    return bool(is_dark_mode(getattr(owner, "_uiWidget", None)))


def _update_stage_icon_and_count(owner: Any, *, state: str, dark: bool) -> None:
    count_label = getattr(owner, "_runStageCountLabel", None)
    if count_label is not None:
        done, total = getattr(owner, "_runProgressCounts", (0, 0))
        has_data = bool(getattr(owner, "_runProgressHasData", False))
        count_label.setText(_stage_count_text(owner, done, total, state=state, has_data=has_data))
    icon_label = getattr(owner, "_runStageIconLabel", None)
    if icon_label is not None:
        frame = int(getattr(owner, "_runSpinnerFrame", 0) or 0)
        pix = _stage_icon_pixmap(state, dark=dark, frame=frame)
        if pix is not None and hasattr(icon_label, "setPixmap"):
            icon_label.setPixmap(pix)


def _on_run_animation_tick(owner: Any) -> None:
    state = str(getattr(owner, "_runState", "idle") or "idle")
    if not run_state_presentation(state).busy:
        stop_run_animation(owner)
        return
    owner._runSpinnerFrame = (int(getattr(owner, "_runSpinnerFrame", 0) or 0) + 1) % _SPINNER_FRAME_COUNT
    dark = _resolve_dark_mode(owner)
    _update_stage_icon_and_count(owner, state=state, dark=dark)


def stop_run_animation(owner: Any) -> None:
    timer = getattr(owner, "_runAnimationTimer", None)
    if timer is not None:
        stop_fn = getattr(timer, "stop", None)
        if callable(stop_fn):
            stop_fn()


def _start_run_animation(owner: Any) -> None:
    timer = getattr(owner, "_runAnimationTimer", None)
    if timer is None:
        return
    is_active = getattr(timer, "isActive", None)
    if callable(is_active):
        try:
            if is_active():
                return
        except Exception:
            pass
    start_fn = getattr(timer, "start", None)
    if callable(start_fn):
        start_fn()


def set_live_stage_text(owner: Any, stage_label: str, detail: str = "") -> None:
    """Update ``_runStageLabel`` immediately during busy phases before/between manifest polls."""

    primary = (stage_label or "").strip()
    extra = (detail or "").strip()
    if primary and extra and extra != primary:
        text = f"{primary} — {extra}"
    else:
        text = primary or extra
    owner._runLiveStageText = text
    owner._runHasLiveStageDetail = bool(extra)
    stage = getattr(owner, "_runStageLabel", None)
    if stage is None:
        return
    if text:
        stage.setText(text)
        _set_widget_visible(stage, True)
    else:
        stage.setText("")
        _set_widget_visible(stage, False)


def refresh_run_progress_styles(owner: Any, *, dark: Optional[bool] = None) -> None:
    """Apply Light/Dark semantic styling and High-DPI vector icon to the stage indicator strip."""

    progress = getattr(owner, "_runProgressBar", None)
    if progress is None:
        return
    from shared.ui_styles import (
        run_progress_bar_style,
        run_stage_count_label_style,
        set_stylesheet_if_changed,
    )

    root = getattr(owner, "_uiWidget", None)
    resolved_dark = _resolve_dark_mode(owner, dark)
    state = str(getattr(owner, "_runState", "idle") or "idle")
    set_stylesheet_if_changed(progress, run_progress_bar_style(state, dark=resolved_dark, widget=root))
    count_label = getattr(owner, "_runStageCountLabel", None)
    if count_label is not None:
        set_stylesheet_if_changed(
            count_label,
            run_stage_count_label_style(state, dark=resolved_dark, widget=root),
        )
    _update_stage_icon_and_count(owner, state=state, dark=resolved_dark)


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
    state_label.setToolTip("Run state: Idle, Importing inputs, Checking environment, Running, Completed, Completed with errors or Cancelled.")
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

    status_frame = getattr(getattr(owner, "ui", None), "statusFrame", None)
    if (
        status_frame is not None
        and hasattr(layout, "indexOf")
        and hasattr(layout, "removeWidget")
        and layout.indexOf(status_frame) >= 0
    ):
        layout.removeWidget(status_frame)
        layout.addWidget(status_frame)

    timer = None
    anim_timer = None
    if hasattr(qt, "QTimer"):
        timer = qt.QTimer()
        timer.setInterval(PROGRESS_POLL_INTERVAL_MS)
        timer.connect("timeout()", owner._poll_run_progress)

        anim_timer = qt.QTimer()
        anim_timer.setInterval(_RUN_ANIMATION_INTERVAL_MS)
        if hasattr(anim_timer, "timeout") and hasattr(anim_timer.timeout, "connect"):
            anim_timer.timeout.connect(lambda: _on_run_animation_tick(owner))
        else:
            anim_timer.connect("timeout()", lambda: _on_run_animation_tick(owner))

    owner._runStateLabel = state_label
    owner._runProgressRowWidget = stage_strip
    owner._runStageIconLabel = stage_icon
    owner._runProgressBar = progress
    owner._runStageCountLabel = count_label
    owner._runStageLabel = stage_label
    owner._runStatusHintLabel = hint
    owner._runProgressTimer = timer
    owner._runAnimationTimer = anim_timer
    owner._runSpinnerFrame = 0
    owner._runBusyStartedAt = None
    owner._runProgressManifestPath = None
    owner._runProgressStartedAt = None
    owner._runProgressHasData = False
    owner._runProgressCounts = (0, 0)
    owner._runLiveStageText = ""
    owner._runHasLiveStageDetail = False
    owner._lastSnapshotStage = None
    set_run_state(owner, "idle")


def set_run_state(owner: Any, state: str) -> None:
    """Update the state chip and the stage indicator strip for ``state``."""

    normalized_state = state or "idle"
    prev_state = str(getattr(owner, "_runState", "idle") or "idle")
    prev_busy = run_state_presentation(prev_state).busy
    presentation = run_state_presentation(normalized_state)
    owner._runState = normalized_state
    if (prev_state == "running") != (normalized_state == "running"):
        _on_running_changed(owner, normalized_state == "running")

    label = getattr(owner, "_runStateLabel", None)
    if label is not None:
        label.setText(_state_label_text(owner, normalized_state, presentation.label))
        label.setStyleSheet(
            f"QLabel#runStateLabel {{ background-color: {presentation.color}; color: white; "
            "border-radius: 4px; padding: 3px 8px; font-weight: 600; }"
        )
        _set_widget_visible(label, normalized_state != "idle")

    status_frame = getattr(getattr(owner, "ui", None), "statusFrame", None)
    if status_frame is not None:
        _set_widget_visible(status_frame, not presentation.busy)

    if presentation.busy:
        if prev_state != normalized_state or getattr(owner, "_runBusyStartedAt", None) is None:
            owner._runBusyStartedAt = time.time()
            owner._runSpinnerFrame = 0
        if (not prev_busy) or (
            prev_state != normalized_state and normalized_state in {"importing", "preflight", "setup"}
        ):
            owner._runProgressHasData = False
            owner._runProgressCounts = (0, 0)
            owner._runHasLiveStageDetail = False
            owner._lastSnapshotStage = None

    progress = getattr(owner, "_runProgressBar", None)
    if progress is None:
        if presentation.busy:
            _start_run_animation(owner)
        else:
            stop_run_animation(owner)
            owner._runBusyStartedAt = None
        return

    done, total = getattr(owner, "_runProgressCounts", (0, 0))
    has_data = bool(getattr(owner, "_runProgressHasData", False))
    visible = normalized_state != "idle"
    stage_strip = getattr(owner, "_runProgressRowWidget", None)
    was_visible = bool(getattr(stage_strip, "visible", False)) if stage_strip is not None else False
    _set_widget_visible(progress, visible)
    _set_widget_visible(stage_strip, visible)

    if presentation.busy:
        if not has_data or total <= 0 or done <= 0:
            progress.setRange(0, 0)  # keep smooth indeterminate pulse until first stage finishes
        else:
            progress.setRange(0, total)
            progress.setValue(done)
        refresh_run_progress_styles(owner)
        _start_run_animation(owner)
        if stage_strip is not None and was_visible != visible:
            queue_fit = getattr(owner, "_queueCompactFit", None)
            if callable(queue_fit):
                queue_fit()
        return

    stop_run_animation(owner)
    owner._runBusyStartedAt = None
    owner._runHasLiveStageDetail = False
    owner._lastSnapshotStage = None
    owner._runLiveStageText = ""
    stage = getattr(owner, "_runStageLabel", None)
    if stage is not None:
        if normalized_state == "idle":
            stage.setText("")
        _set_widget_visible(stage, False)

    if normalized_state == "idle":
        owner._runProgressHasData = False
        owner._runProgressCounts = (0, 0)
        progress.setRange(0, 1)
        progress.setValue(0)
        progress.setFormat(_text(owner, "no_run_in_progress", "No run in progress"))
    elif not has_data or total <= 0 or done <= 0:
        progress.setRange(0, 1)
        progress.setValue(1 if normalized_state in {"completed", "completed_with_blocks", "failed"} else 0)
        progress.setFormat(presentation.label)
    else:
        progress.setRange(0, total)
        progress.setValue(done)

    refresh_run_progress_styles(owner)
    if stage_strip is not None and was_visible != visible:
        queue_fit = getattr(owner, "_queueCompactFit", None)
        if callable(queue_fit):
            queue_fit()


def reset_run_progress(owner: Any) -> None:
    owner._runProgressHasData = False
    owner._runProgressCounts = (0, 0)
    owner._runLiveStageText = ""
    owner._runHasLiveStageDetail = False
    owner._lastSnapshotStage = None
    state = str(getattr(owner, "_runState", "idle") or "idle")
    busy = run_state_presentation(state).busy
    progress = getattr(owner, "_runProgressBar", None)
    if progress is not None:
        if busy:
            progress.setRange(0, 0)
        else:
            progress.setRange(0, 1)
            progress.setValue(0)
        progress.setFormat(progress_bar_format(0, 0))
    stage = getattr(owner, "_runStageLabel", None)
    if stage is not None:
        stage.setText("")
        _set_widget_visible(stage, False)
    refresh_run_progress_styles(owner)


def start_progress_polling(owner: Any, manifest_path: Optional[Path], *, started_at: Optional[float] = None,
                           batch_progress: bool = False) -> None:
    """Poll the orchestration checkpoint manifest while the background run is alive.

    ``batch_progress`` (H3): ``manifest_path`` is the progress file that ``scripts/run_batch_pipeline.py``
    writes after each subject, not an orchestration manifest.
    """

    owner._runProgressManifestPath = Path(manifest_path) if manifest_path else None
    owner._runProgressIsBatch = bool(batch_progress)
    resolved_started_at = started_at if started_at is not None else time.time()
    owner._runProgressStartedAt = resolved_started_at
    owner._runBusyStartedAt = resolved_started_at
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
    if getattr(owner, "_runProgressIsBatch", False):
        apply_batch_progress(owner, Path(path))
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
        state = str(getattr(owner, "_runState", "idle") or "idle")
        busy = run_state_presentation(state).busy
        if progress is not None and total:
            owner._runProgressHasData = True
            owner._runProgressCounts = (done, total)
            if busy and done <= 0:
                progress.setRange(0, 0)
            else:
                progress.setRange(0, total)
                progress.setValue(done)
            progress.setFormat(progress_bar_format(done, total))
            refresh_run_progress_styles(owner)
        owner._set_run_status_table_rows(rows_from_batch_manifest(Path(path), record)[0])
        return
    prev_done = (getattr(owner, "_runProgressCounts", None) or (0, 0))[0]
    snapshot = progress_from_manifest(path, not_before=getattr(owner, "_runProgressStartedAt", None))
    if snapshot is not None:
        apply_progress_snapshot(owner, snapshot)
        if snapshot.done > prev_done and Path(path).is_file() and hasattr(owner, "_refresh_run_history"):
            owner._refresh_run_history(select_path=Path(path))


def apply_batch_progress(owner: Any, path: Path) -> None:
    """Show the done/total subjects and one status row per subject from a batch progress file (H3)."""

    from shared.ui_run_controller import read_batch_progress, rows_from_batch_progress

    record = read_batch_progress(path)
    if record is None:
        return
    rows, done, total = rows_from_batch_progress(record)
    progress = getattr(owner, "_runProgressBar", None)
    if progress is not None and total > 0:
        busy = run_state_presentation(str(getattr(owner, "_runState", "idle") or "idle")).busy
        owner._runProgressHasData = True
        owner._runProgressCounts = (done, total)
        if busy and done <= 0:
            progress.setRange(0, 0)
        else:
            progress.setRange(0, total)
            progress.setValue(done)
        progress.setFormat(progress_bar_format(done, total))
        refresh_run_progress_styles(owner)
    owner._set_run_status_table_rows(rows)


def apply_progress_snapshot(owner: Any, snapshot) -> None:
    """Show done/total stages, the current stage and finished task states."""

    progress = getattr(owner, "_runProgressBar", None)
    state = str(getattr(owner, "_runState", "idle") or "idle")
    busy = run_state_presentation(state).busy
    if progress is not None and snapshot.total > 0:
        owner._runProgressHasData = True
        owner._runProgressCounts = (snapshot.done, snapshot.total)
        if busy and snapshot.done <= 0:
            progress.setRange(0, 0)
        else:
            progress.setRange(0, snapshot.total)
            progress.setValue(snapshot.done)
        progress.setFormat(progress_bar_format(snapshot.done, snapshot.total))
        refresh_run_progress_styles(owner)
    stage = getattr(owner, "_runStageLabel", None)
    if stage is not None:
        if snapshot.current_stage:
            owner._lastSnapshotStage = snapshot.current_stage
            if not getattr(owner, "_runHasLiveStageDetail", False):
                text = _text(owner, "current_stage", "Current stage") + f": {snapshot.current_stage}"
                owner._runLiveStageText = text
                stage.setText(text)
                _set_widget_visible(stage, True)
        elif not getattr(owner, "_runLiveStageText", ""):
            _set_widget_visible(stage, False)
    if getattr(owner, "_tableShowsLiveRun", False):
        for name, status in snapshot.finished_tasks:
            prefix = name.split(".", 1)[0]
            owner._update_run_status_table_item((name, prefix), status)


# ----------------------------------------------------------------------- results


def _set_expanded(item: Any, expanded: bool = True) -> None:
    if item is not None and hasattr(item, "setExpanded"):
        item.setExpanded(bool(expanded))


def _tree_count(obj: Any, attr: str) -> int:
    val = getattr(obj, attr, 0)
    val = val() if callable(val) else val
    return val if isinstance(val, int) else 0


def _iter_tree_items(tree: Any):
    def _walk(item: Any):
        yield item
        for child_index in range(_tree_count(item, "childCount")):
            child = item.child(child_index)
            if child is not None:
                yield from _walk(child)

    for top_index in range(_tree_count(tree, "topLevelItemCount")):
        top_item = tree.topLevelItem(top_index)
        if top_item is not None:
            yield from _walk(top_item)


def _clear_item_children(item: Any) -> None:
    if item is None:
        return
    if hasattr(item, "takeChildren"):
        item.takeChildren()
        return
    if hasattr(item, "removeChild"):
        for child_index in range(_tree_count(item, "childCount") - 1, -1, -1):
            child = item.child(child_index)
            if child is not None:
                item.removeChild(child)


def _user_role(qt_mod: Any, offset: int = 0):
    base = getattr(getattr(qt_mod, "Qt", None), "UserRole", 0)
    return base + offset if offset else base


def install_results_panel(owner: Any) -> None:
    """Build the unified Results directory tree, hidden run selector state and report buttons."""

    if getattr(owner, "_runHistoryComboBox", None) is not None:
        return
    results_layout = _layout_or_fallback(owner, "resultsLayout")
    if results_layout is None:
        return
    import qt

    section_box = getattr(getattr(owner, "ui", None), "resultsCollapsibleButton", None)
    if section_box is not None and hasattr(section_box, "connect"):
        section_box.connect(
            "contentsCollapsed(bool)",
            lambda collapsed: None if collapsed else owner._refresh_run_history(),
        )

    results_body_layout = _install_results_header(owner, qt, results_layout)

    empty_row = qt.QWidget()
    empty_layout = qt.QHBoxLayout(empty_row)
    empty_layout.setContentsMargins(0, 0, 0, 0)
    empty_label = qt.QLabel("No runs yet")
    owner._resultsEmptyLabel = empty_label
    _bind(owner, empty_label, "text", "no_runs_found")
    empty_layout.addWidget(empty_label, 1)
    results_body_layout.addWidget(empty_row)

    results_box = qt.QWidget()
    results_box.setObjectName("resultsTreeContainer")
    results_box.hide()
    results_body_layout.addWidget(results_box)
    owner._resultsEmptyRow = empty_row
    owner._resultsColumns = results_box
    results_box_layout = qt.QVBoxLayout(results_box)
    results_box_layout.setContentsMargins(0, 0, 0, 0)
    results_box_layout.setSpacing(4)

    combo = qt.QComboBox(results_box)
    combo.setObjectName("runHistoryComboBox")
    combo.setSizeAdjustPolicy(qt.QComboBox.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(14)
    combo.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed)
    combo.hide()
    combo.setToolTip(
        "Runs found under the output directory, the subject root, the batch root and folders added with "
        "“Add a run folder…”, newest first. GUI, command-line and batch runs all appear here, "
        "also from earlier Slicer sessions."
    )

    add_folder = qt.QPushButton("Add a run folder…")
    add_folder.setObjectName("addRunFolderButton")
    add_folder.clicked.connect(owner._on_browse_run_directory_clicked)
    _bind(owner, add_folder, "text", "add_run_folder")
    add_folder.hide()
    empty_layout.addWidget(add_folder)

    summary = qt.QLabel("Run summary will appear after a plan or run completes.")
    summary.wordWrap = True
    summary.setObjectName("runSummaryLabel")
    summary.hide()
    results_box_layout.addWidget(summary)

    details = qt.QLabel("No run selected.")
    details.setObjectName("runDetailsLabel")
    details.wordWrap = True
    try:
        details.setTextInteractionFlags(qt.Qt.TextSelectableByMouse | qt.Qt.TextSelectableByKeyboard)
    except Exception:
        pass
    details.hide()
    results_box_layout.addWidget(details)

    review = qt.QLabel("Recorded QC: consult report\nVisual review: pending")
    review.wordWrap = True
    review.setObjectName("reviewStateLabel")
    review.hide()
    results_box_layout.addWidget(review)

    files = qt.QTreeWidget()
    files.setObjectName("runOutputFiles")
    files.setHeaderHidden(True)
    files.setColumnCount(2)
    if hasattr(files, "setIndentation"):
        files.setIndentation(14)
    files.setMinimumHeight(150)
    files.setMaximumHeight(RESULTS_TREE_MAX_HEIGHT)
    files.setFrameShape(qt.QFrame.NoFrame)
    files.setMinimumWidth(0)
    header = files.header()
    if hasattr(header, "setStretchLastSection"):
        header.setStretchLastSection(False)
    header.setSectionResizeMode(0, qt.QHeaderView.Stretch)
    header.setSectionResizeMode(1, qt.QHeaderView.ResizeToContents)
    owner._runOutputFiles = files
    owner._runHistoryTree = files
    owner._runTreeItems = {}
    owner._runTreeRecords = ()
    owner._resultsGroupRuns = {}
    owner._selectingTreeItem = False
    results_box_layout.addWidget(files)

    def _on_tree_item_clicked(item, column) -> None:
        raw_path = item.data(0, _user_role(qt, 0))
        if isinstance(raw_path, (str, Path)) and raw_path and column == 1:
            _toggle_output(owner, item)
        _select_item_run(owner, item)

    def _on_tree_item_double_clicked(item, _column) -> None:
        raw_path = item.data(0, _user_role(qt, 0))
        if isinstance(raw_path, (str, Path)) and raw_path:
            _toggle_output(owner, item)

    files.itemClicked.connect(_on_tree_item_clicked)
    if hasattr(files, "itemExpanded"):
        files.itemExpanded.connect(lambda item: _record_expansion(owner, item, True))
    if hasattr(files, "itemCollapsed"):
        files.itemCollapsed.connect(lambda item: _record_expansion(owner, item, False))
    if hasattr(files, "itemDoubleClicked"):
        files.itemDoubleClicked.connect(_on_tree_item_double_clicked)
    files.setContextMenuPolicy(qt.Qt.CustomContextMenu)
    files.customContextMenuRequested.connect(lambda position: _output_menu(owner, position))

    owner._resultSceneButtons = []

    report_bar = qt.QWidget()
    report_bar.setObjectName("resultsReportBar")
    report_row = qt.QHBoxLayout(report_bar)
    report_row.setContentsMargins(0, 4, 0, 0)
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
    report_bar.hide()
    owner._resultsReportBar = report_bar
    results_body_layout.addWidget(report_bar)

    combo.currentIndexChanged.connect(lambda _index: owner._on_run_history_selection_changed())

    owner._runStatusPanel = getattr(owner.ui, "resultsCollapsibleButton", None) or results_box
    owner._runHistoryComboBox = combo
    owner._runHistoryRecords = ()
    owner._runDetailsLabel = details
    owner._runSummaryLabel = summary
    owner._reviewStateLabel = review
    owner._runStatusTable = None
    owner._runReportButton = report_button
    owner._runOutputTreeButton = output_tree_button
    owner._latestReportPath = None
    owner._latestOutputTreePath = None
    owner._latestResultManifestPath = None
    owner._tableShowsLiveRun = False


# ----------------------------------------------------- results header: collapse and pop-out


def _rebind(owner: Any, widget: Any, prop: str, key: str) -> None:
    """Point a language binding at a new text key (the tooltip changes with the state)."""

    bindings = [entry for entry in (getattr(owner, "_uiLanguageBindings", None) or ())
                if not (entry[0] is widget and entry[1] == prop)]
    owner._uiLanguageBindings = bindings
    _bind(owner, widget, prop, key)


_RESULTS_ICON_SVG_CACHE: dict[tuple[str, bool], bytes] = {}
_RESULTS_ICON_CACHE: dict[tuple[str, bool], Any] = {}


def _results_icon_svg(glyph: str, *, dark: bool) -> bytes:
    """Line icon for the Results header; the stroke follows the light or dark theme."""

    key = (glyph, bool(dark))
    cached = _RESULTS_ICON_SVG_CACHE.get(key)
    if cached is not None:
        return cached
    color = "#C9D1D9" if dark else "#4A5560"
    body = (
        f'<g fill="none" stroke="{color}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M9.5 2.5h4v4"/><path d="M13.5 2.5 8 8"/>'
        '<path d="M12 9.5v3a1 1 0 0 1-1 1H3.5a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1h3"/></g>'
    )
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 16 16">'
           f"{body}</svg>").encode("utf-8")
    _RESULTS_ICON_SVG_CACHE[key] = svg
    return svg


def _results_icon(qt_mod: Any, glyph: str, *, dark: bool) -> Any:
    key = (glyph, bool(dark))
    cached = _RESULTS_ICON_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        pixmap = qt_mod.QPixmap()
        if not pixmap.loadFromData(qt_mod.QByteArray(_results_icon_svg(glyph, dark=dark)), "SVG"):
            return None
        if hasattr(pixmap, "setDevicePixelRatio"):
            pixmap.setDevicePixelRatio(4.0)
        icon = qt_mod.QIcon(pixmap)
    except Exception:
        return None
    _RESULTS_ICON_CACHE[key] = icon
    return icon


def _queue_fit(owner: Any) -> None:
    queue_fit = getattr(owner, "_queueCompactFit", None)
    if callable(queue_fit):
        queue_fit()


def _style_header_button(qt: Any, button: Any) -> None:
    """FD7: the size and the style rules of the module's "..." buttons (``ui_algorithm_picker.picker_button``)."""

    button.setProperty("algorithmPickerButton", True)  # QToolButton[algorithmPickerButton="true"] in ui_styles
    button.setIconSize(qt.QSize(16, 16))
    button.setFixedSize(*RESULTS_HEADER_BUTTON_SIZE)
    button.setCursor(qt.Qt.PointingHandCursor)


def _install_results_header(owner: Any, qt: Any, results_layout: Any) -> Any:
    """Add the header row (collapse, pop-out), the pop-out placeholder row and the body container.

    The body container holds the empty-state row, the tree and the report bar. Collapse and pop-out act
    on this one widget; the tree code never shows or hides it. Returns the body's layout.
    """

    header = qt.QWidget()
    header.setObjectName("resultsHeaderRow")
    header.setFixedHeight(RESULTS_HEADER_BUTTON_SIZE[1])  # FD7: the row is no higher than its buttons
    header_layout = qt.QHBoxLayout(header)
    header_layout.setContentsMargins(0, 0, 0, 0)
    header_layout.setSpacing(2)
    header_layout.addStretch(1)
    collapse = qt.QToolButton()
    collapse.setObjectName("resultsCollapseButton")
    _style_header_button(qt, collapse)
    collapse.setArrowType(qt.Qt.DownArrow)
    collapse.setToolTip(_text(owner, "results_collapse", "Collapse results"))
    _bind(owner, collapse, "tooltip", "results_collapse")
    pop_out = qt.QToolButton()
    pop_out.setObjectName("resultsPopOutButton")
    _style_header_button(qt, pop_out)
    pop_out.setToolButtonStyle(qt.Qt.ToolButtonIconOnly)
    pop_out.setToolTip(_text(owner, "results_pop_out", "Open results in a separate window"))
    _bind(owner, pop_out, "tooltip", "results_pop_out")
    header_layout.addWidget(collapse)
    header_layout.addWidget(pop_out)
    results_layout.addWidget(header)

    popped = qt.QWidget()
    popped.setObjectName("resultsPoppedOutRow")
    popped_layout = qt.QHBoxLayout(popped)
    popped_layout.setContentsMargins(0, 0, 0, 0)
    popped_label = qt.QLabel("")
    popped_label.setObjectName("resultsPoppedOutLabel")
    popped_label.setText(_text(owner, "results_in_window", "Results are shown in a separate window."))
    _bind(owner, popped_label, "text", "results_in_window")
    dock_back = qt.QPushButton("")
    dock_back.setObjectName("resultsDockBackButton")
    dock_back.setText(_text(owner, "results_bring_back", "Bring back"))
    _bind(owner, dock_back, "text", "results_bring_back")
    popped_layout.addWidget(popped_label, 1)
    popped_layout.addWidget(dock_back)
    popped.hide()
    results_layout.addWidget(popped)

    body = qt.QWidget()
    body.setObjectName("resultsBody")
    body_layout = qt.QVBoxLayout(body)
    body_layout.setContentsMargins(0, 0, 0, 0)
    body_layout.setSpacing(4)
    results_layout.addWidget(body)

    owner._resultsLayout = results_layout
    owner._resultsHeaderRow = header
    owner._resultsCollapseButton = collapse
    owner._resultsPopOutButton = pop_out
    owner._resultsPoppedRow = popped
    owner._resultsBody = body
    owner._resultsCollapsed = False
    owner._resultsPoppedOut = False
    owner._resultsDialog = None
    owner._resultsDialogLayout = None
    owner._resultsBodyIndex = None
    owner._resultsWindowSize = RESULTS_WINDOW_SIZE
    collapse.clicked.connect(lambda *_args: toggle_results_collapsed(owner))
    pop_out.clicked.connect(lambda *_args: pop_out_results(owner))
    dock_back.clicked.connect(lambda *_args: dock_results_body(owner))
    refresh_results_window(owner)
    return body_layout


def toggle_results_collapsed(owner: Any) -> None:
    """Hide or show the Results body; everything except the header row. Session state only."""

    body = getattr(owner, "_resultsBody", None)
    if body is None:
        return
    collapsed = not bool(getattr(owner, "_resultsCollapsed", False))
    if collapsed and getattr(owner, "_resultsPoppedOut", False):
        dock_results_body(owner)
    owner._resultsCollapsed = collapsed
    _set_widget_visible(body, not collapsed)
    button = getattr(owner, "_resultsCollapseButton", None)
    if button is not None:
        import qt

        button.setArrowType(qt.Qt.RightArrow if collapsed else qt.Qt.DownArrow)
        key = "results_expand" if collapsed else "results_collapse"
        button.setToolTip(_text(owner, key, "Expand results" if collapsed else "Collapse results"))
        _rebind(owner, button, "tooltip", key)
    _queue_fit(owner)


def _widget_extent(widget: Any, name: str) -> Optional[int]:
    value = getattr(widget, name, None)
    value = value() if callable(value) else value
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def pop_out_results(owner: Any) -> None:
    """Move the Results body into one separate, non-modal window. There is still only one tree."""

    body = getattr(owner, "_resultsBody", None)
    layout = getattr(owner, "_resultsLayout", None)
    if body is None or layout is None or getattr(owner, "_resultsPoppedOut", False):
        return
    import qt
    import slicer

    if getattr(owner, "_resultsCollapsed", False):
        toggle_results_collapsed(owner)
    dialog = getattr(owner, "_resultsDialog", None)
    if dialog is None:
        dialog = qt.QDialog(slicer.util.mainWindow())
        dialog.setObjectName(RESULTS_WINDOW_NAME)
        dialog.setModal(False)
        dialog.setWindowFlags(qt.Qt.Window | qt.Qt.WindowMinMaxButtonsHint | qt.Qt.WindowCloseButtonHint)
        dialog_layout = qt.QVBoxLayout(dialog)
        dialog_layout.setContentsMargins(8, 8, 8, 8)
        # finished(int) covers the title-bar close and Esc; the dialog is hidden and reused, never deleted.
        dialog.connect("finished(int)", lambda *_args: dock_results_body(owner))
        owner._resultsDialog = dialog
        owner._resultsDialogLayout = dialog_layout
    dialog.setWindowTitle(_text(owner, "results_window_title", "Results - SlicerDSpace"))
    owner._resultsPoppedOut = True
    owner._resultsBodyIndex = layout.indexOf(body)
    layout.removeWidget(body)
    owner._resultsDialogLayout.addWidget(body)
    _set_widget_visible(body, True)
    tree = getattr(owner, "_runOutputFiles", None)
    if tree is not None:
        tree.setMaximumHeight(RESULTS_TREE_NO_LIMIT)
    _set_widget_visible(getattr(owner, "_resultsPoppedRow", None), True)
    pop_button = getattr(owner, "_resultsPopOutButton", None)
    if pop_button is not None:
        pop_button.enabled = False
    refresh_results_window(owner)
    width, height = getattr(owner, "_resultsWindowSize", RESULTS_WINDOW_SIZE)
    dialog.resize(width, height)
    dialog.show()
    _queue_fit(owner)


def dock_results_body(owner: Any) -> None:
    """Return the Results body to its place in the panel. Safe to call when nothing is popped out."""

    if not getattr(owner, "_resultsPoppedOut", False):
        return
    owner._resultsPoppedOut = False
    body = getattr(owner, "_resultsBody", None)
    layout = getattr(owner, "_resultsLayout", None)
    dialog = getattr(owner, "_resultsDialog", None)
    if dialog is not None:
        width, height = _widget_extent(dialog, "width"), _widget_extent(dialog, "height")
        if width and height:
            owner._resultsWindowSize = (width, height)
        window_layout = getattr(owner, "_resultsDialogLayout", None)
        if window_layout is not None and body is not None:
            window_layout.removeWidget(body)
    if body is not None and layout is not None:
        index = getattr(owner, "_resultsBodyIndex", None)
        layout.insertWidget(index if isinstance(index, int) and index >= 0 else layout.count(), body)
        _set_widget_visible(body, not getattr(owner, "_resultsCollapsed", False))
    tree = getattr(owner, "_runOutputFiles", None)
    if tree is not None:
        tree.setMaximumHeight(RESULTS_TREE_MAX_HEIGHT)
    _set_widget_visible(getattr(owner, "_resultsPoppedRow", None), False)
    pop_button = getattr(owner, "_resultsPopOutButton", None)
    if pop_button is not None:
        pop_button.enabled = True
    if dialog is not None:
        try:
            dialog.hide()
        except Exception:
            pass
    _queue_fit(owner)


def refresh_results_window(owner: Any, *, dark: Optional[bool] = None) -> None:
    """Theme refresh: redraw the header icon and give the Results window the module style."""

    pop_button = getattr(owner, "_resultsPopOutButton", None)
    if pop_button is None:
        return
    try:
        resolved_dark = _resolve_dark_mode(owner, dark)
    except Exception:
        resolved_dark = False
    try:
        import qt

        icon = _results_icon(qt, "popout", dark=resolved_dark)
        if icon is not None:
            pop_button.setIcon(icon)
    except Exception:
        pass
    dialog = getattr(owner, "_resultsDialog", None)
    root = getattr(owner, "_uiWidget", None)
    if dialog is not None and root is not None:
        from shared.ui_styles import set_stylesheet_if_changed

        attr = getattr(root, "styleSheet", None)
        style = attr() if callable(attr) else attr
        if isinstance(style, str):
            # FD7: the root rule (background, text colour) names the module root; give it to the window too.
            set_stylesheet_if_changed(
                dialog, style.replace(MODULE_ROOT_RULE, f"{MODULE_ROOT_RULE[:-2]}, QDialog#{RESULTS_WINDOW_NAME} {{")
            )
        # FD7: a window does not inherit its parent's palette; the panel body gets the root's palette
        # (``apply_placeholder_palette``: Window, Base, Text, Button ...), so the window gets a copy of it.
        palette_attr = getattr(root, "palette", None)
        try:
            root_palette = palette_attr() if callable(palette_attr) else palette_attr
            if root_palette is not None:
                import qt

                dialog.setPalette(qt.QPalette(root_palette))
        except Exception:
            pass


def _qt_value(obj: Any, name: str) -> Any:
    value = getattr(obj, name, None)
    return value() if callable(value) else value


def _child_items(parent: Any) -> list:
    """The direct child items of a tree widget (top-level items) or of a tree item."""

    if hasattr(parent, "topLevelItemCount"):
        count, getter = _tree_count(parent, "topLevelItemCount"), getattr(parent, "topLevelItem", None)
    else:
        count, getter = _tree_count(parent, "childCount"), getattr(parent, "child", None)
    if getter is None or count <= 0:
        return []
    return [child for child in (getter(index) for index in range(count)) if child is not None]


def _top_item(item: Any) -> Any:
    while item is not None and item.parent() is not None:
        item = item.parent()
    return item


def _item_key(qt_mod: Any, item: Any) -> Optional[tuple]:
    """A key that survives a rebuild: subject, (subject, family) or (subject, file path)."""

    role_key, role_path = _user_role(qt_mod, ROLE_KEY), _user_role(qt_mod, ROLE_PATH)
    parent = item.parent()
    if parent is None:
        key = item.data(0, role_key)
        return ("subject", key) if key else None
    grand = parent.parent()
    if grand is None:
        key = item.data(0, role_key)
        return ("family", parent.data(0, role_key), key) if key else None
    path = item.data(0, role_path)
    return ("file", grand.data(0, role_key), str(path)) if path else None


def _view_snapshot(qt_mod: Any, tree: Any) -> tuple[Optional[tuple], Any]:
    """The current item's key and the scroll position, taken before a rebuild."""

    key = None
    current = getattr(tree, "currentItem", None)
    if callable(current):
        try:
            item = current()
            key = _item_key(qt_mod, item) if item is not None else None
        except Exception:
            key = None
    scroll = None
    try:
        scroll = _qt_value(tree.verticalScrollBar(), "value")
    except Exception:
        pass
    return key, scroll


def _restore_scroll(tree: Any, scroll: Any) -> None:
    if scroll is None:
        return
    try:
        tree.verticalScrollBar().setValue(scroll)
    except Exception:
        pass


def _expansion_state(owner: Any) -> dict:
    """What the user expanded or collapsed in this session: ``subjects[key]`` and ``families[(key, family)]``."""

    state = getattr(owner, "_resultsExpansion", None)
    if not isinstance(state, dict):
        state = {"subjects": {}, "families": {}}
        owner._resultsExpansion = state
    return state


def _record_expansion(owner: Any, item: Any, expanded: bool) -> None:
    """Tree signal: remember a user change. A rebuild that applies the state is not a user change."""

    if getattr(owner, "_resultsApplyingState", False):
        return
    import qt

    key = _item_key(qt, item)
    if key is None:
        return
    state = _expansion_state(owner)
    if key[0] == "subject":
        state["subjects"][key[1]] = bool(expanded)
    elif key[0] == "family":
        state["families"][(key[1], key[2])] = bool(expanded)


def _apply_family_expansion(owner: Any, subject_item: Any, qt_mod: Any) -> None:
    role_key = _user_role(qt_mod, ROLE_KEY)
    subject_key = subject_item.data(0, role_key)
    families = _expansion_state(owner)["families"]
    for family in _child_items(subject_item):
        _set_expanded(family, families.get((subject_key, family.data(0, role_key)), False))


def running_subject_id(owner: Any) -> str:
    """The subject a single-subject job is running now, from the live run state; empty for a batch."""

    if str(getattr(owner, "_runState", "") or "") != "running":
        return ""
    context = getattr(owner, "_backgroundRunContext", None) or {}
    if context.get("batch_size") is not None or _is_campaign(context):
        return ""
    return str(context.get("subject_id") or "").strip()


def _is_campaign(context: dict) -> bool:
    """FD6: an experiment campaign runs many subjects; its context still carries the GUI's subject."""

    return bool(context.get("experiment_manifest_path"))


def _record_stamp(record: RunRecord) -> float:
    return record.timestamp.timestamp() if record.timestamp is not None else float("-inf")


def _subject_groups(owner: Any, records, running: str) -> list[dict]:
    """Group the records by subject, newest first; the subject of a running job comes first.

    A batch record never merges into a subject: it is its own group.
    """

    fallback = _text(owner, "run_selector_label", "Runs")
    groups: dict[str, dict] = {}
    for index, record in enumerate(records):
        label = record.subject_id or fallback
        key = f"batch:{record.manifest_path}" if record.is_batch else label
        group = groups.setdefault(
            key, {"key": key, "label": label, "indices": [], "batch": record.is_batch, "running": False}
        )
        group["indices"].append(index)
    if running:
        group = groups.get(running)
        if group is None:
            group = groups[running] = {"key": running, "label": running, "indices": [], "batch": False,
                                       "running": True}
        group["running"] = not group["batch"]
    for group in groups.values():
        group["indices"].sort(key=lambda index: (-_record_stamp(records[index]), index))
        group["indices"] = tuple(group["indices"])
    return sorted(
        groups.values(),
        key=lambda group: (
            0 if group["running"] else 1,
            -(_record_stamp(records[group["indices"][0]]) if group["indices"] else float("inf")),
            group["indices"][0] if group["indices"] else -1,
        ),
    )


def _read_manifest(record: RunRecord) -> Optional[dict]:
    try:
        manifest = json.loads(record.manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return manifest if isinstance(manifest, dict) and manifest else None


def _fill_subject_children(owner: Any, subject_item: Any, records, indices, manifests: Optional[dict] = None) -> None:
    """Add the families and files of every run of a subject, newest run first, each family and file once."""

    for index in indices:
        record = records[index]
        if record.is_batch:
            continue
        manifest = manifests[index] if manifests and index in manifests else _read_manifest(record)
        if isinstance(manifest, dict) and manifest:
            _populate_manifest_output_items(
                owner, subject_item, manifest, manifest_path=record.manifest_path, run_index=index
            )
    _order_families(subject_item)


def _order_families(subject_item: Any) -> None:
    """Keep the families in their usual order, also when an older run adds one that a newer run lacks."""

    import qt as qt_mod

    role_key = _user_role(qt_mod, ROLE_KEY)
    rank = {family: position for position, family in enumerate(RESULT_FAMILY_ORDER)}
    children = _child_items(subject_item)
    ordered = sorted(children, key=lambda child: rank.get(child.data(0, role_key), len(rank)))
    if ordered != children:
        subject_item.takeChildren()
        for child in ordered:
            subject_item.addChild(child)


def _build_subject_item(owner: Any, qt_mod: Any, tree: Any, records, group: dict) -> Any:
    newest = group["indices"][0] if group["indices"] else None
    if group["running"]:
        status = _text(owner, "results_running", "running")
    else:
        status = status_label(records[newest].status)
    if group["batch"]:
        name = records[newest].run_name or records[newest].manifest_path.parent.name
        text = f"{group['label']} · {name} · {status}"
    else:
        text = f"{group['label']} · {status}"
    item = qt_mod.QTreeWidgetItem(tree, [text])
    item.setData(0, _user_role(qt_mod, ROLE_KEY), group["key"])
    if newest is not None:
        item.setData(0, _user_role(qt_mod, ROLE_MANIFEST), str(records[newest].manifest_path))
        item.setData(0, _user_role(qt_mod, ROLE_RUN), newest)
        item.setToolTip(0, "\n\n".join(run_record_details(records[index]) for index in group["indices"]))
    else:
        item.setToolTip(0, status)
    return item


def _populate_run_tree(owner, records, current_index):
    """Rebuild the Results tree: one item per subject (runs merged), then families, then files."""

    tree = getattr(owner, "_runOutputFiles", None) or getattr(owner, "_runHistoryTree", None)
    if tree is None:
        return
    import qt

    owner._runTreeRecords = tuple(records)  # as scanned; the labels below use the session's end states (FD4)
    records = _with_ended_runs(owner, records)
    running = running_subject_id(owner)
    has_content = bool(records) or bool(running)
    columns = owner._resultsColumns
    columns.setVisible(has_content)
    owner._resultsEmptyLabel.setVisible(not has_content)
    empty_row = getattr(owner, "_resultsEmptyRow", None)
    if empty_row is not None:
        empty_row.setVisible(not has_content)
    report_bar = getattr(owner, "_resultsReportBar", None)
    if report_bar is not None:
        report_bar.setVisible(False)
    snapshot = _view_snapshot(qt, tree)
    state = _expansion_state(owner)
    run_items: dict = {}
    group_runs: dict = {}
    selected_item = None
    owner._resultsApplyingState = True
    try:
        tree.clear()
        for position, group in enumerate(_subject_groups(owner, records, running)):
            item = _build_subject_item(owner, qt, tree, records, group)
            if not group["batch"]:
                _fill_subject_children(owner, item, records, group["indices"])
                _apply_family_expansion(owner, item, qt)
            # Default: only the newest subject is open, and only down to its (collapsed) families.
            _set_expanded(item, state["subjects"].get(group["key"], position == 0))
            for index in group["indices"]:
                run_items[index] = item
                group_runs[index] = group["indices"]
            if current_index in group["indices"]:
                selected_item = item
        owner._runTreeItems = run_items
        owner._resultsGroupRuns = group_runs
        _restore_current_item(qt, tree, snapshot[0], selected_item)
        _restore_scroll(tree, snapshot[1])
    finally:
        owner._resultsApplyingState = False
    _refresh_output_visibility(owner)
    _queue_fit(owner)


def _item_is_expanded(item: Any) -> bool:
    is_expanded = getattr(item, "isExpanded", None)
    return bool(is_expanded()) if callable(is_expanded) else True


def _shown_item(item: Any) -> Any:
    """The item itself, or its highest collapsed ancestor.

    FD2: Qt expands every collapsed ancestor of a new current item (``currentChanged`` -> ``scrollTo``). That
    would undo a collapse of the user, and in a hidden tree it would happen later, outside the guard.
    """

    target = item
    ancestor = item.parent()
    while ancestor is not None:
        if not _item_is_expanded(ancestor):
            target = ancestor
        ancestor = ancestor.parent()
    return target


def _restore_current_item(qt_mod: Any, tree: Any, key: Optional[tuple], selected_subject: Any) -> None:
    """Keep the item the user had when it is still in the selected subject; else select the subject."""

    if key is not None and selected_subject is not None:
        for item in _iter_tree_items(tree):
            if _item_key(qt_mod, item) == key and _top_item(item) is selected_subject:
                tree.setCurrentItem(_shown_item(item))
                return
    if selected_subject is not None:
        tree.setCurrentItem(selected_subject)


def _rebuild_tree(owner: Any) -> None:
    """Rebuild the tree from the records of the last scan (no rescan), e.g. when a run starts or ends."""

    if getattr(owner, "_runOutputFiles", None) is None or getattr(owner, "_resultsColumns", None) is None:
        return
    combo = getattr(owner, "_runHistoryComboBox", None)
    index = _qt_value(combo, "currentIndex") if combo is not None else 0
    _populate_run_tree(owner, getattr(owner, "_runTreeRecords", ()) or (), index if isinstance(index, int) else 0)


def _on_running_changed(owner: Any, now_running: bool) -> None:
    """A single-subject job started or ended: remember its Output Folder, show or relabel its subject."""

    try:
        if now_running:
            _forget_ended_runs(owner, _job_manifest_keys(owner))
            save_run_output_folder(owner)
            subject = running_subject_id(owner)
            if subject:
                _expansion_state(owner)["subjects"][subject] = True
        else:
            _remember_ended_run(owner, str(getattr(owner, "_runState", "") or ""))
        _rebuild_tree(owner)
    except Exception:
        pass


# FD4: a job of this session that ended while its manifest on disk still says "running" (a killed worker leaves
# its checkpoint). Session memory only, keyed by the real path of the manifest; the manifest is never changed.
_ENDED_RUN_STATES = ("failed", "cancelled")


def _manifest_key(path: Any) -> str:
    return os.path.realpath(str(path))


def _manifest_says_running(path: Any) -> bool:
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(manifest, dict) and str(manifest.get("status") or "") == "running"


def _job_manifest_keys(owner: Any) -> set:
    """The manifest of the current job: the path the worker printed and the planned path (not a batch progress file)."""

    paths = [getattr(owner, "_backgroundManifestPath", None)]
    if not getattr(owner, "_runProgressIsBatch", False):
        paths.append(getattr(owner, "_runProgressManifestPath", None))
    return {_manifest_key(path) for path in paths if path}


def _ended_runs(owner: Any) -> dict:
    ended = getattr(owner, "_resultsEndedRuns", None)
    if not isinstance(ended, dict):
        ended = {}
        owner._resultsEndedRuns = ended
    return ended


def _forget_ended_runs(owner: Any, keys) -> None:
    ended = _ended_runs(owner)
    for key in keys:
        ended.pop(key, None)


def _remember_ended_run(owner: Any, state: str) -> None:
    if state not in _ENDED_RUN_STATES:
        return
    ended = _ended_runs(owner)
    for key in _job_manifest_keys(owner):
        if _manifest_says_running(key):
            ended[key] = state


def _with_ended_runs(owner: Any, records) -> tuple:
    """The records as the Results show them: a run that ended in this session is not shown as running.

    An entry is dropped when its manifest no longer says running, or when a job runs for it now.
    """

    records = tuple(records)
    ended = getattr(owner, "_resultsEndedRuns", None)
    if not ended:
        return records
    live = _job_manifest_keys(owner) if str(getattr(owner, "_runState", "") or "") == "running" else set()
    for key in list(ended):
        if key in live or not _manifest_says_running(key):
            ended.pop(key, None)
    if not ended:
        return records
    return tuple(
        replace(record, status=ended[_manifest_key(record.manifest_path)])
        if record.status == "running" and _manifest_key(record.manifest_path) in ended
        else record
        for record in records
    )


def save_run_output_folder(owner: Any) -> None:
    """Remember the Output Folder of a run, so a later Slicer session lists its results too.

    FD5: with an empty Output Folder field the results go to ``<subject root>/outputs``; then the subject root
    holds the run and is remembered. A campaign does not run the GUI's subject; it remembers its own folder
    (``ui_experiments.submit``).
    """

    context = getattr(owner, "_backgroundRunContext", None) or {}
    folder = _node_text(owner, "outputDirectory")
    if not folder and not _is_campaign(context):
        subject_root = context.get("subject_root")
        folder = str(subject_root).strip() if subject_root else _node_text(owner, "inputDirectory")
    if folder:
        save_history_directory(folder)


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


def _item_manifest_path(owner: Any, item: Any, qt_mod: Any) -> Optional[Path]:
    raw_item_manifest = item.data(0, _user_role(qt_mod, 2))
    if isinstance(raw_item_manifest, Path):
        return raw_item_manifest
    if isinstance(raw_item_manifest, str) and raw_item_manifest:
        return Path(raw_item_manifest)
    fallback = getattr(owner, "_latestResultManifestPath", None)
    return Path(fallback) if isinstance(fallback, (str, Path)) and fallback else None


def _toggle_output(owner, item):
    import qt
    import slicer

    raw_path = item.data(0, _user_role(qt, 0))
    if not isinstance(raw_path, (str, Path)) or not raw_path:
        return
    manifest_path = _item_manifest_path(owner, item, qt)
    if manifest_path is None:
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


def _select_item_run(owner: Any, item: Any) -> None:
    """Select the run of a tree item in the run selector, without rebuilding the merged subject."""

    import qt

    combo = getattr(owner, "_runHistoryComboBox", None)
    run_idx = item.data(0, _user_role(qt, ROLE_RUN)) if item is not None else None
    if combo is None or not isinstance(run_idx, int) or isinstance(run_idx, bool):
        return
    if combo.currentIndex != run_idx:
        owner._selectingTreeItem = True
        try:
            combo.setCurrentIndex(run_idx)
        finally:
            owner._selectingTreeItem = False


def _same_path(left: Any, right: Any) -> bool:
    try:
        return Path(left).resolve() == Path(right).resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def _select_manifest_run(owner: Any, manifest_path: Any) -> bool:
    """Select the current record that has ``manifest_path`` in the run selector; False when none has it.

    FD8: reads no tree item, so it is safe after a nested event loop that may have rebuilt the tree.
    """

    combo = getattr(owner, "_runHistoryComboBox", None)
    if combo is None or manifest_path is None:
        return False
    records = getattr(owner, "_runHistoryRecords", ()) or ()
    run_idx = next((index for index, record in enumerate(records)
                    if _same_path(record.manifest_path, manifest_path)), None)
    if run_idx is None:
        return False
    if combo.currentIndex != run_idx:
        owner._selectingTreeItem = True
        try:
            combo.setCurrentIndex(run_idx)
        finally:
            owner._selectingTreeItem = False
    return True


def _output_menu(owner, position):
    import qt
    import slicer

    tree = owner._runOutputFiles
    item = tree.itemAt(position)
    raw_path = item.data(0, _user_role(qt, 0)) if item is not None else None
    manifest_path = _item_manifest_path(owner, item, qt) if item is not None else getattr(owner, "_latestResultManifestPath", None)
    if not isinstance(raw_path, (str, Path)) or not raw_path or manifest_path is None:
        return
    nodes = _output_nodes(slicer, manifest_path, Path(raw_path))
    is_seg = item is not None and item.data(0, _user_role(qt, 1)) == "segmentation"
    if not nodes and not is_seg:
        return
    menu = qt.QMenu(tree)
    zh = str(getattr(owner, "_onboardingLanguage", "en")).startswith("zh")
    edit_action = (
        menu.addAction("编辑组织分割副本" if zh else "Edit tissue copy in Segment Editor")
        if is_seg
        else None
    )
    unload = menu.addAction("从场景卸载" if zh else "Unload from scene") if nodes else None
    chosen = menu.exec_(tree.viewport().mapToGlobal(position))
    if edit_action is not None and chosen == edit_action:
        # FD3: _manage_result_scene acts on the selected run; in a merged subject the file may own an older one.
        # FD8: exec_ ran a nested event loop; a refresh may have deleted `item` and reordered the records,
        # so resolve the run from the manifest path taken before the menu opened, never from the item.
        if _select_manifest_run(owner, manifest_path):
            owner._manage_result_scene("edit_tissue")
    elif unload is not None and chosen == unload:
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
    if tree is None:
        return
    import qt
    from shared.result_scene import owned_nodes

    fallback_manifest = getattr(owner, "_latestResultManifestPath", None)
    by_path_by_manifest = {}

    def _nodes_for_manifest(mpath: Optional[Path]):
        if mpath is None:
            return {}
        key = str(mpath.resolve())
        cached = by_path_by_manifest.get(key)
        if cached is None:
            cached = {}
            for node in owned_nodes(slicer, key):
                cached.setdefault(node.GetAttribute("SlicerDDSurfer.SourcePath"), []).append(node)
            by_path_by_manifest[key] = cached
        return cached

    zh = str(getattr(owner, "_onboardingLanguage", "en")).startswith("zh")
    for item in _iter_tree_items(tree):
        raw_path = item.data(0, _user_role(qt, 0))
        if not isinstance(raw_path, (str, Path)) or not raw_path:
            continue
        if item.text(1) == "↗":
            continue
        mpath = _item_manifest_path(owner, item, qt) or (Path(fallback_manifest) if fallback_manifest else None)
        by_path = _nodes_for_manifest(mpath)
        visible = _output_visible(slicer, by_path.get(str(raw_path), ()))
        item.setIcon(1, _eye_icon(qt, visible))
        item.setText(1, "")
        item.setToolTip(1, ("隐藏" if zh else "Hide") if visible else ("显示" if zh else "Show"))


def _is_internal_cache_path(
    path: Path,
    subject_root: Optional[Path],
    output_root: Optional[Path] = None,
) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    if subject_root is not None:
        try:
            rel = resolved.relative_to(subject_root)
            if "cache" in rel.parts:
                return True
        except ValueError:
            pass
    if output_root is not None:
        workspace = output_root.parent if output_root.name.lower() in {"output", "outputs"} else output_root
        cache_dir = workspace / "cache"
        try:
            resolved.relative_to(cache_dir.resolve())
            return True
        except (OSError, ValueError):
            pass
    posix = resolved.as_posix()
    return "/cache/stages/" in posix or "/cache/reports/" in posix


def _populate_manifest_output_items(
    owner: Any,
    parent: Any,
    manifest: dict[str, Any],
    *,
    manifest_path: Optional[Path] = None,
    run_index: Optional[int] = None,
) -> None:
    import qt
    from shared.result_scene import result_candidates

    subject_root_raw = manifest.get("subject_root")
    subject_root: Optional[Path] = None
    if subject_root_raw:
        try:
            subject_root = Path(str(subject_root_raw)).expanduser().resolve()
        except OSError:
            subject_root = Path(str(subject_root_raw)).expanduser()

    output_root_raw = manifest.get("output_root")
    output_root: Optional[Path] = None
    if output_root_raw:
        try:
            output_root = Path(str(output_root_raw)).expanduser().resolve()
        except OSError:
            output_root = Path(str(output_root_raw)).expanduser()

    candidates, _ = result_candidates(
        manifest,
        allow_bundle_directory_scan=False,
        include_preprocessing=True,
        include_brain_mask=True,
    )
    filtered_candidates = [
        item for item in candidates
        if not _is_internal_cache_path(Path(item["path"]), subject_root, output_root=output_root)
    ]
    visual_paths = {item["path"] for item in filtered_candidates
                    if Path(item["path"]).suffix.lower() not in {".csv", ".tsv"}}
    seg_paths = {item["path"] for item in filtered_candidates
                 if str(item.get("product") or "").startswith("segmentation")}
    groups = {}
    for candidate in filtered_candidates:
        path = candidate["path"]
        groups.setdefault(candidate["product"], {})[path] = path in visual_paths
    for task in manifest.get("tasks") or []:
        if not isinstance(task, dict) or not isinstance(task.get("outputs"), dict):
            continue
        for value in task["outputs"].values():
            if not isinstance(value, str):
                continue
            path = Path(value)
            if (
                path.suffix.lower() == ".json"
                and path.is_absolute()
                and path.is_file()
                and not _is_internal_cache_path(path, subject_root, output_root=output_root)
            ):
                groups.setdefault(str(task.get("name") or "qc"), {})[str(path.resolve())] = False

    resolved_manifest = manifest_path if manifest_path is not None else getattr(owner, "_latestResultManifestPath", None)
    qc_base = Path(str(output_root_raw)).expanduser() if output_root_raw else None
    qc_dir = (qc_base / "06_qc") if qc_base is not None else None
    if (qc_dir is None or not qc_dir.is_dir()) and resolved_manifest is not None:
        fallback_qc_dir = Path(resolved_manifest).parent.parent / "06_qc"
        if fallback_qc_dir.is_dir():
            qc_dir = fallback_qc_dir
    if qc_dir is not None and qc_dir.is_dir():
        for txt_path in sorted(qc_dir.rglob("*.txt")):
            if txt_path.is_file() and not _is_internal_cache_path(txt_path, subject_root, output_root=output_root):
                groups.setdefault("qc", {})[str(txt_path.resolve())] = False

    zh = str(getattr(owner, "_onboardingLanguage", "en")).startswith("zh")
    titles = {
        "anatomy": ("Anatomical priors", "解剖先验"),
        "preprocessing": ("Preprocessing", "预处理"),
        "microstructure": ("Microstructure", "微结构"),
        "tractography": ("Tractography", "纤维追踪"),
        "bundles": ("Fibre bundles", "纤维束"),
        "connectome": ("Connectome", "连接组"),
        "qc": ("Quality control", "质量控制"),
    }
    families = {}
    for product, paths in groups.items():
        family = product.split(".", 1)[0]
        if family in {"surface", "segmentation", "parcellation", "postprocess"}:
            family = "anatomy"
        families.setdefault(family, {}).update(paths)
    role_manifest = _user_role(qt, ROLE_MANIFEST)
    role_run = _user_role(qt, ROLE_RUN)
    role_key = _user_role(qt, ROLE_KEY)
    manifest_str = str(resolved_manifest) if resolved_manifest is not None else None
    family_order = RESULT_FAMILY_ORDER
    ordered_families = [
        *[family for family in family_order if family in families],
        *[family for family in families if family not in family_order],
    ]
    # The parent may already hold families and files of a newer run: add to them, skip what is there.
    existing: dict = {}
    seen_paths: set = set()
    for child in _child_items(parent):
        if not callable(getattr(child, "data", None)):
            continue
        family_key = child.data(0, role_key)
        if family_key:
            existing[family_key] = child
        for leaf in _child_items(child):
            seen_paths.add(str(leaf.data(0, _user_role(qt, ROLE_PATH))))
    for family in ordered_families:
        paths = {path: visual for path, visual in families[family].items() if path not in seen_paths}
        if not paths:
            continue
        seen_paths.update(paths)
        caption = titles.get(family, (family, family))[int(zh)]
        group = existing.get(family)
        if group is None:
            group = qt.QTreeWidgetItem(parent, [f"{caption} ({len(paths)})"])
            group.setData(0, role_key, family)
            if run_index is not None:
                group.setData(0, role_run, run_index)
            if manifest_str is not None and run_index is not None:
                group.setData(0, role_manifest, manifest_str)
            existing[family] = group
            added_to_existing = False
        else:
            added_to_existing = True
        for path, visual in paths.items():
            item = qt.QTreeWidgetItem(group, [Path(path).name, "○" if visual else "↗"])
            item.setData(0, _user_role(qt, ROLE_PATH), path)
            if path in seg_paths:
                item.setData(0, _user_role(qt, ROLE_KIND), "segmentation")
            if manifest_str is not None:
                item.setData(0, role_manifest, manifest_str)
            if run_index is not None:
                item.setData(0, role_run, run_index)
            item.setToolTip(0, path)
        if added_to_existing:
            group.setText(0, f"{caption} ({_tree_count(group, 'childCount')})")


def _target_run_index(owner: Any, run_items: dict, run_index: Optional[int]) -> Optional[int]:
    if run_index in run_items:
        return run_index
    latest = getattr(owner, "_latestResultManifestPath", None)
    if latest is not None:
        wanted = str(latest)
        for index, record in enumerate(getattr(owner, "_runTreeRecords", ()) or ()):
            if index in run_items and str(record.manifest_path) == wanted:
                return index
        return None
    combo = getattr(owner, "_runHistoryComboBox", None)
    combo_idx = getattr(combo, "currentIndex", None)
    return combo_idx if isinstance(combo_idx, int) and combo_idx in run_items else None


def _show_output_files(owner, manifest, *, run_index: Optional[int] = None):
    tree = getattr(owner, "_runOutputFiles", None)
    if tree is None:
        return
    import qt

    run_items = getattr(owner, "_runTreeItems", None)
    if isinstance(run_items, dict) and run_items:
        target_idx = _target_run_index(owner, run_items, run_index)
        if target_idx is not None:
            # Rebuild the subject of this run from all its runs; the run's manifest is the one in hand.
            subject_item = run_items[target_idx]
            indices = (getattr(owner, "_resultsGroupRuns", {}) or {}).get(target_idx, (target_idx,))
            key, scroll = _view_snapshot(qt, tree)
            owner._resultsApplyingState = True
            try:
                _clear_item_children(subject_item)
                _fill_subject_children(
                    owner, subject_item, getattr(owner, "_runTreeRecords", ()), indices,
                    {target_idx: manifest if isinstance(manifest, dict) else {}},
                )
                _apply_family_expansion(owner, subject_item, qt)
                # FD1: the clear removed the current item from the tree, so ``currentItem()`` is None now.
                # The snapshot taken before the clear tells whether it was in this subject.
                if key is not None and key[1] == subject_item.data(0, _user_role(qt, ROLE_KEY)):
                    _restore_current_item(qt, tree, key, subject_item)
                _restore_scroll(tree, scroll)
            finally:
                owner._resultsApplyingState = False
        _refresh_output_visibility(owner)
        return

    tree.clear()
    if isinstance(manifest, dict) and manifest:
        _populate_manifest_output_items(
            owner,
            tree,
            manifest,
            manifest_path=getattr(owner, "_latestResultManifestPath", None),
        )
        for family in _child_items(tree):
            _set_expanded(family, True)
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


def _effective_subject_id_for_owner(owner: Any) -> str:
    if hasattr(owner, "_effective_subject_id"):
        try:
            return str(owner._effective_subject_id() or "").strip()
        except Exception:
            pass
    sid = _node_text(owner, "subjectId")
    if sid:
        return sid
    subj = _node_text(owner, "inputDirectory")
    if subj:
        from shared.subject_id import derive_subject_id

        return str(derive_subject_id(subj) or "").strip()
    return ""


def _is_batch_mode_for_owner(owner: Any) -> bool:
    node = getattr(owner, "_parameterNode", None)
    if not node:
        return False
    cohort = bool(
        getattr(node, "experimentsEnabled", False)
        and getattr(node, "experimentScope", "subject") == "cohort"
    )
    return cohort or bool(getattr(node, "batchMode", False))


def history_roots(owner: Any):
    subject_root_text = _node_text(owner, "inputDirectory")
    output_root_text = _node_text(owner, "outputDirectory")
    batch_root_text = _node_text(owner, "batchRoot")
    # Results of earlier plugin sessions stay: the remembered folders are always scanned.
    extra_dirs = (*(getattr(owner, "_manualRunHistoryDirs", ()) or ()), *load_history_directories())
    return candidate_run_roots(
        subject_root_text=subject_root_text,
        output_root_text=output_root_text,
        batch_root_text=batch_root_text,
        subject_id=_effective_subject_id_for_owner(owner),
        batch_mode=_is_batch_mode_for_owner(owner),
        extra_dirs=extra_dirs,
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
        for record in _with_ended_runs(owner, discovery.records):
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
    if path is None:
        if not getattr(owner, "_runTreeItems", None) and not running_subject_id(owner):
            _show_output_files(owner, {})
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
        if not record.is_batch:
            _show_output_files(owner, manifest)
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
    if getattr(owner, "_selectingTreeItem", False):
        _refresh_output_visibility(owner)
    else:
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
    manual = list(getattr(owner, "_manualRunHistoryDirs", ()) or ())
    if directory not in manual:
        manual.append(directory)
        owner._manualRunHistoryDirs = tuple(manual)
    refresh_run_history(owner, select_path=discovery.records[0].manifest_path)
    slicer.util.showStatusMessage(
        f"Found {len(discovery.records)} run(s) under {directory}; selected the newest.", 4000
    )
