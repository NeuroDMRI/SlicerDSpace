"""Qt glue for the "4. Run" progress area and the "5. Results" run selector.

The pure logic (manifest discovery/validation, progress maths, button rules)
lives in ``shared/run_history.py`` and ``shared/ui_run_state.py``; this module
only builds widgets and moves data between them and the owner widget.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from shared.run_history import (
    ManifestError,
    RunRecord,
    archived_run_manifests,
    candidate_run_roots,
    discover_runs,
    dump_hidden_runs,
    hidden_stamp,
    parse_hidden_runs,
    parse_run_manifest,
    progress_from_manifest,
    read_archived_manifest,
    remember_directory,
    run_manifest_key,
    run_record_brief,
    run_record_details,
    run_record_key,
    run_folder_path,
    run_record_label,
    split_hidden,
    subject_folder_from_roots,
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
# Batch 2: runs removed from the Results list (one JSON string: slot key -> the run's own stamp). Files stay on disk.
RESULTS_HIDDEN_SETTING = "SlicerDSpace/UserSettings/resultsHiddenRuns"
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
RESULT_FAMILY_TITLES = {
    "anatomy": ("Anatomical priors", "解剖先验"),
    "preprocessing": ("Preprocessing", "预处理"),
    "microstructure": ("Microstructure", "微结构"),
    "tractography": ("Tractography", "纤维追踪"),
    "bundles": ("Fibre bundles", "纤维束"),
    "connectome": ("Connectome", "连接组"),
    "qc": ("Quality control", "质量控制"),
}
# Results subgroups (Batch 2): the second level inside these families, in this order. A product of one of these
# families that is not listed gets its own subgroup after the listed ones, captioned with the part after the first dot.
# Captions reuse the GUI's names for the same thing (one name for one thing): for anatomy the onboarding labels
# "segmentation", "parcellation", "surface", "surface_postprocess" (also the live stage names; the anatomical priors
# popup shows shorter forms), and for the algorithms the "..." popup labels (shared.algorithm_selection.LABELS,
# shown untranslated in both languages).
RESULT_SUBGROUP_ORDER = {
    "anatomy": ("segmentation", "parcellation", "surfaces", "postprocess"),
    "microstructure": ("dti", "freewater", "ifw", "noddi", "dki", "mkcurve"),
    "tractography": ("ukf", "csd_act", "dti_det", "saf"),
    "bundles": ("wmql", "wma", "tractcloud"),
    "connectome": ("streamline_count", "sift2_weighted"),
}
RESULT_SUBGROUP_CAPTIONS = {
    ("anatomy", "segmentation"): ("Tissue segmentation", "组织分割"),
    ("anatomy", "parcellation"): ("Brain parcellation", "脑区分割"),
    ("anatomy", "surfaces"): ("Cortical surfaces", "皮层表面"),
    ("anatomy", "postprocess"): ("Surface postprocess", "表面后处理"),
    ("microstructure", "dti"): ("DTI", "DTI"),
    ("microstructure", "freewater"): ("FW", "FW"),
    ("microstructure", "ifw"): ("iFW", "iFW"),
    ("microstructure", "noddi"): ("NODDI", "NODDI"),
    ("microstructure", "dki"): ("DKI", "DKI"),
    ("microstructure", "mkcurve"): ("MK-Curve", "MK-Curve"),
    ("tractography", "ukf"): ("UKF", "UKF"),
    ("tractography", "csd_act"): ("CSD / ACT", "CSD / ACT"),
    ("tractography", "dti_det"): ("DTI", "DTI"),
    ("tractography", "saf"): ("CSD + SAF", "CSD + SAF"),
    ("bundles", "wmql"): ("WMQL", "WMQL"),
    ("bundles", "wma"): ("WMA", "WMA"),
    ("bundles", "tractcloud"): ("TractCloud", "TractCloud"),
    ("connectome", "streamline_count"): ("Fiber count", "Fiber count"),
    ("connectome", "sift2_weighted"): ("SIFT2", "SIFT2"),
}
_SUBGROUP_ALIASES = {("tractography", "act"): "csd_act"}
# Results sources (batch 2): in the bundles and connectome families a subgroup is one (algorithm, tractography source)
# pair, id "<algorithm>@<source>" ("<algorithm>" when the source is unknown), caption "WMQL · UKF". The source order
# after the unknown source, and the captions: the tractography subgroup names above (one name for one thing); the GUI
# has no name yet for a user's own tractogram file, so "supplied" gets its own caption. Review R7: a SAF child
# publishes to "<product>/saf/" and "05_connectome/saf/" (orchestrator._bundle_publish_backend), caption "CSD + SAF".
RESULT_SOURCE_FAMILIES = ("bundles", "connectome")
RESULT_SOURCE_ORDER = ("ukf", "csd_act", "saf", "dti_det", "supplied")
RESULT_TRACKING_SOURCES = ("ukf", "csd_act", "saf", "dti_det")  # what an "@<backend>" suffix or tracking_backend names
# Review R6: the bundle algorithms that always read the tracking variant's own tractogram. The orchestrator config has
# a user-file input for WMA and TractCloud only (wma_input_tractography, tractcloud_input_tractography), and the
# manifest that carries "@<backend>" (multi-tracking parent) records no input file, so only these may take the source
# from the variant (rule 2 step b), unless the manifest records bundle_tractography_backend (see _task_source).
RESULT_OWN_TRACTOGRAM_BUNDLES = ("wmql",)
# Review R8: the published folders of shared/io_layout.py (SubjectLayout.bundles, .connectome, .cache_stages). The
# names are literals here because io_layout only spells them inside the layout properties; a test pins them.
RESULT_BUNDLES_FOLDER = "04_bundles"
RESULT_CONNECTOME_FOLDER = "05_connectome"
RESULT_STAGE_CACHE_PARTS = ("cache", "stages")
RESULT_SOURCE_CAPTIONS = {
    **{source: RESULT_SUBGROUP_CAPTIONS[("tractography", source)] for source in RESULT_TRACKING_SOURCES},
    "supplied": ("supplied file", "自备文件"),
}
RESULT_SOURCE_SEPARATOR = " · "
# The ROLE_KEY of a subgroup row is "<family>::<subgroup>" (e.g. "anatomy::surfaces"). The row kind does not come from
# this text (an unknown family from a manifest may hold "::" in its key): a subgroup row carries SUBGROUP_ROW_KIND.
SUBGROUP_KEY_SEPARATOR = "::"
SUBGROUP_ROW_KIND = "subgroup"  # ROLE_KIND of a subgroup row; file rows use "segmentation" / FIBER_ROW_KIND
# Item data roles of the Results tree, as offsets from Qt.UserRole.
ROLE_PATH, ROLE_KIND, ROLE_MANIFEST, ROLE_RUN, ROLE_KEY = 0, 1, 2, 3, 4
ROLE_FOLDER = 5  # Batch 2: the folder on disk of a subject row (the Open folder button); text, empty when unknown
ROLE_QC_MARK = 6  # a subgroup row whose bundle task's recorded QC is not passed: "failed|unreadable<TAB>reason"


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
    if section_box is not None and hasattr(section_box, "connect") and hasattr(section_box, "collapsed"):
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
    # Batch 2: after a Clear the tree container is hidden; this button is then the only way back.
    show_hidden = qt.QPushButton("")
    show_hidden.setObjectName("resultsShowHiddenButton")
    show_hidden.setText(_show_hidden_text(owner, 0))  # no language binding: the text carries a count (R4)
    show_hidden.clicked.connect(lambda *_args: show_hidden_results(owner))
    show_hidden.hide()
    empty_layout.addWidget(show_hidden)
    owner._resultsShowHiddenButton = show_hidden
    owner._resultsHiddenRecords = ()
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

    if hasattr(qt, "QSlider") and hasattr(qt, "QSpinBox"):
        zh = str(getattr(owner, "_onboardingLanguage", "en") or "en").startswith("zh")
        dwi_stepper = qt.QWidget()
        dwi_stepper.setObjectName("resultsDwiStepper")
        dwi_stepper_layout = qt.QHBoxLayout(dwi_stepper)
        dwi_stepper_layout.setContentsMargins(0, 2, 0, 0)
        dwi_stepper_layout.setSpacing(6)
        dwi_label = qt.QLabel("体积" if zh else "Volume")
        dwi_label.setObjectName("resultsDwiStepperLabel")
        dwi_slider = qt.QSlider(qt.Qt.Horizontal)
        dwi_slider.setObjectName("resultsDwiComponentSlider")
        dwi_slider.setRange(1, 1)
        if hasattr(dwi_slider, "setSingleStep"):
            dwi_slider.setSingleStep(1)
        if hasattr(dwi_slider, "setPageStep"):
            dwi_slider.setPageStep(1)
        if hasattr(dwi_slider, "setTracking"):
            dwi_slider.setTracking(True)
        dwi_slider.setMinimumWidth(60)
        dwi_slider.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed)
        pointing = getattr(qt.Qt, "PointingHandCursor", None)
        if pointing is not None and hasattr(dwi_slider, "setCursor"):
            dwi_slider.setCursor(pointing)
        dwi_slider.setEnabled(False)
        dwi_spin = qt.QSpinBox()
        dwi_spin.setObjectName("resultsDwiComponentSpinBox")
        dwi_spin.setRange(1, 1)
        dwi_spin.setSizePolicy(qt.QSizePolicy.Maximum, qt.QSizePolicy.Fixed)
        dwi_spin.setEnabled(False)
        dwi_info = qt.QLabel()
        dwi_info.setObjectName("resultsDwiStepperInfo")
        if hasattr(dwi_info, "setTextFormat") and hasattr(qt.Qt, "PlainText"):
            dwi_info.setTextFormat(qt.Qt.PlainText)
        dwi_info.setMinimumWidth(0)
        dwi_info.setSizePolicy(qt.QSizePolicy.Maximum, qt.QSizePolicy.Preferred)
        dwi_stepper_layout.addWidget(dwi_label)
        dwi_stepper_layout.addWidget(dwi_slider, 1)
        dwi_stepper_layout.addWidget(dwi_spin)
        dwi_stepper_layout.addWidget(dwi_info)
        dwi_stepper.hide()
        results_box_layout.addWidget(dwi_stepper)

        def _on_results_dwi_change(value) -> None:
            handler = getattr(owner, "_on_results_dwi_component_changed", None)
            if callable(handler):
                handler(value)

        dwi_slider.connect("valueChanged(int)", _on_results_dwi_change)
        dwi_spin.connect("valueChanged(int)", _on_results_dwi_change)
        owner._resultsDwiWidgets = {
            "stepper": dwi_stepper,
            "stepper_label": dwi_label,
            "slider": dwi_slider,
            "spin": dwi_spin,
            "info": dwi_info,
        }

    def _on_tree_item_clicked(item, column) -> None:
        if column == 1 and item.parent() is None:  # Batch 2 add-on: the Open folder button of a subject row
            _open_item_folder(owner, item)
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
    group = f'<g fill="none" stroke="{color}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">'
    if glyph == "list":  # the Results list menu: a list with an x (clear, show hidden)
        body = (
            f"{group}"
            '<path d="M2.5 4h11"/><path d="M2.5 8h11"/><path d="M2.5 12h5"/>'
            '<path d="M10.5 10.5l3 3"/><path d="M13.5 10.5l-3 3"/></g>'
        )
    elif glyph == "folder":  # Open folder, on the Subject ID rows
        # A mid-tone stroke: the selected row has a light highlight also in the dark theme, so the light stroke
        # of the header buttons would vanish there.
        row_group = group.replace(color, "#8A99AB" if dark else "#5F6E80")
        body = (
            f"{row_group}"
            '<path d="M2 4.5a1 1 0 0 1 1-1h3l1.5 1.5H13a1 1 0 0 1 1 1V12a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1z"/></g>'
        )
    elif glyph == "hideall":  # Hide everything shown: an eye with a slash over stacked layers
        body = (
            f"{group}"
            '<path d="M2 5s2.2-3 6-3 6 3 6 3-2.2 3-6 3-6-3-6-3z"/><circle cx="8" cy="5" r="1.2"/>'
            '<path d="M2 10.6l6 2.4 6-2.4"/><path d="M2 13.4l6 2.4 6-2.4"/>'
            '<path d="M3 1.5l10 11"/></g>'
        )
    else:  # "popout"
        body = (
            f"{group}"
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
    """Add the header row (hide-all, menu, pop-out), the pop-out placeholder row and the body container.

    The body container holds the empty-state row, the tree and the report bar. Pop-out acts
    on this one widget; the tree code never shows or hides it. Returns the body's layout.
    """

    header = qt.QWidget()
    header.setObjectName("resultsHeaderRow")
    header.setFixedHeight(RESULTS_HEADER_BUTTON_SIZE[1])  # FD7: the row is no higher than its buttons
    header_layout = qt.QHBoxLayout(header)
    header_layout.setContentsMargins(0, 0, 0, 0)
    header_layout.setSpacing(2)
    header_layout.addStretch(1)
    # Batch 2 add-on: "Hide everything shown", an icon-only button; the text is the tooltip and the accessible name.
    hide_all = qt.QToolButton()
    hide_all.setObjectName("resultsHideAllButton")
    _style_header_button(qt, hide_all)
    hide_all.setToolButtonStyle(qt.Qt.ToolButtonIconOnly)
    hide_all_text = _text(owner, "results_hide_all", HIDE_ALL_TEXT)
    hide_all.setToolTip(hide_all_text)
    hide_all.setAccessibleName(hide_all_text)
    _bind(owner, hide_all, "tooltip", "results_hide_all")
    hide_all.clicked.connect(lambda *_args: hide_everything_shown(owner))
    header_layout.addWidget(hide_all)
    owner._resultsHideAllButton = hide_all
    # Batch 2: the Results list menu (Clear, Show hidden). The menu is filled on each open, in the current language.
    menu_button = qt.QToolButton()
    menu_button.setObjectName("resultsMenuButton")
    _style_header_button(qt, menu_button)
    menu_button.setToolButtonStyle(qt.Qt.ToolButtonIconOnly)
    menu_button.setPopupMode(qt.QToolButton.InstantPopup)
    menu_button.setStyleSheet("QToolButton::menu-indicator { image: none; width: 0px; }")
    menu_button.setToolTip(_text(owner, "results_menu", "Results list: clear it or show hidden results"))
    _bind(owner, menu_button, "tooltip", "results_menu")
    results_menu = qt.QMenu(menu_button)
    _show_menu_tooltips(results_menu)
    results_menu.connect("aboutToShow()", lambda: _fill_results_menu(owner, results_menu))
    menu_button.setMenu(results_menu)
    header_layout.addWidget(menu_button)
    pop_out = qt.QToolButton()
    pop_out.setObjectName("resultsPopOutButton")
    _style_header_button(qt, pop_out)
    pop_out.setToolButtonStyle(qt.Qt.ToolButtonIconOnly)
    pop_out.setToolTip(_text(owner, "results_pop_out", "Open results in a separate window"))
    _bind(owner, pop_out, "tooltip", "results_pop_out")
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
    owner._resultsMenuButton = menu_button
    owner._resultsMenu = results_menu
    owner._resultsCollapseButton = None
    owner._resultsPopOutButton = pop_out
    owner._resultsPoppedRow = popped
    owner._resultsBody = body
    owner._resultsCollapsed = False
    owner._resultsPoppedOut = False
    owner._resultsDialog = None
    owner._resultsDialogLayout = None
    owner._resultsBodyIndex = None
    owner._resultsWindowSize = RESULTS_WINDOW_SIZE
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

    try:
        _refresh_output_visibility(owner, dark=dark)  # B2: the row eyes follow the theme
    except Exception:
        pass
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
        menu_button = getattr(owner, "_resultsMenuButton", None)
        menu_icon = _results_icon(qt, "list", dark=resolved_dark) if menu_button is not None else None
        if menu_icon is not None:
            menu_button.setIcon(menu_icon)
        hide_all_button = getattr(owner, "_resultsHideAllButton", None)
        hide_all_icon = _results_icon(qt, "hideall", dark=resolved_dark) if hide_all_button is not None else None
        if hide_all_icon is not None:
            hide_all_button.setIcon(hide_all_icon)
    except Exception:
        pass
    _refresh_folder_icons(owner, resolved_dark)
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


def _is_subgroup_row(qt_mod: Any, item: Any) -> bool:
    """A subgroup row carries the ``SUBGROUP_ROW_KIND`` tag. The text of its key does not decide the row kind: an
    unknown family (a manifest task such as ``anatomy::surfaces.output``) may hold the separator in its key."""

    return item.data(0, _user_role(qt_mod, ROLE_KIND)) == SUBGROUP_ROW_KIND


def _subgroup_key(family: str, subgroup: str) -> str:
    return f"{family}{SUBGROUP_KEY_SEPARATOR}{subgroup}"


def _result_subgroup(family: str, product: str) -> Optional[str]:
    """The subgroup of a product inside its family; None for a flat family (preprocessing, qc, unknown)."""

    if family not in RESULT_SUBGROUP_ORDER:
        return None
    # A multi-tracking run names a task "<product>@<tracking backend>" (shared/multi_tracking.py); the files of
    # every tracking source go to the algorithm's one subgroup (no third level).
    product = product.split("@", 1)[0]
    head, _, rest = product.partition(".")
    if family == "anatomy":
        if head == "postprocess" or product == "surface.ddsurfer_postprocess":
            return "postprocess"
        if head in ("segmentation", "parcellation"):
            return head
        if head == "surface":
            return "surfaces"
    rest = rest or product
    return _SUBGROUP_ALIASES.get((family, rest), rest)


def _family_caption(family: Any, zh: bool) -> str:
    return RESULT_FAMILY_TITLES.get(family, (family, family))[int(zh)]


def _subgroup_caption(key: str, zh: bool) -> str:
    family, _, subgroup = key.partition(SUBGROUP_KEY_SEPARATOR)
    if family in RESULT_SOURCE_FAMILIES:
        algorithm, _, source = subgroup.partition("@")
        if source in RESULT_SOURCE_CAPTIONS:
            base = RESULT_SUBGROUP_CAPTIONS.get((family, algorithm), (algorithm, algorithm))[int(zh)]
            return f"{base}{RESULT_SOURCE_SEPARATOR}{RESULT_SOURCE_CAPTIONS[source][int(zh)]}"
    return RESULT_SUBGROUP_CAPTIONS.get((family, subgroup), (subgroup, subgroup))[int(zh)]


def _path_parts(value: Any) -> list:
    """The folder names of a path text, for "/" and "\\" separators."""

    return [part for part in re.split(r"[\\/]+", str(value or "").strip()) if part]


def _published_source(parts: list, top: str, source_index: int) -> Optional[str]:
    """Review R8: the source folder of a path in the published layout only: the third name from the end is ``top``
    (``04_bundles/<product>/<source>``: source ``parts[-1]``; ``05_connectome/<source>/<file>``: source
    ``parts[-2]``), and the path is not in the stage cache. A legacy or stage folder is named after the RUN (the CLI
    default run name is ``csd_act``), so it never counts."""

    if len(parts) < 3 or parts[-3] != top or parts[source_index] not in RESULT_SOURCE_ORDER:
        return None
    head = tuple(parts[:-3])
    size = len(RESULT_STAGE_CACHE_PARTS)
    if any(head[i:i + size] == RESULT_STAGE_CACHE_PARTS for i in range(len(head) - size + 1)):
        return None
    return parts[source_index]


def _task_source(family: str, task: dict, manifest: Optional[dict] = None) -> Optional[str]:
    """Results sources rule 2: the tractography source of a bundles or connectome task, or None when unknown.

    a. The published folder: bundles ``outputs["output_dir"]`` (``04_bundles/<product>/<source>``), connectome the
       parent folder of ``outputs["matrix_path"]`` / ``outputs["matrix"]`` (``05_connectome/<source>/``); only a
       known source in the published layout counts (review R8). It comes first: a ukf variant that ran on the user's
       file publishes to ``supplied``.
    b. Bundles only: the ``@<backend>`` suffix of the task name (multi-tracking), else the ``tracking_backend``
       field; only a tracking backend counts, and only when the record shows that the product did not read a
       user-supplied tractogram (review R6): WMQL always (no user-file input), and every bundle algorithm of a run
       whose ``manifest`` records ``bundle_tractography_backend`` (the orchestrator refuses that setting together
       with a WMA or TractCloud input file). Otherwise the manifest cannot tell and the source is unknown. Not for the
       connectome: an old flat connectome file can hold another variant's matrix.
    """

    if family not in RESULT_SOURCE_FAMILIES:
        return None
    outputs = task.get("outputs") if isinstance(task.get("outputs"), dict) else {}
    if family == "bundles":
        found = _published_source(_path_parts(outputs.get("output_dir")), RESULT_BUNDLES_FOLDER, -1)
    else:
        found = _published_source(_path_parts(outputs.get("matrix_path") or outputs.get("matrix")),
                                  RESULT_CONNECTOME_FOLDER, -2)
    if found is not None:
        return found
    if family == "bundles":
        name = str(task.get("name") or "")
        product = str(task.get("product") or name.partition("@")[0])
        no_supplied_file = bool(str((manifest or {}).get("bundle_tractography_backend") or "").strip())
        if product.partition(".")[2] not in RESULT_OWN_TRACTOGRAM_BUNDLES and not no_supplied_file:
            return None
        suffix = name.partition("@")[2]
        if suffix in RESULT_TRACKING_SOURCES:
            return suffix
        backend = str(task.get("tracking_backend") or "")
        if backend in RESULT_TRACKING_SOURCES:
            return backend
    return None


def _subgroup_ranks(family: str, subgroups) -> dict:
    """Sort keys for the subgroups of a family: the algorithm in table order (an unnamed one after the named ones, in
    the order given), then for a sourced family the source: unknown, ukf, csd_act, dti_det, supplied."""

    names = RESULT_SUBGROUP_ORDER.get(family, ())
    unnamed: dict = {}
    ranks = {}
    for subgroup in subgroups:
        algorithm, _, source = str(subgroup).partition("@") if family in RESULT_SOURCE_FAMILIES else (subgroup, "", "")
        if algorithm in names:
            first = (names.index(algorithm), 0)
        else:
            first = (len(names), unnamed.setdefault(algorithm, len(unnamed)))
        source_rank = 0 if not source else (
            1 + RESULT_SOURCE_ORDER.index(source) if source in RESULT_SOURCE_ORDER else len(RESULT_SOURCE_ORDER) + 1)
        ranks[subgroup] = (*first, source_rank)
    return ranks


def _file_count(item: Any, role_path: Any) -> int:
    """The file rows under a family row, through its subgroups."""

    return sum(1 if child.data(0, role_path) else _file_count(child, role_path) for child in _child_items(item))


def _item_key(qt_mod: Any, item: Any) -> Optional[tuple]:
    """A key that survives a rebuild: subject, (subject, family), (subject, family, subgroup) or (subject, path).

    Results subgroups: the row kind comes from the row's data, not from its depth (a file row sits under a family
    or under a subgroup); the subject is always the top item.
    """

    role_key, role_path = _user_role(qt_mod, ROLE_KEY), _user_role(qt_mod, ROLE_PATH)
    parent = item.parent()
    key = item.data(0, role_key)
    if parent is None:
        return ("subject", key) if key else None
    subject = _top_item(item).data(0, role_key)
    path = item.data(0, role_path)
    if path:
        return ("file", subject, str(path))
    if not key:
        return None
    if _is_subgroup_row(qt_mod, item):
        return ("subgroup", subject, parent.data(0, role_key), key)
    return ("family", subject, key) if parent.parent() is None else None


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
    """What the user expanded or collapsed in this session: ``subjects[key]``, ``families[(key, family)]`` and
    ``subgroups[(key, family, subgroup key)]``."""

    state = getattr(owner, "_resultsExpansion", None)
    if not isinstance(state, dict):
        state = {"subjects": {}, "families": {}}
        owner._resultsExpansion = state
    for name in ("subjects", "families", "subgroups"):
        state.setdefault(name, {})
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
    elif key[0] == "subgroup":
        state["subgroups"][(key[1], key[2], key[3])] = bool(expanded)


def _apply_family_expansion(owner: Any, subject_item: Any, qt_mod: Any) -> None:
    """The remembered state of the families and subgroups of a subject; both start collapsed."""

    role_key = _user_role(qt_mod, ROLE_KEY)
    subject_key = subject_item.data(0, role_key)
    state = _expansion_state(owner)
    families, subgroups = state["families"], state["subgroups"]
    for family in _child_items(subject_item):
        family_key = family.data(0, role_key)
        _set_expanded(family, families.get((subject_key, family_key), False))
        for row in _child_items(family):
            row_key = row.data(0, role_key)
            if _is_subgroup_row(qt_mod, row):
                _set_expanded(row, subgroups.get((subject_key, family_key, row_key), False))


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


def _archive_context(owner: Any, records=None) -> tuple[dict, dict]:
    """What the archived manifests of a rebuild need, computed once: the stored hide list (the raw stamps, also for a
    running job) and the current records of each metadata folder (shown and hidden, for the owner of an archive)."""

    shown = tuple(records) if records is not None else tuple(getattr(owner, "_runTreeRecords", ()) or ())
    folders: dict = {}
    for record in (*shown, *(getattr(owner, "_resultsHiddenRecords", ()) or ())):
        if not record.is_batch:
            folders.setdefault(str(Path(record.manifest_path).parent), []).append(record)
    return load_hidden_runs(), folders


def _subject_manifests(owner: Any, records, indices, manifests: Optional[dict], context) -> list:
    """The current and the archived manifests of a subject's runs as one newest-first sequence.

    Each entry is ``(manifest, manifest path, owning run index, metadata folder of the owning record, archived)``.
    An archive is not a run: it carries the index of its owning record. Archives older than the stored hide stamp of
    their owning record stay hidden (Results hide rules); a broken archive is skipped.

    Also kept on ``owner`` for the rows of this rebuild (review R1, R2): ``_resultsRunManifests`` (run index -> the
    current manifest and every archive of the record; the scene nodes of a row are the nodes that any of them owns;
    the paths are resolved only when a row looks for its nodes) and ``_resultsHiddenArchives`` (record key ->
    (record, number of archives that the hide stamp keeps hidden); "Show hidden results" counts and restores them).
    """

    hidden, folders = context if context is not None else _archive_context(owner, records)
    run_manifests = _owner_dict(owner, "_resultsRunManifests")
    hidden_archives = _owner_dict(owner, "_resultsHiddenArchives")
    entries = []
    for position, index in enumerate(indices):
        record = records[index]
        if record.is_batch:
            continue
        metadata = Path(record.manifest_path).parent
        manifest = manifests[index] if manifests and index in manifests else _read_manifest(record)
        if isinstance(manifest, dict) and manifest:
            entries.append((_record_stamp(record), 0, position, manifest, record.manifest_path, index, metadata))
        key = run_record_key(record)
        hide_stamp = (float("-inf") if hidden[key] is None else float(hidden[key])) if key in hidden else None
        archives = archived_run_manifests(record, folders.get(str(metadata), ()))
        run_manifests[index] = (record.manifest_path, *(archive.path for archive in archives))  # resolved on use
        _owner_dict(owner, "_resultsRunManifestKeys").pop(index, None)
        kept_hidden = 0
        for archive in archives:
            parsed = read_archived_manifest(archive)
            if parsed is None:
                continue
            payload, when = parsed
            stamp = when.timestamp() if when is not None else float("-inf")
            if hide_stamp is not None and stamp <= hide_stamp:
                kept_hidden += 1
                continue
            entries.append((stamp, 1, position, payload, archive.path, index, metadata))
        if kept_hidden:
            hidden_archives[key] = (record, kept_hidden)
        else:
            hidden_archives.pop(key, None)
    entries.sort(key=lambda entry: (-entry[0], entry[1], entry[2]))
    return [(*entry[3:], bool(entry[1])) for entry in entries]


def _owner_dict(owner: Any, name: str) -> dict:
    value = getattr(owner, name, None)
    if not isinstance(value, dict):
        value = {}
        setattr(owner, name, value)
    return value


_BROKEN_ARCHIVES_LOGGED: set = set()


def _fill_subject_children(owner: Any, subject_item: Any, records, indices, manifests: Optional[dict] = None,
                           context=None) -> None:
    """Add the families and files of every run of a subject, newest manifest first, each family and file once.

    The archived manifests of the runs (``history/``) add the files that a re-run of the same run name moved out of
    the current manifest. Review R4: an archive with a broken task structure is skipped (one log line), and the
    tasks it put in the memo are taken out again, so an older manifest with the same task still gives its rows.
    """

    memo: dict = {"tasks": set(), "qc_dirs": {}}
    for manifest, manifest_path, index, metadata, archived in _subject_manifests(
            owner, records, indices, manifests, context):
        saved = (set(memo["tasks"]), dict(memo["qc_dirs"])) if archived else None
        try:
            _populate_manifest_output_items(
                owner, subject_item, manifest, manifest_path=manifest_path, run_index=index,
                metadata_dir=metadata, memo=memo,
            )
        except Exception as exc:
            if saved is None:
                raise
            memo["tasks"], memo["qc_dirs"] = saved
            if str(manifest_path) not in _BROKEN_ARCHIVES_LOGGED:
                _BROKEN_ARCHIVES_LOGGED.add(str(manifest_path))
                logging.getLogger(__name__).warning(
                    "Results: skipped the archived manifest %s (%s: %s).", manifest_path, type(exc).__name__, exc)
    _order_families(subject_item)


def _order_families(subject_item: Any) -> None:
    """Keep the families and their subgroups in their usual order, also when an older run adds one that a newer
    run lacks. Call it before the expansion is applied: a taken and re-added row loses its expansion in Qt."""

    import qt as qt_mod

    role_key = _user_role(qt_mod, ROLE_KEY)
    rank = {family: position for position, family in enumerate(RESULT_FAMILY_ORDER)}
    _reorder_children(subject_item, lambda child: rank.get(child.data(0, role_key), len(rank)))
    for family in _child_items(subject_item):
        family_key = family.data(0, role_key)
        if not RESULT_SUBGROUP_ORDER.get(family_key):
            continue
        prefix = _subgroup_key(family_key, "")
        keys = [str(child.data(0, role_key) or "") for child in _child_items(family)]
        ranks = _subgroup_ranks(family_key, [key[len(prefix):] for key in keys if key.startswith(prefix)])
        last = (float("inf"),)
        _reorder_children(family, lambda child: ranks.get(str(child.data(0, role_key) or "")[len(prefix):], last)
                          if str(child.data(0, role_key) or "").startswith(prefix) else last)


def _reorder_children(item: Any, rank) -> None:
    children = _child_items(item)
    ordered = sorted(children, key=rank)  # stable: rows of one rank keep their order (newest run first)
    if ordered != children:
        item.takeChildren()
        for child in ordered:
            item.addChild(child)


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
    _set_folder_button(owner, qt_mod, item, _group_folder(owner, records, group))  # Batch 2 add-on: Open folder
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
    # Filled again by the subjects below (review R1, R2); the empty-state count reads them before the fill.
    owner._resultsRunManifests = {}
    owner._resultsRunManifestKeys = {}
    owner._resultsHiddenArchives = {}
    records = _with_ended_runs(owner, records)
    running = running_subject_id(owner)
    has_content = bool(records) or bool(running)
    columns = owner._resultsColumns
    columns.setVisible(has_content)
    owner._resultsEmptyLabel.setVisible(not has_content)
    empty_row = getattr(owner, "_resultsEmptyRow", None)
    if empty_row is not None:
        empty_row.setVisible(not has_content)
    _update_empty_state(owner, has_content)
    _sync_hide_all_text(owner)
    report_bar = getattr(owner, "_resultsReportBar", None)
    if report_bar is not None:
        report_bar.setVisible(False)
    snapshot = _view_snapshot(qt, tree)
    state = _expansion_state(owner)
    run_items: dict = {}
    group_runs: dict = {}
    selected_item = None
    context = _archive_context(owner, records)
    owner._resultsApplyingState = True
    try:
        tree.clear()
        for position, group in enumerate(_subject_groups(owner, records, running)):
            item = _build_subject_item(owner, qt, tree, records, group)
            if not group["batch"]:
                _fill_subject_children(owner, item, records, group["indices"], context=context)
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


def _scene_nodes_by_run(slicer) -> dict:
    """``{Run attribute: {source path: [nodes]}}`` of every result node, from ONE scene scan (review R1)."""

    from shared.result_scene import result_nodes

    grouped: dict = {}
    for node in result_nodes(slicer):
        run = node.GetAttribute("SlicerDDSurfer.Run")
        if run:
            grouped.setdefault(run, {}).setdefault(node.GetAttribute("SlicerDDSurfer.SourcePath"), []).append(node)
    return grouped


def _row_manifests(owner: Any, item: Any, qt_mod: Any, manifest_path: Any, resolved: Optional[dict] = None):
    """The manifests whose scene nodes belong to a file row (review R1): every manifest of the row's owning record
    (its current manifest and its archives, raw and resolved paths), so a node that was loaded before a re-run moved
    the manifest to ``history/`` still belongs to the row. A row without a known owning record keeps its own
    manifest. Another record's manifests never count. ``resolved`` caches the resolve of a manifest path."""

    raw = str(manifest_path)
    cache = resolved if resolved is not None else {}
    run_idx = item.data(0, _user_role(qt_mod, ROLE_RUN)) if item is not None else None
    if isinstance(run_idx, int) and not isinstance(run_idx, bool):
        paths = (getattr(owner, "_resultsRunManifests", None) or {}).get(run_idx)
        if paths and raw in {str(path) for path in paths}:
            keys_by_run = _owner_dict(owner, "_resultsRunManifestKeys")
            keys = keys_by_run.get(run_idx)
            if keys is None:
                for path in paths:
                    if str(path) not in cache:
                        cache[str(path)] = os.path.realpath(path)
                keys = keys_by_run[run_idx] = frozenset(
                    text for path in paths for text in (str(path), cache[str(path)]))
            return keys
    if raw not in cache:
        cache[raw] = os.path.realpath(raw)
    return frozenset((raw, cache[raw]))


def _nodes_of(grouped: dict, manifests, path: Any) -> list:
    """The nodes of ``path`` that any of ``manifests`` owns, each once."""

    found: list = []
    seen: set = set()
    for key, by_path in grouped.items():  # the few Run values of the scene, in scene order
        if key not in manifests:
            continue
        for node in by_path.get(str(path), ()):
            if id(node) not in seen:
                seen.add(id(node))
                found.append(node)
    return found


def _output_nodes(owner, slicer, item, manifest_path, path):
    import qt

    return _nodes_of(_scene_nodes_by_run(slicer), _row_manifests(owner, item, qt, manifest_path), path)


def _carrier_overlay_nodes(grouped: dict, manifests, surface_path: Any) -> list:
    """Morphometry overlay model nodes in ``manifests`` whose ``CarrierPath`` matches ``surface_path``."""

    from shared.surface_morphometry_display import _file_key

    target = _file_key(surface_path)
    if not target or not grouped:
        return []
    found: list = []
    seen: set = set()
    for key, by_path in grouped.items():
        if key not in manifests:
            continue
        for nodes in by_path.values():
            for node in nodes:
                if id(node) in seen:
                    continue
                if _file_key(node.GetAttribute("SlicerDDSurfer.CarrierPath")) == target:
                    seen.add(id(node))
                    found.append(node)
    return found


def _find_surface_row_manifest(owner: Any, qt_mod: Any, item: Any, surface_path: Path, fallback: Path) -> Path:
    """Return the manifest path owning the cortical surface row for ``surface_path`` in the same subject subtree."""

    from shared.surface_morphometry_display import _file_key

    target = _file_key(surface_path)
    root = _top_item(item) if item is not None else None
    if not target or root is None:
        return fallback
    stack = [root]
    while stack:
        current = stack.pop()
        raw = current.data(0, _user_role(qt_mod, ROLE_PATH))
        if isinstance(raw, (str, Path)) and raw and _file_key(raw) == target:
            found = _item_manifest_path(owner, current, qt_mod)
            if found is not None:
                return found
        stack.extend(_child_items(current))
    return fallback


def _slice_connections(slicer, node):
    layer = "Label" if node.IsA("vtkMRMLLabelMapVolumeNode") else "Background"
    return [(composite, layer) for composite in slicer.util.getNodesByClass("vtkMRMLSliceCompositeNode")
            if getattr(composite, f"Get{layer}VolumeID")() == node.GetID()]


def _output_visible(slicer, nodes):
    from shared.fiber_display import is_fiber_bundle_node, is_fiber_node_visible

    for node in nodes:
        if node.IsA("vtkMRMLVolumeNode"):
            if _slice_connections(slicer, node):
                return True
        elif is_fiber_bundle_node(node):
            if is_fiber_node_visible(node):
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
        from shared.fiber_display import (
            DISPLAY_KIND,
            KIND_TRACTOGRAPHY,
            apply_fiber_display,
            is_fiber_bundle_node,
            set_fiber_node_visible,
        )
        from shared.result_scene import source_stamp
        from shared.surface_morphometry_display import (
            PRODUCT as MORPHOMETRY_PRODUCT,
            _file_key,
            carrier_surface_of_node,
            desired_carrier_surfaces,
            morphometry_row,
            surface_from_obj_path,
        )

        item_kind = item.data(0, _user_role(qt, ROLE_KIND))
        row_manifests = _row_manifests(owner, item, qt, manifest_path)
        nodes = _output_nodes(owner, slicer, item, manifest_path, path)
        is_surface_obj = surface_from_obj_path(path) is not None
        surface_overlays = (
            _carrier_overlay_nodes(_scene_nodes_by_run(slicer), row_manifests, path)
            if is_surface_obj
            else []
        )
        visible = _output_visible(slicer, nodes) or (bool(surface_overlays) and _output_visible(slicer, surface_overlays))
        if is_surface_obj and visible:
            for node in nodes:
                node.SetDisplayVisibility(False)
            for overlay in surface_overlays:
                overlay.SetDisplayVisibility(False)
            _refresh_output_visibility(owner)
            return

        morph_info = morphometry_row(MORPHOMETRY_PRODUCT, path) if item_kind == MORPHOMETRY_ROW_KIND else None
        is_morph_overlay = morph_info is not None and morph_info[0] in {"measure", "annotation"}

        if not visible and nodes and path.is_file():
            stamp = source_stamp(path)
            if any(node.GetAttribute("SlicerDDSurfer.SourceStamp") != stamp for node in nodes):
                for node in nodes:
                    slicer.mrmlScene.RemoveNode(node)
                nodes = []

        if is_morph_overlay:
            if visible:
                reveal_carriers: list[Path] = []
                seen_carriers: set[str] = set()
                for node in nodes:
                    try:
                        node_visible = bool(node.GetDisplayVisibility())
                    except Exception:
                        node_visible = False
                    if node_visible:
                        raw_carrier = node.GetAttribute("SlicerDDSurfer.CarrierPath")
                        if surface_from_obj_path(raw_carrier) is not None:
                            carrier_path = Path(str(raw_carrier))
                            key = _file_key(carrier_path)
                            if key and key not in seen_carriers and carrier_path.is_file():
                                seen_carriers.add(key)
                                reveal_carriers.append(carrier_path)
                    node.SetDisplayVisibility(False)
                grouped_after = _scene_nodes_by_run(slicer) if reveal_carriers else {}
                for carrier_path in reveal_carriers:
                    plain_nodes = _nodes_of(grouped_after, row_manifests, carrier_path)
                    if not plain_nodes:
                        surface_manifest = _find_surface_row_manifest(owner, qt, item, carrier_path, manifest_path)
                        try:
                            owner.logic.loadKeyResultsFromManifest(surface_manifest, source_path=carrier_path)
                        except Exception:
                            pass
                        plain_nodes = _nodes_of(_scene_nodes_by_run(slicer), row_manifests, carrier_path)
                    for plain in plain_nodes:
                        plain.SetDisplayVisibility(True)
                _refresh_output_visibility(owner)
                return

            manifest_data: dict[str, Any] = {}
            try:
                manifest_data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            except Exception:
                manifest_data = {}
            desired = (
                desired_carrier_surfaces(slicer, manifest_data, morph_info[1], path)
                if manifest_data
                else ("white",)
            )
            if not nodes:
                result = owner.logic.loadKeyResultsFromManifest(manifest_path, source_path=path)
                nodes = _output_nodes(owner, slicer, item, manifest_path, path)
                if not nodes:
                    error = next((entry["reason"] for entry in result["skipped"] if entry.get("path") == str(path)),
                                 "This output is missing or no longer available in the run.")
                    raise RuntimeError(error)
            elif any(carrier_surface_of_node(node) is not None for node in nodes):
                loaded_surfaces = {carrier_surface_of_node(node) for node in nodes}
                missing_surfaces = tuple(surf for surf in desired if surf not in loaded_surfaces)
                if missing_surfaces:
                    result = owner.logic.loadKeyResultsFromManifest(
                        manifest_path,
                        source_path=path,
                        morphometry_surfaces=missing_surfaces,
                    )
                    nodes = _output_nodes(owner, slicer, item, manifest_path, path)
                    loaded_surfaces = {carrier_surface_of_node(node) for node in nodes}
                    if any(surf not in loaded_surfaces for surf in missing_surfaces):
                        error = next(
                            (entry["reason"] for entry in result["skipped"] if entry.get("path") == str(path)),
                            "This output is missing or no longer available in the run.",
                        )
                        raise RuntimeError(error)

            if any(carrier_surface_of_node(node) is not None for node in nodes):
                shown_nodes = []
                for node in nodes:
                    show_this = carrier_surface_of_node(node) in desired
                    node.SetDisplayVisibility(show_this)
                    if show_this:
                        shown_nodes.append(node)
            else:
                shown_nodes = list(nodes)
                for node in shown_nodes:
                    node.SetDisplayVisibility(True)

            _show_one_overlay_per_surface(slicer, shown_nodes)
            # Also hide plain surface / sibling overlays loaded from archived manifests of the same run family.
            shown_ids = {node.GetID() for node in shown_nodes}
            shown_carrier_keys = {
                _file_key(node.GetAttribute("SlicerDDSurfer.CarrierPath"))
                for node in shown_nodes
                if surface_from_obj_path(node.GetAttribute("SlicerDDSurfer.CarrierPath")) is not None
            }
            if shown_carrier_keys:
                for key, by_path in _scene_nodes_by_run(slicer).items():
                    if key not in row_manifests:
                        continue
                    for other_list in by_path.values():
                        for other in other_list:
                            if other.GetID() in shown_ids or not other.IsA("vtkMRMLModelNode"):
                                continue
                            other_keys = {
                                _file_key(other.GetAttribute("SlicerDDSurfer.SourcePath")),
                                _file_key(other.GetAttribute("SlicerDDSurfer.CarrierPath")),
                            }
                            if other_keys & shown_carrier_keys and other.GetDisplayVisibility():
                                other.SetDisplayVisibility(False)
            _refresh_output_visibility(owner)
            return

        if not nodes:
            result = owner.logic.loadKeyResultsFromManifest(manifest_path, source_path=path)
            nodes = _output_nodes(owner, slicer, item, manifest_path, path)
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
                if (
                    not visible
                    and item_kind == FIBER_ROW_KIND
                    and node.GetAttribute("SlicerDDSurfer.DisplayKind") != DISPLAY_KIND
                ):
                    apply_fiber_display(
                        slicer,
                        node,
                        node.GetAttribute("SlicerDDSurfer.StreamlineKind") or KIND_TRACTOGRAPHY,
                    )
                if is_fiber_bundle_node(node):
                    set_fiber_node_visible(node, not visible)
                else:
                    node.SetDisplayVisibility(not visible)
        if not visible:
            _show_one_overlay_per_surface(slicer, nodes)
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


def _owning_manifest_path(owner: Any, item: Any, qt_mod: Any, manifest_path: Any) -> Any:
    """The manifest of the current record that owns a file row: the row's own manifest, or for a row of an archived
    manifest (``history/``) the record that ROLE_RUN names."""

    records = getattr(owner, "_runHistoryRecords", ()) or ()
    if manifest_path is None or any(_same_path(record.manifest_path, manifest_path) for record in records):
        return manifest_path
    run_idx = item.data(0, _user_role(qt_mod, ROLE_RUN))
    if isinstance(run_idx, int) and not isinstance(run_idx, bool) and 0 <= run_idx < len(records):
        return records[run_idx].manifest_path
    return manifest_path


def _output_menu(owner, position):
    import qt
    import slicer

    tree = owner._runOutputFiles
    item = tree.itemAt(position)
    if item is None:
        return
    raw_path = item.data(0, _user_role(qt, 0))
    manifest_path = _item_manifest_path(owner, item, qt)
    is_file = isinstance(raw_path, (str, Path)) and bool(raw_path) and manifest_path is not None
    nodes = _output_nodes(owner, slicer, item, manifest_path, Path(raw_path)) if is_file else []
    is_seg = is_file and item.data(0, _user_role(qt, 1)) == "segmentation"
    # Batch 2 + FD8: take the runs of the whole subject now; no tree item is read after the menu closes.
    remove_records, remove_reason = _removal_for_item(owner, qt, item)
    folder = _top_item(item).data(0, _user_role(qt, ROLE_FOLDER))  # also before exec_ (FD8)
    run_manifest = _owning_manifest_path(owner, item, qt, manifest_path) if is_seg else None  # before exec_ (FD8)
    # Review R1: Edit tissue reads the label maps of ONE manifest; a label map of this row that was loaded before a
    # re-run is owned by the manifest path it was loaded from (the current one), not by the archive of the row.
    edit_manifest = next((node.GetAttribute("SlicerDDSurfer.Run") for node in nodes
                          if node.IsA("vtkMRMLLabelMapVolumeNode") and str(
                              node.GetAttribute("SlicerDDSurfer.Product") or "").startswith("segmentation.")),
                         None) if is_seg else None
    edit_manifest = Path(edit_manifest) if edit_manifest else manifest_path
    menu = qt.QMenu(tree)
    zh = str(getattr(owner, "_onboardingLanguage", "en")).startswith("zh")
    edit_action = (
        menu.addAction("编辑组织分割副本" if zh else "Edit tissue copy in Segment Editor")
        if is_seg
        else None
    )
    unload = menu.addAction("从场景卸载" if zh else "Unload from scene") if nodes else None
    if edit_action is not None or unload is not None:
        menu.addSeparator()
    open_folder = menu.addAction(_text(owner, "results_open_folder", OPEN_FOLDER_TEXT))
    remove = menu.addAction(_text(owner, "results_remove", "Remove from Results list"))
    remove.enabled = bool(remove_records)
    remove.toolTip = remove_reason or _text(owner, "results_remove_tooltip", REMOVE_TOOLTIP)
    _show_menu_tooltips(menu)
    chosen = menu.exec_(tree.viewport().mapToGlobal(position))
    if edit_action is not None and chosen == edit_action:
        # FD3: _manage_result_scene acts on the selected run; in a merged subject the file may own an older one.
        # FD8: exec_ ran a nested event loop; a refresh may have deleted `item` and reordered the records,
        # so resolve the run from the manifest path taken before the menu opened, never from the item.
        # Archived runs: a row of an archived manifest selects its owning run, and the action reads the archive.
        if _select_manifest_run(owner, run_manifest):
            if _same_path(run_manifest, edit_manifest):
                owner._manage_result_scene("edit_tissue")
            else:
                selected = getattr(owner, "_latestResultManifestPath", None)
                owner._latestResultManifestPath = edit_manifest
                try:
                    owner._manage_result_scene("edit_tissue")
                finally:
                    owner._latestResultManifestPath = selected
    elif unload is not None and chosen == unload:
        for node in nodes:
            display_nodes = []
            get_count = getattr(node, "GetNumberOfDisplayNodes", None)
            get_nth = getattr(node, "GetNthDisplayNode", None)
            if callable(get_count) and callable(get_nth):
                try:
                    display_nodes = [get_nth(index) for index in range(int(get_count() or 0))]
                except Exception:
                    display_nodes = []
            slicer.mrmlScene.RemoveNode(node)
            for display in display_nodes:
                if display is not None:
                    try:
                        slicer.mrmlScene.RemoveNode(display)
                    except Exception:
                        pass
        _refresh_output_visibility(owner)
    elif chosen == open_folder:
        open_results_folder(owner, folder)
    elif chosen == remove and remove_records:
        _hide_records(owner, remove_records)


# ------------------------------------------------- Batch 2: remove from the Results list, Clear, show hidden
# Only the list changes: no file is deleted and no scene node is removed. A removed run shows again when its slot
# gets a newer run (re-run, resume, cancel), and "Show hidden results" brings back all hidden runs of the scan.

REMOVE_TOOLTIP = "Hide this subject's runs from the Results list. Files on disk and loaded scene nodes are kept."
REMOVE_RUNNING_TOOLTIP = "A job of this subject is running now; it cannot be removed from the list."
CLEAR_TOOLTIP = "Hide every run in the Results list. Files on disk and loaded scene nodes are kept; new runs appear."
REMOVED_STATUS = "Removed {count} run(s) from the list; files are kept."


def _show_menu_tooltips(menu: Any) -> None:
    try:
        menu.setToolTipsVisible(True)
    except Exception:
        pass


def load_hidden_runs() -> dict:
    """The stored hide list; a broken or missing value reads as empty."""

    try:
        import slicer

        value = slicer.app.userSettings().value(RESULTS_HIDDEN_SETTING, "")
    except Exception:
        return {}
    return parse_hidden_runs(value)


def save_hidden_runs(hidden: dict) -> None:
    try:
        import slicer

        slicer.app.userSettings().setValue(RESULTS_HIDDEN_SETTING, dump_hidden_runs(hidden))
    except Exception:
        pass


def _live_run_keys(owner: Any, records=()) -> set:
    """The slots of the job that runs now; they are never hidden.

    A path that a listed record owns gets the key of that record (a campaign file is a ``batch`` key, not a
    ``subject`` key). The ``subject`` key of the path stays in the set for a record that the scan lists under
    another file of the slot (a cancel companion).
    """

    if str(getattr(owner, "_runState", "") or "") != "running":
        return set()
    paths = _job_manifest_keys(owner)
    keys = {run_manifest_key(path) for path in paths}
    if paths:
        keys.update(run_record_key(record) for record in records if _manifest_key(record.manifest_path) in paths)
    return keys


def _run_started_stamp(owner: Any) -> Optional[float]:
    """The start of the current run (``_backgroundRunContext["started_at"]``, UTC ISO text) as a stamp."""

    text = str((getattr(owner, "_backgroundRunContext", None) or {}).get("started_at") or "").strip()
    if not text:
        return None
    try:
        started = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        return started.timestamp()
    except (ValueError, OverflowError, OSError):
        return None


def _active_run_keys(owner: Any, records) -> set:
    """The slots that a running job owns: its live manifests, and every listed run that is running or that is not
    older than the start of the run (a GUI batch or a campaign writes many subjects; only the last manifest path
    is known)."""

    if str(getattr(owner, "_runState", "") or "") != "running":
        return set()
    records = tuple(records)
    keys = _live_run_keys(owner, records)
    started = _run_started_stamp(owner)
    for record in records:
        if record.status == "running" or (started is not None and _record_stamp(record) >= started):
            keys.add(run_record_key(record))
    return keys


def _split_hidden_for_owner(owner: Any, records) -> tuple[tuple, tuple]:
    records = tuple(records)
    hidden = load_hidden_runs()
    active = _active_run_keys(owner, records)
    if active:
        hidden = {key: stamp for key, stamp in hidden.items() if key not in active}
    return split_hidden(records, hidden)


def _is_protected(record: RunRecord, running_subject: str, active_keys: set) -> bool:
    if run_record_key(record) in active_keys:
        return True
    return bool(running_subject) and not record.is_batch and record.subject_id == running_subject


def _clearable_records(owner: Any) -> tuple:
    """The listed runs that a Clear hides: all except the runs of the job that runs now."""

    records = tuple(getattr(owner, "_runHistoryRecords", ()) or ())
    running, active = running_subject_id(owner), _active_run_keys(owner, records)
    return tuple(record for record in records if not _is_protected(record, running, active))


def _removal_for_item(owner: Any, qt_mod: Any, item: Any) -> tuple[tuple, str]:
    """The runs of the item's top subject item, and the reason when they cannot be removed."""

    top = _top_item(item)
    running = running_subject_id(owner)
    reason = _text(owner, "results_remove_running", REMOVE_RUNNING_TOOLTIP)
    if running and top.data(0, _user_role(qt_mod, ROLE_KEY)) == running:
        return (), reason
    run_idx = top.data(0, _user_role(qt_mod, ROLE_RUN))
    if not isinstance(run_idx, int) or isinstance(run_idx, bool):
        return (), ""
    tree_records = getattr(owner, "_runTreeRecords", ()) or ()
    indices = (getattr(owner, "_resultsGroupRuns", {}) or {}).get(run_idx, ())
    records = tuple(tree_records[index] for index in indices if 0 <= index < len(tree_records))
    active = _active_run_keys(owner, tree_records)
    if any(run_record_key(record) in active for record in records):
        return (), reason
    return records, ""


def _hide_records(owner: Any, records) -> None:
    records = tuple(records)
    if not records:
        return
    hidden = load_hidden_runs()
    for record in records:
        key = run_record_key(record)
        hidden.pop(key, None)  # a new hide goes to the end
        hidden[key] = hidden_stamp(record)
    save_hidden_runs(hidden)
    owner._refresh_run_history()
    message = _text(owner, "results_removed_status", REMOVED_STATUS).format(count=len(records))
    try:
        import slicer

        slicer.util.showStatusMessage(message, 4000)
    except Exception:
        pass


def _unhide_records(records) -> None:
    hidden = load_hidden_runs()
    keys = {run_record_key(record) for record in records} & set(hidden)
    if keys:
        save_hidden_runs({key: stamp for key, stamp in hidden.items() if key not in keys})


def clear_results_list(owner: Any) -> None:
    """Hide every listed run except the running subject; new runs appear after it."""

    _hide_records(owner, _clearable_records(owner))


def show_hidden_results(owner: Any) -> None:
    """Show every hidden run that the last scan found again, and (review R2) the archives that a hide stamp keeps
    hidden although their record shows again: the stamps of those records are cleared too."""

    archived = tuple(record for record, _count in (getattr(owner, "_resultsHiddenArchives", None) or {}).values())
    _unhide_records((*(getattr(owner, "_resultsHiddenRecords", ()) or ()), *archived))
    owner._refresh_run_history()


def _hidden_count(owner: Any) -> int:
    """The number for "Show hidden results": the hidden runs, plus the hidden archives of shown runs (review R2)."""

    archives = getattr(owner, "_resultsHiddenArchives", None) or {}
    return len(getattr(owner, "_resultsHiddenRecords", ()) or ()) + sum(count for _record, count in archives.values())


def _show_hidden_text(owner: Any, count: int) -> str:
    text = _text(owner, "results_show_hidden", "Show hidden results")
    zh = str(getattr(owner, "_onboardingLanguage", "en")).startswith("zh")
    return f"{text}（{count}）" if zh else f"{text} ({count})"


def results_menu_entries(owner: Any) -> list:
    """(text, enabled, tooltip, action) of the Results list menu, in the current language and state."""

    count = _hidden_count(owner)
    return [
        (_text(owner, "results_clear", "Clear results list"), bool(_clearable_records(owner)),
         _text(owner, "results_clear_tooltip", CLEAR_TOOLTIP), lambda: clear_results_list(owner)),
        (_show_hidden_text(owner, count), count > 0, "", lambda: show_hidden_results(owner)),
    ]


def _fill_results_menu(owner: Any, menu: Any) -> None:
    menu.clear()
    for text, enabled, tooltip, action_slot in results_menu_entries(owner):
        action = menu.addAction(text)
        action.enabled = bool(enabled)
        if tooltip:
            action.toolTip = tooltip
        action.connect("triggered()", action_slot)


def _update_empty_state(owner: Any, has_content: bool) -> None:
    """The empty row: "no runs yet", or after a Clear the cleared text and the show-hidden button."""

    count = _hidden_count(owner)
    cleared = not has_content and count > 0
    label = getattr(owner, "_resultsEmptyLabel", None)
    if label is not None and not has_content:
        key, fallback = (("results_cleared_empty", "Results list cleared. New runs will appear here.") if cleared
                         else ("no_runs_found", "No runs in output folder yet"))
        label.setText(_text(owner, key, fallback))
        _rebind(owner, label, "text", key)
    button = getattr(owner, "_resultsShowHiddenButton", None)
    if button is not None:
        if cleared:
            button.setText(_show_hidden_text(owner, count))
        _set_widget_visible(button, cleared)


# --------------------------------------------- Batch 2 add-on: Open folder on a subject row, Hide everything shown
# Open folder is the icon of column 1 of a top-level row (the eye of a file row lives in the same column). The folder
# is stored on the row (ROLE_FOLDER) when the tree is built, so a click or a menu entry never reads a record again.
# Hide everything shown turns off displays only: it removes no node and touches no file.

HIDE_ALL_TEXT = "Hide everything shown"
OPEN_FOLDER_TEXT = "Open folder"
FOLDER_MISSING_TEXT = "The folder does not exist: {path}"
FOLDER_UNKNOWN_TEXT = "The folder of this subject is not known yet."
FOLDER_OPEN_FAILED_TEXT = "The system could not open the folder: {path}"
_COMPOSITE_LAYERS = ("Background", "Foreground", "Label")


def _status_message(text: str) -> None:
    try:
        import slicer

        slicer.util.showStatusMessage(text, 4000)
    except Exception:
        pass


def _folder_icon(owner: Any, qt_mod: Any, dark: Optional[bool] = None) -> Any:
    try:
        return _results_icon(qt_mod, "folder", dark=_resolve_dark_mode(owner, dark))
    except Exception:
        return None


def _group_folder(owner: Any, records, group: dict) -> str:
    """The folder of a subject row: that of its newest run; a running subject without a run uses its job roots."""

    if group["indices"]:
        return str(run_folder_path(records[group["indices"][0]]))
    context = getattr(owner, "_backgroundRunContext", None) or {}
    folder = subject_folder_from_roots(context.get("output_root"), context.get("subject_root"))
    return str(folder) if folder is not None else ""


def _set_folder_button(owner: Any, qt_mod: Any, item: Any, folder: str) -> None:
    item.setData(0, _user_role(qt_mod, ROLE_FOLDER), folder)
    icon = _folder_icon(owner, qt_mod)
    if icon is not None:
        item.setIcon(1, icon)
    _set_folder_button_text(owner, qt_mod, item)


def _set_folder_button_text(owner: Any, qt_mod: Any, item: Any) -> None:
    text = _text(owner, "results_open_folder", OPEN_FOLDER_TEXT)
    item.setToolTip(1, text)
    role = getattr(getattr(qt_mod, "Qt", None), "AccessibleTextRole", None)
    if role is not None:
        item.setData(1, role, text)


def _refresh_folder_icons(owner: Any, dark: bool) -> None:
    """A theme change: redraw the Open folder icon of every subject row."""

    tree = getattr(owner, "_runOutputFiles", None)
    if tree is None:
        return
    try:
        import qt

        icon = _folder_icon(owner, qt, dark)
        if icon is None:
            return
        role = _user_role(qt, ROLE_FOLDER)
        for item in _child_items(tree):
            if item.data(0, role) is not None:  # R5: only a subject row has the Open folder button
                item.setIcon(1, icon)
    except Exception:
        pass


def refresh_results_language(owner: Any) -> None:
    """A language switch: the texts that the generic bindings cannot set (R4). No rescan, no tree rebuild.

    The counted Show-hidden button, the accessible name of Hide all, and the tooltip and accessible name of the
    Open folder button of every subject row, and the captions of the family and subgroup rows.
    """

    button = getattr(owner, "_resultsShowHiddenButton", None)
    if button is not None:
        try:
            button.setText(_show_hidden_text(owner, _hidden_count(owner)))
        except Exception:
            pass
    _sync_hide_all_text(owner)
    dwi_widgets = getattr(owner, "_resultsDwiWidgets", None) or {}
    dwi_label = dwi_widgets.get("stepper_label")
    if dwi_label is not None:
        try:
            zh_lang = str(getattr(owner, "_onboardingLanguage", "en") or "en").startswith("zh")
            dwi_label.setText("体积" if zh_lang else "Volume")
        except Exception:
            pass
    tree = getattr(owner, "_runOutputFiles", None)
    if tree is None:
        return
    try:
        import qt

        role = _user_role(qt, ROLE_FOLDER)
        zh = str(getattr(owner, "_onboardingLanguage", "en")).startswith("zh")
        for item in _child_items(tree):
            if item.data(0, role) is not None:
                _set_folder_button_text(owner, qt, item)
                for family in _child_items(item):
                    _relabel_family(qt, family, zh)
            else:  # the flat fallback tree: the families are the top-level rows
                _relabel_family(qt, item, zh)
    except Exception:
        pass
    try:
        _refresh_output_visibility(owner)
    except Exception:
        pass


def _relabel_family(qt_mod: Any, family: Any, zh: bool) -> None:
    """Results subgroups: the caption of a family row and of its subgroup rows in the current language."""

    role_key, role_path = _user_role(qt_mod, ROLE_KEY), _user_role(qt_mod, ROLE_PATH)
    family_key = family.data(0, role_key)
    if not family_key or family.data(0, role_path):
        return
    family.setText(0, f"{_family_caption(family_key, zh)} ({_file_count(family, role_path)})")
    for row in _child_items(family):
        row_key = row.data(0, role_key)
        if _is_subgroup_row(qt_mod, row):
            row.setText(0, _subgroup_text(qt_mod, row, row_key, zh))
            _relabel_morphometry_rows(qt_mod, row, zh)


def open_results_folder(owner: Any, folder: Any) -> None:
    """Open ``folder`` in the system file browser. A missing folder gives a status message and nothing else."""

    text = str(folder or "").strip()
    if not text:
        _status_message(_text(owner, "results_folder_unknown", FOLDER_UNKNOWN_TEXT))
        return
    path = Path(text)
    if not path.is_dir():
        _status_message(_text(owner, "results_folder_missing", FOLDER_MISSING_TEXT).format(path=path))
        return
    try:
        import qt

        opened = qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(str(path)))
    except Exception:
        opened = False
    if opened is False:
        _status_message(_text(owner, "results_folder_open_failed", FOLDER_OPEN_FAILED_TEXT).format(path=path))


def _open_item_folder(owner: Any, item: Any) -> None:
    import qt

    open_results_folder(owner, item.data(0, _user_role(qt, ROLE_FOLDER)))


def _sync_hide_all_text(owner: Any) -> None:
    """The accessible name follows the language (the tooltip is bound; a binding cannot set an accessible name)."""

    button = getattr(owner, "_resultsHideAllButton", None)
    if button is not None:
        try:
            button.setAccessibleName(_text(owner, "results_hide_all", HIDE_ALL_TEXT))
        except Exception:
            pass


def _is_internal_node(node: Any) -> bool:
    """Slice-plane models ("Red Volume Slice" ...) and other nodes that Slicer hides from the user's lists."""

    if node.GetHideFromEditors():
        return True
    for index in range(node.GetNumberOfDisplayNodes()):
        if node.GetNthDisplayNode(index).IsA("vtkMRMLSliceDisplayNode"):
            return True
    return False


def _hide_node_displays(node: Any) -> None:
    for index in range(node.GetNumberOfDisplayNodes()):
        display = node.GetNthDisplayNode(index)
        if display is not None and display.GetVisibility():
            display.SetVisibility(0)


def hide_everything_shown(owner: Any) -> None:
    """Turn off every model, segmentation, slice layer and volume rendering that is shown. There is no "show all".

    The whole scene is acted on, not only the nodes the plugin loaded. Nodes stay in the scene and files stay on disk.
    """

    import slicer

    def nodes_of(class_name: str) -> list:
        try:
            return list(slicer.util.getNodesByClass(class_name))
        except Exception:
            return []

    def each(class_name: str, action) -> None:
        for node in nodes_of(class_name):
            try:
                action(node)
            except Exception:  # one deleted or odd node must not stop the others
                continue

    def hide_model(node: Any) -> None:
        if not _is_internal_node(node):
            _hide_node_displays(node)

    def clear_layers(composite: Any) -> None:
        for layer in _COMPOSITE_LAYERS:
            if getattr(composite, f"Get{layer}VolumeID")():
                getattr(composite, f"Set{layer}VolumeID")(None)

    each("vtkMRMLModelNode", hide_model)
    each("vtkMRMLFiberBundleNode", hide_model)
    each("vtkMRMLSegmentationNode", _hide_node_displays)  # every segmentation, as asked
    each("vtkMRMLSliceCompositeNode", clear_layers)
    each("vtkMRMLVolumeRenderingDisplayNode", lambda display: display.GetVisibility() and display.SetVisibility(0))
    try:
        _refresh_output_visibility(owner)
    except Exception:
        pass


FIBER_ROW_KIND = "fiber_bundle"
_EYE_OUTLINE = {(True, False): "#27824d", (True, True): "#6fcf97", (False, False): "#728195", (False, True): "#9aa7b8"}
# The "fiber eye": three short streamlines in direction colours (red, green, blue) replace the pupil.
_FIBER_EYE_COLORS = {False: {"curves": ("#d64545", "#2f9e44", "#3b6fd8")},
                     True: {"curves": ("#ff7b7b", "#63d67a", "#6ea8ff")}}
_FIBER_EYE_CURVES = ("M7.5 10c1.5-1.4 3 1.4 4.5 0s3-1.4 4.5 0",
                     "M7.5 12c1.5-1.4 3 1.4 4.5 0s3-1.4 4.5 0",
                     "M7.5 14c1.5-1.4 3 1.4 4.5 0s3-1.4 4.5 0")


def _eye_icon(qt, visible, *, kind=None, dark=False):
    """Row eye. Cache key (kind, visible, dark): the icon follows the theme. A fiber-bundle row gets the fiber eye."""
    key = (FIBER_ROW_KIND if kind == FIBER_ROW_KIND else "plain", bool(visible), bool(dark))
    if key not in _EYE_ICONS:
        color = _EYE_OUTLINE[(key[1], key[2])]
        slash = '' if visible else '<path d="M3 3l18 18"/>'
        if key[0] == FIBER_ROW_KIND:
            curves = _FIBER_EYE_COLORS[key[2]]["curves"] if visible else (color,) * 3
            pupil = "".join(f'<path d="{shape}" stroke="{stroke}" stroke-width="1.5"/>'
                            for shape, stroke in zip(_FIBER_EYE_CURVES, curves))
        else:
            pupil = '<circle cx="12" cy="12" r="2.5"/>'
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24">'
               f'<g fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round">'
               '<path d="M2 12s3.6-6 10-6 10 6 10 6-3.6 6-10 6-10-6-10-6z"/>'
               f'{pupil}{slash}</g></svg>')
        pixmap = qt.QPixmap()
        pixmap.loadFromData(qt.QByteArray(svg.encode("utf-8")))
        _EYE_ICONS[key] = qt.QIcon(pixmap)
    return _EYE_ICONS[key]


_EYE_ICONS = {}


def _refresh_output_visibility(owner, *, dark: Optional[bool] = None):
    import slicer

    tree = getattr(owner, "_runOutputFiles", None)
    if tree is None:
        return
    import qt

    fallback_manifest = getattr(owner, "_latestResultManifestPath", None)
    grouped: Optional[dict] = None  # review R1: one scene scan per pass, grouped by the Run attribute
    resolved_manifests: dict = {}  # one resolve (and stat) per manifest path, not one per file row

    def _row_nodes(item: Any, mpath: Optional[Path], raw_path: Any) -> list:
        nonlocal grouped
        if mpath is None:
            return []
        if grouped is None:
            grouped = _scene_nodes_by_run(slicer)
        if not grouped:
            return []
        manifests = _row_manifests(owner, item, qt, mpath, resolved_manifests)
        direct = _nodes_of(grouped, manifests, raw_path)
        from shared.surface_morphometry_display import surface_from_obj_path

        if surface_from_obj_path(raw_path) is not None:
            overlays = _carrier_overlay_nodes(grouped, manifests, raw_path)
            if overlays:
                seen = {id(n) for n in direct}
                return direct + [n for n in overlays if id(n) not in seen]
        return direct

    zh = str(getattr(owner, "_onboardingLanguage", "en")).startswith("zh")
    try:
        resolved_dark = _resolve_dark_mode(owner, dark)
    except Exception:
        resolved_dark = False
    for item in _iter_tree_items(tree):
        raw_path = item.data(0, _user_role(qt, 0))
        if not isinstance(raw_path, (str, Path)) or not raw_path:
            continue
        if item.text(1) == "↗":
            continue
        mpath = _item_manifest_path(owner, item, qt) or (Path(fallback_manifest) if fallback_manifest else None)
        visible = _output_visible(slicer, _row_nodes(item, mpath, raw_path))
        kind = item.data(0, _user_role(qt, ROLE_KIND))
        item.setIcon(1, _eye_icon(qt, visible, kind=kind, dark=resolved_dark))
        item.setText(1, "")
        if kind == FIBER_ROW_KIND:
            tip = ("隐藏纤维束" if zh else "Hide fiber bundle") if visible else ("显示为纤维束" if zh else "Show as fiber bundle")
        else:
            tip = ("隐藏" if zh else "Hide") if visible else ("显示" if zh else "Show")
        item.setToolTip(1, tip)
    refresh_stepper = getattr(owner, "_refresh_input_preview_stepper", None)
    refresh_results = getattr(owner, "_refresh_results_dwi_stepper", None)
    try:
        if callable(refresh_stepper):
            refresh_stepper()
        elif callable(refresh_results):
            refresh_results()
    except Exception:
        pass


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


def _task_memo_key(manifest: dict, task: dict) -> Optional[tuple]:
    """Every field of a task that ``result_candidates`` / ``bundle_outputs`` and the QC-row loop read, plus the roots
    of its manifest. Preprocessing tasks get no key: their outputs decide the brain-mask fallback of the manifest."""

    name = str(task.get("name") or "")
    if name.startswith("preprocessing."):
        return None
    try:
        outputs = json.dumps(task.get("outputs"), sort_keys=True, default=str)
    except (TypeError, ValueError):
        return None
    return (str(manifest.get("subject_root") or ""), str(manifest.get("output_root") or ""), name,
            str(task.get("status") or ""), str(task.get("reason") or ""), str(task.get("qc_path") or ""), outputs)


def _without_repeated_tasks(manifest: dict, memo: dict) -> dict:
    """The manifest without the tasks that a newer manifest of the same rebuild already read unchanged.

    Such a task gives exactly the rows it gave before (same files, same checks), and they are in the tree already, so
    reading it again only repeats the file checks (a re-run of one run name 60 times names the same files 60 times).
    """

    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return manifest
    seen = memo["tasks"]
    kept = []
    for task in tasks:
        key = _task_memo_key(manifest, task) if isinstance(task, dict) else None
        if key is not None and key in seen:
            continue
        if key is not None:
            seen.add(key)
        kept.append(task)
    return manifest if len(kept) == len(tasks) else dict(manifest, tasks=kept)


MORPHOMETRY_ROW_KIND = "surface_morphometry"  # ROLE_KIND of a surface post-processing row (plain eye)


def _morphometry_paths(candidates) -> set:
    """The rows of the surface post-processing files (shared.surface_morphometry_display)."""

    from shared.surface_morphometry_display import morphometry_row

    return {item["path"] for item in candidates if morphometry_row(item.get("product"), item["path"]) is not None}


def _morphometry_tooltip(path: Any, zh: bool) -> str:
    """What the row shows, then its file path."""

    from shared.surface_morphometry_display import row_tooltip

    text = row_tooltip(path, zh=zh)
    return f"{text}\n{path}" if text else str(path)


def _relabel_morphometry_rows(qt_mod: Any, row: Any, zh: bool) -> None:
    for leaf in _child_items(row):
        path = leaf.data(0, _user_role(qt_mod, ROLE_PATH))
        if path and leaf.data(0, _user_role(qt_mod, ROLE_KIND)) == MORPHOMETRY_ROW_KIND:
            leaf.setToolTip(0, _morphometry_tooltip(path, zh))


def _show_one_overlay_per_surface(slicer: Any, nodes) -> None:
    """Surface post-processing: a shown measure or annotation hides the other overlays of its white surface."""

    from shared.surface_morphometry_display import hide_same_surface_siblings

    hide_same_surface_siblings(slicer, nodes)


QC_MARK_TEXT = {"failed": ("QC not passed", "QC 未通过"), "unreadable": ("no QC record", "无 QC 记录")}


def _subgroup_text(qt_mod: Any, row: Any, key: str, zh: bool) -> str:
    """Caption (count), then the QC mark of a bundle task whose recorded QC is not passed. The key, the order, the
    expansion state and the count do not change with the mark."""

    text = f"{_subgroup_caption(key, zh)} ({_tree_count(row, 'childCount')})"
    state, _, reason = str(row.data(0, _user_role(qt_mod, ROLE_QC_MARK)) or "").partition("\t")
    if state in QC_MARK_TEXT:
        mark = QC_MARK_TEXT[state][1 if zh else 0]
        row.setToolTip(0, f"{mark}{'：' if zh else ': '}{reason}")
        text += f" · {mark}"
    return text


def _populate_manifest_output_items(
    owner: Any,
    parent: Any,
    manifest: dict[str, Any],
    *,
    manifest_path: Optional[Path] = None,
    run_index: Optional[int] = None,
    metadata_dir: Optional[Path] = None,
    memo: Optional[dict] = None,
) -> None:
    """Add the files of one manifest under ``parent``; a file that a newer manifest already added is skipped.

    ``metadata_dir`` is the ``00_run_metadata`` folder of the owning record (an archived manifest lives deeper, in
    ``history/<run>/<stamp>/``). ``memo`` is shared by the manifests of one subject rebuild: a task that a newer
    manifest of the rebuild already read with the same name, state and outputs is not read again, and a QC folder is
    scanned once. Both only skip work whose rows are already in the tree.
    """

    import qt
    from shared.result_scene import result_candidates

    if memo is not None:
        manifest = _without_repeated_tasks(manifest, memo)
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
        include_surface_morphometry=True,
        include_qc_not_passed=True,
    )
    filtered_candidates = [
        item for item in candidates
        if not _is_internal_cache_path(Path(item["path"]), subject_root, output_root=output_root)
    ]
    visual_paths = {item["path"] for item in filtered_candidates
                    if Path(item["path"]).suffix.lower() not in {".csv", ".tsv", ".stats"}}
    morphometry_paths = _morphometry_paths(filtered_candidates)
    qc_marks = {item["path"]: f"{item['qc_mark']}\t{item.get('qc_reason') or ''}"
                for item in filtered_candidates if item.get("qc_mark")}
    seg_paths = {item["path"] for item in filtered_candidates
                 if str(item.get("product") or "").startswith("segmentation")}
    from shared.fiber_display import streamline_kind
    fiber_paths = {item["path"] for item in filtered_candidates if streamline_kind(item)}
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
    # memo["qc_dirs"]: folder -> is a folder, for this rebuild. A folder that a newer manifest scanned is not scanned
    # again (its rows are in the tree), and a missing folder is checked once.
    checked = memo["qc_dirs"] if memo is not None else {}

    def _check(folder: Path) -> str:
        key = str(folder)
        if key in checked:
            return "done" if checked[key] else "missing"
        checked[key] = folder.is_dir()
        return "scan" if checked[key] else "missing"

    state = _check(qc_dir) if qc_dir is not None else "missing"
    if state == "missing" and resolved_manifest is not None:
        metadata = Path(metadata_dir) if metadata_dir is not None else Path(resolved_manifest).parent
        qc_dir = metadata.parent / "06_qc"
        state = _check(qc_dir)
    if state == "scan":
        for txt_path in sorted(qc_dir.rglob("*.txt")):
            if txt_path.is_file() and not _is_internal_cache_path(txt_path, subject_root, output_root=output_root):
                groups.setdefault("qc", {})[str(txt_path.resolve())] = False

    zh = str(getattr(owner, "_onboardingLanguage", "en")).startswith("zh")
    # family -> subgroup (None for a flat family) -> {path: visual}
    families: dict = {}
    # Results sources rule 3: the source belongs to the task, so every row of a task (its files and its QC JSON rows,
    # both grouped by the task name) goes to the task's one subgroup.
    task_by_name: dict = {}
    for entry in manifest.get("tasks") or []:
        if isinstance(entry, dict) and entry.get("name"):
            task_by_name.setdefault(str(entry["name"]), entry)
    for product, paths in groups.items():
        family = product.split(".", 1)[0]
        if family in {"surface", "segmentation", "parcellation", "postprocess"}:
            family = "anatomy"
        subgroup = _result_subgroup(family, product)
        source = _task_source(family, task_by_name.get(product) or {"name": product}, manifest) if subgroup else None
        if source is not None:
            subgroup = f"{subgroup}@{source}"
        families.setdefault(family, {}).setdefault(subgroup, {}).update(paths)
    role_manifest = _user_role(qt, ROLE_MANIFEST)
    role_run = _user_role(qt, ROLE_RUN)
    role_key = _user_role(qt, ROLE_KEY)
    role_path = _user_role(qt, ROLE_PATH)
    manifest_str = str(resolved_manifest) if resolved_manifest is not None else None
    family_order = RESULT_FAMILY_ORDER
    ordered_families = [
        *[family for family in family_order if family in families],
        *[family for family in families if family not in family_order],
    ]
    # The parent may already hold families, subgroups and files of a newer run: add to them, skip what is there.
    existing: dict = {}
    existing_subgroups: dict = {}
    seen_paths: set = set()
    for child in _child_items(parent):
        if not callable(getattr(child, "data", None)):
            continue
        family_key = child.data(0, role_key)
        if family_key:
            existing[family_key] = child
        for row in _child_items(child):
            row_path = row.data(0, role_path)
            if row_path:
                seen_paths.add(str(row_path))
                continue
            row_key = row.data(0, role_key)
            if _is_subgroup_row(qt, row):
                existing_subgroups[row_key] = row
            for leaf in _child_items(row):
                leaf_path = leaf.data(0, role_path)
                if leaf_path:
                    seen_paths.add(str(leaf_path))

    def _mark_run(row: Any) -> None:
        # A family or subgroup row carries the run that made it: a click on it selects that run.
        if run_index is not None:
            row.setData(0, role_run, run_index)
        if manifest_str is not None and run_index is not None:
            row.setData(0, role_manifest, manifest_str)

    for family in ordered_families:
        subgroups = families[family]
        ranks = _subgroup_ranks(family, [name for name in subgroups if name is not None])
        fresh: list = []
        for subgroup in sorted(subgroups, key=lambda name: ranks.get(name, (float("inf"),))):
            paths = {path: visual for path, visual in subgroups[subgroup].items() if path not in seen_paths}
            if paths:
                seen_paths.update(paths)
                fresh.append((subgroup, paths))
        if not fresh:
            continue
        caption = _family_caption(family, zh)
        group = existing.get(family)
        if group is None:
            total = sum(len(paths) for _, paths in fresh)
            group = qt.QTreeWidgetItem(parent, [f"{caption} ({total})"])
            group.setData(0, role_key, family)
            _mark_run(group)
            existing[family] = group
            added_to_existing = False
        else:
            added_to_existing = True
        for subgroup, paths in fresh:
            target = group
            if subgroup is not None:
                key = _subgroup_key(family, subgroup)
                target = existing_subgroups.get(key)
                if target is None:
                    target = qt.QTreeWidgetItem(group, [f"{_subgroup_caption(key, zh)} ({len(paths)})"])
                    target.setData(0, role_key, key)
                    target.setData(0, _user_role(qt, ROLE_KIND), SUBGROUP_ROW_KIND)
                    _mark_run(target)
                    existing_subgroups[key] = target
                    sub_existed = False
                else:
                    sub_existed = True
                sub_marked = False
                mark = next((qc_marks[path] for path in paths if path in qc_marks), None)
                if mark is not None and not target.data(0, _user_role(qt, ROLE_QC_MARK)):
                    target.setData(0, _user_role(qt, ROLE_QC_MARK), mark)
                    sub_marked = True
            for path, visual in paths.items():
                item = qt.QTreeWidgetItem(target, [Path(path).name, "○" if visual else "↗"])
                item.setData(0, role_path, path)
                if path in seg_paths:
                    item.setData(0, _user_role(qt, ROLE_KIND), "segmentation")
                elif path in fiber_paths:
                    item.setData(0, _user_role(qt, ROLE_KIND), FIBER_ROW_KIND)
                elif path in morphometry_paths:
                    item.setData(0, _user_role(qt, ROLE_KIND), MORPHOMETRY_ROW_KIND)
                if manifest_str is not None:
                    item.setData(0, role_manifest, manifest_str)
                if run_index is not None:
                    item.setData(0, role_run, run_index)
                leaf_tip = _morphometry_tooltip(path, zh) if path in morphometry_paths else path
                if path in qc_marks:
                    qc_state, _, qc_reason = qc_marks[path].partition("\t")
                    if qc_state in QC_MARK_TEXT:
                        qc_label = QC_MARK_TEXT[qc_state][1 if zh else 0]
                        leaf_tip = f"{leaf_tip}\n{qc_label}{'：' if zh else ': '}{qc_reason}" if qc_reason else f"{leaf_tip}\n{qc_label}"
                item.setToolTip(0, leaf_tip)
            if subgroup is not None and (sub_existed or sub_marked):
                target.setText(0, _subgroup_text(qt, target, key, zh))
        if added_to_existing:
            group.setText(0, f"{caption} ({_file_count(group, role_path)})")


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
    _BROKEN_ARCHIVES_LOGGED.clear()
    previous = select_path or selected_manifest_path(owner)
    discovery = discover_runs(history_roots(owner))
    # Batch 2: the one place where runs removed from the Results list are filtered out. Every reader below (combo,
    # records, tree, item map, target index, empty row, selection) sees only the shown runs.
    records, owner._resultsHiddenRecords = _split_hidden_for_owner(owner, discovery.records)
    owner._runHistoryRecords = records
    combo.blockSignals(True)
    try:
        combo.clear()
        for record in _with_ended_runs(owner, records):
            combo.addItem(run_record_label(record), record.manifest_path.as_posix())
            combo.setItemData(combo.count - 1, run_record_details(record), 3)  # Qt::ToolTipRole
        if not records:
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
            for index, record in enumerate(records):
                if record.manifest_path.as_posix() == wanted:
                    target_index = index
                    break
        combo.setCurrentIndex(target_index)
        owner._runHistoryPreviousIndex = target_index
    finally:
        combo.blockSignals(False)
    _populate_run_tree(owner, records, target_index)
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
    _unhide_records(discovery.records)  # Batch 2: the runs this folder holds are listed, so the message stays true
    manual = list(getattr(owner, "_manualRunHistoryDirs", ()) or ())
    if directory not in manual:
        manual.append(directory)
        owner._manualRunHistoryDirs = tuple(manual)
    refresh_run_history(owner, select_path=discovery.records[0].manifest_path)
    slicer.util.showStatusMessage(
        f"Found {len(discovery.records)} run(s) under {directory}; selected the newest.", 4000
    )
