"""Pure presentation rules for run state, button enablement and log lines.

Kept free of Qt so the rules can be unit tested; the Qt glue lives in
``shared/ui_run_history.py`` and ``shared/module_widget.py``.
"""

from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class RunStatePresentation:
    label: str
    color: str
    busy: bool = False


# Mirrors the states of Slicer's CLI progress bar (Idle, Running, Completed,
# Completed with errors, Cancelled) so the module reads like a core module.
RUN_STATES: dict[str, RunStatePresentation] = {
    "idle": RunStatePresentation("Idle", "#607D8B"),
    "importing": RunStatePresentation("Importing inputs", "#1565C0", busy=True),
    "preflight": RunStatePresentation("Checking environment", "#1565C0", busy=True),
    "setup": RunStatePresentation("Preparing environment", "#1565C0", busy=True),
    "running": RunStatePresentation("Running", "#1565C0", busy=True),
    "completed": RunStatePresentation("Completed", "#2E7D32"),
    "completed_with_blocks": RunStatePresentation("Completed with blocked stages", "#E65100"),
    "failed": RunStatePresentation("Completed with errors", "#C62828"),
    "cancelled": RunStatePresentation("Cancelled", "#E65100"),
}


def run_state_presentation(state: str) -> RunStatePresentation:
    """Return label/colour for a run state; unknown states read as idle."""

    return RUN_STATES.get((state or "").strip(), RUN_STATES["idle"])


def run_state_from_manifest_status(status: str) -> str:
    """Map an orchestration/batch manifest status to a run state key."""

    value = (status or "").strip()
    if value == "partial_failed":
        return "failed"
    if value in RUN_STATES and value not in {"preflight", "importing", "setup"}:
        return value
    return "idle"


def format_elapsed_mmss(seconds: float) -> str:
    """Format an elapsed duration in seconds as ``MM:SS`` or ``HH:MM:SS``."""

    try:
        numeric = float(seconds)
    except (TypeError, ValueError):
        numeric = 0.0
    if not math.isfinite(numeric) or numeric < 0.0:
        numeric = 0.0
    total_seconds = int(numeric)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


_LIVE_STAGE_TAG_RE = re.compile(
    r"^\[(?:Orchestrator|Preprocessing|BrainMask|DDSeg|DDParcel|DDSurfer(?:\s+Postprocess)?|"
    r"Microstructure|Tractography|SAF|UKF|Bundles|WMQL|WMA|TractCloud|Connectome|QC|Report)\]\s*",
    re.IGNORECASE,
)

_LIVE_STAGE_LABELS: dict[str, tuple[str, str]] = {
    "preprocessing": ("00 · Preprocessing", "00 · 预处理"),
    "brain_mask": ("00 · Preprocessing (Brain mask)", "00 · 预处理（脑掩膜）"),
    "ddseg": ("01 · Tissue segmentation (DDSeg)", "01 · 组织分割 (DDSeg)"),
    "ddparcel": ("01 · Brain parcellation (DDParcel)", "01 · 脑区分割 (DDParcel)"),
    "ddsurfer": ("01 · Cortical surface (DDSurfer)", "01 · 皮层表面 (DDSurfer)"),
    "surface_postprocess": ("01 · Surface morphometry", "01 · 表面形态学后处理"),
    "microstructure": ("02 · Microstructure", "02 · 微结构分析"),
    "tractography": ("03 · Tractography", "03 · 纤维追踪"),
    "bundles": ("04 · Fiber bundles", "04 · 纤维束提取"),
    "connectome": ("05 · Connectome", "05 · 连接组"),
    "qc": ("06 · QC & reports", "06 · 质控与报告"),
    "pipeline": ("Pipeline", "流水线执行"),
}

_POSTPROCESS_RE = re.compile(r"(?:^|[\W_])(postprocess|freesurfer|recon-all|surface_morphometry)(?:$|[\W_])", re.IGNORECASE)
_BRAIN_MASK_RE = re.compile(r"(?:^|[\W_])(brainmask|brain_mask|brain mask|dwi_masking)(?:$|[\W_])", re.IGNORECASE)
_PREPROC_RE = re.compile(r"(?:^|[\W_])(preprocessing|dwifslpreproc)(?:$|[\W_])", re.IGNORECASE)
_DDSEG_RE = re.compile(r"(?:^|[\W_])ddseg(?:$|[\W_])", re.IGNORECASE)
_DDPARCEL_RE = re.compile(r"(?:^|[\W_])ddparcel(?:$|[\W_])", re.IGNORECASE)
_DDSURFER_RE = re.compile(r"(?:^|[\W_])ddsurfer(?:$|[\W_])", re.IGNORECASE)
_BUNDLES_RE = re.compile(r"(?:^|[\W_])(wmql|wma|tractcloud|fiber bundles|fibre bundles)(?:$|[\W_])", re.IGNORECASE)
_CONNECTOME_RE = re.compile(r"(?:^|[\W_])(connectome|tck2connectome)(?:$|[\W_])", re.IGNORECASE)
_TRACTOGRAPHY_RE = re.compile(
    r"(?:^|[\W_])(?:tractography|ukf|csd|csd_act|dti_det|saf|tckgen|act_prep)(?:$|[\W_])",
    re.IGNORECASE,
)
_ACT_UPPER_RE = re.compile(r"(?:^|[\W_])ACT(?:$|[\W_])")
_MICROSTRUCTURE_RE = re.compile(
    r"(?:^|[\W_])(microstructure|dti|free\s*water|freewater|free_water|noddi|dki|ifw)(?:$|[\W_])",
    re.IGNORECASE,
)
_QC_RE = re.compile(r"(?:^|[\W_])(qc|report|reports|manifest)(?:$|[\W_])", re.IGNORECASE)


def _concise_log_detail(line: str, limit: int = 96) -> str:
    stripped = _LIVE_STAGE_TAG_RE.sub("", line.strip())
    compact = " ".join(stripped.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def _classify_live_stage_key(text: str) -> Optional[str]:
    lowered = text.lower()
    if lowered.startswith("[background] starting") or lowered.startswith("[setup] ["):
        return None
    if lowered.startswith("[orchestrator] plan ") or lowered.startswith("[orchestrator] run plan:"):
        return None

    # Check explicit surface postprocess before generic [DDSurfer] / DDSurfer tags.
    if lowered.startswith("[ddsurfer postprocess]") or _POSTPROCESS_RE.search(text):
        return "surface_postprocess"
    if lowered.startswith("[brainmask]"):
        return "brain_mask"
    if lowered.startswith("[preprocessing]"):
        return "preprocessing"
    if lowered.startswith("[ddseg]"):
        return "ddseg"
    if lowered.startswith("[ddparcel]"):
        return "ddparcel"
    if lowered.startswith("[ddsurfer]"):
        return "ddsurfer"
    if lowered.startswith("[microstructure]"):
        return "microstructure"
    if lowered.startswith(("[tractography]", "[saf]", "[ukf]")):
        return "tractography"
    if lowered.startswith(("[bundles]", "[wmql]", "[wma]", "[tractcloud]")):
        return "bundles"
    if lowered.startswith("[connectome]"):
        return "connectome"
    if lowered.startswith(("[qc]", "[report]")):
        return "qc"

    if _DDSEG_RE.search(text):
        return "ddseg"
    if _DDPARCEL_RE.search(text):
        return "ddparcel"
    if _DDSURFER_RE.search(text):
        return "ddsurfer"
    if _BRAIN_MASK_RE.search(text):
        return "brain_mask"
    if _PREPROC_RE.search(text):
        return "preprocessing"
    if _CONNECTOME_RE.search(text):
        return "connectome"
    if _BUNDLES_RE.search(text):
        return "bundles"
    if _TRACTOGRAPHY_RE.search(text) or _ACT_UPPER_RE.search(text):
        return "tractography"
    if _MICROSTRUCTURE_RE.search(text):
        return "microstructure"
    if _QC_RE.search(text):
        return "qc"
    if text.startswith("[Orchestrator] Running "):
        return "pipeline"
    return None


def parse_live_stage_from_log_line(line: str, language: str = "en") -> Optional[tuple[str, str]]:
    """Map a real-time stdout log line to ``(localized_stage_label, concise_detail)`` or ``None``."""

    text = (line or "").strip()
    if not text:
        return None
    stage_key = _classify_live_stage_key(text)
    if stage_key is None:
        return None
    is_zh = str(language or "en").strip().lower().startswith("zh")
    en_label, zh_label = _LIVE_STAGE_LABELS[stage_key]
    return (zh_label if is_zh else en_label, _concise_log_detail(text))


def progress_bar_format(done: int, total: int) -> str:
    """Text shown inside the progress bar."""

    if total <= 0:
        return "Waiting for the run plan…"
    noun = "stage" if total == 1 else "stages"
    return f"{done} / {total} {noun}"


@dataclass(frozen=True)
class ButtonState:
    enabled: bool
    tooltip: str


RUN_BUTTON_READY_TOOLTIP = (
    "Check dependencies for the selected targets, then start a monitored background run. "
    "Missing Python packages are installed first only when that option is enabled."
)


def run_button_state(
    *,
    pipeline_active: bool,
    selected_targets: Iterable[str],
    batch_mode: bool,
    subject_id: str,
    subject_root_text: str,
    batch_root_text: str = "",
    topup_errors: Iterable[str] = (),
) -> ButtonState:
    """Keep Run clickable for validation; only an active job disables it."""

    if pipeline_active:
        return ButtonState(False, "A run is in progress. Use 'Cancel running job' to stop it.")
    if not [target for target in selected_targets if target and target != "validation"]:
        return ButtonState(True, "Select at least one product under 2. Analysis.")
    if batch_mode:
        if not (batch_root_text or subject_root_text).strip():
            return ButtonState(True, "Select a Batch folder under Inputs.")
    else:
        # C6: an empty Subject ID is derived from the input path, so only the input is required.
        if not (subject_root_text or "").strip():
            return ButtonState(True, "Fill in Subject root under 1. Inputs.")
    errors = [error for error in topup_errors if error]
    if errors:
        return ButtonState(True, "Fix distortion-correction settings under 3. Route and parameters: " + errors[0])
    return ButtonState(True, RUN_BUTTON_READY_TOOLTIP)


def gui_brain_mask_required(
    targets: Iterable[str],
    *,
    input_known: bool,
    input_mask_exists: bool,
    preprocessing_on: bool,
    mask_backend: str,
    preprocessing_stages: str = "",
) -> bool:
    """C5: provision the brain-mask runtime whenever the closure needs a mask and none is supplied.

    The runtime is also required when the input folder is not chosen yet, so Prepare
    installs it before the first run. A supplied ``input/nodif_brain_mask.nii.gz`` removes the need
    unless ``brain_mask`` is explicitly checked in ``preprocessing_stages`` (or legacy ``preprocessing_on``
    with ``mask_backend != 'provided'`` when ``preprocessing_stages`` is unset).
    """

    from shared.target_closure import MASK_TARGETS

    needs_mask = bool(set(MASK_TARGETS).intersection(targets))
    has_user_mask = bool(input_known and input_mask_exists)
    stages_text = (preprocessing_stages or "").strip()
    if stages_text:
        from shared.algorithm_selection import PREPROCESSING, algorithms
        stages = algorithms(stages_text, PREPROCESSING, "denoise,eddy,topup")
        if not has_user_mask:
            return bool(needs_mask or preprocessing_on)
        return bool(preprocessing_on and "brain_mask" in stages)
    missing_mask = needs_mask and not has_user_mask and (input_known or mask_backend != "provided")
    # Preprocessing (FSL eddy) changes the voxel space and may need a new mask in
    # corrected space even when a supplied input mask already exists.
    return bool(missing_mask or (preprocessing_on and mask_backend != "provided"))


SCENE_ACTION_TOOLTIPS = {
    "reuse": "Load this run's results. Nodes this run already owns and that are unchanged are reused, not duplicated.",
    "replace": "Remove the nodes this run owns, then load its results again.",
    "append": "Add another copy of this run's results next to the existing nodes.",
    "clear": "Remove the nodes this run owns. Source files are never deleted.",
}


def result_actions_state(*, has_subject_run: bool, is_batch: bool, error: str = "") -> ButtonState:
    """Enablement for Load, Replace, Append and Clear."""

    if error:
        return ButtonState(False, error)
    if is_batch:
        return ButtonState(False, "Batch runs list one entry per subject; choose a subject's run to load its images.")
    if not has_subject_run:
        return ButtonState(False, "Choose a run in the list above, or choose “Add a run folder…” in the Run list.")
    return ButtonState(True, "Manage only scene nodes owned by this run; source files are never deleted.")


LOG_LEVEL_COLORS = {
    "error": "#AA3434",
    "warning": "#865F24",
    "stage": "#335B70",
    "success": "#34654E",
    "info": "#425360",
}

DARK_LOG_LEVEL_COLORS = {
    "error": "#FF7B72",
    "warning": "#FFB74D",
    "stage": "#7CC4EA",
    "success": "#66BB6A",
    "info": "#D2DEE8",
}

_ERROR_PATTERN = re.compile(r"\b(error|failed|failure|traceback|exception|fatal)\b", re.IGNORECASE)
_WARNING_PATTERN = re.compile(r"\b(warning|warn|blocked|cancelled|skipped|deprecated)\b", re.IGNORECASE)
_SUCCESS_PATTERN = re.compile(r"\b(completed|passed|succeeded|ready to run|environment ready)\b", re.IGNORECASE)


def log_line_level(line: str) -> str:
    """Classify a pipeline log line for colouring."""

    text = line or ""
    if text.startswith("[Orchestrator] Running ") or text.startswith("[Background] Starting"):
        return "stage"
    if _ERROR_PATTERN.search(text):
        return "error"
    if _WARNING_PATTERN.search(text):
        return "warning"
    if _SUCCESS_PATTERN.search(text):
        return "success"
    return "info"


def log_line_html(line: str, *, dark: bool | None = None) -> str:
    """Escape a log line and wrap it in a coloured span (whitespace preserved)."""

    if dark is None:
        try:
            from shared.ui_styles import is_dark_mode

            use_dark = is_dark_mode()
        except Exception:
            use_dark = False
    else:
        use_dark = bool(dark)
    level = log_line_level(line)
    escaped = html.escape(line or "", quote=False)
    weight = "font-weight:600;" if level in {"stage", "error"} else ""
    palette = DARK_LOG_LEVEL_COLORS if use_dark else LOG_LEVEL_COLORS
    return f'<span style="white-space:pre;color:{palette[level]};{weight}">{escaped}</span>'
