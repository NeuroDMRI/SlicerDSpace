"""Pure presentation rules for run state, button enablement and log lines.

Kept free of Qt so the rules can be unit tested; the Qt glue lives in
``shared/ui_run_history.py`` and ``shared/module_widget.py``.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RunStatePresentation:
    label: str
    color: str
    busy: bool = False


# Mirrors the states of Slicer's CLI progress bar (Idle, Running, Completed,
# Completed with errors, Cancelled) so the module reads like a core module.
RUN_STATES: dict[str, RunStatePresentation] = {
    "idle": RunStatePresentation("Idle", "#607D8B"),
    "preflight": RunStatePresentation("Checking environment", "#1565C0", busy=True),
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
    if value in RUN_STATES and value not in {"preflight"}:
        return value
    return "idle"


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
        return ButtonState(True, "Select at least one product under 2. Products.")
    if batch_mode:
        if not (batch_root_text or subject_root_text).strip():
            return ButtonState(True, "Set 'Batch root' under Analysis settings (or a Subject root) for batch mode.")
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
) -> bool:
    """C5: provision the brain-mask runtime whenever the closure needs a mask and none is supplied.

    The runtime is also required when the input folder is not chosen yet, so Prepare
    installs it before the first run. A supplied ``input/nodif_brain_mask.nii.gz`` removes the need.
    """

    from shared.target_closure import MASK_TARGETS

    needs_mask = bool(set(MASK_TARGETS).intersection(targets))
    has_user_mask = bool(input_known and input_mask_exists)
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


def log_line_html(line: str) -> str:
    """Escape a log line and wrap it in a coloured span (whitespace preserved)."""

    level = log_line_level(line)
    escaped = html.escape(line or "", quote=False)
    weight = "font-weight:600;" if level in {"stage", "error"} else ""
    return f'<span style="white-space:pre;color:{LOG_LEVEL_COLORS[level]};{weight}">{escaped}</span>'
