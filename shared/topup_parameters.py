"""Validation and mapping of topup / distortion-correction inputs from the GUI.

The one preprocessing route (DIPY MP-PCA, DIPY Gibbs removal, FSL eddy/topup
through ``dwifslpreproc``) passes three optional values to ``dwifslpreproc``:

* ``--preprocessing-phase-encoding-direction`` -> ``-pe_dir`` (always passed; W2b: from the sidecar when not given)
* ``--preprocessing-readout-time``             -> ``-readout_time`` (seconds; omitted when no value is known)
* ``--preprocessing-reverse-phase-b0``         -> ``-rpe_pair -se_epi``

This module validates what the user typed and maps it onto exactly those CLI
flags and the matching ``ProductOrchestratorConfig`` fields, so the GUI and
the command line stay in lock step. It is pure Python (no Qt/Slicer).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


# Accepted by MRtrix3 ``phaseencoding.direction`` (dwifslpreproc -pe_dir):
# NIfTI/BIDS axis codes and anatomical axis designators.
NIFTI_PHASE_ENCODING_DIRECTIONS = ("j-", "j", "i", "i-", "k", "k-")
ANATOMICAL_PHASE_ENCODING_DIRECTIONS = ("AP", "PA", "LR", "RL", "IS", "SI")
PHASE_ENCODING_DIRECTIONS = NIFTI_PHASE_ENCODING_DIRECTIONS + ANATOMICAL_PHASE_ENCODING_DIRECTIONS

# W2b (T1 D1): the GUI offers the anatomical designators only (a NIfTI axis code would need the affine to be
# understood). The empty value means "take it from input/dwi.json"; with no sidecar value the run is blocked.
PHASE_ENCODING_CHOICES: tuple[tuple[str, str], ...] = (
    ("", "From sidecar (input/dwi.json)"),
    ("AP", "AP  (anterior → posterior)"),
    ("PA", "PA  (posterior → anterior)"),
    ("LR", "LR  (left → right)"),
    ("RL", "RL  (right → left)"),
    ("IS", "IS  (inferior → superior)"),
    ("SI", "SI  (superior → inferior)"),
)

# Total readout time in seconds. Typical EPI values are 0.02-0.15 s; MRtrix3's
# own examples go up to ~0.7 s. Anything above 1 s is almost certainly a value
# typed in milliseconds.
READOUT_TIME_MIN_S = 0.001
READOUT_TIME_MAX_S = 1.0
READOUT_TIME_TYPICAL_MAX_S = 0.2
DWIFSLPREPROC_DEFAULT_READOUT_S = 0.1

NIFTI_SUFFIXES = (".nii", ".nii.gz", ".mif", ".mif.gz", ".mgh", ".mgz")
NRRD_SUFFIXES = (".nrrd", ".nhdr")
REVERSE_PE_SUFFIXES = NIFTI_SUFFIXES + NRRD_SUFFIXES

CLI_FLAG_PHASE_ENCODING = "--preprocessing-phase-encoding-direction"
CLI_FLAG_READOUT_TIME = "--preprocessing-readout-time"
CLI_FLAG_REVERSE_PHASE_B0 = "--preprocessing-reverse-phase-b0"


@dataclass(frozen=True)
class TopupParameters:
    """Validated distortion-correction inputs; ``None`` means not set."""

    phase_encoding_direction: Optional[str] = None
    readout_time: Optional[float] = None
    reverse_phase_b0: Optional[Path] = None

    @property
    def is_empty(self) -> bool:
        return self.phase_encoding_direction is None and self.readout_time is None and self.reverse_phase_b0 is None


@dataclass(frozen=True)
class TopupValidation:
    """Result of validating the GUI fields."""

    parameters: TopupParameters
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


def route_uses_topup(preprocessing_selected: bool = True, preprocessing_stages: Optional[str] = None) -> bool:
    """True when the topup / eddy PE fields apply (and should be shown)."""

    if not preprocessing_selected:
        return False
    text = (preprocessing_stages or "").strip()
    if not text:
        return True
    from shared.algorithm_selection import PREPROCESSING, algorithms
    stages = algorithms(text, PREPROCESSING, "denoise,eddy")
    return bool({"eddy", "topup"} & set(stages))


def normalize_phase_encoding_direction(value: Any) -> Optional[str]:
    """Return the canonical spelling, ``None`` for empty, or raise ``ValueError``."""

    text = str(value or "").strip()
    if not text:
        return None
    for allowed in PHASE_ENCODING_DIRECTIONS:
        if text.lower() == allowed.lower():
            return allowed
    raise ValueError(
        f"Phase-encoding direction {text!r} is not recognised. Use one of "
        + ", ".join(PHASE_ENCODING_DIRECTIONS)
        + " (the value dwifslpreproc accepts for -pe_dir)."
    )


def parse_readout_time(value: Any) -> Optional[float]:
    """Parse the total readout time in seconds; ``None``/0/empty means not set."""

    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("Total readout time must be a number of seconds, e.g. 0.0646.")
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip().replace(",", ".")
        if text.lower().endswith("s") and not text.lower().endswith("ms"):
            text = text[:-1].strip()
        if not text:
            return None
        if text.lower().endswith("ms"):
            raise ValueError(
                f"Total readout time {value!r} looks like milliseconds. Enter seconds instead "
                "(for example 64.6 ms → 0.0646)."
            )
        try:
            number = float(text)
        except ValueError as exc:
            raise ValueError(f"Total readout time {value!r} is not a number. Enter seconds, e.g. 0.0646.") from exc
    if not math.isfinite(number):
        raise ValueError("Total readout time must be a finite number of seconds.")
    if number == 0.0:
        return None
    if number < 0:
        raise ValueError("Total readout time must be positive (seconds).")
    if number > READOUT_TIME_MAX_S:
        hint = " It looks like milliseconds; divide by 1000." if number >= 1.0 else ""
        raise ValueError(
            f"Total readout time {number:g} s is outside the accepted range "
            f"{READOUT_TIME_MIN_S:g}–{READOUT_TIME_MAX_S:g} s.{hint}"
        )
    if number < READOUT_TIME_MIN_S:
        raise ValueError(
            f"Total readout time {number:g} s is outside the accepted range "
            f"{READOUT_TIME_MIN_S:g}–{READOUT_TIME_MAX_S:g} s."
        )
    return number


def _has_image_suffix(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in REVERSE_PE_SUFFIXES)


def validate_reverse_phase_b0_path(
    reverse_phase_b0: Any = "",
    *,
    batch_mode: bool = False,
    zh: bool = False,
) -> tuple[Optional[Path], Optional[str]]:
    """Validate an optional reverse-PE b=0 file path.

    Returns ``(path, None)`` when a valid file is given, ``(None, None)`` when unset, or ``(None, error)`` when invalid.
    """

    reverse_text = str(reverse_phase_b0 or "").strip()
    if not reverse_text:
        return None, None
    candidate = Path(reverse_text).expanduser()
    if batch_mode:
        if zh:
            return None, (
                "反向相位编码 b=0 图像仅适用于单个受试者，批处理模式会同时运行根目录下所有受试者。"
                "请清空反向相位 b=0 图像，或关闭批处理模式逐个运行。"
            )
        return None, (
            "A reverse-PE b=0 image applies to one subject, but batch mode runs every subject under the "
            "batch root. Clear 'Reverse-PE b=0 image' or turn off Batch mode and run subjects one at a time."
        )
    if not candidate.exists():
        if zh:
            return None, f"找不到反向相位编码 b=0 图像：{candidate}。请选择有效的图像文件。"
        return None, f"Reverse-PE b=0 image not found: {candidate}. Choose an existing NIfTI/NRRD file."
    if not candidate.is_file():
        if zh:
            return None, f"反向相位编码 b=0 图像必须是文件，不能是文件夹：{candidate}。"
        return None, f"Reverse-PE b=0 image must be a file, not a folder: {candidate}."
    if not _has_image_suffix(candidate):
        if zh:
            return None, (
                f"反向相位编码 b=0 图像 {candidate.name} 不是受支持的图像文件"
                "（.nii、.nii.gz、.nrrd、.nhdr、.mif）。请选择反向相位编码 b=0 图像。"
            )
        return None, (
            f"Reverse-PE b=0 image {candidate.name} is not a NIfTI/NRRD/MRtrix image "
            "(.nii, .nii.gz, .nrrd, .nhdr, .mif). Choose the reversed phase-encoding b=0 volume."
        )
    return candidate, None


def topup_unavailable_reason(
    reverse_phase_b0: Any = "",
    *,
    batch_mode: bool = False,
    zh: bool = False,
) -> str:
    """Return why Topup cannot be selected right now, or ``""`` when a valid reverse-PE b=0 file is ready."""

    if batch_mode:
        if zh:
            return (
                "Topup 需要为每个受试者单独提供反向相位编码（例如主 DWI 为 AP 时提供 PA）的 b=0 图像，"
                "批处理模式下不可用。请切换到单受试者模式后再提供该文件。"
            )
        return (
            "Topup requires a per-subject reverse phase-encoding b=0 image (for example, PA matching an AP DWI) "
            "and is not available in Batch mode. Switch to single-subject mode to provide one."
        )
    reverse_text = str(reverse_phase_b0 or "").strip()
    if not reverse_text:
        if zh:
            return (
                "运行 Topup 需要提供与主 DWI 相位编码方向相反的 b=0 图像"
                "（例如主 DWI 为 AP 方向时，提供 PA 方向的 b=0 图像）。"
                "请在下方选择反向 b=0 文件（支持 .nii、.nii.gz、.nrrd、.nhdr、.mif）后即可勾选 Topup。"
            )
        return (
            "Topup requires a reverse phase-encoding b=0 image (for example, a PA b=0 volume matching an AP main DWI). "
            "Select a reverse-PE b=0 file (.nii, .nii.gz, .nrrd, .nhdr, .mif) below to enable Topup."
        )
    _, error = validate_reverse_phase_b0_path(reverse_text, batch_mode=False, zh=zh)
    return error or ""


def sanitize_unavailable_topup_stage(node: Any, *, batch_mode: Optional[bool] = None) -> bool:
    """Strip ``topup`` from a multi-stage selection when Topup cannot run (no reverse-PE b=0 or Batch mode).

    Prevents legacy saved settings (``denoise,eddy,topup`` with no reverse-PE b=0) or switching into Batch mode from
    displaying ``DWI denoising + Eddy`` in the UI while blocking Run on a hidden ``topup`` stage. An explicit
    ``topup``-only selection is left intact so the UI and validation still explain what is missing.
    """

    if node is None:
        return False
    raw_stages_val = getattr(node, "orchestrationPreprocessingStages", "")
    raw_stages = raw_stages_val.strip() if isinstance(raw_stages_val, str) else ""
    if not raw_stages:
        return False
    if batch_mode is None:
        cohort = bool(
            getattr(node, "experimentsEnabled", False) is True
            and getattr(node, "experimentScope", "") == "cohort"
        )
        is_batch = cohort or bool(getattr(node, "batchMode", False) is True)
    else:
        is_batch = bool(batch_mode)
    reverse_b0_val = getattr(node, "preprocessingReversePhaseB0", "")
    reverse_b0 = reverse_b0_val.strip() if isinstance(reverse_b0_val, str) else ""
    if not topup_unavailable_reason(reverse_b0, batch_mode=is_batch):
        return False
    from shared.algorithm_selection import PREPROCESSING, algorithms

    try:
        chosen = algorithms(raw_stages, PREPROCESSING, "denoise,eddy")
    except ValueError:
        return False
    if "topup" not in chosen:
        return False
    kept = [stage for stage in chosen if stage != "topup"]
    if not kept:
        return False
    wanted = ",".join(kept)
    if raw_stages != wanted:
        node.orchestrationPreprocessingStages = wanted
        return True
    return False


def validate_topup_parameters(
    *,
    preprocessing_selected: bool,
    phase_encoding_direction: Any = "",
    readout_time: Any = None,
    reverse_phase_b0: Any = "",
    batch_mode: bool = False,
    subject_root: Any = None,
    output_root: Any = None,
    preprocessing_stages: Optional[str] = None,
) -> TopupValidation:
    """Validate the three GUI fields for the preprocessing route.

    When preprocessing is not selected the fields are ignored and
    nothing is forwarded, so pipeline defaults are untouched.

    W2b rule 4 (T1 D1): with a ``subject_root`` (single-subject runs) the same resolution as the run is applied to the
    chosen direction and the sidecar ``input/dwi.json``. When no direction results, the message is an error, so the
    run is stopped before any stage. Earlier preprocessing outputs do not lift the block: selected preprocessing is reused
    only by the full version-6 key of the current settings, which needs a direction (F5). Batch mode is checked per
    subject by the cohort plan.
    """

    if not route_uses_topup(preprocessing_selected, preprocessing_stages):
        return TopupValidation(parameters=TopupParameters())

    errors: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []

    pe_direction: Optional[str] = None
    try:
        pe_direction = normalize_phase_encoding_direction(phase_encoding_direction)
    except ValueError as exc:
        errors.append(str(exc))

    readout: Optional[float] = None
    try:
        readout = parse_readout_time(readout_time)
    except ValueError as exc:
        errors.append(str(exc))

    reverse_text = str(reverse_phase_b0 or "").strip()
    reverse_path, reverse_error = validate_reverse_phase_b0_path(reverse_text, batch_mode=batch_mode, zh=False)
    if reverse_error:
        errors.append(reverse_error)

    stages_text = (preprocessing_stages or "").strip()
    eddy_or_reverse_needed = True
    effective_has_reverse = bool(reverse_text)
    if stages_text:
        from shared.algorithm_selection import PREPROCESSING, algorithms
        chosen_stages = algorithms(stages_text, PREPROCESSING, "denoise,eddy")
        topup_selected = "topup" in chosen_stages
        if topup_selected and not reverse_text:
            if batch_mode:
                errors.append(
                    "Topup requires a per-subject reverse-PE b=0 image and cannot run in batch mode. "
                    "Uncheck Topup or turn off Batch mode."
                )
            else:
                errors.append(
                    "Topup requires a reverse-PE b=0 image (for example, a PA b=0 matching an AP DWI). "
                    "Select a reverse-PE b=0 file under Preprocessing > … > Topup, or uncheck Topup."
                )
        effective_has_reverse = bool(reverse_text) and topup_selected
        eddy_or_reverse_needed = ("eddy" in chosen_stages) or effective_has_reverse

    if subject_root is not None and not batch_mode and not errors and eddy_or_reverse_needed:
        from shared.pe_resolution import phase_encoding_block_reason

        root_path = Path(str(subject_root)).expanduser()
        if root_path.is_file():
            root_path = root_path.parent
        # F5: the orchestrator reuses selected preprocessing only by the full version-6 key of the current settings, and
        # that key holds the effective PE direction. With no direction there is no matching key, so no outputs of an
        # earlier run (version 5 or 6) make the run possible: the block does not depend on earlier outputs.
        blocked = phase_encoding_block_reason(
            root_path, user_phase_encoding=pe_direction, user_readout_time=readout,
            has_reverse_b0=effective_has_reverse)
        if blocked:
            errors.append(blocked)
    if pe_direction is None and not errors:
        notes.append(
            "Phase-encoding direction: taken from input/dwi.json (PhaseEncodingDirection) when you leave it on "
            "'From sidecar'. If the sidecar has none, preprocessing does not start (most axial clinical DWI: AP)."
        )
    if readout is None and not errors:
        notes.append(
            "Total readout time not set: TotalReadoutTime from input/dwi.json is used when it exists; otherwise "
            f"dwifslpreproc assumes {DWIFSLPREPROC_DEFAULT_READOUT_S:g} s (recorded as readout_assumed)."
        )
    elif readout is not None and readout > READOUT_TIME_TYPICAL_MAX_S:
        warnings.append(
            f"Total readout time {readout:g} s is longer than usual (typically 0.02–{READOUT_TIME_TYPICAL_MAX_S:g} s). "
            "Check the TotalReadoutTime value in the scan's JSON sidecar."
        )
    if reverse_path is None and not reverse_text and not batch_mode:
        notes.append("No reverse-PE b=0 image: eddy runs without topup field estimation (-rpe_none).")

    parameters = TopupParameters(
        phase_encoding_direction=pe_direction,
        readout_time=readout,
        reverse_phase_b0=reverse_path,
    )
    return TopupValidation(
        parameters=parameters,
        errors=tuple(errors),
        warnings=tuple(warnings),
        notes=tuple(notes),
    )


def topup_cli_args(parameters: TopupParameters) -> list[str]:
    """Return ``run_subject_pipeline.py`` flags for the set values only."""

    args: list[str] = []
    if parameters.phase_encoding_direction:
        args.extend([CLI_FLAG_PHASE_ENCODING, parameters.phase_encoding_direction])
    if parameters.readout_time is not None:
        args.extend([CLI_FLAG_READOUT_TIME, repr(float(parameters.readout_time))])
    if parameters.reverse_phase_b0 is not None:
        args.extend([CLI_FLAG_REVERSE_PHASE_B0, Path(parameters.reverse_phase_b0).expanduser().as_posix()])
    return args


def topup_orchestrator_kwargs(parameters: TopupParameters) -> dict[str, Any]:
    """Return the ``ProductOrchestratorConfig`` fields the CLI would set."""

    return {
        "preprocessing_reverse_phase_b0": parameters.reverse_phase_b0,
        "preprocessing_phase_encoding_direction": parameters.phase_encoding_direction,
        "preprocessing_readout_time": parameters.readout_time,
    }


def topup_summary_text(validation: TopupValidation) -> str:
    """Short status line for the GUI hint label."""

    if validation.errors:
        return "Fix before running: " + " ".join(validation.errors)
    parts = []
    if validation.warnings:
        parts.append("Warning: " + " ".join(validation.warnings))
    if validation.notes:
        parts.append(" ".join(validation.notes))
    return " ".join(parts)
