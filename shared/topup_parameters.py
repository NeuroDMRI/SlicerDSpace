"""Validation and mapping of topup / distortion-correction inputs from the GUI.

The one preprocessing route (DIPY MP-PCA, DIPY Gibbs removal, FSL eddy/topup
through ``dwifslpreproc``) passes three optional values to ``dwifslpreproc``:

* ``--preprocessing-phase-encoding-direction`` -> ``-pe_dir``
* ``--preprocessing-readout-time``             -> ``-readout_time`` (seconds)
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

PHASE_ENCODING_CHOICES: tuple[tuple[str, str], ...] = (
    ("", "Not set"),
    ("j-", "j-  (NIfTI axis code)"),
    ("j", "j  (NIfTI axis code)"),
    ("i", "i  (NIfTI axis code)"),
    ("i-", "i-  (NIfTI axis code)"),
    ("k", "k  (NIfTI axis code)"),
    ("k-", "k-  (NIfTI axis code)"),
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


def route_uses_topup(preprocessing_selected: bool = True) -> bool:
    """True when the topup fields apply (and should be shown)."""

    return bool(preprocessing_selected)


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
    return any(name.endswith(suffix) for suffix in NIFTI_SUFFIXES)


def validate_topup_parameters(
    *,
    preprocessing_selected: bool,
    phase_encoding_direction: Any = "",
    readout_time: Any = None,
    reverse_phase_b0: Any = "",
    batch_mode: bool = False,
) -> TopupValidation:
    """Validate the three GUI fields for the preprocessing route.

    When preprocessing is not selected the fields are ignored and
    nothing is forwarded, so pipeline defaults are untouched.
    """

    if not route_uses_topup(preprocessing_selected):
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

    reverse_path: Optional[Path] = None
    reverse_text = str(reverse_phase_b0 or "").strip()
    if reverse_text:
        candidate = Path(reverse_text).expanduser()
        if batch_mode:
            errors.append(
                "A reverse-PE b=0 image applies to one subject, but batch mode runs every subject under the "
                "batch root. Clear 'Reverse-PE b=0 image' or turn off Batch mode and run subjects one at a time."
            )
        elif not candidate.exists():
            errors.append(f"Reverse-PE b=0 image not found: {candidate}. Choose an existing NIfTI file.")
        elif not candidate.is_file():
            errors.append(f"Reverse-PE b=0 image must be a file, not a folder: {candidate}.")
        elif not _has_image_suffix(candidate):
            errors.append(
                f"Reverse-PE b=0 image {candidate.name} is not a NIfTI/MRtrix image "
                "(.nii, .nii.gz, .mif). Choose the reversed phase-encoding b=0 volume."
            )
        else:
            reverse_path = candidate

    if pe_direction is None and not errors:
        warnings.append(
            "Phase-encoding direction is not set. dwifslpreproc stops with "
            "'phase encoding direction must be provided using the -pe_dir option' unless "
            "preprocessing outputs from an earlier run are reused."
        )
    if readout is None and not errors:
        notes.append(
            f"Total readout time not set: dwifslpreproc assumes {DWIFSLPREPROC_DEFAULT_READOUT_S:g} s, "
            "which only affects the field-map scaling."
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
