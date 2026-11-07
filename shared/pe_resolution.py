"""Phase-encoding direction and readout time for the one preprocessing route (W2b / T1 D1).

Each field is resolved on its own.

Phase-encoding direction:
  1. the user value (GUI combo; CLI, where NIfTI axis codes are allowed);
  2. the sidecar ``PhaseEncodingDirection`` of ``input/dwi.json``;
  3. the sidecar ``PhaseEncodingAxis`` only: that axis, positive sign, flag ``pe_sign_assumed`` (gated, see below);
  4. none: preprocessing is blocked at plan time. There is no silent default.

Every value reaches ``dwifslpreproc -pe_dir`` as an anatomical designator (``AP``, ``PA``, ``LR``, ``RL``, ``IS``,
``SI``). MRtrix applies a NIfTI axis code to its own near-RAS image axes, not to the axes of the file, so a code
(``j-``) is converted with the affine of the imported DWI first.

Readout time: the user value, then the sidecar ``TotalReadoutTime``, then none (``-readout_time`` is left out,
``dwifslpreproc`` assumes 0.1 s, flag ``readout_assumed``).

Pure Python. nibabel is imported only when an affine must be read from a file.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

ANATOMICAL_DESIGNATORS = ("AP", "PA", "LR", "RL", "IS", "SI")
SIDECAR_NAME = "dwi.json"

RULE4_MESSAGE = (
    "Phase-encoding direction needed: choose AP, PA, LR, RL, IS or SI in Preprocessing "
    "(most axial clinical DWI: AP)"
)

# An axis whose two largest world components differ by less than this factor is oblique: refused.
OBLIQUE_RATIO = 1.5
# The tolerance dwifslpreproc uses to compare two readout times.
READOUT_CONFLICT_TOLERANCE_S = 5e-3
DWIFSLPREPROC_DEFAULT_READOUT_S = 0.1

# T1 D1 rule 3, M-T1a: rule 3 (a sidecar that names only the PE axis) applies to a mode only after M-T1a measured
# that mode: a consistent sign flip of the DWI and the SE-EPI pair changed the result no more than a repeat run did.
# While a mode is False, axis-only sidecar data in that mode fall to rule 4. Read at call time.
PE_SIGN_ASSUMPTION_MEASURED = {"rpe_none": False, "rpe_pair": False}

# World axis names for the positive direction of x, y, z: the start letter, then the end letter.
_WORLD_AXES = (("L", "R"), ("P", "A"), ("I", "S"))
_AXIS_CODES = {"i": 0, "j": 1, "k": 2}


class PhaseEncodingNeeded(ValueError):
    """Rule 4: no phase-encoding direction is known. The message is the one the user sees."""


def mode_for(has_reverse_b0: bool) -> str:
    return "rpe_pair" if has_reverse_b0 else "rpe_none"


def _is_axis_code(text: str) -> bool:
    return len(text) in (1, 2) and text[0] in _AXIS_CODES and (len(text) == 1 or text[1] == "-")


def axis_code_to_designator(code: str, affine: Any) -> Optional[str]:
    """Convert a NIfTI axis code (``i``, ``j-``, ...) to an anatomical designator with the affine of the DWI.

    Take column a of the 3x3 part of the affine, multiply it by the sign, and let the largest absolute component
    name the world axis (x: L to R, y: P to A, z: I to S); its sign gives the direction. Return None when the axis
    is oblique (its two largest components differ by less than a factor of 1.5) or the affine is unusable.
    """

    text = str(code or "").strip().lower()
    if not _is_axis_code(text):
        return None
    axis = _AXIS_CODES[text[0]]
    sign = -1.0 if text.endswith("-") else 1.0
    try:
        column = [sign * float(affine[row][axis]) for row in range(3)]
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    if not all(math.isfinite(value) for value in column):
        return None
    ranked = sorted(range(3), key=lambda index: abs(column[index]), reverse=True)
    largest, second = abs(column[ranked[0]]), abs(column[ranked[1]])
    if largest == 0.0:
        return None
    if second > 0.0 and largest / second < OBLIQUE_RATIO:
        return None
    start, end = _WORLD_AXES[ranked[0]]
    return start + end if column[ranked[0]] > 0 else end + start


def designator_axis(designator: str) -> frozenset:
    """The world axis of a designator as a set of letters ('AP' and 'PA' both give {'A', 'P'})."""
    return frozenset(designator)


def _clean_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class AcquisitionMetadata:
    """The resolved PE direction and readout time, with sources, flags and conflicts."""

    mode: str
    phase_encoding: Optional[str]
    phase_encoding_source: Optional[str]
    phase_encoding_user: Optional[str]
    phase_encoding_sidecar_raw: Optional[str]
    phase_encoding_axis_sidecar_raw: Optional[str]
    readout_time: Optional[float]
    readout_time_source: str
    readout_time_user: Optional[float]
    readout_time_sidecar_raw: Any
    pe_sign_assumed: bool
    readout_assumed: bool
    sidecar_sha256: Optional[str]
    conflicts: tuple = ()
    warnings: tuple = ()
    error: Optional[str] = None

    @property
    def needs_phase_encoding(self) -> bool:
        return self.phase_encoding is None

    @property
    def readout_time_for_command(self) -> Optional[float]:
        """The value for ``-readout_time``; None when it is left out (the assumed case)."""
        return None if self.readout_assumed else self.readout_time

    def require_phase_encoding(self) -> str:
        if self.phase_encoding is None:
            raise PhaseEncodingNeeded(self.error or RULE4_MESSAGE)
        return self.phase_encoding

    def to_record(self) -> dict:
        """The manifest and QC record: per field the effective value, its source and the raw sidecar value."""
        return {
            "kind": "dwi_preprocessing_acquisition",
            "mode": self.mode,
            "phase_encoding": {
                "value": self.phase_encoding,
                "source": self.phase_encoding_source,
                "user": self.phase_encoding_user,
                "sidecar_raw": self.phase_encoding_sidecar_raw,
                "sidecar_axis_raw": self.phase_encoding_axis_sidecar_raw,
            },
            "readout_time": {
                "value": self.readout_time,
                "source": self.readout_time_source,
                "user": self.readout_time_user,
                "sidecar_raw": self.readout_time_sidecar_raw,
            },
            "pe_sign_assumed": self.pe_sign_assumed,
            "readout_assumed": self.readout_assumed,
            "sidecar_sha256": self.sidecar_sha256,
            "conflicts": [dict(conflict) for conflict in self.conflicts],
            "warnings": list(self.warnings),
        }

    def summary_text(self) -> str:
        """One line for the result panel: values, sources, flags and conflict warnings."""
        parts = []
        if self.phase_encoding is None:
            parts.append("phase encoding not set")
        else:
            parts.append(f"phase encoding {self.phase_encoding} ({self.phase_encoding_source})")
        if self.readout_time is None:
            parts.append("readout time not set")
        else:
            parts.append(f"readout time {self.readout_time:g} s ({self.readout_time_source})")
        if self.pe_sign_assumed:
            parts.append("pe_sign_assumed: the sidecar names only the axis; a positive sign was assumed")
        if self.readout_assumed:
            parts.append("readout_assumed: dwifslpreproc uses its default of 0.1 s")
        text = "; ".join(parts)
        if self.warnings:
            text += "; WARNING: " + " ".join(self.warnings)
        return text


def resolve_acquisition(
    *,
    user_phase_encoding: Optional[str] = None,
    user_readout_time: Optional[float] = None,
    sidecar: Optional[dict] = None,
    affine: Any = None,
    has_reverse_b0: bool = False,
    sidecar_sha256: Optional[str] = None,
) -> AcquisitionMetadata:
    """Resolve both fields. ``affine`` may be a matrix or a callable that returns one (read only when needed)."""

    mode = mode_for(has_reverse_b0)
    sidecar = sidecar if isinstance(sidecar, dict) else {}
    warnings: list[str] = []
    conflicts: list[dict] = []
    loaded: list = []

    def get_affine():
        if not loaded:
            try:
                loaded.append(affine() if callable(affine) else affine)
            except Exception:
                loaded.append(None)
        return loaded[0]

    def convert(code: str) -> Optional[str]:
        text = str(code or "").strip()
        if text.upper() in ANATOMICAL_DESIGNATORS:
            return text.upper()
        return axis_code_to_designator(text, get_affine()) if _is_axis_code(text.lower()) else None

    user_text = str(user_phase_encoding or "").strip() or None
    sidecar_direction = sidecar.get("PhaseEncodingDirection")
    sidecar_direction = sidecar_direction.strip() if isinstance(sidecar_direction, str) else None
    sidecar_axis = sidecar.get("PhaseEncodingAxis")
    sidecar_axis = sidecar_axis.strip() if isinstance(sidecar_axis, str) else None

    sidecar_converted = convert(sidecar_direction) if sidecar_direction else None
    value: Optional[str] = None
    source: Optional[str] = None
    sign_assumed = False
    error: Optional[str] = None
    axis_converted: Optional[str] = None

    if user_text is not None:
        value = convert(user_text)
        source = "user" if value else None
        if value is None:
            error = RULE4_MESSAGE
    elif sidecar_converted is not None:
        value, source = sidecar_converted, "sidecar"
    elif sidecar_direction is None and sidecar_axis and _is_axis_code(sidecar_axis.lower()):
        # Rule 3: only the axis. Allowed for a mode only after M-T1a measured that mode (T1 D1 rule 3, M-T1a).
        if PE_SIGN_ASSUMPTION_MEASURED.get(mode, False):
            axis_converted = convert(sidecar_axis.rstrip("-"))
            if axis_converted is not None:
                value, source, sign_assumed = axis_converted, "sidecar_axis", True
    if value is None and error is None:
        error = RULE4_MESSAGE

    if sidecar_direction and sidecar_converted is None and user_text is None:
        warnings.append(
            f"The sidecar PhaseEncodingDirection {sidecar_direction!r} could not be converted to AP, PA, LR, RL, IS or SI "
            "(unknown value or an oblique image axis)."
        )

    # Conflict: a user value wins; a different sidecar value is recorded and shown.
    if value is not None and source == "user":
        if sidecar_converted is not None and sidecar_converted != value:
            conflicts.append({"field": "phase_encoding", "user": value, "sidecar": sidecar_converted,
                              "sidecar_raw": sidecar_direction})
            warnings.append(f"Phase-encoding direction {value} was set by you; the sidecar says "
                            f"{sidecar_converted} ({sidecar_direction}). Your value is used.")
        elif sidecar_direction is None and sidecar_axis and _is_axis_code(sidecar_axis.lower()):
            axis_only = convert(sidecar_axis.rstrip("-"))
            if axis_only is not None and designator_axis(axis_only) != designator_axis(value):
                conflicts.append({"field": "phase_encoding", "user": value, "sidecar": axis_only,
                                  "sidecar_raw": sidecar_axis})
                warnings.append(f"Phase-encoding direction {value} was set by you; the sidecar names the axis "
                                f"{sidecar_axis} ({axis_only[0]}{axis_only[1]} axis). Your value is used.")

    # Readout time.
    user_readout = _clean_number(user_readout_time)
    if user_readout is not None and user_readout <= 0:
        user_readout = None
    raw_readout = sidecar.get("TotalReadoutTime")
    sidecar_readout = _clean_number(raw_readout)
    if raw_readout is not None and (sidecar_readout is None or sidecar_readout <= 0):
        warnings.append(f"The sidecar TotalReadoutTime {raw_readout!r} is not a positive number and is ignored.")
        sidecar_readout = None
    elif sidecar_readout is not None:
        from shared.topup_parameters import READOUT_TIME_MAX_S, READOUT_TIME_MIN_S
        if not READOUT_TIME_MIN_S <= sidecar_readout <= READOUT_TIME_MAX_S:
            warnings.append(
                f"The sidecar TotalReadoutTime {sidecar_readout:g} s is outside "
                f"{READOUT_TIME_MIN_S:g}-{READOUT_TIME_MAX_S:g} s and is ignored (is it in milliseconds?)."
            )
            sidecar_readout = None
    readout_assumed = False
    if user_readout is not None:
        readout, readout_source = user_readout, "user"
        if sidecar_readout is not None and abs(sidecar_readout - user_readout) > READOUT_CONFLICT_TOLERANCE_S:
            conflicts.append({"field": "readout_time", "user": user_readout, "sidecar": sidecar_readout})
            warnings.append(f"Readout time {user_readout:g} s was set by you; the sidecar says "
                            f"{sidecar_readout:g} s. Your value is used.")
    elif sidecar_readout is not None:
        readout, readout_source = sidecar_readout, "sidecar"
    else:
        readout, readout_source, readout_assumed = DWIFSLPREPROC_DEFAULT_READOUT_S, "assumed", True

    return AcquisitionMetadata(
        mode=mode,
        phase_encoding=value,
        phase_encoding_source=source,
        phase_encoding_user=user_text,
        phase_encoding_sidecar_raw=sidecar_direction,
        phase_encoding_axis_sidecar_raw=sidecar_axis,
        readout_time=readout,
        readout_time_source=readout_source,
        readout_time_user=user_readout,
        readout_time_sidecar_raw=raw_readout,
        pe_sign_assumed=sign_assumed,
        readout_assumed=readout_assumed,
        sidecar_sha256=sidecar_sha256,
        conflicts=tuple(conflicts),
        warnings=tuple(warnings),
        error=error,
    )


def read_sidecar(path: Path) -> tuple[Optional[dict], Optional[str]]:
    """Return (parsed sidecar or None, SHA-256 of the file or None). A broken file gives (None, sha)."""
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError:
        return None, None
    digest = hashlib.sha256(raw).hexdigest()
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None, digest
    return (parsed if isinstance(parsed, dict) else None), digest


def _affine_loader(dwi_path: Path) -> Callable[[], Any]:
    def load():
        import nibabel as nib
        return nib.load(str(dwi_path)).affine
    return load


def resolve_for_inputs(
    *,
    input_dir: Path,
    dwi_path: Path,
    user_phase_encoding: Optional[str] = None,
    user_readout_time: Optional[float] = None,
    has_reverse_b0: bool = False,
) -> AcquisitionMetadata:
    """Resolve with the sidecar ``<input_dir>/dwi.json`` and the affine of the imported DWI."""
    sidecar, digest = read_sidecar(Path(input_dir) / SIDECAR_NAME)
    result = resolve_acquisition(
        user_phase_encoding=user_phase_encoding, user_readout_time=user_readout_time, sidecar=sidecar,
        affine=_affine_loader(Path(dwi_path)), has_reverse_b0=has_reverse_b0, sidecar_sha256=digest,
    )
    if digest is not None and sidecar is None:
        result = _with_warning(result, "The sidecar input/dwi.json is not a readable JSON object and is ignored.")
    return result


def _with_warning(result: AcquisitionMetadata, text: str) -> AcquisitionMetadata:
    from dataclasses import replace
    return replace(result, warnings=(*result.warnings, text))


def resolve_for_subject_root(
    subject_root: Path,
    *,
    user_phase_encoding: Optional[str] = None,
    user_readout_time: Optional[float] = None,
    has_reverse_b0: bool = False,
) -> AcquisitionMetadata:
    """Resolve for the canonical layout ``<subject_root>/input/data.nii.gz`` plus ``input/dwi.json``."""
    input_dir = Path(subject_root) / "input"
    return resolve_for_inputs(
        input_dir=input_dir, dwi_path=input_dir / "data.nii.gz", user_phase_encoding=user_phase_encoding,
        user_readout_time=user_readout_time, has_reverse_b0=has_reverse_b0,
    )


def phase_encoding_block_reason(
    subject_root: Path,
    *,
    user_phase_encoding: Optional[str] = None,
    user_readout_time: Optional[float] = None,
    has_reverse_b0: bool = False,
) -> Optional[str]:
    """The rule-4 message when preprocessing of this subject cannot get a PE direction; else None.

    A subject root that has no ``input/data.nii.gz`` yet (a DICOM folder to be converted, or missing raw data) is not
    judged here: the sidecar does not exist before the import, and a missing input is reported elsewhere.
    """
    root = Path(subject_root)
    if not (root / "input" / "data.nii.gz").is_file():
        return None
    resolved = resolve_for_subject_root(
        root, user_phase_encoding=user_phase_encoding, user_readout_time=user_readout_time,
        has_reverse_b0=has_reverse_b0,
    )
    return resolved.error if resolved.needs_phase_encoding else None
