"""Input validation and lightweight QC for SlicerDSpace subject roots."""

from __future__ import annotations

import importlib.util
import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from shared.io_layout import default_task_output_dir
from shared.brain_mask import brain_mask_config_from_options, inspect_brain_mask_backend, validate_brain_mask
from shared.pipeline import RAW_INPUT_FILES, RawInputBundle, find_subject_root, inspect_subject_directory


ISSUE_LEVELS = ("info", "warning", "error")

PYTHON_DEPENDENCY_GROUPS: dict[str, tuple[str, ...]] = {
    "core": ("nibabel", "SimpleITK"),
    "microstructure": ("dipy", "amico"),
    "noddi": ("amico",),
    "freewater": ("dipy",),
    "tractography": ("dipy", "vtk"),
    "surface": ("torch", "trimesh", "rtree"),
}

COMMAND_DEPENDENCY_GROUPS: dict[str, tuple[str, ...]] = {
    "preprocessing": (
        "dwidenoise",
        "mrdegibbs",
        "dwifslpreproc",
        "dwi2mask",
        "eddy",
        "topup",
    ),
    "tractography": ("tckgen", "tcksift2"),
    "connectome": ("tck2connectome",),
}


@dataclass(frozen=True)
class ValidationIssue:
    level: str
    category: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InputValidationConfig:
    subject_id: str
    input_root: Path
    output_dir: Optional[Path] = None
    t1w_path: Optional[Path] = None
    bids_metadata_path: Optional[Path] = None
    dependency_groups: tuple[str, ...] = ("core",)
    require_phase_encoding: bool = False
    mrtrix_bin_dir: Optional[Path] = None
    write_report: bool = True
    brain_mask_backend: str = "pnl_cnn"
    brain_mask_device: str = "auto"
    pnl_mask_python: Optional[Path] = None
    pnl_mask_script: Optional[Path] = None
    pnl_mask_model_dir: Optional[Path] = None
    pnl_mask_ants_bin_dir: Optional[Path] = None
    pnl_mask_route: str = "auto"
    synthstrip_python: Optional[Path] = None
    postprocess_freesurfer_home: Optional[Path] = None


@dataclass(frozen=True)
class InputValidationResult:
    subject_id: str
    subject_root: Optional[Path]
    report_path: Optional[Path]
    passed: bool
    status: str
    error_count: int
    warning_count: int
    messages: tuple[ValidationIssue, ...]
    metrics: dict[str, Any]


def _check_brain_mask(
    raw: RawInputBundle,
    config: InputValidationConfig,
    messages: list[ValidationIssue],
    metrics: dict[str, Any],
) -> None:
    try:
        metrics["brain_mask_qc"] = validate_brain_mask(raw.dwi_path, raw.mask_path)
    except Exception as exc:
        metrics["brain_mask_qc"] = {"valid": False, "error": str(exc)}
        _add_issue(
            messages, "error", "brain_mask",
            f"Brain mask failed geometry/content QC for selected backend '{config.brain_mask_backend}'.",
            backend=config.brain_mask_backend, error=str(exc),
        )


def _add_issue(
    messages: list[ValidationIssue],
    level: str,
    category: str,
    message: str,
    **details: Any,
) -> None:
    if level not in ISSUE_LEVELS:
        raise ValueError(f"Invalid validation issue level: {level}")
    messages.append(ValidationIssue(level=level, category=category, message=message, details=details))


def _file_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": path.as_posix(), "exists": False}
    stat = path.stat()
    return {
        "path": path.resolve().as_posix(),
        "exists": True,
        "is_file": path.is_file(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_metadata_paths(raw: RawInputBundle, explicit: Optional[Path]) -> list[Path]:
    if explicit is not None:
        return [explicit.expanduser().resolve()]
    input_dir = raw.input_dir
    candidates = [
        input_dir / "data.json",
        input_dir / "dwi.json",
        input_dir / f"{raw.subject_id}_dwi.json",
        raw.subject_root / f"{raw.subject_id}_dwi.json",
    ]
    return candidates


def _parse_bvals(path: Path) -> list[float]:
    return [float(token) for token in path.read_text(encoding="utf-8").split()]


def _parse_bvecs(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append([float(token) for token in stripped.split()])
    return rows


def _normalise_bvec_shape(rows: list[list[float]], n_gradients: int) -> list[list[float]]:
    if len(rows) == 3 and all(len(row) == n_gradients for row in rows):
        return [list(vector) for vector in zip(*rows)]
    if len(rows) == n_gradients and all(len(row) == 3 for row in rows):
        return rows
    raise ValueError(f"Unexpected bvec shape; expected 3x{n_gradients} or {n_gradients}x3.")


def _check_file_layout(raw: RawInputBundle, messages: list[ValidationIssue], metrics: dict[str, Any]) -> None:
    file_metrics: dict[str, Any] = {}
    for key, filename in RAW_INPUT_FILES.items():
        path = raw.input_dir / filename
        info = _file_info(path)
        file_metrics[key] = info
        if not path.is_file():
            _add_issue(messages, "error", "file", f"Missing required input file: input/{filename}", path=path.as_posix())
        elif path.stat().st_size <= 0:
            _add_issue(messages, "error", "file", f"Input file is empty: input/{filename}", path=path.as_posix())
    metrics["files"] = file_metrics


def _check_incomplete_file_layout(
    subject_root: Path,
    messages: list[ValidationIssue],
    metrics: dict[str, Any],
) -> None:
    input_dir = subject_root / "input"
    file_metrics: dict[str, Any] = {}
    for key, filename in RAW_INPUT_FILES.items():
        path = input_dir / filename
        info = _file_info(path)
        file_metrics[key] = info
        if not path.is_file():
            _add_issue(messages, "error", "file", f"Missing required input file: input/{filename}", path=path.as_posix())
        elif path.stat().st_size <= 0:
            _add_issue(messages, "error", "file", f"Input file is empty: input/{filename}", path=path.as_posix())
    metrics["files"] = file_metrics


def _check_metadata(
    raw: RawInputBundle,
    config: InputValidationConfig,
    messages: list[ValidationIssue],
    metrics: dict[str, Any],
) -> None:
    found_path: Optional[Path] = None
    metadata: dict[str, Any] = {}
    for candidate in _candidate_metadata_paths(raw, config.bids_metadata_path):
        if candidate.is_file():
            found_path = candidate
            try:
                metadata = _read_json(candidate)
            except json.JSONDecodeError as exc:
                _add_issue(
                    messages,
                    "error",
                    "metadata",
                    "BIDS metadata JSON exists but cannot be parsed.",
                    path=candidate.as_posix(),
                    error=str(exc),
                )
            break

    metrics["bids_metadata"] = {
        "path": found_path.as_posix() if found_path else None,
        "found": found_path is not None,
        "phase_encoding_direction": metadata.get("PhaseEncodingDirection"),
        "total_readout_time": metadata.get("TotalReadoutTime"),
        "estimated_total_readout_time": metadata.get("EstimatedTotalReadoutTime"),
    }

    if found_path is None:
        level = "error" if config.require_phase_encoding else "warning"
        _add_issue(
            messages,
            level,
            "metadata",
            "No BIDS sidecar was found for phase-encoding/readout-time validation.",
            searched=[path.as_posix() for path in _candidate_metadata_paths(raw, config.bids_metadata_path)],
        )
        return

    phase_direction = metadata.get("PhaseEncodingDirection")
    if phase_direction not in {"i", "i-", "j", "j-", "k", "k-"}:
        level = "error" if config.require_phase_encoding else "warning"
        _add_issue(
            messages,
            level,
            "metadata",
            "PhaseEncodingDirection is missing or not one of i/i-/j/j-/k/k-.",
            path=found_path.as_posix(),
            value=phase_direction,
        )

    readout_time = metadata.get("TotalReadoutTime", metadata.get("EstimatedTotalReadoutTime"))
    if not isinstance(readout_time, (int, float)) or readout_time <= 0:
        level = "error" if config.require_phase_encoding else "warning"
        _add_issue(
            messages,
            level,
            "metadata",
            "TotalReadoutTime or EstimatedTotalReadoutTime is missing or invalid.",
            path=found_path.as_posix(),
            value=readout_time,
        )


def _check_dwi(raw: RawInputBundle, messages: list[ValidationIssue], metrics: dict[str, Any]) -> None:
    try:
        import nibabel as nib
        image = nib.load(raw.dwi_path)
        shape = tuple(int(value) for value in image.shape)
        zooms = tuple(float(value) for value in image.header.get_zooms()[:3])
        finite_geometry = all(math.isfinite(float(value)) for row in image.affine for value in row)
        valid = len(shape) == 4 and all(value > 0 for value in shape)
        valid_geometry = finite_geometry and len(zooms) == 3 and all(math.isfinite(value) and value > 0 for value in zooms)
        metrics['dwi'] = {'valid': valid and valid_geometry, 'shape': list(shape),
                          'volume_count': shape[3] if len(shape) == 4 else None}
        if not valid:
            _add_issue(messages, 'error', 'dwi', 'DWI must be a nonempty 4D image.', shape=list(shape))
        if not valid_geometry:
            _add_issue(messages, 'error', 'dwi', 'DWI has invalid affine or spatial voxel sizes.')
    except Exception as exc:
        metrics['dwi'] = {'valid': False}
        _add_issue(messages, 'error', 'dwi', 'Failed to read the DWI image header.', error=str(exc))


def _check_gradients(raw: RawInputBundle, messages: list[ValidationIssue], metrics: dict[str, Any]) -> None:
    try:
        bvals = _parse_bvals(raw.bval_path)
        bvec_rows = _parse_bvecs(raw.bvec_path)
        bvecs = _normalise_bvec_shape(bvec_rows, len(bvals))
    except Exception as exc:
        _add_issue(messages, "error", "gradient", "Failed to parse bval/bvec gradient files.", error=str(exc))
        metrics["gradients"] = {"valid": False}
        return

    if not bvals:
        _add_issue(messages, "error", "gradient", "bvals file contains no gradients.")
        metrics["gradients"] = {"valid": False}
        return

    negative_bvals = [idx for idx, value in enumerate(bvals) if value < 0]
    nonfinite_bvals = [idx for idx, value in enumerate(bvals) if not math.isfinite(value)]
    bvec_norms = [math.sqrt(sum(component * component for component in vector)) for vector in bvecs]
    nonfinite_bvecs = [
        idx
        for idx, vector in enumerate(bvecs)
        if any(not math.isfinite(component) for component in vector)
    ]
    diffusion_norm_outliers = [
        idx
        for idx, (bval, norm) in enumerate(zip(bvals, bvec_norms))
        if bval > 50 and (norm < 0.5 or norm > 1.5)
    ]
    b0_nonzero_vectors = [
        idx
        for idx, (bval, norm) in enumerate(zip(bvals, bvec_norms))
        if bval <= 50 and norm > 0.2
    ]
    shells = sorted({int(round(value / 100.0) * 100) for value in bvals if math.isfinite(value) and value > 50})
    volume_count = metrics.get('dwi', {}).get('volume_count')
    count_matches = volume_count is None or volume_count == len(bvals)
    finite_norms = [value for value in bvec_norms if math.isfinite(value)]

    metrics["gradients"] = {
        "valid": not (negative_bvals or nonfinite_bvals or nonfinite_bvecs) and count_matches,
        "count": len(bvals),
        "dwi_volume_count": volume_count,
        "b0_count": sum(1 for value in bvals if value <= 50),
        "diffusion_count": sum(1 for value in bvals if value > 50),
        "shells_rounded": shells,
        "bvec_norm_min": min(finite_norms) if finite_norms else None,
        "bvec_norm_max": max(finite_norms) if finite_norms else None,
    }

    if not count_matches:
        _add_issue(messages, 'error', 'gradient', 'Gradient count does not match the DWI volume count.',
                   gradient_count=len(bvals), volume_count=volume_count)

    if negative_bvals:
        _add_issue(messages, "error", "gradient", "bvals contain negative values.", indices=negative_bvals[:20])
    if nonfinite_bvals or nonfinite_bvecs:
        _add_issue(
            messages,
            "error",
            "gradient",
            "Gradient files contain non-finite values.",
            bval_indices=nonfinite_bvals[:20],
            bvec_indices=nonfinite_bvecs[:20],
        )
    if not any(value <= 50 for value in bvals):
        _add_issue(messages, "warning", "gradient", "No b0 volume detected using b<=50 threshold.")
    if not shells:
        _add_issue(messages, "warning", "gradient", "No diffusion-weighted shell detected.")
    if diffusion_norm_outliers:
        _add_issue(
            messages,
            "warning",
            "gradient",
            "Some diffusion b-vectors are not close to unit length.",
            indices=diffusion_norm_outliers[:20],
        )
    if b0_nonzero_vectors:
        _add_issue(
            messages,
            "warning",
            "gradient",
            "Some b0 entries have non-zero b-vectors.",
            indices=b0_nonzero_vectors[:20],
        )


def _check_t1w(path: Optional[Path], messages: list[ValidationIssue], metrics: dict[str, Any]) -> None:
    if path is None:
        metrics["t1w"] = {"provided": False}
        _add_issue(messages, "info", "anatomy", "No T1w image was provided; T1w-dependent branches remain gated.")
        return
    resolved = path.expanduser().resolve()
    metrics["t1w"] = _file_info(resolved)
    if not resolved.is_file():
        _add_issue(messages, "error", "anatomy", "Provided T1w image does not exist.", path=resolved.as_posix())


def _iter_dependencies(groups: Iterable[str]) -> tuple[set[str], set[str], set[str]]:
    unknown: set[str] = set()
    python_modules: set[str] = set()
    commands: set[str] = set()
    for group in groups:
        if group == "all":
            python_modules.update(module for values in PYTHON_DEPENDENCY_GROUPS.values() for module in values)
            commands.update(command for values in COMMAND_DEPENDENCY_GROUPS.values() for command in values)
            continue
        if group in PYTHON_DEPENDENCY_GROUPS:
            python_modules.update(PYTHON_DEPENDENCY_GROUPS[group])
        if group in COMMAND_DEPENDENCY_GROUPS:
            commands.update(COMMAND_DEPENDENCY_GROUPS[group])
        if group not in PYTHON_DEPENDENCY_GROUPS and group not in COMMAND_DEPENDENCY_GROUPS:
            unknown.add(group)
    return unknown, python_modules, commands


def _check_dependencies(
    groups: tuple[str, ...],
    messages: list[ValidationIssue],
    metrics: dict[str, Any],
) -> None:
    unknown, python_modules, commands = _iter_dependencies(groups)
    python_status = {
        module: importlib.util.find_spec(module) is not None
        for module in sorted(python_modules)
    }
    command_status = {
        command: shutil.which(command) is not None
        for command in sorted(commands)
    }
    metrics["dependencies"] = {
        "groups": list(groups),
        "python": python_status,
        "commands": command_status,
        "unknown_groups": sorted(unknown),
    }

    for group in sorted(unknown):
        _add_issue(messages, "warning", "dependency", "Unknown dependency group requested.", group=group)
    for module, available in python_status.items():
        if not available:
            _add_issue(messages, "warning", "dependency", "Python dependency is not importable.", module=module)
    for command, available in command_status.items():
        if not available:
            _add_issue(messages, "warning", "dependency", "External command is not on PATH.", command=command)


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_subject_inputs(config: InputValidationConfig) -> InputValidationResult:
    """Validate a subject-root import and write a lightweight QC JSON report."""

    messages: list[ValidationIssue] = []
    metrics: dict[str, Any] = {}
    subject_root: Optional[Path] = None
    report_path: Optional[Path] = None
    mask_config = brain_mask_config_from_options(config)
    metrics["brain_mask_backend"] = inspect_brain_mask_backend(mask_config)
    metrics["brain_mask_selection"] = {"backend": mask_config.backend}

    if not config.subject_id.strip():
        _add_issue(messages, "error", "file", "subject_id is required.")

    inspected = inspect_subject_directory(config.input_root.expanduser().resolve(), config.subject_id)
    metrics["subject_directory"] = {
        key: value.as_posix() if isinstance(value, Path) else value
        for key, value in inspected.items()
    }

    raw: Optional[RawInputBundle] = None
    if inspected["mode"] == "subject_root":
        subject_root = Path(inspected["subject_root"])
        try:
            raw = find_subject_root(
                config.input_root.expanduser().resolve(),
                config.subject_id,
                mrtrix_bin_dir=config.mrtrix_bin_dir,
                brain_mask_config=mask_config,
            )
        except Exception as exc:
            metrics["brain_mask_qc"] = {"status": "not_run", "error": str(exc)}
            _check_incomplete_file_layout(subject_root, messages, metrics)
            _add_issue(
                messages,
                "error",
                "file",
                f"Brain mask could not be resolved using selected backend '{mask_config.backend}'.",
                backend=mask_config.backend,
                error=str(exc),
            )
        else:
            subject_root = raw.subject_root
            _check_file_layout(raw, messages, metrics)
            _check_dwi(raw, messages, metrics)
            _check_gradients(raw, messages, metrics)
            _check_metadata(raw, config, messages, metrics)
            _check_brain_mask(raw, config, messages, metrics)
    else:
        expanded_input = config.input_root.expanduser().resolve()
        candidate_subject_root = expanded_input
        if not (candidate_subject_root / "input").is_dir() and (expanded_input / config.subject_id).is_dir():
            candidate_subject_root = expanded_input / config.subject_id
        _check_incomplete_file_layout(candidate_subject_root, messages, metrics)
        _add_issue(
            messages,
            "error",
            "file",
            "Could not resolve a subject root with input/data.nii.gz, bvals, and bvecs.",
            input_root=config.input_root.as_posix(),
            mode=inspected["mode"],
        )

    _check_t1w(config.t1w_path, messages, metrics)
    _check_dependencies(config.dependency_groups, messages, metrics)

    error_count = sum(1 for issue in messages if issue.level == "error")
    warning_count = sum(1 for issue in messages if issue.level == "warning")
    status = "passed" if error_count == 0 else "failed"
    passed = error_count == 0

    if config.write_report:
        output_dir = (
            config.output_dir.expanduser().resolve()
            if config.output_dir is not None
            else (default_task_output_dir(subject_root, "validation") if subject_root else config.input_root / "validation")
        )
        report_path = output_dir / f"{config.subject_id or 'unknown'}_input_validation.json"
        _write_report(
            report_path,
            {
                "kind": "input_validation",
                "version": 1,
                "subject_id": config.subject_id,
                "status": status,
                "passed": passed,
                "error_count": error_count,
                "warning_count": warning_count,
                "messages": [
                    {
                        "level": issue.level,
                        "category": issue.category,
                        "message": issue.message,
                        "details": issue.details,
                    }
                    for issue in messages
                ],
                "metrics": metrics,
            },
        )

    return InputValidationResult(
        subject_id=config.subject_id,
        subject_root=subject_root,
        report_path=report_path,
        passed=passed,
        status=status,
        error_count=error_count,
        warning_count=warning_count,
        messages=tuple(messages),
        metrics=metrics,
    )


__all__ = [
    "COMMAND_DEPENDENCY_GROUPS",
    "InputValidationConfig",
    "InputValidationResult",
    "PYTHON_DEPENDENCY_GROUPS",
    "ValidationIssue",
    "validate_subject_inputs",
]
