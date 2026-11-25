"""Shared caching helpers for raw DWI, native DTI, and registered DTI bundles."""

from __future__ import annotations

from shared.cohort_schedule import scheduled_stage

import json
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from uuid import uuid4
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, replace
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Optional

from shared.external_process import (
    copy_file_atomic,
    find_pythonslicer,
    link_or_copy_file,
    merged_startup_environment,
    needs_windows_ascii_staging,
    normalize_slicer_python_launch,
    slicer_runtime_roots,
    subprocess_no_window_kwargs,
    text_subprocess_kwargs,
    windows_ascii_temp_dir,
)
from shared.brain_mask import BrainMaskConfig, generate_brain_mask
from shared.cache_lock import cache_lock

LOGGER = logging.getLogger(__name__)

slicer = sys.modules.get("slicer")
if slicer is None:  # pragma: no cover - depends on the host executable
    # Slicer's private ``python-real`` can find the ``slicer`` package without
    # having initialized the application.  That failed import prints a helper
    # error to stdout before raising, which corrupts JSON-only CLI responses.
    # Reuse the already loaded module in Slicer proper; otherwise treat the
    # optional probe as silent and unavailable.
    try:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            import slicer as _slicer
    except Exception as exc:
        slicer = None  # type: ignore[assignment]
        SLICER_IMPORT_ERROR = exc
    else:
        slicer = _slicer
        SLICER_IMPORT_ERROR = None
else:
    SLICER_IMPORT_ERROR = None

try:  # pragma: no cover - optional runtime dependency
    import SimpleITK as sitk
except ImportError as exc:  # pragma: no cover
    sitk = None  # type: ignore[assignment]
    SIMPLEITK_IMPORT_ERROR = exc
else:
    SIMPLEITK_IMPORT_ERROR = None


RAW_INPUT_FILES = {
    "dwi": "data.nii.gz",
    "bval": "bvals",
    "bvec": "bvecs",
    "mask": "nodif_brain_mask.nii.gz",
}

RAW_REQUIRED_INPUT_KEYS = ("dwi", "bval", "bvec")


@dataclass(frozen=True)
class RawInputBundle:
    subject_root: Path
    subject_id: str
    input_dir: Path
    dwi_path: Path
    bval_path: Path
    bvec_path: Path
    mask_path: Path
    cache_root: Optional[Path] = None

    @property
    def effective_cache_root(self) -> Path:
        return self.cache_root if self.cache_root is not None else self.subject_root / "cache"


@dataclass(frozen=True)
class NativeDTIBundle:
    subject_root: Path
    subject_id: str
    fit_method: str
    bundle_dir: Path
    dwi_nhdr_path: Path
    mask_nhdr_path: Path
    dti_nrrd_path: Path
    b0_nrrd_path: Path
    fa_nrrd_path: Path
    trace_nrrd_path: Path
    mean_diffusivity_nrrd_path: Path
    min_eig_nrrd_path: Path
    mid_eig_nrrd_path: Path
    max_eig_nrrd_path: Path
    ddseg_md_nrrd_path: Path
    ddseg_e1_nrrd_path: Path
    ddseg_e2_nrrd_path: Path
    ddseg_e3_nrrd_path: Path
    fa_nii_path: Path
    trace_nii_path: Path
    mean_diffusivity_nii_path: Path
    min_eig_nii_path: Path
    mid_eig_nii_path: Path
    max_eig_nii_path: Path
    manifest_path: Path
    dec_nrrd_path: Path  # D6: direction-encoded colour FA map


@dataclass(frozen=True)
class RegisteredDTIBundle:
    subject_root: Path
    subject_id: str
    fit_method: str
    atlas_path: Path
    bundle_dir: Path
    fa_reg_path: Path
    trace_reg_path: Path
    mean_diffusivity_reg_path: Path
    min_eig_reg_path: Path
    mid_eig_reg_path: Path
    max_eig_reg_path: Path
    mask_reg_path: Path
    transform_path: Path
    inverse_transform_path: Path
    manifest_path: Path
    scalar_format: str = "nifti"


def _emit(message: str, log: Optional[Callable[[str], None]] = None) -> None:
    LOGGER.info(message)
    if log:
        log(message)


def _lock_log(log: Optional[Callable[[str], None]]) -> Optional[Callable[[str], None]]:
    """H7: route a cache-lock wait notice into the caller's run log; None keeps the lock's stderr default."""
    return (lambda message: _emit("[Shared] " + message, log)) if log else None


def _require_slicer() -> None:
    if SLICER_IMPORT_ERROR is not None or slicer is None:
        raise RuntimeError("Shared DTI bundle generation requires the Slicer Python environment.") from SLICER_IMPORT_ERROR


def _slicer_home_candidates() -> list[Path]:
    candidates: list[Path] = []
    if slicer is not None and hasattr(slicer, "app"):
        try:
            slicer_home = getattr(slicer.app, "slicerHome", "")
            if slicer_home:
                candidates.append(Path(str(slicer_home)).expanduser().resolve())
        except Exception:
            pass
    for env_key in ("SLICER_HOME", "Slicer_HOME", "SLICERDSPACE_SLICER_HOME"):
        env_value = os.environ.get(env_key)
        if env_value:
            candidates.append(Path(env_value).expanduser().resolve())
    # B7 (A-C1 3): the doctor and teem_dti.py read this launcher variable too; its folder is a Slicer runtime root.
    launcher = os.environ.get("SLICERDSPACE_SLICER_EXECUTABLE")
    if launcher:
        candidates.append(Path(launcher).expanduser().resolve().parent)
    executable = Path(sys.executable).expanduser().resolve()
    if executable.stem.lower() in {"python-real", "pythonslicer", "python"}:
        if executable.parent.parent.name == "Python" and executable.parent.parent.parent.name == "lib":
            candidates.append(executable.parent.parent.parent.parent)
        candidates.append(executable.parent.parent)
    if executable.name.startswith("Slicer"):
        candidates.append(executable.parent)
    unique: list[Path] = []
    for candidate in candidates:
        for root in slicer_runtime_roots(candidate):
            if root not in unique:
                unique.append(root)
    return unique


def _resolve_slicer_home() -> Path:
    for candidate in _slicer_home_candidates():
        if (candidate / "lib").is_dir() or (candidate / "bin").is_dir():
            return candidate
    searched = ", ".join(path.as_posix() for path in _slicer_home_candidates()) or "none"
    raise RuntimeError(f"Failed to resolve Slicer home. Candidate paths: {searched}")


def is_raw_subject_root(path: Path) -> bool:
    input_dir = path / "input"
    return input_dir.is_dir() and all((input_dir / RAW_INPUT_FILES[key]).exists() for key in RAW_REQUIRED_INPUT_KEYS)


def _safe_subject_subdir(base_dir: Path, subject_id: str) -> Optional[Path]:
    sid = (subject_id or "").strip()
    if not sid or sid in {".", ".."} or "/" in sid or "\\" in sid or Path(sid).name != sid:
        return None
    return base_dir / sid


def detected_input_format_from_mode(mode: Optional[str]) -> Optional[str]:
    """Map an inspected directory/file ``mode`` to canonical UI format (``nifti``, ``dicom``, ``nrrd``)."""

    normalized = (mode or "").strip().lower()
    if normalized in {"subject_root", "nifti", "legacy_registered"}:
        return "nifti"
    if normalized == "dicom":
        return "dicom"
    if normalized == "nrrd":
        return "nrrd"
    return None


def resolve_subject_root_dir(base_path: Path, subject_id: str = "") -> Path:
    """Resolve ``base_path`` across NIfTI/DICOM/NRRD folders, ``<parent>/<subject_id>``, or direct files."""

    candidate = Path(base_path).expanduser()
    if candidate.is_file():
        expanded_file = candidate.resolve()
        if expanded_file.parent.name == "input" and is_raw_subject_root(expanded_file.parent.parent):
            return expanded_file.parent.parent
        from shared.convertible_input import find_convertible_input

        if find_convertible_input(expanded_file) is not None:
            return candidate
        from shared.dicom_input import _dicom_file

        if _dicom_file(expanded_file):
            return expanded_file.parent
        return candidate

    if not candidate.is_dir():
        return candidate
    expanded = candidate.resolve()
    if expanded.name == "input" and is_raw_subject_root(expanded.parent):
        return expanded.parent
    sid = (subject_id or "").strip()
    if (expanded / "input").is_dir() or (sid and _has_registered_inputs(expanded, sid)):
        return candidate
    from shared.convertible_input import find_convertible_input

    if find_convertible_input(expanded) is not None:
        return candidate
    subject_dir = _safe_subject_subdir(expanded, sid)
    if subject_dir is None or not subject_dir.is_dir():
        return candidate
    if is_raw_subject_root(subject_dir) or (subject_dir / "input").is_dir() or _has_registered_inputs(subject_dir, sid):
        return subject_dir
    conv_sub = find_convertible_input(subject_dir)
    if conv_sub is not None:
        return conv_sub[2]
    from shared.dicom_input import is_dicom_root

    if is_dicom_root(subject_dir):
        return subject_dir
    return candidate


def inspect_subject_directory(base_path: Path, subject_id: str) -> dict[str, Any]:
    """Classify a path across NIfTI, DICOM, and NRRD (directory, parent/<subject_id>, or direct file)."""
    expanded = base_path.expanduser().resolve()
    if expanded.is_file():
        if expanded.parent.name == "input" and is_raw_subject_root(expanded.parent.parent):
            canon_root = expanded.parent.parent
            return {
                "mode": "subject_root",
                "subject_root": canon_root,
                "subject_dir": canon_root,
            }
        from shared.convertible_input import find_convertible_input

        conv_file = find_convertible_input(expanded)
        if conv_file is not None:
            return {
                "mode": conv_file[0],
                "subject_root": conv_file[2],
                "subject_dir": conv_file[2],
                "has_mask": "mask_path" in conv_file[3],
            }
        from shared.dicom_input import _dicom_file

        if _dicom_file(expanded):
            return {"mode": "dicom", "subject_root": expanded.parent, "subject_dir": expanded.parent}
        return {"mode": "missing", "subject_root": None, "subject_dir": expanded.parent}

    if expanded.is_dir() and expanded.name == "input" and is_raw_subject_root(expanded.parent):
        return {
            "mode": "subject_root",
            "subject_root": expanded.parent,
            "subject_dir": expanded.parent,
        }

    subject_dir = _safe_subject_subdir(expanded, subject_id)

    if is_raw_subject_root(expanded):
        return {
            "mode": "subject_root",
            "subject_root": expanded,
            "subject_dir": expanded,
        }

    if subject_dir is not None and subject_dir.is_dir() and is_raw_subject_root(subject_dir):
        return {
            "mode": "subject_root",
            "subject_root": subject_dir,
            "subject_dir": subject_dir,
        }

    if _has_registered_inputs(expanded, subject_id):
        return {
            "mode": "legacy_registered",
            "subject_root": None,
            "subject_dir": expanded,
        }

    if subject_dir is not None and subject_dir.is_dir() and _has_registered_inputs(subject_dir, subject_id):
        return {
            "mode": "legacy_registered",
            "subject_root": None,
            "subject_dir": subject_dir,
        }

    from shared.convertible_input import find_convertible_input
    convertible = find_convertible_input(expanded)
    if convertible is None and subject_dir is not None and subject_dir.is_dir():
        convertible = find_convertible_input(subject_dir)
    if convertible is not None:
        return {
            "mode": convertible[0],
            "subject_root": convertible[2],
            "subject_dir": convertible[2],
            "has_mask": "mask_path" in convertible[3],
        }

    from shared.dicom_input import is_dicom_root
    if subject_dir is not None and subject_dir.is_dir() and is_dicom_root(subject_dir):
        return {"mode": "dicom", "subject_root": subject_dir, "subject_dir": subject_dir}
    if is_dicom_root(expanded):
        return {"mode": "dicom", "subject_root": expanded, "subject_dir": expanded}

    return {
        "mode": "missing",
        "subject_root": None,
        "subject_dir": subject_dir if (subject_dir is not None and subject_dir.is_dir()) else expanded,
    }


def _is_user_supplied_mask(mask_path: Path) -> bool:
    return mask_path.is_file() and not mask_path.with_name("nodif_brain_mask.manifest.json").is_file()


def resolve_raw_brain_mask_path(subject_root: Path, output_root: Optional[Path] = None) -> Path:
    """Resolve native DWI brain mask path, preferring user input then ``preprocessed/``."""
    from shared.io_layout import subject_layout

    resolved_root = Path(subject_root).expanduser().resolve()
    layout = subject_layout(resolved_root, output_root=output_root)
    mask_name = RAW_INPUT_FILES["mask"]
    input_mask = resolved_root / "input" / mask_name
    preproc_mask = layout.preproc / mask_name
    legacy_cached_mask = layout.cache / "brain_mask" / mask_name
    external_workspace = output_root is not None and layout.workspace_root != resolved_root

    if _is_user_supplied_mask(input_mask):
        return input_mask
    if preproc_mask.is_file() or preproc_mask.with_name("nodif_brain_mask.manifest.json").is_file():
        return preproc_mask
    if not external_workspace and (
        input_mask.is_file() or input_mask.with_name("nodif_brain_mask.manifest.json").is_file()
    ):
        return input_mask
    if legacy_cached_mask.is_file():
        return legacy_cached_mask
    return preproc_mask


def _migrate_legacy_generated_mask(src_mask: Path, dest_mask: Path) -> bool:
    """Copy a verified legacy v2 auto-generated mask and receipt into ``dest_mask``."""
    src_receipt = src_mask.with_name("nodif_brain_mask.manifest.json")
    if src_mask.is_symlink() or src_receipt.is_symlink():
        return False
    if not src_mask.is_file() or not src_receipt.is_file():
        return False
    record = json.loads(src_receipt.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("version") != 2:
        return False
    from shared.brain_mask import _sha256

    recorded_sha = (record.get("output") or {}).get("sha256") if isinstance(record.get("output"), dict) else None
    if not recorded_sha or recorded_sha != _sha256(src_mask):
        raise ValueError("Generated brain mask was edited; select provided after review, or explicitly regenerate")

    dest_mask.parent.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest_mask.resolve()
    dest_receipt = dest_mask.with_name("nodif_brain_mask.manifest.json")
    updated = dict(record)
    updated["mask_path"] = str(dest_resolved)
    updated["manifest_path"] = str(dest_resolved.with_name("nodif_brain_mask.manifest.json"))
    output_meta = dict(updated.get("output") or {})
    output_meta["path"] = str(dest_resolved)
    updated["output"] = output_meta

    with tempfile.NamedTemporaryFile(prefix=".brain-mask-", suffix=".nii.gz", dir=dest_mask.parent, delete=False) as tmp:
        staged_mask = Path(tmp.name)
    staged_receipt: Optional[Path] = None
    try:
        shutil.copyfile(src_mask, staged_mask)
        with tempfile.NamedTemporaryFile(
            mode="w", prefix=".mask-record-", suffix=".json", dir=dest_receipt.parent, encoding="utf-8", delete=False
        ) as tmp:
            staged_receipt = Path(tmp.name)
            json.dump(updated, tmp, indent=2, sort_keys=True)
            tmp.write("\n")
        os.replace(staged_mask, dest_mask)
        try:
            os.replace(staged_receipt, dest_receipt)
        except BaseException:
            dest_mask.unlink(missing_ok=True)
            raise
    finally:
        staged_mask.unlink(missing_ok=True)
        if staged_receipt is not None:
            staged_receipt.unlink(missing_ok=True)
    return True


def find_subject_root(
    base_path: Path,
    subject_id: str,
    *,
    brain_mask_config: Optional[BrainMaskConfig] = None,
    generate_missing_mask: bool = True,
    output_root: Optional[Path] = None,
) -> RawInputBundle:
    inspected = inspect_subject_directory(base_path, subject_id)
    if inspected["mode"] != "subject_root":
        raise FileNotFoundError(
            f"Could not resolve a subject root with input/data.nii.gz under '{base_path}' for subject '{subject_id}'."
        )

    from shared.io_layout import subject_layout

    subject_root = Path(inspected["subject_root"]).expanduser().resolve()
    layout = subject_layout(subject_root, output_root=output_root)
    input_dir = subject_root / "input"
    mask_name = RAW_INPUT_FILES["mask"]
    input_mask = input_dir / mask_name
    preproc_mask = layout.preproc / mask_name
    legacy_cached_mask = layout.cache / "brain_mask" / mask_name
    external_workspace = output_root is not None and layout.workspace_root != subject_root

    if _is_user_supplied_mask(input_mask):
        mask_path = input_mask
    else:
        if not external_workspace and input_mask.with_name("nodif_brain_mask.manifest.json").is_file():
            json.loads(input_mask.with_name("nodif_brain_mask.manifest.json").read_text(encoding="utf-8"))
        if generate_missing_mask:
            mask_path = preproc_mask
            if not preproc_mask.exists():
                legacy_sources = [legacy_cached_mask] if external_workspace else [input_mask, legacy_cached_mask]
                for legacy_mask in legacy_sources:
                    if _migrate_legacy_generated_mask(legacy_mask, preproc_mask):
                        break
        else:
            mask_path = resolve_raw_brain_mask_path(subject_root, output_root=output_root)

    generated_record = mask_path.with_name("nodif_brain_mask.manifest.json")
    verify_generated = False
    if generated_record.is_file():
        try:
            verify_generated = json.loads(generated_record.read_text(encoding="utf-8")).get("version") == 2
        except (OSError, ValueError, AttributeError):
            verify_generated = True
    if generate_missing_mask and (not mask_path.exists() or verify_generated):
        generate_brain_mask(
            input_dir / RAW_INPUT_FILES["dwi"],
            input_dir / RAW_INPUT_FILES["bval"],
            input_dir / RAW_INPUT_FILES["bvec"],
            mask_path,
            config=brain_mask_config or BrainMaskConfig(),
        )
    return RawInputBundle(
        subject_root=subject_root,
        subject_id=subject_id,
        input_dir=input_dir,
        dwi_path=input_dir / RAW_INPUT_FILES["dwi"],
        bval_path=input_dir / RAW_INPUT_FILES["bval"],
        bvec_path=input_dir / RAW_INPUT_FILES["bvec"],
        mask_path=mask_path,
        cache_root=layout.cache,
    )


PREPROCESSING_CACHE_KEY_VERSIONS = (5, 6)
NO_CURRENT_PREPROCESSING_MESSAGE = "No current preprocessed DWI for this input; using the raw DWI"


def qc_record_completed(path: Path) -> bool:
    """True when a QC JSON says ``completed`` and no automatic check failed."""
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        checks = record.get("checks") or {}
        return (record.get("status") == "completed" and record.get("automatic_qc_passed") is not False
                and (not isinstance(checks, dict) or not any(value is False for value in checks.values())))
    except Exception:
        return False


@dataclass(frozen=True)
class PreprocessedOutputs:
    """The latest preprocessing outputs that match the current raw input (see ``current_preprocessed_outputs``)."""

    subject_id: str
    preproc_dir: Path
    dwi_path: Path
    bval_path: Path
    bvec_path: Path
    mask_path: Path
    manifest_path: Path
    qc_path: Path
    manifest: dict

    @property
    def cache_key(self) -> dict:
        return self.manifest["cache_key"]

    @property
    def cache_key_version(self) -> int:
        return int(self.cache_key["version"])

    @property
    def route_id(self) -> str:
        """The route id of the outputs; a version-5 manifest has none of the version-6 kind."""
        return str(self.cache_key.get("backend")) if self.cache_key_version >= 6 else "pre-v6"


def _latest_preprocessing_manifest(preproc_dir: Path, subject_id: Optional[str]) -> Optional[tuple[str, Path]]:
    suffix = "_preproc_manifest.json"
    if subject_id is not None:
        path = preproc_dir / f"{subject_id}{suffix}"
        return (subject_id, path) if path.is_file() else None
    try:
        candidates = [path for path in preproc_dir.glob("*" + suffix) if path.is_file()]
    except OSError:
        return None
    if not candidates:
        return None
    latest = max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))
    return latest.name[: -len(suffix)], latest


def current_preprocessed_outputs(
    subject_root: Path,
    output_root: Optional[Path] = None,
    subject_id: Optional[str] = None,
) -> Optional[PreprocessedOutputs]:
    """T1 D1: the preprocessing outputs every stage of a run may use, or None.

    The one decision behind both reuse points (``resolve_preprocessed_input_bundle`` and the orchestrator's
    ``_resolve_analysis_inputs`` / ``_preprocessing_complete``). It needs no preprocessing config. The latest
    preprocessing manifest (or the one of ``subject_id``) must be completed or cached, with a cache key of version 5 or
    6 (so subjects preprocessed before the upgrade keep their outputs); its four output files and its gradient rule
    must be current (the QC JSON is not part of the decision, F4); and its recorded signatures of the raw dwi, bval, bvec and mask files must match the
    current raw input. Anything else gives None, and then every stage uses the raw DWI.
    """

    from shared.io_layout import subject_layout

    root = Path(subject_root).expanduser().resolve()
    layout = subject_layout(root, output_root=Path(output_root) if output_root is not None else None)
    candidates = [
        (layout.stage_cache("00_preprocessing"), layout.preproc),
        (layout.stage_cache("00_preprocessing"), layout.stage_cache("00_preprocessing")),
        (layout.preproc, layout.preproc),
        (layout.preproc_legacy, layout.preproc_legacy),
    ]
    found = None
    manifest_dir = layout.preproc
    files_dir = layout.preproc
    for cand_manifest_dir, cand_files_dir in candidates:
        maybe = _latest_preprocessing_manifest(cand_manifest_dir, subject_id)
        if maybe is not None:
            found_id, _ = maybe
            cand_paths = {
                "dwi": cand_files_dir / f"{found_id}_dwi_final.nii.gz",
                "bval": cand_files_dir / f"{found_id}_dwi_final.bval",
                "bvec": cand_files_dir / f"{found_id}_dwi_final.bvec",
                "mask": cand_files_dir / f"{found_id}_brain_mask_final.nii.gz",
            }
            if all(path.is_file() for path in cand_paths.values()):
                found = maybe
                manifest_dir = cand_manifest_dir
                files_dir = cand_files_dir
                break
    if found is None:
        return None
    found_id, manifest_path = found
    paths = {
        "dwi": files_dir / f"{found_id}_dwi_final.nii.gz",
        "bval": files_dir / f"{found_id}_dwi_final.bval",
        "bvec": files_dir / f"{found_id}_dwi_final.bvec",
        "mask": files_dir / f"{found_id}_brain_mask_final.nii.gz",
    }
    qc_path = manifest_dir / "05_qc" / f"{found_id}_preproc_qc.json"
    if not qc_path.is_file() and (files_dir / "05_qc" / f"{found_id}_preproc_qc.json").is_file():
        qc_path = files_dir / "05_qc" / f"{found_id}_preproc_qc.json"
    if not all(path.is_file() for path in paths.values()):
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict) or manifest.get("status") not in {"completed", "cached"}:
        return None
    from preprocessing.pipeline import preprocessing_gibbs_current, preprocessing_gradients_current
    if not preprocessing_gradients_current(manifest) or not preprocessing_gibbs_current(manifest):
        return None
    cache_key = manifest.get("cache_key")
    if not isinstance(cache_key, dict) or cache_key.get("version") not in PREPROCESSING_CACHE_KEY_VERSIONS:
        return None
    inputs = cache_key.get("inputs")
    if not isinstance(inputs, dict):
        return None
    input_dir = root / "input"
    for key, name in RAW_INPUT_FILES.items():
        record = inputs.get(key)
        expected_path = input_dir / name
        if key == "mask":
            candidates = (input_dir / name, layout.preproc / name, layout.cache / "brain_mask" / name)
            matched = next((cand for cand in candidates if cand.is_file() and _input_record_matches(record, cand)), None)
            if matched is None:
                return None
        elif not _input_record_matches(record, expected_path):
            return None
    return PreprocessedOutputs(
        subject_id=found_id, preproc_dir=files_dir, dwi_path=paths["dwi"], bval_path=paths["bval"],
        bvec_path=paths["bvec"], mask_path=paths["mask"], manifest_path=manifest_path, qc_path=qc_path,
        manifest=manifest,
    )


def _input_record_matches(record: object, path: Path) -> bool:
    """True when a recorded input file names ``path`` (resolved paths equal) with the same size and mtime.

    The producer (``preprocessing.pipeline._file_record``) writes the path as ``as_posix()``, so the stored string can
    differ from ``str(path.resolve())`` (Windows: ``C:/`` against ``C:\\``). Compare resolved paths, not strings.
    """
    if not isinstance(record, dict):
        return False
    try:
        stat = path.stat()
        return (
            Path(str(record.get("path") or "")).resolve() == path.resolve()
            and int(record.get("size", -1)) == stat.st_size
            and int(record.get("mtime_ns", -1)) == stat.st_mtime_ns
        )
    except (OSError, TypeError, ValueError):
        return False


def preprocessed_outputs_usable(
    subject_root: Path,
    output_root: Optional[Path] = None,
    subject_id: Optional[str] = None,
) -> bool:
    """T1 D1: True when ``current_preprocessed_outputs`` finds usable preprocessing outputs for the whole run."""
    return current_preprocessed_outputs(subject_root, output_root, subject_id) is not None


def resolve_preprocessed_input_bundle(
    raw: RawInputBundle,
    *,
    output_root: Optional[Path] = None,
) -> RawInputBundle:
    """Use current preprocessing outputs as an analysis bundle when available.

    Downstream quantitative models must consume the same dMRI, gradients and
    mask emitted by preprocessing. If there are no current preprocessing outputs
    (``preprocessed_outputs_usable``), the canonical input bundle is the explicit
    fallback, for every stage alike.
    """
    from shared.io_layout import subject_layout

    cache_root = subject_layout(raw.subject_root, output_root=output_root).cache if output_root is not None else raw.cache_root
    outputs = current_preprocessed_outputs(raw.subject_root, output_root, raw.subject_id)
    if outputs is None:
        return replace(raw, cache_root=cache_root) if raw.cache_root != cache_root else raw
    return RawInputBundle(
        subject_root=raw.subject_root,
        subject_id=raw.subject_id,
        input_dir=outputs.preproc_dir,
        dwi_path=outputs.dwi_path,
        bval_path=outputs.bval_path,
        bvec_path=outputs.bvec_path,
        mask_path=outputs.mask_path,
        cache_root=cache_root,
    )


def _has_registered_inputs(subject_dir: Path, subject_id: str) -> bool:
    sid = (subject_id or "").strip()
    if not sid:
        return False
    expected = [
        f"{sid}-dti-FractionalAnisotropy-Reg.nii.gz",
        f"{sid}-dti-MinEigenvalue-Reg.nii.gz",
        f"{sid}-dti-MidEigenvalue-Reg.nii.gz",
        f"{sid}-dti-MaxEigenvalue-Reg.nii.gz",
        f"{sid}-dti-Trace-Reg.nii.gz",
        f"{sid}-dti-MeanDiffusivity-Reg.nii.gz",
        f"{sid}-mask-Reg.nii.gz",
        f"{sid}-b0ToAtlasT2.tfm",
    ]
    return subject_dir.is_dir() and all((subject_dir / name).exists() for name in expected)


def _file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _cache_file_signature(path: Path) -> dict[str, Any]:
    return {**_file_signature(path), 'ctime_ns': path.stat().st_ctime_ns}


def _read_manifest(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover - corrupt manifest fallback
        LOGGER.warning("Unable to parse manifest: %s", path)
        return None


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _content_digest(path: str) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else _stream_digest(stream)


@lru_cache(maxsize=128)
def _artifact_digest(path: str, identity: tuple) -> str:
    return _content_digest(path)


def _stream_digest(stream) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(block)
    return digest.hexdigest()


def _artifact_record(path: Path, *, verify_content: bool = False) -> dict:
    stat = path.stat()
    if not path.is_file() or stat.st_size == 0:
        raise ValueError(f'Missing or empty shared artifact: {path}')
    identity = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    digest = (_content_digest(str(path.resolve())) if verify_content
              else _artifact_digest(str(path.resolve()), identity))
    after = path.stat()
    if identity != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError(f'Shared artifact changed while being checked: {path}')
    return dict(path=str(path.resolve()), sha256=digest)


def _artifacts_current(directory: Path, files) -> bool:
    try:
        return _read_manifest(directory / 'artifacts.json') == {
            str(path): _artifact_record(path) for path in files}
    except (OSError, ValueError):
        return False


def _save_artifacts(directory: Path, files) -> None:
    _write_manifest(directory / 'artifacts.json', {str(path): _artifact_record(path) for path in files})


def _gradient_frame_key_pin(raw: RawInputBundle) -> Callable[[str], str]:
    """Item 8 of the gradient-frame fix: keep the old cache keys where the fix changes no output.

    The fix edited nhdr_write.py, teem_dti.py and this file, whose sha256 values are part of the cache keys below.
    For a DWI whose affine the fix provably leaves alone (shared/cache_key_pins.py: axis-aligned, det < 0) each
    current sha256 is mapped to the sha256 of base commit 3c912ad2, so finished runs are reused. Every other DWI
    keeps the real sha256 and is recomputed. A later edit of one of these files changes its sha256; the table then
    no longer matches it and the key changes as it did before the pin.
    """
    try:
        from shared.cache_key_pins import gradient_frame_fix_is_no_op, key_sha256
    except ImportError:
        return lambda digest: digest
    no_op = gradient_frame_fix_is_no_op(raw.dwi_path)
    return lambda digest: key_sha256(digest, no_op)


def _conversion_implementation(pin: Callable[[str], str] = lambda digest: digest) -> dict:
    root = Path(__file__).resolve().parents[1] / 'anatomy/parcellation/conversion'
    return {name: pin(_artifact_record(root / name)['sha256']) for name in ('nhdr_write.py', 'bval_bvec_io.py')}


def _run_cli(cmd: list[str], log: Optional[Callable[[str], None]] = None) -> None:
    _emit(f"[Shared] Running: {' '.join(cmd)}", log)
    exec_cmd = list(cmd)
    if exec_cmd:
        program, norm_args = normalize_slicer_python_launch(exec_cmd[0], exec_cmd[1:])
        exec_cmd = [program, *norm_args]
    process = subprocess.Popen(
        exec_cmd,
        env=merged_startup_environment(build_cli_environment(cmd[0])),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        **subprocess_no_window_kwargs(**text_subprocess_kwargs(bufsize=1)),
    )
    last_nonempty_line = ""

    def on_line(line):
        nonlocal last_nonempty_line
        stripped = line.rstrip()
        if stripped.strip():
            last_nonempty_line = stripped.strip()
        _emit(f"[Shared] {stripped}", log)

    # H8: a surviving grandchild must not keep the stage open after the direct child has exited.
    from shared.line_stream import stream_process_lines
    return_code = stream_process_lines(process, on_line, notice=lambda message: _emit(f"[Shared] {message}", log))
    if return_code != 0:
        cli_name = Path(cmd[0]).name if cmd else "CLI"
        detail = f" ({cli_name}: {last_nonempty_line})" if last_nonempty_line else f" ({cli_name})"
        raise RuntimeError(f"CLI command failed with exit code {return_code}{detail}")


def _get_cli_executable(module_name: str) -> str:
    return resolve_cli_executable(module_name)


def resolve_launcher_executable() -> str:
    _require_slicer()
    if hasattr(slicer, "app"):
        launcher = getattr(slicer.app, "launcherExecutableFilePath", None)
        if callable(launcher):
            launcher = launcher()
        if isinstance(launcher, str) and launcher:
            return Path(launcher).resolve().as_posix()

    slicer_home = _resolve_slicer_home()
    for fallback in (
        slicer_home / "Slicer",
        slicer_home / "Slicer.exe",
        slicer_home / "bin" / "Slicer",
        slicer_home / "bin" / "Slicer.exe",
        slicer_home / "MacOS" / "Slicer",
    ):
        if fallback.is_file():
            return fallback.as_posix()
    raise RuntimeError("Failed to resolve the Slicer launcher executable path.")


def resolve_pythonslicer_executable() -> str:
    """PythonSlicer of the running Slicer, or (outside Slicer, A-C1) of the Slicer the environment names.

    Inside Slicer the order is unchanged. Outside it: ``sys.executable`` when it is PythonSlicer, then
    ``SLICER_HOME`` / ``Slicer_HOME``, then the configured Slicer path (``DTI_SLICER_PATH``, ``SLICER_PATH``).
    """
    if SLICER_IMPORT_ERROR is None and slicer is not None:
        slicer_home = _resolve_slicer_home()
        found = find_pythonslicer(slicer_home)
        if found:
            return found.as_posix()
        raise RuntimeError("Failed to resolve the PythonSlicer executable path.")
    executable = Path(sys.executable)
    if executable.name.lower() in ("pythonslicer", "pythonslicer.exe") and executable.is_file():
        return executable.as_posix()
    homes = list(_slicer_home_candidates())
    for key in ("DTI_SLICER_PATH", "SLICER_PATH"):
        if os.environ.get(key):
            homes.extend(slicer_runtime_roots(Path(os.environ[key]).expanduser().resolve()))
    for home in homes:
        found = find_pythonslicer(home)
        if found:
            return found.as_posix()
    raise RuntimeError(
        "Failed to resolve the PythonSlicer executable outside Slicer. Set SLICER_HOME to the Slicer install "
        "folder (the folder that holds bin/PythonSlicer; on macOS the Slicer.app or its Contents folder), "
        "or run this program with PythonSlicer. Searched: "
        + (", ".join(path.as_posix() for path in homes) or "none"))


NATIVE_TENSOR_SCALARS = ("FractionalAnisotropy", "Trace", "MeanDiffusivity",
                         "MinEigenvalue", "MidEigenvalue", "MaxEigenvalue")


def native_tensor_command(pythonslicer, dwi_header, tensor, baseline, enumeration, scalars, receipt=None,
                          dec=None) -> list[str]:
    """One ``teem_dti.py estimate`` process: tensor, baseline and the scalar maps (operation, path) together.

    No ``--threads``: the child reads ``SLICERDSPACE_STAGE_CPUS`` (set by the cohort scheduler) itself.
    """
    from shared.teem_dti import estimate_command
    return estimate_command(pythonslicer, dwi_header, tensor, baseline, enumeration, scalars, receipt, dec=dec)


def build_cli_environment(cli_executable: str) -> dict[str, str]:
    env = os.environ.copy()
    requested = env.get("OMP_NUM_THREADS", "")
    if (not env.get("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS")
            and requested.isdecimal() and int(requested) > 0):
        # Propagate an explicit budget to ITK consumers only. SlicerDMRI's
        # VTK tensor estimator does not use ITK's thread setting.
        env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = requested
    cli_path = Path(cli_executable).resolve()
    extra_lib_dirs: list[str] = []

    cli_dir = cli_path.parent
    if cli_dir.is_dir():
        extra_lib_dirs.append(str(cli_dir))

    qt_loadable_dir = cli_dir.parent / "qt-loadable-modules"
    if qt_loadable_dir.is_dir():
        extra_lib_dirs.append(str(qt_loadable_dir))

    if sys.platform == "darwin":
        # macOS extension CLIs (UKFTractography for Slicer 5.12) name Slicer's
        # core ITK/VTK libraries by build-machine paths, and the launcher
        # exports no DYLD_LIBRARY_PATH; point dyld at the app's lib/Slicer-*.
        # C5: stop every ancestor walk at the .app/Contents folder, and use the
        # running Slicer's lib only for a CLI that belongs to that Slicer
        # (inside its root, or in a Slicer-managed slicer.org/Extensions-* folder).
        def bounded_parents(path: Path) -> list[Path]:
            result: list[Path] = []
            for parent in path.parents:
                result.append(parent)
                if parent.name == "Contents" and parent.parent.suffix == ".app":
                    break
            return result

        if (cli_dir.parent / "libUKFBase.dylib").is_file():
            extra_lib_dirs.append(str(cli_dir.parent))
        module_root = Path(__file__).resolve().parents[1]
        installed_ext_root = (
            module_root.parents[2]
            if module_root.name == "qt-scripted-modules" and len(module_root.parents) >= 3
            else None
        )
        runtime_root = next((root for root in bounded_parents(Path(sys.executable).resolve())
                             if any(root.glob("lib/Slicer-*"))), None)
        belongs_to_slicer = (
            (runtime_root is not None and runtime_root in cli_path.parents)
            or (installed_ext_root is not None and installed_ext_root in cli_path.parents)
            or any(parent.parent.name == "slicer.org" and parent.name.startswith("Extensions-")
                   for parent in cli_path.parents)
        )
        for root in (*bounded_parents(cli_path), *((runtime_root,) if belongs_to_slicer and runtime_root else ())):
            extra_lib_dirs.extend(str(path) for path in sorted(root.glob("lib/Slicer-*")) if path.is_dir())

    library_var = "PATH" if sys.platform == "win32" else (
        "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH")
    existing = [entry for entry in env.get(library_var, "").split(os.pathsep) if entry]
    merged: list[str] = []
    for entry in [*extra_lib_dirs, *existing]:
        if entry not in merged:
            merged.append(entry)
    env[library_var] = os.pathsep.join(merged)
    return env


@lru_cache(maxsize=1)
def _cli_search_dirs() -> tuple[Path, ...]:
    _require_slicer()
    slicer_home = _resolve_slicer_home()
    candidates: list[Path] = []
    patterns = [
        "lib/Slicer-*/cli-modules",
        "slicer.org/Extensions-*/*/lib/Slicer-*/cli-modules",
        # A stock macOS .app keeps its extensions in Contents/Extensions-<revision>.
        "Extensions-*/*/lib/Slicer-*/cli-modules",
    ]
    for pattern in patterns:
        for path in sorted(slicer_home.glob(pattern)):
            if path.is_dir():
                candidates.append(path.resolve())
    if sys.platform == "darwin":
        # C1: accept only extension builds made for this Slicer's own lib/Slicer-X.Y,
        # newest revision first; a 5.6 or 5.10 CLI must not serve a 5.12 runtime.
        home_dir = Path(os.environ["HOME"]) if os.environ.get("HOME") else Path.home()
        user_extensions = home_dir / "Library/Application Support/slicer.org"
        own_versions = {path.name for path in slicer_home.glob("lib/Slicer-*") if path.is_dir()}

        def revision(path: Path) -> int:
            digits = path.parents[3].name.rpartition("-")[2]
            return int(digits) if digits.isdecimal() else -1

        user_dirs = [path for path in user_extensions.glob("Extensions-*/*/lib/Slicer-*/cli-modules")
                     if path.is_dir() and path.parent.name in own_versions]
        for path in sorted(user_dirs, key=lambda item: (-revision(item), item.as_posix())):
            if path.resolve() not in candidates:
                candidates.append(path.resolve())
    # Extension Manager/development paths need not live under SLICER_HOME.
    # PythonQt exposes searchPaths as a property on some Slicer versions.
    try:
        factory = slicer.app.moduleManager().factoryManager()
        paths = getattr(factory, 'searchPaths', ())
        paths = paths() if callable(paths) else paths
        for value in paths:
            path = Path(str(value))
            if path.is_dir() and path.resolve() not in candidates:
                candidates.append(path.resolve())
    except (AttributeError, TypeError, RuntimeError):
        pass
    return tuple(candidates)


def resolve_cli_executable(module_name: str) -> str:
    _require_slicer()
    modules = getattr(slicer, "modules", None)
    module = getattr(modules, module_name, None) if modules is not None else None
    if module is not None:
        logic = module.cliModuleLogic()
        if logic is not None:
            getter = getattr(logic, "GetCLIExecutable", None)
            if callable(getter):
                cli_exe = getter()
                if cli_exe:
                    return cli_exe
        path = getattr(module, 'path', None)
        path = path() if callable(path) else path
        if path and Path(str(path)).is_file() and Path(str(path)).suffix.lower() not in {'.xml', '.so', '.dll', '.dylib'}:
            return str(path)

    target = module_name.lower()
    windows_executable_suffixes = {".exe", ".bat", ".cmd"}
    for cli_dir in _cli_search_dirs():
        for candidate in cli_dir.iterdir():
            candidate_name = candidate.name.lower()
            is_executable_name = candidate_name == target or (
                candidate.suffix.lower() in windows_executable_suffixes and candidate.stem.lower() == target
            )
            if candidate.is_file() and is_executable_name:
                return candidate.as_posix()

    raise RuntimeError(
        f"Required CLI module is not available: {module_name}. Searched directories: "
        + ", ".join(path.as_posix() for path in _cli_search_dirs())
    )


def _rewrite_nhdr_data_file(nhdr_path: Path, data_path: Path) -> None:
    lines = nhdr_path.read_text(encoding="utf-8").splitlines()
    updated = []
    replaced = False
    for line in lines:
        if line.startswith("data file:"):
            updated.append(f"data file: {data_path.resolve().as_posix()}")
            replaced = True
        else:
            updated.append(line)
    if not replaced:
        updated.append(f"data file: {data_path.resolve().as_posix()}")
    nhdr_path.write_text("\n".join(updated) + "\n", encoding="utf-8")


def input_source_cache_prefix(raw: RawInputBundle) -> str:
    """Cache-key prefix separating features of the raw input from preprocessed DWI.

    Raw inputs keep the historical key; a bundle resolved from 01_preproc gets
    its own ``preproc_`` subdirectory so alternating raw/preprocessed runs do
    not overwrite (thrash) one another's cached DTI features.
    """

    try:
        is_raw = Path(raw.input_dir).resolve() == (Path(raw.subject_root) / "input").resolve()
    except OSError:
        is_raw = True
    return "" if is_raw else "preproc_"


def ensure_diffusion_headers(raw: RawInputBundle, *, log=None) -> tuple[Path, Path]:
    """Convert once for LS/WLS consumers; scaled payloads belong to this cache."""
    from tempfile import TemporaryDirectory
    from anatomy.parcellation.conversion.nhdr_write import nhdr_write

    directory = raw.effective_cache_root / 'diffusion_inputs' / (input_source_cache_prefix(raw) + 'native')
    headers = (directory / 'DWI.nhdr', directory / 'Mask.nhdr')
    manifest = directory / 'manifest.json'
    signature = dict(kind='diffusion_headers', version=2, subject_id=raw.subject_id,
        implementation=_conversion_implementation(_gradient_frame_key_pin(raw)),
        sources={key: _cache_file_signature(getattr(raw, key + '_path')) for key in ('dwi', 'bval', 'bvec', 'mask')})
    with cache_lock(directory / '.lock', log=_lock_log(log)):
        signature['sources'] = {key: _cache_file_signature(getattr(raw, key + '_path'))
                                for key in ('dwi', 'bval', 'bvec', 'mask')}
        if _read_manifest(manifest) == signature:
            receipt = _read_manifest(directory / 'artifacts.json') or {}
            required = set(map(str, headers))
            allowed = required | {str(directory / name) for name in ('DWI.raw.gz', 'Mask.raw.gz')}
            if required <= set(receipt) <= allowed and _artifacts_current(directory, [Path(path) for path in receipt]):
                _emit(f'[Shared] Reusing DWI/mask headers: {directory}', log)
                return headers
        manifest.unlink(missing_ok=True)
        files = list(headers)
        with TemporaryDirectory(prefix='.headers-', dir=directory) as temporary:
            for source, bval, bvec, target in (
                    (raw.dwi_path, raw.bval_path, raw.bvec_path, headers[0]),
                    (raw.mask_path, None, None, headers[1])):
                staged = Path(temporary) / target.name
                nhdr_write(str(source), str(bval) if bval else None, str(bvec) if bvec else None, str(staged))
                scaled = staged.with_suffix('.raw.gz')
                if scaled.is_file():
                    payload = directory / scaled.name
                    shutil.copy2(scaled, payload)
                    files.append(payload)
                else:
                    payload = source
                _rewrite_nhdr_data_file(staged, payload)
                staged.replace(target)
        current = {key: _cache_file_signature(getattr(raw, key + '_path')) for key in ('dwi', 'bval', 'bvec', 'mask')}
        if current != signature['sources']:
            raise RuntimeError('Inputs changed during shared header conversion')
        _save_artifacts(directory, files)
        _write_manifest(manifest, signature)
        _emit(f'[Shared] Prepared DWI/mask headers once: {directory}', log)
        return headers


def _build_native_bundle(raw: RawInputBundle, fit_method: str) -> NativeDTIBundle:
    fit_key = fit_method.lower()
    bundle_dir = raw.effective_cache_root / "native_dti" / f"{input_source_cache_prefix(raw)}{fit_key}"
    subject = raw.subject_id
    return NativeDTIBundle(
        subject_root=raw.subject_root,
        subject_id=subject,
        fit_method=fit_key,
        bundle_dir=bundle_dir,
        dwi_nhdr_path=bundle_dir / "DWI.nhdr",
        mask_nhdr_path=bundle_dir / "Mask.nhdr",
        dti_nrrd_path=bundle_dir / "DTI.nrrd",
        b0_nrrd_path=bundle_dir / "b0.nrrd",
        fa_nrrd_path=bundle_dir / "DTI_FA.nrrd",
        trace_nrrd_path=bundle_dir / "DTI_Trace.nrrd",
        mean_diffusivity_nrrd_path=bundle_dir / "DTI_MeanDiffusivity.nrrd",
        min_eig_nrrd_path=bundle_dir / "DTI_MinEigenvalue.nrrd",
        mid_eig_nrrd_path=bundle_dir / "DTI_MidEigenvalue.nrrd",
        max_eig_nrrd_path=bundle_dir / "DTI_MaxEigenvalue.nrrd",
        ddseg_md_nrrd_path=bundle_dir / "DTI_MD.nrrd",
        ddseg_e1_nrrd_path=bundle_dir / "DTI_E1.nrrd",
        ddseg_e2_nrrd_path=bundle_dir / "DTI_E2.nrrd",
        ddseg_e3_nrrd_path=bundle_dir / "DTI_E3.nrrd",
        fa_nii_path=bundle_dir / f"{subject}-dti-FractionalAnisotropy.nii.gz",
        trace_nii_path=bundle_dir / f"{subject}-dti-Trace.nii.gz",
        mean_diffusivity_nii_path=bundle_dir / f"{subject}-dti-MeanDiffusivity.nii.gz",
        min_eig_nii_path=bundle_dir / f"{subject}-dti-MinEigenvalue.nii.gz",
        mid_eig_nii_path=bundle_dir / f"{subject}-dti-MidEigenvalue.nii.gz",
        max_eig_nii_path=bundle_dir / f"{subject}-dti-MaxEigenvalue.nii.gz",
        manifest_path=bundle_dir / "manifest.json",
        dec_nrrd_path=bundle_dir / f"{subject}-dti-DEC.nrrd",
    )


def _native_manifest(raw: RawInputBundle, bundle: NativeDTIBundle) -> dict[str, Any]:
    pin = _gradient_frame_key_pin(raw)
    return {
        "kind": "native_dti_bundle",
        "version": 4,
        "subject_id": raw.subject_id,
        "fit_method": bundle.fit_method,
        "conversion_implementation": _conversion_implementation(pin),
        "preparation_implementation": pin(_artifact_record(Path(__file__))['sha256']),
        "tensor_implementation": pin(_artifact_record(Path(__file__).with_name('teem_dti.py'))['sha256']),
        "source_files": {
            "dwi": _cache_file_signature(raw.dwi_path),
            "bval": _cache_file_signature(raw.bval_path),
            "bvec": _cache_file_signature(raw.bvec_path),
            "mask": _cache_file_signature(raw.mask_path),
        },
    }


def _native_outputs_complete(bundle: NativeDTIBundle) -> bool:
    required = [
        bundle.dwi_nhdr_path,
        bundle.mask_nhdr_path,
        bundle.dti_nrrd_path,
        bundle.b0_nrrd_path,
        bundle.fa_nrrd_path,
        bundle.trace_nrrd_path,
        bundle.mean_diffusivity_nrrd_path,
        bundle.min_eig_nrrd_path,
        bundle.mid_eig_nrrd_path,
        bundle.max_eig_nrrd_path,
        bundle.ddseg_md_nrrd_path,
        bundle.ddseg_e1_nrrd_path,
        bundle.ddseg_e2_nrrd_path,
        bundle.ddseg_e3_nrrd_path,
        bundle.fa_nii_path,
        bundle.trace_nii_path,
        bundle.mean_diffusivity_nii_path,
        bundle.min_eig_nii_path,
        bundle.mid_eig_nii_path,
        bundle.max_eig_nii_path,
        bundle.dec_nrrd_path,
    ]
    return _artifacts_current(bundle.bundle_dir, required)


def _convert_nrrd_to_nifti(in_path: Path, out_path: Path) -> None:
    from anatomy.parcellation.conversion.nifti_write import nifti_write

    prefix = out_path.as_posix().replace(".nii.gz", "")
    nifti_write(in_path.as_posix(), prefix)


def _is_native_dti_cached(raw: RawInputBundle, fit_method: str = "wls", reuse: bool = True) -> bool:
    if not reuse:
        return False
    try:
        fit_key = str(fit_method).lower()
        if fit_key not in {"ls", "wls"}:
            return False
        bundle = _build_native_bundle(raw, fit_key)
        if not bundle.manifest_path.is_file():
            return False
        expected_manifest = _native_manifest(raw, bundle)
        actual_manifest = _read_manifest(bundle.manifest_path)
        return actual_manifest == expected_manifest and _native_outputs_complete(bundle)
    except Exception:
        return False


def _is_registered_dti_cached(
    raw: RawInputBundle,
    native: NativeDTIBundle,
    atlas_t2: Path,
    fit_method: str = "wls",
    reuse: bool = True,
    scalar_format: str = "nifti",
) -> bool:
    if not reuse:
        return False
    try:
        bundle = _build_registered_bundle(raw, fit_method, atlas_t2, scalar_format)
        if not bundle.manifest_path.is_file():
            return False
        expected_manifest = _registered_manifest(raw, native, bundle)
        actual_manifest = _read_manifest(bundle.manifest_path)
        return actual_manifest == expected_manifest and _registered_outputs_complete(bundle)
    except Exception:
        return False


@scheduled_stage('native_dti')
def ensure_native_dti_bundle(
    raw: RawInputBundle,
    fit_method: str = "wls",
    reuse: bool = True,
    log: Optional[Callable[[str], None]] = None,
) -> NativeDTIBundle:
    """Create or reuse native-space DTI features under cache/native_dti."""
    if fit_method.lower() not in ('ls', 'wls'):
        raise ValueError(f'Unsupported fit method: {fit_method}')
    bundle = _build_native_bundle(raw, fit_method)
    with cache_lock(bundle.bundle_dir / '.lock', log=_lock_log(log)):
        return _ensure_native_dti_bundle(raw, fit_method, reuse, log)


def _ensure_native_dti_bundle(raw, fit_method, reuse, log):
    fit_key = fit_method.lower()
    if fit_key not in {"ls", "wls"}:
        raise ValueError(f"Unsupported fit method: {fit_method}")

    headers = ensure_diffusion_headers(raw, log=log)
    bundle = _build_native_bundle(raw, fit_key)
    expected_manifest = _native_manifest(raw, bundle)
    actual_manifest = _read_manifest(bundle.manifest_path)
    if reuse and actual_manifest == expected_manifest and _native_outputs_complete(bundle):
        from shared.cohort_schedule import mark_reused
        mark_reused('native_dti')
        _emit(f"[Shared] Reusing native DTI bundle: {bundle.bundle_dir}", log)
        return bundle

    _require_slicer()
    _emit(f"[Shared] Building native DTI bundle ({fit_key}) in {bundle.bundle_dir}", log)
    started = time.monotonic()
    bundle.bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle.manifest_path.unlink(missing_ok=True)
    shutil.copy2(headers[0], bundle.dwi_nhdr_path)
    shutil.copy2(headers[1], bundle.mask_nhdr_path)

    receipt_path = bundle.bundle_dir / "native_dti_receipt.json"
    receipt_path.unlink(missing_ok=True)
    scalar_paths = {
        "FractionalAnisotropy": bundle.fa_nrrd_path, "Trace": bundle.trace_nrrd_path,
        "MeanDiffusivity": bundle.mean_diffusivity_nrrd_path, "MinEigenvalue": bundle.min_eig_nrrd_path,
        "MidEigenvalue": bundle.mid_eig_nrrd_path, "MaxEigenvalue": bundle.max_eig_nrrd_path,
    }
    scalar_outputs = [(name, scalar_paths[name]) for name in NATIVE_TENSOR_SCALARS]  # N5: one list of scalar names
    _run_cli(
        native_tensor_command(
            resolve_pythonslicer_executable(),
            bundle.dwi_nhdr_path, bundle.dti_nrrd_path, bundle.b0_nrrd_path,
            "WLS" if fit_key == "wls" else "LS", scalar_outputs, receipt_path,
            dec=bundle.dec_nrrd_path),
        log,
    )
    tensor_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    shutil.copy2(bundle.trace_nrrd_path, bundle.ddseg_md_nrrd_path)
    shutil.copy2(bundle.max_eig_nrrd_path, bundle.ddseg_e1_nrrd_path)
    shutil.copy2(bundle.mid_eig_nrrd_path, bundle.ddseg_e2_nrrd_path)
    shutil.copy2(bundle.min_eig_nrrd_path, bundle.ddseg_e3_nrrd_path)

    _convert_nrrd_to_nifti(bundle.fa_nrrd_path, bundle.fa_nii_path)
    _convert_nrrd_to_nifti(bundle.trace_nrrd_path, bundle.trace_nii_path)
    _convert_nrrd_to_nifti(bundle.mean_diffusivity_nrrd_path, bundle.mean_diffusivity_nii_path)
    _convert_nrrd_to_nifti(bundle.min_eig_nrrd_path, bundle.min_eig_nii_path)
    _convert_nrrd_to_nifti(bundle.mid_eig_nrrd_path, bundle.mid_eig_nii_path)
    _convert_nrrd_to_nifti(bundle.max_eig_nrrd_path, bundle.max_eig_nii_path)

    if expected_manifest != _native_manifest(raw, bundle):
        raise RuntimeError('Inputs changed during shared native DTI preparation')
    files = [getattr(bundle, name) for name in bundle.__dataclass_fields__
             if name.endswith('_path') and name != 'manifest_path']
    _save_artifacts(bundle.bundle_dir, files)
    _write_manifest(bundle.manifest_path, expected_manifest)
    _write_manifest(bundle.bundle_dir / 'native_dti_timing.json', {
        'kind': 'native_dti_timing', 'status': 'completed', 'subject_id': raw.subject_id,
        'runtime_seconds': time.monotonic() - started, 'fit_method': fit_key,
        'source_dwi': str(raw.dwi_path), 'cache_reused': False,
        'tensor_fit': tensor_receipt,
        'timing_scope': 'Fresh native Teem tensor fit and six scalar maps in one process (plus the DEC colour map), compatibility copies and NIfTI conversion; excludes registration and later ROI statistics',
    })
    return bundle


def _atlas_key(atlas_path: Path) -> str:
    stem = atlas_path.name
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    else:
        stem = atlas_path.stem
    return stem.replace(" ", "_")


def _build_registered_bundle(raw: RawInputBundle, fit_method: str, atlas_path: Path,
                             scalar_format: str = 'nifti') -> RegisteredDTIBundle:
    fit_key = fit_method.lower()
    bundle_dir = (
        raw.effective_cache_root / "registered_dti" / f"{input_source_cache_prefix(raw)}{_atlas_key(atlas_path)}__{fit_key}"
    )
    if scalar_format == 'nrrd':
        bundle_dir = bundle_dir.with_name(bundle_dir.name + '__nrrd')
    subject = raw.subject_id
    return RegisteredDTIBundle(
        subject_root=raw.subject_root,
        subject_id=subject,
        fit_method=fit_key,
        atlas_path=atlas_path,
        bundle_dir=bundle_dir,
        fa_reg_path=bundle_dir / f"{subject}-dti-FractionalAnisotropy-Reg.nii.gz",
        trace_reg_path=bundle_dir / f"{subject}-dti-Trace-Reg.nii.gz",
        mean_diffusivity_reg_path=bundle_dir / f"{subject}-dti-MeanDiffusivity-Reg.nii.gz",
        min_eig_reg_path=bundle_dir / f"{subject}-dti-MinEigenvalue-Reg.nii.gz",
        mid_eig_reg_path=bundle_dir / f"{subject}-dti-MidEigenvalue-Reg.nii.gz",
        max_eig_reg_path=bundle_dir / f"{subject}-dti-MaxEigenvalue-Reg.nii.gz",
        mask_reg_path=bundle_dir / f"{subject}-mask-Reg.nii.gz",
        transform_path=bundle_dir / f"{subject}-b0ToAtlasT2.tfm",
        inverse_transform_path=bundle_dir / f"{subject}-b0ToAtlasT2_Inverse.h5",
        manifest_path=bundle_dir / "manifest.json",
        scalar_format=scalar_format,
    )


def _registered_manifest(raw: RawInputBundle, native: NativeDTIBundle, registered: RegisteredDTIBundle) -> dict[str, Any]:
    return {
        "kind": "registered_dti_bundle",
        "version": 2,
        "subject_id": raw.subject_id,
        "fit_method": registered.fit_method,
        "scalar_format": registered.scalar_format,
        "implementation": _gradient_frame_key_pin(raw)(_artifact_record(Path(__file__))['sha256']),
        "atlas": _cache_file_signature(registered.atlas_path),
        "native_bundle": {
            "dir": str(native.bundle_dir.resolve()),
            "manifest": _read_manifest(native.manifest_path),
            "artifacts": _read_manifest(native.bundle_dir / 'artifacts.json'),
        },
    }


def _registered_outputs_complete(bundle: RegisteredDTIBundle) -> bool:
    required = [
        bundle.fa_reg_path,
        bundle.trace_reg_path,
        bundle.mean_diffusivity_reg_path,
        bundle.min_eig_reg_path,
        bundle.mid_eig_reg_path,
        bundle.max_eig_reg_path,
        bundle.mask_reg_path,
        bundle.transform_path,
        bundle.inverse_transform_path,
    ]
    return _artifacts_current(bundle.bundle_dir, required)


def _path_image_suffix(path: Path) -> str:
    lower = path.name.lower()
    for compound in (".nii.gz", ".mif.gz", ".raw.gz", ".tar.gz"):
        if lower.endswith(compound):
            return path.name[-len(compound):]
    return path.suffix


def _write_inverse_transform(forward_path: Path, inverse_path: Path) -> None:
    if SIMPLEITK_IMPORT_ERROR is not None or sitk is None:
        return
    if needs_windows_ascii_staging(forward_path, inverse_path):
        with windows_ascii_temp_dir(prefix="sitk_") as stage_dir:
            staged_in = link_or_copy_file(forward_path, stage_dir / f"forward{_path_image_suffix(forward_path)}")
            staged_out = stage_dir / f"inverse{_path_image_suffix(inverse_path)}"
            transform = sitk.ReadTransform(str(staged_in))
            inverse = transform.GetInverse()
            sitk.WriteTransform(inverse, str(staged_out))
            copy_file_atomic(staged_out, inverse_path)
        return
    transform = sitk.ReadTransform(str(forward_path))
    inverse = transform.GetInverse()
    sitk.WriteTransform(inverse, str(inverse_path))


def ensure_registered_dti_bundle(
    raw: RawInputBundle,
    atlas_t2: Path,
    fit_method: str = "wls",
    reuse: bool = True,
    log: Optional[Callable[[str], None]] = None,
    *,
    scalar_format: str = 'nifti',
) -> RegisteredDTIBundle:
    """Create or reuse registered DTI features under cache/registered_dti.

    S1.2: the native bundle comes first and is admitted as its own ``native_dti`` stage (or reused). The
    ``registered_dti`` admission then covers registration and resampling only. Lock order: the native cache
    lock is taken and released inside ``ensure_native_dti_bundle`` before the registered cache lock is
    taken, so no process holds the registered lock while it waits for the native fit.
    """
    if scalar_format not in ('nifti', 'nrrd'):
        raise ValueError('Registered scalar format must be nifti or nrrd')
    if fit_method.lower() not in ('ls', 'wls'):
        raise ValueError(f'Unsupported fit method: {fit_method}')
    atlas_t2 = atlas_t2.expanduser().resolve()
    if not atlas_t2.is_file():
        raise FileNotFoundError(f"Atlas T2 not found: {atlas_t2}")
    native = ensure_native_dti_bundle(raw, fit_method=fit_method, reuse=True, log=log)
    return _registered_dti_stage(raw, native, atlas_t2, fit_method, reuse, log, scalar_format)


@scheduled_stage('registered_dti')
def _registered_dti_stage(raw, native, atlas_t2, fit_method, reuse, log, scalar_format):
    bundle = _build_registered_bundle(raw, fit_method, atlas_t2, scalar_format)
    with cache_lock(bundle.bundle_dir / '.lock', log=_lock_log(log)):
        return _ensure_registered_dti_bundle(raw, native, atlas_t2, fit_method, reuse, log, scalar_format)


def _ensure_registered_dti_bundle(raw, native, atlas_t2, fit_method, reuse, log, scalar_format):
    bundle = _build_registered_bundle(raw, fit_method, atlas_t2, scalar_format)
    expected_manifest = _registered_manifest(raw, native, bundle)
    actual_manifest = _read_manifest(bundle.manifest_path)
    if reuse and actual_manifest == expected_manifest and _registered_outputs_complete(bundle):
        _emit(f"[Shared] Reusing registered DTI bundle: {bundle.bundle_dir}", log)
        from shared.cohort_schedule import mark_reused
        mark_reused('registered_dti')
        if not bundle.inverse_transform_path.exists():
            _write_inverse_transform(bundle.transform_path, bundle.inverse_transform_path)
        return bundle

    _require_slicer()
    _emit(f"[Shared] Building registered DTI bundle ({bundle.fit_method}) in {bundle.bundle_dir}", log)
    bundle.bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle.manifest_path.unlink(missing_ok=True)

    brainsfit_cli = _get_cli_executable("brainsfit")
    resample_cli = _get_cli_executable("resamplescalarvectordwivolume")

    resample_jobs = [
        (native.fa_nii_path, bundle.fa_reg_path, "linear"),
        (native.trace_nii_path, bundle.trace_reg_path, "linear"),
        (native.mean_diffusivity_nii_path, bundle.mean_diffusivity_reg_path, "linear"),
        (native.min_eig_nii_path, bundle.min_eig_reg_path, "linear"),
        (native.mid_eig_nii_path, bundle.mid_eig_reg_path, "linear"),
        (native.max_eig_nii_path, bundle.max_eig_reg_path, "linear"),
        (raw.mask_path, bundle.mask_reg_path, "nn"),
    ]
    if scalar_format == 'nrrd':
        originals = dict(zip((native.fa_nii_path, native.trace_nii_path, native.mean_diffusivity_nii_path,
                              native.min_eig_nii_path, native.mid_eig_nii_path, native.max_eig_nii_path),
                             (native.fa_nrrd_path, native.trace_nrrd_path, native.mean_diffusivity_nrrd_path,
                              native.min_eig_nrrd_path, native.mid_eig_nrrd_path, native.max_eig_nrrd_path)))
        resample_jobs = [(originals.get(source, source), target, mode) for source, target, mode in resample_jobs]

    if needs_windows_ascii_staging(
        atlas_t2,
        native.b0_nrrd_path,
        bundle.transform_path,
        *(path for pair in resample_jobs for path in pair[:2]),
    ):
        with windows_ascii_temp_dir(prefix="reg_dti_") as stage_dir:
            staged_atlas = link_or_copy_file(atlas_t2, stage_dir / f"atlas{_path_image_suffix(atlas_t2)}")
            staged_b0 = link_or_copy_file(native.b0_nrrd_path, stage_dir / f"b0{_path_image_suffix(native.b0_nrrd_path)}")
            staged_tx = stage_dir / f"transform{_path_image_suffix(bundle.transform_path)}"
            _run_cli(
                [
                    brainsfit_cli,
                    "--fixedVolume",
                    staged_atlas.as_posix(),
                    "--movingVolume",
                    staged_b0.as_posix(),
                    "--linearTransform",
                    staged_tx.as_posix(),
                    "--useRigid",
                    "--useAffine",
                ],
                log,
            )
            copy_file_atomic(staged_tx, bundle.transform_path)
            for idx, (input_path, output_path, interpolation) in enumerate(resample_jobs):
                staged_in = link_or_copy_file(input_path, stage_dir / f"in_{idx}{_path_image_suffix(input_path)}")
                staged_out = stage_dir / f"out_{idx}{_path_image_suffix(output_path)}"
                _run_cli(
                    [
                        resample_cli,
                        "-i",
                        interpolation,
                        staged_in.as_posix(),
                        "--Reference",
                        staged_atlas.as_posix(),
                        "--transformationFile",
                        staged_tx.as_posix(),
                        staged_out.as_posix(),
                    ],
                    log,
                )
                copy_file_atomic(staged_out, output_path)
    else:
        _run_cli(
            [
                brainsfit_cli,
                "--fixedVolume",
                atlas_t2.as_posix(),
                "--movingVolume",
                native.b0_nrrd_path.as_posix(),
                "--linearTransform",
                bundle.transform_path.as_posix(),
                "--useRigid",
                "--useAffine",
            ],
            log,
        )
        for input_path, output_path, interpolation in resample_jobs:
            _run_cli(
                [
                    resample_cli,
                    "-i",
                    interpolation,
                    input_path.as_posix(),
                    "--Reference",
                    atlas_t2.as_posix(),
                    "--transformationFile",
                    bundle.transform_path.as_posix(),
                    output_path.as_posix(),
                ],
                log,
            )

    _write_inverse_transform(bundle.transform_path, bundle.inverse_transform_path)
    if expected_manifest != _registered_manifest(raw, native, bundle):
        raise RuntimeError('Inputs changed during shared registration')
    files = [getattr(bundle, name) for name in bundle.__dataclass_fields__
             if name.endswith('_path') and name not in ('atlas_path', 'manifest_path')]
    _save_artifacts(bundle.bundle_dir, files)
    _write_manifest(bundle.manifest_path, expected_manifest)
    return bundle
