"""Prepare external DWI inputs into the SlicerDSpace subject-root layout."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from shared.external_process import subprocess_no_window_kwargs, text_subprocess_kwargs


ImportSourceKind = Literal["canonical", "dicom", "nifti", "nrrd"]
ImportBackend = Literal["auto", "copy", "dcm2niix", "mrconvert", "dwiconvert"]
ImportOutputKind = Literal["canonical-nifti", "compat-nrrd"]


CANONICAL_INPUT_FILES = ("data.nii.gz", "bvals", "bvecs")
NIFTI_SUFFIXES = (".nii", ".nii.gz")
NRRD_SUFFIXES = (".nrrd", ".nhdr")
DICOM_SUFFIXES = (".dcm", ".ima")


@dataclass(frozen=True)
class InputImportConfig:
    """Options for preparing an external DWI source into a subject root."""

    source: Path
    subject_root: Path
    subject_id: str
    source_kind: ImportSourceKind = "canonical"
    backend: ImportBackend = "auto"
    output_kind: ImportOutputKind = "canonical-nifti"
    bval_path: Path | None = None
    bvec_path: Path | None = None
    metadata_path: Path | None = None
    mask_path: Path | None = None
    dcm2niix_executable: Path | None = None
    mrconvert_executable: Path | None = None
    dwiconvert_executable: Path | None = None
    dry_run: bool = False
    force: bool = False
    # D3: the identity of the user's real source, given by a wrapper that imports a temporary copy of it
    # (detached NRRD header, chosen DICOM series). None: prepare_inputs computes it from ``source``.
    source_signatures: dict[str, Any] | None = None


def detect_import_source_kind(path: Path) -> ImportSourceKind:
    """Detect the most likely import source kind."""

    resolved = path.expanduser()
    if resolved.is_dir():
        input_dir = resolved / "input"
        if input_dir.is_dir() and all((input_dir / name).is_file() for name in CANONICAL_INPUT_FILES):
            return "canonical"
        if all((resolved / name).is_file() for name in CANONICAL_INPUT_FILES):
            return "canonical"
        if _directory_has_dicom_like_files(resolved):
            return "dicom"
        return "dicom"
    name = resolved.name.lower()
    if name.endswith(NRRD_SUFFIXES):
        return "nrrd"
    if name.endswith(NIFTI_SUFFIXES):
        return "nifti"
    raise ValueError(f"Unsupported import source: {path}")


def _directory_has_dicom_like_files(path: Path) -> bool:
    try:
        for child in path.iterdir():
            if child.is_file() and (child.suffix.lower() in DICOM_SUFFIXES or not child.suffix):
                return True
            if child.is_dir() and _directory_has_dicom_like_files(child):
                return True
    except OSError:
        return False
    return False


def packaged_dcm2niix() -> Path | None:
    """Executable of the pip package ``dcm2niix`` installed in this Python, found through its module (B6).

    S-N4: only string or path attributes count; anything else falls back. The wheel puts the executable at
    ``<site>/bin/dcm2niix`` beside the module folder, so that place is searched too.
    """
    try:
        import dcm2niix
    except ImportError:
        return None
    candidates = []
    for value in (getattr(dcm2niix, "bin_path", None), getattr(dcm2niix, "bin", None)):
        if isinstance(value, (str, os.PathLike)) and str(value):
            candidates.append(Path(value))
    module_file = getattr(dcm2niix, "__file__", None)
    if isinstance(module_file, (str, os.PathLike)) and str(module_file):
        folder = Path(module_file).parent
        candidates.extend([
            folder.parent / "bin" / "dcm2niix",
            folder.parent / "Scripts" / "dcm2niix",
            folder.parent.parent / "bin" / "dcm2niix",
            folder.parent.parent / "Scripts" / "dcm2niix",
            folder / "bin" / "dcm2niix",
        ])
    for path in candidates:
        for candidate in (path, path.with_name(path.name + ".exe")):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
    return None


def find_dcm2niix() -> str | None:
    """dcm2niix lookup order: the installed package first, then PATH (B6)."""
    package = packaged_dcm2niix()
    return package.as_posix() if package else shutil.which("dcm2niix")


def _resolve_executable(explicit: Path | None, name: str) -> str:
    if explicit:
        return explicit.expanduser().as_posix()
    found = find_dcm2niix() if name == "dcm2niix" else shutil.which(name)
    if found:
        return found
    return name


def _sidecar_for(source: Path, suffix: str) -> Path:
    name = source.name
    if name.endswith(".nii.gz"):
        return source.with_name(name[:-7] + suffix)
    return source.with_suffix(suffix)


def _copy_file(source: Path, target: Path, *, force: bool) -> None:
    if target.exists():
        try:
            if source.expanduser().resolve() == target.expanduser().resolve():
                return
        except OSError:
            pass
        if not force:
            raise FileExistsError(f"Refusing to overwrite existing file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _write_text_file(content: str, target: Path, *, force: bool) -> None:
    if target.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _run(command: list[str], *, dry_run: bool, cwd: Path | None = None) -> dict[str, Any]:
    if dry_run:
        return {"command": command, "returncode": None, "stdout": "", "stderr": "", "dry_run": True}
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **subprocess_no_window_kwargs(**text_subprocess_kwargs()),
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Input import backend executable was not found: {command[0]}. "
            "Run the command with Slicer Python or select an explicit converter path."
        ) from exc
    result = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "dry_run": False,
    }
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout).strip()
        detail_suffix = f"\n{details[-4000:]}" if details else ""
        raise RuntimeError(
            f"Input import command failed with exit code {completed.returncode}: {' '.join(command)}{detail_suffix}"
        )
    return result


def _lossless_signed_nrrd(source: Path, destination: Path) -> dict[str, Any]:
    """Bridge DWIConvert's short-only reader without enabling lossy conversion."""
    try:
        import nrrd
        import numpy as np
    except ImportError as exc:
        raise RuntimeError('This unsigned NRRD requires NumPy and pynrrd for a checked, lossless conversion.') from exc
    data, header = nrrd.read(str(source))
    if data.dtype != np.dtype('uint16') or data.ndim != 4 or not data.size:
        raise ValueError('Lossless DWIConvert compatibility requires a nonempty uint16 diffusion NRRD.')
    maximum = int(data.max())
    if maximum > np.iinfo(np.int16).max:
        raise ValueError('Unsigned NRRD values exceed int16 range; refusing lossy conversion. Use a compatible converter.')
    gradients = [key for key in header if key.startswith('DWMRI_gradient_')]
    if not gradients or 'DWMRI_b-value' not in header:
        raise ValueError('Diffusion gradient metadata is missing; refusing NRRD compatibility conversion.')
    original = dict(header)
    # A detached source's filename must not redirect writes to its raw data.
    header.pop('data file', None)
    header.pop('type', None)
    nrrd.write(str(destination), data.astype(np.int16), header=header)
    written = nrrd.read_header(str(destination))
    for key in ('space', 'space directions', 'space origin', 'measurement frame', 'kinds',
                'DWMRI_b-value', *gradients):
        if key not in original:
            continue
        left, right = original[key], written.get(key)
        same = (np.array_equal(left, right, equal_nan=True) if key in
                {'space directions', 'space origin', 'measurement frame'} else
                np.array_equal(left, right))
        if not same:
            raise ValueError('NRRD compatibility conversion changes metadata: ' + key)
    return dict(source=str(source), original_dtype='uint16', converted_dtype='int16',
                minimum=int(data.min()), maximum=maximum, source_shape=list(data.shape),
                gradient_count=len(gradients), metadata_preserved=True, lossy_conversion=False)


def _run_nrrd_conversion(command, source, targets, *, dry_run):
    try:
        return _run(command, dry_run=dry_run)
    except RuntimeError as exc:
        message = str(exc)
        if 'Unsupported source pixel type' not in message or 'unsigned_short' not in message:
            raise
        if any(targets[key].exists() for key in ('dwi', 'bvals', 'bvecs')):
            raise RuntimeError('Conversion left partial outputs; inspect them before retrying.') from exc
        with tempfile.TemporaryDirectory(prefix='.signed-nrrd-', dir=targets['input_dir']) as temporary:
            signed = Path(temporary) / 'dwi.nrrd'
            receipt = _lossless_signed_nrrd(source, signed)
            retry = list(command)
            retry[retry.index('--inputVolume') + 1] = str(signed)
            result = _run(retry, dry_run=False)
            result['initial_failure'] = message
            result['lossless_input_cast'] = receipt
            return result


def _nrrd_import_gradient_check(source: Path, targets: dict[str, Path]) -> dict[str, Any]:
    """Gradient frame (batch 2): compare the imported bvecs with the source NRRD's own gradients.

    The world direction of each source gradient is S MF g_file. The FSL rule for the output NIfTI's affine gives
    the bvecs world_to_fsl(g). DWIConvert NrrdToFSL ignores the rotation of the grid, so its bvecs are wrong on
    oblique sources. Returns the largest angle between the imported and the rule's directions and the rule's rows.
    """
    import nibabel as nib
    import nrrd
    import numpy as np

    from shared import gradient_frame

    header = nrrd.read_header(str(source))
    count = sum(1 for key in header if str(key).startswith("DWMRI_gradient_"))
    keys = ["DWMRI_gradient_%04d" % index for index in range(count)]
    if any(key not in header for key in keys):
        raise ValueError(f"Source NRRD gradient keys are not numbered 0000 to N-1: {source}")
    if count == 0:
        return dict(count=0, max_error_deg=0.0, rows=None)
    bvecs = np.loadtxt(targets["bvecs"], ndmin=2)
    if bvecs.shape == (3, count):  # the FSL layout: three rows
        bvecs = bvecs.T
    if bvecs.shape != (count, 3):
        raise ValueError(f"The imported bvec file has {bvecs.shape} entries; the source NRRD has {count} gradients")
    frame = np.asarray(header.get("measurement frame", np.eye(3)), dtype=float).T  # pynrrd rows -> columns
    values = np.array([gradient_frame.parse_nrrd_vector(header[key]) for key in keys])
    world = gradient_frame.nrrd_to_world(values, header["space"], frame)
    affine = nib.load(str(targets["dwi"])).affine
    source_moving = np.linalg.norm(world, axis=1) > gradient_frame.ZERO_GRADIENT_NORM
    imported_moving = np.linalg.norm(bvecs, axis=1) > gradient_frame.ZERO_GRADIENT_NORM
    if not np.array_equal(source_moving, imported_moving):
        raise ValueError("The imported bvec file and the source NRRD disagree on which volumes are b = 0")
    rows = bvecs.copy()
    error = 0.0
    if source_moving.any():
        unit = world[source_moving] / np.linalg.norm(world[source_moving], axis=1, keepdims=True)
        expected = gradient_frame.world_to_fsl(unit, affine)
        rows[source_moving] = expected / np.linalg.norm(expected, axis=1, keepdims=True)
        error = float(gradient_frame.axis_angle_degrees(bvecs[source_moving], rows[source_moving]).max())
    return dict(count=count, max_error_deg=error, rows=rows)


def _correct_nrrd_import_gradients(source: Path, targets: dict[str, Path]) -> dict[str, Any] | None:
    """Rewrite the imported bvec file by the FSL rule when DWIConvert's directions differ; else keep it as written."""
    from shared import gradient_frame

    check = _nrrd_import_gradient_check(source, targets)
    if check["max_error_deg"] <= gradient_frame.KEEP_TOLERANCE_DEG:
        return None
    rows = check["rows"]
    text = "\n".join(" ".join("0" if value == 0 else repr(float(value)) for value in rows[:, axis])
                     for axis in range(3)) + "\n"
    staged = targets["bvecs"].with_name(targets["bvecs"].name + ".gradient-frame")
    staged.write_text(text, encoding="utf-8")
    os.replace(staged, targets["bvecs"])
    return {"rule": "bvec = F R^-1 g (inverse FSL rule for the output NIfTI affine)",
            "source": "the source NRRD gradients and measurement frame",
            "dwiconvert_max_error_deg": check["max_error_deg"], "bvals": "kept", "image": "kept"}


def _nrrd_import_gradients_current(source: Path, targets: dict[str, Path]) -> bool:
    from shared import gradient_frame

    try:
        return _nrrd_import_gradient_check(source, targets)["max_error_deg"] <= gradient_frame.KEEP_TOLERANCE_DEG
    except Exception:  # unreadable or inconsistent: import again
        return False


def _normalize_dcm2niix_outputs(input_dir: Path, *, force: bool, dry_run: bool) -> list[dict[str, str]]:
    expected = {
        "data.bval": "bvals",
        "data.bvec": "bvecs",
        "data.json": "dwi.json",
    }
    actions: list[dict[str, str]] = []
    for source_name, target_name in expected.items():
        source = input_dir / source_name
        target = input_dir / target_name
        actions.append({"source": source.as_posix(), "target": target.as_posix()})
        if dry_run:
            continue
        if source.is_file():
            if target.exists() and not force:
                raise FileExistsError(f"Refusing to overwrite existing file: {target}")
            source.replace(target)
    return actions


def _canonical_targets(subject_root: Path) -> dict[str, Path]:
    input_dir = subject_root / "input"
    return {
        "input_dir": input_dir,
        "dwi": input_dir / "data.nii.gz",
        "bvals": input_dir / "bvals",
        "bvecs": input_dir / "bvecs",
        "metadata": input_dir / "dwi.json",
        "mask": input_dir / "nodif_brain_mask.nii.gz",
        "manifest": input_dir / "input_import_manifest.json",
        "compat_nrrd": subject_root / "cache" / "imported_nrrd" / "data.nhdr",
    }


def build_import_plan(config: InputImportConfig) -> dict[str, Any]:
    """Build a serializable input-import plan without mutating the filesystem."""

    source = config.source.expanduser().resolve()
    subject_root = config.subject_root.expanduser().resolve()
    source_kind = config.source_kind if config.source_kind != "canonical" else detect_import_source_kind(source)
    backend = _select_backend(
        source_kind,
        config.backend,
        config.output_kind,
        dcm2niix_executable=config.dcm2niix_executable,
        dwiconvert_executable=config.dwiconvert_executable,
    )
    targets = _canonical_targets(subject_root)
    command: list[str] | None = None

    if backend == "dcm2niix":
        command = [
            _resolve_executable(config.dcm2niix_executable, "dcm2niix"),
            "-z",
            "y",
            "-ba",
            "y",
            "-f",
            "data",
            "-o",
            targets["input_dir"].as_posix(),
            source.as_posix(),
        ]
    elif backend == "mrconvert":
        command = [
            _resolve_executable(config.mrconvert_executable, "mrconvert"),
            source.as_posix(),
            targets["dwi"].as_posix(),
            "-export_grad_fsl",
            targets["bvecs"].as_posix(),
            targets["bvals"].as_posix(),
            "-json_export",
            targets["metadata"].as_posix(),
        ]
        if config.force:
            command.append("-force")
    elif backend == "dwiconvert" and source_kind == "nrrd":
        command = [
            _resolve_executable(config.dwiconvert_executable, "DWIConvert"),
            "--inputVolume",
            source.as_posix(),
            "--conversionMode",
            "NrrdToFSL",
            "--outputNiftiFile",
            targets["dwi"].as_posix(),
            "--outputBValues",
            targets["bvals"].as_posix(),
            "--outputBVectors",
            targets["bvecs"].as_posix(),
        ]
    elif backend == "dwiconvert" and source_kind == "dicom":
        if config.output_kind == "canonical-nifti":
            command = [
                _resolve_executable(config.dwiconvert_executable, "DWIConvert"),
                "--inputDicomDirectory",
                source.as_posix(),
                "--conversionMode",
                "DicomToFSL",
                "--outputNiftiFile",
                targets["dwi"].as_posix(),
                "--outputBValues",
                targets["bvals"].as_posix(),
                "--outputBVectors",
                targets["bvecs"].as_posix(),
            ]
        else:
            command = [
                _resolve_executable(config.dwiconvert_executable, "DWIConvert"),
                "--inputDicomDirectory",
                source.as_posix(),
                "--conversionMode",
                "DicomToNrrd",
                "--outputVolume",
                targets["compat_nrrd"].as_posix(),
            ]

    return {
        "kind": "slicerdsurfer_input_import_plan",
        "version": 1,
        "subject_id": config.subject_id,
        "source": source.as_posix(),
        "source_kind": source_kind,
        "backend": backend,
        "output_kind": config.output_kind,
        "canonical_layout": {key: value.as_posix() for key, value in targets.items()},
        "command": command,
        "sidecars": {
            "bval": (config.bval_path or _sidecar_for(source, ".bval")).expanduser().as_posix()
            if source_kind == "nifti"
            else None,
            "bvec": (config.bvec_path or _sidecar_for(source, ".bvec")).expanduser().as_posix()
            if source_kind == "nifti"
            else None,
            "metadata": (config.metadata_path or _sidecar_for(source, ".json")).expanduser().as_posix()
            if source_kind == "nifti"
            else None,
            "mask": config.mask_path.expanduser().as_posix() if config.mask_path else None,
        },
        "policy": (
            "External inputs are normalized into input/data.nii.gz, input/bvals, input/bvecs and optional input/dwi.json. "
            "The downstream pipeline continues to consume only the canonical subject-root layout."
        ),
    }


def _select_backend(
    source_kind: ImportSourceKind,
    backend: ImportBackend,
    output_kind: ImportOutputKind,
    *,
    dcm2niix_executable: Path | None = None,
    dwiconvert_executable: Path | None = None,
) -> ImportBackend:
    if backend != "auto":
        return backend
    if source_kind == "canonical":
        return "copy"
    if source_kind == "nifti":
        return "copy"
    if source_kind == "dicom":
        if output_kind == "compat-nrrd":
            return "dwiconvert"
        if dcm2niix_executable is not None or find_dcm2niix():
            return "dcm2niix"
        if dwiconvert_executable is not None or shutil.which("DWIConvert"):
            return "dwiconvert"
        return "dcm2niix"
    if source_kind == "nrrd":
        return "dwiconvert"
    raise ValueError(f"Unsupported source kind: {source_kind}")


def _path_signature(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            return None
        stat = resolved.stat()
    except OSError:
        return None
    return {
        "path": resolved.as_posix(),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def dicom_source_signature(folder: Path, series: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """D3.1: a stat-only signature of a DICOM source (no pixel data is read).

    ``series`` (a ``scan_series`` entry: ``uid`` and ``paths``) names the one series that the tile import
    converts; without it the signature covers every DICOM file in the folder. Run it in an import job,
    never on the GUI thread: it lists and stats every file.
    """
    root = Path(folder).expanduser().resolve()
    if series is None:
        from shared.dicom_input import _inventory

        if not root.is_dir():
            raise FileNotFoundError(f"DICOM folder not found: {root}")
        entries = _inventory(root)
        series_uid = None
    else:
        entries = []
        for item in series.get("paths") or ():
            path = Path(item)
            stat = path.stat()  # a missing series file raises FileNotFoundError, as the import would
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError:
                relative = path.as_posix()
            entries.append({"path": relative, "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
        entries.sort(key=lambda entry: entry["path"])
        series_uid = str(series.get("uid") or "")
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return {
        "folder": root.as_posix(),
        "series_uid": series_uid,
        "count": len(entries),
        "inventory_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def _import_source_signatures(
    config: InputImportConfig,
    source: Path,
    source_kind: ImportSourceKind,
) -> dict[str, Any] | None:
    if config.output_kind != "canonical-nifti":
        return None
    mask_sig = _path_signature(config.mask_path)
    if source_kind == "dicom":
        return {"dicom": dicom_source_signature(source), "mask": mask_sig}
    if source_kind == "canonical":
        src_input = source / "input" if (source / "input").is_dir() else source
        return {
            "dwi": _path_signature(src_input / "data.nii.gz"),
            "bvals": _path_signature(src_input / "bvals"),
            "bvecs": _path_signature(src_input / "bvecs"),
            "metadata": _path_signature(src_input / "dwi.json"),
            "mask": mask_sig if config.mask_path is not None else _path_signature(src_input / "nodif_brain_mask.nii.gz"),
        }
    if source_kind == "nifti":
        bval = (config.bval_path or _sidecar_for(source, ".bval")).expanduser().resolve()
        bvec = (config.bvec_path or _sidecar_for(source, ".bvec")).expanduser().resolve()
        metadata = (config.metadata_path or _sidecar_for(source, ".json")).expanduser().resolve()
        return {
            "dwi": _path_signature(source),
            "bval": _path_signature(bval),
            "bvec": _path_signature(bvec),
            "metadata": _path_signature(metadata),
            "mask": mask_sig,
        }
    return {
        "dwi": _path_signature(source),
        "mask": mask_sig,
    }


def _output_signatures(targets: dict[str, Path]) -> dict[str, Any]:
    signatures = {
        key: _path_signature(targets[key])
        for key in ("dwi", "bvals", "bvecs", "metadata")
    }
    auto_mask_manifest = targets["mask"].with_name("nodif_brain_mask.manifest.json")
    signatures["mask"] = None if auto_mask_manifest.is_file() else _path_signature(targets["mask"])
    return signatures


# ----- D3: the subject folder follows the input identity ---------------------------------------------------------

INTENT_MARKER_NAME = "input_import_intent.json"
SUBJECT_SUFFIX_LIMIT = 99
_OPTIONAL_SIGNATURE_KEYS = frozenset({"metadata", "mask"})
_AUTO_DICOM_RECEIPT = "auto_dicom_import.json"  # shared.dicom_input.RECEIPT (automatic DICOM import)
# FE1: "Check environment dependencies" writes its reports here (relative to the subject root) before any import.
_ENVIRONMENT_SCAFFOLD = ("outputs", "00_run_metadata", "environment")


@dataclass(frozen=True)
class SubjectChoice:
    """The subject folder an import writes to, and why (``new``, ``same_input``, ``retry_same_input``,
    ``legacy_same_source``)."""

    subject_id: str
    subject_root: Path
    decision: str
    base_id: str

    def as_payload(self) -> dict[str, str]:
        return {"base_id": self.base_id, "subject_id": self.subject_id,
                "subject_root": self.subject_root.as_posix(), "decision": self.decision}


def input_identity(signatures: Mapping[str, Any] | None) -> dict[str, Any]:
    """The source signatures without the ``mask`` and ``raw`` entries: another mask for the same DWI is the same
    input, and so is another detached data file under the same NHDR header (FE2: the importer's reuse test still
    sees ``raw`` and converts again)."""
    return {key: value for key, value in (signatures or {}).items() if key not in {"mask", "raw"}}


def import_source_signatures(
    config: InputImportConfig,
    *,
    raw_path: Path | None = None,
    dicom_series: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The source signatures of one GUI import, computed from the user's real source.

    ``raw_path``: the file of the 'Detached data' tile (its stat signature is part of the input).
    ``dicom_series``: the chosen series of a DICOM tile import. A required file that does not exist raises
    ``FileNotFoundError`` before any subject folder is chosen. Lists a DICOM folder: run it in the import job.
    """
    source = config.source.expanduser().resolve()
    source_kind = config.source_kind if config.source_kind != "canonical" else detect_import_source_kind(source)
    if source_kind == "dicom" and dicom_series is not None:
        if config.output_kind != "canonical-nifti":
            raise ValueError("A subject folder can be chosen only for a canonical NIfTI import.")
        signatures: dict[str, Any] | None = {"dicom": dicom_source_signature(source, dicom_series),
                                             "mask": _path_signature(config.mask_path)}
    else:
        signatures = _import_source_signatures(config, source, source_kind)
    if signatures is None:
        raise ValueError("A subject folder can be chosen only for a canonical NIfTI import.")
    required: dict[str, Path] = {}
    if source_kind == "nifti":
        required = {
            "dwi": source,
            "bval": (config.bval_path or _sidecar_for(source, ".bval")).expanduser(),
            "bvec": (config.bvec_path or _sidecar_for(source, ".bvec")).expanduser(),
        }
    elif source_kind == "canonical":
        src_input = source / "input" if (source / "input").is_dir() else source
        required = {"dwi": src_input / "data.nii.gz", "bvals": src_input / "bvals", "bvecs": src_input / "bvecs"}
    elif source_kind == "nrrd":
        required = {"dwi": source}
    if raw_path is not None:
        signatures["raw"] = _path_signature(raw_path)
        required["raw"] = Path(raw_path)
    if config.mask_path is not None and signatures.get("mask") is None:
        required["mask"] = config.mask_path
    for key, path in required.items():
        if signatures.get(key) is None:
            raise FileNotFoundError(f"Input file not found: {path}")
    return signatures


def subject_root_rule(output_text: str, fallback_parent: str | Path) -> Callable[[str], Path]:
    """``root_for_id`` with the GUI rule: ``<Output Folder>/<ID>`` (resolved), else ``<fallback_parent>/<ID>``."""

    def root_for_id(subject_id: str) -> Path:
        from shared.ui_run_controller import resolve_single_subject_workspace_root

        root = resolve_single_subject_workspace_root(output_text, subject_id)
        if root is not None:
            return root.expanduser().resolve()
        return Path(fallback_parent).expanduser() / subject_id

    return root_for_id


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _holds_data(folder: Path, marker: Path, scaffold: Path | None = None, *, in_scaffold: bool = False) -> bool:
    """True when ``folder`` holds anything but hidden entries, empty folders, the intent ``marker`` and the
    files at any depth below ``scaffold`` (the environment-check folder; a symlink there still counts)."""
    try:
        children = list(folder.iterdir())
    except OSError:
        return True  # unreadable: treat as data, never as unused
    for child in children:
        if child.name.startswith("."):
            continue
        if child.is_symlink():
            return True
        if child.is_dir():
            if _holds_data(child, marker, scaffold, in_scaffold=in_scaffold or child == scaffold):
                return True
            continue
        if in_scaffold or child == marker:
            continue
        return True
    return False


def _recorded_source(manifest: Mapping[str, Any]) -> str:
    """The source a legacy manifest names; the tile wrappers record the real source beside a temporary copy."""
    plan = manifest.get("plan") if isinstance(manifest.get("plan"), Mapping) else {}
    series = plan.get("dicom_series") if isinstance(plan.get("dicom_series"), Mapping) else {}
    detached = plan.get("detached_nrrd") if isinstance(plan.get("detached_nrrd"), Mapping) else {}
    return str(series.get("source") or detached.get("header") or plan.get("source") or "")


def _same_path(left: str, right: Path | str | None) -> bool:
    if not left or right is None:
        return False
    try:
        return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()
    except OSError:
        return False


def _same_legacy_series(plan: Mapping[str, Any], identity: Mapping[str, Any]) -> bool:
    """A legacy DICOM series import matches only the same series: another series of the folder is another input."""
    recorded = plan.get("dicom_series") if isinstance(plan.get("dicom_series"), Mapping) else {}
    recorded_uid = str(recorded.get("series_instance_uid") or "")
    dicom = identity.get("dicom") if isinstance(identity.get("dicom"), Mapping) else {}
    current_uid = str(dicom.get("series_uid") or "")
    return not (recorded_uid or current_uid) or recorded_uid == current_uid


def _subject_root_decision(
    root: Path,
    identity: Mapping[str, Any],
    source: Path | None,
    source_kind: str | None,
) -> str | None:
    """Rules 1-4 of D3.2 for one candidate root; ``None`` means "not this input, try the next ID"."""
    if not root.exists() and not root.is_symlink():
        return "new"
    if not root.is_dir():
        return None
    input_dir = root / "input"
    if not _holds_data(root, input_dir / INTENT_MARKER_NAME, root.joinpath(*_ENVIRONMENT_SCAFFOLD)):
        return "new"
    if (input_dir / _AUTO_DICOM_RECEIPT).exists():
        return None  # automatically imported data: never overwritten by a tile import
    manifest = _read_json_object(input_dir / "input_import_manifest.json")
    if manifest is not None and manifest.get("status") == "completed":
        if "source_signatures" in manifest:
            saved = manifest.get("source_signatures")
            if isinstance(saved, Mapping) and input_identity(saved) == identity:
                return "same_input"
            return None
        plan = manifest.get("plan") if isinstance(manifest.get("plan"), Mapping) else {}
        if (source is not None and source_kind and plan.get("source_kind") == source_kind
                and _same_path(_recorded_source(manifest), source)
                and _same_legacy_series(plan, identity)):
            return "legacy_same_source"
        return None
    marker = _read_json_object(input_dir / INTENT_MARKER_NAME)
    if marker is not None and marker.get("identity") == identity:
        return "retry_same_input"
    return None


def choose_subject_for_source(
    base_id: str,
    root_for_id: Callable[[str], Path],
    signatures: Mapping[str, Any] | None,
    *,
    source: Path | None = None,
    source_kind: str | None = None,
) -> SubjectChoice:
    """D3.2: the subject ID and folder for this input: ``base``, ``base_2``, ... ``base_99``; the first match wins.

    ``source`` and ``source_kind`` are needed only for the legacy rule (a manifest made before signatures).
    The function creates nothing; it reads the candidate folders.
    """
    base = str(base_id or "").strip()
    if not base:
        raise ValueError("Enter a valid subject ID.")
    if not signatures:
        raise ValueError("The input has no signature; a subject folder cannot be chosen.")
    identity = input_identity(signatures)
    missing = [key for key, value in identity.items() if value is None and key not in _OPTIONAL_SIGNATURE_KEYS]
    if missing:
        raise FileNotFoundError("Input file not found: " + ", ".join(sorted(missing)))
    # JSON form, so the comparison with a saved manifest or marker is exact.
    identity = json.loads(json.dumps(identity))
    for index in range(1, SUBJECT_SUFFIX_LIMIT + 1):
        candidate = base if index == 1 else f"{base}_{index}"
        root = Path(root_for_id(candidate))
        decision = _subject_root_decision(root, identity, source, source_kind)
        if decision is not None:
            return SubjectChoice(subject_id=candidate, subject_root=root, decision=decision, base_id=base)
    raise ValueError(
        f"Subjects {base} to {base}_{SUBJECT_SUFFIX_LIMIT} already hold other data. "
        "Enter another Subject ID or choose another Output Folder."
    )


def _write_intent_marker(input_dir: Path, signatures: Mapping[str, Any]) -> None:
    marker = {
        "kind": "slicerdsurfer_input_import_intent",
        "version": 1,
        "identity": input_identity(signatures),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (input_dir / INTENT_MARKER_NAME).write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare_inputs(config: InputImportConfig) -> dict[str, Any]:
    """Prepare external inputs and write input/input_import_manifest.json."""

    started = time.monotonic()
    plan = build_import_plan(config)
    targets = _canonical_targets(config.subject_root.expanduser().resolve())
    source = config.source.expanduser().resolve()
    source_kind = plan["source_kind"]
    backend = plan["backend"]
    source_sigs = None
    if not config.dry_run and config.output_kind == "canonical-nifti":
        source_sigs = (dict(config.source_signatures) if config.source_signatures is not None
                       else _import_source_signatures(config, source, source_kind))
    if not config.dry_run and config.force and source_sigs is not None and targets["manifest"].is_file():
        try:
            saved = json.loads(targets["manifest"].read_text(encoding="utf-8"))
        except (OSError, ValueError):
            saved = None
        if (
            isinstance(saved, dict)
            and saved.get("status") == "completed"
            and saved.get("source_signatures") == source_sigs
            and isinstance(saved.get("output_signatures"), dict)
            and saved.get("output_signatures") == _output_signatures(targets)
            and (source_kind != "nrrd" or backend != "dwiconvert" or _nrrd_import_gradients_current(source, targets))
        ):
            return {**saved, "status": "completed", "reused": True, "runtime_seconds": 0.0}
    if not config.dry_run and not config.force:
        if config.output_kind == 'compat-nrrd':
            candidates = (targets['compat_nrrd'], targets['compat_nrrd'].with_suffix('.raw.gz'))
        else:
            candidates = tuple(targets[key] for key in ('dwi', 'bvals', 'bvecs', 'metadata', 'mask'))
            candidates += tuple(targets['input_dir'] / name for name in ('data.bval', 'data.bvec', 'data.json'))
        conflicts = [path for path in candidates if path.exists() or path.is_symlink()]
        if conflicts:
            raise FileExistsError('Refusing to overwrite existing input files without --force: ' +
                                  ', '.join(path.as_posix() for path in conflicts))
    if not config.dry_run:
        targets["input_dir"].mkdir(parents=True, exist_ok=True)
        if source_sigs is not None:
            # D3: before any staged file is removed or written, so a failed import is retried in this folder.
            _write_intent_marker(targets["input_dir"], source_sigs)
        if config.force and config.mask_path is None:
            src_input = (source / "input" if (source / "input").is_dir() else source) if source_kind == "canonical" else None
            same_canonical = False
            if src_input is not None:
                try:
                    same_canonical = src_input.resolve() == targets["input_dir"].resolve()
                except OSError:
                    same_canonical = False
            if not same_canonical and (source_kind != "canonical" or not (src_input / "nodif_brain_mask.nii.gz").is_file()):
                targets["mask"].unlink(missing_ok=True)
                targets["mask"].with_name("nodif_brain_mask.manifest.json").unlink(missing_ok=True)
        if config.force and backend != "copy" and config.output_kind == "canonical-nifti":
            for stale_name in ("data.nii.gz", "data.nii", "bvals", "bvecs", "dwi.json", "data.bval", "data.bvec", "data.json"):
                (targets["input_dir"] / stale_name).unlink(missing_ok=True)
    command_result: dict[str, Any] | None = None
    gradient_receipt: dict[str, Any] | None = None

    if backend == "copy":
        if source_kind == "canonical":
            _copy_canonical_source(source, targets, force=config.force, dry_run=config.dry_run)
        elif source_kind == "nifti":
            _copy_nifti_source(source, targets, config, force=config.force, dry_run=config.dry_run)
    else:
        command = plan.get("command")
        if not command:
            raise ValueError(f"No import command was generated for backend={backend}.")
        cwd = targets["input_dir"] if backend == "dcm2niix" else None
        if backend == 'dwiconvert' and source_kind == 'nrrd':
            command_result = _run_nrrd_conversion(command, source, targets, dry_run=config.dry_run)
            if not config.dry_run and config.output_kind == "canonical-nifti":
                gradient_receipt = _correct_nrrd_import_gradients(source, targets)
        else:
            command_result = _run(command, dry_run=config.dry_run, cwd=cwd)
        if backend == "dcm2niix":
            plan["postprocess"] = _normalize_dcm2niix_outputs(targets["input_dir"], force=config.force, dry_run=config.dry_run)
    if config.mask_path:
        if not config.dry_run:
            targets["mask"].with_name("nodif_brain_mask.manifest.json").unlink(missing_ok=True)
            _copy_file(config.mask_path.expanduser().resolve(), targets["mask"], force=config.force)

    manifest = {
        "kind": "slicerdsurfer_input_import_manifest",
        "version": 1,
        "status": "planned" if config.dry_run else "completed",
        "runtime_seconds": None if config.dry_run else time.monotonic() - started,
        "timing_scope": "Input preparation/conversion and file copies, excluding GUI and Slicer startup",
        "plan": plan,
        "command_result": command_result,
        "outputs": {
            "dwi": targets["dwi"].as_posix(),
            "bvals": targets["bvals"].as_posix(),
            "bvecs": targets["bvecs"].as_posix(),
            "metadata": targets["metadata"].as_posix(),
            "mask": targets["mask"].as_posix() if targets["mask"].exists() or config.mask_path else None,
        },
        **({"source_signatures": source_sigs} if source_sigs is not None else {}),
        **({"gradient_frame": gradient_receipt} if gradient_receipt is not None else {}),
        **({"output_signatures": _output_signatures(targets)} if not config.dry_run else {}),
    }
    if not config.dry_run:
        targets["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _copy_canonical_source(source: Path, targets: dict[str, Path], *, force: bool, dry_run: bool) -> None:
    input_dir = source / "input" if (source / "input").is_dir() else source
    mapping = {
        input_dir / "data.nii.gz": targets["dwi"],
        input_dir / "bvals": targets["bvals"],
        input_dir / "bvecs": targets["bvecs"],
    }
    optional = {
        input_dir / "dwi.json": targets["metadata"],
        input_dir / "nodif_brain_mask.nii.gz": targets["mask"],
    }
    if dry_run:
        return
    for src, dst in mapping.items():
        _copy_file(src, dst, force=force)
    for src, dst in optional.items():
        if src.is_file():
            _copy_file(src, dst, force=force)


def _validate_nifti_voxels_single_pass(
    image,
    source: Path,
    *,
    tee_path: Path | None = None,
) -> None:
    """Decode the full 4D NIfTI payload in one sequential stream pass."""
    import numpy as np
    from shared.diffusion_io import open_fast_gzip_reader

    dtype = image.get_data_dtype()
    itemsize = int(dtype.itemsize)
    total_voxels = int(np.prod(image.shape, dtype=np.int64))
    expected_bytes = total_voxels * itemsize
    offset = max(0, int(getattr(image.dataobj, "offset", 0)))
    slope = getattr(image.dataobj, "slope", 1.0)
    inter = getattr(image.dataobj, "inter", 0.0)
    if (slope is not None and not np.isfinite(slope)) or (inter is not None and not np.isfinite(inter)):
        raise ValueError("DWI contains non-finite values")
    needs_float_check = dtype.kind in ("f", "c")
    elems_per_chunk = max(1, (8 * 1024 * 1024) // max(1, itemsize))
    chunk_bytes = elems_per_chunk * itemsize

    tee_fh = tee_path.open("wb") if tee_path is not None else None
    try:
        with open_fast_gzip_reader(source) as stream:
            remaining_skip = offset
            while remaining_skip > 0:
                skipped = stream.read(min(remaining_skip, 4 * 1024 * 1024))
                if not skipped:
                    raise EOFError("Truncated NIfTI header before voxel payload")
                if tee_fh is not None:
                    tee_fh.write(skipped)
                remaining_skip -= len(skipped)
            remaining = expected_bytes
            carry = b""
            while remaining > 0:
                buf = stream.read(min(remaining, chunk_bytes))
                if not buf:
                    raise EOFError("Truncated NIfTI voxel payload")
                if tee_fh is not None:
                    tee_fh.write(buf)
                remaining -= len(buf)
                if needs_float_check:
                    if carry:
                        buf = carry + buf
                        carry = b""
                    usable = (len(buf) // itemsize) * itemsize
                    if usable < len(buf):
                        carry = buf[usable:]
                        buf = buf[:usable]
                    if buf and not np.isfinite(np.frombuffer(buf, dtype=dtype)).all():
                        raise ValueError("DWI contains non-finite values")
            if carry:
                raise EOFError("Unaligned NIfTI voxel payload")
            while True:
                tail = stream.read(1024 * 1024)
                if not tail:
                    break
                if tee_fh is not None:
                    tee_fh.write(tail)
    finally:
        if tee_fh is not None:
            tee_fh.close()


def _copy_nifti_source(
    source: Path,
    targets: dict[str, Path],
    config: InputImportConfig,
    *,
    force: bool,
    dry_run: bool,
) -> None:
    bval = (config.bval_path or _sidecar_for(source, ".bval")).expanduser().resolve()
    bvec = (config.bvec_path or _sidecar_for(source, ".bvec")).expanduser().resolve()
    metadata = (config.metadata_path or _sidecar_for(source, ".json")).expanduser().resolve()
    if dry_run:
        return
    import nibabel as nib
    import numpy as np
    from shared.diffusion_io import commit_predecoded_nifti_cache, prepare_uncompressed_cache_target

    image = nib.load(str(source))
    if len(image.shape) != 4 or not np.isfinite(image.affine).all():
        raise ValueError("DWI must be a four-dimensional NIfTI with a finite affine")
    bval_bytes = bval.read_bytes()
    bvec_bytes = bvec.read_bytes()
    bval_has_bom = bval_bytes.startswith(b"\xef\xbb\xbf")
    bvec_has_bom = bvec_bytes.startswith(b"\xef\xbb\xbf")
    bval_raw = bval_bytes.decode("utf-8-sig")
    bvec_raw = bvec_bytes.decode("utf-8-sig")
    bvals = np.loadtxt(io.StringIO(bval_raw)).reshape(-1)
    bvecs = np.loadtxt(io.StringIO(bvec_raw), ndmin=2)
    bvecs_transposed = False
    if bvecs.ndim == 2 and bvals.size != 3 and bvecs.shape == (bvals.size, 3):
        bvecs = bvecs.T
        bvecs_transposed = True
    if bvals.size != image.shape[3] or bvecs.shape != (3, bvals.size):
        raise ValueError("Gradient count/shape does not match the DWI volumes (expected 3 x N bvecs)")
    if not np.isfinite(bvals).all() or not np.isfinite(bvecs).all():
        raise ValueError("Gradients contain non-finite values")
    target = targets["dwi"]
    for path in (target, targets["bvals"], targets["bvecs"]):
        if path.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    # Force voxel decoding in one sequential stream pass before committing anything, catching truncated data.
    staged_cache_nii: Path | None = None
    expected_cache_bytes = 0
    if isinstance(image, (nib.Nifti1Image, nib.Nifti2Image)) and hasattr(image.dataobj, "offset"):
        offset = max(0, int(getattr(image.dataobj, "offset", 0)))
        expected_cache_bytes = (
            int(np.prod(image.shape, dtype=np.int64)) * int(image.get_data_dtype().itemsize) + offset
        )
        if source.name.lower().endswith(".nii.gz"):
            staged_cache_nii = prepare_uncompressed_cache_target(target, expected_cache_bytes)
        try:
            _validate_nifti_voxels_single_pass(image, source, tee_path=staged_cache_nii)
        except Exception:
            if staged_cache_nii is not None:
                staged_cache_nii.unlink(missing_ok=True)
            raise
    else:
        for index in range(image.shape[3]):
            if not np.isfinite(np.asanyarray(image.dataobj[..., index])).all():
                raise ValueError("DWI contains non-finite values")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(suffix=".nii.gz", dir=target.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        if source.name.lower().endswith(".nii.gz"):
            shutil.copyfile(source, temporary)
        else:
            with source.open("rb") as src, gzip.open(temporary, "wb", compresslevel=1) as dst:
                shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)
        nib.load(str(temporary))
        os.replace(temporary, target)
    except Exception:
        if staged_cache_nii is not None:
            staged_cache_nii.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)
    if staged_cache_nii is not None:
        commit_predecoded_nifti_cache(target, staged_cache_nii, expected_cache_bytes)
    if bval_has_bom:
        _write_text_file(bval_raw, targets["bvals"], force=force)
    else:
        _copy_file(bval, targets["bvals"], force=force)
    if bvecs_transposed:
        transposed_text = "\n".join(" ".join(f"{float(v):.15g}" for v in row) for row in bvecs) + "\n"
        _write_text_file(transposed_text, targets["bvecs"], force=force)
    elif bvec_has_bom:
        _write_text_file(bvec_raw, targets["bvecs"], force=force)
    else:
        _copy_file(bvec, targets["bvecs"], force=force)
    if metadata.is_file():
        _copy_file(metadata, targets["metadata"], force=force)
