"""Prepare external DWI inputs into the SlicerDSpace subject-root layout."""

from __future__ import annotations

import json
import gzip
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from shared.external_process import text_subprocess_kwargs


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
    """Executable of the pip package ``dcm2niix`` installed in this Python, found through its module (B6)."""
    try:
        import dcm2niix
    except ImportError:
        return None
    for value in (getattr(dcm2niix, "bin_path", None), getattr(dcm2niix, "bin", None)):
        if not value:
            continue
        path = Path(value)
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
    if target.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


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
            **text_subprocess_kwargs(),
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


def prepare_inputs(config: InputImportConfig) -> dict[str, Any]:
    """Prepare external inputs and write input/input_import_manifest.json."""

    started = time.monotonic()
    plan = build_import_plan(config)
    targets = _canonical_targets(config.subject_root.expanduser().resolve())
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
    command_result: dict[str, Any] | None = None
    source = config.source.expanduser().resolve()
    source_kind = plan["source_kind"]
    backend = plan["backend"]

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
        else:
            command_result = _run(command, dry_run=config.dry_run, cwd=cwd)
        if backend == "dcm2niix":
            plan["postprocess"] = _normalize_dcm2niix_outputs(targets["input_dir"], force=config.force, dry_run=config.dry_run)
    if config.mask_path:
        _copy_file(config.mask_path.expanduser().resolve(), targets["mask"], force=config.force) if not config.dry_run else None

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

    image = nib.load(str(source))
    if len(image.shape) != 4 or not np.isfinite(image.affine).all():
        raise ValueError("DWI must be a four-dimensional NIfTI with a finite affine")
    bvals = np.loadtxt(bval).reshape(-1)
    bvecs = np.loadtxt(bvec, ndmin=2)
    if bvals.size != image.shape[3] or bvecs.shape != (3, bvals.size):
        raise ValueError("Gradient count/shape does not match the DWI volumes (expected 3 x N bvecs)")
    if not np.isfinite(bvals).all() or not np.isfinite(bvecs).all():
        raise ValueError("Gradients contain non-finite values")
    # Force voxel decoding before committing anything, catching truncated data.
    for index in range(image.shape[3]):
        if not np.isfinite(np.asanyarray(image.dataobj[..., index])).all():
            raise ValueError("DWI contains non-finite values")
    target = targets["dwi"]
    for path in (target, targets["bvals"], targets["bvecs"]):
        if path.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(suffix=".nii.gz", dir=target.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        if source.name.lower().endswith(".nii.gz"):
            shutil.copyfile(source, temporary)
        else:
            with source.open("rb") as src, gzip.open(temporary, "wb") as dst:
                shutil.copyfileobj(src, dst)
        nib.load(str(temporary))
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    _copy_file(bval, targets["bvals"], force=force)
    _copy_file(bvec, targets["bvecs"], force=force)
    if metadata.is_file():
        _copy_file(metadata, targets["metadata"], force=force)
