"""Automatic detection and transactional staging for raw NIfTI and NRRD inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional


REQUIRED = ("data.nii.gz", "bvals", "bvecs")
RECEIPT = "auto_convertible_import.json"


def _strip_nifti_stem(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def _matching_gradients(nifti_path: Path) -> tuple[Optional[Path], Optional[Path]]:
    folder = nifti_path.parent
    stem = _strip_nifti_stem(nifti_path)
    for bval_name, bvec_name in (
        (f"{stem}.bval", f"{stem}.bvec"),
        (f"{stem}.bvals", f"{stem}.bvecs"),
        ("bvals", "bvecs"),
    ):
        bval = folder / bval_name
        bvec = folder / bvec_name
        if bval.is_file() and bvec.is_file():
            return bval, bvec
    return None, None


def _matching_sidecars(nifti_path: Path) -> dict[str, Path]:
    folder = nifti_path.parent
    stem = _strip_nifti_stem(nifti_path)
    extra: dict[str, Path] = {}
    for json_name in (f"{stem}.json", "dwi.json", "data.json"):
        candidate = folder / json_name
        if candidate.is_file():
            extra["metadata_path"] = candidate
            break
    mask_candidates = [
        "nodif_brain_mask.nii.gz",
        f"{stem}_mask.nii.gz",
        f"{stem}_brain_mask.nii.gz",
    ]
    if stem.lower().endswith("_dwi") and len(stem) > 4:
        prefix = stem[:-4]
        mask_candidates.extend([f"{prefix}_mask.nii.gz", f"{prefix}_brain_mask.nii.gz"])
    for mask_name in mask_candidates:
        candidate = folder / mask_name
        if candidate.is_file() and candidate.resolve() != nifti_path.resolve():
            extra["mask_path"] = candidate
            break
    return extra


def find_convertible_input(path: Path) -> Optional[tuple[str, Path, Path, dict[str, Path]]]:
    """Return ``(kind, source_file, subject_root, extra_paths)`` if convertible."""

    resolved = Path(path).expanduser().resolve()
    if resolved.is_file():
        lower = resolved.name.lower()
        if lower.endswith((".nrrd", ".nhdr")):
            return ("nrrd", resolved, resolved.parent, _matching_sidecars(resolved))
        if lower.endswith((".nii", ".nii.gz")):
            if resolved.parent.name == "input" and all((resolved.parent / f).is_file() for f in REQUIRED):
                return None
            bval, bvec = _matching_gradients(resolved)
            if bval is not None and bvec is not None:
                extra = {"bval_path": bval, "bvec_path": bvec, **_matching_sidecars(resolved)}
                return ("nifti", resolved, resolved.parent, extra)
        return None

    if not resolved.is_dir():
        return None
    if (
        (resolved / "input").is_dir()
        and all((resolved / "input" / f).is_file() for f in REQUIRED)
        and not (resolved / "input" / RECEIPT).is_file()
    ):
        return None

    if all((resolved / f).is_file() for f in REQUIRED):
        source = resolved / "data.nii.gz"
        extra = {
            "bval_path": resolved / "bvals",
            "bvec_path": resolved / "bvecs",
            **_matching_sidecars(source),
        }
        return ("nifti", source, resolved, extra)

    nifti_candidates: list[tuple[Path, Path, Path]] = []
    try:
        entries = sorted(resolved.iterdir())
    except OSError:
        return None

    for child in entries:
        if not child.is_file() or child.is_symlink():
            continue
        lower = child.name.lower()
        if "mask" in lower:
            continue
        if lower.endswith((".nii", ".nii.gz")):
            bval, bvec = _matching_gradients(child)
            if bval is not None and bvec is not None:
                nifti_candidates.append((child, bval, bvec))
    if len(nifti_candidates) == 1:
        source, bval, bvec = nifti_candidates[0]
        extra = {"bval_path": bval, "bvec_path": bvec, **_matching_sidecars(source)}
        return ("nifti", source, resolved, extra)

    nrrd_candidates = [
        child
        for child in entries
        if child.is_file()
        and not child.is_symlink()
        and child.name.lower().endswith((".nrrd", ".nhdr"))
        and "mask" not in child.name.lower()
    ]
    if len(nrrd_candidates) == 1:
        return ("nrrd", nrrd_candidates[0], resolved, _matching_sidecars(nrrd_candidates[0]))
    return None


def _source_signature(source: Path, extra: dict[str, Path]) -> dict[str, Any]:
    tracked = {"source": source, **extra}
    records: dict[str, Any] = {}
    for key, item in tracked.items():
        stat = item.stat()
        records[key] = {
            "path": item.resolve().as_posix(),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return records


def prepare_convertible_root(
    path: Path,
    subject_id: str,
    *,
    dwiconvert: Optional[Path] = None,
    mrconvert: Optional[Path] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Path:
    """Normalize a raw NIfTI or NRRD file/folder into ``<subject_root>/input`` if needed."""

    resolved = Path(path).expanduser().resolve()
    found = find_convertible_input(resolved)
    if found is None:
        return resolved if resolved.is_dir() else resolved.parent

    kind, source_file, subject_root, extra = found
    input_dir = subject_root / "input"
    receipt_path = input_dir / RECEIPT
    signature = _source_signature(source_file, extra)
    if (
        input_dir.is_dir()
        and all((input_dir / f).is_file() for f in REQUIRED)
        and receipt_path.is_file()
    ):
        try:
            saved = json.loads(receipt_path.read_text(encoding="utf-8"))
            if saved.get("kind") == kind and saved.get("signature") == signature:
                return subject_root
        except (OSError, ValueError, TypeError):
            pass

    from shared.input_import import InputImportConfig, import_dwi_input

    if log:
        log(f"[Input] Preparing {kind.upper()} input '{source_file.name}' into {input_dir}")
    import_dwi_input(
        InputImportConfig(
            source=source_file,
            subject_root=subject_root,
            subject_id=subject_id,
            source_kind=kind,  # type: ignore[arg-type]
            dwiconvert_executable=dwiconvert,
            mrconvert_executable=mrconvert,
            force=True,
            **extra,
        )
    )
    receipt_path.write_text(
        json.dumps({"kind": kind, "signature": signature}, indent=2) + "\n",
        encoding="utf-8",
    )
    return subject_root
