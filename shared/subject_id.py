"""Derive a safe subject ID from the chosen input (C6)."""

import re
from pathlib import Path

_FILE_SUFFIXES = (".nii.gz", ".nii", ".nrrd", ".nhdr")
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def derive_subject_id(input_text: str) -> str:
    """Name the subject after the input.

    * a subject root folder (canonical layout) or a DICOM folder: the folder name;
    * a bare DWI file: the file name without ``.nii`` / ``.nii.gz`` (and other volume suffixes).

    Characters outside ``[A-Za-z0-9._-]`` become ``_``. Empty input gives ``""``; the result is never ``.`` or ``..``.
    """

    text = (input_text or "").strip()
    if not text:
        return ""
    path = Path(text).expanduser()
    name = path.name or path.resolve().name
    if name in (".", ".."):  # Q6: a dot name is no subject ID; use the resolved folder name
        name = path.resolve().name
    if name in (".", ".."):
        return ""
    if not path.is_dir():
        lowered = name.lower()
        for suffix in _FILE_SUFFIXES:
            if lowered.endswith(suffix):
                name = name[: -len(suffix)]
                break
    name = _UNSAFE.sub("_", name)
    return "" if name in (".", "..") else name  # Q6: again after the suffix strip and sanitising


def effective_subject_id(typed_id: str, input_text: str) -> str:
    """The typed ID when present, else the ID derived from the input path."""

    return (typed_id or "").strip() or derive_subject_id(input_text)
