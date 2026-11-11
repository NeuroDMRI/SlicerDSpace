"""Conservative defaults for the subject-folder picker, without data edits."""

from pathlib import Path


def subject_folder_defaults(path, subject_id="", *, batch=False):
    root = Path(path).expanduser() if path else None
    if root is None:
        return path or "", subject_id
    if root.is_file():
        lower = root.name.lower()
        if (
            not batch
            and root.parent.name == "input"
            and all((root.parent / name).is_file() for name in ("data.nii.gz", "bvals", "bvecs"))
        ):
            canon_root = root.parent.parent
            inferred = subject_id if str(subject_id).strip() else canon_root.name
            return str(canon_root), inferred
        if lower.endswith((".nrrd", ".nhdr", ".nii", ".nii.gz")):
            inferred = subject_id
            if not batch and not str(subject_id).strip():
                stem = root.name[:-7] if lower.endswith(".nii.gz") else root.stem
                parent_name = root.parent.name
                inferred = parent_name if parent_name and parent_name.lower() not in {"input", "dwi", "data"} else stem
            return str(root), inferred
        from shared.dicom_input import _dicom_file

        if _dicom_file(root):
            dicom_dir = root.parent
            inferred = subject_id if (batch or str(subject_id).strip()) else dicom_dir.name
            return str(dicom_dir), inferred
    if not root.is_dir():
        return path or "", subject_id
    if not batch and root.name == "input" and all((root / name).is_file() for name in ("data.nii.gz", "bvals", "bvecs")):
        root = root.parent
    inferred = subject_id
    if not batch and not str(subject_id).strip():
        inferred = root.name
    return str(root), inferred
