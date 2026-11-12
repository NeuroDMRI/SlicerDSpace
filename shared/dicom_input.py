"""Conservative, transactional normalization of an explicitly selected DICOM root."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

REQUIRED = ("data.nii.gz", "bvals", "bvecs")
RECEIPT = "auto_dicom_import.json"
IGNORED = {".git", "cache", "outputs", "__pycache__"}


def _dicom_file(path):
    if path.is_symlink() or not path.is_file():
        return False
    try:
        with path.open("rb") as stream:
            if stream.read(132)[128:] == b"DICM":
                return True
        if path.suffix.lower() not in {".dcm", ".ima", ""}:
            return False
        import pydicom
        header = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=["SOPClassUID", "SeriesInstanceUID"])
        return bool(header.get("SOPClassUID") and header.get("SeriesInstanceUID"))
    except Exception:
        return False


def _files(root, max_folders=None):
    for index, (folder, directories, names) in enumerate(os.walk(root, followlinks=False)):
        if max_folders is not None and index >= max_folders:
            return
        directories[:] = sorted(name for name in directories if name not in IGNORED and
                                not name.startswith(".dds-dicom-") and not (Path(folder) / name).is_symlink())
        if Path(folder) == root:
            directories[:] = [name for name in directories if not (
                name == "input" and all((root / "input" / file).is_file() for file in REQUIRED))]
        for name in sorted(names):
            path = Path(folder) / name
            if path.suffix.lower() not in {".json", ".txt", ".log", ".csv", ".gz", ".nii", ".nrrd", ".nhdr", ".bval", ".bvec"} and not path.is_symlink():
                yield path


def scan_series(root):
    """Read DICOM headers and group instance paths by Series Instance UID."""
    import pydicom
    from pydicom.errors import InvalidDicomError

    tags = ["SeriesInstanceUID", "SeriesDescription", "SeriesNumber",
            "ProtocolName", "AcquisitionTime", "NumberOfFrames"]
    series = {}
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"DICOM source is not a directory: {root}")
    for path in _files(root):
        try:
            header = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=tags)
        except InvalidDicomError:
            continue
        uid = str(header.get("SeriesInstanceUID") or "")
        if not uid:
            continue
        entry = series.setdefault(uid, {
            "uid": uid,
            "description": str(header.get("SeriesDescription") or header.get("ProtocolName") or uid),
            "number": str(header.get("SeriesNumber") or ""),
            "acquisition_time": str(header.get("AcquisitionTime") or ""),
            "frames": 0,
            "paths": [],
        })
        entry["frames"] += int(header.get("NumberOfFrames") or 1)
        entry["paths"].append(path)
    return tuple(series.values())


def is_dicom_root(root):
    root = Path(root).expanduser()
    if not root.is_dir() or all((root / "input" / name).is_file() for name in REQUIRED):
        return False
    for index, path in enumerate(_files(root, max_folders=256)):
        if index >= 2048:
            break
        if _dicom_file(path):
            return True
    return False


def _inventory(root):
    return [{"path": path.relative_to(root).as_posix(), "size": path.stat().st_size,
             "mtime_ns": path.stat().st_mtime_ns} for path in _files(root) if _dicom_file(path)]


def _sha(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def converter_path(explicit=None):
    explicit = explicit or os.environ.get("SLICERDSPACE_DWICONVERT_EXECUTABLE")
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise FileNotFoundError("Configured DWIConvert is not executable: " + str(path))
        return "dwiconvert", path
    from shared.input_import import find_dcm2niix
    found = find_dcm2niix()  # B6: the installed dcm2niix package before PATH; DWIConvert stays the fallback
    if found:
        return "dcm2niix", Path(found)
    found = shutil.which("DWIConvert")
    if found:
        return "dwiconvert", Path(found)
    from tractography.ukf import resolve_executable
    try:
        return "dwiconvert", resolve_executable(name="DWIConvert")
    except RuntimeError as exc:
        raise FileNotFoundError("DICOM input needs dcm2niix or Slicer's DWIConvert; no conversion or analysis was run.") from exc


def _validate_converted(folder):
    import nibabel as nib
    import numpy as np
    candidates = []
    for image in folder.glob("*.nii*"):
        stem = image.name[:-7] if image.name.endswith(".nii.gz") else image.stem
        gradients = ((folder / "bvals", folder / "bvecs") if stem == "data" and
                     (folder / "bvals").is_file() else (folder / (stem + ".bval"), folder / (stem + ".bvec")))
        if all(path.is_file() for path in gradients):
            candidates.append((image, *gradients))
    if len(candidates) != 1 or candidates[0][0].name != "data.nii.gz":
        raise ValueError("DICOM conversion must yield exactly one unambiguous DWI series with gradients; select a single diffusion series directory.")
    image = nib.load(folder / "data.nii.gz")
    bvals, bvecs = (np.loadtxt(folder / name) for name in ("bvals", "bvecs"))
    if (len(image.shape) != 4 or image.shape[3] < 2 or bvals.size != image.shape[3] or
            bvecs.shape not in {(3, image.shape[3]), (image.shape[3], 3)} or
            not np.isfinite(image.affine).all() or not np.isfinite(bvals).all() or
            not np.isfinite(bvecs).all() or np.any(bvals < 0)):
        raise ValueError("Converted DWI geometry/volume count and finite bvals/bvecs do not agree")
    if not np.isfinite(np.asanyarray(image.dataobj)).all():
        raise ValueError("Converted DWI contains nonfinite values")


def prepare_dicom_root(root, subject_id, *, dwiconvert=None, log=None):
    root = Path(root).expanduser().resolve()
    target = root / "input"
    receipt = target / RECEIPT
    if receipt.is_file():
        saved = json.loads(receipt.read_text())
        if saved.get("kind") != "slicerdsurfer_auto_dicom_input" or saved.get("source_inventory") != _inventory(root):
            raise ValueError("DICOM source changed after conversion; preserve the old input and use a new subject root")
        if (not set(REQUIRED) <= set(saved.get("output_sha256", {})) or
                any(Path(name).name != name for name in saved["output_sha256"])):
            raise ValueError("Incomplete or unsafe DICOM conversion receipt")
        if any((target / name).is_symlink() or not (target / name).is_file() or _sha(target / name) != digest
               for name, digest in saved["output_sha256"].items()):
            raise ValueError("Previously converted DICOM input changed; refusing to reuse it")
        return saved
    if not is_dicom_root(root):
        return None
    if target.exists() or target.is_symlink():
        raise FileExistsError("DICOM import will not replace an existing input directory: " + str(target))
    original = _inventory(root)
    backend, executable = converter_path(dwiconvert)
    if log:
        log("[Input] Converting DICOM using " + backend + "; original files remain unchanged")
    from shared.input_import import InputImportConfig, prepare_inputs
    with tempfile.TemporaryDirectory(prefix=".dds-dicom-", dir=root.parent) as temporary:
        staging = Path(temporary)
        result = prepare_inputs(InputImportConfig(source=root, subject_root=staging, subject_id=subject_id,
            source_kind="dicom", backend=backend,
            dcm2niix_executable=executable if backend == "dcm2niix" else None,
            dwiconvert_executable=executable if backend == "dwiconvert" else None))
        prepared = staging / "input"
        _validate_converted(prepared)
        if _inventory(root) != original:
            raise ValueError("DICOM source changed during conversion; prepared input was not published")
        result["plan"]["subject_root"] = str(root)
        result["plan"]["canonical_layout"] = {key: str(root / Path(value).relative_to(staging))
                                              for key, value in result["plan"]["canonical_layout"].items()}
        result["outputs"] = {key: str(root / Path(value).relative_to(staging)) if value else None
                             for key, value in result["outputs"].items()}
        (prepared / "input_import_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
        record = {"kind": "slicerdsurfer_auto_dicom_input", "status": "completed", "backend": backend,
                  "source_inventory": original, "output_sha256": {
                      path.name: _sha(path) for path in prepared.iterdir() if path.is_file()},
                  "runtime_seconds": result["runtime_seconds"]}
        (prepared / RECEIPT).write_text(json.dumps(record, indent=2) + "\n")
        if target.exists() or target.is_symlink():
            raise FileExistsError("Input directory appeared during conversion; refusing to replace it")
        prepared.rename(target)
    return record
