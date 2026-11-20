"""Input preview (H1): resolve the current DWI and show a light mean-b0 volume in Slicer.

Everything except :func:`show_preview_in_scene` runs without Slicer, so it can be unit
tested. Heavy imports (numpy, nibabel, pynrrd, ``shared.pipeline``) happen inside the
functions that need them. Discovery reuses ``shared.pipeline.inspect_subject_directory``
(the rule that input validation uses) and ``shared.convertible_input.find_convertible_input``;
this module adds no new discovery rule.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional

B0_THRESHOLD = 50.0
PREVIEW_ATTRIBUTE = "SlicerDDSurfer.InputPreview"
MAX_CACHED_PREVIEWS = 6
_CACHE_VERSION = "v2"
# O1: a NIfTI whose voxel data is smaller than this is read in one pass; a larger one is read volume by volume.
_ONE_PASS_BYTES = 1 << 30
# O2: a NRRD is read whole, so a header that announces more voxel data than this is refused.
MAX_NRRD_BYTES = 4 << 30
_MAX_PAYLOAD_FILES = 4096

# (English, Chinese) text per error code. ``{detail}`` is filled from PreviewError.detail.
_MESSAGES: dict[str, tuple[str, str]] = {
    "no_input": ("Select an input first.", "请先选择输入。"),
    "missing": ("No readable DWI was found at the selected input.", "在所选输入中找不到可读取的 DWI。"),
    "dicom_unconverted": ("Convert inputs first (Prepare Inputs)", "请先转换输入（准备输入）"),
    "no_dwi": (
        "This input has no DWI to preview (registered DTI layout).",
        "此输入不含可预览的 DWI（已配准的 DTI 布局）。",
    ),
    "no_bval": ("The b-value file (bval) was not found.", "找不到 b 值文件（bval）。"),
    "no_b0": ("No volume with b <= {detail} was found.", "没有 b <= {detail} 的体积。"),
    "batch_empty": (
        "No subject folder with a readable input was found in the batch folder.",
        "批量文件夹中没有含可读取输入的受试者文件夹。",
    ),
    "missing_dependency": (
        "The input preview needs the Python package {detail}, which is not installed.",
        "预览输入需要 Python 包 {detail}，但尚未安装。",
    ),
    "bad_data": ("{detail}", "{detail}"),
}


class PreviewError(Exception):
    """Expected, user-facing reason why no preview can be made."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = str(detail)

    def text(self, zh: bool = False) -> str:
        english, chinese = _MESSAGES.get(self.code, ("{detail}", "{detail}"))
        return (chinese if zh else english).format(detail=self.detail)


@dataclass(frozen=True)
class PreviewSource:
    """The DWI (and optional bval/mask) that a preview reads."""

    kind: str  # "nifti" or "nrrd"
    dwi: Path
    bval: Optional[Path]
    mask: Optional[Path]
    subject_id: str
    origin: str = ""
    raw: Optional[Path] = None  # C1: the file chosen in the 'Detached data' tile, which replaces the NHDR's own data file


# --------------------------------------------------------------------------- discovery


def _default_inspect(path: Path, subject_id: str) -> dict[str, Any]:
    from shared.pipeline import inspect_subject_directory

    return inspect_subject_directory(Path(path), subject_id)


def _user_mask(input_dir: Path) -> Optional[Path]:
    """The user-supplied native mask (same test as ``shared.pipeline._is_user_supplied_mask``)."""

    mask = input_dir / "nodif_brain_mask.nii.gz"
    if mask.is_file() and not mask.with_name("nodif_brain_mask.manifest.json").is_file():
        return mask
    return None


def _find_convertible(candidate: Path, subject_id: str):
    """Mirror the lookup order of ``inspect_subject_directory`` for raw NIfTI/NRRD."""

    from shared.convertible_input import find_convertible_input
    from shared.pipeline import _safe_subject_subdir

    expanded = Path(candidate).expanduser().resolve()
    found = find_convertible_input(expanded)
    if found is None and expanded.is_dir():
        subject_dir = _safe_subject_subdir(expanded, subject_id)
        if subject_dir is not None and subject_dir.is_dir():
            found = find_convertible_input(subject_dir)
    return found


def resolve_preview_source(inspected: Mapping[str, Any], candidate: Path, subject_id: str) -> PreviewSource:
    """Turn an ``inspect_subject_directory`` result into a preview source."""

    mode = str(inspected.get("mode") or "")
    if mode == "dicom":
        raise PreviewError("dicom_unconverted")
    if mode == "legacy_registered":
        raise PreviewError("no_dwi")
    if mode == "subject_root":
        root = Path(str(inspected["subject_root"]))
        input_dir = root / "input"
        return PreviewSource(
            kind="nifti",
            dwi=input_dir / "data.nii.gz",
            bval=input_dir / "bvals",
            mask=_user_mask(input_dir),
            subject_id=subject_id or root.name,
            origin=str(root),
        )
    if mode in {"nifti", "nrrd"}:
        found = _find_convertible(Path(candidate), subject_id)
        if found is None:
            raise PreviewError("missing")
        kind, source, folder, extra = found
        return PreviewSource(
            kind=kind,
            dwi=Path(source),
            bval=Path(extra["bval_path"]) if extra.get("bval_path") else None,
            mask=Path(extra["mask_path"]) if extra.get("mask_path") else None,
            subject_id=subject_id or Path(folder).name,
            origin=str(folder),
        )
    raise PreviewError("missing")


def resolve_pending_source(
    kind: str,
    source: Any,
    *,
    bval: Any = None,
    mask: Any = None,
    raw: Any = None,
    subject_id: str = "",
) -> PreviewSource:
    """Preview a Prepare-Inputs tile source that has not been imported yet.

    C1: for an NHDR header, ``raw`` is the file chosen in the 'Detached data' tile. The import replaces the header's
    own ``data file`` with it, so the preview does the same.
    """

    kind = (kind or "").strip().lower()
    if kind == "dicom":
        raise PreviewError("dicom_unconverted")
    path = Path(str(source)).expanduser()
    if not path.is_file():
        raise PreviewError("missing")
    mask_path = Path(str(mask)).expanduser() if mask else None
    if mask_path is not None and not mask_path.is_file():
        mask_path = None
    name = subject_id or path.name.split(".")[0]
    if kind == "nrrd" or path.name.lower().endswith((".nrrd", ".nhdr")):
        raw_path = None
        if raw and path.suffix.lower() == ".nhdr":
            # R3-2: resolve like Prepare Inputs does (expanduser, then resolve against the working directory), so the
            # staged header in its temp directory still finds the file.
            raw_path = Path(str(raw)).expanduser().resolve()
            if not raw_path.is_file():
                raise PreviewError("bad_data", "The selected detached data file was not found.")
        return PreviewSource("nrrd", path, None, mask_path, name, str(path.parent), raw_path)
    bval_path = Path(str(bval)).expanduser() if bval else None
    if bval_path is None or not bval_path.is_file():
        from shared.convertible_input import find_convertible_input

        found = find_convertible_input(path)
        bval_path = Path(found[3]["bval_path"]) if found and found[3].get("bval_path") else None
    if bval_path is None:
        raise PreviewError("no_bval")
    return PreviewSource("nifti", path, bval_path, mask_path, name, str(path.parent))


def check_preview_dependencies(source: Optional[PreviewSource] = None, find_spec: Callable[[str], Any] = None) -> None:
    """O11: raise ``missing_dependency`` when nibabel (and pynrrd, for a NRRD source) cannot be imported."""

    find_spec = find_spec or importlib.util.find_spec
    needed = ["nibabel"] + (["pynrrd"] if source is not None and source.kind == "nrrd" else [])
    for package in needed:
        module = "nrrd" if package == "pynrrd" else package
        try:
            found = find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            raise PreviewError("missing_dependency", package)


def first_batch_source(
    batch_root: Any, inspect: Optional[Callable[[Path, str], Mapping[str, Any]]] = None
) -> PreviewSource:
    """First previewable subject of a batch, in the order ``scripts/run_batch_pipeline.discover_subjects`` uses.

    O8: a subject that cannot be previewed (unconverted DICOM, registered DTI layout) is skipped. When no subject can
    be previewed, the reason of the first skipped subject is raised.
    """

    inspect = inspect or _default_inspect
    text = str(batch_root or "").strip()
    root = Path(text).expanduser().resolve() if text else None
    if root is None or not root.is_dir():
        raise PreviewError("no_input")
    first_error: Optional[PreviewError] = None
    for candidate in sorted(root.glob("*")):
        if not candidate.is_dir():
            continue
        base = candidate
        inspected = inspect(candidate, candidate.name)
        if inspected.get("mode") == "missing":
            base = root
            inspected = inspect(root, candidate.name)
        if inspected.get("mode") == "missing":
            continue
        try:
            return resolve_preview_source(inspected, base, candidate.name)
        except PreviewError as exc:
            first_error = first_error or exc
    raise first_error or PreviewError("batch_empty")


def resolve_preview_for_ui(
    *,
    batch_mode: bool,
    batch_root: Any = "",
    input_dir: Any = "",
    subject_id: str = "",
    pending: Optional[Mapping[str, Any]] = None,
    inspect: Optional[Callable[[Path, str], Mapping[str, Any]]] = None,
) -> PreviewSource:
    """Resolve the current subject's DWI the way input validation sees it."""

    inspect = inspect or _default_inspect
    if batch_mode:
        return first_batch_source(batch_root, inspect)
    if pending:
        return resolve_pending_source(
            str(pending.get("kind") or ""),
            pending.get("source"),
            bval=pending.get("bval"),
            mask=pending.get("mask"),
            raw=pending.get("raw"),
            subject_id=subject_id,
        )
    text = str(input_dir or "").strip()
    if not text:
        raise PreviewError("no_input")
    candidate = Path(text).expanduser()
    if not (candidate.is_dir() or candidate.is_file()):
        raise PreviewError("missing")
    return resolve_preview_source(inspect(candidate, subject_id), candidate, subject_id)


# --------------------------------------------------------------------------- mean b0


def read_bvals(path: Any) -> list[float]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    try:
        return [float(token) for token in text.replace(",", " ").split()]
    except ValueError as exc:
        raise PreviewError("bad_data", f"The bval file has a non-numeric entry: {exc}") from exc


def _space_signs(space: Any) -> tuple[float, float, float]:
    """Signs that turn a NRRD ``space`` into RAS (NIfTI) coordinates."""

    text = str(space or "").strip().lower()
    if not text:
        raise PreviewError("bad_data", "The NRRD header has no 'space' field.")
    letters = "".join(part[0] for part in text.split("-")) if "-" in text else text
    if len(letters) != 3 or set(letters) - set("rlapsi"):
        raise PreviewError("bad_data", f"Unsupported NRRD space: {space}")
    signs = []
    for letter, positive in zip(letters, "ras"):
        signs.append(1.0 if letter == positive else -1.0)
    return signs[0], signs[1], signs[2]


def _nrrd_diffusion_axis(header: Mapping[str, Any], ndim: int) -> int:
    import numpy as np

    directions = header.get("space directions")
    candidates: list[int] = []
    if directions is not None:
        for index, row in enumerate(directions):
            if row is None or not np.all(np.isfinite(np.asarray(row, dtype=float))):
                candidates.append(index)
    if not candidates:
        kinds = [str(kind).lower() for kind in header.get("kinds", [])]
        candidates = [i for i, kind in enumerate(kinds) if kind in {"list", "vector", "covariant-vector", "none"}]
    if len(candidates) != 1:
        raise PreviewError("bad_data", "Could not tell which NRRD axis holds the diffusion volumes.")
    return candidates[0]


def _nrrd_bvalues(header: Mapping[str, Any], count: int) -> list[float]:
    try:
        b_max = float(str(header["DWMRI_b-value"]).split()[0])
    except (KeyError, ValueError, IndexError) as exc:
        raise PreviewError("bad_data", "The NRRD header has no usable DWMRI_b-value.") from exc
    values = []
    for index in range(count):
        raw = header.get("DWMRI_gradient_%04d" % index)
        if raw is None:
            raise PreviewError("bad_data", "The NRRD header has no DWMRI_gradient_%04d entry." % index)
        try:
            vector = [float(token) for token in str(raw).split()]
        except ValueError as exc:
            raise PreviewError("bad_data", "DWMRI_gradient_%04d is not numeric." % index) from exc
        values.append(b_max * sum(component * component for component in vector))
    return values


def _mean_of(volumes: Any, indices: list[int], threshold: float):
    import numpy as np

    if not indices:
        raise PreviewError("no_b0", f"{threshold:g}")
    total = None
    for index in indices:
        volume = np.asarray(volumes(index), dtype=np.float32)
        if total is None:
            total = volume.copy()
        else:
            total += volume
    return total / np.float32(len(indices))


def _mean_b0_nifti(source: PreviewSource, threshold: float):
    import nibabel as nib
    import numpy as np

    if source.bval is None or not Path(source.bval).is_file():
        raise PreviewError("no_bval")
    bvals = read_bvals(source.bval)
    # O1: keep_file_open stops a .nii.gz from being re-opened (and decompressed from byte 0) for every volume.
    image = nib.load(str(source.dwi), keep_file_open=True)
    if len(image.shape) != 4:
        raise PreviewError("bad_data", f"The DWI is {len(image.shape)}D; a 4D volume series is required.")
    if len(bvals) != image.shape[3]:
        raise PreviewError(
            "bad_data", f"The bval file has {len(bvals)} entries but the DWI has {image.shape[3]} volumes."
        )
    indices = [index for index, value in enumerate(bvals) if value <= threshold]  # ascending: one forward pass
    voxel_bytes = int(np.prod(image.shape, dtype=np.int64)) * int(image.get_data_dtype().itemsize)
    if voxel_bytes < _ONE_PASS_BYTES:
        data = np.asanyarray(image.dataobj)  # small: read the file once, then slice
        mean = _mean_of(lambda i: data[..., i], indices, threshold)
        del data
    else:
        mean = _mean_of(lambda i: image.dataobj[..., i], indices, threshold)
    header = image.header.copy()
    header.set_data_dtype(np.float32)
    header.set_data_shape(mean.shape)
    affine = np.array(image.affine)
    del image  # closes the file that keep_file_open holds
    return nib.Nifti1Image(mean, affine, header), len(indices)


_NRRD_ITEM_BYTES = {
    "int8": 1, "signed char": 1, "uint8": 1, "uchar": 1, "unsigned char": 1,
    "int16": 2, "short": 2, "short int": 2, "signed short": 2, "signed short int": 2,
    "uint16": 2, "ushort": 2, "unsigned short": 2, "unsigned short int": 2,
    "int32": 4, "int": 4, "signed int": 4, "uint32": 4, "uint": 4, "unsigned int": 4,
    "float": 4,
    "int64": 8, "longlong": 8, "long long": 8, "long long int": 8, "signed long long": 8,
    "signed long long int": 8, "uint64": 8, "ulonglong": 8, "unsigned long long": 8,
    "unsigned long long int": 8, "double": 8,
}
_DATA_FILE_PATTERN = r"^data\s*file\s*:\s*(.+)$"
_MAX_NRRD_HEADER_BYTES = 64 << 10


def _nrrd_announced_bytes(header: Mapping[str, Any]) -> int:
    """Bytes of voxel data that a NRRD header announces (an unknown type counts as 8 bytes per voxel)."""

    count = 1
    for size in header.get("sizes", []):
        count *= int(size)
    return count * _NRRD_ITEM_BYTES.get(str(header.get("type", "")).strip().lower(), 8)


def stage_detached_header(header_path: Any, raw: Any, directory: Any) -> Path:
    """C1: write a copy of an NHDR whose single ``data file`` is ``raw`` (the rule ``import_detached_nrrd`` applies)."""

    header_path, raw = Path(header_path), Path(raw)
    if header_path.suffix.lower() != ".nhdr" or not raw.is_file():
        raise PreviewError("bad_data", "Select an NHDR header and an existing detached data file.")
    header = header_path.read_text(encoding="ascii", errors="replace")
    match = re.search(_DATA_FILE_PATTERN, header, re.IGNORECASE | re.MULTILINE)
    if match is None or match.group(1).upper().startswith("LIST") or "%" in match.group(1):
        raise PreviewError("bad_data", "A single detached data file is required for this header override.")
    staged = Path(directory) / header_path.name
    raw_text = raw.resolve().as_posix()
    try:
        raw_text.encode("ascii")
    except UnicodeEncodeError:
        raise PreviewError(
            "bad_data",
            "The detached data path contains non-ASCII characters; move the file to an ASCII-only folder.",
        ) from None
    header = re.sub(_DATA_FILE_PATTERN, lambda _: f"data file: {raw_text}", header,
                    count=1, flags=re.IGNORECASE | re.MULTILINE)
    staged.write_text(header, encoding="ascii", errors="replace")
    return staged


@contextlib.contextmanager
def _nrrd_read_target(source: PreviewSource) -> Iterator[str]:
    if source.raw is None:
        yield str(source.dwi)
        return
    with tempfile.TemporaryDirectory(prefix="slicerdspace-nhdr-") as temporary:
        yield str(stage_detached_header(source.dwi, source.raw, temporary))


def nrrd_payload_files(path: Any, raw: Any = None) -> list:
    """C2: the data files that a detached NRRD header points to (``raw`` replaces them); empty for attached data."""

    if raw:
        return [Path(raw)]
    path = Path(path)
    # R3-1: a .nrrd header can name detached data too ('data file'), so every extension is inspected. An attached
    # .nrrd is read only up to its first blank line (the header end) and at most _MAX_NRRD_HEADER_BYTES.
    is_nhdr = path.suffix.lower() == ".nhdr"
    try:
        with open(str(path), "rb") as handle:
            chunk = handle.read((1 << 20) if is_nhdr else _MAX_NRRD_HEADER_BYTES)
    except OSError:
        return []
    if not is_nhdr:
        blank = re.search(rb"\r?\n\r?\n", chunk)
        if blank is not None:
            chunk = chunk[:blank.start() + 1]
    text = chunk.decode("ascii", errors="replace")
    match = re.search(_DATA_FILE_PATTERN, text, re.IGNORECASE | re.MULTILINE)
    if match is None:
        return []
    value = match.group(1).strip()
    if value.upper().startswith("LIST"):
        names = [line.strip() for line in text[match.end():].splitlines() if line.strip() and not line.startswith("#")]
    elif "%" in value:
        parts = value.split()
        try:
            low, high, step = int(parts[1]), int(parts[2]), int(parts[3])
            names = [parts[0] % number for number in range(low, high + 1, step or 1)]
        except (ValueError, IndexError, TypeError):
            names = []
    else:
        names = [value]
    return [path.parent / name for name in names[:_MAX_PAYLOAD_FILES]]


def _mean_b0_nrrd(source: PreviewSource, threshold: float):
    import nibabel as nib
    import numpy as np

    try:
        import nrrd
    except ImportError as exc:
        raise PreviewError("bad_data", "Reading NRRD needs pynrrd, which is not installed.") from exc
    try:
        with _nrrd_read_target(source) as target:
            # O2: read the header first; the whole 4D array is loaded below, so refuse a huge one.
            announced = _nrrd_announced_bytes(nrrd.read_header(target))
            if announced > MAX_NRRD_BYTES:
                raise PreviewError(
                    "bad_data",
                    "The NRRD holds about %.1f GiB of voxel data; the preview reads at most %d GiB. "
                    "Convert the input to NIfTI first." % (announced / (1 << 30), MAX_NRRD_BYTES >> 30),
                )
            data, header = nrrd.read(target)
    except PreviewError:
        raise
    except Exception as exc:
        raise PreviewError("bad_data", f"Could not read the NRRD file: {exc}") from exc
    if data.ndim != 4:
        raise PreviewError("bad_data", f"The NRRD is {data.ndim}D; a 4D diffusion series is required.")
    axis = _nrrd_diffusion_axis(header, data.ndim)
    bvals = _nrrd_bvalues(header, data.shape[axis])
    indices = [index for index, value in enumerate(bvals) if value <= threshold]
    volumes = np.moveaxis(data, axis, -1)
    mean = _mean_of(lambda i: volumes[..., i], indices, threshold)
    directions = np.asarray(header.get("space directions"), dtype=float)
    spatial = np.delete(directions, axis, axis=0)
    if spatial.shape != (3, 3) or not np.all(np.isfinite(spatial)):
        raise PreviewError("bad_data", "The NRRD 'space directions' field is not usable.")
    origin = np.asarray(header.get("space origin", [0.0, 0.0, 0.0]), dtype=float)
    affine = np.eye(4)
    affine[:3, :3] = spatial.T
    affine[:3, 3] = origin
    affine = np.diag(list(_space_signs(header.get("space"))) + [1.0]) @ affine
    return nib.Nifti1Image(mean, affine), len(indices)


def _stamp(path: Optional[Path]) -> str:
    if path is None:
        return "-"
    try:
        stat = Path(path).stat()
    except OSError:
        return f"{path}:missing"
    return f"{Path(path).resolve()}:{stat.st_size}:{stat.st_mtime_ns}"


def preview_cache_path(source: PreviewSource, cache_dir: Any, threshold: float = B0_THRESHOLD) -> Path:
    parts = [_CACHE_VERSION, _stamp(source.dwi), _stamp(source.bval), f"{threshold:g}"]
    if source.kind == "nrrd":
        # C2: a detached NRRD header says nothing about its data, so the data files are part of the key.
        parts.extend(_stamp(payload) for payload in nrrd_payload_files(source.dwi, source.raw))
    key = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in (source.subject_id or "subject"))[:48]
    return Path(cache_dir) / f"preview_{safe}_{key}_meanb0.nii.gz"


def _prune_cache(cache_dir: Path, keep: Path) -> None:
    entries = sorted(cache_dir.glob("preview_*_meanb0.nii.gz"), key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in entries[MAX_CACHED_PREVIEWS:]:
        if stale != keep:
            try:
                stale.unlink()
            except OSError:
                pass


def mean_b0_image(source: PreviewSource, cache_dir: Any, threshold: float = B0_THRESHOLD) -> Path:
    """Write (or reuse) the mean of all volumes with b <= ``threshold`` as a 3D NIfTI.

    The file goes into ``cache_dir`` and never into the subject's input folder.
    """

    cache = Path(cache_dir)
    target = preview_cache_path(source, cache, threshold)
    if target.is_file() and target.stat().st_size > 0:
        os.utime(target, None)
        return target
    if not Path(source.dwi).is_file():
        raise PreviewError("missing")
    cache.mkdir(parents=True, exist_ok=True)
    staged = target.with_name("tmp_" + target.name)
    try:
        if source.kind == "nrrd":
            image, _count = _mean_b0_nrrd(source, threshold)
        else:
            image, _count = _mean_b0_nifti(source, threshold)
        import nibabel as nib

        nib.save(image, str(staged))
        os.replace(staged, target)
    except PreviewError:
        raise
    except Exception as exc:
        raise PreviewError("bad_data", f"Could not build the mean b0 image: {exc}") from exc
    finally:
        # O6: never leave a half-written staging file in the cache.
        try:
            staged.unlink()
        except OSError:
            pass
    try:
        _prune_cache(cache, target)
    except OSError:
        pass
    return target


# --------------------------------------------------------------------------- Slicer side


def _remove_previous_preview_nodes(slicer: Any) -> None:
    scene = slicer.mrmlScene
    try:
        nodes = list(slicer.util.getNodesByClass("vtkMRMLVolumeNode"))
    except Exception:
        nodes = []
    for node in nodes:
        if node.GetAttribute(PREVIEW_ATTRIBUTE):
            scene.RemoveNode(node)


def _fit_slice_views(slicer: Any, background_id: str, label_id: Optional[str]) -> None:
    # C3: foreground=None clears the layer; the default 'keep-current' would let an old foreground volume hide the preview.
    layers = {"background": background_id, "foreground": None, "label": label_id}
    if label_id:
        layers["labelOpacity"] = 0.35
    try:
        slicer.util.setSliceViewerLayers(**layers, fit=True)
    except Exception:
        # No layout manager (headless): set the composite nodes directly.
        for composite in slicer.util.getNodesByClass("vtkMRMLSliceCompositeNode"):
            composite.SetBackgroundVolumeID(background_id)
            composite.SetForegroundVolumeID(None)
            composite.SetLabelVolumeID(label_id or "")
            if label_id:
                composite.SetLabelOpacity(0.35)
    try:
        slicer.util.resetSliceViews()
    except Exception:
        pass
    try:
        node = slicer.mrmlScene.GetNodeByID(background_id)
        bounds = [0.0] * 6
        node.GetRASBounds(bounds)
        center = [(bounds[index] + bounds[index + 1]) / 2 for index in (0, 2, 4)]
        slicer.vtkMRMLSliceNode.JumpAllSlices(
            slicer.mrmlScene, *center, slicer.vtkMRMLSliceNode.CenteredJumpSlice
        )
    except Exception:
        pass
    try:
        manager = slicer.app.layoutManager()
        if manager is not None and manager.layout == slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView:
            manager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutConventionalView)
    except Exception:
        pass


def _first_node(loaded: Any) -> Any:
    """The node that ``slicer.util.loadVolume`` returns (a list when a file holds several)."""

    if isinstance(loaded, (list, tuple)):
        return loaded[0] if loaded else None
    return loaded


def show_preview_in_scene(
    slicer: Any,
    mean_b0: Any,
    mask: Any = None,
    *,
    subject_id: str = "",
    log: Optional[Callable[[str], None]] = None,
) -> dict[str, Optional[str]]:
    """Load the mean b0 (and a brain mask as label map) and show them in the slice views."""

    say = log or (lambda _message: None)
    _remove_previous_preview_nodes(slicer)
    name = f"{subject_id or 'subject'}_input"
    # O10: loadVolume returns the node and raises on failure (returnNode=True is deprecated).
    try:
        volume = _first_node(slicer.util.loadVolume(str(mean_b0), properties={"name": f"{name}_meanb0", "show": False}))
    except Exception as exc:
        raise PreviewError("bad_data", f"Slicer could not load {mean_b0}: {exc}") from exc
    if volume is None:
        raise PreviewError("bad_data", f"Slicer could not load {mean_b0}")
    volume.SetAttribute(PREVIEW_ATTRIBUTE, "meanb0")
    label = None
    if mask:
        try:
            label = _first_node(
                slicer.util.loadLabelVolume(str(mask), properties={"name": f"{name}_mask", "show": False})
            )
            if label is None:
                say(f"[Input preview] The brain mask could not be loaded as a label map: {mask}")
            else:
                label.SetAttribute(PREVIEW_ATTRIBUTE, "mask")
        except Exception as exc:
            label = None
            say(f"[Input preview] The brain mask could not be loaded: {exc}")
    _fit_slice_views(slicer, volume.GetID(), label.GetID() if label is not None else None)
    return {"volume_id": volume.GetID(), "mask_id": label.GetID() if label is not None else None}
