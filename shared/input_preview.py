"""Input preview (H1): resolve the current DWI and show it in Slicer.

b2-dwi: 'View input' builds the whole DWI as a core ``vtkMRMLDiffusionWeightedVolumeNode`` in memory
(:func:`read_full_dwi` on the worker thread, :func:`show_dwi_in_scene` on the GUI thread). Above the size limit
(:func:`full_dwi_limit_bytes`), and when a NIfTI has no usable bvec, it falls back to the light mean-b0 volume
(:func:`mean_b0_image`). Batch mode has no preview (b2 inputs-help): the button is hidden there.

Everything except :func:`show_preview_in_scene` and :func:`show_dwi_in_scene` runs without Slicer, so it can be
unit tested. Heavy imports (numpy, nibabel, pynrrd, ``shared.pipeline``) happen inside the
functions that need them. Discovery reuses ``shared.pipeline.inspect_subject_directory``
(the rule that input validation uses) and ``shared.convertible_input.find_convertible_input``;
this module adds no new discovery rule.
"""

from __future__ import annotations

import contextlib
import gzip
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
# O2: a single 3D NRRD volume (or whole NRRD for non-streamable encodings / full DWI) above this limit is refused.
MAX_NRRD_BYTES = 4 << 30
_NRRD_STREAM_CHUNK_BYTES = 8 << 20
_MAX_PAYLOAD_FILES = 4096
# b2-dwi: the full DWI is built when its voxel data is at most min(FULL_DWI_MAX_BYTES, FULL_DWI_RAM_FRACTION * RAM).
FULL_DWI_MAX_BYTES = 4 << 30
FULL_DWI_RAM_FRACTION = 0.25
# b2-dwi: volumes read per block (measured: about 1.3x the voxel array at peak, versus about 3x for get_fdata).
_FILL_BLOCK = 10
_VTK_DTYPES = ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float32", "float64")

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
    "no_bvec": ("The gradient direction file (bvec) was not found.", "找不到梯度方向文件（bvec）。"),
    "no_b0": ("No volume with b <= {detail} was found.", "没有 b <= {detail} 的体积。"),
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
    bvec: Optional[Path] = None  # b2-dwi: FSL gradient directions of a NIfTI DWI (the full view needs them)


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
            bvec=input_dir / "bvecs",
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
            bvec=Path(extra["bvec_path"]) if extra.get("bvec_path") else None,
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
    bvec: Any = None,
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
    bvec_path = Path(str(bvec)).expanduser() if bvec else None
    if bvec_path is not None and not bvec_path.is_file():
        bvec_path = None
    if bval_path is None or not bval_path.is_file() or bvec_path is None:
        from shared.convertible_input import find_convertible_input

        found = find_convertible_input(path)
        extra = found[3] if found else {}
        if bval_path is None or not bval_path.is_file():
            bval_path = Path(extra["bval_path"]) if extra.get("bval_path") else None
        if bvec_path is None and extra.get("bvec_path"):
            bvec_path = Path(extra["bvec_path"])
    if bval_path is None:
        raise PreviewError("no_bval")
    # b2-dwi: a missing bvec is not a resolve error; the full-DWI reader raises 'no_bvec' (the size fallback needs none).
    return PreviewSource("nifti", path, bval_path, mask_path, name, str(path.parent), bvec=bvec_path)


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


def resolve_preview_for_ui(
    *,
    input_dir: Any = "",
    subject_id: str = "",
    pending: Optional[Mapping[str, Any]] = None,
    inspect: Optional[Callable[[Path, str], Mapping[str, Any]]] = None,
) -> PreviewSource:
    """Resolve the current subject's DWI the way input validation sees it."""

    inspect = inspect or _default_inspect
    if pending:
        return resolve_pending_source(
            str(pending.get("kind") or ""),
            pending.get("source"),
            bval=pending.get("bval"),
            mask=pending.get("mask"),
            raw=pending.get("raw"),
            subject_id=subject_id,
            bvec=pending.get("bvec"),
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


def non_ascii_data_path_message() -> str:
    """The plain message for a detached data path that an ASCII NHDR cannot hold (F5; shared with the import)."""

    return "The detached data path contains non-ASCII characters; move the file to an ASCII-only folder."


def _is_ascii(text: str) -> bool:
    try:
        text.encode("ascii")
    except UnicodeEncodeError:
        return False
    return True


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
    if not _is_ascii(raw_text):
        raise PreviewError("bad_data", non_ascii_data_path_message())
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


_NRRD_TYPEMAP_FALLBACK = {
    "signed char": "i1", "int8": "i1", "int8_t": "i1",
    "uchar": "u1", "unsigned char": "u1", "uint8": "u1", "uint8_t": "u1",
    "short": "i2", "short int": "i2", "signed short": "i2", "signed short int": "i2", "int16": "i2", "int16_t": "i2",
    "ushort": "u2", "unsigned short": "u2", "unsigned short int": "u2", "uint16": "u2", "uint16_t": "u2",
    "int": "i4", "signed int": "i4", "int32": "i4", "int32_t": "i4",
    "uint": "u4", "unsigned int": "u4", "uint32": "u4", "uint32_t": "u4",
    "longlong": "i8", "long long": "i8", "long long int": "i8", "signed long long": "i8",
    "signed long long int": "i8", "int64": "i8", "int64_t": "i8",
    "ulonglong": "u8", "unsigned long long": "u8", "unsigned long long int": "u8", "uint64": "u8", "uint64_t": "u8",
    "float": "f4",
    "double": "f8",
}


def _nrrd_numpy_dtype(header: Mapping[str, Any], nrrd_mod: Any):
    import numpy as np

    reader = getattr(nrrd_mod, "reader", None)
    determine = getattr(reader, "_determine_datatype", None)
    if callable(determine):
        return determine(header)
    type_key = str(header.get("type", "")).strip().lower()
    if type_key not in _NRRD_TYPEMAP_FALLBACK:
        raise PreviewError("bad_data", f"Unsupported NRRD data type: {header.get('type')!r}")
    code = _NRRD_TYPEMAP_FALLBACK[type_key]
    if np.dtype(code).itemsize > 1:
        endian = str(header.get("endian", "")).strip().lower()
        if endian == "big":
            code = ">" + code
        elif endian == "little":
            code = "<" + code
        else:
            raise PreviewError("bad_data", f"Unsupported or missing NRRD endianness: {header.get('endian')!r}")
    return np.dtype(code)


def _read_exact_bytes(stream: Any, n_bytes: int) -> bytes:
    if n_bytes <= 0:
        return b""
    chunks: list[bytes] = []
    remaining = int(n_bytes)
    while remaining > 0:
        part = stream.read(remaining)
        if not part:
            raise PreviewError("bad_data", "The NRRD data stream ended before all voxels were read.")
        chunks.append(part)
        remaining -= len(part)
    return chunks[0] if len(chunks) == 1 else b"".join(chunks)


def _skip_exact_bytes(stream: Any, n_bytes: int, chunk_bytes: int = 4 << 20) -> None:
    remaining = int(n_bytes)
    while remaining > 0:
        part = stream.read(min(remaining, chunk_bytes))
        if not part:
            raise PreviewError("bad_data", "The NRRD data stream ended before all voxels were read.")
        remaining -= len(part)


def _stream_mean_b0_nrrd_payload(
    header: Mapping[str, Any],
    target: str,
    attached_fh: Any,
    axis: int,
    indices: list[int],
    nrrd_mod: Any,
):
    """Stream-decode only the requested b0 volume(s) from raw or gzip NRRD/NHDR without allocating the 4D array.

    Returns the float32 3D mean-b0 array on success, or ``None`` when the header uses an encoding or
    multi-file layout that requires falling back to ``nrrd.read``.
    """
    import numpy as np

    encoding = str(header.get("encoding", "")).strip().lower()
    if encoding not in {"raw", "gzip", "gz"}:
        return None
    data_filename = header.get("datafile", header.get("data file", None))
    if data_filename is not None:
        if not isinstance(data_filename, str):
            return None
        stripped = data_filename.strip()
        if not stripped or stripped.upper().startswith("LIST") or "%" in stripped:
            return None
        data_filename = stripped
    line_skip = int(header.get("lineskip", header.get("line skip", 0)))
    byte_skip = int(header.get("byteskip", header.get("byte skip", 0)))
    if line_skip < 0 or byte_skip < -1:
        raise PreviewError("bad_data", "The NRRD header has an invalid line skip or byte skip.")
    if encoding != "raw" and byte_skip != 0:
        return None

    sizes = tuple(int(s) for s in header.get("sizes", []))
    spatial_shape = tuple(s for i, s in enumerate(sizes) if i != axis)
    ng = int(sizes[axis])
    inner = int(np.prod(sizes[:axis], dtype=np.int64))
    outer = int(np.prod(sizes[axis + 1 :], dtype=np.int64))
    dtype = _nrrd_numpy_dtype(header, nrrd_mod)
    itemsize = int(dtype.itemsize)
    total_voxels = int(np.prod(sizes, dtype=np.int64))

    with contextlib.ExitStack() as stack:
        if data_filename is not None:
            data_path = (
                data_filename
                if os.path.isabs(data_filename)
                else os.path.join(os.path.dirname(target), data_filename)
            )
            data_fh = stack.enter_context(open(data_path, "rb"))
        else:
            data_fh = attached_fh

        for _ in range(line_skip):
            data_fh.readline()
        if byte_skip > 0:
            data_fh.seek(byte_skip, os.SEEK_CUR)
        elif byte_skip == -1:
            data_fh.seek(-total_voxels * itemsize, os.SEEK_END)

        if outer == 1:
            # Gradient axis is last (sizes = X, Y, Z, G): each 3D volume is one contiguous block in Fortran order.
            vol_bytes = inner * itemsize
            total = np.zeros(spatial_shape, dtype=np.float32)
            if encoding == "raw":
                base_offset = data_fh.tell()
                data_fh.seek(0, os.SEEK_END)
                if data_fh.tell() < base_offset + ng * vol_bytes:
                    raise PreviewError("bad_data", "The NRRD data file is shorter than the header dimensions.")
                for g in indices:
                    data_fh.seek(base_offset + int(g) * vol_bytes, os.SEEK_SET)
                    raw_bytes = _read_exact_bytes(data_fh, vol_bytes)
                    vol = np.frombuffer(raw_bytes, dtype=dtype).reshape(spatial_shape, order="F")
                    total += vol.astype(np.float32, copy=False)
            else:
                gz = stack.enter_context(gzip.GzipFile(fileobj=data_fh, mode="rb"))
                last_idx = max(indices)
                wanted = set(indices)
                for g in range(last_idx + 1):
                    if g in wanted:
                        raw_bytes = _read_exact_bytes(gz, vol_bytes)
                        vol = np.frombuffer(raw_bytes, dtype=dtype).reshape(spatial_shape, order="F")
                        total += vol.astype(np.float32, copy=False)
                    else:
                        _skip_exact_bytes(gz, vol_bytes)
            return total / np.float32(len(indices))

        # Interleaved gradient axis (e.g. axis == 0, sizes = G, X, Y, Z): stream bounded outer slabs.
        stream = data_fh if encoding == "raw" else stack.enter_context(gzip.GzipFile(fileobj=data_fh, mode="rb"))
        slab_bytes = inner * ng * itemsize
        chunk_outer = max(1, min(outer, _NRRD_STREAM_CHUNK_BYTES // max(1, slab_bytes)))
        total_flat = np.zeros((inner, outer), dtype=np.float32, order="F")
        o0 = 0
        while o0 < outer:
            o1 = min(outer, o0 + chunk_outer)
            cur_outer = o1 - o0
            raw_bytes = _read_exact_bytes(stream, cur_outer * slab_bytes)
            chunk = np.frombuffer(raw_bytes, dtype=dtype).reshape((inner, ng, cur_outer), order="F")
            for g in indices:
                total_flat[:, o0:o1] += chunk[:, int(g), :].astype(np.float32, copy=False)
            o0 = o1
        return (total_flat / np.float32(len(indices))).reshape(spatial_shape, order="F")


def _mean_b0_nrrd(source: PreviewSource, threshold: float):
    import nibabel as nib
    import numpy as np

    try:
        import nrrd
    except ImportError as exc:
        raise PreviewError("bad_data", "Reading NRRD needs pynrrd, which is not installed.") from exc
    try:
        with _nrrd_read_target(source) as target:
            with open(target, "rb") as attached_fh:
                header = nrrd.read_header(attached_fh)
                sizes = tuple(int(s) for s in header.get("sizes", []))
                if len(sizes) != 4:
                    raise PreviewError("bad_data", f"The NRRD is {len(sizes)}D; a 4D diffusion series is required.")
                axis = _nrrd_diffusion_axis(header, len(sizes))
                ng = int(sizes[axis])
                spatial_shape = tuple(s for i, s in enumerate(sizes) if i != axis)
                item_bytes = _NRRD_ITEM_BYTES.get(str(header.get("type", "")).strip().lower(), 8)
                single_vol_bytes = int(np.prod(spatial_shape, dtype=np.int64)) * item_bytes
                announced = _nrrd_announced_bytes(header)
                if single_vol_bytes > MAX_NRRD_BYTES:
                    raise PreviewError(
                        "bad_data",
                        "The NRRD holds about %.1f GiB of voxel data; the preview reads at most %d GiB. "
                        "Convert the input to NIfTI first." % (announced / (1 << 30), MAX_NRRD_BYTES >> 30),
                    )
                bvals = _nrrd_bvalues(header, ng)
                indices = [index for index, value in enumerate(bvals) if value <= threshold]
                if not indices:
                    raise PreviewError("no_b0", f"{threshold:g}")
                mean = _stream_mean_b0_nrrd_payload(header, target, attached_fh, axis, indices, nrrd)
            if mean is None:
                if announced > MAX_NRRD_BYTES:
                    raise PreviewError(
                        "bad_data",
                        "The NRRD holds about %.1f GiB of voxel data; the preview reads at most %d GiB. "
                        "Convert the input to NIfTI first." % (announced / (1 << 30), MAX_NRRD_BYTES >> 30),
                    )
                data, header = nrrd.read(target)
                volumes = np.moveaxis(data, axis, -1)
                mean = _mean_of(lambda i: volumes[..., i], indices, threshold)
    except PreviewError:
        raise
    except Exception as exc:
        raise PreviewError("bad_data", f"Could not read the NRRD file: {exc}") from exc
    return nib.Nifti1Image(mean, _nrrd_ras_affine(header, axis)), len(indices)


def _nrrd_ras_affine(header: Mapping[str, Any], axis: int):
    """IJK-to-RAS matrix of the three spatial NRRD axes (the diffusion axis ``axis`` removed)."""

    import numpy as np

    directions = np.asarray(header.get("space directions"), dtype=float)
    spatial = np.delete(directions, axis, axis=0)
    if spatial.shape != (3, 3) or not np.all(np.isfinite(spatial)):
        raise PreviewError("bad_data", "The NRRD 'space directions' field is not usable.")
    origin = np.asarray(header.get("space origin", [0.0, 0.0, 0.0]), dtype=float)
    affine = np.eye(4)
    affine[:3, :3] = spatial.T
    affine[:3, 3] = origin
    return np.diag(list(_space_signs(header.get("space"))) + [1.0]) @ affine


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


# --------------------------------------------------------------------------- full DWI (b2-dwi)


@dataclass
class FullDwi:
    """A whole DWI read into memory, ready for a core ``vtkMRMLDiffusionWeightedVolumeNode``.

    ``image`` is a ``vtkImageData`` whose N-component scalar array VTK allocated and owns, so no numpy buffer has to
    stay alive. ``gradients`` are unit vectors (zero for a zero vector) in the measurement frame
    ``measurement_frame`` (3x3, gradient frame to RAS); ``bvalues`` are per volume.
    """

    image: Any
    ijk_to_ras: Any
    gradients: Any
    bvalues: list
    measurement_frame: Any
    voxel_bytes: int
    scalar_type: str


@dataclass
class PreparedPreview:
    """What the worker hands to the GUI thread: the full DWI, or the mean-b0 file above the size limit."""

    kind: str  # "dwi" or "meanb0"
    dwi: Optional[FullDwi] = None
    mean_b0: Optional[Path] = None
    estimated_bytes: int = 0
    limit_bytes: int = 0
    # Why the mean b0 is shown: "size" (above the size limit) or "no_gradients" (bvec missing or unusable); "" for a DWI.
    reason: str = "size"


def physical_memory_bytes() -> Optional[int]:
    """Installed RAM in bytes, or None when the platform does not say."""

    try:
        pages, page_size = os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return int(pages) * int(page_size)
    except (AttributeError, ValueError, OSError):
        pass
    if os.name == "nt":
        try:
            import ctypes

            class _MemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(_MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
        except Exception:
            pass
    return None


def available_memory_bytes() -> Optional[int]:
    """Currently available host RAM in bytes, or None when the platform does not say."""

    cap_bytes: Optional[int] = None
    cap_env = os.environ.get("SLICERDSPACE_MAX_RAM_GIB", "").strip()
    if cap_env:
        try:
            parsed = int(float(cap_env) * (1 << 30))
            if parsed > 0:
                cap_bytes = parsed
        except ValueError:
            pass

    avail: Optional[int] = None
    try:
        from shared.platform_info import available_host_ram_bytes

        reported = available_host_ram_bytes()
        if reported is not None and int(reported) > 0:
            avail = int(reported)
    except Exception:
        pass

    if avail is None and os.name == "nt":
        try:
            import ctypes

            class _AvailMemoryStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            status = _AvailMemoryStatus()
            status.dwLength = ctypes.sizeof(_AvailMemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) and status.ullAvailPhys > 0:
                avail = int(status.ullAvailPhys)
        except Exception:
            pass

    if avail is None and os.path.isfile("/proc/meminfo"):
        try:
            with open("/proc/meminfo", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    if line.startswith("MemAvailable:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            kib = int(parts[1])
                            if kib > 0:
                                avail = kib * 1024
                        break
        except Exception:
            pass

    total = physical_memory_bytes()
    if avail is not None and total is not None and total > 0:
        avail = min(avail, total)
    elif avail is None:
        avail = total
    if avail is not None and cap_bytes is not None:
        avail = min(avail, cap_bytes)
    return avail


def full_dwi_limit_bytes(physical: Any = "auto") -> int:
    """Largest voxel data that View input loads as a full DWI: min(4 GiB, 25 % of available RAM); 4 GiB if unknown."""

    if physical == "auto":
        avail = available_memory_bytes()
        total = physical_memory_bytes()
        if avail and total:
            effective = min(int(avail), int(total))
        else:
            effective = avail or total
        if not effective:
            return FULL_DWI_MAX_BYTES
        return int(min(FULL_DWI_MAX_BYTES, int(effective) * FULL_DWI_RAM_FRACTION))
    if not physical:
        return FULL_DWI_MAX_BYTES
    return int(min(FULL_DWI_MAX_BYTES, int(physical) * FULL_DWI_RAM_FRACTION))


def size_notice(estimated: int, limit: int, zh: bool = False) -> str:
    """Status text for the size fallback (no dialog)."""

    gib_estimated, gib_limit = estimated / float(1 << 30), limit / float(1 << 30)
    if zh:
        return (f"该 DWI 约含 {gib_estimated:.1f} GiB 体素数据，超过完整显示的上限 {gib_limit:.1f} GiB"
                f"（4 GiB 与内存 25% 中的较小者），因此改为显示平均 b0 图像。")
    return (f"The DWI holds about {gib_estimated:.1f} GiB of voxel data, more than the {gib_limit:.1f} GiB limit "
            f"for the full view (the smaller of 4 GiB and 25% of RAM), so the mean b0 image is shown instead.")


def gradient_notice(zh: bool = False) -> str:
    """Status text for the gradient fallback: the bvec file is missing or cannot be read (no dialog)."""

    if zh:
        return "找不到梯度方向文件（bvec），或无法读取，因此改为显示平均 b0 图像。"
    return "The gradient table (bvec) was not found or could not be read, so the mean b0 image is shown instead."


def gradient_table_unusable(source: PreviewSource) -> bool:
    """True when a NIfTI source has no usable bvec, so the full DWI cannot be built (b2 inputs-help addon).

    A missing path, a missing file, an unreadable file and a count that does not match the volumes all count. Another
    problem of the DWI itself (not 4D, a bval count mismatch) is left to the readers, which report it as an error.
    """

    if source.kind == "nrrd":
        return False  # the gradients are in the NRRD header
    if source.bvec is None or not Path(source.bvec).is_file():
        return True
    try:
        import nibabel as nib

        shape = tuple(int(size) for size in nib.load(str(source.dwi)).shape)
    except Exception:
        return False  # the header is unreadable: the readers report that
    if len(shape) != 4:
        return False
    try:
        read_bvecs(source.bvec, shape[3])
    except Exception:
        return True
    return False


def default_component(bvalues: Any, threshold: float = B0_THRESHOLD) -> int:
    """The first volume with b <= ``threshold`` (the first volume when there is none)."""

    for index, value in enumerate(bvalues):
        if float(value) <= threshold:
            return index
    return 0


def read_bvecs(path: Any, count: int):
    """FSL gradient directions as an (N, 3) array; 3 x N (FSL) and N x 3 layouts are both accepted."""

    import numpy as np

    text = Path(path).read_text(encoding="utf-8", errors="replace")
    try:
        rows = [[float(token) for token in line.replace(",", " ").split()] for line in text.splitlines()
                if line.strip()]
    except ValueError as exc:
        raise PreviewError("bad_data", f"The bvec file has a non-numeric entry: {exc}") from exc
    if not all(np.isfinite(value) for row in rows for value in row):  # b2 fix W3: "nan" and "inf" parse as floats
        raise PreviewError("bad_data", "The bvec file has a non-finite entry (nan or inf).")
    widths = {len(row) for row in rows}
    if len(widths) == 1:
        values = np.asarray(rows, dtype=float)
        if values.shape == (3, count):
            return values.T.copy()
        if values.shape == (count, 3):
            return values
    shape = f"{len(rows)} x {max(widths) if widths else 0}"
    raise PreviewError("bad_data", f"The bvec file has {shape} entries; 3 x {count} are needed for {count} volumes.")


def _unit_rows(vectors: Any):
    import numpy as np

    vectors = np.asarray(vectors, dtype=float).reshape(-1, 3)
    lengths = np.linalg.norm(vectors, axis=1)
    out = np.zeros_like(vectors)
    nonzero = lengths > 1e-6
    out[nonzero] = vectors[nonzero] / lengths[nonzero, None]
    return out


def gradients_ras(affine: Any, bvecs: Any):
    """FSL bvecs to RAS gradient directions: ``g = R @ F @ b``.

    ``R`` is ``affine[:3, :3]`` with unit columns. ``F = diag(-1, 1, 1)`` when ``det(affine[:3, :3]) > 0`` (FSL
    stores bvecs in a radiological voxel frame), else the identity. Measured equal to MRtrix ``-fslgrad`` (b2-dwi).
    Zero rows stay zero; other rows are returned with unit length.
    """

    import numpy as np

    matrix = np.asarray(affine, dtype=float)[:3, :3]
    norms = np.linalg.norm(matrix, axis=0)
    if not np.all(norms > 0) or not np.all(np.isfinite(matrix)):
        raise PreviewError("bad_data", "The DWI affine has a zero or invalid voxel axis.")
    rotation = matrix / norms
    flip = np.diag([-1.0, 1.0, 1.0]) if np.linalg.det(matrix) > 0 else np.eye(3)
    vectors = np.asarray(bvecs, dtype=float).reshape(-1, 3)
    return _unit_rows((rotation @ flip @ vectors.T).T)


def _build_dtype(dtype: Any):
    import numpy as np

    native = np.dtype(dtype).newbyteorder("=")
    if native.name not in _VTK_DTYPES:
        raise PreviewError("bad_data", f"The DWI voxel type {native.name} cannot be shown.")
    return native


def _nifti_build_dtype(image: Any):
    """On-disk dtype, or float32 when the NIfTI scale slope/intercept is not 1/0."""

    import numpy as np

    slope = float(getattr(image.dataobj, "slope", 1.0))
    inter = float(getattr(image.dataobj, "inter", 0.0))
    if slope != 1.0 or inter != 0.0:
        return np.dtype(np.float32)
    return _build_dtype(image.get_data_dtype())


def estimate_full_dwi_bytes(source: PreviewSource) -> int:
    """Voxel bytes of the full DWI as it would be built; reads only the header."""

    import numpy as np

    if source.kind == "nrrd":
        try:
            import nrrd
        except ImportError as exc:
            raise PreviewError("bad_data", "Reading NRRD needs pynrrd, which is not installed.") from exc
        try:
            with _nrrd_read_target(source) as target:
                return _nrrd_announced_bytes(nrrd.read_header(target))
        except PreviewError:
            raise
        except Exception as exc:
            raise PreviewError("bad_data", f"Could not read the NRRD header: {exc}") from exc
    import nibabel as nib

    try:
        image = nib.load(str(source.dwi))
    except Exception as exc:
        raise PreviewError("bad_data", f"Could not read the NIfTI header: {exc}") from exc
    return int(np.prod(image.shape, dtype=np.int64)) * int(_nifti_build_dtype(image).itemsize)


def _new_vtk_image(shape_ijk: Any, count: int, dtype: Any):
    """A vtkImageData with an uninitialised N-component array that VTK owns, and a numpy view (k, j, i, c) of it."""

    import vtk
    from vtk.util import numpy_support

    nx, ny, nz = (int(size) for size in shape_ijk)
    array = vtk.vtkDataArray.CreateDataArray(numpy_support.get_vtk_array_type(dtype))
    array.SetNumberOfComponents(int(count))
    array.SetNumberOfTuples(nx * ny * nz)
    array.SetName("DWI")
    image = vtk.vtkImageData()
    image.SetDimensions(nx, ny, nz)
    image.GetPointData().SetScalars(array)
    view = numpy_support.vtk_to_numpy(array).reshape(nz, ny, nx, int(count))
    return image, view


def _fill_block(view: Any, block: Any, start: int) -> None:
    """Copy volumes ``start..`` of an (i, j, k, n) block into the (k, j, i, c) view, one slab at a time."""

    stop = start + block.shape[3]
    for z in range(block.shape[2]):
        view[z, :, :, start:stop] = block[:, :, z, :].transpose(1, 0, 2)


def _read_full_dwi_nifti_into(
    source: PreviewSource,
    alloc_buffer: Callable[[tuple[int, int, int], int, Any], tuple[Any, Any]],
    block: int,
) -> tuple[Any, dict[str, Any]]:
    import nibabel as nib
    import numpy as np

    if source.bval is None or not Path(source.bval).is_file():
        raise PreviewError("no_bval")
    bvals = read_bvals(source.bval)
    image = nib.load(str(source.dwi), keep_file_open=True)
    shape = tuple(int(size) for size in image.shape)
    if len(shape) != 4:
        raise PreviewError("bad_data", f"The DWI is {len(shape)}D; a 4D volume series is required.")
    count = shape[3]
    if len(bvals) != count:
        raise PreviewError("bad_data", f"The bval file has {len(bvals)} entries but the DWI has {count} volumes.")
    if source.bvec is None or not Path(source.bvec).is_file():
        raise PreviewError("no_bvec")
    affine = np.array(image.affine, dtype=float)
    gradients = gradients_ras(affine, read_bvecs(source.bvec, count))
    dtype = _nifti_build_dtype(image)
    owner, view = alloc_buffer(shape[:3], count, dtype)
    for start in range(0, count, block):
        # The volumes of one block are contiguous in the file; a block keeps the peak near 1.3x the array (b2-dwi).
        chunk = np.asarray(image.dataobj[..., start:min(count, start + block)])
        _fill_block(view, chunk, start)
        del chunk
    del view, image  # the view points into caller memory; the image closes the file that keep_file_open holds
    meta = {
        "shape_ijk": shape[:3],
        "count": count,
        "dtype": dtype.name,
        "ijk_to_ras": affine,
        "gradients": gradients,
        "bvalues": [float(b) for b in bvals],
        "measurement_frame": np.eye(3),
        "voxel_bytes": int(np.prod(shape, dtype=np.int64)) * dtype.itemsize,
    }
    return owner, meta


def _stream_full_dwi_nrrd_into(
    header: Mapping[str, Any],
    target: str,
    attached_fh: Any,
    axis: int,
    nrrd_mod: Any,
    view: Any,
    block: int,
) -> bool:
    """Stream raw/gzip NRRD voxel data directly into the (nz, ny, nx, count) view without a full 4D copy.

    Returns True when streamed, or False when the encoding/layout requires falling back to ``nrrd.read``.
    """
    import numpy as np

    encoding = str(header.get("encoding", "")).strip().lower()
    if encoding not in {"raw", "gzip", "gz"}:
        return False
    data_filename = header.get("datafile", header.get("data file", None))
    if data_filename is not None:
        if not isinstance(data_filename, str):
            return False
        stripped = data_filename.strip()
        if not stripped or stripped.upper().startswith("LIST") or "%" in stripped:
            return False
        data_filename = stripped
    line_skip = int(header.get("lineskip", header.get("line skip", 0)))
    byte_skip = int(header.get("byteskip", header.get("byte skip", 0)))
    if line_skip < 0 or byte_skip < -1:
        raise PreviewError("bad_data", "The NRRD header has an invalid line skip or byte skip.")
    if encoding != "raw" and byte_skip != 0:
        return False

    sizes = tuple(int(s) for s in header.get("sizes", []))
    if axis not in (0, len(sizes) - 1):
        return False
    dtype = _nrrd_numpy_dtype(header, nrrd_mod)
    itemsize = int(dtype.itemsize)
    total_voxels = int(np.prod(sizes, dtype=np.int64))
    spatial_shape = tuple(s for i, s in enumerate(sizes) if i != axis)
    nx, ny, nz = (int(s) for s in spatial_shape)
    count = int(sizes[axis])

    with contextlib.ExitStack() as stack:
        if data_filename is not None:
            data_path = (
                data_filename
                if os.path.isabs(data_filename)
                else os.path.join(os.path.dirname(target), data_filename)
            )
            data_fh = stack.enter_context(open(data_path, "rb"))
        else:
            data_fh = attached_fh

        for _ in range(line_skip):
            data_fh.readline()
        if byte_skip > 0:
            data_fh.seek(byte_skip, os.SEEK_CUR)
        elif byte_skip == -1:
            data_fh.seek(-total_voxels * itemsize, os.SEEK_END)

        if encoding == "raw":
            base_offset = data_fh.tell()
            data_fh.seek(0, os.SEEK_END)
            if data_fh.tell() < base_offset + total_voxels * itemsize:
                raise PreviewError("bad_data", "The NRRD data file is shorter than the header dimensions.")
            data_fh.seek(base_offset, os.SEEK_SET)
            stream = data_fh
        else:
            stream = stack.enter_context(gzip.GzipFile(fileobj=data_fh, mode="rb"))

        if axis == len(sizes) - 1:
            # sizes = (nx, ny, nz, count) Fortran order: each 3D volume is contiguous.
            vol_bytes = nx * ny * nz * itemsize
            step = max(1, int(block))
            for start in range(0, count, step):
                cur = min(step, count - start)
                raw_bytes = _read_exact_bytes(stream, cur * vol_bytes)
                chunk = np.frombuffer(raw_bytes, dtype=dtype).reshape((nx, ny, nz, cur), order="F")
                _fill_block(view, chunk, start)
                del chunk, raw_bytes
            return True

        # axis == 0: sizes = (count, nx, ny, nz) Fortran order == (nz, ny, nx, count) C order directly!
        slab_bytes = ny * nx * count * itemsize
        slab_step = max(1, min(nz, _NRRD_STREAM_CHUNK_BYTES // max(1, slab_bytes)))
        z0 = 0
        while z0 < nz:
            z1 = min(nz, z0 + slab_step)
            raw_bytes = _read_exact_bytes(stream, (z1 - z0) * slab_bytes)
            chunk = np.frombuffer(raw_bytes, dtype=dtype).reshape((z1 - z0, ny, nx, count))
            view[z0:z1] = chunk
            del chunk, raw_bytes
            z0 = z1
        return True


def _read_full_dwi_nrrd_into(
    source: PreviewSource,
    alloc_buffer: Callable[[tuple[int, int, int], int, Any], tuple[Any, Any]],
    block: int,
) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    try:
        import nrrd
    except ImportError as exc:
        raise PreviewError("bad_data", "Reading NRRD needs pynrrd, which is not installed.") from exc
    owner = None
    try:
        with _nrrd_read_target(source) as target:
            with open(target, "rb") as attached_fh:
                header = nrrd.read_header(attached_fh)
                sizes = tuple(int(s) for s in header.get("sizes", []))
                if len(sizes) != 4:
                    raise PreviewError("bad_data", f"The NRRD is {len(sizes)}D; a 4D diffusion series is required.")
                announced = _nrrd_announced_bytes(header)
                if announced > MAX_NRRD_BYTES:
                    raise PreviewError(
                        "bad_data",
                        "The NRRD holds about %.1f GiB of voxel data; the preview reads at most %d GiB."
                        % (announced / (1 << 30), MAX_NRRD_BYTES >> 30),
                    )
                axis = _nrrd_diffusion_axis(header, len(sizes))
                count = int(sizes[axis])
                spatial_shape = tuple(int(s) for i, s in enumerate(sizes) if i != axis)
                bvals = _nrrd_bvalues(header, count)
                vectors = [[float(token) for token in str(header["DWMRI_gradient_%04d" % i]).split()] for i in range(count)]
                if any(len(vector) != 3 for vector in vectors):
                    raise PreviewError("bad_data", "A DWMRI_gradient entry does not have 3 components.")
                affine = _nrrd_ras_affine(header, axis)
                frame = header.get("measurement frame")
                # pynrrd returns one header vector per row; the NRRD vectors are the matrix columns.
                matrix = np.asarray(frame, dtype=float).T if frame is not None else np.eye(3)
                if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
                    raise PreviewError("bad_data", "The NRRD 'measurement frame' field is not usable.")
                measurement_frame = np.diag(_space_signs(header.get("space"))) @ matrix
                disk_dtype = _nrrd_numpy_dtype(header, nrrd)
                dtype = _build_dtype(disk_dtype)
                owner, view = alloc_buffer(spatial_shape, count, dtype)
                streamed = _stream_full_dwi_nrrd_into(header, target, attached_fh, axis, nrrd, view, block)
            if not streamed:
                data, header = nrrd.read(target)
                volumes = np.moveaxis(data, axis, -1)
                for start in range(0, count, block):
                    _fill_block(view, volumes[..., start:min(count, start + block)], start)
                del volumes, data
            del view
    except PreviewError:
        raise
    except Exception as exc:
        raise PreviewError("bad_data", f"Could not read the NRRD file: {exc}") from exc
    meta = {
        "shape_ijk": spatial_shape,
        "count": count,
        "dtype": dtype.name,
        "ijk_to_ras": affine,
        "gradients": _unit_rows(vectors),
        "bvalues": bvals,
        "measurement_frame": measurement_frame,
        "voxel_bytes": int(np.prod(spatial_shape, dtype=np.int64)) * count * dtype.itemsize,
    }
    return owner, meta


def _read_full_dwi_nifti(source: PreviewSource, block: int) -> FullDwi:
    vtk_image, meta = _read_full_dwi_nifti_into(source, _new_vtk_image, block)
    return FullDwi(
        vtk_image,
        meta["ijk_to_ras"],
        meta["gradients"],
        meta["bvalues"],
        meta["measurement_frame"],
        meta["voxel_bytes"],
        vtk_image.GetScalarTypeAsString(),
    )


def _read_full_dwi_nrrd(source: PreviewSource, block: int) -> FullDwi:
    vtk_image, meta = _read_full_dwi_nrrd_into(source, _new_vtk_image, block)
    return FullDwi(
        vtk_image,
        meta["ijk_to_ras"],
        meta["gradients"],
        meta["bvalues"],
        meta["measurement_frame"],
        meta["voxel_bytes"],
        vtk_image.GetScalarTypeAsString(),
    )


def read_full_dwi_to_shared_memory(source: PreviewSource, block: int = _FILL_BLOCK) -> tuple[Any, dict[str, Any]]:
    """Stream the whole DWI into a POSIX/Windows SharedMemory block in VTK (k, j, i, c) layout.

    Returns ``(shm, metadata_dict)``. Caller owns ``shm`` and must close/unlink it when done.
    """
    from multiprocessing.shared_memory import SharedMemory
    import numpy as np

    if not Path(source.dwi).is_file():
        raise PreviewError("missing")
    created: list[Any] = []

    def alloc_shm(shape_ijk: tuple[int, int, int], count: int, dtype: Any) -> tuple[Any, Any]:
        nx, ny, nz = (int(s) for s in shape_ijk)
        nbytes = int(nx) * int(ny) * int(nz) * int(count) * int(np.dtype(dtype).itemsize)
        shm = SharedMemory(create=True, size=max(1, nbytes))
        created.append(shm)
        view = np.ndarray((nz, ny, nx, int(count)), dtype=np.dtype(dtype), buffer=shm.buf)
        return shm, view

    try:
        if source.kind == "nrrd":
            shm, meta = _read_full_dwi_nrrd_into(source, alloc_shm, block)
        else:
            shm, meta = _read_full_dwi_nifti_into(source, alloc_shm, block)
    except BaseException:
        for shm in created:
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except Exception:
                pass
        raise
    meta["shm_name"] = shm.name
    return shm, meta


def full_dwi_from_shared_memory(meta: Mapping[str, Any]) -> FullDwi:
    """GUI thread: copy a SharedMemory (k, j, i, c) buffer into a VTK-owned vtkImageData in one memcpy."""
    from multiprocessing.shared_memory import SharedMemory
    import numpy as np

    shape_ijk = tuple(int(s) for s in meta["shape_ijk"])
    nx, ny, nz = shape_ijk
    count = int(meta["count"])
    dtype = _build_dtype(np.dtype(str(meta["dtype"])))
    shm = SharedMemory(name=str(meta["shm_name"]), create=False)
    try:
        src = np.ndarray((nz, ny, nx, count), dtype=dtype, buffer=shm.buf)
        vtk_image, view = _new_vtk_image(shape_ijk, count, dtype)
        np.copyto(view, src)
        del src, view
    finally:
        try:
            shm.close()
        except Exception:
            pass
    return FullDwi(
        vtk_image,
        np.asarray(meta["ijk_to_ras"], dtype=float),
        np.asarray(meta["gradients"], dtype=float),
        [float(b) for b in meta["bvalues"]],
        np.asarray(meta["measurement_frame"], dtype=float),
        int(meta["voxel_bytes"]),
        vtk_image.GetScalarTypeAsString(),
    )


def read_full_dwi(source: PreviewSource, block: int = _FILL_BLOCK) -> FullDwi:
    """Read the whole DWI into a VTK array of the on-disk type (runs on the worker thread; writes no file)."""

    if not Path(source.dwi).is_file():
        raise PreviewError("missing")
    try:
        if source.kind == "nrrd":
            return _read_full_dwi_nrrd(source, block)
        return _read_full_dwi_nifti(source, block)
    except PreviewError:
        raise
    except MemoryError as exc:
        raise PreviewError("bad_data", "Not enough memory to load the full DWI.") from exc
    except Exception as exc:
        raise PreviewError("bad_data", f"Could not read the full DWI: {exc}") from exc


def prepare_input_preview(
    source: PreviewSource, cache_dir: Any, *, limit: Optional[int] = None, threshold: float = B0_THRESHOLD
) -> PreparedPreview:
    """Worker-thread step of View input: the full DWI, or the mean b0 when it is above the size limit."""

    if not Path(source.dwi).is_file():
        raise PreviewError("missing")
    limit = full_dwi_limit_bytes() if limit is None else int(limit)
    estimated = estimate_full_dwi_bytes(source)
    if gradient_table_unusable(source):
        # Without gradients the earlier view is kept: the mean b0 (it needs only the bval file).
        return PreparedPreview("meanb0", mean_b0=mean_b0_image(source, cache_dir, threshold),
                               estimated_bytes=estimated, limit_bytes=limit, reason="no_gradients")
    if estimated > limit:
        return PreparedPreview("meanb0", mean_b0=mean_b0_image(source, cache_dir, threshold),
                               estimated_bytes=estimated, limit_bytes=limit, reason="size")
    return PreparedPreview("dwi", dwi=read_full_dwi(source), estimated_bytes=estimated, limit_bytes=limit, reason="")


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
    label = _load_preview_mask(slicer, mask, name, say)
    _fit_slice_views(slicer, volume.GetID(), label.GetID() if label is not None else None)
    return {"volume_id": volume.GetID(), "mask_id": label.GetID() if label is not None else None}


def _load_preview_mask(slicer: Any, mask: Any, name: str, say: Callable[[str], None]) -> Any:
    """The brain mask as a label map tagged as a preview node, or None (a mask failure never stops the preview)."""

    if not mask:
        return None
    try:
        label = _first_node(slicer.util.loadLabelVolume(str(mask), properties={"name": f"{name}_mask", "show": False}))
    except Exception as exc:
        say(f"[Input preview] The brain mask could not be loaded: {exc}")
        return None
    if label is None:
        say(f"[Input preview] The brain mask could not be loaded as a label map: {mask}")
        return None
    label.SetAttribute(PREVIEW_ATTRIBUTE, "mask")
    return label


def remove_preview_nodes(slicer: Any) -> None:
    """Remove every node that an earlier View input added (full DWI, mean b0, mask)."""

    _remove_previous_preview_nodes(slicer)


def build_dwi_volume_node(
    slicer: Any,
    dwi: FullDwi,
    name: str,
    *,
    threshold: float = B0_THRESHOLD,
    attribute: Optional[tuple[str, str]] = None,
) -> tuple[Any, dict[str, Any]]:
    """GUI thread: wrap a :class:`FullDwi` in a core ``vtkMRMLDiffusionWeightedVolumeNode``.

    Shared by ``show_dwi_in_scene`` (input preview) and ``shared.result_scene.load_dwi_result``
    (preprocessed 4D DWI outputs) so every 4D DWI in the scene preserves exact RAS geometry and
    native Slicer Volumes ``DWI Component`` support without SlicerDMRI.
    """

    import numpy as np

    try:
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLDiffusionWeightedVolumeNode", name)
    except Exception as exc:
        raise PreviewError("bad_data", f"Slicer could not create the DWI node: {exc}") from exc
    if node is None:
        raise PreviewError("bad_data", "Slicer could not create the DWI node.")
    count = len(dwi.bvalues)
    component = default_component(dwi.bvalues, threshold)
    try:
        if attribute is not None:
            node.SetAttribute(attribute[0], attribute[1])
        if len(dwi.gradients) != count:
            raise ValueError(f"{len(dwi.gradients)} gradients for {count} volumes")
        node.SetIJKToRASMatrix(slicer.util.vtkMatrixFromArray(np.asarray(dwi.ijk_to_ras, dtype=float)))
        node.SetAndObserveImageData(dwi.image)
        node.SetNumberOfGradients(count)
        for index in range(count):
            node.SetDiffusionGradient(index, [float(value) for value in dwi.gradients[index]])
            node.SetBValue(index, float(dwi.bvalues[index]))
        frame = np.eye(4)
        frame[:3, :3] = np.asarray(dwi.measurement_frame, dtype=float)
        node.SetMeasurementFrameMatrix(slicer.util.vtkMatrixFromArray(frame))
        node.CreateDefaultDisplayNodes()
        display = node.GetDisplayNode()
        if display is not None:
            display.SetAutoWindowLevel(1)  # measured: W/L is recomputed for each component
            display.SetDiffusionComponent(component)
    except Exception as exc:
        try:
            slicer.mrmlScene.RemoveNode(node)
        except Exception:
            pass
        raise PreviewError("bad_data", f"Slicer could not build the DWI node: {exc}") from exc
    return node, {
        "volume_id": node.GetID(),
        "component": component,
        "count": count,
        "b_value": float(dwi.bvalues[component]) if count else 0.0,
    }


def show_dwi_in_scene(
    slicer: Any,
    dwi: FullDwi,
    mask: Any = None,
    *,
    subject_id: str = "",
    log: Optional[Callable[[str], None]] = None,
    threshold: float = B0_THRESHOLD,
) -> dict[str, Any]:
    """GUI thread: wrap a :class:`FullDwi` in a core DWI node, show the first b0 component and the mask overlay.

    The node replaces any earlier preview node. Only core Slicer classes are used (no SlicerDMRI).
    """

    say = log or (lambda _message: None)
    _remove_previous_preview_nodes(slicer)
    name = f"{subject_id or 'subject'}_input"
    node, info = build_dwi_volume_node(
        slicer,
        dwi,
        f"{name}_dwi",
        threshold=threshold,
        attribute=(PREVIEW_ATTRIBUTE, "dwi"),
    )
    label = _load_preview_mask(slicer, mask, name, say)
    _fit_slice_views(slicer, node.GetID(), label.GetID() if label is not None else None)
    return {
        "volume_id": info["volume_id"],
        "mask_id": label.GetID() if label is not None else None,
        "component": info["component"],
        "count": info["count"],
        "b_value": info["b_value"],
    }
