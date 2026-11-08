"""Connectome construction planning and dependency checks."""

from __future__ import annotations

from shared.cohort_schedule import scheduled_stage

import csv
import json
import math
import os
import subprocess
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Optional

from shared import default_task_output_dir_for_dependency
from shared.external_process import startup_environment, resolve_external_command


SUPPORTED_TRACTOGRAPHY_EXTENSIONS = {".trk", ".tck", ".vtk", ".vtp"}
SUPPORTED_PARCELLATION_SUFFIXES = (".nii", ".nii.gz", ".nrrd", ".mgz")


@dataclass(frozen=True)
class ConnectomeConfig:
    subject_id: str
    tractography_path: Path
    parcellation_path: Path
    output_dir: Optional[Path] = None
    metric: str = "streamline_count"
    scalar_map_path: Optional[Path] = None
    sift2_weights_path: Optional[Path] = None
    lut_path: Optional[Path] = None
    mrtrix_bin_dir: Optional[Path] = None
    backend: str = "auto"
    node_source: str = "ddparcel"
    node_filter: str = "gray_matter"
    assignment_radius_mm: float = 4.0
    symmetric: bool = True
    zero_diagonal: bool = True
    # None = not set by the user: ``auto`` with the MRtrix path then compresses (D5).
    compress_mrtrix_dense: Optional[bool] = None
    execute: bool = False


CONNECTOME_BACKENDS = ("auto", "python", "mrtrix")
MRTRIX_CONNECTOME_METRICS = frozenset({"streamline_count", "sift2_weighted_streamline_count"})
TCK2CONNECTOME_MISSING = (
    "MRtrix tck2connectome is needed for this connectome; install MRtrix3 in Prepare, "
    "or choose the Python connectome backend (--connectome-backend python)"
)


def _mrtrix_connectome_input(config: ConnectomeConfig) -> bool:
    return config.tractography_path.suffix.lower() == ".tck" and config.metric in MRTRIX_CONNECTOME_METRICS


def select_connectome_backend(config: ConnectomeConfig, backend_status: dict[str, object]) -> tuple[bool, str, str]:
    """Return (use_mrtrix, backend_used, reason) for the requested backend.

    ``auto``: a ``.tck`` source with a streamline-count metric uses MRtrix ``tck2connectome``;
    any other source or metric uses Python. A missing ``tck2connectome`` is an error, never a fallback.
    ``mrtrix`` and ``python`` keep their earlier meaning.
    """
    eligible = _mrtrix_connectome_input(config)
    have_tool = bool(backend_status.get("tck2connectome"))
    if config.backend == "auto":
        if not eligible:
            return False, "python", "auto: source is not a .tck file or the metric is not a streamline count"
        if not have_tool:
            raise RuntimeError(TCK2CONNECTOME_MISSING)
        return True, "mrtrix", "auto: .tck source with a streamline-count metric"
    if config.backend == "mrtrix":
        if not (eligible and have_tool):
            raise RuntimeError("Requested MRtrix connectome backend is unavailable for this input/metric.")
        return True, "mrtrix", "requested mrtrix"
    return False, "python", "requested python" if config.backend == "python" else f"backend {config.backend!r}: python"


def resolve_compress_mrtrix_dense(config: ConnectomeConfig, use_mrtrix: bool) -> bool:
    """The compression choice of a run: the user value, else on only for ``auto`` with the MRtrix path (D5)."""
    if config.compress_mrtrix_dense is not None:
        return bool(config.compress_mrtrix_dense)
    return bool(use_mrtrix and config.backend == "auto")


def expected_connectome_choice(config: ConnectomeConfig) -> tuple[str, bool]:
    """(backend_used, compress_mrtrix_dense) that a run of ``config`` would record (T-B2).

    Used by the orchestrator to decide whether an earlier connectome is current. The availability of
    ``tck2connectome`` does not enter: a missing tool is a run-time error, not a different result.
    Raises RuntimeError when the requested backend cannot work for this input (as the run does).
    """
    use_mrtrix, backend_used, _reason = select_connectome_backend(config, {"tck2connectome": True})
    return backend_used, resolve_compress_mrtrix_dense(config, use_mrtrix)


@dataclass(frozen=True)
class ConnectomeResult:
    subject_id: str
    output_dir: Path
    matrix_path: Path
    nodes_path: Path
    manifest_path: Path
    qc_path: Path
    backend_status: dict[str, object]
    executed: bool
    return_code: Optional[int] = None


def _emit(message: str, log: Optional[Callable[[str], None]] = None) -> None:
    if log:
        log(message)


def _file_info(path: Path) -> dict[str, object]:
    expanded = path.expanduser()
    if not expanded.exists():
        return {"path": expanded.as_posix(), "exists": False}
    stat = expanded.stat()
    return {
        "path": expanded.resolve().as_posix(),
        "exists": True,
        "is_file": expanded.is_file(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _table_has_data_row(path: Path, *, header_prefix: str) -> bool:
    if not _nonempty_file(path):
        return False
    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            first = next((line.strip() for line in stream if line.strip()), "")
            if not first:
                return False
            if not first.lower().startswith(header_prefix.lower()):
                return True
            return any(line.strip() for line in stream)
    except OSError:
        return False


def _connectome_output_errors(qc: dict[str, object], matrix_path: Path, nodes_path: Path) -> list[str]:
    errors: list[str] = []
    try:
        if int(qc.get("node_count", 0) or 0) <= 0:
            errors.append("node_count<=0")
    except (TypeError, ValueError):
        errors.append("invalid node_count")
    try:
        if int(qc.get("edge_count", 0) or 0) <= 0:
            errors.append("edge_count<=0")
    except (TypeError, ValueError):
        errors.append("invalid edge_count")
    if not _table_has_data_row(nodes_path, header_prefix="node_id\t"):
        errors.append("nodes table has no data rows")
    if not _table_has_data_row(matrix_path, header_prefix="source_node,target_node,"):
        errors.append("matrix has no data edges")
    return errors


def _has_supported_parcellation_suffix(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in SUPPORTED_PARCELLATION_SUFFIXES)


def _default_lut_path() -> Path:
    return Path(__file__).resolve().parents[1] / "anatomy" / "parcellation" / "conversion" / "data" / "FreeSurferColorLUT.txt"


def _resolve_lut_path(config: ConnectomeConfig) -> Optional[Path]:
    lut_path = config.lut_path.expanduser().resolve() if config.lut_path is not None else _default_lut_path()
    return lut_path if lut_path.is_file() else None


def _resolve_command(name: str, bin_dir: Optional[Path] = None) -> Optional[str]:
    return resolve_external_command(name, bin_dir)


def inspect_connectome_backends(bin_dir: Optional[Path] = None) -> dict[str, object]:
    """Inspect optional backends for tractography-to-parcellation assignment."""

    status: dict[str, object] = {
        "mrtrix_bin_dir": bin_dir.expanduser().resolve().as_posix() if bin_dir is not None else None,
        "tck2connectome": _resolve_command("tck2connectome", bin_dir),
        "python": {
            "nibabel": False,
            "numpy": False,
            "vtk": False,
        },
    }
    for module_name in ("nibabel", "numpy", "vtk"):
        try:
            __import__(module_name)
        except Exception:
            status["python"][module_name] = False  # type: ignore[index]
        else:
            status["python"][module_name] = True  # type: ignore[index]
    return status


def _load_label_volume(path: Path):
    try:
        import nibabel as nib
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Connectome execution requires nibabel and numpy for label lookup.") from exc

    image = nib.load(path.as_posix())
    data = np.asarray(image.dataobj)
    inverse_affine = np.linalg.inv(image.affine)
    return data, inverse_affine


def _load_scalar_volume(path: Path):
    try:
        import nibabel as nib
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Scalar-weighted connectome execution requires nibabel and numpy.") from exc

    image = nib.load(path.as_posix())
    data = np.asarray(image.dataobj, dtype=float)
    inverse_affine = np.linalg.inv(image.affine)
    return data, inverse_affine


def _label_at_point(point, data, inverse_affine) -> int:
    import numpy as np

    voxel = np.rint((inverse_affine @ np.array([point[0], point[1], point[2], 1.0]))[:3]).astype(int)
    if np.any(voxel < 0) or np.any(voxel >= np.array(data.shape[:3])):
        return 0
    return int(data[tuple(voxel)])


def _build_endpoint_label_assigner(
    data,
    inverse_affine,
    *,
    allowed_labels: Iterable[int],
    radius_mm: float,
) -> tuple[Callable[[object], int], str]:
    import numpy as np

    allowed = np.asarray(sorted({int(label) for label in allowed_labels}), dtype=data.dtype)
    allowed_mask = np.isin(data, allowed)

    if radius_mm <= 0:
        allowed_set = {int(label) for label in allowed}

        def assign_end_voxel(point: object) -> int:
            label = _label_at_point(point, data, inverse_affine)
            return label if label in allowed_set else 0

        return assign_end_voxel, "end_voxel"

    allowed_voxels = np.argwhere(allowed_mask)
    if not len(allowed_voxels):
        return lambda point: 0, "radial_search_empty"

    affine = np.linalg.inv(inverse_affine)
    world_coordinates = (
        allowed_voxels @ affine[:3, :3].T
        + affine[:3, 3]
    )

    try:
        from scipy.spatial import cKDTree
    except ImportError:  # pragma: no cover - exercised only in minimal environments
        singular_values = np.linalg.svd(affine[:3, :3], compute_uv=False)
        min_spacing = max(float(np.min(singular_values)), 1e-6)
        padding = int(math.ceil(radius_mm / min_spacing)) + 2
        shape = np.asarray(data.shape[:3], dtype=int)

        def assign_local_search(point: object) -> int:
            point_array = np.asarray(point, dtype=float)[:3]
            continuous_voxel = (inverse_affine @ np.append(point_array, 1.0))[:3]
            center = np.rint(continuous_voxel).astype(int)
            lower = np.maximum(center - padding, 0)
            upper = np.minimum(center + padding + 1, shape)
            if np.any(lower >= upper):
                return 0
            slices = tuple(slice(int(lower[axis]), int(upper[axis])) for axis in range(3))
            local_voxels = np.argwhere(allowed_mask[slices])
            if not len(local_voxels):
                return 0
            local_voxels += lower
            local_world = local_voxels @ affine[:3, :3].T + affine[:3, 3]
            distances = np.linalg.norm(local_world - point_array, axis=1)
            nearest_index = int(np.argmin(distances))
            if float(distances[nearest_index]) > radius_mm:
                return 0
            return int(data[tuple(local_voxels[nearest_index])])

        return assign_local_search, "numpy.radial_search"

    tree = cKDTree(world_coordinates)

    def assign_kdtree(point: object) -> int:
        distance, index = tree.query(
            np.asarray(point, dtype=float)[:3],
            k=1,
            distance_upper_bound=radius_mm,
        )
        if not np.isfinite(distance) or int(index) >= len(allowed_voxels):
            return 0
        return int(data[tuple(allowed_voxels[int(index)])])

    return assign_kdtree, "scipy.spatial.cKDTree"


def _scalar_at_point(point, data, inverse_affine) -> Optional[float]:
    import math
    import numpy as np

    voxel = np.rint((inverse_affine @ np.array([point[0], point[1], point[2], 1.0]))[:3]).astype(int)
    if np.any(voxel < 0) or np.any(voxel >= np.array(data.shape[:3])):
        return None
    value = float(data[tuple(voxel)])
    return value if math.isfinite(value) else None


def _iter_trk_tck_streamlines(path: Path) -> Iterable[object]:
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("TRK/TCK connectome execution requires nibabel.") from exc

    tractogram = nib.streamlines.load(path.as_posix())
    yield from tractogram.streamlines


def _iter_vtk_streamlines(path: Path) -> Iterable[list[tuple[float, float, float]]]:
    try:
        import vtk  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment dependent
        if path.suffix.lower() == ".vtk":
            yield from _iter_legacy_vtk_polydata_streamlines(path)
            return
        raise RuntimeError("VTP connectome execution requires the vtk Python module.") from exc

    if path.suffix.lower() == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
    else:
        reader = vtk.vtkPolyDataReader()
    reader.SetFileName(path.as_posix())
    reader.Update()
    polydata = reader.GetOutput()
    points = polydata.GetPoints()
    lines = polydata.GetLines()
    lines.InitTraversal()
    id_list = vtk.vtkIdList()
    while lines.GetNextCell(id_list):
        streamline: list[tuple[float, float, float]] = []
        for index in range(id_list.GetNumberOfIds()):
            streamline.append(points.GetPoint(id_list.GetId(index)))
        if streamline:
            yield streamline


def _read_vtk_ascii_token(blob: bytes, offset: int) -> tuple[str, int]:
    while offset < len(blob) and chr(blob[offset]).isspace():
        offset += 1
    start = offset
    while offset < len(blob) and not chr(blob[offset]).isspace():
        offset += 1
    return blob[start:offset].decode("ascii"), offset


def _iter_legacy_vtk_polydata_streamlines(path: Path) -> Iterable[list[tuple[float, float, float]]]:
    """Read endpoints from legacy VTK POLYDATA without the optional vtk module.

    This supports the common ASCII and BINARY legacy VTK layout used by UKF/WMA
    tractography. It intentionally handles only POINTS plus LINES/POLYLINES.
    """

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Legacy VTK connectome fallback requires numpy.") from exc

    blob = path.read_bytes()
    header = blob[:512].decode("ascii", errors="ignore").upper()
    is_binary = "\nBINARY\n" in header
    points_marker = blob.find(b"POINTS ")
    if points_marker < 0:
        raise RuntimeError(f"Legacy VTK file has no POINTS section: {path}")
    token, offset = _read_vtk_ascii_token(blob, points_marker)
    if token != "POINTS":
        raise RuntimeError(f"Unexpected VTK POINTS marker: {token}")
    n_points_token, offset = _read_vtk_ascii_token(blob, offset)
    n_points = int(n_points_token)
    point_type, offset = _read_vtk_ascii_token(blob, offset)
    while offset < len(blob) and blob[offset] in b" \t\r\n":
        offset += 1

    if point_type.lower() not in {"float", "double"}:
        raise RuntimeError(f"Unsupported VTK point type: {point_type}")
    dtype = ">f4" if point_type.lower() == "float" else ">f8"
    if is_binary:
        count = n_points * 3
        points = np.frombuffer(blob, dtype=dtype, count=count, offset=offset).reshape((n_points, 3))
        offset += points.nbytes
    else:
        text_start = offset
        lines_marker = blob.find(b"\nLINES ", text_start)
        if lines_marker < 0:
            lines_marker = blob.find(b"\nPOLYLINES ", text_start)
        if lines_marker < 0:
            raise RuntimeError(f"Legacy VTK file has no LINES section: {path}")
        points = np.fromstring(blob[text_start:lines_marker].decode("ascii"), sep=" ").reshape((n_points, 3))
        offset = lines_marker + 1

    if blob[offset : offset + len(b"LINES ")].upper() == b"LINES ":
        lines_marker = offset
        marker_len = len(b"LINES ")
    elif blob[offset : offset + len(b"POLYLINES ")].upper() == b"POLYLINES ":
        lines_marker = offset
        marker_len = len(b"POLYLINES ")
    else:
        lines_marker = blob.find(b"\nLINES ", offset)
        marker_len = len(b"\nLINES ")
        if lines_marker < 0:
            lines_marker = blob.find(b"\nPOLYLINES ", offset)
            marker_len = len(b"\nPOLYLINES ")
    if lines_marker < 0:
        raise RuntimeError(f"Legacy VTK file has no LINES/POLYLINES section: {path}")
    offset = lines_marker + marker_len
    n_lines_token, offset = _read_vtk_ascii_token(blob, offset)
    n_lines = int(n_lines_token)
    total_size, offset = _read_vtk_ascii_token(blob, offset)
    n_indices = int(total_size)
    while offset < len(blob) and blob[offset] in b" \t\r\n":
        offset += 1

    if blob[offset : offset + len(b"OFFSETS")].upper() == b"OFFSETS":
        _, offset = _read_vtk_ascii_token(blob, offset)
        offset_type, offset = _read_vtk_ascii_token(blob, offset)
        while offset < len(blob) and blob[offset] in b" \t\r\n":
            offset += 1
        offset_dtype = ">i8" if offset_type.lower().endswith("int64") else ">i4"
        offsets = np.frombuffer(blob, dtype=offset_dtype, count=n_lines, offset=offset)
        offset += offsets.nbytes
        while offset < len(blob) and blob[offset] in b" \t\r\n":
            offset += 1
        connectivity_marker, offset = _read_vtk_ascii_token(blob, offset)
        if connectivity_marker.upper() != "CONNECTIVITY":
            raise RuntimeError(f"Expected VTK CONNECTIVITY section, got: {connectivity_marker}")
        connectivity_type, offset = _read_vtk_ascii_token(blob, offset)
        while offset < len(blob) and blob[offset] in b" \t\r\n":
            offset += 1
        connectivity_dtype = ">i8" if connectivity_type.lower().endswith("int64") else ">i4"
        connectivity = np.frombuffer(blob, dtype=connectivity_dtype, count=n_indices, offset=offset)
        for index in range(n_lines):
            start = int(offsets[index])
            end = int(offsets[index + 1]) if index + 1 < n_lines else n_indices
            if end - start >= 2:
                first = int(connectivity[start])
                last = int(connectivity[end - 1])
                yield [tuple(float(value) for value in points[first]), tuple(float(value) for value in points[last])]
        return

    if is_binary:
        cells = np.frombuffer(blob, dtype=">i4", count=n_indices, offset=offset)
        cursor = 0
        for _ in range(n_lines):
            n_cell_points = int(cells[cursor])
            cursor += 1
            if n_cell_points >= 2:
                first = int(cells[cursor])
                last = int(cells[cursor + n_cell_points - 1])
                yield [tuple(float(value) for value in points[first]), tuple(float(value) for value in points[last])]
            cursor += n_cell_points
    else:
        numbers = np.fromstring(blob[offset:].decode("ascii", errors="ignore"), sep=" ", dtype=np.int64)
        cursor = 0
        for _ in range(n_lines):
            n_cell_points = int(numbers[cursor])
            cursor += 1
            if n_cell_points >= 2:
                first = int(numbers[cursor])
                last = int(numbers[cursor + n_cell_points - 1])
                yield [tuple(float(value) for value in points[first]), tuple(float(value) for value in points[last])]
            cursor += n_cell_points


def _iter_streamlines(path: Path) -> Iterable[object]:
    suffix = path.suffix.lower()
    if suffix in {".trk", ".tck"}:
        yield from _iter_trk_tck_streamlines(path)
        return
    if suffix in {".vtk", ".vtp"}:
        yield from _iter_vtk_streamlines(path)
        return
    raise ValueError(f"Unsupported tractography format: {suffix}")


def _parse_label_names(lut_path: Optional[Path]) -> dict[int, str]:
    if lut_path is None:
        return {}

    names: dict[int, str] = {}
    with lut_path.open("r", encoding="utf-8") as stream:
        first_data_line_seen = False
        header: list[str] | None = None
        for raw_line in stream:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if not first_data_line_seen and not parts[0].lstrip("-").isdigit():
                header = [part.strip().lower() for part in parts]
                first_data_line_seen = True
                continue
            first_data_line_seen = True
            try:
                if header and "label" in header and "name" in header:
                    label = int(parts[header.index("label")])
                    name = parts[header.index("name")]
                else:
                    label = int(parts[0])
                    name = parts[1]
            except (IndexError, ValueError):
                continue
            names[label] = name
    return names


def _write_nodes_table(path: Path, labels: Iterable[int], *, label_names: Optional[dict[int, str]] = None) -> int:
    unique_labels = sorted({int(label) for label in labels if int(label) != 0})
    names = label_names or {}
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("node_id\tlabel\tname\n")
        for index, label in enumerate(unique_labels, start=1):
            stream.write(f"{index}\t{label}\t{names.get(label, f'ROI-{label}')}\n")
    return len(unique_labels)


def _connectome_node_labels(
    labels: Iterable[int],
    *,
    node_filter: str,
) -> list[int]:
    unique_labels = sorted({int(label) for label in labels if int(label) != 0})
    if node_filter == "all":
        return unique_labels

    # Apply the anatomical filter only to FreeSurfer/DDParcel-style maps.
    # Generic user LUTs and compact test parcellations retain all labels.
    if not any(1000 <= label < 3000 for label in unique_labels):
        return unique_labels

    subcortical_gray_matter = {
        10,
        11,
        12,
        13,
        17,
        18,
        26,
        28,
        49,
        50,
        51,
        52,
        53,
        54,
        58,
        60,
    }
    return [
        label
        for label in unique_labels
        if label in subcortical_gray_matter or 1000 <= label < 3000
    ]


def _write_filtered_parcellation(
    source: Path,
    destination: Path,
    *,
    allowed_labels: Iterable[int],
) -> None:
    try:
        import nibabel as nib
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Filtered MRtrix connectome execution requires nibabel and numpy.") from exc

    image = nib.load(source.as_posix())
    data = np.asarray(image.dataobj)
    allowed = np.asarray(sorted({int(label) for label in allowed_labels}), dtype=data.dtype)
    filtered = np.where(np.isin(data, allowed), data, 0)
    destination.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(filtered, image.affine, image.header),
        destination.as_posix(),
    )


def _metric_requires_scalar(metric: str) -> bool:
    return metric.lower() in {"mean_scalar", "mean_fa", "mean_md", "mean_odi"}


def _metric_requires_length(metric: str) -> bool:
    return metric.lower() in {"mean_length_mm", "total_length_mm"}


def _streamline_length_mm(streamline) -> float:
    import numpy as np

    points = np.asarray(streamline, dtype=float)
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _read_sift2_weights(path: Path) -> list[float]:
    weights: list[float] = []
    for line in path.expanduser().read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for token in stripped.split():
            try:
                weights.append(float(token))
            except ValueError:
                continue
    return weights


def _tck_header_streamline_count(path: Path) -> Optional[int]:
    """Return the declared MRtrix TCK streamline count without loading points."""

    if path.expanduser().suffix.lower() != ".tck":
        return None
    try:
        import nibabel as nib
    except ImportError:
        return None

    try:
        header = nib.streamlines.load(path.expanduser().resolve().as_posix(), lazy_load=True).header
        value = header.get("count")
        return int(value) if value is not None else None
    except Exception:
        # This is optional QC metadata.  An unreadable/non-TCK fixture or a
        # backend-specific nibabel header error must not turn an otherwise
        # successful tck2connectome execution into a pipeline failure.
        return None


def _write_python_connectome(
    config: ConnectomeConfig,
    matrix_path: Path,
    nodes_path: Path,
    *,
    lut_path: Optional[Path],
) -> dict[str, object]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Connectome execution requires numpy.") from exc

    data, inverse_affine = _load_label_volume(config.parcellation_path.expanduser().resolve())
    label_names = _parse_label_names(lut_path)
    scalar_data = None
    scalar_inverse_affine = None
    sift2_weights = _read_sift2_weights(config.sift2_weights_path) if config.metric == "sift2_weighted_streamline_count" and config.sift2_weights_path is not None else None
    if _metric_requires_scalar(config.metric):
        if config.scalar_map_path is None:
            raise ValueError(f"metric='{config.metric}' requires scalar_map_path.")
        scalar_data, scalar_inverse_affine = _load_scalar_volume(config.scalar_map_path.expanduser().resolve())
    all_labels = np.unique(data)
    unique_nonzero_labels = sorted({int(label) for label in all_labels if int(label) != 0})
    node_labels = _connectome_node_labels(
        unique_nonzero_labels,
        node_filter=config.node_filter,
    )
    allowed_node_labels = set(node_labels)
    assign_endpoint, assignment_implementation = _build_endpoint_label_assigner(
        data,
        inverse_affine,
        allowed_labels=node_labels,
        radius_mm=config.assignment_radius_mm,
    )
    named_node_count = sum(1 for label in node_labels if label in label_names)
    node_count = _write_nodes_table(nodes_path, node_labels, label_names=label_names)
    edge_counts: Counter[tuple[int, int]] = Counter()
    edge_scalar_sums: Counter[tuple[int, int]] = Counter()
    edge_scalar_counts: Counter[tuple[int, int]] = Counter()
    edge_length_sums: Counter[tuple[int, int]] = Counter()
    edge_length_counts: Counter[tuple[int, int]] = Counter()
    skipped_streamlines = 0
    skipped_non_node_streamlines = 0
    skipped_scalar_streamlines = 0
    total_streamlines = 0
    radially_assigned_endpoints = 0
    endpoint_assigned_streamlines = 0
    matrix_included_streamlines = 0
    self_connection_streamlines = 0

    for streamline_index, streamline in enumerate(_iter_streamlines(config.tractography_path.expanduser().resolve())):
        total_streamlines += 1
        if len(streamline) < 2:
            skipped_streamlines += 1
            continue
        direct_source = _label_at_point(streamline[0], data, inverse_affine)
        direct_target = _label_at_point(streamline[-1], data, inverse_affine)
        source = assign_endpoint(streamline[0])
        target = assign_endpoint(streamline[-1])
        radially_assigned_endpoints += int(source != 0 and source != direct_source)
        radially_assigned_endpoints += int(target != 0 and target != direct_target)
        if source == 0 or target == 0:
            skipped_streamlines += 1
            if (
                (direct_source != 0 and direct_source not in allowed_node_labels)
                or (direct_target != 0 and direct_target not in allowed_node_labels)
            ):
                skipped_non_node_streamlines += 1
            continue
        streamline_weight = None
        if sift2_weights is not None:
            if streamline_index >= len(sift2_weights):
                skipped_streamlines += 1
                continue
            streamline_weight = sift2_weights[streamline_index]
        endpoint_assigned_streamlines += 1
        if source == target:
            self_connection_streamlines += 1
            if config.zero_diagonal:
                continue
        matrix_included_streamlines += 1
        edge = tuple(sorted((source, target))) if config.symmetric else (source, target)
        if streamline_weight is not None:
            edge_counts[edge] += streamline_weight
        else:
            edge_counts[edge] += 1
        if _metric_requires_length(config.metric):
            edge_length_sums[edge] += _streamline_length_mm(streamline)
            edge_length_counts[edge] += 1
        if scalar_data is not None and scalar_inverse_affine is not None:
            samples = [
                value
                for value in (_scalar_at_point(point, scalar_data, scalar_inverse_affine) for point in streamline)
                if value is not None
            ]
            if samples:
                edge_scalar_sums[edge] += float(np.mean(samples))
                edge_scalar_counts[edge] += 1
            else:
                skipped_scalar_streamlines += 1

    with matrix_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source_node", "target_node", config.metric])
        for (source, target), count in sorted(edge_counts.items()):
            if _metric_requires_scalar(config.metric):
                scalar_count = edge_scalar_counts[(source, target)]
                value = (edge_scalar_sums[(source, target)] / scalar_count) if scalar_count else ""
                writer.writerow([source, target, value])
            elif config.metric == "mean_length_mm":
                length_count = edge_length_counts[(source, target)]
                value = (edge_length_sums[(source, target)] / length_count) if length_count else ""
                writer.writerow([source, target, value])
            elif config.metric == "total_length_mm":
                writer.writerow([source, target, edge_length_sums[(source, target)]])
            else:
                writer.writerow([source, target, count])

    return {
        "node_count": node_count,
        "edge_count": len(edge_counts),
        "streamline_count_total": total_streamlines,
        "streamline_count_assigned": sum(edge_counts.values()),
        "streamline_count_endpoint_assigned": endpoint_assigned_streamlines,
        "streamline_count_matrix_included": matrix_included_streamlines,
        "streamline_count_self_connection": self_connection_streamlines,
        "streamline_count_skipped": skipped_streamlines,
        "streamline_count_skipped_non_node": skipped_non_node_streamlines,
        "streamline_count_scalar_sampled": sum(edge_scalar_counts.values()) if _metric_requires_scalar(config.metric) else None,
        "streamline_count_scalar_skipped": skipped_scalar_streamlines if _metric_requires_scalar(config.metric) else None,
        "streamline_count_length_sampled": sum(edge_length_counts.values()) if _metric_requires_length(config.metric) else None,
        "sift2_weight_count": len(sift2_weights) if sift2_weights is not None else None,
        "named_node_count": named_node_count,
        "unnamed_node_count": node_count - named_node_count,
        "node_filter": config.node_filter,
        "assignment_method": "radial_search" if config.assignment_radius_mm > 0 else "end_voxel",
        "assignment_radius_mm": config.assignment_radius_mm,
        "assignment_implementation": assignment_implementation,
        "endpoint_count_radially_assigned": radially_assigned_endpoints,
        "zero_diagonal": config.zero_diagonal,
        "input_label_count": len(unique_nonzero_labels),
        "excluded_node_label_count": len(unique_nonzero_labels) - node_count,
    }


def build_tck2connectome_command(config: ConnectomeConfig, matrix_path: Path) -> tuple[str, ...]:
    command = _resolve_command("tck2connectome", config.mrtrix_bin_dir)
    if command is None:
        raise RuntimeError("tck2connectome was not found on PATH.")
    cmd = [
        command,
        config.tractography_path.expanduser().resolve().as_posix(),
        config.parcellation_path.expanduser().resolve().as_posix(),
        matrix_path.as_posix(),
        "-stat_edge",
        "sum",
    ]
    if config.sift2_weights_path is not None:
        cmd.extend(["-tck_weights_in", config.sift2_weights_path.expanduser().resolve().as_posix()])
    if config.symmetric:
        cmd.append("-symmetric")
    if config.zero_diagonal:
        cmd.append("-zero_diagonal")
    if config.assignment_radius_mm > 0:
        cmd.extend(["-assignment_radial_search", str(config.assignment_radius_mm)])
    else:
        cmd.append("-assignment_end_voxels")
    # SPEC-GAP: ConnectomeConfig has no explicit thread option, so "explicit wins" has no path here;
    # the stage reservation sets tck2connectome -nthreads (the spec names connectome as a site).
    from shared.numeric_threads import reserved_stage_cpus
    reserved = reserved_stage_cpus()
    if reserved is not None:
        cmd.extend(["-nthreads", str(reserved)])
    cmd.append("-force")
    return tuple(cmd)


def _summarize_dense_connectome_matrix(matrix_path: Path, *, symmetric: bool) -> dict[str, object]:
    """Read an MRtrix tck2connectome dense CSV and report basic QC metrics."""

    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Dense connectome QC parsing requires numpy.") from exc

    rows: list[list[float]] = []
    with matrix_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        for row in reader:
            if not row:
                continue
            rows.append([float(value) for value in row if value != ""])
    if not rows:
        return {
            "node_count": 0,
            "edge_count": 0,
            "streamline_count_total": None,
            "streamline_count_assigned": 0,
            "streamline_count_skipped": None,
        }

    matrix = np.asarray(rows, dtype=float)
    if matrix.ndim != 2:
        raise RuntimeError(f"Connectome matrix is not 2D: {matrix_path}")
    node_count = int(matrix.shape[0])
    if symmetric and matrix.shape[0] == matrix.shape[1]:
        upper = np.triu(matrix)
        edge_count = int(np.count_nonzero(upper))
        assigned = float(np.trace(matrix) + np.triu(matrix, k=1).sum())
    else:
        edge_count = int(np.count_nonzero(matrix))
        assigned = float(matrix.sum())

    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "streamline_count_total": None,
        "streamline_count_assigned": int(assigned) if assigned.is_integer() else assigned,
        "streamline_count_skipped": None,
    }


def _load_dense_connectome_matrix(matrix_path: Path):
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Dense connectome matrix parsing requires numpy.") from exc

    rows: list[list[float]] = []
    with matrix_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        for row in reader:
            if not row:
                continue
            rows.append([float(value) for value in row if value != ""])
    if not rows:
        return np.zeros((0, 0), dtype=float)
    matrix = np.asarray(rows, dtype=float)
    if matrix.ndim != 2:
        raise RuntimeError(f"Connectome matrix is not 2D: {matrix_path}")
    return matrix


def _write_compressed_dense_connectome(
    dense_matrix_path: Path,
    sparse_matrix_path: Path,
    labels: Iterable[int],
    *,
    metric: str,
    symmetric: bool,
) -> dict[str, object]:
    """Compress an MRtrix dense matrix to label-valued sparse edge rows.

    MRtrix ``tck2connectome`` can produce very large dense matrices when the
    parcellation uses sparse FreeSurfer-style label values. This helper keeps
    only rows/columns that correspond to labels actually present in the
    parcellation image and writes a compact edge-list CSV.
    """

    import numpy as np

    matrix = _load_dense_connectome_matrix(dense_matrix_path)
    unique_labels = sorted({int(label) for label in labels if int(label) != 0})
    if matrix.shape[0] == len(unique_labels) and matrix.shape[1] == len(unique_labels):
        remap_mode = "sequential_sorted_labels"
        index_by_label = {label: index for index, label in enumerate(unique_labels)}
    else:
        remap_mode = "one_based_label_indices"
        index_by_label = {
            label: label - 1
            for label in unique_labels
            if label > 0 and label - 1 < matrix.shape[0] and label - 1 < matrix.shape[1]
        }

    edge_count = 0
    assigned = 0.0
    dropped_labels = [label for label in unique_labels if label not in index_by_label]
    with sparse_matrix_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source_node", "target_node", metric])
        for source in unique_labels:
            source_index = index_by_label.get(source)
            if source_index is None:
                continue
            target_labels = unique_labels
            if symmetric:
                target_labels = [label for label in unique_labels if label >= source]
            for target in target_labels:
                target_index = index_by_label.get(target)
                if target_index is None:
                    continue
                value = float(matrix[source_index, target_index])
                if not np.isfinite(value) or value == 0:
                    continue
                edge_count += 1
                assigned += value
                writer.writerow([source, target, int(value) if value.is_integer() else value])

    return {
        "node_count": len(unique_labels),
        "edge_count": edge_count,
        "streamline_count_total": None,
        "streamline_count_assigned": int(assigned) if assigned.is_integer() else assigned,
        "streamline_count_skipped": None,
        "dense_node_count": int(matrix.shape[0]) if matrix.ndim == 2 else 0,
        "dense_matrix_path": dense_matrix_path.as_posix(),
        "dense_remap_mode": remap_mode,
        "dense_remapped_label_count": len(index_by_label),
        "dense_dropped_label_count": len(dropped_labels),
        "dense_dropped_labels_preview": dropped_labels[:20],
    }


def _validate_config(config: ConnectomeConfig) -> None:
    if not config.subject_id.strip():
        raise ValueError("subject_id is required.")
    tractography = config.tractography_path.expanduser()
    parcellation = config.parcellation_path.expanduser()
    if not tractography.is_file():
        raise FileNotFoundError(f"Tractography dependency does not exist: {tractography}")
    if tractography.suffix.lower() not in SUPPORTED_TRACTOGRAPHY_EXTENSIONS:
        raise ValueError(f"Unsupported tractography format: {tractography.suffix}")
    if not parcellation.is_file():
        raise FileNotFoundError(f"DDParcel/parcellation dependency does not exist: {parcellation}")
    if not _has_supported_parcellation_suffix(parcellation):
        raise ValueError(f"Unsupported parcellation format: {parcellation.name}")
    if config.metric not in {
        "streamline_count",
        "sift2_weighted_streamline_count",
        "mean_scalar",
        "mean_fa",
        "mean_md",
        "mean_odi",
        "mean_length_mm",
        "total_length_mm",
    }:
        raise ValueError(
            "metric must be one of: streamline_count, sift2_weighted_streamline_count, mean_scalar, mean_fa, mean_md, mean_odi, mean_length_mm, total_length_mm."
        )
    if config.backend not in {"auto", "python", "mrtrix"}:
        raise ValueError("backend must be one of: auto, python, mrtrix.")
    if config.node_filter not in {"gray_matter", "all"}:
        raise ValueError("node_filter must be one of: gray_matter, all.")
    if not math.isfinite(config.assignment_radius_mm) or config.assignment_radius_mm < 0:
        raise ValueError("assignment_radius_mm must be a finite value greater than or equal to zero.")
    if _metric_requires_scalar(config.metric):
        if config.scalar_map_path is None:
            raise ValueError(f"metric='{config.metric}' requires scalar_map_path.")
        if not config.scalar_map_path.expanduser().is_file():
            raise FileNotFoundError(f"Scalar map does not exist: {config.scalar_map_path}")
    if config.metric == "sift2_weighted_streamline_count":
        if config.sift2_weights_path is None:
            raise ValueError("metric='sift2_weighted_streamline_count' requires sift2_weights_path.")
        if not config.sift2_weights_path.expanduser().is_file():
            raise FileNotFoundError(f"SIFT2 weights do not exist: {config.sift2_weights_path}")
        if config.tractography_path.expanduser().suffix.lower() != ".tck":
            raise ValueError("SIFT2 weighted connectome currently requires a .tck tractogram.")


def _infer_connectome_output_dir(config: ConnectomeConfig) -> Path:
    if config.output_dir is not None:
        return config.output_dir.expanduser().resolve()
    return default_task_output_dir_for_dependency(config.tractography_path, "connectome")


@scheduled_stage('connectome')
def run_connectome(config: ConnectomeConfig, *, log: Optional[Callable[[str], None]] = None) -> ConnectomeResult:
    """Prepare connectome outputs that explicitly depend on tractography and DDParcel."""

    _validate_config(config)
    output_dir = _infer_connectome_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    lut_path = _resolve_lut_path(config)
    stem = f"{config.subject_id}-connectome-{config.metric}"
    matrix_path = output_dir / f"{stem}.csv"
    nodes_path = output_dir / f"{config.subject_id}-connectome-nodes.tsv"
    manifest_path = output_dir / f"{stem}_manifest.json"
    qc_path = output_dir / f"{stem}_qc.json"

    backend_status = inspect_connectome_backends(config.mrtrix_bin_dir)

    manifest = {
        "kind": "connectome",
        "version": 1,
        "subject_id": config.subject_id,
        "executed": False,
        "status": "planned",
        "metric": config.metric,
        "requested_backend": config.backend,
        "node_source": config.node_source,
        "node_filter": config.node_filter,
        "assignment_method": "radial_search" if config.assignment_radius_mm > 0 else "end_voxel",
        "assignment_radius_mm": config.assignment_radius_mm,
        "symmetric": config.symmetric,
        "zero_diagonal": config.zero_diagonal,
        "compress_mrtrix_dense": config.compress_mrtrix_dense,
        "backend": backend_status,
        "dependencies": {
            "tractography": _file_info(config.tractography_path),
            "ddparcel_parcellation": _file_info(config.parcellation_path),
            "scalar_map": _file_info(config.scalar_map_path) if config.scalar_map_path is not None else None,
            "sift2_weights": _file_info(config.sift2_weights_path) if config.sift2_weights_path is not None else None,
            "lut": _file_info(lut_path) if lut_path is not None else None,
        },
        "outputs": {
            "matrix": matrix_path.as_posix(),
            "nodes": nodes_path.as_posix(),
            "qc": qc_path.as_posix(),
        },
    }
    qc = {
        "subject_id": config.subject_id,
        "status": "planned",
        "executed": False,
        "metric": config.metric,
        "node_count": None,
        "edge_count": None,
        "streamline_count_total": None,
        "streamline_count_assigned": None,
        "streamline_count_endpoint_assigned": None,
        "streamline_count_matrix_included": None,
        "streamline_count_self_connection": None,
        "streamline_count_skipped": None,
        "streamline_count_skipped_non_node": None,
        "streamline_count_scalar_sampled": None,
        "streamline_count_scalar_skipped": None,
        "streamline_count_length_sampled": None,
        "sift2_weight_count": None,
        "named_node_count": None,
        "unnamed_node_count": None,
        "node_filter": config.node_filter,
        "assignment_method": "radial_search" if config.assignment_radius_mm > 0 else "end_voxel",
        "assignment_radius_mm": config.assignment_radius_mm,
        "assignment_implementation": None,
        "endpoint_count_radially_assigned": None,
        "zero_diagonal": config.zero_diagonal,
        "input_label_count": None,
        "excluded_node_label_count": None,
        "return_code": None,
        "elapsed_seconds": 0.0,
        "backend": "planned",
        "dense_node_count": None,
        "dense_matrix_path": None,
        "dense_remap_mode": None,
        "dense_remapped_label_count": None,
        "dense_dropped_label_count": None,
        "error": None,
    }
    return_code: Optional[int] = None
    execution_error: Optional[str] = None
    start_time = time.monotonic()
    if config.execute:
        manifest["executed"] = True
        manifest["status"] = "running"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        try:
            try:
                use_mrtrix, backend_used, backend_reason = select_connectome_backend(config, backend_status)
            except RuntimeError as exc:
                manifest["backend_used"] = None
                manifest["backend_reason"] = str(exc)
                raise
            manifest["backend_used"] = backend_used
            manifest["backend_reason"] = backend_reason
            compress_dense = resolve_compress_mrtrix_dense(config, use_mrtrix)
            manifest["compress_mrtrix_dense"] = compress_dense
            if use_mrtrix:
                data, _ = _load_label_volume(config.parcellation_path.expanduser().resolve())
                label_names = _parse_label_names(lut_path)
                unique_labels = sorted({int(label) for label in data.ravel() if int(label) != 0})
                node_labels = _connectome_node_labels(
                    unique_labels,
                    node_filter=config.node_filter,
                )
                command_config = config
                if node_labels != unique_labels:
                    filtered_parcellation = output_dir / f"{config.subject_id}-connectome-node-parcellation.nii.gz"
                    _write_filtered_parcellation(
                        config.parcellation_path.expanduser().resolve(),
                        filtered_parcellation,
                        allowed_labels=node_labels,
                    )
                    command_config = replace(config, parcellation_path=filtered_parcellation)
                    manifest["outputs"]["filtered_parcellation"] = filtered_parcellation.as_posix()  # type: ignore[index]

                mrtrix_matrix_path = (
                    output_dir / f"{stem}_dense.csv" if compress_dense else matrix_path
                )
                if compress_dense:
                    manifest["outputs"]["dense_matrix"] = mrtrix_matrix_path.as_posix()  # type: ignore[index]
                command = build_tck2connectome_command(command_config, mrtrix_matrix_path)
                manifest["command"] = list(command)
                command_env = startup_environment()
                if os.name == "nt" and not command_env.get("HOME"):
                    command_env["HOME"] = command_env.get("USERPROFILE") or str(Path.home())
                if config.mrtrix_bin_dir is not None:
                    command_env["PATH"] = os.pathsep.join(
                        filter(None, (str(config.mrtrix_bin_dir.expanduser().resolve()), command_env.get("PATH")))
                    )
                completed = subprocess.run(command, check=True, env=command_env)
                return_code = completed.returncode
                try:
                    _write_nodes_table(nodes_path, node_labels, label_names=label_names)
                    qc.update(
                        {
                            "named_node_count": sum(1 for label in node_labels if label in label_names),
                            "unnamed_node_count": sum(1 for label in node_labels if label not in label_names),
                            "input_label_count": len(unique_labels),
                            "excluded_node_label_count": len(unique_labels) - len(node_labels),
                        }
                    )
                except Exception:
                    _write_nodes_table(nodes_path, [], label_names={})
                if compress_dense:
                    qc.update(
                        _write_compressed_dense_connectome(
                            mrtrix_matrix_path,
                            matrix_path,
                            node_labels,
                            metric=config.metric,
                            symmetric=config.symmetric,
                        )
                    )
                else:
                    summary = _summarize_dense_connectome_matrix(matrix_path, symmetric=config.symmetric)
                    qc.update(summary)
                    # MRtrix allocates rows up to the maximum label value;
                    # unused rows are not anatomical network nodes.
                    qc["dense_node_count"] = summary["node_count"]
                    qc["dense_matrix_path"] = matrix_path.as_posix()
                    qc["node_count"] = len(node_labels)
                total_streamlines = _tck_header_streamline_count(config.tractography_path)
                qc["streamline_count_total"] = total_streamlines
                if config.sift2_weights_path is not None:
                    qc["sift2_weight_count"] = len(_read_sift2_weights(config.sift2_weights_path))
                if (
                    config.metric == "streamline_count"
                    and total_streamlines is not None
                    and not config.zero_diagonal
                ):
                    assigned = qc.get("streamline_count_assigned")
                    if isinstance(assigned, (int, float)):
                        qc["streamline_count_skipped"] = max(0, total_streamlines - int(assigned))
                qc.update(
                    {
                        "backend": "mrtrix3.tck2connectome",
                        "assignment_implementation": "mrtrix3.tck2connectome",
                    }
                )
            else:
                summary = _write_python_connectome(config, matrix_path, nodes_path, lut_path=lut_path)
                qc.update(summary)
                qc.update(
                    {
                        "backend": (
                            "python.radial_search"
                            if config.assignment_radius_mm > 0
                            else "python.endpoint_labels"
                        )
                    }
                )
                return_code = 0
        except Exception as exc:
            execution_error = str(exc)
            return_code = 1
        finally:
            if execution_error is None:
                output_errors = _connectome_output_errors(qc, matrix_path, nodes_path)
                if output_errors:
                    execution_error = "Connectome execution produced incomplete outputs: " + "; ".join(output_errors)
                    return_code = 1
            elapsed_seconds = time.monotonic() - start_time
            qc.update(
                {
                    "status": "failed" if execution_error else "completed",
                    "executed": True,
                    "return_code": return_code,
                    "elapsed_seconds": elapsed_seconds,
                    "error": execution_error,
                }
            )
    else:
        with matrix_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["source_node", "target_node", config.metric])
        try:
            data, _ = _load_label_volume(config.parcellation_path.expanduser().resolve())
            label_names = _parse_label_names(lut_path)
            node_labels = _connectome_node_labels(
                data.ravel(),
                node_filter=config.node_filter,
            )
            _write_nodes_table(nodes_path, node_labels, label_names=label_names)
        except Exception:
            nodes_path.write_text("node_id\tlabel\tname\n", encoding="utf-8")
    # Keep the native manifest and QC lifecycle fields authoritative and in
    # lockstep.  Previously an executed connectome left its manifest at
    # ``running`` even after the QC record had reached ``completed`` or
    # ``failed``.
    manifest["executed"] = bool(qc["executed"])
    manifest["status"] = str(qc["status"])
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    qc_path.write_text(json.dumps(qc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if execution_error is not None:
        raise RuntimeError(f"Connectome execution failed: {execution_error}")
    _emit(f"[Connectome] Prepared connectome scaffold: {manifest_path}", log)

    return ConnectomeResult(
        subject_id=config.subject_id,
        output_dir=output_dir,
        matrix_path=matrix_path,
        nodes_path=nodes_path,
        manifest_path=manifest_path,
        qc_path=qc_path,
        backend_status=backend_status,
        executed=bool(config.execute),
        return_code=return_code,
    )
