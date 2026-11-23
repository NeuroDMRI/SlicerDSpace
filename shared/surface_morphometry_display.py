"""Native display of the surface post-processing results (no SlicerFreeSurfer extension).

The task ``surface.ddsurfer_postprocess`` publishes FreeSurfer-format files in
``outputs/01_anatomical_priors/surface_morphometry/``: ``{lh,rh}.thickness/.curv/.sulc/.area`` (one value per white-surface
vertex), ``{lh,rh}.aparc.annot`` / ``.aparc.a2009s.annot`` (one label per vertex), ``{lh,rh}.inflated`` (a surface) and
``{lh,rh}.<atlas>.stats`` (text tables). Slicer core has no reader for them; this module reads them with
``nibabel.freesurfer`` (and parses ``.stats`` tables natively).

- A measure or an annotation is shown on whichever cortical surface(s) of the same hemisphere are currently open
  (``white``, ``pial``, or ``inflated``), defaulting to ``white`` when none is open: the published
  ``cortical_surface/{hemi}.{white,pial}.obj`` (never the LPS ``.stl``, whose vertex order differs) or
  ``surface_morphometry/{hemi}.inflated``. Each row gets its own model node (carrier geometry + this row's scalar).
- Before the scalar is attached: the value count must equal the point count, and the carrier's sha256 must equal the
  ``native_sha256`` that post-processing recorded in ``structural/manifest.json`` (when that record is readable).
- ``{hemi}.inflated`` becomes its own model, moved from the FreeSurfer surface space to scanner space with the
  volume geometry of the file footer (the transform SlicerFreeSurfer's reader applies).
- ``{hemi}.<atlas>.stats`` opens in an in-Slicer table dialog (with optional ``vtkMRMLTableNode`` export and TSV copy).

The pure functions (rows, carrier, checks, ranges, label mapping, footer transform, sibling rule, stats parsing) run
without the application; :func:`load_morphometry_row` builds scene nodes and :func:`show_stats_dialog` displays tables.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Optional

PREFIX = "SlicerDDSurfer."
PRODUCT = "surface.ddsurfer_postprocess"
SURFACE_PRODUCT = "surface.ddsurfer"
MORPHOMETRY_ROW_KIND = "surface_morphometry"  # ROLE_KIND of a Results row of this product (plain eye)
HEMISPHERES = ("lh", "rh")
SURFACES = ("white", "pial", "inflated")
_SURFACE_TEXT = {"white": ("white", "白质"), "pial": ("pial", "软脑膜"), "inflated": ("inflated", "膨胀")}
_SURFACE_OBJ = re.compile(r"^(lh|rh)\.(white|pial)\.obj$")
_SURFACE_CARRIER = re.compile(r"^(lh|rh)\.(?:(white|pial)\.obj|(inflated))$")
MEASURES = ("thickness", "curv", "sulc", "area")
ANNOTATIONS = ("aparc", "aparc.a2009s")
_KIND_ORDER = ("measure", "annotation", "inflated", "stats")
_ROW = re.compile(r"^(lh|rh)\.(?:(thickness|curv|sulc|area|inflated)|(aparc|aparc\.a2009s)\.annot|(.+)\.stats)$")

# Display: one sequential table for thickness and area, one diverging table for curv and sulc. Both exist in Slicer
# 5.10 (Base/Logic/Resources/ColorFiles at tag v5.10.0) and 5.12 (scene node ids listed in 5.12.4). A fallback keeps
# the click working if an id is missing.
SEQUENTIAL_TABLES = ("vtkMRMLColorTableNodeFileViridis.txt", "vtkMRMLColorTableNodeFileColdToHotRainbow.txt",
                     "vtkMRMLColorTableNodeRainbow")
DIVERGING_TABLES = ("vtkMRMLColorTableNodeFileDivergingBlueRed.txt", "vtkMRMLColorTableNodeFileColdToHotRainbow.txt",
                    "vtkMRMLColorTableNodeRainbow")
THICKNESS_RANGE = (1.0, 4.5)
UNKNOWN_GREY = (0.5, 0.5, 0.5)
LEGEND_TITLES = {"thickness": "Thickness (mm)", "curv": "Curvature (1/mm)", "sulc": "Sulcal depth (mm)",
                 "area": "Vertex area (mm²)"}

# Display-only lateral separation for FreeSurfer inflated surfaces (never modifies files on disk or polydata points).
# FreeSurfer lh.inflated and rh.inflated both sit centered around the RAS origin and visually overlap in 3D view;
# applying -35 mm (lh) / +35 mm (rh) via owned vtkMRMLLinearTransformNode separates them cleanly and can be toggled
# ON/OFF dynamically in one click.
INFLATED_SEPARATION_OFFSET_MM = 35.0
INFLATED_HEMISPHERE_ATTR = PREFIX + "InflatedHemisphere"
INFLATED_TRANSFORM_ATTR = PREFIX + "InflatedTransformNodeID"
INFLATED_OFFSET_ATTR = PREFIX + "InflatedOffsetX"
_INFLATED_SEPARATION_ENABLED: bool = True

_MEASURE_TEXT = {"thickness": ("thickness", "皮层厚度"), "curv": ("curvature", "曲率"),
                 "sulc": ("sulcal depth", "脑沟深度"), "area": ("vertex area", "顶点面积")}
_ATLAS_TEXT = {"aparc": ("Desikan (aparc)", "Desikan（aparc）"),
               "aparc.a2009s": ("Destrieux (aparc.a2009s)", "Destrieux（aparc.a2009s）")}


class MorphometryError(RuntimeError):
    """A clear, bilingual reason why a post-processing row cannot be shown (the Results eye shows it as is)."""


def _bilingual(english: str, chinese: str) -> str:
    return f"{english}\n{chinese}"


NIBABEL_MISSING = _bilingual(
    "nibabel is missing in Slicer's Python, so the surface post-processing results cannot be shown. "
    "Run \"Prepare selected analysis environment\".",
    "Slicer 的 Python 中缺少 nibabel，无法显示表面后处理结果。请运行“准备所选分析环境”。")


# ---------------------------------------------------------------------------------------------------- rows
def morphometry_row(product: Any, path: Any) -> Optional[tuple[str, str, str]]:
    """(kind, hemisphere, name) of a published post-processing file, else None.

    Accepts by product AND file name: a ``x.thickness`` of another product, or an unknown name of this product, is not
    a row. kind: "measure" (name thickness/curv/sulc/area), "annotation" (name aparc/aparc.a2009s), "inflated",
    "stats" (name = the atlas).
    """

    if str(product or "").split("@", 1)[0] != PRODUCT:
        return None
    match = _ROW.match(Path(str(path)).name)
    if match is None:
        return None
    hemi, surface, atlas, stats = match.groups()
    if surface == "inflated":
        return "inflated", hemi, "inflated"
    if surface:
        return "measure", hemi, surface
    if atlas:
        return "annotation", hemi, atlas
    return "stats", hemi, stats


def row_order(product: Any, path: Any) -> tuple:
    """Sort key of a row: lh before rh; thickness, curv, sulc, area, aparc, aparc.a2009s, inflated, stats."""

    row = morphometry_row(product, path)
    if row is None:
        return (len(HEMISPHERES), len(_KIND_ORDER), 0, str(path))
    kind, hemi, name = row
    within = (MEASURES.index(name) if kind == "measure" else
              ANNOTATIONS.index(name) if kind == "annotation" and name in ANNOTATIONS else
              ANNOTATIONS.index(name) if kind == "stats" and name in ANNOTATIONS else len(ANNOTATIONS))
    return (HEMISPHERES.index(hemi), _KIND_ORDER.index(kind), within, Path(str(path)).name)


def row_tooltip(path: Any, *, zh: bool) -> str:
    """What the row's eye does, in the current language (an empty text for a file that is not a row)."""

    row = morphometry_row(PRODUCT, path)
    if row is None:
        return ""
    kind, _hemi, name = row
    if kind == "measure":
        english, chinese = _MEASURE_TEXT[name]
        return (
            f"在已打开的皮层表面（白质/软脑膜/膨胀）上显示{chinese}（默认白质表面）"
            if zh
            else f"Show {english} on the open cortical surface (white, pial, or inflated; defaults to white)"
        )
    if kind == "annotation":
        english, chinese = _ATLAS_TEXT.get(name, (name, name))
        return (
            f"在已打开的皮层表面（白质/软脑膜/膨胀）上显示 {chinese} 分区（默认白质表面）"
            if zh
            else f"Show the {english} parcellation on the open cortical surface (white, pial, or inflated; defaults to white)"
        )
    if kind == "inflated":
        return "显示膨胀表面" if zh else "Show the inflated surface"
    return "在 Slicer 中查看统计表" if zh else "View the statistics table in Slicer"


def surface_from_obj_path(path: Any) -> Optional[tuple[str, str]]:
    """Return ``(hemisphere, surface)`` for ``{lh,rh}.{white,pial}.obj``, else None."""

    if not path:
        return None
    match = _SURFACE_OBJ.match(Path(str(path)).name)
    return (match.group(1), match.group(2)) if match is not None else None


def surface_from_carrier_path(path: Any) -> Optional[tuple[str, str]]:
    """Return ``(hemisphere, surface)`` for ``{lh,rh}.{white,pial}.obj`` or ``{lh,rh}.inflated``, else None."""

    if not path:
        return None
    match = _SURFACE_CARRIER.match(Path(str(path)).name)
    if match is None:
        return None
    hemi, obj_surf, inflated_surf = match.groups()
    return (hemi, obj_surf or inflated_surf)


surface_from_path = surface_from_carrier_path


def carrier_surface_of_node(node: Any) -> Optional[str]:
    """Return ``'white'``, ``'pial'``, or ``'inflated'`` for a morphometry overlay model node, else None."""

    if node is None or not hasattr(node, "GetAttribute"):
        return None
    surf = node.GetAttribute(PREFIX + "CarrierSurface")
    if surf in SURFACES:
        return surf
    carrier = node.GetAttribute(PREFIX + "CarrierPath")
    parsed = surface_from_carrier_path(carrier)
    return parsed[1] if parsed is not None else None


# ------------------------------------------------------------------------------------------ carrier checks
def _completed(task: dict) -> bool:
    status = task.get("status")
    return status in {"completed", "reused"} or (status == "skipped" and task.get("reason") == "completed outputs found")


def surface_carrier(manifest: dict, hemi: str, surface: str = "white") -> Optional[Path]:
    """The published ``{hemi}.{surface}.obj`` (or ``{hemi}.inflated``) that the run records, or None (never ``.stl``)."""

    if surface == "inflated":
        target_product = PRODUCT
        target_name = f"{hemi}.inflated"
    else:
        target_product = SURFACE_PRODUCT
        target_name = f"{hemi}.{surface}.obj"
    for task in manifest.get("tasks") or []:
        if not isinstance(task, dict) or str(task.get("name") or "").split("@", 1)[0] != target_product:
            continue
        if not _completed(task):
            continue
        for value in (task.get("outputs") or {}).values():
            if not isinstance(value, str) or not value:
                continue
            path = Path(value).expanduser()
            if path.is_absolute() and path.name == target_name and path.is_file():
                return path
    return None


def white_carrier(manifest: dict, hemi: str) -> Optional[Path]:
    """The published ``{hemi}.white.obj`` that the run's surface task records, or None (never the ``.stl``)."""

    return surface_carrier(manifest, hemi, "white")


def inflated_carrier(manifest: dict, hemi: str, row_path: Any = None) -> Optional[Path]:
    """The published ``{hemi}.inflated`` from ``surface.ddsurfer_postprocess`` outputs or the published layout."""

    found = surface_carrier(manifest, hemi, "inflated")
    if found is not None:
        return found
    if row_path is not None:
        return published_surface_carrier(manifest, hemi, row_path, "inflated")
    from shared.io_layout import subject_layout

    subject_root, output_root = manifest.get("subject_root"), manifest.get("output_root")
    if not subject_root:
        return None
    try:
        layout = subject_layout(Path(str(subject_root)), output_root=Path(str(output_root)) if output_root else None)
        carrier = layout.surface_morphometry / f"{hemi}.inflated"
        return carrier if carrier.is_file() else None
    except (OSError, RuntimeError, ValueError, TypeError):
        return None


def published_surface_carrier(manifest: dict, hemi: str, row_path: Any, surface: str = "white") -> Optional[Path]:
    """Review M4: ``cortical_surface/{hemi}.{surface}.obj`` (or ``surface_morphometry/{hemi}.inflated``) of the
    published layout (shared.io_layout) of the run's output root, used when the manifest records no completed task
    with that file. Only when the row itself lies in the ``surface_morphometry`` folder of that same layout."""

    from shared.io_layout import subject_layout

    subject_root, output_root = manifest.get("subject_root"), manifest.get("output_root")
    if not subject_root or not row_path:
        return None
    try:
        layout = subject_layout(Path(str(subject_root)), output_root=Path(str(output_root)) if output_root else None)
        if Path(str(row_path)).expanduser().resolve().parent != layout.surface_morphometry.resolve():
            return None
        carrier = (
            layout.surface_morphometry / f"{hemi}.inflated"
            if surface == "inflated"
            else layout.cortical_surface / f"{hemi}.{surface}.obj"
        )
        return carrier if carrier.is_file() else None
    except (OSError, RuntimeError, ValueError, TypeError):
        return None


def published_white_carrier(manifest: dict, hemi: str, row_path: Any) -> Optional[Path]:
    return published_surface_carrier(manifest, hemi, row_path, "white")


def recorded_carrier_sha256(task: dict, hemi: str, surface: str = "white") -> Optional[str]:
    """``native_sha256["{hemi}.{surface}.obj"]`` of the post-processing ``structural/manifest.json``; None when unreadable."""

    outputs = task.get("outputs") or {}
    candidates = [task.get("manifest_path")]
    if outputs.get("structural_root"):
        candidates.append(str(Path(str(outputs["structural_root"])) / "manifest.json"))
    key = f"{hemi}.inflated" if surface == "inflated" else f"{hemi}.{surface}.obj"
    for value in candidates:
        if not value:
            continue
        try:
            record = json.loads(Path(str(value)).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        hashes = record.get("native_sha256") if isinstance(record, dict) else None
        digest = hashes.get(key) if isinstance(hashes, dict) else None
        if isinstance(digest, str) and digest:
            return digest.lower()
    return None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_carrier(manifest: dict, task: dict, hemi: str, row_path: Any = None, surface: str = "white") -> Path:
    """The white, pial, or inflated surface to carry a measure of ``hemi``; a bilingual MorphometryError on error.

    The task's recorded file comes first; else the published file of the row's layout (review M4). The
    checksum check is the same for both."""

    en_surf, zh_surf = _SURFACE_TEXT.get(surface, (surface, surface))
    if surface == "inflated":
        carrier = inflated_carrier(manifest, hemi, row_path)
        if carrier is None:
            raise MorphometryError(_bilingual(
                f"This run has no {hemi}.inflated surface. Re-run Surface postprocess to show this file on the inflated surface.",
                f"此运行没有 {hemi}.inflated {zh_surf}表面。请重新运行“表面后处理”后再在{zh_surf}表面上显示此文件。"))
    else:
        carrier = surface_carrier(manifest, hemi, surface) or published_surface_carrier(manifest, hemi, row_path, surface)
        if carrier is None:
            raise MorphometryError(_bilingual(
                f"This run has no {hemi}.{surface}.obj surface. Older runs publish only the .stl surface, whose vertex order "
                "differs from the post-processing values. Re-run Surface to show this file.",
                f"此运行没有 {hemi}.{surface}.obj {zh_surf}表面。旧的运行只发布 .stl 表面，其顶点顺序与后处理数值不同。"
                "请重新运行“表面重建”后再显示此文件。"))
    expected = recorded_carrier_sha256(task, hemi, surface)
    if expected is not None and file_sha256(carrier) != expected:
        raise MorphometryError(_bilingual(
            f"{carrier.name} is not the {en_surf} surface that surface post-processing used (its checksum differs from "
            "the post-processing record). Re-run Surface postprocess.",
            f"{carrier.name} 不是表面后处理所用的{zh_surf}表面（校验值与后处理记录不同）。请重新运行“表面后处理”。"))
    return carrier


def check_count(values: int, points: int, file_name: str, carrier_name: str) -> None:
    if int(values) != int(points):
        raise MorphometryError(_bilingual(
            f"{file_name} has {values} values, but the surface {carrier_name} has {points} points; "
            "the file does not belong to this surface.",
            f"{file_name} 有 {values} 个数值，但表面 {carrier_name} 有 {points} 个顶点；此文件不属于该表面。"))


def active_carrier_surfaces(slicer, manifest: dict, hemi: str, row_path: Any = None) -> tuple[str, ...]:
    """Return currently visible cortical surface(s) (subset of ``('white', 'pial', 'inflated')``) for ``hemi``."""

    carrier_keys: dict[str, str] = {}
    for surf in SURFACES:
        if surf == "inflated":
            path = inflated_carrier(manifest, hemi, row_path)
        else:
            path = surface_carrier(manifest, hemi, surf) or published_surface_carrier(manifest, hemi, row_path, surf)
        key = _file_key(path)
        if key:
            carrier_keys[surf] = key
    if not carrier_keys:
        return ()
    from shared.result_scene import result_nodes

    try:
        scene_nodes = result_nodes(slicer)
    except Exception:
        return ()
    found: set[str] = set()
    for node in scene_nodes:
        if node is None or not node.IsA("vtkMRMLModelNode"):
            continue
        try:
            if not node.GetDisplayVisibility():
                continue
        except Exception:
            continue
        node_files = {
            _file_key(node.GetAttribute(PREFIX + "SourcePath")),
            _file_key(node.GetAttribute(PREFIX + "CarrierPath")),
        }
        for surf, key in carrier_keys.items():
            if key in node_files:
                found.add(surf)
    return tuple(surf for surf in SURFACES if surf in found)


def desired_carrier_surfaces(slicer, manifest: dict, hemi: str, row_path: Any = None) -> tuple[str, ...]:
    """Target surface(s) for a post-processing overlay: open surface(s) if any, else default to ``('white',)``."""

    active = active_carrier_surfaces(slicer, manifest, hemi, row_path)
    return active if active else ("white",)


# --------------------------------------------------------------------------------------- values and colours
def measure_range(name: str, values) -> tuple[float, float]:
    """Display range: thickness 1.0-4.5 mm; curv/sulc symmetric at the 98th percentile of |v|; area 0 to the 98th."""

    import numpy as np

    if name == "thickness":
        return THICKNESS_RANGE
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if name in {"curv", "sulc"}:
        magnitude = np.abs(finite)
        limit = float(np.percentile(magnitude, 98)) if magnitude.size else 0.0
        if not limit > 0:
            limit = float(magnitude.max()) if magnitude.size and magnitude.max() > 0 else 1.0
        return (-limit, limit)
    positive = finite[finite >= 0]
    high = float(np.percentile(positive, 98)) if positive.size else 0.0
    if not high > 0:
        high = float(positive.max()) if positive.size and positive.max() > 0 else 1.0
    return (0.0, high)


def _text(name) -> str:
    return name.decode("utf-8", "replace") if isinstance(name, bytes) else str(name)


def annotation_colors(raw_ids, ctab, names, entries: Optional[dict] = None
                      ) -> tuple[Any, list[tuple[str, float, float, float]]]:
    """Map raw annotation values to colour-table rows: row 0 = unknown (neutral grey), row j+1 = the j-th used ctab row.

    The table ids (column 4 of ``ctab``) need not be sequential; a value that is in no used entry (FreeSurfer's -1 or
    0 "no label") goes to row 0.

    Review M2: for a version-2 file with sparse table indices nibabel returns one ``ctab`` row per index up to the
    maximum (unused rows are zero) and a COMPACT ``names`` list, so ``names[k]`` does not belong to ``ctab[k]``.
    Used rows and names: ``entries`` ({table index: name}, parsed from the file) when given; else every row with
    ``names[k]`` when ``len(names) == len(ctab)`` (contiguous table, unchanged); else the rows that are not all zero,
    named ``label <index>``. An unused row never becomes a label or matches a vertex.
    """

    import numpy as np

    table = np.asarray(ctab)
    rows = len(table)
    if entries:
        used = [(index, _text(name)) for index, name in sorted(entries.items()) if 0 <= int(index) < rows]
    elif len(names) == rows:
        used = [(index, _text(name)) for index, name in enumerate(names)]
    else:
        used = [(index, f"label {index}") for index in range(rows) if np.any(table[index, :4] != 0)]
    raw = np.asarray(raw_ids).astype(np.int64)
    table_ids = (table[[index for index, _ in used], 4].astype(np.int64) if used else np.zeros(0, dtype=np.int64))
    indices = np.zeros(raw.shape, dtype=np.int32)
    order = np.argsort(table_ids, kind="stable")
    if table_ids.size:
        positions = np.searchsorted(table_ids[order], raw)
        positions = np.clip(positions, 0, table_ids.size - 1)
        hit = table_ids[order][positions] == raw
        indices[hit] = order[positions[hit]].astype(np.int32) + 1
    colors = [("unknown", *UNKNOWN_GREY)]
    for index, text in used:
        red, green, blue = (float(value) / 255 for value in table[index, :3])
        if text.strip().lower() == "unknown":  # the table's own "unknown" entry (aparc: dark purple; a2009s: black)
            red, green, blue = UNKNOWN_GREY
        colors.append((text, red, green, blue))
    return indices, colors


def annotation_table_entries(path) -> Optional[dict]:
    """{table index: name} of a version-2 annotation's colour table, read from the file; None for another layout or
    any read problem (the caller then uses neutral names). Layout (big-endian int32): vertex count, (vertex, value)
    pairs, table flag, -2, max index, table-name length + name, entry count, then per entry: index, name length +
    name, R, G, B, T."""

    import numpy as np

    try:
        data = Path(path).read_bytes()

        def number(offset: int) -> int:
            if offset < 0 or offset + 4 > len(data):
                raise ValueError("truncated annotation")
            return int(np.frombuffer(data, dtype=">i4", count=1, offset=offset)[0])

        offset = 4 + 8 * number(0)
        if number(offset) != 1 or number(offset + 4) != -2:
            return None
        max_index = number(offset + 8)
        offset += 12
        offset += 4 + number(offset)
        count = number(offset)
        offset += 4
        if not 0 <= count <= max(0, max_index) or count > len(data) // 24:
            return None
        entries = {}
        for _ in range(count):
            index, length = number(offset), number(offset + 4)
            if length < 0 or offset + 8 + length + 16 > len(data):
                return None
            name = data[offset + 8:offset + 8 + length].split(b"\x00", 1)[0]
            entries[index] = name.decode("utf-8", "replace")
            offset += 8 + length + 16
        return entries
    except (OSError, ValueError, TypeError):
        return None


def read_annotation(nibabel_module, path) -> tuple[Any, list[tuple[str, float, float, float]]]:
    raw, ctab, names = nibabel_module.freesurfer.read_annot(str(path), orig_ids=True)
    entries = annotation_table_entries(path) if len(names) != len(ctab) else None
    return annotation_colors(raw, ctab, names, entries)


def annotation_scalar_range(color_count: int) -> tuple[float, float]:
    """The manual scalar range that maps integer i to table row i (Slicer's colour-node range made one flat colour)."""

    return (0.0, float(max(1, color_count - 1)))


def surface_to_scanner(metadata) -> Optional[Any]:
    """4x4 transform from FreeSurfer surface (tkr) space to scanner RAS from a surface footer; None without one."""

    import numpy as np

    try:
        shape = np.asarray(metadata["volume"], dtype=np.float64)
        zooms = np.asarray(metadata["voxelsize"], dtype=np.float64)
        cosines = np.vstack([metadata["xras"], metadata["yras"], metadata["zras"]]).astype(np.float64).T
        center = np.asarray(metadata["cras"], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    if shape.shape != (3,) or zooms.shape != (3,) or cosines.shape != (3, 3) or center.shape != (3,):
        return None
    scanner = np.eye(4)
    scanner[:3, :3] = cosines * zooms
    scanner[:3, 3] = center - scanner[:3, :3] @ (shape / 2)
    tkr = np.eye(4)
    tkr[:3, :3] = np.array([[-1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]]) * zooms
    tkr[:3, 3] = -tkr[:3, :3] @ (shape / 2)
    try:
        return scanner @ np.linalg.inv(tkr)
    except np.linalg.LinAlgError:
        return None


def read_inflated(nibabel_module, path) -> tuple[Any, Any]:
    """Points in scanner RAS and faces of a FreeSurfer surface; a reflecting transform also reverses the winding."""

    import numpy as np

    points, faces, metadata = nibabel_module.freesurfer.read_geometry(str(path), read_metadata=True)
    transform = surface_to_scanner(metadata or {})
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if transform is not None:
        points = points @ transform[:3, :3].T + transform[:3, 3]
        if np.linalg.det(transform[:3, :3]) < 0:
            faces = faces[:, ::-1]
    return points, faces


# ------------------------------------------------------------------------ inflated hemisphere display separation
def inflated_separation_enabled() -> bool:
    """Return whether display-only L/R lateral separation (±35 mm X) is enabled for inflated surfaces."""

    return bool(_INFLATED_SEPARATION_ENABLED)


def inflated_separation_offset_mm(
    hemi: str,
    *,
    enabled: Optional[bool] = None,
    offset_mm: float = INFLATED_SEPARATION_OFFSET_MM,
) -> float:
    """RAS X-axis display shift in mm for an inflated hemisphere (``-offset`` for lh, ``+offset`` for rh, else 0)."""

    if hemi not in HEMISPHERES:
        raise ValueError(f"Unsupported hemisphere for inflated separation: {hemi!r}")
    active = _INFLATED_SEPARATION_ENABLED if enabled is None else bool(enabled)
    if not active:
        return 0.0
    magnitude = abs(float(offset_mm))
    return -magnitude if hemi == "lh" else magnitude


def inflated_hemisphere_of_node(node: Any) -> Optional[str]:
    """Return ``'lh'`` or ``'rh'`` if ``node`` is a loaded inflated surface or an overlay on an inflated surface."""

    if node is None or not hasattr(node, "GetAttribute"):
        return None
    try:
        if hasattr(node, "IsA") and not node.IsA("vtkMRMLModelNode"):
            return None
    except Exception:
        return None
    tagged = node.GetAttribute(INFLATED_HEMISPHERE_ATTR)
    if tagged in HEMISPHERES:
        return tagged
    carrier_surf = carrier_surface_of_node(node)
    for attr in (PREFIX + "CarrierPath", PREFIX + "SourcePath"):
        parsed = surface_from_carrier_path(node.GetAttribute(attr))
        if parsed is not None and parsed[1] == "inflated":
            return parsed[0]
        if carrier_surf == "inflated" and parsed is not None:
            return parsed[0]
    if carrier_surf == "inflated":
        parsed_row = morphometry_row(PRODUCT, node.GetAttribute(PREFIX + "SourcePath") or "")
        if parsed_row is not None and parsed_row[1] in HEMISPHERES:
            return parsed_row[1]
    return None


def iter_inflated_model_nodes(slicer, *, visible_only: bool = False) -> list:
    """Return every inflated model node (plain or morphometry overlay on inflated) currently in the Slicer scene."""

    if slicer is None:
        return []
    candidates = []
    util = getattr(slicer, "util", None)
    if util is not None and hasattr(util, "getNodesByClass"):
        try:
            candidates = list(util.getNodesByClass("vtkMRMLModelNode"))
        except Exception:
            candidates = []
    elif getattr(slicer, "mrmlScene", None) is not None and hasattr(slicer.mrmlScene, "GetNodesByClass"):
        try:
            collection = slicer.mrmlScene.GetNodesByClass("vtkMRMLModelNode")
            if collection is not None:
                candidates = [collection.GetItemAsObject(i) for i in range(collection.GetNumberOfItems())]
        except Exception:
            candidates = []
    matched = []
    for node in candidates:
        if inflated_hemisphere_of_node(node) is None:
            continue
        if visible_only:
            try:
                if not node.GetDisplayVisibility():
                    continue
            except Exception:
                continue
        matched.append(node)
    return matched


def has_visible_inflated_surface(slicer) -> bool:
    """Return True when at least one inflated surface (or overlay on an inflated surface) is currently visible."""

    return bool(iter_inflated_model_nodes(slicer, visible_only=True))


def _set_linear_transform_x_offset(transform_node: Any, offset_x: float) -> None:
    """Set a pure X-axis translation on ``transform_node`` without touching polydata vertices or disk files."""

    if transform_node is None:
        return
    try:
        import vtk

        matrix = vtk.vtkMatrix4x4()
        matrix.Identity()
        matrix.SetElement(0, 3, float(offset_x))
    except Exception:
        matrix = None
    setter = getattr(transform_node, "SetMatrixTransformToParent", None)
    if callable(setter) and matrix is not None:
        setter(matrix)
    elif hasattr(transform_node, "SetAndObserveMatrixTransformToParent") and matrix is not None:
        transform_node.SetAndObserveMatrixTransformToParent(matrix)
    if hasattr(transform_node, "SetAttribute"):
        transform_node.SetAttribute(INFLATED_OFFSET_ATTR, f"{float(offset_x):.1f}")


def apply_inflated_separation(
    slicer,
    node: Any,
    hemi: str,
    *,
    enabled: Optional[bool] = None,
    offset_mm: float = INFLATED_SEPARATION_OFFSET_MM,
) -> Any:
    """Attach/update an owned display-only ``vtkMRMLLinearTransformNode`` for ``node`` without modifying disk files."""

    dx = inflated_separation_offset_mm(hemi, enabled=enabled, offset_mm=offset_mm)
    node.SetAttribute(INFLATED_HEMISPHERE_ATTR, hemi)
    node.SetAttribute(INFLATED_OFFSET_ATTR, f"{dx:.1f}")
    scene = getattr(slicer, "mrmlScene", None)
    if scene is None:
        return None
    transform = None
    existing_id = node.GetAttribute(INFLATED_TRANSFORM_ATTR)
    if existing_id and hasattr(scene, "GetNodeByID"):
        candidate = scene.GetNodeByID(existing_id)
        if candidate is not None and (not hasattr(candidate, "IsA") or candidate.IsA("vtkMRMLLinearTransformNode")):
            transform = candidate
    if transform is None:
        node_name = ""
        if hasattr(node, "GetName"):
            try:
                node_name = str(node.GetName() or "").strip()
            except Exception:
                node_name = ""
        transform_name = f"{node_name or (hemi + '.inflated')} display offset"
        transform = scene.AddNewNodeByClass("vtkMRMLLinearTransformNode", transform_name)
        if transform is None:
            return None
        if hasattr(transform, "SetHideFromEditors"):
            try:
                transform.SetHideFromEditors(True)
            except Exception:
                pass
        if hasattr(transform, "SetAttribute"):
            transform.SetAttribute(INFLATED_HEMISPHERE_ATTR, hemi)
        own_nodes(node, [transform])
        node.SetAttribute(INFLATED_TRANSFORM_ATTR, transform.GetID())
    _set_linear_transform_x_offset(transform, dx)
    if hasattr(node, "SetAndObserveTransformNodeID"):
        node.SetAndObserveTransformNodeID(transform.GetID())
    return transform


def set_inflated_separation_enabled(
    slicer,
    enabled: bool,
    *,
    offset_mm: float = INFLATED_SEPARATION_OFFSET_MM,
) -> int:
    """Toggle L/R inflated hemisphere separation ON/OFF for all loaded inflated surfaces and future loads."""

    global _INFLATED_SEPARATION_ENABLED
    _INFLATED_SEPARATION_ENABLED = bool(enabled)
    updated = 0
    for node in iter_inflated_model_nodes(slicer):
        hemi = inflated_hemisphere_of_node(node)
        if hemi in HEMISPHERES:
            apply_inflated_separation(slicer, node, hemi, enabled=_INFLATED_SEPARATION_ENABLED, offset_mm=offset_mm)
            updated += 1
    if slicer is not None and updated:
        _observe_model_removal(slicer)
    return updated


# --------------------------------------------------------------------------------------- one visible overlay
def _file_key(path: Any) -> Optional[str]:
    """One spelling per file: the manifest may name the white surface through a symlink."""

    if not path:
        return None
    try:
        return str(Path(path).resolve())
    except (OSError, RuntimeError, ValueError):
        return str(path)


def hide_same_surface_siblings(slicer, shown_nodes) -> list:
    """Item 7: hide the other overlays of the same carrier and the plain white model of the same run; return them.

    Only a node that carries an overlay (attribute CarrierPath) hides others. Hiding is not unloading.
    """

    from shared.result_scene import result_nodes

    shown_nodes = [node for node in shown_nodes if node is not None]
    keys = {(node.GetAttribute(PREFIX + "Run"), node.GetAttribute(PREFIX + "CarrierPath")) for node in shown_nodes}
    keys = {(run, _file_key(carrier)) for run, carrier in keys if run and carrier}
    if not keys:
        return []
    shown = {node.GetID() for node in shown_nodes}
    hidden = []
    for node in result_nodes(slicer):
        if node.GetID() in shown or not node.IsA("vtkMRMLModelNode"):
            continue
        run = node.GetAttribute(PREFIX + "Run")
        if not any(run == key_run for key_run, _carrier in keys):
            continue
        files = {_file_key(node.GetAttribute(PREFIX + "CarrierPath")), _file_key(node.GetAttribute(PREFIX + "SourcePath"))}
        same = any(run == key_run and carrier in files for key_run, carrier in keys)
        if same and node.GetDisplayVisibility():
            node.SetDisplayVisibility(0)
            hidden.append(node)
    return hidden


# ------------------------------------------------------------------------------- colour table lifetime (item 6)
def remove_owned_color_node(slicer, model_node) -> bool:
    """Remove the annotation colour table that belongs to ``model_node`` (it is made for one row only)."""

    identity = model_node.GetAttribute(PREFIX + "ColorNodeID") if model_node is not None else None
    if not identity:
        return False
    color = slicer.mrmlScene.GetNodeByID(identity)
    if color is None or not hasattr(color, "GetAttribute"):
        return False
    if color.GetAttribute(PREFIX + "OwnerNodeID") != model_node.GetID():
        return False
    slicer.mrmlScene.RemoveNode(color)
    return True


OWNED_NODES_ATTRIBUTE = PREFIX + "OwnedNodeIDs"  # on a row's model: its display, legend and colour node ids


def own_nodes(model_node, nodes) -> None:
    """Review M3: record ``nodes`` (display nodes, legend, colour table) as owned by the row's model. Each owned node
    gets ``OwnerNodeID`` = the model id; the model lists their ids. Ids are kept so that the removal observer does not
    need the model's references, which may no longer resolve once it has left the scene."""

    owner = model_node.GetID()
    identities = (model_node.GetAttribute(OWNED_NODES_ATTRIBUTE) or "").split()
    for node in nodes:
        if node is None or not hasattr(node, "SetAttribute"):
            continue
        identity = node.GetID()
        if not isinstance(identity, str) or not identity:
            continue
        node.SetAttribute(PREFIX + "OwnerNodeID", owner)
        if identity not in identities:
            identities.append(identity)
    model_node.SetAttribute(OWNED_NODES_ATTRIBUTE, " ".join(identities))


def _display_nodes(model_node) -> list:
    count = getattr(model_node, "GetNumberOfDisplayNodes", None)
    if count is None:
        return []
    return [model_node.GetNthDisplayNode(index) for index in range(int(count() or 0))]


def remove_owned_nodes(slicer, model_node) -> int:
    """Remove every node that ``model_node`` owns (only a node whose OwnerNodeID is this model); safe to repeat."""

    if model_node is None:
        return 0
    owner = model_node.GetID()
    identities = (model_node.GetAttribute(OWNED_NODES_ATTRIBUTE) or "").split()
    color = model_node.GetAttribute(PREFIX + "ColorNodeID")
    removed = 0
    for identity in identities + ([color] if color and color not in identities else []):
        node = slicer.mrmlScene.GetNodeByID(identity)
        if node is None or not hasattr(node, "GetAttribute") or node.GetAttribute(PREFIX + "OwnerNodeID") != owner:
            continue
        slicer.mrmlScene.RemoveNode(node)
        removed += 1
    return removed


_REMOVAL_OBSERVER: dict = {}


def _observe_model_removal(slicer) -> None:
    """One scene observer: when a row's model leaves the scene (Unload, Clear, Replace), the display nodes, legend
    and colour table it owns leave too. Nothing while the scene closes; a model without owned nodes (a plain result
    model) removes nothing."""

    scene = slicer.mrmlScene
    if _REMOVAL_OBSERVER.get("scene") is scene or not hasattr(scene, "AddObserver"):
        return
    import vtk

    @vtk.calldata_type(vtk.VTK_OBJECT)
    def on_removed(_caller, _event, node):
        try:
            if node is None or scene.IsClosing():
                return
            if node.GetAttribute(OWNED_NODES_ATTRIBUTE) or node.GetAttribute(PREFIX + "ColorNodeID"):
                remove_owned_nodes(slicer, node)
        except Exception:  # an odd node must not break a scene operation
            pass

    _REMOVAL_OBSERVER.update(scene=scene, callback=on_removed,
                             tag=scene.AddObserver(scene.NodeRemovedEvent, on_removed))


# ---------------------------------------------------------------------------------------------- node builder
def _import_nibabel():
    try:
        import nibabel
        import nibabel.freesurfer  # noqa: F401
    except ImportError as exc:
        raise MorphometryError(NIBABEL_MISSING) from exc
    return nibabel


def _first_existing(slicer, identities) -> Optional[str]:
    for identity in identities:
        if slicer.mrmlScene.GetNodeByID(identity) is not None:
            return identity
    return None


def _postprocess_task(manifest: dict) -> dict:
    return next((task for task in manifest.get("tasks") or [] if isinstance(task, dict)
                 and str(task.get("name") or "").split("@", 1)[0] == PRODUCT), {})


def _load_carrier(slicer, carrier: Path):
    success, node = slicer.util.loadNodeFromFile(
        carrier.as_posix(), "ModelFile",
        properties={"coordinateSystem": slicer.vtkMRMLStorageNode.CoordinateSystemRAS}, returnNode=True)
    if not (success and node):
        raise MorphometryError(_bilingual(f"Slicer could not read {carrier.name}.", f"Slicer 无法读取 {carrier.name}。"))
    return node


def _point_array(values, name: str, *, integer: bool):
    import numpy as np
    from vtk.util.numpy_support import numpy_to_vtk

    data = np.ascontiguousarray(values, dtype=np.int32 if integer else np.float32)
    array = numpy_to_vtk(data, deep=True)
    array.SetName(name)
    return array


def _add_legend(slicer, node, title: str):
    try:
        legend = slicer.modules.colors.logic().AddDefaultColorLegendDisplayNode(node)
        if legend is not None:
            legend.SetTitleText(title)
        return legend
    except Exception:  # Item 6: the legend is optional; the click must not fail without it
        return None


def _build_model_from_points(slicer, points, faces, name: str):
    import numpy as np
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    poly = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float32), deep=True))
    poly.SetPoints(vtk_points)
    id_dtype = np.int64 if vtk.vtkIdTypeArray().GetDataTypeSize() == 8 else np.int32
    cells = vtk.vtkCellArray()
    faces = np.asarray(faces)
    offsets = np.arange(0, 3 * len(faces) + 1, 3, dtype=id_dtype)
    cells.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True),
                  numpy_to_vtkIdTypeArray(np.ascontiguousarray(faces.ravel(), dtype=id_dtype), deep=True))
    poly.SetPolys(cells)
    node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
    node.SetAndObservePolyData(poly)
    node.CreateDefaultDisplayNodes()
    return node


def load_morphometry_row(
    slicer,
    candidate: dict,
    manifest: dict,
    *,
    surfaces: Optional[tuple[str, ...] | list[str]] = None,
) -> bool:
    """Build node(s) for one clicked post-processing row; on any failure nothing stays in the scene.

    For measures and annotations, selects the currently open cortical surface(s) of that hemisphere (``white``,
    ``pial``, and/or ``inflated``), defaulting to ``('white',)`` when none is open. Explicit ``surfaces`` overrides
    automatic detection. Raises MorphometryError (bilingual) for a reason the user can act on. The caller registers
    the new model node(s) (Run, SourcePath ...) like every other result node.
    """

    path = Path(candidate["path"])
    row = morphometry_row(candidate.get("product"), path)
    if row is None:
        raise MorphometryError(_bilingual(f"{path.name} is not a surface post-processing result.",
                                          f"{path.name} 不是表面后处理结果。"))
    kind, hemi, name = row
    if kind == "stats":
        raise MorphometryError(_bilingual(f"{path.name} is a statistics table; open it in the statistics viewer.",
                                          f"{path.name} 是统计表；请在统计表查看器中打开。"))
    nibabel = _import_nibabel()
    from shared.fiber_display import detach_result_storage

    created = []
    try:
        if kind == "inflated":
            points, faces = read_inflated(nibabel, path)
            node = _build_model_from_points(slicer, points, faces, path.name)
            created.append(node)
            display = node.GetDisplayNode()
            if display is not None:
                display.SetColor(0.88, 0.88, 0.82)  # the plain colour of the white surfaces (result_scene)
            node.SetAttribute(PREFIX + "SourceCoordinateSystem", "RAS")
            own_nodes(node, _display_nodes(node))  # review M3: its display node leaves with it
            transform = apply_inflated_separation(slicer, node, hemi)
            if transform is not None:
                created.append(transform)
            _observe_model_removal(slicer)
            return True
        if surfaces is not None:
            target_surfaces = tuple(s for s in surfaces if s in SURFACES) or ("white",)
        else:
            target_surfaces = desired_carrier_surfaces(slicer, manifest, hemi, path)
        task = _postprocess_task(manifest)
        carriers = [(surf, check_carrier(manifest, task, hemi, path, surface=surf)) for surf in target_surfaces]
        if kind == "measure":
            values = nibabel.freesurfer.read_morph_data(str(path))
            colors = None
        else:
            values, colors = read_annotation(nibabel, path)
        import vtk

        array_name = name if kind == "measure" else name.replace(".", "_")
        for index, (surf, carrier) in enumerate(carriers):
            node_name = path.name if (len(carriers) == 1 and surf == "white") else f"{path.name} ({surf})"
            if surf == "inflated":
                points, faces = read_inflated(nibabel, carrier)
                node = _build_model_from_points(slicer, points, faces, node_name)
                created.append(node)
                own_nodes(node, _display_nodes(node))
            else:
                node = _load_carrier(slicer, carrier)
                created.append(node)
                own_nodes(node, _display_nodes(node))  # review M3: the reader's display node belongs to this row
                detach_result_storage(slicer, node)  # a scene Save must never rewrite the published surface
                node.SetName(node_name)
            check_count(len(values), node.GetPolyData().GetNumberOfPoints(), path.name, carrier.name)
            node.SetAttribute(PREFIX + "CarrierPath", str(carrier.resolve()))
            node.SetAttribute(PREFIX + "CarrierSurface", surf)
            node.SetAttribute(PREFIX + "SourceCoordinateSystem", "RAS")
            node.SetAttribute(PREFIX + "MorphometryKind", kind)
            if surf == "inflated":
                transform = apply_inflated_separation(slicer, node, hemi)
                if transform is not None:
                    created.append(transform)
            node.GetPolyData().GetPointData().AddArray(_point_array(values, array_name, integer=kind == "annotation"))
            node.GetPolyData().GetPointData().Modified()
            if node.GetDisplayNode() is None:
                node.CreateDefaultDisplayNodes()
            display = node.GetDisplayNode()
            own_nodes(node, _display_nodes(node))
            display.SetActiveScalar(array_name, vtk.vtkAssignAttribute.POINT_DATA)
            if kind == "measure":
                table = _first_existing(slicer, SEQUENTIAL_TABLES if name in {"thickness", "area"} else DIVERGING_TABLES)
                if table is not None:
                    display.SetAndObserveColorNodeID(table)
                low, high = measure_range(name, values)
            else:
                color_name = f"{path.name} colors" if (len(carriers) == 1 and surf == "white") else f"{path.name} ({surf}) colors"
                color = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLColorTableNode", color_name)
                created.append(color)
                color.SetTypeToUser()
                color.SetNumberOfColors(len(colors))
                for c_idx, (label, red, green, blue) in enumerate(colors):
                    color.SetColor(c_idx, label, red, green, blue, 1.0)
                low, high = annotation_scalar_range(len(colors))
                lookup = color.GetLookupTable()
                if lookup is not None:
                    lookup.SetTableRange(low, high)
                color.SetAttribute(PREFIX + "OwnerNodeID", node.GetID())
                node.SetAttribute(PREFIX + "ColorNodeID", color.GetID())
                own_nodes(node, [color])
                display.SetAndObserveColorNodeID(color.GetID())
            display.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseManualScalarRange)
            display.SetScalarRange(low, high)
            display.SetScalarVisibility(True)
            if kind == "measure" and index == 0:
                legend = _add_legend(slicer, node, LEGEND_TITLES[name])
                own_nodes(node, [legend] + _display_nodes(node))
        _observe_model_removal(slicer)
        return True
    except Exception:
        # Review M3: the rollback removes the model, its display node(s), its legend and its colour table.
        for item in reversed(created):
            try:
                extras = [extra for extra in _display_nodes(item) if extra is not None]
            except Exception:
                extras = []
            for extra in extras:
                try:
                    slicer.mrmlScene.RemoveNode(extra)
                except Exception:
                    pass
            try:
                remove_owned_nodes(slicer, item)
            except Exception:
                pass
            try:
                slicer.mrmlScene.RemoveNode(item)
            except Exception:
                pass
        raise


# ------------------------------------------------------------------------------------------- .stats tables
_MEASURE_LINE = re.compile(r"^#\s*Measure\s+(.+)$")
_COLHEADERS_LINE = re.compile(r"^#\s*ColHeaders\s+(.+)$")
_TABLECOL_LINE = re.compile(r"^#\s*TableCol\s+(\d+)\s+(ColHeader|Units|FieldName)\s+(.+)$")
_META_LINE = re.compile(r"^#\s*(subjectname|hemi|annotation|NRows|NTableCols|CreationTime)\s+(.+)$")
_INT_PATTERN = re.compile(r"^[+-]?\d+$")


def parse_freesurfer_stats(path: Any) -> dict[str, Any]:
    """Parse a FreeSurfer / DDSurfer ``.stats`` table into measures, columns, rows, units, metadata and raw text."""

    raw_text = Path(path).read_text(encoding="utf-8", errors="replace")
    measures: list[tuple[str, str, str, str]] = []
    columns: list[str] = []
    table_col_headers: dict[int, str] = {}
    table_col_units: dict[int, str] = {}
    metadata: dict[str, str] = {}
    data_lines: list[str] = []

    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            m_measure = _MEASURE_LINE.match(stripped)
            if m_measure:
                parts = [part.strip() for part in m_measure.group(1).split(",")]
                if len(parts) >= 5:
                    measures.append((parts[1] or parts[0], parts[2], parts[3], ", ".join(parts[4:])))
                elif len(parts) == 4:
                    measures.append((parts[0], parts[1], parts[2], parts[3]))
                elif len(parts) == 3:
                    measures.append((parts[0], parts[0], parts[1], parts[2]))
                elif len(parts) == 2:
                    measures.append((parts[0], parts[0], parts[1], ""))
                continue
            m_cols = _COLHEADERS_LINE.match(stripped)
            if m_cols:
                columns = m_cols.group(1).split()
                continue
            m_tcol = _TABLECOL_LINE.match(stripped)
            if m_tcol:
                col_index = int(m_tcol.group(1))
                field = m_tcol.group(2)
                value = m_tcol.group(3).strip()
                if field == "ColHeader":
                    table_col_headers[col_index] = value
                elif field == "Units":
                    table_col_units[col_index] = value
                continue
            m_meta = _META_LINE.match(stripped)
            if m_meta:
                metadata[m_meta.group(1)] = m_meta.group(2).strip()
            continue
        data_lines.append(stripped)

    if not columns and table_col_headers:
        columns = [table_col_headers[idx] for idx in sorted(table_col_headers)]

    units: dict[str, str] = {}
    for idx, col_name in table_col_headers.items():
        if idx in table_col_units:
            units[col_name] = table_col_units[idx]
    for pos, col_name in enumerate(columns, 1):
        if col_name not in units and pos in table_col_units:
            units[col_name] = table_col_units[pos]

    rows: list[tuple[str, ...]] = []
    if columns:
        col_count = len(columns)
        for line in data_lines:
            parts = line.split()
            if len(parts) == col_count:
                rows.append(tuple(parts))
            elif len(parts) < col_count:
                rows.append(tuple(parts + [""] * (col_count - len(parts))))
            else:
                rows.append(tuple(parts[: col_count - 1] + [" ".join(parts[col_count - 1 :])]))

    return {
        "measures": measures,
        "columns": columns,
        "rows": rows,
        "units": units,
        "metadata": metadata,
        "raw_text": raw_text,
    }


def stats_to_tsv(parsed: dict[str, Any]) -> str:
    """Format parsed ``.stats`` data as TSV (or raw text when no table columns were parsed)."""

    columns = parsed.get("columns") or []
    rows = parsed.get("rows") or []
    if not columns:
        return str(parsed.get("raw_text") or "")
    lines = ["\t".join(str(col) for col in columns)]
    lines.extend("\t".join(str(cell) for cell in row) for row in rows)
    return "\n".join(lines)


def _parse_numeric_cell(text: str) -> Optional[int | float]:
    stripped = str(text or "").strip()
    if not stripped:
        return None
    if _INT_PATTERN.match(stripped):
        try:
            return int(stripped)
        except ValueError:
            return None
    try:
        value = float(stripped)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def load_stats_table_node(slicer, path: Any) -> Any:
    """Create or update a ``vtkMRMLTableNode`` from a FreeSurfer/DDSurfer ``.stats`` file; None if not tabular."""

    stats_path = Path(path).expanduser()
    parsed = parse_freesurfer_stats(stats_path)
    columns = parsed["columns"]
    rows = parsed["rows"]
    if not columns:
        return None

    import vtk

    source_key = _file_key(stats_path)
    table_node = None
    try:
        existing = slicer.util.getNodesByClass("vtkMRMLTableNode")
    except Exception:
        existing = []
    for candidate in existing or []:
        if candidate is not None and _file_key(candidate.GetAttribute(PREFIX + "StatsSourcePath")) == source_key:
            table_node = candidate
            break
    if table_node is None:
        table_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode", stats_path.name)

    vtk_table = vtk.vtkTable()
    row_count = len(rows)
    for col_idx, col_name in enumerate(columns):
        col_values = [row[col_idx] if col_idx < len(row) else "" for row in rows]
        non_empty = [val for val in col_values if str(val).strip()]
        parsed_nums = [_parse_numeric_cell(val) for val in non_empty]
        if non_empty and all(isinstance(val, int) for val in parsed_nums):
            int_vals = [val for val in parsed_nums if isinstance(val, int)]
            if all(-(2**31) <= val < 2**31 for val in int_vals):
                array = vtk.vtkIntArray()
            else:
                array = vtk.vtkLongLongArray()
            array.SetName(col_name)
            array.SetNumberOfValues(row_count)
            for row_idx, raw in enumerate(col_values):
                val = _parse_numeric_cell(raw)
                array.SetValue(row_idx, int(val) if isinstance(val, int) else 0)
        elif non_empty and all(val is not None for val in parsed_nums):
            array = vtk.vtkDoubleArray()
            array.SetName(col_name)
            array.SetNumberOfValues(row_count)
            for row_idx, raw in enumerate(col_values):
                val = _parse_numeric_cell(raw)
                array.SetValue(row_idx, float(val) if val is not None else float("nan"))
        else:
            array = vtk.vtkStringArray()
            array.SetName(col_name)
            array.SetNumberOfValues(row_count)
            for row_idx, raw in enumerate(col_values):
                array.SetValue(row_idx, str(raw))
        vtk_table.AddColumn(array)

    if hasattr(table_node, "SetAndObserveTable"):
        table_node.SetAndObserveTable(vtk_table)
    if hasattr(table_node, "SetUseColumnNameAsColumnHeader"):
        try:
            table_node.SetUseColumnNameAsColumnHeader(True)
        except Exception:
            pass
    if hasattr(table_node, "SetLocked"):
        try:
            table_node.SetLocked(True)
        except Exception:
            pass
    units = parsed.get("units") or {}
    if hasattr(table_node, "SetColumnUnitLabel"):
        for col_name in columns:
            unit = str(units.get(col_name) or "").strip()
            if unit and unit.lower() not in {"na", "unitless"}:
                try:
                    table_node.SetColumnUnitLabel(col_name, unit)
                except Exception:
                    pass
    table_node.SetAttribute(PREFIX + "StatsSourcePath", str(stats_path.resolve()))
    table_node.SetAttribute(PREFIX + "MorphometryKind", "stats")
    return table_node


def _activate_table_node_in_slicer(slicer, table_node: Any) -> None:
    """Best-effort display of ``table_node`` in Slicer's Table View."""

    if table_node is None:
        return
    try:
        app = getattr(slicer, "app", None)
        app_logic = app.applicationLogic() if (app is not None and hasattr(app, "applicationLogic")) else None
        if app_logic is not None:
            selection = app_logic.GetSelectionNode()
            if selection is not None and hasattr(selection, "SetActiveTableID"):
                selection.SetActiveTableID(table_node.GetID())
            if hasattr(app_logic, "PropagateTableSelection"):
                app_logic.PropagateTableSelection()
        layout_mgr = app.layoutManager() if (app is not None and hasattr(app, "layoutManager")) else None
        if layout_mgr is not None and hasattr(layout_mgr, "setLayout"):
            table_views = getattr(layout_mgr, "tableViewCount", 0)
            if callable(table_views):
                table_views = table_views()
            if not table_views:
                layout_node_cls = getattr(slicer, "vtkMRMLLayoutNode", None)
                table_layout = getattr(layout_node_cls, "SlicerLayoutFourUpTableView", None)
                if table_layout is not None:
                    layout_mgr.setLayout(table_layout)
    except Exception:
        pass


def _connect_qt_signal(widget: Any, signature: str, attr_name: str, callback) -> None:
    connect_fn = getattr(widget, "connect", None)
    if callable(connect_fn):
        try:
            connect_fn(signature, callback)
            return
        except Exception:
            pass
    signal = getattr(widget, attr_name, None)
    if signal is not None and hasattr(signal, "connect"):
        signal.connect(callback)


def _format_stats_summary(parsed: dict[str, Any], *, zh: bool) -> str:
    measures = parsed.get("measures") or []
    if measures:
        parts = []
        for name, description, value, unit in measures:
            label = description or name
            unit_suffix = f" {unit}" if unit and unit.lower() not in {"na", "unitless"} else ""
            parts.append(f"{label}: {value}{unit_suffix}")
        return "\n".join(parts)
    metadata = parsed.get("metadata") or {}
    meta_parts = []
    labels = (
        ("subjectname", "受试者" if zh else "Subject"),
        ("hemi", "半球" if zh else "Hemisphere"),
        ("annotation", "图谱" if zh else "Atlas"),
        ("NRows", "区域数" if zh else "Regions"),
    )
    for key, label in labels:
        if metadata.get(key):
            meta_parts.append(f"{label}: {metadata[key]}")
    return "  ·  ".join(meta_parts)


def show_stats_dialog(owner: Any, path: Any, *, zh: bool = False) -> bool:
    """Display a FreeSurfer/DDSurfer ``.stats`` file inside Slicer in a sortable, read-only Qt dialog."""

    import qt
    import slicer

    stats_path = Path(path).expanduser()
    parent = getattr(owner, "parent", None)
    if parent is None:
        main_win = getattr(getattr(slicer, "util", None), "mainWindow", None)
        if callable(main_win):
            try:
                parent = main_win()
            except Exception:
                parent = None
    if not stats_path.is_file():
        slicer.util.infoDisplay("统计表文件不存在。" if zh else "The statistics file is missing.", parent=parent)
        return False
    try:
        parsed = parse_freesurfer_stats(stats_path)
    except OSError as exc:
        slicer.util.errorDisplay(
            f"无法读取统计表文件：\n{exc}" if zh else f"Failed to read statistics file:\n{exc}",
            parent=parent,
        )
        return False

    dialog = qt.QDialog(parent)
    dialog.setWindowTitle(stats_path.name)
    if hasattr(dialog, "resize"):
        dialog.resize(880, 580)
    if hasattr(dialog, "setMinimumSize"):
        try:
            dialog.setMinimumSize(620, 400)
        except Exception:
            pass

    layout = qt.QVBoxLayout(dialog)

    path_label = qt.QLabel(str(stats_path.resolve()))
    if hasattr(path_label, "setWordWrap"):
        path_label.setWordWrap(True)
    layout.addWidget(path_label)

    summary_text = _format_stats_summary(parsed, zh=zh)
    if summary_text:
        if parsed["measures"] and hasattr(qt, "QGroupBox"):
            summary_box = qt.QGroupBox("汇总指标" if zh else "Summary Measures")
            summary_layout = qt.QVBoxLayout(summary_box)
            summary_label = qt.QLabel(summary_text)
            if hasattr(summary_label, "setWordWrap"):
                summary_label.setWordWrap(True)
            summary_layout.addWidget(summary_label)
            layout.addWidget(summary_box)
        else:
            summary_label = qt.QLabel(summary_text)
            if hasattr(summary_label, "setWordWrap"):
                summary_label.setWordWrap(True)
            layout.addWidget(summary_label)

    columns = parsed["columns"]
    rows = parsed["rows"]
    units = parsed.get("units") or {}
    has_table = bool(columns)

    if has_table:
        filter_bar = qt.QHBoxLayout()
        filter_input = qt.QLineEdit()
        if hasattr(filter_input, "setPlaceholderText"):
            filter_input.setPlaceholderText("筛选区域或数值..." if zh else "Filter regions or values...")
        if hasattr(filter_input, "setClearButtonEnabled"):
            filter_input.setClearButtonEnabled(True)
        count_label = qt.QLabel(f"{len(rows)} 行" if zh else f"{len(rows)} rows")
        filter_bar.addWidget(filter_input, 1)
        filter_bar.addWidget(count_label)
        layout.addLayout(filter_bar)

        table = qt.QTableWidget(len(rows), len(columns), dialog)
        headers = []
        for col in columns:
            unit = str(units.get(col) or "").strip()
            headers.append(f"{col} ({unit})" if (unit and unit.lower() not in {"na", "unitless"}) else col)
        if hasattr(table, "setHorizontalHeaderLabels"):
            table.setHorizontalHeaderLabels(headers)
        if hasattr(table, "setEditTriggers") and hasattr(qt, "QAbstractItemView"):
            no_edit = getattr(qt.QAbstractItemView, "NoEditTriggers", None)
            if no_edit is not None:
                table.setEditTriggers(no_edit)
        if hasattr(table, "setAlternatingRowColors"):
            table.setAlternatingRowColors(True)
        if hasattr(table, "setSelectionBehavior") and hasattr(qt, "QAbstractItemView"):
            select_rows = getattr(qt.QAbstractItemView, "SelectRows", None)
            if select_rows is not None:
                table.setSelectionBehavior(select_rows)
        if hasattr(table, "setSortingEnabled"):
            table.setSortingEnabled(False)

        edit_role = getattr(getattr(qt, "Qt", None), "EditRole", 2)
        align_right = getattr(getattr(qt, "Qt", None), "AlignRight", 0x0002)
        align_vcenter = getattr(getattr(qt, "Qt", None), "AlignVCenter", 0x0080)
        for row_idx, row_values in enumerate(rows):
            for col_idx, cell_text in enumerate(row_values):
                cell_item = qt.QTableWidgetItem(str(cell_text))
                numeric = _parse_numeric_cell(cell_text)
                if numeric is not None:
                    try:
                        cell_item.setData(edit_role, numeric)
                    except Exception:
                        pass
                    try:
                        cell_item.setTextAlignment(align_right | align_vcenter)
                    except Exception:
                        pass
                table.setItem(row_idx, col_idx, cell_item)

        if hasattr(table, "resizeColumnsToContents"):
            try:
                table.resizeColumnsToContents()
            except Exception:
                pass
        if hasattr(table, "setSortingEnabled"):
            table.setSortingEnabled(True)

        def _apply_filter(query_text: str) -> None:
            needle = str(query_text or "").strip().lower()
            shown = 0
            row_total = table.rowCount() if callable(getattr(table, "rowCount", None)) else getattr(table, "rowCount", len(rows))
            col_total = table.columnCount() if callable(getattr(table, "columnCount", None)) else getattr(table, "columnCount", len(columns))
            for r_idx in range(int(row_total)):
                if not needle:
                    matched = True
                else:
                    matched = False
                    for c_idx in range(int(col_total)):
                        cell = table.item(r_idx, c_idx)
                        text_attr = getattr(cell, "text", "") if cell is not None else ""
                        cell_str = str(text_attr() if callable(text_attr) else text_attr).lower()
                        if needle in cell_str:
                            matched = True
                            break
                if hasattr(table, "setRowHidden"):
                    table.setRowHidden(r_idx, not matched)
                if matched:
                    shown += 1
            if needle:
                count_label.setText(f"{shown} / {row_total} 行" if zh else f"{shown} / {row_total} rows")
            else:
                count_label.setText(f"{row_total} 行" if zh else f"{row_total} rows")

        _connect_qt_signal(filter_input, "textChanged(QString)", "textChanged", _apply_filter)
        layout.addWidget(table, 1)
    else:
        raw_view = qt.QPlainTextEdit(dialog)
        if hasattr(raw_view, "setReadOnly"):
            raw_view.setReadOnly(True)
        if hasattr(raw_view, "setPlainText"):
            raw_view.setPlainText(parsed["raw_text"])
        layout.addWidget(raw_view, 1)

    buttons = qt.QHBoxLayout()

    if has_table:
        slicer_table_btn = qt.QPushButton("在 Slicer 表格视图打开" if zh else "Open in Slicer Table View")

        def _open_in_slicer_table() -> None:
            try:
                node = load_stats_table_node(slicer, stats_path)
                _activate_table_node_in_slicer(slicer, node)
            except Exception as exc:
                slicer.util.errorDisplay(str(exc), parent=dialog)

        _connect_qt_signal(slicer_table_btn, "clicked()", "clicked", _open_in_slicer_table)
        buttons.addWidget(slicer_table_btn)

    copy_btn = qt.QPushButton(
        ("复制表格" if has_table else "复制文本") if zh else ("Copy Table (TSV)" if has_table else "Copy Text")
    )

    def _copy_tsv() -> None:
        try:
            qt.QApplication.clipboard().setText(stats_to_tsv(parsed))
        except Exception:
            pass

    _connect_qt_signal(copy_btn, "clicked()", "clicked", _copy_tsv)
    buttons.addWidget(copy_btn)

    external_btn = qt.QPushButton("外部打开" if zh else "Open Externally")

    def _open_externally() -> None:
        open_fn = getattr(owner, "_open_local_path", None)
        if callable(open_fn):
            open_fn(stats_path, missing_message="统计表文件不存在。" if zh else "The statistics file is missing.")
            return
        try:
            qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(str(stats_path.resolve())))
        except Exception:
            pass

    _connect_qt_signal(external_btn, "clicked()", "clicked", _open_externally)
    buttons.addWidget(external_btn)

    if hasattr(buttons, "addStretch"):
        buttons.addStretch(1)

    close_btn = qt.QPushButton("关闭" if zh else "Close")
    _connect_qt_signal(close_btn, "clicked()", "clicked", getattr(dialog, "close", getattr(dialog, "accept", lambda: None)))
    buttons.addWidget(close_btn)
    layout.addLayout(buttons)

    key = _file_key(stats_path) or str(stats_path)
    try:
        dialogs = getattr(owner, "_statsDialogs", None)
        if not isinstance(dialogs, dict):
            dialogs = {}
            setattr(owner, "_statsDialogs", dialogs)
        previous = dialogs.get(key)
        if previous is not None and previous is not dialog and hasattr(previous, "close"):
            try:
                previous.close()
            except Exception:
                pass
        dialogs[key] = dialog
    except Exception:
        pass

    if hasattr(dialog, "show"):
        dialog.show()
    if hasattr(dialog, "raise_"):
        try:
            dialog.raise_()
        except Exception:
            pass
    if hasattr(dialog, "activateWindow"):
        try:
            dialog.activateWindow()
        except Exception:
            pass
    return True
