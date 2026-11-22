"""Native display of the surface post-processing results (no SlicerFreeSurfer extension).

The task ``surface.ddsurfer_postprocess`` publishes FreeSurfer-format files in
``outputs/01_anatomical_priors/surface_morphometry/``: ``{lh,rh}.thickness/.curv/.sulc/.area`` (one value per white-surface
vertex), ``{lh,rh}.aparc.annot`` / ``.aparc.a2009s.annot`` (one label per vertex), ``{lh,rh}.inflated`` (a surface) and
``{lh,rh}.<atlas>.stats`` (text tables). Slicer core has no reader for them; this module reads them with
``nibabel.freesurfer``.

- A measure or an annotation is shown on the WHITE surface of the same hemisphere of the same run: the published
  ``cortical_surface/{hemi}.white.obj`` that the run's surface task records. Post-processing read exactly that file, so
  its vertex order is the order of the values. The LPS ``.stl`` has the same point count but another vertex order, so
  it is never a carrier. Each row gets its own model node (white geometry + this row's scalar).
- Before the scalar is attached: the value count must equal the point count, and the carrier's sha256 must equal the
  ``native_sha256`` that post-processing recorded in ``structural/manifest.json`` (when that record is readable).
- ``{hemi}.inflated`` becomes its own model, moved from the FreeSurfer surface space to scanner space with the
  volume geometry of the file footer (the transform SlicerFreeSurfer's reader applies).

The pure functions (rows, carrier, checks, ranges, label mapping, footer transform, sibling rule) run without the
application; :func:`load_morphometry_row` is the one thin function that builds nodes.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Optional

PREFIX = "SlicerDDSurfer."
PRODUCT = "surface.ddsurfer_postprocess"
SURFACE_PRODUCT = "surface.ddsurfer"
MORPHOMETRY_ROW_KIND = "surface_morphometry"  # ROLE_KIND of a Results row of this product (plain eye)
HEMISPHERES = ("lh", "rh")
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
        return f"在白质表面上显示{chinese}" if zh else f"Show {english} on the white surface"
    if kind == "annotation":
        english, chinese = _ATLAS_TEXT.get(name, (name, name))
        return f"在白质表面上显示 {chinese} 分区" if zh else f"Show the {english} parcellation on the white surface"
    if kind == "inflated":
        return "显示膨胀表面" if zh else "Show the inflated surface"
    return "在 Slicer 外打开统计表" if zh else "Open the statistics table outside Slicer"


# ------------------------------------------------------------------------------------------ carrier checks
def _completed(task: dict) -> bool:
    status = task.get("status")
    return status in {"completed", "reused"} or (status == "skipped" and task.get("reason") == "completed outputs found")


def white_carrier(manifest: dict, hemi: str) -> Optional[Path]:
    """The published ``{hemi}.white.obj`` that the run's surface task records, or None (never the ``.stl``)."""

    for task in manifest.get("tasks") or []:
        if not isinstance(task, dict) or str(task.get("name") or "").split("@", 1)[0] != SURFACE_PRODUCT:
            continue
        if not _completed(task):
            continue
        for value in (task.get("outputs") or {}).values():
            if not isinstance(value, str) or not value:
                continue
            path = Path(value).expanduser()
            if path.is_absolute() and path.name == f"{hemi}.white.obj" and path.is_file():
                return path
    return None


def recorded_carrier_sha256(task: dict, hemi: str) -> Optional[str]:
    """``native_sha256["{hemi}.white.obj"]`` of the post-processing ``structural/manifest.json``; None when unreadable."""

    outputs = task.get("outputs") or {}
    candidates = [task.get("manifest_path")]
    if outputs.get("structural_root"):
        candidates.append(str(Path(str(outputs["structural_root"])) / "manifest.json"))
    for value in candidates:
        if not value:
            continue
        try:
            record = json.loads(Path(str(value)).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        hashes = record.get("native_sha256") if isinstance(record, dict) else None
        value = hashes.get(f"{hemi}.white.obj") if isinstance(hashes, dict) else None
        if isinstance(value, str) and value:
            return value.lower()
    return None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_carrier(manifest: dict, task: dict, hemi: str) -> Path:
    """The white ``.obj`` to carry a measure of ``hemi``; a bilingual MorphometryError when there is none or it differs."""

    carrier = white_carrier(manifest, hemi)
    if carrier is None:
        raise MorphometryError(_bilingual(
            f"This run has no {hemi}.white.obj surface. Older runs publish only the .stl surface, whose vertex order "
            "differs from the post-processing values. Re-run Surface to show this file.",
            f"此运行没有 {hemi}.white.obj 白质表面。旧的运行只发布 .stl 表面，其顶点顺序与后处理数值不同。"
            "请重新运行“表面重建”后再显示此文件。"))
    expected = recorded_carrier_sha256(task, hemi)
    if expected is not None and file_sha256(carrier) != expected:
        raise MorphometryError(_bilingual(
            f"{carrier.name} is not the white surface that surface post-processing used (its checksum differs from "
            "the post-processing record). Re-run Surface postprocess.",
            f"{carrier.name} 不是表面后处理所用的白质表面（校验值与后处理记录不同）。请重新运行“表面后处理”。"))
    return carrier


def check_count(values: int, points: int, file_name: str, carrier_name: str) -> None:
    if int(values) != int(points):
        raise MorphometryError(_bilingual(
            f"{file_name} has {values} values, but the white surface {carrier_name} has {points} points; "
            "the file does not belong to this surface.",
            f"{file_name} 有 {values} 个数值，但白质表面 {carrier_name} 有 {points} 个顶点；此文件不属于该表面。"))


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


def annotation_colors(raw_ids, ctab, names) -> tuple[Any, list[tuple[str, float, float, float]]]:
    """Map raw annotation values to colour-table rows: row 0 = unknown (neutral grey), row k+1 = ctab entry k.

    The table ids (column 4 of ``ctab``) need not be sequential; a value that is in no entry (FreeSurfer's -1 or 0
    "no label") goes to row 0.
    """

    import numpy as np

    raw = np.asarray(raw_ids).astype(np.int64)
    table_ids = np.asarray(ctab)[:, 4].astype(np.int64) if len(ctab) else np.zeros(0, dtype=np.int64)
    indices = np.zeros(raw.shape, dtype=np.int32)
    order = np.argsort(table_ids, kind="stable")
    if table_ids.size:
        positions = np.searchsorted(table_ids[order], raw)
        positions = np.clip(positions, 0, table_ids.size - 1)
        hit = table_ids[order][positions] == raw
        indices[hit] = order[positions[hit]].astype(np.int32) + 1
    colors = [("unknown", *UNKNOWN_GREY)]
    for index, name in enumerate(names):
        text = name.decode("utf-8", "replace") if isinstance(name, bytes) else str(name)
        red, green, blue = (float(value) / 255 for value in np.asarray(ctab)[index, :3])
        if text.strip().lower() == "unknown":  # the table's own "unknown" entry (aparc: dark purple; a2009s: black)
            red, green, blue = UNKNOWN_GREY
        colors.append((text, red, green, blue))
    return indices, colors


def read_annotation(nibabel_module, path) -> tuple[Any, list[tuple[str, float, float, float]]]:
    raw, ctab, names = nibabel_module.freesurfer.read_annot(str(path), orig_ids=True)
    return annotation_colors(raw, ctab, names)


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


_REMOVAL_OBSERVER: dict = {}


def _observe_model_removal(slicer) -> None:
    """One scene observer: when a row's model leaves the scene (Unload, Clear, Replace), its colour table leaves too."""

    scene = slicer.mrmlScene
    if _REMOVAL_OBSERVER.get("scene") is scene or not hasattr(scene, "AddObserver"):
        return
    import vtk

    @vtk.calldata_type(vtk.VTK_OBJECT)
    def on_removed(_caller, _event, node):
        try:
            if node is not None and node.GetAttribute(PREFIX + "ColorNodeID") and not scene.IsClosing():
                remove_owned_color_node(slicer, node)
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


def _add_legend(slicer, node, title: str) -> None:
    try:
        legend = slicer.modules.colors.logic().AddDefaultColorLegendDisplayNode(node)
        if legend is not None:
            legend.SetTitleText(title)
    except Exception:  # Item 6: the legend is optional; the click must not fail without it
        pass


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


def load_morphometry_row(slicer, candidate: dict, manifest: dict) -> bool:
    """Build the node of one clicked post-processing row; on any failure nothing stays in the scene.

    Raises MorphometryError (bilingual) for a reason the user can act on. The caller registers the new model node
    (Run, SourcePath ...) like every other result node.
    """

    path = Path(candidate["path"])
    row = morphometry_row(candidate.get("product"), path)
    if row is None:
        raise MorphometryError(_bilingual(f"{path.name} is not a surface post-processing result.",
                                          f"{path.name} 不是表面后处理结果。"))
    kind, hemi, name = row
    if kind == "stats":
        raise MorphometryError(_bilingual(f"{path.name} is a text table; open it outside Slicer.",
                                          f"{path.name} 是文本表格；请在 Slicer 外打开。"))
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
            return True
        carrier = check_carrier(manifest, _postprocess_task(manifest), hemi)
        if kind == "measure":
            values = nibabel.freesurfer.read_morph_data(str(path))
            colors = None
        else:
            values, colors = read_annotation(nibabel, path)
        node = _load_carrier(slicer, carrier)
        created.append(node)
        detach_result_storage(slicer, node)  # a scene Save must never rewrite the published white surface
        check_count(len(values), node.GetPolyData().GetNumberOfPoints(), path.name, carrier.name)
        node.SetName(path.name)
        node.SetAttribute(PREFIX + "CarrierPath", str(carrier.resolve()))
        node.SetAttribute(PREFIX + "SourceCoordinateSystem", "RAS")
        node.SetAttribute(PREFIX + "MorphometryKind", kind)
        import vtk

        array_name = name if kind == "measure" else name.replace(".", "_")
        node.GetPolyData().GetPointData().AddArray(_point_array(values, array_name, integer=kind == "annotation"))
        node.GetPolyData().GetPointData().Modified()
        if node.GetDisplayNode() is None:
            node.CreateDefaultDisplayNodes()
        display = node.GetDisplayNode()
        display.SetActiveScalar(array_name, vtk.vtkAssignAttribute.POINT_DATA)
        if kind == "measure":
            table = _first_existing(slicer, SEQUENTIAL_TABLES if name in {"thickness", "area"} else DIVERGING_TABLES)
            if table is not None:
                display.SetAndObserveColorNodeID(table)
            low, high = measure_range(name, values)
        else:
            color = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLColorTableNode", f"{path.name} colors")
            created.append(color)
            color.SetTypeToUser()
            color.SetNumberOfColors(len(colors))
            for index, (label, red, green, blue) in enumerate(colors):
                color.SetColor(index, label, red, green, blue, 1.0)
            low, high = annotation_scalar_range(len(colors))
            lookup = color.GetLookupTable()
            if lookup is not None:
                lookup.SetTableRange(low, high)
            color.SetAttribute(PREFIX + "OwnerNodeID", node.GetID())
            node.SetAttribute(PREFIX + "ColorNodeID", color.GetID())
            display.SetAndObserveColorNodeID(color.GetID())
            _observe_model_removal(slicer)
        display.SetScalarRangeFlag(slicer.vtkMRMLDisplayNode.UseManualScalarRange)
        display.SetScalarRange(low, high)
        display.SetScalarVisibility(True)
        if kind == "measure":
            _add_legend(slicer, node, LEGEND_TITLES[name])
        return True
    except Exception:
        for item in reversed(created):
            try:
                slicer.mrmlScene.RemoveNode(item)
            except Exception:
                pass
        raise
