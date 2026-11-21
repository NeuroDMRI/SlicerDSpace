"""Native fiber-bundle display for streamline results: one classifier, one direction colour, one style.

Every streamline output (whole-brain tractography and bundles) is shown as a ``vtkMRMLModelNode`` of polylines,
coloured per point by its direction (DEC). It needs Slicer core only, so the display is the same with and
without the SlicerDMRI extension.
"""

from __future__ import annotations

from typing import Any, Optional

DEC_ARRAY = "SlicerDSpace_DEC"
KIND_TRACTOGRAPHY = "tractography"
KIND_BUNDLE = "bundle"
STREAMLINE_SUFFIXES = (".trk", ".tck", ".vtk", ".vtp")
DISPLAY_KIND = "fiber_bundle"
LINE_WIDTH = {KIND_TRACTOGRAPHY: 1.0, KIND_BUNDLE: 2.0}
# The solid colour is unused while the direction colours show; a neutral grey keeps the Models swatch honest.
NEUTRAL_COLOR = (0.7, 0.7, 0.7)
_NOT_STREAMLINE_KEY_WORDS = ("seed", "fod", "mask", "intermediate")
_PREFIX = "SlicerDDSurfer."


def streamline_kind(product: Any, key: Any = None, path: Any = None) -> Optional[str]:
    """Return "tractography", "bundle" or None from the manifest product and key, never from the suffix alone.

    Accepts either ``(product, key, path)`` or one result candidate dict (``product``, ``key``, ``path``).
    """

    if isinstance(product, dict):
        candidate = product
        product, key, path = candidate.get("product"), candidate.get("key"), candidate.get("path")
    product = str(product or "")
    key = str(key or "").lower().removesuffix("_path")
    if not str(path or "").lower().endswith(STREAMLINE_SUFFIXES):
        return None
    if product.startswith(("tractography.", "saf.")):
        if any(word in key for word in _NOT_STREAMLINE_KEY_WORDS):
            return None
        return KIND_TRACTOGRAPHY
    if product.startswith("bundles."):
        return KIND_BUNDLE
    return None


def direction_colors(poly_data: Any):
    """Per-point uint8 RGB = round(|unit tangent| * 255): R = left-right, G = anterior-posterior, B = superior-inferior.

    The tangent is a central difference inside a polyline and one-sided at its ends. A zero tangent (repeated
    points) takes the direction of the whole line; a point with no direction, or in no line, stays (0, 0, 0).
    """

    import numpy as np
    from vtk.util.numpy_support import vtk_to_numpy

    count = int(poly_data.GetNumberOfPoints()) if poly_data is not None else 0
    colors = np.zeros((count, 3), dtype=np.uint8)
    lines = poly_data.GetLines() if count else None
    if lines is None or lines.GetNumberOfCells() == 0:
        return colors
    points = vtk_to_numpy(poly_data.GetPoints().GetData()).astype(np.float32, copy=False)
    offsets = vtk_to_numpy(lines.GetOffsetsArray()).astype(np.int64, copy=False)
    connectivity = vtk_to_numpy(lines.GetConnectivityArray()).astype(np.int64, copy=False)
    sizes = np.diff(offsets)
    used = sizes > 0
    if not used.any():
        return colors
    starts, ends = offsets[:-1][used], offsets[1:][used] - 1
    tangent = points[connectivity]  # points in line order
    step = np.diff(tangent, axis=0)
    tangent[:-1] = step  # forward difference
    tangent[-1] = 0
    tangent[ends] = 0  # no forward step past the end of a line
    tangent[1:] += step  # plus the backward difference: a central difference inside a line
    inner_starts = starts[starts > 0]
    tangent[inner_starts] -= step[inner_starts - 1]  # no backward step before the start of a line
    del step
    norm = np.sqrt(np.einsum("ij,ij->i", tangent, tangent))
    flat = np.flatnonzero(norm <= 0)
    if flat.size:
        cell = np.searchsorted(starts, flat, side="right") - 1
        tangent[flat] = points[connectivity[ends[cell]]] - points[connectivity[starts[cell]]]
        norm[flat] = np.sqrt(np.einsum("ij,ij->i", tangent[flat], tangent[flat]))
    np.abs(tangent, out=tangent)
    np.divide(tangent, norm[:, None], out=tangent, where=norm[:, None] > 0)
    tangent[norm <= 0] = 0
    tangent *= 255.0
    np.rint(tangent, out=tangent)
    colors[connectivity] = np.clip(tangent, 0, 255).astype(np.uint8)
    return colors


def add_direction_colors(poly_data: Any) -> Any:
    """Store the DEC array on the poly data (replacing an older copy); return the VTK array."""

    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    array = numpy_to_vtk(direction_colors(poly_data), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    array.SetName(DEC_ARRAY)
    point_data = poly_data.GetPointData()
    point_data.RemoveArray(DEC_ARRAY)
    point_data.AddArray(array)
    point_data.Modified()
    return array


def detach_result_storage(slicer: Any, node: Any) -> int:
    """Remove the storage nodes of ``node`` from the scene; return how many were removed.

    The DEC array makes the node's poly data differ from the result file it was read from. Slicer then lists the node
    as modified in the Save dialog with that file as its target, and Python has no way to mark the node as unmodified.
    A node without a storage node is never written to the result file: a scene Save offers a new file instead.
    """

    count = getattr(node, "GetNumberOfStorageNodes", None)
    if count is None:
        return 0
    storages = [node.GetNthStorageNode(index) for index in range(int(count()))]
    removed = 0
    for storage in storages:
        if storage is not None:
            slicer.mrmlScene.RemoveNode(storage)
            removed += 1
    return removed


def apply_fiber_display(slicer: Any, node: Any, kind: Optional[str]) -> bool:
    """The one fiber style: DEC by direct RGB mapping, opaque lines, 2D slice intersections. No tubes."""

    import vtk

    kind = kind if kind in LINE_WIDTH else KIND_TRACTOGRAPHY
    poly_data = node.GetPolyData() if hasattr(node, "GetPolyData") else None
    if poly_data is None:
        return False
    add_direction_colors(poly_data)
    detach_result_storage(slicer, node)  # b2fix D1: a scene Save must not rewrite the published result file
    display = node.GetDisplayNode()
    if display is None and hasattr(node, "CreateDefaultDisplayNodes"):
        node.CreateDefaultDisplayNodes()
        display = node.GetDisplayNode()
    if display is None:
        return False
    direct = getattr(getattr(slicer, "vtkMRMLDisplayNode", None), "UseDirectMapping", 4)
    display.SetActiveScalar(DEC_ARRAY, vtk.vtkAssignAttribute.POINT_DATA)
    display.SetScalarRangeFlag(direct)
    display.SetScalarVisibility(True)
    display.SetColor(*NEUTRAL_COLOR)
    display.SetOpacity(1.0)
    display.SetLineWidth(LINE_WIDTH[kind])
    display.SetVisibility2D(True)
    display.SetSliceDisplayModeToIntersection()
    display.SetSliceIntersectionThickness(3)
    node.SetAttribute(_PREFIX + "DisplayKind", DISPLAY_KIND)
    node.SetAttribute(_PREFIX + "StreamlineKind", kind)
    node.SetAttribute(_PREFIX + "FiberColorMode", "direction")
    return True
