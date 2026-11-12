"""Fail-closed, format-aware contracts shared by planning and execution."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path


def readable_output(path: Path) -> bool:
    """Decode supported scientific formats; other files must be non-empty."""
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return False
        name = path.name.lower()
        if name.endswith((".nii", ".nii.gz", ".mgz")):
            import nibabel as nib
            import numpy as np

            img = nib.load(str(path))
            data = np.asanyarray(img.dataobj)
            return bool(data.size and np.isfinite(data).all()
                        and np.isfinite(img.affine).all()
                        and abs(np.linalg.det(img.affine[:3, :3])) > 1e-12)
        if name.endswith(".stl"):
            import numpy as np
            import trimesh

            mesh = trimesh.load(str(path), process=False)
            return bool(len(mesh.vertices) and len(mesh.faces) and np.isfinite(mesh.vertices).all())
        if name.endswith(".json"):
            record = json.loads(path.read_text())
            return isinstance(record, dict) and bool(record) and record.get("status") not in {
                "failed", "running", "planned", "partial_failed", "cancelled"
            }
        return True
    except Exception:
        return False


def outputs_readable(paths: Iterable[str | Path]) -> bool:
    """Require at least one output and validate every member."""
    found = False
    for path in paths:
        found = True
        if not readable_output(Path(path)):
            return False
    return found


def tractography_readable(tck: Path, trk: Path, weights: Path | None = None) -> bool:
    try:
        import nibabel as nib
        import numpy as np

        counts = []
        for path in dict.fromkeys((tck, trk)):
            count = 0
            for streamline in nib.streamlines.load(str(path), lazy_load=True).streamlines:
                if len(streamline) < 2 or not np.isfinite(streamline).all():
                    return False
                count += 1
            counts.append(count)
        if not counts[0] or len(set(counts)) != 1:
            return False
        if weights is not None:
            values = np.loadtxt(weights, ndmin=1)
            if values.ndim != 1 or values.size != counts[0]:
                return False
            if not np.isfinite(values).all() or (values < 0).any() or not (values > 0).any():
                return False
        return True
    except Exception:
        return False


def vtp_streamline_count(path: str | Path) -> int | None:
    """Decode VTP geometry; return None for corruption or unavailable VTK.

    Empty but valid polydata returns zero. A Piece header alone is not evidence
    that the appended/compressed geometry is readable.
    """
    return polydata_streamline_count(path)


def polydata_streamline_count(path: str | Path) -> int | None:
    """Decode legacy VTK or XML VTP geometry, not only its header."""
    try:
        import numpy as np
        from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
        from vtkmodules.vtkIOLegacy import vtkPolyDataReader
        from vtkmodules.util.numpy_support import vtk_to_numpy

        path = Path(path)
        if not path.is_file() or path.stat().st_size == 0:
            return None
        if path.suffix.lower() == '.vtp':
            reader = vtkXMLPolyDataReader()
        elif path.suffix.lower() == '.vtk':
            reader = vtkPolyDataReader()
        else:
            return None
        errors = []
        reader.AddObserver("ErrorEvent", lambda *_: errors.append(True))
        # Legacy readers otherwise accept partially parsed ASCII geometry.
        reader.AddObserver("WarningEvent", lambda *_: errors.append(True))
        reader.SetFileName(str(path))
        if path.suffix.lower() == '.vtp' and not reader.CanReadFile(str(path)):
            return None
        reader.Update()
        if errors or reader.GetErrorCode():
            return None
        data = reader.GetOutput()
        if path.suffix.lower() == '.vtk':
            import re
            with path.open('rb') as stream:
                head_bytes = stream.read(65536)
            header = head_bytes.split(b'\n', 4)
            if len(header) < 4 or header[2].strip().upper() not in {b'ASCII', b'BINARY'}:
                return None
            is_ascii = header[2].strip().upper() == b'ASCII'
            raw_bytes = path.read_bytes() if is_ascii else head_bytes
            declared = re.search(rb'\bPOINTS\s+(\d+)\s+\w+\s', raw_bytes)
            if declared is None and not is_ascii:
                raw_bytes = path.read_bytes()
                declared = re.search(rb'\bPOINTS\s+(\d+)\s+\w+\s', raw_bytes)
            if declared is None or int(declared.group(1)) != data.GetNumberOfPoints():
                return None
            if is_ascii:
                # vtkDataReader may replace malformed ASCII coordinates with
                # zeros without setting an error code. Validate the original.
                raw = raw_bytes[declared.end():].split(maxsplit=3 * data.GetNumberOfPoints())
                values = np.asarray(raw[:3 * data.GetNumberOfPoints()], dtype=float)
                if values.size != 3 * data.GetNumberOfPoints() or not np.isfinite(values).all():
                    return None
        points = data.GetPoints()
        if points is not None and not np.isfinite(vtk_to_numpy(points.GetData())).all():
            return None
        if data.GetNumberOfVerts() or data.GetNumberOfPolys() or data.GetNumberOfStrips():
            return None
        lines = data.GetLines()
        count = data.GetNumberOfLines()
        offsets = vtk_to_numpy(lines.GetOffsetsArray())
        connectivity = vtk_to_numpy(lines.GetConnectivityArray())
        if len(offsets) != count + 1 or offsets[0] != 0 or offsets[-1] != len(connectivity):
            return None
        if count and (np.any(np.diff(offsets) < 2) or np.any(connectivity < 0)
                      or np.any(connectivity >= data.GetNumberOfPoints())):
            return None
        if count:
            ordered = vtk_to_numpy(points.GetData())[connectivity]
            changes = np.any(np.diff(ordered, axis=0) != 0, axis=1)
            changes[offsets[1:-1] - 1] = False
            if np.any(np.add.reduceat(changes.astype(np.int64), offsets[:-1]) == 0):
                return None
        return count
    except (ImportError, OSError, ValueError, RuntimeError, AttributeError, IndexError):
        return None
