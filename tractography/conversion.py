"""Lossless geometric TRK/TCK export for consumers that require VTK."""

import hashlib
import itertools
import json
from pathlib import Path
import tempfile


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def ensure_vtk_tractogram(source, target):
    """Preserve every streamline in RAS mm; never use a display subsample."""
    import nibabel as nib
    import numpy as np
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
    from connectome.pipeline import _iter_streamlines
    from shared.output_contracts import polydata_streamline_count

    source, target = Path(source).resolve(), Path(target).resolve()
    if source.suffix.lower() not in {'.trk', '.tck'} or target.suffix.lower() != '.vtk' or source == target:
        raise ValueError('Expected a TRK/TCK source and a separate VTK target')
    before = _digest(source)
    sidecar = target.with_suffix('.conversion.json')
    try:
        saved = json.loads(sidecar.read_text(encoding='utf-8'))
        if (saved.get('version') == 1 and saved.get('source') == str(source)
                and saved.get('source_sha256') == before and target.is_file()
                and saved.get('vtk_sha256') == _digest(target)):
            return saved
    except (OSError, ValueError, AttributeError):
        pass
    tracks = []
    for line in nib.streamlines.load(str(source), lazy_load=True).streamlines:
        points = np.asarray(line)
        if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 2
                or not np.isfinite(points).all() or not np.any(np.diff(points, axis=0))):
            raise ValueError('Invalid source streamline; conversion never drops or repairs tracks')
        tracks.append(points)
    if not tracks:
        raise ValueError('Cannot convert an empty tractogram')
    coordinates = np.concatenate(tracks)
    points, lines = vtk.vtkPoints(), vtk.vtkCellArray()
    points.SetData(numpy_to_vtk(coordinates, deep=True))
    id_dtype = np.int64 if vtk.vtkIdTypeArray().GetDataTypeSize() == 8 else np.int32
    offsets = np.cumsum([0, *(len(line) for line in tracks)], dtype=id_dtype)
    lines.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True),
                  numpy_to_vtkIdTypeArray(np.arange(len(coordinates), dtype=id_dtype), deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetLines(lines)
    space = vtk.vtkStringArray()
    space.SetName('SPACE')
    space.InsertNextValue('RAS')
    poly.GetFieldData().AddArray(space)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.vtk-conversion-', dir=target.parent) as temporary:
        staged = Path(temporary) / target.name
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(str(staged))
        writer.SetHeader('SlicerDSpace geometric conversion. SPACE=RAS')
        writer.SetInputData(poly)
        writer.SetFileTypeToBinary()
        if writer.Write() != 1 or polydata_streamline_count(staged) != len(tracks):
            raise ValueError('VTK export failed its decoded geometry/count check')
        maximum = 0.
        for original, restored in itertools.zip_longest(tracks, _iter_streamlines(staged)):
            if original is None or restored is None or np.shape(original) != np.shape(restored):
                raise ValueError('VTK export changed streamline or point counts')
            delta = float(np.max(np.abs(original - np.asarray(restored))))
            if delta > 1e-4:
                raise ValueError('VTK export changed RAS coordinates beyond 0.0001 mm')
            maximum = max(maximum, delta)
        if _digest(source) != before:
            raise ValueError('Source tractogram changed during conversion')
        record = dict(version=1, source=str(source), source_sha256=before,
                      vtk_sha256=_digest(staged), streamline_count=len(tracks),
                      coordinate_system='RAS', maximum_coordinate_delta_mm=maximum,
                      scope='Geometry only; tensor/point arrays are not present in this export')
        staged_json = Path(temporary) / sidecar.name
        staged_json.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        staged.replace(target)
        staged_json.replace(sidecar)
    return record
