"""SlicerDMRI-independent WMA tract scalar summary using VTK and NumPy.

This is a **separately identified method**, not a claim of bitwise/numerical
parity with SlicerDMRI FiberTractMeasurements. It computes the requested
per-tract scalar means and geometry with ordinary VTK point-data arrays.
Validation against the upstream CLI must precede scientific certification.

It deliberately reports unsupported tensor-only inputs instead of silently
inventing FA/MD values; use real comparison fixtures before enabling that route
in a scientific release. The emitted CSVs use the same names as the WMA author
shell, with an explicit sidecar declaring their method.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


class WMAUnsupportedMeasurement(RuntimeError):
    pass


def _scalar_clamp(name):
    label = name.casefold()
    if 'fractionalanisotropy' in label or label == 'fa':
        return (0., 1.)
    if 'relativeanisotropy' in label:
        return (0., 2**0.5)
    if any(s in label for s in ('linearmeasure', 'planarmeasure', 'sphericalmeasure')):
        return (0., 1.)
    return None


def measure_polydata(poly):
    import numpy as np
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    points = poly.GetPoints()
    count = int(points.GetNumberOfPoints()) if points is not None else 0
    lines = poly.GetLines()
    line_count, total_length = 0, 0.
    if lines is not None:
        ids = vtk.vtkIdList()
        lines.InitTraversal()
        while lines.GetNextCell(ids):
            n = ids.GetNumberOfIds()
            if n == 0:
                continue
            line_count += 1
            for index in range(1, n):
                p, q = points.GetPoint(ids.GetId(index - 1)), points.GetPoint(ids.GetId(index))
                total_length += math.dist(p, q)
    row = {
        'Num_Points': count,
        'Num_Fibers': line_count,
        'Mean_Length': total_length / line_count if line_count else math.nan,
        'Num_Clamp_Excluded': 0,
    }
    data = poly.GetPointData()
    for index in range(data.GetNumberOfArrays() if data else 0):
        array = data.GetArray(index)
        if array is None:
            continue
        name = array.GetName() or f'Scalar_{index}'
        components = array.GetNumberOfComponents()
        if components > 1:
            # Do not claim FTM's eigenvalue/FA tensor math without an
            # explicit, independently tested implementation.
            if components in (6, 9):
                raise WMAUnsupportedMeasurement(
                    f'Tensor array {name!r} has {components} components; '
                    'native scalar-only WMA statistics cannot replace FiberTractMeasurements tensor computations.')
            continue
        values = vtk_to_numpy(array).astype(np.float64, copy=False).reshape(-1)
        if values.size != count:
            raise ValueError(f'Inconsistent number of values in {name}')
        finite = np.isfinite(values)
        allowed = _scalar_clamp(name)
        if allowed is not None:
            valid = finite & (values >= allowed[0]) & (values <= allowed[1])
            row['Num_Clamp_Excluded'] += int(np.sum(finite & ~valid))
        else:
            valid = finite
        row[name] = float(np.mean(values[valid])) if valid.any() else math.nan
    return row


def _read_polydata(path):
    import vtk
    reader = vtk.vtkXMLPolyDataReader() if path.suffix.lower() == '.vtp' else vtk.vtkPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    if poly is None:
        raise ValueError(f'Cannot read WMA tract polydata: {path}')
    return poly


def measure_directory(directory: Path, csv_path: Path):
    """Write a stable per-tract table. Never overwrite existing measurements."""
    directory, csv_path = Path(directory), Path(csv_path)
    if not directory.is_dir():
        raise FileNotFoundError(f'WMA measurement input directory missing: {directory}')
    files = sorted([*directory.glob('*.vtp'), *directory.glob('*.vtk')])
    if not files:
        raise FileNotFoundError(f'No WMA VTK/VTP bundles in {directory}')
    rows = []
    for path in files:
        row = {'Name': path.stem}
        row.update(measure_polydata(_read_polydata(path)))
        rows.append(row)
    columns = ['Name', 'Num_Points', 'Num_Fibers', 'Mean_Length', 'Num_Clamp_Excluded']
    columns.extend(sorted(set().union(*(set(row) for row in rows)) - set(columns)))
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        raise FileExistsError(f'Will not replace an existing WMA measurement table: {csv_path}')
    with csv_path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return {'csv': str(csv_path), 'tracts': len(rows), 'scalar_fields': columns[5:]}


def export_wma_case_measurements(output_root, case_id):
    """Compute four WMA measurements without the SlicerDMRI binary.

    Authors' WMA script exports the same four CSV basename locations when
    -d=1; this function operates on already-produced clusters. The method is
    explicitly annotated in a sidecar. Any failure invalidates the export.
    """
    root = Path(output_root) / case_id
    separated = root / 'FiberClustering' / 'SeparatedClusters'
    pairs = [
        (separated / 'tracts_commissural', separated / 'diffusion_measurements_commissural.csv'),
        (separated / 'tracts_left_hemisphere', separated / 'diffusion_measurements_left_hemisphere.csv'),
        (separated / 'tracts_right_hemisphere', separated / 'diffusion_measurements_right_hemisphere.csv'),
        (root / 'AnatomicalTracts', root / 'AnatomicalTracts' / 'diffusion_measurements_anatomical_tracts.csv'),
    ]
    outputs = [measure_directory(directory, destination) for directory, destination in pairs]
    for result in outputs:
        result['csv'] = str(Path(result['csv']).relative_to(root))
    record = {
        'kind': 'slicerdspace_wma_measurements',
        'backend': 'native_vtk_point_scalar_summary',
        'replaces_slicerdmri': False,
        'upstream_fibertractmeasurements_parity_certified': False,
        'method_warning': 'New explicit native scalar-summary method; not validated as equivalent to SlicerDMRI FiberTractMeasurements.',
        'products': outputs,
    }
    sidecar = root / 'wma_native_measurements.json'
    if sidecar.exists():
        raise FileExistsError(sidecar)
    sidecar.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return record
