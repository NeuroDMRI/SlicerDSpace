"""Strict scientific-product fingerprints for a frozen cohort experiment."""

import hashlib
import json
from pathlib import Path


def _array(digest, value):
    import numpy as np
    value = np.asarray(value)
    digest.update(json.dumps([value.shape, value.dtype.str]).encode())
    # Hash stored values, shape and dtype without silently rounding differences.
    flat = value.ravel(order='C')
    for start in range(0, flat.size, 1024*1024):
        digest.update(flat[start:start+1024*1024].tobytes())


def product_hash(path, *, allow_empty=False):
    import numpy as np
    digest = hashlib.sha256()
    if path.name.endswith(('.nii', '.nii.gz', '.mgh', '.mgz')):
        import nibabel as nib
        image = nib.load(path)
        digest.update(str(image.header.get_data_dtype()).encode())
        _array(digest, image.affine)
        _array(digest, np.asanyarray(image.dataobj))
    elif path.suffix == '.npy':
        _array(digest, np.load(path, mmap_mode='r'))
    elif path.suffix == '.txt' and any(word in path.name.lower() for word in ('sift','weight')):
        _array(digest, np.loadtxt(path, comments='#', ndmin=1))
    elif path.suffix in ('.tck', '.trk'):
        import nibabel as nib
        tractogram = nib.streamlines.load(str(path), lazy_load=True).tractogram
        for track in tractogram:
            _array(digest, track.streamline)
            for kind in (track.data_for_streamline, track.data_for_points):
                for key in sorted(kind):
                    digest.update(key.encode())
                    _array(digest, kind[key])
    elif path.suffix in ('.vtk', '.vtp', '.stl'):
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
        reader = {'.vtp': vtk.vtkXMLPolyDataReader, '.vtk': vtk.vtkPolyDataReader,
                  '.stl': vtk.vtkSTLReader}[path.suffix]()
        errors = []
        reader.AddObserver(vtk.vtkCommand.ErrorEvent, lambda *_: errors.append(True))
        if path.suffix == '.vtp' and not reader.CanReadFile(str(path)):
            raise ValueError('Unreadable polydata: '+str(path))
        reader.SetFileName(str(path))
        reader.Update()
        data = reader.GetOutput()
        if errors or reader.GetErrorCode() or (data.GetNumberOfPoints() == 0 and not allow_empty):
            raise ValueError('Empty or unreadable polydata: '+str(path))
        if data.GetPoints() is None:
            digest.update(b'empty-polydata-without-points')
        else:
            _array(digest, vtk_to_numpy(data.GetPoints().GetData()))
        for cells in (data.GetVerts(), data.GetLines(), data.GetPolys(), data.GetStrips()):
            _array(digest, vtk_to_numpy(cells.GetOffsetsArray()))
            _array(digest, vtk_to_numpy(cells.GetConnectivityArray()))
        for attributes in (data.GetPointData(), data.GetCellData(), data.GetFieldData()):
            for i in range(attributes.GetNumberOfArrays()):
                array = attributes.GetAbstractArray(i)
                digest.update(str(array.GetName()).encode())
                if isinstance(array, vtk.vtkDataArray):
                    _array(digest, vtk_to_numpy(array))
                else:
                    digest.update(str([array.GetVariantValue(j).ToString()
                                       for j in range(array.GetNumberOfValues())]).encode())
    else:
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


def scientific_paths(root):
    """Select the same delivered arrays, geometry and saved cluster intermediates."""
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if not path.is_file() or relative.parts[0] not in ('02_microstructure','03_anatomy','04_tractography'):
            continue
        if any(part.startswith(('.', '_')) or part in ('cache', 'logs') for part in relative.parts):
            continue
        if path.name.endswith(('.nii','.nii.gz','.mgh','.mgz','.npy','.tck','.trk','.vtk','.vtp','.stl','.tfm')) or (
                path.suffix in ('.csv', '.tsv') and any('connectome' in part.lower() for part in relative.parts)) or (
                path.suffix == '.txt' and any(word in path.name.lower() for word in ('sift', 'weight', 'transform'))):
            yield path


def scientific_inventory(root):
    rows = {}
    for path in scientific_paths(root):
        relative = path.relative_to(root)
        # Atlas cluster slots may be legitimately empty after classification,
        # outlier removal or hemisphere separation. Final-product QC remains strict.
        cluster = ('FiberClustering' in relative.parts and path.name.startswith('cluster_')
                   and path.suffix in ('.vtk', '.vtp'))
        rows[str(relative)] = product_hash(path, allow_empty=cluster)
    if not rows:
        raise ValueError('No scientific arrays or tractograms found: '+str(root))
    return rows


def compare_inventories(reference, candidate):
    missing = sorted(reference.keys()-candidate.keys())
    extra = sorted(candidate.keys()-reference.keys())
    different = sorted(key for key in reference.keys() & candidate.keys() if reference[key] != candidate[key])
    return dict(status='passed' if reference and not (missing or extra or different) else 'failed',
                product_count=len(reference), missing=missing, extra=extra, different=different,
                scope='Exact array/dtype/affine, streamline ordering, mesh topology/attributes, weights and connectome tables; excludes logs and intermediate MIF files')
