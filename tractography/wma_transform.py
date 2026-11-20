"""Batch the pinned author's VTK transform without changing point arithmetic.

Derived from SlicerDMRI/whitematteranalysis register_two_subjects.py at
4532b911386d2d1bc7ebba63fdd13e65b1660623. Not an upstream release.
All or portions of this licensed product (such portions are the "Software")
have been obtained under license from The Brigham and Women's Hospital, Inc.
See licenses/WMA.txt for the applicable terms and conditions.
"""

import hashlib
from pathlib import Path
import sys
import time


QUALIFIED_VTK = '9.5.2'
_REPORTED_UNQUALIFIED_VTK = set()


def _report_unqualified_vtk(version):
    """F6: say once per run why the faster transform path is off."""
    if version in _REPORTED_UNQUALIFIED_VTK:
        return
    _REPORTED_UNQUALIFIED_VTK.add(version)
    print('WMA fast path qualified only for VTK %s; this Slicer has VTK %s; using the reference path '
          '(same results, slower)' % (QUALIFIED_VTK, version), file=sys.stderr, flush=True)


AUTHOR_SHA256 = 'eb6cb9293df3fe3e2d783e096ea979ae279e94c7f3e9f335b3fc34818a93920f'


def registration_transform_support(package_dir):
    source = Path(package_dir) / 'register_two_subjects.py'
    digest = hashlib.sha256(source.read_bytes()).hexdigest() if source.is_file() else None
    if digest != AUTHOR_SHA256:
        return {'supported': False, 'source_sha256': digest, 'reason': 'Unverified registration source'}
    import vtk
    version = vtk.vtkVersion.GetVTKVersion()
    if version != QUALIFIED_VTK:
        _report_unqualified_vtk(version)
    return {'supported': version == '9.5.2', 'source_sha256': digest, 'vtk': version,
            'reason': 'Verified author source and VTK' if version == '9.5.2' else 'Unqualified VTK; retain author transform'}


class AuthorCompatibleFiberTransform:
    """Use the same vtkTransform on double points, with bounded temporary data."""

    def __init__(self, original, converter, max_bytes=64 * 1024**2):
        if max_bytes < 0:
            raise ValueError('max_bytes must be nonnegative')
        self.__wrapped__ = original
        self.converter = converter
        self.max_bytes = max_bytes
        self.calls = self.bypassed = self.points = 0
        self.seconds = 0.

    def __call__(self, in_array, transform, mode=(1, 1, 1, 1)):
        import numpy as np
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

        if (type(in_array) is not np.ndarray or in_array.ndim != 3 or in_array.shape[0] != 3
                or in_array.dtype not in (np.float32, np.float64) or in_array.size == 0
                or in_array.size * 8 * 4 > self.max_bytes or not np.isfinite(in_array).all()):
            self.bypassed += 1
            return self.__wrapped__(in_array, transform, mode=mode)
        started = time.monotonic()
        positions = np.ascontiguousarray(np.moveaxis(in_array, 0, -1).reshape(-1, 3), dtype=np.float64)
        source = vtk.vtkPoints()
        source.SetData(numpy_to_vtk(positions, deep=False))
        destination = vtk.vtkPoints()
        destination.SetDataTypeToDouble()
        vtk_transform = self.converter(transform, scaled=True, mode=mode)
        vtk_transform.TransformPoints(source, destination)
        result = vtk_to_numpy(destination.GetData()).reshape(in_array.shape[1], in_array.shape[2], 3)
        # Own the output, matching the author's C-order float64 array.
        result = np.ascontiguousarray(np.moveaxis(result, -1, 0))
        self.calls += 1
        self.points += len(positions)
        self.seconds += time.monotonic() - started
        return result

    def summary(self):
        return {'calls': self.calls, 'author_bypasses': self.bypassed, 'points': self.points,
                'bulk_seconds': self.seconds, 'temporary_budget_bytes': self.max_bytes,
                'scope': 'Same author vtkTransform and float64 point order; no objective, sampling, optimizer or classification changes'}
