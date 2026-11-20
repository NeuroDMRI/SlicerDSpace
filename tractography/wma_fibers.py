# Adapted from WMA FiberArray.convert_from_polydata and _calculate_line_indices.
# Copyright (c) 2009-2011, Brigham and Women's Hospital. All rights reserved.
# Copyright (c) 2009-2011, The Slicer Community. All rights reserved.
# Modified for SlicerDSpace, 2026: bounded bulk copying, same sampled points.
# See licenses/WMA.txt for the complete upstream agreement.
"""Preserve author nearest-neighbor parameterization, without point-call loops."""

from functools import wraps
import hashlib
from pathlib import Path
import time

from tractography.wma_transform import _report_unqualified_vtk


QUALIFIED_VTK = '9.5.2'


AUTHOR_SHA256 = '5f1164336b32c01e3d5372bdebb7a4ba5a6cde92b589228c162242eaacdee82d'


def fiber_conversion_support(package_dir: Path) -> dict:
    path = package_dir / 'fibers.py'
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    if digest != AUTHOR_SHA256:
        return dict(supported=False, source_sha256=digest, vtk_version=None,
                    reason='unverified source; original conversion retained')
    import vtk
    version = vtk.vtkVersion.GetVTKVersion()
    supported = digest == AUTHOR_SHA256 and version == '9.5.2'
    if version != QUALIFIED_VTK:
        _report_unqualified_vtk(version)
    return dict(supported=supported, source_sha256=digest, vtk_version=version,
                reason='verified author sampling and VTK arrays' if supported else 'unverified source/VTK; original conversion retained')


class AuthorCompatibleFiberArrayConverter:
    """A descriptor-compatible method wrapper, scoped to an owned WMA process."""

    def __init__(self, fiber_class, *, block_size=4096, max_aux_bytes=64*1024**2):
        if block_size < 1 or max_aux_bytes < 0:
            raise ValueError('Positive block size and nonnegative auxiliary budget required')
        self.fiber_class = fiber_class
        self.original = fiber_class.convert_from_polydata
        self.indexer = fiber_class._calculate_line_indices
        self.block_size = block_size
        self.max_aux_bytes = max_aux_bytes
        self.calls = self.bypassed = self.fibers = 0
        self.seconds = 0.

        @wraps(self.original)
        def method(instance, input_vtk_polydata, points_per_fiber=None):
            return self(instance, input_vtk_polydata, points_per_fiber)
        self.method = method

    def __call__(self, instance, source, points_per_fiber=None):
        import numpy as np
        from vtk.util.numpy_support import vtk_to_numpy
        p = instance.points_per_fiber if points_per_fiber is None else points_per_fiber
        lines = source.GetLines()
        points = source.GetPoints()
        n = source.GetNumberOfLines()
        indexer = getattr(instance._calculate_line_indices, '__func__', None)
        qualified = (type(instance) is self.fiber_class and indexer is self.indexer
                     and not instance.verbose and isinstance(p, (int, np.integer)) and p >= 2
                     and n > 0 and points is not None and lines is not None
                     and 3*n*8 + min(n, self.block_size)*p*48 <= self.max_aux_bytes)
        if not qualified:
            self.bypassed += 1
            return self.original(instance, source, points_per_fiber)
        coordinates = vtk_to_numpy(points.GetData())
        offsets = vtk_to_numpy(lines.GetOffsetsArray())
        ids = vtk_to_numpy(lines.GetConnectivityArray())
        lengths = np.diff(offsets)
        if (coordinates.ndim != 2 or coordinates.shape[1] != 3
                or coordinates.dtype not in (np.float32, np.float64)
                or len(offsets) != n+1 or offsets[0] != 0 or offsets[-1] != len(ids)
                or np.any(lengths <= 0) or ids.min(initial=0) < 0
                or ids.max(initial=-1) >= len(coordinates)):
            self.bypassed += 1
            return self.original(instance, source, points_per_fiber)
        started = time.monotonic()
        outputs = [np.zeros((n, p)) for _ in range(3)]
        for start in range(0, n, self.block_size):
            stop = min(start+self.block_size, n)
            step = (lengths[start:stop]-1.0)/(p-1.0)
            # Same float64 multiply and ties-to-even rounding as the author's
            # int(round(point_index * step)); no arc-length interpolation.
            index = np.rint(step[:, None]*np.arange(p)[None, :]).astype(np.int64)
            index += offsets[start:stop, None]
            selected = coordinates[ids[index]]
            for axis in range(3):
                outputs[axis][start:stop] = selected[..., axis]
        instance.points_per_fiber = p
        instance.number_of_fibers = n
        for axis, values in zip('ras', outputs):
            setattr(instance, 'fiber_array_'+axis, values)
        # Match the author's traversal side effect, not just its output arrays.
        lines.InitTraversal()
        lines.SetTraversalCellId(n)
        if instance.hemispheres:
            instance.calculate_hemispheres()
        self.calls += 1
        self.fibers += n
        self.seconds += time.monotonic()-started

    def summary(self) -> dict:
        return dict(calls=self.calls, bypassed=self.bypassed, fibers=self.fibers,
                    seconds=self.seconds, block_size=self.block_size,
                    max_aux_bytes=self.max_aux_bytes,
                    scope='Author nearest-neighbor array conversion only; no streamline or solver change')
