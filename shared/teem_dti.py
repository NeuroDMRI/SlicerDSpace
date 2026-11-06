"""Diffusion tensor estimation and scalar maps with the Teem library of 3D Slicer core.

This module replaces the SlicerDMRI command-line modules ``DWIToDTIEstimation`` and
``DiffusionTensorScalarMeasurements`` (SlicerDMRI revision a40200b). It uses only parts that
ship with 3D Slicer itself:

* ``vtkTeemNRRDReader`` / ``vtkTeemNRRDWriter`` / ``vtkDiffusionTensorMathematics`` (vtkTeem),
* ``vtkMRMLNRRDStorageNode.ParseDiffusionInformation`` (MRML core),
* the Teem library (``libteem``, Teem 1.12.0) through ``ctypes``.

The one SlicerDMRI class that is not in Slicer core, ``vtkTeemEstimateDiffusionTensor``, is a
thin loop around Teem's ``tenEstimate1TensorSingle_d``. This module sets up the same Teem
estimation context with the same values (method, minimum signal 1.0, one WLS iteration,
B0 estimated, threshold 0/1, no eigenvalue shift), transforms and rescales the gradients
exactly as that class does, and calls the same Teem estimator. The numbers therefore come
from the same compiled code as before; only the loop that visits the voxels moved here.

Differences from the SlicerDMRI loop, both deliberate:

* Threads. The voxels are split into z slabs. Each worker thread owns one Teem context and
  fits whole slabs with ``tenEstimate1TensorVolume4D``. The number of threads is an argument
  (default: ``SLICERDSPACE_STAGE_CPUS``, else the CPUs this process may use). SlicerDMRI used
  every core of the machine, whatever the stage reservation was.
* A voxel whose estimation fails. SlicerDMRI ignored the error code and kept the tensor of the
  previous voxel in the same thread (undefined for the first voxel of a thread extent). Here
  such a voxel gets Teem's own failure output (a NaN tensor), and the count is reported.
  Teem estimation does not fail for finite input; the count is 0 on real data.

Run with Slicer's Python (``PythonSlicer``)::

    PythonSlicer teem_dti.py estimate --enumeration WLS DWI.nhdr DTI.nrrd b0.nrrd \
        --scalar FractionalAnisotropy=DTI_FA.nrrd --scalar Trace=DTI_Trace.nrrd
    PythonSlicer teem_dti.py scalars DTI.nrrd FractionalAnisotropy=DTI_FA.nrrd
    PythonSlicer teem_dti.py selfcheck

Licences: Teem is LGPL-2.1 with a static-linking exception (shipped by Slicer); the
replicated logic follows SlicerDMRI (3D Slicer licence). See THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

import argparse
import ctypes
import glob
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

TEEM_EXPECTED_VERSION_PREFIX = "1.12"
NRRD_TYPE_DOUBLE = 10
TEN_METHOD_LLS = 1
TEN_METHOD_WLS = 2
# tenEstimate1Method values (teem/ten/ten.h); SlicerDMRI enumeration names.
FIT_METHODS = {"LS": TEN_METHOD_LLS, "WLS": TEN_METHOD_WLS}
# vtkTeemEstimateDiffusionTensor defaults (SlicerDMRI a40200b).
MINIMUM_SIGNAL_VALUE = 1.0
NUMBER_OF_WLS_ITERATIONS = 1
KNOWN_B0 = 0.0
GRADIENT_NORM_EPSILON = 10e-12

# DiffusionTensorScalarMeasurements enumeration -> vtkDiffusionTensorMathematics setter suffix.
SCALAR_OPERATIONS = {
    "Trace": "Trace",
    "Determinant": "Determinant",
    "RelativeAnisotropy": "RelativeAnisotropy",
    "FractionalAnisotropy": "FractionalAnisotropy",
    "Mode": "Mode",
    "LinearMeasurement": "LinearMeasure",
    "LinearMeasure": "LinearMeasure",
    "PlanarMeasurement": "PlanarMeasure",
    "PlanarMeasure": "PlanarMeasure",
    "SphericalMeasurement": "SphericalMeasure",
    "SphericalMeasure": "SphericalMeasure",
    "MinEigenvalue": "MinEigenvalue",
    "MidEigenvalue": "MiddleEigenvalue",
    "MaxEigenvalue": "MaxEigenvalue",
    "ColorByOrientation": "ColorByOrientation",
    "ColorByMode": "ColorByMode",
    "MaxEigenvalueProjectionX": "MaxEigenvalueProjectionX",
    "MaxEigenvalueProjectionY": "MaxEigenvalueProjectionY",
    "MaxEigenvalueProjectionZ": "MaxEigenvalueProjectionZ",
    "RAIMaxEigenvecX": "RAIMaxEigenvecX",
    "RAIMaxEigenvecY": "RAIMaxEigenvecY",
    "RAIMaxEigenvecZ": "RAIMaxEigenvecZ",
    "MaxEigenvecX": "MaxEigenvecX",
    "MaxEigenvecY": "MaxEigenvecY",
    "MaxEigenvecZ": "MaxEigenvecZ",
    "D11": "D11",
    "D22": "D22",
    "D33": "D33",
    "ParallelDiffusivity": "ParallelDiffusivity",
    "PerpendicularDiffusivity": "PerpendicularDiffusivity",
    "MeanDiffusivity": "MeanDiffusivity",
}


class TeemError(RuntimeError):
    """The Teem library is missing, has an unexpected layout, or reported an error."""


class _TenEstimateContext(ctypes.Structure):
    """Mirror of ``tenEstimateContext`` in Teem 1.12.0 (teem/ten/ten.h, Slicer/teem 43395a6)."""

    _fields_ = [
        ("bValue", ctypes.c_double),
        ("valueMin", ctypes.c_double),
        ("sigma", ctypes.c_double),
        ("dwiConfThresh", ctypes.c_double),
        ("dwiConfSoft", ctypes.c_double),
        ("_ngrad", ctypes.c_void_p),
        ("_nbmat", ctypes.c_void_p),
        ("skipList", ctypes.c_void_p),
        ("skipListArr", ctypes.c_void_p),
        ("all_f", ctypes.c_void_p),
        ("all_d", ctypes.c_void_p),
        ("simulate", ctypes.c_int),
        ("estimate1Method", ctypes.c_int),
        ("estimateB0", ctypes.c_int),
        ("recordTime", ctypes.c_int),
        ("recordErrorDwi", ctypes.c_int),
        ("recordErrorLogDwi", ctypes.c_int),
        ("recordLikelihoodDwi", ctypes.c_int),
        ("verbose", ctypes.c_int),
        ("negEvalShift", ctypes.c_int),
        ("progress", ctypes.c_int),
        ("WLSIterNum", ctypes.c_uint),
        ("flag", ctypes.c_int * 128),
        ("allNum", ctypes.c_uint),
        ("dwiNum", ctypes.c_uint),
        ("nbmat", ctypes.c_void_p),
        ("nwght", ctypes.c_void_p),
        ("nemat", ctypes.c_void_p),
        ("knownB0", ctypes.c_double),
        ("all", ctypes.c_void_p),
        ("bnorm", ctypes.c_void_p),
        ("allTmp", ctypes.c_void_p),
        ("dwiTmp", ctypes.c_void_p),
        ("dwi", ctypes.c_void_p),
        ("skipLut", ctypes.c_void_p),
        ("estimatedB0", ctypes.c_double),
        ("ten", ctypes.c_double * 7),
        ("conf", ctypes.c_double),
        ("mdwi", ctypes.c_double),
        ("time", ctypes.c_double),
        ("errorDwi", ctypes.c_double),
        ("errorLogDwi", ctypes.c_double),
        ("likelihoodDwi", ctypes.c_double),
    ]


_CONTEXT_P = ctypes.POINTER(_TenEstimateContext)
_DOUBLE_P = ctypes.POINTER(ctypes.c_double)
# Indices flagUnknown+1 .. flagLast-1 that tenEstimateContextNew clears (teem/ten/estimate.c).
_CLEARED_FLAG_RANGE = range(1, 12)

_LIBRARY = None
_LIBRARY_LOCK = threading.Lock()
_BIFF_LOCK = threading.Lock()


# --------------------------------------------------------------------------- library loading
def _library_file_names() -> list[str]:
    if sys.platform == "darwin":
        return ["libteem.1.dylib", "libteem.dylib"]
    if sys.platform.startswith("win"):
        return ["teem.dll", "libteem.dll"]
    return ["libteem.so.1", "libteem.so"]


def _slicer_roots() -> list[Path]:
    roots: list[Path] = []
    try:
        import vtkTeemPython  # noqa: F401  (loads libteem as a dependency)
        module_file = Path(vtkTeemPython.__file__).resolve()
        roots += [module_file.parents[2], module_file.parents[1]]
    except Exception:
        pass
    executable = Path(sys.executable).resolve()
    roots += [executable.parents[1], executable.parent]
    for key in ("SLICER_HOME", "SLICERDSPACE_SLICER_HOME"):
        if os.environ.get(key):
            roots.append(Path(os.environ[key]))
    unique: list[Path] = []
    for root in roots:
        if root not in unique and root.is_dir():
            unique.append(root)
    return unique


def _candidate_library_paths() -> list[str]:
    override = os.environ.get("SLICERDSPACE_TEEM_LIBRARY")
    if override:
        return [override]
    names = _library_file_names()
    found: list[str] = []
    for root in _slicer_roots():
        for pattern in ("lib/Teem-*/{}", "lib/Slicer-*/{}", "lib/{}", "bin/{}", "{}"):
            for name in names:
                found += sorted(glob.glob(str(root / pattern.format(name))))
    return found + names  # bare names last: the loader search path


def _bind(lib) -> None:
    c = ctypes
    signatures = {
        "tenEstimateContextNew": (_CONTEXT_P, []),
        "tenEstimateContextNix": (c.c_void_p, [_CONTEXT_P]),
        "tenEstimateVerboseSet": (None, [_CONTEXT_P, c.c_int]),
        "tenEstimateNegEvalShiftSet": (None, [_CONTEXT_P, c.c_int]),
        "tenEstimateMethodSet": (c.c_int, [_CONTEXT_P, c.c_int]),
        "tenEstimateSigmaSet": (c.c_int, [_CONTEXT_P, c.c_double]),
        "tenEstimateValueMinSet": (c.c_int, [_CONTEXT_P, c.c_double]),
        "tenEstimateBMatricesSet": (c.c_int, [_CONTEXT_P, c.c_void_p, c.c_double, c.c_int]),
        "tenEstimateThresholdSet": (c.c_int, [_CONTEXT_P, c.c_double, c.c_double]),
        "tenEstimateUpdate": (c.c_int, [_CONTEXT_P]),
        "tenEstimate1TensorSingle_d": (c.c_int, [_CONTEXT_P, _DOUBLE_P, _DOUBLE_P]),
        "tenEstimate1TensorVolume4D": (
            c.c_int, [_CONTEXT_P, c.c_void_p, c.POINTER(c.c_void_p), c.c_void_p, c.c_void_p, c.c_int]),
        "tenBMatrixCalc": (c.c_int, [c.c_void_p, c.c_void_p]),
        "nrrdNew": (c.c_void_p, []),
        "nrrdNix": (c.c_void_p, [c.c_void_p]),
        "nrrdNuke": (c.c_void_p, [c.c_void_p]),
        "nrrdWrap_nva": (c.c_int, [c.c_void_p, c.c_void_p, c.c_int, c.c_uint, c.POINTER(c.c_size_t)]),
        "biffGetDone": (c.c_void_p, [c.c_char_p]),
        "airFree": (c.c_void_p, [c.c_void_p]),
    }
    for name, (restype, argtypes) in signatures.items():
        function = getattr(lib, name)
        function.restype = restype
        function.argtypes = argtypes


def load_teem():
    """Load and bind Slicer's Teem library once; raise TeemError when it is not usable."""
    global _LIBRARY
    with _LIBRARY_LOCK:
        if _LIBRARY is not None:
            return _LIBRARY
        errors = []
        for candidate in _candidate_library_paths():
            try:
                lib = ctypes.CDLL(candidate)
                _bind(lib)
            except (OSError, AttributeError) as error:
                errors.append(f"{candidate}: {error}")
                continue
            version = teem_version(lib)
            if not version.startswith(TEEM_EXPECTED_VERSION_PREFIX):
                errors.append(f"{candidate}: Teem {version}, expected {TEEM_EXPECTED_VERSION_PREFIX}.x")
                continue
            lib._slicerdspace_path = candidate
            _check_context_layout(lib)
            _LIBRARY = lib
            return lib
        raise TeemError("Slicer's Teem library was not found or not usable:\n  " + "\n  ".join(errors))


def teem_version(lib) -> str:
    try:
        return (ctypes.c_char_p.in_dll(lib, "airTeemVersion").value or b"").decode()
    except (ValueError, AttributeError):
        return ""


def _take_biff_message(lib) -> str:
    with _BIFF_LOCK:
        pointer = lib.biffGetDone(b"ten")
        if not pointer:
            return ""
        message = ctypes.cast(pointer, ctypes.c_char_p).value or b""
        lib.airFree(pointer)
    return message.decode(errors="replace").strip()


# --------------------------------------------------------------------------- layout self-check
def _check_context_layout(lib) -> None:
    """Raise TeemError unless the struct mirror matches the loaded library.

    tenEstimateContextNew sets known values (NaN doubles, NULL pointers, WLSIterNum 3, ...);
    a wrong field offset reads one of them at the wrong place. Setters then round-trip.
    """
    tec = lib.tenEstimateContextNew()
    if not tec:
        raise TeemError("tenEstimateContextNew returned NULL")
    try:
        ctx = tec.contents
        problems = []
        nan_fields = ("bValue", "valueMin", "sigma", "dwiConfThresh", "dwiConfSoft", "knownB0",
                      "estimatedB0", "conf", "mdwi", "time", "errorDwi", "errorLogDwi", "likelihoodDwi")
        for name in nan_fields:
            if not math.isnan(getattr(ctx, name)):
                problems.append(f"{name}={getattr(ctx, name)!r} (expected NaN)")
        if not all(math.isnan(value) for value in ctx.ten):
            problems.append("ten is not all NaN")
        for name in ("_ngrad", "_nbmat", "all_f", "all_d", "all", "bnorm", "allTmp", "dwiTmp", "dwi", "skipLut"):
            if getattr(ctx, name):
                problems.append(f"{name} is not NULL")
        for name in ("skipListArr", "nbmat", "nwght", "nemat"):
            if not getattr(ctx, name):
                problems.append(f"{name} is NULL")
        expected = dict(simulate=0, estimate1Method=0, estimateB0=1, recordTime=0, recordErrorDwi=0,
                        recordErrorLogDwi=0, recordLikelihoodDwi=0, verbose=0, progress=0, WLSIterNum=3,
                        allNum=0, dwiNum=0)
        for name, value in expected.items():
            if getattr(ctx, name) != value:
                problems.append(f"{name}={getattr(ctx, name)} (expected {value})")
        if any(ctx.flag[index] for index in _CLEARED_FLAG_RANGE):
            problems.append("state flags are not cleared")
        lib.tenEstimateVerboseSet(tec, 7)
        lib.tenEstimateNegEvalShiftSet(tec, 1)
        lib.tenEstimateMethodSet(tec, TEN_METHOD_WLS)
        lib.tenEstimateValueMinSet(tec, 2.5)
        lib.tenEstimateSigmaSet(tec, 0.25)
        lib.tenEstimateThresholdSet(tec, 3.0, 4.0)
        round_trip = dict(verbose=7, negEvalShift=1, estimate1Method=TEN_METHOD_WLS, valueMin=2.5, sigma=0.25,
                          dwiConfThresh=3.0, dwiConfSoft=4.0)
        for name, value in round_trip.items():
            if getattr(ctx, name) != value:
                problems.append(f"setter round trip {name}={getattr(ctx, name)} (expected {value})")
        if problems:
            raise TeemError("tenEstimateContext layout does not match this Teem build: " + "; ".join(problems))
    finally:
        lib.tenEstimateContextNix(tec)


# --------------------------------------------------------------------------- estimation core
class _Gradients:
    """Rescaled gradients and the Teem B-matrix list, kept alive while contexts use them."""

    def __init__(self, lib, gradients: np.ndarray, max_b: float):
        self.lib = lib
        self.max_b = float(max_b)
        self.gradients = np.ascontiguousarray(gradients, dtype=np.float64)
        self.count = self.gradients.shape[0]
        self.ngrad = lib.nrrdNew()
        self.nbmat = lib.nrrdNew()
        sizes = (ctypes.c_size_t * 2)(3, self.count)
        if lib.nrrdWrap_nva(self.ngrad, self.gradients.ctypes.data, NRRD_TYPE_DOUBLE, 2, sizes):
            raise TeemError("nrrdWrap_nva failed for the gradient list")
        if lib.tenBMatrixCalc(self.nbmat, self.ngrad):
            raise TeemError("tenBMatrixCalc failed: " + _take_biff_message(lib))

    def close(self) -> None:
        self.lib.nrrdNix(self.ngrad)  # wrapper only; numpy owns the data
        self.lib.nrrdNuke(self.nbmat)


def _new_context(lib, gradients: _Gradients, method: int, shift_negative: bool):
    """Same settings, in the same order, as vtkTeemEstimateDiffusionTensor::SetTenContext."""
    tec = lib.tenEstimateContextNew()
    if not tec:
        raise TeemError("tenEstimateContextNew returned NULL")
    ctx = tec.contents
    ctx.progress = 0  # SlicerDMRI sets 1; it only prints progress from tenEstimate1TensorVolume4D
    failed = lib.tenEstimateBMatricesSet(tec, gradients.nbmat, gradients.max_b, int(not KNOWN_B0))
    ctx.knownB0 = KNOWN_B0
    lib.tenEstimateVerboseSet(tec, 0)
    failed = failed or lib.tenEstimateMethodSet(tec, method)
    failed = failed or lib.tenEstimateValueMinSet(tec, MINIMUM_SIGNAL_VALUE)
    if method == TEN_METHOD_LLS:
        ctx.recordErrorLogDwi = 1
    elif method == TEN_METHOD_WLS:
        ctx.WLSIterNum = NUMBER_OF_WLS_ITERATIONS
        ctx.recordErrorDwi = 1
    lib.tenEstimateNegEvalShiftSet(tec, int(bool(shift_negative)))
    failed = failed or lib.tenEstimateThresholdSet(tec, 0.0, 1.0)
    failed = failed or lib.tenEstimateUpdate(tec)
    if failed:
        message = _take_biff_message(lib)
        lib.tenEstimateContextNix(tec)
        raise TeemError("Teem estimation setup failed: " + message)
    if ctx.allNum != gradients.count or ctx.estimateB0 != 1:
        lib.tenEstimateContextNix(tec)
        raise TeemError(f"Teem context check failed: allNum={ctx.allNum}, gradients={gradients.count}")
    return tec


def _fit_slab(lib, gradients, method, shift_negative, values: np.ndarray):
    """Fit one C-contiguous block (voxels x gradients). Return (tensors[:, 7], b0, failures)."""
    voxels, count = values.shape
    tec = _new_context(lib, gradients, method, shift_negative)
    ndwi = lib.nrrdNew()
    nten = lib.nrrdNew()
    nb0 = ctypes.c_void_p()
    try:
        as_double = np.ascontiguousarray(values, dtype=np.float64)  # == (double) in[k] in SlicerDMRI
        sizes = (ctypes.c_size_t * 4)(count, voxels, 1, 1)
        if lib.nrrdWrap_nva(ndwi, as_double.ctypes.data, NRRD_TYPE_DOUBLE, 4, sizes):
            raise TeemError("nrrdWrap_nva failed for a DWI slab")
        if not lib.tenEstimate1TensorVolume4D(tec, nten, ctypes.byref(nb0), None, ndwi, NRRD_TYPE_DOUBLE):
            tensors = _copy_nrrd_doubles(nten, voxels * 7).reshape(voxels, 7)
            baseline = _copy_nrrd_doubles(nb0.value, voxels)
            return tensors, baseline, 0
        _take_biff_message(lib)
        return _fit_voxels(lib, tec, as_double)
    finally:
        lib.nrrdNix(ndwi)
        lib.nrrdNuke(nten)
        if nb0.value:
            lib.nrrdNuke(nb0.value)
        lib.tenEstimateContextNix(tec)


def _fit_voxels(lib, tec, as_double: np.ndarray):
    """Per-voxel path for a slab where Teem stopped at a failed voxel."""
    voxels = as_double.shape[0]
    tensors = np.empty((voxels, 7), dtype=np.float64)
    baseline = np.empty(voxels, dtype=np.float64)
    output = (ctypes.c_double * 7)()
    ctx = tec.contents
    failures = 0
    for index in range(voxels):
        row = as_double[index]
        if lib.tenEstimate1TensorSingle_d(tec, output, row.ctypes.data_as(_DOUBLE_P)):
            failures += 1
            _take_biff_message(lib)
        tensors[index] = ctx.ten[:]  # Teem's own state: NaN after a failed estimate
        baseline[index] = ctx.estimatedB0
    return tensors, baseline, failures


def _copy_nrrd_doubles(nrrd_pointer, count: int) -> np.ndarray:
    data = ctypes.cast(nrrd_pointer, ctypes.POINTER(ctypes.c_void_p))[0]  # Nrrd.data is the first field
    if not data:
        raise TeemError("Teem returned an empty output")
    return np.ctypeslib.as_array(ctypes.cast(data, _DOUBLE_P), shape=(count,)).copy()


def estimate_tensors(values: np.ndarray, gradients: np.ndarray, max_b: float, method: str = "WLS",
                     shift_negative: bool = False, threads: int | None = None,
                     slab_voxels: int | None = None):
    """Fit a single tensor per row of ``values`` (voxels x gradients).

    ``gradients`` are the transformed and rescaled gradients (see ``prepare_gradients``).
    Return ``(tensors[voxels, 7] float64, baseline[voxels] float64, failures)``; tensor column 0
    is Teem's confidence value, columns 1-6 are xx, xy, xz, yy, yz, zz.
    """
    if method not in FIT_METHODS:
        raise ValueError(f"Unsupported estimation method {method!r}; use LS or WLS")
    lib = load_teem()
    values = np.ascontiguousarray(values)
    voxels = values.shape[0]
    threads = max(1, int(threads or default_threads()))
    if slab_voxels is None:
        slab_voxels = max(1, min(65536, -(-voxels // (4 * threads))))
    bounds = [(start, min(start + slab_voxels, voxels)) for start in range(0, voxels, slab_voxels)]
    tensors = np.empty((voxels, 7), dtype=np.float64)
    baseline = np.empty(voxels, dtype=np.float64)
    grads = _Gradients(lib, gradients, max_b)
    try:
        def run(bound):
            start, stop = bound
            slab_tensors, slab_baseline, failures = _fit_slab(
                lib, grads, FIT_METHODS[method], shift_negative, values[start:stop])
            tensors[start:stop] = slab_tensors
            baseline[start:stop] = slab_baseline
            return failures

        if threads == 1 or len(bounds) == 1:
            failures = sum(run(bound) for bound in bounds)
        else:
            with ThreadPoolExecutor(max_workers=threads) as pool:
                failures = sum(pool.map(run, bounds))
    finally:
        grads.close()
    return tensors, baseline, failures


def default_threads() -> int:
    requested = os.environ.get("SLICERDSPACE_STAGE_CPUS", "").strip()
    if requested.isdigit() and int(requested) > 0:
        return int(requested)
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


# --------------------------------------------------------------------------- VTK / NRRD layer
def _vtk_modules():
    import vtk
    import vtkTeem
    from vtk.util import numpy_support
    try:
        from MRMLCorePython import vtkMRMLNRRDStorageNode
    except ImportError:
        from slicer import vtkMRMLNRRDStorageNode
    return vtk, vtkTeem, numpy_support, vtkMRMLNRRDStorageNode


def prepare_gradients(reader, vtk, storage_node_class):
    """Gradients in IJK, normalised, rescaled by sqrt(b / max b); as DWIToDTIEstimation does."""
    grads = vtk.vtkDoubleArray()
    bvalues = vtk.vtkDoubleArray()
    if not storage_node_class().ParseDiffusionInformation(reader, grads, bvalues):
        raise TeemError("Error parsing diffusion information")
    measurement_frame = vtk.vtkMatrix4x4()
    measurement_frame.DeepCopy(reader.GetMeasurementFrameMatrix())
    ras_to_ijk_rotation = vtk.vtkMatrix4x4()
    ras_to_ijk_rotation.DeepCopy(reader.GetRasToIjkMatrix())
    for row in range(3):
        ras_to_ijk_rotation.SetElement(row, 3, 0)
    for column in range(3):
        values = [ras_to_ijk_rotation.GetElement(row, column) for row in range(3)]
        if vtk.vtkMath.Normalize(values) != 0.0 and abs(math.fsum(v * v for v in values) - 1.0) > 1e-9:
            raise TeemError("vtkMath.Normalize did not update its argument in place")
        for row in range(3):
            ras_to_ijk_rotation.SetElement(row, column, values[row])
    transform = vtk.vtkTransform()
    transform.PostMultiply()
    transform.SetMatrix(measurement_frame)
    transform.Concatenate(ras_to_ijk_rotation)
    transform.Update()

    count = grads.GetNumberOfTuples()
    max_b = bvalues.GetRange()[1]
    gradients = np.empty((count, 3), dtype=np.float64)
    for index in range(count):
        point = transform.TransformPoint(grads.GetTuple3(index))
        norm = math.sqrt(point[0] * point[0] + point[1] * point[1] + point[2] * point[2])
        if norm > GRADIENT_NORM_EPSILON:
            point = (point[0] / norm, point[1] / norm, point[2] / norm)
        factor = math.sqrt(bvalues.GetValue(index) / max_b)
        gradients[index] = (point[0] * factor, point[1] * factor, point[2] * factor)
    return gradients, max_b, ras_to_ijk_rotation


def _cast_like_c(values: np.ndarray, dtype: np.dtype):
    """``(T) value`` of the SlicerDMRI loop. Integer overflow/NaN (undefined in C) becomes 0."""
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        return values.astype(dtype), 0
    info = np.iinfo(dtype)
    truncated = np.trunc(values)
    valid = np.isfinite(truncated) & (truncated >= info.min) & (truncated <= info.max)
    result = np.zeros(values.shape, dtype=dtype)
    result[valid] = truncated[valid].astype(dtype)
    return result, int((~valid).sum())


def _ijk_to_ras(reader, vtk):
    matrix = vtk.vtkMatrix4x4()
    matrix.DeepCopy(reader.GetRasToIjkMatrix())
    matrix.Invert()
    return matrix


def estimate_file(dwi_path, tensor_path, baseline_path, method="WLS", shift_negative=False,
                  threads=None, log=print) -> dict:
    """DWIToDTIEstimation: read a DWI NRRD, write the tensor NRRD and the baseline NRRD."""
    vtk, vtkTeem, numpy_support, storage_node_class = _vtk_modules()
    threads = max(1, int(threads or default_threads()))
    started = time.monotonic()
    reader = vtkTeem.vtkTeemNRRDReader()
    reader.SetFileName(str(dwi_path))
    reader.Update()
    if reader.GetReadStatus():
        raise TeemError(f"Error reading diffusion file {dwi_path}")
    gradients, max_b, ras_to_ijk_rotation = prepare_gradients(reader, vtk, storage_node_class)
    image = reader.GetOutput()
    scalars = image.GetPointData().GetScalars()
    if scalars is None or scalars.GetNumberOfComponents() != gradients.shape[0]:
        raise TeemError("The input has to have a number of components equal to the number of gradients")
    values = numpy_support.vtk_to_numpy(scalars)
    values = values.reshape(-1, gradients.shape[0])
    read_seconds = time.monotonic() - started

    fit_started = time.monotonic()
    tensors, baseline, failures = estimate_tensors(values, gradients, max_b, method, shift_negative, threads)
    fit_seconds = time.monotonic() - fit_started

    xx, xy, xz, yy, yz, zz = (tensors[:, column] for column in range(1, 7))
    full = np.stack([xx, xy, xz, xy, yy, yz, xz, yz, zz], axis=1).astype(np.float32)
    baseline_cast, cast_out_of_range = _cast_like_c(baseline, values.dtype)
    del tensors, values

    ijk_to_ras = _ijk_to_ras(reader, vtk)
    tensor_image = vtk.vtkImageData()
    tensor_image.CopyStructure(image)
    tensor_array = numpy_support.numpy_to_vtk(full, deep=True, array_type=vtk.VTK_FLOAT)
    tensor_image.GetPointData().SetTensors(tensor_array)
    writer = vtkTeem.vtkTeemNRRDWriter()
    writer.SetInputData(tensor_image)
    writer.SetFileName(str(tensor_path))
    writer.UseCompressionOn()
    writer.SetIJKToRASMatrix(ijk_to_ras)
    ras_to_ijk_rotation.Invert()  # tensor components are in IJK; the frame maps them back
    writer.SetMeasurementFrameMatrix(ras_to_ijk_rotation)
    writer.Write()

    baseline_image = vtk.vtkImageData()
    baseline_image.CopyStructure(image)
    baseline_array = numpy_support.numpy_to_vtk(
        baseline_cast, deep=True, array_type=numpy_support.get_vtk_array_type(baseline_cast.dtype))
    baseline_array.SetName("Baseline")
    baseline_image.GetPointData().SetScalars(baseline_array)
    writer2 = vtkTeem.vtkTeemNRRDWriter()
    writer2.SetInputData(baseline_image)
    writer2.SetFileName(str(baseline_path))
    writer2.UseCompressionOn()
    writer2.SetIJKToRASMatrix(ijk_to_ras)
    writer2.Write()

    lib = load_teem()
    receipt = dict(
        kind="teem_tensor_estimation", method=method, shift_negative_eigenvalues=bool(shift_negative),
        threads=threads, voxels=int(baseline.shape[0]), gradients=int(gradients.shape[0]), max_b=max_b,
        failed_voxels=int(failures), baseline_cast_out_of_range=int(cast_out_of_range),
        teem_version=teem_version(lib), teem_library=getattr(lib, "_slicerdspace_path", ""),
        read_seconds=round(read_seconds, 3), fit_seconds=round(fit_seconds, 3),
        total_seconds=round(time.monotonic() - started, 3),
        reference="SlicerDMRI a40200b DWIToDTIEstimation / vtkTeemEstimateDiffusionTensor")
    if log:
        log(f"[TeemDTI] {method} fit of {receipt['voxels']} voxels x {receipt['gradients']} values with "
            f"{threads} threads in {receipt['fit_seconds']} s (failed voxels: {failures})")
    return receipt


def scalar_files(tensor_path, outputs, threads=None, log=print) -> dict:
    """DiffusionTensorScalarMeasurements for several operations from one read of the tensor."""
    vtk, vtkTeem, _numpy_support, _storage = _vtk_modules()
    threads = max(1, int(threads or default_threads()))
    started = time.monotonic()
    reader = vtkTeem.vtkTeemNRRDReader()
    reader.SetFileName(str(tensor_path))
    reader.Update()
    if reader.GetReadStatus() or reader.GetOutput().GetPointData().GetTensors() is None:
        raise TeemError(f"No tensor data in {tensor_path}")
    for operation, output_path in outputs:
        if operation not in SCALAR_OPERATIONS:
            raise ValueError(f"Operation {operation} not supported")
        maths = vtkTeem.vtkDiffusionTensorMathematics()
        maths.SetInputConnection(0, reader.GetOutputPort())
        getattr(maths, "SetOperationTo" + SCALAR_OPERATIONS[operation])()
        maths.SetNumberOfThreads(threads)
        maths.Update()
        writer = vtkTeem.vtkTeemNRRDWriter()
        writer.SetInputConnection(maths.GetOutputPort())
        writer.SetFileName(str(output_path))
        writer.UseCompressionOn()
        writer.SetIJKToRASMatrix(_ijk_to_ras(reader, vtk))
        writer.Write()
    seconds = round(time.monotonic() - started, 3)
    if log:
        log(f"[TeemDTI] {len(outputs)} scalar maps with {threads} threads in {seconds} s")
    return dict(kind="teem_tensor_scalars", operations=[op for op, _ in outputs], threads=threads, seconds=seconds)


# --------------------------------------------------------------------------- functional self-check
def self_check() -> dict:
    """Load Teem, verify the struct layout, and recover a known tensor from simulated signals."""
    lib = load_teem()
    directions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1],
                           [1, -1, 0], [1, 0, -1], [0, 1, -1], [1, 1, 1], [-1, 1, 1], [1, -1, 1]], float)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    bvalues = np.r_[0.0, np.full(len(directions), 1000.0), np.full(len(directions), 2000.0)]
    unit = np.vstack([np.zeros(3), directions, directions])
    max_b = bvalues.max()
    gradients = unit * np.sqrt(bvalues / max_b)[:, None]
    truth = np.array([[1.7e-3, 0.1e-3, 0.0], [0.1e-3, 0.4e-3, 0.05e-3], [0.0, 0.05e-3, 0.3e-3]])
    b0 = 1000.0
    signal = b0 * np.exp(-bvalues * np.einsum("ni,ij,nj->n", unit, truth, unit))
    values = np.vstack([signal, signal * 0.5]).astype(np.float32)
    report = dict(teem_version=teem_version(lib), library=getattr(lib, "_slicerdspace_path", ""))
    for method in ("LS", "WLS"):
        tensors, baseline, failures = estimate_tensors(values, gradients, max_b, method, threads=2, slab_voxels=1)
        expected = np.array([truth[0, 0], truth[0, 1], truth[0, 2], truth[1, 1], truth[1, 2], truth[2, 2]])
        error = float(np.max(np.abs(tensors[:, 1:] - expected)))
        if failures or error > 1e-6 or abs(baseline[0] - b0) > 1e-2 or abs(baseline[1] - b0 / 2) > 1e-2:
            raise TeemError(f"Teem {method} self-check failed: tensor error {error}, baseline {baseline}")
        report[method] = dict(max_tensor_error=error, baseline=baseline.tolist())
    report["status"] = "ok"
    return report


# --------------------------------------------------------------------------- command line
def _parse_outputs(items):
    outputs = []
    for item in items or []:
        operation, separator, path = item.partition("=")
        if not separator or not path:
            raise SystemExit(f"Expected OPERATION=PATH, got {item!r}")
        outputs.append((operation, path))
    return outputs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    estimate = commands.add_parser("estimate", help="DWIToDTIEstimation replacement")
    estimate.add_argument("--enumeration", choices=sorted(FIT_METHODS), default="LS")
    estimate.add_argument("--shiftNeg", action="store_true")
    estimate.add_argument("--threads", type=int, default=None)
    estimate.add_argument("--scalar", action="append", metavar="OPERATION=PATH")
    estimate.add_argument("--receipt", default=None)
    estimate.add_argument("input_volume")
    estimate.add_argument("output_tensor")
    estimate.add_argument("output_baseline")
    scalars = commands.add_parser("scalars", help="DiffusionTensorScalarMeasurements replacement")
    scalars.add_argument("--threads", type=int, default=None)
    scalars.add_argument("input_tensor")
    scalars.add_argument("outputs", nargs="+", metavar="OPERATION=PATH")
    commands.add_parser("selfcheck", help="functional self-check of the Teem binding")
    args = parser.parse_args(argv)
    try:
        if args.command == "selfcheck":
            print(json.dumps(self_check(), indent=2))
            return 0
        if args.command == "scalars":
            scalar_files(args.input_tensor, _parse_outputs(args.outputs), args.threads)
            return 0
        receipt = estimate_file(args.input_volume, args.output_tensor, args.output_baseline,
                                args.enumeration, args.shiftNeg, args.threads)
        outputs = _parse_outputs(args.scalar)
        if outputs:
            receipt["scalars"] = scalar_files(args.output_tensor, outputs, args.threads)
        if args.receipt:
            Path(args.receipt).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        return 0
    except TeemError as error:
        print(f"[TeemDTI] ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
