"""Compare NRRD outputs voxel by voxel (reference SlicerDMRI CLI vs shared/teem_dti.py).

Usage: PythonSlicer compare_nrrd.py <report.json> <ref.nrrd>::<new.nrrd> [...]
"""
import gzip, hashlib, json, sys
import numpy as np
import vtk, vtkTeem
from vtk.util import numpy_support

def read(path):
    r = vtkTeem.vtkTeemNRRDReader(); r.SetFileName(path); r.Update()
    pd = r.GetOutput().GetPointData()
    arr = pd.GetTensors() if pd.GetTensors() is not None else pd.GetScalars()
    m1 = [r.GetRasToIjkMatrix().GetElement(i, j) for i in range(4) for j in range(4)]
    mf = r.GetMeasurementFrameMatrix()
    m2 = [mf.GetElement(i, j) for i in range(4) for j in range(4)] if mf else None
    return numpy_support.vtk_to_numpy(arr).copy(), m1, m2

def split(path):
    raw = open(path, "rb").read()
    cut = raw.index(b"\n\n") + 2
    header, payload = raw[:cut], raw[cut:]
    try:
        data = gzip.decompress(payload)
    except Exception:
        data = payload
    return header.decode(errors="replace"), hashlib.sha256(raw).hexdigest(), hashlib.sha256(data).hexdigest()

def ulp(a, b):
    if a.dtype == np.float32:
        ia = a.view(np.int32).astype(np.int64); ib = b.view(np.int32).astype(np.int64)
        ia = np.where(ia < 0, -(2**31) - ia, ia); ib = np.where(ib < 0, -(2**31) - ib, ib)
        return np.abs(ia - ib)
    if a.dtype == np.float64:
        ia = a.view(np.int64).astype(object); ib = b.view(np.int64).astype(object)
        return None
    return np.abs(a.astype(np.int64) - b.astype(np.int64))

def compare(ref, new):
    a, ma, fa = read(ref); b, mb, fb = read(new)
    out = dict(reference=ref, candidate=new, shape_ref=list(a.shape), shape_new=list(b.shape),
               dtype_ref=str(a.dtype), dtype_new=str(b.dtype))
    ha, sa, pa = split(ref); hb, sb, pb = split(new)
    out.update(file_sha256_equal=sa == sb, payload_sha256_equal=pa == pb,
               header_equal=ha == hb, ras_to_ijk_equal=ma == mb, measurement_frame_equal=fa == fb)
    if ha != hb:
        out["header_diff"] = [l for l in set(ha.splitlines()) ^ set(hb.splitlines())]
    if a.shape != b.shape or a.dtype != b.dtype:
        return out
    nan_a, nan_b = np.isnan(a) if a.dtype.kind == "f" else np.zeros(a.shape, bool), None
    nan_b = np.isnan(b) if b.dtype.kind == "f" else np.zeros(b.shape, bool)
    equal = (a == b) | (nan_a & nan_b)
    out.update(values=int(a.size), equal_values=int(equal.sum()), nan_ref=int(nan_a.sum()), nan_new=int(nan_b.sum()),
               nan_mask_equal=bool((nan_a == nan_b).all()))
    finite = ~(nan_a | nan_b)
    if finite.any():
        diff = np.abs(a[finite].astype(np.float64) - b[finite].astype(np.float64))
        out["max_abs_diff"] = float(diff.max())
        scale = np.abs(a[finite].astype(np.float64))
        out["max_rel_diff"] = float((diff / np.maximum(scale, 1e-30)).max())
        u = ulp(a[finite], b[finite])
        if u is not None:
            out["max_ulp"] = int(u.max()); out["values_over_1_ulp"] = int((u > 1).sum())
        voxels = equal.reshape(a.shape[0], -1).all(axis=1) if a.ndim > 1 else equal
        out["voxels"] = int(voxels.size); out["voxels_identical"] = int(voxels.sum())
    return out

pairs = [arg.split("::") for arg in sys.argv[2:]]
report = [compare(r, n) for r, n in pairs]
open(sys.argv[1], "w").write(json.dumps(report, indent=1) + "\n")
for item in report:
    print(json.dumps({k: v for k, v in item.items() if k not in ("reference",)}, separators=(",", ":"))[:900])
