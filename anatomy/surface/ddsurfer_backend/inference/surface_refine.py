"""Native-space diffusion-guided cortical surface refinement (pure 5-DTI, CPU)."""
from __future__ import annotations

import ctypes
import hashlib
import logging
import os
import platform
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_fill_holes, gaussian_filter, map_coordinates

LOG = logging.getLogger(__name__)
MODULE_DIR = Path(__file__).resolve().parent
CPP_SOURCE = MODULE_DIR / "surface_refine.cpp"
BUNDLED_BIN_DIR = MODULE_DIR / "bin"
MEDIAL_MASKS_PATH = MODULE_DIR / "assets" / "medial_masks.npz"

_LIB = None
_LIB_FAILED = False
_LIB_SOURCE = "unavailable"
_DLL_DIR_HANDLES = []


def _normalized_plat() -> str:
    sys_name = platform.system().lower()
    mach = platform.machine().lower()
    arch = {"amd64": "x86_64", "x64": "x86_64", "aarch64": "arm64"}.get(mach, mach)
    return f"{sys_name}_{arch}"


def refinement_library_source() -> str:
    """Return where the loaded native refinement engine came from ('bundled', 'compiled', or 'unavailable')."""
    return _LIB_SOURCE


def _bundled_lib_candidates() -> list[Path]:
    sys_name = platform.system().lower()
    plat = _normalized_plat()
    suf = _shared_lib_suffix()
    names = []
    if sys_name == "darwin":
        names.append(f"libsurface_refine_darwin_universal2{suf}")
    names.append(f"libsurface_refine_{plat}{suf}")
    if sys_name == "windows":
        names.append(f"surface_refine_{plat}{suf}")
    return [BUNDLED_BIN_DIR / name for name in names]


def _load_cdll(lib_path: Path):
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        try:
            _DLL_DIR_HANDLES.append(os.add_dll_directory(str(lib_path.parent)))
        except OSError:
            pass
    return ctypes.CDLL(str(lib_path))


def _shared_lib_suffix() -> str:
    sys_name = platform.system().lower()
    if sys_name == "windows":
        return ".dll"
    if sys_name == "darwin":
        return ".dylib"
    return ".so"


def _cache_dir() -> Path:
    base = os.environ.get("DDSURFER_REFINE_CACHE") or os.environ.get("XDG_CACHE_HOME")
    if base:
        candidate = Path(base) / "ddsurfer"
    else:
        candidate = Path.home() / ".cache" / "ddsurfer"
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        probe = candidate / f".probe_{os.getpid()}"
        probe.write_bytes(b"")
        probe.unlink()
        return candidate
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "ddsurfer_refine_cache"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def _find_cxx_compiler() -> Optional[str]:
    candidates = []
    if os.environ.get("CXX"):
        candidates.append(os.environ["CXX"])
    candidates.extend(["g++", "clang++", "c++"])
    for cand in candidates:
        resolved = shutil.which(cand)
        if resolved:
            return resolved
    return None


def _build_or_load_lib():
    global _LIB, _LIB_FAILED, _LIB_SOURCE
    if _LIB is not None:
        return _LIB
    if _LIB_FAILED:
        return None

    lib = None
    for cand in _bundled_lib_candidates():
        if cand.is_file() and cand.stat().st_size > 0:
            try:
                lib = _load_cdll(cand)
                _LIB_SOURCE = "bundled"
                LOG.info("Loaded bundled native surface refinement library: %s", cand)
                break
            except OSError as exc:
                LOG.warning("Bundled native surface refinement library %s could not be loaded (%s); falling back to local compile.", cand, exc)
                lib = None

    if lib is None:
        if not CPP_SOURCE.is_file():
            _LIB_FAILED = True
            return None

        src_hash = hashlib.sha256(CPP_SOURCE.read_bytes()).hexdigest()[:12]
        plat = _normalized_plat()
        cache = _cache_dir()
        lib_path = cache / f"libsurface_refine_{src_hash}_{plat}{_shared_lib_suffix()}"

        if not lib_path.is_file() or not lib_path.stat().st_size:
            cxx = _find_cxx_compiler()
            if cxx is None:
                LOG.warning("No precompiled library or C++17 compiler found; skipping native surface refinement.")
                _LIB_FAILED = True
                return None
            tmp_lib = cache / f".libsurface_refine_{src_hash}_{os.getpid()}{_shared_lib_suffix()}"
            is_win = platform.system().lower() == "windows"
            pic_flags = [] if is_win else ["-fPIC"]
            flag_sets = [
                ["-O3", "-march=native", "-fno-math-errno", "-funroll-loops", "-std=c++17", "-fopenmp", "-shared"] + pic_flags,
                ["-O3", "-fno-math-errno", "-funroll-loops", "-std=c++17", "-fopenmp", "-shared"] + pic_flags,
                ["-O3", "-fno-math-errno", "-std=c++17", "-shared"] + pic_flags,
            ]
            built = False
            last_err = ""
            try:
                for flags in flag_sets:
                    cmd = [cxx, *flags, str(CPP_SOURCE), "-o", str(tmp_lib)]
                    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    if proc.returncode == 0 and tmp_lib.is_file() and tmp_lib.stat().st_size > 0:
                        os.replace(tmp_lib, lib_path)
                        built = True
                        break
                    last_err = proc.stdout.strip()
            finally:
                if tmp_lib.exists():
                    try:
                        tmp_lib.unlink()
                    except OSError:
                        pass
            if not built:
                LOG.warning("Failed to compile native surface refinement library; skipping refinement. %s", last_err[-300:])
                _LIB_FAILED = True
                return None

        try:
            lib = _load_cdll(lib_path)
            _LIB_SOURCE = "compiled"
        except OSError as exc:
            LOG.warning("Failed to load native surface refinement library (%s); skipping refinement.", exc)
            _LIB_FAILED = True
            return None

    c_dp = ctypes.POINTER(ctypes.c_double)
    c_fp = ctypes.POINTER(ctypes.c_float)
    c_ip = ctypes.POINTER(ctypes.c_int)
    c_u8p = ctypes.POINTER(ctypes.c_uint8)
    c_i = ctypes.c_int
    c_d = ctypes.c_double

    lib.refine_dl_surface_physics_flow.argtypes = [
        c_dp, c_i, c_ip, c_i, c_fp,
        c_fp, c_i, c_i, c_i, c_dp,
        c_fp, c_i, c_i, c_i, c_dp,
        c_fp, c_i, c_i, c_i, c_dp,
        c_fp, c_i, c_i, c_i, c_dp,
        c_fp, c_i, c_i, c_i, c_dp,
        c_i, c_i,
        c_d, c_d,
        c_d, c_d,
        c_d, c_d, c_d, c_d,
        c_d, c_d, c_d,
        c_d, c_d, c_d,
        c_d, c_d, c_d, c_i,
        c_i,
    ]
    lib.refine_dl_surface_physics_flow.restype = None

    lib.refine_dl_pial_physics_flow.argtypes = [
        c_dp, c_dp, c_i, c_ip, c_i, c_fp,
        c_fp, c_i, c_i, c_i, c_dp,
        c_fp, c_i, c_i, c_i, c_dp,
        c_fp, c_i, c_i, c_i, c_dp,
        c_fp, c_i, c_i, c_i, c_dp,
        c_i, c_i,
        c_d, c_d,
        c_d, c_d,
        c_d, c_d,
        c_d, c_d, c_d,
        c_d, c_d, c_d,
        c_d, c_d, c_d,
        c_d, c_d, c_d, c_d,
        c_i, c_i,
    ]
    lib.refine_dl_pial_physics_flow.restype = None

    lib.count_self_intersections_tdf.argtypes = [c_dp, c_i, c_ip, c_i, c_i]
    lib.count_self_intersections_tdf.restype = c_i

    lib.fast_affine_resample_tdf.argtypes = [
        c_fp, c_i, c_i, c_i,
        c_fp, c_i, c_i, c_i,
        c_dp, c_i, c_i, c_i,
    ]
    lib.fast_affine_resample_tdf.restype = None

    lib.fast_gaussian_3d_tdf.argtypes = [c_fp, c_fp, c_i, c_i, c_i, c_d, c_i]
    lib.fast_gaussian_3d_tdf.restype = None

    lib.compute_3d_sheetness_tdf.argtypes = [c_fp, c_i, c_i, c_i, c_dp, c_i, c_fp, c_fp, c_i]
    lib.compute_3d_sheetness_tdf.restype = None

    lib.fast_edt_3d_tdf.argtypes = [c_u8p, c_i, c_i, c_i, c_fp, c_fp, c_fp, c_i]
    lib.fast_edt_3d_tdf.restype = None

    lib.compute_vertex_normals_tdf.argtypes = [c_dp, c_i, c_ip, c_i, c_fp, c_i]
    lib.compute_vertex_normals_tdf.restype = None

    lib.fast_gmm_2c_tdf.argtypes = [
        ctypes.POINTER(c_fp), c_i, c_i, c_i, c_i,
        c_u8p, c_ip, c_i,
        c_fp, c_d,
        c_fp, c_fp, c_fp, c_i,
    ]
    lib.fast_gmm_2c_tdf.restype = c_i

    _LIB = lib
    return _LIB


def _prep_f32(arr, affine):
    if arr is None:
        return None, 0, 0, 0, None, None
    a = np.ascontiguousarray(arr, dtype=np.float32)
    M = np.ascontiguousarray(np.linalg.inv(np.asarray(affine, dtype=np.float64)).ravel(), dtype=np.float64)
    return (
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(a.shape[0]), int(a.shape[1]), int(a.shape[2]),
        M.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        (a, M),
    )


def _fast_affine_resample(lib, src, src_aff, dst_shape, dst_aff, order=3, clamp_oob=False, nthreads=8):
    a = np.ascontiguousarray(src, dtype=np.float32)
    dst = np.empty(tuple(int(x) for x in dst_shape), dtype=np.float32)
    M = np.ascontiguousarray(
        (np.linalg.inv(np.asarray(src_aff, dtype=np.float64)) @ np.asarray(dst_aff, dtype=np.float64)).ravel(),
        dtype=np.float64,
    )
    lib.fast_affine_resample_tdf(
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), a.shape[0], a.shape[1], a.shape[2],
        dst.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), dst.shape[0], dst.shape[1], dst.shape[2],
        M.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        int(order), int(1 if clamp_oob else 0), int(nthreads),
    )
    return dst


def _fast_zoom_linear(lib, src, dst_shape, scale, nthreads=8):
    a = np.ascontiguousarray(src, dtype=np.float32)
    dst = np.empty(tuple(int(x) for x in dst_shape), dtype=np.float32)
    M = np.zeros(16, dtype=np.float64)
    M[0] = M[5] = M[10] = float(scale)
    M[15] = 1.0
    lib.fast_affine_resample_tdf(
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), a.shape[0], a.shape[1], a.shape[2],
        dst.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), dst.shape[0], dst.shape[1], dst.shape[2],
        M.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        1, 1, int(nthreads),
    )
    return dst


def _fast_gaussian_3d(lib, vol, sigma, nthreads=8):
    a = np.ascontiguousarray(vol, dtype=np.float32)
    dst = np.empty_like(a)
    lib.fast_gaussian_3d_tdf(
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        dst.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        a.shape[0], a.shape[1], a.shape[2], float(sigma), int(nthreads),
    )
    return dst


def _fast_3d_sheetness(lib, vol, sigmas=(0.95, 1.55), nthreads=8):
    a = np.ascontiguousarray(vol, dtype=np.float32)
    sigs = np.ascontiguousarray(sigmas, dtype=np.float64)
    r_out = np.empty_like(a)
    v_out = np.empty_like(a)
    lib.compute_3d_sheetness_tdf(
        a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), a.shape[0], a.shape[1], a.shape[2],
        sigs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(sigs),
        r_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        v_out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(nthreads),
    )
    return r_out, v_out


def _fast_edt_3d(lib, mask_bool, nn_src=None, need_dist=True, nthreads=8):
    m = np.ascontiguousarray(mask_bool, dtype=np.uint8)
    dist = np.empty(m.shape, dtype=np.float32) if need_dist else None
    dist_p = dist.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) if dist is not None else None
    if nn_src is not None:
        ns = np.ascontiguousarray(nn_src, dtype=np.float32)
        nd = np.empty_like(ns)
        ns_p = ns.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        nd_p = nd.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    else:
        ns = nd = ns_p = nd_p = None
    lib.fast_edt_3d_tdf(
        m.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), m.shape[0], m.shape[1], m.shape[2],
        dist_p, ns_p, nd_p, int(nthreads),
    )
    if nn_src is not None and need_dist:
        return dist, nd
    if nn_src is not None:
        return nd
    return dist


def _fast_vertex_normals(lib, verts, faces, nthreads=8):
    v = np.ascontiguousarray(verts, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.int32)
    nrm = np.empty((len(v), 3), dtype=np.float32)
    lib.compute_vertex_normals_tdf(
        v.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(v),
        f.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), len(f),
        nrm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), int(nthreads),
    )
    return nrm


def _fast_gmm_2c(lib, feats_list, mask, csf_prob, fa, md, mrf_sigma=0.25, n_fit=24000, nthreads=8):
    D = len(feats_list)
    paren = mask & (csf_prob < 0.30) & (fa > 0.04) & (md > 0.35) & (md < 1.25)
    if int(paren.sum()) < 500:
        paren = mask & (csf_prob < 0.45) & (md < 1.6)
    p_idx = np.flatnonzero(paren)
    if len(p_idx) < 32:
        raise ValueError("Insufficient parenchyma voxels for GMM fitting")
    rng = np.random.default_rng(42)
    sub_idx = (rng.choice(p_idx, size=n_fit, replace=False) if len(p_idx) > n_fit else p_idx).astype(np.int32)

    arrs = [np.ascontiguousarray(f, dtype=np.float32) for f in feats_list]
    ptrs = (ctypes.POINTER(ctypes.c_float) * D)(*[a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) for a in arrs])
    m_u8 = np.ascontiguousarray(mask, dtype=np.uint8)
    csf_f = np.ascontiguousarray(csf_prob, dtype=np.float32)
    pwm = np.empty(mask.shape, dtype=np.float32)
    pgm = np.empty(mask.shape, dtype=np.float32)
    pcsf = np.empty(mask.shape, dtype=np.float32)

    rc = lib.fast_gmm_2c_tdf(
        ptrs, D, mask.shape[0], mask.shape[1], mask.shape[2],
        m_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        sub_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), len(sub_idx),
        csf_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), float(mrf_sigma),
        pwm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        pgm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        pcsf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(nthreads),
    )
    if rc != 0:
        raise RuntimeError(f"fast_gmm_2c_tdf failed with code {rc}")
    return pwm, pgm, pcsf


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1.0 - 1e-4)
    return np.log(p / (1.0 - p))


def _load_volume_ras(path: Path) -> Tuple[np.ndarray, np.ndarray, float]:
    """Load 3D scalar volume as float32 + scanner-RAS 4x4 affine + mean voxel spacing (mm)."""
    path = Path(path)
    low = path.name.lower()
    if low.endswith((".nii", ".nii.gz")):
        img = nib.load(str(path))
        arr = np.nan_to_num(img.get_fdata(dtype=np.float32).squeeze(), nan=0.0).astype(np.float32, copy=False)
        aff = np.asarray(img.affine, dtype=np.float64)
        dx = float(np.mean(img.header.get_zooms()[:3]))
        return arr, aff, dx
    import SimpleITK as sitk
    img = sitk.ReadImage(str(path))
    arr = np.nan_to_num(sitk.GetArrayFromImage(img).transpose(2, 1, 0), nan=0.0).astype(np.float32, copy=False)
    aff = np.eye(4, dtype=np.float64)
    aff[:3, :3] = np.asarray(img.GetDirection(), dtype=np.float64).reshape(3, 3) @ np.diag(img.GetSpacing())
    aff[:3, 3] = np.asarray(img.GetOrigin(), dtype=np.float64)
    aff = np.diag([-1.0, -1.0, 1.0, 1.0]) @ aff
    dx = float(np.mean(img.GetSpacing()[:3]))
    return arr, aff, dx


def _native_crop_bounds(lo_ras, hi_ras, nat_aff, nat_shape, pad_vox=5):
    corners = np.array(
        [[x, y, z] for x in (lo_ras[0], hi_ras[0]) for y in (lo_ras[1], hi_ras[1]) for z in (lo_ras[2], hi_ras[2])],
        dtype=np.float64,
    )
    inv_nat = np.linalg.inv(nat_aff)
    ijk = (inv_nat[:3, :3] @ corners.T + inv_nat[:3, 3:4]).T
    nlo = np.maximum(0, np.floor(ijk.min(axis=0)).astype(int) - pad_vox)
    nhi = np.minimum(np.array(nat_shape, dtype=int), np.ceil(ijk.max(axis=0)).astype(int) + pad_vox + 1)
    c_nat_aff = nat_aff.copy()
    c_nat_aff[:3, 3] = (nat_aff @ np.array([nlo[0], nlo[1], nlo[2], 1.0]))[:3]
    sl = (slice(int(nlo[0]), int(nhi[0])), slice(int(nlo[1]), int(nhi[1])), slice(int(nlo[2]), int(nhi[2])))
    return sl, c_nat_aff


def _local_norm_in_mask(lib, vol, mask, par_mask, sigma_mm=10.0, nthreads=8):
    w = par_mask.astype(np.float32)
    v2 = np.ascontiguousarray((vol * w)[::2, ::2, ::2], dtype=np.float32)
    w2 = np.ascontiguousarray(w[::2, ::2, ::2], dtype=np.float32)
    num2 = _fast_gaussian_3d(lib, v2, sigma=sigma_mm * 0.5, nthreads=nthreads)
    den2 = _fast_gaussian_3d(lib, w2, sigma=sigma_mm * 0.5, nthreads=nthreads)
    loc2 = (num2 / np.maximum(den2, 1e-4)).astype(np.float32)
    loc = _fast_zoom_linear(lib, loc2, vol.shape, 0.5, nthreads=nthreads)
    return np.where(mask & (loc > 1e-4), vol / np.maximum(loc, 1e-4), 0.0).astype(np.float32)


def _get_medial_mask(lib, verts: np.ndarray, faces: np.ndarray, hemi: str, nthreads: int = 8) -> np.ndarray:
    if MEDIAL_MASKS_PATH.is_file():
        with np.load(MEDIAL_MASKS_PATH) as data:
            n_expected = int(data[f"{hemi}_len"][0])
            if n_expected == len(verts):
                bits = np.unpackbits(data[f"{hemi}_bits"])[:n_expected]
                return bits.astype(np.float32)
    nrm = _fast_vertex_normals(lib, verts, faces, nthreads=nthreads)
    x_min, x_max = verts[:, 0].min(), verts[:, 0].max()
    y_min, y_max = verts[:, 1].min(), verts[:, 1].max()
    z_min, z_max = verts[:, 2].min(), verts[:, 2].max()
    if hemi == "lh":
        near_med = (verts[:, 0] > x_max - 15.0) & (nrm[:, 0] > 0.35)
    else:
        near_med = (verts[:, 0] < x_min + 15.0) & (nrm[:, 0] < -0.35)
    in_core = (
        (verts[:, 1] > y_min + 0.24 * (y_max - y_min))
        & (verts[:, 1] < y_max - 0.24 * (y_max - y_min))
        & (verts[:, 2] > z_min + 0.20 * (z_max - z_min))
        & (verts[:, 2] < z_max - 0.32 * (z_max - z_min))
    )
    return (near_med & in_core).astype(np.float32)


def _preload_native_dti(dti_paths: Dict[str, Path]) -> Dict[str, Any]:
    keys = ["mask", "fa", "md", "l1", "l2", "l3"]
    with ThreadPoolExecutor(max_workers=len(keys)) as tex:
        loaded = dict(tex.map(lambda k: (k, _load_volume_ras(dti_paths[k])), keys))

    mask_raw, nat_aff, dx = loaded["mask"]
    mask_nat = mask_raw > 0.5
    fa_nat, fa_aff, _ = loaded["fa"]
    if mask_nat.shape != fa_nat.shape or not np.allclose(nat_aff, fa_aff, atol=1e-2):
        raise ValueError("Native brain mask geometry does not match native DTI maps")

    md_nat = loaded["md"][0]
    md_pos = md_nat[mask_nat & (md_nat > 0)]
    md_scale = 1000.0 if (len(md_pos) > 0 and float(np.median(md_pos)) < 0.05) else 1.0
    if md_scale != 1.0:
        md_nat = (md_nat * md_scale).astype(np.float32)
        l1_raw = (loaded["l1"][0] * md_scale).astype(np.float32)
        l2_raw = (loaded["l2"][0] * md_scale).astype(np.float32)
        l3_raw = (loaded["l3"][0] * md_scale).astype(np.float32)
    else:
        l1_raw, l2_raw, l3_raw = loaded["l1"][0], loaded["l2"][0], loaded["l3"][0]

    return dict(
        dx=dx, nat_aff=nat_aff, mask_nat=mask_nat,
        fa_nat=fa_nat, md_nat=md_nat, l1_raw=l1_raw, l2_raw=l2_raw, l3_raw=l3_raw,
    )


def _build_hemi_physics(lib, vw_ras, vp_ras, nat, nthreads=8, margin_mm=10.0) -> Dict[str, Any]:
    nat_aff = nat["nat_aff"]
    mask_nat = nat["mask_nat"]
    if vp_ras is not None:
        lo = np.floor(np.minimum(vw_ras.min(axis=0), vp_ras.min(axis=0)) - margin_mm)
        hi = np.ceil(np.maximum(vw_ras.max(axis=0), vp_ras.max(axis=0)) + margin_mm)
    else:
        lo = np.floor(vw_ras.min(axis=0) - margin_mm)
        hi = np.ceil(vw_ras.max(axis=0) + margin_mm)
    c_shape = tuple((hi - lo + 1).astype(int).tolist())
    c_aff = np.eye(4, dtype=np.float64)
    c_aff[:3, 3] = lo

    nsl, c_nat_aff = _native_crop_bounds(lo, hi, nat_aff, mask_nat.shape, pad_vox=5)
    mask_1mm = binary_fill_holes(
        _fast_affine_resample(lib, mask_nat[nsl].astype(np.float32), c_nat_aff, c_shape, c_aff, order=1, nthreads=nthreads) > 0.5
    )
    dist_outer_mm = _fast_edt_3d(lib, mask_1mm, nthreads=nthreads)

    up = {}
    for k, arr, vmin, vmax in (
        ("fa", nat["fa_nat"], 0.0, 1.0),
        ("md", nat["md_nat"], 0.15, 3.5),
        ("l1", nat["l1_raw"], 0.05, 3.5),
        ("l2", nat["l2_raw"], 0.05, 3.5),
        ("l3", nat["l3_raw"], 0.05, 3.5),
    ):
        v = _fast_affine_resample(lib, arr[nsl], c_nat_aff, c_shape, c_aff, order=3, nthreads=nthreads)
        if vmin > 0.0:
            up[k] = (np.clip(v, vmin, vmax) * mask_1mm).astype(np.float32, copy=False)
        else:
            up[k] = np.clip(v * mask_1mm, vmin, vmax).astype(np.float32, copy=False)

    fa_1mm, md_1mm = up["fa"], up["md"]
    l1_1mm, l2_1mm, l3_1mm = up["l1"], up["l2"], up["l3"]
    cs_inv_1mm = np.clip(1.0 - 3.0 * l3_1mm / np.maximum(l1_1mm + l2_1mm + l3_1mm, 0.15), 0.0, 1.0).astype(np.float32)

    p_csf_md1 = np.clip((md_1mm - 1.25) / 1.35, 0.0, 1.0) * mask_1mm
    csf_phys_1mm = np.clip(0.65 * p_csf_md1 + 0.35 * np.clip((md_1mm - 1.20) / 1.40, 0.0, 1.0), 0.0, 1.0) * mask_1mm
    fa_fwe_1mm = (fa_1mm * np.clip((0.40 - csf_phys_1mm) / 0.25, 0.0, 1.0)).astype(np.float32)

    par0 = mask_1mm & (md_1mm > 0.35) & (md_1mm < 1.25)
    d_t = float(np.clip(np.median(md_1mm[par0]), 0.62, 0.92)) if int(par0.sum()) > 500 else 0.78

    wm_edge_ok = (
        np.clip((fa_1mm - 0.18) / 0.07, 0.0, 1.0)
        * np.clip(((d_t + 0.16) - md_1mm) / 0.14, 0.0, 1.0)
        * np.clip((0.14 - csf_phys_1mm) / 0.08, 0.0, 1.0)
    )
    rim_atten = np.maximum(np.clip((dist_outer_mm - 0.40) / 0.65, 0.0, 1.0), 0.92 * wm_edge_ok).astype(np.float32)

    par1 = mask_1mm & (csf_phys_1mm < 0.20) & (md_1mm > 0.40) & (md_1mm < 1.15)
    fa_loc1 = np.clip(_local_norm_in_mask(lib, fa_fwe_1mm, mask_1mm, par1, 11.0, nthreads), 0.15, 2.80) * mask_1mm
    l3_loc1 = np.clip(_local_norm_in_mask(lib, l3_1mm, mask_1mm, par1, 11.0, nthreads), 0.25, 2.20) * mask_1mm

    def _sr_field(feats_1mm):
        pw, pg, pc = _fast_gmm_2c(lib, feats_1mm, mask_1mm, csf_phys_1mm, fa_fwe_1mm, md_1mm, 0.25, nthreads=nthreads)
        ps = pw + pg + pc
        vl = mask_1mm & (ps > 1e-4)
        pw[vl] /= ps[vl]; pg[vl] /= ps[vl]; pc[vl] /= ps[vl]
        pg = np.clip(pg + pw * (1.0 - rim_atten), 0, 1); pw = pw * rim_atten
        ps2 = pw + pg + pc + 1e-6
        pw = (pw / ps2) * mask_1mm; pg = (pg / ps2) * mask_1mm; pc = (pc / ps2) * mask_1mm
        pd = pw + pg
        l1 = _logit(pw / (pd + 1e-6)).astype(np.float32)
        pn = mask_1mm & (pc < 0.20) & (pd > 0.45) & (md_1mm > 0.45) & (md_1mm < 1.15) & np.isfinite(l1)
        bgm = pn & (fa_fwe_1mm >= 0.11) & (fa_fwe_1mm <= 0.19)
        bwm = pn & (fa_fwe_1mm >= 0.35) & (fa_fwe_1mm <= 0.48)
        sl_raw = float(np.median(l1[bwm]) - np.median(l1[bgm])) if (int(bwm.sum()) >= 150 and int(bgm.sum()) >= 150) else 3.5
        sl = float(np.clip(sl_raw, 2.0, 8.0))
        f50 = float(np.median(fa_fwe_1mm[pn])) if int(pn.sum()) > 800 else 0.23
        tb = pn & (np.abs(fa_fwe_1mm - f50) < 0.035)
        l0i = float(np.median(l1[tb])) if int(tb.sum()) >= 150 else 0.0
        wm_frac = float(((l1 > l0i) & pn).sum() / max(int(pn.sum()), 1))
        ok = (sl_raw >= 2.2) and (0.22 <= wm_frac <= 0.66)
        return l1, sl, l0i, pd, pc, ok

    log_fa = np.log(np.maximum(fa_fwe_1mm, 0.0) + 0.05).astype(np.float32)
    l_v74, sl_v74, l0i_v74, par_dens, p_csf, _ = _sr_field([log_fa])
    l_eigs, sl_eigs, l0i_eigs, _, _, eigs_ok = _sr_field([log_fa, fa_loc1, cs_inv_1mm, l3_loc1])

    z_v74_n = np.clip((l_v74 - l0i_v74) / sl_v74, -3.5, 3.5).astype(np.float32)
    z_eigs_n = np.clip((l_eigs - l0i_eigs) / sl_eigs, -3.5, 3.5).astype(np.float32)
    l_blend = (2.0 * z_v74_n + 2.0 * z_eigs_n).astype(np.float32) if eigs_ok else (4.0 * z_v74_n).astype(np.float32)

    c_base = p_csf.copy()
    low_par_pen = 3.0 * np.clip((0.40 - par_dens) / 0.30, 0.0, 1.0)
    pial_pen = 3.5 * np.clip((c_base - 0.24) / 0.32, 0.0, 1.0)
    outer_pen = 2.0 * np.clip((0.90 - dist_outer_mm) / 0.60, 0.0, 1.0) * np.clip((md_1mm - (d_t + 0.12)) / 0.30, 0.0, 1.0)
    low_fa_outer = 1.5 * np.clip((0.10 - fa_1mm) / 0.04, 0, 1) * np.clip((1.5 - dist_outer_mm) / 0.9, 0, 1)
    pen_tot = (low_par_pen + pial_pen + outer_pen + low_fa_outer).astype(np.float32)

    fa_fwe_c = (fa_1mm * np.clip((0.40 - c_base) / 0.25, 0.0, 1.0)).astype(np.float32)
    r_fa, v_fa = _fast_3d_sheetness(lib, fa_fwe_c, sigmas=(0.95, 1.55), nthreads=nthreads)
    r_l3, v_l3 = _fast_3d_sheetness(lib, -l3_1mm, sigmas=(0.95, 1.55), nthreads=nthreads)
    s_fa = np.clip((r_fa - v_fa) / (float(np.percentile(r_fa[mask_1mm], 95)) + 1e-4), -1.6, 1.6)
    s_l3 = np.clip((r_l3 - v_l3) / (float(np.percentile(r_l3[mask_1mm], 95)) + 1e-4), -1.6, 1.6)
    sheet_joint = (0.55 * s_fa + 0.45 * s_l3).astype(np.float32)

    return dict(
        dx=nat["dx"], d_t=d_t, l_blend=l_blend, sl_wm=4.0, pen_tot=pen_tot,
        c_base=c_base.astype(np.float32), fa=fa_1mm.astype(np.float32),
        md_raw=md_1mm.astype(np.float32), fa_fwe=fa_fwe_c, sheet_joint=sheet_joint,
        excl=(~mask_1mm).astype(np.float32), mask_1mm=mask_1mm, grid_aff=c_aff,
    )


def _self_cal_white(lib, v_dl, faces, rip, b, lam=0.535, fm_cap=0.475, reg_lo=0.40, nthreads=8):
    nrm = _fast_vertex_normals(lib, v_dl, faces, nthreads=nthreads)
    inv1 = np.linalg.inv(b["grid_aff"])
    cort_ids = np.where(rip < 0.5)[0]
    sel_ids = np.random.default_rng(42).choice(cort_ids, size=min(18000, len(cort_ids)), replace=False)
    p_cur = v_dl[sel_ids].astype(np.float32).copy()
    ns = nrm[sel_ids]
    guide = b["fa"]; csf_v = b["c_base"]
    d_mm = float(np.clip(1.05 * b["dx"], 1.35, 2.00))

    for _ in range(6):
        f0 = map_coordinates(guide, inv1[:3, :3] @ p_cur.T + inv1[:3, 3:4], order=1, mode="nearest")
        fm = map_coordinates(guide, inv1[:3, :3] @ (p_cur - d_mm * ns).T + inv1[:3, 3:4], order=1, mode="nearest")
        fmid = map_coordinates(guide, inv1[:3, :3] @ (p_cur + 0.5 * d_mm * ns).T + inv1[:3, 3:4], order=1, mode="nearest")
        fp = map_coordinates(guide, inv1[:3, :3] @ (p_cur + d_mm * ns).T + inv1[:3, 3:4], order=1, mode="nearest")
        fp_safe = np.minimum(fp, fmid)
        c0 = map_coordinates(csf_v, inv1[:3, :3] @ p_cur.T + inv1[:3, 3:4], order=1, mode="nearest")
        fm_c = np.minimum(fm, fm_cap)
        fp_c = np.maximum(fp_safe, 0.08)
        thr = (1.0 - lam) * fp_c + lam * fm_c
        step = np.clip((f0 - thr) / np.maximum(fm_c - fp_c, 0.08) * d_mm, -0.60, 0.60)
        step = np.where((c0 > 0.25) | (f0 < 0.09), -0.45, np.where((f0 > fm_cap) & (c0 < 0.15), +0.45, step))
        p_cur = p_cur + step[:, None] * ns

    ijk_f = inv1[:3, :3] @ p_cur.T + inv1[:3, 3:4]
    l_vol = b["l_blend"]; sl_wm = b["sl_wm"]
    lf = map_coordinates(l_vol, ijk_f, order=1, mode="nearest")
    cf = map_coordinates(csf_v, ijk_f, order=1, mode="nearest")
    ex = map_coordinates(b["excl"], ijk_f, order=1, mode="nearest")
    fm_f = map_coordinates(guide, inv1[:3, :3] @ (p_cur - d_mm * ns).T + inv1[:3, 3:4], order=1, mode="nearest")
    fmid_f = map_coordinates(guide, inv1[:3, :3] @ (p_cur + 0.5 * d_mm * ns).T + inv1[:3, 3:4], order=1, mode="nearest")
    fp_f = map_coordinates(guide, inv1[:3, :3] @ (p_cur + d_mm * ns).T + inv1[:3, 3:4], order=1, mode="nearest")
    mono_ok = (fp_f <= fmid_f + 0.025) & (fm_f - np.minimum(fp_f, fmid_f) > 0.07)
    keep = (cf < 0.22) & (ex < 0.25) & mono_ok & np.isfinite(lf)
    if int(keep.sum()) < 200:
        return np.zeros_like(l_vol, dtype=np.float32)

    l0_glob = float(np.median(lf[keep]))
    shape = l_vol.shape; ds_f = 4
    s4 = (shape[0] // ds_f + 1, shape[1] // ds_f + 1, shape[2] // ds_f + 1)
    num = np.zeros(s4, dtype=np.float32); den = np.zeros(s4, dtype=np.float32)
    ijk_k = np.rint(ijk_f[:, keep] / ds_f).astype(np.int32)
    ok_b = (ijk_k[0] >= 0) & (ijk_k[0] < s4[0]) & (ijk_k[1] >= 0) & (ijk_k[1] < s4[1]) & (ijk_k[2] >= 0) & (ijk_k[2] < s4[2])
    ijk_k = ijk_k[:, ok_b]
    dl_k = np.clip(lf[keep][ok_b] - l0_glob, -0.50 * sl_wm, 0.40 * sl_wm).astype(np.float32)
    np.add.at(num, (ijk_k[0], ijk_k[1], ijk_k[2]), dl_k)
    np.add.at(den, (ijk_k[0], ijk_k[1], ijk_k[2]), 1.0)
    dl_4mm = np.clip(gaussian_filter(num, sigma=2.8) / (gaussian_filter(den, sigma=2.8) + 0.12), -reg_lo * sl_wm, +0.25 * sl_wm).astype(np.float32)
    dl_1mm = _fast_zoom_linear(lib, dl_4mm, shape, 1.0 / ds_f, nthreads=nthreads)
    return (l0_glob + dl_1mm).astype(np.float32)


def _self_cal_pial(lib, vp_dl, faces, rip, md_v, mk_v, grid_aff, dx, d_t, lam_p=0.68, md_cap=2.65, nthreads=8):
    nrm = _fast_vertex_normals(lib, vp_dl, faces, nthreads=nthreads)
    inv1 = np.linalg.inv(grid_aff)
    cort_ids = np.where(rip < 0.5)[0]
    sel_ids = np.random.default_rng(42).choice(cort_ids, size=min(18000, len(cort_ids)), replace=False)
    p_cur = vp_dl[sel_ids].astype(np.float32).copy()
    ns = nrm[sel_ids]
    d_mm = float(np.clip(1.05 * dx, 1.35, 2.00))

    for _ in range(6):
        m0 = map_coordinates(md_v, inv1[:3, :3] @ p_cur.T + inv1[:3, 3:4], order=1, mode="nearest")
        mm = map_coordinates(md_v, inv1[:3, :3] @ (p_cur - d_mm * ns).T + inv1[:3, 3:4], order=1, mode="nearest")
        mmid = map_coordinates(md_v, inv1[:3, :3] @ (p_cur + 0.5 * d_mm * ns).T + inv1[:3, 3:4], order=1, mode="nearest")
        mp = map_coordinates(md_v, inv1[:3, :3] @ (p_cur + d_mm * ns).T + inv1[:3, 3:4], order=1, mode="nearest")
        mk0 = map_coordinates(mk_v, inv1[:3, :3] @ p_cur.T + inv1[:3, 3:4], order=1, mode="nearest")
        mkp = map_coordinates(mk_v, inv1[:3, :3] @ (p_cur + d_mm * ns).T + inv1[:3, 3:4], order=1, mode="nearest")
        mp_safe = np.maximum(mp, mmid)
        mm_c = np.clip(mm, d_t - 0.15, d_t + 0.30)
        mp_c = np.clip(mp_safe, d_t + 0.40, md_cap)
        thr = (1.0 - lam_p) * mm_c + lam_p * mp_c
        step = np.clip((thr - m0) / np.maximum(mp_c - mm_c, 0.25) * d_mm, -0.65, 0.65)
        valid = (mk0 > 0.6) & (mkp > 0.6) & (mp_safe - mm > 0.35)
        p_cur = p_cur + np.where(valid, step, 0.0)[:, None] * ns

    ijk_f = inv1[:3, :3] @ p_cur.T + inv1[:3, 3:4]
    m0_f = map_coordinates(md_v, ijk_f, order=1, mode="nearest")
    mm_f = map_coordinates(md_v, inv1[:3, :3] @ (p_cur - d_mm * ns).T + inv1[:3, 3:4], order=1, mode="nearest")
    mmid_f = map_coordinates(md_v, inv1[:3, :3] @ (p_cur + 0.5 * d_mm * ns).T + inv1[:3, 3:4], order=1, mode="nearest")
    mp_f = map_coordinates(md_v, inv1[:3, :3] @ (p_cur + d_mm * ns).T + inv1[:3, 3:4], order=1, mode="nearest")
    mk0_f = map_coordinates(mk_v, ijk_f, order=1, mode="nearest")
    mkp_f = map_coordinates(mk_v, inv1[:3, :3] @ (p_cur + d_mm * ns).T + inv1[:3, 3:4], order=1, mode="nearest")
    keep = (mk0_f > 0.6) & (mkp_f > 0.6) & (mm_f < d_t + 0.32) & (mp_f >= mmid_f - 0.08) & (mp_f - mm_f > 0.40) & np.isfinite(m0_f)
    if int(keep.sum()) < 200:
        md0_g = float(np.clip(d_t + 0.62, 1.25, 1.55))
        return md0_g, np.full_like(md_v, md0_g, dtype=np.float32)

    md0_g = float(np.clip(np.median(m0_f[keep]), d_t + 0.38, d_t + 0.98))
    shape = md_v.shape; ds_f = 4
    s4 = (shape[0] // ds_f + 1, shape[1] // ds_f + 1, shape[2] // ds_f + 1)
    num = np.zeros(s4, dtype=np.float32); den = np.zeros(s4, dtype=np.float32)
    ijk_k = np.rint(ijk_f[:, keep] / ds_f).astype(np.int32)
    ok_b = (ijk_k[0] >= 0) & (ijk_k[0] < s4[0]) & (ijk_k[1] >= 0) & (ijk_k[1] < s4[1]) & (ijk_k[2] >= 0) & (ijk_k[2] < s4[2])
    ijk_k = ijk_k[:, ok_b]
    dm_k = np.clip(m0_f[keep][ok_b] - md0_g, -0.22, 0.22).astype(np.float32)
    np.add.at(num, (ijk_k[0], ijk_k[1], ijk_k[2]), dm_k)
    np.add.at(den, (ijk_k[0], ijk_k[1], ijk_k[2]), 1.0)
    dm_4 = np.clip(gaussian_filter(num, sigma=3.0) / (gaussian_filter(den, sigma=3.0) + 0.18), -0.16, +0.16).astype(np.float32)
    dm_1 = _fast_zoom_linear(lib, dm_4, shape, 1.0 / ds_f, nthreads=nthreads)
    return md0_g, (md0_g + dm_1).astype(np.float32)


def _refine_one_hemi(lib, hemi: str, vw_in, wf, vp_in, pf, nat, nthreads: int = 8):
    rip = _get_medial_mask(lib, vw_in, wf, hemi=hemi, nthreads=nthreads)
    b = _build_hemi_physics(lib, vw_in, vp_in, nat, nthreads=nthreads)

    # White v3
    l0_use = _self_cal_white(lib, vw_in, wf, rip, b, nthreads=nthreads)
    z_base = ((b["l_blend"] - l0_use) / b["sl_wm"]).astype(np.float32)
    w_band = np.exp(-0.5 * (z_base / 0.72) ** 2) * np.clip((0.26 - b["c_base"]) / 0.12, 0.0, 1.0) * b["mask_1mm"]
    zf = np.where(b["mask_1mm"], z_base + (0.16 * w_band) * b["sheet_joint"] - b["pen_tot"], -3.5).astype(np.float32)

    dx = b["dx"]
    dx_coarse = float(np.clip((dx - 1.35) / 0.65, 0.0, 1.0))
    apv_phys = 0.50 * float(np.clip(dx / 1.25, 0.85, 1.45))
    ed_mm = float(np.clip(0.95 * dx, 1.15, 1.95))
    cp_eff = 0.65 + 0.25 * dx_coarse

    inv1 = np.linalg.inv(b["grid_aff"])
    z0 = map_coordinates(zf, inv1[:3, :3] @ vw_in.T + inv1[:3, 3:4], order=1)[rip < 0.5]
    shift_mag = abs(float(np.median(z0)))
    calm = float(np.clip((0.15 - shift_mag) / 0.10, 0.0, 1.0))
    z_dead_eff = 0.18 + 1.25 * 0.10 * calm * (0.35 + 0.65 * dx_coarse)
    w_anc_eff = 0.42 + 1.25 * 0.18 * calm * (0.40 + 0.60 * dx_coarse)
    max_d_eff = 1.60 - 1.25 * 0.35 * calm * dx_coarse

    vw = np.ascontiguousarray(vw_in, dtype=np.float64).copy()
    f_w = np.ascontiguousarray(wf, dtype=np.int32)
    rip_f = np.ascontiguousarray(rip, dtype=np.float32)
    zf_p, zf_nx, zf_ny, zf_nz, zf_M, _k1 = _prep_f32(zf, b["grid_aff"])
    fa_p, fa_nx, fa_ny, fa_nz, fa_M, _k2 = _prep_f32(b["fa_fwe"], b["grid_aff"])
    cs_p, cs_nx, cs_ny, cs_nz, cs_M, _k3 = _prep_f32(b["c_base"], b["grid_aff"])
    ex_p, ex_nx, ex_ny, ex_nz, ex_M, _k4 = _prep_f32(b["excl"], b["grid_aff"])

    lib.refine_dl_surface_physics_flow(
        vw.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(vw),
        f_w.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), len(f_w),
        rip_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        zf_p, zf_nx, zf_ny, zf_nz, zf_M,
        fa_p, fa_nx, fa_ny, fa_nz, fa_M,
        cs_p, cs_nx, cs_ny, cs_nz, cs_M,
        None, 0, 0, 0, None,
        ex_p, ex_nx, ex_ny, ex_nz, ex_M,
        18, 16, 0.95, 0.58, 0.125, 0.075,
        0.04, z_dead_eff, 2.2, 0.0,
        w_anc_eff, 0.25, 0.14,
        max_d_eff, ed_mm, 0.74,
        apv_phys, cp_eff, 0.0, 2, int(nthreads),
    )

    vp = None
    if vp_in is not None and pf is not None:
        # Pial v4 (reuses White's 1.0mm grid directly!)
        mask_1 = b["mask_1mm"]
        md_1 = b["md_raw"]
        md_nn = _fast_edt_3d(lib, ~mask_1, nn_src=md_1, need_dist=False, nthreads=nthreads)
        mk_f = mask_1.astype(np.float32)
        d_t = b["d_t"]
        md0_g, md0_1mm = _self_cal_pial(lib, vp_in, pf, rip, md_nn, mk_f, b["grid_aff"], dx, d_t, nthreads=nthreads)
        s_md = max(0.28, 0.52 * (md0_g - d_t))
        z_pial = np.clip((md0_1mm - md_nn) / s_md, -3.5, 3.5).astype(np.float32)
        wm_m = (b["fa"] > 0.30) & (md_nn < d_t + 0.16) & (mk_f > 0.5)
        z_pial = np.where(wm_m, np.maximum(z_pial, 1.3), z_pial)
        z_pial = np.where(mk_f > 0.5, z_pial, 0.0).astype(np.float32)

        ed_p = float(np.clip(1.00 * dx, 1.25, 2.00))
        vp = np.ascontiguousarray(vp_in, dtype=np.float64).copy()
        f_p = np.ascontiguousarray(pf, dtype=np.int32)
        zp_p, zp_nx, zp_ny, zp_nz, zp_M, _p1 = _prep_f32(z_pial, b["grid_aff"])
        md_p, md_nx, md_ny, md_nz, md_M, _p2 = _prep_f32(md_nn, b["grid_aff"])
        fa2_p, fa2_nx, fa2_ny, fa2_nz, fa2_M, _p3 = _prep_f32(b["fa"], b["grid_aff"])
        mk_p, mk_nx, mk_ny, mk_nz, mk_M, _p4 = _prep_f32(mk_f, b["grid_aff"])

        lib.refine_dl_pial_physics_flow(
            vp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            vw.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), len(vp),
            f_p.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), len(f_p),
            rip_f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            zp_p, zp_nx, zp_ny, zp_nz, zp_M,
            md_p, md_nx, md_ny, md_nz, md_M,
            fa2_p, fa2_nx, fa2_ny, fa2_nz, fa2_M,
            mk_p, mk_nx, mk_ny, mk_nz, mk_M,
            18, 16, 0.95, 0.58, 0.125, 0.075,
            0.14, 2.2, 0.30, 0.26, 0.14,
            2.2, ed_p, 0.74,
            0.55, d_t, 0.70, 0.0, 0.85, 0.95, 2.65,
            1, int(nthreads),
        )
    return vw, vp


def resolve_dti_paths(dti_dir: Path, subject: str, mask_path: Path) -> Optional[Dict[str, Path]]:
    dti_dir = Path(dti_dir)
    mask_file = Path(mask_path)
    if not mask_file.is_file() or mask_file.stat().st_size == 0:
        return None
    candidates = (
        {
            "fa": dti_dir / f"{subject}-FA.nii.gz",
            "md": dti_dir / f"{subject}-MD.nii.gz",
            "l1": dti_dir / f"{subject}-MaxEigenvalue.nii.gz",
            "l2": dti_dir / f"{subject}-MidEigenvalue.nii.gz",
            "l3": dti_dir / f"{subject}-MinEigenvalue.nii.gz",
            "mask": mask_file,
        },
        {
            "fa": dti_dir / f"{subject}-dti-FractionalAnisotropy.nii.gz",
            "md": dti_dir / f"{subject}-dti-MeanDiffusivity.nii.gz",
            "l1": dti_dir / f"{subject}-dti-MaxEigenvalue.nii.gz",
            "l2": dti_dir / f"{subject}-dti-MidEigenvalue.nii.gz",
            "l3": dti_dir / f"{subject}-dti-MinEigenvalue.nii.gz",
            "mask": mask_file,
        },
        {
            "fa": dti_dir / "DTI_FA.nrrd",
            "md": dti_dir / "DTI_MeanDiffusivity.nrrd",
            "l1": dti_dir / "DTI_MaxEigenvalue.nrrd",
            "l2": dti_dir / "DTI_MidEigenvalue.nrrd",
            "l3": dti_dir / "DTI_MinEigenvalue.nrrd",
            "mask": mask_file,
        },
    )
    for paths in candidates:
        if all(p.is_file() and p.stat().st_size > 0 for p in paths.values()):
            return paths
    return None


def refine_bilateral_surfaces(
    meshes: Dict[str, Tuple[np.ndarray, np.ndarray]],
    dti_dir: Path,
    subject: str,
    mask_path: Path,
    nthreads: int = 8,
) -> Dict[str, np.ndarray]:
    """
    Refine native-RAS surfaces in memory using exported native DTI maps and brain mask.
    `meshes` maps stems ('lh.white', 'lh.pial', 'rh.white', 'rh.pial') -> (vertices, faces).
    Returns updated vertex arrays keyed by the same stems.
    """
    out = {k: np.asarray(v, dtype=np.float64) for k, (v, _) in meshes.items()}
    if not meshes or min(len(v) for v, _ in meshes.values()) < 1000:
        return out
    dti_paths = resolve_dti_paths(dti_dir, subject, mask_path)
    if dti_paths is None:
        return out
    lib = _build_or_load_lib()
    if lib is None:
        return out

    nat = _preload_native_dti(dti_paths)
    if int(nat["mask_nat"].sum()) < 1000:
        return out

    hemis = [h for h in ("lh", "rh") if f"{h}.white" in meshes]
    if not hemis:
        return out
    nt_hemi = max(1, int(nthreads) // len(hemis))

    def _run(h):
        vw_in, wf = meshes[f"{h}.white"]
        vp_in, pf = meshes.get(f"{h}.pial", (None, None))
        vw_out, vp_out = _refine_one_hemi(lib, h, vw_in, wf, vp_in, pf, nat, nthreads=nt_hemi)
        return h, vw_out, vp_out

    if len(hemis) == 2:
        with ThreadPoolExecutor(max_workers=2) as tex:
            results = list(tex.map(_run, hemis))
    else:
        results = [_run(hemis[0])]

    for h, vw_out, vp_out in results:
        if vw_out is not None and np.isfinite(vw_out).all():
            out[f"{h}.white"] = vw_out
        if vp_out is not None and np.isfinite(vp_out).all():
            out[f"{h}.pial"] = vp_out
    return out
