"""Hardware-adaptive CPU and PyTorch CUDA acceleration for DIPY preprocessing.

Accelerates the two in-process DIPY stages of the canonical preprocessing route:
1. ``mppca`` (Veraart et al., 2016; Manjon et al., 2013):
   - Uses the exact dual/primal symmetric covariance identity so ``eigh`` runs on
     ``min(P, D) x min(P, D)`` matrices (e.g. ``125 x 125`` instead of ``288 x 288``
     for multi-shell HCP DWI, a 12.2x matrix-flop reduction).
   - Vectorizes the Marcenko-Pastur rank/variance classifier without Python loops.
   - Streams a bounded z-slab / rolling 5-slice ring buffer (~150 MB VRAM on GPU)
     with automatic VRAM-aware batch scaling and CUDA OOM fallback to multi-threaded CPU.
2. ``gibbs_removal`` (Kellner et al., 2016):
   - Batches 2D slices across 1D sub-voxel Fourier shifts and 2D k-space recombination
     on PyTorch CUDA or multi-threaded NumPy/SciPy FFT workers.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import ctypes
import ctypes.util
import glob
import math
import os
import sys
import threading
from typing import Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from shared.hardware_adapt import (
    MIB,
    adaptive_cuda_scaled_batch,
    empty_cuda_cache_safely,
    is_cuda_oom,
    query_cuda_free_total_bytes,
    run_cuda_chunked_with_oom_fallback,
)
from shared.numeric_threads import effective_cpu_threads


_CUSOLVER_LIB = None
_CUSOLVER_LOAD_ATTEMPTED = False
_CUSOLVER_HANDLES: dict[tuple[int, int, int], tuple[ctypes.c_void_p, ctypes.c_void_p]] = {}
_CUSOLVER_LOCK = threading.Lock()


def _load_cusolver_lib():
    """Locate and bind ``cusolverDnXsyevBatched`` from NVIDIA cuSOLVER (Linux & Windows)."""
    global _CUSOLVER_LIB, _CUSOLVER_LOAD_ATTEMPTED
    if _CUSOLVER_LOAD_ATTEMPTED:
        return _CUSOLVER_LIB
    with _CUSOLVER_LOCK:
        if _CUSOLVER_LOAD_ATTEMPTED:
            return _CUSOLVER_LIB
        _CUSOLVER_LOAD_ATTEMPTED = True
        candidates: list[str] = []
        try:
            import nvidia.cusolver  # type: ignore[import-not-found]

            for base in getattr(nvidia.cusolver, "__path__", []):
                candidates.extend(sorted(glob.glob(os.path.join(base, "lib", "libcusolver.so*"))))
                candidates.extend(sorted(glob.glob(os.path.join(base, "bin", "cusolver64_*.dll"))))
        except Exception:
            pass
        try:
            import torch

            t_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
            candidates.extend(sorted(glob.glob(os.path.join(t_lib, "libcusolver.so*"))))
            candidates.extend(sorted(glob.glob(os.path.join(t_lib, "cusolver64_*.dll"))))
        except Exception:
            pass
        if sys.platform.startswith("linux") and os.path.exists("/proc/self/maps"):
            try:
                with open("/proc/self/maps", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if "libcusolver.so" in line:
                            p = line.strip().split()[-1]
                            if p not in candidates:
                                candidates.append(p)
            except Exception:
                pass
        found = ctypes.util.find_library("cusolver")
        if found:
            candidates.append(found)

        c_void_p = ctypes.c_void_p
        c_int = ctypes.c_int
        c_int64 = ctypes.c_int64
        c_size_t = ctypes.c_size_t
        for path in candidates:
            try:
                lib = ctypes.CDLL(path)
                if not (hasattr(lib, "cusolverDnXsyevBatched") and hasattr(lib, "cusolverDnXsyevBatched_bufferSize")):
                    continue
                lib.cusolverDnCreate.argtypes = [ctypes.POINTER(c_void_p)]
                lib.cusolverDnCreate.restype = c_int
                lib.cusolverDnDestroy.argtypes = [c_void_p]
                lib.cusolverDnDestroy.restype = c_int
                lib.cusolverDnSetStream.argtypes = [c_void_p, c_void_p]
                lib.cusolverDnSetStream.restype = c_int
                lib.cusolverDnCreateParams.argtypes = [ctypes.POINTER(c_void_p)]
                lib.cusolverDnCreateParams.restype = c_int
                lib.cusolverDnDestroyParams.argtypes = [c_void_p]
                lib.cusolverDnDestroyParams.restype = c_int
                lib.cusolverDnXsyevBatched_bufferSize.argtypes = [
                    c_void_p, c_void_p, c_int, c_int, c_int64,
                    c_int, c_void_p, c_int64,
                    c_int, c_void_p,
                    c_int,
                    ctypes.POINTER(c_size_t), ctypes.POINTER(c_size_t),
                    c_int64,
                ]
                lib.cusolverDnXsyevBatched_bufferSize.restype = c_int
                lib.cusolverDnXsyevBatched.argtypes = [
                    c_void_p, c_void_p, c_int, c_int, c_int64,
                    c_int, c_void_p, c_int64,
                    c_int, c_void_p,
                    c_int,
                    c_void_p, c_size_t,
                    c_void_p, c_size_t,
                    c_void_p, c_int64,
                ]
                lib.cusolverDnXsyevBatched.restype = c_int
                _CUSOLVER_LIB = lib
                break
            except Exception:
                continue
        return _CUSOLVER_LIB


def _get_cusolver_handle_params(lib, dev_idx: int):
    key = (os.getpid(), threading.get_ident(), int(dev_idx))
    pair = _CUSOLVER_HANDLES.get(key)
    if pair is not None:
        return pair
    handle = ctypes.c_void_p()
    params = ctypes.c_void_p()
    if lib.cusolverDnCreate(ctypes.byref(handle)) != 0:
        return None
    if lib.cusolverDnCreateParams(ctypes.byref(params)) != 0:
        lib.cusolverDnDestroy(handle)
        return None
    _CUSOLVER_HANDLES[key] = (handle, params)
    return handle, params


def _eigh_cuda_batched(sym):
    """Batched symmetric eigendecomposition on CUDA.

    For ``N <= 32``, PyTorch dispatches cuSOLVER's fast warp-level ``syevjBatched``.
    For ``32 < N <= 512`` (e.g. ``125 x 125`` in 3D ``5x5x5`` MP-PCA), PyTorch's
    default wrapper falls back to a slow per-matrix loop; here we bind NVIDIA's
    native ``cusolverDnXsyevBatched`` with VRAM-aware sub-batching and automatic
    fallback to ``torch.linalg.eigh``.
    """
    import torch

    bsz, n_dim, _ = sym.shape
    if not sym.is_cuda or n_dim <= 32 or n_dim > 512 or bsz < 8:
        return torch.linalg.eigh(sym)

    # Ensure cuSOLVER context is initialized in PyTorch before querying/binding.
    lib = _load_cusolver_lib()
    if lib is None:
        _ = torch.linalg.eigh(sym[:1, :2, :2])
        global _CUSOLVER_LOAD_ATTEMPTED
        _CUSOLVER_LOAD_ATTEMPTED = False
        lib = _load_cusolver_lib()
    if lib is None:
        return torch.linalg.eigh(sym)

    dev = sym.device
    dev_idx = torch.cuda.current_device() if dev.index is None else int(dev.index)
    hp = _get_cusolver_handle_params(lib, dev_idx)
    if hp is None:
        return torch.linalg.eigh(sym)
    handle, params = hp

    cuda_dt = 1 if sym.dtype == torch.float64 else (0 if sym.dtype == torch.float32 else -1)
    if cuda_dt < 0:
        return torch.linalg.eigh(sym)

    # VRAM-aware sub-batch sizing so 4-8 GiB consumer GPUs never OOM on workspace.
    elem_bytes = 8 if cuda_dt == 1 else 4
    est_ws_per_mat = max(65536, n_dim * n_dim * elem_bytes * 16)
    free_info = query_cuda_free_total_bytes(dev)
    if free_info is not None:
        ws_budget = max(int(64 * MIB), int(free_info[0] * 0.25))
        max_sub = max(64, min(4096, ws_budget // est_ws_per_mat))
    else:
        max_sub = 1024
    sub_b = min(bsz, max_sub)

    c_void_p = ctypes.c_void_p
    c_size_t = ctypes.c_size_t
    stream_ptr = c_void_p(torch.cuda.current_stream(dev).cuda_stream)
    if lib.cusolverDnSetStream(handle, stream_ptr) != 0:
        return torch.linalg.eigh(sym)

    eigvals_out = torch.empty((bsz, n_dim), device=dev, dtype=sym.dtype)
    eigvecs_out = torch.empty((bsz, n_dim, n_dim), device=dev, dtype=sym.dtype)
    info_buf = torch.empty((sub_b,), device=dev, dtype=torch.int32)
    d_buf = None
    d_buf_cap = 0

    try:
        for s in range(0, bsz, sub_b):
            e = min(bsz, s + sub_b)
            cur_b = e - s
            a_work = sym[s:e].contiguous().clone()
            w_sub = eigvals_out[s:e]
            ws_dev = c_size_t(0)
            ws_host = c_size_t(0)
            st = lib.cusolverDnXsyevBatched_bufferSize(
                handle, params, 1, 0, n_dim,
                cuda_dt, c_void_p(a_work.data_ptr()), n_dim,
                cuda_dt, c_void_p(w_sub.data_ptr()),
                cuda_dt,
                ctypes.byref(ws_dev), ctypes.byref(ws_host),
                cur_b,
            )
            if st != 0:
                return torch.linalg.eigh(sym)
            need_dev = max(1, int(ws_dev.value))
            if d_buf is None or d_buf_cap < need_dev:
                d_buf = None
                d_buf = torch.empty((need_dev,), device=dev, dtype=torch.uint8)
                d_buf_cap = need_dev
            h_buf = (ctypes.c_uint8 * max(1, int(ws_host.value)))() if ws_host.value > 0 else None
            h_ptr = ctypes.cast(h_buf, c_void_p) if h_buf is not None else c_void_p(0)

            st2 = lib.cusolverDnXsyevBatched(
                handle, params, 1, 0, n_dim,
                cuda_dt, c_void_p(a_work.data_ptr()), n_dim,
                cuda_dt, c_void_p(w_sub.data_ptr()),
                cuda_dt,
                c_void_p(d_buf.data_ptr()), int(ws_dev.value),
                h_ptr, int(ws_host.value),
                c_void_p(info_buf.data_ptr()), cur_b,
            )
            if st2 != 0 or bool(torch.any(info_buf[:cur_b] != 0).item()):
                return torch.linalg.eigh(sym)
            # Fortran column-major eigenvectors -> row-major columns via transpose(-1, -2)
            eigvecs_out[s:e].copy_(a_work.transpose(-1, -2))
        return eigvals_out, eigvecs_out
    except Exception as exc:
        if is_cuda_oom(exc):
            raise
        return torch.linalg.eigh(sym)


def resolve_dipy_accel_device(requested: Optional[str] = None) -> str:
    value = str(
        requested
        if requested is not None
        else os.environ.get("SLICERDSPACE_DIPY_DEVICE", "auto")
    ).strip().lower()
    if value not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported DIPY preprocessing device {value!r}; expected 'auto', 'cpu', or 'cuda'.")
    if value == "cpu":
        return "cpu"
    try:
        import torch

        cuda_ok = bool(torch.cuda.is_available())
    except Exception:
        cuda_ok = False
    if value == "cuda" and not cuda_ok:
        raise RuntimeError("DIPY CUDA acceleration was explicitly requested, but PyTorch CUDA is unavailable.")
    return "cuda" if cuda_ok else "cpu"


def _solve_mppca_core_numpy_fp64(
    patches_64: np.ndarray,
    *,
    dim: int,
    num_samples: int,
    tau_factor_sq: float,
    calc_dtype: np.dtype,
    return_sigma: bool,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    mean = np.mean(patches_64, axis=1, keepdims=True)
    xc = patches_64 - mean
    dual = dim > num_samples
    if dual:
        gram = np.matmul(xc, np.swapaxes(xc, 1, 2)) * (1.0 / float(num_samples))
        gram = 0.5 * (gram + np.swapaxes(gram, 1, 2))
        eigvals, eigvecs = np.linalg.eigh(gram)
    else:
        cov = np.matmul(np.swapaxes(xc, 1, 2), xc) * (1.0 / float(num_samples))
        cov = 0.5 * (cov + np.swapaxes(cov, 1, 2))
        eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0.0)

    m_cls = min(dim, num_samples - 1)
    l_vals = eigvals[:, -m_cls:]
    idx = np.arange(1, m_cls + 1, dtype=np.float64)
    var_all = np.cumsum(l_vals, axis=1) / idx[None, :]
    scale = 4.0 * np.sqrt(idx / float(num_samples))
    r_all = l_vals - l_vals[:, 0:1] - scale[None, :] * var_all
    cond = r_all <= 0
    cond[:, 0] = True
    c_idx = np.max(cond * np.arange(m_cls, dtype=np.int64)[None, :], axis=1)
    this_var = var_all[np.arange(patches_64.shape[0]), c_idx]

    tau = tau_factor_sq * this_var
    kept = eigvals >= tau[:, None]
    if dual:
        pos_tau = tau > 0
        kept = np.where(pos_tau[:, None], kept, True)
        n_kept = np.where(pos_tau, np.sum(kept, axis=1), dim).astype(np.float64)
        u_kept = eigvecs * kept[:, None, :].astype(np.float64)
        proj = np.matmul(np.swapaxes(u_kept, 1, 2), xc)
        xest = (np.matmul(u_kept, proj) + mean).astype(calc_dtype, copy=False)
    else:
        n_kept = np.sum(kept, axis=1).astype(np.float64)
        w_kept = eigvecs * kept[:, None, :].astype(np.float64)
        proj = np.matmul(xc, w_kept)
        xest = (np.matmul(proj, np.swapaxes(w_kept, 1, 2)) + mean).astype(calc_dtype, copy=False)

    this_theta = (1.0 / (1.0 + n_kept)).astype(calc_dtype, copy=False)
    return xest, this_theta, (this_var.astype(calc_dtype, copy=False) if return_sigma else None)


def _solve_mppca_batch_numpy(
    patches: np.ndarray,
    *,
    dim: int,
    num_samples: int,
    tau_factor_sq: float,
    calc_dtype: np.dtype,
    return_sigma: bool,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Exact vectorized MP-PCA patch solver on CPU (FP64 OpenBLAS dsyevd + constant-patch short-circuit)."""
    bsz = patches.shape[0]
    const_theta = np.dtype(calc_dtype).type(1.0 / (1.0 + float(dim)))
    active = np.any(patches != patches[:, 0:1, :], axis=(1, 2))
    n_act = int(np.count_nonzero(active))
    if n_act == 0:
        xest = np.array(patches, dtype=calc_dtype, copy=True)
        this_theta = np.full((bsz,), const_theta, dtype=calc_dtype)
        this_var = np.zeros((bsz,), dtype=calc_dtype) if return_sigma else None
        return xest, this_theta, this_var
    if n_act == bsz:
        return _solve_mppca_core_numpy_fp64(
            np.asarray(patches, dtype=np.float64),
            dim=dim,
            num_samples=num_samples,
            tau_factor_sq=tau_factor_sq,
            calc_dtype=calc_dtype,
            return_sigma=return_sigma,
        )
    xest = np.array(patches, dtype=calc_dtype, copy=True)
    this_theta = np.full((bsz,), const_theta, dtype=calc_dtype)
    this_var = np.zeros((bsz,), dtype=calc_dtype) if return_sigma else None
    x_a, th_a, v_a = _solve_mppca_core_numpy_fp64(
        np.asarray(patches[active], dtype=np.float64),
        dim=dim,
        num_samples=num_samples,
        tau_factor_sq=tau_factor_sq,
        calc_dtype=calc_dtype,
        return_sigma=return_sigma,
    )
    xest[active] = x_a
    this_theta[active] = th_a
    if this_var is not None and v_a is not None:
        this_var[active] = v_a
    return xest, this_theta, this_var


def _solve_mppca_core_torch_fp64(
    patches_t,
    *,
    dim: int,
    num_samples: int,
    tau_factor_sq: float,
    return_sigma: bool,
):
    import torch

    calc_t = patches_t.dtype
    dev = patches_t.device
    patches_64 = patches_t.to(torch.float64)
    mean_64 = patches_64.mean(dim=1, keepdim=True)
    xc_64 = patches_64 - mean_64
    dual = dim > num_samples
    if dual:
        gram = torch.matmul(xc_64, xc_64.transpose(1, 2)) * (1.0 / float(num_samples))
        gram = 0.5 * (gram + gram.transpose(1, 2))
        eigvals, eigvecs = _eigh_cuda_batched(gram)
    else:
        cov = torch.matmul(xc_64.transpose(1, 2), xc_64) * (1.0 / float(num_samples))
        cov = 0.5 * (cov + cov.transpose(1, 2))
        eigvals, eigvecs = _eigh_cuda_batched(cov)
    eigvals = torch.clamp(eigvals, min=0.0)

    m_cls = min(dim, num_samples - 1)
    l_vals = eigvals[:, -m_cls:]
    idx = torch.arange(1, m_cls + 1, device=dev, dtype=torch.float64)
    var_all = torch.cumsum(l_vals, dim=1) / idx.unsqueeze(0)
    scale = 4.0 * torch.sqrt(idx / float(num_samples))
    r_all = l_vals - l_vals[:, 0:1] - scale.unsqueeze(0) * var_all
    cond = r_all <= 0
    cond[:, 0] = True
    pos_idx = torch.arange(m_cls, device=dev, dtype=torch.long).unsqueeze(0)
    c_idx = torch.amax(torch.where(cond, pos_idx, torch.zeros_like(pos_idx)), dim=1)
    this_var = var_all.gather(1, c_idx.unsqueeze(1)).squeeze(1)

    tau = tau_factor_sq * this_var
    kept = eigvals >= tau.unsqueeze(1)
    if dual:
        pos_tau = tau > 0
        kept = torch.where(pos_tau.unsqueeze(1), kept, torch.ones_like(kept))
        n_kept = torch.where(
            pos_tau,
            kept.sum(dim=1),
            torch.full((patches_t.shape[0],), dim, device=dev, dtype=torch.long),
        ).to(torch.float64)
        u_kept = eigvecs * kept.unsqueeze(1).to(torch.float64)
        proj = torch.matmul(u_kept.transpose(1, 2), xc_64)
        xest = (torch.matmul(u_kept, proj) + mean_64).to(calc_t)
    else:
        n_kept = kept.sum(dim=1).to(torch.float64)
        w_kept = eigvecs * kept.unsqueeze(1).to(torch.float64)
        proj = torch.matmul(xc_64, w_kept)
        xest = (torch.matmul(proj, w_kept.transpose(1, 2)) + mean_64).to(calc_t)

    this_theta = (1.0 / (1.0 + n_kept)).to(calc_t)
    return xest, this_theta, (this_var.to(calc_t) if return_sigma else None)


def _solve_mppca_batch_torch(
    patches_t,
    *,
    dim: int,
    num_samples: int,
    tau_factor_sq: float,
    return_sigma: bool,
):
    """Exact vectorized MP-PCA patch solver on PyTorch CUDA (FP64 cusolverDnXsyevBatched + constant-patch short-circuit)."""
    import torch

    bsz = int(patches_t.shape[0])
    calc_t = patches_t.dtype
    dev = patches_t.device
    const_theta = 1.0 / (1.0 + float(dim))
    active = torch.any(patches_t != patches_t[:, 0:1, :], dim=(1, 2))
    n_act = int(active.sum().item())
    if n_act == 0:
        xest = patches_t.clone()
        this_theta = torch.full((bsz,), const_theta, device=dev, dtype=calc_t)
        this_var = torch.zeros((bsz,), device=dev, dtype=calc_t) if return_sigma else None
        return xest, this_theta, this_var
    if n_act == bsz:
        return _solve_mppca_core_torch_fp64(
            patches_t,
            dim=dim,
            num_samples=num_samples,
            tau_factor_sq=tau_factor_sq,
            return_sigma=return_sigma,
        )
    xest = patches_t.clone()
    this_theta = torch.full((bsz,), const_theta, device=dev, dtype=calc_t)
    this_var = torch.zeros((bsz,), device=dev, dtype=calc_t) if return_sigma else None
    x_a, th_a, v_a = _solve_mppca_core_torch_fp64(
        patches_t[active],
        dim=dim,
        num_samples=num_samples,
        tau_factor_sq=tau_factor_sq,
        return_sigma=return_sigma,
    )
    xest[active] = x_a
    this_theta[active] = th_a
    if this_var is not None and v_a is not None:
        this_var[active] = v_a
    return xest, this_theta, this_var


def _mppca_cpu(
    arr: np.ndarray,
    *,
    rx: int,
    ry: int,
    rz: int,
    return_sigma: bool,
    calc_dtype: np.dtype,
    out_dtype: np.dtype,
    max_workers: Optional[int] = None,
):
    sx, sy, sz, dim = arr.shape
    px, py, pz = 2 * rx + 1, 2 * ry + 1, 2 * rz + 1
    num_samples = px * py * pz
    tau_factor = 1.0 + math.sqrt(float(dim) / float(num_samples))
    tau_factor_sq = tau_factor * tau_factor
    nx_c = sx - 2 * rx
    ny_c = sy - 2 * ry
    nz_c = sz - 2 * rz

    n_workers = max(1, int(max_workers if max_workers is not None else effective_cpu_threads(None, default=8)))
    n_workers = min(n_workers, nz_c, 16)
    # Keep per-worker memory small for modest 8-16 GiB RAM machines.
    ny_step = max(1, min(ny_c, max(1, 2048 // max(1, nx_c))))

    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        threadpool_limits = None

    def _process_z_range(z_range: Tuple[int, int]):
        kz0, kz1 = z_range  # patch center k in [kz0, kz1)
        local_z_len = (kz1 - kz0) + 2 * rz
        local_thetax = np.zeros((sx, sy, local_z_len, dim), dtype=calc_dtype)
        local_theta = np.zeros((sx, sy, local_z_len, 1), dtype=calc_dtype)
        local_var = np.zeros((sx, sy, local_z_len), dtype=calc_dtype) if return_sigma else None

        def _inner():
            for k in range(kz0, kz1):
                z_rel = k - kz0
                slab = np.ascontiguousarray(arr[:, :, k - rz : k + rz + 1, :], dtype=calc_dtype)
                win = sliding_window_view(slab, (px, py, pz), axis=(0, 1, 2))
                # win shape: (nx_c, ny_c, 1, dim, px, py, pz)
                for y0 in range(0, ny_c, ny_step):
                    y1 = min(ny_c, y0 + ny_step)
                    ny_b = y1 - y0
                    sub = np.moveaxis(win[:, y0:y1, 0, :, :, :, :], 2, -1)
                    # sub shape: (nx_c, ny_b, px, py, pz, dim)
                    patches = np.ascontiguousarray(sub.reshape(nx_c * ny_b, num_samples, dim))
                    xest, this_theta, this_var = _solve_mppca_batch_numpy(
                        patches,
                        dim=dim,
                        num_samples=num_samples,
                        tau_factor_sq=tau_factor_sq,
                        calc_dtype=calc_dtype,
                        return_sigma=return_sigma,
                    )
                    w_grid = this_theta.reshape(nx_c, ny_b)
                    wx_grid = (xest * this_theta[:, None, None]).reshape(nx_c, ny_b, px, py, pz, dim)
                    wv_grid = (this_var * this_theta).reshape(nx_c, ny_b) if (return_sigma and this_var is not None) else None
                    for dx in range(px):
                        xs = slice(dx, dx + nx_c)
                        for dy in range(py):
                            ys = slice(y0 + dy, y1 + dy)
                            for dz in range(pz):
                                z_dst = z_rel + dz
                                local_thetax[xs, ys, z_dst, :] += wx_grid[:, :, dx, dy, dz, :]
                                local_theta[xs, ys, z_dst, 0] += w_grid
                                if local_var is not None and wv_grid is not None:
                                    local_var[xs, ys, z_dst] += wv_grid

        if threadpool_limits is not None and n_workers > 1:
            with threadpool_limits(limits=1):
                _inner()
        else:
            _inner()
        return kz0, local_thetax, local_theta, local_var

    z_starts = np.linspace(rz, sz - rz, num=n_workers + 1, dtype=int)
    ranges = [(int(z_starts[i]), int(z_starts[i + 1])) for i in range(n_workers) if z_starts[i + 1] > z_starts[i]]

    thetax = np.zeros((sx, sy, sz, dim), dtype=calc_dtype)
    theta = np.zeros((sx, sy, sz, 1), dtype=calc_dtype)
    var_acc = np.zeros((sx, sy, sz), dtype=calc_dtype) if return_sigma else None

    if len(ranges) == 1:
        kz0, l_tx, l_th, l_v = _process_z_range(ranges[0])
        z_slice = slice(kz0 - rz, kz0 - rz + l_tx.shape[2])
        thetax[:, :, z_slice, :] += l_tx
        theta[:, :, z_slice, :] += l_th
        if var_acc is not None and l_v is not None:
            var_acc[:, :, z_slice] += l_v
    else:
        with ThreadPoolExecutor(max_workers=len(ranges)) as pool:
            for kz0, l_tx, l_th, l_v in pool.map(_process_z_range, ranges):
                z_slice = slice(kz0 - rz, kz0 - rz + l_tx.shape[2])
                thetax[:, :, z_slice, :] += l_tx
                theta[:, :, z_slice, :] += l_th
                if var_acc is not None and l_v is not None:
                    var_acc[:, :, z_slice] += l_v

    denoised = thetax / theta
    np.clip(denoised, 0.0, None, out=denoised)
    if return_sigma and var_acc is not None:
        sigma = np.sqrt(np.maximum(var_acc / theta[..., 0], 0.0))
        return denoised.astype(out_dtype, copy=False), sigma.astype(calc_dtype, copy=False)
    return denoised.astype(out_dtype, copy=False)


def _mppca_cuda(
    arr: np.ndarray,
    *,
    rx: int,
    ry: int,
    rz: int,
    return_sigma: bool,
    calc_dtype: np.dtype,
    out_dtype: np.dtype,
):
    import torch

    sx, sy, sz, dim = arr.shape
    px, py, pz = 2 * rx + 1, 2 * ry + 1, 2 * rz + 1
    num_samples = px * py * pz
    tau_factor = 1.0 + math.sqrt(float(dim) / float(num_samples))
    tau_factor_sq = tau_factor * tau_factor
    nx_c = sx - 2 * rx
    ny_c = sy - 2 * ry

    dev = torch.device("cuda")
    torch_dtype = torch.float64 if calc_dtype == np.float64 else torch.float32
    elem_bytes = 8 if torch_dtype == torch.float64 else 4
    per_patch_bytes = max(4096, num_samples * max(dim, num_samples) * elem_bytes * 6)
    patch_batch = adaptive_cuda_scaled_batch(
        per_item_bytes=per_patch_bytes,
        default_batch=4096,
        min_batch=256,
        max_batch=8192,
    )
    ny_step = max(1, min(ny_c, max(1, patch_batch // max(1, nx_c))))

    denoised_out = np.zeros((sx, sy, sz, dim), dtype=out_dtype)
    sigma_out = np.zeros((sx, sy, sz), dtype=calc_dtype) if return_sigma else None

    # Rolling ring buffer of thickness pz (= 5 slices) on GPU: uses ~150 MB VRAM instead of ~7 GB!
    ring_thetax = torch.zeros((sx, sy, pz, dim), device=dev, dtype=torch_dtype)
    ring_theta = torch.zeros((sx, sy, pz, 1), device=dev, dtype=torch_dtype)
    ring_var = torch.zeros((sx, sy, pz), device=dev, dtype=torch_dtype) if return_sigma else None

    def _flush_ring_slice(slot: int, z_global: int) -> None:
        denom = ring_theta[:, :, slot, :]
        den_slice = torch.clamp(ring_thetax[:, :, slot, :] / denom, min=0.0)
        denoised_out[:, :, z_global, :] = den_slice.detach().cpu().numpy().astype(out_dtype, copy=False)
        if sigma_out is not None and ring_var is not None:
            sig_slice = torch.sqrt(torch.clamp(ring_var[:, :, slot] / denom[..., 0], min=0.0))
            sigma_out[:, :, z_global] = sig_slice.detach().cpu().numpy().astype(calc_dtype, copy=False)
        ring_thetax[:, :, slot, :].zero_()
        ring_theta[:, :, slot, :].zero_()
        if ring_var is not None:
            ring_var[:, :, slot].zero_()

    prev_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for k in range(rz, sz - rz):
            slab = np.ascontiguousarray(arr[:, :, k - rz : k + rz + 1, :], dtype=calc_dtype)
            slab_t = torch.from_numpy(slab).to(device=dev, dtype=torch_dtype)
            # Extract patches via zero-copy unfold: (nx_c, ny_c, 1, dim, px, py, pz)
            win_t = slab_t.unfold(0, px, 1).unfold(1, py, 1).unfold(2, pz, 1)
            for y0 in range(0, ny_c, ny_step):
                y1 = min(ny_c, y0 + ny_step)
                ny_b = y1 - y0
                # Permute to (nx_c, ny_b, px, py, pz, dim)
                sub_t = win_t[:, y0:y1, 0, :, :, :, :].permute(0, 1, 3, 4, 5, 2).contiguous()
                patches_t = sub_t.view(nx_c * ny_b, num_samples, dim)
                xest, this_theta, this_var = _solve_mppca_batch_torch(
                    patches_t,
                    dim=dim,
                    num_samples=num_samples,
                    tau_factor_sq=tau_factor_sq,
                    return_sigma=return_sigma,
                )
                w_grid = this_theta.view(nx_c, ny_b)
                wx_grid = (xest * this_theta[:, None, None]).view(nx_c, ny_b, px, py, pz, dim)
                wv_grid = (this_var * this_theta).view(nx_c, ny_b) if (return_sigma and this_var is not None) else None
                for dz in range(pz):
                    slot = (k - rz + dz) % pz
                    for dx in range(px):
                        xs = slice(dx, dx + nx_c)
                        for dy in range(py):
                            ys = slice(y0 + dy, y1 + dy)
                            ring_thetax[xs, ys, slot, :] += wx_grid[:, :, dx, dy, dz, :]
                            ring_theta[xs, ys, slot, 0] += w_grid
                            if ring_var is not None and wv_grid is not None:
                                ring_var[xs, ys, slot] += wv_grid

            # Once k completes, global z = k - rz will never be touched by future k' > k!
            if k < sz - rz - 1:
                done_z = k - rz
                _flush_ring_slice(done_z % pz, done_z)
            else:
                # Final k = sz - rz - 1: flush all remaining pz slices [sz - 2*rz - 1 .. sz - 1]
                for dz in range(pz):
                    done_z = k - rz + dz
                    _flush_ring_slice(done_z % pz, done_z)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
        del ring_thetax, ring_theta, ring_var
        empty_cuda_cache_safely()

    if return_sigma and sigma_out is not None:
        return denoised_out, sigma_out
    return denoised_out


def mppca_accelerated(
    arr: np.ndarray,
    *,
    patch_radius: int = 2,
    return_sigma: bool = False,
    device: Optional[str] = None,
    max_workers: Optional[int] = None,
):
    """Hardware-adaptive CPU/CUDA drop-in replacement for ``dipy.denoise.localpca.mppca``."""
    arr_np = np.asarray(arr)
    if arr_np.ndim != 4:
        raise ValueError(f"PCA denoising expects a 4D array, got shape {arr_np.shape}")

    r_arr = np.array([patch_radius, patch_radius, patch_radius], dtype=int)
    r_arr[np.asarray(arr_np.shape[:3]) == 1] = 0
    ps = 2 * r_arr + 1
    if int(np.prod(ps)) <= 1 or np.any((np.asarray(arr_np.shape[:3]) != 1) & (np.asarray(arr_np.shape[:3]) < ps)):
        from dipy.denoise.localpca import mppca

        return mppca(arr_np, patch_radius=patch_radius, return_sigma=return_sigma, suppress_warning=True)

    calc_dtype = np.dtype(np.float64 if arr_np.dtype == np.float64 else np.float32)
    out_dtype = arr_np.dtype
    rx, ry, rz = int(r_arr[0]), int(r_arr[1]), int(r_arr[2])

    resolved = resolve_dipy_accel_device(device)
    explicit_cuda = str(device or os.environ.get("SLICERDSPACE_DIPY_DEVICE", "auto")).strip().lower() == "cuda"
    if resolved == "cuda":
        try:
            return _mppca_cuda(
                arr_np,
                rx=rx,
                ry=ry,
                rz=rz,
                return_sigma=return_sigma,
                calc_dtype=calc_dtype,
                out_dtype=out_dtype,
            )
        except Exception as exc:
            if is_cuda_oom(exc) or not explicit_cuda:
                empty_cuda_cache_safely()
            else:
                raise

    return _mppca_cpu(
        arr_np,
        rx=rx,
        ry=ry,
        rz=rz,
        return_sigma=return_sigma,
        calc_dtype=calc_dtype,
        out_dtype=out_dtype,
        max_workers=max_workers,
    )


def _gibbs_weights_numpy(shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    g0 = np.zeros(shape, dtype=np.float64)
    g1 = np.zeros(shape, dtype=np.float64)
    k0 = np.linspace(-np.pi, np.pi, num=shape[0])
    k1 = np.linspace(-np.pi, np.pi, num=shape[1])
    k1_m, k0_m = np.meshgrid(k1[1:-1], k0[1:-1])
    cosk0 = 1.0 + np.cos(k0_m)
    cosk1 = 1.0 + np.cos(k1_m)
    g1[1:-1, 1:-1] = cosk0 / (cosk0 + cosk1)
    g0[1:-1, 1:-1] = cosk1 / (cosk0 + cosk1)
    g1[1:-1, 0] = g1[1:-1, -1] = 1.0
    g1[0, 0] = g1[-1, -1] = g1[0, -1] = g1[-1, 0] = 0.5
    g0[0, 1:-1] = g0[-1, 1:-1] = 1.0
    g0[0, 0] = g0[-1, -1] = g0[0, -1] = g0[-1, 0] = 0.5
    return g0, g1


def _tv_axis_last_numpy(xs: np.ndarray, n_points: int) -> np.ndarray:
    pad = n_points + 1
    n_len = xs.shape[-1]
    xp = np.concatenate((xs[..., -pad:], xs, xs[..., :pad]), axis=-1)
    ptv = np.abs(xp[..., pad : pad + n_len] - xp[..., pad + 1 : pad + 1 + n_len])
    ntv = np.abs(xp[..., pad : pad + n_len] - xp[..., pad - 1 : pad - 1 + n_len])
    for n in range(1, n_points):
        ptv = ptv + np.abs(xp[..., pad + n : pad + n + n_len] - xp[..., pad + 1 + n : pad + 1 + n + n_len])
        ntv = ntv + np.abs(xp[..., pad - n : pad - n + n_len] - xp[..., pad - 1 - n : pad - 1 - n + n_len])
    return np.minimum(ptv, ntv)


def _gibbs_1d_batch_numpy(x: np.ndarray, *, axis: int, n_points: int, dtype_float: np.dtype) -> np.ndarray:
    import scipy.fft as sfft

    ssamp = np.linspace(0.02, 0.9, num=45, dtype=dtype_float)
    xs = x.copy() if axis == 1 else np.swapaxes(x, -2, -1).copy()
    tvp = _tv_axis_last_numpy(xs, n_points)
    tvn = tvp.copy()
    isp = xs.copy()
    isn = xs.copy()
    sp = np.zeros(xs.shape, dtype=dtype_float)
    sn = np.zeros(xs.shape, dtype=dtype_float)
    n_len = xs.shape[-1]
    c_fft = sfft.fft(xs, axis=-1)
    k_vec = sfft.fftfreq(n_len, 1.0 / (2.0j * np.pi)).astype(c_fft.dtype, copy=False)

    for s_val in ssamp:
        phase = np.exp(k_vec * s_val)
        img_p = np.abs(sfft.ifft(c_fft * phase, axis=-1))
        tvs_p = _tv_axis_last_numpy(img_p, n_points)
        img_n = np.abs(sfft.ifft(c_fft * np.conj(phase), axis=-1))
        tvs_n = _tv_axis_last_numpy(img_n, n_points)

        mask_p = tvp > tvs_p
        np.copyto(isp, img_p, where=mask_p)
        sp[mask_p] = s_val
        np.copyto(tvp, tvs_p, where=mask_p)

        mask_n = tvn > tvs_n
        np.copyto(isn, img_n, where=mask_n)
        sn[mask_n] = s_val
        np.copyto(tvn, tvs_n, where=mask_n)

    denom = sp + sn
    nz = denom != 0
    xs[nz] = (isp[nz] - isn[nz]) / denom[nz] * sn[nz] + isn[nz]
    return xs if axis == 1 else np.swapaxes(xs, -2, -1)


def _gibbs_2d_batch_numpy(batch: np.ndarray, *, n_points: int, g0: np.ndarray, g1: np.ndarray) -> np.ndarray:
    import scipy.fft as sfft

    dtype_float = np.promote_types(batch.real.dtype, np.float32)
    work = np.asarray(batch, dtype=dtype_float)
    img_c1 = _gibbs_1d_batch_numpy(work, axis=1, n_points=n_points, dtype_float=dtype_float)
    img_c0 = _gibbs_1d_batch_numpy(work, axis=0, n_points=n_points, dtype_float=dtype_float)
    c1 = sfft.fft2(img_c1, axes=(-2, -1))
    c0 = sfft.fft2(img_c0, axes=(-2, -1))
    combined = sfft.fftshift(c1, axes=(-2, -1)) * g1[None, :, :] + sfft.fftshift(c0, axes=(-2, -1)) * g0[None, :, :]
    out = np.abs(sfft.ifft2(combined, axes=(-2, -1)))
    return out.astype(batch.dtype, copy=False)


def _tv_axis_last_torch(xs, n_points: int):
    import torch

    pad = n_points + 1
    n_len = xs.shape[-1]
    xp = torch.cat((xs[..., -pad:], xs, xs[..., :pad]), dim=-1)
    ptv = torch.abs(xp[..., pad : pad + n_len] - xp[..., pad + 1 : pad + 1 + n_len])
    ntv = torch.abs(xp[..., pad : pad + n_len] - xp[..., pad - 1 : pad - 1 + n_len])
    for n in range(1, n_points):
        ptv = ptv + torch.abs(xp[..., pad + n : pad + n + n_len] - xp[..., pad + 1 + n : pad + 1 + n + n_len])
        ntv = ntv + torch.abs(xp[..., pad - n : pad - n + n_len] - xp[..., pad - 1 - n : pad - 1 - n + n_len])
    return torch.minimum(ptv, ntv)


def _gibbs_1d_batch_torch(x_t, *, axis: int, n_points: int):
    import scipy.fft as sfft
    import torch

    dev = x_t.device
    calc_t = x_t.dtype
    np_float = np.float64 if calc_t == torch.float64 else np.float32
    ssamp = np.linspace(0.02, 0.9, num=45, dtype=np_float)
    xs = x_t.clone() if axis == 1 else x_t.transpose(-2, -1).contiguous()
    tvp = _tv_axis_last_torch(xs, n_points)
    tvn = tvp.clone()
    isp = xs.clone()
    isn = xs.clone()
    sp = torch.zeros_like(xs)
    sn = torch.zeros_like(xs)
    n_len = xs.shape[-1]
    c_fft = torch.fft.fft(xs, dim=-1)
    k_np = sfft.fftfreq(n_len, 1.0 / (2.0j * np.pi))
    k_vec = torch.as_tensor(k_np, device=dev, dtype=c_fft.dtype)

    for s_val in ssamp:
        s_float = float(s_val)
        phase = torch.exp(k_vec * s_float)
        img_p = torch.abs(torch.fft.ifft(c_fft * phase, dim=-1))
        tvs_p = _tv_axis_last_torch(img_p, n_points)
        img_n = torch.abs(torch.fft.ifft(c_fft * torch.conj(phase), dim=-1))
        tvs_n = _tv_axis_last_torch(img_n, n_points)

        mask_p = tvp > tvs_p
        isp = torch.where(mask_p, img_p, isp)
        sp = torch.where(mask_p, torch.full_like(sp, s_float), sp)
        tvp = torch.where(mask_p, tvs_p, tvp)

        mask_n = tvn > tvs_n
        isn = torch.where(mask_n, img_n, isn)
        sn = torch.where(mask_n, torch.full_like(sn, s_float), sn)
        tvn = torch.where(mask_n, tvs_n, tvn)

    denom = sp + sn
    nz = denom != 0
    interp = torch.where(nz, (isp - isn) / torch.where(nz, denom, torch.ones_like(denom)) * sn + isn, xs)
    return interp if axis == 1 else interp.transpose(-2, -1).contiguous()


def gibbs_removal_accelerated(
    vol: np.ndarray,
    *,
    slice_axis: int = 2,
    n_points: int = 3,
    inplace: bool = False,
    device: Optional[str] = None,
    max_workers: Optional[int] = None,
) -> np.ndarray:
    """Hardware-adaptive CPU/CUDA drop-in replacement for ``dipy.denoise.gibbs.gibbs_removal``."""
    arr = vol if inplace else np.array(vol, copy=True)
    nd = arr.ndim
    if nd < 2 or nd > 4:
        raise ValueError(f"Data have to be a 4D, 3D or 2D matrix, got shape {arr.shape}")
    if slice_axis > 2:
        raise ValueError("slice_axis must be 0, 1, or 2")

    if nd == 2:
        g0, g1 = _gibbs_weights_numpy(arr.shape)
        arr[:, :] = _gibbs_2d_batch_numpy(arr[None, :, :], n_points=n_points, g0=g0, g1=g1)[0]
        return arr

    if nd == 3:
        work = np.moveaxis(arr, slice_axis, 0)
        inishap = work.shape
        flat = work
    else:
        work = np.moveaxis(arr, (slice_axis, 3), (0, 1))
        inishap = work.shape
        flat = work.reshape((inishap[0] * inishap[1], inishap[2], inishap[3]))

    h_dim, w_dim = int(flat.shape[1]), int(flat.shape[2])
    n_slices = int(flat.shape[0])
    g0, g1 = _gibbs_weights_numpy((h_dim, w_dim))

    resolved = resolve_dipy_accel_device(device)
    explicit_cuda = str(device or os.environ.get("SLICERDSPACE_DIPY_DEVICE", "auto")).strip().lower() == "cuda"

    def _run_cpu_batches(indices_slice: slice) -> np.ndarray:
        sub = np.ascontiguousarray(flat[indices_slice])
        n_sub = sub.shape[0]
        n_workers = max(1, int(max_workers if max_workers is not None else effective_cpu_threads(None, default=8)))
        n_workers = min(n_workers, max(1, (n_sub + 31) // 32), 16)
        batch_sz = 32
        batches = [(s, min(n_sub, s + batch_sz)) for s in range(0, n_sub, batch_sz)]
        out_sub = np.empty_like(sub)

        def _worker(bounds: Tuple[int, int]):
            b0, b1 = bounds
            return b0, b1, _gibbs_2d_batch_numpy(sub[b0:b1], n_points=n_points, g0=g0, g1=g1)

        if n_workers == 1 or len(batches) == 1:
            for b in batches:
                b0, b1, res = _worker(b)
                out_sub[b0:b1] = res
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                for b0, b1, res in pool.map(_worker, batches):
                    out_sub[b0:b1] = res
        return out_sub

    if resolved == "cuda":
        import torch

        dev = torch.device("cuda")
        torch_dtype = torch.float64 if flat.dtype == np.float64 else torch.float32
        elem_bytes = 8 if torch_dtype == torch.float64 else 4
        per_slice_bytes = max(4096, h_dim * w_dim * elem_bytes * 24)
        init_batch = adaptive_cuda_scaled_batch(
            per_item_bytes=per_slice_bytes,
            default_batch=512,
            min_batch=32,
            max_batch=1024,
        )
        g0_t = torch.as_tensor(g0, device=dev, dtype=torch_dtype).unsqueeze(0)
        g1_t = torch.as_tensor(g1, device=dev, dtype=torch_dtype).unsqueeze(0)

        def _cuda_chunk(start: int, end: int) -> np.ndarray:
            sub_np = np.ascontiguousarray(flat[start:end])
            x_t = torch.from_numpy(sub_np).to(device=dev, dtype=torch_dtype)
            img_c1 = _gibbs_1d_batch_torch(x_t, axis=1, n_points=n_points)
            img_c0 = _gibbs_1d_batch_torch(x_t, axis=0, n_points=n_points)
            c1 = torch.fft.fft2(img_c1, dim=(-2, -1))
            c0 = torch.fft.fft2(img_c0, dim=(-2, -1))
            comb = torch.fft.fftshift(c1, dim=(-2, -1)) * g1_t + torch.fft.fftshift(c0, dim=(-2, -1)) * g0_t
            res = torch.abs(torch.fft.ifft2(comb, dim=(-2, -1)))
            return res.detach().cpu().numpy().astype(flat.dtype, copy=False)

        def _cpu_fallback_chunk(start: int, end: int) -> np.ndarray:
            if explicit_cuda:
                raise RuntimeError("Gibbs CUDA chunk failed under explicit CUDA mode.")
            return _run_cpu_batches(slice(start, end))

        try:
            chunks = run_cuda_chunked_with_oom_fallback(
                n_slices,
                initial_chunk_size=init_batch,
                min_cuda_chunk_size=16,
                cuda_fn=_cuda_chunk,
                cpu_fallback_fn=None if explicit_cuda else _cpu_fallback_chunk,
            )
            flat[:, :, :] = np.concatenate(chunks, axis=0) if chunks else flat
        finally:
            del g0_t, g1_t
            empty_cuda_cache_safely()
    else:
        flat[:, :, :] = _run_cpu_batches(slice(0, n_slices))

    if nd == 3:
        out_vol = np.moveaxis(flat, 0, slice_axis)
    else:
        out_vol = np.moveaxis(flat.reshape(inishap), (0, 1), (slice_axis, 3))
    if inplace:
        vol[...] = out_vol
        return vol
    return out_vol
