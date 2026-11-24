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
    GIB,
    MIB,
    adaptive_cuda_scaled_batch,
    adaptive_gpu_scaled_batch,
    empty_cuda_cache_safely,
    empty_gpu_cache_safely,
    is_apple_silicon_host,
    is_cuda_oom,
    is_mps_available,
    optimal_cpu_torch_threads,
    query_cuda_free_total_bytes,
    query_host_available_ram_bytes,
    run_cuda_chunked_with_oom_fallback,
)
from shared.numeric_threads import effective_cpu_threads, reserved_stage_cpus


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
        if sys.platform == "win32":
            for env_key in ("CUDA_PATH", "CUDA_HOME"):
                env_dir = os.environ.get(env_key, "").strip()
                if env_dir:
                    candidates.extend(sorted(glob.glob(os.path.join(env_dir, "bin", "cusolver64_*.dll"))))
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
            dll_handle = None
            try:
                if sys.platform == "win32":
                    adder = getattr(os, "add_dll_directory", None)
                    parent_dir = os.path.dirname(path)
                    if callable(adder) and parent_dir and os.path.isdir(parent_dir):
                        try:
                            dll_handle = adder(parent_dir)
                        except OSError:
                            dll_handle = None
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
                if dll_handle is not None:
                    _CUSOLVER_HANDLES[("_dll_dir", path)] = dll_handle
                break
            except Exception:
                if dll_handle is not None:
                    try:
                        dll_handle.close()
                    except Exception:
                        pass
                continue
            else:
                if dll_handle is not None:
                    try:
                        dll_handle.close()
                    except Exception:
                        pass
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
    if os.name == "nt" and cuda_dt == 1:
        from shared.hardware_adapt import is_consumer_fp64_cuda_gpu

        # Consumer GeForce RTX GPUs have 1/64 FP64 throughput; bound batched 125x125 FP64
        # cuSOLVER launches to stay well below Windows' 2-second WDDM TDR limit.
        tdr_sub_cap = 256 if is_consumer_fp64_cuda_gpu(dev) else 1024
        max_sub = min(max_sub, tdr_sub_cap)
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


_MPS_FFT_FP32_SUPPORTED: Optional[bool] = None


def _mps_supports_fft_float32() -> bool:
    """Return True if the active PyTorch MPS runtime supports float32/complex64 FFT."""
    global _MPS_FFT_FP32_SUPPORTED
    if _MPS_FFT_FP32_SUPPORTED is not None:
        return _MPS_FFT_FP32_SUPPORTED
    try:
        import torch

        if not is_mps_available(torch):
            _MPS_FFT_FP32_SUPPORTED = False
            return False
        probe = torch.zeros((2, 2), dtype=torch.float32, device="mps")
        freq = torch.fft.fft2(probe, dim=(-2, -1))
        _ = torch.fft.ifft2(freq, dim=(-2, -1)).real
        _ = torch.as_tensor(np.zeros((2,), dtype=np.complex64), device="mps", dtype=torch.complex64)
        _MPS_FFT_FP32_SUPPORTED = True
    except Exception:
        _MPS_FFT_FP32_SUPPORTED = False
    return _MPS_FFT_FP32_SUPPORTED


def resolve_dipy_accel_device(requested: Optional[str] = None, *, for_gibbs: bool = False) -> str:
    value = str(
        requested
        if requested is not None
        else os.environ.get("SLICERDSPACE_DIPY_DEVICE", "auto")
    ).strip().lower()
    if value not in {"auto", "cpu", "cuda", "mps"} and not value.startswith("cuda:"):
        raise ValueError(f"Unsupported DIPY preprocessing device {value!r}; expected 'auto', 'cpu', 'mps', or 'cuda'.")
    if value == "cpu":
        return "cpu"
    if value == "mps":
        if for_gibbs and _mps_supports_fft_float32():
            return "mps"
        return "cpu"
    try:
        import torch

        cuda_ok = bool(torch.cuda.is_available())
    except Exception:
        cuda_ok = False
    if (value == "cuda" or value.startswith("cuda:")) and not cuda_ok:
        raise RuntimeError("DIPY CUDA acceleration was explicitly requested, but PyTorch CUDA is unavailable.")
    if cuda_ok:
        if value.startswith("cuda:") and value != "cuda:0":
            return value
        return "cuda"
    if for_gibbs and _mps_supports_fft_float32():
        return "mps"
    return "cpu"


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


# CPU MP-PCA patch batch per worker, sized by bytes like the CUDA path (``_mppca_cuda``).
# Measured 2026-11-22 in Slicer's x86_64 (Rosetta, OpenBLAS Nehalem) PythonSlicer, 8 workers, old and
# new runs interleaved on a machine with background load (the ranges cover the repeated runs; the
# times vary with the load): the former fixed ~2048-patch batch peaked at 12.7-14.9 GB RSS / 50-72 s at
# 288 volumes (145x32x14 slab) and 9.9-12.6 GB / 33-61 s at 100 volumes (64x64x32), because every
# worker holds about six (batch, 125, volumes) arrays. Batches of about 140-240 patches gave
# 3.0-3.2 GB / 28-39 s and 2.1-2.3 GB / 29-42 s; in each interleaved pair the smaller batch was not
# slower. The denoised output differs by at most 6e-7 of the signal maximum (31, 100 and 288 volumes).
# 192 MiB gives 268 patches at <= 125 volumes and
# 116 at 288. The batch is cut in whole x-rows of patch centres, so one call holds at least one row
# (``nx_c`` patches) even when that exceeds the budget. Only the float32 accumulation order changes.
MPPCA_CPU_BATCH_BYTES = 192 * MIB
_MPPCA_CPU_MIN_PATCH_BATCH = 64
_MPPCA_CPU_MAX_PATCH_BATCH = 2048  # the former fixed batch; no input gets a larger batch than before


def _low_ram_worker_cap(avail_ram_bytes: int) -> Optional[int]:
    """Return a conservative CPU worker ceiling for tight available host RAM, or None when ample."""
    if avail_ram_bytes <= 0:
        return None
    if avail_ram_bytes < 2 * GIB:
        return 1
    if avail_ram_bytes < 3 * GIB:
        return 2
    if avail_ram_bytes < 6 * GIB:
        return 4
    if avail_ram_bytes < 10 * GIB:
        return 6
    return None


def _mppca_cpu_batch_bytes(avail_ram_bytes: Optional[int] = None) -> int:
    """Return per-worker CPU MP-PCA byte budget, scaling down only when available host RAM < 10 GiB."""
    avail = query_host_available_ram_bytes() if avail_ram_bytes is None else int(avail_ram_bytes)
    if avail > 0:
        if avail < 2 * GIB:
            return 32 * MIB
        if avail < 6 * GIB:
            return 64 * MIB
        if avail < 10 * GIB:
            return 96 * MIB
    return MPPCA_CPU_BATCH_BYTES


def _mppca_cpu_patch_batch(dim: int, num_samples: int, *, batch_bytes: Optional[int] = None) -> int:
    """Patches per CPU solver call for ``dim`` volumes and ``num_samples`` voxels per patch."""
    per_patch_bytes = max(1, int(num_samples)) * max(int(dim), int(num_samples), 1) * 8 * 6
    budget = MPPCA_CPU_BATCH_BYTES if batch_bytes is None else max(1, int(batch_bytes))
    min_batch = (
        _MPPCA_CPU_MIN_PATCH_BATCH
        if budget >= MPPCA_CPU_BATCH_BYTES
        else max(16, min(_MPPCA_CPU_MIN_PATCH_BATCH, (_MPPCA_CPU_MIN_PATCH_BATCH * budget) // MPPCA_CPU_BATCH_BYTES))
    )
    batch = budget // per_patch_bytes
    return int(max(min_batch, min(_MPPCA_CPU_MAX_PATCH_BATCH, batch)))


def _resolve_dipy_cpu_workers(
    max_workers: Optional[int],
    *,
    avail_ram_bytes: Optional[int] = None,
    apply_low_ram_cap: bool = False,
    allow_unsafe_ram_override: bool = False,
) -> int:
    """Resolve CPU worker count with low-RAM protection and Apple Silicon P-core cap, zero server regression."""
    if max_workers is not None:
        n_workers = max(1, int(max_workers))
    else:
        stage_reserved = reserved_stage_cpus()
        if stage_reserved is not None:
            n_workers = max(1, int(stage_reserved))
        elif is_apple_silicon_host():
            n_workers = optimal_cpu_torch_threads(None, max_conv_threads=8)
        else:
            default_workers = 16 if (os.cpu_count() or 1) >= 32 else 8
            n_workers = effective_cpu_threads(None, default=default_workers)
    if apply_low_ram_cap and not allow_unsafe_ram_override:
        avail = query_host_available_ram_bytes() if avail_ram_bytes is None else int(avail_ram_bytes)
        ram_cap = _low_ram_worker_cap(avail)
        if ram_cap is not None:
            n_workers = min(n_workers, ram_cap)
    return max(1, int(n_workers))


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
    patch_batch: Optional[int] = None,
):
    """CPU MP-PCA. ``patch_batch`` is internal (tests): None = byte-budgeted batch, int = fixed patches per call."""
    sx, sy, sz, dim = arr.shape
    px, py, pz = 2 * rx + 1, 2 * ry + 1, 2 * rz + 1
    num_samples = px * py * pz
    tau_factor = 1.0 + math.sqrt(float(dim) / float(num_samples))
    tau_factor_sq = tau_factor * tau_factor
    nx_c = sx - 2 * rx
    ny_c = sy - 2 * ry
    nz_c = sz - 2 * rz

    avail_ram = query_host_available_ram_bytes()
    n_workers = _resolve_dipy_cpu_workers(max_workers, avail_ram_bytes=avail_ram, apply_low_ram_cap=True)
    n_workers = min(n_workers, nz_c, 16)
    # Per-worker patch batch from the byte budget (MPPCA_CPU_BATCH_BYTES, scaled down under tight RAM), in whole x-rows.
    if patch_batch is None:
        patch_batch = _mppca_cpu_patch_batch(dim, num_samples, batch_bytes=_mppca_cpu_batch_bytes(avail_ram))
    ny_step = max(1, min(ny_c, max(1, int(patch_batch) // max(1, nx_c))))

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
        # Scope BLAS/OpenMP single-threading ONCE outside the thread pool; calling
        # threadpool_limits inside worker threads races on process-global OpenBLAS state.
        from contextlib import nullcontext

        blas_ctx = threadpool_limits(limits=1) if threadpool_limits is not None else nullcontext()
        with blas_ctx, ThreadPoolExecutor(max_workers=len(ranges)) as pool:
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
    device: str = "cuda",
):
    import torch

    sx, sy, sz, dim = arr.shape
    px, py, pz = 2 * rx + 1, 2 * ry + 1, 2 * rz + 1
    num_samples = px * py * pz
    tau_factor = 1.0 + math.sqrt(float(dim) / float(num_samples))
    tau_factor_sq = tau_factor * tau_factor
    nx_c = sx - 2 * rx
    ny_c = sy - 2 * ry

    dev = torch.device(device)
    torch_dtype = torch.float64 if calc_dtype == np.float64 else torch.float32
    # _solve_mppca_core_torch_fp64 always promotes patches to float64 (8 bytes) for exact eigendecomposition.
    per_patch_bytes = max(4096, num_samples * max(dim, num_samples) * 8 * 6)
    patch_batch = adaptive_cuda_scaled_batch(
        per_item_bytes=per_patch_bytes,
        default_batch=4096,
        min_batch=256,
        max_batch=8192,
        device=dev,
        windows_tdr_cap=2048,
        fp64=True,
        consumer_fp64_tdr_cap=512,
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
            y0 = 0
            while y0 < ny_c:
                y1 = min(ny_c, y0 + ny_step)
                ny_b = y1 - y0
                try:
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
                    y0 = y1
                except BaseException as exc:
                    if not is_cuda_oom(exc) or ny_step <= 1:
                        raise
                    empty_cuda_cache_safely()
                    ny_step = max(1, ny_step // 2)

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


def _classify_cuda_fallback_error(exc: BaseException) -> Optional[str]:
    """Classify recoverable CUDA/GPU execution failures for ``auto`` CPU fallback.

    Returns ``"oom"``, ``"tdr_timeout"``, or ``"cuda_runtime"`` for genuine GPU/CUDA failures,
    and ``None`` for non-CUDA errors (which should propagate instead of silently restarting on CPU).
    """
    if is_cuda_oom(exc):
        return "oom"
    msg = str(exc or "").lower()
    tdr_markers = (
        "launch timed out",
        "cudaerrorlaunchtimeout",
        "unspecified launch failure",
        "the launch timed out and was terminated",
        "context is destroyed",
        "device lost",
    )
    if any(marker in msg for marker in tdr_markers):
        return "tdr_timeout"
    cls_name = type(exc).__name__.lower()
    if "accelerator" in cls_name or "cuda" in cls_name:
        return "cuda_runtime"
    if isinstance(exc, RuntimeError):
        cuda_markers = (
            "cuda",
            "cublas",
            "cusolver",
            "cufft",
            "cudnn",
            "no kernel image",
            "illegal memory access",
            "nvrtc",
            "hip error",
            "mps",
        )
        if any(marker in msg for marker in cuda_markers):
            return "cuda_runtime"
    return None


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
    req_raw = str(device or os.environ.get("SLICERDSPACE_DIPY_DEVICE", "auto")).strip().lower()
    explicit_cuda = req_raw == "cuda" or req_raw.startswith("cuda:")
    if resolved.startswith("cuda"):
        try:
            return _mppca_cuda(
                arr_np,
                rx=rx,
                ry=ry,
                rz=rz,
                return_sigma=return_sigma,
                calc_dtype=calc_dtype,
                out_dtype=out_dtype,
                device=resolved,
            )
        except Exception as exc:
            category = _classify_cuda_fallback_error(exc)
            if explicit_cuda:
                if category == "tdr_timeout":
                    print(
                        f"[DIPY MP-PCA] CUDA launch timeout / WDDM TDR on {resolved} ({exc}). "
                        "Explicit CUDA mode does not fall back to CPU; switch to auto/cpu or reduce display GPU load.",
                        file=sys.stderr,
                    )
                raise
            if category is None:
                raise
            label = {
                "oom": "CUDA out of memory",
                "tdr_timeout": "CUDA launch timeout / WDDM TDR",
                "cuda_runtime": "CUDA runtime error",
            }.get(category, "CUDA error")
            print(
                f"[DIPY MP-PCA] {label} on {resolved} ({type(exc).__name__}: {exc}); falling back to CPU.",
                file=sys.stderr,
            )
            empty_cuda_cache_safely()

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
    g0_f = g0.astype(dtype_float, copy=False)
    g1_f = g1.astype(dtype_float, copy=False)
    c1 = sfft.fft2(img_c1, axes=(-2, -1))
    del img_c1
    c0 = sfft.fft2(img_c0, axes=(-2, -1))
    del img_c0
    combined = sfft.fftshift(c1, axes=(-2, -1)) * g1_f[None, :, :] + sfft.fftshift(c0, axes=(-2, -1)) * g0_f[None, :, :]
    del c1, c0
    out = np.abs(sfft.ifft2(combined, axes=(-2, -1)))
    del combined
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
    """Hardware-adaptive CPU/CUDA drop-in replacement for ``dipy.denoise.gibbs.gibbs_removal``.

    Streams bounded 2D slice chunks directly into a single preallocated output buffer (or in-place
    into ``vol`` when ``inplace=True``) via zero-copy stride views so 4D DWI volumes never allocate
    redundant full-volume reshapes, intermediate output arrays, or concatenated chunk lists.
    """
    vol_arr = vol if isinstance(vol, np.ndarray) else np.asarray(vol)
    nd = vol_arr.ndim
    if nd < 2 or nd > 4:
        raise ValueError(f"Data have to be a 4D, 3D or 2D matrix, got shape {vol_arr.shape}")
    if slice_axis > 2:
        raise ValueError("slice_axis must be 0, 1, or 2")

    out = vol_arr if inplace else np.empty_like(vol_arr)
    if nd == 2:
        g0, g1 = _gibbs_weights_numpy(vol_arr.shape)
        out[:, :] = _gibbs_2d_batch_numpy(vol_arr[None, :, :], n_points=n_points, g0=g0, g1=g1)[0]
        return out

    if nd == 3:
        src_view = np.moveaxis(vol_arr, slice_axis, 0)
        dst_view = np.moveaxis(out, slice_axis, 0)
        n_s = int(src_view.shape[0])
        n_v = 1
        h_dim, w_dim = int(src_view.shape[1]), int(src_view.shape[2])
    else:
        src_view = np.moveaxis(vol_arr, (slice_axis, 3), (0, 1))
        dst_view = np.moveaxis(out, (slice_axis, 3), (0, 1))
        n_s = int(src_view.shape[0])
        n_v = int(src_view.shape[1])
        h_dim, w_dim = int(src_view.shape[2]), int(src_view.shape[3])

    n_slices = n_s * n_v
    if n_slices == 0:
        return out

    g0, g1 = _gibbs_weights_numpy((h_dim, w_dim))

    def _gather_slices(start: int, end: int) -> np.ndarray:
        if nd == 3:
            return np.ascontiguousarray(src_view[start:end])
        count = end - start
        v0, s0 = divmod(start, n_s)
        if s0 + count <= n_s:
            return np.ascontiguousarray(src_view[s0 : s0 + count, v0])
        buf = np.empty((count, h_dim, w_dim), dtype=vol_arr.dtype)
        pos = 0
        cur = start
        while cur < end:
            v, s = divmod(cur, n_s)
            take = min(end - cur, n_s - s)
            buf[pos : pos + take] = src_view[s : s + take, v]
            pos += take
            cur += take
        return buf

    def _scatter_slices(start: int, end: int, result: np.ndarray) -> None:
        if nd == 3:
            dst_view[start:end] = result
            return
        count = end - start
        v0, s0 = divmod(start, n_s)
        if s0 + count <= n_s:
            dst_view[s0 : s0 + count, v0] = result
            return
        pos = 0
        cur = start
        while cur < end:
            v, s = divmod(cur, n_s)
            take = min(end - cur, n_s - s)
            dst_view[s : s + take, v] = result[pos : pos + take]
            pos += take
            cur += take

    resolved = resolve_dipy_accel_device(device, for_gibbs=True)
    req_raw = str(device or os.environ.get("SLICERDSPACE_DIPY_DEVICE", "auto")).strip().lower()
    explicit_cuda = req_raw == "cuda" or req_raw.startswith("cuda:")

    done = np.zeros(n_slices, dtype=bool)

    def _run_cpu_range(start: int, end: int) -> None:
        n_sub = end - start
        if n_sub <= 0:
            return
        avail_ram = query_host_available_ram_bytes()
        batch_sz = 16 if (0 < avail_ram < 4 * GIB) else 32
        n_workers = _resolve_dipy_cpu_workers(max_workers, avail_ram_bytes=avail_ram, apply_low_ram_cap=True)
        n_workers = min(n_workers, max(1, (n_sub + batch_sz - 1) // batch_sz), 16)
        batches = [(s, min(end, s + batch_sz)) for s in range(start, end, batch_sz)]

        def _worker(bounds: Tuple[int, int]):
            b0, b1 = bounds
            sub = _gather_slices(b0, b1)
            res = _gibbs_2d_batch_numpy(sub, n_points=n_points, g0=g0, g1=g1)
            return b0, b1, res

        if n_workers == 1 or len(batches) == 1:
            for b in batches:
                b0, b1, res = _worker(b)
                _scatter_slices(b0, b1, res)
                done[b0:b1] = True
                del res
        else:
            from collections import deque
            from contextlib import nullcontext

            try:
                from threadpoolctl import threadpool_limits as _tp_limits
            except ImportError:
                _tp_limits = None
            blas_ctx = _tp_limits(limits=1) if _tp_limits is not None else nullcontext()
            max_inflight = max(n_workers, min(len(batches), n_workers * 2))
            with blas_ctx, ThreadPoolExecutor(max_workers=n_workers) as pool:
                pending: deque = deque()
                batch_iter = iter(batches)
                for _ in range(max_inflight):
                    b = next(batch_iter, None)
                    if b is None:
                        break
                    pending.append(pool.submit(_worker, b))
                while pending:
                    fut = pending.popleft()
                    b0, b1, res = fut.result()
                    _scatter_slices(b0, b1, res)
                    done[b0:b1] = True
                    del res
                    nxt = next(batch_iter, None)
                    if nxt is not None:
                        pending.append(pool.submit(_worker, nxt))

    def _run_cpu_uncompleted(start: int, end: int) -> None:
        if end <= start:
            return
        pending = np.flatnonzero(~done[start:end])
        if pending.size == 0:
            return
        splits = np.flatnonzero(np.diff(pending) > 1) + 1
        for group in np.split(pending, splits):
            _run_cpu_range(start + int(group[0]), start + int(group[-1]) + 1)

    is_cuda_dev = resolved.startswith("cuda")
    if is_cuda_dev or resolved == "mps":
        import torch

        dev = torch.device(resolved)
        torch_dtype = torch.float64 if (is_cuda_dev and vol_arr.dtype == np.float64) else torch.float32
        elem_bytes = 8 if torch_dtype == torch.float64 else 4
        per_slice_bytes = max(4096, h_dim * w_dim * elem_bytes * 24)
        if is_cuda_dev:
            init_batch = adaptive_cuda_scaled_batch(
                per_item_bytes=per_slice_bytes,
                default_batch=512,
                min_batch=32,
                max_batch=1024,
                device=dev,
                windows_tdr_cap=128,
                fp64=(torch_dtype == torch.float64),
                consumer_fp64_tdr_cap=64,
            )
        else:
            init_batch = adaptive_gpu_scaled_batch(
                per_item_bytes=per_slice_bytes,
                default_batch=128,
                min_batch=16,
                max_batch=256,
                device=dev,
            )
        g0_t = None
        g1_t = None
        try:
            g0_t = torch.as_tensor(g0, device=dev, dtype=torch_dtype).unsqueeze(0)
            g1_t = torch.as_tensor(g1, device=dev, dtype=torch_dtype).unsqueeze(0)

            def _gpu_step(start: int, end: int) -> None:
                sub_np = _gather_slices(start, end)
                x_t = torch.from_numpy(sub_np).to(device=dev, dtype=torch_dtype)
                img_c1 = _gibbs_1d_batch_torch(x_t, axis=1, n_points=n_points)
                img_c0 = _gibbs_1d_batch_torch(x_t, axis=0, n_points=n_points)
                c1 = torch.fft.fft2(img_c1, dim=(-2, -1))
                c0 = torch.fft.fft2(img_c0, dim=(-2, -1))
                comb = torch.fft.fftshift(c1, dim=(-2, -1)) * g1_t + torch.fft.fftshift(c0, dim=(-2, -1)) * g0_t
                res = torch.abs(torch.fft.ifft2(comb, dim=(-2, -1)))
                res_np = res.detach().cpu().numpy().astype(vol_arr.dtype, copy=False)
                _scatter_slices(start, end, res_np)
                done[start:end] = True

            def _cuda_step(start: int, end: int) -> None:
                try:
                    _gpu_step(start, end)
                except Exception as exc:
                    if is_cuda_oom(exc):
                        # C6/D8: release the CUDA and the MPS allocator cache before the chunk is halved and retried.
                        empty_gpu_cache_safely()
                    raise

            def _cpu_fallback_step(start: int, end: int) -> None:
                if explicit_cuda:
                    raise RuntimeError("Gibbs CUDA chunk failed under explicit CUDA mode.")
                _run_cpu_uncompleted(start, end)

            run_cuda_chunked_with_oom_fallback(
                n_slices,
                initial_chunk_size=init_batch,
                min_cuda_chunk_size=16,
                cuda_step_fn=_cuda_step,
                cpu_fallback_fn=None if explicit_cuda else _cpu_fallback_step,
            )
        except Exception as exc:
            category = _classify_cuda_fallback_error(exc)
            if explicit_cuda or category is None:
                if explicit_cuda and category == "tdr_timeout":
                    print(
                        f"[DIPY Gibbs] CUDA launch timeout / WDDM TDR on {resolved} ({exc}). "
                        "Explicit CUDA mode does not fall back to CPU; switch to auto/cpu or reduce display GPU load.",
                        file=sys.stderr,
                    )
                raise
            label = {
                "oom": "GPU out of memory",
                "tdr_timeout": "CUDA launch timeout / WDDM TDR",
                "cuda_runtime": "GPU runtime error",
            }.get(category, "GPU error")
            print(
                f"[DIPY Gibbs] {label} on {resolved} ({type(exc).__name__}: {exc}); falling back to CPU.",
                file=sys.stderr,
            )
            empty_gpu_cache_safely()
            _run_cpu_uncompleted(0, n_slices)
        finally:
            del g0_t, g1_t
            empty_gpu_cache_safely()
    else:
        _run_cpu_range(0, n_slices)

    return out
