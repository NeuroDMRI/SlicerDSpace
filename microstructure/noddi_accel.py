"""Computation-preserving CPU and CUDA acceleration for AMICO NODDI fitting.

Implements the exact 3-step AMICO NODDI dictionary solver (unregularized NNLS
isotropic pre-fit, column-normalized elastic-net non-negative Lasso support
selection, and support-restricted debiasing NNLS) in IEEE float64 using batched
active-set Lawson-Hanson quadratic programming on CUDA or multi-threaded CPU,
alongside masked-only signal normalization, parallel OLS tensor orientation
estimation, and concurrent fast-gzip NIfTI export.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import time
from typing import Callable, Optional

import nibabel as nib
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional runtime dependency
    torch = None  # type: ignore[assignment]


VALID_NODDI_DEVICES = {"auto", "cpu", "cuda"}
_KERNEL_MEM_CACHE: dict[str, dict[str, object]] = {}


def resolve_noddi_device(requested: str | None = "auto") -> str:
    req = (requested or "auto").strip().lower()
    if req == "auto":
        req = os.environ.get("SLICERDSPACE_NODDI_DEVICE", "auto").strip().lower()
    if req not in VALID_NODDI_DEVICES:
        raise ValueError(f"noddi_device must be one of {sorted(VALID_NODDI_DEVICES)}, got {requested!r}")
    if req == "cpu":
        return "cpu"
    cuda_ok = bool(torch is not None and torch.cuda.is_available())
    if req == "cuda" and not cuda_ok:
        raise RuntimeError("NODDI CUDA requested, but torch.cuda.is_available() is False.")
    return "cuda" if cuda_ok else "cpu"


def save_nifti_fast(image: nib.Nifti1Image, output_path: Path, compresslevel: int = 1) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    path_str = str(output_path)
    if path_str.endswith(".nii.gz"):
        with open(output_path, "wb", buffering=4 * 1024 * 1024) as raw_fh:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_fh, compresslevel=compresslevel, mtime=0) as gz_fh:
                image.to_file_map({"image": nib.FileHolder(fileobj=gz_fh)})
    else:
        nib.save(image, path_str)


def _try_fast_gunzip_nifti1_4d(dwi_path: Path) -> tuple[nib.Nifti1Image, np.ndarray] | None:
    """Fast streaming gunzip for standard little-endian NIfTI-1 .nii.gz files."""
    if not dwi_path.name.endswith(".nii.gz"):
        return None
    decompressor = shutil.which("pigz") or shutil.which("gzip")
    if not decompressor:
        return None
    try:
        hdr_img = nib.load(str(dwi_path))
        if not isinstance(hdr_img, nib.Nifti1Image):
            return None
        hdr = hdr_img.header
        if hdr.endianness != "<" or len(hdr_img.shape) != 4:
            return None
        vox_offset = int(hdr["vox_offset"])
        if vox_offset < 352:
            return None
        dtype = np.dtype(hdr.get_data_dtype()).newbyteorder("<")
        expected_voxels = int(np.prod(hdr_img.shape, dtype=np.int64))
        expected_bytes = vox_offset + expected_voxels * dtype.itemsize
        proc = subprocess.run(
            [decompressor, "-dc", str(dwi_path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        buf = proc.stdout
        if len(buf) < expected_bytes:
            return None
        raw_arr = np.ndarray(
            shape=hdr_img.shape,
            dtype=dtype,
            buffer=buf,
            offset=vox_offset,
            order="F",
        )
        return hdr_img, raw_arr
    except Exception:
        return None


def _load_masked_normalized_dwi(
    *,
    dwi_path: Path,
    mask_path: Path,
    scheme,
) -> tuple[nib.Nifti1Image, nib.Nifti1Image, np.ndarray, np.ndarray]:
    """Load DWI + mask and reproduce AMICO Evaluation.load_data normalization on masked voxels."""
    mask_img = nib.load(str(mask_path))
    if mask_img.ndim != 3:
        raise ValueError("MASK file is not a 3D image")
    mask_arr = (np.asarray(mask_img.dataobj) > 0).astype(np.uint8)
    mask_bool = mask_arr == 1

    fast_loaded = _try_fast_gunzip_nifti1_4d(dwi_path)
    if fast_loaded is not None:
        dwi_img, raw_4d = fast_loaded
        hdr = dwi_img.header
        slope = float(hdr["scl_slope"]) if np.isfinite(hdr["scl_slope"]) else 0.0
        inter = float(hdr["scl_inter"]) if np.isfinite(hdr["scl_inter"]) else 0.0
        need_scale = slope != 0.0 and (slope != 1.0 or inter != 0.0)
        if np.issubdtype(raw_4d.dtype, np.floating) and not np.isfinite(raw_4d).all():
            raise ValueError("NaN or Inf values detected in raw DWI signal.")
        y_f32 = np.asarray(raw_4d[mask_bool], dtype=np.float32)
        if need_scale:
            y_f32 = y_f32 * np.float32(slope) + np.float32(inter)
    else:
        dwi_img = nib.load(str(dwi_path))
        if dwi_img.ndim != 4:
            raise ValueError("DWI file is not a 4D image")
        full_f32 = dwi_img.get_fdata(dtype=np.float32)
        if not np.isfinite(full_f32).all():
            raise ValueError("NaN or Inf values detected in raw DWI signal.")
        y_f32 = np.asarray(full_f32[mask_bool], dtype=np.float32)

    if dwi_img.shape[:3] != mask_img.shape[:3]:
        raise ValueError("MASK geometry does not match with DWI data")
    if int(scheme.nS) != int(dwi_img.shape[3]):
        raise ValueError("Scheme does not match with DWI data")
    if int(scheme.b0_count) <= 0:
        raise ValueError("No b0 volume to normalize signal with")

    if not np.isfinite(y_f32).all():
        raise ValueError("NaN or Inf values detected in scaled DWI signal.")

    b0_idx = np.asarray(scheme.b0_idx, dtype=np.int64)
    mean_b0 = np.mean(y_f32[:, b0_idx], axis=1, dtype=np.float32)
    norm_factor = np.zeros_like(mean_b0, dtype=np.float32)
    pos = mean_b0 > 0.0
    norm_factor[pos] = np.float32(1.0) / mean_b0[pos]
    y_f32 *= norm_factor[:, None]

    if not np.isfinite(y_f32).all():
        raise ValueError("NaN or Inf values in signal after b0 normalization.")

    y_f64 = y_f32.astype(np.float64)
    y_f64[y_f64 < 0.0] = 0.0
    return dwi_img, mask_img, mask_arr, y_f64


def _estimate_principal_directions_ols(
    y: np.ndarray,
    scheme,
    *,
    nthreads: int,
) -> np.ndarray:
    """Exact parallel equivalent of DIPY TensorModel(gtab, fit_method='OLS').fit(y).directions."""
    import dipy.reconst.dti as dti
    from dipy.core.gradients import gradient_table

    gtab = gradient_table(bvals=scheme.b, bvecs=scheme.raw[:, :3])
    n_vox = y.shape[0]
    if n_vox == 0:
        return np.zeros((0, 3), dtype=np.float32)

    W = dti.design_matrix(gtab)
    pinv_WT = np.ascontiguousarray(np.linalg.pinv(W).T, dtype=np.float64)
    min_diff = float(1e-6 / -W.min())

    dirs = np.empty((n_vox, 3), dtype=np.float32)
    strictly_pos = np.all(y > 0.0, axis=1)
    pos_idx = np.flatnonzero(strictly_pos)
    zero_idx = np.flatnonzero(~strictly_pos)

    if pos_idx.size > 0:
        workers = max(1, min(int(nthreads), (pos_idx.size + 16383) // 16384))
        chunk_sz = (pos_idx.size + workers - 1) // workers

        def _solve_chunk(start: int, stop: int) -> tuple[int, int, np.ndarray]:
            rows = pos_idx[start:stop]
            log_s = np.log(y[rows])
            fit_lo = log_s @ pinv_WT
            dti_params = dti.eig_from_lo_tri(fit_lo, min_diffusivity=min_diff)
            evecs = dti_params.reshape(-1, 4, 3)[:, 1:, :]
            return start, stop, np.asarray(evecs[:, :, 0], dtype=np.float32)

        if workers == 1:
            _, _, chunk_dirs = _solve_chunk(0, pos_idx.size)
            dirs[pos_idx] = chunk_dirs
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = [
                    pool.submit(_solve_chunk, s, min(s + chunk_sz, pos_idx.size))
                    for s in range(0, pos_idx.size, chunk_sz)
                ]
                for fut in futs:
                    s, e, chunk_dirs = fut.result()
                    dirs[pos_idx[s:e]] = chunk_dirs

    if zero_idx.size > 0:
        tm = dti.TensorModel(gtab, fit_method="OLS")
        fallback_dirs = np.asarray(tm.fit(y[zero_idx]).directions, dtype=np.float32)
        if fallback_dirs.ndim == 3:
            fallback_dirs = fallback_dirs[:, 0, :]
        dirs[zero_idx] = fallback_dirs

    return dirs


def _dirs_to_lut_indices(dirs: np.ndarray, htable: np.ndarray, amico_lut) -> np.ndarray:
    """Vectorized spherical hash-table lookup matching AMICO's Cython lut indexing."""
    if dirs.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    ht2 = np.asarray(htable).reshape(181, 181)
    v = np.asarray(dirs, dtype=np.float64).copy()
    flip = v[:, 1] < 0.0
    v[flip] *= -1.0
    _, theta, phi = amico_lut.cart2sphere(
        np.ascontiguousarray(v[:, 0]),
        np.ascontiguousarray(v[:, 1]),
        np.ascontiguousarray(v[:, 2]),
    )
    i0 = np.floor(np.rad2deg(theta) + 0.5).astype(np.int64) % 181
    i1 = np.floor(np.rad2deg(phi) + 0.5).astype(np.int64) % 181
    return np.asarray(ht2[i0, i1], dtype=np.int64)


def _solve_qp_bucket_numpy(
    G: np.ndarray,
    rhs: np.ndarray,
    allowed_mask: np.ndarray | None = None,
    tol: float = 1e-11,
) -> np.ndarray:
    """Batched Lawson-Hanson active-set non-negative QP for one direction bucket in FP64."""
    B, P = rhs.shape
    x_full = np.zeros((B, P), dtype=np.float64)
    if B == 0:
        return x_full

    active = np.arange(B, dtype=np.int64)
    rhs_act = rhs.copy()
    x = np.zeros((B, P), dtype=np.float64)
    passive = np.zeros((B, P), dtype=bool)
    forbid = (~allowed_mask).copy() if allowed_mask is not None else None
    all_cols = np.arange(P, dtype=np.int64)

    for _ in range(P * 2):
        B_a = active.size
        if B_a == 0:
            break
        counts = passive.sum(axis=1)
        k_max = int(counts.max())
        if k_max == 0:
            grad = rhs_act.copy()
        elif k_max >= 32:
            grad = rhs_act - x @ G
        else:
            if k_max < P:
                idx_pad = np.argpartition(-passive.view(np.int8), kth=k_max - 1, axis=1)[:, :k_max]
            else:
                idx_pad = np.broadcast_to(all_cols, (B_a, P))
            row_a = np.arange(B_a)[:, None]
            valid_slot = passive[row_a, idx_pad]
            x_act = np.where(valid_slot, x[row_a, idx_pad], 0.0)
            grad = rhs_act - np.einsum("bk,bkp->bp", x_act, G[idx_pad])

        if forbid is not None:
            grad[passive | forbid] = -np.inf
        else:
            grad[passive] = -np.inf

        best_col = np.argmax(grad, axis=1)
        max_g = grad[np.arange(B_a), best_col]
        keep = max_g > tol
        if not np.all(keep):
            done = ~keep
            x_full[active[done]] = x[done]
            if not np.any(keep):
                active = active[:0]
                break
            active = active[keep]
            rhs_act = rhs_act[keep]
            x = x[keep]
            passive = passive[keep]
            best_col = best_col[keep]
            if forbid is not None:
                forbid = forbid[keep]

        B_a = active.size
        passive[np.arange(B_a), best_col] = True

        in_idx = np.arange(B_a, dtype=np.int64)
        for _ in range(P * 2):
            B_i = in_idx.size
            if B_i == 0:
                break
            pas_sub = passive[in_idx]
            cnt_sub = pas_sub.sum(axis=1)
            k_i = int(cnt_sub.max())
            if k_i == 0:
                break
            if k_i < P:
                cols_pad = np.argpartition(-pas_sub.view(np.int8), kth=k_i - 1, axis=1)[:, :k_i]
            else:
                cols_pad = np.broadcast_to(all_cols, (B_i, P))
            row_i = np.arange(B_i)[:, None]
            valid = pas_sub[row_i, cols_pad]

            G_sub = G[cols_pad[:, :, None], cols_pad[:, None, :]].copy()
            valid_2d = valid[:, :, None] & valid[:, None, :]
            eye_k = np.eye(k_i, dtype=np.float64)[None, :, :]
            G_sub = np.where(valid_2d, G_sub, eye_k)
            rhs_sub = np.where(valid, rhs_act[in_idx[:, None], cols_pad], 0.0)

            try:
                z_pad = np.linalg.solve(G_sub, rhs_sub[..., None])[..., 0]
            except np.linalg.LinAlgError:
                G_reg = G_sub + (1e-13 * eye_k)
                z_pad = np.linalg.solve(G_reg, rhs_sub[..., None])[..., 0]

            feas = np.all((z_pad > 0.0) | (~valid), axis=1)
            if np.any(feas):
                f_rows = in_idx[feas]
                x[f_rows] = 0.0
                x[f_rows[:, None], cols_pad[feas]] = np.where(valid[feas], z_pad[feas], 0.0)
            if np.all(feas):
                break

            infeas = ~feas
            sub_rows = in_idx[infeas]
            cols_inf = cols_pad[infeas]
            val_inf = valid[infeas]
            z_inf = z_pad[infeas]
            x_cur = x[sub_rows[:, None], cols_inf]
            neg_m = val_inf & (z_inf <= 0.0)
            denom = np.maximum(x_cur - z_inf, 1e-30)
            ratio = np.where(neg_m, x_cur / denom, np.inf)
            alpha = np.min(ratio, axis=1, keepdims=True)
            x_new = x_cur + alpha * (z_inf - x_cur)
            still_pos = val_inf & (x_new > 1e-15)
            # Guarantee at least one blocking constraint drops if alpha == 0
            dropped = val_inf & (~still_pos)
            no_drop = ~np.any(dropped, axis=1)
            if np.any(no_drop):
                nd_rows = np.flatnonzero(no_drop)
                for r in nd_rows:
                    cand = np.flatnonzero(neg_m[r])
                    if cand.size > 0:
                        still_pos[r, cand[int(np.argmin(ratio[r, cand]))]] = False

            x[sub_rows] = 0.0
            passive[sub_rows] = False
            x[sub_rows[:, None], cols_inf] = np.where(still_pos, x_new, 0.0)
            passive[sub_rows[:, None], cols_inf] = still_pos
            in_idx = sub_rows

    if active.size > 0:
        x_full[active] = x
    return x_full


def _solve_qp_batched_torch(
    G_all: "torch.Tensor",
    d_idx: "torch.Tensor",
    rhs: "torch.Tensor",
    allowed_mask: "torch.Tensor | None" = None,
    tol: float = 1e-11,
) -> "torch.Tensor":
    """Batched Lawson-Hanson active-set non-negative QP across direction buckets on GPU in FP64."""
    B, P = rhs.shape
    device = rhs.device
    x_full = torch.zeros((B, P), dtype=torch.float64, device=device)
    if B == 0:
        return x_full

    active = torch.arange(B, dtype=torch.int64, device=device)
    d_act = d_idx
    rhs_act = rhs
    x = torch.zeros((B, P), dtype=torch.float64, device=device)
    passive = torch.zeros((B, P), dtype=torch.bool, device=device)
    forbid = (~allowed_mask) if allowed_mask is not None else None

    for _ in range(P * 2):
        B_a = active.numel()
        if B_a == 0:
            break
        counts = passive.sum(dim=1)
        k_max = int(counts.max().item())
        if k_max == 0:
            grad = rhs_act.clone()
        else:
            _, idx_pad = torch.topk(passive.to(torch.int8), k=k_max, dim=1, largest=True, sorted=False)
            valid_slot = torch.gather(passive, 1, idx_pad)
            x_act = torch.where(valid_slot, torch.gather(x, 1, idx_pad), 0.0)
            G_rows = G_all[d_act[:, None], idx_pad, :]
            grad = rhs_act - torch.einsum("bk,bkp->bp", x_act, G_rows)

        if forbid is not None:
            grad.masked_fill_(passive | forbid, -float("inf"))
        else:
            grad.masked_fill_(passive, -float("inf"))

        max_g, best_col = grad.max(dim=1)
        keep = max_g > tol
        if not bool(torch.all(keep).item()):
            done = ~keep
            x_full[active[done]] = x[done]
            if not bool(torch.any(keep).item()):
                active = active[:0]
                break
            active = active[keep]
            d_act = d_act[keep]
            rhs_act = rhs_act[keep]
            x = x[keep]
            passive = passive[keep]
            best_col = best_col[keep]
            if forbid is not None:
                forbid = forbid[keep]

        passive.scatter_(1, best_col[:, None], True)

        in_idx = torch.arange(active.numel(), dtype=torch.int64, device=device)
        for _ in range(P * 2):
            B_i = in_idx.numel()
            if B_i == 0:
                break
            pas_sub = passive[in_idx]
            cnt_sub = pas_sub.sum(dim=1)
            k_i = int(cnt_sub.max().item())
            if k_i == 0:
                break
            _, cols_pad = torch.topk(pas_sub.to(torch.int8), k=k_i, dim=1, largest=True, sorted=False)
            valid = torch.gather(pas_sub, 1, cols_pad)

            G_sub = G_all[d_act[in_idx][:, None, None], cols_pad[:, :, None], cols_pad[:, None, :]]
            valid_2d = valid[:, :, None] & valid[:, None, :]
            eye_k = torch.eye(k_i, dtype=torch.float64, device=device).unsqueeze(0)
            G_sub = torch.where(valid_2d, G_sub, eye_k)
            rhs_sub = torch.where(valid, torch.gather(rhs_act[in_idx], 1, cols_pad), 0.0)

            z_pad, info = torch.linalg.solve_ex(G_sub, rhs_sub)
            if bool(torch.any(info != 0).item()):
                bad = info != 0
                z_pad[bad] = torch.linalg.solve(G_sub[bad] + 1e-13 * eye_k, rhs_sub[bad])

            feas = torch.all((z_pad > 0.0) | (~valid), dim=1)
            if bool(torch.any(feas).item()):
                f_rows = in_idx[feas]
                x_f = torch.zeros((f_rows.numel(), P), dtype=torch.float64, device=device)
                x_f.scatter_(1, cols_pad[feas], torch.where(valid[feas], z_pad[feas], 0.0))
                x[f_rows] = x_f
            if bool(torch.all(feas).item()):
                break

            infeas = ~feas
            sub_rows = in_idx[infeas]
            cols_inf = cols_pad[infeas]
            val_inf = valid[infeas]
            z_inf = z_pad[infeas]
            x_cur = torch.gather(x[sub_rows], 1, cols_inf)
            neg_m = val_inf & (z_inf <= 0.0)
            denom = torch.clamp(x_cur - z_inf, min=1e-30)
            ratio = torch.where(neg_m, x_cur / denom, float("inf"))
            alpha, min_pos = ratio.min(dim=1, keepdim=True)
            x_new = x_cur + alpha * (z_inf - x_cur)
            still_pos = val_inf & (x_new > 1e-15)
            # Ensure progress when alpha == 0 on a newly added non-positive variable
            no_drop = ~torch.any(val_inf & (~still_pos), dim=1, keepdim=True)
            if bool(torch.any(no_drop).item()):
                force_drop = torch.zeros_like(still_pos)
                force_drop.scatter_(1, min_pos, no_drop)
                still_pos = still_pos & (~force_drop)

            x_upd = torch.zeros((sub_rows.numel(), P), dtype=torch.float64, device=device)
            pas_upd = torch.zeros((sub_rows.numel(), P), dtype=torch.bool, device=device)
            x_upd.scatter_(1, cols_inf, torch.where(still_pos, x_new, 0.0))
            pas_upd.scatter_(1, cols_inf, still_pos)
            x[sub_rows] = x_upd
            passive[sub_rows] = pas_upd
            in_idx = sub_rows

    if active.numel() > 0:
        x_full[active] = x
    return x_full


def _fit_noddi_cpu(
    *,
    y: np.ndarray,
    dir_idx: np.ndarray,
    K_wm: np.ndarray,
    K_iso: np.ndarray,
    K_norms: np.ndarray,
    icvf: np.ndarray,
    kappa: np.ndarray,
    dwi_idx: np.ndarray,
    lambda1: float,
    lambda2: float,
    ndirs: int,
    nthreads: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_vox, n_grad = y.shape
    n_wm = K_wm.shape[0]
    P1 = n_wm + 1
    estimates = np.zeros((n_vox, 3), dtype=np.float32)
    rmse = np.zeros(n_vox, dtype=np.float32)
    nrmse = np.zeros(n_vox, dtype=np.float32)
    if n_vox == 0:
        return estimates, rmse, nrmse

    order = np.argsort(dir_idx, kind="stable")
    sorted_d = dir_idx[order]
    boundaries = np.searchsorted(sorted_d, np.arange(ndirs + 1))

    K_iso_64 = K_iso.astype(np.float64)
    K_iso_dwi = K_iso_64[dwi_idx]
    eye_wm = np.eye(n_wm, dtype=np.float64)
    nonempty_dirs = [d for d in range(ndirs) if boundaries[d + 1] > boundaries[d]]

    def _solve_direction_bucket(d: int) -> None:
        s, e = int(boundaries[d]), int(boundaries[d + 1])
        rows = order[s:e]
        Y_d = y[rows]

        A1 = np.empty((n_grad, P1), dtype=np.float64)
        A1[:, :n_wm] = K_wm[:, d, :].T
        A1[:, n_wm] = K_iso_64
        G1 = A1.T @ A1

        rhs1 = Y_d @ A1
        c1 = _solve_qp_bucket_numpy(G1, rhs1)

        y2 = Y_d[:, dwi_idx] - c1[:, n_wm : n_wm + 1] * K_iso_dwi[None, :]
        np.maximum(y2, 0.0, out=y2)

        A2 = A1[dwi_idx, :n_wm] * K_norms
        G2 = A2.T @ A2 + lambda2 * eye_wm
        rhs2 = y2 @ A2 - lambda1
        x2 = _solve_qp_bucket_numpy(G2, rhs2)

        supp = np.empty((e - s, P1), dtype=bool)
        supp[:, :n_wm] = x2 > 0.0
        supp[:, n_wm] = True
        c3 = _solve_qp_bucket_numpy(G1, rhs1, allowed_mask=supp)

        w_wm = c3[:, :n_wm]
        s_wm = np.sum(w_wm, axis=1) + 1e-16
        s_tot = np.sum(c3, axis=1) + 1e-16
        f_iso = c3[:, n_wm] / s_tot
        f_ic = (w_wm @ icvf) / s_wm
        f_od = (2.0 / np.pi) * np.arctan2(1.0, (w_wm @ kappa) / s_wm)

        est_d = np.empty((e - s, 3), dtype=np.float32)
        est_d[:, 0] = f_ic
        est_d[:, 1] = f_od
        est_d[:, 2] = f_iso
        estimates[rows] = est_d

        res = Y_d - (c3 @ A1.T)
        ss_res = np.sum(res * res, axis=1)
        rmse[rows] = np.sqrt(ss_res / float(n_grad)).astype(np.float32)
        ss_y = np.sum(Y_d * Y_d, axis=1)
        nrmse_d = np.zeros(e - s, dtype=np.float64)
        pos_y = ss_y > 1e-16
        nrmse_d[pos_y] = np.sqrt(ss_res[pos_y] / ss_y[pos_y])
        nrmse[rows] = nrmse_d.astype(np.float32)

    workers = max(1, min(int(nthreads), len(nonempty_dirs)))
    if workers == 1:
        for d in nonempty_dirs:
            _solve_direction_bucket(d)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_solve_direction_bucket, nonempty_dirs))

    return estimates, rmse, nrmse


def _fit_noddi_cuda(
    *,
    y: np.ndarray,
    dir_idx: np.ndarray,
    K_wm: np.ndarray,
    K_iso: np.ndarray,
    K_norms: np.ndarray,
    icvf: np.ndarray,
    kappa: np.ndarray,
    dwi_idx: np.ndarray,
    lambda1: float,
    lambda2: float,
    ndirs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert torch is not None
    n_vox, n_grad = y.shape
    n_wm = K_wm.shape[0]
    P1 = n_wm + 1
    estimates = np.zeros((n_vox, 3), dtype=np.float32)
    rmse = np.zeros(n_vox, dtype=np.float32)
    nrmse = np.zeros(n_vox, dtype=np.float32)
    if n_vox == 0:
        return estimates, rmse, nrmse

    dev = torch.device("cuda:0")
    with torch.no_grad():
        # Build (ndirs, n_grad, P1) and Gram matrices on GPU in FP64
        wm_t = torch.as_tensor(K_wm, dtype=torch.float64, device=dev).permute(1, 2, 0).contiguous()
        iso_t = torch.as_tensor(K_iso, dtype=torch.float64, device=dev)
        A1_all = torch.empty((ndirs, n_grad, P1), dtype=torch.float64, device=dev)
        A1_all[:, :, :n_wm] = wm_t
        A1_all[:, :, n_wm] = iso_t.unsqueeze(0)
        G1_all = torch.bmm(A1_all.transpose(1, 2), A1_all)

        dwi_idx_t = torch.as_tensor(dwi_idx, dtype=torch.int64, device=dev)
        norms_t = torch.as_tensor(K_norms, dtype=torch.float64, device=dev)
        A2_all = A1_all.index_select(1, dwi_idx_t)[:, :, :n_wm] * norms_t.unsqueeze(0)
        eye_wm = torch.eye(n_wm, dtype=torch.float64, device=dev).unsqueeze(0)
        G2_all = torch.bmm(A2_all.transpose(1, 2), A2_all) + lambda2 * eye_wm

        iso_dwi_t = iso_t.index_select(0, dwi_idx_t)
        icvf_t = torch.as_tensor(icvf, dtype=torch.float64, device=dev)
        kappa_t = torch.as_tensor(kappa, dtype=torch.float64, device=dev)

        # Sort voxels by direction so RHS GEMMs can operate on contiguous slices
        order = np.argsort(dir_idx, kind="stable")
        sorted_d = dir_idx[order]
        Y_sorted = np.ascontiguousarray(y[order], dtype=np.float64)

        chunk_size = 65536
        est_sorted = np.empty((n_vox, 3), dtype=np.float32)
        rmse_sorted = np.empty(n_vox, dtype=np.float32)
        nrmse_sorted = np.empty(n_vox, dtype=np.float32)

        for start in range(0, n_vox, chunk_size):
            stop = min(start + chunk_size, n_vox)
            B = stop - start
            Y_c = torch.as_tensor(Y_sorted[start:stop], dtype=torch.float64, device=dev)
            d_c = torch.as_tensor(sorted_d[start:stop], dtype=torch.int64, device=dev)

            # Grouped RHS1 computation across unique directions in this sorted chunk
            rhs1 = torch.empty((B, P1), dtype=torch.float64, device=dev)
            uniq_d, counts_d = torch.unique_consecutive(d_c, return_counts=True)
            offsets_d = torch.zeros(uniq_d.numel() + 1, dtype=torch.int64, device=dev)
            torch.cumsum(counts_d, dim=0, out=offsets_d[1:])
            offs_cpu = offsets_d.cpu().tolist()
            uniq_cpu = uniq_d.cpu().tolist()
            for idx_u, d_val in enumerate(uniq_cpu):
                s_u, e_u = offs_cpu[idx_u], offs_cpu[idx_u + 1]
                rhs1[s_u:e_u] = Y_c[s_u:e_u] @ A1_all[d_val]

            c1 = _solve_qp_batched_torch(G1_all, d_c, rhs1)

            y2 = torch.clamp(
                Y_c.index_select(1, dwi_idx_t) - c1[:, n_wm : n_wm + 1] * iso_dwi_t.unsqueeze(0),
                min=0.0,
            )
            rhs2 = torch.empty((B, n_wm), dtype=torch.float64, device=dev)
            for idx_u, d_val in enumerate(uniq_cpu):
                s_u, e_u = offs_cpu[idx_u], offs_cpu[idx_u + 1]
                rhs2[s_u:e_u] = y2[s_u:e_u] @ A2_all[d_val] - lambda1

            x2 = _solve_qp_batched_torch(G2_all, d_c, rhs2)

            supp = torch.empty((B, P1), dtype=torch.bool, device=dev)
            supp[:, :n_wm] = x2 > 0.0
            supp[:, n_wm] = True
            c3 = _solve_qp_batched_torch(G1_all, d_c, rhs1, allowed_mask=supp)

            w_wm = c3[:, :n_wm]
            s_wm = w_wm.sum(dim=1) + 1e-16
            s_tot = c3.sum(dim=1) + 1e-16
            f_iso = c3[:, n_wm] / s_tot
            f_ic = (w_wm @ icvf_t) / s_wm
            f_od = (2.0 / np.pi) * torch.atan2(torch.ones_like(s_wm), (w_wm @ kappa_t) / s_wm)
            est_c = torch.stack((f_ic, f_od, f_iso), dim=1).to(torch.float32)

            y_pred = torch.empty_like(Y_c)
            for idx_u, d_val in enumerate(uniq_cpu):
                s_u, e_u = offs_cpu[idx_u], offs_cpu[idx_u + 1]
                y_pred[s_u:e_u] = c3[s_u:e_u] @ A1_all[d_val].T

            res = Y_c - y_pred
            ss_res = (res * res).sum(dim=1)
            rmse_c = torch.sqrt(ss_res / float(n_grad)).to(torch.float32)
            ss_y = (Y_c * Y_c).sum(dim=1)
            nrmse_c = torch.where(ss_y > 1e-16, torch.sqrt(ss_res / torch.clamp(ss_y, min=1e-16)), 0.0).to(torch.float32)

            est_sorted[start:stop] = est_c.cpu().numpy()
            rmse_sorted[start:stop] = rmse_c.cpu().numpy()
            nrmse_sorted[start:stop] = nrmse_c.cpu().numpy()

        estimates[order] = est_sorted
        rmse[order] = rmse_sorted
        nrmse[order] = nrmse_sorted

    return estimates, rmse, nrmse


def _load_or_build_noddi_kernels(
    *,
    amico_mod,
    cache_root: Path,
    subject_id: str,
    amico_output_dir: Path,
    scheme_path: Path,
    b0_thr: int,
    lmax: int,
    ndirs: int,
    lambda1: float,
    lambda2: float,
    nthreads: int,
    ensure_rotations_fn: Callable[[int, int, Optional[Callable[[str], None]]], None],
    log: Optional[Callable[[str], None]] = None,
) -> tuple[object, dict[str, object], np.ndarray]:
    """Generate/load AMICO NODDI kernels with protocol-level memoization."""
    scheme_bytes = scheme_path.read_bytes()
    cache_key = hashlib.sha256(
        b"|".join(
            [
                str(getattr(amico_mod, "__version__", "")).encode("utf-8"),
                scheme_bytes,
                f"b0={b0_thr}|lmax={lmax}|ndirs={ndirs}|l1={lambda1}|l2={lambda2}".encode("utf-8"),
            ]
        )
    ).hexdigest()[:24]

    scheme = amico_mod.scheme.Scheme(str(scheme_path), int(b0_thr))
    mem_hit = _KERNEL_MEM_CACHE.get(cache_key)
    if mem_hit is not None:
        return scheme, mem_hit["kernels"], mem_hit["htable"]  # type: ignore[return-value]

    disk_cache = cache_root.parent / f"_noddi_resampled_lut_{cache_key}.npz"
    if disk_cache.is_file():
        try:
            with np.load(str(disk_cache), allow_pickle=False) as npz:
                kernels = {
                    "model": "NODDI",
                    "wm": npz["wm"],
                    "iso": npz["iso"],
                    "kappa": npz["kappa"],
                    "icvf": npz["icvf"],
                    "norms": npz["norms"],
                }
                htable = npz["htable"]
            _KERNEL_MEM_CACHE[cache_key] = {"kernels": kernels, "htable": htable}
            return scheme, kernels, htable
        except Exception:
            pass

    ensure_rotations_fn(int(lmax), int(ndirs), log)
    ev = amico_mod.Evaluation(
        study_path=str(cache_root),
        subject=subject_id,
        output_path=str(amico_output_dir),
    )
    ev.set_config("doDirectionalAverage", False)
    ev.set_config("doMergeB0", False)
    ev.set_config("BLAS_nthreads", max(1, int(nthreads)))
    ev.scheme = scheme
    ev.set_model("NODDI")
    ev.set_solver(lambda1=float(lambda1), lambda2=float(lambda2))
    try:
        ev.generate_kernels(regenerate=False, lmax=int(lmax), ndirs=int(ndirs))
    except Exception as exc:
        msg = str(exc)
        if "Auxiliary matrices not found" not in msg and "amico.setup()" not in msg:
            raise
        amico_mod.setup(lmax=int(lmax))
        ev.generate_kernels(regenerate=False, lmax=int(lmax), ndirs=int(ndirs))
    ev.load_kernels()
    kernels = dict(ev.KERNELS)
    htable = np.asarray(ev.htable).copy()

    try:
        disk_cache.parent.mkdir(parents=True, exist_ok=True)
        tmp_npz = disk_cache.with_name(f".tmp_{os.getpid()}_{disk_cache.name}")
        np.savez(
            str(tmp_npz),
            wm=kernels["wm"],
            iso=kernels["iso"],
            kappa=kernels["kappa"],
            icvf=kernels["icvf"],
            norms=kernels["norms"],
            htable=htable,
        )
        os.replace(tmp_npz, disk_cache)
    except OSError:
        pass

    _KERNEL_MEM_CACHE[cache_key] = {"kernels": kernels, "htable": htable}
    return scheme, kernels, htable


def run_accelerated_noddi_fit(
    *,
    amico_mod,
    config,
    cache_root: Path,
    dwi_path: Path,
    mask_path: Path,
    scheme_path: Path,
    amico_output_dir: Path,
    nthreads: int,
    resolved_device: str,
    ensure_rotations_fn: Callable[[int, int, Optional[Callable[[str], None]]], None],
    progress_cb: Optional[Callable[[str, str], None]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> tuple[nib.Nifti1Image, np.ndarray, dict[str, Path], dict[str, np.ndarray], dict[str, object]]:
    """Run accelerated NODDI fitting and write AMICO-compatible raw output NIfTIs."""
    t0 = time.monotonic()
    if progress_cb is not None:
        progress_cb("running", "loading_data")

    scheme = amico_mod.scheme.Scheme(str(scheme_path), int(config.noddi_b0_threshold))
    dwi_img, mask_img, mask_arr, y = _load_masked_normalized_dwi(
        dwi_path=dwi_path,
        mask_path=mask_path,
        scheme=scheme,
    )
    t_load = time.monotonic() - t0

    if progress_cb is not None:
        progress_cb("running", "generating_kernels")
    t1 = time.monotonic()
    scheme, kernels, htable = _load_or_build_noddi_kernels(
        amico_mod=amico_mod,
        cache_root=cache_root,
        subject_id=config.subject_id,
        amico_output_dir=amico_output_dir,
        scheme_path=scheme_path,
        b0_thr=int(config.noddi_b0_threshold),
        lmax=int(config.noddi_lmax),
        ndirs=int(config.noddi_ndirs),
        lambda1=float(config.noddi_solver_lambda1),
        lambda2=float(config.noddi_solver_lambda2),
        nthreads=int(nthreads),
        ensure_rotations_fn=ensure_rotations_fn,
        log=log,
    )
    t_kern = time.monotonic() - t1

    if progress_cb is not None:
        progress_cb("running", "fitting_model")
    t2 = time.monotonic()
    dirs_masked = _estimate_principal_directions_ols(y, scheme, nthreads=int(nthreads))
    dir_idx = _dirs_to_lut_indices(dirs_masked, htable, amico_mod.lut)

    K_wm = np.asarray(kernels["wm"], dtype=np.float32)
    K_iso = np.asarray(kernels["iso"], dtype=np.float32)
    K_norms = np.asarray(kernels["norms"], dtype=np.float64)
    icvf = np.asarray(kernels["icvf"], dtype=np.float64)
    kappa = np.asarray(kernels["kappa"], dtype=np.float64)
    dwi_idx = np.asarray(scheme.dwi_idx, dtype=np.int64)

    if resolved_device == "cuda":
        estimates, rmse_v, nrmse_v = _fit_noddi_cuda(
            y=y,
            dir_idx=dir_idx,
            K_wm=K_wm,
            K_iso=K_iso,
            K_norms=K_norms,
            icvf=icvf,
            kappa=kappa,
            dwi_idx=dwi_idx,
            lambda1=float(config.noddi_solver_lambda1),
            lambda2=float(config.noddi_solver_lambda2),
            ndirs=int(config.noddi_ndirs),
        )
    else:
        estimates, rmse_v, nrmse_v = _fit_noddi_cpu(
            y=y,
            dir_idx=dir_idx,
            K_wm=K_wm,
            K_iso=K_iso,
            K_norms=K_norms,
            icvf=icvf,
            kappa=kappa,
            dwi_idx=dwi_idx,
            lambda1=float(config.noddi_solver_lambda1),
            lambda2=float(config.noddi_solver_lambda2),
            ndirs=int(config.noddi_ndirs),
            nthreads=int(nthreads),
        )
    t_fit = time.monotonic() - t2

    if progress_cb is not None:
        progress_cb("running", "saving_results")
    t3 = time.monotonic()

    vol_shape = mask_arr.shape
    mask_bool = mask_arr == 1
    ndi_map = np.zeros(vol_shape, dtype=np.float32)
    odi_map = np.zeros(vol_shape, dtype=np.float32)
    fwf_map = np.zeros(vol_shape, dtype=np.float32)
    rmse_map = np.zeros(vol_shape, dtype=np.float32)
    nrmse_map = np.zeros(vol_shape, dtype=np.float32)
    dir_map = np.zeros(vol_shape + (3,), dtype=np.float32)

    ndi_map[mask_bool] = estimates[:, 0]
    odi_map[mask_bool] = estimates[:, 1]
    fwf_map[mask_bool] = estimates[:, 2]
    rmse_map[mask_bool] = rmse_v
    nrmse_map[mask_bool] = nrmse_v
    dir_map[mask_bool] = dirs_masked

    amico_output_dir.mkdir(parents=True, exist_ok=True)
    for stale in amico_output_dir.glob("*"):
        if stale.is_file():
            stale.unlink(missing_ok=True)

    amico_ver = str(getattr(amico_mod, "__version__", "2.0"))
    cfg_record = {
        "version": amico_ver,
        "study_path": str(cache_root),
        "subject": config.subject_id,
        "OUTPUT_path": str(amico_output_dir),
        "b0_thr": int(config.noddi_b0_threshold),
        "lmax": int(config.noddi_lmax),
        "ndirs": int(config.noddi_ndirs),
        "nthreads": int(nthreads),
        "device": resolved_device,
        "fit_time": t_fit,
        "solver_params": {
            "lambda1": float(config.noddi_solver_lambda1),
            "lambda2": float(config.noddi_solver_lambda2),
        },
    }
    with (amico_output_dir / "config.pickle").open("wb") as fh:
        pickle.dump(cfg_record, fh, protocol=2)

    affine = dwi_img.affine
    base_hdr = dwi_img.header.copy()
    base_hdr.set_data_dtype(np.float32)

    def _build_map_img(
        arr: np.ndarray,
        cal_min: float,
        cal_max: float,
        descrip: str | None = None,
    ) -> nib.Nifti1Image:
        hdr = base_hdr.copy()
        img = nib.Nifti1Image(arr, affine, hdr)
        img.header["cal_min"] = float(cal_min)
        img.header["cal_max"] = float(cal_max)
        img.header["scl_slope"] = 1.0
        img.header["scl_inter"] = 0.0
        if descrip:
            img.header["descrip"] = descrip.encode("ascii", errors="ignore")[:79]
        return img

    noddi_sources = {
        "noddi_ndi": amico_output_dir / "fit_NDI.nii.gz",
        "noddi_odi": amico_output_dir / "fit_ODI.nii.gz",
        "noddi_fwf": amico_output_dir / "fit_FWF.nii.gz",
        "noddi_rmse": amico_output_dir / "fit_RMSE.nii.gz",
        "noddi_nrmse": amico_output_dir / "fit_NRMSE.nii.gz",
    }
    write_jobs = [
        (_build_map_img(dir_map, -1.0, 1.0), amico_output_dir / "fit_dir.nii.gz"),
        (_build_map_img(rmse_map, 0.0, 1.0), noddi_sources["noddi_rmse"]),
        (_build_map_img(nrmse_map, 0.0, 1.0), noddi_sources["noddi_nrmse"]),
        (
            _build_map_img(ndi_map, float(ndi_map.min()), float(ndi_map.max()), f"Neurite Density Index (AMICO v{amico_ver})"),
            noddi_sources["noddi_ndi"],
        ),
        (
            _build_map_img(odi_map, float(odi_map.min()), float(odi_map.max()), f"Orientation Dispersion Index (AMICO v{amico_ver})"),
            noddi_sources["noddi_odi"],
        ),
        (
            _build_map_img(fwf_map, float(fwf_map.min()), float(fwf_map.max()), f"Free Water Fraction (AMICO v{amico_ver})"),
            noddi_sources["noddi_fwf"],
        ),
    ]
    with ThreadPoolExecutor(max_workers=min(6, max(1, int(nthreads)))) as pool:
        futs = [pool.submit(save_nifti_fast, img, dst, 1) for img, dst in write_jobs]
        for fut in futs:
            fut.result()

    t_save = time.monotonic() - t3
    source_arrays = {
        "noddi_ndi": ndi_map,
        "noddi_odi": odi_map,
        "noddi_fwf": fwf_map,
        "noddi_rmse": rmse_map,
        "noddi_nrmse": nrmse_map,
    }
    exec_meta = {
        "engine": f"batched_lawson_hanson_fp64_{resolved_device}",
        "device": resolved_device,
        "nthreads": int(nthreads),
        "load_seconds": round(t_load, 4),
        "kernel_seconds": round(t_kern, 4),
        "fit_seconds": round(t_fit, 4),
        "save_seconds": round(t_save, 4),
    }
    return mask_img, mask_bool, noddi_sources, source_arrays, exec_meta
