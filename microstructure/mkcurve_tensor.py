# Python translation of lib/DKI/{dki_fit,dki_mk_only,dki_parameters}.m
# Copyright (c) 2017 New York University and University of Antwerp.
# Modifications: Python translation for SlicerDSpace, 2026.
# This file and mkcurve_directions.json are covered by MPL-2.0.
# See licenses/MK-Curve-DKI-MPL-2.0.txt. Not an official upstream Python release.
"""Unconstrained author-style WLLS and numerical kurtosis integration."""

from functools import lru_cache
from itertools import combinations_with_replacement
import json
from math import factorial
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import lstsq
from scipy.linalg.blas import dtrsm
from scipy.linalg.lapack import dgelsy, dgelsy_lwork, dgeqp3, dormqr


# These SciPy families have paired wrapper/direct-driver regression evidence.
# Unknown releases retain their public solver, not a guessed ABI/threshold.
_DIRECT_GELSY = (tuple(scipy.__version__.split('.')[:2]) in {('1', '13'), ('1', '17')}
                 or scipy.__version__ == '1.18.1')


def _design_setup(design):
    scale = np.linalg.norm(design, axis=0)
    if np.any(scale == 0) or np.linalg.cond(design) > 1e15:
        return None
    scaled = design / scale
    if np.linalg.matrix_rank(scaled) < 22:
        return None
    return scale, scaled


@lru_cache(maxsize=4)
def _cached_design_setup(shape, content):
    result = _design_setup(np.frombuffer(content, dtype=np.float64).reshape(shape))
    if result is not None:
        for array in result:
            array.setflags(write=False)
    return result


def _prepared_design(design):
    design = np.asarray(design)
    # Retain layout-dependent reduction order, and bound cache memory. The key
    # contains values, not array identity, so an edited acquisition invalidates it.
    if design.dtype == np.float64 and design.flags.c_contiguous and design.nbytes <= 1024 ** 2:
        return _cached_design_setup(design.shape, design.tobytes())
    return _design_setup(design)


@lru_cache(maxsize=4)
def _cached_ols_qr(shape, content, lwork):
    scaled = np.frombuffer(content, dtype=np.float64).reshape(shape)
    qr, jpvt, tau, _, info = dgeqp3(scaled, lwork)
    if info != 0:
        return None
    columns = shape[1]
    r_upper = np.asfortranarray(qr[:columns, :columns])
    for array in (qr, jpvt, tau, r_upper):
        array.setflags(write=False)
    return qr, jpvt, tau, r_upper


@lru_cache(maxsize=32)
def _gelsy_workspace_nrhs(rows, columns, nrhs):
    size, info = dgelsy_lwork(rows, columns, nrhs, np.finfo(np.float64).eps)
    if info != 0:
        raise ValueError('LAPACK gelsy workspace query failed')
    return int(size)


def _ols_initial(scaled, logs):
    if (not _DIRECT_GELSY or scaled.dtype != np.float64 or logs.dtype != np.float64
            or not scaled.flags.c_contiguous or scaled.ndim != 2 or logs.ndim != 2
            or scaled.shape[0] < scaled.shape[1] or scaled.nbytes > 1024 ** 2 or not len(logs)):
        return lstsq(scaled, logs.T, lapack_driver="gelsy")[0]
    rows, columns = scaled.shape
    nrhs = len(logs)
    sub_lwork = _gelsy_workspace_nrhs(rows, columns, nrhs) - columns
    factorization = _cached_ols_qr(scaled.shape, scaled.tobytes(), sub_lwork)
    if factorization is None:
        return lstsq(scaled, logs.T, lapack_driver="gelsy")[0]
    qr, jpvt, tau, r_upper = factorization
    qtb, _, info = dormqr("L", "T", qr, tau, np.asfortranarray(logs.T), sub_lwork)
    if info != 0:
        return lstsq(scaled, logs.T, lapack_driver="gelsy")[0]
    solved = dtrsm(1.0, r_upper, qtb[:columns], side=0, lower=0, trans_a=0, diag=0)
    b_full = np.empty((rows, nrhs), dtype=np.float64, order="F")
    b_full[jpvt - 1] = solved
    return b_full[:columns]


@lru_cache(maxsize=32)
def _gelsy_workspace(rows, columns):
    size, info = dgelsy_lwork(rows, columns, 1, np.finfo(np.float64).eps)
    if info != 0:
        raise ValueError('LAPACK gelsy workspace query failed')
    return int(size)


def _weighted_qr(matrix, observed):
    """Same SciPy GELSY inputs/rcond/pivots; caller has checked finiteness."""
    if (not _DIRECT_GELSY or matrix.dtype != np.float64 or observed.dtype != np.float64
            or matrix.ndim != 2 or observed.ndim != 1 or matrix.shape[0] < matrix.shape[1]):
        fit, _, rank, _ = lstsq(matrix, observed, lapack_driver='gelsy')
        return fit, rank
    rows, columns = matrix.shape
    _, solution, _, rank, info = dgelsy(
        matrix, observed, np.zeros((columns, 1), dtype=np.int32),
        np.finfo(np.float64).eps, _gelsy_workspace(rows, columns), False, False)
    if info != 0:
        raise ValueError('LAPACK gelsy failed with info=' + str(info))
    return solution[:columns], rank


@lru_cache(maxsize=8)
def _tensor_terms(order):
    indices = tuple(combinations_with_replacement(range(3), order))
    counts = tuple(factorial(order) / np.prod([factorial(index.count(j)) for j in range(3)]) for index in indices)
    return indices, counts


@lru_cache(maxsize=8)
def _tensor_terms_arrays(order):
    indices, counts = _tensor_terms(order)
    idx_array = np.asarray(indices, dtype=np.intp)
    counts_array = np.asarray(counts, dtype=np.float64)
    idx_array.setflags(write=False)
    counts_array.setflags(write=False)
    return idx_array, counts_array


def tensor_basis(directions, order):
    directions = np.asarray(directions, dtype=np.float64)
    _tensor_terms(order)
    idx_array, counts_array = _tensor_terms_arrays(order)
    return np.ascontiguousarray(np.prod(directions[..., idx_array], axis=-1) * counts_array)


@lru_cache(maxsize=1)
def integration_directions():
    directions = np.asarray(json.loads(Path(__file__).with_name("mkcurve_directions.json").read_text()), dtype=float)
    if directions.shape != (256, 3) or not np.isfinite(directions).all():
        raise ValueError("Invalid author MK-Curve integration directions")
    directions.setflags(write=False)
    return directions


@lru_cache(maxsize=1)
def _integration_bases():
    # The author's quadrature is fixed; never cache voxel-dependent fits.
    bases = tuple(tensor_basis(integration_directions(), order).T for order in (2, 4))
    for basis in bases:
        basis.setflags(write=False)
    return bases


@lru_cache(maxsize=1)
def _equator_directions():
    angles = np.arange(256) * (2 * np.pi / 256)
    equator = np.column_stack((np.cos(angles), np.sin(angles), np.zeros(256)))
    equator.setflags(write=False)
    return equator


def design_matrix(bvals, bvecs):
    bvals = np.asarray(bvals, dtype=float).reshape(-1)
    directions = np.asarray(bvecs, dtype=float).copy()
    if directions.shape != (len(bvals), 3) or not np.isfinite(directions).all() or not np.isfinite(bvals).all():
        raise ValueError("MK-Curve gradient dimensions/values are invalid")
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions /= np.where(norms > 0, norms, 1)
    return np.column_stack((np.ones(len(bvals)), -bvals[:, None] * tensor_basis(directions, 2),
                            bvals[:, None] ** 2 / 6 * tensor_basis(directions, 4)))


def fit_wlls(signals, design, *, solver="reference"):
    """Return log(S0), six D and fifteen unnormalised fourth-order coefficients.

    As upstream: OLS initialization, predicted-signal weights, one WLLS solve.
    Column equilibration changes the solver conditioning, not its objective.
    No eigenvalue/kurtosis clipping, constraints, or iterative reweighting.
    """
    if solver not in {"reference", "batched"}:
        raise ValueError("MK-Curve solver must be reference or batched")
    signals = np.asarray(signals, dtype=np.float64)
    if signals.ndim != 2 or signals.shape[1] != len(design):
        raise ValueError("MK-Curve signals must be voxels x measurements")
    output = np.full((len(signals), 22), np.nan)
    prepared = _prepared_design(design)
    if prepared is None:
        return output
    scale, scaled = prepared
    valid = np.isfinite(signals).all(axis=1)
    logs = np.log(np.maximum(signals[valid], np.finfo(float).eps))
    return _fit_logged_signals(logs, np.flatnonzero(valid), len(signals), scale, scaled, solver)


def _fit_logged_signals(logs, indices, count, scale, scaled, solver):
    output = np.full((count, 22), np.nan)
    if not len(logs):
        return output
    initial = _ols_initial(scaled, logs)
    with np.errstate(over="ignore", invalid="ignore"):
        weights = np.exp(scaled @ initial).T
    maxima = weights.max(axis=1) if solver == "reference" else None
    pending = (np.isfinite(weights).all(axis=1) & (maxima > 0)
               if maxima is not None else np.ones(len(logs), dtype=bool))
    if solver == "batched":
        candidates = np.flatnonzero(np.isfinite(weights).all(axis=1) & (weights.max(axis=1) > 0))
        # Equilibrated normal systems are used only when well conditioned.
        # Ill-conditioned or rank-deficient rows retain the pivoted QR solve.
        for start in range(0, len(candidates), 256):
            ids = candidates[start:start + 256]
            w = weights[ids] / weights[ids].max(axis=1, keepdims=True)
            weighted = scaled * w[:, :, None]
            gram = weighted.transpose(0, 2, 1) @ weighted
            eigenvalues = np.linalg.eigvalsh(gram)
            safe = (eigenvalues[:, 0] > 0) & (eigenvalues[:, -1] < 1e6 * eigenvalues[:, 0])
            if not safe.any():
                continue
            rhs = weighted[safe].transpose(0, 2, 1) @ (logs[ids[safe]] * w[safe])[:, :, None]
            try:
                beta = np.linalg.solve(gram[safe], rhs)[:, :, 0] / scale
            except np.linalg.LinAlgError:
                continue
            good = np.isfinite(beta).all(axis=1)
            accepted = ids[safe][good]
            output[indices[accepted]] = beta[good]
            pending[accepted] = False
    remaining = np.flatnonzero(pending)
    scaled_qr = np.asfortranarray(scaled) if _DIRECT_GELSY and scaled.dtype == np.float64 else scaled
    for row, index, observed, weight in zip(remaining, indices[pending], logs[pending], weights[pending]):
        if maxima is None and (not np.isfinite(weight).all() or not np.any(weight > 0)):
            continue
        # Dividing all weights by a common factor leaves WLLS unchanged.
        weight = weight / (maxima[row] if maxima is not None else weight.max())
        fit, rank = _weighted_qr(scaled_qr * weight[:, None], observed * weight)
        if rank == 22:
            output[index] = fit / scale
    return output


@lru_cache(maxsize=4)
def _cached_grid_operator(scaled_shape, scaled_bytes, grid_shape, grid_bytes):
    scaled = np.frombuffer(scaled_bytes, dtype=np.float64).reshape(scaled_shape)
    grid = np.frombuffer(grid_bytes, dtype=np.float64).reshape(grid_shape)
    M, P = scaled.shape
    e0 = np.zeros((1, M), dtype=np.float64)
    e0[0, 0] = 1.0
    h0 = (scaled @ _ols_initial(scaled, e0))[:, 0]
    log_grid = np.log(np.maximum(grid, np.finfo(float).eps))
    S = len(log_grid)
    l_min, l_max = float(log_grid[0]), float(log_grid[-1])
    l_mid = 0.5 * (l_min + l_max)
    K = 15
    k_idx = np.arange(K)
    t_nodes = -np.cos(k_idx * np.pi / (K - 1))
    l_nodes = np.ascontiguousarray(0.5 * (t_nodes + 1.0) * (l_max - l_min) + l_min)
    dl_nodes = l_nodes - l_mid
    t_grid = 2.0 * (log_grid - l_min) / (l_max - l_min) - 1.0
    T_nodes_inv = np.ascontiguousarray(np.linalg.inv(np.cos(np.outer(np.arccos(np.clip(t_nodes, -1.0, 1.0)), k_idx))))
    T_grid = np.ascontiguousarray(np.cos(np.outer(np.arccos(np.clip(t_grid, -1.0, 1.0)), k_idx)))
    A_nodes = np.exp(2.0 * np.outer(dl_nodes, h0[1:]))
    U_svd, S_svd, Vt_svd = np.linalg.svd(A_nodes, full_matrices=False)
    R = int(np.count_nonzero(S_svd > S_svd[0] * 1e-15))
    U_R = np.ascontiguousarray(U_svd[:, :R] * S_svd[:R])
    V_R = np.ascontiguousarray(Vt_svd[:R, :])
    X_nonb0 = scaled[1:, :]
    Z_full = (X_nonb0[:, :, None] * X_nonb0[:, None, :]).reshape(M - 1, P * P)
    P_gram = np.ascontiguousarray((V_R[:, :, None] * Z_full[None, :, :]).transpose(1, 0, 2).reshape(M - 1, R * P * P))
    Q_rhs = np.ascontiguousarray((V_R[:, :, None] * X_nonb0[None, :, :]).transpose(1, 0, 2).reshape(M - 1, R * P))
    X0 = np.ascontiguousarray(scaled[0, :])
    X0_is_e0 = bool(np.all(X0[1:] == 0))
    X0_full = np.ascontiguousarray((X0[:, None] * X0[None, :]).reshape(P * P))
    a0_sq_nodes = np.ascontiguousarray(np.exp(2.0 * dl_nodes * h0[0]))
    dl_grid = log_grid - l_mid
    A_grid_full = np.exp(2.0 * np.outer(dl_grid, h0[1:]))
    U_R_full = np.ascontiguousarray(A_grid_full @ V_R.T)
    a0_sq_full = np.ascontiguousarray(np.exp(2.0 * dl_grid * h0[0]))
    log_grid = np.ascontiguousarray(log_grid)
    for arr in (l_nodes, T_nodes_inv, T_grid, U_R, P_gram, Q_rhs, X0, X0_full, a0_sq_nodes, U_R_full, a0_sq_full, log_grid):
        arr.setflags(write=False)
    return (l_mid, K, S, R, l_nodes, T_nodes_inv, T_grid, U_R, P_gram, Q_rhs,
            X0, X0_is_e0, X0_full, a0_sq_nodes, U_R_full, a0_sq_full, log_grid)


class _SyntheticB0Fitter:
    """Chunk-local invariant logs; every b0 still gets its original WLLS solve."""

    def __init__(self, signals, design, *, solver="reference"):
        if solver not in {"reference", "batched"}:
            raise ValueError("MK-Curve solver must be reference or batched")
        signals = np.asarray(signals, dtype=np.float64)
        if signals.ndim != 2 or signals.shape[1] != len(design) or signals.shape[1] < 1:
            raise ValueError("MK-Curve signals must be voxels x measurements")
        self.count, self.solver = len(signals), solver
        self.prepared = _prepared_design(design)
        self.indices = np.flatnonzero(np.isfinite(signals[:, 1:]).all(axis=1))
        selected = signals[self.indices].copy()
        self._positive_nonb0 = (selected[:, 1:] > 0).all(axis=1)
        selected[:, 0] = 1.0
        self.logs = np.log(np.maximum(selected, np.finfo(float).eps))
        self._work_logs = self.logs.copy()

    def fit(self, b0):
        b0 = np.asarray(b0, dtype=np.float64)
        if b0.shape != (self.count,):
            raise ValueError("Synthetic b0 must have one value per voxel")
        if self.prepared is None:
            return np.full((self.count, 22), np.nan)
        baseline = b0[self.indices]
        valid = np.isfinite(baseline)
        if valid.all():
            indices = self.indices
            logs = self._work_logs
            logs[:, 0] = np.log(np.maximum(baseline, np.finfo(float).eps))
        else:
            indices = self.indices[valid]
            # Fancy indexing owns C-order storage, just like signals[valid] in fit_wlls.
            logs = self.logs[valid]
            logs[:, 0] = np.log(np.maximum(baseline[valid], np.finfo(float).eps))
        return _fit_logged_signals(logs, indices, self.count, *self.prepared, self.solver)

    def fit_grid(self, grid):
        grid = np.ascontiguousarray(grid, dtype=np.float64)
        if (self.prepared is None or len(grid) < 64 or not len(self.indices)
                or grid.ndim != 1 or not np.all(np.diff(grid) > 0)):
            curves = np.empty((len(grid), self.count), dtype=np.float64)
            b0_step = np.empty(self.count, dtype=np.float64)
            for step, value in enumerate(grid):
                b0_step.fill(value)
                curves[step] = mean_kurtosis(self.fit(b0_step))
            return curves
        scale, scaled = self.prepared
        scaled_c = np.ascontiguousarray(scaled, dtype=np.float64)
        (l_mid, K, S, R, l_nodes, T_nodes_inv, T_grid, U_R, P_gram, Q_rhs,
         X0, X0_is_e0, X0_full, a0_sq_nodes, U_R_full, a0_sq_full, log_grid) = _cached_grid_operator(
            scaled_c.shape, scaled_c.tobytes(), grid.shape, grid.tobytes())
        P = scaled_c.shape[1]
        N = len(self.indices)
        curves = np.full((S, self.count), np.nan, dtype=np.float64)
        logs_base = self._work_logs
        logs_base[:, 0] = l_mid
        Y_mid_raw = (scaled_c @ _ols_initial(scaled_c, logs_base)).T
        with np.errstate(over="ignore", invalid="ignore"):
            Y_mid = Y_mid_raw - Y_mid_raw.max(axis=1, keepdims=True)
            W_mid = np.exp(Y_mid)
            W2_nonb0 = W_mid[:, 1:] ** 2
            W2_0 = (W_mid[:, 0] ** 2)[:, None] * a0_sq_nodes[None, :]
        G_basis = (W2_nonb0 @ P_gram).reshape(N, R, P * P)
        B_basis = ((W2_nonb0 * logs_base[:, 1:]) @ Q_rhs).reshape(N, R, P)
        G_nodes = (U_R @ np.ascontiguousarray(G_basis.transpose(1, 0, 2).reshape(R, N * P * P))).reshape(K, N, P, P)
        if X0_is_e0:
            G_nodes[:, :, 0, 0] += W2_0.T * (X0[0] ** 2)
        else:
            G_nodes += (W2_0.T[:, :, None] * X0_full[None, None, :]).reshape(K, N, P, P)
        B_nodes = (U_R @ np.ascontiguousarray(B_basis.transpose(1, 0, 2).reshape(R, N * P))).reshape(K, N, P)
        if X0_is_e0:
            B_nodes[:, :, 0] += W2_0.T * l_nodes[:, None] * X0[0]
        else:
            B_nodes += (W2_0.T * l_nodes[:, None])[:, :, None] * X0[None, None, :]
        finite_w = np.isfinite(W_mid).all(axis=1) & self._positive_nonb0
        safe = np.zeros(N, dtype=bool)
        if np.any(finite_w):
            fw_ids = np.flatnonzero(finite_w)
            eig0 = np.linalg.eigvalsh(G_nodes[0, fw_ids])
            eig1 = np.linalg.eigvalsh(G_nodes[-1, fw_ids])
            ok = ((eig0[:, 0] > 0) & (eig0[:, -1] < 1e6 * eig0[:, 0])
                  & (eig1[:, 0] > 0) & (eig1[:, -1] < 1e6 * eig1[:, 0]))
            safe[fw_ids[ok]] = True
        if np.any(safe):
            s_ids = np.flatnonzero(safe)
            n_s = len(s_ids)
            beta_nodes = np.linalg.solve(G_nodes[:, s_ids], B_nodes[:, s_ids, :, None])[:, :, :, 0] / scale
            C_cheb = T_nodes_inv @ beta_nodes.reshape(K, -1)
            cheb_tail = np.max(np.abs(C_cheb[-2:].reshape(2, n_s, P)), axis=(0, 2))
            beta_all = (T_grid @ C_cheb).reshape(S, n_s, P)
            curves_safe = np.empty((S, n_s), dtype=np.float64)
            for v0 in range(0, n_s, 16):
                v1 = min(v0 + 16, n_s)
                curves_safe[:, v0:v1] = mean_kurtosis(beta_all[:, v0:v1, :].reshape(-1, P)).reshape(S, v1 - v0)
            near_thresh = np.any((np.abs(curves_safe) < 1e-3) | (np.abs(curves_safe - 5.0) < 1e-3), axis=0)
            c_masked = np.where((curves_safe < 0) | (curves_safe > 5.0) | ~np.isfinite(curves_safe), -1.0, curves_safe)
            rev_idx = np.argmax(c_masked[::-1] < 0, axis=0) + 1
            rev_idx[rev_idx == 1] = S
            z_idx = S - rev_idx + 1
            c_valid = np.where(np.arange(S)[:, None] < z_idx[None, :], 0.0, c_masked)
            top2 = np.partition(c_valid, S - 2, axis=0)[-2:]
            close_peak = (top2[1] > 0) & ((top2[1] - top2[0]) < 1e-5)
            need_refine = (~np.isfinite(curves_safe).all(axis=0)) | (cheb_tail > 1e-8) | near_thresh | close_peak
            if np.any(need_refine):
                ref_rel = np.flatnonzero(need_refine)
                ref_ids = s_ids[ref_rel]
                n_ref = len(ref_ids)
                G_sub = (U_R_full @ np.ascontiguousarray(G_basis[ref_ids].transpose(1, 0, 2).reshape(R, n_ref * P * P))).reshape(S, n_ref, P, P)
                W2_0_full = (W_mid[ref_ids, 0] ** 2)[:, None] * a0_sq_full[None, :]
                if X0_is_e0:
                    G_sub[:, :, 0, 0] += W2_0_full.T * (X0[0] ** 2)
                else:
                    G_sub += (W2_0_full.T[:, :, None] * X0_full[None, None, :]).reshape(S, n_ref, P, P)
                B_sub = (U_R_full @ np.ascontiguousarray(B_basis[ref_ids].transpose(1, 0, 2).reshape(R, n_ref * P))).reshape(S, n_ref, P)
                if X0_is_e0:
                    B_sub[:, :, 0] += W2_0_full.T * log_grid[:, None] * X0[0]
                else:
                    B_sub += (W2_0_full.T * log_grid[:, None])[:, :, None] * X0[None, None, :]
                betas_sub = np.linalg.solve(G_sub, B_sub[:, :, :, None])[:, :, :, 0] / scale
                curves_safe[:, ref_rel] = mean_kurtosis(betas_sub.reshape(-1, P)).reshape(S, n_ref)
            curves[:, self.indices[s_ids]] = curves_safe
        if not np.all(safe):
            u_ids = np.flatnonzero(~safe)
            u_indices = self.indices[u_ids]
            u_logs = self.logs[u_ids].copy()
            for step, l_val in enumerate(log_grid):
                u_logs[:, 0] = l_val
                betas_u = _fit_logged_signals(u_logs, u_indices, self.count, scale, scaled, self.solver)
                curves[step, u_indices] = mean_kurtosis(betas_u[u_indices])
        return curves


def mean_kurtosis(coefficients):
    diffusion_basis, kurtosis_basis = _integration_bases()
    adc = coefficients[:, 1:7] @ diffusion_basis
    fourth = coefficients[:, 7:22] @ kurtosis_basis
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = np.mean(fourth / adc ** 2, axis=1)
        md_squared = np.mean(coefficients[:, [1, 4, 6]], axis=1) ** 2
    # Upstream W is normalized by MD^2: the algebraic cancellation is not
    # defined at zero MD, even if a direct numerator/ADC computation is finite.
    result[(md_squared == 0) | ~np.isfinite(md_squared)] = np.nan
    return result


def parameter_maps(coefficients):
    n = len(coefficients)
    maps = {key: np.full(n, np.nan) for key in ("fa", "md", "ad", "rd", "mk", "ak", "rk", "e1", "e2", "e3")}
    maps["mk"] = mean_kurtosis(coefficients)
    finite_indices = np.flatnonzero(np.isfinite(coefficients).all(axis=1))
    if not len(finite_indices):
        return maps
    equator = _equator_directions()
    finite_betas = coefficients[finite_indices]
    m = len(finite_indices)
    matrices = np.empty((m, 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = finite_betas[:, 1]
    matrices[:, 0, 1] = matrices[:, 1, 0] = finite_betas[:, 2]
    matrices[:, 0, 2] = matrices[:, 2, 0] = finite_betas[:, 3]
    matrices[:, 1, 1] = finite_betas[:, 4]
    matrices[:, 1, 2] = matrices[:, 2, 1] = finite_betas[:, 5]
    matrices[:, 2, 2] = finite_betas[:, 6]
    values_all, vectors_all = np.linalg.eigh(matrices)
    values = values_all[:, ::-1]
    directions = vectors_all[:, :, -1]
    md = values.mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        fa = np.sqrt(1.5 * np.sum((values - md[:, None]) ** 2, axis=1) / np.sum(values ** 2, axis=1))
    maps["fa"][finite_indices] = fa
    maps["md"][finite_indices] = md
    maps["ad"][finite_indices] = values[:, 0]
    maps["rd"][finite_indices] = values[:, 1:].mean(axis=1)
    maps["e1"][finite_indices] = values[:, 0]
    maps["e2"][finite_indices] = values[:, 1]
    maps["e3"][finite_indices] = values[:, 2]
    nonzero_md = md != 0
    if not np.any(nonzero_md):
        return maps
    nz_indices = finite_indices[nonzero_md]
    nz_betas = finite_betas[nonzero_md]
    nz_dirs = directions[nonzero_md]
    nz_e1 = values[nonzero_md, 0]
    vx = -nz_dirs[:, 1]
    vy = nz_dirs[:, 0]
    s2 = vx * vx + vy * vy
    c = nz_dirs[:, 2]
    k_nz = len(nz_indices)
    rotations = np.zeros((k_nz, 3, 3), dtype=np.float64)
    rotations[:, 0, 0] = 1.0
    rotations[:, 1, 1] = 1.0
    rotations[:, 2, 2] = 1.0
    valid_rot = s2 >= 1e-24
    if np.any(valid_rot):
        vx_v, vy_v, s2_v, c_v = vx[valid_rot], vy[valid_rot], s2[valid_rot], c[valid_rot]
        factor = (1.0 - c_v) / s2_v
        cross = np.zeros((len(vx_v), 3, 3), dtype=np.float64)
        cross[:, 0, 2] = vy_v
        cross[:, 1, 2] = -vx_v
        cross[:, 2, 0] = -vy_v
        cross[:, 2, 1] = vx_v
        rotations[valid_rot] = np.eye(3)[None, :, :] + cross + (cross @ cross) * factor[:, None, None]
    radial = np.matmul(equator[None, :, :], np.swapaxes(rotations, -1, -2))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        b4_dir = tensor_basis(nz_dirs, 4)
        ak = np.sum(b4_dir * nz_betas[:, 7:22], axis=1) / (nz_e1 ** 2)
        b4_rad = tensor_basis(radial, 4)
        b2_rad = tensor_basis(radial, 2)
        num_rk = np.einsum("kqd,kd->kq", b4_rad, nz_betas[:, 7:22])
        den_rk = np.einsum("kqd,kd->kq", b2_rad, nz_betas[:, 1:7]) ** 2
        rk = np.mean(num_rk / den_rk, axis=1)
    maps["ak"][nz_indices] = ak
    maps["rk"][nz_indices] = rk
    return maps


@lru_cache(maxsize=4)
def _cached_cuda_operators(scaled_shape, scaled_bytes, grid_shape, grid_bytes):
    import torch

    dev = torch.device("cuda")
    cpu_ops = _cached_grid_operator(scaled_shape, scaled_bytes, grid_shape, grid_bytes)
    (l_mid, K, S, R, l_nodes, T_nodes_inv, T_grid, U_R, P_gram, Q_rhs,
     X0, X0_is_e0, X0_full, a0_sq_nodes, U_R_full, a0_sq_full, log_grid) = cpu_ops
    scaled_np = np.frombuffer(scaled_bytes, dtype=np.float64).reshape(scaled_shape)
    pinv_scaled_np = np.linalg.pinv(scaled_np)
    diff_basis_np, kurt_basis_np = _integration_bases()
    equator_np = _equator_directions()
    idx2_np, cnt2_np = _tensor_terms_arrays(2)
    idx4_np, cnt4_np = _tensor_terms_arrays(4)

    def _t(arr, dtype=torch.float64, np_dtype=np.float64):
        return torch.as_tensor(np.array(arr, dtype=np_dtype, copy=True, order="C"), dtype=dtype, device=dev)

    return {
        "l_mid": l_mid,
        "K": K,
        "S": S,
        "R": R,
        "X0_is_e0": X0_is_e0,
        "X0_0": float(X0[0]),
        "scaled": _t(scaled_np),
        "pinv_scaled": _t(pinv_scaled_np),
        "l_nodes": _t(l_nodes),
        "T_nodes_inv": _t(T_nodes_inv),
        "T_grid": _t(T_grid),
        "U_R": _t(U_R),
        "P_gram": _t(P_gram),
        "Q_rhs": _t(Q_rhs),
        "X0": _t(X0),
        "X0_full": _t(X0_full),
        "a0_sq_nodes": _t(a0_sq_nodes),
        "U_R_full": _t(U_R_full),
        "a0_sq_full": _t(a0_sq_full),
        "log_grid": _t(log_grid),
        "diff_basis": _t(diff_basis_np),
        "kurt_basis": _t(kurt_basis_np),
        "equator": _t(equator_np),
        "idx2": _t(idx2_np, dtype=torch.long, np_dtype=np.int64),
        "cnt2": _t(cnt2_np),
        "idx4": _t(idx4_np, dtype=torch.long, np_dtype=np.int64),
        "cnt4": _t(cnt4_np),
    }


def _mean_kurtosis_cuda(coefficients_t, diff_basis_t, kurt_basis_t):
    import torch

    adc = torch.matmul(coefficients_t[:, 1:7], diff_basis_t)
    fourth = torch.matmul(coefficients_t[:, 7:22], kurt_basis_t)
    result = torch.mean(fourth / (adc * adc), dim=1)
    md_sq = torch.mean(coefficients_t[:, [1, 4, 6]], dim=1) ** 2
    bad = (md_sq == 0.0) | (~torch.isfinite(md_sq))
    result = torch.where(bad, torch.tensor(float("nan"), dtype=torch.float64, device=coefficients_t.device), result)
    return result


def _parameter_maps_cuda(coefficients_np, ops):
    import torch

    dev = torch.device("cuda")
    n = len(coefficients_np)
    maps = {key: np.full(n, np.nan, dtype=np.float64) for key in ("fa", "md", "ad", "rd", "mk", "ak", "rk", "e1", "e2", "e3")}
    coeff_t = torch.as_tensor(np.ascontiguousarray(coefficients_np), dtype=torch.float64, device=dev)
    maps["mk"] = _mean_kurtosis_cuda(coeff_t, ops["diff_basis"], ops["kurt_basis"]).cpu().numpy()

    finite_mask_t = torch.isfinite(coeff_t).all(dim=1)
    if not torch.any(finite_mask_t):
        return maps

    fin_idx_t = torch.nonzero(finite_mask_t, as_tuple=False).squeeze(1)
    fin_idx_np = fin_idx_t.cpu().numpy()
    fin_betas = coeff_t[fin_idx_t]
    m = len(fin_idx_np)

    matrices = torch.empty((m, 3, 3), dtype=torch.float64, device=dev)
    matrices[:, 0, 0] = fin_betas[:, 1]
    matrices[:, 0, 1] = matrices[:, 1, 0] = fin_betas[:, 2]
    matrices[:, 0, 2] = matrices[:, 2, 0] = fin_betas[:, 3]
    matrices[:, 1, 1] = fin_betas[:, 4]
    matrices[:, 1, 2] = matrices[:, 2, 1] = fin_betas[:, 5]
    matrices[:, 2, 2] = fin_betas[:, 6]
    vals_asc, vecs_asc = torch.linalg.eigh(matrices)
    values = torch.flip(vals_asc, dims=[-1])
    directions = vecs_asc[:, :, -1]
    md = torch.mean(values, dim=1)
    fa = torch.sqrt(1.5 * torch.sum((values - md.unsqueeze(1)) ** 2, dim=1) / torch.sum(values ** 2, dim=1))

    vals_np = values.cpu().numpy()
    maps["fa"][fin_idx_np] = fa.cpu().numpy()
    maps["md"][fin_idx_np] = md.cpu().numpy()
    maps["ad"][fin_idx_np] = vals_np[:, 0]
    maps["rd"][fin_idx_np] = vals_np[:, 1:].mean(axis=1)
    maps["e1"][fin_idx_np] = vals_np[:, 0]
    maps["e2"][fin_idx_np] = vals_np[:, 1]
    maps["e3"][fin_idx_np] = vals_np[:, 2]

    nz_mask = md != 0.0
    if not torch.any(nz_mask):
        return maps

    nz_rel = torch.nonzero(nz_mask, as_tuple=False).squeeze(1)
    nz_idx_np = fin_idx_np[nz_rel.cpu().numpy()]
    nz_betas = fin_betas[nz_rel]
    nz_dirs = directions[nz_rel]
    nz_e1 = values[nz_rel, 0]

    vx = -nz_dirs[:, 1]
    vy = nz_dirs[:, 0]
    s2 = vx * vx + vy * vy
    c = nz_dirs[:, 2]
    k_nz = len(nz_idx_np)
    eye3 = torch.eye(3, dtype=torch.float64, device=dev)
    rotations = eye3.unsqueeze(0).repeat(k_nz, 1, 1)
    valid_rot = s2 >= 1e-24
    if torch.any(valid_rot):
        vr = torch.nonzero(valid_rot, as_tuple=False).squeeze(1)
        vx_v, vy_v, s2_v, c_v = vx[vr], vy[vr], s2[vr], c[vr]
        factor = (1.0 - c_v) / s2_v
        cross = torch.zeros((len(vr), 3, 3), dtype=torch.float64, device=dev)
        cross[:, 0, 2] = vy_v
        cross[:, 1, 2] = -vx_v
        cross[:, 2, 0] = -vy_v
        cross[:, 2, 1] = vx_v
        rotations[vr] = eye3.unsqueeze(0) + cross + torch.matmul(cross, cross) * factor.view(-1, 1, 1)

    radial = torch.matmul(ops["equator"].unsqueeze(0), rotations.transpose(-1, -2))
    idx4, cnt4 = ops["idx4"], ops["cnt4"]
    idx2, cnt2 = ops["idx2"], ops["cnt2"]

    b4_dir = torch.prod(nz_dirs[:, idx4], dim=-1) * cnt4
    ak = torch.sum(b4_dir * nz_betas[:, 7:22], dim=1) / (nz_e1 * nz_e1)

    # Evaluate 256-direction radial kurtosis in bounded sub-batches to keep GPU memory small
    rk_out = torch.empty(k_nz, dtype=torch.float64, device=dev)
    for r0 in range(0, k_nz, 1024):
        r1 = min(r0 + 1024, k_nz)
        rad_sub = radial[r0:r1]
        bet_sub = nz_betas[r0:r1]
        b4_rad = torch.prod(rad_sub[:, :, idx4], dim=-1) * cnt4
        b2_rad = torch.prod(rad_sub[:, :, idx2], dim=-1) * cnt2
        num_rk = torch.einsum("kqd,kd->kq", b4_rad, bet_sub[:, 7:22])
        den_rk = torch.einsum("kqd,kd->kq", b2_rad, bet_sub[:, 1:7]) ** 2
        rk_out[r0:r1] = torch.mean(num_rk / den_rk, dim=1)

    maps["ak"][nz_idx_np] = ak.cpu().numpy()
    maps["rk"][nz_idx_np] = rk_out.cpu().numpy()
    return maps


class _CudaSyntheticB0Fitter:
    """PyTorch CUDA fp64 WLLS and Chebyshev synthetic-b0 grid solver with CPU QR fallback."""

    def __init__(self, signals, design, grid, *, solver="reference"):
        import torch

        self.cpu_fitter = _SyntheticB0Fitter(signals, design, solver=solver)
        self.count = self.cpu_fitter.count
        self.solver = solver
        self.prepared = self.cpu_fitter.prepared
        self.indices = self.cpu_fitter.indices
        self.dev = torch.device("cuda")
        self.grid_np = np.ascontiguousarray(grid, dtype=np.float64)
        self.ops = None
        if self.prepared is not None and len(self.indices) > 0 and len(self.grid_np) >= 64:
            scale, scaled = self.prepared
            scaled_c = np.ascontiguousarray(scaled, dtype=np.float64)
            self.scale_np = scale
            self.scaled_c = scaled_c
            self.scale_t = torch.as_tensor(np.array(scale, dtype=np.float64, copy=True, order="C"), dtype=torch.float64, device=self.dev)
            self.ops = _cached_cuda_operators(scaled_c.shape, scaled_c.tobytes(), self.grid_np.shape, self.grid_np.tobytes())
            self.logs_base_t = torch.as_tensor(np.ascontiguousarray(self.cpu_fitter.logs), dtype=torch.float64, device=self.dev)
            self.pos_nonb0_t = torch.as_tensor(self.cpu_fitter._positive_nonb0, dtype=torch.bool, device=self.dev)

    def fit(self, b0):
        import torch

        if self.ops is None:
            return self.cpu_fitter.fit(b0)
        b0 = np.asarray(b0, dtype=np.float64)
        baseline = b0[self.indices]
        valid = np.isfinite(baseline)
        if not np.any(valid):
            return np.full((self.count, 22), np.nan, dtype=np.float64)
        if valid.all():
            act_indices = self.indices
            logs_t = self.logs_base_t.clone()
            logs_cpu = self.cpu_fitter._work_logs
            logs_cpu[:, 0] = np.log(np.maximum(baseline, np.finfo(float).eps))
            b0_vals = baseline
        else:
            act_indices = self.indices[valid]
            val_t = torch.as_tensor(np.flatnonzero(valid), dtype=torch.long, device=self.dev)
            logs_t = self.logs_base_t[val_t].clone()
            logs_cpu = self.cpu_fitter.logs[valid]
            logs_cpu[:, 0] = np.log(np.maximum(baseline[valid], np.finfo(float).eps))
            b0_vals = baseline[valid]

        eps = float(np.finfo(float).eps)
        logs_t[:, 0] = torch.log(torch.clamp(torch.as_tensor(np.ascontiguousarray(b0_vals), dtype=torch.float64, device=self.dev), min=eps))
        scaled_t = self.ops["scaled"]
        pinv_t = self.ops["pinv_scaled"]

        ols = torch.matmul(logs_t, pinv_t.T)
        weights = torch.exp(torch.matmul(ols, scaled_t.T))
        w_max = weights.amax(dim=1, keepdim=True)
        w = weights / torch.clamp(w_max, min=1e-300)
        weighted = scaled_t.unsqueeze(0) * w.unsqueeze(2)
        gram = torch.matmul(weighted.mT, weighted)
        L, info = torch.linalg.cholesky_ex(gram)
        diag_L = torch.diagonal(L, dim1=-2, dim2=-1)
        safe = (
            (info == 0)
            & torch.isfinite(weights).all(dim=1)
            & (w_max.squeeze(1) > 0)
            & (diag_L.amin(dim=-1) > 1e-4 * diag_L.amax(dim=-1))
        )

        output = np.full((self.count, 22), np.nan, dtype=np.float64)
        safe_np = safe.cpu().numpy()
        if np.any(safe_np):
            rhs = torch.matmul(weighted[safe].mT, (logs_t[safe] * w[safe]).unsqueeze(2))
            beta = torch.cholesky_solve(rhs, L[safe]).squeeze(-1) / self.scale_t
            beta_np = beta.cpu().numpy()
            good = np.isfinite(beta_np).all(axis=1)
            acc = np.flatnonzero(safe_np)[good]
            output[act_indices[acc]] = beta_np[good]
            safe_np[np.flatnonzero(safe_np)[~good]] = False

        if not np.all(safe_np):
            rem = np.flatnonzero(~safe_np)
            fb = _fit_logged_signals(logs_cpu[rem], np.arange(len(rem)), len(rem), self.scale_np, self.scaled_c, self.solver)
            output[act_indices[rem]] = fb
        return output

    def fit_grid(self):
        import torch

        if self.ops is None:
            return self.cpu_fitter.fit_grid(self.grid_np)
        ops = self.ops
        K, S, R = ops["K"], ops["S"], ops["R"]
        P = 22
        N = len(self.indices)
        curves = np.full((S, self.count), np.nan, dtype=np.float64)

        logs_mid = self.logs_base_t.clone()
        logs_mid[:, 0] = ops["l_mid"]
        Y_mid_raw = torch.matmul(torch.matmul(logs_mid, ops["pinv_scaled"].T), ops["scaled"].T)
        Y_mid = Y_mid_raw - Y_mid_raw.amax(dim=1, keepdim=True)
        W_mid = torch.exp(Y_mid)
        W2_nonb0 = W_mid[:, 1:] ** 2
        W2_0 = (W_mid[:, 0] ** 2).unsqueeze(1) * ops["a0_sq_nodes"].unsqueeze(0)

        G_basis = torch.matmul(W2_nonb0, ops["P_gram"]).reshape(N, R, P * P)
        B_basis = torch.matmul(W2_nonb0 * logs_mid[:, 1:], ops["Q_rhs"]).reshape(N, R, P)
        G_nodes = torch.matmul(ops["U_R"], G_basis.permute(1, 0, 2).reshape(R, N * P * P)).reshape(K, N, P, P)
        if ops["X0_is_e0"]:
            G_nodes[:, :, 0, 0] += W2_0.T * (ops["X0_0"] ** 2)
        else:
            G_nodes += (W2_0.T.unsqueeze(-1) * ops["X0_full"].view(1, 1, -1)).reshape(K, N, P, P)

        B_nodes = torch.matmul(ops["U_R"], B_basis.permute(1, 0, 2).reshape(R, N * P)).reshape(K, N, P)
        if ops["X0_is_e0"]:
            B_nodes[:, :, 0] += W2_0.T * ops["l_nodes"].unsqueeze(1) * ops["X0_0"]
        else:
            B_nodes += (W2_0.T * ops["l_nodes"].unsqueeze(1)).unsqueeze(-1) * ops["X0"].view(1, 1, -1)

        finite_w = torch.isfinite(W_mid).all(dim=1) & self.pos_nonb0_t
        L_nodes, info_nodes = torch.linalg.cholesky_ex(G_nodes)
        diag_L = torch.diagonal(L_nodes, dim1=-2, dim2=-1)
        cond_ok = (info_nodes == 0) & (diag_L.amin(dim=-1) > 1e-3 * diag_L.amax(dim=-1))
        safe_t = finite_w & cond_ok.all(dim=0)
        safe_np = safe_t.cpu().numpy()

        if np.any(safe_np):
            s_idx_t = torch.nonzero(safe_t, as_tuple=False).squeeze(1)
            s_ids_np = s_idx_t.cpu().numpy()
            n_s = len(s_ids_np)
            beta_nodes = torch.cholesky_solve(B_nodes[:, s_idx_t].unsqueeze(-1), L_nodes[:, s_idx_t]).squeeze(-1) / self.scale_t
            C_cheb = torch.matmul(ops["T_nodes_inv"], beta_nodes.reshape(K, -1))
            cheb_tail = torch.amax(torch.abs(C_cheb[-2:].reshape(2, n_s, P)), dim=(0, 2))
            beta_all = torch.matmul(ops["T_grid"], C_cheb).reshape(S, n_s, P)

            curves_safe_t = torch.empty((S, n_s), dtype=torch.float64, device=self.dev)
            for v0 in range(0, n_s, 256):
                v1 = min(v0 + 256, n_s)
                curves_safe_t[:, v0:v1] = _mean_kurtosis_cuda(
                    beta_all[:, v0:v1, :].reshape(-1, P), ops["diff_basis"], ops["kurt_basis"]
                ).reshape(S, v1 - v0)

            curves_safe = curves_safe_t.cpu().numpy()
            cheb_tail_np = cheb_tail.cpu().numpy()
            near_thresh = np.any((np.abs(curves_safe) < 1e-3) | (np.abs(curves_safe - 5.0) < 1e-3), axis=0)
            c_masked = np.where((curves_safe < 0) | (curves_safe > 5.0) | ~np.isfinite(curves_safe), -1.0, curves_safe)
            rev_idx = np.argmax(c_masked[::-1] < 0, axis=0) + 1
            rev_idx[rev_idx == 1] = S
            z_idx = S - rev_idx + 1
            c_valid = np.where(np.arange(S)[:, None] < z_idx[None, :], 0.0, c_masked)
            top2 = np.partition(c_valid, S - 2, axis=0)[-2:]
            close_peak = (top2[1] > 0) & ((top2[1] - top2[0]) < 1e-5)
            need_refine = (~np.isfinite(curves_safe).all(axis=0)) | (cheb_tail_np > 1e-8) | near_thresh | close_peak

            if np.any(need_refine):
                ref_rel_np = np.flatnonzero(need_refine)
                # Refine in bounded GPU sub-batches so S * sub * 22 * 22 stays small
                for r0 in range(0, len(ref_rel_np), 256):
                    r1 = min(r0 + 256, len(ref_rel_np))
                    sub_rel = ref_rel_np[r0:r1]
                    sub_ids_np = s_ids_np[sub_rel]
                    sub_ids_t = torch.as_tensor(sub_ids_np, dtype=torch.long, device=self.dev)
                    n_ref = len(sub_ids_np)
                    G_sub = torch.matmul(
                        ops["U_R_full"], G_basis[sub_ids_t].permute(1, 0, 2).reshape(R, n_ref * P * P)
                    ).reshape(S, n_ref, P, P)
                    W2_0_full = (W_mid[sub_ids_t, 0] ** 2).unsqueeze(1) * ops["a0_sq_full"].unsqueeze(0)
                    if ops["X0_is_e0"]:
                        G_sub[:, :, 0, 0] += W2_0_full.T * (ops["X0_0"] ** 2)
                    else:
                        G_sub += (W2_0_full.T.unsqueeze(-1) * ops["X0_full"].view(1, 1, -1)).reshape(S, n_ref, P, P)
                    B_sub = torch.matmul(
                        ops["U_R_full"], B_basis[sub_ids_t].permute(1, 0, 2).reshape(R, n_ref * P)
                    ).reshape(S, n_ref, P)
                    if ops["X0_is_e0"]:
                        B_sub[:, :, 0] += W2_0_full.T * ops["log_grid"].unsqueeze(1) * ops["X0_0"]
                    else:
                        B_sub += (W2_0_full.T * ops["log_grid"].unsqueeze(1)).unsqueeze(-1) * ops["X0"].view(1, 1, -1)
                    L_sub, info_sub = torch.linalg.cholesky_ex(G_sub)
                    if bool((info_sub == 0).all()):
                        betas_sub = torch.cholesky_solve(B_sub.unsqueeze(-1), L_sub).squeeze(-1) / self.scale_t
                        curves_safe[:, sub_rel] = _mean_kurtosis_cuda(
                            betas_sub.reshape(-1, P), ops["diff_basis"], ops["kurt_basis"]
                        ).reshape(S, n_ref).cpu().numpy()
                    else:
                        u_indices = self.indices[sub_ids_np]
                        u_logs = self.cpu_fitter.logs[sub_ids_np].copy()
                        log_grid_np = np.log(np.maximum(self.grid_np, np.finfo(float).eps))
                        for step, l_val in enumerate(log_grid_np):
                            u_logs[:, 0] = l_val
                            betas_u = _fit_logged_signals(u_logs, np.arange(n_ref), n_ref, self.scale_np, self.scaled_c, self.solver)
                            curves_safe[step, sub_rel] = mean_kurtosis(betas_u)

            curves[:, self.indices[s_ids_np]] = curves_safe

        if not np.all(safe_np):
            u_ids = np.flatnonzero(~safe_np)
            u_indices = self.indices[u_ids]
            u_logs = self.cpu_fitter.logs[u_ids].copy()
            log_grid_np = np.log(np.maximum(self.grid_np, np.finfo(float).eps))
            for step, l_val in enumerate(log_grid_np):
                u_logs[:, 0] = l_val
                betas_u = _fit_logged_signals(u_logs, u_indices, self.count, self.scale_np, self.scaled_c, self.solver)
                curves[step, u_indices] = mean_kurtosis(betas_u[u_indices])

        return curves

