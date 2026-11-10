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
    matrices = np.empty((len(finite_indices), 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = finite_betas[:, 1]
    matrices[:, 0, 1] = matrices[:, 1, 0] = finite_betas[:, 2]
    matrices[:, 0, 2] = matrices[:, 2, 0] = finite_betas[:, 3]
    matrices[:, 1, 1] = finite_betas[:, 4]
    matrices[:, 1, 2] = matrices[:, 2, 1] = finite_betas[:, 5]
    matrices[:, 2, 2] = finite_betas[:, 6]
    values_all, vectors_all = np.linalg.eigh(matrices)
    for pos, index in enumerate(finite_indices):
        beta = finite_betas[pos]
        values, vectors = values_all[pos], vectors_all[pos]
        values, direction = values[::-1], vectors[:, -1]
        md = values.mean()
        with np.errstate(divide="ignore", invalid="ignore"):
            fa = np.sqrt(1.5 * np.sum((values - md) ** 2) / np.sum(values ** 2))
        maps["fa"][index], maps["md"][index] = fa, md
        maps["ad"][index], maps["rd"][index] = values[0], values[1:].mean()
        for key, value in zip(("e1", "e2", "e3"), values):
            maps[key][index] = value
        if md == 0:
            continue
        v = np.array([-direction[1], direction[0], 0.])
        s2, c = np.dot(v, v), direction[2]
        if s2 < 1e-24:
            radial = equator  # Well-defined equatorial plane at either pole.
        else:
            vx, vy, vz = v
            cross = np.array([[0, -vz, vy], [vz, 0, -vx], [-vy, vx, 0]])
            rotation = np.eye(3) + cross + cross @ cross * ((1 - c) / s2)
            radial = equator @ rotation.T
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            ak = (tensor_basis(direction[None, :], 4) @ beta[7:22])[0] / values[0] ** 2
            rk = np.mean((tensor_basis(radial, 4) @ beta[7:22]) / (tensor_basis(radial, 2) @ beta[1:7]) ** 2)
        maps["ak"][index], maps["rk"][index] = ak, rk
    return maps
