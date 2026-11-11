"""Guarded, unsmoothed DIPY DKI with separate quantitative and display outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Callable, Optional

import sys
from shared.pipeline import RawInputBundle
from shared.cohort_schedule import scheduled_stage
from shared.numeric_threads import dki_threads

if sys.platform == "darwin":
    try:
        import torch as _darwin_preload_torch  # noqa: F401
    except Exception:
        pass

DKI_METRICS = ("mk", "ak", "rk", "kfa", "mkt", "rtk")
DKI_FIT_METHODS = ("WLS", "OLS", "MKCURVE", "MKCURVE_MATLAB")
DKI_REFERENCE = "https://docs.dipy.org/stable/examples_built/reconstruction/reconst_dki.html"


@dataclass(frozen=True)
class DkiConfig:
    subject_id: str
    raw_input_bundle: RawInputBundle
    output_dir: Path
    fit_method: str = "MKCURVE"
    b0_threshold: float = 50.0
    shell_tolerance: float = 100.0
    max_bvalue: float = 2500.0
    min_directions_per_shell: int = 15
    voxels_per_chunk: int = 1024
    force_recompute: bool = False
    matlab_executable: Optional[Path] = None
    mkcurve_root: Optional[Path] = None
    display_min_kurtosis: float = 0.0
    display_max_kurtosis: float = 3.0
    dense_metrics: bool = True
    num_threads: int | None = None
    mkcurve_workers: int = 1
    mkcurve_voxels_per_chunk: int = 64
    mkcurve_chunk_schedule: str = 'dynamic'
    device: str = 'cpu'


@dataclass(frozen=True)
class DkiResult:
    subject_id: str
    status: str
    output_dir: Path
    mk_path: Path
    ak_path: Path
    rk_path: Path
    kfa_path: Path
    mkt_path: Path
    rtk_path: Path
    rmse_path: Path
    nrmse_path: Path
    valid_mask_path: Path
    manifest_path: Path
    qc_path: Path
    progress_path: Path
    outputs: dict[str, Path] = field(default_factory=dict)
    mkcurve_outputs: dict[str, Path] = field(default_factory=dict)


def dki_output_paths(subject_id: str, output_dir: Path) -> dict[str, Path]:
    """Paths relative to the DKI stage directory, not the microstructure root."""
    root = Path(output_dir).expanduser().resolve()
    paths = {key: root / f"{subject_id}_dki_{key.upper()}.nii.gz" for key in DKI_METRICS}
    paths.update({f"{key}_raw": root / f"{subject_id}_dki_{key.upper()}_raw.nii.gz" for key in DKI_METRICS})
    paths.update({
        "rmse": root / f"{subject_id}_dki_RMSE.nii.gz",
        "nrmse": root / f"{subject_id}_dki_NRMSE.nii.gz",
        "valid_mask": root / f"{subject_id}_dki_valid_mask.nii.gz",
        "fit_mask": root / f"{subject_id}_dki_fit_mask.nii.gz",
        "params": root / f"{subject_id}_dki_parameters_raw.nii.gz",
        "s0": root / f"{subject_id}_dki_S0.nii.gz",
        "manifest": root / "dki_manifest.json",
        "qc": root / "dki_qc.json",
        "progress": root / "dki_progress.json",
    })
    return paths


def _method(config: DkiConfig) -> str:
    method = config.fit_method.strip().upper()
    if method not in DKI_FIT_METHODS:
        if method == "MK_CURVE":
            raise ValueError("mk_curve is reserved; select MKCURVE for the Python port")
        raise ValueError("DKI fit_method must be WLS, OLS, MKCURVE, or MKCURVE_MATLAB")
    return method


def _validate_options(config: DkiConfig) -> None:
    import numpy as np

    if not config.subject_id.strip() or config.subject_id != config.raw_input_bundle.subject_id:
        raise ValueError("DKI subject_id must match the input bundle")
    _method(config)
    for name in ('voxels_per_chunk', 'mkcurve_workers', 'mkcurve_voxels_per_chunk'):
        if type(getattr(config, name)) is not int or getattr(config, name) < 1:
            raise ValueError('DKI ' + name + ' must be a positive integer')
    if config.num_threads is not None and (type(config.num_threads) is not int or config.num_threads < 1):
        raise ValueError('DKI num_threads must be a positive integer')
    if config.mkcurve_chunk_schedule not in ('dynamic', 'ordered'):
        raise ValueError('MK-Curve chunk schedule must be dynamic or ordered')
    if not isinstance(config.device, str) or config.device.strip().lower() not in ('cpu', 'cuda', 'auto'):
        raise ValueError("DKI device must be 'cpu', 'cuda', or 'auto'")
    for name in ("b0_threshold", "shell_tolerance", "max_bvalue", "display_min_kurtosis", "display_max_kurtosis"):
        if not np.isfinite(getattr(config, name)):
            raise ValueError(f"DKI {name} must be finite")
    if config.b0_threshold < 0 or config.shell_tolerance <= 0 or config.max_bvalue <= config.b0_threshold:
        raise ValueError("DKI requires nonnegative b0_threshold, positive shell_tolerance, and max_bvalue above b0")
    if config.min_directions_per_shell < 15 or config.voxels_per_chunk < 1:
        raise ValueError("DKI requires at least 15 directions per shell and a positive voxels_per_chunk")
    if config.display_min_kurtosis < 0 or config.display_max_kurtosis <= config.display_min_kurtosis:
        raise ValueError("DKI display bounds must satisfy 0 <= min < max")


def _scaled_rank(matrix) -> tuple[int, float]:
    import numpy as np

    scale = np.linalg.norm(matrix, axis=0)
    scaled = matrix / np.where(scale > 0, scale, 1)
    singular = np.linalg.svd(scaled, compute_uv=False)
    tolerance = max(scaled.shape) * np.finfo(float).eps * (singular[0] if singular.size else 1)
    rank = int(np.count_nonzero(singular > tolerance))
    condition = float(singular[0] / singular[-1]) if singular.size and singular[-1] > tolerance else None
    return rank, condition


def _shells(bvals, threshold: float, tolerance: float) -> list[list[int]]:
    # Anchor each shell at its smallest b-value: drifting/chained values must
    # not merge two acquisitions whose endpoints are outside the tolerance.
    shells: list[list[int]] = []
    for index in sorted((i for i, b in enumerate(bvals) if b > threshold), key=lambda i: bvals[i]):
        if not shells or bvals[index] - bvals[shells[-1][0]] > tolerance:
            shells.append([])
        shells[-1].append(index)
    return shells


def _independent_axes(vectors) -> int:
    import numpy as np

    unique = []
    for vector in vectors:
        normalized = vector / np.linalg.norm(vector)
        if not any(abs(float(np.dot(normalized, previous))) > 1 - 1e-6 for previous in unique):
            unique.append(normalized)
    return len(unique)


def _load_protocol(config: DkiConfig, *, require_mask: bool = True):
    import nibabel as nib
    import numpy as np
    from dipy.core.gradients import gradient_table
    from dipy.reconst.utils import dki_design_matrix

    bundle = config.raw_input_bundle
    image = nib.load(str(bundle.dwi_path))
    if len(image.shape) != 4 or image.shape[3] < 22:
        raise ValueError("DKI requires a 4D DWI with at least 22 measurements")
    if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) < 1e-12:
        raise ValueError("DKI DWI must have a finite nonsingular world affine")
    bvals = np.loadtxt(bundle.bval_path, ndmin=1).reshape(-1)
    vectors = np.loadtxt(bundle.bvec_path, ndmin=2)
    if vectors.shape == (3, image.shape[3]):
        vectors = vectors.T
    if len(bvals) != image.shape[3] or vectors.shape != (image.shape[3], 3):
        raise ValueError("DKI bvals/bvecs must match the DWI volume count")
    if not np.isfinite(bvals).all() or np.any(bvals < 0) or not np.isfinite(vectors).all():
        raise ValueError("DKI gradients must be finite and b-values nonnegative")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(np.abs(norms[bvals > config.b0_threshold] - 1) > 0.01):
        raise ValueError("DKI diffusion b-vectors must be unit length within 0.01")
    nonzero_baseline_vectors = int(np.count_nonzero(norms[bvals <= config.b0_threshold] > 0.01))
    # HCP can retain direction entries for nominal b0 volumes. Direction is
    # irrelevant to the b0 model; clear it in the working table only.
    vectors = vectors.copy()
    vectors[bvals <= config.b0_threshold] = 0
    # Select complete shells by their median, not individual scanner b-values.
    # Otherwise a nominal b=3000 shell can be silently truncated at 3000.
    all_shells = _shells(bvals, config.b0_threshold, config.shell_tolerance)
    selected = np.asarray(sorted(
        list(np.flatnonzero(bvals <= config.b0_threshold)) +
        [index for shell in all_shells if np.median(bvals[shell]) <= config.max_bvalue for index in shell]
    ), dtype=int)
    selected_bvals, selected_vectors = bvals[selected], vectors[selected]
    if not np.any(selected_bvals <= config.b0_threshold):
        raise ValueError("DKI requires at least one b0 measurement")
    shells = _shells(selected_bvals, config.b0_threshold, config.shell_tolerance)
    if len(shells) < 2:
        raise ValueError("DKI requires at least two nonzero shells after the maximum-b selection")
    gtab = gradient_table(selected_bvals, bvecs=selected_vectors, b0_threshold=config.b0_threshold, atol=0.01)
    design = dki_design_matrix(gtab)
    shell_info = []
    for indices in shells:
        unique = _independent_axes(selected_vectors[indices])
        angular_rank, _ = _scaled_rank(design[indices, 6:21])
        shell_info.append({
            "bvalue": float(np.mean(selected_bvals[indices])), "volume_count": len(indices),
            "independent_directions": unique, "angular_rank": angular_rank,
        })
        if unique < config.min_directions_per_shell or angular_rank < 15:
            raise ValueError(
                "DKI requires at least 15 independent directions and angular design rank 15 per shell; "
                f"shell {shell_info[-1]['bvalue']:g} has {unique} directions and rank {angular_rank}"
            )
    rank, condition = _scaled_rank(design)
    if rank < 22:
        raise ValueError(f"DKI scaled-column design rank is {rank}; all 22 parameters must be identifiable")
    mask_data = None
    if require_mask or bundle.mask_path.is_file():
        mask_image = nib.load(str(bundle.mask_path))
        mask_data = np.asanyarray(mask_image.dataobj)
        if mask_image.shape != image.shape[:3] or not np.allclose(mask_image.affine, image.affine, rtol=0, atol=1e-4):
            raise ValueError("DKI mask must match the DWI's native shape and affine")
        if not np.isfinite(mask_data).all() or not np.all((mask_data == 0) | (mask_data == 1)) or not np.any(mask_data):
            raise ValueError("DKI mask must be finite, binary and nonempty")
    inspection = {
        "shape": list(image.shape), "selected_volumes": selected.tolist(),
        "excluded_high_b_volumes": int(image.shape[3] - len(selected)),
        "upper_bvalue_policy": "include complete shells whose median b-value is at most max_bvalue",
        "b0_count": int(np.count_nonzero(selected_bvals <= config.b0_threshold)),
        "baseline_bvectors_cleared_in_working_table": nonzero_baseline_vectors,
        "shells": shell_info, "design_rank": rank, "design_condition_scaled": condition,
        "mask_voxels": int(np.count_nonzero(mask_data)) if mask_data is not None else None,
        "geometry_pass": True if mask_data is not None else None,
        "warnings": (["b > 2500 s/mm^2 included: truncated cumulant DKI can be biased; review high-b fits"]
                     if np.any(selected_bvals > 2500) else []),
    }
    return image, mask_data.astype(bool) if mask_data is not None else None, gtab, selected, inspection


def inspect_dki_inputs(config: DkiConfig, *, require_mask: bool = True) -> dict[str, Any]:
    """Check eligibility and tool discovery without constructing or fitting a model."""
    result = {"eligible": False, "available": False, "fit_method": config.fit_method, "reasons": []}
    try:
        _validate_options(config)
        if _method(config).startswith("MKCURVE"):
            if _method(config) == "MKCURVE_MATLAB":
                from microstructure.mkcurve_matlab import inspect_mkcurve_backend
            else:
                from microstructure.mkcurve import inspect_mkcurve_backend

            external = inspect_mkcurve_backend(config.matlab_executable, config.mkcurve_root)
            result["mkcurve_backend"] = external
            if not external["available"]:
                raise ValueError("MKCURVE backend unavailable: " + ", ".join(external["missing_required"]))
        _, _, _, _, inspection = _load_protocol(config, require_mask=require_mask)
        result.update(inspection, eligible=True, available=True, fit_method=_method(config))
    except Exception as exc:
        result["reasons"] = [f"{type(exc).__name__}: {exc}"]
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".json", encoding="utf-8", delete=False) as stream:
        staged = Path(stream.name)
        try:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        except Exception:
            staged.unlink(missing_ok=True)
            raise
    try:
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)


def _signature(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest.hexdigest()}


def expected_dki_signature(config: DkiConfig) -> dict[str, Any]:
    _validate_options(config)
    bundle = config.raw_input_bundle
    libraries = {name: importlib.metadata.version(name) for name in ("dipy", "numpy", "nibabel")}
    signature = {
        "version": 3, "subject_id": config.subject_id,
        "inputs": {key: _signature(getattr(bundle, f"{key}_path")) for key in ("dwi", "bval", "bvec", "mask")},
        "libraries": libraries,
        "settings": {
            "fit_method": _method(config), "standard_fit_method": "WLS" if _method(config).startswith("MKCURVE") else _method(config),
            "b0_threshold": config.b0_threshold, "shell_tolerance": config.shell_tolerance,
            "max_bvalue": config.max_bvalue, "min_directions_per_shell": config.min_directions_per_shell,
            "voxels_per_chunk": config.voxels_per_chunk,
            "num_threads": config.num_threads, "mkcurve_workers": config.mkcurve_workers,
            "mkcurve_voxels_per_chunk": config.mkcurve_voxels_per_chunk,
            "mkcurve_chunk_schedule": config.mkcurve_chunk_schedule,
            "display_bounds": [config.display_min_kurtosis, config.display_max_kurtosis],
            "raw_metric_bounds": None, "smoothing": False, "nrmse_normalization": "fitted_S0",
        },
    }
    if _method(config).startswith("MKCURVE"):
        if _method(config) == "MKCURVE_MATLAB":
            from microstructure.mkcurve_matlab import inspect_mkcurve_backend
        else:
            from microstructure.mkcurve import inspect_mkcurve_backend

        signature["mkcurve_backend"] = inspect_mkcurve_backend(config.matlab_executable, config.mkcurve_root)
    return signature


def dki_outputs_current(config: DkiConfig) -> bool:
    if config.force_recompute:
        return False
    try:
        paths = dki_output_paths(config.subject_id, config.output_dir)
        manifest, qc = _read_json(paths["manifest"]), _read_json(paths["qc"])
        if manifest.get("status") != "completed" or qc.get("status") != "completed":
            return False
        if int(qc.get("valid_voxels", 0)) <= 0:
            return False
        if manifest.get("signature") != expected_dki_signature(config):
            return False
        import nibabel as nib
        import numpy as np

        source = nib.load(str(config.raw_input_bundle.dwi_path))
        records = manifest.get("artifacts") or {}
        for key, path in paths.items():
            if key in {"manifest", "qc", "progress"}:
                continue
            if not path.is_file() or records.get(key) != _signature(path):
                return False
            image = nib.load(str(path))
            shape = source.shape[:3] + ((27,) if key == "params" else ())
            if image.shape != shape or not np.allclose(image.affine, source.affine, rtol=0, atol=1e-4):
                return False
        if _method(config).startswith("MKCURVE"):
            external = manifest.get("mkcurve") or {}
            maps = external.get("maps") or {}
            if not maps or not Path(external.get("record_path", "")).is_file():
                return False
            if external.get("kind") != "mkcurve_python_port" and not external.get("automatic_qc_passed"):
                return False
            if external.get("kind") == "mkcurve_python_port" and external.get("status") != "completed":
                return False
            if external.get("kind") == "mkcurve_python_port" and int(external.get("corrected_valid_voxels", 0)) <= 0:
                return False
            for key, value in (external.get("auxiliary_files") or {}).items():
                if not Path(value).is_file() or (external.get("auxiliary_artifacts") or {}).get(key) != _signature(Path(value)):
                    return False
            if external.get("kind") == "mkcurve_python_port" and manifest.get("mkcurve_record_artifact") != _signature(Path(external["record_path"])):
                return False
            for key, value in maps.items():
                if not Path(value).is_file() or (manifest.get("mkcurve_artifacts") or {}).get(key) != _signature(Path(value)):
                    return False
        return True
    except (OSError, ValueError, ImportError, KeyError):
        return False


def _result(config: DkiConfig, status: str, paths: dict[str, Path], mkcurve: Optional[dict] = None) -> DkiResult:
    return DkiResult(
        subject_id=config.subject_id, status=status, output_dir=paths["manifest"].parent,
        **{f"{key}_path": paths[key] for key in (*DKI_METRICS, "rmse", "nrmse", "valid_mask", "manifest", "qc", "progress")},
        outputs=paths,
        mkcurve_outputs={key: Path(value) for key, value in (mkcurve or {}).get("maps", {}).items()},
    )


def _progress(path: Path, config: DkiConfig, status: str, **details) -> None:
    _write_json(path, {
        "kind": "microstructure_stage_progress", "stage": "dki", "subject_id": config.subject_id,
        "status": status, "updated_at": datetime.now(timezone.utc).isoformat(), **details,
    })


_ORIG_DKI_FIT = None


def _fit_dki_chunk(model, observed, fit_method_name: str):
    """Diagonal-free WLS/OLS fit and batched tensor decomposition matching DIPY."""
    import numpy as np
    from dipy.reconst.dki import (
        decompose_tensor,
        from_lower_triangular,
        params_to_dki_params,
    )

    A = model.design_matrix
    inv_A = model.inverse_design_matrix
    log_s = np.log(np.maximum(observed, model.min_signal))
    n = len(observed)
    res_all = np.empty((n, 22), dtype=np.float64)
    if fit_method_name in ("WLS", "UWLLS"):
        if n <= 64:
            for v in range(n):
                ls = log_s[v]
                ols_r = np.dot(inv_A, ls)
                w = np.exp(np.dot(A, ols_r))
                W_A = np.multiply(w[:, None], A, order="C")
                W_A[W_A == 0] = 0.0
                inv_W_A = np.linalg.pinv(W_A)
                inv_W_A_W = np.multiply(inv_W_A, w[None, :], order="C")
                inv_W_A_W[inv_W_A_W == 0] = 0.0
                res_all[v] = np.dot(inv_W_A_W, ls)
        else:
            col_scale = np.maximum(np.linalg.norm(A, axis=0), 1e-30)
            A_eq = np.ascontiguousarray(A / col_scale)
            ols_r = np.dot(log_s, inv_A.T)
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                w_all = np.exp(np.dot(ols_r, A.T))
            for s in range(0, n, 2048):
                e = min(n, s + 2048)
                w_b = w_all[s:e]
                ls_b = log_s[s:e]
                w_max = np.maximum(np.max(w_b, axis=1, keepdims=True), 1e-300)
                w_norm = np.where(np.isfinite(w_b), w_b / w_max, 0.0)
                W_A_eq = w_norm[:, :, None] * A_eq[None, :, :]
                G_eq = np.matmul(np.swapaxes(W_A_eq, -1, -2), W_A_eq)
                rhs_eq = np.matmul(np.swapaxes(W_A_eq, -1, -2), (w_norm * ls_b)[:, :, None])[..., 0]
                finite_w = np.isfinite(w_b).all(axis=1) & (np.max(w_b, axis=1) > 0)
                diag_g = np.diagonal(G_eq, axis1=-2, axis2=-1)
                cand = finite_w & (np.min(diag_g, axis=-1) > 1e-12 * np.max(diag_g, axis=-1))
                solved = np.zeros(e - s, dtype=bool)
                if np.any(cand):
                    cand_idx = np.flatnonzero(cand)
                    try:
                        L_eq = np.linalg.cholesky(G_eq[cand_idx])
                        dL = np.diagonal(L_eq, axis1=-2, axis2=-1)
                        good_L = np.min(dL, axis=-1) > 1e-6 * np.max(dL, axis=-1)
                        if np.any(good_L):
                            ok_idx = cand_idx[good_L]
                            sol_eq = np.linalg.solve(G_eq[ok_idx], rhs_eq[ok_idx][..., None])[..., 0]
                            res_all[s + ok_idx] = sol_eq / col_scale
                            solved[ok_idx] = True
                    except np.linalg.LinAlgError:
                        for ci in cand_idx:
                            try:
                                L_i = np.linalg.cholesky(G_eq[ci])
                                dLi = np.diag(L_i)
                                if np.min(dLi) > 1e-6 * np.max(dLi):
                                    sol_i = np.linalg.solve(G_eq[ci], rhs_eq[ci])
                                    res_all[s + ci] = sol_i / col_scale
                                    solved[ci] = True
                            except np.linalg.LinAlgError:
                                pass
                for bi in np.flatnonzero(~solved):
                    ls = ls_b[bi]
                    wv = w_b[bi]
                    W_A = np.multiply(wv[:, None], A, order="C")
                    W_A[W_A == 0] = 0.0
                    inv_W_A = np.linalg.pinv(W_A)
                    inv_W_A_W = np.multiply(inv_W_A, wv[None, :], order="C")
                    inv_W_A_W[inv_W_A_W == 0] = 0.0
                    res_all[s + bi] = np.dot(inv_W_A_W, ls)
    else:
        if n <= 64:
            for v in range(n):
                res_all[v] = np.dot(inv_A, log_s[v])
        else:
            res_all[:] = np.dot(log_s, inv_A.T)

    if n <= 64:
        out = np.array([params_to_dki_params(res_all[v], min_diffusivity=model.min_diffusivity) for v in range(n)])
        return out[:, :27], out[:, 27]

    DT = from_lower_triangular(res_all[:, :6])
    evals, evecs = decompose_tensor(DT, min_diffusivity=model.min_diffusivity)
    md_sq = evals.mean(axis=-1) ** 2
    kt = np.where(md_sq[:, None] != 0, res_all[:, 6:21] / md_sq[:, None], 0.0 * res_all[:, 6:21])
    s0 = np.exp(-res_all[:, 21])
    params = np.concatenate((evals, evecs[:, 0, :], evecs[:, 1, :], evecs[:, 2, :], kt), axis=-1)
    return params, s0


def _compute_rtk_cpu(params):
    """Vectorized rescaled radial tensor kurtosis (RTK) matching DIPY DiffusionKurtosisFit.rtk."""
    import numpy as np
    from dipy.reconst.dki import (
        _positive_evals,
        ind_ele,
        mean_diffusivity,
        radial_diffusivity,
        split_dki_param,
    )

    fevals, fevecs, fkt = split_dki_param(params)
    rtk = np.zeros(len(params), dtype=np.float64)
    rel_i = _positive_evals(fevals[:, 0], fevals[:, 1], fevals[:, 2])
    if np.any(rel_i):
        evecs_rel = fevecs[rel_i]
        kt_rel = fkt[rel_i]
        evals_rel = fevals[rel_i]
        idx_tuples = [
            (il, jl, kl, ll, ind_ele[(il + 1) * (jl + 1) * (kl + 1) * (ll + 1)])
            for il in range(3)
            for jl in range(3)
            for kl in range(3)
            for ll in range(3)
        ]
        il_idx = np.array([t[0] for t in idx_tuples], dtype=np.intp)
        jl_idx = np.array([t[1] for t in idx_tuples], dtype=np.intp)
        kl_idx = np.array([t[2] for t in idx_tuples], dtype=np.intp)
        ll_idx = np.array([t[3] for t in idx_tuples], dtype=np.intp)
        ke_idx = np.array([t[4] for t in idx_tuples], dtype=np.intp)
        u1 = evecs_rel[:, :, 1]
        u2 = evecs_rel[:, :, 2]
        wtk = np.empty(len(evals_rel), dtype=np.float64)
        for s in range(0, len(evals_rel), 8192):
            e = min(len(evals_rel), s + 8192)
            kt_81 = kt_rel[s:e, ke_idx]
            u1_b = u1[s:e]
            u2_b = u2[s:e]
            Wyyyy = np.sum(u1_b[:, il_idx] * u1_b[:, jl_idx] * u1_b[:, kl_idx] * u1_b[:, ll_idx] * kt_81, axis=1)
            Wzzzz = np.sum(u2_b[:, il_idx] * u2_b[:, jl_idx] * u2_b[:, kl_idx] * u2_b[:, ll_idx] * kt_81, axis=1)
            Wyyzz = np.sum(u1_b[:, il_idx] * u1_b[:, jl_idx] * u2_b[:, kl_idx] * u2_b[:, ll_idx] * kt_81, axis=1)
            wtk[s:e] = (3.0 / 8.0) * (Wyyyy + Wzzzz + 2.0 * Wyyzz)
        md_r = mean_diffusivity(evals_rel)
        rd_r = radial_diffusivity(evals_rel)
        rtk[rel_i] = wtk * (md_r ** 2) / (rd_r ** 2)
    return rtk


def _predict_dki_chunk(model, params, s0):
    """Vectorized DKI signal prediction reusing model.design_matrix."""
    import numpy as np
    from dipy.reconst.dki import lower_triangular, split_dki_param

    fevals, fevecs, fkt = split_dki_param(params)
    DT_pred = np.matmul(fevecs * fevals[:, None, :], np.swapaxes(fevecs, -1, -2))
    dt_pred = lower_triangular(DT_pred)
    md_pred = (dt_pred[:, 0] + dt_pred[:, 2] + dt_pred[:, 5]) / 3.0
    X_pred = np.concatenate((dt_pred, fkt * (md_pred[:, None] * md_pred[:, None]), -np.log(s0)[:, None]), axis=1)
    A = model.design_matrix
    if len(params) <= 64:
        return np.array([np.exp(np.dot(A, X_pred[v])) for v in range(len(params))])
    return np.exp(np.dot(X_pred, A.T))


def _resolve_dki_device(device: str) -> str:
    req = (device or "cpu").strip().lower()
    if req == "cpu":
        return "cpu"
    try:
        import torch
        has_cuda = bool(torch.cuda.is_available())
    except ImportError:
        has_cuda = False
    if req == "cuda" and not has_cuda:
        raise RuntimeError("DKI CUDA requested, but torch.cuda.is_available() is False")
    return "cuda" if has_cuda else "cpu"


def _fit_dki_chunk_cuda(model, observed, fit_method_name: str):
    """PyTorch CUDA fp64 WLS/OLS solver with CPU pinv fallback on ill-conditioned voxels."""
    import numpy as np
    import torch
    from dipy.reconst.dki import decompose_tensor, from_lower_triangular

    dev = torch.device("cuda")
    A = model.design_matrix
    inv_A = model.inverse_design_matrix
    A_t = torch.as_tensor(np.ascontiguousarray(A), dtype=torch.float64, device=dev)
    inv_A_t = torch.as_tensor(np.ascontiguousarray(inv_A), dtype=torch.float64, device=dev)
    obs_t = torch.as_tensor(np.ascontiguousarray(observed), dtype=torch.float64, device=dev)
    log_s_t = torch.log(torch.clamp(obs_t, min=float(model.min_signal)))

    if fit_method_name in ("WLS", "UWLLS"):
        col_scale = torch.linalg.norm(A_t, dim=0).clamp(min=1e-30)
        A_eq = A_t / col_scale
        ols_r = torch.matmul(log_s_t, inv_A_t.T)
        w_t = torch.exp(torch.matmul(ols_r, A_t.T))
        w_max = w_t.amax(dim=1, keepdim=True).clamp(min=1e-300)
        w_norm = w_t / w_max
        W_A_eq = w_norm.unsqueeze(2) * A_eq.unsqueeze(0)
        G_eq = torch.matmul(W_A_eq.mT, W_A_eq)
        rhs_eq = torch.matmul(W_A_eq.mT, (w_norm * log_s_t).unsqueeze(2))
        L_eq, info_eq = torch.linalg.cholesky_ex(G_eq)
        diag_L = torch.diagonal(L_eq, dim1=-2, dim2=-1)
        well_cond = (
            (info_eq == 0)
            & torch.isfinite(w_t).all(dim=1)
            & (w_t.amax(dim=1) > 0)
            & (diag_L.amin(dim=-1) > 1e-6 * diag_L.amax(dim=-1))
        )
        res_t = torch.empty((len(observed), 22), dtype=torch.float64, device=dev)
        if torch.any(well_cond):
            sol_eq = torch.cholesky_solve(rhs_eq[well_cond], L_eq[well_cond]).squeeze(-1)
            res_t[well_cond] = sol_eq / col_scale
        res_all = res_t.cpu().numpy()
        well_np = well_cond.cpu().numpy()
        if not np.all(well_np):
            log_s_np = np.log(np.maximum(observed, model.min_signal))
            for v in np.flatnonzero(~well_np):
                ls = log_s_np[v]
                ols_v = np.dot(inv_A, ls)
                wv = np.exp(np.dot(A, ols_v))
                W_A = np.multiply(wv[:, None], A, order="C")
                W_A[W_A == 0] = 0.0
                inv_W_A = np.linalg.pinv(W_A)
                inv_W_A_W = np.multiply(inv_W_A, wv[None, :], order="C")
                inv_W_A_W[inv_W_A_W == 0] = 0.0
                res_all[v] = np.dot(inv_W_A_W, ls)
    else:
        res_all = torch.matmul(log_s_t, inv_A_t.T).cpu().numpy()

    DT = from_lower_triangular(res_all[:, :6])
    evals, evecs = decompose_tensor(DT, min_diffusivity=model.min_diffusivity)
    md_sq = evals.mean(axis=-1) ** 2
    kt = np.where(md_sq[:, None] != 0, res_all[:, 6:21] / md_sq[:, None], 0.0 * res_all[:, 6:21])
    s0 = np.exp(-res_all[:, 21])
    params = np.concatenate((evals, evecs[:, 0, :], evecs[:, 1, :], evecs[:, 2, :], kt), axis=-1)
    return params, s0


def _predict_and_rtk_cuda(model, params, s0, observed):
    """GPU fp64 signal residual (RMSE/NRMSE) and rescaled radial tensor kurtosis (RTK)."""
    import numpy as np
    import torch
    from dipy.reconst.dki import (
        _positive_evals,
        ind_ele,
        lower_triangular,
        mean_diffusivity,
        radial_diffusivity,
        split_dki_param,
    )

    dev = torch.device("cuda")
    fevals, fevecs, fkt = split_dki_param(params)
    DT_pred = np.matmul(fevecs * fevals[:, None, :], np.swapaxes(fevecs, -1, -2))
    dt_pred = lower_triangular(DT_pred)
    md_pred = (dt_pred[:, 0] + dt_pred[:, 2] + dt_pred[:, 5]) / 3.0
    X_pred = np.concatenate((dt_pred, fkt * (md_pred[:, None] * md_pred[:, None]), -np.log(s0)[:, None]), axis=1)

    A_t = torch.as_tensor(np.ascontiguousarray(model.design_matrix), dtype=torch.float64, device=dev)
    X_t = torch.as_tensor(np.ascontiguousarray(X_pred), dtype=torch.float64, device=dev)
    obs_t = torch.as_tensor(np.ascontiguousarray(observed), dtype=torch.float64, device=dev)
    s0_t = torch.as_tensor(np.ascontiguousarray(s0), dtype=torch.float64, device=dev)
    pred_t = torch.exp(torch.matmul(X_t, A_t.T))
    rmse_t = torch.sqrt(torch.mean((pred_t - obs_t) ** 2, dim=1))
    nrmse_t = rmse_t / s0_t
    rmse = rmse_t.cpu().numpy()
    nrmse = nrmse_t.cpu().numpy()

    rtk = np.zeros(len(params), dtype=np.float64)
    rel_i = _positive_evals(fevals[:, 0], fevals[:, 1], fevals[:, 2])
    if np.any(rel_i):
        evecs_rel = torch.as_tensor(np.ascontiguousarray(fevecs[rel_i]), dtype=torch.float64, device=dev)
        kt_rel = torch.as_tensor(np.ascontiguousarray(fkt[rel_i]), dtype=torch.float64, device=dev)
        evals_rel = fevals[rel_i]
        idx_tuples = [
            (il, jl, kl, ll, ind_ele[(il + 1) * (jl + 1) * (kl + 1) * (ll + 1)])
            for il in range(3)
            for jl in range(3)
            for kl in range(3)
            for ll in range(3)
        ]
        il_idx = torch.tensor([t[0] for t in idx_tuples], dtype=torch.long, device=dev)
        jl_idx = torch.tensor([t[1] for t in idx_tuples], dtype=torch.long, device=dev)
        kl_idx = torch.tensor([t[2] for t in idx_tuples], dtype=torch.long, device=dev)
        ll_idx = torch.tensor([t[3] for t in idx_tuples], dtype=torch.long, device=dev)
        ke_idx = torch.tensor([t[4] for t in idx_tuples], dtype=torch.long, device=dev)

        kt_81 = kt_rel[:, ke_idx]
        u1 = evecs_rel[:, :, 1]
        u2 = evecs_rel[:, :, 2]
        Wyyyy = torch.sum(u1[:, il_idx] * u1[:, jl_idx] * u1[:, kl_idx] * u1[:, ll_idx] * kt_81, dim=1)
        Wzzzz = torch.sum(u2[:, il_idx] * u2[:, jl_idx] * u2[:, kl_idx] * u2[:, ll_idx] * kt_81, dim=1)
        Wyyzz = torch.sum(u1[:, il_idx] * u1[:, jl_idx] * u2[:, kl_idx] * u2[:, ll_idx] * kt_81, dim=1)
        wtk = (3.0 / 8.0) * (Wyyyy + Wzzzz + 2.0 * Wyyzz).cpu().numpy()
        md_r = mean_diffusivity(evals_rel)
        rd_r = radial_diffusivity(evals_rel)
        rtk[rel_i] = wtk * (md_r ** 2) / (rd_r ** 2)

    return rmse, nrmse, rtk


def _compute_maps(config: DkiConfig, paths: dict[str, Path], log):
    phase_start = time.monotonic()
    phase_times = {key:0.0 for key in ('preparation','signal_read','tensor_fit','metrics_prediction','map_assignment','file_export')}
    import nibabel as nib
    import numpy as np
    from dipy.reconst.dki import DiffusionKurtosisModel
    import dipy
    global _ORIG_DKI_FIT
    if _ORIG_DKI_FIT is None:
        _ORIG_DKI_FIT = DiffusionKurtosisModel.fit
    dense_supported = dipy.__version__.split('.')[:2] in (['1','11'], ['1','12'])
    dense_used = False

    image, mask, gtab, selected, eligibility = _load_protocol(config)
    flat_indices = np.flatnonzero(mask.ravel())
    metrics = {key: {"finite": 0, "negative": 0, "below_display": 0, "above_display": 0} for key in DKI_METRICS}
    counts = {"fitted_voxels": 0, "valid_voxels": 0, "nonfinite_signal_voxels": 0, "nonpositive_signal_voxels": 0}
    fit_method_name = "WLS" if _method(config).startswith("MKCURVE") else _method(config)
    model = DiffusionKurtosisModel(
        gtab, fit_method=fit_method_name,
        return_S0_hat=True, min_signal=1e-6,
    )
    with tempfile.TemporaryDirectory(prefix=".dki-", dir=config.output_dir) as work:
        work = Path(work)
        source_path = config.raw_input_bundle.dwi_path.resolve()
        if source_path.name.endswith(".gz"):
            uncompressed = work / "dwi.nii"
            from shared.cohort_io import io_admission
            with io_admission('dki_decompression'), gzip.open(source_path, "rb") as source, uncompressed.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            input_image = nib.load(str(uncompressed), mmap="r")
        else:
            input_image = nib.load(str(source_path), mmap="r")
        proxy = input_image.dataobj
        unscaled = proxy.get_unscaled()
        arrays = {}
        for key in paths:
            if key in {"manifest", "qc", "progress"}:
                continue
            shape = image.shape[:3] + ((27,) if key == "params" else ())
            dtype = np.uint8 if key in {"valid_mask", "fit_mask"} else np.float32
            array = np.lib.format.open_memmap(work / f"{key}.npy", mode="w+", dtype=dtype, shape=shape)
            array[:] = 0
            if dtype != np.uint8:
                array.reshape((-1, 27) if key == "params" else (-1,))[flat_indices] = np.nan
            arrays[key] = array
        try:
            resolved_device = _resolve_dki_device(config.device)
            from shared.hardware_adapt import (
                adaptive_cuda_scaled_batch,
                available_ram_bytes,
                empty_cuda_cache_safely,
                is_cuda_oom,
            )

            if resolved_device == "cuda" and config.dense_metrics and dense_supported:
                per_voxel_bytes = max(8192, int(len(selected)) * 22 * 32)
                chunk_step = adaptive_cuda_scaled_batch(
                    per_voxel_bytes,
                    max(config.voxels_per_chunk, 16384),
                    min_batch=256,
                    max_batch=65536,
                )
            elif config.dense_metrics and dense_supported and len(flat_indices) > 64 and config.voxels_per_chunk == 1024:
                chunk_step = 8192
            else:
                chunk_step = config.voxels_per_chunk
            phase_times['preparation'] = time.monotonic()-phase_start
            masked_unscaled = None
            pre_bytes = int(len(flat_indices)) * int(len(selected)) * int(np.dtype(unscaled.dtype).itemsize)
            if len(flat_indices) > 64 and pre_bytes <= 1024**3 and available_ram_bytes() >= 4 * (1024**3):
                stamp_pre = time.monotonic()
                masked_unscaled = np.empty((len(flat_indices), len(selected)), dtype=unscaled.dtype)
                for col_idx, vol_idx in enumerate(selected):
                    vol_arr = np.asarray(unscaled[..., int(vol_idx)])
                    masked_unscaled[:, col_idx] = vol_arr.ravel()[flat_indices]
                phase_times['signal_read'] += time.monotonic() - stamp_pre
            last_progress_time = 0.0
            for start in range(0, len(flat_indices), chunk_step):
                stamp = time.monotonic()
                indices = flat_indices[start:start + chunk_step]
                if masked_unscaled is not None:
                    signals = np.asarray(masked_unscaled[start:start + chunk_step], dtype=np.float64)
                else:
                    coordinates = np.unravel_index(indices, mask.shape)
                    signals = np.asarray(unscaled[coordinates[0], coordinates[1], coordinates[2], :][:, selected], dtype=np.float64)
                signals = signals * proxy.slope + proxy.inter
                finite = np.isfinite(signals).all(axis=1)
                positive = (signals > 0).all(axis=1)
                counts["nonfinite_signal_voxels"] += int(np.count_nonzero(~finite))
                counts["nonpositive_signal_voxels"] += int(np.count_nonzero(finite & ~positive))
                fit_indices = indices[finite & positive]
                phase_times['signal_read'] += time.monotonic()-stamp
                if len(fit_indices):
                    stamp = time.monotonic()
                    observed = signals[finite & positive]
                    use_fast_dense = (
                        config.dense_metrics
                        and dense_supported
                        and getattr(type(model), "fit", None) is _ORIG_DKI_FIT
                        and "fit" not in model.__dict__
                    )
                    if use_fast_dense and resolved_device == "cuda":
                        from dipy.reconst.dki import DiffusionKurtosisFit
                        try:
                            params, s0 = _fit_dki_chunk_cuda(model, observed, fit_method_name)
                            phase_times['tensor_fit'] += time.monotonic()-stamp
                            stamp = time.monotonic()
                            fitted = DiffusionKurtosisFit(model, params, model_S0=s0)
                            dense_used = True
                            rmse, nrmse, rtk_cuda = _predict_and_rtk_cuda(model, params, s0, observed)
                            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                                raw = {
                                    key: np.asarray(getattr(fitted, key)(min_kurtosis=None, max_kurtosis=None)).reshape(-1)
                                    for key in ("mk", "ak", "rk", "mkt")
                                }
                                raw["rtk"] = rtk_cuda
                                raw["kfa"] = np.asarray(fitted.kfa).reshape(-1)
                        except BaseException as exc:
                            if not is_cuda_oom(exc):
                                raise
                            empty_cuda_cache_safely()
                            params, s0 = _fit_dki_chunk(model, observed, fit_method_name)
                            phase_times['tensor_fit'] += time.monotonic()-stamp
                            stamp = time.monotonic()
                            fitted = DiffusionKurtosisFit(model, params, model_S0=s0)
                            dense_used = True
                            predictions = _predict_dki_chunk(model, params, s0)
                            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                                raw = {
                                    key: np.asarray(getattr(fitted, key)(min_kurtosis=None, max_kurtosis=None)).reshape(-1)
                                    for key in ("mk", "ak", "rk", "mkt")
                                }
                                raw["rtk"] = (
                                    _compute_rtk_cpu(params)
                                    if len(params) > 64
                                    else np.asarray(fitted.rtk(min_kurtosis=None, max_kurtosis=None)).reshape(-1)
                                )
                                raw["kfa"] = np.asarray(fitted.kfa).reshape(-1)
                                rmse = np.sqrt(np.mean((predictions - observed) ** 2, axis=1))
                                nrmse = rmse / s0
                    elif use_fast_dense:
                        from dipy.reconst.dki import DiffusionKurtosisFit
                        params, s0 = _fit_dki_chunk(model, observed, fit_method_name)
                        phase_times['tensor_fit'] += time.monotonic()-stamp
                        stamp = time.monotonic()
                        fitted = DiffusionKurtosisFit(model, params, model_S0=s0)
                        dense_used = True
                        predictions = _predict_dki_chunk(model, params, s0)
                        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                            raw = {
                                key: np.asarray(getattr(fitted, key)(min_kurtosis=None, max_kurtosis=None)).reshape(-1)
                                for key in ("mk", "ak", "rk", "mkt")
                            }
                            raw["rtk"] = (
                                _compute_rtk_cpu(params)
                                if len(params) > 64
                                else np.asarray(fitted.rtk(min_kurtosis=None, max_kurtosis=None)).reshape(-1)
                            )
                            raw["kfa"] = np.asarray(fitted.kfa).reshape(-1)
                            rmse = np.sqrt(np.mean((predictions - observed) ** 2, axis=1))
                            nrmse = rmse / s0
                    else:
                        fitted = model.fit(observed)
                        params = np.asarray(fitted.model_params)
                        s0 = np.asarray(fitted.model_S0).reshape(-1)
                        phase_times['tensor_fit'] += time.monotonic()-stamp
                        stamp = time.monotonic()
                        if config.dense_metrics and dense_supported:
                            # Keep the author fit/parameters. Evaluate its existing
                            # array-capable metrics without per-voxel Python wrappers.
                            from dipy.reconst.multi_voxel import MultiVoxelFit
                            from dipy.reconst.dki import DiffusionKurtosisFit
                            if isinstance(fitted, MultiVoxelFit):
                                fitted = DiffusionKurtosisFit(model, params, model_S0=s0)
                                dense_used = True
                        predictions = np.asarray(fitted.predict(gtab, S0=s0))
                        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                            raw = {
                                key: np.asarray(getattr(fitted, key)(min_kurtosis=None, max_kurtosis=None)).reshape(-1)
                                for key in DKI_METRICS if key != "kfa"
                            }
                            raw["kfa"] = np.asarray(fitted.kfa).reshape(-1)
                            rmse = np.sqrt(np.mean((predictions - observed) ** 2, axis=1))
                            nrmse = rmse / s0
                    phase_times['metrics_prediction'] += time.monotonic()-stamp
                    stamp = time.monotonic()
                    valid = np.isfinite(params).all(axis=1) & np.isfinite(s0) & (s0 > 0)
                    valid &= (np.asarray(fitted.evals) > model.min_diffusivity).all(axis=1)
                    valid &= np.isfinite(rmse) & np.isfinite(nrmse)
                    for key, values in raw.items():
                        upper = 1.0 if key == "kfa" else config.display_max_kurtosis
                        lower = 0.0 if key == "kfa" else config.display_min_kurtosis
                        arrays[f"{key}_raw"].ravel()[fit_indices] = values
                        arrays[key].ravel()[fit_indices] = np.clip(values, lower, upper)
                        metrics[key]["finite"] += int(np.count_nonzero(np.isfinite(values)))
                        metrics[key]["negative"] += int(np.count_nonzero(values < 0))
                        metrics[key]["below_display"] += int(np.count_nonzero(values < lower))
                        metrics[key]["above_display"] += int(np.count_nonzero(values > upper))
                        valid &= np.isfinite(values) & (values >= lower) & (values <= upper)
                    arrays["params"].reshape(-1, 27)[fit_indices] = params
                    arrays["s0"].ravel()[fit_indices] = s0
                    arrays["rmse"].ravel()[fit_indices] = rmse
                    arrays["nrmse"].ravel()[fit_indices] = nrmse
                    arrays["fit_mask"].ravel()[fit_indices] = 1
                    arrays["valid_mask"].ravel()[fit_indices] = valid.astype(np.uint8)
                    counts["fitted_voxels"] += len(fit_indices)
                    counts["valid_voxels"] += int(np.count_nonzero(valid))
                    phase_times['map_assignment'] += time.monotonic()-stamp
                completed = min(start + len(indices), len(flat_indices))
                now_prog = time.monotonic()
                if start == 0 or completed == len(flat_indices) or (now_prog - last_progress_time >= 1.0):
                    _progress(paths["progress"], config, "running", completed_voxels=completed, total_voxels=len(flat_indices), **counts)
                    last_progress_time = now_prog
                if log:
                    log(f"[DKI] Processed {completed}/{len(flat_indices)} mask voxels ({counts['valid_voxels']} valid)")
            if not counts["fitted_voxels"]:
                raise ValueError("DKI has no masked voxels with finite, strictly positive selected measurements")
            if not counts["valid_voxels"]:
                raise ValueError("DKI has no valid quantitative voxels; inspect acquisition and raw fit quality")
            stamp = time.monotonic()
            for key, array in arrays.items():
                array.flush()
                header = image.header.copy()
                header.set_data_dtype(array.dtype)
                staged = work / paths[key].name
                nib.save(nib.Nifti1Image(array, image.affine, header), str(staged))
                staged.replace(paths[key])
            phase_times['file_export'] = time.monotonic()-stamp
        finally:
            for array in arrays.values():
                array._mmap.close()
            if isinstance(unscaled, np.memmap):
                unscaled._mmap.close()
    phase_times['other'] = max(0.,time.monotonic()-phase_start-sum(phase_times.values()))
    return {
        "kind": "microstructure_dki_qc", "status": "completed", "subject_id": config.subject_id,
        "fit_method": "WLS" if _method(config).startswith("MKCURVE") else _method(config),
        "selected_fit_method": _method(config), "eligibility": eligibility,
        **counts, "mask_voxels": len(flat_indices), "metrics": metrics,
        "invalid_voxels": len(flat_indices) - counts["valid_voxels"],
        "valid_fraction": counts["valid_voxels"] / len(flat_indices),
        "display_bounds": {key: [0, 1] if key == "kfa" else [config.display_min_kurtosis, config.display_max_kurtosis] for key in DKI_METRICS},
        "raw_metric_bounds": None, "smoothing": False, "human_review": "pending",
        "nrmse_definition": "sqrt(mean((prediction - observed)^2)) / fitted_S0; selected volumes only",
        "invalid_policy": "Outside-mask zero; unfitted in-mask NaN; fitted raw values retained regardless of validity. No hole filling.",
        "positive_signal_floor": 1e-6,
        "validity_policy": "finite parameters/residuals; positive diffusion eigenvalues/S0; all six raw metrics within documented display bounds",
        "warnings": eligibility["warnings"] + [
            "Unconstrained WLS/OLS can yield negative or implausible kurtosis; use raw maps and validity mask for quantitative review.",
            "Standard DIPY kurtosis maps are not the author's MK-Curve correction.",
        ],
        "reference": DKI_REFERENCE,
        "metric_evaluation": "DIPY array-level metrics, unchanged fit" if dense_used else "DIPY per-voxel metrics",
        "device": resolved_device,
        "dense_metrics_requested": config.dense_metrics,
        "dense_metrics_supported": dense_supported,
        "phase_runtime_seconds": phase_times,
    }


@scheduled_stage(lambda config: 'microstructure_mkcurve' if _method(config).startswith('MKCURVE') else 'microstructure_dki')
@dki_threads
def run_dki(config: DkiConfig, log: Optional[Callable[[str], None]] = None) -> DkiResult:
    """Fit only eligible acquisitions, with bounded signal chunks and disk-backed maps."""
    paths = dki_output_paths(config.subject_id, config.output_dir)
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    _progress(paths["progress"], config, "checking")
    try:
        eligibility = inspect_dki_inputs(config)
        if not eligibility["eligible"]:
            raise ValueError("DKI acquisition/backend is ineligible: " + "; ".join(eligibility["reasons"]))
        if dki_outputs_current(config):
            previous = _read_json(paths["manifest"])
            if log:
                log("[DKI] Reusing matching completed stage, including raw maps and residual QC")
            _progress(paths["progress"], config, "completed", reused=True)
            return _result(config, "cached", paths, previous.get("mkcurve"))
        signature = expected_dki_signature(config)
        manifest = {"kind": "microstructure_dki_stage", "version": 1, "status": "running", "signature": signature}
        _write_json(paths["manifest"], manifest)
        _progress(paths["progress"], config, "running")
        qc = _compute_maps(config, paths, log)
        mkcurve = None
        if _method(config).startswith("MKCURVE"):
            if _method(config) == "MKCURVE_MATLAB":
                from microstructure.mkcurve_matlab import run_mkcurve
                backend_options = {"matlab_executable": config.matlab_executable, "mkcurve_root": config.mkcurve_root}
            else:
                from microstructure.mkcurve import run_mkcurve
                backend_options = {"voxels_per_chunk": config.mkcurve_voxels_per_chunk,
                                   "num_workers": config.mkcurve_workers, "worker_threads": config.num_threads,
                                   "chunk_schedule": config.mkcurve_chunk_schedule}
                if config.device.strip().lower() != "cpu":
                    backend_options["device"] = config.device

            bundle = config.raw_input_bundle
            try:
                mkcurve = run_mkcurve(
                    bundle.dwi_path, bundle.bval_path, bundle.bvec_path, bundle.mask_path,
                    config.output_dir / "mkcurve", b0_threshold=config.b0_threshold,
                    max_bvalue=config.max_bvalue, log=log, **backend_options,
                )
            except ValueError as exc:
                # D4: MK-Curve is the default; a stop caused by the signal data names the WLS option and never
                # falls back by itself. Option and input errors (T-N2) keep their own message.
                from microstructure.mkcurve import NO_VALID_VOXELS_STOP, SIGNAL_SCALE_STOP
                if not any(reason in str(exc) for reason in (SIGNAL_SCALE_STOP, NO_VALID_VOXELS_STOP)):
                    raise
                raise ValueError(f"{exc}; re-run with DKI fit method WLS (--dki-fit-method WLS)") from exc
            manifest["mkcurve"] = mkcurve
            manifest["mkcurve_record_artifact"] = _signature(Path(mkcurve["record_path"]))
            manifest["mkcurve_artifacts"] = {key: _signature(Path(value)) for key, value in mkcurve["maps"].items()}
            qc["mkcurve"] = {"automatic_qc_passed": mkcurve["automatic_qc_passed"], "human_review": "pending", "record_path": mkcurve["record_path"]}
        # Refuse publication if inputs or options changed during the fit.
        if signature != expected_dki_signature(config):
            raise RuntimeError("DKI inputs/backend changed while fitting; outputs are not reusable")
        manifest.update({"status": "completed", "subject_id": config.subject_id,
                         "artifacts": {key: _signature(path) for key, path in paths.items() if key not in {"manifest", "qc", "progress"}},
                         "outputs": {key: str(path) for key, path in paths.items()},
                         "runtime_seconds": time.monotonic() - start, "smoothing": False, "reference": DKI_REFERENCE})
        _write_json(paths["qc"], qc)
        _write_json(paths["manifest"], manifest)
        _progress(paths["progress"], config, "completed", completed_voxels=qc["mask_voxels"], total_voxels=qc["mask_voxels"], runtime_seconds=time.monotonic() - start)
        return _result(config, "completed", paths, mkcurve)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        _write_json(paths["manifest"], {"kind": "microstructure_dki_stage", "status": "failed", "subject_id": config.subject_id, "error": error})
        _write_json(paths["qc"], {"kind": "microstructure_dki_qc", "status": "failed", "subject_id": config.subject_id, "error": error})
        _progress(paths["progress"], config, "failed", error=error)
        raise
