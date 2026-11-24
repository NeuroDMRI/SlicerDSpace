# Python translation of MK-Curve/run_MKCurve.m and abnormal_voxel_detection.m.
# MK-Curve Contribution and Software License Agreement, BWH, June 2019.
# See licenses/MK-Curve-LICENSE.txt. Modified for SlicerDSpace, 2026.
"""Native Python MK-Curve port, distinct from standard DIPY DKI maps."""

from datetime import datetime, timezone
from contextlib import ExitStack
import hashlib
import importlib.metadata
from pathlib import Path
import tempfile
import time

UPSTREAM_URL = "https://github.com/zhangfanmark/MK-Curve"
UPSTREAM_COMMIT = "9a213e1a5850fe50c998793f31ed852df00242f8"
# Bumped to 3 after fixing the multi-threaded LAPACK dormqr diagonal-mutation race
# and updating the batched WLLS/Rodrigues solver so pre-fix cached MK-Curve outputs invalidate.
PORT_VERSION = 3
# The two stops that come from the signal data. Only these get the WLS hint in the DKI stage (T-N2).
SIGNAL_SCALE_STOP = "Signal scale cannot support the author's integer synthetic-b0 grid; rescale acquisition explicitly"
NO_VALID_VOXELS_STOP = "MK-Curve has no valid corrected quantitative voxels; inspect raw curves and acquisition"


def metric_validity(maps, coefficients_finite, curve_valid, nrmse):
    """Separate per-parameter eligibility from the stricter joint-map gate.

    A failed radial kurtosis must not erase an independently valid mean
    kurtosis. All masks still require a finite fit and positive diffusion
    tensor. Raw values and the original joint gate remain unchanged.
    """
    import numpy as np

    base = np.asarray(coefficients_finite, dtype=bool) & curve_valid & np.isfinite(nrmse)
    for key in ("e1", "e2", "e3"):
        if key in maps:
            base &= np.isfinite(maps[key]) & (maps[key] > 0)
    masks = {}
    for key, values in maps.items():
        lower, upper = (0, 1) if key == "fa" else ((0, 3) if key in {"mk", "ak", "rk"} else (0, .005))
        masks[key] = base & np.isfinite(values) & (values >= lower) & (values <= upper)
    return masks


def inspect_mkcurve_backend(*_legacy_paths):
    """Discover bundled Python code/resources, never require MATLAB."""
    files = [Path(__file__), Path(__file__).with_name("mkcurve_tensor.py"),
             Path(__file__).with_name("mkcurve_directions.json"), Path(__file__).with_name("mkcurve_inputs.py"),
             Path(__file__).with_name('chunk_dispatch.py')]
    missing = [str(path) for path in files if not path.is_file()]
    libraries = {}
    for key in ("numpy", "scipy", "nibabel"):
        try:
            libraries[key] = importlib.metadata.version(key)
        except importlib.metadata.PackageNotFoundError:
            missing.append("Python package " + key)
    return {"available": not missing, "missing_required": missing, "implementation": "python_port",
            "port_version": PORT_VERSION, "upstream_url": UPSTREAM_URL, "upstream_commit": UPSTREAM_COMMIT,
            "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files if path.is_file()},
            "libraries": libraries}


def synthetic_b0_grid(b0):
    import numpy as np

    mean = float(np.mean(b0))
    lower, upper = int(np.floor(mean * .1 + .5)), int(np.floor(mean * 2 + .5))
    step = int(np.floor((upper - lower) / 200 + .5))
    if lower <= 0 or step <= 0:
        raise ValueError(SIGNAL_SCALE_STOP)
    return np.arange(lower, upper + 1, step, dtype=float)


def detect_abnormal(curves, b0, grid, threshold=.5):
    """Literal upstream index convention (MATLAB 1-based translated explicitly)."""
    import numpy as np

    if not 0 <= threshold <= 1:
        raise ValueError("MK-Curve threshold must be in [0, 1]")
    curves = np.array(curves, dtype=float, copy=True)
    grid, b0 = np.asarray(grid, dtype=float), np.asarray(b0, dtype=float)
    if curves.shape != (len(grid), len(b0)) or len(grid) < 3 or not np.all(np.diff(grid) > 0):
        raise ValueError("Invalid MK-Curve matrix or increasing b0 grid")
    finite = np.isfinite(curves).all(axis=0) & np.isfinite(b0)
    curves[:, ~finite] = 0
    curves[curves > 5] = -1
    reverse_index = np.argmax(curves[::-1] < 0, axis=0) + 1
    reverse_index[reverse_index == 1] = len(grid)
    zero_index = len(grid) - reverse_index + 1  # zero-based, upstream pos_zero_MK - 1
    curves[np.arange(len(grid))[:, None] < zero_index[None, :]] = 0
    max_index = np.argmax(curves, axis=0)
    zero_b0, max_b0 = grid[zero_index], grid[max_index]
    max_mk = curves[max_index, np.arange(len(b0))]
    corrected_b0 = zero_b0 * (1 - threshold) + max_b0 * threshold
    first_greater = np.searchsorted(grid, b0, side="right")
    unusual = ((max_index - zero_index <= 2) & (first_greater < len(grid))
               & (first_greater <= max_index + 1 + 5) & finite)
    abnormal = ((b0 <= corrected_b0) | unusual) & finite
    return {"zero_mk_b0": zero_b0, "max_mk_b0": max_b0, "max_mk": max_mk,
            "abnormal_mask": abnormal, "unusual_mask": unusual, "curve_valid_mask": finite,
            "corrected_b0": np.where(abnormal, corrected_b0, b0)}


def repair_nonpositive(data, mask):
    """Upstream sequential positive-neighbour mean, MATLAB voxel ordering."""
    import numpy as np

    repaired, unresolved = 0, 0
    for volume in range(data.shape[-1]):
        image = data[..., volume]
        bad = mask & (~np.isfinite(image) | (image <= 0))
        for flat in np.flatnonzero(bad.ravel(order="F")):
            xyz = np.unravel_index(flat, mask.shape, order="F")
            ranges = tuple(slice(max(0, x - 1), min(size, x + 2)) for x, size in zip(xyz, mask.shape))
            neighbors = image[ranges]
            values = neighbors[np.isfinite(neighbors) & (neighbors > 0)]
            if len(values):
                image[xyz] = values.mean()
                repaired += 1
            else:
                image[xyz] = np.nan
                unresolved += 1
    return {"repaired_measurements": repaired, "unresolved_measurements": unresolved,
            "policy": "sequential 26-neighbour positive-signal mean; working copy only; no parameter-map smoothing"}


def _fit_chunk(payload, *, reuse_signal_logs=None):
    import numpy as np
    from microstructure.mkcurve_tensor import fit_wlls, mean_kurtosis, parameter_maps

    signals, design, grid, threshold, solver = payload
    if reuse_signal_logs is None:
        reuse_signal_logs = True
    b0 = signals[:, 0].copy()
    fitter = None
    if reuse_signal_logs:
        from microstructure.mkcurve_tensor import _SyntheticB0Fitter
        fitter = _SyntheticB0Fitter(signals, design, solver=solver)
    if fitter is not None and len(grid) >= 64:
        original = fitter.fit_fast(b0)
        curves = fitter.fit_grid(grid)
        detection = detect_abnormal(curves, b0, grid, threshold)
        pos_max = detection["curve_valid_mask"] & (detection["max_mk"] > 0)
        if np.any(pos_max):
            mk_at_max = mean_kurtosis(fitter.fit_fast(detection["max_mk_b0"], row_mask=pos_max))
            detection["max_mk"][pos_max] = np.clip(mk_at_max[pos_max], 0.0, None)
        abnormal = detection["abnormal_mask"]
        corrected = original.copy()
        if np.any(abnormal):
            corrected[abnormal] = fitter.fit_fast(detection["corrected_b0"], row_mask=abnormal)[abnormal]
    else:
        original = fitter.fit(b0) if fitter else fit_wlls(signals, design, solver=solver)
        curves = np.empty((len(grid), len(signals)))
        for step, value in enumerate(grid):
            signals[:, 0] = value
            coefficients = fitter.fit(signals[:, 0]) if fitter else fit_wlls(signals, design, solver=solver)
            curves[step] = mean_kurtosis(coefficients)
        detection = detect_abnormal(curves, b0, grid, threshold)
        signals[:, 0] = detection["corrected_b0"]
        corrected = fitter.fit(signals[:, 0]) if fitter else fit_wlls(signals, design, solver=solver)

    orig_maps = parameter_maps(original)
    orig_finite = np.isfinite(original).all(axis=1)
    orig_valid = orig_finite.copy()
    for key, values in orig_maps.items():
        lower, upper = (0, 1) if key == "fa" else ((0, 3) if key in {"mk", "ak", "rk"} else (0, .005))
        orig_valid &= np.isfinite(values) & (values >= lower) & (values <= upper)
        detection[f"original_{key}"] = values
    orig_observed = signals.copy()
    orig_observed[:, 0] = b0
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        predicted = np.exp(original @ design.T)
        detection["original_nrmse"] = np.sqrt(np.mean((predicted - orig_observed) ** 2, axis=1)) / np.exp(original[:, 0])
    orig_valid &= detection["curve_valid_mask"] & np.isfinite(detection["original_nrmse"])
    detection["original_valid_mask"] = orig_valid
    for key, mask in metric_validity(orig_maps, orig_finite,
                                    detection["curve_valid_mask"], detection["original_nrmse"]).items():
        detection[f"original_{key}_valid_mask"] = mask

    # Unchanged (~abnormal) voxels share exact coefficients and b0 with original, so reuse their maps/NRMSE.
    abnormal = detection["abnormal_mask"]
    corr_maps = {key: values.copy() for key, values in orig_maps.items()}
    corr_nrmse = detection["original_nrmse"].copy()
    if np.any(abnormal):
        ab_idx = np.flatnonzero(abnormal)
        ab_maps = parameter_maps(corrected[ab_idx])
        for key, values in ab_maps.items():
            corr_maps[key][ab_idx] = values
        ab_observed = signals[ab_idx].copy()
        ab_observed[:, 0] = detection["corrected_b0"][ab_idx]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            pred_ab = np.exp(corrected[ab_idx] @ design.T)
            corr_nrmse[ab_idx] = np.sqrt(np.mean((pred_ab - ab_observed) ** 2, axis=1)) / np.exp(corrected[ab_idx, 0])
    corr_finite = np.isfinite(corrected).all(axis=1)
    corr_valid = corr_finite.copy()
    for key, values in corr_maps.items():
        lower, upper = (0, 1) if key == "fa" else ((0, 3) if key in {"mk", "ak", "rk"} else (0, .005))
        corr_valid &= np.isfinite(values) & (values >= lower) & (values <= upper)
        detection[f"corrected_{key}"] = values
    detection["corrected_nrmse"] = corr_nrmse
    corr_valid &= detection["curve_valid_mask"] & np.isfinite(corr_nrmse)
    detection["corrected_valid_mask"] = corr_valid
    for key, mask in metric_validity(corr_maps, corr_finite,
                                    detection["curve_valid_mask"], corr_nrmse).items():
        detection[f"corrected_{key}_valid_mask"] = mask
    return original, corrected, curves, detection


def _resolve_mkcurve_device(device: str) -> str:
    dev = (device or "cpu").strip().lower()
    if dev not in ("cpu", "cuda", "mps", "auto") and not dev.startswith("cuda:"):
        raise ValueError("MK-Curve device must be 'cpu', 'cuda', 'mps', or 'auto'")
    if dev in ("cpu", "mps"):
        return "cpu"
    try:
        import torch
        has_cuda = bool(torch.cuda.is_available())
    except Exception:
        has_cuda = False
    if (dev == "cuda" or dev.startswith("cuda:")) and not has_cuda:
        raise RuntimeError("MK-Curve device='cuda' requested, but torch.cuda.is_available() is False")
    if has_cuda and dev.startswith("cuda:") and dev != "cuda:0":
        return dev
    return "cuda" if has_cuda else "cpu"


def _fit_chunk_cuda(payload):
    import numpy as np
    import torch
    from shared.hardware_adapt import empty_cuda_cache_safely, is_cuda_oom
    from microstructure.mkcurve_tensor import (
        _CudaSyntheticB0Fitter,
        _mean_kurtosis_cuda,
        _parameter_maps_cuda,
    )

    if len(payload) >= 7:
        signals, design, grid, threshold, solver, device_str, explicit_cuda = payload[:7]
    elif len(payload) >= 6:
        signals, design, grid, threshold, solver, device_str = payload[:6]
        explicit_cuda = False
    else:
        signals, design, grid, threshold, solver = payload
        device_str = "cuda"
        explicit_cuda = False
    try:
        fitter = _CudaSyntheticB0Fitter(signals, design, grid, solver=solver, device=device_str)
        if fitter.ops is None:
            return _fit_chunk((signals, design, grid, threshold, solver))

        b0 = signals[:, 0].copy()
        original = fitter.fit(b0)
        curves = fitter.fit_grid()
        detection = detect_abnormal(curves, b0, grid, threshold)
        pos_max = detection["curve_valid_mask"] & (detection["max_mk"] > 0)
        if np.any(pos_max):
            mk_coeff = fitter.fit(detection["max_mk_b0"])
            mk_coeff_t = torch.as_tensor(np.ascontiguousarray(mk_coeff), dtype=torch.float64, device=fitter.dev)
            mk_at_max = _mean_kurtosis_cuda(mk_coeff_t, fitter.ops["diff_basis"], fitter.ops["kurt_basis"]).cpu().numpy()
            detection["max_mk"][pos_max] = np.clip(mk_at_max[pos_max], 0.0, None)

        signals[:, 0] = detection["corrected_b0"]
        corrected = fitter.fit(signals[:, 0])
        design_t = torch.as_tensor(np.ascontiguousarray(design), dtype=torch.float64, device=fitter.dev)
        for kind, coefficients in (("original", original), ("corrected", corrected)):
            maps = _parameter_maps_cuda(coefficients, fitter.ops, device_str=device_str)
            valid = np.isfinite(coefficients).all(axis=1)
            for key, values in maps.items():
                lower, upper = (0, 1) if key == "fa" else ((0, 3) if key in {"mk", "ak", "rk"} else (0, .005))
                valid &= np.isfinite(values) & (values >= lower) & (values <= upper)
                detection[f"{kind}_{key}"] = values
            observed = signals.copy()
            if kind == "original":
                observed[:, 0] = b0
            obs_t = torch.as_tensor(np.ascontiguousarray(observed), dtype=torch.float64, device=fitter.dev)
            coeff_t = torch.as_tensor(np.ascontiguousarray(coefficients), dtype=torch.float64, device=fitter.dev)
            pred_t = torch.exp(torch.matmul(coeff_t, design_t.T))
            nrmse_t = torch.sqrt(torch.mean((pred_t - obs_t) ** 2, dim=1)) / torch.exp(coeff_t[:, 0])
            detection[f"{kind}_nrmse"] = nrmse_t.cpu().numpy()
            valid &= detection["curve_valid_mask"] & np.isfinite(detection[f"{kind}_nrmse"])
            detection[f"{kind}_valid_mask"] = valid
            for key, mask in metric_validity(maps, np.isfinite(coefficients).all(axis=1),
                                            detection["curve_valid_mask"], detection[f"{kind}_nrmse"]).items():
                detection[f"{kind}_{key}_valid_mask"] = mask
        return original, corrected, curves, detection
    except BaseException as exc:
        if not is_cuda_oom(exc):
            raise
        empty_cuda_cache_safely()
        if len(signals) > 128:
            mid = len(signals) // 2
            orig_a, corr_a, cur_a, det_a = _fit_chunk_cuda(
                (signals[:mid].copy(), design, grid, threshold, solver, device_str, explicit_cuda)
            )
            orig_b, corr_b, cur_b, det_b = _fit_chunk_cuda(
                (signals[mid:].copy(), design, grid, threshold, solver, device_str, explicit_cuda)
            )
            merged = {k: np.concatenate([det_a[k], det_b[k]], axis=0) for k in det_a}
            return (
                np.concatenate([orig_a, orig_b], axis=0),
                np.concatenate([corr_a, corr_b], axis=0),
                np.concatenate([cur_a, cur_b], axis=1),
                merged,
            )
        if explicit_cuda:
            raise
        return _fit_chunk((signals.copy(), design, grid, threshold, solver))


def _initialize_worker(threads):
    if threads is None:
        return
    import os
    from threadpoolctl import threadpool_limits
    global _worker_thread_limit
    for name in (
        'OMP_NUM_THREADS',
        'OPENBLAS_NUM_THREADS',
        'MKL_NUM_THREADS',
        'VECLIB_MAXIMUM_THREADS',
        'NUMEXPR_NUM_THREADS',
    ):
        os.environ[name] = str(threads)
    _worker_thread_limit = threadpool_limits(limits=threads)


def run_mkcurve(dwi_path, bval_path, bvec_path, mask_path, output_dir, *,
                b0_threshold=50., max_bvalue=3000., voxels_per_chunk=64,
                threshold=.5, grid=None, log=None, num_workers=1, solver="reference",
                worker_threads=None, chunk_schedule='dynamic', device="cpu", **legacy):
    """Scan synthetic b0, identify abnormal curves, refit b0-corrected signals.

    Original inputs remain immutable. Raw untruncated maps, validity masks,
    curve samples, coefficients and provenance are retained for review.
    """
    import gzip
    import nibabel as nib
    import numpy as np
    from microstructure.mkcurve_tensor import design_matrix
    from microstructure.dki import _signature, _write_json
    from microstructure.mkcurve_inputs import _prepare_masked_signals

    if (type(voxels_per_chunk) is not int or voxels_per_chunk < 1
            or (num_workers is not None and (type(num_workers) is not int or num_workers < 1))
            or not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("MK-Curve requires a positive chunk size and finite threshold in [0,1]")
    if worker_threads is not None and (type(worker_threads) is not int or worker_threads < 1):
        raise ValueError('MK-Curve worker_threads must be a positive integer')
    if chunk_schedule not in ('dynamic', 'ordered'):
        raise ValueError('MK-Curve chunk schedule must be dynamic or ordered')
    resolved_device = _resolve_mkcurve_device(device)
    is_cuda_dev = resolved_device.startswith("cuda")
    if not is_cuda_dev and worker_threads is None:
        # Tiny N x 22 LAPACK dgelsy solves suffer severe OpenBLAS thread-barrier spinning
        # (e.g. 3.5x-10x slowdown on macOS Rosetta/Apple Silicon) if BLAS is left unbounded,
        # even when num_workers == 1.
        worker_threads = 1
    if num_workers is None:
        if is_cuda_dev:
            num_workers = 1
        else:
            from shared.numeric_threads import effective_cpu_threads
            budget = effective_cpu_threads(default=8)
            num_workers = max(1, budget // (worker_threads or 1))
    import os
    reservation = os.environ.get('SLICERDSPACE_STAGE_CPUS')
    if reservation and num_workers * (worker_threads or 1) > int(reservation):
        raise ValueError('MK-Curve workers and threads exceed the stage CPU reservation')
    if solver not in {"reference", "batched"}:
        raise ValueError("MK-Curve solver must be reference or batched")
    if any(value is not None for value in legacy.values()):
        raise ValueError("MATLAB/source paths belong to MKCURVE_MATLAB, not the Python port")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    backend = inspect_mkcurve_backend()
    inputs = {key: _signature(Path(path)) for key, path in
              zip(("dwi", "bval", "bvec", "mask"), (dwi_path, bval_path, bvec_path, mask_path))}
    progress = root / "mkcurve_progress.json"
    _write_json(progress, {"status": "running", "implementation": "python_port", "device": resolved_device})
    _write_json(root / "mkcurve_record.json", {"kind": "mkcurve_python_port", "status": "running", "inputs": inputs})
    try:
        with tempfile.TemporaryDirectory(prefix=".mkcurve-", dir=root, ignore_cleanup_errors=True) as temporary, ExitStack() as handles:
            masked_signals, mask, coords, bvals_sel, bvecs_sel, reference, preparation, repair = _prepare_masked_signals(
                Path(dwi_path), Path(bval_path), Path(bvec_path), Path(mask_path), Path(temporary), b0_threshold, max_bvalue,
                allow_nonfinite=True)
            handles.callback(masked_signals._mmap.close)
            count = len(coords[0])
            b0 = masked_signals[:, 0].copy()
            grid = synthetic_b0_grid(b0) if grid is None else np.asarray(grid, dtype=float)
            if len(grid) < 3 or not np.isfinite(grid).all() or np.any(grid <= 0) or not np.all(np.diff(grid) > 0):
                raise ValueError("Synthetic b0 grid must contain at least three finite positive increasing values")
            np.savetxt(root / "synthetic_b0_grid.csv", grid, delimiter=",")
            design = design_matrix(bvals_sel, bvecs_sel)
            curves = np.lib.format.open_memmap(root / "mk_vs_synthetic_b0.npy", mode="w+", dtype="float64", shape=(len(grid), count))
            original_coefficients = np.lib.format.open_memmap(root / "original_coefficients.npy", mode="w+", dtype="float64", shape=(count, 22))
            corrected_coefficients = np.lib.format.open_memmap(root / "corrected_coefficients.npy", mode="w+", dtype="float64", shape=(count, 22))
            results = {}
            try:
                is_cuda_dev = resolved_device.startswith("cuda")
                req_mkc_dev = str(device or "auto").strip().lower()
                explicit_cuda = req_mkc_dev == "cuda" or req_mkc_dev.startswith("cuda:")
                eff_workers = 1 if is_cuda_dev else num_workers
                if is_cuda_dev:
                    from shared.hardware_adapt import adaptive_cuda_scaled_batch
                    per_voxel_bytes = max(16384, int(len(grid)) * 22 * 64)
                    eff_chunk = adaptive_cuda_scaled_batch(
                        per_voxel_bytes,
                        max(voxels_per_chunk, 4096) if voxels_per_chunk <= 512 else voxels_per_chunk,
                        min_batch=128,
                        max_batch=16384,
                        device=resolved_device,
                        windows_tdr_cap=4096,
                        fp64=True,
                        consumer_fp64_tdr_cap=1024,
                    )
                elif count >= 1024 and len(grid) >= 64 and voxels_per_chunk == 64:
                    # Larger BLAS/LAPACK chunks amortise Python/dispatch overhead on full-brain scans
                    # while keeping explicit chunk overrides and small synthetic tests untouched.
                    eff_chunk = 256
                else:
                    eff_chunk = voxels_per_chunk
                chunk_fn = _fit_chunk_cuda if is_cuda_dev else _fit_chunk
                if worker_threads is not None:
                    try:
                        from threadpoolctl import threadpool_limits
                        handles.enter_context(threadpool_limits(limits=worker_threads))
                    except Exception:
                        pass
                if eff_workers > 1:
                    import sys
                    main_file = getattr(sys.modules.get("__main__"), "__file__", None)
                    can_spawn = (
                        "slicerapp" not in Path(sys.executable).stem.lower()
                        and bool(main_file)
                        and Path(main_file).is_file()
                    )
                    # When the vectorized GIL-releasing batched Cholesky + Chebyshev path is active
                    # (len(grid) >= 64), ThreadPoolExecutor on Windows/macOS avoids spawn startup,
                    # 1+ GiB pickle IPC, and multi-process RAM clamping (~30 MiB/thread vs ~1 GiB/proc).
                    use_fast_threads = (
                        not is_cuda_dev
                        and len(grid) >= 64
                        and sys.platform in ("win32", "darwin")
                    )
                    use_threads = use_fast_threads or not can_spawn or (sys.platform in ("win32", "darwin") and count < 256)
                    if use_fast_threads:
                        from shared.hardware_adapt import host_total_ram_gib
                        tot_ram = host_total_ram_gib()
                        if tot_ram is not None and tot_ram <= 10.0:
                            eff_workers = min(eff_workers, 4)
                    elif not use_threads:
                        from shared.hardware_adapt import adaptive_spawn_workers
                        safe_spawn = adaptive_spawn_workers(
                            eff_workers,
                            per_worker_ram_gib=1.0,
                            reserve_ram_gib=4.0,
                            max_workers=max(eff_workers, 8),
                        )
                        if safe_spawn < eff_workers:
                            if safe_spawn <= 1:
                                use_threads = True
                                eff_workers = min(eff_workers, 4)
                                if log:
                                    log(
                                        f"[MK-Curve Python] Low host RAM; using {eff_workers} threads "
                                        f"instead of {num_workers} spawned processes to avoid memory exhaustion"
                                    )
                            else:
                                if log:
                                    log(
                                        f"[MK-Curve Python] Clamping spawned workers from {eff_workers} "
                                        f"to {safe_spawn} for host RAM safety"
                                    )
                                eff_workers = safe_spawn
                    if use_threads:
                        from concurrent.futures import ThreadPoolExecutor
                        # Main thread already pinned BLAS via threadpool_limits above; avoid
                        # concurrent per-thread threadpool_limits() mutation races.
                        pool = ThreadPoolExecutor(max_workers=eff_workers)
                    else:
                        from concurrent.futures import ProcessPoolExecutor
                        import multiprocessing
                        pool = ProcessPoolExecutor(max_workers=eff_workers,
                            mp_context=multiprocessing.get_context("spawn"),
                            initializer=_initialize_worker, initargs=(worker_threads,))
                    handles.callback(pool.shutdown, wait=True, cancel_futures=True)
                else:
                    pool = None
                def payload(start):
                    stop = min(start + eff_chunk, count)
                    sig = np.asarray(masked_signals[start:stop], dtype=np.float64).copy()
                    if is_cuda_dev:
                        return sig, design, grid, threshold, solver, resolved_device, explicit_cuda
                    return sig, design, grid, threshold, solver
                from microstructure.chunk_dispatch import chunk_results
                completed = 0
                last_progress = 0.0
                for start, stop, fitted in chunk_results(count, eff_chunk, eff_workers,
                        payload, chunk_fn, pool=pool, schedule=chunk_schedule):
                    chunk_coords = tuple(c[start:stop] for c in coords)
                    original, corrected, chunk_curves, detection = fitted
                    original_coefficients[start:stop] = original
                    curves[:, start:stop] = chunk_curves
                    corrected_coefficients[start:stop] = corrected
                    for key, values in detection.items():
                        if key not in results:
                            results[key] = np.lib.format.open_memmap(Path(temporary) / f"{key}.npy", mode="w+", dtype=values.dtype, shape=mask.shape)
                            results[key][:] = 0
                        results[key][chunk_coords] = values
                    completed += stop - start
                    if log:
                        log(f"[MK-Curve Python] {completed}/{count} voxels; {len(grid)} synthetic b0 samples")
                    now = time.monotonic()
                    if start == 0 or completed >= count or (now - last_progress) >= 1.0:
                        _write_json(progress, {"status": "running", "completed_voxels": completed, "total_voxels": count,
                                               "num_workers": eff_workers, "solver": solver, "device": resolved_device})
                        last_progress = now
                paths, summaries = {}, {}
                def _export_map(item):
                    key, values = item
                    path = root / f"mkcurve_{key}.nii.gz"
                    header = reference.header.copy()
                    dtype = np.uint8 if values.dtype == bool else np.float32
                    header.set_data_dtype(dtype)
                    img = nib.Nifti1Image(values.astype(dtype), reference.affine, header)
                    with open(path, "wb", buffering=2 * 1024 * 1024) as raw_fh:
                        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_fh, compresslevel=1, mtime=0) as gz_fh:
                            img.to_file_map({"image": nib.FileHolder(fileobj=gz_fh)})
                    selected = values[coords]
                    return key, str(path), {"finite": int(np.isfinite(selected).sum()), "nonzero": int(np.count_nonzero(selected))}

                export_workers = min(8, max(1, os.cpu_count() or 1), len(results)) if len(results) > 1 and count >= 1024 else 1
                if export_workers > 1:
                    from concurrent.futures import ThreadPoolExecutor
                    with ThreadPoolExecutor(max_workers=export_workers) as export_pool:
                        for key, out_path, summary in export_pool.map(_export_map, results.items()):
                            paths[key] = out_path
                            summaries[key] = summary
                else:
                    for item in results.items():
                        key, out_path, summary = _export_map(item)
                        paths[key] = out_path
                        summaries[key] = summary
                record = {"kind": "mkcurve_python_port", "status": "completed", "backend": backend,
                          "method": "author-style unconstrained WLLS, numerical 256-direction MK, synthetic-b0 correction",
                          "settings": {"threshold": threshold, "solver": solver, "device": resolved_device,
                                       "num_workers": num_workers, "effective_workers": eff_workers,
                                       "voxels_per_chunk": eff_chunk,
                                       "worker_threads": worker_threads,
                                       "chunk_schedule": chunk_schedule, "max_inflight_chunks": eff_workers,
                                       "grid_policy": "adaptive author integer grid, independent of directory name",
                                       "grid_samples": len(grid), "b0_threshold": b0_threshold, "max_bvalue": max_bvalue},
                          "maps": paths, "inputs": inputs, "preparation": preparation, "signal_repair": repair,
                          "mask_voxels": count, "corrected_voxels": summaries["abnormal_mask"]["nonzero"],
                          "corrected_valid_voxels": summaries["corrected_valid_mask"]["nonzero"], "summaries": summaries,
                          "automatic_qc_passed": summaries["corrected_valid_mask"]["nonzero"] == count,
                          "metric_valid_voxels": {key: summaries[f"corrected_{key}_valid_mask"]["nonzero"]
                                                  for key in ("fa", "md", "ad", "rd", "mk", "ak", "rk")
                                                  if f"corrected_{key}_valid_mask" in summaries},
                          "validity_policy": "Per-metric finite/range checks with finite coefficients, finite curve/residual and positive diffusion eigenvalues. Joint validity is retained separately; exclusions are not fitted values.",
                          "human_review": "pending", "raw_metric_clipping": False, "smoothing": False,
                          "runtime_seconds": time.monotonic() - started,
                          "created_at": datetime.now(timezone.utc).isoformat(),
                          "differences_from_upstream": ["explicit adaptive grid; no filename-based HCP/Ndyx branch",
                                                       "stable radial plane at principal-axis poles", "nonfinite curves are invalid, never silently corrected"],
                          "curve_storage": "mk_vs_synthetic_b0.npy: grid x masked voxels in MATLAB/Fortran order",
                          "record_path": str(root / "mkcurve_record.json")}
                if record["corrected_valid_voxels"] == 0:
                    raise ValueError(NO_VALID_VOXELS_STOP)
                for array in (curves, original_coefficients, corrected_coefficients):
                    array.flush()
                record["auxiliary_files"] = {key: str(root / filename) for key, filename in
                                             (("curve", "mk_vs_synthetic_b0.npy"), ("grid", "synthetic_b0_grid.csv"),
                                              ("original_coefficients", "original_coefficients.npy"),
                                              ("corrected_coefficients", "corrected_coefficients.npy"))}
                record["auxiliary_artifacts"] = {key: _signature(Path(path)) for key, path in record["auxiliary_files"].items()}
                if any(inputs[key] != _signature(Path(path)) for key, path in
                       zip(("dwi", "bval", "bvec", "mask"), (dwi_path, bval_path, bvec_path, mask_path))):
                    raise RuntimeError("MK-Curve source inputs changed during fitting")
                if backend != inspect_mkcurve_backend():
                    raise RuntimeError("MK-Curve backend changed during fitting")
                _write_json(Path(record["record_path"]), record)
                _write_json(progress, {"status": "completed", "completed_voxels": count, "total_voxels": count})
                return record
            finally:
                if resolved_device.startswith("cuda"):
                    from microstructure.mkcurve_tensor import _cached_cuda_operators
                    from shared.hardware_adapt import empty_cuda_cache_safely

                    _cached_cuda_operators.cache_clear()
                    empty_cuda_cache_safely()
                for array in (curves, original_coefficients, corrected_coefficients, masked_signals, *results.values()):
                    array._mmap.close()
                results.clear()
                del curves, original_coefficients, corrected_coefficients, masked_signals
                import gc
                gc.collect()
    except Exception as exc:
        _write_json(progress, {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        _write_json(root / "mkcurve_record.json", {"kind": "mkcurve_python_port", "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
