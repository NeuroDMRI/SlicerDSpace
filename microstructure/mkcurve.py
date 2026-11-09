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
PORT_VERSION = 2
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
        reuse_signal_logs = solver == "reference"
    b0 = signals[:, 0].copy()
    fitter = None
    if reuse_signal_logs:
        from microstructure.mkcurve_tensor import _SyntheticB0Fitter
        fitter = _SyntheticB0Fitter(signals, design, solver=solver)
    original = fitter.fit(b0) if fitter else fit_wlls(signals, design, solver=solver)
    curves = np.empty((len(grid), len(signals)))
    for step, value in enumerate(grid):
        signals[:, 0] = value
        coefficients = fitter.fit(signals[:, 0]) if fitter else fit_wlls(signals, design, solver=solver)
        curves[step] = mean_kurtosis(coefficients)
    detection = detect_abnormal(curves, b0, grid, threshold)
    signals[:, 0] = detection["corrected_b0"]
    corrected = fitter.fit(signals[:, 0]) if fitter else fit_wlls(signals, design, solver=solver)
    for kind, coefficients in (("original", original), ("corrected", corrected)):
        maps = parameter_maps(coefficients)
        valid = np.isfinite(coefficients).all(axis=1)
        for key, values in maps.items():
            lower, upper = (0, 1) if key == "fa" else ((0, 3) if key in {"mk", "ak", "rk"} else (0, .005))
            valid &= np.isfinite(values) & (values >= lower) & (values <= upper)
            detection[f"{kind}_{key}"] = values
        observed = signals.copy()
        if kind == "original":
            observed[:, 0] = b0
        with np.errstate(over="ignore", invalid="ignore"):
            predicted = np.exp(coefficients @ design.T)
            detection[f"{kind}_nrmse"] = np.sqrt(np.mean((predicted - observed) ** 2, axis=1)) / np.exp(coefficients[:, 0])
        valid &= detection["curve_valid_mask"] & np.isfinite(detection[f"{kind}_nrmse"])
        detection[f"{kind}_valid_mask"] = valid
        for key, mask in metric_validity(maps, np.isfinite(coefficients).all(axis=1),
                                        detection["curve_valid_mask"], detection[f"{kind}_nrmse"]).items():
            detection[f"{kind}_{key}_valid_mask"] = mask
    return original, corrected, curves, detection


def _initialize_worker(threads):
    if threads is None:
        return
    import os
    from threadpoolctl import threadpool_limits
    global _worker_thread_limit
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = str(threads)
    _worker_thread_limit = threadpool_limits(limits=threads)


def run_mkcurve(dwi_path, bval_path, bvec_path, mask_path, output_dir, *,
                b0_threshold=50., max_bvalue=3000., voxels_per_chunk=64,
                threshold=.5, grid=None, log=None, num_workers=1, solver="reference",
                worker_threads=None, chunk_schedule='dynamic', **legacy):
    """Scan synthetic b0, identify abnormal curves, refit b0-corrected signals.

    Original inputs remain immutable. Raw untruncated maps, validity masks,
    curve samples, coefficients and provenance are retained for review.
    """
    import nibabel as nib
    import numpy as np
    from microstructure.mkcurve_tensor import design_matrix
    from microstructure.dki import _signature, _write_json
    from microstructure.mkcurve_inputs import _prepare_inputs

    if (type(voxels_per_chunk) is not int or voxels_per_chunk < 1
            or type(num_workers) is not int or num_workers < 1
            or not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("MK-Curve requires a positive chunk size and finite threshold in [0,1]")
    if worker_threads is not None and (type(worker_threads) is not int or worker_threads < 1):
        raise ValueError('MK-Curve worker_threads must be a positive integer')
    if num_workers > 1 and worker_threads is None:
        worker_threads = 1
    if chunk_schedule not in ('dynamic', 'ordered'):
        raise ValueError('MK-Curve chunk schedule must be dynamic or ordered')
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
    _write_json(progress, {"status": "running", "implementation": "python_port"})
    _write_json(root / "mkcurve_record.json", {"kind": "mkcurve_python_port", "status": "running", "inputs": inputs})
    try:
        with tempfile.TemporaryDirectory(prefix=".mkcurve-", dir=root) as temporary, ExitStack() as handles:
            prepared, reference, preparation = _prepare_inputs(
                Path(dwi_path), Path(bval_path), Path(bvec_path), Path(mask_path), Path(temporary), b0_threshold, max_bvalue,
                allow_nonfinite=True)
            working_image = nib.load(str(prepared[0]), mmap="c")
            data = working_image.dataobj.get_unscaled()
            if isinstance(data, np.memmap):
                handles.callback(data._mmap.close)
            mask = nib.load(str(prepared[3])).get_fdata() > 0
            repair = repair_nonpositive(data, mask)
            coords = np.unravel_index(np.flatnonzero(mask.ravel(order="F")), mask.shape, order="F")
            b0 = data[..., 0][coords]
            grid = synthetic_b0_grid(b0) if grid is None else np.asarray(grid, dtype=float)
            if len(grid) < 3 or not np.isfinite(grid).all() or np.any(grid <= 0) or not np.all(np.diff(grid) > 0):
                raise ValueError("Synthetic b0 grid must contain at least three finite positive increasing values")
            np.savetxt(root / "synthetic_b0_grid.csv", grid, delimiter=",")
            design = design_matrix(np.loadtxt(prepared[1]), np.loadtxt(prepared[2]).T)
            count = len(b0)
            curves = np.lib.format.open_memmap(root / "mk_vs_synthetic_b0.npy", mode="w+", dtype="float64", shape=(len(grid), count))
            original_coefficients = np.lib.format.open_memmap(root / "original_coefficients.npy", mode="w+", dtype="float64", shape=(count, 22))
            corrected_coefficients = np.lib.format.open_memmap(root / "corrected_coefficients.npy", mode="w+", dtype="float64", shape=(count, 22))
            results = {}
            try:
                if worker_threads is not None:
                    from threadpoolctl import threadpool_limits
                    handles.enter_context(threadpool_limits(limits=worker_threads))
                if num_workers > 1:
                    import sys
                    main_file = getattr(sys.modules.get("__main__"), "__file__", None)
                    can_spawn = (
                        "slicerapp" not in Path(sys.executable).stem.lower()
                        and bool(main_file)
                        and Path(main_file).is_file()
                    )
                    if os.name == "nt" and not (can_spawn and count >= 256):
                        from concurrent.futures import ThreadPoolExecutor
                        pool = ThreadPoolExecutor(max_workers=num_workers,
                            initializer=_initialize_worker, initargs=(worker_threads,))
                    else:
                        from concurrent.futures import ProcessPoolExecutor
                        import multiprocessing
                        pool = ProcessPoolExecutor(max_workers=num_workers,
                            mp_context=multiprocessing.get_context("spawn"),
                            initializer=_initialize_worker, initargs=(worker_threads,))
                    handles.callback(pool.shutdown, wait=True, cancel_futures=True)
                else:
                    pool = None
                def payload(start):
                    stop = min(start + voxels_per_chunk, count)
                    chunk_coords = tuple(c[start:stop] for c in coords)
                    return data[chunk_coords].copy(), design, grid, threshold, solver
                from microstructure.chunk_dispatch import chunk_results
                completed = 0
                for start, stop, fitted in chunk_results(count, voxels_per_chunk, num_workers,
                        payload, _fit_chunk, pool=pool, schedule=chunk_schedule):
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
                    _write_json(progress, {"status": "running", "completed_voxels": completed, "total_voxels": count,
                                           "num_workers": num_workers, "solver": solver})
                paths, summaries = {}, {}
                for key, values in results.items():
                    path = root / f"mkcurve_{key}.nii.gz"
                    header = reference.header.copy()
                    dtype = np.uint8 if values.dtype == bool else np.float32
                    header.set_data_dtype(dtype)
                    nib.save(nib.Nifti1Image(values.astype(dtype), reference.affine, header), str(path))
                    paths[key] = str(path)
                    selected = values[coords]
                    summaries[key] = {"finite": int(np.isfinite(selected).sum()), "nonzero": int(np.count_nonzero(selected))}
                record = {"kind": "mkcurve_python_port", "status": "completed", "backend": backend,
                          "method": "author-style unconstrained WLLS, numerical 256-direction MK, synthetic-b0 correction",
                          "settings": {"threshold": threshold, "solver": solver, "num_workers": num_workers,
                                       "voxels_per_chunk": voxels_per_chunk, "worker_threads": worker_threads,
                                       "chunk_schedule": chunk_schedule, "max_inflight_chunks": num_workers,
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
                for array in (curves, original_coefficients, corrected_coefficients, *results.values()):
                    array._mmap.close()
                if isinstance(data, np.memmap):
                    data._mmap.close()
                for holder in working_image.file_map.values():
                    if getattr(holder, "fileobj", None) is not None:
                        holder.fileobj.close()
                results.clear()
                del curves, original_coefficients, corrected_coefficients, data, working_image
                import gc
                gc.collect()
    except Exception as exc:
        _write_json(progress, {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        _write_json(root / "mkcurve_record.json", {"kind": "mkcurve_python_port", "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
