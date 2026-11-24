"""Shared Python input staging for native and optional MATLAB MK-Curve."""

import gc
import gzip
from pathlib import Path
import shutil


def _repair_nonpositive_volume(image, mask):
    """Upstream sequential 26-neighbour positive-signal mean on one 3D volume (MATLAB Fortran order)."""
    import numpy as np

    repaired, unresolved = 0, 0
    bad = mask & (~np.isfinite(image) | (image <= 0))
    if not np.any(bad):
        return 0, 0
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
    return repaired, unresolved


def _open_uncompressed_dwi(dwi_path, directory):
    """Return ``(source_image, close_image, temp_path)`` using shared uncompressed cache or fast gzip."""
    import nibabel as nib

    dwi_p = Path(dwi_path)
    if not str(dwi_p).lower().endswith(".gz"):
        return nib.load(str(dwi_p), mmap=False), None, None

    try:
        from shared.diffusion_io import ensure_cached_uncompressed_nifti, open_fast_gzip_reader
    except Exception:
        ensure_cached_uncompressed_nifti = None
        open_fast_gzip_reader = None

    cached = ensure_cached_uncompressed_nifti(dwi_p) if ensure_cached_uncompressed_nifti is not None else None
    if cached is not None:
        img = nib.load(str(cached), mmap="r", keep_file_open=True)
        return img, img, None

    uncompressed = Path(directory) / "source.nii"
    if open_fast_gzip_reader is not None:
        with open_fast_gzip_reader(dwi_p) as source, uncompressed.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
    else:
        with gzip.open(dwi_p, "rb") as source, uncompressed.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
    img = nib.load(str(uncompressed), mmap="r", keep_file_open=True)
    return img, img, uncompressed


def _close_nib_handles(image):
    if image is None:
        return
    for holder in image.file_map.values():
        if getattr(holder, "fileobj", None) is not None:
            try:
                holder.fileobj.close()
            except Exception:
                pass


def _validate_and_select_inputs(dwi_path, bval_path, bvec_path, mask_path, b0_threshold, max_bvalue):
    import nibabel as nib
    import numpy as np
    from microstructure.dki import _shells

    image = nib.load(str(dwi_path), mmap=False)
    mask_image = nib.load(str(mask_path), mmap=False)
    bvals = np.asarray(np.loadtxt(bval_path), dtype=float).reshape(-1)
    bvecs = np.asarray(np.loadtxt(bvec_path), dtype=float)
    if bvecs.shape == (3, len(bvals)):
        bvecs = bvecs.T
    if len(image.shape) != 4 or image.shape[-1] != len(bvals) or bvecs.shape != (len(bvals), 3):
        raise ValueError("MK-Curve DWI and gradients have incompatible dimensions")
    if mask_image.shape != image.shape[:3] or not np.allclose(mask_image.affine, image.affine, atol=1e-4, rtol=0):
        raise ValueError("MK-Curve mask and DWI must share the same voxel/world geometry")
    mask = np.asarray(mask_image.dataobj)
    if not np.isfinite(mask).all() or not np.isin(mask, (0, 1)).all() or not np.any(mask):
        raise ValueError("MK-Curve requires a finite, nonempty binary brain mask")
    if not np.isfinite(bvals).all() or not np.isfinite(bvecs).all() or np.any(bvals < 0):
        raise ValueError("MK-Curve gradients must be finite and b-values nonnegative")
    baseline = np.flatnonzero(bvals <= b0_threshold)
    weighted = np.asarray(sorted(index for shell in _shells(bvals, b0_threshold, 100.0)
                                 if np.median(bvals[shell]) <= max_bvalue for index in shell), dtype=int)
    if not len(baseline):
        raise ValueError("MK-Curve requires baseline measurements")
    return image, mask, bvals, bvecs, baseline, weighted


def _prepare_masked_signals(dwi_path, bval_path, bvec_path, mask_path, directory, b0_threshold, max_bvalue=3000.0, *, allow_nonfinite=True):
    """Single-pass streaming staging, 26-neighbour repair, and mask extraction (bit-exact to _prepare_inputs + repair_nonpositive).

    Avoids materialising two full-volume float64 copies (~10.5 GiB on HCP 100307) on disk by repairing
    each 3D volume in RAM (~29 MiB) and writing only masked voxels into ``masked_signals.npy``.
    """
    import nibabel as nib
    import numpy as np

    _, mask_raw, bvals, bvecs, baseline, weighted = _validate_and_select_inputs(
        dwi_path, bval_path, bvec_path, mask_path, b0_threshold, max_bvalue
    )
    mask = np.asarray(mask_raw > 0, dtype=bool)
    coords = np.unravel_index(np.flatnonzero(mask.ravel(order="F")), mask.shape, order="F")
    count = len(coords[0])
    n_vols = len(weighted) + 1

    source_image, close_img, temp_uncomp = _open_uncompressed_dwi(dwi_path, directory)
    masked_path = Path(directory) / "masked_signals.npy"
    masked_signals = np.lib.format.open_memmap(masked_path, mode="w+", dtype=np.float64, shape=(count, n_vols))
    repaired_total, unresolved_total = 0, 0
    try:
        total_b0 = np.zeros(source_image.shape[:3], dtype=np.float64)
        for index in baseline:
            volume = np.asarray(source_image.dataobj[..., int(index)], dtype=np.float64)
            if not allow_nonfinite and not np.isfinite(volume[mask]).all():
                raise ValueError("MK-Curve baseline contains NaN/Inf in the brain mask")
            total_b0 += volume
        vol0 = total_b0 / len(baseline)
        rep, unres = _repair_nonpositive_volume(vol0, mask)
        repaired_total += rep
        unresolved_total += unres
        masked_signals[:, 0] = vol0[coords]
        del total_b0, vol0

        for destination, source_idx in enumerate(weighted, start=1):
            volume = np.asarray(source_image.dataobj[..., int(source_idx)], dtype=np.float64).copy()
            if not allow_nonfinite and not np.isfinite(volume[mask]).all():
                raise ValueError("MK-Curve DWI contains NaN/Inf in the brain mask")
            rep, unres = _repair_nonpositive_volume(volume, mask)
            repaired_total += rep
            unresolved_total += unres
            masked_signals[:, destination] = volume[coords]

        header = source_image.header.copy()
        header.set_data_dtype(np.float64)
        affine = source_image.affine.copy()
        reference = nib.Nifti1Image(np.empty((0,), dtype=np.float32), affine, header)
    except Exception:
        masked_signals._mmap.close()
        del masked_signals
        if masked_path.is_file():
            masked_path.unlink()
        raise
    finally:
        _close_nib_handles(close_img)
        del close_img, source_image
        gc.collect()
        if temp_uncomp is not None and temp_uncomp.is_file():
            try:
                temp_uncomp.unlink()
            except OSError:
                pass

    bvals_sel = np.concatenate(([0.0], bvals[weighted]))
    bvecs_sel = np.vstack((np.zeros((1, 3)), bvecs[weighted]))
    preparation = {
        "b0_threshold_s_mm2": b0_threshold, "baseline_count": len(baseline),
        "max_bvalue_s_mm2": max_bvalue, "excluded_high_b_volumes": len(bvals) - len(baseline) - len(weighted),
        "upper_bvalue_policy": "complete shells selected by median, tolerance 100 s/mm2",
        "working_baseline": "mean of selected b0 measurements; b=0; zero b-vector",
        "nonpositive_signal_handling": "upstream check_non_positive_signal uses neighbor means",
        "original_inputs_modified": False,
        "working_storage": "disk-backed float64 masked signals; compressed source expanded once; no float32 input rounding",
    }
    repair = {
        "repaired_measurements": repaired_total, "unresolved_measurements": unresolved_total,
        "policy": "sequential 26-neighbour positive-signal mean; working copy only; no parameter-map smoothing",
    }
    return masked_signals, mask, coords, bvals_sel, bvecs_sel, reference, preparation, repair


def _prepare_inputs(dwi_path, bval_path, bvec_path, mask_path, directory, b0_threshold, max_bvalue=3000.0, *, allow_nonfinite=False):
    import nibabel as nib
    import numpy as np

    _, mask, bvals, bvecs, baseline, weighted = _validate_and_select_inputs(
        dwi_path, bval_path, bvec_path, mask_path, b0_threshold, max_bvalue
    )
    source_image, uncompressed_image, temp_uncomp = _open_uncompressed_dwi(dwi_path, directory)
    # Upstream detects b0 strictly below 10. Make the caller's documented b0
    # convention explicit and collapse all baseline repeats, without editing input.
    selected_signals_path = directory / "selected_signals.npy"
    data = np.lib.format.open_memmap(selected_signals_path, mode="w+", dtype=np.float64,
                                   fortran_order=True, shape=(*source_image.shape[:3], len(weighted) + 1))
    try:
        total = np.zeros(source_image.shape[:3], dtype=np.float64)
        for index in baseline:
            volume = np.asarray(source_image.dataobj[..., int(index)], dtype=np.float64)
            if not allow_nonfinite and not np.isfinite(volume[mask > 0]).all():
                raise ValueError("MK-Curve baseline contains NaN/Inf in the brain mask")
            total += volume
        data[..., 0] = total / len(baseline)
        for destination, source in enumerate(weighted, start=1):
            volume = np.asarray(source_image.dataobj[..., int(source)], dtype=np.float64)
            if not allow_nonfinite and not np.isfinite(volume[mask > 0]).all():
                raise ValueError("MK-Curve DWI contains NaN/Inf in the brain mask")
            data[..., destination] = volume
        header = source_image.header.copy()
        header.set_data_dtype(np.float64)
        affine = source_image.affine.copy()
        dwi = directory / "data.nii"
        bval = directory / "bvals"
        bvec = directory / "bvecs"
        mask_file = directory / "mask.nii.gz"
        nib.save(nib.Nifti1Image(data, affine, header), str(dwi))
    finally:
        data._mmap.close()
        del data
        if selected_signals_path.is_file():
            selected_signals_path.unlink()
        _close_nib_handles(uncompressed_image)
        del uncompressed_image
        del source_image
        gc.collect()
        if temp_uncomp is not None and temp_uncomp.is_file():
            try:
                temp_uncomp.unlink()
            except OSError:
                pass
    reference = nib.Nifti1Image(np.empty((0,), dtype=np.float32), affine, header)
    np.savetxt(bval, np.concatenate(([0.0], bvals[weighted]))[None, :], fmt="%.10g")
    np.savetxt(bvec, np.vstack((np.zeros((1, 3)), bvecs[weighted])).T, fmt="%.10g")
    shutil.copy2(mask_path, mask_file)
    return (dwi, bval, bvec, mask_file), reference, {
        "b0_threshold_s_mm2": b0_threshold, "baseline_count": len(baseline),
        "max_bvalue_s_mm2": max_bvalue, "excluded_high_b_volumes": len(bvals) - len(baseline) - len(weighted),
        "upper_bvalue_policy": "complete shells selected by median, tolerance 100 s/mm2",
        "working_baseline": "mean of selected b0 measurements; b=0; zero b-vector",
        "nonpositive_signal_handling": "upstream check_non_positive_signal uses neighbor means",
        "original_inputs_modified": False,
        "working_storage": "disk-backed float64 volumes; compressed source expanded once; no float32 input rounding",
    }
