"""Shared Python input staging for native and optional MATLAB MK-Curve."""

import gc
import shutil
import gzip

def _prepare_inputs(dwi_path, bval_path, bvec_path, mask_path, directory, b0_threshold, max_bvalue=3000.0, *, allow_nonfinite=False):
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
    uncompressed_image = None
    if str(dwi_path).endswith(".gz"):
        uncompressed = directory / "source.nii"
        with gzip.open(dwi_path, "rb") as source, uncompressed.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
        uncompressed_image = nib.load(str(uncompressed), mmap="r", keep_file_open=True)
        source_image = uncompressed_image
    else:
        source_image = image
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
        if uncompressed_image is not None:
            for holder in uncompressed_image.file_map.values():
                if getattr(holder, "fileobj", None) is not None:
                    holder.fileobj.close()
            del uncompressed_image
        del source_image
        gc.collect()
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
