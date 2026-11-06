"""iFW: free-water fraction corrected for T2 weighting, minus the CSF fraction (T1 design D3).

Formula (SI note, Cho et al. preprint Supplementary Methods 1):

    v   = f * C_T / (C_C * (1 - f) + f * C_T)      C_T = exp(-TE / T2_GM), C_C = exp(-TE / T2_CSF)
    iFW = max(v - CSF_v, 0)

``f`` is the fitted free-water signal fraction. ``derive_ifw`` follows the checks and dtypes of the
manuscript's reference function; the stage ``run_ifw`` adds the grid rule, the echo-time source rule
and the Free Water eligibility rule of the design.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np

IFW_T2_GM_MS = 95.0
IFW_T2_CSF_MS = 1500.0
IFW_REFERENCE = (
    "Cho et al., preprint, Supplementary Methods 1, equations S.1-S.4 "
    "(doi:10.21203/rs.3.rs-3179575/v1); fixed T2_GM = 95 ms and T2_CSF = 1500 ms"
)
GRID_AFFINE_ATOL = 1e-5
FREE_WATER_INELIGIBLE = (
    "Free water needs at least three distinct b-values, b=0 counted (for example 0, 1000 and 2000 s/mm²)"
)
ECHO_TIME_MISSING = "iFW needs the echo time (ms): enter it in Microstructure"
ECHO_TIME_SOURCES = ("user", "sidecar")


class IfwBlocked(ValueError):
    """The iFW stage cannot run for a reason the user can fix; other stages continue."""


class FreeWaterIneligible(IfwBlocked):
    """The acquisition has fewer than three distinct b-values."""


class EchoTimeMissing(IfwBlocked):
    """No echo time from the user or the DWI sidecar."""


def derive_ifw(fw, csf, *, echo_time_ms, tissue_t2_ms, csf_t2_ms):
    """Invert the two-compartment T2 weighting, subtract CSF, project negatives to zero.

    Returns ``(corrected, unprojected, ifw)`` in float64. Port of the manuscript's reference function.
    """
    fw, csf = np.asarray(fw, dtype=np.float64), np.asarray(csf, dtype=np.float64)
    if fw.shape != csf.shape:
        raise ValueError("FW and CSF arrays must share the same physical grid")
    for name, array in (("FW", fw), ("CSF", csf)):
        if not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
            raise ValueError(name + " must be finite fractions in [0,1]")
    if (not np.isfinite([echo_time_ms, tissue_t2_ms, csf_t2_ms]).all()
            or echo_time_ms < 0 or min(tissue_t2_ms, csf_t2_ms) <= 0):
        raise ValueError("TE must be nonnegative; both T2 values must be positive, in ms")
    ct, cc = np.exp(-echo_time_ms / tissue_t2_ms), np.exp(-echo_time_ms / csf_t2_ms)
    if min(ct, cc) <= 0:
        raise ValueError("Relaxation attenuation underflows")
    corrected = fw * ct / (cc * (1 - fw) + fw * ct)
    unprojected = corrected - csf
    return corrected, unprojected, np.maximum(unprojected, 0.0)


def free_water_eligibility(bvals, bvecs=None, *, b0_threshold: float = 50.0) -> tuple[bool, str]:
    """DIPY's own predicate for ``FreeWaterTensorModel``: at least three b-value groups, b=0 counted."""
    from dipy.core.gradients import check_multi_b, gradient_table

    bvals = np.asarray(bvals, dtype=float).ravel()
    if bvecs is None:  # only the b-value groups matter here; DIPY needs unit vectors for the b > 0 rows
        bvecs = np.tile([1.0, 0.0, 0.0], (bvals.size, 1))
        bvecs[bvals <= b0_threshold] = 0.0
    else:
        bvecs = np.asarray(bvecs, dtype=float)
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=b0_threshold)
    if check_multi_b(gtab, 3, non_zero=False):
        return True, ""
    return False, FREE_WATER_INELIGIBLE


def sidecar_echo_time_ms(sidecar_path: Optional[Path]) -> Optional[float]:
    """``EchoTime`` of a BIDS-style JSON sidecar, seconds converted to ms; None when absent or unusable."""
    if sidecar_path is None or not Path(sidecar_path).is_file():
        return None
    try:
        value = json.loads(Path(sidecar_path).read_text(encoding="utf-8")).get("EchoTime")
    except (OSError, ValueError, AttributeError):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value) * 1000.0
    return value if np.isfinite(value) and value >= 0 else None


def resolve_echo_time(user_value_ms: Optional[float], sidecar_path: Optional[Path]) -> tuple[float, str]:
    """TE in ms and its source: the user value, then the sidecar ``EchoTime``; no default (raises)."""
    if user_value_ms is not None:
        return float(user_value_ms), "user"
    from_sidecar = sidecar_echo_time_ms(sidecar_path)
    if from_sidecar is not None:
        return from_sidecar, "sidecar"
    raise EchoTimeMissing(ECHO_TIME_MISSING)


@dataclass(frozen=True)
class IfwConfig:
    subject_id: str
    fw_fraction_path: Path
    csf_probability_path: Path
    mask_path: Path
    bval_path: Path
    bvec_path: Path
    output_dir: Path
    echo_time_ms: Optional[float] = None
    dwi_sidecar_path: Optional[Path] = None
    tissue_t2_ms: float = IFW_T2_GM_MS
    csf_t2_ms: float = IFW_T2_CSF_MS
    roi_label_map: Optional[Path] = None
    lut_path: Optional[Path] = None


@dataclass(frozen=True)
class IfwResult:
    subject_id: str
    status: str
    ifw_path: Path
    fw_volume_fraction_path: Path
    record_path: Path
    echo_time_ms: float
    echo_time_source: str
    roi_stats_csv_path: Optional[Path] = None


def ifw_output_paths(subject_id: str, output_dir: Path) -> dict[str, Path]:
    directory = Path(output_dir)
    return {
        "ifw": directory / f"{subject_id}_ifw.nii.gz",
        "fw_volume_fraction": directory / f"{subject_id}_fw_volume_fraction.nii.gz",
        "record": directory / "ifw_record.json",
        "roi_stats_csv": directory / f"{subject_id}_ROIWise-iFW.csv",
        "roi_stats_json": directory / f"{subject_id}_ROIWise-iFW.json",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(path: Path):
    import nibabel as nib

    return nib.as_closest_canonical(nib.load(str(path)))


def _to_orientation_of(array: np.ndarray, canonical_affine: np.ndarray, native_affine: np.ndarray) -> np.ndarray:
    """Bring a canonical-order array back to the voxel order of an image with ``native_affine``."""
    import nibabel as nib
    from nibabel.orientations import apply_orientation, io_orientation, ornt_transform

    transform = ornt_transform(io_orientation(canonical_affine), io_orientation(native_affine))
    return np.asarray(apply_orientation(array, transform))


def _inputs(config: IfwConfig) -> dict[str, Path]:
    paths = {"fw_fraction": config.fw_fraction_path, "csf_probability": config.csf_probability_path,
             "mask": config.mask_path, "bval": config.bval_path, "bvec": config.bvec_path}
    return {key: Path(value) for key, value in paths.items()}


def _signature(config: IfwConfig, echo_time_ms: float, source: str) -> dict:
    inputs = {key: _sha256(path) for key, path in _inputs(config).items()}
    sidecar = config.dwi_sidecar_path
    if source == "sidecar" and sidecar is not None:
        inputs["dwi_sidecar"] = _sha256(Path(sidecar))
    return {"inputs_sha256": inputs, "echo_time_ms": echo_time_ms, "echo_time_source": source,
            "tissue_t2_ms": float(config.tissue_t2_ms), "csf_t2_ms": float(config.csf_t2_ms)}


def ifw_outputs_current(config: IfwConfig, echo_time_ms: float, source: str) -> bool:
    paths = ifw_output_paths(config.subject_id, config.output_dir)
    if not (paths["ifw"].is_file() and paths["fw_volume_fraction"].is_file() and paths["record"].is_file()):
        return False
    try:
        record = json.loads(paths["record"].read_text(encoding="utf-8"))
        expected = _signature(config, echo_time_ms, source)
    except (OSError, ValueError):
        return False
    if record.get("status") != "completed" or any(record.get(key) != value for key, value in expected.items()):
        return False
    return record.get("output_sha256") == {"ifw": _sha256(paths["ifw"]),
                                           "fw_volume_fraction": _sha256(paths["fw_volume_fraction"])}


def run_ifw(config: IfwConfig, log: Optional[Callable[[str], None]] = None) -> IfwResult:
    """Run the iFW stage on the native grid. Raises :class:`IfwBlocked` for fixable stops."""
    import nibabel as nib

    emit = log or (lambda message: None)
    # Eligibility first: the same DIPY predicate as the Free Water stage, on this stage's gradient table.
    bvals = np.loadtxt(config.bval_path, dtype=float).ravel()
    bvecs = np.loadtxt(config.bvec_path, dtype=float)
    bvecs = bvecs.T if bvecs.ndim == 2 and bvecs.shape[0] == 3 and bvecs.shape[1] != 3 else bvecs
    eligible, message = free_water_eligibility(bvals, bvecs)
    if not eligible:
        raise FreeWaterIneligible(message)
    echo_time_ms, source = resolve_echo_time(config.echo_time_ms, config.dwi_sidecar_path)
    paths = ifw_output_paths(config.subject_id, config.output_dir)
    paths["ifw"].parent.mkdir(parents=True, exist_ok=True)
    if ifw_outputs_current(config, echo_time_ms, source):
        emit("[iFW] Reusing matching completed stage")
        return IfwResult(config.subject_id, "cached", paths["ifw"], paths["fw_volume_fraction"], paths["record"],
                         echo_time_ms, source,
                         paths["roi_stats_csv"] if paths["roi_stats_csv"].is_file() else None)
    started = time.monotonic()
    native = nib.load(str(config.fw_fraction_path))
    images = {"fw_fraction": _canonical(config.fw_fraction_path),
              "csf_probability": _canonical(config.csf_probability_path),
              "mask": _canonical(config.mask_path)}
    reference = images["fw_fraction"]
    for name, image in images.items():
        if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, rtol=0,
                                                              atol=GRID_AFFINE_ATOL):
            raise ValueError("Spatial mismatch: " + name)
    fw = reference.get_fdata(dtype=np.float64)
    csf = images["csf_probability"].get_fdata(dtype=np.float64)
    support = images["mask"].get_fdata(dtype=np.float64) > 0
    corrected, unprojected, ifw = derive_ifw(fw, csf, echo_time_ms=echo_time_ms,
                                             tissue_t2_ms=float(config.tissue_t2_ms),
                                             csf_t2_ms=float(config.csf_t2_ms))
    ifw[~support] = 0
    projected = int(np.sum((unprojected < 0) & support))
    outputs = {}
    for key, array in (("ifw", ifw), ("fw_volume_fraction", corrected)):
        data = _to_orientation_of(array.astype(np.float32), reference.affine, native.affine)
        image = nib.Nifti1Image(data, native.affine)
        image.set_data_dtype(np.float32)
        nib.save(image, str(paths[key]))
        outputs[key] = _sha256(paths[key])
    roi_csv = None
    if config.roi_label_map is not None:
        from microstructure.pipeline import _compute_roi_stats

        _compute_roi_stats(metric_paths={"ifw": paths["ifw"], "fw_volume_fraction": paths["fw_volume_fraction"]},
                           roi_label_map=Path(config.roi_label_map), lut_path=config.lut_path,
                           csv_path=paths["roi_stats_csv"], json_path=paths["roi_stats_json"],
                           brain_mask_path=Path(config.mask_path))
        roi_csv = paths["roi_stats_csv"]
    record = {
        "kind": "microstructure_ifw_stage", "version": 1, "status": "completed", "subject_id": config.subject_id,
        **_signature(config, echo_time_ms, source),
        "formula": "v = f*C_T/(C_C*(1-f)+f*C_T); iFW = max(v - CSF_v, 0); C_T = exp(-TE/T2_GM), C_C = exp(-TE/T2_CSF)",
        "reference": IFW_REFERENCE,
        "grid": "canonical orientation, affine atol 1e-5; maps saved in the voxel order of the Free Water map",
        "dtype": "float64 in memory, float32 saved", "outside_brain_mask": "iFW = 0",
        "negative_voxels_projected_to_zero": projected, "mask_voxels": int(support.sum()),
        "outputs": {key: paths[key].as_posix() for key in ("ifw", "fw_volume_fraction")},
        "output_sha256": outputs,
        "runtime_seconds": time.monotonic() - started,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    paths["record"].write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    emit(f"[iFW] TE {echo_time_ms:g} ms ({source}); wrote {paths['ifw']}")
    return IfwResult(config.subject_id, "completed", paths["ifw"], paths["fw_volume_fraction"], paths["record"],
                     echo_time_ms, source, roi_csv)
