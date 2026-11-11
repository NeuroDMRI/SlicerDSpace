"""Microstructure workflow: default DTI and optional FW, NODDI, and guarded DKI."""

from __future__ import annotations

from shared.cohort_schedule import scheduled_stage

import csv
import hashlib
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from shared.pipeline import NativeDTIBundle, RawInputBundle
from shared.scalar_qc import scalar_qc_current

LOGGER = logging.getLogger(__name__)
FREE_WATER_ENGINES = {"serial", "joblib", "dask", "ray"}
# DIPY's free-water model assumes an isotropic free-water diffusivity of
# 3e-3 mm^2/s. Tissue MD estimates above that value are numerically invalid
# (typically caused by ill-conditioned low-signal voxel fits) and must not be
# propagated into regional statistics.
FREE_WATER_MAX_TISSUE_MD = 3.0e-3
# DIPY's reference free-water example excludes tissue metrics when the fitted
# free-water fraction exceeds 0.7 because tissue FA becomes unstable there.
FREE_WATER_MAX_FRACTION_FOR_TISSUE = 0.7
# NODDI's NDI and ODI are conditional tissue parameters. In voxels dominated
# by the fitted isotropic compartment they are not interpretable and AMICO can
# legitimately return a dictionary-boundary value. Public maps retain bounded
# brain estimates for continuous display; this threshold defines the separate
# validity mask used by quantitative summaries.
NODDI_MAX_ISOTROPIC_FRACTION_FOR_TISSUE = 0.7
NODDI_UPPER_BOUND = 0.99
ROI_STATISTICS_SCHEMA = 2
DTI_KEYS = ("fa", "md", "ad", "rd", "l1", "l2", "l3", "trace")
FREE_WATER_KEYS = ("fwf", "fw_tissue_fa", "fw_tissue_md", "fw_tissue_valid_mask")
NODDI_KEYS = ("noddi_ndi", "noddi_odi", "noddi_fwf", "noddi_tissue_valid_mask", "noddi_rmse", "noddi_nrmse")

try:
    import nibabel as nib
    import numpy as np
except ImportError as exc:  # pragma: no cover - handled at runtime inside Slicer
    nib = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None

import sys
if sys.platform == "darwin":
    try:
        import torch as _darwin_preload_torch  # noqa: F401
    except Exception:
        pass

try:
    from dipy.core.gradients import gradient_table
    from dipy.reconst.fwdti import FreeWaterTensorModel
except ImportError as exc:  # pragma: no cover - optional runtime dependency
    gradient_table = None  # type: ignore[assignment]
    FreeWaterTensorModel = None  # type: ignore[assignment]
    DIPY_IMPORT_ERROR = exc
else:
    DIPY_IMPORT_ERROR = None

try:
    import amico
    from amico.util import fsl2scheme
except ImportError as exc:  # pragma: no cover - optional runtime dependency
    amico = None  # type: ignore[assignment]
    fsl2scheme = None  # type: ignore[assignment]
    AMICO_IMPORT_ERROR = exc
else:
    AMICO_IMPORT_ERROR = None


@dataclass(frozen=True)
class MicrostructureConfig:
    subject_id: str
    raw_input_bundle: RawInputBundle
    native_bundle: NativeDTIBundle
    output_dir: Path
    roi_label_map: Path | None = None
    lut_path: Path | None = None
    free_water_fit_method: str = "WLS"
    free_water_chunk_size: int = 2
    free_water_engine: str = "serial"
    free_water_n_jobs: int | None = None
    free_water_vox_per_chunk: int | None = None
    free_water_max_fraction_for_tissue: float = FREE_WATER_MAX_FRACTION_FOR_TISSUE
    run_free_water: bool = False
    run_noddi: bool = False
    run_dki: bool = False
    dki_fit_method: str = "MKCURVE"
    dki_matlab_executable: Path | None = None
    dki_mkcurve_root: Path | None = None
    dki_execution: dict | None = None
    force_recompute: bool = False
    noddi_directional_average: bool = False
    noddi_b0_threshold: int = 50
    noddi_lmax: int = 12
    noddi_ndirs: int = 500
    noddi_nthreads: int | None = None
    noddi_solver_lambda1: float = 0.5
    noddi_solver_lambda2: float = 0.001
    noddi_max_isotropic_fraction_for_tissue: float = NODDI_MAX_ISOTROPIC_FRACTION_FOR_TISSUE
    max_parallel_models: int = 1
    free_water_device: str = "cpu"


@dataclass(frozen=True)
class MicrostructureResult:
    subject_id: str
    output_dir: Path
    fa_path: Path
    md_path: Path
    ad_path: Path
    rd_path: Path
    l1_path: Path
    l2_path: Path
    l3_path: Path
    trace_path: Path
    fwf_path: Path
    fw_tissue_fa_path: Path
    fw_tissue_md_path: Path
    fw_tissue_valid_mask_path: Path
    free_water_qc_path: Path
    noddi_ndi_path: Path
    noddi_odi_path: Path
    noddi_fwf_path: Path
    noddi_tissue_valid_mask_path: Path
    noddi_rmse_path: Path
    noddi_nrmse_path: Path
    noddi_qc_path: Path
    roi_label_map: Path | None
    roi_stats_csv_path: Path | None
    roi_stats_json_path: Path | None
    dki_result: object | None = None
    compute_schedule: dict | None = None
    dec_path: Path | None = None


def _emit(message: str, log: Optional[Callable[[str], None]] = None) -> None:
    LOGGER.info(message)
    if log:
        log(message)


def _ensure_base_imports_ready() -> None:
    if IMPORT_ERROR is not None:
        raise RuntimeError(
            "Microstructure output generation requires nibabel and numpy."
        ) from IMPORT_ERROR


def _ensure_dipy_ready() -> None:
    if DIPY_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Free Water modeling requires DIPY. Install it in the active Python environment first."
        ) from DIPY_IMPORT_ERROR


def _ensure_amico_ready() -> None:
    if AMICO_IMPORT_ERROR is not None:
        raise RuntimeError(
            "NODDI modeling requires AMICO (dmri-amico). Install it in the active Python environment first."
        ) from AMICO_IMPORT_ERROR


def _file_signature(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _read_json(path: Path) -> Optional[dict[str, object]]:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:  # pragma: no cover - corrupt manifest fallback
        LOGGER.warning("Unable to parse JSON file: %s", path)
        return None


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_progress(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched = {
        **payload,
        "updated_at": _utc_now(),
    }
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(enriched, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _copy_if_needed(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dest.resolve():
        return
    shutil.copy2(src, dest)


def _link_or_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        if dest.exists() and dest.samefile(src):
            return
        if dest.is_symlink():
            dest.unlink()
        else:
            previous = _read_json(dest.with_name(dest.name + ".source.json"))
            if (previous == _file_signature(src) and dest.stat().st_size == src.stat().st_size
                    and dest.stat().st_mtime_ns == src.stat().st_mtime_ns):
                return
            dest.unlink()
    try:
        dest.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dest)
        _write_json(dest.with_name(dest.name + ".source.json"), _file_signature(src))


def _default_lut_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "anatomy"
        / "parcellation"
        / "conversion"
        / "data"
        / "FreeSurferColorLUT.txt"
    )


def _resolve_roi_label_map(config: MicrostructureConfig) -> Path | None:
    if config.roi_label_map is not None:
        roi_path = config.roi_label_map.expanduser().resolve()
        if not roi_path.is_file():
            raise FileNotFoundError(f"ROI label map not found: {roi_path}")
        return roi_path

    from shared.io_layout import subject_layout
    layout = subject_layout(config.native_bundle.subject_root)
    directory = layout.anatomy_input_dir("ddparcel")
    default_candidates = [
        directory / f"{config.subject_id}_ddparcel_wmparc_labels.nii.gz",
        directory / f"{config.subject_id}-DDSurfer-wmparc.nii.gz",
    ]
    if not directory.exists() and not directory.is_symlink():
        default_candidates.append(layout.outputs / "ddparcel" / f"{config.subject_id}-DDSurfer-wmparc.nii.gz")
    for default_path in default_candidates:
        if default_path.is_file():
            return default_path
    return None


def _resolve_lut_path(config: MicrostructureConfig) -> Path | None:
    lut_path = config.lut_path.expanduser().resolve() if config.lut_path is not None else _default_lut_path()
    if lut_path.is_file():
        return lut_path
    return None


def _parse_label_names(labels: list[int], lut_path: Path | None) -> dict[int, str]:
    if lut_path is None:
        return {}

    label_set = set(labels)
    names: dict[int, str] = {}
    with lut_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            try:
                label_id = int(parts[0])
            except ValueError:
                continue
            if label_id in label_set:
                names[label_id] = parts[1]
    return names


def _output_paths(
    subject_id: str,
    output_dir: Path,
) -> dict[str, Path]:
    dti_dir = output_dir / "dti"
    freewater_dir = output_dir / "freewater"
    noddi_dir = output_dir / "noddi"
    if output_dir.name == "02_microstructure":
        roi_dir = output_dir.parent / "05_statistics" / "regional_microstructure"
    else:
        roi_dir = output_dir / "regional_microstructure"
    return {
        "fa": dti_dir / f"{subject_id}_dti_FA.nii.gz",
        "md": dti_dir / f"{subject_id}_dti_MD.nii.gz",
        "ad": dti_dir / f"{subject_id}_dti_AD.nii.gz",
        "rd": dti_dir / f"{subject_id}_dti_RD.nii.gz",
        "l1": dti_dir / f"{subject_id}_dti_L1.nii.gz",
        "l2": dti_dir / f"{subject_id}_dti_L2.nii.gz",
        "l3": dti_dir / f"{subject_id}_dti_L3.nii.gz",
        "trace": dti_dir / f"{subject_id}_dti_Trace.nii.gz",
        "dec": dti_dir / f"{subject_id}-dti-DEC.nrrd",  # D6: direction-encoded colour FA map
        "dti_qc": dti_dir / "dti_qc.json",
        "fwf": freewater_dir / f"{subject_id}_fw_fraction.nii.gz",
        "fw_tissue_fa": freewater_dir / f"{subject_id}_fw_FA.nii.gz",
        "fw_tissue_md": freewater_dir / f"{subject_id}_fw_MD.nii.gz",
        "fw_tissue_valid_mask": freewater_dir / f"{subject_id}_fw_tissue_valid_mask.nii.gz",
        "free_water_qc": freewater_dir / "freewater_qc.json",
        "noddi_ndi": noddi_dir / f"{subject_id}_noddi_NDI.nii.gz",
        "noddi_odi": noddi_dir / f"{subject_id}_noddi_ODI.nii.gz",
        "noddi_fwf": noddi_dir / f"{subject_id}_noddi_IsoVF.nii.gz",
        "noddi_tissue_valid_mask": noddi_dir / f"{subject_id}_noddi_tissue_valid_mask.nii.gz",
        "noddi_rmse": noddi_dir / f"{subject_id}_noddi_RMSE.nii.gz",
        "noddi_nrmse": noddi_dir / f"{subject_id}_noddi_NRMSE.nii.gz",
        "noddi_qc": noddi_dir / "noddi_qc.json",
        "roi_stats_csv": roi_dir / f"{subject_id}_ROIWise-Microstructure.csv",
        "roi_stats_json": roi_dir / f"{subject_id}_ROIWise-Microstructure.json",
        "free_water_manifest": freewater_dir / "freewater_manifest.json",
        "free_water_progress": freewater_dir / "freewater_progress.json",
        "noddi_manifest": noddi_dir / "noddi_manifest.json",
        "noddi_progress": noddi_dir / "noddi_progress.json",
        "manifest": output_dir / "manifest.json",
    }


def _expected_manifest(
    config: MicrostructureConfig,
    roi_label_map: Path | None,
    *,
    free_water_available: bool | None = None,
    noddi_available: bool | None = None,
) -> dict[str, object]:
    free_water_available = bool(config.run_free_water) if free_water_available is None else bool(free_water_available)
    noddi_available = bool(config.run_noddi) if noddi_available is None else bool(noddi_available)
    roi_metrics = ["FA", "MD", "AD", "RD", "L1", "L2", "L3", "TRACE"]
    if free_water_available:
        roi_metrics.extend(["FWF", "FW_TISSUE_FA", "FW_TISSUE_MD"])
    if noddi_available:
        roi_metrics.extend(["NODDI_NDI", "NODDI_ODI", "NODDI_FWF"])
    return {
        "kind": "microstructure_phase3",
        "version": 6,
        "roi_statistics_schema": ROI_STATISTICS_SCHEMA,
        "subject_id": config.subject_id,
        "fit_method": config.native_bundle.fit_method,
        "roi_metrics": roi_metrics,
        "native_bundle": {
            "dir": str(config.native_bundle.bundle_dir.resolve()),
            "manifest": _read_json(config.native_bundle.manifest_path),
        },
        "raw_input_bundle": {
            "dwi": _file_signature(config.raw_input_bundle.dwi_path),
            "bval": _file_signature(config.raw_input_bundle.bval_path),
            "bvec": _file_signature(config.raw_input_bundle.bvec_path),
            "mask": _file_signature(config.raw_input_bundle.mask_path),
        },
        "free_water_model": {
            "fit_method": config.free_water_fit_method.upper(),
            "chunk_size": int(config.free_water_chunk_size),
            "engine": config.free_water_engine,
            "n_jobs": config.free_water_n_jobs,
            "vox_per_chunk": config.free_water_vox_per_chunk,
            "max_tissue_md_mm2_s": FREE_WATER_MAX_TISSUE_MD,
            "max_free_water_fraction_for_tissue": float(config.free_water_max_fraction_for_tissue),
            "run_enabled": bool(config.run_free_water),
            "outputs_available": free_water_available,
        },
        "noddi_model": {
            "name": "NODDI",
            "library": "AMICO",
            "run_enabled": bool(config.run_noddi),
            "outputs_available": noddi_available,
            "directional_average": bool(config.noddi_directional_average),
            "b0_threshold": int(config.noddi_b0_threshold),
            "lmax": int(config.noddi_lmax),
            "ndirs": int(config.noddi_ndirs),
            "nthreads": _noddi_threads(config),
            "solver": {
                "lambda1": float(config.noddi_solver_lambda1),
                "lambda2": float(config.noddi_solver_lambda2),
            },
            "max_isotropic_fraction_for_tissue": float(config.noddi_max_isotropic_fraction_for_tissue),
        },
        "roi_label_map": _file_signature(roi_label_map) if roi_label_map is not None else None,
    }


def _expected_free_water_manifest(config: MicrostructureConfig) -> dict[str, object]:
    return {
        "kind": "microstructure_freewater_stage",
        "version": 2,
        "subject_id": config.subject_id,
        "raw_input_bundle": {
            "dwi": _file_signature(config.raw_input_bundle.dwi_path),
            "bval": _file_signature(config.raw_input_bundle.bval_path),
            "bvec": _file_signature(config.raw_input_bundle.bvec_path),
            "mask": _file_signature(config.raw_input_bundle.mask_path),
        },
        "free_water_model": {
            "fit_method": config.free_water_fit_method.upper(),
            "chunk_size": int(config.free_water_chunk_size),
            "engine": config.free_water_engine,
            "n_jobs": config.free_water_n_jobs,
            "vox_per_chunk": config.free_water_vox_per_chunk,
            "max_tissue_md_mm2_s": FREE_WATER_MAX_TISSUE_MD,
            "max_free_water_fraction_for_tissue": float(config.free_water_max_fraction_for_tissue),
        },
    }


def _expected_noddi_manifest(config: MicrostructureConfig) -> dict[str, object]:
    return {
        "kind": "microstructure_noddi_stage",
        "version": 3,
        "subject_id": config.subject_id,
        "raw_input_bundle": {
            "dwi": _file_signature(config.raw_input_bundle.dwi_path),
            "bval": _file_signature(config.raw_input_bundle.bval_path),
            "bvec": _file_signature(config.raw_input_bundle.bvec_path),
            "mask": _file_signature(config.raw_input_bundle.mask_path),
        },
        "noddi_model": {
            "name": "NODDI",
            "library": "AMICO",
            "directional_average": bool(config.noddi_directional_average),
            "b0_threshold": int(config.noddi_b0_threshold),
            "lmax": int(config.noddi_lmax),
            "ndirs": int(config.noddi_ndirs),
            "nthreads": _noddi_threads(config),
            "solver": {
                "lambda1": float(config.noddi_solver_lambda1),
                "lambda2": float(config.noddi_solver_lambda2),
            },
            "max_isotropic_fraction_for_tissue": float(config.noddi_max_isotropic_fraction_for_tissue),
            "residual_maps": ["RMSE", "NRMSE"],
        },
    }


def _outputs_complete(
    paths: dict[str, Path],
    *,
    roi_enabled: bool,
    free_water_enabled: bool,
    noddi_enabled: bool,
    brain_mask_path: Path | None = None,
) -> bool:
    required = [
        paths["fa"],
        paths["md"],
        paths["ad"],
        paths["rd"],
        paths["l1"],
        paths["l2"],
        paths["l3"],
        paths["trace"],
        paths["dec"],
        paths["manifest"],
    ]
    if free_water_enabled:
        required.extend(
            [
                paths["fwf"],
                paths["fw_tissue_fa"],
                paths["fw_tissue_md"],
                paths["fw_tissue_valid_mask"],
                paths["free_water_qc"],
                paths["free_water_manifest"],
            ]
        )
    if noddi_enabled:
        required.extend(
            [
                paths["noddi_ndi"],
                paths["noddi_odi"],
                paths["noddi_fwf"],
                paths["noddi_tissue_valid_mask"],
                paths["noddi_rmse"],
                paths["noddi_nrmse"],
                paths["noddi_qc"],
                paths["noddi_manifest"],
            ]
        )
    if roi_enabled:
        required.extend([paths["roi_stats_csv"], paths["roi_stats_json"]])
    return (
        all(path.is_file() for path in required)
        and (not roi_enabled or roi_statistics_current(
            paths["roi_stats_json"], paths["roi_stats_csv"], required_metrics=DTI_KEYS,
            brain_mask_path=brain_mask_path))
        and scalar_qc_current(paths["dti_qc"], [paths[key] for key in DTI_KEYS])
        and (not free_water_enabled or _free_water_outputs_complete(paths))
        and (not noddi_enabled or _noddi_outputs_complete(paths))
    )


def _free_water_outputs_complete(paths: dict[str, Path]) -> bool:
    return scalar_qc_current(paths["free_water_qc"], [paths[key] for key in FREE_WATER_KEYS])


def _free_water_maps_complete(paths: dict[str, Path]) -> bool:
    return all(
        paths[key].is_file()
        for key in ("fwf", "fw_tissue_fa", "fw_tissue_md", "fw_tissue_valid_mask")
    )


def _noddi_outputs_complete(paths: dict[str, Path]) -> bool:
    return scalar_qc_current(paths["noddi_qc"], [paths[key] for key in NODDI_KEYS])


def _free_water_outputs_current(config: MicrostructureConfig, paths: dict[str, Path]) -> bool:
    """Return whether reusable Free Water maps match the current input/config."""

    return (
        _free_water_outputs_complete(paths)
        and _read_json(paths["free_water_manifest"]) == _expected_free_water_manifest(config)
    )


def _noddi_outputs_current(config: MicrostructureConfig, paths: dict[str, Path]) -> bool:
    """Return whether reusable NODDI maps match the current input/config."""

    return (
        _noddi_outputs_complete(paths)
        and _read_json(paths["noddi_manifest"]) == _expected_noddi_manifest(config)
    )


def _compute_rd_map(l2_path: Path, l3_path: Path, output_path: Path) -> None:
    _ensure_base_imports_ready()
    l2_img = nib.load(str(l2_path))
    l3_img = nib.load(str(l3_path))
    l2 = np.asarray(l2_img.get_fdata(dtype=np.float32), dtype=np.float32)
    l3 = np.asarray(l3_img.get_fdata(dtype=np.float32), dtype=np.float32)
    rd = (l2 + l3) / np.float32(2.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(rd, l2_img.affine, l2_img.header), str(output_path))


@scheduled_stage('microstructure_dti')
def _export_dti_scalar_maps(config: MicrostructureConfig, paths: dict[str, Path]) -> None:
    # Selected FW/NODDI/DKI fits may be forced; immutable DTI prerequisites
    # remain reusable when their own source and QC receipts still match.
    if _dti_outputs_current(config, paths):
        from shared.cohort_schedule import mark_reused
        mark_reused('microstructure_dti')
        return
    source_before = _dti_source_signature(config)
    native = config.native_bundle
    _copy_if_needed(native.fa_nii_path, paths["fa"])
    _copy_if_needed(native.mean_diffusivity_nii_path, paths["md"])
    _copy_if_needed(native.max_eig_nii_path, paths["l1"])
    _copy_if_needed(native.mid_eig_nii_path, paths["l2"])
    _copy_if_needed(native.min_eig_nii_path, paths["l3"])
    _copy_if_needed(native.trace_nii_path, paths["trace"])
    _copy_if_needed(native.dec_nrrd_path, paths["dec"])
    _copy_if_needed(native.max_eig_nii_path, paths["ad"])
    # RD belongs to the current eigenvalue export, never to a previous input.
    _compute_rd_map(paths["l2"], paths["l3"], paths["rd"])
    _write_dti_qc(config, paths)
    if source_before != _dti_source_signature(config):
        _write_json(paths["dti_qc"], {"status": "failed", "automatic_qc_passed": False,
                                   "error": "Native DTI source changed during scalar export"})
        raise RuntimeError("Native DTI source changed during scalar export; outputs are not reusable")


def _dti_source_signature(config: MicrostructureConfig) -> dict[str, object]:
    native = config.native_bundle
    return {name: _file_signature(Path(getattr(native, name))) for name in (
        "fa_nii_path", "mean_diffusivity_nii_path", "max_eig_nii_path", "mid_eig_nii_path",
        "min_eig_nii_path", "trace_nii_path", "dec_nrrd_path", "manifest_path",
    )} | {"mask": _file_signature(config.raw_input_bundle.mask_path)}


def _dti_outputs_current(config: MicrostructureConfig, paths: dict[str, Path]) -> bool:
    try:
        return (scalar_qc_current(paths["dti_qc"], [paths[key] for key in DTI_KEYS])
                and paths["dec"].is_file()
                and (_read_json(paths["dti_qc"]) or {}).get("source_signature") == _dti_source_signature(config))
    except OSError:
        return False


def _load_qc_maps(config, paths, keys, qc_key):
    """Invalidate old green QC before decoding, including on corrupt geometry."""
    _ensure_base_imports_ready()
    _write_json(paths[qc_key], {"status": "running", "automatic_qc_passed": False})
    try:
        mask_img = nib.load(str(config.raw_input_bundle.mask_path))
        mask_data = np.asarray(mask_img.dataobj)
        if mask_data.ndim != 3 or not np.isfinite(mask_data).all() or not np.any(mask_data > 0):
            raise ValueError("QC requires a finite, nonempty 3D brain mask")
        mask = mask_data > 0
        images = {key: nib.load(str(paths[key])) for key in keys}
        for key, image in images.items():
            if image.shape != mask.shape or not np.allclose(image.affine, mask_img.affine, rtol=0, atol=1e-4):
                raise ValueError(f"{key} does not match the native mask grid")
        arrays = {key: image.get_fdata(dtype=np.float32) for key, image in images.items()}
        return mask_img, mask, arrays
    except Exception as exc:
        _write_json(paths[qc_key], {"status": "failed", "automatic_qc_passed": False,
                                  "subject_id": config.subject_id, "error": str(exc)})
        raise


def _finish_scalar_qc(paths, qc_key, keys, payload):
    passed = all(value is True for value in payload["checks"].values())
    payload.update({"status": "completed" if passed else "failed", "automatic_qc_passed": passed,
                    "human_review": "pending", "updated_at": _utc_now(),
                    "output_artifacts": {str(paths[key].resolve()): _file_signature(paths[key]) for key in keys}})
    _write_json(paths[qc_key], payload)
    if not passed:
        failed = ", ".join(key for key, value in payload["checks"].items() if not value)
        raise RuntimeError(f"{qc_key} failed automatic QC: {failed}")
    return payload


def _write_dti_qc(config, paths):
    _, mask, arrays = _load_qc_maps(config, paths, DTI_KEYS, "dti_qc")
    eigen_sum = arrays["l1"] + arrays["l2"] + arrays["l3"]
    trace_close = np.isclose(arrays["trace"], eigen_sum, rtol=1e-4, atol=1e-8)
    # Slicer's Trace uses the original tensor; its eigensystem scalars clamp
    # negative eigenvalues. Differences are allowed only at that clamp boundary.
    trace_explained = trace_close | ((arrays["l3"] <= 1e-8) & (arrays["trace"] <= eigen_sum + 1e-8))
    checks = {
        "native_grid_matches": True,
        "all_values_finite": all(bool(np.isfinite(array).all()) for array in arrays.values()),
        "fa_in_range": bool(np.all((arrays["fa"][mask] >= 0) & (arrays["fa"][mask] <= 1.00001))),
        "nonempty_brain_signal": bool(np.any(arrays["md"][mask] > 0)),
        "ad_equals_l1": bool(np.allclose(arrays["ad"], arrays["l1"], rtol=1e-4, atol=1e-8)),
        "rd_equals_mean_l2_l3": bool(np.allclose(arrays["rd"], (arrays["l2"] + arrays["l3"]) / 2, rtol=1e-4, atol=1e-8)),
        "eigenvalues_nonnegative_ordered": bool(np.all((arrays["l3"][mask] >= -1e-8)
            & (arrays["l1"][mask] >= arrays["l2"][mask] - 1e-8) & (arrays["l2"][mask] >= arrays["l3"][mask] - 1e-8))),
        "trace_matches_or_negative_eigenvalue_clamp": bool(np.all(trace_explained[mask])),
        "md_equals_eigenvalue_mean": bool(np.allclose(arrays["md"][mask], eigen_sum[mask] / 3, rtol=1e-4, atol=1e-8)),
    }
    return _finish_scalar_qc(paths, "dti_qc", DTI_KEYS, {
        "kind": "dti_export_qc", "version": 1, "subject_id": config.subject_id,
        "source_signature": _dti_source_signature(config), "checks": checks,
        "warnings": ["Scalar integrity/relations are checked; this is not an anatomical or model-fit accuracy certification."],
        "trace_clamp_boundary_voxels": int(np.count_nonzero(mask & ~trace_close)),
        "reference": "https://github.com/Slicer/Slicer/blob/main/Libs/vtkTeem/vtkDiffusionTensorMathematics.cxx",
        "negative_eigenvalue_voxels": int(np.count_nonzero(mask & ((arrays["l1"] < 0) | (arrays["l2"] < 0) | (arrays["l3"] < 0)))),
    })


def _remove_legacy_roi_outputs(output_dir: Path, subject_id: str) -> None:
    legacy_paths = [
        output_dir / f"{subject_id}-ROIWise-FA.csv",
        output_dir / f"{subject_id}-ROIWise-FA.json",
    ]
    for path in legacy_paths:
        path.unlink(missing_ok=True)


def _load_bvals_bvecs(raw_input_bundle: RawInputBundle) -> tuple[np.ndarray, np.ndarray]:
    bvals = np.loadtxt(raw_input_bundle.bval_path)
    bvecs = np.loadtxt(raw_input_bundle.bvec_path)

    bvals = np.atleast_1d(np.asarray(bvals, dtype=np.float64)).reshape(-1)
    bvecs = np.asarray(bvecs, dtype=np.float64)
    if bvecs.ndim == 1:
        bvecs = bvecs.reshape(1, -1)
    if bvecs.shape[0] == 3 and bvecs.shape[1] == bvals.shape[0]:
        bvecs = bvecs.T
    if bvecs.shape != (bvals.shape[0], 3):
        raise ValueError(
            f"Unexpected bvec shape {bvecs.shape}; expected ({bvals.shape[0]}, 3)."
        )
    return bvals, bvecs


def _save_scalar_nifti(data: np.ndarray, reference_img: nib.Nifti1Image, output_path: Path) -> None:
    header = reference_img.header.copy()
    header.set_data_dtype(np.float32)
    image = nib.Nifti1Image(data.astype(np.float32), reference_img.affine, header)
    nib.save(image, str(output_path))


def _save_mask_nifti(data: np.ndarray, reference_img: nib.Nifti1Image, output_path: Path) -> None:
    header = reference_img.header.copy()
    header.set_data_dtype(np.uint8)
    image = nib.Nifti1Image(data.astype(np.uint8), reference_img.affine, header)
    nib.save(image, str(output_path))


def _sanitize_free_water_scalars(
    fwf: np.ndarray,
    tissue_fa: np.ndarray,
    tissue_md: np.ndarray,
    tissue_evals: np.ndarray,
    *,
    max_tissue_md: float = FREE_WATER_MAX_TISSUE_MD,
    max_free_water_fraction: float = FREE_WATER_MAX_FRACTION_FOR_TISSUE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Return free-water maps and an explicit tissue-fit validity mask.

    Tissue FA and MD are unavailable when free water dominates the voxel or
    when DIPY's tensor decomposition reaches a non-positive boundary. The
    original FWF estimate remains available; unavailable tissue metrics are
    zeroed and accompanied by a machine-readable validity mask.
    """

    raw_fwf = np.asarray(fwf, dtype=np.float32)
    raw_fa = np.asarray(tissue_fa, dtype=np.float32)
    raw_md = np.asarray(tissue_md, dtype=np.float32)
    raw_evals = np.asarray(tissue_evals, dtype=np.float32)
    if raw_evals.shape != raw_fa.shape + (3,):
        raise ValueError(
            f"Expected tissue eigenvalues with shape {raw_fa.shape + (3,)}, got {raw_evals.shape}."
        )
    valid_tissue = (
        np.isfinite(raw_fwf)
        & (raw_fwf >= 0.0)
        & (raw_fwf <= float(max_free_water_fraction))
        & np.isfinite(raw_fa)
        & np.isfinite(raw_md)
        & np.all(np.isfinite(raw_evals), axis=-1)
        & np.all(raw_evals > 0.0, axis=-1)
        & (raw_fa >= 0.0)
        & (raw_fa <= 1.0)
        & (raw_md >= 0.0)
        & (raw_md <= float(max_tissue_md))
    )
    invalid_count = int(np.count_nonzero(~valid_tissue))
    clean_fwf = np.clip(
        np.nan_to_num(raw_fwf, nan=0.0, posinf=1.0, neginf=0.0),
        0.0,
        1.0,
    ).astype(np.float32, copy=False)
    clean_fa = np.where(valid_tissue, raw_fa, 0.0).astype(np.float32, copy=False)
    clean_md = np.where(valid_tissue, raw_md, 0.0).astype(np.float32, copy=False)
    return clean_fwf, clean_fa, clean_md, valid_tissue, invalid_count


def _write_free_water_qc(config: MicrostructureConfig, paths: dict[str, Path], *, fit_implementation=None) -> dict[str, object]:
    """Validate saved Free Water maps and persist a reusable scientific QC record."""

    mask_image, mask, arrays = _load_qc_maps(config, paths, FREE_WATER_KEYS, "free_water_qc")
    arrays["valid_mask"] = arrays.pop("fw_tissue_valid_mask")
    valid = arrays["valid_mask"] > 0.0
    brain_voxels = int(np.count_nonzero(mask))
    valid_voxels = int(np.count_nonzero(valid))
    checks = {
        "shape_matches_mask": all(array.shape == mask.shape for array in arrays.values()),
        "affine_matches_mask": True,
        "all_values_finite": all(bool(np.isfinite(array).all()) for array in arrays.values()),
        "valid_mask_binary": bool(np.all(np.isin(arrays["valid_mask"], (0.0, 1.0)))),
        "valid_mask_within_brain": bool(np.all(~valid | mask)),
        "fwf_in_range": bool(
            np.all((arrays["fwf"][mask] >= 0.0) & (arrays["fwf"][mask] <= 1.0))
        ),
        "tissue_fa_in_range": bool(
            np.all((arrays["fw_tissue_fa"][valid] >= 0.0) & (arrays["fw_tissue_fa"][valid] <= 1.0))
        ),
        "tissue_md_in_range": bool(
            np.all(
                (arrays["fw_tissue_md"][valid] >= 0.0)
                & (arrays["fw_tissue_md"][valid] <= FREE_WATER_MAX_TISSUE_MD)
            )
        ),
        "invalid_tissue_metrics_zero": bool(
            np.allclose(arrays["fw_tissue_fa"][~valid], 0.0, atol=1e-7)
            and np.allclose(arrays["fw_tissue_md"][~valid], 0.0, atol=1e-9)
        ),
        "maps_zero_outside_brain": bool(
            all(np.allclose(array[~mask], 0.0, atol=1e-7) for array in arrays.values())
        ),
        "nonempty_valid_tissue": valid_voxels > 0,
        "valid_tissue_fraction_threshold": bool(np.all(arrays["fwf"][valid] <= config.free_water_max_fraction_for_tissue + 1e-6)),
    }
    payload: dict[str, object] = {
        "kind": "freewater_qc",
        "version": 2,
        "status": "completed" if all(checks.values()) else "failed",
        "subject_id": config.subject_id,
        "model": "DIPY FreeWaterTensorModel",
        "source_signature": {key: _file_signature(Path(getattr(config.raw_input_bundle, key + "_path")))
                             for key in ("dwi", "bval", "bvec", "mask")},
        "input": {
            "dwi": str(config.raw_input_bundle.dwi_path.resolve()),
            "bval": str(config.raw_input_bundle.bval_path.resolve()),
            "bvec": str(config.raw_input_bundle.bvec_path.resolve()),
            "mask": str(config.raw_input_bundle.mask_path.resolve()),
        },
        "public_maps": {
            "fwf": str(paths["fwf"].resolve()),
            "tissue_fa": str(paths["fw_tissue_fa"].resolve()),
            "tissue_md": str(paths["fw_tissue_md"].resolve()),
            "tissue_valid_mask": str(paths["fw_tissue_valid_mask"].resolve()),
        },
        "parameters": {
            "max_free_water_fraction_for_tissue": float(config.free_water_max_fraction_for_tissue),
            "max_tissue_md_mm2_s": FREE_WATER_MAX_TISSUE_MD,
        },
        "checks": checks,
        "validity": {
            "brain_mask_voxels": brain_voxels,
            "valid_tissue_voxels": valid_voxels,
            "valid_tissue_fraction": float(valid_voxels / max(1, brain_voxels)),
            "excluded_tissue_voxels": int(np.count_nonzero(mask & ~valid)),
        },
        "summary": {
            "fwf_brain_mask": _scalar_summary(arrays["fwf"], mask),
            "tissue_fa_valid": _scalar_summary(arrays["fw_tissue_fa"], valid),
            "tissue_md_valid": _scalar_summary(arrays["fw_tissue_md"], valid),
        },
        "warnings": [
            "Tissue FA and MD are interpreted only where the tissue-validity mask is non-zero."
        ],
    }
    if fit_implementation is not None:
        payload['fit_implementation'] = fit_implementation
    return _finish_scalar_qc(paths, "free_water_qc", FREE_WATER_KEYS, payload)


def _write_noddi_qc(config, paths, *, source_maps=None, validity=None):
    """Check saved conditional tissue maps AND residuals without filling voxels."""
    _, mask, arrays = _load_qc_maps(config, paths, NODDI_KEYS, "noddi_qc")
    valid = arrays["noddi_tissue_valid_mask"] > 0
    brain_voxels = int(np.count_nonzero(mask))
    valid_voxels = int(np.count_nonzero(valid))
    checks = {
        "native_grid_matches": True,
        "all_values_finite": all(bool(np.isfinite(array).all()) for array in arrays.values()),
        "valid_mask_binary": bool(np.all(np.isin(arrays["noddi_tissue_valid_mask"], (0, 1)))),
        "valid_mask_within_brain": bool(np.all(~valid | mask)),
        "nonempty_valid_tissue": valid_voxels > 0,
        "parameters_in_range": all(bool(np.all((arrays[key] >= 0) & (arrays[key] <= 1))) for key in NODDI_KEYS[:3]),
        "parameters_zero_outside_brain": all(bool(np.all(arrays[key][~mask] == 0)) for key in NODDI_KEYS[:3]),
        "residuals_nonnegative": all(bool(np.all(arrays[key][mask] >= 0)) for key in ("noddi_rmse", "noddi_nrmse")),
        "valid_tissue_fraction_threshold": bool(np.all(arrays["noddi_fwf"][valid] <= config.noddi_max_isotropic_fraction_for_tissue + 1e-6)),
    }
    summaries = {name: _scalar_summary(arrays[key], scope) for name, key, scope in (
        ("ndi_valid_tissue", "noddi_ndi", valid), ("odi_valid_tissue", "noddi_odi", valid),
        ("isovf_brain_mask", "noddi_fwf", mask), ("rmse_valid_tissue", "noddi_rmse", valid),
        ("nrmse_valid_tissue", "noddi_nrmse", valid),
    )}
    warnings = ["NDI/ODI display retains brain estimates; quantitative summaries use the separate tissue-validity mask. Anatomical review remains required."]
    if valid_voxels < brain_voxels / 2:
        warnings.append("Fewer than half the brain-mask voxels are tissue-valid; inspect IsoVF and fit quality.")
    if summaries["nrmse_valid_tissue"]["p95"] > 0.5:
        warnings.append("NRMSE p95 exceeds the local review threshold 0.5; inspect residual maps and acquisition compatibility.")
    return _finish_scalar_qc(paths, "noddi_qc", NODDI_KEYS, {
        "kind": "noddi_qc", "version": 2, "subject_id": config.subject_id,
        "model": "AMICO NODDI", "checks": checks,
        "source_signature": {key: _file_signature(Path(getattr(config.raw_input_bundle, key + "_path")))
                             for key in ("dwi", "bval", "bvec", "mask") if hasattr(config.raw_input_bundle, key + "_path")},
        "source_maps": {key: str(Path(path).resolve()) for key, path in (source_maps or {}).items()},
        "public_maps": {key: str(paths[key].resolve()) for key in NODDI_KEYS},
        "validity": {**(validity or {}), "brain_mask_voxels": brain_voxels, "valid_tissue_voxels": valid_voxels,
                     "valid_tissue_fraction": valid_voxels / max(1, brain_voxels)},
        "summary": summaries, "warnings": warnings,
    })


@scheduled_stage('microstructure_freewater')
def _compute_free_water_maps(
    *,
    config: MicrostructureConfig,
    paths: dict[str, Path],
    log: Optional[Callable[[str], None]] = None,
) -> None:
    from shared.diffusion_io import diffusion_image

    started = time.monotonic()
    signature = _expected_free_water_manifest(config)
    # Without indexed_gzip, every axial slab can re-decompress the entire 4D
    # file. Use one sequential decode and disk-backed slabs, not full-RAM data.
    with diffusion_image(config.raw_input_bundle.dwi_path, paths["fwf"].parent) as image:
        _compute_free_water_maps_impl(config=config, paths=paths, log=log, dwi_img=image, started=started)
    if signature != _expected_free_water_manifest(config):
        _write_json(paths["free_water_qc"], {"status": "failed", "automatic_qc_passed": False,
                                            "error": "Free Water inputs changed during fitting"})
        raise RuntimeError("Free Water inputs changed during fitting; results are not reusable")


_ORIG_FW_FIT = FreeWaterTensorModel.fit if FreeWaterTensorModel is not None else None


def _compute_free_water_maps_impl(
    *,
    config: MicrostructureConfig,
    paths: dict[str, Path],
    dwi_img,
    started: float | None = None,
    log: Optional[Callable[[str], None]] = None,
) -> None:
    raw = config.raw_input_bundle
    fit_method = config.free_water_fit_method.upper()
    chunk_size = max(1, int(config.free_water_chunk_size))
    engine = (config.free_water_engine or "serial").lower()
    n_jobs = config.free_water_n_jobs
    vox_per_chunk = config.free_water_vox_per_chunk
    progress_path = paths["free_water_progress"]
    started = time.monotonic() if started is None else started

    bvals, bvecs = _load_bvals_bvecs(raw)
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=50)
    from dipy.core.gradients import check_multi_b
    from microstructure.ifw import FREE_WATER_INELIGIBLE
    if not check_multi_b(gtab, 3, non_zero=False):
        raise ValueError(FREE_WATER_INELIGIBLE)
    fw_model = FreeWaterTensorModel(gtab, fit_method=fit_method)
    from microstructure.freewater_dispatch import configure_freewater_dispatch
    from microstructure.freewater_wls import _resolve_fw_device, wls_batch_fit
    fit_implementation = configure_freewater_dispatch(fw_model, fit_method, engine)
    use_batch_wls = (
        bool(fit_implementation.get("enabled"))
        and fit_method == "WLS"
        and engine == "serial"
        and _ORIG_FW_FIT is not None
        and getattr(type(fw_model), "fit", None) is _ORIG_FW_FIT
        and "fit" not in fw_model.__dict__
    )
    resolved_fw_device = _resolve_fw_device(config.free_water_device) if use_batch_wls else "cpu"
    if use_batch_wls:
        fit_implementation = {
            **fit_implementation,
            "batched_wls": True,
            "device": resolved_fw_device,
        }
    _emit(f"[Microstructure] Free Water dispatch: {fit_implementation}", log)

    rounded_shells = np.unique(np.round(bvals / 100.0) * 100.0).astype(int).tolist()
    _emit(
        f"[Microstructure] Free Water model: fit={fit_method}, chunk_size={chunk_size}, "
        f"engine={engine}, n_jobs={n_jobs}, vox_per_chunk={vox_per_chunk}, shells={rounded_shells}",
        log,
    )

    mask_img = nib.load(str(raw.mask_path))
    mask_data = mask_img.get_fdata(dtype=np.float32)
    mask = mask_data > 0

    if dwi_img.shape[:3] != mask.shape or not np.allclose(dwi_img.affine, mask_img.affine, rtol=0, atol=1e-4):
        raise ValueError(
            f"DWI shape {dwi_img.shape[:3]} does not match mask shape {mask.shape}."
        )
    if not np.isfinite(mask_data).all() or not mask.any():
        raise ValueError("Free Water requires a finite, nonempty native-space mask")
    if len(dwi_img.shape) != 4 or dwi_img.shape[3] != len(bvals):
        raise ValueError("Free Water DWI/gradient volume counts differ")

    fwf_map = np.zeros(mask.shape, dtype=np.float32)
    fw_tissue_fa_map = np.zeros(mask.shape, dtype=np.float32)
    fw_tissue_md_map = np.zeros(mask.shape, dtype=np.float32)
    fw_tissue_valid_mask = np.zeros(mask.shape, dtype=bool)

    z_size = mask.shape[2]
    if use_batch_wls and chunk_size == 2 and z_size >= 16:
        from shared.hardware_adapt import adaptive_cpu_slab_z
        step_z = adaptive_cpu_slab_z(dwi_img.shape, 16, min_step_z=2)
    else:
        step_z = chunk_size
    chunk_count = (z_size + step_z - 1) // step_z
    gradient_count = dwi_img.shape[3]
    total_voxels = int(mask.sum())
    completed_chunks = 0
    completed_voxels = 0
    invalid_tissue_voxels = 0

    progress_base: dict[str, object] = {
        "kind": "microstructure_stage_progress",
        "version": 1,
        "subject_id": config.subject_id,
        "stage": "freewater",
        "status": "running",
        "fit_method": fit_method,
        "fit_implementation": fit_implementation,
        "chunk_size": step_z,
        "engine": engine,
        "n_jobs": n_jobs,
        "vox_per_chunk": vox_per_chunk,
        "chunk_count": chunk_count,
        "total_voxels": total_voxels,
        "gradient_count": int(gradient_count),
        "shells": rounded_shells,
        "max_free_water_fraction_for_tissue": float(config.free_water_max_fraction_for_tissue),
    }
    _write_progress(
        progress_path,
        {
            **progress_base,
            "completed_chunks": completed_chunks,
            "completed_voxels": completed_voxels,
            "current_chunk": None,
            "elapsed_seconds": 0.0,
        },
    )

    last_progress_time = 0.0
    for chunk_index, z_start in enumerate(range(0, z_size, step_z), start=1):
        z_stop = min(z_size, z_start + step_z)
        mask_chunk = mask[:, :, z_start:z_stop]
        voxel_count = int(mask_chunk.sum())
        if voxel_count == 0:
            completed_chunks += 1
            continue

        _emit(
            f"[Microstructure] Free Water chunk {chunk_index}/{chunk_count}: z={z_start}:{z_stop}, voxels={voxel_count}",
            log,
        )

        data_chunk = np.asarray(dwi_img.dataobj[:, :, z_start:z_stop, :], dtype=np.float32)
        voxel_signals = data_chunk.reshape(-1, gradient_count)[mask_chunk.reshape(-1)]
        if use_batch_wls:
            fit, _ = wls_batch_fit(fw_model, voxel_signals, device=resolved_fw_device)
        else:
            fit_kwargs: dict[str, object] = {"engine": engine}
            if n_jobs is not None:
                fit_kwargs["n_jobs"] = int(n_jobs)
            if vox_per_chunk is not None:
                fit_kwargs["vox_per_chunk"] = int(vox_per_chunk)
            fit = fw_model.fit(voxel_signals, **fit_kwargs)

        fwf, fw_fa, fw_md, valid_tissue, invalid_count = _sanitize_free_water_scalars(
            fit.f,
            fit.fa,
            fit.md,
            fit.evals,
            max_free_water_fraction=config.free_water_max_fraction_for_tissue,
        )
        invalid_tissue_voxels += invalid_count

        fwf_chunk = np.zeros(mask_chunk.shape, dtype=np.float32)
        fw_fa_chunk = np.zeros(mask_chunk.shape, dtype=np.float32)
        fw_md_chunk = np.zeros(mask_chunk.shape, dtype=np.float32)
        valid_chunk = np.zeros(mask_chunk.shape, dtype=bool)

        flat_mask = mask_chunk.reshape(-1)
        fwf_chunk.reshape(-1)[flat_mask] = fwf
        fw_fa_chunk.reshape(-1)[flat_mask] = fw_fa
        fw_md_chunk.reshape(-1)[flat_mask] = fw_md
        valid_chunk.reshape(-1)[flat_mask] = valid_tissue

        fwf_map[:, :, z_start:z_stop] = fwf_chunk
        fw_tissue_fa_map[:, :, z_start:z_stop] = fw_fa_chunk
        fw_tissue_md_map[:, :, z_start:z_stop] = fw_md_chunk
        fw_tissue_valid_mask[:, :, z_start:z_stop] = valid_chunk
        completed_chunks += 1
        completed_voxels += voxel_count
        now_prog = time.monotonic()
        if chunk_index == 1 or chunk_index == chunk_count or (now_prog - last_progress_time >= 1.0):
            _write_progress(
                progress_path,
                {
                    **progress_base,
                    "completed_chunks": completed_chunks,
                    "completed_voxels": completed_voxels,
                    "current_chunk": chunk_index,
                    "z_start": z_start,
                    "z_stop": z_stop,
                    "voxel_count": voxel_count,
                    "invalid_tissue_voxels": invalid_tissue_voxels,
                    "elapsed_seconds": round(now_prog - started, 3),
                },
            )
            last_progress_time = now_prog

    _save_scalar_nifti(fwf_map, mask_img, paths["fwf"])
    _save_scalar_nifti(fw_tissue_fa_map, mask_img, paths["fw_tissue_fa"])
    _save_scalar_nifti(fw_tissue_md_map, mask_img, paths["fw_tissue_md"])
    _save_mask_nifti(fw_tissue_valid_mask, mask_img, paths["fw_tissue_valid_mask"])
    _write_free_water_qc(config, paths, fit_implementation=fit_implementation)
    _write_progress(
        progress_path,
        {
            **progress_base,
            "status": "completed",
            "completed_chunks": completed_chunks,
            "completed_voxels": completed_voxels,
            "invalid_tissue_voxels": invalid_tissue_voxels,
            "max_tissue_md_mm2_s": FREE_WATER_MAX_TISSUE_MD,
            "max_free_water_fraction_for_tissue": float(config.free_water_max_fraction_for_tissue),
            "current_chunk": None,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "outputs": {
                "fwf": paths["fwf"].as_posix(),
                "fw_tissue_fa": paths["fw_tissue_fa"].as_posix(),
                "fw_tissue_md": paths["fw_tissue_md"].as_posix(),
                "fw_tissue_valid_mask": paths["fw_tissue_valid_mask"].as_posix(),
            },
        },
    )


def _amico_cache_root(config: MicrostructureConfig) -> Path:
    # AMICO skips generation when ANY A_*.npy exists. Different protocols,
    # directions or dictionaries must never share that upstream kernel path.
    payload = {"stage": _expected_noddi_manifest(config), "amico_version": getattr(amico, "__version__", None)}
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]
    return config.raw_input_bundle.subject_root / "cache" / "microstructure" / "amico" / key


def _noddi_threads(config) -> int:
    if config.noddi_nthreads is not None:
        count = int(config.noddi_nthreads)
        if count < 1:
            raise ValueError("noddi_nthreads must be positive or None (automatic)")
        return count
    try:
        available = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available = os.cpu_count() or 1
    budget = int(os.environ.get('SLICERDSPACE_STAGE_CPUS', available))
    return max(1, min(8, available, budget))


def _ensure_amico_rotation_matrices(lmax: int, ndirs: int, log) -> None:
    """Run amico.setup() once when AMICO's precomputed rotation matrices are missing.

    AMICO 2.x reports the missing files with util.ERROR(), which calls exit() (SystemExit) instead of
    raising, so the check below loads them before generate_kernels() rather than catching its failure.
    """
    try:
        amico.lut.load_precomputed_rotation_matrices(lmax, ndirs)
    except (Exception, SystemExit):
        _emit("[Microstructure] AMICO rotation matrices missing; running amico.setup() once.", log)
        amico.setup(lmax=lmax)


def _amico_subject_dir(config: MicrostructureConfig) -> Path:
    return _amico_cache_root(config) / config.subject_id


def _amico_output_dir(config: MicrostructureConfig) -> Path:
    return config.output_dir / "noddi" / "_amico_noddi"


def _prepare_amico_inputs(
    *,
    config: MicrostructureConfig,
    log: Optional[Callable[[str], None]] = None,
) -> tuple[Path, Path, Path, Path, Path]:
    if fsl2scheme is None:
        raise RuntimeError("AMICO is unavailable; cannot prepare NODDI inputs.")

    study_dir = _amico_subject_dir(config)
    study_dir.mkdir(parents=True, exist_ok=True)

    raw = config.raw_input_bundle
    dwi_path = study_dir / raw.dwi_path.name
    bval_path = study_dir / raw.bval_path.name
    bvec_path = study_dir / raw.bvec_path.name
    mask_path = study_dir / raw.mask_path.name
    scheme_path = study_dir / "DWI.scheme"

    _link_or_copy(raw.dwi_path, dwi_path)
    _link_or_copy(raw.bval_path, bval_path)
    _link_or_copy(raw.bvec_path, bvec_path)
    _link_or_copy(raw.mask_path, mask_path)
    fsl2scheme(str(bval_path), str(bvec_path), schemeFilename=str(scheme_path))

    _emit(
        f"[Microstructure] AMICO study cache prepared at {study_dir}",
        log,
    )
    return study_dir, dwi_path, mask_path, bval_path, scheme_path


def _sanitize_noddi_maps(
    ndi: np.ndarray,
    odi: np.ndarray,
    isovf: np.ndarray,
    brain_mask: np.ndarray,
    *,
    max_isotropic_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Sanitize public NODDI maps and derive a separate tissue-validity mask.

    AMICO preserves its raw estimates in ``_amico_noddi``. The public NDI/ODI
    maps retain finite, bounded estimates throughout the brain mask so that
    display does not create artificial holes. Quantitative summaries use the
    separate tissue-validity mask because conditional tissue fractions are not
    interpreted in voxels dominated by the isotropic component.
    """
    raw_ndi = np.asarray(ndi, dtype=np.float32)
    raw_odi = np.asarray(odi, dtype=np.float32)
    raw_isovf = np.asarray(isovf, dtype=np.float32)
    mask = np.asarray(brain_mask, dtype=bool)
    if not (raw_ndi.shape == raw_odi.shape == raw_isovf.shape == mask.shape):
        raise ValueError("NODDI maps and brain mask must have identical shapes.")

    finite_and_bounded = (
        np.isfinite(raw_ndi)
        & np.isfinite(raw_odi)
        & np.isfinite(raw_isovf)
        & (raw_ndi >= 0.0)
        & (raw_ndi <= 1.0)
        & (raw_odi >= 0.0)
        & (raw_odi <= 1.0)
        & (raw_isovf >= 0.0)
        & (raw_isovf <= 1.0)
    )
    high_isotropic = raw_isovf > float(max_isotropic_fraction)
    valid_tissue = mask & finite_and_bounded & ~high_isotropic
    valid_brain_estimate = mask & finite_and_bounded
    clean_ndi = np.where(valid_brain_estimate, raw_ndi, 0.0).astype(np.float32, copy=False)
    clean_odi = np.where(valid_brain_estimate, raw_odi, 0.0).astype(np.float32, copy=False)
    clean_isovf = np.where(
        mask,
        np.clip(np.nan_to_num(raw_isovf, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0),
        0.0,
    ).astype(np.float32, copy=False)
    valid_count = int(np.count_nonzero(valid_tissue))
    valid_ndi = raw_ndi[valid_tissue]
    qc = {
        "brain_mask_voxels": int(np.count_nonzero(mask)),
        "valid_tissue_voxels": valid_count,
        "valid_tissue_fraction": float(valid_count / max(1, int(np.count_nonzero(mask)))),
        "excluded_high_isotropic_voxels": int(np.count_nonzero(mask & high_isotropic)),
        "excluded_nonfinite_or_out_of_range_voxels": int(np.count_nonzero(mask & ~finite_and_bounded)),
        "max_isotropic_fraction_for_tissue": float(max_isotropic_fraction),
        "ndi_upper_boundary_voxels_in_valid_tissue": int(
            np.count_nonzero(valid_ndi >= NODDI_UPPER_BOUND)
        ),
        "ndi_upper_boundary_fraction_in_valid_tissue": float(
            np.count_nonzero(valid_ndi >= NODDI_UPPER_BOUND) / max(1, valid_ndi.size)
        ),
    }
    return clean_ndi, clean_odi, clean_isovf, valid_tissue, qc


def _scalar_summary(data: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(data, dtype=np.float32)[np.asarray(mask, dtype=bool)]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "min": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


@scheduled_stage('microstructure_noddi')
def _compute_noddi_maps(
    *,
    config: MicrostructureConfig,
    paths: dict[str, Path],
    log: Optional[Callable[[str], None]] = None,
) -> None:
    if amico is None:
        raise RuntimeError("AMICO is required for NODDI modeling.")
    signature = _expected_noddi_manifest(config)
    progress_path = paths["noddi_progress"]
    started = time.monotonic()

    def write_noddi_progress(status: str, step: str, **extra: object) -> None:
        _write_progress(
            progress_path,
            {
                "kind": "microstructure_stage_progress",
                "version": 1,
                "subject_id": config.subject_id,
                "stage": "noddi",
                "status": status,
                "current_step": step,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "directional_average": bool(config.noddi_directional_average),
                "b0_threshold": int(config.noddi_b0_threshold),
                "lmax": int(config.noddi_lmax),
                "ndirs": int(config.noddi_ndirs),
                "nthreads": _noddi_threads(config),
                **extra,
            },
        )

    write_noddi_progress("running", "checking_shells")

    bvals, _ = _load_bvals_bvecs(config.raw_input_bundle)
    nonzero_shells = sorted(
        {
            int(shell)
            for shell in np.unique(np.round(bvals / 100.0) * 100.0)
            if int(shell) > int(config.noddi_b0_threshold)
        }
    )
    if len(nonzero_shells) < 2:
        raise ValueError(
            "NODDI requires multi-shell diffusion data with at least two non-zero shells."
        )

    write_noddi_progress("running", "preparing_amico_inputs", nonzero_shells=nonzero_shells)
    study_dir, dwi_path, mask_path, _, scheme_path = _prepare_amico_inputs(config=config, log=log)
    amico_output_dir = _amico_output_dir(config)
    amico_output_dir.mkdir(parents=True, exist_ok=True)

    _emit(
        "[Microstructure] AMICO NODDI: "
        f"directional_average={config.noddi_directional_average}, "
        f"b0_thr={config.noddi_b0_threshold}, "
        f"shells={nonzero_shells}",
        log,
    )
    amico.set_verbose(2)
    write_noddi_progress(
        "running",
        "loading_data",
        nonzero_shells=nonzero_shells,
        study_dir=study_dir.as_posix(),
        amico_output_dir=amico_output_dir.as_posix(),
    )
    evaluation = amico.Evaluation(
        study_path=str(_amico_cache_root(config)),
        subject=config.subject_id,
        output_path=str(amico_output_dir),
    )
    evaluation.set_config("doDirectionalAverage", bool(config.noddi_directional_average))
    evaluation.set_config("doSaveModulatedMaps", False)
    evaluation.set_config("doComputeRMSE", True)
    evaluation.set_config("doComputeNRMSE", True)
    evaluation.set_config("nthreads", _noddi_threads(config))
    evaluation.set_config("BLAS_nthreads", 1)
    evaluation.load_data(
        dwi_filename=dwi_path.name,
        scheme_filename=scheme_path.name,
        mask_filename=mask_path.name,
        b0_thr=int(config.noddi_b0_threshold),
    )
    evaluation.set_model("NODDI")
    evaluation.set_solver(
        lambda1=float(config.noddi_solver_lambda1),
        lambda2=float(config.noddi_solver_lambda2),
    )
    write_noddi_progress("running", "generating_kernels", nonzero_shells=nonzero_shells)
    _ensure_amico_rotation_matrices(int(config.noddi_lmax), int(config.noddi_ndirs), log)
    try:
        evaluation.generate_kernels(
            regenerate=False,
            lmax=int(config.noddi_lmax),
            ndirs=int(config.noddi_ndirs),
        )
    except Exception as exc:
        message = str(exc)
        if "Auxiliary matrices not found" not in message and "amico.setup()" not in message:
            raise
        _emit("[Microstructure] AMICO kernels missing; running amico.setup() once and retrying.", log)
        amico.setup(lmax=int(config.noddi_lmax))
        evaluation.generate_kernels(
            regenerate=False,
            lmax=int(config.noddi_lmax),
            ndirs=int(config.noddi_ndirs),
        )
    write_noddi_progress("running", "loading_kernels", nonzero_shells=nonzero_shells)
    evaluation.load_kernels()
    write_noddi_progress("running", "fitting_model", nonzero_shells=nonzero_shells)
    evaluation.fit()
    write_noddi_progress("running", "saving_results", nonzero_shells=nonzero_shells)
    evaluation.save_results()

    source_dir = amico_output_dir
    noddi_sources = {
        "noddi_ndi": source_dir / "fit_NDI.nii.gz",
        "noddi_odi": source_dir / "fit_ODI.nii.gz",
        "noddi_fwf": source_dir / "fit_FWF.nii.gz",
        "noddi_rmse": source_dir / "fit_RMSE.nii.gz",
        "noddi_nrmse": source_dir / "fit_NRMSE.nii.gz",
    }
    for key, src in noddi_sources.items():
        if not src.is_file():
            raise FileNotFoundError(f"AMICO did not produce expected NODDI map: {src}")

    mask_img = nib.load(str(config.raw_input_bundle.mask_path))
    mask = mask_img.get_fdata(dtype=np.float32) > 0.0
    source_images = {key: nib.load(str(src)) for key, src in noddi_sources.items()}
    source_arrays = {
        key: image.get_fdata(dtype=np.float32)
        for key, image in source_images.items()
    }
    for key, image in source_images.items():
        if image.shape != mask.shape or not np.allclose(image.affine, mask_img.affine, atol=1e-4):
            raise ValueError(f"AMICO {key} output does not match the NODDI input mask grid.")
    clean_ndi, clean_odi, clean_isovf, valid_tissue, validity_qc = _sanitize_noddi_maps(
        source_arrays["noddi_ndi"],
        source_arrays["noddi_odi"],
        source_arrays["noddi_fwf"],
        mask,
        max_isotropic_fraction=config.noddi_max_isotropic_fraction_for_tissue,
    )
    _save_scalar_nifti(clean_ndi, mask_img, paths["noddi_ndi"])
    _save_scalar_nifti(clean_odi, mask_img, paths["noddi_odi"])
    _save_scalar_nifti(clean_isovf, mask_img, paths["noddi_fwf"])
    _save_mask_nifti(valid_tissue, mask_img, paths["noddi_tissue_valid_mask"])
    shutil.copy2(noddi_sources["noddi_rmse"], paths["noddi_rmse"])
    shutil.copy2(noddi_sources["noddi_nrmse"], paths["noddi_nrmse"])
    if signature != _expected_noddi_manifest(config):
        _write_json(paths["noddi_qc"], {"status": "failed", "automatic_qc_passed": False,
                                      "error": "NODDI inputs changed during fitting"})
        raise RuntimeError("NODDI inputs changed during fitting; results are not reusable")
    _write_noddi_qc(config, paths, source_maps=noddi_sources, validity=validity_qc)
    write_noddi_progress(
        "completed",
        "completed",
        nonzero_shells=nonzero_shells,
        outputs={
            "noddi_ndi": paths["noddi_ndi"].as_posix(),
            "noddi_odi": paths["noddi_odi"].as_posix(),
            "noddi_fwf": paths["noddi_fwf"].as_posix(),
            "noddi_tissue_valid_mask": paths["noddi_tissue_valid_mask"].as_posix(),
            "noddi_rmse": paths["noddi_rmse"].as_posix(),
            "noddi_nrmse": paths["noddi_nrmse"].as_posix(),
            "noddi_qc": paths["noddi_qc"].as_posix(),
        },
    )


def _roi_validity_mask(path: Path, reference_path: Path) -> np.ndarray:
    mask_img, reference = nib.load(str(path)), nib.load(str(reference_path))
    values = mask_img.get_fdata(dtype=np.float32)
    if mask_img.shape != reference.shape or not np.allclose(mask_img.affine, reference.affine, atol=1e-4):
        raise ValueError(f"Validity mask does not share the scalar's native grid: {path}")
    if values.ndim != 3 or not np.isfinite(values).all() or not np.isin(values, (0, 1)).all():
        raise ValueError(f"Validity mask must be finite, binary and 3D: {path}")
    return values.astype(bool)


def roi_statistics_current(
    json_path: Path, csv_path: Path, *, roi_label_map: Path | None = None,
    required_metrics: tuple[str, ...] = (),
    brain_mask_path: Path | None = None,
    selected_labels: tuple[int, ...] | None = None,
) -> bool:
    """Reject legacy, stale or inconsistent regional tables without refitting maps."""

    try:
        payload = _read_json(json_path) or {}
        if payload.get("schema_version") != ROI_STATISTICS_SCHEMA:
            return False
        if payload.get('selected_labels') != (sorted(set(selected_labels)) if selected_labels is not None else None):
            return False
        if roi_label_map is not None and payload.get("roi_label_map") != str(roi_label_map.resolve()):
            return False
        if brain_mask_path is not None and payload.get("brain_mask") != str(brain_mask_path.resolve()):
            return False
        sources = payload.get("source_signatures") or {}
        metrics = payload.get("feature_maps") or {}
        if not set(required_metrics).issubset(metrics):
            return False
        required_sources = {"atlas", *(f"metric:{key}" for key in metrics)}
        for key, policy in (payload.get("validity_policy") or {}).items():
            if policy == "explicit_validity_mask":
                required_sources.add(f"mask:{key}")
            else:
                # A zero-pair fallback cannot certify explicit tissue validity.
                return False
        if not metrics or not required_sources.issubset(sources):
            return False
        if sources['atlas']['path'] != payload.get('roi_label_map') or any(
            sources[f'metric:{key}']['path'] != path for key, path in metrics.items()
        ):
            return False
        if payload.get('brain_mask') and (sources.get('brain_mask') or {}).get('path') != payload['brain_mask']:
            return False
        if any(record != _file_signature(Path(record["path"])) for record in sources.values()):
            return False
        if payload.get("csv_sha256") != hashlib.sha256(csv_path.read_bytes()).hexdigest():
            return False
        with csv_path.open(encoding="utf-8", newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
        rows = payload["rows"]
        return len(rows) == len(csv_rows) == payload["roi_count"] and all(
            csv_row == {key: "" if value is None else str(value) for key, value in row.items()}
            for csv_row, row in zip(csv_rows, rows)
        )
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False


def _compute_roi_stats(
    *,
    metric_paths: dict[str, Path],
    roi_label_map: Path,
    lut_path: Path | None,
    csv_path: Path,
    json_path: Path,
    validity_masks: dict[str, np.ndarray] | None = None,
    validity_mask_paths: dict[str, Path] | None = None,
    brain_mask_path: Path | None = None,
    selected_labels: tuple[int, ...] | None = None,
    include_quantiles: bool = False,
) -> int:
    started = time.monotonic()
    if not metric_paths:
        raise ValueError("ROI statistics require at least one scalar map.")
    roi_img = nib.load(str(roi_label_map))
    roi_data = roi_img.get_fdata(dtype=np.float32)
    if roi_data.ndim != 3 or not np.isfinite(roi_data).all() or np.any(roi_data < 0):
        raise ValueError("ROI label map must be a finite non-negative 3D image.")
    brain_mask = _roi_validity_mask(brain_mask_path, roi_label_map) if brain_mask_path is not None else None
    metric_arrays: dict[str, np.ndarray] = {}
    metric_fieldnames: list[str] = []

    for metric_key, metric_path in metric_paths.items():
        metric_img = nib.load(str(metric_path))
        metric_data = metric_img.get_fdata(dtype=np.float32)
        if metric_data.shape != roi_data.shape:
            raise ValueError(
                f"ROI label map shape {roi_data.shape} does not match {metric_key} shape {metric_data.shape}."
            )
        if not np.allclose(metric_img.affine, roi_img.affine, atol=1e-4):
            raise ValueError(f"ROI label map affine does not match the {metric_key} image affine.")
        metric_arrays[metric_key] = metric_data
        metric_fieldnames.extend(
            [
                f"{metric_key}_mean",
                f"{metric_key}_std",
                f"{metric_key}_min",
                f"{metric_key}_max",
                f"{metric_key}_valid_voxel_count",
                f"{metric_key}_excluded_voxel_count",
                f"{metric_key}_valid_fraction",
            ]
        )
        if include_quantiles:
            metric_fieldnames.extend(f"{metric_key}_{suffix}" for suffix in ("median", "q25", "q75", "iqr"))

    metric_validity_masks: dict[str, np.ndarray] = dict(validity_masks or {})
    loaded_masks: dict[Path, np.ndarray] = {}
    for metric_key, mask_path in (validity_mask_paths or {}).items():
        if metric_key not in metric_arrays:
            raise ValueError(f"Validity mask supplied for unknown metric: {metric_key}.")
        mask_path = Path(mask_path).resolve()
        if mask_path not in loaded_masks:
            loaded_masks[mask_path] = _roi_validity_mask(mask_path, roi_label_map)
        valid_mask = loaded_masks[mask_path]
        if metric_key in metric_validity_masks and not np.array_equal(metric_validity_masks[metric_key], valid_mask):
            raise ValueError(f"Validity mask array and file disagree for {metric_key}.")
        metric_validity_masks[metric_key] = valid_mask
    for metric_key, valid_mask in metric_validity_masks.items():
        if metric_key not in metric_arrays:
            raise ValueError(f"Validity mask supplied for unknown metric: {metric_key}.")
        values = np.asarray(valid_mask)
        if values.shape != roi_data.shape:
            raise ValueError(f"Validity mask shape does not match ROI label map for {metric_key}.")
        if not np.isfinite(values).all() or not np.isin(values, (0, 1)).all():
            raise ValueError(f"Validity mask must be finite and binary for {metric_key}.")
        metric_validity_masks[metric_key] = values.astype(bool)
    mask_policies = {key: "explicit_validity_mask" for key in metric_validity_masks}
    if {"fw_tissue_fa", "fw_tissue_md"}.issubset(metric_arrays):
        tissue_fa = metric_arrays["fw_tissue_fa"]
        tissue_md = metric_arrays["fw_tissue_md"]
        finite_pair = np.isfinite(tissue_fa) & np.isfinite(tissue_md)
        for key in ("fw_tissue_fa", "fw_tissue_md"):
            if key in metric_validity_masks:
                metric_validity_masks[key] &= finite_pair
            else:
                metric_validity_masks[key] = finite_pair & ((tissue_fa != 0.0) | (tissue_md != 0.0))
                mask_policies[key] = "legacy_zero_pair_fallback_no_explicit_mask"

    rounded_roi = np.rint(roi_data)
    non_integer_mask = np.logical_and(roi_data > 0, np.abs(roi_data - rounded_roi) > 1e-3)
    if np.any(non_integer_mask):
        raise ValueError("ROI label map must contain discrete integer labels.")

    roi_labels = rounded_roi.astype(np.int64)
    # Stable groups retain the C-order voxel sequence used by boolean indexing,
    # without rescanning the entire volume for every region and metric.
    positive_indices = np.flatnonzero(roi_labels.ravel() > 0)
    positive_labels = roi_labels.flat[positive_indices]
    order = np.argsort(positive_labels, kind="stable")
    grouped_indices, grouped_labels = positive_indices[order], positive_labels[order]
    boundaries = np.r_[0, np.flatnonzero(np.diff(grouped_labels)) + 1, len(grouped_labels)]
    groups = [(int(grouped_labels[start]), grouped_indices[start:stop])
              for start, stop in zip(boundaries[:-1], boundaries[1:]) if start < stop]
    if selected_labels is not None:
        if not selected_labels or any(type(label) is not int or label <= 0 for label in selected_labels):
            raise ValueError('Select at least one positive integer ROI label')
        selected_labels = sorted(set(selected_labels))
        absent = set(selected_labels) - {label for label, _ in groups}
        if absent:
            raise ValueError('Selected ROI labels are absent: ' + ', '.join(map(str, sorted(absent))))
        groups = [(label, indices) for label, indices in groups if label in selected_labels]
    unique_labels = [label for label, _ in groups]
    label_names = _parse_label_names(unique_labels, lut_path)

    rows: list[dict[str, object]] = []
    for label, indices in groups:
        row: dict[str, object] = {
            "label_id": label,
            "label_name": label_names.get(label, ""),
        }
        row["roi_voxel_count"] = int(len(indices))
        voxel_count: int | None = None

        for metric_key, metric_data in metric_arrays.items():
            voxels = metric_data.flat[indices]
            metric_valid_mask = np.isfinite(voxels)
            if brain_mask is not None:
                metric_valid_mask &= brain_mask.flat[indices]
            if metric_key in metric_validity_masks:
                metric_valid_mask &= metric_validity_masks[metric_key].flat[indices]
            voxels = voxels[metric_valid_mask]
            if voxel_count is None:
                voxel_count = int(voxels.size)
            row[f"{metric_key}_valid_voxel_count"] = int(voxels.size)
            row[f"{metric_key}_excluded_voxel_count"] = row["roi_voxel_count"] - int(voxels.size)
            row[f"{metric_key}_valid_fraction"] = float(voxels.size / row["roi_voxel_count"])
            if voxels.size == 0:
                for suffix in ("mean", "std", "min", "max", *(("median", "q25", "q75", "iqr") if include_quantiles else ())):
                    row[f"{metric_key}_{suffix}"] = None
                continue
            # Selected-region exports use double reductions, including near-constant ROIs.
            reduction_dtype = np.float64 if selected_labels is not None else None
            row[f"{metric_key}_mean"] = float(np.mean(voxels, dtype=reduction_dtype))
            row[f"{metric_key}_std"] = float(np.std(voxels, dtype=reduction_dtype))
            row[f"{metric_key}_min"] = float(np.min(voxels))
            row[f"{metric_key}_max"] = float(np.max(voxels))
            if include_quantiles:
                q25, median, q75 = np.percentile(voxels.astype(np.float64), (25, 50, 75))
                row.update({f"{metric_key}_median": float(median), f"{metric_key}_q25": float(q25),
                            f"{metric_key}_q75": float(q75), f"{metric_key}_iqr": float(q75 - q25)})

        row["voxel_count"] = int(voxel_count or 0)
        rows.append(row)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label_id", "label_name", "voxel_count", "roi_voxel_count", *metric_fieldnames],
        )
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "schema_version": ROI_STATISTICS_SCHEMA,
        "subject_id": next(iter(metric_paths.values())).name.split("_")[0],
        "roi_label_map": str(roi_label_map.resolve()),
        "brain_mask": str(brain_mask_path.resolve()) if brain_mask_path is not None else None,
        "lut_path": str(lut_path.resolve()) if lut_path is not None else None,
        "feature_maps": {metric_key: str(metric_path.resolve()) for metric_key, metric_path in metric_paths.items()},
        "validity_policy": mask_policies,
        "source_signatures": {
            "atlas": _file_signature(roi_label_map),
            **({"brain_mask": _file_signature(brain_mask_path)} if brain_mask_path is not None else {}),
            **{f"metric:{key}": _file_signature(path) for key, path in metric_paths.items()},
            **{f"mask:{key}": _file_signature(path) for key, path in (validity_mask_paths or {}).items()},
            **({"lut": _file_signature(lut_path)} if lut_path is not None else {}),
        },
        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "voxel_count_basis": next(iter(metric_paths)),
        "statistics_policy": (
            "Per-metric finite valid samples only. Empty estimates are null/blank, never zero-filled; all requested atlas regions retained with coverage."
            if selected_labels is not None else
            "Per-metric finite valid samples only. Empty estimates are null/blank, never zero-filled; all atlas regions retained with coverage."
        ),
        "roi_count": len(rows),
        "selected_labels": selected_labels,
        "runtime_seconds": time.monotonic() - started,
        "timing_scope": "Read scalar maps/atlas and compute/write ROI statistics; excludes scalar model fitting",
        "rows": rows,
    }
    _write_json(json_path, payload)
    return len(rows)


def _run_independent_models(config, log):
    """Compute disjoint scalar products first; publish shared ROI tables once."""
    from dataclasses import replace
    import uuid
    from shared.compute_schedule import ComputeJob, available_resources, run_independent_jobs
    from microstructure.compute_worker import encode
    from shared.pipeline import resolve_pythonslicer_executable
    from shared.pipeline import slicer
    import sys

    cpu, _ = available_resources()
    import os
    cpu = min(cpu, int(os.environ.get('SLICERDSPACE_STAGE_CPUS', cpu)))
    stages = [stage for stage, enabled in (('freewater', config.run_free_water),
              ('noddi', config.run_noddi), ('dki', config.run_dki)) if enabled]
    threads = min(_noddi_threads(config), max(1, cpu - (len(stages)-1)))
    adjusted = replace(config, max_parallel_models=1,
                       noddi_nthreads=threads if config.noddi_nthreads is None else config.noddi_nthreads,
                       free_water_n_jobs=(config.free_water_n_jobs if config.free_water_engine == 'serial' else
                                          config.free_water_n_jobs or max(1, cpu // len(stages))))
    root = config.output_dir / 'compute_schedules' / uuid.uuid4().hex
    root.parent.mkdir(parents=True, exist_ok=True)
    description = root.with_suffix('.json')
    description.write_text(json.dumps(encode(adjusted), indent=2)+'\n', encoding='utf-8')
    image = nib.load(str(config.raw_input_bundle.dwi_path))
    # Conservative reservation for a full loaded DWI, parameter maps and kernels.
    reserve = 2 * 1024**3 + int(np.prod(image.shape[:3])) * (image.shape[3]*8 + 1200)
    jobs = []
    python = resolve_pythonslicer_executable() if getattr(slicer, 'app', None) is not None else sys.executable
    for stage in stages:
        if not adjusted.force_recompute:
            paths = _output_paths(adjusted.subject_id, adjusted.output_dir)
            if stage == 'freewater' and _free_water_outputs_current(adjusted, paths):
                continue
            if stage == 'noddi' and _noddi_outputs_current(adjusted, paths):
                continue
            if stage == 'dki':
                from microstructure.dki import DkiConfig, dki_outputs_current
                if dki_outputs_current(DkiConfig(adjusted.subject_id,adjusted.raw_input_bundle,
                    adjusted.output_dir/'dki',fit_method=adjusted.dki_fit_method,
                    matlab_executable=adjusted.dki_matlab_executable,mkcurve_root=adjusted.dki_mkcurve_root,
                    **(adjusted.dki_execution or {}))):
                    continue
        count = (_noddi_threads(adjusted) if stage == 'noddi' else
                 (1 if adjusted.free_water_engine == 'serial' else adjusted.free_water_n_jobs)
                 if stage == 'freewater' else
                 ((adjusted.dki_execution or {}).get('mkcurve_workers', 1) if adjusted.dki_fit_method == 'MKCURVE' else 1)
                 * ((adjusted.dki_execution or {}).get('num_threads') or 1))
        jobs.append(ComputeJob(stage, (python, str(Path(__file__).with_name('compute_worker.py')),
            '--stage', stage, '--config', str(description)), threads=count, memory_bytes=reserve,
            environment={'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}))
    schedule = run_independent_jobs(jobs, directory=root,
                                    max_workers=config.max_parallel_models, cpu_budget=cpu, log=log)
    return replace(adjusted, force_recompute=False), schedule


def run_microstructure(
    config: MicrostructureConfig,
    log: Optional[Callable[[str], None]] = None,
) -> MicrostructureResult:
    """Export native DTI maps, Free Water maps, and ROI-wise statistics."""

    subject_id = config.subject_id.strip()
    if not subject_id:
        raise ValueError("Subject ID must be provided.")
    if subject_id != config.raw_input_bundle.subject_id:
        raise ValueError("Subject ID does not match the provided raw input bundle.")
    if subject_id != config.native_bundle.subject_id:
        raise ValueError("Subject ID does not match the provided native DTI bundle.")
    if config.raw_input_bundle.subject_root != config.native_bundle.subject_root:
        raise ValueError("Raw input bundle and native DTI bundle must belong to the same subject root.")
    if config.free_water_chunk_size <= 0:
        raise ValueError("free_water_chunk_size must be > 0.")
    if (config.free_water_engine or "serial").lower() not in FREE_WATER_ENGINES:
        raise ValueError("free_water_engine must be one of: serial, joblib, dask, ray.")
    if config.free_water_vox_per_chunk is not None and config.free_water_vox_per_chunk <= 0:
        raise ValueError("free_water_vox_per_chunk must be > 0 when provided.")
    if not 0.0 < float(config.free_water_max_fraction_for_tissue) < 1.0:
        raise ValueError("free_water_max_fraction_for_tissue must be between 0 and 1.")
    if not 0.0 < float(config.noddi_max_isotropic_fraction_for_tissue) < 1.0:
        raise ValueError("noddi_max_isotropic_fraction_for_tissue must be between 0 and 1.")
    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.max_parallel_models not in (1, 2, 3):
        raise ValueError('max_parallel_models must be 1, 2 or 3')
    schedule = None
    if config.max_parallel_models > 1 and sum((config.run_free_water, config.run_noddi, config.run_dki)) > 1:
        # DTI/cache preparation is complete before this entry; worker outputs are
        # independent. No concurrent writes to manifest.json or ROI statistics.
        config, schedule = _run_independent_models(config, log)

    paths = _output_paths(subject_id, output_dir)
    dki_result = None
    if config.run_dki:
        from microstructure.dki import DkiConfig, run_dki

        dki_result = run_dki(DkiConfig(
            subject_id=subject_id, raw_input_bundle=config.raw_input_bundle,
            output_dir=output_dir / "dki", fit_method=config.dki_fit_method,
            matlab_executable=config.dki_matlab_executable, mkcurve_root=config.dki_mkcurve_root,
            force_recompute=config.force_recompute,
            **(config.dki_execution or {}),
        ), log=log)
    expected_free_water_manifest = _expected_free_water_manifest(config)
    if (
        bool(config.run_free_water) and not config.force_recompute
        and _read_json(paths["free_water_manifest"]) == expected_free_water_manifest
        and _free_water_maps_complete(paths)
        and not _free_water_outputs_complete(paths)
    ):
        _emit("[Microstructure] Backfilling scientific QC for reusable Free Water outputs", log)
        _write_free_water_qc(config, paths)
    if (
        bool(config.run_noddi) and not config.force_recompute
        and _read_json(paths["noddi_manifest"]) == _expected_noddi_manifest(config)
        and all(paths[key].is_file() for key in NODDI_KEYS)
        and not _noddi_outputs_complete(paths)
    ):
        _emit("[Microstructure] Revalidating saved NODDI maps and residuals before reuse", log)
        previous = _read_json(paths["noddi_qc"]) or {}
        _write_noddi_qc(config, paths, source_maps=previous.get("source_maps"), validity=previous.get("validity"))
    roi_label_map = _resolve_roi_label_map(config)
    free_water_available = bool(config.run_free_water) or _free_water_outputs_current(config, paths)
    noddi_available = bool(config.run_noddi) or _noddi_outputs_current(config, paths)
    expected_manifest = _expected_manifest(
        config,
        roi_label_map,
        free_water_available=free_water_available,
        noddi_available=noddi_available,
    )
    actual_manifest = _read_json(paths["manifest"])
    expected_manifest["dki_model"] = {
        "run_enabled": bool(config.run_dki),
        "manifest": _read_json(dki_result.manifest_path) if dki_result is not None else None,
    }
    roi_enabled = roi_label_map is not None
    force_recompute = bool(config.force_recompute)

    if (
        not force_recompute
        and actual_manifest == expected_manifest
        and _dti_outputs_current(config, paths)
        and (not config.run_free_water or _free_water_outputs_current(config, paths))
        and (not config.run_noddi or _noddi_outputs_current(config, paths))
        and _outputs_complete(
            paths,
            roi_enabled=roi_enabled,
            free_water_enabled=bool(config.run_free_water),
            noddi_enabled=bool(config.run_noddi),
            brain_mask_path=config.raw_input_bundle.mask_path,
        )
    ):
        _emit(f"[Microstructure] Reusing outputs in {output_dir}", log)
        return MicrostructureResult(
            subject_id=subject_id,
            output_dir=output_dir,
            fa_path=paths["fa"],
            md_path=paths["md"],
            ad_path=paths["ad"],
            rd_path=paths["rd"],
            l1_path=paths["l1"],
            l2_path=paths["l2"],
            l3_path=paths["l3"],
            trace_path=paths["trace"],
            fwf_path=paths["fwf"],
            fw_tissue_fa_path=paths["fw_tissue_fa"],
            fw_tissue_md_path=paths["fw_tissue_md"],
            fw_tissue_valid_mask_path=paths["fw_tissue_valid_mask"],
            free_water_qc_path=paths["free_water_qc"],
            noddi_ndi_path=paths["noddi_ndi"],
            noddi_odi_path=paths["noddi_odi"],
            noddi_fwf_path=paths["noddi_fwf"],
            noddi_tissue_valid_mask_path=paths["noddi_tissue_valid_mask"],
            noddi_rmse_path=paths["noddi_rmse"],
            noddi_nrmse_path=paths["noddi_nrmse"],
            noddi_qc_path=paths["noddi_qc"],
            roi_label_map=roi_label_map,
            roi_stats_csv_path=paths["roi_stats_csv"] if roi_enabled else None,
            roi_stats_json_path=paths["roi_stats_json"] if roi_enabled else None,
            dki_result=dki_result,
            compute_schedule=schedule,
            dec_path=paths["dec"],
        )

    action = 'Reusing verified' if _dti_outputs_current(config, paths) else 'Exporting'
    _emit(f"[Microstructure] {action} DTI scalar maps from native DTI bundle ({config.native_bundle.fit_method})", log)
    _export_dti_scalar_maps(config, paths)
    _remove_legacy_roi_outputs(output_dir, subject_id)

    actual_free_water_manifest = _read_json(paths["free_water_manifest"])
    if bool(config.run_free_water):
        if (
            not force_recompute
            and actual_free_water_manifest == expected_free_water_manifest
            and _free_water_outputs_complete(paths)
        ):
            _emit("[Microstructure] Reusing Free Water stage outputs", log)
        else:
            _ensure_base_imports_ready()
            _ensure_dipy_ready()
            _emit("[Microstructure] Running Free Water stage", log)
            try:
                _compute_free_water_maps(config=config, paths=paths, log=log)
            except Exception as exc:
                _write_progress(
                    paths["free_water_progress"],
                    {
                        "kind": "microstructure_stage_progress",
                        "version": 1,
                        "subject_id": subject_id,
                        "stage": "freewater",
                        "status": "failed",
                        "error": str(exc),
                    },
                )
                raise
            _write_json(paths["free_water_manifest"], expected_free_water_manifest)
    else:
        _emit("[Microstructure] Free Water stage disabled; DTI-only outputs will not include Free Water maps", log)

    expected_noddi_manifest = _expected_noddi_manifest(config)
    actual_noddi_manifest = _read_json(paths["noddi_manifest"])
    if bool(config.run_noddi):
        if (
            not force_recompute
            and actual_noddi_manifest == expected_noddi_manifest
            and _noddi_outputs_complete(paths)
        ):
            _emit("[Microstructure] Reusing NODDI stage outputs", log)
        else:
            _ensure_base_imports_ready()
            _ensure_amico_ready()
            _emit("[Microstructure] Running NODDI stage", log)
            try:
                _compute_noddi_maps(config=config, paths=paths, log=log)
            except Exception as exc:
                _write_progress(
                    paths["noddi_progress"],
                    {
                        "kind": "microstructure_stage_progress",
                        "version": 1,
                        "subject_id": subject_id,
                        "stage": "noddi",
                        "status": "failed",
                        "error": str(exc),
                    },
                )
                raise
            _write_json(paths["noddi_manifest"], expected_noddi_manifest)
    else:
        _emit("[Microstructure] NODDI stage disabled; DTI-only outputs will not include NODDI maps", log)

    if roi_label_map is not None:
        _ensure_base_imports_ready()
        lut_path = _resolve_lut_path(config)
        metric_paths = {
            "fa": paths["fa"],
            "md": paths["md"],
            "ad": paths["ad"],
            "rd": paths["rd"],
            "l1": paths["l1"],
            "l2": paths["l2"],
            "l3": paths["l3"],
            "trace": paths["trace"],
        }
        if _free_water_outputs_current(config, paths):
            metric_paths.update(
                {
                    "fwf": paths["fwf"],
                    "fw_tissue_fa": paths["fw_tissue_fa"],
                    "fw_tissue_md": paths["fw_tissue_md"],
                }
            )
        if _noddi_outputs_current(config, paths):
            metric_paths.update(
                {
                    "noddi_ndi": paths["noddi_ndi"],
                    "noddi_odi": paths["noddi_odi"],
                    "noddi_fwf": paths["noddi_fwf"],
                }
            )
        validity_mask_paths: dict[str, Path] = {}
        if _free_water_outputs_current(config, paths):
            validity_mask_paths.update({key: paths["fw_tissue_valid_mask"] for key in ("fw_tissue_fa", "fw_tissue_md")})
        if dki_result is not None:
            from microstructure.dki import DKI_METRICS

            for key in DKI_METRICS:
                metric_paths[f"dki_{key}"] = dki_result.outputs[f"{key}_raw"]
                validity_mask_paths[f"dki_{key}"] = dki_result.valid_mask_path
            corrected_mask = dki_result.mkcurve_outputs.get("corrected_valid_mask")
            if corrected_mask is not None:
                for key in ("fa", "md", "ad", "rd", "mk", "ak", "rk"):
                    corrected = dki_result.mkcurve_outputs.get(f"corrected_{key}")
                    if corrected is not None:
                        metric_paths[f"mkcurve_corrected_{key}"] = corrected
                        validity_mask_paths[f"mkcurve_corrected_{key}"] = dki_result.mkcurve_outputs.get(
                            f"corrected_{key}_valid_mask", corrected_mask)
        if _noddi_outputs_current(config, paths):
            validity_mask_paths.update({key: paths["noddi_tissue_valid_mask"] for key in ("noddi_ndi", "noddi_odi")})
        _emit(f"[Microstructure] Computing ROI-wise statistics from {roi_label_map}", log)
        roi_count = _compute_roi_stats(
            metric_paths=metric_paths,
            roi_label_map=roi_label_map,
            lut_path=lut_path,
            csv_path=paths["roi_stats_csv"],
            json_path=paths["roi_stats_json"],
            validity_mask_paths=validity_mask_paths,
            brain_mask_path=config.raw_input_bundle.mask_path,
        )
        _emit(
            f"[Microstructure] ROI-wise statistics written for {roi_count} labels",
            log,
        )
    else:
        _emit(
            "[Microstructure] No ROI label map supplied. Wrote the selected model maps; regional statistics were skipped.",
            log,
        )
        paths["roi_stats_csv"].unlink(missing_ok=True)
        paths["roi_stats_json"].unlink(missing_ok=True)

    _write_json(paths["manifest"], expected_manifest)
    return MicrostructureResult(
        subject_id=subject_id,
        output_dir=output_dir,
        fa_path=paths["fa"],
        md_path=paths["md"],
        ad_path=paths["ad"],
        rd_path=paths["rd"],
        l1_path=paths["l1"],
        l2_path=paths["l2"],
        l3_path=paths["l3"],
        trace_path=paths["trace"],
        fwf_path=paths["fwf"],
        fw_tissue_fa_path=paths["fw_tissue_fa"],
        fw_tissue_md_path=paths["fw_tissue_md"],
        fw_tissue_valid_mask_path=paths["fw_tissue_valid_mask"],
        free_water_qc_path=paths["free_water_qc"],
        noddi_ndi_path=paths["noddi_ndi"],
        noddi_odi_path=paths["noddi_odi"],
        noddi_fwf_path=paths["noddi_fwf"],
        noddi_tissue_valid_mask_path=paths["noddi_tissue_valid_mask"],
        noddi_rmse_path=paths["noddi_rmse"],
        noddi_nrmse_path=paths["noddi_nrmse"],
        noddi_qc_path=paths["noddi_qc"],
        roi_label_map=roi_label_map,
        roi_stats_csv_path=paths["roi_stats_csv"] if roi_label_map is not None else None,
        roi_stats_json_path=paths["roi_stats_json"] if roi_label_map is not None else None,
        dki_result=dki_result,
        compute_schedule=schedule,
        dec_path=paths["dec"],
    )
