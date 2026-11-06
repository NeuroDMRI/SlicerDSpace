#!/usr/bin/env python3
"""Audit multi-subject anatomical, DTI, and tractography outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DTI_KEYS = ("fa", "md", "ad", "rd", "l1", "l2", "l3", "trace")
FULL_ROUTE_NAMES = {"freewater", "noddi", "saf", "wmql", "wma", "tractcloud", "connectome"}
TRACTOGRAM_LENGTH_ABS_TOL_MM = 1e-3


def _manifest_for_run(subject_output: Path, run_name: str | None = None) -> Path | None:
    metadata_dir = subject_output / "00_run_metadata"
    if run_name:
        candidate = metadata_dir / f"{run_name}_orchestration_manifest.json"
        return candidate if candidate.is_file() else None
    candidates = list(metadata_dir.glob("*_orchestration_manifest.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def _record_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    value: Any,
    requirement: str,
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "value": value,
            "requirement": requirement,
        }
    )


def _task_outputs(manifest: dict[str, Any], task_name: str) -> dict[str, Any]:
    for task in manifest.get("tasks", []):
        if task.get("name") == task_name and task.get("status") in {"completed", "skipped"}:
            return dict(task.get("outputs") or {})
    return {}


def _task_entry(manifest: dict[str, Any], task_name: str) -> dict[str, Any] | None:
    for task in manifest.get("tasks", []):
        if task.get("name") == task_name:
            return dict(task)
    return None


def _analysis_input_path(manifest: dict[str, Any], key: str) -> Path:
    """Resolve the standardized preprocessing output when the run selected it."""

    preprocessed = _task_outputs(manifest, "preprocessing.dwi").get(key)
    if preprocessed:
        candidate = Path(str(preprocessed))
        if candidate.is_file():
            return candidate
    names = {
        "dwi": "data.nii.gz",
        "bval": "bvals",
        "bvec": "bvecs",
        "mask": "nodif_brain_mask.nii.gz",
    }
    recorded = (manifest.get("analysis_inputs") or {}).get(key)
    if recorded:
        return Path(str(recorded))
    return Path(str(manifest["subject_root"])) / "input" / names[key]


def _audit_orchestration(manifest: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    """Reject incomplete runs before interpreting their scientific products."""

    status = manifest.get("status")
    if status is not None:
        _record_check(
            checks,
            "orchestration.status",
            status == "completed",
            status,
            "completed",
        )
    task_statuses = {
        str(task.get("name") or f"task_{index}"): task.get("status")
        for index, task in enumerate(manifest.get("tasks") or [])
    }
    if task_statuses:
        successful = {"completed", "skipped", "passed"}
        _record_check(
            checks,
            "orchestration.task_statuses",
            all(status in successful for status in task_statuses.values()),
            task_statuses,
            "every selected task completed, passed, or reused a completed output",
        )


def _audit_validation(manifest: dict[str, Any], checks: list[dict[str, Any]]) -> bool:
    """Audit the native input-validation report when the route was selected."""

    task = _task_entry(manifest, "validation")
    if task is None:
        return False
    task_status = task.get("status")
    _record_check(
        checks,
        "validation.task_status",
        task_status in {"completed", "skipped", "passed"},
        task_status,
        "completed, passed, or skipped/reused",
    )
    outputs = dict(task.get("outputs") or {})
    report_path, report = _json_record(
        task.get("qc_path")
        or task.get("manifest_path")
        or outputs.get("report")
    )
    _record_check(
        checks,
        "validation.report_exists",
        report is not None,
        report_path.as_posix() if report_path else None,
        "readable input-validation JSON exists",
    )
    if report is None:
        return True
    error_count = int(report.get("error_count", 0) or 0)
    passed = report.get("passed") is True and report.get("status") == "passed"
    _record_check(
        checks,
        "validation.report_status",
        passed and error_count == 0,
        {
            "status": report.get("status"),
            "passed": report.get("passed"),
            "error_count": error_count,
            "warning_count": int(report.get("warning_count", 0) or 0),
        },
        "passed=true, status=passed, and error_count=0",
    )
    files = dict(((report.get("metrics") or {}).get("files") or {}))
    required_files = {
        key: bool((files.get(key) or {}).get("exists") and (files.get(key) or {}).get("is_file"))
        for key in ("dwi", "bval", "bvec")
    }
    if required_files:
        _record_check(
            checks,
            "validation.required_files",
            all(required_files.values()),
            required_files,
            "DWI, b-values, and b-vectors exist as files",
        )
    gradients = dict(((report.get("metrics") or {}).get("gradients") or {}))
    if gradients:
        _record_check(
            checks,
            "validation.gradients",
            gradients.get("valid") is True,
            gradients,
            "gradient table is valid",
        )
    return True


def _audit_preprocessing(manifest: dict[str, Any], checks: list[dict[str, Any]]) -> bool:
    """Validate the standardized DWI bundle and preprocessing disclosure."""

    import nibabel as nib
    import numpy as np

    task = _task_entry(manifest, "preprocessing.dwi")
    if task is None:
        return False
    outputs = dict(task.get("outputs") or {})
    manifest_path, native_manifest = _json_record(task.get("manifest_path"))
    qc_path, qc = _json_record(task.get("qc_path"))
    _record_check(
        checks,
        "preprocessing.task_status",
        task.get("status") in {"completed", "skipped"},
        task.get("status"),
        "completed or skipped/reused",
    )
    _record_check(
        checks,
        "preprocessing.manifest_cache_key",
        bool(native_manifest and native_manifest.get("cache_key")),
        manifest_path.as_posix() if manifest_path else None,
        "versioned input-and-parameter cache key is recorded",
    )
    _record_check(
        checks,
        "preprocessing.qc_status",
        bool(qc and qc.get("status") == "completed"),
        qc.get("status") if qc else None,
        "completed",
    )

    paths = {
        key: Path(str(outputs.get(key) or ""))
        for key in ("dwi", "bval", "bvec", "mask")
    }
    for key, path in paths.items():
        _record_check(checks, f"preprocessing.{key}.exists", path.is_file(), path.as_posix(), "file exists")
    if not all(path.is_file() for path in paths.values()):
        return True

    dwi_image = nib.load(paths["dwi"].as_posix())
    mask_image = nib.load(paths["mask"].as_posix())
    dwi_shape = dwi_image.shape
    mask = np.asarray(mask_image.dataobj)
    bvals = np.asarray(np.loadtxt(paths["bval"], ndmin=1), dtype=np.float64).reshape(-1)
    bvecs = np.asarray(np.loadtxt(paths["bvec"]), dtype=np.float64)
    if bvecs.ndim == 2 and bvecs.shape[0] == 3:
        bvecs = bvecs.T
    volume_count = int(dwi_shape[3]) if len(dwi_shape) == 4 else 0
    _record_check(checks, "preprocessing.dwi_4d", len(dwi_shape) == 4, list(dwi_shape), "4D DWI")
    _record_check(
        checks,
        "preprocessing.gradient_count",
        volume_count > 0 and bvals.size == volume_count and bvecs.shape == (volume_count, 3),
        {"volumes": volume_count, "bvals": int(bvals.size), "bvec_shape": list(bvecs.shape)},
        "one finite b-value and 3-vector per DWI volume",
    )
    finite_gradients = bool(np.isfinite(bvals).all() and np.isfinite(bvecs).all())
    _record_check(checks, "preprocessing.gradients_finite", finite_gradients, None, "all gradients finite")
    _record_check(
        checks,
        "preprocessing.bvals_nonnegative",
        finite_gradients and bool(np.all(bvals >= 0)),
        float(bvals.min()) if bvals.size and np.isfinite(bvals).all() else None,
        "b-values are finite and non-negative",
    )
    if finite_gradients and bvecs.shape == (volume_count, 3):
        nonzero = bvals > 50.0
        norms = np.linalg.norm(bvecs[nonzero], axis=1)
        normalized_fraction = float(np.mean(np.isclose(norms, 1.0, atol=0.05))) if norms.size else 0.0
        _record_check(
            checks,
            "preprocessing.bvec_norms",
            normalized_fraction >= 0.99,
            normalized_fraction,
            ">= 0.99 of diffusion-weighted b-vectors have norm 1 +/- 0.05",
        )
    mask_valid = bool(
        mask.shape == dwi_shape[:3]
        and np.isfinite(mask).all()
        and np.all(np.isin(mask, (0, 1)))
        and np.count_nonzero(mask) > 0
        and np.allclose(mask_image.affine, dwi_image.affine, atol=1e-4)
    )
    _record_check(
        checks,
        "preprocessing.mask_contract",
        mask_valid,
        {"mask_shape": list(mask.shape), "dwi_shape": list(dwi_shape[:3]), "voxels": int(np.count_nonzero(mask))},
        "binary non-empty mask on the DWI grid",
    )
    affine_valid = bool(
        np.isfinite(dwi_image.affine).all()
        and abs(np.linalg.det(dwi_image.affine[:3, :3])) > 1e-12
    )
    _record_check(checks, "preprocessing.affine", affine_valid,
                  dwi_image.affine.tolist(), "finite, invertible spatial affine")
    # Read one volume at a time; large diffusion series need not fit in RAM.
    nonfinite_count = 0
    brain_nonzero_count = 0
    for volume in range(volume_count):
        values = np.asarray(dwi_image.dataobj[..., volume])
        finite = np.isfinite(values)
        nonfinite_count += int(np.count_nonzero(~finite))
        if mask_valid:
            brain_nonzero_count += int(np.count_nonzero((mask != 0) & finite & (values != 0)))
    _record_check(
        checks, "preprocessing.dwi_finite", volume_count > 0 and nonfinite_count == 0,
        {"nonfinite_voxels": nonfinite_count, "volumes_scanned": volume_count},
        "all output DWI voxels are finite",
    )
    _record_check(
        checks, "preprocessing.brain_signal", brain_nonzero_count > 0,
        {"nonzero_finite_brain_values": brain_nonzero_count},
        "the output series contains nonzero finite signal within the verified mask",
    )
    if qc:
        details = dict(qc.get("backend_details") or {})
        omitted = any(
            str((details.get(key) or {}).get("status") or "")
            in {"not_available", "not_run", "not_supported", "not_supported_by_dipy_backend", "skipped"}
            for key in ("denoise", "gibbs", "motion", "bias_correction", "susceptibility_distortion_correction")
        )
        warnings = list(qc.get("warnings") or [])
        _record_check(
            checks,
            "preprocessing.omitted_steps_disclosed",
            not omitted or bool(warnings),
            warnings,
            "omitted corrections are explicitly disclosed in QC warnings",
        )
    return True


def _audit_dti(manifest: dict[str, Any], checks: list[dict[str, Any]]) -> bool:
    import nibabel as nib
    import numpy as np

    outputs = _task_outputs(manifest, "microstructure.dti")
    if not outputs:
        return False

    mask_path = _analysis_input_path(manifest, "mask")
    _record_check(checks, "dti.mask_exists", mask_path.is_file(), mask_path.as_posix(), "file exists")
    if not mask_path.is_file():
        return True

    mask_image = nib.load(mask_path.as_posix())
    mask = np.asarray(mask_image.dataobj) > 0
    maps: dict[str, Any] = {}
    for key in DTI_KEYS:
        path_value = outputs.get(key)
        path = Path(str(path_value)) if path_value else None
        exists = bool(path and path.is_file())
        _record_check(checks, f"dti.{key}.exists", exists, path.as_posix() if path else None, "file exists")
        if not exists or path is None:
            continue
        image = nib.load(path.as_posix())
        array = np.asarray(image.dataobj, dtype=np.float64)
        maps[key] = array
        _record_check(checks, f"dti.{key}.shape", array.shape == mask.shape, list(array.shape), list(mask.shape))
        _record_check(
            checks,
            f"dti.{key}.affine",
            bool(np.allclose(image.affine, mask_image.affine, atol=1e-4)),
            float(np.max(np.abs(image.affine - mask_image.affine))),
            "maximum absolute difference <= 1e-4",
        )
        _record_check(checks, f"dti.{key}.finite", bool(np.isfinite(array).all()), None, "all values finite")

    if any(key not in maps for key in DTI_KEYS) or not np.any(mask):
        return True

    fa = maps["fa"][mask]
    _record_check(
        checks,
        "dti.fa.range",
        bool(np.min(fa) >= -1e-6 and np.max(fa) <= 1.0 + 1e-6),
        {"min": float(np.min(fa)), "max": float(np.max(fa))},
        "0 <= FA <= 1",
    )
    for key in ("md", "ad", "rd", "l1", "l2", "l3"):
        values = maps[key][mask]
        _record_check(
            checks,
            f"dti.{key}.nonnegative",
            bool(np.min(values) >= -1e-8),
            float(np.min(values)),
            "minimum >= -1e-8 mm^2/s",
        )

    l1, l2, l3 = (maps[key][mask] for key in ("l1", "l2", "l3"))
    md, ad, rd, trace = (maps[key][mask] for key in ("md", "ad", "rd", "trace"))
    eigen_order_fraction = float(np.mean((l1 >= l2 - 1e-8) & (l2 >= l3 - 1e-8)))
    _record_check(checks, "dti.eigenvalue_order", eigen_order_fraction >= 0.999, eigen_order_fraction, ">= 0.999")

    identities = {
        "ad_equals_l1": np.abs(ad - l1),
        "rd_equals_l2_l3_mean": np.abs(rd - (l2 + l3) / 2.0),
        "md_equals_eigenvalue_mean": np.abs(md - (l1 + l2 + l3) / 3.0),
    }
    for name, error in identities.items():
        max_error = float(np.max(error))
        _record_check(checks, f"dti.{name}", max_error <= 1e-6, max_error, "maximum absolute error <= 1e-6")

    # SlicerDMRI computes Trace from the unconstrained tensor while its exported
    # eigenvalue maps clamp negative eigenvalues. Compare the unaffected voxels
    # by coverage instead of requiring an invalid exact identity everywhere.
    trace_close_fraction = float(np.mean(np.isclose(trace, 3.0 * md, rtol=1e-4, atol=1e-8)))
    _record_check(
        checks,
        "dti.trace_md_consistency",
        trace_close_fraction >= 0.95,
        trace_close_fraction,
        ">= 0.95 of masked voxels satisfy Trace ~= 3*MD",
    )
    return True


def _audit_regional_statistics(manifest: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    outputs = _task_outputs(manifest, 'microstructure.dti')
    if not (outputs.get('roi_stats_json') or outputs.get('roi_stats_csv')):
        return
    from scripts.audit_regional_validity import audit_statistics

    try:
        result = audit_statistics(Path(str(outputs.get('roi_stats_json') or '')),
                                  Path(str(outputs.get('roi_stats_csv') or '')), required_metrics=DTI_KEYS)
        value = {key: result.get(key) for key in ('status', 'checks', 'regions', 'metrics', 'comparisons', 'failures')}
        passed = result['automatic_qc_passed'] is True
    except Exception as exc:
        value, passed = {'error': f'{type(exc).__name__}: {exc}'}, False
    _record_check(checks, 'microstructure.regional_statistics', passed, value,
                  'Current source-linked tables and independently recomputed native-grid counts/estimates; invalid samples excluded')


def _audit_act_preparation(manifest: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    from shared.orchestrator import _act_inputs_current

    task = _task_entry(manifest, "tractography.csd_act")
    if not task:
        return
    path = Path(str(task.get("manifest_path") or ""))
    _record_check(
        checks, "tractography.csd_act.act_preparation_current",
        path.is_file() and _act_inputs_current(path), str(path),
        "unchanged ACT inputs with current tissue-mapping version; pre-v3 VentralDC sGM is not accepted",
    )


def _audit_tractography(manifest: dict[str, Any], checks: list[dict[str, Any]]) -> bool:
    import nibabel as nib
    import numpy as np

    task_name = "tractography.dti_det"
    outputs = _task_outputs(manifest, task_name)
    if not outputs:
        task_name = "tractography.csd_act"
        outputs = _task_outputs(manifest, task_name)
    if not outputs:
        task_name = "tractography.ukf"
        outputs = _task_outputs(manifest, task_name)
    path_value = outputs.get("tractogram_trk") or outputs.get("tractogram")
    if not path_value:
        return False
    tractogram_path = Path(str(path_value))
    _record_check(checks, "tractography.exists", tractogram_path.is_file(), tractogram_path.as_posix(), "file exists")
    if not tractogram_path.is_file():
        return True

    tractogram = nib.streamlines.load(tractogram_path.as_posix())
    streamlines = list(tractogram.streamlines)
    _record_check(checks, "tractography.streamline_count", len(streamlines) > 0, len(streamlines), "> 0")
    if not streamlines:
        return True

    points = np.concatenate(streamlines, axis=0)
    lengths = np.asarray(
        [np.linalg.norm(np.diff(streamline, axis=0), axis=1).sum() for streamline in streamlines],
        dtype=np.float64,
    )
    _record_check(checks, "tractography.points_finite", bool(np.isfinite(points).all()), None, "all points finite")
    _record_check(
        checks,
        "tractography.lengths_positive",
        bool(np.isfinite(lengths).all() and np.min(lengths) > 0),
        {
            "min_mm": float(np.min(lengths)),
            "median_mm": float(np.median(lengths)),
            "max_mm": float(np.max(lengths)),
        },
        "all streamline lengths are finite and > 0 mm",
    )

    task = _task_entry(manifest, task_name) or {}
    if task_name == 'tractography.ukf':
        from shared.pipeline import RawInputBundle
        from tractography.ukf import current

        try:
            native_path = Path(str(task['manifest_path']))
            native = json.loads(native_path.read_text())
            signature = native['signature']
            raw_paths = {key: _analysis_input_path(manifest, key) for key in ('dwi', 'bval', 'bvec', 'mask')}
            root = Path(str(manifest['subject_root']))
            raw = RawInputBundle(root, manifest['subject_id'], raw_paths['dwi'].parent,
                                 raw_paths['dwi'], raw_paths['bval'], raw_paths['bvec'], raw_paths['mask'])
            intact = current(raw, native_path.parent, signature['parameters'],
                             Path(signature['executable']['path']), Path(signature['converter']['path']))
            detail = str(native_path)
        except Exception as exc:
            intact, detail = False, str(exc)
        _record_check(checks, 'tractography.ukf.current_artifacts', intact, detail,
                      'Current input/backend identity and intact tractogram, effective mask and QC artifacts')
    tracking_manifest_value = task.get("manifest_path")
    tracking_manifest_path = Path(str(tracking_manifest_value)) if tracking_manifest_value else None
    if tracking_manifest_path and tracking_manifest_path.is_file():
        tracking_manifest = json.loads(tracking_manifest_path.read_text(encoding="utf-8"))
        parameters = tracking_manifest.get("parameters") or {}
        configured_min = parameters.get("min_length_mm")
        configured_max = parameters.get("max_length_mm")
        if task_name == "tractography.csd_act":
            _record_check(
                checks,
                "tractography.csd_act.length_limits_recorded",
                configured_min is not None and configured_max is not None,
                {"min_length_mm": configured_min, "max_length_mm": configured_max},
                "explicit minimum and maximum streamline lengths are recorded",
            )
        if configured_min is not None and configured_max is not None:
            within_configured_range = bool(
                np.min(lengths) + 1e-3 >= float(configured_min)
                and np.max(lengths) - 1e-3 <= float(configured_max)
            )
            _record_check(
                checks,
                "tractography.configured_length_range",
                within_configured_range,
                {
                    "configured_min_mm": float(configured_min),
                    "configured_max_mm": float(configured_max),
                    "observed_min_mm": float(np.min(lengths)),
                    "observed_max_mm": float(np.max(lengths)),
                },
                "all saved streamlines satisfy the configured millimetre range",
            )

    qc_value = task.get("qc_path")
    qc_path = Path(str(qc_value)) if qc_value else None
    if qc_path and qc_path.is_file():
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        if task_name == 'tractography.ukf':
            _record_check(checks, 'tractography.ukf.automatic_qc', qc.get('automatic_qc_passed') is True,
                          qc.get('checks'), 'UKF tensor/FA arrays and native-mask geometry QC passed')
        if task_name == "tractography.csd_act":
            scientific_summary = qc.get("scientific_output_summary")
            _record_check(
                checks,
                "tractography.csd_act.scientific_output_summary",
                isinstance(scientific_summary, dict) and bool(scientific_summary),
                scientific_summary,
                "decoded TCK/TRK/SIFT2 evidence is recorded",
            )
        qc_count = qc.get("streamline_count")
        if qc_count is not None:
            _record_check(
                checks,
                "tractography.qc_streamline_count",
                int(qc_count) == len(streamlines),
                {"qc": int(qc_count), "file": len(streamlines)},
                "QC and saved tractogram streamline counts match",
            )
        qc_min = qc.get("length_min_mm")
        qc_max = qc.get("length_max_mm")
        if qc_min is not None and qc_max is not None:
            summary_matches = bool(
                math.isclose(
                    float(qc_min),
                    float(np.min(lengths)),
                    rel_tol=1e-5,
                    abs_tol=TRACTOGRAM_LENGTH_ABS_TOL_MM,
                )
                and math.isclose(
                    float(qc_max),
                    float(np.max(lengths)),
                    rel_tol=1e-5,
                    abs_tol=TRACTOGRAM_LENGTH_ABS_TOL_MM,
                )
            )
            _record_check(
                checks,
                "tractography.qc_length_summary",
                summary_matches,
                {
                    "qc_min_mm": float(qc_min),
                    "file_min_mm": float(np.min(lengths)),
                    "qc_max_mm": float(qc_max),
                    "file_max_mm": float(np.max(lengths)),
                },
                "QC and saved tractogram length extrema match",
            )

    dwi_path = _analysis_input_path(manifest, "dwi")
    if dwi_path.is_file():
        dwi = nib.load(dwi_path.as_posix())
        corners = np.asarray(
            [
                [i, j, k, 1.0]
                for i in (0, dwi.shape[0] - 1)
                for j in (0, dwi.shape[1] - 1)
                for k in (0, dwi.shape[2] - 1)
            ],
            dtype=np.float64,
        )
        world = (dwi.affine @ corners.T).T[:, :3]
        margin = 2.0 * max(float(value) for value in dwi.header.get_zooms()[:3])
        lower = world.min(axis=0) - margin
        upper = world.max(axis=0) + margin
        within = bool(np.all((points >= lower) & (points <= upper)))
        _record_check(
            checks,
            "tractography.world_bounds",
            within,
            {"point_min": points.min(axis=0).tolist(), "point_max": points.max(axis=0).tolist()},
            "all points within DWI world bounds plus a two-voxel margin",
        )
    return True


def _audit_dki(manifest: dict[str, Any], checks: list[dict[str, Any]]) -> bool:
    """Audit raw DKI estimates separately from bounded display copies."""
    import nibabel as nib
    import numpy as np
    from microstructure.dki import DKI_METRICS

    task = _task_entry(manifest, "microstructure.dki")
    if task is None:
        return False
    _record_check(checks, "dki.task_status", task.get("status") in {"completed", "skipped", "reused"},
                  task.get("status"), "completed or explicitly reused")
    outputs = task.get("outputs") or {}
    qc_path = Path(str(task.get("qc_path") or outputs.get("qc") or ""))
    qc = _json_record(qc_path)[1] if qc_path.is_file() else {}
    _record_check(checks, "dki.qc_completed", (qc or {}).get("status") == "completed",
                  str(qc_path), "matching stage QC exists and execution completed")
    _record_check(checks, "dki.design_rank", (qc or {}).get("eligibility", {}).get("design_rank") == 22,
                  (qc or {}).get("eligibility", {}).get("design_rank"), "22 identifiable parameters")
    arrays = {}
    reference = None
    for key in (*DKI_METRICS, *(name + "_raw" for name in DKI_METRICS), "valid_mask", "fit_mask", "rmse", "nrmse"):
        path = Path(str(outputs.get(key) or ""))
        _record_check(checks, f"dki.{key}.exists", path.is_file(), str(path), "declared file exists")
        if not path.is_file():
            continue
        try:
            image = nib.load(path)
            if reference is None:
                reference = image
                dwi_path = _analysis_input_path(manifest, "dwi")
                if dwi_path.is_file():
                    source = nib.load(dwi_path)
                    _record_check(checks, "dki.source_geometry", source.shape[:3] == image.shape and np.allclose(source.affine, image.affine, atol=1e-4, rtol=0),
                                  list(image.shape), "native DKI grid matches actual analysis DWI")
            array = np.asarray(image.dataobj)
            aligned = array.shape == reference.shape and np.allclose(image.affine, reference.affine, atol=1e-4, rtol=0)
            _record_check(checks, f"dki.{key}.geometry", aligned, list(array.shape), "same shape and affine as MK")
            if aligned:
                arrays[key] = array
        except Exception as exc:
            _record_check(checks, f"dki.{key}.readable", False, str(exc), "readable NIfTI")
    valid, fit = arrays.get("valid_mask"), arrays.get("fit_mask")
    if valid is None or fit is None:
        return True
    binary = np.isfinite(valid).all() and np.isfinite(fit).all() and np.isin(valid, (0, 1)).all() and np.isin(fit, (0, 1)).all()
    _record_check(checks, "dki.masks_binary", binary, None, "finite binary fit and validity masks")
    valid, fit = valid > 0, fit > 0
    _record_check(checks, "dki.valid_subset_fit", not np.any(valid & ~fit), int(valid.sum()), "valid voxels were actually fitted")
    _record_check(checks, "dki.nonempty_valid_fit", bool(valid.any()), int(valid.sum()), "at least one numerically valid voxel; not a quality-rate guarantee")
    for key in DKI_METRICS:
        raw, display = arrays.get(key + "_raw"), arrays.get(key)
        if raw is None or display is None:
            continue
        lower, upper = (qc or {}).get("display_bounds", {}).get(key, (0, 1) if key == "kfa" else (0, 3))
        _record_check(checks, f"dki.{key}.valid_values", bool(np.isfinite(raw[valid]).all() and np.all((raw[valid] >= lower) & (raw[valid] <= upper))),
                      int(valid.sum()), "raw values finite and within documented validity bounds on valid mask")
        _record_check(checks, f"dki.{key}.display_derivation", bool(np.allclose(display[fit], np.clip(raw[fit], lower, upper), equal_nan=True)),
                      None, "display is a separate bounded copy, not altered quantitative values")
    for name in ("valid", "fitted"):
        count = int(valid.sum()) if name == "valid" else int(fit.sum())
        _record_check(checks, f"dki.{name}_count", count == (qc or {}).get(f"{name}_voxels"), count, "independent voxel count matches QC")
    for key in ("rmse", "nrmse"):
        if key in arrays:
            values = arrays[key][valid]
            _record_check(checks, f"dki.{key}.valid_values", bool(np.isfinite(values).all() and np.all(values >= 0)), None,
                          "finite nonnegative residuals on valid voxels")
    mkcurve_path = (qc or {}).get("mkcurve", {}).get("record_path")
    if mkcurve_path:
        _audit_mkcurve(mkcurve_path, reference, checks)
    return True


def _audit_mkcurve(record_path, reference, checks):
    """Independent geometry, validity, correction and reproducibility checks."""
    import nibabel as nib
    import numpy as np
    from microstructure.dki import _signature

    record = _json_record(record_path)[1] or {}
    if record.get("kind") != "mkcurve_python_port":
        return
    _record_check(checks, "mkcurve.record_completed", record.get("status") == "completed",
                  record.get("status"), "completed Python correction with provenance")
    maps = record.get("maps") or {}
    arrays = {}
    for key, value in maps.items():
        try:
            image = nib.load(value)
            aligned = reference is not None and image.shape == reference.shape and np.allclose(image.affine, reference.affine, atol=1e-4, rtol=0)
            _record_check(checks, f"mkcurve.{key}.geometry", aligned, list(image.shape), "native shape and affine")
            if aligned:
                arrays[key] = image.get_fdata()
        except Exception as exc:
            _record_check(checks, f"mkcurve.{key}.readable", False, str(exc), "readable declared image")
    for key in ("corrected_valid_mask", "abnormal_mask", "curve_valid_mask"):
        values = arrays.get(key)
        _record_check(checks, f"mkcurve.{key}.binary", values is not None and np.isin(values, (0, 1)).all(),
                      None, "explicit finite binary mask")
    valid = arrays.get("corrected_valid_mask")
    if valid is not None:
        valid = valid > 0
        _record_check(checks, "mkcurve.nonempty_corrected_valid", bool(valid.any()), int(valid.sum()),
                      "at least one valid corrected quantitative voxel; not a quality-rate guarantee")
        _record_check(checks, "mkcurve.corrected_valid_count", int(valid.sum()) == record.get("corrected_valid_voxels"),
                      int(valid.sum()), "independent valid count agrees with record, not a clinical guarantee")
        for key in ("fa", "md", "ad", "rd", "mk", "ak", "rk", "nrmse"):
            values = arrays.get("corrected_" + key)
            lower, upper = (0, 1) if key == "fa" else ((0, 3) if key in {"mk", "ak", "rk"} else (0, None if key == "nrmse" else .005))
            passed = values is not None and np.isfinite(values[valid]).all() and np.all(values[valid] >= lower)
            if values is not None and upper is not None:
                passed &= np.all(values[valid] <= upper)
            _record_check(checks, f"mkcurve.corrected_{key}.valid_values", passed, None,
                          "raw untruncated values satisfy declared valid-domain conditions")
    required = ("abnormal_mask", "zero_mk_b0", "max_mk_b0", "corrected_b0")
    if all(key in arrays for key in required):
        abnormal = arrays["abnormal_mask"] > 0
        threshold = record.get("settings", {}).get("threshold", .5)
        expected = arrays["zero_mk_b0"] * (1 - threshold) + arrays["max_mk_b0"] * threshold
        _record_check(checks, "mkcurve.b0_correction", np.allclose(arrays["corrected_b0"][abnormal], expected[abnormal], atol=2e-4, rtol=1e-6),
                      int(abnormal.sum()), "corrected b0 equals declared threshold blend at detected voxels")
        _record_check(checks, "mkcurve.corrected_count", int(abnormal.sum()) == record.get("corrected_voxels"),
                      int(abnormal.sum()), "independent abnormal count agrees with record")
    for key, value in (record.get("auxiliary_files") or {}).items():
        try:
            passed = _signature(Path(value)) == record.get("auxiliary_artifacts", {}).get(key)
        except OSError:
            passed = False
        _record_check(checks, f"mkcurve.{key}.hash", passed, value, "curve/grid/coefficients match recorded artifact hash")


def _audit_optional_microstructure(manifest: dict[str, Any], checks: list[dict[str, Any]]) -> set[str]:
    import nibabel as nib
    import numpy as np

    route_specs = {
        "freewater": {
            "task": "microstructure.freewater",
            "maps": {"fwf": (0.0, 1.0), "fw_fa": (0.0, 1.0), "fw_md": (0.0, None)},
        },
        "noddi": {
            "task": "microstructure.noddi",
            "maps": {
                "ndi": (0.0, 1.0),
                "odi": (0.0, 1.0),
                "isovf": (0.0, 1.0),
                "rmse": (0.0, None),
                "nrmse": (0.0, None),
            },
        },
    }
    output_aliases = {
        "fw_fa": "fw_tissue_fa",
        "fw_md": "fw_tissue_md",
        "ndi": "noddi_ndi",
        "odi": "noddi_odi",
        "isovf": "noddi_fwf",
        "rmse": "noddi_rmse",
        "nrmse": "noddi_nrmse",
    }
    selected: set[str] = set()
    mask_path = _analysis_input_path(manifest, "mask")
    mask_image = nib.load(mask_path.as_posix()) if mask_path.is_file() else None
    mask = np.asarray(mask_image.dataobj) > 0 if mask_image is not None else None

    for route, spec in route_specs.items():
        task = _task_entry(manifest, str(spec["task"]))
        if task is None:
            continue
        selected.add(route)
        _record_check(
            checks,
            f"{route}.task_status",
            task.get("status") in {"completed", "skipped"},
            task.get("status"),
            "completed or skipped/reused",
        )
        outputs = dict(task.get("outputs") or {})
        qc_path = None
        qc_payload = None
        route_mask_image = mask_image
        route_mask = mask
        if route in {"freewater", "noddi"}:
            qc_value = outputs.get("qc") or outputs.get("noddi_qc")
            if route == "freewater" and not qc_value:
                valid_value = outputs.get("valid_mask") or outputs.get("fw_tissue_valid_mask")
                qc_value = str(Path(str(valid_value)).parent / "freewater_qc.json") if valid_value else None
            qc_path = Path(str(qc_value)) if qc_value else None
            qc_payload = _json_record(qc_path)[1] if qc_path is not None else None
            recorded_mask = ((qc_payload or {}).get("input") or {}).get("mask")
            recorded_mask_path = Path(str(recorded_mask)) if recorded_mask else None
            if recorded_mask_path is not None and recorded_mask_path.is_file():
                route_mask_image = nib.load(recorded_mask_path.as_posix())
                route_mask = np.asarray(route_mask_image.dataobj) > 0
        for key, (minimum, maximum) in dict(spec["maps"]).items():
            value = outputs.get(key) or outputs.get(output_aliases.get(key, ""))
            path = Path(str(value)) if value else None
            exists = bool(path and path.is_file())
            _record_check(
                checks,
                f"{route}.{key}.exists",
                exists,
                path.as_posix() if path else None,
                "file exists",
            )
            if not exists or path is None:
                continue
            image = nib.load(path.as_posix())
            array = np.asarray(image.dataobj, dtype=np.float64)
            finite = bool(np.isfinite(array).all())
            _record_check(checks, f"{route}.{key}.finite", finite, None, "all values finite")
            if route_mask_image is not None and route_mask is not None:
                _record_check(
                    checks,
                    f"{route}.{key}.shape",
                    array.shape == route_mask.shape,
                    list(array.shape),
                    list(route_mask.shape),
                )
                _record_check(
                    checks,
                    f"{route}.{key}.affine",
                    bool(np.allclose(image.affine, route_mask_image.affine, atol=1e-4)),
                    float(np.max(np.abs(image.affine - route_mask_image.affine))),
                    "maximum absolute difference <= 1e-4",
                )
            values = (
                array[route_mask]
                if route_mask is not None and array.shape == route_mask.shape and np.any(route_mask)
                else array.ravel()
            )
            if not finite or values.size == 0:
                continue
            observed_min = float(np.min(values))
            observed_max = float(np.max(values))
            in_range = observed_min >= float(minimum) - 1e-6
            requirement = f">= {minimum}"
            if maximum is not None:
                in_range = in_range and observed_max <= float(maximum) + 1e-6
                requirement = f"{minimum} <= value <= {maximum}"
            _record_check(
                checks,
                f"{route}.{key}.range",
                in_range,
                {"min": observed_min, "max": observed_max},
                requirement,
            )
        if route in {"freewater", "noddi"}:
            validity_aliases = {
                "freewater": ("valid_mask", "fw_tissue_valid_mask"),
                "noddi": ("valid_mask", "noddi_tissue_valid_mask"),
            }
            validity_value = next((outputs.get(key) for key in validity_aliases[route] if outputs.get(key)), None)
            validity_path = Path(str(validity_value)) if validity_value else None
            validity_exists = bool(validity_path and validity_path.is_file())
            _record_check(
                checks,
                f"{route}.valid_mask.exists",
                validity_exists,
                validity_path.as_posix() if validity_path else None,
                "file exists",
            )
            if validity_exists and validity_path is not None:
                validity = np.asarray(nib.load(validity_path.as_posix()).dataobj)
                _record_check(
                    checks,
                    f"{route}.valid_mask.binary",
                    bool(np.all(np.isin(validity, (0, 1)))),
                    sorted(float(value) for value in np.unique(validity)),
                    "binary 0/1 mask",
                )
                if route == "noddi" and route_mask is not None:
                    for metric_key in ("ndi", "odi"):
                        metric_value = outputs.get(metric_key) or outputs.get(output_aliases[metric_key])
                        if not metric_value:
                            continue
                        metric = np.asarray(nib.load(str(metric_value)).dataobj, dtype=np.float64)
                        outside_brain = metric[~route_mask]
                        _record_check(
                            checks,
                            f"noddi.{metric_key}.zero_outside_brain_mask",
                            bool(np.allclose(outside_brain, 0.0, atol=1e-6)),
                            float(np.max(np.abs(outside_brain))) if outside_brain.size else 0.0,
                            "all values outside the brain mask equal zero",
                        )
            if route == "freewater":
                _record_check(
                    checks,
                    "freewater.qc.completed",
                    bool(qc_payload and qc_payload.get("status") == "completed"),
                    qc_path.as_posix() if qc_path else None,
                    "completed freewater_qc.json",
                )
                if qc_payload:
                    qc_checks = qc_payload.get("checks") or {}
                    valid_fraction = float(
                        ((qc_payload.get("validity") or {}).get("valid_tissue_fraction", 0.0)) or 0.0
                    )
                    _record_check(
                        checks,
                        "freewater.qc.output_contract",
                        bool(qc_checks) and all(value is True for value in qc_checks.values()),
                        qc_checks,
                        "all native Free Water output-contract checks pass",
                    )
                    _record_check(
                        checks,
                        "freewater.qc.valid_tissue_fraction",
                        valid_fraction >= 0.5,
                        valid_fraction,
                        ">= 0.5 of brain-mask voxels have interpretable tissue FA/MD",
                    )
            if route == "noddi":
                _record_check(
                    checks,
                    "noddi.qc.completed",
                    bool(qc_payload and qc_payload.get("status") == "completed"),
                    qc_path.as_posix() if qc_path else None,
                    "completed noddi_qc.json",
                )
                if qc_payload:
                    validity = qc_payload.get("validity") or {}
                    summary = qc_payload.get("summary") or {}
                    nrmse = summary.get("nrmse_valid_tissue") or {}
                    valid_fraction = float(validity.get("valid_tissue_fraction", 0.0) or 0.0)
                    nrmse_p95 = float(nrmse.get("p95", float("inf")) or 0.0)
                    _record_check(
                        checks,
                        "noddi.qc.valid_tissue_fraction",
                        valid_fraction >= 0.5,
                        valid_fraction,
                        ">= 0.5 of brain-mask voxels have finite, bounded tissue estimates with IsoVF below the configured threshold",
                    )
                    _record_check(
                        checks,
                        "noddi.qc.nrmse_p95",
                        math.isfinite(nrmse_p95) and nrmse_p95 <= 0.5,
                        nrmse_p95,
                        "95th percentile normalized fitting error <= 0.5 in tissue-valid voxels",
                    )
    return selected


def _json_record(path_value: Any) -> tuple[Path | None, dict[str, Any] | None]:
    path = Path(str(path_value)) if path_value else None
    if path is None or not path.is_file():
        return path, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return path, None
    return path, payload if isinstance(payload, dict) else None


def _audit_qc_routes(manifest: dict[str, Any], checks: list[dict[str, Any]]) -> set[str]:
    route_specs = {
        "saf": ("tractography.saf", "streamline_count", 1),
        "wmql": ("bundles.wmql", "bundle_count", 1),
        "wma": ("bundles.wma", "anatomical_tract_count", 73),
        "tractcloud": ("bundles.tractcloud", "bundle_count", 1),
    }
    selected: set[str] = set()
    for route, (task_name, count_key, minimum_count) in route_specs.items():
        task = _task_entry(manifest, task_name)
        if task is None:
            continue
        selected.add(route)
        _record_check(
            checks,
            f"{route}.task_status",
            task.get("status") in {"completed", "skipped"},
            task.get("status"),
            "completed or skipped/reused",
        )
        outputs = dict(task.get("outputs") or {})
        qc_path, qc = _json_record(task.get("qc_path") or outputs.get("qc"))
        _record_check(
            checks,
            f"{route}.qc_exists",
            qc is not None,
            qc_path.as_posix() if qc_path else None,
            "readable QC JSON exists",
        )
        if qc is None:
            continue
        _record_check(checks, f"{route}.qc_status", qc.get("status") == "completed", qc.get("status"), "completed")
        summary = dict(qc.get("output_summary") or {})
        if route == "wma" and not {
            "anatomical_tract_readable_count",
            "anatomical_tract_nonempty_count",
        }.issubset(summary):
            _, native_manifest = _json_record(task.get("manifest_path") or outputs.get("manifest"))
            input_value = None
            output_value = None
            if native_manifest:
                input_value = (
                    native_manifest.get("prepared_input_tractography")
                    or native_manifest.get("input_tractography")
                )
                output_value = native_manifest.get("output_dir")
            input_path = Path(str(input_value)) if input_value else None
            output_dir = Path(str(output_value)) if output_value else (qc_path.parent if qc_path else None)
            if input_path is not None and input_path.is_file() and output_dir is not None:
                try:
                    from tractography.wma import summarize_wma_outputs

                    summary.update(summarize_wma_outputs(output_dir, input_path))
                except Exception as exc:
                    _record_check(
                        checks,
                        "wma.legacy_summary_decode",
                        False,
                        f"{type(exc).__name__}: {exc}",
                        "legacy WMA outputs can be decoded with the current content-aware reader",
                    )
        # Recheck files on disk even if an older QC JSON claimed success.
        if route in {"wma", "tractcloud"}:
            from shared.output_contracts import vtp_streamline_count

            if route == "wma":
                tract_dir = summary.get("anatomical_tracts_dir")
                paths = sorted(Path(str(tract_dir)).glob("*.vtp")) if tract_dir else []
                decoded = {path.name: vtp_streamline_count(path) for path in paths}
                summary["anatomical_tract_readable_count"] = sum(v is not None for v in decoded.values())
                summary["anatomical_tract_nonempty_count"] = sum(v is not None and v > 0 for v in decoded.values())
                summary["anatomical_empty_tracts"] = [k for k, v in decoded.items() if v == 0]
                summary["anatomical_tract_count"] = len(paths)
                summary["anatomical_tract_streamline_counts"] = decoded
                from tractography.wma import _wma_output_contract_error, tract_sampling_eligibility

                error = _wma_output_contract_error(summary)
                _record_check(checks, "wma.current_sampling_contract", error is None, error,
                              "current decoded coverage and major-tract sampling contract; anatomical review separate")
                sampling = tract_sampling_eligibility(decoded)
                summary['tract_sampling'] = sampling
                _record_check(checks, "wma.all_tract_sampling", sampling['all_tract_sampling_passed'],
                              sampling['excluded_from_quantification'],
                              "every returned atlas tract satisfies the unchanged sampling minimum; not statistical power or anatomical accuracy")
            else:
                paths = [Path(str(value)) for value in summary.get("bundle_files", [])]
                decoded = {str(path): vtp_streamline_count(path) for path in paths}
                summary["bundle_readable_count"] = sum(v is not None for v in decoded.values())
                summary["bundle_nonempty_count"] = sum(v is not None and v > 0 for v in decoded.values())
        if route == "wmql":
            from tractography.wmql import summarize_wmql_outputs, _wmql_output_contract_error

            _, native_manifest = _json_record(task.get("manifest_path") or outputs.get("manifest"))
            queries = tuple((native_manifest or {}).get("query_selection") or ())
            root = summary.get("root") or (qc_path.parent if qc_path else None)
            if root is not None:
                try:
                    summary = summarize_wmql_outputs(Path(root), str(qc.get("subject_id") or manifest.get("subject_id") or "") or None)
                    error = _wmql_output_contract_error(summary, len(queries), queries)
                except Exception as exc:
                    error = str(exc)
            else:
                error = "Missing WMQL output root"
            _record_check(checks, "wmql.current_sampling_contract", error is None, error,
                          "current decoded requested-query/sampling contract; candidate anatomy remains pending")
        count_value = qc.get(count_key)
        if count_value is None:
            count_value = summary.get(count_key)
        try:
            count = int(count_value)
        except (TypeError, ValueError):
            count = 0
        _record_check(
            checks,
            f"{route}.{count_key}",
            count >= minimum_count,
            count,
            f">= {minimum_count}",
        )

        bundle_files = [Path(str(value)) for value in summary.get("bundle_files", [])]
        if bundle_files:
            valid_files = [path for path in bundle_files if path.is_file() and path.stat().st_size > 0]
            _record_check(
                checks,
                f"{route}.bundle_files",
                len(valid_files) == len(bundle_files) == count,
                {"declared": count, "listed": len(bundle_files), "valid": len(valid_files)},
                "declared, listed, and non-empty bundle counts match",
            )
        if route == "wma":
            readable_count = int(summary.get("anatomical_tract_readable_count", 0) or 0)
            nonempty_count = int(summary.get("anatomical_tract_nonempty_count", 0) or 0)
            _record_check(
                checks,
                "wma.anatomical_tracts_readable",
                readable_count >= minimum_count and readable_count == count,
                readable_count,
                f"all declared VTP files decode; at least {minimum_count} files",
            )
            _record_check(
                checks,
                "wma.anatomical_tracts_nonempty",
                nonempty_count >= 55,
                {
                    "nonempty": nonempty_count,
                    "empty": summary.get("anatomical_empty_tracts") or [],
                },
                ">= 55 of the 73 atlas tracts contain streamlines",
            )
        if route in {"wmql", "tractcloud"}:
            readable_count = int(summary.get("bundle_readable_count", 0) or 0)
            nonempty_count = int(summary.get("bundle_nonempty_count", 0) or 0)
            _record_check(
                checks,
                f"{route}.bundles_decoded_nonempty",
                count > 0 and readable_count == count and nonempty_count == count,
                {"declared": count, "readable": readable_count, "nonempty": nonempty_count},
                "every declared bundle is readable and non-empty",
            )
        if route == "saf":
            for key in ("saf_tractogram", "saf_vtk"):
                record = summary.get(key) or {}
                path = Path(str(record.get("path") or outputs.get("tractogram" if key == "saf_tractogram" else "tractogram_vtk") or ""))
                _record_check(
                    checks,
                    f"saf.{key}_file",
                    path.is_file() and path.stat().st_size > 0,
                    path.as_posix(),
                    "non-empty file exists",
                )
            scientific_summary = qc.get("scientific_output_summary")
            _record_check(
                checks,
                "saf.scientific_output_summary",
                isinstance(scientific_summary, dict)
                and int(scientific_summary.get("streamline_count", 0) or 0) == count
                and bool(scientific_summary.get("all_points_finite")),
                scientific_summary,
                "decoded streamline count matches QC and all points are finite",
            )
    return selected


def _audit_cross_route_inputs(manifest: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    route_pairs = (
        ("tractography.csd_act", "tractogram_vtk", "bundles.tractcloud", "tractcloud.input_matches_same_run_tractography"),
    )
    for source_task_name, source_key, consumer_task_name, check_name in route_pairs:
        source_task = _task_entry(manifest, source_task_name)
        consumer_task = _task_entry(manifest, consumer_task_name)
        if source_task is None or consumer_task is None:
            continue
        expected_value = dict(source_task.get("outputs") or {}).get(source_key)
        _, consumer_manifest = _json_record(consumer_task.get("manifest_path"))
        actual_value = consumer_manifest.get("input_tractography") if consumer_manifest else None
        expected = Path(str(expected_value)).resolve() if expected_value else None
        actual = Path(str(actual_value)).resolve() if actual_value else None
        _record_check(
            checks,
            check_name,
            expected is not None and actual == expected,
            {
                "expected": expected.as_posix() if expected else None,
                "actual": actual.as_posix() if actual else None,
            },
            "consumer input equals the producing task output from this run",
        )

    wma_task = _task_entry(manifest, "bundles.wma")
    if wma_task is not None:
        _, wma_manifest = _json_record(wma_task.get("manifest_path"))
        input_value = wma_manifest.get("input_tractography") if wma_manifest else None
        input_path = Path(str(input_value)).resolve() if input_value else None
        _record_check(
            checks,
            "wma.input_tractography_exists",
            input_path is not None and input_path.is_file() and input_path.stat().st_size > 0,
            input_path.as_posix() if input_path else None,
            "explicit WMA whole-brain input exists and is non-empty",
        )


def _audit_connectomes(manifest: dict[str, Any], checks: list[dict[str, Any]]) -> bool:
    tasks = [
        task
        for task in manifest.get("tasks", [])
        if str(task.get("name") or "").startswith("connectome.")
    ]
    if not tasks:
        return False
    for task in tasks:
        task_name = str(task.get("name"))
        prefix = task_name.replace(".", "_")
        _record_check(
            checks,
            f"{prefix}.task_status",
            task.get("status") in {"completed", "skipped"},
            task.get("status"),
            "completed or skipped/reused",
        )
        outputs = dict(task.get("outputs") or {})
        qc_path, qc = _json_record(task.get("qc_path") or outputs.get("qc"))
        manifest_path, native_manifest = _json_record(task.get("manifest_path") or outputs.get("manifest"))
        _record_check(checks, f"{prefix}.qc_exists", qc is not None, qc_path.as_posix() if qc_path else None, "readable QC JSON exists")
        _record_check(
            checks,
            f"{prefix}.manifest_exists",
            native_manifest is not None,
            manifest_path.as_posix() if manifest_path else None,
            "readable manifest JSON exists",
        )
        if qc is None or native_manifest is None:
            continue
        _record_check(checks, f"{prefix}.qc_status", qc.get("status") == "completed", qc.get("status"), "completed")
        native_outputs = dict(native_manifest.get("outputs") or {})
        matrix_path = Path(str(outputs.get("matrix") or native_outputs.get("matrix") or ""))
        nodes_path = Path(str(outputs.get("nodes") or native_outputs.get("nodes") or ""))
        _record_check(checks, f"{prefix}.matrix_exists", matrix_path.is_file(), matrix_path.as_posix(), "file exists")
        _record_check(checks, f"{prefix}.nodes_exists", nodes_path.is_file(), nodes_path.as_posix(), "file exists")
        if not matrix_path.is_file() or not nodes_path.is_file():
            continue
        with matrix_path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        with nodes_path.open("r", encoding="utf-8", newline="") as stream:
            node_rows = list(csv.DictReader(stream, delimiter="\t"))
        metric = str(qc.get("metric") or native_manifest.get("metric") or "streamline_count")
        if rows and "source_node" not in rows[0]:
            # Raw MRtrix CSVs are label-indexed dense matrices; the nodes TSV
            # defines which rows represent anatomical nodes.
            import numpy as np
            try:
                dense = np.loadtxt(matrix_path, delimiter=",")
                square = dense.ndim == 2 and dense.shape[0] == dense.shape[1]
                _record_check(checks, f"{prefix}.dense_square", square, list(dense.shape), "square matrix")
                _record_check(checks, f"{prefix}.dense_values", bool(np.isfinite(dense).all() and (dense >= 0).all()), None, "finite and non-negative")
                symmetric = bool(native_manifest.get("symmetric", True))
                _record_check(checks, f"{prefix}.dense_symmetry", not symmetric or (square and np.allclose(dense, dense.T)), None, "symmetric when requested")
                labels = [int(row["label"]) for row in node_rows]
                indices = list(range(len(labels))) if dense.shape == (len(labels), len(labels)) else [label - 1 for label in labels]
                mapped = square and all(0 <= index < dense.shape[0] for index in indices)
                _record_check(checks, f"{prefix}.dense_node_mapping", mapped, len(indices), "all node labels map to matrix rows")
                _record_check(checks, f"{prefix}.dense_dimension", qc.get("dense_node_count") == dense.shape[0], qc.get("dense_node_count"), "QC records the raw matrix dimension")
                rows = []
                if mapped:
                    for i, source in enumerate(labels):
                        for j in range(i if symmetric else 0, len(labels)):
                            value = float(dense[indices[i], indices[j]])
                            if value != 0:
                                rows.append({"source_node": source, "target_node": labels[j], metric: value})
            except (OSError, ValueError, IndexError, KeyError, TypeError) as exc:
                _record_check(checks, f"{prefix}.dense_decode", False, str(exc), "readable numerical matrix")
                rows = []
        required_columns = {"source_node", "target_node", metric}
        columns = set(rows[0]) if rows else set()
        _record_check(checks, f"{prefix}.columns", required_columns.issubset(columns), sorted(columns), sorted(required_columns))
        try:
            pairs = [(int(row["source_node"]), int(row["target_node"])) for row in rows]
            values = [float(row[metric]) for row in rows]
        except (KeyError, TypeError, ValueError):
            pairs, values = [], []
        finite_nonnegative = bool(values) and all(math.isfinite(value) and value >= 0 for value in values)
        _record_check(checks, f"{prefix}.values", finite_nonnegative, len(values), "non-empty, finite, and non-negative")
        _record_check(checks, f"{prefix}.unique_edges", len(pairs) == len(set(pairs)), len(pairs), "no duplicate edges")
        _record_check(checks, f"{prefix}.upper_triangle", bool(pairs) and all(a <= b for a, b in pairs), None, "source_node <= target_node")
        node_labels = {int(row["label"]) for row in node_rows if str(row.get("label") or "").strip().lstrip("-").isdigit()}
        edge_labels = {value for pair in pairs for value in pair}
        _record_check(checks, f"{prefix}.node_labels", bool(node_labels) and edge_labels.issubset(node_labels), len(node_labels), "all edge labels occur in nodes TSV")
        edge_count = int(qc.get("edge_count") or 0)
        node_count = int(qc.get("node_count") or 0)
        _record_check(checks, f"{prefix}.edge_count", edge_count == len(rows), {"qc": edge_count, "file": len(rows)}, "QC and CSV match")
        _record_check(checks, f"{prefix}.node_count", node_count == len(node_rows), {"qc": node_count, "file": len(node_rows)}, "QC and TSV match")
        if metric == "streamline_count" and qc.get("streamline_count_assigned") is not None and values:
            assigned = int(qc["streamline_count_assigned"])
            _record_check(
                checks,
                f"{prefix}.assigned_streamlines",
                math.isclose(sum(values), assigned, rel_tol=0.0, abs_tol=1e-6),
                {"qc": assigned, "csv_sum": sum(values)},
                "CSV edge counts sum to QC assigned streamlines",
            )
    return True


def _audit_structural(manifest: dict[str, Any], checks: list[dict[str, Any]]) -> set[str]:
    import nibabel as nib
    import numpy as np
    from nibabel.processing import resample_from_to

    output_root_value = manifest.get("output_root")
    if not output_root_value:
        return set()
    output_root = Path(str(output_root_value))
    subject_id = str(manifest["subject_id"])
    from shared.io_layout import anatomy_input_dir
    ddseg_dir = anatomy_input_dir(output_root, "ddseg")
    ddparcel_dir = anatomy_input_dir(output_root, "ddparcel")
    surface_dir = anatomy_input_dir(output_root, "ddsurfer")
    task_names = {
        "ddseg": "segmentation.ddseg",
        "ddparcel": "parcellation.ddparcel",
        "ddsurfer": "surface.ddsurfer",
    }
    selected = {
        route
        for route, task_name in task_names.items()
        if _task_entry(manifest, task_name) is not None
    }
    if not selected:
        return set()

    mask_path = _analysis_input_path(manifest, "mask")
    if not mask_path.is_file():
        _record_check(checks, "structural.mask_exists", False, mask_path.as_posix(), "file exists")
        return selected
    mask_image = nib.load(mask_path.as_posix())
    mask = np.asarray(mask_image.dataobj) > 0

    ddseg_outputs = _task_outputs(manifest, task_names["ddseg"])
    segmentation_path = Path(
        str(ddseg_outputs.get("segmentation") or ddseg_dir / "SegmentationMap_GMWMCSF.nii.gz")
    )
    probability_paths = {
        "gm": Path(str(ddseg_outputs.get("gm_probability") or ddseg_dir / "ProbabilisticMap_GM.nii.gz")),
        "wm": Path(str(ddseg_outputs.get("wm_probability") or ddseg_dir / "ProbabilisticMap_WM.nii.gz")),
        "csf": Path(str(ddseg_outputs.get("csf_probability") or ddseg_dir / "ProbabilisticMap_CSF.nii.gz")),
    }
    if "ddseg" in selected:
        _record_check(
            checks,
            "structural.ddseg.segmentation_exists",
            segmentation_path.is_file(),
            segmentation_path.as_posix(),
            "file exists",
        )
    if "ddseg" in selected and segmentation_path.is_file():
        segmentation_image = nib.load(segmentation_path.as_posix())
        segmentation = np.asarray(segmentation_image.dataobj, dtype=np.float64)
        rounded = np.rint(segmentation)
        _record_check(
            checks,
            "structural.ddseg.labels_discrete",
            bool(np.allclose(segmentation, rounded, atol=1e-4)),
            sorted(float(value) for value in np.unique(segmentation)),
            "all labels are integers",
        )
        label_set = {int(value) for value in np.unique(rounded)}
        _record_check(
            checks,
            "structural.ddseg.labels",
            label_set.issubset({0, 1, 2, 3}) and {1, 2, 3}.issubset(label_set),
            sorted(label_set),
            "labels are 0/1/2/3 and all three tissue classes are present",
        )
        native_segmentation = np.asarray(
            resample_from_to(segmentation_image, mask_image, order=0).dataobj
        ) > 0
        denominator = int(mask.sum()) + int(native_segmentation.sum())
        dice = (
            2.0 * float(np.count_nonzero(mask & native_segmentation)) / denominator
            if denominator
            else 0.0
        )
        _record_check(
            checks,
            "structural.ddseg.world_overlap",
            dice >= 0.95,
            dice,
            "foreground Dice with input mask after world-space resampling >= 0.95",
        )

        probability_arrays: dict[str, Any] = {}
        for name, path in probability_paths.items():
            exists = path.is_file()
            _record_check(
                checks,
                f"structural.ddseg.{name}_probability_exists",
                exists,
                path.as_posix(),
                "file exists",
            )
            if not exists:
                continue
            image = nib.load(path.as_posix())
            array = np.asarray(image.dataobj, dtype=np.float64)
            probability_arrays[name] = array
            _record_check(
                checks,
                f"structural.ddseg.{name}_probability_shape",
                array.shape == segmentation.shape,
                list(array.shape),
                list(segmentation.shape),
            )
            _record_check(
                checks,
                f"structural.ddseg.{name}_probability_range",
                bool(np.isfinite(array).all() and np.min(array) >= -1e-6 and np.max(array) <= 1.0 + 1e-6),
                {"min": float(np.min(array)), "max": float(np.max(array))},
                "all values finite and within [0, 1]",
            )
        if len(probability_arrays) == 3:
            foreground = rounded > 0
            probability_stack = np.stack(
                [probability_arrays[name] for name in ("gm", "wm", "csf")],
                axis=-1,
            )
            probability_sum = np.sum(probability_stack, axis=-1)
            foreground_mass = probability_sum[foreground]
            valid_mass = np.isfinite(foreground_mass) & (foreground_mass > 0.0) & (
                foreground_mass <= 1.0 + 1e-5
            )
            valid_mass_fraction = float(np.mean(valid_mass)) if foreground_mass.size else 0.0
            _record_check(
                checks,
                "structural.ddseg.probability_mass",
                valid_mass_fraction >= 0.999,
                {
                    "valid_fraction": valid_mass_fraction,
                    "min": float(np.min(foreground_mass)) if foreground_mass.size else None,
                    "max": float(np.max(foreground_mass)) if foreground_mass.size else None,
                },
                ">= 0.999 of segmented voxels have finite, non-zero GM+WM+CSF mass <= 1",
            )
            predicted_labels = np.asarray((2, 1, 3), dtype=np.int16)[
                np.argmax(probability_stack, axis=-1)
            ]
            label_agreement = float(np.mean(predicted_labels[foreground] == rounded[foreground]))
            _record_check(
                checks,
                "structural.ddseg.probability_label_agreement",
                label_agreement >= 0.999,
                label_agreement,
                ">= 0.999 agreement between the maximum tissue probability and segmentation label",
            )

    ddparcel_outputs = _task_outputs(manifest, task_names["ddparcel"])
    parcellation_path = Path(
        str(ddparcel_outputs.get("wmparc") or ddparcel_dir / f"{subject_id}-DDSurfer-wmparc.nii.gz")
    )
    if "ddparcel" in selected:
        _record_check(
            checks,
            "structural.ddparcel.exists",
            parcellation_path.is_file(),
            parcellation_path.as_posix(),
            "file exists",
        )
    if "ddparcel" in selected and parcellation_path.is_file():
        image = nib.load(parcellation_path.as_posix())
        array = np.asarray(image.dataobj, dtype=np.float64)
        rounded = np.rint(array)
        labels = np.unique(rounded)
        _record_check(
            checks,
            "structural.ddparcel.shape",
            array.shape == mask.shape,
            list(array.shape),
            list(mask.shape),
        )
        _record_check(
            checks,
            "structural.ddparcel.affine",
            bool(np.allclose(image.affine, mask_image.affine, atol=1e-4)),
            float(np.max(np.abs(image.affine - mask_image.affine))),
            "maximum absolute affine difference <= 1e-4",
        )
        _record_check(
            checks,
            "structural.ddparcel.labels_discrete",
            bool(np.isfinite(array).all() and np.allclose(array, rounded, atol=1e-4)),
            None,
            "all labels finite and integer-valued",
        )
        nonzero_label_count = int(np.count_nonzero(labels > 0))
        _record_check(
            checks,
            "structural.ddparcel.label_count",
            nonzero_label_count >= 50,
            nonzero_label_count,
            ">= 50 nonzero anatomical labels",
        )

    ddsurfer_outputs = _task_outputs(manifest, task_names["ddsurfer"])
    expected_models = {
        f"{surface}_{hemi}": Path(
            str(
                ddsurfer_outputs.get(
                    {"wm": "white", "pial": "pial"}[surface] + f"_{hemi}"
                )
                or surface_dir
                / "pred_results_SlicerDSpace"
                / "native"
                / subject_id
                / f"{subject_id}_predicted_{surface}_{hemi}.stl"
            )
        )
        for surface in ("wm", "pial")
        for hemi in ("lh", "rh")
    }
    surface_meshes: dict[str, Any] = {}
    for name, model_path in expected_models.items():
        if "ddsurfer" not in selected:
            continue
        exists = model_path.is_file()
        _record_check(
            checks,
            f"structural.ddsurfer.{name}_exists",
            exists,
            model_path.as_posix(),
            "file exists",
        )
        if not exists:
            continue
        import trimesh

        mesh = trimesh.load(model_path.as_posix(), process=True)
        surface_meshes[name] = mesh
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        extents = np.ptp(vertices, axis=0) if vertices.size else np.zeros(3)
        valid = bool(
            len(vertices) >= 1000
            and len(faces) >= 1000
            and np.isfinite(vertices).all()
            and np.all(extents > 10.0)
        )
        _record_check(
            checks,
            f"structural.ddsurfer.{name}_geometry",
            valid,
            {
                "vertices": int(len(vertices)),
                "faces": int(len(faces)),
                "extents_mm": extents.tolist(),
            },
            ">= 1000 vertices/faces, finite coordinates, and > 10 mm extent on each axis",
        )
        _record_check(
            checks,
            f"structural.ddsurfer.{name}_topology",
            bool(
                mesh.is_watertight
                and mesh.is_winding_consistent
                and int(mesh.body_count) == 1
                and int(mesh.euler_number) == 2
            ),
            {
                "watertight": bool(mesh.is_watertight),
                "winding_consistent": bool(mesh.is_winding_consistent),
                "body_count": int(mesh.body_count),
                "euler_number": int(mesh.euler_number),
            },
            "closed, consistently oriented, single-component genus-0 cortical surface",
        )
    for hemi in ("lh", "rh"):
        white = surface_meshes.get(f"wm_{hemi}")
        pial = surface_meshes.get(f"pial_{hemi}")
        if white is None or pial is None:
            continue
        if len(white.vertices) != len(pial.vertices):
            _record_check(
                checks,
                f"structural.ddsurfer.{hemi}_white_pial_correspondence",
                False,
                {"white_vertices": len(white.vertices), "pial_vertices": len(pial.vertices)},
                "white and pial surfaces have corresponding vertex counts",
            )
            continue
        displacement = np.linalg.norm(
            np.asarray(pial.vertices) - np.asarray(white.vertices),
            axis=1,
        )
        displacement_summary = {
            "median_mm": float(np.median(displacement)),
            "p95_mm": float(np.percentile(displacement, 95)),
            "max_mm": float(np.max(displacement)),
        }
        _record_check(
            checks,
            f"structural.ddsurfer.{hemi}_white_pial_correspondence",
            bool(
                np.isfinite(displacement).all()
                and displacement_summary["median_mm"] > 0.1
                and displacement_summary["p95_mm"] < 15.0
            ),
            displacement_summary,
            "finite corresponding-vertex displacement with median > 0.1 mm and p95 < 15 mm",
        )
    return selected


def audit_output_root(
    output_root: Path,
    *,
    require_dti: bool = False,
    require_tractography: bool = False,
    require_structural: bool = False,
    require_full_routes: bool = False,
    min_subjects: int = 1,
    run_name: str | None = None,
) -> dict[str, Any]:
    root = output_root.expanduser().resolve()
    subjects: list[dict[str, Any]] = []
    if (root / "00_run_metadata").is_dir():
        subject_outputs = [root]
    else:
        subject_outputs = sorted(
            path for path in root.iterdir() if path.is_dir() and path.name != "00_batch_metadata"
        )
    for subject_output in subject_outputs:
        manifest_path = _manifest_for_run(subject_output, run_name)
        if manifest_path is None:
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checks: list[dict[str, Any]] = []
        _audit_orchestration(manifest, checks)
        _audit_validation(manifest, checks)
        _audit_preprocessing(manifest, checks)
        has_dti = _audit_dti(manifest, checks)
        _audit_regional_statistics(manifest, checks)
        _audit_act_preparation(manifest, checks)
        has_tractography = _audit_tractography(manifest, checks)
        structural_routes = _audit_structural(manifest, checks)
        optional_microstructure_routes = _audit_optional_microstructure(manifest, checks)
        if _audit_dki(manifest, checks):
            optional_microstructure_routes.add("dki")
        qc_routes = _audit_qc_routes(manifest, checks)
        _audit_cross_route_inputs(manifest, checks)
        has_connectome = _audit_connectomes(manifest, checks)
        if require_dti:
            _record_check(checks, "required.dti", has_dti, has_dti, "DTI outputs present")
        if require_tractography:
            _record_check(
                checks,
                "required.tractography",
                has_tractography,
                has_tractography,
                "tractography output present",
            )
        if require_structural:
            required_routes = {"ddseg", "ddparcel", "ddsurfer"}
            has_structural = required_routes.issubset(structural_routes)
            _record_check(
                checks,
                "required.structural",
                has_structural,
                has_structural,
                "DDSeg, DDParcel, and DDSurfer tasks present",
            )
        if require_full_routes:
            observed_routes = set(optional_microstructure_routes) | set(qc_routes)
            if has_connectome:
                observed_routes.add("connectome")
            missing_routes = sorted(FULL_ROUTE_NAMES - observed_routes)
            _record_check(
                checks,
                "required.full_routes",
                not missing_routes,
                {"observed": sorted(observed_routes), "missing": missing_routes},
                "Free Water, NODDI, SAF, WMQL, WMA, TractCloud, and connectome tasks present",
            )
        failed = [check for check in checks if not check["passed"]]
        subjects.append(
            {
                "subject_id": str(manifest.get("subject_id") or subject_output.name),
                "manifest_path": manifest_path.as_posix(),
                "status": "failed" if failed else "passed",
                "failed_check_count": len(failed),
                "checks": checks,
            }
        )

    enough_subjects = len(subjects) >= int(min_subjects)
    failed_subjects = [subject for subject in subjects if subject["status"] != "passed"]
    status = "passed" if enough_subjects and not failed_subjects else "failed"
    return {
        "kind": "slicerdsurfer_scientific_output_audit",
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "output_root": root.as_posix(),
        "status": status,
        "requirements": {
            "require_dti": bool(require_dti),
            "require_tractography": bool(require_tractography),
            "require_structural": bool(require_structural),
            "require_full_routes": bool(require_full_routes),
            "min_subjects": int(min_subjects),
            "run_name": run_name,
        },
        "summary": {
            "subject_count": len(subjects),
            "passed_subjects": len(subjects) - len(failed_subjects),
            "failed_subjects": len(failed_subjects),
            "minimum_subject_count_met": enough_subjects,
        },
        "subjects": subjects,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True, help="Batch output root containing subject directories.")
    parser.add_argument("--report", type=Path, help="JSON report path. Defaults under 00_batch_metadata.")
    parser.add_argument("--require-dti", action="store_true", help="Fail when a subject lacks completed DTI outputs.")
    parser.add_argument(
        "--require-tractography",
        action="store_true",
        help="Fail when a subject lacks completed DTI or CSD/ACT tractography output.",
    )
    parser.add_argument(
        "--require-structural",
        action="store_true",
        help="Fail unless a subject has DDSeg, DDParcel, and DDSurfer outputs.",
    )
    parser.add_argument(
        "--require-full-routes",
        action="store_true",
        help="Fail unless Free Water, NODDI, SAF, WMQL, WMA, TractCloud, and connectome outputs are audited.",
    )
    parser.add_argument("--min-subjects", type=int, default=1, help="Minimum number of audited subjects.")
    parser.add_argument(
        "--run-name",
        help="Audit this exact orchestration run instead of the most recently modified manifest.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.min_subjects < 1:
        raise SystemExit("--min-subjects must be >= 1")
    report = audit_output_root(
        args.output_root,
        require_dti=bool(args.require_dti),
        require_tractography=bool(args.require_tractography),
        require_structural=bool(args.require_structural),
        require_full_routes=bool(args.require_full_routes),
        min_subjects=int(args.min_subjects),
        run_name=args.run_name,
    )
    report_path = args.report or (
        args.output_root.expanduser().resolve() / "00_batch_metadata" / "scientific_output_audit.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], sort_keys=True))
    print(report_path)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
