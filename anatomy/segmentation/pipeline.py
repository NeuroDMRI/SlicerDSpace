from __future__ import annotations

from shared.cohort_schedule import scheduled_stage

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import nibabel as nib

from .config import DDSegConfig
from .preprocess import OrientedMask, load_mask, masking_and_normalizing, stack_features, split_views
from .inference import combine_views, normalize_class_scores, predict_view
from .orientation import from_model_orientation
from .qc import write_ddseg_qc_record
from .utils import padding_unpadding
from .dti_slicer import run_dti_feature_extraction


def _feature_folder(cfg: DDSegConfig) -> Path:
    if cfg.dti_feature_dir is not None:
        return cfg.dti_feature_dir
    if not (cfg.dwi_nii and cfg.bval and cfg.bvec):
        raise ValueError("DTI generation requires --dwi_nii, --bval, and --bvec.")
    dti_out = cfg.output_folder / "DTI"
    run_dti_feature_extraction(cfg.dwi_nii, cfg.bval, cfg.bvec, dti_out, cfg.slicer_base, cfg.slicer_ext)
    return dti_out


def _is_cuda_tdr_or_launch_timeout(exc: BaseException) -> bool:
    msg = str(exc).lower()
    markers = (
        "cudaerrorlaunchtimeout",
        "the launch timed out and was terminated",
        "launch timed out",
        "unspecified launch failure",
        "cudaerrorlaunchfailure",
        "cudaerrordevicelost",
        "device lost",
    )
    return any(tok in msg for tok in markers)


def predict_model_space(
    stacked: np.ndarray,
    cfg: DDSegConfig,
    *,
    allow_cpu_fallback: bool | None = None,
    execution_info: dict[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the three view models on a padded model-orientation stack."""

    import os
    import torch
    from .model_loader import load_models  # lazy: torch is only needed for inference
    from shared.environment import resolve_auto_torch_device
    from shared.hardware_adapt import empty_gpu_cache_safely, is_gpu_oom, optimal_cpu_torch_threads

    if allow_cpu_fallback is None:
        allow_cpu_fallback = str(cfg.device or "auto").strip().lower() in {"", "auto"}
    resolved_device = resolve_auto_torch_device(cfg.device)
    cfg = replace(cfg, device=resolved_device)
    actual_device = cfg.device
    fallback_from: str | None = None
    fallback_reason: str | None = None
    views = split_views(stacked)

    def _run_on_device(target_device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        prev_threads = None
        if str(target_device).startswith("cpu") and not os.environ.get("SLICERDSPACE_STAGE_CPUS"):
            prev_threads = torch.get_num_threads()
            torch.set_num_threads(optimal_cpu_torch_threads(max_conv_threads=8, server_max_threads=36))
        try:
            models = load_models(cfg.weights_dir, "dti", target_device)
            pred_axial = predict_view(models.axial, views["axial"], cfg.apply_softmax)
            pred_sagittal = predict_view(models.sagittal, views["sagittal"], cfg.apply_softmax)
            pred_coronal = predict_view(models.coronal, views["coronal"], cfg.apply_softmax)
            return pred_axial, pred_sagittal, pred_coronal
        finally:
            if prev_threads is not None:
                torch.set_num_threads(prev_threads)
            if str(target_device).startswith(("cuda", "mps")):
                empty_gpu_cache_safely()

    try:
        pred_axial, pred_sagittal, pred_coronal = _run_on_device(cfg.device)
    except RuntimeError as exc:
        if not (
            allow_cpu_fallback
            and str(cfg.device).startswith(("cuda", "mps"))
            and (is_gpu_oom(exc) or _is_cuda_tdr_or_launch_timeout(exc))
        ):
            raise
        empty_gpu_cache_safely()
        fallback_from = cfg.device
        fallback_reason = str(exc)
        actual_device = "cpu"
        print(f"[DDSeg] GPU error/OOM on {cfg.device} ({exc}); falling back to CPU", flush=True)
        pred_axial, pred_sagittal, pred_coronal = _run_on_device("cpu")
    if execution_info is not None:
        execution_info["device"] = actual_device
        execution_info["resolved_device"] = resolved_device
        if fallback_from is not None:
            execution_info["device_fallback"] = True
            execution_info["fallback_from"] = fallback_from
            execution_info["fallback_reason"] = fallback_reason
    combined = combine_views(pred_axial, pred_sagittal, pred_coronal)
    prob_maps = combined["prob_maps"]
    if not cfg.apply_softmax:
        # Preserve the legacy/default fused-logit argmax while ensuring files
        # named ProbabilisticMap_* contain actual normalized probabilities.
        prob_maps = normalize_class_scores(prob_maps)
    return combined["labels"], prob_maps


def restore_disk_orientation(
    labels: np.ndarray,
    prob_maps: np.ndarray,
    mask: OrientedMask,
) -> tuple[np.ndarray, np.ndarray]:
    """Unpad, mask and map model-orientation predictions back to the mask's voxel grid."""

    model_mask = mask.model_mask
    stacked = np.concatenate([labels[..., None].astype(np.float32), prob_maps.astype(np.float32)], axis=3)
    stacked = padding_unpadding(stacked, model_mask.shape, "unpadding")
    labels_model = np.array(stacked[..., 0])
    probs_model = np.array(stacked[..., 1:])
    labels_model[~model_mask] = 0
    probs_model[~model_mask] = 0
    return (
        from_model_orientation(labels_model, mask.affine),
        from_model_orientation(probs_model, mask.affine),
    )


def _save_like_mask(data: np.ndarray, mask: OrientedMask, path: Path) -> None:
    header = mask.image.header.copy()
    header.set_data_dtype(np.float32)
    image = nib.Nifti1Image(np.asarray(data, dtype=np.float32), mask.affine, header)
    image.set_qform(mask.affine, code=int(mask.image.header["qform_code"]) or 1)
    image.set_sform(mask.affine, code=int(mask.image.header["sform_code"]) or 1)
    nib.save(image, path.as_posix())


@scheduled_stage('segmentation', gpu=True)
def run_ddseg(cfg: DDSegConfig) -> None:
    from shared.environment import resolve_auto_torch_device
    requested_device = cfg.device
    resolved_device = resolve_auto_torch_device(cfg.device)
    cfg = replace(cfg, device=resolved_device)
    print(f"[DDSeg] Requested {requested_device}; resolved {cfg.device} in the inference runtime", flush=True)
    cfg.output_folder.mkdir(parents=True, exist_ok=True)
    (cfg.output_folder / "ddseg_qc.json").unlink(missing_ok=True)
    feature_folder = _feature_folder(cfg)

    mask = load_mask(cfg.input_mask_nii)
    feature_maps, geometry = masking_and_normalizing(feature_folder, mask, cfg.parameter_type)
    stacked = stack_features(feature_maps, cfg.parameter_type)
    allow_cpu_fallback = str(requested_device or "auto").strip().lower() in {"", "auto"}
    execution_info: dict[str, object] = {
        "requested_device": requested_device,
        "resolved_device": resolved_device,
        "device": resolved_device,
    }
    labels_model, probs_model = predict_model_space(
        stacked,
        cfg,
        allow_cpu_fallback=allow_cpu_fallback,
        execution_info=execution_info,
    )
    executed_device = str(execution_info.get("device") or cfg.device)
    cfg_for_qc = replace(cfg, device=executed_device)
    labels, prob_maps = restore_disk_orientation(labels_model, probs_model, mask)

    outputs = {
        "segmentation": cfg.output_folder / "SegmentationMap_GMWMCSF.nii.gz",
        "gm_probability": cfg.output_folder / "ProbabilisticMap_GM.nii.gz",
        "wm_probability": cfg.output_folder / "ProbabilisticMap_WM.nii.gz",
        "csf_probability": cfg.output_folder / "ProbabilisticMap_CSF.nii.gz",
    }
    _save_like_mask(labels, mask, outputs["segmentation"])
    _save_like_mask(prob_maps[..., 2], mask, outputs["gm_probability"])
    _save_like_mask(prob_maps[..., 1], mask, outputs["wm_probability"])
    _save_like_mask(prob_maps[..., 3], mask, outputs["csf_probability"])
    qc_geometry = {**geometry, "feature_dir": Path(feature_folder).resolve().as_posix()}
    write_ddseg_qc_record(
        cfg_for_qc,
        labels=labels,
        probability_maps=prob_maps,
        outputs=outputs,
        geometry=qc_geometry,
    )
    qc_path = cfg.output_folder / "ddseg_qc.json"
    if qc_path.is_file():
        record = json.loads(qc_path.read_text(encoding="utf-8"))
        record["device"] = executed_device
        record["requested_device"] = requested_device
        record["resolved_device"] = resolved_device
        if execution_info.get("fallback_from"):
            record["device_fallback"] = True
            record["fallback_from"] = execution_info["fallback_from"]
            record["fallback_reason"] = execution_info.get("fallback_reason")
        tmp_path = qc_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp_path.replace(qc_path)
