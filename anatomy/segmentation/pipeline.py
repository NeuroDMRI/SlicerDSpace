from __future__ import annotations

from shared.cohort_schedule import scheduled_stage

from dataclasses import replace
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


def predict_model_space(stacked: np.ndarray, cfg: DDSegConfig) -> tuple[np.ndarray, np.ndarray]:
    """Run the three view models on a padded model-orientation stack."""

    from .model_loader import load_models  # lazy: torch is only needed for inference
    from shared.environment import resolve_auto_torch_device

    cfg = replace(cfg, device=resolve_auto_torch_device(cfg.device))
    views = split_views(stacked)
    models = load_models(cfg.weights_dir, "dti", cfg.device)
    pred_axial = predict_view(models.axial, views["axial"], cfg.apply_softmax)
    pred_sagittal = predict_view(models.sagittal, views["sagittal"], cfg.apply_softmax)
    pred_coronal = predict_view(models.coronal, views["coronal"], cfg.apply_softmax)
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
    cfg = replace(cfg, device=resolve_auto_torch_device(cfg.device))
    print(f"[DDSeg] Requested {requested_device}; resolved {cfg.device} in the inference runtime", flush=True)
    cfg.output_folder.mkdir(parents=True, exist_ok=True)
    (cfg.output_folder / "ddseg_qc.json").unlink(missing_ok=True)
    feature_folder = _feature_folder(cfg)

    mask = load_mask(cfg.input_mask_nii)
    feature_maps, geometry = masking_and_normalizing(feature_folder, mask, cfg.parameter_type)
    stacked = stack_features(feature_maps, cfg.parameter_type)
    labels_model, probs_model = predict_model_space(stacked, cfg)
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
    write_ddseg_qc_record(
        cfg,
        labels=labels,
        probability_maps=prob_maps,
        outputs=outputs,
        geometry={**geometry, "feature_dir": Path(feature_folder).resolve().as_posix()},
    )
