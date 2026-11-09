from __future__ import annotations

import numpy as np
from typing import Dict, List

def _softmax(x: np.ndarray) -> np.ndarray:
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)


def _softmax_nhwc(x: np.ndarray) -> np.ndarray:
    exp = np.exp(x - np.max(x, axis=3, keepdims=True))
    return exp / np.sum(exp, axis=3, keepdims=True)


def normalize_class_scores(scores: np.ndarray) -> np.ndarray:
    """Convert a final NHWC class-score volume into probabilities."""

    if scores.ndim != 4:
        raise ValueError(f"Expected a 4D NHWC score volume, got shape {scores.shape}")
    return _softmax_nhwc(scores)


def predict_view(model, slices: List[np.ndarray], apply_softmax: bool) -> np.ndarray:
    scores = []
    cached_bg_score = None
    for slc in slices:
        is_uniform_bg = slc.shape[0] >= 64 and bool(np.all(slc == -100.0))
        if is_uniform_bg and cached_bg_score is not None:
            scores.append(cached_bg_score.copy())
            continue
        if model.input_layout == "NHWC":
            x = np.expand_dims(slc, axis=0).astype(np.float32)
        else:
            x = np.expand_dims(slc, axis=0).transpose(0, 3, 1, 2).astype(np.float32)
        y = model.forward(x)
        if model.output_layout == "NHWC":
            prob = _softmax_nhwc(y) if apply_softmax else y
            score = prob[0]
        else:
            prob = _softmax(y) if apply_softmax else y
            score = prob[0].transpose(1, 2, 0)
        if is_uniform_bg:
            cached_bg_score = score.copy()
        scores.append(score)
    return np.stack(scores, axis=2)


def combine_views(pred_axial: np.ndarray, pred_sagittal: np.ndarray, pred_coronal: np.ndarray) -> Dict[str, np.ndarray]:
    pred_sagittal = np.transpose(pred_sagittal, (2, 0, 1, 3))
    pred_coronal = np.transpose(pred_coronal, (0, 2, 1, 3))
    prob_sum = pred_axial + pred_sagittal + pred_coronal
    prob_maps = prob_sum / 3.0
    # MATLAB subtracts 1 because its indices are 1-based; NumPy argmax is 0-based already.
    labels = np.argmax(prob_maps, axis=3).astype(np.int16)
    return {"prob_maps": prob_maps, "labels": labels}

