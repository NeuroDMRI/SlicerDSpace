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


def _uniform_slice_key(slc: np.ndarray):
    if not isinstance(slc, np.ndarray) or slc.ndim < 2 or slc.size == 0:
        return None
    pixel = slc[(0,) * (slc.ndim - 1)]
    if not np.isfinite(pixel).all() or not np.array_equal(slc, np.broadcast_to(pixel, slc.shape)):
        return None
    return (slc.shape, slc.dtype.str, np.ascontiguousarray(pixel).tobytes())


def _decode_batch_output(y: np.ndarray, output_layout: str, apply_softmax: bool) -> np.ndarray:
    if output_layout == "NHWC":
        return _softmax_nhwc(y) if apply_softmax else y
    prob = _softmax(y) if apply_softmax else y
    return prob.transpose(0, 2, 3, 1)


def predict_view(model, slices: List[np.ndarray], apply_softmax: bool) -> np.ndarray:
    if not slices:
        return np.empty((0,), dtype=np.float32)

    uniform_cache: Dict[tuple, np.ndarray] = {}
    slice_keys = [_uniform_slice_key(slc) for slc in slices]
    infer_indices: List[int] = []
    pending_uniform_keys: Dict[tuple, int] = {}
    for idx, key in enumerate(slice_keys):
        if key is None:
            infer_indices.append(idx)
        elif key not in pending_uniform_keys:
            pending_uniform_keys[key] = len(infer_indices)
            infer_indices.append(idx)

    inferred_scores: Dict[int, np.ndarray] = {}
    batch_size = 8
    dev_str = str(getattr(model, "device", "cpu")).lower()
    prev_cpu_threads = None
    if dev_str.startswith("cpu"):
        try:
            import torch
            from shared.hardware_adapt import enable_cpu_flush_denormal, optimal_cpu_torch_threads

            enable_cpu_flush_denormal(torch)
            target_threads = optimal_cpu_torch_threads(max_conv_threads=8, server_max_threads=36)
            current_threads = int(torch.get_num_threads())
            if current_threads > target_threads:
                prev_cpu_threads = current_threads
                torch.set_num_threads(target_threads)
            if target_threads >= 24:
                batch_size = 16
        except Exception:
            pass
    if dev_str.startswith(("cuda", "mps")):
        from shared.hardware_adapt import query_gpu_free_total_bytes

        mem = query_gpu_free_total_bytes(dev_str)
        if mem is not None:
            min_bytes = int(4.0 * (1024**3)) if dev_str.startswith("mps") else int(2.0 * (1024**3))
            if mem[0] < int(1.2 * (1024**3)):
                batch_size = 1
            elif mem[0] < min_bytes:
                batch_size = 2

    batch_supported = True
    pos = 0
    try:
        while pos < len(infer_indices):
            step = batch_size if batch_supported else 1
            chunk_idx = infer_indices[pos : pos + step]
            arr = np.stack([slices[i] for i in chunk_idx], axis=0).astype(np.float32, copy=False)
            x = arr if model.input_layout == "NHWC" else arr.transpose(0, 3, 1, 2)
            if len(chunk_idx) > 1:
                try:
                    y = model.forward(np.ascontiguousarray(x))
                    if y.shape[0] != len(chunk_idx):
                        raise ValueError("Model did not preserve batch dimension")
                    decoded = _decode_batch_output(y, model.output_layout, apply_softmax)
                    for local_i, orig_i in enumerate(chunk_idx):
                        inferred_scores[orig_i] = decoded[local_i]
                    pos += len(chunk_idx)
                    continue
                except Exception as exc:
                    from shared.hardware_adapt import empty_gpu_cache_safely, is_gpu_oom

                    empty_gpu_cache_safely()
                    if dev_str.startswith(("cuda", "mps")) and is_gpu_oom(exc) and batch_size > 2:
                        batch_size = max(1, batch_size // 2)
                    else:
                        batch_supported = False
                    continue

            y1 = model.forward(np.ascontiguousarray(x))
            decoded1 = _decode_batch_output(y1, model.output_layout, apply_softmax)
            inferred_scores[chunk_idx[0]] = decoded1[0]
            pos += 1
    finally:
        if prev_cpu_threads is not None:
            try:
                import torch

                torch.set_num_threads(prev_cpu_threads)
            except Exception:
                pass
        if dev_str.startswith(("cuda", "mps")):
            from shared.hardware_adapt import empty_gpu_cache_safely

            empty_gpu_cache_safely()

    for key, infer_pos in pending_uniform_keys.items():
        if len(uniform_cache) < 16:
            uniform_cache[key] = inferred_scores[infer_indices[infer_pos]]

    scores = []
    for idx, key in enumerate(slice_keys):
        if key is not None and key in uniform_cache:
            scores.append(uniform_cache[key].copy())
        else:
            scores.append(inferred_scores[idx])
    return np.stack(scores, axis=2)


def combine_views(pred_axial: np.ndarray, pred_sagittal: np.ndarray, pred_coronal: np.ndarray) -> Dict[str, np.ndarray]:
    pred_sagittal = np.transpose(pred_sagittal, (2, 0, 1, 3))
    pred_coronal = np.transpose(pred_coronal, (0, 2, 1, 3))
    prob_sum = pred_axial + pred_sagittal + pred_coronal
    prob_maps = prob_sum / 3.0
    # MATLAB subtracts 1 because its indices are 1-based; NumPy argmax is 0-based already.
    labels = np.argmax(prob_maps, axis=3).astype(np.int16)
    return {"prob_maps": prob_maps, "labels": labels}
