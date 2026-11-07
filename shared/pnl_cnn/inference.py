"""Original normalization, slice orientation, and weighted-view mask contract."""

import os

import numpy as np
import torch

from .weights import load_model

VIEWS = ("sagittal", "coronal", "axial")


def select_device(requested="auto"):
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("Device must be auto, cpu or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in the selected PyTorch runtime")
    resolved = "cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return {"requested": requested, "resolved": resolved, "framework": "pytorch",
            "torch_version": torch.__version__, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "reason": "First caller-visible CUDA device" if resolved == "cuda" else "CPU selected or CUDA unavailable"}


def normalize(volume, percentile=99):
    data = np.asarray(volume, dtype=np.float32)
    if data.shape != (256, 256, 256) or not np.isfinite(data).all():
        raise ValueError("Registered b0 must be finite on the IIT 256^3 grid")
    if not 0 < percentile <= 100:
        raise ValueError("Percentile must be in (0,100]")
    scale = np.percentile(data, percentile)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Normalization percentile must be positive")
    # NumPy 2 otherwise promotes the NumPy float64 percentile scalar. Keras
    # received float32 in the author's NumPy 1.x pipeline.
    data = np.divide(data, np.float32(scale), dtype=np.float32)
    data[data > 1] = 1
    data[data < 0] = 0
    return data


def orient(volume, view):
    if view not in VIEWS:
        raise ValueError("Unknown anatomical view")
    return np.swapaxes(volume, 0, VIEWS.index(view))


def aggregate(probabilities):
    # Keep the author's float32 operation order and strict threshold.
    fused = np.add(np.add(np.multiply(probabilities["sagittal"], 0.1),
                          np.multiply(probabilities["coronal"], 0.4)),
                   np.multiply(probabilities["axial"], 0.5))
    return fused, (fused > 0.45).astype(np.float32)


@torch.inference_mode()
def predict(volume, model_dir, *, device="cpu", batch_size=1, all_outputs=False):
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    if volume.shape != (256, 256, 256) or not np.isfinite(volume).all():
        raise ValueError("Expected a finite normalized IIT-grid volume")
    probabilities = {}
    for view in VIEWS:
        print("[PNL Torch] Predicting " + view, flush=True)
        model = load_model(model_dir, view, device)
        slices = orient(volume, view)
        output = np.empty(volume.shape, dtype=np.float32)
        for start in range(0, 256, batch_size):
            inputs = torch.from_numpy(np.ascontiguousarray(slices[start:start + batch_size, None])).to(device)
            result = model(inputs, segmentation_only=not all_outputs)
            if all_outputs:
                result = result[0]
            output[start:start + len(inputs)] = result[:, 0].cpu().numpy()
        if not np.isfinite(output).all():
            raise RuntimeError("CNN produced nonfinite probabilities")
        probabilities[view] = orient(output, view)
        del model
    fused, mask = aggregate(probabilities)
    return probabilities, fused, mask
