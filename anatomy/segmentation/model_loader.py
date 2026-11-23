from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch


class InferenceModel(Protocol):
    def forward(self, x: np.ndarray) -> np.ndarray:  # x: N,C,H,W
        ...
    input_layout: str
    output_layout: str


@dataclass
class TorchScriptModel:
    module: torch.nn.Module
    device: str
    input_layout: str = "NCHW"
    output_layout: str = "NCHW"

    def forward(self, x: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(x)
        if str(self.device).lower().startswith("cpu") and tensor.ndim == 4:
            tensor = tensor.to(device=self.device, memory_format=torch.channels_last)
        else:
            tensor = tensor.to(self.device)
        with torch.inference_mode():
            out = self.module(tensor)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.contiguous().cpu().numpy()


@dataclass
class ModelBundle:
    axial: InferenceModel
    sagittal: InferenceModel
    coronal: InferenceModel


def load_torch_model(path: Path, device: str) -> TorchScriptModel:
    if not path.exists():
        raise FileNotFoundError(f"Missing model: {path}")
    from shared.hardware_adapt import (
        enable_cpu_flush_denormal,
        should_enable_cudnn_benchmark,
        zero_subnormal_module_weights,
    )

    dev_str = str(device).lower()
    if dev_str.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = should_enable_cudnn_benchmark(device)
    elif dev_str.startswith("cpu"):
        enable_cpu_flush_denormal(torch)
    module = torch.jit.load(path.as_posix(), map_location=device)
    module.eval()
    if dev_str.startswith("cpu"):
        zero_subnormal_module_weights(module)
        try:
            module = module.to(memory_format=torch.channels_last)
        except Exception:
            pass
    return TorchScriptModel(module=module, device=device)


def _pick_model(weights_dir: Path, base: str, device: str) -> InferenceModel:
    pt_path = weights_dir / f"{base}.pt"
    return load_torch_model(pt_path, device)


def load_models(weights_dir: Path, prefix: str, device: str) -> ModelBundle:
    axial = _pick_model(weights_dir, f"{prefix}_axial", device)
    sagittal = _pick_model(weights_dir, f"{prefix}_sagittal", device)
    coronal = _pick_model(weights_dir, f"{prefix}_coronal", device)
    return ModelBundle(axial=axial, sagittal=sagittal, coronal=coronal)
