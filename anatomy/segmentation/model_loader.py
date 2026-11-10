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
        tensor = torch.from_numpy(x).to(self.device)
        with torch.inference_mode():
            out = self.module(tensor)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out.cpu().numpy()


@dataclass
class ModelBundle:
    axial: InferenceModel
    sagittal: InferenceModel
    coronal: InferenceModel


def load_torch_model(path: Path, device: str) -> TorchScriptModel:
    if not path.exists():
        raise FileNotFoundError(f"Missing model: {path}")
    module = torch.jit.load(path.as_posix(), map_location=device)
    module.eval()
    return TorchScriptModel(module=module, device=device)


def _pick_model(weights_dir: Path, base: str, device: str) -> InferenceModel:
    pt_path = weights_dir / f"{base}.pt"
    return load_torch_model(pt_path, device)


def load_models(weights_dir: Path, prefix: str, device: str) -> ModelBundle:
    axial = _pick_model(weights_dir, f"{prefix}_axial", device)
    sagittal = _pick_model(weights_dir, f"{prefix}_sagittal", device)
    coronal = _pick_model(weights_dir, f"{prefix}_coronal", device)
    return ModelBundle(axial=axial, sagittal=sagittal, coronal=coronal)
