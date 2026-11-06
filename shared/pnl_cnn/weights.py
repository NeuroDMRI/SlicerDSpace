# Modified version of PNLBrainMask-Torch @ 8d201a5 (Brigham and Women's Hospital licence, see shared/pnl_cnn/LICENSE.txt); modified by SlicerDSpace: only the checked weight loader is kept (constants, sha256, check, load_model); download, acquire_author_weights and convert are removed with their imports.
"""Checked author assets and deterministic, TensorFlow-free weight conversion."""

import hashlib
import json
from pathlib import Path

from .model import CompNet

SOURCE_COMMIT = "d7fe0fd4976c2d5a3489526f75d2a00dae3e1fe1"
SOURCE_URL = "https://github.com/pnlbwh/CNN-Diffusion-MRIBrain-Segmentation"
MODEL_URL = SOURCE_URL + "/releases/download/v0.3/model_folder.tar.gz"
MODEL_SHA256 = "c67210fd59e5bcaf8e0e730bdde9d8f28b817148a74d9586a39c9b6e86df5191"
TEMPLATE_URL = "https://www.nitrc.org/frs/download.php/11290/IITmean_b0_256.nii.gz"
TEMPLATE_SHA256 = "2bb1a7bbe448702033a515e76164df6472828422ecef5cb4f996d91ec126dead"
ARCHITECTURE_SHA256 = "989fd5a64d9307268441c3525d59b31d59c20db864f0262ef53ad53ec495bdae"
AUTHOR_FILES = {
    "axial": ("weights-axial-improvement-08.h5", "e11d3c410769c68844109a86eed40ebedc655ea03c14f6f54f4f13749c5ead18"),
    "coronal": ("weights-coronal-improvement-08.h5", "248dcf90752d4e0a2e22a136c5f0ca6d3c9cb9b628d131ce869d9ef6cbea702c"),
    "sagittal": ("weights-sagittal-improvement-09.h5", "27aaafddac65617e0653c4da4fd0c650529fcc8150b4efd6b66135070ebc01bc"),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(path, expected):
    if sha256(path) != expected:
        raise ValueError("SHA256 mismatch: " + str(path))


def load_model(directory, view, device="cpu"):
    import torch

    directory = Path(directory)
    receipt = json.loads((directory / "weights.json").read_text())
    if receipt["format"] != 1 or receipt["architecture_sha256"] != ARCHITECTURE_SHA256:
        raise ValueError("Unrecognized converted model")
    record = receipt["views"][view]
    if record["file"] != f"{view}.pt" or record["source_sha256"] != AUTHOR_FILES[view][1]:
        raise ValueError("Unexpected model provenance")
    path = directory / f"{view}.pt"
    check(path, record["sha256"])
    model = CompNet()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.to(device).eval()
