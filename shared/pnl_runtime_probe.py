"""Validate converted PNL weights in their external PyTorch interpreter."""

import hashlib
import json
from pathlib import Path
import sys


def main():
    import nibabel as nib
    import torch
    import pnl_brain_mask
    from pnl_brain_mask.weights import load_model

    if pnl_brain_mask.__version__ != "0.1.0":
        raise RuntimeError("PNL requires the reviewed pnl-brain-mask 0.1.0 package")
    version = tuple(map(int, torch.__version__.split("+")[0].split(".")[:2]))
    if not (2, 4) <= version < (3, 0):
        raise RuntimeError("PNL requires compatible PyTorch >=2.4,<3")
    torch.set_num_threads(1)
    root = Path(sys.argv[1])
    if nib.load(str(root / "IITmean_b0_256.nii.gz")).shape != (256, 256, 256):
        raise RuntimeError("Unexpected PNL template grid")
    with torch.inference_mode():
        for view in ("axial", "coronal", "sagittal"):
            model = load_model(root, view, "cpu")
            outputs = model(torch.zeros((1, 1, 256, 256)))
            if not all(torch.isfinite(value).all() for value in outputs):
                raise RuntimeError("PNL forward check produced nonfinite values")
    package = Path(pnl_brain_mask.__file__).parent
    sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(package.glob("*.py"))}
    print(json.dumps({"framework": "pytorch", "torch_version": torch.__version__,
                      "package_version": pnl_brain_mask.__version__, "device": "cpu",
                      "package_source_sha256": sources, "forward_checks": "three_views_three_heads"}))


if __name__ == "__main__":
    main()
