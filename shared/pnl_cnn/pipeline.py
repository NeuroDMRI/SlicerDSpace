# Modified version of PNLBrainMask-Torch @ 8d201a5 (Brigham and Women's Hospital licence, see shared/pnl_cnn/LICENSE.txt); modified by SlicerDSpace: only the 3-D input branch of mean_b0 is kept (the 4-D mean, ANTs calls, run and finish_native are removed with their imports); a non-3-D input raises ValueError.
"""Modified, shell-free PNL native-grid pipeline with unchanged ANTs operations.

Original source: pnlbwh/CNN-Diffusion-MRIBrain-Segmentation, pinned in weights.py.
No implicit anatomical reorientation, optional morphology, or retraining.
"""

import numpy as np


def mean_b0(image, bvals=None, threshold=50):
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("b0 threshold must be finite and nonnegative")
    if image.ndim == 3:
        data = image.get_fdata()
    else:
        raise ValueError("Expected 3D b0 NIfTI (the 4D DWI branch is not vendored)")
    if not np.isfinite(data).all():
        raise ValueError("Input contains NaN/Inf; refusing silent sanitization")
    return np.maximum(data, 0)
