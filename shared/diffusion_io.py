"""Bounded-memory sequential decompression for repeated DWI chunk reads."""

from contextlib import contextmanager
import gzip
from pathlib import Path
import shutil
import tempfile


@contextmanager
def diffusion_image(path, work_dir):
    import nibabel as nib
    import numpy as np

    path = Path(path)
    image = nib.load(str(path), mmap="r", keep_file_open=False)
    if not path.name.endswith(".nii.gz"):
        yield image
        return
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    needed = int(np.prod(image.shape)) * image.dataobj.dtype.itemsize + int(image.dataobj.offset)
    if shutil.disk_usage(work_dir).free < needed + 16 * 1024 * 1024:
        raise RuntimeError(f"Insufficient temporary disk space for one-pass DWI decompression ({needed} bytes required)")
    with tempfile.TemporaryDirectory(prefix=".dwi-read-", dir=work_dir) as temporary:
        uncompressed = Path(temporary) / "data.nii"
        from shared.cohort_io import io_admission
        with io_admission('dwi_decompression'), gzip.open(path, "rb") as source, uncompressed.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
        yield nib.load(str(uncompressed), mmap="r", keep_file_open=False)
