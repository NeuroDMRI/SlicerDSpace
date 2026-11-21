"""Cache-key pins for the gradient-frame fix (batch 2, item 8): keep finished runs where the fix is a no-op.

The native-DTI, diffusion-header and registered-DTI cache keys (shared/pipeline.py) hold the sha256 of
anatomy/parcellation/conversion/nhdr_write.py, shared/teem_dti.py and shared/pipeline.py. The gradient-frame fix
edited all three files. Its changes are no-ops for a DWI whose affine passes ``gradient_frame_fix_is_no_op``:

- nhdr_write.py flips the bvecs only when det > 0, so with det < 0 the header is byte-identical;
- teem_dti.py keeps the SlicerDMRI column-normalised RasToIjk when it equals the row-normalised one; on an
  axis-aligned grid both are the same signed permutation, so the tensor fit is bit-identical;
- the pipeline.py edit only adds this pin.

For such a DWI the key uses the base-commit (3c912ad2) sha256 in place of the current one, so old caches are
reused. Every other DWI keeps the real sha256 and is recomputed. This module is hashed by no cache key, so it can
hold the hashes of the three files.

A later edit of one of the three files changes its sha256: the table below then no longer holds it, the real
sha256 is used and the key changes as before. The stale entry is then inert: delete it (the key test in
Testing/Python/test_b2_gradient_frame.py is skipped from then on and says so). Each file has two entries: LF bytes (the repository) and CRLF bytes (a Windows
checkout with core.autocrlf, ``* text=auto`` in .gitattributes).
"""

from __future__ import annotations

from pathlib import Path

# {current sha256: base 3c912ad2 sha256}; computed from the final content of each file.
BASE_HASH_PINS = {
    # anatomy/parcellation/conversion/nhdr_write.py (LF, CRLF)
    "94edba60217da6fcbbe0bd27a6aa143b670f341d1eae5f1df1d09cf55736d972": "14a19586994a7587e8dde1a986b879ceebfca57e950d7f0d3393b85afdca506f",
    "02b1b27b3d12b6affe39fe480f45b28d432a02b2fa764f18fb2ea2d0585783ba": "e590ca9942cadb19180f9b9a6b1331834c3d4ffe683e826f567bc4bc9e82e2ce",
    # shared/teem_dti.py (LF, CRLF)
    "c2514ba7e56cb9b28c6027477f974b42c4c85b33824efe4ce1b3c9d07deb190a": "7a3058c95d33e3a446cec695e4d1edf2da980abf6dd5ce618531246b230dfa71",
    "38a2a61c21a86034f7bdfca67ed625264dbbc9bd7d305175a033b1724c378b69": "c0039fb6120ab2fc107444cdf111ec73335d237582458e0ffa5b73fd82a926fa",
    # shared/pipeline.py (LF, CRLF)
    "e4a3965e857cbc227b55dae7b24031aa9c94036147b15970e78104e201f3d018": "2427fa6d0ee34c427c643e66e1ea75a6b093ce09cccc20ca08a2288d05792cbb",
    "46363bf5edb046485c19361e26a46b182b2c565a1557695ef6635cd6d0dcb9cf": "7bc8c8a662f9bbef9b960b7deebf33005141c4e554a54eae53fcf1d271ce8bc4",
}


def gradient_frame_fix_is_no_op(dwi_path) -> bool:
    """True only when the 3 x 3 part of the DWI affine has exactly one nonzero entry in each row and each column
    (an axis-aligned grid, any axis order and voxel size) and a negative determinant.

    The affine is nibabel's ``header.get_best_affine()``, the matrix nhdr_write.py uses. No tolerance is applied:
    a grid that is off-axis by any amount counts as affected. An unreadable file or a non-finite affine counts as
    affected too, so the real sha256 is used.
    """
    try:
        import nibabel
        import numpy

        matrix = numpy.asarray(nibabel.load(str(Path(dwi_path))).header.get_best_affine(), dtype=float)[:3, :3]
        if not numpy.all(numpy.isfinite(matrix)):
            return False
        nonzero = matrix != 0.0
        if not (numpy.all(nonzero.sum(axis=0) == 1) and numpy.all(nonzero.sum(axis=1) == 1)):
            return False
        return bool(numpy.linalg.det(matrix) < 0.0)
    except Exception:
        return False


def key_sha256(digest: str, no_op: bool) -> str:
    """The sha256 that goes into a cache key: the base-commit value for a no-op DWI, else the real one."""
    return BASE_HASH_PINS.get(digest, digest) if no_op else digest
