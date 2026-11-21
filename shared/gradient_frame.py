"""World (RAS) frame of diffusion gradient directions: FSL bvecs, NIfTI affines and NRRD headers.

Standard library and numpy only. The rule is held here once (gradient frame fix, batch 2):

* FSL bvecs ``b`` of a NIfTI with affine 3 x 3 part ``A``: ``R`` = ``A`` with each column normalised,
  ``F`` = diag(-1, 1, 1) when det(A) > 0, else the identity; the world (RAS) direction is ``g = R F b``.
  This equals MRtrix ``-fslgrad``.
* An NRRD header stores the measurement-frame vectors as matrix COLUMNS (``MF``); a ``DWMRI_gradient_*``
  value ``g_file`` has the world direction ``S MF g_file``, where ``S`` maps the header ``space`` to RAS.

Sheared affines and NIfTI files whose qform and sform disagree are out of scope: the functions use the
affine they are given and normalise its columns.
"""

from __future__ import annotations

import math

import numpy as np

FSL_FLIP = np.diag([-1.0, 1.0, 1.0])

# Header ``space`` value -> the matrix that maps that space to RAS (each is its own inverse).
NRRD_SPACE_TO_RAS = {
    "right-anterior-superior": np.diag([1.0, 1.0, 1.0]),
    "ras": np.diag([1.0, 1.0, 1.0]),
    "left-posterior-superior": np.diag([-1.0, -1.0, 1.0]),
    "lps": np.diag([-1.0, -1.0, 1.0]),
    "left-anterior-superior": np.diag([-1.0, 1.0, 1.0]),
    "las": np.diag([-1.0, 1.0, 1.0]),
}

# A NRRD gradient whose norm is at most this value is a b = 0 entry and stays as written.
ZERO_GRADIENT_NORM = 1e-12

# A converter's directions that are within this angle of the rule are kept as written (byte-identical outputs).
# It is the acceptance level of the route tests, not a science threshold.
KEEP_TOLERANCE_DEG = 0.01

# Off-diagonal tolerance of ``is_signed_diagonal``: a direction error below about 0.001 degree.
SIGNED_DIAGONAL_TOLERANCE = 1e-5


def _matrix3(affine) -> np.ndarray:
    matrix = np.asarray(affine, dtype=np.float64)
    if matrix.shape not in ((3, 3), (4, 4)):
        raise ValueError(f"Expected a 3 x 3 or 4 x 4 affine, got shape {matrix.shape}")
    matrix = matrix[:3, :3]
    if not np.isfinite(matrix).all():
        raise ValueError("The affine is not finite")
    return matrix


def _rows(vectors) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"Expected directions as an (N, 3) array, got shape {array.shape}")
    return array


def affine_rotation(affine) -> np.ndarray:
    """``R``: the 3 x 3 part of ``affine`` with each column normalised (the voxel size removed)."""
    matrix = _matrix3(affine)
    norms = np.linalg.norm(matrix, axis=0)
    if np.any(norms <= 0.0):
        raise ValueError("The affine has a zero-length axis")
    return matrix / norms


def determinant_positive(affine) -> bool:
    """True when det(A) > 0: FSL then negates the first bvec component (radiological voxel order)."""
    return bool(np.linalg.det(_matrix3(affine)) > 0.0)


def fsl_flip(affine) -> np.ndarray:
    """``F``: diag(-1, 1, 1) when det(A) > 0, else the identity."""
    return FSL_FLIP.copy() if determinant_positive(affine) else np.eye(3)


def voxel_axis_bvecs(bvecs, affine) -> np.ndarray:
    """``F b`` as an (N, 3) array: the directions in the image-axis frame (i, j, k).

    A consumer that maps directions to the world through the affine (DIPY tracking, peaks, voxel-axis
    direction maps) needs these vectors. For det(A) < 0 the input array is returned unchanged (same object
    values, no arithmetic), so results for those images stay bit-identical.
    """
    rows = _rows(bvecs)
    if not determinant_positive(affine):
        return rows
    flipped = rows.copy()
    flipped[:, 0] = -flipped[:, 0]
    return flipped


def fsl_to_world(bvecs, affine) -> np.ndarray:
    """World (RAS) directions ``R F b`` as an (N, 3) array; the norm of each row is kept."""
    rows = _rows(bvecs)
    return (affine_rotation(affine) @ fsl_flip(affine) @ rows.T).T


def world_to_fsl(world, affine) -> np.ndarray:
    """The inverse rule: FSL bvecs ``F R^-1 g`` for world (RAS) directions ``g`` and the target affine."""
    rows = _rows(world)
    return (fsl_flip(affine) @ np.linalg.inv(affine_rotation(affine)) @ rows.T).T


def nrrd_space_to_ras(space: str) -> np.ndarray:
    key = str(space or "").strip().lower()
    if key not in NRRD_SPACE_TO_RAS:
        raise ValueError(f"Unsupported NRRD space for diffusion gradients: {space!r}")
    return NRRD_SPACE_TO_RAS[key].copy()


def nrrd_to_world(gradients, space: str, measurement_frame) -> np.ndarray:
    """World (RAS) vectors ``S MF g_file`` of NRRD gradients; ``measurement_frame`` has the vectors as columns."""
    frame = np.asarray(measurement_frame, dtype=np.float64).reshape(3, 3)
    return (nrrd_space_to_ras(space) @ frame @ _rows(gradients).T).T


def world_to_nrrd(world, space: str, measurement_frame) -> np.ndarray:
    """NRRD header vectors ``MF^-1 S g`` that a reader maps back to the world (RAS) vectors ``g``."""
    frame = np.asarray(measurement_frame, dtype=np.float64).reshape(3, 3)
    return (np.linalg.inv(frame) @ nrrd_space_to_ras(space) @ _rows(world).T).T


def is_signed_diagonal(affine, tolerance: float = SIGNED_DIAGONAL_TOLERANCE) -> bool:
    """True when ``R`` is a diagonal matrix of +-1 within ``tolerance`` (no rotation, no axis permutation)."""
    rotation = affine_rotation(affine)
    off_diagonal = rotation - np.diag(np.diag(rotation))
    return bool(np.max(np.abs(off_diagonal)) <= tolerance)


def axis_angle_degrees(first, second) -> np.ndarray:
    """Per-row angle in degrees between two (N, 3) direction arrays; g and -g are the same axis."""
    a, b = _rows(first), _rows(second)
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cosine = np.ones(len(a))
    valid = norms > 0.0
    cosine[valid] = np.abs(np.sum(a[valid] * b[valid], axis=1)) / norms[valid]
    return np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0)))


def format_nrrd_vector(vector) -> str:
    """One ``DWMRI_gradient_*`` value in the style DWIConvert writes (``repr`` floats, three spaces)."""
    return "   ".join(repr(float(value)) for value in np.asarray(vector, dtype=np.float64).reshape(3))


def parse_nrrd_vector(text: str) -> np.ndarray:
    values = np.array([float(part) for part in str(text).split()], dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError(f"Invalid NRRD gradient value: {text!r}")
    return values


def check_nrrd_gradients_match_fsl(values, bvals, bvecs, *, norm_tolerance: float = 1e-3) -> None:
    """Fail unless the NRRD gradient values are the bval/bvec rows in the same count and order.

    A nonzero value must have the norm sqrt(b / max b) (the b-value scaling DWIConvert writes) and a
    nonzero bvec row. A zero value is a b = 0 entry and is not checked further.
    """
    bvals = np.asarray(bvals, dtype=np.float64).reshape(-1)
    rows = _rows(bvecs)
    values = list(values)
    if not (len(values) == len(bvals) == len(rows)):
        raise ValueError(
            f"Gradient count differs: the NRRD header has {len(values)} gradients, the bval file "
            f"{len(bvals)} values and the bvec file {len(rows)} directions")
    max_b = float(bvals.max()) if len(bvals) else 0.0
    for index, text in enumerate(values):
        norm = float(np.linalg.norm(parse_nrrd_vector(text)))
        if norm <= ZERO_GRADIENT_NORM:
            continue
        expected = math.sqrt(max(float(bvals[index]), 0.0) / max_b) if max_b > 0 else 0.0
        if abs(norm - expected) > norm_tolerance or float(np.linalg.norm(rows[index])) <= ZERO_GRADIENT_NORM:
            raise ValueError(
                f"NRRD gradient {index} does not match bval/bvec row {index} (norm {norm:.6f}, expected "
                f"{expected:.6f}): the gradient order differs from the bvec file")


def nrrd_gradient_values_for_world(values, world, space: str, measurement_frame) -> list[str]:
    """New ``DWMRI_gradient_*`` values: the direction of ``world[i]`` in the header frame, the norm of ``values[i]``.

    A zero value stays as written (b = 0). A nonzero value with a zero world direction is an error.
    """
    values = list(values)
    world = _rows(world)
    if len(values) != len(world):
        raise ValueError(f"Gradient count differs: {len(values)} header values, {len(world)} directions")
    result = []
    for index, (text, direction) in enumerate(zip(values, world)):
        norm = float(np.linalg.norm(parse_nrrd_vector(text)))
        if norm <= ZERO_GRADIENT_NORM:
            result.append(text)
            continue
        length = float(np.linalg.norm(direction))
        if length <= ZERO_GRADIENT_NORM:
            raise ValueError(f"NRRD gradient {index} is nonzero but its bvec row is zero")
        in_frame = world_to_nrrd(direction / length, space, measurement_frame)[0]
        result.append(format_nrrd_vector(in_frame / np.linalg.norm(in_frame) * norm))
    return result


def splice_nrrd_gradient_lines(content: bytes, values) -> bytes:
    """Replace the ``DWMRI_gradient_NNNN:=`` lines of an NRRD file's header; every other byte stays as written.

    ``values`` is the list of new values in index order. The header ends at the first blank line; an attached
    payload after it is copied unchanged.
    """
    values = list(values)
    head, separator, payload = content.partition(b"\n\n")
    if not separator:
        raise ValueError("The NRRD file has no header terminator")
    lines = head.split(b"\n")
    seen = set()
    for number, line in enumerate(lines):
        if not line.startswith(b"DWMRI_gradient_"):
            continue
        key, _, _ = line.partition(b":=")
        index = int(key[len(b"DWMRI_gradient_"):])
        if index >= len(values) or index in seen:
            raise ValueError(f"Unexpected NRRD gradient key: {key.decode(errors='replace')}")
        seen.add(index)
        lines[number] = key + b":=" + values[index].encode("ascii")
    if len(seen) != len(values):
        raise ValueError(f"The NRRD header has {len(seen)} gradient lines, expected {len(values)}")
    return b"\n".join(lines) + separator + payload
