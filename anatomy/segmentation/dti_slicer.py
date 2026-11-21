from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

from shared.external_process import find_pythonslicer, slicer_runtime_roots, subprocess_no_window_kwargs
from shared.teem_dti import estimate_command


import sys


def _launcher(cli: Path, slicer_exe: Optional[Path]) -> list[str]:
    if sys.platform == "darwin" and slicer_exe and slicer_exe.parent.name == "MacOS":
        return [str(cli)]
    if slicer_exe and slicer_exe.exists():
        return [str(slicer_exe), "--launch", str(cli)]
    return [str(cli)]


def _find_slicer_lib_dir(slicer_base: Path) -> Path:
    for root in slicer_runtime_roots(slicer_base):
        slicer_lib = root / "lib"
        if slicer_lib.is_dir():
            candidates = sorted(p for p in slicer_lib.iterdir() if p.is_dir() and p.name.startswith("Slicer-"))
            if candidates:
                return candidates[0]
    raise FileNotFoundError(f"No Slicer-* library directory under install/app root: {slicer_base}")


def _cli_path(directory: Path, name: str) -> Path:
    for candidate in (directory / (name + ".exe"), directory / name):
        if candidate.is_file():
            return candidate
    return directory / name


def correct_dwiconvert_gradients(dwi_nrrd: Path, nii_file: Path, bval_file: Path, bvec_file: Path) -> Optional[dict]:
    """Gradient frame (batch 2): set the DWIConvert FSLToNrrd gradient directions by the FSL rule g = R F b.

    DWIConvert is right only for axis-aligned det < 0 grids; there the file is checked and left byte-identical
    (returns None). For every other grid each gradient line gets the rule's direction in the header frame and
    keeps the norm DWIConvert wrote; only the gradient lines of the header change, the payload bytes do not.
    """
    import nibabel as nib
    import nrrd
    import numpy as np

    from shared import gradient_frame

    header = nrrd.read_header(str(dwi_nrrd))
    count = sum(1 for key in header if key.startswith("DWMRI_gradient_"))
    keys = ["DWMRI_gradient_%04d" % index for index in range(count)]
    if any(key not in header for key in keys):
        raise RuntimeError("DWIConvert gradient keys are not numbered 0000 to N-1: " + str(dwi_nrrd))
    values = [header[key] for key in keys]
    bvals = np.loadtxt(str(bval_file)).reshape(-1)
    bvecs = np.loadtxt(str(bvec_file), ndmin=2)
    if bvecs.shape == (3, len(bvals)):
        bvecs = bvecs.T
    gradient_frame.check_nrrd_gradients_match_fsl(values, bvals, bvecs)
    affine = nib.load(str(nii_file)).affine
    space = header["space"]
    frame = np.asarray(header.get("measurement frame", np.eye(3)), dtype=float).T  # pynrrd rows -> columns
    world = gradient_frame.fsl_to_world(bvecs, affine)
    written = gradient_frame.nrrd_to_world([gradient_frame.parse_nrrd_vector(v) for v in values], space, frame)
    moving = np.linalg.norm(written, axis=1) > gradient_frame.ZERO_GRADIENT_NORM
    error = float(gradient_frame.axis_angle_degrees(written[moving], world[moving]).max()) if moving.any() else 0.0
    if not gradient_frame.determinant_positive(affine) and gradient_frame.is_signed_diagonal(affine):
        if error > gradient_frame.KEEP_TOLERANCE_DEG:
            raise RuntimeError(f"DWIConvert gradients differ from the FSL rule by {error:.3f} degrees on an "
                               f"axis-aligned grid: {dwi_nrrd}")
        return None
    updated = gradient_frame.nrrd_gradient_values_for_world(values, world, space, frame)
    content = gradient_frame.splice_nrrd_gradient_lines(dwi_nrrd.read_bytes(), updated)
    staged = dwi_nrrd.with_name(dwi_nrrd.name + ".gradients")
    staged.write_bytes(content)
    os.replace(staged, dwi_nrrd)
    return dict(rule="g = R F b", dwiconvert_max_error_deg=error)


def run_dti_feature_extraction(
    nii_file: Path,
    bval_file: Path,
    bvec_file: Path,
    output_folder: Path,
    slicer_base: Optional[Path] = None,
    slicer_ext: Optional[Path] = None,  # deprecated: no longer needed
) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)
    dti_e3 = output_folder / "DTI_E3.nrrd"
    if dti_e3.exists():
        return

    if slicer_base is None:
        raise ValueError("slicer_base is required for DTI feature extraction.")
    # slicer_ext is accepted for old callers and ignored: the tensor fit uses Slicer core only (teem_dti.py).

    slicer_lib_dir = _find_slicer_lib_dir(slicer_base)
    dwi_convert = _cli_path(slicer_lib_dir / "cli-modules", "DWIConvert")
    python_slicer = find_pythonslicer(slicer_base)
    if python_slicer is None:
        raise FileNotFoundError(f"PythonSlicer not found in the Slicer install: {slicer_base}")
    slicer_exe = next((path for root in slicer_runtime_roots(slicer_base)
                       for path in (root / "Slicer.exe", root / "Slicer", root / "MacOS/Slicer")
                       if path.is_file()), None)

    dwi_nrrd = output_folder / "DWI.nrrd"
    dti_nrrd = output_folder / "DTI.nrrd"
    b0_nrrd = output_folder / "b0.nrrd"
    fa_nrrd = output_folder / "DTI_FA.nrrd"
    md_nrrd = output_folder / "DTI_MD.nrrd"
    e1_nrrd = output_folder / "DTI_E1.nrrd"
    e2_nrrd = output_folder / "DTI_E2.nrrd"
    e3_nrrd = output_folder / "DTI_E3.nrrd"

    # Argument lists passed to subprocess.run() without shell=True: no shell
    # is involved, so paths containing spaces, semicolons, or other shell
    # metacharacters are passed through literally with no quoting needed.
    dwi_convert_cmd = _launcher(dwi_convert, slicer_exe) + [
        "--conversionMode",
        "FSLToNrrd",
        "--inputBVectors",
        str(bvec_file),
        "--inputBValues",
        str(bval_file),
        "--fslNIFTIFile",
        str(nii_file),
        "--outputVolume",
        str(dwi_nrrd),
        "--allowLossyConversion",
    ]

    # One PythonSlicer process fits the tensor and writes the five scalar maps (E1 = max, E3 = min eigenvalue).
    tensor_cmd = estimate_command(
        python_slicer, dwi_nrrd, dti_nrrd, b0_nrrd, "LS",
        (("FractionalAnisotropy", fa_nrrd), ("Trace", md_nrrd), ("MaxEigenvalue", e1_nrrd),
         ("MidEigenvalue", e2_nrrd), ("MinEigenvalue", e3_nrrd)))

    subprocess.run(dwi_convert_cmd, check=True, **subprocess_no_window_kwargs())
    if dwi_nrrd.is_file():  # a missing file is reported by the fit and the final check below
        correct_dwiconvert_gradients(dwi_nrrd, nii_file, bval_file, bvec_file)
    subprocess.run(tensor_cmd, check=True, **subprocess_no_window_kwargs())

    if not dti_e3.exists():
        raise RuntimeError("DTI parameter extraction failed. Check Slicer CLI paths.")
