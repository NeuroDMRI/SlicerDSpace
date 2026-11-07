from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

from shared.external_process import find_pythonslicer, slicer_runtime_roots
from shared.teem_dti import estimate_command


def _launcher(cli: Path, slicer_exe: Optional[Path]) -> list[str]:
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

    for cmd in [dwi_convert_cmd, tensor_cmd]:
        subprocess.run(cmd, check=True)

    if not dti_e3.exists():
        raise RuntimeError("DTI parameter extraction failed. Check Slicer CLI paths.")
