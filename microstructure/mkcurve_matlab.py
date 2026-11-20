"""Optional reference adapter for the authors' MATLAB MK-Curve implementation.

This is not a Python reimplementation of MK-Curve. Upstream code remains in a
separately configured checkout, with its own license and MATLAB requirements.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import uuid
from collections import deque
from pathlib import Path

from shared.external_process import external_python_environment, text_subprocess_kwargs

UPSTREAM_URL = "https://github.com/zhangfanmark/MK-Curve"
METRICS = ("mk", "ak", "rk", "fa", "md", "ad", "rd")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_mkcurve_backend(matlab_executable=None, mkcurve_root=None) -> dict:
    """Check configuration without invoking MATLAB or accepting a license."""
    executable = matlab_executable or os.environ.get("SLICERDSPACE_MKCURVE_MATLAB")
    root_value = mkcurve_root or os.environ.get("SLICERDSPACE_MKCURVE_ROOT")
    if executable:
        executable = shutil.which(str(executable)) or str(Path(executable).expanduser())
    else:
        executable = shutil.which("matlab")
    root = Path(root_value).expanduser().resolve() if root_value else None
    missing = []
    if not executable or not Path(executable).is_file() or not os.access(executable, os.X_OK):
        missing.append("MATLAB executable (R2019a or newer, with -batch support)")
    required = ("run_MKCurve.m", "prepare_data.m", "abnormal_voxel_detection.m",
                "MK_versus_synthetic_b0.m", "lib/DKI/dki_fit.m",
                "lib/functions/save_nii_parameters.m", "LICENSE")
    if root is None:
        missing.append("author MK-Curve source directory")
    else:
        missing.extend(str(root / name) for name in required if not (root / name).is_file())
    sources = {}
    if root is not None and not missing:
        sources = {path.relative_to(root).as_posix(): _hash(path)
                   for path in sorted(root.rglob("*.m")) if ".git" not in path.parts}
        sources["LICENSE"] = _hash(root / "LICENSE")
    return {"available": not missing, "missing_required": missing,
            "matlab_executable": str(executable) if executable else None,
            "mkcurve_root": str(root) if root else None, "source_sha256": sources,
            "upstream_url": UPSTREAM_URL}


def _matlab_literal(value) -> str:
    text = str(value)
    if any(char in text for char in ("\n", "\r", "\x00")):
        raise ValueError("MK-Curve paths cannot contain control characters")
    return "'" + text.replace("'", "''") + "'"


from microstructure.mkcurve_inputs import _prepare_inputs

def run_mkcurve(dwi_path, bval_path, bvec_path, mask_path, output_dir, *,
                matlab_executable=None, mkcurve_root=None, b0_threshold=50.0, max_bvalue=3000.0,
                log=None) -> dict:
    """Run original MK-Curve; expose untruncated MAT estimates, not clipped PNGs.

    Acquisition eligibility must already have passed the common DKI gate.
    Each invocation has a new work directory because upstream caches by filename.
    """
    import nibabel as nib
    import numpy as np
    from scipy.io import loadmat

    backend = inspect_mkcurve_backend(matlab_executable, mkcurve_root)
    if not backend["available"]:
        raise RuntimeError("MK-Curve unavailable: " + "; ".join(backend["missing_required"]))
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    work = output / ("author_run_" + uuid.uuid4().hex)
    work.mkdir()
    inputs, reference, preprocessing = _prepare_inputs(
        dwi_path, bval_path, bvec_path, mask_path, work, b0_threshold, max_bvalue)
    expression = ("cd(" + _matlab_literal(backend["mkcurve_root"]) + "); "
                  "run_MKCurve(" + ",".join(_matlab_literal(path) for path in (*inputs, work)) + ",0);")
    command = [backend["matlab_executable"], "-batch", expression]
    tail = deque(maxlen=50)
    log_path = work / "matlab.log"
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, env=external_python_environment(),
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   **text_subprocess_kwargs(bufsize=1))
        def on_line(line):
            handle.write(line)
            handle.flush()
            tail.append(line.rstrip())
            if log:
                log("[MK-Curve] " + line.rstrip())

        try:
            # H8: a surviving grandchild must not keep the stage open after MATLAB has exited.
            from shared.line_stream import stream_process_lines
            code = stream_process_lines(process, on_line,
                                        notice=(lambda message: log("[MK-Curve] " + message)) if log else None)
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        # stream_process_lines closes the pipe itself once its reader has finished; a surviving
        # descendant keeps it open (held by the daemon reader), so no close here.
    if code:
        raise RuntimeError(f"MK-Curve MATLAB failed ({code}): " + "\n".join(tail))
    original_mat = work / "out" / "original_parameters.mat"
    corrected_mat = work / "out" / "threshold_0.50" / "fixed_parameters.mat"
    paths = {}
    summaries = {}
    brain = np.asarray(nib.load(str(mask_path)).dataobj) > 0
    for label, mat in (("original", original_mat), ("corrected", corrected_mat)):
        if not mat.is_file():
            raise RuntimeError(f"MK-Curve did not produce expected author estimates: {mat}")
        arrays = loadmat(mat)
        for metric in METRICS:
            values = np.asarray(arrays[metric], dtype=np.float64)
            if values.shape != reference.shape[:3]:
                raise ValueError(f"MK-Curve {label} {metric} shape disagrees with source DWI")
            values[~brain] = 0
            path = work / f"{label}_{metric.upper()}_untruncated.nii.gz"
            header = reference.header.copy()
            header.set_data_dtype(np.float32)
            nib.save(nib.Nifti1Image(values.astype(np.float32), reference.affine, header), str(path))
            key = f"{label}_{metric}"
            paths[key] = str(path)
            valid = np.isfinite(values[brain])
            summaries[key] = {"brain_voxels": int(brain.sum()), "nonfinite_count": int((~valid).sum()),
                              "negative_count": int((values[brain][valid] < 0).sum())}
    record = {"kind": "mkcurve_author_adapter", "version": 1, "status": "completed",
              "method": "original MATLAB MK-Curve; threshold=0.5", "backend": backend,
              "command": command, "log": str(log_path), "preprocessing": preprocessing,
              "source_files": {str(Path(path).resolve()): _hash(Path(path))
                               for path in (dwi_path, bval_path, bvec_path, mask_path)},
              "maps": paths, "summary": summaries, "original_mat": str(original_mat),
              "corrected_mat": str(corrected_mat), "human_review": "pending",
              "automatic_qc_passed": all(item["nonfinite_count"] == 0 for item in summaries.values()),
              "notes": ["Raw MAT estimates exported without upstream NIfTI truncation.",
                        "KFA/MKT/RTK are not supplied by this author route; no substitute computed.",
                        "Upstream corrected NIfTI and working DWI remain separate in author_run."]}
    record_path = work / "adapter_record.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    record["record_path"] = str(record_path)
    return record
