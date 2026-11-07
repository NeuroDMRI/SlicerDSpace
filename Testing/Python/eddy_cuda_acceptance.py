"""Opt-in real MRtrix/FSL preprocessing acceptance in a fresh output directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from preprocessing.pipeline import DwiPreprocessingConfig, run_dwi_preprocessing
from shared.pipeline import find_subject_root


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-root", type=Path, required=True)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mrtrix-bin-dir", type=Path, required=True)
    parser.add_argument("--fsl-bin-dir", type=Path, required=True)
    parser.add_argument("--phase-encoding-direction", required=True)
    parser.add_argument("--readout-time", type=float, required=True)
    parser.add_argument("--reverse-phase-b0", type=Path)
    parser.add_argument("--eddy-slm", choices=("none", "linear", "quadratic"), default="none")
    parser.add_argument("--eddy-threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("Use a new output directory; this acceptance never overwrites earlier results")
    args.output_dir.mkdir(parents=True)
    if args.device == "cpu":
        # An explicit CPU benchmark must not select the installed CUDA executable.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    receipt = {"kind": "eddy_cuda_acceptance", "status": "running",
               "config": {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()},
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "manual_scientific_review": "pending"}
    start = time.perf_counter()
    try:
        import nibabel as nib
        import numpy as np

        raw = find_subject_root(args.subject_root, args.subject_id, generate_missing_mask=False)
        sources = [raw.dwi_path, raw.bval_path, raw.bvec_path, raw.mask_path]
        if args.reverse_phase_b0:
            sources.append(args.reverse_phase_b0)
        receipt["source_sha256"] = {str(path): digest(path) for path in sources}
        result = run_dwi_preprocessing(DwiPreprocessingConfig(
            subject_id=args.subject_id, input_root=args.subject_root,
            output_dir=args.output_dir / "preprocessing", execute=True,
            mrtrix_bin_dir=args.mrtrix_bin_dir, fsl_bin_dir=args.fsl_bin_dir,
            phase_encoding_direction=args.phase_encoding_direction,
            readout_time=args.readout_time, reverse_phase_b0=args.reverse_phase_b0,
            eddy_threads=args.eddy_threads, eddy_second_level_model=args.eddy_slm,
            brain_mask_backend="pnl_cnn",
        ), log=print)
        receipt["runtime_seconds"] = time.perf_counter() - start
        qc = json.loads(result.qc_path.read_text())
        device = qc["backend_details"]["motion"]["device_execution"]
        image = nib.load(result.final_dwi_path)
        source_image = nib.load(raw.dwi_path)
        data = image.get_fdata(dtype=np.float32)
        bvals = np.loadtxt(result.final_bval_path).reshape(-1)
        bvecs = np.loadtxt(result.final_bvec_path)
        norms = np.linalg.norm(bvecs, axis=0)
        dw = bvals > 50
        checks = {
            "pipeline_completed": result.status == "completed",
            f"{args.device}_selected": device["resolved"] == args.device,
            "no_cpu_retry": device["cuda_error_cpu_retry"] is False,
            "dwi_same_shape": image.shape == source_image.shape,
            "dwi_same_world_grid": bool(np.allclose(image.affine, source_image.affine)),
            "dwi_finite": bool(np.isfinite(data).all()),
            "dwi_nonempty": bool(np.any(data > 0)),
            "gradient_count": bvals.size == image.shape[3] and bvecs.shape == (3, image.shape[3]),
            "gradients_finite": bool(np.isfinite(bvals).all() and np.isfinite(bvecs).all()),
            "dw_gradients_unit_norm": bool(dw.any() and np.allclose(norms[dw], 1, atol=0.01)),
            "mask_native_grid": qc["brain_mask_qc"]["geometry_pass"] is True,
            "inputs_unchanged": receipt["source_sha256"] == {str(path): digest(path) for path in sources},
        }
        receipt.update(status="PASS" if all(checks.values()) else "FAIL", checks=checks,
                       device_execution=device, manifest=str(result.manifest_path), qc=str(result.qc_path))
        receipt["output_sha256"] = {str(path): digest(path) for path in (
            result.final_dwi_path, result.final_bval_path, result.final_bvec_path, result.final_mask_path)}
    except Exception as exc:
        receipt.update(status="FAIL", runtime_seconds=time.perf_counter() - start,
                       error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    (args.output_dir / "acceptance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
