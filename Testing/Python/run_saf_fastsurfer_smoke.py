#!/usr/bin/env python3
"""Run a small SAF smoke test against a FastSurfer/FreeSurfer directory.

The script can run in normal Python through ``tractography.run_saf`` or inside
Slicer through ``SlicerDSpaceLogic.runSAFPipeline``. It creates a tiny
synthetic .tck tractogram from cortex surface vertices so the SAF backend can be
exercised without MRtrix3 tractography. Input surfaces must already be in native
dMRI space.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _set_thread_defaults() -> None:
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def _make_synthetic_inputs(subject_root: Path, fs_dir: Path) -> tuple[Path, Path]:
    import nibabel as nib
    import nibabel.freesurfer.io as fsio
    import numpy as np

    backend_dir = REPO_ROOT / "tractography" / "saf_backend"
    if backend_dir.as_posix() not in sys.path:
        sys.path.insert(0, backend_dir.as_posix())
    from Filters.filters_label import filters_label
    from Support.get_surface_transform import get_surface_transform

    input_dir = subject_root / "cache" / "stages" / "03_tractography" / "saf_test_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    streamlines = []
    for hemi in ("rh", "lh"):
        white_file = fs_dir / "surf" / f"{hemi}.white"
        vertices, _ = fsio.read_geometry(white_file.as_posix())
        vertices = vertices + get_surface_transform(white_file.as_posix())[:, 3]
        used = filters_label(vertices, (fs_dir / "label" / f"{hemi}.cortex.label").as_posix())
        indices = np.flatnonzero(used)
        for offset in range(0, min(2000, len(indices) - 50), 200):
            start = vertices[indices[offset]]
            end = vertices[indices[offset + 40]]
            middle = (start + end) / 2.0 * 0.92
            streamlines.append(np.vstack([start, middle, end]).astype(np.float32))

    tck_path = input_dir / "synthetic_saf_input.tck"
    tractogram = nib.streamlines.Tractogram(streamlines, affine_to_rasmm=np.eye(4))
    nib.streamlines.save(tractogram, tck_path.as_posix())

    fod_path = input_dir / "dummy_fod.nii.gz"
    if not fod_path.exists():
        nib.save(nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), np.eye(4)), fod_path.as_posix())
    return tck_path, fod_path


def _write_dummy_brain_t1(output_dir: Path) -> None:
    import nibabel as nib
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), np.eye(4)),
        (output_dir / "brain_T1.nii.gz").as_posix(),
    )


def main() -> int:
    _set_thread_defaults()
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-root", type=Path, required=True)
    parser.add_argument("--subject-id", default="100307")
    parser.add_argument("--fs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mrtrix-bin-dir", type=Path, default=None)
    parser.add_argument("--mrtrix-prefix", default="")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--use-slicer-logic", action="store_true")
    args = parser.parse_args()

    subject_root = args.subject_root.expanduser().resolve()
    fs_dir = args.fs_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else subject_root / "cache" / "stages" / "03_tractography" / "saf_fastsurfer_smoke"
    )
    tck_path, fod_path = _make_synthetic_inputs(subject_root, fs_dir)
    _write_dummy_brain_t1(output_dir)

    if args.use_slicer_logic:
        from SlicerDSpace import SlicerDSpaceLogic

        logic = SlicerDSpaceLogic()
        result = logic.runSAFPipeline(
            subject_id=args.subject_id,
            input_root=subject_root,
            fs_dir=fs_dir,
            fod_path=fod_path,
            output_dir=output_dir,
            pregenerated_tractogram=tck_path,
            mrtrix_bin_dir=args.mrtrix_bin_dir,
            mrtrix_prefix=args.mrtrix_prefix,
            maxlen=None,
            force_recompute=True,
            execute=args.execute,
        )
    else:
        from tractography import SAFConfig, run_saf

        result = run_saf(
            SAFConfig(
                subject_id=args.subject_id,
                input_root=subject_root,
                fs_dir=fs_dir,
                fod_path=fod_path,
                output_dir=output_dir,
                pregenerated_tractogram=tck_path,
                mrtrix_bin_dir=args.mrtrix_bin_dir,
                mrtrix_prefix=args.mrtrix_prefix,
                maxlen=None,
                force_recompute=True,
                execute=args.execute,
            ),
            log=print,
        )

    print("SAF_FASTSURFER_SMOKE_OK")
    print(result.saf_tractogram)
    print(result.manifest_path)
    print(result.qc_path)
    return 0


if __name__ == "__main__":
    status = main()
    if "slicer" in sys.modules:
        import slicer

        slicer.util.exit(status)
    raise SystemExit(status)
