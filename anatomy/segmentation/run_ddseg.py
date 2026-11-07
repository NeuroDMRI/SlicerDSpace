from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from anatomy.segmentation.config import DDSegConfig
from anatomy.segmentation.pipeline import run_ddseg


def _path_or_none(value: str | None) -> Path | None:
    if not value:
        return None
    return Path(value).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DDSeg inside PythonSlicer.")
    parser.add_argument("--mask", required=True, help="Path to nodif_brain_mask.nii.gz")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--weights-dir", required=True, help="DDSeg weights directory")
    parser.add_argument("--parameter-type", default="DTI", help="Feature bundle type")
    parser.add_argument("--device", default="auto", help="Torch device; auto prefers the first visible CUDA GPU")
    parser.add_argument("--apply-softmax", action="store_true", help="Apply softmax to model outputs")
    parser.add_argument("--dti-feature-dir", help="Existing DTI feature bundle directory")
    parser.add_argument("--dwi-nii", help="Raw DWI NIfTI")
    parser.add_argument("--bval", help="bvals path")
    parser.add_argument("--bvec", help="bvecs path")
    parser.add_argument("--slicer-base", help="Slicer installation root")
    parser.add_argument("--slicer-ext", help="deprecated; no longer needed")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    cfg = DDSegConfig(
        input_mask_nii=Path(args.mask).expanduser().resolve(),
        parameter_type=args.parameter_type,
        output_folder=Path(args.output).expanduser().resolve(),
        weights_dir=Path(args.weights_dir).expanduser().resolve(),
        dti_feature_dir=_path_or_none(args.dti_feature_dir),
        device=args.device,
        apply_softmax=bool(args.apply_softmax),
        dwi_nii=_path_or_none(args.dwi_nii),
        bval=_path_or_none(args.bval),
        bvec=_path_or_none(args.bvec),
        slicer_base=_path_or_none(args.slicer_base),
        slicer_ext=_path_or_none(args.slicer_ext),
    )
    run_ddseg(cfg)


if __name__ == "__main__":
    main()
