#!/usr/bin/env python3
"""Standalone CLI worker for DDParcel-guided ROI tractography (side task)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_int_list(raw: str) -> tuple[int, ...]:
    values = []
    for token in str(raw or "").replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    return tuple(dict.fromkeys(values))


def _parse_name_list(raw: str) -> tuple[str, ...]:
    if not str(raw or "").strip():
        return ()
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run standalone ROI tractography from DDParcel brain region(s).")
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--subject-root", required=True)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--parcellation", required=True)
    parser.add_argument("--algorithm", required=True, choices=("ukf", "csd_act", "dti_det", "saf"))
    parser.add_argument("--label-ids", required=True, help="Comma-separated DDParcel integer label IDs.")
    parser.add_argument("--region-names", default="", help="Comma-separated region names matching label IDs.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--select-streamlines", type=int, default=10000)
    parser.add_argument("--dilate-cortical-voxels", type=int, default=1)
    parser.add_argument("--ukf-parameters", default="{}")
    parser.add_argument("--ukf-executable", default="")
    parser.add_argument("--dwiconvert-executable", default="")
    parser.add_argument("--csd-act-parameters", default="{}")
    parser.add_argument("--dti-parameters", default="{}")
    parser.add_argument("--saf-parameters", default="{}")
    parser.add_argument("--mrtrix-bin-dir", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    from shared.roi_tracking import ROITrackingConfig, run_roi_tractography

    args = build_parser().parse_args(argv)
    try:
        label_ids = _parse_int_list(args.label_ids)
        region_names = _parse_name_list(args.region_names)
        config = ROITrackingConfig(
            subject_id=args.subject_id,
            subject_root=Path(args.subject_root),
            output_root=Path(args.output_root) if args.output_root.strip() else None,
            parcellation_path=Path(args.parcellation),
            algorithm=args.algorithm,
            label_ids=label_ids,
            region_names=region_names,
            device=args.device,
            select_streamlines=int(args.select_streamlines),
            dilate_cortical_voxels=int(args.dilate_cortical_voxels),
            ukf_parameters=json.loads(args.ukf_parameters or "{}"),
            ukf_executable=Path(args.ukf_executable) if args.ukf_executable.strip() else None,
            dwiconvert_executable=Path(args.dwiconvert_executable) if args.dwiconvert_executable.strip() else None,
            csd_act_parameters=json.loads(args.csd_act_parameters or "{}"),
            dti_parameters=json.loads(args.dti_parameters or "{}"),
            saf_parameters=json.loads(args.saf_parameters or "{}"),
            mrtrix_bin_dir=Path(args.mrtrix_bin_dir) if args.mrtrix_bin_dir.strip() else None,
        )
        result = run_roi_tractography(config, log=lambda msg: print(msg, flush=True))
        print("ROI_MANIFEST:" + Path(result["manifest_path"]).as_posix(), flush=True)
        return 0
    except Exception as exc:
        print(f"[ROI Tracking] ERROR: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
