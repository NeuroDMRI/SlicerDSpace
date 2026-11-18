#!/usr/bin/env python3
"""Summarize an existing real-subject CSD/ACT -> WMQL -> connectome run.

This script is intentionally read-only. It validates and consolidates the QC
artifacts produced by ``run_csd_act_real_subject_smoke.py`` so long-run evidence
does not depend on stale hand-written notes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


CONNECTOME_METRICS = (
    "streamline_count",
    "sift2_weighted_streamline_count",
    "mean_fa",
    "mean_md",
    "mean_length_mm",
    "total_length_mm",
    "mrtrix_dense_compressed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-root", default="/data/chengjin/SlicerDDSurfer-test/100307")
    parser.add_argument("--subject-id", default="100307")
    parser.add_argument("--run-name", default="csd_act_100k")
    parser.add_argument("--expected-streamlines", type=int, default=100000)
    parser.add_argument("--min-bundle-count", type=int, default=4)
    parser.add_argument("--min-assigned-streamlines", type=int, default=90000)
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON. Defaults to outputs/00_run_metadata/<run-name>_real_subject_audit.json.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _du_bytes(path: Path) -> int | None:
    if not path.exists():
        return None
    result = subprocess.run(
        ["du", "-sb", path.as_posix()],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return int(result.stdout.split()[0])


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "disk_usage_bytes": _du_bytes(path),
    }


def _connectome_qc_path(connectome_root: Path, subject_id: str, run_name: str, metric: str) -> Path:
    run_dir = connectome_root / run_name
    if metric == "streamline_count":
        return run_dir / f"{subject_id}-connectome-streamline_count_qc.json"
    if metric == "mrtrix_dense_compressed":
        return run_dir / metric / f"{subject_id}-connectome-streamline_count_qc.json"
    return run_dir / metric / f"{subject_id}-connectome-{metric}_qc.json"


def _summarize_connectomes(connectome_root: Path, subject_id: str, run_name: str) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for metric in CONNECTOME_METRICS:
        qc_path = _connectome_qc_path(connectome_root, subject_id, run_name, metric)
        if not qc_path.is_file():
            summaries[metric] = {"status": "missing", "qc": qc_path.as_posix()}
            continue
        qc = _read_json(qc_path)
        summaries[metric] = {
            "status": qc.get("status"),
            "backend": qc.get("backend"),
            "qc": qc_path.as_posix(),
            "node_count": qc.get("node_count"),
            "named_node_count": qc.get("named_node_count"),
            "unnamed_node_count": qc.get("unnamed_node_count"),
            "edge_count": qc.get("edge_count"),
            "elapsed_seconds": qc.get("elapsed_seconds"),
            "streamline_count_total": qc.get("streamline_count_total"),
            "streamline_count_assigned": qc.get("streamline_count_assigned"),
            "streamline_count_skipped": qc.get("streamline_count_skipped"),
            "streamline_count_scalar_sampled": qc.get("streamline_count_scalar_sampled"),
            "streamline_count_length_sampled": qc.get("streamline_count_length_sampled"),
            "sift2_weight_count": qc.get("sift2_weight_count"),
        }
    return summaries


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from shared import normalize_run_name, subject_layout

    subject_root = Path(args.subject_root).expanduser().resolve()
    subject_id = args.subject_id
    run_name = normalize_run_name(args.run_name)
    layout = subject_layout(subject_root)
    outputs = layout.outputs
    metadata_root = layout.run_metadata

    csd_dir = layout.stage_cache("03_tractography/csd" if run_name == "csd_act" else f"03_tractography/csd/{run_name}")
    tractogram_dir = layout.tractograms / run_name
    wmql_dir = layout.stage_cache("04_bundles/wmql") / run_name
    connectome_root = layout.stage_cache("05_connectome")
    if not (csd_dir / "csd_act_qc.json").is_file() and (outputs / "04_tractography").is_dir():
        legacy_tract = outputs / "04_tractography"
        csd_dir = (legacy_tract / "csd") if run_name == "csd_act" else (legacy_tract / "csd" / run_name)
        tractogram_dir = legacy_tract / "tractograms" / run_name
        wmql_dir = legacy_tract / "bundles" / "wmql" / run_name
        connectome_root = legacy_tract / "connectome"

    csd_qc_path = csd_dir / "csd_act_qc.json"
    wmql_qc_path = wmql_dir / "wmql_qc.json"
    orchestration_path = metadata_root / f"{run_name}_orchestration_manifest.json"

    csd_qc = _read_json(csd_qc_path)
    wmql_qc = _read_json(wmql_qc_path)
    output_summary = wmql_qc.get("output_summary") if isinstance(wmql_qc.get("output_summary"), dict) else {}
    bundles = output_summary.get("bundles") if isinstance(output_summary.get("bundles"), list) else []

    tractogram_tck = tractogram_dir / f"{subject_id}_csd_act_ifod2.tck"
    tractogram_trk = tractogram_dir / f"{subject_id}_csd_act_ifod2.trk"
    sift2_weights = tractogram_dir / f"{subject_id}_csd_act_ifod2_sift2_weights.txt"

    connectomes = _summarize_connectomes(connectome_root, subject_id, run_name)
    streamline_connectome = connectomes.get("streamline_count", {})

    checks = {
        "csd_act_completed": csd_qc.get("status") == "completed",
        "csd_act_executed": bool(csd_qc.get("executed")),
        "streamline_count_matches_expected": streamline_connectome.get("streamline_count_total") == args.expected_streamlines,
        "wmql_bundle_count_at_least_minimum": int(output_summary.get("bundle_count") or 0) >= args.min_bundle_count,
        "connectome_assigned_at_least_minimum": float(streamline_connectome.get("streamline_count_assigned") or 0) >= args.min_assigned_streamlines,
        "all_connectomes_completed": all(summary.get("status") == "completed" for summary in connectomes.values()),
    }

    summary = {
        "kind": "slicerddsurfer_real_subject_csd_act_audit",
        "version": 1,
        "subject_id": subject_id,
        "subject_root": subject_root.as_posix(),
        "run_name": run_name,
        "expected_streamlines": args.expected_streamlines,
        "status": "completed" if all(checks.values()) else "attention_required",
        "checks": checks,
        "inputs": {
            "csd_qc": csd_qc_path.as_posix(),
            "wmql_qc": wmql_qc_path.as_posix(),
            "orchestration_manifest": orchestration_path.as_posix() if orchestration_path.is_file() else None,
        },
        "disk_usage": {
            "csd_dir_bytes": _du_bytes(csd_dir),
            "tractogram_dir_bytes": _du_bytes(tractogram_dir),
            "wmql_dir_bytes": _du_bytes(wmql_dir),
            "connectome_root_bytes": _du_bytes(connectome_root),
        },
        "csd_act": {
            "status": csd_qc.get("status"),
            "executed": csd_qc.get("executed"),
            "elapsed_seconds": csd_qc.get("elapsed_seconds"),
            "wm_fod_exists": csd_qc.get("wm_fod_exists"),
            "wm_fod_norm_exists": csd_qc.get("wm_fod_norm_exists"),
            "tractogram_exists": csd_qc.get("tractogram_exists"),
            "tractogram_trk_exists": csd_qc.get("tractogram_trk_exists"),
            "sift2_weights_exists": csd_qc.get("sift2_weights_exists"),
            "tractogram_tck": _file_record(tractogram_tck),
            "tractogram_trk": _file_record(tractogram_trk),
            "sift2_weights": _file_record(sift2_weights),
        },
        "wmql": {
            "status": wmql_qc.get("status"),
            "executed": wmql_qc.get("executed"),
            "elapsed_seconds": wmql_qc.get("elapsed_seconds"),
            "bundle_count": output_summary.get("bundle_count"),
            "bundles": bundles,
        },
        "connectomes": connectomes,
    }

    output_path = Path(args.output).expanduser().resolve() if args.output else metadata_root / f"{run_name}_real_subject_audit.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[csd-act-audit] summary={output_path}")
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
