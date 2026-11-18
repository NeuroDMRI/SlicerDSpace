#!/usr/bin/env python3
"""Aggregate SlicerDSpace manifests into publication-oriented benchmark tables and figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ORCHESTRATION_KIND = "slicerddsurfer_product_orchestration"
COMPUTE_STATUSES = {"completed", "passed", "success"}
REUSE_STATUSES = {"skipped", "reused"}


def _seconds(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def discover_manifests(inputs: Iterable[Path]) -> list[Path]:
    """Discover orchestration manifests from files and output trees."""

    discovered: set[Path] = set()
    for raw in inputs:
        path = raw.expanduser().resolve()
        candidates = [path] if path.is_file() else path.rglob("*_orchestration_manifest.json") if path.is_dir() else []
        for candidate in candidates:
            payload = _read_json(candidate)
            if payload and payload.get("kind") == ORCHESTRATION_KIND:
                discovered.add(candidate)
    return sorted(discovered)


def _first_numeric_key(payload: Any, key: str) -> float | None:
    if isinstance(payload, dict):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        for child in payload.values():
            found = _first_numeric_key(child, key)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for child in payload:
            found = _first_numeric_key(child, key)
            if found is not None:
                return found
    return None


def discover_stage_metrics(inputs: Iterable[Path]) -> list[dict[str, Any]]:
    """Read authoritative stage QC/progress files without report/metadata copies."""

    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for raw in inputs:
        root = raw.expanduser().resolve()
        search_root = root.parent if root.is_file() else root
        if not search_root.is_dir():
            continue
        for path in search_root.rglob("*.json"):
            if path in seen:
                continue
            seen.add(path)
            relative_parts = set(path.relative_to(search_root).parts)
            if relative_parts & {"00_run_metadata", "06_slicer_scene_report", "06_qc", "reports", "display_previews", "legacy_outputs", ".debug_outputs"}:
                continue
            if not (path.name.endswith("_qc.json") or path.name.endswith("_progress.json")):
                continue
            payload = _read_json(path)
            if not payload:
                continue
            elapsed = _first_numeric_key(payload, "elapsed_seconds")
            if elapsed is None:
                continue
            stage = str(payload.get("stage") or payload.get("kind") or path.stem)
            rows.append(
                {
                    "stage": stage,
                    "status": str(payload.get("status") or "unknown").lower(),
                    "elapsed_seconds": elapsed,
                    "elapsed_minutes": elapsed / 60.0,
                    "source_path": path.as_posix(),
                }
            )
    return sorted(rows, key=lambda row: (row["stage"], row["source_path"]))


def _task_size_lookup(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = list(profile.get("slowest_tasks") or []) + list(profile.get("largest_task_outputs") or [])
    lookup: dict[str, dict[str, Any]] = {}
    for row in rows:
        name = str(row.get("name") or "")
        if not name:
            continue
        current = lookup.setdefault(name, {})
        current.update({key: value for key, value in row.items() if value is not None})
    return lookup


def rows_from_manifest(path: Path, payload: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return one run row, task rows, and top-directory rows."""

    subject_id = str(payload.get("subject_id") or "unknown")
    run_name = str(payload.get("run_name") or path.stem.replace("_orchestration_manifest", ""))
    profile = payload.get("performance_profile") if isinstance(payload.get("performance_profile"), dict) else {}
    tasks = list(payload.get("tasks") or [])
    status_counts = Counter(str(task.get("status") or "unknown").lower() for task in tasks)
    completed_runtimes = [
        float(task["runtime_seconds"])
        for task in tasks
        if str(task.get("status") or "").lower() in COMPUTE_STATUSES and _seconds(task.get("runtime_seconds")) is not None
    ]
    reused_runtimes = [
        float(task["runtime_seconds"])
        for task in tasks
        if str(task.get("status") or "").lower() in REUSE_STATUSES and _seconds(task.get("runtime_seconds")) is not None
    ]
    top_dirs = list(profile.get("top_dirs") or [])
    output_bytes = sum(int(row.get("size_bytes") or 0) for row in top_dirs) if top_dirs else None
    run_row = {
        "manifest_path": path.as_posix(),
        "subject_id": subject_id,
        "run_name": run_name,
        "execute": bool(payload.get("execute")),
        "resume": bool(payload.get("resume")),
        "tractography_backend": payload.get("tractography_backend") or "",
        "bundle_backend": payload.get("bundle_backend") or "",
        "connectome_backend": payload.get("connectome_backend") or "",
        "ddseg_device": payload.get("resolved_ddseg_device") or payload.get("ddseg_device") or "",
        "surface_device": payload.get("resolved_surface_device") or payload.get("surface_device") or "",
        "targets": ",".join(str(value) for value in payload.get("targets", []) or []),
        "elapsed_seconds": _seconds(payload.get("elapsed_seconds", profile.get("orchestration_elapsed_seconds"))),
        "completed_task_runtime_seconds": sum(completed_runtimes),
        "resume_check_runtime_seconds": sum(reused_runtimes),
        "task_count": len(tasks),
        "completed_tasks": sum(status_counts.get(status, 0) for status in COMPUTE_STATUSES),
        "skipped_or_reused_tasks": sum(status_counts.get(status, 0) for status in REUSE_STATUSES),
        "blocked_tasks": status_counts.get("blocked", 0),
        "failed_tasks": status_counts.get("failed", 0),
        "output_snapshot_available": bool(top_dirs),
        "output_bytes": output_bytes,
        "output_gib": output_bytes / (1024**3) if output_bytes is not None else None,
    }
    size_lookup = _task_size_lookup(profile)
    task_rows: list[dict[str, Any]] = []
    for task in tasks:
        name = str(task.get("name") or "unknown")
        sizes = size_lookup.get(name, {})
        status = str(task.get("status") or "unknown").lower()
        task_rows.append(
            {
                "manifest_path": path.as_posix(),
                "subject_id": subject_id,
                "run_name": run_name,
                "task": name,
                "status": status,
                "runtime_seconds": _seconds(task.get("runtime_seconds")),
                "is_compute_runtime": status in COMPUTE_STATUSES and _seconds(task.get("runtime_seconds")) is not None and bool(payload.get('execute')),
                "reason": str(task.get("reason") or ""),
                "started_at": str(task.get("started_at") or ""),
                "ended_at": str(task.get("ended_at") or ""),
                "declared_output_bytes": int(sizes.get("declared_output_bytes") or 0),
                "declared_output_files": int(sizes.get("declared_output_files") or 0),
            }
        )
    directory_rows = [
        {
            "manifest_path": path.as_posix(),
            "subject_id": subject_id,
            "run_name": run_name,
            "directory": str(row.get("path") or ""),
            "size_bytes": int(row.get("size_bytes") or 0),
            "size_gib": int(row.get("size_bytes") or 0) / (1024**3),
            "file_count": int(row.get("file_count") or 0),
        }
        for row in top_dirs
    ]
    return run_row, task_rows, directory_rows


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def _statistics(values: list[float]) -> dict[str, Any]:
    values = [value for value in values if _seconds(value) is not None]
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "p95": _percentile(values, 0.95),
    }


def _pretty_stage_name(value: str) -> str:
    text = str(value or "").lower()
    if text == "noddi":
        return "NODDI"
    if text == "freewater":
        return "Free Water"
    if "mrtrix_csd_act" in text:
        return "CSD/ACT tractography"
    if text == "saf_qc":
        return "Short-association fibres (SAF)"
    if text == "surface_qc":
        return "DDSurfer surface reconstruction"
    if "wmql" in text:
        return "WMQL bundle segmentation"
    if "tractcloud" in text:
        return "TractCloud bundle parcellation"
    if "sift2" in text and "connectome" in text:
        return "Connectome (SIFT2-weighted)"
    if "connectome" in text:
        return "Connectome (streamline count)"
    return str(value).replace("_", " ")


def _pretty_directory_name(value: str) -> str:
    return {
        "00_run_metadata": "Run metadata",
        "01_anatomical_priors": "Anatomical priors",
        "01_preproc": "Preprocessing (legacy directory)",
        "02_microstructure": "Microstructure",
        "03_anatomy": "Anatomical reconstruction (legacy directory)",
        "03_structural": "Anatomical reconstruction (legacy directory)",
        "03_tractography": "Tractography",
        "04_bundles": "Bundles",
        "04_tractography": "Tractography, bundles and connectomes (legacy directory)",
        "05_connectome": "Connectome",
        "05_statistics": "Statistics (legacy directory)",
        "06_qc": "QC summaries",
        "06_slicer_scene_report": "Slicer scene and report (legacy directory)",
    }.get(str(value), str(value).replace("_", " "))


def _pretty_task_name(value: str) -> str:
    return {
        "validation": "Input validation",
        "microstructure.dti": "DTI scalar mapping",
        "microstructure.freewater": "Free Water modelling",
        "microstructure.noddi": "NODDI modelling",
        "tractography.dti_det": "DTI deterministic tractography",
        "tractography.csd_act": "CSD/ACT tractography",
        "tractography.saf": "Short-association fibres (SAF)",
        "bundles.wmql": "WMQL bundle segmentation",
        "bundles.wma": "WMA bundle clustering",
        "bundles.tractcloud": "TractCloud bundle parcellation",
        "connectome.streamline_count": "Connectome (streamline count)",
        "connectome.sift2_weighted": "Connectome (SIFT2-weighted)",
    }.get(str(value), str(value).replace(".", " / ").replace("_", " "))


def summarize_stage_metrics(stage_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate authoritative stage measurements for cohort-safe reporting."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stage_rows:
        grouped[_pretty_stage_name(str(row["stage"]))].append(row)
    summary_rows: list[dict[str, Any]] = []
    for stage, rows in grouped.items():
        values = [float(row["elapsed_seconds"]) for row in rows
                  if row['status'] in COMPUTE_STATUSES and _seconds(row.get('elapsed_seconds')) is not None]
        stats = _statistics(values)
        status_counts = Counter(str(row["status"]) for row in rows)
        summary_rows.append(
            {
                "stage": stage,
                "n": len(values),
                "completed_n": sum(status_counts.get(status, 0) for status in COMPUTE_STATUSES),
                "statuses": ";".join(f"{key}:{value}" for key, value in sorted(status_counts.items())),
                "mean_seconds": stats["mean"],
                "median_seconds": stats["median"],
                "q1_seconds": _percentile(values, 0.25),
                "q3_seconds": _percentile(values, 0.75),
                "minimum_seconds": stats["minimum"],
                "maximum_seconds": stats["maximum"],
                "p95_seconds": stats["p95"],
                "median_minutes": float(stats["median"]) / 60.0 if stats['median'] is not None else None,
            }
        )
    return sorted(summary_rows, key=lambda row: float(row["median_seconds"] or 0), reverse=True)


def _report_generation_system() -> dict[str, Any]:
    memory_gib = None
    try:
        memory_gib = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
    except (AttributeError, OSError, ValueError):
        pass
    processor = platform.processor().strip()
    generic_processors = {"", "x86_64", "amd64", "aarch64", "arm64"}
    if processor.lower() in generic_processors and sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
                if line.lower().startswith("model name"):
                    processor = line.split(":", 1)[1].strip()
                    break
        except (OSError, IndexError):
            pass
    gpu_names: list[str] = []
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
        gpu_names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        pass
    git_revision = None
    git_dirty = None
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
        git_revision = completed.stdout.strip() or None
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
        git_dirty = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "scope": "report_generation_host_not_necessarily_execution_host",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": processor or None,
        "python": sys.version.split()[0],
        "logical_cpu_count": os.cpu_count(),
        "memory_gib": memory_gib,
        "gpus": gpu_names,
        "slicerdsurfer_git_revision": git_revision,
        "slicerdsurfer_git_dirty": git_dirty,
    }


def build_summary(run_rows: list[dict[str, Any]], task_rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in task_rows if row['is_compute_runtime']]
    reused = [row for row in task_rows if row["status"] in REUSE_STATUSES and _seconds(row['runtime_seconds']) is not None]
    status_counts = Counter(row["status"] for row in task_rows)
    by_task: dict[str, list[float]] = defaultdict(list)
    for row in completed:
        by_task[row["task"]].append(float(row["runtime_seconds"]))
    total_tasks = len(task_rows)
    output_sizes = [float(row["output_gib"]) for row in run_rows if row.get("output_gib") is not None]
    return {
        "kind": "slicerddsurfer_protocol_benchmark_summary",
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "run_count": len(run_rows),
        "subject_count": len({row["subject_id"] for row in run_rows}),
        "manifest_count": len({row["manifest_path"] for row in run_rows}),
        "report_generation_system": _report_generation_system(),
        "run_elapsed_seconds": _statistics([row["elapsed_seconds"] for row in run_rows]),
        "completed_task_runtime_seconds": _statistics([float(row["runtime_seconds"]) for row in completed]),
        "resume_check_runtime_seconds": _statistics([float(row["runtime_seconds"]) for row in reused]),
        "task_status_counts": dict(sorted(status_counts.items())),
        "task_status_percent": {
            key: 100.0 * value / total_tasks if total_tasks else 0.0
            for key, value in sorted(status_counts.items())
        },
        "output_size_gib": _statistics(output_sizes),
        "output_snapshot_run_count": len(output_sizes),
        "completed_runtime_by_task": {
            task: _statistics(values)
            for task, values in sorted(by_task.items())
        },
        "interpretation_notes": [
            "Only tasks with status=completed, passed, or success are treated as algorithm compute runtimes.",
            "Skipped/reused task runtimes represent resume-detection overhead, not original algorithm runtime.",
            "Output sizes are snapshots of existing output directories at manifest creation time.",
            "The recorded system describes the report-generation host; verify it against run provenance before manuscript use.",
            "These descriptive benchmarks require cohort size, hardware, software versions, and acquisition characteristics before manuscript use.",
        ],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_figures(
    output_dir: Path,
    run_rows: list[dict[str, Any]],
    task_rows: list[dict[str, Any]],
    directory_rows: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [f"Figures not generated because matplotlib is unavailable: {exc}"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    colors = ["#235347", "#D58545", "#4C78A8", "#B44B4B", "#7A6FAF", "#6B7D74"]
    written: list[str] = []

    completed_by_task: dict[str, list[float]] = defaultdict(list)
    for row in task_rows:
        if row['is_compute_runtime']:
            completed_by_task[row["task"]].append(float(row["runtime_seconds"]))
    if completed_by_task:
        ordered = sorted(completed_by_task, key=lambda key: statistics.median(completed_by_task[key]), reverse=True)
        labels = [_pretty_task_name(name) for name in ordered]
        values = [statistics.median(completed_by_task[name]) for name in ordered]
        lower_errors = [
            values[index] - float(_percentile(completed_by_task[name], 0.25) or values[index])
            for index, name in enumerate(ordered)
        ]
        upper_errors = [
            float(_percentile(completed_by_task[name], 0.75) or values[index]) - values[index]
            for index, name in enumerate(ordered)
        ]
        maximum_value = max(values) or 1.0
        fig, ax = plt.subplots(figsize=(7.4, max(4.2, len(labels) * 0.55)))
        bars = ax.barh(labels[::-1], values[::-1], color=colors[0])
        positions = [bar.get_y() + bar.get_height() / 2 for bar in bars]
        ax.errorbar(
            values[::-1],
            positions,
            xerr=[lower_errors[::-1], upper_errors[::-1]],
            fmt="none",
            ecolor="#173B33",
            elinewidth=1.0,
            capsize=2.5,
        )
        for bar, value, task in zip(bars, values[::-1], ordered[::-1]):
            ax.text(
                value + maximum_value * 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3g} (n={len(completed_by_task[task])})",
                va="center",
            )
        ax.set_xlabel("Median completed runtime (s)")
        ax.set_title("Task runtime from orchestration records")
        ax.grid(axis="x", color="#D9E1DD", linewidth=0.7)
        ax.set_xlim(0, maximum_value * 1.25)
        fig.tight_layout()
        for suffix in ("png", "svg"):
            path = output_dir / f"task_runtime.{suffix}"
            fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
            written.append(path.as_posix())
        plt.close(fig)

    statuses = Counter(row["status"] for row in task_rows)
    if statuses:
        labels = list(sorted(statuses))
        values = [statuses[label] for label in labels]
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.bar(labels, values, color=colors[: len(labels)])
        ax.set_ylabel("Task records")
        ax.set_title("Execution status distribution")
        ax.grid(axis="y", color="#D9E1DD", linewidth=0.7)
        fig.tight_layout()
        for suffix in ("png", "svg"):
            path = output_dir / f"task_status_distribution.{suffix}"
            fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
            written.append(path.as_posix())
        plt.close(fig)

    directory_sizes: dict[str, list[float]] = defaultdict(list)
    for row in directory_rows:
        directory_sizes[row["directory"]].append(float(row["size_gib"]))
    if directory_sizes:
        ordered = sorted(directory_sizes, key=lambda key: statistics.median(directory_sizes[key]), reverse=True)
        values = [statistics.median(directory_sizes[name]) for name in ordered]
        labels = [_pretty_directory_name(name) for name in ordered]
        maximum_value = max(values) or 1.0
        fig, ax = plt.subplots(figsize=(7.4, max(4.2, len(labels) * 0.5)))
        bars = ax.barh(labels[::-1], values[::-1], color=colors[2])
        for bar, value in zip(bars, values[::-1]):
            label = f"{value:.2f}" if value >= 0.01 else f"{value:.2g}"
            ax.text(
                value + maximum_value * 0.012,
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center",
            )
        ax.set_xlabel("Median output size (GiB)")
        ax.set_title("Output storage by product directory")
        ax.grid(axis="x", color="#D9E1DD", linewidth=0.7)
        ax.set_xlim(0, maximum_value * 1.18)
        fig.tight_layout()
        for suffix in ("png", "svg"):
            path = output_dir / f"output_storage.{suffix}"
            fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
            written.append(path.as_posix())
        plt.close(fig)
    stage_summary_rows = [row for row in summarize_stage_metrics(stage_rows) if row['median_seconds'] is not None]
    if stage_summary_rows:
        labels = [str(row["stage"]) for row in stage_summary_rows]
        values = [float(row["median_seconds"]) / 60.0 for row in stage_summary_rows]
        lower_errors = [
            (float(row["median_seconds"]) - float(row["q1_seconds"])) / 60.0
            for row in stage_summary_rows
        ]
        upper_errors = [
            (float(row["q3_seconds"]) - float(row["median_seconds"])) / 60.0
            for row in stage_summary_rows
        ]
        maximum_value = max(values) or 1.0
        fig, ax = plt.subplots(figsize=(7.4, max(4.6, len(labels) * 0.48)))
        bars = ax.barh(labels[::-1], values[::-1], color=colors[1])
        positions = [bar.get_y() + bar.get_height() / 2 for bar in bars]
        ax.errorbar(
            values[::-1],
            positions,
            xerr=[lower_errors[::-1], upper_errors[::-1]],
            fmt="none",
            ecolor="#784F2F",
            elinewidth=1.0,
            capsize=2.5,
        )
        for bar, value, row in zip(bars, values[::-1], stage_summary_rows[::-1]):
            ax.text(
                value + maximum_value * 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f} (n={row['n']})",
                va="center",
            )
        ax.set_xlabel("Wall time (min)")
        ax.set_title("Stage wall time from QC/progress records")
        ax.grid(axis="x", color="#D9E1DD", linewidth=0.7)
        ax.set_xlim(0, maximum_value * 1.28)
        fig.tight_layout()
        for suffix in ("png", "svg"):
            path = output_dir / f"stage_wall_time.{suffix}"
            fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
            written.append(path.as_posix())
        plt.close(fig)
    return written


def generate_benchmark_report(inputs: list[Path], output_dir: Path) -> dict[str, Any]:
    manifests = discover_manifests(inputs)
    if not manifests:
        raise FileNotFoundError("No SlicerDSpace orchestration manifests were found in the supplied inputs.")
    run_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    directory_rows: list[dict[str, Any]] = []
    stage_rows = discover_stage_metrics(inputs)
    stage_summary_rows = summarize_stage_metrics(stage_rows)
    for manifest in manifests:
        payload = _read_json(manifest)
        if not payload:
            continue
        run_row, task_part, directory_part = rows_from_manifest(manifest, payload)
        run_rows.append(run_row)
        task_rows.extend(task_part)
        directory_rows.extend(directory_part)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "runs.csv", run_rows, list(run_rows[0]))
    _write_csv(output_dir / "tasks.csv", task_rows, list(task_rows[0]) if task_rows else ["manifest_path"])
    _write_csv(output_dir / "output_directories.csv", directory_rows, list(directory_rows[0]) if directory_rows else ["manifest_path"])
    _write_csv(
        output_dir / "stage_metrics.csv",
        stage_rows,
        list(stage_rows[0]) if stage_rows else ["stage", "status", "elapsed_seconds", "elapsed_minutes", "source_path"],
    )
    _write_csv(
        output_dir / "stage_summary.csv",
        stage_summary_rows,
        list(stage_summary_rows[0])
        if stage_summary_rows
        else [
            "stage",
            "n",
            "completed_n",
            "statuses",
            "mean_seconds",
            "median_seconds",
            "q1_seconds",
            "q3_seconds",
            "minimum_seconds",
            "maximum_seconds",
            "p95_seconds",
            "median_minutes",
        ],
    )
    summary = build_summary(run_rows, task_rows)
    summary["authoritative_stage_wall_time"] = {
        row["stage"]: {
            key: value
            for key, value in row.items()
            if key != "stage"
        }
        for row in stage_summary_rows
    }
    figure_paths = _plot_figures(output_dir, run_rows, task_rows, directory_rows, stage_rows)
    summary["input_manifests"] = [path.as_posix() for path in manifests]
    summary["outputs"] = {
        "runs_csv": (output_dir / "runs.csv").as_posix(),
        "tasks_csv": (output_dir / "tasks.csv").as_posix(),
        "output_directories_csv": (output_dir / "output_directories.csv").as_posix(),
        "stage_metrics_csv": (output_dir / "stage_metrics.csv").as_posix(),
        "stage_summary_csv": (output_dir / "stage_summary.csv").as_posix(),
        "figures": figure_paths,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Nature Protocols-oriented SlicerDSpace benchmark CSVs and figures.")
    parser.add_argument("--input", action="append", required=True, type=Path, help="Manifest file or output tree. May be repeated.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Destination for CSV, JSON, PNG and SVG outputs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = generate_benchmark_report(args.input, args.output_dir)
    except Exception as exc:
        print(f"Benchmark generation failed: {exc}")
        return 2
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
