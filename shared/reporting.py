"""Subject-level report/provenance generation for SlicerDSpace outputs."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .io_layout import normalize_run_name, subject_layout


@dataclass(frozen=True)
class SubjectReportPaths:
    """Paths written by the subject report generator."""

    report_html: Path
    provenance_json: Path
    qc_summary_json: Path
    summary_md: Path
    report_html_latest: Path
    provenance_json_latest: Path
    qc_summary_json_latest: Path
    summary_md_latest: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "report_html": self.report_html.as_posix(),
            "provenance_json": self.provenance_json.as_posix(),
            "qc_summary_json": self.qc_summary_json.as_posix(),
            "summary_md": self.summary_md.as_posix(),
            "report_html_latest": self.report_html_latest.as_posix(),
            "provenance_json_latest": self.provenance_json_latest.as_posix(),
            "qc_summary_json_latest": self.qc_summary_json_latest.as_posix(),
            "summary_md_latest": self.summary_md_latest.as_posix(),
        }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _collect_output_tree(output_root: Path, *, max_files: int = 2500) -> dict[str, Any]:
    top_dirs: dict[str, int] = {}
    files: list[dict[str, Any]] = []
    total_files = 0
    total_bytes = 0
    if not output_root.exists():
        return {
            "output_root": output_root.as_posix(),
            "exists": False,
            "top_dirs": top_dirs,
            "files": files,
            "file_count": 0,
            "total_bytes": 0,
            "truncated": False,
        }
    for path in sorted(output_root.rglob("*")):
        if not path.is_file():
            continue
        total_files += 1
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        total_bytes += size
        try:
            top_name = path.relative_to(output_root).parts[0]
        except (IndexError, ValueError):
            top_name = ""
        if top_name:
            top_dirs[top_name] = top_dirs.get(top_name, 0) + 1
        if len(files) < max_files:
            files.append(
                {
                    "path": _relative(path, output_root),
                    "size_bytes": size,
                }
            )
    return {
        "output_root": output_root.as_posix(),
        "exists": True,
        "top_dirs": top_dirs,
        "files": files,
        "file_count": total_files,
        "total_bytes": total_bytes,
        "truncated": total_files > len(files),
    }


def _collect_qc_files(output_root: Path, *, extra_roots: tuple[Path, ...] = ()) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    roots = (output_root, *extra_roots)
    for root in roots:
        if not root or not root.exists():
            continue
        base = output_root.parent if root != output_root else output_root
        for path in sorted(root.rglob("*qc.json")):
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            payload = _read_json(path)
            records.append(
                {
                    "path": _relative(path, base),
                    "status": payload.get("status"),
                    "kind": payload.get("kind"),
                    "summary": {
                        key: value
                        for key, value in payload.items()
                        if key
                        in {
                            "status",
                            "streamline_count",
                            "assigned_streamlines",
                            "edge_count",
                            "node_count",
                            "bundle_count",
                            "anatomical_tract_count",
                            "warnings",
                            "errors",
                        }
                    },
                }
            )
    return records


def _task_summary(manifest: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in manifest.get("tasks", []):
        status = str(task.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return counts


def _format_bytes(value: Any) -> str:
    try:
        size = float(value)
    except (TypeError, ValueError):
        return "0 B"
    units = ("B", "KB", "MB", "GB", "TB")
    unit = 0
    while size >= 1024 and unit < len(units) - 1:
        size /= 1024
        unit += 1
    if unit == 0:
        return f"{int(size)} {units[unit]}"
    return f"{size:.1f} {units[unit]}"


def _format_seconds(value: Any) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "0.0 s"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, rem = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {rem:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m"


def _task_status_for_humans(status: Any) -> str:
    value = str(status or "unknown")
    return {
        "completed": "Completed",
        "passed": "Passed",
        "skipped": "Reused existing output",
        "blocked": "Blocked",
        "failed": "Failed",
        "planned": "Planned",
        "running": "Running",
        "will_run": "Will run",
    }.get(value, value.replace("_", " ").title())


def _overall_status(task_counts: dict[str, int]) -> tuple[str, str]:
    failed = task_counts.get("failed", 0)
    blocked = task_counts.get("blocked", 0)
    planned = task_counts.get("planned", 0) + task_counts.get("will_run", 0)
    running = task_counts.get("running", 0)
    completed_like = sum(task_counts.get(key, 0) for key in ("completed", "passed", "skipped"))
    total = sum(task_counts.values())
    unknown = total - failed - blocked - planned - running - completed_like
    if failed:
        return "failed", f"{failed} task(s) failed. Review failed task QC/logs first."
    if running:
        return "running", f"{running} task(s) are still running."
    if blocked and completed_like:
        return "partial", f"Core results are available; {blocked} optional task(s) are blocked."
    if blocked:
        return "blocked", f"{blocked} task(s) are blocked before execution."
    if planned and completed_like:
        return "planned", f"{planned} task(s) are planned; {completed_like} task(s) completed or reused."
    if planned:
        return "planned", f"{planned} task(s) are planned. Run with execution enabled to generate outputs."
    if unknown:
        return "unknown", f"{unknown} task(s) have unrecognized status. Review task QC/logs."
    if not total:
        return "unknown", "No task records were found in the orchestration manifest."
    return "completed", "All selected tasks completed or reused existing outputs."


def _task_family(name: str) -> str:
    if "." in name:
        prefix = name.split(".", 1)[0]
    else:
        prefix = name
    return {
        "validation": "Validation",
        "preprocessing": "Preprocessing",
        "segmentation": "Segmentation",
        "parcellation": "Parcellation",
        "surface": "Surface",
        "microstructure": "Microstructure",
        "tractography": "Tractography",
        "bundles": "Fiber bundles",
        "connectome": "Connectome",
    }.get(prefix, prefix.replace("_", " ").title())


def _short_reason(reason: Any, *, max_len: int = 170) -> str:
    text = " ".join(str(reason or "").split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip(" ,;:") + "..."


def _blocked_hint(name: str, reason: Any) -> str:
    lowered = f"{name} {reason}".lower()
    if "freesurfer" in lowered or "recon-all" in lowered or "mri_" in lowered:
        return "Configure FreeSurfer tools and FREESURFER_HOME, or leave surface postprocess disabled."
    if "wma" in lowered or "org atlas" in lowered:
        return "Configure WMA root, ORG atlas directory, and the WMA/Slicer executable path."
    if "tractcloud" in lowered:
        return "Provide a VTK/VTP tractogram and TractCloud backend/model configuration."
    if "tckgen" in lowered or "mrtrix" in lowered or "tckconvert" in lowered:
        return "Configure MRtrix3 bin directory or add MRtrix commands to PATH."
    if "missing" in lowered:
        return "Install or configure the missing dependency, then rerun the same command."
    return "Review the task QC/log for the exact blocker."


def _collect_task_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in manifest.get("tasks", []):
        name = str(task.get("name") or "")
        status = str(task.get("status") or "unknown")
        rows.append(
            {
                "family": _task_family(name),
                "name": name,
                "status": status,
                "status_label": _task_status_for_humans(status),
                "reason": _short_reason(task.get("reason")),
                "raw_reason": str(task.get("reason") or ""),
                "qc_path": task.get("qc_path") or task.get("normalized_qc_path"),
                "manifest_path": task.get("manifest_path") or task.get("normalized_manifest_path"),
            }
        )
    return rows


def _collect_blocked_items(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    items: dict[tuple[str, str], dict[str, Any]] = {}

    def add(name: Any, reason: Any) -> None:
        key_name = str(name or "")
        raw_reason = str(reason or "")
        signature = (_short_reason(raw_reason, max_len=220), _blocked_hint(key_name, raw_reason))
        existing = items.get(signature)
        candidate = {
            "name": key_name,
            "stage": _task_family(key_name),
            "reason": _short_reason(raw_reason),
            "raw_reason": raw_reason,
            "suggested_action": _blocked_hint(key_name, raw_reason),
        }
        if existing is None or ("." in key_name and "." not in str(existing.get("name") or "")):
            items[signature] = candidate

    for task in manifest.get("tasks", []):
        if task.get("status") != "blocked":
            continue
        add(task.get("name"), task.get("reason"))
    for name, reason in (manifest.get("preflight_blocked_targets") or {}).items():
        add(name, reason)
    return sorted(items.values(), key=lambda item: str(item.get("name") or ""))


def _collect_key_outputs(manifest: dict[str, Any], output_tree: dict[str, Any], *, max_items: int = 24) -> list[dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    output_root = Path(str(output_tree.get("output_root") or ""))

    def add(path_value: Any, label: str, source: str) -> None:
        if not path_value:
            return
        path = Path(str(path_value))
        rel = _relative(path, output_root) if output_root.as_posix() not in ("", ".") else path.as_posix()
        if rel in outputs:
            return
        outputs[rel] = {"label": label, "path": rel, "source": source}

    file_paths = [str(record.get("path") or "") for record in output_tree.get("files", [])]

    def first_containing(*tokens: str) -> Optional[str]:
        for path in file_paths:
            if all(token in path for token in tokens):
                return path
        return None

    def add_first(label: str, source: str, *tokens: str) -> None:
        match = first_containing(*tokens)
        if match:
            add(match, label, source)

    def add_parent(label: str, source: str, *tokens: str) -> None:
        match = first_containing(*tokens)
        if match:
            add(Path(match).parent.as_posix(), label, source)

    add_parent("DTI scalar maps", "microstructure", "02_microstructure/diffusion_tensor/")
    if not any("02_microstructure/diffusion_tensor/" in p for p in outputs):
        add_parent("DTI scalar maps", "microstructure", "02_microstructure/dti/")
    add_first("ROI-wise microstructure table", "microstructure", "ROIWise-Microstructure.csv")
    add_first("Brain parcellation", "parcellation", "_parcellation.nii.gz")
    if not any(item.get("source") == "parcellation" for item in outputs.values()):
        add_first("DDParcel wmparc", "parcellation", "DDSurfer-wmparc.nii.gz")
    add_first("Tissue segmentation", "segmentation", "SegmentationMap_GMWMCSF")
    add_first("Cortical surface mesh", "surface", "cortical_surface/lh.white.obj")
    add_first("ACT tractogram TCK", "tractography", "_act.tck")
    if not any(item.get("label") == "ACT tractogram TCK" for item in outputs.values()):
        add_first("ACT tractogram TCK", "tractography", "csd_act_ifod2.tck")
    add_first("ACT tractogram TRK", "tractography", "_act.trk")
    if not any(item.get("label") == "ACT tractogram TRK" for item in outputs.values()):
        add_first("ACT tractogram TRK", "tractography", "csd_act_ifod2.trk")
    add_parent("WMQL bundles", "bundles", "04_bundles/query_bundle_extraction")
    add_first("Streamline-count connectome", "connectome", "_streamline_count.csv")
    add_first("SIFT2-weighted connectome", "connectome", "_sift2_weighted.csv")
    add_first("Per-stage QC summary", "qc", "06_qc/")

    for task in manifest.get("tasks", []):
        source = str(task.get("name") or "task")
        for key, value in (task.get("outputs") or {}).items():
            if str(key).lower() in {"bval", "bvec", "dwi", "mask", "ad", "fa", "md", "rd", "l1", "l2", "l3", "trace"}:
                continue
            if len(outputs) >= max_items:
                break
            add(value, str(key).replace("_", " ").upper(), source)
        if len(outputs) >= max_items:
            break
    return list(outputs.values())[:max_items]


def _large_output_rows(output_tree: dict[str, Any], *, max_items: int = 8) -> list[dict[str, Any]]:
    files = sorted(output_tree.get("files", []), key=lambda item: int(item.get("size_bytes") or 0), reverse=True)
    return [
        {
            "path": item.get("path"),
            "size": _format_bytes(item.get("size_bytes")),
            "size_bytes": item.get("size_bytes"),
        }
        for item in files[:max_items]
    ]


def _performance_profile(manifest: dict[str, Any]) -> dict[str, Any]:
    profile = manifest.get("performance_profile")
    return profile if isinstance(profile, dict) else {}


def _performance_task_rows(profile: dict[str, Any], key: str, *, max_items: int = 8) -> list[dict[str, Any]]:
    rows = []
    for item in profile.get(key, [])[:max_items]:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "task": item.get("name"),
                "status": item.get("status"),
                "runtime": _format_seconds(item.get("runtime_seconds")),
                "outputs": _format_bytes(item.get("declared_output_bytes")),
                "files": item.get("declared_output_files", 0),
            }
        )
    return rows


def _performance_top_dir_rows(profile: dict[str, Any], *, max_items: int = 8) -> list[dict[str, Any]]:
    rows = []
    for item in profile.get("largest_top_dirs", [])[:max_items]:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "path": item.get("path"),
                "size": _format_bytes(item.get("size_bytes")),
                "files": item.get("file_count", 0),
            }
        )
    return rows


def _build_executive_summary(
    *,
    manifest: dict[str, Any],
    output_tree: dict[str, Any],
    qc_records: list[dict[str, Any]],
) -> dict[str, Any]:
    task_counts = _task_summary(manifest)
    status, interpretation = _overall_status(task_counts)
    return {
        "status": status,
        "interpretation": interpretation,
        "subject_id": manifest.get("subject_id"),
        "run_name": manifest.get("run_name"),
        "tasks": task_counts,
        "output_files": output_tree.get("file_count", 0),
        "output_size": _format_bytes(output_tree.get("total_bytes", 0)),
        "elapsed_seconds": manifest.get("elapsed_seconds", 0),
        "elapsed_time": _format_seconds(manifest.get("elapsed_seconds", 0)),
        "qc_files": len(qc_records),
        "blocked_count": task_counts.get("blocked", 0),
    }


def _build_provenance(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    output_tree: dict[str, Any],
    qc_records: list[dict[str, Any]],
) -> dict[str, Any]:
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    executive_summary = _build_executive_summary(
        manifest=manifest,
        output_tree=output_tree,
        qc_records=qc_records,
    )
    performance_profile = _performance_profile(manifest)
    return {
        "kind": "slicerddsurfer_subject_provenance",
        "version": 1,
        "generated_at": generated_at,
        "summary": executive_summary,
        "subject_id": manifest.get("subject_id"),
        "subject_root": manifest.get("subject_root"),
        "output_root": manifest.get("output_root"),
        "run_name": manifest.get("run_name"),
        "orchestration_manifest": manifest_path.as_posix(),
        "targets": manifest.get("targets", []),
        "run_targets": manifest.get("run_targets", []),
        "run_plan_summary": manifest.get("run_plan_summary", {}),
        "performance_profile_path": manifest.get("performance_profile_path"),
        "performance_profile": performance_profile,
        "task_summary": _task_summary(manifest),
        "preflight_blocked_targets": manifest.get("preflight_blocked_targets", {}),
        "archived_development_outputs": manifest.get("archived_development_outputs", {}),
        "output_tree": {
            "top_dirs": output_tree.get("top_dirs", {}),
            "file_count": output_tree.get("file_count", 0),
            "total_bytes": output_tree.get("total_bytes", 0),
            "truncated": output_tree.get("truncated", False),
        },
        "qc_file_count": len(qc_records),
    }


def _build_qc_summary(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    output_tree: dict[str, Any],
    qc_records: list[dict[str, Any]],
) -> dict[str, Any]:
    tasks = manifest.get("tasks", [])
    executive_summary = _build_executive_summary(
        manifest=manifest,
        output_tree=output_tree,
        qc_records=qc_records,
    )
    task_rows = _collect_task_rows(manifest)
    blocked_items = _collect_blocked_items(manifest)
    key_outputs = _collect_key_outputs(manifest, output_tree)
    performance_profile = _performance_profile(manifest)
    return {
        "kind": "slicerddsurfer_subject_qc_summary",
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": executive_summary,
        "subject_id": manifest.get("subject_id"),
        "run_name": manifest.get("run_name"),
        "orchestration_manifest": manifest_path.as_posix(),
        "run_plan_summary": manifest.get("run_plan_summary", {}),
        "performance_profile_path": manifest.get("performance_profile_path"),
        "performance_profile": performance_profile,
        "performance_slowest_tasks": _performance_task_rows(performance_profile, "slowest_tasks"),
        "performance_largest_task_outputs": _performance_task_rows(performance_profile, "largest_task_outputs"),
        "performance_largest_top_dirs": _performance_top_dir_rows(performance_profile),
        "task_summary": _task_summary(manifest),
        "readable_task_table": task_rows,
        "blocked_items": blocked_items,
        "key_outputs": key_outputs,
        "largest_outputs": _large_output_rows(output_tree),
        "tasks": [
            {
                "name": task.get("name"),
                "status": task.get("status"),
                "reason": task.get("reason"),
                "qc_path": task.get("qc_path"),
                "manifest_path": task.get("manifest_path"),
            }
            for task in tasks
        ],
        "preflight_blocked_targets": manifest.get("preflight_blocked_targets", {}),
        "output_tree_audit_path": manifest.get("output_tree_audit_path"),
        "output_tree_warnings": manifest.get("output_tree_warnings", []),
        "output_tree": {
            "output_root": output_tree.get("output_root"),
            "exists": output_tree.get("exists"),
            "top_dirs": output_tree.get("top_dirs", {}),
            "file_count": output_tree.get("file_count", 0),
            "total_bytes": output_tree.get("total_bytes", 0),
            "total_size": _format_bytes(output_tree.get("total_bytes", 0)),
            "truncated": output_tree.get("truncated", False),
            "files": output_tree.get("files", []),
        },
        "qc_files": qc_records,
    }


def _html_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "<p class=\"muted\">None.</p>"
    head = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(row.get(column, '') or ''))}</td>" for column in columns)
        body.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _status_badge(status: Any) -> str:
    raw = str(status or "unknown")
    label = _task_status_for_humans(raw)
    css = raw.replace("_", "-")
    return f"<span class=\"badge badge-{html.escape(css)}\">{html.escape(label)}</span>"


def _render_cards(cards: list[tuple[str, Any, str]]) -> str:
    return "".join(
        "<div class=\"card\">"
        f"<div class=\"label\">{html.escape(label)}</div>"
        f"<div class=\"value\">{html.escape(str(value))}</div>"
        f"<div class=\"hint\">{html.escape(hint)}</div>"
        "</div>"
        for label, value, hint in cards
    )


def _render_blocked_items(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p class=\"ok-text\">No blocked targets.</p>"
    rendered = []
    for row in rows:
        rendered.append(
            "<div class=\"blocker\">"
            f"<div class=\"blocker-title\">{html.escape(str(row.get('name') or 'blocked'))}</div>"
            f"<div class=\"blocker-reason\">{html.escape(str(row.get('reason') or ''))}</div>"
            f"<div class=\"blocker-action\">Next: {html.escape(str(row.get('suggested_action') or 'Review QC.'))}</div>"
            "<details><summary>Full reason</summary>"
            f"<p>{html.escape(str(row.get('raw_reason') or ''))}</p>"
            "</details>"
            "</div>"
        )
    return "".join(rendered)


def _render_task_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p class=\"muted\">No task records found.</p>"
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('family') or ''))}</td>"
            f"<td><code>{html.escape(str(row.get('name') or ''))}</code></td>"
            f"<td>{_status_badge(row.get('status'))}</td>"
            f"<td>{html.escape(str(row.get('reason') or ''))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Product</th><th>Task</th><th>Status</th><th>Meaning / reason</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def _render_path_list(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p class=\"muted\">No key outputs found.</p>"
    rendered = []
    for row in rows:
        label = str(row.get("label") or row.get("path") or "output")
        path = str(row.get("path") or "")
        source = str(row.get("source") or "")
        rendered.append(
            "<li>"
            f"<span class=\"path-label\">{html.escape(label)}</span>"
            f"<code>{html.escape(path)}</code>"
            f"<span class=\"source\">{html.escape(source)}</span>"
            "</li>"
        )
    return f"<ul class=\"paths\">{''.join(rendered)}</ul>"


def _build_markdown_summary(
    *,
    manifest: dict[str, Any],
    provenance: dict[str, Any],
    qc_summary: dict[str, Any],
    paths: SubjectReportPaths,
) -> str:
    summary = qc_summary.get("summary", {})
    blocked = qc_summary.get("blocked_items", [])
    task_rows = qc_summary.get("readable_task_table", [])
    key_outputs = qc_summary.get("key_outputs", [])
    slowest_tasks = qc_summary.get("performance_slowest_tasks", [])
    largest_top_dirs = qc_summary.get("performance_largest_top_dirs", [])
    lines = [
        f"# SlicerDSpace Run Summary: {manifest.get('subject_id') or 'subject'}",
        "",
        f"- Status: {summary.get('status')} - {summary.get('interpretation')}",
        f"- Run name: `{manifest.get('run_name') or ''}`",
        f"- Generated at: `{provenance.get('generated_at') or ''}`",
        f"- Output root: `{manifest.get('output_root') or ''}`",
        f"- Elapsed: {summary.get('elapsed_time') or '0.0 s'}",
        f"- Output files: {summary.get('output_files', 0)} ({summary.get('output_size', '0 B')})",
        f"- QC files: {summary.get('qc_files', 0)}",
        "",
        "## Blocked Items",
        "",
    ]
    if blocked:
        for item in blocked:
            lines.append(f"- `{item.get('name')}`: {item.get('reason')}")
            lines.append(f"  Action: {item.get('suggested_action')}")
    else:
        lines.append("- None.")
    lines.extend(["", "## Task Status", ""])
    for row in task_rows:
        lines.append(f"- `{row.get('name')}`: {row.get('status_label')} - {row.get('reason')}")
    lines.extend(["", "## Performance Profile", ""])
    if slowest_tasks:
        lines.append("Slowest tasks:")
        for row in slowest_tasks[:8]:
            lines.append(f"- `{row.get('task')}`: {row.get('runtime')} ({row.get('status')})")
    else:
        lines.append("- No task runtime data available.")
    if largest_top_dirs:
        lines.append("")
        lines.append("Largest top-level output directories:")
        for row in largest_top_dirs[:8]:
            lines.append(f"- `{row.get('path')}`: {row.get('size')} across {row.get('files')} file(s)")
    lines.extend(["", "## Key Outputs", ""])
    for row in key_outputs[:16]:
        lines.append(f"- {row.get('label')}: `{row.get('path')}`")
    lines.extend(
        [
            "",
            "## Report Files",
            "",
            f"- HTML: `{paths.report_html.as_posix()}`",
            f"- QC JSON: `{paths.qc_summary_json.as_posix()}`",
            f"- Provenance JSON: `{paths.provenance_json.as_posix()}`",
            "",
        ]
    )
    return "\n".join(lines)


def _build_html_report(
    *,
    manifest: dict[str, Any],
    provenance: dict[str, Any],
    qc_summary: dict[str, Any],
    paths: SubjectReportPaths,
) -> str:
    summary = qc_summary.get("summary", {})
    tasks = qc_summary.get("readable_task_table", [])
    output_rows = [
        {"top_dir": key, "file_count": value, "meaning": _task_family(key.split("_", 1)[-1]) if key[:2].isdigit() else key}
        for key, value in sorted(qc_summary.get("output_tree", {}).get("top_dirs", {}).items())
    ]
    blocked_rows = qc_summary.get("blocked_items", [])
    key_outputs = qc_summary.get("key_outputs", [])
    largest_outputs = qc_summary.get("largest_outputs", [])
    performance_profile = qc_summary.get("performance_profile", {})
    slowest_tasks = qc_summary.get("performance_slowest_tasks", [])
    largest_task_outputs = qc_summary.get("performance_largest_task_outputs", [])
    largest_top_dirs = qc_summary.get("performance_largest_top_dirs", [])
    status = str(summary.get("status") or "unknown")
    title = f"SlicerDSpace Report - {manifest.get('subject_id') or 'subject'}"
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --ink: #172026;
      --muted: #63717a;
      --line: #d8e0df;
      --paper: #fffdf7;
      --wash: #f0f5f2;
      --accent: #0f766e;
      --warning: #b45309;
      --danger: #b91c1c;
      --ok: #166534;
      --blue: #1d4ed8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at 12% 10%, rgba(15, 118, 110, .13), transparent 26rem),
        linear-gradient(135deg, #f8fbf7 0%, #f6f0e7 100%);
      font-family: Aptos, "IBM Plex Sans", "Helvetica Neue", sans-serif;
      line-height: 1.45;
    }}
    main {{ max-width: 1180px; margin: 28px auto; padding: 0 20px 36px; }}
    .hero {{
      border: 1px solid rgba(23, 32, 38, .16);
      border-radius: 22px;
      padding: 28px;
      background: rgba(255, 253, 247, .92);
      box-shadow: 0 24px 70px rgba(23, 32, 38, .10);
    }}
    h1 {{ margin: 0 0 8px; font-size: clamp(28px, 4vw, 46px); letter-spacing: -.04em; }}
    h2 {{ margin: 32px 0 10px; font-size: 20px; letter-spacing: -.02em; }}
    .subtitle {{ color: var(--muted); margin: 0; max-width: 880px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 22px 0 0; }}
    .card {{ border: 1px solid var(--line); border-radius: 16px; padding: 14px; background: #ffffffcc; }}
    .label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .08em; }}
    .value {{ font-weight: 750; font-size: 22px; word-break: break-word; letter-spacing: -.02em; }}
    .hint {{ color: var(--muted); font-size: 12px; margin-top: 3px; }}
    .panel {{ border: 1px solid var(--line); border-radius: 18px; padding: 18px; background: rgba(255, 253, 247, .78); margin-top: 16px; }}
    table {{ width: 100%; border-collapse: collapse; margin: 10px 0 4px; font-size: 13px; background: white; border-radius: 14px; overflow: hidden; }}
    th, td {{ border-bottom: 1px solid #e8eeec; padding: 9px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #edf4ef; color: #32413c; font-size: 12px; text-transform: uppercase; letter-spacing: .06em; }}
    tr:last-child td {{ border-bottom: 0; }}
    code {{ background: #edf4ef; padding: 2px 5px; border-radius: 6px; word-break: break-all; }}
    details {{ margin-top: 8px; color: var(--muted); }}
    summary {{ cursor: pointer; }}
    .muted, .source {{ color: var(--muted); }}
    .ok-text {{ color: var(--ok); font-weight: 700; }}
    .status-line {{ display: inline-flex; align-items: center; gap: 10px; margin: 12px 0 0; font-weight: 700; }}
    .badge {{ display: inline-flex; align-items: center; border-radius: 999px; padding: 3px 9px; font-size: 12px; font-weight: 750; white-space: nowrap; }}
    .badge-completed, .badge-passed, .badge-skipped {{ color: var(--ok); background: #dcfce7; }}
    .badge-planned, .badge-running {{ color: var(--blue); background: #dbeafe; }}
    .badge-blocked {{ color: var(--warning); background: #ffedd5; }}
    .badge-failed {{ color: var(--danger); background: #fee2e2; }}
    .badge-partial {{ color: var(--warning); background: #ffedd5; }}
    .badge-unknown {{ color: var(--blue); background: #dbeafe; }}
    .blocker {{ border-left: 4px solid var(--warning); padding: 12px 14px; background: #fff8ed; border-radius: 12px; margin: 10px 0; }}
    .blocker-title {{ font-weight: 800; }}
    .blocker-reason {{ margin: 4px 0; }}
    .blocker-action {{ color: #7c2d12; font-weight: 700; }}
    .paths {{ list-style: none; padding: 0; margin: 8px 0 0; }}
    .paths li {{ display: grid; grid-template-columns: minmax(130px, 220px) 1fr auto; gap: 10px; padding: 8px 0; border-bottom: 1px solid #e8eeec; align-items: baseline; }}
    .paths li:last-child {{ border-bottom: 0; }}
    .path-label {{ font-weight: 750; }}
    .footer {{ color: var(--muted); font-size: 12px; margin-top: 28px; }}
    @media (max-width: 720px) {{
      main {{ margin: 12px auto; padding: 0 10px 24px; }}
      .hero {{ padding: 18px; }}
      .paths li {{ grid-template-columns: 1fr; gap: 3px; }}
      table {{ font-size: 12px; }}
    }}
  </style>
</head>
<body>
<main>
  <section class=\"hero\">
    <h1>{html.escape(title)}</h1>
    <p class=\"subtitle\">{html.escape(str(summary.get('interpretation') or ''))}</p>
    <div class=\"status-line\">{_status_badge(status)} <span class=\"muted\">Generated {html.escape(str(provenance.get('generated_at') or ''))}</span></div>
    <div class=\"grid\">
      {_render_cards([
          ('Run name', manifest.get('run_name') or '', 'Use this name for resume and report lookup.'),
          ('Tasks', ', '.join(f"{k}: {v}" for k, v in sorted((summary.get('tasks') or {}).items())) or '0', 'Skipped means existing outputs were reused.'),
          ('Elapsed', summary.get('elapsed_time') or '0.0 s', 'Total product-orchestrator wall time.'),
          ('Outputs', f"{summary.get('output_files', 0)} files", str(summary.get('output_size') or '0 B')),
          ('QC files', summary.get('qc_files', 0), 'Machine-readable checks collected from each step.'),
      ])}
    </div>
  </section>

  <section class=\"panel\">
    <h2>Blocked Items and Next Actions</h2>
    {_render_blocked_items(blocked_rows)}
  </section>

  <section class=\"panel\">
    <h2>Task Status</h2>
    {_render_task_table(tasks)}
  </section>

  <section class=\"panel\">
    <h2>Performance Profile</h2>
    <p class=\"muted\">Use this section to identify the next optimization target. Skipped task runtimes measure resume detection overhead, not original compute time.</p>
    <div class=\"grid\">
      {_render_cards([
          ('Pre-report elapsed', _format_seconds(performance_profile.get('orchestration_elapsed_seconds')), 'Elapsed at the pre-report snapshot.'),
          ('Task runtime sum', _format_seconds(performance_profile.get('task_runtime_seconds_total')), 'Sum of task intervals; concurrent intervals may overlap.'),
          ('Profile JSON', performance_profile.get('path') or manifest.get('performance_profile_path') or '', 'Machine-readable profile artifact.'),
      ])}
    </div>
    <h2>Slowest Tasks</h2>
    {_html_table(slowest_tasks, ['task', 'status', 'runtime', 'outputs', 'files'])}
    <h2>Largest Task Outputs</h2>
    {_html_table(largest_task_outputs, ['task', 'status', 'runtime', 'outputs', 'files'])}
    <h2>Largest Top-Level Output Directories</h2>
    {_html_table(largest_top_dirs, ['path', 'size', 'files'])}
  </section>

  <section class=\"panel\">
    <h2>Key Outputs</h2>
    {_render_path_list(key_outputs)}
  </section>

  <section class=\"panel\">
    <h2>Output Tree</h2>
    {_html_table(output_rows, ['top_dir', 'file_count'])}
  </section>

  <section class=\"panel\">
    <h2>Largest Runtime Files</h2>
    {_html_table(largest_outputs, ['path', 'size'])}
  </section>

  <section class=\"panel\">
    <h2>Report Artifacts</h2>
    <ul class=\"paths\">
      <li><span class=\"path-label\">Markdown summary</span><code>{html.escape(paths.summary_md.as_posix())}</code><span class=\"source\">human-readable</span></li>
      <li><span class=\"path-label\">Annotated output tree</span><code>{html.escape((paths.report_html.parent / 'output_tree.html').as_posix())}</code><span class=\"source\">human-readable</span></li>
      <li><span class=\"path-label\">Provenance</span><code>{html.escape(paths.provenance_json.as_posix())}</code><span class=\"source\">machine-readable</span></li>
      <li><span class=\"path-label\">QC summary</span><code>{html.escape(paths.qc_summary_json.as_posix())}</code><span class=\"source\">machine-readable</span></li>
    </ul>
  </section>

  <div class=\"footer\">
    SlicerDSpace keeps detailed manifests and QC JSON for reproducibility. This HTML report intentionally summarizes the run first; use the JSON files for exhaustive paths and raw task metadata.
  </div>
</main>
</body>
</html>
"""


def write_subject_reports(
    *,
    subject_root: Path,
    manifest_path: Path,
    output_root: Optional[Path] = None,
    run_name: Optional[str] = None,
) -> SubjectReportPaths:
    """Write HTML, provenance, and QC summary artifacts for one run."""

    manifest_path = manifest_path.expanduser().resolve()
    manifest = _read_json(manifest_path)
    subject_root = subject_root.expanduser().resolve()
    layout = subject_layout(subject_root, output_root=output_root)
    normalized_run = normalize_run_name(run_name or str(manifest.get("run_name") or "run"))
    report_dir = layout.slicer_scene_report
    paths = SubjectReportPaths(
        report_html=report_dir / f"{normalized_run}_report.html",
        provenance_json=report_dir / f"{normalized_run}_provenance.json",
        qc_summary_json=report_dir / f"{normalized_run}_qc_summary.json",
        summary_md=report_dir / f"{normalized_run}_summary.md",
        report_html_latest=report_dir / "report.html",
        provenance_json_latest=report_dir / "provenance.json",
        qc_summary_json_latest=report_dir / "qc_summary.json",
        summary_md_latest=report_dir / "summary.md",
    )
    output_tree = _collect_output_tree(layout.outputs)
    qc_records = _collect_qc_files(layout.outputs, extra_roots=(layout.cache_stages,))
    provenance = _build_provenance(
        manifest_path=manifest_path,
        manifest=manifest,
        output_tree=output_tree,
        qc_records=qc_records,
    )
    qc_summary = _build_qc_summary(
        manifest_path=manifest_path,
        manifest=manifest,
        output_tree=output_tree,
        qc_records=qc_records,
    )
    report_html = _build_html_report(
        manifest=manifest,
        provenance=provenance,
        qc_summary=qc_summary,
        paths=paths,
    )
    summary_md = _build_markdown_summary(
        manifest=manifest,
        provenance=provenance,
        qc_summary=qc_summary,
        paths=paths,
    )
    _write_json(paths.provenance_json, provenance)
    _write_json(paths.qc_summary_json, qc_summary)
    _write_text(paths.report_html, report_html)
    _write_text(paths.summary_md, summary_md)
    _write_json(paths.provenance_json_latest, provenance)
    _write_json(paths.qc_summary_json_latest, qc_summary)
    _write_text(paths.report_html_latest, report_html)
    _write_text(paths.summary_md_latest, summary_md)
    return paths
