"""Concise human-readable QC text summaries under ``outputs/06_qc/``."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
from pathlib import Path
from typing import Any, Optional


_SKIP_QC_STATUSES = frozenset({"planned", "will_run", "blocked"})
_NOISY_SCALAR_KEYS = frozenset(
    {
        "schema",
        "kind",
        "version",
        "created_at",
        "started_at",
        "ended_at",
        "subject_id",
        "run_name",
        "status",
        "reason",
        "reason_category",
        "task_name",
        "reused",
        "runtime_seconds",
        "elapsed_seconds",
        "human_review",
    }
)
_NESTED_SUMMARY_KEYS = (
    "output_summary",
    "scientific_output_summary",
    "eligibility",
    "five_tt_stats",
    "bundle_assessment",
    "anatomical_prior_status",
    "tract_sampling",
    "sampling_screen",
)

QC_TASK_SUBPATHS: dict[str, Path] = {
    "preprocessing.dwi": Path("00_preprocessing.txt"),
    "segmentation.ddseg": Path("01_anatomical_priors/tissue_segmentation.txt"),
    "parcellation.ddparcel": Path("01_anatomical_priors/brain_parcellation.txt"),
    "surface.ddsurfer": Path("01_anatomical_priors/cortical_surface.txt"),
    "surface.ddsurfer_postprocess": Path("01_anatomical_priors/surface_morphometry.txt"),
    "microstructure.dti": Path("02_microstructure/diffusion_tensor.txt"),
    "microstructure.freewater": Path("02_microstructure/free_water.txt"),
    "microstructure.ifw": Path("02_microstructure/interstitial_free_water.txt"),
    "microstructure.noddi": Path("02_microstructure/neurite_microstructure.txt"),
    "microstructure.dki": Path("02_microstructure/diffusion_kurtosis.txt"),
    "tractography.dti_det": Path("03_tractography/single_tensor_tractography.txt"),
    "tractography.ukf": Path("03_tractography/multi_tensor_tractography.txt"),
    "tractography.csd_act": Path("03_tractography/anatomically_constrained_tractography.txt"),
    "tractography.act": Path("03_tractography/anatomically_constrained_tractography.txt"),
    "tractography.saf": Path("03_tractography/short_association_fibers.txt"),
    "bundles.wmql": Path("04_bundles/query_bundle_extraction.txt"),
    "bundles.wma": Path("04_bundles/fiber_clustering.txt"),
    "bundles.tractcloud": Path("04_bundles/bundle_identification.txt"),
    "connectome.streamline_count": Path("05_connectome/streamline_count.txt"),
    "connectome.sift2_weighted": Path("05_connectome/sift2_weighted.txt"),
}

QC_TASK_TITLES: dict[str, str] = {
    "preprocessing.dwi": "00 Preprocessing (DWI Preprocessing)",
    "segmentation.ddseg": "01 Anatomical Priors — Tissue Segmentation",
    "parcellation.ddparcel": "01 Anatomical Priors — Brain Parcellation",
    "surface.ddsurfer": "01 Anatomical Priors — Cortical Surface",
    "surface.ddsurfer_postprocess": "01 Anatomical Priors — Surface Morphometry",
    "microstructure.dti": "02 Microstructure — Diffusion Tensor (DTI)",
    "microstructure.freewater": "02 Microstructure — Free-Water Elimination",
    "microstructure.ifw": "02 Microstructure — Interstitial Free Water",
    "microstructure.noddi": "02 Microstructure — Neurite Microstructure (NODDI)",
    "microstructure.dki": "02 Microstructure — Diffusion Kurtosis (DKI)",
    "tractography.dti_det": "03 Tractography — Single-Tensor Deterministic",
    "tractography.ukf": "03 Tractography — Multi-Tensor Unscented Kalman Filter",
    "tractography.csd_act": "03 Tractography — Anatomically Constrained (CSD-ACT)",
    "tractography.act": "03 Tractography — Anatomically Constrained (ACT)",
    "tractography.saf": "03 Tractography — Short Association Fibers",
    "bundles.wmql": "04 Bundles — Query Bundle Extraction (WMQL)",
    "bundles.wma": "04 Bundles — Fiber Clustering (WMA)",
    "bundles.tractcloud": "04 Bundles — Bundle Identification (TractCloud)",
    "connectome.streamline_count": "05 Connectome — Streamline Count",
    "connectome.sift2_weighted": "05 Connectome — SIFT2-Weighted",
}


def qc_summary_path_for_task(qc_root: Path, task_name: str) -> Optional[Path]:
    """Return the canonical ``06_qc`` text file path for an orchestrator task."""

    rel = QC_TASK_SUBPATHS.get(task_name)
    if rel is not None:
        return qc_root / rel
    if task_name.startswith("connectome."):
        suffix = task_name.split(".", 1)[1].strip()
        if suffix:
            return qc_root / "05_connectome" / f"{suffix}.txt"
    return None


def _task_attr(task: Any, key: str, default: Any = None) -> Any:
    if isinstance(task, Mapping):
        return task.get(key, default)
    return getattr(task, key, default)


def _format_scalar(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, str):
        text = value.strip()
        if not text or len(text) > 160:
            return None
        return text
    return None


def _read_qc_json(qc_path: Any) -> dict[str, Any]:
    if not qc_path:
        return {}
    path = Path(qc_path)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _extract_qc_metrics(payload: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    metrics: list[str] = []
    warnings: list[str] = []

    checks = payload.get("checks")
    if isinstance(checks, Mapping) and checks:
        passed = sum(1 for val in checks.values() if val is True)
        failed_names = [str(name) for name, val in checks.items() if val is False]
        metrics.append(f"checks_passed: {passed}/{len(checks)}")
        if failed_names:
            metrics.append("failed_checks: " + ", ".join(failed_names[:10]))

    for key, value in payload.items():
        if key in _NOISY_SCALAR_KEYS or key in {"checks", "warnings", "scientific_warnings", "sampling_warnings", "error"}:
            continue
        formatted = _format_scalar(value)
        if formatted is not None:
            metrics.append(f"{key}: {formatted}")

    for section_key in _NESTED_SUMMARY_KEYS:
        section = payload.get(section_key)
        if not isinstance(section, Mapping):
            continue
        for sub_key, sub_val in section.items():
            formatted = _format_scalar(sub_val)
            if formatted is not None:
                metrics.append(f"{section_key}.{sub_key}: {formatted}")

    for warn_key in ("error", "warnings", "scientific_warnings", "sampling_warnings"):
        item = payload.get(warn_key)
        if isinstance(item, str) and item.strip():
            warnings.append(f"{warn_key}: {item.strip()}")
        elif isinstance(item, (list, tuple)):
            for entry in item[:10]:
                text = str(entry).strip()
                if text:
                    warnings.append(f"{warn_key}: {text}")

    return metrics[:24], warnings[:10]


def _is_inside_dir(candidate: Path, parent: Path) -> bool:
    try:
        candidate.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _display_path(raw: str, subject_root: Optional[Path]) -> str:
    candidate = Path(raw)
    if subject_root is not None:
        try:
            return candidate.relative_to(subject_root).as_posix()
        except ValueError:
            try:
                return candidate.resolve().relative_to(subject_root.resolve()).as_posix()
            except ValueError:
                pass
    return raw


def format_task_qc_text(
    task: Any,
    *,
    subject_id: str,
    run_name: Optional[str] = None,
    subject_root: Optional[Path] = None,
    cache_root: Optional[Path] = None,
) -> str:
    """Format a concise plain-text QC summary for one executed/reused task."""

    task_name = str(_task_attr(task, "name", "") or "").strip()
    status = str(_task_attr(task, "status", "") or "").strip()
    reason = str(_task_attr(task, "reason", "") or "").strip()
    runtime = _task_attr(task, "runtime_seconds")
    regional_stats = str(_task_attr(task, "regional_statistics", "") or "").strip()
    qc_path = _task_attr(task, "qc_path")
    outputs = _task_attr(task, "outputs") or {}

    title = QC_TASK_TITLES.get(task_name, task_name)
    lines = [
        f"Task: {title} ({task_name})",
        f"Subject: {subject_id}",
    ]
    if run_name:
        lines.append(f"Run: {run_name}")
    lines.append(f"Status: {status}")
    if reason:
        lines.append(f"Note: {reason}")
    if runtime is not None:
        try:
            lines.append(f"Runtime: {float(runtime):.2f} s")
        except (TypeError, ValueError):
            pass
    if regional_stats:
        lines.append(f"Regional statistics: {regional_stats}")

    qc_payload = _read_qc_json(qc_path)
    metrics, warnings = _extract_qc_metrics(qc_payload)
    if metrics:
        lines.append("")
        lines.append("QC metrics:")
        lines.extend(f"  - {item}" for item in metrics)
    if warnings:
        lines.append("")
        lines.append("Warnings / errors:")
        lines.extend(f"  - {item}" for item in warnings)

    if isinstance(outputs, Mapping):
        output_lines: list[str] = []
        for key, value in outputs.items():
            if value is None:
                continue
            text = str(value).strip()
            if not text:
                continue
            displayed = _display_path(text, subject_root)
            rel_parts = Path(displayed).parts
            in_cache = (rel_parts and rel_parts[0] == "cache") or (
                subject_root is not None and _is_inside_dir(Path(text), subject_root / "cache")
            ) or (
                cache_root is not None and _is_inside_dir(Path(text), cache_root)
            )
            if in_cache:
                continue
            output_lines.append(f"  - {key}: {displayed}")
        if output_lines:
            lines.append("")
            lines.append("Outputs:")
            lines.extend(output_lines)

    return "\n".join(lines).rstrip() + "\n"


def write_qc_text_summaries(
    qc_root: Path,
    tasks: Iterable[Any],
    *,
    subject_id: str,
    run_name: Optional[str] = None,
    subject_root: Optional[Path] = None,
    cache_root: Optional[Path] = None,
) -> dict[str, Path]:
    """Write concise ``.txt`` QC summaries for all executed or reused tasks."""

    effective_cache_root = cache_root
    if effective_cache_root is None:
        for candidate in (qc_root.parent.parent / "cache", qc_root.parent / "cache"):
            if candidate.is_dir():
                effective_cache_root = candidate
                break

    written: dict[str, Path] = {}
    for task in tasks:
        task_name = str(_task_attr(task, "name", "") or "").strip()
        status = str(_task_attr(task, "status", "") or "").strip().lower()
        if not task_name or status in _SKIP_QC_STATUSES:
            continue
        target_path = qc_summary_path_for_task(qc_root, task_name)
        if target_path is None:
            continue
        content = format_task_qc_text(
            task,
            subject_id=subject_id,
            run_name=run_name,
            subject_root=subject_root,
            cache_root=effective_cache_root,
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(content, encoding="utf-8")
        written[task_name] = target_path
    return written
