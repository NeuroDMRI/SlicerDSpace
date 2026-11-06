"""Pure run-controller helpers for the SlicerDSpace UI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from shared.orchestrator import normalize_orchestrator_targets, orchestrator_closure
from shared.topup_parameters import TopupParameters, topup_cli_args, topup_orchestrator_kwargs
from shared.user_config import canonical_microstructure_stages


@dataclass(frozen=True)
class ProductTargetFlags:
    preprocessing: bool = False
    surface: bool = False
    surface_postprocess: bool = False
    parcellation: bool = False
    segmentation: bool = False
    microstructure: bool = False
    tractography: bool = False
    bundles: bool = False
    connectome: bool = False


@dataclass(frozen=True)
class BackendSelection:
    preprocessing: str = "dipy"
    tractography: str = "csd_act"
    bundles: str = "wmql"
    microstructure_stages: str = "dti"
    connectome: str = "python"


@dataclass(frozen=True)
class StreamlineControlState:
    enabled: bool
    label: str
    tooltip: str


@dataclass(frozen=True)
class RunSelection:
    """Normalized UI state needed to build run kwargs."""

    subject_id: str
    subject_root_text: str
    output_root_text: str
    selected_targets: tuple[str, ...]
    run_name: str
    tractography_backend: str
    bundle_backend: str
    microstructure_freewater: bool
    microstructure_noddi: bool
    select_streamlines: int
    ddseg_device: str
    surface_device: str
    connectome_backend: str
    connectome_compress_mrtrix_dense: bool
    preprocessing_backend: str = "dipy"
    wma_input_tractography_text: str = ""
    topup: TopupParameters = TopupParameters()
    brain_mask_backend: str = "pnl_cnn"
    brain_mask_device: str = "auto"
    microstructure_dki: bool = False
    dki_fit_method: str = "WLS"
    wmql_query_preset: str = "official"
    wmql_query_selection: str = ""
    wma_atlas: str = "org"
    max_parallel_models: int = 1
    ukf_parameters_text: str = '{}'
    # Targets the user ticked, before the prerequisite closure; the CLI recomputes the closure
    # from these so the run manifest can record which targets were added automatically.
    requested_targets: tuple[str, ...] = ()


@dataclass(frozen=True)
class BatchRunOptions:
    """Validated batch-run options."""

    batch_root: Path
    batch_size: int
    arrange_inputs: str


def default_run_name_for_backend(_backend: str) -> str:
    """Return the neutral default run name shown in the UI."""

    return "Test1"


def validate_single_subject_inputs(subject_root_text: str, subject_id: str) -> str:
    """Return an error message if single-subject run inputs are incomplete."""

    if not (subject_root_text or "").strip() or not (subject_id or "").strip():
        return "Subject ID and subject root are required before running."
    return ""


def validate_batch_run_options(
    batch_root_text: str,
    *,
    fallback_subject_root_text: str,
    batch_size: int,
    arrange_inputs: str,
) -> tuple[BatchRunOptions | None, str]:
    """Validate and normalize batch-run options."""

    root_text = ((batch_root_text or "").strip() or (fallback_subject_root_text or "").strip())
    if not root_text:
        return None, "Batch root is required in Batch mode."
    batch_root = Path(root_text).expanduser()
    if not batch_root.is_dir():
        return None, f"Batch root does not exist:\n{batch_root}"
    return (
        BatchRunOptions(
            batch_root=batch_root,
            batch_size=max(1, int(batch_size or 1)),
            arrange_inputs=(arrange_inputs or "none").strip() or "none",
        ),
        "",
    )


def is_auto_generated_run_name(run_name: str) -> bool:
    """Return True for defaults or legacy backend-derived names."""

    normalized = (run_name or "").strip()
    if normalized in {"", "Test1"}:
        return True
    if normalized.endswith("_ui"):
        legacy_stem = normalized[:-3]
        return legacy_stem in {"csd_act", "tractography", "saf", "dti_det"}
    return False


def microstructure_stage_flags(stages_text: str, *, legacy_freewater: bool = False, legacy_noddi: bool = False) -> tuple[bool, bool]:
    """Preserve the two-flag compatibility API for older widgets and callers."""

    freewater, noddi, _ = microstructure_optional_stage_flags(
        stages_text, legacy_freewater=legacy_freewater, legacy_noddi=legacy_noddi,
    )
    return freewater, noddi


def microstructure_optional_stage_flags(
    stages_text: str, *, legacy_freewater: bool = False, legacy_noddi: bool = False, legacy_dki: bool = False,
) -> tuple[bool, bool, bool]:
    """Return independently selected Free Water, NODDI and DKI flags."""

    text = (stages_text or "").strip().lower()
    if not text:
        stages = {"dti"}
        if legacy_freewater:
            stages.add("freewater")
        if legacy_noddi:
            stages.add("noddi")
        if legacy_dki:
            stages.add("dki")
    else:
        stages = {stage.strip().lower() for stage in text.replace("+", ",").split(",") if stage.strip()}
    stages = set(canonical_microstructure_stages(stages))
    return "freewater" in stages, "noddi" in stages, "dki" in stages


def selected_workflows(flags: ProductTargetFlags) -> list[str]:
    """Return human-readable selected workflow names."""

    selected: list[str] = []
    if flags.preprocessing:
        selected.append("DWI Preprocessing")
    if flags.surface:
        selected.append("Surface / DDSurfer native STL")
    if flags.surface_postprocess:
        selected.append("Surface postprocess")
    if flags.parcellation:
        selected.append("Parcellation / DDParcel")
    if flags.segmentation:
        selected.append("Segmentation / DDSeg")
    if flags.microstructure:
        selected.append("Microstructure")
    if flags.tractography:
        selected.append("Tractography")
    if flags.bundles:
        selected.append("Fiber bundles")
    if flags.connectome:
        selected.append("Connectome")
    return selected


def selected_orchestrator_targets(flags: ProductTargetFlags) -> tuple[str, ...]:
    """Return legacy orchestrator targets excluding preprocessing/native surface."""

    targets = ["validation"]
    if flags.segmentation:
        targets.append("segmentation")
    if flags.parcellation:
        targets.append("parcellation")
    if flags.surface_postprocess:
        targets.append("surface_postprocess")
    if flags.microstructure:
        targets.append("microstructure")
    if flags.tractography:
        targets.append("tractography")
    if flags.bundles:
        targets.append("bundles")
    if flags.connectome:
        targets.append("connectome")
    return tuple(targets)


def selected_closure(
    flags: ProductTargetFlags,
    *,
    tractography_backend: str,
    bundle_backend: str,
    microstructure_freewater: bool,
    microstructure_noddi: bool,
    microstructure_dki: bool = False,
):
    """Return the prerequisite closure (C1) of the ticked targets."""

    product_targets = ["validation"]
    if flags.preprocessing:
        product_targets.append("preprocessing")
    if flags.segmentation:
        product_targets.append("segmentation")
    if flags.parcellation:
        product_targets.append("parcellation")
    if flags.surface:
        product_targets.append("surface")
    if flags.surface_postprocess:
        product_targets.append("surface_postprocess")
    if flags.microstructure:
        product_targets.append("microstructure")
    if flags.tractography:
        product_targets.append("tractography")
    if flags.bundles:
        product_targets.append("bundles")
    if flags.connectome:
        product_targets.append("connectome")
    return orchestrator_closure(
        tuple(product_targets),
        tractography_backend=(tractography_backend or "csd_act").strip() or "csd_act",
        bundle_backend=(bundle_backend or "wmql").strip() or "wmql",
        microstructure_run_free_water=microstructure_freewater,
        microstructure_run_noddi=microstructure_noddi,
        microstructure_run_dki=microstructure_dki,
    )


def selected_product_targets(
    flags: ProductTargetFlags,
    *,
    tractography_backend: str,
    bundle_backend: str,
    microstructure_freewater: bool,
    microstructure_noddi: bool,
    microstructure_dki: bool = False,
) -> tuple[str, ...]:
    """Return normalized product targets after backend expansion and prerequisite closure."""

    return selected_closure(
        flags,
        tractography_backend=tractography_backend,
        bundle_backend=bundle_backend,
        microstructure_freewater=microstructure_freewater,
        microstructure_noddi=microstructure_noddi,
        microstructure_dki=microstructure_dki,
    ).targets


def selected_environment_targets(product_targets: tuple[str, ...]) -> tuple[str, ...]:
    """Return unique environment targets to preflight."""

    return tuple(dict.fromkeys(target for target in product_targets if target))


def selected_products_summary(targets: tuple[str, ...], backends: BackendSelection) -> tuple[str, str]:
    """Build the compact selected-products summary and target tooltip text."""

    visible_targets = [target for target in targets if target != "validation"]
    target_text = ", ".join(visible_targets) if visible_targets else "none"
    micro_stages = canonical_microstructure_stages(backends.microstructure_stages)
    summary = (
        f"Selected targets: {target_text}\n"
        f"Backends: preprocessing={(backends.preprocessing or 'dipy').strip() or 'dipy'}, "
        f"tractography={(backends.tractography or 'csd_act').strip() or 'csd_act'}, "
        f"bundles={(backends.bundles or 'wmql').strip() or 'wmql'}, "
        f"microstructure={','.join(micro_stages)}, "
        f"connectome={(backends.connectome or 'python').strip() or 'python'}"
    )
    return summary, target_text


def streamline_control_state(tractography_backend: str) -> StreamlineControlState:
    """Return label, enabled state and tooltip for streamline-count controls."""

    backend = (tractography_backend or "csd_act").strip() or "csd_act"
    from shared.algorithm_selection import algorithms
    uses_act_streamlines = bool(set(algorithms(backend)) & {"csd_act", "saf"})
    if len(algorithms(backend)) > 1:
        description = (
            "Requested streamlines for selected MRtrix CSD/ACT or SAF variants only. "
            "UKF uses its own seeds-per-voxel setting; DTI deterministic tracking ignores this count."
        )
    elif backend == "csd_act":
        description = (
            "Number of streamlines requested from MRtrix CSD/ACT. "
            "One million is recommended for analysis-scale bundle and connectome coverage."
        )
    elif backend == "saf":
        description = (
            "Number of streamlines generated during the MRtrix CSD/ACT stage before SAF filtering. "
            "One million is recommended when SAF, bundle analysis or connectome outputs are selected."
        )
    elif backend == 'ukf':
        description = 'Official UKF uses seeds per voxel in Advanced settings, not an ACT target streamline count.'
    else:
        description = (
            "DTI deterministic tracking does not use MRtrix ACT streamline selection. "
            "This setting is ignored for the current tractography backend."
        )
    return StreamlineControlState(
        enabled=uses_act_streamlines,
        label="ACT/SAF streamlines" if uses_act_streamlines else "Streamline count",
        tooltip=description,
    )


def active_bundle_backend_sections(bundle_backend: str) -> set[str]:
    """Return optional backend path sections that should be highlighted."""

    backend = (bundle_backend or "wmql").strip().lower() or "wmql"
    active_sections: set[str] = set()
    from shared.algorithm_selection import algorithms, BUNDLES
    selected = algorithms(backend, BUNDLES, "wmql")
    if "wma" in selected:
        active_sections.add("wma")
    if "tractcloud" in selected:
        active_sections.add("tractcloud")
    return active_sections


def task_keys_for_running_line(line: str) -> tuple[str, ...]:
    """Return status-table keys affected by one orchestrator progress line."""

    lowered = (line or "").lower()
    if "dwi preprocessing" in lowered:
        return ("preprocessing", "preprocessing.dwi")
    if "ddseg" in lowered:
        return ("segmentation", "segmentation.ddseg")
    if "ddparcel" in lowered:
        return ("parcellation", "parcellation.ddparcel")
    if "ddsurfer surface" in lowered:
        return ("surface", "surface.ddsurfer")
    if "ddsurfer postprocess" in lowered:
        return ("surface_postprocess", "surface.ddsurfer_postprocess")
    if "microstructure" in lowered:
        return ("microstructure", "microstructure.dti")
    if "csd/act" in lowered:
        return ("tractography", "tractography.csd_act")
    if "dti tractography" in lowered:
        return ("tractography", "tractography.dti_det")
    if "wmql" in lowered:
        return ("bundles", "bundles.wmql")
    if "saf" in lowered:
        return ("saf", "tractography.saf")
    if "wma" in lowered:
        return ("wma", "bundles.wma")
    if "tractcloud" in lowered:
        return ("tractcloud", "bundles.tractcloud")
    if "connectome" in lowered:
        return ("connectome", "connectome.streamline_count")
    return ()


SUMMARY_STATE_ORDER = ("completed", "passed", "reused", "skipped", "partial", "blocked", "cancelled", "failed")


def run_summary_from_manifest(manifest: dict[str, Any]) -> tuple[str, str]:
    """Return a compact run summary from a manifest's QC summary path."""

    report_paths = manifest.get("report_paths") or {}
    qc_path = ""
    if isinstance(report_paths, dict):
        qc_path = str(report_paths.get("qc_summary_json_latest") or report_paths.get("qc_summary_json") or "")
    if not qc_path:
        return "", ""
    path = Path(qc_path).expanduser()
    if not path.is_file():
        return "", ""
    try:
        qc_summary = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"Report summary unavailable: {exc}", "unknown"
    summary = qc_summary.get("summary") if isinstance(qc_summary, dict) else {}
    if not isinstance(summary, dict):
        return "", ""
    status = str(summary.get("status") or "unknown")
    interpretation = str(summary.get("interpretation") or "").strip()
    elapsed = str(summary.get("elapsed_time") or "").strip()
    tasks = summary.get("tasks") if isinstance(summary.get("tasks"), dict) else {}
    ranked = {state: index for index, state in enumerate(SUMMARY_STATE_ORDER)}
    shown = sorted(
        ((state, count) for state, count in tasks.items() if count),
        key=lambda item: (ranked.get(item[0], len(ranked)), item[0]),
    )
    first = status.replace("_", " ").capitalize() + (f" · {elapsed}" if elapsed else "")
    lines = [first]
    if shown:
        lines.append(" · ".join(f"{count} {state.replace('_', ' ')}" for state, count in shown))
    if interpretation and status not in {"completed", "passed"}:
        lines.append(interpretation)
    return "\n".join(lines), status


RUN_STATUS_WORDS = {"completed", "passed", "failed", "running", "partial", "blocked", "planned", "unknown", "cancelled"}


def summary_names_run_status(text: str) -> bool:
    """True when a run-summary label names a run's status: the manifest summary ("Completed · 9m 22s") or a
    "Status: …" message; False for a plan, a fallback line or an empty label."""

    text = str(text or "").strip()
    if text.startswith("Status:"):
        return True
    head = text.split("\n", 1)[0].split(" · ", 1)[0].strip().lower().replace(" ", "_")
    return head in RUN_STATUS_WORDS


def preflight_blockers_for_targets(doctor_result: Any, selected_targets: tuple[str, ...]) -> dict[str, str]:
    """Return environment blockers limited to selected targets."""

    from shared import blocked_target_reasons

    selected = set(selected_targets)
    return {target: reason for target, reason in blocked_target_reasons(doctor_result).items() if target in selected}


def build_single_subject_run_kwargs(
    selection: RunSelection,
    external_paths: dict[str, Optional[Path]],
    *,
    preflight_blocked_targets: dict[str, str],
) -> dict[str, Any]:
    """Build product-orchestration kwargs for one subject."""

    return {
        **_ukf_run_options(selection),
        "subject_id": selection.subject_id,
        "subject_root": Path(selection.subject_root_text).expanduser(),
        "output_root": Path(selection.output_root_text).expanduser() if selection.output_root_text else None,
        "targets": selection.selected_targets,
        "requested_targets": selection.requested_targets,
        "run_name": selection.run_name,
        "tractography_backend": selection.tractography_backend,
        "bundle_backend": selection.bundle_backend,
        "wmql_query_preset": selection.wmql_query_preset,
        "wmql_query_selection": _wmql_selection(selection.wmql_query_selection, selection.wmql_query_preset),
        "max_parallel_models": selection.max_parallel_models,
        "preprocessing_backend": selection.preprocessing_backend,
        "brain_mask_backend": selection.brain_mask_backend,
        "brain_mask_device": selection.brain_mask_device,
        "resume": True,
        "microstructure_run_free_water": selection.microstructure_freewater,
        "microstructure_run_noddi": selection.microstructure_noddi,
        "microstructure_run_dki": selection.microstructure_dki,
        "microstructure_dki_fit_method": selection.dki_fit_method,
        "microstructure_dki_matlab_executable": external_paths.get("dki_matlab_executable"),
        "microstructure_dki_mkcurve_root": external_paths.get("dki_mkcurve_root"),
        "select_streamlines": max(1, int(selection.select_streamlines)),
        "ddseg_device": selection.ddseg_device,
        "surface_device": selection.surface_device,
        "tractcloud_device": "cuda" if selection.ddseg_device.startswith("cuda") else selection.ddseg_device,
        "model_assets_dir": external_paths.get("model_assets_dir"),
        "mrtrix_bin_dir": external_paths.get("mrtrix_bin_dir"),
        "preprocessing_mrtrix_bin_dir": external_paths.get("mrtrix_bin_dir"),
        "preprocessing_fsl_bin_dir": external_paths.get("preprocessing_fsl_bin_dir"),
        "postprocess_freesurfer_home": external_paths.get("postprocess_freesurfer_home"),
        "postprocess_runtime": external_paths.get("postprocess_runtime"),
        "postprocess_license_file": external_paths.get("postprocess_license_file"),
        "pnl_mask_python": external_paths.get("pnl_mask_python"),
        "pnl_mask_script": external_paths.get("pnl_mask_script"),
        "pnl_mask_model_dir": external_paths.get("pnl_mask_model_dir"),
        "pnl_mask_ants_bin_dir": external_paths.get("pnl_mask_ants_bin_dir"),
        "synthstrip_python": external_paths.get("synthstrip_python"),
        "tract_querier_root": external_paths.get("tract_querier_root"),
        "wma_root": external_paths.get("wma_root"),
        "wma_atlas_dir": external_paths.get("wma_atlas_dir"),
        "wma_atlas": selection.wma_atlas,
        "wma_input_tractography": (
            Path(selection.wma_input_tractography_text).expanduser()
            if selection.wma_input_tractography_text.strip()
            else None
        ),
        "slicer_executable": external_paths.get("slicer_executable"),
        "fiber_tract_measurements_cli": external_paths.get("fiber_tract_measurements_cli"),
        "tractcloud_root": external_paths.get("tractcloud_root"),
        "tractcloud_python_executable": external_paths.get("tractcloud_python_executable"),
        "tractcloud_data_dir": external_paths.get("tractcloud_data_dir"),
        "connectome_backend": selection.connectome_backend,
        "connectome_compress_mrtrix_dense": selection.connectome_compress_mrtrix_dense,
        "preflight_blocked_targets": dict(preflight_blocked_targets),
        **topup_orchestrator_kwargs(selection.topup),
    }


def build_batch_run_kwargs(selection: RunSelection, *, batch_root: Path) -> dict[str, Any]:
    """Build product-orchestration kwargs for a batch run."""

    return {
        **_ukf_run_options(selection),
        "subject_id": "batch",
        "subject_root": batch_root,
        "output_root": Path(selection.output_root_text).expanduser() if selection.output_root_text else None,
        "targets": selection.selected_targets,
        "requested_targets": selection.requested_targets,
        "run_name": selection.run_name,
        "tractography_backend": selection.tractography_backend,
        "bundle_backend": selection.bundle_backend,
        "wmql_query_preset": selection.wmql_query_preset,
        "wma_atlas": selection.wma_atlas,
        "wmql_query_selection": _wmql_selection(selection.wmql_query_selection, selection.wmql_query_preset),
        "max_parallel_models": selection.max_parallel_models,
        "preprocessing_backend": selection.preprocessing_backend,
        "brain_mask_backend": selection.brain_mask_backend,
        "brain_mask_device": selection.brain_mask_device,
        "resume": True,
        "microstructure_run_free_water": selection.microstructure_freewater,
        "microstructure_run_noddi": selection.microstructure_noddi,
        "microstructure_run_dki": selection.microstructure_dki,
        "microstructure_dki_fit_method": selection.dki_fit_method,
        "select_streamlines": max(1, int(selection.select_streamlines)),
        "ddseg_device": selection.ddseg_device,
        "surface_device": selection.surface_device,
        "tractcloud_device": "cuda" if selection.ddseg_device.startswith("cuda") else selection.ddseg_device,
        "model_assets_dir": None,
        "connectome_backend": selection.connectome_backend,
        "connectome_compress_mrtrix_dense": selection.connectome_compress_mrtrix_dense,
        "preflight_blocked_targets": {},
        # One reverse-PE b=0 file cannot describe every subject, so batch runs
        # only forward the per-protocol values (validation blocks the file).
        **topup_orchestrator_kwargs(
            TopupParameters(
                phase_encoding_direction=selection.topup.phase_encoding_direction,
                readout_time=selection.topup.readout_time,
            )
        ),
    }


def partial_run_notice_text(blockers: dict[str, str], runnable_targets: tuple[str, ...]) -> str:
    """Return the message for a partial run caused by environment blockers."""

    blocked_text = "\n".join(f"- {target}: {reason}" for target, reason in blockers.items())
    runnable_text = ", ".join(target for target in runnable_targets if target != "validation") or "none"
    return (
        "Some selected products are blocked by missing dependencies. "
        "Runnable targets will still run.\n\n"
        f"Blocked:\n{blocked_text}\n\n"
        f"Runnable targets: {runnable_text}"
    )


def rows_from_plan(plan: Any) -> tuple[list[dict[str, str]], dict[str, int], str]:
    """Build status-table rows and summary state from an orchestration plan."""

    rows = []
    for item in getattr(plan, "items", ()) or ():
        status = "queued" if item.status == "will_run" else item.status
        rows.append(
            {
                "target": item.target,
                "status": status,
                "reason": item.reason,
                "manifest_path": "",
                "qc_path": "",
                "logs_dir": "",
            }
        )
    summary = getattr(plan, "to_dict", lambda: {"summary": {}})().get("summary", {})
    blocked = int(summary.get("blocked", 0) or 0)
    will_run = int(summary.get("will_run", 0) or 0)
    skipped = int(summary.get("skipped", 0) or 0)
    state = "blocked" if blocked and not (will_run or skipped) else ("partial" if blocked else "planned")
    return rows, {"will_run": will_run, "skipped": skipped, "blocked": blocked}, state


def blocked_plan_items(plan: Any) -> list[Any]:
    """Return blocked plan items."""

    return [item for item in getattr(plan, "items", ()) or () if item.status == "blocked"]


def environment_blockers_from_plan(plan: Any) -> dict[str, str]:
    """Return environment-related blockers from a run plan."""

    return {
        item.target: item.reason
        for item in blocked_plan_items(plan)
        if getattr(item, "reason_category", "") == "environment"
    }


def runnable_targets_from_plan(plan: Any) -> tuple[str, ...]:
    """Return non-blocked targets from a run plan."""

    return tuple(item.target for item in getattr(plan, "items", ()) or () if item.status != "blocked")


def blocked_plan_log_text(items: list[Any]) -> str:
    """Return a compact log line for blocked plan items."""

    return "; ".join(f"{item.target}: {item.reason}" for item in items)


def rows_from_result(result: Any, manifest_data: dict[str, Any]) -> list[dict[str, str]]:
    """Build status-table rows from an in-process orchestration result."""

    manifest_tasks = {task.get("name"): task for task in manifest_data.get("tasks", []) if isinstance(task, dict)}
    rows = []
    for task in getattr(result, "tasks", ()) or ():
        manifest_task = manifest_tasks.get(task.name, {})
        rows.append(
            {
                "target": task.name,
                "status": task.status,
                "reason": task.reason,
                "manifest_path": manifest_task.get("normalized_manifest_path")
                or (task.manifest_path.as_posix() if task.manifest_path else ""),
                "qc_path": manifest_task.get("normalized_qc_path") or (task.qc_path.as_posix() if task.qc_path else ""),
                "logs_dir": manifest_task.get("normalized_logs_dir", ""),
            }
        )
    return rows


def rows_from_single_subject_manifest(manifest: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Build status-table rows and counts from a single-subject manifest."""

    rows = []
    counts = {"blocked": 0, "failed": 0, "completed_like": 0}
    for task in manifest.get("tasks", []):
        if not isinstance(task, dict):
            continue
        status = str(task.get("status") or "")
        if status == "blocked":
            counts["blocked"] += 1
        if status == "failed":
            counts["failed"] += 1
        if status in {"completed", "passed", "skipped"}:
            counts["completed_like"] += 1
        rows.append(
            {
                "target": str(task.get("name") or ""),
                "status": status,
                "reason": str(task.get("reason") or ""),
                "manifest_path": str(task.get("normalized_manifest_path") or task.get("manifest_path") or ""),
                "qc_path": str(task.get("normalized_qc_path") or task.get("qc_path") or ""),
                "logs_dir": str(task.get("normalized_logs_dir") or ""),
            }
        )
    if not rows and str(manifest.get("status") or "") == "cancelled":
        # GUI cancellation manifests record targets but no tasks.
        reason = str(manifest.get("reason") or "Run was cancelled.")
        for target in manifest.get("targets") or ():
            rows.append(
                {
                    "target": str(target),
                    "status": "cancelled",
                    "reason": reason,
                    "manifest_path": "",
                    "qc_path": "",
                    "logs_dir": "",
                }
            )
    return rows, counts


def rows_from_batch_manifest(manifest_path: Path, manifest: dict[str, Any]) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Build status-table rows and counts from a batch-run manifest."""

    rows = []
    counts = {"completed": 0, "failed": 0, "skipped": 0, "total": len(manifest.get("subjects", []) or [])}
    for result in manifest.get("results", []) or []:
        if not isinstance(result, dict):
            continue
        returncode = int(result.get("returncode", 1))
        status = "completed" if returncode == 0 else "failed"
        counts[status] += 1
        rows.append(
            {
                "target": str(result.get("subject_id") or "") + (" / " + result['variant'] if result.get('variant') else ''),
                "status": status,
                "reason": str(result.get('reason') or f"return code {returncode}"),
                "manifest_path": str(result.get('run_manifest_path') or manifest_path),
                "qc_path": "",
                "logs_dir": str(Path(str(result.get("stdout_log") or "")).parent) if result.get("stdout_log") else "",
            }
        )
    skipped = manifest.get("skipped", []) or []
    if manifest.get('experiment'):
        finished = {item.get('job_id') for item in manifest.get('results', [])}
        for job in manifest.get('subjects', []):
            if job.get('job_id') not in finished:
                rows.append(dict(target=job.get('subject_id', '') + ' / ' + job.get('variant', ''),
                                 status='cancelled' if manifest.get('status') == 'cancelled' else job.get('status', 'pending'),
                                 reason='', manifest_path=job.get('run_manifest_path', ''), qc_path='', logs_dir=''))
    counts["skipped"] = len(skipped)
    for item in skipped:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "target": str(item.get("subject_id") or item.get("path") or ""),
                "status": "skipped",
                "reason": str(item.get("reason") or "missing input"),
                "manifest_path": str(manifest_path),
                "qc_path": "",
                "logs_dir": "",
            }
        )
    return rows, counts


EXTERNAL_PATH_CLI_FLAGS: tuple[tuple[str, str], ...] = (
    ("model_assets_dir", "--model-assets-dir"),
    ("mrtrix_bin_dir", "--mrtrix-bin-dir"),
    ("mrtrix_bin_dir", "--preprocessing-mrtrix-bin-dir"),
    ("preprocessing_fsl_bin_dir", "--preprocessing-fsl-bin-dir"),
    ("postprocess_freesurfer_home", "--postprocess-freesurfer-home"),
    ("postprocess_runtime", "--postprocess-runtime"),
    ("postprocess_license_file", "--postprocess-license-file"),
    ("pnl_mask_python", "--pnl-mask-python"),
    ("pnl_mask_script", "--pnl-mask-script"),
    ("pnl_mask_model_dir", "--pnl-mask-model-dir"),
    ("pnl_mask_ants_bin_dir", "--pnl-mask-ants-bin-dir"),
    ("synthstrip_python", "--synthstrip-python"),
    ("dki_matlab_executable", "--dki-matlab-executable"),
    ("dki_mkcurve_root", "--dki-mkcurve-root"),
    ("tract_querier_root", "--tract-querier-root"),
    ("wma_root", "--wma-root"),
    ("wma_atlas_dir", "--wma-atlas-dir"),
    ("slicer_executable", "--slicer-executable"),
    ("fiber_tract_measurements_cli", "--fiber-tract-measurements-cli"),
    ("tractcloud_root", "--tractcloud-root"),
    ("tractcloud_python_executable", "--tractcloud-python-executable"),
    ("tractcloud_data_dir", "--tractcloud-data-dir"),
)


def _path_text(value: Path) -> str:
    return value.expanduser().as_posix()


def topup_parameters_from_run_kwargs(common_run_kwargs: dict[str, Any]) -> TopupParameters:
    """Recover the topup values carried in orchestration kwargs."""

    reverse = common_run_kwargs.get("preprocessing_reverse_phase_b0")
    return TopupParameters(
        phase_encoding_direction=common_run_kwargs.get("preprocessing_phase_encoding_direction") or None,
        readout_time=common_run_kwargs.get("preprocessing_readout_time"),
        reverse_phase_b0=Path(reverse) if reverse else None,
    )


def _append_optional_path_args(args: list[str], external_paths: dict[str, Optional[Path]]) -> None:
    for key, flag in EXTERNAL_PATH_CLI_FLAGS:
        value = external_paths.get(key)
        if value is not None:
            args.extend([flag, _path_text(value)])


def microstructure_stages_arg(common_run_kwargs: dict[str, Any]) -> str:
    """Return the CLI microstructure stage string from run kwargs."""

    stages = ["dti"]
    if common_run_kwargs.get("microstructure_run_free_water"):
        stages.append("freewater")
    if common_run_kwargs.get("microstructure_run_noddi"):
        stages.append("noddi")
    if common_run_kwargs.get("microstructure_run_dki"):
        stages.append("dki")
    return ",".join(stages)


def build_background_cli_args(
    common_run_kwargs: dict[str, Any],
    external_paths: dict[str, Optional[Path]],
    *,
    repo_root: Path,
    preprocessing_backend: str = "dipy",
) -> list[str]:
    """Build the Slicer launcher args for one subject background execution."""

    cli_script = repo_root / "shared" / "run_subject_pipeline_cli.py"
    if not cli_script.is_file():
        cli_script = repo_root / "run_subject_pipeline.py"
    output_root = common_run_kwargs.get("output_root")
    args = [
        "--launch",
        "PythonSlicer",
        cli_script.as_posix(),
        "--subject-root",
        _path_text(common_run_kwargs["subject_root"]),
        "--subject-id",
        str(common_run_kwargs["subject_id"]),
        "--run-name",
        str(common_run_kwargs["run_name"]),
        "--targets",
        ",".join(common_run_kwargs.get("requested_targets") or common_run_kwargs["targets"]),
        "--execute",
        "--print-manifest-only",
        "--preprocessing-backend",
        (preprocessing_backend or "dipy").strip() or "dipy",
        "--brain-mask-backend",
        str(common_run_kwargs.get("brain_mask_backend", "pnl_cnn")),
        "--brain-mask-device",
        str(common_run_kwargs.get("brain_mask_device", "auto")),
        "--tractography-backend",
        str(common_run_kwargs["tractography_backend"]),
        "--bundle-backend",
        str(common_run_kwargs["bundle_backend"]),
        "--wma-atlas", str(common_run_kwargs.get("wma_atlas", "org")),
        "--wmql-query-preset",
        str(common_run_kwargs.get("wmql_query_preset", "official")),
        '--wmql-query-selection', ','.join(common_run_kwargs.get('wmql_query_selection', ())),
        '--max-parallel-models', str(common_run_kwargs.get('max_parallel_models', 1)),
        "--connectome-backend",
        str(common_run_kwargs["connectome_backend"]),
        "--select-streamlines",
        str(common_run_kwargs["select_streamlines"]),
        "--ddseg-device",
        str(common_run_kwargs["ddseg_device"]),
        "--surface-device",
        str(common_run_kwargs["surface_device"]),
        "--tractcloud-device",
        str(common_run_kwargs.get("tractcloud_device", "auto")),
    ]
    if output_root is not None:
        args.extend(["--output-root", _path_text(output_root)])
    _append_ukf_args(args, common_run_kwargs)
    args.extend(["--microstructure-stages", microstructure_stages_arg(common_run_kwargs)])
    args.extend(["--dki-fit-method", str(common_run_kwargs.get("microstructure_dki_fit_method", "WLS"))])
    if common_run_kwargs.get("connectome_compress_mrtrix_dense"):
        args.append("--compress-mrtrix-dense")
    wma_input = common_run_kwargs.get("wma_input_tractography")
    if wma_input is not None:
        args.extend(["--wma-input-tractography", _path_text(wma_input)])
    args.extend(topup_cli_args(topup_parameters_from_run_kwargs(common_run_kwargs)))
    _append_optional_path_args(args, external_paths)
    return args


def build_batch_background_cli_args(
    common_run_kwargs: dict[str, Any],
    external_paths: dict[str, Optional[Path]],
    *,
    repo_root: Path,
    batch_root: Path,
    batch_size: int,
    arrange_inputs: str,
    preprocessing_backend: str = "dipy",
) -> list[str]:
    """Build the Slicer launcher args for batch background execution."""

    batch_script = repo_root / "scripts" / "run_batch_pipeline.py"
    output_root = common_run_kwargs.get("output_root")
    args = [
        "--launch",
        "PythonSlicer",
        batch_script.as_posix(),
        "--batch-root",
        _path_text(batch_root),
        "--batch-size",
        str(max(1, int(batch_size))),
        "--run-name",
        str(common_run_kwargs["run_name"]),
        "--arrange-inputs",
        arrange_inputs,
        "--execute",
        "--targets",
        ",".join(common_run_kwargs.get("requested_targets") or common_run_kwargs["targets"]),
        "--preprocessing-backend",
        (preprocessing_backend or "dipy").strip() or "dipy",
        "--brain-mask-backend",
        str(common_run_kwargs.get("brain_mask_backend", "pnl_cnn")),
        "--brain-mask-device",
        str(common_run_kwargs.get("brain_mask_device", "auto")),
        "--tractography-backend",
        str(common_run_kwargs["tractography_backend"]),
        "--bundle-backend",
        str(common_run_kwargs["bundle_backend"]),
        "--wma-atlas", str(common_run_kwargs.get("wma_atlas", "org")),
        "--wmql-query-preset",
        str(common_run_kwargs.get("wmql_query_preset", "official")),
        '--wmql-query-selection', ','.join(common_run_kwargs.get('wmql_query_selection', ())),
        '--max-parallel-models', str(common_run_kwargs.get('max_parallel_models', 1)),
        "--connectome-backend",
        str(common_run_kwargs["connectome_backend"]),
        "--select-streamlines",
        str(common_run_kwargs["select_streamlines"]),
        "--ddseg-device",
        str(common_run_kwargs["ddseg_device"]),
        "--surface-device",
        str(common_run_kwargs["surface_device"]),
        "--tractcloud-device",
        str(common_run_kwargs.get("tractcloud_device", "auto")),
    ]
    if output_root is not None:
        args.extend(["--output-root", _path_text(output_root)])
    _append_ukf_args(args, common_run_kwargs)
    args.extend(["--microstructure-stages", microstructure_stages_arg(common_run_kwargs)])
    args.extend(["--dki-fit-method", str(common_run_kwargs.get("microstructure_dki_fit_method", "WLS"))])
    if common_run_kwargs.get("connectome_compress_mrtrix_dense"):
        args.append("--compress-mrtrix-dense")
    # Forwarded unchanged to every subject's run_subject_pipeline.py call.
    args.extend(topup_cli_args(topup_parameters_from_run_kwargs(common_run_kwargs)))
    _append_optional_path_args(args, external_paths)
    return args


_BATCH_MANIFEST_LINE_PREFIX = "[Batch] manifest:"


def _wmql_selection(value, preset='official'):
    from tractography.wmql_protocol import canonical_wmql_selection
    return canonical_wmql_selection(value, preset)


def _ukf_run_options(selection):
    from shared.algorithm_selection import contains_tracking
    if not contains_tracking(selection.tractography_backend, 'ukf'):
        return {}
    from tractography.ukf import parameters, resolve_executable
    from dataclasses import asdict
    options = asdict(parameters(json.loads(selection.ukf_parameters_text or '{}')))
    try:
        binary = resolve_executable()
    except (OSError, RuntimeError):
        binary = None  # Dependency Doctor supplies the actionable installation message.
    try:
        converter = resolve_executable(name='DWIConvert')
    except (OSError, RuntimeError):
        converter = None
    return dict(ukf_parameters=options, ukf_executable=binary, dwiconvert_executable=converter)


def _append_ukf_args(args, kwargs):
    from shared.algorithm_selection import contains_tracking
    if contains_tracking(kwargs.get('tractography_backend'), 'ukf'):
        args.extend(('--ukf-parameters', json.dumps(kwargs.get('ukf_parameters', {}), sort_keys=True)))
        if kwargs.get('ukf_executable'):
            args.extend(('--ukf-executable', str(kwargs['ukf_executable'])))
        if kwargs.get('dwiconvert_executable'):
            args.extend(('--dwiconvert-executable', str(kwargs['dwiconvert_executable'])))


def parse_manifest_line(line: str) -> Optional[str]:
    """Return the manifest path encoded in one line of background process output.

    Recognizes two formats emitted on the (merged) stdout stream:
    - A bare line ending in "_orchestration_manifest.json" -- the
      single-subject pipeline CLI's manifest filename convention.
    - "[Batch] manifest: <path>" -- scripts/run_batch_pipeline.py's batch
      manifest, whose filename ("batch_<timestamp>.json") does not match
      the single-subject convention and so needs an explicit marker.

    Returns None if the line matches neither format.
    """

    stripped = (line or "").strip()
    if not stripped:
        return None
    if stripped.endswith("_orchestration_manifest.json"):
        return stripped
    if stripped.startswith(_BATCH_MANIFEST_LINE_PREFIX):
        path_text = stripped[len(_BATCH_MANIFEST_LINE_PREFIX):].strip()
        return path_text or None
    return None


def extract_background_manifest_path(buffer_text: str, explicit_path: Optional[Path] = None) -> Optional[Path]:
    """Extract the latest existing orchestration/batch manifest path from process output."""

    if explicit_path:
        return explicit_path
    candidates: list[Path] = []
    for line in (buffer_text or "").splitlines():
        path_text = parse_manifest_line(line)
        if path_text:
            candidates.append(Path(path_text))
    return next((path for path in reversed(candidates) if path.is_file()), None)


def classify_background_run_outcome(*, exit_code: int, cancelled: bool, manifest_available: bool) -> str:
    """Classify how a finished background pipeline QProcess should be handled.

    A manifest, when present, is authoritative: the single-subject pipeline
    CLI returns exit code 2 (not 0) whenever any task failed, even though it
    still writes a complete manifest with correct per-task statuses. That
    manifest must be loaded and its per-row statuses shown -- via
    "manifest" -- instead of the whole run being marked as a blanket
    process failure. "process_failed" is reserved for what it actually
    means: the process exited non-zero and produced no manifest to explain
    why (e.g. a crash before any task ran).

    Returns one of "cancelled", "manifest", "process_failed", "no_manifest".
    """

    if cancelled:
        return "cancelled"
    if manifest_available:
        return "manifest"
    if exit_code != 0:
        return "process_failed"
    return "no_manifest"


def cancelled_manifest_path_and_payload(
    context: dict[str, Any],
    *,
    cancelled_at: str,
) -> tuple[Path, dict[str, Any]]:
    """Build the cancellation manifest path and JSON payload."""

    subject_root = Path(context.get("subject_root") or ".").expanduser().resolve()
    output_root = context.get("output_root")
    outputs = Path(output_root).expanduser().resolve() if output_root else subject_root / "outputs"
    run_name = str(context.get("run_name") or "slicerdsurfer")
    manifest_path = outputs / "00_run_metadata" / f"{run_name}_cancelled_orchestration_manifest.json"
    payload = {
        "kind": "slicerddsurfer_product_orchestration",
        "version": 1,
        "status": "cancelled",
        "subject_id": str(context.get("subject_id") or ""),
        "subject_root": subject_root.as_posix(),
        "output_root": outputs.as_posix(),
        "run_name": run_name,
        "targets": list(context.get("targets") or []),
        "started_at": context.get("started_at"),
        "cancelled_at": cancelled_at,
        "reason": "User cancelled the background SlicerDSpace job from the UI.",
    }
    return manifest_path, payload
