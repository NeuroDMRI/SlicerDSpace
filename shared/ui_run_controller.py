"""Pure run-controller helpers for the SlicerDSpace UI."""

from __future__ import annotations

import json
import math
import os
import re
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

from shared.orchestrator import normalize_orchestrator_targets, orchestrator_closure
from shared.topup_parameters import TopupParameters, topup_cli_args, topup_orchestrator_kwargs
from shared.user_config import canonical_microstructure_stages, selected_microstructure_stages


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
    connectome_compress_mrtrix_dense: Optional[bool]
    wma_input_tractography_text: str = ""
    topup: TopupParameters = TopupParameters()
    brain_mask_backend: str = "pnl_cnn"
    brain_mask_device: str = "auto"
    pnl_mask_route: str = "auto"
    microstructure_dki: bool = False
    microstructure_ifw: bool = False
    ifw_echo_time_ms: Optional[float] = None
    dki_fit_method: str = "MKCURVE"
    wmql_query_preset: str = "official"
    wmql_query_selection: str = ""
    wma_atlas: str = "org"
    max_parallel_models: int = 1
    ukf_parameters_text: str = '{}'
    csd_act_parameters_text: str = '{}'
    dti_parameters_text: str = '{}'
    saf_parameters_text: str = '{}'
    preprocessing_dipy_run_gibbs: bool = True
    microstructure_fit_method: str = "wls"
    freewater_parameters_text: str = '{}'
    ifw_parameters_text: str = '{}'
    tractcloud_parameters_text: str = '{}'
    # Targets the user ticked, before the prerequisite closure; the CLI recomputes the closure
    # from these so the run manifest can record which targets were added automatically.
    requested_targets: tuple[str, ...] = ()
    # Algorithms after the closure (WMA adds UKF). Only used to ship UKF options; the CLI and the
    # orchestrator get the user's ``tractography_backend`` and recompute the closure with provenance.
    closure_tractography_backend: str = ""
    preprocessing_stages: str = ""
    connectome_metrics: str = ""
    # Batch 2 item 5: tractography source per bundle algorithm ({"wmql"|"wma"|"tractcloud": source}); a missing key
    # is "auto" (today's routing). Only explicit values reach the CLI as --bundle-source tokens.
    bundle_sources: dict = field(default_factory=dict)


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


def microstructure_dti_enabled(stages_text: str) -> bool:
    """True when DTI was explicitly selected in Microstructure (defaults to True when unset)."""

    text = (stages_text or "").strip().lower()
    if not text:
        return True
    return "dti" in selected_microstructure_stages(text)


def microstructure_ifw_enabled(stages_text: str, *, legacy_ifw: bool = False) -> bool:
    """True when the iFW stage is selected (T-C2). The three-flag API above keeps its shape."""

    text = (stages_text or "").strip().lower()
    if not text:
        return bool(legacy_ifw)
    return "ifw" in canonical_microstructure_stages(text)


@dataclass(frozen=True)
class DataFilteredRun:
    """The microstructure part of a run request after the data-blocked algorithms are removed (B3)."""

    flags: ProductTargetFlags
    stages: tuple[str, ...]  # the stored stages that still run, in catalog order (empty: the family is left out)
    skipped: tuple[str, ...]  # the stored stages that the data cannot run, in catalog order
    dti: bool
    freewater: bool
    noddi: bool
    dki: bool
    ifw: bool
    # b2-os F1: the tracking algorithms of the request when an OS block removed one (SAF); "" keeps the stored text.
    tractography_backend: str = ""


def filter_run_for_os_blocked_surface(run: DataFilteredRun, tractography_backend: str, closure_of):
    """b2-os F1: skip the OS-blocked Surface postprocess and every ticked target whose closure requires it.

    ``closure_of(flags, tractography_backend)`` returns the prerequisite closure of a request. Returns
    ``(DataFilteredRun, os_skipped)`` or ``None`` when the request does not need Surface postprocess. ``os_skipped``
    names what this filter left out, in order: ``surface_postprocess`` (when ticked), ``saf`` (the tracking
    algorithm, when it would run), then the product targets that still need the step (for example Tractography and
    Fiber bundles when SAF is the only tracking algorithm). No other algorithm is put in the place of SAF.
    """

    from dataclasses import fields

    from shared.algorithm_selection import algorithms

    tracking = (tractography_backend or "csd_act").strip() or "csd_act"
    full = closure_of(run.flags, tracking)
    if "surface_postprocess" not in full.targets:
        return None
    flags, skipped = run.flags, []
    if flags.surface_postprocess:
        flags = replace(flags, surface_postprocess=False)
        skipped.append("surface_postprocess")
    if "saf" in full.targets and "saf" in full.tractography_algorithms:
        kept = tuple(name for name in algorithms(tracking) if name != "saf")
        skipped.append("saf")
        if kept:
            tracking = ",".join(kept)
    for name in (field.name for field in fields(ProductTargetFlags)):
        if getattr(flags, name) and "surface_postprocess" in closure_of(
                ProductTargetFlags(**{name: True}), tracking).targets:
            flags = replace(flags, **{name: False})
            skipped.append(name)
    return replace(run, flags=flags, skipped=tuple(run.skipped) + tuple(skipped),
                   tractography_backend=tracking), tuple(skipped)


def filter_run_for_data(flags: ProductTargetFlags, stages_text: str, blocked) -> Optional[DataFilteredRun]:
    """Remove data-blocked microstructure stages from a run request; ``None`` when nothing is removed.

    ``stages_text`` is the stored selection. It is never rewritten: only the request changes. When no stored stage
    is left, the Microstructure target is left out of the request (the other families still run).
    """

    if not flags.microstructure:
        return None
    stored = selected_microstructure_stages(stages_text)
    skipped = tuple(stage for stage in stored if stage in blocked)
    if not skipped:
        return None
    kept = tuple(stage for stage in stored if stage not in skipped)
    if not kept:
        flags = replace(flags, microstructure=False)
    return DataFilteredRun(
        flags=flags, stages=kept, skipped=skipped,
        dti="dti" in kept, freewater="freewater" in kept, noddi="noddi" in kept,
        dki="dki" in kept, ifw="ifw" in kept,
    )


def compress_mrtrix_dense_value(state: Any) -> Optional[bool]:
    """Three GUI states to the run value (T-N1): "on" -> True, "off" -> False, anything else "unset" -> None."""

    text = str(state or "unset").strip().lower()
    return {"on": True, "off": False}.get(text)


def parse_ifw_echo_time_ms(text: Any) -> Optional[float]:
    """Echo time (ms) from the GUI field: empty is unset (None); anything else must be a number >= 0."""

    value = str(text if text is not None else "").strip()
    if not value:
        return None
    try:
        number = float(value)
    except ValueError:
        raise ValueError("Echo time (ms) must be a number, for example 89.5") from None
    if not math.isfinite(number) or number < 0:
        raise ValueError("Echo time (ms) must be zero or a positive number")
    return number


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
    provided_tractography: tuple[str, ...] = (),
    microstructure_ifw: bool = False,
    microstructure_dti: bool = True,
    bundle_sources: Optional[dict] = None,
):
    """Return the prerequisite closure (C1) of the ticked targets.

    ``provided_tractography`` names targets whose tractogram the user supplied as a file (K2).
    ``bundle_sources`` is the tractography source per bundle algorithm (batch 2 item 5).
    """

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
        microstructure_run_ifw=microstructure_ifw,
        microstructure_run_dti=microstructure_dti,
        provided_tractography=tuple(provided_tractography),
        bundle_sources=bundle_sources,
    )


def run_bundle_sources(selection: RunSelection, *, with_wma_input: bool = True) -> dict:
    """Explicit bundle sources of the selected bundle algorithms (batch 2 item 5).

    A source of an algorithm that is not selected is dropped. With a supplied WMA tractogram file
    (``with_wma_input``), the WMA source is dropped: the file is the WMA input.
    """

    from shared.algorithm_selection import BUNDLES, algorithms
    from shared.bundle_sources import normalize_bundle_sources
    try:
        selected = set(algorithms(selection.bundle_backend, BUNDLES, "wmql"))  # "all" = every bundle algorithm
    except ValueError:
        selected = set()
    sources = {algorithm: source for algorithm, source in
               normalize_bundle_sources(selection.bundle_sources or {}, strict=False).items() if algorithm in selected}
    if with_wma_input and selection.wma_input_tractography_text.strip():
        sources.pop("wma", None)
    return sources


def selected_product_targets(
    flags: ProductTargetFlags,
    *,
    tractography_backend: str,
    bundle_backend: str,
    microstructure_freewater: bool,
    microstructure_noddi: bool,
    microstructure_dki: bool = False,
    provided_tractography: tuple[str, ...] = (),
    microstructure_ifw: bool = False,
    microstructure_dti: bool = True,
) -> tuple[str, ...]:
    """Return normalized product targets after backend expansion and prerequisite closure."""

    return selected_closure(
        flags,
        tractography_backend=tractography_backend,
        bundle_backend=bundle_backend,
        microstructure_freewater=microstructure_freewater,
        microstructure_noddi=microstructure_noddi,
        microstructure_dki=microstructure_dki,
        microstructure_ifw=microstructure_ifw,
        microstructure_dti=microstructure_dti,
        provided_tractography=provided_tractography,
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
        f"Backends: tractography={(backends.tractography or 'csd_act').strip() or 'csd_act'}, "
        f"bundles={(backends.bundles or 'wmql').strip() or 'wmql'}, "
        f"microstructure={','.join(micro_stages)}, "
        f"connectome={(backends.connectome or 'python').strip() or 'python'}"
    )
    return summary, target_text


def streamline_control_state(tractography_backend: str) -> StreamlineControlState:
    """Return label, enabled state and tooltip for streamline-count controls."""

    backend = (tractography_backend or "ukf").strip() or "ukf"
    from shared.algorithm_selection import algorithms
    uses_act_streamlines = bool(set(algorithms(backend)) & {"csd_act", "saf"})
    if len(algorithms(backend)) > 1:
        description = (
            "Requested streamlines for selected CSD/ACT or SAF variants only. "
            "UKF uses its own seeds-per-voxel setting; DTI deterministic tracking ignores this count."
        )
    elif backend == "csd_act":
        description = (
            "Number of streamlines requested from CSD/ACT. "
            "One million is recommended for analysis-scale bundle and connectome coverage."
        )
    elif backend == "saf":
        description = (
            "Number of streamlines generated during the CSD/ACT stage before SAF filtering. "
            "One million is recommended when SAF, bundle analysis or connectome outputs are selected."
        )
    elif backend == 'ukf':
        description = 'Official UKF uses seeds per voxel in Advanced settings, not an ACT target streamline count.'
    else:
        description = (
            "DTI deterministic tracking does not use ACT streamline selection. "
            "This setting is ignored for the current tractography backend."
        )
    return StreamlineControlState(
        enabled=uses_act_streamlines,
        label="ACT/SAF streamlines" if uses_act_streamlines else "Streamline count",
        tooltip=description,
    )


def active_bundle_backend_sections(bundle_backend: str) -> set[str]:
    """Return optional backend path sections that should be highlighted."""

    backend = (bundle_backend or "tractcloud").strip().lower() or "tractcloud"
    active_sections: set[str] = set()
    from shared.algorithm_selection import algorithms, BUNDLES
    selected = algorithms(backend, BUNDLES, "tractcloud")
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
        return ("connectome", "connectome.streamline_count", "connectome.sift2_weighted")
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


def resolve_single_subject_workspace_root(output_root_text: str, subject_id: str = "") -> Optional[Path]:
    """Resolve the single-subject workspace directory under ``<output_root>/<subject_id>``.

    Strips a trailing ``output``/``outputs`` component when already present so
    Scheme A colocates ``input/``, ``preprocessed/``, ``cache/``, and ``outputs/``
    under the same ``<output_root>/<subject_id>`` parent directory. Empty input
    returns ``None`` so callers fall back to the default workspace/subject root.
    """

    raw = str(output_root_text or "").strip()
    if not raw:
        return None
    base = Path(raw).expanduser()
    clean_sid = str(subject_id or "").strip().replace("/", "_").replace("\\", "_")
    if not clean_sid:
        return base.parent if base.name.lower() in {"output", "outputs"} else base
    if base.name.lower() in {"output", "outputs"} and base.parent.name == clean_sid:
        return base.parent
    if base.name == clean_sid:
        return base
    return base / clean_sid


def resolve_single_subject_output_root(output_root_text: str, subject_id: str = "") -> Optional[Path]:
    """Resolve the single-subject output directory under ``<output_root>/<subject_id>/outputs``.

    When ``output_root_text`` already ends with ``<subject_id>`` or
    ``<subject_id>/outputs``, duplicate ``<subject_id>``/``outputs`` segments are
    not nested. Empty input returns ``None`` so callers fall back to
    ``<subject_root>/outputs``.
    """

    raw = str(output_root_text or "").strip()
    if not raw:
        return None
    base = Path(raw).expanduser()
    clean_sid = str(subject_id or "").strip().replace("/", "_").replace("\\", "_")
    if not clean_sid:
        return base if base.name.lower() in {"output", "outputs"} else (base / "outputs")
    if base.name.lower() in {"output", "outputs"} and base.parent.name == clean_sid:
        return base
    if base.name == clean_sid:
        return base / "outputs"
    return base / clean_sid / "outputs"


def build_single_subject_run_kwargs(
    selection: RunSelection,
    external_paths: dict[str, Optional[Path]],
    *,
    preflight_blocked_targets: dict[str, str],
) -> dict[str, Any]:
    """Build product-orchestration kwargs for one subject."""

    from shared.pipeline import resolve_subject_root_dir

    subject_root = resolve_subject_root_dir(Path(selection.subject_root_text).expanduser(), selection.subject_id)
    return {
        **_ukf_run_options(selection),
        **_tractography_param_run_options(selection),
        **_extra_algorithm_run_options(selection),
        "subject_id": selection.subject_id,
        "subject_root": subject_root,
        "output_root": resolve_single_subject_output_root(selection.output_root_text, selection.subject_id),
        "targets": selection.selected_targets,
        "requested_targets": selection.requested_targets,
        "run_name": selection.run_name,
        "tractography_backend": selection.tractography_backend,
        "bundle_backend": selection.bundle_backend,
        "wmql_query_preset": selection.wmql_query_preset,
        "wmql_query_selection": _wmql_selection(selection.wmql_query_selection, selection.wmql_query_preset),
        "max_parallel_models": selection.max_parallel_models,
        "brain_mask_backend": selection.brain_mask_backend,
        "brain_mask_device": selection.brain_mask_device,
        "pnl_mask_route": selection.pnl_mask_route,
        "resume": True,
        "microstructure_run_free_water": selection.microstructure_freewater,
        "microstructure_run_noddi": selection.microstructure_noddi,
        "microstructure_run_dki": selection.microstructure_dki,
        "microstructure_run_ifw": selection.microstructure_ifw,
        "microstructure_ifw_echo_time_ms": selection.ifw_echo_time_ms,
        "microstructure_dki_fit_method": selection.dki_fit_method,
        "microstructure_dki_matlab_executable": external_paths.get("dki_matlab_executable"),
        "microstructure_dki_mkcurve_root": external_paths.get("dki_mkcurve_root"),
        "select_streamlines": max(1, int(selection.select_streamlines)),
        "ddseg_device": selection.ddseg_device,
        "surface_device": selection.surface_device,
        "tractcloud_device": selection.ddseg_device,
        "model_assets_dir": external_paths.get("model_assets_dir"),
        "mrtrix_bin_dir": external_paths.get("mrtrix_bin_dir"),
        "preprocessing_mrtrix_bin_dir": external_paths.get("mrtrix_bin_dir"),
        "preprocessing_fsl_bin_dir": external_paths.get("preprocessing_fsl_bin_dir"),
        "preprocessing_execution_backend": os.environ.get("SLICERDSPACE_PREPROCESSING_EXECUTION_BACKEND")
        or ("native" if sys.platform == "win32" else None),
        "postprocess_freesurfer_home": external_paths.get("postprocess_freesurfer_home"),
        "postprocess_runtime": external_paths.get("postprocess_runtime"),
        "postprocess_license_file": external_paths.get("postprocess_license_file"),
        "pnl_mask_python": external_paths.get("pnl_mask_python"),
        "pnl_mask_script": external_paths.get("pnl_mask_script"),
        "pnl_mask_model_dir": external_paths.get("pnl_mask_model_dir"),
        "pnl_mask_ants_bin_dir": external_paths.get("pnl_mask_ants_bin_dir"),
        "tract_querier_root": external_paths.get("tract_querier_root"),
        "wmql_python_executable": external_paths.get("wmql_python_executable"),
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
        **({"preprocessing_stages": selection.preprocessing_stages} if selection.preprocessing_stages else {}),
        **({"connectome_metrics": selection.connectome_metrics} if selection.connectome_metrics else {}),
        **({"bundle_sources": run_bundle_sources(selection)} if run_bundle_sources(selection) else {}),
        "preflight_blocked_targets": dict(preflight_blocked_targets),
        **topup_orchestrator_kwargs(selection.topup),
    }


def build_batch_run_kwargs(selection: RunSelection, *, batch_root: Path) -> dict[str, Any]:
    """Build product-orchestration kwargs for a batch run."""

    return {
        **_ukf_run_options(selection),
        **_tractography_param_run_options(selection),
        **_extra_algorithm_run_options(selection),
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
        "brain_mask_backend": selection.brain_mask_backend,
        "brain_mask_device": selection.brain_mask_device,
        "pnl_mask_route": selection.pnl_mask_route,
        "resume": True,
        "microstructure_run_free_water": selection.microstructure_freewater,
        "microstructure_run_noddi": selection.microstructure_noddi,
        "microstructure_run_dki": selection.microstructure_dki,
        "microstructure_run_ifw": selection.microstructure_ifw,
        "microstructure_ifw_echo_time_ms": selection.ifw_echo_time_ms,
        "microstructure_dki_fit_method": selection.dki_fit_method,
        "select_streamlines": max(1, int(selection.select_streamlines)),
        "ddseg_device": selection.ddseg_device,
        "surface_device": selection.surface_device,
        "tractcloud_device": selection.ddseg_device,
        "model_assets_dir": None,
        "preprocessing_execution_backend": os.environ.get("SLICERDSPACE_PREPROCESSING_EXECUTION_BACKEND")
        or ("native" if sys.platform == "win32" else None),
        "connectome_backend": selection.connectome_backend,
        "connectome_compress_mrtrix_dense": selection.connectome_compress_mrtrix_dense,
        **({"preprocessing_stages": selection.preprocessing_stages} if selection.preprocessing_stages else {}),
        **({"connectome_metrics": selection.connectome_metrics} if selection.connectome_metrics else {}),
        # Batch runs do not forward the single-subject WMA input file, so the WMA source is kept.
        **({"bundle_sources": run_bundle_sources(selection, with_wma_input=False)}
           if run_bundle_sources(selection, with_wma_input=False) else {}),
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
    ("dki_matlab_executable", "--dki-matlab-executable"),
    ("dki_mkcurve_root", "--dki-mkcurve-root"),
    ("tract_querier_root", "--tract-querier-root"),
    ("wmql_python_executable", "--wmql-python-executable"),
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
    if common_run_kwargs.get("microstructure_run_ifw"):
        stages.append("ifw")
    return ",".join(stages)


def _append_ifw_args(args, common_run_kwargs):
    """Forward the iFW echo time (ms) and the researcher T2 values; none is added when unset."""
    for key, flag in (("microstructure_ifw_echo_time_ms", "--ifw-echo-time-ms"),
                      ("microstructure_ifw_t2_gm_ms", "--ifw-t2-gm-ms"),
                      ("microstructure_ifw_t2_csf_ms", "--ifw-t2-csf-ms")):
        value = common_run_kwargs.get(key)
        if value is not None:
            args.extend([flag, str(value)])


def _wmql_selection_cli_tokens(common_run_kwargs: dict[str, Any]) -> list[str]:
    """Format ``--wmql-query-selection`` without emitting an empty argv token.

    On Windows, passing an empty ``""`` token to ``Slicer.exe --launch PythonSlicer``
    causes ``CTKAppLauncher`` to dump its ``--launcher-*`` help table to stdout.
    """

    raw = common_run_kwargs.get("wmql_query_selection", ())
    joined = str(raw).strip() if isinstance(raw, str) else ",".join(str(item) for item in (raw or ()) if str(item))
    if joined:
        return ["--wmql-query-selection", joined]
    return ["--wmql-query-selection="]


def bundle_source_cli_tokens(sources) -> list[str]:
    """``--bundle-source ALG=SOURCE`` for each explicit value; nothing when every source is ``auto``."""

    from shared.bundle_sources import cli_tokens
    return cli_tokens(sources or {})


def build_background_cli_args(
    common_run_kwargs: dict[str, Any],
    external_paths: dict[str, Optional[Path]],
    *,
    repo_root: Path,
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
        "--brain-mask-backend",
        str(common_run_kwargs.get("brain_mask_backend") or "pnl_cnn"),
        "--brain-mask-device",
        str(common_run_kwargs.get("brain_mask_device") or "auto"),
        "--pnl-mask-route",
        str(common_run_kwargs.get("pnl_mask_route") or "auto"),
        "--tractography-backend",
        str(common_run_kwargs["tractography_backend"]),
        "--bundle-backend",
        str(common_run_kwargs["bundle_backend"]),
        "--wma-atlas", str(common_run_kwargs.get("wma_atlas") or "org"),
        "--wmql-query-preset",
        str(common_run_kwargs.get("wmql_query_preset") or "official"),
        *_wmql_selection_cli_tokens(common_run_kwargs),
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
        str(common_run_kwargs.get("tractcloud_device") or "auto"),
    ]
    if output_root is not None:
        args.extend(["--output-root", _path_text(output_root)])
    _append_ukf_args(args, common_run_kwargs)
    _append_tractography_param_args(args, common_run_kwargs)
    args.extend(["--microstructure-stages", microstructure_stages_arg(common_run_kwargs)])
    args.extend(["--dki-fit-method", str(common_run_kwargs.get("microstructure_dki_fit_method", "MKCURVE") or "MKCURVE")])
    _append_compress_arg(args, common_run_kwargs)
    _append_ifw_args(args, common_run_kwargs)
    _append_extra_algorithm_args(args, common_run_kwargs)
    if common_run_kwargs.get("preprocessing_execution_backend"):
        args.extend(["--preprocessing-execution-backend", str(common_run_kwargs["preprocessing_execution_backend"])])
    if common_run_kwargs.get("preprocessing_stages"):
        args.extend(["--preprocessing-stages", str(common_run_kwargs["preprocessing_stages"])])
    if common_run_kwargs.get("connectome_metrics"):
        args.extend(["--connectome-metrics", str(common_run_kwargs["connectome_metrics"])])
    args.extend(bundle_source_cli_tokens(common_run_kwargs.get("bundle_sources")))
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
    progress_file: Optional[Path] = None,
) -> list[str]:
    """Build the Slicer launcher args for batch background execution.

    ``progress_file`` (H3) is where the batch script writes its per-subject progress record for the GUI to poll.
    """

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
        arrange_inputs or "none",
        "--execute",
        "--targets",
        ",".join(common_run_kwargs.get("requested_targets") or common_run_kwargs["targets"]),
        "--brain-mask-backend",
        str(common_run_kwargs.get("brain_mask_backend") or "pnl_cnn"),
        "--brain-mask-device",
        str(common_run_kwargs.get("brain_mask_device") or "auto"),
        "--pnl-mask-route",
        str(common_run_kwargs.get("pnl_mask_route") or "auto"),
        "--tractography-backend",
        str(common_run_kwargs["tractography_backend"]),
        "--bundle-backend",
        str(common_run_kwargs["bundle_backend"]),
        "--wma-atlas", str(common_run_kwargs.get("wma_atlas") or "org"),
        "--wmql-query-preset",
        str(common_run_kwargs.get("wmql_query_preset") or "official"),
        *_wmql_selection_cli_tokens(common_run_kwargs),
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
        str(common_run_kwargs.get("tractcloud_device") or "auto"),
    ]
    if output_root is not None:
        args.extend(["--output-root", _path_text(output_root)])
    if progress_file is not None:
        args.extend(["--progress-file", _path_text(progress_file)])
    _append_ukf_args(args, common_run_kwargs)
    _append_tractography_param_args(args, common_run_kwargs)
    args.extend(["--microstructure-stages", microstructure_stages_arg(common_run_kwargs)])
    args.extend(["--dki-fit-method", str(common_run_kwargs.get("microstructure_dki_fit_method", "MKCURVE") or "MKCURVE")])
    _append_compress_arg(args, common_run_kwargs)
    _append_ifw_args(args, common_run_kwargs)
    _append_extra_algorithm_args(args, common_run_kwargs)
    if common_run_kwargs.get("preprocessing_execution_backend"):
        args.extend(["--preprocessing-execution-backend", str(common_run_kwargs["preprocessing_execution_backend"])])
    if common_run_kwargs.get("preprocessing_stages"):
        args.extend(["--preprocessing-stages", str(common_run_kwargs["preprocessing_stages"])])
    if common_run_kwargs.get("connectome_metrics"):
        args.extend(["--connectome-metrics", str(common_run_kwargs["connectome_metrics"])])
    # Forwarded unchanged to every subject's run_subject_pipeline.py call.
    args.extend(bundle_source_cli_tokens(common_run_kwargs.get("bundle_sources")))
    args.extend(topup_cli_args(topup_parameters_from_run_kwargs(common_run_kwargs)))
    _append_optional_path_args(args, external_paths)
    return args


_BATCH_MANIFEST_LINE_PREFIX = "[Batch] manifest:"


def _wmql_selection(value, preset='official'):
    from tractography.wmql_protocol import canonical_wmql_selection
    return canonical_wmql_selection(value, preset)


def _ukf_run_options(selection):
    from shared.algorithm_selection import contains_tracking
    if not contains_tracking(selection.closure_tractography_backend or selection.tractography_backend, 'ukf'):
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



def _append_compress_arg(args, common_run_kwargs):
    """None = not set (nothing is passed); True and False are passed explicitly."""
    value = common_run_kwargs.get("connectome_compress_mrtrix_dense")
    if value is not None:
        args.append("--compress-mrtrix-dense" if value else "--no-compress-mrtrix-dense")


def _append_ukf_args(args, kwargs):
    from shared.algorithm_selection import contains_tracking
    if contains_tracking(kwargs.get('tractography_backend'), 'ukf') or 'ukf_parameters' in kwargs:
        args.extend(('--ukf-parameters', json.dumps(kwargs.get('ukf_parameters', {}), sort_keys=True)))
        if kwargs.get('ukf_executable'):
            args.extend(('--ukf-executable', str(kwargs['ukf_executable'])))
        if kwargs.get('dwiconvert_executable'):
            args.extend(('--dwiconvert-executable', str(kwargs['dwiconvert_executable'])))


def _tractography_param_run_options(selection: RunSelection) -> dict[str, Any]:
    """Return validated non-empty tractography override dicts for CSD/ACT, DTI, and SAF."""

    from shared.algorithm_selection import contains_tracking
    from shared.ui_tractography_params import (
        parse_csd_act_overrides,
        parse_dti_overrides,
        parse_saf_overrides,
    )

    tracking = selection.closure_tractography_backend or selection.tractography_backend
    options: dict[str, Any] = {}
    csd_overrides = parse_csd_act_overrides(selection.csd_act_parameters_text or "{}")
    if csd_overrides and (contains_tracking(tracking, "csd_act") or contains_tracking(tracking, "saf")):
        options["csd_act_parameters"] = csd_overrides
    dti_overrides = parse_dti_overrides(selection.dti_parameters_text or "{}")
    if dti_overrides and contains_tracking(tracking, "dti_det"):
        options["dti_parameters"] = dti_overrides
    saf_overrides = parse_saf_overrides(selection.saf_parameters_text or "{}")
    if saf_overrides and contains_tracking(tracking, "saf"):
        options["saf_parameters"] = saf_overrides
    return options


def _extra_algorithm_run_options(selection: RunSelection) -> dict[str, Any]:
    """Return validated non-default overrides for audited preprocessing/microstructure/bundle algorithms."""

    from shared.ui_tractography_params import (
        parse_freewater_overrides,
        parse_ifw_overrides,
        parse_tractcloud_overrides,
        validate_microstructure_dti_fit_method,
    )

    options: dict[str, Any] = {}
    if selection.preprocessing_dipy_run_gibbs is False:
        options["preprocessing_dipy_run_gibbs"] = False
    fit_method = validate_microstructure_dti_fit_method(selection.microstructure_fit_method or "wls")
    if fit_method != "wls":
        options["microstructure_fit_method"] = fit_method
    fw_overrides = parse_freewater_overrides(selection.freewater_parameters_text or "{}")
    if "max_fraction_for_tissue" in fw_overrides and fw_overrides["max_fraction_for_tissue"] != 0.7:
        options["microstructure_free_water_max_fraction_for_tissue"] = fw_overrides["max_fraction_for_tissue"]
    ifw_overrides = parse_ifw_overrides(selection.ifw_parameters_text or "{}")
    if "t2_gm_ms" in ifw_overrides and ifw_overrides["t2_gm_ms"] != 95.0:
        options["microstructure_ifw_t2_gm_ms"] = ifw_overrides["t2_gm_ms"]
    if "t2_csf_ms" in ifw_overrides and ifw_overrides["t2_csf_ms"] != 1500.0:
        options["microstructure_ifw_t2_csf_ms"] = ifw_overrides["t2_csf_ms"]
    tc_overrides = parse_tractcloud_overrides(selection.tractcloud_parameters_text or "{}")
    if "batch_size" in tc_overrides and tc_overrides["batch_size"] != 2048:
        options["tractcloud_batch_size"] = tc_overrides["batch_size"]
    return options


def _append_tractography_param_args(args: list[str], kwargs: dict[str, Any]) -> None:
    """Append --csd-act-parameters / --dti-parameters / --saf-parameters only when non-empty."""

    for key, flag in (
        ("csd_act_parameters", "--csd-act-parameters"),
        ("dti_parameters", "--dti-parameters"),
        ("saf_parameters", "--saf-parameters"),
    ):
        value = kwargs.get(key)
        if isinstance(value, dict) and value:
            args.extend((flag, json.dumps(value, sort_keys=True)))


def _append_extra_algorithm_args(args: list[str], kwargs: dict[str, Any]) -> None:
    """Append non-default audited algorithm flags without altering default CLI invocations."""

    if kwargs.get("preprocessing_dipy_run_gibbs") is False:
        args.append("--preprocessing-skip-gibbs")
    fit_method = kwargs.get("microstructure_fit_method")
    if fit_method and str(fit_method).strip().lower() != "wls":
        args.extend(("--micro-fit-method", str(fit_method).strip().lower()))
    fw_max = kwargs.get("microstructure_free_water_max_fraction_for_tissue")
    if fw_max is not None and float(fw_max) != 0.7:
        args.extend(("--microstructure-free-water-max-fraction-for-tissue", str(float(fw_max))))
    tc_batch = kwargs.get("tractcloud_batch_size")
    if tc_batch is not None and int(tc_batch) != 2048:
        args.extend(("--tractcloud-batch-size", str(int(tc_batch))))


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


# H4: the launcher (CTK app launcher behind PythonSlicer) turns a worker that a signal killed into exit code 1
# plus this line ("error: [<path>/python-real] exit abnormally - Report the problem."), checked on macOS.
WORKER_ABNORMAL_EXIT_MARKER = "exit abnormally"
_MAX_SIGNAL = 64


def _exit_status_is_crash(exit_status: Any) -> bool:
    """True for ``QProcess.CrashExit`` (an int, an enum or its name); ``None`` and ``NormalExit`` are False."""

    if exit_status is None:
        return False
    for convert in (int, lambda value: int(getattr(value, "value"))):
        try:
            return convert(exit_status) == 1
        except (TypeError, ValueError, AttributeError):
            continue
    return "crash" in str(exit_status).lower()


def worker_kill_info(exit_code: int, exit_status: Any = None, *, abnormal_exit_seen: bool = False) -> Optional[dict[str, Any]]:
    """Return ``{"signal": N or None}`` when the worker was killed (by a signal, a crash or the OOM killer), else None.

    Signals show as ``CrashExit``, a negative exit code, ``128 + N`` (a shell or wrapper), or the launcher's
    "exit abnormally" line. Large codes (Windows NTSTATUS such as 0xC0000005) are crashes without a signal number.
    A normal exit with a small positive code (1, 2, ...) is a failure, not a kill.
    """

    try:
        code = int(exit_code)
    except (TypeError, ValueError):
        code = 0
    if _exit_status_is_crash(exit_status):
        number = abs(code) if 0 < abs(code) <= _MAX_SIGNAL else None
        return {"signal": number}
    if -_MAX_SIGNAL <= code < 0:
        return {"signal": -code}
    if 128 < code <= 128 + _MAX_SIGNAL:
        return {"signal": code - 128}
    if abnormal_exit_seen and code != 0:
        return {"signal": None}
    return None


_WINDOWS_NTSTATUS_LABELS = {
    0xC00000FD: "stack overflow (0xC00000FD)",
    0xC0000005: "access violation (0xC0000005)",
    0xC0000374: "heap corruption (0xC0000374)",
    0xC0000409: "stack buffer overrun (0xC0000409)",
}


def worker_killed_message(info: dict[str, Any], *, exit_code: Optional[int] = None) -> str:
    number = info.get("signal")
    if number is None and exit_code is not None:
        try:
            unsigned = int(exit_code) & 0xFFFFFFFF
        except (TypeError, ValueError):
            unsigned = 0
        nt_label = _WINDOWS_NTSTATUS_LABELS.get(unsigned)
        if nt_label:
            return (
                f"The worker crashed with a native Windows {nt_label}. "
                "Inspect the last output lines below (or close other memory-heavy apps) and run again."
            )
    where = f" (signal {number})" if number else ""
    return (f"The worker was killed{where}, possibly out of memory. "
            "Lower the number of parallel models or close other apps, then run again.")


def format_last_output(lines: Any, limit: int = 20) -> str:
    """The last ``limit`` non-empty output lines as one block for a failure message (H4)."""

    tail = [str(line) for line in list(lines or ()) if str(line).strip()][-limit:]
    return "\n".join(tail)


def classify_background_run_outcome(
    *,
    exit_code: int,
    cancelled: bool,
    manifest_available: bool,
    exit_status: Any = None,
    manifest_final: bool = True,
    abnormal_exit_seen: bool = False,
) -> str:
    """Classify how a finished background pipeline QProcess should be handled.

    A manifest, when present, is authoritative: the single-subject pipeline
    CLI returns exit code 2 (not 0) whenever any task failed, even though it
    still writes a complete manifest with correct per-task statuses. That
    manifest must be loaded and its per-row statuses shown -- via
    "manifest" -- instead of the whole run being marked as a blanket
    process failure. "process_failed" is reserved for what it actually
    means: the process exited non-zero and produced no manifest to explain
    why (e.g. a crash before any task ran).

    H4: a worker that a signal killed (see :func:`worker_kill_info`) is "worker_killed". A cancel wins over it
    (Cancel itself ends in a crash exit) and so does a final manifest (a crash while the process shut down after
    it wrote its result). A manifest that still says "running" is only a checkpoint, so it does not hide a kill:
    pass ``manifest_final=False`` for it.

    Returns one of "cancelled", "manifest", "worker_killed", "process_failed", "no_manifest".
    """

    if cancelled:
        return "cancelled"
    if manifest_available and manifest_final:
        return "manifest"
    if worker_kill_info(exit_code, exit_status, abnormal_exit_seen=abnormal_exit_seen) is not None:
        return "worker_killed"
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


# ------------------------------------------------------------------ worker liveness (H2)
# The CLI prints "[Heartbeat] ..." every 60 s and "[Stall warning] ..." lines (shared/run_heartbeat.py). A batch
# relays those lines of each subject with a "[<subject id>] " prefix (scripts/run_batch_pipeline.py). The GUI keeps
# heartbeats out of the log, shows them in the stage label, shows a stall as a warning and never kills anything.

HEARTBEAT_SECONDS_ENV_NAME = "SLICERDSPACE_HEARTBEAT_SECONDS"
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 60.0
SILENT_HEARTBEAT_INTERVALS = 3
_RELAYED_PREFIX = re.compile(r"^\[(?P<subject>[^\]\s][^\]]*)\] (?=\[(?:Heartbeat|Stall warning)\])")
_LAST_OUTPUT_SHOWN_AFTER_SECONDS = 30.0


def heartbeat_interval_seconds() -> float:
    """The heartbeat interval the worker uses: ``SLICERDSPACE_HEARTBEAT_SECONDS`` or 60 s."""

    try:
        value = float(os.environ.get(HEARTBEAT_SECONDS_ENV_NAME, ""))
    except ValueError:
        return DEFAULT_HEARTBEAT_INTERVAL_SECONDS
    return value if value > 0 else DEFAULT_HEARTBEAT_INTERVAL_SECONDS


def split_relayed_subject(line: str) -> tuple[Optional[str], str]:
    """Split ``"[sub-01] [Heartbeat] ..."`` into ``("sub-01", "[Heartbeat] ...")``; a plain line gives ``(None, line)``."""

    text = line.strip()
    match = _RELAYED_PREFIX.match(text)
    if not match:
        return None, text
    return match.group("subject"), text[match.end():]


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds}s" if seconds < 60 else f"{seconds // 60}m"


def worker_silence_seconds(now: float, last_output_at: float, interval_seconds: float,
                           intervals: int = SILENT_HEARTBEAT_INTERVALS) -> Optional[float]:
    """Seconds without any output line when that reaches ``intervals`` heartbeat intervals, else None (pure)."""

    silent = now - last_output_at
    return silent if silent >= intervals * interval_seconds else None


def worker_silence_message(silent_seconds: float) -> str:
    minutes = max(1, int(silent_seconds // 60))
    return (f"No output from the worker for {minutes} min (no heartbeat). "
            "It may be hung; press Cancel to stop it.")


def format_heartbeat_status(fields: dict[str, Any], *, subject: Optional[str] = None, cpu_active: Optional[bool] = None,
                            last_output_age: Optional[float] = None, zh: bool = False) -> str:
    """The stage-label text for one heartbeat: ``<stage> - running for Xm, CPU active``."""

    stage = str(fields.get("stage") or "").strip()
    if not stage or stage == "-":
        stage = "流程" if zh else "Pipeline"
    if subject:
        stage = f"{subject}: {stage}"
    try:
        elapsed = format_duration(float(fields.get("elapsed") or 0))
    except (TypeError, ValueError):
        elapsed = "0s"
    stalled = fields.get("stalled_min")
    show_age = last_output_age is not None and last_output_age >= _LAST_OUTPUT_SHOWN_AFTER_SECONDS
    if zh:
        parts = [f"{stage} - 已运行 {elapsed}"]
        if stalled is not None:
            parts.append(f"{stalled} 分钟无进展")
        elif cpu_active is not None:
            parts.append("CPU 活跃" if cpu_active else "CPU 空闲")
        if show_age:
            parts.append(f"最后输出在 {format_duration(last_output_age)}前")
        return "，".join(parts)
    parts = [f"{stage} - running for {elapsed}"]
    if stalled is not None:
        parts.append(f"no progress for {stalled} min")
    elif cpu_active is not None:
        parts.append("CPU active" if cpu_active else "CPU idle")
    if show_age:
        parts.append(f"last output {format_duration(last_output_age)} ago")
    return ", ".join(parts)


class WorkerLiveness:
    """Track the output of one background worker (pure: the caller passes the clock value).

    ``observe`` classifies a line and returns a dict with ``kind`` (``heartbeat``, ``stall`` or ``output``),
    the parsed fields, the relayed subject and ``resumed`` (True when the line ends a stall or a silence
    warning). A heartbeat without ``stalled_min`` also ends a stall.
    """

    def __init__(self, started_at: float, interval_seconds: Optional[float] = None) -> None:
        self.interval_seconds = float(interval_seconds) if interval_seconds else heartbeat_interval_seconds()
        self.last_line_at = started_at      # any line, heartbeat included: "is the worker alive?"
        self.last_output_at = started_at    # a line that is not a heartbeat: "is it making visible progress?"
        self.silent = False
        self._stalled: set = set()          # subjects with an open stall (None = the single run)
        self._cpu: dict[Optional[str], float] = {}

    @property
    def stalled(self) -> bool:
        return bool(self._stalled)

    def observe(self, line: str, now: float) -> dict[str, Any]:
        from shared.run_heartbeat import parse_heartbeat_line, parse_stall_warning

        subject, bare = split_relayed_subject(line)
        was_warning = bool(self._stalled) or self.silent
        self.last_line_at = now
        self.silent = False
        heartbeat = parse_heartbeat_line(bare)
        if heartbeat is not None:
            cpu = heartbeat.get("cpu_s")
            cpu_active = None
            if isinstance(cpu, (int, float)):
                before = self._cpu.get(subject)
                cpu_active = before is None or cpu > before
                self._cpu[subject] = float(cpu)
            if heartbeat.get("stalled_min") is not None:
                self._stalled.add(subject)
            else:
                self._stalled.discard(subject)
            return {"kind": "heartbeat", "fields": heartbeat, "subject": subject, "cpu_active": cpu_active,
                    "resumed": was_warning and not self._stalled}
        stall = parse_stall_warning(bare)
        if stall is not None:
            self._stalled.add(subject)
            return {"kind": "stall", "stall": stall, "subject": subject, "resumed": False}
        self.last_output_at = now
        # Output of one subject (or of the single run) ends only that subject's stall; a finished subject ends its own.
        self._stalled.discard(subject)
        finished = re.match(r"^\[Batch\] done (\S+)", bare)
        if finished:
            self._stalled.discard(finished.group(1))
        return {"kind": "output", "subject": subject, "resumed": was_warning and not self._stalled}

    def silence(self, now: float) -> Optional[float]:
        """Seconds of silence when the worker has been silent for three heartbeat intervals, else None."""

        return worker_silence_seconds(now, self.last_line_at, self.interval_seconds)

    def last_output_age(self, now: float) -> float:
        return max(0.0, now - self.last_output_at)


# --------------------------------------------------------------- batch progress (H3)

def rows_from_batch_progress(record: dict[str, Any]) -> tuple[list[dict[str, str]], int, int]:
    """Status-table rows plus ``(done, total)`` from the batch script's progress record."""

    results = {str(item.get("subject_id")): item for item in record.get("results") or () if isinstance(item, dict)}
    running = {str(item) for item in record.get("running") or ()}
    subjects = [str(item) for item in record.get("subjects") or ()]
    rows = []
    for subject_id in subjects:
        result = results.get(subject_id)
        if result is not None:
            try:
                code = int(result.get("returncode", 1))
            except (TypeError, ValueError):
                code = 1
            status = str(result.get("status") or "")
            state = "cancelled" if status == "cancelled" else ("completed" if code == 0 else "failed")
            reason = status or f"return code {code}"
        elif subject_id in running:
            state, reason = "running", ""
        else:
            state, reason = "queued", ""
        rows.append({"target": subject_id, "status": state, "reason": reason, "manifest_path": "", "qc_path": "", "logs_dir": ""})
    try:
        total = int(record.get("total") or len(subjects))
    except (TypeError, ValueError):
        total = len(subjects)
    return rows, len(results), total


def read_batch_progress(path: Path) -> Optional[dict[str, Any]]:
    """The batch progress record, or None while it is missing, half-written or of another kind."""

    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(record, dict) and record.get("kind") == "slicerdsurfer_batch_progress":
        return record
    return None


# ------------------------------------------------------------------ plan notices (LE hooks)
# Plan blockers of these categories hold a reason the user can act on (LE-1 NODDI shells are
# "acquisition_or_backend", LE-2 incomplete weights are "data", LE-3 GPU memory is "environment").
USER_ACTIONABLE_BLOCK_CATEGORIES = frozenset({"environment", "data", "acquisition_or_backend"})


def actionable_plan_blockers(plan: Any) -> dict[str, str]:
    """Blocked plan items whose reason the user can act on, as ``{target: reason}``."""

    return {
        item.target: item.reason
        for item in blocked_plan_items(plan)
        if getattr(item, "reason_category", "") in USER_ACTIONABLE_BLOCK_CATEGORIES
    }


def plan_warning_lines(plan: Any) -> list[str]:
    """``plan.warnings`` (LE-4 low disk space and similar advice) as clean text lines."""

    return [str(text).strip() for text in (getattr(plan, "warnings", ()) or ()) if str(text).strip()]


def is_advisory_memory_warning(warning_text: str) -> bool:
    """Return True for soft recommended-RAM advice that should be logged without a blocking modal."""

    text = str(warning_text or "").strip()
    return text.startswith("Available host memory for ") and "below the recommended " in text


def modal_plan_warnings(warnings: list[str]) -> list[str]:
    """Filter plan warnings down to high-severity items worth interrupting the user with a modal."""

    return [text for text in warnings if not is_advisory_memory_warning(text)]


def plan_notice_text(blockers: dict[str, str], runnable_targets: tuple[str, ...], warnings: list[str]) -> str:
    """The notice shown before a run starts: blocked products with their reasons, then warnings."""

    parts = []
    if blockers:
        blocked_text = "\n".join(f"- {target}: {reason}" for target, reason in blockers.items())
        runnable_text = ", ".join(target for target in runnable_targets if target != "validation") or "none"
        parts.append(
            "Some selected products cannot run with the current inputs or setup. "
            "Runnable targets will still run.\n\n"
            f"Blocked:\n{blocked_text}\n\n"
            f"Runnable targets: {runnable_text}"
        )
    if warnings:
        parts.append("Warnings:\n" + "\n".join(f"- {text}" for text in warnings))
    return "\n\n".join(parts)
