"""Lightweight subject-root product orchestrator with resume support.

This is intentionally small: it gives the Slicer UI one backend entrypoint for
target-product execution while leaving each product pipeline responsible for its
own scientific details and manifest/QC files.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from shared.io_layout import (
    ANATOMY_OUTPUT_NAME,
    LEGACY_ANATOMY_OUTPUT_NAME,
    archive_development_outputs,
    canonical_connectome_output_dir,
    materialize_canonical_output_tree,
    normalize_run_name,
    subject_layout,
)
from shared.validation import InputValidationConfig, validate_subject_inputs
from shared.environment import resolve_auto_torch_device
from shared.external_process import interpreter_path
from shared.model_assets import model_asset_path
from shared.output_contracts import outputs_readable, tractography_readable


SUPPORTED_TARGETS = {
    "validation",
    "preprocessing",
    "segmentation",
    "parcellation",
    "surface",
    "surface_postprocess",
    "microstructure",
    "microstructure_freewater",
    "microstructure_noddi",
    "microstructure_dki",
    "tractography",
    "bundles",
    "connectome",
    "saf",
    "wma",
    "tractcloud",
}

TRACTOGRAPHY_BACKENDS = {"csd_act", "ukf", "dti_det", "saf"}
BUNDLE_BACKENDS = {"wmql", "wma", "tractcloud"}
WMA_MIN_ANATOMICAL_TRACT_COUNT = 73
WMA_MIN_NONEMPTY_ANATOMICAL_TRACT_COUNT = 55

from shared.target_closure import TARGET_ORDER, expand_closure  # noqa: E402


@dataclass(frozen=True)
class ProductOrchestratorConfig:
    subject_id: str
    subject_root: Path
    targets: tuple[str, ...]
    output_root: Optional[Path] = None
    run_name: str = "csd_act"
    tractography_backend: str = "csd_act"
    ukf_parameters: dict[str, Any] = field(default_factory=dict)
    ukf_executable: Optional[Path] = None
    dwiconvert_executable: Optional[Path] = None
    bundle_backend: str = "wmql"
    bundle_tractography_backend: Optional[str] = None
    wmql_query_preset: str = "official"
    wmql_query_selection: tuple[str, ...] = ()
    wmql_atlas_mode: str = 'native'
    execute: bool = False
    resume: bool = True
    preprocessing_backend: str = "dipy"
    brain_mask_backend: str = "pnl_cnn"
    brain_mask_device: str = "auto"
    pnl_mask_python: Optional[Path] = None
    pnl_mask_script: Optional[Path] = None
    pnl_mask_model_dir: Optional[Path] = None
    pnl_mask_ants_bin_dir: Optional[Path] = None
    synthstrip_python: Optional[Path] = None
    preprocessing_force: bool = False
    preprocessing_dipy_denoise_method: str = "mppca"
    preprocessing_dipy_run_gibbs: bool = True
    preprocessing_dipy_run_motion_correction: bool = True
    preprocessing_mrtrix_bin_dir: Optional[Path] = None
    preprocessing_fsl_bin_dir: Optional[Path] = None
    preprocessing_reverse_phase_b0: Optional[Path] = None
    preprocessing_phase_encoding_direction: Optional[str] = None
    preprocessing_readout_time: Optional[float] = None
    preprocessing_eddy_threads: Optional[int] = None
    preprocessing_eddy_second_level_model: Optional[str] = None
    microstructure_fit_method: str = "wls"
    microstructure_run_free_water: bool = False
    microstructure_run_noddi: bool = False
    microstructure_run_dki: bool = False
    max_parallel_models: int = 1
    microstructure_dki_fit_method: str = "WLS"
    microstructure_dki_matlab_executable: Optional[Path] = None
    microstructure_dki_mkcurve_root: Optional[Path] = None
    microstructure_dki_execution: dict = field(default_factory=dict)
    microstructure_free_water_chunk_size: int = 2
    microstructure_free_water_engine: str = "serial"
    microstructure_free_water_n_jobs: Optional[int] = None
    microstructure_free_water_vox_per_chunk: Optional[int] = None
    microstructure_free_water_max_fraction_for_tissue: float = 0.7
    postprocess_native_dir: Optional[Path] = None
    postprocess_mri_dir: Optional[Path] = None
    postprocess_backend_root: Optional[Path] = None
    postprocess_mode: str = "whole"
    postprocess_python_bin: Optional[Path] = None
    postprocess_freesurfer_home: Optional[Path] = None
    postprocess_runtime: Optional[Path] = None
    postprocess_license_file: Optional[Path] = None
    postprocess_atlases: Optional[tuple[str, ...]] = None
    postprocess_force: bool = False
    atlas_t2: Optional[Path] = None
    surface_atlas_t2: Optional[Path] = None
    surface_python_bin: Optional[Path] = None
    surface_precision: str = 'fp32'
    surface_cpu_threads: int = 4
    surface_max_gpu_gib: float = 24.
    model_assets_dir: Optional[Path] = None
    ddseg_device: str = "auto"
    surface_device: str = "auto"
    select_streamlines: int = 1_000_000
    dti_min_length_mm: float = 10.0
    dti_max_length_mm: float = 250.0
    mrtrix_bin_dir: Optional[Path] = None
    # Off (None) by default: MRtrix3 keeps using its own default thread
    # count for tckgen/tcksift2. Pinning this does NOT by itself make
    # tckgen bit-reproducible -- see tractography/csd.py's CSDActConfig
    # .nthreads docstring and the "reproducibility_note" the CSD/ACT
    # manifest records for every run.
    mrtrix_nthreads: Optional[int] = None
    # MRtrix3 RNG seed (MRTRIX_RNG_SEED). When set, tckgen is pinned to
    # -nthreads 0 so the tractogram is bit-reproducible; None = unseeded.
    mrtrix_random_seed: Optional[int] = None
    # CSD/ACT seeding: "dynamic" (default, -seed_dynamic on the WM FOD) or
    # "gmwmi" (-seed_gmwmi on the 5tt2gmwmi interface).
    tractography_seeding: str = "dynamic"
    # World-space QC gate: minimum Dice between DDSeg WM and DDParcel WM.
    ddseg_min_wm_dice: float = 0.70
    # ACT anatomy must come from the CSD DWI unless this is set.
    allow_anatomy_space_mismatch: bool = False
    # Explicit exploratory fallback; the GUI retains complete DDSeg + DDParcel priors.
    allow_act_ddseg_only: bool = False
    tract_querier_root: Optional[Path] = None
    connectome_backend: str = "python"
    connectome_compress_mrtrix_dense: bool = False
    saf_fs_dir: Optional[Path] = None
    saf_pregenerated_tractogram: Optional[Path] = None
    saf_tckgen_options: str = ""
    saf_mrtrix_prefix: str = ""
    wma_input_tractography: Optional[Path] = None
    wma_atlas_dir: Optional[Path] = None
    wma_atlas: str = "org"
    wma_root: Optional[Path] = None
    wma_num_threads: Optional[int] = None
    slicer_executable: Optional[Path] = None
    wma_use_virtual_x: bool = False
    wma_export_diffusion_measures: bool = False
    fiber_tract_measurements_cli: Optional[Path] = None
    tractcloud_input_tractography: Optional[Path] = None
    tractcloud_root: Optional[Path] = None
    tractcloud_python_executable: Optional[Path] = None
    tractcloud_data_dir: Optional[Path] = None
    tractcloud_device: str = "auto"
    tractcloud_batch_size: int = 2048
    tractcloud_random_seed: int = 0
    preflight_blocked_targets: dict[str, str] = field(default_factory=dict)
    preflight_blocked_by_backend: dict[str, dict[str, str]] = field(default_factory=dict)
    # C3 provenance: target added by the prerequisite closure -> targets that required it.
    # Callers that already applied the closure (CLI, GUI) pass it here.
    target_required_by: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # Targets the user explicitly skipped (CLI --skip-targets); the closure does not re-add them.
    excluded_targets: tuple[str, ...] = ()
    # Set once the closure was applied, so tracking variants do not expand again.
    closure_applied: bool = False
    # Algorithm added by the closure -> targets that required it; its variant runs only those targets.
    closure_algorithm_requesters: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ProductTaskRecord:
    name: str
    status: str
    reason: str = ""
    manifest_path: Optional[Path] = None
    qc_path: Optional[Path] = None
    outputs: dict[str, str | None] = field(default_factory=dict)
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    runtime_seconds: Optional[float] = None
    logs: tuple[str, ...] = ()
    # C8: "skipped_no_parcellation" when microstructure ran without a parcellation (no ROI statistics).
    regional_statistics: Optional[str] = None


@dataclass(frozen=True)
class RunPlanItem:
    target: str
    task_name: str
    status: str
    reason: str = ""
    reason_category: str = ""
    runnable: bool = False
    will_run: bool = False
    reused: bool = False
    inputs: dict[str, str | None] = field(default_factory=dict)
    outputs: dict[str, str | None] = field(default_factory=dict)
    dependencies: dict[str, str | None] = field(default_factory=dict)
    requested: bool = True
    required_by: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "target": self.target,
            "requested": self.requested,
            "required_by": list(self.required_by),
            "task_name": self.task_name,
            "status": self.status,
            "reason": self.reason,
            "reason_category": self.reason_category,
            "runnable": self.runnable,
            "will_run": self.will_run,
            "reused": self.reused,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "dependencies": self.dependencies,
        }


@dataclass(frozen=True)
class RunPlanResult:
    subject_id: str
    subject_root: Path
    output_root: Path
    run_name: str
    targets: tuple[str, ...]
    run_plan_path: Path
    items: tuple[RunPlanItem, ...]

    @property
    def blocked_items(self) -> tuple[RunPlanItem, ...]:
        return tuple(item for item in self.items if item.status == "blocked")

    @property
    def will_run_items(self) -> tuple[RunPlanItem, ...]:
        return tuple(item for item in self.items if item.will_run)

    @property
    def skipped_items(self) -> tuple[RunPlanItem, ...]:
        return tuple(item for item in self.items if item.status == "skipped")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "slicerddsurfer_run_plan",
            "version": 1,
            "subject_id": self.subject_id,
            "subject_root": self.subject_root.as_posix(),
            "output_root": self.output_root.as_posix(),
            "run_name": self.run_name,
            "targets": list(self.targets),
            "summary": {
                "will_run": len(self.will_run_items),
                "skipped": len(self.skipped_items),
                "blocked": len(self.blocked_items),
            },
            "items": [item.to_dict() for item in self.items],
        }


@dataclass(frozen=True)
class ProductOrchestratorResult:
    subject_id: str
    subject_root: Path
    run_manifest_path: Path
    tasks: tuple[ProductTaskRecord, ...]
    run_plan_path: Optional[Path] = None


def _emit(message: str, log: Optional[Callable[[str], None]] = None) -> None:
    if log:
        log(message)


def _start_task_timer(label: str, log: Optional[Callable[[str], None]] = None) -> float:
    started = time.monotonic()
    _emit(f"[Orchestrator] {label} started", log)
    return started


def _finish_task_timer(label: str, started: float, log: Optional[Callable[[str], None]] = None) -> None:
    _emit(f"[Orchestrator] {label} completed in {time.monotonic() - started:0.1f}s", log)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _qc_completed(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        record = _read_json(path)
        checks = record.get("checks") or {}
        return (record.get("status") == "completed" and record.get("automatic_qc_passed") is not False
                and (not isinstance(checks, dict) or not any(value is False for value in checks.values())))
    except Exception:
        return False


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _wmql_complete(
    qc_path: Path,
    *,
    manifest_path: Optional[Path] = None,
    input_tractography: Optional[Path] = None,
    parcellation: Optional[Path] = None,
    query_preset: str = "official",
    query_selection: tuple[str, ...] = (),
    atlas_mode: str = 'native',
) -> bool:
    """Reuse frozen, readable extraction, retaining official coverage warnings as warnings."""

    try:
        from tractography.wmql import (summarize_wmql_outputs, _wmql_output_contract_error, WMQL_QC_VERSION,
                                       bundled_wmql_query_path, wmql_query_cache_current,
                                       canonical_wmql_selection, official_query_names, AUTHOR_ALIAS_SHA256)
        from tractography.wmql_protocol import CATALOG_PRESETS, preset_query_names
        import hashlib

        native_manifest_path = manifest_path or qc_path.with_name("wmql_manifest.json")
        manifest = _read_json(native_manifest_path)
        qc = _read_json(qc_path)
        if manifest.get('atlas_mode', 'native') != atlas_mode:
            return False
        if atlas_mode != 'native':
            from tractography.wmql_atlas import prepared_atlas_current
            if not prepared_atlas_current(manifest.get('atlas_preparation') or {}, parcellation):
                return False
        official = query_preset == 'official'
        catalogue = query_preset in CATALOG_PRESETS
        if (qc.get('version') != WMQL_QC_VERSION or qc.get('status') != 'completed' or qc.get('error')
                or (catalogue and (not (qc.get('official_multibundle') if official else qc.get('coverage_report_mode'))
                                  or not qc.get('execution_passed')
                                  or not qc.get('output_integrity_passed')))
                or (not catalogue and qc.get('automatic_qc_passed') is not True)):
            return False
        selected = canonical_wmql_selection(query_selection, query_preset) if catalogue else tuple(query_selection)
        if catalogue and (manifest.get('query_preset', 'official') != query_preset
                          or tuple(manifest.get('query_selection', ())) != selected):
            return False
        query = Path(manifest.get('query_path') or '')
        if not query.is_file() or manifest.get('query_sha256') != hashlib.sha256(query.read_bytes()).hexdigest():
            return False
        if manifest.get('query_sha256') != hashlib.sha256(bundled_wmql_query_path(query_preset).read_bytes()).hexdigest():
            return False
        if not wmql_query_cache_current(manifest):
            return False
        if catalogue and not any(Path(path).name == 'FreeSurfer.qry' and digest == AUTHOR_ALIAS_SHA256
                                for path, digest in manifest.get('query_dependencies', {}).items()):
            return False
        expected_dependencies = {
            key: path
            for key, path in {
                "input_tractography": input_tractography,
                "atlas": parcellation,
            }.items()
            if path is not None
        }
        if expected_dependencies and not _manifest_inputs_match(
            native_manifest_path,
            expected_dependencies,
            "dependencies",
        ):
            return False
        summary = summarize_wmql_outputs(qc_path.parent, manifest.get('subject_id'))
        requested = (selected or preset_query_names(query_preset)) if catalogue else tuple(manifest.get('query_selection') or ())
        expected_count = 1 if catalogue else max(1, len(requested))
        bundle_count = int(summary.get("bundle_count", 0) or 0)
        readable_count = int(summary.get("bundle_readable_count", 0) or 0)
        nonempty_count = int(summary.get("bundle_nonempty_count", 0) or 0)
        if bundle_count < expected_count:
            return False
        return readable_count == bundle_count and nonempty_count >= expected_count and _wmql_output_contract_error(
            summary, expected_count, requested, require_sampling=not catalogue) is None
    except (OSError, TypeError, ValueError):
        return False


def _connectome_complete(
    qc_path: Path,
    matrix_path: Path,
    nodes_path: Path,
    *,
    manifest_path: Optional[Path] = None,
    tractography: Optional[Path] = None,
    parcellation: Optional[Path] = None,
    sift2_weights: Optional[Path] = None,
) -> bool:
    """Return true only when QC counts and both connectome tables have data."""

    if not _qc_completed(qc_path):
        return False
    try:
        qc = _read_json(qc_path)
        expected_dependencies = {
            key: path
            for key, path in {
                "tractography": tractography,
                "ddparcel_parcellation": parcellation,
                "sift2_weights": sift2_weights,
            }.items()
            if path is not None
        }
        if expected_dependencies and not _manifest_inputs_match(
            manifest_path or qc_path.with_name(qc_path.name.replace("_qc.json", "_manifest.json")),
            expected_dependencies,
            "dependencies",
        ):
            return False
        if int(qc.get("node_count", 0) or 0) <= 0 or int(qc.get("edge_count", 0) or 0) <= 0:
            return False
        if not _table_has_data_row(nodes_path, header_prefix="node_id\t"):
            return False
        return _table_has_data_row(matrix_path, header_prefix="source_node,target_node,")
    except (OSError, TypeError, ValueError):
        return False


def _table_has_data_row(path: Path, *, header_prefix: str) -> bool:
    """Accept dense headerless matrices or require a row after a known header."""

    if not _nonempty_file(path):
        return False
    try:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            first = next((line.strip() for line in stream if line.strip()), "")
            if not first:
                return False
            if not first.lower().startswith(header_prefix.lower()):
                return True
            return any(line.strip() for line in stream)
    except OSError:
        return False


def _find_first_existing(candidates: list[Path]) -> Optional[Path]:
    return next((path for path in candidates if path.is_file()), None)


def _resolve_optional_path(path: Optional[Path]) -> Optional[Path]:
    return path.expanduser().resolve() if path is not None else None


def _optional_interpreter(path: Optional[Path]) -> Optional[Path]:
    return interpreter_path(path) if path is not None else None


def _resolve_slicer_executable(config: ProductOrchestratorConfig) -> Optional[Path]:
    explicit = _resolve_optional_path(config.slicer_executable)
    if explicit is not None:
        return explicit if explicit.is_file() else None
    env_value = Path(os.environ["SLICERDSPACE_SLICER_EXECUTABLE"]).expanduser().resolve() if os.environ.get("SLICERDSPACE_SLICER_EXECUTABLE") else None
    if env_value is not None and env_value.is_file():
        return env_value
    candidates = [
        Path("/data/chengjin/software/Slicer-5.10.0-linux-amd64/Slicer"),
        Path("/data/chengjin/software/Slicer/Slicer"),
    ]
    return _find_first_existing(candidates)


def _resolved_device_for_execution(requested: str, *, execute: bool, target_will_run: bool) -> str:
    normalized = (requested or "auto").strip().lower() or "auto"
    if normalized not in {"", "auto"}:
        return resolve_auto_torch_device(normalized)
    if not target_will_run:
        return "not_resolved_target_not_run"
    if not execute:
        return "not_resolved_execute_false"
    return resolve_auto_torch_device(normalized)


def _resolved_tractcloud_device_for_execution(requested: str, *, execute: bool, target_will_run: bool) -> str:
    normalized = (requested or "auto").strip().lower() or "auto"
    if normalized not in {"", "auto"}:
        return normalized
    if not target_will_run:
        return "not_resolved_target_not_run"
    if not execute:
        return "not_resolved_execute_false"
    # Resolve inside TractCloud's selected Python, not the caller's runtime.
    return "auto"


def _normalize_bundle_backends(bundle_backend: str) -> tuple[str, ...]:
    value = (bundle_backend or "wmql").strip().lower()
    if value in {"", "default"}:
        value = "wmql"
    value = value.replace("+", ",").replace(";", ",")
    tokens: list[str] = []
    for token in (part.strip() for part in value.split(",")):
        if not token:
            continue
        if token == "all":
            tokens.extend(("wmql", "wma", "tractcloud"))
        elif token in {"none", "off", "skip"}:
            continue
        else:
            tokens.append(token)
    unknown = sorted(set(tokens) - BUNDLE_BACKENDS)
    if unknown:
        raise ValueError("bundle_backend must contain only: all, wmql, wma, tractcloud.")
    return tuple(dict.fromkeys(tokens))


def _primary_tractography_backend(config: ProductOrchestratorConfig) -> str:
    return "csd_act" if config.tractography_backend == "saf" else config.tractography_backend


def _ukf_raw(config, inputs):
    from shared.pipeline import RawInputBundle
    return RawInputBundle(subject_root=config.subject_root, subject_id=config.subject_id,
        input_dir=inputs['dwi'].parent, dwi_path=inputs['dwi'], bval_path=inputs['bval'],
        bvec_path=inputs['bvec'], mask_path=inputs['mask'])


def _ukf_current(config, inputs, directory):
    from tractography.ukf import current
    return current(_ukf_raw(config, inputs), directory, config.ukf_parameters,
                   config.ukf_executable, config.dwiconvert_executable)


def orchestrator_closure(
    targets: tuple[str, ...],
    *,
    tractography_backend: str = "csd_act",
    bundle_backend: str = "wmql",
    microstructure_run_free_water: bool = False,
    microstructure_run_noddi: bool = False,
    microstructure_run_dki: bool = False,
    wma_source_explicit: bool = False,
    provided_tractography: tuple[str, ...] = (),
    excluded: tuple[str, ...] = (),
):
    """Return the prerequisite closure (see ``shared.target_closure``) of the selected targets."""

    from shared.algorithm_selection import algorithms
    return expand_closure(
        targets,
        tractography_algorithms=algorithms(tractography_backend),
        bundle_backends=_normalize_bundle_backends(bundle_backend),
        microstructure_free_water=microstructure_run_free_water,
        microstructure_noddi=microstructure_run_noddi,
        microstructure_dki=microstructure_run_dki,
        wma_source_explicit=wma_source_explicit,
        provided_tractography=provided_tractography,
        excluded=excluded,
    )


def normalize_orchestrator_targets(
    targets: tuple[str, ...],
    *,
    tractography_backend: str = "csd_act",
    bundle_backend: str = "wmql",
    microstructure_run_free_water: bool = False,
    microstructure_run_noddi: bool = False,
    microstructure_run_dki: bool = False,
    wma_source_explicit: bool = False,
    provided_tractography: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Expand product-level targets and add every prerequisite target (C1 closure)."""

    return orchestrator_closure(
        targets,
        tractography_backend=tractography_backend,
        bundle_backend=bundle_backend,
        microstructure_run_free_water=microstructure_run_free_water,
        microstructure_run_noddi=microstructure_run_noddi,
        microstructure_run_dki=microstructure_run_dki,
        wma_source_explicit=wma_source_explicit,
        provided_tractography=provided_tractography,
    ).targets


def _resolve_bundle_input_tractography(
    subject_root: Path,
    explicit_path: Optional[Path],
    *,
    output_root: Optional[Path] = None,
    run_name: Optional[str] = None,
    current_run_tractography: Optional[Path] = None,
    require_current_run: bool = False,
    allow_planned_current_run: bool = False,
) -> Optional[Path]:
    resolved_explicit = _resolve_optional_path(explicit_path)
    if resolved_explicit is not None:
        return resolved_explicit if resolved_explicit.is_file() else None
    layout = subject_layout(subject_root, output_root=output_root)
    candidates: list[Path] = []
    if run_name:
        current_run_candidates = [
            current_run_tractography.expanduser().resolve()
            if current_run_tractography is not None
            else layout.tractograms / normalize_run_name(run_name) / "missing_csd_act_ifod2.vtk"
        ]
        current_run_input = next((path.resolve() for path in current_run_candidates if path.is_file()), None)
        if current_run_input is not None:
            return current_run_input
        if require_current_run:
            return current_run_candidates[0].resolve() if allow_planned_current_run else None
        candidates.extend(current_run_candidates)
    candidates.extend(sorted(layout.tractography.glob("**/*.vt*")))
    candidates.extend(sorted(subject_root.glob("tractography/**/*.vt*")) + sorted(subject_root.glob("tractography/*.vt*")))
    excluded_parts = {"saf", "bundles", "wmql", "wma", "tractcloud", "tract_cloud_dl"}
    return next(
        (
            path.resolve()
            for path in candidates
            if path.suffix.lower() in {".vtk", ".vtp"}
            and path.is_file()
            and not excluded_parts.intersection(part.lower() for part in path.parts)
        ),
        None,
    )


def _resolve_wma_input_tractography(
    subject_root: Path,
    explicit_path: Optional[Path],
    *,
    output_root: Optional[Path] = None,
) -> Optional[Path]:
    """Resolve an explicitly marked WMA-compatible whole-brain tractogram.

    ORG atlas clustering expects broad long-range white-matter coverage.  The
    CSD-ACT tractogram produced for connectome construction is therefore not
    reused implicitly.  Automatic discovery is deliberately limited to files
    whose names identify UKF or a dedicated WMA/ORG input; every other
    tractogram must be selected explicitly by the user.
    """

    resolved_explicit = _resolve_optional_path(explicit_path)
    if resolved_explicit is not None:
        return resolved_explicit if resolved_explicit.is_file() else None

    layout = subject_layout(subject_root, output_root=output_root)
    excluded_parts = {"saf", "bundles", "wma", "wmql", "tractcloud", "tract_cloud_dl"}
    candidates = sorted(layout.tractography.glob("**/*.vtk"))
    candidates.extend(sorted(layout.tractography.glob("**/*.vtp")))
    candidates.extend(sorted(layout.tractography.glob("**/*.tck")))
    candidates.extend(sorted((subject_root / "tractography").glob("**/*.vtk")))
    candidates.extend(sorted((subject_root / "tractography").glob("**/*.vtp")))
    candidates.extend(sorted((subject_root / "tractography").glob("**/*.tck")))
    compatible_name_markers = ("ukf", "wma_input", "wma-input", "org_input", "org-input")
    for path in candidates:
        if not path.is_file():
            continue
        if excluded_parts.intersection(part.lower() for part in path.parts):
            continue
        if not any(marker in path.stem.lower() for marker in compatible_name_markers):
            continue
        return path.resolve()
    return None


def _resolve_mrtrix_command(command: str, bin_dir: Optional[Path]) -> Optional[Path]:
    candidate_dirs: list[Path] = []
    if bin_dir is not None:
        candidate_dirs.append(bin_dir.expanduser())
    for env_name in ("SLICERDSPACE_MRTRIX_BIN_DIR", "MRTRIX_BIN_DIR"):
        value = os.environ.get(env_name)
        if value:
            candidate_dirs.append(Path(value).expanduser())
    executable_names = (f"{command}.exe", command) if os.name == "nt" else (command,)
    for candidate_dir in candidate_dirs:
        for executable_name in executable_names:
            candidate = candidate_dir / executable_name
            if candidate.is_file():
                return candidate.resolve()
    resolved = shutil.which(command)
    return Path(resolved).resolve() if resolved else None


def _resolve_best_saf_pregenerated_tractogram(subject_root: Path, explicit_path: Optional[Path], output_root: Optional[Path] = None) -> Optional[Path]:
    resolved_explicit = _resolve_optional_path(explicit_path)
    if resolved_explicit is not None:
        return resolved_explicit if resolved_explicit.is_file() else None
    layout = subject_layout(subject_root, output_root=output_root)
    candidates = sorted(layout.tractograms.glob("csd_act*/**/*.tck"))
    if not candidates:
        return None
    candidates = [path.resolve() for path in candidates if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_size, path.as_posix()))


def _saf_output_dir(layout, run_name: str) -> Path:
    return layout.saf if run_name == "saf" else (layout.saf / run_name)


def _wma_output_dir(layout, run_name: str) -> Path:
    return layout.bundles_wma / run_name


def _tractcloud_output_dir(layout, run_name: str) -> Path:
    return layout.bundles_tractcloud / run_name


def _wma_anatomical_tract_count(output_dir: Path) -> int:
    return len(list(output_dir.glob("**/AnatomicalTracts/*.vtp"))) if output_dir.is_dir() else 0


def _wma_output_summary(output_dir: Path) -> dict[str, Any]:
    qc_path = output_dir / "wma_org_atlas_qc.json"
    if not qc_path.is_file():
        return {}
    try:
        qc = _read_json(qc_path)
    except Exception:
        return {}
    summary = qc.get("output_summary")
    return dict(summary) if isinstance(summary, dict) else {}


def _wma_complete(
    output_dir: Path,
    *,
    input_tractography: Optional[Path] = None,
    atlas_dir: Optional[Path] = None,
    atlas_family: str = "org",
) -> bool:
    if input_tractography is None:
        return False
    qc_path = output_dir / "wma_org_atlas_qc.json"
    manifest_path = output_dir / "wma_org_atlas_manifest.json"
    if not _qc_completed(qc_path):
        return False
    try:
        from tractography.wma import summarize_wma_outputs, _wma_output_contract_error
        from shared.bundle_qc import bundle_record_error

        if bundle_record_error('bundles.wma', _read_json(qc_path)):
            return False

        manifest = _read_json(manifest_path)
        if manifest.get("atlas_family", "org") != atlas_family:
            return False
        expected_dependencies = {
            key: path
            for key, path in {
                "input_tractography": input_tractography,
                "atlas_dir": atlas_dir,
            }.items()
            if path is not None
        }
        if expected_dependencies and not _manifest_inputs_match(
            manifest_path,
            expected_dependencies,
            "dependencies",
        ):
            return False
        prepared_input = Path(str(manifest.get("prepared_input_tractography") or ""))
        if not prepared_input.is_file():
            return False
        summary = summarize_wma_outputs(output_dir, prepared_input)
        if atlas_family != "org":
            from tractography.wma import inspect_wma_org_atlas
            if atlas_dir is None:
                return False
            atlas_status = inspect_wma_org_atlas(atlas_dir, atlas_family)
            if not atlas_status["available"]:
                return False
            summary["atlas_contract"] = atlas_status["output_contract"]
    except Exception:
        return False
    return (_wma_output_contract_error(summary) is None
            and summary.get('tract_sampling', {}).get('all_tract_sampling_passed') is True)


def _tractcloud_complete(
    output_dir: Path,
    *,
    input_tractography: Optional[Path] = None,
    random_seed: int = 0,
    config: Optional[ProductOrchestratorConfig] = None,
) -> bool:
    qc_path = output_dir / "tractcloud_qc.json"
    manifest_path = output_dir / "tractcloud_manifest.json"
    if not _qc_completed(qc_path):
        return False
    if input_tractography is None:
        return False
    try:
        from tractography.tractcloud import summarize_tractcloud_outputs, ADAPTER_REVISION, model_input_signature
        from shared.bundle_qc import bundle_record_error

        if bundle_record_error('bundles.tractcloud', _read_json(qc_path)):
            return False

        manifest = json.loads(manifest_path.read_text())
        if manifest.get('status') != 'completed' or manifest.get('executed') is not True:
            return False
        if manifest.get('random_seed') != random_seed:
            return False
        if manifest.get('adapter_revision') != ADAPTER_REVISION:
            return False
        if config is not None:
            if manifest.get('batch_size') != config.tractcloud_batch_size:
                return False
            if config.tractcloud_device not in {'auto', ''} and manifest.get('device') != config.tractcloud_device:
                return False
            if config.tractcloud_root is not None and manifest.get('tractcloud_root') != str(config.tractcloud_root.expanduser().resolve()):
                return False
            if manifest.get('dependencies', {}).get('model_inputs') != model_input_signature(config.tractcloud_data_dir):
                return False

        if input_tractography is not None and not _manifest_inputs_match(
            manifest_path,
            {"input_tractography": input_tractography},
            "dependencies",
        ):
            return False
        summary = summarize_tractcloud_outputs(output_dir)
        count = int(summary.get("bundle_count", 0) or 0)
        return (
            count > 0
            and int(summary.get("bundle_readable_count", 0) or 0) == count
            and int(summary.get("bundle_nonempty_count", 0) or 0) == count
            and int(summary.get("assigned_streamline_count", 0) or 0) > 0
        )
    except Exception:
        return False


def _saf_complete(
    output_dir: Path,
    *,
    fod_path: Optional[Path] = None,
    fs_dir: Optional[Path] = None,
) -> bool:
    qc_path = output_dir / "saf_qc.json"
    manifest_path = output_dir / "saf_manifest.json"
    tck_path = output_dir / "SAF.tck"
    vtk_path = output_dir / "SAF.vtk"
    if not _qc_completed(qc_path) or not tractography_readable(tck_path, tck_path):
        return False
    try:
        qc = _read_json(qc_path)
        manifest = _read_json(manifest_path)
        from tractography.saf_backend.provenance import IMPLEMENTATION_REVISION

        for record in (qc, manifest):
            implementation = record.get("implementation")
            if not isinstance(implementation, dict) or implementation.get("revision") != IMPLEMENTATION_REVISION:
                return False
        expected_dependencies = {
            key: path
            for key, path in {"fod": fod_path, "fs_dir": fs_dir}.items()
            if path is not None
        }
        if expected_dependencies and not _manifest_inputs_match(
            manifest_path,
            expected_dependencies,
            "inputs",
        ):
            return False
        scientific_summary = qc.get("scientific_output_summary")
        if not isinstance(scientific_summary, dict):
            return False
        length_summary = scientific_summary.get("length_mm")
        if not isinstance(length_summary, dict):
            return False
        configured_max = (manifest.get("parameters") or {}).get("maxlen")
        return (
            int(qc.get("streamline_count", 0) or 0) > 0
            and int(scientific_summary.get("streamline_count", 0) or 0)
            == int(qc.get("streamline_count", 0) or 0)
            and bool(scientific_summary.get("all_points_finite"))
            and (
                configured_max is None
                or float(length_summary.get("max", float("inf")))
                <= float(configured_max) + 1e-3
            )
            and not qc.get("streamline_count_error")
            and not qc.get("vtk_export_error")
            and _nonempty_file(vtk_path)
        )
    except (OSError, TypeError, ValueError):
        return False


def _normalize_targets(config: ProductOrchestratorConfig) -> tuple[str, ...]:
    if not config.closure_applied:
        return _with_closure(config).targets
    # Tracking variants carry a deliberate subset of the closed targets: only validate and order it.
    requested = {target.strip().lower() for target in config.targets if target.strip()}
    unsupported = sorted(requested - set(TARGET_ORDER))
    if unsupported:
        raise ValueError(f"Unsupported orchestrator target(s): {', '.join(unsupported)}")
    return tuple(target for target in TARGET_ORDER if target in requested)


def _provided_tractography(config: ProductOrchestratorConfig) -> tuple[str, ...]:
    """Bundle targets whose tractogram the user supplied as a file (no tracking is needed for them)."""

    return tuple(name for name, value in (
        ("wma", config.wma_input_tractography), ("tractcloud", config.tractcloud_input_tractography),
        ("saf", config.saf_pregenerated_tractogram)) if value is not None)


def _with_closure(config: ProductOrchestratorConfig) -> ProductOrchestratorConfig:
    """Apply the prerequisite closure once: targets, tractography algorithms and provenance."""

    if config.closure_applied:
        return config
    closure = orchestrator_closure(
        config.targets,
        tractography_backend=config.tractography_backend,
        bundle_backend=config.bundle_backend,
        microstructure_run_free_water=config.microstructure_run_free_water,
        microstructure_run_noddi=config.microstructure_run_noddi,
        microstructure_run_dki=config.microstructure_run_dki,
        wma_source_explicit=bool(config.bundle_tractography_backend),
        provided_tractography=_provided_tractography(config),
        excluded=config.excluded_targets,  # K5: skipped targets are never added or expanded
    )
    from shared.target_closure import provenance_map
    required_by = {target: tuple(value) for target, value in
                   provenance_map(closure, config.target_required_by).items() if target in closure.targets}
    targets = closure.targets
    backend = config.tractography_backend
    if "tractography" in targets:
        backend = closure.algorithms_text
    return replace(config, targets=targets, tractography_backend=backend,
                   target_required_by=required_by, closure_applied=True,
                   closure_algorithm_requesters={**config.closure_algorithm_requesters,
                                                 **{key: tuple(value) for key, value in closure.added_algorithms.items()}})


def closure_notice_for(config: ProductOrchestratorConfig) -> str:
    """C4 one-line notice for a config whose closure was applied (empty when nothing was added)."""

    from shared.algorithm_selection import algorithms
    from shared.target_closure import Closure, closure_notice
    required_by = {target: tuple(value) for target, value in config.target_required_by.items()}
    ordered = {target: required_by[target] for target in config.targets if target in required_by}
    return closure_notice(Closure(
        targets=tuple(config.targets), tractography_algorithms=algorithms(config.tractography_backend),
        requested=tuple(t for t in config.targets if t not in ordered), required_by=ordered,
        added_algorithms={key: tuple(value) for key, value in config.closure_algorithm_requesters.items()}))


def _upstream_reason(selected, upstream: str, missing_text: str) -> str:
    """Q2: a stage that is already in this run must not be told to "select" it."""

    if upstream in selected:
        from shared.target_closure import DISPLAY_NAMES
        return f"{DISPLAY_NAMES.get(upstream, upstream)} is in this run but produced no output (see its task)"
    return missing_text


REGIONAL_STATISTICS_SKIPPED = "skipped_no_parcellation"


def regional_statistics_note(parcellation, log=None) -> Optional[str]:
    """C8: parcellation is not added for microstructure; say so when ROI statistics are skipped."""

    if parcellation is not None:
        return None
    _emit("[Microstructure] Regional statistics skipped: parcellation was not selected", log)
    return REGIONAL_STATISTICS_SKIPPED


def mask_failure_message(reason: str) -> str:
    """C7: one wording for the log line and the blocked reasons when the automatic mask fails."""

    return f"Automatic brain mask failed: {reason}. Provide a brain mask file (NIfTI) or run Prepare to install the mask runtime."


def _task_name_for_target(target: str, config: ProductOrchestratorConfig) -> str:
    if target == "preprocessing":
        return "preprocessing.dwi"
    if target == "segmentation":
        return "segmentation.ddseg"
    if target == "parcellation":
        return "parcellation.ddparcel"
    if target == "surface":
        return "surface.ddsurfer"
    if target == "surface_postprocess":
        return "surface.ddsurfer_postprocess"
    if target == "microstructure":
        return "microstructure.dti"
    if target == "microstructure_freewater":
        return "microstructure.freewater"
    if target == "microstructure_noddi":
        return "microstructure.noddi"
    if target == "microstructure_dki":
        return "microstructure.dki"
    if target == "tractography":
        return "tractography." + _primary_tractography_backend(config)
    if target == "bundles":
        return "bundles.wmql"
    if target == "connectome":
        return "connectome.streamline_count"
    if target == "saf":
        return "tractography.saf"
    if target == "wma":
        return "bundles.wma"
    if target == "tractcloud":
        return "bundles.tractcloud"
    return target


def _resolve_raw_inputs(subject_root: Path) -> dict[str, Path]:
    input_dir = subject_root / "input"
    return {
        "dwi": input_dir / "data.nii.gz",
        "bval": input_dir / "bvals",
        "bvec": input_dir / "bvecs",
        "mask": input_dir / "nodif_brain_mask.nii.gz",
    }


def _resolve_analysis_inputs(
    subject_root: Path,
    subject_id: str,
    output_root: Optional[Path] = None,
) -> dict[str, Path]:
    """Return the completed preprocessing bundle, or canonical raw inputs."""

    if _preprocessing_complete(subject_root, subject_id, output_root):
        outputs = _preprocessing_outputs(subject_root, subject_id, output_root)
        return {key: outputs[key] for key in ("dwi", "bval", "bvec", "mask")}
    return _resolve_raw_inputs(subject_root)


def _file_record_matches(record: object, path: Path) -> bool:
    if not isinstance(record, dict) or not path.exists():
        return False
    try:
        stat = path.stat()
        return (
            Path(str(record.get("path") or "")).resolve() == path.resolve()
            and int(record.get("size", -1)) == stat.st_size
            and int(record.get("mtime_ns", -1)) == stat.st_mtime_ns
        )
    except (OSError, TypeError, ValueError):
        return False


def _manifest_inputs_match(manifest_path: Path, inputs: dict[str, Path], field: str) -> bool:
    try:
        manifest = _read_json(manifest_path)
    except Exception:
        return False
    records = manifest.get(field)
    if not isinstance(records, dict):
        return False
    return all(_file_record_matches(records.get(key), path) for key, path in inputs.items())


def _dti_bundle_source_current(config, manifest_path, inputs):
    if not _manifest_inputs_match(manifest_path, inputs, 'raw_inputs'):
        return False
    try:
        manifest = _read_json(manifest_path)
        params = manifest['parameters']
        return (manifest.get('status') == 'completed'
                and params.get('fit_method', '').lower() == config.microstructure_fit_method.lower()
                and params.get('tracking_method') == 'deterministic'
                and params.get('output_format') == 'trk'
                and params.get('max_seed_voxels') == config.select_streamlines
                and params.get('min_length_mm') == config.dti_min_length_mm
                and params.get('max_length_mm') == config.dti_max_length_mm)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False


def _dti_bundle_qc_acceptable(config, qc_path, tractogram):
    if not _qc_completed(qc_path) or not _nonempty_file(tractogram):
        return False
    try:
        qc = _read_json(qc_path)
        return (int(qc.get('streamline_count') or 0) > 0
                and bool(qc.get('streamlines_exists'))
                and float(qc['length_min_mm']) + 1e-3 >= config.dti_min_length_mm
                and float(qc['length_max_mm']) - 1e-3 <= config.dti_max_length_mm)
    except (OSError, ValueError, TypeError, KeyError):
        return False


def _csd_bundle_parameters_current(config, manifest_path):
    try:
        params = _read_json(manifest_path)['parameters']
        return (params.get('select_streamlines') == config.select_streamlines
                and params.get('min_length_mm') == config.dti_min_length_mm
                and params.get('max_length_mm') == config.dti_max_length_mm
                and params.get('random_seed') == config.mrtrix_random_seed
                and params.get('seeding') == config.tractography_seeding
                and params.get('nthreads') == config.mrtrix_nthreads)
    except (OSError, ValueError, TypeError, KeyError):
        return False


def _preprocessing_outputs(subject_root: Path, subject_id: str, output_root: Optional[Path] = None) -> dict[str, Path]:
    layout = subject_layout(subject_root, output_root=output_root)
    preproc_dir = layout.preproc
    return {
        "dwi": preproc_dir / f"{subject_id}_dwi_final.nii.gz",
        "bval": preproc_dir / f"{subject_id}_dwi_final.bval",
        "bvec": preproc_dir / f"{subject_id}_dwi_final.bvec",
        "mask": preproc_dir / f"{subject_id}_brain_mask_final.nii.gz",
        "manifest": preproc_dir / f"{subject_id}_preproc_manifest.json",
        "qc": preproc_dir / "05_qc" / f"{subject_id}_preproc_qc.json",
    }


def _preprocessing_config(config: ProductOrchestratorConfig, subject_root: Path, output_dir: Path):
    from preprocessing import DwiPreprocessingConfig

    return DwiPreprocessingConfig(
        subject_id=config.subject_id, input_root=subject_root, output_dir=output_dir,
        backend=config.preprocessing_backend, execute=True, skip=False,
        brain_mask_backend=config.brain_mask_backend,
        brain_mask_device=config.brain_mask_device,
        pnl_mask_python=config.pnl_mask_python,
        pnl_mask_script=config.pnl_mask_script,
        pnl_mask_model_dir=config.pnl_mask_model_dir,
        pnl_mask_ants_bin_dir=config.pnl_mask_ants_bin_dir,
        synthstrip_python=config.synthstrip_python,
        postprocess_freesurfer_home=config.postprocess_freesurfer_home,
        force=config.preprocessing_force or (not config.resume),
        dipy_denoise_method=config.preprocessing_dipy_denoise_method,
        dipy_run_gibbs=config.preprocessing_dipy_run_gibbs,
        dipy_run_motion_correction=config.preprocessing_dipy_run_motion_correction,
        mrtrix_bin_dir=config.preprocessing_mrtrix_bin_dir,
        fsl_bin_dir=config.preprocessing_fsl_bin_dir,
        reverse_phase_b0=config.preprocessing_reverse_phase_b0,
        phase_encoding_direction=config.preprocessing_phase_encoding_direction,
        readout_time=config.preprocessing_readout_time,
        eddy_threads=config.preprocessing_eddy_threads,
        eddy_second_level_model=config.preprocessing_eddy_second_level_model,
    )


def _preprocessing_complete(
    subject_root: Path, subject_id: str, output_root: Optional[Path] = None,
    *, config: Optional[ProductOrchestratorConfig] = None,
) -> bool:
    outputs = _preprocessing_outputs(subject_root, subject_id, output_root)
    if not all(path.is_file() for key, path in outputs.items() if key != "qc") or not _qc_completed(outputs["qc"]):
        return False
    try:
        manifest = _read_json(outputs["manifest"])
        if manifest.get("status") not in {"completed", "cached"}:
            return False
        from preprocessing.pipeline import preprocessing_gradients_current
        if not preprocessing_gradients_current(manifest):
            return False
        cache_key = manifest.get("cache_key")
        raw = _resolve_raw_inputs(subject_root)
        if isinstance(cache_key, dict) and not all(
            _file_record_matches((cache_key.get("inputs") or {}).get(key), path)
            for key, path in raw.items()
        ):
            return False
        if config is not None:
            from preprocessing.pipeline import _preprocessing_cache_key, inspect_preprocessing_backend
            from shared.pipeline import RawInputBundle

            expected = _preprocessing_config(config, subject_root, outputs["dwi"].parent)
            if expected.force:
                return False
            bundle = RawInputBundle(
                subject_root=subject_root, subject_id=subject_id, input_dir=subject_root / "input",
                dwi_path=raw["dwi"], bval_path=raw["bval"], bvec_path=raw["bvec"], mask_path=raw["mask"],
            )
            if cache_key != _preprocessing_cache_key(bundle, expected, inspect_preprocessing_backend(expected)):
                return False
    except (OSError, ValueError, TypeError, AttributeError):
        return False
    return True


def _resolve_segmentation(subject_root: Path, output_root: Optional[Path] = None) -> Optional[Path]:
    layout = subject_layout(subject_root, output_root=output_root)
    directory = layout.anatomy_input_dir("ddseg")
    candidates = [directory / "SegmentationMap_GMWMCSF.nii.gz"]
    if not directory.exists() and not directory.is_symlink():
        candidates.extend([
            subject_root / ".legacy_flat_outputs" / "ddseg" / "SegmentationMap_GMWMCSF.nii.gz",
            subject_root / "segmentation" / "SegmentationMap_GMWMCSF.nii.gz",
        ])
    return next((path for path in candidates if path.is_file()), None)


def _anatomy_input_bundle(
    subject_root: Path,
    subject_id: str,
    output_root: Optional[Path],
    mrtrix_bin_dir: Optional[Path],
    brain_mask_config=None,
):
    """DWI bundle for DDSeg/DDParcel/DDSurfer: the preprocessed DWI when preprocessing completed.

    Tractography (CSD/ACT) consumes the preprocessed DWI, so the anatomy used
    for ACT must be derived from the same image; otherwise distortion/motion
    correction displaces the 5TT against the FODs.
    """

    from shared.pipeline import find_subject_root, resolve_preprocessed_input_bundle

    raw_bundle = find_subject_root(subject_root, subject_id, mrtrix_bin_dir=mrtrix_bin_dir,
                                  brain_mask_config=brain_mask_config)
    return resolve_preprocessed_input_bundle(raw_bundle, output_root=output_root)


def _write_anatomy_source(path: Path, bundle, prepared_source=None) -> None:
    from shared.pipeline import _cache_file_signature, _write_manifest
    sources = {key: _cache_file_signature(getattr(bundle, key + '_path'))
               for key in ('dwi', 'bval', 'bvec', 'mask')}
    if prepared_source is not None and sources != prepared_source:
        raise RuntimeError('Anatomical inputs changed after shared preparation; outputs are not reusable')
    record = {
        "kind": "anatomy_source",
        "version": 2,
        "subject_id": bundle.subject_id,
        "dwi": Path(bundle.dwi_path).resolve().as_posix(),
        "mask": Path(bundle.mask_path).resolve().as_posix(),
        "preprocessed": Path(bundle.input_dir).resolve() != (Path(bundle.subject_root) / "input").resolve(),
        "source_files": sources,
    }
    _write_manifest(path, record)


def _anatomy_source_current(path, subject_root, output_root):
    from shared.pipeline import _cache_file_signature
    try:
        record = _read_json(path)
        if record.get('version') != 2 or not record.get('subject_id'):
            return False
        inputs = _resolve_analysis_inputs(subject_root, record['subject_id'], output_root)
        return record.get('source_files') == {key: _cache_file_signature(value) for key, value in inputs.items()}
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def _anatomy_dwi_for_check(sources: dict[str, Optional[str]]) -> Optional[Path]:
    """Recorded anatomy DWI to compare against the CSD DWI (a mismatching one first)."""

    csd_dwi = sources.get("csd_dwi")
    recorded = [value for key, value in sources.items() if key != "csd_dwi" and value]
    for value in recorded:
        if csd_dwi and Path(value).resolve() != Path(csd_dwi).resolve():
            return Path(value)
    return Path(recorded[0]) if recorded else None


def _run_ddseg_world_qc_gate(
    config: "ProductOrchestratorConfig",
    segmentation: Optional[Path],
    parcellation: Optional[Path],
    qc_path: Path,
    tasks: list,
    log: Optional[Callable[[str], None]],
) -> Optional[str]:
    """World-space DDSeg-vs-DDParcel WM overlap gate; returns a failure reason or None."""

    if not config.execute:
        return None
    if segmentation is None or parcellation is None:
        _emit("[Orchestrator] WARNING: DDSeg world-space QC gate skipped (DDSeg or DDParcel output missing).", log)
        return None
    from anatomy.segmentation.world_qc import write_ddseg_world_qc

    try:
        record = write_ddseg_world_qc(segmentation, parcellation, qc_path, min_wm_dice=config.ddseg_min_wm_dice)
        wm_dice = float(record["metrics"]["wm_dice"])
        passed = record["status"] == "passed"
        reason = f"DDSeg WM vs DDParcel WM world-space Dice {wm_dice:.3f} (gate >= {config.ddseg_min_wm_dice:.2f})"
    except Exception as exc:
        passed = False
        reason = f"world-space QC could not be computed: {type(exc).__name__}: {exc}"
    if not passed:
        _emit(f"[Orchestrator] ERROR: DDSeg world-space QC gate FAILED: {reason}", log)
    else:
        _emit(f"[Orchestrator] {reason}", log)
    tasks.append(
        ProductTaskRecord(
            name="segmentation.ddseg_world_qc",
            status="completed" if passed else "failed",
            reason=reason,
            qc_path=qc_path,
            outputs={"qc": qc_path.as_posix()},
        )
    )
    return None if passed else reason


def _recorded_source_dwi(path: Path, key: str) -> Optional[str]:
    try:
        value = _read_json(path).get(key)
    except Exception:
        return None
    return str(value) if value else None


def _resolve_parcellation(subject_root: Path, subject_id: str, output_root: Optional[Path] = None) -> Optional[Path]:
    layout = subject_layout(subject_root, output_root=output_root)
    directory = layout.anatomy_input_dir("ddparcel")
    candidates = [
        directory / f"{subject_id}_ddparcel_wmparc_labels.nii.gz",
        directory / f"{subject_id}-DDSurfer-wmparc.nii.gz",
    ]
    if not directory.exists() and not directory.is_symlink():
        candidates.extend([
            subject_root / ".legacy_flat_outputs" / "ddparcel" / f"{subject_id}-DDSurfer-wmparc.nii.gz",
            subject_root / "parcellation" / "wmparc.nii.gz",
        ])
    return next((path for path in candidates if path.is_file()), None)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_atlas_t2(model_assets_dir: Optional[Path] = None) -> Path:
    return model_asset_path(
        "Resources/Data/100HCP-population-mean-T2.nii.gz",
        repo_root=_repo_root(),
        explicit_root=model_assets_dir,
    )


def _default_ddseg_weights(model_assets_dir: Optional[Path] = None) -> Path:
    return model_asset_path("segmentation/weights", repo_root=_repo_root(), explicit_root=model_assets_dir)


def _segmentation_outputs(subject_root: Path, output_root: Optional[Path] = None) -> dict[str, Path]:
    layout = subject_layout(subject_root, output_root=output_root)
    directory = layout.anatomy_input_dir("ddseg")
    return {
        "segmentation": directory / "SegmentationMap_GMWMCSF.nii.gz",
        "gm_probability": directory / "ProbabilisticMap_GM.nii.gz",
        "wm_probability": directory / "ProbabilisticMap_WM.nii.gz",
        "csf_probability": directory / "ProbabilisticMap_CSF.nii.gz",
        "qc": directory / "ddseg_qc.json",
    }


def _segmentation_complete(subject_root: Path, output_root: Optional[Path] = None) -> bool:
    outputs = _segmentation_outputs(subject_root, output_root)
    if not all(_nonempty_file(path) for path in outputs.values()):
        return False
    try:
        qc = _read_json(outputs["qc"])
        probability_qc = qc.get("probability_qc")
        if not isinstance(probability_qc, dict):
            return False
        mass_fraction = float(probability_qc.get("tissue_mass_valid_fraction", 0.0))
        label_agreement = float(probability_qc.get("label_agreement", 0.0))
        schema_version = int(qc.get('schema_version') or 0)
        from shared.pipeline import _artifact_record
        artifacts = {key: _artifact_record(path, verify_content=True) for key, path in outputs.items() if key != 'qc'}
        if qc.get('output_artifacts') != artifacts:
            return False
    except Exception:
        return False
    from anatomy.segmentation.orientation import ORIENTATION_CONVENTION
    from anatomy.segmentation.qc import DDSEG_QC_SCHEMA_VERSION
    source = outputs['qc'].parent / 'ddseg_source.json'
    if source.exists():
        if not _anatomy_source_current(source, subject_root, output_root):
            return False
    elif qc.get('source_dwi'):
        # Upgrade old executed-run records rather than blessing their path alone.
        return False

    return (
        qc.get("status") == "completed"
        and schema_version >= DDSEG_QC_SCHEMA_VERSION
        and qc.get("orientation_convention") == ORIENTATION_CONVENTION
        and qc.get("probability_semantics") == "four_class_softmax_tissue_channels_v1"
        and bool(probability_qc.get("finite_and_bounded"))
        and mass_fraction >= 0.999
        and label_agreement >= 0.999
    )


def _parcellation_outputs(subject_root: Path, subject_id: str, output_root: Optional[Path] = None) -> dict[str, Path]:
    layout = subject_layout(subject_root, output_root=output_root)
    directory = layout.anatomy_input_dir("ddparcel")
    return {
        "wmparc": directory / f"{subject_id}-DDSurfer-wmparc.nii.gz",
        "wmparc_reg": directory / f"{subject_id}-DDSurfer-wmparc-Reg.mgz",
        "fa_reg_norm": directory / f"{subject_id}-dti-FractionalAnisotropy-Reg-NormMasked.nii.gz",
        "trace_reg_norm": directory / f"{subject_id}-dti-Trace-Reg-NormMasked.nii.gz",
        "min_eig_reg_norm": directory / f"{subject_id}-dti-MinEigenvalue-Reg-NormMasked.nii.gz",
        "mid_eig_reg_norm": directory / f"{subject_id}-dti-MidEigenvalue-Reg-NormMasked.nii.gz",
    }


def _parcellation_complete(subject_root: Path, subject_id: str, output_root: Optional[Path] = None) -> bool:
    from shared.scalar_qc import scalar_qc_current

    outputs = _parcellation_outputs(subject_root, subject_id, output_root)
    source = outputs['wmparc'].parent / 'ddparcel_source.json'
    if source.exists() and not _anatomy_source_current(source, subject_root, output_root):
        return False
    receipt = outputs['wmparc'].parent / 'ddparcel_preparation.json'
    if receipt.exists():
        from shared.pipeline import _artifact_record, _cache_file_signature
        try:
            record = _read_json(receipt)
            inputs = _resolve_analysis_inputs(subject_root, subject_id, output_root)
            if record.get('source_inputs') != {key: _cache_file_signature(path) for key, path in inputs.items()}:
                return False
            if not record.get('inputs') or any(_artifact_record(Path(path))['sha256'] != digest
                    for path, digest in record['inputs'].items()):
                return False
        except (OSError, ValueError, TypeError, AttributeError):
            return False
    return scalar_qc_current(outputs["wmparc"].parent / "ddparcel_qc.json", list(outputs.values()))


def _surface_outputs(subject_root: Path, subject_id: str, output_root: Optional[Path] = None) -> dict[str, Path]:
    native_dir = subject_layout(subject_root, output_root=output_root).anatomy_input_dir("ddsurfer") / "pred_results_SlicerDSpace" / "native" / subject_id
    return {
        "white_lh": native_dir / f"{subject_id}_predicted_wm_lh.stl",
        "white_rh": native_dir / f"{subject_id}_predicted_wm_rh.stl",
        "pial_lh": native_dir / f"{subject_id}_predicted_pial_lh.stl",
        "pial_rh": native_dir / f"{subject_id}_predicted_pial_rh.stl",
    }


def _surface_complete(subject_root: Path, subject_id: str, output_root: Optional[Path] = None,
                      *, config: Optional[ProductOrchestratorConfig] = None) -> bool:
    from anatomy.surface.backend_bridge import sha256, upstream_record
    surface_root = subject_layout(subject_root, output_root=output_root).anatomy_input_dir("ddsurfer")
    try:
        manifest = json.loads((surface_root / 'surface_manifest.json').read_text())
        qc = json.loads((surface_root / 'surface_qc.json').read_text())
        if (manifest.get('upstream_commit') != upstream_record()['commit']
                or manifest.get('status') != 'completed' or not qc.get('automatic_qc_passed')):
            return False
        # Raw resume must reach upstream preflight, which verifies the complete
        # input/runtime signature and final-result receipt before skipping work.
        if manifest.get('input_mode') != 'registered' or manifest.get('version', 0) < 3:
            return False
        if (manifest.get('subject_id') != subject_id
                or manifest.get('subject_root') != str(subject_root.resolve())):
            return False
        expected = config or ProductOrchestratorConfig(subject_id=subject_id, subject_root=subject_root, targets=('surface',))
        if (manifest.get('precision') != expected.surface_precision
                or manifest.get('cpu_threads') != expected.surface_cpu_threads
                or manifest.get('max_gpu_gib') != expected.surface_max_gpu_gib):
            return False
        if expected.surface_python_bin and manifest.get('runtime', {}).get('executable') != str(interpreter_path(expected.surface_python_bin)):
            return False
        if expected.surface_atlas_t2 and manifest.get('resources', {}).get('registration_atlas') != str(expected.surface_atlas_t2.resolve()):
            return False
        inputs, outputs = manifest.get('input_sha256', {}), manifest.get('output_sha256', {})
        if len(inputs) < 15 or len(outputs) != 8:
            return False
        for path, digest in {**inputs, **outputs}.items():
            if Path(path).is_symlink() or sha256(path) != digest:
                return False
        required = set(map(str, _surface_outputs(subject_root, subject_id, output_root).values()))
        native = surface_root / 'pred_results_SlicerDSpace/native' / subject_id
        required.update(str(native / f'{hemi}.{kind}.obj') for hemi in ('lh', 'rh') for kind in ('white', 'pial'))
        if not required.issubset(outputs):
            return False
    except (OSError, ValueError, TypeError, AttributeError):
        return False
    return outputs_readable(_surface_outputs(subject_root, subject_id, output_root).values())


def _surface_postprocess_native_dir(
    subject_root: Path,
    subject_id: str,
    output_root: Optional[Path] = None,
) -> Path:
    return (
        subject_layout(subject_root, output_root=output_root).anatomy_input_dir("ddsurfer")
        / "pred_results_SlicerDSpace"
        / "native"
        / subject_id
    )


def _surface_postprocess_native_ready(native_dir: Path, subject_id: str, mode: str) -> bool:
    hemispheres = ("lh",) if mode == "left" else ("rh",) if mode == "right" else ("lh", "rh")
    if all((native_dir / f'{hemisphere}.{surface}.obj').is_file()
           for hemisphere in hemispheres for surface in ('white', 'pial')):
        return True
    return all(
        (native_dir / f"{subject_id}_predicted_{surface}_{hemisphere}_orig.stl").is_file()
        for hemisphere in hemispheres
        for surface in ("wm", "pial")
    )


def _surface_postprocess_wmparc_path(subject_id: str, mri_dir: Path) -> Optional[Path]:
    for candidate in (
        mri_dir / "DDSurfer-wmparc.nii.gz",
        mri_dir / f"{subject_id}-DDSurfer-wmparc.nii.gz",
    ):
        if candidate.is_file():
            return candidate
    return None


def _surface_postprocess_outputs(subject_root: Path, subject_id: str, output_root: Optional[Path] = None) -> dict[str, Path]:
    layout = subject_layout(subject_root, output_root=output_root)
    structural_root = layout.anatomy_input_dir("ddsurfer/postprocess")
    stats_root = layout.ddsurfer_postprocess_stats
    return {
        "manifest": structural_root / "manifest.json",
        "qc": structural_root / "qc.json",
        "surf": structural_root / "surf",
        "label": structural_root / "label",
        "stats": stats_root / "stats",
    }


def _surface_postprocess_required_files(structural_root: Path, mode: str = "whole") -> tuple[Path, ...]:
    hemispheres = ("lh",) if mode == "left" else ("rh",) if mode == "right" else ("lh", "rh")
    return (
        structural_root / "mri" / "brain.mgz",
        *(structural_root / "surf" / f"{hemisphere}.{surface}" for hemisphere in hemispheres for surface in ("white", "pial", "inflated", "sphere", "curv")),
    )


def _surface_postprocess_complete(
    subject_root: Path,
    subject_id: str,
    output_root: Optional[Path] = None,
    *,
    mode: str = "whole",
    native_dir: Optional[Path] = None,
    atlases: Optional[tuple[str, ...]] = None,
    runtime: Optional[Path] = None,
    freesurfer_home: Optional[Path] = None,
) -> bool:
    outputs = _surface_postprocess_outputs(subject_root, subject_id, output_root)
    structural_root = outputs["manifest"].parent
    try:
        manifest = json.loads(outputs['manifest'].read_text())
    except (OSError, ValueError):
        return False
    native_dir = native_dir or Path(manifest.get('native_dir') or
                                    _surface_postprocess_native_dir(subject_root, subject_id, output_root))
    hemis = ('lh',) if mode == 'left' else ('rh',) if mode == 'right' else ('lh', 'rh')
    if (manifest.get('kind') == 'ddsurfer_surface_only_postprocess' or
            any((native_dir / f'{hemi}.{kind}.obj').is_file() for hemi in hemis for kind in ('white', 'pial'))):
        from anatomy.surface.backend_bridge import native_surface_hashes, upstream_record
        try:
            current = native_surface_hashes(native_dir, mode)
            if (manifest.get('status') != 'completed'
                    or manifest.get('upstream_commit') != upstream_record()['commit']
                    or any(manifest.get('native_sha256', {}).get(k) != v for k, v in current.items())):
                return False
            expected_atlases = list(atlases or ('aparc', 'aparc.a2009s'))
            if manifest.get('cache_signature', {}).get('atlases') != expected_atlases:
                return False
            signature = manifest.get('cache_signature') or {}
            if runtime is not None or freesurfer_home is not None:
                from shared.surface_runtime import surface_runtime_identity
                if signature.get('surface_runtime') != surface_runtime_identity(runtime, freesurfer_home):
                    return False
            if signature.get('mode') != mode:
                return False
            runtime = signature.get('surface_runtime')
            if runtime is not None:
                from anatomy.surface.backend_bridge import sha256
                record = Path(runtime['home']) / ('runtime.json' if runtime['backend'] == 'native' else 'build-stamp.txt')
                if (sha256(record) if record.is_file() else None) != runtime.get('record_sha256'):
                    return False
            if signature.get('brain_source'):
                from anatomy.surface.backend_bridge import sha256
                if sha256(Path(signature['brain_source'])) != signature.get('brain_sha256'):
                    return False
            qc = _read_json(outputs['qc'])
            integrity = qc.get('integrity') or {}
            checks = integrity.get('checks')
            if (integrity.get('version') != 1 or integrity.get('automatic_qc_passed') is not True
                    or not isinstance(checks, dict) or not checks or not all(value is True for value in checks.values())):
                return False
            from shared.scalar_qc import _signature
            from anatomy.surface.postprocess_qc import postprocess_output_paths
            artifacts = integrity.get('output_artifacts')
            required = postprocess_output_paths(structural_root, hemis, expected_atlases, stats_dir=outputs['stats'])
            if artifacts != {str(path.resolve()): _signature(path) for path in required}:
                return False
            if not all(_nonempty_file(structural_root / 'label' / f'{hemi}.{atlas}.annot')
                       and _nonempty_file(outputs['stats'] / f'{hemi}.{atlas}.stats')
                       for hemi in hemis for atlas in expected_atlases):
                return False
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return False
    return (
        _nonempty_file(outputs["manifest"])
        and _qc_completed(outputs["qc"])
        and all(_nonempty_file(path) for path in _surface_postprocess_required_files(structural_root, mode))
    )


def _surface_postprocess_saf_fs_dir(subject_root: Path, subject_id: str, output_root: Optional[Path] = None) -> Optional[Path]:
    """Return a SAF-compatible postprocess root when whole-surface outputs are complete."""

    outputs = _surface_postprocess_outputs(subject_root, subject_id, output_root)
    structural_root = outputs["manifest"].parent
    if not _surface_postprocess_complete(subject_root, subject_id, output_root):
        return None
    return structural_root


def _act_prep_outputs(subject_root: Path, subject_id: str, output_root: Optional[Path] = None) -> dict[str, Path]:
    layout = subject_layout(subject_root, output_root=output_root)
    return {
        "five_tt": layout.act_prep / f"{subject_id}_5tt_from_labels.nii.gz",
        "gmwmi": layout.act_prep / f"{subject_id}_gmwmi_from_labels.nii.gz",
        "manifest": layout.act_prep / "act_prep_manifest.json",
        "qc": layout.act_prep / "act_prep_qc.json",
    }


def _act_anatomy_current(qc: dict, manifest: dict, *, allow_ddseg_only: bool = False) -> bool:
    """Require anatomical evidence separately from the MRtrix format check."""

    parcellation = (manifest.get("inputs") or {}).get("parcellation")
    if not isinstance(parcellation, dict) or not parcellation.get("exists"):
        return bool(allow_ddseg_only and manifest.get("allow_ddseg_only") is True
                    and qc.get("anatomical_prior_status") == "ddseg_only"
                    and qc.get("sgm_from_ddparcel") is False)
    try:
        volume = float(qc["five_tt_stats"]["channel_volume_voxel_units"]["sgm"])
        return (qc.get("sgm_from_ddparcel") is True and math.isfinite(volume) and volume > 0
                and _file_record_matches(parcellation, Path(str(parcellation.get("path") or ""))))
    except (KeyError, TypeError, ValueError):
        return False


def _act_prep_complete(subject_root: Path, subject_id: str, output_root: Optional[Path] = None, *, allow_ddseg_only: bool = False) -> bool:
    from tractography.act_prep import ACT_PREP_VERSION

    outputs = _act_prep_outputs(subject_root, subject_id, output_root)
    if not (outputs["five_tt"].is_file() and outputs["gmwmi"].is_file() and _qc_completed(outputs["qc"])):
        return False
    try:
        qc = _read_json(outputs["qc"])
        return (int(qc.get("version") or 0) >= ACT_PREP_VERSION
                and _act_anatomy_current(qc, _read_json(outputs["manifest"]), allow_ddseg_only=allow_ddseg_only))
    except Exception:
        return False


def _microstructure_dti_outputs(subject_root: Path, subject_id: str, output_root: Optional[Path] = None) -> dict[str, Path]:
    layout = subject_layout(subject_root, output_root=output_root)
    dti_dir = layout.microstructure_dti
    outputs = {
        "fa": dti_dir / f"{subject_id}_dti_FA.nii.gz",
        "md": dti_dir / f"{subject_id}_dti_MD.nii.gz",
        "ad": dti_dir / f"{subject_id}_dti_AD.nii.gz",
        "rd": dti_dir / f"{subject_id}_dti_RD.nii.gz",
        "l1": dti_dir / f"{subject_id}_dti_L1.nii.gz",
        "l2": dti_dir / f"{subject_id}_dti_L2.nii.gz",
        "l3": dti_dir / f"{subject_id}_dti_L3.nii.gz",
        "trace": dti_dir / f"{subject_id}_dti_Trace.nii.gz",
        "qc": dti_dir / "dti_qc.json",
        "manifest": layout.microstructure / "manifest.json",
    }
    if _resolve_parcellation(subject_root, subject_id, output_root) is not None:
        outputs.update({
            "roi_stats_csv": layout.regional_microstructure / f"{subject_id}_ROIWise-Microstructure.csv",
            "roi_stats_json": layout.regional_microstructure / f"{subject_id}_ROIWise-Microstructure.json",
        })
    return outputs


def _microstructure_dti_complete(subject_root: Path, subject_id: str, output_root: Optional[Path] = None, *, fit_method=None) -> bool:
    from microstructure.pipeline import DTI_KEYS, roi_statistics_current
    from shared.scalar_qc import scalar_qc_current

    outputs = _microstructure_dti_outputs(subject_root, subject_id, output_root)
    if fit_method is not None:
        try:
            recorded_method = str(_read_json(outputs["manifest"]).get("fit_method") or "").lower()
        except (OSError, ValueError, AttributeError):
            return False
        if recorded_method != str(fit_method).lower():
            return False
    atlas = _resolve_parcellation(subject_root, subject_id, output_root)
    if atlas is not None:
        regional = subject_layout(subject_root, output_root=output_root).regional_microstructure
        if not roi_statistics_current(
            regional / f"{subject_id}_ROIWise-Microstructure.json",
            regional / f"{subject_id}_ROIWise-Microstructure.csv",
            roi_label_map=atlas,
            required_metrics=DTI_KEYS,
            brain_mask_path=_resolve_analysis_inputs(subject_root, subject_id, output_root)["mask"],
        ):
            return False
    return scalar_qc_current(outputs["qc"], [outputs[key] for key in ("fa", "md", "ad", "rd", "l1", "l2", "l3", "trace")]) and _manifest_inputs_match(
        outputs["manifest"],
        _resolve_analysis_inputs(subject_root, subject_id, output_root),
        "raw_input_bundle",
    )


def _microstructure_freewater_outputs(subject_root: Path, subject_id: str, output_root: Optional[Path] = None) -> dict[str, Path]:
    layout = subject_layout(subject_root, output_root=output_root)
    freewater_dir = layout.microstructure_freewater
    return {
        "fwf": freewater_dir / f"{subject_id}_fw_fraction.nii.gz",
        "fw_fa": freewater_dir / f"{subject_id}_fw_FA.nii.gz",
        "fw_md": freewater_dir / f"{subject_id}_fw_MD.nii.gz",
        "valid_mask": freewater_dir / f"{subject_id}_fw_tissue_valid_mask.nii.gz",
        "qc": freewater_dir / "freewater_qc.json",
        "manifest": freewater_dir / "freewater_manifest.json",
        "progress": freewater_dir / "freewater_progress.json",
    }


def _microstructure_freewater_complete(subject_root: Path, subject_id: str, output_root: Optional[Path] = None, *, config=None) -> bool:
    from shared.scalar_qc import scalar_qc_current

    outputs = _microstructure_freewater_outputs(subject_root, subject_id, output_root)
    return _optional_microstructure_options_match(config, outputs["manifest"], "freewater") and scalar_qc_current(outputs["qc"], [outputs[key] for key in ("fwf", "fw_fa", "fw_md", "valid_mask")]) and _manifest_inputs_match(
        outputs["manifest"],
        _resolve_analysis_inputs(subject_root, subject_id, output_root),
        "raw_input_bundle",
    )


def _microstructure_noddi_outputs(subject_root: Path, subject_id: str, output_root: Optional[Path] = None) -> dict[str, Path]:
    layout = subject_layout(subject_root, output_root=output_root)
    noddi_dir = layout.microstructure_noddi
    return {
        "ndi": noddi_dir / f"{subject_id}_noddi_NDI.nii.gz",
        "odi": noddi_dir / f"{subject_id}_noddi_ODI.nii.gz",
        "isovf": noddi_dir / f"{subject_id}_noddi_IsoVF.nii.gz",
        "valid_mask": noddi_dir / f"{subject_id}_noddi_tissue_valid_mask.nii.gz",
        "rmse": noddi_dir / f"{subject_id}_noddi_RMSE.nii.gz",
        "nrmse": noddi_dir / f"{subject_id}_noddi_NRMSE.nii.gz",
        "qc": noddi_dir / "noddi_qc.json",
        "manifest": noddi_dir / "noddi_manifest.json",
        "progress": noddi_dir / "noddi_progress.json",
    }


def _microstructure_noddi_complete(subject_root: Path, subject_id: str, output_root: Optional[Path] = None, *, config=None) -> bool:
    from shared.scalar_qc import scalar_qc_current

    outputs = _microstructure_noddi_outputs(subject_root, subject_id, output_root)
    return _optional_microstructure_options_match(config, outputs["manifest"], "noddi") and scalar_qc_current(outputs["qc"], [outputs[key] for key in ("ndi", "odi", "isovf", "valid_mask", "rmse", "nrmse")]) and _manifest_inputs_match(
        outputs["manifest"],
        _resolve_analysis_inputs(subject_root, subject_id, output_root),
        "raw_input_bundle",
    )


def _optional_microstructure_options_match(config, manifest_path, stage):
    if config is None:
        return True
    from microstructure.pipeline import MicrostructureConfig, _expected_free_water_manifest, _expected_noddi_manifest
    from shared.pipeline import RawInputBundle

    try:
        inputs = _resolve_analysis_inputs(config.subject_root, config.subject_id, config.output_root)
        raw = RawInputBundle(config.subject_root, config.subject_id, inputs["dwi"].parent,
                             inputs["dwi"], inputs["bval"], inputs["bvec"], inputs["mask"])
        options = MicrostructureConfig(config.subject_id, raw, None, manifest_path.parent,
            free_water_chunk_size=config.microstructure_free_water_chunk_size,
            free_water_engine=config.microstructure_free_water_engine,
            free_water_n_jobs=config.microstructure_free_water_n_jobs,
            free_water_vox_per_chunk=config.microstructure_free_water_vox_per_chunk,
            free_water_max_fraction_for_tissue=config.microstructure_free_water_max_fraction_for_tissue)
        expected = _expected_free_water_manifest(options) if stage == "freewater" else _expected_noddi_manifest(options)
        return _read_json(manifest_path) == expected
    except (OSError, KeyError, ValueError):
        return False


def _act_inputs_current(csd_manifest_path: Path, *, allow_ddseg_only: bool = False) -> bool:
    """True when the CSD manifest's 5TT/GMWMI are unchanged, current-version ACT prep outputs."""

    from tractography.act_prep import ACT_PREP_VERSION

    try:
        dependencies = _read_json(csd_manifest_path).get("dependencies") or {}
        act_paths = {key: Path(str((dependencies.get(key) or {}).get("path") or "")) for key in ("five_tt", "gmwmi")}
        if not all(_file_record_matches(dependencies.get(key), path) for key, path in act_paths.items()):
            return False
        act_dir = act_paths["five_tt"].parent
        act_qc = _read_json(act_dir / "act_prep_qc.json")
        act_manifest = _read_json(act_dir / "act_prep_manifest.json")
        if act_qc.get("status") != "completed" or int(act_qc.get("version") or 0) < ACT_PREP_VERSION:
            return False
        if not _act_anatomy_current(act_qc, act_manifest, allow_ddseg_only=allow_ddseg_only):
            return False
        inputs = act_manifest.get("inputs") or {}
        for key in ("segmentation", "parcellation"):
            record = inputs.get(key)
            if isinstance(record, dict) and record.get("exists"):
                if not _file_record_matches(record, Path(str(record.get("path") or ""))):
                    return False
        for record in (inputs.get("probabilities") or {}).values():
            if isinstance(record, dict) and record.get("exists"):
                if not _file_record_matches(record, Path(str(record.get("path") or ""))):
                    return False
        return True
    except Exception:
        return False


def _csd_act_complete(
    qc_path: Path,
    manifest_path: Path,
    tractogram: Path,
    tractogram_trk: Path,
    tractogram_vtk: Path,
    weights: Path,
    analysis_inputs: dict[str, Path],
    *,
    allow_ddseg_only: bool = False,
) -> bool:
    """Require current inputs and decoded output evidence before CSD/ACT reuse.

    The recorded 5TT/GMWMI must still be the current-version ACT prep
    outputs built from the current DDSeg/DDParcel files, so a tractogram built
    from an outdated 5TT (e.g. mirrored DDSeg or pre-v3 ventral-DC sGM) is never reused.
    """

    from tractography.csd import CSD_ACT_MANIFEST_VERSION

    if not _qc_completed(qc_path) or not _manifest_inputs_match(
        manifest_path,
        analysis_inputs,
        "dependencies",
    ):
        return False
    if not _act_inputs_current(manifest_path, allow_ddseg_only=allow_ddseg_only):
        return False
    try:
        qc = _read_json(qc_path)
        manifest = _read_json(manifest_path)
        summary = qc.get("scientific_output_summary")
        parameters = manifest.get("parameters")
        if not isinstance(summary, dict) or not isinstance(parameters, dict):
            return False
        tck = summary.get("tck")
        trk = summary.get("trk")
        vtk = summary.get("vtk")
        sift2 = summary.get("sift2")
        if not all(isinstance(item, dict) for item in (tck, trk, vtk, sift2)):
            return False
        if int(manifest.get("version") or 0) < CSD_ACT_MANIFEST_VERSION:
            return False
        expected_count = int(parameters.get("select_streamlines", 0) or 0)
        length_summary = tck.get("length_mm")
        if not isinstance(length_summary, dict):
            return False
        return (
            expected_count > 0
            and int(tck.get("streamline_count", 0) or 0) == expected_count
            and int(trk.get("streamline_count", 0) or 0) == expected_count
            and int(vtk.get("streamline_count", 0) or 0) == expected_count
            and int(sift2.get("weight_count", 0) or 0) == expected_count
            and bool(tck.get("all_points_finite"))
            and bool(trk.get("all_points_finite"))
            and bool(sift2.get("all_finite"))
            and bool(sift2.get("all_nonnegative"))
            and float(length_summary.get("min", -1.0)) + 1e-3
            >= float(parameters.get("min_length_mm", 0.0))
            and float(length_summary.get("max", float("inf"))) - 1e-3
            <= float(parameters.get("max_length_mm", 0.0))
            and _file_record_matches(tck.get("file"), tractogram)
            and _file_record_matches(trk.get("file"), tractogram_trk)
            and _file_record_matches(vtk.get("file"), tractogram_vtk)
            and _file_record_matches(sift2.get("file"), weights)
        )
    except (OSError, TypeError, ValueError):
        return False


def _dki_config(config: ProductOrchestratorConfig, inputs: dict[str, Path]):
    from microstructure.dki import DkiConfig
    from shared.pipeline import RawInputBundle

    root = config.subject_root.expanduser().resolve()
    return DkiConfig(
        subject_id=config.subject_id,
        raw_input_bundle=RawInputBundle(
            subject_root=root, subject_id=config.subject_id, input_dir=inputs["dwi"].parent,
            **{f"{key}_path": inputs[key] for key in ("dwi", "bval", "bvec", "mask")},
        ),
        output_dir=subject_layout(root, output_root=config.output_root).microstructure_dki,
        fit_method=config.microstructure_dki_fit_method,
        matlab_executable=config.microstructure_dki_matlab_executable,
        mkcurve_root=config.microstructure_dki_mkcurve_root,
        force_recompute=not config.resume,
        **config.microstructure_dki_execution,
    )


def _microstructure_requested_outputs_complete(
    subject_root: Path,
    subject_id: str,
    *,
    output_root: Optional[Path],
    run_free_water: bool,
    run_noddi: bool,
) -> bool:
    if not _microstructure_dti_complete(subject_root, subject_id, output_root):
        return False
    if run_free_water and not _microstructure_freewater_complete(subject_root, subject_id, output_root):
        return False
    if run_noddi and not _microstructure_noddi_complete(subject_root, subject_id, output_root):
        return False
    return True


def _task_from_result(name: str, result, *, status: Optional[str] = None, reason: str = "") -> ProductTaskRecord:
    # Product result dataclasses expose their generated artifacts as Path
    # fields (for example matrix_path, bundle output_dir, manifest_path, and
    # qc_path).  Preserve those fields in the orchestration record instead of
    # silently emitting outputs={}, while deliberately ignoring arbitrary
    # strings such as commands, backend names, and status messages.
    outputs: dict[str, str | None] = {}
    result_attributes = getattr(result, "__dict__", {})
    if isinstance(result_attributes, dict):
        for key, value in result_attributes.items():
            if key.startswith("_"):
                continue
            if isinstance(value, Path):
                outputs[key] = value.as_posix()

    backend_status = getattr(result, "status", None) or 'completed'
    # A successful process cannot override a failed current QC contract, and
    # an explicit success must never promote a backend's reported failure.
    result_status = (backend_status if backend_status in {'failed', 'partial_failed', 'cancelled'}
                     else status or backend_status)
    result_reason = reason
    if not result_reason and result_status in {"failed", "partial_failed"}:
        result_reason = f"{name} reported {result_status}; inspect its QC record"
    return ProductTaskRecord(
        name=name,
        status=result_status,
        reason=result_reason,
        manifest_path=getattr(result, "manifest_path", None),
        qc_path=getattr(result, "qc_path", None),
        outputs=outputs,
    )


def _path_dict(paths: dict[str, Path | None]) -> dict[str, str | None]:
    return {key: value.as_posix() if isinstance(value, Path) else None for key, value in paths.items()}


def _missing_file_labels(paths: dict[str, Path]) -> list[str]:
    return [label for label, path in paths.items() if not path.is_file()]


def _run_plan_path(layout, run_name: str) -> Path:
    return layout.run_metadata / f"{run_name}_run_plan.json"


def _write_run_plan(plan: RunPlanResult) -> Path:
    plan.run_plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan.run_plan_path.write_text(json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan.run_plan_path


def _validate_surface_settings(config: ProductOrchestratorConfig) -> None:
    if config.surface_precision not in ('fp32', 'bf16', 'auto'):
        raise ValueError('surface_precision must be fp32, bf16 or auto.')
    if config.surface_cpu_threads < 1:
        raise ValueError('surface_cpu_threads must be positive.')
    if not math.isfinite(config.surface_max_gpu_gib) or config.surface_max_gpu_gib <= 0:
        raise ValueError('surface_max_gpu_gib must be finite and positive.')


def _validate_tractography_settings(config: ProductOrchestratorConfig) -> None:
    from tractography.wmql_protocol import CATALOG_PRESETS, canonical_wmql_selection
    if config.wmql_atlas_mode not in ('native', 'ddparcel_wm'):
        raise ValueError('Unknown WMQL atlas mode: ' + config.wmql_atlas_mode)
    if config.wmql_atlas_mode != 'native' and config.wmql_query_preset not in CATALOG_PRESETS:
        raise ValueError('Experimental DDParcel WM preparation requires a bundled WMQL catalogue')
    if config.wmql_query_preset in CATALOG_PRESETS:
        canonical_wmql_selection(config.wmql_query_selection, config.wmql_query_preset)
    if config.tractography_backend == 'ukf':
        from tractography.ukf import parameters
        parameters(config.ukf_parameters)
    from tractography.csd import SEEDING_MODES

    if config.mrtrix_random_seed is not None and config.mrtrix_random_seed < 0:
        raise ValueError("mrtrix_random_seed must be >= 0 when set.")
    if isinstance(config.tractcloud_random_seed, bool) or not isinstance(config.tractcloud_random_seed, int) or not 0 <= config.tractcloud_random_seed < 2**32:
        raise ValueError('tractcloud_random_seed must be an integer in [0, 2**32).')
    if config.tractography_seeding not in SEEDING_MODES:
        raise ValueError(f"tractography_seeding must be one of: {', '.join(SEEDING_MODES)}.")
    if not 0.0 <= float(config.ddseg_min_wm_dice) <= 1.0:
        raise ValueError("ddseg_min_wm_dice must be within [0, 1].")


def _with_shared_bundle_resources(config):
    from shared.optional_assets import resolve_wma_atlas_directory, resolve_tractcloud_data_directory
    from tractography.wma_atlases import atlas_definition
    atlas_definition(config.wma_atlas)
    if config.wma_atlas_dir is None and ("wma" in config.targets or
            "bundles" in config.targets and "wma" in _normalize_bundle_backends(config.bundle_backend)):
        atlas = resolve_wma_atlas_directory(asset_root=config.model_assets_dir, atlas_family=config.wma_atlas)
        if atlas is not None:
            config = replace(config, wma_atlas_dir=atlas)
    if config.tractcloud_data_dir is None and ("tractcloud" in config.targets or
            "bundles" in config.targets and "tractcloud" in _normalize_bundle_backends(config.bundle_backend)):
        data = resolve_tractcloud_data_directory(asset_root=config.model_assets_dir)
        if data is not None:
            config = replace(config, tractcloud_data_dir=data)
    return config


def _validate_bundle_tractography_source(config: ProductOrchestratorConfig) -> None:
    source = config.bundle_tractography_backend
    if not source:
        return
    from shared.algorithm_selection import algorithms
    if source not in {'csd_act', 'ukf', 'dti_det'}:
        raise ValueError('Bundle source must be csd_act, ukf, or dti_det; SAF is not a whole-brain tractogram.')
    if source not in algorithms(config.tractography_backend):
        raise ValueError('The bundle source must also be selected as a tractography backend.')
    if config.wma_input_tractography is not None or config.tractcloud_input_tractography is not None:
        raise ValueError('A bundle source backend cannot be combined with explicit WMA or TractCloud input files.')
    if not set(config.targets).intersection({'bundles', 'wma', 'tractcloud'}) or 'tractography' not in config.targets:
        raise ValueError('An explicit bundle source requires both a bundle target and tractography.')


def plan_subject_products(config: ProductOrchestratorConfig, *, write: bool = True) -> RunPlanResult:
    """Build a target-level run plan without executing product algorithms."""

    config = _with_shared_bundle_resources(_with_closure(config))
    from shared.algorithm_selection import algorithms
    _validate_bundle_tractography_source(config)
    if len(algorithms(config.tractography_backend)) > 1:
        from shared.multi_tracking import plan
        return plan(config, write=write)

    if not config.subject_id.strip():
        raise ValueError("subject_id is required.")
    if config.select_streamlines <= 0:
        raise ValueError("select_streamlines must be > 0.")
    if config.dti_min_length_mm < 0 or config.dti_max_length_mm <= config.dti_min_length_mm:
        raise ValueError("DTI streamline lengths require 0 <= dti_min_length_mm < dti_max_length_mm.")
    if config.mrtrix_nthreads is not None and config.mrtrix_nthreads < 0:
        raise ValueError("mrtrix_nthreads must be >= 0 when set.")
    _validate_tractography_settings(config)
    _validate_surface_settings(config)
    from shared.brain_mask import brain_mask_config_from_options, inspect_brain_mask_backend
    mask_backend = inspect_brain_mask_backend(brain_mask_config_from_options(config))
    if config.microstructure_free_water_chunk_size <= 0:
        raise ValueError("microstructure_free_water_chunk_size must be > 0.")
    if config.microstructure_free_water_engine not in {"serial", "joblib", "dask", "ray"}:
        raise ValueError("microstructure_free_water_engine must be one of: serial, joblib, dask, ray.")
    if config.microstructure_free_water_vox_per_chunk is not None and config.microstructure_free_water_vox_per_chunk <= 0:
        raise ValueError("microstructure_free_water_vox_per_chunk must be > 0.")
    if not 0.0 < float(config.microstructure_free_water_max_fraction_for_tissue) < 1.0:
        raise ValueError("microstructure_free_water_max_fraction_for_tissue must be between 0 and 1.")
    if config.tractography_backend not in TRACTOGRAPHY_BACKENDS:
        raise ValueError("tractography_backend must be one of: csd_act, ukf, dti_det, saf.")
    bundle_backends = _normalize_bundle_backends(config.bundle_backend)
    if config.connectome_backend not in {"python", "mrtrix"}:
        raise ValueError("connectome_backend must be one of: python, mrtrix.")

    targets = _normalize_targets(config)
    selected = set(targets)
    preflight_blocked_targets = {
        target: reason
        for target, reason in (config.preflight_blocked_targets or {}).items()
        if target in selected and reason
    }
    subject_root = config.subject_root.expanduser().resolve()
    output_root = config.output_root.expanduser().resolve() if config.output_root is not None else None
    layout = subject_layout(subject_root, output_root=output_root)
    run_name = normalize_run_name(config.run_name)
    primary_tractography_backend = _primary_tractography_backend(config)
    raw = _resolve_raw_inputs(subject_root)
    analysis_inputs = _resolve_analysis_inputs(subject_root, config.subject_id, output_root)
    raw_missing = _missing_file_labels(raw)
    from shared.dicom_input import is_dicom_root, converter_path
    dicom_pending = bool(raw_missing and is_dicom_root(subject_root))
    dicom_error = None
    if dicom_pending:
        try:
            converter_path(config.dwiconvert_executable)
        except FileNotFoundError as exc:
            dicom_error = str(exc)
        raw_missing = ["mask"]
    mask_can_be_generated = raw_missing == ["mask"] and mask_backend["available"] and config.brain_mask_backend != "provided"
    if mask_can_be_generated:
        raw_missing = []
    model_assets_dir = config.model_assets_dir.expanduser().resolve() if config.model_assets_dir is not None else None
    atlas_t2 = config.atlas_t2.expanduser().resolve() if config.atlas_t2 is not None else _default_atlas_t2(model_assets_dir)

    preprocessing_outputs = _preprocessing_outputs(subject_root, config.subject_id, output_root)
    segmentation_outputs = _segmentation_outputs(subject_root, output_root)
    parcellation_outputs = _parcellation_outputs(subject_root, config.subject_id, output_root)
    surface_outputs = _surface_outputs(subject_root, config.subject_id, output_root)
    micro_outputs = _microstructure_dti_outputs(subject_root, config.subject_id, output_root)
    freewater_outputs = _microstructure_freewater_outputs(subject_root, config.subject_id, output_root)
    noddi_outputs = _microstructure_noddi_outputs(subject_root, config.subject_id, output_root)
    postprocess_outputs = _surface_postprocess_outputs(subject_root, config.subject_id, output_root)

    csd_dir = layout.csd if run_name == "csd_act" else layout.csd / run_name
    tractogram_dir = layout.tractograms / run_name
    dti_tractogram_dir = layout.tractograms_dti_det / run_name
    wmql_dir = layout.bundles_wmql / run_name
    saf_dir = _saf_output_dir(layout, run_name)
    wma_dir = _wma_output_dir(layout, run_name)
    tractcloud_dir = _tractcloud_output_dir(layout, run_name)
    connectome_dir = canonical_connectome_output_dir(subject_root, run_name, "streamline_count", output_root=output_root)
    sift2_connectome_dir = canonical_connectome_output_dir(
        subject_root,
        run_name,
        "sift2_weighted_streamline_count",
        output_root=output_root,
    )
    csd_qc = csd_dir / "csd_act_qc.json"
    dti_stem = f"{config.subject_id}-dti-deterministic"
    dti_qc = dti_tractogram_dir / f"{dti_stem}_qc.json"
    dti_manifest = dti_tractogram_dir / f"{dti_stem}_manifest.json"
    csd_tractogram = tractogram_dir / f"{config.subject_id}_csd_act_ifod2.tck"
    csd_tractogram_trk = csd_tractogram.with_suffix(".trk")
    csd_tractogram_vtk = csd_tractogram.with_suffix(".vtk")
    sift2_weights = csd_tractogram.with_name(f"{csd_tractogram.stem}_sift2_weights.txt")
    dti_tractogram_trk = dti_tractogram_dir / f"{dti_stem}.trk"
    tractogram = csd_tractogram if primary_tractography_backend == "csd_act" else dti_tractogram_trk
    tractogram_trk = csd_tractogram_trk if primary_tractography_backend == "csd_act" else dti_tractogram_trk
    from tractography.ukf import paths as ukf_paths
    ukf_dir = layout.tractograms / 'ukf' / run_name
    ukf_outputs = ukf_paths(config.subject_id, ukf_dir)
    if primary_tractography_backend == 'ukf':
        tractogram, tractogram_trk = ukf_outputs['tractogram'], ukf_outputs['tractogram_trk']
    current_vtk = (tractogram if primary_tractography_backend == 'ukf' else
                   dti_tractogram_trk.with_suffix('.vtk') if primary_tractography_backend == 'dti_det' else
                   csd_tractogram_vtk)
    ukf_cached = primary_tractography_backend != 'ukf' or _ukf_current(config, analysis_inputs, ukf_dir)
    saf_tractogram = saf_dir / "SAF.tck"
    if config.wma_input_tractography is not None:
        wma_input_tractography = _resolve_wma_input_tractography(
            subject_root,
            config.wma_input_tractography,
            output_root=output_root,
        )
    else:
        wma_input_tractography = _resolve_wma_input_tractography(
            subject_root,
            None,
            output_root=output_root,
        )
    if primary_tractography_backend == 'ukf' and config.wma_input_tractography is None:
        wma_input_tractography = tractogram
    elif config.bundle_tractography_backend:
        wma_input_tractography = current_vtk
    tractcloud_input_tractography = _resolve_bundle_input_tractography(
        subject_root,
        config.tractcloud_input_tractography,
        output_root=output_root,
        run_name=run_name,
        current_run_tractography=current_vtk,
        require_current_run=True,
        allow_planned_current_run=True,
    )
    slicer_executable = _resolve_slicer_executable(config)
    segmentation = _resolve_segmentation(subject_root, output_root)
    parcellation = _resolve_parcellation(subject_root, config.subject_id, output_root)

    items: list[RunPlanItem] = []
    available: dict[str, bool] = {}

    def target_completed(target: str) -> bool:
        if target == "preprocessing":
            return _preprocessing_complete(subject_root, config.subject_id, output_root, config=config)
        if target == "segmentation":
            return _segmentation_complete(subject_root, output_root)
        if target == "parcellation":
            return _parcellation_complete(subject_root, config.subject_id, output_root)
        if target == "surface":
            return _surface_complete(subject_root, config.subject_id, output_root, config=config)
        if target == "surface_postprocess":
            if config.postprocess_force:
                return False
            return _surface_postprocess_complete(
                subject_root,
                config.subject_id,
                output_root,
                mode=config.postprocess_mode,
                native_dir=_resolve_optional_path(config.postprocess_native_dir),
                atlases=config.postprocess_atlases,
                runtime=_resolve_optional_path(config.postprocess_runtime),
                freesurfer_home=_resolve_optional_path(config.postprocess_freesurfer_home),
            )
        if target == "microstructure":
            return _microstructure_dti_complete(subject_root, config.subject_id, output_root, fit_method=config.microstructure_fit_method)
        if target == "microstructure_freewater":
            return _microstructure_freewater_complete(subject_root, config.subject_id, output_root, config=config)
        if target == "microstructure_noddi":
            return _microstructure_noddi_complete(subject_root, config.subject_id, output_root, config=config)
        if target == "microstructure_dki":
            from microstructure.dki import dki_outputs_current
            return dki_outputs_current(_dki_config(config, analysis_inputs))
        if target == "tractography":
            if primary_tractography_backend == 'ukf':
                return ukf_cached
            if primary_tractography_backend == "csd_act":
                return _csd_act_complete(
                    csd_qc,
                    csd_dir / "csd_act_manifest.json",
                    csd_tractogram,
                    csd_tractogram_trk,
                    csd_tractogram_vtk,
                    sift2_weights,
                    analysis_inputs,
                    allow_ddseg_only=config.allow_act_ddseg_only,
                )
            return _qc_completed(dti_qc) and tractogram_trk.is_file()
        if target == "bundles":
            return _wmql_complete(
                wmql_dir / "wmql_qc.json",
                query_preset=config.wmql_query_preset,
                query_selection=config.wmql_query_selection,
                atlas_mode=config.wmql_atlas_mode,
                input_tractography=tractogram_trk,
                parcellation=parcellation,
            )
        if target == "saf":
            fod_candidate = csd_dir / f"{config.subject_id}_wmfod_norm.mif"
            fod_path = fod_candidate if fod_candidate.is_file() else csd_dir / f"{config.subject_id}_wmfod.mif"
            fs_dir = _resolve_optional_path(config.saf_fs_dir) or _surface_postprocess_saf_fs_dir(
                subject_root,
                config.subject_id,
                output_root,
            )
            return _saf_complete(saf_dir, fod_path=fod_path, fs_dir=fs_dir)
        if target == "wma":
            return _wma_complete(
                wma_dir,
                input_tractography=wma_input_tractography,
                atlas_dir=config.wma_atlas_dir,
                atlas_family=config.wma_atlas,
            )
        if target == "tractcloud":
            return _tractcloud_complete(
                tractcloud_dir,
                input_tractography=tractcloud_input_tractography,
                random_seed=config.tractcloud_random_seed,
                config=config,
            )
        if target == "connectome":
            connectome_stem = f"{config.subject_id}-connectome-streamline_count"
            return _connectome_complete(
                connectome_dir / f"{connectome_stem}_qc.json",
                connectome_dir / f"{connectome_stem}.csv",
                connectome_dir / f"{config.subject_id}-connectome-nodes.tsv",
                manifest_path=connectome_dir / f"{connectome_stem}_manifest.json",
                tractography=tractogram,
                parcellation=parcellation,
            )
        return False

    def add(
        target: str,
        status: str,
        reason: str = "",
        reason_category: str = "",
        *,
        inputs: dict[str, Path | None] | None = None,
        outputs: dict[str, Path | None] | None = None,
        dependencies: dict[str, str | None] | None = None,
        provides: bool = True,
    ) -> None:
        item = RunPlanItem(
            target=target,
            task_name=_task_name_for_target(target, config),
            status=status,
            reason=reason,
            reason_category=reason_category,
            runnable=status != "blocked",
            will_run=status == "will_run",
            reused=status == "skipped",
            inputs=_path_dict(inputs or {}),
            outputs=_path_dict(outputs or {}),
            dependencies=dependencies or {},
            requested=target not in config.target_required_by,
            required_by=tuple(config.target_required_by.get(target, ())),
        )
        items.append(item)
        available[target] = provides and status in {"will_run", "skipped"}

    def raw_input_block(target: str, outputs: dict[str, Path | None] | None = None) -> bool:
        if raw_missing:
            if raw_missing == ["mask"] and not mask_backend["available"]:
                add(target, "blocked", mask_failure_message(
                    f"brain mask backend {config.brain_mask_backend} unavailable ("
                    + ", ".join(mask_backend["missing_required"]) + ")"), "environment", inputs=raw,
                    outputs=outputs, provides=False)
                return True
            add(
                target,
                "blocked",
                "missing raw input: " + ", ".join(raw_missing),
                "data",
                inputs=raw,
                outputs=outputs,
                provides=False,
            )
            return True
        return False

    for target in targets:
        if dicom_error:
            add(target, "blocked", dicom_error, "environment", inputs=raw, provides=False)
            continue
        if target in preflight_blocked_targets and not target_completed(target):
            add(
                target,
                "blocked",
                preflight_blocked_targets[target],
                "environment",
                provides=False,
            )
            continue

        if target == "validation":
            add(
                target,
                "will_run",
                "validate subject inputs and write QC report",
                "runnable",
                inputs=raw,
                outputs={"report": layout.run_metadata / "validation" / f"{config.subject_id}_input_validation.json"},
                provides=False,
            )
            continue

        if target == "preprocessing":
            if config.resume and _preprocessing_complete(subject_root, config.subject_id, output_root, config=config):
                add(target, "skipped", "completed outputs found", "reuse", inputs=raw, outputs=preprocessing_outputs)
            elif not raw_input_block(target, preprocessing_outputs):
                add(target, "will_run", "ready to run DWI preprocessing", "runnable", inputs=raw, outputs=preprocessing_outputs)
            continue

        if target == "segmentation":
            if config.resume and _segmentation_complete(subject_root, output_root):
                add(target, "skipped", "completed outputs found", "reuse", inputs=raw, outputs=segmentation_outputs)
            elif not raw_input_block(target, segmentation_outputs):
                add(target, "will_run", "ready to run DDSeg", "runnable", inputs=raw, outputs=segmentation_outputs)
            continue

        if target == "parcellation":
            if config.resume and _parcellation_complete(subject_root, config.subject_id, output_root):
                add(target, "skipped", "completed outputs found", "reuse", inputs=raw, outputs=parcellation_outputs)
            elif not atlas_t2.is_file():
                add(
                    target,
                    "blocked",
                    f"missing atlas T2: {atlas_t2}",
                    "data",
                    inputs={**raw, "atlas_t2": atlas_t2},
                    outputs=parcellation_outputs,
                    provides=False,
                )
            elif not raw_input_block(target, parcellation_outputs):
                add(target, "will_run", "ready to run DDParcel", "runnable", inputs={**raw, "atlas_t2": atlas_t2}, outputs=parcellation_outputs)
            continue

        if target == "surface":
            if config.resume and _surface_complete(subject_root, config.subject_id, output_root, config=config):
                add(target, "skipped", "completed outputs found", "reuse", inputs=raw, outputs=surface_outputs)
            elif not raw_input_block(target, surface_outputs):
                add(target, "will_run", "ready to run DDSurfer surface", "runnable", inputs=raw, outputs=surface_outputs)
            continue

        if target == "surface_postprocess":
            explicit_native_dir = _resolve_optional_path(config.postprocess_native_dir)
            explicit_mri_dir = _resolve_optional_path(config.postprocess_mri_dir)
            native_dir = explicit_native_dir or _surface_postprocess_native_dir(
                subject_root,
                config.subject_id,
                output_root,
            )
            mri_dir = explicit_mri_dir or layout.anatomy_input_dir("ddparcel")
            native_ready = _surface_postprocess_native_ready(native_dir, config.subject_id, config.postprocess_mode)
            wmparc_path = _surface_postprocess_wmparc_path(config.subject_id, mri_dir)
            native_will_run = explicit_native_dir is None and available.get("surface", False)
            parcellation_will_run = explicit_mri_dir is None and available.get("parcellation", False)
            current_backend = config.postprocess_backend_root is None and (
                native_will_run or (native_dir / 'lh.white.obj').is_file() or (native_dir / 'rh.white.obj').is_file())
            dependencies = {
                "surface": "available" if native_ready else "will_run" if native_will_run else "missing",
                "parcellation": 'not_required' if current_backend else "available" if wmparc_path else "will_run" if parcellation_will_run else "missing",
            }
            if config.resume and not config.postprocess_force and _surface_postprocess_complete(
                subject_root,
                config.subject_id,
                output_root,
                mode=config.postprocess_mode,
                native_dir=explicit_native_dir,
                atlases=config.postprocess_atlases,
                runtime=_resolve_optional_path(config.postprocess_runtime),
                freesurfer_home=_resolve_optional_path(config.postprocess_freesurfer_home),
            ):
                add(target, "skipped", "completed outputs found", "reuse", outputs=postprocess_outputs)
            elif explicit_native_dir is not None and not explicit_native_dir.is_dir():
                add(target, "blocked", "invalid explicit postprocess inputs: postprocess_native_dir", "data", inputs={"native_dir": native_dir}, outputs=postprocess_outputs, provides=False)
            elif explicit_mri_dir is not None and not explicit_mri_dir.is_dir():
                add(target, "blocked", "invalid explicit postprocess inputs: postprocess_mri_dir", "data", inputs={"mri_dir": mri_dir}, outputs=postprocess_outputs, provides=False)
            elif not native_ready and not native_will_run:
                add(
                    target,
                    "blocked",
                    "missing DDSurfer *_orig.stl inputs; select Surface or provide postprocess_native_dir",
                    "upstream",
                    inputs={"native_dir": native_dir, "mri_dir": mri_dir, "wmparc": wmparc_path},
                    outputs=postprocess_outputs,
                    dependencies=dependencies,
                    provides=False,
                )
            elif not current_backend and wmparc_path is None and not parcellation_will_run:
                add(
                    target,
                    "blocked",
                    "missing DDSurfer wmparc input; select Parcellation or provide postprocess_mri_dir",
                    "upstream",
                    inputs={"native_dir": native_dir, "mri_dir": mri_dir, "wmparc": wmparc_path},
                    outputs=postprocess_outputs,
                    dependencies=dependencies,
                    provides=False,
                )
            else:
                add(
                    target,
                    "will_run",
                    "ready to run DDSurfer postprocess",
                    "runnable",
                    inputs={"native_dir": native_dir, "mri_dir": mri_dir, "wmparc": wmparc_path},
                    outputs=postprocess_outputs,
                    dependencies=dependencies,
                )
            continue

        if target == "microstructure":
            outputs = {**micro_outputs}
            if config.resume and _microstructure_dti_complete(subject_root, config.subject_id, output_root, fit_method=config.microstructure_fit_method):
                add(target, "skipped", "completed DTI microstructure outputs found", "reuse", inputs=analysis_inputs, outputs=outputs)
            elif not raw_input_block(target, outputs):
                add(target, "will_run", "ready to run Microstructure DTI", "runnable", inputs=analysis_inputs, outputs=outputs, dependencies={"parcellation": "optional ROI stats"})
            continue

        if target == "microstructure_freewater":
            dti_ready = _microstructure_dti_complete(subject_root, config.subject_id, output_root) or available.get("microstructure", False)
            if config.resume and _microstructure_freewater_complete(subject_root, config.subject_id, output_root, config=config):
                add(target, "skipped", "completed Free Water outputs found", "reuse", inputs=analysis_inputs, outputs=freewater_outputs, dependencies={"microstructure": "available" if dti_ready else None})
            elif not dti_ready:
                add(target, "blocked", _upstream_reason(selected, "microstructure", "missing Microstructure DTI outputs; select microstructure or provide existing output"), "upstream", inputs=raw, outputs=freewater_outputs, dependencies={"microstructure": "missing"}, provides=False)
            elif not raw_input_block(target, freewater_outputs):
                add(target, "will_run", "ready to run Microstructure Free Water", "runnable", inputs=analysis_inputs, outputs=freewater_outputs, dependencies={"microstructure": "available" if _microstructure_dti_complete(subject_root, config.subject_id, output_root) else "will_run"})
            continue

        if target == "microstructure_noddi":
            dti_ready = _microstructure_dti_complete(subject_root, config.subject_id, output_root) or available.get("microstructure", False)
            if config.resume and _microstructure_noddi_complete(subject_root, config.subject_id, output_root, config=config):
                add(target, "skipped", "completed NODDI outputs found", "reuse", inputs=analysis_inputs, outputs=noddi_outputs, dependencies={"microstructure": "available" if dti_ready else None})
            elif not dti_ready:
                add(target, "blocked", _upstream_reason(selected, "microstructure", "missing Microstructure DTI outputs; select microstructure or provide existing output"), "upstream", inputs=raw, outputs=noddi_outputs, dependencies={"microstructure": "missing"}, provides=False)
            elif not raw_input_block(target, noddi_outputs):
                add(target, "will_run", "ready to run Microstructure NODDI", "runnable", inputs=analysis_inputs, outputs=noddi_outputs, dependencies={"microstructure": "available" if _microstructure_dti_complete(subject_root, config.subject_id, output_root) else "will_run"})
            continue

        if target == "microstructure_dki":
            from microstructure.dki import dki_output_paths, dki_outputs_current, inspect_dki_inputs

            dki_config = _dki_config(config, analysis_inputs)
            outputs = dki_output_paths(config.subject_id, dki_config.output_dir)
            if raw_input_block(target, outputs):
                continue
            # Planning checks protocol metadata, never fits or generates a mask.
            inspection = inspect_dki_inputs(dki_config, require_mask=False)
            if not inspection["eligible"]:
                add(target, "blocked", "; ".join(inspection["reasons"]), "acquisition_or_backend",
                    inputs=analysis_inputs, outputs=outputs, provides=False)
            elif config.resume and dki_outputs_current(dki_config):
                add(target, "skipped", "matching completed DKI outputs found", "reuse",
                    inputs=analysis_inputs, outputs=outputs)
            else:
                add(target, "will_run", "eligible multi-shell DKI; final mask checked before fitting", "runnable",
                    inputs=analysis_inputs, outputs=outputs)
            continue

        if target == "tractography":
            if primary_tractography_backend == 'ukf':
                from tractography.ukf import resolve_executable
                if config.resume and ukf_cached:
                    add(target, 'skipped', 'matching official UKF outputs found', 'reuse', outputs=ukf_outputs)
                elif not raw_input_block(target, ukf_outputs):
                    try:
                        resolve_executable(config.ukf_executable)
                        resolve_executable(config.dwiconvert_executable, name='DWIConvert')
                    except (OSError, RuntimeError) as exc:
                        add(target, 'blocked', str(exc), 'environment', outputs=ukf_outputs, provides=False)
                    else:
                        add(target, 'will_run', 'official UKF CLI; explicit author defaults/overrides', 'runnable',
                            inputs=analysis_inputs, outputs=ukf_outputs)
                continue
            tractography_qc = csd_qc if primary_tractography_backend == "csd_act" else dti_qc
            tractography_outputs = {
                "tractogram": tractogram,
                "tractogram_trk": tractogram_trk,
                "tractogram_vtk": current_vtk,
                "qc": tractography_qc,
                "manifest": csd_dir / "csd_act_manifest.json" if primary_tractography_backend == "csd_act" else dti_manifest,
            }
            complete = config.resume and (
                _csd_act_complete(
                    csd_qc,
                    csd_dir / "csd_act_manifest.json",
                    csd_tractogram,
                    csd_tractogram_trk,
                    csd_tractogram_vtk,
                    sift2_weights,
                    analysis_inputs,
                    allow_ddseg_only=config.allow_act_ddseg_only,
                )
                and (not config.bundle_tractography_backend or _csd_bundle_parameters_current(config, csd_dir / 'csd_act_manifest.json'))
                if primary_tractography_backend == "csd_act"
                else _qc_completed(tractography_qc) and tractogram_trk.is_file()
                and (not config.bundle_tractography_backend or (_dti_bundle_source_current(config, dti_manifest, analysis_inputs)
                     and _dti_bundle_qc_acceptable(config, dti_qc, tractogram_trk)))
            )
            segmentation_ready = segmentation is not None or available.get("segmentation", False)
            if complete:
                add(target, "skipped", "completed outputs found", "reuse", inputs=analysis_inputs, outputs=tractography_outputs, dependencies={"segmentation": "available" if segmentation_ready else None})
            elif raw_input_block(target, tractography_outputs):
                pass
            elif primary_tractography_backend == "csd_act" and not segmentation_ready:
                add(target, "blocked", _upstream_reason(selected, "segmentation", "missing DDSeg segmentation; select segmentation or provide existing output"), "upstream", inputs=raw, outputs=tractography_outputs, dependencies={"segmentation": "missing"}, provides=False)
            elif (primary_tractography_backend == "csd_act" and not config.allow_act_ddseg_only
                  and parcellation is None and not available.get("parcellation", False)):
                add(target, "blocked", _upstream_reason(selected, "parcellation", "missing DDParcel anatomical labels for ACT; select parcellation or provide existing output"), "upstream", inputs=raw, outputs=tractography_outputs, dependencies={"parcellation": "missing"}, provides=False)
            else:
                add(target, "will_run", f"ready to run tractography backend {config.tractography_backend}", "runnable", inputs=analysis_inputs, outputs=tractography_outputs, dependencies={"segmentation": "available" if segmentation is not None else ("will_run" if available.get("segmentation", False) else None)})
            continue

        if target == "bundles":
            wmql_outputs = {"qc": wmql_dir / "wmql_qc.json", "manifest": wmql_dir / "wmql_manifest.json", "bundle_dir": wmql_dir}
            parcellation_ready = parcellation is not None or available.get("parcellation", False)
            tractogram_ready = (available.get('tractography', False) if primary_tractography_backend == 'ukf'
                                else tractogram_trk.is_file() or available.get("tractography", False))
            if config.bundle_tractography_backend:
                tractogram_ready = available.get('tractography', False)
            source_reused = any(item.target == 'tractography' and item.status == 'skipped' for item in items)
            if config.resume and ukf_cached and (not config.bundle_tractography_backend or source_reused) and _wmql_complete(
                wmql_outputs["qc"],
                query_preset=config.wmql_query_preset,
                query_selection=config.wmql_query_selection,
                atlas_mode=config.wmql_atlas_mode,
                manifest_path=wmql_outputs["manifest"],
                input_tractography=tractogram_trk,
                parcellation=parcellation,
            ):
                add(target, "skipped", "completed outputs found", "reuse", outputs=wmql_outputs, dependencies={"parcellation": "available", "tractogram": "available"})
            elif not parcellation_ready:
                add(target, "blocked", _upstream_reason(selected, "parcellation", "missing parcellation; select parcellation or provide existing output"), "upstream", outputs=wmql_outputs, dependencies={"parcellation": "missing"}, provides=False)
            elif not tractogram_ready:
                add(target, "blocked", _upstream_reason(selected, "tractography", "missing TRK tractogram; select tractography or provide existing output"), "upstream", outputs=wmql_outputs, dependencies={"tractogram": "missing"}, provides=False)
            else:
                add(target, "will_run", "ready to run WMQL bundles", "runnable", inputs={"tractogram_trk": tractogram_trk, "parcellation": parcellation}, outputs=wmql_outputs, dependencies={"parcellation": "available" if parcellation is not None else "will_run", "tractogram": "available" if tractogram_trk.is_file() else "will_run"})
            continue

        if target == "saf":
            saf_outputs = {"tractogram": saf_tractogram, "tractogram_vtk": saf_dir / "SAF.vtk", "qc": saf_dir / "saf_qc.json", "manifest": saf_dir / "saf_manifest.json"}
            fod_candidate = csd_dir / f"{config.subject_id}_wmfod_norm.mif"
            fod_path = fod_candidate if fod_candidate.is_file() else (csd_dir / f"{config.subject_id}_wmfod.mif")
            fs_dir = _resolve_optional_path(config.saf_fs_dir) or _surface_postprocess_saf_fs_dir(subject_root, config.subject_id, output_root)
            if config.resume and _saf_complete(saf_dir, fod_path=fod_path, fs_dir=fs_dir):
                add(target, "skipped", "completed outputs found", "reuse", outputs=saf_outputs)
            else:
                fod_will_run = (
                    primary_tractography_backend == "csd_act"
                    and "tractography" in selected
                    and available.get("tractography", False)
                )
                if fs_dir is None and not available.get("surface_postprocess", False):
                    add(target, "blocked", "missing --saf-fs-dir or completed DDSurfer postprocess outputs", "upstream", outputs=saf_outputs, provides=False)
                elif not fod_path.is_file() and not fod_will_run:
                    add(target, "blocked", "missing CSD FOD output for SAF", "upstream", outputs=saf_outputs, dependencies={"fod": "missing"}, provides=False)
                else:
                    add(target, "will_run", "ready to run SAF", "runnable", inputs={"fs_dir": fs_dir, "fod": fod_path}, outputs=saf_outputs, dependencies={"surface_postprocess": "available" if fs_dir else "will_run", "fod": "available" if fod_path.is_file() else "will_run:tractography.csd_act"})
            continue

        if target == "wma":
            wma_outputs = {"qc": wma_dir / "wma_org_atlas_qc.json", "manifest": wma_dir / "wma_org_atlas_manifest.json", "output_dir": wma_dir}
            wma_input_available = bool(
                wma_input_tractography is not None and wma_input_tractography.is_file()
            )
            wma_input_will_run = (primary_tractography_backend == 'ukf' and
                                 config.wma_input_tractography is None and available.get('tractography', False))
            if primary_tractography_backend == 'ukf' and config.wma_input_tractography is None:
                wma_input_available = ukf_cached
            if config.bundle_tractography_backend:
                wma_input_available = False
                wma_input_will_run = available.get('tractography', False)
            if (
                config.resume
                and (ukf_cached or config.wma_input_tractography is not None)
                and (not config.bundle_tractography_backend or any(item.target == 'tractography' and item.status == 'skipped' for item in items))
                and _wma_complete(
                    wma_dir,
                    input_tractography=wma_input_tractography,
                    atlas_dir=config.wma_atlas_dir,
                    atlas_family=config.wma_atlas,
                )
            ):
                add(target, "skipped", "completed outputs found", "reuse", outputs=wma_outputs)
            elif not wma_input_available and not wma_input_will_run:
                add(
                    target,
                    "blocked",
                    "Select UKF in Advanced settings for automatic UKF-to-WMA tracking, or optionally provide an existing WMA-compatible input; CSD is not silently substituted",
                    "data",
                    outputs=wma_outputs,
                    provides=False,
                )
            elif (
                wma_input_tractography is not None
                and wma_input_tractography.suffix.lower() == ".tck"
                and _resolve_mrtrix_command("tckconvert", config.mrtrix_bin_dir) is None
            ):
                add(
                    target,
                    "blocked",
                    "TCK input requires MRtrix3 tckconvert",
                    "environment",
                    inputs={"input_tractography": wma_input_tractography},
                    outputs=wma_outputs,
                    provides=False,
                )
            elif config.wma_atlas_dir is None:
                add(target, "blocked", "missing --wma-atlas-dir", "data", inputs={"input_tractography": wma_input_tractography}, outputs=wma_outputs, provides=False)
            elif slicer_executable is None:
                add(target, "blocked", "missing Slicer executable for WMA", "environment", inputs={"input_tractography": wma_input_tractography}, outputs=wma_outputs, provides=False)
            else:
                add(
                    target,
                    "will_run",
                    "ready to run WMA " + config.wma_atlas + " atlas",
                    "runnable",
                    inputs={"input_tractography": wma_input_tractography, "atlas_dir": config.wma_atlas_dir},
                    outputs=wma_outputs,
                    dependencies={
                        "input_tractography": "available" if wma_input_available else "will_run:tractography"
                    },
                )
            continue

        if target == "tractcloud":
            tractcloud_outputs = {"qc": tractcloud_dir / "tractcloud_qc.json", "manifest": tractcloud_dir / "tractcloud_manifest.json", "output_dir": tractcloud_dir}
            tractcloud_input_available = bool(
                tractcloud_input_tractography is not None and tractcloud_input_tractography.is_file()
            )
            tractcloud_input_will_run = bool(
                tractcloud_input_tractography == current_vtk
                and (available.get("tractography", False) or
                     (primary_tractography_backend == 'dti_det' and _qc_completed(dti_qc)
                      and tractogram_trk.is_file()))
            )
            if config.bundle_tractography_backend:
                tractcloud_input_available = False
                tractcloud_input_will_run = available.get('tractography', False)
            if config.resume and (ukf_cached or config.tractcloud_input_tractography is not None) and (not config.bundle_tractography_backend or any(item.target == 'tractography' and item.status == 'skipped' for item in items)) and _tractcloud_complete(
                tractcloud_dir,
                input_tractography=tractcloud_input_tractography,
                random_seed=config.tractcloud_random_seed,
                config=config,
            ):
                add(target, "skipped", "completed outputs found", "reuse", outputs=tractcloud_outputs)
            elif not tractcloud_input_available and not tractcloud_input_will_run:
                add(target, "blocked", "missing whole-brain VTK/VTP tractography for TractCloud", "data", outputs=tractcloud_outputs, provides=False)
            else:
                add(target, "will_run", "ready to run TractCloud; same-run VTK is prepared automatically", "runnable", inputs={"input_tractography": tractcloud_input_tractography}, outputs=tractcloud_outputs, dependencies={"input_tractography": "available" if tractcloud_input_available else "will_run:tractography." + primary_tractography_backend})
            continue

        if target == "connectome":
            conn_qc = connectome_dir / f"{config.subject_id}-connectome-streamline_count_qc.json"
            conn_matrix = connectome_dir / f"{config.subject_id}-connectome-streamline_count.csv"
            conn_nodes = connectome_dir / f"{config.subject_id}-connectome-nodes.tsv"
            conn_outputs = {
                "qc": conn_qc,
                "manifest": connectome_dir / f"{config.subject_id}-connectome-streamline_count_manifest.json",
                "matrix": conn_matrix,
                "nodes": conn_nodes,
                "output_dir": connectome_dir,
                "sift2_output_dir": sift2_connectome_dir,
            }
            parcellation_ready = parcellation is not None or available.get("parcellation", False)
            tractogram_ready = (available.get('tractography', False) if primary_tractography_backend == 'ukf'
                                else tractogram.is_file() or available.get("tractography", False))
            if config.resume and ukf_cached and _connectome_complete(
                conn_qc,
                conn_matrix,
                conn_nodes,
                manifest_path=conn_outputs["manifest"],
                tractography=tractogram,
                parcellation=parcellation,
            ):
                add(target, "skipped", "completed outputs found", "reuse", outputs=conn_outputs)
            elif not parcellation_ready:
                add(target, "blocked", _upstream_reason(selected, "parcellation", "missing parcellation; select parcellation or provide existing output"), "upstream", outputs=conn_outputs, dependencies={"parcellation": "missing"}, provides=False)
            elif not tractogram_ready:
                add(target, "blocked", _upstream_reason(selected, "tractography", "missing tractogram; select tractography or provide existing output"), "upstream", outputs=conn_outputs, dependencies={"tractogram": "missing"}, provides=False)
            elif primary_tractography_backend != "csd_act" and config.connectome_backend == "mrtrix":
                add(target, "blocked", "UKF/DTI tractography currently supports python connectome backend only", "configuration", outputs=conn_outputs, provides=False)
            else:
                add(
                    target,
                    "will_run",
                    "ready to run connectome",
                    "runnable",
                    inputs={"tractogram": tractogram, "parcellation": parcellation, "sift2_weights": sift2_weights if primary_tractography_backend == 'csd_act' and sift2_weights.is_file() else None},
                    outputs=conn_outputs,
                    dependencies={"parcellation": "available" if parcellation is not None else "will_run", "tractogram": "available" if tractogram.is_file() else "will_run"},
                )
            continue

    plan = RunPlanResult(
        subject_id=config.subject_id,
        subject_root=subject_root,
        output_root=layout.outputs,
        run_name=run_name,
        targets=targets,
        run_plan_path=_run_plan_path(layout, run_name),
        items=tuple(items),
    )
    if write:
        _write_run_plan(plan)
    return plan


def _safe_task_dir_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name).strip("_") or "task"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _archive_existing_run_records(layout, run_name: str) -> dict[str, Path]:
    """Preserve prior run-level provenance before reusing a run name."""

    sources = {
        "orchestration_manifest": layout.run_metadata / f"{run_name}_orchestration_manifest.json",
        "run_plan": _run_plan_path(layout, run_name),
    }
    existing = {key: path for key, path in sources.items() if path.is_file()}
    if not existing:
        return {}

    revision = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    history_dir = layout.run_metadata / "history" / run_name / revision
    history_dir.mkdir(parents=True, exist_ok=False)
    archived: dict[str, Path] = {}
    for key, source in existing.items():
        destination = history_dir / source.name
        shutil.copy2(source, destination)
        archived[key] = destination
    return archived


class _TimedTaskList(list):
    def __init__(
        self,
        start_monotonic: float,
        on_append: Optional[Callable[[list[ProductTaskRecord]], None]] = None,
    ) -> None:
        super().__init__()
        self._last_monotonic = start_monotonic
        self._last_wall = datetime.now(timezone.utc)
        self._on_append = on_append

    def append(self, task: ProductTaskRecord) -> None:  # type: ignore[override]
        now = time.monotonic()
        now_wall = datetime.now(timezone.utc)
        runtime = max(0.0, now - self._last_monotonic)
        ended_at = now_wall.isoformat(timespec="seconds")
        started_at = task.started_at or self._last_wall.isoformat(timespec="seconds")
        timed_task = replace(
            task,
            started_at=started_at,
            ended_at=task.ended_at or ended_at,
            runtime_seconds=task.runtime_seconds if task.runtime_seconds is not None else runtime,
            logs=task.logs
            or (
                f"{task.name} {task.status}" + (f": {task.reason}" if task.reason else ""),
            ),
        )
        super().append(timed_task)
        self._last_monotonic = now
        self._last_wall = now_wall
        if self._on_append is not None:
            self._on_append(self)


def _task_provenance(plan: RunPlanResult, task_name: str) -> dict[str, object]:
    """C3: ``requested`` / ``required_by`` for a manifest task entry."""

    item = next((item for item in plan.items if item.task_name == task_name), None)
    return {"requested": item.requested if item else True, "required_by": list(item.required_by) if item else []}


def _write_orchestration_checkpoint(
    *,
    path: Path,
    config: ProductOrchestratorConfig,
    plan: RunPlanResult,
    targets: tuple[str, ...],
    tasks: list[ProductTaskRecord],
    previous_run_records: dict[str, Path],
) -> None:
    payload = {
        "kind": "slicerddsurfer_product_orchestration",
        "version": 1,
        "status": "running",
        "subject_id": config.subject_id,
        "subject_root": config.subject_root.expanduser().resolve().as_posix(),
        "output_root": plan.output_root.as_posix(),
        "run_name": plan.run_name,
        "targets": list(targets),
        "execute": bool(config.execute),
        "run_plan_path": plan.run_plan_path.as_posix(),
        "bundle_tractography_backend": config.bundle_tractography_backend,
        "preflight_blocked_by_backend": config.preflight_blocked_by_backend,
        "previous_run_records": {key: value.as_posix() for key, value in previous_run_records.items()},
        "updated_at": _now_utc(),
        "tasks": [
            {
                "name": task.name,
                "status": task.status,
                "reason": task.reason,
                "manifest_path": task.manifest_path.as_posix() if task.manifest_path else None,
                "qc_path": task.qc_path.as_posix() if task.qc_path else None,
                "outputs": task.outputs,
                "started_at": task.started_at,
                "ended_at": task.ended_at,
                "runtime_seconds": task.runtime_seconds,
                **_task_provenance(plan, task.name),
                **({"regional_statistics": task.regional_statistics} if task.regional_statistics else {}),
            }
            for task in tasks
        ],
        "closure": {"required_by": {key: list(value) for key, value in config.target_required_by.items()}},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mark_orchestration_failed(config: ProductOrchestratorConfig, error: BaseException) -> Path:
    subject_root = config.subject_root.expanduser().resolve()
    output_root = config.output_root.expanduser().resolve() if config.output_root is not None else None
    layout = subject_layout(subject_root, output_root=output_root)
    run_name = normalize_run_name(config.run_name)
    path = layout.run_metadata / f"{run_name}_orchestration_manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        payload = {
            "kind": "slicerddsurfer_product_orchestration",
            "version": 1,
            "subject_id": config.subject_id,
            "subject_root": subject_root.as_posix(),
            "output_root": layout.outputs.as_posix(),
            "run_name": run_name,
            "targets": list(_normalize_targets(config)),
            "execute": bool(config.execute),
            "tasks": [],
        }
    payload.update(
        {
            "status": "failed",
            "ended_at": _now_utc(),
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_normalized_task_records(
    *,
    layout,
    run_name: str,
    config: ProductOrchestratorConfig,
    tasks: list[ProductTaskRecord],
    plan: RunPlanResult,
    orchestration_manifest_path: Path,
) -> dict[str, dict[str, str]]:
    """Write one stable manifest/qc/logs directory per task.

    Product-specific pipelines keep their native manifest/QC paths. This layer
    gives the Slicer UI, CLI, and regression scripts one schema to inspect.
    """

    task_root = layout.run_metadata / "tasks" / run_name
    task_root.mkdir(parents=True, exist_ok=True)
    plan_by_task = {item.task_name: item for item in plan.items}
    normalized: dict[str, dict[str, str]] = {}
    for task in tasks:
        task_dir = task_root / _safe_task_dir_name(task.name)
        logs_dir = task_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        plan_item = plan_by_task.get(task.name)
        reused = task.status == "skipped" or bool(plan_item.reused if plan_item else False)
        normalized_manifest_path = task_dir / "manifest.json"
        normalized_qc_path = task_dir / "qc.json"
        log_path = logs_dir / "task.log"
        record = {
            "kind": "slicerddsurfer_task_manifest",
            "version": 1,
            "schema": "slicerddsurfer.task.v1",
            "subject_id": config.subject_id,
            "subject_root": config.subject_root.expanduser().resolve().as_posix(),
            "output_root": layout.outputs.as_posix(),
            "run_name": run_name,
            "target": plan_item.target if plan_item else task.name.split(".", 1)[0],
            "task_name": task.name,
            "status": task.status,
            "reason": task.reason,
            "reason_category": plan_item.reason_category if plan_item else "",
            "requested": plan_item.requested if plan_item else True,
            "required_by": list(plan_item.required_by) if plan_item else [],
            "reused": reused,
            "started_at": task.started_at,
            "ended_at": task.ended_at,
            "runtime_seconds": task.runtime_seconds,
            **({"regional_statistics": task.regional_statistics} if task.regional_statistics else {}),
            "created_at": _now_utc(),
            "inputs": plan_item.inputs if plan_item else {},
            "outputs": task.outputs or (plan_item.outputs if plan_item else {}),
            "dependencies": plan_item.dependencies if plan_item else {},
            "native_manifest_path": task.manifest_path.as_posix() if task.manifest_path else None,
            "native_qc_path": task.qc_path.as_posix() if task.qc_path else None,
            "orchestration_manifest_path": orchestration_manifest_path.as_posix(),
            "run_plan_path": plan.run_plan_path.as_posix(),
            "qc_path": normalized_qc_path.as_posix(),
            "logs_dir": logs_dir.as_posix(),
        }
        qc = {
            "kind": "slicerddsurfer_task_qc",
            "version": 1,
            "schema": "slicerddsurfer.task_qc.v1",
            "subject_id": config.subject_id,
            "run_name": run_name,
            "task_name": task.name,
            "status": task.status,
            "reason": task.reason,
            "reason_category": plan_item.reason_category if plan_item else "",
            "reused": reused,
            "started_at": task.started_at,
            "ended_at": task.ended_at,
            "runtime_seconds": task.runtime_seconds,
            "checks": {
                "normalized_manifest_written": True,
                "native_manifest_exists": bool(task.manifest_path and task.manifest_path.is_file()),
                "native_qc_exists": bool(task.qc_path and task.qc_path.is_file()),
                "declared_output_count": len(task.outputs),
            },
            "created_at": record["created_at"],
        }
        normalized_manifest_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        normalized_qc_path.write_text(json.dumps(qc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        log_path.write_text(
            f"{record['created_at']} {task.name} {task.status}"
            + (f" ({task.reason})" if task.reason else "")
            + "\n"
            + "\n".join(task.logs)
            + ("\n" if task.logs else ""),
            encoding="utf-8",
        )
        normalized[task.name] = {
            "manifest_path": normalized_manifest_path.as_posix(),
            "qc_path": normalized_qc_path.as_posix(),
            "logs_dir": logs_dir.as_posix(),
        }
    return normalized


def _expected_output_top_dirs(targets: tuple[str, ...]) -> set[str]:
    expected = {"00_run_metadata"}
    selected = set(targets)
    if "preprocessing" in selected:
        expected.add("01_preproc")
    if selected & {"microstructure", "microstructure_freewater", "microstructure_noddi", "microstructure_dki"}:
        expected.update({"02_microstructure", "05_statistics"})
    if selected & {"segmentation", "parcellation", "surface", "surface_postprocess"}:
        expected.add(ANATOMY_OUTPUT_NAME)
    if selected & {"tractography", "bundles", "saf", "wma", "tractcloud", "connectome"}:
        expected.add("04_tractography")
    expected.add("06_slicer_scene_report")
    return expected


def _audit_output_tree(layout, *, subject_id: str, selected_targets: tuple[str, ...] = ()) -> dict[str, object]:
    expected_top_dirs = _expected_output_top_dirs(selected_targets)
    optional_missing = {"06_slicer_scene_report"}
    warnings: list[str] = []
    outputs = layout.outputs
    present_top_dirs = sorted(path.name for path in outputs.iterdir() if path.is_dir()) if outputs.is_dir() else []
    missing = sorted(name for name in expected_top_dirs - optional_missing if not (
        (outputs / name).exists() or
        (name == ANATOMY_OUTPUT_NAME and (outputs / LEGACY_ANATOMY_OUTPUT_NAME).is_dir())
    ))
    for name in missing:
        warnings.append(f"missing canonical output directory: {name}")
    nested_outputs = outputs / subject_id / "outputs"
    if nested_outputs.exists():
        warnings.append(f"unexpected nested subject outputs: {nested_outputs.as_posix()}")
    legacy_names = {
        "microstructure",
        "ddseg",
        "ddparcel",
        "ddsurfer",
        "tractography",
        "connectome",
        "validation",
    }
    legacy_present = sorted(name for name in legacy_names if (outputs / name).exists())
    for name in legacy_present:
        warnings.append(f"legacy top-level output should be moved under canonical tree or cache: {name}")
    audit = {
        "kind": "slicerddsurfer_output_tree_audit",
        "version": 1,
        "schema": "slicerddsurfer.output_tree_audit.v1",
        "subject_id": subject_id,
        "output_root": outputs.as_posix(),
        "present_top_dirs": present_top_dirs,
        "expected_top_dirs": sorted(expected_top_dirs),
        "warnings": warnings,
        "status": "warning" if warnings else "passed",
        "created_at": _now_utc(),
    }
    audit_path = layout.run_metadata / "output_tree_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit["path"] = audit_path.as_posix()
    return audit


def _write_output_tree_semantics(
    *,
    layout,
    subject_id: str,
    requested_targets: tuple[str, ...],
    run_targets: tuple[str, ...],
    tasks: list[ProductTaskRecord],
) -> dict[str, object]:
    """Write user-facing reasons for canonical output directory state."""

    task_status_by_target = {
        task.name.split(".", 1)[0]: {
            "task_name": task.name,
            "status": task.status,
            "reason": task.reason,
        }
        for task in tasks
    }
    requested = set(requested_targets)
    runnable = set(run_targets)
    entries: list[dict[str, Any]] = []
    directory_targets: list[tuple[str, tuple[str, ...], str]] = [
        ("00_run_metadata", ("validation",), "Run metadata, validation, environment and task records."),
        ("01_preproc", ("preprocessing",), "DWI preprocessing outputs."),
        ("02_microstructure", ("microstructure", "microstructure_freewater", "microstructure_noddi", "microstructure_dki"), "Microstructure model outputs."),
        (ANATOMY_OUTPUT_NAME, ("segmentation", "parcellation", "surface", "surface_postprocess"), "Segmentation, parcellation and surface products."),
        ("04_tractography", ("tractography", "bundles", "saf", "wma", "tractcloud", "connectome"), "Tractography, bundle extraction, SAF and connectome products."),
        ("05_statistics", ("microstructure", "microstructure_freewater", "microstructure_noddi", "microstructure_dki", "surface_postprocess"), "Regional summaries and post-process statistics."),
        ("06_slicer_scene_report", tuple(TARGET_ORDER), "HTML/Markdown reports and output tree views."),
    ]
    if (layout.outputs / LEGACY_ANATOMY_OUTPUT_NAME).is_dir():
        directory_targets.append((LEGACY_ANATOMY_OUTPUT_NAME,
                                  ("segmentation", "parcellation", "surface", "surface_postprocess"),
                                  "Historical anatomy outputs retained in place for compatible reading."))
    for relative, targets, description in directory_targets:
        path = layout.outputs / relative
        selected_here = bool(requested & set(targets)) or relative == "00_run_metadata"
        runnable_here = bool(runnable & set(targets)) or relative == "00_run_metadata"
        file_count, size_bytes = _path_stats(path)
        if not selected_here:
            state = "not_selected_existing" if path.exists() else "not_selected_absent"
        elif not runnable_here:
            state = "blocked_existing" if path.exists() else "blocked_absent"
        elif path.exists() and file_count:
            state = "selected_has_outputs"
        elif path.exists():
            state = "selected_empty"
        else:
            state = "selected_not_created_yet"
        related_tasks = [
            task_status_by_target[target]
            for target in targets
            if target in task_status_by_target
        ]
        entries.append(
            {
                "path": relative,
                "absolute_path": path.as_posix(),
                "state": state,
                "selected": selected_here,
                "runnable": runnable_here,
                "exists": path.exists(),
                "file_count": file_count,
                "size_bytes": size_bytes,
                "related_targets": list(targets),
                "related_tasks": related_tasks,
                "description": description,
            }
        )
    payload = {
        "kind": "slicerddsurfer_output_tree_semantics",
        "version": 1,
        "schema": "slicerddsurfer.output_tree_semantics.v1",
        "subject_id": subject_id,
        "output_root": layout.outputs.as_posix(),
        "requested_targets": list(requested_targets),
        "run_targets": list(run_targets),
        "entries": entries,
        "created_at": _now_utc(),
    }
    path = layout.run_metadata / "output_tree_semantics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload["path"] = path.as_posix()
    return payload


def _path_stats(path: Path) -> tuple[int, int]:
    """Return (file_count, size_bytes) for an existing file or directory."""

    if not path.exists():
        return 0, 0
    if path.is_file():
        try:
            return 1, path.stat().st_size
        except OSError:
            return 1, 0
    if not path.is_dir():
        return 0, 0
    file_count = 0
    size_bytes = 0
    for child in path.rglob("*"):
        if not child.is_file():
            continue
        file_count += 1
        try:
            size_bytes += child.stat().st_size
        except OSError:
            pass
    return file_count, size_bytes


def _path_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _declared_task_output_stats(task: ProductTaskRecord, *, scan_records=None) -> tuple[int, int]:
    candidates: list[Path] = []
    for value in (task.outputs or {}).values():
        if not value:
            continue
        path = Path(str(value)).expanduser()
        if not path.exists():
            continue
        try:
            normalized_path = path.resolve()
        except OSError:
            normalized_path = path
        candidates.append(normalized_path)

    roots: list[Path] = []
    for path in candidates:
        if any(path == existing or _path_within(path, existing) for existing in roots):
            continue
        if path.is_dir():
            roots = [existing for existing in roots if not _path_within(existing, path)]
        roots.append(path)

    total_files = 0
    total_bytes = 0
    for path in roots:
        started = time.monotonic()
        file_count, size_bytes = _path_stats(path)
        if scan_records is not None:
            scan_records.append(dict(path=path.as_posix(), file_count=file_count, size_bytes=size_bytes,
                                     seconds=time.monotonic() - started))
        total_files += file_count
        total_bytes += size_bytes
    return total_files, total_bytes


def _top_dir_stats(output_root: Path) -> list[dict[str, object]]:
    stats: dict[str, dict[str, object]] = {}
    if not output_root.is_dir():
        return []
    for path in output_root.rglob("*"):
        if not path.is_file():
            continue
        try:
            top_name = path.relative_to(output_root).parts[0]
        except (IndexError, ValueError):
            continue
        row = stats.setdefault(top_name, {"path": top_name, "file_count": 0, "size_bytes": 0})
        row["file_count"] = int(row["file_count"]) + 1
        try:
            row["size_bytes"] = int(row["size_bytes"]) + path.stat().st_size
        except OSError:
            pass
    return [stats[name] for name in sorted(stats)]


def _build_performance_profile(
    *,
    layout,
    run_name: str,
    tasks: list[ProductTaskRecord],
    orchestration_elapsed_seconds: float,
) -> dict[str, object]:
    task_rows = []
    scan_records = []
    for task in tasks:
        output_files, output_bytes = _declared_task_output_stats(task, scan_records=scan_records)
        task_rows.append(
            {
                "name": task.name,
                "status": task.status,
                "runtime_seconds": float(task.runtime_seconds or 0.0),
                "declared_output_files": output_files,
                "declared_output_bytes": output_bytes,
                "reason": task.reason,
                "started_at": task.started_at,
                "ended_at": task.ended_at,
                "wall_time_fraction": (float(task.runtime_seconds or 0.0) / orchestration_elapsed_seconds
                                       if orchestration_elapsed_seconds > 0 else None),
            }
        )

    output_root = layout.outputs
    top_scan_started = time.monotonic()
    top_dirs = _top_dir_stats(output_root)
    top_scan_seconds = time.monotonic() - top_scan_started

    profile = {
        "kind": "slicerddsurfer_performance_profile",
        "version": 1,
        "schema": "slicerddsurfer.performance_profile.v1",
        "run_name": run_name,
        "output_root": output_root.as_posix(),
        "created_at": _now_utc(),
        "orchestration_elapsed_seconds": orchestration_elapsed_seconds,
        "task_runtime_seconds_total": sum(float(row["runtime_seconds"]) for row in task_rows),
        "task_intervals": task_rows,
        "directory_statistics": dict(declared_output_queries=scan_records, top_level_scan_seconds=top_scan_seconds,
                                     total_scan_seconds=top_scan_seconds + sum(row['seconds'] for row in scan_records)),
        "profile_notes": [
            "Task runtime_seconds use explicit subprocess intervals when available; otherwise they measure elapsed time between orchestrator task appends.",
            "Skipped task runtimes measure resume detection overhead, not the original compute runtime.",
            "Declared output sizes are computed from task output paths that exist when the manifest is written.",
            "Concurrent task intervals overlap: summed task seconds/fractions are not workflow wall time or speedup. Compare fresh matched complete runs, not sums of historical tasks.",
        ],
        "slowest_tasks": sorted(task_rows, key=lambda row: float(row["runtime_seconds"]), reverse=True)[:8],
        "largest_task_outputs": sorted(task_rows, key=lambda row: int(row["declared_output_bytes"]), reverse=True)[:8],
        "largest_top_dirs": sorted(top_dirs, key=lambda row: int(row["size_bytes"]), reverse=True)[:8],
        "top_dirs": top_dirs,
    }
    profile_path = layout.run_metadata / f"{run_name}_performance_profile.json"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile["path"] = profile_path.as_posix()
    profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return profile


def _parallel_model_products(config, targets, parcellation, log):
    _emit('[Orchestrator] Running independent microstructure model processes',log)
    from microstructure import MicrostructureConfig, run_microstructure
    from shared.pipeline import find_subject_root, resolve_preprocessed_input_bundle, ensure_native_dti_bundle
    from shared.brain_mask import brain_mask_config_from_options

    raw = find_subject_root(config.subject_root, config.subject_id,
        mrtrix_bin_dir=config.mrtrix_bin_dir or config.preprocessing_mrtrix_bin_dir,
        brain_mask_config=brain_mask_config_from_options(config))
    raw = resolve_preprocessed_input_bundle(raw, output_root=config.output_root)
    native = ensure_native_dti_bundle(raw, fit_method=config.microstructure_fit_method, reuse=True, log=log)
    layout = subject_layout(config.subject_root, output_root=config.output_root)
    result = run_microstructure(MicrostructureConfig(config.subject_id,raw,native,layout.microstructure,
        roi_label_map=parcellation,run_free_water='microstructure_freewater' in targets,
        run_noddi='microstructure_noddi' in targets,run_dki='microstructure_dki' in targets,
        dki_fit_method=config.microstructure_dki_fit_method,
        dki_matlab_executable=config.microstructure_dki_matlab_executable,
        dki_mkcurve_root=config.microstructure_dki_mkcurve_root,
        dki_execution=config.microstructure_dki_execution,
        free_water_chunk_size=config.microstructure_free_water_chunk_size,
        free_water_engine=config.microstructure_free_water_engine,
        free_water_n_jobs=config.microstructure_free_water_n_jobs,
        free_water_vox_per_chunk=config.microstructure_free_water_vox_per_chunk,
        free_water_max_fraction_for_tissue=config.microstructure_free_water_max_fraction_for_tissue,
        force_recompute=not config.resume,max_parallel_models=config.max_parallel_models),log=log)
    rows = {row['name']:row for row in result.compute_schedule['jobs']}
    getters = {'freewater':_microstructure_freewater_outputs,'noddi':_microstructure_noddi_outputs}
    tasks = {}
    for stage,row in rows.items():
        if stage == 'dki':
            product = result.dki_result
            outputs = dict(product.outputs)
            outputs.update({f'mkcurve_{key}':value for key,value in product.mkcurve_outputs.items()})
            manifest,qc = product.manifest_path,product.qc_path
        else:
            outputs = getters[stage](config.subject_root,config.subject_id,config.output_root)
            manifest,qc = outputs['manifest'],outputs['qc']
        tasks['microstructure_'+stage] = ProductTaskRecord(name='microstructure.'+stage,status='completed',
            reason='Independent process; shared ROI statistics published after all model fits',
            manifest_path=manifest,qc_path=qc,outputs={key:str(value) for key,value in outputs.items()},
            started_at=row['started_at'],ended_at=row['ended_at'],runtime_seconds=row['runtime_seconds'],
            logs=(row['log'],))
    return tasks


def _run_subject_products_impl(
    config: ProductOrchestratorConfig,
    *,
    log: Optional[Callable[[str], None]] = None,
) -> ProductOrchestratorResult:
    """Run or plan selected products from a shared subject root."""

    config = _with_shared_bundle_resources(_with_closure(config))
    if not config.subject_id.strip():
        raise ValueError("subject_id is required.")
    if config.select_streamlines <= 0:
        raise ValueError("select_streamlines must be > 0.")
    if config.max_parallel_models not in (1, 2, 3):
        raise ValueError('max_parallel_models must be 1, 2 or 3')
    if config.dti_min_length_mm < 0 or config.dti_max_length_mm <= config.dti_min_length_mm:
        raise ValueError("DTI streamline lengths require 0 <= dti_min_length_mm < dti_max_length_mm.")
    if config.mrtrix_nthreads is not None and config.mrtrix_nthreads < 0:
        raise ValueError("mrtrix_nthreads must be >= 0 when set.")
    _validate_tractography_settings(config)
    _validate_surface_settings(config)
    if config.microstructure_free_water_chunk_size <= 0:
        raise ValueError("microstructure_free_water_chunk_size must be > 0.")
    if config.microstructure_free_water_engine not in {"serial", "joblib", "dask", "ray"}:
        raise ValueError("microstructure_free_water_engine must be one of: serial, joblib, dask, ray.")
    if config.microstructure_free_water_vox_per_chunk is not None and config.microstructure_free_water_vox_per_chunk <= 0:
        raise ValueError("microstructure_free_water_vox_per_chunk must be > 0.")
    if not 0.0 < float(config.microstructure_free_water_max_fraction_for_tissue) < 1.0:
        raise ValueError("microstructure_free_water_max_fraction_for_tissue must be between 0 and 1.")
    if config.tractography_backend not in TRACTOGRAPHY_BACKENDS:
        raise ValueError("tractography_backend must be one of: csd_act, ukf, dti_det, saf.")
    bundle_backends = _normalize_bundle_backends(config.bundle_backend)
    if config.connectome_backend not in {"python", "mrtrix"}:
        raise ValueError("connectome_backend must be one of: python, mrtrix.")

    targets = _normalize_targets(config)
    subject_root = config.subject_root.expanduser().resolve()
    output_root = config.output_root.expanduser().resolve() if config.output_root is not None else None
    layout = subject_layout(subject_root, output_root=output_root)
    run_name = normalize_run_name(config.run_name)
    primary_tractography_backend = _primary_tractography_backend(config)
    start = time.monotonic()
    previous_run_records = _archive_existing_run_records(layout, run_name)
    plan = plan_subject_products(config, write=True)
    planning_seconds = time.monotonic() - start
    manifest_path = layout.run_metadata / f"{run_name}_orchestration_manifest.json"
    tasks: list[ProductTaskRecord] = _TimedTaskList(
        start,
        on_append=lambda current: _write_orchestration_checkpoint(
            path=manifest_path,
            config=config,
            plan=plan,
            targets=targets,
            tasks=current,
            previous_run_records=previous_run_records,
        ),
    )
    _write_orchestration_checkpoint(
        path=manifest_path,
        config=config,
        plan=plan,
        targets=targets,
        tasks=tasks,
        previous_run_records=previous_run_records,
    )
    plan_by_target = {item.target: item for item in plan.items}
    preflight_blocked_targets = {
        target: reason
        for target, reason in (config.preflight_blocked_targets or {}).items()
        if target in targets and reason and plan_by_target.get(target) is not None and plan_by_target[target].status == "blocked"
    }
    run_targets = tuple(target for target in targets if target not in preflight_blocked_targets)

    input_started = time.monotonic()
    materialize_canonical_output_tree(subject_root, output_root=output_root, mode="symlink", archive_legacy=False)
    mask_mrtrix_bin_dir = config.mrtrix_bin_dir or config.preprocessing_mrtrix_bin_dir
    from shared.brain_mask import brain_mask_config_from_options
    selected_mask_config = brain_mask_config_from_options(config)
    initial_raw = _resolve_raw_inputs(subject_root)
    mask_failure = ""
    if (
        config.execute
        and not initial_raw["mask"].is_file()
        and all(initial_raw[key].is_file() for key in ("dwi", "bval", "bvec"))
    ):
        from shared.pipeline import find_subject_root

        try:
            find_subject_root(
                subject_root,
                config.subject_id,
                mrtrix_bin_dir=mask_mrtrix_bin_dir,
                brain_mask_config=selected_mask_config,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            mask_failure = mask_failure_message(str(exc))
            _emit(f"[Brain mask] {mask_failure}", log)
    raw = _resolve_raw_inputs(subject_root)

    for target in targets:
        if target in preflight_blocked_targets:
            tasks.append(
                ProductTaskRecord(
                    name=_task_name_for_target(target, config),
                    status="blocked",
                    reason=preflight_blocked_targets[target],
                )
            )

    dwi_targets = {'preprocessing', 'segmentation', 'parcellation', 'surface', 'microstructure',
                   'microstructure_freewater', 'microstructure_noddi', 'microstructure_dki', 'tractography'}
    if 'saf' in run_targets and config.saf_pregenerated_tractogram is None:
        dwi_targets.add('saf')
    input_validation_path = None
    input_blocked_targets = {}
    if "validation" in run_targets or (config.execute and dwi_targets.intersection(run_targets)):
        validation = validate_subject_inputs(
            InputValidationConfig(
                subject_id=config.subject_id,
                input_root=subject_root,
                output_dir=layout.run_metadata / "validation",
                dependency_groups=("core",),
                mrtrix_bin_dir=mask_mrtrix_bin_dir,
                brain_mask_backend=config.brain_mask_backend,
                brain_mask_device=config.brain_mask_device,
                pnl_mask_python=config.pnl_mask_python,
                pnl_mask_script=config.pnl_mask_script,
                pnl_mask_model_dir=config.pnl_mask_model_dir,
                pnl_mask_ants_bin_dir=config.pnl_mask_ants_bin_dir,
                synthstrip_python=config.synthstrip_python,
                postprocess_freesurfer_home=config.postprocess_freesurfer_home,
            )
        )
        input_validation_path = validation.report_path
        if 'validation' in run_targets or not validation.passed:
            tasks.append(ProductTaskRecord(
                name="validation",
                status=validation.status,
                reason=f"errors={validation.error_count}; warnings={validation.warning_count}",
                manifest_path=validation.report_path,
                qc_path=validation.report_path,
            ))
        if config.execute and not validation.passed:
            errors = [issue for issue in validation.messages if issue.level == 'error']
            invalid_data = any(issue.category in {'file', 'dwi', 'gradient', 'brain_mask'} for issue in errors)
            for target in run_targets:
                if (invalid_data and target in dwi_targets) or (
                        target == 'preprocessing' and any(issue.category == 'metadata' for issue in errors)):
                    reason = 'Input validation failed: ' + '; '.join(issue.message for issue in errors)
                    if mask_failure and any(issue.category == 'brain_mask' or 'mask' in issue.message.lower()
                                            for issue in errors):
                        reason = mask_failure + ' ' + reason
                    input_blocked_targets[target] = reason
                    tasks.append(ProductTaskRecord(name=_task_name_for_target(target, config), status='blocked',
                                                   reason=reason, qc_path=validation.report_path))
            run_targets = tuple(target for target in run_targets if target not in input_blocked_targets)

    input_seconds = time.monotonic() - input_started
    execution_started = time.monotonic()
    if "preprocessing" in run_targets:
        preprocessing_outputs = _preprocessing_outputs(subject_root, config.subject_id, output_root)
        if config.resume and _preprocessing_complete(subject_root, config.subject_id, output_root, config=config):
            tasks.append(
                ProductTaskRecord(
                    name="preprocessing.dwi",
                    status="skipped",
                    reason="completed outputs found",
                    manifest_path=preprocessing_outputs["manifest"],
                    qc_path=preprocessing_outputs["qc"],
                    outputs={key: path.as_posix() for key, path in preprocessing_outputs.items() if key not in {"manifest", "qc"}},
                )
            )
        elif not config.execute:
            tasks.append(ProductTaskRecord(name="preprocessing.dwi", status="planned", reason="execute=False"))
        elif any(not path.is_file() for path in raw.values()):
            missing = [label for label, path in raw.items() if not path.is_file()]
            tasks.append(ProductTaskRecord(name="preprocessing.dwi", status="blocked", reason="missing raw input: " + ", ".join(missing)))
        else:
            from preprocessing import run_dwi_preprocessing

            _emit("[Orchestrator] Running DWI preprocessing", log)
            result = run_dwi_preprocessing(
                _preprocessing_config(config, subject_root, layout.preproc),
                log=log,
            )
            tasks.append(
                ProductTaskRecord(
                    name="preprocessing.dwi",
                    status="completed" if result.status in {"completed", "cached"} else result.status,
                    reason="cached outputs reused" if result.status == "cached" else "",
                    manifest_path=result.manifest_path,
                    qc_path=result.qc_path,
                    outputs={
                        "dwi": result.final_dwi_path.as_posix(),
                        "bval": result.final_bval_path.as_posix(),
                        "bvec": result.final_bvec_path.as_posix(),
                        "mask": result.final_mask_path.as_posix(),
                    },
                )
            )

    preprocessing_outputs = _preprocessing_outputs(subject_root, config.subject_id, output_root)
    analysis_inputs = _resolve_analysis_inputs(subject_root, config.subject_id, output_root)
    segmentation = _resolve_segmentation(subject_root, output_root)
    parcellation = _resolve_parcellation(subject_root, config.subject_id, output_root)
    model_assets_dir = config.model_assets_dir.expanduser().resolve() if config.model_assets_dir is not None else None
    atlas_t2 = config.atlas_t2.expanduser().resolve() if config.atlas_t2 is not None else _default_atlas_t2(model_assets_dir)
    segmentation_outputs = _segmentation_outputs(subject_root, output_root)
    parcellation_outputs = _parcellation_outputs(subject_root, config.subject_id, output_root)
    surface_outputs = _surface_outputs(subject_root, config.subject_id, output_root)
    micro_outputs = _microstructure_dti_outputs(subject_root, config.subject_id, output_root)
    csd_dir = layout.csd if run_name == "csd_act" else layout.csd / run_name
    tractogram_dir = layout.tractograms / run_name
    dti_tractogram_dir = layout.tractograms_dti_det / run_name
    wmql_dir = layout.bundles_wmql / run_name
    saf_dir = _saf_output_dir(layout, run_name)
    wma_dir = _wma_output_dir(layout, run_name)
    tractcloud_dir = _tractcloud_output_dir(layout, run_name)
    connectome_dir = canonical_connectome_output_dir(subject_root, run_name, "streamline_count", output_root=output_root)
    sift2_connectome_dir = canonical_connectome_output_dir(
        subject_root,
        run_name,
        "sift2_weighted_streamline_count",
        output_root=output_root,
    )
    csd_qc = csd_dir / "csd_act_qc.json"
    dti_stem = f"{config.subject_id}-dti-deterministic"
    dti_qc = dti_tractogram_dir / f"{dti_stem}_qc.json"
    dti_manifest = dti_tractogram_dir / f"{dti_stem}_manifest.json"
    csd_tractogram = tractogram_dir / f"{config.subject_id}_csd_act_ifod2.tck"
    csd_tractogram_trk = csd_tractogram.with_suffix(".trk")
    csd_tractogram_vtk = csd_tractogram.with_suffix(".vtk")
    sift2_weights = csd_tractogram.with_name(f"{csd_tractogram.stem}_sift2_weights.txt")
    dti_tractogram_trk = dti_tractogram_dir / f"{dti_stem}.trk"
    tractogram = csd_tractogram if primary_tractography_backend == "csd_act" else dti_tractogram_trk
    tractogram_trk = csd_tractogram_trk if primary_tractography_backend == "csd_act" else dti_tractogram_trk
    from tractography.ukf import paths as ukf_paths
    ukf_dir = layout.tractograms / 'ukf' / run_name
    ukf_outputs = ukf_paths(config.subject_id, ukf_dir)
    if primary_tractography_backend == 'ukf':
        tractogram, tractogram_trk = ukf_outputs['tractogram'], ukf_outputs['tractogram_trk']
    current_vtk = (tractogram if primary_tractography_backend == 'ukf' else
                   dti_tractogram_trk.with_suffix('.vtk') if primary_tractography_backend == 'dti_det' else
                   csd_tractogram_vtk)
    saf_tractogram = saf_dir / "SAF.tck"
    if config.wma_input_tractography is not None:
        wma_input_tractography = _resolve_wma_input_tractography(
            subject_root,
            config.wma_input_tractography,
            output_root=output_root,
        )
    else:
        wma_input_tractography = _resolve_wma_input_tractography(
            subject_root,
            None,
            output_root=output_root,
        )
    if primary_tractography_backend == 'ukf' and config.wma_input_tractography is None:
        wma_input_tractography = tractogram
    elif config.bundle_tractography_backend:
        wma_input_tractography = current_vtk
    tractcloud_input_tractography = _resolve_bundle_input_tractography(
        subject_root,
        config.tractcloud_input_tractography,
        output_root=output_root,
        run_name=run_name,
        current_run_tractography=current_vtk,
        require_current_run=True,
        allow_planned_current_run=True,
    )
    slicer_executable = _resolve_slicer_executable(config)
    ddseg_device = config.ddseg_device
    if ddseg_device == "auto" and (not config.execute or not {"segmentation", "parcellation"} & set(run_targets)):
        ddseg_device = "not_resolved_execute_false" if {"segmentation", "parcellation"} & set(run_targets) else "not_resolved_target_not_run"
    surface_device = config.surface_device
    if surface_device == "auto" and (not config.execute or "surface" not in run_targets):
        surface_device = "not_resolved_execute_false" if "surface" in run_targets else "not_resolved_target_not_run"
    tractcloud_device = _resolved_tractcloud_device_for_execution(
        config.tractcloud_device,
        execute=config.execute,
        target_will_run="tractcloud" in run_targets,
    )

    if "segmentation" in run_targets:
        if config.resume and _segmentation_complete(subject_root, output_root):
            tasks.append(
                ProductTaskRecord(
                    name="segmentation.ddseg",
                    status="skipped",
                    reason="completed outputs found",
                    outputs={key: path.as_posix() for key, path in segmentation_outputs.items()},
                )
            )
        elif not config.execute:
            tasks.append(ProductTaskRecord(name="segmentation.ddseg", status="planned", reason="execute=False"))
        elif any(not path.is_file() for path in raw.values()):
            missing = [label for label, path in raw.items() if not path.is_file()]
            tasks.append(ProductTaskRecord(name="segmentation.ddseg", status="blocked", reason="missing raw input: " + ", ".join(missing)))
        else:
            from anatomy.segmentation.config import DDSegConfig
            from anatomy.segmentation.pipeline import run_ddseg
            from shared.pipeline import ensure_native_dti_bundle

            _emit("[Orchestrator] Running DDSeg", log)
            task_started = _start_task_timer("DDSeg", log)
            raw_bundle = _anatomy_input_bundle(subject_root, config.subject_id, output_root, mask_mrtrix_bin_dir, selected_mask_config)
            _emit(f"[Orchestrator] DDSeg input DWI: {raw_bundle.dwi_path}", log)
            native_bundle = ensure_native_dti_bundle(raw_bundle, fit_method="ls", reuse=True, log=log)
            run_ddseg(
                DDSegConfig(
                    input_mask_nii=raw_bundle.mask_path,
                    parameter_type="DTI",
                    output_folder=layout.ddseg,
                    weights_dir=_default_ddseg_weights(model_assets_dir),
                    dti_feature_dir=native_bundle.bundle_dir,
                    device=ddseg_device,
                    source_dwi=raw_bundle.dwi_path,
                )
            )
            _write_anatomy_source(layout.ddseg / 'ddseg_source.json', raw_bundle,
                                  _read_json(native_bundle.manifest_path).get('source_files'))
            _finish_task_timer("DDSeg", task_started, log)
            segmentation_outputs = _segmentation_outputs(subject_root, output_root)
            tasks.append(
                ProductTaskRecord(
                    name="segmentation.ddseg",
                    status="completed",
                    outputs={key: path.as_posix() for key, path in segmentation_outputs.items()},
                )
            )
            segmentation = _resolve_segmentation(subject_root, output_root)

    if "parcellation" in run_targets:
        if config.resume and _parcellation_complete(subject_root, config.subject_id, output_root):
            tasks.append(
                ProductTaskRecord(
                    name="parcellation.ddparcel",
                    status="skipped",
                    reason="completed outputs found",
                    qc_path=parcellation_outputs["wmparc"].parent / "ddparcel_qc.json",
                    outputs={key: path.as_posix() for key, path in parcellation_outputs.items()},
                )
            )
        elif not config.execute:
            tasks.append(ProductTaskRecord(name="parcellation.ddparcel", status="planned", reason="execute=False"))
        elif not atlas_t2.is_file():
            tasks.append(ProductTaskRecord(name="parcellation.ddparcel", status="blocked", reason=f"missing atlas T2: {atlas_t2}"))
        elif any(not path.is_file() for path in raw.values()):
            missing = [label for label, path in raw.items() if not path.is_file()]
            tasks.append(ProductTaskRecord(name="parcellation.ddparcel", status="blocked", reason="missing raw input: " + ", ".join(missing)))
        else:
            from anatomy.parcellation import DDParcelRegisteredConfig, run_ddparcel_from_registered_bundle
            from shared.pipeline import ensure_registered_dti_bundle

            _emit("[Orchestrator] Running DDParcel", log)
            task_started = _start_task_timer("DDParcel", log)
            raw_bundle = _anatomy_input_bundle(subject_root, config.subject_id, output_root, mask_mrtrix_bin_dir, selected_mask_config)
            _emit(f"[Orchestrator] DDParcel input DWI: {raw_bundle.dwi_path}", log)
            registered = ensure_registered_dti_bundle(
                raw_bundle,
                atlas_t2=atlas_t2,
                fit_method="wls",
                reuse=True,
                log=log,
            )
            run_ddparcel_from_registered_bundle(
                DDParcelRegisteredConfig(
                    subject_id=config.subject_id,
                    registered_bundle=registered,
                    output_dir=layout.ddparcel,
                    native_mask=raw_bundle.mask_path,
                    weights_dir=model_asset_path(
                        "parcellation/weights",
                        repo_root=_repo_root(),
                        explicit_root=model_assets_dir,
                    ),
                    device=ddseg_device,
                    force_recompute=not config.resume,
                ),
                log=log,
            )
            _write_anatomy_source(layout.ddparcel / "ddparcel_source.json", raw_bundle,
                _read_json(registered.manifest_path).get('native_bundle', {}).get('manifest', {}).get('source_files'))
            _finish_task_timer("DDParcel", task_started, log)
            parcellation_outputs = _parcellation_outputs(subject_root, config.subject_id, output_root)
            tasks.append(
                ProductTaskRecord(
                    name="parcellation.ddparcel",
                    status="completed",
                    qc_path=layout.ddparcel / "ddparcel_qc.json",
                    outputs={key: path.as_posix() for key, path in parcellation_outputs.items()},
                )
            )
            parcellation = _resolve_parcellation(subject_root, config.subject_id, output_root)

    ddseg_world_qc_state: dict[str, Optional[str]] = {}

    def ddseg_world_qc_gate() -> Optional[str]:
        """Run the DDSeg/DDParcel world-space gate once per invocation; return a failure reason."""

        if "failure" not in ddseg_world_qc_state:
            world_qc_path = layout.ddseg / "ddseg_world_qc.json"
            if segmentation is not None and segmentation.parent != layout.ddseg:
                world_qc_path = layout.run_metadata / "anatomy_world_qc" / "ddseg_world_qc.json"
            ddseg_world_qc_state["failure"] = _run_ddseg_world_qc_gate(
                config,
                segmentation,
                parcellation,
                world_qc_path,
                tasks,
                log,
            )
        return ddseg_world_qc_state["failure"]

    if any(
        task.name in {"segmentation.ddseg", "parcellation.ddparcel"} and task.status == "completed" for task in tasks
    ):
        ddseg_world_qc_gate()

    if "surface" in run_targets:
        if config.resume and _surface_complete(subject_root, config.subject_id, output_root, config=config):
            tasks.append(
                ProductTaskRecord(
                    name="surface.ddsurfer",
                    status="skipped",
                    reason="completed outputs found",
                    outputs={key: path.as_posix() for key, path in surface_outputs.items()},
                )
            )
        elif not config.execute:
            tasks.append(ProductTaskRecord(name="surface.ddsurfer", status="planned", reason="execute=False"))
        else:
            from anatomy.surface import SurfaceConfig, inspect_surface_resources, run_surface

            surface_config = SurfaceConfig(
                subject_id=config.subject_id,
                subject_root=subject_root,
                output_dir=layout.ddsurfer,
                predict_mode="all",
                device=surface_device,
                model_assets_dir=model_assets_dir,
                registration_atlas=_resolve_optional_path(config.surface_atlas_t2),
                python_bin=_optional_interpreter(config.surface_python_bin),
                precision=config.surface_precision,
                cpu_threads=config.surface_cpu_threads,
                max_gpu_gib=config.surface_max_gpu_gib,
                resume=config.resume,
                reuse_shared_preprocessing=True,
                execute=True,
                input_bundle=(
                    _anatomy_input_bundle(subject_root, config.subject_id, output_root, mask_mrtrix_bin_dir, selected_mask_config)
                    if not any(not path.is_file() for path in raw.values())
                    else None
                ),
            )
            resources = inspect_surface_resources(surface_config)
            if not resources["available"]:
                missing = resources.get("missing", [])
                tasks.append(
                    ProductTaskRecord(
                        name="surface.ddsurfer",
                        status="blocked",
                        reason="missing DDSurfer resources: " + ", ".join(str(item) for item in missing),
                    )
                )
            elif any(not path.is_file() for path in raw.values()):
                missing = [label for label, path in raw.items() if not path.is_file()]
                tasks.append(ProductTaskRecord(name="surface.ddsurfer", status="blocked", reason="missing raw input: " + ", ".join(missing)))
            else:
                _emit("[Orchestrator] Running DDSurfer surface", log)
                task_started = _start_task_timer("DDSurfer surface", log)
                from anatomy.surface.pipeline import (
                    prepare_shared_registered_bundle, record_surface_failure, uses_shared_raw_route)
                from shared.cohort_schedule import cohort_active

                # S1 (cohort mode only): native_dti and registered_dti get their own admissions, outside the surface one.
                # Outside cohort mode the preparation stays inside run_surface (the 121345b path).
                if cohort_active() and uses_shared_raw_route(surface_config):
                    prepare_started = time.monotonic()
                    try:
                        prepare_shared_registered_bundle(surface_config, resources, log)
                    except Exception as exc:  # R1: same failed records as run_surface
                        raise record_surface_failure(surface_config, resources, exc, prepare_started) from exc
                result = run_surface(surface_config, log=log)
                _finish_task_timer("DDSurfer surface", task_started, log)
                tasks.append(
                    ProductTaskRecord(
                        name="surface.ddsurfer",
                        status="completed",
                        manifest_path=result.manifest_path,
                        qc_path=result.qc_path,
                        outputs={f"model_{index}": path.as_posix() for index, path in enumerate(result.generated_models, start=1)},
                    )
                )

    if "surface_postprocess" in run_targets:
        postprocess_outputs = _surface_postprocess_outputs(subject_root, config.subject_id, output_root)
        if config.resume and not config.postprocess_force and _surface_postprocess_complete(
            subject_root,
            config.subject_id,
            output_root,
            mode=config.postprocess_mode,
            native_dir=_resolve_optional_path(config.postprocess_native_dir),
            atlases=config.postprocess_atlases,
            runtime=_resolve_optional_path(config.postprocess_runtime),
            freesurfer_home=_resolve_optional_path(config.postprocess_freesurfer_home),
        ):
            tasks.append(
                ProductTaskRecord(
                    name="surface.ddsurfer_postprocess",
                    status="skipped",
                    reason="completed outputs found",
                    manifest_path=postprocess_outputs["manifest"],
                    qc_path=postprocess_outputs["qc"],
                    outputs={key: path.as_posix() for key, path in postprocess_outputs.items()},
                )
            )
        elif not config.execute:
            tasks.append(ProductTaskRecord(name="surface.ddsurfer_postprocess", status="planned", reason="execute=False"))
        else:
            from anatomy.surface import DDSurferPostprocessConfig, run_ddsurfer_postprocess

            explicit_native_dir = _resolve_optional_path(config.postprocess_native_dir)
            explicit_mri_dir = _resolve_optional_path(config.postprocess_mri_dir)
            native_dir = explicit_native_dir or _surface_postprocess_native_dir(
                subject_root,
                config.subject_id,
                output_root,
            )
            mri_dir = explicit_mri_dir or layout.anatomy_input_dir("ddparcel")
            if explicit_native_dir is not None and not explicit_native_dir.is_dir():
                tasks.append(
                    ProductTaskRecord(
                        name="surface.ddsurfer_postprocess",
                        status="blocked",
                        reason="invalid explicit postprocess inputs: postprocess_native_dir",
                    )
                )
            elif explicit_mri_dir is not None and not explicit_mri_dir.is_dir():
                tasks.append(
                    ProductTaskRecord(
                        name="surface.ddsurfer_postprocess",
                        status="blocked",
                        reason="invalid explicit postprocess inputs: postprocess_mri_dir",
                    )
                )
            elif not _surface_postprocess_native_ready(native_dir, config.subject_id, config.postprocess_mode):
                tasks.append(
                    ProductTaskRecord(
                        name="surface.ddsurfer_postprocess",
                        status="blocked",
                        reason="missing DDSurfer *_orig.stl inputs; run Surface or provide postprocess_native_dir",
                    )
                )
            elif (config.postprocess_backend_root is not None or
                  not any((native_dir / f'{hemi}.white.obj').is_file() for hemi in ('lh', 'rh'))) and _surface_postprocess_wmparc_path(config.subject_id, mri_dir) is None:
                tasks.append(
                    ProductTaskRecord(
                        name="surface.ddsurfer_postprocess",
                        status="blocked",
                        reason="missing DDSurfer wmparc input; run Parcellation or provide postprocess_mri_dir",
                    )
                )
            else:
                _emit("[Orchestrator] Running DDSurfer postprocess", log)
                result = run_ddsurfer_postprocess(
                    DDSurferPostprocessConfig(
                        subject_id=config.subject_id,
                        subject_root=subject_root,
                        native_dir=native_dir,
                        mri_dir=mri_dir,
                        output_root=subject_layout(subject_root, output_root=output_root).outputs,
                        backend_root=_resolve_optional_path(config.postprocess_backend_root),
                        mode=config.postprocess_mode,
                        python_bin=_optional_interpreter(config.postprocess_python_bin),
                        freesurfer_home=_resolve_optional_path(config.postprocess_freesurfer_home),
                        runtime=_resolve_optional_path(config.postprocess_runtime),
                        license_file=_resolve_optional_path(config.postprocess_license_file),
                        atlases=config.postprocess_atlases,
                        force=bool(config.postprocess_force),
                    ),
                    log=log,
                )
                postprocess_complete = _surface_postprocess_complete(
                    subject_root,
                    config.subject_id,
                    output_root,
                    mode=config.postprocess_mode,
                    native_dir=native_dir,
                    atlases=config.postprocess_atlases,
                    runtime=_resolve_optional_path(config.postprocess_runtime),
                    freesurfer_home=_resolve_optional_path(config.postprocess_freesurfer_home),
                )
                tasks.append(
                    ProductTaskRecord(
                        name="surface.ddsurfer_postprocess",
                        status="completed" if postprocess_complete else "failed",
                        reason="" if postprocess_complete else "DDSurfer postprocess did not produce all required non-empty MRI/surface outputs",
                        manifest_path=result.manifest_path,
                        qc_path=result.qc_path,
                        outputs={
                            "structural_root": result.structural_root.as_posix(),
                            "stats_root": result.stats_root.as_posix(),
                            "staged_input_dir": result.staged_input_dir.as_posix(),
                        },
                    )
                )

    if "microstructure" in run_targets:
        micro_outputs = _microstructure_dti_outputs(subject_root, config.subject_id, output_root)
        freewater_outputs = _microstructure_freewater_outputs(subject_root, config.subject_id, output_root)
        noddi_outputs = _microstructure_noddi_outputs(subject_root, config.subject_id, output_root)
        # K6: when a parcellation is available to this run, outputs without ROI statistics are not complete.
        roi_statistics_missing = parcellation is not None and not (
            layout.regional_microstructure / f"{config.subject_id}_ROIWise-Microstructure.json").is_file()
        if (config.resume and not roi_statistics_missing
                and _microstructure_dti_complete(subject_root, config.subject_id, output_root, fit_method=config.microstructure_fit_method)):
            outputs = {key: path.as_posix() for key, path in micro_outputs.items() if key != "manifest"}
            tasks.append(
                ProductTaskRecord(
                    name="microstructure.dti",
                    status="skipped",
                    reason="completed DTI microstructure outputs found",
                    regional_statistics=regional_statistics_note(parcellation, log),  # K6: keep the C8 status on resume
                    manifest_path=micro_outputs["manifest"],
                    qc_path=micro_outputs["qc"],
                    outputs=outputs,
                )
            )
        elif not config.execute:
            tasks.append(ProductTaskRecord(name="microstructure.dti", status="planned", reason="execute=False"))
        elif any(not path.is_file() for path in raw.values()):
            missing = [label for label, path in raw.items() if not path.is_file()]
            tasks.append(ProductTaskRecord(name="microstructure.dti", status="blocked", reason="missing raw input: " + ", ".join(missing)))
        else:
            from microstructure import MicrostructureConfig, run_microstructure
            from shared.pipeline import ensure_native_dti_bundle, find_subject_root, resolve_preprocessed_input_bundle

            _emit("[Orchestrator] Running Microstructure DTI", log)
            task_started = _start_task_timer("Microstructure DTI", log)
            raw_bundle = find_subject_root(
                subject_root,
                config.subject_id,
                mrtrix_bin_dir=mask_mrtrix_bin_dir,
                brain_mask_config=selected_mask_config,
            )
            analysis_input_bundle = resolve_preprocessed_input_bundle(raw_bundle, output_root=output_root)
            native_bundle = ensure_native_dti_bundle(
                analysis_input_bundle,
                fit_method=config.microstructure_fit_method,
                # This shared cache is keyed by its own input manifest. A fresh
                # product run must not force an unrelated Slicer CLI rebuild.
                reuse=True,
                log=log,
            )
            result = run_microstructure(
                MicrostructureConfig(
                    subject_id=config.subject_id,
                    raw_input_bundle=analysis_input_bundle,
                    native_bundle=native_bundle,
                    output_dir=layout.microstructure,
                    roi_label_map=parcellation,
                    run_free_water=False,
                    run_noddi=False,
                    free_water_chunk_size=config.microstructure_free_water_chunk_size,
                    free_water_engine=config.microstructure_free_water_engine,
                    free_water_n_jobs=config.microstructure_free_water_n_jobs,
                    free_water_vox_per_chunk=config.microstructure_free_water_vox_per_chunk,
                    free_water_max_fraction_for_tissue=config.microstructure_free_water_max_fraction_for_tissue,
                    force_recompute=not config.resume,
                ),
                log=log,
            )
            _finish_task_timer("Microstructure DTI", task_started, log)
            tasks.append(
                ProductTaskRecord(
                    name="microstructure.dti",
                    status="completed",
                    regional_statistics=regional_statistics_note(parcellation, log),
                    manifest_path=layout.microstructure / "manifest.json",
                    qc_path=layout.microstructure_dti / "dti_qc.json",
                    outputs={
                        "fa": result.fa_path.as_posix(),
                        "md": result.md_path.as_posix(),
                        "ad": result.ad_path.as_posix(),
                        "rd": result.rd_path.as_posix(),
                        "l1": result.l1_path.as_posix(),
                        "l2": result.l2_path.as_posix(),
                        "l3": result.l3_path.as_posix(),
                        "trace": result.trace_path.as_posix(),
                        "roi_stats_csv": result.roi_stats_csv_path.as_posix() if result.roi_stats_csv_path else None,
                        "roi_stats_json": result.roi_stats_json_path.as_posix() if result.roi_stats_json_path else None,
                    },
                )
            )

    parallel_tasks = {}
    parallel_targets = tuple(target for target in run_targets
        if target in {'microstructure_freewater','microstructure_noddi','microstructure_dki'}
        and (plan_by_target.get(target) is None or plan_by_target[target].status != 'blocked'))
    if config.execute and config.max_parallel_models > 1 and len(parallel_targets) > 1:
        parallel_tasks = _parallel_model_products(config,parallel_targets,parcellation,log)

    if "microstructure_freewater" in run_targets:
        freewater_outputs = _microstructure_freewater_outputs(subject_root, config.subject_id, output_root)
        if 'microstructure_freewater' in parallel_tasks:
            tasks.append(parallel_tasks['microstructure_freewater'])
        elif config.resume and _microstructure_freewater_complete(subject_root, config.subject_id, output_root, config=config):
            tasks.append(
                ProductTaskRecord(
                    name="microstructure.freewater",
                    status="skipped",
                    reason="completed Free Water outputs found",
                    manifest_path=freewater_outputs["manifest"],
                    qc_path=freewater_outputs["qc"],
                    outputs={key: path.as_posix() for key, path in freewater_outputs.items() if key != "manifest"},
                )
            )
        elif not config.execute:
            tasks.append(ProductTaskRecord(name="microstructure.freewater", status="planned", reason="execute=False"))
        elif any(not path.is_file() for path in raw.values()):
            missing = [label for label, path in raw.items() if not path.is_file()]
            tasks.append(ProductTaskRecord(name="microstructure.freewater", status="blocked", reason="missing raw input: " + ", ".join(missing)))
        else:
            from microstructure import MicrostructureConfig, run_microstructure
            from shared.pipeline import ensure_native_dti_bundle, find_subject_root, resolve_preprocessed_input_bundle

            _emit("[Orchestrator] Running Microstructure Free Water", log)
            raw_bundle = find_subject_root(
                subject_root,
                config.subject_id,
                mrtrix_bin_dir=mask_mrtrix_bin_dir,
                brain_mask_config=selected_mask_config,
            )
            analysis_input_bundle = resolve_preprocessed_input_bundle(raw_bundle, output_root=output_root)
            native_bundle = ensure_native_dti_bundle(
                analysis_input_bundle,
                fit_method=config.microstructure_fit_method,
                reuse=True,
                log=log,
            )
            result = run_microstructure(
                MicrostructureConfig(
                    subject_id=config.subject_id,
                    raw_input_bundle=analysis_input_bundle,
                    native_bundle=native_bundle,
                    output_dir=layout.microstructure,
                    roi_label_map=parcellation,
                    run_free_water=True,
                    run_noddi=False,
                    free_water_chunk_size=config.microstructure_free_water_chunk_size,
                    free_water_engine=config.microstructure_free_water_engine,
                    free_water_n_jobs=config.microstructure_free_water_n_jobs,
                    free_water_vox_per_chunk=config.microstructure_free_water_vox_per_chunk,
                    free_water_max_fraction_for_tissue=config.microstructure_free_water_max_fraction_for_tissue,
                    force_recompute=not config.resume,
                ),
                log=log,
            )
            tasks.append(
                ProductTaskRecord(
                    name="microstructure.freewater",
                    status="completed",
                    manifest_path=freewater_outputs["manifest"],
                    qc_path=result.free_water_qc_path,
                    outputs={
                        "fwf": result.fwf_path.as_posix(),
                        "fw_fa": result.fw_tissue_fa_path.as_posix(),
                        "fw_md": result.fw_tissue_md_path.as_posix(),
                        "valid_mask": result.fw_tissue_valid_mask_path.as_posix(),
                        "qc": result.free_water_qc_path.as_posix(),
                    },
                )
            )

    if "microstructure_noddi" in run_targets:
        noddi_outputs = _microstructure_noddi_outputs(subject_root, config.subject_id, output_root)
        if 'microstructure_noddi' in parallel_tasks:
            tasks.append(parallel_tasks['microstructure_noddi'])
        elif config.resume and _microstructure_noddi_complete(subject_root, config.subject_id, output_root, config=config):
            tasks.append(
                ProductTaskRecord(
                    name="microstructure.noddi",
                    status="skipped",
                    reason="completed NODDI outputs found",
                    manifest_path=noddi_outputs["manifest"],
                    qc_path=noddi_outputs["qc"],
                    outputs={key: path.as_posix() for key, path in noddi_outputs.items() if key != "manifest"},
                )
            )
        elif not config.execute:
            tasks.append(ProductTaskRecord(name="microstructure.noddi", status="planned", reason="execute=False"))
        elif any(not path.is_file() for path in raw.values()):
            missing = [label for label, path in raw.items() if not path.is_file()]
            tasks.append(ProductTaskRecord(name="microstructure.noddi", status="blocked", reason="missing raw input: " + ", ".join(missing)))
        else:
            from microstructure import MicrostructureConfig, run_microstructure
            from shared.pipeline import ensure_native_dti_bundle, find_subject_root, resolve_preprocessed_input_bundle

            _emit("[Orchestrator] Running Microstructure NODDI", log)
            raw_bundle = find_subject_root(
                subject_root,
                config.subject_id,
                mrtrix_bin_dir=mask_mrtrix_bin_dir,
                brain_mask_config=selected_mask_config,
            )
            analysis_input_bundle = resolve_preprocessed_input_bundle(raw_bundle, output_root=output_root)
            native_bundle = ensure_native_dti_bundle(
                analysis_input_bundle,
                fit_method=config.microstructure_fit_method,
                reuse=True,
                log=log,
            )
            result = run_microstructure(
                MicrostructureConfig(
                    subject_id=config.subject_id,
                    raw_input_bundle=analysis_input_bundle,
                    native_bundle=native_bundle,
                    output_dir=layout.microstructure,
                    roi_label_map=parcellation,
                    # Free-water has its own preceding task. A fresh run must
                    # not fit it a second time while entering NODDI.
                    run_free_water=False,
                    run_noddi=True,
                    free_water_chunk_size=config.microstructure_free_water_chunk_size,
                    free_water_engine=config.microstructure_free_water_engine,
                    free_water_n_jobs=config.microstructure_free_water_n_jobs,
                    free_water_vox_per_chunk=config.microstructure_free_water_vox_per_chunk,
                    free_water_max_fraction_for_tissue=config.microstructure_free_water_max_fraction_for_tissue,
                    force_recompute=not config.resume,
                ),
                log=log,
            )
            tasks.append(
                ProductTaskRecord(
                    name="microstructure.noddi",
                    status="completed",
                    manifest_path=noddi_outputs["manifest"],
                    qc_path=result.noddi_qc_path,
                    outputs={
                        "noddi_ndi": result.noddi_ndi_path.as_posix(),
                        "noddi_odi": result.noddi_odi_path.as_posix(),
                        "noddi_fwf": result.noddi_fwf_path.as_posix(),
                        "noddi_tissue_valid_mask": result.noddi_tissue_valid_mask_path.as_posix(),
                        "noddi_rmse": result.noddi_rmse_path.as_posix(),
                        "noddi_nrmse": result.noddi_nrmse_path.as_posix(),
                        "noddi_qc": result.noddi_qc_path.as_posix(),
                    },
                )
            )

    if "microstructure_dki" in run_targets:
        from microstructure.dki import run_dki

        dki_plan = plan_by_target.get("microstructure_dki")
        if 'microstructure_dki' in parallel_tasks:
            tasks.append(parallel_tasks['microstructure_dki'])
        elif not config.execute:
            tasks.append(ProductTaskRecord(name="microstructure.dki", status="planned", reason="execute=False"))
        elif dki_plan and dki_plan.status == "blocked":
            tasks.append(ProductTaskRecord(name="microstructure.dki", status="blocked", reason=dki_plan.reason))
        else:
            _emit("[Orchestrator] Running guarded Microstructure DKI", log)
            # Resolve again after preprocessing: raw-gradient eligibility is not
            # permission to fit stale gradients or a mask from a different grid.
            try:
                result = run_dki(_dki_config(config, _resolve_analysis_inputs(subject_root, config.subject_id, output_root)), log=log)
                outputs = {key: str(path) for key, path in result.outputs.items() if key not in {"manifest", "progress"}}
                outputs.update({f"mkcurve_{key}": str(path) for key, path in result.mkcurve_outputs.items()})
                tasks.append(ProductTaskRecord(
                    name="microstructure.dki", status="skipped" if result.status == "cached" else "completed",
                    reason="matching DKI outputs reused" if result.status == "cached" else "",
                    manifest_path=result.manifest_path, qc_path=result.qc_path, outputs=outputs,
                ))
            except Exception as exc:
                tasks.append(ProductTaskRecord(name="microstructure.dki", status="failed", reason=str(exc)))
                _emit(f"[Orchestrator] DKI failed; other selected products can continue: {exc}", log)

    if "tractography" in run_targets and primary_tractography_backend == 'ukf':
        from tractography.ukf import run_ukf
        try:
            if not config.execute:
                tasks.append(ProductTaskRecord(name='tractography.ukf', status='planned', reason='execute=False'))
            else:
                result = run_ukf(_ukf_raw(config, analysis_inputs), ukf_dir,
                    options=config.ukf_parameters, executable=config.ukf_executable,
                    converter=config.dwiconvert_executable, force=not config.resume, log=log)
                tasks.append(ProductTaskRecord(name='tractography.ukf',
                    status='skipped' if result['status'] == 'cached' else 'completed',
                    qc_path=ukf_outputs['qc'], manifest_path=ukf_outputs['manifest'],
                    outputs={key: str(path) for key, path in ukf_outputs.items() if key not in {'qc', 'manifest'}}))
        except Exception as exc:
            tasks.append(ProductTaskRecord(name='tractography.ukf', status='failed', reason=str(exc),
                qc_path=ukf_outputs['qc'] if ukf_outputs['qc'].is_file() else None,
                manifest_path=ukf_outputs['manifest'] if ukf_outputs['manifest'].is_file() else None))
            _emit('[Orchestrator] UKF failed; downstream UKF consumers blocked: ' + str(exc), log)

    ukf_upstream_invalid = (primary_tractography_backend == 'ukf' and config.execute and
        (any(task.name == 'tractography.ukf' and task.status not in {'completed', 'skipped'} for task in tasks)
         or not _ukf_current(config, analysis_inputs, ukf_dir)))

    if "tractography" in run_targets and primary_tractography_backend != 'ukf':
        tractography_qc = csd_qc if primary_tractography_backend == "csd_act" else dti_qc
        tractography_manifest = csd_dir / "csd_act_manifest.json" if primary_tractography_backend == "csd_act" else dti_manifest
        tractography_task_name = "tractography.csd_act" if primary_tractography_backend == "csd_act" else "tractography.dti_det"
        tractography_complete = (
            _csd_act_complete(
                csd_qc,
                csd_dir / "csd_act_manifest.json",
                csd_tractogram,
                csd_tractogram_trk,
                csd_tractogram_vtk,
                sift2_weights,
                analysis_inputs,
                allow_ddseg_only=config.allow_act_ddseg_only,
            )
            and (not config.bundle_tractography_backend or _csd_bundle_parameters_current(config, csd_dir / 'csd_act_manifest.json'))
            if primary_tractography_backend == "csd_act"
            else _qc_completed(tractography_qc) and tractogram_trk.is_file()
            and (not config.bundle_tractography_backend or (_dti_bundle_source_current(config, dti_manifest, analysis_inputs)
                 and _dti_bundle_qc_acceptable(config, dti_qc, tractogram_trk)))
        )
        if config.resume and tractography_complete:
            tasks.append(
                ProductTaskRecord(
                    name=tractography_task_name,
                    status="skipped",
                    reason="completed outputs found",
                    qc_path=tractography_qc,
                    manifest_path=tractography_manifest,
                    outputs={
                        "tractogram": tractogram.as_posix(),
                        "tractogram_trk": tractogram_trk.as_posix(),
                        "tractogram_vtk": current_vtk.as_posix() if primary_tractography_backend == 'csd_act' or current_vtk.is_file() else None,
                    },
                )
            )
        elif not config.execute:
            tasks.append(ProductTaskRecord(name=tractography_task_name, status="planned", reason="execute=False"))
        elif any(not path.is_file() for path in raw.values()):
            missing = [label for label, path in raw.items() if not path.is_file()]
            tasks.append(ProductTaskRecord(name=tractography_task_name, status="blocked", reason="missing raw input: " + ", ".join(missing)))
        elif primary_tractography_backend == "csd_act" and segmentation is None:
            tasks.append(ProductTaskRecord(name=tractography_task_name, status="blocked", reason="missing DDSeg segmentation"))
        elif primary_tractography_backend == "csd_act" and parcellation is None and not config.allow_act_ddseg_only:
            tasks.append(ProductTaskRecord(name=tractography_task_name, status="blocked", reason="missing DDParcel anatomical labels for ACT"))
        elif primary_tractography_backend == "csd_act" and ddseg_world_qc_gate() is not None:
            tasks.append(
                ProductTaskRecord(
                    name=tractography_task_name,
                    status="blocked",
                    reason="DDSeg world-space QC gate failed: " + str(ddseg_world_qc_state["failure"]),
                )
            )
        else:
            if primary_tractography_backend == "csd_act":
                from tractography.act_prep import ACTPrepConfig, run_act_prep
                from tractography.csd import CSDActConfig, run_csd_act

                _emit(f"[Orchestrator] Running CSD/ACT {run_name}", log)
                task_started = _start_task_timer(f"CSD/ACT {run_name}", log)
                anatomy_sources = {
                    "ddseg_source_dwi": _recorded_source_dwi(layout.anatomy_input_dir("ddseg") / "ddseg_qc.json", "source_dwi"),
                    "ddparcel_source_dwi": _recorded_source_dwi(layout.anatomy_input_dir("ddparcel") / "ddparcel_source.json", "dwi"),
                    "csd_dwi": analysis_inputs["dwi"].resolve().as_posix(),
                }
                if parcellation is None:
                    _emit("[Orchestrator] WARNING: Explicit DDSeg-only ACT fallback; subcortical GM priors are unavailable.", log)
                act_prep = run_act_prep(
                    ACTPrepConfig(
                        subject_id=config.subject_id,
                        segmentation_path=segmentation,
                        output_dir=layout.act_prep,
                        execute=True,
                        parcellation_path=parcellation,
                        mrtrix_bin_dir=config.mrtrix_bin_dir,
                        anatomy_source=anatomy_sources,
                        allow_ddseg_only=config.allow_act_ddseg_only,
                    ),
                    log=log,
                )
                result = run_csd_act(
                    CSDActConfig(
                        subject_id=config.subject_id,
                        dwi_path=analysis_inputs["dwi"],
                        bval_path=analysis_inputs["bval"],
                        bvec_path=analysis_inputs["bvec"],
                        mask_path=analysis_inputs["mask"],
                        five_tt_path=act_prep.five_tt_path,
                        gmwmi_path=act_prep.gmwmi_path,
                        output_dir=csd_dir,
                        tractogram_dir=tractogram_dir,
                        select_streamlines=config.select_streamlines,
                        min_length_mm=config.dti_min_length_mm,
                        max_length_mm=config.dti_max_length_mm,
                        use_sift2=True,
                        export_trk=True,
                        export_vtk=True,
                        mrtrix_bin_dir=config.mrtrix_bin_dir,
                        nthreads=config.mrtrix_nthreads,
                        random_seed=config.mrtrix_random_seed,
                        seeding=config.tractography_seeding,
                        anatomy_source_dwi=_anatomy_dwi_for_check(anatomy_sources),
                        allow_anatomy_space_mismatch=config.allow_anatomy_space_mismatch,
                        act_prep_qc_path=act_prep.qc_path,
                        execute=True,
                    ),
                    log=log,
                )
                _finish_task_timer(f"CSD/ACT {run_name}", task_started, log)
                tasks.append(
                    ProductTaskRecord(
                        name=tractography_task_name,
                        status="completed",
                        manifest_path=result.manifest_path,
                        qc_path=result.qc_path,
                        outputs={
                            "tractogram": result.tractogram_path.as_posix(),
                            "tractogram_trk": result.tractogram_trk_path.as_posix() if result.tractogram_trk_path else None,
                            "tractogram_vtk": result.tractogram_vtk_path.as_posix() if result.tractogram_vtk_path else None,
                            "sift2_weights": result.sift2_weights_path.as_posix() if result.sift2_weights_path else None,
                        },
                    )
                )
            else:
                from tractography.dti import DTITrackingConfig, run_dti_tracking
                from shared.pipeline import RawInputBundle

                _emit(f"[Orchestrator] Running DTI tractography {run_name}", log)
                task_started = _start_task_timer(f"DTI tractography {run_name}", log)
                result = run_dti_tracking(
                    DTITrackingConfig(
                        subject_id=config.subject_id,
                        input_root=subject_root,
                        input_bundle=RawInputBundle(
                            subject_root=subject_root,
                            subject_id=config.subject_id,
                            input_dir=analysis_inputs["dwi"].parent,
                            dwi_path=analysis_inputs["dwi"],
                            bval_path=analysis_inputs["bval"],
                            bvec_path=analysis_inputs["bvec"],
                            mask_path=analysis_inputs["mask"],
                        ),
                        output_dir=dti_tractogram_dir,
                        fit_method=config.microstructure_fit_method,
                        tracking_method="deterministic",
                        output_format="trk",
                        # DTI produces at most one seed per selected voxel at
                        # the default density.  Honor the unified CLI's
                        # streamline budget so whole-brain runs do not seed
                        # every eligible voxel and exhaust memory on Windows.
                        max_seed_voxels=config.select_streamlines,
                        min_length_mm=config.dti_min_length_mm,
                        max_length_mm=config.dti_max_length_mm,
                        execute=True,
                    ),
                    log=log,
                )
                _finish_task_timer(f"DTI tractography {run_name}", task_started, log)
                tasks.append(
                    ProductTaskRecord(
                        name=tractography_task_name,
                        status="completed",
                        manifest_path=result.manifest_path,
                        qc_path=result.qc_path,
                        outputs={"tractogram_trk": result.streamlines_path.as_posix()},
                    )
                )

    bundle_source_invalid = False
    if config.execute and config.bundle_tractography_backend:
        tracking_tasks = [task for task in tasks if task.name == 'tractography.' + primary_tractography_backend]
        bundle_source_invalid = not tracking_tasks or tracking_tasks[-1].status not in {'completed', 'skipped'}
        if primary_tractography_backend == 'ukf':
            bundle_source_invalid = bundle_source_invalid or ukf_upstream_invalid
        elif primary_tractography_backend == 'dti_det':
            bundle_source_invalid = bundle_source_invalid or not (_dti_bundle_qc_acceptable(config, dti_qc, tractogram_trk)
                                                                  and _dti_bundle_source_current(config, dti_manifest, analysis_inputs))
        else:
            bundle_source_invalid = bundle_source_invalid or not _csd_act_complete(
                csd_qc, csd_dir / 'csd_act_manifest.json', csd_tractogram,
                csd_tractogram_trk, csd_tractogram_vtk, sift2_weights, analysis_inputs,
                allow_ddseg_only=config.allow_act_ddseg_only)
            bundle_source_invalid = bundle_source_invalid or not _csd_bundle_parameters_current(config, csd_dir / 'csd_act_manifest.json')

    if "bundles" in run_targets:
        wmql_qc = wmql_dir / "wmql_qc.json"
        if bundle_source_invalid:
            tasks.append(ProductTaskRecord(name='bundles.wmql', status='blocked', reason='selected tractography did not pass current-run QC'))
        elif ukf_upstream_invalid:
            tasks.append(ProductTaskRecord(name='bundles.wmql', status='blocked', reason='current UKF run is not QC-valid'))
        elif config.resume and _wmql_complete(
            wmql_qc,
            query_preset=config.wmql_query_preset,
            query_selection=config.wmql_query_selection,
            atlas_mode=config.wmql_atlas_mode,
            input_tractography=tractogram_trk,
            parcellation=parcellation,
        ):
            tasks.append(ProductTaskRecord(name="bundles.wmql", status="skipped", reason="completed outputs found", qc_path=wmql_qc, manifest_path=wmql_dir / "wmql_manifest.json"))
        elif not config.execute:
            tasks.append(ProductTaskRecord(name="bundles.wmql", status="planned", reason="execute=False"))
        elif parcellation is None:
            tasks.append(ProductTaskRecord(name="bundles.wmql", status="blocked", reason="missing parcellation"))
        elif not tractogram_trk.is_file():
            tasks.append(ProductTaskRecord(name="bundles.wmql", status="blocked", reason="missing TRK tractogram"))
        else:
            from tractography.wmql import WMQLConfig, run_wmql_bundle_segmentation

            _emit(f"[Orchestrator] Running WMQL bundles {run_name}", log)
            task_started = _start_task_timer(f"WMQL bundles {run_name}", log)
            result = run_wmql_bundle_segmentation(
                WMQLConfig(
                    subject_id=config.subject_id,
                    input_tractography=tractogram_trk,
                    atlas_path=parcellation,
                    output_dir=wmql_dir,
                    tract_querier_root=config.tract_querier_root,
                    query_preset=config.wmql_query_preset,
                    query_selection=config.wmql_query_selection,
                    atlas_mode=config.wmql_atlas_mode,
                    execute=True,
                ),
                log=log,
            )
            _finish_task_timer(f"WMQL bundles {run_name}", task_started, log)
            wmql_complete = _wmql_complete(
                result.qc_path,
                manifest_path=result.manifest_path,
                input_tractography=tractogram_trk,
                parcellation=parcellation,
                query_preset=config.wmql_query_preset,
                query_selection=config.wmql_query_selection,
                atlas_mode=config.wmql_atlas_mode,
            )
            tasks.append(
                _task_from_result(
                    "bundles.wmql",
                    result,
                    status="completed" if wmql_complete else "failed",
                    reason="" if wmql_complete else _read_json(result.qc_path).get('error', "WMQL scientific QC failed"),
                )
            )

    if "saf" in run_targets:
        saf_qc = saf_dir / "saf_qc.json"
        fod_candidate = csd_dir / f"{config.subject_id}_wmfod_norm.mif"
        fod_path = fod_candidate if fod_candidate.is_file() else (csd_dir / f"{config.subject_id}_wmfod.mif")
        fs_dir = _resolve_optional_path(config.saf_fs_dir) or _surface_postprocess_saf_fs_dir(subject_root, config.subject_id, output_root)
        if config.resume and _saf_complete(saf_dir, fod_path=fod_path, fs_dir=fs_dir):
            saf_qc_data = _read_json(saf_qc)
            saf_vtk = saf_dir / "SAF.vtk"
            streamline_count = saf_qc_data.get("streamline_count")
            tasks.append(
                ProductTaskRecord(
                    name="tractography.saf",
                    status="skipped",
                    reason="completed outputs found",
                    qc_path=saf_qc,
                    manifest_path=saf_dir / "saf_manifest.json",
                    outputs={
                        "tractogram": saf_tractogram.as_posix(),
                        "tractogram_vtk": saf_vtk.as_posix() if saf_vtk.is_file() else None,
                        "streamline_count": str(streamline_count) if streamline_count is not None else None,
                    },
                )
            )
        elif not config.execute:
            tasks.append(ProductTaskRecord(name="tractography.saf", status="planned", reason="execute=False"))
        else:
            pregenerated_tractogram = _resolve_optional_path(config.saf_pregenerated_tractogram)
            saf_tckgen_options = config.saf_tckgen_options
            saf_maxlen = 40.0
            if pregenerated_tractogram is None:
                preproc_outputs = _preprocessing_outputs(subject_root, config.subject_id, output_root)
                seed_mask = preproc_outputs["mask"] if preproc_outputs["mask"].is_file() else raw["mask"]
                if not seed_mask.is_file():
                    tasks.append(
                        ProductTaskRecord(
                            name="tractography.saf",
                            status="blocked",
                            reason="missing pregenerated tractogram and no brain mask available for SAF tckgen",
                        )
                    )
                    seed_mask = None
                elif not saf_tckgen_options.strip():
                    saf_tckgen_options = (
                        f"-seed_image {seed_mask.expanduser().resolve().as_posix()} "
                        "-cutoff 0.05 -select 1000000 -maxlength 40"
                    )
            if fs_dir is None:
                tasks.append(
                    ProductTaskRecord(
                        name="tractography.saf",
                        status="blocked",
                        reason="missing --saf-fs-dir or completed whole DDSurfer postprocess outputs",
                    )
                )
            elif not fod_path.is_file():
                tasks.append(ProductTaskRecord(name="tractography.saf", status="blocked", reason="missing CSD FOD output for SAF"))
            elif pregenerated_tractogram is None and 'seed_mask' in locals() and seed_mask is None:
                pass
            else:
                from tractography import SAFConfig, run_saf

                _emit(f"[Orchestrator] Running SAF {run_name}", log)
                result = run_saf(
                    SAFConfig(
                        subject_id=config.subject_id,
                        input_root=subject_root,
                        fs_dir=fs_dir,
                        fod_path=fod_path,
                        output_dir=saf_dir,
                        pregenerated_tractogram=pregenerated_tractogram,
                        tckgen_options=saf_tckgen_options,
                        mrtrix_bin_dir=config.mrtrix_bin_dir,
                        mrtrix_prefix=config.saf_mrtrix_prefix,
                        maxlen=saf_maxlen,
                        force_recompute=not config.resume,
                        execute=True,
                    ),
                    log=log,
                )
                tasks.append(
                    ProductTaskRecord(
                        name="tractography.saf",
                        status="completed",
                        qc_path=result.qc_path,
                        manifest_path=result.manifest_path,
                        outputs={
                            "tractogram": result.saf_tractogram.as_posix(),
                            "tractogram_vtk": result.saf_vtk_path.as_posix() if result.saf_vtk_path else None,
                            "streamline_count": str(result.streamline_count) if result.streamline_count is not None else None,
                        },
                    )
                )

    bundle_conversion_error = None
    if (config.execute and config.bundle_tractography_backend == 'dti_det'
            and set(run_targets).intersection({'wma', 'tractcloud'}) and not bundle_source_invalid):
        from tractography.conversion import ensure_vtk_tractogram
        try:
            ensure_vtk_tractogram(tractogram_trk, current_vtk)
        except (OSError, ValueError, RuntimeError) as exc:
            bundle_conversion_error = 'Automatic DTI-to-VTK conversion failed: ' + str(exc)

    if "wma" in run_targets:
        wma_qc = wma_dir / "wma_org_atlas_qc.json"
        wma_summary = _wma_output_summary(wma_dir)
        anatomical_tract_count = int(wma_summary.get("anatomical_tract_count", 0) or 0)
        nonempty_tract_count = int(wma_summary.get("anatomical_tract_nonempty_count", 0) or 0)
        wma_resume_complete = bool(config.resume and _wma_complete(
            wma_dir,
            input_tractography=wma_input_tractography,
            atlas_dir=config.wma_atlas_dir,
            atlas_family=config.wma_atlas,
        ))
        if bundle_source_invalid or bundle_conversion_error:
            tasks.append(ProductTaskRecord(name='bundles.wma', status='blocked', reason=bundle_conversion_error or 'selected tractography did not pass current-run QC'))
        elif ukf_upstream_invalid and config.wma_input_tractography is None and not wma_resume_complete:
            # F1: a complete WMA result stays reusable. _wma_complete checks the path, size and
            # mtime of its input tractogram against the WMA manifest, so it is not a stale result.
            tasks.append(ProductTaskRecord(name='bundles.wma', status='blocked', reason='current UKF run is not QC-valid'))
        elif wma_resume_complete:
            tasks.append(
                ProductTaskRecord(
                    name="bundles.wma",
                    status="skipped",
                    reason="completed outputs found",
                    qc_path=wma_qc,
                    manifest_path=wma_dir / "wma_org_atlas_manifest.json",
                    outputs={
                        "anatomical_tract_count": str(anatomical_tract_count),
                        "anatomical_tract_nonempty_count": str(nonempty_tract_count),
                    },
                )
            )
        elif not config.execute:
            reason = "execute=False"
            if config.resume and _qc_completed(wma_qc):
                reason = "execute=False; existing QC present but anatomical tracts missing"
            tasks.append(ProductTaskRecord(name="bundles.wma", status="planned", reason=reason))
        else:
            if config.wma_input_tractography is not None:
                wma_input_tractography = _resolve_wma_input_tractography(
                    subject_root,
                    config.wma_input_tractography,
                    output_root=output_root,
                )
            elif primary_tractography_backend != 'ukf' and not config.bundle_tractography_backend:
                wma_input_tractography = _resolve_wma_input_tractography(
                    subject_root,
                    None,
                    output_root=output_root,
                )
            if config.resume and _wma_complete(
                wma_dir,
                input_tractography=wma_input_tractography,
                atlas_dir=config.wma_atlas_dir,
                atlas_family=config.wma_atlas,
            ):
                tasks.append(ProductTaskRecord(name="bundles.wma", status="skipped", reason="completed outputs found", qc_path=wma_qc, manifest_path=wma_dir / "wma_org_atlas_manifest.json", outputs={"anatomical_tract_count": str(anatomical_tract_count), "anatomical_tract_nonempty_count": str(nonempty_tract_count)}))
            elif wma_input_tractography is None:
                tasks.append(
                    ProductTaskRecord(
                        name="bundles.wma",
                        status="blocked",
                        reason="Select --tractography-backend ukf for automatic UKF-to-WMA tracking, or optionally provide --wma-input-tractography; no silent CSD substitution",
                    )
                )
            elif not wma_input_tractography.is_file():
                tasks.append(ProductTaskRecord(name='bundles.wma', status='blocked', reason='selected tractography file is missing'))
            elif (
                wma_input_tractography.suffix.lower() == ".tck"
                and _resolve_mrtrix_command("tckconvert", config.mrtrix_bin_dir) is None
            ):
                tasks.append(
                    ProductTaskRecord(
                        name="bundles.wma",
                        status="blocked",
                        reason="TCK input requires MRtrix3 tckconvert",
                    )
                )
            elif config.wma_atlas_dir is None:
                tasks.append(ProductTaskRecord(name="bundles.wma", status="blocked", reason="missing --wma-atlas-dir"))
            elif slicer_executable is None:
                tasks.append(ProductTaskRecord(name="bundles.wma", status="blocked", reason="missing Slicer executable for WMA"))
            else:
                from tractography import WMAOrgAtlasConfig, run_wma_org_atlas

                _emit(f"[Orchestrator] Running WMA {run_name}", log)
                result = run_wma_org_atlas(
                    WMAOrgAtlasConfig(
                        subject_id=config.subject_id,
                        input_tractography=wma_input_tractography,
                        atlas_dir=config.wma_atlas_dir,
                        atlas_family=config.wma_atlas,
                        output_dir=wma_dir,
                        slicer_executable=slicer_executable,
                        wma_root=config.wma_root,
                        num_threads=config.wma_num_threads,
                        mrtrix_bin_dir=config.mrtrix_bin_dir,
                        use_virtual_x=config.wma_use_virtual_x,
                        export_diffusion_measures=config.wma_export_diffusion_measures,
                        fiber_tract_measurements_cli=config.fiber_tract_measurements_cli,
                        execute=True,
                    ),
                    log=log,
                )
                wma_complete = _wma_complete(
                    wma_dir,
                    input_tractography=wma_input_tractography,
                    atlas_dir=config.wma_atlas_dir,
                    atlas_family=config.wma_atlas,
                )
                tasks.append(
                    _task_from_result(
                        "bundles.wma",
                        result,
                        status="completed" if wma_complete else "failed",
                        reason="" if wma_complete else "WMA outputs did not satisfy the anatomical-tract QC contract",
                    )
                )

    if "tractcloud" in run_targets:
        tractcloud_qc = tractcloud_dir / "tractcloud_qc.json"
        conversion_error = bundle_conversion_error
        if (config.execute and primary_tractography_backend == 'dti_det'
                and config.tractcloud_input_tractography is None and not config.bundle_tractography_backend):
            if not _qc_completed(dti_qc) or not tractogram_trk.is_file():
                conversion_error = 'Current DTI tractography is not QC-valid; no historical CSD fallback'
            else:
                from tractography.conversion import ensure_vtk_tractogram
                try:
                    ensure_vtk_tractogram(tractogram_trk, current_vtk)
                except (OSError, ValueError, RuntimeError) as exc:
                    conversion_error = 'Automatic DTI-to-VTK conversion failed: ' + str(exc)
        tractcloud_input_tractography = _resolve_bundle_input_tractography(
            subject_root,
            config.tractcloud_input_tractography,
            output_root=output_root,
            run_name=run_name,
            current_run_tractography=current_vtk,
            require_current_run=True,
        )
        if bundle_source_invalid:
            tasks.append(ProductTaskRecord(name='bundles.tractcloud', status='blocked', reason='selected tractography did not pass current-run QC'))
        elif conversion_error:
            tasks.append(ProductTaskRecord(name='bundles.tractcloud', status='blocked', reason=conversion_error))
        elif ukf_upstream_invalid and config.tractcloud_input_tractography is None:
            tasks.append(ProductTaskRecord(name='bundles.tractcloud', status='blocked', reason='current UKF run is not QC-valid'))
        elif config.resume and _tractcloud_complete(
            tractcloud_dir,
            input_tractography=tractcloud_input_tractography,
            random_seed=config.tractcloud_random_seed,
            config=config,
        ):
            from tractography.tractcloud import summarize_tractcloud_outputs

            output_summary = summarize_tractcloud_outputs(tractcloud_dir)
            tasks.append(
                ProductTaskRecord(
                    name="bundles.tractcloud",
                    status="skipped",
                    reason="completed outputs found",
                    qc_path=tractcloud_qc,
                    manifest_path=tractcloud_dir / "tractcloud_manifest.json",
                    outputs={"bundle_count": str(output_summary.get("bundle_count"))},
                )
            )
        elif not config.execute:
            reason = "execute=False"
            if config.resume and _qc_completed(tractcloud_qc):
                reason = "execute=False; existing QC present but no bundle outputs found"
            tasks.append(ProductTaskRecord(name="bundles.tractcloud", status="planned", reason=reason))
        else:
            if tractcloud_input_tractography is None:
                tasks.append(ProductTaskRecord(name="bundles.tractcloud", status="blocked", reason="missing whole-brain VTK/VTP tractography for TractCloud"))
            else:
                from tractography import TractCloudConfig, run_tractcloud_parcellation

                _emit(f"[Orchestrator] Running TractCloud {run_name}", log)
                result = run_tractcloud_parcellation(
                    TractCloudConfig(
                        subject_id=config.subject_id,
                        input_tractography=tractcloud_input_tractography,
                        output_dir=tractcloud_dir,
                        tractcloud_root=config.tractcloud_root,
                        python_executable=config.tractcloud_python_executable,
                        data_dir=config.tractcloud_data_dir,
                        model_assets_dir=config.model_assets_dir,
                        device=tractcloud_device,
                        batch_size=config.tractcloud_batch_size,
                        random_seed=config.tractcloud_random_seed,
                        execute=True,
                    ),
                    log=log,
                )
                tasks.append(_task_from_result("bundles.tractcloud", result))

    if "connectome" in run_targets:
        weighted_connectome_handled = False
        conn_qc = connectome_dir / f"{config.subject_id}-connectome-streamline_count_qc.json"
        conn_matrix = connectome_dir / f"{config.subject_id}-connectome-streamline_count.csv"
        conn_nodes = connectome_dir / f"{config.subject_id}-connectome-nodes.tsv"
        conn_manifest = connectome_dir / f"{config.subject_id}-connectome-streamline_count_manifest.json"
        if ukf_upstream_invalid:
            tasks.append(ProductTaskRecord(name='connectome.streamline_count', status='blocked', reason='current UKF run is not QC-valid'))
        elif config.resume and _connectome_complete(
            conn_qc,
            conn_matrix,
            conn_nodes,
            manifest_path=conn_manifest,
            tractography=tractogram,
            parcellation=parcellation,
        ):
            tasks.append(ProductTaskRecord(
                name="connectome.streamline_count", status="skipped", reason="completed outputs found",
                qc_path=conn_qc, manifest_path=conn_manifest,
                outputs={"matrix_path": conn_matrix.as_posix(), "nodes_path": conn_nodes.as_posix()},
            ))
        elif not config.execute:
            tasks.append(ProductTaskRecord(name="connectome.streamline_count", status="planned", reason="execute=False"))
        elif parcellation is None:
            tasks.append(ProductTaskRecord(name="connectome.streamline_count", status="blocked", reason="missing parcellation"))
        elif primary_tractography_backend == "csd_act" and not tractogram.is_file():
            tasks.append(ProductTaskRecord(name="connectome.streamline_count", status="blocked", reason="missing TCK tractogram"))
        elif primary_tractography_backend != "csd_act" and config.connectome_backend == "mrtrix":
            tasks.append(ProductTaskRecord(name="connectome.streamline_count", status="blocked", reason="UKF/DTI tractography currently supports python connectome backend only"))
        elif not tractogram.is_file():
            tasks.append(ProductTaskRecord(name="connectome.streamline_count", status="blocked", reason="missing tractogram"))
        else:
            from connectome import ConnectomeConfig, run_connectome

            _emit(f"[Orchestrator] Running connectome {run_name}", log)
            task_started = _start_task_timer(f"connectome {run_name}", log)
            result = run_connectome(
                ConnectomeConfig(
                    subject_id=config.subject_id,
                    tractography_path=tractogram,
                    parcellation_path=parcellation,
                    output_dir=connectome_dir,
                    metric="streamline_count",
                    backend=config.connectome_backend,
                    mrtrix_bin_dir=config.mrtrix_bin_dir,
                    compress_mrtrix_dense=config.connectome_compress_mrtrix_dense,
                    execute=True,
                ),
                log=log,
            )
            _finish_task_timer(f"connectome {run_name}", task_started, log)
            tasks.append(_task_from_result("connectome.streamline_count", result))
            if primary_tractography_backend == 'csd_act' and sift2_weights.is_file():
                weighted = run_connectome(
                    ConnectomeConfig(
                        subject_id=config.subject_id,
                        tractography_path=tractogram,
                        parcellation_path=parcellation,
                        output_dir=sift2_connectome_dir,
                        metric="sift2_weighted_streamline_count",
                        sift2_weights_path=sift2_weights,
                        backend=config.connectome_backend,
                        mrtrix_bin_dir=config.mrtrix_bin_dir,
                        compress_mrtrix_dense=config.connectome_compress_mrtrix_dense,
                        execute=True,
                    ),
                    log=log,
                )
                tasks.append(_task_from_result("connectome.sift2_weighted", weighted))
                weighted_connectome_handled = True

        if not weighted_connectome_handled and primary_tractography_backend == 'csd_act' and sift2_weights.is_file():
            sift2_qc = sift2_connectome_dir / f"{config.subject_id}-connectome-sift2_weighted_streamline_count_qc.json"
            sift2_matrix = sift2_connectome_dir / f"{config.subject_id}-connectome-sift2_weighted_streamline_count.csv"
            sift2_nodes = sift2_connectome_dir / f"{config.subject_id}-connectome-nodes.tsv"
            if config.resume and _connectome_complete(
                sift2_qc,
                sift2_matrix,
                sift2_nodes,
                manifest_path=sift2_connectome_dir / f"{config.subject_id}-connectome-sift2_weighted_streamline_count_manifest.json",
                tractography=tractogram,
                parcellation=parcellation,
                sift2_weights=sift2_weights,
            ):
                tasks.append(
                    ProductTaskRecord(
                        name="connectome.sift2_weighted",
                        status="skipped",
                        reason="completed outputs found",
                        qc_path=sift2_qc,
                        manifest_path=sift2_connectome_dir / f"{config.subject_id}-connectome-sift2_weighted_streamline_count_manifest.json",
                        outputs={"matrix_path": sift2_matrix.as_posix(), "nodes_path": sift2_nodes.as_posix()},
                    )
                )
            elif not config.execute:
                tasks.append(ProductTaskRecord(name="connectome.sift2_weighted", status="planned", reason="execute=False"))
            elif parcellation is None:
                tasks.append(ProductTaskRecord(name="connectome.sift2_weighted", status="blocked", reason="missing parcellation"))
            elif not tractogram.is_file():
                tasks.append(ProductTaskRecord(name="connectome.sift2_weighted", status="blocked", reason="missing TCK tractogram"))
            else:
                from connectome import ConnectomeConfig, run_connectome

                weighted = run_connectome(
                    ConnectomeConfig(
                        subject_id=config.subject_id,
                        tractography_path=tractogram,
                        parcellation_path=parcellation,
                        output_dir=sift2_connectome_dir,
                        metric="sift2_weighted_streamline_count",
                        sift2_weights_path=sift2_weights,
                        backend=config.connectome_backend,
                        mrtrix_bin_dir=config.mrtrix_bin_dir,
                        compress_mrtrix_dense=config.connectome_compress_mrtrix_dense,
                        execute=True,
                    ),
                    log=log,
                )
                tasks.append(_task_from_result("connectome.sift2_weighted", weighted))

    execution_seconds = time.monotonic() - execution_started
    metadata_started = time.monotonic()
    normalized_task_records = _write_normalized_task_records(
        layout=layout,
        run_name=run_name,
        config=config,
        tasks=tasks,
        plan=plan,
        orchestration_manifest_path=manifest_path,
    )
    archived_outputs = archive_development_outputs(
        subject_root,
        output_root=output_root,
        current_run_name=run_name,
        current_tractography_backend=primary_tractography_backend,
        # Run folders are stable references for QC, bundles and scene loading.
        # Changing the backend must not move another completed run's files.
        archive_alternate_tractograms=False,
    )
    output_tree_audit = _audit_output_tree(layout, subject_id=config.subject_id, selected_targets=targets)
    output_tree_semantics = _write_output_tree_semantics(
        layout=layout,
        subject_id=config.subject_id,
        requested_targets=targets,
        run_targets=run_targets,
        tasks=tasks,
    )
    task_payloads = []
    for task in tasks:
        payload = {
            "name": task.name,
            "status": task.status,
            "reason": task.reason,
            "manifest_path": task.manifest_path.as_posix() if task.manifest_path else None,
            "qc_path": task.qc_path.as_posix() if task.qc_path else None,
            "outputs": task.outputs,
            "started_at": task.started_at,
            "ended_at": task.ended_at,
            "runtime_seconds": task.runtime_seconds,
            **_task_provenance(plan, task.name),
            **({"regional_statistics": task.regional_statistics} if task.regional_statistics else {}),
        }
        payload.update(
            {
                f"normalized_{key}": value
                for key, value in normalized_task_records.get(task.name, {}).items()
            }
        )
        task_payloads.append(payload)

    elapsed_seconds = time.monotonic() - start
    performance_profile = _build_performance_profile(
        layout=layout,
        run_name=run_name,
        tasks=tasks,
        orchestration_elapsed_seconds=elapsed_seconds,
    )
    phase_timings = dict(planning=planning_seconds, input_preparation_and_validation=input_seconds,
                         product_execution=execution_seconds, metadata_and_profile=time.monotonic() - metadata_started)

    manifest = {
        "kind": "slicerddsurfer_product_orchestration",
        "version": 1,
        "status": (
            "failed"
            if any(task.status in {"failed", "partial_failed"} for task in tasks)
            else "completed_with_blocks"
            if any(task.status == "blocked" for task in tasks)
            else "completed"
            if config.execute
            else "planned"
        ),
        "subject_id": config.subject_id,
        "subject_root": subject_root.as_posix(),
        "output_root": output_root.as_posix() if output_root is not None else layout.outputs.as_posix(),
        "targets": list(targets),
        "run_targets": list(run_targets),
        "closure": {"required_by": {key: list(value) for key, value in config.target_required_by.items()}},
        "run_plan_path": plan.run_plan_path.as_posix(),
        "previous_run_records": {key: value.as_posix() for key, value in previous_run_records.items()},
        "run_plan_summary": plan.to_dict()["summary"],
        "output_tree_audit_path": output_tree_audit.get("path"),
        "output_tree_warnings": output_tree_audit.get("warnings", []),
        "output_tree_semantics_path": output_tree_semantics.get("path"),
        "output_tree_semantics_summary": {
            "states": {
                state: sum(1 for entry in output_tree_semantics.get("entries", []) if entry.get("state") == state)
                for state in sorted({entry.get("state") for entry in output_tree_semantics.get("entries", [])})
            }
        },
        "archived_development_outputs": {key: value.as_posix() for key, value in archived_outputs.items()},
        "preflight_blocked_targets": preflight_blocked_targets,
        "input_validation_path": input_validation_path.as_posix() if input_validation_path else None,
        "input_blocked_targets": input_blocked_targets,
        "preflight_blocked_by_backend": config.preflight_blocked_by_backend,
        "run_name": run_name,
        "tractography_backend": config.tractography_backend,
        "bundle_tractography_backend": config.bundle_tractography_backend,
        "ukf_parameters": config.ukf_parameters,
        "ukf_executable": str(config.ukf_executable) if config.ukf_executable else None,
        "dwiconvert_executable": str(config.dwiconvert_executable) if config.dwiconvert_executable else None,
        "primary_tractography_backend": primary_tractography_backend,
        "bundle_backend": config.bundle_backend,
        "wmql_query_preset": config.wmql_query_preset,
        "wmql_atlas_mode": config.wmql_atlas_mode,
        "wma_atlas": config.wma_atlas,
        "wmql_query_selection": list(config.wmql_query_selection),
        "max_parallel_models": config.max_parallel_models,
        "bundle_backends": list(bundle_backends),
        "execute": config.execute,
        "resume": config.resume,
        "preprocessing_backend": config.preprocessing_backend,
        "brain_mask_backend": config.brain_mask_backend,
        "brain_mask_device": config.brain_mask_device,
        "brain_mask_configuration": {
            name: str(getattr(config, name)) if getattr(config, name) else None
            for name in ("pnl_mask_python", "pnl_mask_script", "pnl_mask_model_dir", "pnl_mask_ants_bin_dir")
        },
        "preprocessing_force": config.preprocessing_force,
        "preprocessing_dipy_denoise_method": config.preprocessing_dipy_denoise_method,
        "preprocessing_dipy_run_gibbs": config.preprocessing_dipy_run_gibbs,
        "preprocessing_dipy_run_motion_correction": config.preprocessing_dipy_run_motion_correction,
        "preprocessing_mrtrix_bin_dir": config.preprocessing_mrtrix_bin_dir.as_posix() if config.preprocessing_mrtrix_bin_dir else None,
        "preprocessing_fsl_bin_dir": config.preprocessing_fsl_bin_dir.as_posix() if config.preprocessing_fsl_bin_dir else None,
        "preprocessing_reverse_phase_b0": config.preprocessing_reverse_phase_b0.as_posix() if config.preprocessing_reverse_phase_b0 else None,
        "preprocessing_phase_encoding_direction": config.preprocessing_phase_encoding_direction,
        "preprocessing_readout_time": config.preprocessing_readout_time,
        "preprocessing_eddy_threads": config.preprocessing_eddy_threads,
        "preprocessing_eddy_second_level_model": config.preprocessing_eddy_second_level_model,
        "microstructure_fit_method": config.microstructure_fit_method,
        "microstructure_run_free_water": config.microstructure_run_free_water,
        "microstructure_run_noddi": config.microstructure_run_noddi,
        "microstructure_run_dki": config.microstructure_run_dki,
        "microstructure_dki_fit_method": config.microstructure_dki_fit_method,
        "microstructure_dki_execution": config.microstructure_dki_execution,
        "wma_num_threads": config.wma_num_threads,
        "microstructure_dki_matlab_executable": str(config.microstructure_dki_matlab_executable) if config.microstructure_dki_matlab_executable else None,
        "microstructure_dki_mkcurve_root": str(config.microstructure_dki_mkcurve_root) if config.microstructure_dki_mkcurve_root else None,
        "microstructure_free_water_chunk_size": config.microstructure_free_water_chunk_size,
        "microstructure_free_water_engine": config.microstructure_free_water_engine,
        "microstructure_free_water_n_jobs": config.microstructure_free_water_n_jobs,
        "microstructure_free_water_vox_per_chunk": config.microstructure_free_water_vox_per_chunk,
        "microstructure_free_water_max_fraction_for_tissue": config.microstructure_free_water_max_fraction_for_tissue,
        "microstructure_stages": [
            stage
            for stage, enabled in (
                ("dti", True),
                ("freewater", config.microstructure_run_free_water),
                ("noddi", config.microstructure_run_noddi),
                ("dki", config.microstructure_run_dki),
            )
            if enabled
        ],
        "postprocess_native_dir": config.postprocess_native_dir.expanduser().resolve().as_posix() if config.postprocess_native_dir else None,
        "postprocess_mri_dir": config.postprocess_mri_dir.expanduser().resolve().as_posix() if config.postprocess_mri_dir else None,
        "postprocess_backend_root": config.postprocess_backend_root.expanduser().resolve().as_posix() if config.postprocess_backend_root else None,
        "postprocess_mode": config.postprocess_mode,
        "postprocess_python_bin": _optional_interpreter(config.postprocess_python_bin).as_posix() if config.postprocess_python_bin else None,
        "postprocess_freesurfer_home": config.postprocess_freesurfer_home.expanduser().resolve().as_posix() if config.postprocess_freesurfer_home else None,
        "postprocess_runtime": config.postprocess_runtime.expanduser().resolve().as_posix() if config.postprocess_runtime else None,
        "postprocess_license_file": config.postprocess_license_file.expanduser().resolve().as_posix() if config.postprocess_license_file else None,
        "postprocess_atlases": list(config.postprocess_atlases) if config.postprocess_atlases else None,
        "postprocess_force": config.postprocess_force,
        "atlas_t2": atlas_t2.as_posix(),
        "ddseg_device": config.ddseg_device,
        "surface_device": config.surface_device,
        "surface_atlas_t2": config.surface_atlas_t2.expanduser().resolve().as_posix() if config.surface_atlas_t2 else None,
        "surface_python_bin": _optional_interpreter(config.surface_python_bin).as_posix() if config.surface_python_bin else None,
        "surface_precision": config.surface_precision,
        "surface_cpu_threads": config.surface_cpu_threads,
        "surface_max_gpu_gib": config.surface_max_gpu_gib,
        "resolved_ddseg_device": ddseg_device,
        "resolved_surface_device": surface_device,
        "select_streamlines": config.select_streamlines,
        "dti_min_length_mm": config.dti_min_length_mm,
        "dti_max_length_mm": config.dti_max_length_mm,
        "mrtrix_nthreads": config.mrtrix_nthreads,
        "mrtrix_random_seed": config.mrtrix_random_seed,
        "tractography_seeding": config.tractography_seeding,
        "ddseg_min_wm_dice": config.ddseg_min_wm_dice,
        "allow_anatomy_space_mismatch": config.allow_anatomy_space_mismatch,
        "allow_act_ddseg_only": config.allow_act_ddseg_only,
        "anatomy_input": "preprocessed DWI when preprocessing completed, else raw input (DDSeg/DDParcel/DDSurfer)",
        "connectome_backend": config.connectome_backend,
        "connectome_compress_mrtrix_dense": config.connectome_compress_mrtrix_dense,
        "saf_fs_dir": config.saf_fs_dir.expanduser().resolve().as_posix() if config.saf_fs_dir else None,
        "saf_pregenerated_tractogram": config.saf_pregenerated_tractogram.expanduser().resolve().as_posix() if config.saf_pregenerated_tractogram else None,
        "saf_surface_space": "native_dMRI",
        "saf_mrtrix_prefix": config.saf_mrtrix_prefix,
        "wma_input_tractography": wma_input_tractography.as_posix() if wma_input_tractography else None,
        "wma_atlas_dir": config.wma_atlas_dir.expanduser().resolve().as_posix() if config.wma_atlas_dir else None,
        "wma_root": config.wma_root.expanduser().resolve().as_posix() if config.wma_root else None,
        "slicer_executable": slicer_executable.as_posix() if slicer_executable else None,
        "wma_use_virtual_x": config.wma_use_virtual_x,
        "wma_export_diffusion_measures": config.wma_export_diffusion_measures,
        "fiber_tract_measurements_cli": config.fiber_tract_measurements_cli.expanduser().resolve().as_posix() if config.fiber_tract_measurements_cli else None,
        "tractcloud_input_tractography": tractcloud_input_tractography.as_posix() if tractcloud_input_tractography else None,
        "tractcloud_root": config.tractcloud_root.expanduser().resolve().as_posix() if config.tractcloud_root else None,
        "tractcloud_python_executable": _optional_interpreter(config.tractcloud_python_executable).as_posix() if config.tractcloud_python_executable else None,
        "tractcloud_data_dir": config.tractcloud_data_dir.expanduser().resolve().as_posix() if config.tractcloud_data_dir else None,
        "tractcloud_device": config.tractcloud_device,
        "resolved_tractcloud_device": tractcloud_device,
        "tractcloud_batch_size": config.tractcloud_batch_size,
        "tractcloud_random_seed": config.tractcloud_random_seed,
        "elapsed_seconds": elapsed_seconds,
        "phase_timings_seconds": phase_timings,
        "performance_profile_path": performance_profile.get("path"),
        "performance_profile": performance_profile,
        "tasks": task_payloads,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_started = time.monotonic()
    try:
        from shared.output_tree_report import write_output_tree_report

        output_tree_report_paths = write_output_tree_report(
            output_root=layout.outputs,
            destination=layout.slicer_scene_report / f"{run_name}_output_tree.html",
            latest_destination=layout.slicer_scene_report / "output_tree.html",
            subject_id=config.subject_id,
            run_name=run_name,
        )
        manifest["output_tree_report_paths"] = output_tree_report_paths.to_dict()
    except Exception as exc:  # pragma: no cover - report generation must not break pipeline execution
        manifest["output_tree_report_error"] = str(exc)
    phase_timings['output_tree_report'] = time.monotonic() - report_started
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    subject_report_started = time.monotonic()
    try:
        from shared.reporting import write_subject_reports

        report_paths = write_subject_reports(
            subject_root=subject_root,
            output_root=output_root,
            manifest_path=manifest_path,
            run_name=run_name,
        )
        manifest["report_paths"] = report_paths.to_dict()
        refreshed_output_tree_audit = _audit_output_tree(layout, subject_id=config.subject_id, selected_targets=targets)
        refreshed_output_tree_semantics = _write_output_tree_semantics(
            layout=layout,
            subject_id=config.subject_id,
            requested_targets=targets,
            run_targets=run_targets,
            tasks=tasks,
        )
        manifest["output_tree_audit_path"] = refreshed_output_tree_audit.get("path")
        manifest["output_tree_warnings"] = refreshed_output_tree_audit.get("warnings", [])
        manifest["output_tree_semantics_path"] = refreshed_output_tree_semantics.get("path")
        manifest["output_tree_semantics_summary"] = {
            "states": {
                state: sum(
                    1
                    for entry in refreshed_output_tree_semantics.get("entries", [])
                    if entry.get("state") == state
                )
                for state in sorted({entry.get("state") for entry in refreshed_output_tree_semantics.get("entries", [])})
            }
        }
    except Exception as exc:  # pragma: no cover - reporting must not break pipeline execution
        manifest["report_error"] = str(exc)
    phase_timings['subject_reports_and_refresh'] = time.monotonic() - subject_report_started
    manifest['wall_time_seconds'] = time.monotonic() - start
    manifest['wall_time_scope'] = 'Orchestrator planning through product execution and report generation; final telemetry write, CLI preflight and GUI loading are outside this interval.'
    performance_profile['phase_timings_seconds'] = dict(phase_timings)
    performance_profile['wall_time_seconds'] = manifest['wall_time_seconds']
    performance_profile['wall_time_scope'] = manifest['wall_time_scope']
    Path(performance_profile['path']).write_text(json.dumps(performance_profile, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ProductOrchestratorResult(
        subject_id=config.subject_id,
        subject_root=subject_root,
        run_manifest_path=manifest_path,
        tasks=tuple(tasks),
        run_plan_path=plan.run_plan_path,
    )


def run_subject_products(
    config: ProductOrchestratorConfig,
    *,
    log: Optional[Callable[[str], None]] = None,
) -> ProductOrchestratorResult:
    """Run products while preserving a failed manifest if any stage raises."""

    try:
        fresh_closure = not config.closure_applied  # Q1: multi-tracking children arrive already closed
        config = _with_closure(config)
        notice = closure_notice_for(config)
        if fresh_closure and notice and config.execute:
            # Plan-only runs keep stdout as pure JSON, so the notice is written for real runs only.
            _emit("[Targets] " + notice, log)
        _validate_bundle_tractography_source(config)
        if config.execute:
            if not config.subject_id.strip() or config.select_streamlines <= 0:
                raise ValueError("subject_id is required and select_streamlines must be positive")
            _normalize_targets(config)
            from shared.dicom_input import prepare_dicom_root
            prepare_dicom_root(config.subject_root, config.subject_id, dwiconvert=config.dwiconvert_executable, log=log)
        from shared.algorithm_selection import algorithms
        if len(algorithms(config.tractography_backend)) > 1:
            from shared.multi_tracking import run
            return run(config, log=log)
        return _run_subject_products_impl(config, log=log)
    except Exception as exc:
        _mark_orchestration_failed(config, exc)
        raise
