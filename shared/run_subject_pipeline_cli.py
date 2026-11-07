#!/usr/bin/env python3
"""Unified CLI entrypoint for subject-root SlicerDSpace execution."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import (
    EnvironmentDoctorConfig,
    EnvironmentDoctorResult,
    ProductOrchestratorConfig,
    blocked_target_reasons,
    plan_subject_products,
    run_environment_doctor,
    run_subject_products,
    normalize_orchestrator_targets,
    subject_layout,
    summarize_environment_doctor,
    write_environment_report,
)
from shared.preprocessing_route import add_removed_flags
from shared.run_cancellation import orchestration_manifest_path, write_cancelled_manifest
from shared.brain_mask import (
    BRAIN_MASK_BACKENDS, PNL_MASK_ROUTES, PNL_NATIVE_EXPERIMENTAL_REASON, unsupported_backend_message,
)
from shared.dependency_config import (
    DEPENDENCY_CONFIG_KEYS,
    add_dependency_config_arguments,
    apply_dependency_config,
    write_dependency_config,
)
from shared.python_environment import resolve_python_executable
from shared.user_config import (
    apply_user_config_to_args,
    coupled_explicit_cli_destinations,
    default_user_config_path,
    explicit_cli_destinations,
    load_user_config,
    user_config_from_cli_args,
    write_user_config,
)


DEFAULT_PRODUCT_TARGETS = (
    "preprocessing",
    "segmentation",
    "parcellation",
    "surface",
    "microstructure",
    "tractography",
    "bundles",
    "connectome",
)
OPTIONAL_ALGORITHM_TARGETS = (
    "microstructure_freewater",
    "microstructure_noddi",
    "microstructure_dki",
    "microstructure_ifw",
    "surface_postprocess",
    "saf",
    "wma",
    "tractcloud",
)
ALL_CLI_TARGETS = ("validation",) + DEFAULT_PRODUCT_TARGETS + OPTIONAL_ALGORITHM_TARGETS

def _split_csv_or_repeat(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    items: list[str] = []
    for value in values:
        for token in value.split(","):
            token = token.strip().lower()
            if token:
                items.append(token)
    return tuple(items)


def _split_csv_preserve_case(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    items: list[str] = []
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if token:
                items.append(token)
    return tuple(items)


def _resolve_microstructure_stages(args: argparse.Namespace) -> tuple[str, ...]:
    raw_values = _split_csv_or_repeat(args.microstructure_stages)
    if not raw_values:
        raw_values = ("dti",)
    expanded: list[str] = []
    aliases = {
        "dt": "dti",
        "dti": "dti",
        "fa": "dti",
        "fa_md": "dti",
        "free_water": "freewater",
        "freewater": "freewater",
        "fw": "freewater",
        "noddi": "noddi",
        "dki": "dki",
        "ifw": "ifw",
    }
    for value in raw_values:
        if value in {"all", "full"}:
            expanded.extend(("dti", "freewater", "noddi", "dki"))
            continue
        try:
            expanded.append(aliases[value])
        except KeyError as exc:
            raise ValueError("microstructure stages must contain only: dti, freewater, noddi, dki, ifw, all.") from exc
    if args.run_free_water:
        expanded.append("freewater")
    if args.run_noddi:
        expanded.append("noddi")
    if getattr(args, "run_dki", False) or "microstructure_dki" in _split_csv_or_repeat(args.targets):
        expanded.append("dki")
    if getattr(args, "run_ifw", False) or "microstructure_ifw" in _split_csv_or_repeat(args.targets):
        expanded.append("ifw")
    if "ifw" in expanded:
        expanded.append("freewater")  # iFW needs the Free Water fraction
    if any(stage in expanded for stage in ("freewater", "noddi", "dki", "ifw")):
        expanded.append("dti")
    return tuple(stage for stage in ("dti", "freewater", "noddi", "dki", "ifw") if stage in set(expanded))


def _infer_subject_id(subject_root: Path) -> str:
    from shared.subject_id import derive_subject_id
    return derive_subject_id(str(subject_root.resolve()))


def _apply_dependency_config(args: argparse.Namespace) -> None:
    apply_dependency_config(args, args.dependency_config)


def _write_dependency_config(args: argparse.Namespace, path: Path) -> Path:
    return write_dependency_config(args, path)


def _resolve_closure(args: argparse.Namespace):
    """Selected CLI targets with every prerequisite added (C1/C2), minus --skip-targets."""
    include = list(_split_csv_or_repeat(args.targets))
    exclude = set(_split_csv_or_repeat(args.skip_targets))

    if args.all:
        include = list(DEFAULT_PRODUCT_TARGETS)
    elif not include:
        include = ["microstructure", "tractography", "bundles", "connectome"]

    if args.with_validation:
        include = ["validation", *include]

    resolved: list[str] = []
    seen: set[str] = set()
    for target in include:
        if target == "all":
            for expanded in DEFAULT_PRODUCT_TARGETS:
                if expanded not in seen and expanded not in exclude:
                    resolved.append(expanded)
                    seen.add(expanded)
            continue
        if target not in seen and target not in exclude:
            resolved.append(target)
            seen.add(target)
    from shared.orchestrator import orchestrator_closure
    return orchestrator_closure(
        tuple(resolved),
        tractography_backend=args.tractography_backend,
        bundle_backend=args.bundle_backend,
        microstructure_run_free_water=bool(args.run_free_water),
        microstructure_run_noddi=bool(args.run_noddi),
        microstructure_run_dki=bool(getattr(args, "run_dki", False)),
        microstructure_run_ifw=bool(getattr(args, "run_ifw", False)),
        wma_source_explicit=bool(getattr(args, "bundle_tractography_backend", None)),
        provided_tractography=tuple(name for name, value in (
            ("wma", getattr(args, "wma_input_tractography", None)),
            ("tractcloud", getattr(args, "tractcloud_input_tractography", None)),
            ("saf", getattr(args, "saf_pregenerated_tractogram", None))) if value),
        excluded=tuple(exclude),  # K5: the closure never adds or expands a skipped target
    ), exclude


def _resolve_targets(args: argparse.Namespace) -> tuple[str, ...]:
    closure, exclude = _resolve_closure(args)
    return closure.targets


def resolve_analysis_options(args, *, explicit_dests=()):
    """Pure CLI/GUI experiment validation before any job writes outputs."""
    for field in ('dki_threads', 'dki_voxels_per_chunk', 'mkcurve_workers', 'mkcurve_voxels_per_chunk', 'wma_threads'):
        value = getattr(args, field, None)
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError('--' + field.replace('_', '-') + ' must be a positive integer')
    microstructure_stages = _resolve_microstructure_stages(args)
    args.run_free_water = 'freewater' in microstructure_stages
    args.run_noddi = 'noddi' in microstructure_stages
    args.run_dki = 'dki' in microstructure_stages
    args.run_ifw = 'ifw' in microstructure_stages
    for name in ('ifw_echo_time_ms', 'ifw_t2_gm_ms', 'ifw_t2_csf_ms'):
        value = getattr(args, name, None)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not value == value or value in (float('inf'), float('-inf'))
                                  or value < 0 or (name != 'ifw_echo_time_ms' and value <= 0)):
            raise ValueError('--' + name.replace('_', '-') + (' must be zero or a positive number' if name == 'ifw_echo_time_ms'
                                                              else ' must be a positive number'))
    if args.bundle_tractography_backend:
        from shared.algorithm_selection import algorithms
        if args.wma_input_tractography or args.tractcloud_input_tractography:
            raise ValueError('--bundle-tractography-backend cannot be combined with explicit WMA or TractCloud input files')
        if 'tractography' in _split_csv_or_repeat(args.skip_targets):
            raise ValueError('--bundle-tractography-backend requires tractography; remove --skip-targets tractography')
        tracking = list(algorithms(args.tractography_backend))
        source = args.bundle_tractography_backend
        if 'tractography_backend' not in explicit_dests:
            tracking = [source]
        elif source not in tracking:
            tracking.append(source)
        args.tractography_backend = ','.join(tracking)
    from tractography.ukf import parameters
    from tractography.wmql_protocol import CATALOG_PRESETS, canonical_wmql_selection
    parameters(json.loads(args.ukf_parameters))
    if args.wmql_query_preset in CATALOG_PRESETS:
        canonical_wmql_selection(args.wmql_query_selection, args.wmql_query_preset)
    closure, exclude = _resolve_closure(args)
    targets = closure.targets
    if not targets:
        raise ValueError('No targets selected. Use --all or --targets.')
    # C2/C3: later stages (doctor, config, cohort planner) see the closed algorithm set and provenance.
    args.closure_tractography_backend = closure.algorithms_text if 'tractography' in targets else args.tractography_backend
    args.target_required_by = {target: tuple(value) for target, value in closure.required_by.items() if target in targets}
    args.excluded_targets = tuple(sorted(exclude))
    # K4: what the user asked for (not the closed set) is what --save-user-config saves.
    args.requested_targets = tuple(target for target in closure.requested if target in targets)
    args.closure_algorithm_requesters = {key: tuple(value) for key, value in closure.added_algorithms.items()}
    if args.bundle_tractography_backend:
        if not set(targets).intersection({'bundles', 'wma', 'tractcloud'}):
            raise ValueError('--bundle-tractography-backend requires a bundle target')
        if 'tractography' not in targets:
            targets = (*targets, 'tractography')
    unsupported = sorted(set(targets) - set(ALL_CLI_TARGETS))
    if unsupported:
        raise ValueError(f'Unsupported targets: {", ".join(unsupported)}')
    return targets


def _environment_config_from_args(
    args: argparse.Namespace,
    *,
    targets: tuple[str, ...],
    output_root: Path | None,
) -> EnvironmentDoctorConfig:
    subject_root = args.subject_root.expanduser().resolve()
    layout = subject_layout(subject_root, output_root=output_root)
    from shared.environment import run_needs_brain_mask
    brain_mask_required = run_needs_brain_mask(subject_root, targets=targets,
        subject_id=args.subject_id, output_root=output_root,
        brain_mask_backend=args.brain_mask_backend,
        force=args.preprocessing_force, resume=not args.no_resume)
    return EnvironmentDoctorConfig(
        targets=targets,
        repo_root=REPO_ROOT,
        output_dir=args.doctor_output_dir.expanduser().resolve()
        if args.doctor_output_dir
        else layout.run_metadata / "environment",
        python_executable=args.python_executable or args.doctor_python_executable,
        python_path_entries=tuple(
            Path(entry).expanduser().resolve()
            for entry in sys.path
            if entry and Path(entry).expanduser().exists()
        ),
        check_slicer_cli_modules=True,
        python_environment=args.python_environment,
        conda_executable=args.conda_executable,
        brain_mask_backend=args.brain_mask_backend,
        brain_mask_device=args.brain_mask_device,
        pnl_mask_route=args.pnl_mask_route,
        brain_mask_required=bool(brain_mask_required),
        pnl_mask_python=args.pnl_mask_python,
        pnl_mask_script=args.pnl_mask_script,
        pnl_mask_model_dir=args.pnl_mask_model_dir,
        pnl_mask_ants_bin_dir=args.pnl_mask_ants_bin_dir,
        tractography_backend=getattr(args, 'closure_tractography_backend', None) or args.tractography_backend,
        ukf_executable=getattr(args, 'ukf_executable', None),
        dwiconvert_executable=getattr(args, 'dwiconvert_executable', None),
        connectome_backend=args.connectome_backend,
        ddseg_device=args.ddseg_device,
        surface_device=args.surface_device,
        tractcloud_device=args.tractcloud_device,
        run_free_water=bool(args.run_free_water),
        run_noddi=bool(args.run_noddi),
        run_dki=bool(getattr(args, "run_dki", False)),
        dki_fit_method=getattr(args, "dki_fit_method", "MKCURVE"),
        dki_matlab_executable=getattr(args, "dki_matlab_executable", None),
        dki_mkcurve_root=getattr(args, "dki_mkcurve_root", None),
        mrtrix_bin_dir=args.mrtrix_bin_dir,
        preprocessing_mrtrix_bin_dir=args.preprocessing_mrtrix_bin_dir,
        preprocessing_fsl_bin_dir=args.preprocessing_fsl_bin_dir,
        preprocessing_execution_backend=getattr(args, "preprocessing_execution_backend", None),
        preprocessing_reverse_phase=bool(getattr(args, "preprocessing_reverse_phase_b0", None)),
        postprocess_freesurfer_home=args.postprocess_freesurfer_home,
        postprocess_runtime=args.postprocess_runtime,
        postprocess_license_file=args.postprocess_license_file,
        model_assets_dir=args.model_assets_dir,
        wma_root=args.wma_root,
        wma_atlas_dir=args.wma_atlas_dir,
        wma_atlas=args.wma_atlas,
        wma_export_diffusion_measures=bool(getattr(args, "wma_export_diffusion_measures", False)),
        slicer_executable=args.slicer_executable,
        fiber_tract_measurements_cli=args.fiber_tract_measurements_cli,
        tract_querier_root=args.tract_querier_root,
        wmql_python_executable=args.wmql_python_executable,
        tractcloud_root=args.tractcloud_root,
        tractcloud_python_executable=args.tractcloud_python_executable,
        tractcloud_data_dir=args.tractcloud_data_dir,
    )


def _run_environment_doctor_from_args(
    args: argparse.Namespace,
    *,
    targets: tuple[str, ...],
    output_root: Path | None,
) -> tuple[EnvironmentDoctorResult, dict[str, str]]:
    config = _environment_config_from_args(args, targets=targets, output_root=output_root)
    result = run_environment_doctor(config)
    output_dir = config.output_dir
    if output_dir is None:
        output_dir = subject_layout(args.subject_root.expanduser().resolve(), output_root=output_root).run_metadata / "environment"
    written = write_environment_report(result, output_dir)
    return result, {key: path.as_posix() for key, path in written.items()}


def _print_run_plan_summary(plan, *, stream) -> None:
    summary = plan.to_dict()["summary"]
    print(
        "Run plan: "
        f"{summary['will_run']} will run, {summary['skipped']} skipped/reused, {summary['blocked']} blocked.",
        file=stream,
    )
    for item in plan.items:
        detail = f" ({item.reason})" if item.reason else ""
        print(f"- {item.target}: {item.status}{detail}", file=stream)


def _make_sigterm_handler(manifest_path: Path, *, subject_id: str, run_name: str):
    """Return a signal handler that writes a terminal "cancelled" manifest.

    Installed via ``signal.signal(signal.SIGTERM, handler)`` before
    ``run_subject_products()`` starts, so that if the whole process group is
    killed (see shared/process_group.py), the manifest ends up recording
    "cancelled" instead of being left at "running" or never written.
    Returned as a plain function (rather than only installed inline) so it
    can be called directly in tests with a fake manifest path, without
    sending a real signal.
    """

    def _handle_sigterm(signum, frame) -> None:
        write_cancelled_manifest(
            manifest_path,
            subject_id=subject_id,
            run_name=run_name,
            cancelled_at=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        )
        sys.exit(143)

    return _handle_sigterm


def _cli_log(message: str) -> None:
    """Run log: the closure notice goes to stderr so stdout stays the JSON result for blocked runs (GUI merges both).

    W2b: so do the notice that no current preprocessed DWI is used and the rule-4 phase-encoding block message.
    """
    to_stderr = message.startswith(("[Targets]", "[Orchestrator] No current preprocessed DWI", "[Orchestrator] Phase-encoding"))
    print(message, file=sys.stderr if to_stderr else sys.stdout)


def _phase_encoding_block(plan) -> str:
    """W2b rule 4: the message of a preprocessing plan item that is blocked for want of a PE direction, else ''."""
    for item in getattr(plan, "items", ()) or ():
        if item.status == "blocked" and item.reason_category == "phase_encoding":
            return item.reason
    return ""


def _brain_mask_backend_argument(value: str) -> str:
    """W5 / SC-2: a removed or unknown brain mask backend fails with a message that names pnl_cnn and provided."""
    if value not in BRAIN_MASK_BACKENDS:
        raise argparse.ArgumentTypeError(unsupported_backend_message(value))
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run SlicerDSpace from one subject-root entrypoint. "
            "Use --all for the full product tree, or --targets/--skip-targets to customize."
        )
    )
    parser.add_argument(
        "--subject-root",
        type=Path,
        help="Subject root directory. Defaults to the most recently saved user-config path when available.",
    )
    parser.add_argument("--subject-id", help="Optional subject id. Defaults to the subject-root directory name.")
    parser.add_argument("--output-root", type=Path, help="Optional output root. Defaults to <subject-root>/outputs.")
    parser.add_argument("--run-name", default="Test1", help="Run name used for tractography/connectome subtrees.")
    parser.add_argument(
        "--user-config",
        type=Path,
        help=(
            "Shared GUI/CLI user configuration JSON. Defaults to the platform-specific SlicerDSpace config path "
            "or $SLICERDSPACE_USER_CONFIG."
        ),
    )
    parser.add_argument("--no-user-config", action="store_true", help="Do not load saved GUI/CLI defaults.")
    parser.add_argument(
        "--save-user-config",
        nargs="?",
        const="__default__",
        metavar="PATH",
        help="Save resolved paths and run defaults to PATH, or to the default user config when PATH is omitted.",
    )
    parser.add_argument("--print-user-config", action="store_true", help="Print the loaded shared user configuration and exit.")
    parser.add_argument(
        "--language",
        choices=("en", "zh"),
        help="Slicer beginner-UI language stored when --save-user-config is used.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run the full product tree: segmentation, parcellation, surface, microstructure, tractography, bundles, connectome.",
    )
    parser.add_argument(
        "--targets",
        action="append",
        help=(
            "Targets to include. May be repeated or comma-separated. "
            f"Choices: {', '.join(ALL_CLI_TARGETS)}."
        ),
    )
    parser.add_argument(
        "--skip-targets",
        action="append",
        help="Targets to skip. May be repeated or comma-separated.",
    )
    parser.add_argument("--with-validation", action="store_true", help="Also run shared input validation.")
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Check local environment for selected targets, write install plan, and exit before running products.",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Run environment preflight, write a target run plan, print it as JSON, and exit before execution.",
    )
    parser.add_argument(
        "--doctor-output-dir",
        type=Path,
        help="Optional environment doctor output directory. Defaults to <outputs>/00_run_metadata/environment.",
    )
    parser.add_argument(
        "--dependency-config",
        type=Path,
        help=(
            "Optional JSON file containing saved external dependency paths. "
            "Explicit CLI path arguments override this file."
        ),
    )
    parser.add_argument(
        "--save-dependency-config",
        type=Path,
        help="Write the resolved external dependency path configuration to this JSON file.",
    )
    parser.add_argument(
        "--python-executable",
        type=Path,
        help="Optional Python executable to use for dependency checks. Defaults to the current Python.",
    )
    parser.add_argument(
        "--python-environment",
        help="Optional conda environment name or Python executable path for dependency checks.",
    )
    parser.add_argument(
        "--conda-executable",
        type=Path,
        help="Optional conda or mamba executable used to resolve --python-environment names.",
    )
    parser.add_argument(
        "--doctor-python-executable",
        type=Path,
        help="Deprecated alias for --python-executable.",
    )
    parser.add_argument(
        "--skip-environment-check",
        action="store_true",
        help="Skip the pre-execution dependency doctor. Only use when running in a known-good managed environment.",
    )
    parser.add_argument(
        "--ignore-environment-blockers",
        action="store_true",
        help="Deprecated compatibility flag. Partial execution is now the default.",
    )
    parser.add_argument(
        "--strict-environment",
        action="store_true",
        help="Abort before execution when any selected target has missing required dependencies.",
    )
    parser.add_argument("--execute", action="store_true", help="Actually execute workflows. Default is dry-run planning only.")
    parser.add_argument("--no-resume", action="store_true", help="Recompute every stage of this run, including prerequisites added automatically.")
    parser.add_argument(
        "--brain-mask-backend",
        default="pnl_cnn",
        type=_brain_mask_backend_argument,
        choices=BRAIN_MASK_BACKENDS,
        help=(
            "Brain mask backend (default: pnl_cnn). Missing dependencies are reported without falling back. "
            "provided requires an existing input mask. synthstrip, mrtrix and dipy were removed (W5)."
        ),
    )
    parser.add_argument("--brain-mask-device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="PNL device: auto prefers the first visible GPU in that runtime; cuda requires GPU; cpu disables CUDA.")
    parser.add_argument("--pnl-mask-route", choices=PNL_MASK_ROUTES, default="auto",
                        help=("Route of the pnl_cnn brain mask. auto: managed on Linux (conda runner with ANTs), native on macOS and Windows. "
                              "managed: the conda runner. native (experimental): SimpleITK registration in Slicer's Python; it "
                              + PNL_NATIVE_EXPERIMENTAL_REASON + "."))
    # W2a (D1): preprocessing has one route. The three removed flags stay registered only to fail with the message.
    add_removed_flags(parser)
    parser.add_argument("--preprocessing-force", action="store_true", help="Force rerun preprocessing even if outputs exist.")
    parser.add_argument("--preprocessing-skip-gibbs", action="store_true",
                        help="Skip DIPY Gibbs removal (for partial-Fourier data). Preprocessing has one route: DIPY MP-PCA, "
                             "DIPY Gibbs removal, FSL eddy/topup through dwifslpreproc.")
    parser.add_argument("--preprocessing-mrtrix-bin-dir", type=Path, help="Optional MRtrix3 bin dir for preprocessing.")
    parser.add_argument("--preprocessing-fsl-bin-dir", type=Path, help="Optional FSL bin dir for preprocessing.")
    parser.add_argument("--preprocessing-execution-backend", choices=("legacy", "native"),
                        help="FSL correction runner. Default: legacy. Native uses the shared experimental CPU runner.")
    parser.add_argument("--preprocessing-reverse-phase-b0", type=Path, help="Optional reverse phase-encoding b0 for MRtrix/FSL preprocessing.")
    parser.add_argument("--preprocessing-phase-encoding-direction", help="Phase-encoding direction for preprocessing: AP, PA, LR, RL, IS, SI or a NIfTI axis code (i, i-, j, j-, k, k-). Without it the direction comes from input/dwi.json (PhaseEncodingDirection); if neither has it, preprocessing is blocked (most axial clinical DWI: AP).")
    parser.add_argument("--preprocessing-readout-time", type=float, help="Total readout time in seconds for preprocessing. Without it: TotalReadoutTime from input/dwi.json, else dwifslpreproc assumes 0.1 s (recorded as readout_assumed).")
    parser.add_argument(
        "--preprocessing-eddy-threads",
        type=int,
        help="CPU thread count for FSL eddy. Defaults to the available CPU count, capped at 8.",
    )
    parser.add_argument(
        "--preprocessing-eddy-slm",
        choices=("none", "linear", "quadratic"),
        help=(
            "FSL eddy second-level EC model. Use linear when MRtrix reports "
            "strongly asymmetric diffusion-direction sampling."
        ),
    )
    parser.add_argument(
        "--tractography-backend",
        default="csd_act",
        help="One or comma-separated tracking backends: csd_act, ukf, dti_det, saf. Multiple choices use isolated run variants.",
    )
    parser.add_argument('--ukf-parameters', default='{}',
                        help='JSON overrides of official UKF defaults; never applies CSD streamline budgets.')
    parser.add_argument('--ukf-executable', type=Path,
                        help='Optional installed UKFTractography CLI; GUI discovers the extension automatically.')
    parser.add_argument('--dwiconvert-executable', type=Path,
                        help='Optional official DWIConvert CLI; GUI forwards its discovered path automatically.')
    parser.add_argument(
        "--bundle-backend",
        default="wmql",
        help=(
            "Bundle backend selection. Use wmql, wma, tractcloud, all, or comma-separated values. "
            "When target bundles is selected this expands to the concrete backend targets."
        ),
    )
    parser.add_argument(
        "--bundle-tractography-backend",
        choices=("csd_act", "ukf", "dti_det"),
        help=("Use this whole-brain tracking backend as the source for all selected bundle methods. "
              "The source is run automatically; SAF is a short-fibre product and is not a whole-brain input."),
    )
    parser.add_argument(
        "--connectome-backend",
        default="auto",
        choices=("auto", "python", "mrtrix"),
        help=("Connectome backend. auto: MRtrix tck2connectome for a .tck source with a streamline-count "
              "metric, Python for any other source or metric; no silent fallback."),
    )
    parser.add_argument('--max-parallel-models', type=int, choices=(1,2,3), default=1,
                        help='Independent Free-water/NODDI/DKI processes; resource-bounded; default sequential.')
    parser.add_argument("--wmql-query-preset", choices=("official", "cerebellar", "official_cerebellar", "ddparcel_act", "author", "published_endpoints"), default="official",
                        help="official: 48 unchanged author outputs (default); official_cerebellar: add three regional cerebellar selections; cerebellar: regional selections only. Remaining choices are historical protocols.")
    parser.add_argument('--wmql-query-selection', default='',
                        help='Comma-separated queries from the selected preset, e.g. AF,CST or cerebellar_brainstem. Empty/all runs the selected preset in full.')
    parser.add_argument('--wmql-atlas-mode', choices=('native', 'ddparcel_wm'), default='native',
                        help='native retains the supplied labels (default). ddparcel_wm explicitly enables experimental 5-mm adjacent-WM territories; native DDParcel output is unchanged and anatomical review remains required.')
    parser.add_argument(
        "--select-streamlines",
        type=int,
        default=1_000_000,
        help="Requested streamline count for tractography. Defaults to 1,000,000 for analysis-scale bundle/connectome coverage.",
    )
    parser.add_argument(
        "--dti-min-length-mm",
        type=float,
        default=10.0,
        help="Minimum retained DTI deterministic streamline length in millimetres.",
    )
    parser.add_argument(
        "--dti-max-length-mm",
        type=float,
        default=250.0,
        help="Maximum retained DTI deterministic streamline length in millimetres.",
    )
    parser.add_argument(
        "--mrtrix-nthreads",
        type=int,
        default=None,
        help=(
            "Optional: pin MRtrix3's own -nthreads for tckgen/tcksift2. Off (unset) by default, "
            "meaning MRtrix3 uses its own default thread count. NOTE: pinning this does NOT by "
            "itself make tckgen bit-reproducible -- multi-threaded track generation is not "
            "bit-reproducible even with a fixed random seed; only -nthreads 0 is."
        ),
    )
    parser.add_argument(
        "--mrtrix-seed",
        type=int,
        default=None,
        help=(
            "Optional MRtrix3 RNG seed (exported as MRTRIX_RNG_SEED). When set, tckgen runs with "
            "-nthreads 0 so the tractogram is bit-reproducible (slower); tcksift2 keeps --mrtrix-nthreads."
        ),
    )
    parser.add_argument(
        "--tckgen-seeding",
        default="dynamic",
        choices=("dynamic", "gmwmi"),
        help="CSD/ACT seeding: dynamic (default, -seed_dynamic on the WM FOD) or gmwmi (-seed_gmwmi).",
    )
    parser.add_argument(
        "--ddseg-min-wm-dice",
        type=float,
        default=0.70,
        help="World-space QC gate: minimum Dice of DDSeg WM vs DDParcel WM before ACT may run.",
    )
    parser.add_argument(
        "--allow-act-ddseg-only",
        action="store_true",
        help="Explicit exploratory ACT fallback without DDParcel; records missing subcortical GM priors in QC. Disabled by default.",
    )
    parser.add_argument(
        "--allow-anatomy-space-mismatch",
        action="store_true",
        help="Allow ACT anatomy derived from a different DWI than the CSD input (not recommended).",
    )
    parser.add_argument(
        "--micro-fit-method",
        default="wls",
        choices=("wls", "ls"),
        help="DTI fit method used by microstructure and DTI tracking.",
    )
    parser.add_argument(
        "--microstructure-stages",
        action="append",
        help=(
            "Microstructure stages to run. May be repeated or comma-separated. "
            "Choices: dti, freewater, noddi, dki, ifw, all. Defaults to dti only; optional stages are independent. "
            "ifw is not part of all: it needs the echo time and also runs Free Water and DDSeg."
        ),
    )
    parser.add_argument("--run-free-water", action="store_true", help="Enable Free Water stage under microstructure.")
    parser.add_argument("--run-noddi", action="store_true", help="Enable NODDI stage under microstructure.")
    parser.add_argument("--run-dki", action="store_true", help="Enable the optional DKI stage under microstructure.")
    parser.add_argument("--run-ifw", action="store_true", help="Enable the iFW stage (also runs Free Water and DDSeg).")
    parser.add_argument(
        "--ifw-echo-time-ms", type=float, default=None,
        help="iFW echo time in ms. Without it the stage reads EchoTime from the DWI JSON sidecar, else it stops. No default.",
    )
    parser.add_argument("--ifw-t2-gm-ms", type=float, default=95.0,
                        help="iFW tissue T2 in ms (Cho et al. preprint, Suppl. Methods 1: 95).")
    parser.add_argument("--ifw-t2-csf-ms", type=float, default=1500.0,
                        help="iFW CSF T2 in ms (Cho et al. preprint, Suppl. Methods 1: 1500).")
    parser.add_argument(
        "--dki-fit-method", choices=("WLS", "OLS", "MKCURVE", "MKCURVE_MATLAB"), default="MKCURVE",
        help="DKI: WLS/OLS, bundled Python MKCURVE, or optional author reference MKCURVE_MATLAB. No fallback.",
    )
    parser.add_argument(
        "--microstructure-free-water-chunk-size",
        type=int,
        default=2,
        help=(
            "Z-slice chunk size for DIPY Free Water fitting. Smaller values write progress more often; "
            "defaults to 2."
        ),
    )
    parser.add_argument('--dki-threads', type=int, help='Numerical threads per DKI/MK-Curve worker; automatic reservation if omitted.')
    parser.add_argument('--dki-voxels-per-chunk', type=int, default=1024, help='Bounded standard DKI fit block (default: 1024).')
    parser.add_argument('--mkcurve-workers', type=int, default=1, help='Independent MK-Curve voxel workers (default: 1).')
    parser.add_argument('--mkcurve-voxels-per-chunk', type=int, default=64, help='MK-Curve voxel block (default: 64).')
    parser.add_argument('--mkcurve-chunk-schedule', choices=('dynamic', 'ordered'), default='dynamic',
                        help='Refill voxel work on completion; ordered is retained for controlled comparisons.')
    parser.add_argument('--wma-threads', type=int, help='WMA author worker count; otherwise use the admitted stage reservation.')
    parser.add_argument(
        "--microstructure-free-water-engine",
        default="serial",
        choices=("serial", "joblib", "dask", "ray"),
        help="DIPY Free Water fit engine. Use joblib/dask/ray for explicit parallel long regressions.",
    )
    parser.add_argument(
        "--microstructure-free-water-n-jobs",
        type=int,
        help="Worker count passed to DIPY Free Water parallel engines. Omit for DIPY default.",
    )
    parser.add_argument(
        "--microstructure-free-water-vox-per-chunk",
        type=int,
        help="Voxel chunk size passed to DIPY Free Water parallel engines. Omit for DIPY default.",
    )
    parser.add_argument(
        "--microstructure-free-water-max-fraction-for-tissue",
        type=float,
        default=0.7,
        help=(
            "Maximum fitted free-water fraction retained for corrected tissue FA/MD. "
            "Defaults to DIPY's reference threshold of 0.7."
        ),
    )
    parser.add_argument("--postprocess-native-dir", type=Path, help="DDSurfer indexed native RAS OBJ directory (legacy *_orig.stl also supported).")
    parser.add_argument("--postprocess-mri-dir", type=Path, help="Optional native brain.mgz directory; wmparc is required only by the legacy backend.")
    parser.add_argument("--postprocess-backend-root", type=Path, help="Optional override for bundled DDSurfer postprocess backend.")
    parser.add_argument("--postprocess-mode", default="whole", choices=("whole", "left", "right"), help="DDSurfer postprocess hemisphere mode.")
    parser.add_argument("--postprocess-python-bin", type=Path, help="Optional Python interpreter for DDSurfer postprocess backend.")
    parser.add_argument(
        "--postprocess-freesurfer-home",
        type=Path,
        help="Optional FreeSurfer home for DDSurfer postprocessing.",
    )
    parser.add_argument("--postprocess-atlases", action="append", help="Optional comma-separated atlas list for DDSurfer postprocess.")
    parser.add_argument("--postprocess-force", action="store_true", help="Force restaging DDSurfer postprocess inputs.")
    parser.add_argument("--mrtrix-bin-dir", type=Path, help="Optional MRtrix3 bin directory.")
    parser.add_argument("--tract-querier-root", type=Path, help="Optional TractQuerier checkout root.")
    parser.add_argument("--atlas-t2", type=Path, help="Optional DDParcel atlas T2 path override.")
    parser.add_argument("--surface-atlas-t2", type=Path, help="Matching DDSurfer 1-mm atlas override (separate from DDParcel).")
    parser.add_argument("--surface-python-bin", type=Path, help="Optional isolated DDSurfer Python runtime; also supports DDSURFER_PYTHON_BIN.")
    parser.add_argument("--surface-precision", choices=('auto', 'bf16', 'fp32'), default='fp32')
    parser.add_argument("--surface-cpu-threads", type=int, default=None,
                        help="DDSurfer inference CPU thread budget. Default: the stage CPU reservation of a cohort run, else 4.")
    parser.add_argument("--surface-max-gpu-gib", type=float, default=24., help="DDSurfer inference GPU allocation budget in GiB.")
    parser.add_argument("--ddseg-device", default="auto", help="DDSeg torch device. auto uses cuda:0 when available, otherwise cpu.")
    parser.add_argument("--surface-device", default="auto", help="DDSurfer torch device. auto uses cuda:0 when available, otherwise cpu.")
    parser.add_argument(
        "--compress-mrtrix-dense",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=("Compress MRtrix dense connectome CSV into ROI-label edge list. "
              "Not set: on for the auto backend with the MRtrix path, off otherwise."),
    )
    parser.add_argument(
        "--print-manifest-only",
        action="store_true",
        help="Print only the orchestration manifest path instead of the full JSON result.",
    )
    parser.add_argument(
        "--saf-fs-dir",
        type=Path,
        help="Optional FreeSurfer/FastSurfer directory for SAF. Defaults to completed whole DDSurfer postprocess outputs when available.",
    )
    parser.add_argument("--saf-pregenerated-tractogram", type=Path, help="Optional pregenerated tractogram for SAF.")
    parser.add_argument("--saf-tckgen-options", default="", help="Optional MRtrix tckgen options string for SAF when it generates its own tractogram.")
    parser.add_argument("--saf-mrtrix-prefix", default="", help="Optional command prefix for SAF MRtrix calls.")
    parser.add_argument(
        "--wma-input-tractography",
        type=Path,
        help=(
            "Whole-brain VTK/VTP/TCK input for WMA ORG clustering. UKF tractography is recommended; "
            "the connectome CSD-ACT tractogram is not reused automatically because WMA requires "
            "broad long-range bundle coverage."
        ),
    )
    parser.add_argument("--wma-atlas", choices=("org", "east-west", "naba"), default="org",
                        help="WMA atlas family: ORG (default), East-West adults, or NABA neonates/adults.")
    parser.add_argument("--wma-atlas-dir", type=Path, help="Optional directory for the selected WMA atlas.")
    parser.add_argument("--wma-root", type=Path, help="Optional WhiteMatterAnalysis checkout root.")
    parser.add_argument("--slicer-executable", type=Path, help="Slicer executable used by WMA.")
    parser.add_argument("--wma-use-virtual-x", action="store_true",
                        help="Request xvfb-run for WMA; selected automatically on headless Linux.")
    parser.add_argument("--wma-export-diffusion-measures", action="store_true", help="Enable WMA diffusion measurements export.")
    parser.add_argument("--fiber-tract-measurements-cli", type=Path, help="Optional SlicerDMRI FiberTractMeasurements CLI for WMA.")
    parser.add_argument("--tractcloud-input-tractography", type=Path, help="Optional VTK/VTP tractography input for TractCloud.")
    parser.add_argument("--tractcloud-root", type=Path, help="Optional TractCloud source checkout root.")
    parser.add_argument("--tractcloud-python-executable", type=Path, help="Optional Python executable for TractCloud.")
    parser.add_argument("--tractcloud-data-dir", type=Path, help="Optional model/data directory for TractCloud.")
    parser.add_argument("--tractcloud-device", default="auto", choices=("auto", "cpu", "cuda"), help="TractCloud device.")
    parser.add_argument("--tractcloud-batch-size", type=int, default=2048, help="TractCloud batch size.")
    add_dependency_config_arguments(
        parser,
        exclude=(
            "python_executable",
            "python_environment",
            "conda_executable",
            "mrtrix_bin_dir",
            "preprocessing_mrtrix_bin_dir",
            "preprocessing_fsl_bin_dir",
            "postprocess_freesurfer_home",
            "wma_root",
            "wma_atlas_dir",
            "slicer_executable",
            "fiber_tract_measurements_cli",
            "tract_querier_root",
            "tractcloud_root",
            "tractcloud_python_executable",
            "tractcloud_data_dir",
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    cli_started = time.monotonic()
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    explicit_dests = coupled_explicit_cli_destinations(
        explicit_cli_destinations(parser, raw_argv)
    )
    args = parser.parse_args(raw_argv)
    try:
        config_path = args.user_config.expanduser() if args.user_config else default_user_config_path()
        user_config = load_user_config(config_path) if not args.no_user_config else None
        if args.print_user_config:
            print(json.dumps(user_config or {}, indent=2, ensure_ascii=False, sort_keys=True))
            return 0
        _apply_dependency_config(args)
        dependency_profile_dests = {
            key
            for key in DEPENDENCY_CONFIG_KEYS
            if args.dependency_config is not None and getattr(args, key, None) is not None
        }
        if user_config is not None:
            apply_user_config_to_args(
                args,
                user_config,
                explicit_dests=explicit_dests | dependency_profile_dests,
            )
        if 'wma_atlas_dir' not in explicit_dests:
            from shared.optional_assets import inherited_wma_atlas_directory
            args.wma_atlas_dir = inherited_wma_atlas_directory(args.wma_atlas_dir, args.wma_atlas,
                                                               args.model_assets_dir)
        if args.python_environment and args.python_executable is None:
            args.python_executable = resolve_python_executable(
                python_environment=args.python_environment,
                conda_executable=args.conda_executable,
            )
    except Exception as exc:
        parser.error(str(exc))
    if args.brain_mask_backend not in BRAIN_MASK_BACKENDS:
        parser.error(unsupported_backend_message(args.brain_mask_backend))
    if args.subject_root is None:
        parser.error(
            "--subject-root is required unless a recent subject_root is available in the shared user config. "
            f"Current config path: {config_path}"
        )
    if args.save_dependency_config:
        written_config = _write_dependency_config(args, args.save_dependency_config)
        print(f"Dependency config written: {written_config}", file=sys.stderr)

    try:
        targets = resolve_analysis_options(args, explicit_dests=explicit_dests)
    except ValueError as exc:
        parser.error(str(exc))

    subject_root = args.subject_root.expanduser().resolve()
    subject_id = (args.subject_id or _infer_subject_id(subject_root)).strip()
    args.subject_root = subject_root
    args.subject_id = subject_id
    output_root = args.output_root.expanduser().resolve() if args.output_root else None
    args.output_root = output_root
    if args.save_user_config:
        save_path = config_path if args.save_user_config == "__default__" else Path(args.save_user_config).expanduser()
        saved_payload = user_config_from_cli_args(
            args,
            language=args.language or str((user_config or {}).get("ui", {}).get("language") or "en"),
            base_config=user_config,
            resolved_targets=getattr(args, "requested_targets", None) or targets,
            resolved_stages=_resolve_microstructure_stages(args),
        )
        written = write_user_config(saved_payload, save_path)
        print(f"User config written: {written}", file=sys.stderr)

    if args.doctor:
        doctor_result, doctor_paths = _run_environment_doctor_from_args(
            args,
            targets=targets,
            output_root=output_root,
        )
        print(
            json.dumps(
                {
                    "environment": doctor_result.to_dict(),
                    "report_paths": doctor_paths,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    preflight_blocked_targets: dict[str, str] = {}
    doctor_result: EnvironmentDoctorResult | None = None
    doctor_paths: dict[str, str] = {}
    environment_started = time.monotonic()
    if (args.execute or args.plan) and not args.skip_environment_check:
        doctor_result, doctor_paths = _run_environment_doctor_from_args(
            args,
            targets=targets,
            output_root=output_root,
        )
        print(summarize_environment_doctor(doctor_result), file=sys.stderr)
        print(f"Environment report: {doctor_paths['json']}", file=sys.stderr)
        print(f"Python install helper: {doctor_paths['python_sh']}", file=sys.stderr)
        print(f"External tool notes: {doctor_paths['external_notes']}", file=sys.stderr)
        preflight_blocked_targets = {
            target: reason
            for target, reason in blocked_target_reasons(doctor_result).items()
            if target in targets
        }
        if preflight_blocked_targets and args.strict_environment and args.execute:
            print(
                "Aborting before execution because required dependencies are missing. "
                "Install/configure them, rerun --doctor, or omit --strict-environment to run available targets only.",
                file=sys.stderr,
            )
            return 2
        if preflight_blocked_targets:
            mode = "execution" if args.execute else "planning"
            print(
                f"Continuing with partial {mode}. Blocked target(s): "
                + "; ".join(f"{target}: {reason}" for target, reason in preflight_blocked_targets.items()),
                file=sys.stderr,
            )

    environment_seconds = time.monotonic() - environment_started
    config = ProductOrchestratorConfig(
        subject_id=subject_id,
        subject_root=subject_root,
        output_root=output_root,
        targets=targets,
        run_name=args.run_name,
        tractography_backend=getattr(args, 'closure_tractography_backend', None) or args.tractography_backend,
        bundle_backend=args.bundle_backend,
        bundle_tractography_backend=args.bundle_tractography_backend,
        ukf_parameters=json.loads(args.ukf_parameters),
        ukf_executable=args.ukf_executable,
        dwiconvert_executable=args.dwiconvert_executable,
        execute=bool(args.execute),
        wmql_query_preset=args.wmql_query_preset,
        wmql_atlas_mode=args.wmql_atlas_mode,
        wmql_query_selection=tuple(value.strip() for value in args.wmql_query_selection.split(',') if value.strip()),
        max_parallel_models=args.max_parallel_models,
        resume=not args.no_resume,
        brain_mask_backend=args.brain_mask_backend,
        brain_mask_device=args.brain_mask_device,
        pnl_mask_route=args.pnl_mask_route,
        pnl_mask_python=args.pnl_mask_python,
        pnl_mask_script=args.pnl_mask_script,
        pnl_mask_model_dir=args.pnl_mask_model_dir,
        pnl_mask_ants_bin_dir=args.pnl_mask_ants_bin_dir,
        preprocessing_force=bool(args.preprocessing_force),
        preprocessing_dipy_run_gibbs=not args.preprocessing_skip_gibbs,
        preprocessing_mrtrix_bin_dir=args.preprocessing_mrtrix_bin_dir,
        preprocessing_fsl_bin_dir=args.preprocessing_fsl_bin_dir,
        preprocessing_execution_backend=args.preprocessing_execution_backend,
        preprocessing_reverse_phase_b0=args.preprocessing_reverse_phase_b0,
        preprocessing_phase_encoding_direction=args.preprocessing_phase_encoding_direction,
        preprocessing_readout_time=args.preprocessing_readout_time,
        preprocessing_eddy_threads=args.preprocessing_eddy_threads,
        preprocessing_eddy_second_level_model=args.preprocessing_eddy_slm,
        microstructure_fit_method=args.micro_fit_method,
        microstructure_run_free_water=bool(args.run_free_water),
        microstructure_run_noddi=bool(args.run_noddi),
        microstructure_run_dki=bool(args.run_dki),
        microstructure_run_ifw=bool(getattr(args, "run_ifw", False)),
        microstructure_ifw_echo_time_ms=getattr(args, "ifw_echo_time_ms", None),
        microstructure_ifw_t2_gm_ms=getattr(args, "ifw_t2_gm_ms", 95.0),
        microstructure_ifw_t2_csf_ms=getattr(args, "ifw_t2_csf_ms", 1500.0),
        microstructure_dki_fit_method=args.dki_fit_method,
        microstructure_dki_matlab_executable=args.dki_matlab_executable,
        microstructure_dki_mkcurve_root=args.dki_mkcurve_root,
        microstructure_dki_execution=dict(num_threads=args.dki_threads,
            voxels_per_chunk=args.dki_voxels_per_chunk, mkcurve_workers=args.mkcurve_workers,
            mkcurve_voxels_per_chunk=args.mkcurve_voxels_per_chunk, mkcurve_chunk_schedule=args.mkcurve_chunk_schedule),
        wma_num_threads=args.wma_threads,
        microstructure_free_water_chunk_size=args.microstructure_free_water_chunk_size,
        microstructure_free_water_engine=args.microstructure_free_water_engine,
        microstructure_free_water_n_jobs=args.microstructure_free_water_n_jobs,
        microstructure_free_water_vox_per_chunk=args.microstructure_free_water_vox_per_chunk,
        microstructure_free_water_max_fraction_for_tissue=args.microstructure_free_water_max_fraction_for_tissue,
        postprocess_native_dir=args.postprocess_native_dir,
        postprocess_mri_dir=args.postprocess_mri_dir,
        postprocess_backend_root=args.postprocess_backend_root,
        postprocess_mode=args.postprocess_mode,
        postprocess_python_bin=args.postprocess_python_bin,
        postprocess_freesurfer_home=args.postprocess_freesurfer_home,
        postprocess_runtime=args.postprocess_runtime,
        postprocess_license_file=args.postprocess_license_file,
        postprocess_atlases=_split_csv_preserve_case(args.postprocess_atlases) or None,
        postprocess_force=bool(args.postprocess_force),
        atlas_t2=args.atlas_t2,
        surface_atlas_t2=args.surface_atlas_t2,
        surface_python_bin=args.surface_python_bin,
        surface_precision=args.surface_precision,
        surface_cpu_threads=args.surface_cpu_threads,
        surface_max_gpu_gib=args.surface_max_gpu_gib,
        model_assets_dir=args.model_assets_dir,
        ddseg_device=args.ddseg_device,
        surface_device=args.surface_device,
        select_streamlines=args.select_streamlines,
        dti_min_length_mm=args.dti_min_length_mm,
        dti_max_length_mm=args.dti_max_length_mm,
        mrtrix_bin_dir=args.mrtrix_bin_dir,
        mrtrix_nthreads=args.mrtrix_nthreads,
        mrtrix_random_seed=args.mrtrix_seed,
        tractography_seeding=args.tckgen_seeding,
        ddseg_min_wm_dice=args.ddseg_min_wm_dice,
        allow_anatomy_space_mismatch=bool(args.allow_anatomy_space_mismatch),
        allow_act_ddseg_only=bool(args.allow_act_ddseg_only),
        tract_querier_root=args.tract_querier_root,
        wmql_python_executable=args.wmql_python_executable,
        connectome_backend=args.connectome_backend,
        connectome_compress_mrtrix_dense=args.compress_mrtrix_dense,
        saf_fs_dir=args.saf_fs_dir,
        saf_pregenerated_tractogram=args.saf_pregenerated_tractogram,
        saf_tckgen_options=args.saf_tckgen_options,
        saf_mrtrix_prefix=args.saf_mrtrix_prefix,
        wma_input_tractography=args.wma_input_tractography,
        wma_atlas_dir=args.wma_atlas_dir,
        wma_atlas=args.wma_atlas,
        wma_root=args.wma_root,
        slicer_executable=args.slicer_executable,
        wma_use_virtual_x=bool(args.wma_use_virtual_x),
        wma_export_diffusion_measures=bool(args.wma_export_diffusion_measures),
        fiber_tract_measurements_cli=args.fiber_tract_measurements_cli,
        tractcloud_input_tractography=args.tractcloud_input_tractography,
        tractcloud_root=args.tractcloud_root,
        tractcloud_python_executable=args.tractcloud_python_executable,
        tractcloud_data_dir=args.tractcloud_data_dir,
        tractcloud_device=args.tractcloud_device,
        tractcloud_batch_size=args.tractcloud_batch_size,
        preflight_blocked_targets=preflight_blocked_targets,
        preflight_blocked_by_backend=(getattr(doctor_result, 'tracking_backend_blockers', {}) if doctor_result else {}),
        target_required_by=dict(getattr(args, 'target_required_by', {}) or {}),
        excluded_targets=tuple(getattr(args, 'excluded_targets', ()) or ()),
        closure_algorithm_requesters=dict(getattr(args, 'closure_algorithm_requesters', {}) or {}),
    )

    if args.plan:
        plan = plan_subject_products(config, write=True)
        print(
            json.dumps(
                {
                    "environment": doctor_result.to_dict() if doctor_result is not None else None,
                    "environment_report_paths": doctor_paths,
                    "run_plan_path": plan.run_plan_path.as_posix(),
                    "run_plan": plan.to_dict(),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        if reason := _phase_encoding_block(plan):
            print(reason, file=sys.stderr)
            return 2
        return 0

    if args.execute:
        plan = plan_subject_products(config, write=True)
        _print_run_plan_summary(plan, stream=sys.stderr)
        if reason := _phase_encoding_block(plan):
            # W2b rule 4 (T1 D1): no phase-encoding direction, no preprocessing; stop before any stage starts.
            print(reason, file=sys.stderr)
            return 2
        # If Cancel kills this process's whole process group (see
        # shared/process_group.py), leave a terminal "cancelled" manifest
        # instead of letting run_subject_products() vanish mid-flight with
        # the manifest stuck at "running" (or never written at all).
        cancellation_manifest_path = orchestration_manifest_path(
            config.subject_root, output_root=config.output_root, run_name=config.run_name
        )
        signal.signal(
            signal.SIGTERM,
            _make_sigterm_handler(
                cancellation_manifest_path,
                subject_id=config.subject_id,
                run_name=config.run_name,
            ),
        )

    orchestration_started = time.monotonic()
    result = run_subject_products(config, log=_cli_log)
    orchestration_seconds = time.monotonic() - orchestration_started
    cli_timing = dict(main_wall_seconds=time.monotonic() - cli_started,
                      environment_check_seconds=environment_seconds, orchestration_call_seconds=orchestration_seconds,
                      scope='CLI main entry through environment checks, planning and orchestration return; excludes Python startup and final response/telemetry writes.')
    manifest_payload = {}
    try:
        manifest_payload = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
        task_payloads = manifest_payload.get("tasks", [])
        manifest_payload['cli_timing'] = cli_timing
        from shared.pipeline import _write_manifest
        _write_manifest(result.run_manifest_path, manifest_payload)
    except Exception:
        task_payloads = [
            {
                "name": task.name,
                "status": task.status,
                "reason": task.reason,
                "manifest_path": task.manifest_path.as_posix() if task.manifest_path else None,
                "qc_path": task.qc_path.as_posix() if task.qc_path else None,
                "outputs": task.outputs,
            }
            for task in result.tasks
        ]
    payload = {
        "subject_id": result.subject_id,
        "subject_root": result.subject_root.as_posix(),
        "run_manifest_path": result.run_manifest_path.as_posix(),
        "run_plan_path": result.run_plan_path.as_posix() if result.run_plan_path else None,
        "tasks": task_payloads,
        "cli_timing": cli_timing,
    }

    if args.print_manifest_only:
        print(result.run_manifest_path.as_posix())
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))

    # The same rule as the run manifest's "failed" status (orchestrator.py): partial_failed counts.
    failure_like_statuses = {"failed", "partial_failed"}
    if args.execute and any(task.status in failure_like_statuses for task in result.tasks):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
