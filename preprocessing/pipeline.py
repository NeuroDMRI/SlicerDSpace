"""Optional DWI preprocessing: one route (W2a / D1).

The route is DIPY MP-PCA denoising, then DIPY Gibbs-ringing removal, then FSL
eddy (and topup when a reverse-PE b=0 image is given) through MRtrix3
``dwifslpreproc`` by default. The experimental native runner calls the same
FSL tools directly. The correction input is the denoised, Gibbs-corrected
NIfTI, with the voxel order and affine of the imported DWI and the original FSL
gradients.

This stage is intentionally optional. Product pipelines may consume raw
``input/data.nii.gz`` directly, or users may run this stage first to generate a
shared corrected DWI under ``outputs/01_preproc``.
"""

from __future__ import annotations

from shared.cohort_schedule import scheduled_stage

from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import math
import os
import shutil
import subprocess
import importlib.util
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional

from shared.io_layout import default_task_output_dir
from shared.external_process import (
    fsl_root_from_bin_dir,
    managed_tool_environment,
    native_arm64_argv,
    subprocess_no_window_kwargs,
    text_subprocess_kwargs,
)
from shared.wsl_bridge import preprocessing_launcher
from shared.brain_mask import (
    BrainMaskConfig,
    brain_mask_config_from_options,
    generate_brain_mask,
    brain_mask_cache_inspection,
    inspect_brain_mask_backend,
    validate_brain_mask,
)
from shared.pipeline import RawInputBundle, find_subject_root
from shared.preprocessing_route import (
    CACHE_KEY_VERSION, ROUTE_ID as _ROUTE_ID, resolve_preprocessing_execution_backend, route_id as _route_id,
)
from shared.pe_resolution import AcquisitionMetadata, resolve_for_inputs


REQUIRED_COMMANDS = ("dwifslpreproc", "dwi2mask")
REVERSE_PHASE_COMMANDS = ("mrconvert", "dwiextract", "mrcat")
DIAGNOSTIC_COMMANDS = ("eddy", "topup")
PYTHON_MODULES = ("nibabel", "numpy", "dipy")
# W2a (D1): the one route. W2b: the cache key (version 6) and the manifest carry the route id of the run,
# `dipy_mppca+gibbs+fsl_eddy[+topup]` (without `+gibbs` when Gibbs removal is skipped); ROUTE_ID is the base string.
ROUTE_ID = _ROUTE_ID
# Old manifests (the removed DIPY between-volume route) still exist on disk and are read, never written.
DIPY_BVEC_ROTATION_POLICY = "inverse_sampling_affine_polar_v1"


def preprocessing_gradients_current(manifest: dict) -> bool:
    """Reject motion-corrected products of the removed DIPY route that used the superseded rotation rule (old manifests)."""
    key = manifest.get("cache_key") or {}
    if not isinstance(key, dict) or key.get("backend") != "dipy":
        return True
    parameters = key.get("parameters") or {}
    return (parameters.get("dipy_run_motion_correction") is False
            or parameters.get("dipy_bvec_rotation_policy") == DIPY_BVEC_ROTATION_POLICY)


def _default_eddy_threads() -> int:
    # An explicit config value never reaches here. The stage CPU reservation comes next.
    from shared.numeric_threads import reserved_stage_cpus
    reserved = reserved_stage_cpus()
    if reserved is not None:
        return reserved
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if callable(process_cpu_count):
        count = process_cpu_count()
    elif hasattr(os, "sched_getaffinity"):
        count = len(os.sched_getaffinity(0))
    else:
        count = os.cpu_count()
    n_cpus = int(count or 1)
    if n_cpus >= 64:
        cap = 32
    elif n_cpus >= 32:
        cap = 16
    else:
        cap = 8
    return max(1, min(cap, n_cpus))


@dataclass(frozen=True)
class DwiPreprocessingConfig:
    subject_id: str
    input_root: Path
    output_dir: Optional[Path] = None
    execute: bool = False
    skip: bool = False
    force: bool = False
    dipy_run_gibbs: bool = True
    b0_threshold: float = 50.0
    mrtrix_bin_dir: Optional[Path] = None
    fsl_bin_dir: Optional[Path] = None
    reverse_phase_b0: Optional[Path] = None
    phase_encoding_direction: Optional[str] = None
    readout_time: Optional[float] = None
    eddy_threads: Optional[int] = None
    eddy_second_level_model: Optional[str] = None
    brain_mask_backend: str = "pnl_cnn"
    brain_mask_device: str = "auto"
    pnl_mask_python: Optional[Path] = None
    pnl_mask_script: Optional[Path] = None
    pnl_mask_model_dir: Optional[Path] = None
    pnl_mask_ants_bin_dir: Optional[Path] = None
    pnl_mask_route: str = "auto"
    execution_backend: Optional[str] = None
    dipy_device: str = "auto"
    run_denoise: bool = True
    run_eddy: bool = True
    run_topup: bool = True
    output_root: Optional[Path] = None


def _effective_reverse_phase_b0(config: DwiPreprocessingConfig) -> Optional[Path]:
    return config.reverse_phase_b0 if config.run_topup else None


def _effective_run_eddy(config: DwiPreprocessingConfig) -> bool:
    return bool(config.run_eddy or (config.run_topup and _effective_reverse_phase_b0(config) is not None))


@dataclass(frozen=True)
class DwiPreprocessingResult:
    subject_id: str
    status: str
    output_dir: Path
    manifest_path: Path
    qc_path: Path
    final_dwi_path: Path
    final_bval_path: Path
    final_bvec_path: Path
    final_mask_path: Path
    # W2b: the resolved phase-encoding direction and readout time (None for an old caller or a skipped run).
    acquisition: Optional[AcquisitionMetadata] = None


def run_route_id(config: DwiPreprocessingConfig) -> str:
    """The route id of this run: `dipy_mppca[+gibbs]+fsl_eddy[+topup]` (or the active stage subset)."""
    effective_eddy = _effective_run_eddy(config)
    return _route_id(
        gibbs=bool(config.run_denoise and config.dipy_run_gibbs),
        topup=bool(effective_eddy and _effective_reverse_phase_b0(config) is not None),
        denoise=bool(config.run_denoise),
        eddy=effective_eddy,
    )


def resolve_acquisition_for(raw: RawInputBundle, config: DwiPreprocessingConfig) -> AcquisitionMetadata:
    """W2b (D1): resolve the PE direction and readout time from the user values and `input/dwi.json`.

    The sidecar sits next to the imported DWI; the affine is the one of the imported DWI. This function never raises
    for a missing direction: `.error` holds the rule-4 message and `require_phase_encoding()` raises it.
    """
    return resolve_for_inputs(
        input_dir=Path(raw.dwi_path).parent, dwi_path=raw.dwi_path,
        user_phase_encoding=config.phase_encoding_direction, user_readout_time=config.readout_time,
        has_reverse_b0=_effective_reverse_phase_b0(config) is not None,
    )


def _brain_mask_config(config: DwiPreprocessingConfig) -> BrainMaskConfig:
    selected = brain_mask_config_from_options(config)
    if config.eddy_threads is not None:
        selected = replace(selected, threads=_resolved_eddy_threads(config))
    return selected


def _resolve_raw_inputs(config: DwiPreprocessingConfig) -> RawInputBundle:
    # find_subject_root may generate a raw mask. Preprocessing needs only the
    # corrected-space mask, and planning must never launch mask inference.
    return find_subject_root(
        config.input_root, config.subject_id,
        brain_mask_config=_brain_mask_config(config), generate_missing_mask=False,
        output_root=config.output_root,
    )


def _resolve_output_dir(raw: RawInputBundle, output_dir: Optional[Path], *, output_root: Optional[Path] = None) -> Path:
    if output_dir is not None:
        return output_dir.expanduser().resolve()
    return default_task_output_dir(raw.subject_root, "preproc", output_root=output_root)


def _command_search_path(config: DwiPreprocessingConfig) -> str:
    entries: list[str] = []
    for candidate in (config.mrtrix_bin_dir, config.fsl_bin_dir):
        if candidate is not None:
            entries.append(candidate.expanduser().resolve().as_posix())
    existing = os.environ.get("PATH", "")
    if existing:
        entries.append(existing)
    return os.pathsep.join(entries)


def _fsl_root_from_bin_dir(bin_dir: Path) -> Path:
    """Return ``FSLDIR`` for either ``$FSLDIR/bin`` or ``share/fsl/bin``."""

    return fsl_root_from_bin_dir(bin_dir)


def _resolve_command(command: str, config: DwiPreprocessingConfig) -> Optional[str]:
    return shutil.which(command, path=_command_search_path(config))


def _resolved_eddy_threads(config: DwiPreprocessingConfig) -> int:
    eddy_threads = getattr(config, "eddy_threads", None)
    if eddy_threads is None:
        return _default_eddy_threads()
    if eddy_threads < 1:
        raise ValueError("eddy_threads must be >= 1.")
    return int(eddy_threads)


def _resolved_eddy_options(config: DwiPreprocessingConfig) -> str:
    options = ["--cnr_maps", "--residuals", f"--nthr={_resolved_eddy_threads(config)}"]
    if config.eddy_second_level_model is not None:
        model = config.eddy_second_level_model.strip().lower()
        if model not in {"none", "linear", "quadratic"}:
            raise ValueError(
                "eddy_second_level_model must be one of: none, linear, quadratic."
            )
        options.append(f"--slm={model}")
    return " ".join(options)


def _external_command_environment(config: DwiPreprocessingConfig) -> dict[str, str]:
    # B3: one helper builds the shim, FSLDIR, FSLOUTPUTTYPE and PATH for eddy, topup and dwifslpreproc.
    return managed_tool_environment(mrtrix_bin_dir=config.mrtrix_bin_dir, fsl_bin_dir=config.fsl_bin_dir)


def inspect_preprocessing_backend(config: DwiPreprocessingConfig) -> dict[str, Any]:
    """Check the selected correction runner and the common Python dependencies."""
    active_python_modules = PYTHON_MODULES if config.run_denoise else ("nibabel", "numpy")
    python_status = {module: importlib.util.find_spec(module) is not None for module in active_python_modules}
    missing_python = [module for module, available in python_status.items() if not available]
    execution_backend = resolve_preprocessing_execution_backend(config.execution_backend)
    effective_eddy = _effective_run_eddy(config)
    effective_reverse = _effective_reverse_phase_b0(config) is not None
    native = {}
    if not effective_eddy:
        required_commands: list[str] = []
        diagnostic_commands: list[str] = []
        commands: dict[str, Optional[str]] = {}
        missing_commands: list[str] = []
    elif execution_backend == "native":
        from preprocessing.fsl_runtime import inspect_native_fsl_runtime

        native = inspect_native_fsl_runtime(
            config.mrtrix_bin_dir, config.fsl_bin_dir, reverse_phase=effective_reverse,
        )
        required_commands = native["required_commands"]
        commands = native["commands"]
        missing_commands = native["missing_commands"]
        diagnostic_commands = []
    else:
        required_commands = list(REQUIRED_COMMANDS)
        if effective_reverse:
            required_commands.extend(REVERSE_PHASE_COMMANDS)
        diagnostic_commands = list(DIAGNOSTIC_COMMANDS)
        commands = {
            command: _resolve_command(command, config)
            for command in (*required_commands, *diagnostic_commands)
        }
        missing_commands = [command for command in required_commands if commands[command] is None]
    missing_diagnostic = [command for command in diagnostic_commands if commands[command] is None]
    missing_runtime = native["missing_required"] if native else missing_commands
    return {
        "kind": "dwi_preprocessing_backend",
        "backend": run_route_id(config),
        "execution_backend": execution_backend,
        **({name: native[name] for name in ("fsl_root", "topup_config")} if native else {}),
        "python": python_status,
        "missing_python": missing_python,
        "required_commands": required_commands,
        "diagnostic_commands": diagnostic_commands,
        "commands": commands,
        "missing_commands": missing_commands,
        "missing_required": [*(f"python module: {module}" for module in missing_python), *missing_runtime],
        "missing_diagnostic": missing_diagnostic,
        "available": not missing_python and not missing_runtime,
        "capabilities": {
            "denoise": "mppca" if config.run_denoise else False,
            "gibbs_removal": bool(config.run_denoise and config.dipy_run_gibbs),
            "susceptibility_distortion_correction": bool(effective_eddy and effective_reverse),
            "eddy_current_modeling": effective_eddy,
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _link_or_copy(src: Path, dest: Path) -> None:
    src = src.expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        dest.symlink_to(src)
        return
    except OSError:
        pass
    try:
        os.link(src, dest)
        return
    except OSError:
        pass
    shutil.copy2(src, dest)


def _module_version(name: str) -> Optional[str]:
    try:
        from importlib import metadata
        return metadata.version(name)
    except Exception:
        return None


def _file_record(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": path.as_posix(), "exists": False}
    stat = path.stat()
    record: dict[str, Any] = {
        "path": path.resolve().as_posix(),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if path.is_file():
        from shared.cache_reuse import sha256_file

        record["sha256"] = sha256_file(path)
    return record


PREPROCESSING_EXECUTION_ONLY_PARAMS = frozenset({"eddy_threads"})
_BRAIN_MASK_PATH_OPTION_KEYS = frozenset({
    "pnl_mask_python",
    "pnl_mask_script",
    "pnl_mask_model_dir",
    "pnl_mask_ants_bin_dir",
    "brain_mask_device",
})


def _brain_mask_cache_key_matches(recorded: object, expected: object) -> bool:
    from shared.cache_reuse import file_records_equivalent

    if not isinstance(recorded, dict) or not isinstance(expected, dict):
        return False
    if recorded.get("backend") != expected.get("backend"):
        return False
    rec_insp = recorded.get("inspection") if isinstance(recorded.get("inspection"), dict) else {}
    exp_insp = expected.get("inspection") if isinstance(expected.get("inspection"), dict) else {}
    if rec_insp.get("pnl_route_resolved") != exp_insp.get("pnl_route_resolved"):
        return False
    native_pnl = (
        expected.get("backend") == "pnl_cnn"
        and exp_insp.get("pnl_route_resolved") == "native"
    )
    ignored_opts = set(_BRAIN_MASK_PATH_OPTION_KEYS)
    if native_pnl:
        ignored_opts.add("threads")
    rec_opts = {
        k: v for k, v in (recorded.get("options") or {}).items()
        if k not in ignored_opts
    }
    exp_opts = {
        k: v for k, v in (expected.get("options") or {}).items()
        if k not in ignored_opts
    }
    if rec_opts != exp_opts:
        return False
    rec_models = (recorded.get("assets") or {}).get("model_files")
    exp_models = (expected.get("assets") or {}).get("model_files")
    if isinstance(exp_models, list):
        if not isinstance(rec_models, list) or len(rec_models) != len(exp_models):
            return False
        for rec_item, exp_item in zip(rec_models, exp_models):
            if not file_records_equivalent(rec_item, exp_item, allow_relocated=True):
                return False
    return True


def preprocessing_cache_key_matches(recorded: object, expected: object) -> bool:
    """Compare preprocessing v6 cache keys on scientific inputs & settings.

    Tolerates `.py` source edits (`fsl_native.py`, `fsl_runtime.py`,
    `mrtrix_preproc_runner.py`), `eddy_threads`, `CUDA_VISIBLE_DEVICES`, tool
    install-path/mtime churn, and subject-directory relocation while preserving
    strict invalidation for route, execution backend, sidecar hash, PE/readout
    settings, scientific parameters, brain-mask settings, and raw DWI inputs.
    """
    from shared.cache_reuse import file_records_equivalent

    if not isinstance(recorded, dict) or not isinstance(expected, dict):
        return False
    for key in ("kind", "version", "execution_backend", "backend", "dwi_sidecar_sha256"):
        if recorded.get(key) != expected.get(key):
            return False
    rec_inputs = recorded.get("inputs")
    exp_inputs = expected.get("inputs")
    if not isinstance(rec_inputs, dict) or not isinstance(exp_inputs, dict):
        return False
    if set(rec_inputs.keys()) != set(exp_inputs.keys()):
        return False
    for key, exp_rec in exp_inputs.items():
        if not file_records_equivalent(rec_inputs.get(key), exp_rec, allow_relocated=True):
            return False
    rec_params = recorded.get("parameters")
    exp_params = expected.get("parameters")
    if not isinstance(rec_params, dict) or not isinstance(exp_params, dict):
        return False
    rec_sci = {k: v for k, v in rec_params.items() if k not in PREPROCESSING_EXECUTION_ONLY_PARAMS}
    exp_sci = {k: v for k, v in exp_params.items() if k not in PREPROCESSING_EXECUTION_ONLY_PARAMS}
    if rec_sci != exp_sci:
        return False
    if "topup_config" in expected or "topup_config" in recorded:
        if not file_records_equivalent(recorded.get("topup_config"), expected.get("topup_config"), allow_relocated=True):
            return False
    if not _brain_mask_cache_key_matches(recorded.get("brain_mask"), expected.get("brain_mask")):
        return False
    return True


def _reverse_phase_stem(image_path: Path) -> str:
    name = image_path.name
    lower = name.lower()
    for suffix in (".nii.gz", ".mif.gz", ".nii", ".mif", ".nrrd", ".nhdr"):
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return image_path.stem


def _reverse_phase_sidecar_records(reverse_phase: Optional[Path]) -> dict[str, Any]:
    """Return file records for sibling gradient/detached-data sidecars affecting reverse-PE b0 selection."""
    if reverse_phase is None:
        return {}
    records: dict[str, Any] = {}
    stem = _reverse_phase_stem(reverse_phase)
    for key, suffixes in (
        ("reverse_phase_bval", (".bval", ".bvals")),
        ("reverse_phase_bvec", (".bvec", ".bvecs")),
    ):
        for suffix in suffixes:
            candidate = reverse_phase.parent / f"{stem}{suffix}"
            if candidate.is_file():
                records[key] = _file_record(candidate)
                break
    if reverse_phase.name.lower().endswith(".nhdr") and reverse_phase.is_file():
        try:
            import nrrd

            header = nrrd.read_header(str(reverse_phase))
            data_file = header.get("data file") or header.get("datafile")
            if isinstance(data_file, str) and data_file.strip():
                candidate = Path(data_file.strip())
                if not candidate.is_absolute():
                    candidate = (reverse_phase.parent / candidate).resolve()
                if candidate.is_file():
                    records["reverse_phase_data"] = _file_record(candidate)
        except Exception:
            pass
    return records


def _preprocessing_cache_key(
    raw: RawInputBundle,
    config: DwiPreprocessingConfig,
    backend: dict[str, Any],
    mask_backend: Optional[dict[str, Any]] = None,
    acquisition: Optional[AcquisitionMetadata] = None,
) -> dict[str, Any]:
    """Describe every input and setting that can change preprocessing output.

    Version 6 (W2b): the route id of the run, the effective PE direction and readout time with their sources, and the
    SHA-256 of `input/dwi.json`.
    """

    if acquisition is None:
        acquisition = resolve_acquisition_for(raw, config)
    effective_reverse = _effective_reverse_phase_b0(config)
    reverse_phase = effective_reverse.expanduser().resolve() if effective_reverse else None
    executables = {
        name: _file_record(Path(value))
        for name, value in dict(backend.get("commands") or {}).items()
        if value
    }
    eddy_path = dict(backend.get("commands") or {}).get("eddy")
    eddy_binaries = {
        path.name: _file_record(path)
        for path in (Path(eddy_path).parent.glob("eddy*") if eddy_path else ())
        if path.is_file()
    }
    return {
        "kind": "dwi_preprocessing_cache_key",
        "version": CACHE_KEY_VERSION,
        **({
            "execution_backend": "native",
            "native_dispatch": _file_record(Path(__file__).with_name("fsl_native.py")),
            "native_runtime": _file_record(Path(__file__).with_name("fsl_runtime.py")),
            "topup_config": _file_record(Path(backend["topup_config"])) if backend.get("topup_config") else None,
        } if backend.get("execution_backend") == "native" else {}),
        "backend": run_route_id(config),
        "dwi_sidecar_sha256": acquisition.sidecar_sha256,
        "inputs": {
            "dwi": _file_record(raw.dwi_path),
            "bval": _file_record(raw.bval_path),
            "bvec": _file_record(raw.bvec_path),
            "mask": _file_record(raw.mask_path),
            "reverse_phase_b0": _file_record(reverse_phase) if reverse_phase else None,
            **_reverse_phase_sidecar_records(reverse_phase),
        },
        "parameters": {
            "dipy_run_gibbs": bool(config.dipy_run_gibbs),
            **({"run_denoise": False} if not config.run_denoise else {}),
            **({"run_eddy": False} if not config.run_eddy else {}),
            **({"run_topup": False} if not config.run_topup else {}),
            "b0_threshold": float(config.b0_threshold),
            "phase_encoding_direction": acquisition.phase_encoding,
            "phase_encoding_source": acquisition.phase_encoding_source,
            "pe_sign_assumed": acquisition.pe_sign_assumed,
            "readout_time": acquisition.readout_time,
            "readout_time_source": acquisition.readout_time_source,
            "eddy_threads": _resolved_eddy_threads(config),
            "eddy_second_level_model": (
                config.eddy_second_level_model.strip().lower()
                if config.eddy_second_level_model is not None
                else None
            ),
        },
        "executables": executables,
        "runtime_bridge_configs": {
            str(path): _file_record(path)
            for value in dict(backend.get("commands") or {}).values() if value
            for path in (Path(str(value) + ".conf"), Path(value).parent / "python.exe.conf")
            if path.is_file()
        },
        "eddy_binaries": eddy_binaries,
        "dipy_version": _module_version("dipy"),
        "eddy_cuda_visibility": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "eddy_device_policy": _file_record(Path(__file__).resolve().parents[1] / "shared/mrtrix_preproc_runner.py"),
        "brain_mask": _brain_mask_cache_key(
            config, mask_backend if mask_backend is not None else inspect_brain_mask_backend(_brain_mask_config(config)),
        ),
    }


def _brain_mask_cache_key(
    config: DwiPreprocessingConfig, inspection: dict[str, Any],
) -> dict[str, Any]:
    selected = _brain_mask_config(config)
    # B12: the native PNL route leaves the thread count out of the key (M-A2); the managed route keeps it.
    inspection = brain_mask_cache_inspection(inspection)
    native_pnl = selected.backend == "pnl_cnn" and inspection.get("pnl_route_resolved") == "native"
    options = {
        name: (value.expanduser().resolve().as_posix() if isinstance(value, Path) else value)
        for name, value in vars(selected).items() if not (native_pnl and name == "threads")
    }
    assets: dict[str, Any] = {}
    for name, value in vars(selected).items():
        if isinstance(value, Path):
            assets[name] = _file_record(value.expanduser().resolve())
    if config.pnl_mask_model_dir is not None:
        model_dir = config.pnl_mask_model_dir.expanduser().resolve()
        assets["model_files"] = [
            _file_record(path) for path in sorted(model_dir.rglob("*")) if path.is_file()
        ]

    def collect_tools(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                collect_tools(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect_tools(item)
        elif isinstance(value, (str, Path)) and Path(value).is_absolute():
            path = Path(value)
            if path.is_file():
                assets[path.as_posix()] = _file_record(path)

    collect_tools(inspection)
    return {"backend": selected.backend, "options": options, "inspection": inspection, "assets": assets}


def _run_command(
    cmd: list[str], *, env: dict[str, str], log: Optional[Callable[[str], None]], capture: bool = False,
) -> str:
    if log:
        log("[Preproc] Running: " + " ".join(cmd))
    process = subprocess.Popen(
        native_arm64_argv(cmd, env),  # C6/D8: native arm64 MRtrix under a Rosetta Slicer
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        **subprocess_no_window_kwargs(**text_subprocess_kwargs(bufsize=1)),
    )
    output = []

    def on_line(line):
        if capture:
            output.append(line)
        if log:
            log("[Preproc] " + line.rstrip())

    # H8: a surviving grandchild must not keep the stage open after the direct child has exited.
    from shared.line_stream import stream_process_lines
    return_code = stream_process_lines(process, on_line,
                                       notice=(lambda message: log("[Preproc] " + message)) if log else None)
    if return_code != 0:
        raise RuntimeError(f"Preprocessing command failed with exit code {return_code}: {' '.join(cmd)}")
    return "".join(output)


def _planned_paths(output_dir: Path, subject: str) -> dict[str, Path]:
    return {
        "import_dir": output_dir / "00_import",
        "gradients_dir": output_dir / "01_gradients",
        "denoise_dir": output_dir / "02_denoise_degibbs",
        "motion_dir": output_dir / "03_distortion_motion",
        "mask_dir": output_dir / "04_bias_normalize",
        "qc_dir": output_dir / "05_qc",
        "final_dwi": output_dir / f"{subject}_dwi_final.nii.gz",
        "final_bval": output_dir / f"{subject}_dwi_final.bval",
        "final_bvec": output_dir / f"{subject}_dwi_final.bvec",
        "final_mask": output_dir / f"{subject}_brain_mask_final.nii.gz",
        "manifest": output_dir / f"{subject}_preproc_manifest.json",
        "qc": output_dir / "05_qc" / f"{subject}_preproc_qc.json",
        "eddy_parameters": output_dir / "05_qc" / f"{subject}_eddy_parameters.txt",
        "eddy_movement_rms": output_dir / "05_qc" / f"{subject}_eddy_movement_rms.txt",
        "eddy_outlier_map": output_dir / "05_qc" / f"{subject}_eddy_outlier_map.txt",
        "eddy_outlier_report": output_dir / "05_qc" / f"{subject}_eddy_outlier_report.txt",
        "eddy_cnr_maps": output_dir / "05_qc" / f"{subject}_eddy_cnr_maps.nii.gz",
        "eddy_residuals": output_dir / "05_qc" / f"{subject}_eddy_residuals.nii.gz",
    }


def _correction_scratch(paths: dict[str, Path]) -> Path:
    return paths.get("native_scratch", paths["motion_dir"] / "dwifslpreproc_tmp")


def _collect_dwifslpreproc_diagnostics(paths: dict[str, Path], subject: str) -> dict[str, dict[str, Any]]:
    scratch_dir = _correction_scratch(paths)
    destinations = {
        "eddy_parameters": (
            scratch_dir / "dwi_post_eddy.eddy_parameters",
            paths["eddy_parameters"],
        ),
        "eddy_movement_rms": (
            scratch_dir / "dwi_post_eddy.eddy_movement_rms",
            paths["eddy_movement_rms"],
        ),
        "eddy_restricted_movement_rms": (
            scratch_dir / "dwi_post_eddy.eddy_restricted_movement_rms",
            paths["motion_dir"] / f"{subject}_eddy_restricted_movement_rms.txt",
        ),
        "eddy_outlier_map": (
            scratch_dir / "dwi_post_eddy.eddy_outlier_map",
            paths["eddy_outlier_map"],
        ),
        "eddy_outlier_report": (
            scratch_dir / "dwi_post_eddy.eddy_outlier_report",
            paths["eddy_outlier_report"],
        ),
        "eddy_cnr_maps": (
            scratch_dir / "dwi_post_eddy.eddy_cnr_maps.nii.gz",
            paths["eddy_cnr_maps"],
        ),
        "eddy_residuals": (
            scratch_dir / "dwi_post_eddy.eddy_residuals.nii.gz",
            paths["eddy_residuals"],
        ),
        "topup_field_coefficients": (
            scratch_dir / "field_fieldcoef.nii.gz",
            paths["motion_dir"] / f"{subject}_topup_fieldcoef.nii.gz",
        ),
        "topup_field_map": (
            scratch_dir / "field_map.nii.gz",
            paths["motion_dir"] / f"{subject}_topup_field_map.nii.gz",
        ),
        "topup_motion_parameters": (
            scratch_dir / "field_movpar.txt",
            paths["motion_dir"] / f"{subject}_topup_motion_parameters.txt",
        ),
        "topup_log": (
            scratch_dir / "topup_in.topup_log",
            paths["qc_dir"] / f"{subject}_topup.log",
        ),
    }
    records: dict[str, dict[str, Any]] = {}
    for name, (source, destination) in destinations.items():
        if source.is_file():
            shutil.copy2(source, destination)
        elif "native_scratch" in paths:
            destination.unlink(missing_ok=True)
        records[name] = _file_record(destination)
    return records


def _prepare_reverse_b0_nrrd(
    source: Path,
    dest: Path,
    *,
    b0_threshold: float = 50.0,
    reference_path: Optional[Path] = None,
) -> None:
    """Convert a 3D or 4D NRRD/NHDR reverse-PE b=0 image to a single-volume 4D RAS NIfTI.

    When ``reference_path`` (the imported main DWI NIfTI) is provided, the voxel axes are reoriented
    to match the reference image's exact axis orientation (for example, LAS from ``DWIConvert NrrdToFSL``)
    while preserving physical RAS coordinates, so downstream ``mrcat`` and FSL ``fslmerge``/``topup`` see
    identical voxel-to-RAS orientation matrices.
    """
    import nibabel as nib
    import nrrd
    import numpy as np
    from shared.input_preview import _nrrd_bvalues, _nrrd_diffusion_axis, _nrrd_ras_affine, _space_signs

    data, header = nrrd.read(str(source))
    if data.ndim == 3:
        directions = np.asarray(header.get("space directions"), dtype=float)
        if directions.shape != (3, 3) or not np.all(np.isfinite(directions)):
            raise ValueError(f"Reverse phase-encoded b=0 NRRD has invalid space directions: {source}")
        origin = np.asarray(header.get("space origin", [0.0, 0.0, 0.0]), dtype=float)
        affine = np.eye(4)
        affine[:3, :3] = directions.T
        affine[:3, 3] = origin
        ras_affine = np.diag(list(_space_signs(header.get("space"))) + [1.0]) @ affine
        vol3d = np.asarray(data, dtype=np.float32)
    elif data.ndim == 4:
        axis = _nrrd_diffusion_axis(header, data.ndim)
        volumes = np.moveaxis(data, axis, -1)
        vol_index = 0
        try:
            bvals = _nrrd_bvalues(header, volumes.shape[3])
            vol_index = next((i for i, b in enumerate(bvals) if b <= b0_threshold), 0)
        except Exception:
            vol_index = 0
        ras_affine = _nrrd_ras_affine(header, axis)
        vol3d = np.asarray(volumes[..., vol_index], dtype=np.float32)
    else:
        raise ValueError(f"reverse_phase_b0 NRRD must be 3D or 4D; received {data.ndim}D ({data.shape}).")
    if reference_path is not None and Path(reference_path).is_file():
        ref_img = nib.load(str(reference_path))
        ref_affine = np.asarray(ref_img.affine, dtype=float)
        try:
            xform = nib.orientations.ornt_transform(
                nib.orientations.io_orientation(ras_affine),
                nib.orientations.io_orientation(ref_affine),
            )
            if not np.allclose(xform, [[0, 1], [1, 1], [2, 1]]):
                orig_shape = vol3d.shape
                vol3d = np.ascontiguousarray(nib.orientations.apply_orientation(vol3d, xform), dtype=np.float32)
                ras_affine = ras_affine @ nib.orientations.inv_ornt_aff(xform, orig_shape)
        except Exception:
            pass
        validate_reverse_b0_spatial_grid(
            ref_img.shape,
            ref_affine,
            vol3d.shape,
            ras_affine,
            forward_label="main DWI",
            reverse_label=f"reverse phase-encoded b=0 ({source.name})",
        )
    vol = vol3d[..., np.newaxis]
    dest.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(vol, ras_affine), str(dest))


_OPPOSITE_PE_DESIGNATORS = {
    "AP": "PA",
    "PA": "AP",
    "LR": "RL",
    "RL": "LR",
    "IS": "SI",
    "SI": "IS",
}


def validate_reverse_b0_spatial_grid(
    forward_shape: tuple[int, ...] | list[int],
    forward_affine: Any,
    reverse_shape: tuple[int, ...] | list[int],
    reverse_affine: Any,
    *,
    forward_label: str = "main DWI",
    reverse_label: str = "reverse phase-encoded b=0",
    spacing_rtol: float = 1e-4,
    rotation_atol: float = 1e-4,
    origin_atol_mm: float = 1e-2,
) -> None:
    """Validate that forward and reverse b=0 images share the exact same 3D spatial grid before ``mrcat``/Topup.

    MRtrix ``mrcat`` adopts the first image's affine header when concatenating volumes of identical 3D matrix size,
    which would silently hide origin, spacing, or orientation mismatches from downstream Topup checks.
    """
    import numpy as np

    fwd_dims = tuple(int(v) for v in tuple(forward_shape)[:3])
    rev_dims = tuple(int(v) for v in tuple(reverse_shape)[:3])
    if len(fwd_dims) != 3 or any(d <= 0 for d in fwd_dims):
        raise ValueError(f"Invalid {forward_label} spatial shape: {tuple(forward_shape)}")
    if len(rev_dims) != 3 or any(d <= 0 for d in rev_dims):
        raise ValueError(f"Invalid {reverse_label} spatial shape: {tuple(reverse_shape)}")
    if fwd_dims != rev_dims:
        raise ValueError(
            f"{reverse_label} spatial grid does not match {forward_label} before Topup merge: "
            f"3D dimensions differ ({rev_dims} vs {fwd_dims})."
        )
    fwd_aff = np.asarray(forward_affine, dtype=np.float64)
    rev_aff = np.asarray(reverse_affine, dtype=np.float64)
    if fwd_aff.shape != (4, 4) or not np.isfinite(fwd_aff).all() or abs(float(np.linalg.det(fwd_aff[:3, :3]))) < 1e-12:
        raise ValueError(f"{forward_label} has a non-finite or singular spatial affine.")
    if rev_aff.shape != (4, 4) or not np.isfinite(rev_aff).all() or abs(float(np.linalg.det(rev_aff[:3, :3]))) < 1e-12:
        raise ValueError(f"{reverse_label} has a non-finite or singular spatial affine.")

    fwd_spacing = np.linalg.norm(fwd_aff[:3, :3], axis=0)
    rev_spacing = np.linalg.norm(rev_aff[:3, :3], axis=0)
    if np.any(fwd_spacing <= 0) or np.any(rev_spacing <= 0):
        raise ValueError("Topup spatial grid validation encountered non-positive voxel spacing.")
    rel_spacing = np.abs(rev_spacing - fwd_spacing) / (0.5 * (fwd_spacing + rev_spacing))
    if float(np.max(rel_spacing)) > float(spacing_rtol):
        raise ValueError(
            f"{reverse_label} spatial grid does not match {forward_label} before Topup merge: "
            f"voxel sizes differ ({tuple(round(float(v), 5) for v in rev_spacing)} vs "
            f"{tuple(round(float(v), 5) for v in fwd_spacing)} mm, tolerance {spacing_rtol:g})."
        )

    fwd_rot = fwd_aff[:3, :3] / fwd_spacing[None, :]
    rev_rot = rev_aff[:3, :3] / rev_spacing[None, :]
    max_rot_diff = float(np.max(np.abs(rev_rot - fwd_rot)))
    if max_rot_diff > float(rotation_atol):
        raise ValueError(
            f"{reverse_label} spatial grid does not match {forward_label} before Topup merge: "
            f"orientation/rotation matrix differs (max element diff {max_rot_diff:.6g}, tolerance {rotation_atol:g})."
        )

    origin_diff = np.abs(rev_aff[:3, 3] - fwd_aff[:3, 3])
    max_origin_diff = float(np.max(origin_diff))
    if max_origin_diff > float(origin_atol_mm):
        raise ValueError(
            f"{reverse_label} spatial grid does not match {forward_label} before Topup merge: "
            f"origin differs by {max_origin_diff:.3f} mm (tolerance {origin_atol_mm:g} mm; "
            f"forward={np.round(fwd_aff[:3, 3], 3).tolist()}, reverse={np.round(rev_aff[:3, 3], 3).tolist()})."
        )


def _read_nrrd_spatial_grid_aligned(source: Path, reference_affine: Any) -> tuple[tuple[int, int, int], Any]:
    """Read an NRRD/NHDR header's 3D shape and RAS affine after the same orientation alignment as ``_prepare_reverse_b0_nrrd``."""
    import nibabel as nib
    import nrrd
    import numpy as np
    from shared.input_preview import _nrrd_diffusion_axis, _nrrd_ras_affine, _space_signs

    header = nrrd.read_header(str(source))
    raw_sizes = header.get("sizes")
    sizes = () if raw_sizes is None else tuple(int(s) for s in raw_sizes)
    if len(sizes) == 3:
        directions = np.asarray(header.get("space directions"), dtype=float)
        if directions.shape != (3, 3) or not np.all(np.isfinite(directions)):
            raise ValueError(f"Reverse phase-encoded b=0 NRRD has invalid space directions: {source}")
        origin = np.asarray(header.get("space origin", [0.0, 0.0, 0.0]), dtype=float)
        affine = np.eye(4)
        affine[:3, :3] = directions.T
        affine[:3, 3] = origin
        ras_affine = np.diag(list(_space_signs(header.get("space"))) + [1.0]) @ affine
        shape3d = sizes
    elif len(sizes) == 4:
        axis = _nrrd_diffusion_axis(header, 4)
        ras_affine = _nrrd_ras_affine(header, axis)
        shape3d = tuple(sizes[i] for i in range(4) if i != axis)
    else:
        raise ValueError(f"reverse_phase_b0 must be a 3D or 4D image; received shape {sizes}.")

    ref_affine = np.asarray(reference_affine, dtype=float)
    try:
        xform = nib.orientations.ornt_transform(
            nib.orientations.io_orientation(ras_affine),
            nib.orientations.io_orientation(ref_affine),
        )
        if not np.allclose(xform, [[0, 1], [1, 1], [2, 1]]):
            orig_shape = shape3d
            shape3d = tuple(orig_shape[int(axis_idx)] for axis_idx in xform[:, 0])
            ras_affine = ras_affine @ nib.orientations.inv_ornt_aff(xform, orig_shape)
    except Exception:
        pass
    return shape3d, ras_affine


def _read_mif_header_dict(path: Path) -> Optional[dict[str, Any]]:
    """Parse ``dim``, ``vox``, and ``transform`` from an MRtrix ``.mif`` / ``.mif.gz`` header in pure Python."""
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    try:
        with opener(path, "rb") as stream:
            first = stream.readline(256)
            if not first.startswith(b"mrtrix image"):
                return None
            dims: list[int] = []
            vox: list[float] = []
            transform_rows: list[list[float]] = []
            for _ in range(512):
                raw_line = stream.readline(4096)
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").strip()
                if line == "END":
                    break
                if ":" not in line:
                    continue
                key, _, value = line.partition(":")
                key = key.strip().lower()
                val = value.strip()
                if key == "dim":
                    dims = [int(tok.strip()) for tok in val.split(",") if tok.strip()]
                elif key == "vox":
                    vox = [float(tok.strip()) for tok in val.split(",") if tok.strip()]
                elif key == "transform":
                    row = [float(tok.strip()) for tok in val.split(",") if tok.strip()]
                    if len(row) == 4:
                        transform_rows.append(row)
    except OSError:
        return None
    except ValueError:
        return None
    if len(dims) < 3 or len(vox) < 3 or len(transform_rows) < 3:
        return None
    return {"size": dims, "spacing": vox, "transform": transform_rows[:3]}


def _mrinfo_header_dict(image_path: Path, mrinfo_cmd: str, env: dict[str, str]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=".mrinfo-") as tmp:
        out_json = Path(tmp) / "hdr.json"
        _run_command([mrinfo_cmd, image_path.as_posix(), "-json_all", out_json.as_posix()], env=env, log=None, capture=True)
        if not out_json.is_file() or out_json.stat().st_size == 0:
            raise RuntimeError(f"mrinfo failed to inspect image header: {image_path}")
        return json.loads(out_json.read_text(encoding="utf-8"))


def _validate_reverse_b0_sidecar_metadata(
    reverse_path: Path,
    reverse_affine: Any,
    forward_pe: Optional[str],
) -> None:
    """Validate optional sibling JSON sidecar on the reverse-PE b=0 image (PE direction & TotalReadoutTime)."""
    from shared.pe_resolution import ANATOMICAL_DESIGNATORS, _is_axis_code, axis_code_to_designator, designator_axis

    stem = _reverse_phase_stem(reverse_path)
    sidecar_path = reverse_path.parent / f"{stem}.json"
    if not sidecar_path.is_file():
        return
    try:
        payload = json.loads(sidecar_path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Reverse phase-encoded b=0 sidecar {sidecar_path.name} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Reverse phase-encoded b=0 sidecar {sidecar_path.name} must contain a JSON object.")

    if "TotalReadoutTime" in payload and payload["TotalReadoutTime"] is not None:
        raw_ro = payload["TotalReadoutTime"]
        if isinstance(raw_ro, bool) or not isinstance(raw_ro, (int, float)):
            raise ValueError(
                f"Reverse phase-encoded b=0 sidecar {sidecar_path.name} has non-numeric TotalReadoutTime: {raw_ro!r}."
            )
        ro_val = float(raw_ro)
        if not math.isfinite(ro_val) or ro_val <= 0.0:
            raise ValueError(
                f"Reverse phase-encoded b=0 sidecar {sidecar_path.name} TotalReadoutTime must be positive and finite "
                f"(received {raw_ro!r})."
            )

    raw_pe = payload.get("PhaseEncodingDirection")
    if raw_pe is not None:
        if not isinstance(raw_pe, str) or not raw_pe.strip():
            raise ValueError(
                f"Reverse phase-encoded b=0 sidecar {sidecar_path.name} has invalid PhaseEncodingDirection: {raw_pe!r}."
            )
        pe_text = raw_pe.strip()
        if pe_text.upper() in ANATOMICAL_DESIGNATORS:
            rev_designator: Optional[str] = pe_text.upper()
        elif _is_axis_code(pe_text.lower()):
            rev_designator = axis_code_to_designator(pe_text, reverse_affine) if reverse_affine is not None else None
            if rev_designator is None and reverse_affine is not None:
                raise ValueError(
                    f"Reverse phase-encoded b=0 sidecar {sidecar_path.name} PhaseEncodingDirection {raw_pe!r} "
                    "cannot be mapped cleanly to an anatomical axis (oblique or invalid affine)."
                )
        else:
            raise ValueError(
                f"Reverse phase-encoded b=0 sidecar {sidecar_path.name} has unrecognized PhaseEncodingDirection "
                f"{raw_pe!r}; expected one of AP, PA, LR, RL, IS, SI or i, i-, j, j-, k, k-."
            )
        if rev_designator is not None and forward_pe is not None:
            expected_rev = _OPPOSITE_PE_DESIGNATORS.get(forward_pe.upper())
            if expected_rev is not None and rev_designator != expected_rev:
                raise ValueError(
                    f"Reverse phase-encoded b=0 PhaseEncodingDirection ({rev_designator}, raw {raw_pe!r} in "
                    f"{sidecar_path.name}) is not opposite to main DWI PhaseEncodingDirection ({forward_pe}); "
                    f"Topup requires {expected_rev}."
                )
    elif payload.get("PhaseEncodingAxis") is not None:
        raw_axis = payload.get("PhaseEncodingAxis")
        if not isinstance(raw_axis, str) or not _is_axis_code(raw_axis.strip().lower()):
            raise ValueError(
                f"Reverse phase-encoded b=0 sidecar {sidecar_path.name} has invalid PhaseEncodingAxis: {raw_axis!r}."
            )
        if reverse_affine is not None and forward_pe is not None:
            axis_desig = axis_code_to_designator(raw_axis.strip().rstrip("-"), reverse_affine)
            if axis_desig is not None and designator_axis(axis_desig) != designator_axis(forward_pe.upper()):
                raise ValueError(
                    f"Reverse phase-encoded b=0 PhaseEncodingAxis ({raw_axis!r} -> {axis_desig} axis in "
                    f"{sidecar_path.name}) is incompatible with main DWI PhaseEncodingDirection ({forward_pe})."
                )


def validate_topup_and_eddy_inputs(
    raw: RawInputBundle,
    config: DwiPreprocessingConfig,
    acquisition: AcquisitionMetadata,
    *,
    backend: Optional[dict[str, Any]] = None,
    env: Optional[dict[str, str]] = None,
) -> None:
    """Validate PE direction, readout time, and forward/reverse b=0 spatial grids before merge/Topup/Eddy."""
    if not _effective_run_eddy(config):
        return
    if config.readout_time is not None:
        if isinstance(config.readout_time, bool):
            raise ValueError("TotalReadoutTime (readout_time) must be a positive finite number of seconds.")
        try:
            ro_val = float(config.readout_time)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"TotalReadoutTime (readout_time) must be a positive finite number: {config.readout_time!r}.") from exc
        if not math.isfinite(ro_val) or ro_val <= 0.0:
            raise ValueError(
                f"TotalReadoutTime (readout_time) must be positive and finite (received {config.readout_time!r})."
            )
    if config.phase_encoding_direction is not None and str(config.phase_encoding_direction).strip():
        if acquisition.phase_encoding is None:
            raise ValueError(
                f"Invalid phase_encoding_direction {config.phase_encoding_direction!r}: must be AP, PA, LR, RL, IS, SI "
                "or a non-oblique NIfTI axis code (i, i-, j, j-, k, k-)."
            )
    forward_pe = acquisition.require_phase_encoding()
    if acquisition.readout_time is not None:
        if not math.isfinite(float(acquisition.readout_time)) or float(acquisition.readout_time) <= 0.0:
            raise ValueError(f"Resolved TotalReadoutTime must be positive and finite (got {acquisition.readout_time!r}).")

    effective_reverse = _effective_reverse_phase_b0(config)
    if effective_reverse is None:
        return
    reverse_path = effective_reverse.expanduser().resolve()
    if not reverse_path.is_file():
        raise FileNotFoundError(f"Reverse phase-encoded b=0 image not found: {reverse_path}")

    import nibabel as nib
    import numpy as np

    fwd_img = nib.load(str(raw.dwi_path))
    fwd_shape = fwd_img.shape
    fwd_affine = np.asarray(fwd_img.affine, dtype=float)
    rev_name_lower = reverse_path.name.lower()
    rev_affine: Any = None

    if rev_name_lower.endswith((".nrrd", ".nhdr")):
        rev_shape3d, rev_affine = _read_nrrd_spatial_grid_aligned(reverse_path, fwd_affine)
        validate_reverse_b0_spatial_grid(
            fwd_shape,
            fwd_affine,
            rev_shape3d,
            rev_affine,
            forward_label="main DWI",
            reverse_label=f"reverse phase-encoded b=0 ({reverse_path.name})",
        )
    elif rev_name_lower.endswith((".mif", ".mif.gz")):
        from preprocessing.fsl_native import validate_b0_pair_headers

        mrinfo_cmd = ((backend or {}).get("commands") or {}).get("mrinfo") or _resolve_command("mrinfo", config)
        if mrinfo_cmd and Path(mrinfo_cmd).is_file():
            eff_env = env if env is not None else _external_command_environment(config)
            fwd_hdr = _mrinfo_header_dict(raw.dwi_path, str(mrinfo_cmd), eff_env)
            rev_hdr = _mrinfo_header_dict(reverse_path, str(mrinfo_cmd), eff_env)
            validate_b0_pair_headers(fwd_hdr, rev_hdr)
            rev_spacing = np.asarray(rev_hdr["spacing"][:3], dtype=float)
            rev_tf = np.asarray(rev_hdr["transform"][:3], dtype=float)
            rev_affine = np.eye(4)
            rev_affine[:3, :3] = rev_tf[:3, :3] * rev_spacing[None, :]
            rev_affine[:3, 3] = rev_tf[:3, 3]
        else:
            mif_hdr = _read_mif_header_dict(reverse_path)
            if mif_hdr is not None:
                rev_spacing = np.asarray(mif_hdr["spacing"][:3], dtype=float)
                rev_tf = np.asarray(mif_hdr["transform"][:3], dtype=float)
                rev_affine = np.eye(4)
                rev_affine[:3, :3] = rev_tf[:3, :3] * rev_spacing[None, :]
                rev_affine[:3, 3] = rev_tf[:3, 3]
                # Compare in MRtrix's RAS canonical representation against forward DWI.
                fwd_can = nib.as_closest_canonical(fwd_img)
                validate_reverse_b0_spatial_grid(
                    fwd_can.shape,
                    fwd_can.affine,
                    tuple(mif_hdr["size"][:3]),
                    rev_affine,
                    forward_label="main DWI",
                    reverse_label=f"reverse phase-encoded b=0 ({reverse_path.name})",
                )
    else:
        try:
            rev_img = nib.load(str(reverse_path))
        except Exception as exc:
            raise ValueError(f"Unable to read reverse phase-encoded b=0 image: {reverse_path}") from exc
        rev_affine = np.asarray(rev_img.affine, dtype=float)
        validate_reverse_b0_spatial_grid(
            fwd_shape,
            fwd_affine,
            rev_img.shape,
            rev_affine,
            forward_label="main DWI",
            reverse_label=f"reverse phase-encoded b=0 ({reverse_path.name})",
        )

    _validate_reverse_b0_sidecar_metadata(reverse_path, rev_affine, forward_pe)


def _reverse_b0_nifti_volume_index(image_path: Path, volume_count: int, b0_threshold: float) -> int:
    """Pick the first b=0 volume index for a 4D reverse-PE NIfTI using a matching sibling ``.bval``/``.bvals`` file.

    Falls back to ``0`` when no matching sibling bval file exists (for example, a pure multi-b0 reverse-PE stack).
    Raises ``ValueError`` if a matching sibling bval file explicitly confirms there is no volume with ``b <= b0_threshold``.
    """
    import math

    name = image_path.name
    lower = name.lower()
    if lower.endswith(".nii.gz"):
        stem = name[:-7]
    elif lower.endswith(".nii"):
        stem = name[:-4]
    else:
        stem = image_path.stem
    for suffix in (".bval", ".bvals"):
        candidate = image_path.parent / f"{stem}{suffix}"
        if not candidate.is_file():
            continue
        try:
            tokens = candidate.read_text(encoding="utf-8", errors="ignore").split()
            bvals = [float(token) for token in tokens]
        except ValueError:
            continue
        if len(bvals) != volume_count or not all(math.isfinite(b) for b in bvals):
            continue
        match = next((index for index, b in enumerate(bvals) if b <= float(b0_threshold)), None)
        if match is None:
            raise ValueError(
                f"Reverse phase-encoded image {image_path.name} has no b=0 volume "
                f"(minimum b-value in {candidate.name} is {min(bvals):g}, threshold {float(b0_threshold):g})."
            )
        return match
    return 0


def _build_commands(raw: RawInputBundle, config: DwiPreprocessingConfig, paths: dict[str, Path], backend: dict[str, Any],
                    acquisition: Optional[AcquisitionMetadata] = None) -> list[list[str]]:
    if acquisition is None:
        acquisition = resolve_acquisition_for(raw, config)
    subject = raw.subject_id
    denoised = paths["denoise_dir"] / f"{subject}_dwi_denoised.nii.gz"
    noise = paths["denoise_dir"] / f"{subject}_dwi_noise.nii.gz"
    degibbs = paths["denoise_dir"] / f"{subject}_dwi_degibbs.nii.gz"
    effective_eddy = _effective_run_eddy(config)
    effective_reverse = _effective_reverse_phase_b0(config) if effective_eddy else None
    reverse_phase_commands: list[list[str]] = []
    reverse_phase_b0 = (
        effective_reverse.expanduser().resolve()
        if effective_reverse is not None
        else None
    )
    if reverse_phase_b0 is not None:
        if not reverse_phase_b0.is_file():
            raise FileNotFoundError(f"Reverse phase-encoded b=0 image not found: {reverse_phase_b0}")
        reverse_name_lower = reverse_phase_b0.name.lower()
        normalized_reverse = paths["motion_dir"] / f"{subject}_reverse_phase_b0_4d.nii.gz"
        if reverse_name_lower.endswith((".nrrd", ".nhdr")):
            try:
                import nrrd

                header = nrrd.read_header(str(reverse_phase_b0))
                raw_sizes = header.get("sizes")
                sizes = () if raw_sizes is None else tuple(int(s) for s in raw_sizes)
                reverse_ndim = len(sizes)
            except Exception as exc:
                raise ValueError(f"Unable to read reverse phase-encoded b=0 NRRD image: {reverse_phase_b0}") from exc
            if reverse_ndim not in {3, 4} or any(s <= 0 for s in sizes):
                raise ValueError(
                    "reverse_phase_b0 must be a 3D or 4D image; "
                    f"received shape {sizes}."
                )
            reverse_phase_commands.append(
                [
                    "prepare-reverse-b0",
                    reverse_phase_b0.as_posix(),
                    normalized_reverse.as_posix(),
                    raw.dwi_path.as_posix(),
                ]
            )
            reverse_phase_b0 = normalized_reverse
        elif not reverse_name_lower.endswith((".mif", ".mif.gz")):
            import nibabel as nib

            try:
                reverse_shape = nib.load(str(reverse_phase_b0)).shape
            except Exception as exc:
                raise ValueError(f"Unable to read reverse phase-encoded b=0 image: {reverse_phase_b0}") from exc
            reverse_ndim = len(reverse_shape)
            if reverse_ndim not in {3, 4}:
                raise ValueError(
                    "reverse_phase_b0 must be a 3D image or a 4D image; "
                    f"received shape {reverse_shape}."
                )
            reverse_volume_count = int(reverse_shape[3]) if reverse_ndim == 4 else 1
            if reverse_volume_count < 1:
                raise ValueError(
                    "reverse_phase_b0 must contain at least one reverse phase-encoded b=0 volume; "
                    f"received shape {reverse_shape}."
                )
            if reverse_ndim == 3:
                normalize_cmd = [
                    backend["commands"]["mrconvert"] or "mrconvert",
                    reverse_phase_b0.as_posix(),
                    normalized_reverse.as_posix(),
                    "-axes",
                    "0,1,2,-1",
                ]
                if config.force:
                    normalize_cmd.append("-force")
                reverse_phase_commands.append(normalize_cmd)
                reverse_phase_b0 = normalized_reverse
            elif reverse_volume_count > 1:
                b0_index = _reverse_b0_nifti_volume_index(
                    reverse_phase_b0, reverse_volume_count, config.b0_threshold
                )
                normalize_cmd = [
                    backend["commands"]["mrconvert"] or "mrconvert",
                    reverse_phase_b0.as_posix(),
                    normalized_reverse.as_posix(),
                    "-coord",
                    "3",
                    str(b0_index),
                    "-axes",
                    "0,1,2,-1",
                ]
                if config.force:
                    normalize_cmd.append("-force")
                reverse_phase_commands.append(normalize_cmd)
                reverse_phase_b0 = normalized_reverse
        forward_b0s = paths["motion_dir"] / f"{subject}_forward_b0s.mif"
        forward_b0 = paths["motion_dir"] / f"{subject}_forward_b0_4d.nii.gz"
        se_epi_pair = paths["motion_dir"] / f"{subject}_se_epi_pair.mif"
        reverse_phase_commands.extend(
            [
                [
                    backend["commands"]["dwiextract"] or "dwiextract",
                    degibbs.as_posix(),
                    forward_b0s.as_posix(),
                    "-fslgrad",
                    raw.bvec_path.as_posix(),
                    raw.bval_path.as_posix(),
                    "-bzero",
                ],
                [
                    backend["commands"]["mrconvert"] or "mrconvert",
                    forward_b0s.as_posix(),
                    forward_b0.as_posix(),
                    "-coord",
                    "3",
                    "0",
                    "-axes",
                    "0,1,2,-1",
                ],
                [
                    backend["commands"]["mrcat"] or "mrcat",
                    forward_b0.as_posix(),
                    reverse_phase_b0.as_posix(),
                    se_epi_pair.as_posix(),
                    "-axis",
                    "3",
                ],
            ]
        )
        if config.force:
            for command in reverse_phase_commands:
                if command[0] != "prepare-reverse-b0" and "-force" not in command:
                    command.append("-force")
        reverse_phase_b0 = se_epi_pair

    if not effective_eddy:
        correction_commands = [
            ["copy", degibbs.as_posix(), paths["final_dwi"].as_posix()],
            ["copy", raw.bval_path.as_posix(), paths["final_bval"].as_posix()],
            ["copy", raw.bvec_path.as_posix(), paths["final_bvec"].as_posix()],
        ]
    elif backend.get("execution_backend") == "native":
        correction_cmd = ["native-fsl-correction", degibbs.as_posix()]
        if reverse_phase_b0 is not None:
            correction_cmd.append(reverse_phase_b0.as_posix())
        correction_commands = [correction_cmd]
    else:
        correction_cmd = [
            backend["commands"]["dwifslpreproc"] or "dwifslpreproc",
            degibbs.as_posix(),
            paths["final_dwi"].as_posix(),
            "-fslgrad",
            raw.bvec_path.as_posix(),
            raw.bval_path.as_posix(),
            "-export_grad_fsl",
            paths["final_bvec"].as_posix(),
            paths["final_bval"].as_posix(),
            "-scratch",
            (paths["motion_dir"] / "dwifslpreproc_tmp").as_posix(),
            "-nocleanup",
            "-eddy_options",
            _resolved_eddy_options(config),
        ]
        if config.force:
            correction_cmd.append("-force")
        if reverse_phase_b0 is not None:
            correction_cmd.extend(["-rpe_pair", "-se_epi", reverse_phase_b0.as_posix()])
        else:
            correction_cmd.append("-rpe_none")
        # W2b (D1): `-pe_dir <designator>` is always passed (it is mandatory with -rpe_none and -rpe_pair; no -json_import,
        # because the JSON cannot replace it). `-readout_time` only when a value is known: dwifslpreproc assumes 0.1 s.
        # A plan with no direction (rule 4) shows no -pe_dir; executing such a plan fails first.
        if acquisition.phase_encoding is not None:
            correction_cmd.extend(["-pe_dir", acquisition.phase_encoding])
        if acquisition.readout_time_for_command is not None:
            correction_cmd.extend(["-readout_time", str(acquisition.readout_time_for_command)])
        correction_commands = [correction_cmd]

    # W2a (D1): DIPY MP-PCA, then DIPY Gibbs removal (a copy when skipped, so the next step reads one fixed path).
    # These two steps run in this process; the MRtrix3 and FSL steps below run as commands.
    if config.run_denoise:
        denoise_commands = [
            ["dipy", "mppca", raw.dwi_path.as_posix(), denoised.as_posix(), noise.as_posix()],
            ["dipy", "gibbs_removal" if config.dipy_run_gibbs else "skip_gibbs_removal",
             denoised.as_posix(), degibbs.as_posix()],
        ]
    else:
        denoise_commands = [
            ["copy", raw.dwi_path.as_posix(), degibbs.as_posix()],
        ]

    return [
        *denoise_commands,
        *reverse_phase_commands,
        *correction_commands,
        [
            "optional-output",
            paths["eddy_parameters"].as_posix(),
            paths["eddy_movement_rms"].as_posix(),
            paths["eddy_outlier_map"].as_posix(),
            paths["eddy_outlier_report"].as_posix(),
            paths["eddy_cnr_maps"].as_posix(),
            paths["eddy_residuals"].as_posix(),
        ],
    ]


def _collect_eddy_qc_artifacts(paths: dict[str, Path]) -> dict[str, Any]:
    """Copy Eddy products retained by ``dwifslpreproc -nocleanup`` into QC.

    MRtrix writes these files below its scratch directory with the fixed
    ``dwi_post_eddy`` prefix.  They are useful pipeline outputs, but are not
    copied by ``dwifslpreproc`` itself.  In particular, Eddy's outlier map is
    a text matrix rather than a NIfTI image, so preserve it as ``.txt``.
    """

    scratch_dir = _correction_scratch(paths)
    source_prefix = scratch_dir / "dwi_post_eddy"
    specifications = {
        "eddy_parameters": (".eddy_parameters", "text/eddy-parameters"),
        "eddy_movement_rms": (".eddy_movement_rms", "text/eddy-movement-rms"),
        "eddy_outlier_map": (".eddy_outlier_map", "text/eddy-outlier-map"),
        "eddy_outlier_report": (".eddy_outlier_report", "text/eddy-outlier-report"),
        "eddy_cnr_maps": (".eddy_cnr_maps.nii.gz", "application/nifti+gzip"),
        "eddy_residuals": (".eddy_residuals.nii.gz", "application/nifti+gzip"),
    }
    artifact_records: dict[str, Any] = {}
    for key, (suffix, content_type) in specifications.items():
        source = Path(f"{source_prefix}{suffix}")
        destination = paths[key]
        record = {
            "source": source.as_posix(),
            "destination": destination.as_posix(),
            "content_type": content_type,
            "status": "missing",
        }
        if source.is_file():
            destination.parent.mkdir(parents=True, exist_ok=True)
            src_stat = source.stat()
            if not (
                destination.is_file()
                and not destination.is_symlink()
                and destination.stat().st_size == src_stat.st_size
                and destination.stat().st_mtime_ns == src_stat.st_mtime_ns
            ):
                shutil.copy2(source, destination)
            record.update({"status": "copied", "size": destination.stat().st_size})
        artifact_records[key] = record
    return {
        "scratch_dir": scratch_dir.as_posix(),
        "artifacts": artifact_records,
    }


def _materialize_dirs(paths: dict[str, Path]) -> None:
    for key in ("import_dir", "gradients_dir", "denoise_dir", "motion_dir", "mask_dir", "qc_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)


def _copy_input_sidecars(raw: RawInputBundle, paths: dict[str, Path]) -> None:
    _link_or_copy(raw.dwi_path, paths["import_dir"] / raw.dwi_path.name)
    _link_or_copy(raw.bval_path, paths["import_dir"] / raw.bval_path.name)
    _link_or_copy(raw.bvec_path, paths["import_dir"] / raw.bvec_path.name)
    if raw.mask_path.is_file():
        _link_or_copy(raw.mask_path, paths["import_dir"] / raw.mask_path.name)
    _link_or_copy(raw.bval_path, paths["gradients_dir"] / f"{raw.subject_id}_gradients_checked.bval")
    _link_or_copy(raw.bvec_path, paths["gradients_dir"] / f"{raw.subject_id}_gradients_checked.bvec")


def _save_float32(array: Any, source_img: Any, path: Path) -> None:
    """Save with the affine and header of the imported DWI, but as float32.

    The source header may name an integer type. Saving denoised floats in that type would round them
    (and write a scale), so the eddy input would differ from the DIPY result.
    """
    import nibabel as nib
    import numpy as np

    image = nib.Nifti1Image(np.asarray(array, dtype=np.float32), source_img.affine, source_img.header)
    image.set_data_dtype(np.float32)
    out_str = str(path)
    disable_accel = os.environ.get("SLICERDSPACE_DISABLE_DIPY_ACCEL", "").strip().lower() in {"1", "true", "yes"}
    if out_str.endswith(".nii.gz") and not disable_accel:
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.GzipFile(filename=out_str, mode="wb", compresslevel=1) as gz:
            holder = nib.FileHolder(fileobj=gz)
            image.to_file_map({"image": holder, "header": holder})
    else:
        nib.save(image, out_str)


def _run_fused_mppca_gibbs(
    mppca_cmd: list[str],
    gibbs_cmd: list[str],
    *,
    log: Optional[Callable[[str], None]],
    device: Optional[str] = None,
    max_workers: Optional[int] = None,
) -> None:
    """Run consecutive MP-PCA + Gibbs removal in RAM without re-gunzipping intermediate 4D NIfTI."""
    import nibabel as nib
    import numpy as np
    from preprocessing.dipy_accel import (
        gibbs_removal_accelerated,
        mppca_accelerated,
        resolve_dipy_accel_device,
    )

    def emit(message: str) -> None:
        if log:
            log("[Preproc:DIPY] " + message)

    source_img = nib.load(mppca_cmd[2])
    data = source_img.get_fdata(dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"DIPY preprocessing expects a 4D DWI image, got shape {data.shape}")
    resolved_dev = resolve_dipy_accel_device(device)
    emit(f"Loaded {mppca_cmd[2]} shape={data.shape}")
    emit(f"Running fused MP-PCA denoising + Gibbs removal in RAM (device={resolved_dev})")
    denoised, sigma = mppca_accelerated(
        data,
        patch_radius=2,
        return_sigma=True,
        device=device,
        max_workers=max_workers,
    )
    del data
    with ThreadPoolExecutor(max_workers=3) as io_pool:
        fut_noise = io_pool.submit(_save_float32, sigma, source_img, Path(mppca_cmd[4]))
        fut_denoised = io_pool.submit(_save_float32, denoised, source_img, Path(mppca_cmd[3]))
        corrected = gibbs_removal_accelerated(
            denoised,
            slice_axis=2,
            inplace=False,
            device=device,
            max_workers=max_workers,
        )
        fut_degibbs = io_pool.submit(_save_float32, corrected, source_img, Path(gibbs_cmd[3]))
        fut_noise.result()
        fut_denoised.result()
        fut_degibbs.result()


def _run_dipy_step(
    command: list[str],
    *,
    log: Optional[Callable[[str], None]],
    device: Optional[str] = None,
    max_workers: Optional[int] = None,
) -> dict[str, Any]:
    """Run one in-process DIPY step of the plan: ``mppca`` or ``gibbs_removal`` (``skip_gibbs_removal`` copies)."""
    step = command[1]

    def emit(message: str) -> None:
        if log:
            log("[Preproc:DIPY] " + message)

    if step == "skip_gibbs_removal":
        emit("Skipping Gibbs removal")
        shutil.copy2(command[2], command[3])
        return {"status": "skipped"}
    try:
        import nibabel as nib
        import numpy as np
    except Exception as exc:
        raise RuntimeError("DIPY preprocessing requires nibabel, numpy, and dipy.") from exc
    source_img = nib.load(command[2])
    data = source_img.get_fdata(dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"DIPY preprocessing expects a 4D DWI image, got shape {data.shape}")
    emit(f"Loaded {command[2]} shape={data.shape}")
    disable_accel = os.environ.get("SLICERDSPACE_DISABLE_DIPY_ACCEL", "").strip().lower() in {"1", "true", "yes"}
    if step == "mppca":
        try:
            from dipy.denoise.localpca import mppca
        except Exception as exc:
            raise RuntimeError("DIPY preprocessing requires nibabel, numpy, and dipy.") from exc
        if disable_accel:
            emit("Running MP-PCA denoising (baseline DIPY)")
            denoised, sigma = mppca(data, patch_radius=2, return_sigma=True)
        else:
            from preprocessing.dipy_accel import mppca_accelerated, resolve_dipy_accel_device

            resolved_dev = resolve_dipy_accel_device(device)
            emit(f"Running MP-PCA denoising (device={resolved_dev})")
            denoised, sigma = mppca_accelerated(
                data,
                patch_radius=2,
                return_sigma=True,
                device=device,
                max_workers=max_workers,
            )
        _save_float32(denoised, source_img, Path(command[3]))
        _save_float32(sigma, source_img, Path(command[4]))
        return {"status": "completed", "method": "dipy_mppca", "noise_map": command[4]}
    if step == "gibbs_removal":
        try:
            from dipy.denoise.gibbs import gibbs_removal
        except Exception as exc:
            raise RuntimeError("DIPY preprocessing requires nibabel, numpy, and dipy.") from exc
        if disable_accel:
            emit("Running Gibbs removal (baseline DIPY)")
            corrected = gibbs_removal(data, slice_axis=2, inplace=False, num_processes=1)
        else:
            from preprocessing.dipy_accel import gibbs_removal_accelerated, resolve_dipy_accel_device

            resolved_dev = resolve_dipy_accel_device(device)
            emit(f"Running Gibbs removal (device={resolved_dev})")
            corrected = gibbs_removal_accelerated(
                data,
                slice_axis=2,
                inplace=False,
                device=device,
                max_workers=max_workers,
            )
        _save_float32(corrected, source_img, Path(command[3]))
        return {"status": "completed", "method": "dipy_gibbs_removal"}
    raise ValueError(f"Unknown DIPY preprocessing step: {step}")


def _mrtrix_backend_details(
    paths: dict[str, Path],
    subject_id: str,
    config: DwiPreprocessingConfig,
    acquisition: Optional[AcquisitionMetadata] = None,
    backend: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    native = backend is not None and backend.get("execution_backend") == "native"
    effective_eddy = _effective_run_eddy(config)
    effective_reverse = _effective_reverse_phase_b0(config) if effective_eddy else None
    if native:
        env = {"FSLDIR": backend.get("fsl_root"), "FSLOUTPUTTYPE": "NIFTI_GZ"}
        device_execution = (
            {"resolved_device": "cpu", "executable": (backend.get("commands") or {}).get("eddy")}
            if effective_eddy else None
        )
    else:
        env = _external_command_environment(config)
        device_record = paths["qc_dir"] / "eddy_device.json"
        device_execution = json.loads(device_record.read_text(encoding="utf-8")) if (effective_eddy and device_record.is_file()) else None
    return {
        "execution_backend": "native" if native else "legacy",
        "fsl_environment": {
            "FSLDIR": env.get("FSLDIR"),
            "FSLOUTPUTTYPE": env.get("FSLOUTPUTTYPE"),
        },
        "route": run_route_id(config),
        **({"acquisition": acquisition.to_record()} if acquisition is not None else {}),
        "denoise": ({"status": "completed", "method": "dipy_mppca"} if config.run_denoise
                    else {"status": "skipped", "method": "dipy_mppca"}),
        "gibbs": ({"status": "completed", "method": "dipy_gibbs_removal"} if (config.run_denoise and config.dipy_run_gibbs)
                  else {"status": "skipped", "method": "dipy_gibbs_removal"}),
        "motion": ({"status": "completed", "method": "fsl_eddy", "device_execution": device_execution} if effective_eddy
                   else {"status": "skipped", "method": "fsl_eddy", "device_execution": None}),
        "bias_correction": {
            "status": "not_run",
            "reason": "Bias-field correction is not included in the MRtrix/FSL command sequence.",
        },
        "susceptibility_distortion_correction": {
            "status": "completed" if effective_reverse is not None else "not_run",
            "method": "fsl_topup" if effective_reverse is not None else "rpe_none",
        },
        "diagnostics": _collect_dwifslpreproc_diagnostics(paths, subject_id),
        "eddy_qc_collection": _collect_eddy_qc_artifacts(paths),
        "scratch_dir": _correction_scratch(paths).as_posix(),
    }


def _execute_preprocessing_backend(
    raw: RawInputBundle,
    config: DwiPreprocessingConfig,
    paths: dict[str, Path],
    backend: dict[str, Any],
    commands: list[list[str]],
    *,
    log: Optional[Callable[[str], None]],
    acquisition: Optional[AcquisitionMetadata] = None,
    native_commands: Optional[list[list[str]]] = None,
    force: bool = False,
) -> dict[str, Any]:
    native = backend.get("execution_backend") == "native"
    if native and _effective_run_eddy(config):
        from preprocessing.fsl_runtime import native_fsl_environment

        env = native_fsl_environment(backend.get("commands") or {}, backend.get("fsl_root"))
    else:
        env = _external_command_environment(config)

    def run_native_command(argv: list[str]) -> str:
        if native_commands is not None:
            native_commands.append(list(argv))
        return _run_command(argv, env=env, log=log, capture=True)

    disable_accel = os.environ.get("SLICERDSPACE_DISABLE_DIPY_ACCEL", "").strip().lower() in {"1", "true", "yes"}
    effective_threads = _resolved_eddy_threads(config)
    idx = 0
    while idx < len(commands):
        command = commands[idx]
        if command[0] == "dipy":
            if (
                not disable_accel
                and command[1] == "mppca"
                and idx + 1 < len(commands)
                and commands[idx + 1][:2] == ["dipy", "gibbs_removal"]
                and commands[idx + 1][2] == command[3]
            ):
                if log:
                    log("[Preproc] Fused DIPY steps: mppca + gibbs_removal")
                _run_fused_mppca_gibbs(
                    command,
                    commands[idx + 1],
                    log=log,
                    device=config.dipy_device,
                    max_workers=effective_threads,
                )
                idx += 2
                continue
            if log:
                log("[Preproc] DIPY step: " + " ".join(command[1:]))
            _run_dipy_step(
                command,
                log=log,
                device=config.dipy_device,
                max_workers=effective_threads,
            )
            idx += 1
            continue
        if command[0] == "copy":
            shutil.copy2(command[1], command[2])
            idx += 1
            continue
        if command[0] == "prepare-reverse-b0":
            ref_path = Path(command[3]) if len(command) > 3 and command[3] else None
            _prepare_reverse_b0_nrrd(
                Path(command[1]),
                Path(command[2]),
                b0_threshold=config.b0_threshold,
                reference_path=ref_path,
            )
            idx += 1
            continue
        if command[0] == "optional-output":
            idx += 1
            continue
        if Path(command[0]).stem.lower() == "mrcat" and len(command) >= 4:
            if acquisition is None:
                acquisition = resolve_acquisition_for(raw, config)
            validate_topup_and_eddy_inputs(raw, config, acquisition, backend=backend, env=env)
            # Also verify the exact two inputs about to be concatenated when they are real images on disk.
            fwd_in, rev_in = Path(command[1]), Path(command[2])
            if fwd_in.is_file() and rev_in.is_file():
                if fwd_in.name.lower().endswith((".nii", ".nii.gz")) and rev_in.name.lower().endswith((".nii", ".nii.gz")):
                    import nibabel as nib

                    try:
                        fwd_loaded = nib.load(str(fwd_in))
                        rev_loaded = nib.load(str(rev_in))
                    except Exception:
                        fwd_loaded = rev_loaded = None
                    if fwd_loaded is not None and rev_loaded is not None:
                        validate_reverse_b0_spatial_grid(
                            fwd_loaded.shape,
                            fwd_loaded.affine,
                            rev_loaded.shape,
                            rev_loaded.affine,
                            forward_label=f"forward b=0 ({fwd_in.name})",
                            reverse_label=f"reverse b=0 ({rev_in.name})",
                        )
        if command[0] == "native-fsl-correction":
            from preprocessing.fsl_native import run_fsl_correction

            if acquisition is None:
                acquisition = resolve_acquisition_for(raw, config)
            run_fsl_correction(
                input_dwi=Path(command[1]), input_bvec=raw.bvec_path, input_bval=raw.bval_path,
                output_dwi=paths["final_dwi"], output_bvec=paths["final_bvec"], output_bval=paths["final_bval"],
                scratch_dir=_correction_scratch(paths), phase_encoding=acquisition.require_phase_encoding(),
                readout_time=acquisition.readout_time_for_command,
                se_epi_pair=Path(command[2]) if len(command) == 3 else None,
                topup_config=Path(backend["topup_config"]) if backend["topup_config"] else None,
                commands=backend["commands"], eddy_options=_resolved_eddy_options(config).split(),
                run_stage=run_native_command, force=force,
            )
        elif native:
            run_native_command(command)
        else:
            exec_cmd = list(command)
            if not disable_accel and effective_threads > 1:
                tool_name = Path(exec_cmd[0]).name.lower()
                if tool_name in {"dwifslpreproc", "mrconvert", "dwiextract", "mrcat"} and "-nthreads" not in exec_cmd:
                    exec_cmd.extend(["-nthreads", str(effective_threads)])
                if tool_name == "dwifslpreproc" and "-rpe_pair" in exec_cmd and "-topup_options" not in exec_cmd:
                    exec_cmd.extend(["-topup_options", f" --nthr={effective_threads}"])
            runtime_command = preprocessing_launcher(exec_cmd, paths["qc_dir"] / "eddy_device.json", env)
            _run_command(runtime_command, env=env, log=log)
        idx += 1
    return _mrtrix_backend_details(paths, raw.subject_id, config, acquisition, backend)


def _preprocessing_warnings(
    backend: dict[str, Any],
    backend_details: dict[str, Any],
) -> list[str]:
    # W2b: the conflict warnings and the assumption flags of the PE and readout resolution (result panel and QC).
    acquisition = backend_details.get("acquisition") or {}
    warnings = list(acquisition.get("warnings") or [])
    if acquisition.get("pe_sign_assumed"):
        warnings.append("pe_sign_assumed: the sidecar names only the phase-encoding axis; a positive sign was assumed.")
    if acquisition.get("readout_assumed"):
        if backend.get("execution_backend") == "native":
            warnings.append("readout_assumed: no readout time was given; preprocessing assumed 0.1 s.")
        else:
            warnings.append("readout_assumed: no readout time was given; dwifslpreproc assumed 0.1 s.")
    warnings += [
        f"Diagnostic command not found: {command}"
        for command in backend.get("missing_diagnostic", [])
    ]
    capabilities = backend.get("capabilities") or {}
    if "denoise" in capabilities and capabilities["denoise"] in {False, None, "none", "skip", ""}:
        warnings.append("Denoising was not applied by the selected preprocessing configuration.")
    for capability, message in (
        ("gibbs_removal", "Gibbs-ringing removal was not applied."),
        ("susceptibility_distortion_correction", "Susceptibility-distortion correction was not applied."),
    ):
        if capabilities.get(capability) is False:
            warnings.append(message)
    bias_status = backend_details.get("bias_correction") or {}
    if bias_status.get("status") in {"not_available", "not_supported", "not_run", "skipped"}:
        warnings.append("Bias-field correction was not applied by the selected backend.")
    return list(dict.fromkeys(warnings))


def _finalize_brain_mask(
    raw: RawInputBundle,
    config: DwiPreprocessingConfig,
    paths: dict[str, Path],
    backend_details: dict[str, Any],
    *,
    log: Optional[Callable[[str], None]],
) -> dict[str, Any]:
    selected = _brain_mask_config(config)
    motion_applied = (backend_details.get("motion") or {}).get("status") == "completed"
    preserve_input = selected.backend == "provided"
    warnings: list[str] = []
    if preserve_input:
        if not raw.mask_path.is_file():
            raise FileNotFoundError(
                f"Selected brain-mask backend '{selected.backend}' requires a supplied mask: {raw.mask_path}"
            )
        validate_brain_mask(paths["final_dwi"], raw.mask_path)
        # Replace the output itself, never follow a previous output link back
        # to the canonical supplied mask.
        with tempfile.TemporaryDirectory(dir=paths["final_mask"].parent, prefix=".provided-mask-") as temporary:
            temporary_mask = Path(temporary) / paths["final_mask"].name
            shutil.copy2(raw.mask_path, temporary_mask)
            os.replace(temporary_mask, paths["final_mask"])
        qc = validate_brain_mask(paths["final_dwi"], paths["final_mask"])
        if motion_applied:
            warnings.append(
                "Supplied brain mask was copied after motion correction. Geometry QC cannot verify "
                "anatomical alignment; human review of the mask on the corrected DWI is required."
            )
        generation = {
            "backend": "provided",
            "requested_backend": selected.backend,
            "status": "copied",
            "method": "input_mask",
            "source": raw.mask_path.as_posix(),
        }
    else:
        if paths["final_mask"].is_symlink() or (
            paths["final_mask"].is_file() and paths["final_mask"].stat().st_nlink > 1
        ):
            paths["final_mask"].unlink()
        generation = dict(generate_brain_mask(
            paths["final_dwi"], paths["final_bval"], paths["final_bvec"], paths["final_mask"],
            config=selected, force=True, log=log,
        ))
        generation.update({
            "backend": selected.backend,
            "method": f"{selected.backend}_corrected_dwi",
            "source": paths["final_dwi"].as_posix(),
        })
        qc = validate_brain_mask(paths["final_dwi"], paths["final_mask"])
    generation.update({
        "selected_backend": selected.backend,
        "qc": qc,
        "motion_applied": motion_applied,
        "human_review_required": qc["human_review"] == "pending",
        "alignment_review_required": preserve_input and motion_applied,
        "warnings": warnings,
    })
    # Copying preserves the input timestamp, but freshness is relative to the
    # corrected DWI. A provided mask must not be refreshed on every cache hit.
    if paths["final_mask"].stat().st_mtime_ns < paths["final_dwi"].stat().st_mtime_ns:
        stat = paths["final_mask"].stat()
        os.utime(paths["final_mask"], ns=(stat.st_atime_ns, paths["final_dwi"].stat().st_mtime_ns))
    return generation


def _detach_linked_backend_outputs(paths: dict[str, Path]) -> None:
    candidates = [paths[key] for key in ("final_dwi", "final_bval", "final_bvec")]
    for key in ("denoise_dir", "motion_dir"):
        candidates.extend(path for path in paths[key].rglob("*") if path.is_file() or path.is_symlink())
    for path in candidates:
        if path.is_symlink() or (path.is_file() and path.stat().st_nlink > 1):
            path.unlink()


@scheduled_stage('preprocessing', gpu=True)
def run_dwi_preprocessing(
    config: DwiPreprocessingConfig,
    *,
    log: Optional[Callable[[str], None]] = None,
) -> DwiPreprocessingResult:
    raw = _resolve_raw_inputs(config)
    output_dir = _resolve_output_dir(raw, config.output_dir, output_root=config.output_root)
    paths = _planned_paths(output_dir, raw.subject_id)
    backend = inspect_preprocessing_backend(config)
    native = backend["execution_backend"] == "native"
    if native:
        paths["native_scratch"] = paths["motion_dir"] / "fsl_native_tmp"
    mask_backend = inspect_brain_mask_backend(_brain_mask_config(config))
    acquisition = resolve_acquisition_for(raw, config)
    if config.execute and not config.skip and config.run_topup and not config.run_eddy and config.reverse_phase_b0 is None:
        raise ValueError(
            "Topup was selected without a reverse phase-encoding b=0 image (--preprocessing-reverse-phase-b0). "
            "Provide a reverse-PE b=0 image or disable Topup."
        )
    if config.execute and not config.skip and _effective_run_eddy(config):
        # Rule 4 (T1 D1) + pre-merge Topup geometry & PE/readout validation:
        # fails fast before the manifest, the scratch folder, denoising, or mrcat.
        validate_topup_and_eddy_inputs(raw, config, acquisition, backend=backend)
    commands = _build_commands(raw, config, paths, backend, acquisition)
    command_records = [{"argv": command} for command in commands]
    try:
        previous_manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        previous_manifest = {}
    cache_key = _preprocessing_cache_key(raw, config, backend, mask_backend, acquisition)

    output_dir.mkdir(parents=True, exist_ok=True)
    status = "skipped" if config.skip else ("planned" if not config.execute else "running" if native else "completed")

    manifest: dict[str, Any] = {
        "kind": "dwi_preprocessing",
        "version": 2,
        "subject_id": raw.subject_id,
        "status": status,
        "optional": True,
        "execute": config.execute,
        "skip": config.skip,
        "backend": backend,
        "brain_mask_backend": mask_backend,
        "acquisition": acquisition.to_record(),
        "inputs": {
            "subject_root": raw.subject_root.as_posix(),
            "dwi": raw.dwi_path.as_posix(),
            "bval": raw.bval_path.as_posix(),
            "bvec": raw.bvec_path.as_posix(),
            "mask": raw.mask_path.as_posix(),
            "reverse_phase_b0": config.reverse_phase_b0.as_posix() if config.reverse_phase_b0 else None,
        },
        "parameters": {
            "eddy_threads": _resolved_eddy_threads(config),
            "eddy_second_level_model": (
                config.eddy_second_level_model.strip().lower()
                if config.eddy_second_level_model is not None
                else None
            ),
            "brain_mask_backend": _brain_mask_config(config).backend,
            "final_mask_method": (
                "input_mask" if config.brain_mask_backend == "provided"
                else f"{config.brain_mask_backend}_corrected_dwi"
            ),
        },
        "outputs": {key: value.as_posix() for key, value in paths.items() if key not in {"import_dir", "gradients_dir", "denoise_dir", "motion_dir", "mask_dir", "qc_dir"}},
        "commands": command_records,
        "cache_key": cache_key,
    }
    if native:
        manifest["native_commands"] = []
    _write_json(paths["manifest"], manifest)

    if config.skip:
        _write_json(paths["qc"], {"kind": "dwi_preprocessing_qc", "status": "skipped", "subject_id": raw.subject_id, "checks": []})
        return DwiPreprocessingResult(raw.subject_id, "skipped", output_dir, paths["manifest"], paths["qc"], paths["final_dwi"], paths["final_bval"], paths["final_bvec"], paths["final_mask"])

    if not config.execute:
        _write_json(paths["qc"], {
            "kind": "dwi_preprocessing_qc", "status": "planned", "subject_id": raw.subject_id,
            "backend_available": backend["available"], "brain_mask_backend": mask_backend,
        })
        return DwiPreprocessingResult(raw.subject_id, "planned", output_dir, paths["manifest"], paths["qc"], paths["final_dwi"], paths["final_bval"], paths["final_bvec"], paths["final_mask"])

    if not backend["available"]:
        missing = ", ".join(backend.get("missing_required", []))
        error = f"DWI preprocessing is unavailable ({ROUTE_ID}): {missing}"
        if native:
            manifest.update(status="failed", error=error)
            _write_json(paths["manifest"], manifest)
            _write_json(paths["qc"], {"kind": "dwi_preprocessing_qc", "status": "failed",
                                      "subject_id": raw.subject_id, "backend_available": False, "error": error})
        raise RuntimeError(error)

    cached_core_ready = (
        not config.force
        and previous_manifest.get("status") in {"completed", "cached"}
        and preprocessing_cache_key_matches(previous_manifest.get("cache_key"), cache_key)
        and paths["final_dwi"].is_file()
        and paths["final_bval"].is_file()
        and paths["final_bvec"].is_file()
    )
    mask_refresh_status = "not_needed"
    if cached_core_ready:
        if native:
            manifest["native_commands"] = previous_manifest.get("native_commands", [])
        try:
            previous_qc = json.loads(paths["qc"].read_text(encoding="utf-8")) if paths["qc"].is_file() else {}
        except (OSError, ValueError):
            previous_qc = {}
        previous_details = previous_qc.get("backend_details") if isinstance(previous_qc, dict) else None
        backend_details: dict[str, Any] = (
            dict(previous_details) if isinstance(previous_details, dict) else {}
        )
        backend_details = _mrtrix_backend_details(paths, raw.subject_id, config, acquisition, backend)
        try:
            mask_is_stale = (
                not paths["final_mask"].is_file()
                or paths["final_mask"].stat().st_mtime_ns < paths["final_dwi"].stat().st_mtime_ns
            )
            mask_generation = previous_qc.get("mask_generation") or previous_manifest.get("mask_generation") or {}
            if mask_is_stale:
                _materialize_dirs(paths)
                mask_generation = _finalize_brain_mask(raw, config, paths, backend_details, log=log)
                mask_refresh_status = (
                    "copied_from_input" if mask_generation["status"] == "copied"
                    else "regenerated_from_corrected_dwi"
                )
            else:
                mask_generation = dict(mask_generation)
                mask_generation["qc"] = validate_brain_mask(paths["final_dwi"], paths["final_mask"])
        except Exception as exc:
            manifest.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            _write_json(paths["manifest"], manifest)
            _write_json(paths["qc"], {
                "kind": "dwi_preprocessing_qc", "status": "failed", "subject_id": raw.subject_id,
                "brain_mask_backend": mask_backend, "error": manifest["error"],
            })
            raise
        manifest.update({"status": "cached", "mask_generation": mask_generation})
        manifest["parameters"]["final_mask_method"] = mask_generation.get("method")
        _write_json(paths["manifest"], manifest)
        _write_json(
            paths["qc"],
            {
                "kind": "dwi_preprocessing_qc",
                "version": 1,
                "status": "completed",
                "reuse_status": "cached",
                "mask_refresh_status": mask_refresh_status,
                "subject_id": raw.subject_id,
                "files": {
                    "final_dwi": _file_record(paths["final_dwi"]),
                    "final_bval": _file_record(paths["final_bval"]),
                    "final_bvec": _file_record(paths["final_bvec"]),
                    "final_mask": _file_record(paths["final_mask"]),
                    "eddy_parameters": _file_record(paths["eddy_parameters"]),
                    "eddy_movement_rms": _file_record(paths["eddy_movement_rms"]),
                    "eddy_outlier_map": _file_record(paths["eddy_outlier_map"]),
                    "eddy_outlier_report": _file_record(paths["eddy_outlier_report"]),
                    "eddy_cnr_maps": _file_record(paths["eddy_cnr_maps"]),
                    "eddy_residuals": _file_record(paths["eddy_residuals"]),
                },
                "backend_details": backend_details,
                "warnings": _preprocessing_warnings(backend, backend_details) + mask_generation.get("warnings", []),
                "acquisition": acquisition.to_record(),
                "brain_mask_backend": mask_backend,
                "brain_mask_qc": mask_generation["qc"],
                "mask_generation": mask_generation,
            },
        )
        return DwiPreprocessingResult(raw.subject_id, "cached", output_dir, paths["manifest"], paths["qc"], paths["final_dwi"], paths["final_bval"], paths["final_bvec"], paths["final_mask"], acquisition)

    _materialize_dirs(paths)
    overwrite_existing = config.force or bool(previous_manifest)
    if overwrite_existing:
        # A cache miss may leave old intermediates/final files. Overwrite them
        # instead of failing at the first MRtrix3 command, and never resume an old eddy scratch.
        commands = _build_commands(raw, replace(config, force=True), paths, backend, acquisition)
        manifest["commands"] = [{"argv": command} for command in commands]
        manifest["overwrite_existing"] = True
    if overwrite_existing:
        shutil.rmtree(_correction_scratch(paths), ignore_errors=True)
    _detach_linked_backend_outputs(paths)
    _copy_input_sidecars(raw, paths)
    manifest["status"] = "running"
    _write_json(paths["manifest"], manifest)
    _write_json(
        paths["qc"],
        {
            "kind": "dwi_preprocessing_qc",
            "status": "running",
            "subject_id": raw.subject_id,
            "backend_available": backend["available"],
        },
    )

    try:
        backend_details = _execute_preprocessing_backend(
            raw,
            config,
            paths,
            backend,
            commands,
            log=log,
            acquisition=acquisition,
            native_commands=manifest.get("native_commands"),
            force=overwrite_existing,
        )

        if not native:
            if not paths["final_bval"].exists():
                shutil.copy2(raw.bval_path, paths["final_bval"])
            if not paths["final_bvec"].exists():
                shutil.copy2(raw.bvec_path, paths["final_bvec"])
        mask_generation = _finalize_brain_mask(raw, config, paths, backend_details, log=log)

        required_outputs = (
            paths["final_dwi"],
            paths["final_bval"],
            paths["final_bvec"],
            paths["final_mask"],
        )
        missing_outputs = [path.as_posix() for path in required_outputs if not path.is_file()]
        if missing_outputs:
            raise RuntimeError(
                "DWI preprocessing finished without required output(s): " + ", ".join(missing_outputs)
            )
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(paths["manifest"], manifest)
        _write_json(
            paths["qc"],
            {
                "kind": "dwi_preprocessing_qc",
                "version": 1,
                "subject_id": raw.subject_id,
                "status": "failed",
                "backend_available": backend["available"],
                "brain_mask_backend": mask_backend,
                "error": manifest["error"],
            },
        )
        raise

    qc_payload = {
        "kind": "dwi_preprocessing_qc",
        "version": 1,
        "subject_id": raw.subject_id,
        "status": "completed",
        "files": {
            "final_dwi": _file_record(paths["final_dwi"]),
            "final_bval": _file_record(paths["final_bval"]),
            "final_bvec": _file_record(paths["final_bvec"]),
            "final_mask": _file_record(paths["final_mask"]),
            "eddy_parameters": _file_record(paths["eddy_parameters"]),
            "eddy_movement_rms": _file_record(paths["eddy_movement_rms"]),
            "eddy_outlier_map": _file_record(paths["eddy_outlier_map"]),
            "eddy_outlier_report": _file_record(paths["eddy_outlier_report"]),
            "eddy_cnr_maps": _file_record(paths["eddy_cnr_maps"]),
            "eddy_residuals": _file_record(paths["eddy_residuals"]),
        },
        "warnings": _preprocessing_warnings(backend, backend_details) + mask_generation.get("warnings", []),
        "acquisition": acquisition.to_record(),
        "brain_mask_backend": mask_backend,
        "brain_mask_qc": mask_generation["qc"],
        "mask_generation": mask_generation,
        "backend_details": backend_details,
    }
    manifest["status"] = "completed"
    manifest["mask_generation"] = mask_generation
    manifest["parameters"]["final_mask_method"] = mask_generation["method"]
    _write_json(paths["manifest"], manifest)
    _write_json(paths["qc"], qc_payload)

    return DwiPreprocessingResult(raw.subject_id, "completed", output_dir, paths["manifest"], paths["qc"], paths["final_dwi"], paths["final_bval"], paths["final_bvec"], paths["final_mask"], acquisition)
