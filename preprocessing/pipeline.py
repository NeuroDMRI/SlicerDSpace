"""Optional DWI preprocessing wrapper around DIPY or MRtrix3/FSL commands.

This stage is intentionally optional. Product pipelines may consume raw
``input/data.nii.gz`` directly, or users may run this stage first to generate a
shared corrected DWI under ``outputs/01_preproc``.
"""

from __future__ import annotations

from shared.cohort_schedule import scheduled_stage

import json
import os
import shutil
import subprocess
import importlib.util
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional

from shared.io_layout import default_task_output_dir
from shared.external_process import fsl_root_from_bin_dir, managed_tool_environment, text_subprocess_kwargs
from shared.wsl_bridge import preprocessing_launcher
from shared.brain_mask import (
    BrainMaskConfig,
    brain_mask_config_from_options,
    generate_brain_mask,
    inspect_brain_mask_backend,
    validate_brain_mask,
)
from shared.pipeline import RawInputBundle, find_subject_root


REQUIRED_COMMANDS = ("dwidenoise", "mrdegibbs", "dwifslpreproc", "dwi2mask")
REVERSE_PHASE_COMMANDS = ("mrconvert", "dwiextract", "mrcat")
DIAGNOSTIC_COMMANDS = ("eddy", "topup")
DIPY_BVEC_ROTATION_POLICY = "inverse_sampling_affine_polar_v1"


def preprocessing_gradients_current(manifest: dict) -> bool:
    """Reject motion-corrected DIPY products using the superseded rotation rule."""
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
    return max(1, min(8, int(count or 1)))


@dataclass(frozen=True)
class DwiPreprocessingConfig:
    subject_id: str
    input_root: Path
    output_dir: Optional[Path] = None
    backend: str = "dipy"
    execute: bool = False
    skip: bool = False
    force: bool = False
    dipy_denoise_method: str = "mppca"
    dipy_run_gibbs: bool = True
    dipy_run_motion_correction: bool = True
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
    synthstrip_python: Optional[Path] = None
    postprocess_freesurfer_home: Optional[Path] = None


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
    )


def _resolve_output_dir(raw: RawInputBundle, output_dir: Optional[Path]) -> Path:
    if output_dir is not None:
        return output_dir.expanduser().resolve()
    return default_task_output_dir(raw.subject_root, "preproc")


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
    if config.eddy_threads is None:
        return _default_eddy_threads()
    if config.eddy_threads < 1:
        raise ValueError("eddy_threads must be >= 1.")
    return int(config.eddy_threads)


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
    backend_name = config.backend.lower()
    if backend_name == "dipy":
        python_modules = ("nibabel", "numpy", "dipy")
        python_status = {
            module: importlib.util.find_spec(module) is not None
            for module in python_modules
        }
        missing_python = [module for module, available in python_status.items() if not available]
        return {
            "kind": "dwi_preprocessing_backend",
            "backend": "dipy",
            "python": python_status,
            "missing_python": missing_python,
            "available": not missing_python,
            "capabilities": {
                "denoise": config.dipy_denoise_method,
                "gibbs_removal": bool(config.dipy_run_gibbs),
                "between_volume_motion_correction": bool(config.dipy_run_motion_correction),
                "susceptibility_distortion_correction": False,
                "eddy_current_modeling": False,
            },
        }
    if backend_name not in {"mrtrix_fsl", "mrtrix", "fsl"}:
        return {
            "kind": "dwi_preprocessing_backend",
            "backend": backend_name,
            "available": False,
            "missing_required": [f"unsupported backend: {config.backend}"],
        }
    required_commands = list(REQUIRED_COMMANDS)
    if config.reverse_phase_b0 is not None:
        required_commands.extend(REVERSE_PHASE_COMMANDS)
    commands = {
        command: _resolve_command(command, config)
        for command in (*required_commands, *DIAGNOSTIC_COMMANDS)
    }
    missing_required = [command for command in required_commands if commands[command] is None]
    missing_diagnostic = [command for command in DIAGNOSTIC_COMMANDS if commands[command] is None]
    return {
        "kind": "dwi_preprocessing_backend",
        "backend": "mrtrix_fsl",
        "required_commands": required_commands,
        "diagnostic_commands": list(DIAGNOSTIC_COMMANDS),
        "commands": commands,
        "missing_required": missing_required,
        "missing_diagnostic": missing_diagnostic,
        "available": not missing_required,
        "capabilities": {
            "denoise": "mppca",
            "gibbs_removal": True,
            "between_volume_motion_correction": True,
            "susceptibility_distortion_correction": config.reverse_phase_b0 is not None,
            "eddy_current_modeling": True,
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


def _file_record(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": path.as_posix(), "exists": False}
    stat = path.stat()
    return {
        "path": path.resolve().as_posix(),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _preprocessing_cache_key(
    raw: RawInputBundle,
    config: DwiPreprocessingConfig,
    backend: dict[str, Any],
    mask_backend: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Describe every input and setting that can change preprocessing output."""

    reverse_phase = config.reverse_phase_b0.expanduser().resolve() if config.reverse_phase_b0 else None
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
        "version": 5,
        "backend": str(backend.get("backend") or config.backend).lower(),
        "inputs": {
            "dwi": _file_record(raw.dwi_path),
            "bval": _file_record(raw.bval_path),
            "bvec": _file_record(raw.bvec_path),
            "mask": _file_record(raw.mask_path),
            "reverse_phase_b0": _file_record(reverse_phase) if reverse_phase else None,
        },
        "parameters": {
            "dipy_denoise_method": config.dipy_denoise_method.lower(),
            "dipy_run_gibbs": bool(config.dipy_run_gibbs),
            "dipy_run_motion_correction": bool(config.dipy_run_motion_correction),
            **({"dipy_bvec_rotation_policy": DIPY_BVEC_ROTATION_POLICY}
               if str(backend.get("backend") or config.backend).lower() == "dipy" and config.dipy_run_motion_correction else {}),
            "b0_threshold": float(config.b0_threshold),
            "phase_encoding_direction": config.phase_encoding_direction,
            "readout_time": config.readout_time,
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
        "eddy_cuda_visibility": os.environ.get("CUDA_VISIBLE_DEVICES")
        if backend.get("backend") == "mrtrix_fsl" else None,
        "eddy_device_policy": _file_record(Path(__file__).resolve().parents[1] / "shared/mrtrix_preproc_runner.py")
        if backend.get("backend") == "mrtrix_fsl" else None,
        "brain_mask": _brain_mask_cache_key(
            config, mask_backend if mask_backend is not None else inspect_brain_mask_backend(_brain_mask_config(config)),
        ),
    }


def _brain_mask_cache_key(
    config: DwiPreprocessingConfig, inspection: dict[str, Any],
) -> dict[str, Any]:
    selected = _brain_mask_config(config)
    options = {
        name: (value.expanduser().resolve().as_posix() if isinstance(value, Path) else value)
        for name, value in vars(selected).items()
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


def _run_command(cmd: list[str], *, env: dict[str, str], log: Optional[Callable[[str], None]]) -> None:
    if log:
        log("[Preproc] Running: " + " ".join(cmd))
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        **text_subprocess_kwargs(bufsize=1),
    )
    if process.stdout:
        for line in process.stdout:
            if log:
                log("[Preproc] " + line.rstrip())
        process.stdout.close()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Preprocessing command failed with exit code {return_code}: {' '.join(cmd)}")


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


def _collect_dwifslpreproc_diagnostics(paths: dict[str, Path], subject: str) -> dict[str, dict[str, Any]]:
    scratch_dir = paths["motion_dir"] / "dwifslpreproc_tmp"
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
        records[name] = _file_record(destination)
    return records


def _build_commands(raw: RawInputBundle, config: DwiPreprocessingConfig, paths: dict[str, Path], backend: dict[str, Any]) -> list[list[str]]:
    if backend.get("backend") == "dipy":
        return [
            ["dipy", "denoise", config.dipy_denoise_method, raw.dwi_path.as_posix(), (paths["denoise_dir"] / f"{raw.subject_id}_dwi_denoised.nii.gz").as_posix()],
            ["dipy", "gibbs_removal", (paths["denoise_dir"] / f"{raw.subject_id}_dwi_degibbs.nii.gz").as_posix()]
            if config.dipy_run_gibbs
            else ["dipy", "skip_gibbs_removal"],
            ["dipy", "motion_correction", (paths["motion_dir"] / f"{raw.subject_id}_dwi_preproc.nii.gz").as_posix()]
            if config.dipy_run_motion_correction
            else ["dipy", "skip_motion_correction"],
        ]

    subject = raw.subject_id
    denoised = paths["denoise_dir"] / f"{subject}_dwi_denoised.nii.gz"
    noise = paths["denoise_dir"] / f"{subject}_dwi_noise.nii.gz"
    degibbs = paths["denoise_dir"] / f"{subject}_dwi_degibbs.nii.gz"
    reverse_phase_commands: list[list[str]] = []
    reverse_phase_b0 = (
        config.reverse_phase_b0.expanduser().resolve()
        if config.reverse_phase_b0 is not None
        else None
    )
    if reverse_phase_b0 is not None:
        if not reverse_phase_b0.is_file():
            raise FileNotFoundError(f"Reverse phase-encoded b=0 image not found: {reverse_phase_b0}")
        import nibabel as nib

        try:
            reverse_shape = nib.load(str(reverse_phase_b0)).shape
        except Exception as exc:
            raise ValueError(f"Unable to read reverse phase-encoded b=0 image: {reverse_phase_b0}") from exc
        reverse_ndim = len(reverse_shape)
        if reverse_ndim not in {3, 4}:
            raise ValueError(
                "reverse_phase_b0 must be a 3D image or a single-volume 4D image; "
                f"received shape {reverse_shape}."
            )
        reverse_volume_count = int(reverse_shape[3]) if reverse_ndim == 4 else 1
        if reverse_volume_count != 1:
            raise ValueError(
                "reverse_phase_b0 must contain exactly one reverse phase-encoded b=0 volume; "
                f"received shape {reverse_shape}."
            )
        if reverse_ndim == 3:
            normalized_reverse = paths["motion_dir"] / f"{subject}_reverse_phase_b0_4d.nii.gz"
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
                if "-force" not in command:
                    command.append("-force")
        reverse_phase_b0 = se_epi_pair

    dwifslpreproc_cmd = [
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
        dwifslpreproc_cmd.append("-force")
    if reverse_phase_b0 is not None:
        dwifslpreproc_cmd.extend(["-rpe_pair", "-se_epi", reverse_phase_b0.as_posix()])
    else:
        dwifslpreproc_cmd.append("-rpe_none")
    if config.phase_encoding_direction:
        dwifslpreproc_cmd.extend(["-pe_dir", config.phase_encoding_direction])
    if config.readout_time is not None:
        dwifslpreproc_cmd.extend(["-readout_time", str(config.readout_time)])

    dwidenoise_cmd = [
        backend["commands"]["dwidenoise"] or "dwidenoise",
        raw.dwi_path.as_posix(),
        denoised.as_posix(),
        "-noise",
        noise.as_posix(),
    ]
    mrdegibbs_cmd = [
        backend["commands"]["mrdegibbs"] or "mrdegibbs",
        denoised.as_posix(),
        degibbs.as_posix(),
    ]
    if config.force:
        dwidenoise_cmd.append("-force")
        mrdegibbs_cmd.append("-force")

    return [
        dwidenoise_cmd,
        mrdegibbs_cmd,
        *reverse_phase_commands,
        dwifslpreproc_cmd,
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

    scratch_dir = paths["motion_dir"] / "dwifslpreproc_tmp"
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


def _load_bvecs(path: Path) -> list[list[float]]:
    rows = [[float(token) for token in line.split()] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) == 3:
        return [list(vector) for vector in zip(*rows)]
    return rows


def _write_bvecs(path: Path, vectors: list[list[float]]) -> None:
    if not vectors:
        path.write_text("", encoding="utf-8")
        return
    rows = list(zip(*vectors))
    text = "\n".join(" ".join(f"{float(value):.10g}" for value in row) for row in rows) + "\n"
    path.write_text(text, encoding="utf-8")


def _rotate_bvecs_best_effort(raw_bvec_path: Path, output_bvec_path: Path, affines: Any) -> dict[str, Any]:
    """Reorient gradients using DIPY's reference-to-moving sampling transforms."""
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Gradient reorientation requires NumPy") from exc

    bvecs = _load_bvecs(raw_bvec_path)
    affine_array = np.asarray(affines)
    input_affine_shape = list(affine_array.shape)
    affine_layout: Optional[str] = None
    if (
        affine_array.ndim == 3
        and affine_array.shape[:2] == (4, 4)
        and affine_array.shape[-1] == len(bvecs)
    ):
        # DIPY 1.12 returns one affine per volume with shape (4, 4, N).
        # Move the volume axis first so the rotation loop below also remains
        # compatible with the historical/list-derived (N, 4, 4) layout.
        affine_array = np.moveaxis(affine_array, -1, 0)
        affine_layout = "4x4xN"
    elif (
        affine_array.ndim == 3
        and affine_array.shape[-2:] == (4, 4)
        and affine_array.shape[0] == len(bvecs)
    ):
        affine_layout = "Nx4x4"

    if affine_layout is None:
        raise ValueError(f"Motion affine shape {input_affine_shape} does not match {len(bvecs)} gradients")
    if not np.isfinite(affine_array).all() or not np.isfinite(bvecs).all():
        raise ValueError("Motion affines and gradients must be finite for reorientation")

    rotated: list[list[float]] = []
    for vector, affine in zip(bvecs, affine_array):
        vec = np.asarray(vector, dtype=float)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-8:
            rotated.append([0.0, 0.0, 0.0])
            continue
        linear = affine[:3, :3]
        if np.linalg.matrix_rank(linear) < 3:
            raise ValueError("Singular motion affine cannot define a gradient rotation")
        try:
            u, _, vt = np.linalg.svd(linear)
            rotation = u @ vt
        except np.linalg.LinAlgError as exc:
            raise ValueError("Motion affine polar decomposition failed") from exc
        # DIPY affines map reference positions to moving samples; gradients use R^-1.
        new_vec = rotation.T @ vec
        new_norm = float(np.linalg.norm(new_vec))
        if new_norm > 1e-8:
            new_vec = new_vec / new_norm
        rotated.append([float(new_vec[0]), float(new_vec[1]), float(new_vec[2])])
    _write_bvecs(output_bvec_path, rotated)
    return {
        "status": "rotated",
        "method": "inverse_polar_rotation_of_dipy_sampling_affines",
        "gradient_count": len(rotated),
        "affine_layout": affine_layout,
        "affine_shape": input_affine_shape,
    }


def _run_dipy_backend(
    raw: RawInputBundle,
    config: DwiPreprocessingConfig,
    paths: dict[str, Path],
    *,
    log: Optional[Callable[[str], None]],
) -> dict[str, Any]:
    try:
        import nibabel as nib
        import numpy as np
        from dipy.align import motion_correction
        from dipy.core.gradients import gradient_table
        from dipy.denoise.gibbs import gibbs_removal
        from dipy.denoise.localpca import localpca, mppca
        try:
            from dipy.denoise.patch2self import patch2self
        except Exception:  # pragma: no cover - version-dependent optional path
            patch2self = None
    except Exception as exc:
        raise RuntimeError("DIPY preprocessing backend requires nibabel, numpy, and dipy.") from exc

    def emit(message: str) -> None:
        if log:
            log("[Preproc:DIPY] " + message)

    subject = raw.subject_id
    denoise_method = config.dipy_denoise_method.lower()
    if denoise_method in {"none", "skip", ""} and not config.dipy_run_gibbs and not config.dipy_run_motion_correction:
        emit("Fast path: linking raw DWI/gradients directly to final preprocessing outputs")
        denoised_path = paths["denoise_dir"] / f"{subject}_dwi_denoised.nii.gz"
        degibbs_path = paths["denoise_dir"] / f"{subject}_dwi_degibbs.nii.gz"
        corrected_path = paths["motion_dir"] / f"{subject}_dwi_preproc.nii.gz"
        _link_or_copy(raw.dwi_path, denoised_path)
        _link_or_copy(denoised_path, degibbs_path)
        _link_or_copy(degibbs_path, corrected_path)
        _link_or_copy(corrected_path, paths["final_dwi"])
        _link_or_copy(raw.bval_path, paths["final_bval"])
        _link_or_copy(raw.bvec_path, paths["final_bvec"])
        return {
            "denoise": {"method": denoise_method, "status": "skipped"},
            "gibbs": {"status": "skipped"},
            "motion": {"status": "skipped", "bvec_rotation": {"status": "copied"}},
            "bias_correction": {
                "status": "not_run",
                "reason": "fast path requested with no denoise/gibbs/motion operations",
            },
            "susceptibility_distortion_correction": {
                "status": "not_supported_by_dipy_backend",
                "recommended_backend": "mrtrix_fsl",
            },
            "files": {
                "denoised": denoised_path.as_posix(),
                "degibbs": degibbs_path.as_posix(),
                "motion_corrected": corrected_path.as_posix(),
            },
            "fast_path": True,
        }

    emit(f"Loading source DWI from {raw.dwi_path}")
    source_img = nib.load(str(raw.dwi_path))
    emit("Materializing DWI array into memory")
    data = source_img.get_fdata(dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"DIPY preprocessing expects a 4D DWI image, got shape {data.shape}")
    emit(f"Loaded DWI shape={data.shape}")

    emit("Loading gradients")
    bvals = np.loadtxt(str(raw.bval_path), ndmin=1)
    bvecs = np.loadtxt(str(raw.bvec_path))
    if bvecs.ndim == 2 and bvecs.shape[0] == 3:
        bvecs = bvecs.T
    gtab = gradient_table(bvals, bvecs=bvecs, b0_threshold=config.b0_threshold)
    emit(f"Gradient table prepared with {int(len(bvals))} volumes")

    denoised = data
    noise_path = paths["denoise_dir"] / f"{subject}_dwi_noise.nii.gz"
    denoise_status: dict[str, Any] = {"method": denoise_method}
    if denoise_method in {"none", "skip", ""}:
        emit("Skipping denoise")
        denoise_status["status"] = "skipped"
    elif denoise_method == "mppca":
        emit("Running mppca denoise")
        denoised, sigma = mppca(data, patch_radius=2, return_sigma=True)
        nib.save(nib.Nifti1Image(np.asarray(sigma, dtype=np.float32), source_img.affine, source_img.header), str(noise_path))
        denoise_status["status"] = "completed"
        denoise_status["noise_map"] = noise_path.as_posix()
    elif denoise_method == "localpca":
        emit("Running localpca denoise")
        denoised = localpca(data, sigma=None, patch_radius=2)
        denoise_status["status"] = "completed"
    elif denoise_method == "patch2self":
        if patch2self is None:
            raise RuntimeError("DIPY patch2self is not available in this environment.")
        emit("Running patch2self denoise")
        denoised = patch2self(data, bvals, b0_threshold=config.b0_threshold, verbose=False)
        denoise_status["status"] = "completed"
    else:
        raise ValueError(f"Unsupported DIPY denoise method: {config.dipy_denoise_method}")

    denoised_path = paths["denoise_dir"] / f"{subject}_dwi_denoised.nii.gz"
    nib.save(nib.Nifti1Image(np.asarray(denoised, dtype=np.float32), source_img.affine, source_img.header), str(denoised_path))

    if config.dipy_run_gibbs:
        emit("Running Gibbs removal")
        degibbs = gibbs_removal(np.asarray(denoised, dtype=np.float32), slice_axis=2, inplace=False, num_processes=1)
        gibbs_status = {"status": "completed"}
    else:
        emit("Skipping Gibbs removal")
        degibbs = np.asarray(denoised, dtype=np.float32)
        gibbs_status = {"status": "skipped"}
    degibbs_path = paths["denoise_dir"] / f"{subject}_dwi_degibbs.nii.gz"
    nib.save(nib.Nifti1Image(np.asarray(degibbs, dtype=np.float32), source_img.affine, source_img.header), str(degibbs_path))

    if config.dipy_run_motion_correction:
        emit("Running between-volume motion correction")
        corrected_img, affine_array = motion_correction(
            nib.Nifti1Image(np.asarray(degibbs, dtype=np.float32), source_img.affine, source_img.header),
            gtab,
            b0_ref=0,
        )
        corrected_path = paths["motion_dir"] / f"{subject}_dwi_preproc.nii.gz"
        nib.save(corrected_img, str(corrected_path))
        affine_path = paths["motion_dir"] / f"{subject}_dipy_motion_affines.npy"
        np.save(str(affine_path), np.asarray(affine_array))
        rotation_status = _rotate_bvecs_best_effort(raw.bvec_path, paths["final_bvec"], affine_array)
        motion_status = {
            "status": "completed",
            "affines": affine_path.as_posix(),
            "bvec_rotation": rotation_status,
        }
    else:
        emit("Skipping motion correction")
        corrected_path = paths["motion_dir"] / f"{subject}_dwi_preproc.nii.gz"
        shutil.copy2(degibbs_path, corrected_path)
        shutil.copy2(raw.bvec_path, paths["final_bvec"])
        motion_status = {"status": "skipped", "bvec_rotation": {"status": "copied"}}

    shutil.copy2(corrected_path, paths["final_dwi"])
    shutil.copy2(raw.bval_path, paths["final_bval"])

    return {
        "denoise": denoise_status,
        "gibbs": gibbs_status,
        "motion": motion_status,
        "bias_correction": {
            "status": "not_available",
            "reason": "DIPY bias_correction module is not available in the tested 1.11.0 environment.",
        },
        "susceptibility_distortion_correction": {
            "status": "not_supported_by_dipy_backend",
            "recommended_backend": "mrtrix_fsl",
        },
        "files": {
            "denoised": denoised_path.as_posix(),
            "degibbs": degibbs_path.as_posix(),
            "motion_corrected": corrected_path.as_posix(),
        },
    }


def _mrtrix_backend_details(
    paths: dict[str, Path],
    subject_id: str,
    config: DwiPreprocessingConfig,
) -> dict[str, Any]:
    env = _external_command_environment(config)
    return {
        "fsl_environment": {
            "FSLDIR": env.get("FSLDIR"),
            "FSLOUTPUTTYPE": env.get("FSLOUTPUTTYPE"),
        },
        "denoise": {"status": "completed", "method": "dwidenoise_mppca"},
        "gibbs": {"status": "completed", "method": "mrdegibbs"},
        "motion": {"status": "completed", "method": "fsl_eddy",
                   "device_execution": json.loads((paths["qc_dir"] / "eddy_device.json").read_text())
                   if (paths["qc_dir"] / "eddy_device.json").is_file() else None},
        "bias_correction": {
            "status": "not_run",
            "reason": "Bias-field correction is not included in the MRtrix/FSL command sequence.",
        },
        "susceptibility_distortion_correction": {
            "status": "completed" if config.reverse_phase_b0 is not None else "not_run",
            "method": "fsl_topup" if config.reverse_phase_b0 is not None else "rpe_none",
        },
        "diagnostics": _collect_dwifslpreproc_diagnostics(paths, subject_id),
        "eddy_qc_collection": _collect_eddy_qc_artifacts(paths),
        "scratch_dir": (paths["motion_dir"] / "dwifslpreproc_tmp").as_posix(),
    }


def _execute_preprocessing_backend(
    raw: RawInputBundle,
    config: DwiPreprocessingConfig,
    paths: dict[str, Path],
    backend: dict[str, Any],
    commands: list[list[str]],
    *,
    log: Optional[Callable[[str], None]],
) -> dict[str, Any]:
    if backend.get("backend") == "dipy":
        return _run_dipy_backend(raw, config, paths, log=log)

    env = _external_command_environment(config)
    for command in commands:
        if command[0] == "copy":
            shutil.copy2(command[1], command[2])
            continue
        if command[0] == "optional-output":
            continue
        runtime_command = preprocessing_launcher(command, paths["qc_dir"] / "eddy_device.json", env)
        _run_command(runtime_command, env=env, log=log)
    return _mrtrix_backend_details(paths, raw.subject_id, config)


def _preprocessing_warnings(
    backend: dict[str, Any],
    backend_details: dict[str, Any],
) -> list[str]:
    warnings = [
        f"Diagnostic command not found: {command}"
        for command in backend.get("missing_diagnostic", [])
    ]
    capabilities = backend.get("capabilities") or {}
    if "denoise" in capabilities and capabilities["denoise"] in {False, None, "none", "skip", ""}:
        warnings.append("Denoising was not applied by the selected preprocessing configuration.")
    for capability, message in (
        ("gibbs_removal", "Gibbs-ringing removal was not applied."),
        ("between_volume_motion_correction", "Between-volume motion correction was not applied."),
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
    preserve_input = selected.backend == "provided" or (
        backend_details.get("fast_path") is True and not motion_applied and raw.mask_path.is_file()
    )
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
            "method": "dwi2mask_corrected_dwi" if selected.backend == "mrtrix" else f"{selected.backend}_corrected_dwi",
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
    output_dir = _resolve_output_dir(raw, config.output_dir)
    paths = _planned_paths(output_dir, raw.subject_id)
    backend = inspect_preprocessing_backend(config)
    mask_backend = inspect_brain_mask_backend(_brain_mask_config(config))
    commands = _build_commands(raw, config, paths, backend)
    command_records = [{"argv": command} for command in commands]
    try:
        previous_manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    except (OSError, ValueError):
        previous_manifest = {}
    cache_key = _preprocessing_cache_key(raw, config, backend, mask_backend)

    output_dir.mkdir(parents=True, exist_ok=True)
    status = "skipped" if config.skip else ("planned" if not config.execute else "completed")

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
        "inputs": {
            "subject_root": raw.subject_root.as_posix(),
            "dwi": raw.dwi_path.as_posix(),
            "bval": raw.bval_path.as_posix(),
            "bvec": raw.bvec_path.as_posix(),
            "mask": raw.mask_path.as_posix(),
            "reverse_phase_b0": config.reverse_phase_b0.as_posix() if config.reverse_phase_b0 else None,
        },
        "parameters": {
            "eddy_threads": _resolved_eddy_threads(config)
            if backend.get("backend") == "mrtrix_fsl"
            else None,
            "eddy_second_level_model": (
                config.eddy_second_level_model.strip().lower()
                if backend.get("backend") == "mrtrix_fsl"
                and config.eddy_second_level_model is not None
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
        missing = ", ".join(backend.get("missing_required", backend.get("missing_python", [])))
        raise RuntimeError(f"DWI preprocessing backend is unavailable ({config.backend}): {missing}")

    cached_core_ready = (
        not config.force
        and previous_manifest.get("status") in {"completed", "cached"}
        and previous_manifest.get("cache_key") == cache_key
        and paths["final_dwi"].is_file()
        and paths["final_bval"].is_file()
        and paths["final_bvec"].is_file()
    )
    mask_refresh_status = "not_needed"
    if cached_core_ready:
        try:
            previous_qc = json.loads(paths["qc"].read_text(encoding="utf-8")) if paths["qc"].is_file() else {}
        except (OSError, ValueError):
            previous_qc = {}
        previous_details = previous_qc.get("backend_details") if isinstance(previous_qc, dict) else None
        backend_details: dict[str, Any] = (
            dict(previous_details) if isinstance(previous_details, dict) else {}
        )
        if backend.get("backend") == "mrtrix_fsl":
            backend_details = _mrtrix_backend_details(paths, raw.subject_id, config)
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
                "brain_mask_backend": mask_backend,
                "brain_mask_qc": mask_generation["qc"],
                "mask_generation": mask_generation,
            },
        )
        return DwiPreprocessingResult(raw.subject_id, "cached", output_dir, paths["manifest"], paths["qc"], paths["final_dwi"], paths["final_bval"], paths["final_bvec"], paths["final_mask"])

    _materialize_dirs(paths)
    overwrite_existing = config.force or bool(previous_manifest)
    if overwrite_existing and backend.get("backend") == "mrtrix_fsl":
        # A cache miss may leave old intermediates/final files. Overwrite them
        # instead of failing at dwidenoise, and never resume an old eddy scratch.
        commands = _build_commands(raw, replace(config, force=True), paths, backend)
        manifest["commands"] = [{"argv": command} for command in commands]
        manifest["overwrite_existing"] = True
    if overwrite_existing:
        shutil.rmtree(paths["motion_dir"] / "dwifslpreproc_tmp", ignore_errors=True)
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
        )

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

    return DwiPreprocessingResult(raw.subject_id, "completed", output_dir, paths["manifest"], paths["qc"], paths["final_dwi"], paths["final_bval"], paths["final_bvec"], paths["final_mask"])
