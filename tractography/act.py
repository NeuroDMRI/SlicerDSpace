"""ACT tractography planning and dependency checks.

This module treats ACT as an optional MRtrix3-backed workflow. Standard ACT
needs a 5TT image and a GMWMI seed image; DDParcel is recorded as the required
node/parcellation dependency and can be used later to derive fallback tissue
models, but it is not treated as a complete ACT tissue model by itself.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from shared.external_process import managed_tool_environment


SUPPORTED_ACT_OUTPUT_FORMATS = {"tck"}
REQUIRED_MRTRIX_COMMANDS = ("mrconvert", "dwi2response", "dwi2fod", "tckgen", "tcksift2", "tck2connectome")


@dataclass(frozen=True)
class ACTTrackingConfig:
    subject_id: str
    parcellation_path: Path
    output_dir: Path
    tractography_path: Optional[Path] = None
    dwi_path: Optional[Path] = None
    bval_path: Optional[Path] = None
    bvec_path: Optional[Path] = None
    mask_path: Optional[Path] = None
    five_tt_path: Optional[Path] = None
    gmwmi_path: Optional[Path] = None
    wm_fod_path: Optional[Path] = None
    backend: str = "mrtrix3"
    algorithm: str = "iFOD2"
    seed_count: int = 1_000_000
    output_format: str = "tck"
    sift2: bool = True
    execute: bool = False


@dataclass(frozen=True)
class ACTTrackingResult:
    subject_id: str
    output_dir: Path
    streamlines_path: Path
    weights_path: Optional[Path]
    manifest_path: Path
    qc_path: Path
    commands: tuple[tuple[str, ...], ...]
    backend_status: dict[str, object]
    standard_act_ready: bool
    executed: bool
    return_code: Optional[int] = None


def _emit(message: str, log: Optional[Callable[[str], None]] = None) -> None:
    if log:
        log(message)


def _file_info(path: Optional[Path]) -> dict[str, object]:
    if path is None:
        return {"path": None, "exists": False}
    expanded = path.expanduser()
    if not expanded.exists():
        return {"path": expanded.as_posix(), "exists": False}
    stat = expanded.stat()
    return {
        "path": expanded.resolve().as_posix(),
        "exists": True,
        "is_file": expanded.is_file(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def inspect_mrtrix_backend() -> dict[str, object]:
    """Return MRtrix3 command availability without assuming install location."""

    commands: dict[str, object] = {}
    missing: list[str] = []
    for command in REQUIRED_MRTRIX_COMMANDS:
        resolved = shutil.which(command)
        commands[command] = resolved
        if resolved is None:
            missing.append(command)
    return {
        "name": "mrtrix3",
        "available": not missing,
        "commands": commands,
        "missing_commands": missing,
    }


def _command(name: str) -> str:
    return shutil.which(name) or name


def build_mrtrix_act_commands(config: ACTTrackingConfig, streamlines_path: Path, weights_path: Optional[Path]) -> tuple[tuple[str, ...], ...]:
    """Build an MRtrix3 ACT command sequence without assuming install paths."""

    if not config.dwi_path:
        raise ValueError("dwi_path is required to build ACT commands.")
    if not config.mask_path:
        raise ValueError("mask_path is required to build ACT commands.")
    if not config.five_tt_path:
        raise ValueError("five_tt_path is required to build ACT commands.")
    if not config.gmwmi_path:
        raise ValueError("gmwmi_path is required to build ACT commands.")
    dwi_mif = config.output_dir.expanduser().resolve() / f"{config.subject_id}-dwi.mif"
    wm_response = config.output_dir.expanduser().resolve() / f"{config.subject_id}-wm_response.txt"
    gm_response = config.output_dir.expanduser().resolve() / f"{config.subject_id}-gm_response.txt"
    csf_response = config.output_dir.expanduser().resolve() / f"{config.subject_id}-csf_response.txt"
    wm_fod = config.wm_fod_path.expanduser().resolve() if config.wm_fod_path else config.output_dir.expanduser().resolve() / f"{config.subject_id}-wm_fod.mif"
    gm_fod = config.output_dir.expanduser().resolve() / f"{config.subject_id}-gm.mif"
    csf_fod = config.output_dir.expanduser().resolve() / f"{config.subject_id}-csf.mif"

    mrconvert = [_command("mrconvert"), config.dwi_path.expanduser().resolve().as_posix(), dwi_mif.as_posix()]
    if config.bvec_path and config.bval_path:
        mrconvert.extend(
            [
                "-fslgrad",
                config.bvec_path.expanduser().resolve().as_posix(),
                config.bval_path.expanduser().resolve().as_posix(),
            ]
        )

    commands: list[tuple[str, ...]] = [
        tuple(mrconvert),
        (
            _command("dwi2response"),
            "dhollander",
            dwi_mif.as_posix(),
            wm_response.as_posix(),
            gm_response.as_posix(),
            csf_response.as_posix(),
            "-mask",
            config.mask_path.expanduser().resolve().as_posix(),
        ),
        (
            _command("dwi2fod"),
            "msmt_csd",
            dwi_mif.as_posix(),
            wm_response.as_posix(),
            wm_fod.as_posix(),
            gm_response.as_posix(),
            gm_fod.as_posix(),
            csf_response.as_posix(),
            csf_fod.as_posix(),
            "-mask",
            config.mask_path.expanduser().resolve().as_posix(),
        ),
        (
            _command("tckgen"),
            wm_fod.as_posix(),
            streamlines_path.as_posix(),
            "-algorithm",
            config.algorithm,
            "-act",
            config.five_tt_path.expanduser().resolve().as_posix(),
            "-seed_gmwmi",
            config.gmwmi_path.expanduser().resolve().as_posix(),
            "-backtrack",
            "-crop_at_gmwmi",
            "-select",
            str(config.seed_count),
        ),
    ]
    if config.sift2 and weights_path is not None:
        commands.append(
            (
                _command("tcksift2"),
                streamlines_path.as_posix(),
                wm_fod.as_posix(),
                weights_path.as_posix(),
                "-act",
                config.five_tt_path.expanduser().resolve().as_posix(),
            )
        )
    return tuple(commands)


def _validate_config(config: ACTTrackingConfig) -> None:
    if not config.subject_id.strip():
        raise ValueError("subject_id is required.")
    if config.backend.lower() != "mrtrix3":
        raise ValueError("Only the optional mrtrix3 ACT backend is currently supported.")
    if config.output_format.lower() not in SUPPORTED_ACT_OUTPUT_FORMATS:
        raise ValueError(f"output_format must be one of {sorted(SUPPORTED_ACT_OUTPUT_FORMATS)}.")
    if config.seed_count <= 0:
        raise ValueError("seed_count must be > 0.")
    if not config.parcellation_path.expanduser().is_file():
        raise FileNotFoundError(f"DDParcel/parcellation dependency does not exist: {config.parcellation_path}")


def run_act_tracking(config: ACTTrackingConfig, *, log: Optional[Callable[[str], None]] = None) -> ACTTrackingResult:
    """Prepare an ACT tractography manifest and QC record.

    ``execute=False`` is intentionally useful in normal Python and Slicer import
    tests. ``execute=True`` is held until MRtrix3 command execution is wired and
    validated on a small subject.
    """

    _validate_config(config)
    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_format = config.output_format.lower()
    stem = f"{config.subject_id}-act-{config.algorithm.lower()}"
    streamlines_path = config.tractography_path.expanduser().resolve() if config.tractography_path else output_dir / f"{stem}.{output_format}"
    weights_path = output_dir / f"{stem}_sift2_weights.txt" if config.sift2 else None
    manifest_path = output_dir / f"{stem}_manifest.json"
    qc_path = output_dir / f"{stem}_qc.json"
    backend_status = inspect_mrtrix_backend()
    standard_act_ready = bool(config.five_tt_path and config.five_tt_path.expanduser().is_file() and config.gmwmi_path and config.gmwmi_path.expanduser().is_file())
    commands: tuple[tuple[str, ...], ...] = ()
    if standard_act_ready and config.dwi_path and config.mask_path:
        commands = build_mrtrix_act_commands(config, streamlines_path, weights_path)

    if config.execute:
        if not backend_status["available"]:
            missing = ", ".join(str(item) for item in backend_status["missing_commands"])
            raise RuntimeError(f"MRtrix3 backend is incomplete. Missing command(s): {missing}")
        if not standard_act_ready:
            raise RuntimeError("Standard ACT execution requires existing five_tt_path and gmwmi_path.")
        if not config.dwi_path or not config.mask_path:
            raise RuntimeError("ACT execution requires dwi_path and mask_path.")
        commands = build_mrtrix_act_commands(config, streamlines_path, weights_path)

    manifest = {
        "kind": "act_tracking",
        "version": 1,
        "subject_id": config.subject_id,
        "backend": backend_status,
        "executed": False,
        "status": "planned",
        "parameters": {
            "algorithm": config.algorithm,
            "seed_count": config.seed_count,
            "output_format": output_format,
            "sift2": config.sift2,
        },
        "dependencies": {
            "ddparcel_parcellation": _file_info(config.parcellation_path),
            "dwi": _file_info(config.dwi_path),
            "bval": _file_info(config.bval_path),
            "bvec": _file_info(config.bvec_path),
            "mask": _file_info(config.mask_path),
            "five_tt": _file_info(config.five_tt_path),
            "gmwmi": _file_info(config.gmwmi_path),
            "wm_fod": _file_info(config.wm_fod_path),
        },
        "act_readiness": {
            "standard_act_ready": standard_act_ready,
            "reason": "5TT and GMWMI are present" if standard_act_ready else "Standard ACT requires 5TT and GMWMI; DDParcel alone is not sufficient.",
            "ddparcel_role": "connectome_nodes_and_possible_tissue_model_source",
        },
        "outputs": {
            "streamlines": streamlines_path.as_posix(),
            "sift2_weights": weights_path.as_posix() if weights_path else None,
            "qc": qc_path.as_posix(),
        },
        "commands": [list(command) for command in commands],
    }
    qc = {
        "subject_id": config.subject_id,
        "status": "planned",
        "executed": False,
        "backend_available": bool(backend_status["available"]),
        "standard_act_ready": standard_act_ready,
        "streamline_count": None,
        "sift2_weight_count": None,
        "warnings": [] if standard_act_ready else ["Standard ACT is not ready without 5TT and GMWMI."],
    }
    return_code: Optional[int] = None
    execution_error: Optional[str] = None
    start_time = time.monotonic()
    if config.execute:
        manifest["status"] = "running"
        manifest["executed"] = True
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        resolved = next((Path(path) for path in backend_status["commands"].values() if path), None)
        command_env = managed_tool_environment(mrtrix_bin_dir=resolved.parent if resolved else None)
        from tractography.csd import _mrtrix_csd_invocation
        try:
            for command in commands:
                argv, env = _mrtrix_csd_invocation(command, command_env)
                _emit(f"[Tractography] Running ACT command: {' '.join(argv)}", log)
                completed = subprocess.run(argv, check=True, env=env)
                return_code = completed.returncode
        except subprocess.CalledProcessError as exc:
            return_code = exc.returncode
            execution_error = str(exc)
        finally:
            elapsed_seconds = time.monotonic() - start_time
            qc.update(
                {
                    "status": "failed" if execution_error else "completed",
                    "executed": True,
                    "return_code": return_code,
                    "elapsed_seconds": elapsed_seconds,
                    "streamlines_exists": streamlines_path.is_file(),
                    "sift2_weights_exists": weights_path.is_file() if weights_path else None,
                    "error": execution_error,
                }
            )
            manifest["status"] = qc["status"]
            manifest["return_code"] = return_code
    else:
        qc["return_code"] = None
        qc["elapsed_seconds"] = 0.0
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    qc_path.write_text(json.dumps(qc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if execution_error is not None:
        raise RuntimeError(f"ACT execution failed: {execution_error}")
    _emit(f"[Tractography] Prepared ACT scaffold: {manifest_path}", log)

    return ACTTrackingResult(
        subject_id=config.subject_id,
        output_dir=output_dir,
        streamlines_path=streamlines_path,
        weights_path=weights_path,
        manifest_path=manifest_path,
        qc_path=qc_path,
        commands=commands,
        backend_status=backend_status,
        standard_act_ready=standard_act_ready,
        executed=bool(config.execute),
        return_code=return_code,
    )
