"""MRtrix3 CSD/FOD/ACT tractography backend.

The implementation intentionally treats MRtrix3 as an external backend. It
records dependency status and the exact command chain in dry-run mode, then runs
the same commands when ``execute=True`` and MRtrix3 is available on PATH.
"""

from __future__ import annotations

from shared.cohort_schedule import scheduled_stage

import json
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from shared.external_process import external_python_environment, resolve_external_command


SEEDING_MODES = ("dynamic", "gmwmi")
CSD_ACT_MANIFEST_VERSION = 3

REQUIRED_MRTRIX_COMMANDS = (
    "mrconvert",
    "dwi2response",
    "dwi2fod",
    "mtnormalise",
    "tckgen",
    "tcksift2",
)


@dataclass(frozen=True)
class CSDActConfig:
    subject_id: str
    dwi_path: Path
    bval_path: Path
    bvec_path: Path
    mask_path: Path
    five_tt_path: Path
    gmwmi_path: Path
    output_dir: Path
    tractogram_dir: Optional[Path] = None
    algorithm: str = "iFOD2"
    select_streamlines: int = 1_000_000
    min_length_mm: float = 10.0
    max_length_mm: float = 250.0
    step_size_mm: Optional[float] = None
    cutoff: Optional[float] = None
    # Seed for MRtrix3's RNG (exported as MRTRIX_RNG_SEED). None = unseeded.
    random_seed: Optional[int] = None
    # "dynamic" (default): -seed_dynamic on the normalised WM FOD (SIFT-style
    # seeding; far less short-streamline bias than GMWMI seeding).
    # "gmwmi": -seed_gmwmi on the 5tt2gmwmi interface image.
    seeding: str = "dynamic"
    # MRtrix3 "-nthreads" for tcksift2, and for tckgen when unseeded (None =
    # MRtrix default). When random_seed is set, tckgen is always pinned to
    # "-nthreads 0" (single-threaded): that is the only configuration in which
    # a seeded tckgen run is bit-reproducible, because multi-threaded tckgen
    # consumes the RNG stream in a run-dependent order (verified 2026-09-26 on
    # HCP 100307: identical streamline data, only the header timestamp
    # differs). The manifest's "reproducibility" block records the outcome.
    nthreads: Optional[int] = None
    use_sift2: bool = True
    # DWI the anatomy (DDSeg/DDParcel -> 5TT) was derived from. When set, it
    # must be the CSD DWI unless allow_anatomy_space_mismatch is True.
    anatomy_source_dwi: Optional[Path] = None
    allow_anatomy_space_mismatch: bool = False
    act_prep_qc_path: Optional[Path] = None
    export_trk: bool = False
    export_vtk: bool = False
    mrtrix_bin_dir: Optional[Path] = None
    execute: bool = False


@dataclass(frozen=True)
class CSDActResult:
    subject_id: str
    output_dir: Path
    wm_fod_path: Path
    wm_fod_norm_path: Path
    tractogram_path: Path
    tractogram_trk_path: Optional[Path]
    tractogram_vtk_path: Optional[Path]
    sift2_weights_path: Optional[Path]
    manifest_path: Path
    qc_path: Path
    commands: tuple[tuple[str, ...], ...]
    backend_status: dict[str, object]
    executed: bool
    return_code: Optional[int] = None


def _emit(message: str, log: Optional[Callable[[str], None]] = None) -> None:
    if log:
        log(message)


@contextmanager
def _timed_step(records, name, persist, command=None):
    """Persist partial progress without changing the command or its RNG state."""
    entry = {'step': name, 'index': len(records), 'status': 'running',
             'started_at': datetime.now(timezone.utc).isoformat(),
             'command': list(command) if command is not None else None}
    records.append(entry)
    started = time.monotonic()
    persist()
    try:
        yield entry
    except BaseException as exc:
        entry.update(status='failed', error=str(exc), return_code=getattr(exc, 'returncode', None))
        raise
    else:
        if entry['status'] == 'running':
            entry['status'] = 'completed'
    finally:
        entry.update(elapsed_seconds=time.monotonic() - started,
                     ended_at=datetime.now(timezone.utc).isoformat())
        persist()


def _file_info(path: Path) -> dict[str, object]:
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


def _resolve_command(name: str, bin_dir: Optional[Path] = None) -> Optional[str]:
    return resolve_external_command(name, bin_dir)


def inspect_mrtrix_csd_backend(
    bin_dir: Optional[Path] = None,
    *,
    require_tckconvert: bool = False,
) -> dict[str, object]:
    commands: dict[str, Optional[str]] = {}
    missing: list[str] = []
    required_commands = (*REQUIRED_MRTRIX_COMMANDS, "tckconvert") if require_tckconvert else REQUIRED_MRTRIX_COMMANDS
    for command in required_commands:
        resolved = _resolve_command(command, bin_dir)
        commands[command] = resolved
        if resolved is None:
            missing.append(command)
    return {
        "name": "mrtrix3",
        "bin_dir": bin_dir.expanduser().resolve().as_posix() if bin_dir is not None else None,
        "available": not missing,
        "commands": commands,
        "missing_commands": missing,
    }


def _cmd(name: str, bin_dir: Optional[Path] = None) -> str:
    return _resolve_command(name, bin_dir) or name


def _validate_config(config: CSDActConfig) -> None:
    if not config.subject_id.strip():
        raise ValueError("subject_id is required.")
    if config.select_streamlines <= 0:
        raise ValueError("select_streamlines must be > 0.")
    if config.min_length_mm <= 0 or config.max_length_mm <= config.min_length_mm:
        raise ValueError("CSD/ACT streamline lengths require 0 < min_length_mm < max_length_mm.")
    if config.random_seed is not None and config.random_seed < 0:
        raise ValueError("random_seed must be >= 0.")
    if config.seeding not in SEEDING_MODES:
        raise ValueError(f"seeding must be one of {', '.join(SEEDING_MODES)}; got {config.seeding!r}.")
    if config.nthreads is not None and config.nthreads < 0:
        raise ValueError("nthreads must be >= 0 when set.")
    if anatomy_space_matches(config) is False and not config.allow_anatomy_space_mismatch:
        raise ValueError(
            "ACT anatomy was derived from a different DWI than the CSD input "
            f"(anatomy: {config.anatomy_source_dwi}; CSD: {config.dwi_path}). Run DDSeg/DDParcel on the same "
            "(preprocessed) DWI, or explicitly allow the mismatch."
        )
    for label, path in (
        ("dwi_path", config.dwi_path),
        ("bval_path", config.bval_path),
        ("bvec_path", config.bvec_path),
        ("mask_path", config.mask_path),
        ("five_tt_path", config.five_tt_path),
        ("gmwmi_path", config.gmwmi_path),
    ):
        if not path.expanduser().is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")


def anatomy_space_matches(config: CSDActConfig) -> Optional[bool]:
    if config.anatomy_source_dwi is None:
        return None
    return Path(config.anatomy_source_dwi).expanduser().resolve() == Path(config.dwi_path).expanduser().resolve()


def effective_nthreads(config: CSDActConfig) -> Optional[int]:
    """MRtrix -nthreads: the explicit value, else the stage CPU reservation, else None (MRtrix default)."""

    if config.nthreads is not None:
        return int(config.nthreads)
    from shared.numeric_threads import reserved_stage_cpus

    return reserved_stage_cpus()


def tckgen_thread_count(config: CSDActConfig) -> Optional[int]:
    """Effective tckgen -nthreads: 0 when seeded, else the explicit value or the stage reservation (None = MRtrix default)."""

    if config.random_seed is not None:
        return 0
    return effective_nthreads(config)


def sift2_mu_path(tractogram: Path) -> Path:
    return tractogram.with_name(f"{tractogram.stem}_sift2_mu.txt")


def reproducibility_record(config: CSDActConfig) -> dict[str, object]:
    threads = tckgen_thread_count(config)
    seeded = config.random_seed is not None
    return {
        "random_seed": config.random_seed,
        "seed_mechanism": "MRTRIX_RNG_SEED environment variable" if seeded else None,
        "nthreads_requested": config.nthreads,
        "tckgen_nthreads": threads,
        "tckgen_nthreads_pinned_for_seed": seeded,
        "pretracking_nthreads": effective_nthreads(config),
        "tcksift2_nthreads": effective_nthreads(config),
        "tckgen_bit_reproducible": bool(seeded and threads == 0),
        "note": (
            "A seeded tckgen run is bit-reproducible only single-threaded (-nthreads 0); this was verified "
            "on HCP 100307 (identical streamline data, header timestamp differs). With more threads the RNG "
            "stream is consumed in a run-dependent order and repeats are only statistically equivalent."
        ),
    }


def build_csd_act_commands(config: CSDActConfig) -> tuple[tuple[str, ...], ...]:
    output_dir = config.output_dir.expanduser().resolve()
    tractogram_dir = (
        config.tractogram_dir.expanduser().resolve()
        if config.tractogram_dir is not None
        else output_dir.parent / "tractograms" / "csd_act"
    )
    dwi_mif = output_dir / f"{config.subject_id}_dwi.mif"
    mask_mif = output_dir / f"{config.subject_id}_mask.mif"
    wm_response = output_dir / f"{config.subject_id}_response_wm.txt"
    gm_response = output_dir / f"{config.subject_id}_response_gm.txt"
    csf_response = output_dir / f"{config.subject_id}_response_csf.txt"
    wm_fod = output_dir / f"{config.subject_id}_wmfod.mif"
    gm = output_dir / f"{config.subject_id}_gm.mif"
    csf = output_dir / f"{config.subject_id}_csf.mif"
    wm_fod_norm = output_dir / f"{config.subject_id}_wmfod_norm.mif"
    gm_norm = output_dir / f"{config.subject_id}_gm_norm.mif"
    csf_norm = output_dir / f"{config.subject_id}_csf_norm.mif"
    tractogram = tractogram_dir / f"{config.subject_id}_csd_act_{config.algorithm.lower()}.tck"
    weights = tractogram.with_name(f"{tractogram.stem}_sift2_weights.txt")

    commands: list[tuple[str, ...]] = [
        (
            _cmd("mrconvert", config.mrtrix_bin_dir),
            config.dwi_path.expanduser().resolve().as_posix(),
            dwi_mif.as_posix(),
            "-fslgrad",
            config.bvec_path.expanduser().resolve().as_posix(),
            config.bval_path.expanduser().resolve().as_posix(),
            "-force",
        ),
        (
            _cmd("mrconvert", config.mrtrix_bin_dir),
            config.mask_path.expanduser().resolve().as_posix(),
            mask_mif.as_posix(),
            "-force",
        ),
        (
            _cmd("dwi2response", config.mrtrix_bin_dir),
            "dhollander",
            dwi_mif.as_posix(),
            wm_response.as_posix(),
            gm_response.as_posix(),
            csf_response.as_posix(),
            "-mask",
            mask_mif.as_posix(),
            "-force",
        ),
        (
            _cmd("dwi2fod", config.mrtrix_bin_dir),
            "msmt_csd",
            dwi_mif.as_posix(),
            wm_response.as_posix(),
            wm_fod.as_posix(),
            gm_response.as_posix(),
            gm.as_posix(),
            csf_response.as_posix(),
            csf.as_posix(),
            "-mask",
            mask_mif.as_posix(),
            "-force",
        ),
        (
            _cmd("mtnormalise", config.mrtrix_bin_dir),
            wm_fod.as_posix(),
            wm_fod_norm.as_posix(),
            gm.as_posix(),
            gm_norm.as_posix(),
            csf.as_posix(),
            csf_norm.as_posix(),
            "-mask",
            mask_mif.as_posix(),
            "-force",
        ),
    ]
    if config.seeding == "dynamic":
        seed_options = ["-seed_dynamic", wm_fod_norm.as_posix()]
    else:
        seed_options = ["-seed_gmwmi", config.gmwmi_path.expanduser().resolve().as_posix()]
    tckgen = [
        _cmd("tckgen", config.mrtrix_bin_dir),
        wm_fod_norm.as_posix(),
        tractogram.as_posix(),
        "-algorithm",
        config.algorithm,
        "-act",
        config.five_tt_path.expanduser().resolve().as_posix(),
        *seed_options,
        "-backtrack",
        "-crop_at_gmwmi",
        "-select",
        str(config.select_streamlines),
        "-minlength",
        str(config.min_length_mm),
        "-maxlength",
        str(config.max_length_mm),
        "-force",
    ]
    if config.step_size_mm is not None:
        tckgen.extend(["-step", str(config.step_size_mm)])
    if config.cutoff is not None:
        tckgen.extend(["-cutoff", str(config.cutoff)])
    tckgen_threads = tckgen_thread_count(config)
    if tckgen_threads is not None:
        tckgen.extend(["-nthreads", str(tckgen_threads)])
    commands.append(tuple(tckgen))
    if config.use_sift2:
        tcksift2 = [
            _cmd("tcksift2", config.mrtrix_bin_dir),
            tractogram.as_posix(),
            wm_fod_norm.as_posix(),
            weights.as_posix(),
            "-act",
            config.five_tt_path.expanduser().resolve().as_posix(),
            "-out_mu",
            sift2_mu_path(tractogram).as_posix(),
            "-force",
        ]
        if effective_nthreads(config) is not None:
            tcksift2.extend(["-nthreads", str(effective_nthreads(config))])
        commands.append(tuple(tcksift2))
    if config.export_vtk:
        commands.append(
            (
                _cmd("tckconvert", config.mrtrix_bin_dir),
                tractogram.as_posix(),
                tractogram.with_suffix(".vtk").as_posix(),
                "-force",
            )
        )
    if effective_nthreads(config) is not None:
        commands = [
            command if "-nthreads" in command else (*command, "-nthreads", str(effective_nthreads(config)))
            for command in commands
        ]
    return tuple(commands)


def _tractogram_summary(path: Path) -> dict[str, object]:
    """Decode every streamline and return the evidence used by CSD/ACT QC."""

    import nibabel as nib
    import numpy as np

    lengths: list[float] = []
    all_points_finite = True
    for streamline in nib.streamlines.load(str(path), lazy_load=True).streamlines:
        points = np.asarray(streamline, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 3:
            all_points_finite = False
            continue
        if not np.isfinite(points).all():
            all_points_finite = False
            continue
        lengths.append(float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()))
    values = np.asarray(lengths, dtype=np.float64)
    summary: dict[str, object] = {
        "file": _file_info(path),
        "streamline_count": int(values.size),
        "all_points_finite": bool(all_points_finite),
    }
    if values.size:
        summary["fraction_shorter_than_20mm"] = float(np.mean(values < 20.0))
        summary["length_mm"] = {
            "min": float(np.min(values)),
            "p05": float(np.percentile(values, 5)),
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }
    return summary


def _sift2_summary(path: Path, expected_count: int) -> dict[str, object]:
    import numpy as np

    values = np.asarray(np.loadtxt(path, ndmin=1), dtype=np.float64).reshape(-1)
    return {
        "file": _file_info(path),
        "weight_count": int(values.size),
        "expected_count": int(expected_count),
        "all_finite": bool(np.isfinite(values).all()),
        "all_nonnegative": bool(np.all(values >= 0.0)),
        "positive_count": int(np.count_nonzero(values > 0.0)),
        "sum": float(np.sum(values)) if values.size else 0.0,
    }


def _sift2_mu_summary(path: Path) -> dict[str, object]:
    import numpy as np

    values = np.asarray(np.loadtxt(path, ndmin=1), dtype=np.float64).reshape(-1)
    value = float(values[0]) if values.size == 1 else None
    return {
        "file": _file_info(path),
        "value": value,
        "valid": bool(value is not None and np.isfinite(value) and value > 0.0),
    }


def _vtk_streamline_summary(path: Path) -> dict[str, object]:
    import re

    match = None
    with path.open("rb") as stream:
        for line in stream:
            match = re.match(rb"(?i)^\s*LINES\s+(\d+)\s+", line)
            if match is not None:
                break
    if match is None:
        raise ValueError("VTK LINES count was not found")
    return {
        "file": _file_info(path),
        "streamline_count": int(match.group(1)),
    }


def _validate_generated_outputs(
    config: CSDActConfig,
    tractogram: Path,
    tractogram_trk: Optional[Path],
    tractogram_vtk: Optional[Path],
    weights: Optional[Path],
) -> tuple[dict[str, object], Optional[str]]:
    """Validate generated tractography before a run can report completion."""

    try:
        tck_summary = _tractogram_summary(tractogram)
        summaries: dict[str, object] = {"tck": tck_summary}
        expected_count = int(config.select_streamlines)
        count = int(tck_summary["streamline_count"])
        length_summary = dict(tck_summary.get("length_mm") or {})
        valid = (
            bool(tck_summary["all_points_finite"])
            and count == expected_count
            and bool(length_summary)
            and float(length_summary["min"]) + 1e-3 >= float(config.min_length_mm)
            and float(length_summary["max"]) - 1e-3 <= float(config.max_length_mm)
        )
        if tractogram_trk is not None:
            trk_summary = _tractogram_summary(tractogram_trk)
            summaries["trk"] = trk_summary
            valid = valid and int(trk_summary["streamline_count"]) == count and bool(
                trk_summary["all_points_finite"]
            )
        if tractogram_vtk is not None:
            vtk_summary = _vtk_streamline_summary(tractogram_vtk)
            summaries["vtk"] = vtk_summary
            valid = valid and int(vtk_summary["streamline_count"]) == count
        if weights is not None:
            weight_summary = _sift2_summary(weights, count)
            summaries["sift2"] = weight_summary
            mu_summary = _sift2_mu_summary(sift2_mu_path(tractogram))
            summaries["sift2_mu"] = mu_summary
            valid = valid and (
                int(weight_summary["weight_count"]) == count
                and bool(weight_summary["all_finite"])
                and bool(weight_summary["all_nonnegative"])
                and int(weight_summary["positive_count"]) > 0
                and bool(mu_summary["valid"])
            )
        if not valid:
            return summaries, "Generated CSD/ACT tractography failed its scientific output contract."
        return summaries, None
    except Exception as exc:
        return {}, f"Unable to validate generated CSD/ACT outputs: {type(exc).__name__}: {exc}"


def export_tck_to_trk(tck_path: Path, reference_image_path: Path, trk_path: Path) -> Path:
    """Export an MRtrix TCK tractogram as TRK for TractQuerier/WMQL users.

    MRtrix is the preferred backend for ACT, but TractQuerier reads TRK in the
    current Python environment. Keeping this conversion inside the CSD/ACT
    backend avoids hidden one-off shell steps between ACT and WMQL.
    """

    try:
        import nibabel as nib
        from nibabel.streamlines import Field
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("TCK to TRK export requires nibabel.") from exc

    tck = tck_path.expanduser().resolve()
    reference = reference_image_path.expanduser().resolve()
    trk = trk_path.expanduser().resolve()
    if not tck.is_file():
        raise FileNotFoundError(f"TCK tractogram does not exist: {tck}")
    if not reference.is_file():
        raise FileNotFoundError(f"Reference image does not exist: {reference}")
    trk.parent.mkdir(parents=True, exist_ok=True)

    reference_image = nib.load(reference.as_posix())
    tractogram = nib.streamlines.load(tck.as_posix()).tractogram
    header = nib.streamlines.trk.TrkFile.create_empty_header()
    header[Field.VOXEL_TO_RASMM] = reference_image.affine
    header[Field.VOXEL_SIZES] = reference_image.header.get_zooms()[:3]
    header[Field.DIMENSIONS] = reference_image.shape[:3]
    header[Field.VOXEL_ORDER] = "RAS"
    nib.streamlines.save(nib.streamlines.TrkFile(tractogram, header), trk.as_posix())
    return trk


@scheduled_stage('csd_act')
def run_csd_act(config: CSDActConfig, *, log: Optional[Callable[[str], None]] = None) -> CSDActResult:
    _validate_config(config)
    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tractogram_dir = (
        config.tractogram_dir.expanduser().resolve()
        if config.tractogram_dir is not None
        else output_dir.parent / "tractograms" / "csd_act"
    )
    tractogram_dir.mkdir(parents=True, exist_ok=True)

    wm_fod = output_dir / f"{config.subject_id}_wmfod.mif"
    wm_fod_norm = output_dir / f"{config.subject_id}_wmfod_norm.mif"
    tractogram = tractogram_dir / f"{config.subject_id}_csd_act_{config.algorithm.lower()}.tck"
    weights = tractogram.with_name(f"{tractogram.stem}_sift2_weights.txt") if config.use_sift2 else None
    manifest_path = output_dir / "csd_act_manifest.json"
    qc_path = output_dir / "csd_act_qc.json"
    tractogram_trk = tractogram.with_suffix(".trk") if config.export_trk else None
    tractogram_vtk = tractogram.with_suffix(".vtk") if config.export_vtk else None
    backend_status = inspect_mrtrix_csd_backend(
        config.mrtrix_bin_dir,
        require_tckconvert=config.export_vtk,
    )
    commands = build_csd_act_commands(config)

    manifest: dict[str, object] = {
        "kind": "mrtrix_csd_act",
        "version": CSD_ACT_MANIFEST_VERSION,
        "subject_id": config.subject_id,
        "executed": bool(config.execute),
        "working_directory": output_dir.as_posix(),
        "status": "running" if config.execute else "planned",
        "backend": backend_status,
        "parameters": {
            "algorithm": config.algorithm,
            "select_streamlines": config.select_streamlines,
            "min_length_mm": config.min_length_mm,
            "max_length_mm": config.max_length_mm,
            "step_size_mm": config.step_size_mm,
            "cutoff": config.cutoff,
            "random_seed": config.random_seed,
            "seeding": config.seeding,
            "seed_image": (
                wm_fod_norm.as_posix()
                if config.seeding == "dynamic"
                else config.gmwmi_path.expanduser().resolve().as_posix()
            ),
            "nthreads": config.nthreads,
            "tckgen_nthreads": tckgen_thread_count(config),
            "use_sift2": config.use_sift2,
            "export_trk": config.export_trk,
            "export_vtk": config.export_vtk,
            "tractogram_dir": tractogram_dir.as_posix(),
            "mrtrix_bin_dir": config.mrtrix_bin_dir.expanduser().resolve().as_posix()
            if config.mrtrix_bin_dir is not None
            else None,
        },
        "dependencies": {
            "dwi": _file_info(config.dwi_path),
            "bval": _file_info(config.bval_path),
            "bvec": _file_info(config.bvec_path),
            "mask": _file_info(config.mask_path),
            "five_tt": _file_info(config.five_tt_path),
            "gmwmi": _file_info(config.gmwmi_path),
        },
        "outputs": {
            "wm_fod": wm_fod.as_posix(),
            "wm_fod_norm": wm_fod_norm.as_posix(),
            "tractogram": tractogram.as_posix(),
            "tractogram_trk": tractogram_trk.as_posix() if tractogram_trk else None,
            "tractogram_vtk": tractogram_vtk.as_posix() if tractogram_vtk else None,
            "sift2_weights": weights.as_posix() if weights else None,
            "sift2_mu": sift2_mu_path(tractogram).as_posix() if weights else None,
            "qc": qc_path.as_posix(),
        },
        "commands": [list(command) for command in commands],
        "anatomy_space": {
            "anatomy_source_dwi": Path(config.anatomy_source_dwi).expanduser().resolve().as_posix()
            if config.anatomy_source_dwi is not None
            else None,
            "csd_dwi": Path(config.dwi_path).expanduser().resolve().as_posix(),
            "same_dwi": anatomy_space_matches(config),
            "mismatch_allowed": bool(config.allow_anatomy_space_mismatch),
        },
        "reproducibility": reproducibility_record(config),
        "reproducibility_note": reproducibility_record(config)["note"],
    }
    qc: dict[str, object] = {
        "kind": "mrtrix_csd_act_qc",
        "version": CSD_ACT_MANIFEST_VERSION,
        "subject_id": config.subject_id,
        "status": "planned",
        "executed": False,
        "backend_available": bool(backend_status["available"]),
        "missing_commands": backend_status["missing_commands"],
        "wm_fod_exists": wm_fod.is_file(),
        "wm_fod_norm_exists": wm_fod_norm.is_file(),
        "tractogram_exists": tractogram.is_file(),
        "tractogram_trk_exists": tractogram_trk.is_file() if tractogram_trk else None,
        "tractogram_vtk_exists": tractogram_vtk.is_file() if tractogram_vtk else None,
        "sift2_weights_exists": weights.is_file() if weights else None,
        "return_code": None,
        "elapsed_seconds": 0.0,
        "error": None,
        "scientific_output_summary": {},
    }
    if config.act_prep_qc_path is not None:
        preparation = json.loads(config.act_prep_qc_path.read_text(encoding="utf-8"))
        if preparation.get("status") != "completed":
            raise ValueError("CSD/ACT requires completed ACT preparation QC.")
        prior_record = {"qc_path": str(config.act_prep_qc_path.resolve()),
            **{key: preparation.get(key) for key in (
                "anatomical_prior_status", "sgm_from_ddparcel", "pathology_channel", "warnings", "anatomical_review")}}
        manifest["act_preparation"] = prior_record
        qc["act_preparation"] = prior_record
        manifest["dependencies"]["act_prep_qc"] = _file_info(config.act_prep_qc_path)

    return_code: Optional[int] = None
    error: Optional[str] = None
    phase_timings: list[dict[str, object]] = []
    manifest['phase_timings'] = phase_timings
    qc['phase_timings'] = phase_timings
    def persist_timings():
        temporary = manifest_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        temporary.replace(manifest_path)

    start = time.monotonic()
    if config.execute:
        if not backend_status["available"]:
            missing = ", ".join(str(item) for item in backend_status["missing_commands"])
            raise RuntimeError(f"MRtrix3 CSD/ACT backend is incomplete. Missing command(s): {missing}")
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        # MRtrix3 ignores "-config RandomSeed <n>" on the command line; the
        # seed must be provided via the MRTRIX_RNG_SEED environment variable.
        command_env = external_python_environment(
            extra_env={"MRTRIX_RNG_SEED": str(config.random_seed)} if config.random_seed is not None else None,
            path_prepend=[config.mrtrix_bin_dir] if config.mrtrix_bin_dir is not None else None,
        )
        try:
            for command in commands:
                _emit(f"[Tractography] Running MRtrix CSD/ACT command: {' '.join(command)}", log)
                with _timed_step(phase_timings, Path(command[0]).name, persist_timings, command) as phase:
                    completed = subprocess.run(command, check=True, env=command_env, cwd=output_dir)
                    return_code = completed.returncode
                    phase['return_code'] = return_code
            if config.export_trk and tractogram_trk is not None:
                _emit(f"[Tractography] Exporting ACT TCK to TRK for WMQL: {tractogram_trk}", log)
                with _timed_step(phase_timings, 'export_tck_to_trk', persist_timings):
                    export_tck_to_trk(tractogram, config.dwi_path, tractogram_trk)
            with _timed_step(phase_timings, 'scientific_output_validation', persist_timings) as phase:
                scientific_summary, validation_error = _validate_generated_outputs(
                    config, tractogram, tractogram_trk, tractogram_vtk, weights,
                )
                qc["scientific_output_summary"] = scientific_summary
                if validation_error is not None:
                    error = validation_error
                    phase.update(status='failed', error=validation_error)
        except subprocess.CalledProcessError as exc:
            return_code = exc.returncode
            error = str(exc)
        except Exception as exc:
            return_code = 1
            error = str(exc)
        qc.update(
            {
                "status": "failed" if error else "completed",
                "executed": True,
                "return_code": return_code,
                "elapsed_seconds": time.monotonic() - start,
                "wm_fod_exists": wm_fod.is_file(),
                "wm_fod_norm_exists": wm_fod_norm.is_file(),
                "tractogram_exists": tractogram.is_file(),
                "tractogram_trk_exists": tractogram_trk.is_file() if tractogram_trk else None,
                "tractogram_vtk_exists": tractogram_vtk.is_file() if tractogram_vtk else None,
                "sift2_weights_exists": weights.is_file() if weights else None,
                "error": error,
            }
        )
        manifest["status"] = qc["status"]
        manifest["return_code"] = return_code
        manifest['runtime_seconds'] = qc['elapsed_seconds']
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    qc_path.write_text(json.dumps(qc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if error is not None:
        raise RuntimeError(f"MRtrix CSD/ACT execution failed: {error}")

    return CSDActResult(
        subject_id=config.subject_id,
        output_dir=output_dir,
        wm_fod_path=wm_fod,
        wm_fod_norm_path=wm_fod_norm,
        tractogram_path=tractogram,
        tractogram_trk_path=tractogram_trk,
        tractogram_vtk_path=tractogram_vtk,
        sift2_weights_path=weights,
        manifest_path=manifest_path,
        qc_path=qc_path,
        commands=commands,
        backend_status=backend_status,
        executed=bool(config.execute),
        return_code=return_code,
    )
