"""SAF short association fibre workflow facade.

The implementation keeps the translated SAF algorithm bundled under
``tractography/saf_backend`` and exposes a SlicerDSpace-style wrapper with
input validation, backend checks, manifest/QC output, and explicit execution.
"""

from __future__ import annotations

from shared.cohort_schedule import scheduled_stage

import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from shared import default_task_output_dir, find_subject_root
from shared.external_process import startup_environment, text_subprocess_kwargs, resolve_external_command
from .saf_backend.provenance import implementation_record


REQUIRED_SAF_PYTHON_PACKAGES = ("numpy", "scipy", "sklearn", "nibabel")
BASE_MRTRIX_COMMANDS = ("mrconvert",)
TRACTOGRAPHY_MRTRIX_COMMANDS = ("tckgen",)
LENGTH_FILTER_MRTRIX_COMMANDS = ("tckstats",)
OPTIONAL_MRTRIX_COMMANDS = ("tcksample",)
FREESURFER_COMMANDS = ("recon-all",)
SAF_INTERMEDIATE_NAMES = (
    "DWI_lh_pial.txt",
    "DWI_lh_white.txt",
    "DWI_rh_pial.txt",
    "DWI_rh_white.txt",
    "DWI_seeds_out.tck",
    "DWI_seeds_out.txt",
    "PS.mat",
    "SD.mat",
    "WSM_coords.txt",
    "brain_T1.nii.gz",
    "lengths.txt",
    "mcc.tck",
    "mgn.tck",
    "rej.tck",
    "surf_data",
    "surf_tracks.tck",
)


@dataclass(frozen=True)
class SAFConfig:
    """Configuration for Python SAF short association fibre extraction."""

    subject_id: str
    input_root: Path
    fs_dir: Path
    fod_path: Path
    output_dir: Optional[Path] = None
    fa_path: Optional[Path] = None
    pregenerated_tractogram: Optional[Path] = None
    streamline_weights: Optional[Path] = None
    tckgen_options: str = ""
    mrtrix_bin_dir: Optional[Path] = None
    mrtrix_prefix: str = ""
    notrunc: bool = False
    fast_intersection: bool = True
    gwg_hard: bool = False
    pialfilter: bool = False
    remesh_seeds: bool | float = False
    maxlen: Optional[float] = 90.0
    random_seed: int = 12345
    scalar2surf: tuple[Path, ...] = field(default_factory=tuple)
    device: str = "auto"
    force_recompute: bool = False
    diagnostics: bool = False
    execute: bool = False


@dataclass(frozen=True)
class SAFResult:
    """Prepared or completed SAF run."""

    subject_id: str
    output_dir: Path
    saf_tractogram: Path
    manifest_path: Path
    qc_path: Path
    executed: bool
    backend_status: dict[str, Any]
    saf_vtk_path: Optional[Path] = None
    streamline_count: Optional[int] = None
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
        "is_dir": expanded.is_dir(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _configured_mrtrix_bin_dir(mrtrix_bin_dir: Optional[Path] = None) -> Optional[Path]:
    if mrtrix_bin_dir is not None:
        return mrtrix_bin_dir.expanduser().resolve()
    env_value = os.environ.get("SLICERDSPACE_MRTRIX_BIN") or os.environ.get("MRTRIX_BIN")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return None


def _resolve_command(command: str, *, bin_dir: Optional[Path] = None) -> Optional[str]:
    found = resolve_external_command(command, bin_dir, require_executable=True)
    return Path(found).resolve().as_posix() if found else None


def _command_status(
    commands: tuple[str, ...],
    *,
    bin_dir: Optional[Path] = None,
    prefix: str = "",
) -> dict[str, Optional[str]]:
    if prefix and bin_dir is None:
        return {command: f"{prefix}{command}" for command in commands}
    return {command: _resolve_command(command, bin_dir=bin_dir) for command in commands}


def _package_status(packages: tuple[str, ...]) -> dict[str, bool]:
    return {package: importlib.util.find_spec(package) is not None for package in packages}


def bundled_saf_backend_dir() -> Path:
    return Path(__file__).resolve().parent / "saf_backend"


def inspect_saf_backend(
    *,
    mrtrix_bin_dir: Optional[Path] = None,
    mrtrix_prefix: str = "",
    require_recon_all: bool = False,
    require_mrconvert: bool = True,
    require_tckgen: bool = True,
    require_tckstats: bool = True,
    require_tcksample: bool = False,
) -> dict[str, Any]:
    """Inspect Python SAF, MRtrix3, and optional FreeSurfer dependencies."""

    packages = _package_status(REQUIRED_SAF_PYTHON_PACKAGES)
    resolved_mrtrix_bin_dir = _configured_mrtrix_bin_dir(mrtrix_bin_dir)
    required_mrtrix_commands = list(BASE_MRTRIX_COMMANDS) if require_mrconvert else []
    if require_tckgen:
        required_mrtrix_commands.extend(TRACTOGRAPHY_MRTRIX_COMMANDS)
    if require_tckstats:
        required_mrtrix_commands.extend(LENGTH_FILTER_MRTRIX_COMMANDS)
    mrtrix_required = _command_status(
        tuple(required_mrtrix_commands),
        bin_dir=resolved_mrtrix_bin_dir,
        prefix=mrtrix_prefix,
    )
    mrtrix_optional = _command_status(
        OPTIONAL_MRTRIX_COMMANDS,
        bin_dir=resolved_mrtrix_bin_dir,
        prefix=mrtrix_prefix,
    )
    freesurfer = _command_status(FREESURFER_COMMANDS) if require_recon_all else {}
    missing_required = [
        f"python:{name}" for name, available in packages.items() if not available
    ]
    missing_required.extend(
        f"mrtrix:{name}" for name, resolved in mrtrix_required.items() if resolved is None
    )
    missing_required.extend(
        f"freesurfer:{name}" for name, resolved in freesurfer.items() if resolved is None
    )
    missing_optional = [
        f"mrtrix:{name}" for name, resolved in mrtrix_optional.items() if resolved is None
    ]
    if require_tcksample:
        missing_required.extend(missing_optional)
        missing_optional = []
    backend_dir = bundled_saf_backend_dir()
    return {
        "name": "python_saf",
        "available": not missing_required and backend_dir.is_dir(),
        "backend_dir": backend_dir.as_posix(),
        "backend_dir_exists": backend_dir.is_dir(),
        "packages": packages,
        "mrtrix": {
            "bin_dir": resolved_mrtrix_bin_dir.as_posix() if resolved_mrtrix_bin_dir else None,
            "bin_dir_exists": resolved_mrtrix_bin_dir.is_dir() if resolved_mrtrix_bin_dir else None,
            "prefix": mrtrix_prefix,
            "discovery": "explicit_or_env" if resolved_mrtrix_bin_dir else ("prefix_unchecked" if mrtrix_prefix else "PATH"),
        },
        "mrtrix_required": mrtrix_required,
        "mrtrix_optional": mrtrix_optional,
        "freesurfer": freesurfer,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
    }


def _validate_config(config: SAFConfig) -> None:
    if not config.subject_id.strip():
        raise ValueError("subject_id is required.")
    if not config.fs_dir.expanduser().is_dir():
        raise FileNotFoundError(f"FreeSurfer directory not found: {config.fs_dir}")
    if not config.fod_path.expanduser().is_file():
        raise FileNotFoundError(f"FOD image not found: {config.fod_path}")
    if config.pregenerated_tractogram is not None and not config.pregenerated_tractogram.expanduser().is_file():
        raise FileNotFoundError(f"Pregenerated tractogram not found: {config.pregenerated_tractogram}")
    if config.streamline_weights is not None and not config.streamline_weights.expanduser().is_file():
        raise FileNotFoundError(f"Streamline weights not found: {config.streamline_weights}")
    if config.mrtrix_bin_dir is not None and not config.mrtrix_bin_dir.expanduser().is_dir():
        raise FileNotFoundError(f"MRtrix bin directory not found: {config.mrtrix_bin_dir}")
    for scalar in config.scalar2surf:
        if not scalar.expanduser().is_file():
            raise FileNotFoundError(f"Scalar-to-surface image not found: {scalar}")
    if config.maxlen is not None and config.maxlen <= 0:
        raise ValueError("maxlen must be > 0 or None.")
    if config.random_seed < 0:
        raise ValueError("random_seed must be >= 0.")
    dev = (config.device or "auto").strip().lower()
    if dev not in {"auto", "cpu", "cuda"} and not dev.startswith("cuda:"):
        raise ValueError("SAF device must be 'auto', 'cpu', or 'cuda'")


def _resolve_saf_device(device: str = "auto") -> str:
    dev = (device or "auto").strip().lower()
    if dev not in {"auto", "cpu", "cuda"} and not dev.startswith("cuda:"):
        raise ValueError("SAF device must be 'auto', 'cpu', or 'cuda'")
    return "cpu"


def _effective_tckgen_options(config: SAFConfig) -> str:
    """Return the tckgen options SAF should use.

    MRtrix3 ignores ``-config RandomSeed <n>`` passed on the command line, so
    the reproducible seed is applied via the ``MRTRIX_RNG_SEED`` environment
    variable (see ``_mrtrix_rng_seed_env``) instead of being injected here.
    """

    return config.tckgen_options.strip()


def _tckgen_command_options(config: SAFConfig) -> str:
    """tckgen options for the SAF command line: an explicit ``-nthreads`` wins, else the stage reservation.

    SPEC-GAP: the stage CPU reservation is applied only here, not in the recorded options,
    so the cache record does not depend on the admitted CPU count.
    """

    from shared.numeric_threads import reserved_stage_cpus

    options = _effective_tckgen_options(config)
    reserved = reserved_stage_cpus()
    if reserved is None or "-nthreads" in options.split():
        return options
    return f"{options} -nthreads {reserved}".strip()


@contextmanager
def _mrtrix_rng_seed_env(seed: int) -> Iterator[None]:
    """Temporarily export MRTRIX_RNG_SEED for the SAF backend's subprocess calls.

    The bundled SAF backend shells out to MRtrix3 commands (see
    ``saf_backend/Support/sys_exec.py``); those subprocesses inherit their
    environment from ``os.environ`` at call time, so setting it here (and
    restoring it afterward) makes MRtrix3's RNG seed reproducible without
    relying on the ineffective ``-config RandomSeed`` command-line option.
    """

    previous = os.environ.get("MRTRIX_RNG_SEED")
    os.environ["MRTRIX_RNG_SEED"] = str(seed)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("MRTRIX_RNG_SEED", None)
        else:
            os.environ["MRTRIX_RNG_SEED"] = previous


def _resolved_mrtrix_command(config: SAFConfig, command: str) -> Optional[str]:
    bin_dir = _configured_mrtrix_bin_dir(config.mrtrix_bin_dir)
    return _resolve_command(command, bin_dir=bin_dir)


def _run_mrtrix_command(
    args: list[str],
    *,
    config: Optional[SAFConfig] = None,
    log: Optional[Callable[[str], None]] = None,
) -> subprocess.CompletedProcess[str]:
    _emit("[SAF] " + " ".join(args), log)
    env = startup_environment()
    if sys.platform == "win32" and not env.get("HOME"):
        env["HOME"] = env.get("USERPROFILE") or str(Path.home())
    if config is not None:
        bin_dir = _configured_mrtrix_bin_dir(config.mrtrix_bin_dir)
        if bin_dir is not None:
            env["PATH"] = os.pathsep.join(filter(None, (str(bin_dir), env.get("PATH"))))
    return subprocess.run(
        args, check=True, capture_output=True, env=env, **text_subprocess_kwargs()
    )


def _count_tck_streamlines(tck_path: Path, *, config: SAFConfig, log: Optional[Callable[[str], None]] = None) -> tuple[Optional[int], Optional[str]]:
    tckstats = _resolved_mrtrix_command(config, "tckstats")
    if tckstats is None:
        return None, "mrtrix:tckstats not found"
    try:
        completed = _run_mrtrix_command([tckstats, "-output", "count", tck_path.as_posix()], config=config, log=log)
        return int(completed.stdout.strip().split()[0]), None
    except Exception as exc:
        return None, str(exc)


def _export_tck_to_vtk(
    tck_path: Path,
    vtk_path: Path,
    *,
    config: SAFConfig,
    log: Optional[Callable[[str], None]] = None,
) -> tuple[bool, Optional[str]]:
    tckconvert = _resolved_mrtrix_command(config, "tckconvert")
    if tckconvert is None:
        return False, "mrtrix:tckconvert not found"
    try:
        _run_mrtrix_command([tckconvert, tck_path.as_posix(), vtk_path.as_posix(), "-force"], config=config, log=log)
        return vtk_path.is_file(), None if vtk_path.is_file() else "tckconvert finished without VTK output"
    except Exception as exc:
        return False, str(exc)


def _build_saf_args(config: SAFConfig, output_dir: Path) -> list[Any]:
    args: list[Any] = [
        "fs_dir",
        config.fs_dir.expanduser().resolve().as_posix(),
        "fod",
        config.fod_path.expanduser().resolve().as_posix(),
        "outf",
        output_dir.as_posix(),
    ]
    if config.fa_path is not None:
        args.extend(["fa", config.fa_path.expanduser().resolve().as_posix()])
    if config.notrunc:
        args.append("notrunc")
    if not config.fast_intersection:
        args.extend(["fastx", "off"])
    if config.gwg_hard:
        args.append("gwg_hard")
    if config.pialfilter:
        args.append("pialfilter")
    if config.remesh_seeds:
        args.extend(["remesh_seeds", config.remesh_seeds])
    if config.maxlen is None:
        args.extend(["maxlen", ""])
    else:
        args.extend(["maxlen", float(config.maxlen)])
    if config.pregenerated_tractogram is None:
        args.extend(["tckgen", _tckgen_command_options(config)])
    elif config.tckgen_options:
        args.extend(["tckgen", config.tckgen_options])
    if config.pregenerated_tractogram is not None:
        args.extend(["noseed", config.pregenerated_tractogram.expanduser().resolve().as_posix()])
    if config.streamline_weights is not None:
        args.extend(["sl_weights", config.streamline_weights.expanduser().resolve().as_posix()])
    if config.mrtrix_prefix:
        args.extend(["mrtrix_prefix", config.mrtrix_prefix])
    mrtrix_bin_dir = _configured_mrtrix_bin_dir(config.mrtrix_bin_dir)
    if mrtrix_bin_dir is not None:
        args.extend(["mrtrix_bin_dir", mrtrix_bin_dir.as_posix()])
    if config.force_recompute:
        args.append("force")
    if config.diagnostics:
        args.append("dgn")
    for scalar in config.scalar2surf:
        args.extend(["scalar2surf", scalar.expanduser().resolve().as_posix()])
    return args


@contextmanager
def _saf_backend_import_path() -> Iterator[None]:
    backend_dir = bundled_saf_backend_dir()
    backend = backend_dir.as_posix()
    added = backend not in sys.path
    if added:
        sys.path.insert(0, backend)
    try:
        yield
    finally:
        if added:
            try:
                sys.path.remove(backend)
            except ValueError:
                pass


def _saf_cache_dir(output_dir: Path) -> Path:
    return output_dir / "cache"


def _saf_output_or_cache_path(output_dir: Path, name: str) -> Path:
    direct = output_dir / name
    if direct.exists() or direct.is_symlink():
        return direct
    return _saf_cache_dir(output_dir) / name


def _remove_cache_entry(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _cache_saf_intermediates(output_dir: Path) -> dict[str, str]:
    """Move non-product SAF artifacts into a stable, replaceable cache."""

    moved: dict[str, str] = {}
    cache_dir = _saf_cache_dir(output_dir)
    for name in SAF_INTERMEDIATE_NAMES:
        src = output_dir / name
        if not src.exists() and not src.is_symlink():
            continue
        cache_dir.mkdir(parents=True, exist_ok=True)
        dest = cache_dir / name
        _remove_cache_entry(dest)
        # Versions before the stable-cache fix appended .1, .2, ... on every
        # forced rerun. Remove only those numeric legacy siblings.
        for legacy in cache_dir.glob(f"{name}.*"):
            suffix = legacy.name[len(name) + 1 :]
            if suffix.isdigit():
                _remove_cache_entry(legacy)
        shutil.move(src.as_posix(), dest.as_posix())
        moved[name] = dest.as_posix()
    return moved


def _summarize_saf_outputs(output_dir: Path) -> dict[str, Any]:
    saf_tractogram = output_dir / "SAF.tck"
    saf_vtk = output_dir / "SAF.vtk"
    surf_data = _saf_output_or_cache_path(output_dir, "surf_data")
    return {
        "saf_tractogram": _file_info(saf_tractogram),
        "saf_vtk": _file_info(saf_vtk),
        "weights": _file_info(output_dir / "SAF.wts"),
        "log": _file_info(output_dir / "log.txt"),
        "ps_mat": _file_info(_saf_output_or_cache_path(output_dir, "PS.mat")),
        "sd_mat": _file_info(_saf_output_or_cache_path(output_dir, "SD.mat")),
        "surface_data_dir": _file_info(surf_data),
        "surface_data_files": sorted(path.name for path in surf_data.iterdir()) if surf_data.is_dir() else [],
    }


def _tractogram_summary(path: Path) -> dict[str, Any]:
    """Decode all SAF streamlines and summarize their physical lengths."""

    import nibabel as nib
    import numpy as np

    tck = nib.streamlines.load(str(path), lazy_load=False).streamlines
    if hasattr(tck, "_data") and hasattr(tck, "_offsets") and hasattr(tck, "_lengths"):
        data = np.asarray(tck._data, dtype=np.float64)
        offsets = np.asarray(tck._offsets, dtype=np.int64)
        sl_lengths = np.asarray(tck._lengths, dtype=np.int64)
        valid_sl = sl_lengths >= 2
        all_points_finite = bool(np.all(valid_sl)) and bool(np.isfinite(data).all())
        if len(sl_lengths) > 0 and all_points_finite:
            seg_counts = sl_lengths - 1
            total_segs = int(np.sum(seg_counts))
            if total_segs > 0:
                group_starts = np.empty(len(seg_counts), dtype=np.int64)
                group_starts[0] = 0
                if len(seg_counts) > 1:
                    np.cumsum(seg_counts[:-1], out=group_starts[1:])
                p0_idx = np.repeat(offsets, seg_counts) + (
                    np.arange(total_segs, dtype=np.int64) - np.repeat(group_starts, seg_counts)
                )
                seg_norms = np.linalg.norm(data[p0_idx + 1] - data[p0_idx], axis=1)
                values = np.add.reduceat(seg_norms, group_starts)
            else:
                values = np.zeros(0, dtype=np.float64)
            return _format_tractogram_summary(path, values, all_points_finite)

    lengths: list[float] = []
    all_points_finite = True
    for streamline in tck:
        points = np.asarray(streamline, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 3:
            all_points_finite = False
            continue
        if not np.isfinite(points).all():
            all_points_finite = False
            continue
        lengths.append(float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()))
    values = np.asarray(lengths, dtype=np.float64)
    return _format_tractogram_summary(path, values, all_points_finite)


def _format_tractogram_summary(path: Path, values: Any, all_points_finite: bool) -> dict[str, Any]:
    import numpy as np

    summary: dict[str, Any] = {
        "file": _file_info(path),
        "streamline_count": int(values.size),
        "all_points_finite": bool(all_points_finite),
    }
    if values.size:
        summary["length_mm"] = {
            "min": float(np.min(values)),
            "p05": float(np.percentile(values, 5)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
        }
    return summary


@scheduled_stage('saf')
def run_saf(config: SAFConfig, *, log: Optional[Callable[[str], None]] = None) -> SAFResult:
    """Prepare or execute Python SAF for one subject.

    ``execute=False`` writes manifest/QC only. ``execute=True`` imports the
    bundled Python SAF backend in-process and runs the translated SAF pipeline.
    """

    _validate_config(config)
    raw = find_subject_root(config.input_root.expanduser().resolve(), config.subject_id)
    output_dir = (
        config.output_dir.expanduser().resolve()
        if config.output_dir is not None
        else default_task_output_dir(raw.subject_root, "saf")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "saf_manifest.json"
    qc_path = output_dir / "saf_qc.json"
    saf_tractogram = output_dir / "SAF.tck"
    saf_vtk_path = output_dir / "SAF.vtk"
    backend_status = inspect_saf_backend(
        mrtrix_bin_dir=config.mrtrix_bin_dir,
        mrtrix_prefix=config.mrtrix_prefix,
        require_recon_all=False,
        require_mrconvert=not (output_dir / "brain_T1.nii.gz").is_file(),
        require_tckgen=config.pregenerated_tractogram is None,
        require_tckstats=config.maxlen is not None or config.streamline_weights is not None,
        require_tcksample=bool(config.fa_path or config.scalar2surf),
    )
    args = _build_saf_args(config, output_dir)
    manifest = {
        "kind": "saf_short_association_fibres",
        "implementation": implementation_record(),
        "version": 2,
        "subject_id": raw.subject_id,
        "backend": backend_status,
        "executed": bool(config.execute),
        "status": "planned" if not config.execute else "running",
        "parameters": {
            "surface_space": "native_dMRI",
            "notrunc": config.notrunc,
            "fast_intersection": config.fast_intersection,
            "gwg_hard": config.gwg_hard,
            "pialfilter": config.pialfilter,
            "remesh_seeds": config.remesh_seeds,
            "maxlen": config.maxlen,
            "random_seed": config.random_seed,
            "tckgen_options": (
                _effective_tckgen_options(config)
                if config.pregenerated_tractogram is None
                else config.tckgen_options
            ),
            "mrtrix_bin_dir": config.mrtrix_bin_dir.expanduser().resolve().as_posix() if config.mrtrix_bin_dir else None,
            "mrtrix_prefix": config.mrtrix_prefix,
            "effective_mrtrix_prefix": config.mrtrix_prefix,
            "initial_tractogram_source": "pregenerated" if config.pregenerated_tractogram else "mrtrix_tckgen",
            "force_recompute": config.force_recompute,
            "diagnostics": config.diagnostics,
        },
        "reproducibility_note": (
            "random_seed is forwarded to tckgen via MRTRIX_RNG_SEED. A "
            "thread count can be pinned by passing -nthreads N as part of "
            "tckgen_options above (it is recorded verbatim there when set); "
            "multi-threaded tckgen is NOT bit-reproducible even with a "
            "fixed random_seed, because threads consume the shared "
            "random-number stream in a run-dependent order. Only "
            "-nthreads 0 (fully single-threaded) is bit-reproducible for a "
            "given random_seed."
        ),
        "inputs": {
            "subject_root": raw.subject_root.as_posix(),
            "fs_dir": _file_info(config.fs_dir),
            "fod": _file_info(config.fod_path),
            "fa": _file_info(config.fa_path),
            "pregenerated_tractogram": _file_info(config.pregenerated_tractogram),
            "streamline_weights": _file_info(config.streamline_weights),
            "scalar2surf": [_file_info(path) for path in config.scalar2surf],
        },
        "saf_args": [str(arg) for arg in args],
        "outputs": {
            "saf_tractogram": saf_tractogram.as_posix(),
            "qc": qc_path.as_posix(),
            "summary": _summarize_saf_outputs(output_dir),
        },
    }
    resolved_saf_device = _resolve_saf_device(config.device) if config.execute else (config.device or "auto")
    manifest["parameters"]["device"] = resolved_saf_device
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _emit(f"[SAF] Manifest: {manifest_path}", log)
    start_time = time.monotonic()
    return_code: Optional[int] = None
    execution_error: Optional[str] = None
    execution_exception: Optional[Exception] = None
    if config.execute:
        if not backend_status["available"]:
            missing = ", ".join(str(item) for item in backend_status["missing_required"])
            return_code = 1
            execution_error = f"SAF backend is incomplete. Missing: {missing}"
        else:
            cwd = Path.cwd()
            prev_saf_dev = os.environ.get("SLICERDSPACE_SAF_DEVICE")
            os.environ["SLICERDSPACE_SAF_DEVICE"] = resolved_saf_device
            try:
                with _mrtrix_rng_seed_env(config.random_seed), _saf_backend_import_path():
                    module = importlib.import_module("SAF_pipeline")
                    module.SAF_pipeline(*args)
                return_code = 0
            except Exception as exc:
                return_code = 1
                execution_error = str(exc)
                execution_exception = exc
            finally:
                if prev_saf_dev is None:
                    os.environ.pop("SLICERDSPACE_SAF_DEVICE", None)
                else:
                    os.environ["SLICERDSPACE_SAF_DEVICE"] = prev_saf_dev
                os.chdir(cwd)

    elapsed_seconds = time.monotonic() - start_time
    output_summary = _summarize_saf_outputs(output_dir)
    if config.execute and execution_error is None and not saf_tractogram.is_file():
        execution_error = "SAF backend finished without SAF.tck."
        return_code = 1
    streamline_count: Optional[int] = None
    streamline_count_error: Optional[str] = None
    vtk_export_error: Optional[str] = None
    scientific_output_summary: dict[str, Any] = {}
    cached_intermediates: dict[str, str] = {}
    if config.execute and execution_error is None and saf_tractogram.is_file():
        streamline_count, streamline_count_error = _count_tck_streamlines(saf_tractogram, config=config, log=log)
        vtk_ok, vtk_export_error = _export_tck_to_vtk(saf_tractogram, saf_vtk_path, config=config, log=log)
        if vtk_ok:
            vtk_export_error = None
        try:
            scientific_output_summary = _tractogram_summary(saf_tractogram)
        except Exception as exc:
            execution_error = f"Unable to decode SAF.tck for QC: {type(exc).__name__}: {exc}"
        if execution_error is None:
            measured_count = int(scientific_output_summary.get("streamline_count", 0) or 0)
            length_summary = scientific_output_summary.get("length_mm") or {}
            if streamline_count_error or streamline_count is None or streamline_count <= 0:
                execution_error = streamline_count_error or "SAF.tck contains no streamlines."
            elif measured_count != streamline_count or not scientific_output_summary.get("all_points_finite"):
                execution_error = "SAF.tck failed decoded streamline-count or finite-coordinate QC."
            elif config.maxlen is not None and (
                not length_summary
                or float(length_summary.get("max", float("inf"))) > float(config.maxlen) + 1e-3
            ):
                execution_error = f"SAF.tck contains streamlines longer than the configured {config.maxlen:g} mm limit."
            elif vtk_export_error or not saf_vtk_path.is_file():
                execution_error = vtk_export_error or "SAF VTK export is missing."
        if not config.diagnostics:
            cached_intermediates = _cache_saf_intermediates(output_dir)
        output_summary = _summarize_saf_outputs(output_dir)
    qc = {
        "kind": "saf_qc",
        "implementation": implementation_record(),
        "human_review": "pending",
        "scientific_scope": "Decoded tractogram count, finite coordinates, physical length limit and VTK export; not anatomical validation or MATLAB end-to-end equivalence.",
        "version": 2,
        "subject_id": raw.subject_id,
        "status": "failed" if execution_error else ("completed" if config.execute else "planned"),
        "executed": bool(config.execute),
        "return_code": return_code,
        "elapsed_seconds": elapsed_seconds,
        "backend_available": bool(backend_status["available"]),
        "streamline_count": streamline_count,
        "streamline_count_error": streamline_count_error,
        "vtk_export_error": vtk_export_error,
        "cached_intermediates": cached_intermediates,
        "output_summary": output_summary,
        "scientific_output_summary": scientific_output_summary,
        "error": execution_error,
    }
    qc_path.write_text(json.dumps(qc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest["status"] = qc["status"]
    manifest["return_code"] = return_code
    manifest["streamline_count"] = streamline_count
    manifest["streamline_count_error"] = streamline_count_error
    manifest["vtk_export_error"] = vtk_export_error
    manifest["cached_intermediates"] = cached_intermediates
    manifest["error"] = execution_error
    manifest["outputs"]["summary"] = output_summary
    manifest["scientific_output_summary"] = scientific_output_summary
    manifest["outputs"]["saf_vtk"] = saf_vtk_path.as_posix()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if execution_error is not None:
        error = RuntimeError(f"SAF execution failed: {execution_error}")
        if execution_exception is not None:
            raise error from execution_exception
        raise error

    return SAFResult(
        subject_id=raw.subject_id,
        output_dir=output_dir,
        saf_tractogram=saf_tractogram,
        manifest_path=manifest_path,
        qc_path=qc_path,
        executed=bool(config.execute),
        backend_status=backend_status,
        saf_vtk_path=saf_vtk_path if saf_vtk_path.is_file() else None,
        streamline_count=streamline_count,
        return_code=return_code,
    )
