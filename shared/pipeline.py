"""Shared caching helpers for raw DWI, native DTI, and registered DTI bundles."""

from __future__ import annotations

from shared.cohort_schedule import scheduled_stage

import json
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import time
from uuid import uuid4
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from functools import lru_cache
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Optional

from shared.external_process import merged_startup_environment, text_subprocess_kwargs, slicer_runtime_roots
from shared.brain_mask import BrainMaskConfig, generate_brain_mask
from shared.cache_lock import cache_lock

LOGGER = logging.getLogger(__name__)

slicer = sys.modules.get("slicer")
if slicer is None:  # pragma: no cover - depends on the host executable
    # Slicer's private ``python-real`` can find the ``slicer`` package without
    # having initialized the application.  That failed import prints a helper
    # error to stdout before raising, which corrupts JSON-only CLI responses.
    # Reuse the already loaded module in Slicer proper; otherwise treat the
    # optional probe as silent and unavailable.
    try:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            import slicer as _slicer
    except Exception as exc:
        slicer = None  # type: ignore[assignment]
        SLICER_IMPORT_ERROR = exc
    else:
        slicer = _slicer
        SLICER_IMPORT_ERROR = None
else:
    SLICER_IMPORT_ERROR = None

try:  # pragma: no cover - optional runtime dependency
    import SimpleITK as sitk
except ImportError as exc:  # pragma: no cover
    sitk = None  # type: ignore[assignment]
    SIMPLEITK_IMPORT_ERROR = exc
else:
    SIMPLEITK_IMPORT_ERROR = None


RAW_INPUT_FILES = {
    "dwi": "data.nii.gz",
    "bval": "bvals",
    "bvec": "bvecs",
    "mask": "nodif_brain_mask.nii.gz",
}

RAW_REQUIRED_INPUT_KEYS = ("dwi", "bval", "bvec")


@dataclass(frozen=True)
class RawInputBundle:
    subject_root: Path
    subject_id: str
    input_dir: Path
    dwi_path: Path
    bval_path: Path
    bvec_path: Path
    mask_path: Path


@dataclass(frozen=True)
class NativeDTIBundle:
    subject_root: Path
    subject_id: str
    fit_method: str
    bundle_dir: Path
    dwi_nhdr_path: Path
    mask_nhdr_path: Path
    dti_nrrd_path: Path
    b0_nrrd_path: Path
    fa_nrrd_path: Path
    trace_nrrd_path: Path
    mean_diffusivity_nrrd_path: Path
    min_eig_nrrd_path: Path
    mid_eig_nrrd_path: Path
    max_eig_nrrd_path: Path
    ddseg_md_nrrd_path: Path
    ddseg_e1_nrrd_path: Path
    ddseg_e2_nrrd_path: Path
    ddseg_e3_nrrd_path: Path
    fa_nii_path: Path
    trace_nii_path: Path
    mean_diffusivity_nii_path: Path
    min_eig_nii_path: Path
    mid_eig_nii_path: Path
    max_eig_nii_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class RegisteredDTIBundle:
    subject_root: Path
    subject_id: str
    fit_method: str
    atlas_path: Path
    bundle_dir: Path
    fa_reg_path: Path
    trace_reg_path: Path
    mean_diffusivity_reg_path: Path
    min_eig_reg_path: Path
    mid_eig_reg_path: Path
    max_eig_reg_path: Path
    mask_reg_path: Path
    transform_path: Path
    inverse_transform_path: Path
    manifest_path: Path
    scalar_format: str = "nifti"


def _emit(message: str, log: Optional[Callable[[str], None]] = None) -> None:
    LOGGER.info(message)
    if log:
        log(message)


def _require_slicer() -> None:
    if SLICER_IMPORT_ERROR is not None or slicer is None:
        raise RuntimeError("Shared DTI bundle generation requires the Slicer Python environment.") from SLICER_IMPORT_ERROR


def _slicer_home_candidates() -> list[Path]:
    candidates: list[Path] = []
    if slicer is not None and hasattr(slicer, "app"):
        try:
            slicer_home = getattr(slicer.app, "slicerHome", "")
            if slicer_home:
                candidates.append(Path(str(slicer_home)).expanduser().resolve())
        except Exception:
            pass
    for env_key in ("SLICER_HOME", "Slicer_HOME"):
        env_value = os.environ.get(env_key)
        if env_value:
            candidates.append(Path(env_value).expanduser().resolve())
    executable = Path(sys.executable).expanduser().resolve()
    if executable.stem.lower() in {"python-real", "pythonslicer", "python"}:
        candidates.append(executable.parent.parent)
    if executable.name.startswith("Slicer"):
        candidates.append(executable.parent)
    unique: list[Path] = []
    for candidate in candidates:
        for root in slicer_runtime_roots(candidate):
            if root not in unique:
                unique.append(root)
    return unique


def _resolve_slicer_home() -> Path:
    for candidate in _slicer_home_candidates():
        if (candidate / "lib").is_dir() or (candidate / "bin").is_dir():
            return candidate
    searched = ", ".join(path.as_posix() for path in _slicer_home_candidates()) or "none"
    raise RuntimeError(f"Failed to resolve Slicer home. Candidate paths: {searched}")


def is_raw_subject_root(path: Path) -> bool:
    input_dir = path / "input"
    return input_dir.is_dir() and all((input_dir / RAW_INPUT_FILES[key]).exists() for key in RAW_REQUIRED_INPUT_KEYS)


def inspect_subject_directory(base_path: Path, subject_id: str) -> dict[str, Any]:
    """Classify a path as a new-style subject root or legacy registered directory."""
    expanded = base_path.expanduser().resolve()
    subject_dir = expanded / subject_id

    if is_raw_subject_root(expanded):
        return {
            "mode": "subject_root",
            "subject_root": expanded,
            "subject_dir": expanded,
        }

    if subject_dir.is_dir() and is_raw_subject_root(subject_dir):
        return {
            "mode": "subject_root",
            "subject_root": subject_dir,
            "subject_dir": subject_dir,
        }

    if _has_registered_inputs(expanded, subject_id):
        return {
            "mode": "legacy_registered",
            "subject_root": None,
            "subject_dir": expanded,
        }

    if subject_dir.is_dir() and _has_registered_inputs(subject_dir, subject_id):
        return {
            "mode": "legacy_registered",
            "subject_root": None,
            "subject_dir": subject_dir,
        }

    from shared.dicom_input import is_dicom_root
    if is_dicom_root(expanded):
        return {"mode": "dicom", "subject_root": expanded, "subject_dir": expanded}

    return {
        "mode": "missing",
        "subject_root": None,
        "subject_dir": subject_dir if subject_dir.is_dir() else expanded,
    }


def find_subject_root(
    base_path: Path,
    subject_id: str,
    *,
    mrtrix_bin_dir: Optional[Path] = None,
    brain_mask_config: Optional[BrainMaskConfig] = None,
    generate_missing_mask: bool = True,
) -> RawInputBundle:
    inspected = inspect_subject_directory(base_path, subject_id)
    if inspected["mode"] != "subject_root":
        raise FileNotFoundError(
            f"Could not resolve a subject root with input/data.nii.gz under '{base_path}' for subject '{subject_id}'."
        )

    subject_root = Path(inspected["subject_root"])
    input_dir = subject_root / "input"
    mask_path = input_dir / RAW_INPUT_FILES["mask"]
    generated_record = mask_path.with_name("nodif_brain_mask.manifest.json")
    verify_generated = False
    if generated_record.is_file():
        try:
            verify_generated = json.loads(generated_record.read_text(encoding="utf-8")).get("version") == 2
        except (OSError, ValueError, AttributeError):
            verify_generated = True
    if generate_missing_mask and (not mask_path.exists() or verify_generated):
        generate_brain_mask(
            input_dir / RAW_INPUT_FILES["dwi"],
            input_dir / RAW_INPUT_FILES["bval"],
            input_dir / RAW_INPUT_FILES["bvec"],
            mask_path,
            config=brain_mask_config or BrainMaskConfig(mrtrix_bin_dir=mrtrix_bin_dir),
        )
    return RawInputBundle(
        subject_root=subject_root,
        subject_id=subject_id,
        input_dir=input_dir,
        dwi_path=input_dir / RAW_INPUT_FILES["dwi"],
        bval_path=input_dir / RAW_INPUT_FILES["bval"],
        bvec_path=input_dir / RAW_INPUT_FILES["bvec"],
        mask_path=mask_path,
    )


def resolve_preprocessed_input_bundle(
    raw: RawInputBundle,
    *,
    output_root: Optional[Path] = None,
) -> RawInputBundle:
    """Use completed preprocessing outputs as an analysis bundle when available.

    Downstream quantitative models must consume the same dMRI, gradients and
    mask emitted by preprocessing. If the user has not run preprocessing, the
    canonical input bundle remains the explicit fallback.
    """
    resolved_output_root = (
        output_root.expanduser().resolve()
        if output_root is not None
        else raw.subject_root / "outputs"
    )
    preproc_dir = resolved_output_root / "01_preproc"
    manifest_path = preproc_dir / f"{raw.subject_id}_preproc_manifest.json"
    candidates = {
        "dwi": preproc_dir / f"{raw.subject_id}_dwi_final.nii.gz",
        "bval": preproc_dir / f"{raw.subject_id}_dwi_final.bval",
        "bvec": preproc_dir / f"{raw.subject_id}_dwi_final.bvec",
        "mask": preproc_dir / f"{raw.subject_id}_brain_mask_final.nii.gz",
    }
    if not manifest_path.is_file() or not all(path.is_file() for path in candidates.values()):
        return raw
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return raw
    if manifest.get("status") not in {"completed", "cached"}:
        return raw
    from preprocessing.pipeline import preprocessing_gradients_current
    if not preprocessing_gradients_current(manifest):
        return raw
    if "cache_key" in manifest:
        cache_key = manifest.get("cache_key")
        inputs = cache_key.get("inputs") if isinstance(cache_key, dict) else None
        if not isinstance(inputs, dict):
            return raw
        for key, path in {"dwi": raw.dwi_path, "bval": raw.bval_path,
                          "bvec": raw.bvec_path, "mask": raw.mask_path}.items():
            record = inputs.get(key)
            try:
                current = _file_signature(path)
                if not isinstance(record, dict) or any(record.get(field) != value for field, value in current.items()):
                    return raw
            except OSError:
                return raw
    return RawInputBundle(
        subject_root=raw.subject_root,
        subject_id=raw.subject_id,
        input_dir=preproc_dir,
        dwi_path=candidates["dwi"],
        bval_path=candidates["bval"],
        bvec_path=candidates["bvec"],
        mask_path=candidates["mask"],
    )


def _has_registered_inputs(subject_dir: Path, subject_id: str) -> bool:
    expected = [
        f"{subject_id}-dti-FractionalAnisotropy-Reg.nii.gz",
        f"{subject_id}-dti-MinEigenvalue-Reg.nii.gz",
        f"{subject_id}-dti-MidEigenvalue-Reg.nii.gz",
        f"{subject_id}-dti-MaxEigenvalue-Reg.nii.gz",
        f"{subject_id}-dti-Trace-Reg.nii.gz",
        f"{subject_id}-dti-MeanDiffusivity-Reg.nii.gz",
        f"{subject_id}-mask-Reg.nii.gz",
        f"{subject_id}-b0ToAtlasT2.tfm",
    ]
    return subject_dir.is_dir() and all((subject_dir / name).exists() for name in expected)


def _file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _cache_file_signature(path: Path) -> dict[str, Any]:
    return {**_file_signature(path), 'ctime_ns': path.stat().st_ctime_ns}


def _read_manifest(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # pragma: no cover - corrupt manifest fallback
        LOGGER.warning("Unable to parse manifest: %s", path)
        return None


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _content_digest(path: str) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else _stream_digest(stream)


@lru_cache(maxsize=128)
def _artifact_digest(path: str, identity: tuple) -> str:
    return _content_digest(path)


def _stream_digest(stream) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(block)
    return digest.hexdigest()


def _artifact_record(path: Path, *, verify_content: bool = False) -> dict:
    stat = path.stat()
    if not path.is_file() or stat.st_size == 0:
        raise ValueError(f'Missing or empty shared artifact: {path}')
    identity = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    digest = (_content_digest(str(path.resolve())) if verify_content
              else _artifact_digest(str(path.resolve()), identity))
    after = path.stat()
    if identity != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError(f'Shared artifact changed while being checked: {path}')
    return dict(path=str(path.resolve()), sha256=digest)


def _artifacts_current(directory: Path, files) -> bool:
    try:
        return _read_manifest(directory / 'artifacts.json') == {
            str(path): _artifact_record(path) for path in files}
    except (OSError, ValueError):
        return False


def _save_artifacts(directory: Path, files) -> None:
    _write_manifest(directory / 'artifacts.json', {str(path): _artifact_record(path) for path in files})


def _conversion_implementation() -> dict:
    root = Path(__file__).resolve().parents[1] / 'anatomy/parcellation/conversion'
    return {name: _artifact_record(root / name)['sha256'] for name in ('nhdr_write.py', 'bval_bvec_io.py')}


def _run_cli(cmd: list[str], log: Optional[Callable[[str], None]] = None) -> None:
    _emit(f"[Shared] Running: {' '.join(cmd)}", log)
    process = subprocess.Popen(
        cmd,
        env=merged_startup_environment(build_cli_environment(cmd[0])),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        **text_subprocess_kwargs(bufsize=1),
    )
    if process.stdout:
        for line in process.stdout:
            _emit(f"[Shared] {line.rstrip()}", log)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"CLI command failed with exit code {return_code}")


def _get_cli_executable(module_name: str) -> str:
    return resolve_cli_executable(module_name)


def resolve_launcher_executable() -> str:
    _require_slicer()
    if hasattr(slicer, "app"):
        launcher = getattr(slicer.app, "launcherExecutableFilePath", None)
        if callable(launcher):
            launcher = launcher()
        if isinstance(launcher, str) and launcher:
            return Path(launcher).resolve().as_posix()

    slicer_home = _resolve_slicer_home()
    for fallback in (
        slicer_home / "Slicer",
        slicer_home / "Slicer.exe",
        slicer_home / "bin" / "Slicer",
        slicer_home / "bin" / "Slicer.exe",
        slicer_home / "MacOS" / "Slicer",
    ):
        if fallback.is_file():
            return fallback.as_posix()
    raise RuntimeError("Failed to resolve the Slicer launcher executable path.")


def resolve_pythonslicer_executable() -> str:
    _require_slicer()
    slicer_home = _resolve_slicer_home()
    candidates = [
        slicer_home / "bin" / "PythonSlicer",
        slicer_home / "bin" / "PythonSlicer.exe",
        slicer_home / "PythonSlicer",
        slicer_home / "PythonSlicer.exe",
        slicer_home / "MacOS" / "PythonSlicer",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.as_posix()
    raise RuntimeError("Failed to resolve the PythonSlicer executable path.")


def build_cli_environment(cli_executable: str) -> dict[str, str]:
    env = os.environ.copy()
    requested = env.get("OMP_NUM_THREADS", "")
    if (not env.get("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS")
            and requested.isdecimal() and int(requested) > 0):
        # Propagate an explicit budget to ITK consumers only. SlicerDMRI's
        # VTK tensor estimator does not use ITK's thread setting.
        env["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = requested
    cli_path = Path(cli_executable).resolve()
    extra_lib_dirs: list[str] = []

    cli_dir = cli_path.parent
    if cli_dir.is_dir():
        extra_lib_dirs.append(cli_dir.as_posix())

    qt_loadable_dir = cli_dir.parent / "qt-loadable-modules"
    if qt_loadable_dir.is_dir():
        extra_lib_dirs.append(qt_loadable_dir.as_posix())

    if sys.platform == "darwin":
        # macOS extension CLIs (UKFTractography for Slicer 5.12) name Slicer's
        # core ITK/VTK libraries by build-machine paths, and the launcher
        # exports no DYLD_LIBRARY_PATH; point dyld at the app's lib/Slicer-*.
        # C5: stop every ancestor walk at the .app/Contents folder, and use the
        # running Slicer's lib only for a CLI that belongs to that Slicer
        # (inside its root, or in a Slicer-managed slicer.org/Extensions-* folder).
        def bounded_parents(path: Path) -> list[Path]:
            result: list[Path] = []
            for parent in path.parents:
                result.append(parent)
                if parent.name == "Contents" and parent.parent.suffix == ".app":
                    break
            return result

        runtime_root = next((root for root in bounded_parents(Path(sys.executable).resolve())
                             if any(root.glob("lib/Slicer-*"))), None)
        belongs_to_slicer = (
            (runtime_root is not None and runtime_root in cli_path.parents)
            or any(parent.parent.name == "slicer.org" and parent.name.startswith("Extensions-")
                   for parent in cli_path.parents)
        )
        for root in (*bounded_parents(cli_path), *((runtime_root,) if belongs_to_slicer and runtime_root else ())):
            extra_lib_dirs.extend(path.as_posix() for path in sorted(root.glob("lib/Slicer-*")) if path.is_dir())

    library_var = "PATH" if os.name == "nt" or sys.platform == "win32" else (
        "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH")
    existing = [entry for entry in env.get(library_var, "").split(os.pathsep) if entry]
    merged: list[str] = []
    for entry in [*extra_lib_dirs, *existing]:
        if entry not in merged:
            merged.append(entry)
    env[library_var] = os.pathsep.join(merged)
    return env


@lru_cache(maxsize=1)
def _cli_search_dirs() -> tuple[Path, ...]:
    _require_slicer()
    slicer_home = _resolve_slicer_home()
    candidates: list[Path] = []
    patterns = [
        "lib/Slicer-*/cli-modules",
        "slicer.org/Extensions-*/*/lib/Slicer-*/cli-modules",
        # A stock macOS .app keeps its extensions in Contents/Extensions-<revision>.
        "Extensions-*/*/lib/Slicer-*/cli-modules",
    ]
    for pattern in patterns:
        for path in sorted(slicer_home.glob(pattern)):
            if path.is_dir():
                candidates.append(path.resolve())
    if sys.platform == "darwin":
        # C1: accept only extension builds made for this Slicer's own lib/Slicer-X.Y,
        # newest revision first; a 5.6 or 5.10 CLI must not serve a 5.12 runtime.
        user_extensions = Path.home() / "Library/Application Support/slicer.org"
        own_versions = {path.name for path in slicer_home.glob("lib/Slicer-*") if path.is_dir()}

        def revision(path: Path) -> int:
            digits = path.parents[3].name.rpartition("-")[2]
            return int(digits) if digits.isdecimal() else -1

        user_dirs = [path for path in user_extensions.glob("Extensions-*/*/lib/Slicer-*/cli-modules")
                     if path.is_dir() and path.parent.name in own_versions]
        for path in sorted(user_dirs, key=lambda item: (-revision(item), item.as_posix())):
            if path.resolve() not in candidates:
                candidates.append(path.resolve())
    # Extension Manager/development paths need not live under SLICER_HOME.
    # PythonQt exposes searchPaths as a property on some Slicer versions.
    try:
        factory = slicer.app.moduleManager().factoryManager()
        paths = getattr(factory, 'searchPaths', ())
        paths = paths() if callable(paths) else paths
        for value in paths:
            path = Path(str(value))
            if path.is_dir() and path.resolve() not in candidates:
                candidates.append(path.resolve())
    except (AttributeError, TypeError, RuntimeError):
        pass
    return tuple(candidates)


def resolve_cli_executable(module_name: str) -> str:
    _require_slicer()
    modules = getattr(slicer, "modules", None)
    module = getattr(modules, module_name, None) if modules is not None else None
    if module is not None:
        logic = module.cliModuleLogic()
        if logic is not None:
            getter = getattr(logic, "GetCLIExecutable", None)
            if callable(getter):
                cli_exe = getter()
                if cli_exe:
                    return cli_exe
        path = getattr(module, 'path', None)
        path = path() if callable(path) else path
        if path and Path(str(path)).is_file() and Path(str(path)).suffix.lower() not in {'.xml', '.so', '.dll', '.dylib'}:
            return str(path)

    target = module_name.lower()
    windows_executable_suffixes = {".exe", ".bat", ".cmd"}
    for cli_dir in _cli_search_dirs():
        for candidate in cli_dir.iterdir():
            candidate_name = candidate.name.lower()
            is_executable_name = candidate_name == target or (
                candidate.suffix.lower() in windows_executable_suffixes and candidate.stem.lower() == target
            )
            if candidate.is_file() and is_executable_name:
                return candidate.as_posix()

    raise RuntimeError(
        f"Required CLI module is not available: {module_name}. Searched directories: "
        + ", ".join(path.as_posix() for path in _cli_search_dirs())
    )


def _rewrite_nhdr_data_file(nhdr_path: Path, data_path: Path) -> None:
    lines = nhdr_path.read_text(encoding="utf-8").splitlines()
    updated = []
    replaced = False
    for line in lines:
        if line.startswith("data file:"):
            updated.append(f"data file: {data_path.resolve().as_posix()}")
            replaced = True
        else:
            updated.append(line)
    if not replaced:
        updated.append(f"data file: {data_path.resolve().as_posix()}")
    nhdr_path.write_text("\n".join(updated) + "\n", encoding="utf-8")


def input_source_cache_prefix(raw: RawInputBundle) -> str:
    """Cache-key prefix separating features of the raw input from preprocessed DWI.

    Raw inputs keep the historical key; a bundle resolved from 01_preproc gets
    its own ``preproc_`` subdirectory so alternating raw/preprocessed runs do
    not overwrite (thrash) one another's cached DTI features.
    """

    try:
        is_raw = Path(raw.input_dir).resolve() == (Path(raw.subject_root) / "input").resolve()
    except OSError:
        is_raw = True
    return "" if is_raw else "preproc_"


def ensure_diffusion_headers(raw: RawInputBundle, *, log=None) -> tuple[Path, Path]:
    """Convert once for LS/WLS consumers; scaled payloads belong to this cache."""
    from tempfile import TemporaryDirectory
    from anatomy.parcellation.conversion.nhdr_write import nhdr_write

    directory = raw.subject_root / 'cache/diffusion_inputs' / (input_source_cache_prefix(raw) + 'native')
    headers = (directory / 'DWI.nhdr', directory / 'Mask.nhdr')
    manifest = directory / 'manifest.json'
    signature = dict(kind='diffusion_headers', version=2, subject_id=raw.subject_id,
        implementation=_conversion_implementation(),
        sources={key: _cache_file_signature(getattr(raw, key + '_path')) for key in ('dwi', 'bval', 'bvec', 'mask')})
    with cache_lock(directory / '.lock'):
        signature['sources'] = {key: _cache_file_signature(getattr(raw, key + '_path'))
                                for key in ('dwi', 'bval', 'bvec', 'mask')}
        if _read_manifest(manifest) == signature:
            receipt = _read_manifest(directory / 'artifacts.json') or {}
            required = set(map(str, headers))
            allowed = required | {str(directory / name) for name in ('DWI.raw.gz', 'Mask.raw.gz')}
            if required <= set(receipt) <= allowed and _artifacts_current(directory, [Path(path) for path in receipt]):
                _emit(f'[Shared] Reusing DWI/mask headers: {directory}', log)
                return headers
        manifest.unlink(missing_ok=True)
        files = list(headers)
        with TemporaryDirectory(prefix='.headers-', dir=directory) as temporary:
            for source, bval, bvec, target in (
                    (raw.dwi_path, raw.bval_path, raw.bvec_path, headers[0]),
                    (raw.mask_path, None, None, headers[1])):
                staged = Path(temporary) / target.name
                nhdr_write(str(source), str(bval) if bval else None, str(bvec) if bvec else None, str(staged))
                scaled = staged.with_suffix('.raw.gz')
                if scaled.is_file():
                    payload = directory / scaled.name
                    shutil.copy2(scaled, payload)
                    files.append(payload)
                else:
                    payload = source
                _rewrite_nhdr_data_file(staged, payload)
                staged.replace(target)
        current = {key: _cache_file_signature(getattr(raw, key + '_path')) for key in ('dwi', 'bval', 'bvec', 'mask')}
        if current != signature['sources']:
            raise RuntimeError('Inputs changed during shared header conversion')
        _save_artifacts(directory, files)
        _write_manifest(manifest, signature)
        _emit(f'[Shared] Prepared DWI/mask headers once: {directory}', log)
        return headers


def _build_native_bundle(raw: RawInputBundle, fit_method: str) -> NativeDTIBundle:
    fit_key = fit_method.lower()
    bundle_dir = raw.subject_root / "cache" / "native_dti" / f"{input_source_cache_prefix(raw)}{fit_key}"
    subject = raw.subject_id
    return NativeDTIBundle(
        subject_root=raw.subject_root,
        subject_id=subject,
        fit_method=fit_key,
        bundle_dir=bundle_dir,
        dwi_nhdr_path=bundle_dir / "DWI.nhdr",
        mask_nhdr_path=bundle_dir / "Mask.nhdr",
        dti_nrrd_path=bundle_dir / "DTI.nrrd",
        b0_nrrd_path=bundle_dir / "b0.nrrd",
        fa_nrrd_path=bundle_dir / "DTI_FA.nrrd",
        trace_nrrd_path=bundle_dir / "DTI_Trace.nrrd",
        mean_diffusivity_nrrd_path=bundle_dir / "DTI_MeanDiffusivity.nrrd",
        min_eig_nrrd_path=bundle_dir / "DTI_MinEigenvalue.nrrd",
        mid_eig_nrrd_path=bundle_dir / "DTI_MidEigenvalue.nrrd",
        max_eig_nrrd_path=bundle_dir / "DTI_MaxEigenvalue.nrrd",
        ddseg_md_nrrd_path=bundle_dir / "DTI_MD.nrrd",
        ddseg_e1_nrrd_path=bundle_dir / "DTI_E1.nrrd",
        ddseg_e2_nrrd_path=bundle_dir / "DTI_E2.nrrd",
        ddseg_e3_nrrd_path=bundle_dir / "DTI_E3.nrrd",
        fa_nii_path=bundle_dir / f"{subject}-dti-FractionalAnisotropy.nii.gz",
        trace_nii_path=bundle_dir / f"{subject}-dti-Trace.nii.gz",
        mean_diffusivity_nii_path=bundle_dir / f"{subject}-dti-MeanDiffusivity.nii.gz",
        min_eig_nii_path=bundle_dir / f"{subject}-dti-MinEigenvalue.nii.gz",
        mid_eig_nii_path=bundle_dir / f"{subject}-dti-MidEigenvalue.nii.gz",
        max_eig_nii_path=bundle_dir / f"{subject}-dti-MaxEigenvalue.nii.gz",
        manifest_path=bundle_dir / "manifest.json",
    )


def _native_manifest(raw: RawInputBundle, bundle: NativeDTIBundle) -> dict[str, Any]:
    return {
        "kind": "native_dti_bundle",
        "version": 3,
        "subject_id": raw.subject_id,
        "fit_method": bundle.fit_method,
        "conversion_implementation": _conversion_implementation(),
        "preparation_implementation": _artifact_record(Path(__file__))['sha256'],
        "source_files": {
            "dwi": _cache_file_signature(raw.dwi_path),
            "bval": _cache_file_signature(raw.bval_path),
            "bvec": _cache_file_signature(raw.bvec_path),
            "mask": _cache_file_signature(raw.mask_path),
        },
    }


def _native_outputs_complete(bundle: NativeDTIBundle) -> bool:
    required = [
        bundle.dwi_nhdr_path,
        bundle.mask_nhdr_path,
        bundle.dti_nrrd_path,
        bundle.b0_nrrd_path,
        bundle.fa_nrrd_path,
        bundle.trace_nrrd_path,
        bundle.mean_diffusivity_nrrd_path,
        bundle.min_eig_nrrd_path,
        bundle.mid_eig_nrrd_path,
        bundle.max_eig_nrrd_path,
        bundle.ddseg_md_nrrd_path,
        bundle.ddseg_e1_nrrd_path,
        bundle.ddseg_e2_nrrd_path,
        bundle.ddseg_e3_nrrd_path,
        bundle.fa_nii_path,
        bundle.trace_nii_path,
        bundle.mean_diffusivity_nii_path,
        bundle.min_eig_nii_path,
        bundle.mid_eig_nii_path,
        bundle.max_eig_nii_path,
    ]
    return _artifacts_current(bundle.bundle_dir, required)


def _convert_nrrd_to_nifti(in_path: Path, out_path: Path) -> None:
    from anatomy.parcellation.conversion.nifti_write import nifti_write

    prefix = out_path.as_posix().replace(".nii.gz", "")
    nifti_write(in_path.as_posix(), prefix)


@scheduled_stage('native_dti')
def ensure_native_dti_bundle(
    raw: RawInputBundle,
    fit_method: str = "wls",
    reuse: bool = True,
    log: Optional[Callable[[str], None]] = None,
) -> NativeDTIBundle:
    """Create or reuse native-space DTI features under cache/native_dti."""
    if fit_method.lower() not in ('ls', 'wls'):
        raise ValueError(f'Unsupported fit method: {fit_method}')
    bundle = _build_native_bundle(raw, fit_method)
    with cache_lock(bundle.bundle_dir / '.lock'):
        return _ensure_native_dti_bundle(raw, fit_method, reuse, log)


def _ensure_native_dti_bundle(raw, fit_method, reuse, log):
    fit_key = fit_method.lower()
    if fit_key not in {"ls", "wls"}:
        raise ValueError(f"Unsupported fit method: {fit_method}")

    headers = ensure_diffusion_headers(raw, log=log)
    bundle = _build_native_bundle(raw, fit_key)
    expected_manifest = _native_manifest(raw, bundle)
    actual_manifest = _read_manifest(bundle.manifest_path)
    if reuse and actual_manifest == expected_manifest and _native_outputs_complete(bundle):
        from shared.cohort_schedule import mark_reused
        mark_reused('native_dti')
        _emit(f"[Shared] Reusing native DTI bundle: {bundle.bundle_dir}", log)
        return bundle

    _require_slicer()
    _emit(f"[Shared] Building native DTI bundle ({fit_key}) in {bundle.bundle_dir}", log)
    started = time.monotonic()
    bundle.bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle.manifest_path.unlink(missing_ok=True)
    shutil.copy2(headers[0], bundle.dwi_nhdr_path)
    shutil.copy2(headers[1], bundle.mask_nhdr_path)

    dti_cli = _get_cli_executable("dwitodtiestimation")
    scalar_cli = _get_cli_executable("diffusiontensorscalarmeasurements")
    fit_enum = "WLS" if fit_key == "wls" else "LS"

    _run_cli(
        [
            dti_cli,
            "--enumeration",
            fit_enum,
            bundle.dwi_nhdr_path.as_posix(),
            bundle.dti_nrrd_path.as_posix(),
            bundle.b0_nrrd_path.as_posix(),
        ],
        log,
    )

    scalar_outputs = [
        ("FractionalAnisotropy", bundle.fa_nrrd_path),
        ("Trace", bundle.trace_nrrd_path),
        ("MeanDiffusivity", bundle.mean_diffusivity_nrrd_path),
        ("MinEigenvalue", bundle.min_eig_nrrd_path),
        ("MidEigenvalue", bundle.mid_eig_nrrd_path),
        ("MaxEigenvalue", bundle.max_eig_nrrd_path),
    ]
    for enum_name, output_path in scalar_outputs:
        _run_cli(
            [
                scalar_cli,
                "--enumeration",
                enum_name,
                bundle.dti_nrrd_path.as_posix(),
                output_path.as_posix(),
            ],
            log,
        )

    shutil.copy2(bundle.trace_nrrd_path, bundle.ddseg_md_nrrd_path)
    shutil.copy2(bundle.max_eig_nrrd_path, bundle.ddseg_e1_nrrd_path)
    shutil.copy2(bundle.mid_eig_nrrd_path, bundle.ddseg_e2_nrrd_path)
    shutil.copy2(bundle.min_eig_nrrd_path, bundle.ddseg_e3_nrrd_path)

    _convert_nrrd_to_nifti(bundle.fa_nrrd_path, bundle.fa_nii_path)
    _convert_nrrd_to_nifti(bundle.trace_nrrd_path, bundle.trace_nii_path)
    _convert_nrrd_to_nifti(bundle.mean_diffusivity_nrrd_path, bundle.mean_diffusivity_nii_path)
    _convert_nrrd_to_nifti(bundle.min_eig_nrrd_path, bundle.min_eig_nii_path)
    _convert_nrrd_to_nifti(bundle.mid_eig_nrrd_path, bundle.mid_eig_nii_path)
    _convert_nrrd_to_nifti(bundle.max_eig_nrrd_path, bundle.max_eig_nii_path)

    if expected_manifest != _native_manifest(raw, bundle):
        raise RuntimeError('Inputs changed during shared native DTI preparation')
    files = [getattr(bundle, name) for name in bundle.__dataclass_fields__
             if name.endswith('_path') and name != 'manifest_path']
    _save_artifacts(bundle.bundle_dir, files)
    _write_manifest(bundle.manifest_path, expected_manifest)
    _write_manifest(bundle.bundle_dir / 'native_dti_timing.json', {
        'kind': 'native_dti_timing', 'status': 'completed', 'subject_id': raw.subject_id,
        'runtime_seconds': time.monotonic() - started, 'fit_method': fit_key,
        'source_dwi': str(raw.dwi_path), 'cache_reused': False,
        'itk_default_threads': build_cli_environment(dti_cli).get('ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'),
        'thread_control_scope': 'ITK environment only; not a thread limit on the VTK tensor estimator',
        'timing_scope': 'Fresh native tensor fitting, six scalar measurements, compatibility copies and NIfTI conversion; excludes registration and later ROI statistics',
    })
    return bundle


def _atlas_key(atlas_path: Path) -> str:
    stem = atlas_path.name
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    else:
        stem = atlas_path.stem
    return stem.replace(" ", "_")


def _build_registered_bundle(raw: RawInputBundle, fit_method: str, atlas_path: Path,
                             scalar_format: str = 'nifti') -> RegisteredDTIBundle:
    fit_key = fit_method.lower()
    bundle_dir = (
        raw.subject_root / "cache" / "registered_dti" / f"{input_source_cache_prefix(raw)}{_atlas_key(atlas_path)}__{fit_key}"
    )
    if scalar_format == 'nrrd':
        bundle_dir = bundle_dir.with_name(bundle_dir.name + '__nrrd')
    subject = raw.subject_id
    return RegisteredDTIBundle(
        subject_root=raw.subject_root,
        subject_id=subject,
        fit_method=fit_key,
        atlas_path=atlas_path,
        bundle_dir=bundle_dir,
        fa_reg_path=bundle_dir / f"{subject}-dti-FractionalAnisotropy-Reg.nii.gz",
        trace_reg_path=bundle_dir / f"{subject}-dti-Trace-Reg.nii.gz",
        mean_diffusivity_reg_path=bundle_dir / f"{subject}-dti-MeanDiffusivity-Reg.nii.gz",
        min_eig_reg_path=bundle_dir / f"{subject}-dti-MinEigenvalue-Reg.nii.gz",
        mid_eig_reg_path=bundle_dir / f"{subject}-dti-MidEigenvalue-Reg.nii.gz",
        max_eig_reg_path=bundle_dir / f"{subject}-dti-MaxEigenvalue-Reg.nii.gz",
        mask_reg_path=bundle_dir / f"{subject}-mask-Reg.nii.gz",
        transform_path=bundle_dir / f"{subject}-b0ToAtlasT2.tfm",
        inverse_transform_path=bundle_dir / f"{subject}-b0ToAtlasT2_Inverse.h5",
        manifest_path=bundle_dir / "manifest.json",
        scalar_format=scalar_format,
    )


def _registered_manifest(raw: RawInputBundle, native: NativeDTIBundle, registered: RegisteredDTIBundle) -> dict[str, Any]:
    return {
        "kind": "registered_dti_bundle",
        "version": 2,
        "subject_id": raw.subject_id,
        "fit_method": registered.fit_method,
        "scalar_format": registered.scalar_format,
        "implementation": _artifact_record(Path(__file__))['sha256'],
        "atlas": _cache_file_signature(registered.atlas_path),
        "native_bundle": {
            "dir": str(native.bundle_dir.resolve()),
            "manifest": _read_manifest(native.manifest_path),
            "artifacts": _read_manifest(native.bundle_dir / 'artifacts.json'),
        },
    }


def _registered_outputs_complete(bundle: RegisteredDTIBundle) -> bool:
    required = [
        bundle.fa_reg_path,
        bundle.trace_reg_path,
        bundle.mean_diffusivity_reg_path,
        bundle.min_eig_reg_path,
        bundle.mid_eig_reg_path,
        bundle.max_eig_reg_path,
        bundle.mask_reg_path,
        bundle.transform_path,
        bundle.inverse_transform_path,
    ]
    return _artifacts_current(bundle.bundle_dir, required)


def _write_inverse_transform(forward_path: Path, inverse_path: Path) -> None:
    if SIMPLEITK_IMPORT_ERROR is not None or sitk is None:
        return
    transform = sitk.ReadTransform(str(forward_path))
    inverse = transform.GetInverse()
    sitk.WriteTransform(inverse, str(inverse_path))


@scheduled_stage('registered_dti')
def ensure_registered_dti_bundle(
    raw: RawInputBundle,
    atlas_t2: Path,
    fit_method: str = "wls",
    reuse: bool = True,
    log: Optional[Callable[[str], None]] = None,
    *,
    scalar_format: str = 'nifti',
) -> RegisteredDTIBundle:
    """Create or reuse registered DTI features under cache/registered_dti."""
    if scalar_format not in ('nifti', 'nrrd'):
        raise ValueError('Registered scalar format must be nifti or nrrd')
    if fit_method.lower() not in ('ls', 'wls'):
        raise ValueError(f'Unsupported fit method: {fit_method}')
    atlas_t2 = atlas_t2.expanduser().resolve()
    bundle = _build_registered_bundle(raw, fit_method, atlas_t2, scalar_format)
    with cache_lock(bundle.bundle_dir / '.lock'):
        return _ensure_registered_dti_bundle(raw, atlas_t2, fit_method, reuse, log, scalar_format)


def _ensure_registered_dti_bundle(raw, atlas_t2, fit_method, reuse, log, scalar_format):
    atlas_t2 = atlas_t2.expanduser().resolve()
    if not atlas_t2.is_file():
        raise FileNotFoundError(f"Atlas T2 not found: {atlas_t2}")

    native = ensure_native_dti_bundle(raw, fit_method=fit_method, reuse=True, log=log)
    bundle = _build_registered_bundle(raw, fit_method, atlas_t2, scalar_format)
    expected_manifest = _registered_manifest(raw, native, bundle)
    actual_manifest = _read_manifest(bundle.manifest_path)
    if reuse and actual_manifest == expected_manifest and _registered_outputs_complete(bundle):
        _emit(f"[Shared] Reusing registered DTI bundle: {bundle.bundle_dir}", log)
        from shared.cohort_schedule import mark_reused
        mark_reused('registered_dti')
        if not bundle.inverse_transform_path.exists():
            _write_inverse_transform(bundle.transform_path, bundle.inverse_transform_path)
        return bundle

    _require_slicer()
    _emit(f"[Shared] Building registered DTI bundle ({bundle.fit_method}) in {bundle.bundle_dir}", log)
    bundle.bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle.manifest_path.unlink(missing_ok=True)

    brainsfit_cli = _get_cli_executable("brainsfit")
    resample_cli = _get_cli_executable("resamplescalarvectordwivolume")

    _run_cli(
        [
            brainsfit_cli,
            "--fixedVolume",
            atlas_t2.as_posix(),
            "--movingVolume",
            native.b0_nrrd_path.as_posix(),
            "--linearTransform",
            bundle.transform_path.as_posix(),
            "--useRigid",
            "--useAffine",
        ],
        log,
    )

    resample_jobs = [
        (native.fa_nii_path, bundle.fa_reg_path, "linear"),
        (native.trace_nii_path, bundle.trace_reg_path, "linear"),
        (native.mean_diffusivity_nii_path, bundle.mean_diffusivity_reg_path, "linear"),
        (native.min_eig_nii_path, bundle.min_eig_reg_path, "linear"),
        (native.mid_eig_nii_path, bundle.mid_eig_reg_path, "linear"),
        (native.max_eig_nii_path, bundle.max_eig_reg_path, "linear"),
        (raw.mask_path, bundle.mask_reg_path, "nn"),
    ]
    if scalar_format == 'nrrd':
        originals = dict(zip((native.fa_nii_path, native.trace_nii_path, native.mean_diffusivity_nii_path,
                              native.min_eig_nii_path, native.mid_eig_nii_path, native.max_eig_nii_path),
                             (native.fa_nrrd_path, native.trace_nrrd_path, native.mean_diffusivity_nrrd_path,
                              native.min_eig_nrrd_path, native.mid_eig_nrrd_path, native.max_eig_nrrd_path)))
        resample_jobs = [(originals.get(source, source), target, mode) for source, target, mode in resample_jobs]
    for input_path, output_path, interpolation in resample_jobs:
        _run_cli(
            [
                resample_cli,
                "-i",
                interpolation,
                input_path.as_posix(),
                "--Reference",
                atlas_t2.as_posix(),
                "--transformationFile",
                bundle.transform_path.as_posix(),
                output_path.as_posix(),
            ],
            log,
        )

    _write_inverse_transform(bundle.transform_path, bundle.inverse_transform_path)
    if expected_manifest != _registered_manifest(raw, native, bundle):
        raise RuntimeError('Inputs changed during shared registration')
    files = [getattr(bundle, name) for name in bundle.__dataclass_fields__
             if name.endswith('_path') and name not in ('atlas_path', 'manifest_path')]
    _save_artifacts(bundle.bundle_dir, files)
    _write_manifest(bundle.manifest_path, expected_manifest)
    return bundle
