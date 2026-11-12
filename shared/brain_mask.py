"""Selectable native-DWI masking, with external runtimes and explicit provenance."""

from __future__ import annotations

from shared.cohort_schedule import scheduled_stage

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import gzip
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Any, Callable, Optional

from shared.external_process import python_tool_environment, text_subprocess_kwargs


BRAIN_MASK_BACKENDS = ("pnl_cnn", "provided")
# W5 (D2): these mask backends are gone. Old manifests, receipts and saved settings may still name them (SC-2).
REMOVED_BRAIN_MASK_BACKENDS = ("synthstrip", "mrtrix", "dipy")


def unsupported_backend_message(backend: str) -> str:
    """SC-2: the text for a brain mask backend that is removed or unknown; it names the two that remain."""
    kind = "was removed" if backend in REMOVED_BRAIN_MASK_BACKENDS else "is not known"
    return f"Brain mask backend {backend!r} {kind}. Use pnl_cnn (PNL CNN) or provided (a supplied mask)."
BRAIN_MASK_DEVICES = ("auto", "cpu", "cuda")
# T2 design SC-1a: route of the pnl_cnn backend. auto -> managed on Linux (WSL included), native elsewhere.
PNL_MASK_ROUTES = ("auto", "managed", "native")
PNL_ROUTE_NAMES = {"managed": "managed-ants", "native": "native-sitk"}
PNL_ROUTE_LABELS = {"auto": "auto", "managed": "managed", "native": "native (experimental)"}
# SC-1b: the reason text of the experimental status of the native route.
PNL_NATIVE_EXPERIMENTAL_REASON = "did not meet the M-A parity rule on 2 of 24 cases (T2 design §3)"
# SC-1e: the native registration always runs with one SimpleITK thread (the T3 configuration).
PNL_REGISTRATION_THREADS = 1
# T2 design A1/A7: the pnl_cnn backend runs this module in Slicer's Python, with the vendored CNN code.
PNL_NATIVE_SCRIPT = Path(__file__).with_name("pnl_mask_native.py")
PNL_VENDOR_DIR = Path(__file__).with_name("pnl_cnn")
PNL_TEMPLATE_NAME = "IITmean_b0_256.nii.gz"
PNL_NATIVE_PYTHON_MISSING = "PythonSlicer (Slicer's Python with SlicerPyTorch runs the native PNL mask)"


@dataclass(frozen=True)
class BrainMaskConfig:
    backend: str = "pnl_cnn"
    brain_mask_device: str = "auto"
    pnl_mask_python: Optional[Path] = None
    pnl_mask_script: Optional[Path] = None
    pnl_mask_model_dir: Optional[Path] = None
    pnl_mask_ants_bin_dir: Optional[Path] = None
    pnl_mask_route: str = "auto"
    b0_threshold: float = 50.0
    threads: int = 4
    model_assets_dir: Optional[Path] = None


def brain_mask_config_from_options(options) -> BrainMaskConfig:
    return BrainMaskConfig(
        backend=getattr(options, "brain_mask_backend", "pnl_cnn"),
        brain_mask_device=getattr(options, "brain_mask_device", "auto"),
        pnl_mask_route=getattr(options, "pnl_mask_route", None) or "auto",
        **{name: getattr(options, name, None) for name in (
            "pnl_mask_python", "pnl_mask_script", "pnl_mask_model_dir", "pnl_mask_ants_bin_dir", "model_assets_dir")},
        b0_threshold=getattr(options, "b0_threshold", 50.0),
    )


def _path(value, env_key=None, *, resolve=True):
    value = value or (os.environ.get(env_key) if env_key else None)
    if not value:
        return None
    path = Path(value).expanduser().absolute()
    return path.resolve() if resolve else path


def _manifest_path_for_mask(mask_path: Path) -> Path:
    name = mask_path.name.removesuffix(".gz").removesuffix(".nii")
    return mask_path.with_name(name + ".manifest.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": _sha256(path)}


_pnl_native_python_problem = ""  # B7: why the last `_pnl_native_python()` call found nothing


def _pnl_native_python() -> Optional[Path]:
    """PythonSlicer for the A1 subprocess, found as for ``teem_dti.py``; None outside a Slicer Python."""
    global _pnl_native_python_problem
    _pnl_native_python_problem = ""
    try:
        # Imported here: shared.pipeline imports this module.
        from shared.pipeline import resolve_pythonslicer_executable
        return Path(resolve_pythonslicer_executable())
    except (ImportError, RuntimeError) as exc:
        _pnl_native_python_problem = str(exc)
        return None


def _pnl_native_python_checked() -> Optional[Path]:
    """``_pnl_native_python()`` with a clean problem text, so a replaced resolver never shows an old message."""
    global _pnl_native_python_problem
    _pnl_native_python_problem = ""
    return _pnl_native_python()


def pnl_child_threads() -> int:
    """Thread count the native PNL child uses (A6): ``SLICERDSPACE_STAGE_CPUS``, else affinity, else CPU count."""
    from shared.pnl_mask_native import default_threads
    return default_threads()


def pnl_native_resources() -> list[Path]:
    """A7 cache resources: ``pnl_mask_native.py`` and every vendored file in UPSTREAM.json (O5: with the JSON)."""
    record = json.loads((PNL_VENDOR_DIR / "UPSTREAM.json").read_text(encoding="utf-8"))
    return [PNL_NATIVE_SCRIPT, PNL_VENDOR_DIR / "UPSTREAM.json",
            *(PNL_VENDOR_DIR / relative for relative in sorted(record["files"]))]


def resolve_pnl_mask_route(route: str | None = "auto", system: str | None = None) -> str:
    """SC-1a: ``managed`` or ``native`` for the setting ``auto`` | ``managed`` | ``native``.

    ``auto`` is ``managed`` on Linux and ``native`` on macOS and Windows. On Windows the mask runs in Windows Slicer's
    Python (only the MRtrix and FSL tools go through WSL), and the managed route is refused there; the managed ``pnl_cnn``
    runtime has no macOS or Windows build, so native is the only PNL route on those systems.
    """
    route = route or "auto"
    if route not in PNL_MASK_ROUTES:
        raise ValueError("pnl_mask_route must be one of: " + ", ".join(PNL_MASK_ROUTES))
    if route != "auto":
        return route
    # S1 (SC-1 fix round): Windows resolves auto to native (spec SC-1a); managed only on Linux.
    return "managed" if (system or platform.system()) == "Linux" else "native"


def pnl_route_record(route: str) -> dict[str, str]:
    """SC-1b: ``route`` and, for the native route, the experimental status with its reason (receipt, run record)."""
    record = {"route": PNL_ROUTE_NAMES[route]}
    if route == "native":
        record.update(status="experimental", reason=PNL_NATIVE_EXPERIMENTAL_REASON)
    return record


def _generated_route(mask: Path) -> dict[str, str] | None:
    """The route of the receipt next to `mask`, only when that receipt describes these very bytes (F2b).

    A pnl_cnn receipt gives its ``pnl_route``. A receipt of a removed generating backend (SC1-CARRY-1, W5 SC-2) still
    reads and gives that backend as the route, for example ``{"route": "synthstrip"}``.
    """
    record_path = _manifest_path_for_mask(mask)
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict):
        return None
    recorded = (record.get("output") or {}).get("sha256") if isinstance(record.get("output"), dict) else None
    if not recorded or recorded != _sha256(mask):
        return None
    backend = record.get("backend")
    route = record.get("pnl_route")
    if backend == "pnl_cnn":
        if isinstance(route, dict) and route.get("route"):
            return {key: str(value) for key, value in route.items()}
        return None
    if backend in REMOVED_BRAIN_MASK_BACKENDS:
        return {"route": str(backend)}
    return None


def mask_provenance(mask_path: Path, *, preprocessing_manifest: dict | None = None) -> dict[str, str]:
    """F2 / F2b: route, status and reason of the mask that a run consumes.

    (a) The receipt next to the mask gives its ``pnl_route`` if the backend is pnl_cnn and ``pnl_route`` is present; a
    receipt of another generating backend gives that backend's name as the route. Both need the receipt's
    ``output.sha256`` to equal the SHA-256 of the mask now (a stale receipt never labels a later mask); this
    holds also for a reused cache entry. (b) Otherwise, for a preprocessing final mask, a preprocessing manifest whose
    ``mask_generation`` has backend ``provided`` and a ``source`` gives the provenance of that source (same rule), when
    the final mask is a byte copy of it. (c) Otherwise a mask file with no receipt at all is ``supplied``; a receipt that
    does not match, or has no usable route, is ``unknown``; no mask file is ``unknown``.
    """
    mask = Path(mask_path)
    if not mask.is_file():
        return {"route": "unknown"}
    route = _generated_route(mask)
    if route:
        return route
    generation = (preprocessing_manifest or {}).get("mask_generation")
    if isinstance(generation, dict) and generation.get("backend") == "provided" and generation.get("source"):
        source = Path(str(generation["source"]))
        if source.is_file() and _sha256(source) == _sha256(mask):
            return mask_provenance(source)
    return {"route": "unknown" if _manifest_path_for_mask(mask).is_file() else "supplied"}


def _inspect_pnl_managed(config: BrainMaskConfig, system: str | None, missing: list, tools: dict, resources: list) -> None:
    """The managed route: the pre-W-A tool lookup of 70fb635^1 (conda Python, ``pnl-brain-mask`` script, ANTs)."""
    resources.append(Path(__file__).with_name("pnl_mask_runner.py"))
    # Resolving a venv's Python symlink would select the base interpreter
    # and lose that environment's installed PNLBrainMask-Torch package.
    python = _path(config.pnl_mask_python, "SLICERDSPACE_PNL_MASK_PYTHON", resolve=False)
    script = _path(config.pnl_mask_script, "SLICERDSPACE_PNL_MASK_SCRIPT")
    if script is None and python is not None:
        script = python.parent / "pnl-brain-mask"
    ants = _path(config.pnl_mask_ants_bin_dir, "ANTSPATH")
    for name, path in (("pnl_mask_python", python), ("pnl_mask_script", script)):
        if path is None or not path.is_file():
            missing.append(name)
        else:
            tools[name] = str(path)
    if python is not None and python.is_file() and not os.access(python, os.X_OK):
        missing.append("executable pnl_mask_python")
    if script and script.is_file():
        resources.append(script)
    search = os.pathsep.join(str(p) for p in (ants, python.parent if python else None) if p)
    search += os.pathsep + os.environ.get("PATH", "")
    for command in ("antsRegistrationSyNQuick.sh", "antsApplyTransforms", "ImageMath"):
        found = shutil.which(command, path=search)
        if found:
            tools[command] = str(Path(found).resolve())
        else:
            missing.append(command)
    tools["ants_bin_dir"] = str(ants) if ants else ""
    if (system or platform.system()) == "Windows" or os.name == "nt":
        missing.append("PNL CNN requires a POSIX external runtime (use Linux/WSL pipeline execution)")


def _pnl_settings(route: str) -> dict[str, Any]:
    """The route-dependent part of the cache signature (SC-1f: ``pnl_route`` differs per route)."""
    if route == "managed":
        return {"pnl_framework": "pytorch", "pnl_route": PNL_ROUTE_NAMES["managed"], "pnl_package": "pnl-brain-mask",
                "pnl_inverse_interpolation": "ANTs default linear"}
    return {"pnl_framework": "pytorch", "pnl_route": PNL_ROUTE_NAMES["native"],
            "pnl_package": "vendored PNLBrainMask-Torch 8d201a5 (shared/pnl_cnn)",
            "pnl_registration": "SimpleITK rigid, ANTs rigid-stage parameters",
            "pnl_registration_threads": PNL_REGISTRATION_THREADS,
            "pnl_inverse_interpolation": "SimpleITK linear, float32"}


def inspect_brain_mask_backend(config: BrainMaskConfig | None = None, *, system: str | None = None) -> dict[str, Any]:
    """Inspect files/tools only; no framework import, model download or inference."""
    config = config or BrainMaskConfig()
    if config.backend not in BRAIN_MASK_BACKENDS:
        raise ValueError(unsupported_backend_message(config.backend))
    if config.brain_mask_device not in BRAIN_MASK_DEVICES:
        raise ValueError("brain_mask_device must be one of: " + ", ".join(BRAIN_MASK_DEVICES))
    if config.pnl_mask_route not in PNL_MASK_ROUTES:
        raise ValueError("pnl_mask_route must be one of: " + ", ".join(PNL_MASK_ROUTES))
    if not math.isfinite(config.b0_threshold) or config.b0_threshold < 0 or config.threads < 1:
        raise ValueError("Mask b0 threshold must be finite/nonnegative and threads must be positive")
    missing = []
    tools, resources = {}, []
    threads = config.threads
    route = None
    if config.backend == "pnl_cnn":
        route = resolve_pnl_mask_route(config.pnl_mask_route, system)
        if route == "managed":
            # SC-1a: the conda runner of 70fb635^1 (ANTs 2.3.0); it uses config.threads as before W-A.
            _inspect_pnl_managed(config, system, missing, tools, resources)
        else:
            # A-O4: the child ignores config.threads; the signature records the count it will use.
            threads = pnl_child_threads()
            # A7 (T2 design v4): Slicer's Python runs shared/pnl_mask_native.py (A1) with the vendored CNN code.
            # The conda runner settings (pnl_mask_python, pnl_mask_script, pnl_mask_ants_bin_dir) are not read.
            python = _pnl_native_python_checked()
            if python is None or not python.is_file() or not os.access(python, os.X_OK):
                # B7 (A-C1 2): when the resolver found nothing, its message (with "Set SLICER_HOME") reaches the user.
                reason = _pnl_native_python_problem if python is None else ""
                missing.append(PNL_NATIVE_PYTHON_MISSING + (": " + reason if reason else ""))
            else:
                tools["pythonslicer"] = str(python)
            try:
                vendored = pnl_native_resources()
            except (OSError, ValueError, KeyError):
                vendored = []
                missing.append(f"PNL native file {PNL_VENDOR_DIR / 'UPSTREAM.json'}")
            for file in vendored:
                if file.is_file():
                    resources.append(file)
                else:
                    missing.append(f"PNL native file {file}")
        models = _path(config.pnl_mask_model_dir, "SLICERDSPACE_PNL_MASK_MODEL_DIR")
        if models is None:
            from shared.optional_assets import verified_resource_directory
            models = verified_resource_directory("pnl-cnn", asset_root=config.model_assets_dir)
        if models is None or not models.is_dir():
            missing.append("pnl_mask_model_dir")
        else:
            tools["model_dir"] = str(models)
            for name in ("weights.json", PNL_TEMPLATE_NAME):
                file = models / name
                if not file.is_file():
                    missing.append(name)
                else:
                    resources.append(file)
            for view in ("sagittal", "coronal", "axial"):
                weights = models / (view + ".pt")
                if not weights.is_file():
                    missing.append(view + ".pt (convert/download PNLBrainMask-Torch weights; legacy HDF5 is not a Torch runtime)")
                else:
                    resources.append(weights)
    return {"backend": config.backend, "available": not missing, "missing_required": missing,
            "pnl_route_resolved": route,
            "tool_paths": tools, "resources": [str(p) for p in resources],
            "settings": {"b0_threshold": config.b0_threshold, "threads": threads,
                         "pnl_percentile": 99, "pnl_cleanup": "official_fill_holes_no_optional_filter",
                         "device": config.brain_mask_device if config.backend == "pnl_cnn" else "cpu",
                         **(_pnl_settings(route) if config.backend == "pnl_cnn" else {})},
            "note": "File discovery is not runtime/model compatibility or anatomical QC"}


def brain_mask_cache_inspection(inspection: dict[str, Any]) -> dict[str, Any]:
    """B12: the inspection as the cache signature and the preprocessing cache key hold it.

    The native PNL route leaves the thread count out (M-A2, 2026-11-07: 24 cases, 0 differing voxels between 1 and 8
    inference threads with the registration pinned to 1 thread); the receipt still records ``child_threads``. The
    managed route keeps its threads (ANTs was not measured). The device stays in the key.
    """
    if inspection.get("backend") != "pnl_cnn" or inspection.get("pnl_route_resolved") != "native":
        return inspection
    settings = {key: value for key, value in inspection["settings"].items() if key != "threads"}
    return dict(inspection, settings=settings)


def validate_brain_mask(dwi_path: Path, mask_path: Path) -> dict[str, Any]:
    import nibabel as nib
    import numpy as np

    dwi, mask = nib.load(str(dwi_path)), nib.load(str(mask_path))
    data = np.asarray(mask.dataobj)
    if data.ndim != 3 or data.shape != dwi.shape[:3] or not np.allclose(mask.affine, dwi.affine, rtol=0, atol=1e-4):
        raise ValueError("Brain mask must match the DWI's native 3D voxel grid and world affine")
    if not np.isfinite(data).all() or not np.all((data == 0) | (data == 1)) or not np.any(data):
        raise ValueError("Brain mask must be finite, binary and nonempty")
    fraction = float(np.count_nonzero(data) / data.size)
    warnings = ["Inspect brain boundaries, cerebellum and brainstem before quantitative use"]
    if fraction < 0.01 or fraction > 0.9:
        warnings.append("Unusual mask/FOV fraction; requires anatomical review")
    return {"geometry_pass": True, "binary": True, "finite": True,
            "brain_voxels": int(np.count_nonzero(data)), "mask_fov_fraction": fraction,
            "human_review": "pending", "warnings": warnings}


def _save_nifti_fast(image_obj, output_path: Path) -> None:
    import nibabel as nib

    out_str = str(output_path)
    if out_str.endswith(".nii.gz"):
        with gzip.GzipFile(filename=out_str, mode="wb", compresslevel=1) as gz:
            holder = nib.FileHolder(fileobj=gz)
            image_obj.to_file_map({"image": holder, "header": holder})
    else:
        nib.save(image_obj, out_str)


def _accumulate_b0_single_pass_gzip(image, dwi_path: Path, indices) -> "Optional[Any]":
    import numpy as np

    if not str(dwi_path).endswith(".nii.gz"):
        return None
    offset = int(image.header.get_data_offset())
    if offset < 352 or len(image.shape) != 4:
        return None
    disk_dtype = np.dtype(image.header.get_data_dtype())
    if disk_dtype.kind not in ("i", "u", "f") or disk_dtype.itemsize < 1:
        return None
    shape3d = tuple(int(dim) for dim in image.shape[:3])
    vol_elems = shape3d[0] * shape3d[1] * shape3d[2]
    vol_bytes = vol_elems * disk_dtype.itemsize
    slope, inter = image.header.get_slope_inter()
    use_scale = (
        slope is not None and np.isfinite(slope) and slope != 0.0
        and (float(slope) != 1.0 or (inter is not None and np.isfinite(inter) and float(inter) != 0.0))
    )
    slope_f = float(slope) if use_scale else 1.0
    inter_f = float(inter) if (use_scale and inter is not None and np.isfinite(inter)) else 0.0

    wanted = {int(idx) for idx in indices}
    max_idx = max(wanted)
    total = np.zeros(shape3d, dtype=np.float64)
    skip_chunk = 4 * 1024 * 1024

    def _read_exact(stream, nbytes: int) -> bytes:
        chunks = []
        rem = nbytes
        while rem > 0:
            part = stream.read(min(rem, skip_chunk))
            if not part:
                raise EOFError("Truncated gzipped NIfTI payload while reading b0 volumes")
            chunks.append(part)
            rem -= len(part)
        return chunks[0] if len(chunks) == 1 else b"".join(chunks)

    with gzip.open(dwi_path, "rb") as stream:
        if offset > 0:
            _read_exact(stream, offset)
        for vol_idx in range(max_idx + 1):
            if vol_idx in wanted:
                raw_bytes = _read_exact(stream, vol_bytes)
                vol = np.ndarray(shape3d, dtype=disk_dtype, buffer=raw_bytes, order="F").astype(np.float32)
                if use_scale:
                    vol = vol * np.float32(slope_f) + np.float32(inter_f)
                if not np.isfinite(vol).all():
                    raise ValueError("Baseline images contain NaN/Inf")
                total += vol
            else:
                rem = vol_bytes
                while rem > 0:
                    skipped = stream.read(min(rem, skip_chunk))
                    if not skipped:
                        raise EOFError("Truncated gzipped NIfTI payload while skipping non-b0 volume")
                    rem -= len(skipped)
    return total


def _mean_b0(dwi_path, bval_path, bvec_path, output, threshold):
    import nibabel as nib
    import numpy as np

    image = nib.load(str(dwi_path), keep_file_open=True)
    bvals = np.loadtxt(str(bval_path)).reshape(-1)
    bvecs = np.loadtxt(str(bvec_path))
    if image.ndim != 4 or len(bvals) != image.shape[3] or not np.isfinite(bvals).all() or np.any(bvals < 0):
        raise ValueError("Mask generation requires 4D DWI and matching finite/nonnegative b-values")
    if bvecs.shape not in ((3, len(bvals)), (len(bvals), 3)) or not np.isfinite(bvecs).all():
        raise ValueError("Mask generation requires matching finite b-vectors")
    indices = np.flatnonzero(bvals <= threshold)
    if not len(indices):
        raise ValueError(f"No baseline measurements with b <= {threshold}")
    disable_fast = os.environ.get("SLICERDSPACE_DISABLE_PNL_ACCEL", "").strip().lower() in {"1", "true", "yes"}
    total = None if disable_fast else _accumulate_b0_single_pass_gzip(image, Path(dwi_path), indices)
    if total is None:
        total = np.zeros(image.shape[:3], np.float64)
        for index in indices:
            volume = np.asarray(image.dataobj[..., int(index)], dtype=np.float32)
            if not np.isfinite(volume).all():
                raise ValueError("Baseline images contain NaN/Inf")
            total += volume
    mean = (total / len(indices)).astype(np.float32)
    if not np.any(mean > 0):
        raise ValueError("Mean b0 has no positive signal")
    header = image.header.copy()
    header.set_data_dtype(np.float32)
    out_img = nib.Nifti1Image(mean, image.affine, header)
    if disable_fast:
        nib.save(out_img, str(output))
    else:
        _save_nifti_fast(out_img, Path(output))
    return len(indices)


def _run(command, env, log):
    # Inherit the pipeline's owned group so GUI cancellation reaches ANTs.
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               **text_subprocess_kwargs(bufsize=1))
    tail = deque(maxlen=50)
    try:
        for line in process.stdout:
            tail.append(line.rstrip())
            if log:
                log("[Brain mask] " + line.rstrip())
        code = process.wait()
    finally:
        process.stdout.close()
    if code:
        raise RuntimeError(f"Brain-mask command failed ({code}): " + "\n".join(tail))
    return "\n".join(tail)


@scheduled_stage('brain_mask', gpu=True)
def generate_brain_mask(dwi_path: Path, bval_path: Path, bvec_path: Path, output_mask_path: Path,
                        *, config: BrainMaskConfig | None = None, force: bool = False,
                        log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Run the selected backend. Supplied masks take priority; never silently fall back."""
    started = time.monotonic()
    config = config or BrainMaskConfig()
    paths = [Path(p).expanduser().resolve() for p in (dwi_path, bval_path, bvec_path)]
    output = Path(output_mask_path).expanduser()
    if output.is_symlink() or output.resolve() in paths:
        raise ValueError("Generated brain mask must not overwrite a source input or symlink")
    output = output.resolve()
    record_path = _manifest_path_for_mask(output)
    if record_path.is_symlink():
        raise ValueError("Mask provenance must not be a symlink")
    if config.backend not in BRAIN_MASK_BACKENDS:
        raise ValueError(unsupported_backend_message(config.backend))
    existing = {}
    if record_path.is_file():
        existing = json.loads(record_path.read_text(encoding="utf-8"))
    if output.is_file() and not force:
        qc = validate_brain_mask(paths[0], output)
        if config.backend == "provided" or existing.get("version") != 2:
            return {"status": "reused", "backend": "provided", "requested_backend": config.backend,
                    "previous_backend": existing.get("backend"), "mask_path": str(output), "qc": qc,
                    "manifest_path": str(record_path) if existing else None,
                    "note": "Existing user/legacy mask preserved; no automatic inference"}
    if config.backend == "provided":
        if not output.is_file():
            raise FileNotFoundError("Provided-mask route requires an existing native brain mask")
        return {"status": "reused", "backend": "provided", "mask_path": str(output),
                "qc": validate_brain_mask(paths[0], output)}
    backend = inspect_brain_mask_backend(config)
    route = backend["pnl_route_resolved"]
    if not backend["available"]:
        raise FileNotFoundError(f"Brain mask backend {config.backend} unavailable: " + ", ".join(backend["missing_required"]) +
                                ". Configure Advanced > Environment and models, choose another backend, or supply a verified mask.")
    resource_paths = [Path(p) for p in backend["resources"]]
    all_hash_targets = [*paths, *resource_paths]
    if len(all_hash_targets) > 1 and os.environ.get("SLICERDSPACE_DISABLE_PNL_ACCEL", "").strip().lower() not in {"1", "true", "yes"}:
        with ThreadPoolExecutor(max_workers=min(8, len(all_hash_targets))) as pool:
            fingerprints = list(pool.map(_fingerprint, all_hash_targets))
        input_fps = fingerprints[:len(paths)]
        resource_fps = fingerprints[len(paths):]
    else:
        input_fps = [_fingerprint(p) for p in paths]
        resource_fps = [_fingerprint(p) for p in resource_paths]
    signature = {"backend": config.backend, "settings": brain_mask_cache_inspection(backend)["settings"],
                 "inputs": input_fps,
                 "resources": resource_fps,
                 "tool_paths": backend["tool_paths"]}
    if output.exists() and not force and existing.get("version") == 2:
        if existing.get("output", {}).get("sha256") != _sha256(output):
            raise ValueError("Generated brain mask was edited; select provided after review, or explicitly regenerate")
        if signature == existing.get("signature"):
            return dict(existing, status="reused", qc=validate_brain_mask(paths[0], output))
    output.parent.mkdir(parents=True, exist_ok=True)
    tools = backend["tool_paths"]
    phase_seconds = {"preflight_and_hashing": time.monotonic() - started}
    with tempfile.TemporaryDirectory(prefix="sdds-mask-", dir="/tmp" if os.name == "posix" else None) as directory:
        # Keep all generated baseline/registration intermediates private.
        work = Path(directory)
        baseline = work / "mean_b0.nii.gz"
        phase_started = time.monotonic()
        count = _mean_b0(*paths, baseline, config.b0_threshold)
        phase_seconds["b0_preparation"] = time.monotonic() - phase_started
        candidate = work / "mask.nii.gz"
        prepend = []
        # A-O4: for pnl_cnn the effective child count (backend settings), else config.threads.
        threads = backend["settings"]["threads"]
        env_extra = {"OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
                     "OPENBLAS_NUM_THREADS": str(threads), "PYTHONUNBUFFERED": "1"}
        if route == "managed":
            # SC-1a: the pre-W-A conda runner (70fb635^1), ANTs 2.3.0 in the managed pnl_cnn runtime.
            models = work / "models"
            models.symlink_to(tools["model_dir"], target_is_directory=True)
            prepend.append(Path(tools["pnl_mask_python"]).parent)
            if tools.get("ants_bin_dir"):
                prepend.insert(0, Path(tools["ants_bin_dir"]))
                env_extra["ANTSPATH"] = tools["ants_bin_dir"]
            device_record = work / "pnl_device.json"
            command = [tools["pnl_mask_python"], str(Path(__file__).with_name("pnl_mask_runner.py")),
                       "--device", config.brain_mask_device, "--record", str(device_record),
                       "--script", tools["pnl_mask_script"], "--", "mask", "--input", str(baseline),
                       "--output", str(candidate), "--model-dir", str(models), "--percentile", "99",
                       "--threads", str(config.threads)]
            if tools.get("ants_bin_dir"):
                command.extend(("--ants-bin", tools["ants_bin_dir"]))
        else:
            # A1: one Slicer-Python process. No --threads: the child reads SLICERDSPACE_STAGE_CPUS (A6).
            # The receipt (route, versions, device, threads, phase times) becomes device_execution.
            models = Path(tools["model_dir"])
            device_record = work / "pnl_device.json"
            command = [tools["pythonslicer"], str(PNL_NATIVE_SCRIPT), "--input", str(baseline),
                       "--output", str(candidate), "--model-dir", str(models),
                       "--template", str(models / PNL_TEMPLATE_NAME),
                       "--device", config.brain_mask_device, "--record", str(device_record)]
        if route == "managed":
            from shared.managed_pnl import pnl_python_environment
            env = pnl_python_environment(tools["pnl_mask_python"], env_extra, path_prepend=prepend)
        else:
            env = python_tool_environment(tools["pythonslicer"], env_extra)
            # A6: the child reads the scheduler's CPU reservation; Slicer's startup environment does not hold it.
            for passthrough_key in ("SLICERDSPACE_STAGE_CPUS", "SLICERDSPACE_DISABLE_PNL_ACCEL", "CUDA_VISIBLE_DEVICES"):
                if os.environ.get(passthrough_key):
                    env[passthrough_key] = os.environ[passthrough_key]
        phase_started = time.monotonic()
        tail = _run(command, env, log)
        phase_seconds["external_process"] = time.monotonic() - phase_started
        device = json.loads(device_record.read_text(encoding="utf-8"))
        if not candidate.is_file():
            raise RuntimeError(f"{config.backend} returned success but did not create its declared mask")
        phase_started = time.monotonic()
        qc = validate_brain_mask(paths[0], candidate)
        phase_seconds["native_mask_qc"] = time.monotonic() - phase_started
        with tempfile.NamedTemporaryFile(prefix=".brain-mask-", suffix=".nii.gz", dir=output.parent, delete=False) as tmp:
            staged = Path(tmp.name)
        staging_record = None
        try:
            shutil.copyfile(candidate, staged)
            manifest = {"kind": "generated_brain_mask", "version": 2, "status": "completed", "backend": config.backend,
                        "requested_backend": config.backend, "mask_path": str(output), "manifest_path": str(record_path),
                        "signature": signature, "output": dict(_fingerprint(staged), path=str(output)), "b0_volumes": count,
                        "command": command, "stdout_tail": tail, "qc": qc,
                        "device_execution": device,
                        "phase_seconds": phase_seconds,
                        "phase_timing_scope": "Non-overlapping measured phases, not a complete sum of wall time; external process includes interpreter/model startup and upstream registration/inference/cleanup",
                        # SC-1b: the route. SPEC-GAP SC-1: the experimental status and reason of the native route are
                        # nested in `pnl_route` (same keys as the run record: route, status, reason), because the
                        # receipt's own `status` ("completed") is read by cache reuse and the callers.
                        **({"route": PNL_ROUTE_NAMES[route], "pnl_route": pnl_route_record(route)}
                           if config.backend == "pnl_cnn" else {}),
                        **({"child_threads": threads} if config.backend == "pnl_cnn" and route == "native" else {}),
                        "thread_environment": {key: env.get(key) for key in (
                            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                            "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "TF_ENABLE_ONEDNN_OPTS")},
                        "runtime_seconds": time.monotonic() - started,
                        "timing_scope": "Fresh mask generation, input/resource hashing, b0 preparation and automatic QC; excludes GUI startup/downloads",
                        "note": "Automatic geometry/integrity checks passed; anatomical quality awaits human review"}
            with tempfile.NamedTemporaryFile(mode="w", prefix=".mask-record-", suffix=".json", dir=record_path.parent,
                                             encoding="utf-8", delete=False) as tmp:
                staging_record = Path(tmp.name)
                json.dump(manifest, tmp, indent=2, sort_keys=True)
                tmp.write("\n")
            backup = None
            try:
                if output.exists():
                    with tempfile.NamedTemporaryFile(prefix=".mask-backup-", dir=output.parent, delete=False) as tmp:
                        backup = Path(tmp.name)
                    shutil.copy2(output, backup)
                os.replace(staged, output)
                try:
                    os.replace(staging_record, record_path)
                except BaseException:
                    # Two files cannot be renamed atomically as a pair. Roll
                    # back the mask if publishing its provenance fails.
                    if backup is not None:
                        os.replace(backup, output)
                    else:
                        output.unlink(missing_ok=True)
                    raise
            finally:
                staging_record.unlink(missing_ok=True)
                if backup is not None:
                    backup.unlink(missing_ok=True)
        finally:
            staged.unlink(missing_ok=True)
            if staging_record is not None:
                staging_record.unlink(missing_ok=True)
    return manifest
