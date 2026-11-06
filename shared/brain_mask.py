"""Selectable native-DWI masking, with external runtimes and explicit provenance."""

from __future__ import annotations

from shared.cohort_schedule import scheduled_stage

from collections import deque
from dataclasses import dataclass
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

from shared.external_process import external_python_environment, python_tool_environment, text_subprocess_kwargs


BRAIN_MASK_BACKENDS = ("pnl_cnn", "synthstrip", "mrtrix", "provided", "dipy")
BRAIN_MASK_DEVICES = ("auto", "cpu", "cuda")
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
    synthstrip_python: Optional[Path] = None
    mrtrix_bin_dir: Optional[Path] = None
    freesurfer_home: Optional[Path] = None
    b0_threshold: float = 50.0
    threads: int = 4
    model_assets_dir: Optional[Path] = None


def brain_mask_config_from_options(options) -> BrainMaskConfig:
    return BrainMaskConfig(
        backend=getattr(options, "brain_mask_backend", "pnl_cnn"),
        brain_mask_device=getattr(options, "brain_mask_device", "auto"),
        **{name: getattr(options, name, None) for name in (
            "pnl_mask_python", "pnl_mask_script", "pnl_mask_model_dir", "pnl_mask_ants_bin_dir", "synthstrip_python", "model_assets_dir")},
        mrtrix_bin_dir=getattr(options, "mrtrix_bin_dir", None) or getattr(options, "preprocessing_mrtrix_bin_dir", None),
        freesurfer_home=getattr(options, "postprocess_freesurfer_home", None),
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


def resolve_dwi2mask(explicit_bin_dir: Optional[Path] = None) -> Optional[Path]:
    for folder in (explicit_bin_dir, os.environ.get("SLICERDSPACE_MRTRIX_BIN_DIR"), os.environ.get("MRTRIX_BIN_DIR")):
        if folder:
            candidate = Path(folder).expanduser().resolve() / "dwi2mask"
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
    found = shutil.which("dwi2mask")
    return Path(found).resolve() if found else None


def _pnl_native_python() -> Optional[Path]:
    """PythonSlicer for the A1 subprocess, found as for ``teem_dti.py``; None outside a Slicer Python."""
    try:
        # Imported here: shared.pipeline imports this module.
        from shared.pipeline import resolve_pythonslicer_executable
        return Path(resolve_pythonslicer_executable())
    except (ImportError, RuntimeError):
        return None


def pnl_child_threads() -> int:
    """Thread count the native PNL child uses (A6): ``SLICERDSPACE_STAGE_CPUS``, else affinity, else CPU count."""
    from shared.pnl_mask_native import default_threads
    return default_threads()


def pnl_native_resources() -> list[Path]:
    """A7 cache resources: ``pnl_mask_native.py`` and every vendored file in UPSTREAM.json (O5: with the JSON)."""
    record = json.loads((PNL_VENDOR_DIR / "UPSTREAM.json").read_text(encoding="utf-8"))
    return [PNL_NATIVE_SCRIPT, PNL_VENDOR_DIR / "UPSTREAM.json",
            *(PNL_VENDOR_DIR / relative for relative in sorted(record["files"]))]


def inspect_brain_mask_backend(config: BrainMaskConfig | None = None, *, system: str | None = None) -> dict[str, Any]:
    """Inspect files/tools only; no framework import, model download or inference."""
    config = config or BrainMaskConfig()
    if config.backend not in BRAIN_MASK_BACKENDS:
        raise ValueError("brain_mask_backend must be one of: " + ", ".join(BRAIN_MASK_BACKENDS))
    if config.brain_mask_device not in BRAIN_MASK_DEVICES:
        raise ValueError("brain_mask_device must be one of: " + ", ".join(BRAIN_MASK_DEVICES))
    if not math.isfinite(config.b0_threshold) or config.b0_threshold < 0 or config.threads < 1:
        raise ValueError("Mask b0 threshold must be finite/nonnegative and threads must be positive")
    missing = []
    tools, resources = {}, []
    settings = {}
    threads = config.threads
    if config.backend == "pnl_cnn":
        # A-O4: the child ignores config.threads; the signature records the count it will use.
        threads = pnl_child_threads()
        # A7 (T2 design v4): Slicer's Python runs shared/pnl_mask_native.py (A1) with the vendored CNN code.
        # The conda runner settings (pnl_mask_python, pnl_mask_script, pnl_mask_ants_bin_dir) are not read;
        # they stay in the config until A8 removes the managed profile.
        python = _pnl_native_python()
        if python is None or not python.is_file() or not os.access(python, os.X_OK):
            missing.append(PNL_NATIVE_PYTHON_MISSING)
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
    elif config.backend == "synthstrip":
        if (system or platform.system()) == "Windows" or os.name == "nt":
            missing.append("Full FreeSurfer SynthStrip runtime is not native Windows; use a reviewed supplied mask or an explicitly configured Linux/WSL pipeline")
        home = _path(config.freesurfer_home, "FREESURFER_HOME")
        executable = home / "bin/mri_synthstrip" if home else _path(shutil.which("mri_synthstrip"))
        if executable is None or not executable.is_file() or not os.access(executable, os.X_OK):
            missing.append("FreeSurfer mri_synthstrip")
        else:
            tools["mri_synthstrip"] = str(executable)
            resources.append(executable)
            home = home or executable.parent.parent
            model = home / "models/synthstrip.1.pt"
            if not model.is_file():
                from shared.optional_assets import verified_resource_directory
                shared = verified_resource_directory("synthstrip", asset_root=config.model_assets_dir)
                if shared:
                    model = shared / "synthstrip.1.pt"
            if model.is_file():
                resources.append(model)
                if model != home / "models/synthstrip.1.pt":
                    tools["synthstrip_model"] = str(model)
            else:
                missing.append("FreeSurfer models/synthstrip.1.pt")
            tools["freesurfer_home"] = str(home)
            python = _path(config.synthstrip_python, "SLICERDSPACE_SYNTHSTRIP_PYTHON", resolve=False) or home / "bin/fspython"
            script = home / "python/scripts/mri_synthstrip"
            if (config.synthstrip_python or os.environ.get("SLICERDSPACE_SYNTHSTRIP_PYTHON")) and (
                    not python.is_file() or not os.access(python, os.X_OK)):
                missing.append("executable configured synthstrip_python")
            if python.is_file() and script.is_file():
                tools["synthstrip_python"], tools["synthstrip_script"] = str(python), str(script)
                resources.extend((script, Path(__file__).with_name("synthstrip_mask_runner.py")))
            elif config.brain_mask_device != "cpu":
                missing.append("SynthStrip fspython and python/scripts/mri_synthstrip for auto/CUDA device discovery (or explicitly select CPU)")
    elif config.backend == "dipy":
        from importlib.metadata import PackageNotFoundError, version
        try:
            implementation = version('dipy')
        except PackageNotFoundError:
            implementation = 'not installed'
        # Python-package availability is checked in the selected interpreter by
        # Dependency Doctor; no external executable or model is required here.
        settings = dict(implementation='dipy.segment.mask.median_otsu',
                        implementation_version=implementation, median_radius=4,
                        numpass=4, dilate=1, autocrop=False)
        resources.append(Path(__file__))
    elif config.backend == "mrtrix":
        executable = resolve_dwi2mask(config.mrtrix_bin_dir)
        if executable is None:
            missing.append("MRtrix3 dwi2mask")
        else:
            tools["dwi2mask"] = str(executable)
            resources.append(executable)
    return {"backend": config.backend, "available": not missing, "missing_required": missing,
            "tool_paths": tools, "resources": [str(p) for p in resources],
            "settings": {"b0_threshold": config.b0_threshold, "threads": threads,
                         "pnl_percentile": 99, "pnl_cleanup": "official_fill_holes_no_optional_filter",
                         "device": config.brain_mask_device if config.backend in ("pnl_cnn", "synthstrip") else "cpu",
                         **({"pnl_framework": "pytorch", "pnl_route": "native-sitk",
                             "pnl_package": "vendored PNLBrainMask-Torch 8d201a5 (shared/pnl_cnn)",
                             "pnl_registration": "SimpleITK rigid, ANTs rigid-stage parameters",
                             "pnl_inverse_interpolation": "SimpleITK linear, float32"} if config.backend == "pnl_cnn" else {}),
                         **settings},
            "note": "File discovery is not runtime/model compatibility or anatomical QC"}


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
    nib.save(nib.Nifti1Image(mean, image.affine, header), str(output))
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
        raise ValueError("Unknown brain mask backend: " + config.backend)
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
    if not backend["available"]:
        raise FileNotFoundError(f"Brain mask backend {config.backend} unavailable: " + ", ".join(backend["missing_required"]) +
                                ". Configure Advanced > Environment and models, choose another backend, or supply a verified mask.")
    signature = {"backend": config.backend, "settings": backend["settings"],
                 "inputs": [_fingerprint(p) for p in paths],
                 "resources": [_fingerprint(Path(p)) for p in backend["resources"]],
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
        if config.backend == "dipy":
            command = []
        elif config.backend == "pnl_cnn":
            # A1: one Slicer-Python process. No --threads: the child reads SLICERDSPACE_STAGE_CPUS (A6).
            # The receipt (route, versions, device, threads, phase times) becomes device_execution.
            models = Path(tools["model_dir"])
            device_record = work / "pnl_device.json"
            command = [tools["pythonslicer"], str(PNL_NATIVE_SCRIPT), "--input", str(baseline),
                       "--output", str(candidate), "--model-dir", str(models),
                       "--template", str(models / PNL_TEMPLATE_NAME),
                       "--device", config.brain_mask_device, "--record", str(device_record)]
        elif config.backend == "synthstrip":
            env_extra["FREESURFER_HOME"] = tools["freesurfer_home"]
            prepend.append(Path(tools["mri_synthstrip"]).parent)
            command = [tools["mri_synthstrip"], "-i", str(baseline), "-m", str(candidate)]
            if "synthstrip_model" in tools:
                command.extend(("--model", tools["synthstrip_model"]))
            if "synthstrip_python" in tools:
                device_record = work / "synthstrip_device.json"
                command = [tools["synthstrip_python"], str(Path(__file__).with_name("synthstrip_mask_runner.py")),
                           "--device", config.brain_mask_device, "--record", str(device_record),
                           "--script", tools["synthstrip_script"], "--", *command[1:]]
        else:
            command = [tools["dwi2mask"], str(paths[0]), str(candidate), "-fslgrad", str(paths[2]), str(paths[1]), "-force"]
        env = external_python_environment(env_extra, path_prepend=prepend)
        if config.backend == "pnl_cnn":
            env = python_tool_environment(tools["pythonslicer"], env_extra)
            # A6: the child reads the scheduler's CPU reservation; Slicer's startup environment does not hold it.
            if os.environ.get("SLICERDSPACE_STAGE_CPUS"):
                env["SLICERDSPACE_STAGE_CPUS"] = os.environ["SLICERDSPACE_STAGE_CPUS"]
        phase_started = time.monotonic()
        if config.backend == 'dipy':
            import nibabel as nib
            import numpy as np
            from dipy.segment.mask import median_otsu
            baseline_image = nib.load(baseline)
            _, mask = median_otsu(np.asarray(baseline_image.dataobj), median_radius=4,
                                  numpass=4, dilate=1, autocrop=False)
            header = baseline_image.header.copy()
            header.set_data_dtype(np.uint8)
            nib.save(nib.Nifti1Image(mask.astype(np.uint8), baseline_image.affine, header), candidate)
            tail = 'DIPY median_otsu: mean b0, radius=4, passes=4, dilation=1, native grid'
            if log:
                log('[Brain mask] ' + tail)
            phase_seconds['native_python_mask'] = time.monotonic() - phase_started
            device = dict(requested=config.brain_mask_device, resolved='cpu',
                          reason='DIPY median_otsu is a CPU algorithm')
        else:
            tail = _run(command, env, log)
            phase_seconds["external_process"] = time.monotonic() - phase_started
            device = json.loads(device_record.read_text(encoding="utf-8")) if config.backend == "pnl_cnn" or "synthstrip_python" in tools else None
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
                        **({"child_threads": threads} if config.backend == "pnl_cnn" else {}),
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


def generate_brain_mask_with_dwi2mask(dwi_path, bval_path, bvec_path, output_mask_path,
                                     *, mrtrix_bin_dir=None, force=False):
    """Compatibility wrapper for callers explicitly requesting MRtrix."""
    return generate_brain_mask(dwi_path, bval_path, bvec_path, output_mask_path,
                               config=BrainMaskConfig(backend="mrtrix", mrtrix_bin_dir=mrtrix_bin_dir), force=force)
