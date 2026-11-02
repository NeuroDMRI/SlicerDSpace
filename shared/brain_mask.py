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

from shared.external_process import external_python_environment, text_subprocess_kwargs


BRAIN_MASK_BACKENDS = ("pnl_cnn", "synthstrip", "mrtrix", "provided", "dipy")
BRAIN_MASK_DEVICES = ("auto", "cpu", "cuda")


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


def inspect_brain_mask_backend(config: BrainMaskConfig | None = None, *, system: str | None = None) -> dict[str, Any]:
    """Inspect files/tools only; no TensorFlow import, model download or inference."""
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
    if config.backend == "pnl_cnn":
        resources.append(Path(__file__).with_name("pnl_mask_runner.py"))
        # Resolving a venv's Python symlink would select the base interpreter
        # and lose that environment's TensorFlow and conversion packages.
        python = _path(config.pnl_mask_python, "SLICERDSPACE_PNL_MASK_PYTHON", resolve=False)
        script = _path(config.pnl_mask_script, "SLICERDSPACE_PNL_MASK_SCRIPT")
        models = _path(config.pnl_mask_model_dir, "SLICERDSPACE_PNL_MASK_MODEL_DIR")
        if models is None:
            from shared.optional_assets import verified_resource_directory
            models = verified_resource_directory("pnl-cnn", asset_root=config.model_assets_dir)
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
        if models is None or not models.is_dir():
            missing.append("pnl_mask_model_dir")
        else:
            tools["model_dir"] = str(models)
            for name in ("CompNetBasicModel.json", "IITmean_b0_256.nii.gz"):
                file = models / name
                if not file.is_file():
                    missing.append(name)
                else:
                    resources.append(file)
            for view in ("sagittal", "coronal", "axial"):
                weights = sorted(models.glob(f"weights-{view}-improvement-*.h5"))
                # The official glob(...)[-1] is ambiguous with multiple files.
                if len(weights) != 1:
                    missing.append(f"exactly one weights-{view}-improvement-*.h5")
                else:
                    resources.extend(weights)
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
            "settings": {"b0_threshold": config.b0_threshold, "threads": config.threads,
                         "pnl_percentile": 99, "pnl_cleanup": "official_fill_holes_no_optional_filter",
                         "device": config.brain_mask_device if config.backend in ("pnl_cnn", "synthstrip") else "cpu",
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
        # PNL constructs shell commands internally. Safe aliases isolate its
        # workspace and cleanup from user inputs, including paths with spaces.
        work = Path(directory)
        baseline = work / "mean_b0.nii.gz"
        phase_started = time.monotonic()
        count = _mean_b0(*paths, baseline, config.b0_threshold)
        phase_seconds["b0_preparation"] = time.monotonic() - phase_started
        candidate = work / "mask.nii.gz"
        prepend = []
        env_extra = {"OMP_NUM_THREADS": str(config.threads), "MKL_NUM_THREADS": str(config.threads),
                     "OPENBLAS_NUM_THREADS": str(config.threads), "PYTHONUNBUFFERED": "1"}
        if config.backend == "dipy":
            command = []
        elif config.backend == "pnl_cnn":
            models = work / "models"
            models.symlink_to(tools["model_dir"], target_is_directory=True)
            listing = work / "b0_list.txt"
            listing.write_text(str(baseline) + "\n", encoding="utf-8")
            prepend.append(Path(tools["pnl_mask_python"]).parent)
            if tools.get("ants_bin_dir"):
                prepend.insert(0, Path(tools["ants_bin_dir"]))
                env_extra["ANTSPATH"] = tools["ants_bin_dir"]
            device_record = work / "pnl_device.json"
            command = [tools["pnl_mask_python"], str(Path(__file__).with_name("pnl_mask_runner.py")),
                       "--device", config.brain_mask_device, "--record", str(device_record),
                       "--script", tools["pnl_mask_script"], "--", "-i", str(listing),
                       "-f", str(models), "-nproc", "1", "-p", "99", "-qc", "0"]
            candidate = work / "mean_b0_bse-multi_BrainMask.nii.gz"
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
            from shared.managed_pnl import pnl_python_environment
            env = pnl_python_environment(tools["pnl_mask_python"], env_extra, path_prepend=prepend)
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
