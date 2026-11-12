"""Local DDSurfer postprocess backend bundled inside SlicerDSpace."""

from __future__ import annotations

from shared.cohort_schedule import scheduled_stage

import os
import shutil
import subprocess
import sys
import json
import hashlib
from uuid import uuid4
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from shared.environment import resolve_freesurfer_home
from shared.external_process import interpreter_path, merged_startup_environment, text_subprocess_kwargs

import nibabel as nib
import numpy as np
from nibabel.freesurfer.mghformat import MGHImage
from nibabel.processing import resample_from_to

from shared.io_layout import ANATOMY_OUTPUT_NAME, anatomy_input_dir, subject_layout
from shared.staging_paths import resolve_staging_subject_dir

REQUIRED_MRI_FILES = ("brain.mgz", "DDSurfer-wmparc.nii.gz")
FS_BRAIN_TEMPLATE_SHAPE = (256, 256, 256)
FS_BRAIN_TEMPLATE_AFFINE = np.array(
    [
        [-1.0, 0.0, 0.0, 127.70000457763672],
        [0.0, 0.0, 1.0, -145.0],
        [0.0, -1.0, 0.0, 147.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class DDSurferPostprocessConfig:
    subject_id: str
    output_root: Path
    subject_root: Optional[Path] = None
    native_dir: Optional[Path] = None
    mri_dir: Optional[Path] = None
    backend_root: Optional[Path] = None
    mode: str = "whole"
    python_bin: Optional[Path] = None
    freesurfer_home: Optional[Path] = None
    runtime: Optional[Path] = None
    license_file: Optional[Path] = None
    sphere_seed: Optional[int] = None
    stage_mode: str = "copy"
    atlases: Optional[tuple[str, ...]] = None
    force: bool = False


@dataclass(frozen=True)
class DDSurferPostprocessResult:
    manifest_path: Path
    qc_path: Path
    structural_root: Path
    stats_root: Path
    staged_input_dir: Path


def _emit(message: str, log: Optional[Callable[[str], None]] = None) -> None:
    if log is not None:
        log(message)


def _copy_or_link(src: Path, dest: Path, *, mode: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        return
    if mode in {"copy", "inplace"}:
        shutil.copy2(src, dest, follow_symlinks=True)
        return
    if mode != "symlink":
        raise ValueError("stage_mode must be 'copy', 'symlink', or 'inplace'")
    try:
        dest.symlink_to(src)
    except OSError:
        shutil.copy2(src, dest)


def _symlink_canonical_output(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_symlink():
        try:
            if dest.resolve(strict=True) == src.resolve():
                return
        except (FileNotFoundError, OSError):
            dest.unlink()
        else:
            dest.unlink()
    elif dest.exists():
        return
    try:
        dest.symlink_to(src, target_is_directory=True)
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dest, dirs_exist_ok=True)
        elif src.is_file():
            shutil.copy2(src, dest)


def _default_backend_root() -> Path:
    return Path(__file__).resolve().parent / "ddsurfer_postprocess_release"


def _default_freesurfer_home() -> Optional[Path]:
    return resolve_freesurfer_home()


def _resolve_bash_executable(python_bin: Path, *, platform: Optional[str] = None) -> str:
    """Prefer the installer-managed Slicer WSL bridge on Windows.

    The system ``bash.exe`` launches WSL but does not translate Windows/UNC
    arguments or map PythonSlicer to the managed Linux runtime.  The Windows
    dependency installer places a configured bridge beside ``Slicer.exe``.
    """

    if (platform or os.name) != "nt":
        return "bash"

    candidate_roots: list[Path] = []
    for executable in (python_bin, Path(sys.executable)):
        executable_parent = executable.expanduser().parent
        candidate_roots.append(
            executable_parent.parent if executable_parent.name.lower() == "bin" else executable_parent
        )
    for root in candidate_roots:
        candidate = root / "bash.exe"
        configuration = root / "bash.exe.conf"
        if candidate.is_file() and configuration.is_file():
            return str(candidate)
    return shutil.which("bash") or "bash"


def _resolve_subject_root(config: DDSurferPostprocessConfig) -> Path:
    if config.subject_root is not None:
        return config.subject_root.expanduser().resolve()
    output_root = config.output_root.expanduser().resolve()
    if output_root.name == "outputs":
        return output_root.parent
    raise ValueError("subject_root is required when output_root is not <subject_root>/outputs.")


def _resolve_native_dir(config: DDSurferPostprocessConfig, subject_root: Path) -> Path:
    if config.native_dir is not None:
        return config.native_dir.expanduser().resolve()
    layout = subject_layout(subject_root, output_root=config.output_root)
    return (layout.anatomy_input_dir("ddsurfer") / "pred_results_SlicerDSpace" / "native" / config.subject_id).resolve()


def _resolve_wmparc_path(subject_id: str, mri_dir: Path) -> Path:
    candidates = [
        mri_dir / "DDSurfer-wmparc.nii.gz",
        mri_dir / f"{subject_id}-DDSurfer-wmparc.nii.gz",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Required wmparc file missing under {mri_dir}. Expected one of: {', '.join(str(p) for p in candidates)}")


def _load_b0_or_fa_source(subject_root: Path, subject_id: str, output_root: Path) -> tuple[nib.spatialimages.SpatialImage, str]:
    layout = subject_layout(subject_root, output_root=output_root)
    fa_path = layout.microstructure_dti / f"{subject_id}_dti_FA.nii.gz"
    if fa_path.is_file():
        return nib.load(str(fa_path)), "fa"

    preproc_dwi = layout.preproc / f"{subject_id}_dwi_final.nii.gz"
    preproc_bval = layout.preproc / f"{subject_id}_dwi_final.bval"
    if preproc_dwi.is_file():
        dwi_img = nib.load(str(preproc_dwi))
        if len(dwi_img.shape) == 4:
            dwi = np.asarray(dwi_img.dataobj, dtype=np.float32)
            if preproc_bval.is_file():
                bvals = np.loadtxt(preproc_bval, dtype=np.float32, ndmin=1)
                b0_idx = np.where(bvals <= 50.0)[0]
                if b0_idx.size == 0:
                    b0_idx = np.array([0], dtype=int)
            else:
                b0_idx = np.array([0], dtype=int)
            brain = dwi[..., b0_idx].mean(axis=-1, dtype=np.float32)
        else:
            brain = np.asarray(dwi_img.dataobj, dtype=np.float32)
        return nib.Nifti1Image(brain, dwi_img.affine), "b0_mean"

    raw_dwi = subject_root / "input" / "data.nii.gz"
    raw_bval = subject_root / "input" / "bvals"
    if raw_dwi.is_file():
        raw_img = nib.load(str(raw_dwi))
        if len(raw_img.shape) == 4:
            dwi = np.asarray(raw_img.dataobj, dtype=np.float32)
            if raw_bval.is_file():
                bvals = np.loadtxt(raw_bval, dtype=np.float32, ndmin=1)
                b0_idx = np.where(bvals <= 50.0)[0]
                if b0_idx.size == 0:
                    b0_idx = np.array([0], dtype=int)
            else:
                b0_idx = np.array([0], dtype=int)
            brain = dwi[..., b0_idx].mean(axis=-1, dtype=np.float32)
        else:
            brain = np.asarray(raw_img.dataobj, dtype=np.float32)
        return nib.Nifti1Image(brain, raw_img.affine), "raw_b0_mean"

    raise FileNotFoundError("Unable to derive brain.mgz from subject data. Expected preprocessed DWI, raw DWI, or FA map.")


def _generate_brain_mgz(*, subject_root: Path, subject_id: str, output_root: Path, generated_mri_dir: Path) -> Path:
    generated_mri_dir.mkdir(parents=True, exist_ok=True)
    output_path = generated_mri_dir / "brain.mgz"
    if output_path.is_file():
        existing = nib.load(str(output_path))
        if tuple(existing.shape) == tuple(FS_BRAIN_TEMPLATE_SHAPE) and np.allclose(existing.affine, FS_BRAIN_TEMPLATE_AFFINE):
            return output_path
    source_img, source_kind = _load_b0_or_fa_source(subject_root, subject_id, output_root)
    data = np.asarray(source_img.get_fdata(dtype=np.float32), dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    finite = data[np.isfinite(data)]
    if finite.size:
        lo = float(np.percentile(finite, 1.0))
        hi = float(np.percentile(finite, 99.0))
        if hi > lo:
            data = np.clip((data - lo) / (hi - lo), 0.0, 1.0)
        else:
            data = np.zeros_like(data, dtype=np.float32)
    brain_u8 = np.round(data * 255.0).astype(np.uint8)
    source_like = nib.Nifti1Image(brain_u8.astype(np.float32), source_img.affine)
    resampled = resample_from_to(
        source_like,
        (tuple(int(v) for v in FS_BRAIN_TEMPLATE_SHAPE), np.asarray(FS_BRAIN_TEMPLATE_AFFINE, dtype=np.float32)),
        order=1,
    )
    resampled_u8 = np.round(np.clip(np.asarray(resampled.dataobj, dtype=np.float32), 0.0, 255.0)).astype(np.uint8)
    brain_mgz = MGHImage(resampled_u8, np.asarray(FS_BRAIN_TEMPLATE_AFFINE, dtype=np.float32))
    nib.save(brain_mgz, str(output_path))
    (generated_mri_dir / "brain_source.txt").write_text(source_kind + "\n", encoding="utf-8")
    return output_path


def _expose_canonical_postprocess_outputs(*, output_root: Path, subject_id: str) -> tuple[Path, Path]:
    nested_subject_root = output_root / subject_id
    nested_structural_root = anatomy_input_dir(nested_subject_root)
    nested_stats_root = nested_subject_root / "05_statistics"
    canonical_structural_root = output_root / ANATOMY_OUTPUT_NAME
    canonical_stats_root = output_root / "05_statistics"
    if nested_structural_root.exists():
        _symlink_canonical_output(nested_structural_root, canonical_structural_root)
    if nested_stats_root.exists():
        _symlink_canonical_output(nested_stats_root, canonical_stats_root)
    return (
        canonical_structural_root / "ddsurfer" / "postprocess",
        canonical_stats_root / "ddsurfer" / "postprocess",
    )


def stage_postprocess_inputs(
    *,
    subject_id: str,
    native_dir: Path,
    wmparc_path: Path,
    brain_mgz_path: Path,
    staging_inputs_root: Path,
    mode: str,
    force: bool,
    stage_mode: str = "copy",
) -> Path:
    subject_input_dir = resolve_staging_subject_dir(staging_inputs_root, subject_id, for_delete=force)
    surf_dir = subject_input_dir / "surf"
    mri_out_dir = subject_input_dir / "mri"
    if force and subject_input_dir.exists():
        shutil.rmtree(subject_input_dir)
    surf_dir.mkdir(parents=True, exist_ok=True)
    mri_out_dir.mkdir(parents=True, exist_ok=True)

    _copy_or_link(brain_mgz_path, mri_out_dir / "brain.mgz", mode=stage_mode)
    _copy_or_link(wmparc_path, mri_out_dir / "DDSurfer-wmparc.nii.gz", mode=stage_mode)

    hemi_tokens: tuple[str, ...]
    normalized_mode = mode.lower()
    if normalized_mode == "left":
        hemi_tokens = ("_lh_",)
    elif normalized_mode == "right":
        hemi_tokens = ("_rh_",)
    else:
        hemi_tokens = ("_lh_", "_rh_")

    stl_paths = sorted(
        path
        for path in native_dir.glob(f"{subject_id}_predicted_*_orig.stl")
        if any(token in path.name for token in hemi_tokens)
    )
    if not stl_paths:
        raise FileNotFoundError(f"No *_orig.stl files found under {native_dir}")
    for stl in stl_paths:
        _copy_or_link(stl, surf_dir / stl.name, mode=stage_mode)
    return subject_input_dir


@scheduled_stage('surface_postprocess')
def run_ddsurfer_postprocess(
    config: DDSurferPostprocessConfig,
    *,
    log: Optional[Callable[[str], None]] = None,
) -> DDSurferPostprocessResult:
    if (not config.subject_id.strip() or Path(config.subject_id).name != config.subject_id
            or config.subject_id in ('.', '..', 'fsaverage')):
        raise ValueError('subject_id must be a nonempty directory name')
    if config.mode not in ('whole', 'left', 'right'):
        raise ValueError('mode must be whole, left, or right')
    if config.backend_root is None:
        subject_root = _resolve_subject_root(config)
        native_dir = _resolve_native_dir(config, subject_root)
        hemis = ('lh',) if config.mode == 'left' else ('rh',) if config.mode == 'right' else ('lh', 'rh')
        if all((native_dir / f'{hemi}.{kind}.obj').is_file() for hemi in hemis for kind in ('white', 'pial')):
            return _run_current_postprocess(config, subject_root, native_dir, log)
    if config.runtime is not None:
        raise ValueError('Local components require indexed native white/pial OBJ inputs; legacy STL processing still needs its full external backend')
    backend_root = (config.backend_root or _default_backend_root()).expanduser().resolve()
    run_script = backend_root / "run.sh"
    if not run_script.is_file():
        raise FileNotFoundError(f"DDSurfer postprocess backend missing run.sh: {run_script}")

    output_root = config.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    subject_root = _resolve_subject_root(config)
    layout = subject_layout(subject_root, output_root=output_root)
    native_dir = _resolve_native_dir(config, subject_root)
    mri_dir = config.mri_dir.expanduser().resolve() if config.mri_dir is not None else layout.anatomy_input_dir("ddparcel")
    wmparc_path = _resolve_wmparc_path(config.subject_id, mri_dir)
    brain_mgz_path = mri_dir / "brain.mgz"

    runtime_root = layout.anatomy / "cache" / "ddsurfer_postprocess_runtime"
    generated_mri_dir = runtime_root / "generated_mri" / config.subject_id
    if not brain_mgz_path.is_file():
        brain_mgz_path = _generate_brain_mgz(
            subject_root=subject_root,
            subject_id=config.subject_id,
            output_root=output_root,
            generated_mri_dir=generated_mri_dir,
        )
    staging_inputs_root = runtime_root / "inputs"
    staged_input_dir = stage_postprocess_inputs(
        subject_id=config.subject_id,
        native_dir=native_dir,
        wmparc_path=wmparc_path,
        brain_mgz_path=brain_mgz_path,
        staging_inputs_root=staging_inputs_root,
        mode=config.mode,
        force=config.force,
        stage_mode=config.stage_mode,
    )

    python_bin = interpreter_path(config.python_bin) if config.python_bin else Path(sys.executable)
    freesurfer_home = config.freesurfer_home.expanduser().resolve() if config.freesurfer_home else _default_freesurfer_home()

    cmd = [
        _resolve_bash_executable(python_bin),
        str(run_script),
        "--subject-id",
        config.subject_id,
        "--mode",
        config.mode,
        "--inputs-dir",
        str(staging_inputs_root),
        "--outputs-dir",
        str(output_root),
        "--nest-subject-outputs",
        "0",
        "--stage-mode",
        config.stage_mode,
        "--python-bin",
        str(python_bin),
    ]
    if freesurfer_home is not None:
        cmd.extend(["--freesurfer-home", str(freesurfer_home)])
    if config.atlases:
        cmd.extend(["--atlases", ",".join(config.atlases)])

    _emit("[DDSurfer Postprocess] Running: " + " ".join(cmd), log)
    env = merged_startup_environment()
    env["SLICERDSPACE_ANATOMY_DIRNAME"] = ANATOMY_OUTPUT_NAME
    if os.name == "nt":
        wslenv = env.get("WSLENV", "").strip(":")
        entry = "SLICERDSPACE_ANATOMY_DIRNAME/u"
        env["WSLENV"] = f"{wslenv}:{entry}" if wslenv else entry
    process = subprocess.Popen(
        cmd,
        cwd=str(backend_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        **text_subprocess_kwargs(bufsize=1),
    )
    try:
        if process.stdout:
            for line in process.stdout:
                _emit(line.rstrip(), log)
        return_code = process.wait()
    finally:
        if process.stdout:
            process.stdout.close()
    if return_code != 0:
        raise RuntimeError(f"DDSurfer postprocess failed with exit code {return_code}")

    structural_root = anatomy_input_dir(output_root, "ddsurfer/postprocess")
    stats_root = output_root / "05_statistics" / "ddsurfer" / "postprocess"
    return DDSurferPostprocessResult(
        manifest_path=structural_root / "manifest.json",
        qc_path=structural_root / "qc.json",
        structural_root=structural_root,
        stats_root=stats_root,
        staged_input_dir=staged_input_dir,
    )


def _run_current_postprocess(config, subject_root, native_dir, log):
    from .backend_bridge import BACKEND_ROOT, native_surface_hashes, run_backend, sha256, upstream_record
    layout = subject_layout(subject_root, output_root=config.output_root)
    from shared.surface_runtime import resolve_surface_runtime, surface_runtime_identity
    backend, home = resolve_surface_runtime(config.runtime, config.freesurfer_home)
    brain = config.mri_dir / 'brain.mgz' if config.mri_dir is not None else None
    brain = brain if brain is not None and brain.is_file() else None
    inputs = native_surface_hashes(native_dir, config.mode)
    signature = dict(upstream_commit=upstream_record()['commit'], native_dir=str(native_dir),
                     native_sha256=inputs, mode=config.mode,
                     atlases=list(config.atlases or ('aparc', 'aparc.a2009s')),
                     freesurfer_home=str(home), brain_source=str(brain) if brain else None,
                     brain_sha256=sha256(brain) if brain else None)
    signature['surface_runtime'] = surface_runtime_identity(config.runtime, config.freesurfer_home)
    signature['sphere_seed'] = config.sphere_seed
    cache_key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:20]
    if config.force:
        cache_key += '-' + uuid4().hex[:12]
    runtime = layout.anatomy / 'cache/ddsurfer_surface_only' / cache_key
    fs_subject = runtime / config.subject_id
    command = [BACKEND_ROOT / 'postprocessing/pipeline.py', '--subject', config.subject_id,
               '--output-root', runtime, '--hemi', 'both' if config.mode == 'whole' else config.mode,
               '--atlases', ','.join(config.atlases or ('aparc', 'aparc.a2009s')), '--backend', backend]
    hemis = ('lh',) if config.mode == 'left' else ('rh',) if config.mode == 'right' else ('lh', 'rh')
    for hemi in hemis:
        for kind in ('white', 'pial'):
            command += [f'--{hemi}-{kind}', native_dir / f'{hemi}.{kind}.obj']
    command += ['--runtime' if backend == 'native' else '--freesurfer-home', home]
    from shared.surface_license import resolve_surface_license
    activation = resolve_surface_license(config.license_file, freesurfer_home=home)
    if activation is not None:
        command += ['--license-file', activation]
    if config.sphere_seed is not None:
        command += ['--sphere-seed', str(config.sphere_seed)]
    if brain is not None:
        command += ['--brain-source', brain]
    structural = layout.ddsurfer_postprocess
    stats = layout.ddsurfer_postprocess_stats
    structural.mkdir(parents=True, exist_ok=True)
    stats.mkdir(parents=True, exist_ok=True)
    manifest = dict(kind='ddsurfer_surface_only_postprocess', upstream_commit=upstream_record()['commit'],
                    subject_id=config.subject_id, status='running', native_dir=str(native_dir),
                    command=list(map(str, command)), requires_ddparcel=False,
                    native_sha256=inputs, cache_signature=signature, cache_key=cache_key)
    manifest_path, qc_path = structural / 'manifest.json', structural / 'qc.json'
    prior_is_current = False
    if manifest_path.is_file():
        try:
            prior_is_current = json.loads(manifest_path.read_text()).get('kind') == 'ddsurfer_surface_only_postprocess'
        except Exception:
            prior_is_current = False
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    try:
        run_backend(command, python_bin=config.python_bin, log=log, log_path=structural / 'backend.log')
        checks = json.loads((fs_subject / 'logs/qc.json').read_text())
        from .postprocess_qc import validate_postprocess_outputs
        integrity = validate_postprocess_outputs(fs_subject, hemis, signature['atlases'])
        (fs_subject / 'logs/integrity_qc.json').write_text(json.dumps(integrity, indent=2) + '\n')
        if not integrity['automatic_qc_passed']:
            failed = [name for name, passed in integrity['checks'].items() if not passed]
            raise ValueError('Surface postprocess integrity failed: ' + ', '.join(failed))
        if native_surface_hashes(native_dir, config.mode) != inputs or (brain is not None and sha256(brain) != signature['brain_sha256']):
            raise RuntimeError('Surface postprocess inputs changed while running; outputs are not current')
        for source, destination in [(fs_subject / name, structural / name) for name in
                                    ('surf', 'label', 'mri', 'fsaverage', 'logs')] + [(fs_subject / 'stats', stats / 'stats')]:
            if destination.exists() and not destination.is_symlink():
                if prior_is_current:
                    shutil.rmtree(destination) if destination.is_dir() else destination.unlink()
                else:
                    backup = destination.parent / ('legacy_before_' + upstream_record()['commit'][:12]) / destination.name
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    if backup.exists():
                        raise FileExistsError(f'Existing postprocess output conflicts with preserved backup: {backup}')
                    destination.rename(backup)
            _symlink_canonical_output(source, destination)
        # Track the public copies too on platforms without directory symlinks.
        canonical_pairs = [(fs_subject / name, structural / name) for name in
                           ('surf', 'label', 'mri', 'fsaverage', 'logs')] + [(fs_subject / 'stats', stats / 'stats')]
        all_linked = all(
            dest.is_symlink() and dest.resolve() == src.resolve()
            for src, dest in canonical_pairs
        )
        if not all_linked:
            integrity = validate_postprocess_outputs(structural, hemis, signature['atlases'], stats_dir=stats / 'stats')
            if not integrity['automatic_qc_passed']:
                raise ValueError('Published surface postprocess products failed integrity QC')
        manifest['status'] = 'completed'
        qc = dict(status='completed', upstream_commit=upstream_record()['commit'],
                  automatic_qc=checks, automatic_qc_passed=True, integrity=integrity, human_review='pending')
    except Exception as exc:
        manifest['status'] = 'failed'
        qc = dict(status='failed', error=str(exc))
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        qc_path.write_text(json.dumps(qc, indent=2) + '\n')
    return DDSurferPostprocessResult(manifest_path, qc_path, structural, stats, native_dir)
