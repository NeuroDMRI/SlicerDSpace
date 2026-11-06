"""DDParcel preprocessing pipeline adapted for SlicerDSpace."""

from __future__ import annotations

from shared.cohort_schedule import scheduled_stage

import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from shared.external_process import text_subprocess_kwargs
from shared.model_assets import model_asset_path
from shared.pipeline import (RegisteredDTIBundle, build_cli_environment, native_tensor_command,
                             resolve_cli_executable, resolve_pythonslicer_executable,
                             _artifact_record, _artifacts_current, _save_artifacts,
                             _read_manifest, _write_manifest)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DDParcelConfig:
    subject_id: str
    input_root: Path
    output_root: Path
    atlas_t2: Path
    weights_dir: Optional[Path] = None
    flip: int = 1
    device: str = "auto"


@dataclass(frozen=True)
class DDParcelRegisteredConfig:
    subject_id: str
    registered_bundle: RegisteredDTIBundle
    output_dir: Path
    native_mask: Path
    weights_dir: Optional[Path] = None
    flip: int = 1
    device: str = "auto"
    force_recompute: bool = False


def _log_message(message: str, log: Optional[Callable[[str], None]] = None) -> None:
    LOGGER.info(message)
    if log:
        log(message)


def _run_cli(cmd: list[str], log: Optional[Callable[[str], None]] = None) -> None:
    _log_message(f"[DDParcel] Running: {' '.join(cmd)}", log)
    process = subprocess.Popen(
        cmd,
        env=build_cli_environment(cmd[0]),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        **text_subprocess_kwargs(bufsize=1),
    )
    if process.stdout:
        for line in process.stdout:
            _log_message(f"[DDParcel] {line.rstrip()}", log)
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"CLI command failed with exit code {return_code}")


def _run_normalize(input_path: Path, mask_path: Path, output_path: Path, flip: int) -> None:
    from . import normalize as normalize_module

    argv_backup = sys.argv[:]
    try:
        sys.argv = [
            "normalize.py",
            "--input",
            input_path.as_posix(),
            "--mask",
            mask_path.as_posix(),
            "--output",
            output_path.as_posix(),
            "--flip",
            str(int(flip)),
        ]
        normalize_module.main()
    finally:
        sys.argv = argv_backup


def _run_ddsurfer_pred(
    output_dir: Path,
    subject_id: str,
    weights_dir: Path,
    log: Optional[Callable[[str], None]],
    *,
    device: str = "auto",
) -> None:
    from . import DDSurfer_Pred as ddsurfer_pred

    argv_backup = sys.argv[:]
    try:
        sys.argv = [
            "DDSurfer_Pred.py",
            "--in_dir",
            output_dir.as_posix(),
            "--out_dir",
            output_dir.as_posix(),
            "--weights_dir",
            weights_dir.as_posix(),
            "--subject_id",
            subject_id,
        ]
        options = ddsurfer_pred.options_parse()
    finally:
        sys.argv = argv_backup

    normalized_device = (device or "auto").strip().lower() or "auto"
    options.require_cuda = normalized_device == "cuda" or normalized_device.startswith("cuda:")
    if normalized_device == "cpu":
        options.no_cuda = True
        options.gpu = "-1"
    elif normalized_device in {"auto", "cuda"}:
        options.no_cuda = False
        options.gpu = "0"
    elif normalized_device.startswith("cuda:"):
        options.no_cuda = False
        options.gpu = normalized_device.split(":", 1)[1]
    else:
        raise ValueError("DDParcel device must be one of: auto, cpu, cuda, or cuda:N.")

    logger = logging.getLogger("ddparcel.ddsurfer")
    logger.setLevel(logging.INFO)
    local_handlers: list[logging.Handler] = []
    if log:
        class CallbackHandler(logging.Handler):
            def emit(self, record):
                log(record.getMessage())

        callback_handler = CallbackHandler()
        logger.addHandler(callback_handler)
        local_handlers.append(callback_handler)
    elif not logger.handlers:
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        logger.addHandler(stream_handler)
        local_handlers.append(stream_handler)

    try:
        _log_message(
            f"[DDParcel] Launching DDSurfer prediction for {subject_id} with weights from {weights_dir}",
            log,
        )
        ddsurfer_pred.fastsurfercnn(options.iname, options.oname, logger, options)
        _log_message(f"[DDParcel] DDSurfer prediction finished for {subject_id}", log)
    finally:
        for handler in local_handlers:
            logger.removeHandler(handler)


def _default_weights_dir(explicit: Optional[Path] = None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    return model_asset_path("parcellation/weights", repo_root=Path(__file__).resolve().parents[2])


def _registered_task_signature(config, weights_dir):
    bundle = config.registered_bundle
    sources = [getattr(bundle, name) for name in bundle.__dataclass_fields__ if name.endswith('_path')]
    sources += [config.native_mask, Path(__file__), Path(__file__).with_name('normalize.py'),
                Path(__file__).with_name('DDSurfer_Pred.py')]
    weights = sorted(path for path in weights_dir.rglob('*') if path.suffix in ('.pkl', '.pt', '.pth'))
    if not weights:
        raise FileNotFoundError(f'No DDParcel checkpoints in {weights_dir}')
    sources += weights
    registered_record = _read_manifest(bundle.manifest_path) or {}
    source_inputs = registered_record.get('native_bundle', {}).get('manifest', {}).get('source_files', {})
    return dict(version=1, subject_id=config.subject_id, flip=config.flip, device=config.device,
                source_inputs=source_inputs,
                inputs={str(path.resolve()): _artifact_record(path)['sha256'] for path in sources})


@scheduled_stage('parcellation', gpu=True)
def run_ddparcel_from_registered_bundle(
    config: DDParcelRegisteredConfig,
    log: Optional[Callable[[str], None]] = None,
) -> None:
    """Run the DDParcel task-specific tail starting from a cached registered DTI bundle."""

    subject_id = config.subject_id.strip()
    if not subject_id:
        raise ValueError("Subject ID must be provided.")
    if subject_id != config.registered_bundle.subject_id:
        raise ValueError("Subject ID does not match the provided registered bundle.")
    if not config.native_mask.is_file():
        raise FileNotFoundError(f"Native-space mask not found: {config.native_mask}")
    if not config.registered_bundle.inverse_transform_path.is_file():
        raise FileNotFoundError(f'DDParcel requires an inverse transform: {config.registered_bundle.inverse_transform_path}')

    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = config.registered_bundle

    nii_fa_reg_norm = output_dir / f"{subject_id}-dti-FractionalAnisotropy-Reg-NormMasked.nii.gz"
    nii_trace_reg_norm = output_dir / f"{subject_id}-dti-Trace-Reg-NormMasked.nii.gz"
    nii_min_eig_reg_norm = output_dir / f"{subject_id}-dti-MinEigenvalue-Reg-NormMasked.nii.gz"
    nii_mid_eig_reg_norm = output_dir / f"{subject_id}-dti-MidEigenvalue-Reg-NormMasked.nii.gz"

    mgz_wmparc_reg = output_dir / f"{subject_id}-DDSurfer-wmparc-Reg.mgz"
    nii_wmparc = output_dir / f"{subject_id}-DDSurfer-wmparc.nii.gz"
    weights_dir = _default_weights_dir(config.weights_dir)
    signature = _registered_task_signature(config, weights_dir)
    receipt = output_dir / 'ddparcel_preparation.json'
    outputs = [nii_fa_reg_norm, nii_trace_reg_norm, nii_min_eig_reg_norm, nii_mid_eig_reg_norm,
               mgz_wmparc_reg, nii_wmparc]
    if (config.force_recompute or _read_manifest(receipt) != signature
            or not _artifacts_current(output_dir, outputs)):
        _write_manifest(output_dir / 'ddparcel_qc.json', {'status': 'running', 'automatic_qc_passed': False})
        receipt.unlink(missing_ok=True)
        for path in outputs:
            path.unlink(missing_ok=True)
        _log_message('[DDParcel] Inputs/settings changed or output receipt missing; rebuilding task-specific products', log)

    _log_message(f"[DDParcel] Starting task-specific stage from cached registered bundle for {subject_id}", log)
    _log_message(f"[DDParcel] Registered bundle: {bundle.bundle_dir}", log)

    if not nii_mid_eig_reg_norm.exists():
        _run_normalize(bundle.fa_reg_path, bundle.mask_reg_path, nii_fa_reg_norm, config.flip)
        _run_normalize(bundle.trace_reg_path, bundle.mask_reg_path, nii_trace_reg_norm, config.flip)
        _run_normalize(bundle.min_eig_reg_path, bundle.mask_reg_path, nii_min_eig_reg_norm, config.flip)
        _run_normalize(bundle.mid_eig_reg_path, bundle.mask_reg_path, nii_mid_eig_reg_norm, config.flip)

    if not mgz_wmparc_reg.exists():
        _run_ddsurfer_pred(output_dir, subject_id, weights_dir, log, device=config.device)

    if not nii_wmparc.exists():
        cli_exe = resolve_cli_executable("resamplescalarvectordwivolume")
        if not bundle.inverse_transform_path.exists():
            raise FileNotFoundError(f"DDParcel requires the inverse transform for native labels: {bundle.inverse_transform_path}")
        else:
            cmd = [
                cli_exe,
                "--Reference",
                config.native_mask.as_posix(),
                "--transformationFile",
                bundle.inverse_transform_path.as_posix(),
                "--interpolation",
                "nn",
                mgz_wmparc_reg.as_posix(),
                nii_wmparc.as_posix(),
            ]
            _run_cli(cmd, log)

    if (
        nii_fa_reg_norm.exists()
        and nii_trace_reg_norm.exists()
        and nii_min_eig_reg_norm.exists()
        and nii_mid_eig_reg_norm.exists()
    ):
        from .qc import validate_outputs

        validate_outputs(subject_id, output_dir, config.native_mask)
        if signature != _registered_task_signature(config, weights_dir):
            _write_manifest(output_dir / 'ddparcel_qc.json', {'status': 'failed', 'automatic_qc_passed': False,
                'error': 'DDParcel inputs/settings changed during execution'})
            raise RuntimeError('DDParcel inputs/settings changed during execution')
        _save_artifacts(output_dir, outputs)
        _write_manifest(receipt, signature)
        _log_message(f"[DDParcel] Completed successfully for {subject_id}", log)
    else:
        raise RuntimeError(f"[DDParcel] Pipeline failed for {subject_id}")


def run_ddparcel(config: DDParcelConfig, log: Optional[Callable[[str], None]] = None) -> None:
    from .conversion.nhdr_write import nhdr_write
    from .conversion.nifti_write import nifti_write

    subject_id = config.subject_id.strip()
    if not subject_id:
        raise ValueError("Subject ID must be provided.")

    input_dir = config.input_root / subject_id / "T1w" / "Diffusion"
    output_dir = config.output_root / subject_id
    output_dir.mkdir(parents=True, exist_ok=True)

    atlas_t2 = config.atlas_t2.expanduser().resolve()
    if not atlas_t2.is_file():
        raise FileNotFoundError(f"Atlas T2 not found: {atlas_t2}")

    dwi = input_dir / "data.nii.gz"
    bval = input_dir / "bvals"
    bvec = input_dir / "bvecs"
    mask = input_dir / "nodif_brain_mask.nii.gz"

    for required in (dwi, bval, bvec, mask):
        if not required.exists():
            raise FileNotFoundError(f"Required input missing: {required}")

    nrrd_dwi = input_dir / f"{subject_id}.nhdr"
    nrrd_mask = input_dir / f"{subject_id}-mask.nhdr"

    nrrd_dti = output_dir / f"{subject_id}-dti.nhdr"
    nrrd_b0 = output_dir / f"{subject_id}-b0.nhdr"

    nrrd_fa = output_dir / f"{subject_id}-dti-FractionalAnisotropy.nhdr"
    nrrd_trace = output_dir / f"{subject_id}-dti-Trace.nhdr"
    nrrd_min_eig = output_dir / f"{subject_id}-dti-MinEigenvalue.nhdr"
    nrrd_mid_eig = output_dir / f"{subject_id}-dti-MidEigenvalue.nhdr"
    nrrd_max_eig = output_dir / f"{subject_id}-dti-MaxEigenvalue.nhdr"
    nrrd_md = output_dir / f"{subject_id}-dti-MeanDiffusivity.nhdr"

    nii_fa = output_dir / f"{subject_id}-dti-FractionalAnisotropy.nii.gz"
    nii_trace = output_dir / f"{subject_id}-dti-Trace.nii.gz"
    nii_min_eig = output_dir / f"{subject_id}-dti-MinEigenvalue.nii.gz"
    nii_mid_eig = output_dir / f"{subject_id}-dti-MidEigenvalue.nii.gz"
    nii_max_eig = output_dir / f"{subject_id}-dti-MaxEigenvalue.nii.gz"
    nii_md = output_dir / f"{subject_id}-dti-MeanDiffusivity.nii.gz"

    tfm = output_dir / f"{subject_id}-b0ToAtlasT2.tfm"
    tfm_inv = output_dir / f"{subject_id}-b0ToAtlasT2_Inverse.h5"

    nii_fa_reg = output_dir / f"{subject_id}-dti-FractionalAnisotropy-Reg.nii.gz"
    nii_trace_reg = output_dir / f"{subject_id}-dti-Trace-Reg.nii.gz"
    nii_min_eig_reg = output_dir / f"{subject_id}-dti-MinEigenvalue-Reg.nii.gz"
    nii_mid_eig_reg = output_dir / f"{subject_id}-dti-MidEigenvalue-Reg.nii.gz"
    nii_md_reg = output_dir / f"{subject_id}-dti-MeanDiffusivity-Reg.nii.gz"
    nii_max_eig_reg = output_dir / f"{subject_id}-dti-MaxEigenvalue-Reg.nii.gz"
    nii_mask_reg = output_dir / f"{subject_id}-mask-Reg.nii.gz"

    nii_fa_reg_norm = output_dir / f"{subject_id}-dti-FractionalAnisotropy-Reg-NormMasked.nii.gz"
    nii_trace_reg_norm = output_dir / f"{subject_id}-dti-Trace-Reg-NormMasked.nii.gz"
    nii_min_eig_reg_norm = output_dir / f"{subject_id}-dti-MinEigenvalue-Reg-NormMasked.nii.gz"
    nii_mid_eig_reg_norm = output_dir / f"{subject_id}-dti-MidEigenvalue-Reg-NormMasked.nii.gz"

    mgz_wmparc_reg = output_dir / f"{subject_id}-DDSurfer-wmparc-Reg.mgz"
    nii_wmparc = output_dir / f"{subject_id}-DDSurfer-wmparc.nii.gz"

    _log_message(f"[DDParcel] Starting pipeline for {subject_id}", log)

    if not nrrd_mask.exists():
        _log_message("[DDParcel] Converting NIfTI to NHDR", log)
        nhdr_write(dwi.as_posix(), bval.as_posix(), bvec.as_posix(), nrrd_dwi.as_posix())
        nhdr_write(mask.as_posix(), None, None, nrrd_mask.as_posix())

    # W2: one teem_dti.py process writes the tensor, the baseline and the six scalar maps (.nhdr: detached data).
    need_nifti = not nii_mid_eig.exists()
    if not nrrd_b0.exists() or need_nifti:
        _run_cli(
            native_tensor_command(
                resolve_pythonslicer_executable(), nrrd_dwi, nrrd_dti, nrrd_b0, "WLS",
                (
                    ("FractionalAnisotropy", nrrd_fa),
                    ("Trace", nrrd_trace),
                    ("MinEigenvalue", nrrd_min_eig),
                    ("MidEigenvalue", nrrd_mid_eig),
                    ("MaxEigenvalue", nrrd_max_eig),
                    ("MeanDiffusivity", nrrd_md),
                )),
            log,
        )

    if need_nifti:
        nifti_write(nrrd_fa.as_posix(), nrrd_fa.as_posix().replace(".nhdr", ""))
        nifti_write(nrrd_trace.as_posix(), nrrd_trace.as_posix().replace(".nhdr", ""))
        nifti_write(nrrd_min_eig.as_posix(), nrrd_min_eig.as_posix().replace(".nhdr", ""))
        nifti_write(nrrd_mid_eig.as_posix(), nrrd_mid_eig.as_posix().replace(".nhdr", ""))
        nifti_write(nrrd_max_eig.as_posix(), nrrd_max_eig.as_posix().replace(".nhdr", ""))
        nifti_write(nrrd_md.as_posix(), nrrd_md.as_posix().replace(".nhdr", ""))

    if not tfm.exists():
        cli_exe = resolve_cli_executable("brainsfit")
        cmd = [
            cli_exe,
            "--fixedVolume",
            atlas_t2.as_posix(),
            "--movingVolume",
            nrrd_b0.as_posix(),
            "--linearTransform",
            tfm.as_posix(),
            "--useRigid",
            "--useAffine",
        ]
        _run_cli(cmd, log)

    if not nii_mask_reg.exists():
        cli_exe = resolve_cli_executable("resamplescalarvectordwivolume")
        for input_path, output_path, interp in (
            (nii_fa, nii_fa_reg, "linear"),
            (nii_trace, nii_trace_reg, "linear"),
            (nii_min_eig, nii_min_eig_reg, "linear"),
            (nii_mid_eig, nii_mid_eig_reg, "linear"),
            (nii_md, nii_md_reg, "linear"),
            (nii_max_eig, nii_max_eig_reg, "linear"),
            (mask, nii_mask_reg, "nn"),
        ):
            cmd = [
                cli_exe,
                "-i",
                interp,
                input_path.as_posix(),
                "--Reference",
                atlas_t2.as_posix(),
                "--transformationFile",
                tfm.as_posix(),
                output_path.as_posix(),
            ]
            _run_cli(cmd, log)

    if not nii_mid_eig_reg_norm.exists():
        _run_normalize(nii_fa_reg, nii_mask_reg, nii_fa_reg_norm, config.flip)
        _run_normalize(nii_trace_reg, nii_mask_reg, nii_trace_reg_norm, config.flip)
        _run_normalize(nii_min_eig_reg, nii_mask_reg, nii_min_eig_reg_norm, config.flip)
        _run_normalize(nii_mid_eig_reg, nii_mask_reg, nii_mid_eig_reg_norm, config.flip)

    if not mgz_wmparc_reg.exists():
        weights_dir = _default_weights_dir(config.weights_dir)
        _run_ddsurfer_pred(output_dir, subject_id, weights_dir, log, device=config.device)

    if not nii_wmparc.exists():
        cli_exe = resolve_cli_executable("resamplescalarvectordwivolume")
        if not tfm_inv.exists():
            raise FileNotFoundError(f"DDParcel requires the inverse transform for native labels: {tfm_inv}")
        else:
            cmd = [
                cli_exe,
                "--Reference",
                mask.as_posix(),
                "--transformationFile",
                tfm_inv.as_posix(),
                "--interpolation",
                "nn",
                mgz_wmparc_reg.as_posix(),
                nii_wmparc.as_posix(),
            ]
            _run_cli(cmd, log)

    if (
        nii_fa_reg_norm.exists()
        and nii_trace_reg_norm.exists()
        and nii_min_eig_reg_norm.exists()
        and nii_mid_eig_reg_norm.exists()
    ):
        from .qc import validate_outputs

        validate_outputs(subject_id, output_dir, mask)
        _log_message(f"[DDParcel] Completed successfully for {subject_id}", log)
    else:
        raise RuntimeError(f"[DDParcel] Pipeline failed for {subject_id}")
