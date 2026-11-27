"""Slicer facade for the manifest-configured, unchanged DDSurfer backend."""
from __future__ import annotations

from shared.cohort_schedule import scheduled_stage

import json
import math
import os
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional

from shared import find_subject_root, inspect_subject_directory
from shared import model_assets as _model_assets
from shared.model_assets import model_asset_candidates
from .backend_bridge import BACKEND_ROOT, native_products, run_backend, runtime_info, sha256, upstream_record, python_executable

ATLAS_NAME = '100HCP-population-mean-T2-1mm.nii.gz'


@dataclass(frozen=True)
class SurfaceConfig:
    subject_id: str
    subject_root: Path
    output_dir: Path
    predict_mode: str = 'all'
    device: str = 'auto'
    template_dir: Optional[Path] = None
    checkpoint_root: Optional[Path] = None
    registration_atlas: Optional[Path] = None
    model_assets_dir: Optional[Path] = None
    cleanup_intermediates: bool = False
    execute: bool = False
    input_bundle: Optional[object] = None
    precision: str = 'fp32'
    python_bin: Optional[Path] = None
    slicer_home: Optional[Path] = None
    cpu_threads: Optional[int] = None  # None: stage CPU reservation, else 4
    max_gpu_gib: float = 24.
    resume: bool = False
    auto_mask: bool = False
    mask_threads: int = 4
    freesurfer_home: Optional[Path] = None
    reuse_shared_preprocessing: bool = False


@dataclass(frozen=True)
class SurfaceResult:
    subject_id: str
    output_dir: Path
    manifest_path: Path
    qc_path: Path
    generated_models: tuple[Path, ...]
    executed: bool


def _resolve_resource(explicit, env_key, relative, required_child=None, model_assets_dir=None):
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    if os.environ.get(env_key):
        return Path(os.environ[env_key]).expanduser().resolve()
    # Codex P1 (fix round): walk the same candidates as before (configured root, per-user default root, bundled copy,
    # checkout) so the environment doctor and the run agree. Opus P2 (bundled fallback): every candidate that is not
    # the user's own configured root passes the sha256 check of model_asset_path; the first that verifies is used.
    # A file that fails its sha256 is never used: when no candidate verifies, the first failure is raised.
    repo_root = Path(__file__).resolve().parents[2]
    relative_depth = len(Path(relative).parts)
    configured = bool(model_assets_dir is not None and str(model_assets_dir).strip()) or bool(
        os.environ.get(_model_assets.MODEL_ASSETS_ENV))
    manifest_records = _model_assets._load_asset_manifest_records(repo_root)
    bundled_roots = _model_assets.bundled_model_asset_roots(repo_root)
    first_error = partial = None
    for index, candidate in enumerate(model_asset_candidates(relative, repo_root=repo_root, explicit_root=model_assets_dir)):
        if not (candidate / required_child if required_child else candidate).exists():
            continue
        root = candidate.parents[relative_depth - 1]
        if index == 0 and configured:
            # The configured root is the user's own choice and is not verified (as in model_asset_path). A folder
            # left by an interrupted download is skipped when a verified copy exists later in the order.
            if candidate.is_dir() and _model_assets.incomplete_manifest_files(root, relative, manifest_records):
                partial = candidate
                continue
            return candidate.resolve()
        if _model_assets._has_verified_required_asset(root, (relative,), manifest_records=manifest_records):
            return candidate.resolve()
        if first_error is None:
            first_error = _model_assets._unverified_asset_error(
                root, relative, candidate, manifest_records, bundled=root in bundled_roots)
    if partial is not None:
        return partial.resolve()
    if first_error is not None:
        raise first_error
    return None


def resolve_surface_template_dir(template_dir=None, *, model_assets_dir=None):
    return _resolve_resource(template_dir, 'DDSURFER_TEMPLATE_DIR', 'surface/template',
                             'ddsurfer_lh_template.obj', model_assets_dir)


def resolve_surface_checkpoint_root(checkpoint_root=None, *, model_assets_dir=None):
    return _resolve_resource(checkpoint_root, 'DDSURFER_CHECKPOINT_DIR', 'surface/weights',
                             'manifest.json', model_assets_dir)


def resolve_surface_registration_atlas(registration_atlas=None, *, model_assets_dir=None):
    return _resolve_resource(registration_atlas, 'DDSURFER_REGISTRATION_ATLAS',
                             f'surface/template/{ATLAS_NAME}', model_assets_dir=model_assets_dir)


def inspect_surface_resources(config: SurfaceConfig) -> dict:
    template = resolve_surface_template_dir(config.template_dir, model_assets_dir=config.model_assets_dir)
    weights = resolve_surface_checkpoint_root(config.checkpoint_root, model_assets_dir=config.model_assets_dir)
    atlas = resolve_surface_registration_atlas(config.registration_atlas, model_assets_dir=config.model_assets_dir)
    missing, errors = [], []
    model_config = None
    if weights is None:
        missing.append('surface/weights/manifest.json')
    if template is None:
        missing.append('surface/template/ddsurfer_lh_template.obj')
    if atlas is None or not atlas.is_file():
        missing.append(f'surface/template/{ATLAS_NAME}')
    if weights is not None and template is not None:
        try:
            model_config = json.loads((weights / 'manifest.json').read_text(encoding='utf-8'))
            expected_assets = json.loads((Path(__file__).resolve().parents[2] / 'models/assets_manifest.json').read_text(encoding='utf-8'))['asset_records']
            for filename in ('manifest.json', 'SHA256SUMS'):
                path = weights / filename
                if sha256(path) != expected_assets[f'surface/weights/{filename}']['sha256']:
                    raise ValueError(f'Model release metadata mismatch: {path}')
            checksums = {}
            for line in (weights / 'SHA256SUMS').read_text(encoding='utf-8').splitlines():
                if line.strip():
                    digest, filename = line.split(maxsplit=1)
                    filename = filename.lstrip('*')
                    if Path(filename).name != filename or filename in checksums:
                        raise ValueError('Invalid checkpoint checksum entry')
                    checksums[filename] = digest
            for hemi in ('left', 'right'):
                entry = model_config['templates'][hemi]
                if Path(entry['file']).name != entry['file']:
                    raise ValueError('Invalid template filename')
                paths = [(template / entry['file'], entry['sha256'])]
                for kind in ('white', 'pial') if config.predict_mode == 'all' else ('white',):
                    filename = model_config['checkpoint_files'][hemi][kind]
                    if Path(filename).name != filename:
                        raise ValueError('Invalid checkpoint filename')
                    paths.append((weights / filename, checksums[filename]))
                for path, digest in paths:
                    if not path.is_file():
                        missing.append(str(path))
                    elif sha256(path) != digest:
                        errors.append(f'Checksum mismatch: {path}')
            if atlas is not None and atlas.is_file():
                expected = expected_assets[f'surface/template/{ATLAS_NAME}']['sha256']
                if sha256(atlas) != expected:
                    errors.append('DDSurfer requires the matching trained 1-mm atlas; the DDParcel atlas is not interchangeable')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(str(exc))
    return dict(available=not missing and not errors, missing=missing + errors,
                template_dir=str(template) if template else None,
                checkpoint_root=str(weights) if weights else None,
                registration_atlas=str(atlas) if atlas else None,
                upstream_commit=upstream_record()['commit'], model_config=model_config)


def _raw_command(config, resources, output_dir):
    from shared.environment import effective_surface_max_gpu_gib

    effective_max_gpu = effective_surface_max_gpu_gib(config.max_gpu_gib)
    bundle = config.input_bundle or find_subject_root(config.subject_root, config.subject_id)
    command = [BACKEND_ROOT / 'run_ddsurfer_pipeline.py', '--subject', config.subject_id,
               '--output-root', output_dir / 'upstream', '--predict-mode', config.predict_mode,
               '--device', config.device, '--precision', config.precision,
               '--cpu-threads', str(config.cpu_threads), '--max-gpu-gib', str(effective_max_gpu),
               '--checkpoint-root', resources['checkpoint_root'], '--template-dir', resources['template_dir']]
    for flag, attribute in (('dwi', 'dwi_path'), ('bval', 'bval_path'), ('bvec', 'bvec_path'), ('mask', 'mask_path')):
        path = getattr(bundle, attribute)
        if flag == 'mask' and path is None and config.auto_mask:
            continue
        if path is None or not Path(path).is_file():
            raise FileNotFoundError(f'DDSurfer requires a prepared {flag}; run input preparation first')
        command.extend([f'--{flag}', path])
    if config.auto_mask:
        command.extend(['--auto-mask', '--mask-threads', str(config.mask_threads)])
        if config.freesurfer_home is not None:
            command.extend(['--freesurfer-home', config.freesurfer_home])
    if config.resume:
        command.append('--resume')
    if not config.cleanup_intermediates:
        command.append('--keep-cache')
    home = config.slicer_home or os.environ.get('DTI_SLICER_PATH') or os.environ.get('SLICER_HOME')
    if home is None:
        from shared.pipeline import resolve_launcher_executable
        home = Path(resolve_launcher_executable()).parent
    # N9: a separately deployed backend cannot find shared/teem_dti.py from its own location.
    env = dict(DTI_SLICER_PATH=str(home), DTI_REFERENCE_IMAGE=resources['registration_atlas'])
    if not os.environ.get('SLICERDSPACE_TEEM_DTI_SCRIPT'):  # T4: a user value wins
        env['SLICERDSPACE_TEEM_DTI_SCRIPT'] = str(Path(__file__).resolve().parents[2] / 'shared/teem_dti.py')
    return command, env


def _merge_execution_devices(records):
    """One ``{part: device}`` map from prediction records (F1); None for records from before the field existed."""
    merged = {}
    for record in records:
        for part, device in ((record or {}).get('execution_devices') or {}).items():
            merged.setdefault(part, [])
            if device not in merged[part]:
                merged[part].append(device)
    return {part: ','.join(devices) for part, devices in merged.items()} or None


def _execution_device_note(requested, resolved_device, devices):
    """The log line for a run whose volume network did not run on the resolved device, else ''."""
    vf_device = (devices or {}).get('vf_net')
    if not vf_device or vf_device == str(resolved_device).split(':')[0]:
        return ''
    reason = 'no MPS Conv3d in this torch build' if str(resolved_device).startswith('mps') else 'see the backend log'
    return f'[DDSurfer] Requested {requested}; resolved {resolved_device} (volume network on {vf_device}: {reason})'


def _run_registered(config, resources, output_dir, subject_dir, *, native_dti_dir=None, native_mask_path=None):
    from .preprocessing import SubjectPreprocessingConfig, preprocess_subject
    from .prediction import PredictionStageConfig, predict_subject
    metrics = {name: subject_dir / f'{config.subject_id}-dti-{name}-Reg.nii.gz' for name in
               ('FractionalAnisotropy', 'MinEigenvalue', 'MidEigenvalue', 'MaxEigenvalue', 'MeanDiffusivity')}
    prepared = preprocess_subject(SubjectPreprocessingConfig(config.subject_id, metrics,
        subject_dir / f'{config.subject_id}-mask-Reg.nii.gz',
        subject_dir / f'{config.subject_id}-b0ToAtlasT2.tfm', output_dir,
        Path(resources['registration_atlas']), python_bin=config.python_bin))
    records = []
    hemi_threads = config.cpu_threads
    parallel_hemis = False
    if str(config.device).strip().lower().startswith('cpu') and not os.environ.get('SLICERDSPACE_STAGE_CPUS'):
        import sys as _sys
        if _sys.platform.startswith('linux') and config.cpu_threads == 4:
            cpus = os.cpu_count() or 1
            if cpus >= 32:
                from shared.hardware_adapt import host_total_ram_gib
                tot_gib = host_total_ram_gib()
                if tot_gib is not None and tot_gib > 48.0:
                    parallel_hemis = True
                    hemi_threads = min(36, max(8, cpus // 4))

    def _run_hemi(hemi):
        predicted = predict_subject(PredictionStageConfig(config.subject_id, hemi, config.predict_mode,
            config.device, 0.02, 2, 3, Path(resources['template_dir']), Path(resources['checkpoint_root']),
            prepared.preprocessed_dir.parent, output_dir / 'pred_results_SlicerDSpace',
            precision=config.precision, python_bin=config.python_bin,
            cpu_threads=hemi_threads, max_gpu_gib=config.max_gpu_gib))
        hemi_recs = []
        for mesh in (predicted.wm_mesh_path, predicted.pial_mesh_path):  # read before cleanup deletes them (F1)
            try:
                hemi_recs.append(json.loads(Path(mesh).with_suffix('.json').read_text(encoding='utf-8')))
            except (OSError, ValueError, TypeError):
                pass
        return hemi_recs

    if parallel_hemis:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2) as pool:
            for hemi_recs in pool.map(_run_hemi, ('left', 'right')):
                records.extend(hemi_recs)
    else:
        for hemi in ('left', 'right'):
            records.extend(_run_hemi(hemi))
    native_cmd = [
        BACKEND_ROOT / 'inference/native.py',
        '--subject', config.subject_id,
        '--data-root', prepared.preprocessed_dir.parent,
        '--pred-root', output_dir / 'pred_results_SlicerDSpace',
        '--predict-mode', config.predict_mode,
        '--output-dir', output_dir / 'pred_results_SlicerDSpace/native' / config.subject_id,
    ]
    if native_dti_dir is not None and native_mask_path is not None:
        refine_threads = max(int(config.cpu_threads or 4), min(16, os.cpu_count() or 4))
        native_cmd += [
            '--refine',
            '--dti-dir', Path(native_dti_dir),
            '--mask', Path(native_mask_path),
            '--threads', str(refine_threads),
        ]
    run_backend(native_cmd, python_bin=config.python_bin)
    if config.cleanup_intermediates:
        for path in (prepared.raw_dir, prepared.preprocessed_dir,
                     output_dir / 'pred_results_SlicerDSpace/mni' / config.subject_id):
            shutil.rmtree(path, ignore_errors=True)
    return _merge_execution_devices(records)


def uses_shared_raw_route(config):
    """True when ``run_surface`` takes the shared raw route for a prepared input bundle."""
    return bool(config.reuse_shared_preprocessing and not config.auto_mask and config.input_bundle is not None)


def prepare_shared_registered_bundle(config, resources, log=None):
    """Obtain the raw bundle and the shared registered bundle (WLS, NRRD scalars, reuse).

    The cohort orchestrator calls this before ``run_surface``, and only in cohort mode, so that ``native_dti``
    and ``registered_dti`` are admitted as their own stages (S1.2), not inside the ``surface`` admission.
    """
    from shared.pipeline import ensure_registered_dti_bundle

    raw = config.input_bundle or find_subject_root(config.subject_root, config.subject_id)
    registered = ensure_registered_dti_bundle(raw, Path(resources['registration_atlas']),
        fit_method='wls', reuse=True, log=log, scalar_format='nrrd')
    return raw, registered


SURFACE_PIPELINE_CONTRACT_VERSION = 2


def _normalized_surface_input_hashes(mapping) -> list[tuple[str, str]]:
    """Compare surface inputs by filename + content hash, ignoring `.py` source files."""
    if not isinstance(mapping, dict):
        return []
    entries = [
        (os.path.normcase(Path(str(path)).name), str(digest))
        for path, digest in mapping.items()
        if Path(str(path)).suffix.lower() != '.py'
    ]
    return sorted(entries)


def _surface_outputs_match(recorded_outputs, required_paths: list[Path]) -> bool:
    """Verify every required OBJ/STL surface matches its recorded SHA-256 (relocation-safe)."""
    if not isinstance(recorded_outputs, dict) or len(recorded_outputs) < len(required_paths):
        return False
    unused = dict(recorded_outputs)
    for path in required_paths:
        if not path.is_file() or path.is_symlink():
            return False
        resolved = str(path.resolve())
        key = resolved if resolved in unused else next(
            (k for k in unused if os.path.normcase(Path(str(k)).name) == os.path.normcase(path.name)),
            None,
        )
        if key is None:
            return False
        if sha256(path) != unused.pop(key):
            return False
    return True


def _run_shared_raw(config, resources, output_dir, manifest, previous, log):
    raw, registered = prepare_shared_registered_bundle(config, resources, log)
    paths = [getattr(registered, name) for name in registered.__dataclass_fields__
             if name.endswith('_path')]
    paths += [getattr(raw, name + '_path') for name in ('dwi', 'bval', 'bvec', 'mask')]
    paths += [Path(resources['checkpoint_root']) / 'manifest.json']
    paths += [Path(resources['checkpoint_root']) / filename for hemi in ('left', 'right')
              for kind, filename in resources['model_config']['checkpoint_files'][hemi].items()
              if kind == 'white' or config.predict_mode == 'all']
    paths += [Path(resources['template_dir']) / resources['model_config']['templates'][hemi]['file']
              for hemi in ('left', 'right')]
    manifest['input_mode'] = 'shared_raw'
    manifest['pipeline_contract_version'] = SURFACE_PIPELINE_CONTRACT_VERSION
    manifest['input_dwi'] = str(raw.dwi_path)
    manifest['input_sha256'] = {str(path.resolve()): sha256(path) for path in paths}
    manifest['shared_preprocessing'] = dict(registered_bundle=str(registered.bundle_dir),
        native_bundle=json.loads(registered.manifest_path.read_text(encoding='utf-8'))['native_bundle']['dir'],
        fit_method='wls', scalar_resampling_source='nrrd', atlas=resources['registration_atlas'],
        scope='Shared native tensor/scalars; trained surface atlas and five-channel normalization remain independent')
    # Only scientifically meaningful contract fields gate reuse; execution knobs
    # (device, cpu_threads, max_gpu_gib, runtime install path) are recorded as
    # provenance without forcing recomputation.
    scientific_keys = ('input_mode', 'subject_id', 'predict_mode', 'precision', 'upstream_commit')
    outputs = previous.get('output_sha256', {})
    native = output_dir / 'pred_results_SlicerDSpace/native' / config.subject_id
    required_paths = []
    for hemi in ('lh', 'rh'):
        for kind in ('white', 'pial') if config.predict_mode == 'all' else ('white',):
            token = 'wm' if kind == 'white' else 'pial'
            required_paths.extend((
                native / f'{hemi}.{kind}.obj',
                native / f'{config.subject_id}_predicted_{token}_{hemi}.stl',
            ))
    contract_ok = (
        previous.get('pipeline_contract_version', SURFACE_PIPELINE_CONTRACT_VERSION)
        == SURFACE_PIPELINE_CONTRACT_VERSION
    )
    inputs_ok = (
        bool(manifest['input_sha256'])
        and _normalized_surface_input_hashes(previous.get('input_sha256'))
        == _normalized_surface_input_hashes(manifest['input_sha256'])
    )
    reusable = (
        config.resume
        and previous.get('status') == 'completed'
        and contract_ok
        and all(previous.get(key) == manifest.get(key) for key in scientific_keys)
        and inputs_ok
        and _surface_outputs_match(outputs, required_paths)
    )
    if reusable:
        if log:
            log('[DDSurfer] Reusing verified native surfaces and shared preprocessing')
        if previous.get('execution_devices'):
            manifest['execution_devices'] = previous['execution_devices']
    else:
        native_bundle_dir = Path(manifest['shared_preprocessing']['native_bundle'])
        native_mask = native_bundle_dir / 'Mask.nhdr'
        if not native_mask.is_file():
            native_mask = raw.mask_path
        devices = _run_registered(
            config, resources, output_dir, registered.bundle_dir,
            native_dti_dir=native_bundle_dir,
            native_mask_path=native_mask,
        )
        if isinstance(devices, dict):
            manifest['execution_devices'] = devices
    if manifest['input_sha256'] != {str(path.resolve()): sha256(path) for path in paths}:
        raise RuntimeError('Shared surface inputs changed during prediction')
    return reusable


def resolved_cpu_threads(config):
    """Explicit thread budget, else the stage CPU reservation (``SLICERDSPACE_STAGE_CPUS``), else 4."""
    if config.cpu_threads is not None:
        return config.cpu_threads
    from shared.numeric_threads import reserved_stage_cpus
    return reserved_stage_cpus() or 4


def _surface_records(config, resources, output_dir):
    """The base manifest and QC records that ``run_surface`` starts from."""
    manifest = dict(kind='ddsurfer_surface', version=3, subject_id=config.subject_id,
        subject_root=str(config.subject_root.expanduser().resolve()), output_dir=str(output_dir),
        input_dwi=str(config.input_bundle.dwi_path) if config.input_bundle is not None else None,
        predict_mode=config.predict_mode, device=config.device, precision=config.precision,
        cpu_threads=resolved_cpu_threads(config), max_gpu_gib=config.max_gpu_gib,
        resume=config.resume, auto_mask=config.auto_mask, mask_threads=config.mask_threads,
        upstream_commit=resources['upstream_commit'], resources=resources,
        executed=config.execute, status='planned', outputs={})
    qc = dict(subject_id=config.subject_id, status='planned', executed=False, model_count=0,
              elapsed_seconds=0., error=None, human_review='pending')
    return manifest, qc


def _mark_surface_failed(manifest, qc, exc, executed):
    manifest['status'] = 'failed'
    qc.update(status='failed', executed=executed, error=str(exc), automatic_qc_passed=False)


def _write_surface_records(manifest_path, qc_path, manifest, qc, started):
    qc['elapsed_seconds'] = time.monotonic() - started
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    qc_path.write_text(json.dumps(qc, indent=2) + '\n', encoding='utf-8')


def record_surface_failure(config, resources, exc, started):
    """Write the failed surface records for a failure outside ``run_surface`` (R1) and return the error to raise.

    It writes the same files and fields as the failure handler of ``run_surface``.
    """
    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, qc = _surface_records(config, resources, output_dir)
    _mark_surface_failed(manifest, qc, exc, config.execute)
    _write_surface_records(output_dir / 'surface_manifest.json', output_dir / 'surface_qc.json', manifest, qc, started)
    return RuntimeError(f'DDSurfer surface execution failed: {exc}')


@scheduled_stage('surface', gpu=True)
def run_surface(config: SurfaceConfig, *, log: Optional[Callable[[str], None]] = None) -> SurfaceResult:
    if not config.subject_id.strip() or Path(config.subject_id).name != config.subject_id or config.subject_id in ('.', '..', 'fsaverage'):
        raise ValueError('subject_id must be a nonempty directory name')
    if config.predict_mode not in ('all', 'wm') or config.precision not in ('auto', 'bf16', 'fp32'):
        raise ValueError('Invalid prediction mode or precision')
    config = replace(config, cpu_threads=resolved_cpu_threads(config))
    if config.cpu_threads < 1 or config.mask_threads < 1:
        raise ValueError('DDSurfer thread budgets must be positive')
    if not math.isfinite(config.max_gpu_gib) or config.max_gpu_gib <= 0:
        raise ValueError('DDSurfer GPU budget must be finite and positive')
    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path, qc_path = output_dir / 'surface_manifest.json', output_dir / 'surface_qc.json'
    resources = inspect_surface_resources(config)
    try:
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        previous = {}
    manifest, qc = _surface_records(config, resources, output_dir)
    products = []
    started, failure = time.monotonic(), None
    try:
        if config.execute:
            if not resources['available']:
                raise FileNotFoundError('DDSurfer resources invalid: ' + ', '.join(resources['missing']))
            from shared.environment import (
                effective_surface_max_gpu_gib,
                resolve_auto_torch_device,
                surface_gpu_memory_decision,
            )
            requested_surface_device = str(config.device or 'auto').strip().lower()
            surface_py = Path(python_executable(config.python_bin))
            resolved_surface_device = resolve_auto_torch_device(config.device, surface_py)
            if requested_surface_device in {'', 'auto'} and resolved_surface_device.startswith('cuda'):
                gpu_action, gpu_note = surface_gpu_memory_decision(
                    'auto', surface_py, resolved_device=resolved_surface_device
                )
                if gpu_action == 'cpu':
                    resolved_surface_device = 'cpu'
                    manifest['gpu_memory_fallback_note'] = gpu_note
                    if log:
                        log(f'[DDSurfer] {gpu_note}')
            effective_max_gpu = effective_surface_max_gpu_gib(config.max_gpu_gib)
            config = replace(config, device=resolved_surface_device, max_gpu_gib=effective_max_gpu)
            manifest['resolved_device'] = config.device
            manifest['effective_max_gpu_gib'] = config.max_gpu_gib
            if log:
                log(f'[DDSurfer] Requested {manifest["device"]}; resolved {config.device} in the surface runtime')
            manifest['status'] = 'running'
            manifest['runtime'] = runtime_info(config.python_bin)
            manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
            inspected = inspect_subject_directory(config.subject_root, config.subject_id)
            if inspected['mode'] == 'missing':
                for candidate in (config.subject_root, config.subject_root / config.subject_id):
                    required = [candidate / f'{config.subject_id}-dti-{name}-Reg.nii.gz' for name in
                                ('FractionalAnisotropy', 'MinEigenvalue', 'MidEigenvalue', 'MaxEigenvalue', 'MeanDiffusivity')]
                    required += [candidate / f'{config.subject_id}-mask-Reg.nii.gz', candidate / f'{config.subject_id}-b0ToAtlasT2.tfm']
                    if all(p.is_file() for p in required):
                        inspected = dict(mode='legacy_registered', subject_dir=str(candidate))
                        break
            native_dir = output_dir / 'pred_results_SlicerDSpace/native' / config.subject_id
            reused_shared_raw = False
            if (config.reuse_shared_preprocessing and not config.auto_mask
                    and (config.input_bundle is not None or inspected['mode'] == 'subject_root')):
                reused_shared_raw = bool(_run_shared_raw(config, resources, output_dir, manifest, previous, log))
            elif config.input_bundle is not None or inspected['mode'] == 'subject_root':
                manifest['input_mode'] = 'raw'
                if config.auto_mask:
                    raw = config.input_bundle or find_subject_root(config.subject_root, config.subject_id, generate_missing_mask=False)
                    if raw.mask_path is None or not Path(raw.mask_path).is_file():
                        from shared.brain_mask import BrainMaskConfig, generate_brain_mask
                        mask_path = output_dir / 'native_brain_mask.nii.gz'
                        mask_record = generate_brain_mask(raw.dwi_path, raw.bval_path, raw.bvec_path, mask_path,
                            config=BrainMaskConfig(backend='pnl_cnn',  # W5 (D2): the one mask generator
                                brain_mask_device=('cuda' if config.device.startswith('cuda') else ('mps' if config.device.startswith('mps') else 'cpu')),
                                threads=config.mask_threads), log=log)
                        manifest['mask_generation'] = mask_record
                        config = replace(config, input_bundle=replace(raw, mask_path=mask_path))
                    # Supply the native mask explicitly, bypassing the pinned
                    # upstream helper's independent CPU-only default.
                command, env = _raw_command(config, resources, output_dir)
                run_backend(command, python_bin=config.python_bin, env=env, log=log,
                            log_path=output_dir / 'backend.log')
                upstream = output_dir / 'upstream' / config.subject_id
                native_dir.mkdir(parents=True, exist_ok=True)
                for hemi in ('lh', 'rh'):
                    for kind in ('white', 'pial') if config.predict_mode == 'all' else ('white',):
                        shutil.copy2(upstream / 'ddsurfer' / f'{hemi}.{kind}.obj', native_dir / f'{hemi}.{kind}.obj')
                manifest['upstream_pipeline'] = json.loads((upstream / 'logs/pipeline.json').read_text(encoding='utf-8'))
                devices = _merge_execution_devices(manifest['upstream_pipeline'].get('surfaces', {}).values())
                if devices:
                    manifest['execution_devices'] = devices
                manifest['outputs']['native_dti'] = str(upstream / 'dti')
            elif inspected['mode'] != 'missing':
                manifest['input_mode'] = 'registered'
                directory = Path(inspected['subject_dir'])
                paths = [directory / f'{config.subject_id}-dti-{name}-Reg.nii.gz' for name in
                         ('FractionalAnisotropy', 'MinEigenvalue', 'MidEigenvalue', 'MaxEigenvalue', 'MeanDiffusivity')]
                paths += [directory / f'{config.subject_id}-mask-Reg.nii.gz',
                          directory / f'{config.subject_id}-b0ToAtlasT2.tfm',
                          Path(resources['registration_atlas']), Path(resources['checkpoint_root']) / 'manifest.json']
                paths += [Path(resources['checkpoint_root']) / filename for hemi in ('left', 'right')
                          for kind, filename in resources['model_config']['checkpoint_files'][hemi].items()
                          if kind == 'white' or config.predict_mode == 'all']
                paths += [Path(resources['template_dir']) / resources['model_config']['templates'][hemi]['file']
                          for hemi in ('left', 'right')]
                manifest['input_sha256'] = {str(p.resolve()): sha256(p) for p in paths}
                devices = _run_registered(config, resources, output_dir, Path(inspected['subject_dir']))
                if isinstance(devices, dict):
                    manifest['execution_devices'] = devices
            else:
                raise FileNotFoundError(f'No DDSurfer subject input: {config.subject_root}')
            note = _execution_device_note(manifest['device'], config.device, manifest.get('execution_devices'))
            if note and log:
                log(note)
            products, checks = native_products(
                native_dir, config.subject_id, config.predict_mode, export_stl=not reused_shared_raw
            )
            refinement_record = None
            for ref_cand in (
                native_dir.parent / 'logs' / 'lh.white.json',
                output_dir / 'upstream' / config.subject_id / 'logs' / 'lh.white.json',
            ):
                if ref_cand.is_file():
                    try:
                        rec = json.loads(ref_cand.read_text(encoding='utf-8')).get('surface_refinement')
                        if isinstance(rec, dict):
                            refinement_record = rec
                            break
                    except (OSError, ValueError, TypeError):
                        pass
            if refinement_record is None and reused_shared_raw and isinstance(previous.get('surface_refinement'), dict):
                refinement_record = previous['surface_refinement']
            if refinement_record is not None:
                manifest['surface_refinement'] = refinement_record
                qc['surface_refinement'] = refinement_record
            qc.update(status='completed', executed=True, model_count=len(products), surfaces=checks,
                      automatic_qc_passed=all(c['finite'] and c['watertight'] and c['euler_number'] == 2 for c in checks.values()))
            manifest['outputs']['models'] = list(map(str, products))
            manifest['outputs']['indexed_RAS_models'] = [c['indexed_master'] for c in checks.values()]
            manifest['output_sha256'] = {str(p.resolve()): sha256(p) for p in
                                        [*products, *(Path(c['indexed_master']) for c in checks.values())]}
            manifest['status'] = 'completed'
    except (KeyboardInterrupt, SystemExit) as exc:
        manifest['status'] = 'cancelled'
        qc.update(status='cancelled', executed=config.execute, error=str(exc), automatic_qc_passed=False)
        raise
    except Exception as exc:
        failure = exc
        _mark_surface_failed(manifest, qc, exc, config.execute)
    finally:
        _write_surface_records(manifest_path, qc_path, manifest, qc, started)
    if failure is not None:
        raise RuntimeError(f'DDSurfer surface execution failed: {failure}') from failure
    if log:
        log(f'[Surface] Manifest: {manifest_path}')
    return SurfaceResult(config.subject_id, output_dir, manifest_path, qc_path, tuple(products), config.execute)
