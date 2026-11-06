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
    candidates = model_asset_candidates(relative, repo_root=Path(__file__).resolve().parents[2],
                                        explicit_root=model_assets_dir)
    for candidate in candidates:
        if (candidate / required_child if required_child else candidate).exists():
            return candidate.resolve()
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
            model_config = json.loads((weights / 'manifest.json').read_text())
            expected_assets = json.loads((Path(__file__).resolve().parents[2] / 'models/assets_manifest.json').read_text())['asset_records']
            for filename in ('manifest.json', 'SHA256SUMS'):
                path = weights / filename
                if sha256(path) != expected_assets[f'surface/weights/{filename}']['sha256']:
                    raise ValueError(f'Model release metadata mismatch: {path}')
            checksums = {}
            for line in (weights / 'SHA256SUMS').read_text().splitlines():
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
    bundle = config.input_bundle or find_subject_root(config.subject_root, config.subject_id)
    command = [BACKEND_ROOT / 'run_ddsurfer_pipeline.py', '--subject', config.subject_id,
               '--output-root', output_dir / 'upstream', '--predict-mode', config.predict_mode,
               '--device', config.device, '--precision', config.precision,
               '--cpu-threads', str(config.cpu_threads), '--max-gpu-gib', str(config.max_gpu_gib),
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


def _run_registered(config, resources, output_dir, subject_dir):
    from .preprocessing import SubjectPreprocessingConfig, preprocess_subject
    from .prediction import PredictionStageConfig, predict_subject
    metrics = {name: subject_dir / f'{config.subject_id}-dti-{name}-Reg.nii.gz' for name in
               ('FractionalAnisotropy', 'MinEigenvalue', 'MidEigenvalue', 'MaxEigenvalue', 'MeanDiffusivity')}
    prepared = preprocess_subject(SubjectPreprocessingConfig(config.subject_id, metrics,
        subject_dir / f'{config.subject_id}-mask-Reg.nii.gz',
        subject_dir / f'{config.subject_id}-b0ToAtlasT2.tfm', output_dir,
        Path(resources['registration_atlas']), python_bin=config.python_bin))
    for hemi in ('left', 'right'):
        predict_subject(PredictionStageConfig(config.subject_id, hemi, config.predict_mode,
            config.device, 0.02, 2, 3, Path(resources['template_dir']), Path(resources['checkpoint_root']),
            prepared.preprocessed_dir.parent, output_dir / 'pred_results_SlicerDSpace',
            precision=config.precision, python_bin=config.python_bin,
            cpu_threads=config.cpu_threads, max_gpu_gib=config.max_gpu_gib))
    run_backend([BACKEND_ROOT / 'inference/native.py', '--subject', config.subject_id,
        '--data-root', prepared.preprocessed_dir.parent, '--pred-root', output_dir / 'pred_results_SlicerDSpace',
        '--predict-mode', config.predict_mode, '--output-dir',
        output_dir / 'pred_results_SlicerDSpace/native' / config.subject_id], python_bin=config.python_bin)
    if config.cleanup_intermediates:
        for path in (prepared.raw_dir, prepared.preprocessed_dir,
                     output_dir / 'pred_results_SlicerDSpace/mni' / config.subject_id):
            shutil.rmtree(path, ignore_errors=True)


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


def _run_shared_raw(config, resources, output_dir, manifest, previous, log):
    raw, registered = prepare_shared_registered_bundle(config, resources, log)
    paths = [getattr(registered, name) for name in registered.__dataclass_fields__
             if name.endswith('_path')]
    paths += [getattr(raw, name + '_path') for name in ('dwi', 'bval', 'bvec', 'mask')]
    paths += [Path(__file__), Path(__file__).with_name('preprocessing.py'),
              Path(__file__).with_name('prediction.py'), Path(__file__).with_name('backend_bridge.py'),
              Path(resources['checkpoint_root']) / 'manifest.json']
    paths += [Path(resources['checkpoint_root']) / filename for hemi in ('left', 'right')
              for kind, filename in resources['model_config']['checkpoint_files'][hemi].items()
              if kind == 'white' or config.predict_mode == 'all']
    paths += [Path(resources['template_dir']) / resources['model_config']['templates'][hemi]['file']
              for hemi in ('left', 'right')]
    manifest['input_mode'] = 'shared_raw'
    manifest['input_dwi'] = str(raw.dwi_path)
    manifest['input_sha256'] = {str(path.resolve()): sha256(path) for path in paths}
    manifest['shared_preprocessing'] = dict(registered_bundle=str(registered.bundle_dir),
        native_bundle=json.loads(registered.manifest_path.read_text())['native_bundle']['dir'],
        fit_method='wls', scalar_resampling_source='nrrd', atlas=resources['registration_atlas'],
        scope='Shared native tensor/scalars; trained surface atlas and five-channel normalization remain independent')
    keys = ('input_mode', 'input_sha256', 'subject_id', 'predict_mode', 'device', 'resolved_device',
            'precision', 'cpu_threads', 'max_gpu_gib', 'upstream_commit', 'runtime')
    outputs = previous.get('output_sha256', {})
    native = output_dir / 'pred_results_SlicerDSpace/native' / config.subject_id
    required = set()
    for hemi in ('lh', 'rh'):
        for kind in ('white', 'pial') if config.predict_mode == 'all' else ('white',):
            token = 'wm' if kind == 'white' else 'pial'
            required.update((str((native / f'{hemi}.{kind}.obj').resolve()),
                             str((native / f'{config.subject_id}_predicted_{token}_{hemi}.stl').resolve())))
    reusable = config.resume and previous.get('status') == 'completed' and required.issubset(outputs)
    if reusable and all(previous.get(key) == manifest.get(key) for key in keys):
        try:
            reusable = all(sha256(path) == digest for path, digest in outputs.items())
        except OSError:
            reusable = False
    else:
        reusable = False
    if reusable:
        if log:
            log('[DDSurfer] Reusing verified native surfaces and shared preprocessing')
    else:
        _run_registered(config, resources, output_dir, registered.bundle_dir)
    if manifest['input_sha256'] != {str(path.resolve()): sha256(path) for path in paths}:
        raise RuntimeError('Shared surface inputs changed during prediction')


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
        previous = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        previous = {}
    manifest, qc = _surface_records(config, resources, output_dir)
    products = []
    started, failure = time.monotonic(), None
    try:
        if config.execute:
            if not resources['available']:
                raise FileNotFoundError('DDSurfer resources invalid: ' + ', '.join(resources['missing']))
            from shared.environment import resolve_auto_torch_device
            config = replace(config, device=resolve_auto_torch_device(config.device, Path(python_executable(config.python_bin))))
            manifest['resolved_device'] = config.device
            if log:
                log(f'[DDSurfer] Requested {manifest["device"]}; resolved {config.device} in the surface runtime')
            manifest['status'] = 'running'
            manifest['runtime'] = runtime_info(config.python_bin)
            manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
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
            if (config.reuse_shared_preprocessing and not config.auto_mask
                    and (config.input_bundle is not None or inspected['mode'] == 'subject_root')):
                _run_shared_raw(config, resources, output_dir, manifest, previous, log)
            elif config.input_bundle is not None or inspected['mode'] == 'subject_root':
                manifest['input_mode'] = 'raw'
                if config.auto_mask:
                    raw = config.input_bundle or find_subject_root(config.subject_root, config.subject_id, generate_missing_mask=False)
                    if raw.mask_path is None or not Path(raw.mask_path).is_file():
                        from shared.brain_mask import BrainMaskConfig, generate_brain_mask
                        mask_path = output_dir / 'native_brain_mask.nii.gz'
                        mask_record = generate_brain_mask(raw.dwi_path, raw.bval_path, raw.bvec_path, mask_path,
                            config=BrainMaskConfig(backend='synthstrip', freesurfer_home=config.freesurfer_home,
                                brain_mask_device='cuda' if config.device.startswith('cuda') else 'cpu', threads=config.mask_threads), log=log)
                        manifest['mask_generation'] = mask_record
                        config = replace(config, input_bundle=replace(raw, mask_path=mask_path))
                    # Supply the native mask explicitly, bypassing the pinned
                    # upstream helper's independent CPU-only SynthStrip default.
                command, env = _raw_command(config, resources, output_dir)
                run_backend(command, python_bin=config.python_bin, env=env, log=log,
                            log_path=output_dir / 'backend.log')
                upstream = output_dir / 'upstream' / config.subject_id
                native_dir.mkdir(parents=True, exist_ok=True)
                for hemi in ('lh', 'rh'):
                    for kind in ('white', 'pial') if config.predict_mode == 'all' else ('white',):
                        shutil.copy2(upstream / 'ddsurfer' / f'{hemi}.{kind}.obj', native_dir / f'{hemi}.{kind}.obj')
                manifest['upstream_pipeline'] = json.loads((upstream / 'logs/pipeline.json').read_text())
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
                _run_registered(config, resources, output_dir, Path(inspected['subject_dir']))
            else:
                raise FileNotFoundError(f'No DDSurfer subject input: {config.subject_root}')
            products, checks = native_products(native_dir, config.subject_id, config.predict_mode)
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
