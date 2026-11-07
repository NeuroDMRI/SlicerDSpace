"""Validated, isolated experiment plans using the subject CLI's parameter schema."""

from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import io
import json
from functools import lru_cache
from pathlib import Path
import re
import shlex
import uuid


PARAMETERS = frozenset('''targets skip_targets all with_validation no_resume
brain_mask_backend brain_mask_device pnl_mask_route preprocessing_backend preprocessing_force
preprocessing_dipy_denoise_method preprocessing_skip_gibbs preprocessing_skip_motion
preprocessing_phase_encoding_direction preprocessing_readout_time preprocessing_eddy_threads preprocessing_eddy_slm
tractography_backend ukf_parameters bundle_backend bundle_tractography_backend connectome_backend
max_parallel_models wmql_query_preset wmql_query_selection select_streamlines dti_min_length_mm dti_max_length_mm
mrtrix_nthreads mrtrix_seed tckgen_seeding micro_fit_method microstructure_stages run_free_water run_noddi run_dki
run_ifw ifw_echo_time_ms ifw_t2_gm_ms ifw_t2_csf_ms
dki_fit_method microstructure_free_water_chunk_size microstructure_free_water_engine microstructure_free_water_n_jobs
microstructure_free_water_vox_per_chunk microstructure_free_water_max_fraction_for_tissue
postprocess_mode postprocess_atlases surface_precision surface_cpu_threads surface_max_gpu_gib
ddseg_device surface_device compress_mrtrix_dense tractcloud_device tractcloud_batch_size'''.split())


@lru_cache(maxsize=1)
def cli_parser():
    from shared.run_subject_pipeline_cli import build_parser
    parser = build_parser()
    parser.allow_abbrev = False
    return parser


def parse_cli(args):
    try:
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            return cli_parser().parse_args(args)
    except SystemExit as exc:
        raise ValueError(errors.getvalue().strip() or 'Invalid CLI parameters') from exc


def parameter_args(value):
    """Accept data arguments, never a shell command or execution/installation flags."""
    args = shlex.split(value) if isinstance(value, str) else value
    if not isinstance(args, list) or not all(isinstance(v, str) and '\0' not in v for v in args):
        raise ValueError('Parameter arguments must be a string or a JSON array of strings')
    actions = cli_parser()._option_string_actions
    for token in args:
        if token.startswith('--') or token in ('-h', '-?', '--'):
            flag = token.split('=', 1)[0]
            action = actions.get(flag)
            if action is None or action.dest not in PARAMETERS:
                raise ValueError(f'{flag} is not an experiment parameter. Configure paths in Environment; run/output settings are managed by the experiment runner.')
    parse_cli(args)
    return list(args)


def variants_from_json(text):
    records = json.loads(text) if isinstance(text, str) else text
    if not isinstance(records, list) or not 1 <= len(records) <= 64:
        raise ValueError('Provide 1 to 64 named experiment variants')
    variants, names = [], set()
    for item in records:
        if not isinstance(item, dict) or set(item) - {'name', 'args'}:
            raise ValueError('Each variant has only name and args fields')
        name = item.get('name', '')
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', name):
            raise ValueError('Variant names use 1-64 letters, numbers, dots, hyphens or underscores, starting with a letter/number')
        reserved = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}
        if name.endswith('.') or name.split('.', 1)[0].upper() in reserved:
            raise ValueError('Variant name is not portable to Windows: ' + name)
        if name.casefold() in names:
            raise ValueError('Duplicate variant name: ' + name)
        names.add(name.casefold())
        variants.append(dict(name=name, args=parameter_args(item.get('args', []))))
    return variants


def make_request(*, base_args, subject_root, subject_id, output_root, run_name, scope='subject',
                 cohort_root='', jobs=1, options='', variants='[{"name":"baseline","args":[]}]'):
    if scope not in {'subject', 'cohort'} or not 1 <= int(jobs) <= 8:
        raise ValueError('Choose subject/cohort scope and 1-8 concurrent subjects')
    campaign = re.sub(r'[^A-Za-z0-9_.-]', '_', run_name or 'experiment').strip('.')[:48] or 'experiment'
    campaign += '_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
    return dict(kind='slicerddsurfer_experiment_request', version=1, base_args=list(base_args),
                subject_root=str(subject_root), subject_id=subject_id, scope=scope, cohort_root=str(cohort_root),
                output_root=str(Path(output_root).expanduser().absolute() / 'experiments' / campaign),
                jobs=int(jobs), options=parameter_args(options), variants=variants_from_json(variants))


def compile_plan(request, *, python_executable, pipeline_script):
    from scripts.run_batch_pipeline import BatchSubject, discover_subjects
    if request.get('kind') != 'slicerddsurfer_experiment_request' or request.get('version') != 1:
        raise ValueError('Unsupported experiment request')
    jobs = request.get('jobs', 1)
    if type(jobs) is not int or not 1 <= jobs <= 8:
        raise ValueError('Concurrent subjects must be an integer in [1, 8]')
    scope = request.get('scope')
    if scope == 'cohort':
        if not str(request.get('cohort_root', '')).strip():
            raise ValueError('Cohort root is required')
        subjects, skipped = discover_subjects(Path(request['cohort_root']), arrange_inputs='none')
    elif scope == 'subject':
        if not str(request.get('subject_root', '')).strip():
            raise ValueError('Subject root is required')
        subject = Path(request['subject_root']).expanduser().resolve()
        if not subject.is_dir():
            raise ValueError('Subject root must be an existing directory')
        subject_id = request.get('subject_id', '')
        if not isinstance(subject_id, str) or not subject_id.strip():
            raise ValueError('Subject ID is required')
        subjects, skipped = [BatchSubject(subject_id, subject, 'GUI', 'subject')], []
    else:
        raise ValueError('Experiment scope must be subject or cohort')
    if not subjects:
        raise ValueError('No prepared subjects found; import inputs before running a cohort')
    identifiers = [s.subject_id.casefold() for s in subjects]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('Duplicate subject IDs in cohort')
    if any(any(ch in s.subject_id for ch in '/\\\0\r\n') or s.subject_id in {'.','..'} for s in subjects):
        raise ValueError('Subject IDs must not contain path separators or control characters')
    variants = variants_from_json(request['variants'])
    options = parameter_args(request.get('options', []))
    base = list(request['base_args'])
    forbidden = {'--help', '-h', '--execute', '--plan', '--doctor', '--skip-environment-check',
                 '--ignore-environment-blockers', '--save-user-config', '--save-dependency-config',
                 '--print-user-config', '--user-config', '--dependency-config'}
    # Base arguments come from the existing GUI serializer, with identity/output
    # and lifecycle flags removed; recheck imported requests as well.
    forbidden.update(('--subject-root', '--subject-id', '--output-root', '--run-name'))
    if not all(isinstance(arg, str) for arg in base) or any(arg.split('=', 1)[0] in forbidden for arg in base):
        raise ValueError('Base arguments contain reserved identity or execution options')
    parse_cli(base)
    output = Path(request['output_root']).expanduser().resolve()
    if output.exists():
        raise ValueError('Experiment output directory already exists; create a new campaign')
    if len(subjects) * len(variants) > 4096:
        raise ValueError('Limit each campaign to 4096 jobs; split larger studies into separate campaigns')
    from shared.run_subject_pipeline_cli import resolve_analysis_options, _resolve_microstructure_stages
    from shared.user_config import explicit_cli_destinations
    templates = []
    # Parameters are shared across subjects. Validate each variant once, before
    # materialising paths, instead of rebuilding the parser for every job.
    for variant in variants:
        args = [*base, *options, *variant['args'], '--no-user-config', '--strict-environment']
        resolved = parse_cli(args)
        targets = resolve_analysis_options(resolved, explicit_dests=explicit_cli_destinations(cli_parser(), args))
        if resolved.select_streamlines < 1 or resolved.dti_min_length_mm < 0 or resolved.dti_max_length_mm <= resolved.dti_min_length_mm:
            raise ValueError('Invalid streamline count or DTI length interval')
        effective = dict(targets=list(targets), tractography=resolved.tractography_backend,
                         bundles=resolved.bundle_backend, bundle_source=resolved.bundle_tractography_backend,
                         microstructure=list(_resolve_microstructure_stages(resolved)),
                         fit_method=resolved.micro_fit_method, mask=resolved.brain_mask_backend)
        templates.append((variant, args, effective))
    rows = []
    for index, subject in enumerate(subjects):
        for variant, parameters, effective in templates:
            child_output = output / f'subject_{index + 1:04d}' / variant['name']
            if child_output.is_relative_to(subject.subject_root.resolve() / 'input'):
                raise ValueError('Experiment outputs must not be inside source input data')
            args = [*parameters,
                    '--subject-root', str(subject.subject_root), '--subject-id', subject.subject_id,
                    '--output-root', str(child_output), '--run-name', variant['name'],
                    '--execute', '--print-manifest-only']
            rows.append(dict(job_id=f'{index + 1:04d}_{variant["name"]}', subject_id=subject.subject_id,
                             subject_root=str(subject.subject_root), variant=variant['name'],
                             effective_config=effective,
                             output_root=str(child_output), command=[str(python_executable), str(pipeline_script), *args],
                             run_manifest_path=str(child_output / '00_run_metadata' / f'{variant["name"]}_orchestration_manifest.json')))
    return dict(kind='slicerdsurfer_batch_run', experiment=True, version=1, execute=True, request=request,
                output_root=str(output), run_name=output.name, jobs=jobs,
                batch_root=str(request.get('cohort_root') or request['subject_root']),
                status='planned', subjects=rows, results=[], skipped=skipped,
                manifest_path=str(output / '00_batch_metadata' / 'experiments.json'),
                planned_commands=[row['command'] for row in rows],
                scheduling='Subjects may run concurrently; variants of the same subject run sequentially')
