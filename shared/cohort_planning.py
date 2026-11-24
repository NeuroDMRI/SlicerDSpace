"""Conservative header-only estimates for optional cohort stage admission."""

import json
import math
from pathlib import Path


GPU_CAPABLE_STAGES = frozenset({
    'brain_mask',
    'preprocessing',
    'segmentation',
    'parcellation',
    'surface',
    'tractcloud',
    'microstructure_freewater',
    'microstructure_noddi',
    'microstructure_mkcurve',
})


def available_cohort_resources(
    fraction=.7,
    *,
    proc_cgroup_path=Path('/proc/self/cgroup'),
    cgroup_root=Path('/sys/fs/cgroup'),
):
    from shared.compute_schedule import available_resources
    from shared.hardware_adapt import linux_cgroup_limits

    cpus, memory = available_resources()
    memory = int(memory / .7 * fraction)
    # Respect nested Linux cgroup v1/v2 container limits in addition to CPU affinity and host RAM.
    limits = linux_cgroup_limits(proc_cgroup_path=proc_cgroup_path, cgroup_root=cgroup_root)
    if limits['cpu_limit'] is not None:
        cpus = min(cpus, limits['cpu_limit'])
    if limits['memory_available_bytes'] is not None:
        free = int(limits['memory_available_bytes'] * fraction)
        memory = min(memory, free) if memory else free
    return cpus, memory


def _stage_uses_gpu_in_plan(stage, arguments):
    """Return True when ``stage`` is GPU-capable under the supplied CLI ``arguments``."""
    if stage not in GPU_CAPABLE_STAGES:
        return False
    if stage == 'microstructure_mkcurve':
        fit_method = _option(arguments, '--dki-fit-method', '').strip().upper()
        if fit_method == 'MKCURVE_MATLAB':
            return False
    if stage == 'microstructure_noddi' and '--noddi-directional-average' in arguments:
        return False

    def _effective_device(flags):
        for flag in flags:
            val = _option(arguments, flag, '').strip().lower()
            if val and val != 'auto':
                return val
        return 'auto'

    if stage == 'brain_mask':
        dev = _effective_device(('--brain-mask-device',))
    elif stage == 'preprocessing':
        preproc_stages = _option(arguments, '--preprocessing-stages', '').strip().lower()
        if preproc_stages:
            stage_tokens = {tok.strip() for tok in preproc_stages.split(',') if tok.strip()}
            if 'denoise' not in stage_tokens and 'all' not in stage_tokens:
                return False
        dev = _effective_device(('--brain-mask-device', '--ddseg-device', '--surface-device'))
    elif stage in {'segmentation', 'parcellation'}:
        dev = _effective_device(('--ddseg-device',))
    elif stage == 'surface':
        dev = _effective_device(('--surface-device',))
    elif stage == 'tractcloud':
        dev = _effective_device(('--tractcloud-device',))
    elif stage in {'microstructure_freewater', 'microstructure_noddi', 'microstructure_mkcurve'}:
        dev = _effective_device(('--ddseg-device', '--surface-device', '--brain-mask-device'))
        if dev in {'cpu', 'mps'}:
            return False
        return True
    else:
        dev = 'auto'
    return dev != 'cpu'


def _option(arguments, flag, default=''):
    value = default
    for i, item in enumerate(arguments):
        if item.startswith(flag + '='):
            value = item.split('=', 1)[1]
        elif item == flag and i + 1 < len(arguments):
            value = arguments[i + 1]
    return value


def _options(arguments, flag):
    return [item.split('=', 1)[1] if item.startswith(flag+'=') else arguments[i+1]
            for i, item in enumerate(arguments)
            if item.startswith(flag+'=') or (item == flag and i+1 < len(arguments))]


def _fitted_ram_gb(stage, signal, ram_fit):
    """S2: 1 + 1.25 * P * (signal / signal_ref) GiB for a stage measured in M1, else None."""
    entry = ram_fit.get(stage)
    if entry is None:
        return None
    peak_gib, signal_ref = entry
    if not (math.isfinite(peak_gib) and peak_gib > 0 and math.isfinite(signal_ref) and signal_ref > 0):
        return None
    return 1 + 1.25 * peak_gib * (signal / signal_ref)


def default_duration_seconds(stage, cpu, signal, defaults):
    """S3: t_p * c_p / min(n, c_p) * (signal / signal_ref) for a stage measured in M1, else 0.

    p is the smallest probed CPU count that is at least n, or the largest probed count.
    """
    # SPEC-GAP: the table shape is {stage: {'signal_ref': float, 'probes': {p: (t_p, c_p)}}};
    # the spec names the content but not the layout. A malformed entry gives 0 (unknown).
    entry = defaults.get(stage)
    if not entry:
        return 0
    probes, signal_ref = entry.get('probes') or {}, entry.get('signal_ref')
    if not probes or not isinstance(signal_ref, (int, float)) or not math.isfinite(signal_ref) or signal_ref <= 0:
        return 0
    counts = sorted(probes)
    p = next((count for count in counts if count >= cpu), counts[-1])
    t_p, c_p = probes[p]
    if not (math.isfinite(t_p) and t_p > 0 and math.isfinite(c_p) and c_p > 0):
        return 0
    return t_p * c_p / min(cpu, c_p) * (signal / signal_ref)


def image_size(root):
    """Read dimensions only; never decompress the full DWI or change its files."""
    for directory in (Path(root) / 'input', Path(root)):
        for filename in ('data.nii.gz', 'data.nii', 'data.nhdr', 'data.nrrd'):
            path = directory / filename
            if not path.is_file():
                continue
            try:
                if path.suffix in ('.nhdr', '.nrrd'):
                    import nrrd
                    header = nrrd.read_header(str(path))
                    shape = tuple(int(value) for value in header['sizes'])
                    list_axis = next((i for i, kind in enumerate(header.get('kinds', ()))
                                      if kind in ('list', 'vector')), None)
                    if len(shape) == 4 and list_axis is not None:
                        shape = tuple(size for i, size in enumerate(shape) if i != list_axis) + (shape[list_axis],)
                else:
                    import nibabel as nib
                    shape = tuple(nib.load(str(path)).shape)
                if len(shape) != 4 or any(value <= 0 for value in shape):
                    continue
                return dict(shape=list(shape), source=str(path), estimated=False)
            except Exception:
                # Normal input validation reports malformed files. Planning uses
                # a documented conservative size when the header is unavailable.
                continue
    return dict(shape=[160, 192, 160, 128], source=None, estimated=True)


def cohort_phase_encoding_blocks(subjects, arguments, *, selected=None, output_root=None):
    """W2b rule 4 (T1 D1): {subject_id: message} for subjects whose preprocessing would run with no PE direction.

    Empty when preprocessing is not selected. Earlier preprocessing outputs do not lift the block (F5): selected
    preprocessing is reused only by the full version-6 key of the current settings, which needs a PE direction, so a
    subject with no direction (user value or sidecar) cannot reuse earlier outputs, version 5 or 6. `output_root` is
    the batch output base; it is kept for callers and no longer changes the decision.
    """
    from shared.cohort_targets import selected_stages
    from shared.pe_resolution import phase_encoding_block_reason

    if 'preprocessing' not in (selected_stages(arguments) if selected is None else selected):
        return {}
    has_reverse = bool(_option(arguments, '--preprocessing-reverse-phase-b0'))
    preproc_stages_text = (_option(arguments, '--preprocessing-stages') or '').strip()
    if preproc_stages_text:
        from shared.algorithm_selection import PREPROCESSING, algorithms
        chosen_preproc = algorithms(preproc_stages_text, PREPROCESSING, 'denoise,eddy,topup')
        has_reverse = has_reverse and ('topup' in chosen_preproc)
        if 'eddy' not in chosen_preproc and not has_reverse:
            return {}
    direction = _option(arguments, '--preprocessing-phase-encoding-direction') or None
    readout = _option(arguments, '--preprocessing-readout-time')
    try:
        readout = float(readout) if readout else None
    except ValueError:
        readout = None  # the subject command reports a bad value
    blocks = {}
    for subject in subjects:
        reason = phase_encoding_block_reason(subject.subject_root, user_phase_encoding=direction,
                                             user_readout_time=readout, has_reverse_b0=has_reverse)
        if reason:
            blocks[subject.subject_id] = reason
    return blocks


def cohort_phase_encoding_message(blocks):
    """One error text for the cohort plan: the rule-4 message once, then the subjects that need it."""
    if not blocks:
        return ''
    first = next(iter(blocks.values()))
    return f"{first}. Blocked subjects ({len(blocks)}): " + ', '.join(sorted(blocks))


def plan_cohort(subjects, arguments, *, jobs=None, cpu_budget=None, memory_gb=None,
                stage_cpus=None, stage_memory_gb=None, gpu_slots=1, available=None,
                adaptive=False, resource_fraction=.8, history_path=None, tuning_profile=None,
                output_root=None, allow_low_memory=False):
    if not math.isfinite(resource_fraction) or not 0 < resource_fraction <= 1:
        raise ValueError('Resource fraction must be greater than zero and at most one')
    cpus, memory = available if available is not None else available_cohort_resources(resource_fraction if adaptive else .7)
    cpu = cpu_budget if cpu_budget is not None else max(1, math.floor(cpus * (resource_fraction if adaptive else 1)))
    ram = memory_gb if memory_gb is not None else memory / 1024**3
    if not math.isfinite(ram) or ram <= 0 or cpu < 1:
        raise ValueError('Available CPU/RAM could not be established; provide --batch-cpus and --batch-memory-gb')
    if (jobs is not None and jobs < 1) or gpu_slots < 1:
        raise ValueError('Subject and GPU-slot limits must be positive')
    from shared.cohort_targets import selected_stages
    from shared import cohort_defaults
    # Read the M1 tables at call time so tests (and a later M1 update) can replace them.
    cpu_requests, ram_fit = cohort_defaults.STAGE_CPU_REQUESTS, cohort_defaults.RAM_FIT
    duration_defaults = cohort_defaults.DURATION_DEFAULTS
    selected = selected_stages(arguments)
    if blocks := cohort_phase_encoding_blocks(subjects, arguments, selected=selected, output_root=output_root):
        raise ValueError(cohort_phase_encoding_message(blocks))
    hints, stage_hints = [], {}
    for flag, stages in (('--surface-cpu-threads', {'surface'}), ('--mrtrix-nthreads', {'csd_act', 'saf', 'connectome'}),
                         ('--preprocessing-eddy-threads', {'preprocessing'}),
                         ('--microstructure-free-water-n-jobs', {'microstructure_freewater'}),
                         ('--wma-threads', {'wma'}), ('--dki-threads', {'microstructure_dki', 'microstructure_mkcurve'})):
        if not stages & selected:
            continue
        value = _option(arguments, flag)
        if value:
            count = int(value)
            if count < 0:
                raise ValueError(flag + ' must not request unbounded workers in resource mode')
            hints.append(count)
            for stage in stages & selected:
                stage_hints[stage] = max(1, count)
    if 'microstructure_mkcurve' in selected:
        workers = int(_option(arguments, '--mkcurve-workers', '1'))
        if workers < 1:
            raise ValueError('--mkcurve-workers must be positive')
        needed = workers * stage_hints.get('microstructure_mkcurve', 1)
        hints.append(needed)
        stage_hints['microstructure_mkcurve'] = needed
    threads = stage_cpus if stage_cpus is not None else min(cpu, max([min(4 if adaptive else 8, cpu), *hints]))
    if threads < 1 or threads > cpu or any(hint > threads for hint in hints):
        raise ValueError('Algorithm thread request or per-stage CPU reservation exceeds the cohort budget')
    if stage_memory_gb is not None and (not math.isfinite(stage_memory_gb) or stage_memory_gb <= 0):
        raise ValueError('--batch-stage-memory-gb must be finite and positive')

    from shared.cohort_profile import read_history, history_estimate, workload_key
    from shared.compute_tuning import read_profile, lookup_profile, runtime_arguments
    tuned_profiles = read_profile(tuning_profile) if adaptive else {}
    history = read_history(history_path) if adaptive else {}
    profiles, inputs, selected_peaks, subject_arguments = {}, {}, [], {}
    plan_warnings, low_memory_mitigations = [], {}
    simple_dti = (_option(arguments, '--microstructure-stages') == 'dti' and
                  not any(flag in arguments for flag in ('--run-free-water', '--run-noddi', '--run-dki')))
    for subject in subjects:
        image = image_size(subject.subject_root)
        inputs[subject.subject_id] = image
        voxels = math.prod(image['shape'][:3])
        signal = voxels * image['shape'][3] * 8 / 1024**3
        maps = voxels * 1200 / 1024**3
        estimates = dict(brain_mask=2 + signal, native_dti=2 + 2 * signal,
            registered_dti=3 + 2 * signal, preprocessing=3 + 3 * signal,
            segmentation=3 + .5 * signal, parcellation=12 + .25 * signal,
            surface=6 + signal, surface_postprocess=4 + .5 * signal,
            microstructure=2 + (1.5 if simple_dti else 2.5) * signal + maps,
            csd_act=3 + 1.5 * signal + maps, ukf=4 + signal,
            dti_tracking=2 + signal, saf=4 + .5 * signal,
            wmql=8, wma=16, tractcloud=12, connectome=4)
        estimates.update(microstructure_dti=2+.5*signal,
                         microstructure_freewater=2+1.5*signal+maps,
                         microstructure_noddi=2+2*signal+maps,
                         microstructure_dki=2+signal+maps,
                         microstructure_mkcurve=2+signal+maps,
                         microstructure_ifw=2+.25*signal+maps)
        profile = {}
        tuned_arguments = []
        for stage, estimate in estimates.items():
            # SPEC-GAP: S2 does not name the scheduler mode; the fit applies in adaptive mode only,
            # like S1.1 and S3, so `resources` mode keeps the header heuristic.
            fitted = _fitted_ram_gb(stage, signal, ram_fit) if adaptive else None
            if fitted is not None:
                estimate = fitted
            profile[stage] = dict(cpu=threads if stage not in {'wmql', 'connectome'} else min(2, threads),
                memory_gb=stage_memory_gb if stage_memory_gb is not None else math.ceil(estimate * 4) / 4)
            # SPEC-GAP: a table value is capped at the CPU budget (a request above it would fail validation).
            if adaptive and stage_cpus is None and stage in cpu_requests:
                profile[stage]['cpu'] = max(1, min(int(cpu_requests[stage]), cpu))
            if adaptive and stage_cpus is None and stage in {'dti_tracking', 'tractcloud'}:
                profile[stage]['cpu'] = min(2, cpu)
            if adaptive and stage_cpus is None and stage in {'wmql', 'connectome', 'wma',
                    'microstructure_dti', 'microstructure_freewater', 'microstructure_dki', 'microstructure_mkcurve',
                    'microstructure_ifw'}:
                profile[stage]['cpu'] = 1
                if stage == 'microstructure_freewater':
                    profile[stage]['cpu'] = max(1, int(_option(arguments, '--microstructure-free-water-n-jobs', '1')))
            if stage in stage_hints:
                profile[stage]['cpu'] = stage_hints[stage]
            flags = ('--wma-threads',) if stage == 'wma' else (
                ('--dki-threads', '--mkcurve-workers', '--mkcurve-voxels-per-chunk') if stage.endswith('mkcurve')
                else ('--dki-threads', '--dki-voxels-per-chunk'))
            if (adaptive and stage in selected and stage_cpus is None and tuned_profiles
                    and not any(_option(arguments, flag) for flag in flags)):
                tuned = lookup_profile(tuned_profiles, stage, image['shape'], arguments,
                    cpu_budget=cpu, memory_gb=ram, jobs=jobs or min(len(subjects), 8) or 1,
                    memory_floor_gb=profile[stage]['memory_gb'])
                if tuned:
                    settings = tuned['settings']
                    profile[stage]['cpu'] = settings['threads'] * settings['workers']
                    profile[stage]['calibration'] = tuned
                    if stage_memory_gb is None:
                        profile[stage]['memory_gb'] = max(profile[stage]['memory_gb'], math.ceil(tuned['memory_gb'] * 4) / 4)
                    tuned_arguments += runtime_arguments(stage, settings)
            if adaptive:
                request = profile[stage]
                stage_gpu = _stage_uses_gpu_in_plan(stage, [*arguments, *tuned_arguments])
                key = workload_key(stage, image['shape'], [*arguments, *tuned_arguments], request['cpu'], stage_gpu)
                feedback = history_estimate(history, key)
                request.update(gpu=int(stage_gpu),
                               io=int(stage in {'registered_dti', 'preprocessing', 'wmql', 'connectome'}),
                               profile_key=key, duration_seconds=feedback['duration_seconds'] if feedback else 0,
                               estimate_source='header heuristic' if feedback is None else 'header + measured high-water mark')
                if feedback and stage_memory_gb is None:
                    request['memory_gb'] = max(request['memory_gb'], math.ceil(feedback['memory_gb'] * 4) / 4)
        if stage_cpus is not None:
            for request in profile.values():
                request['cpu'] = threads
        if adaptive:
            # S3: a stage without history gets the M1 default duration (0 = unknown).
            for stage, request in profile.items():
                if not request['duration_seconds']:
                    request['duration_seconds'] = default_duration_seconds(
                        stage, request['cpu'], signal, duration_defaults)
        peaks = [profile[stage]['memory_gb'] for stage in selected if stage in profile]
        peak = max(peaks, default=stage_memory_gb or 1)
        if peak > ram:
            if not allow_low_memory:
                raise ValueError(
                    f'{subject.subject_id}: estimated stage RAM {peak:g} GiB exceeds budget {ram:.1f} GiB. '
                    'Review available memory, pass --low-memory on modest-RAM hosts, or override reservations after a pilot.'
                )
            orig_stage_mem = {stage: profile[stage]['memory_gb'] for stage in selected if stage in profile}
            sub_mitigations = []
            for stage in selected:
                if stage in profile and profile[stage]['memory_gb'] > ram:
                    profile[stage]['memory_gb'] = ram
                    profile[stage]['cpu'] = max(1, min(profile[stage]['cpu'], 4, cpu))
            if 'microstructure_dki' in selected and not _option([*arguments, *tuned_arguments], '--dki-voxels-per-chunk'):
                tuned_arguments += ['--dki-voxels-per-chunk', '8192']
                sub_mitigations.append(dict(stage='microstructure_dki', applied_args=['--dki-voxels-per-chunk', '8192']))
            if 'microstructure_mkcurve' in selected and not _option([*arguments, *tuned_arguments], '--mkcurve-voxels-per-chunk'):
                tuned_arguments += ['--mkcurve-voxels-per-chunk', '1024']
                sub_mitigations.append(dict(stage='microstructure_mkcurve', applied_args=['--mkcurve-voxels-per-chunk', '1024']))
            if 'ukf' in selected and 'ukf' in profile and orig_stage_mem.get('ukf', 0) > ram:
                ukf_orig = orig_stage_mem['ukf']
                ukf_cpu = max(1, min(profile['ukf']['cpu'], 2, cpu))
                profile['ukf']['cpu'] = ukf_cpu
                raw_ukf_opts = _option([*arguments, *tuned_arguments], '--ukf-parameters', '')
                try:
                    ukf_opts = json.loads(raw_ukf_opts) if raw_ukf_opts else {}
                    if not isinstance(ukf_opts, dict):
                        ukf_opts = {}
                except ValueError:
                    ukf_opts = {}
                merged_ukf = dict(ukf_opts)
                prev_threads = merged_ukf.get('numThreads')
                if prev_threads is None or prev_threads == -1 or (isinstance(prev_threads, int) and prev_threads > ukf_cpu):
                    merged_ukf['numThreads'] = ukf_cpu
                explicit_high_seeds = isinstance(merged_ukf.get('seedsPerVoxel'), int) and merged_ukf['seedsPerVoxel'] > 1
                if 'seedsPerVoxel' not in merged_ukf:
                    merged_ukf['seedsPerVoxel'] = 1
                ukf_json = json.dumps(merged_ukf, separators=(',', ':'))
                tuned_arguments += ['--ukf-parameters', ukf_json]
                seed_note = (
                    f" Note: explicit seedsPerVoxel={merged_ukf['seedsPerVoxel']} increases VTK polydata RAM."
                    if explicit_high_seeds
                    else ""
                )
                ukf_warning = (
                    f"{subject.subject_id}: UKF estimated RAM ({ukf_orig:g} GiB) exceeds low-memory budget "
                    f"({ram:.1f} GiB); clamped reservation to {ram:.1f} GiB and {ukf_cpu} thread(s) with "
                    f"--ukf-parameters={ukf_json}.{seed_note} Native UKF C++ memory is not hard-capped by cohort "
                    "reservations and may still exceed the budget on multi-shell high-resolution DWI; "
                    "consider GPU UKF (--useGPU) or CSD/ACT if memory pressure occurs."
                )
                plan_warnings.append(ukf_warning)
                sub_mitigations.append(dict(
                    stage='ukf',
                    estimated_memory_gb=ukf_orig,
                    budget_memory_gb=ram,
                    clamped_cpu=ukf_cpu,
                    applied_ukf_parameters=merged_ukf,
                    applied_args=['--ukf-parameters', ukf_json],
                    warning=ukf_warning,
                ))
            for stage, orig_mem in orig_stage_mem.items():
                if orig_mem > ram and stage not in {'ukf', 'microstructure_dki', 'microstructure_mkcurve'}:
                    sub_mitigations.append(dict(
                        stage=stage,
                        estimated_memory_gb=orig_mem,
                        budget_memory_gb=ram,
                        clamped_cpu=profile[stage]['cpu'],
                    ))
            if sub_mitigations:
                low_memory_mitigations[subject.subject_id] = sub_mitigations
            peak = min(peak, ram)
        selected_peaks.append(peak)
        profiles[subject.subject_id] = profile
        subject_arguments[subject.subject_id] = tuned_arguments
    peak = max(selected_peaks, default=stage_memory_gb or min(1, ram))
    capacity = max(1, min(cpu // threads, int(ram // peak)))
    count = jobs if jobs is not None else max(1, min(len(subjects), capacity))
    if adaptive and jobs is None:
        # Waiters keep interpreter state: retain RAM headroom and bound the frontier.
        count = max(1, min(len(subjects), int(max(.5, ram - peak) / .5), cpu))
    if allow_low_memory and jobs is None and any(
            any(req.get('memory_gb', 0) >= ram for st, req in prof.items() if st in selected)
            for prof in profiles.values()):
        count = 1
    return dict(cpu_budget=cpu, memory_gb=ram, stage_cpus=threads, stage_memory_gb=peak,
                memory_reservations_fixed=stage_memory_gb is not None,
                low_memory_mode=bool(allow_low_memory),
                low_memory_mitigations=low_memory_mitigations,
                warnings=plan_warnings,
                gpu_slots=gpu_slots, jobs=count, jobs_automatic=jobs is None,
                subject_profiles=profiles, input_headers=inputs, selected_stages=sorted(selected),
                subject_arguments=subject_arguments, tuning_profile=str(tuning_profile) if tuning_profile else None,
                estimation='Header-based conservative heuristic, not measured peak usage',
                idle_process_headroom='Frontier bounded by cohort size and CPU/RAM budget; adaptive allows 0.5 GiB per waiting interpreter',
                adaptive=adaptive, resource_fraction=resource_fraction if adaptive else None,
                history_path=str(history_path) if history_path else None,
                policy='fair_backfill' if adaptive else 'first_fit', io_slots=2)
