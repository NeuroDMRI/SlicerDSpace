"""Conservative header-only estimates for optional cohort stage admission."""

import math
from pathlib import Path


def available_cohort_resources(fraction=.7):
    from shared.compute_schedule import available_resources
    cpus, memory = available_resources()
    memory = int(memory / .7 * fraction)
    # Respect common container limits in addition to CPU affinity and host RAM.
    for quota, period in ((Path('/sys/fs/cgroup/cpu.max'), None),
                          (Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us'),
                           Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us'))):
        try:
            q, p = quota.read_text().split() if period is None else (quota.read_text(), period.read_text())
            if q.strip() != 'max' and int(q) > 0:
                cpus = min(cpus, max(1, int(q) // int(p)))
        except (OSError, ValueError, ZeroDivisionError):
            pass
    for limit, usage in (('memory.max', 'memory.current'),
                         ('memory/memory.limit_in_bytes', 'memory/memory.usage_in_bytes')):
        try:
            ceiling = int((Path('/sys/fs/cgroup') / limit).read_text())
            used = int((Path('/sys/fs/cgroup') / usage).read_text())
            free = int(max(0, ceiling - used) * fraction)
            memory = min(memory, free) if memory else free
        except (OSError, ValueError):
            pass
    return cpus, memory


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


def plan_cohort(subjects, arguments, *, jobs=None, cpu_budget=None, memory_gb=None,
                stage_cpus=None, stage_memory_gb=None, gpu_slots=1, available=None,
                adaptive=False, resource_fraction=.8, history_path=None, tuning_profile=None):
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
    selected = selected_stages(arguments)
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
                         microstructure_mkcurve=2+signal+maps)
        profile = {}
        tuned_arguments = []
        for stage, estimate in estimates.items():
            profile[stage] = dict(cpu=threads if stage not in {'wmql', 'connectome'} else min(2, threads),
                memory_gb=stage_memory_gb if stage_memory_gb is not None else math.ceil(estimate * 4) / 4)
            if adaptive and stage_cpus is None and stage in {'wmql', 'connectome', 'wma',
                    'microstructure_dti', 'microstructure_freewater', 'microstructure_dki', 'microstructure_mkcurve'}:
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
                key = workload_key(stage, image['shape'], [*arguments, *tuned_arguments], request['cpu'],
                                   stage in {'brain_mask', 'segmentation', 'parcellation', 'surface', 'tractcloud'})
                feedback = history_estimate(history, key)
                request.update(io=int(stage in {'registered_dti', 'preprocessing', 'wmql', 'connectome'}),
                               profile_key=key, duration_seconds=feedback['duration_seconds'] if feedback else 0,
                               estimate_source='header heuristic' if feedback is None else 'header + measured high-water mark')
                if feedback and stage_memory_gb is None:
                    request['memory_gb'] = max(request['memory_gb'], math.ceil(feedback['memory_gb'] * 4) / 4)
        if stage_cpus is not None:
            for request in profile.values():
                request['cpu'] = threads
        peaks = [profile[stage]['memory_gb'] for stage in selected if stage in profile]
        peak = max(peaks, default=stage_memory_gb or 1)
        if peak > ram:
            raise ValueError(f'{subject.subject_id}: estimated stage RAM {peak:g} GiB exceeds budget {ram:.1f} GiB. '
                             'Review available memory or override reservations after a pilot.')
        selected_peaks.append(peak)
        profiles[subject.subject_id] = profile
        subject_arguments[subject.subject_id] = tuned_arguments
    peak = max(selected_peaks, default=stage_memory_gb or min(1, ram))
    capacity = max(1, min(cpu // threads, int(ram // peak)))
    count = jobs if jobs is not None else max(1, min(len(subjects), capacity, 8))
    if adaptive and jobs is None:
        # Waiters keep interpreter state: retain RAM headroom and bound the frontier.
        count = max(1, min(len(subjects), 8, int(max(.5, ram - peak) / .5), max(2, cpu)))
    return dict(cpu_budget=cpu, memory_gb=ram, stage_cpus=threads, stage_memory_gb=peak,
                memory_reservations_fixed=stage_memory_gb is not None,
                gpu_slots=gpu_slots, jobs=count, jobs_automatic=jobs is None,
                subject_profiles=profiles, input_headers=inputs, selected_stages=sorted(selected),
                subject_arguments=subject_arguments, tuning_profile=str(tuning_profile) if tuning_profile else None,
                estimation='Header-based conservative heuristic, not measured peak usage',
                idle_process_headroom='Automatic frontier capped at 8; adaptive mode allows 0.5 GiB per waiting interpreter',
                adaptive=adaptive, resource_fraction=resource_fraction if adaptive else None,
                history_path=str(history_path) if history_path else None,
                policy='fair_backfill' if adaptive else 'first_fit', io_slots=2)
