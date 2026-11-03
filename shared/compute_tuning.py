"""Bounded, exact-output-checked runtime configuration selection."""

import hashlib
import json
import math
from pathlib import Path
import statistics


def candidates(stage, max_cpus=8):
    if stage not in ('microstructure_dki', 'microstructure_mkcurve', 'wma'):
        return []
    if type(max_cpus) is not int or max_cpus < 1:
        raise ValueError('Tuning CPU ceiling must be a positive integer')
    threads = [value for value in (1, 2, 4, 8) if value <= max_cpus]
    rows = [dict(threads=value, workers=1, chunk=64 if stage.endswith('mkcurve') else 1024)
            for value in threads]
    if stage.endswith('mkcurve'):
        rows += [dict(threads=1, workers=value, chunk=chunk)
                 for value in (2, 4, 8) if value <= max_cpus for chunk in (32, 64, 128)]
    return rows


def config_id(config):
    return f'w{config["workers"]}-t{config["threads"]}-c{config["chunk"]}'


def tuning_key(stage, shape, arguments):
    from shared.cohort_planning import _option
    from shared.cohort_profile import workload_key
    # Target selection and unrelated models do not change this node's identity.
    flags = {'--dki-fit-method': 'MKCURVE' if stage.endswith('mkcurve') else 'WLS'}
    if stage == 'wma':
        flags = {'--wma-root': '', '--wma-atlas': 'org', '--wma-atlas-dir': ''}
    settings = [value for flag, default in sorted(flags.items())
                for value in (flag, _option(arguments, flag, default))]
    return workload_key(stage, list(shape), settings, 1, False)


def select_configuration(rows, configs, *, cpu_budget, memory_gb, jobs, objective='throughput', minimum_gain=.05,
                         memory_floor_gb=0.):
    """Select within measured candidates; latency and capacity estimates stay distinct."""
    if objective not in ('throughput', 'latency'):
        raise ValueError('Tuning objective must be throughput or latency')
    if (cpu_budget < 1 or jobs < 1 or not math.isfinite(memory_gb) or memory_gb <= 0
            or not math.isfinite(memory_floor_gb) or memory_floor_gb < 0):
        raise ValueError('Positive finite tuning resource budgets are required')
    reference = config_id(configs[0])
    base_rows = [row for row in rows if row['config'] == reference]
    cases = {(row['input'], row['repeat']) for row in base_rows}
    inputs = {item[0] for item in cases}
    if (not cases or len(base_rows) != len(cases) or any(sum(item[0] == name for item in cases) < 3 for name in inputs)
            or any(row.get('status') != 'completed' or not row.get('exact') for row in base_rows)):
        raise ValueError('A complete exact baseline is required for tuning')
    summaries = []
    for config in configs:
        if any(type(config.get(name)) is not int or config[name] < 1 for name in ('workers', 'threads', 'chunk')):
            raise ValueError('Candidate settings must be positive integer counts')
        ident = config_id(config)
        measured = [row for row in rows if row['config'] == ident]
        if ({(row['input'], row['repeat']) for row in measured} != cases
                or len(measured) != len(cases)
                or any(row.get('status') != 'completed' or not row.get('exact') for row in measured)):
            continue
        if any(not math.isfinite(row['seconds']) or row['seconds'] <= 0
               or not math.isfinite(row['rss_bytes']) or row['rss_bytes'] <= 0 for row in measured):
            continue
        cpu = config['workers'] * config['threads']
        ram = max(memory_floor_gb, max(row['rss_bytes'] for row in measured) / 1024**3 * 1.3)
        capacity = min(jobs, cpu_budget // cpu, int(memory_gb // ram))
        if capacity < 1:
            continue
        seconds = statistics.median(row['seconds'] for row in measured)
        score = seconds / capacity if objective == 'throughput' else seconds
        summaries.append(dict(config=ident, settings=config, seconds=seconds, memory_gb=ram,
                              estimated_capacity=capacity, score=score, samples=len(measured)))
    base = next((row for row in summaries if row['config'] == reference), None)
    if base is None:
        raise ValueError('No baseline configuration fits the tuning resource budget')
    best = min(summaries, key=lambda row: (row['score'], row['settings']['workers'] * row['settings']['threads']))
    if best['score'] > base['score'] * (1 - minimum_gain):
        best = base
    return dict(selected=best, candidates=summaries, objective=objective,
                minimum_gain=minimum_gain, selection_scope='Best accepted measured candidate; capacity is an estimate, not cohort throughput')


def read_profile(path):
    if not path or not Path(path).is_file():
        return {}
    try:
        if Path(path).stat().st_size > 8 * 1024**2:
            return {}
        record = json.loads(Path(path).read_text())
        return record if (record.get('version') == 1 and record.get('status') == 'passed'
                          and isinstance(record.get('profiles'),dict)) else {}
    except (ValueError, OSError, AttributeError):
        return {}


def lookup_profile(profile, stage, shape, arguments, *, cpu_budget, memory_gb, jobs, memory_floor_gb=0.):
    if stage not in ('microstructure_dki', 'microstructure_mkcurve', 'wma'):
        return None
    key = tuning_key(stage, shape, arguments)
    item = profile.get('profiles', {}).get(key)
    if not item:
        return None
    try:
        result = select_configuration(item['measurements'], item['configs'], cpu_budget=cpu_budget,
                                      memory_gb=memory_gb, jobs=jobs, objective=item.get('objective', 'throughput'),
                                      memory_floor_gb=memory_floor_gb)
        selected = result['selected']
        settings = selected['settings']
        for name in ('workers', 'threads', 'chunk'):
            if type(settings[name]) is not int or settings[name] < 1:
                return None
        return dict(**selected, key=key, source='exact-checked runtime calibration')
    except (ValueError, KeyError, TypeError, OverflowError, IndexError, AttributeError):
        return None


def runtime_arguments(stage, settings):
    if stage == 'wma':
        return ['--wma-threads', str(settings['threads'])]
    arguments = ['--dki-threads', str(settings['threads'])]
    if stage == 'microstructure_mkcurve':
        arguments += ['--mkcurve-workers', str(settings['workers']),
                      '--mkcurve-voxels-per-chunk', str(settings['chunk'])]
    else:
        arguments += ['--dki-voxels-per-chunk', str(settings['chunk'])]
    return arguments


def array_fingerprint(arrays):
    """Preserve dtype, shape, NaN payloads and numerical values without rounding."""
    import numpy as np
    digest = hashlib.sha256()
    for key, values in sorted(arrays.items()):
        array = np.asarray(values)
        digest.update(json.dumps([key, array.shape, array.dtype.str]).encode())
        digest.update(array.tobytes(order='C'))
    return digest.hexdigest()
