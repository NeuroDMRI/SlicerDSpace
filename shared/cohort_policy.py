"""Pure resource-admission policy: backfill that never delays the head reservation."""

import math

RESOURCES = ('cpu', 'memory_gb', 'gpu', 'io')
_PIPELINE_STAGES = {
    'brain_mask', 'native_dti', 'registered_dti', 'preprocessing',
    'segmentation', 'parcellation', 'surface', 'surface_postprocess',
    'microstructure_dti', 'microstructure_freewater', 'microstructure_noddi',
    'microstructure_dki', 'microstructure_mkcurve', 'microstructure_ifw',
    'dti_tracking', 'csd_act', 'ukf', 'saf', 'wmql', 'wma', 'tractcloud', 'connectome',
}


def fits(row, active, limits):
    return all(sum(item.get(key, 0) for item in active) + row.get(key, 0) <= limits[key]
               for key in RESOURCES)


def reservation_time(head, active, limits, now):
    remaining = list(active)
    if fits(head, remaining, limits):
        return now
    def release(item):
        end = item.get('started_at', now) + item.get('duration_seconds', 0)
        return end if item.get('duration_seconds', 0) > 0 and end > now else math.inf
    releases = sorted({release(item) for item in active if math.isfinite(release(item))})
    for end in releases:
        remaining = [item for item in active
                     if release(item) > end]
        if fits(head, remaining, limits):
            return end
    return math.inf


def _stage_priority(row, gpu_waiting):
    if row.get('cache_hit_hint'):
        return (0, 0, row.get('queued_at', 0))
    if row.get('gpu', 0) > 0 or row.get('io', 0) > 0:
        return (1, row.get('cpu', 0), row.get('queued_at', 0))
    cpu = row.get('cpu', 0)
    if cpu <= 2:
        return (2, cpu, row.get('queued_at', 0))
    if cpu <= 4:
        return (3, cpu, row.get('queued_at', 0))
    stage = row.get('stage', '')
    if gpu_waiting < 2 and stage == 'native_dti':
        return (4, cpu, row.get('queued_at', 0))
    return (5, cpu, row.get('queued_at', 0))


def _mold_candidate(row, active, limits):
    if row.get('stage') == 'native_dti' and row.get('cpu', 0) > 8:
        used_cpu = sum(item.get('cpu', 0) for item in active)
        free_cpu = limits['cpu'] - used_cpu
        heavy_native_active = any(
            item.get('stage') == 'native_dti' and item.get('cpu', 0) >= 8
            for item in active
        )
        downstream_cpu = sum(
            item.get('cpu', 0) for item in active if item.get('stage') != 'native_dti'
        )
        headroom = max(0, 4 - downstream_cpu) if (heavy_native_active and limits['cpu'] >= 24) else 0
        max_cpu = free_cpu - headroom
        if max_cpu >= 8:
            effective_cpu = min(int(row['cpu']), int(max_cpu))
            if effective_cpu != row['cpu']:
                return dict(row, cpu=effective_cpu)
        else:
            return None
    return row


def choose_ready(waiting, active, limits, now, *, ready_order='fifo'):
    """Backfill only without delaying the head's estimated start.

    Unknown runtimes can use only resources the head does not need. Runtime
    estimates are advisory. There is no ageing barrier: the head's reservation is
    recomputed on every call, and a row past its expected end never releases.
    """
    if not waiting:
        return None
    if ready_order not in ('fifo', 'mem_thread'):
        raise ValueError('Unknown ready-task ordering')
    if ready_order == 'mem_thread' and any(row.get('stage') in _PIPELINE_STAGES for row in waiting):
        gpu_waiting = sum(1 for row in waiting if row.get('gpu', 0) > 0)
        ordered = sorted(waiting, key=lambda row: _stage_priority(row, gpu_waiting))
        head = _mold_candidate(ordered[0], active, limits) or ordered[0]
        if fits(head, active, limits):
            ordered[0].update(head)
            return ordered[0]
        start = reservation_time(head, active, limits, now)
        for raw_row in ordered[1:]:
            row = _mold_candidate(raw_row, active, limits)
            if row is None or not fits(row, active, limits):
                continue
            hypothetical = dict(row, started_at=now)
            projected = reservation_time(head, [*active, hypothetical], limits, now)
            safe_without_runtime = not math.isfinite(start) and fits(head, [row], limits)
            if safe_without_runtime or (math.isfinite(start) and projected <= start):
                raw_row.update(row)
                return raw_row
        return None
    head = waiting[0]
    if fits(head, active, limits):
        return head
    start = reservation_time(head, active, limits, now)
    candidates = waiting[1:]
    if ready_order == 'mem_thread':
        # Nipype MultiProc's mem_thread key, restricted to safe backfill candidates.
        candidates = sorted(candidates, key=lambda row: (row['memory_gb'], row['cpu']))
    for row in candidates:
        if not fits(row, active, limits):
            continue
        # Holding this request forever must still permit the protected request.
        hypothetical = dict(row, started_at=now)
        projected = reservation_time(head, [*active, hypothetical], limits, now)
        safe_without_runtime = not math.isfinite(start) and fits(head, [row], limits)
        if safe_without_runtime or (math.isfinite(start) and projected <= start):
            return row
    return None
