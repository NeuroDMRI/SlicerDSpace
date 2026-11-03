"""Pure resource-admission policy, with bounded bypass and head reservations."""

import math

RESOURCES = ('cpu', 'memory_gb', 'gpu', 'io')


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


def choose_ready(waiting, active, limits, now, *, max_wait=120., max_bypasses=8, ready_order='fifo'):
    """Backfill only without delaying the head's estimated start; drain on ageing.

    Unknown runtimes can use only resources the head does not need. Runtime
    estimates are advisory; the ageing barrier also protects against overruns.
    """
    if not waiting:
        return None
    if ready_order not in ('fifo','mem_thread'):
        raise ValueError('Unknown ready-task ordering')
    head = waiting[0]
    if fits(head, active, limits):
        return head
    if now - head['queued_at'] >= max_wait or head.get('bypasses', 0) >= max_bypasses:
        return None
    start = reservation_time(head, active, limits, now)
    candidates=waiting[1:]
    if ready_order=='mem_thread':
        # Nipype MultiProc's mem_thread key, restricted to safe backfill candidates.
        candidates=sorted(candidates,key=lambda row:(row['memory_gb'],row['cpu']))
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
