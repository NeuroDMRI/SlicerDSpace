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
    releases = sorted({max(now, item.get('started_at', now) + item['duration_seconds'])
                       for item in active if item.get('duration_seconds', 0) > 0})
    for end in releases:
        remaining = [item for item in active
                     if not item.get('duration_seconds') or
                     max(now, item.get('started_at', now) + item['duration_seconds']) > end]
        if fits(head, remaining, limits):
            return end
    return math.inf


def choose_ready(waiting, active, limits, now, *, max_wait=120., max_bypasses=8):
    """Backfill only without delaying the head's estimated start; drain on ageing.

    Unknown runtimes can use only resources the head does not need. Runtime
    estimates are advisory; the ageing barrier also protects against overruns.
    """
    if not waiting:
        return None
    head = waiting[0]
    if fits(head, active, limits):
        return head
    if now - head['queued_at'] >= max_wait or head.get('bypasses', 0) >= max_bypasses:
        return None
    start = reservation_time(head, active, limits, now)
    for row in waiting[1:]:
        if not fits(row, active, limits):
            continue
        duration = row.get('duration_seconds', 0)
        # Holding this request forever must still permit the protected request.
        hypothetical = dict(row, started_at=now)
        projected = reservation_time(head, [*active, hypothetical], limits, now)
        safe_without_runtime = not math.isfinite(start) and fits(head, [row], limits)
        if safe_without_runtime or (math.isfinite(start) and projected <= start):
            return row
    return None
