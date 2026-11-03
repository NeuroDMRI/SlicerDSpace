"""Bounded completion-driven chunks with explicit original output offsets."""

from concurrent.futures import FIRST_COMPLETED, wait


def chunk_results(count, chunk_size, workers, payload, fit, *, pool=None, schedule='dynamic'):
    if schedule not in ('dynamic', 'ordered'):
        raise ValueError('Chunk schedule must be dynamic or ordered')
    if pool is None:
        for start in range(0, count, chunk_size):
            yield start, min(start + chunk_size, count), fit(payload(start))
        return
    starts = iter(range(0, count, chunk_size))
    pending = {}

    def submit_next():
        start = next(starts, None)
        if start is not None:
            pending[pool.submit(fit, payload(start))] = start

    try:
        for _ in range(workers):
            submit_next()
        while pending:
            if schedule == 'ordered':
                done = [min(pending, key=pending.get)]
            else:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                done = sorted(done, key=pending.get)
            for future in done:
                start = pending.pop(future)
                result = future.result()
                submit_next()
                yield start, min(start + chunk_size, count), result
    finally:
        for future in pending:
            future.cancel()
