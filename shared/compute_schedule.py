"""Resource-bounded independent subprocesses; no MRML/Qt work in workers."""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from graphlib import TopologicalSorter
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time


@dataclass(frozen=True)
class ComputeJob:
    name: str
    command: tuple[str, ...]
    threads: int = 1
    memory_bytes: int = 0
    environment: dict | None = None
    dependencies: tuple[str, ...] = ()


def available_resources():
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = os.cpu_count() or 1
    try:
        import psutil
        memory = int(psutil.virtual_memory().available * .7)
    except ImportError:
        memory = 0
    return cpus, memory


def _save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def _stop(process):
    try:
        import psutil
        children = psutil.Process(process.pid).children(recursive=True)
    except (ImportError, OSError):
        children = []
    except Exception:
        children = []
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        for child in reversed(children):
            try:
                child.terminate()
            except Exception:
                pass
        try:
            process.terminate()
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    for child in children:
        try:
            if child.is_running():
                child.kill()
        except Exception:
            pass


def run_independent_jobs(jobs, *, directory, max_workers=1, cpu_budget=None,
                         memory_budget=None, log=None):
    """Jobs have disjoint outputs and optional named, acyclic dependencies.

    Children inherit the workflow's process group, so Slicer's normal whole-tree
    cancellation still applies. Failures stop remaining children, never publish
    a successful aggregate. Per-job logs retain failures and incomplete work.
    """
    jobs = list(jobs)
    if (max_workers < 1 or len({job.name for job in jobs}) != len(jobs)
            or any(not re.fullmatch(r'[A-Za-z0-9_-]+', job.name) for job in jobs)):
        raise ValueError('Positive worker count and unique job names are required')
    by_name = {job.name:job for job in jobs}
    if any(set(job.dependencies)-by_name.keys() for job in jobs):
        raise ValueError('Compute dependency is not present in this schedule')
    graph = TopologicalSorter({job.name:job.dependencies for job in jobs})
    graph.prepare()
    cpus, memory = available_resources()
    cpus = cpus if cpu_budget is None else cpu_budget
    memory = memory if memory_budget is None else memory_budget
    if cpus < 1 or memory < 0 or any(job.threads < 1 or job.memory_bytes < 0 for job in jobs):
        raise ValueError('Invalid CPU or memory budget')
    if any(job.threads > cpus or (memory and job.memory_bytes > memory) for job in jobs):
        raise ValueError('A compute job exceeds the available resource budget')
    requested_workers = max_workers
    if not memory:
        max_workers = 1
        if requested_workers > 1 and log:
            log('[Compute] Available memory is unknown; using sequential model execution')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    report = directory / 'schedule.json'
    start = time.monotonic()
    rows = {job.name: dict(name=job.name, status='queued', threads=job.threads,
                           memory_bytes=job.memory_bytes, command=list(job.command),
                           dependencies=list(job.dependencies)) for job in jobs}
    record = dict(kind='independent_model_schedule', status='running', max_workers=max_workers,
                  requested_workers=requested_workers,
                  cpu_budget=cpus, memory_budget=memory, jobs=list(rows.values()),
                  scope='Model subprocess wall time, not an algorithmic speedup',
                  readiness='graphlib.TopologicalSorter', completion='concurrent.futures.FIRST_COMPLETED')
    active, waiting, handles = {}, [], []
    executor = ThreadPoolExecutor(max_workers=max_workers,thread_name_prefix='model-exit')
    previous_signal = None
    def cancel(_signum, _frame):
        raise InterruptedError('Workflow cancelled')
    if threading.current_thread() is threading.main_thread():
        previous_signal = signal.signal(signal.SIGTERM, cancel)
    _save(report, record)
    try:
        while graph.is_active():
            waiting.extend(by_name[name] for name in graph.get_ready())
            for job in waiting.copy():
                if len(active) >= max_workers:
                    break
                used_cpu = sum(item[0].threads for item in active.values())
                used_mem = sum(item[0].memory_bytes for item in active.values())
                if used_cpu + job.threads > cpus or (memory and used_mem + job.memory_bytes > memory):
                    continue
                stream = (directory / (job.name + '.log')).open('wb')
                handles.append(stream)
                environment = os.environ.copy()
                environment.update(job.environment or {})
                process = subprocess.Popen(job.command, env=environment, stdout=stream, stderr=subprocess.STDOUT)
                active[executor.submit(process.wait)] = job, process, time.monotonic()
                waiting.remove(job)
                rows[job.name].update(status='running', started_at=datetime.now(timezone.utc).isoformat(),
                                     start_offset_seconds=time.monotonic()-start, pid=process.pid,
                                     log=str(stream.name))
                if log:
                    log(f'[Compute] Running {job.name} ({job.threads} CPU threads reserved)')
                _save(report, record)
            if not active:
                raise RuntimeError('Ready compute jobs cannot fit the declared resources')
            done, _ = wait(active,timeout=.25,return_when=FIRST_COMPLETED)
            failures=[]
            for future in done:
                job, process, began = active.pop(future)
                name, code = job.name, future.result()
                rows[name].update(status='completed' if code == 0 else 'failed', exit_code=code,
                                  ended_at=datetime.now(timezone.utc).isoformat(),
                                  runtime_seconds=time.monotonic()-began)
                _save(report, record)
                if code:
                    failures.append((name,code))
                else:
                    graph.done(name)
                if log:
                    log(f'[Compute] {rows[name]["status"]} {name}: {rows[name]["runtime_seconds"]:.3f} s')
            if failures:
                name,code=failures[0]
                raise RuntimeError(f'{name} failed with exit code {code}; see {rows[name]["log"]}')
        record['status'] = 'completed'
        return record
    except BaseException as exc:
        record.update(status='cancelled' if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else 'failed',
                      error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        for job, process, began in active.values():
            if process.poll() is None:
                _stop(process)
            rows[job.name].update(status='cancelled', runtime_seconds=time.monotonic()-began)
        executor.shutdown(wait=True,cancel_futures=True)
        for row in rows.values():
            if row['status']=='queued':
                row['status']='not_run'
        for stream in handles:
            stream.close()
        if previous_signal is not None:
            signal.signal(signal.SIGTERM, previous_signal)
        record['elapsed_seconds'] = time.monotonic()-start
        record['job_seconds_sum'] = sum(row.get('runtime_seconds', 0) for row in rows.values())
        _save(report, record)
