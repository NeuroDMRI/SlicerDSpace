"""Scope Linux CPU placement for native tools that ignore thread variables."""

from contextlib import contextmanager
import os
import shutil


def cpu_allocation(budget, requested=None):
    """Select CPUs from the caller's allocation, preserving an explicit CPU set."""
    available = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else []
    if not available:
        if requested:
            raise ValueError('Explicit CPU placement is unavailable on this platform')
        return []
    selected = list(requested) if requested is not None else available
    if (any(type(cpu) is not int or cpu < 0 for cpu in selected)
            or len(selected) != len(set(selected)) or len(selected) < budget
            or not set(selected) <= set(available)):
        raise ValueError('CPU affinity must contain unique allowed CPUs and cover the CPU budget')
    return selected[:budget]


def isolated_command(command, cpus):
    """Apply placement before exec, so launchers and new threads inherit it."""
    if not cpus:
        return list(command)
    selected = cpu_allocation(len(cpus), cpus)
    taskset = shutil.which('taskset')
    if not taskset:
        raise RuntimeError('CPU-isolated benchmarking requires util-linux taskset')
    return [taskset, '--cpu-list', ','.join(map(str, selected)), *command]


def affinity_violations(pid, cpus):
    """Inspect live descendant threads without changing their placement."""
    if not cpus or not hasattr(os, 'sched_getaffinity'):
        return []
    import psutil
    try:
        owner = psutil.Process(pid)
        processes = [owner, *owner.children(recursive=True)]
    except psutil.NoSuchProcess:
        return []
    allowed = set(cpus)
    issues = []
    for process in processes:
        try:
            for thread in process.threads():
                try:
                    observed = os.sched_getaffinity(thread.id)
                except ProcessLookupError:
                    continue
                if not observed <= allowed:
                    issues.append(dict(pid=process.pid, tid=thread.id, observed=sorted(observed)))
                    if len(issues) >= 32:
                        return issues
        except psutil.NoSuchProcess:
            continue
    return issues


@contextmanager
def cpu_placement(cpus):
    if not cpus or not hasattr(os, 'sched_setaffinity'):
        yield
        return
    import psutil
    process = psutil.Process()
    original = os.sched_getaffinity(0)
    if not set(cpus) <= original:
        raise ValueError('Stage CPU placement would escape the inherited CPU allocation')
    previous = {}
    try:
        for thread in process.threads():
            try:
                previous[thread.id] = os.sched_getaffinity(thread.id)
                os.sched_setaffinity(thread.id, cpus)
            except ProcessLookupError:
                pass
        yield
    finally:
        for thread in process.threads():
            try:
                os.sched_setaffinity(thread.id, previous.get(thread.id, original))
            except ProcessLookupError:
                pass
