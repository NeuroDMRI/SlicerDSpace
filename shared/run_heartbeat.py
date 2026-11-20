"""Liveness heartbeat for long subject-run stages (H2).

UKF, WMA, WMQL and eddy can run for hours without a line of output, so a hung tool
looks like a slow one. While a run executes, a daemon thread prints one line every
``SLICERDSPACE_HEARTBEAT_SECONDS`` seconds (default 60). It never kills anything:
a silent but live step only produces a warning.

Line formats (the GUI can parse them with :func:`parse_heartbeat_line`)::

    [Heartbeat] stage=<current task> elapsed=<s> cpu_s=<s> children=<n> log_bytes=<n> [stalled_min=<N>]
    [Stall warning] <stage>: no CPU progress and no output for <N> min. The step may be hung; press Cancel to stop it, or wait if the tool is known to be quiet.

* ``stage``: the task the run log named last (``[Orchestrator] Running <X>``, ``<X> started``, or a tool
  wrapper line such as ``[UKF] Running UKFTractography``, which names stage ``UKF``); ``-`` after a
  ``<X> completed in`` line. It may contain spaces; it ends where ``elapsed=`` starts. Stall detection
  does not depend on the stage name: it runs for the whole run, so a tool that never names itself is still caught.
* ``elapsed``: whole seconds since the stage became current.
* ``cpu_s``: user+system CPU seconds of this process and all its descendants, summed per process
  over the run (a descendant that has exited keeps its last sampled CPU). Left out without psutil.
* ``children``: number of live descendants. Left out without psutil.
* ``log_bytes``: size of the active stage log, when one is known: the most recently written ``*.log``
  below the output root that changed since the stage began, except the file this process's stdout goes to and
  the files named in ``SLICERDSPACE_HEARTBEAT_EXCLUDE``. Left out otherwise.
* ``stalled_min``: whole minutes without progress. Only present once that reaches
  ``SLICERDSPACE_STALL_WARN_MINUTES`` (default 10). Progress means more than 1% of one core of new CPU
  time since the last sample, a run-log line, or growth of the active stage log.
* ``[Stall warning]`` is printed once per threshold period (10 min by default) while the stall lasts.
"""

from __future__ import annotations

import os
import re
import stat
import sys
import threading
import time
from typing import Callable, Optional

HEARTBEAT_SECONDS_ENV = "SLICERDSPACE_HEARTBEAT_SECONDS"
STALL_WARN_MINUTES_ENV = "SLICERDSPACE_STALL_WARN_MINUTES"
EXCLUDE_ENV = "SLICERDSPACE_HEARTBEAT_EXCLUDE"  # os.pathsep-separated files that are not stage progress
DEFAULT_HEARTBEAT_SECONDS = 60.0
DEFAULT_STALL_WARN_MINUTES = 10.0
HEARTBEAT_PREFIX = "[Heartbeat]"
STALL_PREFIX = "[Stall warning]"
_MAX_SCAN = 200000
_FIELD = re.compile(r"(\w+)=(.*?)(?=\s+\w+=|$)")
_STAGE_LINE = re.compile(r"^\[Orchestrator\] (?:Running (?P<run>.+)|(?P<started>.+) started|(?P<done>.+) completed in [\d.]+s)\s*$")
# A tool wrapper that announces itself, e.g. "[UKF] Running UKFTractography (GPU ...)": the stage is its tag.
# "[Preproc] Running: <command>" has a colon and does not match.
_TOOL_LINE = re.compile(r"^\[(?P<tag>[A-Za-z][\w /-]*)\] Running (?!:)")


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, ""))
    except ValueError:
        return default
    return value if value > 0 else default


def format_heartbeat(stage: str, elapsed: float, cpu_s: Optional[float] = None, children: Optional[int] = None,
                     log_bytes: Optional[int] = None, stalled_min: Optional[int] = None) -> str:
    parts = [f"{HEARTBEAT_PREFIX} stage={stage or '-'} elapsed={int(elapsed)}"]
    if cpu_s is not None:
        parts.append(f"cpu_s={cpu_s:.0f}")
    if children is not None:
        parts.append(f"children={children}")
    if log_bytes is not None:
        parts.append(f"log_bytes={log_bytes}")
    if stalled_min is not None:
        parts.append(f"stalled_min={stalled_min}")
    return " ".join(parts)


def format_stall_warning(stage: str, minutes: int) -> str:
    return (f"{STALL_PREFIX} {stage}: no CPU progress and no output for {minutes} min. "
            "The step may be hung; press Cancel to stop it, or wait if the tool is known to be quiet.")


def parse_heartbeat_line(line: str) -> Optional[dict]:
    """Return the fields of a ``[Heartbeat]`` line (numbers as int/float, ``stage`` as text), or None."""
    line = line.strip()
    if not line.startswith(HEARTBEAT_PREFIX):
        return None
    fields = {}
    for key, value in _FIELD.findall(line[len(HEARTBEAT_PREFIX):]):
        try:
            fields[key] = int(value)
        except ValueError:
            try:
                fields[key] = float(value)
            except ValueError:
                fields[key] = value.strip()
    return fields if "stage" in fields else None


def parse_stall_warning(line: str) -> Optional[dict]:
    """Return ``{stage, minutes}`` for a ``[Stall warning]`` line, or None."""
    match = re.match(r"^\[Stall warning\] (?P<stage>.+): no CPU progress and no output for (?P<min>\d+) min\.", line.strip())
    return {"stage": match.group("stage"), "minutes": int(match.group("min"))} if match else None


class _TreeCpu:
    """Cumulative CPU seconds of this process tree; an exited descendant keeps its last sampled value."""

    def __init__(self) -> None:
        self._last: dict = {}
        try:
            import psutil
            self._psutil = psutil
            self.available = True
        except ImportError:
            self._psutil = None
            self.available = False

    def sample(self) -> tuple[Optional[float], Optional[int]]:
        if not self.available:
            return None, None
        psutil = self._psutil
        try:
            root = psutil.Process()
            children = root.children(recursive=True)
        except psutil.Error:
            return sum(self._last.values()), 0
        for process in [root, *children]:
            try:
                times = process.cpu_times()
                self._last[(process.pid, process.create_time())] = times.user + times.system
            except psutil.Error:
                continue
        return sum(self._last.values()), len(children)


def own_output_identities(extra=()) -> set:
    """P-a: ``(device, inode)`` of the files this process's own output goes to, which must not count as progress.

    Each heartbeat line grows the file that stdout is redirected to (the batch script's ``<id>.stdout.log``), so
    the log scanner would see "growth" at every tick. The files are ``sys.stdout`` when it is a regular file,
    and every path in ``SLICERDSPACE_HEARTBEAT_EXCLUDE`` (plus ``extra``).
    """
    found = set()
    try:
        info = os.fstat(sys.stdout.fileno())
        if stat.S_ISREG(info.st_mode):
            found.add((info.st_dev, info.st_ino))
    except (AttributeError, OSError, ValueError):
        pass
    paths = [item for item in os.environ.get(EXCLUDE_ENV, "").split(os.pathsep) if item]
    for path in [*paths, *(str(item) for item in extra)]:
        try:
            info = os.stat(path)
        except OSError:
            continue
        found.add((info.st_dev, info.st_ino))
    return found


def _newest_log_size(root, since_wall: float, exclude=frozenset()) -> Optional[int]:
    """Size of the most recently written ``*.log`` below ``root`` that changed after ``since_wall``.

    A file whose ``(device, inode)`` is in ``exclude`` is skipped (see :func:`own_output_identities`).
    """
    if not root:
        return None
    best, scanned = None, 0
    for folder, _dirs, names in os.walk(root):
        for name in names:
            scanned += 1
            if scanned > _MAX_SCAN:
                return best[1] if best else None
            if not name.endswith(".log"):
                continue
            try:
                info = os.stat(os.path.join(folder, name))
            except OSError:
                continue
            if exclude and (info.st_dev, info.st_ino) in exclude:
                continue
            if info.st_mtime >= since_wall and (best is None or info.st_mtime > best[0]):
                best = (info.st_mtime, info.st_size)
    return best[1] if best else None


class Heartbeat:
    """Daemon thread that prints a heartbeat line per interval; see the module docstring."""

    def __init__(self, *, interval: Optional[float] = None, stall_minutes: Optional[float] = None,
                 out: Optional[Callable[[str], None]] = None, log_root=None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.interval = interval if interval is not None else _env_float(HEARTBEAT_SECONDS_ENV, DEFAULT_HEARTBEAT_SECONDS)
        self.stall_seconds = 60.0 * (stall_minutes if stall_minutes is not None
                                     else _env_float(STALL_WARN_MINUTES_ENV, DEFAULT_STALL_WARN_MINUTES))
        self._out = out or self._write_stdout
        self.log_root = log_root
        self._clock = clock
        self._cpu = _TreeCpu()
        self._lock = threading.Lock()
        self._stage: Optional[str] = None
        self._stage_started = clock()
        self._stage_wall = time.time()
        self._activity = 0
        self._last_cpu: Optional[float] = None
        self._last_log_bytes: Optional[int] = None
        self._last_seen_activity = 0
        self._progress_at = clock()
        self._warned_at: Optional[float] = None
        self._previous_tick = clock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def _write_stdout(line: str) -> None:
        sys.stdout.write(line + "\n")  # one write, so the line is not split by another thread's print
        sys.stdout.flush()

    # --- stage bookkeeping -------------------------------------------------------------------
    def set_stage(self, name: Optional[str]) -> None:
        with self._lock:
            self._stage = name or None
            self._stage_started = self._clock()
            self._stage_wall = time.time()
            self._progress_at = self._clock()
            self._warned_at = None
            self._last_log_bytes = None

    def observe_log_line(self, message: str) -> None:
        """Count a run-log line as output, and follow the current task named by an orchestrator line."""
        with self._lock:
            self._activity += 1
        text = message.strip()
        tool = _TOOL_LINE.match(text)
        if tool and not text.startswith("[Orchestrator]"):
            self.set_stage(tool.group("tag").strip())
            return
        match = _STAGE_LINE.match(text)
        if match:
            if match.group("done"):
                self.set_stage(None)
            else:
                self.set_stage((match.group("run") or match.group("started")).strip())

    def wrap_log(self, log: Optional[Callable[[str], None]]) -> Callable[[str], None]:
        def wrapped(message: str) -> None:
            self.observe_log_line(str(message))
            if log is not None:
                log(message)
        return wrapped

    # --- one sample --------------------------------------------------------------------------
    def tick(self) -> list[str]:
        """Take one sample, emit its lines, and return them."""
        now = self._clock()
        cpu_s, children = self._cpu.sample()
        with self._lock:
            stage, started, since_wall, activity = self._stage, self._stage_started, self._stage_wall, self._activity
        log_bytes = _newest_log_size(self.log_root, since_wall, own_output_identities())
        with self._lock:
            progressed = activity != self._last_seen_activity
            self._last_seen_activity = activity
            if cpu_s is not None:
                gained = cpu_s - (self._last_cpu if self._last_cpu is not None else cpu_s)
                if gained > 0.01 * max(now - self._previous_tick, 0.0):
                    progressed = True
                self._last_cpu = cpu_s
            if log_bytes is not None and log_bytes != self._last_log_bytes:
                if self._last_log_bytes is not None:
                    progressed = True
                self._last_log_bytes = log_bytes
            self._previous_tick = now
            if progressed:
                self._progress_at = now
                self._warned_at = None
            stalled = now - self._progress_at
            warn = stalled >= self.stall_seconds and (
                self._warned_at is None or now - self._warned_at >= self.stall_seconds)
            if warn:
                self._warned_at = now
        stalled_min = int(stalled // 60) if stalled >= self.stall_seconds else None
        lines = [format_heartbeat(stage or "-", now - started, cpu_s, children, log_bytes, stalled_min)]
        if warn:
            lines.append(format_stall_warning(stage or "pipeline (between stages)", int(stalled // 60)))
        for line in lines:
            try:
                self._out(line)
            except Exception:
                pass
        return lines

    # --- thread ------------------------------------------------------------------------------
    def start(self) -> "Heartbeat":
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="run-heartbeat", daemon=True)
            self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.tick()
            except Exception:
                pass

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)
