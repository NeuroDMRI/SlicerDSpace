"""Sample stage process trees and retain bounded, hardware-specific feedback."""

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import threading
import time
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

from shared.cache_lock import cache_lock

_GIB = float(1024 ** 3)


@lru_cache(maxsize=1)
def _implementation():
    root = Path(__file__).resolve().parents[1]
    sources = ('shared/cohort_profile.py', 'shared/cohort_planning.py', 'shared/cohort_schedule.py',
               'shared/cohort_policy.py', 'shared/cohort_resources.py', 'shared/compute_schedule.py',
               'shared/compute_tuning.py', 'shared/numeric_threads.py', 'shared/cohort_targets.py',
               'shared/hardware_adapt.py', 'scripts/autotune_batch.py',
               'preprocessing/dipy_accel.py',
               'microstructure/pipeline.py', 'microstructure/dki.py', 'microstructure/mkcurve.py',
               'microstructure/mkcurve_tensor.py', 'microstructure/mkcurve_inputs.py', 'microstructure/chunk_dispatch.py',
               'microstructure/freewater_dispatch.py', 'microstructure/freewater_wls.py',
               'microstructure/freewater_protocol.py', 'tractography/wma.py',
               'tractography/wma_registration.py', 'tractography/wma_transform.py',
               'tractography/saf.py', 'tractography/dti.py', 'connectome/pipeline.py')
    result = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources}
    for name in ('numpy', 'scipy', 'dipy', 'torch', 'vtk', 'dmri-amico'):
        try:
            result['package:' + name] = version(name)
        except PackageNotFoundError:
            result['package:' + name] = None
    return result


def workload_key(stage, shape, arguments, cpu, gpu):
    # Match numerical settings, data dimensions, CPU reservation and implementation.
    payload = dict(stage=stage, shape=shape, arguments=arguments, cpu=cpu, gpu=gpu,
                   machine=platform.machine(), processor=platform.processor(), node=platform.node(),
                   python=platform.python_version(), code=_implementation())
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def read_history(path):
    if not path or not Path(path).is_file():
        return {}
    try:
        if Path(path).stat().st_size > 8 * 1024**2:
            return {}
        result = json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(result, dict) or result.get('version') != 1 or not isinstance(result.get('profiles'), dict):
            return {}
        result['profiles'] = {key: [row for row in rows if isinstance(row, dict)][-12:]
                              for key, rows in result['profiles'].items() if isinstance(rows, list)}
        return result
    except (OSError, ValueError):
        return {}


def history_estimate(history, key):
    rows = history.get('profiles', {}).get(key, [])[-12:]
    rows = [row for row in rows if row.get('outcome') == 'completed' and not row.get('cache_hit') and row.get('samples', 0) >= 2
            and all(isinstance(row.get(field), (int, float)) and math.isfinite(row[field]) and row[field] > 0
                    for field in ('peak_tree_rss_bytes', 'execution_seconds'))]
    if not rows:
        return None
    return dict(samples=len(rows), memory_gb=max(row['peak_tree_rss_bytes'] for row in rows) / 1024**3 * 1.3,
                duration_seconds=max(row['execution_seconds'] for row in rows) * 1.25)


def record_history(path, key, measurement):
    if not path or not key:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with cache_lock(path.with_suffix('.lock')):
        history = read_history(path) or dict(version=1, profiles={})
        profiles = history.setdefault('profiles', {})
        profiles[key] = (profiles.get(key, []) + [measurement])[-12:]
        # Preserve recent workload keys rather than growing indefinitely.
        while len(profiles) > 512:
            del profiles[next(iter(profiles))]
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(history, indent=2) + '\n', encoding='utf-8')
        tmp.replace(path)


def _finite_float_or_none(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


def _windows_process_counters(pid):
    """Read Windows process memory (WorkingSet, PrivateUsage, PageFaultCount) and I/O via Win32 ctypes."""
    empty = dict(rss_bytes=None, private_bytes=None, page_faults=None, major_page_faults=None,
                 io_read_bytes=None, io_write_bytes=None, cpu_seconds=None)
    if os.name != 'nt':
        return empty
    try:
        import ctypes

        class _ProcessMemoryCountersEx(ctypes.Structure):
            _fields_ = [
                ('cb', ctypes.c_uint32),
                ('PageFaultCount', ctypes.c_uint32),
                ('PeakWorkingSetSize', ctypes.c_size_t),
                ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t),
                ('PeakPagefileUsage', ctypes.c_size_t),
                ('PrivateUsage', ctypes.c_size_t),
            ]

        class _IoCounters(ctypes.Structure):
            _fields_ = [
                ('ReadOperationCount', ctypes.c_uint64),
                ('WriteOperationCount', ctypes.c_uint64),
                ('OtherOperationCount', ctypes.c_uint64),
                ('ReadTransferCount', ctypes.c_uint64),
                ('WriteTransferCount', ctypes.c_uint64),
                ('OtherTransferCount', ctypes.c_uint64),
            ]

        class _FileTime(ctypes.Structure):
            _fields_ = [('dwLowDateTime', ctypes.c_uint32), ('dwHighDateTime', ctypes.c_uint32)]

        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int

        is_self = int(pid) == os.getpid()
        handle = kernel32.GetCurrentProcess() if is_self else kernel32.OpenProcess(0x1000, 0, int(pid))
        if not handle:
            return empty
        try:
            mem_fn = getattr(kernel32, 'K32GetProcessMemoryInfo', None)
            if mem_fn is None:
                try:
                    mem_fn = getattr(ctypes.WinDLL('psapi', use_last_error=True), 'GetProcessMemoryInfo', None)
                except OSError:
                    mem_fn = None
            if mem_fn is not None:
                mem_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
                mem_fn.restype = ctypes.c_int
                pmc = _ProcessMemoryCountersEx()
                pmc.cb = ctypes.sizeof(_ProcessMemoryCountersEx)
                if mem_fn(handle, ctypes.byref(pmc), pmc.cb):
                    empty['rss_bytes'] = int(pmc.WorkingSetSize)
                    empty['private_bytes'] = int(pmc.PrivateUsage)
                    empty['page_faults'] = int(pmc.PageFaultCount)
            io_fn = getattr(kernel32, 'GetProcessIoCounters', None)
            if io_fn is not None:
                io_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                io_fn.restype = ctypes.c_int
                io_cnt = _IoCounters()
                if io_fn(handle, ctypes.byref(io_cnt)):
                    empty['io_read_bytes'] = int(io_cnt.ReadTransferCount)
                    empty['io_write_bytes'] = int(io_cnt.WriteTransferCount)
            times_fn = getattr(kernel32, 'GetProcessTimes', None)
            if times_fn is not None:
                times_fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
                times_fn.restype = ctypes.c_int
                ft_c, ft_e, ft_k, ft_u = _FileTime(), _FileTime(), _FileTime(), _FileTime()
                if times_fn(handle, ctypes.byref(ft_c), ctypes.byref(ft_e), ctypes.byref(ft_k), ctypes.byref(ft_u)):
                    k_100ns = (ft_k.dwHighDateTime << 32) | ft_k.dwLowDateTime
                    u_100ns = (ft_u.dwHighDateTime << 32) | ft_u.dwLowDateTime
                    empty['cpu_seconds'] = float(k_100ns + u_100ns) / 1e7
        finally:
            if not is_self:
                kernel32.CloseHandle(handle)
    except Exception:
        pass
    return empty


def _linux_process_counters(pid, *, proc_root=Path('/proc'), page_size=None, clk_tck=None):
    """Parse Linux ``/proc/<pid>/stat``, ``/proc/<pid>/status``, and ``/proc/<pid>/io``."""
    out = dict(rss_bytes=None, private_bytes=None, page_faults=None, major_page_faults=None,
               io_read_bytes=None, io_write_bytes=None, cpu_seconds=None, create_token=None)
    pdir = Path(proc_root) / str(int(pid))
    if page_size is None:
        try:
            page_size = int(os.sysconf('SC_PAGE_SIZE')) if hasattr(os, 'sysconf') else 4096
        except (OSError, ValueError):
            page_size = 4096
    if clk_tck is None:
        try:
            clk_tck = int(os.sysconf('SC_CLK_TCK')) if hasattr(os, 'sysconf') else 100
        except (OSError, ValueError):
            clk_tck = 100
    try:
        raw_stat = (pdir / 'stat').read_text(encoding='utf-8')
        rparen = raw_stat.rfind(')')
        if rparen >= 0:
            fields = raw_stat[rparen + 2:].split()
            # fields[0] is proc stat field #3 (state).
            if len(fields) > 21:
                minflt = int(fields[7])
                majflt = int(fields[9])
                utime = int(fields[11])
                stime = int(fields[12])
                starttime = int(fields[19])
                rss_pages = int(fields[21])
                out['page_faults'] = max(0, minflt + majflt)
                out['major_page_faults'] = max(0, majflt)
                out['cpu_seconds'] = float(utime + stime) / float(max(1, clk_tck))
                out['rss_bytes'] = max(0, rss_pages) * int(page_size)
                out['create_token'] = starttime
    except (OSError, ValueError):
        pass
    try:
        for line in (pdir / 'status').read_text(encoding='utf-8').splitlines():
            if ':' not in line:
                continue
            key, val = line.split(':', 1)
            tokens = val.strip().split()
            if not tokens:
                continue
            if key == 'RssAnon':
                out['private_bytes'] = int(tokens[0]) * 1024
            elif key == 'VmRSS' and out['rss_bytes'] is None:
                out['rss_bytes'] = int(tokens[0]) * 1024
    except (OSError, ValueError):
        pass
    try:
        for line in (pdir / 'io').read_text(encoding='utf-8').splitlines():
            if ':' not in line:
                continue
            key, val = line.split(':', 1)
            key = key.strip()
            if key == 'read_bytes':
                out['io_read_bytes'] = int(val.strip())
            elif key == 'write_bytes':
                out['io_write_bytes'] = int(val.strip())
    except (OSError, ValueError):
        pass
    return out


def _linux_descendant_pids(root_pid, *, proc_root=Path('/proc')):
    """Return ``[root_pid, ...descendants]`` using ``/proc/<pid>/task/*/children`` without psutil."""
    ordered = []
    seen = set()
    queue = [int(root_pid)]
    while queue:
        pid = queue.pop(0)
        if pid in seen:
            continue
        seen.add(pid)
        ordered.append(pid)
        task_dir = Path(proc_root) / str(pid) / 'task'
        try:
            for child_file in task_dir.glob('*/children'):
                try:
                    for token in child_file.read_text(encoding='utf-8').split():
                        if token.isdigit():
                            cid = int(token)
                            if cid not in seen:
                                queue.append(cid)
                except OSError:
                    continue
        except OSError:
            continue
    return ordered


def _macos_rusage_counters():
    """Best-effort macOS/POSIX stdlib ``resource.getrusage`` counters for the current process."""
    out = dict(page_faults=None, major_page_faults=None, cpu_seconds=None, rss_bytes=None)
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        maj = max(0, int(getattr(usage, 'ru_majflt', 0)))
        minf = max(0, int(getattr(usage, 'ru_minflt', 0)))
        out['major_page_faults'] = maj
        out['page_faults'] = maj + minf
        out['cpu_seconds'] = float(getattr(usage, 'ru_utime', 0.0) + getattr(usage, 'ru_stime', 0.0))
        maxrss = int(getattr(usage, 'ru_maxrss', 0))
        if maxrss > 0:
            # macOS ru_maxrss is in bytes; Linux/BSD is in KiB.
            out['rss_bytes'] = maxrss if sys.platform == 'darwin' else maxrss * 1024
    except Exception:
        pass
    return out


def _default_system_memory_snapshot():
    try:
        from shared.hardware_adapt import query_host_memory_snapshot

        return query_host_memory_snapshot(respect_env_cap=False)
    except Exception:
        return {}


class StageMonitor:
    """Sample process-tree RSS/private memory, page faults, I/O, active stage, and system headroom."""

    def __init__(
        self,
        interval=.2,
        *,
        stage=None,
        stage_provider=None,
        timeline_path=None,
        max_timeline_samples=2048,
        memory_snapshot_fn=None,
    ):
        self.interval = interval
        self.stop_event = threading.Event()
        self.peak = 0
        self.peak_private_bytes = None
        self.samples = 0
        self.cpu = {}
        self.io = {}
        self.io_read = {}
        self.io_write = {}
        self.page_faults = {}
        self.major_page_faults = {}
        self.start_cpu = {}
        self.start_io = {}
        self.start_io_read = {}
        self.start_io_write = {}
        self.start_page_faults = {}
        self.start_major_page_faults = {}
        self.started = time.monotonic()
        self.thread = None
        self.latest = {}
        self.timeline = []
        self.phase_transitions = []
        self._lock = threading.Lock()
        self._stage = stage
        self._stage_provider = stage_provider
        self.timeline_path = Path(timeline_path) if timeline_path else None
        self._max_timeline_samples = max(16, int(max_timeline_samples))
        self._memory_snapshot_fn = memory_snapshot_fn or _default_system_memory_snapshot
        self._min_sys_avail_gb = None
        self._min_sys_commit_avail_gb = None
        self._min_sys_swap_free_gb = None
        self._initial_sys_swap_free_gb = None
        self._latest_sys_swap_free_gb = None
        if stage is not None:
            self.phase_transitions.append(dict(stage=stage, time=time.time(), elapsed_seconds=0.0))

    def set_stage(self, stage):
        """Update the active stage/phase label and record a timestamped transition."""
        now_epoch = time.time()
        elapsed = time.monotonic() - self.started
        with self._lock:
            if stage != self._stage:
                self._stage = stage
                self.phase_transitions.append(dict(stage=stage, time=now_epoch, elapsed_seconds=elapsed))

    def _current_stage(self, now_epoch, elapsed):
        if callable(self._stage_provider):
            try:
                provided = self._stage_provider()
            except Exception:
                provided = None
            if provided is not None:
                with self._lock:
                    if provided != self._stage:
                        self._stage = provided
                        self.phase_transitions.append(dict(stage=provided, time=now_epoch, elapsed_seconds=elapsed))
        with self._lock:
            return self._stage

    def _collect_process_rows(self):
        """Collect per-process metrics across owner + descendants using psutil and OS-native APIs."""
        rows = []
        psutil = None
        try:
            import psutil as _psutil
            psutil = _psutil
        except ImportError:
            psutil = None

        if psutil is not None:
            try:
                owner = psutil.Process(os.getpid())
                processes = [owner, *owner.children(recursive=True)]
            except psutil.Error:
                processes = []
            for process in processes:
                try:
                    pid = int(process.pid)
                    ident = (pid, process.create_time())
                    mem_info = process.memory_info()
                    rss = int(mem_info.rss)
                    private_b = getattr(mem_info, 'private', None)
                    if private_b is not None:
                        private_b = int(private_b)
                    pf = getattr(mem_info, 'num_page_faults', None)
                    if pf is None:
                        pf = getattr(mem_info, 'pfaults', None)
                    pf = int(pf) if pf is not None else None
                    majflt = getattr(mem_info, 'pageins', None)
                    majflt = int(majflt) if majflt is not None else None
                    times = process.cpu_times()
                    cpu_sec = float(times.user + times.system)
                    io_read_b = None
                    io_write_b = None
                    try:
                        counters = process.io_counters()
                        io_read_b = int(counters.read_bytes)
                        io_write_b = int(counters.write_bytes)
                    except (psutil.Error, AttributeError):
                        pass

                    if os.name == 'nt' and (private_b is None or pf is None or io_read_b is None):
                        native = _windows_process_counters(pid)
                        if private_b is None:
                            private_b = native['private_bytes']
                        if pf is None:
                            pf = native['page_faults']
                        if io_read_b is None:
                            io_read_b = native['io_read_bytes']
                        if io_write_b is None:
                            io_write_b = native['io_write_bytes']
                    elif sys.platform.startswith('linux'):
                        native = _linux_process_counters(pid)
                        if private_b is None:
                            private_b = native['private_bytes']
                        if pf is None:
                            pf = native['page_faults']
                        if majflt is None:
                            majflt = native['major_page_faults']
                        if io_read_b is None:
                            io_read_b = native['io_read_bytes']
                        if io_write_b is None:
                            io_write_b = native['io_write_bytes']
                    elif sys.platform == 'darwin' and pid == os.getpid() and (pf is None or majflt is None):
                        native = _macos_rusage_counters()
                        if pf is None:
                            pf = native['page_faults']
                        if majflt is None:
                            majflt = native['major_page_faults']

                    rows.append(dict(
                        pid=pid, ident=ident, rss_bytes=rss, private_bytes=private_b,
                        cpu_seconds=cpu_sec, io_read_bytes=io_read_b, io_write_bytes=io_write_b,
                        page_faults=pf, major_page_faults=majflt,
                    ))
                except psutil.Error:
                    continue
            return rows

        # Zero-dependency fallback when psutil is unavailable.
        pids = _linux_descendant_pids(os.getpid()) if sys.platform.startswith('linux') else [os.getpid()]
        for pid in pids:
            if os.name == 'nt':
                native = _windows_process_counters(pid)
                ident = (pid, 0)
            elif sys.platform.startswith('linux'):
                native = _linux_process_counters(pid)
                ident = (pid, native.get('create_token') or 0)
            else:
                native = _macos_rusage_counters()
                ident = (pid, 0)
            if native.get('rss_bytes') is None and native.get('cpu_seconds') is None:
                continue
            rows.append(dict(
                pid=pid,
                ident=ident,
                rss_bytes=int(native.get('rss_bytes') or 0),
                private_bytes=native.get('private_bytes'),
                cpu_seconds=float(native.get('cpu_seconds') or 0.0),
                io_read_bytes=native.get('io_read_bytes'),
                io_write_bytes=native.get('io_write_bytes'),
                page_faults=native.get('page_faults'),
                major_page_faults=native.get('major_page_faults'),
            ))
        return rows

    def sample(self):
        now_epoch = time.time()
        elapsed = time.monotonic() - self.started
        stage = self._current_stage(now_epoch, elapsed)
        proc_rows = self._collect_process_rows()
        if not proc_rows:
            return

        rss = 0
        private_total = 0
        have_private = False
        pids = []
        for proc in proc_rows:
            ident = proc['ident']
            pids.append(proc['pid'])
            rss += int(proc['rss_bytes'])
            if proc['private_bytes'] is not None:
                private_total += int(proc['private_bytes'])
                have_private = True
            cpu_val = float(proc['cpu_seconds'])
            self.start_cpu.setdefault(ident, cpu_val if self.samples == 0 else 0.0)
            self.cpu[ident] = cpu_val
            if proc['io_read_bytes'] is not None or proc['io_write_bytes'] is not None:
                r_b = int(proc['io_read_bytes'] or 0)
                w_b = int(proc['io_write_bytes'] or 0)
                tot_io = r_b + w_b
                self.start_io.setdefault(ident, tot_io if self.samples == 0 else 0)
                self.io[ident] = tot_io
                self.start_io_read.setdefault(ident, r_b if self.samples == 0 else 0)
                self.io_read[ident] = r_b
                self.start_io_write.setdefault(ident, w_b if self.samples == 0 else 0)
                self.io_write[ident] = w_b
            if proc['page_faults'] is not None:
                pf_val = int(proc['page_faults'])
                self.start_page_faults.setdefault(ident, pf_val if self.samples == 0 else 0)
                self.page_faults[ident] = pf_val
            if proc['major_page_faults'] is not None:
                mpf_val = int(proc['major_page_faults'])
                self.start_major_page_faults.setdefault(ident, mpf_val if self.samples == 0 else 0)
                self.major_page_faults[ident] = mpf_val

        self.peak = max(self.peak, rss)
        if have_private:
            self.peak_private_bytes = private_total if self.peak_private_bytes is None else max(self.peak_private_bytes, private_total)
        self.samples += 1

        try:
            sys_mem = self._memory_snapshot_fn() or {}
        except Exception:
            sys_mem = {}
        sys_avail_gb = _finite_float_or_none(sys_mem.get('available_ram_gib'))
        sys_commit_limit_gb = _finite_float_or_none(sys_mem.get('commit_limit_gib'))
        sys_commit_total_gb = _finite_float_or_none(sys_mem.get('commit_total_gib'))
        sys_commit_avail_gb = _finite_float_or_none(sys_mem.get('commit_available_gib'))
        sys_swap_total_gb = _finite_float_or_none(sys_mem.get('swap_total_gib'))
        sys_swap_free_gb = _finite_float_or_none(sys_mem.get('swap_free_gib'))
        sys_eff_avail_gb = _finite_float_or_none(sys_mem.get('effective_available_ram_gib'))

        if sys_avail_gb is not None:
            self._min_sys_avail_gb = sys_avail_gb if self._min_sys_avail_gb is None else min(self._min_sys_avail_gb, sys_avail_gb)
        if sys_commit_avail_gb is not None:
            self._min_sys_commit_avail_gb = (
                sys_commit_avail_gb if self._min_sys_commit_avail_gb is None else min(self._min_sys_commit_avail_gb, sys_commit_avail_gb)
            )
        if sys_swap_free_gb is not None:
            if self._initial_sys_swap_free_gb is None:
                self._initial_sys_swap_free_gb = sys_swap_free_gb
            self._latest_sys_swap_free_gb = sys_swap_free_gb
            self._min_sys_swap_free_gb = (
                sys_swap_free_gb if self._min_sys_swap_free_gb is None else min(self._min_sys_swap_free_gb, sys_swap_free_gb)
            )

        cpu_total = sum(max(0.0, value - self.start_cpu[key]) for key, value in self.cpu.items())
        io_total = sum(max(0, value - self.start_io[key]) for key, value in self.io.items())
        io_read_total = (
            sum(max(0, value - self.start_io_read[key]) for key, value in self.io_read.items())
            if self.io_read else None
        )
        io_write_total = (
            sum(max(0, value - self.start_io_write[key]) for key, value in self.io_write.items())
            if self.io_write else None
        )
        pf_total = (
            sum(max(0, value - self.start_page_faults[key]) for key, value in self.page_faults.items())
            if self.page_faults else None
        )
        mpf_total = (
            sum(max(0, value - self.start_major_page_faults[key]) for key, value in self.major_page_faults.items())
            if self.major_page_faults else None
        )

        # Keep ``self.latest`` free of 'time'/'elapsed_seconds' so CohortTelemetry.sample()
        # can merge ``dict(time=..., elapsed_seconds=..., **self.monitor.latest)`` cleanly.
        snapshot = dict(
            rss_bytes=rss,
            rss_gb=round(rss / _GIB, 6),
            private_bytes=private_total if have_private else None,
            private_gb=round(private_total / _GIB, 6) if have_private else None,
            process_count=len(pids),
            pids=pids,
            cpu_seconds=cpu_total,
            io_bytes=io_total,
            io_read_bytes=io_read_total,
            io_write_bytes=io_write_total,
            page_faults=pf_total,
            major_page_faults=mpf_total,
            stage=stage,
            system_available_ram_gb=sys_avail_gb,
            system_commit_limit_gb=sys_commit_limit_gb,
            system_commit_total_gb=sys_commit_total_gb,
            system_commit_available_gb=sys_commit_avail_gb,
            system_swap_total_gb=sys_swap_total_gb,
            system_swap_free_gb=sys_swap_free_gb,
            system_effective_available_ram_gb=sys_eff_avail_gb,
        )
        self.latest = snapshot

        timeline_row = dict(time=now_epoch, elapsed_seconds=elapsed, **{k: v for k, v in snapshot.items() if k != 'pids'})
        if len(self.timeline) >= self._max_timeline_samples:
            # Retain bounded timeline by keeping first point, every even interior point, and latest.
            self.timeline = [self.timeline[0], *self.timeline[1:-1:2], self.timeline[-1]]
        self.timeline.append(timeline_row)
        if self.timeline_path is not None:
            try:
                self.timeline_path.parent.mkdir(parents=True, exist_ok=True)
                with self.timeline_path.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(timeline_row, allow_nan=False) + '\n')
            except OSError:
                pass

    def start(self):
        self.sample()
        def run():
            while not self.stop_event.wait(self.interval):
                self.sample()
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        return self

    def finish(self, outcome):
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        self.sample()
        elapsed = time.monotonic() - self.started
        cpu = sum(max(0.0, value - self.start_cpu[key]) for key, value in self.cpu.items())
        io_total = sum(max(0, value - self.start_io[key]) for key, value in self.io.items())
        io_read_total = (
            sum(max(0, value - self.start_io_read[key]) for key, value in self.io_read.items())
            if self.io_read else None
        )
        io_write_total = (
            sum(max(0, value - self.start_io_write[key]) for key, value in self.io_write.items())
            if self.io_write else None
        )
        pf_total = (
            sum(max(0, value - self.start_page_faults[key]) for key, value in self.page_faults.items())
            if self.page_faults else None
        )
        mpf_total = (
            sum(max(0, value - self.start_major_page_faults[key]) for key, value in self.major_page_faults.items())
            if self.major_page_faults else None
        )
        swap_used_delta_gb = None
        if self._initial_sys_swap_free_gb is not None and self._min_sys_swap_free_gb is not None:
            swap_used_delta_gb = round(max(0.0, self._initial_sys_swap_free_gb - self._min_sys_swap_free_gb), 6)
        swap_or_commit_pressure = bool(
            (self._min_sys_commit_avail_gb is not None and self._min_sys_commit_avail_gb < 2.0)
            or (self._min_sys_avail_gb is not None and self._min_sys_avail_gb < 1.5)
            or (swap_used_delta_gb is not None and swap_used_delta_gb >= 0.25)
            or (mpf_total is not None and mpf_total >= 1000)
        )
        with self._lock:
            stage = self._stage
            transitions = list(self.phase_transitions)
        return dict(
            outcome=outcome,
            samples=self.samples,
            execution_seconds=elapsed,
            peak_tree_rss_bytes=self.peak,
            peak_tree_rss_gb=round(self.peak / _GIB, 6),
            peak_tree_private_bytes=self.peak_private_bytes,
            peak_tree_private_gb=round(self.peak_private_bytes / _GIB, 6) if self.peak_private_bytes is not None else None,
            sampled_cpu_seconds=cpu,
            mean_cpu_cores=cpu / max(elapsed, 1e-9),
            sampled_io_bytes=io_total,
            sampled_io_read_bytes=io_read_total,
            sampled_io_write_bytes=io_write_total,
            sampled_page_faults=pf_total,
            sampled_major_page_faults=mpf_total,
            min_system_available_ram_gb=self._min_sys_avail_gb,
            min_system_commit_available_gb=self._min_sys_commit_avail_gb,
            min_system_swap_free_gb=self._min_sys_swap_free_gb,
            swap_used_delta_gb=swap_used_delta_gb,
            swap_or_commit_pressure=swap_or_commit_pressure,
            stage=stage,
            phase_transitions=transitions,
            timeline_samples_retained=len(self.timeline),
            timeline_path=str(self.timeline_path) if self.timeline_path else None,
            interval_seconds=self.interval,
            scope='Sampled owner plus descendants; short-lived processes and between-sample peaks may be missed',
        )
