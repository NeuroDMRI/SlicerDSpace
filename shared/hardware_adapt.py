"""Hardware-adaptive chunking, VRAM budgeting, and OOM fallback helpers.

Designed so SlicerDSpace runs smoothly across both modest user workstations/laptops
(4-8 CPU cores, 8-16 GiB RAM, CPU-only or 4-8 GiB consumer GPUs) and high-end
multi-GPU servers without changing numerical results.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
from typing import Any, Callable


GIB = 1024 ** 3
MIB = 1024 ** 2
WINDOWS_CUDA_ALLOC_CONF = "garbage_collection_threshold:0.8,max_split_size_mb:128"


def find_nvidia_smi() -> str | None:
    """Locate ``nvidia-smi`` across PATH and standard Windows DriverStore / NVSMI / WSL locations."""
    found = shutil.which("nvidia-smi")
    if found:
        return found
    candidates: list[Path] = []
    if sys.platform == "win32":
        sys_root = Path(os.environ.get("SystemRoot") or r"C:\Windows")
        candidates.append(sys_root / "System32" / "nvidia-smi.exe")
        for prog_env in ("ProgramFiles", "ProgramW6432"):
            prog = os.environ.get(prog_env)
            if prog:
                candidates.append(Path(prog) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe")
        candidates.append(Path(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe"))
    elif sys.platform.startswith("linux"):
        candidates.append(Path("/usr/lib/wsl/lib/nvidia-smi"))
    for cand in candidates:
        try:
            if cand.is_file():
                return str(cand)
        except OSError:
            continue
    return None


def global_vram_cap_bytes() -> int | None:
    """Read optional global VRAM cap in bytes from ``SLICERDSPACE_MAX_VRAM_GIB`` (or alias ``SLICERDSPACE_MAX_VRAM_GB``)."""
    import math

    for env_name in ("SLICERDSPACE_MAX_VRAM_GIB", "SLICERDSPACE_MAX_VRAM_GB"):
        raw = os.environ.get(env_name, "").strip()
        if not raw:
            continue
        try:
            val_gib = float(raw)
        except ValueError:
            continue
        if math.isfinite(val_gib) and val_gib > 0.0:
            cap_b = int(val_gib * GIB)
            if cap_b > 0:
                return cap_b
    return None


def _apply_vram_cap(free_b: int, total_b: int, *, used_b: int = 0) -> tuple[int, int]:
    """Clamp ``(free_bytes, total_bytes)`` to any global VRAM cap while keeping ``0 <= free <= total``."""
    free_i = max(0, int(free_b))
    total_i = max(0, int(total_b))
    cap_b = global_vram_cap_bytes()
    if cap_b is not None:
        total_i = min(total_i, cap_b)
        remaining_cap = max(0, cap_b - max(0, int(used_b)))
        free_i = min(free_i, cap_b, remaining_cap)
    if total_i > 0:
        free_i = min(free_i, total_i)
    return int(free_i), int(total_i)


def query_cuda_free_total_bytes(device: Any = None) -> tuple[int, int] | None:
    """Return ``(free_bytes, total_bytes)`` for ``device``, or ``None`` if unavailable."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        dev = torch.device("cuda" if device is None else device)
        if dev.type != "cuda":
            return None
        idx = torch.cuda.current_device() if dev.index is None else int(dev.index)
        free_b, total_b = torch.cuda.mem_get_info(idx)
        used_b = 0
        try:
            mem_alloc_fn = getattr(torch.cuda, "memory_allocated", None)
            if callable(mem_alloc_fn):
                used_b = max(0, int(mem_alloc_fn(idx)))
        except Exception:
            used_b = 0
        return _apply_vram_cap(int(free_b), int(total_b), used_b=used_b)
    except Exception:
        return None


def is_mps_available(torch_mod: Any = None) -> bool:
    """Return True when PyTorch Apple Metal (MPS) backend is available."""
    try:
        mod = torch_mod
        if mod is None:
            import torch as mod

        mps_mod = getattr(getattr(mod, "backends", None), "mps", None)
        fn = getattr(mps_mod, "is_available", None)
        return bool(callable(fn) and fn() is True)
    except Exception:
        return False


def query_mps_free_total_bytes() -> tuple[int, int] | None:
    """Return conservative ``(free_bytes, total_bytes)`` for Apple MPS unified memory."""
    try:
        import torch

        if not is_mps_available(torch):
            return None
        mps_ns = getattr(torch, "mps", None)
        rec_fn = getattr(mps_ns, "recommended_max_memory", None)
        total_b = int(rec_fn()) if callable(rec_fn) else 0
        if total_b <= 0:
            total_b = max(int(2 * GIB), int(query_host_available_ram_bytes() * 0.60))
        cur_fn = getattr(mps_ns, "current_allocated_memory", None)
        drv_fn = getattr(mps_ns, "driver_allocated_memory", None)
        used_cur = int(cur_fn()) if callable(cur_fn) else 0
        used_drv = int(drv_fn()) if callable(drv_fn) else 0
        used_b = max(0, used_cur, used_drv)
        if global_vram_cap_bytes() is not None:
            return _apply_vram_cap(max(0, total_b - used_b), total_b, used_b=used_b)
        free_b = min(total_b, max(int(64 * MIB), total_b - used_b))
        return int(free_b), int(total_b)
    except Exception:
        return None


def query_gpu_free_total_bytes(device: Any = None) -> tuple[int, int] | None:
    """Return ``(free_bytes, total_bytes)`` for CUDA or MPS ``device``."""
    info: tuple[int, int] | None = None
    try:
        import torch

        if device is not None:
            dev = device if isinstance(device, torch.device) else torch.device(str(device))
            if dev.type == "mps":
                info = query_mps_free_total_bytes()
            elif dev.type == "cuda":
                info = query_cuda_free_total_bytes(dev)
            else:
                return None
            return _apply_vram_cap(info[0], info[1]) if info is not None else None
    except Exception:
        pass
    info = query_cuda_free_total_bytes(device)
    if info is None:
        info = query_mps_free_total_bytes()
    return _apply_vram_cap(info[0], info[1]) if info is not None else None


_DATACENTER_HIGH_FP64_GPU_MARKERS = (
    "H100",
    "H200",
    "H800",
    "A100",
    "A800",
    "A30",
    "V100",
    "GV100",
    "P100",
    "GP100",
    "B100",
    "B200",
    "GB200",
    "GH200",
    "TITAN V",
    "TESLA K",
)


def is_consumer_fp64_cuda_gpu(device: Any = None) -> bool:
    """Return True when ``device`` is a consumer/workstation NVIDIA GPU with 1/64 FP64 throughput.

    Datacenter GPUs with 1/2 FP64 units (H100, H200, A100, A30, V100, P100, B100/B200,
    TITAN V) return False so server FP64 workloads stay at full throughput.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        dev = torch.device("cuda" if device is None else device)
        if dev.type != "cuda":
            return False
        idx = torch.cuda.current_device() if dev.index is None else int(dev.index)
        name = str(torch.cuda.get_device_name(idx) or "").upper()
        if not name:
            return os.name == "nt"
        if any(marker in name for marker in _DATACENTER_HIGH_FP64_GPU_MARKERS):
            return False
        return True
    except Exception:
        return False


def should_enable_cudnn_benchmark(
    device: Any = None,
    *,
    min_free_gib: float = 6.0,
    windows_min_free_gib: float = 8.0,
) -> bool:
    """Enable cuDNN benchmark only when free VRAM comfortably covers algorithm-search workspaces.

    On 4-8 GiB Windows WDDM GPUs (where DWM + Slicer 3D OpenGL leave only 3.0-6.8 GiB free),
    ``cudnn.benchmark = True`` can allocate large temporary workspaces that trigger OOM or
    silent WDDM shared-RAM spillover.
    """
    info = query_cuda_free_total_bytes(device)
    if info is None:
        return os.name != "nt"
    free_b, _ = info
    threshold_gib = float(windows_min_free_gib) if os.name == "nt" else float(min_free_gib)
    return free_b >= int(threshold_gib * GIB)


def is_cuda_oom(exc: BaseException) -> bool:
    """Return True when ``exc`` represents a CUDA or Apple MPS out-of-memory condition."""
    msg = str(exc).lower()
    if "cudaerrormemoryallocation" in msg:
        return True
    if ("out of memory" in msg or "alloc_failed" in msg) and (
        "cuda" in msg or "cublas" in msg or "cudnn" in msg or "cusolver" in msg or "alloc" in msg or "mps" in msg
    ):
        return True
    if "mps backend out of memory" in msg or "invalid buffer size" in msg or "mtlbuffer" in msg:
        return True
    if "metal" in msg and "memory" in msg:
        return True
    cls_name = type(exc).__name__.lower()
    return "outofmemory" in cls_name


is_gpu_oom = is_cuda_oom


def empty_gpu_cache_safely() -> None:
    """Synchronize/release cached GPU allocator blocks on initialized CUDA or MPS backends."""
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        if is_mps_available(torch):
            mps_ns = getattr(torch, "mps", None)
            sync_fn = getattr(mps_ns, "synchronize", None)
            empty_fn = getattr(mps_ns, "empty_cache", None)
            if callable(sync_fn):
                sync_fn()
            if callable(empty_fn):
                empty_fn()
    except Exception:
        pass


def empty_cuda_cache_safely() -> None:
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    except Exception:
        pass


def adaptive_cuda_batch_size(
    per_item_bytes: int,
    default_batch: int,
    *,
    min_batch: int = 64,
    max_batch: int = 131072,
    safety_fraction: float = 0.30,
    device: Any = None,
) -> int:
    """Scale CUDA batch/chunk size to fit current free VRAM with safety margin."""
    base = max(int(min_batch), min(int(max_batch), int(default_batch)))
    info = query_cuda_free_total_bytes(device)
    if info is None:
        return base
    free_b, _ = info
    budget = max(int(64 * MIB), int(free_b * float(safety_fraction)))
    denom = max(1, int(per_item_bytes))
    fitted = budget // denom
    return max(int(min_batch), min(int(max_batch), min(base, int(fitted)) if free_b < 12 * GIB else min(int(max_batch), max(base, min(base, int(fitted))))))


def adaptive_gpu_scaled_batch(
    per_item_bytes: int,
    default_batch: int,
    *,
    min_batch: int = 64,
    max_batch: int = 131072,
    safety_fraction: float = 0.30,
    device: Any = None,
) -> int:
    """Scale batch size down on low-memory CUDA or MPS GPUs while preserving ``default_batch`` when memory suffices."""
    base = max(int(min_batch), min(int(max_batch), int(default_batch)))
    info = query_gpu_free_total_bytes(device)
    if info is None:
        return base
    free_b, _ = info
    budget = max(int(32 * MIB), int(free_b * float(safety_fraction)))
    max_safe = max(int(min_batch), budget // max(1, int(per_item_bytes)))
    return max(int(min_batch), min(base, max_safe))


def adaptive_cuda_scaled_batch(
    per_item_bytes: int,
    default_batch: int,
    *,
    min_batch: int = 64,
    max_batch: int = 131072,
    safety_fraction: float = 0.30,
    device: Any = None,
    windows_tdr_cap: int | None = None,
    fp64: bool = False,
    consumer_fp64_tdr_cap: int | None = None,
) -> int:
    """Scale batch size down on low-VRAM GPUs while keeping ``default_batch`` when VRAM suffices.

    On Windows WDDM (``os.name == 'nt'``), bounds batch sizes by ``windows_tdr_cap`` (and further by
    ``consumer_fp64_tdr_cap`` when ``fp64=True`` on 1/64-FP64 consumer GeForce GPUs) to avoid
    2-second WDDM TDR driver resets and silent shared-GPU-memory spillover into system RAM.
    On Linux/POSIX (including H100/A100 servers), TDR caps are never applied.
    """
    base = max(int(min_batch), min(int(max_batch), int(default_batch)))
    if os.name == "nt":
        if windows_tdr_cap is not None:
            base = max(int(min_batch), min(base, int(windows_tdr_cap)))
        if fp64 and is_consumer_fp64_cuda_gpu(device):
            fp64_cap = consumer_fp64_tdr_cap
            if fp64_cap is None and windows_tdr_cap is not None:
                fp64_cap = max(int(min_batch), int(windows_tdr_cap) // 4)
            if fp64_cap is not None:
                base = max(int(min_batch), min(base, int(fp64_cap)))
    info = query_cuda_free_total_bytes(device)
    if info is None:
        return base
    free_b, _ = info
    eff_safety = float(safety_fraction)
    if os.name == "nt" and free_b < 10 * GIB:
        # Leave extra headroom on 4-8 GiB WDDM GPUs so DWM + Slicer OpenGL never push
        # PyTorch allocations into WDDM shared system RAM (20x slowdown).
        eff_safety = min(eff_safety, 0.25)
    budget = max(int(32 * MIB), int(free_b * eff_safety))
    max_safe = max(int(min_batch), budget // max(1, int(per_item_bytes)))
    return max(int(min_batch), min(base, max_safe))


def run_cuda_chunked_with_oom_fallback(
    n_items: int,
    initial_chunk: int | None = None,
    cuda_step_fn: Callable[[int, int], Any] | None = None,
    *,
    min_chunk: int = 64,
    cpu_fallback_fn: Callable[[int, int], Any] | None = None,
    initial_chunk_size: int | None = None,
    min_cuda_chunk_size: int | None = None,
    cuda_fn: Callable[[int, int], Any] | None = None,
) -> Any:
    """Execute ``cuda_step_fn(start, stop)`` or value-returning ``cuda_fn(start, stop)`` across ``[0, n_items)``, halving chunk size on CUDA OOM."""
    return_list = cuda_fn is not None
    step_fn = cuda_fn if cuda_fn is not None else cuda_step_fn
    if step_fn is None:
        raise ValueError("Either cuda_step_fn or cuda_fn must be provided.")
    eff_init = int(initial_chunk_size if initial_chunk_size is not None else (initial_chunk if initial_chunk is not None else 1024))
    eff_min = int(min_cuda_chunk_size if min_cuda_chunk_size is not None else min_chunk)

    total = int(n_items)
    if total <= 0:
        return [] if return_list else {"chunks": 0, "oom_retries": 0, "cpu_fallback_items": 0, "final_chunk": eff_init}
    chunk = max(eff_min, eff_init)
    floor = max(1, eff_min)
    start = 0
    chunks = 0
    oom_retries = 0
    cpu_items = 0
    collected: list[Any] = []
    while start < total:
        stop = min(total, start + chunk)
        try:
            res = step_fn(start, stop)
            if return_list:
                collected.append(res)
            start = stop
            chunks += 1
        except BaseException as exc:
            if not is_cuda_oom(exc):
                raise
            oom_retries += 1
            empty_cuda_cache_safely()
            if chunk > floor:
                chunk = max(floor, chunk // 2)
                continue
            if cpu_fallback_fn is not None:
                res_fb = cpu_fallback_fn(start, total)
                if return_list:
                    collected.append(res_fb)
                cpu_items = total - start
                start = total
                break
            raise
    if return_list:
        return collected
    return {
        "chunks": chunks,
        "oom_retries": oom_retries,
        "cpu_fallback_items": cpu_items,
        "final_chunk": chunk,
    }


def query_host_available_ram_bytes() -> int:
    """Return conservative available host RAM in bytes."""
    cap_b: int | None = None
    cap_env = os.environ.get("SLICERDSPACE_MAX_RAM_GIB", "").strip()
    if cap_env:
        try:
            parsed = int(float(cap_env) * GIB)
            if parsed > 0:
                cap_b = parsed
        except ValueError:
            pass
    try:
        from shared.compute_schedule import available_resources

        _, mem_b = available_resources()
        if int(mem_b) > 0:
            eff_b = int(mem_b)
            if sys.platform.startswith("linux"):
                cg_avail = linux_cgroup_limits().get("memory_available_bytes")
                if cg_avail is not None:
                    eff_b = min(eff_b, int(cg_avail * 0.7))
            return min(eff_b, cap_b) if cap_b is not None else eff_b
    except Exception:
        pass
    return min(8 * GIB, cap_b) if cap_b is not None else 8 * GIB


available_ram_bytes = query_host_available_ram_bytes


def adaptive_cpu_slab_z(
    shape_4d: tuple[int, ...],
    default_step_z: int = 16,
    *,
    min_step_z: int = 2,
    bytes_per_element: int = 32,
    ram_fraction: float = 0.08,
    bytes_per_voxel_factor: float | None = None,
    min_slab: int | None = None,
    max_slab: int | None = None,
) -> int:
    """Bound axial slab thickness ``step_z`` so low-RAM laptops never spike memory."""
    eff_min = int(min_slab if min_slab is not None else min_step_z)
    eff_base = max(eff_min, int(max_slab if max_slab is not None else default_step_z))
    eff_bytes = max(4, int(round(float(bytes_per_voxel_factor) * 4.0)) if bytes_per_voxel_factor is not None else int(bytes_per_element))
    if len(shape_4d) < 3:
        return eff_base
    nx, ny = max(1, int(shape_4d[0])), max(1, int(shape_4d[1]))
    nv = max(1, int(shape_4d[3])) if len(shape_4d) >= 4 else 1
    per_slice = nx * ny * nv * eff_bytes
    avail = query_host_available_ram_bytes()
    budget = max(int(64 * MIB), int(avail * float(ram_fraction)))
    max_slices = max(eff_min, budget // max(1, per_slice))
    return max(eff_min, min(eff_base, int(max_slices)))


def host_total_ram_gib(*, respect_env_cap: bool = True) -> float | None:
    """Return physical host RAM in GiB accurately across Linux, Windows, and macOS (incl. Rosetta 2)."""
    cap_gib: float | None = None
    if respect_env_cap:
        cap_env = os.environ.get("SLICERDSPACE_MAX_RAM_GIB", "").strip()
        if cap_env:
            try:
                parsed = float(cap_env)
                if parsed > 0:
                    cap_gib = parsed
            except ValueError:
                pass

    total_gib: float | None = None
    if sys.platform == "darwin":
        try:
            import subprocess

            sysctl = "/usr/sbin/sysctl" if os.path.exists("/usr/sbin/sysctl") else "sysctl"
            res = subprocess.run([sysctl, "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
            if res.returncode == 0:
                val = int(res.stdout.strip())
                if val > 0:
                    total_gib = float(val) / float(GIB)
        except Exception:
            pass
    if total_gib is None:
        try:
            import psutil

            total = int(psutil.virtual_memory().total)
            if total > 0:
                total_gib = float(total) / float(GIB)
        except Exception:
            pass
    if total_gib is None and os.name == "nt":
        try:
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_uint32),
                    ("dwMemoryLoad", ctypes.c_uint32),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64),
                ]

            stat = _MemoryStatusEx()
            stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)) and stat.ullTotalPhys > 0:
                total_gib = float(stat.ullTotalPhys) / float(GIB)
        except Exception:
            pass
    if total_gib is None and hasattr(os, "sysconf"):
        try:
            pages = int(os.sysconf("SC_PHYS_PAGES"))
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            if pages > 0 and page_size > 0:
                ratio = 1
                if sys.platform == "darwin":
                    try:
                        from shared.platform_info import rosetta_page_ratio

                        ratio = max(1, int(rosetta_page_ratio()))
                    except Exception:
                        ratio = 1
                total_gib = float(pages * page_size * ratio) / float(GIB)
        except (OSError, ValueError):
            pass

    if sys.platform.startswith("linux"):
        cg_limit = linux_cgroup_limits().get("memory_limit_bytes")
        if cg_limit is not None and cg_limit > 0:
            cg_gib = float(cg_limit) / float(GIB)
            total_gib = cg_gib if total_gib is None else min(total_gib, cg_gib)

    if total_gib is None:
        return cap_gib
    return min(total_gib, cap_gib) if cap_gib is not None else total_gib


def is_apple_silicon_host() -> bool:
    """Return True on Apple Silicon macOS (both native arm64 and Rosetta 2)."""
    if sys.platform != "darwin":
        return False
    try:
        from shared.platform_info import host_machine

        return host_machine().lower() in {"arm64", "aarch64"}
    except Exception:
        return False


def enable_cpu_flush_denormal(torch_mod: Any = None) -> bool:
    """Enable flush-to-zero for subnormal floats in PyTorch CPU kernels (prevents x86/Rosetta stalls)."""
    try:
        mod = torch_mod
        if mod is None:
            import torch as mod

        fn = getattr(mod, "set_flush_denormal", None)
        if callable(fn):
            return bool(fn(True))
    except Exception:
        pass
    return False


def adaptive_spawn_workers(
    requested_workers: int,
    *,
    per_worker_ram_gib: float = 2.5,
    reserve_ram_gib: float = 6.0,
    max_workers: int = 8,
    platform_name: str | None = None,
    total_ram_gib: float | None = None,
    available_ram_gib: float | None = None,
) -> int:
    """Cap multiprocessing ``spawn`` worker count on Windows/macOS or low-RAM hosts to prevent OOM.

    High-RAM Linux servers (>48 GiB total and available RAM) preserve ``requested_workers`` unchanged.
    Both total physical RAM (minus ``reserve_ram_gib``) and currently available RAM strictly bound
    the worker count so tight available memory throttles down to 1 worker instead of overcommitting.
    """
    req = max(1, int(requested_workers))
    if req <= 1:
        return 1
    plat = sys.platform if platform_name is None else str(platform_name).lower()
    tot = host_total_ram_gib() if total_ram_gib is None else float(total_ram_gib)
    is_spawn = plat in {"win32", "darwin"}
    if available_ram_gib is None:
        avail_b = query_host_available_ram_bytes()
        avail = float(avail_b) / float(GIB) if avail_b > 0 else None
    else:
        avail = max(0.0, float(available_ram_gib))
    if not is_spawn and tot is not None and tot > 48.0 and (avail is None or avail > 48.0):
        return req
    if tot is None and avail is None:
        return min(req, 4 if is_spawn else req)
    per_worker = max(0.5, float(per_worker_ram_gib))
    limits: list[int] = [req, int(max_workers)]
    if tot is not None:
        tot_budget = max(1, int((tot - float(reserve_ram_gib)) // per_worker))
        if tot <= 12.0:
            tot_budget = min(tot_budget, 2)
        elif tot <= 20.0 and is_spawn:
            tot_budget = min(tot_budget, 3)
        elif tot <= 36.0 and is_spawn:
            tot_budget = min(tot_budget, 4)
        limits.append(tot_budget)
    if avail is not None:
        # ``available_ram_gib`` already reflects currently free headroom (excluding OS/parent process
        # and the 30% safety factor from ``available_resources``), so bound new workers strictly by it.
        avail_budget = max(1, int(max(0.0, avail) // per_worker))
        limits.append(avail_budget)
    return max(1, min(limits))


def _parse_linux_cpulist(spec: str) -> list[int]:
    """Parse Linux sysfs cpulist strings such as ``'0-35,72-107'`` into sorted CPU IDs."""
    cpus: list[int] = []
    for part in spec.strip().split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            lo_s, hi_s = token.split("-", 1)
            if lo_s.isdigit() and hi_s.isdigit():
                lo, hi = int(lo_s), int(hi_s)
                if hi >= lo:
                    cpus.extend(range(lo, hi + 1))
        elif token.isdigit():
            cpus.append(int(token))
    return sorted(set(cpus))


_CGROUP_UNLIMITED_BYTES = 1 << 60


def _cgroup_ancestor_dirs(base: Path, rel_path: str) -> list[Path]:
    """Return ancestor directories from leaf ``base / rel_path`` up to ``base`` (inclusive)."""
    parts = [part for part in rel_path.strip().strip("/").split("/") if part and part not in (".", "..")]
    chain = [base]
    cur = base
    for part in parts:
        cur = cur / part
        chain.append(cur)
    out: list[Path] = []
    for item in reversed(chain):
        if item not in out:
            out.append(item)
    return out


def linux_cgroup_limits(
    *,
    proc_cgroup_path: Path | str = Path("/proc/self/cgroup"),
    cgroup_root: Path | str = Path("/sys/fs/cgroup"),
) -> dict[str, int | None]:
    """Resolve the tightest effective Linux cgroup v1/v2 CPU and memory limits across all ancestors.

    Reads ``proc_cgroup_path`` (default ``/proc/self/cgroup``) to locate the current process's
    nested cgroup v2 unified path and/or cgroup v1 ``cpu``/``cpuset``/``memory`` controller paths,
    then walks every ancestor from leaf up to ``cgroup_root`` taking the minimum effective quota
    and memory headroom. Also checks root cgroup files as a fallback when ``/proc/self/cgroup``
    is absent or container cgroup namespaces bind-mount the leaf at root.
    """
    root = Path(cgroup_root)
    v2_rel_paths: list[str] = []
    v1_mounts: dict[str, list[tuple[str, str]]] = {"cpu": [], "cpuset": [], "memory": []}

    try:
        raw_cgroup = Path(proc_cgroup_path).read_text(encoding="utf-8")
    except OSError:
        raw_cgroup = ""

    for line in raw_cgroup.splitlines():
        parts = line.strip().split(":", 2)
        if len(parts) != 3:
            continue
        hier_id, controllers_raw, cg_path = parts
        cg_path = cg_path.strip()
        if hier_id == "0" and controllers_raw == "":
            if cg_path not in v2_rel_paths:
                v2_rel_paths.append(cg_path)
            continue
        controllers = [c.strip() for c in controllers_raw.split(",") if c.strip()]
        for key in ("cpu", "cpuset", "memory"):
            if key in controllers:
                # Check both combined mount dir (e.g. cpu,cpuacct) and single controller dir (cpu).
                for mount_name in (controllers_raw.strip(), key):
                    if mount_name and (mount_name, cg_path) not in v1_mounts[key]:
                        v1_mounts[key].append((mount_name, cg_path))

    # Always include root mount fallback so host-bind-mounted containers and root-only stubs work.
    if "" not in v2_rel_paths:
        v2_rel_paths.append("")
    for key in ("cpu", "cpuset", "memory"):
        if (key, "") not in v1_mounts[key]:
            v1_mounts[key].append((key, ""))

    cpu_limit: int | None = None
    memory_limit_bytes: int | None = None
    memory_available_bytes: int | None = None

    def _min_opt(cur: int | None, val: int) -> int:
        return val if cur is None else min(cur, val)

    # 1. cgroup v2 unified hierarchy walk
    seen_v2: set[Path] = set()
    for rel in v2_rel_paths:
        for directory in _cgroup_ancestor_dirs(root, rel):
            if directory in seen_v2:
                continue
            seen_v2.add(directory)
            # CPU quota (cpu.max -> "<quota> <period>")
            try:
                tokens = (directory / "cpu.max").read_text(encoding="utf-8").split()
                if len(tokens) >= 2 and tokens[0] != "max":
                    quota_us, period_us = int(tokens[0]), int(tokens[1])
                    if quota_us > 0 and period_us > 0:
                        cpu_limit = _min_opt(cpu_limit, max(1, quota_us // period_us))
            except (OSError, ValueError, ZeroDivisionError):
                pass
            # Effective cpuset
            for cpuset_file in ("cpuset.cpus.effective", "cpuset.cpus"):
                try:
                    cpulist = _parse_linux_cpulist((directory / cpuset_file).read_text(encoding="utf-8"))
                    if cpulist:
                        cpu_limit = _min_opt(cpu_limit, len(cpulist))
                        break
                except OSError:
                    pass
            # Memory limit & remaining headroom
            try:
                raw_max = (directory / "memory.max").read_text(encoding="utf-8").strip()
                if raw_max and raw_max != "max":
                    ceiling = int(raw_max)
                    if 0 < ceiling < _CGROUP_UNLIMITED_BYTES:
                        memory_limit_bytes = _min_opt(memory_limit_bytes, ceiling)
                        try:
                            used = int((directory / "memory.current").read_text(encoding="utf-8").strip())
                            if used >= 0:
                                memory_available_bytes = _min_opt(memory_available_bytes, max(0, ceiling - used))
                        except (OSError, ValueError):
                            pass
            except (OSError, ValueError):
                pass

    # 2. cgroup v1 CPU quota walk
    seen_v1_cpu: set[Path] = set()
    for mount_name, rel in v1_mounts["cpu"]:
        for directory in _cgroup_ancestor_dirs(root / mount_name, rel):
            if directory in seen_v1_cpu:
                continue
            seen_v1_cpu.add(directory)
            try:
                quota_us = int((directory / "cpu.cfs_quota_us").read_text(encoding="utf-8").strip())
                period_us = int((directory / "cpu.cfs_period_us").read_text(encoding="utf-8").strip())
                if quota_us > 0 and period_us > 0:
                    cpu_limit = _min_opt(cpu_limit, max(1, quota_us // period_us))
            except (OSError, ValueError, ZeroDivisionError):
                pass

    # 3. cgroup v1 cpuset walk
    seen_v1_cpuset: set[Path] = set()
    for mount_name, rel in v1_mounts["cpuset"]:
        for directory in _cgroup_ancestor_dirs(root / mount_name, rel):
            if directory in seen_v1_cpuset:
                continue
            seen_v1_cpuset.add(directory)
            for cpuset_file in ("cpuset.effective_cpus", "cpuset.cpus.effective", "cpuset.cpus"):
                try:
                    cpulist = _parse_linux_cpulist((directory / cpuset_file).read_text(encoding="utf-8"))
                    if cpulist:
                        cpu_limit = _min_opt(cpu_limit, len(cpulist))
                        break
                except OSError:
                    pass

    # 4. cgroup v1 memory walk
    seen_v1_mem: set[Path] = set()
    for mount_name, rel in v1_mounts["memory"]:
        for directory in _cgroup_ancestor_dirs(root / mount_name, rel):
            if directory in seen_v1_mem:
                continue
            seen_v1_mem.add(directory)
            try:
                ceiling = int((directory / "memory.limit_in_bytes").read_text(encoding="utf-8").strip())
                if 0 < ceiling < _CGROUP_UNLIMITED_BYTES:
                    memory_limit_bytes = _min_opt(memory_limit_bytes, ceiling)
                    try:
                        used = int((directory / "memory.usage_in_bytes").read_text(encoding="utf-8").strip())
                        if used >= 0:
                            memory_available_bytes = _min_opt(memory_available_bytes, max(0, ceiling - used))
                    except (OSError, ValueError):
                        pass
            except (OSError, ValueError):
                pass

    return {
        "cpu_limit": cpu_limit,
        "memory_limit_bytes": memory_limit_bytes,
        "memory_available_bytes": memory_available_bytes,
    }


def _read_windows_memory_counters() -> dict[str, int | None]:
    """Query Windows physical RAM and system commit charge via Win32 ctypes (zero extra deps)."""
    result: dict[str, int | None] = {
        "total_phys_bytes": None,
        "avail_phys_bytes": None,
        "commit_limit_bytes": None,
        "commit_total_bytes": None,
        "commit_avail_bytes": None,
    }
    if os.name != "nt":
        return result
    try:
        import ctypes

        class _PerformanceInformation(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_uint32),
                ("CommitTotal", ctypes.c_size_t),
                ("CommitLimit", ctypes.c_size_t),
                ("CommitPeak", ctypes.c_size_t),
                ("PhysicalTotal", ctypes.c_size_t),
                ("PhysicalAvailable", ctypes.c_size_t),
                ("SystemCache", ctypes.c_size_t),
                ("KernelTotal", ctypes.c_size_t),
                ("KernelPaged", ctypes.c_size_t),
                ("KernelNonpaged", ctypes.c_size_t),
                ("PageSize", ctypes.c_size_t),
                ("HandleCount", ctypes.c_uint32),
                ("ProcessCount", ctypes.c_uint32),
                ("ThreadCount", ctypes.c_uint32),
            ]

        info = _PerformanceInformation()
        info.cb = ctypes.sizeof(_PerformanceInformation)
        perf_fn = None
        for dll_name, fn_name in (("kernel32", "K32GetPerformanceInfo"), ("psapi", "GetPerformanceInfo")):
            try:
                dll = ctypes.WinDLL(dll_name, use_last_error=True)
                cand = getattr(dll, fn_name, None)
                if cand is not None:
                    perf_fn = cand
                    break
            except OSError:
                continue
        if perf_fn is not None:
            perf_fn.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            perf_fn.restype = ctypes.c_int
            if perf_fn(ctypes.byref(info), info.cb) and info.PageSize > 0:
                page = int(info.PageSize)
                if info.PhysicalTotal > 0:
                    result["total_phys_bytes"] = int(info.PhysicalTotal) * page
                    result["avail_phys_bytes"] = int(info.PhysicalAvailable) * page
                if info.CommitLimit > 0:
                    result["commit_limit_bytes"] = int(info.CommitLimit) * page
                    result["commit_total_bytes"] = int(info.CommitTotal) * page
                    result["commit_avail_bytes"] = max(0, int(info.CommitLimit) - int(info.CommitTotal)) * page
    except Exception:
        pass

    if result["total_phys_bytes"] is None or result["commit_avail_bytes"] is None:
        try:
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_uint32),
                    ("dwMemoryLoad", ctypes.c_uint32),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64),
                ]

            stat = _MemoryStatusEx()
            stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                if stat.ullTotalPhys > 0:
                    if result["total_phys_bytes"] is None:
                        result["total_phys_bytes"] = int(stat.ullTotalPhys)
                    if result["avail_phys_bytes"] is None:
                        result["avail_phys_bytes"] = int(stat.ullAvailPhys)
                if stat.ullTotalPageFile > 0:
                    if result["commit_limit_bytes"] is None:
                        result["commit_limit_bytes"] = int(stat.ullTotalPageFile)
                    if result["commit_avail_bytes"] is None:
                        result["commit_avail_bytes"] = int(stat.ullAvailPageFile)
                    if result["commit_total_bytes"] is None:
                        result["commit_total_bytes"] = max(0, int(stat.ullTotalPageFile) - int(stat.ullAvailPageFile))
        except Exception:
            pass
    return result


def _parse_linux_meminfo(meminfo_path: Path | str = Path("/proc/meminfo")) -> dict[str, int]:
    """Parse Linux ``/proc/meminfo`` fields into bytes."""
    parsed: dict[str, int] = {}
    try:
        text = Path(meminfo_path).read_text(encoding="utf-8")
    except OSError:
        return parsed
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        tokens = rest.strip().split()
        if not tokens:
            continue
        try:
            val = int(tokens[0])
        except ValueError:
            continue
        unit = tokens[1].lower() if len(tokens) > 1 else ""
        scale = 1024 if unit == "kb" else (1024 * 1024 if unit == "mb" else 1)
        parsed[key.strip()] = val * scale
    return parsed


def query_host_memory_snapshot(
    *,
    respect_env_cap: bool = True,
    meminfo_path: Path | str = Path("/proc/meminfo"),
    proc_cgroup_path: Path | str = Path("/proc/self/cgroup"),
    cgroup_root: Path | str = Path("/sys/fs/cgroup"),
) -> dict[str, float | None]:
    """Return a cross-OS snapshot of host physical RAM, available RAM, and commit/swap headroom (GiB).

    Keys:
      - ``total_ram_gib``: installed physical RAM (or cgroup memory ceiling on Linux when tighter)
      - ``available_ram_gib``: currently available physical RAM (Rosetta-corrected on macOS)
      - ``commit_limit_gib``, ``commit_total_gib``, ``commit_available_gib``: Windows commit charge
      - ``swap_total_gib``, ``swap_free_gib``: swap capacity/headroom on Linux/macOS
      - ``effective_available_ram_gib``: ``min(available_ram_gib, commit_available_gib)`` when both
        are known, otherwise whichever is observable
    """
    cap_gib: float | None = None
    if respect_env_cap:
        cap_env = os.environ.get("SLICERDSPACE_MAX_RAM_GIB", "").strip()
        if cap_env:
            try:
                parsed_cap = float(cap_env)
                if parsed_cap > 0:
                    cap_gib = parsed_cap
            except ValueError:
                pass

    total_gib: float | None = None
    avail_gib: float | None = None
    commit_limit_gib: float | None = None
    commit_total_gib: float | None = None
    commit_avail_gib: float | None = None
    swap_total_gib: float | None = None
    swap_free_gib: float | None = None

    if sys.platform == "win32":
        win = _read_windows_memory_counters()
        if win["total_phys_bytes"] is not None:
            total_gib = float(win["total_phys_bytes"]) / float(GIB)
        if win["avail_phys_bytes"] is not None:
            avail_gib = float(win["avail_phys_bytes"]) / float(GIB)
        if win["commit_limit_bytes"] is not None:
            commit_limit_gib = float(win["commit_limit_bytes"]) / float(GIB)
        if win["commit_total_bytes"] is not None:
            commit_total_gib = float(win["commit_total_bytes"]) / float(GIB)
        if win["commit_avail_bytes"] is not None:
            commit_avail_gib = float(win["commit_avail_bytes"]) / float(GIB)
    elif sys.platform.startswith("linux"):
        info = _parse_linux_meminfo(meminfo_path)
        if info.get("MemTotal", 0) > 0:
            total_gib = float(info["MemTotal"]) / float(GIB)
        if "MemAvailable" in info:
            avail_gib = float(info["MemAvailable"]) / float(GIB)
        elif "MemFree" in info:
            approx = info["MemFree"] + info.get("Buffers", 0) + info.get("Cached", 0)
            avail_gib = float(approx) / float(GIB)
        if "SwapTotal" in info:
            swap_total_gib = float(info["SwapTotal"]) / float(GIB)
        if "SwapFree" in info:
            swap_free_gib = float(info["SwapFree"]) / float(GIB)
        cg = linux_cgroup_limits(proc_cgroup_path=proc_cgroup_path, cgroup_root=cgroup_root)
        if cg["memory_limit_bytes"] is not None:
            cg_tot = float(cg["memory_limit_bytes"]) / float(GIB)
            total_gib = cg_tot if total_gib is None else min(total_gib, cg_tot)
        if cg["memory_available_bytes"] is not None:
            cg_avail = float(cg["memory_available_bytes"]) / float(GIB)
            avail_gib = cg_avail if avail_gib is None else min(avail_gib, cg_avail)
    elif sys.platform == "darwin":
        total_gib = host_total_ram_gib(respect_env_cap=False)
        try:
            from shared.platform_info import available_host_ram_bytes

            avail_b = available_host_ram_bytes()
            if avail_b is not None and avail_b >= 0:
                avail_gib = float(avail_b) / float(GIB)
        except Exception:
            pass
        try:
            import psutil

            sw = psutil.swap_memory()
            swap_total_gib = float(sw.total) / float(GIB)
            swap_free_gib = float(sw.free) / float(GIB)
        except Exception:
            pass

    if total_gib is None or avail_gib is None:
        try:
            import psutil

            vm = psutil.virtual_memory()
            if total_gib is None and int(vm.total) > 0:
                total_gib = float(vm.total) / float(GIB)
            if avail_gib is None and int(vm.available) >= 0:
                avail_gib = float(vm.available) / float(GIB)
        except Exception:
            pass
    if total_gib is None:
        total_gib = host_total_ram_gib(respect_env_cap=False)

    if cap_gib is not None:
        total_gib = cap_gib if total_gib is None else min(total_gib, cap_gib)
        if avail_gib is not None:
            avail_gib = min(avail_gib, cap_gib)
        if commit_avail_gib is not None:
            commit_avail_gib = min(commit_avail_gib, cap_gib)

    eff_candidates = [v for v in (avail_gib, commit_avail_gib) if v is not None]
    effective_avail_gib = min(eff_candidates) if eff_candidates else None

    return {
        "total_ram_gib": total_gib,
        "available_ram_gib": avail_gib,
        "commit_limit_gib": commit_limit_gib,
        "commit_total_gib": commit_total_gib,
        "commit_available_gib": commit_avail_gib,
        "swap_total_gib": swap_total_gib,
        "swap_free_gib": swap_free_gib,
        "effective_available_ram_gib": effective_avail_gib,
    }


def linux_single_socket_physical_cores(logical_cpus: int | None = None) -> int:
    """Return physical core count of NUMA node 0 on Linux (fallback: ``logical_cpus // 2``).

    Keeping PyTorch CPU conv2d/conv3d OpenMP threads within a single NUMA socket avoids
    cross-socket UPI/QPI synchronization overhead on dual-socket Xeon/EPYC servers while
    still scaling past the 8-16 thread consumer desktop cap.
    """
    fallback = max(1, int(logical_cpus or os.cpu_count() or 4))
    node0_path = Path("/sys/devices/system/node/node0/cpulist")
    try:
        if node0_path.is_file():
            node0_cpus = _parse_linux_cpulist(node0_path.read_text(encoding="utf-8"))
            if node0_cpus:
                phys_pairs: set[tuple[str, str]] = set()
                for cpu_id in node0_cpus:
                    topo = Path(f"/sys/devices/system/cpu/cpu{cpu_id}/topology")
                    core_p = topo / "core_id"
                    pkg_p = topo / "physical_package_id"
                    if core_p.is_file():
                        core_id = core_p.read_text(encoding="utf-8").strip()
                        pkg_id = pkg_p.read_text(encoding="utf-8").strip() if pkg_p.is_file() else "0"
                        phys_pairs.add((pkg_id, core_id))
                if phys_pairs:
                    return max(1, len(phys_pairs))
                return max(1, len(node0_cpus) // 2 if len(node0_cpus) >= 16 else len(node0_cpus))
    except OSError:
        pass
    return max(1, fallback // 2 if fallback >= 32 else fallback)


def optimal_cpu_torch_threads(
    reserved_cpus: int | None = None,
    *,
    max_conv_threads: int = 8,
    server_max_threads: int | None = None,
    platform_name: str | None = None,
    is_apple_silicon: bool | None = None,
) -> int:
    """Choose PyTorch CPU intra-op thread count avoiding OpenMP contention on small 2D slices and E-cores.

    When ``server_max_threads`` is supplied on high-core, high-RAM (>48 GiB) Linux servers without an
    explicit stage reservation, scales up to single-socket physical cores (capped by ``server_max_threads``)
    so dual-socket Xeon/EPYC hosts are not artificially throttled to desktop thread counts.
    """
    eff_cap = max(int(max_conv_threads), int(server_max_threads)) if server_max_threads is not None else int(max_conv_threads)
    env_stage = os.environ.get("SLICERDSPACE_STAGE_CPUS", "").strip()
    if reserved_cpus is None and env_stage.isdigit() and int(env_stage) > 0:
        reserved_cpus = int(env_stage)
    if reserved_cpus is not None and int(reserved_cpus) > 0:
        return max(1, min(int(reserved_cpus), eff_cap))
    mem_b = 0
    try:
        from shared.compute_schedule import available_resources

        cpus, mem_b = available_resources()
        cpus = max(1, int(cpus))
        mem_b = int(mem_b)
    except Exception:
        cpus = max(1, int(os.cpu_count() or 4))
    plat = sys.platform if platform_name is None else str(platform_name).lower()
    apple = is_apple_silicon_host() if is_apple_silicon is None else bool(is_apple_silicon)
    if plat == "darwin" and apple and cpus >= 6:
        # Leave efficiency cores headroom so OpenMP barriers do not stall behind E-cores.
        p_cap = max(4, cpus - 2) if cpus <= 10 else max(6, (cpus * 3) // 4)
        return max(1, min(cpus, p_cap, int(max_conv_threads)))
    if server_max_threads is not None and plat.startswith("linux") and cpus >= 32:
        tot_gib = (float(mem_b) / float(GIB)) if mem_b > 0 else host_total_ram_gib()
        if tot_gib is not None and tot_gib > 48.0:
            socket_phys = linux_single_socket_physical_cores(cpus)
            target = min(cpus, socket_phys, int(server_max_threads))
            return max(int(max_conv_threads), target)
    return max(1, min(cpus, int(max_conv_threads)))


def zero_subnormal_module_weights(module: Any, *, min_normal: float = 1.17549435e-38) -> int:
    """Zero IEEE-754 subnormal float32 parameters/buffers in-place (bit-identical output, no CPU traps).

    Trained checkpoints exported from CUDA (where hardware flushes subnormals for free) can carry
    tens of millions of subnormal float32 weights (e.g. ~99% of DDSeg, ~31% of DDParcel). On CPUs
    where ``torch.set_flush_denormal(True)`` is unsupported (such as ARM64) or OpenMP worker threads
    were already spawned before FTZ/DAZ was enabled, zeroing subnormal weights at load time eliminates
    the 20x-35x x86 microcode assist penalty with zero numerical drift.
    """
    try:
        import torch
    except Exception:
        return 0
    if module is None:
        return 0
    zeroed = 0
    seen: set[int] = set()
    threshold = float(min_normal)
    with torch.no_grad():
        for iter_name in ("parameters", "buffers"):
            fn = getattr(module, iter_name, None)
            if not callable(fn):
                continue
            try:
                tensors = list(fn())
            except Exception:
                continue
            for tensor in tensors:
                if not isinstance(tensor, torch.Tensor):
                    continue
                tid = id(tensor)
                if tid in seen or tensor.dtype != torch.float32:
                    continue
                seen.add(tid)
                data = tensor.detach()
                abs_data = data.abs()
                mask = (abs_data > 0.0) & (abs_data < threshold)
                if bool(mask.any().item()):
                    zeroed += int(mask.sum().item())
                    data.masked_fill_(mask, 0.0)
    return zeroed
