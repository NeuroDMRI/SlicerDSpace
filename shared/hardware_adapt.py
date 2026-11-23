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
        cap_env = os.environ.get("SLICERDSPACE_MAX_VRAM_GIB", "").strip()
        if cap_env:
            try:
                cap_b = int(float(cap_env) * GIB)
                if cap_b > 0:
                    used_b = max(0, int(torch.cuda.memory_allocated(idx)))
                    total_b = min(int(total_b), cap_b)
                    free_b = min(int(free_b), max(int(64 * MIB), cap_b - used_b))
            except ValueError:
                pass
        return int(free_b), int(total_b)
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
        cap_env = os.environ.get("SLICERDSPACE_MAX_VRAM_GIB", "").strip()
        if cap_env:
            try:
                cap_b = int(float(cap_env) * GIB)
                if cap_b > 0:
                    total_b = min(total_b, cap_b)
            except ValueError:
                pass
        free_b = max(int(64 * MIB), total_b - used_b)
        return int(free_b), int(total_b)
    except Exception:
        return None


def query_gpu_free_total_bytes(device: Any = None) -> tuple[int, int] | None:
    """Return ``(free_bytes, total_bytes)`` for CUDA or MPS ``device``."""
    try:
        import torch

        if device is not None:
            dev = device if isinstance(device, torch.device) else torch.device(str(device))
            if dev.type == "mps":
                return query_mps_free_total_bytes()
            if dev.type == "cuda":
                return query_cuda_free_total_bytes(dev)
            return None
    except Exception:
        pass
    info = query_cuda_free_total_bytes(device)
    if info is not None:
        return info
    return query_mps_free_total_bytes()


def is_cuda_oom(exc: BaseException) -> bool:
    """Return True when ``exc`` represents a CUDA or Apple MPS out-of-memory condition."""
    msg = str(exc).lower()
    if ("out of memory" in msg or "alloc_failed" in msg) and (
        "cuda" in msg or "cublas" in msg or "cudnn" in msg or "alloc" in msg or "mps" in msg
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
) -> int:
    """Scale batch size down on low-VRAM GPUs while keeping ``default_batch`` when VRAM suffices."""
    base = max(int(min_batch), min(int(max_batch), int(default_batch)))
    if os.name == "nt" and windows_tdr_cap is not None:
        base = max(int(min_batch), min(base, int(windows_tdr_cap)))
    info = query_cuda_free_total_bytes(device)
    if info is None:
        return base
    free_b, _ = info
    budget = max(int(32 * MIB), int(free_b * float(safety_fraction)))
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
            return min(int(mem_b), cap_b) if cap_b is not None else int(mem_b)
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

    High-RAM Linux servers (>48 GiB total RAM) preserve ``requested_workers`` unchanged.
    """
    req = max(1, int(requested_workers))
    if req <= 1:
        return 1
    plat = sys.platform if platform_name is None else str(platform_name).lower()
    tot = host_total_ram_gib() if total_ram_gib is None else float(total_ram_gib)
    is_spawn = plat in {"win32", "darwin"}
    if not is_spawn and tot is not None and tot > 48.0:
        return req
    if available_ram_gib is None:
        avail_b = query_host_available_ram_bytes()
        avail = float(avail_b) / float(GIB) if avail_b > 0 else None
    else:
        avail = float(available_ram_gib)
    # On spawn platforms (Windows/macOS), use conservative usable budget bounded by both total and available RAM.
    if tot is not None and avail is not None:
        usable = min(tot, max(avail + 2.0, tot * 0.65))
    elif tot is not None:
        usable = tot
    elif avail is not None:
        usable = avail
    else:
        return min(req, 4 if is_spawn else req)
    if not is_spawn and usable > 48.0:
        return req
    budget_workers = max(1, int((usable - float(reserve_ram_gib)) // max(0.5, float(per_worker_ram_gib))))
    if tot is not None and tot <= 12.0:
        budget_workers = min(budget_workers, 2)
    elif tot is not None and tot <= 20.0 and is_spawn:
        budget_workers = min(budget_workers, 3)
    elif tot is not None and tot <= 36.0 and is_spawn:
        budget_workers = min(budget_workers, 4)
    return max(1, min(req, int(max_workers), budget_workers))


def optimal_cpu_torch_threads(
    reserved_cpus: int | None = None,
    *,
    max_conv_threads: int = 8,
    platform_name: str | None = None,
    is_apple_silicon: bool | None = None,
) -> int:
    """Choose PyTorch CPU intra-op thread count avoiding OpenMP contention on small 2D slices and E-cores."""
    env_stage = os.environ.get("SLICERDSPACE_STAGE_CPUS", "").strip()
    if reserved_cpus is None and env_stage.isdigit() and int(env_stage) > 0:
        reserved_cpus = int(env_stage)
    if reserved_cpus is not None and int(reserved_cpus) > 0:
        return max(1, min(int(reserved_cpus), int(max_conv_threads)))
    try:
        from shared.compute_schedule import available_resources

        cpus, _ = available_resources()
        cpus = max(1, int(cpus))
    except Exception:
        cpus = max(1, int(os.cpu_count() or 4))
    plat = sys.platform if platform_name is None else str(platform_name).lower()
    apple = is_apple_silicon_host() if is_apple_silicon is None else bool(is_apple_silicon)
    if plat == "darwin" and apple and cpus >= 6:
        # Leave efficiency cores headroom so OpenMP barriers do not stall behind E-cores.
        p_cap = max(4, cpus - 2) if cpus <= 10 else max(6, (cpus * 3) // 4)
        return max(1, min(cpus, p_cap, int(max_conv_threads)))
    return max(1, min(cpus, int(max_conv_threads)))
