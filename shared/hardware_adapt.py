"""Hardware-adaptive chunking, VRAM budgeting, and OOM fallback helpers.

Designed so SlicerDSpace runs smoothly across both modest user workstations/laptops
(4-8 CPU cores, 8-16 GiB RAM, CPU-only or 4-8 GiB consumer GPUs) and high-end
multi-GPU servers without changing numerical results.
"""

from __future__ import annotations

import os
from typing import Any, Callable


GIB = 1024 ** 3
MIB = 1024 ** 2


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


def is_cuda_oom(exc: BaseException) -> bool:
    """Return True when ``exc`` represents a CUDA out-of-memory condition."""
    msg = str(exc).lower()
    if "out of memory" in msg and ("cuda" in msg or "cublas" in msg or "cudnn" in msg or "alloc" in msg):
        return True
    cls_name = type(exc).__name__.lower()
    return "outofmemory" in cls_name


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


def adaptive_cuda_scaled_batch(
    per_item_bytes: int,
    default_batch: int,
    *,
    min_batch: int = 64,
    max_batch: int = 131072,
    safety_fraction: float = 0.30,
    device: Any = None,
) -> int:
    """Scale batch size down on low-VRAM GPUs while keeping ``default_batch`` when VRAM suffices."""
    base = max(int(min_batch), min(int(max_batch), int(default_batch)))
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


def optimal_cpu_torch_threads(reserved_cpus: int | None = None, *, max_conv_threads: int = 8) -> int:
    """Choose PyTorch CPU intra-op thread count avoiding OpenMP contention on small 2D slices."""
    env_stage = os.environ.get("SLICERDSPACE_STAGE_CPUS", "").strip()
    if reserved_cpus is None and env_stage.isdigit() and int(env_stage) > 0:
        reserved_cpus = int(env_stage)
    if reserved_cpus is not None and int(reserved_cpus) > 0:
        return max(1, min(int(reserved_cpus), int(max_conv_threads)))
    try:
        from shared.compute_schedule import available_resources

        cpus, _ = available_resources()
        return max(1, min(int(cpus), int(max_conv_threads)))
    except Exception:
        return max(1, min(os.cpu_count() or 4, int(max_conv_threads)))
