"""Machine architecture of Slicer's Python and of the host (T2 design B1).

Stable Slicer on macOS is x86_64 only. On Apple Silicon it runs under Rosetta 2, so
``platform.machine()`` reports ``x86_64`` inside it while the host is ``arm64``.
Anything installed into Slicer's Python must match ``slicer_machine()``. External
runtimes (conda environments, MRtrix3, FSL) run outside Slicer and choose their build
by ``host_machine()``.
"""

from __future__ import annotations

import functools
import os
import platform
import subprocess

_SYSCTL = "/usr/sbin/sysctl"


def slicer_machine() -> str:
    """Architecture of the running (Slicer) Python process."""
    return platform.machine()


def _sysctl_is_one(name: str) -> bool:
    # R-N4: the fixed system path first, so a PATH entry named sysctl cannot answer; the bare name if it is absent.
    program = _SYSCTL if os.path.exists(_SYSCTL) else "sysctl"
    try:
        result = subprocess.run([program, "-n", name], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and result.stdout.strip() == "1"


@functools.lru_cache(maxsize=1)
def _apple_silicon_host() -> bool:
    """R-N4: the answer cannot change while the process runs, so sysctl is called once."""
    return _sysctl_is_one("sysctl.proc_translated") or _sysctl_is_one("hw.optional.arm64")


def host_machine() -> str:
    """Architecture of the machine; ``arm64`` on Apple Silicon even under Rosetta."""
    machine = platform.machine()
    if platform.system() == "Darwin" and _apple_silicon_host():
        return "arm64"
    return machine


_ARCH = "/usr/bin/arch"
_NATIVE_ARM64_PAGE_BYTES = 16384


def _native_page_bytes() -> int:
    """Native host page size. Under Rosetta, ``arch -arm64 sysctl`` reports it; else 16 KiB on Apple Silicon."""
    program = _SYSCTL if os.path.exists(_SYSCTL) else "sysctl"
    try:
        result = subprocess.run([_ARCH, "-arm64", program, "-n", "hw.pagesize"],
                                capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            value = int(result.stdout.strip())
            if value > 0:
                return value
    except (OSError, subprocess.TimeoutExpired, ValueError):
        pass
    return _NATIVE_ARM64_PAGE_BYTES if _sysctl_is_one("hw.optional.arm64") else 0


@functools.lru_cache(maxsize=1)
def rosetta_page_ratio() -> int:
    """M1/F2: factor by which psutil under-reports page-based memory under Rosetta; 1 when it does not.

    Under Rosetta on Apple Silicon the process page size is 4096, but the Mach VM page counts that
    psutil reads are in native 16384-byte pages, so available/free are too low by native/process.
    """
    try:
        if platform.system() != "Darwin" or not _sysctl_is_one("sysctl.proc_translated"):
            return 1
        process_page = int(os.sysconf("SC_PAGE_SIZE"))
        native_page = _native_page_bytes()
        if process_page > 0 and native_page > process_page and native_page % process_page == 0:
            return native_page // process_page
    except Exception:
        pass
    return 1


def available_host_ram_bytes():
    """Available host RAM in bytes, corrected for the Rosetta page-size mismatch; ``None`` without psutil.

    Never raises. On any failure of the correction the uncorrected psutil value is returned.
    """
    try:
        import psutil
        memory = psutil.virtual_memory()
        available = int(memory.available)
    except Exception:
        return None
    try:
        ratio = rosetta_page_ratio()
        if ratio > 1:
            available *= ratio
            total = int(getattr(memory, "total", 0) or 0)
            if total > 0:
                available = min(available, total)
    except Exception:
        pass
    return available
