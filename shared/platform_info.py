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
