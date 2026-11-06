"""Machine architecture of Slicer's Python and of the host (T2 design B1).

Stable Slicer on macOS is x86_64 only. On Apple Silicon it runs under Rosetta 2, so
``platform.machine()`` reports ``x86_64`` inside it while the host is ``arm64``.
Anything installed into Slicer's Python must match ``slicer_machine()``. External
runtimes (conda environments, MRtrix3, FSL) run outside Slicer and choose their build
by ``host_machine()``.
"""

from __future__ import annotations

import platform
import subprocess


def slicer_machine() -> str:
    """Architecture of the running (Slicer) Python process."""
    return platform.machine()


def _sysctl_is_one(name: str) -> bool:
    try:
        result = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and result.stdout.strip() == "1"


def host_machine() -> str:
    """Architecture of the machine; ``arm64`` on Apple Silicon even under Rosetta."""
    machine = platform.machine()
    if platform.system() == "Darwin" and (_sysctl_is_one("sysctl.proc_translated")
                                          or _sysctl_is_one("hw.optional.arm64")):
        return "arm64"
    return machine
