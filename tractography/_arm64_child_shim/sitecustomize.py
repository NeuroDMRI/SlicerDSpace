"""Site hook for Rosetta-translated Python running MRtrix scripts (dwi2response / 5ttgen).

When Slicer's x86_64 Python executes a Python-based MRtrix script on Apple Silicon,
universal Mach-O child binaries (dwiextract, amp2response, mrmath, maskfilter, etc.)
would otherwise inherit x86_64 Rosetta translation. Wrapping subprocess.Popen here
launches those universal child executables natively with ``/usr/bin/arch -arm64``.
"""

import os
import shutil
import struct
import subprocess
import sys

_ARCH = "/usr/bin/arch"
_DISABLE_ENV = "SLICERDSPACE_DISABLE_ARM64_EXEC"
_ARM64_CPUTYPE = 0x0100000C
_FAT_RECORD = {0xCAFEBABE: 20, 0xCAFEBABF: 32}


def _arm64_disabled(env):
    for source in (os.environ, env or {}):
        if str(source.get(_DISABLE_ENV, "")).strip().lower() in {"1", "true", "yes", "on"}:
            return True
    return False


def _has_arm64_slice(path):
    try:
        with open(path, "rb") as handle:
            head = handle.read(8)
            if len(head) < 8:
                return False
            magic, count = struct.unpack(">II", head)
            record = _FAT_RECORD.get(magic)
            if record is None or not 0 < count <= 32:
                return False
            table = handle.read(record * count)
        return any(
            struct.unpack(">I", table[i * record : i * record + 4])[0] == _ARM64_CPUTYPE
            for i in range(count)
            if len(table) >= i * record + 4
        )
    except OSError:
        return False


if sys.platform == "darwin" and os.path.exists(_ARCH) and not _arm64_disabled(None):
    _orig_popen = subprocess.Popen

    class _Arm64UniversalPopen(_orig_popen):
        def __init__(self, args, *a, **kw):
            try:
                env = kw.get("env")
                if (
                    isinstance(args, (list, tuple))
                    and args
                    and not kw.get("shell", False)
                    and str(args[0]) != _ARCH
                    and not _arm64_disabled(env)
                ):
                    prog = str(args[0])
                    if os.sep not in prog:
                        prog = shutil.which(prog, path=(env or os.environ).get("PATH")) or prog
                    base = os.path.basename(prog).lower()
                    if (
                        os.path.isfile(prog)
                        and not base.startswith("python")
                        and not base.startswith("pythonslicer")
                        and _has_arm64_slice(prog)
                    ):
                        src_env = env if env is not None else os.environ
                        forwarded = [
                            part
                            for key, value in src_env.items()
                            if str(key).startswith("DYLD_") and value
                            for part in ("-e", f"{key}={value}")
                        ]
                        args = [_ARCH, "-arm64", *forwarded, os.path.abspath(prog), *args[1:]]
            except Exception:
                pass
            super().__init__(args, *a, **kw)

    subprocess.Popen = _Arm64UniversalPopen
