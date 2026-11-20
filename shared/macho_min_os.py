"""Pure-Python reader for the minimum macOS version a Mach-O binary asks for (G1).

Reads only the Mach-O header and the load commands, never the whole file. It does not need ``otool``,
so the setup code can compare a bundled runtime with the running macOS before the first ``dlopen``.

Every function returns ``None`` when the answer is unknown and never raises on a bad or odd file.
A version is a tuple of ints ``(major, minor, patch)``.
"""

from __future__ import annotations

import os
import platform
import stat
import struct
from typing import Dict, Optional, Tuple

Version = Tuple[int, int, int]

MH_MAGIC_64 = 0xFEEDFACF
FAT_MAGIC = 0xCAFEBABE
FAT_MAGIC_64 = 0xCAFEBABF
CPU_TYPE_X86_64 = 0x01000007
CPU_TYPE_ARM64 = 0x0100000C
LC_VERSION_MIN_MACOSX = 0x24
LC_BUILD_VERSION = 0x32
PLATFORM_MACOS = 1

_THIN_HEADER_SIZE = 32
_MAX_FAT_ARCHS = 16  # 0xCAFEBABE is also the magic of a Java class file; a large "count" is not a fat file
_MAX_LOAD_COMMAND_BYTES = 4 * 1024 * 1024
_MAX_FILES_PER_DIR = 50000
_CPU_BY_ARCH = {"arm64": CPU_TYPE_ARM64, "aarch64": CPU_TYPE_ARM64, "x86_64": CPU_TYPE_X86_64, "amd64": CPU_TYPE_X86_64}


def _unpack_version(value: int) -> Version:
    """Decode the packed ``xxxx.yy.zz`` form."""
    return (value >> 16) & 0xFFFF, (value >> 8) & 0xFF, value & 0xFF


def format_version(version: Optional[Version]) -> str:
    """``(15, 0, 0)`` -> ``"15.0"``; ``(10, 13, 2)`` -> ``"10.13.2"``; ``None`` -> ``""``."""
    if not version:
        return ""
    major, minor, patch = (tuple(version) + (0, 0, 0))[:3]
    return "%d.%d.%d" % (major, minor, patch) if patch else "%d.%d" % (major, minor)


def parse_version_text(text) -> Optional[Version]:
    """``"15.0"`` -> ``(15, 0, 0)``; anything that is not dotted ints -> ``None``."""
    try:
        parts = [int(part) for part in str(text).strip().split(".")[:3]]
    except (TypeError, ValueError):
        return None
    if not parts:
        return None
    return tuple(parts + [0] * (3 - len(parts)))  # type: ignore[return-value]


def current_macos_version() -> Optional[Version]:
    """The running macOS version, or ``None`` when this is not macOS (or the version is not readable)."""
    if platform.system() != "Darwin":
        return None
    return parse_version_text(platform.mac_ver()[0]) if platform.mac_ver()[0] else None


def _host_arch() -> str:
    try:
        from shared.platform_info import host_machine
        return host_machine().lower()
    except Exception:
        return platform.machine().lower()


def _thin_min_os(stream, offset: int, file_size: int) -> Optional[Version]:
    """Minimum macOS of the thin 64-bit Mach-O that starts at ``offset``; reads the header and the commands only."""
    stream.seek(offset)
    header = stream.read(_THIN_HEADER_SIZE)
    if len(header) < _THIN_HEADER_SIZE or struct.unpack_from("<I", header, 0)[0] != MH_MAGIC_64:
        return None
    ncmds, sizeofcmds = struct.unpack_from("<II", header, 16)
    if sizeofcmds > _MAX_LOAD_COMMAND_BYTES or offset + _THIN_HEADER_SIZE + sizeofcmds > file_size:
        return None
    commands = stream.read(sizeofcmds)
    if len(commands) < sizeofcmds:
        return None
    position = 0
    found = None
    for _ in range(ncmds):
        if position + 8 > len(commands):
            break
        command, size = struct.unpack_from("<II", commands, position)
        if size < 8 or position + size > len(commands):
            break
        if command == LC_BUILD_VERSION and size >= 16:
            platform_id, minos = struct.unpack_from("<II", commands, position + 8)
            if platform_id == PLATFORM_MACOS:
                found = _unpack_version(minos)
        elif command == LC_VERSION_MIN_MACOSX and size >= 12 and found is None:
            found = _unpack_version(struct.unpack_from("<I", commands, position + 8)[0])
        position += size
    return found


def _fat_slices(stream, file_size: int):
    """Return ``[(cputype, offset)]`` of a fat file, or ``None`` when the file is not a plausible fat file."""
    head = stream.read(8)
    if len(head) < 8:
        return None
    magic, count = struct.unpack(">II", head)
    if magic not in (FAT_MAGIC, FAT_MAGIC_64) or not 0 < count <= _MAX_FAT_ARCHS:
        return None
    wide = magic == FAT_MAGIC_64
    record_size = 32 if wide else 20
    table = stream.read(record_size * count)
    if len(table) < record_size * count:
        return None
    slices = []
    for index in range(count):
        if wide:
            cputype, _sub, offset, size, _align, _res = struct.unpack_from(">IIQQII", table, index * record_size)
        else:
            cputype, _sub, offset, size, _align = struct.unpack_from(">IIIII", table, index * record_size)
        if offset + _THIN_HEADER_SIZE > file_size or offset + size > file_size:
            return None
        slices.append((cputype, offset))
    return slices


def macho_min_macos(path, arch: Optional[str] = None) -> Optional[Version]:
    """Minimum macOS version of one Mach-O file, or ``None`` (not Mach-O, no version command, unreadable).

    For a fat file the slice for ``arch`` is read (default: the host architecture, ``arm64`` on Apple Silicon
    also under Rosetta). When no slice matches, the highest minimum of all slices is returned.
    """
    wanted = _CPU_BY_ARCH.get((arch or _host_arch()).lower())
    try:
        with open(path, "rb") as stream:
            file_size = os.fstat(stream.fileno()).st_size
            if file_size < 8:
                return None
            magic = struct.unpack("<I", stream.read(4))[0]
            if magic == MH_MAGIC_64:
                return _thin_min_os(stream, 0, file_size)
            stream.seek(0)
            slices = _fat_slices(stream, file_size)
            if not slices:
                return None
            chosen = [offset for cputype, offset in slices if cputype == wanted] if wanted else []
            versions = [_thin_min_os(stream, offset, file_size) for offset in (chosen or [o for _c, o in slices])]
            versions = [version for version in versions if version is not None]
            return max(versions) if versions else None
    except (OSError, ValueError, struct.error, OverflowError):
        return None


_FILE_CACHE: Dict[tuple, Optional[Version]] = {}


def _file_min_macos(path: str, stat_result, arch: str) -> Optional[Version]:
    key = (path, stat_result.st_mtime_ns, stat_result.st_size, arch)
    if key not in _FILE_CACHE:
        _FILE_CACHE[key] = macho_min_macos(path, arch)
    return _FILE_CACHE[key]


def min_macos_for_dir(directory, arch: Optional[str] = None) -> Optional[Version]:
    """Highest minimum macOS over the Mach-O files in ``directory/bin`` and ``directory/lib``, or ``None``.

    Scripts, other non-Mach-O files and symlinks are skipped (so a symlink loop cannot hang the walk).
    Each file is read once per (path, modification time, size, arch); a repeat call only stats the files.
    It never raises.
    """
    try:
        root = os.path.abspath(os.fspath(directory))
        arch = (arch or _host_arch()).lower()
        best = None
        seen = 0
        capped = False
        for sub in ("bin", "lib"):
            top = os.path.join(root, sub)
            if capped:
                break
            if os.path.islink(top):
                continue
            for current, dirs, names in os.walk(top, followlinks=False):
                dirs[:] = [name for name in dirs if not os.path.islink(os.path.join(current, name))]
                for name in names:
                    seen += 1
                    if seen > _MAX_FILES_PER_DIR:
                        capped = True  # O7: the cap ends the whole walk, not only this directory
                        break
                    path = os.path.join(current, name)
                    try:
                        info = os.lstat(path)
                    except OSError:
                        continue
                    if not stat.S_ISREG(info.st_mode):
                        continue
                    version = _file_min_macos(path, info, arch)
                    if version is not None and (best is None or version > best):
                        best = version
                if capped:
                    break
        return best
    except Exception:
        return None
