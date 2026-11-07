"""Locate a native CPU FSL and MRtrix runtime without executing discovery candidates."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Mapping

from shared.external_process import FSL_OUTPUT_TYPE, external_python_environment, fsl_root_from_bin_dir


MRTRIX_COMMANDS = ("mrconvert", "mrinfo", "dwi2mask", "maskfilter")
REVERSE_MRTRIX_COMMANDS = ("dwiextract", "mrcat", "mrtransform", "mrcalc")
FSL_COMMANDS = ("eddy",)
REVERSE_FSL_COMMANDS = ("topup", "applytopup")
TOPUP_CONFIG = Path("etc", "flirtsch", "b02b0.cnf")
MACH_O_MAGICS = (
    b"\xfe\xed\xfa\xce", b"\xce\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"\xcf\xfa\xed\xfe",
    b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca", b"\xca\xfe\xba\xbf", b"\xbf\xba\xfe\xca",
)


def _native_binary(path: Path, system: str) -> str | None:
    """Reject shell wrappers, WSL bridges, and executable formats for other hosts."""
    try:
        if not path.is_file() or not os.access(path, os.X_OK):
            return None
        resolved = path.resolve(strict=True)
        if any(candidate.with_name(candidate.name + ".slicerdsurfer-owner").is_file()
               for candidate in (path, resolved)):
            return None
        with resolved.open("rb") as stream:
            magic = stream.read(4)
            if system == "windows":
                if not magic.startswith(b"MZ"):
                    return None
                stream.seek(60)
                offset = int.from_bytes(stream.read(4), "little")
                stream.seek(offset)
                matched = offset >= 64 and stream.read(4) == b"PE\x00\x00"
            elif system == "darwin":
                matched = magic in MACH_O_MAGICS
            else:
                matched = system == "linux" and magic == b"\x7fELF"
    except (OSError, RuntimeError, ValueError):
        return None
    return str(resolved) if matched else None


def _candidate_dirs(configured: Path | str | None, tool: str, system: str) -> tuple[Path, ...]:
    if configured is not None:
        return (Path(configured).expanduser().resolve(),)

    from shared.managed_runtimes import MRTRIX_TARBALL, PROFILES, runtime_root

    if tool == "fsl":
        settings = ("SLICERDSPACE_FSL_BIN_DIR", "FSL_BIN_DIR")
        homes = ("FSLDIR",)
        managed = (PROFILES["fsl_cpu"]["directory"], PROFILES["fsl_cuda"]["directory"])
    else:
        settings = ("SLICERDSPACE_MRTRIX_BIN_DIR", "MRTRIX_BIN_DIR")
        homes = ("SLICERDSPACE_MRTRIX_HOME", "MRTRIX_HOME", "MRTRIX3_HOME")
        managed = (MRTRIX_TARBALL["directory"] if system == "darwin" else PROFILES["mrtrix"]["directory"],)

    candidates = [Path(os.environ[key]).expanduser() for key in settings if os.environ.get(key)]
    candidates.extend(Path(os.environ[key]).expanduser() / "bin" for key in homes if os.environ.get(key))
    if system != "windows":
        base = runtime_root()
        candidates.extend(base / name / "bin" for name in managed)
    candidates.extend(Path(entry).expanduser() for entry in os.get_exec_path() if entry)
    return tuple(dict.fromkeys(directory.resolve() for directory in candidates))


def _select_commands(directories, names, system, *, fsl=False, require_topup_config=False):
    """Select one prefix per tool family, so binaries and data never come from separate installations."""
    selected = None
    selected_commands = dict.fromkeys(names)
    best_score = -1
    for directory in directories:
        if fsl and directory.parts[-3:] == ("share", "fsl", "bin"):
            directory = fsl_root_from_bin_dir(directory) / "bin"
        commands = {}
        for name in names:
            binary = "eddy_cpu" if name == "eddy" else name
            suffix = ".exe" if system == "windows" else ""
            commands[name] = _native_binary(directory / (binary + suffix), system)
        topup_ok = not require_topup_config or (fsl_root_from_bin_dir(directory) / TOPUP_CONFIG).is_file()
        score = sum(path is not None for path in commands.values()) + int(topup_ok)
        if score > best_score:
            selected, selected_commands, best_score = directory, commands, score
        if topup_ok and all(commands.values()):
            break
    return selected, selected_commands


def inspect_native_fsl_runtime(
    mrtrix_bin_dir: Path | str | None,
    fsl_bin_dir: Path | str | None,
    *,
    reverse_phase: bool,
    platform: str | None = None,
) -> dict:
    """Inspect the exact commands and topup data needed by the native route."""
    system = (platform or sys.platform).lower()
    system = {"win32": "windows", "macos": "darwin"}.get(system, system)
    mrtrix_names = MRTRIX_COMMANDS + (REVERSE_MRTRIX_COMMANDS if reverse_phase else ())
    fsl_names = FSL_COMMANDS + (REVERSE_FSL_COMMANDS if reverse_phase else ())
    mrtrix_bin, mrtrix_commands = _select_commands(
        _candidate_dirs(mrtrix_bin_dir, "mrtrix", system), mrtrix_names, system,
    )
    fsl_bin, fsl_commands = _select_commands(
        _candidate_dirs(fsl_bin_dir, "fsl", system), fsl_names, system, fsl=True,
        require_topup_config=reverse_phase,
    )
    fsl_root = fsl_root_from_bin_dir(fsl_bin) if fsl_bin is not None else None
    topup_config = fsl_root / TOPUP_CONFIG if reverse_phase and fsl_root is not None else None
    if topup_config is not None and not topup_config.is_file():
        topup_config = None
    commands = {**mrtrix_commands, **fsl_commands}
    missing_commands = [name for name, path in commands.items() if path is None]
    missing_required = missing_commands + (["topup_config"] if reverse_phase and topup_config is None else [])
    return {
        "commands": commands,
        "required_commands": list(commands),
        "missing_commands": missing_commands,
        "missing_required": missing_required,
        "available": not missing_required,
        "eddy_device": "cpu",
        "mrtrix_bin_dir": str(mrtrix_bin) if mrtrix_bin is not None else None,
        "fsl_bin_dir": str(fsl_bin) if fsl_bin is not None else None,
        "fsl_root": str(fsl_root) if fsl_root is not None else None,
        "topup_config": str(topup_config) if topup_config is not None else None,
    }


def native_fsl_environment(
    commands: Mapping[str, str | None],
    fsl_root: Path | str | None,
) -> dict[str, str]:
    """Use the resolved native binaries without creating interpreter shims."""
    paths = tuple(dict.fromkeys(Path(command).parent for command in commands.values() if command))
    extra = {"FSLOUTPUTTYPE": FSL_OUTPUT_TYPE}
    if fsl_root is not None:
        extra["FSLDIR"] = str(fsl_root)
    env = external_python_environment(extra, path_prepend=paths)
    if sys.platform == "win32" and not env.get("HOME"):
        # MRtrix resolves its Windows configuration via HOME.
        env["HOME"] = env.get("USERPROFILE") or str(Path.home())
    return env
