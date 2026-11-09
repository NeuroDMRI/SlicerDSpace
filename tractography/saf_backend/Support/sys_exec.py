import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from shared.external_process import resolve_external_command, startup_environment, text_subprocess_kwargs


def _mrtrix_seed_extra_env(PS):
    """Return an ``extra_env`` dict forwarding the MRtrix RNG seed, if any.

    ``startup_environment()`` uses ``slicer.util.startupEnvironment()``
    inside Slicer, which is a fixed snapshot captured at Slicer's own
    process launch time -- it does not reflect anything set on
    ``os.environ`` afterward. ``tractography/saf.py`` exports
    ``MRTRIX_RNG_SEED`` via ``os.environ`` (SAF runs in-process, not as a
    separate subprocess) for the duration of the SAF backend call; without
    explicitly forwarding it here via ``extra_env``, that seed would never
    reach the MRtrix commands this helper actually launches. ``PS`` (the
    SAF backend's own settings object) is checked too, for any caller that
    sets a seed there directly instead of via the environment.
    """

    seed = os.environ.get("MRTRIX_RNG_SEED") or getattr(PS, "random_seed", None)
    return {"MRTRIX_RNG_SEED": str(seed)} if seed is not None else None


def sys_exec(cmd, PS):
    """Run MRtrix argv or an explicitly configured legacy shell command."""

    extra_env = _mrtrix_seed_extra_env(PS)
    env = startup_environment(extra_env=extra_env)
    if not isinstance(cmd, str):
        cmd = [str(arg) for arg in cmd]
        prefix = getattr(PS, "mrtrix_prefix", "")
        if prefix:
            cmd = f"{prefix} {shlex.join(cmd)}"
        else:
            # Native MRtrix reads HOME for its configuration on Windows too.
            if sys.platform == "win32" and not env.get("HOME"):
                env["HOME"] = env.get("USERPROFILE") or str(Path.home())
            bin_dir = getattr(PS, "mrtrix_bin_dir", "")
            if bin_dir:
                env["PATH"] = os.pathsep.join(filter(None, (str(bin_dir), env.get("PATH"))))
                executable = resolve_external_command(cmd[0], Path(bin_dir), require_executable=True)
            else:
                executable = shutil.which(cmd[0], path=env.get("PATH", "")) or resolve_external_command(cmd[0], None, require_executable=True)
                if executable and Path(executable).is_file():
                    env["PATH"] = os.pathsep.join(filter(None, (str(Path(executable).resolve().parent), env.get("PATH"))))
            if not executable:
                raise FileNotFoundError(f"SAF command {cmd[0]} was not found. Configure the MRtrix bin directory.")
            cmd[0] = os.path.abspath(executable)

    shell = isinstance(cmd, str)
    if shell and sys.platform == "win32":
        # Backend paths use POSIX quoting, which cmd.exe does not understand.
        # Decode those quotes to argv, preserve the MRtrix PATH prefix in the
        # environment, and let the selected native/WSL executable handle argv.
        cmd = shlex.split(cmd)
        if cmd and cmd[0].startswith("PATH="):
            prefix = cmd.pop(0)[len("PATH="):]
            if not prefix.endswith(":$PATH"):
                raise ValueError("Unsupported SAF PATH prefix on Windows; configure MRtrix bin directory")
            env["PATH"] = prefix[:-len(":$PATH")] + os.pathsep + env.get("PATH", "")
        if not cmd or any(token in {"&&", "||", "|", ";", ">", "<"} for token in cmd):
            raise ValueError("Windows SAF requires a single executable command, not shell operators")
        command = shutil.which(cmd[0], path=env.get("PATH", ""))
        if command:
            cmd[0] = command
        shell = False
    if getattr(PS, 'dgn', False):
        return subprocess.run(
            cmd,
            shell=shell,
            check=True,
            env=env,
        )
    return subprocess.run(
        cmd,
        shell=shell,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        **text_subprocess_kwargs(),
    )
