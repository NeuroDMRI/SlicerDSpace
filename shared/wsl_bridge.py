"""Inspect plugin-owned WSL bridges without executing Windows binaries."""

from pathlib import Path

WSL_INSTALL_COMMAND = "wsl --install -d Ubuntu"
WSL_MISSING_LABEL = "Windows needs WSL 2 with Ubuntu for MRtrix3, FSL and FreeSurfer"


def decode_wsl_output(raw):
    """`wsl.exe -l -v` prints UTF-16LE (usually with a BOM) and CRLF; decode it to plain text."""
    if isinstance(raw, str):
        text = raw
    elif raw.startswith((b"\xff\xfe", b"\xfe\xff")) or b"\x00" in raw:
        text = raw.decode("utf-16", errors="replace")
    else:
        text = raw.decode("utf-8", errors="replace")
    return text.replace("\ufeff", "").replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")


def parse_distributions(raw):
    """Parse `wsl.exe -l -v` into [{name, state, version, default}]; the header line is skipped."""
    found = []
    for line in decode_wsl_output(raw).split("\n"):
        if not line.strip():
            continue
        default = line.lstrip().startswith("*")
        parts = line.replace("*", " ", 1).split() if default else line.split()
        if len(parts) < 3 or not parts[-1].isdigit():
            continue
        found.append({"name": parts[0], "state": " ".join(parts[1:-1]), "version": int(parts[-1]), "default": default})
    return found


def list_distributions(run=None):
    """Installed WSL distributions; an empty list when wsl.exe is missing, fails or times out."""
    import subprocess

    runner = run or subprocess.run
    try:
        done = runner(["wsl.exe", "-l", "-v"], capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return []
    if getattr(done, "returncode", 1) != 0:
        return []
    return parse_distributions(done.stdout or b"")


def select_distribution(distributions):
    """The default distribution if it is Ubuntu* on WSL 2, else the first Ubuntu* on WSL 2, else None (B5)."""
    def usable(item):
        return str(item.get("name", "")).lower().startswith("ubuntu") and item.get("version") == 2

    for item in distributions:
        if item.get("default") and usable(item):
            return item["name"]
    return next((item["name"] for item in distributions if usable(item)), None)


def saved_distribution(config=None):
    """The distribution name saved in the user config by the Windows setup, or None."""
    if config is None:
        from shared.user_config import load_user_config

        try:
            config = load_user_config()
        except (OSError, ValueError):
            return None
    value = str(config.get("wsl_distribution") or "").strip()
    return value or None


def check_bridge_distribution(executable, expected):
    """Refuse a bridge that was written for another distribution than the saved one."""
    config = bridge_config(executable)
    if config is not None and expected and config["distribution"] != expected:
        raise ValueError("WSL bridge %s uses distribution '%s' but the saved distribution is '%s'; rerun Windows runtime setup"
                         % (executable, config["distribution"], expected))
    return config


def bridge_config(executable):
    executable = Path(executable)
    marker = Path(str(executable) + ".slicerdsurfer-owner")
    if not marker.is_file():
        return None
    path = Path(str(executable) + ".conf")
    if not path.is_file():
        raise FileNotFoundError("WSL bridge configuration missing; rerun Windows runtime setup: " + str(path))
    fields, arguments = {}, []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator:
            raise ValueError("Invalid WSL bridge configuration: " + str(path))
        key = key.strip().lower()
        if key == "arg":
            arguments.append(value)
        elif key in {"target", "distribution"}:
            fields[key] = value
    if not fields.get("distribution") or not fields.get("target", "").startswith("/"):
        raise ValueError("WSL bridge needs a distribution and absolute Linux target: " + str(path))
    return dict(fields, arguments=arguments)


def preprocessing_launcher(command, record_path, environment):
    """Run the same strict eddy policy in the actual Linux/WSL MRtrix Python."""
    import shutil

    script = Path(command[0])
    if script.name.lower() not in {"dwifslpreproc", "dwifslpreproc.exe"}:
        return command
    policy = Path(__file__).with_name("mrtrix_preproc_runner.py")
    config = bridge_config(script)
    if config is not None:
        check_bridge_distribution(script, saved_distribution())
        python = script.parent / "python.exe"
        python_config = bridge_config(python)
        if not python.is_file() or python_config is None:
            raise FileNotFoundError("WSL MRtrix Python bridge missing; rerun scripts/install_deps_windows.ps1")
        if (python_config["distribution"] != config["distribution"]
                or python_config["target"] != config["target"]
                or len(config["arguments"]) != 1
                or not config["arguments"][0].startswith("/")
                or Path(config["arguments"][0]).name != "dwifslpreproc"
                or len(python_config["arguments"]) != 1
                or Path(python_config["arguments"][0]).parent != Path(config["arguments"][0]).parent
                or Path(python_config["arguments"][0]).name != "python"):
            raise ValueError("WSL MRtrix script/Python bridges must use the same distribution and runtime")
        target = config["arguments"][0]
    elif script.is_file() and "mrtrix3" in script.read_text(encoding="utf-8", errors="replace"):
        python = shutil.which("python", path=environment.get("PATH", ""))
        if not python:
            raise FileNotFoundError("The selected MRtrix external environment has no Python interpreter")
        target = str(script)
    elif script.suffix.lower() == ".exe":
        raise RuntimeError("Native Windows dwifslpreproc is not supported by this FSL route; configure the managed WSL bridges")
    else:
        return command
    return [str(python), str(policy), "--script", target, "--record", str(record_path), "--", *command[1:]]
