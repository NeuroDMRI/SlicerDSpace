"""Resolve Python executables from explicit paths or conda environment names."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from shared.external_process import (
    external_python_environment,
    interpreter_path,
    slicer_runtime_roots,
    text_subprocess_kwargs,
)


def setup_python_command(python, *, slicer_home, launcher, current_python=None):
    """Slicer's raw interpreter needs its launcher; external venvs keep their identity."""
    if not python:
        return launcher, ['--launch', 'PythonSlicer']
    selected = interpreter_path(python)
    current = interpreter_path(current_python or sys.executable)
    home = Path(slicer_home).expanduser().resolve()
    slicer_bins = {(root / 'bin').resolve() for root in (slicer_runtime_roots(home) or (home,))}
    own_python = (selected.parent.resolve() in slicer_bins and selected.name.lower() in
                  {'python-real', 'python-real.exe', 'pythonslicer', 'pythonslicer.exe'})
    if selected == current or own_python:
        return launcher, ['--launch', 'PythonSlicer']
    return str(selected), []


def _looks_like_path(value: str) -> bool:
    return "/" in value or "\\" in value or value.startswith(".") or value.startswith("~")


def resolve_python_executable(
    *,
    python_executable: Optional[Path] = None,
    python_environment: Optional[str] = None,
    conda_executable: Optional[Path] = None,
) -> Path:
    """Resolve the Python executable used for dependency checks and pip installs.

    Resolution order:
    1. Explicit python executable path.
    2. ``python_environment`` as a path to a Python executable.
    3. ``python_environment`` as a conda environment name via ``conda run``.
    4. Current Python executable.
    """

    if python_executable is not None:
        return interpreter_path(python_executable)

    env_text = str(python_environment or "").strip()
    if env_text:
        if _looks_like_path(env_text):
            return interpreter_path(env_text)
        conda = conda_executable.expanduser().resolve() if conda_executable is not None else None
        if conda is None:
            resolved = shutil.which("conda") or shutil.which("mamba")
            if resolved:
                conda = Path(resolved).resolve()
        if conda is None:
            raise FileNotFoundError(
                f"Cannot resolve conda environment '{env_text}' because conda/mamba was not found. "
                "Set --conda-executable or pass --python-executable."
            )
        if not conda.is_file():
            raise FileNotFoundError(f"Configured conda/mamba executable does not exist: {conda}")
        completed = subprocess.run(
            [
                conda.as_posix(),
                "run",
                "-n",
                env_text,
                "python",
                "-c",
                "import sys; print(sys.executable)",
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=external_python_environment(),
            **text_subprocess_kwargs(),
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Cannot resolve conda environment '{env_text}' with {conda}: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        python_path = completed.stdout.strip().splitlines()[-1].strip() if completed.stdout.strip() else ""
        if not python_path:
            raise RuntimeError(f"Conda environment '{env_text}' did not report a Python executable.")
        return interpreter_path(python_path)

    return interpreter_path(sys.executable)
