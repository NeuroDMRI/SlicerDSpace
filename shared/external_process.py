"""Helpers for launching external commands safely from Slicer or headless CLI."""

from __future__ import annotations

import os
import sys
import shutil
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Iterable, Mapping, Optional


def resolve_external_command(name: str, bin_dir: Optional[Path] = None, *, require_executable=False) -> Optional[str]:
    """Prefer configured binaries, including Windows .exe, without shell lookup."""
    if bin_dir is not None:
        names = (name + '.exe', name) if sys.platform == 'win32' and not name.lower().endswith('.exe') else (name,)
        for basename in names:
            candidate = Path(bin_dir).expanduser() / basename
            if candidate.is_file() and (not require_executable or os.access(candidate, os.X_OK)):
                return candidate.resolve().as_posix()
    return shutil.which(name)


def interpreter_path(path: Path | str) -> Path:
    """Absolute, normalized interpreter path that keeps a virtualenv's ``bin/python`` symlink."""
    # C6: abspath removes ``..`` like ``sys.executable`` does; it does not follow symlinks.
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def slicer_runtime_roots(home: Path | str) -> tuple[Path, ...]:
    """Accept an install root or macOS .app without hardcoding an OS layout."""
    root = Path(home).expanduser()
    candidates = (root, root / "Contents", root.parent if root.name == "MacOS" else root)
    return tuple(dict.fromkeys(path for path in candidates if path.is_dir()))


def find_pythonslicer(home: Path | str) -> Optional[Path]:
    """PythonSlicer of a Slicer install root or macOS .app (Linux bin/, macOS Contents/bin/, Windows bin/*.exe)."""
    for root in slicer_runtime_roots(home):
        for relative in ("bin/PythonSlicer", "bin/PythonSlicer.exe", "PythonSlicer", "PythonSlicer.exe",
                         "MacOS/PythonSlicer"):
            if (root / relative).is_file():
                return root / relative
    return None


def startup_environment(extra_env: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """Return an environment suitable for external tools.

    When running inside Slicer, ``slicer.util.startupEnvironment()`` strips
    Slicer's runtime library-path overrides that can break external tools.
    Outside Slicer we fall back to the current process environment.
    """

    env: dict[str, str]
    try:
        slicer_module = sys.modules.get("slicer")
        if slicer_module is None:
            # See shared.pipeline: Slicer's uninitialized python-real helper
            # may print an import error before raising.  Environment probing
            # must stay silent because callers can emit machine-readable JSON.
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                import slicer as slicer_module  # type: ignore[import-not-found]

        env = dict(slicer_module.util.startupEnvironment())
    except Exception:
        env = os.environ.copy()

    if extra_env:
        env.update({str(key): str(value) for key, value in extra_env.items() if value is not None})
    return env


def merged_startup_environment(extra_env: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """Alias used at call sites where an existing env dict is extended."""

    return startup_environment(extra_env)


def _path_parts(value: Optional[str]) -> list[str]:
    if not value:
        return []
    return [part for part in value.split(os.pathsep) if part]


def _looks_like_slicer_runtime_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    return any(
        marker in normalized
        for marker in (
            "/slicer-",
            "/slicer.app/",
            "/slicer.org/extensions",
            "/lib/slicer-",
            "/qt-scripted-modules",
            "/qt-loadable-modules",
            "/cli-modules",
            "/pythonqt",
        )
    )


def _dedupe_paths(paths: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for path in paths:
        if not path:
            continue
        key = os.path.normpath(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def text_subprocess_kwargs(**overrides: object) -> dict[str, object]:
    """Return kwargs for a text-mode ``subprocess.run``/``Popen`` call.

    Passing only ``text=True`` lets Python decode subprocess output with
    ``locale.getpreferredencoding()``, which is not guaranteed to be UTF-8
    (notably on Windows). A byte sequence that is valid UTF-8 but not valid
    under that locale encoding then raises ``UnicodeDecodeError`` partway
    through an otherwise-successful run, losing output instead of just a
    few unrepresentable characters. Force UTF-8 and replace undecodable
    bytes instead of raising.

    Extra keyword arguments (e.g. ``bufsize=1`` for a streaming ``Popen``)
    are merged into the returned dict so call sites can still do
    ``subprocess.Popen(cmd, **text_subprocess_kwargs(bufsize=1), ...)``.
    """

    return {"text": True, "encoding": "utf-8", "errors": "replace", **overrides}


def external_python_environment(
    extra_env: Optional[Mapping[str, str]] = None,
    *,
    path_prepend: Optional[Iterable[Path | str]] = None,
    strip_slicer_paths: bool = True,
) -> dict[str, str]:
    """Return an environment for external Python-based command-line tools.

    Slicer and ``PythonSlicer`` export ``PYTHONHOME``/``PYTHONPATH`` and prepend
    Slicer runtime directories to ``PATH``. Python scripts launched via
    ``#!/usr/bin/env python`` can then import Slicer's stdlib with a different
    interpreter and fail with errors such as ``SRE module mismatch``. This
    helper keeps the user's normal environment shape but removes Slicer Python
    overrides before launching external Python tools such as MRtrix scripts.
    """

    env = startup_environment()
    for key in (
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONNOUSERSITE",
        "PYTHONEXECUTABLE",
        "PYTHONUSERBASE",
    ):
        env.pop(key, None)

    prepend = [Path(path).expanduser().as_posix() for path in (path_prepend or [])]
    for key in ("PATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH"):
        parts = _path_parts(env.get(key))
        if strip_slicer_paths:
            parts = [part for part in parts if not _looks_like_slicer_runtime_path(part)]
        if key == "PATH":
            parts = prepend + parts
        env[key] = os.pathsep.join(_dedupe_paths(parts))

    if extra_env:
        env.update({str(key): str(value) for key, value in extra_env.items() if value is not None})
    return env


def is_slicer_python(python_executable: Path | str) -> bool:
    """True when the interpreter is Slicer's own (PythonSlicer, python-real or a path inside a Slicer install)."""
    path = Path(python_executable).expanduser().absolute()
    # C9: a virtualenv is never Slicer's Python, even under a folder named like "Slicer-envs".
    in_venv = (path.parent.parent / "pyvenv.cfg").is_file()
    return path.name in {"PythonSlicer", "python-real"} or (not in_venv and _looks_like_slicer_runtime_path(str(path)))


def python_tool_environment(python_executable: Path | str,
                            extra_env: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """Keep Slicer's launcher environment only for its own interpreter.

    Preserve a selected virtualenv executable's symlink path, but do not let
    Slicer's PYTHONHOME or library overrides leak into that environment.
    """
    path = Path(python_executable).expanduser().absolute()
    if is_slicer_python(path):
        return startup_environment(extra_env)
    return external_python_environment(extra_env, path_prepend=[path.parent])
