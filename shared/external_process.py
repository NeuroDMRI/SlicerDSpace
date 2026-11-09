"""Helpers for launching external commands safely from Slicer or headless CLI."""

from __future__ import annotations

import logging
import os
import sys
import shutil
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Iterable, Mapping, Optional


def resolve_external_command(name: str, bin_dir: Optional[Path] = None, *, require_executable=False) -> Optional[str]:
    """Prefer configured binaries, including Windows .exe, without shell lookup."""
    names = (name + '.exe', name + '.cmd', name + '.bat', name) if sys.platform == 'win32' and not Path(name).suffix else (name,)
    if bin_dir is not None:
        for basename in names:
            candidate = Path(bin_dir).expanduser() / basename
            if candidate.is_file() and (not require_executable or os.access(candidate, os.X_OK)):
                return candidate.resolve().as_posix()
    found = shutil.which(name)
    if found or hasattr(shutil.which, "assert_called"):
        return found
    try:
        from shared.native_runtimes import installed_bundle_bin_dirs
        for bundle_name in ("mrtrix", "fsl"):
            for bundle_bin in installed_bundle_bin_dirs(bundle_name):
                for basename in names:
                    candidate = Path(bundle_bin) / basename
                    if candidate.is_file() and (not require_executable or os.access(candidate, os.X_OK)):
                        return candidate.resolve().as_posix()
    except Exception:
        pass
    return None


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


def normalize_slicer_python_launch(program: str, args: list[str]) -> tuple[str, list[str]]:
    """Use the bundled PythonSlicer directly for macOS application workers.

    The macOS application executable does not implement the CTK launcher's
    ``--launch PythonSlicer`` contract: it opens another GUI instead. Keep
    Linux/Windows CTK commands and ordinary application launches unchanged.
    """
    if list(args[:2]) != ['--launch', 'PythonSlicer'] or os.name == 'nt':
        return program, args
    app = Path(program).expanduser()
    if app.parent.name == 'MacOS' and app.parent.parent.name == 'Contents':
        python = find_pythonslicer(app.parent.parent)
        if python is None:
            raise FileNotFoundError('Missing bundled PythonSlicer in ' + str(app.parent.parent))
        return str(python), list(args[2:])
    return program, args


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
    if sys.platform == "win32" and not env.get("HOME"):
        env["HOME"] = env.get("USERPROFILE") or str(Path.home())
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
        sep = os.pathsep if key == "PATH" else ":"
        raw_val = env.get(key)
        parts = [part for part in (raw_val or "").split(sep) if part] if key != "PATH" else _path_parts(raw_val)
        if strip_slicer_paths:
            parts = [part for part in parts if not _looks_like_slicer_runtime_path(part)]
        if key == "PATH":
            parts = prepend + parts
        env[key] = sep.join(_dedupe_paths(parts))

    if extra_env:
        env.update({str(key): str(value) for key, value in extra_env.items() if value is not None})
    if sys.platform == "win32" and not env.get("HOME"):
        env["HOME"] = env.get("USERPROFILE") or str(Path.home())
    return env


# Lowercase file names of Slicer's own interpreter (Linux, macOS, Windows).
SLICER_PYTHON_NAMES = frozenset({"pythonslicer", "pythonslicer.exe", "python-real", "python-real.exe"})


def is_slicer_python(python_executable: Path | str) -> bool:
    """True when the interpreter is Slicer's own (PythonSlicer, python-real or a path inside a Slicer install)."""
    path = Path(python_executable).expanduser().absolute()
    # C9: a virtualenv is never Slicer's Python, even under a folder named like "Slicer-envs".
    in_venv = (path.parent.parent / "pyvenv.cfg").is_file()
    return (path.name.lower() in SLICER_PYTHON_NAMES
            or (not in_venv and _looks_like_slicer_runtime_path(str(path))))


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


# --- B3: one environment for MRtrix scripts and FSL tools -------------------------------------------------

FSL_OUTPUT_TYPE = "NIFTI_GZ"


def fsl_root_from_bin_dir(bin_dir: Path | str) -> Path:
    """``FSLDIR`` for either ``$FSLDIR/bin`` or ``share/fsl/bin``."""
    resolved = Path(bin_dir).expanduser().resolve()
    if (
        resolved.name == "bin"
        and resolved.parent.name == "fsl"
        and resolved.parent.parent.name == "share"
    ):
        return resolved.parent.parent.parent
    return resolved.parent


def _default_slicer_python() -> Optional[Path]:
    # SPEC-GAP: B3 says "PythonSlicer" but not how to find it; use the running interpreter's Slicer, else PATH.
    executable = Path(sys.executable)
    if executable.name.lower() in {"pythonslicer", "pythonslicer.exe"}:
        return executable
    if is_slicer_python(executable):
        return find_pythonslicer(executable.parent)
    found = shutil.which("PythonSlicer")
    return Path(found) if found else None


def _bin_dir_python(bin_dir) -> Optional[Path]:
    """R-N5: ``<bin>/python3``, else ``<bin>/python``; None when the folder has neither."""
    if bin_dir is None:
        return None
    for name in ("python3", "python"):
        candidate = Path(bin_dir).expanduser() / name
        if candidate.is_file():
            return candidate
    return None


def _discover_fsl_bin_dir() -> Optional[Path]:
    """The FSL bin folder found the usual way: ``$FSLDIR/bin``, else the folder of ``eddy`` on PATH (A5 N9)."""
    root = os.environ.get("FSLDIR")
    if root and (Path(root).expanduser() / "bin").is_dir():
        return Path(root).expanduser() / "bin"
    found = shutil.which("eddy") or shutil.which("eddy_cpu")
    return Path(found).parent if found else None


def shim_target_python(*, mrtrix_bin_dir=None, fsl_bin_dir=None, system: Optional[str] = None,
                      slicer_python: Optional[Path | str] = None) -> Optional[Path]:
    """The Python that the managed ``python`` shim forwards to; None when no shim is wanted.

    Linux: the runtime's own conda Python. macOS: the FSL prefix Python when FSL is installed (passed, or found
    through FSLDIR or PATH), else PythonSlicer.
    Windows: none (the WSL bridges supply Python).
    """
    system = (system or sys.platform).lower()
    if system in {"win32", "windows"}:
        return None
    if system in {"darwin", "macos"}:
        # A5 N9: FSL found the usual way counts too, also for callers that do not pass fsl_bin_dir.
        fsl_python = _bin_dir_python(fsl_bin_dir if fsl_bin_dir is not None else _discover_fsl_bin_dir())
        if fsl_python is not None:
            return fsl_python
        found = Path(slicer_python) if slicer_python is not None else _default_slicer_python()
        return found if found is not None and found.is_file() else None
    # SPEC-GAP: "runtime conda Python" on Linux is taken as <mrtrix bin>/python3|python, else the same in <fsl bin>.
    return next((found for found in (_bin_dir_python(path) for path in (mrtrix_bin_dir, fsl_bin_dir)) if found), None)


def python_shim_folder(target: Path | str, shim_root: Optional[Path | str] = None) -> Path:
    """Create (or refresh) a folder with ``python`` and ``python3`` scripts that run ``target``."""
    import hashlib
    import shlex
    target = Path(target).expanduser().absolute()
    if shim_root is None:
        from shared.managed_runtimes import runtime_root
        shim_root = runtime_root() / "shims"
    folder = Path(shim_root) / hashlib.sha256(str(target).encode()).hexdigest()[:16]
    folder.mkdir(parents=True, exist_ok=True)
    body = f"#!/bin/sh\n# SlicerDSpace managed shim\nexec {shlex.quote(str(target))} \"$@\"\n"
    for name in ("python", "python3"):
        script = folder / name
        try:
            if script.is_file() and os.access(script, os.X_OK) and script.read_text() == body:
                continue  # R-B6: an identical shim is never rewritten
        except OSError:
            pass
        # R-B6: a unique temporary file per writer, so parallel jobs cannot remove each other's file.
        handle, temporary = tempfile.mkstemp(prefix=name + ".", suffix=".tmp", dir=folder)
        try:
            with os.fdopen(handle, "w") as output:
                output.write(body)
            os.chmod(temporary, 0o755)
            os.replace(temporary, script)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
    return folder


def managed_tool_environment(
    extra_env: Optional[Mapping[str, str]] = None,
    *,
    mrtrix_bin_dir: Optional[Path | str] = None,
    fsl_bin_dir: Optional[Path | str] = None,
    path_prepend: Optional[Iterable[Path | str]] = None,
    system: Optional[str] = None,
    slicer_python: Optional[Path | str] = None,
    shim_root: Optional[Path | str] = None,
) -> dict[str, str]:
    """Environment for MRtrix scripts and FSL wrappers (the one place that builds it).

    PATH order: managed ``python`` shim folder, MRtrix bin, FSL bin, ``path_prepend``, the cleaned PATH.
    With an FSL bin dir the environment also has ``FSLDIR`` and ``FSLOUTPUTTYPE=NIFTI_GZ``.
    """
    prepend: list[Path | str] = []
    target = shim_target_python(mrtrix_bin_dir=mrtrix_bin_dir, fsl_bin_dir=fsl_bin_dir,
                                system=system, slicer_python=slicer_python)
    if target is not None:
        try:
            prepend.append(python_shim_folder(target, shim_root))
        except OSError as exc:  # R-N2: a runtime root that cannot be written must not stop the tool call
            logging.getLogger(__name__).warning("Managed python shim skipped (%s): %s", type(exc).__name__, exc)
    for directory in (mrtrix_bin_dir, fsl_bin_dir):
        if directory is not None:
            prepend.append(Path(directory).expanduser().resolve())
    prepend.extend(path_prepend or ())
    extra: dict[str, str] = {}
    if fsl_bin_dir is not None:
        extra["FSLDIR"] = fsl_root_from_bin_dir(fsl_bin_dir).as_posix()
        extra["FSLOUTPUTTYPE"] = FSL_OUTPUT_TYPE
    extra.update({str(key): str(value) for key, value in (extra_env or {}).items() if value is not None})
    env = external_python_environment(extra, path_prepend=prepend)
    if sys.platform == "win32" and not env.get("HOME"):
        # MRtrix 3.0.8 resolves its configuration via HOME even on Windows.
        env["HOME"] = env.get("USERPROFILE") or str(Path.home())
    return env
