"""Shared cross-platform helpers for Testing/Python unit tests."""

from __future__ import annotations

import functools
import os
from pathlib import Path
import sys
import tempfile
import unittest

_CAN_SYMLINK: bool | None = None


def can_create_symlinks() -> bool:
    """Return True if the current process can create filesystem symlinks."""
    global _CAN_SYMLINK
    if _CAN_SYMLINK is not None:
        return _CAN_SYMLINK
    if os.name != "nt":
        _CAN_SYMLINK = True
        return True
    try:
        with tempfile.TemporaryDirectory(prefix="slicerdspace-symlink-probe-") as tmp:
            root = Path(tmp)
            target = root / "target.txt"
            target.write_text("ok", encoding="utf-8")
            link = root / "link.txt"
            os.symlink(target, link)
            _CAN_SYMLINK = True
    except (OSError, NotImplementedError):
        _CAN_SYMLINK = False
    return _CAN_SYMLINK


def skip_unless_symlinks(obj):
    """Decorator skipping tests that create OS symlinks on non-elevated Windows hosts."""
    return unittest.skipUnless(
        can_create_symlinks(),
        "OS symlink creation requires Developer Mode or elevated privileges on Windows",
    )(obj)


def test_python_launcher() -> Path:
    """Return an invocable Python launcher (preferring PythonSlicer.exe over python-real.exe)."""
    exe = Path(sys.executable)
    if os.name == "nt" and exe.name.lower() == "python-real.exe":
        sibling = exe.with_name("PythonSlicer.exe")
        if sibling.is_file():
            return sibling
    return exe


def write_fake_python_cli(path: Path, py_body: str, *, launcher: Path | None = None) -> Path:
    """Create a self-contained single-file fake CLI executable (`.cmd` polyglot on Windows, script on POSIX).

    On Windows, if ``path`` has no executable suffix (``.cmd``/``.bat``/``.exe``), appends ``.cmd`` so
    ``shutil.which`` and ``CreateProcess`` both resolve and execute it cleanly.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    py_exec = launcher or test_python_launcher()
    if os.name == "nt":
        target = path if path.suffix.lower() in {".cmd", ".bat"} else path.with_name(path.name + ".cmd")
        extra_paths = [p for p in sys.path if "bundled-python" in p.lower() and os.path.isdir(p)]
        path_setup = f"sys.path[0:0]={extra_paths!r}; " if extra_paths else ""
        bootstrap = (
            f"import sys; {path_setup}sys.argv=sys.argv[1:]; "
            "src=open(sys.argv[0], encoding='utf-8').read().split('#__PY__\\n', 1)[1]; "
            "exec(compile(src, sys.argv[0], 'exec'), {'__name__': '__main__', '__file__': sys.argv[0]})"
        )
        content = (
            "@echo off\n"
            f'"{py_exec}" -c "{bootstrap}" "%~f0" %*\n'
            "exit /b %ERRORLEVEL%\n"
            "#__PY__\n"
            + py_body.lstrip("\n")
        )
        target.write_text(content, encoding="utf-8", newline="\n")
        return target
    path.write_text(f"#!{py_exec}\n" + py_body.lstrip("\n"), encoding="utf-8")
    path.chmod(0o755)
    return path
