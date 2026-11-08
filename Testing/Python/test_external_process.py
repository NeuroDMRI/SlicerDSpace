#!/usr/bin/env python3
"""Tests for shared/external_process.py's subprocess kwargs helper."""

from __future__ import annotations

import subprocess
import os
import sys
import unittest
import tempfile
from unittest.mock import patch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.external_process import (
    external_python_environment, interpreter_path, python_tool_environment, text_subprocess_kwargs)


class TextSubprocessKwargsTest(unittest.TestCase):
    def test_windows_configured_executables_are_used_by_all_mrtrix_clients(self):
        from tractography.csd import _resolve_command as csd_command
        from tractography.act_prep import _resolve_command as act_command
        from connectome.pipeline import _resolve_command as connectome_command
        from tractography.saf import _resolve_command as saf_command
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executable = root / 'mrconvert.exe'
            executable.touch()
            executable.chmod(0o755)
            with patch('shared.external_process.sys.platform', 'win32'):
                for resolver in (csd_command, act_command, connectome_command):
                    self.assertEqual(resolver('mrconvert', root), executable.resolve().as_posix())
                self.assertEqual(saf_command('mrconvert', bin_dir=root), executable.resolve().as_posix())
    def test_macos_external_python_does_not_inherit_slicer_dylib_paths(self):
        environment = {"PATH": "/usr/bin", "DYLD_LIBRARY_PATH": "/Applications/Slicer.app/Contents/lib:/opt/real-tool/lib",
                       "DYLD_FALLBACK_LIBRARY_PATH": "/Applications/Slicer.app/Contents/lib"}
        with patch("shared.external_process.startup_environment", return_value=environment.copy()):
            actual = external_python_environment()
        self.assertEqual(actual["DYLD_LIBRARY_PATH"], "/opt/real-tool/lib")
        self.assertEqual(actual["DYLD_FALLBACK_LIBRARY_PATH"], "")
    def test_c6_interpreter_path_normalizes_dotdot_and_keeps_the_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            real = root / "base/bin/python3"
            real.parent.mkdir(parents=True)
            real.touch()
            link = root / "venv/bin/python"
            link.parent.mkdir(parents=True)
            try:
                link.symlink_to(real)
            except OSError as exc:
                if sys.platform == "win32":
                    self.skipTest(f"Windows symlink privilege unavailable: {exc}")
                raise
            cwd = root / "work"
            cwd.mkdir()
            old = os.getcwd()
            os.chdir(cwd)
            try:
                actual = interpreter_path("../venv/bin/python")
            finally:
                os.chdir(old)
            self.assertEqual(actual, link)
            self.assertNotIn("..", actual.parts)
            self.assertTrue(actual.is_symlink())

    def test_c9_virtualenv_under_a_slicer_named_folder_is_not_slicers_python(self) -> None:
        startup = {"PATH": "/usr/bin", "PYTHONHOME": "/slicer/lib/Python"}
        with tempfile.TemporaryDirectory() as tmp:
            python = Path(tmp).resolve() / "Slicer-envs/ddsurfer/bin/python"
            python.parent.mkdir(parents=True)
            python.touch()
            with patch("shared.external_process.startup_environment", return_value=startup.copy()):
                self.assertEqual(python_tool_environment(python)["PYTHONHOME"], "/slicer/lib/Python")  # no venv yet
                (python.parent.parent / "pyvenv.cfg").write_text("home = /usr/bin\n")
                self.assertNotIn("PYTHONHOME", python_tool_environment(python))

    def test_defaults_force_utf8_with_replacement(self) -> None:
        kwargs = text_subprocess_kwargs()
        self.assertEqual(
            kwargs,
            {"text": True, "encoding": "utf-8", "errors": "replace"},
        )

    def test_overrides_are_merged_in(self) -> None:
        kwargs = text_subprocess_kwargs(bufsize=1)
        self.assertEqual(
            kwargs,
            {"text": True, "encoding": "utf-8", "errors": "replace", "bufsize": 1},
        )

    def test_kwargs_are_accepted_by_a_real_subprocess_call(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-c", "print('ok')"],
            capture_output=True,
            check=True,
            **text_subprocess_kwargs(),
        )
        self.assertEqual(completed.stdout.strip(), "ok")

    def test_replace_errors_do_not_raise_on_undecodable_bytes(self) -> None:
        # A byte sequence that is not valid under a non-UTF-8 locale, but
        # would be silently mis-decoded (or raise) without errors="replace"
        # and an explicit encoding. \xff is invalid in UTF-8 on its own.
        script = "import sys; sys.stdout.buffer.write(b'before\\xffafter')"
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            check=True,
            **text_subprocess_kwargs(),
        )
        self.assertIn("before", completed.stdout)
        self.assertIn("after", completed.stdout)


if __name__ == "__main__":
    unittest.main()
