#!/usr/bin/env python3
"""Tests for Python/conda environment resolution."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from shared.python_environment import resolve_python_executable


class PythonEnvironmentTest(unittest.TestCase):
    def test_explicit_python_executable_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            python = Path(tmp) / "PythonSlicer"
            python.write_text("", encoding="utf-8")

            self.assertEqual(resolve_python_executable(python_executable=python), python)

    def test_python_environment_can_be_python_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            python = Path(tmp) / "python"
            python.write_text("", encoding="utf-8")

            self.assertEqual(resolve_python_executable(python_environment=python.as_posix()), python)

    def test_python_environment_name_resolves_through_conda_run(self) -> None:
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected_python = root / "envs" / "slicerdsurfer" / "bin" / "python"
            expected_python.parent.mkdir(parents=True)
            expected_python.write_text("", encoding="utf-8")
            if sys.platform == "win32":
                conda = root / "conda.cmd"
                conda.write_text(
                    "@echo off\r\n"
                    'if not "%~1"=="run" exit /b 1\r\n'
                    'if not "%~2"=="-n" exit /b 1\r\n'
                    'if not "%~3"=="slicerdsurfer" exit /b 1\r\n'
                    f"echo {expected_python.as_posix()}\r\n",
                    encoding="utf-8",
                )
            else:
                conda = root / "conda"
                conda.write_text(
                    "#!/usr/bin/env python3\n"
                    "import sys\n"
                    "assert sys.argv[1:4] == ['run', '-n', 'slicerdsurfer']\n"
                    f"print({expected_python.as_posix()!r})\n",
                    encoding="utf-8",
                )
                conda.chmod(0o755)

            self.assertEqual(
                resolve_python_executable(python_environment="slicerdsurfer", conda_executable=conda),
                expected_python,
            )

    def test_conda_env_name_requires_conda_or_explicit_python(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "conda"):
            resolve_python_executable(python_environment="definitely_missing_env", conda_executable=Path("/missing/conda"))


if __name__ == "__main__":
    unittest.main()
