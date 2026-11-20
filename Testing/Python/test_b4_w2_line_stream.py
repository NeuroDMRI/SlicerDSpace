"""W2 / H8: a surviving grandchild must not keep a stage open after the direct child exits."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from shared.line_stream import DESCENDANTS_NOTICE, stream_process_lines

POSIX = unittest.skipUnless(os.name == "posix", "needs /bin/sh and sleep")
# The direct child prints, starts a background grandchild that keeps the pipe for 8 s, and exits at once.
LEAVER = ["/bin/sh", "-c", "echo first line; sleep 8 & exit 3"]
LEAVER_OK = ["/bin/sh", "-c", "echo first line; sleep 8 & exit 0"]
LIMIT = 6.0  # grace is 5 s; the unfixed code returns after about 8 s


def _popen(command):
    return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)


class LineStreamTests(unittest.TestCase):
    @POSIX
    def test_h8_helper_returns_the_exit_code_after_grace_and_reports_the_notice(self):
        lines, notes = [], []
        started = time.monotonic()
        code = stream_process_lines(_popen(LEAVER), lines.append, notice=notes.append, grace_seconds=0.5)
        self.assertLess(time.monotonic() - started, 3)
        self.assertEqual(code, 3)
        self.assertEqual(lines, ["first line\n"])
        self.assertEqual(notes, [DESCENDANTS_NOTICE])

    @POSIX
    def test_h8_normal_child_has_all_lines_and_no_notice(self):
        lines, notes = [], []
        process = _popen(["/bin/sh", "-c", "for i in 1 2 3; do echo line$i; done"])
        self.assertEqual(stream_process_lines(process, lines.append, notice=notes.append), 0)
        self.assertEqual(lines, ["line1\n", "line2\n", "line3\n"])
        self.assertEqual(notes, [])
        self.assertTrue(process.stdout.closed)

    @POSIX
    def test_h8_late_output_of_a_slow_writer_is_still_delivered_within_grace(self):
        lines = []
        process = _popen(["/bin/sh", "-c", "sleep 0.3; echo late"])
        stream_process_lines(process, lines.append)
        self.assertEqual(lines, ["late\n"])

    @POSIX
    def test_h8_brain_mask_run(self):
        from shared.brain_mask import _run
        logs = []
        started = time.monotonic()
        self.assertEqual(_run(LEAVER_OK, dict(os.environ), logs.append).strip(), "first line")
        self.assertLess(time.monotonic() - started, LIMIT)
        self.assertIn("[Brain mask] first line", logs)
        self.assertIn("[Brain mask] " + DESCENDANTS_NOTICE, logs)

    @POSIX
    def test_h8_brain_mask_failure_still_reports_the_exit_code(self):
        from shared.brain_mask import _run
        started = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, r"failed \(3\)"):
            _run(LEAVER, dict(os.environ), None)
        self.assertLess(time.monotonic() - started, LIMIT)

    @POSIX
    def test_h8_shared_pipeline_run_cli(self):
        from shared.pipeline import _run_cli
        logs = []
        started = time.monotonic()
        _run_cli(LEAVER_OK, logs.append)
        self.assertLess(time.monotonic() - started, LIMIT)
        self.assertIn("[Shared] first line", logs)
        self.assertIn("[Shared] " + DESCENDANTS_NOTICE, logs)
        with self.assertRaisesRegex(RuntimeError, r"exit code 3 \(sh: first line\)"):
            _run_cli(LEAVER, None)

    @POSIX
    def test_h8_preprocessing_run_command(self):
        from preprocessing.pipeline import _run_command
        logs = []
        started = time.monotonic()
        out = _run_command(LEAVER_OK, env=dict(os.environ), log=logs.append, capture=True)
        self.assertLess(time.monotonic() - started, LIMIT)
        self.assertEqual(out, "first line\n")
        self.assertIn("[Preproc] " + DESCENDANTS_NOTICE, logs)

    @POSIX
    def test_h8_surface_backend_bridge(self):
        from anatomy.surface import backend_bridge
        logs = []
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "leaver.sh"
            script.write_text("#!/bin/sh\necho first line\nsleep 8 &\nexit 0\n")
            script.chmod(0o755)
            started = time.monotonic()
            # python_executable() returns the configured binary; the "interpreter" here is the shell script.
            backend_bridge.run_backend([], python_bin=str(script), log=logs.append, log_path=Path(tmp) / "run.log")
            self.assertLess(time.monotonic() - started, LIMIT)
            self.assertIn("first line\n", (Path(tmp) / "run.log").read_text())
        self.assertIn("first line", logs)
        self.assertIn("[DDSurfer] " + DESCENDANTS_NOTICE, logs)


if __name__ == "__main__":
    unittest.main()
