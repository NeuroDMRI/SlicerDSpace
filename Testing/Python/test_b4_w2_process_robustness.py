"""W2: H7 waiting logs, H9 process-group fallback, X12 explicit text encoding."""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

POSIX = unittest.skipUnless(os.name == "posix", "POSIX process groups and flock")


class CacheLockWaitLogTests(unittest.TestCase):
    @POSIX
    def test_h7_a_waiting_lock_logs_path_holder_and_seconds(self):
        from shared.cache_lock import cache_lock
        logs = []
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.lock"
            with cache_lock(path):
                with self.assertRaises(TimeoutError):
                    with cache_lock(path, timeout=1.0, log=logs.append, log_interval=0.3):
                        self.fail("entered a held lock")
        self.assertGreaterEqual(len(logs), 2)
        self.assertRegex(logs[0], rf"^waiting for .*cache\.lock \(held by pid {os.getpid()}\) for \d+ s$")

    @POSIX
    def test_h7_default_sink_is_stderr_and_a_free_lock_logs_nothing(self):
        from shared.cache_lock import cache_lock
        import io
        from contextlib import redirect_stderr
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.lock"
            err = io.StringIO()
            with redirect_stderr(err):
                with cache_lock(path, log_interval=0.1):
                    time.sleep(0.25)
            self.assertEqual(err.getvalue(), "")
            with cache_lock(path):
                with redirect_stderr(err):
                    with self.assertRaises(TimeoutError):
                        with cache_lock(path, timeout=0.5, log_interval=0.2):
                            pass
            self.assertIn("waiting for", err.getvalue())


class CohortWaitLogTests(unittest.TestCase):
    def test_h7_wait_description_names_the_blocking_resource_and_queue_position(self):
        from shared.cohort_schedule import _wait_description
        row = dict(id="b", cpu=2, memory_gb=1, gpu=0, io=0)
        state = dict(waiting=[dict(id="a"), row])
        limits = dict(cpu=2, memory_gb=8, gpu=1, io=2)
        used = dict(cpu=2, memory_gb=1, gpu=0, io=0)
        self.assertEqual(_wait_description(row, state, used, limits), ("cpu", 1))
        used["cpu"] = 0
        self.assertEqual(_wait_description(row, state, used, limits), ("queue", 1))

    def test_h7_queued_stage_logs_once_then_on_the_interval(self):
        from shared import cohort_schedule as cs
        logs = []

        @cs.scheduled_stage("fit")
        def work(*, log=None):
            return "done"

        with tempfile.TemporaryDirectory() as tmp:
            folder = cs.create_schedule(Path(tmp) / "queue", cpu_budget=1, memory_gb=4, stage_cpus=1, stage_memory_gb=1)
            with patch.dict(os.environ, {cs.ENV: str(folder)}), patch.object(cs, "WAIT_LOG_SECONDS", 0.4):
                result = []
                with cs.stage_admission("holder"):
                    worker = threading.Thread(target=lambda: result.append(work(log=logs.append)))
                    worker.start()
                    time.sleep(1.1)
                    held = list(logs)
                worker.join(timeout=10)
        self.assertEqual(result, ["done"])
        self.assertGreaterEqual(len(held), 2)
        self.assertEqual(held[0], "[Cohort] fit waiting for cpu (0 ahead)")
        self.assertLessEqual(len(held), 4)  # once, then every interval, not every 50 ms poll


class ProcessGroupFallbackTests(unittest.TestCase):
    @POSIX
    def test_h9_python_fallback_makes_the_target_its_own_group_leader(self):
        """The pid that the caller holds (Popen.pid) must be the group id, or killpg(pid) misses the tree."""
        from shared import process_group
        script = "import os; print(os.getpid(), os.getpgid(0))"
        with tempfile.TemporaryDirectory() as tmp:
            plain = Path(tmp) / "python3"  # a plain interpreter that is not a Slicer one
            plain.symlink_to(Path(sys.executable).resolve())
            with patch.object(process_group.sys, "executable", str(plain)), \
                    patch.object(process_group.shutil, "which", return_value=None), \
                    patch.object(process_group, "_perl_setpgrp_launcher", return_value=None), \
                    patch.object(process_group, "_python_setsid_launcher", return_value=str(plain)):
                program, args, grouped = process_group.grouped_launch_command(sys.executable, ["-c", script])
            self.assertTrue(grouped)
            self.assertEqual(program, str(plain))
            process = subprocess.Popen([program, *args], stdout=subprocess.PIPE, text=True)
            output = process.communicate(timeout=30)[0].split()
        self.assertEqual(process.returncode, 0)
        self.assertEqual(int(output[0]), process.pid)  # exec in place: same pid as the caller holds
        self.assertEqual(int(output[1]), process.pid)  # and it leads its own group

    @POSIX
    def test_h9_slicer_interpreters_are_not_used_because_they_change_the_pid(self):
        """PythonSlicer starts python-real as a child; grouped=True would make Cancel miss the group."""
        from shared import process_group
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("PythonSlicer", "python-real"):
                fake = Path(tmp) / name
                fake.write_text("#!/bin/sh\n")
                fake.chmod(0o755)
                with patch.object(process_group.sys, "executable", str(fake)):
                    self.assertIsNone(process_group._python_setsid_launcher(), name)
            plain = Path(tmp) / "python3"
            plain.write_text("#!/bin/sh\n")
            plain.chmod(0o755)
            with patch.object(process_group.sys, "executable", str(plain)):
                self.assertEqual(process_group._python_setsid_launcher(), str(plain))

    @POSIX
    def test_h9_python_fallback_passes_a_single_argument_unchanged(self):
        from shared import process_group
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "has space.sh"
            script.write_text("#!/bin/sh\necho ran\n")
            script.chmod(0o755)
            with patch.object(process_group.shutil, "which", return_value=None), \
                    patch.object(process_group, "_perl_setpgrp_launcher", return_value=None), \
                    patch.object(process_group, "_python_setsid_launcher", return_value=sys.executable):
                program, args, grouped = process_group.grouped_launch_command(str(script), [])
            out = subprocess.run([program, *args], capture_output=True, text=True, timeout=30)
        self.assertTrue(grouped)
        self.assertEqual(out.stdout.strip(), "ran")

    @POSIX
    def test_h9_a_warning_is_logged_when_no_grouping_is_possible(self):
        from shared import process_group
        with patch.object(process_group.shutil, "which", return_value=None), \
                patch.object(process_group, "_perl_setpgrp_launcher", return_value=None), \
                patch.object(process_group, "_python_setsid_launcher", return_value=None), \
                patch.object(process_group, "_warned_ungrouped", False):
            with self.assertLogs(process_group._logger, level="WARNING") as captured:
                program, args, grouped = process_group.grouped_launch_command("/bin/echo", ["x"])
        self.assertFalse(grouped)
        self.assertIn("Cancel will stop only the launcher", captured.output[0])
        self.assertIn("Install util-linux", captured.output[0])


class ExplicitEncodingTests(unittest.TestCase):
    def test_x12_every_text_mode_subprocess_call_names_utf8_and_replace(self):
        for name in ("shared/managed_runtimes.py", "shared/managed_pnl.py"):
            tree = ast.parse((REPO / name).read_text(encoding="utf-8"))
            calls = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "run":
                    keywords = {k.arg: k.value for k in node.keywords}
                    if "text" in keywords:
                        calls += 1
                        self.assertEqual(getattr(keywords.get("encoding"), "value", None), "utf-8", f"{name}:{node.lineno}")
                        self.assertEqual(getattr(keywords.get("errors"), "value", None), "replace", f"{name}:{node.lineno}")
            self.assertGreater(calls, 0, name)

    def test_x12_all_changed_modules_still_parse_with_python_3_9(self):
        for name in ("shared/run_heartbeat.py", "shared/line_stream.py", "shared/process_group.py",
                     "shared/cache_lock.py", "shared/cohort_schedule.py", "shared/managed_runtimes.py",
                     "shared/managed_pnl.py", "shared/run_subject_pipeline_cli.py", "shared/brain_mask.py",
                     "shared/pipeline.py", "preprocessing/pipeline.py", "anatomy/surface/backend_bridge.py"):
            ast.parse((REPO / name).read_text(encoding="utf-8"), feature_version=(3, 9))


if __name__ == "__main__":
    unittest.main()
