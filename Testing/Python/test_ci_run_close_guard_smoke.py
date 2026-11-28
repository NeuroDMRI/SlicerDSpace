#!/usr/bin/env python3
"""Tests for .github/scripts/ci_run_close_guard_smoke.py and its workflow step (backlog "close guard / Author" row B).

No Slicer, network or GitHub is needed; the workflow YAML is read as text.
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
HELPER = REPO / ".github" / "scripts" / "ci_run_close_guard_smoke.py"
WORKFLOW = REPO / ".github" / "workflows" / "slicer-compat-matrix.yml"


def _load_helper():
    spec = importlib.util.spec_from_file_location("ci_run_close_guard_smoke_under_test", str(HELPER))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BuildCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_helper()

    def test_linux_runs_under_xvfb_with_main_window(self):
        command = self.mod.build_command("/opt/Slicer/Slicer", "linux", xvfb="/usr/bin/xvfb-run")
        self.assertEqual(command[:2], ["/usr/bin/xvfb-run", "-a"])
        self.assertNotIn("--no-main-window", command)
        self.assertEqual(command[-2], "--python-script")
        self.assertTrue(command[-1].endswith("slicer_close_guard_smoke.py"))

    def test_linux_default_xvfb_name(self):
        self.assertEqual(self.mod.build_command("S", "linux")[:2], ["xvfb-run", "-a"])

    def test_darwin_and_windows_have_no_xvfb(self):
        for platform in ("darwin", "win32"):
            with self.subTest(platform=platform):
                command = self.mod.build_command("S", platform, xvfb="/usr/bin/xvfb-run")
                self.assertEqual(command[0], "S")
                self.assertNotIn("-a", command)
                self.assertNotIn("--no-main-window", command)
                for flag in ("--no-splash", "--disable-settings", "--disable-cli-modules"):
                    self.assertIn(flag, command)


class JudgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_helper()

    def _judge(self, payload=None, raw=None):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            if payload is not None:
                path.write_text(json.dumps(payload), encoding="utf-8")
            elif raw is not None:
                path.write_text(raw, encoding="utf-8")
            return self.mod.judge(path)

    def test_missing_report_fails(self):
        ok, text = self._judge()
        self.assertFalse(ok)
        self.assertIn("no report", text)

    def test_unreadable_report_fails(self):
        self.assertFalse(self._judge(raw="{not json")[0])

    def test_passed_false_fails_and_lists_checks_that_are_not_true(self):
        ok, text = self._judge({"passed": False, "checks": {"guard_installed": True, "close_refused": False}})
        self.assertFalse(ok)
        self.assertIn("close_refused", text)
        self.assertNotIn("guard_installed", text)

    def test_passed_missing_or_not_true_fails(self):
        self.assertFalse(self._judge({"checks": {}})[0])
        self.assertFalse(self._judge({"passed": "yes"})[0])

    def test_passed_true_passes(self):
        self.assertTrue(self._judge({"passed": True, "checks": {"guard_installed": True}})[0])


class TimeoutTests(unittest.TestCase):
    """Fix round: Codex P1 (timeout verdict comes from the report) and Opus P2 (kill the whole tree)."""

    @classmethod
    def setUpClass(cls):
        cls.mod = _load_helper()

    def _main(self, platform, report=None):
        """Run main() with a fake Popen whose wait() times out. `report` is written at launch, as the smoke would."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            process = mock.MagicMock()
            process.pid = 4242
            process.wait.side_effect = [subprocess.TimeoutExpired("slicer", 1), -9]

            def fake_popen(command, **kwargs):
                if report is not None:
                    (tmp / "slicer_close_guard_smoke.json").write_text(json.dumps(report), encoding="utf-8")
                return process

            argv = ["--slicer-exe", "S", "--artifacts-dir", str(tmp), "--timeout", "1"]
            with mock.patch.object(sys, "platform", platform), \
                    mock.patch.object(subprocess, "Popen", side_effect=fake_popen) as popen, \
                    mock.patch.object(shutil, "which", return_value=None), \
                    mock.patch.object(subprocess, "run") as run, \
                    mock.patch.object(os, "killpg", create=True) as killpg, \
                    mock.patch.object(os, "getpgid", create=True, return_value=777) as getpgid:
                code = self.mod.main(argv)
        return code, popen, run, killpg, getpgid, process

    def test_codex_p1_timeout_with_passed_report_passes(self):
        code, *_ = self._main("linux", report={"passed": True, "checks": {}})
        self.assertEqual(code, 0)

    def test_codex_p1_timeout_without_report_fails(self):
        code, *_ = self._main("linux", report=None)
        self.assertEqual(code, 1)

    def test_codex_p1_timeout_with_failed_report_fails(self):
        code, *_ = self._main("linux", report={"passed": False, "checks": {"close_refused": False}})
        self.assertEqual(code, 1)

    def test_opus_p2_posix_timeout_kills_the_process_group(self):
        _code, popen, run, killpg, getpgid, process = self._main("linux", report=None)
        self.assertIs(popen.call_args.kwargs.get("start_new_session"), True)
        getpgid.assert_called_once_with(4242)
        killpg.assert_called_once()
        self.assertEqual(killpg.call_args.args[0], 777)
        run.assert_not_called()
        process.kill.assert_not_called()
        self.assertEqual(process.wait.call_count, 2)

    def test_opus_p2_posix_killpg_error_falls_back_to_kill(self):
        with mock.patch.object(os, "killpg", create=True, side_effect=OSError("gone")), \
                mock.patch.object(os, "getpgid", create=True, return_value=1):
            process = mock.MagicMock()
            process.pid = 5
            self.mod.kill_tree(process, "linux")
        process.kill.assert_called_once()

    def test_opus_p2_windows_timeout_runs_taskkill_tree(self):
        _code, popen, run, killpg, _getpgid, process = self._main("win32", report=None)
        self.assertNotIn("start_new_session", popen.call_args.kwargs)
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], ["taskkill", "/T", "/F", "/PID", "4242"])
        killpg.assert_not_called()
        process.kill.assert_not_called()

    def test_opus_p2_windows_taskkill_error_falls_back_to_kill(self):
        with mock.patch.object(subprocess, "run", side_effect=OSError("no taskkill")):
            process = mock.MagicMock()
            process.pid = 5
            self.mod.kill_tree(process, "win32")
        process.kill.assert_called_once()


class WorkflowStepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        text = WORKFLOW.read_text(encoding="utf-8")
        cls.repo_only = cls._job_block(text, "repo-only")
        cls.full_install = cls._job_block(text, "full-install")

    @staticmethod
    def _job_block(text, name):
        match = re.search(r"^  %s:\n(.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)" % re.escape(name), text, re.S | re.M)
        assert match, name
        return match.group(1)

    @staticmethod
    def _steps(block):
        return re.split(r"^      - ", block, flags=re.M)[1:]

    def test_repo_only_job_requires_the_smoke_except_on_windows(self):
        steps = [s for s in self._steps(self.repo_only) if "ci_run_close_guard_smoke.py" in s]
        self.assertEqual(len(steps), 1)
        step = steps[0]
        # Run 46121498394: PASS on the 4 macOS/Linux jobs; Windows has no OpenGL 3.2 and crashes
        # before the smoke runs. So the step fails the job on macOS/Linux and is report only on Windows.
        lines = re.findall(r"(?m)^        continue-on-error:.*$", step)
        self.assertEqual(lines, ["        continue-on-error: ${{ runner.os == 'Windows' }}"])
        self.assertRegex(step, r"(?m)^        timeout-minutes: 10$")
        self.assertIn("if: ${{ !cancelled() }}", step)
        self.assertIn("SLICER_EXE: ${{ steps.slicer.outputs.slicer_exe }}", step)
        self.assertIn("$RUNNER_TEMP/ci-artifacts", step)

    def test_step_sits_between_the_smoke_and_the_unit_test_install(self):
        names = re.findall(r"(?m)^      - name: (.*)$", self.repo_only)
        smoke = names.index("Run the headless smoke through real Slicer")
        install = names.index("Install the unit-test dependencies for PythonSlicer (own directory)")
        guard = [i for i, n in enumerate(names) if "close guard smoke" in n]
        self.assertEqual(len(guard), 1)
        self.assertTrue(smoke < guard[0] < install)

    def test_full_install_job_does_not_run_it(self):
        self.assertNotIn("ci_run_close_guard_smoke.py", self.full_install)


if __name__ == "__main__":
    unittest.main()
