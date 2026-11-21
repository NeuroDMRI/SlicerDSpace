"""Finding I1/B1: a reinstall over an installed SlicerDSpace must succeed (Python installer and website JS copy).

The fake follows the Slicer start-up facts read from Slicer v5.10.0 and v5.12.4:
S1 start-up first uninstalls the extensions scheduled for removal, then marks every installed extension as loaded;
S3 uninstalling a loaded extension is refused; S4 installing an installed extension returns False.
"""

from __future__ import annotations

import os
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from test_b4_w4_platform import _InstallerCopies


class FakeSlicerState:
    """The extension state on disk and in the settings file, shared by every Slicer start."""

    def __init__(self, installed: bool = False, scheduled: bool = False, locked: bool = False,
                 cancel_works: bool = True, schedule_after_install: bool = False) -> None:
        self.installed = installed
        self.scheduled = scheduled
        self.locked = locked  # a Windows file lock: the start-up removal cannot delete the files
        self.cancel_works = cancel_works
        self.schedule_after_install = schedule_after_install
        self.starts = 0
        self.cancelled = []
        self.history = []

    def start(self) -> "FakeEm":
        self.starts += 1
        self.history.append((self.installed, self.scheduled))
        if self.scheduled and not self.locked:  # S1: uninstallScheduledExtensions runs before aboutToLoadExtensions
            self.installed = False
            self.scheduled = False
        return FakeEm(self)


class FakeEm:
    slicerRevision = "34045"
    slicerOs = "linux"
    slicerArch = "amd64"

    def __init__(self, state: FakeSlicerState) -> None:
        self.state = state
        self.loaded = state.installed  # S2: everything installed at start-up is marked loaded

    def setInteractive(self, value): pass
    def isExtensionInstalled(self, name): return self.state.installed
    def scheduleExtensionForUninstall(self, name): self.state.scheduled = True
    def uninstallScheduledExtensions(self):
        if self.state.scheduled and not self.loaded and not self.state.locked:  # S3
            self.state.installed = False
            self.state.scheduled = False
    def updateModel(self): pass
    def cancelExtensionScheduledForUninstall(self, name):
        self.state.cancelled.append(name)
        if self.state.cancel_works:
            self.state.scheduled = False
    def isExtensionScheduledForUninstall(self, name): return self.state.scheduled
    def installExtension(self, name, meta, archive, a, b):
        if self.state.installed:  # S4
            return False
        self.state.installed = True
        if self.state.schedule_after_install:
            self.state.scheduled = True
        return True
    def extensionModulePaths(self, name): return []
    def extensionsInstallPath(self): return ""


def _run_installer_function(ns: dict, state: FakeSlicerState, tmp: Path, slicer_arg: str = "/fake/Slicer", expected_exe: str = None):
    """Run the installer against the fake; each helper launch is a new Slicer start. Return (calls, logs, error)."""
    name = "_install_extension_in_slicer" if "_install_extension_in_slicer" in ns else "_install_extension"
    logs: list = []
    ns["_log"] = logs.append
    calls: list = []
    real_run = subprocess.run
    qt_stub = types.ModuleType("qt")
    qt_stub.QTimer = types.SimpleNamespace(singleShot=lambda ms, fn: fn())

    # The installer launches os.path.abspath(slicer_exe); on Windows that adds a drive letter to /fake/Slicer.
    expected_exe = expected_exe or os.path.abspath(slicer_arg)

    def fake_run(cmd, *args, **kwargs):
        # Finding P2-1: on Linux without DISPLAY the installer prefixes the helper command with [xvfb-run, -a].
        # Strip that prefix before matching, and record it, so the prefix stays tested.
        orig_cmd = cmd
        xvfb_prefix: list = []
        if cmd and len(cmd) > 2 and os.path.basename(cmd[0]) == "xvfb-run" and cmd[1] == "-a":
            xvfb_prefix, cmd = list(cmd[:2]), list(cmd[2:])
        if not cmd or cmd[0] not in (slicer_arg, expected_exe):
            return real_run(orig_cmd, *args, **kwargs)
        if cmd[0] != expected_exe:
            # POSIX subprocess resolves a relative executable against the cwd of the child, not of the caller.
            raise FileNotFoundError(cmd[0])
        seen: dict = {}
        em = state.start()
        slicer_stub = types.ModuleType("slicer")
        slicer_stub.app = types.SimpleNamespace(
            extensionsManagerModel=lambda: em, majorVersion=5, minorVersion=10, exit=lambda code: seen.update(code=code)
        )
        script = cmd.index("--python-script") + 1
        helper, script_args = cmd[script], cmd[script + 1:]
        calls.append({"cmd": list(cmd), "cwd": kwargs.get("cwd"), "args": script_args, "xvfb_prefix": xvfb_prefix})
        saved_argv = sys.argv
        sys.argv = [helper, *script_args]
        try:
            with patch.dict(sys.modules, {"qt": qt_stub, "slicer": slicer_stub}):
                exec(compile(Path(helper).read_text(encoding="utf-8"), helper, "exec"), {"__name__": "helper"})
        finally:
            sys.argv = saved_argv
        return subprocess.CompletedProcess(orig_cmd, seen.get("code", 1))

    archive = tmp / "ext.zip"
    archive.write_bytes(b"PK")
    error = None
    with patch.object(subprocess, "run", fake_run):
        try:
            ns[name](slicer_arg, archive)
        except RuntimeError as exc:
            error = str(exc)
    return calls, logs, error


class I1ReinstallTests(_InstallerCopies, unittest.TestCase):
    def _each(self, **state_args):
        import tempfile

        for label, ns in self.copies():
            with self.subTest(label), tempfile.TemporaryDirectory() as tmp:
                state = FakeSlicerState(**state_args)
                yield state, _run_installer_function(ns, state, Path(tmp))

    def test_b1_reinstall_over_an_installed_copy_succeeds_with_two_helper_starts(self) -> None:
        for state, (calls, logs, error) in self._each(installed=True):
            self.assertIsNone(error)
            self.assertEqual(state.starts, 2)
            self.assertEqual([c["args"][-1] for c in calls], ["1", "2"])
            self.assertTrue(state.installed)
            self.assertFalse(state.scheduled)
            self.assertEqual(state.cancelled, [])
            self.assertTrue(any("second time" in line for line in logs), logs)
            self.assertFalse(any("[WARN]" in line and "scheduled for removal" in line for line in logs), logs)

    def test_b1_fresh_install_starts_slicer_once(self) -> None:
        for state, (calls, logs, error) in self._each(installed=False):
            self.assertIsNone(error)
            self.assertEqual(state.starts, 1)
            self.assertEqual(calls[0]["args"][-1], "1")
            self.assertTrue(state.installed)
            self.assertEqual(state.cancelled, [])
            self.assertFalse(any("scheduled for removal" in line for line in logs))

    def test_b1_pass_one_only_schedules_the_removal(self) -> None:
        for state, (calls, logs, error) in self._each(installed=True):
            self.assertIsNone(error)
            # State seen by each Slicer start (before its start-up uninstall): pass 2 finds the old copy still
            # installed and scheduled, so pass 1 did nothing but schedule it.
            self.assertEqual(state.history, [(True, False), (True, True)])

    def test_b1_windows_lock_pass_two_cancels_the_schedule_and_reports_the_clear_error(self) -> None:
        for state, (calls, logs, error) in self._each(installed=True, locked=True):
            self.assertIsNotNone(error)
            self.assertIn("Close all 3D Slicer windows and run the installer again", error)
            self.assertNotIn("could not cancel", error)
            self.assertEqual(state.starts, 2)
            self.assertEqual(state.cancelled, ["SlicerDSpace"])
            self.assertFalse(state.scheduled)
            self.assertTrue(state.installed)

    def test_b1_lock_with_a_failing_cancel_says_so(self) -> None:
        for state, (calls, logs, error) in self._each(installed=True, locked=True, cancel_works=False):
            self.assertIn("could not cancel the removal schedule", error)
            self.assertTrue(state.scheduled)

    def test_i1_schedule_left_after_install_is_cancelled_and_warned(self) -> None:
        for state, (calls, logs, error) in self._each(installed=False, schedule_after_install=True):
            self.assertIsNone(error)
            self.assertEqual(state.cancelled, ["SlicerDSpace"])
            self.assertFalse(state.scheduled)
            self.assertTrue(any("[WARN]" in line and "scheduled for removal" in line and "cancelled that schedule" in line
                                for line in logs), logs)

    def test_p2_2_a_cancel_that_did_not_work_is_not_reported_as_cancelled(self) -> None:
        for state, (calls, logs, error) in self._each(installed=False, schedule_after_install=True, cancel_works=False):
            self.assertIsNone(error)
            self.assertTrue(state.scheduled)
            self.assertFalse(any("cancelled that schedule" in line for line in logs), logs)
            self.assertTrue(any("could not cancel that schedule" in line for line in logs), logs)

    def test_i1_no_warning_when_nothing_was_scheduled(self) -> None:
        for state, (calls, logs, error) in self._each(installed=False):
            self.assertIsNone(error)
            self.assertEqual(state.cancelled, [])
            self.assertFalse(any("scheduled for removal" in line for line in logs))

    def test_o_p2_1_helper_runs_in_its_own_temp_folder_not_in_the_caller_cwd(self) -> None:
        for state, (calls, logs, error) in self._each(installed=True):
            self.assertIsNone(error)
            self.assertEqual(len(calls), 2)
            for call in calls:
                self.assertIsNotNone(call["cwd"])
                receipt = Path(call["args"][1])
                self.assertEqual(Path(call["cwd"]), receipt.parent)
                self.assertTrue(receipt.name == "receipt.json")

    def test_r1_a_relative_slicer_path_is_made_absolute_before_the_cwd_changes(self) -> None:
        import os
        import tempfile

        for label, ns in self.copies():
            with self.subTest(label), tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as user_cwd:
                user_cwd = os.path.realpath(user_cwd)
                state = FakeSlicerState(installed=False)
                old = os.getcwd()
                os.chdir(user_cwd)  # the user's folder: the relative path below is valid only here
                try:
                    calls, logs, error = _run_installer_function(
                        ns, state, Path(tmp), slicer_arg="Slicer-5.10.0-linux-amd64/Slicer",
                        expected_exe=os.path.join(user_cwd, "Slicer-5.10.0-linux-amd64", "Slicer"))
                finally:
                    os.chdir(old)
                self.assertIsNone(error)
                self.assertEqual(len(calls), 1)
                for call in calls:
                    self.assertTrue(os.path.isabs(call["cmd"][0]), call["cmd"][0])
                    self.assertNotEqual(Path(call["cwd"]), Path(user_cwd))

    def test_p2_1_on_linux_without_display_the_helper_runs_under_xvfb_run_and_the_fake_accepts_it(self) -> None:
        import shutil
        import tempfile

        # Simulate the Linux CI runner on any host: platform linux, DISPLAY unset, xvfb-run on PATH.
        real_which = shutil.which
        fake_which = lambda cmd, *a, **k: "/usr/bin/xvfb-run" if cmd == "xvfb-run" else real_which(cmd, *a, **k)
        for label, ns in self.copies():
            with self.subTest(label), tempfile.TemporaryDirectory() as tmp:
                state = FakeSlicerState(installed=True)
                env_without_display = {k: v for k, v in os.environ.items() if k != "DISPLAY"}
                with patch.object(sys, "platform", "linux"), patch.dict(os.environ, env_without_display, clear=True), \
                        patch.object(shutil, "which", fake_which):
                    calls, logs, error = _run_installer_function(ns, state, Path(tmp))
                self.assertIsNone(error)
                self.assertEqual(state.starts, 2)
                self.assertEqual(len(calls), 2)
                for call in calls:
                    self.assertEqual(call["xvfb_prefix"], ["/usr/bin/xvfb-run", "-a"])
                    self.assertEqual(call["cmd"][0], os.path.abspath("/fake/Slicer"))


if __name__ == "__main__":
    unittest.main()
