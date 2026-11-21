"""Finding GAP: the graphical Setup extension runs inside a live 3D Slicer, where an installed SlicerDSpace is loaded.

Slicer v5.10.0 and v5.12.4: S2 an installed extension is marked loaded at start-up; S3 uninstalling a loaded extension is
refused and the schedule stays; S4 installing an installed extension returns False. The fake in
test_installer_reinstall_i1 follows these facts.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from test_installer_reinstall_i1 import FakeSlicerState

REPO_ROOT = Path(__file__).resolve().parents[2]
JS_PATH = REPO_ROOT / "docs" / "site" / "assets" / "js" / "installer-customizer.js"


class TwoExtensionSlicer:
    """Setup and SlicerDSpace together; the S1-S4 facts per extension name. A finished Setup scheduled itself for removal."""

    def __init__(self, scheduled=("SlicerDSpaceSetup",), cancel_works: bool = True) -> None:
        self.installed = {"SlicerDSpace", "SlicerDSpaceSetup"}
        self.scheduled = set(scheduled)
        self.cancel_works = cancel_works
        self.cancel_calls = []

    def start(self):
        self.installed -= self.scheduled  # S1: the start-up removal runs first
        self.scheduled = set()
        return self

    def isExtensionInstalled(self, name): return name in self.installed
    def isExtensionScheduledForUninstall(self, name): return name in self.scheduled
    def scheduleExtensionForUninstall(self, name): self.scheduled.add(name)

    def cancelExtensionScheduledForUninstall(self, name):
        self.cancel_calls.append(name)
        if self.cancel_works:
            self.scheduled.discard(name)


class _Loose(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return MagicMock(name=name)


def setup_module_source(state_js: str = "{ os: 'linux', wmaEnabled: false }") -> str:
    node = shutil.which("node")
    if node is None:
        raise unittest.SkipTest("node is not installed in PATH")
    code = ("const ic = require(process.argv[1]);"
            f"process.stdout.write(ic.buildSetupExtensionFiles({state_js}).find((f) => f.name.endsWith('SlicerDSpaceSetup.py')).data);")
    return subprocess.run([node, "-e", code, str(JS_PATH)], capture_output=True, text=True, encoding="utf-8", errors="strict", check=True).stdout


def load_setup_module(minor: int = 10, source: str = None) -> dict:
    """Exec the graphical Setup module against stub qt and slicer modules; return its namespace."""
    source = source or setup_module_source()
    slm = types.ModuleType("slicer.ScriptedLoadableModule")
    slm.ScriptedLoadableModule = type("ScriptedLoadableModule", (), {"__init__": lambda self, parent: None})
    slm.ScriptedLoadableModuleWidget = type("ScriptedLoadableModuleWidget", (), {"__init__": lambda self, parent=None: None})
    slicer_stub = _Loose("slicer")
    slicer_stub.ScriptedLoadableModule = slm
    slicer_stub.app = types.SimpleNamespace(majorVersion=5, minorVersion=minor)
    namespace: dict = {"__name__": "setup_under_test", "__file__": "SlicerDSpaceSetup.py"}
    with patch.dict(sys.modules, {"qt": _Loose("qt"), "slicer": slicer_stub, "slicer.ScriptedLoadableModule": slm}):
        exec(compile(source, "SlicerDSpaceSetup.py", "exec"), namespace)
    namespace["_slicer_stub"] = slicer_stub
    return namespace


class GapGuiReinstallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = setup_module_source()

    def _ns(self, minor: int = 10) -> dict:
        return load_setup_module(minor, self.source)

    def test_gap_install_over_a_loaded_copy_cancels_the_schedule_and_raises_a_clear_error(self) -> None:
        ns = self._ns()
        state = FakeSlicerState(installed=True)
        em = state.start()  # the running Slicer window: the old copy is loaded
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError) as caught:
                ns["_install_extension_gui"](em, Path(tmp) / "ext.zip", tmp, lambda _m: None)
        self.assertIn("Restart 3D Slicer", str(caught.exception))
        self.assertFalse(state.scheduled, "a failed install must not leave SlicerDSpace scheduled for removal")
        self.assertTrue(state.installed)
        self.assertEqual(state.cancelled, ["SlicerDSpace"])

    def test_gap_failed_cancel_is_reported_in_the_error(self) -> None:
        ns = self._ns()
        state = FakeSlicerState(installed=True, cancel_works=False)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError) as caught:
                ns["_install_extension_gui"](state.start(), Path(tmp) / "ext.zip", tmp, lambda _m: None)
        self.assertIn("could not be cancelled", str(caught.exception))

    def test_gap_installed_copy_is_removed_at_restart_and_the_new_copy_installs_after_it(self) -> None:
        for minor in (10, 12):  # 10: the archive is installed as it is; 12: the stub path for another Slicer build
            with self.subTest(minor=minor), tempfile.TemporaryDirectory() as tmp:
                ns = self._ns(minor)
                state = FakeSlicerState(installed=True)
                logs, restarts = [], []
                note = ns["_handle_installed_copy"](state.start(), lambda: True, lambda: restarts.append(1) or True, logs.append)
                self.assertTrue(note.startswith("Restarting"), note)
                self.assertEqual(restarts, [1])
                self.assertTrue(state.installed and state.scheduled)  # nothing was installed or downloaded in this session
                self.assertTrue(any("scheduled for removal" in line for line in logs), logs)
                em_after_restart = state.start()  # S1: the start-up removes the old copy
                self.assertFalse(state.installed)
                ns["_install_extension_gui"](em_after_restart, Path(tmp) / "ext.zip", tmp, logs.append)
                self.assertTrue(state.installed)
                self.assertFalse(state.scheduled)

    def test_gap_cancel_in_the_dialog_changes_nothing(self) -> None:
        ns = self._ns()
        state = FakeSlicerState(installed=True)
        restarts = []
        note = ns["_handle_installed_copy"](state.start(), lambda: False, lambda: restarts.append(1) or True, lambda _m: None)
        self.assertTrue(note.startswith("Cancelled"), note)
        self.assertEqual((state.installed, state.scheduled, restarts), (True, False, []))

    def test_gap_without_a_restart_function_the_user_is_told_to_restart(self) -> None:
        ns = self._ns()
        state = FakeSlicerState(installed=True)
        note = ns["_handle_installed_copy"](state.start(), lambda: True, lambda: False, lambda _m: None)
        self.assertIn("Restart 3D Slicer yourself", note)
        self.assertTrue(state.scheduled)

    def test_s_setup_reinstall_after_a_completed_setup_cancels_the_setup_self_removal(self) -> None:
        ns = self._ns()
        slicer = TwoExtensionSlicer()
        restarts = []
        note = ns["_handle_installed_copy"](slicer, lambda: True, lambda: restarts.append(1) or True, lambda _m: None)
        self.assertTrue(note.startswith("Restarting"), note)
        self.assertEqual(restarts, [1])
        self.assertEqual(slicer.cancel_calls, ["SlicerDSpaceSetup"])
        self.assertEqual(slicer.scheduled, {"SlicerDSpace"})
        slicer.start()
        self.assertEqual(slicer.installed, {"SlicerDSpaceSetup"}, "Setup must survive the restart to install the replacement")

    def test_s_setup_failed_cancel_schedules_nothing_and_does_not_restart(self) -> None:
        ns = self._ns()
        slicer = TwoExtensionSlicer(cancel_works=False)
        logs, restarts = [], []
        note = ns["_handle_installed_copy"](slicer, lambda: True, lambda: restarts.append(1) or True, logs.append)
        self.assertEqual(restarts, [])
        self.assertNotIn("SlicerDSpace", slicer.scheduled)
        self.assertEqual(slicer.scheduled, {"SlicerDSpaceSetup"})
        self.assertIn("Restart 3D Slicer first", note)
        self.assertIn(note, logs)
        slicer.start()
        self.assertIn("SlicerDSpace", slicer.installed)

    def test_s_setup_without_a_self_schedule_is_unchanged(self) -> None:
        ns = self._ns()
        slicer = TwoExtensionSlicer(scheduled=())
        restarts = []
        note = ns["_handle_installed_copy"](slicer, lambda: True, lambda: restarts.append(1) or True, lambda _m: None)
        self.assertTrue(note.startswith("Restarting"), note)
        self.assertEqual((slicer.cancel_calls, slicer.scheduled, restarts), ([], {"SlicerDSpace"}, [1]))

    def test_gap_nothing_to_handle_when_not_installed(self) -> None:
        ns = self._ns()
        asked = []
        note = ns["_handle_installed_copy"](FakeSlicerState(installed=False).start(), lambda: asked.append(1) or True, lambda: True, lambda _m: None)
        self.assertIsNone(note)
        self.assertEqual(asked, [])

    def test_gap_a_schedule_that_stays_after_install_is_cancelled_or_reported(self) -> None:
        for cancel_works in (True, False):
            with self.subTest(cancel_works=cancel_works), tempfile.TemporaryDirectory() as tmp:
                ns = self._ns()
                state = FakeSlicerState(installed=False, schedule_after_install=True, cancel_works=cancel_works)
                logs = []
                ns["_install_extension_gui"](state.start(), Path(tmp) / "ext.zip", tmp, logs.append)
                self.assertEqual(state.scheduled, not cancel_works)
                self.assertEqual(any("[WARN]" in line and "could not cancel" in line for line in logs), not cancel_works, logs)

    def test_gap_start_gate_runs_before_the_download_and_downloads_ready_uses_the_guarded_install(self) -> None:
        source = self.source
        start = source.index("    def _on_start():")
        body = source[start:source.index("    def _on_cancel():")]
        self.assertLess(body.index('if state["done"]'), body.index("_handle_installed_copy("))
        self.assertLess(body.index("_handle_installed_copy("), body.index('state["running"] = True'))
        self.assertLess(body.index('state["running"] = True'), body.index("_start_worker_process("))
        ready = source[source.index('elif kind == "downloads_ready":'):source.index('elif kind == "all_done":')]
        self.assertIn("_install_extension_gui(", ready)
        self.assertNotIn("uninstallScheduledExtensions", ready)
        self.assertEqual(source.count("uninstallScheduledExtensions()"), 1)


if __name__ == "__main__":
    unittest.main()
