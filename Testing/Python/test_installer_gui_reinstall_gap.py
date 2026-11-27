"""Finding GAP: the graphical Setup extension runs inside a live 3D Slicer, where an installed SlicerDSpace is loaded.

Slicer v5.10.0 and v5.12.4: S2 an installed extension is marked loaded at start-up; S3 uninstalling a loaded extension is
refused and the schedule stays; S4 installing an installed extension returns False. The fake in
test_installer_reinstall_i1 follows these facts.
"""

from __future__ import annotations

import ast
import queue
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
    """Setup and SlicerDSpace together; the S1-S4 facts per extension name. A finished Setup scheduled itself for removal.

    `loaded` holds the extensions that were installed at the last start() (S2). uninstallScheduledExtensions follows
    qSlicerExtensionsManagerModel::uninstallScheduledExtensions of Slicer v5.10.0 and v5.12.4: it loops over every scheduled
    extension, skips a loaded one (it stays scheduled, S3) and goes on with the next. `stop_at_first_refusal` models a Slicer
    that stops the loop at the first refusal; no real Slicer does that, a test sets it to prove that the loop matters.
    """

    def __init__(self, scheduled=("SlicerDSpaceSetup",), cancel_works: bool = True,
                 installed=("SlicerDSpace", "SlicerDSpaceSetup"), stop_at_first_refusal: bool = False) -> None:
        self.installed = set(installed)
        self.loaded = set(self.installed)  # the window opened with these extensions installed
        self.scheduled = set(scheduled)
        self._schedule_order = list(scheduled)  # the loop visits the extensions in this order
        self.cancel_works = cancel_works
        self.stop_at_first_refusal = stop_at_first_refusal
        self.cancel_calls = []
        self.uninstall_runs = []  # per uninstallScheduledExtensions call: (removed, refused)

    slicerRevision = "34045"
    slicerOs = "linux"
    slicerArch = "amd64"

    def start(self):
        self.installed -= self.scheduled  # S1: the start-up removal runs first
        self.scheduled = set()
        self._schedule_order = []
        self.loaded = set(self.installed)  # S2
        return self

    def isExtensionInstalled(self, name): return name in self.installed
    def isExtensionScheduledForUninstall(self, name): return name in self.scheduled

    def scheduleExtensionForUninstall(self, name):
        if name not in self.scheduled:
            self._schedule_order.append(name)
        self.scheduled.add(name)

    def cancelExtensionScheduledForUninstall(self, name):
        self.cancel_calls.append(name)
        if self.cancel_works:
            self.scheduled.discard(name)
            if name in self._schedule_order:
                self._schedule_order.remove(name)

    def uninstallScheduledExtensions(self):
        removed, refused = [], []
        for name in list(self._schedule_order):
            if name in self.loaded:  # S3: refused, the schedule stays
                refused.append(name)
                if self.stop_at_first_refusal:
                    break
                continue
            self.installed.discard(name)
            self.scheduled.discard(name)
            self._schedule_order.remove(name)
            removed.append(name)
        self.uninstall_runs.append((removed, refused))

    def setInteractive(self, value): pass
    def updateModel(self): pass

    def installExtension(self, name, meta, archive, a, b):
        if name in self.installed:  # S4
            return False
        self.installed.add(name)
        return True


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
    namespace["__setup_source__"] = source
    return namespace


def run_poll_handler(ns: dict, em, kind: str, payload: dict, state: dict = None, install=None) -> dict:
    """Run the real `_poll` handler of the dialog (cut out of the Setup module source) for one queued item.

    The handler is a closure inside the dialog builder; its closure names become module globals of a copy of the module
    namespace. `_SESSION_INSTALLED` stays the module's own list (the copy is shallow). Returns the stubs for asserts.
    """
    tree = ast.parse(ns["__setup_source__"])
    poll_def = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_poll")
    state = state if state is not None else {"running": True, "done": False}
    state["queue"] = queue.Queue()
    state["queue"].put((kind, payload))
    stubs = {name: MagicMock(name=name) for name in ("pbar", "status_lbl", "poll_timer", "start_btn", "cancel_btn", "_append_log",
                                                     "_set_controls_enabled", "_dequarantine", "_run_phase_c", "qt")}
    stubs["_resolve_mod_root"] = lambda em_: Path("/fake/ext/lib/Slicer-5.10/qt-scripted-modules/mod")
    stubs["_start_phase_c_worker"] = MagicMock(name="_start_phase_c_worker", return_value=True)
    stubs["slicer"] = types.SimpleNamespace(app=types.SimpleNamespace(extensionsManagerModel=lambda: em, majorVersion=5, minorVersion=10))
    stubs["state"] = state
    if install is not None:
        stubs["_install_extension_gui"] = install
    globals_ = dict(ns)
    globals_.update(stubs)
    exec(compile(ast.Module(body=[poll_def], type_ignores=[]), "SlicerDSpaceSetup.py", "exec"), globals_)
    globals_["_poll"]()
    stubs["state"] = state
    return stubs


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
        self.assertIn("Restart 3D Slicer (this removes Setup)", note)
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
        self.assertLess(body.index('if state["done"]'), body.index("_installed_copy_note("))
        self.assertLess(body.index("_installed_copy_note("), body.index('state["running"] = True'))
        self.assertLess(body.index('state["running"] = True'), body.index("_start_worker_process("))
        ready = source[source.index('elif kind == "downloads_ready":'):source.index('elif kind == "all_done":')]
        self.assertIn("_install_extension_gui(", ready)
        self.assertNotIn("uninstallScheduledExtensions", ready)
        self.assertEqual(source.count("uninstallScheduledExtensions()"), 1)

    def test_opus_r2_p2_c_installed_copy_note_is_skipped_when_this_session_installed_the_copy(self) -> None:
        ns = self._ns()
        asked, restarts = [], []
        ask = lambda: asked.append(1) or True
        restart = lambda: restarts.append(1) or True
        state = FakeSlicerState(installed=True)
        em = state.start()
        note = ns["_installed_copy_note"]({"installed_here": True}, em, ask, restart, lambda _m: None)
        self.assertIsNone(note)
        self.assertEqual((asked, restarts, state.scheduled), ([], [], False))
        note = ns["_installed_copy_note"]({}, em, ask, restart, lambda _m: None)
        self.assertTrue(note.startswith("Restarting"), note)
        self.assertEqual((asked, restarts, state.scheduled), ([1], [1], True))

    def test_reinstall_fix_p2_a_the_installed_here_flag_lasts_across_setup_dialogs_of_one_session(self) -> None:
        # Backlog "Opus (reinstall fix) P2-1": the module loads once per 3D Slicer session, the state dict once per dialog.
        ns = self._ns()
        asked, restarts = [], []
        ask = lambda: asked.append(1) or True
        restart = lambda: restarts.append(1) or True
        state = FakeSlicerState(installed=True)
        em = state.start()
        ns["_SESSION_INSTALLED"][0] = True  # what downloads_ready does in the first dialog, next to state["installed_here"]
        second_dialog_state = {}
        note = ns["_installed_copy_note"](second_dialog_state, em, ask, restart, lambda _m: None)
        self.assertIsNone(note)
        self.assertEqual((asked, restarts, state.scheduled), ([], [], False))

    def test_opus_r2_p2_c_on_start_uses_the_note_and_downloads_ready_records_the_install(self) -> None:
        source = self.source
        start = source.index("    def _on_start():")
        body = source[start:source.index("    def _on_cancel():")]
        self.assertIn("_installed_copy_note(state, slicer.app.extensionsManagerModel(),", body)
        self.assertNotIn("_handle_installed_copy(", body)
        ready = source[source.index('elif kind == "downloads_ready":'):source.index('elif kind == "all_done":')]
        self.assertLess(ready.index("_install_extension_gui("), ready.index('state["installed_here"] = True'))
        self.assertLess(ready.index('state["installed_here"] = True'), ready.index("_resolve_mod_root("))

    def test_opus_r2_p2_c_a_second_install_in_the_same_session_replaces_the_copy_installed_here(self) -> None:
        ns = self._ns()
        state = FakeSlicerState(installed=False)
        em = state.start()  # the running window had no copy at its start, so the copy installed below is not loaded
        with tempfile.TemporaryDirectory() as tmp:
            ns["_install_extension_gui"](em, Path(tmp) / "ext.zip", tmp, lambda _m: None)
            self.assertTrue(state.installed)
            ns["_install_extension_gui"](em, Path(tmp) / "ext.zip", tmp, lambda _m: None)  # the retry: no error
        self.assertTrue(state.installed)
        self.assertFalse(state.scheduled)

    def test_reinstall_p2_second_round_p2_1_downloads_ready_sets_the_session_flag_only_after_a_good_install(self) -> None:
        # Backlog "Opus (reinstall P2 fix) P2-1": the real downloads_ready branch of the dialog's _poll handler runs here.
        payload = {"work_dir": "/fake/work", "ext_archive": "/fake/work/ext.zip"}
        ns = self._ns()
        self.assertFalse(ns["_SESSION_INSTALLED"][0])
        state = FakeSlicerState(installed=False)
        stubs = run_poll_handler(ns, state.start(), "downloads_ready", dict(payload), install=None)
        self.assertTrue(state.installed, "the real _install_extension_gui ran against the fake")
        self.assertTrue(stubs["state"].get("installed_here"))
        self.assertTrue(ns["_SESSION_INSTALLED"][0])
        # A new dialog of the same session (a fresh state dict) gets no restart prompt.
        asked, restarts = [], []
        note = ns["_installed_copy_note"]({}, state.start(), lambda: asked.append(1) or True, lambda: restarts.append(1) or True, lambda _m: None)
        self.assertIsNone(note)
        self.assertEqual((asked, restarts), ([], []))

        # The install raises: no flag in the module, none in the state.
        ns = self._ns()
        failing = FakeSlicerState(installed=True)  # a loaded copy: _install_extension_gui raises RuntimeError
        stubs = run_poll_handler(ns, failing.start(), "downloads_ready", dict(payload))
        self.assertFalse(ns["_SESSION_INSTALLED"][0])
        self.assertFalse(stubs["state"].get("installed_here"))
        self.assertIn("Install failed", stubs["status_lbl"].setText.call_args[0][0])
        self.assertFalse(failing.scheduled)

    def test_reinstall_p2_second_round_p2_2_reopening_setup_after_all_done_replaces_the_unloaded_copy(self) -> None:
        # Backlog "Opus (reinstall P2 fix) P2-2": Slicer started with Setup loaded and SlicerDSpace not installed. This
        # session installs SlicerDSpace; all_done schedules Setup for removal; the user opens Setup again and starts again.
        with tempfile.TemporaryDirectory() as tmp:
            ns = self._ns()
            slicer = TwoExtensionSlicer(scheduled=(), installed=("SlicerDSpaceSetup",)).start()
            self.assertEqual(slicer.loaded, {"SlicerDSpaceSetup"})
            payload = {"work_dir": tmp, "ext_archive": str(Path(tmp) / "ext.zip")}
            run_poll_handler(ns, slicer, "downloads_ready", dict(payload))  # first dialog: the real guarded install
            self.assertEqual(slicer.installed, {"SlicerDSpace", "SlicerDSpaceSetup"})
            self.assertTrue(ns["_SESSION_INSTALLED"][0])
            run_poll_handler(ns, slicer, "all_done", {"mod_root": "/fake", "warnings": []})
            self.assertEqual(slicer.scheduled, {"SlicerDSpaceSetup"})  # Setup removes itself at the next start
            # The second dialog of the same session: a fresh state dict, a Setup that is scheduled for removal.
            asked, restarts, logs = [], [], []
            note = ns["_installed_copy_note"]({}, slicer, lambda: asked.append(1) or True, lambda: restarts.append(1) or True, logs.append)
            self.assertIsNone(note)
            self.assertEqual((asked, restarts, slicer.cancel_calls), ([], [], []))
            ns["_install_extension_gui"](slicer, Path(tmp) / "ext.zip", tmp, logs.append)  # no RuntimeError
            self.assertEqual(slicer.uninstall_runs[-1], (["SlicerDSpace"], ["SlicerDSpaceSetup"]))
            self.assertEqual(slicer.installed, {"SlicerDSpace", "SlicerDSpaceSetup"})
            self.assertNotIn("SlicerDSpace", slicer.scheduled)
            self.assertEqual(slicer.scheduled, {"SlicerDSpaceSetup"})
            self.assertFalse([line for line in logs if "[WARN]" in line], logs)
            after_restart = slicer.start()  # Setup removes itself at the next start; SlicerDSpace stays
            self.assertEqual(after_restart.installed, {"SlicerDSpace"})

    def test_opus_r3_p2_setup_that_cannot_cancel_its_own_removal_says_to_restart_and_install_setup_again(self) -> None:
        ns = self._ns()
        slicer = TwoExtensionSlicer(cancel_works=False)
        logs, restarts = [], []
        note = ns["_handle_installed_copy"](slicer, lambda: True, lambda: restarts.append(1) or True, logs.append)
        self.assertEqual(restarts, [])
        self.assertIn("this removes Setup", note)
        self.assertIn("install SlicerDSpaceSetup again", note)
        self.assertNotIn("open Setup again", note)
        self.assertIn(note, logs)


if __name__ == "__main__":
    unittest.main()
