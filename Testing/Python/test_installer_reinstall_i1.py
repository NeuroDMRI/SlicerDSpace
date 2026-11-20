"""Finding I1: a reinstall must not leave SlicerDSpace scheduled for uninstall (Python installer and website JS copy)."""

from __future__ import annotations

import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from test_b4_w4_platform import _InstallerCopies


class _FakeEm:
    def __init__(self, loaded: bool, schedule_after_install: bool = False) -> None:
        self.loaded = loaded
        self.schedule_after_install = schedule_after_install
        self.scheduled = False
        self.cancelled = []
        self.slicerRevision = "34045"
        self.slicerOs = "linux"
        self.slicerArch = "amd64"

    def setInteractive(self, value): pass
    def isExtensionInstalled(self, name): return self.loaded
    def scheduleExtensionForUninstall(self, name): self.scheduled = True
    def uninstallScheduledExtensions(self):
        # Slicer refuses to remove an extension whose modules are loaded; the schedule stays.
        if not self.loaded:
            self.scheduled = False
    def updateModel(self): pass
    def cancelExtensionScheduledForUninstall(self, name):
        self.cancelled.append(name)
        self.scheduled = False
    def isExtensionScheduledForUninstall(self, name): return self.scheduled
    def installExtension(self, name, meta, archive, a, b):
        self.loaded = True
        if self.schedule_after_install:
            self.scheduled = True
        return True
    def extensionModulePaths(self, name): return []
    def extensionsInstallPath(self): return ""


def _run_installer_function(ns: dict, em: _FakeEm, tmp: Path):
    """Run the installer's helper script against a fake extension manager; return (cmd, logs, error)."""
    name = "_install_extension_in_slicer" if "_install_extension_in_slicer" in ns else "_install_extension"
    logs: list = []
    ns["_log"] = logs.append
    seen: dict = {}
    real_run = subprocess.run
    qt_stub = types.ModuleType("qt")
    qt_stub.QTimer = types.SimpleNamespace(singleShot=lambda ms, fn: fn())
    slicer_stub = types.ModuleType("slicer")
    slicer_stub.app = types.SimpleNamespace(
        extensionsManagerModel=lambda: em, majorVersion=5, minorVersion=10, exit=lambda code: seen.update(code=code)
    )

    def fake_run(cmd, *args, **kwargs):
        if not cmd or cmd[0] != "/fake/Slicer":
            return real_run(cmd, *args, **kwargs)
        seen["cmd"] = list(cmd)
        helper = cmd[cmd.index("--python-script") + 1]
        saved_argv = sys.argv
        sys.argv = [helper, cmd[-2], cmd[-1]]
        try:
            with patch.dict(sys.modules, {"qt": qt_stub, "slicer": slicer_stub}):
                exec(compile(Path(helper).read_text(encoding="utf-8"), helper, "exec"), {"__name__": "helper"})
        finally:
            sys.argv = saved_argv
        return subprocess.CompletedProcess(cmd, seen.get("code", 1))

    archive = tmp / "ext.zip"
    archive.write_bytes(b"PK")
    error = None
    with patch.object(subprocess, "run", fake_run):
        try:
            ns[name]("/fake/Slicer", archive)
        except RuntimeError as exc:
            error = str(exc)
    return seen["cmd"], logs, error


class I1ReinstallTests(_InstallerCopies, unittest.TestCase):
    def test_i1_helper_ignores_installed_extension_and_fails_when_it_cannot_be_removed(self) -> None:
        import tempfile

        for label, ns in self.copies():
            with self.subTest(label), tempfile.TemporaryDirectory() as tmp:
                em = _FakeEm(loaded=True)
                cmd, logs, error = _run_installer_function(ns, em, Path(tmp))
                idx = cmd.index("--modules-to-ignore")
                self.assertEqual(cmd[idx + 1], "SlicerDSpace")
                self.assertIn("--no-main-window", cmd)
                self.assertIsNotNone(error)
                self.assertIn("Close all 3D Slicer windows", error)
                self.assertEqual(em.cancelled, ["SlicerDSpace"])
                self.assertFalse(em.scheduled)

    def test_i1_schedule_left_after_install_is_cancelled_and_warned(self) -> None:
        import tempfile

        for label, ns in self.copies():
            with self.subTest(label), tempfile.TemporaryDirectory() as tmp:
                em = _FakeEm(loaded=False, schedule_after_install=True)
                cmd, logs, error = _run_installer_function(ns, em, Path(tmp))
                self.assertIsNone(error)
                self.assertEqual(em.cancelled, ["SlicerDSpace"])
                self.assertFalse(em.scheduled)
                self.assertTrue(any("[WARN]" in line and "scheduled for removal" in line for line in logs), logs)

    def test_i1_no_warning_when_nothing_was_scheduled(self) -> None:
        import tempfile

        for label, ns in self.copies():
            with self.subTest(label), tempfile.TemporaryDirectory() as tmp:
                em = _FakeEm(loaded=False)
                cmd, logs, error = _run_installer_function(ns, em, Path(tmp))
                self.assertIsNone(error)
                self.assertEqual(em.cancelled, [])
                self.assertFalse(any("scheduled for removal" in line for line in logs))


if __name__ == "__main__":
    unittest.main()
