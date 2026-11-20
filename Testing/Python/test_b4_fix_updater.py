"""Batch 4 fix round, group updater: U1 (early stdlib-only recovery), U2 (one updater at a time), P-h (Slicer pick)."""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from Testing.Python.test_b4_w4_platform import _InstallerCopies, _updater_fixture

REPO_ROOT = Path(__file__).resolve().parents[2]
_CHILD_ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}


def _dead_pid() -> int:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def _tree_bytes(root: Path) -> dict:
    """Every file under root except the empty lock file the updater creates and keeps."""

    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*"))
            if p.is_file() and p.name != ".slicerdspace_update.lock"}


def _broken_install(root: Path, journal: dict) -> Path:
    """A module root whose new shared/__init__.py imports a module the killed worker never wrote."""

    (root / "shared" / "__pycache__").mkdir(parents=True)
    shutil.copy2(REPO_ROOT / "SlicerDSpace.py", root / "SlicerDSpace.py")
    source = REPO_ROOT / "shared" / "update_journal.py"
    if source.is_file():
        shutil.copy2(source, root / "shared" / "update_journal.py")
    (root / "shared" / "__init__.py").write_text("from .not_written_yet import VALUE\n", encoding="utf-8")
    (root / "shared" / "new_mod.py").write_text("NEW = 1\n", encoding="utf-8")
    (root / "shared" / "__pycache__" / "__init__.cpython-312.pyc").write_bytes(b"stale")
    journal_dir = root / ".slicerdspace_update_journal"
    (journal_dir / "backup" / "shared").mkdir(parents=True)
    (journal_dir / "backup" / "shared" / "__init__.py").write_text("VALUE = 'old'\n", encoding="utf-8")
    (journal_dir / "journal.json").write_text(json.dumps(journal), encoding="utf-8")
    return root


def _journal(pid: int, **extra) -> dict:
    return {"kind": "slicerdspace_update_journal", "version": 1, "target_version": "9", "pid": pid,
            "backed_up": ["shared/__init__.py"], "created": ["shared/new_mod.py"], **extra}


# ---------------------------------------------------------------- U1
class U1EarlyRecoveryTests(unittest.TestCase):
    def test_U1_update_journal_loads_by_path_with_a_broken_shared_package(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = _broken_install(Path(td), _journal(_dead_pid()))
            probe = (
                "import importlib.util, sys\n"
                "before = set(sys.modules)\n"
                "spec = importlib.util.spec_from_file_location('_probe_update_journal', sys.argv[1])\n"
                "module = importlib.util.module_from_spec(spec)\n"
                "spec.loader.exec_module(module)\n"
                "new = sorted(m.split('.')[0] for m in set(sys.modules) - before)\n"
                "bad = [m for m in new if m not in sys.stdlib_module_names and m not in ('psutil', '_probe_update_journal')]\n"
                "print('NONSTDLIB', bad)\n"
                "print('HAS_RECOVER', callable(getattr(module, 'recover_interrupted_update', None)))\n"
            )
            run = subprocess.run([sys.executable, "-c", probe, str(root / "shared" / "update_journal.py")],
                                 capture_output=True, text=True, cwd=str(root), env=_CHILD_ENV, timeout=120)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("NONSTDLIB []", run.stdout)
            self.assertIn("HAS_RECOVER True", run.stdout)

    def test_U1_startup_recovery_runs_before_any_shared_or_scripts_import(self) -> None:
        tree = ast.parse((REPO_ROOT / "SlicerDSpace.py").read_text(encoding="utf-8"), feature_version=(3, 9))
        call_index = first_package_import = None
        for index, node in enumerate(tree.body):
            if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                    and getattr(node.value.func, "id", "") == "_recover_interrupted_update" and call_index is None):
                call_index = index
            names = []
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            if first_package_import is None and any(n.split(".")[0] in ("shared", "scripts") for n in names):
                first_package_import = index
        self.assertIsNotNone(call_index, "SlicerDSpace.py does not call _recover_interrupted_update() at module level")
        self.assertLess(call_index, first_package_import)

    def test_U1_startup_block_rolls_back_and_the_package_imports_again(self) -> None:
        source = (REPO_ROOT / "SlicerDSpace.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_recover_interrupted_update"]
        self.assertTrue(nodes, "no _recover_interrupted_update() in SlicerDSpace.py")
        block = "import os\n" + ast.get_source_segment(source, nodes[0]) + "\n_recover_interrupted_update()\n"
        with tempfile.TemporaryDirectory() as td:
            root = _broken_install(Path(td), _journal(_dead_pid()))
            probe = (
                "import sys\n"
                "sys.path.insert(0, sys.argv[2])\n"
                "exec(compile(open(sys.argv[1]).read(), sys.argv[3], 'exec'), {'__file__': sys.argv[3], '__name__': 'probe'})\n"
                "import shared\n"
                "print('VALUE', shared.VALUE)\n"
            )
            block_file = Path(td) / "block.py"
            block_file.write_text(block, encoding="utf-8")
            run = subprocess.run([sys.executable, "-c", probe, str(block_file), str(root), str(root / "SlicerDSpace.py")],
                                 capture_output=True, text=True, cwd=str(root), env=_CHILD_ENV, timeout=120)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("VALUE old", run.stdout)
            self.assertIn("rolled back an interrupted update", run.stdout)
            self.assertFalse((root / "shared" / "new_mod.py").exists())
            self.assertFalse((root / "shared" / "__pycache__" / "__init__.cpython-312.pyc").exists())
            self.assertFalse((root / ".slicerdspace_update_journal").exists())


# ---------------------------------------------------------------- U2
_WORKER = r"""
import json, sys, time
from pathlib import Path
from shared.code_updater import download_and_apply_code_overlay
args = json.loads(sys.argv[1])
def hook(rel):
    if rel == "SlicerDSpace.py":  # R3-N1: the entry file is written last
        Path(args["ready"]).write_text("1")
        time.sleep(300)
download_and_apply_code_overlay(module_root=Path(args["inst"]), manifest=args["manifest"],
                                offline_dir=Path(args["offline"]), _apply_hook=hook)
"""


class U2OneUpdaterTests(unittest.TestCase):
    def test_U2_second_worker_refuses_while_the_first_applies_then_dead_owner_rolls_back(self) -> None:
        from shared.code_updater import (
            JOURNAL_DIRNAME, compute_code_tree_sha256, download_and_apply_code_overlay, recover_interrupted_update,
        )

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            inst, orig_sha, _v2, manifest, offline = _updater_fixture(td)
            ready = td / "ready"
            args = {"inst": str(inst), "manifest": manifest, "offline": str(offline), "ready": str(ready)}
            worker = subprocess.Popen([sys.executable, "-c", _WORKER, json.dumps(args)], cwd=str(REPO_ROOT),
                                      env=_CHILD_ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 120
                while not ready.exists():
                    if worker.poll() is not None or time.monotonic() > deadline:
                        self.fail("first worker did not reach the apply step: " + worker.stdout.read().decode())
                    time.sleep(0.05)
                self.assertIn("'v2'", (inst / "shared" / "helper.py").read_text(encoding="utf-8"))
                self.assertNotIn("VALUE = 2", (inst / "SlicerDSpace.py").read_text(encoding="utf-8"))
                journal_file = inst / JOURNAL_DIRNAME / "journal.json"
                self.assertTrue(journal_file.is_file())
                before = _tree_bytes(inst)

                with self.assertRaisesRegex(RuntimeError, "Another SlicerDSpace update"):
                    download_and_apply_code_overlay(module_root=inst, manifest=manifest, offline_dir=offline)
                self.assertEqual(_tree_bytes(inst), before)
                self.assertEqual(recover_interrupted_update(inst)["status"], "update_in_progress")
                self.assertEqual(_tree_bytes(inst), before)
            finally:
                worker.kill()
                worker.wait()
                worker.stdout.close()

            result = recover_interrupted_update(inst)
            self.assertEqual(result["status"], "rolled_back")
            self.assertEqual(compute_code_tree_sha256(inst)[0], orig_sha)
            self.assertFalse((inst / JOURNAL_DIRNAME).exists())

    def _journal_with_backup(self, inst: Path, journal: dict) -> Path:
        from shared.code_updater import JOURNAL_DIRNAME

        journal_dir = inst / JOURNAL_DIRNAME
        (journal_dir / "backup" / "shared").mkdir(parents=True)
        shutil.copy2(inst / "shared" / "helper.py", journal_dir / "backup" / "shared" / "helper.py")
        (journal_dir / "journal.json").write_text(json.dumps(journal), encoding="utf-8")
        return journal_dir / "journal.json"

    def test_U2_live_owner_is_not_treated_as_dead_when_psutil_is_missing(self) -> None:
        import psutil
        from shared.code_updater import recover_interrupted_update

        worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        try:
            created = psutil.Process(worker.pid).create_time()
            for label, extra, no_psutil in (("with create time, no psutil", {"pid_created": created}, True),
                                            ("no create time, no psutil", {}, True),
                                            ("no create time, psutil", {}, False)):
                with self.subTest(label), tempfile.TemporaryDirectory() as td:
                    inst, _orig, _v2, _manifest, _offline = _updater_fixture(Path(td))
                    _unavailable_lock(inst)  # R3-B1: the owner check decides only when the lock cannot be opened
                    journal = {"kind": "slicerdspace_update_journal", "version": 1, "target_version": "9",
                               "pid": worker.pid, "backed_up": ["shared/helper.py"], "created": [], **extra}
                    journal_file = self._journal_with_backup(inst, journal)
                    modules = {"psutil": None} if no_psutil else {}
                    with patch.dict(sys.modules, modules):
                        result = recover_interrupted_update(inst)
                    self.assertEqual(result["status"], "update_in_progress")
                    self.assertTrue(journal_file.is_file())
        finally:
            worker.kill()
            worker.wait()

    def test_U2_without_psutil_a_dead_or_too_old_owner_is_rolled_back(self) -> None:
        from shared.code_updater import recover_interrupted_update

        worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        try:
            for label, pid, age in (("dead pid", _dead_pid(), 0), ("live pid older than the age limit", worker.pid, 3 * 3600)):
                with self.subTest(label), tempfile.TemporaryDirectory() as td:
                    inst, _orig, _v2, _manifest, _offline = _updater_fixture(Path(td))
                    _unavailable_lock(inst)  # R3-B1: with a free lock the owner check is skipped altogether
                    journal = {"kind": "slicerdspace_update_journal", "version": 1, "target_version": "9",
                               "pid": pid, "started": time.time() - age, "backed_up": ["shared/helper.py"], "created": []}
                    journal_file = self._journal_with_backup(inst, journal)
                    old = time.time() - age
                    os.utime(journal_file, (old, old))
                    with patch.dict(sys.modules, {"psutil": None}):
                        result = recover_interrupted_update(inst)
                    self.assertEqual(result["status"], "rolled_back")
        finally:
            worker.kill()
            worker.wait()

    def test_U2_apply_refuses_when_the_journal_cannot_be_rolled_back(self) -> None:
        from shared.code_updater import JOURNAL_DIRNAME, download_and_apply_code_overlay

        with tempfile.TemporaryDirectory() as td:
            inst, _orig, _v2, manifest, offline = _updater_fixture(Path(td))
            journal_dir = inst / JOURNAL_DIRNAME
            journal_dir.mkdir()
            (journal_dir / "journal.json").write_text(json.dumps(
                {"kind": "slicerdspace_update_journal", "version": 1, "target_version": "9", "pid": _dead_pid(),
                 "backed_up": ["shared/helper.py"], "created": []}), encoding="utf-8")
            before = _tree_bytes(inst)
            with self.assertRaises(RuntimeError):
                download_and_apply_code_overlay(module_root=inst, manifest=manifest, offline_dir=offline)
            self.assertEqual(_tree_bytes(inst), before)


# ---------------------------------------------------------------- R3 (third fix round)
def _unavailable_lock(inst: Path) -> None:
    """A directory at the lock path: os.open fails (EISDIR), so acquire_update_lock returns "unavailable"."""

    (inst / ".slicerdspace_update.lock").mkdir()


def _live_unrelated_process() -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])


class R3UpdaterLockTests(unittest.TestCase):
    def _journal_with_backup(self, inst: Path, journal: dict) -> Path:
        return U2OneUpdaterTests._journal_with_backup(self, inst, journal)

    def test_r3_U2_apply_refuses_without_changing_files_when_the_lock_is_unavailable(self) -> None:
        from shared.code_updater import JOURNAL_DIRNAME, download_and_apply_code_overlay
        from shared.update_journal import acquire_update_lock

        with tempfile.TemporaryDirectory() as td:
            inst, _orig, _v2, manifest, offline = _updater_fixture(Path(td))
            _unavailable_lock(inst)
            self.assertEqual(acquire_update_lock(inst), ("unavailable", None))
            before = _tree_bytes(inst)
            with self.assertRaisesRegex(RuntimeError, "update lock"):
                download_and_apply_code_overlay(module_root=inst, manifest=manifest, offline_dir=offline)
            self.assertEqual(_tree_bytes(inst), before)
            self.assertFalse((inst / JOURNAL_DIRNAME).exists())

    def test_r3_B1_free_lock_rolls_back_a_live_unrelated_pid_without_psutil(self) -> None:
        from shared.code_updater import compute_code_tree_sha256, download_and_apply_code_overlay, recover_interrupted_update

        other = _live_unrelated_process()
        try:
            journal = {"kind": "slicerdspace_update_journal", "version": 1, "target_version": "9", "pid": other.pid,
                       "started": time.time(), "backed_up": ["shared/helper.py"], "created": []}
            with self.subTest("startup recovery"), tempfile.TemporaryDirectory() as td:
                inst, orig_sha, _v2, _manifest, _offline = _updater_fixture(Path(td))
                journal_file = self._journal_with_backup(inst, journal)
                (inst / "shared" / "helper.py").write_text("half written\n", encoding="utf-8")
                with patch.dict(sys.modules, {"psutil": None}):
                    result = recover_interrupted_update(inst)
                self.assertEqual(result["status"], "rolled_back")
                self.assertFalse(journal_file.exists())
                self.assertEqual(compute_code_tree_sha256(inst)[0], orig_sha)
            with self.subTest("apply with the lock held"), tempfile.TemporaryDirectory() as td:
                inst, _orig, v2_sha, manifest, offline = _updater_fixture(Path(td))
                self._journal_with_backup(inst, journal)
                with patch.dict(sys.modules, {"psutil": None}):
                    result = download_and_apply_code_overlay(module_root=inst, manifest=manifest, offline_dir=offline)
                self.assertEqual(result["status"], "applied")
                self.assertEqual(compute_code_tree_sha256(inst)[0], v2_sha)
        finally:
            other.kill()
            other.wait()

    def test_r3_B1_unavailable_lock_keeps_the_owner_check(self) -> None:
        import psutil
        from shared.code_updater import recover_interrupted_update

        owner = _live_unrelated_process()
        try:
            with tempfile.TemporaryDirectory() as td:
                inst, _orig, _v2, _manifest, _offline = _updater_fixture(Path(td))
                _unavailable_lock(inst)
                journal_file = self._journal_with_backup(inst, {
                    "kind": "slicerdspace_update_journal", "version": 1, "target_version": "9", "pid": owner.pid,
                    "pid_created": psutil.Process(owner.pid).create_time(), "started": time.time(),
                    "backed_up": ["shared/helper.py"], "created": []})
                before = _tree_bytes(inst)
                self.assertEqual(recover_interrupted_update(inst)["status"], "update_in_progress")
                self.assertEqual(_tree_bytes(inst), before)
                owner.kill()
                owner.wait()
                self.assertEqual(recover_interrupted_update(inst)["status"], "rolled_back")  # provably dead owner
                self.assertFalse(journal_file.exists())
        finally:
            owner.kill()
            owner.wait()

    def test_r3_N1_SlicerDSpace_py_is_written_after_every_other_runtime_file(self) -> None:
        from shared.code_updater import download_and_apply_code_overlay

        with tempfile.TemporaryDirectory() as td:
            inst, _orig, _v2, manifest, offline = _updater_fixture(Path(td))
            obsolete = inst / "shared" / "obsolete_r3.py"
            obsolete.write_text("OLD = 1\n", encoding="utf-8")
            order, at_entry = [], {}

            def hook(rel: str) -> None:
                order.append(rel)
                if rel == "SlicerDSpace.py":
                    at_entry["helper"] = (inst / "shared" / "helper.py").read_text(encoding="utf-8")
                    at_entry["obsolete_left"] = obsolete.exists()

            result = download_and_apply_code_overlay(module_root=inst, manifest=manifest, offline_dir=offline,
                                                     _apply_hook=hook)
            self.assertEqual(result["status"], "applied")
            self.assertGreater(len(order), 2)
            self.assertEqual(order[-1], "SlicerDSpace.py")
            self.assertEqual(order.count("SlicerDSpace.py"), 1)
            self.assertIn("'v2'", at_entry["helper"])
            self.assertFalse(at_entry["obsolete_left"])


# ---------------------------------------------------------------- P-h
class PhSupportedSlicerTests(_InstallerCopies, unittest.TestCase):
    def _win(self, base: Path, name: str, minor: int) -> Path:
        directory = base / name
        (directory / "bin").mkdir(parents=True)
        (directory / "Slicer.exe").write_bytes(b"x")
        (directory / "bin" / "PythonSlicer.exe").write_bytes(b"x")
        (directory / "lib" / ("Slicer-5." + str(minor))).mkdir(parents=True)
        return directory / "Slicer.exe"

    def _mac(self, base: Path, name: str, minor: int) -> Path:
        contents = base / name / "Contents"
        (contents / "MacOS").mkdir(parents=True)
        (contents / "bin").mkdir(parents=True)
        (contents / "MacOS" / "Slicer").write_bytes(b"x")
        (contents / "bin" / "PythonSlicer").write_bytes(b"x")
        (contents / "lib" / ("Slicer-5." + str(minor))).mkdir(parents=True)
        return contents / "MacOS" / "Slicer"

    def _linux(self, base: Path, name: str, minor: int) -> Path:
        directory = base / name
        (directory / "bin").mkdir(parents=True)
        (directory / "Slicer").write_bytes(b"x")
        (directory / "bin" / "PythonSlicer").write_bytes(b"x")
        (directory / "lib" / ("Slicer-5." + str(minor))).mkdir(parents=True)
        return directory / "Slicer"

    def test_P_h_a_stable_release_wins_over_a_newer_preview(self) -> None:
        for kind, ns in self.copies():
            with self.subTest(copy=kind), tempfile.TemporaryDirectory() as td:
                base = Path(td) / "slicer.org"
                old = self._win(base, "Slicer 5.8.1", 8)
                stable = self._win(base, "Slicer 5.12.1", 12)
                self._win(base, "Slicer 5.13.0-2026-11-01", 13)
                self._win(base, "Slicer 5.11.0-2026-01-01", 11)
                with patch.dict(os.environ, {"LOCALAPPDATA": td}), patch.object(sys, "platform", "win32"), \
                        patch.dict(ns, {"_log": lambda _m: None}):
                    self.assertEqual(Path(ns["_pick_supported_slicer"](str(old), False)), stable)

    def test_P_h_a_preview_is_used_only_when_nothing_else_fits_and_is_named(self) -> None:
        for kind, ns in self.copies():
            with self.subTest(copy=kind), tempfile.TemporaryDirectory() as td:
                base = Path(td) / "slicer.org"
                old = self._win(base, "Slicer 5.8.1", 8)
                preview = self._win(base, "Slicer 5.13.0-2026-11-01", 13)
                logs = []
                with patch.dict(os.environ, {"LOCALAPPDATA": td}), patch.object(sys, "platform", "win32"), \
                        patch.dict(ns, {"_log": logs.append}):
                    self.assertEqual(Path(ns["_pick_supported_slicer"](str(old), False)), preview)
                self.assertTrue(any("5.13" in m and "5.10" in m and "5.12" in m for m in logs), logs)

    def test_P_h_macos_scans_the_applications_folders(self) -> None:
        for kind, ns in self.copies():
            with self.subTest(copy=kind), tempfile.TemporaryDirectory() as td:
                apps, home_apps = Path(td) / "Applications", Path(td) / "home" / "Applications"
                first = self._mac(apps, "Slicer.app", 8)
                self._mac(apps, "Slicer-5.13-preview.app", 13)
                best = self._mac(home_apps, "Slicer 5.12.app", 12)
                dirs = ((str(apps), "*Slicer*.app"), (str(home_apps), "*Slicer*.app"))
                with patch.object(sys, "platform", "darwin"), patch.dict(ns, {"_log": lambda _m: None,
                                                                             "_MACOS_SLICER_DIRS": dirs}):
                    self.assertEqual(Path(ns["_pick_supported_slicer"](str(first), False)), best)
                    with self.assertRaisesRegex(RuntimeError, r"found 5\.8"):
                        ns["_pick_supported_slicer"](str(first), True)  # an explicit path is never replaced
                    shutil.rmtree(Path(td) / "home")
                    shutil.rmtree(apps / "Slicer-5.13-preview.app")
                    with self.assertRaisesRegex(RuntimeError, r"Requires 3D Slicer 5\.10 or later \(found 5\.8\)"):
                        ns["_pick_supported_slicer"](str(first), False)

    def test_P_h_linux_scans_the_common_folders(self) -> None:
        for kind, ns in self.copies():
            with self.subTest(copy=kind), tempfile.TemporaryDirectory() as td:
                opt = Path(td) / "opt"
                first = self._linux(Path(td) / "home", "Slicer-5.8.1-linux-amd64", 8)
                best = self._linux(opt, "Slicer-5.10.0-linux-amd64", 10)
                dirs = ((str(Path(td) / "home"), "Slicer*"), (str(opt), "Slicer*"))
                with patch.object(sys, "platform", "linux"), patch.dict(ns, {"_log": lambda _m: None,
                                                                            "_LINUX_SLICER_DIRS": dirs}):
                    self.assertEqual(Path(ns["_pick_supported_slicer"](str(first), False)), best)


if __name__ == "__main__":
    unittest.main()
