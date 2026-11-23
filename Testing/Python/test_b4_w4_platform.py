"""Batch 4 task W4: platform, installer and updater findings (X4, X2, F3, F5, F8/F2, H6/X5, F6)."""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
JS_PATH = REPO_ROOT / "docs" / "site" / "assets" / "js" / "installer-customizer.js"
PE = b"MZ" + bytes(58) + (64).to_bytes(4, "little") + b"PE\x00\x00"


def _program(directory: Path, name: str, content: bytes = PE) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(content)
    path.chmod(0o755)
    return path


# ---------------------------------------------------------------- X4 / X2
class X4WslBridgeFallbackTests(unittest.TestCase):
    def _bridge_dir(self, root: Path, names) -> Path:
        directory = root / "bridge"
        directory.mkdir(parents=True)
        for name in names:
            (directory / name).write_bytes(PE)
            (directory / (name + ".conf")).write_text("distribution=Ubuntu\ntarget=/x\n", encoding="utf-8")
            (directory / (name + ".slicerdsurfer-owner")).write_text("bridge\n", encoding="utf-8")
        return directory

    def test_x4_wsl_bridge_folders_fall_back_to_native_bundles(self) -> None:
        from preprocessing.fsl_runtime import inspect_native_fsl_runtime

        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            mrtrix_bridge = self._bridge_dir(root / "m", ["mrconvert.exe", "mrinfo.exe", "dwi2mask.exe", "maskfilter.exe"])
            fsl_bridge = self._bridge_dir(root / "f", ["eddy_cpu.exe"])
            native_mrtrix, native_fsl = root / "mrtrix-3-windows-x86_64" / "bin", root / "fsl-6-windows-x86_64" / "bin"
            for name in ("mrconvert", "mrinfo", "dwi2mask", "maskfilter"):
                _program(native_mrtrix, name + ".exe")
            native_eddy = _program(native_fsl, "eddy_cpu.exe")

            def fake_bundles(tool, system=None, **_kw):
                return (native_mrtrix,) if tool == "mrtrix" else (native_fsl,)

            with patch("shared.native_runtimes.installed_bundle_bin_dirs", side_effect=fake_bundles), \
                    patch.dict(os.environ, {"SLICERDSPACE_FSL_BIN_DIR": "", "FSL_BIN_DIR": "", "SLICERDSPACE_MRTRIX_BIN_DIR": "",
                                            "MRTRIX_BIN_DIR": ""}):
                status = inspect_native_fsl_runtime(mrtrix_bridge, fsl_bridge, reverse_phase=False, platform="Windows")
        self.assertTrue(status["available"], status)
        self.assertEqual(status["commands"]["eddy"], str(native_eddy))
        self.assertEqual(Path(status["mrtrix_bin_dir"]), native_mrtrix)

    def test_x4_bridge_folder_without_a_native_bundle_does_not_fall_back_to_the_environment(self) -> None:
        from preprocessing.fsl_runtime import _candidate_dirs

        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            bridge = self._bridge_dir(root, ["eddy_cpu.exe"])
            other = root / "other"
            _program(other, "eddy_cpu.exe")
            with patch("shared.native_runtimes.installed_bundle_bin_dirs", return_value=()), \
                    patch.dict(os.environ, {"SLICERDSPACE_FSL_BIN_DIR": str(other)}):
                self.assertEqual(_candidate_dirs(bridge, "fsl", "windows"), ())

    def test_x4_a_real_native_folder_is_still_used_alone(self) -> None:
        from preprocessing.fsl_runtime import _candidate_dirs

        with tempfile.TemporaryDirectory() as td:
            native = Path(td).resolve() / "bin"
            _program(native, "eddy_cpu.exe")
            self.assertEqual(_candidate_dirs(native, "fsl", "windows"), (native,))

    def test_x4_ps1_keeps_native_settings_when_a_native_bundle_is_installed(self) -> None:
        text = (REPO_ROOT / "scripts" / "install_deps_windows.ps1").read_text(encoding="utf-8")
        self.assertIn("function Test-NativeBundleInstalled", text)
        merge = text.index("foreach ($entry in $externalDependencies.GetEnumerator())")
        guard = text.index("Test-NativeBundleInstalled $nativeKeyTools[$entry.Key]")
        self.assertLess(merge, guard)
        guard_function = text[text.index("function Test-NativeBundleInstalled"):merge]
        for root_name in ("SLICERDSPACE_BUNDLED_RUNTIMES_ROOT", "bundled-runtimes", "SLICERDSPACE_RUNTIME_ROOT", "runtimes"):
            self.assertIn(root_name, guard_function)
        for key in ("mrtrix_bin_dir", "preprocessing_mrtrix_bin_dir", "preprocessing_fsl_bin_dir"):
            self.assertIn(f"{key} = ", text[text.index("$nativeKeyTools"):guard])

    def test_x2_bridge_reads_wslpath_as_utf8_with_a_timeout(self) -> None:
        source = (REPO_ROOT / "scripts" / "wsl" / "WslCommandBridge.cs").read_text(encoding="utf-8")
        start = source.index("private static string ConvertDrivePath")
        body = source[start:source.index("private static string ConvertArgument")]
        self.assertIn("StandardOutputEncoding = new UTF8Encoding(false)", body)
        self.assertIn("StandardErrorEncoding = new UTF8Encoding(false)", body)
        self.assertIn("WaitForExit(WslPathTimeoutMilliseconds)", body)
        self.assertIn("WslPathTimeoutMilliseconds = 30000", source)
        self.assertIn("did not finish within 30 s", body)
        self.assertNotIn("ReadToEnd()", body)

    def test_x2_ps1_decodes_wsl_output_as_utf8_before_wslpath(self) -> None:
        text = (REPO_ROOT / "scripts" / "install_deps_windows.ps1").read_text(encoding="utf-8")
        setting = text.index("[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)")
        self.assertLess(setting, text.index("function Convert-ToWslPath"))
        self.assertLess(setting, text.index("wslpath -a -u $portablePath"))
        self.assertLess(text.index("wsl.exe --list --quiet"), setting)


# ---------------------------------------------------------------- F3
class F3BundledPythonReasonTests(unittest.TestCase):
    def setUp(self) -> None:
        from shared import bundled_python

        self.bp = bundled_python
        self.bp._PROBLEMS.clear()
        self.bp._LOGGED_PROBLEMS.clear()
        self.addCleanup(self.bp._PROBLEMS.clear)
        self.addCleanup(self.bp._LOGGED_PROBLEMS.clear)
        env_patch = patch.dict(os.environ, {}, clear=False)
        env_patch.start()
        self.addCleanup(env_patch.stop)
        os.environ.pop(self.bp.BUNDLED_PYTHON_ENV, None)

    def _root(self, td: str, meta: dict) -> Path:
        root = Path(td) / "ext" / "bundled-python"
        (root / "site-packages").mkdir(parents=True)
        (root / "python-runtime.json").write_text(json.dumps(meta), encoding="utf-8")
        return root

    def test_f3_incompatible_bundle_records_a_reason_with_the_fix(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td, {"python_version": "3.99", "arch": "x86_64"})
            self.assertFalse(self.bp._runtime_compatible(root))
        reason = self.bp.activation_problem()
        self.assertIn("bundled Python stack is for Python 3.99 x86_64", reason)
        self.assertIn("this Slicer runs Python %d.%d on " % sys.version_info[:2], reason)
        self.assertIn("install the SlicerDSpace package built for this Slicer", reason)

    def test_f3_compatible_bundle_has_no_problem(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td, {"python_version": "%d.%d" % sys.version_info[:2]})
            self.assertTrue(self.bp._runtime_compatible(root))
        self.assertIsNone(self.bp.activation_problem())

    def test_f3_activation_logs_the_reason_once(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = self._root(td, {"python_version": "3.99", "arch": "x86_64"})
            with patch.dict(os.environ, {self.bp.BUNDLED_PYTHON_ENV: str(root)}), \
                    patch.object(self.bp, "_clear_macos_quarantine_once", return_value=False), \
                    self.assertLogs("shared.bundled_python", level="WARNING") as captured:
                self.bp.activate_bundled_python()
                self.bp.activate_bundled_python()
        self.assertEqual(len(captured.records), 1)
        self.assertIn("is not used", captured.output[0])


# ---------------------------------------------------------------- F5 / F8 / F2 helpers
def _extract_payload(script: str) -> str:
    begin, end = "# __SLICERDSPACE_PY_BEGIN__\n", "# __SLICERDSPACE_PY_END__"
    script = script.replace("\r\n", "\n")
    return script.split(begin, 1)[1].split(end, 1)[0]


def _python_payload(target_os: str = "linux") -> str:
    from shared.installer_script_builder import generate_installer_script

    return _extract_payload(generate_installer_script(target_os=target_os, mode="base", slicer_path="", language="en"))


def _js_payload(target_os: str = "linux") -> str:
    node = shutil.which("node")
    if node is None:
        raise unittest.SkipTest("node is not installed in PATH")
    code = ("const ic = require(process.argv[1]);"
            "process.stdout.write(ic.buildScript({ os: '%s' }));" % target_os)
    out = subprocess.run([node, "-e", code, str(JS_PATH)], capture_output=True, text=True, encoding="utf-8",
                         check=True).stdout
    return _extract_payload(out)


def _load(payload: str) -> dict:
    namespace: dict = {"__name__": "w4_payload", "__file__": "payload.py"}
    exec(compile(payload, "payload.py", "exec"), namespace)
    return namespace


class _InstallerCopies:
    """Run each test on the Python mirror and on the website JS copy."""

    def copies(self):
        yield "python", _load(_python_payload("windows"))
        try:
            yield "js", _load(_js_payload("windows"))
        except unittest.SkipTest:
            pass


class F5SkipForeignCliModulesTests(_InstallerCopies, unittest.TestCase):
    def _archive(self, td: Path) -> Path:
        path = td / "base.zip"
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("SlicerDSpace/lib/Slicer-5.10/cli-modules/UKFTractography.exe", b"ukf")
            zf.writestr("SlicerDSpace/lib/Slicer-5.10/cli-modules/UKFBase.dll", b"dll")
            zf.writestr("SlicerDSpace/lib/Slicer-5.10/qt-scripted-modules/SlicerDSpace.py", b"# module\n")
            zf.writestr("SlicerDSpace/share/Slicer-5.10/SlicerDSpace.s4ext", b"arch amd64\n")
        return path

    def test_f5_cli_modules_are_not_copied_into_another_slicer_version(self) -> None:
        for kind, ns in self.copies():
            with self.subTest(copy=kind), tempfile.TemporaryDirectory() as td:
                td = Path(td)
                logs = []
                root = td / "ext"
                extract = ns["_extract_base_archive_remapped"]
                kwargs = {"log_cb": logs.append} if kind == "js" else {}
                with patch.dict(ns, {"_log": logs.append}), patch.object(sys, "platform", "win32"):
                    extract(self._archive(td), root, "34645", "Slicer-5.12", **kwargs)
                self.assertFalse((root / "lib" / "Slicer-5.12" / "cli-modules").exists())
                self.assertTrue((root / "lib" / "Slicer-5.12" / "qt-scripted-modules" / "SlicerDSpace.py").is_file())
                self.assertTrue(any("UKFTractography for this Slicer must come from the Extension Manager (SlicerDMRI)" in m
                                    for m in logs), logs)

    def test_f5_macos_keeps_its_cli_modules(self) -> None:
        """macOS UKF is a standalone bundle: the F5 skip must not change the macOS layout."""
        for kind, ns in self.copies():
            with self.subTest(copy=kind), tempfile.TemporaryDirectory() as td:
                td = Path(td)
                root = td / "ext"
                kwargs = {"log_cb": lambda _m: None} if kind == "js" else {}
                with patch.dict(ns, {"_log": lambda _m: None}), patch.object(sys, "platform", "darwin"):
                    ns["_extract_base_archive_remapped"](self._archive(td), root, "34645", "Slicer-5.12", **kwargs)
                self.assertTrue((root / "lib" / "Slicer-5.12" / "cli-modules" / "UKFTractography.exe").is_file())

    def test_f5_same_version_keeps_cli_modules(self) -> None:
        for kind, ns in self.copies():
            with self.subTest(copy=kind), tempfile.TemporaryDirectory() as td:
                td = Path(td)
                root = td / "ext"
                kwargs = {"log_cb": lambda _m: None} if kind == "js" else {}
                ns["_extract_base_archive_remapped"](self._archive(td), root, "34045", "Slicer-5.10", **kwargs)
                self.assertTrue((root / "lib" / "Slicer-5.10" / "cli-modules" / "UKFTractography.exe").is_file())


class F8F2SlicerChoiceTests(_InstallerCopies, unittest.TestCase):
    def _slicer(self, base: Path, name: str, minor) -> Path:
        directory = base / name
        (directory / "bin").mkdir(parents=True)
        (directory / "Slicer.exe").write_bytes(b"x")
        (directory / "bin" / "PythonSlicer.exe").write_bytes(b"x")
        if minor is not None:
            (directory / "lib" / ("Slicer-5." + str(minor))).mkdir(parents=True)
        return directory / "Slicer.exe"

    def test_f8_reads_version_from_lib_folder(self) -> None:
        for kind, ns in self.copies():
            with self.subTest(copy=kind), tempfile.TemporaryDirectory() as td:
                exe = self._slicer(Path(td), "Slicer 5.8.1", 8)
                self.assertEqual(ns["_slicer_lib_version"](str(exe)), (5, 8))
                exe = self._slicer(Path(td), "Slicer 5.12.0", 12)
                self.assertEqual(ns["_slicer_lib_version"](str(exe)), (5, 12))

    def test_f8_picks_the_newest_supported_slicer_on_windows(self) -> None:
        for kind, ns in self.copies():
            with self.subTest(copy=kind), tempfile.TemporaryDirectory() as td:
                base = Path(td) / "slicer.org"
                old = self._slicer(base, "Slicer 5.6.2", 6)
                self._slicer(base, "Slicer 5.10.0", 10)
                newest = self._slicer(base, "Slicer 5.12.1", 12)
                logs = []
                with patch.dict(os.environ, {"LOCALAPPDATA": td}), patch.object(sys, "platform", "win32"), \
                        patch.dict(ns, {"_log": logs.append}):
                    chosen = ns["_pick_supported_slicer"](str(old), False)
                self.assertEqual(Path(chosen), newest)
                self.assertTrue(any("5.12" in m and str(newest) in m for m in logs), logs)

    def test_f8_an_explicit_slicer_is_not_replaced(self) -> None:
        for kind, ns in self.copies():
            with self.subTest(copy=kind), tempfile.TemporaryDirectory() as td:
                base = Path(td) / "slicer.org"
                explicit = self._slicer(base, "Slicer 5.10.0", 10)
                self._slicer(base, "Slicer 5.12.1", 12)
                with patch.dict(os.environ, {"LOCALAPPDATA": td}), patch.object(sys, "platform", "win32"), \
                        patch.dict(ns, {"_log": lambda _m: None}):
                    self.assertEqual(Path(ns["_pick_supported_slicer"](str(explicit), True)), explicit)

    def test_f2_unsupported_version_stops_before_any_download(self) -> None:
        for kind, ns in self.copies():
            with self.subTest(copy=kind), tempfile.TemporaryDirectory() as td:
                exe = self._slicer(Path(td), "Slicer 5.8.1", 8)

                def no_download(*_a, **_k):
                    raise AssertionError("download started before the version gate")

                # P-h (fix round): macOS/Linux now scan the usual Slicer folders too; keep this host's Slicers out.
                patches = {"_acquire_spec": no_download, "_pick_cache_dir": no_download,
                           "_is_intel_mac": lambda: False, "_log": lambda _m: None,
                           "_posix_slicer_candidates": lambda: []}
                ns["CONFIG"] = dict(ns["CONFIG"], slicer_path="")
                with patch.dict(os.environ, {"SLICERDSPACE_RESOLVED_SLICER": str(exe), "LOCALAPPDATA": td}), \
                        patch.dict(ns, patches):
                    with self.assertRaisesRegex(RuntimeError, r"Requires 3D Slicer 5\.10 or later \(found 5\.8\)"):
                        ns["main"]()

    def test_f2_unknown_layout_is_left_to_the_in_slicer_gate(self) -> None:
        for kind, ns in self.copies():
            with self.subTest(copy=kind), tempfile.TemporaryDirectory() as td:
                exe = self._slicer(Path(td), "Slicer", None)
                with patch.dict(ns, {"_log": lambda _m: None}):
                    self.assertEqual(ns["_pick_supported_slicer"](str(exe), True), str(exe))


# ---------------------------------------------------------------- H6 / X5
def _updater_fixture(td: Path):
    from Testing.Python.test_code_updater import _build_tar_gz, _copy_minimal_install
    from shared.code_updater import RUNTIME_ABI_VERSION, compute_code_tree_sha256

    inst = _copy_minimal_install(td / "inst")
    orig_sha, _ = compute_code_tree_sha256(inst)
    new_tree = _copy_minimal_install(
        td / "v2",
        extra_files={"SlicerDSpace.py": b"# entry v2\nVALUE = 2\n",
                     "shared/helper.py": b"def version():\n    return 'v2'\n"},
    )
    v2_sha, v2_files = compute_code_tree_sha256(new_tree)
    archive = td / "good.tar.gz"
    size, sha = _build_tar_gz(archive, {rel: (new_tree / rel).read_bytes() for rel in v2_files})
    offline = td / "off"
    offline.mkdir()
    shutil.copy2(archive, offline / archive.name)
    manifest = {
        "schema_version": 1, "version": "0.1.2", "runtime_abi_version": RUNTIME_ABI_VERSION,
        "code_sha256": v2_sha,
        "overlay": {"filename": archive.name, "url": "http://127.0.0.1:1/good.tar.gz", "sha256": sha, "size_bytes": size},
    }
    return inst, orig_sha, v2_sha, manifest, offline


class _Killed(BaseException):
    """Stands for a worker killed by the watchdog: not an Exception, so no in-process rollback runs."""


class H6UpdateJournalTests(unittest.TestCase):
    def test_h6_killed_apply_is_rolled_back_on_next_start(self) -> None:
        from shared.code_updater import (
            JOURNAL_DIRNAME, compute_code_tree_sha256, download_and_apply_code_overlay, recover_interrupted_update,
        )

        with tempfile.TemporaryDirectory() as td:
            inst, orig_sha, _v2, manifest, offline = _updater_fixture(Path(td))

            def kill_at_entry(rel: str) -> None:
                if rel == "SlicerDSpace.py":  # R3-N1: written last, so every other file is already new here
                    raise _Killed()

            with self.assertRaises(_Killed):
                download_and_apply_code_overlay(module_root=inst, manifest=manifest, offline_dir=offline,
                                                _apply_hook=kill_at_entry)
            self.assertIn("'v2'", (inst / "shared" / "helper.py").read_text(encoding="utf-8"))
            self.assertNotIn("VALUE = 2", (inst / "SlicerDSpace.py").read_text(encoding="utf-8"))
            self.assertTrue((inst / JOURNAL_DIRNAME / "journal.json").is_file())
            self.assertNotEqual(compute_code_tree_sha256(inst)[0], orig_sha)

            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                result = recover_interrupted_update(inst)
            self.assertEqual(result["status"], "rolled_back")
            self.assertIn("rolled back an interrupted update", out.getvalue())
            self.assertEqual(compute_code_tree_sha256(inst)[0], orig_sha)
            self.assertFalse((inst / JOURNAL_DIRNAME).exists())
            self.assertIsNone(recover_interrupted_update(inst))

    def test_h6_journal_is_removed_after_success(self) -> None:
        from shared.code_updater import JOURNAL_DIRNAME, download_and_apply_code_overlay, recover_interrupted_update

        with tempfile.TemporaryDirectory() as td:
            inst, _orig, v2_sha, manifest, offline = _updater_fixture(Path(td))
            phases = []
            result = download_and_apply_code_overlay(module_root=inst, manifest=manifest, offline_dir=offline,
                                                     phase_callback=phases.append)
            self.assertEqual(result["status"], "applied")
            self.assertEqual(phases, ["downloading", "verifying", "applying"])
            self.assertFalse((inst / JOURNAL_DIRNAME).exists())
            self.assertIsNone(recover_interrupted_update(inst))
            self.assertIn("VALUE = 2", (inst / "SlicerDSpace.py").read_text(encoding="utf-8"))

    def test_h6_completed_journal_never_rolls_back_a_good_update(self) -> None:
        from shared.code_updater import JOURNAL_DIRNAME, recover_interrupted_update

        with tempfile.TemporaryDirectory() as td:
            inst = Path(td)
            (inst / JOURNAL_DIRNAME).mkdir()
            (inst / JOURNAL_DIRNAME / "journal.json").write_text('{"completed": true}', encoding="utf-8")
            self.assertIsNone(recover_interrupted_update(inst))
            self.assertFalse((inst / JOURNAL_DIRNAME).exists())

    def test_h6_missing_backup_is_reported_and_keeps_the_journal(self) -> None:
        from shared.code_updater import JOURNAL_DIRNAME, _write_update_journal, recover_interrupted_update

        with tempfile.TemporaryDirectory() as td:
            inst = Path(td)
            _write_update_journal(inst / JOURNAL_DIRNAME, target_version="9", backed_up=["shared/a.py"], created=[])
            with contextlib.redirect_stdout(io.StringIO()) as out:
                result = recover_interrupted_update(inst)
            self.assertEqual(result["status"], "rollback_incomplete")
            self.assertTrue((inst / JOURNAL_DIRNAME / "journal.json").is_file())
            self.assertIn("Reinstall the SlicerDSpace package", out.getvalue())

    def test_h6_journal_of_a_running_worker_is_not_rolled_back(self) -> None:
        import psutil
        from shared.code_updater import JOURNAL_DIRNAME, recover_interrupted_update

        with tempfile.TemporaryDirectory() as td:
            inst, orig_sha, _v2, manifest, offline = _updater_fixture(Path(td))
            # R3-B1: a running updater holds the update lock; with a free lock the owner check is skipped.
            worker = subprocess.Popen(
                [sys.executable, "-c", "import sys, time; from pathlib import Path; "
                 "from shared.update_journal import acquire_update_lock; "
                 "print(acquire_update_lock(Path(sys.argv[1]))[0], flush=True); time.sleep(60)", str(inst)],
                cwd=str(REPO_ROOT), stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(worker.stdout.readline().strip(), "acquired")
                journal_dir = inst / JOURNAL_DIRNAME
                (journal_dir / "backup").mkdir(parents=True)
                (journal_dir / "journal.json").write_text(json.dumps({
                    "kind": "slicerdspace_update_journal", "version": 1, "target_version": "9",
                    "pid": worker.pid, "pid_created": psutil.Process(worker.pid).create_time(),
                    "backed_up": ["shared/helper.py"], "created": []}), encoding="utf-8")
                result = recover_interrupted_update(inst)
                self.assertEqual(result["status"], "update_in_progress")
                self.assertTrue((journal_dir / "journal.json").is_file())
            finally:
                worker.kill()
                worker.wait()
                worker.stdout.close()

    def test_h6_startup_hook_runs_before_the_other_modules_import(self) -> None:
        text = (REPO_ROOT / "SlicerDSpace.py").read_text(encoding="utf-8")
        # U1 (fix round): recovery now runs before the first `shared` import, not after activate_bundled_python().
        hook = text.index("\n_recover_interrupted_update()\n")
        self.assertLess(hook, text.index("from shared.bundled_python import"))
        self.assertLess(hook, text.index("from shared.algorithm_selection import"))

    def test_x5_os_replace_retries_permission_errors(self) -> None:
        import shared.code_updater as cu

        calls = {"n": 0}
        real = os.replace

        def flaky(src, dst):
            if ".upd." in str(src):
                calls["n"] += 1
                if calls["n"] <= 3:
                    raise PermissionError("locked by scanner")
            return real(src, dst)

        with tempfile.TemporaryDirectory() as td:
            inst, orig_sha, v2_sha, manifest, offline = _updater_fixture(Path(td))
            with patch.object(cu, "_REPLACE_RETRY_INTERVAL", 0.001), patch.object(cu.os, "replace", side_effect=flaky):
                result = cu.download_and_apply_code_overlay(module_root=inst, manifest=manifest, offline_dir=offline)
            self.assertEqual(result["status"], "applied")
            self.assertGreaterEqual(calls["n"], 4)

    def test_x5_permission_error_that_never_clears_raises_and_rolls_back(self) -> None:
        import shared.code_updater as cu

        real = os.replace

        def locked(src, dst):
            if ".upd." in str(src):
                raise PermissionError("locked")
            return real(src, dst)

        with tempfile.TemporaryDirectory() as td:
            inst, orig_sha, v2_sha, manifest, offline = _updater_fixture(Path(td))
            with patch.object(cu, "_REPLACE_RETRY_SECONDS", 0.05), patch.object(cu, "_REPLACE_RETRY_INTERVAL", 0.005), \
                    patch.object(cu.os, "replace", side_effect=locked):
                with self.assertRaises(PermissionError):
                    cu.download_and_apply_code_overlay(module_root=inst, manifest=manifest, offline_dir=offline)
            self.assertEqual(cu.compute_code_tree_sha256(inst)[0], orig_sha)
            self.assertFalse((inst / cu.JOURNAL_DIRNAME).exists())

    def test_h6_worker_prints_phase_markers(self) -> None:
        from shared.code_updater import main

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            inst, _orig, _v2, manifest, offline = _updater_fixture(td)
            manifest_file = td / "manifest.json"
            manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
            out = io.StringIO()
            with patch.dict(os.environ, {"SLICERDSPACE_OFFLINE_DIR": str(offline)}), contextlib.redirect_stdout(out), \
                    patch("shared.code_updater.load_network_settings", return_value={"offline_assets_dir": str(offline)}):
                main(["apply", "--module-root", str(inst), "--manifest-file", str(manifest_file),
                      "--result-file", str(td / "result.json")])
            text = out.getvalue()
            self.assertIn("SLICERDSPACE_UPDATE_STAGE:downloading", text)
            self.assertIn("SLICERDSPACE_UPDATE_STAGE:applying", text)
            self.assertLess(text.index("STAGE:downloading"), text.index("STAGE:applying"))


class H6WatchdogTests(unittest.TestCase):
    class _Process:
        def __init__(self, chunks):
            self.chunks = list(chunks)

        def readAllStandardOutput(self):
            return self.chunks.pop(0) if self.chunks else ""

    class _Owner:
        def __init__(self, job):
            self._codeUpdateJob = job
            self._codeUpdateState = {"phase": "downloading", "percent": 0, "elapsed_sec": 0}
            self.logs = []

        def appendLog(self, message):
            self.logs.append(message)

    def _tick(self, job):
        from shared import ui_update_panel as panel

        owner = self._Owner(job)
        with patch.object(panel, "_kill_job_process") as kill, patch.object(panel, "_finalize_worker") as finalize, \
                patch.object(panel, "_stop_timer"), patch.object(panel, "_sync_idle_texts"):
            panel._on_watchdog_tick(owner)
        return owner, kill, finalize

    def _job(self, chunks=(), **extra):
        now = time.monotonic()
        job = {"kind": "apply", "process": self._Process(chunks), "completed": False, "watchdog_seconds": 240,
               "started_monotonic": now - 1000, "last_activity_monotonic": now - 1000, "output_tail": "", "line_buffer": ""}
        job.update(extra)
        return job

    def test_h6_watchdog_kills_a_worker_with_no_progress_before_applying(self) -> None:
        _owner, kill, finalize = self._tick(self._job())
        kill.assert_called_once()
        finalize.assert_called_once()

    def test_h6_download_progress_resets_the_watchdog(self) -> None:
        _owner, kill, finalize = self._tick(self._job(["SLICERDSPACE_UPDATE_PROGRESS:7\n"]))
        kill.assert_not_called()
        finalize.assert_not_called()

    def test_h6_watchdog_never_kills_after_the_applying_marker(self) -> None:
        job = self._job(["SLICERDSPACE_UPDATE_STAGE:applying\n"])
        owner, kill, finalize = self._tick(job)
        kill.assert_not_called()
        finalize.assert_not_called()
        self.assertTrue(job["applying"])
        job["applying_since"] = time.monotonic() - 1000
        owner, kill, finalize = self._tick(job)
        kill.assert_not_called()
        self.assertTrue(any("still applying" in message for message in owner.logs), owner.logs)

    def test_h6_closing_the_panel_does_not_kill_an_applying_worker(self) -> None:
        from shared import ui_update_panel as panel

        job = self._job(applying=True)
        owner = self._Owner(job)
        with patch.object(panel, "_kill_job_process") as kill, patch.object(panel, "_stop_timer"):
            panel.cleanup_update_panel(owner)
        kill.assert_not_called()


# ---------------------------------------------------------------- F6
class F6WmaFastPathMessageTests(unittest.TestCase):
    def _run(self, module, support, source_name, digest_attr, **kwargs):
        with tempfile.TemporaryDirectory() as td:
            package = Path(td)
            (package / source_name).write_bytes(b"author source")
            import hashlib

            digest = hashlib.sha256(b"author source").hexdigest()
            vtk = types.ModuleType("vtk")
            vtk.vtkVersion = types.SimpleNamespace(GetVTKVersion=lambda: kwargs.get("version", "9.6.2"))
            from tractography import wma_transform

            wma_transform._REPORTED_UNQUALIFIED_VTK.clear()
            err = io.StringIO()
            with patch.object(module, digest_attr, digest), patch.dict(sys.modules, {"vtk": vtk}), \
                    contextlib.redirect_stderr(err):
                first = support(package)
                support(package)
            return first, err.getvalue()

    def test_f6_transform_logs_the_reference_path_reason_once(self) -> None:
        from tractography import wma_transform as module

        record, text = self._run(module, module.registration_transform_support, "register_two_subjects.py", "AUTHOR_SHA256")
        self.assertFalse(record["supported"])
        message = ("WMA fast path qualified only for VTK 9.5.2; this Slicer has VTK 9.6.2; "
                   "using the reference path (same results, slower)")
        self.assertEqual(text.count(message), 1, text)

    def test_f6_fiber_conversion_logs_the_reference_path_reason_once(self) -> None:
        from tractography import wma_fibers as module

        record, text = self._run(module, module.fiber_conversion_support, "fibers.py", "AUTHOR_SHA256")
        self.assertFalse(record["supported"])
        self.assertEqual(text.count("this Slicer has VTK 9.6.2; using the reference path (same results, slower)"), 1, text)

    def test_f6_both_fast_paths_share_one_line_per_run(self) -> None:
        from tractography import wma_fibers, wma_transform

        record, text = self._run(wma_transform, wma_transform.registration_transform_support,
                                 "register_two_subjects.py", "AUTHOR_SHA256")
        with tempfile.TemporaryDirectory() as td:
            import hashlib

            (Path(td) / "fibers.py").write_bytes(b"fibers source")
            vtk = types.ModuleType("vtk")
            vtk.vtkVersion = types.SimpleNamespace(GetVTKVersion=lambda: "9.6.2")
            err = io.StringIO()
            with patch.object(wma_fibers, "AUTHOR_SHA256", hashlib.sha256(b"fibers source").hexdigest()), \
                    patch.dict(sys.modules, {"vtk": vtk}), contextlib.redirect_stderr(err):
                wma_fibers.fiber_conversion_support(Path(td))
        self.assertEqual(text.count("using the reference path"), 1)
        self.assertEqual(err.getvalue(), "")

    def test_f6_qualified_vtk_logs_nothing(self) -> None:
        from tractography import wma_transform as module

        record, text = self._run(module, module.registration_transform_support, "register_two_subjects.py",
                                 "AUTHOR_SHA256", version="9.5.2")
        self.assertTrue(record["supported"])
        self.assertEqual(text, "")


if __name__ == "__main__":
    unittest.main()
