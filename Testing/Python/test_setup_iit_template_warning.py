"""Finding IIT-WARN: 2026-11-27 real install: nitrc.org timed out, wizard still said Setup complete; PNL CNN blocked.

The installer fetches IITmean_b0_256.nii.gz from nitrc.org only. When that download fails, the old installers wrote one
[WARN] line and still ended with "Setup complete!". The PNL CNN brain-mask backend is blocked without the file, so the
user found out only later.

Decided behaviour: the install still finishes, but the end state says that the PNL CNN template is missing and what to
do. Three copies behave the same: the graphical Setup module (phase C and its all_done handler), the headless website
script (JS) and the Python script builder. The headless copies return the exit code 10; the .sh and .bat wrappers turn
it into a [WARN] line and exit 0.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TESTS_ROOT = Path(__file__).resolve().parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

import test_installer_prune_fallback as prune
from test_b4_w4_platform import _js_payload, _load, _python_payload
from test_installer_gui_reinstall_gap import JS_PATH, load_setup_module, setup_module_source

_Isolated, _Project, _on_ci = prune._Isolated, prune._Project, prune._on_ci

RELPATH = "brain_mask/pnl_cnn_torch/IITmean_b0_256.nii.gz"
TEMPLATE_BYTES = b"iit-template-test-bytes"
TEMPLATE_SHA = hashlib.sha256(TEMPLATE_BYTES).hexdigest()
SPEC = {"relpath": RELPATH, "sha256": TEMPLATE_SHA, "size_bytes": len(TEMPLATE_BYTES), "urls": []}

WARN_EN = ("The PNL CNN brain-mask template (IITmean_b0_256.nii.gz) was not installed: the download from nitrc.org "
           "failed. Before you generate a brain mask, open Environment and installation in SlicerDSpace to download "
           "it, or provide your own mask.")
WARN_ZH = ("PNL CNN 脑掩膜模板（IITmean_b0_256.nii.gz）未能安装：从 nitrc.org 下载失败。生成脑掩膜之前，"
           "请在 SlicerDSpace 的“环境与安装”中下载它，或提供自己的掩膜。")
UKF_NOTE = ("Bundled UKFTractography for linux-vtk9 was skipped for this Slicer VTK version; install SlicerDMRI via "
            "Extension Manager if UKF is needed.")
RESTART = "Restart 3D Slicer to open Modules -> Diffusion -> SlicerDSpace."
EXIT_WARN = 10
BUNDLED_ENV = "SLICERDSPACE_BUNDLED_MODEL_ASSETS_ROOT"


@contextlib.contextmanager
def _bundled_env(td: str, mode):
    """Codex P1: the plugin also reads model files from the directory in SLICERDSPACE_BUNDLED_MODEL_ASSETS_ROOT.
    mode: None or "unset" (variable removed), "with" (a directory that holds the valid template), "without" (a directory
    without it). The variable never leaks out of the test."""
    with patch.dict(os.environ):
        os.environ.pop(BUNDLED_ENV, None)
        if mode in ("with", "without"):
            root = Path(td) / "env-bundled-assets"
            root.mkdir()
            if mode == "with":
                _put_template(root)
            os.environ[BUNDLED_ENV] = str(root)
        yield


def _need_node() -> None:
    if shutil.which("node") is None:
        if _on_ci():
            raise AssertionError("node is required on CI for the JS installer copies (node is not in PATH)")
        raise unittest.SkipTest("node is not installed in PATH")


def _put_template(root: Path) -> Path:
    path = root / RELPATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(TEMPLATE_BYTES)
    return path


def _copies() -> dict:
    """The loaded Python text of the three copies that hold the shared helper."""
    _need_node()
    return {
        "python_builder": _load(_python_payload("linux")),
        "js_script": _load(_js_payload("linux")),
        "js_setup_module": load_setup_module(),
    }


class HelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.copies = _copies()

    def _each(self):
        for label, ns in self.copies.items():
            with self.subTest(copy=label):
                self.assertIn("_pnl_template_warning", ns, "the helper is missing")
                yield label, ns["_pnl_template_warning"]

    def test_iit_warn_missing_template_gives_the_warning_in_english_and_chinese(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            roots = [Path(td) / "a", Path(td) / "b"]
            for _label, fn in self._each():
                self.assertEqual(fn(True, SPEC, roots), WARN_EN)
                self.assertEqual(fn(True, SPEC, roots, "en"), WARN_EN)
                self.assertEqual(fn(True, SPEC, roots, language="zh"), WARN_ZH)
                self.assertEqual(fn(True, SPEC, roots, "zh-CN"), WARN_ZH)

    def test_iit_warn_three_copies_return_the_same_text(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            roots = [Path(td)]
            texts = {}
            for label, ns in self.copies.items():
                self.assertIn("_pnl_template_warning", ns, f"{label}: the helper is missing")
                fn = ns["_pnl_template_warning"]
                texts[label] = (fn(True, SPEC, roots, "en"), fn(True, SPEC, roots, "zh"))
            self.assertEqual(len(set(texts.values())), 1, texts)

    def test_iit_warn_template_present_in_any_root_gives_no_warning(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            first, second = Path(td) / "first", Path(td) / "second"
            first.mkdir()
            _put_template(second)
            for _label, fn in self._each():
                self.assertEqual(fn(True, SPEC, [first, second]), "")
                self.assertEqual(fn(True, SPEC, [second, first]), "")

    def test_iit_warn_pnl_not_selected_or_no_spec_gives_no_warning(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            for _label, fn in self._each():
                self.assertEqual(fn(False, SPEC, [Path(td)]), "")
                self.assertEqual(fn(True, {}, [Path(td)]), "")

    def test_iit_warn_wrong_sha256_on_disk_gives_the_warning(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bad = Path(td) / "bad"
            (bad / RELPATH).parent.mkdir(parents=True)
            (bad / RELPATH).write_bytes(b"truncated download")
            for _label, fn in self._each():
                self.assertEqual(fn(True, SPEC, [bad]), WARN_EN)

    def test_iit_warn_spec_without_sha256_accepts_any_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _put_template(Path(td))
            spec = {"relpath": RELPATH, "sha256": ""}
            for _label, fn in self._each():
                self.assertEqual(fn(True, spec, [Path(td)]), "")
                self.assertEqual(fn(True, spec, [Path(td) / "elsewhere"]), WARN_EN)

    def test_iit_warn_helper_text_is_ascii_in_the_shipped_copies(self) -> None:
        # The embedded Python is ASCII only: the Chinese text travels as \u escapes.
        _need_node()
        for label, source in (("js_script", _js_payload("linux")), ("js_setup_module", setup_module_source())):
            with self.subTest(copy=label):
                self.assertIn("def _pnl_template_warning(", source)
                start = source.index("def _pnl_template_warning(")
                end = source.index("\ndef ", start + 1)
                self.assertTrue(all(ord(ch) < 128 for ch in source[start:end]))

    def test_iit_warn_python_twin_script_stays_ascii(self) -> None:
        # The twin's payload sits inside the .sh/.bat file and was ASCII only before this fix (escapes, like its
        # code-update hint). A literal Chinese string would put non-ASCII bytes into a file that cmd.exe reads.
        source = _python_payload("linux")
        self.assertIn("def _pnl_template_warning(", source)
        self.assertEqual([ch for ch in source if ord(ch) > 127], [])


class CompletionMessagesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _need_node()
        cls.ns = load_setup_module()

    def _messages(self, skipped, warnings):
        self.assertIn("_completion_messages", self.ns, "_completion_messages is missing")
        return self.ns["_completion_messages"](skipped, warnings)

    def test_iit_warn_nothing_to_report_keeps_the_old_text(self) -> None:
        status, lines = self._messages([], [])
        self.assertEqual(status, "Setup complete! " + RESTART)
        self.assertEqual(lines, ["Setup finished successfully."])

    def test_iit_warn_skipped_cli_only_keeps_the_old_text(self) -> None:
        status, lines = self._messages(["linux-vtk9"], [])
        self.assertEqual(status, "Setup complete! " + UKF_NOTE)
        self.assertEqual(lines, ["[WARN] " + status, "Setup finished successfully."])

    def test_iit_warn_warning_only_does_not_say_complete(self) -> None:
        status, lines = self._messages([], [WARN_EN])
        self.assertTrue(status.startswith("Setup finished with a warning:"), status)
        self.assertIn(WARN_EN, status)
        self.assertNotIn("Setup complete!", status)
        self.assertTrue(status.rstrip().endswith(RESTART), status)
        self.assertEqual(lines, ["[WARN] " + WARN_EN, "Setup finished with warnings."])
        self.assertNotIn("Setup finished successfully.", lines)

    def test_iit_warn_warning_and_skipped_cli_show_both(self) -> None:
        status, lines = self._messages(["linux-vtk9"], [WARN_EN])
        self.assertTrue(status.startswith("Setup finished with a warning:"), status)
        self.assertIn(WARN_EN, status)
        self.assertIn("Bundled UKFTractography for linux-vtk9", status)
        self.assertIn("install SlicerDMRI via Extension Manager", status)
        self.assertNotIn("Setup complete!", status)
        self.assertTrue(status.rstrip().endswith(RESTART), status)
        warn_lines = [line for line in lines if line.startswith("[WARN] ")]
        self.assertEqual(len(warn_lines), 2, lines)
        self.assertTrue(any(WARN_EN in line for line in warn_lines), lines)
        self.assertTrue(any("UKFTractography" in line for line in warn_lines), lines)
        self.assertEqual(lines[-1], "Setup finished with warnings.")
        self.assertFalse([line for line in lines if "Setup complete!" in line], lines)

    def test_iit_warn_two_warnings_are_counted(self) -> None:
        status, lines = self._messages([], ["first problem.", "second problem."])
        self.assertTrue(status.startswith("Setup finished with 2 warnings:"), status)
        self.assertIn("first problem.", status)
        self.assertIn("second problem.", status)
        self.assertEqual([line for line in lines if line.startswith("[WARN] ")], ["[WARN] first problem.", "[WARN] second problem."])


def _phase_c(ns: dict, project: _Project, *, sel_model_packs, iit_file="", language="en") -> list:
    """Run the Setup module's phase C against a temp project; return its (kind, payload) events."""
    config = dict(ns["CONFIG"])
    config.update(model_assets_dir=str(project.assets), selected_packs=[], language=language,
                  artifacts=dict(config["artifacts"], iit_template=SPEC))
    ns["CONFIG"] = config
    events: list = []
    info = {
        "mod_root": str(project.module_root), "ext_temps": [], "mirror": "", "proxy": "", "offdir": "",
        "iit_file": str(iit_file) if iit_file else "", "pack_archives": [], "model_archives": [], "asset_temps": [],
        "sel_packs": [], "sel_model_packs": sel_model_packs,
    }
    ns["_run_phase_c_core"](info, lambda kind, payload: events.append((kind, payload)))
    return events


class PhaseCTests(unittest.TestCase):
    """The graphical Setup module: the all_done payload carries the warning (phase C can run in a worker process)."""

    @classmethod
    def setUpClass(cls) -> None:
        _need_node()

    def _run(self, *, sel_model_packs, on_disk: bool = False, downloaded: bool = False, language: str = "en",
             env_mode=None):
        ns = load_setup_module()
        with tempfile.TemporaryDirectory() as td, _bundled_env(td, env_mode):
            project = _Project(Path(td), release_package=True)
            if on_disk:
                _put_template(project.assets)
            iit_file = None
            if downloaded:
                iit_file = Path(td) / "downloaded" / "IITmean_b0_256.nii.gz"
                iit_file.parent.mkdir()
                iit_file.write_bytes(TEMPLATE_BYTES)
            with _Isolated(project, release_package=True):
                events = _phase_c(ns, project, sel_model_packs=sel_model_packs, iit_file=iit_file, language=language)
            errors = [p for k, p in events if k == "error"]
            self.assertEqual(errors, [])
            self.assertEqual(events[-1][0], "all_done", events[-3:])
            return events[-1][1], project.assets / RELPATH

    def test_iit_warn_phase_c_missing_template_carries_the_warning(self) -> None:
        payload, _path = self._run(sel_model_packs=["pnl_cnn", "ddseg"])
        self.assertEqual(payload.get("warnings"), [WARN_EN], payload)
        self.assertIn("skipped_cli_subdirs", payload)
        self.assertIn("mod_root", payload)

    def test_iit_warn_phase_c_default_selection_includes_pnl_cnn(self) -> None:
        payload, _path = self._run(sel_model_packs=None)
        self.assertEqual(payload.get("warnings"), [WARN_EN], payload)

    def test_iit_warn_phase_c_chinese_warning(self) -> None:
        payload, _path = self._run(sel_model_packs=["pnl_cnn"], language="zh")
        self.assertEqual(payload.get("warnings"), [WARN_ZH], payload)

    def test_iit_warn_phase_c_reinstall_with_template_on_disk_has_no_warning(self) -> None:
        payload, _path = self._run(sel_model_packs=["pnl_cnn"], on_disk=True)
        self.assertEqual(payload.get("warnings"), [], payload)

    def test_iit_warn_phase_c_downloaded_template_has_no_warning(self) -> None:
        payload, path = self._run(sel_model_packs=["pnl_cnn"], downloaded=True)
        self.assertEqual(payload.get("warnings"), [], payload)

    def test_iit_warn_codex_p1_phase_c_template_in_the_env_bundled_root_has_no_warning(self) -> None:
        payload, _path = self._run(sel_model_packs=["pnl_cnn"], env_mode="with")
        self.assertEqual(payload.get("warnings"), [], payload)

    def test_iit_warn_codex_p1_phase_c_env_root_without_the_template_or_unset_keeps_the_warning(self) -> None:
        for mode in ("without", "unset"):
            with self.subTest(env=mode):
                payload, _path = self._run(sel_model_packs=["pnl_cnn"], env_mode=mode)
                self.assertEqual(payload.get("warnings"), [WARN_EN], payload)

    def test_iit_warn_phase_c_pnl_cnn_not_selected_has_no_warning(self) -> None:
        payload, _path = self._run(sel_model_packs=["ddseg"])
        self.assertEqual(payload.get("warnings"), [], payload)


def _run_main(ns: dict, project: _Project, *, selected_model_packs, on_disk: bool, language: str = "en"):
    """Run main() of a generated script: the download fails, the real _sync_iit_template runs. Return (rc, logs)."""
    logs: list = []
    slicer_exe = project.root / "Slicer"
    slicer_exe.write_text("", encoding="utf-8")
    archive = project.root / "ext.zip"
    archive.write_bytes(b"PK")
    if on_disk:
        _put_template(project.assets)
    config = dict(ns["CONFIG"])
    config.update(model_assets_dir=str(project.assets), selected_packs=[], selected_model_packs=selected_model_packs,
                  mirror_prefix="", proxy_url="", offline_dir="", check_script_dir=False, language=language,
                  artifacts=dict(config["artifacts"], iit_template=SPEC))
    ns["CONFIG"] = config

    def failing_download(*_args, **_kwargs):
        raise OSError("timed out")

    ns["_log"] = logs.append
    ns["_download"] = failing_download
    ns["_pick_supported_slicer"] = lambda exe, explicit: exe
    ns["_warn_linux_glibc_floor"] = lambda *a, **k: None
    ns["_is_intel_mac"] = lambda: False
    ns["_pick_cache_dir"] = lambda need=0: project.root
    ns["_acquire_spec"] = lambda spec, cache_dir, *a, **k: (archive, [])
    ns["_install_extension_in_slicer" if "_install_extension_in_slicer" in ns else "_install_extension"] = lambda *a, **k: {}
    ns["_cleanup_temp_files"] = lambda paths: None
    ns["_find_module_root"] = lambda exe, receipt: project.module_root
    with patch.dict(os.environ, {"SLICERDSPACE_RESOLVED_SLICER": str(slicer_exe)}):
        rc = ns["main"]()
    return rc, logs


class HeadlessMainTests(unittest.TestCase):
    """The headless script (JS website copy and Python builder): the exit code 10 and the last log line."""

    @classmethod
    def setUpClass(cls) -> None:
        _need_node()
        cls.payloads = {"python_builder": _python_payload("linux"), "js_script": _js_payload("linux")}

    def _run(self, *, selected_model_packs, on_disk: bool, language: str = "en", env_mode=None):
        for label, payload in self.payloads.items():
            ns = _load(payload)
            with self.subTest(copy=label), tempfile.TemporaryDirectory() as td:
                project = _Project(Path(td), release_package=True)
                project.config_path.write_text('{"external_dependencies": {}}', encoding="utf-8")
                with _bundled_env(td, env_mode), _Isolated(project, release_package=True):
                    rc, logs = _run_main(ns, project, selected_model_packs=selected_model_packs, on_disk=on_disk,
                                         language=language)
                yield label, rc, logs

    def test_iit_warn_headless_missing_template_returns_the_warning_exit_code(self) -> None:
        for selection in (["pnl_cnn"], None):
            for label, rc, logs in self._run(selected_model_packs=selection, on_disk=False):
                with self.subTest(copy=label, selection=selection):
                    self.assertEqual(rc, EXIT_WARN, logs[-4:])
                    self.assertIn("[WARN] " + WARN_EN, logs)
                    self.assertEqual(logs[-1], "Installation completed with warnings.")
                    self.assertNotIn("Installation completed successfully.", logs)

    def test_iit_warn_headless_chinese_warning(self) -> None:
        for label, rc, logs in self._run(selected_model_packs=["pnl_cnn"], on_disk=False, language="zh"):
            with self.subTest(copy=label):
                self.assertEqual(rc, EXIT_WARN)
                self.assertIn("[WARN] " + WARN_ZH, logs)

    def test_iit_warn_headless_download_note_makes_no_unverified_claim(self) -> None:
        for label, rc, logs in self._run(selected_model_packs=["pnl_cnn"], on_disk=False):
            with self.subTest(copy=label):
                notes = [line for line in logs if "NITRC IIT template" in line and "download" in line]
                self.assertEqual(len(notes), 1, logs)
                self.assertTrue(notes[0].startswith("[WARN] "), notes)
                self.assertNotIn("first run", notes[0])

    def test_iit_warn_headless_template_present_returns_zero(self) -> None:
        for label, rc, logs in self._run(selected_model_packs=["pnl_cnn"], on_disk=True):
            with self.subTest(copy=label):
                self.assertEqual(rc, 0, logs[-4:])
                self.assertEqual(logs[-1], "Installation completed successfully.")
                self.assertFalse([line for line in logs if line.startswith("[WARN] The PNL CNN")], logs)

    def test_iit_warn_codex_p1_headless_template_in_the_env_bundled_root_returns_zero(self) -> None:
        for label, rc, logs in self._run(selected_model_packs=["pnl_cnn"], on_disk=False, env_mode="with"):
            with self.subTest(copy=label):
                self.assertEqual(rc, 0, logs[-4:])
                self.assertEqual(logs[-1], "Installation completed successfully.")

    def test_iit_warn_codex_p1_headless_env_root_without_the_template_or_unset_keeps_the_warning(self) -> None:
        for mode in ("without", "unset"):
            for label, rc, logs in self._run(selected_model_packs=["pnl_cnn"], on_disk=False, env_mode=mode):
                with self.subTest(copy=label, env=mode):
                    self.assertEqual(rc, EXIT_WARN, logs[-4:])
                    self.assertEqual(logs[-1], "Installation completed with warnings.")

    def test_iit_warn_headless_pnl_cnn_not_selected_returns_zero(self) -> None:
        for label, rc, logs in self._run(selected_model_packs=["ddseg"], on_disk=False):
            with self.subTest(copy=label):
                self.assertEqual(rc, 0, logs[-4:])
                self.assertEqual(logs[-1], "Installation completed successfully.")


def _scripts(target_os: str) -> dict:
    from shared.installer_script_builder import generate_installer_script

    out = {"python_builder": generate_installer_script(target_os=target_os, mode="base", slicer_path="", language="en")}
    _need_node()
    code = ("const ic = require(process.argv[1]);"
            "process.stdout.write(ic.buildScript({ os: '%s' }));" % target_os)
    out["js_script"] = subprocess.run(["node", "-e", code, str(JS_PATH)], capture_output=True, text=True,
                                      encoding="utf-8", check=True).stdout
    return out


class WrapperTests(unittest.TestCase):
    """The .sh and .bat wrappers: 0 is OK, 10 is a warning (the wrapper exits 0), any other code is an error."""

    def test_iit_warn_bat_wrappers_have_the_three_way_branch(self) -> None:
        for label, text in _scripts("windows").items():
            with self.subTest(copy=label):
                text = text.replace("\r\n", "\n")
                head = text.split("# __SLICERDSPACE_PY_BEGIN__", 1)[0]
                warn_line = "[WARN] Setup finished with warnings. Read the [WARN] lines above. Open 3D Slicer"
                self.assertIn(warn_line, head)
                ok = head.index("[OK] Setup complete")
                warn = head.index("[WARN] Setup finished with warnings. Read the [WARN] lines above. Open 3D Slicer")
                err = head.index("[ERROR] Installer exited with code %SD_RC%")
                self.assertLess(ok, warn)
                self.assertLess(warn, err)
                self.assertIn("%SD_RC%==10", head)
                between = head[warn:err]
                self.assertIn('set "SD_RC=0"', between)
                self.assertLess(head.index('set "SD_RC=0"', warn), head.index("exit /b %SD_RC%", warn))
                # The error branch still passes the real code on.
                self.assertIn("exit /b %SD_RC%", head)

    def test_iit_warn_sh_wrappers_run_the_three_way_branch(self) -> None:
        if os.name == "nt":
            self.skipTest("the .sh wrapper and its POSIX PythonSlicer stub do not run on Windows")
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is not installed")
        for label, text in _scripts("linux").items():
            for stub_rc, want_rc, want_text, not_text in (
                (0, 0, "[OK] Setup complete!", "[WARN]"),
                (EXIT_WARN, 0, "[WARN] Setup finished with warnings. Read the [WARN] lines above. Open 3D Slicer -> Modules -> Diffusion -> SlicerDSpace.", "[ERROR]"),
                (3, 3, "[ERROR] Installer exited with code 3.", "[OK]"),
            ):
                with self.subTest(copy=label, stub_rc=stub_rc), tempfile.TemporaryDirectory() as td:
                    home = Path(td) / "home"
                    (home / "Slicer-test" / "bin").mkdir(parents=True)
                    (home / "Slicer-test" / "Slicer").write_text("", encoding="utf-8")
                    py = home / "Slicer-test" / "bin" / "PythonSlicer"
                    py.write_text(f"#!/bin/sh\nexit {stub_rc}\n", encoding="utf-8")
                    py.chmod(0o755)
                    script = Path(td) / "install.sh"
                    script.write_text(text, encoding="utf-8")
                    env = {"HOME": str(home), "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                           "SLICERDSPACE_INSTALLER_NONINTERACTIVE": "1"}
                    done = subprocess.run([bash, str(script)], env=env, capture_output=True, text=True,
                                          stdin=subprocess.DEVNULL, timeout=60)
                    self.assertEqual(done.returncode, want_rc, done.stdout + done.stderr)
                    self.assertIn(want_text, done.stdout)
                    self.assertNotIn(not_text, done.stdout)


CI_HELPER = REPO_ROOT / ".github" / "scripts" / "ci_install_dspace.py"


class _FakePopen:
    """A stand-in for subprocess.Popen: prints two lines and returns the given exit code."""

    def __init__(self, code: int):
        self._code = code
        self.stdout = iter(["[SlicerDSpace Installer] [WARN] x\n", "last line\n"])

    def __call__(self, *args, **kwargs):
        return self

    def wait(self):
        return self._code


class CiInstallHelperTests(unittest.TestCase):
    """Opus B1: .github/scripts/ci_install_dspace.py runs the installer payload without the .sh/.bat wrapper. The
    warning exit code 10 must not fail the CI install step (a nitrc.org timeout passed before the warning existed)."""

    @classmethod
    def setUpClass(cls) -> None:
        from test_ci_compat_workflow import load

        cls.mod = load(CI_HELPER, "ci_install_dspace_iit_warning_under_test")

    def _run(self, code: int):
        from shared.installer_script_builder import load_installer_artifacts

        sha = load_installer_artifacts()["bundles"]["linux_x64_base"].get("sha256", "")
        logs: list = []
        with tempfile.TemporaryDirectory() as td, \
                patch.object(self.mod.subprocess, "Popen", _FakePopen(code)), \
                patch.object(self.mod, "log", logs.append):
            out = self.mod.run_website_installer(Path("/x/PythonSlicer"), Path("/x/Slicer"), "linux", Path(td),
                                                 "linux_x64_base", sha)
        return out, logs

    def test_iit_warn_opus_b1_exit_code_10_returns_the_output_and_logs_a_line(self) -> None:
        self.assertTrue(hasattr(self.mod, "INSTALLER_WARNING_EXIT_CODE"), "the named constant is missing")
        out, logs = self._run(10)
        self.assertIn("last line", out)
        self.assertTrue([line for line in logs if "warning" in line.lower() and "[WARN]" in line], logs)

    def test_iit_warn_opus_b1_exit_code_0_returns_and_other_codes_raise(self) -> None:
        out, _logs = self._run(0)
        self.assertIn("last line", out)
        with self.assertRaises(SystemExit) as caught:
            self._run(3)
        self.assertIn("exited with code 3", str(caught.exception))

    def test_iit_warn_opus_b1_constant_equals_the_twin_exit_code(self) -> None:
        self.assertTrue(hasattr(self.mod, "INSTALLER_WARNING_EXIT_CODE"), "the named constant is missing")
        twin = _load(_python_payload("linux"))["_EXIT_TEMPLATE_WARNING"]
        self.assertEqual(self.mod.INSTALLER_WARNING_EXIT_CODE, twin)
        if shutil.which("node") is not None:
            self.assertEqual(self.mod.INSTALLER_WARNING_EXIT_CODE, _load(_js_payload("linux"))["_EXIT_TEMPLATE_WARNING"])


class CompileTests(unittest.TestCase):
    def test_iit_warn_generated_modules_compile_for_every_os(self) -> None:
        _need_node()
        for os_key in ("linux", "windows", "macos"):
            with self.subTest(os=os_key):
                src = setup_module_source("{ os: '%s', wmaEnabled: false }" % os_key)
                compile(src, f"SlicerDSpaceSetup-{os_key}.py", "exec")
                compile(_js_payload(os_key), f"js-script-{os_key}.py", "exec")
                compile(_python_payload(os_key), f"py-script-{os_key}.py", "exec")


if __name__ == "__main__":
    unittest.main()
