"""W5: static checks for the real-Slicer compatibility matrix (workflow, helpers, smoke script).

No Slicer, network or GitHub is needed. PyYAML is used when present; otherwise text checks run.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "slicer-compat-matrix.yml"
SCRIPTS = REPO / ".github" / "scripts"
SMOKE = REPO / "Testing" / "Python" / "slicer_ci_compat_smoke.py"
HELPERS = ("ci_install_slicer.py", "ci_install_dspace.py", "ci_run_slicer_smoke.py", "ci_glibc_scan.py", "ci_debug_state.py")

try:
    import yaml
except ImportError:  # pragma: no cover - depends on the interpreter
    yaml = None


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class WorkflowStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = WORKFLOW.read_text(encoding="utf-8")
        cls.doc = yaml.safe_load(cls.text) if yaml else None

    def test_trigger_is_manual_only(self):
        if self.doc is not None:
            on = self.doc.get("on", self.doc.get(True))  # YAML 1.1 parses the key "on" as True
            self.assertEqual(set(on), {"workflow_dispatch"})
        else:
            self.assertRegex(self.text, r"(?m)^on:\n  workflow_dispatch:")
        self.assertNotRegex(self.text, r"(?m)^  (push|pull_request|schedule):")

    def test_both_jobs_exist(self):
        if self.doc is not None:
            self.assertEqual(set(self.doc["jobs"]), {"repo-only", "full-install"})
        else:
            self.assertRegex(self.text, r"(?m)^  repo-only:")
            self.assertRegex(self.text, r"(?m)^  full-install:")

    def test_matrix_has_three_os_two_slicer_versions_in_each_job(self):
        jobs = self.doc["jobs"] if self.doc is not None else None
        for job in ("repo-only", "full-install"):
            if jobs is not None:
                strategy = jobs[job]["strategy"]
                self.assertIs(strategy["fail-fast"], False)
                self.assertEqual(set(strategy["matrix"]["os"]), {"ubuntu-22.04", "windows-2022", "macos-15"})
                self.assertEqual(set(strategy["matrix"]["slicer"]), {"5.10.0", "5.12.4"})
                self.assertEqual(jobs[job]["timeout-minutes"], 90)
        self.assertEqual(self.text.count("os: [ubuntu-22.04, windows-2022, macos-15]"), 2)
        self.assertEqual(self.text.count("slicer: ['5.10.0', '5.12.4']"), 2)
        self.assertEqual(self.text.count("fail-fast: false"), 2)
        self.assertEqual(self.text.count("timeout-minutes: 90"), 2)

    def test_unit_suite_is_report_only_with_40_minute_limit(self):
        self.assertEqual(self.text.count("continue-on-error: true"), 2)
        self.assertEqual(self.text.count("timeout-minutes: 40"), 2)
        self.assertEqual(self.text.count("-m unittest discover -s Testing/Python -p 'test_*.py'"), 2)

    def test_full_install_unit_suite_is_skipped_on_macos_after_runner_loss(self):
        # Run 44825474979: both macOS full-install jobs lost the runner in this step.
        step = "Unit test suite under the real PythonSlicer (report only)"
        guards = re.findall(re.escape(step) + r"\n\s+if: (.*)", self.text)
        self.assertEqual(len(guards), 2)
        self.assertNotIn("macOS", guards[0])
        self.assertIn("runner.os != 'macOS'", guards[1])

    def test_only_github_token_is_used(self):
        self.assertNotIn("secrets.", self.text)
        self.assertEqual(self.text.count("github.token"), 1)

    def test_every_job_uploads_artifacts_even_on_failure(self):
        self.assertEqual(self.text.count("actions/upload-artifact@v4"), 2)
        self.assertEqual(self.text.count("if: ${{ always() }}"), 2)

    def test_slicer_download_is_cached_by_os_and_version(self):
        self.assertEqual(self.text.count("key: slicer-dl-${{ matrix.os }}-${{ matrix.slicer }}"), 2)

    def test_rosetta_install_command(self):
        self.assertIn("softwareupdate --install-rosetta --agree-to-license", self.text)

    def test_full_install_runs_website_installer_overlay_smoke_and_glibc_scan(self):
        for needle in ("ci_install_dspace.py", "--mode full-install --pipeline required", "ci_glibc_scan.py", "--limit 2.34"):
            self.assertIn(needle, self.text)
        self.assertIn("--mode repo-only --pipeline optional", self.text)
        self.assertIn("scripts/stage_slicer_extension.py", self.text)

    def test_g1_glibc_limit_is_the_documented_linux_floor(self):
        self.assertNotIn("--limit 2.28", self.text)
        self.assertNotIn("allowed to be", self.text.split("on:\n", 1)[0])
        self.assertIn("judge_blocked_task", self.text.split("on:\n", 1)[0])
        scan = (SCRIPTS / "ci_glibc_scan.py").read_text(encoding="utf-8")
        self.assertIn('DEFAULT_LIMIT = "2.34"', scan)
        self.assertIn("documented Linux floor", scan)

    def test_uses_only_allowed_action_versions(self):
        for use in re.findall(r"uses: (\S+)", self.text):
            self.assertRegex(use, r"^actions/(checkout@v4|setup-python@v5|cache@v4|upload-artifact@v4)$")


class SyntaxTests(unittest.TestCase):
    def test_scripts_parse_as_python_39(self):
        for path in [SMOKE] + [SCRIPTS / name for name in HELPERS] + [Path(__file__)]:
            with self.subTest(path=path.name):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 9))


class SlicerDownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load(SCRIPTS / "ci_install_slicer.py", "ci_install_slicer_under_test")

    def test_revision_table_matches_repackage_script(self):
        text = (SCRIPTS / "repackage_release_assets.py").read_text(encoding="utf-8")
        pairs = set(re.findall(r'\("(5\.\d+)", "(\d+)"\)', text))  # MACOS_SLICER_REVISIONS lists several 5.12 revisions
        self.assertEqual(set(self.mod.REVISIONS), {"5.10", "5.12"})
        for minor, revision in self.mod.REVISIONS.items():
            self.assertIn((minor, revision), pairs)

    def test_api_os_names(self):
        self.assertEqual(self.mod.api_os_name("linux"), "linux")
        self.assertEqual(self.mod.api_os_name("win32"), "win")
        self.assertEqual(self.mod.api_os_name("darwin"), "macosx")
        with self.assertRaises(SystemExit):
            self.mod.api_os_name("freebsd")

    def _record(self, **over):
        record = {"download_url": "/bitstream/abc", "revision": "34045", "sha512": "x" * 128, "size": 10,
                  "version": "5.10.0", "name": "Slicer_linux_amd64_34045"}
        record.update(over)
        return record

    def test_validate_record_makes_url_absolute(self):
        out = self.mod.validate_record(self._record(), "linux", "5.10.0")
        self.assertEqual(out["download_url"], "https://download.slicer.org/bitstream/abc")

    def test_validate_record_rejects_wrong_revision(self):
        with self.assertRaises(SystemExit) as ctx:
            self.mod.validate_record(self._record(revision="99999"), "linux", "5.10.0")
        self.assertIn("revision", str(ctx.exception))

    def test_validate_record_rejects_wrong_version_and_missing_fields(self):
        with self.assertRaises(SystemExit):
            self.mod.validate_record(self._record(version="5.12.4"), "linux", "5.10.0")
        with self.assertRaises(SystemExit):
            self.mod.validate_record(self._record(sha512=""), "linux", "5.10.0")

    def test_find_release_reports_html_answer_with_cause(self):
        class Reply:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return b"<!DOCTYPE html><html></html>"

        with self.assertRaises(SystemExit) as ctx:
            self.mod.find_release("windows", "5.10.0", opener=lambda url: Reply())
        self.assertIn("win", str(ctx.exception))


class DspaceInstallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load(SCRIPTS / "ci_install_dspace.py", "ci_install_dspace_under_test")

    def test_base_asset_names_match_published_artifacts(self):
        artifacts = json.loads((REPO / "docs" / "website" / "installer_artifacts.json").read_text(encoding="utf-8"))
        for key, (target_os, template, bundle_key) in self.mod.BASE_ASSETS.items():
            with self.subTest(platform=key):
                self.assertEqual(artifacts["bundles"][bundle_key]["filename"], template % self.mod.RELEASE_TAG)
                self.assertEqual(artifacts["bundles"][bundle_key]["os"], target_os)

    def test_sha256sums_parser(self):
        table = self.mod.parse_sha256sums("%s  a.zip\n%s *b.tar.gz\nnot a line\n" % ("a" * 64, "B" * 64))
        self.assertEqual(table, {"a.zip": "a" * 64, "b.tar.gz": "b" * 64})

    def test_installed_module_dir_is_read_from_installer_output(self):
        out = "x\n[SlicerDSpace Installer] SlicerDSpace extension installed at: /a/b/lib/Slicer-5.12/qt-scripted-modules\ny\n"
        self.assertEqual(self.mod.find_installed_module_dir(out), Path("/a/b/lib/Slicer-5.12/qt-scripted-modules"))
        self.assertIsNone(self.mod.find_installed_module_dir("nothing"))

    def test_scheduled_uninstall_of_the_extension_is_cancelled_in_the_ini(self):
        # Run 44816716788: the smoke start printed 'Successfully uninstalled extension "SlicerDSpace"'.
        text = "[Extensions]\nScheduledForUninstall=SlicerDSpace, Other\nInstallPath=Extensions-34645\n"
        new, changed = self.mod.strip_scheduled_uninstall(text)
        self.assertTrue(changed)
        self.assertIn("ScheduledForUninstall=Other\n", new)
        self.assertIn("InstallPath=Extensions-34645", new)
        new, changed = self.mod.strip_scheduled_uninstall("ScheduledForUninstall=SlicerDSpace\r\n")
        self.assertEqual((new, changed), ("ScheduledForUninstall=\r\n", True))
        for untouched in ("ScheduledForUninstall=@Invalid()\n", "ScheduledForUninstall=SlicerDMRI\n", "x=1\n"):
            self.assertEqual(self.mod.strip_scheduled_uninstall(untouched), (untouched, False))

    def test_website_installer_generates_for_all_three_systems(self):
        sys.path.insert(0, str(REPO))
        try:
            from shared.installer_script_builder import generate_installer_script, load_installer_artifacts
        finally:
            sys.path.remove(str(REPO))
        artifacts = load_installer_artifacts()
        for target in ("linux", "windows", "macos"):
            with self.subTest(target=target):
                script = generate_installer_script(target_os=target, mode="base", slicer_path="/x/Slicer", selected_packs=(),
                                                   offline_dir="/x/off", check_script_dir=False, artifacts=artifacts)
                self.assertIn("__SLICERDSPACE_PY_BEGIN__", script)
                self.assertIn("__SLICERDSPACE_PY_END__", script)


class GlibcScanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load(SCRIPTS / "ci_glibc_scan.py", "ci_glibc_scan_under_test")

    SAMPLE = """
libfoo.so:     file format elf64-x86-64

DYNAMIC SYMBOL TABLE:
0000000000000000      DF *UND*  0000000000000000  GLIBC_2.2.5 memcpy
0000000000000000      DF *UND*  0000000000000000  GLIBC_2.4   __stack_chk_fail
0000000000000000      DF *UND*  0000000000000000  GLIBC_2.28  fcntl64
0000000000000000      DF *UND*  0000000000000000  GLIBCXX_3.4.21 _ZNSsC1Ev
0000000000000000      DF *UND*  0000000000000000  GLIBCXX_3.4.9 _ZdlPv
0000000000001000 g    DF .text  0000000000000010  GLIBC_2.34  provided_here_not_required
"""

    def test_parse_uses_undefined_symbols_only_and_integer_compare(self):
        self.assertEqual(self.mod.parse_objdump(self.SAMPLE), ("2.28", "3.4.21"))

    def test_two_digit_minor_beats_one_digit(self):
        self.assertEqual(self.mod.max_version(["2.4", "2.28", "2.9"]), "2.28")
        self.assertGreater(self.mod.version_tuple("2.28"), self.mod.version_tuple("2.4"))

    def test_violations_use_the_limit_and_ignore_glibcxx(self):
        rows = [{"file": "a", "glibc": "2.28", "glibcxx": "3.4.30"}, {"file": "b", "glibc": "2.29", "glibcxx": ""},
                {"file": "c", "glibc": "", "glibcxx": "3.4.9"}]
        self.assertEqual([r["file"] for r in self.mod.violations(rows, "2.28")], ["b"])
        self.assertIn("<-- above limit", self.mod.format_table(rows, "2.28"))

    def test_elf_detection_and_empty_root_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.so").write_bytes(b"\x7fELF" + b"\0" * 8)
            (root / "b.txt").write_text("hello", encoding="utf-8")
            self.assertEqual([p.name for p in self.mod.iter_elf_files([root])], ["a.so"])
            self.assertEqual(self.mod.main([str(root / "missing")]), 1)


class RunSmokeHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load(SCRIPTS / "ci_run_slicer_smoke.py", "ci_run_slicer_smoke_under_test")

    def test_linux_uses_xvfb_and_no_main_window(self):
        command = self.mod.build_command("/s/Slicer", "linux", "/m", "full-install", xvfb="/usr/bin/xvfb-run")
        self.assertEqual(command[:2], ["/usr/bin/xvfb-run", "-a"])
        self.assertIn("--no-main-window", command)
        self.assertNotIn("--additional-module-paths", command)

    def test_repo_only_registers_the_module_dir_and_windows_has_no_xvfb(self):
        command = self.mod.build_command("S.exe", "win32", "M", "repo-only")
        self.assertEqual(command[0], "S.exe")
        self.assertEqual(command[command.index("--additional-module-paths") + 1], "M")

    def test_missing_or_failed_report_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.json"
            self.assertFalse(self.mod.judge(path)[0])
            path.write_text(json.dumps({"status": "fail", "checks": [{"name": "x", "status": "fail", "detail": "d"}]}), encoding="utf-8")
            self.assertFalse(self.mod.judge(path)[0])
            path.write_text(json.dumps({"status": "pass", "checks": []}), encoding="utf-8")
            self.assertTrue(self.mod.judge(path)[0])


class SmokeScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load(SMOKE, "slicer_ci_compat_smoke_under_test")  # not __main__: no Qt timer starts

    def test_shadow_paths_are_purged_and_module_dir_goes_first(self):
        module_dir = "/repo/stage/SlicerDSpace/lib/Slicer-5.12/qt-scripted-modules"
        installed = "/opt/Slicer/Extensions-34645/SlicerDSpace/lib/Slicer-5.12/qt-scripted-modules"
        other = "/opt/Slicer/Extensions-34645/SlicerDMRI/lib/Slicer-5.12/qt-scripted-modules"
        new_path, removed = self.mod.purge_shadow_paths(["", "/usr/lib", installed, other, module_dir], module_dir)
        self.assertEqual(new_path[0], module_dir)
        self.assertEqual(new_path.count(module_dir), 1)
        self.assertEqual(removed, [installed])
        self.assertIn(other, new_path)  # an unrelated extension is not touched

    def test_installed_module_dir_is_not_purged_when_it_is_the_target(self):
        module_dir = "/opt/Slicer/Extensions-34645/SlicerDSpace/lib/Slicer-5.12/qt-scripted-modules"
        new_path, removed = self.mod.purge_shadow_paths([module_dir, "/usr/lib"], module_dir)
        self.assertEqual((new_path, removed), ([module_dir, "/usr/lib"], []))

    def test_worker_import_failure_names_the_package(self):
        text = "Traceback\n  File x\nModuleNotFoundError: No module named 'nibabel.nifti1'\n"
        self.assertEqual(self.mod.classify_worker_failure(text), "nibabel")
        self.assertEqual(self.mod.classify_worker_failure("ValueError: bad data"), "")

    def test_synthetic_subject_round_trips_and_has_expected_signal(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is not installed in this interpreter")
        with tempfile.TemporaryDirectory() as tmp:
            inp = self.mod.make_synthetic_subject(Path(tmp))
            dwi = self.mod.read_nifti_gz(inp / "data.nii.gz")
            mask = self.mod.read_nifti_gz(inp / "nodif_brain_mask.nii.gz")
            self.assertEqual(dwi.shape, (16, 16, 8, 7))
            self.assertEqual(mask.shape, (16, 16, 8))
            self.assertEqual(len((inp / "bvals").read_text().split()), 7)
            self.assertEqual(len((inp / "bvecs").read_text().strip().splitlines()), 3)
            self.assertTrue(np.allclose(dwi[..., 0], 1000.0))
            self.assertTrue(np.all(dwi[..., 1:] < 1000.0))

    def test_smoke_exits_through_slicer_util_exit_and_writes_report_incrementally(self):
        text = SMOKE.read_text(encoding="utf-8")
        self.assertIn("slicer.util.exit(code)", text)
        self.assertIn("os.replace", text)
        for key in ("numpy", "torch", "dipy", "vtk"):
            self.assertIn('"%s"' % key, text)
        self.assertIn("environment_doctor", text)

    def test_blocked_task_is_accepted_only_for_documented_missing_dependencies(self):
        # Run 44816716788 repo-only: exit 0, no FA, task blocked "preflight missing dependencies: dipy ...".
        # The CLI documents partial execution as the default (--strict-environment aborts instead).
        reason = "preflight missing dependencies: dipy (not found), nibabel (not found), nrrd (not found)"
        self.assertEqual(self.mod.missing_packages_from_reason(reason), ["dipy", "nibabel", "nrrd"])
        self.assertEqual(self.mod.missing_packages_from_reason("something else"), [])
        absent = {n: {"present": False} for n in ("dipy", "nibabel", "nrrd")}
        self.assertTrue(self.mod.judge_blocked_task("blocked", reason, absent)[0])
        present = dict(absent, dipy={"present": True})
        self.assertFalse(self.mod.judge_blocked_task("blocked", reason, present)[0])  # block is not explained
        self.assertFalse(self.mod.judge_blocked_task("blocked", reason, {})[0])  # packages were not probed
        self.assertFalse(self.mod.judge_blocked_task("blocked", "no GPU", absent)[0])
        self.assertFalse(self.mod.judge_blocked_task("completed", "", absent)[0])

    def test_task_outcome_is_read_from_the_orchestration_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self.mod.read_task_outcome(tmp)[0], "")
            meta = Path(tmp) / "outputs" / "00_run_metadata"
            meta.mkdir(parents=True)
            (meta / "Test1_orchestration_manifest.json").write_text(json.dumps(
                {"status": "completed_with_blocks", "tasks": [{"name": "microstructure.dti", "status": "blocked", "reason": "r"}]}))
            self.assertEqual(self.mod.read_task_outcome(tmp), ("blocked", "r"))
            self.assertEqual(self.mod.read_task_outcome(tmp, "other")[0], "")

    def test_missing_environment_fails_without_slicer(self):
        report = self.mod.REPORT
        saved = (dict(report), list(report["checks"]))
        try:
            with tempfile.TemporaryDirectory() as tmp, _env(SLICERDSPACE_CI_MODULE_DIR=tmp, SLICERDSPACE_CI_WORK=None,
                                                          SLICERDSPACE_CI_REPORT=None):
                self.assertEqual(self.mod.main(), 1)
                self.assertEqual(report["status"], "fail")
        finally:
            report.clear()
            report.update(saved[0])
            report["checks"] = saved[1]


class _env:
    """Set (or unset with None) environment variables for a with-block."""

    def __init__(self, **values):
        self.values = values
        self.saved = {}

    def __enter__(self):
        import os

        for key, value in self.values.items():
            self.saved[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        return self

    def __exit__(self, *exc):
        import os

        for key, value in self.saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        return False


if __name__ == "__main__":
    unittest.main()
