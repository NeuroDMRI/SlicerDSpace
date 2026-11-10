"""Exercise the fresh-clone harness with tiny local Git fixtures, without assets."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(all(shutil.which(tool) for tool in ("bash", "git", "make", "python3")),
                     "Fresh-clone harness needs bash, Git, Make and python3")
class FreshCloneSmokeTest(unittest.TestCase):
    def run_fixture(self, directory: Path, *, fail: bool):
        source = directory / "source-with-arbitrary-name"
        source.mkdir()
        scripts = source / "scripts"
        scripts.mkdir()
        (scripts / "download_model_assets.py").write_text(
            'print("fixture asset plan; no model download")\n', encoding="utf-8")
        (source / "Makefile").write_text(
            "test-fast:\n\t@echo fixture stdout\n\t@echo fixture stderr >&2\n"
            + ("\t@exit 7\n" if fail else ""), encoding="utf-8")
        env = os.environ.copy()
        for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONNOUSERSITE"):
            env.pop(key, None)
        env.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull})
        for args in (("init",), ("add", "."),
                     ("-c", "user.name=Smoke Test", "-c", "user.email=smoke@example.invalid",
                      "commit", "-m", "Create harness fixture")):
            subprocess.run(["git", "-C", str(source), *args], env=env,
                           check=True, capture_output=True, text=True, timeout=30)
        revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"],
                                           env=env, text=True, timeout=30).strip()
        smoke_root = directory / "smoke"
        env.update({"SMOKE_REPO_URL": str(source), "SMOKE_REF": revision,
                    "SMOKE_ROOT": str(smoke_root), "SMOKE_ASSET_MODE": "plan",
                    "SMOKE_ASSET_PROFILE": "core",
                    "SMOKE_REPORT": str(smoke_root / "fresh_clone_smoke_report.json")})
        result = subprocess.run(["bash", str(ROOT / "scripts/smoke_fresh_clone.sh")],
                                env=env, capture_output=True, text=True, timeout=60)
        report = json.loads((smoke_root / "fresh_clone_smoke_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["resolved_commit"], revision)
        self.assertEqual(Path(report["clone_dir"]).name, "checkout")
        self.assertEqual([step["name"] for step in report["steps"]],
                         ["git_clone", "git_checkout", "model_assets_plan_core", "make_test_fast"])
        return result, report

    def test_success_reports_all_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            result, report = self.run_fixture(Path(directory), fail=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(report["status"], "passed")
            self.assertTrue(all(step["status"] == "passed" for step in report["steps"]))

    def test_failure_keeps_logs_visible_and_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            result, report = self.run_fixture(Path(directory), fail=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(report["status"], "failed")
            failure = report["steps"][-1]
            self.assertEqual(failure["status"], "failed")
            self.assertNotEqual(failure["exit_code"], 0)
            self.assertIn("Fresh-clone step failed: make_test_fast", result.stderr)
            self.assertIn("fixture stdout", result.stderr)
            self.assertIn("fixture stderr", result.stderr)
            self.assertIn("fixture stdout", Path(failure["stdout"]).read_text(encoding="utf-8"))
            self.assertIn("fixture stderr", Path(failure["stderr"]).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
