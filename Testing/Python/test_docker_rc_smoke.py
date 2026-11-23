"""Contract tests for the isolated Linux release-candidate smoke runner."""

from __future__ import annotations

import shutil
import subprocess
import unittest
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "scripts" / "docker" / "run_rc_smoke.sh"
SLICER_INSTALL_SMOKE = REPO_ROOT / "Testing" / "Python" / "slicer_extension_install_smoke.py"


class DockerRCSmokeContractTest(unittest.TestCase):
    def test_runner_keeps_model_assets_outside_the_checkout(self) -> None:
        script = RUNNER.read_text(encoding="utf-8")

        self.assertIn('SMOKE_MODEL_ASSET_ROOT="${SMOKE_MODEL_ASSET_ROOT:-/work/model-assets}"', script)
        self.assertIn('target = asset_root / relative', script)
        self.assertIn('--asset-root "$SMOKE_MODEL_ASSET_ROOT" --print-plan', script)
        self.assertIn('--asset-root "$SMOKE_MODEL_ASSET_ROOT" --verify-only', script)
        self.assertIn('--model-assets-dir "$SMOKE_MODEL_ASSET_ROOT"', script)

    def test_runner_supports_a_no_cache_image_build(self) -> None:
        script = RUNNER.read_text(encoding="utf-8")

        self.assertIn('DOCKER_BUILD_NO_CACHE="${DOCKER_BUILD_NO_CACHE:-0}"', script)
        self.assertIn('DOCKER_BUILD_ARGS+=(--no-cache)', script)

    def test_install_smoke_cannot_block_on_first_run_dialogs(self) -> None:
        script = SLICER_INSTALL_SMOKE.read_text(encoding="utf-8")

        self.assertIn('update_user_config({"ui": {"first_run_wizard_shown": True}})', script)

    @unittest.skipIf(os.name == "nt", "POSIX bash syntax check runs on POSIX hosts")
    @unittest.skipUnless(shutil.which("bash"), "bash is required")
    def test_runner_has_valid_bash_syntax(self) -> None:
        completed = subprocess.run(
            ["bash", "-n", str(RUNNER)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
