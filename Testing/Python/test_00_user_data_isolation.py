"""B2: the test process never touches the real per-user SlicerDSpace data folder.

``unittest discover`` imports this module first (the name sorts before every other ``test_*.py``), so the isolation
holds for the whole run. Tests that run alone import ``user_data_isolation`` themselves.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import user_data_isolation

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HOME = user_data_isolation.install()


class UserDataIsolationTest(unittest.TestCase):
    def assertInHome(self, path) -> None:
        self.assertTrue(user_data_isolation.is_inside(path, HOME), f"{path} is outside {HOME}")

    def test_every_per_user_path_resolves_inside_the_temporary_home(self) -> None:
        from shared.managed_runtimes import runtime_root
        from shared.model_assets import _sha256_cache_path, default_user_model_assets_root
        from shared.user_config import default_user_config_path

        self.assertInHome(Path.home())
        self.assertInHome(Path("~/.cache").expanduser())
        self.assertInHome(default_user_config_path())
        self.assertInHome(_sha256_cache_path())
        self.assertInHome(default_user_model_assets_root())
        self.assertInHome(runtime_root() / "shims")

    def test_clear_true_keeps_the_isolation_variables(self) -> None:
        from shared.model_assets import _sha256_cache_path

        with patch.dict(os.environ, {"PATH": "/usr/bin"}, clear=True):
            self.assertInHome(Path.home())
            self.assertInHome(_sha256_cache_path())
        with patch.dict(os.environ, {}, clear=True):
            self.assertInHome(Path.home())
        self.assertInHome(Path.home())


if __name__ == "__main__":
    unittest.main()
