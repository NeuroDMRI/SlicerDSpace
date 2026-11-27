"""Finding PLUGIN-ROOTS-PERM (27 November 2026): bundled_model_asset_roots raised on an unreadable candidate folder.

On Python 3.12, ``Path.is_dir()`` raises ``PermissionError`` for a folder that it cannot traverse (confirmed on Slicer's
Python 3.12.10, errno 13). ``shared.model_assets.bundled_model_asset_roots`` probed each candidate without a guard, so
one unreadable ``bundled-model-assets`` folder broke every caller (brain mask, surface, model checks). The installer's
copy of this logic (``_pnl_template_roots``, f84c72dd) skips such a candidate; the plugin now does the same.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.model_assets import BUNDLED_MODEL_ASSETS_ENV, bundled_model_asset_roots


class UnreadableBundledCandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        if os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0):
            self.skipTest("needs POSIX permissions and a non-root user")
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        self.locked = self.root / "locked"
        (self.locked / "inner").mkdir(parents=True)
        self.ext = self.root / "ext"
        self.mod_root = self.ext / "lib" / "Slicer-5.10" / "qt-scripted-modules"
        self.mod_root.mkdir(parents=True)
        self.locked.chmod(0)
        self.addCleanup(self._td.cleanup)
        self.addCleanup(self.locked.chmod, 0o755)
        try:
            (self.locked / "inner").is_dir()
        except PermissionError:
            pass
        else:
            self.skipTest("this Python does not raise on an untraversable folder")

    def test_plugin_roots_perm_unreadable_qt_candidate_is_skipped_and_the_ext_folder_kept(self) -> None:
        (self.ext / "bundled-model-assets").mkdir()
        (self.mod_root / "bundled-model-assets").symlink_to(self.locked / "inner", target_is_directory=True)
        with patch.dict(os.environ):
            os.environ.pop(BUNDLED_MODEL_ASSETS_ENV, None)
            roots = bundled_model_asset_roots(self.mod_root)
        self.assertEqual(roots, ((self.ext / "bundled-model-assets").resolve(),))

    def test_plugin_roots_perm_unreadable_ext_candidate_is_skipped_and_the_qt_folder_kept(self) -> None:
        (self.ext / "bundled-model-assets").symlink_to(self.locked / "inner", target_is_directory=True)
        (self.mod_root / "bundled-model-assets").mkdir()
        with patch.dict(os.environ):
            os.environ.pop(BUNDLED_MODEL_ASSETS_ENV, None)
            roots = bundled_model_asset_roots(self.mod_root)
        self.assertEqual(roots, ((self.mod_root / "bundled-model-assets").resolve(),))

    def test_plugin_roots_perm_unreadable_env_folder_gives_no_root(self) -> None:
        (self.ext / "bundled-model-assets").mkdir()
        with patch.dict(os.environ, {BUNDLED_MODEL_ASSETS_ENV: str(self.locked / "inner")}):
            roots = bundled_model_asset_roots(self.mod_root)
        self.assertEqual(roots, ())


if __name__ == "__main__":
    unittest.main()
