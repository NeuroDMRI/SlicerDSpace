"""The website's "Download Slicer Setup" button must preselect the ORG atlas (fan93 clean install, 2026-11-22).

The page built its setup zip from a state with wmaEnabled false, so the wizard opened with no atlas ticked, while the
release's static setup zips and the paper's Equipment setup preselect ORG.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path

JS_PATH = Path(__file__).resolve().parents[2] / "docs" / "site" / "assets" / "js" / "installer-customizer.js"


def _embedded_config(os_key: str) -> dict:
    node = shutil.which("node")
    if node is None:
        raise unittest.SkipTest("node is not installed in PATH")
    code = ("const ic = require(process.argv[1]);"
            f"const st = ic.defaultSetupState('{os_key}');"
            "const f = ic.buildSetupExtensionFiles(st).find((x) => x.name.endsWith('SlicerDSpaceSetup.py'));"
            "process.stdout.write(f.data);")
    src = subprocess.run([node, "-e", code, str(JS_PATH)], capture_output=True, text=True, encoding="utf-8",
                         errors="strict", check=True).stdout
    line = next(l for l in src.splitlines() if l.startswith("CONFIG = _json.loads('"))
    return json.loads(line[len("CONFIG = _json.loads('"):-2])


class SetupDefaultAtlasTests(unittest.TestCase):
    def test_page_button_state_preselects_org_on_every_os(self) -> None:
        for os_key in ("linux", "windows", "macos"):
            with self.subTest(os=os_key):
                cfg = _embedded_config(os_key)
                self.assertEqual(cfg["selected_packs"], ["wma_org"])
                self.assertIs(cfg["auto_start"], False)

    def test_page_button_uses_the_default_state(self) -> None:
        # The helper alone is not enough: the page must build its state from it (review P2, 2026-11-22).
        source = JS_PATH.read_text(encoding="utf-8")
        self.assertIn("var state = defaultSetupState(detectOS());", source)
        start = source.index("function initCustomizer()")
        self.assertNotIn("wmaEnabled: false", source[start:start + 2000])


if __name__ == "__main__":
    unittest.main()
