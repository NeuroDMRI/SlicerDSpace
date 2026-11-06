"""W-B10: setup scripts are documented and worded with PythonSlicer, never a system Python."""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SYSTEM_PYTHON = re.compile(r"\bpython3?\s+\S*scripts[/\\](?:download_model_assets|setup_\w+)\.py")


class SetupScriptLauncherTests(unittest.TestCase):
    def test_no_setup_doc_or_message_requires_a_system_python(self):
        files = [ROOT / "README.md", ROOT / "README.zh-CN.md", ROOT / "shared/ui_environment_panel.py",
                 ROOT / ".agents/skills/slicerdspace/references/setup.md",
                 *sorted((ROOT / "docs/setup").glob("*.md")), *sorted((ROOT / "docs/assets").glob("*.md"))]
        offenders = [str(path.relative_to(ROOT)) for path in files if SYSTEM_PYTHON.search(path.read_text(encoding="utf-8"))]
        self.assertEqual(offenders, [])

    def test_environment_panel_command_uses_pythonslicer(self):
        text = (ROOT / "shared/ui_environment_panel.py").read_text(encoding="utf-8")
        self.assertIn("PythonSlicer scripts/download_model_assets.py --profile core", text)


if __name__ == "__main__":
    unittest.main()
