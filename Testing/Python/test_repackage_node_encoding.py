"""Backlog item "Opus P2 (node encoding)": build_setup_zips decodes node output as UTF-8.

repackage_release_assets.build_setup_zips ran the node subprocess with text=True and no
encoding. Under an ASCII locale (CI runners), a non-ASCII node error message raised
UnicodeDecodeError, which hid the real RuntimeError. The fix passes encoding="utf-8" and
errors="replace" (a non-UTF-8 byte becomes U+FFFD and cannot hide the error either).

The test does not depend on the host locale. It patches the module's subprocess.run with a
fake that runs a real child process and forces encoding="ascii" when text=True is used
without an encoding argument.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / ".github" / "scripts" / "repackage_release_assets.py"

NON_ASCII_MESSAGE = "Fehler: Datei nicht gefunden ü"

_CHILD_CODE = (
    "import sys\n"
    f"sys.stderr.buffer.write({NON_ASCII_MESSAGE!r}.encode('utf-8'))\n"
    "sys.stderr.flush()\n"
    "sys.exit(1)\n"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("repackage_release_assets_encoding_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class RepackageNodeEncodingTests(unittest.TestCase):
    def test_node_non_ascii_error_is_reported_under_ascii_locale(self) -> None:
        mod = _load_module()
        real_run = subprocess.run
        node_calls = []

        def fake_run(cmd, *args, **kwargs):
            if cmd and cmd[0] == "fake-node":
                node_calls.append(list(cmd))
                if kwargs.get("text") and "encoding" not in kwargs:
                    kwargs["encoding"] = "ascii"  # simulate an ASCII locale
                return real_run([sys.executable, "-c", _CHILD_CODE], *args, **kwargs)
            return real_run(cmd, *args, **kwargs)

        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(mod.shutil, "which", return_value="fake-node"), \
                 mock.patch.object(mod.subprocess, "run", side_effect=fake_run):
                with self.assertRaises(RuntimeError) as ctx:
                    mod.build_setup_zips(Path(td), {}, published=False)
        self.assertEqual(len(node_calls), 1)
        self.assertIn(NON_ASCII_MESSAGE, str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
