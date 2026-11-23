from pathlib import Path
import sys
import tempfile
import unittest
from win_test_helpers import skip_unless_symlinks
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from slicer_acceptance import private_slicer_home, private_user_settings


class AcceptanceIsolationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / "source"
        (self.source / "bin").mkdir(parents=True)
        (self.source / "bin/SlicerApp-real").touch()
        (self.source / "slicer.org").mkdir()
        (self.source / "slicer.org/Slicer-34045.ini").write_text("shared-settings")

    def prepare(self):
        with patch("slicer_acceptance.subprocess.check_output",
                   return_value=f"SLICER_HOME={self.source}\n"):
            return private_slicer_home("Slicer", self.root / "private")

    @skip_unless_symlinks
    def test_runtime_links_do_not_share_revision_settings(self):
        settings = self.prepare()
        self.assertEqual((self.root / "private/bin").resolve(), self.source / "bin")
        self.assertFalse((self.root / "private/slicer.org").is_symlink())
        self.assertFalse((self.root / "private/slicer.org/Slicer-34045.ini").exists())
        self.assertIn(f"SLICER_HOME={self.root / 'private'}", settings.read_text())
        self.assertEqual((self.source / "slicer.org/Slicer-34045.ini").read_text(), "shared-settings")

    @skip_unless_symlinks
    def test_rerun_reuses_only_matching_runtime_links(self):
        self.assertEqual(self.prepare(), self.prepare())

    def test_existing_runtime_directory_is_not_replaced(self):
        (self.root / "private/bin").mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, "Refusing to replace"):
            self.prepare()

    def test_missing_launcher_home_is_rejected(self):
        with patch("slicer_acceptance.subprocess.check_output", return_value="PATH=/bin\n"):
            with self.assertRaisesRegex(RuntimeError, "one SLICER_HOME"):
                private_slicer_home("Slicer", self.root / "private")

    def test_user_profile_is_offline_and_does_not_use_shared_dicom_database(self):
        settings = private_user_settings(self.root)
        content = settings.read_text()
        self.assertIn(f"DatabaseDirectory_0.8.1={self.root / 'DICOMDatabase'}", content)
        self.assertIn("AutoInstallDependencies=false", content)
        self.assertIn("AutoUpdateCheck=false", content)
        settings.write_text("existing-test-settings")
        private_user_settings(self.root)
        self.assertEqual(settings.read_text(), "existing-test-settings")


if __name__ == "__main__":
    unittest.main()
