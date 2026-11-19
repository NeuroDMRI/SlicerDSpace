"""Keep public website artifacts separate from private docs and the extension."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.audit_pages_site import audit
from scripts.extension_runtime.runtime_resources import runtime_files
from scripts import release_audit

ROOT = Path(__file__).resolve().parents[2]


class PagesPublicationTest(unittest.TestCase):
    def test_current_public_site_passes_allowlist(self):
        result = audit(ROOT)
        self.assertEqual(result["status"], "pass", result)
        self.assertGreaterEqual(result["file_count"], 42)
        self.assertTrue((ROOT / 'docs/site/assets/img/fig-runtime-optimizations.png').is_file())
        self.assertFalse((ROOT / 'docs/site/assets/img/guide-bot-hat.svg').exists())
        self.assertFalse((ROOT / 'docs/site/assets/js/guide-bot.js').exists())
        self.assertFalse((ROOT / 'docs/site/assets/css/guide-bot.css').exists())

    def test_standalone_performance_page_is_not_publishable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            site = root / "docs/site"
            site.mkdir(parents=True)
            for name in (".nojekyll", "index.html", "404.html"):
                (site / name).touch()
            (site / "performance.html").write_text("Retired standalone page")
            result = audit(root)
            self.assertEqual(result["status"], "fail")
            self.assertIn("Unexpected public file: performance.html", result["findings"])

    def test_private_documents_and_symlinks_are_not_publishable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            site = root / "docs/site"
            site.mkdir(parents=True)
            for name in (".nojekyll", "index.html", "404.html"):
                (site / name).touch()
            private = root / "docs/private.md"
            private.write_text("Not a public website file")
            self.assertEqual(audit(root)["status"], "pass")
            (site / "internal.json").write_text("{}")
            self.assertEqual(audit(root)["status"], "fail")
            (site / "internal.json").unlink()
            data = site / 'assets/data'
            data.mkdir(parents=True)
            (data / 'private.json').write_text('{}')
            self.assertEqual(audit(root)['status'], 'fail')
            (data / 'private.json').unlink()
            images = site / 'assets/img'
            images.mkdir(parents=True)
            (images / 'guide-bot-hat.svg').write_text('<svg/>')
            self.assertIn('Unexpected public file: assets/img/guide-bot-hat.svg', audit(root)['findings'])
            (images / 'guide-bot-hat.svg').unlink()
            (images / 'unapproved.svg').write_text('<svg/>')
            self.assertIn('Unexpected public file: assets/img/unapproved.svg', audit(root)['findings'])
            (images / 'unapproved.svg').unlink()
            (site / "leak.txt").symlink_to(private)
            self.assertEqual(audit(root)["status"], "fail")

    def test_website_and_development_records_are_not_runtime_resources(self):
        self.assertFalse(any(path.startswith(("docs/site/", "docs/website/")) for path in runtime_files(ROOT)))

    def test_workflow_uploads_only_the_public_artifact(self):
        workflow = (ROOT / ".github/workflows/pages.yml").read_text()
        self.assertIn("path: docs/site", workflow)
        self.assertNotIn("path: docs\n", workflow)
        self.assertNotIn("path: .\n", workflow)
        self.assertIn("needs.pages-status.outputs.enabled == 'true'", workflow)

    def test_only_the_frozen_public_film_has_a_non_lfs_exception(self):
        self.assertTrue(release_audit._verified_public_film(release_audit.PUBLIC_FILM))
        self.assertFalse(release_audit._verified_public_film("other.mp4"))
        with patch.object(release_audit, "PUBLIC_FILM_SHA256", "0" * 64):
            self.assertFalse(release_audit._verified_public_film(release_audit.PUBLIC_FILM))


if __name__ == "__main__":
    unittest.main()
