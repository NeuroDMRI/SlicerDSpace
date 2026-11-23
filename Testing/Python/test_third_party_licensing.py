#!/usr/bin/env python3
"""Tests for third-party licensing/notice files (no YAML dependency; see below).

CITATION.cff is checked with plain string assertions rather than by parsing
it as YAML: PyYAML is not installed in this repository's CI environment, and
this module must not introduce a new test dependency that is unavailable
there.
"""

from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


class LicenseFileTest(unittest.TestCase):
    def test_license_no_longer_links_the_gitignored_broken_path(self) -> None:
        text = (REPO_ROOT / "LICENSE.txt").read_text(encoding="utf-8")
        self.assertNotIn("third_party/REFERENCE_SOURCES.md", text)
        self.assertIn("THIRD_PARTY_NOTICES.md", text)


class ThirdPartyNoticesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = (REPO_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")

    def test_file_exists_at_repo_root(self) -> None:
        self.assertTrue((REPO_ROOT / "THIRD_PARTY_NOTICES.md").is_file())

    def test_documents_fastsurfer_derived_code(self) -> None:
        self.assertIn("FastSurfer", self.text)
        self.assertIn("anatomy/parcellation/models/networks.py", self.text)
        self.assertIn("anatomy/parcellation/data_loader/conform.py", self.text)
        self.assertIn("Apache", self.text)

    def test_documents_saf_backend(self) -> None:
        self.assertIn("saf_backend", self.text)
        self.assertIn("MIT", self.text)
        self.assertIn("10.1016/j.neuroimage.2022.119423", self.text)

    def test_documents_pnlbwh_conversion_honestly(self) -> None:
        self.assertIn("pnlbwh/conversion", self.text)
        self.assertIn("anatomy/parcellation/conversion", self.text)
        self.assertIn("BWH Conversion Contribution and Software License", self.text)
        self.assertIn("anatomy/parcellation/conversion/LICENSE.txt", self.text)
        self.assertIn("modified integration", self.text)
        self.assertIn("not the project's BSD-3-Clause license", self.text)

    def test_documents_freesurfer_color_lut(self) -> None:
        self.assertIn("FreeSurferColorLUT.txt", self.text)
        self.assertIn("FreeSurfer", self.text)

    def test_documents_all_three_verified_method_paper_dois(self) -> None:
        # All three DOIs were individually verified against Crossref
        # (https://api.crossref.org/works/<doi>) before being recorded here
        # and in CITATION.cff's `references:` section.
        self.assertIn("10.1002/advs.76596", self.text)
        self.assertIn("10.1016/j.neuroimage.2021.117934", self.text)
        self.assertIn("10.1109/TMI.2023.3331691", self.text)

    def test_no_longer_narrates_the_review_process(self) -> None:
        # This file documents third-party licensing/attribution facts, not
        # the history of the code review that prompted fixing them.
        self.assertNotIn("the review that produced this notice", self.text)
        self.assertNotIn(".gitignore", self.text)


class LicensesDirectoryTest(unittest.TestCase):
    def test_apache_license_text_is_present_and_genuine(self) -> None:
        path = REPO_ROOT / "licenses" / "Apache-2.0.txt"
        self.assertTrue(path.is_file())
        text = path.read_text(encoding="utf-8")
        self.assertIn("Apache License", text)
        self.assertIn("Version 2.0, January 2004", text)
        self.assertIn("TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION", text)
        # A genuine copy of the license is long; a stub/placeholder would not be.
        self.assertGreater(len(text.splitlines()), 100)

    def test_fastsurfer_notice_lists_every_derived_file(self) -> None:
        path = REPO_ROOT / "licenses" / "NOTICE-FastSurfer.txt"
        self.assertTrue(path.is_file())
        text = path.read_text(encoding="utf-8")
        for derived_file in (
            "anatomy/parcellation/models/networks.py",
            "anatomy/parcellation/models/sub_module.py",
            "anatomy/parcellation/data_loader/augmentation.py",
            "anatomy/parcellation/data_loader/conform.py",
            "anatomy/parcellation/data_loader/load_neuroimaging_data.py",
        ):
            self.assertIn(derived_file, text)
            # Every listed file must actually carry the header this notice
            # reproduces, so the notice cannot silently drift from the code.
            header = (REPO_ROOT / derived_file).read_text(encoding="utf-8")
            self.assertIn("Licensed under the Apache License, Version 2.0", header)


class FastSurferModifiedNoticesTest(unittest.TestCase):
    """Apache-2.0 Section 4(b) requires modified files to carry a notice."""

    def test_files_actually_changed_from_upstream_carry_a_modified_notice(self) -> None:
        for modified_file in (
            "anatomy/parcellation/models/networks.py",
            "anatomy/parcellation/models/sub_module.py",
            "anatomy/parcellation/data_loader/conform.py",
            "anatomy/parcellation/data_loader/load_neuroimaging_data.py",
        ):
            text = (REPO_ROOT / modified_file).read_text(encoding="utf-8")
            self.assertIn(
                "Modified by the SlicerDSpace project",
                text,
                f"{modified_file} was changed from its FastSurfer original but has no Apache-2.0 4(b) notice",
            )

    def test_unmodified_fastsurfer_files_are_not_falsely_marked_as_modified(self) -> None:
        # These carry the same Apache header but were not touched, so they
        # must not claim a modification that did not happen.
        for unmodified_file in (
            "anatomy/parcellation/data_loader/augmentation.py",
        ):
            text = (REPO_ROOT / unmodified_file).read_text(encoding="utf-8")
            self.assertNotIn("Modified by the SlicerDSpace project", text)


class CMakePackagingTest(unittest.TestCase):
    def setUp(self) -> None:
        from scripts.extension_runtime.runtime_resources import runtime_files
        self.files = runtime_files(REPO_ROOT)

    def test_saf_backend_license_is_packaged(self) -> None:
        self.assertIn("tractography/saf_backend/LICENSE.md", self.files)

    def test_third_party_notices_and_licenses_directory_are_packaged(self) -> None:
        self.assertIn("THIRD_PARTY_NOTICES.md", self.files)
        self.assertIn("licenses/Apache-2.0.txt", self.files)
        self.assertIn("licenses/NOTICE-FastSurfer.txt", self.files)
        self.assertIn("LICENSE.txt", self.files)
        self.assertNotIn("LICENSE", self.files)


class CitationCffTest(unittest.TestCase):
    def setUp(self) -> None:
        self.text = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")

    def test_has_a_references_section_with_ddsurfer(self) -> None:
        self.assertIn("references:", self.text)
        self.assertIn("DDSurfer", self.text)

    def test_records_the_verified_ddsurfer_doi(self) -> None:
        # Verified directly against Crossref
        # (https://api.crossref.org/works/10.1002/advs.76596) before being
        # recorded here as a `doi:` field rather than left as a TODO.
        self.assertIn('doi: "10.1002/advs.76596"', self.text)

    def test_no_longer_has_a_todo_for_the_doi(self) -> None:
        self.assertNotIn("TODO", self.text)

    def test_records_ddparcel_and_ddseg_method_paper_references(self) -> None:
        self.assertIn("DDParcel", self.text)
        self.assertIn('doi: "10.1109/TMI.2023.3331691"', self.text)
        self.assertIn('doi: "10.1016/j.neuroimage.2021.117934"', self.text)


if __name__ == "__main__":
    unittest.main()
