#!/usr/bin/env python3
"""Finding "batch scan: subject resolves to the Batch Folder".

``discover_subjects`` used to let a subfolder without input fall back to the Batch Folder itself, so an empty
subfolder became a subject whose input was the whole Batch Folder (results in ``<Batch Folder>/outputs``).
The rule now: a subject never resolves to the Batch Folder itself, except mode ``legacy_registered``, a file
entry of ``--subject-list`` that is input itself and a subject-list entry that is the Batch Folder itself.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from win_test_helpers import skip_unless_symlinks
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_batch_pipeline.py"
NEW_REASON = "no input in this subfolder; the input found is the Batch Folder itself"
OLD_REASON = "missing required input/data.nii.gz, bvals, or bvecs"
# Opus (scan fix) P2-3: a scanned candidate that is the Batch Folder itself gets this neutral reason.
SELF_REASON = "this subfolder is the Batch Folder itself"

spec = importlib.util.spec_from_file_location("_batch_scan_guard_batch", SCRIPT)
assert spec is not None and spec.loader is not None
batch_cli = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = batch_cli
spec.loader.exec_module(batch_cli)


def write_canonical(folder: Path) -> None:
    (folder / "input").mkdir(parents=True, exist_ok=True)
    for name in ("data.nii.gz", "bvals", "bvecs"):
        (folder / "input" / name).write_bytes(b"x")


def write_dicom(folder: Path) -> None:
    (folder / "series1").mkdir(parents=True, exist_ok=True)
    (folder / "series1" / "IM0001").write_bytes(b"\0" * 128 + b"DICM" + b"\0" * 64)


def write_direct_nifti(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("dwi.nii.gz", "dwi.bval", "dwi.bvec"):
        (folder / name).write_bytes(b"x")


def write_registered(folder: Path, sid: str) -> None:
    for suffix in ("dti-FractionalAnisotropy-Reg.nii.gz", "dti-MinEigenvalue-Reg.nii.gz",
                   "dti-MidEigenvalue-Reg.nii.gz", "dti-MaxEigenvalue-Reg.nii.gz", "dti-Trace-Reg.nii.gz",
                   "dti-MeanDiffusivity-Reg.nii.gz", "mask-Reg.nii.gz", "b0ToAtlasT2.tfm"):
        (folder / f"{sid}-{suffix}").write_bytes(b"x")


class BatchFolderGuardTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.batch = self.root / "batch"
        self.batch.mkdir()

    def reasons(self, skipped):
        return {item["subject_id"]: item["reason"] for item in skipped}

    def test_empty_subfolder_next_to_a_dicom_subfolder_is_skipped(self):
        (self.batch / "sub-b").mkdir()
        write_dicom(self.batch / "sub-a")
        subjects, skipped = batch_cli.discover_subjects(self.batch)
        by_id = {subject.subject_id: subject for subject in subjects}
        self.assertEqual(set(by_id), {"sub-a"})
        self.assertEqual(by_id["sub-a"].mode, "dicom")
        self.assertEqual(by_id["sub-a"].subject_root, (self.batch / "sub-a").resolve())
        self.assertEqual(self.reasons(skipped), {"sub-b": NEW_REASON})

    def test_direct_nifti_in_the_batch_folder_plus_an_empty_subfolder_gives_no_subject(self):
        write_direct_nifti(self.batch)
        (self.batch / "sub-b").mkdir()
        subjects, skipped = batch_cli.discover_subjects(self.batch)
        self.assertEqual(subjects, [])
        self.assertEqual(self.reasons(skipped), {"sub-b": NEW_REASON})

    def test_batch_folder_that_is_a_subject_root_gives_no_subject(self):
        write_canonical(self.batch)
        (self.batch / "outputs").mkdir()
        subjects, skipped = batch_cli.discover_subjects(self.batch)
        self.assertEqual(subjects, [])
        self.assertEqual(self.reasons(skipped), {"input": NEW_REASON, "outputs": NEW_REASON})

    def test_legacy_registered_files_in_the_batch_folder_stay_accepted(self):
        write_registered(self.batch, "sub-l")
        for make_folder in (False, True):
            if make_folder:
                (self.batch / "sub-l").mkdir()
            listing = self.root / "list.txt"
            listing.write_text("sub-l\n", encoding="utf-8")
            subjects, skipped = batch_cli.discover_subjects(self.batch, subject_list=listing)
            self.assertEqual([(s.subject_id, s.mode) for s in subjects], [("sub-l", "legacy_registered")], make_folder)
            self.assertEqual(skipped, [], make_folder)

    def test_include_missing_keeps_the_rejected_subject_with_mode_missing(self):
        write_dicom(self.batch / "sub-a")
        (self.batch / "sub-b").mkdir()
        subjects, skipped = batch_cli.discover_subjects(self.batch, include_missing=True)
        modes = {subject.subject_id: subject.mode for subject in subjects}
        self.assertEqual(modes, {"sub-a": "dicom", "sub-b": "missing"})
        self.assertEqual(self.reasons(skipped), {"sub-b": NEW_REASON})

    def test_plain_missing_subfolder_keeps_the_old_reason(self):
        (self.batch / "sub-b").mkdir()
        subjects, skipped = batch_cli.discover_subjects(self.batch)
        self.assertEqual(subjects, [])
        self.assertEqual(self.reasons(skipped), {"sub-b": OLD_REASON})

    def test_subject_list_file_entry_in_the_batch_folder_keeps_its_behaviour(self):
        write_direct_nifti(self.batch)
        listing = self.root / "list.txt"
        listing.write_text("dwi.nii.gz\n", encoding="utf-8")
        subjects, skipped = batch_cli.discover_subjects(self.batch, subject_list=listing)
        # Pinned from the unchanged code: the file's parent (the Batch Folder) is the root; mode nifti.
        self.assertEqual([(s.subject_id, s.subject_root) for s in subjects], [("dwi.nii.gz", self.batch)])
        self.assertEqual(skipped, [])

    def test_subject_list_entry_that_is_the_batch_folder_keeps_its_behaviour(self):
        write_canonical(self.batch)
        listing = self.root / "list.txt"
        listing.write_text(f"{self.batch}\n", encoding="utf-8")
        subjects, skipped = batch_cli.discover_subjects(self.batch, subject_list=listing)
        self.assertEqual([(s.subject_id, s.subject_root, s.mode) for s in subjects],
                         [("batch", self.batch, "subject_root")])
        self.assertEqual(skipped, [])

    @skip_unless_symlinks
    def test_review_b1_scanned_alias_of_the_batch_folder_is_not_a_subject(self):
        """Review B1 (Opus) / P1 (Codex): the "entry is the Batch Folder itself" exemption is for --subject-list only.

        In scan (glob) mode a symlink back to the Batch Folder resolved to the Batch Folder and was exempt, so the
        Batch Folder became a subject again.
        """
        write_direct_nifti(self.batch)
        (self.batch / "alias").symlink_to(self.batch, target_is_directory=True)
        subjects, skipped = batch_cli.discover_subjects(self.batch)
        self.assertEqual(subjects, [])
        self.assertEqual(self.reasons(skipped), {"batch": SELF_REASON})

    @skip_unless_symlinks
    def test_review_b1_scanned_alias_beside_a_dicom_subject(self):
        """Review B1: the alias must not take the DICOM sibling's files through the Batch Folder."""
        write_dicom(self.batch / "sub-a")
        (self.batch / "loop").symlink_to(self.batch, target_is_directory=True)
        subjects, skipped = batch_cli.discover_subjects(self.batch)
        self.assertEqual([(s.subject_id, s.subject_root, s.mode) for s in subjects],
                         [("sub-a", (self.batch / "sub-a").resolve(), "dicom")])
        self.assertEqual(self.reasons(skipped), {"batch": SELF_REASON})

    def test_review_b1_glob_that_selects_the_batch_folder_itself(self):
        """Review B1: a recursive glob yields the Batch Folder itself as a candidate; it is not a subject."""
        write_canonical(self.batch)
        subjects, skipped = batch_cli.discover_subjects(self.batch, subject_glob="**")
        self.assertNotIn(self.batch.resolve(), [s.subject_root for s in subjects])
        self.assertEqual(self.reasons(skipped).get("batch"), SELF_REASON)

    def test_review_help_legacy_files_keep_an_empty_subfolder_in_scan_mode(self):
        """Review P1 (Codex) / P2-5 (Opus): supported fallbacks keep an empty subfolder as a subject (scan mode)."""
        (self.batch / "sub-l").mkdir()
        write_registered(self.batch, "sub-l")
        subjects, skipped = batch_cli.discover_subjects(self.batch)
        self.assertEqual([(s.subject_id, s.mode) for s in subjects], [("sub-l", "legacy_registered")])
        self.assertEqual(skipped, [])

    def test_review_help_input_in_the_output_folder_keeps_an_empty_subfolder(self):
        """Review P1 (Codex) / P2-5 (Opus): input prepared in the Output Folder keeps an empty subfolder."""
        (self.batch / "sub-01").mkdir()
        out = self.root / "out"
        write_canonical(out / "sub-01")
        subjects, skipped = batch_cli.discover_subjects(self.batch, output_root=out)
        self.assertEqual([(s.subject_id, s.subject_root) for s in subjects], [("sub-01", (out / "sub-01").resolve())])
        self.assertEqual(skipped, [])

    def test_review_help_sentence_is_true_for_the_supported_fallbacks(self):
        """Review P1 (Codex) / P2-5 (Opus): "a subfolder with no readable input is skipped" was false for the two
        fallbacks above. The help now says a subfolder is skipped when the scan finds no input for it."""
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from shared import ui_onboarding

        for lang, sentence, old in (
            ("en", "A subfolder is skipped when the scan finds no input for it.",
             "A subfolder with no readable input is skipped."),
            ("zh", "扫描找不到输入的子文件夹会被跳过。", "没有可读取输入的子文件夹会被跳过。"),
        ):
            text = ui_onboarding.onboarding_text(lang, "help_batch_folder")
            self.assertIn(sentence, text, lang)
            self.assertNotIn(old, text, lang)

    def test_nested_subject_list_entry_stays_accepted(self):
        write_canonical(self.batch / "group" / "sub-n")
        listing = self.root / "list.txt"
        # A subject-list entry that names a nested folder by path is not the Batch Folder itself.
        listing.write_text("group/sub-n\n", encoding="utf-8")
        subjects, skipped = batch_cli.discover_subjects(self.batch, subject_list=listing)
        self.assertEqual([(s.subject_id, s.subject_root) for s in subjects],
                         [("sub-n", (self.batch / "group" / "sub-n").resolve())])

    def test_normal_cohort_is_unchanged(self):
        write_canonical(self.batch / "sub-01")
        write_canonical(self.batch / "sub-02")
        (self.batch / "notes").mkdir()
        subjects, skipped = batch_cli.discover_subjects(self.batch)
        self.assertEqual({s.subject_id: s.mode for s in subjects}, {"sub-01": "subject_root", "sub-02": "subject_root"})
        self.assertEqual(self.reasons(skipped), {"notes": OLD_REASON})

    def run_main(self, *extra):
        fake = self.root / "pipeline.py"
        fake.write_text("import sys\nsys.exit(0)\n", encoding="utf-8")
        manifest = self.root / "batch.json"
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            batch_cli.main(["--batch-root", str(self.batch), "--output-root", str(self.root / "out"),
                            "--pipeline-script", str(fake), "--python-executable", sys.executable,
                            "--manifest", str(manifest), *extra])
        lines = [line for line in out.getvalue().splitlines() if line.startswith("[Batch] Skipped ")]
        return lines, json.loads(manifest.read_text())

    def test_main_prints_one_skipped_line_per_skipped_subject(self):
        write_canonical(self.batch / "sub-01")
        write_dicom(self.batch / "sub-d")
        (self.batch / "sub-b").mkdir()
        lines, data = self.run_main("--subject-limit", "1")
        # A DICOM file in sub-d makes the Batch Folder look like DICOM input, so sub-b is the guarded case.
        self.assertEqual(lines, [f"[Batch] Skipped sub-b: {NEW_REASON}"])  # no line for the "subject limit" skip
        self.assertIn("subject limit", {item["reason"] for item in data["skipped"]})

    def test_main_prints_the_old_reason_for_a_plain_missing_subfolder(self):
        write_canonical(self.batch / "sub-01")
        (self.batch / "notes").mkdir()
        lines, _ = self.run_main()
        self.assertEqual(lines, [f"[Batch] Skipped notes: {OLD_REASON}"])


def write_raw_files_directly(folder: Path) -> None:
    """The four names that make ``_arrange_subject_input`` fire (``write_direct_nifti`` does not)."""
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
        (folder / name).write_bytes(b"x")


class BatchScanGuardRowsTest(unittest.TestCase):
    """Backlog rows "22 November 2026: batch scan guard": Opus (scan) P2-1..P2-4, Opus (scan re-review) P2-1..P2-2."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.batch = self.root / "batch"
        self.batch.mkdir()

    def reasons_by_path(self, skipped):
        return {item["path"]: item["reason"] for item in skipped}

    @skip_unless_symlinks
    def test_opus_scan_p2_1_symlink_to_the_parent_is_not_a_subject(self):
        write_dicom(self.batch / "sub-a")
        (self.batch / "up").symlink_to(self.batch.parent, target_is_directory=True)
        subjects, skipped = batch_cli.discover_subjects(self.batch)
        for subject in subjects:
            self.assertNotIn(subject.subject_root, [self.batch, *self.batch.parents], subject)
        self.assertEqual([s.subject_id for s in subjects], ["sub-a"])
        self.assertEqual(self.reasons_by_path(skipped), {self.root.as_posix(): batch_cli.BATCH_FOLDER_ANCESTOR_REASON})

    def test_opus_scan_p2_2_nested_glob_runs_one_subject_per_root(self):
        write_canonical(self.batch / "sub-b")
        (self.batch / "site1" / "sub-b").mkdir(parents=True)
        subjects, skipped = batch_cli.discover_subjects(self.batch, subject_glob="*/*")
        self.assertEqual([(s.subject_id, s.subject_root) for s in subjects], [("sub-b", self.batch / "sub-b")])
        self.assertEqual(
            self.reasons_by_path(skipped),
            {(self.batch / "sub-b" / "input").as_posix(): f"same input as subject sub-b: {self.batch / 'sub-b'}"},
        )

    def test_opus_scan_p2_2_include_missing_keeps_the_duplicate_as_missing_in_candidate_order(self):
        write_canonical(self.batch / "sub-b")
        (self.batch / "site1" / "sub-b").mkdir(parents=True)
        subjects, _ = batch_cli.discover_subjects(self.batch, subject_glob="*/*", include_missing=True)
        self.assertEqual([(s.subject_id, s.mode) for s in subjects], [("sub-b", "subject_root"), ("input", "missing")])

    def test_opus_scan_p2_2_survivor_is_the_id_named_like_the_folder_even_when_it_comes_second(self):
        write_canonical(self.batch / "sub-b")
        listing = self.root / "list.txt"
        listing.write_text("sub-b/input\nsub-b\n", encoding="utf-8")
        subjects, skipped = batch_cli.discover_subjects(self.batch, subject_list=listing)
        self.assertEqual([(s.subject_id, s.subject_root) for s in subjects], [("sub-b", self.batch / "sub-b")])
        self.assertEqual([item["subject_id"] for item in skipped], ["input"])

    def test_opus_scan_p2_3_help_documents_the_same_id_lookup(self):
        # argparse wraps the help text (also after hyphens), so compare without any whitespace.
        text = "".join(batch_cli.build_parser().format_help().split())
        for sentence in ("a subject folder (absolute or relative to --batch-root) or a subject ID",
                         "An entry with no input of its own is looked up as <batch-root>/<entry name>"):
            self.assertIn("".join(sentence.split()), text)

    def test_opus_scan_p2_4_text_file_entry_does_not_take_the_batch_folder_as_input(self):
        write_dicom(self.batch / "sub-a")
        (self.batch / "sub-b").mkdir()
        (self.batch / "sub-b" / "notes.txt").write_text("not an image\n", encoding="utf-8")
        listing = self.root / "list.txt"
        listing.write_text("sub-a\nsub-b/notes.txt\n", encoding="utf-8")
        subjects, skipped = batch_cli.discover_subjects(self.batch, subject_list=listing)
        self.assertEqual([s.subject_id for s in subjects], ["sub-a"])
        self.assertEqual(self.reasons_by_path(skipped), {(self.batch / "sub-b" / "notes.txt").as_posix(): NEW_REASON})

    @skip_unless_symlinks
    def test_opus_scan_rr_p2_1_alias_of_the_batch_folder_is_rejected_before_arrange(self):
        write_raw_files_directly(self.batch)
        (self.batch / "alias").symlink_to(self.batch, target_is_directory=True)
        subjects, skipped = batch_cli.discover_subjects(self.batch, arrange_inputs="copy")
        self.assertEqual(subjects, [])
        self.assertEqual(self.reasons_by_path(skipped), {self.batch.as_posix(): SELF_REASON})
        self.assertFalse((self.batch / "input").exists())

    @skip_unless_symlinks
    def test_opus_scan_rr_p2_2_alias_with_output_root_writes_nothing_and_gives_no_subject(self):
        write_raw_files_directly(self.batch)
        (self.batch / "alias").symlink_to(self.batch, target_is_directory=True)
        out = self.root / "out"
        subjects, skipped = batch_cli.discover_subjects(self.batch, arrange_inputs="copy", output_root=out)
        self.assertEqual(subjects, [])
        self.assertFalse((out / "batch").exists())
        self.assertEqual(self.reasons_by_path(skipped), {self.batch.as_posix(): SELF_REASON})

    def test_opus_scan_rr_p2_2_recursive_glob_gives_one_subject_for_a_nested_input_folder(self):
        write_canonical(self.batch / "sub-1")
        subjects, skipped = batch_cli.discover_subjects(self.batch, subject_glob="**")
        self.assertEqual([(s.subject_id, s.subject_root) for s in subjects], [("sub-1", self.batch / "sub-1")])
        self.assertEqual(len(skipped), 2, skipped)

    def test_two_nifti_file_entries_in_one_folder_stay_two_subjects(self):
        folder = self.batch / "sub-c"
        folder.mkdir()
        for stem in ("a", "b"):
            for suffix in (".nii.gz", ".bval", ".bvec"):
                (folder / f"{stem}{suffix}").write_bytes(b"x")
        listing = self.root / "list.txt"
        listing.write_text("sub-c/a.nii.gz\nsub-c/b.nii.gz\n", encoding="utf-8")
        subjects, skipped = batch_cli.discover_subjects(self.batch, subject_list=listing)
        self.assertEqual([(s.subject_id, s.subject_root) for s in subjects],
                         [("a.nii.gz", folder), ("b.nii.gz", folder)])
        self.assertEqual(skipped, [])

    def test_two_legacy_registered_subjects_in_the_batch_folder_are_both_kept(self):
        write_registered(self.batch, "sub-l1")
        write_registered(self.batch, "sub-l2")
        listing = self.root / "list.txt"
        listing.write_text("sub-l1\nsub-l2\n", encoding="utf-8")
        subjects, skipped = batch_cli.discover_subjects(self.batch, subject_list=listing)
        self.assertEqual([(s.subject_id, s.mode) for s in subjects],
                         [("sub-l1", "legacy_registered"), ("sub-l2", "legacy_registered")])
        self.assertEqual(skipped, [])

    def test_subject_list_entry_that_is_an_ancestor_of_the_batch_folder_is_not_rejected(self):
        """Design: the subject-list exemption stays; only scan mode rejects ancestors."""
        write_canonical(self.root)
        listing = self.root / "list.txt"
        listing.write_text(f"{self.root}\n", encoding="utf-8")
        subjects, _ = batch_cli.discover_subjects(self.batch, subject_list=listing)
        self.assertEqual([s.subject_root for s in subjects], [self.root])

    def test_opus_scan_p2_1_inspected_root_above_the_batch_folder_is_rejected_in_scan_mode_only(self):
        """The post-inspection guard: an inspected root above the Batch Folder (not reachable by a plain folder)."""
        (self.batch / "sub-x").mkdir()
        fake = {"mode": "nifti", "subject_root": self.root, "subject_dir": self.root}
        original = batch_cli.inspect_subject_directory
        batch_cli.inspect_subject_directory = lambda *args, **kwargs: dict(fake)
        self.addCleanup(setattr, batch_cli, "inspect_subject_directory", original)
        subjects, skipped = batch_cli.discover_subjects(self.batch)
        self.assertEqual(subjects, [])
        self.assertEqual([item["reason"] for item in skipped], [batch_cli.BATCH_FOLDER_ANCESTOR_REASON])
        listing = self.root / "list.txt"
        listing.write_text("sub-x\n", encoding="utf-8")
        subjects, _ = batch_cli.discover_subjects(self.batch, subject_list=listing)
        self.assertEqual([s.subject_root for s in subjects], [self.root])




class BatchScanGuardFixRowsTest(unittest.TestCase):
    """Backlog rows "27 November 2026: batch scan guard fixes": Opus (scan fix) P2-1..P2-3, Codex (scan fix) 1..3."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.batch = self.root / "batch"
        self.batch.mkdir()

    def reasons_by_path(self, skipped):
        return {item["path"]: item["reason"] for item in skipped}

    def same_input(self, subject_id, subject_root):
        return f"same input as subject {subject_id}: {subject_root}"

    def test_opus_scan_fix_p2_1_recursive_glob_gives_one_subject_for_a_nested_dicom_layout(self):
        write_dicom(self.batch / "sub-1" / "dicom")  # batch/sub-1/dicom/series1/IM0001
        subjects, skipped = batch_cli.discover_subjects(self.batch, subject_glob="**")
        sub1 = self.batch / "sub-1"
        self.assertEqual([(s.subject_id, s.subject_root, s.mode) for s in subjects], [("sub-1", sub1, "dicom")])
        reasons = self.reasons_by_path(skipped)
        self.assertEqual(reasons[(sub1 / "dicom").as_posix()], self.same_input("sub-1", sub1))
        self.assertEqual(reasons[(sub1 / "dicom" / "series1").as_posix()], self.same_input("sub-1", sub1))

    def test_opus_scan_fix_p2_1_post_pass_outermost_root_wins_when_the_inner_root_comes_first(self):
        """Sorted glob order does not sort resolved roots: ``dicom`` is accepted before ``t/sub-1``, whose root
        (through the ``<Batch Folder>/<subject ID>`` fallback) is ``batch/sub-1``, a folder above ``dicom``."""
        write_dicom(self.batch / "sub-1" / "dicom")
        (self.batch / "t" / "sub-1").mkdir(parents=True)
        subjects, skipped = batch_cli.discover_subjects(self.batch, subject_glob="*/*")
        sub1 = self.batch / "sub-1"
        self.assertEqual([(s.subject_id, s.subject_root) for s in subjects], [("sub-1", sub1)])
        self.assertEqual(self.reasons_by_path(skipped), {(sub1 / "dicom").as_posix(): self.same_input("sub-1", sub1)})
        subjects, _ = batch_cli.discover_subjects(self.batch, subject_glob="*/*", include_missing=True)
        self.assertEqual([(s.subject_id, s.mode) for s in subjects], [("dicom", "missing"), ("sub-1", "dicom")])

    def test_opus_scan_fix_p2_2_recursive_glob_with_arrange_copy_writes_nothing_for_a_nested_input_folder(self):
        write_raw_files_directly(self.batch / "sub-1" / "input")
        subjects, skipped = batch_cli.discover_subjects(self.batch, subject_glob="**", arrange_inputs="copy")
        sub1 = self.batch / "sub-1"
        self.assertFalse((sub1 / "input" / "input").exists())
        self.assertEqual([(s.subject_id, s.subject_root) for s in subjects], [("sub-1", sub1)])
        self.assertEqual(self.reasons_by_path(skipped)[(sub1 / "input").as_posix()], self.same_input("sub-1", sub1))

    def test_opus_scan_fix_p2_2_recursive_glob_with_arrange_copy_and_output_root_writes_nothing(self):
        write_raw_files_directly(self.batch / "sub-1" / "input")
        out = self.root / "out"
        subjects, _ = batch_cli.discover_subjects(self.batch, subject_glob="**", arrange_inputs="copy", output_root=out)
        self.assertFalse((out / "input").exists())
        self.assertFalse((self.batch / "sub-1" / "input" / "input").exists())
        self.assertEqual([(s.subject_id, s.subject_root) for s in subjects], [("sub-1", self.batch / "sub-1")])

    def test_opus_scan_fix_p2_2_guard_direct_raw_files_are_still_arranged(self):
        write_raw_files_directly(self.batch / "sub-1")
        subjects, _ = batch_cli.discover_subjects(self.batch, subject_glob="**", arrange_inputs="copy")
        self.assertEqual([(s.subject_id, s.mode) for s in subjects], [("sub-1", "subject_root+arranged")])
        self.assertTrue((self.batch / "sub-1" / "input" / "data.nii.gz").is_file())

    def test_opus_scan_fix_p2_3_recursive_glob_gives_the_batch_folder_a_neutral_reason(self):
        write_canonical(self.batch / "sub-1")
        _, skipped = batch_cli.discover_subjects(self.batch, subject_glob="**")
        self.assertEqual(self.reasons_by_path(skipped)[self.batch.as_posix()], SELF_REASON)

    def test_codex_scan_fix_1_text_file_in_the_canonical_input_of_the_batch_folder_is_rejected(self):
        write_canonical(self.batch)
        (self.batch / "input" / "notes.txt").write_text("not an image\n", encoding="utf-8")
        listing = self.root / "list.txt"
        listing.write_text("input/notes.txt\n", encoding="utf-8")
        subjects, skipped = batch_cli.discover_subjects(self.batch, subject_list=listing)
        self.assertEqual(subjects, [])
        self.assertEqual(self.reasons_by_path(skipped), {(self.batch / "input" / "notes.txt").as_posix(): NEW_REASON})

    def test_codex_scan_fix_1_decision_canonical_input_file_entry_is_not_input_itself(self):
        """Executor decision: only ``find_convertible_input`` and the DICOM check make a file input itself (no name
        list). A canonical ``input/data.nii.gz`` entry is read through its folder, as the entry ``input`` is."""
        write_canonical(self.batch)
        listing = self.root / "list.txt"
        listing.write_text("input/data.nii.gz\n", encoding="utf-8")
        subjects, skipped = batch_cli.discover_subjects(self.batch, subject_list=listing)
        self.assertEqual(subjects, [])
        self.assertEqual([item["reason"] for item in skipped], [NEW_REASON])

    def test_codex_scan_fix_1_guard_convertible_file_entries_in_the_batch_folder_stay_subjects(self):
        for stem in ("a", "b"):
            for suffix in (".nii.gz", ".bval", ".bvec"):
                (self.batch / f"{stem}{suffix}").write_bytes(b"x")
        listing = self.root / "list.txt"
        listing.write_text("a.nii.gz\n", encoding="utf-8")
        subjects, skipped = batch_cli.discover_subjects(self.batch, subject_list=listing)
        self.assertEqual([(s.subject_id, s.subject_root, s.mode) for s in subjects], [("a.nii.gz", self.batch, "nifti")])
        self.assertEqual(skipped, [])
        listing.write_text("a.nii.gz\nb.nii.gz\n", encoding="utf-8")
        subjects, skipped = batch_cli.discover_subjects(self.batch, subject_list=listing)
        self.assertEqual([(s.subject_id, s.subject_root) for s in subjects],
                         [("a.nii.gz", self.batch), ("b.nii.gz", self.batch)])
        self.assertEqual(skipped, [])

    def test_codex_scan_fix_2_file_entry_read_through_the_output_folder_uses_the_root_key(self):
        (self.batch / "notes.txt").write_text("not an image\n", encoding="utf-8")
        out = self.root / "out"
        write_canonical(out / "notes.txt")
        listing = self.root / "list.txt"
        listing.write_text(f"notes.txt\n{out / 'notes.txt' / 'input'}\n", encoding="utf-8")
        subjects, skipped = batch_cli.discover_subjects(self.batch, subject_list=listing, output_root=out)
        self.assertEqual([(s.subject_id, s.subject_root) for s in subjects], [("notes.txt", out / "notes.txt")])
        self.assertEqual([(item["subject_id"], item["reason"]) for item in skipped],
                         [("input", self.same_input("notes.txt", out / "notes.txt"))])

    @skip_unless_symlinks
    def test_codex_scan_fix_3_legacy_files_above_the_batch_folder_are_rejected_in_scan_mode(self):
        cohort = self.root  # legacy sub-a-* files in /cohort; the Batch Folder is /cohort/batch
        write_registered(cohort, "sub-a")
        (self.batch / "sub-a").mkdir()
        (self.batch / "sub-a" / "sub-a").symlink_to(cohort, target_is_directory=True)
        subjects, skipped = batch_cli.discover_subjects(self.batch)
        self.assertNotIn(cohort, [s.subject_root for s in subjects])
        self.assertEqual(subjects, [])
        self.assertEqual(self.reasons_by_path(skipped),
                         {(self.batch / "sub-a").as_posix(): batch_cli.BATCH_FOLDER_ANCESTOR_REASON})

    def test_codex_scan_fix_3_guard_legacy_files_in_the_batch_folder_itself_stay_accepted(self):
        write_registered(self.batch, "sub-l")
        (self.batch / "sub-l").mkdir()
        subjects, skipped = batch_cli.discover_subjects(self.batch)
        self.assertEqual([(s.subject_id, s.subject_root, s.mode) for s in subjects],
                         [("sub-l", self.batch, "legacy_registered")])
        self.assertEqual(skipped, [])

if __name__ == "__main__":
    unittest.main()
