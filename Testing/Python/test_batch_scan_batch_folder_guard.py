#!/usr/bin/env python3
"""Finding "batch scan: subject resolves to the Batch Folder".

``discover_subjects`` used to let a subfolder without input fall back to the Batch Folder itself, so an empty
subfolder became a subject whose input was the whole Batch Folder (results in ``<Batch Folder>/outputs``).
The rule now: a subject never resolves to the Batch Folder itself, except mode ``legacy_registered``, a file
entry of ``--subject-list`` and a subject-list entry that is the Batch Folder itself.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_batch_pipeline.py"
NEW_REASON = "no input in this subfolder; the input found is the Batch Folder itself"
OLD_REASON = "missing required input/data.nii.gz, bvals, or bvecs"

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


if __name__ == "__main__":
    unittest.main()
