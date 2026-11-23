#!/usr/bin/env python3
"""Package D3: the subject folder follows the input identity (D3.1 DICOM signature, D3.2 subject choice).

The new names are imported inside each test so that every test fails on its own before the change.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from win_test_helpers import skip_unless_symlinks
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

from shared.input_import import InputImportConfig, prepare_inputs


def write_nifti(folder: Path, name: str = "dwi", fill: float = 1.0, bvals: str = "0 1000") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / f"{name}.nii.gz"
    nib.save(nib.Nifti1Image(np.full((2, 2, 2, 2), fill, dtype=np.float32), np.eye(4)), str(source))
    (folder / f"{name}.bval").write_text(bvals + "\n", encoding="utf-8")
    (folder / f"{name}.bvec").write_text("0 1\n0 0\n0 0\n", encoding="utf-8")
    return source


def snapshot(root: Path) -> dict:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def mtimes(root: Path) -> dict:
    return {path.relative_to(root).as_posix(): path.stat().st_mtime_ns
            for path in sorted(root.rglob("*")) if path.is_file()}


def dicom_folder(base: Path, count: int = 3) -> Path:
    root = base / "dicom"
    root.mkdir(parents=True)
    for index in range(count):
        (root / f"im{index}.dcm").write_bytes(b"\0" * 128 + b"DICM" + bytes([index + 1]) * 16)
    return root


def fake_dcm2niix(command, **_kwargs):
    folder = Path(command[command.index("-o") + 1])
    nib.save(nib.Nifti1Image(np.ones((2, 3, 4, 3), dtype=np.float32), np.eye(4)), str(folder / "data.nii.gz"))
    (folder / "data.bval").write_text("0 1000 1000\n")
    (folder / "data.bvec").write_text("0 1 0\n0 0 1\n0 0 0\n")
    return {"returncode": 0, "command": command}


def fake_dwiconvert(command, **_kwargs):
    def target(flag):
        return Path(command[command.index(flag) + 1])

    nib.save(nib.Nifti1Image(np.ones((2, 2, 2, 2), dtype=np.float32), np.eye(4)), str(target("--outputNiftiFile")))
    target("--outputBValues").write_text("0 1000\n")
    target("--outputBVectors").write_text("0 1\n0 0\n0 0\n")
    return {"returncode": 0, "command": command}


def write_detached_nrrd(folder: Path, payload: bytes = b"\x01\x00" * 16) -> tuple[Path, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    raw = folder / "dwi.raw"
    raw.write_bytes(payload)
    header = folder / "dwi.nhdr"
    header.write_text("NRRD0004\ntype: short\ndimension: 4\nsizes: 2 2 2 2\nencoding: raw\ndata file: dwi.raw\n",
                      encoding="ascii")
    return header, raw


class SubjectChoiceCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name).resolve()
        self.out = self.tmp / "out"
        self.out.mkdir()

    def root_for_id(self, subject_id: str) -> Path:
        return self.out / subject_id

    def config(self, source: Path, **extra) -> InputImportConfig:
        return InputImportConfig(source=source, subject_root=self.out / "3167", subject_id="3167",
                                 source_kind=extra.pop("source_kind", "nifti"), force=True, **extra)

    def choose(self, cfg: InputImportConfig, base: str = "3167"):
        from shared.input_import import choose_subject_for_source, import_source_signatures

        return choose_subject_for_source(base, self.root_for_id, import_source_signatures(cfg),
                                         source=cfg.source, source_kind=cfg.source_kind)

    def import_into(self, cfg: InputImportConfig, base: str = "3167"):
        choice = self.choose(cfg, base)
        manifest = prepare_inputs(replace(cfg, subject_root=choice.subject_root, subject_id=choice.subject_id))
        return choice, manifest


class ChooseSubjectTest(SubjectChoiceCase):
    def test_d3_2_empty_output_folder_gives_the_base_id_new(self) -> None:
        choice = self.choose(self.config(write_nifti(self.tmp / "a")))
        self.assertEqual((choice.subject_id, choice.subject_root, choice.decision), ("3167", self.out / "3167", "new"))
        self.assertEqual(choice.base_id, "3167")
        self.assertFalse((self.out / "3167").exists())  # choosing creates nothing

    def test_d3_2_same_nifti_again_is_same_input_and_prepare_inputs_reuses_untouched_files(self) -> None:
        cfg = self.config(write_nifti(self.tmp / "a"))
        self.import_into(cfg)
        before = mtimes(self.out / "3167")
        choice, manifest = self.import_into(cfg)
        self.assertEqual((choice.subject_id, choice.decision), ("3167", "same_input"))
        self.assertTrue(manifest.get("reused"))
        self.assertEqual(mtimes(self.out / "3167"), before)

    def test_d3_2_another_input_goes_to_base_2_and_leaves_the_first_folder_byte_identical(self) -> None:
        first = self.config(write_nifti(self.tmp / "a"))
        self.import_into(first)
        kept = snapshot(self.out / "3167")
        other_file = self.config(write_nifti(self.tmp / "b", fill=2.0))
        choice, _ = self.import_into(other_file)
        self.assertEqual((choice.subject_id, choice.subject_root, choice.decision),
                         ("3167_2", self.out / "3167_2", "new"))
        self.assertEqual(snapshot(self.out / "3167"), kept)

    def test_d3_2_same_path_with_new_size_or_mtime_or_other_bvals_is_another_input(self) -> None:
        source = write_nifti(self.tmp / "a")
        self.import_into(self.config(source))
        kept = snapshot(self.out / "3167")
        stamp = source.stat().st_mtime_ns + 5_000_000_000
        os.utime(source, ns=(stamp, stamp))  # same bytes, another mtime
        self.assertEqual(self.choose(self.config(source)).subject_id, "3167_2")
        write_nifti(self.tmp / "a", fill=3.0)  # another size or content
        self.assertEqual(self.choose(self.config(source)).subject_id, "3167_2")
        other_bvals = self.tmp / "other.bval"
        other_bvals.write_text("0 2000\n", encoding="utf-8")
        self.assertEqual(self.choose(self.config(source, bval_path=other_bvals)).subject_id, "3167_2")
        self.assertEqual(snapshot(self.out / "3167"), kept)

    def test_d3_2_the_mapping_is_stable_back_and_forth_and_a_third_input_takes_base_3(self) -> None:
        a = self.config(write_nifti(self.tmp / "a"))
        b = self.config(write_nifti(self.tmp / "b", fill=2.0))
        c = self.config(write_nifti(self.tmp / "c", fill=3.0))
        self.assertEqual(self.import_into(a)[0].subject_id, "3167")
        self.assertEqual(self.import_into(b)[0].subject_id, "3167_2")
        back = self.import_into(a)[0]
        self.assertEqual((back.subject_id, back.decision), ("3167", "same_input"))
        again = self.import_into(b)[0]
        self.assertEqual((again.subject_id, again.decision), ("3167_2", "same_input"))
        self.assertEqual(self.import_into(c)[0].subject_id, "3167_3")

    def test_d3_2_only_the_mask_differs_stays_in_the_same_folder(self) -> None:
        source = write_nifti(self.tmp / "a")
        mask = self.tmp / "mask.nii.gz"
        nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4)), str(mask))
        self.import_into(self.config(source))
        choice = self.choose(self.config(source, mask_path=mask))
        self.assertEqual((choice.subject_id, choice.decision), ("3167", "same_input"))

    def test_d3_2_hand_placed_input_data_without_a_manifest_is_skipped_and_not_overwritten(self) -> None:
        hand = self.out / "3167" / "input"
        hand.mkdir(parents=True)
        (hand / "data.nii.gz").write_bytes(b"hand placed")
        (hand / "bvals").write_text("0\n")
        choice, _ = self.import_into(self.config(write_nifti(self.tmp / "a")))
        self.assertEqual(choice.subject_id, "3167_2")
        self.assertEqual((hand / "data.nii.gz").read_bytes(), b"hand placed")

    def test_d3_2_legacy_manifest_without_signatures_and_the_same_source_path_is_used(self) -> None:
        source = write_nifti(self.tmp / "a")
        self.import_into(self.config(source))
        manifest_path = self.out / "3167" / "input" / "input_import_manifest.json"
        legacy = json.loads(manifest_path.read_text())
        legacy.pop("source_signatures")
        manifest_path.write_text(json.dumps(legacy))
        (self.out / "3167" / "input" / "input_import_intent.json").unlink(missing_ok=True)
        choice = self.choose(self.config(source))
        self.assertEqual((choice.subject_id, choice.decision), ("3167", "legacy_same_source"))
        other = self.choose(self.config(write_nifti(self.tmp / "b", fill=2.0)))
        self.assertEqual((other.subject_id, other.decision), ("3167_2", "new"))
        # The forced import then runs as today and writes the signature.
        prepare_inputs(replace(self.config(source), subject_root=choice.subject_root))
        self.assertIn("source_signatures", json.loads(manifest_path.read_text()))

    def test_d3_2_base_2_used_by_other_data_gives_base_3_and_99_used_is_a_clear_error(self) -> None:
        a = self.config(write_nifti(self.tmp / "a"))
        self.import_into(a)
        (self.out / "3167_2" / "outputs").mkdir(parents=True)
        (self.out / "3167_2" / "outputs" / "result.txt").write_text("x")
        b = self.config(write_nifti(self.tmp / "b", fill=2.0))
        self.assertEqual(self.choose(b).subject_id, "3167_3")
        for index in range(3, 100):
            (self.out / f"3167_{index}" / "outputs").mkdir(parents=True)
            (self.out / f"3167_{index}" / "outputs" / "result.txt").write_text("x")
        with self.assertRaises(ValueError) as caught:
            self.choose(b)
        self.assertIn("3167_99", str(caught.exception))

    def test_d3_2_results_without_a_staged_input_are_not_unused(self) -> None:
        cfg = self.config(write_nifti(self.tmp / "a"))
        for folder in ("outputs", "preprocessed", "session_2026"):
            with self.subTest(folder=folder):
                root = self.out / "3167"
                (root / folder).mkdir(parents=True)
                (root / folder / "x.nii.gz").write_bytes(b"x")
                self.assertEqual(self.choose(cfg).subject_id, "3167_2")
                (root / folder / "x.nii.gz").unlink()
                (root / folder).rmdir()
        (self.out / "3167" / "notes.txt").write_text("a file at the top")
        self.assertEqual(self.choose(cfg).subject_id, "3167_2")

    def test_d3_2_empty_input_hidden_entries_and_empty_scaffolding_count_as_unused(self) -> None:
        root = self.out / "3167"
        (root / "input").mkdir(parents=True)
        (root / ".DS_Store").write_bytes(b"x")
        (root / "input" / ".signed-nrrd-tmp").mkdir()
        (root / "cache" / "uncompressed_nifti").mkdir(parents=True)
        (root / "cache" / "uncompressed_nifti" / ".staging-1.nii").write_bytes(b"x")
        choice = self.choose(self.config(write_nifti(self.tmp / "a")))
        self.assertEqual((choice.subject_id, choice.decision), ("3167", "new"))

    @skip_unless_symlinks
    def test_d3_2_fe1_only_the_environment_folder_is_ignored_and_a_symlink_there_still_counts(self) -> None:
        cfg = self.config(write_nifti(self.tmp / "a"))
        root = self.out / "3167"
        environment = root / "outputs" / "00_run_metadata" / "environment"
        (environment / "deeper").mkdir(parents=True)
        (environment / "environment_doctor.json").write_text("{}")
        (environment / "deeper" / "setup-1.log").write_text("log")
        first = self.choose(cfg)
        self.assertEqual((first.subject_id, first.decision), ("3167", "new"))
        others =(root / "outputs" / "00_run_metadata" / "Test1_orchestration_manifest.json",
                  root / "outputs" / "01_preprocessing" / "x.txt",
                  root / "outputs" / "x.txt",
                  root / "input" / "outputs" / "00_run_metadata" / "environment" / "x.json")
        for other in others:
            with self.subTest(other=other.relative_to(root).as_posix()):
                other.parent.mkdir(parents=True, exist_ok=True)
                other.write_text("x")
                self.assertEqual(self.choose(cfg).subject_id, "3167_2")
                other.unlink()
        (environment / "link.json").symlink_to(self.tmp / "a" / "dwi.bval")
        self.assertEqual(self.choose(cfg).subject_id, "3167_2")

    def test_d3_2_a_root_with_only_another_inputs_intent_marker_is_unused(self) -> None:
        root = self.out / "3167" / "input"
        root.mkdir(parents=True)
        (root / "input_import_intent.json").write_text(json.dumps({"identity": {"dwi": None}}))
        choice = self.choose(self.config(write_nifti(self.tmp / "a")))
        self.assertEqual((choice.subject_id, choice.decision), ("3167", "new"))

    def test_d3_2_a_failed_import_is_retried_in_the_same_folder_and_another_input_moves_on(self) -> None:
        x = self.config(write_nifti(self.tmp / "x"))
        y = self.config(write_nifti(self.tmp / "y", fill=2.0))
        with patch("shared.input_import._copy_file", side_effect=RuntimeError("disk full")):
            with self.assertRaises(RuntimeError):
                self.import_into(x)
        staged = self.out / "3167" / "input"
        self.assertTrue((staged / "data.nii.gz").is_file())  # staged before the failure
        self.assertFalse((staged / "input_import_manifest.json").exists())
        self.assertTrue((staged / "input_import_intent.json").is_file())
        retry = self.choose(x)
        self.assertEqual((retry.subject_id, retry.decision), ("3167", "retry_same_input"))
        other = self.choose(y)
        self.assertEqual((other.subject_id, other.decision), ("3167_2", "new"))
        done = self.import_into(x)[1]
        self.assertEqual(done["status"], "completed")

    def test_d3_2_a_missing_required_file_raises_before_any_folder_is_chosen(self) -> None:
        from shared.input_import import import_source_signatures

        source = write_nifti(self.tmp / "a")
        (self.tmp / "a" / "dwi.bvec").unlink()
        with self.assertRaises(FileNotFoundError) as caught:
            import_source_signatures(self.config(source))
        self.assertIn("dwi.bvec", str(caught.exception))
        with self.assertRaises(FileNotFoundError):
            import_source_signatures(self.config(self.tmp / "missing.nii.gz"))
        self.assertEqual(list(self.out.iterdir()), [])

    def test_d3_2_an_automatically_imported_dicom_root_is_skipped(self) -> None:
        folder = dicom_folder(self.tmp)
        root = self.out / "3167" / "input"
        root.mkdir(parents=True)
        (root / "data.nii.gz").write_bytes(b"auto")
        (root / "auto_dicom_import.json").write_text(json.dumps({"kind": "slicerdsurfer_auto_dicom_input"}))
        (root / "input_import_manifest.json").write_text(json.dumps(
            {"status": "completed", "plan": {"source": str(folder), "source_kind": "dicom"}}))
        from shared.input_import import choose_subject_for_source, import_source_signatures

        cfg = self.config(folder, source_kind="dicom")
        choice = choose_subject_for_source("3167", self.root_for_id, import_source_signatures(cfg),
                                           source=folder, source_kind="dicom")
        self.assertEqual(choice.subject_id, "3167_2")

    def test_d3_2_the_root_rule_is_the_gui_rule_including_the_nested_case(self) -> None:
        from shared.input_import import subject_root_rule

        rule = subject_root_rule(str(self.out), str(self.tmp / "cfg" / "subjects"))
        self.assertEqual(rule("3167_2"), self.out / "3167_2")
        fallback = subject_root_rule("", str(self.tmp / "cfg" / "subjects"))
        self.assertEqual(fallback("3167_2"), self.tmp / "cfg" / "subjects" / "3167_2")
        # Pinned, not designed: an Output Folder named after the base ID puts the suffixed root inside it.
        nested = subject_root_rule(str(self.out / "3167"), "")
        self.assertEqual(nested("3167"), self.out / "3167")
        self.assertEqual(nested("3167_2"), self.out / "3167" / "3167_2")


class IntentMarkerTest(SubjectChoiceCase):
    def test_d3_2_prepare_inputs_writes_the_intent_marker_without_the_mask(self) -> None:
        from shared.input_import import import_source_signatures, input_identity

        cfg = self.config(write_nifti(self.tmp / "a"))
        prepare_inputs(cfg)
        marker = json.loads((self.out / "3167" / "input" / "input_import_intent.json").read_text())
        self.assertEqual(marker["identity"], json.loads(json.dumps(input_identity(import_source_signatures(cfg)))))
        self.assertNotIn("mask", marker["identity"])
        self.assertTrue(marker.get("created_at"))

    def test_d3_2_a_dry_run_writes_no_marker(self) -> None:
        prepare_inputs(replace(self.config(write_nifti(self.tmp / "a")), dry_run=True))
        self.assertFalse((self.out / "3167").exists())


class DicomSignatureTest(SubjectChoiceCase):
    def test_d3_1_an_unchanged_folder_gives_the_same_signature_and_a_change_another(self) -> None:
        from shared.input_import import dicom_source_signature

        folder = dicom_folder(self.tmp)
        first = dicom_source_signature(folder)
        self.assertEqual(first, dicom_source_signature(folder))
        self.assertEqual((first["folder"], first["count"], first["series_uid"]), (folder.as_posix(), 3, None))
        (folder / "im9.dcm").write_bytes(b"\0" * 128 + b"DICM" + b"added")
        added = dicom_source_signature(folder)
        self.assertNotEqual(added, first)
        (folder / "im9.dcm").unlink()
        self.assertEqual(dicom_source_signature(folder), first)
        (folder / "im0.dcm").unlink()
        self.assertNotEqual(dicom_source_signature(folder), first)
        stamp = (folder / "im1.dcm").stat().st_mtime_ns + 3_000_000_000
        before_touch = dicom_source_signature(folder)
        os.utime(folder / "im1.dcm", ns=(stamp, stamp))
        self.assertNotEqual(dicom_source_signature(folder), before_touch)

    def test_d3_1_import_source_signatures_returns_a_dicom_signature_with_the_mask(self) -> None:
        from shared.input_import import _import_source_signatures

        folder = dicom_folder(self.tmp)
        cfg = self.config(folder, source_kind="dicom")
        signatures = _import_source_signatures(cfg, folder, "dicom")
        self.assertEqual(set(signatures), {"dicom", "mask"})
        self.assertEqual(signatures["dicom"]["count"], 3)

    def dicom_config(self, folder: Path) -> InputImportConfig:
        return self.config(folder, source_kind="dicom", backend="dcm2niix", dcm2niix_executable=Path("/converter"))

    def test_d3_1_an_unchanged_dicom_source_is_reused_and_a_touched_file_converts_again(self) -> None:
        folder = dicom_folder(self.tmp)
        cfg = self.dicom_config(folder)
        with patch("shared.input_import._run", side_effect=fake_dcm2niix) as run:
            first = prepare_inputs(cfg)
            self.assertIn("dicom", first["source_signatures"])
            second = prepare_inputs(cfg)
            self.assertTrue(second.get("reused"))
            self.assertEqual(run.call_count, 1)
            stamp = (folder / "im1.dcm").stat().st_mtime_ns + 3_000_000_000
            os.utime(folder / "im1.dcm", ns=(stamp, stamp))
            third = prepare_inputs(cfg)
            self.assertFalse(third.get("reused", False))
            self.assertEqual(run.call_count, 2)

    def test_d3_1_an_old_dicom_manifest_without_signatures_converts_again_and_gains_one(self) -> None:
        folder = dicom_folder(self.tmp)
        cfg = self.dicom_config(folder)
        with patch("shared.input_import._run", side_effect=fake_dcm2niix) as run:
            prepare_inputs(cfg)
            manifest_path = self.out / "3167" / "input" / "input_import_manifest.json"
            old = json.loads(manifest_path.read_text())
            old.pop("source_signatures")
            manifest_path.write_text(json.dumps(old))
            again = prepare_inputs(cfg)
            self.assertFalse(again.get("reused", False))
            self.assertEqual(run.call_count, 2)
        self.assertIn("source_signatures", json.loads(manifest_path.read_text()))


class JobChoiceTest(SubjectChoiceCase):
    """The import job (thread or child process) chooses the target; the wrappers keep a stable identity."""

    def request(self, mode: str, source: Path, kind: str, base: str = "3167", **extra) -> dict:
        return {
            "mode": mode,
            "config": {"source": str(source), "subject_root": str(self.out / base), "subject_id": base,
                       "source_kind": kind, "force": True, **extra.pop("config", {})},
            "subject_choice": {"base_id": base, "output_dir": str(self.out),
                               "fallback_parent": str(self.tmp / "cfg" / "subjects")},
            **extra,
        }

    def test_d3_3_execute_request_reports_the_chosen_target(self) -> None:
        from shared.prepare_inputs_cli import execute_request

        a = write_nifti(self.tmp / "a")
        b = write_nifti(self.tmp / "b", fill=2.0)
        first = execute_request(self.request("prepare", a, "nifti",
                                             config={"bval_path": str(a.with_name("dwi.bval"))}))
        self.assertEqual(first["subject_choice"], {"base_id": "3167", "subject_id": "3167",
                                                   "subject_root": (self.out / "3167").as_posix(),
                                                   "decision": "new"})
        second = execute_request(self.request("prepare", b, "nifti"))
        self.assertEqual((second["subject_choice"]["subject_id"], second["subject_choice"]["subject_root"]),
                         ("3167_2", (self.out / "3167_2").as_posix()))
        self.assertTrue((self.out / "3167_2" / "input" / "data.nii.gz").is_file())
        saved = json.loads((self.out / "3167_2" / "input" / "input_import_manifest.json").read_text())
        self.assertNotIn("subject_choice", saved)  # the target is reported fresh each time, never stored
        again = execute_request(self.request("prepare", b, "nifti"))
        self.assertEqual((again["subject_choice"]["decision"], again.get("reused")), ("same_input", True))

    def test_d3_3_a_missing_file_fails_the_job_before_any_folder_is_made(self) -> None:
        from shared.prepare_inputs_cli import execute_request

        with self.assertRaises(FileNotFoundError):
            execute_request(self.request("prepare", self.tmp / "gone.nii.gz", "nifti"))
        self.assertEqual(list(self.out.iterdir()), [])

    def series(self, folder: Path) -> dict:
        return {"uid": "1.2.3", "description": "dwi", "number": "1", "acquisition_time": "", "frames": 3,
                "paths": [str(path) for path in sorted(folder.glob("*.dcm"))]}

    def test_d3_1_the_dicom_series_import_gets_a_stable_identity_and_is_reused(self) -> None:
        from shared.prepare_inputs_cli import execute_request

        folder = dicom_folder(self.tmp)
        with patch("shared.dicom_input.converter_path", return_value=("dcm2niix", Path("/converter"))), \
                patch("shared.input_import._run", side_effect=fake_dcm2niix) as run:
            first = execute_request(self.request("dicom", folder, "dicom", series_entry=self.series(folder)))
            receipt = self.out / "3167" / "input" / "input_import_manifest.json"
            saved = json.loads(receipt.read_text())
            self.assertEqual(saved["source_signatures"]["dicom"]["folder"], folder.as_posix())
            self.assertEqual(saved["source_signatures"]["dicom"]["series_uid"], "1.2.3")
            self.assertEqual(saved["plan"]["dicom_series"]["source"], str(folder))
            before = receipt.stat().st_mtime_ns
            second = execute_request(self.request("dicom", folder, "dicom", series_entry=self.series(folder)))
            self.assertEqual(run.call_count, 1)
            self.assertEqual((second["subject_choice"]["subject_id"], second["subject_choice"]["decision"]),
                             ("3167", "same_input"))
            self.assertTrue(second.get("reused"))
            self.assertEqual(receipt.stat().st_mtime_ns, before)  # a reuse rewrites nothing
            self.assertEqual(first["subject_choice"]["subject_id"], "3167")
            stamp = (folder / "im2.dcm").stat().st_mtime_ns + 3_000_000_000
            os.utime(folder / "im2.dcm", ns=(stamp, stamp))
            third = execute_request(self.request("dicom", folder, "dicom", series_entry=self.series(folder)))
            self.assertEqual(third["subject_choice"]["subject_id"], "3167_2")

    def test_d3_2_a_legacy_dicom_series_manifest_matches_on_the_recorded_series_folder(self) -> None:
        from shared.prepare_inputs_cli import execute_request

        folder = dicom_folder(self.tmp)
        input_dir = self.out / "3167" / "input"
        input_dir.mkdir(parents=True)
        (input_dir / "data.nii.gz").write_bytes(b"old")
        (input_dir / "input_import_manifest.json").write_text(json.dumps({
            "status": "completed",
            "plan": {"source": "/tmp/slicerdspace-series-xyz", "source_kind": "dicom",
                     "dicom_series": {"source": str(folder), "series_instance_uid": "1.2.3"}}}))
        with patch("shared.dicom_input.converter_path", return_value=("dcm2niix", Path("/converter"))), \
                patch("shared.input_import._run", side_effect=fake_dcm2niix):
            result = execute_request(self.request("dicom", folder, "dicom", series_entry=self.series(folder)))
        self.assertEqual((result["subject_choice"]["subject_id"], result["subject_choice"]["decision"]),
                         ("3167", "legacy_same_source"))

    def test_d3_2_a_legacy_dicom_manifest_of_another_series_in_the_same_folder_is_skipped(self) -> None:
        from shared.prepare_inputs_cli import execute_request

        folder = dicom_folder(self.tmp)
        input_dir = self.out / "3167" / "input"
        input_dir.mkdir(parents=True)
        (input_dir / "data.nii.gz").write_bytes(b"old")
        (input_dir / "input_import_manifest.json").write_text(json.dumps({
            "status": "completed",
            "plan": {"source": "/tmp/slicerdspace-series-xyz", "source_kind": "dicom",
                     "dicom_series": {"source": str(folder), "series_instance_uid": "9.9.9"}}}))
        with patch("shared.dicom_input.converter_path", return_value=("dcm2niix", Path("/converter"))), \
                patch("shared.input_import._run", side_effect=fake_dcm2niix):
            result = execute_request(self.request("dicom", folder, "dicom", series_entry=self.series(folder)))
        self.assertEqual((result["subject_choice"]["subject_id"], result["subject_choice"]["decision"]),
                         ("3167_2", "new"))
        self.assertEqual((input_dir / "data.nii.gz").read_bytes(), b"old")

    def test_d3_2_fe2_the_detached_nrrd_import_is_stable_and_a_changed_data_file_converts_again_in_place(
            self) -> None:
        # FE2 changed the last step: the data file is not part of the input identity (the header is), so a
        # changed data file stays in the same subject folder; the importer's reuse test still converts again.
        from shared.prepare_inputs_cli import execute_request

        header, raw = write_detached_nrrd(self.tmp / "n")
        chosen_raw = self.tmp / "n" / "chosen.raw"
        chosen_raw.write_bytes(raw.read_bytes())
        with patch("shared.input_import._run", side_effect=fake_dwiconvert) as run:
            first = execute_request(self.request("detached_nrrd", header, "nrrd", raw_path=str(chosen_raw)))
            self.assertEqual(first["subject_choice"]["subject_id"], "3167")
            second = execute_request(self.request("detached_nrrd", header, "nrrd", raw_path=str(chosen_raw)))
            self.assertEqual((second["subject_choice"]["subject_id"], second["subject_choice"]["decision"]),
                             ("3167", "same_input"))
            self.assertTrue(second.get("reused"))
            self.assertEqual(run.call_count, 1)
            chosen_raw.write_bytes(b"\x02\x00" * 16)
            stamp = chosen_raw.stat().st_mtime_ns + 3_000_000_000
            os.utime(chosen_raw, ns=(stamp, stamp))
            third = execute_request(self.request("detached_nrrd", header, "nrrd", raw_path=str(chosen_raw)))
            self.assertEqual((third["subject_choice"]["subject_id"], third["subject_choice"]["decision"]),
                             ("3167", "same_input"))
            self.assertFalse(third.get("reused", False))
            self.assertEqual(run.call_count, 2)

    def test_d3_2_fe2_a_plain_nhdr_import_then_the_same_data_file_chosen_explicitly_keeps_the_id(self) -> None:
        from shared.prepare_inputs_cli import execute_request

        header, raw = write_detached_nrrd(self.tmp / "n")
        with patch("shared.input_import._run", side_effect=fake_dwiconvert):
            plain = execute_request(self.request("prepare", header, "nrrd"))
            self.assertEqual(plain["subject_choice"]["subject_id"], "3167")
            explicit = execute_request(self.request("detached_nrrd", header, "nrrd", raw_path=str(raw)))
        self.assertEqual((explicit["subject_choice"]["subject_id"], explicit["subject_choice"]["decision"]),
                         ("3167", "same_input"))
        self.assertFalse((self.out / "3167_2").exists())

    def test_d3_2_fe2_another_data_file_under_the_same_header_keeps_the_id_and_is_not_reused(self) -> None:
        from shared.prepare_inputs_cli import execute_request

        header, _raw = write_detached_nrrd(self.tmp / "n")
        data_a = self.tmp / "n" / "a.raw"
        data_a.write_bytes(b"\x01\x00" * 16)
        data_b = self.tmp / "n" / "b.raw"
        data_b.write_bytes(b"\x05\x00" * 16)
        with patch("shared.input_import._run", side_effect=fake_dwiconvert) as run:
            first = execute_request(self.request("detached_nrrd", header, "nrrd", raw_path=str(data_a)))
            self.assertEqual(first["subject_choice"]["subject_id"], "3167")
            second = execute_request(self.request("detached_nrrd", header, "nrrd", raw_path=str(data_b)))
            self.assertEqual((second["subject_choice"]["subject_id"], second["subject_choice"]["decision"]),
                             ("3167", "same_input"))
            self.assertFalse(second.get("reused", False))  # the staged data is converted again from B
            self.assertEqual(run.call_count, 2)
        receipt = json.loads((self.out / "3167" / "input" / "input_import_manifest.json").read_text())
        self.assertEqual(receipt["plan"]["detached_nrrd"]["data"], str(data_b.resolve()))
        self.assertEqual(receipt["source_signatures"]["raw"]["path"], data_b.resolve().as_posix())
        self.assertFalse((self.out / "3167_2").exists())

    def test_d3_1_fe3_a_reused_series_import_is_validated_and_an_invalid_one_drops_the_receipt(self) -> None:
        from shared.prepare_inputs_cli import execute_request

        def fake_bad_dcm2niix(command, **_kwargs):
            folder = Path(command[command.index("-o") + 1])
            nib.save(nib.Nifti1Image(np.ones((2, 3, 4, 3), dtype=np.float32), np.eye(4)), str(folder / "data.nii.gz"))
            (folder / "data.bval").write_text("0 1000\n")  # two b-values for three volumes: invalid
            (folder / "data.bvec").write_text("0 1\n0 0\n0 0\n")
            return {"returncode": 0, "command": command}

        folder = dicom_folder(self.tmp)
        receipt = self.out / "3167" / "input" / "input_import_manifest.json"
        with patch("shared.dicom_input.converter_path", return_value=("dcm2niix", Path("/converter"))), \
                patch("shared.input_import._run", side_effect=fake_bad_dcm2niix) as run:
            # The job is killed after prepare_inputs wrote its completed manifest, before the validation ended.
            with patch("shared.dicom_input._validate_converted", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    execute_request(self.request("dicom", folder, "dicom", series_entry=self.series(folder)))
            self.assertEqual(json.loads(receipt.read_text())["status"], "completed")
            with self.assertRaises(ValueError):
                execute_request(self.request("dicom", folder, "dicom", series_entry=self.series(folder)))
            self.assertEqual(run.call_count, 1)  # the retry reached the reuse branch, not a new conversion
        self.assertFalse(receipt.exists())

    def test_d3_1_fe3_a_reused_valid_series_import_keeps_the_receipt_bytes(self) -> None:
        import shared.dicom_input as dicom_input
        from shared.prepare_inputs_cli import execute_request

        folder = dicom_folder(self.tmp)
        receipt = self.out / "3167" / "input" / "input_import_manifest.json"
        with patch("shared.dicom_input.converter_path", return_value=("dcm2niix", Path("/converter"))), \
                patch("shared.input_import._run", side_effect=fake_dcm2niix) as run:
            execute_request(self.request("dicom", folder, "dicom", series_entry=self.series(folder)))
            before = receipt.read_bytes()
            with patch.object(dicom_input, "_validate_converted", wraps=dicom_input._validate_converted) as validate:
                again = execute_request(self.request("dicom", folder, "dicom", series_entry=self.series(folder)))
            self.assertTrue(again.get("reused"))
            self.assertEqual(run.call_count, 1)
            self.assertEqual(validate.call_count, 1)  # the reused series is validated too
        self.assertEqual(receipt.read_bytes(), before)

    def test_d3_3_scan_dicom_and_a_request_without_a_choice_behave_as_before(self) -> None:
        from shared.prepare_inputs_cli import execute_request

        a = write_nifti(self.tmp / "a")
        request = self.request("prepare", a, "nifti")
        request.pop("subject_choice")
        result = execute_request(request)
        self.assertNotIn("subject_choice", result)
        self.assertTrue((self.out / "3167" / "input" / "data.nii.gz").is_file())


if __name__ == "__main__":
    unittest.main()
