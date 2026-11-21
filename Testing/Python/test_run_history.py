#!/usr/bin/env python3
"""Tests for run discovery and manifest parsing used by the Results run selector."""

from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from shared.run_history import (
    BATCH_MANIFEST_KIND,
    SUBJECT_MANIFEST_KIND,
    ManifestError,
    candidate_run_roots,
    discover_manifest_paths,
    discover_runs,
    dump_hidden_runs,
    hidden_stamp,
    manifest_path_for_plan,
    parse_hidden_runs,
    run_manifest_key,
    run_record_key,
    split_hidden,
    manifest_timestamp,
    parse_run_manifest,
    progress_from_manifest,
    progress_from_payloads,
    remember_directory,
    run_record_brief,
    run_record_details,
    run_folder_path,
    run_record_label,
    subject_folder_from_roots,
)


def _write(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) if not isinstance(payload, str) else payload, encoding="utf-8")
    return path


def _subject_manifest(**overrides):
    payload = {
        "kind": SUBJECT_MANIFEST_KIND,
        "version": 1,
        "status": "completed",
        "subject_id": "sub-01",
        "run_name": "GUI1",
        "targets": ["segmentation", "microstructure"],
        "tasks": [],
    }
    payload.update(overrides)
    return payload


class ManifestParsingTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_subject_manifest_record_fields(self) -> None:
        path = _write(
            self.root / "outputs/00_run_metadata/GUI1_orchestration_manifest.json",
            _subject_manifest(ended_at="2026-09-20T10:15:00+00:00", run_targets=["segmentation"]),
        )
        record = parse_run_manifest(path)
        self.assertEqual(record.kind, "subject")
        self.assertEqual(record.subject_id, "sub-01")
        self.assertEqual(record.run_name, "GUI1")
        self.assertEqual(record.targets, ("segmentation",))
        self.assertEqual(record.status, "completed")
        self.assertEqual(record.timestamp, datetime(2026, 9, 20, 10, 15, tzinfo=timezone.utc))
        self.assertFalse(record.is_batch)

    def test_batch_manifest_record_fields(self) -> None:
        path = _write(
            self.root / "batch_outputs/00_batch_metadata/batch_20260921_083000.json",
            {
                "kind": BATCH_MANIFEST_KIND,
                "version": 1,
                "run_name": "Test1",
                "execute": True,
                "forwarded_args": ["--targets", "segmentation,connectome", "--execute"],
                "subjects": [{"subject_id": "a"}, {"subject_id": "b"}],
                "results": [{"subject_id": "a", "returncode": 0}, {"subject_id": "b", "returncode": 2}],
            },
        )
        record = parse_run_manifest(path)
        self.assertTrue(record.is_batch)
        self.assertEqual(record.subject_count, 2)
        self.assertEqual(record.subject_id, "batch (2 subjects)")
        self.assertEqual(record.targets, ("segmentation", "connectome"))
        self.assertEqual(record.status, "failed")
        # No ISO time in a batch manifest: the file name carries it.
        self.assertEqual(record.timestamp.strftime("%Y%m%d_%H%M%S"), "20260921_083000")
        self.assertIn("pick one subject", run_record_details(record))

    def test_wrong_kind_is_rejected_with_actionable_message(self) -> None:
        path = _write(self.root / "x_orchestration_manifest.json", {"kind": "slicerddsurfer_run_plan", "version": 1})
        with self.assertRaises(ManifestError) as ctx:
            parse_run_manifest(path)
        self.assertIn("not a SlicerDSpace run manifest", str(ctx.exception))
        self.assertIn("00_run_metadata", str(ctx.exception))

    def test_unsupported_and_non_integer_versions_are_rejected(self) -> None:
        newer = _write(self.root / "a_orchestration_manifest.json", _subject_manifest(version=2))
        with self.assertRaises(ManifestError) as ctx:
            parse_run_manifest(newer)
        self.assertIn("version 2", str(ctx.exception))
        self.assertIn("Update the extension", str(ctx.exception))
        for bad in ("1", True, None, 1.0):
            path = _write(self.root / "b_orchestration_manifest.json", _subject_manifest(version=bad))
            with self.assertRaises(ManifestError, msg=repr(bad)):
                parse_run_manifest(path)

    def test_invalid_json_and_missing_file(self) -> None:
        broken = _write(self.root / "c_orchestration_manifest.json", "{not json")
        with self.assertRaisesRegex(ManifestError, "not valid JSON"):
            parse_run_manifest(broken)
        with self.assertRaisesRegex(ManifestError, "not found"):
            parse_run_manifest(self.root / "missing_orchestration_manifest.json")
        not_object = _write(self.root / "d_orchestration_manifest.json", "[1, 2]")
        with self.assertRaisesRegex(ManifestError, "JSON object"):
            parse_run_manifest(not_object)

    def test_timestamp_preference_order(self) -> None:
        path = self.root / "t.json"
        _write(path, {})
        self.assertEqual(
            manifest_timestamp({"ended_at": "2026-01-02T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z"}, path).day,
            2,
        )
        self.assertEqual(manifest_timestamp({"cancelled_at": "2026-01-03T00:00:00Z"}, path).day, 3)
        from_tasks = manifest_timestamp(
            {"tasks": [{"ended_at": "2026-01-04T00:00:00+00:00"}, {"ended_at": "2026-01-05T00:00:00+00:00"}]}, path
        )
        self.assertEqual(from_tasks.day, 5)

    def test_timestamp_falls_back_to_file_mtime(self) -> None:
        path = _write(self.root / "m_orchestration_manifest.json", _subject_manifest())
        stamp = time.mktime((2025, 6, 7, 8, 9, 10, 0, 0, -1))
        os.utime(path, (stamp, stamp))
        record = parse_run_manifest(path)
        self.assertAlmostEqual(record.timestamp.timestamp(), stamp, delta=1)

    def test_status_derived_from_tasks_when_missing(self) -> None:
        path = _write(
            self.root / "s_orchestration_manifest.json",
            _subject_manifest(status="", tasks=[{"name": "a", "status": "completed"}, {"name": "b", "status": "blocked"}]),
        )
        self.assertEqual(parse_run_manifest(path).status, "completed_with_blocks")

    def test_label_contains_time_subject_run_and_status(self) -> None:
        path = _write(
            self.root / "l_orchestration_manifest.json",
            _subject_manifest(status="completed_with_blocks", ended_at="2026-09-20T10:15:00+00:00"),
        )
        label = run_record_label(parse_run_manifest(path))
        self.assertIn("sub-01", label)
        self.assertIn("GUI1", label)
        self.assertIn("completed, some blocked", label)
        self.assertRegex(label, r"^2026-09-2\d \d\d:\d\d")


class RecordBriefTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _record(self, **overrides):
        path = _write(self.root / "b_orchestration_manifest.json", _subject_manifest(**overrides))
        return parse_run_manifest(path)

    def test_brief_uses_friendly_product_names_in_recorded_order(self) -> None:
        record = self._record(targets=[
            "microstructure", "microstructure_freewater", "microstructure_noddi", "microstructure_dki",
        ])
        self.assertEqual(run_record_brief(record), "Products: DTI, free water, NODDI, DKI")
        record = self._record(targets=["segmentation", "parcellation", "surface"])
        self.assertEqual(run_record_brief(record), "Products: DDSeg, DDParcel, DDSurfer surfaces")

    def test_brief_falls_back_to_raw_name_for_unknown_target(self) -> None:
        record = self._record(targets=["segmentation", "future_product"])
        self.assertEqual(run_record_brief(record), "Products: DDSeg, future_product")

    def test_brief_without_targets_says_not_recorded(self) -> None:
        self.assertEqual(run_record_brief(self._record(targets=[])), "Products: not recorded")

    def test_brief_of_batch_keeps_the_batch_sentence_and_details_stay_unchanged(self) -> None:
        batch = parse_run_manifest(_write(self.root / "batch_x.json", {
            "kind": BATCH_MANIFEST_KIND, "version": 1, "status": "completed", "subjects": [],
        }))
        self.assertTrue(batch.is_batch)
        self.assertEqual(
            run_record_brief(batch), "Batch run: pick one subject's run in the list to load its images."
        )
        record = self._record(targets=["segmentation"])
        self.assertIn("Targets: segmentation", run_record_details(record))
        self.assertIn("Manifest: ", run_record_details(record))

    def test_every_orchestrator_target_has_a_friendly_label(self) -> None:
        from shared.orchestrator import TARGET_ORDER
        from shared.run_history import TARGET_LABELS

        for target in TARGET_ORDER:
            with self.subTest(target=target):
                self.assertIn(target, TARGET_LABELS)
                self.assertTrue(TARGET_LABELS[target].strip())


class DiscoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_discovers_gui_cli_cancelled_and_batch_runs_but_not_history_or_tasks(self) -> None:
        meta = self.root / "sub-01/outputs/00_run_metadata"
        gui = _write(meta / "GUI1_orchestration_manifest.json", _subject_manifest(ended_at="2026-09-20T10:00:00Z"))
        cli = _write(
            meta / "cli_orchestration_manifest.json",
            _subject_manifest(run_name="cli", ended_at="2026-09-22T10:00:00Z"),
        )
        cancelled = _write(
            meta / "GUI2_cancelled_orchestration_manifest.json",
            _subject_manifest(run_name="GUI2", status="cancelled", cancelled_at="2026-09-21T10:00:00Z"),
        )
        _write(meta / "history/GUI1/20260101T000000Z/GUI1_orchestration_manifest.json", _subject_manifest())
        _write(meta / "tasks/GUI1/segmentation_ddseg/manifest.json", {"kind": "slicerddsurfer_task_manifest", "version": 1})
        _write(meta / "GUI1_run_plan.json", {"kind": "slicerddsurfer_run_plan", "version": 1})
        batch = _write(
            self.root / "batch_outputs/00_batch_metadata/batch_20260923_101010.json",
            {"kind": BATCH_MANIFEST_KIND, "version": 1, "subjects": [], "results": []},
        )
        batch_subject = _write(
            self.root / "batch_outputs/sub-02/00_run_metadata/Test1_orchestration_manifest.json",
            _subject_manifest(subject_id="sub-02", run_name="Test1", ended_at="2026-09-23T11:00:00Z"),
        )

        discovery = discover_runs([self.root])
        paths = [record.manifest_path for record in discovery.records]
        self.assertEqual(set(paths), {gui, cli, cancelled, batch, batch_subject})
        self.assertEqual(paths[0], batch_subject, "newest first")
        self.assertEqual(paths[-1], gui)
        self.assertFalse(discovery.truncated)

    def test_invalid_manifests_are_skipped_with_reason(self) -> None:
        meta = self.root / "outputs/00_run_metadata"
        _write(meta / "good_orchestration_manifest.json", _subject_manifest())
        _write(meta / "bad_orchestration_manifest.json", _subject_manifest(version=99))
        discovery = discover_runs([self.root])
        self.assertEqual(len(discovery.records), 1)
        self.assertEqual(len(discovery.skipped), 1)
        self.assertIn("version 99", discovery.skipped[0][1])

    def test_depth_limit_and_manifest_file_roots(self) -> None:
        deep = _write(
            self.root / "a/b/c/d/00_run_metadata/deep_orchestration_manifest.json", _subject_manifest()
        )
        self.assertEqual(discover_manifest_paths([self.root])[0], [])
        self.assertEqual(discover_manifest_paths([self.root], max_depth=5)[0], [deep])
        self.assertEqual(discover_manifest_paths([deep])[0], [deep])
        self.assertEqual(discover_manifest_paths([deep.parent])[0], [deep])

    def test_directory_cap_marks_truncated(self) -> None:
        for index in range(12):
            (self.root / f"d{index:02d}").mkdir()
        _paths, truncated = discover_manifest_paths([self.root], max_dirs=5)
        self.assertTrue(truncated)

    def test_symlink_cycles_do_not_loop_and_duplicates_collapse(self) -> None:
        manifest = _write(self.root / "real/00_run_metadata/r_orchestration_manifest.json", _subject_manifest())
        try:
            os.symlink(self.root, self.root / "real" / "loop")
            os.symlink(self.root / "real", self.root / "alias")
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")
        paths, _ = discover_manifest_paths([self.root, self.root / "alias"], max_depth=6)
        self.assertEqual(len(paths), 1)
        self.assertEqual(os.path.realpath(paths[0]), os.path.realpath(manifest))

    def test_missing_roots_are_ignored(self) -> None:
        self.assertEqual(discover_runs([self.root / "nope", "", None]).records, ())


class RootsAndSettingsTest(unittest.TestCase):
    def test_candidate_roots_cover_output_subject_batch_and_extra(self) -> None:
        roots = candidate_run_roots(
            subject_root_text="/data/sub-01",
            output_root_text="",
            batch_root_text="/data/study",
            extra_dirs=["/data/old_runs", "", "/data/sub-01"],
        )
        self.assertEqual(
            [root.as_posix() for root in roots],
            ["/data/sub-01/outputs", "/data/sub-01", "/data/study/batch_outputs", "/data/study", "/data/old_runs"],
        )

    def test_explicit_output_root_replaces_default_outputs(self) -> None:
        roots = candidate_run_roots(subject_root_text="/s", output_root_text="/o")
        self.assertEqual([root.as_posix() for root in roots], ["/o", "/s"])

    def test_remember_directory_is_mru_and_capped(self) -> None:
        self.assertEqual(remember_directory(["/a", "/b"], "/b"), ("/b", "/a"))
        self.assertEqual(remember_directory(["/a", "/b", "/c"], "/d", limit=2), ("/d", "/a"))
        self.assertEqual(remember_directory(["/a", ""], ""), ("/a",))


class CancelledManifestRowsTest(unittest.TestCase):
    def test_gui_cancellation_manifest_yields_cancelled_rows_per_target(self) -> None:
        from shared.ui_run_controller import cancelled_manifest_path_and_payload, rows_from_single_subject_manifest

        with tempfile.TemporaryDirectory() as tmp:
            path, payload = cancelled_manifest_path_and_payload(
                {"subject_root": tmp, "subject_id": "sub-01", "run_name": "GUI1", "targets": ["segmentation", "connectome"]},
                cancelled_at="2026-09-26T10:00:00Z",
            )
            _write(path, payload)
            record = parse_run_manifest(path)
            self.assertEqual(record.status, "cancelled")
            rows, counts = rows_from_single_subject_manifest(payload)
        self.assertEqual([row["target"] for row in rows], ["segmentation", "connectome"])
        self.assertEqual({row["status"] for row in rows}, {"cancelled"})
        self.assertIn("cancelled", rows[0]["reason"])
        self.assertEqual(counts["failed"], 0)

    def test_candidate_roots_prioritize_subject_subfolder_when_subject_id_given(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "Downloads"
            roots = candidate_run_roots(output_root_text=str(out), subject_id="try1")
            self.assertEqual(roots[0], out / "try1" / "outputs")
            self.assertIn(out / "try1", roots)
            self.assertIn(out, roots)
            batch_roots = candidate_run_roots(output_root_text=str(out), subject_id="try1", batch_mode=True)
            self.assertEqual(batch_roots[0], out)

    def test_discovery_finds_scheme_a_outputs_and_skips_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Deep cache dirs must be skipped so max_dirs budget is not exhausted.
            for i in range(12):
                (root / "cache" / f"dir_{i:02d}" / "nested").mkdir(parents=True)
            scheme_a_manifest = _write(
                root / "sub-01" / "outputs" / "00_run_metadata" / "Test1_orchestration_manifest.json",
                _subject_manifest(subject_id="sub-01", run_name="Test1", status="completed"),
            )
            discovery = discover_runs([root], max_depth=3, max_dirs=5)
            self.assertEqual(len(discovery.records), 1)
            self.assertEqual(discovery.records[0].manifest_path, scheme_a_manifest)

    def test_discovery_prioritizes_metadata_folders_and_deduplicates_cancelled_companion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(12):
                (root / f"a{i:02d}" / "nested").mkdir(parents=True)
            meta = root / "try1" / "00_run_metadata"
            primary = _write(
                meta / "Test1_orchestration_manifest.json",
                _subject_manifest(
                    subject_id="try1",
                    run_name="Test1",
                    status="cancelled",
                    tasks=[{"name": "microstructure.dti", "status": "completed"}],
                    completed_at="2026-11-18T18:57:20Z",
                ),
            )
            _write(
                meta / "Test1_cancelled_orchestration_manifest.json",
                {
                    "kind": SUBJECT_MANIFEST_KIND,
                    "version": 1,
                    "subject_id": "try1",
                    "run_name": "Test1",
                    "status": "cancelled",
                    "targets": ["microstructure"],
                    "cancelled_at": "2026-11-18T18:57:21Z",
                },
            )
            discovery = discover_runs([root], max_depth=3, max_dirs=5)
            self.assertEqual(len(discovery.records), 1)
            self.assertEqual(discovery.records[0].manifest_path, primary)


class HiddenRunsTest(unittest.TestCase):
    """Batch 2, Results "Remove from Results list" / "Clear": the identity of a run slot and the stamp rule."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def subject(self, folder: str, name: str, **fields):
        path = _write(self.root / folder / "00_run_metadata" / name, _subject_manifest(**fields))
        return parse_run_manifest(path)

    def test_b2_results_key_is_the_discover_runs_slot_for_main_and_cancelled_companion(self) -> None:
        main = self.subject("s1", "Test1_orchestration_manifest.json", ended_at="2026-11-01T10:00:00Z")
        companion = self.subject("s1", "Test1_cancelled_orchestration_manifest.json", cancelled_at="2026-11-01T11:00:00Z")
        other_name = self.subject("s1", "Test2_orchestration_manifest.json")
        other_folder = self.subject("s2", "Test1_orchestration_manifest.json")
        self.assertEqual(run_record_key(main), run_record_key(companion))
        self.assertNotEqual(run_record_key(main), run_record_key(other_name))
        self.assertNotEqual(run_record_key(main), run_record_key(other_folder))
        self.assertEqual(run_manifest_key(main.manifest_path), run_record_key(main))
        self.assertEqual(run_manifest_key(companion.manifest_path), run_record_key(main))
        # discover_runs keeps one record per key.
        self.assertEqual(len(discover_runs([self.root]).records), 3)

    def test_b2_results_batch_key_is_the_real_manifest_path(self) -> None:
        path = _write(self.root / "b" / "00_batch_metadata" / "batch_20261101_100000.json",
                      {"kind": BATCH_MANIFEST_KIND, "version": 1, "subjects": [], "results": []})
        record = parse_run_manifest(path)
        self.assertIn(os.path.realpath(path), run_record_key(record))
        self.assertTrue(run_record_key(record).startswith("batch"))

    def test_b2_results_stamp_rule_equal_stays_hidden_newer_shows(self) -> None:
        old = self.subject("s1", "Test1_orchestration_manifest.json", ended_at="2026-11-01T10:00:00Z")
        other = self.subject("s2", "Test1_orchestration_manifest.json", ended_at="2026-11-01T09:00:00Z")
        hidden = {run_record_key(old): hidden_stamp(old)}
        shown, gone = split_hidden((old, other), hidden)
        self.assertEqual((shown, gone), ((other,), (old,)))
        rerun = self.subject("s1", "Test1_orchestration_manifest.json", ended_at="2026-11-02T09:00:00Z")
        self.assertEqual(split_hidden((rerun, other), hidden), ((rerun, other), ()))
        cancelled = self.subject("s1", "Test1_cancelled_orchestration_manifest.json",
                                 status="cancelled", cancelled_at="2026-11-01T10:05:00Z")
        self.assertEqual(split_hidden((cancelled,), hidden), ((cancelled,), ()))

    def test_b2_results_missing_stamps(self) -> None:
        record = self.subject("s1", "Test1_orchestration_manifest.json", ended_at="2026-11-01T10:00:00Z")
        no_stamp = replace(record, timestamp=None)
        key = run_record_key(record)
        self.assertIsNone(hidden_stamp(no_stamp))
        self.assertEqual(split_hidden((no_stamp,), {key: None}), ((), (no_stamp,)))
        self.assertEqual(split_hidden((no_stamp,), {key: 5.0}), ((), (no_stamp,)))
        self.assertEqual(split_hidden((record,), {key: None}), ((record,), ()))

    def test_b2_results_stored_value_round_trip_and_broken_values_read_as_empty(self) -> None:
        hidden = {"subject\t/a\tTest1": 1700000000.25, "batch\t/b.json": None}
        self.assertEqual(parse_hidden_runs(dump_hidden_runs(hidden)), hidden)
        for broken in (None, "", "not json", "[1, 2]", "42", b"{}", ["{}"], {"a": 1}):
            with self.subTest(value=broken):
                self.assertEqual(parse_hidden_runs(broken), {})
        self.assertEqual(parse_hidden_runs('{"a": 1.5, "b": "x", "c": true, "": 2}'), {"a": 1.5})

    def test_b2_results_stored_map_is_capped_and_drops_the_oldest_stamps(self) -> None:
        hidden = {f"k{i}": float(i) for i in range(10)}
        hidden["none"] = None
        kept = parse_hidden_runs(dump_hidden_runs(hidden, limit=4))
        self.assertEqual(set(kept), {"k6", "k7", "k8", "k9"})


class ProgressTest(unittest.TestCase):
    PLAN = {
        "items": [
            {"target": "preprocessing", "task_name": "preprocessing.dwi", "will_run": False, "status": "skipped"},
            {"target": "segmentation", "task_name": "segmentation.ddseg", "will_run": True},
            {"target": "parcellation", "task_name": "parcellation.ddparcel", "will_run": True},
            {"target": "microstructure", "task_name": "microstructure.dti", "will_run": True},
        ]
    }

    def test_progress_counts_will_run_items_and_names_current_stage(self) -> None:
        manifest = {
            "status": "running",
            "tasks": [
                {"name": "segmentation.ddseg", "status": "completed"},
                {"name": "segmentation.extra_substage", "status": "completed"},
            ],
        }
        snapshot = progress_from_payloads(manifest, self.PLAN)
        self.assertEqual((snapshot.done, snapshot.total), (1, 3))
        self.assertEqual(snapshot.current_stage, "parcellation.ddparcel")
        self.assertEqual(snapshot.percent, 33)

    def test_progress_matches_by_target_prefix_and_clamps(self) -> None:
        manifest = {
            "status": "completed",
            "tasks": [
                {"name": "segmentation.ddseg", "status": "completed"},
                {"name": "parcellation.other_name", "status": "failed"},
                {"name": "microstructure.dti", "status": "blocked"},
            ],
        }
        snapshot = progress_from_payloads(manifest, self.PLAN)
        self.assertEqual((snapshot.done, snapshot.total, snapshot.current_stage), (3, 3, ""))
        self.assertEqual(snapshot.fraction, 1.0)

    def test_progress_without_plan_counts_finished_tasks(self) -> None:
        snapshot = progress_from_payloads({"tasks": [{"name": "a", "status": "completed"}]}, None)
        self.assertEqual((snapshot.done, snapshot.total), (1, 1))
        self.assertEqual(progress_from_payloads({}, None).fraction, 0.0)

    def test_progress_from_manifest_reads_plan_and_ignores_stale_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            meta = Path(tmp) / "00_run_metadata"
            plan_path = _write(meta / "GUI1_run_plan.json", self.PLAN)
            manifest = manifest_path_for_plan(plan_path, "GUI1")
            self.assertEqual(manifest.name, "GUI1_orchestration_manifest.json")
            self.assertIsNone(progress_from_manifest(manifest))
            _write(
                manifest,
                _subject_manifest(
                    status="running",
                    run_plan_path=plan_path.as_posix(),
                    tasks=[{"name": "segmentation.ddseg", "status": "completed"}],
                ),
            )
            snapshot = progress_from_manifest(manifest)
            self.assertEqual((snapshot.done, snapshot.total), (1, 3))
            old = time.time() - 3600
            os.utime(manifest, (old, old))
            self.assertIsNone(progress_from_manifest(manifest, not_before=time.time()))
            self.assertIsNotNone(progress_from_manifest(manifest, not_before=old))


class RunFolderPathTest(unittest.TestCase):
    """Batch 2, Results "Open folder": the folder of a run record on disk (pure; the UI checks that it exists)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_b2_open_folder_a_subject_record_gives_the_folder_that_holds_input_and_outputs(self) -> None:
        path = _write(self.root / "out" / "sub1" / "outputs" / "00_run_metadata" / "R1_orchestration_manifest.json",
                      _subject_manifest(output_root=str(self.root / "elsewhere" / "outputs")))
        record = parse_run_manifest(path)
        # The place of the manifest wins over the stored output_root: a moved run folder keeps working.
        self.assertEqual(run_folder_path(record), self.root / "out" / "sub1")

    def test_b2_open_folder_a_subject_record_with_a_manifest_outside_the_usual_layout_uses_output_root(self) -> None:
        path = _write(self.root / "loose" / "R1_orchestration_manifest.json",
                      _subject_manifest(output_root=str(self.root / "ws" / "sub9" / "outputs")))
        self.assertEqual(run_folder_path(parse_run_manifest(path)), self.root / "ws" / "sub9")
        path = _write(self.root / "loose" / "R2_orchestration_manifest.json",
                      _subject_manifest(output_root=str(self.root / "ws" / "custom")))
        self.assertEqual(run_folder_path(parse_run_manifest(path)), self.root / "ws" / "custom")

    def test_b2_open_folder_a_batch_record_gives_its_batch_output_folder(self) -> None:
        # run_batch_pipeline: the manifest is <output_base>/00_batch_metadata/batch_*.json; ``output_root`` is stored
        # only with --output-root, and then it is the output base. An existing stored root wins; a stale one does not.
        payload = {"kind": BATCH_MANIFEST_KIND, "version": 1, "run_name": "x", "subjects": [], "results": [],
                   "ended_at": "2026-11-04T10:00:00Z"}
        elsewhere = self.root / "elsewhere"
        elsewhere.mkdir()
        custom = _write(self.root / "b1" / "00_batch_metadata" / "batch_1.json", {**payload, "output_root": str(elsewhere)})
        self.assertEqual(run_folder_path(parse_run_manifest(custom)), elsewhere)  # --manifest put the file elsewhere
        stale = _write(self.root / "b2" / "00_batch_metadata" / "batch_2.json",
                       {**payload, "output_root": str(self.root / "gone")})
        self.assertEqual(run_folder_path(parse_run_manifest(stale)), self.root / "b2")  # the run folder was moved
        unset = _write(self.root / "b3" / "00_batch_metadata" / "batch_3.json", {**payload, "output_root": None})
        self.assertEqual(run_folder_path(parse_run_manifest(unset)), self.root / "b3")
        loose = _write(self.root / "loose_batch.json", payload)
        self.assertEqual(run_folder_path(parse_run_manifest(loose)), self.root)

    def test_b2_open_folder_the_last_resort_is_the_folder_of_the_manifest_file(self) -> None:
        record = replace(parse_run_manifest(_write(self.root / "x" / "R_orchestration_manifest.json",
                                                   _subject_manifest())), output_root="")
        self.assertEqual(run_folder_path(record), self.root / "x")  # the folder of the file is the last resort
        self.assertIsNone(subject_folder_from_roots(None, None))
        self.assertIsNone(subject_folder_from_roots("", " "))

    def test_b2_open_folder_roots_of_a_running_job(self) -> None:
        self.assertEqual(subject_folder_from_roots(self.root / "o" / "s1" / "outputs", self.root / "in" / "s1"),
                         self.root / "o" / "s1")
        self.assertEqual(subject_folder_from_roots(None, self.root / "in" / "s1"), self.root / "in" / "s1")
        self.assertEqual(subject_folder_from_roots(self.root / "o" / "s1", None), self.root / "o" / "s1")


if __name__ == "__main__":
    unittest.main()
