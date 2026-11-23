"""Results: the files of a COMPLETED bundle task whose recorded QC is not passed (failed or unreadable).

The Results tree lists them with a mark on the subgroup row; only the tree and the load of one clicked row see them.
Bulk loading (load key results, the automatic set) keeps today's filter and skip messages.
"""

from __future__ import annotations

import importlib
import json
import shutil
import sys
import tempfile
import types
import unittest
from win_test_helpers import skip_unless_symlinks
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from shared.result_scene import result_candidates as REAL_CANDIDATES  # before any Harness patches the module

WMQL_FILES = ("AF_left.trk", "AF_right.trk", "CST_left.trk")


class Run:
    """A run folder: a WMQL task whose QC record failed, a TractCloud task whose QC record is unreadable and a
    failed WMQL@csd_act task."""

    def __init__(self, root: Path):
        self.root = root
        out = root / "outputs"
        self.bundles = out / "04_bundles"
        stage = root / "cache" / "stages" / "04_bundles" / "wmql" / "Test1"
        stage.mkdir(parents=True)
        published = self.bundles / "query_bundle_extraction"
        published.mkdir(parents=True)
        stage_files, self.wmql_files = [], []  # the QC record names the stage files; Results shows the published copies
        for name in WMQL_FILES:
            path = stage / name
            path.write_text("trk", encoding="utf-8")
            stage_files.append(path)
            self.wmql_files.append(Path(shutil.copy2(path, published / name)))
        self.wmql_qc = stage / "wmql_qc.json"
        self.wmql_qc.write_text(json.dumps({
            "kind": "wmql_bundle_segmentation_qc", "status": "completed", "version": 6,
            "automatic_qc_passed": False, "checks": {},
            "output_summary": {"bundle_files": [str(path) for path in stage_files], "bundle_count": 3}}),
            encoding="utf-8")
        cloud = self.bundles / "bundle_identification" / "AF.vtp"
        cloud.parent.mkdir(parents=True, exist_ok=True)
        cloud.write_text("vtp", encoding="utf-8")
        self.cloud_file = cloud
        self.cloud_qc = root / "cache" / "stages" / "04_bundles" / "tract_cloud_dl" / "tractcloud_qc.json"
        self.cloud_qc.parent.mkdir(parents=True)
        self.cloud_qc.write_text("{not json", encoding="utf-8")
        failed = self.bundles / "query_bundle_extraction" / "csd_act" / "AF.trk"
        failed.parent.mkdir(parents=True, exist_ok=True)
        failed.write_text("trk", encoding="utf-8")
        self.failed_file = failed
        self.manifest = {
            "kind": "slicerddsurfer_product_orchestration", "version": 1, "status": "completed",
            "subject_id": "sub1", "run_name": "Test1", "ended_at": "2026-11-01T10:00:00Z",
            "subject_root": str(root), "output_root": str(out),
            "tasks": [
                {"name": "bundles.wmql", "status": "completed", "qc_path": str(self.wmql_qc),
                 "outputs": {"output_dir": str(self.bundles / "query_bundle_extraction"), "qc_path": str(self.wmql_qc)}},
                {"name": "bundles.tractcloud", "status": "completed", "qc_path": str(self.cloud_qc),
                 "outputs": {"AF": str(cloud), "output_dir": str(cloud.parent)}},
                {"name": "bundles.wmql@csd_act", "status": "failed",
                 "outputs": {"AF": str(failed)}},
            ]}
        meta = out / "00_run_metadata"
        meta.mkdir(parents=True)
        self.manifest_path = meta / "Test1_orchestration_manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")


class TempRootTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name).resolve()
        self.run = Run(self.root)


class CandidatesTest(TempRootTest):
    def test_b2_qc_default_result_candidates_still_skip_the_not_passed_tasks_with_the_old_messages(self) -> None:
        from shared.result_scene import result_candidates

        candidates, skipped = result_candidates(self.run.manifest)
        self.assertEqual(candidates, [])
        reasons = {entry["product"]: (entry["reason"], entry.get("reason_category")) for entry in skipped}
        self.assertEqual(reasons["bundles.wmql"], ("Recorded automatic QC failed", "qc_failed"))
        self.assertEqual(reasons["bundles.tractcloud"], ("Unreadable bundle QC", "qc_failed"))
        self.assertEqual(reasons["bundles.wmql@csd_act"][1], "task_failed")

    def test_b2_qc_the_flag_lists_the_files_of_completed_tasks_with_a_mark_and_keeps_failed_tasks_hidden(self) -> None:
        from shared.result_scene import result_candidates

        candidates, skipped = result_candidates(self.run.manifest, include_qc_not_passed=True,
                                                allow_bundle_directory_scan=False)
        by_path = {item["path"]: item for item in candidates}
        for path in self.run.wmql_files:
            item = by_path[str(path)]
            self.assertEqual((item["product"], item["qc_mark"], item["qc_reason"]),
                             ("bundles.wmql", "failed", "Recorded automatic QC failed"))
        cloud = by_path[str(self.run.cloud_file)]
        self.assertEqual((cloud["qc_mark"], cloud["qc_reason"]), ("unreadable", "Unreadable bundle QC"))
        self.assertNotIn(str(self.run.failed_file), by_path)
        self.assertEqual([entry["product"] for entry in skipped if entry.get("reason_category")],
                         ["bundles.wmql@csd_act"])

    def test_b2_qc_a_task_without_any_qc_record_is_marked_as_no_record(self) -> None:
        from shared.result_scene import result_candidates

        task = dict(self.run.manifest["tasks"][1])
        task.pop("qc_path")
        candidates, _ = result_candidates({"tasks": [task]}, include_qc_not_passed=True)
        self.assertEqual([(item["path"], item["qc_mark"]) for item in candidates],
                         [(str(self.run.cloud_file), "unreadable")])

    def test_b2_qc_a_passed_task_gets_no_mark(self) -> None:
        from shared.result_scene import result_candidates

        with mock.patch("shared.result_scene.bundle_record_error", return_value=None):
            candidates, _ = result_candidates({"tasks": [self.run.manifest["tasks"][0]]}, include_qc_not_passed=True)
        self.assertEqual(len(candidates), 3)
        self.assertTrue(all("qc_mark" not in item for item in candidates))

    def test_b2_qc_the_automatic_set_is_unchanged(self) -> None:
        from shared.auto_results import automatic_result_candidates

        selected, skipped, _ = automatic_result_candidates(self.run.manifest)
        self.assertEqual(selected, [])
        self.assertIn("qc_failed", {entry.get("reason_category") for entry in skipped})


class PublishedTask:
    """Review M1: a bundle task shaped like a real published one (``_publish_bundles_outputs``): the task records
    ``output_dir``, ``manifest_path``, ``qc_path`` and ``bundle_count``, not one output per file. The backend's own
    manifest (``wmql_manifest.json``) names the STAGE files in ``outputs.summary.bundle_files``; the published copies
    lie in ``output_dir`` under the same names."""

    def __init__(self, root: Path, *, name="bundles.wmql", summary_key="bundle_files", manifest_name="wmql_manifest.json",
                 qc_name="wmql_qc.json", suffix=".trk"):
        self.stage = root / "cache" / "stages" / "04_bundles" / name.split(".", 1)[1] / "Test1"
        self.stage.mkdir(parents=True)
        self.output_dir = root / "outputs" / "04_bundles" / name.split(".", 1)[1] / "csd_act"
        self.output_dir.mkdir(parents=True)
        self.names = [f"Test1_desc-{name}_{bundle}{suffix}" for bundle in ("af_left", "af_right", "cst_left")]
        stage_files = []
        self.published = []
        for file_name in self.names:
            stage_file = self.stage / file_name
            stage_file.write_text("bundle", encoding="utf-8")
            stage_files.append(str(stage_file))
            self.published.append(Path(shutil.copy2(stage_file, self.output_dir / file_name)))
        empty = self.stage / f"Test1_desc-{name}_empty{suffix}"
        empty.write_text("", encoding="utf-8")
        (self.output_dir / empty.name).write_text("", encoding="utf-8")
        summary = {summary_key: stage_files + [str(empty)]}
        if summary_key == "bundle_files":
            summary["empty_bundle_files"] = [str(empty)]
        else:
            summary["anatomical_empty_tracts"] = [empty.name]
        self.manifest_file = self.stage / manifest_name
        self.manifest_file.write_text(json.dumps({"kind": "bundle_manifest", "outputs": {
            "qc": str(self.stage / qc_name), "summary": summary}}), encoding="utf-8")
        self.qc_file = self.stage / qc_name  # not written: the QC record is missing
        self.task = {"name": name, "status": "skipped", "reason": "completed outputs found",
                     "manifest_path": str(self.manifest_file), "qc_path": str(self.qc_file),
                     "outputs": {"output_dir": str(self.output_dir), "bundle_count": "3"}}


class UnreadableQcPublishedTaskTest(unittest.TestCase):
    """Review M1 (Codex P1): with a missing or invalid QC record, a real published task got no rows and no mark."""

    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()

    def _rows(self, task, **kwargs):
        from shared.result_scene import result_candidates

        kwargs.setdefault("include_qc_not_passed", True)
        kwargs.setdefault("allow_bundle_directory_scan", False)  # as the Results tree calls it
        candidates, skipped = result_candidates({"tasks": [task]}, **kwargs)
        return candidates, skipped

    def test_b2_morph_fix_M1_a_missing_qc_record_lists_the_bundle_manifest_files_with_the_no_record_mark(self) -> None:
        for name, key, manifest_name, suffix in (
                ("bundles.wmql", "bundle_files", "wmql_manifest.json", ".trk"),
                ("bundles.tractcloud", "bundle_files", "tractcloud_manifest.json", ".vtp"),
                ("bundles.wma", "anatomical_tract_files", "wma_org_atlas_manifest.json", ".vtp")):
            with self.subTest(task=name):
                task = PublishedTask(self.root / name, name=name, summary_key=key, manifest_name=manifest_name,
                                     suffix=suffix)
                candidates, _ = self._rows(task.task)
                self.assertEqual(sorted(item["path"] for item in candidates), sorted(map(str, task.published)))
                self.assertTrue(all(item["qc_mark"] == "unreadable" for item in candidates))
                self.assertEqual({item["key"] for item in candidates}, {"bundle_" + file for file in task.names})

    def test_b2_morph_fix_M1_an_invalid_qc_record_lists_the_bundle_manifest_files_with_the_no_record_mark(self) -> None:
        task = PublishedTask(self.root)
        task.qc_file.write_text("{not json", encoding="utf-8")
        candidates, _ = self._rows(task.task)
        self.assertEqual(sorted(item["path"] for item in candidates), sorted(map(str, task.published)))
        self.assertEqual({(item["qc_mark"], item["qc_reason"]) for item in candidates},
                         {("unreadable", "Unreadable bundle QC")})

    def test_b2_morph_fix_M1_without_the_bundle_manifest_the_task_stays_unlisted(self) -> None:
        task = PublishedTask(self.root)
        task.manifest_file.unlink()
        self.assertEqual(self._rows(task.task)[0], [])
        task.manifest_file.write_text("{not json", encoding="utf-8")
        self.assertEqual(self._rows(task.task)[0], [])

    def test_b2_morph_fix_M1_with_the_flag_off_the_task_is_skipped_as_before(self) -> None:
        task = PublishedTask(self.root)
        candidates, skipped = self._rows(task.task, include_qc_not_passed=False)
        self.assertEqual(candidates, [])
        self.assertEqual([(entry["product"], entry["reason"], entry["reason_category"]) for entry in skipped],
                         [("bundles.wmql", "Unreadable bundle QC", "qc_failed")])
        from shared.result_scene import bundle_outputs

        self.assertEqual(bundle_outputs(task.task), {})  # the default (bulk and automatic) resolution is unchanged

    @skip_unless_symlinks
    def test_b2_morph_fix_M1_a_file_outside_the_output_dir_is_not_listed(self) -> None:
        task = PublishedTask(self.root)
        outside = self.root / "elsewhere" / "Test1_desc-bundles.wmql_outside.trk"
        outside.parent.mkdir()
        outside.write_text("bundle", encoding="utf-8")
        escape = task.output_dir / "link_out.trk"  # inside by name, outside once resolved
        escape.symlink_to(outside)
        record = json.loads(task.manifest_file.read_text(encoding="utf-8"))
        record["outputs"]["summary"]["bundle_files"] += [str(outside), str(task.stage / "link_out.trk")]
        task.manifest_file.write_text(json.dumps(record), encoding="utf-8")
        candidates, _ = self._rows(task.task)
        self.assertEqual(sorted(item["path"] for item in candidates), sorted(map(str, task.published)))

    def test_b2_morph_fix3_M1_a_same_name_file_of_another_variant_is_not_listed(self) -> None:
        """Third round (Opus and Codex re-review): the manifest route must keep the published-copy check of the QC
        route. Another tracking variant overwrote one published file in a shared folder; the task's own stage file
        still exists and differs, so the published file is not this task's file and must not be listed."""
        task = PublishedTask(self.root)
        task.published[0].write_text("the bundle of another tracking variant", encoding="utf-8")
        candidates, _ = self._rows(task.task)
        self.assertEqual(sorted(item["path"] for item in candidates), sorted(map(str, task.published[1:])))

    def test_b2_morph_fix_M1_the_tree_shows_the_rows_with_the_no_record_mark(self) -> None:
        from shared import ui_run_history
        from shared.run_history import parse_run_manifest
        from test_ui_results_tree import USER_ROLE, Harness

        task = PublishedTask(self.root)
        out = self.root / "outputs"
        manifest = {"kind": "slicerddsurfer_product_orchestration", "version": 1, "status": "completed",
                    "subject_id": "sub1", "run_name": "Test1", "ended_at": "2026-11-01T10:00:00Z",
                    "subject_root": str(self.root), "output_root": str(out), "tasks": [task.task]}
        meta = out / "00_run_metadata"
        meta.mkdir(parents=True)
        path = meta / "Test1_orchestration_manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        h = Harness(self)
        with mock.patch("shared.result_scene.result_candidates", side_effect=REAL_CANDIDATES):
            h.refresh([parse_run_manifest(path)])
        subject = next(item for item in h.tree.items if item.text(0).startswith("sub1"))
        role_key = USER_ROLE + ui_run_history.ROLE_KEY
        groups = {subject.child(i).data(0, role_key): subject.child(i) for i in range(subject.childCount())}
        self.assertIn("bundles", groups, "the task's bundle files are not listed")
        bundles = groups["bundles"]
        subgroups = [bundles.child(i) for i in range(bundles.childCount())]
        self.assertEqual(len(subgroups), 1, [row.data(0, role_key) for row in subgroups])
        wmql = subgroups[0]
        self.assertTrue(str(wmql.data(0, role_key)).startswith("bundles::wmql"), wmql.data(0, role_key))
        self.assertTrue(wmql.text(0).startswith("WMQL"), wmql.text(0))
        self.assertTrue(wmql.text(0).endswith("(3) · no QC record"), wmql.text(0))
        self.assertIn("Unreadable bundle QC", wmql.tooltips[0])


class LoadGateTest(TempRootTest):
    def _logic(self):
        with mock.patch.dict(sys.modules, {"slicer": types.ModuleType("slicer")}):
            module_logic = importlib.import_module("shared.module_logic")

        class Logic(module_logic.SharedLogicMixin):
            def log(self, message):
                pass

        fake = SimpleNamespace(mrmlScene=mock.MagicMock(), util=SimpleNamespace(getNodesByClass=lambda kind: []))
        patcher = mock.patch.object(module_logic, "slicer", fake)
        patcher.start()
        self.addCleanup(patcher.stop)
        return Logic()

    def test_b2_qc_only_the_load_of_one_clicked_row_turns_the_flag_on(self) -> None:
        logic = self._logic()
        seen = []

        def spy(manifest, **kwargs):
            seen.append(kwargs)
            return [], []

        with mock.patch("shared.result_scene.result_candidates", side_effect=spy):
            logic.loadKeyResultsFromManifest(self.run.manifest_path)
            logic.loadKeyResultsFromManifest(self.run.manifest_path, source_path=self.run.wmql_files[0])
        self.assertFalse(seen[0].get("include_qc_not_passed", False))
        self.assertTrue(seen[1].get("include_qc_not_passed"))

    def test_b2_qc_one_clicked_row_reaches_the_fibre_loader_and_bulk_does_not(self) -> None:
        logic = self._logic()
        calls = []
        logic._load_fiber_bundle_node = lambda path, name, **kwargs: calls.append(Path(path)) or False
        bulk = logic.loadKeyResultsFromManifest(self.run.manifest_path)
        self.assertEqual(calls, [])
        self.assertIn("Recorded automatic QC failed", [entry.get("reason") for entry in bulk["skipped"]])
        logic.loadKeyResultsFromManifest(self.run.manifest_path, source_path=self.run.wmql_files[1])
        self.assertEqual(calls, [self.run.wmql_files[1]])


class TreeTest(TempRootTest):
    def _tree(self, language="en"):
        from shared import ui_run_history
        from shared.run_history import parse_run_manifest
        from test_ui_results_tree import USER_ROLE, Harness

        h = Harness(self, language=language)
        with mock.patch("shared.result_scene.result_candidates", side_effect=REAL_CANDIDATES):
            h.refresh([parse_run_manifest(self.run.manifest_path)])
        subject = next(item for item in h.tree.items if item.text(0).startswith("sub1"))
        role_key = USER_ROLE + ui_run_history.ROLE_KEY
        bundles = next(subject.child(i) for i in range(subject.childCount())
                       if subject.child(i).data(0, role_key) == "bundles")
        rows = {bundles.child(i).data(0, role_key): bundles.child(i) for i in range(bundles.childCount())}
        return h, bundles, rows, USER_ROLE, ui_run_history

    def test_b2_qc_the_tree_lists_the_subgroups_with_a_mark_and_keeps_key_order_and_count(self) -> None:
        h, bundles, rows, USER_ROLE, urh = self._tree()
        self.assertEqual(list(rows), ["bundles::wmql", "bundles::tractcloud"])
        wmql, cloud = rows["bundles::wmql"], rows["bundles::tractcloud"]
        self.assertEqual(wmql.text(0), "WMQL (3) · QC not passed")  # the QC JSON row is in the cache: not listed
        self.assertEqual(cloud.text(0), "TractCloud (1) · no QC record")
        self.assertIn("Recorded automatic QC failed", wmql.tooltips[0])
        self.assertIn("Unreadable bundle QC", cloud.tooltips[0])
        self.assertTrue(bundles.text(0).startswith("Fibre bundles (4)"))
        leaves = [wmql.child(i) for i in range(wmql.childCount())]
        trk = [leaf for leaf in leaves if leaf.text(0).endswith(".trk")]
        self.assertEqual(len(trk), 3)
        self.assertTrue(all(leaf.text(1) != "↗" and 1 in leaf.icons for leaf in trk))  # normal rows with an eye
        self.assertEqual(urh._item_key(h.qt, wmql), ("subgroup", "sub1", "bundles", "bundles::wmql"))
        self.h, self.rows = h, rows

    def test_b2_qc_the_mark_follows_the_language_in_place_and_in_a_rebuild(self) -> None:
        from shared.ui_onboarding import apply_onboarding_language

        self.test_b2_qc_the_tree_lists_the_subgroups_with_a_mark_and_keeps_key_order_and_count()
        apply_onboarding_language(self.h.owner, "zh")
        self.assertEqual(self.rows["bundles::wmql"].text(0), "WMQL (3) · QC 未通过")
        self.assertEqual(self.rows["bundles::tractcloud"].text(0), "TractCloud (1) · 无 QC 记录")
        self.assertIn("QC 未通过", self.rows["bundles::wmql"].tooltips[0])
        apply_onboarding_language(self.h.owner, "en")
        self.assertEqual(self.rows["bundles::wmql"].text(0), "WMQL (3) · QC not passed")
        _h, _bundles, rows, _role, _urh = self._tree(language="zh")
        self.assertEqual(rows["bundles::wmql"].text(0), "WMQL (3) · QC 未通过")


if __name__ == "__main__":
    unittest.main()
