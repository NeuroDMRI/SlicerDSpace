#!/usr/bin/env python3
"""Results: the files of archived runs of a subject (history manifests).

User report (batch 2, subject 3167): "The Results part seems incomplete. I ran many things, for example bundles and
tractography, but Results shows only preprocessing, microstructure and quality control. Every result that was run
should be shown." Cause: every run used one run name; a new run moves the earlier manifest to
``00_run_metadata/history/<run>/<UTC stamp>/`` and the Results tree read only the current manifest.

The tests reuse the working fakes of ``test_ui_results_tree`` (fake ``qt``, tree, items, menus, settings).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import shared.result_scene as result_scene
from shared import run_history, ui_run_history
from shared.run_history import SUBJECT_MANIFEST_KIND, parse_run_manifest
from test_ui_results_tree import (
    USER_ROLE,
    FakeNode,
    FakeSettings,
    Harness,
    child_texts,
    file_rows,
    find_child,
    make_manifest,
)

ROLE_PATH = USER_ROLE + ui_run_history.ROLE_PATH
ROLE_MANIFEST = USER_ROLE + ui_run_history.ROLE_MANIFEST
ROLE_RUN = USER_ROLE + ui_run_history.ROLE_RUN
REAL_RESULT_CANDIDATES = result_scene.result_candidates


def archive_manifest(root: Path, subject: str, run: str, folder: str, ended_at, candidates=(), *, tasks=None,
                     status: str = "completed") -> Path:
    """Write ``<subject>/outputs/00_run_metadata/history/<run>/<folder>/<run>_orchestration_manifest.json``."""

    path = (root / subject / "outputs" / "00_run_metadata" / "history" / run / folder
            / f"{run}_orchestration_manifest.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"kind": SUBJECT_MANIFEST_KIND, "version": 1, "status": status, "subject_id": subject,
               "run_name": run, "tasks": list(tasks or []),
               "cands": [{"product": product, "path": file} for product, file in candidates]}
    if ended_at:
        payload["ended_at"] = ended_at
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def archive_current(record, folder: str) -> Path:
    """What ``_archive_existing_run_records`` does when a run of the same name starts: move the manifest away."""

    target = record.manifest_path.parent / "history" / record.run_name / folder / record.manifest_path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(record.manifest_path), str(target))
    return target


def every_file(item) -> list:
    return [leaf.data(0, ROLE_PATH) for leaf in file_rows(item)]


class ArchivedTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        getattr(run_history, "clear_archived_manifest_cache", lambda: None)()

    def harness(self, records, **kwargs) -> Harness:
        h = Harness(self, **kwargs)
        h.tree.viewport = lambda: SimpleNamespace(mapToGlobal=lambda position: position)
        h.refresh(records)
        return h

    def subject(self, h: Harness, name: str = "sub1"):
        return next(item for item in h.tree.items if item.text(0).startswith(name))

    def shape_3167(self):
        """The report: the current manifest has microstructure only; three archives hold the other products."""

        current = make_manifest(self.root, "sub1", "Test1", "2026-11-21T10:02:00Z", [
            ("preprocessing.dwi", "/x/dwi.nii.gz"), ("microstructure.dti", "/x/fa.nii.gz")])
        archive_manifest(self.root, "sub1", "Test1", "20261121T093748306561Z", "2026-11-21T09:37:00Z", [
            ("preprocessing.dwi", "/x/dwi.nii.gz"), ("segmentation.ddseg", "/x/tissue.nii.gz"),
            ("surface.ddsurfer", "/x/lh.white.vtk"), ("tractography.ukf", "/x/ukf.vtk")])
        archive_manifest(self.root, "sub1", "Test1", "20261121T095457753382Z", "2026-11-21T09:54:00Z", [
            ("tractography.ukf", "/x/ukf.vtk"), ("bundles.wmql", "/x/wmql_AF.trk"),
            ("bundles.tractcloud", "/x/tc_AF.vtp"), ("parcellation.ddparcel", "/x/parc.nii.gz")])
        archive_manifest(self.root, "sub1", "Test1", "20261121T100043181692Z", "2026-11-21T10:00:00Z", [
            ("connectome.streamline_count", "/x/count.csv"), ("microstructure.dti", "/x/fa.nii.gz")])
        return current


class ShapeTest(ArchivedTestBase):
    def test_archived_runs_3167_shape_every_family_shows_each_file_once_and_the_run_list_is_unchanged(self) -> None:
        """Rules 1 and 2: the archived manifests add their files; they are not new runs."""

        current = self.shape_3167()
        h = self.harness([current])
        subject = self.subject(h)
        self.assertEqual(child_texts(subject), [
            "Preprocessing (1)", "Anatomical priors (3)", "Microstructure (1)", "Tractography (1)",
            "Fibre bundles (2)", "Connectome (1)"])
        files = every_file(subject)
        self.assertEqual(len(files), len(set(files)))
        self.assertEqual(sorted(files), sorted([
            "/x/dwi.nii.gz", "/x/fa.nii.gz", "/x/tissue.nii.gz", "/x/lh.white.vtk", "/x/parc.nii.gz",
            "/x/ukf.vtk", "/x/wmql_AF.trk", "/x/tc_AF.vtp", "/x/count.csv"]))
        # The run list, the tree records, the combo box and the hidden count get no entry for an archive.
        self.assertEqual(h.owner._runHistoryRecords, (current,))
        self.assertEqual(h.owner._runTreeRecords, (current,))
        self.assertEqual([item["data"] for item in h.combo.items if item["data"]], [current.manifest_path.as_posix()])
        self.assertEqual(h.owner._resultsHiddenRecords, ())
        self.assertEqual(h.tree.topLevelItemCount, 1)

    def test_archived_runs_discover_runs_still_lists_only_the_current_manifest(self) -> None:
        """Rule 2: ``discover_manifest_paths`` and ``discover_runs`` keep their results; a separate function lists
        the archives of one run record."""

        current = self.shape_3167()
        found = run_history.discover_runs([self.root])
        self.assertEqual([record.manifest_path for record in found.records], [current.manifest_path])
        archives = run_history.archived_run_manifests(current)
        self.assertEqual([entry.path.parent.name for entry in archives],
                         ["20261121T100043181692Z", "20261121T095457753382Z", "20261121T093748306561Z"])

    def test_archived_runs_a_path_of_the_current_manifest_and_an_archive_shows_once_with_the_current_manifest(self):
        """Rule 3 (b): the newer manifest gives the row."""

        current = self.shape_3167()
        h = self.harness([current])
        micro = find_child(self.subject(h), "Microstructure")
        leaves = file_rows(micro)
        self.assertEqual([leaf.data(0, ROLE_PATH) for leaf in leaves], ["/x/fa.nii.gz"])
        self.assertEqual(leaves[0].data(0, ROLE_MANIFEST), str(current.manifest_path))
        self.assertEqual(leaves[0].data(0, ROLE_RUN), 0)
        # Review R5: the archive that also names fa.nii.gz was read (its archive-only file shows), so the single
        # fa row above is the result of the dedupe and not of an archive that was never found.
        count = file_rows(find_child(self.subject(h), "Connectome"))
        self.assertEqual([leaf.data(0, ROLE_PATH) for leaf in count], ["/x/count.csv"])
        self.assertEqual(Path(count[0].data(0, ROLE_MANIFEST)).parent.name, "20261121T100043181692Z")

    def test_archived_runs_a_row_of_an_archive_carries_the_archive_and_the_index_of_its_owning_record(self) -> None:
        """Rule 4: ROLE_MANIFEST is the archived manifest; ROLE_RUN is the index of the owning current record."""

        other = make_manifest(self.root, "sub1", "Other", "2026-11-22T10:00:00Z", [("qc.summary", "/x/qc.txt")])
        current = self.shape_3167()
        h = self.harness([other, current])
        bundles = find_child(self.subject(h), "Fibre bundles")
        leaf = next(row for row in file_rows(bundles) if row.data(0, ROLE_PATH) == "/x/wmql_AF.trk")
        archive = (current.manifest_path.parent / "history" / "Test1" / "20261121T095457753382Z"
                   / "Test1_orchestration_manifest.json")
        self.assertEqual(leaf.data(0, ROLE_MANIFEST), str(archive))
        self.assertEqual(leaf.data(0, ROLE_RUN), 1)  # Test1 is index 1; the archive belongs to Test1, not Other

    def test_archived_runs_order_is_newest_manifest_first_across_archives(self) -> None:
        """Rule 3 (c): time = the manifest's own time; the folder name only breaks a missing time."""

        current = make_manifest(self.root, "sub1", "Test1", "2026-11-10T10:00:00Z", [("qc.summary", "/x/qc.txt")])
        newer = archive_manifest(self.root, "sub1", "Test1", "20261101T000000000000Z", "2026-11-05T10:00:00Z",
                                 [("tractography.ukf", "/x/ukf.vtk"), ("tractography.ukf", "/x/ukf_b.vtk")])
        archive_manifest(self.root, "sub1", "Test1", "20261109T000000000000Z", "2026-11-02T10:00:00Z",
                         [("tractography.ukf", "/x/ukf.vtk"), ("tractography.ukf", "/x/ukf_a.vtk")])
        h = self.harness([current])
        ukf = find_child(find_child(self.subject(h), "Tractography"), "UKF")
        self.assertEqual([leaf.data(0, ROLE_PATH) for leaf in file_rows(ukf)],
                         ["/x/ukf.vtk", "/x/ukf_b.vtk", "/x/ukf_a.vtk"])
        self.assertEqual(file_rows(ukf)[0].data(0, ROLE_MANIFEST), str(newer))

    def test_archived_runs_an_archive_of_another_run_name_goes_to_the_record_of_that_name(self) -> None:
        """Rule 2: an archive goes to the record with the same run name, else to the newest record of the folder."""

        run_a = make_manifest(self.root, "sub1", "A", "2026-11-03T10:00:00Z", [("qc.summary", "/x/a.txt")])
        run_b = make_manifest(self.root, "sub1", "B", "2026-11-04T10:00:00Z", [("qc.summary", "/x/b.txt")])
        archive_manifest(self.root, "sub1", "A", "20261101T000000000000Z", "2026-11-01T10:00:00Z",
                         [("tractography.ukf", "/x/a_ukf.vtk")])
        archive_manifest(self.root, "sub1", "Gone", "20261101T000000000000Z", "2026-11-01T09:00:00Z",
                         [("tractography.ukf", "/x/gone_ukf.vtk")])
        self.assertEqual([e.path.parent.parent.name for e in run_history.archived_run_manifests(run_a, (run_a, run_b))],
                         ["A"])
        self.assertEqual([e.path.parent.parent.name for e in run_history.archived_run_manifests(run_b, (run_a, run_b))],
                         ["Gone"])
        h = self.harness([run_b, run_a])
        ukf = find_child(find_child(self.subject(h), "Tractography"), "UKF")
        runs = {leaf.data(0, ROLE_PATH): leaf.data(0, ROLE_RUN) for leaf in file_rows(ukf)}
        self.assertEqual(runs, {"/x/a_ukf.vtk": 1, "/x/gone_ukf.vtk": 0})


class FilterTest(ArchivedTestBase):
    def test_archived_runs_a_missing_file_is_not_shown_and_a_broken_archive_is_skipped(self) -> None:
        """Rule 5 (d): the same task filter as for a current manifest; a broken archive gives no error."""

        files = self.root / "files"
        files.mkdir()
        (files / "ukf.vtk").write_text("x", encoding="utf-8")
        (files / "fa.nii.gz").write_text("x", encoding="utf-8")
        current = make_manifest(self.root, "sub1", "Test1", "2026-11-10T10:00:00Z", [])
        payload = json.loads(current.manifest_path.read_text(encoding="utf-8"))
        payload.pop("cands")
        payload["tasks"] = [{"name": "microstructure.dti", "status": "completed",
                             "outputs": {"fa": str(files / "fa.nii.gz")}}]
        current.manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        archive_manifest(self.root, "sub1", "Test1", "20261102T000000000000Z", "2026-11-02T10:00:00Z", tasks=[
            {"name": "tractography.ukf", "status": "completed",
             "outputs": {"tractogram": str(files / "ukf.vtk"), "gone": str(files / "gone.vtk")}},
            {"name": "bundles.tractcloud", "status": "failed", "outputs": {"x": str(files / "ukf.vtk")}}])
        broken = archive_manifest(self.root, "sub1", "Test1", "20261103T000000000000Z", "2026-11-03T10:00:00Z")
        broken.write_text("{not json", encoding="utf-8")
        wrong_kind = archive_manifest(self.root, "sub1", "Test1", "20261104T000000000000Z", "2026-11-04T10:00:00Z")
        wrong_kind.write_text(json.dumps({"kind": "other", "version": 1}), encoding="utf-8")
        h = Harness(self)
        h.tree.viewport = lambda: SimpleNamespace(mapToGlobal=lambda position: position)
        with mock.patch("shared.result_scene.result_candidates", side_effect=REAL_RESULT_CANDIDATES):
            h.refresh([parse_run_manifest(current.manifest_path)])
        subject = self.subject(h)
        self.assertEqual(child_texts(subject), ["Microstructure (1)", "Tractography (1)"])
        self.assertEqual(every_file(find_child(subject, "Tractography")), [str((files / "ukf.vtk").resolve())])
        h.slicer.util.errorDisplay.assert_not_called()

    def test_archived_runs_the_manifest_cache_is_refreshed_when_the_file_changes(self) -> None:
        """Rule 7 (g): one parse per (path, mtime, size); a changed file is parsed again."""

        current = make_manifest(self.root, "sub1", "Test1", "2026-11-10T10:00:00Z", [("qc.summary", "/x/qc.txt")])
        path = archive_manifest(self.root, "sub1", "Test1", "20261102T000000000000Z", "2026-11-02T10:00:00Z",
                                [("tractography.ukf", "/x/old.vtk")])
        h = self.harness([current])
        self.assertIn("/x/old.vtk", every_file(self.subject(h)))
        real_load = run_history.load_manifest_json
        with mock.patch.object(run_history, "load_manifest_json", side_effect=real_load) as load:
            h.refresh([current])
            self.assertEqual([c.args[0] for c in load.call_args_list if Path(c.args[0]) == path], [])  # cache hit
            archive_manifest(self.root, "sub1", "Test1", "20261102T000000000000Z", "2026-11-02T10:00:00Z",
                             [("tractography.ukf", "/x/new_name.vtk")])
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
            h.refresh([current])
        files = every_file(self.subject(h))
        self.assertIn("/x/new_name.vtk", files)
        self.assertNotIn("/x/old.vtk", files)


class HideTest(ArchivedTestBase):
    """Rule 6: an archive follows its owning record; a hide stamp keeps the archives from before the hide hidden."""

    def setup_hidden(self):
        self.record = make_manifest(self.root, "sub1", "Test1", "2026-11-01T10:00:00Z",
                                    [("microstructure.dti", "/x/fa.nii.gz")])
        archive_manifest(self.root, "sub1", "Test1", "20261030T000000000000Z", "2026-10-30T10:00:00Z",
                         [("tractography.ukf", "/x/before_hide.vtk")])
        h = self.harness([self.record], settings=FakeSettings())
        self.assertIn("/x/before_hide.vtk", every_file(self.subject(h)))
        ui_run_history._hide_records(h.owner, [self.record])
        return h

    def test_archived_runs_hide_state_1_a_hidden_record_hides_its_archives(self) -> None:
        h = self.setup_hidden()
        self.assertEqual(h.tree.topLevelItemCount, 0)
        self.assertEqual(h.owner._resultsHiddenRecords, (self.record,))  # the count shows the record, not archives

    def test_archived_runs_hide_state_2_a_newer_run_shows_only_the_archives_newer_than_the_hide(self) -> None:
        h = self.setup_hidden()
        # Two re-runs of the same run name after the hide: each moves the current manifest to history/.
        archive_current(self.record, "20261102T000000000000Z")  # the manifest that was hidden
        rerun1 = make_manifest(self.root, "sub1", "Test1", "2026-11-02T10:00:00Z", [
            ("bundles.wmql", "/x/after_hide.trk")])
        archive_current(rerun1, "20261103T000000000000Z")
        rerun2 = make_manifest(self.root, "sub1", "Test1", "2026-11-03T10:00:00Z", [("qc.summary", "/x/qc.txt")])
        h.refresh([rerun2])
        files = every_file(self.subject(h))
        self.assertEqual(sorted(files), ["/x/after_hide.trk", "/x/qc.txt"])
        self.assertEqual(h.owner._resultsHiddenRecords, ())

    def test_archived_runs_hide_state_3_show_hidden_brings_back_the_record_and_all_its_archives(self) -> None:
        h = self.setup_hidden()
        ui_run_history.show_hidden_results(h.owner)
        self.assertEqual(sorted(every_file(self.subject(h))), ["/x/before_hide.vtk", "/x/fa.nii.gz"])


class RowActionTest(ArchivedTestBase):
    def scene(self, h: Harness):
        """A tiny scene: the fake logic loads one model node owned by the manifest path it is given."""

        self.nodes: list = []
        loads: list = []

        class Node(FakeNode):
            def IsA(self, class_name):
                return class_name == self.class_name

        def load(manifest_path, *, source_path=None, **_kw):
            loads.append((Path(manifest_path), Path(source_path)))
            node = Node(Path(source_path).name, "vtkMRMLModelNode", attributes={
                "SlicerDDSurfer.Run": str(Path(manifest_path).resolve()),
                "SlicerDDSurfer.SourcePath": str(source_path)})
            self.nodes.append(node)
            return {"loaded": [node], "skipped": []}

        def owned(_slicer, run_id):
            return [node for node in self.nodes if node.GetAttribute("SlicerDDSurfer.Run") == str(run_id)]

        h.owner.logic = SimpleNamespace(loadKeyResultsFromManifest=load)
        h.owner.parent = None
        h.slicer.util.getNodesByClass.side_effect = lambda name: [n for n in self.nodes if n.class_name == name]
        h.slicer.mrmlScene.RemoveNode.side_effect = lambda node: self.nodes.remove(node)
        # Both scene readers: ``owned_nodes`` (one manifest) and ``result_nodes`` (every result node of the scene).
        self.owned_patch = mock.patch("shared.result_scene.owned_nodes", side_effect=owned)
        self.scan_patch = mock.patch("shared.result_scene.result_nodes", side_effect=lambda _slicer: list(self.nodes))
        for patch in (self.owned_patch, self.scan_patch):
            patch.start()
            self.addCleanup(patch.stop)
        return loads

    def model_node(self, source: str, manifest: Path, class_name: str = "vtkMRMLModelNode", **attributes):
        """A node that the plugin loaded earlier: owned by ``manifest`` (resolved, as the logic writes it)."""

        class Node(FakeNode):
            def IsA(self, name):
                return name == self.class_name

        node = Node(Path(source).name, class_name, attributes={
            "SlicerDDSurfer.Run": str(Path(manifest).resolve()), "SlicerDDSurfer.SourcePath": source, **attributes})
        self.nodes.append(node)
        return node

    def test_archived_runs_the_eye_of_an_archived_row_loads_and_hide_everything_hides_it(self) -> None:
        """Rule 4 (f): eye toggle, icon refresh and "hide everything shown" on a row of an archive."""

        current = self.shape_3167()
        h = self.harness([current])
        loads = self.scene(h)
        surfaces = find_child(find_child(self.subject(h), "Anatomical priors"), "Cortical surfaces")
        leaf = file_rows(surfaces)[0]
        archive = Path(leaf.data(0, ROLE_MANIFEST))
        self.assertEqual(archive.parent.parent.name, "Test1")  # history/Test1/<stamp>/
        self.assertEqual(leaf.tooltips[1], "Show")
        h.tree.itemClicked.emit(leaf, 1)  # the eye
        h.slicer.util.errorDisplay.assert_not_called()
        self.assertEqual(loads, [(archive, Path("/x/lh.white.vtk"))])
        self.assertEqual([node.GetDisplayVisibility() for node in self.nodes], [1])
        self.assertEqual(leaf.tooltips[1], "Hide")  # the icon refresh found the node of the archive
        self.assertEqual(h.combo.currentIndex, 0)  # the click selected the owning run
        ui_run_history.hide_everything_shown(h.owner)
        self.assertEqual([node.GetDisplayVisibility() for node in self.nodes], [0])
        self.assertEqual(leaf.tooltips[1], "Show")
        h.tree.itemDoubleClicked.emit(leaf, 0)  # a double click shows the same node again (no second load)
        self.assertEqual(len(loads), 1)
        self.assertEqual([node.GetDisplayVisibility() for node in self.nodes], [1])

    def test_archived_runs_edit_tissue_on_an_archived_row_uses_the_archive_and_selects_the_owning_run(self) -> None:
        """Rule 4: the right-click menu works for a row of an archive like for a row of the current manifest."""

        other = make_manifest(self.root, "sub2", "R", "2026-11-23T10:00:00Z", [("qc.summary", "/x/q.txt")])
        current = self.shape_3167()
        h = self.harness([other, current])
        self.assertEqual(h.combo.currentIndex, 0)
        seg = file_rows(find_child(find_child(self.subject(h), "Anatomical priors"), "Tissue segmentation"))[0]
        archive = Path(seg.data(0, ROLE_MANIFEST))
        seen = []
        h.owner._manage_result_scene = lambda mode: seen.append(
            (mode, ui_run_history.validated_selected_manifest(h.owner)[0]))
        h.tree.itemAt = lambda _position: seg
        h.pick = lambda menu: next(action for action in menu.actions if action.text.startswith("Edit tissue"))
        ui_run_history._output_menu(h.owner, (3, 3))
        self.assertEqual(seen, [("edit_tissue", archive)])
        self.assertEqual(h.combo.currentIndex, 1)  # the owning run (Test1) is selected
        self.assertEqual(h.owner._latestResultManifestPath, current.manifest_path)  # restored after the action

    def test_archived_runs_a_rebuild_keeps_the_current_archived_row(self) -> None:
        """Rule 4: restore of the current item after a rebuild works for a row of an archive."""

        current = self.shape_3167()
        h = self.harness([current])
        leaf = file_rows(find_child(self.subject(h), "Fibre bundles"))[0]
        h.tree.setCurrentItem(leaf)
        h.refresh([current])
        self.assertEqual(h.tree.currentItem().data(0, ROLE_PATH), leaf.data(0, ROLE_PATH))
        ui_run_history.show_record(h.owner, current, json.loads(current.manifest_path.read_text("utf-8")))
        self.assertEqual(h.tree.currentItem().data(0, ROLE_PATH), leaf.data(0, ROLE_PATH))
        self.assertEqual(len(file_rows(self.subject(h))), 9)  # the show_record rebuild keeps the archives


# --------------------------------------------------------------------------------------- review fix round (R1-R5)

class FixR1NodesTest(RowActionTest):
    """Review R1: the nodes of a row are the nodes with the row's file that ANY manifest of the row's owning record
    owns (its current manifest and its archives). A node loaded before a re-run is owned by the current manifest
    path; after the re-run the row comes from the archive."""

    def rerun_without_tractography(self):
        record = make_manifest(self.root, "sub1", "Test1", "2026-11-01T10:00:00Z", [
            ("microstructure.dti", "/x/fa.nii.gz"), ("tractography.ukf", "/x/ukf.vtk")])
        h = self.harness([record])
        loads = self.scene(h)
        ukf = file_rows(find_child(find_child(self.subject(h), "Tractography"), "UKF"))[0]
        h.tree.itemClicked.emit(ukf, 1)  # run 1: the eye loads the tractogram, owned by the CURRENT manifest path
        self.assertEqual(loads, [(record.manifest_path, Path("/x/ukf.vtk"))])
        archive = archive_current(record, "20261102T000000000000Z")
        rerun = make_manifest(self.root, "sub1", "Test1", "2026-11-02T10:00:00Z", [
            ("microstructure.dti", "/x/fa.nii.gz")])
        h.refresh([rerun])
        ukf = file_rows(find_child(find_child(self.subject(h), "Tractography"), "UKF"))[0]
        self.assertEqual(ukf.data(0, ROLE_MANIFEST), str(archive))  # settled: the row carries the archive
        return h, loads, ukf, rerun

    def test_b2_results_fix_R1_the_eye_of_an_archived_row_sees_a_node_loaded_before_the_rerun(self) -> None:
        h, loads, ukf, _rerun = self.rerun_without_tractography()
        self.assertEqual(len(self.nodes), 1)
        self.assertEqual(ukf.tooltips[1], "Hide fiber bundle")  # the eye is on: the node is visible
        h.tree.itemClicked.emit(ukf, 1)
        h.slicer.util.errorDisplay.assert_not_called()
        self.assertEqual(len(loads), 1)  # no second copy
        self.assertEqual(len(self.nodes), 1)
        self.assertEqual(self.nodes[0].GetDisplayVisibility(), 0)  # the click hid the node
        self.assertEqual(ukf.tooltips[1], "Show as fiber bundle")

    def test_b2_results_fix_R1_unload_is_offered_for_a_node_loaded_before_the_rerun(self) -> None:
        h, _loads, ukf, _rerun = self.rerun_without_tractography()
        offered = []
        h.tree.itemAt = lambda _position: ukf
        h.pick = lambda menu: offered.extend(action.text for action in menu.actions) or next(
            (action for action in menu.actions if action.text == "Unload from scene"), None)
        ui_run_history._output_menu(h.owner, (3, 3))
        self.assertIn("Unload from scene", offered)
        self.assertEqual(self.nodes, [])  # Unload removed the node
        self.assertEqual(ukf.tooltips[1], "Show as fiber bundle")

    def test_b2_results_fix_R1_a_node_of_another_record_with_the_same_file_is_not_matched(self) -> None:
        """Two records must not see each other's nodes: the archive of Test1 is not Other's manifest."""

        current = make_manifest(self.root, "sub1", "Test1", "2026-11-03T10:00:00Z", [("qc.summary", "/x/qc.txt")])
        archive_manifest(self.root, "sub1", "Test1", "20261102T000000000000Z", "2026-11-02T10:00:00Z",
                         [("tractography.ukf", "/x/ukf.vtk")])
        other = make_manifest(self.root, "sub1", "Other", "2026-11-04T10:00:00Z", [("qc.summary", "/x/qc2.txt")])
        h = self.harness([other, current])
        loads = self.scene(h)
        self.model_node("/x/ukf.vtk", other.manifest_path)  # same file, another record
        ui_run_history._refresh_output_visibility(h.owner)
        ukf = file_rows(find_child(find_child(self.subject(h), "Tractography"), "UKF"))[0]
        self.assertEqual(ukf.data(0, ROLE_RUN), 1)  # the row belongs to Test1
        self.assertEqual(ukf.tooltips[1], "Show as fiber bundle")
        h.tree.itemClicked.emit(ukf, 1)
        self.assertEqual(len(loads), 1)  # Test1's archive loads its own copy

    def test_b2_results_fix_R1_one_scene_scan_per_refresh_pass(self) -> None:
        """Opus P2 (same code): one scene scan grouped by the Run attribute, not one ``owned_nodes`` per manifest."""

        current = self.shape_3167()
        h = self.harness([current])
        self.scene(h)
        self.model_node("/x/ukf.vtk", current.manifest_path)
        with mock.patch("shared.result_scene.owned_nodes", side_effect=AssertionError("one scan per manifest")), \
                mock.patch("shared.result_scene.result_nodes", side_effect=lambda _s: list(self.nodes)) as scan:
            ui_run_history._refresh_output_visibility(h.owner)
        self.assertEqual(scan.call_count, 1)
        ukf = file_rows(find_child(find_child(self.subject(h), "Tractography"), "UKF"))[0]
        self.assertEqual(ukf.tooltips[1], "Hide fiber bundle")

    def test_b2_results_fix_R1_edit_tissue_on_an_archived_row_reads_the_manifest_that_owns_the_loaded_labels(self):
        """Edit tissue copy: ``_manage_result_scene`` looks for the labels of ONE manifest; for a row of an archive
        whose label map was loaded before the re-run, that is the current manifest path that owns the node."""

        record = make_manifest(self.root, "sub1", "Test1", "2026-11-01T10:00:00Z", [
            ("segmentation.ddseg", "/x/tissue.nii.gz")])
        h = self.harness([record])
        self.scene(h)
        self.model_node("/x/tissue.nii.gz", record.manifest_path, "vtkMRMLLabelMapVolumeNode",
                        **{"SlicerDDSurfer.Product": "segmentation.ddseg"})
        archive_current(record, "20261102T000000000000Z")
        rerun = make_manifest(self.root, "sub1", "Test1", "2026-11-02T10:00:00Z", [("qc.summary", "/x/qc.txt")])
        h.refresh([rerun])
        seg = file_rows(find_child(find_child(self.subject(h), "Anatomical priors"), "Tissue segmentation"))[0]
        self.assertNotEqual(seg.data(0, ROLE_MANIFEST), str(rerun.manifest_path))
        seen = []
        h.owner._manage_result_scene = lambda mode: seen.append(
            (mode, ui_run_history.validated_selected_manifest(h.owner)[0]))
        h.tree.itemAt = lambda _position: seg
        h.pick = lambda menu: next(action for action in menu.actions if action.text.startswith("Edit tissue"))
        ui_run_history._output_menu(h.owner, (3, 3))
        self.assertEqual(seen, [("edit_tissue", rerun.manifest_path)])


class FixR2ShowHiddenTest(ArchivedTestBase):
    """Review R2 (recorded scope change to rule 6): "Show hidden results" also brings back the stamp-filtered
    archives of records that are shown. They count for the menu entry; the action clears their stamps."""

    def test_b2_results_fix_R2_show_hidden_counts_and_restores_the_hidden_archives_of_a_shown_record(self) -> None:
        a = make_manifest(self.root, "subA", "Test1", "2026-11-01T10:00:00Z", [("microstructure.dti", "/a/fa.nii")])
        archive_manifest(self.root, "subA", "Test1", "20261030T000000000000Z", "2026-10-30T10:00:00Z",
                         [("tractography.ukf", "/a/before_hide.vtk")])
        b = make_manifest(self.root, "subB", "Run", "2026-11-01T11:00:00Z", [("microstructure.dti", "/b/fa.nii")])
        h = self.harness([a, b], settings=FakeSettings())
        ui_run_history._hide_records(h.owner, [a, b])
        # subA re-runs after the hide; subB stays hidden.
        archive_current(a, "20261102T000000000000Z")
        rerun = make_manifest(self.root, "subA", "Test1", "2026-11-02T10:00:00Z", [("qc.summary", "/a/qc.txt")])
        h.refresh([rerun, b])
        self.assertEqual(every_file(self.subject(h, "subA")), ["/a/qc.txt"])  # automatic rule: only the newer run
        self.assertEqual(h.owner._resultsHiddenRecords, (b,))  # an archive is not a run record
        entry = ui_run_history.results_menu_entries(h.owner)[1]
        # One hidden record (subB) + two hidden archives of subA (its manifest from before the hide, and the older).
        self.assertEqual(entry[:2], ("Show hidden results (3)", True))
        entry[3]()  # the action
        self.assertEqual(sorted(every_file(self.subject(h, "subA"))),
                         ["/a/before_hide.vtk", "/a/fa.nii", "/a/qc.txt"])
        self.assertEqual(every_file(self.subject(h, "subB")), ["/b/fa.nii"])
        self.assertEqual(ui_run_history.load_hidden_runs(), {})
        self.assertEqual(ui_run_history.results_menu_entries(h.owner)[1][:2], ("Show hidden results (0)", False))

    def test_b2_results_fix_R2_the_hidden_archives_alone_enable_the_entry(self) -> None:
        """No record is hidden any more; only the stamp-filtered archives remain: the entry is enabled."""

        a = make_manifest(self.root, "subA", "Test1", "2026-11-01T10:00:00Z", [("microstructure.dti", "/a/fa.nii")])
        h = self.harness([a], settings=FakeSettings())
        ui_run_history._hide_records(h.owner, [a])
        archive_current(a, "20261102T000000000000Z")
        rerun = make_manifest(self.root, "subA", "Test1", "2026-11-02T10:00:00Z", [("qc.summary", "/a/qc.txt")])
        h.refresh([rerun])
        self.assertEqual(h.owner._resultsHiddenRecords, ())
        self.assertEqual(ui_run_history.results_menu_entries(h.owner)[1][:2], ("Show hidden results (1)", True))
        ui_run_history.show_hidden_results(h.owner)
        self.assertEqual(sorted(every_file(self.subject(h, "subA"))), ["/a/fa.nii", "/a/qc.txt"])


class FixR3TimeTest(ArchivedTestBase):
    """Review R3: the time of an archive is its recorded time fields, then the ``<stamp>`` folder name; the file
    mtime only when both are missing."""

    def test_b2_results_fix_R3_a_touched_old_archive_without_a_recorded_time_does_not_win_a_path(self) -> None:
        current = make_manifest(self.root, "sub1", "Test1", "2026-11-10T10:00:00Z", [("qc.summary", "/x/qc.txt")])
        recorded = archive_manifest(self.root, "sub1", "Test1", "20261105T000000000000Z", "2026-11-05T10:00:00Z",
                                    [("tractography.ukf", "/x/ukf.vtk")])
        touched = archive_manifest(self.root, "sub1", "Test1", "20261001T000000000000Z", None,
                                   [("tractography.ukf", "/x/ukf.vtk")])
        os.utime(touched)  # copied or touched today: the mtime is newer than every recorded time
        h = self.harness([current])
        ukf = file_rows(find_child(find_child(self.subject(h), "Tractography"), "UKF"))[0]
        self.assertEqual(ukf.data(0, ROLE_MANIFEST), str(recorded))
        entry = next(e for e in run_history.archived_run_manifests(current) if e.path == touched)
        self.assertEqual(run_history.read_archived_manifest(entry)[1].isoformat(), "2026-10-01T00:00:00+00:00")

    def test_b2_results_fix_R3_the_mtime_is_the_last_fallback(self) -> None:
        current = make_manifest(self.root, "sub1", "Test1", "2026-11-10T10:00:00Z", [("qc.summary", "/x/qc.txt")])
        copied = archive_manifest(self.root, "sub1", "Test1", "manual_copy", None, [("tractography.ukf", "/x/u.vtk")])
        entry = next(e for e in run_history.archived_run_manifests(current) if e.path == copied)
        when = run_history.read_archived_manifest(entry)[1]
        self.assertAlmostEqual(when.timestamp(), copied.stat().st_mtime, delta=1.0)


class FixR4BrokenArchiveTest(ArchivedTestBase):
    """Review R4: an archive that parses but has a broken task structure is skipped with one log line; the other
    results still show."""

    def test_b2_results_fix_R4_a_broken_task_structure_in_an_archive_skips_only_that_archive(self) -> None:
        files = self.root / "files"
        files.mkdir()
        for name in ("fa.nii.gz", "ukf.vtk", "old.vtk"):
            (files / name).write_text("x", encoding="utf-8")
        current = make_manifest(self.root, "sub1", "Test1", "2026-11-10T10:00:00Z", [])
        payload = json.loads(current.manifest_path.read_text(encoding="utf-8"))
        payload.pop("cands")
        payload["tasks"] = [{"name": "microstructure.dti", "status": "completed",
                             "outputs": {"fa": str(files / "fa.nii.gz")}}]
        current.manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        good = {"name": "tractography.ukf", "status": "completed", "outputs": {"tractogram": str(files / "ukf.vtk")}}
        broken = archive_manifest(self.root, "sub1", "Test1", "20261105T000000000000Z", "2026-11-05T10:00:00Z",
                                  tasks=[good, {"name": "bundles.wmql", "status": "completed", "outputs": 42}])
        # An older archive with the same good task (and one more file): the broken archive must not have used it up.
        archive_manifest(self.root, "sub1", "Test1", "20261101T000000000000Z", "2026-11-01T10:00:00Z", tasks=[
            good, {"name": "tractography.dti_det", "status": "completed",
                   "outputs": {"tractogram": str(files / "old.vtk")}}])
        h = Harness(self)
        h.tree.viewport = lambda: SimpleNamespace(mapToGlobal=lambda position: position)
        with mock.patch("shared.result_scene.result_candidates", side_effect=REAL_RESULT_CANDIDATES), \
                self.assertLogs("shared.ui_run_history", level="WARNING") as logs:
            h.refresh([parse_run_manifest(current.manifest_path)])
        subject = self.subject(h)
        self.assertEqual(sorted(Path(path).name for path in every_file(subject)), ["fa.nii.gz", "old.vtk", "ukf.vtk"])
        ukf = next(leaf for leaf in file_rows(subject) if leaf.data(0, ROLE_PATH).endswith("ukf.vtk"))
        self.assertEqual(Path(ukf.data(0, ROLE_MANIFEST)).parent.name, "20261101T000000000000Z")
        self.assertEqual(len(logs.output), 1)
        self.assertIn(str(broken), logs.output[0])
        h.slicer.util.errorDisplay.assert_not_called()


class FixR5MemoTest(ArchivedTestBase):
    """Review R5: tests for the skip rule of one subject rebuild (``_without_repeated_tasks`` / ``_task_memo_key``)
    and for the QC-folder memo with ``metadata_dir``."""

    def test_b2_results_fix_R5_an_identical_task_of_an_older_manifest_is_skipped(self) -> None:
        memo = {"tasks": set(), "qc_dirs": {}}
        task = {"name": "tractography.ukf", "status": "completed", "reason": "", "qc_path": "/q.json",
                "outputs": {"tractogram": "/x/ukf.vtk"}}
        newer = {"subject_root": "/s", "output_root": "/s/outputs", "tasks": [task]}
        self.assertIs(ui_run_history._without_repeated_tasks(newer, memo), newer)
        older = dict(newer, tasks=[dict(task)])
        self.assertEqual(ui_run_history._without_repeated_tasks(older, memo)["tasks"], [])

    def test_b2_results_fix_R5_a_task_that_differs_in_a_field_the_listing_reads_is_not_skipped(self) -> None:
        task = {"name": "tractography.ukf", "status": "completed", "reason": "", "qc_path": "/q.json",
                "outputs": {"tractogram": "/x/ukf.vtk"}}
        base = {"subject_root": "/s", "output_root": "/s/outputs", "tasks": [task]}
        changes = {"status": dict(task, status="failed"), "reason": dict(task, reason="qc failed"),
                   "qc_path": dict(task, qc_path="/other.json"),
                   "outputs": dict(task, outputs={"tractogram": "/y.vtk"}),
                   "name": dict(task, name="tractography.csd_act")}
        for field, changed in changes.items():
            with self.subTest(field=field):
                memo = {"tasks": set(), "qc_dirs": {}}
                ui_run_history._without_repeated_tasks(base, memo)
                self.assertEqual(ui_run_history._without_repeated_tasks(dict(base, tasks=[changed]), memo)["tasks"],
                                 [changed])
        for root in ("subject_root", "output_root"):
            with self.subTest(root=root):
                memo = {"tasks": set(), "qc_dirs": {}}
                ui_run_history._without_repeated_tasks(base, memo)
                moved = dict(base, **{root: "/elsewhere"})
                self.assertEqual(ui_run_history._without_repeated_tasks(moved, memo)["tasks"], [task])

    def test_b2_results_fix_R5_a_preprocessing_task_is_never_skipped(self) -> None:
        memo = {"tasks": set(), "qc_dirs": {}}
        task = {"name": "preprocessing.dwi", "status": "completed", "outputs": {"dwi": "/x/dwi.nii.gz"}}
        manifest = {"tasks": [task]}
        self.assertIsNone(ui_run_history._task_memo_key(manifest, task))
        ui_run_history._without_repeated_tasks(manifest, memo)
        self.assertEqual(ui_run_history._without_repeated_tasks(dict(manifest), memo)["tasks"], [task])

    def test_b2_results_fix_R5_an_archive_reads_qc_rows_from_the_owning_records_folder_once(self) -> None:
        """An archive without ``output_root`` falls back to ``<owning 00_run_metadata>/../06_qc`` (not to a folder
        under ``history/``), and the folder is scanned once for all manifests of the rebuild."""

        current = make_manifest(self.root, "sub1", "Test1", "2026-11-10T10:00:00Z", [("qc.summary", "/x/q.txt")])
        # The current manifest names another output root with its own QC folder.
        elsewhere = self.root / "elsewhere"
        (elsewhere / "06_qc").mkdir(parents=True)
        (elsewhere / "06_qc" / "current.txt").write_text("x", encoding="utf-8")
        payload = json.loads(current.manifest_path.read_text(encoding="utf-8"))
        payload["output_root"] = str(elsewhere)
        current.manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        current = parse_run_manifest(current.manifest_path)
        # The archives name no output root: their QC folder is the one next to the owning 00_run_metadata.
        qc_dir = current.manifest_path.parent.parent / "06_qc"
        qc_dir.mkdir()
        (qc_dir / "report.txt").write_text("x", encoding="utf-8")
        newer = archive_manifest(self.root, "sub1", "Test1", "20261102T000000000000Z", "2026-11-02T10:00:00Z",
                                 [("tractography.ukf", "/x/ukf02.vtk")])
        archive_manifest(self.root, "sub1", "Test1", "20261101T000000000000Z", "2026-11-01T10:00:00Z",
                         [("tractography.ukf", "/x/ukf01.vtk")])
        real_rglob = Path.rglob
        scanned = []

        def rglob(path, pattern):
            scanned.append(str(path))
            return real_rglob(path, pattern)

        h = Harness(self)
        real_fill = ui_run_history._fill_subject_children
        with mock.patch.object(Path, "rglob", autospec=True, side_effect=rglob), \
                mock.patch.object(ui_run_history, "_fill_subject_children", side_effect=real_fill) as fills:
            h.refresh([current])
        subject = self.subject(h)
        report = str((qc_dir / "report.txt").resolve())
        self.assertEqual([path for path in every_file(subject) if path.endswith(".txt") and "/x/" not in path],
                         [str((elsewhere / "06_qc" / "current.txt").resolve()), report])
        # Each folder once per subject rebuild (a refresh rebuilds the subject again when it selects the run).
        self.assertEqual(scanned, [str(elsewhere / "06_qc"), str(qc_dir)] * fills.call_count)
        row = next(leaf for leaf in file_rows(subject) if leaf.data(0, ROLE_PATH) == report)
        self.assertEqual(row.data(0, ROLE_MANIFEST), str(newer))


class P2ResultsPolishTest(ArchivedTestBase):
    """Regression tests for archive cache error/eviction, broken-log reset, QGroupBox guard, display cleanup, and language refresh."""

    def test_archive_cache_never_caches_read_error_and_evicts_past_max(self) -> None:
        current = make_manifest(self.root, "sub1", "Test1", "2026-11-10T10:00:00Z", [("qc.summary", "/x/q.txt")])
        broken = archive_manifest(self.root, "sub1", "Test1", "20261102T000000000000Z", "2026-11-02T10:00:00Z", [])
        valid_text = broken.read_text(encoding="utf-8")
        broken.write_text("{not-json", encoding="utf-8")
        entry = next(e for e in run_history.archived_run_manifests(current) if e.path == broken)
        run_history._ARCHIVE_CACHE.clear()
        self.assertIsNone(run_history.read_archived_manifest(entry))
        self.assertEqual(len(run_history._ARCHIVE_CACHE), 0)
        # Fixing the file in place succeeds immediately because the error was not cached.
        broken.write_text(valid_text, encoding="utf-8")
        self.assertIsNotNone(run_history.read_archived_manifest(entry))
        # Bounded eviction check
        with mock.patch.object(run_history, "_ARCHIVE_CACHE_MAX", 2):
            run_history._ARCHIVE_CACHE.clear()
            for idx in range(4):
                p = archive_manifest(self.root, "sub1", "Test1", f"2026110{idx+3}T000000000000Z", f"2026-11-0{idx+3}T10:00:00Z", [])
                e = next(x for x in run_history.archived_run_manifests(current) if x.path == p)
                run_history.read_archived_manifest(e)
            self.assertLessEqual(len(run_history._ARCHIVE_CACHE), 2)

    def test_refresh_run_history_clears_broken_archives_logged_and_language_refreshes_visibility(self) -> None:
        current = make_manifest(self.root, "sub1", "Test1", "2026-11-10T10:00:00Z", [("microstructure.dti", "/x/fa.nii.gz")])
        h = self.harness([current])
        ui_run_history._BROKEN_ARCHIVES_LOGGED.add("/fake/broken.json")
        h.refresh([current])
        self.assertEqual(ui_run_history._BROKEN_ARCHIVES_LOGGED, set())
        with mock.patch.object(ui_run_history, "_refresh_output_visibility") as vis:
            ui_run_history.refresh_results_language(h.owner)
        vis.assert_called_once_with(h.owner)

    def test_unload_removes_owned_display_nodes_with_model_node(self) -> None:
        current = make_manifest(self.root, "sub1", "Test1", "2026-11-10T10:00:00Z", [("tractography.ukf", "/x/ukf.vtk")])
        h = self.harness([current])
        disp = mock.MagicMock(name="ModelDisplay_1")
        node = mock.MagicMock(name="ModelNode_1")
        node.GetNumberOfDisplayNodes.return_value = 1
        node.GetNthDisplayNode.return_value = disp
        h.slicer.mrmlScene = mock.MagicMock()
        item = next(leaf for leaf in file_rows(self.subject(h)) if leaf.data(0, ROLE_PATH) == "/x/ukf.vtk")
        h.tree.itemAt = lambda _pos: item

        class _PickUnloadMenu:
            def __init__(self, _parent=None):
                self.actions = []

            def addAction(self, text):
                act = SimpleNamespace(text=text, enabled=True, toolTip="")
                self.actions.append(act)
                return act

            def addSeparator(self):
                return None

            def exec_(self, _pos):
                return next(a for a in self.actions if "Unload" in a.text)

        h.qt.QMenu = _PickUnloadMenu
        import sys as _sys

        with mock.patch.object(ui_run_history, "_output_nodes", return_value=[node]), \
                mock.patch.object(ui_run_history, "_refresh_output_visibility"), \
                mock.patch.dict(_sys.modules, {"qt": h.qt, "slicer": h.slicer}):
            ui_run_history._output_menu(h.owner, (0, 0))
        removed = [call.args[0] for call in h.slicer.mrmlScene.RemoveNode.call_args_list]
        self.assertIn(disp, removed)
        self.assertIn(node, removed)

    def test_manage_result_scene_clear_covers_archived_manifests_of_same_run(self) -> None:
        import sys as _sys
        from test_ui_results_tree import widget_methods

        current = make_manifest(self.root, "sub1", "Test1", "2026-11-10T10:00:00Z", [("qc.summary", "/x/q.txt")])
        arch = archive_manifest(self.root, "sub1", "Test1", "20261102T000000000000Z", "2026-11-02T10:00:00Z",
                                [("tractography.ukf", "/x/ukf.vtk")])
        fake_slicer = mock.MagicMock()
        fake_slicer.util.confirmYesNoDisplay.return_value = True
        Widget = widget_methods(
            "_manage_result_scene",
            validated_selected_manifest=lambda _owner: (current.manifest_path, None),
            slicer=fake_slicer,
        )
        widget = Widget()
        widget.parent = None
        widget._latestResultManifestPath = current.manifest_path
        widget._resultsRunManifests = {0: (current.manifest_path, arch)}
        cleared_paths = []
        widget.logic = SimpleNamespace(clearResultsFromManifest=lambda mp: (cleared_paths.append(Path(mp)), 1)[1])
        widget.appendLog = lambda _msg: None
        widget._manage_result_scene("clear")
        self.assertIn(current.manifest_path.resolve(), cleared_paths)
        self.assertIn(arch.resolve(), cleared_paths)

    def test_results_tree_omits_ukf_trk_and_cross_archive_dti_trk_when_vtk_is_present(self) -> None:
        files = self.root / "files"
        files.mkdir()
        ukf_vtk = files / "sub1_ukf.vtk"
        ukf_trk = files / "sub1_ukf.trk"
        dti_vtk = files / "sub1_dti.vtk"
        dti_trk = files / "sub1_dti.trk"
        for path in (ukf_vtk, ukf_trk, dti_vtk, dti_trk):
            path.write_text("x", encoding="utf-8")
        current = make_manifest(self.root, "sub1", "Test1", "2026-11-10T10:00:00Z", [])
        payload = json.loads(current.manifest_path.read_text(encoding="utf-8"))
        payload.pop("cands")
        payload["tasks"] = [
            {
                "name": "tractography.ukf",
                "status": "completed",
                "outputs": {"tractogram": str(ukf_vtk), "tractogram_trk": str(ukf_trk)},
            },
            {
                "name": "tractography.dti_det",
                "status": "completed",
                "outputs": {"tractogram_vtk": str(dti_vtk), "tractogram_trk": str(dti_trk)},
            },
        ]
        current.manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        # An older archive had only the standalone DTI .trk before VTK conversion existed.
        archive_manifest(
            self.root,
            "sub1",
            "Test1",
            "20261102T000000000000Z",
            "2026-11-02T10:00:00Z",
            tasks=[{
                "name": "tractography.dti_det",
                "status": "completed",
                "outputs": {"tractogram_trk": str(dti_trk)},
            }],
        )
        h = Harness(self)
        h.tree.viewport = lambda: SimpleNamespace(mapToGlobal=lambda position: position)
        with mock.patch("shared.result_scene.result_candidates", side_effect=REAL_RESULT_CANDIDATES):
            h.refresh([parse_run_manifest(current.manifest_path)])
        tractography = find_child(self.subject(h), "Tractography")
        self.assertEqual(
            sorted(every_file(tractography)),
            sorted([str(ukf_vtk.resolve()), str(dti_vtk.resolve())]),
        )
        self.assertTrue(ukf_trk.is_file())
        self.assertTrue(dti_trk.is_file())


if __name__ == "__main__":
    unittest.main()
