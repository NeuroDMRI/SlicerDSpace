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
        patch = mock.patch("shared.result_scene.owned_nodes", side_effect=owned)
        patch.start()
        self.addCleanup(patch.stop)
        return loads

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


if __name__ == "__main__":
    unittest.main()
