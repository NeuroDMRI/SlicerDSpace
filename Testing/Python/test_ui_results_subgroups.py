#!/usr/bin/env python3
"""Results subgroups: a second-level folder inside the result families.

User request (batch 2): "The Results part can get a second-level folder. For example the second level of Anatomical
shows Tissue segmentation, Cortical surfaces and so on." The tree is Subject -> Family -> Subgroup -> file for the
anatomy, microstructure, tractography, bundles and connectome families; preprocessing and qc stay flat.

The tests reuse the working fakes of ``test_ui_results_tree`` (fake ``qt``, tree, items, menus).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from shared import ui_run_history
from shared.algorithm_selection import LABELS
from shared.ui_onboarding import apply_onboarding_language, onboarding_text
from test_ui_results_tree import (
    USER_ROLE,
    FakeItem,
    FakeSettings,
    Harness,
    child_texts,
    find_child,
    make_manifest,
)

ROLE_PATH = USER_ROLE + ui_run_history.ROLE_PATH
ROLE_RUN = USER_ROLE + ui_run_history.ROLE_RUN
ROLE_MANIFEST = USER_ROLE + ui_run_history.ROLE_MANIFEST
ROLE_KEY = USER_ROLE + ui_run_history.ROLE_KEY

FULL_RUN = [
    ("preprocessing.dwi", "/x/dwi.nii.gz"),
    ("preprocessing.brain_mask", "/x/mask.nii.gz"),
    ("surface.ddsurfer", "/x/lh.white.vtk"),
    ("surface.ddsurfer", "/x/rh.white.vtk"),
    ("segmentation.ddseg", "/x/tissue.nii.gz"),
    ("surface.ddsurfer_postprocess", "/x/lh.thickness"),
    ("postprocess.morphometry", "/x/aseg.stats"),
    ("parcellation.ddparcel", "/x/parc.nii.gz"),
    ("microstructure.noddi", "/x/ndi.nii.gz"),
    ("microstructure.dti", "/x/fa.nii.gz"),
    ("microstructure.dti", "/x/md.nii.gz"),
    ("microstructure.mkcurve", "/x/mk.nii.gz"),
    ("microstructure.ifw", "/x/ifw.nii.gz"),
    ("microstructure.freewater", "/x/fw.nii.gz"),
    ("microstructure.dki", "/x/dki_mk.nii.gz"),
    ("tractography.saf", "/x/saf.tck"),
    ("tractography.act", "/x/act_seeds.nii.gz"),
    ("tractography.csd_act", "/x/act.tck"),
    ("tractography.dti_det", "/x/dti.vtk"),
    ("tractography.ukf", "/x/ukf.vtk"),
    ("bundles.tractcloud", "/x/tc_AF.vtp"),
    ("bundles.wmql", "/x/wmql_AF.trk"),
    ("bundles.wma", "/x/wma_AF.vtp"),
    ("connectome.sift2_weighted", "/x/sift2.csv"),
    ("connectome.streamline_count", "/x/count.csv"),
    ("qc.summary", "/x/qc.txt"),
]


def rows(item) -> list:
    return [item.child(i) for i in range(item.childCount())]


def leaf_names(item) -> list[str]:
    return [row.text(0) for row in rows(item)]


class SubgroupTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def manifest(self, subject="sub1", run="run1", stamp="2026-11-01T10:00:00Z", candidates=None):
        return make_manifest(self.root, subject, run, stamp, list(FULL_RUN if candidates is None else candidates))

    def harness(self, records, **kwargs) -> Harness:
        h = Harness(self, **kwargs)
        h.tree.viewport = lambda: SimpleNamespace(mapToGlobal=lambda position: position)
        h.refresh(records)
        return h

    def subject(self, h: Harness, name: str = "sub1"):
        return next(item for item in h.tree.items if item.text(0).startswith(name))


class LayoutTest(SubgroupTestBase):
    def test_results_subgroups_five_families_get_a_second_level_in_table_order_with_counts(self) -> None:
        """Results subgroups: Subject -> Family -> Subgroup -> file; captions with counts (rules 3 and 5)."""

        h = self.harness([self.manifest()])
        subject = self.subject(h)
        self.assertEqual(child_texts(subject), [
            "Preprocessing (2)", "Anatomical priors (6)", "Microstructure (7)", "Tractography (5)",
            "Fibre bundles (3)", "Connectome (2)", "Quality control (1)"])
        expected = {
            "Anatomical priors": ["Tissue segmentation (1)", "Brain parcellation (1)", "Cortical surfaces (2)",
                                  "Surface postprocess (2)"],
            "Microstructure": ["DTI (2)", "FW (1)", "iFW (1)", "NODDI (1)", "DKI (1)", "MK-Curve (1)"],
            "Tractography": ["UKF (1)", "CSD / ACT (2)", "DTI (1)", "CSD + SAF (1)"],
            "Fibre bundles": ["WMQL (1)", "WMA (1)", "TractCloud (1)"],
            "Connectome": ["Fiber count (1)", "SIFT2 (1)"],
        }
        for family, subgroups in expected.items():
            with self.subTest(family=family):
                self.assertEqual(child_texts(find_child(subject, family)), subgroups)
        anatomy = find_child(subject, "Anatomical priors")
        self.assertEqual(leaf_names(find_child(anatomy, "Cortical surfaces")), ["lh.white.vtk", "rh.white.vtk"])
        self.assertEqual(leaf_names(find_child(anatomy, "Surface postprocess")), ["lh.thickness", "aseg.stats"])
        tracts = find_child(subject, "Tractography")
        self.assertEqual(leaf_names(find_child(tracts, "CSD / ACT")), ["act_seeds.nii.gz", "act.tck"])
        dti = find_child(find_child(subject, "Microstructure"), "DTI")
        for leaf in rows(dti):
            self.assertTrue(leaf.data(0, ROLE_PATH))
            self.assertEqual(leaf.text(1), "")  # the eye column was set by the visibility refresh

    def test_results_subgroups_preprocessing_and_qc_stay_flat(self) -> None:
        """Results subgroups rule 4: preprocessing and qc keep Family -> file; an unknown family stays flat."""

        h = self.harness([self.manifest(candidates=[
            ("preprocessing.dwi", "/x/dwi.nii.gz"), ("qc.summary", "/x/qc.txt"), ("custom.thing", "/x/c.dat")])])
        subject = self.subject(h)
        self.assertEqual(child_texts(subject), ["Preprocessing (1)", "Quality control (1)", "custom (1)"])
        for family in ("Preprocessing", "Quality control", "custom"):
            leaf = find_child(subject, family).child(0)
            self.assertTrue(leaf.data(0, ROLE_PATH), family)
            self.assertEqual(leaf.childCount(), 0)

    def test_results_subgroups_one_subgroup_still_shows_the_level(self) -> None:
        """Results subgroups rule 3: the level is there also when a family has only one subgroup."""

        h = self.harness([self.manifest(candidates=[("microstructure.dti", "/x/fa.nii.gz")])])
        micro = find_child(self.subject(h), "Microstructure")
        self.assertEqual(child_texts(micro), ["DTI (1)"])
        self.assertEqual(leaf_names(micro.child(0)), ["fa.nii.gz"])

    def test_results_subgroups_an_unnamed_product_gets_its_own_subgroup_after_the_named_ones(self) -> None:
        """Results subgroups rule 2: caption = the part of the product after the first dot, as written."""

        h = self.harness([self.manifest(candidates=[
            ("microstructure.fa", "/x/fa.nii.gz"), ("microstructure.dti", "/x/dti_fa.nii.gz"),
            ("tractography.regions.v2", "/x/regions.csv"), ("tractography.ukf", "/x/ukf.vtk")])])
        subject = self.subject(h)
        self.assertEqual(child_texts(find_child(subject, "Microstructure")), ["DTI (1)", "fa (1)"])
        self.assertEqual(child_texts(find_child(subject, "Tractography")), ["UKF (1)", "regions.v2 (1)"])

    def test_results_subgroups_a_tracking_variant_suffix_joins_the_algorithms_subgroup(self) -> None:
        """Results subgroups rule 10: a multi-tracking run names tasks ``<product>@<tracking backend>``
        (shared/multi_tracking.py); the files of several tracking sources land in one subgroup, no third level."""

        h = self.harness([self.manifest(candidates=[
            ("bundles.wmql", "/x/wmql_AF.trk"), ("bundles.wmql@ukf", "/x/ukf/wmql_AF.trk"),
            ("bundles.wma@csd_act", "/x/act/wma_CST.vtp"), ("tractography.csd_act@csd_act", "/x/act.tck"),
            ("tractography.ukf@ukf", "/x/ukf.vtk"), ("connectome.sift2_weighted@csd_act", "/x/sift2.csv")])])
        subject = self.subject(h)
        bundles = find_child(subject, "Fibre bundles")
        self.assertEqual(child_texts(bundles), ["WMQL (2)", "WMA (1)"])
        self.assertEqual(leaf_names(find_child(bundles, "WMQL")), ["wmql_AF.trk", "wmql_AF.trk"])
        self.assertEqual(child_texts(find_child(subject, "Tractography")), ["UKF (1)", "CSD / ACT (1)"])
        self.assertEqual(child_texts(find_child(subject, "Connectome")), ["SIFT2 (1)"])

    def test_results_subgroups_role_key_cannot_collide_with_a_family_or_subject_key(self) -> None:
        """Results subgroups rule 9: a subgroup row carries a ``family::subgroup`` text key."""

        h = self.harness([self.manifest()])
        subject = self.subject(h)
        family_keys = {row.data(0, ROLE_KEY) for row in rows(subject)}
        self.assertTrue(family_keys <= set(ui_run_history.RESULT_FAMILY_ORDER))
        anatomy = find_child(subject, "Anatomical priors")
        keys = [row.data(0, ROLE_KEY) for row in rows(anatomy)]
        self.assertEqual(keys, ["anatomy::segmentation", "anatomy::parcellation", "anatomy::surfaces",
                                "anatomy::postprocess"])
        for row in rows(anatomy):
            self.assertIsNone(row.data(0, ROLE_PATH))
        self.assertFalse(set(keys) & family_keys)
        self.assertNotIn(subject.data(0, ROLE_KEY), keys)

    def test_results_subgroups_captions_use_the_guis_own_names(self) -> None:
        """Results subgroups rule 1: one name for one thing; the captions are the names the GUI already shows."""

        captions = getattr(ui_run_history, "RESULT_SUBGROUP_CAPTIONS", {})
        for language, column in (("en", 0), ("zh", 1)):
            for subgroup, key in (("segmentation", "segmentation"), ("parcellation", "parcellation"),
                                  ("surfaces", "surface"), ("postprocess", "surface_postprocess")):
                self.assertEqual(captions.get(("anatomy", subgroup), ("", ""))[column], onboarding_text(language, key))
        for family, names in (("microstructure", ("dti", "freewater", "ifw", "noddi", "dki")),
                              ("tractography", ("ukf", "csd_act", "dti_det", "saf")),
                              ("bundles", ("wmql", "wma", "tractcloud"))):
            for name in names:
                self.assertEqual(captions.get((family, name)), (LABELS[name], LABELS[name]))
        self.assertEqual(captions.get(("connectome", "streamline_count")), (LABELS["fiber_count"],) * 2)
        self.assertEqual(captions.get(("connectome", "sift2_weighted")), (LABELS["sift2"],) * 2)


class MergeTest(SubgroupTestBase):
    def test_results_subgroups_an_older_run_adds_files_and_subgroups_in_table_order(self) -> None:
        """Results subgroups rule 6: newest run first, each file once, through the subgroup level."""

        new = self.manifest(run="runB", stamp="2026-11-02T10:00:00Z", candidates=[
            ("microstructure.noddi", "/x/ndi.nii.gz"), ("microstructure.dti", "/x/fa.nii.gz")])
        old = self.manifest(run="runA", stamp="2026-11-01T10:00:00Z", candidates=[
            ("microstructure.dti", "/x/fa.nii.gz"), ("microstructure.dti", "/x/md.nii.gz"),
            ("microstructure.freewater", "/x/fw.nii.gz"), ("microstructure.zz_extra", "/x/extra.nii.gz"),
            ("bundles.wma", "/x/wma.vtp")])
        h = self.harness([new, old])
        subject = self.subject(h)
        self.assertEqual(child_texts(subject), ["Microstructure (5)", "Fibre bundles (1)"])
        micro = find_child(subject, "Microstructure")
        self.assertEqual(child_texts(micro), ["DTI (2)", "FW (1)", "NODDI (1)", "zz_extra (1)"])
        dti = find_child(micro, "DTI")
        self.assertEqual(leaf_names(dti), ["fa.nii.gz", "md.nii.gz"])  # fa once: the newest run owns it
        self.assertEqual(dti.child(0).data(0, ROLE_MANIFEST), str(new.manifest_path))
        self.assertEqual(dti.child(1).data(0, ROLE_MANIFEST), str(old.manifest_path))
        # A subgroup carries the run that made it, like a family row.
        self.assertEqual(dti.data(0, ROLE_MANIFEST), str(new.manifest_path))
        self.assertEqual(find_child(micro, "FW").data(0, ROLE_MANIFEST), str(old.manifest_path))
        self.assertEqual(find_child(micro, "FW").data(0, ROLE_RUN), 1)

    def test_results_subgroups_show_record_rebuild_keeps_the_merge(self) -> None:
        """Results subgroups rule 6: the subject rebuild of ``show_record`` merges through the subgroups too."""

        new = self.manifest(run="runB", stamp="2026-11-02T10:00:00Z", candidates=[
            ("microstructure.noddi", "/x/ndi.nii.gz")])
        old = self.manifest(run="runA", stamp="2026-11-01T10:00:00Z", candidates=[
            ("microstructure.dti", "/x/fa.nii.gz"), ("microstructure.noddi", "/x/odi.nii.gz")])
        h = self.harness([new, old])
        manifest = json.loads(new.manifest_path.read_text(encoding="utf-8"))
        ui_run_history.show_record(h.owner, new, manifest)
        micro = find_child(self.subject(h), "Microstructure")
        self.assertEqual(micro.text(0), "Microstructure (3)")
        self.assertEqual(child_texts(micro), ["DTI (1)", "NODDI (2)"])


class ExpansionTest(SubgroupTestBase):
    def two_subjects(self):
        return [self.manifest("sub2", stamp="2026-11-02T10:00:00Z"), self.manifest("sub1")]

    def test_results_subgroups_start_collapsed(self) -> None:
        """Results subgroups rule 7: a subgroup starts collapsed, also under an expanded family."""

        h = self.harness(self.two_subjects())
        micro = find_child(self.subject(h, "sub2"), "Microstructure")
        micro.setExpanded(True)
        h.refresh(self.two_subjects())
        micro = find_child(self.subject(h, "sub2"), "Microstructure")
        self.assertTrue(micro.expanded)
        self.assertFalse(any(row.expanded for row in rows(micro)))

    def test_results_subgroups_user_expansion_survives_a_rebuild_per_subject_family_and_subgroup(self) -> None:
        """Results subgroups rule 7: the remembered state is separate per subject, family and subgroup."""

        h = self.harness(self.two_subjects())
        sub2, sub1 = self.subject(h, "sub2"), self.subject(h, "sub1")
        find_child(find_child(sub2, "Microstructure"), "DTI").setExpanded(True)
        find_child(find_child(sub1, "Tractography"), "DTI").setExpanded(True)
        h.refresh(self.two_subjects())
        sub2, sub1 = self.subject(h, "sub2"), self.subject(h, "sub1")
        self.assertTrue(find_child(find_child(sub2, "Microstructure"), "DTI").expanded)
        self.assertFalse(find_child(find_child(sub2, "Tractography"), "DTI").expanded)
        self.assertFalse(find_child(find_child(sub1, "Microstructure"), "DTI").expanded)
        self.assertTrue(find_child(find_child(sub1, "Tractography"), "DTI").expanded)
        self.assertFalse(find_child(find_child(sub2, "Microstructure"), "NODDI").expanded)
        state = h.owner._resultsExpansion["subgroups"]
        self.assertTrue(state[("sub2", "microstructure", "microstructure::dti")])
        self.assertTrue(state[("sub1", "tractography", "tractography::dti_det")])
        # A user collapse is remembered too, also through the show_record rebuild of one subject.
        find_child(find_child(sub2, "Microstructure"), "DTI").setExpanded(False)
        records = self.two_subjects()
        ui_run_history.show_record(h.owner, records[0], json.loads(records[0].manifest_path.read_text("utf-8")))
        self.assertFalse(find_child(find_child(self.subject(h, "sub2"), "Microstructure"), "DTI").expanded)

    def test_results_subgroups_a_rebuild_keeps_the_current_file_inside_a_subgroup_and_the_scroll(self) -> None:
        """Results subgroups rule 8: ``_item_key`` / ``_restore_current_item`` / ``_view_snapshot`` one level deeper."""

        records = self.two_subjects()
        h = self.harness(records)
        dti = find_child(find_child(self.subject(h, "sub2"), "Microstructure"), "DTI")
        h.tree.setCurrentItem(dti.child(1))
        h.tree.bar.value = 21
        h.refresh(records)
        current = h.tree.currentItem()
        self.assertIsNotNone(current)
        self.assertEqual(current.data(0, ROLE_PATH), "/x/md.nii.gz")
        self.assertEqual(h.tree.bar.value, 21)
        # FD1: the show_record rebuild of the selected subject keeps it too.
        ui_run_history.show_record(h.owner, records[0], json.loads(records[0].manifest_path.read_text("utf-8")))
        current = h.tree.currentItem()
        self.assertIsNotNone(current)
        self.assertEqual(current.data(0, ROLE_PATH), "/x/md.nii.gz")

    def test_results_subgroups_a_rebuild_keeps_a_current_subgroup(self) -> None:
        """Results subgroups rule 8: a subgroup row as the current item survives a rebuild."""

        records = self.two_subjects()
        h = self.harness(records)
        micro = find_child(self.subject(h, "sub2"), "Microstructure")
        h.tree.setCurrentItem(find_child(micro, "NODDI"))
        h.refresh(records)
        current = h.tree.currentItem()
        self.assertIsNotNone(current)
        self.assertEqual(current.text(0), "NODDI (1)")
        self.assertEqual(ui_run_history._item_key(h.qt, current),
                         ("subgroup", "sub2", "microstructure", "microstructure::noddi"))

    def test_results_subgroups_a_collapsed_subgroup_over_the_current_file_stays_collapsed(self) -> None:
        """Results subgroups rule 8 (FD2): the highest collapsed ancestor becomes current, not the file."""

        records = self.two_subjects()
        h = self.harness(records)
        dti = find_child(find_child(self.subject(h, "sub2"), "Microstructure"), "DTI")
        h.tree.setCurrentItem(dti.child(0))  # Qt expands the family and the subgroup
        dti.setExpanded(False)  # the user collapses the subgroup
        h.refresh(records)
        dti = find_child(find_child(self.subject(h, "sub2"), "Microstructure"), "DTI")
        self.assertFalse(dti.expanded)
        self.assertIs(h.tree.currentItem(), dti)


class RowActionTest(SubgroupTestBase):
    def two_runs(self):
        new = self.manifest(run="runB", stamp="2026-11-02T10:00:00Z", candidates=[("microstructure.dti", "/x/fa.nii")])
        old = self.manifest(run="runA", stamp="2026-11-01T10:00:00Z", candidates=[
            ("microstructure.noddi", "/x/ndi.nii"), ("segmentation.ddseg", "/x/tissue_seg.nii.gz")])
        return new, old

    def test_results_subgroups_the_eye_click_on_a_file_inside_a_subgroup_toggles_it_and_selects_its_run(self) -> None:
        """Results subgroups rule 8: the eye (column 1) and the run selection work one level deeper."""

        new, old = self.two_runs()
        h = self.harness([new, old])
        leaf = find_child(find_child(self.subject(h), "Microstructure"), "NODDI").child(0)
        self.assertIn(1, leaf.icons)  # the visibility refresh gave the eye icon to the deeper row
        with mock.patch.object(ui_run_history, "_toggle_output") as toggle:
            h.tree.itemClicked.emit(leaf, 1)
            toggle.assert_called_once_with(h.owner, leaf)
        self.assertEqual(h.combo.currentIndex, 1)  # runA, the run of the file
        with mock.patch.object(ui_run_history, "_toggle_output") as toggle:
            h.tree.itemDoubleClicked.emit(leaf, 0)
            toggle.assert_called_once_with(h.owner, leaf)

    def test_results_subgroups_a_click_on_a_subgroup_selects_its_run_and_toggles_nothing(self) -> None:
        new, old = self.two_runs()
        h = self.harness([new, old])
        noddi = find_child(find_child(self.subject(h), "Microstructure"), "NODDI")
        self.assertNotIn(1, noddi.icons)  # a subgroup row has no eye and no folder button
        with mock.patch.object(ui_run_history, "_toggle_output") as toggle, \
                mock.patch.object(ui_run_history, "_open_item_folder") as folder:
            h.tree.itemClicked.emit(noddi, 1)
            toggle.assert_not_called()
            folder.assert_not_called()
        self.assertEqual(h.combo.currentIndex, 1)

    def test_results_subgroups_remove_on_a_subgroup_acts_on_the_whole_subject(self) -> None:
        """Results subgroups rule 8: ``_removal_for_item`` on a subgroup row acts like on a family row."""

        new, old = self.two_runs()
        h = self.harness([new, old], settings=FakeSettings())
        noddi = find_child(find_child(self.subject(h), "Microstructure"), "NODDI")
        records, reason = ui_run_history._removal_for_item(h.owner, h.qt, noddi)
        self.assertEqual((tuple(records), reason), ((new, old), ""))
        h.tree.itemAt = lambda _position: noddi
        h.pick = lambda menu: next(action for action in menu.actions if action.text == "Remove from Results list")
        ui_run_history._output_menu(h.owner, (3, 3))
        self.assertEqual(h.labels(), [])

    def test_results_subgroups_file_actions_of_a_deeper_row_stay_in_the_context_menu(self) -> None:
        new, old = self.two_runs()
        h = self.harness([new, old])
        leaf = find_child(find_child(self.subject(h), "Anatomical priors"), "Tissue segmentation").child(0)
        h.tree.itemAt = lambda _position: leaf
        opened = []
        h.pick = lambda menu: next(action for action in menu.actions if action.text == "Open folder")
        with mock.patch.object(ui_run_history, "open_results_folder", side_effect=lambda _o, f: opened.append(f)):
            ui_run_history._output_menu(h.owner, (3, 3))
        self.assertEqual([action.text for action in h.menus[-1].actions],
                         ["Edit tissue copy in Segment Editor", "Open folder", "Remove from Results list"])
        self.assertEqual(opened, [self.subject(h).data(0, USER_ROLE + ui_run_history.ROLE_FOLDER)])

    def test_results_subgroups_a_language_switch_renames_families_and_subgroups_without_a_rebuild(self) -> None:
        """Results subgroups rule 8: the captions of families and subgroups follow the language (no rescan)."""

        new, old = self.two_runs()
        h = self.harness([new, old])
        subject = self.subject(h)
        micro = find_child(subject, "Microstructure")
        with mock.patch.object(ui_run_history, "discover_runs", side_effect=AssertionError("rescan")):
            apply_onboarding_language(h.owner, "zh")
        self.assertIs(self.subject(h), subject)
        self.assertEqual(child_texts(subject), ["解剖先验 (1)", "微结构 (2)"])
        self.assertEqual(child_texts(find_child(subject, "解剖先验")), ["组织分割 (1)"])
        self.assertEqual(child_texts(micro), ["DTI (1)", "NODDI (1)"])
        apply_onboarding_language(h.owner, "en")
        self.assertEqual(child_texts(subject), ["Anatomical priors (1)", "Microstructure (2)"])
        self.assertEqual(child_texts(find_child(subject, "Anatomical")), ["Tissue segmentation (1)"])

    def test_results_subgroups_a_rebuild_in_chinese_has_chinese_subgroup_captions(self) -> None:
        h = self.harness([self.manifest()], language="zh")
        anatomy = find_child(self.subject(h), "解剖先验")
        self.assertEqual(child_texts(anatomy), ["组织分割 (1)", "脑区分割 (1)", "皮层表面 (2)", "表面后处理 (2)"])

    def test_results_subgroups_the_fallback_tree_without_subjects_has_subgroups_too(self) -> None:
        """``_show_output_files`` without run items puts the families at the top level; subgroups start collapsed."""

        h = Harness(self)
        h.owner._runTreeItems = {}
        manifest = {"cands": [("microstructure.dti", "/x/fa.nii")], "tasks": []}
        manifest["cands"] = [{"product": product, "path": path} for product, path in manifest["cands"]]
        ui_run_history._show_output_files(h.owner, manifest)
        family = h.tree.topLevelItem(0)
        self.assertEqual(family.text(0), "Microstructure (1)")
        self.assertTrue(family.expanded)
        self.assertEqual(child_texts(family), ["DTI (1)"])
        self.assertFalse(family.child(0).expanded)
        self.assertEqual(family.child(0).child(0).data(0, ROLE_PATH), "/x/fa.nii")
        self.assertIn(1, family.child(0).child(0).icons)

    def test_results_subgroups_a_theme_change_gives_no_folder_icon_to_a_subgroup(self) -> None:
        new, old = self.two_runs()
        h = self.harness([new, old])
        noddi = find_child(find_child(self.subject(h), "Microstructure"), "NODDI")
        with mock.patch.object(ui_run_history, "_folder_icon", return_value="folder-icon"):
            ui_run_history._refresh_folder_icons(h.owner, True)
        self.assertNotIn(1, noddi.icons)
        self.assertEqual(self.subject(h).icons[1], "folder-icon")


if __name__ == "__main__":
    unittest.main()
