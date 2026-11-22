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


def exact(item, text: str):
    """The child with exactly this caption ("WMQL (1)" must not find "WMQL · UKF (1)")."""

    return next(row for row in rows(item) if row.text(0) == text)


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

    def test_results_subgroups_a_tracking_variant_suffix_gives_a_bundle_subgroup_per_source(self) -> None:
        """Results subgroups rule 10, changed on purpose by "Results sources" (batch 2): a multi-tracking run names
        tasks ``<product>@<tracking backend>`` (shared/multi_tracking.py). A bundle subgroup is now one (algorithm,
        source) pair, so the two equal WMQL file names no longer share one subgroup. Tractography keeps one subgroup
        per backend, and the connectome takes no source from the ``@`` suffix (Results sources rule 2b)."""

        h = self.harness([self.manifest(candidates=[
            ("bundles.wmql", "/x/wmql_AF.trk"), ("bundles.wmql@ukf", "/x/ukf/wmql_AF.trk"),
            ("bundles.wma@csd_act", "/x/act/wma_CST.vtp"), ("tractography.csd_act@csd_act", "/x/act.tck"),
            ("tractography.ukf@ukf", "/x/ukf.vtk"), ("connectome.sift2_weighted@csd_act", "/x/sift2.csv")])])
        subject = self.subject(h)
        bundles = find_child(subject, "Fibre bundles")
        # Review R6 (changed on purpose): "bundles.wma@csd_act" gets no source part; a WMA task of a variant may have
        # read the user's tractogram, and the manifest cannot tell.
        self.assertEqual(child_texts(bundles), ["WMQL (1)", "WMQL · UKF (1)", "WMA (1)"])
        self.assertEqual(leaf_names(exact(bundles, "WMQL (1)")), ["wmql_AF.trk"])
        self.assertEqual(exact(bundles, "WMQL · UKF (1)").child(0).data(0, ROLE_PATH), "/x/ukf/wmql_AF.trk")
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

    def test_review_codex_p1_a_family_key_with_the_separator_is_still_a_family_row(self) -> None:
        """Review (Codex P1): a manifest task named ``anatomy::surfaces.output`` gives an unknown, flat family whose
        key contains ``::`` and equals the key of the Cortical surfaces subgroup. The row kind must come from a tag on
        the row, not from the text of the key: the family keeps its own remembered expansion."""

        candidates = list(FULL_RUN) + [("anatomy::surfaces.output", "/x/odd.nii.gz")]
        records = [self.manifest(candidates=candidates)]
        h = self.harness(records)
        odd = find_child(self.subject(h), "anatomy::surfaces")
        self.assertEqual([row.data(0, ROLE_PATH) for row in rows(odd)], ["/x/odd.nii.gz"])  # flat: family > file
        self.assertEqual(ui_run_history._item_key(h.qt, odd), ("family", "sub1", "anatomy::surfaces"))
        odd.setExpanded(True)
        state = h.owner._resultsExpansion
        self.assertTrue(state["families"][("sub1", "anatomy::surfaces")])
        self.assertEqual(state["subgroups"], {})
        h.refresh(records)
        subject = self.subject(h)
        self.assertTrue(find_child(subject, "anatomy::surfaces").expanded)
        surfaces = find_child(find_child(subject, "Anatomical priors"), "Cortical surfaces")
        self.assertFalse(surfaces.expanded)
        self.assertEqual(ui_run_history._item_key(h.qt, surfaces), ("subgroup", "sub1", "anatomy", "anatomy::surfaces"))

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


# ------------------------------------------------------------------------------------------------ Results sources
# Batch 2 "Results sources": a subgroup of the bundles and connectome families is one (algorithm, tractography source)
# pair. The bundle folders are ``outputs/04_bundles/<product folder>/<source>/`` with equal file names in each source.

BUNDLES = "/d/outputs/04_bundles"


def task(name: str, files=(), **outputs_and_fields):
    """A manifest task plus the files that ``result_candidates`` gives for it (``product`` = the task name)."""

    fields = {key: outputs_and_fields.pop(key) for key in ("tracking_backend", "product") if key in outputs_and_fields}
    return dict(name=name, status="completed", outputs=dict(outputs_and_fields), **fields), list(files)


class SourceTestBase(SubgroupTestBase):
    def source_manifest(self, tasks, subject="sub1", run="run1", stamp="2026-11-01T10:00:00Z", **fields):
        record = make_manifest(self.root, subject, run, stamp,
                               [(entry["name"], path) for entry, files in tasks for path in files])
        payload = json.loads(record.manifest_path.read_text(encoding="utf-8"))
        payload["tasks"] = [entry for entry, _files in tasks]
        payload.update(fields)
        record.manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        return record

    def bundles(self, h: Harness, subject: str = "sub1"):
        return find_child(self.subject(h, subject), "Fibre bundles")


class SourceTest(SourceTestBase):
    def test_results_sources_a_multi_tracking_run_gives_one_subgroup_per_source_with_equal_file_names(self) -> None:
        """Results sources (a): ``bundles.wmql@ukf`` and ``bundles.wmql@csd_act`` publish equal file names into two
        source folders; each source is its own subgroup. Rule 3: the task's QC JSON row goes to the same subgroup."""

        qc = self.root / "wmql_ukf_qc.json"
        qc.write_text("{}", encoding="utf-8")
        h = self.harness([self.source_manifest([
            task("bundles.wmql@ukf", [f"{BUNDLES}/query_bundle_extraction/ukf/AF.trk"],
                 output_dir=f"{BUNDLES}/query_bundle_extraction/ukf", qc_path=str(qc)),
            task("bundles.wmql@csd_act", [f"{BUNDLES}/query_bundle_extraction/csd_act/AF.trk"],
                 output_dir=f"{BUNDLES}/query_bundle_extraction/csd_act")])])
        bundles = self.bundles(h)
        self.assertEqual(child_texts(bundles), ["WMQL · UKF (2)", "WMQL · CSD / ACT (1)"])
        ukf, act = rows(bundles)
        self.assertEqual([row.data(0, ROLE_PATH) for row in rows(ukf)],
                         [f"{BUNDLES}/query_bundle_extraction/ukf/AF.trk", str(qc.resolve())])
        self.assertEqual([row.data(0, ROLE_PATH) for row in rows(act)],
                         [f"{BUNDLES}/query_bundle_extraction/csd_act/AF.trk"])
        self.assertEqual([ukf.data(0, ROLE_KEY), act.data(0, ROLE_KEY)], ["bundles::wmql@ukf", "bundles::wmql@csd_act"])
        self.assertEqual(ui_run_history._item_key(h.qt, ukf), ("subgroup", "sub1", "bundles", "bundles::wmql@ukf"))

    def test_results_sources_two_single_backend_runs_give_two_subgroups(self) -> None:
        """Results sources (b): two runs of one subject, plain ``bundles.wmql`` in both, published to ``ukf`` and
        ``csd_act``: the published folder gives the source."""

        new = self.source_manifest([task("bundles.wmql", [f"{BUNDLES}/query_bundle_extraction/csd_act/AF.trk"],
                                         output_dir=f"{BUNDLES}/query_bundle_extraction/csd_act")],
                                   run="runB", stamp="2026-11-02T10:00:00Z")
        old = self.source_manifest([task("bundles.wmql", [f"{BUNDLES}/query_bundle_extraction/ukf/AF.trk"],
                                         output_dir=f"{BUNDLES}/query_bundle_extraction/ukf")], run="runA")
        h = self.harness([new, old])
        bundles = self.bundles(h)
        self.assertEqual(child_texts(bundles), ["WMQL · UKF (1)", "WMQL · CSD / ACT (1)"])
        self.assertEqual(exact(bundles, "WMQL · UKF (1)").child(0).data(0, ROLE_MANIFEST), str(old.manifest_path))
        self.assertEqual(exact(bundles, "WMQL · CSD / ACT (1)").child(0).data(0, ROLE_MANIFEST),
                         str(new.manifest_path))

    def test_results_sources_each_step_of_the_source_rule(self) -> None:
        """Results sources (c): folder before ``@``; ``supplied`` only from the folder; the ``tracking_backend``
        field; a folder that is not a source; an old flat folder without ``@`` gives no source part."""

        h = self.harness([self.source_manifest([
            # a. the folder wins over the @ suffix (a ukf variant that ran on the user's file publishes to supplied)
            task("bundles.wmql@ukf", [f"{BUNDLES}/query_bundle_extraction/supplied/AF.trk"],
                 output_dir=f"{BUNDLES}/query_bundle_extraction/supplied"),
            # a. not a source folder -> b. the @ suffix (WMQL: it reads only the variant's own tractogram)
            task("bundles.wmql@csd_act", [f"{BUNDLES}/query_bundle_extraction/wmql_out/CST.trk"],
                 output_dir=f"{BUNDLES}/query_bundle_extraction/wmql_out"),
            # Review R6: b. names no backend for WMA or TractCloud (they may have read the user's file)
            task("bundles.wma@csd_act", [f"{BUNDLES}/fiber_clustering/wma_out/CST.vtp"],
                 output_dir=f"{BUNDLES}/fiber_clustering/wma_out"),
            task("bundles.tractcloud", [f"{BUNDLES}/bundle_identification/x/AF.vtp"], tracking_backend="dti_det"),
            # c. an old flat folder and no @ suffix -> no source part
            task("bundles.wmql", [f"{BUNDLES}/query_bundle_extraction/CST.trk"],
                 output_dir=f"{BUNDLES}/query_bundle_extraction"),
            # b counts only for a tracking backend: "@supplied" and "@other" give no source
            task("bundles.wma@supplied", [f"{BUNDLES}/fiber_clustering/AF.vtp"]),
            task("bundles.tractcloud@other", [f"{BUNDLES}/bundle_identification/UF.vtp"]),
        ])])
        bundles = self.bundles(h)
        # Review R6 (changed on purpose): "WMA · CSD / ACT" and "TractCloud · DTI" are gone; those tasks join the
        # subgroup without a source. Step b keeps its WMQL cases (the @ suffix here, the tracking_backend field in
        # test_b2_results_fix_R6_...).
        self.assertEqual(child_texts(bundles), [
            "WMQL (1)", "WMQL · CSD / ACT (1)", "WMQL · supplied file (1)", "WMA (2)", "TractCloud (2)"])
        self.assertEqual(leaf_names(exact(bundles, "WMQL (1)")), ["CST.trk"])
        self.assertEqual(exact(bundles, "WMQL · CSD / ACT (1)").child(0).data(0, ROLE_PATH),
                         f"{BUNDLES}/query_bundle_extraction/wmql_out/CST.trk")
        self.assertEqual(leaf_names(exact(bundles, "WMA (2)")), ["CST.vtp", "AF.vtp"])
        self.assertEqual(leaf_names(exact(bundles, "TractCloud (2)")), ["AF.vtp", "UF.vtp"])

    def test_results_sources_connectome_from_the_source_folder_only(self) -> None:
        """Results sources (d): a per-source connectome folder gives the source; an old flat connectome file gives
        none, also for ``@ukf`` / ``@csd_act`` task names (a flat file can hold another variant's matrix)."""

        connectome = "/d/outputs/05_connectome"
        h = self.harness([self.source_manifest([
            task("connectome.streamline_count@ukf",
                 [f"{connectome}/ukf/s_streamline_count.csv", f"{connectome}/ukf/s_nodes.tsv"],
                 matrix_path=f"{connectome}/ukf/s_streamline_count.csv", nodes_path=f"{connectome}/ukf/s_nodes.tsv"),
            task("connectome.streamline_count", [f"{connectome}/csd_act/s_streamline_count.csv"],
                 matrix=f"{connectome}/csd_act/s_streamline_count.csv"),
        ])])
        found = find_child(self.subject(h), "Connectome")
        self.assertEqual(child_texts(found), ["Fiber count · UKF (2)", "Fiber count · CSD / ACT (1)"])
        flat = self.harness([self.source_manifest([
            task("connectome.streamline_count@ukf", [f"{connectome}/s_streamline_count.csv"],
                 matrix_path=f"{connectome}/s_streamline_count.csv"),
            task("connectome.streamline_count@csd_act", [f"{connectome}/s_streamline_count.csv"],
                 matrix_path=f"{connectome}/s_streamline_count.csv"),
        ], subject="sub2")])
        self.assertEqual(child_texts(find_child(self.subject(flat, "sub2"), "Connectome")), ["Fiber count (1)"])

    def test_results_sources_a_single_and_a_multi_run_share_one_subgroup(self) -> None:
        """Results sources (e): ``bundles.wmql`` published to ``ukf`` and ``bundles.wmql@ukf`` are one subgroup;
        each file shows once."""

        folder = f"{BUNDLES}/query_bundle_extraction/ukf"
        new = self.source_manifest([task("bundles.wmql@ukf", [f"{folder}/AF.trk", f"{folder}/UF.trk"],
                                         output_dir=folder)], run="multi", stamp="2026-11-02T10:00:00Z")
        old = self.source_manifest([task("bundles.wmql", [f"{folder}/AF.trk", f"{folder}/CST.trk"],
                                         output_dir=folder)], run="single")
        h = self.harness([new, old])
        bundles = self.bundles(h)
        self.assertEqual(child_texts(bundles), ["WMQL · UKF (3)"])
        self.assertEqual(leaf_names(bundles.child(0)), ["AF.trk", "UF.trk", "CST.trk"])
        self.assertEqual(bundles.text(0), "Fibre bundles (3)")

    def test_results_sources_order_algorithm_then_source_also_after_an_older_run_adds_one(self) -> None:
        """Results sources (f): algorithm in table order, then unknown, ukf, csd_act, (saf, review R7), dti_det,
        supplied; an unnamed algorithm after the named ones."""

        def wmql(source):
            return task(f"bundles.wmql@{source}", [f"{BUNDLES}/query_bundle_extraction/{source}/AF.trk"])

        h = self.harness([self.source_manifest([
            # (review R6: step b names no backend outside WMQL, so these two carry their published folder)
            task("bundles.zz@ukf", [f"{BUNDLES}/zz/AF.vtp"], output_dir=f"{BUNDLES}/zz/ukf"),
            task("bundles.tractcloud@ukf", [f"{BUNDLES}/bundle_identification/ukf/AF.vtp"],
                 output_dir=f"{BUNDLES}/bundle_identification/ukf"),
            task("bundles.wmql", [f"{BUNDLES}/query_bundle_extraction/AF.trk"]),
            task("bundles.wmql@supplied", [f"{BUNDLES}/query_bundle_extraction/supplied/AF.trk"],
                 output_dir=f"{BUNDLES}/query_bundle_extraction/supplied"),
            wmql("dti_det"), wmql("csd_act"), wmql("ukf"),
        ])])
        self.assertEqual(child_texts(self.bundles(h)), [
            "WMQL (1)", "WMQL · UKF (1)", "WMQL · CSD / ACT (1)", "WMQL · DTI (1)", "WMQL · supplied file (1)",
            "TractCloud · UKF (1)", "zz · UKF (1)"])
        # An older run adds subgroups that sort before the newer run's one.
        new = self.source_manifest([wmql("csd_act")], subject="sub2", run="runB", stamp="2026-11-02T10:00:00Z")
        old = self.source_manifest([wmql("ukf"), task("bundles.tractcloud", [f"{BUNDLES}/bundle_identification/x.vtp"]),
                                    task("bundles.wmql", [f"{BUNDLES}/query_bundle_extraction/CST.trk"])],
                                   subject="sub2", run="runA")
        h = self.harness([new, old])
        self.assertEqual(child_texts(self.bundles(h, "sub2")), [
            "WMQL (1)", "WMQL · UKF (1)", "WMQL · CSD / ACT (1)", "TractCloud (1)"])


class SourceRowTest(SourceTestBase):
    def two_sources(self):
        return [self.source_manifest([
            task("bundles.wmql@ukf", [f"{BUNDLES}/query_bundle_extraction/ukf/AF.trk"]),
            task("bundles.wmql@csd_act", [f"{BUNDLES}/query_bundle_extraction/csd_act/AF.trk"]),
            task("bundles.wma", [f"{BUNDLES}/fiber_clustering/supplied/AF.vtp"],
                 output_dir=f"{BUNDLES}/fiber_clustering/supplied"),
            task("connectome.streamline_count", ["/d/outputs/05_connectome/ukf/s_streamline_count.csv"],
                 matrix_path="/d/outputs/05_connectome/ukf/s_streamline_count.csv"),
        ])]

    def test_results_sources_a_language_switch_relabels_the_source_captions_in_place(self) -> None:
        """Results sources (g): ``refresh_results_language`` renames the captions with a source part, no rebuild."""

        h = self.harness(self.two_sources())
        subject = self.subject(h)
        bundles = self.bundles(h)
        with mock.patch.object(ui_run_history, "discover_runs", side_effect=AssertionError("rescan")):
            apply_onboarding_language(h.owner, "zh")
        self.assertIs(self.subject(h), subject)
        self.assertEqual(child_texts(bundles), ["WMQL · UKF (1)", "WMQL · CSD / ACT (1)", "WMA · 自备文件 (1)"])
        self.assertEqual(child_texts(find_child(subject, "连接组")), ["Fiber count · UKF (1)"])
        apply_onboarding_language(h.owner, "en")
        self.assertEqual(child_texts(bundles), ["WMQL · UKF (1)", "WMQL · CSD / ACT (1)", "WMA · supplied file (1)"])
        zh = self.harness(self.two_sources(), language="zh")
        self.assertEqual(child_texts(find_child(self.subject(zh), "纤维束")),
                         ["WMQL · UKF (1)", "WMQL · CSD / ACT (1)", "WMA · 自备文件 (1)"])

    def test_results_sources_expansion_is_remembered_per_source(self) -> None:
        """Results sources (h): the expansion memory keeps "WMQL · UKF" and "WMQL · CSD / ACT" apart."""

        records = self.two_sources()
        h = self.harness(records)
        exact(self.bundles(h), "WMQL · UKF (1)").setExpanded(True)
        h.refresh(records)
        bundles = self.bundles(h)
        self.assertTrue(exact(bundles, "WMQL · UKF (1)").expanded)
        self.assertFalse(exact(bundles, "WMQL · CSD / ACT (1)").expanded)
        state = h.owner._resultsExpansion["subgroups"]
        self.assertTrue(state[("sub1", "bundles", "bundles::wmql@ukf")])
        self.assertNotIn(("sub1", "bundles", "bundles::wmql@csd_act"), state)
        self.assertNotIn(("sub1", "bundles", "bundles::wmql"), state)
        # The current item inside a source subgroup survives a rebuild too.
        h.tree.setCurrentItem(exact(bundles, "WMQL · CSD / ACT (1)").child(0))
        h.refresh(records)
        self.assertEqual(h.tree.currentItem().data(0, ROLE_PATH), f"{BUNDLES}/query_bundle_extraction/csd_act/AF.trk")


# ------------------------------------------------------------------------------------------ review fix round (R6-R8)

class FixSourceTest(SourceTestBase):
    def test_b2_results_fix_R6_step_b_names_no_backend_for_a_bundle_task_that_may_have_read_a_supplied_file(self):
        """Review R6: an old multi-tracking run (flat bundle folders) ran WMA / TractCloud on a user-supplied
        tractogram inside a variant; the parent manifest records the variant (``@ukf``, ``tracking_backend``) but no
        input file, so it cannot tell: no source part. WMQL has no supplied-file input (the orchestrator config has
        ``wma_input_tractography`` and ``tractcloud_input_tractography`` only), so its variant is its source."""

        h = self.harness([self.source_manifest([
            task("bundles.tractcloud@ukf", [f"{BUNDLES}/bundle_identification/AF.vtp"],
                 output_dir=f"{BUNDLES}/bundle_identification", product="bundles.tractcloud",
                 tracking_backend="ukf"),
            task("bundles.wma@csd_act", [f"{BUNDLES}/fiber_clustering/CST.vtp"],
                 output_dir=f"{BUNDLES}/fiber_clustering", product="bundles.wma", tracking_backend="csd_act"),
            task("bundles.wmql@ukf", [f"{BUNDLES}/query_bundle_extraction/AF.trk"],
                 output_dir=f"{BUNDLES}/query_bundle_extraction", product="bundles.wmql", tracking_backend="ukf"),
        ])])
        self.assertEqual(child_texts(self.bundles(h)), ["WMQL · UKF (1)", "WMA (1)", "TractCloud (1)"])
        # The tracking_backend field (no @ suffix) still names the source of a WMQL task.
        field = self.harness([self.source_manifest([
            task("bundles.wmql", [f"{BUNDLES}/query_bundle_extraction/AF.trk"],
                 output_dir=f"{BUNDLES}/query_bundle_extraction", tracking_backend="dti_det")], subject="sub2")])
        self.assertEqual(child_texts(self.bundles(field, "sub2")), ["WMQL · DTI (1)"])

    def test_b2_results_fix_R6_a_run_that_records_a_bundle_source_shows_no_supplied_file(self) -> None:
        """Review R6, the other case: a multi-tracking parent with ``bundle_tractography_backend`` shows that no
        bundle product read a user file (orchestrator._validate_bundle_tractography_source refuses that setting with a
        WMA or TractCloud input file), so step b names the variant of WMA and TractCloud too. Guard: it passes on
        4525ef5b and keeps R6 from hiding a source that the record does show."""

        h = self.harness([self.source_manifest([
            task("bundles.tractcloud@ukf", [f"{BUNDLES}/bundle_identification/AF.vtp"],
                 output_dir=f"{BUNDLES}/bundle_identification", product="bundles.tractcloud",
                 tracking_backend="ukf"),
            task("bundles.wma@csd_act", [f"{BUNDLES}/fiber_clustering/CST.vtp"],
                 output_dir=f"{BUNDLES}/fiber_clustering", product="bundles.wma", tracking_backend="csd_act"),
        ], bundle_tractography_backend="csd_act")])
        self.assertEqual(child_texts(self.bundles(h)), ["WMA · CSD / ACT (1)", "TractCloud · UKF (1)"])

    def test_b2_results_fix_R6_the_published_supplied_folder_still_wins(self) -> None:
        """Step a is unchanged: a TractCloud task of a ukf variant that published to ``supplied`` is "supplied"."""

        h = self.harness([self.source_manifest([
            task("bundles.tractcloud@ukf", [f"{BUNDLES}/bundle_identification/supplied/AF.vtp"],
                 output_dir=f"{BUNDLES}/bundle_identification/supplied", tracking_backend="ukf")])])
        self.assertEqual(child_texts(self.bundles(h)), ["TractCloud · supplied file (1)"])

    def test_b2_results_fix_R7_saf_is_a_source_after_csd_act(self) -> None:
        """Review R7: a SAF child publishes to ``<product>/saf/`` and ``05_connectome/saf/``; caption "CSD + SAF"
        (the GUI's tractography caption, both languages), order after ``csd_act``."""

        connectome = "/d/outputs/05_connectome"

        def wmql(source):
            return task(f"bundles.wmql@{source}", [f"{BUNDLES}/query_bundle_extraction/{source}/AF.trk"],
                        output_dir=f"{BUNDLES}/query_bundle_extraction/{source}")

        records = [self.source_manifest([
            wmql("dti_det"), wmql("saf"), wmql("csd_act"),
            task("connectome.streamline_count", [f"{connectome}/saf/s_streamline_count.csv"],
                 matrix_path=f"{connectome}/saf/s_streamline_count.csv"),
            task("connectome.streamline_count@csd_act", [f"{connectome}/csd_act/s_streamline_count.csv"],
                 matrix_path=f"{connectome}/csd_act/s_streamline_count.csv"),
        ])]
        h = self.harness(records)
        self.assertEqual(child_texts(self.bundles(h)),
                         ["WMQL · CSD / ACT (1)", "WMQL · CSD + SAF (1)", "WMQL · DTI (1)"])
        self.assertEqual(exact(self.bundles(h), "WMQL · CSD + SAF (1)").data(0, ROLE_KEY), "bundles::wmql@saf")
        self.assertEqual(child_texts(find_child(self.subject(h), "Connectome")),
                         ["Fiber count · CSD / ACT (1)", "Fiber count · CSD + SAF (1)"])
        zh = self.harness(records, language="zh")
        self.assertEqual(child_texts(find_child(self.subject(zh), "纤维束")),
                         ["WMQL · CSD / ACT (1)", "WMQL · CSD + SAF (1)", "WMQL · DTI (1)"])

    def test_b2_results_fix_R8_a_run_named_folder_outside_the_published_layout_is_not_a_source(self) -> None:
        """Review R8: legacy and stage folders are named after the RUN (the CLI default run name is ``csd_act``);
        only ``04_bundles/<product>/<source>`` and ``05_connectome/<source>/<file>`` outside the stage cache count."""

        legacy = "/d/outputs/04_tractography"
        stages = "/d/cache/stages"
        h = self.harness([self.source_manifest([
            task("bundles.wmql", [f"{legacy}/bundles/wmql/csd_act/AF.trk"],
                 output_dir=f"{legacy}/bundles/wmql/csd_act"),
            # (a file under the stage cache is never listed, so the listed file lies elsewhere)
            task("bundles.tractcloud", [f"{BUNDLES}/bundle_identification/AF.vtp"],
                 output_dir=f"{stages}/04_bundles/tract_cloud_dl/ukf"),
            task("connectome.streamline_count", [f"{legacy}/connectome/csd_act/s_streamline_count.csv"],
                 matrix_path=f"{legacy}/connectome/csd_act/s_streamline_count.csv"),
            task("connectome.sift2_weighted", ["/d/outputs/05_connectome/s_sift2.csv"],
                 matrix=f"{stages}/05_connectome/ukf/s_sift2.csv"),
        ])])
        self.assertEqual(child_texts(self.bundles(h)), ["WMQL (1)", "TractCloud (1)"])
        self.assertEqual(child_texts(find_child(self.subject(h), "Connectome")), ["Fiber count (1)", "SIFT2 (1)"])

    def test_b2_results_fix_R8_the_folder_names_are_the_layouts_names(self) -> None:
        from shared.io_layout import subject_layout

        layout = subject_layout(self.root / "s", output_root=self.root / "s" / "outputs")
        self.assertEqual(ui_run_history.RESULT_BUNDLES_FOLDER, layout.bundles.name)
        self.assertEqual(ui_run_history.RESULT_CONNECTOME_FOLDER, layout.connectome.name)
        self.assertEqual(ui_run_history.RESULT_STAGE_CACHE_PARTS, layout.cache_stages.parts[-2:])


if __name__ == "__main__":
    unittest.main()
