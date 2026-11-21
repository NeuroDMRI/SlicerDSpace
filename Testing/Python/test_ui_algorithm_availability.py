"""Package B, GUI side: gate inputs (B1), picker popup "?" markers (B2), summary and Run request (B3).

The data rules and their agreement with the planner are tested in ``test_algorithm_availability.py``. Qt is faked.
"""

from __future__ import annotations

import contextlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import nibabel as nib

from shared import algorithm_availability as availability
from shared import ui_algorithm_picker as picker
from shared.topup_parameters import TopupParameters
from shared.ui_run_controller import ProductTargetFlags, filter_run_for_data
from Testing.Python.test_ui_run_history import fake_qt, object_name

SINGLE_SHELL = "0 0 1000 1000 1000 1000"
THREE_GROUPS = "0 0 1000 1000 2000 2000"


def write_subject(root: Path, bvals: str, *, with_dwi: bool = True) -> Path:
    """A canonical subject root: ``input/{data.nii.gz,bvals,bvecs}``."""
    folder = root / "input"
    folder.mkdir(parents=True, exist_ok=True)
    count = len(bvals.split())
    (folder / "bvals").write_text(bvals + "\n")
    rows = np.zeros((3, count))
    rows[0, [i for i, b in enumerate(bvals.split()) if float(b) > 0]] = 1.0
    (folder / "bvecs").write_text("\n".join(" ".join(f"{v:g}" for v in row) for row in rows) + "\n")
    if with_dwi:
        nib.save(nib.Nifti1Image(np.zeros((3, 3, 3, count), np.float32), np.eye(4)), str(folder / "data.nii.gz"))
    return root


def write_gradients(dwi: Path, bval: Path, bvec: Path, bvals: str) -> None:
    """A DWI with its gradient files at explicit paths (the preprocessing ``*_dwi_final.*`` files)."""
    dwi.parent.mkdir(parents=True, exist_ok=True)
    count = len(bvals.split())
    bval.write_text(bvals + "\n")
    rows = np.zeros((3, count))
    rows[0, [i for i, b in enumerate(bvals.split()) if float(b) > 0]] = 1.0
    bvec.write_text("\n".join(" ".join(f"{v:g}" for v in row) for row in rows) + "\n")
    nib.save(nib.Nifti1Image(np.zeros((3, 3, 3, count), np.float32), np.eye(4)), str(dwi))


def write_final(subject: Path, bvals: str, output_root=None, subject_id="sub") -> dict:
    """F1: write the preprocessing outputs where ``shared.orchestrator._preprocessing_outputs`` looks for them."""
    from shared.orchestrator import _preprocessing_outputs

    outputs = _preprocessing_outputs(subject.resolve(), subject_id, output_root)
    write_gradients(outputs["dwi"], outputs["bval"], outputs["bvec"], bvals)
    return outputs


def make_node(root=None, stages="dti,noddi", **extra):
    values = dict(
        inputDirectory=str(root) if root else "", subjectId="sub", batchMode=False,
        experimentsEnabled=False, experimentScope="subject",
        targetPreprocessing=False, targetSurface=False, targetSurfacePostprocess=False, targetParcellation=False,
        targetSegmentation=False, targetMicrostructure=True, targetTractography=False, targetBundles=False,
        targetConnectome=False,
        orchestrationPreprocessingStages="denoise,eddy,topup", orchestrationMicrostructureStages=stages,
        orchestrationTractographyBackend="csd_act", orchestrationBundleBackend="wmql",
        orchestrationConnectomeMetrics="sift2", orchestrationRunFreeWater=False, orchestrationRunNoddi=False,
        orchestrationRunDki=False, orchestrationRunIfw=False, ukfParameters="{}", wmaInputTractography="",
    )
    values.update(extra)
    return SimpleNamespace(**values)


def make_owner(root=None, *, pending=None, tiles=None, batch=False, stages="dti,noddi", lang="en", **node_extra):
    tiles = tiles or {}
    return SimpleNamespace(
        ui=SimpleNamespace(), _uiWidget=None, _pipelineActive=False, _selectedSetupActive=False,
        _onboardingLanguage=lang, _lastEnvironmentResult=None, _lastEnvironmentSignature=None,
        _parameterNode=make_node(root, stages, batchMode=batch, **node_extra),
        _input_preview_pending_source=lambda: pending,
        _input_preview_is_batch=lambda: batch,
        _prepare_inputs_widget_text=lambda key: tiles.get(key, ""),
        _effective_subject_id=lambda: "sub",
    )


class TempCase(unittest.TestCase):
    def setUp(self):
        availability.clear_cache()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)


class GateInputsTest(TempCase):
    def test_b1_the_canonical_subject_root_gives_the_planner_paths(self):
        from shared.orchestrator import _resolve_raw_inputs

        subject = write_subject(self.root / "s1", SINGLE_SHELL)
        inputs = availability.current_gate_inputs(make_owner(subject))
        expected = _resolve_raw_inputs(subject.resolve())  # F1: the planner resolves the subject root
        for key in ("dwi", "bval", "bvec"):
            self.assertEqual(Path(inputs[key]), expected[key], key)

    def test_b1_a_nifti_tile_source_uses_the_bval_and_bvec_tiles(self):
        tiles = {"bval": str(self.root / "my.bvals"), "bvec": str(self.root / "my.bvecs")}
        pending = {"kind": "nifti", "source": str(self.root / "scan.nii.gz"), "bval": tiles["bval"]}
        inputs = availability.current_gate_inputs(make_owner(pending=pending, tiles=tiles))
        self.assertEqual(Path(inputs["dwi"]), self.root / "scan.nii.gz")
        self.assertEqual(Path(inputs["bval"]), self.root / "my.bvals")
        self.assertEqual(Path(inputs["bvec"]), self.root / "my.bvecs")

    def test_b1_a_nifti_tile_source_without_sidecar_tiles_uses_the_files_next_to_it(self):
        pending = {"kind": "nifti", "source": str(self.root / "scan.nii.gz"), "bval": ""}
        inputs = availability.current_gate_inputs(make_owner(pending=pending))
        self.assertEqual(Path(inputs["bval"]), self.root / "scan.bval")
        self.assertEqual(Path(inputs["bvec"]), self.root / "scan.bvec")
        pending["source"] = str(self.root / "scan.nii")
        inputs = availability.current_gate_inputs(make_owner(pending=pending))
        self.assertEqual(Path(inputs["bval"]), self.root / "scan.bval")

    def test_b1_the_tile_source_wins_over_a_folder_that_holds_other_data(self):
        subject = write_subject(self.root / "s1", SINGLE_SHELL)
        pending = {"kind": "nifti", "source": str(self.root / "scan.nii.gz"), "bval": ""}
        inputs = availability.current_gate_inputs(make_owner(subject, pending=pending))
        self.assertEqual(Path(inputs["bval"]), self.root / "scan.bval")

    def test_b1_nrrd_and_dicom_tile_sources_are_unknown(self):
        for kind in ("nrrd", "dicom"):
            pending = {"kind": kind, "source": str(self.root / "x"), "bval": ""}
            self.assertIsNone(availability.current_gate_inputs(make_owner(pending=pending)), kind)
            self.assertEqual(availability.current_data_blocks(make_owner(pending=pending)), {}, kind)

    def test_b1_no_input_and_batch_mode_are_unknown(self):
        self.assertIsNone(availability.current_gate_inputs(make_owner()))
        subject = write_subject(self.root / "s1", SINGLE_SHELL)
        self.assertIsNone(availability.current_gate_inputs(make_owner(subject, batch=True)))
        self.assertEqual(availability.current_data_blocks(make_owner(subject, batch=True)), {})

    def test_b1_owners_without_a_usable_node_never_raise(self):
        for owner in (None, SimpleNamespace(), SimpleNamespace(_parameterNode=None), mock.MagicMock(),
                      SimpleNamespace(_parameterNode=SimpleNamespace())):
            self.assertIsNone(availability.current_gate_inputs(owner), owner)
            self.assertEqual(availability.current_data_blocks(owner), {}, owner)

    def test_b1_missing_bvals_are_unknown(self):
        subject = write_subject(self.root / "s1", SINGLE_SHELL)
        (subject / "input" / "bvals").unlink()
        self.assertEqual(availability.current_data_blocks(make_owner(subject)), {})

    def test_b1_single_shell_data_blocks_noddi_and_free_water_but_not_dti(self):
        subject = write_subject(self.root / "s1", SINGLE_SHELL)
        blocks = availability.current_data_blocks(make_owner(subject))
        self.assertLessEqual({"noddi", "freewater", "ifw"}, set(blocks))
        self.assertNotIn("dti", blocks)
        self.assertEqual(availability.current_data_blocks(make_owner(write_subject(self.root / "s2", THREE_GROUPS))
                                                          ).keys() & {"noddi", "freewater", "ifw"}, set())


class GateFollowsPlannerTest(TempCase):
    """F1: the gate reads the files of ``_resolve_analysis_inputs`` (final files once preprocessing is complete)."""

    def test_f1_complete_preprocessing_makes_the_gate_read_the_final_files_and_noddi_is_not_blocked(self):
        subject = write_subject(self.root / "s1", SINGLE_SHELL)  # raw: one shell
        final = write_final(subject, THREE_GROUPS)  # final: two shells
        with mock.patch("shared.orchestrator._preprocessing_complete", return_value=True):
            owner = make_owner(subject)
            inputs = availability.current_gate_inputs(owner)
            blocks = availability.current_data_blocks(owner)
        self.assertEqual(Path(inputs["bval"]), final["bval"])
        self.assertEqual(Path(inputs["bvec"]), final["bvec"])
        self.assertEqual(Path(inputs["dwi"]), final["dwi"])
        self.assertNotIn("noddi", blocks)

    def test_f1_the_reverse_final_single_shell_blocks_noddi_although_the_raw_files_fit(self):
        subject = write_subject(self.root / "s1", THREE_GROUPS)  # raw: two shells
        final = write_final(subject, SINGLE_SHELL)  # final: one shell
        with mock.patch("shared.orchestrator._preprocessing_complete", return_value=True):
            owner = make_owner(subject)
            self.assertEqual(Path(availability.current_gate_inputs(owner)["bval"]), final["bval"])
            self.assertIn("noddi", availability.current_data_blocks(owner))

    def test_f1_incomplete_preprocessing_keeps_the_raw_files(self):
        subject = write_subject(self.root / "s1", SINGLE_SHELL)
        write_final(subject, THREE_GROUPS)
        with mock.patch("shared.orchestrator._preprocessing_complete", return_value=False):
            owner = make_owner(subject)
            self.assertEqual(Path(availability.current_gate_inputs(owner)["bval"]), subject.resolve() / "input" / "bvals")
            self.assertIn("noddi", availability.current_data_blocks(owner))


class PendingImportTest(TempCase):
    """F2: a source that the run would (re)import makes the data unknown: no block, no skip."""

    def convertible_root_with_stale_receipt(self) -> Path:
        root = self.root / "conv"
        write_gradients(root / "data.nii.gz", root / "bvals", root / "bvecs", THREE_GROUPS)  # the new source
        write_subject(root, SINGLE_SHELL)  # the old import in input/
        from shared.convertible_input import RECEIPT

        (root / "input" / RECEIPT).write_text('{"kind": "nifti", "signature": {"source": "old"}}\n')
        return root

    def test_f2_a_convertible_source_with_a_stale_receipt_gives_no_block(self):
        owner = make_owner(self.convertible_root_with_stale_receipt())
        self.assertIsNone(availability.current_gate_inputs(owner))
        self.assertEqual(availability.current_data_blocks(owner), {})
        self.assertEqual(availability.current_data_blocks(owner, fresh=True), {})

    def test_f2_a_dicom_receipt_in_a_separate_workspace_gives_no_block(self):
        from shared.dicom_input import RECEIPT
        from shared.io_layout import subject_layout
        from shared.ui_run_controller import resolve_single_subject_output_root

        subject = write_subject(self.root / "s1", SINGLE_SHELL)
        output_text = str(self.root / "out")
        output_root = resolve_single_subject_output_root(output_text, "sub").resolve()
        workspace = subject_layout(subject.resolve(), output_root=output_root).workspace_root
        (workspace / "input").mkdir(parents=True)
        (workspace / "input" / RECEIPT).write_text("{}\n")  # the run would switch to this converted input
        owner = make_owner(subject, outputDirectory=output_text)
        self.assertIsNone(availability.current_gate_inputs(owner))
        self.assertEqual(availability.current_data_blocks(owner), {})


class GateCostTest(TempCase):
    """F3: the per-event path lists no folder for a non-canonical input and resolves once per cache period."""

    def test_f3_an_unconverted_dicom_folder_is_unknown_without_resolving_or_listing(self):
        folder = self.root / "dicom"
        folder.mkdir()
        owner = make_owner(folder)
        inspected = {"mode": "dicom", "subject_root": folder, "subject_dir": folder}
        with mock.patch("shared.pipeline.inspect_subject_directory", return_value=inspected), \
                mock.patch("shared.pipeline.resolve_subject_root_dir") as resolve, \
                mock.patch("shared.convertible_input.find_convertible_input") as find:
            for _ in range(3):
                self.assertIsNone(availability.current_gate_inputs(owner))
                self.assertEqual(availability.current_data_blocks(owner), {})
        resolve.assert_not_called()
        find.assert_not_called()

    def test_f3_a_burst_of_refreshes_within_the_cache_period_resolves_once(self):
        import shared.orchestrator as orchestrator
        import shared.pipeline as pipeline

        subject = write_subject(self.root / "s1", SINGLE_SHELL)
        owner = make_owner(subject)
        clock = [100.0]
        with mock.patch("shared.algorithm_availability._monotonic", lambda: clock[0], create=True), \
                mock.patch("shared.pipeline.resolve_subject_root_dir", wraps=pipeline.resolve_subject_root_dir) as resolve, \
                mock.patch("shared.orchestrator._preprocessing_complete",
                           wraps=orchestrator._preprocessing_complete) as complete:
            for _ in range(5):
                self.assertIn("noddi", availability.current_data_blocks(owner))
            self.assertEqual(resolve.call_count, 1)
            self.assertEqual(complete.call_count, 1)
            clock[0] += 1.5  # the cache period is over
            availability.current_data_blocks(owner)
            self.assertEqual(resolve.call_count, 2)
            self.assertEqual(complete.call_count, 2)
            availability.current_data_blocks(owner, fresh=True)  # the Run filter never uses the cached value
            self.assertEqual(resolve.call_count, 3)

    def test_f3_a_new_input_folder_is_resolved_at_once(self):
        first = write_subject(self.root / "s1", SINGLE_SHELL)
        second = write_subject(self.root / "s2", THREE_GROUPS)
        owner = make_owner(first)
        with mock.patch("shared.algorithm_availability._monotonic", lambda: 100.0, create=True):
            self.assertIn("noddi", availability.current_data_blocks(owner))
            owner._parameterNode.inputDirectory = str(second)
            self.assertNotIn("noddi", availability.current_data_blocks(owner))


class FormatTabTest(TempCase):
    """F4: switching the Prepare Inputs format tab refreshes the family summary."""

    def test_f4_a_format_tab_switch_refreshes_the_summary_and_its_tooltip(self):
        from shared import ui_prepare_inputs as prep

        scan = self.root / "scan.nii.gz"
        write_gradients(scan, self.root / "scan.bval", self.root / "scan.bvec", SINGLE_SHELL)
        fields = {"format_pages": mock.MagicMock(), "format_tabs": mock.MagicMock(), "drop_hint": mock.MagicMock()}
        for index, kind in enumerate(("nifti", "nrrd", "dicom")):
            fields[kind + "_page"] = index
            fields[kind + "_source"] = mock.MagicMock()
        names = ("targetPreprocessingCheckBox", "targetMicrostructureCheckBox", "targetTractographyCheckBox",
                 "targetBundlesCheckBox", "targetConnectomeCheckBox")
        owner = make_owner(stages="dti,noddi")
        owner.ui = SimpleNamespace(**{name: FakeText() for name in names})
        owner._algorithmSummaryLabels = {name: FakeText() for name in names}
        owner._algorithmButtons = {}
        owner._prepareInputWidgets = fields
        owner._input_preview_pending_source = lambda: {"kind": fields["source_kind"], "source": str(scan), "bval": ""}
        summary = owner._algorithmSummaryLabels["targetMicrostructureCheckBox"]
        with mock.patch("shared.algorithm_availability._monotonic", lambda: 100.0, create=True):
            prep.show_input_format(owner, "nifti")  # panel construction
            from shared.ui_compact import refresh_compact_algorithms

            refresh_compact_algorithms(owner)
            self.assertEqual(summary.text, "DTI")
            self.assertIn("NODDI: ", summary.toolTip)
            prep.show_input_format(owner, "dicom")  # the user switches the tab: a DICOM source is unknown
            self.assertEqual(summary.text, "DTI + NODDI")
            self.assertNotIn("NODDI: ", summary.toolTip)
            prep.show_input_format(owner, "nifti")
            self.assertEqual(summary.text, "DTI")

    def test_f4_panel_construction_and_an_unchanged_kind_do_not_refresh(self):
        from shared import ui_prepare_inputs as prep

        fields = {"format_pages": mock.MagicMock(), "format_tabs": mock.MagicMock(), "drop_hint": mock.MagicMock(),
                  "nifti_page": 0, "nifti_source": mock.MagicMock()}
        owner = make_owner()
        owner._prepareInputWidgets = fields
        with mock.patch("shared.ui_compact.refresh_compact_algorithms") as refresh:
            prep.show_input_format(owner, "nifti")
            prep.show_input_format(owner, "nifti")
        refresh.assert_not_called()


class PopupBase(TempCase):
    """Run ``choose_algorithms`` with fake Qt. ``chosen`` is the set of enabled boxes the user ticks."""

    def popup(self, owner, family="microstructure", *, chosen=None, blocks=None, fsl_offered=True, ui=None):
        qt = fake_qt()
        if ui is not None:
            owner.ui = ui
            qt.QComboBox = type("FakeComboBox", (), {})
            qt.QDoubleSpinBox = type("FakeDoubleSpinBox", (), {})
        dialog_factory = qt.QDialog.side_effect

        def make_dialog(*args, **kwargs):
            dialog = dialog_factory(*args, **kwargs)

            def fake_exec():
                if chosen is None:
                    return 0
                for widget in qt.created:
                    if widget.kind == "QCheckBox" and str(widget.objectName).startswith("algorithm_"):
                        disabled = any(call.args == (False,) for call in widget.setEnabled.call_args_list)
                        if not disabled:
                            widget.checked = widget.objectName.removeprefix("algorithm_") in chosen
                return qt.QDialog.Accepted

            dialog.exec_.side_effect = fake_exec
            return dialog

        qt.QDialog.side_effect = make_dialog
        slicer = mock.MagicMock(name="slicer")
        patches = [
            mock.patch.dict(sys.modules, {"qt": qt, "slicer": slicer, "ctk": mock.MagicMock()}),
            mock.patch("shared.selected_setup.fsl_preprocessing_offered", return_value=fsl_offered),
            mock.patch("shared.ui_compact.refresh_compact_algorithms"),
            mock.patch("shared.ui_on_demand_resources.maybe_prompt_on_demand_install"),
        ]
        if blocks is not None:
            patches.append(mock.patch("shared.algorithm_availability.current_data_blocks", return_value=blocks))
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            accepted = picker.choose_algorithms(owner, family)
        self.qt = qt
        self.boxes = {w.objectName: w for w in qt.created if w.kind == "QCheckBox"}
        return accepted

    def markers(self):
        return {object_name(w): w for w in self.qt.created if object_name(w).startswith("algorithmUnavailable_")}


class PopupTest(PopupBase):
    def single_shell_owner(self, stages="dti,noddi"):
        subject = write_subject(self.root / "s1", SINGLE_SHELL)
        return make_owner(subject, stages=stages)

    def test_b2_a_data_blocked_row_is_disabled_unchecked_and_carries_a_question_mark_with_the_reason(self):
        owner = self.single_shell_owner()
        self.popup(owner)
        box = self.boxes["algorithm_noddi"]
        reason = availability.current_data_blocks(owner)["noddi"].text("en")
        self.assertIn("at least 2 non-zero b-shells", reason)
        self.assertFalse(box.checked)
        box.setEnabled.assert_called_with(False)
        box.setToolTip.assert_called_with(reason)
        marker = self.markers()["algorithmUnavailable_noddi"]
        self.assertIn(marker.kind, {"QToolButton", "QLabel"})
        marker.setToolTip.assert_called_with(reason)
        marker.setFixedSize.assert_called_with(18, 18)
        marker.setFocusPolicy.assert_called_with(self.qt.Qt.NoFocus)

    def test_b2_dti_stays_enabled_and_has_no_marker(self):
        self.popup(self.single_shell_owner())
        self.assertTrue(self.boxes["algorithm_dti"].checked)
        self.boxes["algorithm_dti"].setEnabled.assert_not_called()
        self.assertNotIn("algorithmUnavailable_dti", self.markers())

    def test_b2_the_marker_is_the_last_widget_of_its_row_after_the_box(self):
        self.popup(self.single_shell_owner())
        marker = self.markers()["algorithmUnavailable_noddi"]
        box = self.boxes["algorithm_noddi"]
        rows = [w for w in self.qt.created if w.kind == "QHBoxLayout"
                and any(c[0] == "addWidget" and c[1][0] is marker for c in w.method_calls)]
        self.assertEqual(len(rows), 1)
        added = [c[1][0] for c in rows[0].method_calls if c[0] == "addWidget"]
        self.assertIs(added[0], box)
        self.assertIs(added[-1], marker)

    def test_b2_a_blocked_row_with_a_settings_gear_disables_the_gear_and_keeps_the_marker_last(self):
        dki_ui = SimpleNamespace(
            orchestrationDkiFitMethodComboBox=SimpleNamespace(toolTip="fit tip"),
            orchestrationDkiFitMethodLabel=SimpleNamespace(text="Fit method"))
        owner = make_owner(stages="dti,dki")
        owner._parameterNode.orchestrationDkiFitMethod = "MKCURVE"
        block = availability.Block("dki", "dki_acquisition", {"detail": "too few volumes"})
        self.popup(owner, blocks={"dki": block}, ui=dki_ui)
        reason = block.text("en")
        gear = next(w for w in self.qt.created if object_name(w) == "algorithmSettings_dki")
        gear.setEnabled.assert_called_with(False)
        gear.setToolTip.assert_called_with(reason)
        marker = self.markers()["algorithmUnavailable_dki"]
        row = next(w for w in self.qt.created if w.kind == "QHBoxLayout"
                   and any(c[0] == "addWidget" and c[1][0] is marker for c in w.method_calls))
        added = [c[1][0] for c in row.method_calls if c[0] == "addWidget"]
        self.assertEqual(added, [self.boxes["algorithm_dki"], gear, marker])

    def test_b2_the_reason_is_shown_in_chinese(self):
        owner = self.single_shell_owner()
        owner._onboardingLanguage = "zh"
        self.popup(owner)
        self.markers()["algorithmUnavailable_noddi"].setToolTip.assert_called_with(
            availability.current_data_blocks(owner)["noddi"].text("zh"))
        self.assertIn("NODDI 至少需要 2 个非零 b 值壳层", self.boxes["algorithm_noddi"].setToolTip.call_args[0][0])

    def test_b2_apply_keeps_a_stored_blocked_algorithm_in_catalog_order(self):
        owner = self.single_shell_owner("dti,noddi")
        self.assertTrue(self.popup(owner, chosen={"dti"}))
        self.assertEqual(owner._parameterNode.orchestrationMicrostructureStages, "dti,noddi")
        self.assertTrue(owner._parameterNode.orchestrationRunNoddi)

    def test_b2_apply_does_not_add_a_blocked_algorithm_that_was_not_stored(self):
        owner = self.single_shell_owner("dti")
        self.assertTrue(self.popup(owner, chosen={"dti"}))
        self.assertEqual(owner._parameterNode.orchestrationMicrostructureStages, "dti")

    def test_b2_unticking_everything_keeps_the_stored_blocked_value_valid_and_does_not_clear_the_family(self):
        owner = self.single_shell_owner("dti,noddi")
        self.assertTrue(self.popup(owner, chosen=set()))
        node = owner._parameterNode
        self.assertEqual(node.orchestrationMicrostructureStages, "noddi")
        self.assertTrue(node.targetMicrostructure)
        self.assertNotIn("microstructure", getattr(owner, "_clearedAlgorithmFamilies", set()))

    def test_b2_unticking_everything_clears_the_family_as_before_when_nothing_blocked_was_stored(self):
        owner = self.single_shell_owner("dti")
        self.assertTrue(self.popup(owner, chosen=set()))
        self.assertFalse(owner._parameterNode.targetMicrostructure)
        self.assertIn("microstructure", owner._clearedAlgorithmFamilies)

    def test_b2_ticking_new_algorithms_replaces_the_enabled_choices_and_keeps_blocked_ones(self):
        subject = write_subject(self.root / "s2", THREE_GROUPS)  # nothing blocked
        owner = make_owner(subject, stages="dti,noddi")
        self.popup(owner, chosen={"dti", "freewater"})
        self.assertEqual(owner._parameterNode.orchestrationMicrostructureStages, "dti,freewater")

    def test_b2_no_blocks_means_no_markers_and_the_old_behaviour(self):
        subject = write_subject(self.root / "s2", THREE_GROUPS)
        owner = make_owner(subject, stages="dti,noddi")
        self.popup(owner, blocks={})
        self.assertEqual(self.markers(), {})
        for box in self.boxes.values():
            self.assertFalse(any(call.args == (False,) for call in box.setEnabled.call_args_list))
        self.assertTrue(self.boxes["algorithm_noddi"].checked)

    def test_b2_without_fsl_the_eddy_and_topup_rows_get_the_marker_and_keep_their_text(self):
        owner = make_owner(stages="dti")
        owner._parameterNode.orchestrationPreprocessingStages = "denoise,eddy,topup"
        self.popup(owner, "preprocessing", blocks={}, fsl_offered=False)
        markers = self.markers()
        self.assertEqual(set(markers), {"algorithmUnavailable_eddy", "algorithmUnavailable_topup"})
        for name in ("eddy", "topup"):
            tip = self.boxes["algorithm_" + name].setToolTip.call_args[0][0]
            self.assertIn("Eddy and Topup need FSL", tip)
            markers["algorithmUnavailable_" + name].setToolTip.assert_called_with(tip)
        self.assertTrue(any(object_name(w) == "algorithmPickerAvailabilityNote" for w in self.qt.created))

    def test_b2_with_fsl_offered_no_preprocessing_marker_is_made(self):
        owner = make_owner(stages="dti")
        self.popup(owner, "preprocessing", blocks={}, fsl_offered=True)
        self.assertEqual(self.markers(), {})

    def test_b2_data_blocks_never_touch_other_families(self):
        owner = make_owner(stages="dti")
        blocks = {"noddi": availability.Block("noddi", "noddi_needs_two_shells", {"count": 1, "shells": [1000.0]})}
        self.popup(owner, "tractography", blocks=blocks)
        self.assertEqual(self.markers(), {})


class FakeText:
    def __init__(self):
        self.text, self.toolTip, self.enabled_calls = "", "", []

    def setText(self, value):
        self.text = value

    def setToolTip(self, value):
        self.toolTip = value

    def setAccessibleName(self, value):
        self.accessibleName = value

    def setEnabled(self, value):
        self.enabled_calls.append(value)


class SummaryTest(TempCase):
    def owner(self, root, stages, lang="en"):
        names = ("targetPreprocessingCheckBox", "targetMicrostructureCheckBox", "targetTractographyCheckBox",
                 "targetBundlesCheckBox", "targetConnectomeCheckBox")
        ui = SimpleNamespace(**{name: FakeText() for name in names})
        owner = make_owner(root, stages=stages, lang=lang)
        owner.ui = ui
        owner._algorithmSummaryLabels = {name: FakeText() for name in names}
        owner._algorithmButtons = {}
        return owner

    def refresh(self, owner):
        from shared.ui_compact import refresh_compact_algorithms

        refresh_compact_algorithms(owner)
        return owner._algorithmSummaryLabels["targetMicrostructureCheckBox"]

    def test_b3_the_summary_lists_what_will_run_and_the_tooltip_names_each_skipped_reason(self):
        owner = self.owner(write_subject(self.root / "s1", SINGLE_SHELL), "dti,noddi")
        summary = self.refresh(owner)
        self.assertEqual(summary.text, "DTI")
        reason = availability.current_data_blocks(owner)["noddi"].text("en")
        self.assertIn("NODDI: " + reason, summary.toolTip)
        self.assertEqual(owner.ui.targetMicrostructureCheckBox.toolTip, summary.toolTip)
        self.assertEqual(summary.enabled_calls, [])

    def test_b3_one_tooltip_line_per_skipped_algorithm_in_catalog_order(self):
        owner = self.owner(write_subject(self.root / "s1", SINGLE_SHELL), "dti,freewater,noddi,ifw")
        summary = self.refresh(owner)
        self.assertEqual(summary.text, "DTI")
        lines = [line for line in summary.toolTip.splitlines() if line.split(":")[0] in {"FW", "NODDI", "iFW"}]
        self.assertEqual([line.split(":")[0] for line in lines], ["FW", "NODDI", "iFW"])

    def test_b3_the_tooltip_reason_is_chinese_in_chinese(self):
        owner = self.owner(write_subject(self.root / "s1", SINGLE_SHELL), "dti,noddi", lang="zh")
        summary = self.refresh(owner)
        self.assertIn("NODDI: NODDI 至少需要 2 个非零 b 值壳层", summary.toolTip)

    def test_b3_without_blocks_the_summary_equals_the_stored_selection_and_nothing_is_muted(self):
        from shared.ui_compact import compact_algorithm_labels

        owner = self.owner(write_subject(self.root / "s2", THREE_GROUPS), "dti,noddi")
        summary = self.refresh(owner)
        self.assertEqual(summary.text, compact_algorithm_labels(owner._parameterNode)["targetMicrostructureCheckBox"])
        self.assertEqual(summary.text, "DTI + NODDI")
        self.assertEqual(summary.enabled_calls, [])
        self.assertNotIn("not available", summary.toolTip)

    def test_b3_when_every_stored_algorithm_is_blocked_the_summary_says_so_and_is_muted(self):
        owner = self.owner(write_subject(self.root / "s1", SINGLE_SHELL), "noddi")
        summary = self.refresh(owner)
        self.assertEqual(summary.text, "NODDI · not available for this data")
        self.assertEqual(summary.enabled_calls, [False])
        owner._onboardingLanguage = "zh"
        summary = self.refresh(owner)
        self.assertEqual(summary.text, "NODDI · 当前数据不可用")

    def test_b3_the_summary_follows_the_data_and_unmutes_when_the_data_fits(self):
        owner = self.owner(write_subject(self.root / "s1", SINGLE_SHELL), "noddi")
        summary = self.refresh(owner)
        self.assertEqual(summary.enabled_calls, [False])
        write_subject(self.root / "s1", THREE_GROUPS)  # the user chose other data
        summary = self.refresh(owner)
        self.assertEqual(summary.text, "NODDI")
        self.assertEqual(summary.enabled_calls, [False, True])
        self.assertEqual(owner._parameterNode.orchestrationMicrostructureStages, "noddi")  # never rewritten

    def test_b3_the_stored_selection_is_not_rewritten_by_a_refresh(self):
        owner = self.owner(write_subject(self.root / "s1", SINGLE_SHELL), "dti,noddi")
        before = dict(vars(owner._parameterNode))
        self.refresh(owner)
        self.assertEqual(vars(owner._parameterNode), before)

    def test_b3_a_tile_change_refreshes_the_summary(self):
        import ast

        source = (REPO_ROOT / "shared" / "ui_prepare_inputs.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        handler = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "text_changed")
        called = {getattr(c.func, "id", getattr(c.func, "attr", "")) for c in ast.walk(handler) if isinstance(c, ast.Call)}
        self.assertIn("refresh_compact_algorithms", called)


class FilterRunTest(unittest.TestCase):
    def flags(self, **kw):
        return ProductTargetFlags(microstructure=True, **kw)

    def test_b3_nothing_blocked_returns_none_so_the_old_path_is_used(self):
        self.assertIsNone(filter_run_for_data(self.flags(), "dti,noddi", set()))
        self.assertIsNone(filter_run_for_data(self.flags(), "dti,noddi", {"dki"}))
        self.assertIsNone(filter_run_for_data(ProductTargetFlags(), "dti,noddi", {"noddi"}))  # family not ticked

    def test_b3_a_blocked_stored_stage_is_removed_from_the_request(self):
        run = filter_run_for_data(self.flags(), "dti,freewater,noddi", {"noddi", "ifw"})
        self.assertEqual(run.skipped, ("noddi",))
        self.assertEqual(run.stages, ("dti", "freewater"))
        self.assertEqual((run.dti, run.freewater, run.noddi, run.dki, run.ifw), (True, True, False, False, False))
        self.assertTrue(run.flags.microstructure)

    def test_b3_a_family_with_nothing_left_is_left_out(self):
        run = filter_run_for_data(self.flags(tractography=True), "noddi", {"noddi"})
        self.assertEqual(run.skipped, ("noddi",))
        self.assertFalse(run.flags.microstructure)
        self.assertTrue(run.flags.tractography)
        self.assertEqual(run.stages, ())

    def test_b3_the_closure_of_the_filtered_run_has_no_noddi(self):
        from shared.ui_run_controller import selected_closure

        run = filter_run_for_data(self.flags(), "dti,noddi", {"noddi"})
        closure = selected_closure(
            run.flags, tractography_backend="csd_act", bundle_backend="wmql", microstructure_dti=run.dti,
            microstructure_freewater=run.freewater, microstructure_noddi=run.noddi,
            microstructure_dki=run.dki, microstructure_ifw=run.ifw)
        self.assertNotIn("microstructure_noddi", closure.targets)
        self.assertIn("microstructure", closure.targets)
        self.assertNotIn("microstructure_noddi", closure.requested)


class RunRequestTest(TempCase):
    """``_onRunSelectedClicked`` with the stored ``dti,noddi`` and single-shell data."""

    def build_widget(self, root, stages="dti,noddi", batch=False):
        from unittest.mock import MagicMock

        mods = {name: MagicMock() for name in ("qt", "slicer", "vtk", "slicer.ScriptedLoadableModule", "ctk")}
        self._patch = mock.patch.dict(sys.modules, mods)
        self._patch.start()
        self.addCleanup(self._patch.stop)
        sys.modules.pop("shared.module_widget", None)
        self.addCleanup(sys.modules.pop, "shared.module_widget", None)
        import shared.module_widget as mw

        widget = mw.SharedWidgetMixin()
        owner = make_owner(root, stages=stages, batch=batch, targetMicrostructure=True)
        node = owner._parameterNode
        node.outputDirectory = ""
        node.device = node.ddsegDevice = "auto"
        node.autoInstallPythonDependencies = False
        node.orchestrationRunName = "Test1"
        node.orchestrationSelectStreamlines = 1000
        node.connectomeBackend = "auto"
        node.connectomeCompressMrtrixDenseState = "unset"
        node.brainMaskBackend, node.brainMaskDevice, node.pnlMaskRoute = "pnl_cnn", "auto", "auto"
        node.maxParallelModels = 1
        widget._parameterNode = node
        widget.logic = SimpleNamespace(planSelectedProductOrchestration=lambda **kw: SimpleNamespace(
            run_plan_path=Path("/x/plan.json"), run_name="Test1", items=[], warnings=[]))
        widget._pipelineActive = False
        widget._prepareInputWidgets = {"source_kind": "nifti"}
        widget._onboardingLanguage = "en"
        widget.parent = None
        self.logged, self.status, self.started = [], [], []
        stubs = dict(
            validateInputs=lambda manual=False: True,
            _prepare_inputs_widget_text=lambda key: "",
            _input_preview_pending_source=lambda: None,
            _input_preview_is_batch=lambda: batch,
            _effective_subject_id=lambda: "sub",
            _save_common_user_settings=lambda: None,
            _apply_shared_inputs_to_workflow_fields=lambda: None,
            _topup_validation=lambda batch_mode=False: SimpleNamespace(errors=[], warnings=[], parameters=TopupParameters()),
            _environment_dependency_path_kwargs=lambda: {},
            _environment_check_kwargs=lambda **kw: {},
            _environment_check_signature=lambda kw: "sig",
            _cached_environment_result=lambda sig: object(),
            _environment_doctor_result_for_run=lambda kw, sig: SimpleNamespace(statuses=[]),
            _update_run_plan_table=lambda plan: None,
            _setStatus=lambda *a: self.status.append(a),
            _summarise_missing=lambda items: "missing " + ",".join(items),
            appendLog=lambda text: self.logged.append(text),
            _startBackgroundPipeline=lambda kwargs, paths, selected: self.started.append((kwargs, selected)),
            _startBackgroundBatchPipeline=lambda *a, **k: self.started.append(("batch", a, k)),
            _ifw_echo_time_ms=lambda: None,
            _provided_tractography=lambda: (),
        )
        for name, fn in stubs.items():
            setattr(widget, name, fn)
        widget._setupRestartRequired = False
        widget._setupFunctionalFailure = False
        widget._selectedSetupActive = False
        widget._inputImportPending = False
        widget.ui = SimpleNamespace()
        widget._mw = mw
        return widget

    def click(self, widget):
        mw = widget._mw
        with mock.patch.object(mw, "preflight_blockers_for_targets", return_value={}), \
                mock.patch.object(mw, "build_single_subject_run_kwargs",
                                  wraps=mw.build_single_subject_run_kwargs), \
                mock.patch("shared.setup_resume.pending_setup", return_value=None), \
                mock.patch("shared.ui_input_inspection.inspect_input_directory", return_value={"mode": "subject_root"}), \
                mock.patch("shared.ui_compact.sync_detected_input_format"):
            widget._onRunSelectedClicked()

    def test_b3_run_request_without_noddi_node_unchanged_one_skip_line(self):
        subject = write_subject(self.root / "s1", SINGLE_SHELL)
        widget = self.build_widget(subject, "dti,noddi")
        before = dict(vars(widget._parameterNode))
        self.click(widget)
        self.assertEqual(len(self.started), 1, self.status)
        kwargs, selected = self.started[0]
        self.assertFalse(kwargs["microstructure_run_noddi"])
        self.assertNotIn("microstructure_noddi", kwargs["targets"])
        self.assertNotIn("microstructure_noddi", kwargs["requested_targets"])
        self.assertIn("microstructure", kwargs["targets"])
        self.assertEqual(selected, ["Microstructure"])
        self.assertEqual(vars(widget._parameterNode), before)  # the stored selection is untouched
        skipped = [line for line in self.logged if line.startswith("Skipped NODDI")]
        self.assertEqual(len(skipped), 1, self.logged)
        self.assertIn("at least 2 non-zero b-shells", skipped[0])

    def test_b3_run_request_without_free_water_and_ifw_has_no_auto_added_segmentation(self):
        subject = write_subject(self.root / "s1", SINGLE_SHELL)
        widget = self.build_widget(subject, "dti,freewater,ifw")
        self.click(widget)
        self.assertEqual(len(self.started), 1, self.status)
        kwargs, _ = self.started[0]
        for key in ("microstructure_run_free_water", "microstructure_run_ifw", "microstructure_run_noddi",
                    "microstructure_run_dki"):
            self.assertFalse(kwargs[key], key)
        for name in ("microstructure_ifw", "microstructure_freewater", "segmentation"):
            self.assertNotIn(name, kwargs["targets"])
            self.assertNotIn(name, kwargs["requested_targets"])
        skipped = [line.split(":")[0] for line in self.logged if line.startswith("Skipped")]
        self.assertEqual(skipped, ["Skipped FW", "Skipped iFW"])
        self.assertEqual(widget._parameterNode.orchestrationMicrostructureStages, "dti,freewater,ifw")

    def test_b3_run_request_drops_dki_that_the_acquisition_cannot_run(self):
        subject = write_subject(self.root / "s1", SINGLE_SHELL)  # 6 volumes: too few for DKI
        widget = self.build_widget(subject, "dti,dki")
        self.click(widget)
        kwargs, _ = self.started[0]
        self.assertFalse(kwargs["microstructure_run_dki"])
        self.assertNotIn("microstructure_dki", kwargs["targets"])
        self.assertEqual([line.split(":")[0] for line in self.logged if line.startswith("Skipped")], ["Skipped DKI"])

    def test_b3_run_request_keeps_noddi_when_the_data_fits(self):
        subject = write_subject(self.root / "s1", THREE_GROUPS)
        widget = self.build_widget(subject, "dti,noddi")
        self.click(widget)
        kwargs, _ = self.started[0]
        self.assertTrue(kwargs["microstructure_run_noddi"])
        self.assertIn("microstructure_noddi", kwargs["targets"])
        self.assertEqual([line for line in self.logged if line.startswith("Skipped")], [])

    def test_b3_run_with_only_blocked_algorithms_shows_the_empty_selection_message_and_starts_nothing(self):
        subject = write_subject(self.root / "s1", SINGLE_SHELL)
        widget = self.build_widget(subject, "noddi")
        self.click(widget)
        self.assertEqual(self.started, [])
        self.assertEqual(self.status[-1], ("error", "missing Target products"))
        self.assertEqual(len([line for line in self.logged if line.startswith("Skipped NODDI")]), 1)

    def test_b3_the_family_is_left_out_but_other_families_still_run(self):
        subject = write_subject(self.root / "s1", SINGLE_SHELL)
        widget = self.build_widget(subject, "noddi")
        widget._parameterNode.targetConnectome = False
        widget._parameterNode.targetPreprocessing = True
        self.click(widget)
        kwargs, selected = self.started[0]
        self.assertEqual(selected, ["DWI Preprocessing"])
        self.assertFalse(any(t.startswith("microstructure") for t in kwargs["targets"]))

    def capture_plan(self, widget):
        self.plan_kwargs = []

        def plan(**kwargs):
            self.plan_kwargs.append(kwargs)
            return SimpleNamespace(run_plan_path=Path("/x/plan.json"), run_name="Test1", items=[], warnings=[])

        widget.logic = SimpleNamespace(planSelectedProductOrchestration=plan)

    def test_f1_run_filter_keeps_noddi_when_the_final_files_fit(self):
        subject = write_subject(self.root / "s1", SINGLE_SHELL)
        write_final(subject, THREE_GROUPS)
        widget = self.build_widget(subject, "dti,noddi")
        with mock.patch("shared.orchestrator._preprocessing_complete", return_value=True):
            self.click(widget)
        kwargs, _ = self.started[0]
        self.assertTrue(kwargs["microstructure_run_noddi"])
        self.assertIn("microstructure_noddi", kwargs["targets"])
        self.assertEqual([line for line in self.logged if line.startswith("Skipped")], [])

    def test_f1_run_filter_drops_noddi_when_the_final_files_do_not_fit(self):
        subject = write_subject(self.root / "s1", THREE_GROUPS)
        write_final(subject, SINGLE_SHELL)
        widget = self.build_widget(subject, "dti,noddi")
        with mock.patch("shared.orchestrator._preprocessing_complete", return_value=True):
            self.click(widget)
        kwargs, _ = self.started[0]
        self.assertFalse(kwargs["microstructure_run_noddi"])
        self.assertEqual(len([line for line in self.logged if line.startswith("Skipped NODDI")]), 1)

    def test_f1_the_gate_asks_the_planner_predicate_with_the_planner_roots(self):
        subject = write_subject(self.root / "s1", SINGLE_SHELL)
        widget = self.build_widget(subject, "dti,noddi")
        widget._parameterNode.outputDirectory = str(self.root / "out")
        self.capture_plan(widget)
        from shared.ui_run_controller import resolve_single_subject_output_root

        output_root = resolve_single_subject_output_root(str(self.root / "out"), "sub").resolve()
        write_final(subject, THREE_GROUPS, output_root=output_root)
        with mock.patch("shared.orchestrator._preprocessing_complete", return_value=True) as complete:
            self.click(widget)
        planner = self.plan_kwargs[0]
        expected = (Path(planner["subject_root"]).expanduser().resolve(), "sub",
                    Path(planner["output_root"]).expanduser().resolve())
        self.assertIn(mock.call(*expected), complete.call_args_list)
        kwargs, _ = self.started[0]
        self.assertTrue(kwargs["microstructure_run_noddi"])

    def test_f2_run_with_a_convertible_source_and_a_stale_receipt_skips_nothing(self):
        root = self.root / "conv"
        write_gradients(root / "data.nii.gz", root / "bvals", root / "bvecs", THREE_GROUPS)
        write_subject(root, SINGLE_SHELL)
        from shared.convertible_input import RECEIPT

        (root / "input" / RECEIPT).write_text('{"kind": "nifti", "signature": {"source": "old"}}\n')
        widget = self.build_widget(root, "dti,noddi")
        self.click(widget)
        kwargs, _ = self.started[0]
        self.assertTrue(kwargs["microstructure_run_noddi"])
        self.assertEqual([line for line in self.logged if line.startswith("Skipped")], [])

    def test_f3_the_run_filter_resolves_fresh_and_ignores_the_summary_cache(self):
        subject = write_subject(self.root / "s1", SINGLE_SHELL)
        write_final(subject, THREE_GROUPS)
        widget = self.build_widget(subject, "dti,noddi")
        with mock.patch("shared.algorithm_availability._monotonic", lambda: 100.0, create=True):
            with mock.patch("shared.orchestrator._preprocessing_complete", return_value=False):
                self.assertIn("noddi", availability.current_data_blocks(widget))  # the summary caches the raw files
            with mock.patch("shared.orchestrator._preprocessing_complete", return_value=True):
                self.click(widget)  # preprocessing finished inside the cache period
        kwargs, _ = self.started[0]
        self.assertTrue(kwargs["microstructure_run_noddi"])
        self.assertEqual([line for line in self.logged if line.startswith("Skipped")], [])

    def test_b3_batch_mode_is_not_filtered(self):
        subject = write_subject(self.root / "s1", SINGLE_SHELL)
        widget = self.build_widget(subject, "dti,noddi", batch=True)
        widget._parameterNode.batchMode = True
        widget._parameterNode.batchRoot = str(self.root)
        batch_options = SimpleNamespace(batch_root=self.root, batch_size=1, arrange_inputs="none")
        mw = widget._mw
        with mock.patch.object(mw, "validate_batch_run_options", return_value=(batch_options, "")), \
                mock.patch.object(mw, "preflight_blockers_for_targets", return_value={}), \
                mock.patch("shared.setup_resume.pending_setup", return_value=None):
            widget._onRunSelectedClicked()
        self.assertEqual(len(self.started), 1, self.status)
        self.assertEqual([line for line in self.logged if line.startswith("Skipped")], [])


if __name__ == "__main__":
    unittest.main()
