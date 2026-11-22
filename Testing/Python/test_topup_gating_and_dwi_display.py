#!/usr/bin/env python3
"""Tests for Topup reverse-PE gating, Eddy/Topup error separation, 4D DWI result loading, and volume slider."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import nibabel as nib
import nrrd
import numpy as np

from Testing.Python.test_input_preview import FakeDwiSlicer, FakeLabel, FakeSpin, fake_full_dwi
from Testing.Python.test_ui_run_history import by_name, fake_qt, object_name


def _write_las_4d_nifti(folder: Path, stem: str = "sub01_dwi_final") -> tuple[Path, Path, Path, np.ndarray]:
    """Write a left-handed LAS 4D NIfTI (matching DWIConvert NrrdToFSL output) plus bval/bvec."""
    affine = np.array(
        [
            [-2.0, 0.0, 0.0, 100.0],
            [0.0, 2.0, 0.0, 64.0],
            [0.0, 0.0, 2.0, -50.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    data = np.arange(4 * 5 * 6 * 3, dtype=np.int16).reshape(4, 5, 6, 3)
    dwi = folder / f"{stem}.nii.gz"
    bval = folder / f"{stem}.bval"
    bvec = folder / f"{stem}.bvec"
    nib.save(nib.Nifti1Image(data, affine), str(dwi))
    bval.write_text("0 1000 1000\n", encoding="utf-8")
    bvec.write_text("0 1 0\n0 0 1\n0 0 0\n", encoding="utf-8")
    return dwi, bval, bvec, affine


class TopupGatingPopupTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _make_ui_and_owner(self, *, reverse_b0: str = "", stages: str = "denoise,eddy", lang: str = "en", batch: bool = False):
        orig_combo = mock.MagicMock(name="origCombo")
        orig_combo.count = 2
        orig_combo.itemText.side_effect = lambda i: ("From sidecar", "AP")[i]
        orig_combo.itemData.side_effect = lambda i: ("", "AP")[i]
        orig_combo.currentIndex = 0
        orig_combo.toolTip = "PE tip"

        orig_spin = mock.MagicMock(name="origSpin")
        orig_spin.minimum = 0.0
        orig_spin.maximum = 1.0
        orig_spin.decimals = 4
        orig_spin.singleStep = 0.01
        orig_spin.value = 0.0
        orig_spin.toolTip = "Readout tip"

        ui_ns = SimpleNamespace(
            preprocessingPhaseEncodingDirectionComboBox=orig_combo,
            preprocessingPhaseEncodingDirectionLabel=SimpleNamespace(text="Phase-encoding direction"),
            preprocessingReadoutTimeSpinBox=orig_spin,
            preprocessingReadoutTimeLabel=SimpleNamespace(text="Total readout time (s)"),
            preprocessingReversePhaseB0Path=SimpleNamespace(toolTip="Reverse tip"),
            preprocessingReversePhaseB0Label=SimpleNamespace(text="Reverse-PE b=0 image"),
        )
        node = SimpleNamespace(
            targetPreprocessing=True,
            orchestrationPreprocessingStages=stages,
            preprocessingPhaseEncodingDirection="",
            preprocessingReadoutTime=0.0,
            preprocessingReversePhaseB0=reverse_b0,
            batchMode=batch,
            experimentsEnabled=False,
            experimentScope="subject",
            inputDirectory=str(self.root),
        )
        owner = SimpleNamespace(
            ui=ui_ns,
            _uiWidget=None,
            _pipelineActive=False,
            _selectedSetupActive=False,
            _onboardingLanguage=lang,
            _parameterNode=node,
            _lastEnvironmentResult=None,
            _lastEnvironmentSignature=None,
        )
        return owner, orig_combo, orig_spin

    def _open_popup(self, owner, *, qt=None, interact=None, fsl_offered: bool = True):
        from shared import ui_algorithm_picker as picker

        qt = qt or fake_qt()
        qt.QComboBox = type("FakeComboBox", (), {})
        qt.QDoubleSpinBox = type("FakeDoubleSpinBox", (), {})
        slicer = mock.MagicMock(name="slicer")
        orig_dialog = qt.QDialog.side_effect

        def make_dialog(*args, **kwargs):
            dlg = orig_dialog(*args, **kwargs)
            dlg.exec_.side_effect = lambda: (interact(qt, dlg) if callable(interact) else qt.QDialog.Rejected)
            return dlg

        qt.QDialog.side_effect = make_dialog
        with (
            mock.patch.dict(sys.modules, {"qt": qt, "slicer": slicer, "ctk": mock.MagicMock()}),
            mock.patch("shared.selected_setup.fsl_preprocessing_offered", return_value=fsl_offered),
        ):
            accepted = picker.choose_algorithms(owner, "preprocessing")
        return qt, accepted

    def test_topup_is_disabled_with_hover_and_click_question_mark_until_reverse_b0_is_supplied(self) -> None:
        owner, _, _ = self._make_ui_and_owner(reverse_b0="", lang="zh")
        qt, _ = self._open_popup(owner)

        topup_box = next(w for w in qt.created if w.kind == "QCheckBox" and w.objectName == "algorithm_topup")
        eddy_box = next(w for w in qt.created if w.kind == "QCheckBox" and w.objectName == "algorithm_eddy")
        eddy_gear = by_name(qt, "algorithmSettings_eddy")
        topup_gear = by_name(qt, "algorithmSettings_topup")
        marker = by_name(qt, "algorithmUnavailable_topup")
        reason = by_name(qt, "algorithmUnavailableReason_topup")
        path_edit = by_name(qt, "topupReversePhaseB0LineEdit")
        browse_btn = by_name(qt, "topupReversePhaseB0BrowseButton")
        clear_btn = by_name(qt, "topupReversePhaseB0ClearButton")

        self.assertFalse(topup_box.checked)
        topup_box.setEnabled.assert_called_with(False)
        topup_gear.setEnabled.assert_called_with(False)
        eddy_box.setEnabled.assert_not_called()
        eddy_gear.setEnabled.assert_not_called()

        self.assertEqual(marker.setText.call_args[0][0], "?")
        self.assertIn("相位编码方向相反", marker.setToolTip.call_args[0][0])
        reason.setVisible.assert_called_once_with(False)
        # Clicking '?' toggles the inline explanation under Topup.
        marker.clicked.connect.call_args[0][0](False)
        reason.setVisible.assert_called_with(True)
        marker.clicked.connect.call_args[0][0](False)
        reason.setVisible.assert_called_with(False)
        self.assertIsNotNone(path_edit)
        self.assertIsNotNone(browse_btn)
        clear_btn.setVisible.assert_called_with(False)

    def test_providing_and_clearing_reverse_b0_toggles_topup_enabled_and_question_mark_live(self) -> None:
        rev_nrrd = self.root / "sub01_PA_b0.nrrd"
        rev_nrrd.write_bytes(b"NRRD0004\n")
        owner, _, _ = self._make_ui_and_owner(reverse_b0="", lang="en")
        observed = {}

        def interact(qt, dlg):
            topup_box = next(w for w in qt.created if w.kind == "QCheckBox" and w.objectName == "algorithm_topup")
            topup_gear = by_name(qt, "algorithmSettings_topup")
            marker = by_name(qt, "algorithmUnavailable_topup")
            path_edit = by_name(qt, "topupReversePhaseB0LineEdit")
            clear_btn = by_name(qt, "topupReversePhaseB0ClearButton")
            on_text = path_edit.textChanged.connect.call_args[0][0]
            on_clear = clear_btn.clicked.connect.call_args[0][0]

            # 1) Select a valid .nrrd reverse-PE file -> Topup enables, auto-checks, gear enables, '?' hides.
            path_edit.setText(str(rev_nrrd))
            on_text(str(rev_nrrd))
            observed["after_select_checked"] = topup_box.checked
            observed["after_select_enabled_calls"] = [c.args for c in topup_box.setEnabled.call_args_list]
            observed["after_select_gear_calls"] = [c.args for c in topup_gear.setEnabled.call_args_list]
            observed["after_select_marker_vis"] = [c.args for c in marker.setVisible.call_args_list]

            # 2) Clear the file -> Topup unchecks, disables, gear disables, '?' reappears.
            on_clear()
            observed["after_clear_checked"] = topup_box.checked
            observed["after_clear_enabled_last"] = topup_box.setEnabled.call_args[0][0]
            observed["after_clear_gear_last"] = topup_gear.setEnabled.call_args[0][0]
            observed["after_clear_marker_last"] = marker.setVisible.call_args[0][0]

            # 3) Re-select valid file and Apply -> commits reverse-PE path and includes 'topup'.
            path_edit.setText(str(rev_nrrd))
            on_text(str(rev_nrrd))
            return qt.QDialog.Accepted

        _, accepted = self._open_popup(owner, interact=interact)
        self.assertTrue(accepted)
        self.assertTrue(observed["after_select_checked"])
        self.assertEqual(observed["after_select_enabled_calls"][-1], (True,))
        self.assertEqual(observed["after_select_gear_calls"][-1], (True,))
        self.assertEqual(observed["after_select_marker_vis"][-1], (False,))
        self.assertFalse(observed["after_clear_checked"])
        self.assertFalse(observed["after_clear_enabled_last"])
        self.assertFalse(observed["after_clear_gear_last"])
        self.assertTrue(observed["after_clear_marker_last"])
        self.assertEqual(owner._parameterNode.preprocessingReversePhaseB0, str(rev_nrrd))
        self.assertIn("topup", owner._parameterNode.orchestrationPreprocessingStages)

    def test_batch_mode_disables_topup_with_question_mark_and_hides_inline_file_picker(self) -> None:
        owner, _, _ = self._make_ui_and_owner(reverse_b0="", lang="en", batch=True)
        qt, _ = self._open_popup(owner)
        topup_box = next(w for w in qt.created if w.kind == "QCheckBox" and w.objectName == "algorithm_topup")
        marker = by_name(qt, "algorithmUnavailable_topup")
        topup_box.setEnabled.assert_called_with(False)
        self.assertIn("batch", marker.setToolTip.call_args[0][0].lower())
        self.assertEqual([w for w in qt.created if object_name(w) == "topupReversePhaseB0LineEdit"], [])


class TopupValidationAndEddyErrorTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_explicit_topup_stage_without_reverse_b0_is_rejected_not_silently_skipped(self) -> None:
        from preprocessing.pipeline import DwiPreprocessingConfig, run_dwi_preprocessing
        from shared.orchestrator import ProductOrchestratorConfig, _preprocessing_pe_block
        from shared.topup_parameters import validate_topup_parameters

        val = validate_topup_parameters(
            preprocessing_selected=True,
            preprocessing_stages="brain_mask,topup",
            phase_encoding_direction="AP",
            reverse_phase_b0="",
        )
        self.assertFalse(val.ok)
        self.assertTrue(any("reverse-pe" in err.lower() for err in val.errors))

        cfg = ProductOrchestratorConfig(
            subject_id="sub01",
            subject_root=self.root,
            targets=("preprocessing",),
            preprocessing_stages="brain_mask,topup",
            preprocessing_phase_encoding_direction="AP",
            preprocessing_reverse_phase_b0=None,
        )
        block = _preprocessing_pe_block(cfg, self.root)
        self.assertIsNotNone(block)
        self.assertIn("reverse", block.lower())

        subj = self.root / "sub01"
        inp = subj / "input"
        inp.mkdir(parents=True)
        dwi, bval, bvec, _ = _write_las_4d_nifti(inp, "data")
        (inp / "bvals").write_text(bval.read_text(encoding="utf-8"), encoding="utf-8")
        (inp / "bvecs").write_text(bvec.read_text(encoding="utf-8"), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "reverse phase-encoding"):
            run_dwi_preprocessing(
                DwiPreprocessingConfig(
                    subject_id="sub01",
                    input_root=subj,
                    execute=True,
                    run_denoise=False,
                    run_eddy=False,
                    run_topup=True,
                    reverse_phase_b0=None,
                    phase_encoding_direction="AP",
                )
            )

    def test_nrrd_and_multivolume_4d_reverse_b0_are_accepted(self) -> None:
        from preprocessing.pipeline import _prepare_reverse_b0_nrrd
        from shared.topup_parameters import validate_topup_parameters

        rev_nrrd = self.root / "rev_pa.nrrd"
        vol = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
        nrrd.write(
            str(rev_nrrd),
            vol,
            {
                "space": "left-posterior-superior",
                "space directions": np.diag([2.0, 2.0, 2.0]),
                "space origin": np.array([-100.0, -64.0, -50.0]),
            },
        )
        val = validate_topup_parameters(
            preprocessing_selected=True,
            preprocessing_stages="denoise,eddy,topup",
            phase_encoding_direction="AP",
            reverse_phase_b0=str(rev_nrrd),
        )
        self.assertTrue(val.ok, val.errors)
        self.assertEqual(val.parameters.reverse_phase_b0, rev_nrrd)

        out_nii = self.root / "rev_pa_4d.nii.gz"
        _prepare_reverse_b0_nrrd(rev_nrrd, out_nii)
        loaded = nib.load(str(out_nii))
        self.assertEqual(loaded.shape, (3, 4, 5, 1))
        expected_ras = np.array(
            [
                [-2.0, 0.0, 0.0, 100.0],
                [0.0, -2.0, 0.0, 64.0],
                [0.0, 0.0, 2.0, -50.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        np.testing.assert_allclose(loaded.affine, expected_ras)

    def test_eddy_only_error_dialog_points_to_eddy_settings_not_topup(self) -> None:
        from Testing.Python.test_b2_fix_round import widget_methods
        from shared.topup_parameters import TopupParameters, TopupValidation

        captured = {}
        Widget = widget_methods(
            "_show_topup_error",
            show_actionable_error_dialog=lambda **kwargs: captured.update(kwargs),
        )
        me = Widget()
        me.parent = None
        me._parameterNode = SimpleNamespace(orchestrationPreprocessingStages="denoise,eddy")
        validation = TopupValidation(parameters=TopupParameters(), errors=("Phase-encoding direction needed",))
        me._show_topup_error(validation)
        self.assertIn("Eddy", captured["what"])
        self.assertNotIn("topup", captured["what"].lower())
        self.assertIn("Eddy > Settings", captured["next_step"])
        self.assertIn("-rpe_none", captured["details"])

        captured.clear()
        me._parameterNode = SimpleNamespace(orchestrationPreprocessingStages="denoise,eddy,topup")
        me._show_topup_error(validation)
        self.assertIn("Topup", captured["what"])
        self.assertIn("Reverse-PE", captured["next_step"])
        self.assertIn("-rpe_pair", captured["details"])


class DwiResultLoadingTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_load_dwi_result_loads_las_4d_nifti_as_dwi_node_with_exact_ras_affine(self) -> None:
        from shared.result_scene import is_dwi_result, load_dwi_result

        dwi, bval, bvec, affine = _write_las_4d_nifti(self.root, "test-7_dwi_final")
        candidate = {"product": "preprocessing.dwi", "key": "dwi", "path": str(dwi)}
        manifest = {
            "subject_id": "test-7",
            "tasks": [
                {
                    "name": "preprocessing.dwi",
                    "outputs": {"dwi": str(dwi), "bval": str(bval), "bvec": str(bvec)},
                }
            ],
        }
        self.assertTrue(is_dwi_result(candidate))
        fake = FakeDwiSlicer()
        node = load_dwi_result(fake, dwi, "test-7_preprocessing.dwi_dwi", candidate, manifest)
        self.assertIsNotNone(node)
        self.assertEqual(node.cls, "vtkMRMLDiffusionWeightedVolumeNode")
        self.assertEqual(node.name, "test-7_preprocessing.dwi_dwi")
        np.testing.assert_allclose(node.ijk, affine)
        self.assertEqual(node.GetNumberOfGradients(), 3)
        self.assertEqual([node.GetBValue(i) for i in range(3)], [0.0, 1000.0, 1000.0])
        self.assertEqual(node.display.component, 0)

    def test_load_dwi_result_fallback_without_bvec_still_preserves_exact_las_affine(self) -> None:
        from shared.result_scene import load_dwi_result

        dwi, bval, bvec, affine = _write_las_4d_nifti(self.root, "test-7_dwi_final")
        bvec.unlink()
        candidate = {"product": "preprocessing.dwi", "key": "dwi", "path": str(dwi)}
        manifest = {"subject_id": "test-7", "tasks": [{"name": "preprocessing.dwi", "outputs": {"dwi": str(dwi), "bval": str(bval)}}]}
        fake = FakeDwiSlicer()
        node = load_dwi_result(fake, dwi, "test-7_preprocessing.dwi_dwi", candidate, manifest)
        self.assertIsNotNone(node)
        self.assertEqual(node.cls, "vtkMRMLScalarVolumeNode")
        np.testing.assert_allclose(node.ijk, affine)


class VolumeSliderStepperTest(unittest.TestCase):
    def test_slider_is_installed_and_synced_with_spinbox_and_display_node(self) -> None:
        from Testing.Python.test_b2_fix_round import widget_methods
        from shared import input_preview as ip
        from shared import ui_compact

        qt = fake_qt()
        layout = mock.MagicMock(name="configurationLayout")
        owner = SimpleNamespace(
            ui=SimpleNamespace(configurationLayout=layout),
            _onboardingLanguage="en",
            _on_input_preview_clicked=lambda: None,
            _on_input_preview_component_changed=lambda v: None,
            _refresh_input_preview_controls=lambda: None,
        )
        with (
            mock.patch.dict(sys.modules, {"qt": qt}),
            mock.patch("shared.ui_prepare_inputs.place_input_preview_button", lambda _o: None),
        ):
            ui_compact.install_input_preview_controls(owner)

        slider = by_name(qt, "inputPreviewComponentSlider")
        spin = by_name(qt, "inputPreviewComponentSpinBox")
        self.assertIs(owner._inputPreviewWidgets["slider"], slider)
        self.assertIs(owner._inputPreviewWidgets["spin"], spin)
        slider.setEnabled.assert_called_with(False)

        # Verify bidirectional runtime sync between slider, spinbox, and DWI display node.
        scene = {}
        fake_slicer = SimpleNamespace(mrmlScene=SimpleNamespace(GetNodeByID=lambda i: scene.get(i)))
        fake_vtk = SimpleNamespace(vtkCommand=SimpleNamespace(ModifiedEvent=33))
        Widget = widget_methods(
            "_input_preview_dwi_node",
            "_refresh_input_preview_stepper",
            "_on_input_preview_component_changed",
            "_on_input_preview_display_modified",
            "_input_preview_is_batch",
            slicer=fake_slicer,
            vtk=fake_vtk,
        )
        me = Widget()
        me.observers = []
        me.addObserver = lambda obj, event, method: me.observers.append((obj, event, method.__name__))
        me.removeObserver = lambda obj, event, method: me.observers.remove((obj, event, method.__name__))
        me._onboardingLanguage = "en"
        me._parameterNode = None
        fake_slider, fake_spin = FakeSpin(), FakeSpin()
        me._inputPreviewWidgets = {
            "stepper": FakeLabel(),
            "slider": fake_slider,
            "spin": fake_spin,
            "info": FakeLabel(),
            "stepper_label": FakeLabel(),
        }
        fake = FakeDwiSlicer()
        info = ip.show_dwi_in_scene(fake, fake_full_dwi((1000, 0, 2000, 5)), None, subject_id="s")
        node = fake.nodes[-1]
        scene[info["volume_id"]] = node
        me._inputPreviewDwiNodeId = info["volume_id"]

        me._refresh_input_preview_stepper()
        self.assertTrue(fake_slider.enabled and fake_spin.enabled)
        self.assertEqual((fake_slider.minimum, fake_slider.maximum, fake_slider.value), (1, 4, 2))
        self.assertEqual((fake_spin.minimum, fake_spin.maximum, fake_spin.value), (1, 4, 2))

        # Dragging the slider updates both the spinbox and the MRML display node.
        me._on_input_preview_component_changed(4)
        self.assertEqual(fake_spin.value, 4)
        self.assertEqual(fake_slider.value, 4)
        self.assertEqual(node.display.component, 3)

        # Changing Volumes > DWI Component updates both slider and spinbox.
        node.display.SetDiffusionComponent(0)
        me._on_input_preview_display_modified(node.display, "ModifiedEvent")
        self.assertEqual(fake_slider.value, 1)
        self.assertEqual(fake_spin.value, 1)
        self.assertEqual(me._inputPreviewWidgets["info"].text, "/ 4  (b = 1000)")

    def test_results_dwi_slider_links_to_active_results_4d_dwi_and_swaps_cleanly(self) -> None:
        from Testing.Python.test_b2_fix_round import widget_methods
        from shared import input_preview as ip
        from shared import ui_run_history

        qt = fake_qt()
        results_layout = mock.MagicMock(name="resultsLayout")
        owner = SimpleNamespace(
            ui=SimpleNamespace(resultsLayout=results_layout, resultsCollapsibleButton=None),
            _onboardingLanguage="en",
            _on_browse_run_directory_clicked=lambda: None,
            _on_open_latest_report_clicked=lambda: None,
            _on_open_output_tree_report_clicked=lambda: None,
            _on_run_history_selection_changed=lambda: None,
            _refresh_run_history=lambda: None,
        )
        with mock.patch.dict(sys.modules, {"qt": qt}):
            ui_run_history.install_results_panel(owner)
        self.assertIs(owner._resultsDwiWidgets["slider"], by_name(qt, "resultsDwiComponentSlider"))
        self.assertIs(owner._resultsDwiWidgets["spin"], by_name(qt, "resultsDwiComponentSpinBox"))
        by_name(qt, "resultsDwiStepper").hide.assert_called()

        scene = {}
        red_composite = SimpleNamespace(
            _bg=None,
            _fg=None,
            GetLayoutName=lambda: "Red",
            GetBackgroundVolumeID=lambda: red_composite._bg,
            GetForegroundVolumeID=lambda: red_composite._fg,
        )
        fake_slicer = SimpleNamespace(
            mrmlScene=SimpleNamespace(GetNodeByID=lambda i: scene.get(i)),
            util=SimpleNamespace(getNodesByClass=lambda cls: [red_composite] if cls == "vtkMRMLSliceCompositeNode" else []),
        )
        fake_vtk = SimpleNamespace(vtkCommand=SimpleNamespace(ModifiedEvent=33))
        Widget = widget_methods(
            "_input_preview_dwi_node",
            "_active_results_dwi_node",
            "_refresh_results_dwi_stepper",
            "_on_results_dwi_component_changed",
            "_on_results_dwi_display_modified",
            "_refresh_input_preview_stepper",
            "_on_input_preview_component_changed",
            "_on_input_preview_display_modified",
            "_input_preview_is_batch",
            slicer=fake_slicer,
            vtk=fake_vtk,
        )
        me = Widget()
        me.observers = []
        me.addObserver = lambda obj, event, method: me.observers.append((obj, event, method.__name__))
        me.removeObserver = lambda obj, event, method: me.observers.remove((obj, event, method.__name__))
        me._onboardingLanguage = "en"
        me._parameterNode = SimpleNamespace(batchMode=False)

        in_slider, in_spin, in_stepper, in_info = FakeSpin(), FakeSpin(), FakeLabel(), FakeLabel()
        res_slider, res_spin, res_stepper, res_info, res_label = FakeSpin(), FakeSpin(), FakeLabel(), FakeLabel(), FakeLabel()
        res_stepper.visible = False
        me._inputPreviewWidgets = {
            "stepper": in_stepper, "slider": in_slider, "spin": in_spin, "info": in_info, "stepper_label": FakeLabel(),
        }
        me._resultsDwiWidgets = {
            "stepper": res_stepper, "slider": res_slider, "spin": res_spin, "info": res_info, "stepper_label": res_label,
        }

        # Build an input preview DWI AND a Results-loaded preprocessed DWI.
        fake_builder = FakeDwiSlicer()
        in_info_dict = ip.show_dwi_in_scene(fake_builder, fake_full_dwi((0, 1000)), None, subject_id="sub01")
        in_node = fake_builder.nodes[-1]
        in_node.GetName = lambda: "sub01_input_dwi"
        scene[in_info_dict["volume_id"]] = in_node
        me._inputPreviewDwiNodeId = in_info_dict["volume_id"]

        res_node, _ = ip.build_dwi_volume_node(
            fake_builder, fake_full_dwi((1000, 0, 2000, 500, 3000)), "sub01_preprocessing.dwi_dwi"
        )
        res_node.SetAttribute("SlicerDDSurfer.Product", "preprocessing.dwi")
        res_node.GetName = lambda: "sub01_preprocessing.dwi_dwi"
        scene[res_node.GetID()] = res_node

        # 1) When Red background is the input preview DWI, only the Inputs slider is visible.
        red_composite._bg = in_node.GetID()
        me._refresh_input_preview_stepper()
        self.assertTrue(in_stepper.visible)
        self.assertFalse(res_stepper.visible)

        # 2) Clicking the eye on Preprocessed DWI sets Red background -> Results slider takes over; Inputs slider hides.
        red_composite._bg = res_node.GetID()
        me._refresh_input_preview_stepper()
        self.assertTrue(res_stepper.visible and res_slider.enabled and res_spin.enabled)
        self.assertFalse(in_stepper.visible)
        self.assertEqual((res_slider.minimum, res_slider.maximum, res_slider.value), (1, 5, 2))
        self.assertEqual(res_info.text, "/ 5  (b = 0)")
        self.assertIn("sub01_preprocessing.dwi_dwi", res_label.tooltip)

        # 3) Scrubbing the Results slider drives the Results DWI display node and syncs spinbox.
        me._on_results_dwi_component_changed(5)
        self.assertEqual(res_spin.value, 5)
        self.assertEqual(res_node.display.component, 4)
        me._on_results_dwi_display_modified(res_node.display, "ModifiedEvent")
        self.assertEqual(res_info.text, "/ 5  (b = 3000)")

        # 4) Even in Batch mode, Results 4D DWI slider remains usable when viewing a batch subject's DWI!
        me._parameterNode = SimpleNamespace(batchMode=True)
        me._refresh_input_preview_stepper()
        self.assertTrue(res_stepper.visible and res_slider.enabled)
        self.assertFalse(in_stepper.visible)

        # 5) Hiding the Results DWI restores the Inputs slider (in Single mode) and detaches Results display observer.
        me._parameterNode = SimpleNamespace(batchMode=False)
        red_composite._bg = None
        me._refresh_input_preview_stepper()
        self.assertFalse(res_stepper.visible)
        self.assertTrue(in_stepper.visible)
        self.assertNotIn(res_node.display, [obs[0] for obs in me.observers])


class LegacyConfigBatchHygieneAnd4dNiftiReverseB0Test(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_sanitize_unavailable_topup_stage_migrates_legacy_config_and_batch_mode(self) -> None:
        from shared.topup_parameters import sanitize_unavailable_topup_stage, validate_topup_parameters

        # Legacy saved config ("denoise,eddy,topup" with empty reverse-PE b=0) strips topup -> UI and Run agree!
        legacy_node = SimpleNamespace(
            targetPreprocessing=True,
            orchestrationPreprocessingStages="denoise,eddy,topup",
            preprocessingReversePhaseB0="",
            batchMode=False,
        )
        self.assertTrue(sanitize_unavailable_topup_stage(legacy_node))
        self.assertEqual(legacy_node.orchestrationPreprocessingStages, "denoise,eddy")
        val = validate_topup_parameters(
            preprocessing_selected=True,
            preprocessing_stages=legacy_node.orchestrationPreprocessingStages,
            phase_encoding_direction="AP",
            reverse_phase_b0="",
        )
        self.assertTrue(val.ok, val.errors)

        # Explicit "topup"-only is NOT stripped so UI and validation still explain what is missing.
        topup_only = SimpleNamespace(
            targetPreprocessing=True,
            orchestrationPreprocessingStages="topup",
            preprocessingReversePhaseB0="",
            batchMode=False,
        )
        self.assertFalse(sanitize_unavailable_topup_stage(topup_only))
        self.assertEqual(topup_only.orchestrationPreprocessingStages, "topup")

        # Valid reverse-PE file keeps "denoise,eddy,topup" in single-subject mode, but strips "topup" in Batch mode.
        rev = self.root / "rev_pa.nii.gz"
        rev.write_bytes(b"\x00")
        valid_node = SimpleNamespace(
            targetPreprocessing=True,
            orchestrationPreprocessingStages="denoise,eddy,topup",
            preprocessingReversePhaseB0=str(rev),
            batchMode=False,
        )
        self.assertFalse(sanitize_unavailable_topup_stage(valid_node))
        self.assertEqual(valid_node.orchestrationPreprocessingStages, "denoise,eddy,topup")
        self.assertTrue(sanitize_unavailable_topup_stage(valid_node, batch_mode=True))
        self.assertEqual(valid_node.orchestrationPreprocessingStages, "denoise,eddy")

    def test_set_batch_mode_clears_single_subject_reverse_b0_and_strips_topup(self) -> None:
        import contextlib
        from Testing.Python.test_b2_fix_round import widget_methods

        rev = self.root / "rev_pa.nii.gz"
        rev.write_bytes(b"\x00")
        fake_slicer = SimpleNamespace(util=SimpleNamespace(NodeModify=lambda _n: contextlib.nullcontext()))
        Widget = widget_methods("_set_batch_mode", slicer=fake_slicer)
        me = Widget()
        me.logs = []
        me.appendLog = me.logs.append
        me._parameterNode = SimpleNamespace(
            batchMode=False,
            orchestrationPreprocessingStages="denoise,eddy,topup",
            preprocessingReversePhaseB0=str(rev),
        )
        me._set_batch_mode(True)
        self.assertTrue(me._parameterNode.batchMode)
        self.assertEqual(me._parameterNode.preprocessingReversePhaseB0, "")
        self.assertEqual(me._parameterNode.orchestrationPreprocessingStages, "denoise,eddy")
        self.assertTrue(any("reverse-PE b=0" in line for line in me.logs))

    def test_batch_mode_picker_offers_clear_button_if_stale_reverse_b0_lingers(self) -> None:
        from shared import ui_algorithm_picker as picker

        rev = self.root / "stale_pa.nii.gz"
        rev.write_bytes(b"\x00")
        qt = fake_qt()
        qt.QComboBox = type("FakeComboBox", (), {})
        qt.QDoubleSpinBox = type("FakeDoubleSpinBox", (), {})
        node = SimpleNamespace(
            targetPreprocessing=True,
            orchestrationPreprocessingStages="denoise,eddy",
            preprocessingPhaseEncodingDirection="AP",
            preprocessingReadoutTime=0.0,
            preprocessingReversePhaseB0=str(rev),
            batchMode=True,
            experimentsEnabled=False,
            experimentScope="subject",
        )
        owner = SimpleNamespace(
            ui=SimpleNamespace(),
            _uiWidget=None,
            _pipelineActive=False,
            _selectedSetupActive=False,
            _onboardingLanguage="en",
            _parameterNode=node,
            _lastEnvironmentResult=None,
            _lastEnvironmentSignature=None,
        )
        orig_dialog = qt.QDialog.side_effect

        def make_dialog(*args, **kwargs):
            dlg = orig_dialog(*args, **kwargs)

            def interact():
                btn = by_name(qt, "topupBatchClearReverseButton")
                btn.clicked.connect.call_args[0][0]()
                return qt.QDialog.Accepted

            dlg.exec_.side_effect = interact
            return dlg

        qt.QDialog.side_effect = make_dialog
        with (
            mock.patch.dict(sys.modules, {"qt": qt, "slicer": mock.MagicMock(), "ctk": mock.MagicMock()}),
            mock.patch("shared.selected_setup.fsl_preprocessing_offered", return_value=True),
        ):
            accepted = picker.choose_algorithms(owner, "preprocessing")
        self.assertTrue(accepted)
        self.assertEqual(node.preprocessingReversePhaseB0, "")

    def test_4d_nifti_reverse_b0_selects_b0_index_from_sibling_bval(self) -> None:
        from preprocessing.pipeline import (
            DwiPreprocessingConfig,
            _build_commands,
            _planned_paths,
            _reverse_b0_nifti_volume_index,
        )
        from shared.pipeline import RawInputBundle

        # 4D reverse-PE NIfTI whose b=0 volume is at index 2 (not index 0).
        rev_nii = self.root / "sub01_PA.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((4, 5, 6, 4), dtype=np.float32), np.eye(4)), str(rev_nii))
        rev_bval = self.root / "sub01_PA.bval"
        rev_bval.write_text("1000 1000 0 2000\n", encoding="utf-8")
        self.assertEqual(_reverse_b0_nifti_volume_index(rev_nii, 4, 50.0), 2)

        # Verify _build_commands emits -coord 3 2 for mrconvert.
        inp = self.root / "sub01" / "input"
        inp.mkdir(parents=True)
        dwi, bval, bvec, _ = _write_las_4d_nifti(inp, "data")
        raw = RawInputBundle(
            subject_id="sub01",
            subject_root=self.root / "sub01",
            input_dir=inp,
            dwi_path=dwi,
            bval_path=bval,
            bvec_path=bvec,
            mask_path=inp / "nodif_brain_mask.nii.gz",
        )
        cfg = DwiPreprocessingConfig(
            subject_id="sub01",
            input_root=self.root / "sub01",
            output_dir=self.root / "out",
            reverse_phase_b0=rev_nii,
            phase_encoding_direction="AP",
        )
        paths = _planned_paths(self.root / "out", "sub01")
        backend = {"commands": {"mrconvert": "mrconvert", "dwiextract": "dwiextract", "mrcat": "mrcat", "dwifslpreproc": "dwifslpreproc"}}
        commands = _build_commands(raw, cfg, paths, backend)
        rev_cmd = next(cmd for cmd in commands if cmd[0] == "mrconvert" and rev_nii.name in cmd[1])
        coord_idx = rev_cmd.index("-coord")
        self.assertEqual(rev_cmd[coord_idx : coord_idx + 3], ["-coord", "3", "2"])

        # Sibling .bval with NO b<=threshold volume raises a clear ValueError.
        rev_bval.write_text("1000 1000 1500 2000\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "no b=0 volume"):
            _reverse_b0_nifti_volume_index(rev_nii, 4, 50.0)

        # Without a sibling .bval, falls back to index 0 (multi-b0 stack).
        rev_bval.unlink()
        self.assertEqual(_reverse_b0_nifti_volume_index(rev_nii, 4, 50.0), 0)

    def test_nrrd_reverse_b0_reorients_to_match_las_reference_affine_and_voxels(self) -> None:
        from preprocessing.pipeline import _prepare_reverse_b0_nrrd

        # Main DWI in LAS (DWIConvert NrrdToFSL output): Y axis flipped relative to LPS NRRD.
        las_ref = self.root / "main_las.nii.gz"
        las_affine = np.array(
            [
                [-2.0, 0.0, 0.0, 100.0],
                [0.0, 2.0, 0.0, 58.0],
                [0.0, 0.0, 2.0, -50.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        # Voxels in LPS order (size 3x4x5): flipping axis 1 maps j -> 3-j and shifts origin 58 + 3*2 = 64 -> LPS origin -64.
        lps_vol = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
        las_vol = np.ascontiguousarray(lps_vol[:, ::-1, :])
        nib.save(nib.Nifti1Image(las_vol[..., np.newaxis], las_affine), str(las_ref))

        rev_nrrd = self.root / "rev_lps.nrrd"
        nrrd.write(
            str(rev_nrrd),
            lps_vol,
            {
                "space": "left-posterior-superior",
                "space directions": np.diag([2.0, 2.0, 2.0]),
                "space origin": np.array([-100.0, -64.0, -50.0]),
            },
        )
        out_nii = self.root / "rev_aligned_las.nii.gz"
        _prepare_reverse_b0_nrrd(rev_nrrd, out_nii, reference_path=las_ref)
        loaded = nib.load(str(out_nii))
        np.testing.assert_allclose(loaded.affine, las_affine)
        np.testing.assert_allclose(np.asarray(loaded.dataobj[..., 0]), las_vol)


if __name__ == "__main__":
    unittest.main()
