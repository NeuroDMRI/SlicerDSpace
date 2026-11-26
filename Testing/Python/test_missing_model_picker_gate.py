"""Unit tests for missing-model/atlas disabled picker rows, '?' download actions, and pre-run reminders."""

from __future__ import annotations

import contextlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import ui_algorithm_picker as picker
from shared.ui_on_demand_resources import (
    core_profile_title,
    detect_missing_direct_algorithms,
    installed_wma_atlases,
)
from Testing.Python.test_ui_run_history import fake_qt, object_name


def _populate_profile(asset_root: Path, manifest_name: str, profile: str) -> None:
    payload = json.loads((REPO_ROOT / "models" / manifest_name).read_text(encoding="utf-8"))
    spec = payload["profiles"][profile]
    rels = list(spec.get("required_paths") or ())
    for child in spec.get("include_profiles") or ():
        rels.extend(payload["profiles"][child].get("required_paths") or ())
    for rel in rels:
        target = asset_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"weights")


def _populate_wma_atlas(asset_root: Path, family: str) -> None:
    from tractography.wma_atlases import atlas_definition

    rel_dir = "bundles/wma" if family == "org" else f"bundles/wma/{family}"
    base = asset_root / rel_dir
    for child in atlas_definition(family)["directories"]:
        (base / child).mkdir(parents=True, exist_ok=True)
    (base / "extracted_resource_index.json").write_text("{}", encoding="utf-8")


def _make_owner(asset_root: Path, *, lang: str = "en", brain_mask_backend: str = "pnl_cnn", **node_extra):
    node = SimpleNamespace(
        inputDirectory="",
        subjectId="sub",
        batchMode=False,
        experimentsEnabled=False,
        experimentScope="subject",
        targetPreprocessing=True,
        targetSurface=False,
        targetSurfacePostprocess=False,
        targetParcellation=True,
        targetSegmentation=True,
        targetMicrostructure=True,
        targetTractography=True,
        targetBundles=True,
        targetConnectome=True,
        brainMaskBackend=brain_mask_backend,
        pnlMaskRoute="native",
        wmaAtlas="org",
        orchestrationPreprocessingStages="brain_mask,denoise",
        orchestrationMicrostructureStages="dti",
        orchestrationTractographyBackend="csd_act",
        orchestrationBundleBackend="wmql,tractcloud",
        orchestrationConnectomeMetrics="sift2",
        orchestrationRunFreeWater=False,
        orchestrationRunNoddi=False,
        orchestrationRunDki=False,
        orchestrationRunIfw=False,
        ukfParameters="{}",
        wmaInputTractography="",
        **node_extra,
    )
    return SimpleNamespace(
        ui=SimpleNamespace(),
        _uiWidget=None,
        _pipelineActive=False,
        _selectedSetupActive=False,
        _onboardingLanguage=lang,
        _lastEnvironmentResult=None,
        _lastEnvironmentSignature=None,
        _parameterNode=node,
        _environment_dependency_path_kwargs=lambda: {"model_assets_dir": str(asset_root)},
        _input_preview_pending_source=lambda: None,
        _input_preview_is_batch=lambda: False,
        _prepare_inputs_widget_text=lambda _key: "",
        _effective_subject_id=lambda: "sub",
        _anatomicalPriorsRow=(None, None, None, None, None),
    )


class MissingModelPickerGateTests(unittest.TestCase):
    def _run_picker(self, owner, family: str, *, during=None):
        qt = fake_qt()
        qt.QTimer.singleShot.side_effect = lambda _ms, fn: fn()
        dialog_factory = qt.QDialog.side_effect

        def make_dialog(*args, **kwargs):
            dialog = dialog_factory(*args, **kwargs)

            def fake_exec():
                if during is not None:
                    outcome = during(qt, dialog)
                    if outcome is not None:
                        return outcome
                return 0

            dialog.exec_.side_effect = fake_exec
            return dialog

        qt.QDialog.side_effect = make_dialog
        slicer = mock.MagicMock(name="slicer")
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.dict(sys.modules, {"qt": qt, "slicer": slicer, "ctk": mock.MagicMock()}))
            stack.enter_context(mock.patch("shared.selected_setup.fsl_preprocessing_offered", return_value=True))
            stack.enter_context(mock.patch("shared.ui_compact.refresh_compact_algorithms"))
            stack.enter_context(mock.patch("shared.ui_on_demand_resources.maybe_prompt_on_demand_install"))
            stack.enter_context(mock.patch("shared.ui_on_demand_resources.bundled_model_asset_roots", return_value=()))
            dl_mock = stack.enter_context(mock.patch("shared.ui_on_demand_resources.prompt_download_missing_resource"))
            picker.choose_algorithms(owner, family)
        return qt, dl_mock

    def test_anatomical_priors_disables_only_missing_ddsurfer_and_wires_download(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            assets = Path(td)
            _populate_profile(assets, "assets_manifest.json", "segmentation")
            _populate_profile(assets, "assets_manifest.json", "parcellation")
            owner = _make_owner(assets)

            def click_download(qt, _dlg):
                dl_btn = next(w for w in qt.created if object_name(w) == "algorithmDownloadButton_targetSurface")
                dl_btn.clicked.connect.call_args[0][0]()
                return 0

            qt, dl_mock = self._run_picker(owner, "anatomical_priors", during=click_download)
            boxes = {str(w.objectName): w for w in qt.created if w.kind == "QCheckBox"}
            markers = {object_name(w): w for w in qt.created if object_name(w).startswith("algorithmUnavailable_")}

            boxes["algorithm_targetSurface"].setEnabled.assert_called_with(False)
            boxes["algorithm_targetSegmentation"].setEnabled.assert_not_called()
            boxes["algorithm_targetParcellation"].setEnabled.assert_not_called()
            self.assertEqual(set(markers), {"algorithmUnavailable_targetSurface"})
            # Reason text stays hidden until '?' is clicked, while Download button stays visible.
            reason_lbl = next(w for w in qt.created if object_name(w) == "algorithmUnavailableReason_targetSurface")
            reason_lbl.setVisible.assert_called_with(False)
            dl_panel = next(w for w in qt.created if object_name(w) == "algorithmMissingAssetPanel_targetSurface")
            dl_panel.setVisible.assert_not_called()
            dl_mock.assert_called_once()
            self.assertEqual(dl_mock.call_args[0][1]["pack_id"], "ddsurfer")

    def test_bundles_disables_wma_only_when_no_wma_atlas_is_installed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            assets = Path(td)
            _populate_profile(assets, "external_assets_manifest.json", "tractcloud")
            owner = _make_owner(assets)

            # 0 WMA atlases installed -> WMA disabled + '?' + atlas selector + Download button
            def pick_naba_and_download(qt, _dlg):
                combo = next(w for w in qt.created if object_name(w) == "algorithmDownloadAtlas_wma")
                combo.currentIndex = 1  # NABA
                dl_btn = next(w for w in qt.created if object_name(w) == "algorithmDownloadButton_wma")
                dl_btn.clicked.connect.call_args[0][0]()
                return 0

            qt, dl_mock = self._run_picker(owner, "bundles", during=pick_naba_and_download)
            boxes = {str(w.objectName): w for w in qt.created if w.kind == "QCheckBox"}
            markers = {object_name(w): w for w in qt.created if object_name(w).startswith("algorithmUnavailable_")}
            boxes["algorithm_wma"].setEnabled.assert_called_with(False)
            boxes["algorithm_tractcloud"].setEnabled.assert_not_called()
            boxes["algorithm_wmql"].setEnabled.assert_not_called()
            self.assertEqual(set(markers), {"algorithmUnavailable_wma"})
            dl_mock.assert_called_once()
            self.assertEqual(dl_mock.call_args[0][1]["pack_id"], "wma_naba")
            self.assertEqual(dl_mock.call_args[0][1]["atlas_family"], "naba")

            # Install ANY ONE atlas (naba) -> WMA is enabled with no '?' marker
            _populate_wma_atlas(assets, "naba")
            with mock.patch("shared.ui_on_demand_resources.bundled_model_asset_roots", return_value=()):
                self.assertEqual(installed_wma_atlases(owner), ("naba",))
            qt2, _ = self._run_picker(owner, "bundles")
            boxes2 = {str(w.objectName): w for w in qt2.created if w.kind == "QCheckBox"}
            markers2 = {object_name(w): w for w in qt2.created if object_name(w).startswith("algorithmUnavailable_")}
            boxes2["algorithm_wma"].setEnabled.assert_not_called()
            self.assertEqual(markers2, {})

    def test_preprocessing_disables_pnl_cnn_only_when_missing_and_not_provided(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            assets = Path(td)
            owner_pnl = _make_owner(assets, brain_mask_backend="pnl_cnn")
            with mock.patch("shared.ui_on_demand_resources.bundled_model_asset_roots", return_value=()):
                missing = detect_missing_direct_algorithms(owner_pnl, "preprocessing")
            self.assertEqual(set(missing), {"brain_mask"})
            self.assertEqual(missing["brain_mask"]["pack_id"], "pnl_cnn")

            owner_prov = _make_owner(assets, brain_mask_backend="provided")
            with mock.patch("shared.ui_on_demand_resources.bundled_model_asset_roots", return_value=()):
                self.assertEqual(detect_missing_direct_algorithms(owner_prov, "preprocessing"), {})

    def test_indirect_dependents_stay_enabled_and_pre_run_confirm_names_friendly_weights(self) -> None:
        from shared.setup_components import plan_components
        from shared.ui_selected_setup import _compact_component_rows

        with tempfile.TemporaryDirectory() as td:
            assets = Path(td)  # Empty: DDSeg, DDParcel, DDSurfer all missing
            owner = _make_owner(assets)
            # Tractography (CSD/ACT, SAF) and Connectome must stay enabled in their pickers
            for family in ("tractography", "connectome"):
                qt, _ = self._run_picker(owner, family)
                markers = {object_name(w): w for w in qt.created if object_name(w).startswith("algorithmUnavailable_")}
                self.assertEqual(markers, {}, family)

            # Pre-run confirmation compact rows format core model profiles with friendly titles
            plan = {
                "asset_profiles": ["segmentation", "parcellation"],
                "core_model_profile_bytes": 672_000_000,
            }
            components = plan_components(plan)
            import html

            for lang, expected_seg, expected_parc in (
                ("en", core_profile_title("segmentation", False), core_profile_title("parcellation", False)),
                ("zh", core_profile_title("segmentation", True), core_profile_title("parcellation", True)),
            ):
                rows = "\n".join(_compact_component_rows(SimpleNamespace(_onboardingLanguage=lang), components, set()))
                self.assertIn(html.escape(expected_seg), rows)
                self.assertIn(html.escape(expected_parc), rows)


if __name__ == "__main__":
    unittest.main()
