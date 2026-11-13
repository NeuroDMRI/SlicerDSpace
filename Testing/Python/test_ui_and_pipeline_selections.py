"""End-to-end unit tests for UI refinements, input format selector, preprocessing stages, and connectome modes."""

from __future__ import annotations

from dataclasses import replace
import inspect
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from preprocessing.pipeline import DwiPreprocessingConfig, run_dwi_preprocessing
from shared.algorithm_selection import CONNECTOME, PREPROCESSING, algorithms
from shared.convertible_input import find_convertible_input
from shared.environment import EnvironmentDoctorConfig, build_dependency_specs, run_needs_brain_mask
from shared.module_logic import SharedLogicMixin
from shared.orchestrator import (
    ProductOrchestratorConfig,
    _effective_preprocessing_mask_backend,
    _preprocessing_config,
    _resolved_connectome_flags,
    _resolved_preprocessing_stages,
    _task_name_for_target,
    plan_subject_products,
)
from shared.pipeline import inspect_subject_directory
from shared.run_subject_pipeline_cli import build_parser
from shared.topup_parameters import TopupParameters, route_uses_topup, validate_topup_parameters
from shared.ui_algorithm_picker import DEFAULT_CHOICES, FAMILIES, selected_algorithms, set_algorithms
from shared.ui_branding import install_header_wordmark
from shared.ui_compact import compact_algorithm_labels, install_input_format_selector, refresh_input_format_selector
from shared.ui_run_controller import (
    RunSelection,
    build_background_cli_args,
    build_batch_background_cli_args,
    build_single_subject_run_kwargs,
)
from shared.ui_run_state import gui_brain_mask_required
from shared.ui_styles import SLICERDSPACE_UI_STYLE


class _FakeComboBox:
    def __init__(self) -> None:
        self.items: list[tuple[str, str]] = []
        self.currentIndex = 0
        self.objectName = ""
        self.toolTip = ""
        self.accessibleName = ""
        self.signals_blocked = False
        self.currentIndexChanged = mock.MagicMock()

    def setObjectName(self, name: str) -> None:
        self.objectName = name

    def addItem(self, label: str, data: str) -> None:
        self.items.append((label, data))

    def count(self) -> int:
        return len(self.items)

    def setItemText(self, idx: int, text: str) -> None:
        _, data = self.items[idx]
        self.items[idx] = (text, data)

    def findData(self, value: str) -> int:
        for idx, (_, data) in enumerate(self.items):
            if data == value:
                return idx
        return -1

    def itemData(self, idx: int) -> str:
        return self.items[idx][1]

    def setCurrentIndex(self, idx: int) -> None:
        self.currentIndex = idx

    def blockSignals(self, flag: bool) -> None:
        self.signals_blocked = flag

    def setSizePolicy(self, *_args) -> None:
        pass

    def setToolTip(self, text: str) -> None:
        self.toolTip = text

    def setPlaceholderText(self, text: str) -> None:
        self.placeholderText = text

    def setAccessibleName(self, text: str) -> None:
        self.accessibleName = text


class _FakeWidget:
    def __init__(self, text: str = "") -> None:
        self.text = text
        self.objectName = ""
        self.toolTip = ""
        self.accessibleName = ""
        self.placeholderText = ""
        self.children: list[object] = []

    def setObjectName(self, name: str) -> None:
        self.objectName = name

    def setText(self, text: str) -> None:
        self.text = text

    def setToolTip(self, text: str) -> None:
        self.toolTip = text

    def setPlaceholderText(self, text: str) -> None:
        self.placeholderText = text

    def setAccessibleName(self, text: str) -> None:
        self.accessibleName = text

    def setFixedSize(self, *_args) -> None:
        pass

    def setCursor(self, *_args) -> None:
        pass

    def setFocusPolicy(self, *_args) -> None:
        pass

    def installEventFilter(self, filt) -> None:
        self.event_filter = filt

    def height(self) -> int:
        return 22

    def mapToGlobal(self, pt):
        return pt

    def addWidget(self, child, *_args) -> None:
        self.children.append(child)

    def addStretch(self, *_args) -> None:
        pass

    def setContentsMargins(self, *_args) -> None:
        pass

    def setSpacing(self, *_args) -> None:
        pass


def _write_raw_subject(root: Path, *, with_mask: bool = False, auto_mask_receipt: bool = False) -> Path:
    input_dir = root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "data.nii.gz").write_bytes(b"fake-dwi")
    (input_dir / "bvals").write_text("0 1000 1000 1000\n", encoding="utf-8")
    (input_dir / "bvecs").write_text("0 1 0 0\n0 0 1 0\n0 0 0 1\n", encoding="utf-8")
    (input_dir / "dwi.json").write_text(
        json.dumps({"PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.05}),
        encoding="utf-8",
    )
    if with_mask:
        (input_dir / "nodif_brain_mask.nii.gz").write_bytes(b"fake-mask")
    if auto_mask_receipt:
        (input_dir / "nodif_brain_mask.manifest.json").write_text("{}", encoding="utf-8")
    return root


class UiAndPipelineSelectionsTest(unittest.TestCase):
    def test_header_logo_enlarged_and_top_collapsible_buttons_hidden(self) -> None:
        branding_source = (Path(__file__).resolve().parents[2] / "shared" / "ui_branding.py").read_text(encoding="utf-8")
        self.assertIn("setFixedSize(176, 32)", branding_source)
        self.assertTrue(callable(install_header_wordmark))
        sheet = SLICERDSPACE_UI_STYLE
        self.assertIn("font-size: 17px;", sheet)
        self.assertIn("QToolButton#inputFormatHelpButton", sheet)
        self.assertIn('QToolButton[algorithmPickerButton="true"]', sheet)
        self.assertIn('QToolButton[algorithmPickerButton="true"]:hover:!disabled', sheet)

        source = (Path(__file__).resolve().parents[2] / "SlicerDSpace.py").read_text(encoding="utf-8")
        self.assertIn('"reloadCollapsibleButton"', source)
        self.assertIn('"helpCollapsibleButton"', source)

    def test_input_format_selector_and_hover_only_help_button(self) -> None:
        class _FakeQObject:
            def __init__(self, _parent=None) -> None:
                pass

        fake_tooltip = mock.MagicMock()
        fake_qt = SimpleNamespace(
            QObject=_FakeQObject,
            QEvent=SimpleNamespace(Enter=10, Leave=11),
            QPoint=lambda x, y: (x, y),
            QToolTip=fake_tooltip,
            QLabel=_FakeWidget,
            QWidget=lambda: _FakeWidget(),
            QHBoxLayout=lambda _parent=None: _FakeWidget(),
            QComboBox=_FakeComboBox,
            QToolButton=_FakeWidget,
            QSizePolicy=SimpleNamespace(Expanding=1, Fixed=0),
            Qt=SimpleNamespace(WhatsThisCursor=1, TabFocus=2, NoFocus=0, AlignVCenter=4),
        )
        form = mock.MagicMock()
        ui = SimpleNamespace(
            configurationFormLayout=form,
            inputDirectoryPath=_FakeWidget(),
            subjectIdLineEdit=_FakeWidget(),
            outputDirectoryPath=_FakeWidget(),
        )
        node = SimpleNamespace(inputFormat="nifti")
        owner = SimpleNamespace(ui=ui, _parameterNode=node, _onboardingLanguage="en")

        with mock.patch.dict(sys.modules, {"qt": fake_qt}):
            install_input_format_selector(owner)
            # Hovering shows tooltip immediately; leaving hides it
            owner._inputFormatHelpFilter.eventFilter(
                ui.inputFormatHelpButton, SimpleNamespace(type=lambda: 10)
            )
            fake_tooltip.showText.assert_called_once()
            owner._inputFormatHelpFilter.eventFilter(
                ui.inputFormatHelpButton, SimpleNamespace(type=lambda: 11)
            )
            fake_tooltip.hideText.assert_called_once()

        self.assertIsNotNone(getattr(ui, "inputFormatComboBox", None))
        self.assertIsNotNone(getattr(ui, "inputFormatHelpButton", None))
        # No static paragraph widget in the row: only combo + '?' hover button
        self.assertEqual([ item[1] for item in ui.inputFormatComboBox.items ], ["nifti", "dicom", "nrrd"])
        self.assertEqual(ui.inputFormatHelpButton.text, "?")
        self.assertIn("NIfTI", ui.inputFormatHelpButton.toolTip)
        self.assertIn("DICOM", ui.inputFormatHelpButton.toolTip)
        self.assertIn("NRRD", ui.inputFormatHelpButton.toolTip)
        self.assertIn("Subject root", ui.inputFormatHelpButton.toolTip)

        node.inputFormat = "dicom"
        owner._onboardingLanguage = "zh"
        refresh_input_format_selector(owner, "zh")
        self.assertEqual(ui.inputFormatComboBox.currentIndex, 1)
        self.assertIn("DICOM", ui.inputDirectoryPath.placeholderText)
        self.assertIn("受试者根目录", ui.inputFormatHelpButton.toolTip)

    def test_preprocessing_brain_mask_default_unchecked_and_fallback_rules(self) -> None:
        # Default preprocessing picker selection leaves brain_mask unchecked
        self.assertEqual(PREPROCESSING, ("brain_mask", "denoise", "eddy", "topup"))
        self.assertEqual(FAMILIES["preprocessing"][2], "orchestrationPreprocessingStages")
        self.assertEqual(DEFAULT_CHOICES["preprocessing"], "denoise,eddy,topup")
        self.assertEqual(selected_algorithms(None, "preprocessing"), ("denoise", "eddy", "topup"))
        self.assertEqual(compact_algorithm_labels(None)["targetPreprocessingCheckBox"], "DWI denoising + Eddy + Topup")

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            no_mask_root = _write_raw_subject(base / "sub_no_mask", with_mask=False)
            user_mask_root = _write_raw_subject(base / "sub_user_mask", with_mask=True)
            auto_mask_root = _write_raw_subject(base / "sub_auto_mask", with_mask=True, auto_mask_receipt=True)

            # Case 1: brain_mask unchecked (default) + NO user mask -> MUST run pnl_cnn automatically
            cfg_no_mask = ProductOrchestratorConfig(
                subject_id="sub_no_mask",
                subject_root=no_mask_root,
                targets=("preprocessing",),
                preprocessing_stages="denoise,eddy,topup",
            )
            self.assertEqual(_effective_preprocessing_mask_backend(cfg_no_mask, no_mask_root), "pnl_cnn")
            self.assertTrue(
                run_needs_brain_mask(
                    no_mask_root,
                    targets=("preprocessing",),
                    subject_id="sub_no_mask",
                    preprocessing_stages="denoise,eddy,topup",
                )
            )
            self.assertTrue(
                gui_brain_mask_required(
                    ("preprocessing",),
                    input_known=True,
                    input_mask_exists=False,
                    preprocessing_on=True,
                    mask_backend="pnl_cnn",
                    preprocessing_stages="denoise,eddy,topup",
                )
            )

            # Case 2: brain_mask unchecked (default) + user supplied mask -> reuse 'provided' (do NOT run PNL)
            cfg_user_mask = ProductOrchestratorConfig(
                subject_id="sub_user_mask",
                subject_root=user_mask_root,
                targets=("preprocessing",),
                preprocessing_stages="denoise,eddy,topup",
            )
            self.assertEqual(_effective_preprocessing_mask_backend(cfg_user_mask, user_mask_root), "provided")
            self.assertFalse(
                run_needs_brain_mask(
                    user_mask_root,
                    targets=("preprocessing",),
                    subject_id="sub_user_mask",
                    preprocessing_stages="denoise,eddy,topup",
                )
            )
            self.assertFalse(
                gui_brain_mask_required(
                    ("preprocessing",),
                    input_known=True,
                    input_mask_exists=True,
                    preprocessing_on=True,
                    mask_backend="pnl_cnn",
                    preprocessing_stages="denoise,eddy,topup",
                )
            )

            # Case 3: brain_mask explicitly checked + user supplied mask -> MUST run pnl_cnn
            cfg_checked = ProductOrchestratorConfig(
                subject_id="sub_user_mask",
                subject_root=user_mask_root,
                targets=("preprocessing",),
                preprocessing_stages="brain_mask,denoise,eddy,topup",
            )
            self.assertEqual(_effective_preprocessing_mask_backend(cfg_checked, user_mask_root), "pnl_cnn")
            self.assertTrue(
                run_needs_brain_mask(
                    user_mask_root,
                    targets=("preprocessing",),
                    subject_id="sub_user_mask",
                    preprocessing_stages="brain_mask,denoise,eddy,topup",
                )
            )
            self.assertTrue(
                gui_brain_mask_required(
                    ("preprocessing",),
                    input_known=True,
                    input_mask_exists=True,
                    preprocessing_on=True,
                    mask_backend="pnl_cnn",
                    preprocessing_stages="brain_mask,denoise,eddy,topup",
                )
            )

            # Case 4: auto-generated mask receipt present -> NOT treated as user-supplied mask
            cfg_auto_mask = ProductOrchestratorConfig(
                subject_id="sub_auto_mask",
                subject_root=auto_mask_root,
                targets=("preprocessing",),
                preprocessing_stages="denoise,eddy,topup",
            )
            self.assertEqual(_effective_preprocessing_mask_backend(cfg_auto_mask, auto_mask_root), "pnl_cnn")
            self.assertTrue(
                run_needs_brain_mask(
                    auto_mask_root,
                    targets=("preprocessing",),
                    subject_id="sub_auto_mask",
                    preprocessing_stages="denoise,eddy,topup",
                )
            )

    def test_selective_preprocessing_stages_and_environment_specs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _write_raw_subject(Path(tmp) / "sub01", with_mask=True)
            # Remove dwi.json to prove brain_mask+denoise without eddy/topup does NOT block on phase encoding
            (root / "input" / "dwi.json").unlink()

            cfg_denoise_only = ProductOrchestratorConfig(
                subject_id="sub01",
                subject_root=root,
                targets=("preprocessing",),
                preprocessing_stages="denoise",
                execute=False,
            )
            self.assertEqual(_resolved_preprocessing_stages(cfg_denoise_only), ("denoise",))
            pre_cfg = replace(_preprocessing_config(cfg_denoise_only, root, root / "outputs"), execute=False)
            self.assertTrue(pre_cfg.run_denoise)
            self.assertFalse(pre_cfg.run_eddy)
            self.assertFalse(pre_cfg.run_topup)

            planned = run_dwi_preprocessing(pre_cfg)
            self.assertEqual(planned.status, "planned")
            manifest = json.loads(planned.manifest_path.read_text(encoding="utf-8"))
            command_heads = [cmd["argv"][0] for cmd in manifest["commands"]]
            self.assertEqual(command_heads[:2], ["dipy", "dipy"])
            self.assertNotIn("dwifslpreproc", command_heads)
            self.assertNotIn("native-fsl-correction", command_heads)

            # brain_mask-only skips DIPY, Eddy, and Topup while forcing PNL mask regeneration
            cfg_mask_only = ProductOrchestratorConfig(
                subject_id="sub01",
                subject_root=root,
                targets=("preprocessing",),
                preprocessing_stages="brain_mask",
                execute=False,
            )
            pre_mask_only = replace(_preprocessing_config(cfg_mask_only, root, root / "outputs_mask"), execute=False)
            self.assertEqual(pre_mask_only.brain_mask_backend, "pnl_cnn")
            self.assertFalse(pre_mask_only.run_denoise)
            self.assertFalse(pre_mask_only.run_eddy)
            self.assertFalse(pre_mask_only.run_topup)
            planned_mask_only = run_dwi_preprocessing(pre_mask_only)
            mask_manifest = json.loads(planned_mask_only.manifest_path.read_text(encoding="utf-8"))
            mask_heads = [cmd["argv"][0] for cmd in mask_manifest["commands"]]
            self.assertNotIn("dipy", mask_heads)
            self.assertNotIn("dwifslpreproc", mask_heads)
            self.assertNotIn("native-fsl-correction", mask_heads)

            # Route / topup controls hide when neither eddy nor topup is selected
            self.assertFalse(route_uses_topup(True, preprocessing_stages="brain_mask,denoise"))
            self.assertTrue(route_uses_topup(True, preprocessing_stages="eddy"))
            val = validate_topup_parameters(
                preprocessing_selected=True,
                preprocessing_stages="brain_mask,denoise",
                subject_root=root,
            )
            self.assertEqual(val.errors, ())

            # Environment doctor omits FSL eddy/topup requirement when neither eddy nor topup is selected
            env_denoise = EnvironmentDoctorConfig(
                targets=("preprocessing",),
                repo_root=Path(__file__).resolve().parents[2],
                preprocessing_stages="denoise",
                preprocessing_execution_backend="legacy",
            )
            keys_denoise = {spec.key for spec in build_dependency_specs(env_denoise)}
            self.assertNotIn("command:eddy", keys_denoise)
            self.assertIn("python:dipy", keys_denoise)

            env_eddy_only = EnvironmentDoctorConfig(
                targets=("preprocessing",),
                repo_root=Path(__file__).resolve().parents[2],
                preprocessing_stages="eddy",
                preprocessing_execution_backend="legacy",
            )
            keys_eddy = {spec.key for spec in build_dependency_specs(env_eddy_only)}
            self.assertIn("command:eddy", keys_eddy)
            self.assertNotIn("python:dipy", keys_eddy)

    def test_connectome_default_sift2_and_fiber_count_modes(self) -> None:
        self.assertEqual(CONNECTOME, ("sift2", "fiber_count"))
        self.assertEqual(FAMILIES["connectome"][2], "orchestrationConnectomeMetrics")
        self.assertEqual(DEFAULT_CHOICES["connectome"], "sift2")
        self.assertEqual(selected_algorithms(None, "connectome"), ("sift2",))
        self.assertEqual(compact_algorithm_labels(None)["targetConnectomeCheckBox"], "SIFT2")
        self.assertEqual(algorithms("fiber_count", CONNECTOME, "sift2"), ("fiber_count",))
        self.assertEqual(algorithms("sift2,fiber_count", CONNECTOME, "sift2"), ("sift2", "fiber_count"))

        with tempfile.TemporaryDirectory() as tmp:
            root = _write_raw_subject(Path(tmp) / "sub01", with_mask=True)
            # _resolved_connectome_flags returns (run_fiber_count, run_sift2).
            # Legacy API default (connectome_metrics=None) preserves backward compatibility (True, True);
            # UI/CLI default (connectome_metrics="sift2") runs SIFT2 only: (False, True)
            cfg_api_legacy = ProductOrchestratorConfig(
                subject_id="sub01",
                subject_root=root,
                targets=("connectome",),
                tractography_backend="csd_act",
            )
            self.assertEqual(_resolved_connectome_flags(cfg_api_legacy, "csd_act"), (True, True))

            cfg_ui_default = ProductOrchestratorConfig(
                subject_id="sub01",
                subject_root=root,
                targets=("connectome",),
                tractography_backend="csd_act",
                connectome_metrics="sift2",
            )
            self.assertEqual(_resolved_connectome_flags(cfg_ui_default, "csd_act"), (False, True))
            self.assertEqual(_task_name_for_target("connectome", cfg_ui_default), "connectome.sift2_weighted")

            cfg_count = ProductOrchestratorConfig(
                subject_id="sub01",
                subject_root=root,
                targets=("connectome",),
                tractography_backend="csd_act",
                connectome_metrics="fiber_count",
            )
            self.assertEqual(_resolved_connectome_flags(cfg_count, "csd_act"), (True, False))
            self.assertEqual(_task_name_for_target("connectome", cfg_count), "connectome.streamline_count")

            cfg_both = ProductOrchestratorConfig(
                subject_id="sub01",
                subject_root=root,
                targets=("connectome",),
                tractography_backend="csd_act",
                connectome_metrics="sift2,fiber_count",
            )
            self.assertEqual(_resolved_connectome_flags(cfg_both, "csd_act"), (True, True))
            self.assertEqual(_task_name_for_target("connectome", cfg_both), "connectome.streamline_count")

            # Non-CSD backend (UKF) falls back cleanly to fiber_count since SIFT2 weights are CSD-only
            cfg_ukf = ProductOrchestratorConfig(
                subject_id="sub01",
                subject_root=root,
                targets=("connectome",),
                tractography_backend="ukf",
                connectome_metrics="sift2",
            )
            self.assertEqual(_resolved_connectome_flags(cfg_ukf, "ukf"), (True, False))

            plan_count = plan_subject_products(cfg_count, write=False)
            conn_item = next(item for item in plan_count.items if item.target == "connectome")
            self.assertEqual(conn_item.task_name, "connectome.streamline_count")
            self.assertTrue(str(conn_item.outputs["matrix"]).endswith("sub01-connectome-streamline_count.csv"))
            self.assertIsNone(conn_item.outputs["sift2_output_dir"])

            plan_sift2 = plan_subject_products(cfg_ui_default, write=False)
            conn_sift2 = next(item for item in plan_sift2.items if item.target == "connectome")
            self.assertEqual(conn_sift2.task_name, "connectome.sift2_weighted")
            self.assertTrue(str(conn_sift2.outputs["matrix"]).endswith("sub01-connectome-sift2_weighted_streamline_count.csv"))

    def test_convertible_nifti_and_nrrd_detection_and_cli_wiring(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            # Raw NIfTI folder (not yet inside input/data.nii.gz)
            nifti_dir = base / "raw_nifti_subj"
            nifti_dir.mkdir()
            (nifti_dir / "sub01_dwi.nii.gz").write_bytes(b"fake-nii")
            (nifti_dir / "sub01_dwi.bval").write_text("0 1000\n", encoding="utf-8")
            (nifti_dir / "sub01_dwi.bvec").write_text("0 1\n0 0\n0 0\n", encoding="utf-8")
            (nifti_dir / "sub01_mask.nii.gz").write_bytes(b"fake-mask")

            conv_nii = find_convertible_input(nifti_dir)
            self.assertIsNotNone(conv_nii)
            self.assertEqual(conv_nii[0], "nifti")
            self.assertIn("mask_path", conv_nii[3])
            inspected_nii = inspect_subject_directory(nifti_dir, "sub01")
            self.assertEqual(inspected_nii["mode"], "nifti")
            self.assertTrue(inspected_nii["has_mask"])

            # Raw NRRD file path
            nrrd_dir = base / "raw_nrrd_subj"
            nrrd_dir.mkdir()
            nrrd_file = nrrd_dir / "sub01_dwi.nrrd"
            nrrd_file.write_bytes(b"NRRD0004\n")
            conv_nrrd = find_convertible_input(nrrd_file)
            self.assertIsNotNone(conv_nrrd)
            self.assertEqual(conv_nrrd[0], "nrrd")
            self.assertEqual(conv_nrrd[2], nrrd_dir.resolve())
            inspected_nrrd = inspect_subject_directory(nrrd_file, "sub01")
            self.assertEqual(inspected_nrrd["mode"], "nrrd")

        # Verify RunSelection -> kwargs -> SlicerDSpaceLogic signatures & CLI parser roundtrip
        selection = RunSelection(
            subject_id="sub01",
            subject_root_text="/tmp/sub01",
            output_root_text="/tmp/out",
            selected_targets=("preprocessing", "connectome"),
            requested_targets=("preprocessing", "connectome"),
            run_name="Test1",
            tractography_backend="csd_act",
            bundle_backend="wmql",
            microstructure_freewater=False,
            microstructure_noddi=False,
            select_streamlines=1000000,
            ddseg_device="auto",
            surface_device="auto",
            connectome_backend="auto",
            connectome_compress_mrtrix_dense=None,
            topup=TopupParameters(),
            preprocessing_stages="brain_mask,denoise,eddy",
            connectome_metrics="fiber_count",
        )
        run_kwargs = build_single_subject_run_kwargs(selection, {}, preflight_blocked_targets={})
        self.assertEqual(run_kwargs["preprocessing_stages"], "brain_mask,denoise,eddy")
        self.assertEqual(run_kwargs["connectome_metrics"], "fiber_count")

        # Ensure SharedLogicMixin methods accept every key in build_single_subject_run_kwargs
        plan_sig = inspect.signature(SharedLogicMixin.planSelectedProductOrchestration)
        run_sig = inspect.signature(SharedLogicMixin.runSelectedProductOrchestration)
        env_sig = inspect.signature(SharedLogicMixin.runEnvironmentDependencyCheck)
        for key in run_kwargs:
            self.assertIn(key, plan_sig.parameters)
            self.assertIn(key, run_sig.parameters)
        self.assertIn("preprocessing_stages", env_sig.parameters)

        repo_root = Path(__file__).resolve().parents[2]
        cli_args = build_background_cli_args(run_kwargs, {}, repo_root=repo_root)
        self.assertIn("--preprocessing-stages", cli_args)
        self.assertIn("brain_mask,denoise,eddy", cli_args)
        self.assertIn("--connectome-metrics", cli_args)
        self.assertIn("fiber_count", cli_args)
        # Strip "--launch PythonSlicer <script>" prefix before argparse
        parsed = build_parser().parse_args(cli_args[3:])
        self.assertEqual(parsed.preprocessing_stages, "brain_mask,denoise,eddy")
        self.assertEqual(parsed.connectome_metrics, "fiber_count")

        batch_cli_args = build_batch_background_cli_args(
            run_kwargs,
            {},
            repo_root=repo_root,
            batch_root=Path("/tmp/batch"),
            batch_size=2,
            arrange_inputs="none",
        )
        self.assertIn("--preprocessing-stages", batch_cli_args)
        self.assertIn("--connectome-metrics", batch_cli_args)

        # Picker state update roundtrip
        fake_slicer = SimpleNamespace(util=SimpleNamespace(NodeModify=lambda _node: mock.MagicMock()))
        owner = SimpleNamespace(
            _parameterNode=SimpleNamespace(),
            _onboardingLanguage="en",
            _refresh_advanced_run_settings=mock.MagicMock(),
            _refresh_topup_controls=mock.MagicMock(),
            _schedule_common_user_settings_save=mock.MagicMock(),
        )
        with mock.patch.dict(sys.modules, {"slicer": fake_slicer}):
            set_algorithms(owner, "preprocessing", ["brain_mask", "eddy"])
            set_algorithms(owner, "connectome", ["fiber_count"])
        self.assertEqual(owner._parameterNode.orchestrationPreprocessingStages, "brain_mask,eddy")
        self.assertEqual(owner._parameterNode.orchestrationConnectomeMetrics, "fiber_count")

    def test_input_format_popover_supports_hover_and_click_to_pin_with_pythonqt_int_geometry(self) -> None:
        class _FakeSignal:
            def __init__(self) -> None:
                self._slots = []

            def connect(self, fn) -> None:
                self._slots.append(fn)

            def emit(self, *args) -> None:
                for fn in list(self._slots):
                    fn(*args)

        class _FakePythonQtWidget(_FakeWidget):
            def __init__(self, text: str = "") -> None:
                super().__init__(text)
                # PythonQt exposes width/height/sizeHint as properties, NOT callable methods!
                self.width = 22
                self.height = 22
                self.sizeHint = SimpleNamespace(width=340, height=160)
                self.visible = False
                self.styleSheet = ""
                self.properties: dict[str, object] = {}
                self.clicked = _FakeSignal()

            def show(self) -> None:
                self.visible = True

            def hide(self) -> None:
                was_visible = self.visible
                self.visible = False
                filt = getattr(self, "event_filter", None)
                if was_visible and filt is not None:
                    filt.eventFilter(self, SimpleNamespace(type=lambda: 18))

            def raise_(self) -> None:
                pass

            def move(self, x: int, y: int) -> None:
                self.pos = (x, y)

            def resize(self, w: int, h: int) -> None:
                self.width = w
                self.height = h

            def setWindowFlags(self, flags: int) -> None:
                self.window_flags = flags

            def setAttribute(self, *_args) -> None:
                pass

            def setWordWrap(self, flag: bool) -> None:
                self.wordWrap = flag

            def setFixedWidth(self, w: int) -> None:
                self.width = w

            def setStyleSheet(self, css: str) -> None:
                self.styleSheet = css

            def setProperty(self, key: str, value: object) -> None:
                self.properties[key] = value

            def addLayout(self, child) -> None:
                self.children.append(child)

        class _FakeQObject:
            def __init__(self, _parent=None) -> None:
                pass

        fake_qt = SimpleNamespace(
            QObject=_FakeQObject,
            QEvent=SimpleNamespace(Enter=10, Leave=11, Hide=18, Close=19, ToolTip=110),
            QPoint=lambda x, y: SimpleNamespace(x=x, y=y),
            QToolTip=mock.MagicMock(),
            QFrame=_FakePythonQtWidget,
            QLabel=_FakePythonQtWidget,
            QWidget=lambda: _FakePythonQtWidget(),
            QHBoxLayout=lambda _parent=None: _FakePythonQtWidget(),
            QVBoxLayout=lambda _parent=None: _FakePythonQtWidget(),
            QComboBox=_FakeComboBox,
            QToolButton=_FakePythonQtWidget,
            QSizePolicy=SimpleNamespace(Expanding=1, Fixed=0),
            Qt=SimpleNamespace(
                WhatsThisCursor=1,
                PointingHandCursor=2,
                NoFocus=0,
                AlignVCenter=4,
                ToolTip=8,
                Popup=16,
                FramelessWindowHint=32,
                WA_ShowWithoutActivating=1,
                WA_TransparentForMouseEvents=2,
                TextSelectableByMouse=4,
            ),
        )
        ui = SimpleNamespace(
            configurationFormLayout=mock.MagicMock(),
            inputDirectoryPath=_FakePythonQtWidget(),
            subjectIdLineEdit=_FakePythonQtWidget(),
            outputDirectoryPath=_FakePythonQtWidget(),
        )
        owner = SimpleNamespace(ui=ui, _parameterNode=SimpleNamespace(inputFormat="nifti"), _onboardingLanguage="en")

        with mock.patch.dict(sys.modules, {"qt": fake_qt}):
            install_input_format_selector(owner)
            hover_card = owner._inputFormatHoverPopover
            pinned_card = owner._inputFormatPinnedPopover
            close_btn = owner._inputFormatHelpCloseButton

            # 1. Hover shows hover_card (without close button); leaving auto-closes it without clicking
            owner._inputFormatHelpFilter.eventFilter(ui.inputFormatHelpButton, SimpleNamespace(type=lambda: 10))
            self.assertTrue(hover_card.visible)
            self.assertFalse(pinned_card.visible)
            owner._inputFormatHelpFilter.eventFilter(ui.inputFormatHelpButton, SimpleNamespace(type=lambda: 11))
            self.assertFalse(hover_card.visible)

            # 2. Clicking '?' pins pinned_card open; leaving '?' does NOT close pinned_card
            ui.inputFormatHelpButton.clicked.emit()
            self.assertTrue(pinned_card.visible)
            self.assertFalse(hover_card.visible)
            self.assertTrue(owner._inputFormatHelpState["pinned"])
            owner._inputFormatHelpFilter.eventFilter(ui.inputFormatHelpButton, SimpleNamespace(type=lambda: 11))
            self.assertTrue(pinned_card.visible)

            # 3. Clicking the '×' close button closes and unpins pinned_card
            close_btn.clicked.emit()
            self.assertFalse(pinned_card.visible)
            self.assertFalse(owner._inputFormatHelpState["pinned"])

            # 4. Re-pinning and clicking outside (Qt.Popup emits Hide event) auto-closes and unpins
            ui.inputFormatHelpButton.clicked.emit()
            self.assertTrue(pinned_card.visible)
            self.assertTrue(owner._inputFormatHelpState["pinned"])
            pinned_card.hide()
            self.assertFalse(pinned_card.visible)
            self.assertFalse(owner._inputFormatHelpState["pinned"])

    def test_products_tooltips_are_concise_and_uestc_acknowledgement_is_set(self) -> None:
        from shared.ui_compact import refresh_compact_algorithms

        ui = SimpleNamespace(
            targetPreprocessingCheckBox=_FakeWidget(),
            targetMicrostructureCheckBox=_FakeWidget(),
            targetTractographyCheckBox=_FakeWidget(),
            targetBundlesCheckBox=_FakeWidget(),
            targetConnectomeCheckBox=_FakeWidget(),
        )
        owner = SimpleNamespace(
            ui=ui,
            _parameterNode=SimpleNamespace(
                orchestrationPreprocessingStages="denoise,eddy,topup",
                orchestrationMicrostructureStages="dti,freewater,noddi",
                orchestrationTractographyBackend="csd_act",
                orchestrationBundleBackend="wmql",
                orchestrationConnectomeMetrics="sift2",
            ),
            _onboardingLanguage="en",
            _algorithmSummaryLabels={name: _FakeWidget() for name in vars(ui)},
            _algorithmButtons={},
        )
        refresh_compact_algorithms(owner, "en")
        self.assertIn("DTI, FW, NODDI, DKI / MK-Curve, iFW", ui.targetMicrostructureCheckBox.toolTip)
        self.assertIn("Selected: DTI + FW + NODDI", ui.targetMicrostructureCheckBox.toolTip)
        self.assertNotIn("not completed results or QC status", ui.targetMicrostructureCheckBox.toolTip)
        self.assertEqual(
            owner._algorithmSummaryLabels["targetMicrostructureCheckBox"].toolTip,
            ui.targetMicrostructureCheckBox.toolTip,
        )

        refresh_compact_algorithms(owner, "zh")
        self.assertIn("扩散微结构参数图", ui.targetMicrostructureCheckBox.toolTip)
        self.assertIn("当前已选：DTI + FW + NODDI", ui.targetMicrostructureCheckBox.toolTip)

        repo_root = Path(__file__).resolve().parents[2]
        for rel in ("SlicerDSpace.py", "CMakeLists.txt", "SlicerDSpace.s4ext"):
            text = (repo_root / rel).read_text(encoding="utf-8")
            self.assertNotIn("ShanghaiTech", text)
            self.assertIn("UESTC", text)
            self.assertIn("Fan's Lab", text)
            self.assertIn("Chengjin Li", text)

    def test_microstructure_dti_is_user_toggleable_and_auto_closed_when_needed(self) -> None:
        from shared.algorithm_selection import MICROSTRUCTURE
        from shared.target_closure import closure_notice
        from shared.ui_run_controller import ProductTargetFlags, microstructure_dti_enabled, selected_closure
        from shared.user_config import canonical_microstructure_stages, normalize_user_config, selected_microstructure_stages

        self.assertEqual(MICROSTRUCTURE, ("dti", "freewater", "noddi", "dki", "ifw"))
        self.assertEqual(selected_microstructure_stages("noddi"), ("noddi",))
        self.assertEqual(canonical_microstructure_stages("noddi"), ("dti", "noddi"))
        self.assertFalse(microstructure_dti_enabled("noddi"))
        self.assertTrue(microstructure_dti_enabled("dti,noddi"))
        self.assertEqual(
            normalize_user_config({"run_defaults": {"microstructure_stages": ["noddi"]}})["run_defaults"]["microstructure_stages"],
            ["noddi"],
        )

        fake_slicer = SimpleNamespace(util=SimpleNamespace(NodeModify=lambda _node: mock.MagicMock()))
        node = SimpleNamespace(
            orchestrationMicrostructureStages="dti",
            orchestrationRunFreeWater=False,
            orchestrationRunNoddi=False,
            orchestrationRunDki=False,
            orchestrationRunIfw=False,
        )
        owner = SimpleNamespace(
            _parameterNode=node,
            _refresh_advanced_run_settings=lambda: None,
            _schedule_common_user_settings_save=lambda: None,
        )
        with mock.patch.dict(sys.modules, {"slicer": fake_slicer}):
            set_algorithms(owner, "microstructure", ["noddi"])
        self.assertEqual(node.orchestrationMicrostructureStages, "noddi")
        self.assertTrue(node.orchestrationRunNoddi)
        self.assertEqual(compact_algorithm_labels(node)["targetMicrostructureCheckBox"], "NODDI")

        closure = selected_closure(
            ProductTargetFlags(microstructure=True),
            tractography_backend="csd_act",
            bundle_backend="wmql",
            microstructure_dti=False,
            microstructure_freewater=False,
            microstructure_noddi=True,
        )
        self.assertEqual(closure.requested, ("validation", "microstructure_noddi"))
        self.assertIn("microstructure", closure.targets)
        self.assertIn("microstructure_noddi", closure.targets)
        self.assertIn("DTI (needed by NODDI)", closure_notice(closure))

    def test_algorithm_picker_row_has_independent_checkbox_and_popup(self) -> None:
        from shared import ui_algorithm_picker

        focus_callbacks = []
        app_instance = SimpleNamespace(
            focusChanged=SimpleNamespace(
                connect=lambda cb: focus_callbacks.append(cb),
                disconnect=lambda cb: focus_callbacks.remove(cb) if cb in focus_callbacks else None,
            )
        )
        style_mock = SimpleNamespace(pixelMetric=lambda _m: 16)
        created_dialogs = []

        def _make_dialog(_parent=None):
            dlg = mock.MagicMock(name="QDialog")
            dlg.exec_.return_value = 0
            created_dialogs.append(dlg)
            return dlg

        _make_dialog.Accepted = 1

        created_boxes = []

        def _make_checkbox(label=""):
            cb = mock.MagicMock(name=f"QCheckBox({label})")
            cb.label_text = label
            cb.checked = False
            created_boxes.append(cb)
            return cb

        click_areas = []

        def make_click_area():
            button = mock.MagicMock()
            click_areas.append(button)
            return button

        fake_qt = SimpleNamespace(
            QApplication=SimpleNamespace(style=lambda: style_mock, instance=lambda: app_instance, primaryScreen=lambda: None),
            QStyle=SimpleNamespace(PM_IndicatorWidth=1, PM_CheckBoxLabelSpacing=2),
            QFormLayout=SimpleNamespace(FieldRole=1, LabelRole=0),
            QSizePolicy=SimpleNamespace(Fixed=0, Ignored=1, Preferred=2),
            Qt=SimpleNamespace(
                AlignVCenter=4,
                WA_TransparentForMouseEvents=51,
                PointingHandCursor=13,
                ClickFocus=2,
                NoFocus=0,
                Popup=8,
                ToolButtonIconOnly=0,
                ScrollBarAlwaysOff=1,
                FramelessWindowHint=2048,
            ),
            QSize=lambda *args: args,
            QFrame=SimpleNamespace(NoFrame=0),
            QScrollArea=lambda _parent: mock.MagicMock(name="QScrollArea"),
            QWidget=lambda: mock.MagicMock(name="QWidget"),
            QHBoxLayout=lambda _p=None: mock.MagicMock(name="QHBoxLayout"),
            QVBoxLayout=lambda _p=None: mock.MagicMock(name="QVBoxLayout"),
            QLabel=lambda *a: mock.MagicMock(name="QLabel", ctor_args=a),
            QPushButton=make_click_area,
            QToolButton=lambda: mock.MagicMock(name="QToolButton"),
            QDialog=_make_dialog,
            QDialogButtonBox=mock.MagicMock(Accepted=1, Apply=2, Cancel=4),
            QCheckBox=_make_checkbox,
            QTimer=lambda: mock.MagicMock(name="QTimer"),
        )
        boxes = {}
        for _, (name, _, _) in FAMILIES.items():
            cb = mock.MagicMock(name=name)
            cb.isEnabled.return_value = True
            boxes[name] = cb
        owner = SimpleNamespace(
            ui=SimpleNamespace(targetProductsFormLayout=mock.MagicMock(), **boxes),
            _parameterNode=SimpleNamespace(orchestrationMicrostructureStages="dti,noddi"),
            _onboardingLanguage="en",
        )
        fake_slicer = SimpleNamespace(util=SimpleNamespace(mainWindow=lambda: None))
        with mock.patch.dict(sys.modules, {"qt": fake_qt, "slicer": fake_slicer}), mock.patch.object(
            ui_algorithm_picker, "refresh_picker_icons"
        ):
            ui_algorithm_picker.install_algorithm_pickers(owner)
            for cb in boxes.values():
                cb.setFixedWidth.assert_called_once_with(18)
            owner._algorithmButtons["microstructure"].isVisible.return_value = False
            ui_algorithm_picker.choose_algorithms(owner, "microstructure")
            boxes["targetMicrostructureCheckBox"].click.assert_not_called()
            row = next(button for button in click_areas if button.objectName == "microstructureAlgorithmRow")
            on_click = row.clicked.connect.call_args.args[0]
            on_click(False)
            boxes["targetMicrostructureCheckBox"].click.assert_called_once_with()

        self.assertEqual(len(focus_callbacks), 0)
        self.assertEqual(len(created_dialogs), 1)
        created_dialogs[0].setWindowFlags.assert_called_once_with(8 | 2048)
        dti_boxes = [b for b in created_boxes if "DTI" in b.label_text]
        self.assertEqual(len(dti_boxes), 1)
        self.assertEqual(dti_boxes[0].label_text, "DTI")

    def test_compute_device_selector_surface_parcellation_and_dti_labels(self) -> None:
        from shared.algorithm_selection import LABELS
        from shared.ui_compact import install_compute_device_selector, refresh_compute_device_selector
        from shared.ui_onboarding import onboarding_text
        from shared.ui_workflow_text import WORKFLOW_TEXT

        self.assertEqual(LABELS["dti_det"], "DTI")
        self.assertEqual(WORKFLOW_TEXT["en"]["compact_postprocess"], "Cortical morphometry / parcellation")
        self.assertEqual(WORKFLOW_TEXT["zh"]["compact_postprocess"], "皮层形态学 / 分区")
        self.assertEqual(onboarding_text("en", "torch_device"), "Compute device")
        self.assertEqual(onboarding_text("zh", "torch_device"), "计算设备")

        combo = _FakeComboBox()
        combo.clear = lambda: combo.items.clear()
        combo.show = lambda: None
        label = _FakeWidget()
        label.show = lambda: None
        targets_layout = _FakeWidget()
        fake_qt = SimpleNamespace(
            QWidget=lambda: _FakeWidget(),
            QHBoxLayout=lambda _p=None: _FakeWidget(),
            QComboBox=SimpleNamespace(AdjustToMinimumContentsLengthWithIcon=0),
            QSizePolicy=SimpleNamespace(Expanding=1, Fixed=0),
            Qt=SimpleNamespace(AlignVCenter=4, PointingHandCursor=13),
        )
        owner = SimpleNamespace(
            ui=SimpleNamespace(
                unifiedTorchDeviceLabel=label,
                unifiedTorchDeviceComboBox=combo,
                targetsLayout=targets_layout,
                environmentFormLayout=None,
            ),
            _parameterNode=SimpleNamespace(device="cuda:1", ddsegDevice="cuda:1"),
            _onboardingLanguage="en",
        )
        with (
            mock.patch.dict(sys.modules, {"qt": fake_qt}),
            mock.patch("shared.ui_compact._detect_cuda_devices", return_value=(("cuda:0", "CUDA:0"), ("cuda:1", "CUDA:1"))),
        ):
            install_compute_device_selector(owner)
            self.assertIn(owner._computeDeviceRow, targets_layout.children)
            self.assertEqual(label.text, "Compute device")
            self.assertEqual([item[1] for item in combo.items], ["auto", "cpu", "cuda:0", "cuda:1"])
            self.assertEqual(combo.currentIndex, 3)
            refresh_compute_device_selector(owner, "zh")
            self.assertEqual(label.text, "计算设备")
            self.assertIn("自动", combo.items[0][0])

    def test_available_torch_skips_pytorch_extension_and_stale_slicer_paths_are_ignored(self) -> None:
        import types
        from shared.selected_setup import build_setup_plan
        from shared.ui_selected_setup import _compact_component_rows

        ready_report = SimpleNamespace(
            statuses=(
                SimpleNamespace(
                    key="python:torch",
                    kind="python",
                    label="torch",
                    status="available",
                    required=True,
                    targets=("surface",),
                    install_group="python",
                    install_hint="",
                    details={"import": "torch", "package": "torch>=2.0.0"},
                ),
                SimpleNamespace(
                    key="python:nibabel",
                    kind="python",
                    label="nibabel",
                    status="available",
                    required=True,
                    targets=("surface",),
                    install_group="python",
                    install_hint="",
                    details={"import": "nibabel", "package": "nibabel>=5.0"},
                ),
            ),
            targets=("validation", "surface"),
            blocker_count=0,
        )
        plan = build_setup_plan(ready_report, installed_extensions=(), loaded_extensions=[], need_torch=True)
        self.assertEqual(plan["extensions"], [])

        owner = SimpleNamespace(_onboardingLanguage="en")
        compact_rows = _compact_component_rows(
            owner,
            [{"title": "Python packages: nibabel>=5.0", "size_bytes": 0, "licence": {"id": "pypi_packages", "name": "MIT: nibabel", "url": "https://pypi.org"}}],
            {"pypi_packages"},
        )
        self.assertEqual(len(compact_rows), 1)
        self.assertNotIn("size not known", compact_rows[0])
        self.assertNotIn("no local file", compact_rows[0])

        from shared.ui_environment_panel import is_other_slicer_installation_path

        with tempfile.TemporaryDirectory() as tmp:
            s510 = Path(tmp) / "Slicer 5.10.0"
            s56 = Path(tmp) / "Slicer 5.6.2"
            (s510 / "bin").mkdir(parents=True)
            (s56 / "bin").mkdir(parents=True)
            (s510 / "bin" / "PythonSlicer.exe").write_text("", encoding="utf-8")
            (s56 / "bin" / "PythonSlicer.exe").write_text("", encoding="utf-8")
            self.assertTrue(is_other_slicer_installation_path(str(s56 / "bin" / "PythonSlicer.exe"), s510))
            self.assertFalse(is_other_slicer_installation_path(str(s510 / "bin" / "PythonSlicer.exe"), s510))

    def test_parent_cohort_subject_root_resolution_clean_cli_args_and_stage_indicator(self) -> None:
        from shared.orchestrator import ProductOrchestratorConfig, plan_subject_products
        from shared.pipeline import resolve_subject_root_dir
        from shared.run_subject_pipeline_cli import build_parser
        from shared.ui_run_controller import (
            RunSelection,
            build_background_cli_args,
            build_batch_background_cli_args,
            build_single_subject_run_kwargs,
        )
        from shared.ui_styles import (
            run_progress_bar_style,
            run_stage_count_label_style,
            run_stage_indicator_palette,
        )
        from shared.user_config import explicit_cli_destinations

        with tempfile.TemporaryDirectory() as tmp:
            cohort = Path(tmp) / "demo_sub"
            sub_input = cohort / "100307" / "input"
            sub_input.mkdir(parents=True)
            for fname in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (sub_input / fname).write_bytes(b"stub")

            resolved = resolve_subject_root_dir(cohort, "100307")
            self.assertEqual(resolved.resolve(), (cohort / "100307").resolve())
            self.assertEqual(resolve_subject_root_dir(cohort / "100307", "100307").resolve(), (cohort / "100307").resolve())

            sel = RunSelection(
                subject_id="100307",
                subject_root_text=str(cohort),
                output_root_text="",
                selected_targets=("validation", "surface"),
                requested_targets=("surface",),
                run_name="Test1",
                tractography_backend="csd_act",
                bundle_backend="wmql",
                microstructure_freewater=False,
                microstructure_noddi=False,
                select_streamlines=100000,
                ddseg_device="cpu",
                surface_device="cpu",
                connectome_backend="auto",
                connectome_compress_mrtrix_dense=False,
            )
            single_kwargs = build_single_subject_run_kwargs(sel, {}, preflight_blocked_targets={})
            self.assertEqual(Path(single_kwargs["subject_root"]).resolve(), (cohort / "100307").resolve())

            plan = plan_subject_products(
                ProductOrchestratorConfig(
                    subject_id="100307",
                    subject_root=cohort,
                    targets=("validation", "surface"),
                    surface_device="cpu",
                    ddseg_device="cpu",
                ),
                write=False,
            )
            by_target = {item.target: item for item in plan.items}
            self.assertNotIn("missing raw input", by_target["surface"].reason)

            repo_root = Path(__file__).resolve().parents[2]
            cli_args = build_background_cli_args(single_kwargs, {}, repo_root=repo_root)
            batch_args = build_batch_background_cli_args(
                single_kwargs, {}, repo_root=repo_root, batch_root=cohort, batch_size=1, arrange_inputs="none"
            )
            self.assertNotIn("", cli_args)
            self.assertNotIn("", batch_args)
            self.assertIn("--wmql-query-selection=", cli_args)
            self.assertIn("--wmql-query-selection=", batch_args)

            parser = build_parser()
            parsed = parser.parse_args(cli_args[3:])
            self.assertEqual(parsed.wmql_query_selection, "")
            self.assertIn("wmql_query_selection", explicit_cli_destinations(parser, cli_args[3:]))

        button_blues = {"#2f6b8a", "#335b70", "#3b7ea1", "#9eb8c8"}
        for dark in (False, True):
            for state in ("running", "preflight", "completed", "completed_with_blocks", "cancelled", "failed"):
                pal = run_stage_indicator_palette(state, dark=dark)
                self.assertNotIn(pal["chunk"].lower(), button_blues)
                self.assertNotIn(pal["icon_bg"].lower(), button_blues)
                bar_qss = run_progress_bar_style(state, dark=dark)
                self.assertIn("max-height: 6px;", bar_qss)
                self.assertIn(pal["chunk"], bar_qss)
                lbl_qss = run_stage_count_label_style(state, dark=dark)
                self.assertIn(pal["text"], lbl_qss)

    def test_cross_format_auto_detection_and_ui_sync(self) -> None:
        from shared.pipeline import (
            detected_input_format_from_mode,
            inspect_subject_directory,
            resolve_subject_root_dir,
        )
        from shared.ui_compact import sync_detected_input_format
        from shared.ui_input_defaults import subject_folder_defaults

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # 1. Canonical NIfTI (<root>/input, <parent>/<id>/input, <root>/input/data.nii.gz)
            canon_parent = root / "cohort_nifti"
            canon_sub = canon_parent / "100307"
            (canon_sub / "input").mkdir(parents=True)
            for fname in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (canon_sub / "input" / fname).write_bytes(b"stub")

            for candidate in (canon_parent, canon_sub, canon_sub / "input", canon_sub / "input" / "data.nii.gz"):
                inspected = inspect_subject_directory(candidate, "100307")
                self.assertEqual(detected_input_format_from_mode(inspected["mode"]), "nifti")
                self.assertEqual(resolve_subject_root_dir(candidate, "100307").resolve(), canon_sub.resolve())
            self.assertEqual(
                subject_folder_defaults(str(canon_sub / "input" / "data.nii.gz")),
                (str(canon_sub), "100307"),
            )

            # 2. Raw convertible NIfTI (folder + direct .nii.gz + parent/<id>)
            raw_parent = root / "cohort_raw_nifti"
            raw_sub = raw_parent / "sub02"
            raw_sub.mkdir(parents=True)
            (raw_sub / "dwi.nii.gz").write_bytes(b"stub")
            (raw_sub / "dwi.bval").write_text("0 1000\n", encoding="utf-8")
            (raw_sub / "dwi.bvec").write_text("0 1\n0 0\n0 0\n", encoding="utf-8")
            for candidate in (raw_parent, raw_sub, raw_sub / "dwi.nii.gz"):
                inspected = inspect_subject_directory(candidate, "sub02")
                self.assertEqual(inspected["mode"], "nifti")
                self.assertEqual(detected_input_format_from_mode(inspected["mode"]), "nifti")

            # 3. NRRD (folder, parent/<id>, and direct .nrrd with .nii.gz mask sidecar)
            nrrd_parent = root / "cohort_nrrd"
            nrrd_sub = nrrd_parent / "sub03"
            nrrd_sub.mkdir(parents=True)
            (nrrd_sub / "scan.nrrd").write_text("NRRD0005\n", encoding="utf-8")
            (nrrd_sub / "nodif_brain_mask.nii.gz").write_bytes(b"stub")
            for candidate in (nrrd_parent, nrrd_sub, nrrd_sub / "scan.nrrd"):
                inspected = inspect_subject_directory(candidate, "sub03")
                self.assertEqual(inspected["mode"], "nrrd")
                self.assertTrue(inspected["has_mask"])
                self.assertEqual(detected_input_format_from_mode(inspected["mode"]), "nrrd")

            # 4. DICOM (folder, parent/<id>, and direct .dcm file)
            dicom_parent = root / "cohort_dicom"
            dicom_sub = dicom_parent / "sub04"
            dicom_sub.mkdir(parents=True)
            dcm_file = dicom_sub / "IM0001.dcm"
            dcm_file.write_bytes(b"\0" * 128 + b"DICM")
            for candidate in (dicom_parent, dicom_sub, dcm_file):
                inspected = inspect_subject_directory(candidate, "sub04")
                self.assertEqual(inspected["mode"], "dicom")
                self.assertEqual(detected_input_format_from_mode(inspected["mode"]), "dicom")
                self.assertEqual(resolve_subject_root_dir(candidate, "sub04").resolve(), dicom_sub.resolve())
            self.assertEqual(subject_folder_defaults(str(dcm_file)), (str(dicom_sub), "sub04"))

            # 5. UI sync: auto-switches NRRD -> NIfTI on path signature change, preserves manual dropdown
            # switch while path is unchanged (so user can browse), and force-syncs on Run click.
            class _FakeCombo:
                def __init__(self):
                    self.items = [("NIfTI", "nifti"), ("DICOM", "dicom"), ("NRRD", "nrrd")]
                    self.currentIndex = 2
                    self._blocked = False

                def findData(self, value):
                    for idx, (_, data) in enumerate(self.items):
                        if data == value:
                            return idx
                    return -1

                def blockSignals(self, flag):
                    prev = self._blocked
                    self._blocked = flag
                    return prev

                def setCurrentIndex(self, idx):
                    self.currentIndex = idx

            combo = _FakeCombo()
            node = SimpleNamespace(inputFormat="nrrd")
            path_widget = SimpleNamespace(filters=0, nameFilters=["old"], placeholderText="", setPlaceholderText=lambda t: setattr(path_widget, "placeholderText", t))
            owner = SimpleNamespace(
                ui=SimpleNamespace(inputFormatComboBox=combo, inputDirectoryPath=path_widget),
                _inputFormatComboBox=combo,
                _parameterNode=node,
                _onboardingLanguage="en",
            )
            # Path signature changes while user had NRRD selected -> auto-syncs to NIfTI
            sig1 = (str(canon_parent), "100307")
            self.assertEqual(sync_detected_input_format(owner, "subject_root", signature=sig1), "nifti")
            self.assertEqual(node.inputFormat, "nifti")
            self.assertEqual(combo.currentIndex, 0)

            # User manually switches dropdown to NRRD to browse; same signature does NOT snap it back
            node.inputFormat = "nrrd"
            combo.currentIndex = 2
            self.assertEqual(sync_detected_input_format(owner, "subject_root", signature=sig1), "nifti")
            self.assertEqual(node.inputFormat, "nrrd")
            self.assertEqual(combo.currentIndex, 2)

            # Clicking Run (force=True) re-syncs dropdown to actual detected format (NIfTI)
            self.assertEqual(sync_detected_input_format(owner, "subject_root", signature=sig1, force=True), "nifti")
            self.assertEqual(node.inputFormat, "nifti")
            self.assertEqual(combo.currentIndex, 0)

            # Selecting a DICOM path next auto-syncs to DICOM
            sig2 = (str(dicom_sub), "sub04")
            self.assertEqual(sync_detected_input_format(owner, "dicom", signature=sig2), "dicom")
            self.assertEqual(node.inputFormat, "dicom")
            self.assertEqual(combo.currentIndex, 1)


if __name__ == "__main__":
    unittest.main()
