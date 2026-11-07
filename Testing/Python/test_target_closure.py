#!/usr/bin/env python3
"""Automatic prerequisite closure (C1-C8): one ticked target runs every stage it needs.

All tests are pure or mocked: no numpy, nibabel or Slicer is needed.
"""

from __future__ import annotations

import contextlib
import io
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

from shared.target_closure import TARGET_ORDER, closure_notice, expand_closure  # noqa: E402


def _ordered(names):
    return tuple(name for name in TARGET_ORDER if name in set(names))


# (requested targets, keyword arguments, expected targets, expected required_by, expected algorithms)
C1_TABLE = [
    (("surface",), {}, ("surface",), {}, ("csd_act",)),
    (("segmentation",), {}, ("segmentation",), {}, ("csd_act",)),
    (("parcellation",), {}, ("parcellation",), {}, ("csd_act",)),
    (("microstructure",), {}, ("microstructure",), {}, ("csd_act",)),
    (("microstructure_noddi",), {}, ("microstructure", "microstructure_noddi"),
     {"microstructure": ("microstructure_noddi",)}, ("csd_act",)),
    (("tractography",), {}, ("segmentation", "parcellation", "tractography"),
     {"segmentation": ("tractography",), "parcellation": ("tractography",)}, ("csd_act",)),
    (("tractography",), {"tractography_algorithms": ("dti_det",)}, ("tractography",), {}, ("dti_det",)),
    (("tractography",), {"tractography_algorithms": ("ukf",)}, ("tractography",), {}, ("ukf",)),
    (("bundles",), {}, ("segmentation", "parcellation", "tractography", "bundles"),
     {"segmentation": ("tractography",), "parcellation": ("tractography", "bundles"), "tractography": ("bundles",)},
     ("csd_act",)),
    (("connectome",), {}, ("segmentation", "parcellation", "tractography", "connectome"),
     {"segmentation": ("tractography",), "parcellation": ("tractography", "connectome"),
      "tractography": ("connectome",)}, ("csd_act",)),
    (("tractcloud",), {}, ("segmentation", "parcellation", "tractography", "tractcloud"),
     {"segmentation": ("tractography",), "parcellation": ("tractography",), "tractography": ("tractcloud",)},
     ("csd_act",)),
    (("wma",), {}, ("tractography", "wma"), {"tractography": ("wma",)}, ("ukf",)),
    (("surface_postprocess",), {}, ("surface", "surface_postprocess"), {"surface": ("surface_postprocess",)},
     ("csd_act",)),
    (("saf",), {}, ("segmentation", "parcellation", "surface", "surface_postprocess", "tractography", "saf"),
     {"segmentation": ("tractography",), "parcellation": ("tractography",), "surface": ("surface_postprocess",),
      "surface_postprocess": ("saf",), "tractography": ("saf",)}, ("csd_act",)),
]


class C1ClosureRulesTest(unittest.TestCase):
    def test_every_single_target_expands_to_its_closure(self) -> None:
        for requested, kwargs, targets, required_by, algorithms in C1_TABLE:
            with self.subTest(requested=requested, kwargs=kwargs):
                closure = expand_closure(requested, **kwargs)
                self.assertEqual(closure.targets, targets)
                self.assertEqual(closure.targets, _ordered(closure.targets), "TARGET_ORDER")
                self.assertEqual(closure.required_by, required_by)
                self.assertEqual(closure.tractography_algorithms, algorithms)
                self.assertEqual(set(closure.requested) & set(closure.required_by), set())

    def test_closure_is_idempotent_and_preprocessing_is_never_added(self) -> None:
        for requested, kwargs, targets, _required_by, algorithms in C1_TABLE:
            with self.subTest(requested=requested):
                again = expand_closure(targets, **{**kwargs, "tractography_algorithms": algorithms})
                self.assertEqual(again.targets, targets)
                self.assertNotIn("preprocessing", targets)
        self.assertEqual(expand_closure(("preprocessing",)).targets, ("preprocessing",))

    def test_orchestrator_entry_point_applies_the_same_closure(self) -> None:
        from shared.orchestrator import normalize_orchestrator_targets
        self.assertEqual(normalize_orchestrator_targets(("connectome",)),
                         ("segmentation", "parcellation", "tractography", "connectome"))
        self.assertEqual(normalize_orchestrator_targets(("tractography",), tractography_backend="dti_det"),
                         ("tractography",))

    def test_wma_and_a_provided_tractogram(self) -> None:
        self.assertEqual(expand_closure(("wma",), tractography_algorithms=("csd_act",)).tractography_algorithms, ("ukf",))
        self.assertEqual(expand_closure(("wma", "bundles")).tractography_algorithms, ("csd_act", "ukf"))
        provided = expand_closure(("wma",), provided_tractography=("wma",))
        self.assertEqual(provided.targets, ("wma",))

    def test_skipped_target_is_not_added_back_by_the_orchestrator(self) -> None:
        from shared.orchestrator import ProductOrchestratorConfig, _with_closure
        config = _with_closure(ProductOrchestratorConfig(
            subject_id="s", subject_root=Path("/none"), targets=("connectome",), excluded_targets=("segmentation",)))
        self.assertNotIn("segmentation", config.targets)
        self.assertIn("parcellation", config.targets)


class C2ConsumersTest(unittest.TestCase):
    def test_plan_for_connectome_alone_has_no_select_x_blocked_item(self) -> None:
        from shared.orchestrator import ProductOrchestratorConfig, plan_subject_products
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "s"
            (root / "input").mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (root / "input" / name).write_text("x")
            config = ProductOrchestratorConfig(subject_id="s", subject_root=root, targets=("connectome",),
                                               atlas_t2=root / "input" / "data.nii.gz", execute=True)
            plan = plan_subject_products(config, write=False)
            targets = [item.target for item in plan.items]
            self.assertEqual(targets, ["segmentation", "parcellation", "tractography", "connectome"])
            for item in plan.items:
                self.assertNotIn("select parcellation", item.reason)
                self.assertNotIn("select segmentation", item.reason)
                self.assertEqual(item.status, "will_run", item.reason)

    def test_blocked_safety_net_remains_when_the_user_skips_a_prerequisite(self) -> None:
        from shared.orchestrator import ProductOrchestratorConfig, plan_subject_products
        with tempfile.TemporaryDirectory() as tmp:
            config = ProductOrchestratorConfig(subject_id="s", subject_root=Path(tmp) / "s", targets=("connectome",),
                                               excluded_targets=("parcellation",), execute=True)
            items = {item.target: item for item in plan_subject_products(config, write=False).items}
            self.assertNotIn("parcellation", items)
            self.assertEqual(items["connectome"].status, "blocked")
            self.assertIn("select parcellation", items["connectome"].reason)

    def test_cohort_planner_reserves_the_prerequisite_stages(self) -> None:
        from shared.cohort_targets import selected_stages
        stages = selected_stages(["--targets", "connectome"])
        self.assertTrue({"segmentation", "parcellation", "csd_act", "connectome"} <= stages, stages)
        self.assertNotIn("surface", stages)
        self.assertEqual(selected_stages(["--targets", "wma"]) & {"ukf", "wma", "csd_act"}, {"ukf", "wma"})

    def test_cli_resolves_targets_through_the_closure(self) -> None:
        from shared.run_subject_pipeline_cli import _resolve_targets, build_parser
        args, _ = build_parser().parse_known_args(["--subject-root", "/none", "--targets", "tractography"])
        self.assertEqual(_resolve_targets(args), ("segmentation", "parcellation", "tractography"))
        args, _ = build_parser().parse_known_args(
            ["--subject-root", "/none", "--targets", "tractography", "--skip-targets", "segmentation"])
        self.assertEqual(_resolve_targets(args), ("parcellation", "tractography"))

    def test_gui_target_selection_and_setup_plan_follow_the_closure(self) -> None:
        from shared.environment import EnvironmentDoctorConfig, run_environment_doctor
        from shared.ui_run_controller import ProductTargetFlags, selected_product_targets
        import shared.ui_selected_setup as ui

        targets = selected_product_targets(
            ProductTargetFlags(tractography=True), tractography_backend="csd_act", bundle_backend="wmql",
            microstructure_freewater=False, microstructure_noddi=False)
        self.assertEqual(targets, ("validation", "segmentation", "parcellation", "tractography"))
        with tempfile.TemporaryDirectory() as tmp:
            report = run_environment_doctor(EnvironmentDoctorConfig(
                targets=targets, repo_root=REPO_ROOT, model_assets_dir=Path(tmp), tractography_backend="csd_act",
                loaded_slicer_modules=()))
        # The result of a stale caller holds only the ticked target: the plan still follows the closure.
        stale = SimpleNamespace(statuses=report.statuses, targets=("tractography",), blocker_count=report.blocker_count)
        owner = mock.Mock()
        owner._selected_closure.return_value = expand_closure(("tractography",))
        owner._parameterNode = SimpleNamespace(brainMaskBackend="provided")
        owner._tractography_backend_text.return_value = "csd_act"
        owner._environment_dependency_path_kwargs.return_value = {}
        owner._closure_notice_text.return_value = "Also runs automatically: DDSeg, DDParcel (needed by CSD/ACT tractography)"
        manager = mock.Mock()
        manager.isExtensionInstalled.return_value = False
        fake_slicer = SimpleNamespace(app=SimpleNamespace(extensionsManagerModel=lambda: manager),
                                      modules=SimpleNamespace())
        with mock.patch.dict("sys.modules", {"slicer": fake_slicer}):
            plan = ui.selected_setup_plan(owner, stale)
        self.assertIn("PyTorch", plan["extensions"])
        self.assertEqual(plan["asset_profiles"], ["segmentation", "parcellation"])
        self.assertNotIn("UKFTractography", plan["extensions"])


class N3SetupPlanFallbackDoesNotExpandTest(unittest.TestCase):
    def test_n3_fallback_with_the_doctor_shape_adds_no_segmentation_parcellation_or_torch(self) -> None:
        owner = mock.Mock()
        owner._selected_closure.return_value = None
        owner._parameterNode = SimpleNamespace(brainMaskBackend="provided")
        owner._environment_dependency_path_kwargs.return_value = {}
        owner._closure_notice_text.return_value = ""
        result = SimpleNamespace(statuses=(), targets=("validation", "tractography", "wma"), blocker_count=0)
        plan = _plan_for_setup(owner, result)
        self.assertNotIn("PyTorch", plan["extensions"])
        self.assertEqual(plan["asset_profiles"], [])
        self.assertEqual(plan["closure_targets"], ["validation", "tractography", "wma"])

    def test_n3_ukf_fallback_with_a_missing_required_ukf_item_keeps_ukf_in_the_plan(self) -> None:
        from shared.environment import DependencyStatus
        owner = mock.Mock()
        owner._selected_closure.return_value = None
        owner._parameterNode = SimpleNamespace(brainMaskBackend="provided")
        owner._environment_dependency_path_kwargs.return_value = {}
        owner._closure_notice_text.return_value = ""
        item = DependencyStatus(key="command:UKFTractography", kind="command", label="UKFTractography",
                                status="missing", required=True, targets=("tractography",), install_group="ukf")
        result = SimpleNamespace(statuses=(item,), targets=("validation", "tractography", "wma"), blocker_count=1)
        plan = _plan_for_setup(owner, result)
        self.assertIn("UKFTractography", plan["extensions"])  # fails at fcb94b5: fallback fixed "csd_act"
        self.assertNotIn("PyTorch", plan["extensions"])
        self.assertEqual(plan["asset_profiles"], [])
        self.assertEqual(plan["closure_targets"], ["validation", "tractography", "wma"])
        self.assertEqual([entry["key"] for entry in plan["manual_items"]], [])
        # Guard: no UKF item in the doctor result -> UKF stays out of the fallback plan.
        plan = _plan_for_setup(owner, SimpleNamespace(statuses=(), targets=result.targets, blocker_count=0))
        self.assertNotIn("UKFTractography", plan["extensions"])


class C2GuiPathKeepsProvenanceTest(unittest.TestCase):
    def test_gui_bundles_all_selection_gives_the_cli_the_user_backend_and_keeps_ukf_variant_minimal(self) -> None:
        from shared import ui_run_controller as controller
        from shared.multi_tracking import variants
        from shared.orchestrator import ProductOrchestratorConfig, _with_closure
        from shared.run_subject_pipeline_cli import build_parser, resolve_analysis_options

        flags = controller.ProductTargetFlags(bundles=True)
        closure = controller.selected_closure(
            flags, tractography_backend="csd_act", bundle_backend="all",
            microstructure_freewater=False, microstructure_noddi=False)
        self.assertEqual(closure.algorithms_text, "csd_act,ukf")
        selection = controller.RunSelection(
            subject_id="s", subject_root_text="/none/s", output_root_text="", selected_targets=closure.targets,
            run_name="r", tractography_backend="csd_act", bundle_backend="all", microstructure_freewater=False,
            microstructure_noddi=False, select_streamlines=1000, ddseg_device="cpu", surface_device="cpu",
            connectome_backend="mrtrix", connectome_compress_mrtrix_dense=False,
            requested_targets=closure.requested, closure_tractography_backend=closure.algorithms_text)
        kwargs = controller.build_single_subject_run_kwargs(selection, {}, preflight_blocked_targets={})
        self.assertEqual(kwargs["tractography_backend"], "csd_act")
        self.assertIn("ukf_parameters", kwargs)  # UKF options still ship: the closure runs UKF
        argv = controller.build_background_cli_args(kwargs, {}, repo_root=REPO_ROOT)
        self.assertIn("--ukf-parameters", argv)
        args = build_parser().parse_args(argv[3:])
        resolve_analysis_options(args)
        config = _with_closure(ProductOrchestratorConfig(
            subject_id="s", subject_root=Path("/none/s"), targets=("bundles",), bundle_backend="all",
            tractography_backend=args.tractography_backend,
            closure_algorithm_requesters=dict(args.closure_algorithm_requesters)))
        by_backend = {child.tractography_backend: child.targets for child in variants(config)}
        self.assertEqual(by_backend["ukf"], ("tractography", "wma"))


class C3ProvenanceTest(unittest.TestCase):
    def test_plan_items_and_manifest_tasks_record_requested_and_required_by(self) -> None:
        from shared.orchestrator import ProductOrchestratorConfig, run_subject_products
        with tempfile.TemporaryDirectory() as tmp:
            result = run_subject_products(ProductOrchestratorConfig(
                subject_id="s", subject_root=Path(tmp) / "s", targets=("surface_postprocess",), execute=False))
            plan = json.loads(result.run_plan_path.read_text(encoding="utf-8"))
            manifest = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
        by_target = {item["target"]: item for item in plan["items"]}
        self.assertEqual((by_target["surface"]["requested"], by_target["surface"]["required_by"]),
                         (False, ["surface_postprocess"]))
        self.assertEqual((by_target["surface_postprocess"]["requested"], by_target["surface_postprocess"]["required_by"]),
                         (True, []))
        tasks = {task["name"]: task for task in manifest["tasks"]}
        self.assertEqual((tasks["surface.ddsurfer"]["requested"], tasks["surface.ddsurfer"]["required_by"]),
                         (False, ["surface_postprocess"]))
        self.assertEqual(manifest["closure"]["required_by"], {"surface": ["surface_postprocess"]})


class C4NoticeTest(unittest.TestCase):
    def test_notice_names_the_added_targets_and_their_requesters(self) -> None:
        self.assertEqual(closure_notice(expand_closure(("tractography",))),
                         "Also runs automatically: DDSeg, DDParcel (needed by CSD/ACT tractography)")
        self.assertEqual(closure_notice(expand_closure(("surface_postprocess",))),
                         "Also runs automatically: DDSurfer (needed by Surface postprocess)")

    def test_notice_is_empty_when_nothing_is_added(self) -> None:
        self.assertEqual(closure_notice(expand_closure(("surface",))), "")
        self.assertEqual(closure_notice(expand_closure(("tractography",), tractography_algorithms=("dti_det",))), "")

    def test_run_log_gets_one_line_at_the_start_of_a_real_run(self) -> None:
        from shared import orchestrator
        lines: list[str] = []
        config = orchestrator.ProductOrchestratorConfig(
            subject_id="s", subject_root=Path("/none"), targets=("tractography",), execute=True)
        with mock.patch.object(orchestrator, "_run_subject_products_impl", return_value="done") as impl:
            orchestrator.run_subject_products(config, log=lines.append)
        impl.assert_called_once()
        self.assertEqual([line for line in lines if line.startswith("[Targets]")],
                         ["[Targets] Also runs automatically: DDSeg, DDParcel (needed by CSD/ACT tractography)"])

    def test_plan_only_run_keeps_the_log_quiet(self) -> None:
        from shared import orchestrator
        lines: list[str] = []
        config = orchestrator.ProductOrchestratorConfig(
            subject_id="s", subject_root=Path("/none"), targets=("tractography",), execute=False)
        with mock.patch.object(orchestrator, "_run_subject_products_impl", return_value="done"):
            orchestrator.run_subject_products(config, log=lines.append)
        self.assertEqual(lines, [])

    def test_setup_plan_dialog_shows_the_same_text(self) -> None:
        import shared.ui_selected_setup as ui
        from shared.selected_setup import build_setup_plan
        plan = build_setup_plan(SimpleNamespace(statuses=(), targets=("tractography",), blocker_count=0),
                                system="Linux", machine="x86_64")
        plan["also_runs"] = "Also runs automatically: DDSeg, DDParcel (needed by CSD/ACT tractography)"
        qt = mock.MagicMock()
        fake_slicer = SimpleNamespace(util=SimpleNamespace(mainWindow=lambda: None))
        with mock.patch.dict("sys.modules", {"qt": qt, "slicer": fake_slicer}):
            ui._confirm(SimpleNamespace(_onboardingLanguage="en"), plan)
        texts = [call.args[0] for call in qt.QLabel.call_args_list if call.args]
        self.assertTrue(any(plan["also_runs"] in text for text in texts), texts)


class C5BrainMaskRuntimeTest(unittest.TestCase):
    NOT_AVAILABLE = "Automatic brain mask is not available on this system yet. Provide a brain mask file (NIfTI)."

    def _missing_mask_report(self):
        item = SimpleNamespace(key="brain_mask_backend", label="Brain mask", install_group="brain_mask", kind="setting",
                               targets=("tractography",), required=True, status="missing",
                               install_hint="Install the mask runtime", details={})
        return SimpleNamespace(statuses=(item,), targets=("tractography",), blocker_count=1)

    def test_linux_x86_64_installs_the_runtime_macos_arm64_is_unsupported_with_clear_text(self) -> None:
        from shared.selected_setup import build_setup_plan
        linux = build_setup_plan(self._missing_mask_report(), system="Linux", machine="x86_64")
        self.assertEqual(linux["managed_profiles"], ["pnl_cnn"])
        self.assertEqual(linux["unsupported_profiles"], [])
        self.assertEqual(linux["unsupported_notes"], {})
        # SC-1c: on macOS the auto route is native: no managed profile, and nothing is reported unsupported.
        native = build_setup_plan(self._missing_mask_report(), system="Darwin", machine="arm64")
        self.assertEqual((native["managed_profiles"], native["unsupported_profiles"]), ([], []))
        mac = build_setup_plan(self._missing_mask_report(), system="Darwin", machine="arm64", pnl_mask_route="managed")
        self.assertEqual(mac["managed_profiles"], [])
        self.assertEqual(mac["unsupported_profiles"], ["pnl_cnn"])
        self.assertEqual(mac["unsupported_notes"], {"pnl_cnn": self.NOT_AVAILABLE})
        self.assertEqual([item["hint"] for item in mac["manual_items"]], [self.NOT_AVAILABLE])

    def test_mask_runtime_is_required_for_a_mask_target_without_a_user_mask_even_with_no_input_folder(self) -> None:
        from shared.ui_run_state import gui_brain_mask_required
        base = dict(input_known=False, input_mask_exists=False, preprocessing_on=False, mask_backend="pnl_cnn")
        self.assertTrue(gui_brain_mask_required(("validation", "tractography"), **base))
        self.assertTrue(gui_brain_mask_required(("microstructure",), **base))
        self.assertFalse(gui_brain_mask_required(("validation",), **base))
        self.assertFalse(gui_brain_mask_required(("tractography",), **{**base, "mask_backend": "provided"}))
        self.assertFalse(gui_brain_mask_required(
            ("tractography",), **{**base, "input_known": True, "input_mask_exists": True}))
        self.assertTrue(gui_brain_mask_required(
            ("tractography",), **{**base, "input_known": True, "input_mask_exists": False, "mask_backend": "provided"}))


class C6SubjectIdTest(unittest.TestCase):
    def test_canonical_layout_uses_the_subject_root_folder_name(self) -> None:
        from shared.subject_id import derive_subject_id
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sub-0042"
            (root / "input").mkdir(parents=True)
            self.assertEqual(derive_subject_id(str(root)), "sub-0042")

    def test_dicom_folder_uses_the_folder_name(self) -> None:
        from shared.subject_id import derive_subject_id
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "DWI_DICOM"
            folder.mkdir()
            (folder / "IM0001.dcm").write_text("x")
            self.assertEqual(derive_subject_id(str(folder)), "DWI_DICOM")

    def test_bare_file_uses_the_name_without_nifti_suffix(self) -> None:
        from shared.subject_id import derive_subject_id
        with tempfile.TemporaryDirectory() as tmp:
            for name, expected in (("dwi.nii.gz", "dwi"), ("case7.nii", "case7")):
                path = Path(tmp) / name
                path.write_text("x")
                self.assertEqual(derive_subject_id(str(path)), expected)

    def test_unsafe_characters_become_underscores(self) -> None:
        from shared.subject_id import derive_subject_id
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "Müller 01 (a)"
            folder.mkdir()
            self.assertEqual(derive_subject_id(str(folder)), "M_ller_01__a_")
            self.assertEqual(derive_subject_id(str(Path(tmp) / "a.b-c_d")), "a.b-c_d")

    def test_q6_dot_names_after_the_suffix_strip_give_an_empty_id(self) -> None:
        from shared.subject_id import derive_subject_id
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("..nii.gz", "...nii", ".nii.gz"):
                path = Path(tmp) / name
                path.write_text("x")
                with self.subTest(name=name):
                    self.assertNotIn(derive_subject_id(str(path)), (".", ".."))
            self.assertEqual(derive_subject_id(str(Path(tmp) / "..nii.gz")), "")
            self.assertEqual(derive_subject_id(str(Path(tmp) / "...nii")), "")

    def test_typed_id_wins_and_empty_input_gives_empty_id(self) -> None:
        from shared.subject_id import derive_subject_id, effective_subject_id
        self.assertEqual(effective_subject_id(" typed ", "/x/other"), "typed")
        self.assertEqual(effective_subject_id("", "/x/other"), "other")
        self.assertEqual(derive_subject_id(""), "")

    def test_run_button_does_not_need_a_typed_id(self) -> None:
        from shared.ui_run_state import run_button_state
        state = run_button_state(selected_targets=("validation", "surface"), subject_id="", subject_root_text="/x/sub",
                                 batch_mode=False, batch_root_text="", pipeline_active=False)
        self.assertTrue(state.enabled)
        self.assertNotIn("Subject ID", state.tooltip)


class C7MaskFailureTest(unittest.TestCase):
    EXPECTED = ("Automatic brain mask failed: boom. Provide a brain mask file (NIfTI) "
                "or run Prepare to install the mask runtime.")

    def test_message_text(self) -> None:
        from shared.orchestrator import mask_failure_message
        self.assertEqual(mask_failure_message("boom"), self.EXPECTED)

    def test_blocked_reason_uses_the_message_when_the_backend_is_unavailable(self) -> None:
        from shared.orchestrator import ProductOrchestratorConfig, plan_subject_products
        unavailable = {"available": False, "missing_required": ["pnl python"], "missing_optional": [], "backend": "pnl_cnn"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "s"
            (root / "input").mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs"):
                (root / "input" / name).write_text("x")
            config = ProductOrchestratorConfig(subject_id="s", subject_root=root, targets=("microstructure",),
                                               execute=True, brain_mask_backend="pnl_cnn")
            with mock.patch("shared.brain_mask.inspect_brain_mask_backend", return_value=unavailable):
                items = {item.target: item for item in plan_subject_products(config, write=False).items}
        reason = items["microstructure"].reason
        self.assertTrue(reason.startswith("Automatic brain mask failed: "), reason)
        self.assertTrue(reason.endswith("Provide a brain mask file (NIfTI) or run Prepare to install the mask runtime."),
                        reason)


class C8RegionalStatisticsTest(unittest.TestCase):
    def test_microstructure_does_not_add_parcellation(self) -> None:
        for requested in (("microstructure",), ("microstructure_noddi",), ("microstructure_dki",)):
            self.assertNotIn("parcellation", expand_closure(requested).targets)

    def test_skip_is_logged_and_recorded_without_parcellation(self) -> None:
        from shared.orchestrator import regional_statistics_note
        lines: list[str] = []
        self.assertEqual(regional_statistics_note(None, lines.append), "skipped_no_parcellation")
        self.assertEqual(lines, ["[Microstructure] Regional statistics skipped: parcellation was not selected"])
        lines.clear()
        self.assertIsNone(regional_statistics_note(Path("/x/parc.nii.gz"), lines.append))
        self.assertEqual(lines, [])

    def test_manifest_task_entry_carries_the_skip_record(self) -> None:
        from shared.orchestrator import (ProductOrchestratorConfig, ProductTaskRecord, _write_orchestration_checkpoint,
                                         plan_subject_products)
        with tempfile.TemporaryDirectory() as tmp:
            config = ProductOrchestratorConfig(subject_id="s", subject_root=Path(tmp) / "s", targets=("microstructure",))
            plan = plan_subject_products(config, write=False)
            record = ProductTaskRecord(name="microstructure.dti", status="completed",
                                       regional_statistics="skipped_no_parcellation")
            path = Path(tmp) / "manifest.json"
            _write_orchestration_checkpoint(path=path, config=config, plan=plan, targets=plan.targets, tasks=[record],
                                            previous_run_records={})
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["tasks"][0]["regional_statistics"], "skipped_no_parcellation")


# ---------------------------------------------------------------------------------------------
# Fix round (K1-K6, Q1-Q7)
# ---------------------------------------------------------------------------------------------

def _widget_method(name, **namespace):
    """Exercise one SharedWidgetMixin method in isolation (the module needs Slicer and Qt)."""
    import ast
    path = REPO_ROOT / "shared" / "module_widget.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def _plan_for_setup(owner, report):
    import shared.ui_selected_setup as ui
    manager = mock.Mock()
    manager.isExtensionInstalled.return_value = False
    fake_slicer = SimpleNamespace(app=SimpleNamespace(extensionsManagerModel=lambda: manager), modules=SimpleNamespace())
    with mock.patch.dict("sys.modules", {"slicer": fake_slicer}):
        return ui.selected_setup_plan(owner, report)


def _fresh_subject(tmp: str) -> Path:
    root = Path(tmp) / "s"
    (root / "input").mkdir(parents=True)
    for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
        (root / "input" / name).write_text("x")
    return root


class K1SetupPlanUsesTheOneClosureTest(unittest.TestCase):
    def test_k1_wma_alone_with_csd_act_backend_does_not_ask_for_torch_or_anatomy(self) -> None:
        from shared.algorithm_selection import algorithms
        closure = expand_closure(("wma",), tractography_algorithms=algorithms("csd_act"))
        self.assertEqual(closure.targets, ("tractography", "wma"))
        # The doctor result holds the real closed shape: validation, tractography, wma (algorithms: ukf).
        doctor = SimpleNamespace(statuses=(), targets=("validation", "tractography", "wma"), blocker_count=0)
        owner = mock.Mock()
        owner._selected_closure.return_value = closure
        owner._parameterNode = SimpleNamespace(brainMaskBackend="provided")
        owner._tractography_backend_text.return_value = "csd_act"  # the user's own text, not the closed algorithms
        owner._environment_dependency_path_kwargs.return_value = {}
        owner._closure_notice_text.return_value = ""
        plan = _plan_for_setup(owner, doctor)
        self.assertNotIn("PyTorch", plan["extensions"])
        self.assertIn("UKFTractography", plan["extensions"])
        self.assertEqual(plan["asset_profiles"], [])
        self.assertEqual(plan["closure_targets"], ["tractography", "wma"])
        for name in ("segmentation", "parcellation"):
            self.assertNotIn(name, plan["closure_targets"])


class K2GuiHonoursASuppliedTractogramTest(unittest.TestCase):
    def test_k2_controller_closure_with_a_supplied_wma_tractogram_has_no_tractography_or_ukf(self) -> None:
        from shared import ui_run_controller as controller
        closure = controller.selected_closure(
            controller.ProductTargetFlags(bundles=True), tractography_backend="csd_act", bundle_backend="wma",
            microstructure_freewater=False, microstructure_noddi=False, provided_tractography=("wma",))
        self.assertEqual(closure.targets, ("validation", "wma"))  # acceptance case wma_ukf with an input file
        self.assertNotIn("ukf", closure.tractography_algorithms)
        without = controller.selected_closure(
            controller.ProductTargetFlags(bundles=True), tractography_backend="csd_act", bundle_backend="wma",
            microstructure_freewater=False, microstructure_noddi=False)
        self.assertEqual(without.targets, ("validation", "tractography", "wma"))

    def test_k2_widget_passes_the_wma_input_parameter_to_the_closure(self) -> None:
        from shared import ui_run_controller as controller
        provided = _widget_method("_provided_tractography")
        closure_method = _widget_method("_selected_closure", selected_closure=controller.selected_closure)
        for text, expected in (("/data/wma_input.vtk", ("wma",)), ("   ", ()), ("", ())):
            with self.subTest(text=text):
                node = SimpleNamespace(wmaInputTractography=text)
                owner = SimpleNamespace(
                    _parameterNode=node,
                    _product_target_flags=lambda: controller.ProductTargetFlags(bundles=True),
                    _tractography_backend_text=lambda: "csd_act", _bundle_backend_text=lambda: "wma",
                    _microstructure_stage_flags=lambda: (False, False), _microstructure_dki_enabled=lambda: False,
                    _microstructure_ifw_enabled=lambda: False)
                owner._provided_tractography = lambda owner=owner: provided(owner)
                self.assertEqual(provided(owner), expected)
                targets = closure_method(owner).targets
                self.assertEqual(targets, ("validation", "wma") if expected else ("validation", "tractography", "wma"))


class K3SafKeepsItsFodTest(unittest.TestCase):
    def test_k3_supplied_saf_tractogram_still_needs_the_tractography_stage(self) -> None:
        closure = expand_closure(("saf",), provided_tractography=("saf",))
        self.assertEqual(closure.targets, ("segmentation", "parcellation", "surface", "surface_postprocess",
                                           "tractography", "saf"))
        self.assertEqual(closure.tractography_algorithms, ("csd_act",))
        # WMA and TractCloud keep the shortcut.
        self.assertEqual(expand_closure(("wma",), provided_tractography=("wma",)).targets, ("wma",))
        self.assertEqual(expand_closure(("tractcloud",), provided_tractography=("tractcloud",)).targets, ("tractcloud",))

    def test_k3_cli_saf_with_a_pregenerated_tractogram(self) -> None:
        from shared.run_subject_pipeline_cli import _resolve_targets, build_parser
        args, _ = build_parser().parse_known_args(
            ["--subject-root", "/none", "--targets", "saf", "--saf-pregenerated-tractogram", "/none/saf.tck"])
        targets = _resolve_targets(args)
        for name in ("tractography", "segmentation", "parcellation", "surface_postprocess"):
            self.assertIn(name, targets)

    def test_k3_dry_run_plan_reports_saf_will_run_with_the_fod_from_the_tractography_stage(self) -> None:
        from shared.orchestrator import ProductOrchestratorConfig, plan_subject_products
        with tempfile.TemporaryDirectory() as tmp:
            root = _fresh_subject(tmp)
            tractogram = Path(tmp) / "saf.tck"
            tractogram.write_text("x")
            config = ProductOrchestratorConfig(subject_id="s", subject_root=root, targets=("saf",),
                                               saf_pregenerated_tractogram=tractogram,
                                               atlas_t2=root / "input" / "data.nii.gz", execute=True)
            items = {item.target: item for item in plan_subject_products(config, write=False).items}
        self.assertEqual(items["saf"].status, "will_run", items["saf"].reason)
        self.assertEqual(items["saf"].dependencies["fod"], "will_run:tractography.csd_act")
        self.assertNotIn("missing CSD FOD", items["saf"].reason)


class K4SavedDefaultsHoldTheUsersTicksTest(unittest.TestCase):
    def test_k4_gui_saves_the_requested_targets_not_the_closed_set(self) -> None:
        from shared import ui_run_controller as controller
        saved: dict = {}
        node = SimpleNamespace(
            orchestrationMicrostructureStages="dti", subjectId="s", inputDirectory="", outputDirectory="",
            orchestrationRunName="r", ddsegDevice="cpu", device="cpu",
            orchestrationTractographyBackend="csd_act", orchestrationBundleBackend="wmql",
            orchestrationSelectStreamlines=1000, autoInstallPythonDependencies=False, connectomeBackend="python")
        closure = controller.selected_closure(
            controller.ProductTargetFlags(connectome=True), tractography_backend="csd_act", bundle_backend="wmql",
            microstructure_freewater=False, microstructure_noddi=False)
        self.assertIn("tractography", closure.targets)
        save = _widget_method(
            "_save_common_user_settings", slicer=SimpleNamespace(app=SimpleNamespace(userSettings=mock.Mock())),
            canonical_microstructure_stages=lambda text: ("dti",), ENVIRONMENT_DEPENDENCY_PATHS=(),
            update_user_config=lambda payload: (saved.update(payload) or ({}, Path("/none/config.json"))))
        owner = SimpleNamespace(
            _parameterNode=node, _common_user_setting_specs=lambda: (), _user_settings_key=lambda name: name,
            _requested_product_targets=lambda: closure.requested, _selected_product_targets=lambda: closure.targets,
            appendLog=lambda text: self.fail(text))
        save(owner)
        targets = saved["run_defaults"]["targets"]
        self.assertIn("connectome", targets)
        for name in ("segmentation", "parcellation", "tractography"):
            self.assertNotIn(name, targets)

    def test_k4_cli_save_user_config_saves_only_the_requested_targets(self) -> None:
        from shared import run_subject_pipeline_cli as cli
        written: list = []
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(cli, "write_user_config", side_effect=lambda payload, path: written.append(payload) or path), \
                mock.patch.object(cli, "_run_environment_doctor_from_args",
                                  return_value=(SimpleNamespace(to_dict=lambda: {}), {})), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            code = cli.main(["--subject-root", str(Path(tmp) / "s"), "--targets", "connectome", "--no-user-config",
                             "--save-user-config", str(Path(tmp) / "x.json"), "--doctor"])
        self.assertEqual(code, 0)
        self.assertEqual(len(written), 1)
        self.assertEqual(written[0]["run_defaults"]["targets"], ["connectome"])


class K5SkippedTargetsAreNeverExpandedTest(unittest.TestCase):
    def test_k5_closure_property_over_every_target_and_every_single_skip(self) -> None:
        for requested in TARGET_ORDER:
            for skip in TARGET_ORDER:
                with self.subTest(requested=requested, skip=skip):
                    closure = expand_closure((requested,), excluded=(skip,))
                    self.assertNotIn(skip, closure.targets)
                    self.assertNotIn(skip, closure.requested)
                    self.assertNotIn(skip, closure.required_by)
                    for requesters in closure.required_by.values():
                        self.assertNotIn(skip, requesters)
                        self.assertTrue(set(requesters) <= set(closure.targets))
                    # nothing is in the closure only because the skipped target asked for it
                    for added in closure.added:
                        self.assertTrue(closure.required_by[added])

    def test_k5_cli_skip_tractography_leaves_no_target_that_only_tractography_required(self) -> None:
        from shared.run_subject_pipeline_cli import _resolve_closure, build_parser
        args, _ = build_parser().parse_known_args(
            ["--subject-root", "/none", "--targets", "connectome", "--skip-targets", "tractography"])
        closure, _exclude = _resolve_closure(args)
        self.assertEqual(closure.targets, ("parcellation", "connectome"))
        self.assertNotIn("segmentation", closure.targets)
        self.assertNotIn("tractography", closure.required_by)
        for requesters in closure.required_by.values():
            self.assertNotIn("tractography", requesters)

    def test_k5_orchestrator_passes_the_skip_set_into_the_closure(self) -> None:
        from shared.orchestrator import ProductOrchestratorConfig, _with_closure
        config = _with_closure(ProductOrchestratorConfig(
            subject_id="s", subject_root=Path("/none"), targets=("connectome",), excluded_targets=("tractography",)))
        self.assertEqual(config.targets, ("parcellation", "connectome"))
        for requesters in config.target_required_by.values():
            self.assertNotIn("tractography", requesters)


class K6ResumeKeepsTheRegionalStatisticsStatusTest(unittest.TestCase):
    def _run(self, root: Path, lines: list, *, complete: bool, runs: list):
        from shared import orchestrator
        config = orchestrator.ProductOrchestratorConfig(
            subject_id="s", subject_root=root, targets=("microstructure",), execute=True, resume=True)

        def fake_run(_config, log=None):
            runs.append(1)
            return SimpleNamespace(**{name: Path(name) for name in (
                "fa_path", "md_path", "ad_path", "rd_path", "l1_path", "l2_path", "l3_path", "trace_path")},
                roi_stats_csv_path=None, roi_stats_json_path=None)

        fake_micro = SimpleNamespace(MicrostructureConfig=lambda **kw: SimpleNamespace(**kw), run_microstructure=fake_run)
        fake_pipeline = SimpleNamespace(
            ensure_native_dti_bundle=lambda *a, **k: None, find_subject_root=lambda *a, **k: None,
            resolve_preprocessed_input_bundle=lambda *a, **k: None)
        passed = SimpleNamespace(report_path=root / "v.json", passed=True, status="passed", error_count=0,
                                 warning_count=0, messages=())
        with mock.patch.dict("sys.modules", {"microstructure": fake_micro, "shared.pipeline": fake_pipeline}), \
                mock.patch.object(orchestrator, "validate_subject_inputs", return_value=passed), \
                mock.patch.object(orchestrator, "_microstructure_dti_complete", return_value=complete):
            result = orchestrator.run_subject_products(config, log=lines.append)
        manifest = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
        return {task["name"]: task for task in manifest["tasks"]}

    def test_k6_resume_without_parcellation_keeps_skipped_no_parcellation_and_the_log_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _fresh_subject(tmp)
            runs: list = []
            first_lines: list = []
            first = self._run(root, first_lines, complete=False, runs=runs)
            self.assertEqual(first["microstructure.dti"]["status"], "completed")
            self.assertEqual(first["microstructure.dti"]["regional_statistics"], "skipped_no_parcellation")
            lines: list = []
            second = self._run(root, lines, complete=True, runs=runs)
        self.assertEqual(len(runs), 1)
        self.assertEqual(second["microstructure.dti"]["status"], "skipped")
        self.assertEqual(second["microstructure.dti"]["regional_statistics"], "skipped_no_parcellation")
        self.assertIn("[Microstructure] Regional statistics skipped: parcellation was not selected", lines)

    def test_k6_resume_with_a_parcellation_now_present_and_no_roi_file_reruns_the_stage(self) -> None:
        from shared.io_layout import subject_layout
        with tempfile.TemporaryDirectory() as tmp:
            root = _fresh_subject(tmp)
            parcellation = subject_layout(root).anatomy_input_dir("ddparcel") / "s_ddparcel_wmparc_labels.nii.gz"
            parcellation.parent.mkdir(parents=True, exist_ok=True)
            parcellation.write_text("x")
            runs: list = []
            lines: list = []
            tasks = self._run(root, lines, complete=True, runs=runs)  # outputs claim to be complete, ROI file is missing
        self.assertEqual(len(runs), 1)
        self.assertEqual(tasks["microstructure.dti"]["status"], "completed")
        self.assertNotIn("regional_statistics", tasks["microstructure.dti"])


class Q1NoticeOncePerRunTest(unittest.TestCase):
    def test_q1_a_config_that_is_already_closed_does_not_repeat_the_notice(self) -> None:
        from shared import orchestrator
        config = orchestrator._with_closure(orchestrator.ProductOrchestratorConfig(
            subject_id="s", subject_root=Path("/none"), targets=("tractography",), execute=True))
        self.assertTrue(config.closure_applied)
        lines: list[str] = []
        with mock.patch.object(orchestrator, "_run_subject_products_impl", return_value="done"):
            orchestrator.run_subject_products(config, log=lines.append)
        self.assertEqual([line for line in lines if line.startswith("[Targets]")], [])

    def test_q1_multi_tracking_logs_the_notice_once(self) -> None:
        from shared import orchestrator
        lines: list[str] = []
        config = orchestrator.ProductOrchestratorConfig(
            subject_id="s", subject_root=Path("/none"), targets=("bundles",), bundle_backend="all",
            tractography_backend="csd_act", execute=True)
        children: list = []

        def run_children(parent, log=None):
            for child in ({"tractography_backend": "csd_act"}, {"tractography_backend": "ukf"}):
                children.append(orchestrator.run_subject_products(orchestrator.replace(parent, **child), log=log))
            return "done"

        with mock.patch("shared.multi_tracking.run", side_effect=run_children), \
                mock.patch.object(orchestrator, "_run_subject_products_impl", return_value="done"):
            orchestrator.run_subject_products(config, log=lines.append)
        self.assertEqual(len(children), 2)
        self.assertEqual(len([line for line in lines if line.startswith("[Targets]")]), 1, lines)


class Q2DownstreamReasonTest(unittest.TestCase):
    def test_q2_a_stage_in_this_run_is_not_told_to_select_itself(self) -> None:
        from shared.orchestrator import _upstream_reason
        text = "missing DDSeg segmentation; select segmentation or provide existing output"
        self.assertEqual(_upstream_reason({"segmentation", "tractography"}, "segmentation", text),
                         "DDSeg is in this run but produced no output (see its task)")
        self.assertEqual(_upstream_reason({"tractography"}, "segmentation", text), text)

    def test_q2_plan_names_the_selected_upstream_stage(self) -> None:
        from shared.orchestrator import ProductOrchestratorConfig, plan_subject_products
        with tempfile.TemporaryDirectory() as tmp:
            config = ProductOrchestratorConfig(subject_id="s", subject_root=Path(tmp) / "s", targets=("connectome",),
                                               execute=True)
            items = {item.target: item for item in plan_subject_products(config, write=False).items}
        for item in items.values():
            self.assertNotIn("select segmentation", item.reason)
            self.assertNotIn("select parcellation", item.reason)
            self.assertNotIn("select tractography", item.reason)


class Q3NoticeNamesAddedAlgorithmsTest(unittest.TestCase):
    def test_q3_wma_alone_names_ukf_tractography(self) -> None:
        self.assertEqual(closure_notice(expand_closure(("wma",))),
                         "Also runs automatically: UKF tractography (needed by WMA)")

    def test_q3_ukf_added_to_a_requested_tractography_run(self) -> None:
        self.assertEqual(closure_notice(expand_closure(("wma", "tractography"))),
                         "Also runs automatically: DDSeg, DDParcel (needed by CSD/ACT tractography); "
                         "UKF tractography (needed by WMA)")

    def test_q3_run_notice_for_a_closed_config_includes_the_algorithm(self) -> None:
        from shared.orchestrator import ProductOrchestratorConfig, _with_closure, closure_notice_for
        config = _with_closure(ProductOrchestratorConfig(subject_id="s", subject_root=Path("/none"), targets=("wma",)))
        self.assertEqual(closure_notice_for(config), "Also runs automatically: UKF tractography (needed by WMA)")


class Q4SafAlgorithmIsRequestedTest(unittest.TestCase):
    def test_q4_picking_the_saf_algorithm_requests_the_saf_target(self) -> None:
        closure = expand_closure(("tractography",), tractography_algorithms=("saf",))
        self.assertIn("saf", closure.requested)
        self.assertNotIn("saf", closure.required_by)
        self.assertEqual(closure.provenance("saf"), {"requested": True, "required_by": []})
        self.assertNotIn("SAF (", closure_notice(closure))
        self.assertNotIn("SAF,", closure_notice(closure))
        # A saf target added by another rule keeps its provenance.
        added = expand_closure(("tractography",), tractography_algorithms=("csd_act",))
        self.assertNotIn("saf", added.targets)

    def test_q4_cli_connectome_with_the_saf_backend_requests_saf(self) -> None:
        from shared.run_subject_pipeline_cli import _resolve_closure, build_parser
        args, _ = build_parser().parse_known_args(
            ["--subject-root", "/none", "--targets", "connectome", "--tractography-backend", "saf"])
        closure, _exclude = _resolve_closure(args)
        self.assertIn("tractography", closure.targets)
        self.assertIn("saf", closure.requested)
        self.assertEqual(closure.provenance("saf"), {"requested": True, "required_by": []})
        self.assertNotIn("saf", closure.added)
        self.assertNotIn("SAF (needed", closure_notice(closure))  # SAF may be named as a requester only
        self.assertIn("surface_postprocess", closure.targets)

    def test_q4_a_skipped_saf_stays_out(self) -> None:
        closure = expand_closure(("connectome",), tractography_algorithms=("saf",), excluded=("saf",))
        self.assertNotIn("saf", closure.targets)
        self.assertNotIn("saf", closure.requested)


class Q5Q6Q7SmallFixesTest(unittest.TestCase):
    def test_q5_environment_uses_the_shared_mask_target_list(self) -> None:
        from shared.environment import run_needs_brain_mask
        with tempfile.TemporaryDirectory() as tmp:
            root = _fresh_subject(tmp)
            (root / "input" / "nodif_brain_mask.nii.gz").unlink()
            from shared.target_closure import MASK_TARGETS
            for target in MASK_TARGETS:
                with self.subTest(target=target):
                    self.assertTrue(run_needs_brain_mask(root, targets=(target,), subject_id="s", output_root=None,
                                                         brain_mask_backend="pnl_cnn"))

    def test_q6_cli_subject_id_uses_the_shared_sanitiser(self) -> None:
        from shared.run_subject_pipeline_cli import _infer_subject_id
        from shared.subject_id import derive_subject_id
        with tempfile.TemporaryDirectory() as tmp:
            odd = Path(tmp) / "pat ient"
            odd.mkdir()
            self.assertEqual(_infer_subject_id(odd), "pat_ient")
        for text in (".", "..", "a/.."):
            self.assertNotIn(derive_subject_id(text), (".", ".."))

    def test_q7_no_resume_help_names_the_prerequisites(self) -> None:
        from shared.run_subject_pipeline_cli import build_parser
        action = next(a for a in build_parser()._actions if "--no-resume" in a.option_strings)
        self.assertEqual(action.help, "Recompute every stage of this run, including prerequisites added automatically.")


if __name__ == "__main__":
    unittest.main()
