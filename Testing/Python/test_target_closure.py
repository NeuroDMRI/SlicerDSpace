#!/usr/bin/env python3
"""Automatic prerequisite closure (C1-C8): one ticked target runs every stage it needs.

All tests are pure or mocked: no numpy, nibabel or Slicer is needed.
"""

from __future__ import annotations

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
        mac = build_setup_plan(self._missing_mask_report(), system="Darwin", machine="arm64")
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


if __name__ == "__main__":
    unittest.main()
