#!/usr/bin/env python3
"""Unit tests for tractography and audited algorithm parameter panels, validation, and end-to-end wiring."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from shared import ui_algorithm_picker
from shared.experiments import compile_plan, make_request, parameter_args
from shared.orchestrator import (
    ProductOrchestratorConfig,
    _effective_csd_act_options,
    _effective_dti_tracking_options,
    _effective_saf_options,
    run_subject_products,
)
from shared.run_subject_pipeline_cli import build_parser, resolve_analysis_options
from shared.ui_run_controller import (
    RunSelection,
    build_background_cli_args,
    build_batch_background_cli_args,
    build_batch_run_kwargs,
    build_single_subject_run_kwargs,
)
from shared.ui_settings import COMMON_USER_SETTING_SPECS
from shared.ui_tractography_params import (
    CSDActParameters,
    DTIParameters,
    SAFParameters,
    csd_act_overrides_json,
    dti_backend_fit_method,
    dti_overrides_json,
    get_node_bool_parameter,
    get_node_parameter,
    parse_csd_act_overrides,
    parse_dti_overrides,
    parse_freewater_overrides,
    parse_ifw_overrides,
    parse_saf_overrides,
    parse_tractcloud_overrides,
    saf_overrides_json,
    set_node_parameter,
    validate_csd_act_parameters,
    validate_dti_parameters,
    validate_microstructure_dti_fit_method,
    validate_saf_parameters,
)
from test_ui_run_history import by_name, fake_qt, object_name
import test_ui_run_history as _ui_run_history_mod


CLOSE = 19


class _FakeQObject:
    pass


class _CloseEvent:
    def __init__(self, kind=CLOSE):
        self._kind = kind
        self.ignored = 0

    def type(self):
        return self._kind

    def ignore(self):
        self.ignored += 1


class _InteractiveCombo:
    """Minimal combo double that tracks added items, index, and itemData."""

    def __init__(self):
        self.kind = "QComboBox"
        self.items: list[tuple[str, object]] = []
        self.currentIndex = 0
        self.setObjectName = mock.MagicMock()

    def addItem(self, label, data=None):
        self.items.append((label, data))

    def setCurrentIndex(self, idx):
        self.currentIndex = int(idx)

    def itemData(self, idx):
        return self.items[int(idx)][1] if 0 <= int(idx) < len(self.items) else None

    def itemText(self, idx):
        return self.items[int(idx)][0] if 0 <= int(idx) < len(self.items) else ""

    @property
    def currentText(self):
        return self.itemText(self.currentIndex)

    def setToolTip(self, _tip):
        pass


def _interactive_qt():
    qt = fake_qt()
    qt.QObject = _FakeQObject
    qt.QEvent.Close = CLOSE

    def make_combo(*_args, **_kwargs):
        combo = _InteractiveCombo()
        qt.created.append(combo)
        return combo

    qt.QComboBox.side_effect = make_combo
    return qt


class ParameterValidationTest(unittest.TestCase):
    def test_csd_act_defaults_round_trip_and_auto_normalization(self) -> None:
        self.assertEqual(parse_csd_act_overrides("{}"), {})
        self.assertEqual(parse_csd_act_overrides(""), {})
        self.assertEqual(parse_csd_act_overrides(None), {})
        defaults = validate_csd_act_parameters("{}")
        self.assertEqual(defaults, CSDActParameters())
        self.assertEqual(csd_act_overrides_json(defaults), "{}")

        # 0 / 0.0 map cleanly to None for auto/unseeded fields.
        auto_parsed = parse_csd_act_overrides(
            json.dumps({"step_size_mm": 0.0, "cutoff": 0, "random_seed": 0})
        )
        self.assertEqual(auto_parsed, {"step_size_mm": None, "cutoff": None, "random_seed": None})

        custom = {
            "select_streamlines": 500_000,
            "algorithm": "SD_Stream",
            "seeding": "gmwmi",
            "min_length_mm": 15.0,
            "max_length_mm": 180.0,
            "step_size_mm": 0.5,
            "cutoff": 0.08,
            "use_sift2": False,
            "random_seed": 42,
        }
        validated = validate_csd_act_parameters(json.dumps(custom))
        self.assertEqual(validated.algorithm, "SD_Stream")
        self.assertEqual(validated.seeding, "gmwmi")
        self.assertEqual(validated.step_size_mm, 0.5)
        self.assertEqual(validated.cutoff, 0.08)
        self.assertFalse(validated.use_sift2)
        self.assertEqual(validated.random_seed, 42)
        round_tripped = validate_csd_act_parameters(csd_act_overrides_json(validated))
        self.assertEqual(round_tripped, validated)

    def test_csd_act_rejects_invalid_values(self) -> None:
        bad_payloads = [
            "not-json",
            "[1, 2]",
            {"unknown_key": 1},
            {"select_streamlines": 500},
            {"select_streamlines": 10_000_001},
            {"select_streamlines": True},
            {"algorithm": "Fact"},
            {"seeding": "white"},
            {"min_length_mm": 0.5},
            {"max_length_mm": 600.0},
            {"min_length_mm": 50.0, "max_length_mm": 50.0},
            {"min_length_mm": 80.0, "max_length_mm": 40.0},
            {"step_size_mm": -0.1},
            {"step_size_mm": 6.0},
            {"cutoff": 1.5},
            {"cutoff": float("nan")},
            {"use_sift2": "yes"},
            {"random_seed": -1},
            {"random_seed": True},
        ]
        for payload in bad_payloads:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                validate_csd_act_parameters(payload)

    def test_dti_defaults_round_trip_and_ols_backend_mapping(self) -> None:
        self.assertEqual(parse_dti_overrides("{}"), {})
        defaults = validate_dti_parameters("{}")
        self.assertEqual(defaults, DTIParameters())
        self.assertEqual(dti_overrides_json(defaults), "{}")

        # Both "ols" and "ls" are accepted and map to "ls" for DTITrackingConfig.
        self.assertEqual(validate_dti_parameters({"fit_method": "ols"}).fit_method, "ols")
        self.assertEqual(validate_dti_parameters({"fit_method": "ls"}).fit_method, "ols")
        self.assertEqual(dti_backend_fit_method("ols"), "ls")
        self.assertEqual(dti_backend_fit_method("ls"), "ls")
        self.assertEqual(dti_backend_fit_method("wls"), "wls")

        custom = {
            "fit_method": "ols",
            "step_size_mm": 0.75,
            "max_angle_deg": 45.0,
            "seed_density": 2,
            "min_fa": 0.15,
            "min_length_mm": 12.0,
            "max_length_mm": 200.0,
            "random_seed": 99,
        }
        validated = validate_dti_parameters(json.dumps(custom))
        self.assertEqual(validate_dti_parameters(dti_overrides_json(validated)), validated)

    def test_dti_rejects_invalid_values(self) -> None:
        bad_payloads = [
            {"unknown": 1},
            {"fit_method": "nlls"},
            {"step_size_mm": 0.05},
            {"step_size_mm": 5.5},
            {"max_angle_deg": 2.0},
            {"max_angle_deg": 95.0},
            {"seed_density": 0},
            {"seed_density": 6},
            {"seed_density": True},
            {"min_fa": 0.0},
            {"min_fa": 0.99},
            {"min_length_mm": 60.0, "max_length_mm": 60.0},
            {"random_seed": -5},
        ]
        for payload in bad_payloads:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                validate_dti_parameters(payload)

    def test_saf_defaults_round_trip_and_invalid_values(self) -> None:
        self.assertEqual(parse_saf_overrides("{}"), {})
        defaults = validate_saf_parameters("{}")
        self.assertEqual(defaults, SAFParameters())
        self.assertEqual(saf_overrides_json(defaults), "{}")

        custom = {
            "select_streamlines": 250_000,
            "maxlen": 60.0,
            "notrunc": True,
            "fast_intersection": False,
            "gwg_hard": True,
            "pialfilter": True,
            "random_seed": 2026,
        }
        validated = validate_saf_parameters(json.dumps(custom))
        self.assertEqual(validate_saf_parameters(saf_overrides_json(validated)), validated)

        for payload in (
            {"unknown": 1},
            {"select_streamlines": 100},
            {"maxlen": 5.0},
            {"maxlen": 350.0},
            {"notrunc": 1},
            {"fast_intersection": "true"},
            {"gwg_hard": 0},
            {"pialfilter": "no"},
            {"random_seed": -1},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                validate_saf_parameters(payload)

    def test_audited_other_algorithm_validators_and_mrml_helpers(self) -> None:
        self.assertEqual(validate_microstructure_dti_fit_method("wls"), "wls")
        self.assertEqual(validate_microstructure_dti_fit_method("ls"), "ls")
        self.assertEqual(validate_microstructure_dti_fit_method("ols"), "ls")
        with self.assertRaises(ValueError):
            validate_microstructure_dti_fit_method("bad")

        self.assertEqual(parse_freewater_overrides("{}"), {})
        self.assertEqual(parse_freewater_overrides({"max_fraction_for_tissue": 0.6}), {"max_fraction_for_tissue": 0.6})
        with self.assertRaises(ValueError):
            parse_freewater_overrides({"max_fraction_for_tissue": 1.0})

        self.assertEqual(parse_ifw_overrides("{}"), {})
        self.assertEqual(parse_ifw_overrides({"t2_gm_ms": 85.0, "t2_csf_ms": 1400.0}),
                         {"t2_gm_ms": 85.0, "t2_csf_ms": 1400.0})
        with self.assertRaises(ValueError):
            parse_ifw_overrides({"t2_gm_ms": 0.0})

        self.assertEqual(parse_tractcloud_overrides("{}"), {})
        self.assertEqual(parse_tractcloud_overrides({"batch_size": 512}), {"batch_size": 512})
        with self.assertRaises(ValueError):
            parse_tractcloud_overrides({"batch_size": 16})

        # MRML parameterNodeWrapper fallback persists via SetParameter / GetParameter.
        store: dict[str, str] = {}
        raw = SimpleNamespace(
            GetParameter=lambda key: store.get(key, ""),
            SetParameter=lambda key, val: store.__setitem__(key, val),
        )
        wrapper = SimpleNamespace(_is_parameter_node_wrapper=True, allParameters={}, parameterNode=raw)
        set_node_parameter(wrapper, "csdActParameters", '{"cutoff": 0.08}')
        set_node_parameter(wrapper, "preprocessingRunGibbs", False)
        self.assertEqual(store["csdActParameters"], '{"cutoff": 0.08}')
        self.assertEqual(store["preprocessingRunGibbs"], "false")
        fresh_wrapper = SimpleNamespace(_is_parameter_node_wrapper=True, allParameters={}, parameterNode=raw)
        self.assertEqual(get_node_parameter(fresh_wrapper, "csdActParameters", "{}"), '{"cutoff": 0.08}')
        self.assertFalse(get_node_bool_parameter(fresh_wrapper, "preprocessingRunGibbs", True))


class EndToEndWiringTest(unittest.TestCase):
    def _base_selection(self, root: Path, **overrides) -> RunSelection:
        base = RunSelection(
            subject_id="sub-01",
            subject_root_text=(root / "sub-01").as_posix(),
            output_root_text=(root / "out").as_posix(),
            selected_targets=("validation", "tractography"),
            run_name="Test1",
            tractography_backend="csd_act",
            bundle_backend="wmql",
            microstructure_freewater=False,
            microstructure_noddi=False,
            select_streamlines=1_000_000,
            ddseg_device="cpu",
            surface_device="cpu",
            connectome_backend="auto",
            connectome_compress_mrtrix_dense=None,
        )
        return replace(base, **overrides)

    def test_defaults_emit_no_extra_cli_flags_and_identical_orchestrator_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selection = self._base_selection(root, tractography_backend="csd_act,dti_det,saf")
            single = build_single_subject_run_kwargs(selection, {}, preflight_blocked_targets={})
            batch = build_batch_run_kwargs(selection, batch_root=root / "batch")
            for key in (
                "csd_act_parameters",
                "dti_parameters",
                "saf_parameters",
                "preprocessing_dipy_run_gibbs",
                "microstructure_fit_method",
                "microstructure_free_water_max_fraction_for_tissue",
                "microstructure_ifw_t2_gm_ms",
                "microstructure_ifw_t2_csf_ms",
                "tractcloud_batch_size",
            ):
                self.assertNotIn(key, single)
                self.assertNotIn(key, batch)

            single_cli = build_background_cli_args(single, {}, repo_root=root)
            batch_cli = build_batch_background_cli_args(
                batch, {}, repo_root=root, batch_root=root / "batch", batch_size=1, arrange_inputs="none",
            )
            for flag in (
                "--csd-act-parameters",
                "--dti-parameters",
                "--saf-parameters",
                "--preprocessing-skip-gibbs",
                "--micro-fit-method",
                "--microstructure-free-water-max-fraction-for-tissue",
                "--ifw-t2-gm-ms",
                "--ifw-t2-csf-ms",
                "--tractcloud-batch-size",
            ):
                self.assertNotIn(flag, single_cli)
                self.assertNotIn(flag, batch_cli)

            # Orchestrator effective options with default {} match legacy defaults verbatim.
            cfg = ProductOrchestratorConfig(subject_id="sub-01", subject_root=root / "sub-01", targets=("tractography",))
            self.assertEqual(
                _effective_csd_act_options(cfg),
                {
                    "algorithm": "iFOD2",
                    "select_streamlines": 1_000_000,
                    "seeding": "dynamic",
                    "min_length_mm": 10.0,
                    "max_length_mm": 250.0,
                    "step_size_mm": None,
                    "cutoff": None,
                    "use_sift2": True,
                    "random_seed": None,
                },
            )
            self.assertEqual(
                _effective_dti_tracking_options(cfg),
                {
                    "fit_method": "wls",
                    "step_size_mm": 0.5,
                    "max_angle_deg": 30.0,
                    "seed_density": 1,
                    "min_fa": 0.2,
                    "min_length_mm": 10.0,
                    "max_length_mm": 250.0,
                    "random_seed": 12345,
                },
            )
            self.assertEqual(
                _effective_saf_options(cfg),
                {
                    "select_streamlines": 1_000_000,
                    "maxlen": 40.0,
                    "notrunc": False,
                    "fast_intersection": True,
                    "gwg_hard": False,
                    "pialfilter": False,
                    "random_seed": 12345,
                },
            )

    def test_custom_params_round_trip_from_run_controller_through_cli_parser_and_experiments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject = root / "sub-01"
            subject.mkdir()
            csd_json = json.dumps({"algorithm": "SD_Stream", "seeding": "gmwmi", "cutoff": 0.08, "use_sift2": False, "random_seed": 7})
            dti_json = json.dumps({"fit_method": "ols", "step_size_mm": 0.7, "max_angle_deg": 40.0, "seed_density": 2, "min_fa": 0.15})
            saf_json = json.dumps({"select_streamlines": 400_000, "maxlen": 65.0, "notrunc": True, "gwg_hard": True, "pialfilter": True, "random_seed": 77})

            selection = self._base_selection(
                root,
                tractography_backend="csd_act,dti_det,saf",
                csd_act_parameters_text=csd_json,
                dti_parameters_text=dti_json,
                saf_parameters_text=saf_json,
                preprocessing_dipy_run_gibbs=False,
                microstructure_fit_method="ls",
                freewater_parameters_text='{"max_fraction_for_tissue": 0.6}',
                ifw_parameters_text='{"t2_gm_ms": 85.0, "t2_csf_ms": 1400.0}',
                tractcloud_parameters_text='{"batch_size": 512}',
            )
            single = build_single_subject_run_kwargs(selection, {}, preflight_blocked_targets={})
            single_cli = build_background_cli_args(single, {}, repo_root=root)

            # Parse emitted CLI tokens through the real run_subject_pipeline_cli parser.
            parsed = build_parser().parse_args(single_cli[3:])
            resolve_analysis_options(parsed)
            self.assertEqual(parse_csd_act_overrides(parsed.csd_act_parameters)["algorithm"], "SD_Stream")
            self.assertEqual(parse_csd_act_overrides(parsed.csd_act_parameters)["use_sift2"], False)
            self.assertEqual(parse_dti_overrides(parsed.dti_parameters)["fit_method"], "ols")
            self.assertEqual(parse_saf_overrides(parsed.saf_parameters)["maxlen"], 65.0)
            self.assertTrue(parsed.preprocessing_skip_gibbs)
            self.assertEqual(parsed.micro_fit_method, "ls")
            self.assertEqual(parsed.microstructure_free_water_max_fraction_for_tissue, 0.6)
            self.assertEqual(parsed.ifw_t2_gm_ms, 85.0)
            self.assertEqual(parsed.ifw_t2_csf_ms, 1400.0)
            self.assertEqual(parsed.tractcloud_batch_size, 512)

            # Batch CLI also carries all flags.
            batch = build_batch_run_kwargs(selection, batch_root=root / "batch")
            batch_cli = build_batch_background_cli_args(
                batch, {}, repo_root=root, batch_root=root / "batch", batch_size=1, arrange_inputs="none",
            )
            for flag in (
                "--csd-act-parameters",
                "--dti-parameters",
                "--saf-parameters",
                "--preprocessing-skip-gibbs",
                "--micro-fit-method",
                "--microstructure-free-water-max-fraction-for-tissue",
                "--ifw-t2-gm-ms",
                "--ifw-t2-csf-ms",
                "--tractcloud-batch-size",
            ):
                self.assertIn(flag, batch_cli)

            # Experiments schema accepts valid params and rejects invalid overrides at preview.
            self.assertEqual(
                parameter_args(["--csd-act-parameters", csd_json, "--dti-parameters", dti_json, "--saf-parameters", saf_json]),
                ["--csd-act-parameters", csd_json, "--dti-parameters", dti_json, "--saf-parameters", saf_json],
            )
            for bad_flag in (
                '--csd-act-parameters \'{"unknown": 1}\'',
                '--dti-parameters \'{"min_fa": 0.0}\'',
                '--saf-parameters \'{"maxlen": 5}\'',
            ):
                req = make_request(
                    base_args=["--targets", "validation", "--brain-mask-backend", "provided"],
                    subject_root=subject,
                    subject_id="sub-01",
                    output_root=root / "exp_out",
                    run_name="test",
                    options=bad_flag,
                )
                with self.subTest(bad_flag=bad_flag), self.assertRaises(ValueError):
                    compile_plan(req, python_executable=sys.executable, pipeline_script=Path("pipeline.py"))

    def test_orchestrator_passes_custom_params_to_csd_dti_and_saf_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject_root = root / "sub01"
            input_dir = subject_root / "input"
            input_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (input_dir / name).write_text("x", encoding="utf-8")

            # 1) CSD/ACT config dispatch
            from shared import subject_layout

            layout = subject_layout(subject_root)
            captured_csd = {}
            layout.ddseg.mkdir(parents=True, exist_ok=True)
            (layout.ddseg / "SegmentationMap_GMWMCSF.nii.gz").write_text("seg", encoding="utf-8")
            layout.ddparcel.mkdir(parents=True, exist_ok=True)
            (layout.ddparcel / "sub01_ddparcel_wmparc_labels.nii.gz").write_text("parc", encoding="utf-8")

            def fake_act_prep(cfg, *, log=None):
                cfg.output_dir.mkdir(parents=True, exist_ok=True)
                five_tt = cfg.output_dir / "5tt.nii.gz"
                gmwmi = cfg.output_dir / "gmwmi.nii.gz"
                qc = cfg.output_dir / "act_prep_qc.json"
                for p in (five_tt, gmwmi, qc):
                    p.write_text("{}", encoding="utf-8")
                return SimpleNamespace(five_tt_path=five_tt, gmwmi_path=gmwmi, qc_path=qc)

            def fake_run_csd(cfg, *, log=None):
                captured_csd["cfg"] = cfg
                cfg.output_dir.mkdir(parents=True, exist_ok=True)
                cfg.tractogram_dir.mkdir(parents=True, exist_ok=True)
                tck = cfg.tractogram_dir / f"{cfg.subject_id}_csd_act_{cfg.algorithm.lower()}.tck"
                trk = tck.with_suffix(".trk")
                vtk = tck.with_suffix(".vtk")
                manifest = cfg.output_dir / "csd_act_manifest.json"
                qc = cfg.output_dir / "csd_act_qc.json"
                for p in (tck, trk, vtk):
                    p.write_text("data", encoding="utf-8")
                manifest.write_text("{}", encoding="utf-8")
                qc.write_text('{"status":"completed"}', encoding="utf-8")
                return SimpleNamespace(
                    tractogram_path=tck,
                    tractogram_trk_path=trk,
                    tractogram_vtk_path=vtk,
                    sift2_weights_path=None,
                    manifest_path=manifest,
                    qc_path=qc,
                )

            fake_validation = SimpleNamespace(
                passed=True,
                status="completed",
                error_count=0,
                warning_count=0,
                messages=(),
                report_path=root / "validation_report.json",
            )
            with (
                mock.patch("shared.orchestrator.validate_subject_inputs", return_value=fake_validation),
                mock.patch("tractography.act_prep.run_act_prep", side_effect=fake_act_prep),
                mock.patch("tractography.csd.run_csd_act", side_effect=fake_run_csd),
                mock.patch("shared.orchestrator._run_ddseg_world_qc_gate", return_value=None),
            ):
                run_subject_products(
                    ProductOrchestratorConfig(
                        subject_id="sub01",
                        subject_root=subject_root,
                        targets=("tractography",),
                        tractography_backend="csd_act",
                        csd_act_parameters={
                            "select_streamlines": 200_000,
                            "algorithm": "SD_Stream",
                            "seeding": "gmwmi",
                            "min_length_mm": 15.0,
                            "max_length_mm": 190.0,
                            "step_size_mm": 0.4,
                            "cutoff": 0.09,
                            "use_sift2": False,
                            "random_seed": 123,
                        },
                        excluded_targets=("segmentation", "parcellation"),
                        execute=True,
                    )
                )
            csd_cfg = captured_csd["cfg"]
            self.assertEqual(csd_cfg.select_streamlines, 200_000)
            self.assertEqual(csd_cfg.algorithm, "SD_Stream")
            self.assertEqual(csd_cfg.seeding, "gmwmi")
            self.assertEqual(csd_cfg.min_length_mm, 15.0)
            self.assertEqual(csd_cfg.max_length_mm, 190.0)
            self.assertEqual(csd_cfg.step_size_mm, 0.4)
            self.assertEqual(csd_cfg.cutoff, 0.09)
            self.assertFalse(csd_cfg.use_sift2)
            self.assertEqual(csd_cfg.random_seed, 123)

            # 2) DTI deterministic config dispatch (verifies "ols" -> "ls" mapping!)
            captured_dti = {}

            def fake_run_dti(cfg, *, log=None):
                captured_dti["cfg"] = cfg
                cfg.output_dir.mkdir(parents=True, exist_ok=True)
                trk = cfg.output_dir / f"{cfg.subject_id}-dti-deterministic.trk"
                manifest = cfg.output_dir / f"{cfg.subject_id}-dti-deterministic_manifest.json"
                qc = cfg.output_dir / f"{cfg.subject_id}-dti-deterministic_qc.json"
                trk.write_text("trk", encoding="utf-8")
                manifest.write_text("{}", encoding="utf-8")
                qc.write_text('{"status":"completed"}', encoding="utf-8")
                return SimpleNamespace(streamlines_path=trk, manifest_path=manifest, qc_path=qc)

            with (
                mock.patch("shared.orchestrator.validate_subject_inputs", return_value=fake_validation),
                mock.patch("tractography.dti.run_dti_tracking", side_effect=fake_run_dti),
            ):
                run_subject_products(
                    ProductOrchestratorConfig(
                        subject_id="sub01",
                        subject_root=subject_root,
                        targets=("tractography",),
                        tractography_backend="dti_det",
                        dti_parameters={
                            "fit_method": "ols",
                            "step_size_mm": 0.8,
                            "max_angle_deg": 45.0,
                            "seed_density": 2,
                            "min_fa": 0.15,
                            "min_length_mm": 14.0,
                            "max_length_mm": 210.0,
                            "random_seed": 777,
                        },
                        execute=True,
                    )
                )
            dti_cfg = captured_dti["cfg"]
            self.assertEqual(dti_cfg.fit_method, "ls")
            self.assertEqual(dti_cfg.step_size_mm, 0.8)
            self.assertEqual(dti_cfg.max_angle_deg, 45.0)
            self.assertEqual(dti_cfg.seed_density, 2)
            self.assertEqual(dti_cfg.min_fa, 0.15)
            self.assertEqual(dti_cfg.min_length_mm, 14.0)
            self.assertEqual(dti_cfg.max_length_mm, 210.0)
            self.assertEqual(dti_cfg.random_seed, 777)

            # 3) SAF config dispatch
            csd_dir = layout.csd / "Test1"
            csd_dir.mkdir(parents=True, exist_ok=True)
            (csd_dir / "sub01_wmfod_norm.mif").write_text("fod", encoding="utf-8")
            fs_dir = root / "fs_sub01"
            fs_dir.mkdir()
            captured_saf = {}

            def fake_run_saf(cfg, *, log=None):
                captured_saf["cfg"] = cfg
                cfg.output_dir.mkdir(parents=True, exist_ok=True)
                tck = cfg.output_dir / "SAF.tck"
                vtk = cfg.output_dir / "SAF.vtk"
                manifest = cfg.output_dir / "saf_manifest.json"
                qc = cfg.output_dir / "saf_qc.json"
                tck.write_text("tck", encoding="utf-8")
                vtk.write_text("vtk", encoding="utf-8")
                manifest.write_text("{}", encoding="utf-8")
                qc.write_text('{"status":"completed"}', encoding="utf-8")
                return SimpleNamespace(
                    saf_tractogram=tck,
                    saf_vtk_path=vtk,
                    streamline_count=1234,
                    manifest_path=manifest,
                    qc_path=qc,
                )

            with (
                mock.patch("shared.orchestrator.validate_subject_inputs", return_value=fake_validation),
                mock.patch("tractography.run_saf", side_effect=fake_run_saf),
            ):
                run_subject_products(
                    ProductOrchestratorConfig(
                        subject_id="sub01",
                        subject_root=subject_root,
                        run_name="Test1",
                        targets=("saf",),
                        saf_fs_dir=fs_dir,
                        saf_parameters={
                            "select_streamlines": 300_000,
                            "maxlen": 75.0,
                            "notrunc": True,
                            "fast_intersection": False,
                            "gwg_hard": True,
                            "pialfilter": True,
                            "random_seed": 4242,
                        },
                        excluded_targets=("segmentation", "parcellation", "surface_postprocess", "tractography"),
                        execute=True,
                    )
                )
            saf_cfg = captured_saf["cfg"]
            self.assertEqual(saf_cfg.maxlen, 75.0)
            self.assertIn("-select 300000", saf_cfg.tckgen_options)
            self.assertIn("-maxlength 75", saf_cfg.tckgen_options)
            self.assertTrue(saf_cfg.notrunc)
            self.assertFalse(saf_cfg.fast_intersection)
            self.assertTrue(saf_cfg.gwg_hard)
            self.assertTrue(saf_cfg.pialfilter)
            self.assertEqual(saf_cfg.random_seed, 4242)

    def test_module_logic_accepts_all_run_controller_kwargs(self) -> None:
        import inspect
        from shared.module_logic import SharedLogicMixin

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selection = self._base_selection(
                root,
                tractography_backend="csd_act,dti_det,saf",
                csd_act_parameters_text='{"cutoff": 0.08}',
                dti_parameters_text='{"min_fa": 0.15}',
                saf_parameters_text='{"maxlen": 70.0}',
                preprocessing_dipy_run_gibbs=False,
                microstructure_fit_method="ls",
                freewater_parameters_text='{"max_fraction_for_tissue": 0.6}',
                ifw_parameters_text='{"t2_gm_ms": 85.0, "t2_csf_ms": 1400.0}',
                tractcloud_parameters_text='{"batch_size": 512}',
            )
            kwargs = build_single_subject_run_kwargs(selection, {}, preflight_blocked_targets={})
            for method_name in ("runSelectedProductOrchestration", "planSelectedProductOrchestration"):
                sig = inspect.signature(getattr(SharedLogicMixin, method_name))
                for key in kwargs:
                    self.assertIn(key, sig.parameters, f"{method_name} missing parameter {key!r}")


class AlgorithmPickerGearAndStagingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.qt = _interactive_qt()
        helper = _ui_run_history_mod.AlgorithmRowTest(
            "test_summary_text_aligns_with_a_plain_checkbox_and_rows_are_vertically_centred"
        )
        self.owner, self.slicer = helper._build_full_picker_owner(self.qt)
        self.node = self.owner._parameterNode
        self.node.orchestrationSelectStreamlines = 1_000_000
        self.node.csdActParameters = "{}"
        self.node.dtiParameters = "{}"
        self.node.safParameters = "{}"
        self.node.preprocessingRunGibbs = True
        self.node.microstructureFitMethod = "wls"
        self.node.freeWaterParameters = "{}"
        self.node.ifwParameters = "{}"
        self.node.tractcloudParameters = "{}"

    def _callback(self, name: str, signal: str):
        return next(c.args[1] for c in by_name(self.qt, name).connect.call_args_list if c.args[0] == signal)

    def _run_picker(self, family: str, scenario):
        orig = self.qt.QDialog.side_effect

        def make_dialog(*args, **kwargs):
            dlg = orig(*args, **kwargs)

            def fake_exec():
                flt = dlg.installEventFilter.call_args.args[0]
                scenario(flt, dlg)
                return self.qt.QDialog.Accepted if dlg.accept.called else 0

            dlg.exec_.side_effect = fake_exec
            return dlg

        self.qt.QDialog.side_effect = make_dialog
        with mock.patch.dict(sys.modules, {"qt": self.qt, "slicer": self.slicer}):
            return ui_algorithm_picker.choose_algorithms(self.owner, family)

    def test_all_four_tractography_algorithms_have_gear_buttons_in_en_and_zh(self) -> None:
        for lang in ("en", "zh"):
            with self.subTest(lang=lang):
                self.qt = _interactive_qt()
                helper = _ui_run_history_mod.AlgorithmRowTest(
            "test_summary_text_aligns_with_a_plain_checkbox_and_rows_are_vertically_centred"
        )
                self.owner, self.slicer = helper._build_full_picker_owner(self.qt)
                self.owner._onboardingLanguage = lang
                self._run_picker("tractography", lambda _flt, _dlg: None)
                gear_names = {object_name(w) for w in self.qt.created if w.kind == "QToolButton"}
                self.assertTrue(
                    {
                        "algorithmSettings_csd_act",
                        "algorithmSettings_ukf",
                        "algorithmSettings_dti_det",
                        "algorithmSettings_saf",
                    }
                    <= gear_names
                )
                # Key parameter controls exist for all 3 newly added backends.
                all_names = {object_name(w) for w in self.qt.created}
                self.assertTrue(
                    {
                        "csd_act_select_streamlines",
                        "csd_act_algorithm",
                        "csd_act_seeding",
                        "csd_act_use_sift2",
                        "csd_act_restore_defaults",
                        "dti_det_fit_method",
                        "dti_det_min_fa",
                        "dti_det_max_angle_deg",
                        "dti_det_restore_defaults",
                        "saf_select_streamlines",
                        "saf_maxlen",
                        "saf_gwg_hard",
                        "saf_restore_defaults",
                    }
                    <= all_names
                )

    def test_audited_other_families_expose_expected_gear_buttons(self) -> None:
        for family, expected_gears in (
            ("preprocessing", {"algorithmSettings_denoise"}),
            ("microstructure", {"algorithmSettings_dti", "algorithmSettings_freewater", "algorithmSettings_ifw"}),
            ("bundles", {"algorithmSettings_wmql", "algorithmSettings_wma", "algorithmSettings_tractcloud"}),
        ):
            with self.subTest(family=family):
                self.qt = _interactive_qt()
                helper = _ui_run_history_mod.AlgorithmRowTest(
            "test_summary_text_aligns_with_a_plain_checkbox_and_rows_are_vertically_centred"
        )
                self.owner, self.slicer = helper._build_full_picker_owner(self.qt)
                self._run_picker(family, lambda _flt, _dlg: None)
                gear_names = {object_name(w) for w in self.qt.created if w.kind == "QToolButton"}
                self.assertTrue(expected_gears <= gear_names)

    def test_look_only_outside_click_commits_nothing_for_tractography_and_other_families(self) -> None:
        for family in ("tractography", "preprocessing", "microstructure", "bundles"):
            with self.subTest(family=family):
                self.qt = _interactive_qt()
                helper = _ui_run_history_mod.AlgorithmRowTest(
            "test_summary_text_aligns_with_a_plain_checkbox_and_rows_are_vertically_centred"
        )
                self.owner, self.slicer = helper._build_full_picker_owner(self.qt)
                self.owner._parameterNode.orchestrationSelectStreamlines = 500_000
                self.owner._parameterNode.csdActParameters = "{}"
                self.owner._parameterNode.dtiParameters = "{}"
                self.owner._parameterNode.safParameters = "{}"
                self.owner._parameterNode.preprocessingRunGibbs = True
                self.owner._parameterNode.microstructureFitMethod = "wls"
                self.owner._parameterNode.freeWaterParameters = "{}"
                self.owner._parameterNode.ifwParameters = "{}"
                self.owner._parameterNode.tractcloudParameters = "{}"

                accepted = self._run_picker(family, lambda flt, dlg: flt.eventFilter(dlg, _CloseEvent()))
                self.assertFalse(accepted)
                self.assertFalse(self.owner._parameterNode.targetTractography)
                self.assertEqual(self.owner._parameterNode.csdActParameters, "{}")
                self.assertEqual(self.owner._parameterNode.orchestrationSelectStreamlines, 500_000)

    def test_staged_apply_commits_and_syncs_streamlines_and_cancel_discards(self) -> None:
        # Cancel discards staged edits.
        def edit_then_cancel(_flt, _dlg):
            by_name(self.qt, "csd_act_select_streamlines").setValue(250_000)
            by_name(self.qt, "csd_act_algorithm").setCurrentIndex(1)
            by_name(self.qt, "dti_det_min_fa").setValue(0.35)
            by_name(self.qt, "saf_maxlen").setValue(60.0)
            self._callback("algorithmPickerCancelButton", "clicked()")()

        self.assertFalse(self._run_picker("tractography", edit_then_cancel))
        self.assertEqual(self.node.csdActParameters, "{}")
        self.assertEqual(self.node.dtiParameters, "{}")
        self.assertEqual(self.node.safParameters, "{}")
        self.assertEqual(self.node.orchestrationSelectStreamlines, 1_000_000)

        # Apply commits staged edits and syncs orchestrationSelectStreamlines.
        self.qt = _interactive_qt()
        helper = _ui_run_history_mod.AlgorithmRowTest(
            "test_summary_text_aligns_with_a_plain_checkbox_and_rows_are_vertically_centred"
        )
        self.owner, self.slicer = helper._build_full_picker_owner(self.qt)
        self.node = self.owner._parameterNode
        self.node.orchestrationSelectStreamlines = 1_000_000

        def edit_then_apply(_flt, _dlg):
            by_name(self.qt, "csd_act_select_streamlines").setValue(250_000)
            by_name(self.qt, "csd_act_algorithm").setCurrentIndex(1)
            by_name(self.qt, "csd_act_use_sift2").setChecked(False)
            by_name(self.qt, "dti_det_fit_method").setCurrentIndex(1)
            by_name(self.qt, "dti_det_min_fa").setValue(0.35)
            by_name(self.qt, "saf_maxlen").setValue(60.0)
            by_name(self.qt, "saf_gwg_hard").setChecked(True)
            self._callback("algorithmPickerApplyButton", "clicked()")()

        self.assertTrue(self._run_picker("tractography", edit_then_apply))
        self.assertEqual(self.node.orchestrationSelectStreamlines, 250_000)
        csd_saved = json.loads(self.node.csdActParameters)
        self.assertEqual(csd_saved["select_streamlines"], 250_000)
        self.assertEqual(csd_saved["algorithm"], "SD_Stream")
        self.assertFalse(csd_saved["use_sift2"])
        dti_saved = json.loads(self.node.dtiParameters)
        self.assertEqual(dti_saved["fit_method"], "ols")
        self.assertEqual(dti_saved["min_fa"], 0.35)
        saf_saved = json.loads(self.node.safParameters)
        self.assertEqual(saf_saved["maxlen"], 60.0)
        self.assertTrue(saf_saved["gwg_hard"])

    def test_invalid_length_range_refuses_apply_and_keeps_popup_open(self) -> None:
        seen = {}

        def set_bad_lengths(_flt, dlg):
            by_name(self.qt, "csd_act_min_length_mm").setValue(80.0)
            by_name(self.qt, "csd_act_max_length_mm").setValue(40.0)
            self._callback("algorithmPickerApplyButton", "clicked()")()
            note = by_name(self.qt, "algorithmPickerNote")
            seen["accepted_after_error"] = dlg.accept.call_count
            seen["note_shown"] = note.show.called
            seen["note_text"] = note.setText.call_args.args[0]

        accepted = self._run_picker("tractography", set_bad_lengths)
        self.assertFalse(accepted)
        self.assertEqual(seen["accepted_after_error"], 0)
        self.assertTrue(seen["note_shown"])
        self.assertIn("min_length_mm", seen["note_text"])
        self.assertEqual(self.node.csdActParameters, "{}")

    def test_restore_defaults_button_resets_fields_to_empty_json(self) -> None:
        self.node.csdActParameters = '{"algorithm": "SD_Stream", "cutoff": 0.08, "select_streamlines": 500000}'
        self.node.orchestrationSelectStreamlines = 500_000

        def click_restore_then_apply(_flt, _dlg):
            reset_btn = by_name(self.qt, "csd_act_restore_defaults")
            reset_btn.clicked.connect.call_args.args[0](False)
            self._callback("algorithmPickerApplyButton", "clicked()")()

        self.assertTrue(self._run_picker("tractography", click_restore_then_apply))
        self.assertEqual(self.node.csdActParameters, "{}")
        self.assertEqual(self.node.orchestrationSelectStreamlines, 1_000_000)

    def test_common_user_setting_specs_persist_new_parameters(self) -> None:
        specs = dict(COMMON_USER_SETTING_SPECS)
        self.assertEqual(specs["csdActParameters"], str)
        self.assertEqual(specs["dtiParameters"], str)
        self.assertEqual(specs["safParameters"], str)
        self.assertEqual(specs["preprocessingRunGibbs"], bool)
        self.assertEqual(specs["microstructureFitMethod"], str)
        self.assertEqual(specs["freeWaterParameters"], str)
        self.assertEqual(specs["ifwParameters"], str)
        self.assertEqual(specs["tractcloudParameters"], str)


if __name__ == "__main__":
    unittest.main()
