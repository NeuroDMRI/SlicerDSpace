#!/usr/bin/env python3
"""T1 design D3 (W3): the iFW stage, its echo-time rule, grids, eligibility and closure entries."""

from __future__ import annotations

import json
import math
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
sys.path.insert(0, str(Path(__file__).resolve().parent))


def reference_derive_ifw(fw, csf, *, echo_time_ms, tissue_t2_ms, csf_t2_ms):
    """Transcription of `derive_ifw` in paper-latex scripts/figures/prepare_box2_ifw.py:38-58 (fact sheet)."""
    fw, csf = np.asarray(fw, dtype=np.float64), np.asarray(csf, dtype=np.float64)
    if fw.shape != csf.shape:
        raise ValueError('FW and CSF arrays must share the same physical grid')
    for name, array in (('FW', fw), ('CSF', csf)):
        if not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
            raise ValueError(name+' must be finite fractions in [0,1]')
    if (not np.isfinite([echo_time_ms, tissue_t2_ms, csf_t2_ms]).all()
            or echo_time_ms < 0 or min(tissue_t2_ms, csf_t2_ms) <= 0):
        raise ValueError('TE must be nonnegative; both T2 values must be positive, in ms')
    ct, cc = np.exp(-echo_time_ms/tissue_t2_ms), np.exp(-echo_time_ms/csf_t2_ms)
    if min(ct, cc) <= 0:
        raise ValueError('Relaxation attenuation underflows')
    corrected = fw*ct / (cc*(1-fw)+fw*ct)
    unprojected = corrected-csf
    return corrected, unprojected, np.maximum(unprojected, 0.)


class IfwFormulaTest(unittest.TestCase):
    def test_constants_and_citation(self):
        from microstructure import ifw
        self.assertEqual((ifw.IFW_T2_GM_MS, ifw.IFW_T2_CSF_MS), (95.0, 1500.0))
        self.assertIn("Supplementary Methods 1", ifw.IFW_REFERENCE)
        self.assertIn("10.21203/rs.3.rs-3179575", ifw.IFW_REFERENCE)

    def test_formula_against_hand_values(self):
        from microstructure.ifw import derive_ifw
        # TE = T2_GM: C_T = exp(-1); C_C = exp(-95/1500). Written out by hand, not through the module.
        ct, cc = math.exp(-1.0), math.exp(-95.0 / 1500.0)
        fw = np.array([0.0, 0.25, 0.5, 1.0])
        csf = np.array([0.0, 0.0, 0.1, 0.2])
        corrected, raw, ifw_map = derive_ifw(fw, csf, echo_time_ms=95.0, tissue_t2_ms=95.0, csf_t2_ms=1500.0)
        expected = [0.0, 0.25 * ct / (cc * 0.75 + 0.25 * ct), 0.5 * ct / (cc * 0.5 + 0.5 * ct), 1.0]
        np.testing.assert_allclose(corrected, expected, rtol=0, atol=1e-15)
        np.testing.assert_allclose(raw, np.array(expected) - csf, rtol=0, atol=1e-15)
        self.assertEqual(ifw_map[0], 0.0)           # no free water and no CSF
        self.assertEqual(ifw_map[3], 0.8)           # v = 1 and CSF = 0.2
        self.assertTrue(np.all(ifw_map >= 0))
        # TE = 0 leaves f unchanged; an equal T2 for both compartments leaves f unchanged.
        np.testing.assert_allclose(derive_ifw(fw, csf, echo_time_ms=0.0, tissue_t2_ms=95.0, csf_t2_ms=1500.0)[0], fw)
        np.testing.assert_allclose(derive_ifw(fw, csf, echo_time_ms=89.5, tissue_t2_ms=95.0, csf_t2_ms=95.0)[0], fw)

    def test_negative_values_project_to_zero(self):
        from microstructure.ifw import derive_ifw
        _, raw, ifw_map = derive_ifw([0.0, 0.5], [0.4, 0.0], echo_time_ms=0.0, tissue_t2_ms=95.0, csf_t2_ms=1500.0)
        self.assertAlmostEqual(raw[0], -0.4)
        self.assertEqual(ifw_map[0], 0.0)
        self.assertEqual(ifw_map[1], 0.5)

    def test_equals_reference_function_on_synthetic_arrays(self):
        from microstructure.ifw import derive_ifw
        rng = np.random.default_rng(20261107)
        for shape in ((7,), (4, 5, 6)):
            fw, csf = rng.random(shape), rng.random(shape)
            fw.flat[0], csf.flat[0] = 0.0, 1.0
            for te in (0.0, 30.0, 89.5, 120.0):
                got = derive_ifw(fw, csf, echo_time_ms=te, tissue_t2_ms=95.0, csf_t2_ms=1500.0)
                want = reference_derive_ifw(fw, csf, echo_time_ms=te, tissue_t2_ms=95.0, csf_t2_ms=1500.0)
                for left, right in zip(got, want):
                    self.assertEqual(left.dtype, np.float64)
                    np.testing.assert_array_equal(left, right)

    def test_input_checks_match_the_reference(self):
        from microstructure.ifw import derive_ifw
        good = np.full((2, 2), 0.3)
        cases = [
            (dict(fw=good, csf=np.full((2, 3), 0.1)), {}),
            (dict(fw=np.full((2, 2), 1.5), csf=good), {}),
            (dict(fw=good, csf=np.full((2, 2), np.nan)), {}),
            (dict(fw=good, csf=-good), {}),
            (dict(fw=good, csf=good), dict(echo_time_ms=-1.0)),
            (dict(fw=good, csf=good), dict(tissue_t2_ms=0.0)),
            (dict(fw=good, csf=good), dict(csf_t2_ms=float("nan"))),
            (dict(fw=good, csf=good), dict(echo_time_ms=1e6)),
        ]
        for arrays, override in cases:
            options = {**dict(echo_time_ms=89.5, tissue_t2_ms=95.0, csf_t2_ms=1500.0), **override}
            with self.assertRaises(ValueError) as ours:
                derive_ifw(arrays["fw"], arrays["csf"], **options)
            with self.assertRaises(ValueError) as theirs:
                reference_derive_ifw(arrays["fw"], arrays["csf"], **options)
            self.assertEqual(str(ours.exception), str(theirs.exception))


class EchoTimeSourceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def sidecar(self, payload):
        path = self.root / "data.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_user_value_wins_over_the_sidecar(self):
        from microstructure.ifw import resolve_echo_time
        self.assertEqual(resolve_echo_time(70.0, self.sidecar({"EchoTime": 0.0895})), (70.0, "user"))
        self.assertEqual(resolve_echo_time(0.0, self.sidecar({"EchoTime": 0.0895})), (0.0, "user"))

    def test_sidecar_converts_seconds_to_ms(self):
        from microstructure.ifw import resolve_echo_time, sidecar_echo_time_ms
        value, source = resolve_echo_time(None, self.sidecar({"EchoTime": 0.0895}))
        self.assertAlmostEqual(value, 89.5, places=9)
        self.assertEqual(source, "sidecar")
        self.assertIsNone(sidecar_echo_time_ms(self.sidecar({"EchoTime": "0.0895"})))
        self.assertIsNone(sidecar_echo_time_ms(self.sidecar({"EchoTime": -0.01})))
        self.assertIsNone(sidecar_echo_time_ms(self.sidecar({"RepetitionTime": 3.0})))

    def test_no_source_stops_with_the_message_and_has_no_default(self):
        from microstructure.ifw import EchoTimeMissing, IfwBlocked, resolve_echo_time
        for sidecar in (None, self.root / "absent.json", self.sidecar({})):
            with self.assertRaises(EchoTimeMissing) as caught:
                resolve_echo_time(None, sidecar)
            self.assertEqual(str(caught.exception), "iFW needs the echo time (ms): enter it in Microstructure")
        self.assertTrue(issubclass(EchoTimeMissing, IfwBlocked))


def _write_nifti(path, data, affine=None):
    import nibabel as nib
    image = nib.Nifti1Image(np.asarray(data), np.eye(4) if affine is None else affine)
    nib.save(image, str(path))
    return path


class IfwStageTest(unittest.TestCase):
    SHAPE = (4, 5, 6)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        rng = np.random.default_rng(7)
        self.fw = rng.random(self.SHAPE).astype(np.float32)
        self.csf = (rng.random(self.SHAPE) * 0.5).astype(np.float32)
        self.mask = np.zeros(self.SHAPE, dtype=np.uint8)
        self.mask[1:3, 1:4, 1:5] = 1
        (self.root / "bvals").write_text("0 0 1000 1000 2000 2000\n")
        (self.root / "bvecs").write_text("0 0 1 0 0 1\n0 0 0 1 0 0\n0 0 0 0 1 0\n")
        self.paths = dict(
            fw=_write_nifti(self.root / "fw.nii.gz", self.fw),
            csf=_write_nifti(self.root / "csf.nii.gz", self.csf),
            mask=_write_nifti(self.root / "mask.nii.gz", self.mask),
        )

    def config(self, **override):
        from microstructure.ifw import IfwConfig
        values = dict(subject_id="S1", fw_fraction_path=self.paths["fw"], csf_probability_path=self.paths["csf"],
                      mask_path=self.paths["mask"], bval_path=self.root / "bvals", bvec_path=self.root / "bvecs",
                      output_dir=self.root / "out", echo_time_ms=89.5)
        values.update(override)
        return IfwConfig(**values)

    def test_outputs_match_the_reference_masking_and_dtype(self):
        import nibabel as nib
        from microstructure.ifw import run_ifw
        result = run_ifw(self.config())
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.ifw_path.name, "S1_ifw.nii.gz")
        self.assertEqual(result.fw_volume_fraction_path.name, "S1_fw_volume_fraction.nii.gz")
        corrected, _, want = reference_derive_ifw(self.fw, self.csf, echo_time_ms=89.5, tissue_t2_ms=95.0,
                                                  csf_t2_ms=1500.0)
        want[self.mask == 0] = 0
        for path, expected in ((result.ifw_path, want), (result.fw_volume_fraction_path, corrected)):
            image = nib.load(str(path))
            self.assertEqual(image.get_data_dtype(), np.dtype(np.float32))
            np.testing.assert_array_equal(np.asarray(image.dataobj), expected.astype(np.float32))
        self.assertTrue(np.all(np.asarray(nib.load(str(result.ifw_path)).dataobj)[self.mask == 0] == 0))

    def test_saved_maps_keep_the_voxel_order_of_the_free_water_map(self):
        import nibabel as nib
        from microstructure.ifw import run_ifw
        affine = np.diag([-2.0, 2.0, 3.0, 1.0])
        affine[:3, 3] = [10.0, -20.0, 5.0]  # LAS-like, as the HCP data
        for key, data in (("fw", self.fw), ("csf", self.csf), ("mask", self.mask)):
            _write_nifti(self.paths[key], data, affine)
        result = run_ifw(self.config())
        saved = nib.load(str(result.ifw_path))
        np.testing.assert_allclose(saved.affine, affine)
        # The reference compares in canonical order: put the saved map through as_closest_canonical.
        canonical_fw = nib.as_closest_canonical(nib.load(str(self.paths["fw"])))
        canonical_csf = nib.as_closest_canonical(nib.load(str(self.paths["csf"])))
        canonical_mask = nib.as_closest_canonical(nib.load(str(self.paths["mask"])))
        _, _, want = reference_derive_ifw(canonical_fw.get_fdata(dtype=np.float64),
                                          canonical_csf.get_fdata(dtype=np.float64), echo_time_ms=89.5,
                                          tissue_t2_ms=95.0, csf_t2_ms=1500.0)
        want[canonical_mask.get_fdata() <= 0] = 0
        got = np.asarray(nib.as_closest_canonical(saved).dataobj)
        np.testing.assert_array_equal(got, want.astype(np.float32))

    def test_record_holds_te_source_t2_hashes_and_citation(self):
        import hashlib
        from microstructure.ifw import IFW_REFERENCE, run_ifw
        sidecar = self.root / "data.json"
        sidecar.write_text(json.dumps({"EchoTime": 0.0895}), encoding="utf-8")
        result = run_ifw(self.config(echo_time_ms=None, dwi_sidecar_path=sidecar))
        self.assertEqual((round(result.echo_time_ms, 9), result.echo_time_source), (89.5, "sidecar"))
        record = json.loads(result.record_path.read_text(encoding="utf-8"))
        self.assertEqual(record["echo_time_source"], "sidecar")
        self.assertAlmostEqual(record["echo_time_ms"], 89.5, places=9)
        self.assertEqual((record["tissue_t2_ms"], record["csf_t2_ms"]), (95.0, 1500.0))
        self.assertEqual(record["reference"], IFW_REFERENCE)
        expected = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in (
            ("fw_fraction", self.paths["fw"]), ("csf_probability", self.paths["csf"]), ("mask", self.paths["mask"]),
            ("bval", self.root / "bvals"), ("bvec", self.root / "bvecs"), ("dwi_sidecar", sidecar))}
        self.assertEqual(record["inputs_sha256"], expected)
        # A user value does not list the sidecar.
        user = run_ifw(self.config(output_dir=self.root / "out2", dwi_sidecar_path=sidecar))
        user_record = json.loads(user.record_path.read_text(encoding="utf-8"))
        self.assertEqual(user_record["echo_time_source"], "user")
        self.assertNotIn("dwi_sidecar", user_record["inputs_sha256"])

    def test_stage_stops_without_an_echo_time(self):
        from microstructure.ifw import EchoTimeMissing, run_ifw
        with self.assertRaisesRegex(EchoTimeMissing, r"iFW needs the echo time \(ms\): enter it in Microstructure"):
            run_ifw(self.config(echo_time_ms=None))
        self.assertFalse((self.root / "out" / "S1_ifw.nii.gz").exists())

    def test_grid_mismatch_is_an_error_not_a_resample(self):
        from microstructure.ifw import run_ifw
        shifted = np.eye(4)
        shifted[:3, 3] = [0.0, 0.0, 1e-3]
        _write_nifti(self.paths["csf"], self.csf, shifted)
        with self.assertRaisesRegex(ValueError, "Spatial mismatch: csf_probability"):
            run_ifw(self.config())
        _write_nifti(self.paths["csf"], self.csf[:, :, :5])
        with self.assertRaisesRegex(ValueError, "Spatial mismatch"):
            run_ifw(self.config())
        # Inside the 1e-5 tolerance the grids count as equal.
        close = np.eye(4)
        close[:3, 3] = [0.0, 0.0, 5e-6]
        _write_nifti(self.paths["csf"], self.csf, close)
        self.assertEqual(run_ifw(self.config()).status, "completed")

    def test_matching_outputs_are_reused_and_a_changed_input_or_te_reruns(self):
        from microstructure.ifw import run_ifw
        first = run_ifw(self.config())
        self.assertEqual(run_ifw(self.config()).status, "cached")
        self.assertEqual(run_ifw(self.config(echo_time_ms=60.0)).status, "completed")
        _write_nifti(self.paths["csf"], self.csf * 0.5)
        self.assertEqual(run_ifw(self.config(echo_time_ms=60.0)).status, "completed")
        self.assertEqual(first.status, "completed")

    def test_two_shell_data_is_ineligible_before_any_output(self):
        from microstructure.ifw import FREE_WATER_INELIGIBLE, FreeWaterIneligible, run_ifw
        (self.root / "bvals").write_text("0 0 1000 1000 1000 1000\n")
        with self.assertRaises(FreeWaterIneligible) as caught:
            run_ifw(self.config())
        self.assertEqual(str(caught.exception), FREE_WATER_INELIGIBLE)
        self.assertFalse((self.root / "out" / "S1_ifw.nii.gz").exists())

    def test_regional_table_includes_ifw(self):
        import csv
        from microstructure.ifw import run_ifw
        labels = np.zeros(self.SHAPE, dtype=np.int32)
        labels[1:3, 1:4, 1:3] = 7
        labels[1:3, 1:4, 3:5] = 9
        _write_nifti(self.root / "labels.nii.gz", labels)
        result = run_ifw(self.config(roi_label_map=self.root / "labels.nii.gz"))
        self.assertTrue(result.roi_stats_csv_path and result.roi_stats_csv_path.is_file())
        with result.roi_stats_csv_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        header = " ".join(rows[0])
        self.assertIn("ifw", header)
        self.assertIn("fw_volume_fraction", header)


class FreeWaterEligibilityTest(unittest.TestCase):
    def test_predicate_is_dipys_check_multi_b_with_three_groups_and_b0_counted(self):
        from dipy.core.gradients import check_multi_b
        from microstructure import ifw
        with mock.patch("dipy.core.gradients.check_multi_b", wraps=check_multi_b) as spy:
            eligible, message = ifw.free_water_eligibility([0, 0, 1000, 1000, 2000, 2000])
        self.assertTrue(eligible)
        self.assertEqual(message, "")
        args, kwargs = spy.call_args
        self.assertEqual(args[1], 3)
        self.assertIs(kwargs.get("non_zero"), False)

    def test_message_and_cases(self):
        from microstructure import ifw
        self.assertEqual(ifw.FREE_WATER_INELIGIBLE,
                         "Free water needs at least three distinct b-values, b=0 counted "
                         "(for example 0, 1000 and 2000 s/mm²)")
        for bvals, expected in (([0, 0, 1000, 1000], False), ([0, 1000, 2000], True), ([1000, 2000, 3000], True),
                                ([0, 5, 1000, 1005], False), ([0, 1000, 1000, 1990, 2000], True)):
            self.assertEqual(ifw.free_water_eligibility(bvals)[0], expected, bvals)

    def test_free_water_fit_stops_with_the_message_on_two_shells(self):
        from microstructure import pipeline
        from microstructure.ifw import FREE_WATER_INELIGIBLE
        root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        (root / "bvals").write_text("0 0 1000 1000\n")
        (root / "bvecs").write_text("0 0 1 0\n0 0 0 1\n0 0 0 0\n")
        config = SimpleNamespace(raw_input_bundle=SimpleNamespace(bval_path=root / "bvals", bvec_path=root / "bvecs"),
                                 free_water_fit_method="NLS", free_water_chunk_size=2, free_water_engine="serial",
                                 free_water_n_jobs=None, free_water_vox_per_chunk=None)
        with self.assertRaisesRegex(ValueError, "three distinct b-values"):
            pipeline._compute_free_water_maps_impl(config=config, paths={"free_water_progress": root / "progress.json"}, dwi_img=None)
        self.assertIn("three distinct b-values", FREE_WATER_INELIGIBLE)


class IfwClosureAndCliTest(unittest.TestCase):
    def test_closure_entries(self):
        from shared import target_closure as tc
        self.assertIn("microstructure_ifw", tc.TARGET_ORDER)
        self.assertEqual(tc.STATIC_REQUIREMENTS["microstructure_ifw"],
                         ("microstructure", "microstructure_freewater", "segmentation"))
        self.assertIn("microstructure_ifw", tc.MASK_TARGETS)
        self.assertEqual(tc.DISPLAY_NAMES["microstructure_ifw"], "iFW")
        for closure in (tc.expand_closure(("microstructure_ifw",)),
                        tc.expand_closure(("microstructure",), microstructure_ifw=True)):
            self.assertEqual(set(closure.targets),
                             {"microstructure", "microstructure_freewater", "microstructure_ifw", "segmentation"})
        added = tc.expand_closure(("microstructure_ifw",))
        self.assertEqual(set(added.required_by), {"microstructure", "microstructure_freewater", "segmentation"})
        self.assertIn("iFW", tc.closure_notice(added))
        self.assertNotIn("microstructure_ifw", tc.expand_closure(("microstructure",), microstructure_free_water=True).targets)
        skipped = tc.expand_closure(("microstructure_ifw",), excluded=("segmentation",))
        self.assertNotIn("segmentation", skipped.targets)

    def test_cli_stage_flags_and_echo_time(self):
        from shared.run_subject_pipeline_cli import build_parser, resolve_analysis_options
        parser = build_parser()
        base = ["--subject-root", "/inputs/S1"]
        args = parser.parse_args(base + ["--microstructure-stages", "ifw", "--ifw-echo-time-ms", "89.5"])
        resolve_analysis_options(args)
        self.assertTrue(args.run_ifw)
        self.assertTrue(args.run_free_water)  # iFW needs the Free Water fraction
        self.assertEqual((args.ifw_echo_time_ms, args.ifw_t2_gm_ms, args.ifw_t2_csf_ms), (89.5, 95.0, 1500.0))
        flag = parser.parse_args(base + ["--run-ifw"])
        resolve_analysis_options(flag)
        self.assertTrue(flag.run_ifw)
        self.assertIsNone(flag.ifw_echo_time_ms)  # no default echo time
        every = parser.parse_args(base + ["--microstructure-stages", "all"])
        resolve_analysis_options(every)
        self.assertFalse(every.run_ifw)  # "all" does not add iFW
        researcher = parser.parse_args(base + ["--run-ifw", "--ifw-t2-gm-ms", "80", "--ifw-t2-csf-ms", "1200"])
        resolve_analysis_options(researcher)
        self.assertEqual((researcher.ifw_t2_gm_ms, researcher.ifw_t2_csf_ms), (80.0, 1200.0))
        for bad in (["--ifw-echo-time-ms", "-1"], ["--ifw-t2-gm-ms", "0"], ["--ifw-t2-csf-ms", "-5"]):
            invalid = parser.parse_args(base + ["--run-ifw"] + bad)
            with self.assertRaises(ValueError):
                resolve_analysis_options(invalid)

    def test_cli_target_name_and_closure(self):
        from shared.run_subject_pipeline_cli import _resolve_targets, build_parser, resolve_analysis_options
        args = build_parser().parse_args(["--subject-root", "/inputs/S1", "--targets", "microstructure_ifw",
                                          "--tractography-backend", "dti_det"])
        resolve_analysis_options(args)
        self.assertEqual(set(_resolve_targets(args)),
                         {"microstructure", "microstructure_freewater", "microstructure_ifw", "segmentation"})

    def test_label_logic_and_layout_sites_know_ifw(self):
        import ast
        from shared.io_layout import subject_layout
        from shared.run_history import TARGET_LABELS
        self.assertEqual(TARGET_LABELS["microstructure_ifw"], "iFW")
        self.assertEqual(subject_layout(Path("/subject")).microstructure_ifw.name, "ifw")
        tree = ast.parse((REPO_ROOT / "shared/module_logic.py").read_text(encoding="utf-8"))
        wanted = {"microstructure_run_ifw", "microstructure_ifw_echo_time_ms", "microstructure_ifw_t2_gm_ms",
                  "microstructure_ifw_t2_csf_ms"}
        found = [{arg.arg for arg in node.args.kwonlyargs + node.args.args} for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef) and node.name in {"planSelectedProductOrchestration",
                                                                         "runSelectedProductOrchestration"}]
        self.assertEqual(len(found), 2)
        for names in found:
            self.assertTrue(wanted <= names)

    def test_controller_forwards_stage_and_echo_time(self):
        from shared.ui_run_controller import _append_ifw_args, microstructure_stages_arg
        kwargs = {"microstructure_run_free_water": True, "microstructure_run_ifw": True,
                  "microstructure_ifw_echo_time_ms": 89.5}
        self.assertEqual(microstructure_stages_arg(kwargs), "dti,freewater,ifw")
        args = []
        _append_ifw_args(args, kwargs)
        self.assertEqual(args, ["--ifw-echo-time-ms", "89.5"])
        args = []
        _append_ifw_args(args, {})
        self.assertEqual(args, [])

    def test_stage_order_keeps_all_without_ifw(self):
        from shared.user_config import MICROSTRUCTURE_STAGE_ORDER, canonical_microstructure_stages
        self.assertEqual(MICROSTRUCTURE_STAGE_ORDER[-1], "ifw")
        self.assertEqual(canonical_microstructure_stages("all"), ("dti", "freewater", "noddi", "dki"))
        self.assertEqual(canonical_microstructure_stages("ifw,freewater"), ("dti", "freewater", "ifw"))


class IfwOrchestratorPlanTest(unittest.TestCase):
    def setUp(self):
        from test_product_orchestrator import _write_valid_raw_subject
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.subject = Path(self.temporary.name) / "100307"
        _write_valid_raw_subject(self.subject)
        (self.subject / "input" / "bvals").write_text("0 1000 2000 2000")

    def plan(self, **override):
        from shared.orchestrator import ProductOrchestratorConfig, plan_subject_products
        values = dict(subject_id="100307", subject_root=self.subject, targets=("microstructure", "microstructure_ifw"),
                      microstructure_run_ifw=True, execute=False)
        values.update(override)
        return {item.target: item for item in plan_subject_products(ProductOrchestratorConfig(**values), write=False).items}

    def test_plan_adds_the_prerequisites_and_stops_without_te(self):
        items = self.plan()
        self.assertEqual(items["microstructure_ifw"].status, "blocked")
        self.assertEqual(items["microstructure_ifw"].reason, "iFW needs the echo time (ms): enter it in Microstructure")
        self.assertEqual(items["microstructure_freewater"].status, "will_run")  # other stages continue
        self.assertEqual(items["segmentation"].status, "will_run")

    def test_plan_runs_with_a_user_te_or_a_sidecar_te(self):
        self.assertEqual(self.plan(microstructure_ifw_echo_time_ms=89.5)["microstructure_ifw"].status, "will_run")
        (self.subject / "input" / "data.json").write_text(json.dumps({"EchoTime": 0.0895}), encoding="utf-8")
        self.assertEqual(self.plan()["microstructure_ifw"].status, "will_run")

    def test_two_shell_acquisition_blocks_free_water_and_ifw_with_the_message(self):
        from microstructure.ifw import FREE_WATER_INELIGIBLE
        (self.subject / "input" / "bvals").write_text("0 1000 1000 1000")
        items = self.plan(microstructure_ifw_echo_time_ms=89.5)
        self.assertEqual(items["microstructure_freewater"].status, "blocked")
        self.assertEqual(items["microstructure_freewater"].reason, FREE_WATER_INELIGIBLE)
        self.assertEqual(items["microstructure_ifw"].status, "blocked")
        self.assertEqual(items["microstructure"].status, "will_run")  # other stages continue

    def test_manifest_fields(self):
        from shared.orchestrator import ProductOrchestratorConfig, run_subject_products
        result = run_subject_products(ProductOrchestratorConfig(
            subject_id="100307", subject_root=self.subject, targets=("microstructure_ifw",),
            microstructure_run_ifw=True, microstructure_ifw_echo_time_ms=89.5, execute=False))
        manifest = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
        self.assertTrue(manifest["microstructure_run_ifw"])
        self.assertEqual(manifest["microstructure_ifw_echo_time_ms"], 89.5)
        self.assertEqual(manifest["microstructure_ifw_t2_gm_ms"], 95.0)
        self.assertIn("ifw", manifest["microstructure_stages"])
        self.assertIn("microstructure.ifw", {task.name for task in result.tasks})


if __name__ == "__main__":
    unittest.main()
