#!/usr/bin/env python3
"""Package B1: the Qt-free module that says which algorithms this DATA cannot run.

The main property is agreement with the run planner: the GUI must never grey out an algorithm that the planner
would run. Every rule here is checked against ``shared.orchestrator`` or ``microstructure.dki`` on the same files.
"""

from __future__ import annotations

import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import nibabel as nib
import numpy as np

from shared import algorithm_availability as availability
from shared.algorithm_availability import Block, data_blocks
from test_microstructure_dki import AFFINE, synthetic_case, write_case

BVAL_FIXTURES = {
    "single_shell": "0 0 1000 1000 1000 1000",
    "two_shells": "0 0 1000 1000 2000 2000",
    "three_groups": "0 0 1000 1000 2000 2000 3000 3000",
    "no_b0_two_shells": "1000 1000 2000 2000",
    "no_b0_single_shell": "1000 1000 1000",
    "b0_only": "0 0 0 5",
    "rounding_edge": "0 1049 1049 2049 2049",
    "unreadable": "0 abc 1000 xyz",
    "empty": "",
}


# F5: the DKI fixture defects of ``DkiAgreementTest._case`` -> (code, numbers that the Chinese text names).
DKI_CODES = {
    "single": ("dki_needs_two_shells", {"shells": 1, "max_bvalue": 2500}),
    "no_b0": ("dki_needs_b0", {}),
    "rank": ("dki_shell_directions", {"bvalue": 1000, "directions": 1, "rank": 1, "min_directions": 15,
                                      "min_rank": 15}),
    "nan": ("dki_gradients_invalid", {}),
    "nonunit": ("dki_bvectors_not_unit", {"tolerance": 0.01}),
    "few_volumes": ("dki_needs_4d_volumes", {"dimensions": 4, "volumes": 10, "min_volumes": 22}),
}


def bval_inputs(root: Path, text: str | None, name: str = "case") -> dict[str, Path]:
    """The planner's input shape (``dwi``, ``bval``, ``bvec``, ``mask``); only the bvals file is written."""

    folder = Path(root) / name / "input"
    folder.mkdir(parents=True, exist_ok=True)
    inputs = {"dwi": folder / "data.nii.gz", "bval": folder / "bvals", "bvec": folder / "bvecs",
              "mask": folder / "nodif_brain_mask.nii.gz"}
    if text is not None:
        inputs["bval"].write_text(text, encoding="utf-8")
    return inputs


class PlannerAgreementTest(unittest.TestCase):
    """B1: ``data_blocks`` blocks exactly when the planner's own check blocks (and never raises)."""

    def setUp(self) -> None:
        availability.clear_cache()
        self.addCleanup(availability.clear_cache)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_b1_noddi_agrees_with_the_planner_on_every_bvals_fixture(self) -> None:
        from shared.orchestrator import _noddi_ineligible

        for name, text in {**BVAL_FIXTURES, "missing_file": None}.items():
            with self.subTest(fixture=name):
                inputs = bval_inputs(self.tmp.name, text, name)
                self.assertEqual("noddi" in data_blocks(inputs), bool(_noddi_ineligible(inputs)))

    def test_b1_free_water_and_ifw_agree_with_the_planner_on_every_bvals_fixture(self) -> None:
        from shared.orchestrator import _free_water_ineligible

        for name, text in {**BVAL_FIXTURES, "missing_file": None}.items():
            with self.subTest(fixture=name):
                inputs = bval_inputs(self.tmp.name, text, name)
                try:
                    planner_blocks = bool(_free_water_ineligible(inputs))
                except Exception:  # the planner itself fails on this file; the GUI must simply not block
                    blocks = data_blocks(inputs)
                    self.assertNotIn("freewater", blocks)
                    self.assertNotIn("ifw", blocks)
                    continue
                blocks = data_blocks(inputs)
                self.assertEqual("freewater" in blocks, planner_blocks)
                self.assertEqual("ifw" in blocks, planner_blocks)

    def test_b1_the_known_fixtures_have_the_expected_verdicts(self) -> None:
        verdict = {}
        for name, text in BVAL_FIXTURES.items():
            verdict[name] = set(data_blocks(bval_inputs(self.tmp.name, text, name)))
        self.assertEqual(verdict["single_shell"], {"noddi", "freewater", "ifw"})
        self.assertEqual(verdict["two_shells"], set())  # b0 plus two shells are three b-value groups
        self.assertEqual(verdict["no_b0_two_shells"], {"freewater", "ifw"})  # two groups; NODDI needs only the shells
        self.assertEqual(verdict["three_groups"], set())
        self.assertEqual(verdict["no_b0_single_shell"], {"noddi", "freewater", "ifw"})
        self.assertIn("noddi", verdict["empty"])  # the planner blocks NODDI on an empty file (0 shells)
        self.assertEqual(verdict["unreadable"], set())  # unreadable b-values mean unknown

    def test_b1_unknown_inputs_give_no_block(self) -> None:
        self.assertEqual(data_blocks({}), {})
        self.assertEqual(data_blocks({"bval": None}), {})
        self.assertEqual(data_blocks(bval_inputs(self.tmp.name, None, "none")), {})
        self.assertEqual(data_blocks({"bval": Path(self.tmp.name) / "no" / "bvals"}), {})
        self.assertEqual(data_blocks(None), {})

    def test_b1_a_rule_that_raises_never_reaches_the_caller_and_is_logged_once(self) -> None:
        inputs = bval_inputs(self.tmp.name, BVAL_FIXTURES["single_shell"], "boom")
        with mock.patch("microstructure.ifw.free_water_eligibility", side_effect=RuntimeError("boom")), \
                self.assertLogs("shared.algorithm_availability", level="WARNING") as logged:
            first = data_blocks(inputs)
            (Path(inputs["bval"])).write_text("0 0 1000 1000 1000 1000 ", encoding="utf-8")  # new signature, same failure
            second = data_blocks(inputs)
        self.assertIn("noddi", first)  # the NODDI rule is isolated from the failing free-water rule
        self.assertNotIn("freewater", first)
        self.assertNotIn("freewater", second)
        self.assertEqual(len([line for line in logged.output if "boom" in line]), 1)

    def test_b1_block_has_a_stable_code_numbers_and_both_languages(self) -> None:
        blocks = data_blocks(bval_inputs(self.tmp.name, BVAL_FIXTURES["single_shell"], "text"))
        noddi, freewater, ifw = blocks["noddi"], blocks["freewater"], blocks["ifw"]
        self.assertIsInstance(noddi, Block)
        self.assertEqual(noddi.code, "noddi_needs_two_shells")
        self.assertEqual(noddi.numbers["count"], 1)
        self.assertEqual(noddi.numbers["shells"], [1000])
        self.assertEqual(freewater.code, ifw.code)
        self.assertEqual(freewater.code, "free_water_needs_three_b_values")
        from shared.orchestrator import _noddi_ineligible

        planner_text = _noddi_ineligible(bval_inputs(self.tmp.name, BVAL_FIXTURES["single_shell"], "text"))
        self.assertTrue(noddi.text("en").startswith(planner_text.split(" Deselect")[0]))
        self.assertIn("at least 2 non-zero b-shells", noddi.text("en"))
        self.assertIn("1000", noddi.text("en"))
        self.assertRegex(noddi.text("zh"), r"[一-鿿]")
        self.assertIn("1000", noddi.text("zh"))
        from microstructure.ifw import FREE_WATER_INELIGIBLE

        self.assertEqual(freewater.text("en"), FREE_WATER_INELIGIBLE)
        self.assertRegex(freewater.text("zh"), r"[一-鿿]")
        self.assertEqual(noddi.text("zh-CN"), noddi.text("zh"))
        self.assertEqual(noddi.text("anything"), noddi.text("en"))

    def test_b1_cache_reads_the_bvals_file_once_for_unchanged_files(self) -> None:
        inputs = bval_inputs(self.tmp.name, BVAL_FIXTURES["single_shell"], "cache")
        real = np.loadtxt
        reads: list[str] = []

        def counting(fname, *args, **kwargs):
            reads.append(str(fname))
            return real(fname, *args, **kwargs)

        with mock.patch("numpy.loadtxt", counting):
            first = data_blocks(inputs)
            second = data_blocks(inputs)
            self.assertEqual(first.keys(), second.keys())
            self.assertEqual(reads.count(str(inputs["bval"])), 1)
            inputs["bval"].write_text(BVAL_FIXTURES["three_groups"], encoding="utf-8")
            third = data_blocks(inputs)
        self.assertEqual(third, {})
        self.assertEqual(reads.count(str(inputs["bval"])), 2)  # a changed file is read again

    def test_b1_the_result_is_a_copy_the_caller_may_change(self) -> None:
        inputs = bval_inputs(self.tmp.name, BVAL_FIXTURES["single_shell"], "copy")
        data_blocks(inputs).clear()
        self.assertIn("noddi", data_blocks(inputs))


class DkiAgreementTest(unittest.TestCase):
    """B1: DKI is blocked for the acquisition only, and the data rule is the one ``inspect_dki_inputs`` runs."""

    def setUp(self) -> None:
        availability.clear_cache()
        self.addCleanup(availability.clear_cache)

    def _case(self, defect: str | None, fit_method: str = "WLS"):
        from microstructure.dki import DkiConfig
        from shared.pipeline import RawInputBundle

        case = synthetic_case()
        if defect == "single":
            case["bvals"][case["bvals"] > 0] = 1000
        elif defect == "no_b0":
            case["bvals"][:4] = 1000
            case["bvecs"][:4] = [1, 0, 0]
        elif defect == "rank":
            case["bvecs"][4:] = [1, 0, 0]
        elif defect == "nan":
            case["bvecs"][6, 0] = np.nan
        elif defect == "nonunit":
            case["bvecs"][6] *= 2
        elif defect == "few_volumes":
            for key in ("bvals", "bvecs"):
                case[key] = case[key][:10]
            case["data"] = case["data"][..., :10]
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        paths = write_case(root, case)
        bundle = RawInputBundle(root, "synthetic", root / "input", **{f"{key}_path": value for key, value in paths.items()})
        return paths, DkiConfig("synthetic", bundle, root / "outputs/dki", fit_method=fit_method)

    def test_b1_dki_blocks_exactly_when_the_planner_inspection_rejects_the_acquisition(self) -> None:
        from microstructure.dki import inspect_dki_inputs

        for defect in (None, "single", "no_b0", "rank", "nan", "nonunit", "few_volumes"):
            with self.subTest(defect=defect):
                paths, config = self._case(defect)
                eligible = inspect_dki_inputs(config, require_mask=False)["eligible"]
                blocks = data_blocks(paths)
                self.assertEqual("dki" in blocks, not eligible)
                if defect is not None:
                    self.assertEqual(blocks["dki"].code, DKI_CODES[defect][0])  # F5: one stable code per failure
                    self.assertTrue(blocks["dki"].numbers["detail"])
                    self.assertIn(blocks["dki"].numbers["detail"], blocks["dki"].text("en"))
                    self.assertRegex(blocks["dki"].text("zh"), r"[一-鿿]")

    def test_b1_dki_is_not_blocked_for_a_backend_or_option_problem(self) -> None:
        from microstructure.dki import inspect_dki_inputs

        paths, config = self._case(None, fit_method="MKCURVE_MATLAB")
        self.assertFalse(inspect_dki_inputs(config, require_mask=False)["eligible"])  # the planner blocks: no backend
        with mock.patch("microstructure.mkcurve.inspect_mkcurve_backend", side_effect=AssertionError("backend asked")), \
                mock.patch("microstructure.mkcurve_matlab.inspect_mkcurve_backend", side_effect=AssertionError("asked")):
            self.assertNotIn("dki", data_blocks(paths))

    def test_b1_dki_ignores_the_mask_and_unreadable_or_missing_files(self) -> None:
        paths, _config = self._case(None)
        paths["mask"].write_bytes(b"not a nifti")  # a mask problem is the planner's, not an acquisition block
        self.assertNotIn("dki", data_blocks(paths))
        availability.clear_cache()
        paths["bvec"].write_text("a b c\n", encoding="utf-8")  # unreadable gradients: unknown
        self.assertNotIn("dki", data_blocks(paths))
        availability.clear_cache()
        paths["bvec"].unlink()  # a missing file is reported by the planner's input check
        self.assertNotIn("dki", data_blocks(paths))
        availability.clear_cache()
        paths["dwi"].write_bytes(b"garbage")
        self.assertNotIn("dki", data_blocks(paths))

    def test_b1_the_dki_inspection_result_is_unchanged_by_the_split(self) -> None:
        """The refactor of ``_load_protocol`` keeps the keys, their order and the reasons."""
        from microstructure.dki import inspect_dki_inputs

        paths, config = self._case(None)
        info = inspect_dki_inputs(config)
        self.assertTrue(info["eligible"])
        self.assertEqual(
            list(info),
            ["eligible", "available", "fit_method", "reasons", "shape", "selected_volumes", "excluded_high_b_volumes",
             "upper_bvalue_policy", "b0_count", "baseline_bvectors_cleared_in_working_table", "shells", "design_rank",
             "design_condition_scaled", "mask_voxels", "geometry_pass", "warnings"],
        )
        for defect, fragment in (("single", "ValueError: DKI requires at least two nonzero shells"),
                                 ("no_b0", "ValueError: DKI requires at least one b0 measurement"),
                                 ("nonunit", "ValueError: DKI diffusion b-vectors must be unit length"),
                                 ("few_volumes", "ValueError: DKI requires a 4D DWI with at least 22 measurements")):
            _paths, bad = self._case(defect)
            reasons = inspect_dki_inputs(bad)["reasons"]
            self.assertTrue(reasons and reasons[0].startswith(fragment), (defect, reasons))

    def test_f5_each_dki_acquisition_failure_has_its_code_numbers_and_a_chinese_text(self) -> None:
        """F5: a stable code per DKI failure, its numbers, and a Chinese text without the English exception text."""
        for defect, (code, numbers) in DKI_CODES.items():
            with self.subTest(defect=defect):
                paths, _config = self._case(defect)
                block = data_blocks(paths)["dki"]
                self.assertEqual(block.code, code)
                for key, value in numbers.items():
                    self.assertEqual(block.numbers[key], value, key)
                detail = block.numbers["detail"]
                self.assertEqual(block.text("en"), f"This DWI does not meet the DKI acquisition requirements: {detail}")
                zh = block.text("zh")
                self.assertRegex(zh, r"[一-鿿]")
                self.assertNotIn(detail, zh)
                self.assertNotIn("DKI requires", zh)
                for value in numbers.values():
                    self.assertIn(f"{value:g}", zh)

    def test_f5_the_planner_messages_and_their_order_are_unchanged(self) -> None:
        """F5: the codes ride on plain ``ValueError``; ``inspect_dki_inputs`` reports the same English text."""
        from microstructure.dki import inspect_dki_inputs

        expected = {
            "single": "ValueError: DKI requires at least two nonzero shells after the maximum-b selection",
            "no_b0": "ValueError: DKI requires at least one b0 measurement",
            "nan": "ValueError: DKI gradients must be finite and b-values nonnegative",
            "nonunit": "ValueError: DKI diffusion b-vectors must be unit length within 0.01",
            "few_volumes": "ValueError: DKI requires a 4D DWI with at least 22 measurements",
            "rank": "ValueError: DKI requires at least 15 independent directions and angular design rank 15 per shell; "
                    "shell 1000 has 1 directions and rank 1",
        }
        for defect, reason in expected.items():
            with self.subTest(defect=defect):
                _paths, config = self._case(defect)
                self.assertEqual(inspect_dki_inputs(config, require_mask=False)["reasons"], [reason])

    def test_f5_the_header_and_count_codes_format_in_both_languages(self) -> None:
        """F5: the codes that the file fixtures do not reach (not 4D, singular affine, count mismatch)."""
        from microstructure.dki import dki_acquisition_issue

        case = synthetic_case()
        shape = case["data"].shape
        cases = {
            "not_4d": (SimpleNamespace(shape=shape[:3], affine=AFFINE), case["bvals"], case["bvecs"],
                       "dki_needs_4d_volumes", {"dimensions": 3, "volumes": None}, "3D"),
            "affine": (SimpleNamespace(shape=shape, affine=np.zeros((4, 4))), case["bvals"], case["bvecs"],
                       "dki_affine_invalid", {}, "仿射"),
            "count": (SimpleNamespace(shape=shape, affine=AFFINE), case["bvals"][:-1], case["bvecs"],
                      "dki_gradient_count_mismatch", {"volumes": shape[3], "bvals": shape[3] - 1}, f"{shape[3] - 1}"),
        }
        for name, (image, bvals, vectors, code, numbers, fragment) in cases.items():
            with self.subTest(case=name):
                issue = dki_acquisition_issue(image, bvals, vectors)
                self.assertIs(type(issue), ValueError)
                self.assertEqual(issue.code, code)
                for key, value in numbers.items():
                    self.assertEqual(issue.numbers[key], value, key)
                block = Block("dki", issue.code, {**issue.numbers, "detail": str(issue)})
                self.assertIn(str(issue), block.text("en"))
                zh = block.text("zh")
                self.assertIn(fragment, zh)
                self.assertNotIn(str(issue), zh)

    def test_f5_a_value_error_without_a_code_keeps_the_generic_code_and_a_chinese_text(self) -> None:
        paths, _config = self._case(None)
        with mock.patch("microstructure.dki._check_acquisition", side_effect=ValueError("dipy says no")):
            block = data_blocks(paths)["dki"]
        self.assertEqual(block.code, "dki_acquisition")
        self.assertIn("dipy says no", block.text("en"))
        self.assertRegex(block.text("zh"), r"[一-鿿]")
        self.assertNotIn("dipy says no", block.text("zh"))

    def test_b1_dki_acquisition_problem_returns_the_message_or_empty(self) -> None:
        from microstructure.dki import dki_acquisition_problem

        case = synthetic_case()
        image = SimpleNamespace(shape=case["data"].shape, affine=AFFINE)
        self.assertEqual(dki_acquisition_problem(image, case["bvals"], case["bvecs"]), "")
        bad = case["bvals"].copy()
        bad[bad > 0] = 1000
        self.assertIn("at least two nonzero shells", dki_acquisition_problem(image, bad, case["bvecs"]))
        flat = SimpleNamespace(shape=(2, 3, 2), affine=AFFINE)
        self.assertIn("4D DWI", dki_acquisition_problem(flat, case["bvals"], case["bvecs"]))


if __name__ == "__main__":
    unittest.main()
