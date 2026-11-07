#!/usr/bin/env python3
"""Tests for GUI topup / distortion-correction parameter validation and mapping."""

from __future__ import annotations

import dataclasses
import tempfile
import unittest
from pathlib import Path

from shared.orchestrator import ProductOrchestratorConfig
from shared.run_subject_pipeline_cli import build_parser
from shared.ui_run_controller import (
    RunSelection,
    build_background_cli_args,
    build_batch_background_cli_args,
    build_batch_run_kwargs,
    build_single_subject_run_kwargs,
)
from shared.topup_parameters import (
    PHASE_ENCODING_CHOICES,
    PHASE_ENCODING_DIRECTIONS,
    READOUT_TIME_MAX_S,
    READOUT_TIME_MIN_S,
    TopupParameters,
    normalize_phase_encoding_direction,
    parse_readout_time,
    route_uses_topup,
    topup_cli_args,
    topup_orchestrator_kwargs,
    topup_summary_text,
    validate_topup_parameters,
)


class PhaseEncodingTest(unittest.TestCase):
    def test_allowed_values_match_mrtrix_designators(self) -> None:
        self.assertEqual(
            set(PHASE_ENCODING_DIRECTIONS),
            {"i", "i-", "j", "j-", "k", "k-", "AP", "PA", "LR", "RL", "IS", "SI"},
        )
        self.assertEqual(PHASE_ENCODING_CHOICES[0], ("", "Not set"))
        self.assertEqual({value for value, _ in PHASE_ENCODING_CHOICES[1:]}, set(PHASE_ENCODING_DIRECTIONS))

    def test_normalisation_is_case_insensitive_and_canonical(self) -> None:
        self.assertEqual(normalize_phase_encoding_direction("ap"), "AP")
        self.assertEqual(normalize_phase_encoding_direction(" J- "), "j-")
        self.assertIsNone(normalize_phase_encoding_direction(""))
        self.assertIsNone(normalize_phase_encoding_direction(None))

    def test_unknown_direction_is_rejected(self) -> None:
        for bad in ("y", "j+", "0,1,0", "anterior"):
            with self.assertRaisesRegex(ValueError, "not recognised", msg=bad):
                normalize_phase_encoding_direction(bad)


class ReadoutTimeTest(unittest.TestCase):
    def test_boundaries(self) -> None:
        self.assertEqual(parse_readout_time(READOUT_TIME_MIN_S), READOUT_TIME_MIN_S)
        self.assertEqual(parse_readout_time(READOUT_TIME_MAX_S), READOUT_TIME_MAX_S)
        for bad in (READOUT_TIME_MIN_S / 2, READOUT_TIME_MAX_S + 0.001):
            with self.assertRaisesRegex(ValueError, "outside the accepted range", msg=repr(bad)):
                parse_readout_time(bad)

    def test_unset_values(self) -> None:
        for unset in (None, 0, 0.0, "", "  "):
            self.assertIsNone(parse_readout_time(unset), repr(unset))

    def test_text_input_and_units(self) -> None:
        self.assertAlmostEqual(parse_readout_time("0.0646"), 0.0646)
        self.assertAlmostEqual(parse_readout_time("0,0646"), 0.0646)
        self.assertAlmostEqual(parse_readout_time("0.05 s"), 0.05)
        with self.assertRaisesRegex(ValueError, "milliseconds"):
            parse_readout_time("64.6 ms")
        with self.assertRaisesRegex(ValueError, "milliseconds"):
            parse_readout_time(64.6)

    def test_invalid_numbers(self) -> None:
        for bad in ("abc", -0.05, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError, msg=repr(bad)):
                parse_readout_time(bad)


class ValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.b0 = self.root / "sub-01_dir-PA_epi.nii.gz"
        self.b0.write_bytes(b"\x00")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_route_gate(self) -> None:
        self.assertTrue(route_uses_topup(True))
        self.assertFalse(route_uses_topup(False))

    def test_unselected_preprocessing_ignores_fields_and_forwards_nothing(self) -> None:
        result = validate_topup_parameters(
            preprocessing_selected=False,
            phase_encoding_direction="nonsense",
            readout_time=-3,
            reverse_phase_b0="/does/not/exist.nii.gz",
        )
        self.assertTrue(result.ok)
        self.assertTrue(result.parameters.is_empty)
        self.assertEqual(topup_cli_args(result.parameters), [])

    def test_valid_mrtrix_fsl_inputs(self) -> None:
        result = validate_topup_parameters(
            preprocessing_selected=True,
            phase_encoding_direction="ap",
            readout_time=0.0646,
            reverse_phase_b0=str(self.b0),
        )
        self.assertTrue(result.ok, result.errors)
        self.assertEqual(result.parameters, TopupParameters("AP", 0.0646, self.b0))
        self.assertEqual(result.warnings, ())

    def test_missing_and_bad_reverse_b0(self) -> None:
        missing = validate_topup_parameters(
            preprocessing_selected=True, phase_encoding_direction="j-",
            reverse_phase_b0=str(self.root / "nope.nii.gz"),
        )
        self.assertIn("not found", missing.errors[0])
        folder = validate_topup_parameters(
            preprocessing_selected=True, phase_encoding_direction="j-",
            reverse_phase_b0=str(self.root),
        )
        self.assertIn("must be a file", folder.errors[0])
        text_file = self.root / "notes.txt"
        text_file.write_text("x")
        wrong_type = validate_topup_parameters(
            preprocessing_selected=True, phase_encoding_direction="j-",
            reverse_phase_b0=str(text_file),
        )
        self.assertIn("not a NIfTI", wrong_type.errors[0])

    def test_batch_mode_blocks_single_reverse_b0(self) -> None:
        result = validate_topup_parameters(
            preprocessing_selected=True, phase_encoding_direction="j-",
            reverse_phase_b0=str(self.b0), batch_mode=True,
        )
        self.assertFalse(result.ok)
        self.assertIn("batch mode", result.errors[0].lower())

    def test_warnings_and_notes_for_incomplete_but_runnable_inputs(self) -> None:
        result = validate_topup_parameters(preprocessing_selected=True)
        self.assertTrue(result.ok)
        self.assertIn("-pe_dir", result.warnings[0])
        self.assertTrue(any("0.1 s" in note for note in result.notes))
        self.assertTrue(any("-rpe_none" in note for note in result.notes))
        long_readout = validate_topup_parameters(
            preprocessing_selected=True, phase_encoding_direction="PA", readout_time=0.5
        )
        self.assertTrue(long_readout.ok)
        self.assertIn("longer than usual", long_readout.warnings[0])
        self.assertIn("Warning:", topup_summary_text(long_readout))

    def test_errors_are_collected_and_summarised(self) -> None:
        result = validate_topup_parameters(
            preprocessing_selected=True, phase_encoding_direction="x", readout_time=5,
        )
        self.assertEqual(len(result.errors), 2)
        self.assertTrue(topup_summary_text(result).startswith("Fix before running:"))


class CliWiringTest(unittest.TestCase):
    """The GUI must set exactly what the command line sets."""

    BASE = ["--subject-root", "/tmp/sub-01", "--subject-id", "sub-01"]

    def test_cli_args_round_trip_through_real_parser(self) -> None:
        parameters = TopupParameters("j-", 0.0959097, Path("/data/sub-01/fmap/sub-01_dir-PA_epi.nii.gz"))
        args = build_parser().parse_args(self.BASE + topup_cli_args(parameters))
        expected = topup_orchestrator_kwargs(parameters)
        self.assertEqual(args.preprocessing_phase_encoding_direction, expected["preprocessing_phase_encoding_direction"])
        self.assertEqual(args.preprocessing_readout_time, expected["preprocessing_readout_time"])
        self.assertEqual(args.preprocessing_reverse_phase_b0, expected["preprocessing_reverse_phase_b0"])

    def test_unset_values_emit_no_flags_and_keep_cli_defaults(self) -> None:
        self.assertEqual(topup_cli_args(TopupParameters()), [])
        args = build_parser().parse_args(self.BASE)
        self.assertEqual(
            topup_orchestrator_kwargs(TopupParameters()),
            {
                "preprocessing_reverse_phase_b0": args.preprocessing_reverse_phase_b0,
                "preprocessing_phase_encoding_direction": args.preprocessing_phase_encoding_direction,
                "preprocessing_readout_time": args.preprocessing_readout_time,
            },
        )

    def test_orchestrator_config_has_the_mapped_fields_with_none_defaults(self) -> None:
        fields = {field.name: field for field in dataclasses.fields(ProductOrchestratorConfig)}
        for key in topup_orchestrator_kwargs(TopupParameters()):
            self.assertIn(key, fields)
            self.assertIsNone(fields[key].default)



class RunControllerWiringTest(unittest.TestCase):
    """GUI selection -> orchestration kwargs -> background CLI args."""

    TOPUP = TopupParameters("PA", 0.0646, Path("/data/sub-01/fmap/b0_PA.nii.gz"))

    def _selection(self, topup: TopupParameters) -> RunSelection:
        return RunSelection(
            subject_id="sub-01",
            subject_root_text="/data/sub-01",
            output_root_text="",
            selected_targets=("validation", "preprocessing"),
            run_name="GUI1",
            tractography_backend="csd_act",
            bundle_backend="wmql",
            microstructure_freewater=False,
            microstructure_noddi=False,
            select_streamlines=1000000,
            ddseg_device="auto",
            surface_device="auto",
            connectome_backend="python",
            connectome_compress_mrtrix_dense=False,
            topup=topup,
        )

    def _repo(self, tmp: str) -> Path:
        repo = Path(tmp)
        (repo / "shared").mkdir()
        (repo / "shared" / "run_subject_pipeline_cli.py").write_text("", encoding="utf-8")
        (repo / "scripts").mkdir()
        return repo

    def test_single_subject_kwargs_and_cli_carry_all_three_values(self) -> None:
        kwargs = build_single_subject_run_kwargs(self._selection(self.TOPUP), {}, preflight_blocked_targets={})
        self.assertEqual(kwargs["preprocessing_phase_encoding_direction"], "PA")
        self.assertEqual(kwargs["preprocessing_readout_time"], 0.0646)
        self.assertEqual(kwargs["preprocessing_reverse_phase_b0"], self.TOPUP.reverse_phase_b0)
        with tempfile.TemporaryDirectory() as tmp:
            args = build_background_cli_args(kwargs, {}, repo_root=self._repo(tmp))
        cli_part = args[args.index("--subject-root"):]
        parsed = build_parser().parse_args(cli_part)
        self.assertEqual(parsed.preprocessing_phase_encoding_direction, "PA")
        self.assertEqual(parsed.preprocessing_readout_time, 0.0646)
        self.assertEqual(parsed.preprocessing_reverse_phase_b0, self.TOPUP.reverse_phase_b0)

    def test_unset_topup_leaves_cli_untouched(self) -> None:
        kwargs = build_single_subject_run_kwargs(self._selection(TopupParameters()), {}, preflight_blocked_targets={})
        with tempfile.TemporaryDirectory() as tmp:
            args = build_background_cli_args(kwargs, {}, repo_root=self._repo(tmp))
        self.assertFalse([arg for arg in args if "phase" in arg or "readout" in arg])

    def test_batch_forwards_protocol_values_but_never_a_reverse_b0(self) -> None:
        kwargs = build_batch_run_kwargs(self._selection(self.TOPUP), batch_root=Path("/data/study"))
        self.assertIsNone(kwargs["preprocessing_reverse_phase_b0"])
        with tempfile.TemporaryDirectory() as tmp:
            args = build_batch_background_cli_args(
                kwargs, {}, repo_root=self._repo(tmp), batch_root=Path("/data/study"), batch_size=1,
                arrange_inputs="none",
            )
        self.assertEqual(args[args.index("--preprocessing-phase-encoding-direction") + 1], "PA")
        self.assertEqual(float(args[args.index("--preprocessing-readout-time") + 1]), 0.0646)
        self.assertNotIn("--preprocessing-reverse-phase-b0", args)


if __name__ == "__main__":
    unittest.main()
