#!/usr/bin/env python3
"""W2b (T1 D1 part 2): phase-encoding and readout resolution, and the dwifslpreproc command."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import pe_resolution as pe  # noqa: E402


def affine_from_axes(i_axis, j_axis, k_axis):
    """Affine whose columns are the given world vectors (x, y, z), unit voxels."""
    return [[i_axis[r], j_axis[r], k_axis[r], 0.0] for r in range(3)] + [[0.0, 0.0, 0.0, 1.0]]


RAS = affine_from_axes((1, 0, 0), (0, 1, 0), (0, 0, 1))
LAS = affine_from_axes((-1, 0, 0), (0, 1, 0), (0, 0, 1))
# The first voxel axis runs along world y; the second along world x.
PERMUTED = affine_from_axes((0, 1, 0), (1, 0, 0), (0, 0, 1))
# The first voxel axis is 45 degrees between x and y: oblique.
OBLIQUE = affine_from_axes((1, 1, 0), (-1, 1, 0), (0, 0, 1))


class ConversionTest(unittest.TestCase):
    """W2b: the conversion rule of D1, with its four tests."""

    def test_w2b_las_file_with_j_minus_is_ap(self):
        # LAS: i runs toward the left, j toward anterior. j- is toward posterior, so the PE runs A to P.
        self.assertEqual(pe.axis_code_to_designator("j-", LAS), "AP")

    def test_w2b_ras_file_with_i_is_lr(self):
        self.assertEqual(pe.axis_code_to_designator("i", RAS), "LR")

    def test_w2b_file_whose_first_voxel_axis_is_anatomical_y(self):
        # i runs along +y: i is PA; i- is AP; j runs along +x: j is LR.
        self.assertEqual(pe.axis_code_to_designator("i", PERMUTED), "PA")
        self.assertEqual(pe.axis_code_to_designator("i-", PERMUTED), "AP")
        self.assertEqual(pe.axis_code_to_designator("j", PERMUTED), "LR")

    def test_w2b_oblique_axis_is_refused_with_the_rule4_message(self):
        self.assertIsNone(pe.axis_code_to_designator("i", OBLIQUE))
        result = pe.resolve_acquisition(user_phase_encoding="i", affine=OBLIQUE)
        self.assertIsNone(result.phase_encoding)
        self.assertEqual(result.error, pe.RULE4_MESSAGE)
        with self.assertRaisesRegex(pe.PhaseEncodingNeeded, "Phase-encoding direction needed"):
            result.require_phase_encoding()

    def test_w2b_z_axis_and_sign_convention(self):
        self.assertEqual(pe.axis_code_to_designator("k", RAS), "IS")
        self.assertEqual(pe.axis_code_to_designator("k-", RAS), "SI")
        self.assertEqual(pe.axis_code_to_designator("j", RAS), "PA")
        self.assertEqual(pe.axis_code_to_designator("i-", RAS), "RL")


class PhaseEncodingRulesTest(unittest.TestCase):
    def test_w2b_rule1_user_value_wins_and_a_nifti_code_is_converted(self):
        result = pe.resolve_acquisition(user_phase_encoding="j-", affine=LAS, sidecar={"PhaseEncodingDirection": "j-"})
        self.assertEqual((result.phase_encoding, result.phase_encoding_source), ("AP", "user"))
        self.assertFalse(result.pe_sign_assumed)
        self.assertEqual(pe.resolve_acquisition(user_phase_encoding="pa").phase_encoding, "PA")

    def test_w2b_rule2_sidecar_direction_is_converted_with_the_affine(self):
        result = pe.resolve_acquisition(sidecar={"PhaseEncodingDirection": "i"}, affine=PERMUTED)
        self.assertEqual((result.phase_encoding, result.phase_encoding_source), ("PA", "sidecar"))
        self.assertEqual(result.phase_encoding_sidecar_raw, "i")
        self.assertFalse(result.pe_sign_assumed)

    def test_w2b_rule4_nothing_known_is_blocked_with_the_message(self):
        for sidecar in (None, {}, {"PhaseEncodingDirection": "weird"}):
            result = pe.resolve_acquisition(sidecar=sidecar, affine=RAS)
            self.assertIsNone(result.phase_encoding, sidecar)
            self.assertEqual(result.error, pe.RULE4_MESSAGE)
        self.assertIn("most axial clinical DWI: AP", pe.RULE4_MESSAGE)

    def test_w2b_rule3_table_false_axis_only_falls_to_rule4_in_both_modes(self):
        self.assertEqual(pe.PE_SIGN_ASSUMPTION_MEASURED, {"rpe_none": False, "rpe_pair": False})
        for reverse in (False, True):
            result = pe.resolve_acquisition(sidecar={"PhaseEncodingAxis": "j"}, affine=RAS, has_reverse_b0=reverse)
            self.assertIsNone(result.phase_encoding)
            self.assertFalse(result.pe_sign_assumed)
            self.assertEqual(result.error, pe.RULE4_MESSAGE)

    def test_w2b_rule3_table_true_gives_sidecar_axis_with_the_flag(self):
        with patch.dict(pe.PE_SIGN_ASSUMPTION_MEASURED, {"rpe_none": True, "rpe_pair": False}):
            none_mode = pe.resolve_acquisition(sidecar={"PhaseEncodingAxis": "j"}, affine=LAS)
            self.assertEqual((none_mode.phase_encoding, none_mode.phase_encoding_source), ("PA", "sidecar_axis"))
            self.assertTrue(none_mode.pe_sign_assumed)
            # The other mode is not measured: it still falls to rule 4.
            pair_mode = pe.resolve_acquisition(sidecar={"PhaseEncodingAxis": "j"}, affine=LAS, has_reverse_b0=True)
            self.assertIsNone(pair_mode.phase_encoding)
        with patch.dict(pe.PE_SIGN_ASSUMPTION_MEASURED, {"rpe_none": False, "rpe_pair": True}):
            pair_mode = pe.resolve_acquisition(sidecar={"PhaseEncodingAxis": "i"}, affine=RAS, has_reverse_b0=True)
            self.assertEqual((pair_mode.phase_encoding, pair_mode.phase_encoding_source), ("LR", "sidecar_axis"))
            self.assertTrue(pair_mode.pe_sign_assumed)

    def test_w2b_a_sidecar_direction_beats_the_axis(self):
        with patch.dict(pe.PE_SIGN_ASSUMPTION_MEASURED, {"rpe_none": True, "rpe_pair": True}):
            result = pe.resolve_acquisition(
                sidecar={"PhaseEncodingDirection": "j-", "PhaseEncodingAxis": "j"}, affine=RAS)
        self.assertEqual((result.phase_encoding, result.phase_encoding_source), ("AP", "sidecar"))
        self.assertFalse(result.pe_sign_assumed)


class ReadoutRulesTest(unittest.TestCase):
    def test_w2b_readout_rule1_user_value(self):
        result = pe.resolve_acquisition(user_readout_time=0.05, sidecar={"TotalReadoutTime": 0.065})
        self.assertEqual((result.readout_time, result.readout_time_source), (0.05, "user"))
        self.assertFalse(result.readout_assumed)
        self.assertEqual(result.readout_time_for_command, 0.05)

    def test_w2b_readout_rule2_sidecar_value(self):
        result = pe.resolve_acquisition(sidecar={"TotalReadoutTime": 0.065})
        self.assertEqual((result.readout_time, result.readout_time_source), (0.065, "sidecar"))
        self.assertFalse(result.readout_assumed)
        self.assertEqual(result.readout_time_for_command, 0.065)

    def test_w2b_readout_rule3_none_is_assumed_and_left_out_of_the_command(self):
        result = pe.resolve_acquisition(sidecar={})
        self.assertTrue(result.readout_assumed)
        self.assertEqual(result.readout_time_source, "assumed")
        self.assertIsNone(result.readout_time_for_command)
        self.assertEqual(result.readout_time, pe.DWIFSLPREPROC_DEFAULT_READOUT_S)

    def test_w2b_an_unusable_sidecar_readout_is_ignored_with_a_warning(self):
        for bad in (-1, 0, "0.05", 65.0):
            result = pe.resolve_acquisition(sidecar={"TotalReadoutTime": bad})
            self.assertTrue(result.readout_assumed, bad)
            self.assertTrue(result.warnings, bad)
            self.assertEqual(result.readout_time_sidecar_raw, bad)


class ConflictTest(unittest.TestCase):
    def test_w2b_pe_conflict_records_both_values_and_warns(self):
        result = pe.resolve_acquisition(user_phase_encoding="AP", sidecar={"PhaseEncodingDirection": "j"}, affine=RAS)
        self.assertEqual(result.phase_encoding, "AP")
        self.assertEqual(len(result.conflicts), 1)
        conflict = result.conflicts[0]
        self.assertEqual((conflict["user"], conflict["sidecar"]), ("AP", "PA"))
        self.assertIn("PA", " ".join(result.warnings))
        self.assertIn("WARNING", result.summary_text())
        record = result.to_record()
        self.assertEqual(record["phase_encoding"]["user"], "AP")
        self.assertEqual(record["phase_encoding"]["sidecar_raw"], "j")
        self.assertEqual(record["conflicts"], [dict(conflict)])

    def test_w2b_agreement_is_not_a_conflict_even_in_another_spelling(self):
        result = pe.resolve_acquisition(user_phase_encoding="j-", sidecar={"PhaseEncodingDirection": "j-"}, affine=LAS)
        self.assertEqual(result.conflicts, ())
        self.assertEqual(result.warnings, ())

    def test_w2b_readout_conflict_uses_the_5e3_tolerance(self):
        near = pe.resolve_acquisition(user_readout_time=0.0650, sidecar={"TotalReadoutTime": 0.0690})
        self.assertEqual(near.conflicts, ())
        far = pe.resolve_acquisition(user_readout_time=0.0650, sidecar={"TotalReadoutTime": 0.0750})
        self.assertEqual(len(far.conflicts), 1)
        self.assertEqual((far.conflicts[0]["user"], far.conflicts[0]["sidecar"]), (0.065, 0.075))
        self.assertEqual(far.readout_time, 0.065)


def write_subject(root: Path, *, affine=None, sidecar=None, subject_id="100307") -> Path:
    import nibabel as nib
    import numpy as np

    subject = root / subject_id
    input_dir = subject / "input"
    input_dir.mkdir(parents=True)
    data = np.zeros((5, 5, 5, 3), dtype=np.float32)
    data[..., 0] = 1.0
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[1:4, 1:4, 1:4] = 1
    matrix = np.eye(4) if affine is None else np.array(affine, dtype=float)
    nib.save(nib.Nifti1Image(data, matrix), str(input_dir / "data.nii.gz"))
    nib.save(nib.Nifti1Image(mask, matrix), str(input_dir / "nodif_brain_mask.nii.gz"))
    (input_dir / "bvals").write_text("0 1000 1000\n", encoding="utf-8")
    (input_dir / "bvecs").write_text("0 1 0\n0 0 1\n0 0 0\n", encoding="utf-8")
    if sidecar is not None:
        (input_dir / "dwi.json").write_text(json.dumps(sidecar), encoding="utf-8")
    return subject


def plan_commands(subject: Path, **config_kwargs):
    from preprocessing import DwiPreprocessingConfig, run_dwi_preprocessing

    result = run_dwi_preprocessing(
        DwiPreprocessingConfig(subject_id="100307", input_root=subject, execute=False, **config_kwargs))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    for record in manifest["commands"]:
        if Path(record["argv"][0]).name == "dwifslpreproc":
            return record["argv"], manifest
    raise AssertionError("no dwifslpreproc command in the plan")


class CommandTest(unittest.TestCase):
    def test_w2b_pe_dir_is_a_designator_converted_from_the_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            subject = write_subject(Path(tmp), affine=LAS, sidecar={"PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.065})
            argv, manifest = plan_commands(subject)
            self.assertEqual(argv[argv.index("-pe_dir") + 1], "AP")
            self.assertEqual(argv[argv.index("-readout_time") + 1], "0.065")
            self.assertNotIn("-json_import", argv)
            self.assertNotIn("-rpe_header", argv)

    def test_w2b_user_nifti_code_is_converted_before_the_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            subject = write_subject(Path(tmp), affine=PERMUTED)
            argv, _ = plan_commands(subject, phase_encoding_direction="i")
            self.assertEqual(argv[argv.index("-pe_dir") + 1], "PA")

    def test_w2b_readout_time_flag_only_when_known(self):
        with tempfile.TemporaryDirectory() as tmp:
            subject = write_subject(Path(tmp))
            argv, manifest = plan_commands(subject, phase_encoding_direction="AP")
            self.assertEqual(argv[argv.index("-pe_dir") + 1], "AP")
            self.assertNotIn("-readout_time", argv)
            argv, manifest = plan_commands(subject, phase_encoding_direction="AP", readout_time=0.05)
            self.assertEqual(argv[argv.index("-readout_time") + 1], "0.05")

    def test_w2b_executing_without_a_direction_fails_before_any_stage(self):
        from preprocessing import DwiPreprocessingConfig, run_dwi_preprocessing

        with tempfile.TemporaryDirectory() as tmp:
            subject = write_subject(Path(tmp))
            with self.assertRaisesRegex(pe.PhaseEncodingNeeded, "Phase-encoding direction needed"):
                run_dwi_preprocessing(DwiPreprocessingConfig(subject_id="100307", input_root=subject, execute=True))
            self.assertFalse((subject / "outputs" / "01_preproc").exists() and any(
                (subject / "outputs" / "01_preproc").iterdir()))


if __name__ == "__main__":
    unittest.main()
