#!/usr/bin/env python3
"""W2b (T1 D1 part 2): the manifest and QC record, and the version-6 preprocessing cache key."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from test_preprocessing_pipeline import _fake_pnl_mask, _write_fake_backend, _write_nifti_subject  # noqa: E402


def make_config(subject: Path, bin_dir: Path, **kwargs):
    from preprocessing import DwiPreprocessingConfig

    return DwiPreprocessingConfig(subject_id="100307", input_root=subject, execute=True,
                                  mrtrix_bin_dir=bin_dir, fsl_bin_dir=bin_dir, **kwargs)


def cache_key_for(subject: Path, config):
    from preprocessing.pipeline import _preprocessing_cache_key, inspect_preprocessing_backend
    from shared.pipeline import find_subject_root

    raw = find_subject_root(subject, "100307", generate_missing_mask=False)
    return _preprocessing_cache_key(raw, config, inspect_preprocessing_backend(config))


class W2bCacheKeyTest(unittest.TestCase):
    def setUp(self) -> None:
        patcher = patch("preprocessing.pipeline.generate_brain_mask", side_effect=_fake_pnl_mask)
        patcher.start()
        self.addCleanup(patcher.stop)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.subject = _write_nifti_subject(self.root)
        self.bin_dir = self.root / "bin"
        _write_fake_backend(self.bin_dir)

    def test_w2b_key_is_version_6_with_the_route_id_of_the_run(self) -> None:
        key = cache_key_for(self.subject, make_config(self.subject, self.bin_dir))
        self.assertEqual(key["version"], 6)
        self.assertEqual(key["backend"], "dipy_mppca+gibbs+fsl_eddy")
        skipped = cache_key_for(self.subject, make_config(self.subject, self.bin_dir, dipy_run_gibbs=False))
        self.assertEqual(skipped["backend"], "dipy_mppca+fsl_eddy")
        reverse = self.root / "reverse.nii.gz"
        reverse.write_bytes(b"x")
        with_topup = cache_key_for(self.subject, make_config(self.subject, self.bin_dir, reverse_phase_b0=reverse))
        self.assertEqual(with_topup["backend"], "dipy_mppca+gibbs+fsl_eddy+topup")
        both = cache_key_for(self.subject, make_config(
            self.subject, self.bin_dir, reverse_phase_b0=reverse, dipy_run_gibbs=False))
        self.assertEqual(both["backend"], "dipy_mppca+fsl_eddy+topup")

    def test_w2b_key_holds_effective_values_with_sources_and_the_sidecar_hash(self) -> None:
        key = cache_key_for(self.subject, make_config(self.subject, self.bin_dir))
        parameters = key["parameters"]
        self.assertEqual((parameters["phase_encoding_direction"], parameters["phase_encoding_source"]), ("AP", "sidecar"))
        self.assertEqual((parameters["readout_time"], parameters["readout_time_source"]), (0.05, "sidecar"))
        self.assertFalse(parameters["pe_sign_assumed"])
        sidecar = (self.subject / "input" / "dwi.json").read_bytes()
        self.assertEqual(key["dwi_sidecar_sha256"], hashlib.sha256(sidecar).hexdigest())
        user = cache_key_for(self.subject, make_config(self.subject, self.bin_dir, phase_encoding_direction="PA"))
        self.assertEqual((user["parameters"]["phase_encoding_direction"], user["parameters"]["phase_encoding_source"]),
                         ("PA", "user"))

    def test_w2b_key_without_a_sidecar_has_a_null_hash_and_assumed_readout(self) -> None:
        (self.subject / "input" / "dwi.json").unlink()
        key = cache_key_for(self.subject, make_config(self.subject, self.bin_dir, phase_encoding_direction="AP"))
        self.assertIsNone(key["dwi_sidecar_sha256"])
        self.assertEqual(key["parameters"]["readout_time_source"], "assumed")

    def test_w2b_a_user_pe_change_with_an_unchanged_sidecar_reruns_preprocessing(self) -> None:
        from preprocessing import run_dwi_preprocessing

        ap = make_config(self.subject, self.bin_dir, phase_encoding_direction="AP")
        pa = make_config(self.subject, self.bin_dir, phase_encoding_direction="PA")
        self.assertNotEqual(cache_key_for(self.subject, ap), cache_key_for(self.subject, pa))
        first = run_dwi_preprocessing(ap)
        self.assertEqual(first.status, "completed")
        self.assertEqual(run_dwi_preprocessing(ap).status, "cached")
        changed = run_dwi_preprocessing(pa)
        self.assertEqual(changed.status, "completed")
        manifest = json.loads(changed.manifest_path.read_text(encoding="utf-8"))
        self.assertTrue(manifest["overwrite_existing"])
        command = next(record["argv"] for record in manifest["commands"] if Path(record["argv"][0]).name == "dwifslpreproc")
        self.assertEqual(command[command.index("-pe_dir") + 1], "PA")

    def test_w2b_the_orchestrator_resume_check_sees_a_pe_change(self) -> None:
        from preprocessing import run_dwi_preprocessing
        from shared import ProductOrchestratorConfig
        from shared.orchestrator import _preprocessing_complete, _preprocessing_config

        config = ProductOrchestratorConfig(
            subject_id="100307", subject_root=self.subject, targets=("preprocessing",), execute=True,
            brain_mask_backend="pnl_cnn", preprocessing_mrtrix_bin_dir=self.bin_dir,
            preprocessing_fsl_bin_dir=self.bin_dir, preprocessing_phase_encoding_direction="AP", resume=True)
        run_dwi_preprocessing(_preprocessing_config(config, self.subject, self.subject / "outputs" / "01_preproc"))
        self.assertTrue(_preprocessing_complete(self.subject, "100307", config=config))
        from dataclasses import replace
        self.assertFalse(_preprocessing_complete(
            self.subject, "100307", config=replace(config, preprocessing_phase_encoding_direction="PA")))
        # A readout time that differs from the one recorded also invalidates the outputs.
        self.assertFalse(_preprocessing_complete(
            self.subject, "100307", config=replace(config, preprocessing_readout_time=0.09)))

    def test_w2b_a_changed_sidecar_changes_the_key(self) -> None:
        before = cache_key_for(self.subject, make_config(self.subject, self.bin_dir, phase_encoding_direction="AP"))
        (self.subject / "input" / "dwi.json").write_text(json.dumps({"EchoTime": 0.09}), encoding="utf-8")
        after = cache_key_for(self.subject, make_config(self.subject, self.bin_dir, phase_encoding_direction="AP"))
        self.assertNotEqual(before["dwi_sidecar_sha256"], after["dwi_sidecar_sha256"])
        self.assertNotEqual(before, after)


class W2bRecordTest(unittest.TestCase):
    def setUp(self) -> None:
        patcher = patch("preprocessing.pipeline.generate_brain_mask", side_effect=_fake_pnl_mask)
        patcher.start()
        self.addCleanup(patcher.stop)
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.subject = _write_nifti_subject(self.root)
        self.bin_dir = self.root / "bin"
        _write_fake_backend(self.bin_dir)

    def _run(self, **kwargs):
        from preprocessing import run_dwi_preprocessing

        result = run_dwi_preprocessing(make_config(self.subject, self.bin_dir, **kwargs))
        manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
        qc = json.loads(result.qc_path.read_text(encoding="utf-8"))
        return result, manifest, qc

    def test_w2b_manifest_and_qc_record_values_sources_flags_and_the_sidecar_hash(self) -> None:
        result, manifest, qc = self._run()
        for record in (manifest["acquisition"], qc["acquisition"], qc["backend_details"]["acquisition"]):
            self.assertEqual(record["phase_encoding"]["value"], "AP")
            self.assertEqual(record["phase_encoding"]["source"], "sidecar")
            self.assertEqual(record["phase_encoding"]["sidecar_raw"], "j-")
            self.assertEqual(record["readout_time"]["value"], 0.05)
            self.assertEqual(record["readout_time"]["source"], "sidecar")
            self.assertEqual(record["readout_time"]["sidecar_raw"], 0.05)
            self.assertFalse(record["pe_sign_assumed"])
            self.assertFalse(record["readout_assumed"])
            self.assertEqual(record["sidecar_sha256"], manifest["cache_key"]["dwi_sidecar_sha256"])
        self.assertIsNotNone(manifest["acquisition"]["sidecar_sha256"])
        self.assertEqual(result.acquisition.phase_encoding, "AP")
        self.assertEqual(manifest["backend"]["backend"], "dipy_mppca+gibbs+fsl_eddy")
        self.assertEqual(qc["backend_details"]["route"], "dipy_mppca+gibbs+fsl_eddy")

    def test_w2b_readout_assumed_flag_is_recorded_and_shown(self) -> None:
        (self.subject / "input" / "dwi.json").unlink()
        result, manifest, qc = self._run(phase_encoding_direction="AP")
        self.assertTrue(manifest["acquisition"]["readout_assumed"])
        self.assertEqual(manifest["acquisition"]["readout_time"]["source"], "assumed")
        self.assertIsNone(manifest["acquisition"]["sidecar_sha256"])
        self.assertTrue(any("readout_assumed" in warning for warning in qc["warnings"]))
        self.assertIn("readout_assumed", result.acquisition.summary_text())

    def test_w2b_a_conflict_records_both_values_and_shows_a_warning(self) -> None:
        result, manifest, qc = self._run(phase_encoding_direction="PA")
        record = manifest["acquisition"]
        self.assertEqual(record["phase_encoding"]["value"], "PA")
        self.assertEqual(record["phase_encoding"]["source"], "user")
        self.assertEqual(record["phase_encoding"]["sidecar_raw"], "j-")
        self.assertEqual(record["conflicts"][0]["user"], "PA")
        self.assertEqual(record["conflicts"][0]["sidecar"], "AP")
        self.assertTrue(any("sidecar says AP" in warning for warning in qc["warnings"]))
        self.assertIn("WARNING", result.acquisition.summary_text())

    def test_w2b_cached_run_keeps_the_record_and_the_warnings(self) -> None:
        first, _, first_qc = self._run(phase_encoding_direction="PA")
        second, manifest, qc = self._run(phase_encoding_direction="PA")
        self.assertEqual(second.status, "cached")
        self.assertEqual(manifest["acquisition"]["conflicts"][0]["sidecar"], "AP")
        self.assertEqual(qc["acquisition"], first_qc["acquisition"])
        self.assertEqual(qc["warnings"], first_qc["warnings"])


if __name__ == "__main__":
    unittest.main()
