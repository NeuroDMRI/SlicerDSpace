#!/usr/bin/env python3
"""W2b (T1 D1 part 2): the one reuse predicate for preprocessed outputs and the run record fields."""

from __future__ import annotations

import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RAW_NAMES = {"dwi": "data.nii.gz", "bval": "bvals", "bvec": "bvecs", "mask": "nodif_brain_mask.nii.gz"}
OUTPUT_NAMES = ("dwi_final.nii.gz", "dwi_final.bval", "dwi_final.bvec", "brain_mask_final.nii.gz")


def make_subject(root: Path) -> Path:
    subject = Path(root).resolve() / "100307"
    (subject / "input").mkdir(parents=True)
    for name in RAW_NAMES.values():
        (subject / "input" / name).write_text("raw " + name, encoding="utf-8")
    return subject


def write_preprocessing(subject: Path, *, status: str = "completed", cache_key="v5", qc="completed",
                        missing: str = "", route: str = "dipy_mppca+gibbs+fsl_eddy") -> Path:
    """Write a preprocessing folder. `cache_key`: "v5", "v6", "v4", None (absent) or "bad" (not a dict)."""
    from shared.pipeline import _file_signature

    preproc = subject / "outputs" / "01_preproc"
    (preproc / "05_qc").mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_NAMES:
        path = preproc / f"100307_{name}"
        if missing and name == missing:
            path.unlink(missing_ok=True)
        else:
            path.write_text("preprocessed " + name, encoding="utf-8")
    manifest = {"status": status}
    if cache_key in {"v4", "v5", "v6"}:
        version = int(cache_key[1])
        manifest["cache_key"] = {
            "version": version, "backend": route if version >= 6 else "dipy_mppca+fsl_eddy",
            "inputs": {key: _file_signature(subject / "input" / name) for key, name in RAW_NAMES.items()},
        }
    elif cache_key == "bad":
        manifest["cache_key"] = "not a dict"
    (preproc / "100307_preproc_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    qc_path = preproc / "05_qc" / "100307_preproc_qc.json"
    if qc == "missing":
        qc_path.unlink(missing_ok=True)
    else:
        qc_path.write_text(json.dumps({"status": "completed" if qc == "completed" else "failed"}), encoding="utf-8")
    return preproc


def reuse_points(subject: Path) -> dict:
    """What each of the two reuse points decides for this subject (True = preprocessed outputs)."""
    from shared.orchestrator import _preprocessing_complete, _resolve_analysis_inputs
    from shared.pipeline import RawInputBundle, preprocessed_outputs_usable, resolve_preprocessed_input_bundle

    preproc = subject / "outputs" / "01_preproc"
    input_dir = subject / "input"
    raw = RawInputBundle(subject, "100307", input_dir, input_dir / RAW_NAMES["dwi"], input_dir / RAW_NAMES["bval"],
                         input_dir / RAW_NAMES["bvec"], input_dir / RAW_NAMES["mask"])
    bundle = resolve_preprocessed_input_bundle(raw)
    inputs = _resolve_analysis_inputs(subject, "100307")
    return {
        "predicate": preprocessed_outputs_usable(subject, None, "100307"),
        "complete": _preprocessing_complete(subject, "100307"),
        "bundle": bundle.dwi_path == preproc / "100307_dwi_final.nii.gz",
        "orchestrator": inputs["dwi"] == preproc / "100307_dwi_final.nii.gz",
    }


class W2bReusePredicateTest(unittest.TestCase):
    def test_w2b_a_version_5_manifest_with_unchanged_raw_input_is_used_at_both_points(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subject = make_subject(Path(tmp))
            write_preprocessing(subject, cache_key="v5")
            self.assertEqual(set(reuse_points(subject).values()), {True})

    def test_w2b_a_version_6_manifest_with_unchanged_raw_input_is_used_at_both_points(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subject = make_subject(Path(tmp))
            write_preprocessing(subject, cache_key="v6")
            self.assertEqual(set(reuse_points(subject).values()), {True})

    def test_w2b_a_changed_raw_input_gives_raw_at_both_points(self) -> None:
        for name in RAW_NAMES.values():
            with self.subTest(changed=name), tempfile.TemporaryDirectory() as tmp:
                subject = make_subject(Path(tmp))
                write_preprocessing(subject, cache_key="v5")
                self.assertEqual(set(reuse_points(subject).values()), {True})
                (subject / "input" / name).write_text("changed after preprocessing, longer", encoding="utf-8")
                self.assertEqual(set(reuse_points(subject).values()), {False})

    def test_w2b_property_the_two_reuse_points_always_agree(self) -> None:
        """Over manifest states (status, cache key, QC, missing file, raw change) both points give one answer."""
        states = itertools.product(
            ("completed", "cached", "failed", "running"),
            ("v4", "v5", "v6", None, "bad"),
            ("completed", "failed", "missing"),
            ("", "dwi_final.nii.gz", "dwi_final.bval", "brain_mask_final.nii.gz"),
            (False, True),
        )
        checked = 0
        with tempfile.TemporaryDirectory() as tmp:
            subject = make_subject(Path(tmp))
            for status, cache_key, qc, missing, raw_changed in states:
                for name in RAW_NAMES.values():
                    (subject / "input" / name).write_text("raw " + name, encoding="utf-8")
                write_preprocessing(subject, status=status, cache_key=cache_key, qc=qc, missing=missing)
                if raw_changed:
                    (subject / "input" / "bvals").write_text("raw bvals changed later", encoding="utf-8")
                expected = (status in {"completed", "cached"} and cache_key in {"v5", "v6"} and qc == "completed"
                            and not missing and not raw_changed)
                decisions = reuse_points(subject)
                with self.subTest(status=status, cache_key=cache_key, qc=qc, missing=missing, raw_changed=raw_changed):
                    self.assertEqual(set(decisions.values()), {expected}, decisions)
                checked += 1
        self.assertEqual(checked, 4 * 5 * 3 * 4 * 2)

    def test_w2b_no_preprocessing_at_all_gives_raw(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subject = make_subject(Path(tmp))
            self.assertEqual(set(reuse_points(subject).values()), {False})


class W2bRunRecordInputTest(unittest.TestCase):
    def record(self, subject: Path) -> dict:
        from shared.run_record import build_run_record

        meta = subject / "outputs" / "00_run_metadata"
        meta.mkdir(parents=True, exist_ok=True)
        manifest = {"kind": "slicerddsurfer_product_orchestration", "subject_id": "100307", "run_name": "Test1",
                    "status": "completed", "execute": True, "targets": ["microstructure"],
                    "subject_root": subject.as_posix(), "tasks": []}
        path = meta / "Test1_orchestration_manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return build_run_record(path)

    def test_w2b_the_run_record_stores_version_route_id_and_cache_key_of_a_version_5_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subject = make_subject(Path(tmp))
            write_preprocessing(subject, cache_key="v5")
            used = self.record(subject)["preprocessed_input"]
            self.assertTrue(used["used"])
            self.assertEqual(used["cache_key_version"], 5)
            self.assertEqual(used["route_id"], "pre-v6")
            self.assertEqual(used["cache_key"]["version"], 5)

    def test_w2b_the_run_record_stores_version_route_id_and_cache_key_of_a_version_6_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subject = make_subject(Path(tmp))
            write_preprocessing(subject, cache_key="v6", route="dipy_mppca+fsl_eddy+topup")
            used = self.record(subject)["preprocessed_input"]
            self.assertEqual((used["cache_key_version"], used["route_id"]), (6, "dipy_mppca+fsl_eddy+topup"))
            self.assertEqual(used["cache_key"]["backend"], "dipy_mppca+fsl_eddy+topup")

    def test_w2b_the_run_record_says_when_the_raw_dwi_was_used(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subject = make_subject(Path(tmp))
            write_preprocessing(subject, cache_key="v5")
            (subject / "input" / "bvecs").write_text("changed", encoding="utf-8")
            used = self.record(subject)["preprocessed_input"]
            self.assertEqual(used, {"used": False,
                                    "message": "No current preprocessed DWI for this input; using the raw DWI"})


class W2bRunLogTest(unittest.TestCase):
    def run_products(self, subject: Path, targets) -> list[str]:
        from shared import ProductOrchestratorConfig
        from shared.orchestrator import run_subject_products

        lines: list[str] = []
        run_subject_products(ProductOrchestratorConfig(
            subject_id="100307", subject_root=subject, targets=targets, execute=False), log=lines.append)
        return lines

    def test_w2b_the_run_log_says_the_raw_dwi_is_used_when_no_preprocessing_is_current(self) -> None:
        message = "No current preprocessed DWI for this input; using the raw DWI"
        with tempfile.TemporaryDirectory() as tmp:
            subject = make_subject(Path(tmp))
            write_preprocessing(subject, cache_key="v5")
            self.assertFalse(any(message in line for line in self.run_products(subject, ("segmentation",))))
            (subject / "input" / "bvals").write_text("changed", encoding="utf-8")
            self.assertTrue(any(message in line for line in self.run_products(subject, ("segmentation",))))


if __name__ == "__main__":
    unittest.main()
