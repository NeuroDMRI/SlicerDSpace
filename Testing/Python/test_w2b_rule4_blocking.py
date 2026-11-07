#!/usr/bin/env python3
"""W2b (T1 D1 rule 4): preprocessing with no phase-encoding direction is blocked at plan time.

The three places D1 names: the GUI run button, the headless plan (CLI) and the cohort plan. Targets that do not need
preprocessing are not affected.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from test_w2b_pe_resolution import LAS, OBLIQUE, write_subject  # noqa: E402

from shared.pe_resolution import RULE4_MESSAGE  # noqa: E402

CLI_PATH = REPO_ROOT / "shared" / "run_subject_pipeline_cli.py"
SIDECAR = {"PhaseEncodingDirection": "j-", "TotalReadoutTime": 0.05}


def config_for(subject: Path, targets, **kwargs):
    from shared import ProductOrchestratorConfig

    return ProductOrchestratorConfig(subject_id="100307", subject_root=subject, targets=tuple(targets), **kwargs)


class W2bPlanBlockTest(unittest.TestCase):
    """The headless plan and the GUI plan: `plan_subject_products` (the GUI run builds the same plan)."""

    def plan_item(self, subject: Path, targets=("preprocessing",), **kwargs):
        from shared import plan_subject_products

        plan = plan_subject_products(config_for(subject, targets, **kwargs), write=False)
        return next(item for item in plan.items if item.target == targets[0])

    def test_w2b_no_direction_blocks_the_preprocessing_item_with_the_rule4_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            item = self.plan_item(write_subject(Path(tmp)))
            self.assertEqual(item.status, "blocked")
            self.assertEqual(item.reason, RULE4_MESSAGE)
            self.assertEqual(item.reason_category, "phase_encoding")
            self.assertIn("most axial clinical DWI: AP", item.reason)

    def test_w2b_a_sidecar_or_a_user_direction_leaves_the_item_runnable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with_sidecar = write_subject(Path(tmp) / "a", sidecar=SIDECAR)
            self.assertEqual(self.plan_item(with_sidecar).status, "will_run")
            without = write_subject(Path(tmp) / "b")
            self.assertEqual(self.plan_item(without, preprocessing_phase_encoding_direction="AP").status, "will_run")
            self.assertEqual(self.plan_item(without, preprocessing_phase_encoding_direction="j-").status, "will_run")

    def test_w2b_an_oblique_nifti_code_is_blocked_too(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subject = write_subject(Path(tmp), affine=OBLIQUE)
            item = self.plan_item(subject, preprocessing_phase_encoding_direction="i")
            self.assertEqual((item.status, item.reason), ("blocked", RULE4_MESSAGE))
            self.assertEqual(self.plan_item(subject, preprocessing_phase_encoding_direction="AP").status, "will_run")

    def test_w2b_a_run_without_the_preprocessing_target_is_not_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subject = write_subject(Path(tmp))
            from shared import plan_subject_products

            plan = plan_subject_products(config_for(subject, ("microstructure",)), write=False)
            self.assertNotIn("preprocessing", [item.target for item in plan.items])
            self.assertEqual([item for item in plan.items if item.reason_category == "phase_encoding"], [])


class W2bRunBlockTest(unittest.TestCase):
    def test_w2b_an_executing_run_is_blocked_before_any_stage(self) -> None:
        from shared import run_subject_products

        with tempfile.TemporaryDirectory() as tmp:
            subject = write_subject(Path(tmp))
            lines: list[str] = []
            result = run_subject_products(
                config_for(subject, ("preprocessing", "microstructure"), execute=True), log=lines.append)
            tasks = {task.name: task for task in result.tasks}
            self.assertEqual(tasks["preprocessing.dwi"].status, "blocked")
            self.assertEqual(tasks["preprocessing.dwi"].reason, RULE4_MESSAGE)
            self.assertEqual(tasks["microstructure.dti"].status, "blocked")
            self.assertIn(RULE4_MESSAGE, tasks["microstructure.dti"].reason)
            self.assertFalse((subject / "outputs" / "01_preproc" / "100307_preproc_manifest.json").exists())
            self.assertFalse((subject / "outputs" / "02_microstructure").exists())
            self.assertIn("[Orchestrator] " + RULE4_MESSAGE, lines)

    def test_w2b_an_executing_run_without_the_preprocessing_target_has_no_block(self) -> None:
        from shared import run_subject_products

        with tempfile.TemporaryDirectory() as tmp:
            subject = write_subject(Path(tmp))
            result = run_subject_products(config_for(subject, ("validation",), execute=True))
            self.assertFalse([task for task in result.tasks if RULE4_MESSAGE in task.reason])

    def test_w2b_the_preprocessing_task_reason_shows_values_sources_and_a_conflict_warning(self) -> None:
        from shared import run_subject_products
        from test_preprocessing_pipeline import _fake_pnl_mask, _write_fake_backend
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            subject = write_subject(Path(tmp), affine=LAS, sidecar=SIDECAR)
            bin_dir = Path(tmp) / "bin"
            _write_fake_backend(bin_dir)
            with patch("preprocessing.pipeline.generate_brain_mask", side_effect=_fake_pnl_mask):
                result = run_subject_products(config_for(
                    subject, ("preprocessing",), execute=True, brain_mask_backend="pnl_cnn",
                    preprocessing_mrtrix_bin_dir=bin_dir, preprocessing_fsl_bin_dir=bin_dir,
                    preprocessing_phase_encoding_direction="PA"))
            task = next(task for task in result.tasks if task.name == "preprocessing.dwi")
            self.assertEqual(task.status, "completed", task.reason)
            self.assertIn("phase encoding PA (user)", task.reason)
            self.assertIn("readout time 0.05 s (sidecar)", task.reason)
            self.assertIn("WARNING", task.reason)
            self.assertIn("sidecar says AP", task.reason)


class W2bCliBlockTest(unittest.TestCase):
    def run_cli(self, subject: Path, *args: str) -> subprocess.CompletedProcess:
        # The environment check is skipped: this test machine has no MRtrix3 or FSL, which would block first.
        return subprocess.run([sys.executable, str(CLI_PATH), "--subject-root", str(subject),
                               "--skip-environment-check", *args],
                              capture_output=True, text=True)

    def test_w2b_cli_plan_is_blocked_with_exit_2_and_the_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subject = write_subject(Path(tmp))
            done = self.run_cli(subject, "--targets", "preprocessing", "--plan")
            self.assertEqual(done.returncode, 2, done.stderr)
            self.assertIn(RULE4_MESSAGE, done.stderr)
            plan = json.loads(done.stdout)["run_plan"]
            item = next(item for item in plan["items"] if item["target"] == "preprocessing")
            self.assertEqual((item["status"], item["reason"]), ("blocked", RULE4_MESSAGE))

    def test_w2b_cli_execute_stops_with_exit_2_before_any_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subject = write_subject(Path(tmp))
            done = self.run_cli(subject, "--targets", "preprocessing,microstructure", "--execute")
            self.assertEqual(done.returncode, 2, done.stderr)
            self.assertIn(RULE4_MESSAGE, done.stderr)
            self.assertFalse((subject / "outputs" / "01_preproc").exists())
            self.assertFalse((subject / "outputs" / "02_microstructure").exists())

    def test_w2b_cli_with_a_sidecar_or_a_user_direction_or_no_preprocessing_target_is_not_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bare = write_subject(Path(tmp) / "a")
            with_sidecar = write_subject(Path(tmp) / "b", sidecar=SIDECAR)
            self.assertEqual(self.run_cli(with_sidecar, "--targets", "preprocessing", "--plan").returncode, 0)
            self.assertEqual(self.run_cli(bare, "--targets", "preprocessing", "--plan",
                                          "--preprocessing-phase-encoding-direction", "AP").returncode, 0)
            done = self.run_cli(bare, "--targets", "microstructure", "--plan")
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertNotIn(RULE4_MESSAGE, done.stderr)


class W2bGuiBlockTest(unittest.TestCase):
    """The GUI run handler shows the validation errors and the run button tooltip names the first one."""

    def validate(self, subject: Path, **kwargs):
        from shared.topup_parameters import validate_topup_parameters

        kwargs.setdefault("preprocessing_selected", True)
        return validate_topup_parameters(subject_root=subject, **kwargs)

    def test_w2b_gui_no_direction_is_an_error_that_stops_the_run(self) -> None:
        from shared.ui_run_state import run_button_state

        with tempfile.TemporaryDirectory() as tmp:
            subject = write_subject(Path(tmp))
            result = self.validate(subject)
            self.assertFalse(result.ok)
            self.assertEqual(result.errors, (RULE4_MESSAGE,))
            self.assertIn("phase-encoding", " ".join(result.errors).lower())  # the combo is marked as wrong
            state = run_button_state(
                pipeline_active=False, selected_targets=("preprocessing",), batch_mode=False, subject_id="100307",
                subject_root_text=str(subject), topup_errors=result.errors)
            self.assertIn(RULE4_MESSAGE, state.tooltip)

    def test_w2b_gui_sidecar_user_direction_other_targets_and_batch_mode_have_no_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bare = write_subject(Path(tmp) / "a")
            with_sidecar = write_subject(Path(tmp) / "b", sidecar=SIDECAR)
            self.assertTrue(self.validate(with_sidecar).ok)
            self.assertTrue(self.validate(bare, phase_encoding_direction="PA").ok)
            self.assertTrue(self.validate(bare, preprocessing_selected=False).ok)  # the closure never adds preprocessing
            self.assertTrue(self.validate(bare, batch_mode=True).ok)  # checked per subject by the cohort plan
            self.assertTrue(self.validate(bare).errors)

    def test_w2b_gui_combo_offers_from_sidecar_and_the_six_designators_only(self) -> None:
        from shared.topup_parameters import PHASE_ENCODING_CHOICES

        self.assertEqual([value for value, _ in PHASE_ENCODING_CHOICES], ["", "AP", "PA", "LR", "RL", "IS", "SI"])
        self.assertIn("sidecar", PHASE_ENCODING_CHOICES[0][1].lower())

    def test_w2b_gui_outputs_of_an_earlier_run_that_are_current_need_no_direction(self) -> None:
        from test_w2b_reuse_predicate import make_subject, write_preprocessing

        with tempfile.TemporaryDirectory() as tmp:
            subject = make_subject(Path(tmp))
            (subject / "input" / "data.nii.gz").unlink()
            import nibabel as nib
            import numpy as np
            nib.save(nib.Nifti1Image(np.zeros((3, 3, 3, 2), dtype=np.float32), np.eye(4)),
                     str(subject / "input" / "data.nii.gz"))
            write_preprocessing(subject, cache_key="v5")
            self.assertTrue(self.validate(subject).ok)
            (subject / "input" / "bvals").write_text("changed", encoding="utf-8")
            self.assertFalse(self.validate(subject).ok)


class W2bCohortBlockTest(unittest.TestCase):
    ARGS = ["--targets", "preprocessing"]

    def subjects(self, root: Path, *, sidecar_for=()):
        from types import SimpleNamespace

        found = []
        for subject_id in ("sub-01", "sub-02", "sub-03"):
            subject = write_subject(root, sidecar=SIDECAR if subject_id in sidecar_for else None, subject_id=subject_id)
            found.append(SimpleNamespace(subject_id=subject_id, subject_root=subject, mode="input"))
        return found

    def plan(self, subjects, arguments):
        from shared.cohort_planning import plan_cohort

        return plan_cohort(subjects, arguments, available=(8, 64 * 1024**3), cpu_budget=8, memory_gb=64)

    def test_w2b_cohort_plan_names_the_blocked_subjects_and_the_rule4_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subjects = self.subjects(Path(tmp), sidecar_for=("sub-02",))
            with self.assertRaises(ValueError) as caught:
                self.plan(subjects, self.ARGS)
            message = str(caught.exception)
            self.assertIn(RULE4_MESSAGE, message)
            self.assertIn("sub-01", message)
            self.assertIn("sub-03", message)
            self.assertNotIn("sub-02", message)

    def test_w2b_cohort_plan_passes_with_a_direction_sidecars_or_without_the_preprocessing_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subjects = self.subjects(Path(tmp))
            plan = self.plan(subjects, [*self.ARGS, "--preprocessing-phase-encoding-direction", "AP"])
            self.assertIn("preprocessing", plan["selected_stages"])
            plan = self.plan(subjects, ["--targets", "microstructure"])  # preprocessing is not selected
            self.assertNotIn("preprocessing", plan["selected_stages"])
        with tempfile.TemporaryDirectory() as tmp:
            everyone = self.subjects(Path(tmp), sidecar_for=("sub-01", "sub-02", "sub-03"))
            self.assertIn("preprocessing", self.plan(everyone, self.ARGS)["selected_stages"])

    def test_w2b_batch_cli_dry_run_is_blocked_before_any_subject_starts(self) -> None:
        spec = importlib.util.spec_from_file_location("run_batch_pipeline_w2b", REPO_ROOT / "scripts" / "run_batch_pipeline.py")
        batch = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = batch
        spec.loader.exec_module(batch)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.subjects(root / "subjects")
            err = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as caught:
                batch.main(["--batch-root", str(root / "subjects"), "--output-root", str(root / "outputs"),
                            "--manifest", str(root / "batch.json"), *self.ARGS])
            self.assertEqual(caught.exception.code, 2)
            self.assertIn("Phase-encoding direction needed", err.getvalue())
            self.assertFalse((root / "batch.json").exists())


if __name__ == "__main__":
    unittest.main()
