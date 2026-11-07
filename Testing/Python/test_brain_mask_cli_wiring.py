"""Brain-mask CLI contracts, config precedence, and batch forwarding."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import run_batch_pipeline as batch_cli
from shared import run_subject_pipeline_cli as cli


BACKENDS = ("pnl_cnn", "provided")
REMOVED_BACKENDS = ("synthstrip", "mrtrix", "dipy")
PATH_KEYS = ("pnl_mask_python", "pnl_mask_script", "pnl_mask_model_dir", "pnl_mask_ants_bin_dir")


def _path_options(root: Path) -> dict[str, Path]:
    return {key: root / "external tools" / key for key in (*PATH_KEYS, "postprocess_freesurfer_home")}


def _path_flags(paths: dict[str, Path]) -> list[str]:
    return [token for key, value in paths.items() for token in ("--" + key.replace("_", "-"), str(value))]


class BrainMaskCliWiringTest(unittest.TestCase):
    def test_parser_defaults_and_canonical_choices(self) -> None:
        parser = cli.build_parser()
        defaults = parser.parse_args([])
        self.assertEqual(defaults.brain_mask_backend, "pnl_cnn")
        for key in PATH_KEYS:
            self.assertIsNone(getattr(defaults, key))
        for backend in BACKENDS:
            self.assertEqual(parser.parse_args(["--brain-mask-backend", backend]).brain_mask_backend, backend)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            parser.parse_args(["--brain-mask-backend", "auto"])
        self.assertEqual(raised.exception.code, 2)

    def test_w5_removed_backends_fail_on_the_cli_naming_the_two_that_remain(self) -> None:
        """W5 (SC-2 rule 1): each removed backend is refused with a message that names pnl_cnn and provided."""
        for backend in REMOVED_BACKENDS:
            for flags in (["--brain-mask-backend", backend], [f"--brain-mask-backend={backend}"]):
                with self.subTest(backend=backend, flags=flags):
                    stderr = io.StringIO()
                    with patch.object(cli, "run_subject_products") as run, redirect_stderr(stderr), \
                            self.assertRaises(SystemExit) as raised:
                        cli.build_parser().parse_args(flags)
                    self.assertEqual(raised.exception.code, 2)
                    text = stderr.getvalue()
                    self.assertIn(backend, text)
                    self.assertIn("pnl_cnn", text)
                    self.assertIn("provided", text)
                    run.assert_not_called()
        # The full entry point stops before any work, too.
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(cli, "run_subject_products") as run, redirect_stderr(io.StringIO()) as stderr, \
                    self.assertRaises(SystemExit):
                cli.main(["--subject-root", tmp, "--no-user-config", "--brain-mask-backend", "synthstrip"])
            run.assert_not_called()
            self.assertIn("pnl_cnn", stderr.getvalue())

    def _invoke(self, subject: Path, flags: list[str]):
        result = SimpleNamespace(
            subject_id=subject.name, subject_root=subject, run_manifest_path=subject / "manifest.json",
            run_plan_path=None, tasks=(),
        )
        plan = SimpleNamespace(
            run_plan_path=subject / "plan.json", items=(),
            to_dict=lambda: {"summary": {"will_run": 0, "skipped": 0, "blocked": 0}},
        )
        doctor = SimpleNamespace(to_dict=lambda: {"kind": "environment_doctor"})
        with ExitStack() as stack:
            run = stack.enter_context(patch.object(cli, "run_subject_products", return_value=result))
            planning = stack.enter_context(patch.object(cli, "plan_subject_products", return_value=plan))
            checking = stack.enter_context(patch.object(cli, "run_environment_doctor", return_value=doctor))
            stack.enter_context(patch.object(cli, "write_environment_report", return_value={
                key: subject / key for key in ("json", "python_sh", "external_notes")
            }))
            stack.enter_context(patch.object(cli, "summarize_environment_doctor", return_value="Checked"))
            stack.enter_context(patch.object(cli, "blocked_target_reasons", return_value={}))
            stack.enter_context(patch.object(cli.signal, "signal"))
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(redirect_stderr(io.StringIO()))
            self.assertEqual(cli.main(["--subject-root", str(subject), "--targets", "microstructure", *flags]), 0)
        return run, planning, checking

    def test_partial_failed_task_gives_exit_code_2_like_the_failed_run_manifest(self) -> None:
        # EDDEN sub-01 WMA run, 2026-10-23: the run manifest said "failed" (bundles.wma partial_failed),
        # but the CLI exited 0, so a batch counted the subject as a success.
        with tempfile.TemporaryDirectory() as tmp:
            subject = Path(tmp) / "sub-01"
            for status, code in (("partial_failed", 2), ("failed", 2), ("completed", 0), ("blocked", 0)):
                with self.subTest(status=status):
                    task = SimpleNamespace(name="bundles.wma", status=status, reason="", manifest_path=None,
                                           qc_path=None, outputs={})
                    result = SimpleNamespace(subject_id="sub-01", subject_root=subject,
                                             run_manifest_path=subject / "missing.json", run_plan_path=None,
                                             tasks=(task,))
                    plan = SimpleNamespace(run_plan_path=subject / "plan.json", items=(),
                                           to_dict=lambda: {"summary": {"will_run": 1, "skipped": 0, "blocked": 0}})
                    with ExitStack() as stack:
                        stack.enter_context(patch.object(cli, "run_subject_products", return_value=result))
                        stack.enter_context(patch.object(cli, "plan_subject_products", return_value=plan))
                        stack.enter_context(patch.object(cli.signal, "signal"))
                        stack.enter_context(redirect_stdout(io.StringIO()))
                        stack.enter_context(redirect_stderr(io.StringIO()))
                        self.assertEqual(cli.main(["--subject-root", str(subject), "--no-user-config",
                                                   "--targets", "wma", "--execute"]), code)

    def test_doctor_plan_execution_and_dry_run_forward_every_choice_and_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subject = Path(tmp) / "sub-01"
            paths = _path_options(Path(tmp))
            for backend in BACKENDS:
                for mode in ((), ("--plan",), ("--execute",), ("--doctor", "--execute")):
                    with self.subTest(backend=backend, mode=mode):
                        run, plan, doctor = self._invoke(subject, [
                            "--no-user-config", "--brain-mask-backend", backend, *_path_flags(paths), *mode,
                        ])
                        calls = [call for mock in (run, plan, doctor) for call in mock.call_args_list]
                        self.assertTrue(calls)
                        for call in calls:
                            config = call.args[0]
                            self.assertEqual(config.brain_mask_backend, backend)
                            for key, path in paths.items():
                                self.assertEqual(getattr(config, key), path)
                        if "--doctor" in mode:
                            doctor.assert_called_once()
                            plan.assert_not_called()
                            run.assert_not_called()

    def test_explicit_flags_override_both_saved_config_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            user_config = root / "user.json"
            dependency_config = root / "dependencies.json"
            saved_paths = _path_options(root / "saved")
            profile_paths = _path_options(root / "profile")
            explicit_paths = _path_options(root / "explicit")
            user_config.write_text(json.dumps({
                "run_defaults": {"brain_mask_backend": "provided"},
                "external_dependencies": {key: str(path) for key, path in saved_paths.items()},
            }), encoding="utf-8")
            dependency_config.write_text(json.dumps({
                "external_dependencies": {key: str(path) for key, path in profile_paths.items()},
            }), encoding="utf-8")
            common = ["--user-config", str(user_config)]
            for extra, expected_backend, expected_paths in (
                ([], "provided", saved_paths),
                (["--dependency-config", str(dependency_config)], "provided", profile_paths),
                (["--dependency-config", str(dependency_config), "--brain-mask-backend=pnl_cnn",
                  *_path_flags(explicit_paths)], "pnl_cnn", explicit_paths),
            ):
                with self.subTest(extra=extra):
                    run, _, _ = self._invoke(root / "sub-01", [*common, *extra])
                    config = run.call_args.args[0]
                    self.assertEqual(config.brain_mask_backend, expected_backend)
                    for key, path in expected_paths.items():
                        self.assertEqual(getattr(config, key), path)

    def test_saved_backend_is_rejected_rather_than_falling_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "user.json"
            config.write_text(json.dumps({"run_defaults": {"brain_mask_backend": "auto"}}), encoding="utf-8")
            with (
                patch.object(cli, "run_subject_products") as run,
                patch.object(cli, "run_environment_doctor") as doctor,
                redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised,
            ):
                cli.main(["--subject-root", tmp, "--user-config", str(config)])
            self.assertEqual(raised.exception.code, 2)
            run.assert_not_called()
            doctor.assert_not_called()

    def test_save_and_reload_preserve_backend_and_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _path_options(root)
            user_config = root / "user.json"
            dependency_config = root / "dependency.json"
            self._invoke(root / "sub-01", [
                "--no-user-config", "--brain-mask-backend", "pnl_cnn", *_path_flags(paths),
                "--save-user-config", str(user_config), "--save-dependency-config", str(dependency_config),
            ])
            saved = json.loads(user_config.read_text(encoding="utf-8"))
            self.assertEqual(saved["run_defaults"]["brain_mask_backend"], "pnl_cnn")
            for file in (user_config, dependency_config):
                payload = json.loads(file.read_text(encoding="utf-8"))
                for key, value in paths.items():
                    self.assertEqual(payload["external_dependencies"][key], str(value))
            run, _, _ = self._invoke(root / "sub-02", ["--user-config", str(user_config)])
            for key, value in paths.items():
                self.assertEqual(getattr(run.call_args.args[0], key), value)

    def test_cli_orchestrator_handoff_preserves_input_validation_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject = root / "sub-01"
            paths = _path_options(root)
            validation_result = SimpleNamespace(status="passed", error_count=0, warning_count=0, report_path=None)
            for backend in BACKENDS:
                with (
                    self.subTest(backend=backend),
                    patch("shared.orchestrator.validate_subject_inputs", return_value=validation_result) as validate,
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()),
                ):
                    self.assertEqual(cli.main([
                        "--subject-root", str(subject), "--targets", "validation", "--no-user-config",
                        "--brain-mask-backend", backend, *_path_flags(paths),
                    ]), 0)
                validate.assert_called_once()
                config = validate.call_args.args[0]
                self.assertEqual(config.brain_mask_backend, backend)
                for key, path in paths.items():
                    if hasattr(config, key):  # W5: the validation config no longer carries the FreeSurfer home
                        self.assertEqual(getattr(config, key), path)
                self.assertTrue(all(hasattr(config, key) for key in PATH_KEYS))

    def test_doctor_marks_only_known_mask_requirements(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subject = Path(tmp) / "sub-01"
            args = cli.build_parser().parse_args(["--subject-root", str(subject), "--no-user-config"])

            def required(targets=("microstructure",)):
                return cli._environment_config_from_args(args, targets=targets, output_root=None).brain_mask_required

            self.assertFalse(required())
            input_dir = subject / "input"
            input_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs"):
                (input_dir / name).touch()
            self.assertTrue(required())
            self.assertFalse(required(("validation",)))
            self.assertFalse(required(("wma", "tractcloud", "surface_postprocess")))
            (input_dir / "nodif_brain_mask.nii.gz").touch()
            self.assertFalse(required())
            self.assertTrue(required(("preprocessing",)))
            args.brain_mask_backend = "provided"
            self.assertFalse(required(("preprocessing",)))
            args.preprocessing_force = True
            self.assertFalse(required(("preprocessing",)))
            args.preprocessing_force = False
            args.brain_mask_backend = "pnl_cnn"
            preproc = subject / "outputs" / "01_preproc"
            preproc.mkdir(parents=True)
            for suffix in ("dwi_final.nii.gz", "dwi_final.bval", "dwi_final.bvec", "brain_mask_final.nii.gz"):
                (preproc / f"sub-01_{suffix}").touch()
            self.assertFalse(required(("preprocessing",)))
            args.preprocessing_backend = "mrtrix_fsl"
            self.assertFalse(required(("preprocessing",)))
            args.preprocessing_force = True
            self.assertTrue(required(("preprocessing",)))
            args.preprocessing_force = False
            args.no_resume = True
            self.assertTrue(required(("preprocessing",)))
            args.no_resume = False
            os.utime(preproc / "sub-01_brain_mask_final.nii.gz", ns=(1, 1))
            self.assertTrue(required(("preprocessing",)))
            args.brain_mask_backend = "provided"
            self.assertFalse(required(("preprocessing",)))
            args.brain_mask_backend = "pnl_cnn"
            args.preprocessing_backend = "dipy"
            args.preprocessing_dipy_denoise_method = "none"
            args.preprocessing_skip_gibbs = True
            args.preprocessing_skip_motion = True
            self.assertFalse(required(("preprocessing",)))
            args.brain_mask_backend = "provided"
            (input_dir / "nodif_brain_mask.nii.gz").unlink()
            self.assertTrue(required(("preprocessing",)))

    def test_real_doctor_does_not_invoke_configured_mask_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subject = Path(tmp) / "sub-01"
            input_dir = subject / "input"
            input_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs"):
                (input_dir / name).touch()
            marker = Path(tmp) / "inference-ran"
            script = Path(tmp) / "dwi_masking.py"
            script.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n", encoding="ascii")
            completed = subprocess.run([
                sys.executable, str(Path(cli.__file__)), "--subject-root", str(subject),
                "--no-user-config", "--targets", "microstructure", "--doctor", "--execute",
                "--pnl-mask-python", sys.executable, "--pnl-mask-script", str(script),
                "--pnl-mask-model-dir", tmp, "--pnl-mask-ants-bin-dir", tmp,
            ], capture_output=True, text=True, timeout=120)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout)["environment"]["kind"], "environment_doctor")
            self.assertFalse(marker.exists())
            self.assertFalse((input_dir / "nodif_brain_mask.nii.gz").exists())
            self.assertEqual(list(subject.rglob("*orchestration_manifest.json")), [])

    def test_batch_forwards_backend_and_paths_to_every_subject(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for subject_id in ("sub-01", "sub-02", "sub-03"):
                input_dir = root / subject_id / "input"
                input_dir.mkdir(parents=True)
                for name in ("data.nii.gz", "bvals", "bvecs"):
                    (input_dir / name).touch()
            paths = _path_options(root)

            def run_subject(command, subject, log_dir):
                return batch_cli.SubjectRunResult(
                    subject.subject_id, subject.subject_root, command, 0,
                    log_dir / "stdout", log_dir / "stderr", 0.0,
                    run_status='planned', status_source='child_manifest',
                )

            for backend in BACKENDS:
                with self.subTest(backend=backend):
                    forwarded = ["--brain-mask-backend=" + backend, *_path_flags(paths), "--no-user-config"]
                    manifest = root / "batch.json"
                    with (
                        patch.object(batch_cli, "_run_subject", side_effect=run_subject) as run,
                        redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()),
                    ):
                        self.assertEqual(batch_cli.main([
                            "--batch-root", str(root), "--jobs", "2", "--manifest", str(manifest), *forwarded,
                        ]), 0)
                    payload = json.loads(manifest.read_text(encoding="utf-8"))
                    self.assertEqual(payload["forwarded_args"], forwarded)
                    self.assertEqual(run.call_count, 3)
                    for command in payload["planned_commands"]:
                        self.assertEqual(command[-len(forwarded):], forwarded)
                        args = cli.build_parser().parse_args(command[2:])
                        self.assertEqual(args.brain_mask_backend, backend)
                        for key, path in paths.items():
                            self.assertEqual(getattr(args, key), path)


if __name__ == "__main__":
    unittest.main()
