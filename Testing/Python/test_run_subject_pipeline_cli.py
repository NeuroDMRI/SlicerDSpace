#!/usr/bin/env python3
"""Tests for the unified run_subject_pipeline CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CLI_PATH = REPO_ROOT / "run_subject_pipeline.py"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class RunSubjectPipelineCliTest(unittest.TestCase):
    def test_ddseg_only_act_fallback_requires_an_explicit_option(self):
        from shared.run_subject_pipeline_cli import build_parser
        parser = build_parser()
        self.assertFalse(parser.parse_args([]).allow_act_ddseg_only)
        self.assertTrue(parser.parse_args(["--allow-act-ddseg-only"]).allow_act_ddseg_only)

    def test_surface_precision_and_resource_defaults_follow_upstream(self) -> None:
        from shared.run_subject_pipeline_cli import build_parser
        defaults = build_parser().parse_args([])
        self.assertEqual(defaults.surface_precision, 'fp32')
        # None: the stage CPU reservation of a cohort run, else 4 (T3 S1.3).
        self.assertIsNone(defaults.surface_cpu_threads)
        self.assertEqual(defaults.surface_max_gpu_gib, 24.)
        selected = build_parser().parse_args(['--surface-precision', 'bf16',
            '--surface-cpu-threads', '2', '--surface-max-gpu-gib', '18'])
        self.assertEqual(selected.surface_precision, 'bf16')
        self.assertEqual(selected.surface_cpu_threads, 2)
        self.assertEqual(selected.surface_max_gpu_gib, 18.)

    def setUp(self) -> None:
        self._user_config_temp = tempfile.TemporaryDirectory()
        self._previous_user_config = os.environ.get("SLICERDSPACE_USER_CONFIG")
        os.environ["SLICERDSPACE_USER_CONFIG"] = str(
            Path(self._user_config_temp.name) / "isolated-user-config.json"
        )

    def tearDown(self) -> None:
        if self._previous_user_config is None:
            os.environ.pop("SLICERDSPACE_USER_CONFIG", None)
        else:
            os.environ["SLICERDSPACE_USER_CONFIG"] = self._previous_user_config
        self._user_config_temp.cleanup()

    def test_shared_user_config_reuses_subject_and_resolved_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject_root = root / "sub-01"
            input_dir = subject_root / "input"
            input_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (input_dir / name).write_text("x", encoding="utf-8")
            config_path = root / "saved-user-config.json"

            subprocess.run(
                [
                    sys.executable,
                    str(CLI_PATH),
                    "--user-config",
                    str(config_path),
                    "--subject-root",
                    str(subject_root),
                    "--targets",
                    "validation,microstructure",
                    "--microstructure-stages",
                    "freewater",
                    "--run-name",
                    "SavedRun",
                    "--save-user-config",
                    str(config_path),
                    "--print-manifest-only",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            saved = json.loads(config_path.read_text(encoding="utf-8"))
            completed = subprocess.run(
                [
                    sys.executable,
                    str(CLI_PATH),
                    "--user-config",
                    str(config_path),
                    "--print-manifest-only",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = json.loads(Path(completed.stdout.strip()).read_text(encoding="utf-8"))

        self.assertEqual(saved["recent"]["subject_root"], subject_root.resolve().as_posix())
        self.assertEqual(
            saved["run_defaults"]["targets"],
            ["validation", "microstructure", "microstructure_freewater"],
        )
        self.assertEqual(saved["run_defaults"]["microstructure_stages"], ["dti", "freewater"])
        self.assertEqual(manifest["subject_id"], "sub-01")
        self.assertEqual(manifest["run_name"], "SavedRun")
        self.assertEqual(manifest["targets"], ["validation", "microstructure", "microstructure_freewater"])

    def test_explicit_dependency_profile_overrides_saved_dependency_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # The environment doctor resolves user-supplied bin dirs before
            # reporting them; resolve here too so the comparison holds under
            # macOS's /tmp -> /private/tmp symlink.
            root = Path(tmp).resolve()
            subject_root = root / "sub-01"
            subject_root.mkdir()
            saved_bin = root / "saved" / "mrtrix" / "bin"
            profile_bin = root / "profile" / "mrtrix" / "bin"
            saved_bin.mkdir(parents=True)
            profile_bin.mkdir(parents=True)
            for bin_dir in (saved_bin, profile_bin):
                command = bin_dir / "mrconvert"
                command.write_text("#!/usr/bin/env sh\n", encoding="utf-8")
                command.chmod(0o755)
            user_config = root / "user-config.json"
            dependency_config = root / "dependency-config.json"
            user_config.write_text(
                json.dumps(
                    {
                        "recent": {"subject_root": subject_root.as_posix()},
                        "external_dependencies": {"mrtrix_bin_dir": saved_bin.as_posix()},
                    }
                ),
                encoding="utf-8",
            )
            dependency_config.write_text(
                json.dumps(
                    {"external_dependencies": {"mrtrix_bin_dir": profile_bin.as_posix()}}
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(CLI_PATH),
                    "--user-config",
                    str(user_config),
                    "--dependency-config",
                    str(dependency_config),
                    "--targets",
                    "tractography",
                    "--doctor",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            payload = json.loads(completed.stdout)
            mrconvert = next(
                item
                for item in payload["environment"]["dependencies"]
                if item["key"] == "command:mrconvert"
            )

        self.assertTrue(mrconvert["details"]["resolved"].startswith(profile_bin.as_posix()))

    def test_dry_run_all_writes_manifest_and_json_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            input_dir = subject_root / "input"
            input_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (input_dir / name).write_text("x", encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(CLI_PATH),
                    "--subject-root",
                    str(subject_root),
                    "--all",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(completed.stdout)
            self.assertEqual(payload["subject_id"], "100307")
            self.assertTrue(payload["run_manifest_path"].endswith("_orchestration_manifest.json"))
            task_names = [task["name"] for task in payload["tasks"]]
            self.assertIn("preprocessing.dwi", task_names)
            self.assertIn("segmentation.ddseg", task_names)
            self.assertIn("parcellation.ddparcel", task_names)
            self.assertIn("surface.ddsurfer", task_names)
            self.assertIn("microstructure.dti", task_names)
            self.assertIn("tractography.csd_act", task_names)
            self.assertIn("bundles.wmql", task_names)
            self.assertIn("connectome.streamline_count", task_names)

    def test_targets_and_skip_targets_override_default_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "subjectA"
            input_dir = subject_root / "input"
            input_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (input_dir / name).write_text("x", encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(CLI_PATH),
                    "--subject-root",
                    str(subject_root),
                    "--targets",
                    "validation,microstructure,tractography,connectome",
                    "--skip-targets",
                    "connectome",
                    "--tractography-backend",
                    "dti_det",
                    "--print-manifest-only",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            manifest_path = Path(completed.stdout.strip())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["subject_id"], "subjectA")
            self.assertEqual(manifest["targets"], ["validation", "microstructure", "tractography"])
            self.assertEqual(manifest["tractography_backend"], "dti_det")

    def test_cli_accepts_optional_algorithm_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "149741"
            input_dir = subject_root / "input"
            tractography_dir = subject_root / "tractography"
            input_dir.mkdir(parents=True)
            tractography_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (input_dir / name).write_text("x", encoding="utf-8")
            (tractography_dir / "149741_ukf_pp_downsampled.vtk").write_text("vtk", encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(CLI_PATH),
                    "--subject-root",
                    str(subject_root),
                    "--targets",
                    "saf,wma,tractcloud",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(completed.stdout)
            task_names = [task["name"] for task in payload["tasks"]]
            # C1: the closure adds the anatomy, surface and CSD/ACT tracking that SAF/WMA/TractCloud need.
            self.assertEqual([name for name in task_names if name.split("@")[0] in
                              {"tractography.saf", "bundles.wma", "bundles.tractcloud"}],
                             ["tractography.saf@csd_act", "bundles.tractcloud@csd_act", "bundles.wma@ukf"])
            for added in ("segmentation.ddseg", "parcellation.ddparcel", "surface.ddsurfer_postprocess"):
                self.assertIn(added, task_names)

    def test_cli_expands_bundle_backend_from_product_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "149741"
            input_dir = subject_root / "input"
            input_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (input_dir / name).write_text("x", encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(CLI_PATH),
                    "--subject-root",
                    str(subject_root),
                    "--targets",
                    "bundles",
                    "--bundle-backend",
                    "all",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(completed.stdout)
            task_names = [task["name"] for task in payload["tasks"]]
            self.assertEqual([name for name in task_names if name.startswith("bundles.")],
                             ["bundles.wmql@csd_act", "bundles.tractcloud@csd_act", "bundles.wma@ukf"])
            self.assertIn("parcellation.ddparcel", task_names)

    def test_cli_microstructure_stages_drive_freewater_noddi_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            input_dir = subject_root / "input"
            input_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (input_dir / name).write_text("x", encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(CLI_PATH),
                    "--subject-root",
                    str(subject_root),
                    "--targets",
                    "microstructure",
                    "--microstructure-stages",
                    "all",
                    "--print-manifest-only",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            manifest_path = Path(completed.stdout.strip())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["microstructure_stages"], ["dti", "freewater", "noddi", "dki"])
            self.assertTrue(manifest["microstructure_run_free_water"])
            self.assertTrue(manifest["microstructure_run_noddi"])

    def test_cli_records_microstructure_free_water_chunk_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            input_dir = subject_root / "input"
            input_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (input_dir / name).write_text("x", encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(CLI_PATH),
                    "--subject-root",
                    str(subject_root),
                    "--targets",
                    "microstructure",
                    "--microstructure-stages",
                    "freewater",
                    "--microstructure-free-water-chunk-size",
                    "1",
                    "--microstructure-free-water-engine",
                    "joblib",
                    "--microstructure-free-water-n-jobs",
                    "4",
                    "--microstructure-free-water-vox-per-chunk",
                    "128",
                    "--microstructure-free-water-max-fraction-for-tissue",
                    "0.65",
                    "--print-manifest-only",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            manifest_path = Path(completed.stdout.strip())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["microstructure_free_water_chunk_size"], 1)
            self.assertEqual(manifest["microstructure_free_water_engine"], "joblib")
            self.assertEqual(manifest["microstructure_free_water_n_jobs"], 4)
            self.assertEqual(manifest["microstructure_free_water_vox_per_chunk"], 128)
            self.assertEqual(manifest["microstructure_free_water_max_fraction_for_tissue"], 0.65)
            self.assertEqual(manifest["microstructure_stages"], ["dti", "freewater"])

    def test_cli_accepts_surface_postprocess_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "877168"
            subject_root.mkdir(parents=True)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(CLI_PATH),
                    "--subject-root",
                    str(subject_root),
                    "--targets",
                    "surface_postprocess",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(completed.stdout)
            self.assertEqual([task["name"] for task in payload["tasks"]], ["surface.ddsurfer", "surface.ddsurfer_postprocess"])
            self.assertEqual([task["requested"] for task in payload["tasks"]], [False, True])
            self.assertEqual(payload["tasks"][0]["required_by"], ["surface_postprocess"])
            self.assertEqual(payload["tasks"][1]["status"], "planned")

    def test_cli_records_surface_postprocess_atlases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "877168"
            subject_root.mkdir(parents=True)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(CLI_PATH),
                    "--subject-root",
                    str(subject_root),
                    "--targets",
                    "surface_postprocess",
                    "--postprocess-atlases",
                    "aparc,aparc.a2009s",
                    "--print-manifest-only",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            manifest_path = Path(completed.stdout.strip())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["postprocess_atlases"], ["aparc", "aparc.a2009s"])

    def test_cli_records_preprocessing_route_options(self) -> None:
        """W2a (D1): the only route option left is --preprocessing-skip-gibbs."""
        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            input_dir = subject_root / "input"
            input_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (input_dir / name).write_text("x", encoding="utf-8")

            completed = subprocess.run(
                [sys.executable, str(CLI_PATH), "--subject-root", str(subject_root), "--targets", "preprocessing",
                 "--preprocessing-skip-gibbs", "--print-manifest-only"],
                check=True, capture_output=True, text=True,
            )

            manifest = json.loads(Path(completed.stdout.strip()).read_text(encoding="utf-8"))
            self.assertFalse(manifest["preprocessing_dipy_run_gibbs"])
            for removed in ("preprocessing_backend", "preprocessing_dipy_denoise_method",
                            "preprocessing_dipy_run_motion_correction"):
                self.assertNotIn(removed, manifest)

    def test_w2a_each_removed_preprocessing_flag_fails_with_the_single_route_message(self) -> None:
        """W2a (D1): an old route, denoise or motion flag stops the run and names the one route."""
        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            (subject_root / "input").mkdir(parents=True)
            for flag in (["--preprocessing-backend", "dipy"], ["--preprocessing-backend", "mrtrix_fsl"],
                         ["--preprocessing-backend=dipy"], ["--preprocessing-dipy-denoise-method", "localpca"],
                         ["--preprocessing-dipy-denoise-method=none"], ["--preprocessing-skip-motion"]):
                with self.subTest(flag=flag):
                    completed = subprocess.run(
                        [sys.executable, str(CLI_PATH), "--subject-root", str(subject_root), "--targets", "preprocessing",
                         *flag, "--print-manifest-only"],
                        capture_output=True, text=True,
                    )
                    name = flag[0].split("=")[0]
                    self.assertEqual(completed.returncode, 2, completed.stderr)
                    self.assertIn("Preprocessing has one route: DIPY MP-PCA, DIPY Gibbs removal, FSL eddy/topup "
                                  f"through dwifslpreproc; remove {name}", completed.stderr)

    def test_doctor_writes_environment_report_and_install_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            input_dir = subject_root / "input"
            input_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (input_dir / name).write_text("x", encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(CLI_PATH),
                    "--subject-root",
                    str(subject_root),
                    "--targets",
                    "validation",
                    "--doctor",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(completed.stdout)
            self.assertEqual(payload["environment"]["kind"], "environment_doctor")
            report_paths = payload["report_paths"]
            self.assertTrue(Path(report_paths["json"]).is_file())
            self.assertTrue(Path(report_paths["python_sh"]).is_file())
            self.assertTrue(Path(report_paths["python_ps1"]).is_file())
            self.assertTrue(Path(report_paths["external_notes"]).is_file())

    def test_doctor_records_requested_cuda_device(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            input_dir = subject_root / "input"
            input_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (input_dir / name).write_text("x", encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(CLI_PATH),
                    "--subject-root",
                    str(subject_root),
                    "--targets",
                    "segmentation",
                    "--ddseg-device",
                    "cuda",
                    "--doctor",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(completed.stdout)
            dependency_keys = {item["key"] for item in payload["environment"]["dependencies"]}
            self.assertIn("device:segmentation:cuda", dependency_keys)

    def test_plan_writes_target_plan_without_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            input_dir = subject_root / "input"
            input_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (input_dir / name).write_text("x", encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(CLI_PATH),
                    "--subject-root",
                    str(subject_root),
                    "--targets",
                    "validation,tractography",
                    "--plan",
                    "--skip-environment-check",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(completed.stdout)
            plan = payload["run_plan"]
            self.assertEqual(plan["kind"], "slicerddsurfer_run_plan")
            self.assertTrue(Path(payload["run_plan_path"]).is_file())
            items = {item["target"]: item for item in plan["items"]}
            self.assertEqual(items["validation"]["status"], "will_run")
            self.assertEqual(items["tractography"]["status"], "blocked")
            self.assertEqual(items["tractography"]["reason_category"], "upstream")

    def test_execute_allows_partial_run_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "subjectA"
            subject_root.mkdir(parents=True)
            missing_wma_root = Path(tmp) / "missing-wma"
            missing_ukf = Path(tmp) / "missing-ukf"
            env = {key: value for key, value in os.environ.items() if "WMA" not in key and "SLICERDSPACE_WMA" not in key}

            completed = subprocess.run(
                [
                    sys.executable,
                    str(CLI_PATH),
                    "--subject-root",
                    str(subject_root),
                    "--no-user-config",
                    "--targets",
                    "wma",
                    "--wma-root",
                    str(missing_wma_root),
                    "--ukf-executable",
                    str(missing_ukf),
                    "--execute",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )

            payload = json.loads(completed.stdout)
            self.assertEqual(payload["tasks"][0]["status"], "blocked")
            self.assertIn("preflight missing dependencies", payload["tasks"][0]["reason"])
            manifest = json.loads(Path(payload["run_manifest_path"]).read_text(encoding="utf-8"))
            self.assertIn("wma", manifest["preflight_blocked_targets"])

    def test_strict_environment_aborts_on_preflight_blockers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "subjectA"
            subject_root.mkdir(parents=True)
            missing_wma_root = Path(tmp) / "missing-wma"
            env = {key: value for key, value in os.environ.items() if "WMA" not in key and "SLICERDSPACE_WMA" not in key}

            completed = subprocess.run(
                [
                    sys.executable,
                    str(CLI_PATH),
                    "--subject-root",
                    str(subject_root),
                    "--no-user-config",
                    "--targets",
                    "wma",
                    "--wma-root",
                    str(missing_wma_root),
                    "--execute",
                    "--strict-environment",
                ],
                capture_output=True,
                text=True,
                env=env,
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn("Aborting before execution", completed.stderr)


class MrtrixNthreadsArgumentTest(unittest.TestCase):
    """--mrtrix-nthreads is an optional, off-by-default passthrough setting."""

    def test_defaults_to_none(self) -> None:
        from shared.run_subject_pipeline_cli import build_parser

        args = build_parser().parse_args([])
        self.assertIsNone(args.mrtrix_nthreads)

    def test_accepts_an_explicit_value(self) -> None:
        from shared.run_subject_pipeline_cli import build_parser

        args = build_parser().parse_args(["--mrtrix-nthreads", "4"])
        self.assertEqual(args.mrtrix_nthreads, 4)

    def test_reaches_the_orchestrator_config(self) -> None:
        from shared.orchestrator import ProductOrchestratorConfig
        from shared.run_subject_pipeline_cli import build_parser

        args = build_parser().parse_args(["--mrtrix-nthreads", "8"])
        self.assertEqual(args.mrtrix_nthreads, 8)
        # ProductOrchestratorConfig.mrtrix_nthreads must exist with the same
        # off-by-default (None) semantics as the CLI flag, so
        # args.mrtrix_nthreads can be forwarded to it directly.
        self.assertIsNone(ProductOrchestratorConfig.__dataclass_fields__["mrtrix_nthreads"].default)


if __name__ == "__main__":
    unittest.main()
