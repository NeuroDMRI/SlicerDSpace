#!/usr/bin/env python3
"""Tests for subject-root product orchestration and resume planning."""

from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _write_test_volume(path: Path) -> None:
    import nibabel as nib
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.ones((2, 2, 2), dtype=np.float32)
    image = nib.MGHImage(data, np.eye(4)) if path.name.endswith(".mgz") else nib.Nifti1Image(data, np.eye(4))
    nib.save(image, str(path))


def _write_test_surface(path: Path) -> None:
    import trimesh

    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.creation.icosphere(subdivisions=1).export(path)


def _ddseg_artifact_fixture(directory: Path):
    from shared.pipeline import _artifact_record
    names = dict(segmentation='SegmentationMap_GMWMCSF', gm_probability='ProbabilisticMap_GM',
                 wm_probability='ProbabilisticMap_WM', csf_probability='ProbabilisticMap_CSF')
    return {key: _artifact_record(directory / (name + '.nii.gz'), verify_content=True) for key, name in names.items()}


def _write_valid_raw_subject(subject_root: Path, shape=(5, 5, 5)):
    import nibabel as nib
    import numpy as np
    directory = subject_root / 'input'
    directory.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.ones((*shape, 4), dtype=np.float32), np.eye(4)), directory / 'data.nii.gz')
    mask = np.zeros(shape, dtype=np.uint8)
    if min(shape) <= 2:
        mask[...] = 1
    else:
        mask[1:4, 1:4, 1:4] = 1
    nib.save(nib.Nifti1Image(mask, np.eye(4)), directory / 'nodif_brain_mask.nii.gz')
    (directory / 'bvals').write_text('0 1000 1000 1000')
    (directory / 'bvecs').write_text('0 1 0 0\n0 0 1 0\n0 0 0 1\n')


def _write_test_tractogram(path: Path, *, length_mm: float = 20.0, count: int = 1) -> None:
    import nibabel as nib
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    streamline = np.asarray(((0.0, 0.0, 0.0), (length_mm, 0.0, 0.0)), dtype=np.float32)
    tractogram = nib.streamlines.Tractogram([streamline] * count, affine_to_rasmm=np.eye(4))
    nib.streamlines.save(tractogram, path.as_posix())


def _file_record(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": path.resolve().as_posix(),
        "exists": True,
        "is_file": path.is_file(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _wmql_query_fixture(folder):
    import shutil
    from tractography.wmql import bundled_wmql_query_path, wmql_query_dependencies
    query = bundled_wmql_query_path()
    # Packaged rules are available without an installed author backend.
    directories = [query.parent]
    dependencies = wmql_query_dependencies(query, directories)
    snapshot = folder / 'queries'
    snapshot.mkdir(exist_ok=True)
    for source in dependencies:
        shutil.copy2(source, snapshot / Path(source).name)
    staged = snapshot / query.name
    return dict(query_path=str(query), staged_query=str(staged),
                query_sha256=hashlib.sha256(query.read_bytes()).hexdigest(),
                query_dependencies=dependencies, query_snapshot_dependencies=wmql_query_dependencies(staged, [snapshot]),
                query_include_dirs=list(map(str, directories)))


def _write_current_act_prep(layout, subject_id: str, segmentation: Path) -> dict[str, Path]:
    """Write current ACT prep outputs recording the current segmentation."""

    from tractography.act_prep import ACT_PREP_VERSION

    layout.act_prep.mkdir(parents=True, exist_ok=True)
    five_tt = layout.act_prep / f"{subject_id}_5tt_from_labels.nii.gz"
    gmwmi = layout.act_prep / f"{subject_id}_gmwmi_from_labels.nii.gz"
    five_tt.write_text("5tt", encoding="utf-8")
    gmwmi.write_text("gmwmi", encoding="utf-8")
    seg_record = {**_file_record(segmentation), "exists": True}
    parcellation = layout.act_prep / "source_parcellation.nii.gz"
    parcellation.write_text("parcellation", encoding="utf-8")
    (layout.act_prep / "act_prep_manifest.json").write_text(
        json.dumps({"version": ACT_PREP_VERSION, "status": "completed", "inputs": {"segmentation": seg_record,
            "parcellation": {**_file_record(parcellation), "exists": True}}}) + "\n",
        encoding="utf-8",
    )
    (layout.act_prep / "act_prep_qc.json").write_text(
        json.dumps({"version": ACT_PREP_VERSION, "status": "completed", "sgm_from_ddparcel": True,
            "five_tt_stats": {"channel_volume_voxel_units": {"sgm": 1.0}}}) + "\n", encoding="utf-8"
    )
    return {"five_tt": five_tt, "gmwmi": gmwmi}


class ProductOrchestratorTest(unittest.TestCase):
    def test_report_time_is_included_in_wall_clock_but_not_legacy_elapsed(self):
        import time
        from shared import ProductOrchestratorConfig, run_subject_products
        def delayed_report(**kwargs):
            time.sleep(.03)
            raise RuntimeError('synthetic report failure')
        with tempfile.TemporaryDirectory() as tmp, mock.patch('shared.reporting.write_subject_reports', side_effect=delayed_report):
            result = run_subject_products(ProductOrchestratorConfig('timing', Path(tmp), targets=('validation',), execute=False))
            manifest = json.loads(result.run_manifest_path.read_text())
            self.assertGreaterEqual(manifest['wall_time_seconds'] - manifest['elapsed_seconds'], .03)
            self.assertGreaterEqual(manifest['phase_timings_seconds']['subject_reports_and_refresh'], .03)
            self.assertEqual(manifest['report_error'], 'synthetic report failure')

    def test_reusing_run_name_archives_previous_manifest_and_plan(self) -> None:
        from shared import ProductOrchestratorConfig, run_subject_products

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            run_name = "repeatable-run"
            first = run_subject_products(
                ProductOrchestratorConfig(
                    subject_id="100307",
                    subject_root=subject_root,
                    targets=("microstructure",),
                    run_name=run_name,
                    execute=False,
                )
            )
            first_manifest = json.loads(first.run_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(first_manifest["targets"], ["microstructure"])
            self.assertEqual(first_manifest["previous_run_records"], {})

            second = run_subject_products(
                ProductOrchestratorConfig(
                    subject_id="100307",
                    subject_root=subject_root,
                    targets=("surface_postprocess",),
                    run_name=run_name,
                    execute=False,
                )
            )
            second_manifest = json.loads(second.run_manifest_path.read_text(encoding="utf-8"))
            archived = second_manifest["previous_run_records"]

            self.assertEqual(second_manifest["targets"], ["surface", "surface_postprocess"])
            self.assertEqual(set(archived), {"orchestration_manifest", "run_plan"})
            for path in archived.values():
                self.assertTrue(Path(path).is_file())
            archived_manifest = json.loads(Path(archived["orchestration_manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(archived_manifest["targets"], ["microstructure"])

    def test_unhandled_stage_error_writes_failed_orchestration_manifest(self) -> None:
        from shared import ProductOrchestratorConfig, run_subject_products

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            config = ProductOrchestratorConfig(
                subject_id="100307",
                subject_root=subject_root,
                targets=("surface",),
                run_name="failure-checkpoint",
                execute=True,
            )
            with mock.patch(
                "shared.orchestrator._run_subject_products_impl",
                side_effect=RuntimeError("synthetic stage failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic stage failure"):
                    run_subject_products(config)

            manifest_path = (
                subject_root
                / "outputs"
                / "00_run_metadata"
                / "failure-checkpoint_orchestration_manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["failure"]["type"], "RuntimeError")
            self.assertEqual(manifest["failure"]["message"], "synthetic stage failure")

    def test_dry_run_plans_microstructure_target(self) -> None:
        from shared import ProductOrchestratorConfig, run_subject_products

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            input_dir = subject_root / "input"
            input_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (input_dir / name).write_text("x", encoding="utf-8")
            result = run_subject_products(
                ProductOrchestratorConfig(
                    subject_id="100307",
                    subject_root=subject_root,
                    targets=("microstructure",),
                    execute=False,
                )
            )

            self.assertEqual(len(result.tasks), 1)
            self.assertEqual(result.tasks[0].name, "microstructure.dti")
            self.assertEqual(result.tasks[0].status, "planned")

    def test_rejects_invalid_microstructure_free_water_chunk_size(self) -> None:
        from shared import ProductOrchestratorConfig, run_subject_products

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            input_dir = subject_root / "input"
            input_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (input_dir / name).write_text("x", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "microstructure_free_water_chunk_size"):
                run_subject_products(
                    ProductOrchestratorConfig(
                        subject_id="100307",
                        subject_root=subject_root,
                        targets=("microstructure",),
                        execute=False,
                        microstructure_free_water_chunk_size=0,
                    )
                )

            with self.assertRaisesRegex(ValueError, "microstructure_free_water_engine"):
                run_subject_products(
                    ProductOrchestratorConfig(
                        subject_id="100307",
                        subject_root=subject_root,
                        targets=("microstructure",),
                        execute=False,
                        microstructure_free_water_engine="invalid",
                    )
                )

            with self.assertRaisesRegex(ValueError, "microstructure_free_water_max_fraction_for_tissue"):
                run_subject_products(
                    ProductOrchestratorConfig(
                        subject_id="100307",
                        subject_root=subject_root,
                        targets=("microstructure",),
                        execute=False,
                        microstructure_free_water_max_fraction_for_tissue=1.0,
                    )
                )

    def test_dry_run_plans_structural_targets(self) -> None:
        from shared import ProductOrchestratorConfig, run_subject_products

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            input_dir = subject_root / "input"
            input_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (input_dir / name).write_text("x", encoding="utf-8")

            result = run_subject_products(
                ProductOrchestratorConfig(
                    subject_id="100307",
                    subject_root=subject_root,
                    targets=("segmentation", "parcellation", "surface"),
                    execute=False,
                )
            )

            statuses = {task.name: task.status for task in result.tasks}
            self.assertEqual(statuses["segmentation.ddseg"], "planned")
            self.assertEqual(statuses["parcellation.ddparcel"], "planned")
            self.assertEqual(statuses["surface.ddsurfer"], "planned")

    def test_dry_run_plans_preprocessing_target(self) -> None:
        from shared import ProductOrchestratorConfig, run_subject_products

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            input_dir = subject_root / "input"
            input_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (input_dir / name).write_text("x", encoding="utf-8")

            result = run_subject_products(
                ProductOrchestratorConfig(
                    subject_id="100307",
                    subject_root=subject_root,
                    targets=("preprocessing",),
                    execute=False,
                )
            )

            self.assertEqual(len(result.tasks), 1)
            self.assertEqual(result.tasks[0].name, "preprocessing.dwi")
            self.assertEqual(result.tasks[0].status, "planned")

    def test_preprocessing_resume_checks_inputs_settings_force_and_failed_state(self) -> None:
        from dataclasses import replace
        from preprocessing import run_dwi_preprocessing
        from shared import ProductOrchestratorConfig, plan_subject_products
        from shared.orchestrator import _preprocessing_complete, _preprocessing_config
        from test_preprocessing_pipeline import _write_fake_backend, _write_subject

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject = _write_subject(root)
            bin_dir = root / "bin"
            _write_fake_backend(bin_dir)
            config = ProductOrchestratorConfig(
                subject_id="100307", subject_root=subject, targets=("preprocessing",),
                execute=False,
                brain_mask_backend="provided",
                preprocessing_mrtrix_bin_dir=bin_dir, preprocessing_fsl_bin_dir=bin_dir,
            )
            result = run_dwi_preprocessing(_preprocessing_config(config, subject, subject / "outputs/01_preproc"))
            self.assertTrue(_preprocessing_complete(subject, "100307", config=config))
            self.assertEqual(plan_subject_products(config, write=False).items[0].status, "skipped")
            changed = (
                replace(config, preprocessing_dipy_run_gibbs=False),
                replace(config, preprocessing_eddy_second_level_model="linear"),
                replace(config, preprocessing_force=True),
            )
            for settings in changed:
                self.assertFalse(_preprocessing_complete(subject, "100307", config=settings))
                self.assertEqual(plan_subject_products(settings, write=False).items[0].status, "will_run")
            manifest = json.loads(result.manifest_path.read_text())
            manifest["status"] = "failed"
            result.manifest_path.write_text(json.dumps(manifest))
            self.assertFalse(_preprocessing_complete(subject, "100307"))
            manifest["status"] = "completed"
            result.manifest_path.write_text(json.dumps(manifest))
            (subject / "input/bvals").write_text("0 2000 2000\n")
            self.assertFalse(_preprocessing_complete(subject, "100307", config=config))
            self.assertFalse(_preprocessing_complete(subject, "100307"))

    def test_dry_run_plans_selected_tractography_bundle_connectome_targets(self) -> None:
        from shared import ProductOrchestratorConfig, run_subject_products

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            input_dir = subject_root / "input"
            input_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (input_dir / name).write_text("x", encoding="utf-8")

            result = run_subject_products(
                ProductOrchestratorConfig(
                    subject_id="100307",
                    subject_root=subject_root,
                    targets=("tractography", "bundles", "connectome"),
                    run_name="csd_act_test",
                    execute=False,
                )
            )

            statuses = {task.name: task.status for task in result.tasks}
            self.assertEqual(statuses["tractography.csd_act"], "planned")
            self.assertEqual(statuses["bundles.wmql"], "planned")
            self.assertEqual(statuses["connectome.streamline_count"], "planned")
            manifest = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["run_name"], "csd_act_test")
            self.assertEqual(manifest["tractography_backend"], "csd_act")
            self.assertFalse(manifest["execute"])
            self.assertEqual(manifest["select_streamlines"], 1000000)
            self.assertEqual(manifest["connectome_backend"], "auto")

    def test_preflight_blocked_targets_are_recorded_but_do_not_stop_other_targets(self) -> None:
        from shared import ProductOrchestratorConfig, run_subject_products

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            input_dir = subject_root / "input"
            input_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (input_dir / name).write_text("x", encoding="utf-8")

            result = run_subject_products(
                ProductOrchestratorConfig(
                    subject_id="100307",
                    subject_root=subject_root,
                    targets=("microstructure", "tractography", "connectome"),
                    execute=False,
                    preflight_blocked_targets={"tractography": "preflight missing dependencies: MRtrix3"},
                )
            )

            statuses = {task.name: task.status for task in result.tasks}
            self.assertEqual(statuses["tractography.csd_act"], "blocked")
            self.assertEqual(statuses["microstructure.dti"], "planned")
            self.assertEqual(statuses["connectome.streamline_count"], "planned")
            manifest = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["preflight_blocked_targets"]["tractography"], "preflight missing dependencies: MRtrix3")
            self.assertNotIn("tractography", manifest["run_targets"])

    def test_run_plan_distinguishes_upstream_and_environment_blockers(self) -> None:
        from shared import ProductOrchestratorConfig, plan_subject_products

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            input_dir = subject_root / "input"
            input_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (input_dir / name).write_text("x", encoding="utf-8")
            atlas_t2 = Path(tmp) / "atlas_t2.nii.gz"
            atlas_t2.write_text("atlas", encoding="utf-8")

            plan = plan_subject_products(
                ProductOrchestratorConfig(
                    subject_id="100307",
                    subject_root=subject_root,
                    targets=("segmentation", "parcellation", "tractography", "bundles", "connectome"),
                    execute=True,
                    atlas_t2=atlas_t2,
                    preflight_blocked_targets={"tractography": "preflight missing dependencies: MRtrix3"},
                )
            )

            statuses = {item.target: item.status for item in plan.items}
            categories = {item.target: item.reason_category for item in plan.items}
            self.assertEqual(statuses["segmentation"], "will_run")
            self.assertEqual(statuses["parcellation"], "will_run")
            self.assertEqual(statuses["tractography"], "blocked")
            self.assertEqual(categories["tractography"], "environment")
            self.assertEqual(statuses["bundles"], "blocked")
            self.assertEqual(categories["bundles"], "upstream")
            self.assertEqual(statuses["connectome"], "blocked")
            self.assertEqual(categories["connectome"], "upstream")
            self.assertTrue(plan.run_plan_path.is_file())

    def test_orchestrator_writes_normalized_task_manifest_qc_and_logs(self) -> None:
        from shared import ProductOrchestratorConfig, run_subject_products

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            input_dir = subject_root / "input"
            input_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (input_dir / name).write_text("x", encoding="utf-8")

            result = run_subject_products(
                ProductOrchestratorConfig(
                    subject_id="100307",
                    subject_root=subject_root,
                    targets=("microstructure", "tractography"),
                    execute=False,
                    preflight_blocked_targets={"tractography": "preflight missing dependencies: MRtrix3"},
                )
            )

            manifest = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(Path(manifest["run_plan_path"]).is_file())
            self.assertTrue(Path(manifest["output_tree_audit_path"]).is_file())
            self.assertIsInstance(manifest["output_tree_warnings"], list)
            # C1: tractography is blocked by preflight; its prerequisites DDSeg and DDParcel are added to the plan.
            plan_items = {item["target"]: item for item in json.loads(Path(manifest["run_plan_path"]).read_text())["items"]}
            self.assertEqual(plan_items["microstructure"]["status"], "will_run")
            self.assertEqual([plan_items[name]["required_by"] for name in ("segmentation", "parcellation")],
                             [["tractography"], ["tractography"]])
            self.assertEqual(sum(item["status"] == "will_run" for item in plan_items.values()),
                             manifest["run_plan_summary"]["will_run"])
            for task in manifest["tasks"]:
                normalized_manifest = Path(task["normalized_manifest_path"])
                normalized_qc = Path(task["normalized_qc_path"])
                normalized_logs = Path(task["normalized_logs_dir"])
                self.assertTrue(normalized_manifest.is_file())
                self.assertTrue(normalized_qc.is_file())
                self.assertTrue(normalized_logs.is_dir())
                task_manifest = json.loads(normalized_manifest.read_text(encoding="utf-8"))
                self.assertEqual(task_manifest["schema"], "slicerddsurfer.task.v1")
                self.assertIn("inputs", task_manifest)
                self.assertIn("outputs", task_manifest)
                self.assertIn("dependencies", task_manifest)
                self.assertIn("runtime_seconds", task_manifest)

    def test_task_from_result_preserves_generated_path_outputs(self) -> None:
        from shared import orchestrator

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            class _BackendResult:
                def __init__(self) -> None:
                    self.output_dir = root / "bundles"
                    self.matrix_path = root / "connectome.csv"
                    self.manifest_path = root / "manifest.json"
                    self.qc_path = root / "qc.json"
                    self.command = ("tool", "--flag")
                    self.executed = True
                    self.backend_status = {"available": True}

            task = orchestrator._task_from_result("connectome.streamline_count", _BackendResult())

            self.assertEqual(
                task.outputs,
                {
                    "output_dir": (root / "bundles").as_posix(),
                    "matrix_path": (root / "connectome.csv").as_posix(),
                    "manifest_path": (root / "manifest.json").as_posix(),
                    "qc_path": (root / "qc.json").as_posix(),
                },
            )
            self.assertNotIn("command", task.outputs)
            self.assertGreater(len(task.outputs), 0)

    def test_output_tree_semantics_marks_unselected_directories(self) -> None:
        from shared import ProductOrchestratorConfig, run_subject_products

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            input_dir = subject_root / "input"
            input_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (input_dir / name).write_text("x", encoding="utf-8")

            result = run_subject_products(
                ProductOrchestratorConfig(
                    subject_id="100307",
                    subject_root=subject_root,
                    targets=("microstructure",),
                    execute=False,
                )
            )

            manifest = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
            audit = json.loads(Path(manifest["output_tree_audit_path"]).read_text(encoding="utf-8"))
            semantics = json.loads(Path(manifest["output_tree_semantics_path"]).read_text(encoding="utf-8"))
            self.assertNotIn("04_tractography", audit["expected_top_dirs"])
            by_path = {entry["path"]: entry for entry in semantics["entries"]}
            self.assertEqual(by_path["04_tractography"]["state"], "not_selected_absent")
            self.assertTrue(by_path["02_microstructure"]["selected"])

    def test_dry_run_honors_output_root_and_advanced_connectome_options(self) -> None:
        from shared import ProductOrchestratorConfig, run_subject_products

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            output_root = Path(tmp) / "custom_outputs"
            input_dir = subject_root / "input"
            input_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (input_dir / name).write_text("x", encoding="utf-8")

            result = run_subject_products(
                ProductOrchestratorConfig(
                    subject_id="100307",
                    subject_root=subject_root,
                    output_root=output_root,
                    targets=("tractography", "connectome"),
                    run_name="csd_act_100k",
                    execute=False,
                    select_streamlines=100000,
                    connectome_backend="mrtrix",
                    connectome_compress_mrtrix_dense=True,
                )
            )

            # run_manifest_path is written under the resolved output root
            # (see manifest["output_root"] below); resolve here too so the
            # comparison holds under macOS's /tmp -> /private/tmp symlink.
            self.assertTrue(result.run_manifest_path.is_relative_to(output_root.resolve()))
            manifest = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["output_root"], output_root.resolve().as_posix())
            self.assertEqual(manifest["tractography_backend"], "csd_act")
            self.assertEqual(manifest["select_streamlines"], 100000)
            self.assertEqual(manifest["connectome_backend"], "mrtrix")
            self.assertTrue(manifest["connectome_compress_mrtrix_dense"])
            profile = manifest["performance_profile"]
            self.assertEqual(profile["schema"], "slicerddsurfer.performance_profile.v1")
            self.assertGreaterEqual(manifest['wall_time_seconds'], manifest['elapsed_seconds'])
            self.assertGreaterEqual(manifest['phase_timings_seconds']['subject_reports_and_refresh'], 0)
            self.assertEqual(profile['wall_time_seconds'], manifest['wall_time_seconds'])
            self.assertTrue(Path(manifest["performance_profile_path"]).is_file())
            self.assertIn("slowest_tasks", profile)
            self.assertIn("largest_top_dirs", profile)
            self.assertTrue(any(row["path"] == "00_run_metadata" for row in profile["top_dirs"]))
            report_paths = manifest["report_paths"]
            output_tree_report_paths = manifest["output_tree_report_paths"]
            report_html = Path(report_paths["report_html"])
            output_tree_html = Path(output_tree_report_paths["report_html_latest"])
            qc_summary = json.loads(Path(report_paths["qc_summary_json"]).read_text(encoding="utf-8"))
            self.assertIn("Performance Profile", report_html.read_text(encoding="utf-8"))
            self.assertTrue(output_tree_html.is_file())
            self.assertIn("Annotated output tree", report_html.read_text(encoding="utf-8"))
            self.assertEqual(qc_summary["performance_profile"]["schema"], "slicerddsurfer.performance_profile.v1")
            self.assertIn("00_run_metadata", qc_summary["output_tree"]["top_dirs"])

    def test_dry_run_does_not_probe_auto_torch_devices(self) -> None:
        from shared import ProductOrchestratorConfig, run_subject_products

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            input_dir = subject_root / "input"
            input_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (input_dir / name).write_text("x", encoding="utf-8")

            with mock.patch(
                "shared.orchestrator.resolve_auto_torch_device",
                side_effect=AssertionError("dry-run must not probe torch devices"),
            ):
                result = run_subject_products(
                    ProductOrchestratorConfig(
                        subject_id="100307",
                        subject_root=subject_root,
                        targets=("segmentation", "surface", "tractcloud"),
                        ddseg_device="auto",
                        surface_device="auto",
                        tractcloud_device="auto",
                        execute=False,
                    )
                )

            manifest = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["resolved_ddseg_device"], "not_resolved_execute_false")
            self.assertEqual(manifest["resolved_surface_device"], "not_resolved_execute_false")
            self.assertEqual(manifest["resolved_tractcloud_device"], "not_resolved_execute_false")

    def test_dry_run_supports_dti_tractography_backend(self) -> None:
        from shared import ProductOrchestratorConfig, run_subject_products

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            input_dir = subject_root / "input"
            input_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (input_dir / name).write_text("x", encoding="utf-8")
            existing_tractogram = (
                subject_root
                / "outputs"
                / "04_tractography"
                / "tractograms"
                / "dti_det"
                / "windows_full"
                / "existing.trk"
            )
            existing_tractogram.parent.mkdir(parents=True)
            existing_tractogram.write_text("existing", encoding="utf-8")

            result = run_subject_products(
                ProductOrchestratorConfig(
                    subject_id="100307",
                    subject_root=subject_root,
                    targets=("tractography", "bundles", "connectome"),
                    run_name="dti_det_test",
                    tractography_backend="dti_det",
                    connectome_backend="python",
                    execute=False,
                )
            )

            statuses = {task.name: task.status for task in result.tasks}
            self.assertEqual(statuses["tractography.dti_det"], "planned")
            self.assertEqual(statuses["bundles.wmql"], "planned")
            self.assertEqual(statuses["connectome.streamline_count"], "planned")
            manifest = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["tractography_backend"], "dti_det")
            self.assertTrue(existing_tractogram.is_file())

    def test_microstructure_optional_stage_gates_are_explicit(self) -> None:
        from shared import ProductOrchestratorConfig, run_subject_products

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            input_dir = subject_root / "input"
            input_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (input_dir / name).write_text("x", encoding="utf-8")

            result = run_subject_products(
                ProductOrchestratorConfig(
                    subject_id="100307",
                    subject_root=subject_root,
                    targets=("microstructure",),
                    execute=False,
                    microstructure_run_free_water=True,
                    microstructure_run_noddi=True,
                )
            )

            self.assertEqual([task.name for task in result.tasks], ["microstructure.dti", "microstructure.freewater", "microstructure.noddi"])
            self.assertEqual([task.status for task in result.tasks], ["planned", "planned", "planned"])
            manifest = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(manifest["microstructure_run_free_water"])
            self.assertTrue(manifest["microstructure_run_noddi"])
            self.assertEqual(manifest["microstructure_stages"], ["dti", "freewater", "noddi"])

    def test_bundle_backend_expands_product_target_to_concrete_backends(self) -> None:
        from shared import ProductOrchestratorConfig, run_subject_products

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "149741"
            input_dir = subject_root / "input"
            input_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (input_dir / name).write_text("x", encoding="utf-8")

            result = run_subject_products(
                ProductOrchestratorConfig(
                    subject_id="149741",
                    subject_root=subject_root,
                    targets=("bundles",),
                    bundle_backend="all",
                    execute=False,
                )
            )

            # C1: the closure adds DDSeg, DDParcel and tractography; WMA runs on a UKF tractogram.
            self.assertEqual([task.name for task in result.tasks], [
                "segmentation.ddseg", "parcellation.ddparcel", "tractography.csd_act@csd_act", "bundles.wmql@csd_act",
                "bundles.tractcloud@csd_act", "tractography.ukf@ukf", "bundles.wma@ukf"])
            manifest = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["targets"], ["segmentation", "parcellation", "tractography", "bundles", "tractcloud", "wma"])
            self.assertEqual(manifest["bundle_backends"], ["wmql", "wma", "tractcloud"])

    def test_wma_plan_does_not_reuse_connectome_csd_tractogram_implicitly(self) -> None:
        from shared import ProductOrchestratorConfig, plan_subject_products

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            input_dir = subject_root / "input"
            input_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (input_dir / name).write_text("x", encoding="utf-8")
            slicer_executable = Path(tmp) / "Slicer"
            slicer_executable.write_text("", encoding="utf-8")
            atlas_dir = Path(tmp) / "ORG-Atlases"
            atlas_dir.mkdir()
            mrtrix_bin = Path(tmp) / "mrtrix" / "bin"
            mrtrix_bin.mkdir(parents=True)
            (mrtrix_bin / "tckconvert").write_text("", encoding="utf-8")

            plan = plan_subject_products(
                ProductOrchestratorConfig(
                    subject_id="100307",
                    subject_root=subject_root,
                    targets=("segmentation", "parcellation", "tractography", "wma"),
                    atlas_t2=input_dir / "data.nii.gz",
                    bundle_backend="wma",
                    execute=True,
                    wma_atlas_dir=atlas_dir,
                    slicer_executable=slicer_executable,
                    ukf_executable=Path(tmp) / "missing-ukf",
                    mrtrix_bin_dir=mrtrix_bin,
                ),
                write=False,
            )

            # C1: WMA now runs on its own UKF tracking variant; it never takes the CSD tractogram of the other variant.
            items = {item.task_name: item for item in plan.items}
            self.assertEqual(items["tractography.csd_act@csd_act"].status, "will_run")
            self.assertEqual(items["bundles.wma@ukf"].status, "blocked")
            self.assertIn("UKF", items["bundles.wma@ukf"].reason)
            self.assertNotIn("bundles.wma@csd_act", items)
            self.assertNotIn("input_tractography", items["bundles.wma@ukf"].inputs)

    def test_wma_no_resume_does_not_take_inner_cache_shortcut(self):
        from shared import ProductOrchestratorConfig, run_subject_products
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'input_ukf.vtk'
            source.write_text('fixture')
            atlas = root / 'atlas'
            atlas.mkdir()
            slicer = root / 'Slicer'
            slicer.touch()
            config = ProductOrchestratorConfig(subject_id='100307', subject_root=root / 'subject',
                targets=('wma',), wma_input_tractography=source, wma_atlas_dir=atlas,
                slicer_executable=slicer, execute=True, resume=False)
            with mock.patch('shared.orchestrator._wma_complete', return_value=True), mock.patch(
                    'tractography.run_wma_org_atlas', side_effect=RuntimeError('fresh WMA called')) as run:
                with self.assertRaisesRegex(RuntimeError, 'fresh WMA called'):
                    run_subject_products(config)
            run.assert_called_once()

    def test_saf_tractography_backend_expands_surface_postprocess_and_saf(self) -> None:
        from shared import ProductOrchestratorConfig, run_subject_products

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            input_dir = subject_root / "input"
            input_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (input_dir / name).write_text("x", encoding="utf-8")

            result = run_subject_products(
                ProductOrchestratorConfig(
                    subject_id="100307",
                    subject_root=subject_root,
                    targets=("tractography",),
                    tractography_backend="saf",
                    execute=False,
                )
            )

            # C1: SAF needs the surfaces (DDSurfer) and the ACT anatomy (DDSeg, DDParcel) for its CSD FOD.
            self.assertEqual([task.name for task in result.tasks], [
                "segmentation.ddseg", "parcellation.ddparcel", "surface.ddsurfer", "surface.ddsurfer_postprocess",
                "tractography.csd_act", "tractography.saf"])
            manifest = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["tractography_backend"], "saf")
            self.assertEqual(manifest["primary_tractography_backend"], "csd_act")

    def test_saf_plan_only_accepts_current_csd_tractography_as_future_fod(self) -> None:
        from shared import ProductOrchestratorConfig, plan_subject_products, subject_layout

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            input_dir = subject_root / "input"
            input_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (input_dir / name).write_text("x", encoding="utf-8")
            fs_dir = Path(tmp) / "freesurfer_subject"
            fs_dir.mkdir()

            dti_plan = plan_subject_products(
                ProductOrchestratorConfig(
                    subject_id="100307",
                    subject_root=subject_root,
                    targets=("tractography", "saf"),
                    tractography_backend="dti_det",
                    saf_fs_dir=fs_dir,
                    execute=True,
                    resume=False,
                )
            )
            dti_items = {item.target: item for item in dti_plan.items}
            self.assertEqual(dti_items["tractography"].status, "will_run")
            self.assertEqual(dti_items["saf"].status, "blocked")
            self.assertEqual(dti_items["saf"].dependencies["fod"], "missing")

            layout = subject_layout(subject_root)
            layout.ddseg.mkdir(parents=True, exist_ok=True)
            for name in (
                "SegmentationMap_GMWMCSF.nii.gz",
                "ProbabilisticMap_GM.nii.gz",
                "ProbabilisticMap_WM.nii.gz",
                "ProbabilisticMap_CSF.nii.gz",
            ):
                (layout.ddseg / name).write_text(name, encoding="utf-8")

            csd_plan = plan_subject_products(
                ProductOrchestratorConfig(
                    subject_id="100307",
                    subject_root=subject_root,
                    targets=("parcellation", "tractography", "saf"),
                    tractography_backend="csd_act",
                    atlas_t2=input_dir / "data.nii.gz",
                    saf_fs_dir=fs_dir,
                    execute=True,
                    resume=False,
                )
            )
            csd_items = {item.target: item for item in csd_plan.items}
            self.assertEqual(csd_items["tractography"].status, "will_run")
            self.assertEqual(csd_items["saf"].status, "will_run")
            self.assertEqual(csd_items["saf"].dependencies["fod"], "will_run:tractography.csd_act")

    def test_dry_run_supports_optional_algorithm_targets(self) -> None:
        from shared import ProductOrchestratorConfig, run_subject_products

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "149741"
            input_dir = subject_root / "input"
            tractography_dir = subject_root / "tractography"
            input_dir.mkdir(parents=True)
            tractography_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (input_dir / name).write_text("x", encoding="utf-8")
            (tractography_dir / "149741_ukf_pp_downsampled.vtk").write_text("vtk", encoding="utf-8")

            result = run_subject_products(
                ProductOrchestratorConfig(
                    subject_id="149741",
                    subject_root=subject_root,
                    targets=("saf", "wma", "tractcloud"),
                    execute=False,
                )
            )

            statuses = {task.name: task.status for task in result.tasks}
            self.assertEqual(statuses["tractography.saf@csd_act"], "planned")
            self.assertEqual(statuses["bundles.wma@ukf"], "planned")
            self.assertEqual(statuses["bundles.tractcloud@csd_act"], "planned")
            manifest = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["targets"], ["segmentation", "parcellation", "surface", "surface_postprocess",
                                                   "tractography", "saf", "tractcloud", "wma"])

    def test_tractcloud_uses_same_run_whole_brain_tractography_not_saf(self) -> None:
        from shared import ProductOrchestratorConfig, plan_subject_products, subject_layout

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            layout = subject_layout(subject_root)
            old_saf_dir = layout.saf / "old-run"
            old_saf_dir.mkdir(parents=True)
            (old_saf_dir / "SAF.vtk").write_text("old vtk", encoding="utf-8")

            run_name = "new-run"
            csd_dir = layout.csd / run_name
            csd_dir.mkdir(parents=True)
            (csd_dir / "100307_wmfod_norm.mif").write_text("fod", encoding="utf-8")
            layout.ddseg.mkdir(parents=True)
            (layout.ddseg / "SegmentationMap_GMWMCSF.nii.gz").write_text("seg", encoding="utf-8")
            fs_dir = Path(tmp) / "native-dmri-surfaces"
            fs_dir.mkdir()

            input_dir = subject_root / "input"
            input_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (input_dir / name).write_text("x", encoding="utf-8")

            plan = plan_subject_products(
                ProductOrchestratorConfig(
                    subject_id="100307",
                    subject_root=subject_root,
                    targets=("parcellation", "tractography", "saf", "tractcloud"),
                    atlas_t2=input_dir / "data.nii.gz",
                    run_name=run_name,
                    saf_fs_dir=fs_dir,
                    execute=True,
                    resume=False,
                )
            )

            items = {item.target: item for item in plan.items}
            self.assertEqual(items["saf"].status, "will_run")
            self.assertEqual(items["tractcloud"].status, "will_run")
            self.assertEqual(
                items["tractcloud"].inputs["input_tractography"],
                items["tractography"].outputs["tractogram_vtk"],
            )
            self.assertEqual(
                items["tractcloud"].dependencies["input_tractography"],
                "will_run:tractography.csd_act",
            )
            self.assertNotEqual(
                items["tractcloud"].inputs["input_tractography"],
                (old_saf_dir / "SAF.vtk").resolve(),
            )

    def test_execute_saf_defaults_to_1m_mask_seeding(self) -> None:
        from shared import ProductOrchestratorConfig, run_subject_products, subject_layout

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            input_dir = subject_root / "input"
            input_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (input_dir / name).write_text("x", encoding="utf-8")

            _write_valid_raw_subject(subject_root)
            layout = subject_layout(subject_root)
            csd_dir = layout.csd
            csd_dir.mkdir(parents=True)
            (csd_dir / "100307_wmfod_norm.mif").write_text("fod", encoding="utf-8")

            post_dir = layout.ddsurfer_postprocess
            (post_dir / "mri").mkdir(parents=True)
            (post_dir / "surf").mkdir()
            for name in ("brain.mgz",):
                (post_dir / "mri" / name).write_text(name, encoding="utf-8")
            for name in (
                "lh.white",
                "rh.white",
                "lh.pial",
                "rh.pial",
                "lh.inflated",
                "rh.inflated",
                "lh.sphere",
                "rh.sphere",
                "lh.curv",
                "rh.curv",
            ):
                (post_dir / "surf" / name).write_text(name, encoding="utf-8")
            (post_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
            (post_dir / "qc.json").write_text('{"status":"completed"}\n', encoding="utf-8")

            captured = {}

            class _FakeResult:
                def __init__(self, output_dir: Path) -> None:
                    self.qc_path = output_dir / "saf_qc.json"
                    self.manifest_path = output_dir / "saf_manifest.json"
                    self.saf_tractogram = output_dir / "SAF.tck"
                    self.saf_vtk_path = output_dir / "SAF.vtk"
                    self.streamline_count = 63589
                    self.qc_path.parent.mkdir(parents=True, exist_ok=True)
                    self.qc_path.write_text('{"status":"completed"}\n', encoding="utf-8")
                    self.manifest_path.write_text("{}\n", encoding="utf-8")
                    self.saf_tractogram.write_text("tck", encoding="utf-8")
                    self.saf_vtk_path.write_text("vtk", encoding="utf-8")

            def _fake_run_saf(config, *, log=None):
                captured["tckgen_options"] = config.tckgen_options
                return _FakeResult(config.output_dir)

            with mock.patch("tractography.run_saf", side_effect=_fake_run_saf):
                result = run_subject_products(
                    ProductOrchestratorConfig(
                        subject_id="100307",
                        subject_root=subject_root,
                        targets=("saf",),
                        execute=True,
                        excluded_targets=("segmentation", "parcellation", "surface_postprocess", "tractography"),
                    )
                )

            self.assertIn("-select 1000000", captured["tckgen_options"])
            self.assertIn("-seed_image", captured["tckgen_options"])
            self.assertEqual(result.tasks[0].outputs["streamline_count"], "63589")
            self.assertTrue(result.tasks[0].outputs["tractogram_vtk"].endswith("SAF.vtk"))

    def test_dry_run_supports_surface_postprocess_target(self) -> None:
        from shared import ProductOrchestratorConfig, run_subject_products

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "877168"
            result = run_subject_products(
                ProductOrchestratorConfig(
                    subject_id="877168",
                    subject_root=subject_root,
                    targets=("surface_postprocess",),
                    execute=False,
                )
            )

            # C1: surface postprocess needs the DDSurfer surfaces, so DDSurfer is added and marked as required.
            self.assertEqual([task.name for task in result.tasks], ["surface.ddsurfer", "surface.ddsurfer_postprocess"])
            self.assertEqual(result.tasks[1].status, "planned")
            manifest = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["targets"], ["surface", "surface_postprocess"])
            self.assertEqual([(task["requested"], task["required_by"]) for task in manifest["tasks"]],
                             [(False, ["surface_postprocess"]), (True, [])])

    def test_surface_postprocess_plan_requires_native_surfaces_and_wmparc(self) -> None:
        from shared import ProductOrchestratorConfig, plan_subject_products, subject_layout

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            input_dir = subject_root / "input"
            input_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (input_dir / name).write_text("x", encoding="utf-8")
            config = ProductOrchestratorConfig(
                subject_id="100307",
                subject_root=subject_root,
                targets=("surface_postprocess",),
                excluded_targets=("surface",),  # safety net: DDSurfer skipped on purpose, existing output is required
                execute=True,
                resume=False,
            )

            missing_surface = {item.target: item for item in plan_subject_products(config).items}
            self.assertEqual(missing_surface["surface_postprocess"].status, "blocked")
            self.assertIn("*_orig.stl", missing_surface["surface_postprocess"].reason)

            layout = subject_layout(subject_root)
            native_dir = layout.ddsurfer / "pred_results_SlicerDSpace" / "native" / "100307"
            native_dir.mkdir(parents=True)
            for surface in ("wm", "pial"):
                for hemisphere in ("lh", "rh"):
                    (native_dir / f"100307_predicted_{surface}_{hemisphere}_orig.stl").write_text(
                        "surface",
                        encoding="utf-8",
                    )
            layout.ddparcel.mkdir(parents=True)

            missing_wmparc = {item.target: item for item in plan_subject_products(config).items}
            self.assertEqual(missing_wmparc["surface_postprocess"].status, "blocked")
            self.assertIn("wmparc", missing_wmparc["surface_postprocess"].reason)

            (layout.ddparcel / "100307-DDSurfer-wmparc.nii.gz").write_text("labels", encoding="utf-8")
            ready = {item.target: item for item in plan_subject_products(config).items}
            self.assertEqual(ready["surface_postprocess"].status, "will_run")

    def test_surface_postprocess_plan_accepts_selected_upstream_products(self) -> None:
        from shared import ProductOrchestratorConfig, plan_subject_products

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            input_dir = subject_root / "input"
            input_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (input_dir / name).write_text("x", encoding="utf-8")
            atlas_t2 = Path(tmp) / "atlas.nii.gz"
            atlas_t2.write_text("atlas", encoding="utf-8")

            plan = plan_subject_products(
                ProductOrchestratorConfig(
                    subject_id="100307",
                    subject_root=subject_root,
                    targets=("surface", "parcellation", "surface_postprocess"),
                    atlas_t2=atlas_t2,
                    execute=True,
                    resume=False,
                )
            )
            items = {item.target: item for item in plan.items}

            self.assertEqual(items["surface"].status, "will_run")
            self.assertEqual(items["parcellation"].status, "will_run")
            self.assertEqual(items["surface_postprocess"].status, "will_run")
            self.assertEqual(items["surface_postprocess"].dependencies["surface"], "will_run")
            self.assertEqual(items["surface_postprocess"].dependencies["parcellation"], "not_required")

    def test_surface_postprocess_resume_and_fresh_run_require_all_core_files(self) -> None:
        from shared import ProductOrchestratorConfig, plan_subject_products, run_subject_products, subject_layout

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            layout = subject_layout(subject_root)
            post_dir = layout.ddsurfer_postprocess
            (post_dir / "mri").mkdir(parents=True)
            (post_dir / "surf").mkdir()
            (post_dir / "mri" / "brain.mgz").write_text("brain", encoding="utf-8")
            for hemisphere in ("lh", "rh"):
                for surface in ("white", "pial", "inflated", "sphere", "curv"):
                    (post_dir / "surf" / f"{hemisphere}.{surface}").write_text("surface", encoding="utf-8")
            (post_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
            (post_dir / "qc.json").write_text('{"status":"completed"}\n', encoding="utf-8")

            resume_config = ProductOrchestratorConfig(
                subject_id="100307",
                subject_root=subject_root,
                targets=("surface_postprocess",),
                excluded_targets=("surface",),
                execute=True,
                resume=True,
            )
            self.assertEqual(plan_subject_products(resume_config).items[0].status, "skipped")

            (post_dir / "surf" / "rh.curv").unlink()
            left_resume_config = ProductOrchestratorConfig(
                subject_id="100307",
                subject_root=subject_root,
                targets=("surface_postprocess",),
                excluded_targets=("surface",),
                execute=True,
                resume=True,
                postprocess_mode="left",
            )
            self.assertEqual(plan_subject_products(left_resume_config).items[0].status, "skipped")
            self.assertNotEqual(plan_subject_products(resume_config).items[0].status, "skipped")
            (post_dir / "surf" / "rh.curv").write_text("surface", encoding="utf-8")

            (post_dir / "surf" / "lh.curv").unlink()
            self.assertNotEqual(plan_subject_products(resume_config).items[0].status, "skipped")

            native_dir = layout.ddsurfer / "pred_results_SlicerDSpace" / "native" / "100307"
            native_dir.mkdir(parents=True)
            for hemisphere in ("lh", "rh"):
                for surface in ("wm", "pial"):
                    (native_dir / f"100307_predicted_{surface}_{hemisphere}_orig.stl").write_text("surface", encoding="utf-8")
            layout.ddparcel.mkdir(parents=True)
            (layout.ddparcel / "100307-DDSurfer-wmparc.nii.gz").write_text("labels", encoding="utf-8")

            class _IncompleteResult:
                def __init__(self) -> None:
                    self.structural_root = post_dir
                    self.stats_root = layout.ddsurfer_postprocess_stats
                    self.staged_input_dir = post_dir / "input"
                    self.manifest_path = post_dir / "manifest.json"
                    self.qc_path = post_dir / "qc.json"

            class _FakePostprocessConfig:
                def __init__(self, **kwargs) -> None:
                    self.__dict__.update(kwargs)

            from anatomy import surface
            with mock.patch.dict(
                surface.__dict__,
                {
                    "DDSurferPostprocessConfig": _FakePostprocessConfig,
                    "run_ddsurfer_postprocess": mock.Mock(return_value=_IncompleteResult()),
                },
            ):
                fresh_result = run_subject_products(
                    ProductOrchestratorConfig(
                        subject_id="100307",
                        subject_root=subject_root,
                        targets=("surface_postprocess",),
                        excluded_targets=("surface",),
                        execute=True,
                        resume=False,
                    )
                )
            self.assertEqual(fresh_result.tasks[0].status, "failed")
            self.assertIn("required non-empty", fresh_result.tasks[0].reason)

    def test_surface_postprocess_execute_false_preserves_requested_atlases(self) -> None:
        from shared import ProductOrchestratorConfig, run_subject_products

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "877168"
            result = run_subject_products(
                ProductOrchestratorConfig(
                    subject_id="877168",
                    subject_root=subject_root,
                    targets=("surface_postprocess",),
                    execute=False,
                    postprocess_atlases=("aparc", "aparc.a2009s"),
                )
            )

            manifest = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["targets"], ["surface", "surface_postprocess"])
            self.assertEqual(manifest["postprocess_atlases"], ["aparc", "aparc.a2009s"])

    def test_resume_skips_completed_existing_outputs(self) -> None:
        from shared import ProductOrchestratorConfig, run_subject_products, subject_layout

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            input_dir = subject_root / "input"
            input_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (input_dir / name).write_text("input", encoding="utf-8")
            _write_valid_raw_subject(subject_root, shape=(2, 2, 2))
            layout = subject_layout(subject_root)
            run_name = "csd_act_test"
            csd_dir = layout.csd / run_name
            tract_dir = layout.tractograms / run_name
            wmql_dir = layout.bundles_wmql / run_name
            conn_dir = layout.connectome / run_name
            seg_dir = layout.ddseg
            parcel_dir = layout.ddparcel
            surface_dir = layout.ddsurfer / "pred_results_SlicerDSpace" / "native" / "100307"
            for directory in (csd_dir, tract_dir, wmql_dir, conn_dir, seg_dir, parcel_dir, surface_dir):
                directory.mkdir(parents=True)
            for name in (
                "SegmentationMap_GMWMCSF.nii.gz",
                "ProbabilisticMap_GM.nii.gz",
                "ProbabilisticMap_WM.nii.gz",
                "ProbabilisticMap_CSF.nii.gz",
            ):
                (seg_dir / name).write_text(name, encoding="utf-8")
            (seg_dir / "ddseg_qc.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "schema_version": 2,
                        "orientation_convention": "affine_reoriented_ras_model_input_v2",
                        "probability_semantics": "four_class_softmax_tissue_channels_v1",
                        "output_artifacts": _ddseg_artifact_fixture(seg_dir),
                        "probability_qc": {
                            "finite_and_bounded": True,
                            "tissue_mass_valid_fraction": 1.0,
                            "label_agreement": 1.0,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            for name in (
                "100307-DDSurfer-wmparc.nii.gz",
                "100307-DDSurfer-wmparc-Reg.mgz",
                "100307-dti-FractionalAnisotropy-Reg-NormMasked.nii.gz",
                "100307-dti-Trace-Reg-NormMasked.nii.gz",
                "100307-dti-MinEigenvalue-Reg-NormMasked.nii.gz",
                "100307-dti-MidEigenvalue-Reg-NormMasked.nii.gz",
            ):
                _write_test_volume(parcel_dir / name)
            for name in (
                "100307_predicted_wm_lh.stl",
                "100307_predicted_wm_rh.stl",
                "100307_predicted_pial_lh.stl",
                "100307_predicted_pial_rh.stl",
            ):
                _write_test_surface(surface_dir / name)
            csd_tck = tract_dir / "100307_csd_act_ifod2.tck"
            from anatomy.surface.backend_bridge import upstream_record
            (layout.ddsurfer / 'surface_manifest.json').write_text(json.dumps(
                dict(status='completed', upstream_commit=upstream_record()['commit'])))
            (layout.ddsurfer / 'surface_qc.json').write_text(json.dumps(dict(automatic_qc_passed=True)))
            csd_trk = tract_dir / "100307_csd_act_ifod2.trk"
            csd_vtk = tract_dir / "100307_csd_act_ifod2.vtk"
            csd_weights = tract_dir / "100307_csd_act_ifod2_sift2_weights.txt"
            _write_test_tractogram(csd_tck)
            _write_test_tractogram(csd_trk)
            csd_vtk.write_text(
                "# vtk DataFile Version 3.0\nASCII\nDATASET POLYDATA\n"
                "POINTS 2 float\n0 0 0 20 0 0\nLINES 1 3\n2 0 1\n",
                encoding="utf-8",
            )
            csd_weights.write_text("1.0\n", encoding="utf-8")
            raw_paths = {
                "dwi": input_dir / "data.nii.gz",
                "bval": input_dir / "bvals",
                "bvec": input_dir / "bvecs",
                "mask": input_dir / "nodif_brain_mask.nii.gz",
            }
            csd_summary = {
                "tck": {
                    "file": _file_record(csd_tck),
                    "streamline_count": 1,
                    "all_points_finite": True,
                    "length_mm": {"min": 20.0, "median": 20.0, "max": 20.0},
                },
                "trk": {
                    "file": _file_record(csd_trk),
                    "streamline_count": 1,
                    "all_points_finite": True,
                },
                "vtk": {
                    "file": _file_record(csd_vtk),
                    "streamline_count": 1,
                },
                "sift2": {
                    "file": _file_record(csd_weights),
                    "weight_count": 1,
                    "all_finite": True,
                    "all_nonnegative": True,
                },
            }
            (csd_dir / "csd_act_qc.json").write_text(
                json.dumps({"status": "completed", "scientific_output_summary": csd_summary}) + "\n",
                encoding="utf-8",
            )
            act_inputs = _write_current_act_prep(layout, "100307", seg_dir / "SegmentationMap_GMWMCSF.nii.gz")
            (csd_dir / "csd_act_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "version": 3,
                        "parameters": {
                            "select_streamlines": 1,
                            "min_length_mm": 10.0,
                            "max_length_mm": 250.0,
                        },
                        "dependencies": {
                            **{key: _file_record(path) for key, path in raw_paths.items()},
                            **{key: _file_record(path) for key, path in act_inputs.items()},
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            wmql_bundle = wmql_dir / "100307_desc-wmql_CST.left.trk"
            _write_test_tractogram(wmql_bundle, count=20)
            (wmql_dir / "wmql_qc.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "version": 6,
                        "official_multibundle": True,
                        "execution_passed": True,
                        "output_integrity_passed": True,
                        "automatic_qc_passed": True,
                        "output_summary": {"bundle_files": [wmql_bundle.as_posix()]},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            parcellation_path = parcel_dir / "100307-DDSurfer-wmparc.nii.gz"
            from microstructure.pipeline import _file_signature
            from shared.orchestrator import _parcellation_outputs

            (parcel_dir / "ddparcel_qc.json").write_text(json.dumps({
                "status": "completed", "automatic_qc_passed": True, "checks": {"synthetic_fixture": True},
                "output_artifacts": {str(path.resolve()): _file_signature(path)
                    for path in _parcellation_outputs(subject_root, "100307").values()},
            }))
            (wmql_dir / "wmql_manifest.json").write_text(
                json.dumps(
                    {
                        "query_selection": [],
                        **_wmql_query_fixture(wmql_dir),
                        "dependencies": {
                            "input_tractography": _file_record(csd_trk),
                            "atlas": _file_record(parcellation_path),
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (conn_dir / "100307-connectome-streamline_count_qc.json").write_text(
                '{"status": "completed", "node_count": 2, "edge_count": 1}\n', encoding="utf-8"
            )
            (conn_dir / "100307-connectome-streamline_count_manifest.json").write_text(
                json.dumps(
                    {
                        "dependencies": {
                            "tractography": _file_record(csd_tck),
                            "ddparcel_parcellation": _file_record(parcellation_path),
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (conn_dir / "100307-connectome-streamline_count.csv").write_text(
                "source_node,target_node,streamline_count\n10,20,1\n", encoding="utf-8"
            )
            (conn_dir / "100307-connectome-nodes.tsv").write_text(
                "node_id\tlabel\tname\n10\t10\tLeft\n20\t20\tRight\n", encoding="utf-8"
            )
            weighted_dir = conn_dir / "sift2_weighted_streamline_count"
            weighted_dir.mkdir()
            weighted_stem = "100307-connectome-sift2_weighted_streamline_count"
            (weighted_dir / f"{weighted_stem}_qc.json").write_text(
                '{"status": "completed", "node_count": 2, "edge_count": 1}\n',
                encoding="utf-8",
            )
            (weighted_dir / f"{weighted_stem}.csv").write_text(
                "source_node,target_node,sift2_weighted_streamline_count\n10,20,1.0\n",
                encoding="utf-8",
            )
            (weighted_dir / "100307-connectome-nodes.tsv").write_text(
                "node_id\tlabel\tname\n10\t10\tLeft\n20\t20\tRight\n",
                encoding="utf-8",
            )
            (weighted_dir / f"{weighted_stem}_manifest.json").write_text(
                json.dumps(
                    {
                        "dependencies": {
                            "tractography": _file_record(csd_tck),
                            "ddparcel_parcellation": _file_record(parcellation_path),
                            "sift2_weights": _file_record(csd_weights),
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            micro_dir = layout.microstructure_dti
            micro_dir.mkdir(parents=True)
            for suffix in ("FA", "MD", "AD", "RD", "L1", "L2", "L3", "Trace"):
                _write_test_volume(micro_dir / f"100307_dti_{suffix}.nii.gz")
            (micro_dir / "100307-dti-DEC.nrrd").write_bytes(b"dec")  # D6: a completed DTI stage has the DEC map
            (micro_dir / "dti_qc.json").write_text(json.dumps({
                "status": "completed", "automatic_qc_passed": True, "checks": {"synthetic_fixture": True},
                "output_artifacts": {str(path.resolve()): _file_signature(path) for path in micro_dir.glob("*.nii.gz")},
            }))
            (layout.microstructure / "manifest.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "raw_input_bundle": {key: _file_record(path) for key, path in raw_paths.items()},
                        "fit_method": "wls",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            from microstructure.pipeline import DTI_KEYS, _compute_roi_stats

            _compute_roi_stats(
                metric_paths={key: micro_dir / f"100307_dti_{'Trace' if key == 'trace' else key.upper()}.nii.gz"
                              for key in DTI_KEYS},
                roi_label_map=parcellation_path,
                brain_mask_path=raw_paths['mask'],
                lut_path=None,
                csv_path=layout.regional_microstructure / "100307_ROIWise-Microstructure.csv",
                json_path=layout.regional_microstructure / "100307_ROIWise-Microstructure.json",
            )

            from types import SimpleNamespace
            verified_surface = SimpleNamespace(
                manifest_path=layout.ddsurfer / 'surface_manifest.json',
                qc_path=layout.ddsurfer / 'surface_qc.json',
                generated_models=tuple(surface_dir.glob('*.stl')),
            )
            with mock.patch('anatomy.surface.inspect_surface_resources', return_value=dict(available=True)), \
                    mock.patch('anatomy.surface.pipeline.prepare_shared_registered_bundle') as shared_bundle, \
                    mock.patch('anatomy.surface.run_surface', return_value=verified_surface) as surface_run:
                result = run_subject_products(
                    ProductOrchestratorConfig(
                        subject_id="100307",
                        subject_root=subject_root,
                        targets=("segmentation", "parcellation", "surface", "microstructure", "tractography", "bundles", "connectome"),
                        run_name=run_name,
                        select_streamlines=1,
                        # T-B2: the fixture connectome has no backend_used, so it is a python connectome; the
                        # default auto choice would resolve to MRtrix for this .tck source and re-run it.
                        connectome_backend="python",
                        execute=True,
                        resume=True,
                    )
                )
            self.assertTrue(surface_run.call_args.args[0].resume)
            self.assertEqual(surface_run.call_args.args[0].precision, 'fp32')
            shared_bundle.assert_not_called()  # R2: outside cohort mode the preparation stays inside run_surface

            self.assertEqual(
                [task.status for task in result.tasks],
                ["skipped", "skipped", "completed", "skipped", "skipped", "skipped", "skipped", "skipped"],
            )
            self.assertTrue(all("completed" in task.reason for task in result.tasks if task.status == 'skipped'))

    def test_resume_rejects_deleted_wmql_and_connectome_products(self) -> None:
        from shared import ProductOrchestratorConfig, plan_subject_products, subject_layout

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            input_dir = subject_root / "input"
            input_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (input_dir / name).write_text("x", encoding="utf-8")

            run_name = "resume_artifacts"
            layout = subject_layout(subject_root)
            layout.ddparcel.mkdir(parents=True)
            parcellation = layout.ddparcel / "100307-DDSurfer-wmparc.nii.gz"
            parcellation.write_text("parcel", encoding="utf-8")
            tract_dir = layout.tractograms_dti_det / run_name
            tract_dir.mkdir(parents=True)
            tractogram = tract_dir / "100307-dti-deterministic.trk"
            _write_test_tractogram(tractogram)

            wmql_dir = layout.bundles_wmql / run_name
            wmql_dir.mkdir(parents=True)
            bundle = wmql_dir / "100307_desc-wmql_CST.left.trk"
            _write_test_tractogram(bundle, count=20)
            (wmql_dir / "wmql_qc.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "version": 6,
                        "official_multibundle": True,
                        "execution_passed": True,
                        "output_integrity_passed": True,
                        "automatic_qc_passed": True,
                        "output_summary": {"bundle_files": [bundle.as_posix()]},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (wmql_dir / "wmql_manifest.json").write_text(
                json.dumps(
                    {
                        "query_selection": [],
                        **_wmql_query_fixture(wmql_dir),
                        "dependencies": {
                            "input_tractography": _file_record(tractogram),
                            "atlas": _file_record(parcellation),
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            conn_dir = layout.connectome / run_name
            conn_dir.mkdir(parents=True)
            conn_qc = conn_dir / "100307-connectome-streamline_count_qc.json"
            conn_matrix = conn_dir / "100307-connectome-streamline_count.csv"
            conn_nodes = conn_dir / "100307-connectome-nodes.tsv"
            conn_qc.write_text('{"status": "completed", "node_count": 2, "edge_count": 1}\n', encoding="utf-8")
            conn_matrix.write_text("source_node,target_node,streamline_count\n10,20,1\n", encoding="utf-8")
            conn_nodes.write_text("node_id\tlabel\tname\n10\t10\tLeft\n20\t20\tRight\n", encoding="utf-8")
            (conn_dir / "100307-connectome-streamline_count_manifest.json").write_text(
                json.dumps(
                    {
                        "dependencies": {
                            "tractography": _file_record(tractogram),
                            "ddparcel_parcellation": _file_record(parcellation),
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            config = ProductOrchestratorConfig(
                subject_id="100307",
                subject_root=subject_root,
                targets=("bundles", "connectome"),
                run_name=run_name,
                tractography_backend="dti_det",
                execute=True,
                resume=True,
            )
            complete_plan = plan_subject_products(config)
            self.assertEqual({item.target: item.status for item in complete_plan.items if item.requested}, {"bundles": "skipped", "connectome": "skipped"})
            from shared import run_subject_products
            from shared.result_scene import result_candidates
            result = run_subject_products(config)
            payload = json.loads(result.run_manifest_path.read_text(encoding="utf-8"))
            candidates, _ = result_candidates(payload)
            self.assertTrue({conn_matrix.resolve().as_posix(), conn_nodes.resolve().as_posix()} <=
                            {candidate["path"] for candidate in candidates})

            conn_matrix.write_text("source_node,target_node,streamline_count\n", encoding="utf-8")
            conn_nodes.write_text("node_id\tlabel\tname\n", encoding="utf-8")
            header_only_plan = plan_subject_products(config)
            self.assertEqual({item.target: item.status for item in header_only_plan.items}["connectome"], "will_run")

            conn_nodes.write_text("node_id\tlabel\tname\n10\t10\tLeft\n20\t20\tRight\n", encoding="utf-8")
            bundle.write_text("", encoding="utf-8")
            conn_matrix.write_text("", encoding="utf-8")
            empty_plan = plan_subject_products(config)
            empty_statuses = {item.target: item.status for item in empty_plan.items}
            self.assertEqual(empty_statuses["bundles"], "will_run")
            self.assertEqual(empty_statuses["connectome"], "will_run")

            _write_test_tractogram(bundle)
            conn_matrix.write_text("source_node,target_node,streamline_count\n10,20,1\n", encoding="utf-8")
            bundle.unlink()
            conn_matrix.unlink()
            incomplete_plan = plan_subject_products(config)
            statuses = {item.target: item.status for item in incomplete_plan.items}
            self.assertEqual(statuses["bundles"], "will_run")
            self.assertEqual(statuses["connectome"], "will_run")

    def test_plan_resume_uses_external_output_root_for_structural_dependencies(self) -> None:
        from shared import ProductOrchestratorConfig, plan_subject_products, subject_layout

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            output_root = Path(tmp) / "external_outputs" / "100307"
            input_dir = subject_root / "input"
            input_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (input_dir / name).write_text("x", encoding="utf-8")

            layout = subject_layout(subject_root, output_root=output_root)
            layout.ddseg.mkdir(parents=True)
            for name in (
                "SegmentationMap_GMWMCSF.nii.gz",
                "ProbabilisticMap_GM.nii.gz",
                "ProbabilisticMap_WM.nii.gz",
                "ProbabilisticMap_CSF.nii.gz",
            ):
                (layout.ddseg / name).write_text(name, encoding="utf-8")
            (layout.ddseg / "ddseg_qc.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "schema_version": 2,
                        "orientation_convention": "affine_reoriented_ras_model_input_v2",
                        "probability_semantics": "four_class_softmax_tissue_channels_v1",
                        "output_artifacts": _ddseg_artifact_fixture(layout.ddseg),
                        "probability_qc": {
                            "finite_and_bounded": True,
                            "tissue_mass_valid_fraction": 1.0,
                            "label_agreement": 1.0,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            layout.ddparcel.mkdir(parents=True)
            for name in (
                "100307-DDSurfer-wmparc.nii.gz",
                "100307-DDSurfer-wmparc-Reg.mgz",
                "100307-dti-FractionalAnisotropy-Reg-NormMasked.nii.gz",
                "100307-dti-Trace-Reg-NormMasked.nii.gz",
                "100307-dti-MinEigenvalue-Reg-NormMasked.nii.gz",
                "100307-dti-MidEigenvalue-Reg-NormMasked.nii.gz",
            ):
                _write_test_volume(layout.ddparcel / name)

            from microstructure.pipeline import _file_signature
            from shared.orchestrator import _parcellation_outputs

            (layout.ddparcel / "ddparcel_qc.json").write_text(json.dumps({
                "status": "completed", "automatic_qc_passed": True, "checks": {"synthetic_fixture": True},
                "output_artifacts": {str(path.resolve()): _file_signature(path)
                    for path in _parcellation_outputs(subject_root, "100307", output_root).values()},
            }))

            surface_dir = layout.ddsurfer / "pred_results_SlicerDSpace" / "native" / "100307"
            surface_dir.mkdir(parents=True)
            for name in (
                "100307_predicted_wm_lh.stl",
                "100307_predicted_wm_rh.stl",
                "100307_predicted_pial_lh.stl",
                "100307_predicted_pial_rh.stl",
            ):
                _write_test_surface(surface_dir / name)
            from anatomy.surface.backend_bridge import upstream_record
            (layout.ddsurfer / 'surface_manifest.json').write_text(json.dumps(
                dict(status='completed', upstream_commit=upstream_record()['commit'])))
            (layout.ddsurfer / 'surface_qc.json').write_text(json.dumps(dict(automatic_qc_passed=True)))

            plan = plan_subject_products(
                ProductOrchestratorConfig(
                    subject_id="100307",
                    subject_root=subject_root,
                    output_root=output_root,
                    targets=("segmentation", "parcellation", "surface", "tractography", "connectome"),
                    execute=True,
                    resume=True,
                ),
                write=False,
            )

            statuses = {item.target: item.status for item in plan.items}
            reasons = {item.target: item.reason for item in plan.items}
            dependencies = {item.target: item.dependencies for item in plan.items}
            self.assertEqual(statuses["segmentation"], "skipped")
            self.assertEqual(statuses["parcellation"], "skipped")
            self.assertEqual(statuses["surface"], "will_run")
            self.assertEqual(statuses["tractography"], "will_run")
            self.assertEqual(statuses["connectome"], "will_run")
            self.assertEqual(dependencies["tractography"]["segmentation"], "available")
            self.assertEqual(dependencies["connectome"]["parcellation"], "available")
            self.assertEqual(dependencies["connectome"]["tractogram"], "will_run")
            self.assertNotIn("missing DDSeg", reasons["tractography"])
            self.assertNotIn("missing parcellation", reasons["connectome"])
            self.assertEqual(plan.output_root, output_root.resolve())

    def test_execute_blocks_missing_dependencies_instead_of_crashing(self) -> None:
        from shared import ProductOrchestratorConfig, run_subject_products

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            result = run_subject_products(
                ProductOrchestratorConfig(
                    subject_id="100307",
                    subject_root=subject_root,
                    targets=("segmentation", "parcellation", "surface", "microstructure", "tractography", "bundles", "connectome"),
                    execute=True,
                )
            )

            statuses = {task.name: task.status for task in result.tasks}
            self.assertEqual(statuses["segmentation.ddseg"], "blocked")
            self.assertEqual(statuses["parcellation.ddparcel"], "blocked")
            self.assertEqual(statuses["surface.ddsurfer"], "blocked")
            self.assertEqual(statuses["microstructure.dti"], "blocked")
            self.assertEqual(statuses["tractography.csd_act"], "blocked")
            self.assertEqual(statuses["bundles.wmql"], "blocked")
            self.assertEqual(statuses["connectome.streamline_count"], "blocked")

    def test_execute_saf_auto_discovers_completed_ddsurfer_postprocess_outputs(self) -> None:
        from shared import ProductOrchestratorConfig, run_subject_products, subject_layout

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "877168"
            input_dir = subject_root / "input"
            input_dir.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (input_dir / name).write_text("x", encoding="utf-8")

            _write_valid_raw_subject(subject_root)
            layout = subject_layout(subject_root)
            csd_dir = layout.csd
            csd_dir.mkdir(parents=True)
            (csd_dir / "877168_wmfod_norm.mif").write_text("fod", encoding="utf-8")

            micro_dir = layout.microstructure_dti
            micro_dir.mkdir(parents=True)
            (micro_dir / "877168_dti_FA.nii.gz").write_text("fa", encoding="utf-8")

            post_dir = layout.ddsurfer_postprocess
            (post_dir / "mri").mkdir(parents=True)
            (post_dir / "surf").mkdir()
            (post_dir / "label").mkdir()
            (layout.ddsurfer_postprocess_stats / "stats").mkdir(parents=True)
            for name in ("brain.mgz",):
                (post_dir / "mri" / name).write_text(name, encoding="utf-8")
            for name in (
                "lh.white",
                "rh.white",
                "lh.pial",
                "rh.pial",
                "lh.inflated",
                "rh.inflated",
                "lh.sphere",
                "rh.sphere",
                "lh.curv",
                "rh.curv",
            ):
                (post_dir / "surf" / name).write_text(name, encoding="utf-8")
            (post_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
            (post_dir / "qc.json").write_text('{"status":"completed"}\n', encoding="utf-8")

            captured: dict[str, Path] = {}

            class _FakeResult:
                def __init__(self, output_dir: Path) -> None:
                    self.qc_path = output_dir / "saf_qc.json"
                    self.manifest_path = output_dir / "saf_manifest.json"
                    self.saf_tractogram = output_dir / "SAF.tck"
                    self.saf_vtk_path = None
                    self.streamline_count = None
                    self.qc_path.parent.mkdir(parents=True, exist_ok=True)
                    self.qc_path.write_text('{"status":"completed"}\n', encoding="utf-8")
                    self.manifest_path.write_text("{}\n", encoding="utf-8")
                    self.saf_tractogram.write_text("tck", encoding="utf-8")

            def _fake_run_saf(config, *, log=None):
                captured["fs_dir"] = config.fs_dir
                return _FakeResult(config.output_dir)

            with mock.patch("tractography.run_saf", side_effect=_fake_run_saf):
                result = run_subject_products(
                    ProductOrchestratorConfig(
                        subject_id="877168",
                        subject_root=subject_root,
                        targets=("saf",),
                        execute=True,
                        excluded_targets=("segmentation", "parcellation", "surface_postprocess", "tractography"),
                    )
                )

            self.assertEqual(result.tasks[0].name, "tractography.saf")
            self.assertEqual(result.tasks[0].status, "completed")
            self.assertEqual(captured["fs_dir"], post_dir)


class MrtrixNthreadsValidationTest(unittest.TestCase):
    """Invalid mrtrix_nthreads must fail fast, before any target runs."""

    def test_negative_mrtrix_nthreads_is_rejected_by_run_subject_products(self) -> None:
        from shared import ProductOrchestratorConfig, run_subject_products

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            with self.assertRaises(ValueError):
                run_subject_products(
                    ProductOrchestratorConfig(
                        subject_id="100307",
                        subject_root=subject_root,
                        targets=("tractography",),
                        execute=False,
                        mrtrix_nthreads=-1,
                    )
                )

    def test_negative_mrtrix_nthreads_is_rejected_by_plan_subject_products(self) -> None:
        from shared import ProductOrchestratorConfig, plan_subject_products

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            with self.assertRaises(ValueError):
                plan_subject_products(
                    ProductOrchestratorConfig(
                        subject_id="100307",
                        subject_root=subject_root,
                        targets=("tractography",),
                        mrtrix_nthreads=-1,
                    )
                )


if __name__ == "__main__":
    unittest.main()
