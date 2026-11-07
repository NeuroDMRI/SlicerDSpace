#!/usr/bin/env python3
"""Tests for pure run-controller UI helpers."""

from __future__ import annotations

import json
import unittest
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from shared.ui_run_controller import (
    BackendSelection,
    ProductTargetFlags,
    RunSelection,
    active_bundle_backend_sections,
    blocked_plan_items,
    blocked_plan_log_text,
    build_background_cli_args,
    build_batch_run_kwargs,
    build_batch_background_cli_args,
    build_single_subject_run_kwargs,
    cancelled_manifest_path_and_payload,
    classify_background_run_outcome,
    default_run_name_for_backend,
    extract_background_manifest_path,
    environment_blockers_from_plan,
    is_auto_generated_run_name,
    microstructure_stage_flags,
    parse_manifest_line,
    partial_run_notice_text,
    rows_from_batch_manifest,
    rows_from_plan,
    rows_from_result,
    rows_from_single_subject_manifest,
    run_summary_from_manifest,
    selected_environment_targets,
    selected_orchestrator_targets,
    selected_product_targets,
    selected_products_summary,
    selected_workflows,
    streamline_control_state,
    task_keys_for_running_line,
    validate_batch_run_options,
    validate_single_subject_inputs,
    runnable_targets_from_plan,
)


class UIRunControllerTest(unittest.TestCase):
    def test_run_name_helpers_keep_neutral_default_and_legacy_detection(self) -> None:
        self.assertEqual(default_run_name_for_backend("csd_act"), "Test1")
        self.assertTrue(is_auto_generated_run_name(""))
        self.assertTrue(is_auto_generated_run_name("Test1"))
        self.assertTrue(is_auto_generated_run_name("csd_act_ui"))
        self.assertFalse(is_auto_generated_run_name("subject42"))

    def test_microstructure_stage_flags_support_all_and_legacy_booleans(self) -> None:
        self.assertEqual(microstructure_stage_flags("dti"), (False, False))
        self.assertEqual(microstructure_stage_flags("dti,freewater,noddi"), (True, True))
        self.assertEqual(microstructure_stage_flags("all"), (True, True))
        self.assertEqual(microstructure_stage_flags("", legacy_freewater=True), (True, False))

    def test_selected_targets_expand_backends(self) -> None:
        flags = ProductTargetFlags(
            preprocessing=True,
            segmentation=True,
            microstructure=True,
            tractography=True,
            bundles=True,
            connectome=True,
        )

        targets = selected_product_targets(
            flags,
            tractography_backend="saf",
            bundle_backend="all",
            microstructure_freewater=True,
            microstructure_noddi=False,
        )

        self.assertIn("preprocessing", targets)
        self.assertIn("microstructure_freewater", targets)
        self.assertIn("surface_postprocess", targets)
        self.assertIn("saf", targets)
        self.assertIn("wma", targets)
        self.assertIn("tractcloud", targets)
        self.assertEqual(selected_environment_targets(("validation", "validation", "bundles")), ("validation", "bundles"))

    def test_selected_workflow_and_orchestrator_labels(self) -> None:
        flags = ProductTargetFlags(surface=True, surface_postprocess=True, parcellation=True, bundles=True)

        self.assertEqual(
            selected_workflows(flags),
            [
                "Surface / DDSurfer native STL",
                "Surface postprocess",
                "Parcellation / DDParcel",
                "Fiber bundles",
            ],
        )
        self.assertEqual(
            selected_orchestrator_targets(flags),
            ("validation", "parcellation", "surface_postprocess", "bundles"),
        )

    def test_surface_postprocess_is_an_explicit_independent_product_target(self) -> None:
        targets = selected_product_targets(
            ProductTargetFlags(surface_postprocess=True),
            tractography_backend="dti_det",
            bundle_backend="wmql",
            microstructure_freewater=False,
            microstructure_noddi=False,
        )

        # C1: surface postprocess needs the DDSurfer surfaces, so the closure adds "surface".
        self.assertEqual(targets, ("validation", "surface", "surface_postprocess"))

    def test_summary_and_streamline_state_are_backend_aware(self) -> None:
        summary, target_text = selected_products_summary(
            ("validation", "tractography", "bundles"),
            BackendSelection(tractography="dti_det", bundles="wmql", microstructure_stages="dti"),
        )

        self.assertEqual(target_text, "tractography, bundles")
        self.assertIn("tractography=dti_det", summary)
        self.assertIn("bundles=wmql", summary)
        dti_state = streamline_control_state("dti_det")
        self.assertFalse(dti_state.enabled)
        self.assertIn("ignored", dti_state.tooltip)
        saf_state = streamline_control_state("saf")
        self.assertTrue(saf_state.enabled)
        self.assertEqual(saf_state.label, "ACT/SAF streamlines")

    def test_active_bundle_backend_sections(self) -> None:
        self.assertEqual(active_bundle_backend_sections("wmql"), set())
        self.assertEqual(active_bundle_backend_sections("wma"), {"wma"})
        self.assertEqual(active_bundle_backend_sections("tractcloud"), {"tractcloud"})
        self.assertEqual(active_bundle_backend_sections("all"), {"wma", "tractcloud"})
        self.assertEqual(active_bundle_backend_sections("wmql,tractcloud"), {"tractcloud"})

    def test_multi_tracking_count_control_explains_ukf_independence(self):
        state = streamline_control_state('csd_act,ukf')
        self.assertTrue(state.enabled)
        self.assertIn('seeds-per-voxel', state.tooltip)
        self.assertFalse(streamline_control_state('ukf,dti_det').enabled)

    def test_task_keys_for_running_line_maps_orchestrator_messages(self) -> None:
        self.assertEqual(
            task_keys_for_running_line("[Orchestrator] Running DWI preprocessing"),
            ("preprocessing", "preprocessing.dwi"),
        )
        self.assertEqual(
            task_keys_for_running_line("[Orchestrator] Running TractCloud bundle extraction"),
            ("tractcloud", "bundles.tractcloud"),
        )
        self.assertEqual(task_keys_for_running_line("unrelated"), ())

    def test_run_summary_from_manifest_reads_qc_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qc_path = root / "qc_summary.json"
            qc_path.write_text(
                '{"summary": {"status": "passed", "elapsed_time": "2 s", '
                '"interpretation": "ready", "tasks": {"completed": 2}}}',
                encoding="utf-8",
            )

            message, status = run_summary_from_manifest(
                {"report_paths": {"qc_summary_json_latest": qc_path.as_posix()}}
            )

            self.assertEqual(status, "passed")
            # The interpretation repeats the counts for a passed run, so it is hidden.
            self.assertEqual(message, "Passed · 2 s\n2 completed")

    def _summary_for(self, summary: dict) -> tuple[str, str]:
        with tempfile.TemporaryDirectory() as tmp:
            qc_path = Path(tmp) / "qc_summary.json"
            qc_path.write_text(json.dumps({"summary": summary}), encoding="utf-8")
            return run_summary_from_manifest({"report_paths": {"qc_summary_json_latest": qc_path.as_posix()}})

    def test_finished_run_summaries_are_recognised_in_the_new_format(self) -> None:
        # slicer_run_plan_panel_smoke.py waited for "Status:", which f6bab13 removed from the manifest summary
        from shared.ui_run_controller import summary_names_run_status

        for status in ("completed", "failed", "partial", "blocked", "running", "planned", "unknown"):
            message, _ = self._summary_for({"status": status, "elapsed_time": "4 s", "tasks": {"completed": 1}})
            self.assertTrue(summary_names_run_status(message), message)
        self.assertTrue(summary_names_run_status("Status: failed | Background process exited with code 1."))
        for text in ("", "Plan: 3 task(s) will run.", "Run finished. Manifest: /tmp/m.json",
                     "Report summary unavailable: bad json"):
            self.assertFalse(summary_names_run_status(text), text)

    def test_run_summary_orders_task_states_and_uses_plain_separators(self) -> None:
        message, status = self._summary_for({
            "status": "failed",
            "elapsed_time": "9m 22s",
            "interpretation": "Some selected tasks failed.",
            "tasks": {
                "zeta": 1, "alpha": 4, "failed": 1, "cancelled": 1, "blocked": 1, "partial": 1,
                "skipped": 2, "reused": 1, "passed": 1, "completed": 2, "unused": 0,
            },
        })

        self.assertEqual(status, "failed")
        lines = message.split("\n")
        self.assertEqual(lines[0], "Failed · 9m 22s")
        self.assertEqual(
            lines[1],
            "2 completed · 1 passed · 1 reused · 2 skipped · 1 partial · 1 blocked · 1 cancelled · 1 failed"
            " · 4 alpha · 1 zeta",
        )
        self.assertEqual(lines[2], "Some selected tasks failed.")
        self.assertEqual(len(lines), 3)
        self.assertNotIn("|", message)
        self.assertNotIn("=", message)

    def test_run_summary_hides_interpretation_for_completed_and_elapsed_when_missing(self) -> None:
        message, status = self._summary_for({
            "status": "completed", "interpretation": "All selected tasks completed.",
            "tasks": {"completed": 2, "skipped": 2},
        })

        self.assertEqual(status, "completed")
        self.assertEqual(message, "Completed\n2 completed · 2 skipped")

    def test_rows_from_plan_normalizes_will_run_and_counts(self) -> None:
        @dataclass
        class Item:
            target: str
            status: str
            reason: str

        class Plan:
            items = [Item("tractography", "will_run", ""), Item("bundles", "blocked", "missing backend")]

            def to_dict(self):
                return {"summary": {"will_run": 1, "skipped": 0, "blocked": 1}}

        rows, summary, state = rows_from_plan(Plan())

        self.assertEqual(rows[0]["status"], "queued")
        self.assertEqual(rows[1]["reason"], "missing backend")
        self.assertEqual(summary, {"will_run": 1, "skipped": 0, "blocked": 1})
        self.assertEqual(state, "partial")

    def test_rows_from_result_uses_normalized_manifest_paths(self) -> None:
        @dataclass
        class Task:
            name: str
            status: str
            reason: str
            manifest_path: Path | None = None
            qc_path: Path | None = None

        class Result:
            tasks = [Task("segmentation", "completed", "", Path("/raw/manifest.json"), Path("/raw/qc.json"))]

        rows = rows_from_result(
            Result(),
            {
                "tasks": [
                    {
                        "name": "segmentation",
                        "normalized_manifest_path": "/normalized/manifest.json",
                        "normalized_qc_path": "/normalized/qc.json",
                        "normalized_logs_dir": "/normalized/logs",
                    }
                ]
            },
        )

        self.assertEqual(rows[0]["manifest_path"], "/normalized/manifest.json")
        self.assertEqual(rows[0]["qc_path"], "/normalized/qc.json")
        self.assertEqual(rows[0]["logs_dir"], "/normalized/logs")

    def test_rows_from_manifests_build_counts(self) -> None:
        rows, counts = rows_from_single_subject_manifest(
            {
                "tasks": [
                    {"name": "segmentation", "status": "completed", "reason": "", "normalized_logs_dir": "/logs"},
                    {"name": "tractography", "status": "blocked", "reason": "missing"},
                    {"name": "bundles", "status": "failed", "reason": "error"},
                ]
            }
        )

        self.assertEqual(len(rows), 3)
        self.assertEqual(counts, {"blocked": 1, "failed": 1, "completed_like": 1})

        batch_rows, batch_counts = rows_from_batch_manifest(
            Path("/tmp/batch_manifest.json"),
            {
                "subjects": ["sub-01", "sub-02", "sub-03"],
                "results": [
                    {"subject_id": "sub-01", "returncode": 0, "stdout_log": "/tmp/sub-01/stdout.log"},
                    {"subject_id": "sub-02", "returncode": 2, "stdout_log": "/tmp/sub-02/stdout.log"},
                ],
                "skipped": [{"subject_id": "sub-03", "reason": "missing input"}],
            },
        )

        self.assertEqual(len(batch_rows), 3)
        self.assertEqual(batch_counts, {"completed": 1, "failed": 1, "skipped": 1, "total": 3})

    def _run_selection(self, root: Path) -> RunSelection:
        return RunSelection(
            subject_id="sub-01",
            subject_root_text=(root / "sub-01").as_posix(),
            output_root_text=(root / "out").as_posix(),
            selected_targets=("validation", "tractography", "bundles"),
            run_name="Test1",
            tractography_backend="csd_act",
            bundle_backend="wmql",
            microstructure_freewater=True,
            microstructure_noddi=False,
            select_streamlines=0,
            ddseg_device="cpu",
            surface_device="cpu",
            connectome_backend="python",
            connectome_compress_mrtrix_dense=True,
            wma_input_tractography_text=(root / "sub-01" / "tractography" / "sub-01_ukf.vtk").as_posix(),
        )

    def test_build_run_kwargs_normalizes_single_subject_and_batch_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selection = self._run_selection(root)
            external_paths = {
                "mrtrix_bin_dir": root / "mrtrix" / "bin",
                "preprocessing_fsl_bin_dir": root / "fsl" / "bin",
                "postprocess_freesurfer_home": root / "freesurfer",
                "tract_querier_root": root / "tract_querier",
                "wma_root": root / "wma",
                "wma_atlas_dir": root / "ORG-Atlases",
                "slicer_executable": root / "Slicer",
                "fiber_tract_measurements_cli": root / "FiberTractMeasurements",
                "tractcloud_root": root / "TractCloud",
                "tractcloud_python_executable": root / "tractcloud-python",
                "tractcloud_data_dir": root / "tractcloud-data",
            }

            single = build_single_subject_run_kwargs(
                selection,
                external_paths,
                preflight_blocked_targets={"bundles": "missing tract_querier"},
            )
            batch = build_batch_run_kwargs(selection, batch_root=root / "batch")

        self.assertEqual(single["subject_id"], "sub-01")
        self.assertEqual(single["targets"], ("validation", "tractography", "bundles"))
        self.assertEqual(single["select_streamlines"], 1)
        self.assertNotIn("preprocessing_backend", single)  # W2a (D1): preprocessing has one route
        self.assertEqual(single["mrtrix_bin_dir"], external_paths["mrtrix_bin_dir"])
        self.assertEqual(single["preprocessing_mrtrix_bin_dir"], external_paths["mrtrix_bin_dir"])
        self.assertEqual(single["postprocess_freesurfer_home"], external_paths["postprocess_freesurfer_home"])
        self.assertEqual(single["wma_root"], external_paths["wma_root"])
        self.assertEqual(single["wma_atlas_dir"], external_paths["wma_atlas_dir"])
        self.assertEqual(
            single["wma_input_tractography"],
            root / "sub-01" / "tractography" / "sub-01_ukf.vtk",
        )
        self.assertEqual(single["slicer_executable"], external_paths["slicer_executable"])
        self.assertEqual(single["tractcloud_root"], external_paths["tractcloud_root"])
        self.assertEqual(single["preflight_blocked_targets"], {"bundles": "missing tract_querier"})
        self.assertEqual(batch["subject_id"], "batch")
        self.assertEqual(batch["subject_root"], root / "batch")
        self.assertEqual(batch["preflight_blocked_targets"], {})

    def test_run_input_validation_helpers_return_actionable_errors(self) -> None:
        self.assertEqual(
            validate_single_subject_inputs("", "sub-01"),
            "Subject ID and subject root are required before running.",
        )
        self.assertEqual(validate_single_subject_inputs("/data/sub-01", "sub-01"), "")

        options, error = validate_batch_run_options(
            "",
            fallback_subject_root_text="",
            batch_size=0,
            arrange_inputs="",
        )
        self.assertIsNone(options)
        self.assertEqual(error, "Batch root is required in Batch mode.")

        with tempfile.TemporaryDirectory() as tmp:
            options, error = validate_batch_run_options(
                "",
                fallback_subject_root_text=tmp,
                batch_size=0,
                arrange_inputs="",
            )

        self.assertEqual(error, "")
        self.assertIsNotNone(options)
        self.assertEqual(options.batch_size, 1)
        self.assertEqual(options.arrange_inputs, "none")

    def test_plan_blocker_helpers_extract_environment_subset(self) -> None:
        @dataclass
        class Item:
            target: str
            status: str
            reason: str
            reason_category: str

        class Plan:
            items = [
                Item("tractography", "blocked", "missing mrtrix", "environment"),
                Item("bundles", "blocked", "missing atlas", "input"),
                Item("connectome", "will_run", "", ""),
            ]

        plan = Plan()

        self.assertEqual([item.target for item in blocked_plan_items(plan)], ["tractography", "bundles"])
        self.assertEqual(environment_blockers_from_plan(plan), {"tractography": "missing mrtrix"})
        self.assertEqual(runnable_targets_from_plan(plan), ("connectome",))
        self.assertEqual(
            blocked_plan_log_text(blocked_plan_items(plan)),
            "tractography: missing mrtrix; bundles: missing atlas",
        )

    def test_partial_run_notice_text_excludes_validation_from_runnable_preview(self) -> None:
        text = partial_run_notice_text(
            {"bundles": "missing tract_querier"},
            ("validation", "tractography"),
        )

        self.assertIn("- bundles: missing tract_querier", text)
        self.assertIn("Runnable targets: tractography", text)

    def _common_run_kwargs(self, root: Path) -> dict[str, object]:
        return {
            "subject_root": root / "sub-01",
            "subject_id": "sub-01",
            "run_name": "Test1",
            "targets": ("validation", "preprocessing", "tractography"),
            "tractography_backend": "csd_act",
            "bundle_backend": "wmql",
            "connectome_backend": "python",
            "select_streamlines": 100000,
            "ddseg_device": "cpu",
            "surface_device": "cpu",
            "output_root": root / "out",
            "microstructure_run_free_water": True,
            "microstructure_run_noddi": False,
            "connectome_compress_mrtrix_dense": True,
            "wma_input_tractography": root / "sub-01" / "tractography" / "sub-01_ukf.vtk",
        }

    def test_build_background_cli_args_includes_paths_and_microstructure_stages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo_root = root / "repo"
            (repo_root / "shared").mkdir(parents=True)
            (repo_root / "shared" / "run_subject_pipeline_cli.py").write_text("", encoding="utf-8")

            args = build_background_cli_args(
                self._common_run_kwargs(root),
                {"mrtrix_bin_dir": root / "mrtrix" / "bin", "wma_root": root / "wma"},
                repo_root=repo_root,
            )

            self.assertIn((repo_root / "shared" / "run_subject_pipeline_cli.py").as_posix(), args)
            self.assertIn("--print-manifest-only", args)
            self.assertIn("--compress-mrtrix-dense", args)
            self.assertNotIn("--preprocessing-backend", args)
            self.assertEqual(args[args.index("--microstructure-stages") + 1], "dti,freewater")
            self.assertEqual(args[args.index("--mrtrix-bin-dir") + 1], (root / "mrtrix" / "bin").as_posix())
            self.assertEqual(args[args.index("--preprocessing-mrtrix-bin-dir") + 1], (root / "mrtrix" / "bin").as_posix())
            self.assertEqual(args[args.index("--wma-root") + 1], (root / "wma").as_posix())
            self.assertEqual(
                args[args.index("--wma-input-tractography") + 1],
                (root / "sub-01" / "tractography" / "sub-01_ukf.vtk").as_posix(),
            )

    def test_build_batch_background_cli_args_uses_batch_script_and_bounds_batch_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo_root = root / "repo"
            args = build_batch_background_cli_args(
                self._common_run_kwargs(root),
                {},
                repo_root=repo_root,
                batch_root=root / "batch",
                batch_size=0,
                arrange_inputs="symlink",
            )

            self.assertIn((repo_root / "scripts" / "run_batch_pipeline.py").as_posix(), args)
            self.assertEqual(args[args.index("--batch-size") + 1], "1")
            self.assertEqual(args[args.index("--arrange-inputs") + 1], "symlink")
            self.assertNotIn("--print-manifest-only", args)
            self.assertNotIn("--wma-input-tractography", args)

    def test_multi_selection_reaches_single_and_batch_workers_with_ukf_options(self):
        root = Path('/tmp/algorithms')
        options = self._common_run_kwargs(root)
        options.update(tractography_backend='csd_act,ukf', bundle_backend='wmql,tractcloud',
                       ukf_parameters={'seedsPerVoxel': 3}, ukf_executable=root / 'UKFTractography',
                       dwiconvert_executable=root / 'DWIConvert')
        commands = [build_background_cli_args(options, {}, repo_root=root),
                    build_batch_background_cli_args(options, {}, repo_root=root, batch_root=root / 'batch',
                                                    batch_size=1, arrange_inputs='copy')]
        for command in commands:
            self.assertEqual(command[command.index('--tractography-backend') + 1], 'csd_act,ukf')
            self.assertEqual(command[command.index('--bundle-backend') + 1], 'wmql,tractcloud')
            self.assertIn('"seedsPerVoxel": 3', command[command.index('--ukf-parameters') + 1])
            self.assertEqual(command[command.index('--ukf-executable') + 1], str(root / 'UKFTractography'))
            self.assertEqual(command[command.index('--dwiconvert-executable') + 1], str(root / 'DWIConvert'))

    def test_selected_atlas_reaches_gui_single_and_batch_children(self):
        root = Path('/tmp/selected-atlas')
        for family in ('org', 'east-west', 'naba'):
            selection = replace(self._run_selection(root), wma_atlas=family, bundle_backend='wma')
            single = build_single_subject_run_kwargs(selection, {}, preflight_blocked_targets={})
            batch = build_batch_run_kwargs(selection, batch_root=root / 'batch')
            self.assertEqual(single['wma_atlas'], family)
            self.assertEqual(batch['wma_atlas'], family)
            commands = [build_background_cli_args(single, {}, repo_root=root),
                        build_batch_background_cli_args(batch, {}, repo_root=root,
                            batch_root=root / 'batch', batch_size=1, arrange_inputs='copy')]
            for command in commands:
                self.assertEqual(command[command.index('--wma-atlas') + 1], family)

    def test_extract_background_manifest_path_uses_explicit_or_latest_existing_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first_orchestration_manifest.json"
            second = root / "second_orchestration_manifest.json"
            first.write_text("{}", encoding="utf-8")
            second.write_text("{}", encoding="utf-8")

            self.assertEqual(extract_background_manifest_path("", explicit_path=first), first)
            self.assertEqual(
                extract_background_manifest_path(f"log\n{first}\nmissing_orchestration_manifest.json\n{second}\n"),
                second,
            )

    def test_extract_background_manifest_path_recognizes_batch_marker_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch_manifest = root / "00_batch_metadata" / "batch_20260101_000000.json"
            batch_manifest.parent.mkdir(parents=True)
            batch_manifest.write_text("{}", encoding="utf-8")

            self.assertEqual(
                extract_background_manifest_path(f"some log line\n[Batch] manifest: {batch_manifest}\nmore log\n"),
                batch_manifest,
            )

    def test_parse_manifest_line_recognizes_single_subject_and_batch_formats(self) -> None:
        self.assertEqual(
            parse_manifest_line("/out/Test1_orchestration_manifest.json"),
            "/out/Test1_orchestration_manifest.json",
        )
        self.assertEqual(
            parse_manifest_line("  /out/Test1_orchestration_manifest.json  "),
            "/out/Test1_orchestration_manifest.json",
        )
        self.assertEqual(
            parse_manifest_line("[Batch] manifest: /out/00_batch_metadata/batch_20260101_000000.json"),
            "/out/00_batch_metadata/batch_20260101_000000.json",
        )

    def test_parse_manifest_line_ignores_unrelated_lines(self) -> None:
        for line in ("", "   ", "[Orchestrator] Running validation", "not a manifest line", "[Batch] manifest:"):
            self.assertIsNone(parse_manifest_line(line))

    def test_classify_background_run_outcome_cancelled_takes_priority(self) -> None:
        outcome = classify_background_run_outcome(exit_code=1, cancelled=True, manifest_available=True)
        self.assertEqual(outcome, "cancelled")

    def test_classify_background_run_outcome_prefers_manifest_over_nonzero_exit(self) -> None:
        # Exit code 2 ("some tasks failed") with a manifest present must load
        # the manifest, not be treated as a blanket process failure.
        outcome = classify_background_run_outcome(exit_code=2, cancelled=False, manifest_available=True)
        self.assertEqual(outcome, "manifest")

    def test_classify_background_run_outcome_success_with_manifest(self) -> None:
        outcome = classify_background_run_outcome(exit_code=0, cancelled=False, manifest_available=True)
        self.assertEqual(outcome, "manifest")

    def test_classify_background_run_outcome_process_failed_without_manifest(self) -> None:
        outcome = classify_background_run_outcome(exit_code=1, cancelled=False, manifest_available=False)
        self.assertEqual(outcome, "process_failed")

    def test_classify_background_run_outcome_no_manifest_on_clean_exit(self) -> None:
        outcome = classify_background_run_outcome(exit_code=0, cancelled=False, manifest_available=False)
        self.assertEqual(outcome, "no_manifest")

    def test_cancelled_manifest_payload_resolves_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path, payload = cancelled_manifest_path_and_payload(
                {
                    "subject_id": "sub-01",
                    "subject_root": root / "sub-01",
                    "output_root": root / "custom-output",
                    "run_name": "Test1",
                    "targets": ("validation", "tractography"),
                    "started_at": "2026-06-23T00:00:00Z",
                },
                cancelled_at="2026-06-23T00:01:00Z",
            )

            self.assertEqual(
                manifest_path,
                (root / "custom-output" / "00_run_metadata" / "Test1_cancelled_orchestration_manifest.json").resolve(),
            )
            self.assertEqual(payload["status"], "cancelled")
            self.assertEqual(payload["targets"], ["validation", "tractography"])
            self.assertEqual(payload["cancelled_at"], "2026-06-23T00:01:00Z")


if __name__ == "__main__":
    unittest.main()
