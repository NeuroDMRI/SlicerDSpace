#!/usr/bin/env python3
"""Tests for the SlicerDSpace batch CLI helpers."""

from __future__ import annotations

import importlib.util
import contextlib
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BATCH_CLI_PATH = REPO_ROOT / "scripts" / "run_batch_pipeline.py"

spec = importlib.util.spec_from_file_location("run_batch_pipeline", BATCH_CLI_PATH)
assert spec is not None and spec.loader is not None
batch_cli = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = batch_cli
spec.loader.exec_module(batch_cli)


def _write_raw_subject(subject_root: Path, *, nested_input: bool = True) -> None:
    input_dir = subject_root / "input" if nested_input else subject_root
    input_dir.mkdir(parents=True, exist_ok=True)
    for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
        (input_dir / name).write_text("x", encoding="utf-8")


class RunBatchPipelineCliTest(unittest.TestCase):
    FAKE_PIPELINE = '''import argparse, json, sys, time
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--subject-root")
p.add_argument("--subject-id")
p.add_argument("--output-root")
p.add_argument("--event-dir", type=Path)
p.add_argument("--fail-first", action="store_true")
a, extra = p.parse_known_args()
started = time.time()
a.event_dir.mkdir(exist_ok=True)
(a.event_dir / (a.subject_id + ".started")).write_text(str(started))
if a.subject_id in ("sub-01", "sub-02"):
    deadline = time.monotonic() + 5
    while len(list(a.event_dir.glob("*.started"))) < 2:
        if time.monotonic() > deadline:
            sys.exit(9)
        time.sleep(.01)
failed = a.fail_first and a.subject_id == "sub-01"
time.sleep(.02 if failed else .2)
dest = Path(a.output_root)
dest.mkdir(parents=True)
(dest / "receipt.json").write_text(json.dumps(dict(id=a.subject_id, start=started, end=time.time(), extra=extra)))
print(a.subject_id)
sys.exit(3 if failed else 0)
'''

    def run_fake_batch(self, root, *extra, jobs=2):
        for subject in ("sub-01", "sub-02", "sub-03"):
            _write_raw_subject(root / "subjects" / subject)
        fake = root / "pipeline.py"
        fake.write_text(self.FAKE_PIPELINE, encoding="utf-8")
        manifest = root / "batch.json"
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            status = batch_cli.main([
                "--batch-root", str(root / "subjects"), "--output-root", str(root / "outputs"),
                "--pipeline-script", str(fake), "--python-executable", sys.executable,
                "--manifest", str(manifest), "--execute", *(['--jobs', str(jobs)] if jobs is not None else []),
                "--event-dir", str(root / "events"), *extra,
            ])
        return status, json.loads(manifest.read_text())

    def test_processes_overlap_and_forward_multiple_algorithm_choices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            choices = ["--microstructure-stages", "dti,freewater,noddi", "--tractography-backend", "csd_act,ukf",
                       "--bundle-backend", "wmql,tractcloud"]
            status, manifest = self.run_fake_batch(root, *choices)
            self.assertEqual(status, 0)
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(len(manifest["results"]), 3)
            self.assertEqual(manifest["not_launched"], [])
            rows = [json.loads((root / "outputs" / name / "receipt.json").read_text())
                    for name in ("sub-01", "sub-02")]
            self.assertLess(max(r["start"] for r in rows), min(r["end"] for r in rows))
            for row in rows:
                self.assertEqual(row["extra"][-len(choices):], choices)
            logs = [Path(item["stdout_log"]) for item in manifest["results"]]
            self.assertEqual(len(set(logs)), 3)
            self.assertTrue(all(path.read_text().strip() in ("sub-01", "sub-02", "sub-03") for path in logs))

    def test_fail_fast_still_records_every_already_started_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, manifest = self.run_fake_batch(Path(tmp), "--fail-first")
            self.assertEqual(status, 2)
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual({r["subject_id"]: r["returncode"] for r in manifest["results"]},
                             {"sub-01": 3, "sub-02": 0})
            self.assertEqual(manifest["not_launched"], ["sub-03"])

    def test_continue_on_error_keeps_remaining_subjects(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, manifest = self.run_fake_batch(Path(tmp), "--fail-first", "--continue-on-error")
            self.assertEqual(status, 2)
            self.assertEqual(len(manifest["results"]), 3)
            self.assertEqual(manifest["not_launched"], [])

    def test_missing_python_is_logged_as_a_subject_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            subject = batch_cli.BatchSubject("sub-01", Path(tmp), "test", "subject_root")
            result = batch_cli._run_subject([str(Path(tmp) / "missing-python")], subject, Path(tmp) / "logs")
            self.assertEqual(result.returncode, 127)
            self.assertIn("Could not start", result.stderr_log.read_text())

    def test_default_pipeline_entry_is_shipped_in_extension(self) -> None:
        args = batch_cli.build_parser().parse_args(["--batch-root", "/tmp/batch"])
        self.assertEqual(args.pipeline_script, REPO_ROOT / "shared/run_subject_pipeline_cli.py")
        self.assertTrue(args.pipeline_script.is_file())

    def test_discover_subject_roots_under_batch_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            batch_root = Path(tmp)
            _write_raw_subject(batch_root / "100307")
            _write_raw_subject(batch_root / "100408")
            (batch_root / "notes").mkdir()

            subjects, skipped = batch_cli.discover_subjects(batch_root)

            self.assertEqual([subject.subject_id for subject in subjects], ["100307", "100408"])
            self.assertEqual([subject.mode for subject in subjects], ["subject_root", "subject_root"])
            self.assertEqual([item["subject_id"] for item in skipped], ["notes"])

    def test_arrange_direct_raw_files_into_input_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "sub-01"
            _write_raw_subject(subject_root, nested_input=False)

            subjects, skipped = batch_cli.discover_subjects(Path(tmp), arrange_inputs="copy")

            self.assertEqual(skipped, [])
            self.assertEqual(len(subjects), 1)
            self.assertEqual(subjects[0].mode, "subject_root+arranged")
            self.assertTrue((subject_root / "input" / "data.nii.gz").is_file())

    def test_build_subject_command_defaults_to_test1_and_subject_output_subdir(self) -> None:
        subject = batch_cli.BatchSubject(
            subject_id="100307",
            subject_root=Path("/tmp/batch/100307"),
            source="glob",
            mode="subject_root",
        )

        command = batch_cli.build_subject_command(
            subject,
            pipeline_script=Path("/repo/run_subject_pipeline.py"),
            python_executable=Path("/usr/bin/python3"),
            output_root=Path("/tmp/outputs"),
            run_name="Test1",
            execute=True,
            forwarded_args=["--all", "--bundle-backend", "all"],
        )

        self.assertIn("--run-name", command)
        self.assertEqual(command[command.index("--run-name") + 1], "Test1")
        # build_subject_command() resolves the output root (e.g. macOS's
        # /tmp -> /private/tmp symlink); compare against the resolved form.
        expected_output_root = (Path("/tmp/outputs").resolve() / "100307").as_posix()
        self.assertEqual(command[command.index("--output-root") + 1], expected_output_root)
        self.assertIn("--execute", command)
        self.assertEqual(command[-3:], ["--all", "--bundle-backend", "all"])

    def test_parser_accepts_batch_size_alias(self) -> None:
        args, forwarded = batch_cli.build_parser().parse_known_args(
            ["--batch-root", "/tmp/batch", "--batch-size", "3", "--targets", "microstructure"]
        )

        self.assertEqual(args.jobs, 3)
        self.assertEqual(forwarded, ["--targets", "microstructure"])

    def test_resource_mode_passes_shared_schedule_without_changing_subject_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = ("import sys\nsys.path.insert(0, " + repr(str(REPO_ROOT)) + ")\n" + self.FAKE_PIPELINE)
            script = script.replace('failed = a.fail_first',
                'from shared.cohort_schedule import stage_admission\n'
                "with stage_admission('fixture', gpu=True): time.sleep(.05)\nfailed = a.fail_first")
            with patch.object(self, 'FAKE_PIPELINE', script):
                status, manifest = self.run_fake_batch(root, '--scheduler', 'resources', '--batch-cpus', '2',
                    '--batch-memory-gb', '2', '--batch-stage-cpus', '1', '--batch-stage-memory-gb', '1')
            self.assertEqual(status, 0)
            self.assertEqual(manifest['scheduler'], 'resources')
            schedule = Path(manifest['resource_schedule'])
            events = [json.loads(line) for line in (schedule / 'events.jsonl').read_text().splitlines()]
            self.assertEqual(len([row for row in events if row['event'] == 'completed']), 3)
            self.assertEqual(json.loads((schedule / 'state.json').read_text()), dict(active=[], waiting=[]))
            self.assertTrue(all(item['status'] == 'process_completed' for item in manifest['results']))

    def test_resource_mode_rejects_nested_model_parallelism(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                batch_cli.main(['--batch-root', tmp, '--scheduler', 'resources', '--max-parallel-models', '2'])

    def test_adaptive_mode_supports_budget_and_subject_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, manifest = self.run_fake_batch(Path(tmp), '--scheduler', 'adaptive',
                '--resource-budget', '80%', '--subject-limit', '2', '--batch-cpus', '2',
                '--batch-memory-gb', '4', '--batch-stage-memory-gb', '1', jobs=None)
            self.assertEqual(status, 0)
            self.assertEqual(len(manifest['results']), 2)
            self.assertEqual(len(manifest['planned_commands']), 2)
            self.assertEqual(manifest['subject_limit'], 2)
            self.assertTrue(any(row['reason'] == 'subject limit' for row in manifest['skipped']))
            self.assertEqual(manifest['resource_plan']['policy'], 'fair_backfill')
            self.assertIn('stage_outcomes', manifest['resource_measurements'])

    def test_resource_mode_can_omit_jobs_and_all_resource_parameters(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch('shared.cohort_planning.available_cohort_resources', return_value=(16, 64 * 1024**3)), \
             patch('shared.cohort_affinity.os.sched_getaffinity', return_value=set(range(16)), create=True):
            status, manifest = self.run_fake_batch(Path(tmp), '--scheduler', 'resources', jobs=None)
            self.assertEqual(status, 0)
            self.assertEqual(manifest['jobs'], 2)
            self.assertTrue(manifest['resource_plan']['jobs_automatic'])
            self.assertTrue(manifest['resource_plan']['input_headers']['sub-01']['estimated'])
            self.assertIn('subject_profiles', json.loads((Path(manifest['resource_schedule']) / 'config.json').read_text()))

    def test_actual_cli_blocked_subject_is_not_a_completed_batch(self):
        from test_input_execution_gate import subject
        for allow_partial in (False, True):
            with self.subTest(allow_partial=allow_partial), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                subject(root / 'subjects')
                manifest = root / 'batch.json'
                args = ['--batch-root', str(root / 'subjects'), '--output-root', str(root / 'outputs'),
                        '--manifest', str(manifest), '--execute', '--targets', 'wma',
                        '--wma-root', str(root / 'no-wma'), '--brain-mask-backend', 'provided']
                if allow_partial:
                    args.append('--allow-partial-success')
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    code = batch_cli.main(args)
                record = json.loads(manifest.read_text())
                self.assertEqual(code, 0 if allow_partial else 2)
                self.assertEqual(record['status'], 'completed_with_blocks')
                self.assertEqual(record['subject_status_counts'], {'completed_with_blocks': 1})
                self.assertGreater(record['task_status_counts']['blocked'], 0)
                self.assertEqual(record['results'][0]['status_source'], 'child_manifest')

    def test_stale_manifest_is_not_used_for_a_new_custom_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stale = root / 'outputs/00_run_metadata/Test1_orchestration_manifest.json'
            stale.parent.mkdir(parents=True)
            stale.write_text(json.dumps(dict(subject_id='sub-01', run_name='Test1', status='completed_with_blocks', tasks=[dict(status='blocked')])))
            runner = root / 'custom.py'
            runner.write_text('pass\n')
            result = batch_cli._run_subject([sys.executable, str(runner), '--output-root', str(root / 'outputs')],
                batch_cli.BatchSubject('sub-01', root, 'test', 'subject_root'), root / 'logs')
            self.assertEqual(result.run_status, 'process_completed')
            self.assertEqual(result.status_source, 'process_exit_code')
            self.assertIsNone(result.task_counts)
            self.assertIn('fresh', result.manifest_error)


if __name__ == "__main__":
    unittest.main()
