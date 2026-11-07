"""Experiment planning never executes a shell or mixes variant outputs."""

import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from shared.experiments import compile_plan, make_request, parameter_args, variants_from_json
from scripts.run_experiments import run_plan


class ExperimentsTest(unittest.TestCase):
    def request(self, root, **kw):
        subject = root / 'subject'
        subject.mkdir(exist_ok=True)
        return make_request(base_args=['--targets', 'validation', '--brain-mask-backend', 'provided'],
                            subject_root=subject, subject_id='subject', output_root=root / 'out', run_name='test', **kw)

    def plan(self, request, script=None):
        return compile_plan(request, python_executable=sys.executable, pipeline_script=script or Path('pipeline.py'))

    def test_cli_schema_and_reserved_lifecycle_options(self):
        self.assertEqual(parameter_args('--mrtrix-seed 42 --select-streamlines=123'), ['--mrtrix-seed','42','--select-streamlines=123'])
        for value in ('--execute', '--skip-environment-check', '--run-name x', '--output-root /tmp',
                      '--mrtrix-nth 2', '--user-config /tmp/x', '--unknown 4', '-h'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parameter_args(value)
        with self.assertRaises(ValueError):
            parameter_args('--brain-mask-backend invalid')

    def test_variants_reject_collisions_and_path_traversal(self):
        for variants in ([], [{'name':'../x'}], [{'name':'CON'}], [{'name':'x.'}], [{'name':'X'},{'name':'x'}], [{'name':'x','command':'oops'}]):
            with self.subTest(variants=variants), self.assertRaises(ValueError):
                variants_from_json(variants)

    def test_override_precedence_identity_and_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = self.request(root, options='--mrtrix-seed 42', variants=[
                {'name':'one','args':['--mrtrix-seed','2026']}, {'name':'two','args':[]}])
            plan = self.plan(request)
            self.assertEqual(len(plan['subjects']), 2)
            a, b = plan['subjects']
            self.assertNotEqual(a['output_root'], b['output_root'])
            from shared.experiments import parse_cli
            self.assertEqual(parse_cli(a['command'][2:]).mrtrix_seed, 2026)
            self.assertEqual(parse_cli(b['command'][2:]).mrtrix_seed, 42)
            self.assertIn('--no-user-config', a['command'])
            self.assertIn('--strict-environment', a['command'])
            self.assertFalse(Path(plan['output_root']).exists())
            Path(plan['output_root']).mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                self.plan(request)

    def test_bad_later_variant_prevents_any_job_from_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = self.request(Path(tmp))
            request['variants'].append(dict(name='bad', args=['--select-streamlines','0']))
            with self.assertRaises(ValueError):
                self.plan(request)
            self.assertFalse(Path(request['output_root']).exists())

    def test_invalid_ukf_and_bundle_source_fail_at_preview(self):
        for options in ('--ukf-parameters \'{"unknown": 1}\'', '--targets unknown',
                        '--targets microstructure --bundle-tractography-backend ukf'):
            with tempfile.TemporaryDirectory() as tmp, self.subTest(options=options):
                request = self.request(Path(tmp), options=options)
                with self.assertRaises(ValueError):
                    self.plan(request)

    def test_empty_cohort_root_cannot_scan_current_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = self.request(Path(tmp), scope='cohort', cohort_root='')
            with self.assertRaisesRegex(ValueError, 'Cohort root is required'):
                self.plan(request)

    def test_scheduler_runs_serial_subject_variants_and_retains_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            request = self.request(root, variants=[{'name':'one','args':[]},{'name':'two','args':[]}])
            plan = self.plan(request)
            # An actual small Python child writes the required run record. This
            # is a scheduler fixture, not simulated algorithm/QC evidence.
            program = "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); p.parent.mkdir(parents=True); p.write_text(json.dumps(dict(execute=True,subject_id='subject',status='completed',tasks=[dict(name='validation',status='passed')])))"
            for row in plan['subjects']:
                row['command'] = [sys.executable, '-c', program, row['run_manifest_path']]
            plan['subjects'][0]['command'] = [sys.executable, '-c', 'raise SystemExit(2)']
            code = run_plan(plan)
            self.assertEqual(code, 2)
            recorded = json.loads(Path(plan['manifest_path']).read_text())
            self.assertEqual([r['variant'] for r in recorded['results']], ['one','two'])
            self.assertEqual([r['returncode'] for r in recorded['results']], [2,0])
            self.assertEqual([r['status'] for r in recorded['results']], ['failed','completed'])
            self.assertEqual(recorded['status'], 'failed')
            self.assertEqual(recorded['not_launched'], [])

    def test_cancellation_does_not_launch_remaining_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.plan(self.request(Path(tmp), variants=[{'name':'one'}, {'name':'two'}]))
            plan['subjects'][0]['command'] = [sys.executable, '-c', 'import time; time.sleep(30)']
            event = threading.Event()
            timer = threading.Timer(0.3, event.set)
            timer.start()
            before = time.monotonic()
            try:
                self.assertEqual(run_plan(plan, cancel_event=event), 130)
            finally:
                timer.join()
            self.assertLess(time.monotonic() - before, 5)
            self.assertEqual(plan['status'], 'cancelled')
            self.assertEqual(plan['not_launched'], ['0001_two'])

    def test_missing_child_record_is_not_a_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.plan(self.request(Path(tmp)))
            plan['subjects'][0]['command'] = [sys.executable, '-c', 'pass']
            self.assertEqual(run_plan(plan), 2)
            self.assertIn('No such file', plan['results'][0]['reason'])

    def test_native_preset_does_not_turn_off_anatomy_or_force_cpu(self):
        from shared.ui_experiments import native_route_values
        values = native_route_values()
        self.assertEqual(values['brainMaskBackend'], 'pnl_cnn')
        self.assertEqual(values['orchestrationTractographyBackend'], 'dti_det')
        self.assertFalse(any('Device' in key or key.startswith('target') for key in values))


if __name__ == '__main__':
    unittest.main()
