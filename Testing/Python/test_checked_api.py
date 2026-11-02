"""The recommended Python entry checks dependencies before shared execution."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from shared import ProductOrchestratorConfig, run_subject_workflow
from shared import api
from shared.environment import EnvironmentDoctorResult, TargetEnvironmentStatus


class CheckedApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = ProductOrchestratorConfig('subject', self.root, ('microstructure',), execute=True)

    def report(self, blocked=False, **options):
        return EnvironmentDoctorResult(platform='linux', host_platform='linux', python_executable='python',
            targets=('microstructure',), statuses=(),
            target_statuses=(TargetEnvironmentStatus('microstructure', 'blocked' if blocked else 'runnable',
                                                     ('missing-tool',) if blocked else ()),),
            blocker_count=int(blocked), missing_python_packages=(), missing_external_groups=(), **options)

    def test_public_entry_preflights_once_before_execution(self):
        self.assertIs(run_subject_workflow, api.run_subject_workflow)
        calls = []
        with patch.object(api, 'run_environment_doctor', side_effect=lambda _: calls.append('check') or self.report()) as doctor, \
                patch.object(api, 'write_environment_report', side_effect=lambda *_: calls.append('report') or {'json': self.root / 'report.json'}), \
                patch.object(api, 'run_subject_products', side_effect=lambda *_args, **_kw: calls.append('execute') or 'result') as run:
            self.assertEqual(run_subject_workflow(self.config), 'result')
        self.assertEqual(calls, ['check', 'report', 'execute'])
        doctor.assert_called_once()
        run.assert_called_once()

    def test_strict_blockers_write_report_but_do_not_launch_products(self):
        with patch.object(api, 'run_environment_doctor', return_value=self.report(True)), \
                patch.object(api, 'run_subject_products') as run, self.assertRaises(RuntimeError):
            run_subject_workflow(self.config, strict_environment=True)
        run.assert_not_called()
        self.assertTrue((api.environment_for_products(self.config).output_dir / 'environment_doctor.json').is_file())

    def test_partial_mode_keeps_blockers_and_does_not_modify_caller_config(self):
        config = replace(self.config, preflight_blocked_targets={'wma': 'previous blocker'},
                         preflight_blocked_by_backend={'ukf': {'bundles': 'previous bundle blocker'}})
        report = self.report(True, tracking_backend_blockers={'ukf': {'tractography': 'UKF unavailable'}})
        with patch.object(api, 'run_environment_doctor', return_value=report), \
                patch.object(api, 'run_subject_products') as run:
            run_subject_workflow(config)
        checked = run.call_args.args[0]
        self.assertIn('microstructure', checked.preflight_blocked_targets)
        self.assertEqual(checked.preflight_blocked_targets['wma'], 'previous blocker')
        self.assertEqual(checked.preflight_blocked_by_backend['ukf'],
                         {'bundles': 'previous bundle blocker', 'tractography': 'UKF unavailable'})
        self.assertEqual(config.preflight_blocked_targets, {'wma': 'previous blocker'})

    def test_prechecked_callers_explicitly_skip_and_cli_is_not_wrapped_twice(self):
        with patch.object(api, 'run_environment_doctor') as doctor, patch.object(api, 'run_subject_products') as run:
            run_subject_workflow(self.config, check_environment=False)
        doctor.assert_not_called()
        self.assertIs(run.call_args.args[0], self.config)
        from shared import run_subject_pipeline_cli
        self.assertIsNot(run_subject_pipeline_cli.run_subject_products, run_subject_workflow)

    def test_planning_retains_blockers_without_strict_execution_error(self):
        with patch.object(api, 'run_environment_doctor', return_value=self.report(True)), \
                patch.object(api, 'run_subject_products') as run:
            run_subject_workflow(replace(self.config, execute=False), strict_environment=True)
        run.assert_called_once()
        self.assertIn('microstructure', run.call_args.args[0].preflight_blocked_targets)

    def test_selected_models_devices_and_backends_are_checked(self):
        config = replace(self.config, microstructure_run_free_water=True, microstructure_run_noddi=True,
                         microstructure_run_dki=True, microstructure_dki_fit_method='MKCURVE',
                         targets=('microstructure', 'bundles'), bundle_backend='wma',
                         tractography_backend='csd_act,ukf', wma_atlas='naba', ddseg_device='cpu')
        doctor = api.environment_for_products(config)
        self.assertTrue(doctor.run_free_water and doctor.run_noddi and doctor.run_dki)
        self.assertEqual(doctor.dki_fit_method, 'MKCURVE')
        self.assertEqual(doctor.wma_atlas, 'naba')
        self.assertEqual(doctor.ddseg_device, 'cpu')
        self.assertIn('wma', doctor.targets)
        self.assertIn('tractography', doctor.targets)

    def test_raw_mask_requirement_is_shared_with_cli(self):
        inputs = self.root / 'input'
        inputs.mkdir()
        for name in ('data.nii.gz', 'bvals', 'bvecs'):
            (inputs / name).write_bytes(b'fixture')
        self.assertTrue(api.environment_for_products(self.config).brain_mask_required)
        (inputs / 'nodif_brain_mask.nii.gz').write_bytes(b'fixture')
        self.assertFalse(api.environment_for_products(self.config).brain_mask_required)

    def test_empty_selection_fails_before_dependency_probes(self):
        with patch.object(api, 'run_environment_doctor') as doctor, self.assertRaises(ValueError):
            run_subject_workflow(replace(self.config, targets=()))
        doctor.assert_not_called()


if __name__ == '__main__':
    unittest.main()
