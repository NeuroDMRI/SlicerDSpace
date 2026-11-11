"""Selection and multi-backend orchestration without expensive tracking jobs."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from shared.algorithm_selection import algorithms, BUNDLES, choice_value, selection_choices, TRACKING
from shared.environment import EnvironmentDoctorConfig, build_dependency_specs
from shared.orchestrator import ProductOrchestratorConfig, ProductOrchestratorResult, RunPlanItem, RunPlanResult
from shared.multi_tracking import variants, run, plan
from shared.ui_algorithm_picker import set_algorithms, refresh_picker_language, FAMILIES
from shared.bundle_qc import bundle_record_error


class MultiAlgorithmTests(unittest.TestCase):
    def test_compact_picker_keeps_accessible_translated_action_and_selection(self):
        buttons = {family: SimpleNamespace() for family in FAMILIES}
        for button in buttons.values():
            button.setAccessibleName = lambda text, button=button: setattr(button, 'accessibleName', text)
        node = SimpleNamespace(orchestrationPreprocessingStages='denoise,eddy,topup',
                               orchestrationMicrostructureStages='dti,noddi',
                               orchestrationTractographyBackend='csd_act,ukf',
                               orchestrationBundleBackend='wmql,wma',
                               orchestrationConnectomeMetrics='sift2,fiber_count')
        owner = SimpleNamespace(_algorithmButtons=buttons, _parameterNode=node)
        before = vars(node).copy()
        for language in ('en', 'zh'):
            owner._onboardingLanguage = language
            refresh_picker_language(owner)
            self.assertEqual(len({button.accessibleName for button in buttons.values()}), 5)
            for button in buttons.values():
                self.assertEqual(button.text, '...')
                self.assertIn('选择' if language == 'zh' else 'Select', button.accessibleName)
                self.assertIn('可多选' if language == 'zh' else 'multiple allowed', button.toolTip)
            self.assertIn('Eddy', buttons['preprocessing'].toolTip)
            self.assertIn('NODDI', buttons['microstructure'].toolTip)
            self.assertIn('UKF', buttons['tractography'].toolTip)
            self.assertIn('WMA', buttons['bundles'].toolTip)
            self.assertIn('SIFT2', buttons['connectome'].toolTip)
            self.assertEqual(vars(node), before)

    def test_exact_subsets_and_invalid_values(self):
        self.assertEqual(algorithms("ukf,csd_act,ukf"), ("ukf", "csd_act"))
        self.assertEqual(algorithms("wmql,tractcloud", BUNDLES, "wmql"), ("wmql", "tractcloud"))
        self.assertEqual(algorithms("all", BUNDLES, "wmql"), BUNDLES)
        self.assertEqual(len(selection_choices(TRACKING)), 15)
        self.assertEqual(choice_value('ukf,csd_act'), 'csd_act,ukf')
        self.assertEqual(choice_value('tractcloud,wmql', BUNDLES, 'wmql'), 'wmql,tractcloud')
        for value in ("typo", "ukf,typo", ","):
            with self.assertRaises(ValueError):
                algorithms(value)

    def test_variants_preserve_options_and_only_run_shared_products_once(self):
        config = ProductOrchestratorConfig("demo", Path('/input'),
            ('validation', 'preprocessing', 'parcellation', 'microstructure', 'tractography', 'bundles', 'connectome'),
            run_name="comparison", tractography_backend="csd_act,ukf", bundle_backend="wmql,tractcloud",
            ukf_parameters={"seedsPerVoxel": 3}, microstructure_run_noddi=True)
        first, second = tuple(variants(config))
        self.assertEqual((first.run_name, second.run_name), ("comparison__csd_act", "comparison__ukf"))
        self.assertNotIn('preprocessing', second.targets)
        self.assertNotIn('microstructure', second.targets)
        self.assertEqual(second.targets, ('tractography', 'bundles', 'connectome'))
        self.assertFalse(second.microstructure_run_noddi)
        self.assertEqual(second.ukf_parameters, {"seedsPerVoxel": 3})
        self.assertEqual(second.bundle_backend, "wmql,tractcloud")

    def test_doctor_covers_every_selected_tracking_method(self):
        from scripts.install_dependencies import build_parser
        args = build_parser().parse_args(['--tractography-backend', 'ukf,csd_act'])
        self.assertEqual(args.tractography_backend, 'csd_act,ukf')
        config = EnvironmentDoctorConfig(targets=('tractography',), repo_root=Path('/repo'),
                                         tractography_backend='csd_act,dti_det')
        keys = {item.key for item in build_dependency_specs(config)}
        self.assertTrue(any('tckgen' in key for key in keys))
        self.assertTrue(any('dipy' in key for key in keys))
        self.assertEqual(len(keys), len(build_dependency_specs(config)))

    def test_plan_preserves_dependency_blocks_and_separate_tracking_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = ProductOrchestratorConfig('demo', root, ('validation', 'tractography'),
                tractography_backend='csd_act,ukf', run_name='comparison')
            def fake_plan(child, write=True):
                return RunPlanResult('demo', root, root / 'outputs', child.run_name, child.targets,
                    root / (child.run_name + '_run_plan.json'),
                    (RunPlanItem('tractography', 'tractography.' + child.tractography_backend, 'blocked',
                        reason='Missing selected backend', reason_category='environment',
                        outputs={'tractogram': root / child.run_name / 'tractogram.vtk'}),))
            with patch('shared.orchestrator.plan_subject_products', side_effect=fake_plan):
                result = plan(config, write=False)
            self.assertEqual([item.status for item in result.items], ['blocked', 'blocked'])
            self.assertEqual(len({item.task_name for item in result.items}), 2)
            self.assertEqual(len({item.outputs['tractogram'] for item in result.items}), 2)

    def test_cancelled_or_empty_picker_never_mutates_parameters(self):
        owner = SimpleNamespace(_parameterNode=SimpleNamespace())
        with self.assertRaises(ValueError):
            set_algorithms(owner, 'bundles', [])
        self.assertEqual(vars(owner._parameterNode), {})

    def test_variant_suffix_cannot_bypass_bundle_qc(self):
        self.assertIsNotNone(bundle_record_error('bundles.wmql@ukf', {'status': 'completed', 'automatic_qc_passed': False}))

    def test_sequential_variants_save_separate_manifests_and_keep_failures(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = ProductOrchestratorConfig('demo', root, ('validation', 'tractography', 'bundles'),
                output_root=root / 'output', run_name='compare', tractography_backend='csd_act,ukf', execute=True)
            calls = []
            def fake_plan(child, write=True):
                return RunPlanResult('demo', root, root / 'output', child.run_name, child.targets,
                    root / 'output/00_run_metadata' / (child.run_name + '_run_plan.json'),
                    (RunPlanItem('tractography', 'tractography.' + child.tractography_backend, 'will_run', will_run=True),))
            def fake_run(child, log=None):
                calls.append(child.tractography_backend)
                path = root / (child.run_name + '.json')
                path.write_text(json.dumps({'tasks': [dict(name='tractography.' + child.tractography_backend,
                    status='partial_failed' if child.tractography_backend == 'ukf' else 'completed', outputs={})]}))
                return ProductOrchestratorResult('demo', root, path, ())
            with patch('shared.orchestrator.plan_subject_products', side_effect=fake_plan), \
                    patch('shared.orchestrator.run_subject_products', side_effect=fake_run):
                result = run(config)
            self.assertEqual(calls, ['csd_act', 'ukf'])
            manifest = json.loads(result.run_manifest_path.read_text())
            self.assertEqual(manifest['status'], 'failed')
            self.assertEqual([task['status'] for task in manifest['tasks']], ['completed', 'partial_failed'])
            self.assertEqual(len({task['name'] for task in manifest['tasks']}), 2)
            self.assertEqual(len(manifest['child_runs']), 2)


if __name__ == '__main__':
    unittest.main()
