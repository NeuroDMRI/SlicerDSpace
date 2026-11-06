"""Full-workflow selection and safe GPU admission for real cohort benchmarks."""

from pathlib import Path
import json
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.benchmark_heterogeneous_cohort import (build_record, check_plan, idle_devices,
                                                   preflight, validate_campaign_inputs, verify)
from scripts.benchmark_cohort_compare import scientific_inventory
from shared.cohort_targets import selected_stages


class HeterogeneousCampaignTests(unittest.TestCase):
    def base(self):
        return dict(subjects=[dict(subject=str(i)) for i in range(12)], common=['--max-parallel-models', '2'],
                    routes={'tracts': ['--wma-root', '/wma', '--wma-atlas-dir', '/atlas',
                                      '--slicer-executable', '/slicer', '--ukf-executable', '/ukf']})

    def test_complete_route_includes_ddsurfer_and_scientific_end_products(self):
        record = build_record(self.base(), Path('/snapshot'), {}, True, list(range(56, 72)), 128)
        stages = selected_stages(record['routes']['tracts'])
        self.assertTrue({'segmentation', 'parcellation', 'surface', 'csd_act', 'connectome',
                         'microstructure_freewater', 'microstructure_noddi'} <= stages)
        self.assertNotIn('wma', stages)
        self.assertEqual(record['common'], ['--max-parallel-models', '1'])
        self.assertIsNone(record['jobs'])
        self.assertEqual([t['policy'] for t in record['trials']], ['subject', 'adaptive'])
        arguments = record['routes']['tracts']
        self.assertNotIn('--bundle-tractography-backend', arguments)
        self.assertEqual(arguments[arguments.index('--connectome-backend')+1], 'mrtrix')

    def test_cpu_route_prioritises_ukf_wma_without_gpu_anatomy(self):
        record = build_record(self.base(), Path('/snapshot'), {}, False, list(range(32, 56)), 160)
        stages = selected_stages(record['routes']['tracts'])
        self.assertTrue({'ukf', 'wma'} <= stages)
        self.assertFalse({'segmentation', 'parcellation', 'surface'} & stages)
        self.assertFalse(record['requires_idle_gpu'])
        arguments = record['routes']['tracts']
        self.assertEqual(arguments[arguments.index('--bundle-tractography-backend')+1], 'ukf')

    def test_spare_vram_on_busy_gpu_does_not_admit_the_campaign(self):
        self.assertEqual(idle_devices([['0', '100', '35000', '81559']]), [])
        self.assertEqual(idle_devices([['0', '0', '35000', '81559']]), [])
        self.assertEqual(idle_devices([['0', '0', '50', '81559']]), [0])

    def test_preflight_uses_installed_entry_and_checks_doctor_blockers(self):
        for blocked in ([], ['surface']):
            with self.subTest(blocked=blocked), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root/'preflight').mkdir()
                (root/'preflight/environment_doctor.json').write_text(json.dumps(
                    {'summary': {'blocked_targets': blocked, 'runnable_targets': ['wma']}}))
                plan_path = root/'preflight-output/00_run_metadata/benchmark_run_plan.json'
                plan_path.parent.mkdir(parents=True)
                plan_path.write_text(json.dumps(dict(summary=dict(blocked=0),
                    items=[dict(target='wma', status='will_run')])))
                record = dict(launcher='/slicer', snapshot='/snapshot', common=[], routes={'tracts': []},
                              requires_idle_gpu=True,
                              subjects=[dict(subject='100307', source='/input')])
                with patch('scripts.benchmark_heterogeneous_cohort.subprocess.run') as run:
                    if blocked:
                        with self.assertRaisesRegex(RuntimeError, 'surface'):
                            preflight(record, root, {})
                    else:
                        result = preflight(record, root, {})
                        self.assertEqual(result['blocked_targets'], [])
                        self.assertEqual(result['task_plan'], {'blocked': 0})
                self.assertEqual(run.call_args.args[0][3], '/snapshot/shared/run_subject_pipeline_cli.py')
                self.assertEqual(run.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], '0')
                self.assertEqual(run.call_count, 1 if blocked else 2)

    def test_plan_rejects_blocked_or_missing_wma_despite_available_dependencies(self):
        ready = [dict(target='tractography', status='will_run'), dict(target='wma', status='will_run')]
        check_plan(dict(items=ready), ['tractography', 'wma'])
        for items in ([], ready[:1], [ready[0], dict(target='wma', status='blocked', reason='missing source')]):
            with self.subTest(items=items), self.assertRaisesRegex(RuntimeError, 'Unrunnable'):
                check_plan(dict(items=items), ['tractography', 'wma'])

    def test_preflight_retries_only_transient_dependency_timeout_and_keeps_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = dict(launcher='/slicer', snapshot='/snapshot', common=[], routes={'tracts': []},
                          subjects=[dict(subject='A', source='/input')])
            plan_path = root/'preflight-output/00_run_metadata/benchmark_run_plan.json'
            plan_path.parent.mkdir(parents=True)
            plan_path.write_text(json.dumps(dict(summary={'blocked': 0},
                items=[dict(target='wma', status='will_run')])))

            def doctor(command, **kwargs):
                if '--doctor' not in command:
                    return
                folder = Path(command[command.index('--doctor-output-dir')+1])
                folder.mkdir()
                blocked = folder.name == 'preflight'
                result = dict(summary=dict(blocked_targets=['wma'] if blocked else [], runnable_targets=['wma']),
                    dependencies=[dict(required=True, status='missing', targets=['wma'],
                                       details={'error': 'Dependency probe timed out after 30s'})] if blocked else [])
                (folder/'environment_doctor.json').write_text(json.dumps(result))

            with patch('scripts.benchmark_heterogeneous_cohort.subprocess.run', side_effect=doctor) as run, \
                    patch('scripts.benchmark_heterogeneous_cohort.time.sleep') as sleep:
                result = preflight(record, root, {})
            self.assertFalse(result['blocked_targets'])
            self.assertEqual(run.call_count, 3)
            sleep.assert_called_once_with(10)
            self.assertTrue((root/'preflight/environment_doctor.json').exists())
            self.assertTrue((root/'preflight-retry-2/environment_doctor.json').exists())

    def test_dependency_timeout_retries_are_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = dict(launcher='/slicer', snapshot='/snapshot', common=[], routes={'tracts': []},
                          subjects=[dict(subject='A', source='/input')])

            def doctor(command, **kwargs):
                folder = Path(command[command.index('--doctor-output-dir')+1])
                folder.mkdir()
                (folder/'environment_doctor.json').write_text(json.dumps(dict(
                    summary=dict(blocked_targets=['surface']), dependencies=[dict(required=True,
                    status='missing', targets=['surface'], details={'error': 'Dependency probe timed out after 30s'})])))

            with patch('scripts.benchmark_heterogeneous_cohort.subprocess.run', side_effect=doctor) as run, \
                    patch('scripts.benchmark_heterogeneous_cohort.time.sleep') as sleep:
                with self.assertRaisesRegex(RuntimeError, 'surface'):
                    preflight(record, root, {})
            self.assertEqual(run.call_count, 3)
            self.assertEqual(sleep.call_count, 2)

    def test_each_subject_gets_input_preflight_and_failed_report_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = dict(subjects=[dict(subject='A', source='/source/A'), dict(subject='B', source='/source/B')])
            (root/'campaign.json').write_text(json.dumps(record))
            with patch('tractography.ukf.validate_inputs', side_effect=[{'effective_brain_voxels': 100}, ValueError('bad mask')]) as validate:
                with self.assertRaisesRegex(ValueError, 'bad mask'):
                    validate_campaign_inputs(root)
            report = json.loads((root/'ukf_input_checks.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertEqual([row['status'] for row in report['subjects']], ['passed', 'failed'])
            self.assertEqual([call.args[0].subject_id for call in validate.call_args_list], ['A', 'B'])

    def test_inventory_covers_native_surface_and_connectome_matrix(self):
        import vtk
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            surface = root/'03_anatomy/surface/lh.stl'
            surface.parent.mkdir(parents=True)
            sphere = vtk.vtkSphereSource()
            sphere.Update()
            writer = vtk.vtkSTLWriter()
            writer.SetInputData(sphere.GetOutput())
            writer.SetFileName(str(surface))
            writer.Write()
            matrix = root/'04_tractography/connectome/matrix.csv'
            matrix.parent.mkdir(parents=True)
            matrix.write_text('0,1\n1,0\n')
            nodes = matrix.with_name('nodes.tsv')
            nodes.write_text('node_id\tlabel\tname\n1\t1024\tctx-lh-precentral\n')
            before = scientific_inventory(root)
            self.assertIn(str(surface.relative_to(root)), before)
            self.assertIn(str(matrix.relative_to(root)), before)
            self.assertIn(str(nodes.relative_to(root)), before)
            matrix.write_text('0,2\n2,0\n')
            self.assertNotEqual(before, scientific_inventory(root))
            matrix.write_text('0,1\n1,0\n')
            self.assertEqual(before, scientific_inventory(root))
            nodes.write_text('node_id\tlabel\tname\n1\t2024\tctx-rh-precentral\n')
            self.assertNotEqual(before, scientific_inventory(root))

    def test_corrected_audit_keeps_original_timing_and_failed_audit_records(self):
        for changed in (False, True):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                trials = []
                for policy, seconds in (('subject', 120), ('adaptive', 100)):
                    folder = root/policy
                    folder.mkdir()
                    (folder/'scientific_fingerprints.json').write_text(json.dumps({'A': {'array': 'same'}}))
                    trials.append(dict(id=policy, route='tracts', policy=policy, threads=4, status='completed',
                        output=str(folder), qc_status='failed', measurement={'execution_seconds': seconds}))
                record = dict(status='completed_pending_acceptance', subjects=[dict(subject='A', source='/source', sha256={})],
                              trials=trials, reference_policy='subject', cpus=8, memory_gb=32, jobs=None)
                path = root/'campaign.json'
                path.write_text(json.dumps(record))
                before = path.read_bytes()
                with patch('scripts.audit_scientific_outputs.audit_output_root',
                           return_value=dict(status='passed', summary={'passed_subjects': 1})), \
                        patch('scripts.benchmark_cohort_compare.scientific_inventory',
                              return_value={'array': 'changed' if changed else 'same'}):
                    args = SimpleNamespace(output=root, max_hours=1)
                    if changed:
                        with self.assertRaisesRegex(RuntimeError, 'products changed'):
                            verify(args)
                    else:
                        self.assertEqual(verify(args), 0)
                        result = json.loads((root/'post_validation/primary_comparison.json').read_text())
                        self.assertEqual(result['comparisons'][0]['status'], 'passed')
                        self.assertAlmostEqual(result['comparisons'][0]['speed_ratio'], 1.2)
                self.assertEqual(path.read_bytes(), before)

    def test_post_validation_never_accepts_failed_scientific_execution(self):
        for trials in ([], [{'status': 'failed'}]):
            with self.subTest(trials=trials), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root/'campaign.json').write_text(json.dumps(dict(status='failed', trials=trials)))
                with self.assertRaisesRegex(RuntimeError, 'Scientific execution incomplete'):
                    verify(SimpleNamespace(output=root, max_hours=1))
                result = json.loads((root/'post_validation/verification.json').read_text())
                self.assertEqual(result['status'], 'failed')

    def test_documented_reader_error_can_rebuild_separate_fingerprints_without_editing_timing(self):
        import numpy as np
        for changed_after_run in (False, True):
            with self.subTest(changed_after_run=changed_after_run), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                trials = []
                for policy, seconds in (('subject', 120), ('adaptive', 100)):
                    folder = root/policy
                    product = folder/'outputs/A/02_microstructure/fa.npy'
                    product.parent.mkdir(parents=True)
                    np.save(product, np.array([0.2, 0.5], dtype=np.float32))
                    (folder/'independent_qc.json').write_text(json.dumps({'status': 'passed'}))
                    trials.append(dict(id=policy, route='tracts', policy=policy, threads=4, status='completed',
                        output=str(folder), qc_status='audit_error', ended_at=time.time()-(10 if changed_after_run else 0),
                        audit_error='ValueError: Empty or unreadable polydata: /FiberClustering/cluster_00552.vtp',
                        measurement={'execution_seconds': seconds}))
                record = dict(status='completed_pending_acceptance', subjects=[dict(subject='A', source='/source', sha256={})],
                              trials=trials, reference_policy='subject', cpus=8, memory_gb=32, jobs=None)
                path = root/'campaign.json'
                path.write_text(json.dumps(record))
                before = path.read_bytes()
                args = SimpleNamespace(output=root, max_hours=1, verification_name='post_validation_v2',
                                       rebuild_missing_fingerprints=True)
                with patch('scripts.audit_scientific_outputs.audit_output_root',
                           return_value=dict(status='passed', summary={'passed_subjects': 1})):
                    if changed_after_run:
                        with self.assertRaisesRegex(RuntimeError, 'after the timed trial'):
                            verify(args)
                    else:
                        self.assertEqual(verify(args), 0)
                        result = json.loads((root/'post_validation_v2/primary_comparison.json').read_text())
                        self.assertAlmostEqual(result['comparisons'][0]['speed_ratio'], 1.2)
                        repaired = json.loads((root/'post_validation_v2/campaign.json').read_text())
                        self.assertEqual(repaired['trials'][0]['inventory_provenance']['file_count'], 1)
                self.assertEqual(path.read_bytes(), before)
                self.assertFalse((root/'subject/scientific_fingerprints.json').exists())

    def test_missing_inventory_without_documented_reader_error_cannot_be_rebuilt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trial = dict(id='subject', status='completed', output=str(root), qc_status='passed')
            (root/'campaign.json').write_text(json.dumps(dict(status='completed', subjects=[], trials=[trial])))
            with patch('scripts.audit_scientific_outputs.audit_output_root',
                       return_value=dict(status='passed', summary={'passed_subjects': 0})), \
                    self.assertRaisesRegex(RuntimeError, 'documented reader-only repair'):
                verify(SimpleNamespace(output=root, max_hours=1, rebuild_missing_fingerprints=True))


if __name__ == '__main__':
    unittest.main()
