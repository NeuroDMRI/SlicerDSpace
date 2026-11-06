"""Backend handoff and result-state regressions, without fitting fake anatomy."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from shared.orchestrator import ProductOrchestratorConfig, _task_from_result, plan_subject_products
from shared.ui_run_controller import RunSelection, _ukf_run_options, _append_ukf_args


class BundleAutomationTest(unittest.TestCase):
    def test_gui_forwards_converter_without_background_module_registry(self):
        selection = RunSelection(
            subject_id='s', subject_root_text='/tmp/s', output_root_text='',
            selected_targets=('bundles',), run_name='run', tractography_backend='ukf',
            bundle_backend='wma', microstructure_freewater=False, microstructure_noddi=False,
            select_streamlines=1000, ddseg_device='auto', surface_device='auto',
            connectome_backend='python', connectome_compress_mrtrix_dense=False)
        binary, converter = Path('/installed extension/UKFTractography'), Path('/another extension/DWIConvert')
        with mock.patch('tractography.ukf.resolve_executable', side_effect=[binary, converter]):
            options = _ukf_run_options(selection)
        self.assertEqual(options['dwiconvert_executable'], converter)
        args = []
        _append_ukf_args(args, dict(tractography_backend='ukf', **options))
        self.assertEqual(args[args.index('--dwiconvert-executable') + 1], str(converter))
        from shared.run_subject_pipeline_cli import build_parser
        parsed = build_parser().parse_args(['--subject-root', '/tmp/s', *args])
        self.assertEqual(parsed.ukf_executable, binary)
        self.assertEqual(parsed.dwiconvert_executable, converter)

    def test_current_qc_failure_wins_over_backend_completed_status(self):
        for name in ('bundles.wmql', 'bundles.wma'):
            self.assertEqual(_task_from_result(name, SimpleNamespace(status='completed'), status='failed').status, 'failed')
            self.assertEqual(_task_from_result(name, SimpleNamespace(status='failed'), status='completed').status, 'failed')

    def test_dti_tractcloud_plan_never_selects_old_csd(self):
        from shared.io_layout import subject_layout
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / 'input'
            inputs.mkdir()
            for name in ('data.nii.gz', 'bvals', 'bvecs', 'nodif_brain_mask.nii.gz'):
                (inputs / name).write_text('fixture')
            layout = subject_layout(root)
            old = layout.tractograms / 'old' / 's_csd_act_ifod2.vtk'
            old.parent.mkdir(parents=True)
            old.write_text('never a source for the selected DTI run')
            config = ProductOrchestratorConfig(subject_id='s', subject_root=root,
                targets=('tractography', 'bundles'), bundle_backend='tractcloud',
                tractography_backend='dti_det', run_name='new', execute=False, resume=False)
            plan = plan_subject_products(config)
            items = {item.target: item for item in plan.items}
            cloud = items['tractcloud']
            self.assertEqual(cloud.status, 'will_run')
            self.assertEqual(cloud.inputs['input_tractography'], items['tractography'].outputs['tractogram_vtk'])
            self.assertIn('dti-deterministic.vtk', cloud.inputs['input_tractography'])
            self.assertEqual(cloud.dependencies['input_tractography'], 'will_run:tractography.dti_det')
            # C1: a bundles-only selection now adds DTI tractography itself (and still never takes the old CSD file).
            plan = plan_subject_products(replace(config, targets=('bundles',)))
            cloud = next(item for item in plan.items if item.target == 'tractcloud')
            self.assertEqual(cloud.status, 'will_run')
            self.assertIn('dti-deterministic.vtk', cloud.inputs['input_tractography'])
            # With tractography skipped on purpose the blocked safety net still applies.
            plan = plan_subject_products(replace(config, targets=('bundles',), excluded_targets=('tractography',)))
            self.assertEqual(next(item for item in plan.items if item.target == 'tractcloud').status, 'blocked')

    def test_legacy_wma_cache_cannot_skip_current_sampling_revalidation(self):
        from shared.orchestrator import _wma_complete
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'wma_org_atlas_qc.json').write_text(json.dumps({'status': 'completed'}))
            with mock.patch('tractography.wma.summarize_wma_outputs') as decode:
                self.assertFalse(_wma_complete(root, input_tractography=root / 'ukf.vtk'))
                decode.assert_not_called()


if __name__ == '__main__':
    unittest.main()
