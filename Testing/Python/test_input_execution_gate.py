"""Real input validation must precede DWI preparation, including on resume."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import nibabel as nib
import numpy as np

from shared import ProductOrchestratorConfig, run_subject_products


def subject(root):
    folder = root / 'sub-gate' / 'input'
    folder.mkdir(parents=True)
    nib.save(nib.Nifti1Image(np.ones((5, 5, 5, 4), dtype=np.float32), np.eye(4)), folder / 'data.nii.gz')
    mask = np.zeros((5, 5, 5), dtype=np.uint8)
    mask[1:4, 1:4, 1:4] = 1
    nib.save(nib.Nifti1Image(mask, np.eye(4)), folder / 'nodif_brain_mask.nii.gz')
    (folder / 'bvals').write_text('0 1000 1000 1000')
    (folder / 'bvecs').write_text('0 1 0 0\n0 0 1 0\n0 0 0 1\n')
    return folder.parent


class InputExecutionGateTests(unittest.TestCase):
    def test_invalid_input_never_reaches_dti_or_inference(self):
        for defect in ('affine', 'gradient_count', 'nonfinite'):
            for resume in (False, True):
                with self.subTest(defect=defect, resume=resume), tempfile.TemporaryDirectory() as tmp:
                    root = subject(Path(tmp))
                    if defect == 'affine':
                        path = root / 'input/nodif_brain_mask.nii.gz'
                        image = nib.load(path)
                        affine = image.affine.copy()
                        affine[0, 3] = 10
                        nib.save(nib.Nifti1Image(np.asarray(image.dataobj), affine), path)
                    elif defect == 'gradient_count':
                        (root / 'input/bvals').write_text('0 1000 1000')
                        (root / 'input/bvecs').write_text('0 1 0\n0 0 1\n0 0 0\n')
                    else:
                        (root / 'input/bvals').write_text('0 1000 1000 inf')
                    with mock.patch('shared.pipeline.ensure_native_dti_bundle') as prepare, mock.patch(
                            'anatomy.segmentation.pipeline.run_ddseg') as infer:
                        result = run_subject_products(ProductOrchestratorConfig(
                            'sub-gate', root, targets=('segmentation',), execute=True,
                            resume=resume, brain_mask_backend='provided'))
                    prepare.assert_not_called()
                    infer.assert_not_called()
                    tasks = {task.name: task for task in result.tasks}
                    self.assertEqual(tasks['segmentation.ddseg'].status, 'blocked')
                    record = json.loads(result.run_manifest_path.read_text())
                    self.assertEqual(record['status'], 'failed')
                    self.assertIn('segmentation', record['input_blocked_targets'])
                    self.assertTrue(Path(record['input_validation_path']).is_file())

    def test_bad_raw_input_does_not_block_independent_existing_bundle_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = subject(Path(tmp))
            (root / 'input/bvals').write_text('0 1000 1000 inf')
            with mock.patch('shared.orchestrator._wma_complete', return_value=True):
                result = run_subject_products(ProductOrchestratorConfig(
                    'sub-gate', root, targets=('validation', 'wma'), execute=True, resume=True,
                    brain_mask_backend='provided'))
            self.assertEqual(next(task for task in result.tasks if task.name == 'bundles.wma').status, 'skipped')

    def test_bad_raw_input_still_blocks_wma_when_the_existing_result_is_not_current(self):
        # F1: the resume reuse applies only when _wma_complete verifies the existing result.
        with tempfile.TemporaryDirectory() as tmp:
            root = subject(Path(tmp))
            (root / 'input/bvals').write_text('0 1000 1000 inf')
            with mock.patch('shared.orchestrator._wma_complete', return_value=False):
                result = run_subject_products(ProductOrchestratorConfig(
                    'sub-gate', root, targets=('validation', 'wma'), execute=True, resume=True,
                    brain_mask_backend='provided'))
            self.assertEqual(next(task for task in result.tasks if task.name == 'bundles.wma').status, 'blocked')

    def _wma_task_with_valid_input(self, ukf_patch, ukf_current):
        with tempfile.TemporaryDirectory() as tmp:
            root = subject(Path(tmp))
            with mock.patch('shared.orchestrator._wma_complete', return_value=True), \
                    mock.patch('shared.orchestrator._ukf_current', return_value=ukf_current), \
                    mock.patch('tractography.ukf.run_ukf', **ukf_patch):
                result = run_subject_products(ProductOrchestratorConfig(
                    'sub-gate', root, targets=('wma',), tractography_backend='ukf', execute=True, resume=True,
                    brain_mask_backend='provided'))
            return {task.name: task for task in result.tasks}

    def test_b1_failed_ukf_run_keeps_wma_blocked_even_with_a_complete_wma_result(self):
        # B1: valid input, UKF raises. The WMA result may come from an older tractogram.
        tasks = self._wma_task_with_valid_input(dict(side_effect=RuntimeError('ukf crashed')), False)
        self.assertEqual(tasks['tractography.ukf'].status, 'failed')
        self.assertEqual(tasks['bundles.wma'].status, 'blocked')

    def test_b1_stale_ukf_output_keeps_wma_blocked_even_with_a_complete_wma_result(self):
        # B1: valid input, UKF did not run and its output is not current.
        tasks = self._wma_task_with_valid_input(dict(return_value={'status': 'completed'}), False)
        self.assertEqual(tasks['bundles.wma'].status, 'blocked')
