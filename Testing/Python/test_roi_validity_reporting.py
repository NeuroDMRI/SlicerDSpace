"""Invalid scalar estimates must not become regional zero measurements."""

import json
import csv
import hashlib
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np

from microstructure.pipeline import _compute_roi_stats, _roi_validity_mask, roi_statistics_current


class RegionalValidityTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.roi = self.volume('roi', [1, 1, 2, 2])

    def volume(self, name, values, affine=None):
        path = self.root / f'{name}.nii.gz'
        nib.save(nib.Nifti1Image(np.asarray(values, dtype=np.float32).reshape(4, 1, 1),
                                np.eye(4) if affine is None else affine), path)
        return path

    def compute(self, paths, masks=None):
        _compute_roi_stats(metric_paths=paths, roi_label_map=self.roi, lut_path=None,
                           csv_path=self.root / 'stats.csv', json_path=self.root / 'stats.json',
                           validity_masks=masks)
        return json.loads((self.root / 'stats.json').read_text())

    def test_metric_specific_counts_and_empty_estimates_not_zero(self):
        scalar = self.volume('scalar', [0.2, 0.4, 0.6, 0.8])
        mk = self.volume('mk', [1.2, 999, 888, 777])
        result = self.compute({'fa': scalar, 'mkcurve_corrected_mk': mk},
                              {'mkcurve_corrected_mk': np.asarray([1, 0, 0, 0]).reshape(4, 1, 1)})
        self.assertEqual(result['schema_version'], 2)
        first, second = result['rows']
        self.assertEqual(first['roi_voxel_count'], 2)
        self.assertEqual(first['fa_valid_voxel_count'], 2)
        self.assertEqual(first['mkcurve_corrected_mk_valid_voxel_count'], 1)
        self.assertEqual(first['mkcurve_corrected_mk_valid_fraction'], .5)
        self.assertAlmostEqual(first['mkcurve_corrected_mk_mean'], 1.2, places=5)
        self.assertEqual(second['mkcurve_corrected_mk_excluded_voxel_count'], 2)
        self.assertIsNone(second['mkcurve_corrected_mk_mean'])

    def test_explicit_freewater_mask_cannot_be_overwritten_by_nonzero_pair(self):
        fa = self.volume('fw_fa', [0, .9, .4, .6])
        md = self.volume('fw_md', [0, .002, .001, .001])
        valid = np.asarray([1, 0, 1, 1]).reshape(4, 1, 1)
        result = self.compute({'fw_tissue_fa': fa, 'fw_tissue_md': md},
                              {'fw_tissue_fa': valid, 'fw_tissue_md': valid})
        first = result['rows'][0]
        self.assertEqual(first['fw_tissue_fa_valid_voxel_count'], 1)
        self.assertEqual(first['fw_tissue_fa_mean'], 0)
        self.assertEqual(result['validity_policy']['fw_tissue_fa'], 'explicit_validity_mask')

    def test_empty_first_metric_does_not_drop_other_metrics_or_regions(self):
        scalar = self.volume('scalar', [.1, .2, .3, .4])
        result = self.compute({'one': scalar, 'two': scalar}, {'one': np.zeros((4, 1, 1))})
        self.assertEqual(result['roi_count'], 2)
        for row in result['rows']:
            self.assertIsNone(row['one_mean'])
            self.assertEqual(row['one_valid_voxel_count'], 0)
            self.assertEqual(row['two_valid_voxel_count'], 2)

    def test_nonfinite_or_nonbinary_validity_is_rejected(self):
        scalar = self.volume('scalar', [1, 2, 3, 4])
        for invalid in (np.nan, .5, 2):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'finite and binary'):
                self.compute({'scalar': scalar}, {'scalar': np.array([1, invalid, 0, 1]).reshape(4, 1, 1)})

    def test_native_validity_mask_grid_and_values_are_verified(self):
        scalar = self.volume('scalar', [1, 2, 3, 4])
        affine = np.eye(4)
        affine[0, 3] = 10
        mask = self.volume('shifted_mask', [1, 0, 1, 1], affine)
        with self.assertRaisesRegex(ValueError, 'native grid'):
            _roi_validity_mask(mask, scalar)
        mask = self.volume('bad_mask', [1, 0, np.nan, 1])
        with self.assertRaisesRegex(ValueError, 'finite, binary'):
            _roi_validity_mask(mask, scalar)

    def test_cache_rejects_legacy_tables_changed_sources_and_changed_csv(self):
        scalar = self.volume('scalar', [1, 2, 3, 4])
        self.compute({'scalar': scalar})
        current = lambda: roi_statistics_current(self.root / 'stats.json', self.root / 'stats.csv',
                                                 roi_label_map=self.roi)
        self.assertTrue(current())
        original = (self.root / 'stats.json').read_text()
        payload = json.loads(original)
        payload['schema_version'] = 1
        (self.root / 'stats.json').write_text(json.dumps(payload))
        self.assertFalse(current())
        (self.root / 'stats.json').write_text(original)
        (self.root / 'stats.csv').write_text('corrupted table\n')
        self.assertFalse(current())
        self.compute({'scalar': scalar})
        self.volume('scalar', [5, 6, 7, 8])
        self.assertFalse(current())

    def test_file_backed_validity_is_required_for_reusable_masked_tables(self):
        scalar = self.volume('scalar', [1, 999, 3, 4])
        valid = self.volume('valid', [1, 0, 1, 1])
        kwargs = dict(metric_paths={'dki_mk': scalar}, roi_label_map=self.roi, lut_path=None,
                      csv_path=self.root / 'stats.csv', json_path=self.root / 'stats.json')
        _compute_roi_stats(**kwargs, validity_masks={'dki_mk': np.array([1, 0, 1, 1]).reshape(4, 1, 1)})
        self.assertFalse(roi_statistics_current(kwargs['json_path'], kwargs['csv_path']))
        _compute_roi_stats(**kwargs, validity_mask_paths={'dki_mk': valid})
        self.assertTrue(roi_statistics_current(kwargs['json_path'], kwargs['csv_path']))
        self.volume('valid', [1, 1, 1, 1])
        self.assertFalse(roi_statistics_current(kwargs['json_path'], kwargs['csv_path']))

    def test_conflicting_validity_array_and_file_is_rejected(self):
        scalar = self.volume('scalar', [1, 2, 3, 4])
        valid = self.volume('valid', [1, 0, 1, 1])
        with self.assertRaisesRegex(ValueError, 'array and file disagree'):
            _compute_roi_stats(metric_paths={'scalar': scalar}, roi_label_map=self.roi, lut_path=None,
                               csv_path=self.root / 'stats.csv', json_path=self.root / 'stats.json',
                               validity_masks={'scalar': np.ones((4, 1, 1))},
                               validity_mask_paths={'scalar': valid})

    def test_orchestrator_cannot_resume_dti_past_old_regional_statistics(self):
        from unittest.mock import patch
        from microstructure.pipeline import DTI_KEYS
        from shared.io_layout import subject_layout
        from shared.orchestrator import _microstructure_dti_complete

        subject = self.root / 'subject'
        layout = subject_layout(subject)
        regional = layout.regional_microstructure
        scalar = self.volume('scalar', [1, 2, 3, 4])
        json_path = regional / 'test_ROIWise-Microstructure.json'
        csv_path = regional / 'test_ROIWise-Microstructure.csv'
        brain = self.volume('brain', [1, 1, 1, 1])
        dec = layout.microstructure_dti / 'test-dti-DEC.nrrd'  # D6: the DTI stage is complete only with the DEC map
        dec.parent.mkdir(parents=True, exist_ok=True)
        dec.write_bytes(b'dec')
        with patch('shared.orchestrator._resolve_parcellation', return_value=self.roi), \
                patch('shared.orchestrator._resolve_analysis_inputs', return_value={'mask': brain}), \
                patch('shared.scalar_qc.scalar_qc_current', return_value=True), \
                patch('shared.orchestrator._manifest_inputs_match', return_value=True):
            self.assertFalse(_microstructure_dti_complete(subject, 'test'))
            _compute_roi_stats(metric_paths={key: scalar for key in DTI_KEYS}, roi_label_map=self.roi, lut_path=None,
                               csv_path=csv_path, json_path=json_path, brain_mask_path=brain)
            self.assertTrue(_microstructure_dti_complete(subject, 'test'))
            self.volume('roi', [1, 2, 2, 2])
            self.assertFalse(_microstructure_dti_complete(subject, 'test'))

    def test_brain_mask_excludes_atlas_boundary_outside_the_fitted_domain(self):
        scalar = self.volume('scalar', [1, 0, 3, 4])
        brain = self.volume('brain', [1, 0, 1, 1])
        csv_path, json_path = self.root / 'stats.csv', self.root / 'stats.json'
        _compute_roi_stats(metric_paths={'fa': scalar}, roi_label_map=self.roi, lut_path=None,
                           csv_path=csv_path, json_path=json_path, brain_mask_path=brain)
        first = json.loads(json_path.read_text())['rows'][0]
        self.assertEqual(first['roi_voxel_count'], 2)
        self.assertEqual(first['fa_valid_voxel_count'], 1)
        self.assertEqual(first['fa_mean'], 1)
        self.assertTrue(roi_statistics_current(json_path, csv_path, brain_mask_path=brain))
        self.volume('brain', [1, 1, 1, 1])
        self.assertFalse(roi_statistics_current(json_path, csv_path, brain_mask_path=brain))

    def test_independent_audit_checks_raw_estimates_without_repair(self):
        from unittest.mock import patch
        from scripts.audit_regional_validity import audit_statistics

        scalar = self.volume('scalar', [.5, 999, 2, 3])
        valid = self.volume('valid', [1, 0, 0, 0])
        csv_path, json_path = self.root / 'stats.csv', self.root / 'stats.json'
        _compute_roi_stats(metric_paths={'mkcurve_corrected_mk': scalar}, roi_label_map=self.roi, lut_path=None,
                           csv_path=csv_path, json_path=json_path, validity_mask_paths={'mkcurve_corrected_mk': valid})
        before = {path: path.read_bytes() for path in (scalar, valid, self.roi, csv_path, json_path)}
        with patch('scripts.audit_regional_validity._compute_roi_stats', side_effect=AssertionError('No repair')):
            result = audit_statistics(json_path, csv_path)
        self.assertTrue(result['automatic_qc_passed'], result)
        self.assertEqual(result['coverage']['mkcurve_corrected_mk']['excluded_voxels'], 3)
        self.assertEqual(before, {path: path.read_bytes() for path in before})

        payload = json.loads(json_path.read_text())
        payload['rows'][0]['mkcurve_corrected_mk_mean'] = 7.0
        with csv_path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=payload['rows'][0].keys())
            writer.writeheader()
            writer.writerows(payload['rows'])
        payload['csv_sha256'] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
        json_path.write_text(json.dumps(payload))
        self.assertTrue(roi_statistics_current(json_path, csv_path))
        result = audit_statistics(json_path, csv_path)
        self.assertFalse(result['automatic_qc_passed'])
        self.assertIn('mean', [item['failure'] for item in result['failures']])

    def test_joint_audit_rejects_missing_declared_regional_tables(self):
        from scripts.audit_scientific_outputs import _audit_regional_statistics

        checks = []
        manifest = dict(tasks=[dict(name='microstructure.dti', status='completed',
            outputs=dict(roi_stats_json=str(self.root / 'absent.json'), roi_stats_csv=str(self.root / 'absent.csv')))])
        _audit_regional_statistics(manifest, checks)
        self.assertEqual(checks[0]['name'], 'microstructure.regional_statistics')
        self.assertFalse(checks[0]['passed'])


if __name__ == '__main__':
    unittest.main()
