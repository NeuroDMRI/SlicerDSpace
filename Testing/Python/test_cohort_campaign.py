"""Full-volume campaign safety and exact product comparison contracts."""

from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np

from scripts.benchmark_cohort_compare import product_hash, compare_inventories
from scripts.benchmark_cohort_workflows import inspect_cohort


class CohortCampaignTests(unittest.TestCase):
    def test_inventory_checks_gradient_count_before_accepting_subject(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'A/T1w/Diffusion'
            folder.mkdir(parents=True)
            nib.save(nib.Nifti1Image(np.ones((3,3,3,4)), np.eye(4)),folder/'data.nii.gz')
            nib.save(nib.Nifti1Image(np.ones((3,3,3)), np.eye(4)),folder/'nodif_brain_mask.nii.gz')
            np.savetxt(folder/'bvals', [0,1000,2000,3000])
            np.savetxt(folder/'bvecs', np.ones((3,4)))
            rows, _ = inspect_cohort(Path(tmp), 1)
            self.assertEqual(rows[0]['shape'], [3,3,3,4])
            np.savetxt(folder/'bvals', [0,1000,2000])
            with self.assertRaisesRegex(ValueError, 'Only 0 eligible'):
                inspect_cohort(Path(tmp), 1)

    def test_nifti_digest_preserves_dtype_affine_and_invalid_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'x.nii.gz'
            values = np.ones((3,3,3),dtype=np.float32)
            values[0,0,0] = np.nan
            nib.save(nib.Nifti1Image(values,np.eye(4)),path)
            original = product_hash(path)
            nib.save(nib.Nifti1Image(values,np.eye(4)),path)
            self.assertEqual(product_hash(path), original)
            values[0,0,0] = 0
            nib.save(nib.Nifti1Image(values,np.eye(4)),path)
            self.assertNotEqual(product_hash(path), original)

    def test_tractogram_digest_detects_reordered_streamlines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'x.tck'
            tracks = [np.array([[0,0,0],[i,1,0]], dtype=np.float32) for i in (1,2)]
            nib.streamlines.save(nib.streamlines.Tractogram(tracks, affine_to_rasmm=np.eye(4)),str(path))
            original = product_hash(path)
            nib.streamlines.save(nib.streamlines.Tractogram(tracks[::-1], affine_to_rasmm=np.eye(4)),str(path))
            self.assertNotEqual(product_hash(path), original)

    def test_missing_or_empty_product_sets_cannot_pass(self):
        self.assertEqual(compare_inventories({}, {})['status'], 'failed')
        self.assertEqual(compare_inventories({'raw':'a'}, {})['status'], 'failed')
        self.assertEqual(compare_inventories({'raw':'a'}, {'raw':'a'})['status'], 'passed')

    def test_sift_weights_ignore_path_comments_but_keep_numeric_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'sift2_weights.txt'
            path.write_text('# command_history: /reference/tracks.tck\n1 2 3\n')
            original = product_hash(path)
            path.write_text('# command_history: /candidate/tracks.tck\n1 2 3\n')
            self.assertEqual(product_hash(path), original)
            path.write_text('# command_history: /candidate/tracks.tck\n3 2 1\n')
            self.assertNotEqual(product_hash(path), original)


if __name__ == '__main__':
    unittest.main()
