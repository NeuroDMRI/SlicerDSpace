"""Full-volume campaign safety and exact product comparison contracts."""

from pathlib import Path
import tempfile
import unittest
import json

import nibabel as nib
import numpy as np

from scripts.benchmark_cohort_compare import product_hash, compare_inventories
from scripts.benchmark_cohort_workflows import inspect_cohort, select_campaign, update_comparisons


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

    def test_selecting_pending_arms_preserves_the_original_and_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/'original.json'
            source.write_text(json.dumps(dict(status='prepared',subjects=[{'subject':'A'}],notes=[],
                snapshot='/frozen/science',source_hashes={'model.py':'hash'},trials=[
                    dict(id=policy,policy=policy,route='dki',threads=4,status='pending')
                    for policy in ('sequential','subject','stage','adaptive')])))
            original=source.read_bytes()
            selected=select_campaign(source,root/'selected',policies=['subject','adaptive'],routes=['dki'])
            result=json.loads(selected.read_text())
            self.assertEqual(source.read_bytes(),original)
            self.assertEqual(result['snapshot'],'/frozen/science')
            self.assertEqual([row['policy'] for row in result['trials']],['subject','adaptive'])
            self.assertEqual(result['reference_policy'],'subject')
            self.assertEqual(result['omitted_trials'],['sequential','stage'])

    def test_selecting_an_active_arm_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/'original.json'
            source.write_text(json.dumps(dict(trials=[dict(id='subject',policy='subject',route='dki',
                threads=4,status='running'),dict(id='adaptive',policy='adaptive',route='dki',threads=4,status='pending')])))
            with self.assertRaisesRegex(ValueError,'finish active work'):
                select_campaign(source,root/'selected',policies=['subject','adaptive'],routes=['dki'])
            self.assertFalse((root/'selected').exists())

    def test_primary_pair_needs_both_scientific_equivalence_and_qc(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            trials=[]
            for policy,seconds in (('subject',120),('adaptive',100)):
                folder=root/policy
                folder.mkdir()
                (folder/'scientific_fingerprints.json').write_text(json.dumps({'A':{'params':'same'}}))
                trials.append(dict(id=policy,route='dki',threads=4,policy=policy,status='completed',
                    output=str(folder),qc_status='passed',measurement=dict(execution_seconds=seconds)))
            record=dict(reference_policy='subject',subjects=[{'subject':'A'}],trials=trials,cpus=8,memory_gb=16,jobs=2)
            update_comparisons(record,root)
            result=json.loads((root/'primary_comparison.json').read_text())['comparisons'][0]
            self.assertEqual(result['status'],'passed')
            self.assertAlmostEqual(result['speed_ratio'],1.2)
            trials[1]['qc_status']='failed'
            update_comparisons(record,root)
            result=json.loads((root/'primary_comparison.json').read_text())['comparisons'][0]
            self.assertIsNone(result['speed_ratio'])
            self.assertFalse((root/'subject/comparison.json').exists())

    def test_primary_pair_rejects_an_incomplete_participant_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            trials=[]
            for policy,subjects in (('subject',{'A':{'params':'same'},'B':{'params':'same'}}),
                                    ('adaptive',{'A':{'params':'same'}})):
                folder=root/policy
                folder.mkdir()
                (folder/'scientific_fingerprints.json').write_text(json.dumps(subjects))
                trials.append(dict(id=policy,route='dki',threads=4,policy=policy,status='completed',
                    output=str(folder),qc_status='passed',measurement=dict(execution_seconds=100)))
            record=dict(reference_policy='subject',subjects=[{'subject':'A'},{'subject':'B'}],trials=trials,
                        cpus=8,memory_gb=16,jobs=2)
            update_comparisons(record,root)
            self.assertEqual(trials[1]['comparison_status'],'failed')

    def test_boundary_handover_requires_completed_subject_and_no_started_stage(self):
        from scripts.benchmark_cohort_priority import validate_handover
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            boundary=root/'trials/stage'
            boundary.mkdir(parents=True)
            (boundary/'priority_boundary_hold.json').write_text('{}')
            source=root/'campaign.json'
            trials=[dict(policy=policy,status='completed' if policy in ('sequential','subject') else 'pending')
                    for policy in ('sequential','subject','stage','adaptive')]
            source.write_text(json.dumps(dict(trials=trials)))
            (root/'driver.log').write_text('FileExistsError: '+str(boundary))
            record=dict(source=str(source),boundary=str(boundary))
            self.assertEqual(validate_handover(record)['trials'],trials)
            trials[1]['status']='running'
            source.write_text(json.dumps(dict(trials=trials)))
            with self.assertRaisesRegex(ValueError,'did not finish'):
                validate_handover(record)
            trials[1]['status']='completed'
            source.write_text(json.dumps(dict(trials=trials)))
            (boundary/'stdout.log').write_text('Unexpected trial already started')
            with self.assertRaisesRegex(ValueError,'Unexpected work'):
                validate_handover(record)

    def test_completed_evidence_is_reused_without_changing_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            evidence=root/'subject'
            evidence.mkdir()
            for name in ('independent_qc.json','scientific_fingerprints.json'):
                (evidence/name).write_text('{}')
            source=root/'original.json'
            measurement=dict(execution_seconds=123.456,peak_tree_rss_bytes=12345)
            source.write_text(json.dumps(dict(notes=[],trials=[
                dict(id='subject',policy='subject',route='dki',threads=4,status='completed',
                     output=str(evidence),measurement=measurement),
                dict(id='adaptive',policy='adaptive',route='dki',threads=4,status='pending')],
                snapshot='/frozen/original')))
            selected=select_campaign(source,root/'selected',policies=['subject','adaptive'],routes=['dki'])
            row=json.loads(selected.read_text())['trials'][0]
            self.assertEqual(row['measurement'],measurement)
            self.assertEqual(row['output'],str(evidence))
            self.assertEqual(len(row['inherited_evidence']),2)


if __name__ == '__main__':
    unittest.main()
