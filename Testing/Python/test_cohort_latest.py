"""Latest cohorts retain historical evidence and isolate tuning from scheduling."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.benchmark_cohort_latest import prepare_campaign, function_ast
from scripts.benchmark_cohort_workflows import update_comparisons


class LatestCampaignTests(unittest.TestCase):
    def test_latest_campaign_contains_a_fixed_and_a_tuned_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'source.json'
            source.write_text(json.dumps(dict(status='prepared',subjects=[dict(subject='A')],notes=[],
                trials=[dict(id='old',route='dki',policy='subject',status='pending')],snapshot='/old')))
            profile = root/'profile.json'
            profile.write_text('{}')
            result = prepare_campaign(source,root/'latest',dict(snapshot='/new',hashes={}), 'dki',profile)
            record = json.loads(result.read_text())
            self.assertEqual([row['policy'] for row in record['trials']],['subject','adaptive','adaptive_tuned'])
            self.assertTrue(all(row['snapshot']=='/new' for row in record['trials']))
            self.assertIn('compute_profile_sha256',record['trials'][-1])
            self.assertEqual(json.loads(source.read_text())['snapshot'],'/old')

    def test_runtime_selection_effect_is_compared_to_the_fixed_latest_scheduler(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for policy,seconds in (('subject',120),('adaptive',100),('adaptive_tuned',80)):
                output = root/policy
                output.mkdir()
                (output/'scientific_fingerprints.json').write_text(json.dumps({'A':{'params':'same'}}))
                rows.append(dict(id=policy,route='dki',policy=policy,threads=4,status='completed',
                                 output=str(output),qc_status='passed',measurement=dict(execution_seconds=seconds)))
            update_comparisons(dict(trials=rows,subjects=[dict(subject='A')],reference_policy='subject',
                                    cpus=16,memory_gb=32,jobs=4),root)
            result = json.loads((root/'configuration_comparison.json').read_text())['comparisons'][0]
            self.assertEqual(result['reference'],'adaptive')
            self.assertEqual(result['candidate'],'adaptive_tuned')
            self.assertAlmostEqual(result['speed_ratio'],1.25)

    def test_execution_changes_do_not_mask_a_numerical_kernel_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'fit.py'
            path.write_text('def fit(x):\n    return x * 2\n')
            original = function_ast(path,'fit')
            path.write_text('def fit(x):\n    return x * 3\n')
            self.assertNotEqual(original,function_ast(path,'fit'))


if __name__ == '__main__':
    unittest.main()
