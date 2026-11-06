"""Bounded runtime search, subset planning and indexed voxel completion."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from microstructure.chunk_dispatch import chunk_results
from shared.compute_tuning import candidates, config_id, select_configuration, tuning_key, calibration_options
from shared.cohort_planning import plan_cohort
from shared.cohort_targets import selected_stages
from shared.numeric_threads import numerical_threads

GIB = 1024**3


class DynamicChunkTests(unittest.TestCase):
    def test_completed_chunk_refills_before_the_slow_head_finishes(self):
        starts, finishes, lock = {}, {}, threading.Lock()
        def fit(index):
            with lock:
                starts[index] = time.monotonic()
            time.sleep(.35 if index == 0 else .015)
            with lock:
                finishes[index] = time.monotonic()
            return index
        with ThreadPoolExecutor(max_workers=2) as pool:
            rows = list(chunk_results(5, 1, 2, lambda index:index, fit, pool=pool))
        self.assertLess(starts[2], finishes[0])
        self.assertEqual(sorted(rows), [(i, i+1, i) for i in range(5)])
        self.assertNotEqual([row[0] for row in rows], list(range(5)))

    def test_pending_payloads_stay_bounded_and_tail_offsets_are_preserved(self):
        made, consumed, peak = 0, 0, 0
        def payload(index):
            nonlocal made, peak
            made += 1
            peak = max(peak, made-consumed)
            return index
        with ThreadPoolExecutor(max_workers=3) as pool:
            for start, stop, result in chunk_results(11, 2, 3, payload, lambda index:index, pool=pool):
                self.assertEqual(start, result)
                self.assertEqual(stop, min(start+2, 11))
                consumed += 1
        # The replacement is submitted before yielding the completed result.
        self.assertLessEqual(peak, 4)
        self.assertEqual(consumed, 6)

    def test_ordered_control_and_serial_do_not_lose_chunks(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            rows = list(chunk_results(7, 3, 2, lambda index:index, lambda index:index, pool=pool, schedule='ordered'))
        self.assertEqual(rows, [(0,3,0), (3,6,3), (6,7,6)])
        self.assertEqual(rows, list(chunk_results(7,3,1,lambda index:index,lambda index:index)))


class ComputeTuningTests(unittest.TestCase):
    def test_calibration_uses_production_shell_cutoff_instead_of_hardcoded_3000(self):
        from microstructure.dki import DkiConfig
        options=calibration_options('microstructure_mkcurve')
        self.assertEqual(options['max_bvalue'], DkiConfig.__dataclass_fields__['max_bvalue'].default)
        self.assertEqual(options['max_bvalue'], 2500.)
        with patch('shared.cohort_profile.workload_key', side_effect=lambda *args:args) as key:
            before=tuning_key('microstructure_mkcurve',[8,8,8,288],[])
            with patch('shared.compute_tuning.calibration_options', return_value={**options,'max_bvalue':3000.}):
                after=tuning_key('microstructure_mkcurve',[8,8,8,288],[])
        self.assertNotEqual(before,after)

    def test_mkcurve_calibration_selects_the_actual_production_measurements(self):
        import numpy as np
        from scripts.autotune_batch import prepare_sample
        from Testing.Python.test_microstructure_dki import synthetic_case, write_case
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'subject'
            write_case(root, synthetic_case(noisy=True))
            sample = Path(tmp)/'sample'
            record = prepare_sample(root, sample, 'microstructure_mkcurve', 8)
            bvals = np.loadtxt(sample/'bvals').reshape(-1)
            self.assertEqual(record['selected_measurements'], 61)
            self.assertEqual(record['protocol']['max_bvalue'], 2500.)
            self.assertEqual(bvals.shape, (61,))
            self.assertLessEqual(bvals.max(), 2500.)

    def test_malformed_profiles_do_not_replace_declared_defaults(self):
        from shared.compute_tuning import read_profile, lookup_profile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'profile.json'
            path.write_text(json.dumps(dict(version=1,status='passed',profiles=[])))
            self.assertEqual(read_profile(path),{})
            key = tuning_key('microstructure_dki',[8,8,8,90],[])
            profile = dict(profiles={key:dict(measurements=[],configs=[])})
            self.assertIsNone(lookup_profile(profile,'microstructure_dki',[8,8,8,90],[],
                                            cpu_budget=4,memory_gb=4,jobs=2))

    def test_calibration_cli_runs_the_actual_models_and_reuses_matching_profile(self):
        import subprocess
        import sys
        from Testing.Python.test_microstructure_dki import synthetic_case, write_case
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'subject'
            write_case(root,synthetic_case(noisy=True))
            output = Path(tmp)/'profile.json'
            command = [sys.executable,str(Path(__file__).resolve().parents[2]/'scripts/autotune_batch.py'),
                       '--subject-root',str(root),'--stages','microstructure_dki,microstructure_mkcurve',
                       '--output',str(output),'--max-cpus','1','--cpu-budget','1','--sample-voxels','8']
            result = subprocess.run(command,capture_output=True,text=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stderr+'\n'+result.stdout)
            profile = json.loads(output.read_text())
            self.assertEqual(len(profile['profiles']),2)
            for row in profile['profiles'].values():
                self.assertEqual(len(row['measurements']),3)
                self.assertTrue(all(item['exact'] for item in row['measurements']))
            result = subprocess.run(command,capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr)
            reused = json.loads(output.read_text())
            self.assertEqual(profile['profiles'],reused['profiles'])

    def records(self):
        configs = [dict(threads=1,workers=1,chunk=64), dict(threads=4,workers=1,chunk=64)]
        rows = [dict(input='A',repeat=repeat,config=config_id(config),status='completed',exact=True,
                     seconds=4 if config['threads']==1 else 3,rss_bytes=GIB)
                for config in configs for repeat in range(3)]
        return configs, rows

    def test_latency_and_batch_capacity_can_choose_different_configurations(self):
        configs, rows = self.records()
        throughput = select_configuration(rows,configs,cpu_budget=4,memory_gb=16,jobs=4)
        latency = select_configuration(rows,configs,cpu_budget=4,memory_gb=16,jobs=4,objective='latency')
        self.assertEqual(throughput['selected']['settings']['threads'],1)
        self.assertEqual(latency['selected']['settings']['threads'],4)

    def test_mismatching_or_incomplete_candidates_cannot_win(self):
        configs, rows = self.records()
        rows[-1]['exact'] = False
        choice = select_configuration(rows,configs,cpu_budget=4,memory_gb=16,jobs=1)
        self.assertEqual(len(choice['candidates']),1)
        rows.pop()
        choice = select_configuration(rows,configs,cpu_budget=4,memory_gb=16,jobs=1)
        self.assertEqual(choice['selected']['settings']['threads'],1)

    def test_small_or_noisy_gains_retain_the_baseline(self):
        configs, rows = self.records()
        for row in rows:
            row['seconds'] = 4 if row['config']==config_id(configs[0]) else 3.95
        choice = select_configuration(rows,configs,cpu_budget=4,memory_gb=16,jobs=1)
        self.assertEqual(choice['selected']['settings']['threads'],1)

    def test_search_is_bounded_and_uses_workers_in_cpu_count(self):
        for config in candidates('microstructure_mkcurve',4):
            self.assertLessEqual(config['workers']*config['threads'],4)
        self.assertEqual(candidates('surface',4),[])

    def test_key_is_per_method_not_whole_pipeline_selection(self):
        base = ['--dki-fit-method','WLS']
        self.assertEqual(tuning_key('microstructure_dki',[8,8,8,90],base),
            tuning_key('microstructure_dki',[8,8,8,90],base+['--targets','microstructure,tractography','--run-noddi']))
        self.assertNotEqual(tuning_key('microstructure_dki',[8,8,8,90],base),
                            tuning_key('microstructure_dki',[8,8,8,90],['--dki-fit-method','OLS']))

    def test_environment_and_library_limits_restore_on_failure(self):
        with patch.dict(os.environ,{'OMP_NUM_THREADS':'7','SLICERDSPACE_STAGE_CPUS':'2'}):
            with self.assertRaisesRegex(RuntimeError,'probe failure'):
                with numerical_threads(1,workers=2):
                    self.assertEqual(os.environ['OMP_NUM_THREADS'],'1')
                    raise RuntimeError('probe failure')
            self.assertEqual(os.environ['OMP_NUM_THREADS'],'7')
            with self.assertRaisesRegex(ValueError,'reservation'):
                with numerical_threads(2,workers=2):
                    self.fail('Oversubscription was admitted')


class SubsetPlanningTests(unittest.TestCase):
    def plan(self, arguments, **extra):
        return plan_cohort([SimpleNamespace(subject_id='A',subject_root=Path('/unused/A'))],arguments,
                           cpu_budget=4,memory_gb=32,jobs=2,adaptive=True,available=(4,32*GIB),**extra)

    def test_single_model_does_not_select_all_models_or_tracking(self):
        selected = set(self.plan(['--targets','microstructure','--microstructure-stages','dki'])['selected_stages'])
        self.assertIn('microstructure_dki',selected)
        self.assertIn('microstructure_dti',selected)
        self.assertFalse(selected & {'microstructure_mkcurve','microstructure_noddi','csd_act','wma','surface'})

    def test_existing_bundle_only_does_not_request_new_tracking(self):
        selected = selected_stages(['--targets','wma','--wma-input-tractography','/existing.vtp'])
        self.assertEqual(selected,{'wma'})

    def test_repeated_target_and_skip_semantics_match_subject_cli(self):
        selected = selected_stages(['--targets','surface','--targets','microstructure',
                                   '--microstructure-stages','noddi','--skip-targets','surface'])
        self.assertIn('microstructure_noddi',selected)
        self.assertNotIn('surface',selected)
        self.assertNotIn('registered_dti',selected)

    def test_validation_only_does_not_reserve_the_full_pipeline(self):
        self.assertEqual(self.plan(['--targets','validation'])['selected_stages'],[])

    def test_unused_algorithm_hints_do_not_block_a_small_route(self):
        self.plan(['--targets','connectome','--surface-cpu-threads','1000','--mkcurve-workers','1000'])

    def test_selected_explicit_worker_and_thread_product_is_reserved(self):
        plan = self.plan(['--targets','microstructure_dki','--dki-fit-method','MKCURVE',
                          '--mkcurve-workers','2','--dki-threads','2'])
        self.assertEqual(plan['subject_profiles']['A']['microstructure_mkcurve']['cpu'],4)
        with self.assertRaises(ValueError):
            self.plan(['--targets','microstructure_dki','--dki-fit-method','MKCURVE',
                       '--mkcurve-workers','3','--dki-threads','2'])

    def test_matching_calibration_applies_only_to_selected_automatic_models(self):
        from shared.cohort_planning import image_size
        arguments = ['--targets','microstructure_dki','--dki-fit-method','WLS']
        configs = [dict(threads=1,workers=1,chunk=1024),dict(threads=2,workers=1,chunk=1024)]
        rows = [dict(config=config_id(config),input='A',repeat=repeat,status='completed',exact=True,
                     seconds=10 if config['threads']==1 else 2,rss_bytes=GIB)
                for config in configs for repeat in range(3)]
        key = tuning_key('microstructure_dki',image_size('/unused/A')['shape'],arguments)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'profile.json'
            path.write_text(json.dumps(dict(version=1,status='passed',profiles={key:dict(configs=configs,measurements=rows)})))
            plan = self.plan(arguments,tuning_profile=path)
            self.assertEqual(plan['subject_profiles']['A']['microstructure_dki']['cpu'],2)
            self.assertIn('--dki-threads',plan['subject_arguments']['A'])
            explicit = self.plan(arguments+['--dki-threads','1'],tuning_profile=path)
            self.assertEqual(explicit['subject_profiles']['A']['microstructure_dki']['cpu'],1)
            self.assertEqual(explicit['subject_arguments']['A'],[])


if __name__ == '__main__':
    unittest.main()
