"""Actual isolated processes: overlap, resource bounds, failure and GPU policy."""

from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from shared.compute_schedule import ComputeJob, run_independent_jobs


class ComputeScheduleTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def jobs(self, count=2, threads=1, memory=1):
        return [ComputeJob(str(i), (sys.executable, '-c',
            'import time; time.sleep(.4)'), threads=threads, memory_bytes=memory) for i in range(count)]

    def run_jobs(self, jobs, name, **kwargs):
        return run_independent_jobs(jobs, directory=self.root/name, max_workers=2,
                                    cpu_budget=4, memory_budget=10, **kwargs)

    def test_disjoint_processes_overlap_and_keep_true_job_durations(self):
        record = self.run_jobs(self.jobs(), 'parallel')
        rows = record['jobs']
        self.assertTrue(all(row['status']=='completed' for row in rows))
        self.assertLess(abs(rows[0]['start_offset_seconds']-rows[1]['start_offset_seconds']), .3)
        self.assertGreater(record['job_seconds_sum'], record['elapsed_seconds']*1.3)

    def test_cpu_and_memory_limits_serialize_without_changing_jobs(self):
        for field in ('threads','memory'):
            record = self.run_jobs(self.jobs(threads=3 if field=='threads' else 1,
                memory=6 if field=='memory' else 1), field)
            self.assertGreaterEqual(record['jobs'][1]['start_offset_seconds'],
                record['jobs'][0]['start_offset_seconds'] + record['jobs'][0]['runtime_seconds'] - .05)

    def test_oversized_job_rejected_before_start_and_unknown_memory_serializes(self):
        with self.assertRaisesRegex(ValueError, 'resource budget'):
            self.run_jobs(self.jobs(memory=11), 'too_large')
        record=run_independent_jobs(self.jobs(), directory=self.root/'unknown', max_workers=2,
                                    cpu_budget=4, memory_budget=0)
        self.assertEqual(record['max_workers'],1)

    def test_failure_cancels_running_sibling_and_retains_evidence(self):
        jobs=[ComputeJob('fail',(sys.executable,'-c','import time;time.sleep(.1);raise SystemExit(7)')),
              ComputeJob('sibling',(sys.executable,'-c','import time;time.sleep(30)'))]
        with self.assertRaisesRegex(RuntimeError, 'exit code 7'):
            self.run_jobs(jobs, 'failed')
        report=json.loads((self.root/'failed/schedule.json').read_text())
        self.assertEqual(report['status'],'failed')
        self.assertEqual([row['status'] for row in report['jobs']],['failed','cancelled'])
        self.assertLess(report['elapsed_seconds'],10)

    def test_gpu_environment_inherited_and_parent_environment_unmodified(self):
        path=self.root/'device.json'
        code='import os,json;from pathlib import Path;Path('+repr(str(path))+').write_text(json.dumps(dict(cuda=os.environ.get("CUDA_VISIBLE_DEVICES"),omp=os.environ.get("OMP_NUM_THREADS"))))'
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'2','OMP_NUM_THREADS':'8'}):
            self.run_jobs([ComputeJob('probe',(sys.executable,'-c',code),environment={'OMP_NUM_THREADS':'1'})],'environment')
            self.assertEqual(os.environ['OMP_NUM_THREADS'],'8')
        self.assertEqual(json.loads(path.read_text()),{'cuda':'2','omp':'1'})

    def test_invalid_names_or_duplicate_jobs_cannot_escape_logs(self):
        for names in (('same','same'),('../escape','valid')):
            with self.assertRaises(ValueError):
                self.run_jobs([ComputeJob(name,(sys.executable,'-c','')) for name in names],'invalid')

    def test_config_roundtrip_and_cli_ui_expose_parallel_limit(self):
        from microstructure.compute_worker import encode,decode
        from microstructure.pipeline import MicrostructureConfig
        from shared.pipeline import RawInputBundle
        from shared.run_subject_pipeline_cli import build_parser
        from shared.ui_settings import COMMON_USER_SETTING_SPECS
        raw=RawInputBundle(self.root,'fixture',self.root,*[self.root/name for name in ('dwi','bval','bvec','mask')])
        config=MicrostructureConfig('fixture',raw,None,self.root,max_parallel_models=2)
        self.assertEqual(decode(json.loads(json.dumps(encode(config)))),config)
        args=build_parser().parse_args(['--subject-root',str(self.root),'--subject-id','fixture','--max-parallel-models','3'])
        self.assertEqual(args.max_parallel_models,3)
        self.assertIn(('maxParallelModels',int),COMMON_USER_SETTING_SPECS)

    def test_fresh_serial_noddi_does_not_refit_preceding_freewater(self):
        from types import SimpleNamespace
        from shared.orchestrator import ProductOrchestratorConfig,run_subject_products
        from shared.pipeline import RawInputBundle
        source=self.root/'input'
        source.mkdir()
        for name in ('data.nii.gz','bvals','bvecs','nodif_brain_mask.nii.gz'):
            (source/name).write_text('fixture')
        from test_product_orchestrator import _write_valid_raw_subject
        _write_valid_raw_subject(self.root)
        (source/'bvals').write_text('0 1000 2000 2000')  # D3: Free Water needs three b-value groups
        raw=RawInputBundle(self.root,'fixture',source,*[source/name for name in ('data.nii.gz','bvals','bvecs','nodif_brain_mask.nii.gz')])
        names=('fa','md','ad','rd','l1','l2','l3','trace','fwf','fw_tissue_fa','fw_tissue_md',
               'fw_tissue_valid_mask','free_water_qc','noddi_ndi','noddi_odi','noddi_fwf',
               'noddi_tissue_valid_mask','noddi_rmse','noddi_nrmse','noddi_qc')
        result=SimpleNamespace(**{name+'_path':self.root/(name+'.nii.gz') for name in names},
                               roi_stats_csv_path=None,roi_stats_json_path=None)
        with patch('shared.pipeline.find_subject_root',return_value=raw), \
             patch('shared.pipeline.resolve_preprocessed_input_bundle',return_value=raw), \
             patch('shared.pipeline.ensure_native_dti_bundle',return_value=SimpleNamespace()), \
             patch('microstructure.run_microstructure',return_value=result) as compute:
            run_subject_products(ProductOrchestratorConfig('fixture',self.root,('microstructure',),
                execute=True,resume=False,microstructure_run_free_water=True,microstructure_run_noddi=True))
        options=[call.args[0] for call in compute.call_args_list]
        self.assertEqual(len(options),3)
        self.assertEqual(sum(config.run_free_water for config in options),1)
        self.assertEqual(sum(config.run_noddi for config in options),1)

    def test_valid_cached_models_are_not_launched_or_reported_as_fresh_work(self):
        from types import SimpleNamespace
        from microstructure.pipeline import MicrostructureConfig,_run_independent_models
        from shared.pipeline import RawInputBundle
        raw=RawInputBundle(self.root,'fixture',self.root,*[self.root/name for name in ('dwi','bval','bvec','mask')])
        config=MicrostructureConfig('fixture',raw,None,self.root/'outputs',
            run_free_water=True,run_noddi=True,max_parallel_models=2)
        with patch('microstructure.pipeline.nib.load',return_value=SimpleNamespace(shape=(2,2,2,8))), \
             patch('microstructure.pipeline._free_water_outputs_current',return_value=True), \
             patch('microstructure.pipeline._noddi_outputs_current',return_value=True), \
             patch('shared.compute_schedule.run_independent_jobs',return_value={'jobs':[]}) as schedule:
            adjusted,record=_run_independent_models(config,None)
        self.assertEqual(schedule.call_args.args[0],[])
        self.assertEqual(record['jobs'],[])
        self.assertIsNone(adjusted.free_water_n_jobs)
        self.assertFalse(adjusted.force_recompute)

    def test_orchestrator_preserves_overlapping_model_intervals(self):
        from types import SimpleNamespace
        from shared.orchestrator import ProductOrchestratorConfig,_parallel_model_products
        from shared.pipeline import RawInputBundle
        raw=RawInputBundle(self.root,'fixture',self.root,*[self.root/name for name in ('dwi','bval','bvec','mask')])
        rows=[dict(name=name,started_at='2026-10-20T00:00:00+00:00',
                   ended_at='2026-10-20T00:00:02+00:00',runtime_seconds=2.,log=str(self.root/(name+'.log')))
              for name in ('freewater','noddi','dki')]
        result=SimpleNamespace(compute_schedule={'jobs':rows},dki_result=SimpleNamespace(
            outputs={'mk':self.root/'mk.nii.gz'},mkcurve_outputs={},
            manifest_path=self.root/'dki_manifest.json',qc_path=self.root/'dki_qc.json'))
        config=ProductOrchestratorConfig('fixture',self.root,('microstructure',),
            execute=True,resume=False,max_parallel_models=3)
        targets=tuple('microstructure_'+row['name'] for row in rows)
        with patch('shared.pipeline.find_subject_root',return_value=raw), \
             patch('shared.pipeline.resolve_preprocessed_input_bundle',return_value=raw), \
             patch('shared.pipeline.ensure_native_dti_bundle',return_value=None), \
             patch('microstructure.run_microstructure',return_value=result) as compute:
            records=_parallel_model_products(config,targets,None,None)
        self.assertEqual(len(records),3)
        options=compute.call_args.args[0]
        self.assertTrue(options.run_free_water and options.run_noddi and options.run_dki)
        self.assertEqual(options.max_parallel_models,3)
        self.assertTrue(options.force_recompute)
        self.assertEqual(len({record.started_at for record in records.values()}),1)
        self.assertTrue(all(record.runtime_seconds==2. for record in records.values()))
        self.assertTrue(all(record.status=='completed' for record in records.values()))


if __name__=='__main__':
    unittest.main()
