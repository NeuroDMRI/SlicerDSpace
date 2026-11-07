"""T3 scheduler contracts: S1.1 requests, S1.3 thread precedence, S2 RAM fit, S3 durations and backfill."""

import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from shared import cohort_defaults
from shared.cohort_planning import default_duration_seconds, image_size, plan_cohort
from shared.cohort_policy import choose_ready, fits, reservation_time
from shared.cohort_profile import record_history
from shared.cohort_schedule import ENV, create_schedule, stage_admission

GIB = 1024**3
ABSENT_SIGNAL = math.prod(image_size('/absent')['shape'][:3]) * image_size('/absent')['shape'][3] * 8 / GIB


def subjects(count=1):
    return [SimpleNamespace(subject_id=f'S{i}', subject_root=Path('/absent') / f'S{i}') for i in range(count)]


def adaptive_profile(arguments=('--targets', 'surface'), *, cpus=32, **kwargs):
    plan = plan_cohort(subjects(), list(arguments), adaptive=True, available=(cpus, 256 * GIB), **kwargs)
    return plan, plan['subject_profiles']['S0']


class StageRequestTableTests(unittest.TestCase):
    """S1.1: adaptive CPU requests come from cohort_defaults.STAGE_CPU_REQUESTS."""

    def test_table_sets_the_adaptive_request_and_unlisted_stages_keep_theirs(self):
        with patch.dict(cohort_defaults.STAGE_CPU_REQUESTS, {'native_dti': 8, 'surface': 2}, clear=True):
            plan, profile = adaptive_profile()
        self.assertEqual((profile['native_dti']['cpu'], profile['surface']['cpu']), (8, 2))
        self.assertEqual(profile['brain_mask']['cpu'], 4)
        self.assertEqual(profile['microstructure_dki']['cpu'], 1)
        self.assertEqual(plan['stage_cpus'], 4)

    def test_table_value_is_capped_at_the_cpu_budget(self):
        with patch.dict(cohort_defaults.STAGE_CPU_REQUESTS, {'native_dti': 16}, clear=True):
            _, profile = adaptive_profile(cpus=8)
        self.assertEqual(profile['native_dti']['cpu'], 6)

    def test_batch_stage_cpus_overrides_the_table(self):
        with patch.dict(cohort_defaults.STAGE_CPU_REQUESTS, {'native_dti': 8, 'surface': 2}, clear=True):
            _, profile = adaptive_profile(stage_cpus=3)
        self.assertEqual({row['cpu'] for name, row in profile.items() if name in ('native_dti', 'surface', 'brain_mask')}, {3})

    def test_explicit_thread_flag_overrides_the_table(self):
        with patch.dict(cohort_defaults.STAGE_CPU_REQUESTS, {'surface': 2}, clear=True):
            _, profile = adaptive_profile(('--targets', 'surface', '--surface-cpu-threads', '6'))
        self.assertEqual(profile['surface']['cpu'], 6)

    def test_table_is_ignored_outside_adaptive_mode(self):
        with patch.dict(cohort_defaults.STAGE_CPU_REQUESTS, {'native_dti': 8}, clear=True):
            plan = plan_cohort(subjects(), ['--targets', 'surface'], available=(32, 256 * GIB))
        self.assertEqual(plan['subject_profiles']['S0']['native_dti']['cpu'], plan['stage_cpus'])

    def test_shipped_table_holds_the_m1_choices(self):
        """M1 fill (m1_bcdbc95): the largest candidate with efficiency >= 0.75 per bracketed stage."""
        expected = {'native_dti': 16, 'registered_dti': 2, 'segmentation': 2, 'parcellation': 2, 'surface': 2,
                    'microstructure_noddi': 8}
        self.assertEqual(cohort_defaults.STAGE_CPU_REQUESTS, expected)
        _, profile = adaptive_profile(cpus=32)
        self.assertEqual({stage: profile[stage]['cpu'] for stage in expected}, expected)

    def test_m1_fill_explicit_thread_flag_above_four_sets_only_its_own_stages(self):
        """Backlog (Codex P1 = Opus N1, routed to the M1 fill): --mrtrix-nthreads 8 sets csd_act and connectome
        to 8; the bracketed stages keep their M1 table values."""
        arguments = ('--targets', 'tractography,connectome', '--tractography-backend', 'csd_act',
                     '--mrtrix-nthreads', '8')
        _, profile = adaptive_profile(arguments, cpus=32)
        self.assertEqual((profile['csd_act']['cpu'], profile['connectome']['cpu']), (8, 8))
        expected = {'native_dti': 16, 'registered_dti': 2, 'segmentation': 2, 'parcellation': 2, 'surface': 2,
                    'microstructure_noddi': 8}
        self.assertEqual({stage: profile[stage]['cpu'] for stage in expected}, expected)


class ThreadPrecedenceTests(unittest.TestCase):
    """S1.3: explicit value > SLICERDSPACE_STAGE_CPUS > built-in default."""

    FAKE_PIPELINE_ENV = '''import argparse, json, os, sys
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--subject-root")
p.add_argument("--subject-id")
p.add_argument("--output-root")
p.add_argument("--event-dir", type=Path)
a, extra = p.parse_known_args()
dest = Path(a.output_root)
dest.mkdir(parents=True)
keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "SLICERDSPACE_STAGE_CPUS")
(dest / "receipt.json").write_text(json.dumps(dict(extra=extra, env={k: os.environ.get(k) for k in keys})))
'''

    def run_batch(self, scheduler, *extra):
        from test_run_batch_pipeline import RunBatchPipelineCliTest
        case = RunBatchPipelineCliTest()
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(RunBatchPipelineCliTest, 'FAKE_PIPELINE', self.FAKE_PIPELINE_ENV):
            root = Path(tmp)
            flags = ['--scheduler', scheduler, '--batch-cpus', '4', '--batch-memory-gb', '8',
                     '--batch-stage-memory-gb', '1']
            status, manifest = case.run_fake_batch(root, *flags, *extra, jobs=1)
            receipt = json.loads((root / 'outputs/sub-01/receipt.json').read_text())
        self.assertEqual(status, 0)
        return manifest['planned_commands'][0], receipt

    THREAD_FLAGS = ('--surface-cpu-threads', '--mrtrix-nthreads', '--preprocessing-eddy-threads')

    def test_adaptive_driver_does_not_append_thread_flags(self):
        command, receipt = self.run_batch('adaptive')
        for flag in self.THREAD_FLAGS:
            self.assertNotIn(flag, command)
            self.assertNotIn(flag, receipt['extra'])

    def test_adaptive_driver_keeps_the_users_thread_flags(self):
        command, _ = self.run_batch('adaptive', '--mrtrix-nthreads', '2')
        self.assertEqual(command.count('--mrtrix-nthreads'), 1)
        self.assertEqual(command[command.index('--mrtrix-nthreads') + 1], '2')
        self.assertNotIn('--surface-cpu-threads', command)

    def test_resources_mode_still_appends_the_plan_stage_cpus(self):
        command, _ = self.run_batch('resources', '--batch-stage-cpus', '2')
        for flag in self.THREAD_FLAGS:
            self.assertEqual(command[command.index(flag) + 1], '2')

    def test_subject_mode_has_no_resource_plan_flags_or_thread_env(self):
        from test_run_batch_pipeline import RunBatchPipelineCliTest
        case = RunBatchPipelineCliTest()
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(RunBatchPipelineCliTest, 'FAKE_PIPELINE', self.FAKE_PIPELINE_ENV), \
             patch.dict(os.environ, {'OMP_NUM_THREADS': '7'}):
            status, manifest = case.run_fake_batch(Path(tmp), jobs=1)
            receipt = json.loads((Path(tmp) / 'outputs/sub-01/receipt.json').read_text())
        self.assertEqual(status, 0)
        for flag in self.THREAD_FLAGS:
            self.assertNotIn(flag, manifest['planned_commands'][0])
        self.assertEqual(receipt['env']['OMP_NUM_THREADS'], '7')

    def test_adaptive_subject_process_starts_with_one_blas_thread(self):
        _, receipt = self.run_batch('adaptive')
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
            self.assertEqual(receipt['env'][key], '1')
        self.assertIsNotNone(receipt['env']['SLICERDSPACE_STAGE_CPUS'])

    def test_resources_subject_process_keeps_plan_stage_cpus_in_blas_variables(self):
        _, receipt = self.run_batch('resources', '--batch-stage-cpus', '2')
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
            self.assertEqual(receipt['env'][key], '2')

    def test_reserved_stage_cpus_reads_only_a_positive_integer(self):
        from shared.numeric_threads import reserved_stage_cpus
        for value, expected in (('3', 3), (' 5 ', 5), ('0', None), ('-1', None), ('x', None), ('', None)):
            with patch.dict(os.environ, {'SLICERDSPACE_STAGE_CPUS': value}):
                self.assertEqual(reserved_stage_cpus(), expected, value)
        with patch.dict(os.environ):
            os.environ.pop('SLICERDSPACE_STAGE_CPUS', None)
            self.assertIsNone(reserved_stage_cpus())

    def test_mrtrix_csd_commands(self):
        from test_tractography_csd_seeding import _config
        from tractography.csd import build_csd_act_commands, reproducibility_record, tckgen_thread_count

        def thread_values(commands):
            return {Path(command[0]).name: command[command.index('-nthreads') + 1] if '-nthreads' in command else None
                    for command in commands}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ):
                os.environ.pop('SLICERDSPACE_STAGE_CPUS', None)
                free = thread_values(build_csd_act_commands(_config(root)))
                self.assertIsNone(free['tckgen'])
                with patch.dict(os.environ, {'SLICERDSPACE_STAGE_CPUS': '3'}):
                    reserved = thread_values(build_csd_act_commands(_config(root)))
                    self.assertEqual(reserved['tckgen'], '3')
                    self.assertEqual(reserved['tcksift2'], '3')
                    self.assertTrue(all(value == '3' for value in reserved.values()), reserved)
                    explicit = thread_values(build_csd_act_commands(_config(root, nthreads=5)))
                    self.assertTrue(all(value == '5' for value in explicit.values()), explicit)
                    seeded = _config(root, random_seed=7)
                    self.assertEqual(tckgen_thread_count(seeded), 0)
                    self.assertEqual(thread_values(build_csd_act_commands(seeded))['tckgen'], '0')
                    self.assertIsNone(reproducibility_record(_config(root))['nthreads_requested'])
                    self.assertEqual(reproducibility_record(_config(root))['tcksift2_nthreads'], 3)

    def test_connectome_command(self):
        from connectome.pipeline import ConnectomeConfig, build_tck2connectome_command
        config = ConnectomeConfig('S', Path('/t.tck'), Path('/p.nii.gz'))
        with patch('connectome.pipeline._resolve_command', return_value='tck2connectome'):
            with patch.dict(os.environ):
                os.environ.pop('SLICERDSPACE_STAGE_CPUS', None)
                self.assertNotIn('-nthreads', build_tck2connectome_command(config, Path('/m.csv')))
                os.environ['SLICERDSPACE_STAGE_CPUS'] = '2'
                command = build_tck2connectome_command(config, Path('/m.csv'))
                self.assertEqual(command[command.index('-nthreads') + 1], '2')
                self.assertEqual(command[-1], '-force')

    def test_saf_tckgen_options(self):
        from tractography.saf import SAFConfig, _tckgen_command_options, _effective_tckgen_options
        base = dict(subject_id='S', input_root=Path('.'), fs_dir=Path('.'), fod_path=Path('.'))
        with patch.dict(os.environ):
            os.environ.pop('SLICERDSPACE_STAGE_CPUS', None)
            self.assertEqual(_tckgen_command_options(SAFConfig(**base, tckgen_options='-select 9')), '-select 9')
            os.environ['SLICERDSPACE_STAGE_CPUS'] = '4'
            self.assertEqual(_tckgen_command_options(SAFConfig(**base, tckgen_options='-select 9')), '-select 9 -nthreads 4')
            self.assertEqual(_tckgen_command_options(SAFConfig(**base)), '-nthreads 4')
            explicit = SAFConfig(**base, tckgen_options='-nthreads 1 -select 9')
            self.assertEqual(_tckgen_command_options(explicit), '-nthreads 1 -select 9')
            # The recorded options do not depend on the admitted CPU count.
            self.assertEqual(_effective_tckgen_options(SAFConfig(**base, tckgen_options='-select 9')), '-select 9')

    def test_eddy_threads(self):
        from preprocessing.pipeline import DwiPreprocessingConfig, _resolved_eddy_options, _resolved_eddy_threads
        with patch.dict(os.environ):
            os.environ.pop('SLICERDSPACE_STAGE_CPUS', None)
            default = _resolved_eddy_threads(DwiPreprocessingConfig('S', Path('.')))
            self.assertGreaterEqual(default, 1)
            os.environ['SLICERDSPACE_STAGE_CPUS'] = '3'
            self.assertEqual(_resolved_eddy_threads(DwiPreprocessingConfig('S', Path('.'))), 3)
            self.assertIn('--nthr=3', _resolved_eddy_options(DwiPreprocessingConfig('S', Path('.'))))
            self.assertEqual(_resolved_eddy_threads(DwiPreprocessingConfig('S', Path('.'), eddy_threads=2)), 2)

    def test_ddsurfer_threads(self):
        from anatomy.surface.pipeline import SurfaceConfig, resolved_cpu_threads, _surface_records
        config = SurfaceConfig('S', Path('/s'), Path('/o'))
        explicit = SurfaceConfig('S', Path('/s'), Path('/o'), cpu_threads=2)
        with patch.dict(os.environ):
            os.environ.pop('SLICERDSPACE_STAGE_CPUS', None)
            self.assertEqual(resolved_cpu_threads(config), 4)
            os.environ['SLICERDSPACE_STAGE_CPUS'] = '6'
            self.assertEqual(resolved_cpu_threads(config), 6)
            self.assertEqual(resolved_cpu_threads(explicit), 2)
            manifest, _ = _surface_records(config, dict(upstream_commit='x'), Path('/o'))
            self.assertEqual(manifest['cpu_threads'], 6)

    def test_surface_cli_and_orchestrator_default_to_the_reservation(self):
        from shared.run_subject_pipeline_cli import build_parser
        from shared.orchestrator import ProductOrchestratorConfig
        self.assertIsNone(build_parser().parse_args([]).surface_cpu_threads)
        self.assertIsNone(ProductOrchestratorConfig(subject_id='S', subject_root=Path('/s'), targets=('surface',)).surface_cpu_threads)
        self.assertEqual(build_parser().parse_args(['--surface-cpu-threads', '2']).surface_cpu_threads, 2)

    def test_surface_resume_check_ignores_the_recorded_threads_only_inside_a_cohort(self):
        from shared import orchestrator
        manifest = dict(upstream_commit='c', status='completed', input_mode='registered', version=3,
                        subject_id='S', subject_root=str(Path('/s').resolve()), precision='fp32',
                        cpu_threads=2, max_gpu_gib=24.)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 's'
            surface_root = root / 'surface'
            surface_root.mkdir(parents=True)
            (surface_root / 'surface_manifest.json').write_text(json.dumps(dict(manifest, subject_root=str(root.resolve()))))
            (surface_root / 'surface_qc.json').write_text(json.dumps(dict(automatic_qc_passed=True)))
            layout = SimpleNamespace(anatomy_input_dir=lambda name: surface_root)
            seen = []

            def after_threads(*args, **kwargs):
                seen.append('past-thread-check')
                return False
            for cohort, expected in (({}, False), ({ENV: '/x'}, True)):
                seen.clear()
                with patch.dict(os.environ, cohort), \
                     patch.object(orchestrator, 'subject_layout', return_value=layout), \
                     patch('anatomy.surface.backend_bridge.upstream_record', return_value=dict(commit='c')), \
                     patch.object(orchestrator, 'interpreter_path', side_effect=after_threads, create=True):
                    config = orchestrator.ProductOrchestratorConfig(
                        subject_id='S', subject_root=root, targets=('surface',), surface_python_bin=Path('/py'))
                    orchestrator._surface_complete(root, 'S', config=config)
                self.assertEqual(bool(seen), expected, cohort)


class TorchThreadTests(unittest.TestCase):
    """S1.3: an already imported torch follows the reservation, then is restored."""

    def fake_torch(self):
        state = dict(threads=7, calls=[])
        module = types.ModuleType('torch')
        module.get_num_threads = lambda: state['threads']

        def set_num_threads(count):
            state['calls'].append(count)
            state['threads'] = count
        module.set_num_threads = set_num_threads
        return module, state

    def run_stage(self, policy, torch_module):
        with tempfile.TemporaryDirectory() as tmp:
            folder = create_schedule(Path(tmp) / 'schedule', cpu_budget=4, memory_gb=4, stage_cpus=4, stage_memory_gb=4,
                                     policy=policy, subject_profiles={'A': {'fit': dict(cpu=2, memory_gb=1)}})
            modules = {'torch': torch_module} if torch_module is not None else {}
            with patch.dict(os.environ, {ENV: str(folder), 'SLICERDSPACE_COHORT_SUBJECT': 'A'}), \
                 patch.dict(sys.modules, modules):
                if torch_module is None:
                    sys.modules.pop('torch', None)
                with stage_admission('fit'):
                    inside = torch_module.get_num_threads() if torch_module is not None else None
        return inside

    def test_adaptive_admission_sets_and_restores_torch_threads(self):
        torch, state = self.fake_torch()
        self.assertEqual(self.run_stage('fair_backfill', torch), 2)
        self.assertEqual(state['threads'], 7)
        self.assertEqual(state['calls'], [2, 7])

    def test_torch_threads_are_restored_after_a_failed_stage(self):
        torch, state = self.fake_torch()
        with tempfile.TemporaryDirectory() as tmp:
            folder = create_schedule(Path(tmp) / 'schedule', cpu_budget=4, memory_gb=4, stage_cpus=4, stage_memory_gb=4,
                                     policy='fair_backfill', subject_profiles={'A': {'fit': dict(cpu=2, memory_gb=1)}})
            with patch.dict(os.environ, {ENV: str(folder), 'SLICERDSPACE_COHORT_SUBJECT': 'A'}), \
                 patch.dict(sys.modules, {'torch': torch}):
                with self.assertRaises(ValueError):
                    with stage_admission('fit'):
                        raise ValueError('scientific error')
        self.assertEqual(state['threads'], 7)

    def test_resources_mode_leaves_torch_alone(self):
        torch, state = self.fake_torch()
        self.assertEqual(self.run_stage('first_fit', torch), 7)
        self.assertEqual(state['calls'], [])

    def test_torch_is_not_imported_by_the_scheduler(self):
        self.assertIsNone(self.run_stage('fair_backfill', None))


class RamFitTests(unittest.TestCase):
    """S2: 1 + 1.25 * P * (signal / signal_ref) GiB, rounded up to 0.25 GiB, for stages in RAM_FIT."""

    def memory(self, table, stage='native_dti', **kwargs):
        with patch.dict(cohort_defaults.RAM_FIT, table, clear=True):
            _, profile = adaptive_profile(**kwargs)
        return profile[stage]['memory_gb']

    def test_formula_at_the_reference_signal(self):
        self.assertEqual(self.memory({'native_dti': (8.0, ABSENT_SIGNAL)}), 11.0)

    def test_formula_scales_with_signal_and_rounds_up_to_a_quarter(self):
        # 1 + 1.25 * 1.0 * 1.1 = 2.375 -> 2.5
        self.assertEqual(self.memory({'native_dti': (1.0, ABSENT_SIGNAL / 1.1)}), 2.5)
        # 1 + 1.25 * 2.0 * 0.5 = 2.25 exactly
        self.assertEqual(self.memory({'native_dti': (2.0, ABSENT_SIGNAL * 2)}), 2.25)

    def test_stage_outside_the_table_keeps_the_header_heuristic(self):
        # Baseline with an empty table (the shipped table holds M1 values since the M1 fill).
        self.assertEqual(self.memory({'native_dti': (8.0, ABSENT_SIGNAL)}, 'surface'), self.memory({}, 'surface'))

    def test_invalid_entry_falls_back_to_the_header_heuristic(self):
        self.assertEqual(self.memory({'native_dti': (8.0, 0)}), self.memory({}))

    def test_fit_is_not_applied_outside_adaptive_mode(self):
        with patch.dict(cohort_defaults.RAM_FIT, {'native_dti': (80.0, ABSENT_SIGNAL)}, clear=True):
            fitted = plan_cohort(subjects(), ['--targets', 'surface'], available=(32, 512 * GIB))
        plain = plan_cohort(subjects(), ['--targets', 'surface'], available=(32, 512 * GIB))
        self.assertEqual(fitted['subject_profiles'], plain['subject_profiles'])

    def test_stage_memory_override_wins(self):
        self.assertEqual(self.memory({'native_dti': (8.0, ABSENT_SIGNAL)}, stage_memory_gb=3), 3)

    def test_history_only_raises_the_fitted_value(self):
        subject = subjects()
        arguments = ['--targets', 'surface']
        with tempfile.TemporaryDirectory() as tmp, patch.dict(cohort_defaults.RAM_FIT, {'native_dti': (8.0, ABSENT_SIGNAL)}, clear=True):
            path = Path(tmp) / 'history.json'
            plan = plan_cohort(subject, arguments, adaptive=True, available=(32, 256 * GIB), history_path=path)
            key = plan['subject_profiles']['S0']['native_dti']['profile_key']
            self.assertEqual(plan['subject_profiles']['S0']['native_dti']['memory_gb'], 11.0)
            for gib, expected in ((.5, 11.0), (20, 26.0)):
                record_history(path, key, dict(outcome='completed', samples=3,
                                               peak_tree_rss_bytes=gib * GIB, execution_seconds=10))
                updated = plan_cohort(subject, arguments, adaptive=True, available=(32, 256 * GIB), history_path=path)
                self.assertEqual(updated['subject_profiles']['S0']['native_dti']['memory_gb'], expected)


class DurationDefaultTests(unittest.TestCase):
    """S3: default duration t_p * c_p / min(n, c_p) * (signal / signal_ref)."""

    TABLE = {'native_dti': dict(signal_ref=10.0, probes={2: (1000., 2.0), 4: (600., 3.5), 8: (400., 6.0)})}

    def test_formula_and_choice_of_p(self):
        # n -> p: 1 -> 2, 2 -> 2, 3 -> 4, 4 -> 4, 8 -> 8, 16 -> 8 (largest probed count)
        expected = {1: 1000 * 2.0 / 1, 2: 1000 * 2.0 / 2, 3: 600 * 3.5 / 3, 4: 600 * 3.5 / 3.5,
                    8: 400 * 6.0 / 6.0, 16: 400 * 6.0 / 6.0}
        for n, seconds in expected.items():
            self.assertAlmostEqual(default_duration_seconds('native_dti', n, 10.0, self.TABLE), seconds, msg=str(n))

    def test_signal_scales_the_estimate(self):
        self.assertAlmostEqual(default_duration_seconds('native_dti', 4, 5.0, self.TABLE), 300.0)

    def test_unmeasured_or_invalid_stage_is_unknown(self):
        self.assertEqual(default_duration_seconds('surface', 4, 10.0, self.TABLE), 0)
        self.assertEqual(default_duration_seconds('native_dti', 4, 10.0, {}), 0)
        for bad in (dict(signal_ref=0, probes={4: (1., 1.)}), dict(signal_ref=1, probes={}),
                    dict(signal_ref=1, probes={4: (0., 1.)})):
            self.assertEqual(default_duration_seconds('native_dti', 4, 10.0, {'native_dti': bad}), 0)

    def test_planner_uses_the_request_signal_and_the_adaptive_mode(self):
        table = {'native_dti': dict(signal_ref=ABSENT_SIGNAL * 2, probes={4: (600., 3.5)})}
        with patch.dict(cohort_defaults.DURATION_DEFAULTS, table, clear=True):
            _, profile = adaptive_profile()
            plan = plan_cohort(subjects(), ['--targets', 'surface'], available=(32, 256 * GIB))
        self.assertAlmostEqual(profile['native_dti']['duration_seconds'], 300.0)
        self.assertEqual(profile['surface']['duration_seconds'], 0)
        self.assertNotIn('duration_seconds', plan['subject_profiles']['S0']['native_dti'])

    def test_planner_uses_the_final_request_for_n(self):
        table = {'native_dti': dict(signal_ref=ABSENT_SIGNAL, probes={2: (1000., 2.0), 8: (400., 6.0)})}
        with patch.dict(cohort_defaults.DURATION_DEFAULTS, table, clear=True), \
             patch.dict(cohort_defaults.STAGE_CPU_REQUESTS, {'native_dti': 8}, clear=True):
            _, profile = adaptive_profile()
            _, forced = adaptive_profile(stage_cpus=2)
        self.assertAlmostEqual(profile['native_dti']['duration_seconds'], 400.0)
        self.assertAlmostEqual(forced['native_dti']['duration_seconds'], 1000.0)

    def test_m1_fill_shipped_tables_give_ram_and_duration_for_the_measured_stages(self):
        """M1 fill: the shipped RAM_FIT and DURATION_DEFAULTS reach the plan for a measured stage
        (native_dti at its table request) and leave an unmeasured stage unknown."""
        self.assertEqual(set(cohort_defaults.RAM_FIT), set(cohort_defaults.DURATION_DEFAULTS))
        self.assertIn('native_dti', cohort_defaults.RAM_FIT)
        _, profile = adaptive_profile(cpus=32)
        ratio = ABSENT_SIGNAL / cohort_defaults.SIGNAL_REF
        peak, _ = cohort_defaults.RAM_FIT['native_dti']
        self.assertEqual(profile['native_dti']['memory_gb'], math.ceil((1 + 1.25 * peak * ratio) * 4) / 4)
        t16, c16 = cohort_defaults.DURATION_DEFAULTS['native_dti']['probes'][16]
        self.assertAlmostEqual(profile['native_dti']['duration_seconds'], t16 * c16 / min(16, c16) * ratio)
        self.assertNotIn('brain_mask', cohort_defaults.DURATION_DEFAULTS)
        self.assertEqual(profile['brain_mask']['duration_seconds'], 0)

    def test_history_wins_over_the_default(self):
        table = {'native_dti': dict(signal_ref=ABSENT_SIGNAL, probes={4: (600., 3.5)})}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(cohort_defaults.DURATION_DEFAULTS, table, clear=True):
            path = Path(tmp) / 'history.json'
            plan = plan_cohort(subjects(), ['--targets', 'surface'], adaptive=True, available=(32, 256 * GIB), history_path=path)
            key = plan['subject_profiles']['S0']['native_dti']['profile_key']
            self.assertAlmostEqual(plan['subject_profiles']['S0']['native_dti']['duration_seconds'], 600.0)
            record_history(path, key, dict(outcome='completed', samples=3, peak_tree_rss_bytes=GIB, execution_seconds=40))
            updated = plan_cohort(subjects(), ['--targets', 'surface'], adaptive=True, available=(32, 256 * GIB), history_path=path)
        self.assertAlmostEqual(updated['subject_profiles']['S0']['native_dti']['duration_seconds'], 50.0)


def task(name, cpu, duration=0, **extra):
    return dict(id=name, cpu=cpu, memory_gb=1, gpu=0, io=0, duration_seconds=duration, queued_at=0, **extra)


class BarrierRemovalTests(unittest.TestCase):
    limits = dict(cpu=8, memory_gb=16, gpu=1, io=2)

    def test_old_ageing_barrier_no_longer_blocks_safe_backfill(self):
        head, small = task('head', 8), task('small', 2, 5)
        # Waited 300 s: the old rule (120 s) stopped all backfill.
        self.assertEqual(choose_ready([head, small], [task('running', 4, 30, started_at=290)],
                                      self.limits, 300)['id'], 'small')
        active = [task('running', 4, 30, started_at=90)]
        # Overtaken 8 times: the old rule (8 bypasses) stopped all backfill.
        self.assertEqual(choose_ready([dict(head, bypasses=8), small], active, self.limits, 100)['id'], 'small')

    def test_unsafe_backfill_is_still_refused_after_a_long_wait(self):
        active = [task('running', 4, 30, started_at=290)]
        head, long = task('head', 8), task('long', 2, 300)
        self.assertIsNone(choose_ready([head, long], active, self.limits, 300))

    def test_overrun_row_never_releases_so_only_head_compatible_work_backfills(self):
        active = [task('overrun', 4, 10, started_at=0)]
        head = task('head', 8)
        self.assertEqual(reservation_time(head, active, self.limits, 500), math.inf)
        self.assertIsNone(choose_ready([head, task('small', 2, 5)], active, self.limits, 500))

    def test_policy_has_no_ageing_parameters(self):
        with self.assertRaises(TypeError):
            choose_ready([task('head', 1)], [], self.limits, 0, max_wait=1.)


def simulate(rows, limits, ready_order):
    """Ideal event model: admission is evaluated at each release time; exact durations; no arrivals."""
    waiting, active, now = [dict(row) for row in rows], [], 0
    reservation, started, overtaken = {}, {}, 0
    while waiting:
        active = [row for row in active if row['started_at'] + row['duration_seconds'] > now]
        while waiting:
            head = waiting[0]
            reservation.setdefault(head['id'], reservation_time(head, active, limits, now))
            chosen = choose_ready(waiting, active, limits, now, ready_order=ready_order)
            if chosen is None:
                break
            if chosen is not head:
                overtaken += 1
            waiting.remove(chosen)
            chosen['started_at'] = now
            started[chosen['id']] = now
            active.append(chosen)
        if waiting:
            if not active:
                raise AssertionError('deadlock: nothing active and the head does not start')
            now = min(row['started_at'] + row['duration_seconds'] for row in active)
    return reservation, started, overtaken


class ReservationInvariantTests(unittest.TestCase):
    """S3 invariant: with exact durations and no new arrivals, no head starts after its reservation."""

    limits = dict(cpu=8, memory_gb=16, gpu=1, io=2)

    def workload(self, rng, count=30):
        return [dict(id=f'r{i}', cpu=rng.randint(1, 8), memory_gb=rng.randint(1, 10), gpu=int(rng.random() < .25),
                     io=int(rng.random() < .25), duration_seconds=rng.randint(1, 60), queued_at=0)
                for i in range(count)]

    def test_no_head_starts_later_than_its_reservation(self):
        rng = random.Random(20261107)
        overtakes = 0
        for ready_order in ('fifo', 'mem_thread'):
            for seed in range(25):
                rows = self.workload(rng)
                reservation, started, overtaken = simulate(rows, self.limits, ready_order)
                overtakes += overtaken
                self.assertEqual(set(started), {row['id'] for row in rows})
                for name, bound in reservation.items():
                    self.assertLessEqual(started[name], bound, f'{ready_order} seed {seed}: {name}')
        # The workloads must actually exercise backfill, or the invariant would hold trivially.
        self.assertGreater(overtakes, 50)

    def test_every_row_eventually_starts_without_a_barrier(self):
        rng = random.Random(7)
        rows = self.workload(rng, 40)
        _, started, _ = simulate(rows, self.limits, 'mem_thread')
        self.assertEqual(len(started), 40)

    def test_fits_helper_is_consistent_with_the_simulation(self):
        row = task('x', 8)
        self.assertTrue(fits(row, [], self.limits))
        self.assertFalse(fits(row, [task('y', 1)], self.limits))


if __name__ == '__main__':
    unittest.main()
