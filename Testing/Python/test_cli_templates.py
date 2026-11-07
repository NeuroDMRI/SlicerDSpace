"""Keep published CLI recipes executable against the current argument contracts."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.extension_runtime.runtime_resources import runtime_files
from scripts.prepare_inputs import build_parser as input_parser
from scripts.run_batch_pipeline import build_parser as batch_parser
from shared.experiments import compile_plan
from shared.run_subject_pipeline_cli import build_parser, resolve_analysis_options
from shared.user_config import explicit_cli_destinations


ROOT = Path(__file__).resolve().parents[2]
GUIDES = [ROOT / 'docs/setup' / name for name in ('CLI_TEMPLATES.md', 'CLI_TEMPLATES.zh-CN.md')]
EXAMPLES = ROOT / 'docs/setup/examples'


def blocks(path, kind):
    text = path.read_text(encoding='utf-8')
    matches = re.findall(r'<!-- cli-' + kind + r': ([\w-]+) -->\s*```\w+\n(.*?)\n```', text, re.S)
    if len(dict(matches)) != len(matches):
        raise AssertionError('Duplicate CLI recipe identifiers in ' + str(path))
    return dict(matches)


def argv(code):
    return shlex.split(code.replace('\\\n', ' '))


def parse_route(code):
    tokens = argv(code)
    if tokens[0] != 'pipeline':
        raise AssertionError('Not a subject pipeline recipe')
    parser = build_parser()
    parser.allow_abbrev = False
    arguments = ['--no-user-config', '--strict-environment',
                 '--subject-root', '/inputs/sub-01', '--subject-id', 'sub-01',
                 '--output-root', '/outputs/' + tokens[1], '--run-name', tokens[1],
                 '--model-assets-dir', '/assets', *tokens[2:]]
    args = parser.parse_args(arguments)
    targets = resolve_analysis_options(args, explicit_dests=explicit_cli_destinations(parser, arguments))
    return args, targets


class CliTemplatesTest(unittest.TestCase):
    def test_languages_publish_identical_recipe_arguments(self):
        english, chinese = (blocks(path, 'template') for path in GUIDES)
        self.assertEqual(len(english), 42)
        self.assertEqual(english.keys(), chinese.keys())
        for name in english:
            with self.subTest(recipe=name):
                self.assertEqual(argv(english[name]), argv(chinese[name]))
        self.assertEqual(blocks(GUIDES[0], 'setup'), blocks(GUIDES[1], 'setup'))

    def test_subject_recipes_resolve_current_parameters_without_execution(self):
        names = set()
        for name, code in blocks(GUIDES[0], 'template').items():
            if not code.startswith('pipeline '):
                continue
            with self.subTest(recipe=name):
                args, targets = parse_route(code)
                self.assertTrue(targets)
                self.assertTrue(args.plan)
                self.assertFalse(args.execute)
                self.assertTrue(args.strict_environment)
                self.assertTrue(args.no_user_config)
                self.assertFalse(args.skip_environment_check)
                self.assertFalse(args.allow_anatomy_space_mismatch)
                self.assertFalse(args.no_resume)
                self.assertNotIn(args.run_name, names)
                names.add(args.run_name)
        self.assertEqual(len(names), 34)

    def test_bundle_routes_keep_source_and_method_independent(self):
        recipes = blocks(GUIDES[0], 'template')
        for name in ('bundle-wmql', 'bundle-wmql-subset', 'bundle-wma', 'bundle-tractcloud', 'bundle-multiple'):
            with self.subTest(recipe=name):
                args, targets = parse_route(recipes[name])
                self.assertEqual(args.bundle_tractography_backend, 'ukf')
                self.assertIn('ukf', args.tractography_backend.split(','))
                self.assertIn('tractography', targets)
        args, targets = parse_route(recipes['bundle-multiple'])
        self.assertEqual(set(args.bundle_backend.split(',')), {'wmql', 'wma', 'tractcloud'})
        self.assertTrue({'bundles', 'wma', 'tractcloud'}.issubset(targets))
        self.assertEqual(args.tractography_backend, 'csd_act,ukf')
        args, _ = parse_route(recipes['bundle-existing'])
        self.assertIsNone(args.bundle_tractography_backend)
        self.assertIsNotNone(args.wma_input_tractography)

    def test_optional_models_and_saf_are_explicit(self):
        recipes = blocks(GUIDES[0], 'template')
        args, targets = parse_route(recipes['micro-dti'])
        self.assertEqual(args.brain_mask_backend, 'provided')
        self.assertFalse(args.run_free_water or args.run_noddi or args.run_dki)
        args, targets = parse_route(recipes['micro-parallel'])
        self.assertTrue(args.run_free_water and args.run_noddi and args.run_dki)
        self.assertEqual(args.max_parallel_models, 2)
        args, _ = parse_route(recipes['micro-mkcurve'])
        self.assertEqual(args.dki_fit_method, 'MKCURVE')
        args, targets = parse_route(recipes['tract-saf'])
        self.assertIn('saf', targets)
        self.assertIn('surface_postprocess', targets)
        self.assertIsNone(args.bundle_tractography_backend)

    def test_input_imports_are_previews_with_explicit_formats(self):
        recipes = blocks(GUIDES[0], 'template')
        for name, source_kind in (('import-dicom', 'dicom'), ('import-nifti', 'nifti'), ('import-nrrd', 'nrrd')):
            with self.subTest(recipe=name):
                tokens = argv(recipes[name])
                self.assertEqual(tokens[1:3], ['--launch', 'PythonSlicer'])
                self.assertTrue(tokens[3].endswith('/scripts/prepare_inputs.py'))
                args = input_parser().parse_args(tokens[4:])
                self.assertTrue(args.dry_run)
                self.assertFalse(args.force)
                self.assertEqual(args.source_kind, source_kind)
                self.assertIsNotNone(args.subject_id)

    def test_cohort_forwards_the_same_dti_contract(self):
        tokens = argv(blocks(GUIDES[0], 'template')['cohort'])
        self.assertTrue(tokens[3].endswith('/scripts/run_batch_pipeline.py'))
        args, forward = batch_parser().parse_known_args(tokens[4:])
        self.assertFalse(args.execute)
        self.assertEqual(args.jobs, 2)
        resolved = build_parser().parse_args(forward)
        self.assertTrue(resolved.no_user_config)
        self.assertTrue(resolved.strict_environment)
        self.assertEqual(resolved.brain_mask_backend, 'provided')
        self.assertEqual(resolve_analysis_options(resolved), ('validation', 'microstructure'))

    def test_experiments_recipe_validates_without_launching_jobs(self):
        from scripts import run_experiments
        tokens = argv(blocks(GUIDES[0], 'template')['experiments'])
        with tempfile.TemporaryDirectory() as directory:
            request = Path(directory) / 'request.json'
            request.write_text('{}', encoding='utf-8')
            arguments = [str(request) if v == '/path/to/edited_experiment_request.json' else v for v in tokens[4:]]
            with patch.object(run_experiments, 'compile_plan', return_value={'status': 'planned'}) as planner, \
                    patch.object(run_experiments, 'run_plan') as runner, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_experiments.main(arguments), 0)
            planner.assert_called_once()
            runner.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec('nibabel') and importlib.util.find_spec('numpy'), 'nibabel/numpy unavailable')
    def test_regional_recipes_use_direct_arguments_and_optional_catalog(self):
        from scripts import analyze_regions
        recipes = blocks(GUIDES[0], 'template')
        for name in ('region-inspect', 'region-run'):
            tokens = argv(recipes[name])
            request = json.loads((EXAMPLES / 'region_request.json').read_text())
            with self.subTest(recipe=name), patch.object(sys, 'argv', [tokens[3], *tokens[4:]]), \
                    patch.object(signal, 'signal'), patch.object(analyze_regions, 'inspect_run', return_value={}) as inspect, \
                    patch.object(analyze_regions, 'write_json') as write, \
                    patch.object(analyze_regions, 'read_json', return_value=request), \
                    patch.object(analyze_regions, 'run_analysis') as run, contextlib.redirect_stdout(io.StringIO()):
                analyze_regions.main()
                if name == 'region-inspect':
                    inspect.assert_called_once()
                    self.assertEqual(inspect.call_args.args[0], Path('/path/to/run_orchestration_manifest.json'))
                    write.assert_not_called()
                    run.assert_not_called()
                else:
                    inspect.assert_not_called()
                    run.assert_called_once()
                    direct = run.call_args.args[0]
                    self.assertEqual(direct['selected_regions'], ['1024', '2024'])
                    self.assertEqual(direct['metrics'], ['fa', 'md'])
                    self.assertEqual(direct['output_dir'], str(Path('/path/to/new_region_results').resolve()))

    def test_report_uses_the_route_specific_manifest(self):
        from scripts import generate_subject_report
        from types import SimpleNamespace
        tokens = argv(blocks(GUIDES[0], 'template')['report'])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()  # F4: macOS /var is a symlink to /private/var
            manifest = root / 'outputs/dti/00_run_metadata/dti_orchestration_manifest.json'
            manifest.parent.mkdir(parents=True)
            manifest.write_text('{}')
            replacements = {'$SUBJECT': str(root / 'subject'), '$OUTPUTS/dti': str(root / 'outputs/dti'),
                            '$OUTPUTS/dti/00_run_metadata/dti_orchestration_manifest.json': str(manifest)}
            arguments = [replacements.get(v, v) for v in tokens[4:]]
            output = SimpleNamespace(**{name: root / name for name in ('report_html', 'summary_md', 'provenance_json', 'qc_summary_json')})
            with patch.object(generate_subject_report, 'write_subject_reports', return_value=output) as writer, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(generate_subject_report.main(arguments), 0)
            self.assertEqual(writer.call_args.kwargs['manifest_path'], manifest)
            self.assertEqual(writer.call_args.kwargs['output_root'], root / 'outputs/dti')

    def test_example_experiments_compile_two_subjects_and_two_variants_without_writes(self):
        request = json.loads((EXAMPLES / 'dti_experiments.json').read_text())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for subject in ('sub-01', 'sub-02'):
                inputs = root / 'study' / subject / 'input'
                inputs.mkdir(parents=True)
                for name in ('data.nii.gz', 'bvals', 'bvecs', 'nodif_brain_mask.nii.gz'):
                    (inputs / name).touch()
            request.update(cohort_root=str(root / 'study'), output_root=str(root / 'results'))
            plan = compile_plan(request, python_executable=sys.executable,
                                pipeline_script=ROOT / 'shared/run_subject_pipeline_cli.py')
            self.assertEqual(len(plan['subjects']), 4)
            self.assertEqual({r['effective_config']['fit_method'] for r in plan['subjects']}, {'wls', 'ls'})
            self.assertEqual(len({r['output_root'] for r in plan['subjects']}), 4)
            self.assertFalse((root / 'results').exists())

    @unittest.skipUnless(importlib.util.find_spec('nibabel') and importlib.util.find_spec('numpy'), 'nibabel/numpy unavailable')
    def test_region_request_example_runs_on_small_qc_signed_maps(self):
        import nibabel as nib
        import numpy as np
        from shared.region_analysis import run_analysis, sha256
        request = json.loads((EXAMPLES / 'region_request.json').read_text())
        self.assertEqual(request['tract_rule'], 'through_any')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atlas, fa, md = (root / name for name in ('atlas.nii.gz', 'fa.nii.gz', 'md.nii.gz'))
            labels = np.full((3, 3, 3), request['selected_labels'][0], dtype=np.int32)
            labels[1:] = request['selected_labels'][1]
            (root / 'input').mkdir()
            nib.save(nib.Nifti1Image(np.ones(labels.shape, dtype=np.uint8), np.eye(4)),
                     root / 'input/nodif_brain_mask.nii.gz')
            for path, data in ((atlas, labels), (fa, np.full(labels.shape, 0.6)), (md, np.full(labels.shape, 0.001))):
                nib.save(nib.Nifti1Image(data, np.eye(4)), path)
            tasks = []
            for name, outputs in (('parcellation.ddparcel', {'wmparc': atlas}), ('microstructure.dti', {'fa': fa, 'md': md})):
                qc = root / (name + '_qc.json')
                paths = {key: str(value) for key, value in outputs.items()}
                qc.write_text(json.dumps(dict(status='completed', automatic_qc_passed=True,
                    source_signatures={key: dict(path=value, sha256=sha256(Path(value))) for key, value in paths.items()})))
                tasks.append(dict(name=name, status='completed', outputs=paths, qc_path=str(qc)))
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps(dict(kind='slicerddsurfer_product_orchestration', status='completed',
                subject_id='fixture', subject_root=str(root), output_root=str(root / 'outputs'), tasks=tasks)))
            request.update(source_manifest=str(manifest), output_dir=str(root / 'regions'))
            run_analysis(request)
            statistics = list((root / 'regions').glob('*.csv'))
            self.assertTrue(statistics)

    def test_guides_and_json_examples_ship_in_runtime(self):
        selected = set(runtime_files(ROOT))
        for path in [*GUIDES, *EXAMPLES.glob('*.json')]:
            self.assertIn(path.relative_to(ROOT).as_posix(), selected)
        self.assertFalse(any(path.startswith('.agents/') for path in selected))

    @unittest.skipUnless(shutil.which('bash'), 'Bash unavailable')
    def test_bash_snippets_have_valid_syntax(self):
        commands = [blocks(GUIDES[0], 'setup')['bash'], *blocks(GUIDES[0], 'template').values()]
        result = subprocess.run(['bash', '-n'], input='\n'.join(commands), text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which('bash') and os.name != 'nt', 'POSIX Bash fixture required')
    def test_bash_wrapper_preserves_spaces_json_and_output_isolation(self):
        setup = blocks(GUIDES[0], 'setup')['bash']
        with tempfile.TemporaryDirectory(prefix='CLI examples ') as directory:
            root = Path(directory)
            launcher = root / 'fake Slicer'
            launcher.write_text('#!' + sys.executable + '\nimport json,os,sys\n'
                'with open(os.environ["DOC_TEST_LOG"],"a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n')
            launcher.chmod(0o700)
            script = setup + '\n' + '\n'.join([
                'export SLICER=' + shlex.quote(str(launcher)),
                'export OUTPUTS=' + shlex.quote(str(root / 'analysis folder')),
                'export SUBJECT=' + shlex.quote(str(root / 'subject folder')),
                'DEPENDENCY_ARGS=(--dependency-config ' + shlex.quote(str(root / 'dependency config.json')) + ')',
                blocks(GUIDES[0], 'template')['tract-ukf'],
                blocks(GUIDES[0], 'template')['micro-dti'],
            ])
            log = root / 'argv.jsonl'
            env = dict(os.environ, DOC_TEST_LOG=str(log))
            result = subprocess.run(['bash', '-e'], input=script, text=True, capture_output=True, env=env)
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [json.loads(line) for line in log.read_text().splitlines()]
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][calls[0].index('--ukf-parameters') + 1], '{"numThreads":8}')
            for call, run in zip(calls, ('tracks_ukf', 'dti')):
                self.assertEqual(call[call.index('--subject-root') + 1], str(root / 'subject folder'))
                self.assertEqual(call[call.index('--output-root') + 1], str(root / 'analysis folder' / run))
                self.assertEqual(call[call.index('--dependency-config') + 1], str(root / 'dependency config.json'))
                self.assertIn('--plan', call)
                self.assertNotIn('--execute', call)

    def test_powershell_wrapper_syntax_when_available(self):
        pwsh = os.environ.get('SLICERDSPACE_TEST_PWSH') or shutil.which('pwsh')
        if not pwsh:
            self.skipTest('PowerShell not installed')
        code = blocks(GUIDES[0], 'setup')['powershell']
        command = '$t=$null; $e=$null; [void][System.Management.Automation.Language.Parser]::ParseInput($env:DOC_TEST_PS,[ref]$t,[ref]$e); if($e.Count){$e | Out-String | Write-Error; exit 1}'
        result = subprocess.run([pwsh, '-NoProfile', '-NonInteractive', '-Command', command],
                                capture_output=True, text=True, env=dict(os.environ, DOC_TEST_PS=code))
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
