"""B9: pass and fail rule of each per-feature check, with mocked subprocess and a fake torch."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from shared import setup_selftest as selftest

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'


def result(code=0, out='', err=''):
    return SimpleNamespace(returncode=code, stdout=out, stderr=err)


class ToolFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.mrtrix, self.fsl = self.root / 'mrtrix/bin', self.root / 'fsl/bin'
        self.mrtrix.mkdir(parents=True)
        self.fsl.mkdir(parents=True)

    def runner(self, answers):
        """answers: tool name -> result; every call is recorded as (name, args, env)."""
        calls = []

        def run(argv, **kwargs):
            name = Path(argv[0]).name
            calls.append((name, list(argv[1:]), kwargs.get('env')))
            answer = answers[name]
            if isinstance(answer, Exception):
                raise answer
            return answer
        return patch.object(selftest.subprocess, 'run', side_effect=run), calls


class MrtrixCheckTests(ToolFixture):
    def good(self):
        return {'mrconvert': result(0, '== mrconvert 3.0.8 ==\n'),
                'dwi2response': result(0, 'dwi2response: usage ...'),
                'dwifslpreproc': result(0, 'dwifslpreproc: usage ...')}

    def test_all_three_commands_run_through_the_managed_environment(self):
        patcher, calls = self.runner(self.good())
        with patcher:
            checks = selftest.check_mrtrix(self.mrtrix, self.fsl, shim_root=self.root / 'shims')
        self.assertEqual(checks, ['mrtrix:mrconvert_version', 'mrtrix:dwi2response_help', 'mrtrix:dwifslpreproc_help'])
        self.assertEqual([(name, args) for name, args, _ in calls],
                         [('mrconvert', ['--version']), ('dwi2response', ['--help']), ('dwifslpreproc', ['--help'])])
        for _, _, env in calls:  # the B3 environment, not the bare process environment
            self.assertEqual(env['FSLOUTPUTTYPE'], 'NIFTI_GZ')
            self.assertIn(str(self.mrtrix), env['PATH'].split(os.pathsep))

    def test_wrong_version_fails(self):
        for text in ('== mrconvert 3.0.7 ==', '== mrconvert 13.0.8 ==', ''):
            patcher, _ = self.runner(dict(self.good(), mrconvert=result(0, text)))
            with patcher, self.assertRaisesRegex(RuntimeError, 'does not report 3.0.8'):
                selftest.check_mrtrix(self.mrtrix, shim_root=self.root / 'shims')

    def test_a_script_that_cannot_find_python_fails(self):
        patcher, _ = self.runner(dict(self.good(), dwifslpreproc=result(127, '', 'env: python: No such file')))
        with patcher, self.assertRaisesRegex(RuntimeError, r'dwifslpreproc --help failed \(exit 127\)'):
            selftest.check_mrtrix(self.mrtrix, shim_root=self.root / 'shims')

    def test_help_text_without_the_tool_name_fails(self):
        patcher, _ = self.runner(dict(self.good(), dwi2response=result(0, 'something else')))
        with patcher, self.assertRaisesRegex(RuntimeError, 'dwi2response --help failed'):
            selftest.check_mrtrix(self.mrtrix, shim_root=self.root / 'shims')

    def test_a_missing_binary_is_reported_not_raised_as_oserror(self):
        patcher, _ = self.runner(dict(self.good(), mrconvert=FileNotFoundError('no mrconvert')))
        with patcher, self.assertRaisesRegex(RuntimeError, 'mrconvert could not run'):
            selftest.check_mrtrix(self.mrtrix, shim_root=self.root / 'shims')


class FslCheckTests(ToolFixture):
    def good(self):
        return {name: result(1, '', f'{name}\nUsage: {name} [options]') for name in ('eddy', 'eddy_cpu', 'topup')}

    def test_exit_1_with_usage_text_passes_for_all_three(self):
        patcher, calls = self.runner(self.good())
        with patcher:
            checks = selftest.check_fsl(self.fsl, shim_root=self.root / 'shims')
        self.assertEqual(checks, ['fsl:eddy_help', 'fsl:eddy_cpu_help', 'fsl:topup_help'])
        for _, args, env in calls:
            self.assertEqual(args, ['--help'])
            self.assertEqual(env['FSLDIR'], str(self.fsl.parent))
            self.assertEqual(env['FSLOUTPUTTYPE'], 'NIFTI_GZ')
            self.assertIn(str(self.fsl), env['PATH'].split(os.pathsep))

    def test_exit_127_fslpython_not_found_fails(self):
        patcher, _ = self.runner(dict(self.good(), eddy=result(127, '', 'fslpython not found')))
        with patcher, self.assertRaisesRegex(RuntimeError, r'eddy --help should print its usage and exit with 1; got exit 127'):
            selftest.check_fsl(self.fsl, shim_root=self.root / 'shims')

    def test_exit_0_and_exit_1_without_usage_fail(self):
        for answer in (result(0, 'Usage: eddy'), result(1, '', 'segmentation fault')):
            patcher, _ = self.runner(dict(self.good(), eddy_cpu=answer))
            with patcher, self.assertRaisesRegex(RuntimeError, 'eddy_cpu --help'):
                selftest.check_fsl(self.fsl, shim_root=self.root / 'shims')

    def test_topup_is_checked_last_and_can_fail_alone(self):
        patcher, calls = self.runner(dict(self.good(), topup=result(127)))
        with patcher, self.assertRaisesRegex(RuntimeError, 'topup --help'):
            selftest.check_fsl(self.fsl, shim_root=self.root / 'shims')
        self.assertEqual([name for name, _, _ in calls], ['eddy', 'eddy_cpu', 'topup'])


class Dcm2niixCheckTests(ToolFixture):
    def test_version_text_passes_with_exit_0_or_1(self):
        for code in (0, 1):
            patcher, calls = self.runner({'dcm2niix': result(code, "Chris Rorden's dcm2niiX version v1.0.20261029")})
            with patcher:
                self.assertEqual(selftest.check_dcm2niix(self.root / 'dcm2niix'), ['dcm2niix:version'])
            self.assertEqual(calls[0][1], ['-v'])

    def test_other_output_or_exit_fails(self):
        for answer in (result(2, 'dcm2niix version 1'), result(0, 'hello'), result(127, '', 'not found')):
            patcher, _ = self.runner({'dcm2niix': answer})
            with patcher, self.assertRaisesRegex(RuntimeError, 'did not report a version'):
                selftest.check_dcm2niix(self.root / 'dcm2niix')


class FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array, dtype=np.float32)
        self.shape = self.array.shape


def fake_torch(conv=None, finite=True):
    torch = ModuleType('torch')
    torch.float32 = 'float32'

    @contextmanager
    def inference_mode():
        yield
    torch.inference_mode = inference_mode
    torch.zeros = lambda shape, dtype=None, device=None: FakeTensor(np.zeros(shape))
    torch.ones = lambda shape, device=None: FakeTensor(np.ones(shape))
    torch.full_like = lambda tensor, value: FakeTensor(np.full(tensor.shape, value))
    torch.equal = lambda a, b: bool(np.array_equal(a.array, b.array))
    torch.isfinite = lambda tensor: SimpleNamespace(all=lambda: finite)

    def conv2d(x, w):
        kernel = w.array.shape[-1]
        size = x.array.shape[-1] - kernel + 1
        out = np.array([[x.array[0, 0, i:i + kernel, j:j + kernel].sum() for j in range(size)] for i in range(size)])
        return FakeTensor(out.reshape(1, 1, size, size) if conv is None else conv)
    torch.nn = SimpleNamespace(functional=SimpleNamespace(conv2d=conv2d))
    return torch


class TorchConvolutionTests(unittest.TestCase):
    def test_correct_convolution_passes_and_names_the_device(self):
        self.assertEqual(selftest.check_torch_convolution(fake_torch(), 'cuda:0'), 'torch:convolution:cuda:0')

    def test_wrong_values_or_shape_fail(self):
        for wrong in (np.full((1, 1, 2, 2), 8), np.full((1, 1, 3, 3), 9)):
            with self.assertRaisesRegex(RuntimeError, 'convolution returned a wrong result'):
                selftest.check_torch_convolution(fake_torch(conv=wrong), 'cpu')


class NativeMaskCheckTests(unittest.TestCase):
    def modules(self, shape=(1, 1, 256, 256), resolved='cpu', finite=True):
        calls = []
        torch = fake_torch(finite=finite)
        inference = SimpleNamespace(VIEWS=('sagittal', 'coronal', 'axial'),
                                    select_device=lambda requested: calls.append(('device', requested)) or {'resolved': resolved})

        def load_model(directory, view, device):
            calls.append(('load', str(directory), view, device))

            def model(inputs, segmentation_only):
                calls.append(('forward', view, tuple(inputs.shape), segmentation_only))
                return FakeTensor(np.zeros(shape))
            return model
        return torch, inference, SimpleNamespace(load_model=load_model), calls

    def test_module_absent_on_this_base_skips_cleanly(self):
        self.assertFalse(selftest.native_mask_module_present())  # W-A is on another branch
        checks, skipped = selftest.check_native_mask({'model_dir': '/m'})
        self.assertEqual(checks, [])
        self.assertEqual(skipped[0]['check'], 'pnl_native_mask')
        self.assertIn('not present', skipped[0]['reason'])

    def test_three_views_each_run_one_256_slice_on_the_selected_device(self):
        torch, inference, weights, calls = self.modules(resolved='cuda')
        with patch.object(selftest, 'native_mask_module_present', return_value=True):
            checks, skipped = selftest.check_native_mask({'model_dir': '/m', 'device': 'auto'},
                                                         torch=torch, inference=inference, weights=weights)
        self.assertEqual(checks, ['pnl_native_mask:sagittal:cuda', 'pnl_native_mask:coronal:cuda', 'pnl_native_mask:axial:cuda'])
        self.assertEqual(skipped, [])
        self.assertEqual(calls[0], ('device', 'auto'))
        self.assertEqual([c for c in calls if c[0] == 'load'],
                         [('load', '/m', view, 'cuda') for view in ('sagittal', 'coronal', 'axial')])
        self.assertEqual([c for c in calls if c[0] == 'forward'],
                         [('forward', view, (1, 1, 256, 256), True) for view in ('sagittal', 'coronal', 'axial')])

    def test_wrong_output_shape_or_nonfinite_output_fails(self):
        with patch.object(selftest, 'native_mask_module_present', return_value=True):
            for kwargs in ({'shape': (1, 1, 128, 128)}, {'finite': False}):
                torch, inference, weights, _ = self.modules(**kwargs)
                with self.assertRaisesRegex(RuntimeError, 'sagittal view gave a wrong or nonfinite slice'):
                    selftest.check_native_mask({'model_dir': '/m'}, torch=torch, inference=inference, weights=weights)

    def test_a_view_that_cannot_load_fails_the_check(self):
        torch, inference, _, _ = self.modules()
        broken = SimpleNamespace(load_model=lambda *a: (_ for _ in ()).throw(ValueError('SHA256 mismatch')))
        with patch.object(selftest, 'native_mask_module_present', return_value=True), \
                self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            selftest.check_native_mask({'model_dir': '/m'}, torch=torch, inference=inference, weights=broken)


class RunChecksDispatchTests(ToolFixture):
    def test_request_keys_select_the_feature_checks_and_skips_are_reported(self):
        answers = {'mrconvert': result(0, 'mrconvert 3.0.8'), 'dwi2response': result(0, 'dwi2response'),
                   'dwifslpreproc': result(0, 'dwifslpreproc'),
                   **{name: result(1, 'Usage') for name in ('eddy', 'eddy_cpu', 'topup')},
                   'dcm2niix': result(0, 'dcm2niix version v1')}
        patcher, _ = self.runner(answers)
        request = {'mrtrix_bin_dir': str(self.mrtrix), 'fsl_bin_dir': str(self.fsl), 'dcm2niix': str(self.root / 'dcm2niix'),
                   'pnl_native_mask': {'model_dir': '/m'}}
        with patcher, patch.dict(os.environ, {'SLICERDSPACE_RUNTIME_ROOT': str(self.root / 'rt')}):
            report = selftest.run_checks(request)
        self.assertEqual(report['status'], 'passed')
        self.assertEqual([c for c in report['checks'] if c.startswith(('mrtrix', 'fsl', 'dcm2niix'))],
                         ['mrtrix:mrconvert_version', 'mrtrix:dwi2response_help', 'mrtrix:dwifslpreproc_help',
                          'fsl:eddy_help', 'fsl:eddy_cpu_help', 'fsl:topup_help', 'dcm2niix:version'])
        self.assertEqual(report['skipped'][0]['check'], 'pnl_native_mask')

    def test_a_failing_feature_fails_the_whole_request(self):
        patcher, _ = self.runner({'mrconvert': result(127, '', 'x')})
        with patcher, self.assertRaises(RuntimeError):
            selftest.run_checks({'mrtrix_bin_dir': str(self.mrtrix)})

    def test_requests_without_feature_keys_are_unchanged(self):
        report = selftest.run_checks({'imports': []})
        self.assertEqual((report['status'], report['checks']), ('passed', []))
        self.assertNotIn('skipped', report)


class ScriptTests(ToolFixture):
    def load_script(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('check_setup_functionality', SCRIPTS / 'check_setup_functionality.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_cli_options_become_request_keys_and_the_result_file_is_written(self):
        script = self.load_script()
        out = self.root / 'result.json'
        with patch.object(script, 'print'), patch('shared.setup_selftest.run_checks',
                                                  return_value={'kind': 'x', 'status': 'passed'}) as run:
            code = script.main(['--result', str(out), '--mrtrix-bin-dir', str(self.mrtrix), '--fsl-bin-dir', str(self.fsl),
                                '--dcm2niix', '/bin/d2n', '--pnl-model-dir', '/m', '--device', 'cpu'])
        self.assertEqual(code, 0)
        run.assert_called_once_with({'mrtrix_bin_dir': str(self.mrtrix), 'fsl_bin_dir': str(self.fsl), 'dcm2niix': '/bin/d2n',
                                     'pnl_native_mask': {'model_dir': '/m', 'device': 'cpu'}})
        self.assertEqual(json.loads(out.read_text())['status'], 'passed')

    def test_failed_check_gives_exit_code_1(self):
        script = self.load_script()
        out = self.root / 'result.json'
        with patch.object(script, 'print'), patch('shared.setup_selftest.run_checks', side_effect=RuntimeError('boom')):
            code = script.main(['--result', str(out), '--dcm2niix', '/bin/d2n'])
        self.assertEqual(code, 1)
        self.assertIn('boom', json.loads(out.read_text())['error'])


if __name__ == '__main__':
    unittest.main()
