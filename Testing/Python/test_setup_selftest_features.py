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
            if callable(answer):
                answer = answer(list(argv[1:]))
            if isinstance(answer, Exception):
                raise answer
            return answer
        return patch.object(selftest.subprocess, 'run', side_effect=run), calls


def mrtrix_script(name):
    """MRtrix scripts take `-help`; `--help` exits 1 with an argparse error (measured 2026-11-07, MRtrix 3.0.8)."""
    return lambda args: (result(0, f'{name}: usage ...') if args == ['-help']
                         else result(1, '', 'Error: unrecognized arguments: --help'))


class MrtrixCheckTests(ToolFixture):
    def good(self, version='3.0.8'):
        return {'mrconvert': result(0, f'== mrconvert {version} ==\n'),
                'dwi2response': mrtrix_script('dwi2response'), 'dwifslpreproc': mrtrix_script('dwifslpreproc')}

    def managed(self):
        """Treat self.root as the plugin runtime root, so self.mrtrix is a plugin-managed prefix."""
        return patch('shared.managed_runtimes.runtime_root', return_value=self.root)

    def test_all_three_commands_run_through_the_managed_environment(self):
        patcher, calls = self.runner(self.good())
        with patcher:
            checks = selftest.check_mrtrix(self.mrtrix, self.fsl, shim_root=self.root / 'shims')
        self.assertEqual(checks, ['mrtrix:mrconvert_version', 'mrtrix:dwi2response_help', 'mrtrix:dwifslpreproc_help'])
        self.assertEqual([(name, args) for name, args, _ in calls],
                         [('mrconvert', ['--version']), ('dwi2response', ['-help']), ('dwifslpreproc', ['-help'])])
        for _, _, env in calls:  # the B3 environment, not the bare process environment
            self.assertEqual(env['FSLOUTPUTTYPE'], 'NIFTI_GZ')
            self.assertIn(str(self.mrtrix), env['PATH'].split(os.pathsep))

    def test_help_uses_a_single_dash_and_the_old_double_dash_call_fails_the_mock(self):
        """R-B1: a check that calls `--help` gets MRtrix's exit 1 and must fail."""
        for name in ('dwi2response', 'dwifslpreproc'):
            self.assertEqual(mrtrix_script(name)(['--help']).returncode, 1)
        patcher, calls = self.runner(self.good())
        with patcher:
            selftest.check_mrtrix(self.mrtrix, shim_root=self.root / 'shims')
        self.assertTrue(all(args != ['--help'] for _, args, _ in calls))

    def test_managed_prefix_needs_exactly_the_pinned_version(self):
        """R-B4: the plugin-managed prefix with 3.0.4 fails."""
        for text in ('== mrconvert 3.0.4 ==', '== mrconvert 3.0.7 ==', '== mrconvert 13.0.8 ==', ''):
            patcher, _ = self.runner(dict(self.good(), mrconvert=result(0, text)))
            with patcher, self.managed(), self.assertRaisesRegex(RuntimeError, 'does not report 3.0.8'):
                selftest.check_mrtrix(self.mrtrix, shim_root=self.root / 'shims')

    def test_user_set_mrtrix_of_another_3_x_release_passes(self):
        """R-B4: a working user-set MRtrix 3.0.4 passes; a failed or non-MRtrix banner does not."""
        patcher, _ = self.runner(self.good('3.0.4'))
        with patcher:
            self.assertIn('mrtrix:mrconvert_version', selftest.check_mrtrix(self.mrtrix, shim_root=self.root / 'shims'))
        for answer in (result(1, '== mrconvert 3.0.4 =='), result(0, 'something else'), result(0, '== mrconvert 2.9 ==')):
            patcher, _ = self.runner(dict(self.good(), mrconvert=answer))
            with patcher, self.assertRaisesRegex(RuntimeError, r'does not report 3\.x'):
                selftest.check_mrtrix(self.mrtrix, shim_root=self.root / 'shims')

    def test_a_script_that_cannot_find_python_fails(self):
        patcher, _ = self.runner(dict(self.good(), dwifslpreproc=result(127, '', 'env: python: No such file')))
        with patcher, self.assertRaisesRegex(RuntimeError, r'dwifslpreproc -help failed \(exit 127\)'):
            selftest.check_mrtrix(self.mrtrix, shim_root=self.root / 'shims')

    def test_help_text_without_the_tool_name_fails(self):
        patcher, _ = self.runner(dict(self.good(), dwi2response=result(0, 'something else')))
        with patcher, self.assertRaisesRegex(RuntimeError, 'dwi2response -help failed'):
            selftest.check_mrtrix(self.mrtrix, shim_root=self.root / 'shims')

    def test_a_missing_binary_is_reported_not_raised_as_oserror(self):
        patcher, _ = self.runner(dict(self.good(), mrconvert=FileNotFoundError('no mrconvert')))
        with patcher, self.assertRaisesRegex(RuntimeError, 'mrconvert could not run'):
            selftest.check_mrtrix(self.mrtrix, shim_root=self.root / 'shims')


class FslCheckTests(ToolFixture):
    def good(self):
        return {name: result(1, '', f'{name}\nUsage: {name} [options]') for name in ('eddy', 'eddy_cpu', 'eddy_openmp', 'topup')}

    def managed(self):
        """Treat self.root as the plugin runtime root, so self.fsl is a plugin-managed prefix."""
        return patch('shared.managed_runtimes.runtime_root', return_value=self.root)

    def present(self, *names):
        for name in names:
            (self.fsl / name).write_text('#!/bin/sh\n')

    def test_exit_1_with_usage_text_passes_for_all_three(self):
        self.present('eddy_cpu')
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
        self.present('eddy_cpu')
        for answer in (result(0, 'Usage: eddy'), result(1, '', 'segmentation fault')):
            patcher, _ = self.runner(dict(self.good(), eddy_cpu=answer))
            with patcher, self.assertRaisesRegex(RuntimeError, 'eddy_cpu --help'):
                selftest.check_fsl(self.fsl, shim_root=self.root / 'shims')

    def test_topup_is_checked_last_and_can_fail_alone(self):
        self.present('eddy_cpu')
        patcher, calls = self.runner(dict(self.good(), topup=result(127)))
        with patcher, self.assertRaisesRegex(RuntimeError, 'topup --help'):
            selftest.check_fsl(self.fsl, shim_root=self.root / 'shims')
        self.assertEqual([name for name, _, _ in calls], ['eddy', 'eddy_cpu', 'topup'])

    def test_user_set_fsl_with_eddy_openmp_and_no_eddy_cpu_passes(self):
        """G2: FSL before 6.0.7 has eddy_openmp, not eddy_cpu. Fails before the fix: eddy_cpu was always run."""
        self.present('eddy_openmp')
        patcher, calls = self.runner(self.good())
        with patcher:
            checks = selftest.check_fsl(self.fsl, shim_root=self.root / 'shims')
        self.assertEqual(checks, ['fsl:eddy_help', 'fsl:eddy_openmp_help', 'fsl:topup_help'])
        self.assertEqual([name for name, _, _ in calls], ['eddy', 'eddy_openmp', 'topup'])

    def test_user_set_fsl_needs_only_eddy_and_topup(self):
        patcher, calls = self.runner(self.good())
        with patcher:
            checks = selftest.check_fsl(self.fsl, shim_root=self.root / 'shims')
        self.assertEqual(checks, ['fsl:eddy_help', 'fsl:topup_help'])
        for missing in ('eddy', 'topup'):
            patcher, _ = self.runner(dict(self.good(), **{missing: FileNotFoundError(missing)}))
            with patcher, self.assertRaisesRegex(RuntimeError, missing + ' could not run'):
                selftest.check_fsl(self.fsl, shim_root=self.root / 'shims')

    def test_a_present_eddy_openmp_that_fails_fails_the_user_set_fsl(self):
        self.present('eddy_openmp')
        patcher, _ = self.runner(dict(self.good(), eddy_openmp=result(127)))
        with patcher, self.assertRaisesRegex(RuntimeError, 'eddy_openmp --help'):
            selftest.check_fsl(self.fsl, shim_root=self.root / 'shims')

    def test_managed_prefix_without_eddy_cpu_fails(self):
        """G2: the plugin-managed FSL prefix must have eddy_cpu."""
        self.present('eddy_openmp')
        patcher, calls = self.runner(dict(self.good(), eddy_cpu=FileNotFoundError('eddy_cpu')))
        with patcher, self.managed(), self.assertRaisesRegex(RuntimeError, 'eddy_cpu could not run'):
            selftest.check_fsl(self.fsl, shim_root=self.root / 'shims')
        patcher, calls = self.runner(self.good())
        with patcher, self.managed():
            checks = selftest.check_fsl(self.fsl, shim_root=self.root / 'shims')
        self.assertEqual(checks, ['fsl:eddy_help', 'fsl:eddy_cpu_help', 'fsl:topup_help'])


class Dcm2niixCheckTests(ToolFixture):
    def test_version_text_passes_with_exit_0_or_3(self):
        """R-B2: the real dcm2niix exits 3 after `-v` (measured 2026-11-07)."""
        for code in (0, 3):
            patcher, calls = self.runner({'dcm2niix': result(code, "Chris Rorden's dcm2niiX version v1.0.20261029")})
            with patcher:
                self.assertEqual(selftest.check_dcm2niix(self.root / 'dcm2niix'), ['dcm2niix:version'])
            self.assertEqual(calls[0][1], ['-v'])

    def test_other_output_or_exit_fails(self):
        for answer in (result(1, 'dcm2niix version 1'), result(2, 'dcm2niix version 1'), result(0, 'hello'),
                       result(127, '', 'not found')):
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

    def test_module_absent_is_a_failure_not_a_skip(self):
        """R-B5: a requested PNL check with shared/pnl_mask_native.py absent fails."""
        torch, inference, weights, calls = self.modules()
        with patch.object(selftest, 'native_mask_module_present', return_value=False), \
                self.assertRaisesRegex(RuntimeError, 'pnl_mask_native.py is not present'):
            selftest.check_native_mask({'model_dir': '/m'}, torch=torch, inference=inference, weights=weights)
        self.assertEqual(calls, [])  # nothing ran

    def test_three_views_each_run_one_256_slice_on_the_selected_device(self):
        torch, inference, weights, calls = self.modules(resolved='cuda')
        with patch.object(selftest, 'native_mask_module_present', return_value=True):
            checks = selftest.check_native_mask({'model_dir': '/m', 'device': 'auto'},
                                                torch=torch, inference=inference, weights=weights)
        self.assertEqual(checks, ['pnl_native_mask:sagittal:cuda', 'pnl_native_mask:coronal:cuda', 'pnl_native_mask:axial:cuda'])
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
    def test_request_keys_select_the_feature_checks(self):
        answers = {'mrconvert': result(0, 'mrconvert 3.0.8'), 'dwi2response': mrtrix_script('dwi2response'),
                   'dwifslpreproc': mrtrix_script('dwifslpreproc'),
                   **{name: result(1, 'Usage') for name in ('eddy', 'eddy_cpu', 'topup')},
                   'dcm2niix': result(3, 'dcm2niix version v1')}
        patcher, _ = self.runner(answers)
        (self.fsl / 'eddy_cpu').write_text('#!/bin/sh\n')
        request = {'mrtrix_bin_dir': str(self.mrtrix), 'fsl_bin_dir': str(self.fsl), 'dcm2niix': str(self.root / 'dcm2niix'),
                   'pnl_native_mask': {'model_dir': '/m'}}
        with patcher, patch.dict(os.environ, {'SLICERDSPACE_RUNTIME_ROOT': str(self.root / 'rt')}), \
                patch.object(selftest, 'check_native_mask', return_value=['pnl_native_mask:axial:cpu']) as native:
            report = selftest.run_checks(request)
        native.assert_called_once_with({'model_dir': '/m'})
        self.assertEqual(report['status'], 'passed')
        self.assertIn('pnl_native_mask:axial:cpu', report['checks'])
        self.assertEqual([c for c in report['checks'] if c.startswith(('mrtrix', 'fsl', 'dcm2niix'))],
                         ['mrtrix:mrconvert_version', 'mrtrix:dwi2response_help', 'mrtrix:dwifslpreproc_help',
                          'fsl:eddy_help', 'fsl:eddy_cpu_help', 'fsl:topup_help', 'dcm2niix:version'])

    def test_a_failing_feature_fails_the_whole_request(self):
        patcher, _ = self.runner({'mrconvert': result(127, '', 'x')})
        with patcher, self.assertRaises(RuntimeError):
            selftest.run_checks({'mrtrix_bin_dir': str(self.mrtrix)})

    def test_requests_without_feature_keys_are_unchanged(self):
        report = selftest.run_checks({'imports': []})
        self.assertEqual((report['status'], report['checks']), ('passed', []))
        self.assertNotIn('skipped', report)

    def test_requested_native_mask_with_module_absent_fails_the_run(self):
        """R-B5 through run_checks (no real inference: the module check fails first)."""
        with patch.object(selftest, 'native_mask_module_present', return_value=False), \
                self.assertRaisesRegex(RuntimeError, 'not present'):
            selftest.run_checks({'pnl_native_mask': {'model_dir': '/m'}})


def status(key, kind, *, group='', details=None, required=True, state='available'):
    return SimpleNamespace(key=key, kind=kind, label=key, status=state, required=required, targets=('preprocessing',),
                           details=details or {}, install_group=group)


def tool_report(*items):
    return SimpleNamespace(statuses=list(items), targets=('preprocessing',))


MRTRIX = status('command:dwifslpreproc', 'command', group='mrtrix', details={'resolved': '/opt/mrtrix3/bin/dwifslpreproc'})
FSL = status('command:eddy', 'command', group='fsl', details={'resolved': '/opt/fsl/bin/eddy'})
PNL = status('brain_mask_backend', 'setting', details={
    'backend': 'pnl_cnn', 'tool_paths': {'model_dir': '/models/pnl-cnn', 'pythonslicer': '/slicer/PythonSlicer'},
    'settings': {'device': 'cuda'}})
SLICER_NUMPY = status('python:numpy', 'python', details={'import': 'numpy', 'python_executable': '/slicer/PythonSlicer'})


class RequestsForReportToolTests(unittest.TestCase):
    """R-B3: the builder used by the GUI check and the setup self_check step adds the B9 feature checks."""

    def setUp(self):
        # G1: the fallback asks DICOM import for dcm2niix; keep these tests independent of the machine's PATH.
        patcher = patch('shared.input_import.packaged_dcm2niix', return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = patch('shared.input_import.shutil.which', return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_report_with_mrtrix_fsl_and_the_pnl_method_gives_all_keys(self):
        requests = selftest.requests_for_report(tool_report(SLICER_NUMPY, MRTRIX, FSL, PNL),
                                                backend_paths={'dcm2niix_executable': '/opt/d2n/dcm2niix'})
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(request['python'], '/slicer/PythonSlicer')
        self.assertEqual(request['imports'], ['numpy'])
        self.assertEqual(request['mrtrix_bin_dir'], '/opt/mrtrix3/bin')
        self.assertEqual(request['fsl_bin_dir'], '/opt/fsl/bin')
        self.assertEqual(request['dcm2niix'], '/opt/d2n/dcm2niix')
        self.assertEqual(request['pnl_native_mask'], {'model_dir': '/models/pnl-cnn', 'device': 'cuda'})

    def test_tools_only_report_gets_a_request_for_the_report_interpreter(self):
        requests = selftest.requests_for_report(tool_report(MRTRIX, FSL))
        self.assertEqual(len(requests), 1)
        self.assertIsNone(requests[0]['python'])  # None: the interpreter that ran the environment check
        self.assertEqual((requests[0]['mrtrix_bin_dir'], requests[0]['fsl_bin_dir']), ('/opt/mrtrix3/bin', '/opt/fsl/bin'))
        self.assertNotIn('pnl_native_mask', requests[0])
        self.assertNotIn('dcm2niix', requests[0])

    def test_pnl_check_goes_to_the_pythonslicer_the_backend_reports_when_no_group_is_slicer(self):
        plain = status('python:numpy', 'python', details={'import': 'numpy', 'python_executable': '/venv/bin/python'})
        requests = {r['python']: r for r in selftest.requests_for_report(tool_report(plain, PNL))}
        self.assertNotIn('pnl_native_mask', requests['/venv/bin/python'])
        self.assertEqual(requests['/slicer/PythonSlicer']['pnl_native_mask']['model_dir'], '/models/pnl-cnn')

    def test_unselected_missing_optional_and_other_backends_add_nothing(self):
        optional = status('command:tckconvert', 'command', group='mrtrix', required=False,
                          details={'resolved': '/opt/mrtrix3/bin/tckconvert'})
        missing = status('command:eddy', 'command', group='fsl', state='missing', details={'resolved': None})
        other = status('brain_mask_backend', 'setting', details={'backend': 'synthstrip', 'tool_paths': {'model_dir': '/m'}})
        self.assertEqual(selftest.requests_for_report(tool_report(SLICER_NUMPY, optional, missing, other)),
                         [dict(python='/slicer/PythonSlicer', imports=['numpy'], pythonpath=[], models=[], devices=['cpu'])])

    def test_dcm2niix_comes_from_the_report_entry_first(self):
        entry = status('command:dcm2niix', 'command', group='dcm2niix', details={'resolved': '/opt/d2n/dcm2niix'})
        request = selftest.requests_for_report(tool_report(entry), backend_paths={'packaged_dcm2niix': '/other'})[0]
        self.assertEqual(request['dcm2niix'], '/opt/d2n/dcm2niix')
        request = selftest.requests_for_report(tool_report(SLICER_NUMPY), backend_paths={'packaged_dcm2niix': '/pkg/dcm2niix'})[0]
        self.assertEqual(request['dcm2niix'], '/pkg/dcm2niix')


    def test_dcm2niix_from_the_packaged_wheel_when_the_report_has_no_field(self):
        """G1: no report field and no path setting; the wheel in Slicer's Python is what DICOM import uses."""
        with patch('shared.input_import.packaged_dcm2niix', return_value=Path('/slicer/site/bin/dcm2niix')):
            request = selftest.requests_for_report(tool_report(SLICER_NUMPY))[0]
        self.assertEqual(request['dcm2niix'], '/slicer/site/bin/dcm2niix')
        self.assertEqual(request['python'], '/slicer/PythonSlicer')  # Slicer's Python request

    def test_dcm2niix_on_path_when_there_is_no_wheel_and_none_means_no_check(self):
        with patch('shared.input_import.shutil.which', return_value='/usr/local/bin/dcm2niix'):
            request = selftest.requests_for_report(tool_report(SLICER_NUMPY))[0]
        self.assertEqual(request['dcm2niix'], '/usr/local/bin/dcm2niix')
        request = selftest.requests_for_report(tool_report(SLICER_NUMPY))[0]  # DWIConvert fallback
        self.assertNotIn('dcm2niix', request)

    def test_mrtrix_used_only_as_the_brain_mask_backend_gets_its_check(self):
        """G3: the report has no MRtrix command entry, only the required brain_mask_backend setting."""
        mask = status('brain_mask_backend', 'setting', details={
            'backend': 'mrtrix', 'tool_paths': {'dwi2mask': '/opt/mrtrix3/bin/dwi2mask'}})
        request = selftest.requests_for_report(tool_report(SLICER_NUMPY, mask))[0]
        self.assertEqual(request['mrtrix_bin_dir'], '/opt/mrtrix3/bin')
        self.assertNotIn('fsl_bin_dir', request)
        missing = status('brain_mask_backend', 'setting', state='missing', details={
            'backend': 'mrtrix', 'tool_paths': {'dwi2mask': '/opt/mrtrix3/bin/dwi2mask'}})
        self.assertNotIn('mrtrix_bin_dir', selftest.requests_for_report(tool_report(SLICER_NUMPY, missing))[0])
        # a command entry of the preprocessing target still wins
        request = selftest.requests_for_report(tool_report(SLICER_NUMPY, MRTRIX, mask))[0]
        self.assertEqual(request['mrtrix_bin_dir'], '/opt/mrtrix3/bin')


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

    def test_n1_help_text_names_the_calls_the_checks_make(self):
        """N-1 guard: the options describe -help for MRtrix scripts, --help for FSL and -v for dcm2niix."""
        import io
        from contextlib import redirect_stdout
        script = self.load_script()
        out = io.StringIO()
        with redirect_stdout(out), self.assertRaises(SystemExit):
            script.main(['--help'])
        text = ' '.join(out.getvalue().split())
        self.assertIn('dwi2response -help and dwifslpreproc -help', text)
        self.assertNotIn('dwi2response --help', text)
        self.assertIn('eddy and topup --help', text)
        self.assertIn('dcm2niix -v', text)

    def test_failed_check_gives_exit_code_1(self):
        script = self.load_script()
        out = self.root / 'result.json'
        with patch.object(script, 'print'), patch('shared.setup_selftest.run_checks', side_effect=RuntimeError('boom')):
            code = script.main(['--result', str(out), '--dcm2niix', '/bin/d2n'])
        self.assertEqual(code, 1)
        self.assertIn('boom', json.loads(out.read_text())['error'])


if __name__ == '__main__':
    unittest.main()
