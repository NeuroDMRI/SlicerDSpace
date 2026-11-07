"""Native FSL discovery and route-specific dependency checks."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


def _write_program(directory, name, content=b'\x7fELF'):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(content)
    path.chmod(0o755)
    return path


def _write_linux_mrtrix(directory, *, pair=False):
    names = ['mrconvert', 'mrinfo', 'dwi2mask', 'maskfilter']
    if pair:
        names.extend(['dwiextract', 'mrcat', 'mrtransform', 'mrcalc'])
    for name in names:
        _write_program(directory, name)


class NativeFslRuntimeTests(unittest.TestCase):
    def test_fsl_share_wrappers_resolve_to_prefix_binaries(self):
        from preprocessing.fsl_runtime import inspect_native_fsl_runtime, native_fsl_environment

        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            mrtrix = root / 'MRtrix commands'
            fsl_root = root / 'FSL runtime'
            wrappers = fsl_root / 'share' / 'fsl' / 'bin'
            _write_linux_mrtrix(mrtrix)
            _write_program(wrappers, 'eddy_cpu', b'#!/bin/sh\nexit 1\n')
            eddy = _write_program(fsl_root / 'bin', 'eddy_cpu')
            _write_program(fsl_root / 'bin', 'eddy', b'#!/bin/sh\nexit 1\n')
            status = inspect_native_fsl_runtime(mrtrix, wrappers, reverse_phase=False, platform='Linux')

        self.assertTrue(status['available'])
        self.assertEqual(status['commands']['eddy'], str(eddy))
        self.assertNotIn('topup', status['required_commands'])
        environment = native_fsl_environment(status['commands'], status['fsl_root'])
        self.assertEqual(environment['FSLDIR'], str(fsl_root))
        self.assertEqual(environment['FSLOUTPUTTYPE'], 'NIFTI_GZ')

    def test_reverse_phase_requires_topup_and_its_configuration(self):
        from preprocessing.fsl_runtime import inspect_native_fsl_runtime

        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            mrtrix, fsl = root / 'mrtrix', root / 'fsl' / 'bin'
            _write_linux_mrtrix(mrtrix, pair=True)
            for name in ('eddy_cpu', 'topup'):
                _write_program(fsl, name)
            status = inspect_native_fsl_runtime(mrtrix, fsl, reverse_phase=True, platform='Linux')
            self.assertFalse(status['available'])
            self.assertIn('applytopup', status['missing_commands'])
            self.assertIn('topup_config', status['missing_required'])

            _write_program(fsl, 'applytopup')
            topup_config = fsl.parent / 'etc' / 'flirtsch' / 'b02b0.cnf'
            topup_config.parent.mkdir(parents=True)
            topup_config.write_text('--subsamp=1\n', encoding='utf-8')
            status = inspect_native_fsl_runtime(mrtrix, fsl, reverse_phase=True, platform='Linux')

        self.assertTrue(status['available'])
        self.assertEqual(status['topup_config'], str(topup_config))

    def test_explicit_windows_prefix_rejects_a_wsl_eddy_bridge(self):
        from preprocessing.fsl_runtime import inspect_native_fsl_runtime

        pe = b'MZ' + bytes(58) + (64).to_bytes(4, 'little') + b'PE\x00\x00'
        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            mrtrix, fsl = root / 'mrtrix', root / 'fsl' / 'bin'
            for name in ('mrconvert', 'mrinfo', 'dwi2mask', 'maskfilter'):
                _write_program(mrtrix, name + '.exe', pe)
            eddy = _write_program(fsl, 'eddy_cpu.exe', pe)
            marker = eddy.with_name(eddy.name + '.slicerdsurfer-owner')
            marker.touch()
            other = root / 'different' / 'bin'
            _write_program(other, 'eddy_cpu.exe', pe)
            with patch.dict(os.environ, {'SLICERDSPACE_FSL_BIN_DIR': str(other)}):
                status = inspect_native_fsl_runtime(mrtrix, fsl, reverse_phase=False, platform='Windows')
            self.assertFalse(status['available'])
            self.assertEqual(status['missing_commands'], ['eddy'])

            marker.unlink()
            status = inspect_native_fsl_runtime(mrtrix, fsl, reverse_phase=False, platform='Windows')

        self.assertTrue(status['available'])
        self.assertEqual(status['commands']['eddy'], str(eddy))

    def test_native_doctor_checks_binaries_without_legacy_wrapper(self):
        from shared.environment import EnvironmentDoctorConfig, build_dependency_specs, _check_spec

        with TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            mrtrix, fsl = root / 'mrtrix', root / 'fsl' / 'bin'
            _write_linux_mrtrix(mrtrix)
            _write_program(fsl, 'eddy_cpu')
            config = EnvironmentDoctorConfig(
                targets=('preprocessing',), repo_root=root, platform_name='Linux',
                preprocessing_execution_backend='native',
                preprocessing_mrtrix_bin_dir=mrtrix, preprocessing_fsl_bin_dir=fsl,
            )
            specs = build_dependency_specs(config)
            runtime = {spec.label: _check_spec(spec, 'Linux')
                       for spec in specs if spec.install_group in {'native-fsl', 'native-mrtrix'}}

        self.assertNotIn('command:dwifslpreproc', {spec.key for spec in specs})
        self.assertNotIn('command:topup', {spec.key for spec in specs})
        self.assertEqual(set(runtime), {'mrconvert', 'mrinfo', 'dwi2mask', 'maskfilter', 'eddy_cpu'})
        self.assertTrue(all(status.status == 'available' for status in runtime.values()))


if __name__ == '__main__':
    unittest.main()
