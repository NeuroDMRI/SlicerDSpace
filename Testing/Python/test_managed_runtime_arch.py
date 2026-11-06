"""B4: conda architecture follows the host, never Slicer's own Python. No network, no real conda."""

from contextlib import nullcontext
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from shared import managed_runtimes as runtime
from shared import platform_info

MAC_VERSION = ('15.1', ('', '', ''), '')


def apple_silicon_under_rosetta():
    """Slicer is x86_64 (Rosetta) on an arm64 Mac; sysctl reports the translation."""
    return (patch.object(platform_info.platform, 'system', return_value='Darwin'),
            patch.object(platform_info.platform, 'machine', return_value='x86_64'),
            patch.object(platform_info.subprocess, 'run',
                         return_value=SimpleNamespace(returncode=0, stdout='1\n', stderr='')))


class CondaArchitectureTests(unittest.TestCase):
    def setUp(self):
        guard = patch.object(runtime.urllib.request, 'urlopen', side_effect=AssertionError('tests must not use the network'))
        guard.start()
        self.addCleanup(guard.stop)

    def rosetta(self):
        """Only for tests that exercise the sysctl path itself."""
        for patcher in apple_silicon_under_rosetta():
            patcher.start()
            self.addCleanup(patcher.stop)

    def arm64_mac(self):
        """An Apple Silicon host; the module-level host choice is fixed so that subprocess mocks stay independent."""
        for patcher in (patch.object(runtime.platform, 'system', return_value='Darwin'),
                        patch.object(runtime, 'host_machine', return_value='arm64')):
            patcher.start()
            self.addCleanup(patcher.stop)

    def conda_info(self, platform_name):
        return patch.object(runtime.subprocess, 'run', return_value=SimpleNamespace(
            returncode=0, stdout=json.dumps({'platform': platform_name}), stderr=''))

    def test_arm64_host_under_rosetta_selects_osx_arm64_and_the_arm64_installer(self):
        self.rosetta()
        self.assertEqual(runtime.conda_subdir(), 'osx-arm64')
        url, sha = runtime.bootstrap_artifact()
        self.assertTrue(url.endswith('/26.7.2-0/Miniforge3-26.7.2-0-MacOSX-arm64.sh'))
        self.assertEqual(sha, 'd70bfa2e97afcda96927c9b9ca0e2316cb7750e4ce651c94388267cbe9588711')
        self.assertEqual(runtime.install_plan(('fsl_cpu',), Path('/nowhere'))['conda_subdir'], 'osx-arm64')

    def test_intel_mac_and_linux_choose_their_own_subdir_and_installer(self):
        with patch.object(runtime.platform, 'system', return_value='Darwin'), \
                patch.object(runtime, 'host_machine', return_value='x86_64'):
            self.assertEqual(runtime.conda_subdir(), 'osx-64')
            self.assertTrue(runtime.bootstrap_artifact()[0].endswith('MacOSX-x86_64.sh'))
        with patch.object(runtime.platform, 'system', return_value='Linux'), \
                patch.object(runtime, 'host_machine', return_value='x86_64'):
            self.assertEqual(runtime.conda_subdir(), 'linux-64')
            self.assertEqual(runtime.bootstrap_artifact(), (runtime.MINIFORGE_URL, runtime.MINIFORGE_SHA256))
        self.assertIsNone(runtime.conda_subdir('Windows', 'AMD64'))
        self.assertIsNone(runtime.conda_subdir('Linux', 'aarch64'))

    def test_darwin_allow_list_is_the_mrtrix_tarball_and_the_fsl_cpu_profile_on_both_hosts(self):
        with patch.object(runtime.platform, 'mac_ver', return_value=MAC_VERSION):
            for machine in ('x86_64', 'arm64'):
                for profiles in (('mrtrix',), ('fsl_cpu',), ('mrtrix', 'fsl_cpu')):
                    self.assertTrue(runtime.supported_platform('Darwin', machine, profiles=profiles), (machine, profiles))
                for profiles in (('fsl_cuda',), ('pnl_cnn',), ('mrtrix', 'pnl_cnn'), ()):
                    self.assertFalse(runtime.supported_platform('Darwin', machine, profiles=profiles), (machine, profiles))
                self.assertFalse(runtime.supported_platform('Darwin', machine))

    def test_supported_platform_uses_the_host_machine_not_slicers(self):
        with patch.object(runtime, 'host_machine', return_value='arm64') as host, \
                patch.object(runtime.platform, 'mac_ver', return_value=MAC_VERSION), \
                patch.object(runtime.platform, 'machine', return_value='x86_64'):
            self.assertTrue(runtime.supported_platform('Darwin', profiles=('fsl_cpu',)))
            host.assert_called()

    def test_fsl_cpu_build_string_follows_the_subdir(self):
        for subdir, build in (('linux-64', 'hb35a746_0'), ('osx-64', 'h7ffe42b_0'), ('osx-arm64', 'hed4507f_0')):
            with patch.object(runtime, 'conda_subdir', return_value=subdir):
                packages = runtime.profile_spec('fsl_cpu')['packages']
            self.assertEqual(packages, (f'fsl-eddy=2602.0={build}', 'fsl-topup=2203.5', 'python=3.12'))

    def test_path_conda_of_another_platform_is_not_used(self):
        self.arm64_mac()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            name = 'miniforge-26.7.2-0-osx-arm64'
            (root / 'Miniforge3-26.7.2-0-MacOSX-arm64.sh').write_text('fixture, not executed')

            def installed(argv, extra_env=None):
                (root / name / 'bin').mkdir(parents=True)
                (root / name / 'bin/conda').write_text('fixture')

            with patch.object(runtime.shutil, 'which', return_value='/opt/conda/bin/conda'), \
                    patch.object(runtime, 'file_hash', return_value=runtime.MINIFORGE_MACOS_ARM64_SHA256), \
                    patch.object(runtime, '_run', side_effect=installed), self.conda_info('osx-64'):
                self.assertEqual(runtime.ensure_conda(root), str(root / name / 'bin/conda'))
            marker = json.loads((root / (name + '.bootstrap.json')).read_text())
            self.assertEqual((marker['subdir'], marker['status']), ('osx-arm64', 'ready'))

    def test_path_conda_of_the_expected_platform_is_used(self):
        self.arm64_mac()
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(runtime.shutil, 'which', return_value='/opt/conda/bin/conda'), \
                self.conda_info('osx-arm64'), patch.object(runtime, '_run') as run:
            self.assertEqual(runtime.ensure_conda(Path(temp)), '/opt/conda/bin/conda')
            run.assert_not_called()

    def test_unreadable_conda_info_means_the_path_conda_is_not_used(self):
        for bad in (SimpleNamespace(returncode=1, stdout='', stderr='x'),
                    SimpleNamespace(returncode=0, stdout='nope', stderr='')):
            with patch.object(runtime.subprocess, 'run', return_value=bad):
                self.assertIsNone(runtime._path_conda_platform('/opt/conda/bin/conda'))

    def bootstrap_fixture(self, root, marker_subdir):
        bootstrap = root / 'miniforge-26.7.2-0-osx-arm64'
        (bootstrap / 'bin').mkdir(parents=True)
        (bootstrap / 'bin/conda').write_text('fixture')
        (root / 'miniforge-26.7.2-0-osx-arm64.bootstrap.json').write_text(json.dumps(
            {'kind': 'slicerdsurfer_miniforge_bootstrap', 'prefix': str(bootstrap), 'subdir': marker_subdir}))
        return bootstrap

    def test_bootstrap_of_another_subdir_is_not_reused(self):
        self.arm64_mac()
        with tempfile.TemporaryDirectory() as temp, patch.object(runtime.shutil, 'which', return_value=None):
            self.bootstrap_fixture(Path(temp), 'osx-64')
            with self.assertRaisesRegex(RuntimeError, 'not built for osx-arm64'):
                runtime.ensure_conda(Path(temp))

    def test_bootstrap_with_the_same_subdir_is_reused_without_download(self):
        self.arm64_mac()
        with tempfile.TemporaryDirectory() as temp, patch.object(runtime.shutil, 'which', return_value=None):
            bootstrap = self.bootstrap_fixture(Path(temp), 'osx-arm64')
            with patch.object(runtime.subprocess, 'run', return_value=SimpleNamespace(returncode=0)), \
                    patch.object(runtime.urllib.request, 'urlopen') as download:
                self.assertEqual(runtime.ensure_conda(Path(temp)), str(bootstrap / 'bin/conda'))
                download.assert_not_called()

    def test_every_create_and_install_runs_with_conda_subdir(self):
        self.arm64_mac()
        with tempfile.TemporaryDirectory() as temp, patch.object(runtime, 'setup_lock', return_value=nullcontext()), \
                patch.object(runtime.platform, 'mac_ver', return_value=MAC_VERSION), \
                patch.object(runtime, 'ensure_conda', return_value='conda'), patch.object(runtime, '_run') as run, \
                patch.object(runtime.subprocess, 'run', return_value=SimpleNamespace(stdout='lock')), \
                patch.object(runtime, 'verify_prefix', return_value={'external_dependencies': {}, 'command_sha256': {}}):
            runtime.provision(('fsl_cpu',), root=Path(temp), accept_fsl_license=True)
            command, environment = run.call_args.args
            self.assertEqual(environment, {'CONDA_SUBDIR': 'osx-arm64'})
            self.assertIn('fsl-eddy=2602.0=hed4507f_0', command)

    def test_run_puts_conda_subdir_into_the_child_environment(self):
        with patch.object(runtime.subprocess, 'run') as run:
            runtime._run(['conda', 'create'], {'CONDA_SUBDIR': 'osx-arm64'})
        self.assertEqual(run.call_args.kwargs['env']['CONDA_SUBDIR'], 'osx-arm64')

    def prefix_with_record(self, root, subdir):
        prefix = root / 'prefix'
        (prefix / 'bin').mkdir(parents=True)
        (prefix / 'conda-meta').mkdir()
        for name in runtime.PROFILES['fsl_cpu']['commands']:
            (prefix / 'bin' / name).write_text('# fixture\n')
            (prefix / 'bin' / name).chmod(0o755)
        for package in runtime.profile_spec('fsl_cpu')['packages']:
            name, version = package.split('=')[:2]
            (prefix / 'conda-meta' / (name + '.json')).write_text(json.dumps(dict(name=name, version=version, subdir=subdir)))
        return prefix

    def test_prefix_with_an_osx_64_record_fails_verification_on_an_arm64_host(self):
        self.arm64_mac()
        with tempfile.TemporaryDirectory() as temp:
            prefix = self.prefix_with_record(Path(temp), 'osx-64')
            with self.assertRaisesRegex(RuntimeError, 'another architecture; use Repair'):
                runtime.verify_prefix('fsl_cpu', prefix)

    def test_matching_and_noarch_records_pass_verification(self):
        self.arm64_mac()
        with tempfile.TemporaryDirectory() as temp:
            prefix = self.prefix_with_record(Path(temp), 'osx-arm64')
            record = json.loads((prefix / 'conda-meta/python.json').read_text())
            (prefix / 'conda-meta/python.json').write_text(json.dumps(dict(record, subdir='noarch')))
            self.assertEqual(set(runtime.verify_prefix('fsl_cpu', prefix)['command_sha256']),
                             {'eddy', 'eddy_cpu', 'topup'})


if __name__ == '__main__':
    unittest.main()
