"""First-use delivery checks are offline, scoped and functional, not import-only."""

import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from shared import backend_sources as sources
from shared.selected_setup import build_setup_plan
from shared.setup_selftest import requests_for_report, run_checks
from shared.ui_selected_setup import python_setup_commands


class BackendSourceTests(unittest.TestCase):
    def fixture(self, directory, *, member='package/entry.py', link=False):
        archive = directory / 'fixture.tar.gz'
        with tarfile.open(archive, 'w:gz') as bundle:
            entry = tarfile.TarInfo('author/' + member)
            content = b'# synthetic source fixture\n'
            entry.size = len(content)
            if link:
                entry.type, entry.linkname, entry.size = tarfile.SYMTYPE, '/outside', 0
            bundle.addfile(entry, io.BytesIO(content))
        return dict(repository='example/fixture', revision='a' * 40,
                    sha256=sources.file_hash(archive), bytes=archive.stat().st_size,
                    setting='fixture_root', entry='package/entry.py', paths=('package',), requirements=())

    def test_plan_and_permission_gate_do_not_create_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'absent'
            self.assertEqual(sources.prepare_sources(('wmql',), root=root)['status'], 'planned')
            with self.assertRaises(PermissionError):
                sources.prepare_sources(('wmql',), root=root, execute=True)
            self.assertFalse(root.exists())

    def test_offline_install_reuses_verified_sources_without_git_or_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            spec = self.fixture(folder)
            with patch.dict(sources.SOURCES, {'fixture': spec}, clear=True), patch('urllib.request.urlopen') as download:
                result = sources.prepare_sources(('fixture',), root=folder / 'installed',
                    offline_root=folder, execute=True, accept_terms=True)
                path = Path(result['external_dependencies']['fixture_root'])
                self.assertTrue((path / 'package/entry.py').is_file())
                self.assertEqual(sources.verify_source('fixture', path), path)
                self.assertEqual(sources.prepare_sources(('fixture',), root=folder / 'installed',
                    execute=True, accept_terms=True)['status'], 'completed')
                download.assert_not_called()
                (path / 'package/entry.py').write_text('changed')
                with self.assertRaisesRegex(ValueError, 'changed'):
                    sources.prepare_sources(('fixture',), root=folder / 'installed', execute=True, accept_terms=True)
                self.assertEqual((path / 'package/entry.py').read_text(), 'changed')

    def test_unsafe_archive_is_rejected_without_publishing_a_prefix(self):
        for member, link in (('../outside', False), ('package/entry.py', True)):
            with self.subTest(member=member, link=link), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                spec = self.fixture(folder, member=member, link=link)
                with patch.dict(sources.SOURCES, {'fixture': spec}, clear=True):
                    with self.assertRaisesRegex(ValueError, 'Unsafe'):
                        sources.prepare_sources(('fixture',), root=folder / 'installed', offline_root=folder,
                                                execute=True, accept_terms=True)
                    self.assertFalse(sources.profile_path('fixture', folder / 'installed').exists())

    def test_unrecorded_source_file_invalidates_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            with patch.dict(sources.SOURCES, {'fixture': self.fixture(folder)}, clear=True):
                record = sources.prepare_sources(('fixture',), root=folder / 'installed', offline_root=folder,
                                                 execute=True, accept_terms=True)
                root = Path(record['external_dependencies']['fixture_root'])
                (root / 'package/injected.py').write_text('extra')
                with self.assertRaisesRegex(ValueError, 'unrecorded'):
                    sources.verify_source('fixture', root)


class SetupDeliveryTests(unittest.TestCase):
    def test_backend_dependency_requests_retain_the_external_interpreter(self):
        report = SimpleNamespace(statuses=[self.missing('python:tractcloud', 'python', 'tractcloud')])
        plan = build_setup_plan(report, backend_paths={'tractcloud_python_executable': Path('/chosen/python')})
        self.assertEqual({item['python_executable'] for item in plan['python_requests']}, {'/chosen/python'})
        with tempfile.TemporaryDirectory() as tmp:
            owner = Mock()
            owner._latestEnvironmentReportDir = tmp
            owner._setup_python_command.side_effect = lambda executable: (executable, [])
            commands, _ = python_setup_commands(owner, plan['python_packages'], requests=plan['python_requests'])
            self.assertEqual(len(commands), 1)
            self.assertEqual(commands[0][0], '/chosen/python')
            payload = json.loads(Path(commands[0][1][commands[0][1].index('--request') + 1]).read_text())
            self.assertIn('numpy>=1.21', payload['requirements'])
            self.assertIn('vtk>=9', payload['requirements'])

    def test_windows_does_not_prepare_a_bash_only_wma_source_as_a_runtime(self):
        report = SimpleNamespace(statuses=[self.missing('command:wm_apply_ORG_atlas_to_subject.sh', 'wma')])
        plan = build_setup_plan(report, system='Windows', machine='AMD64')
        self.assertEqual(plan['backend_profiles'], [])
        self.assertEqual(plan['unsupported_backend_profiles'], ['wma'])
        self.assertEqual(len(plan['manual_items']), 1)

    def test_old_macos_is_unqualified_before_any_installation(self):
        from shared.managed_runtimes import install_plan
        with patch('platform.system', return_value='Darwin'), patch('platform.machine', return_value='x86_64'), \
                patch('platform.mac_ver', return_value=('13.6', (), '')):
            self.assertFalse(install_plan(('mrtrix',))['platform_supported'])

    def test_unpublished_missing_surface_bundle_is_not_offered_as_downloadable(self):
        item = SimpleNamespace(key='surface_postprocess_runtime', label='Surface components', install_group='surface-runtime',
            kind='setting', required=True, status='missing', targets=('surface_postprocess',),
            install_hint='Use a verified offline bundle', details={'build_required': True})
        with tempfile.TemporaryDirectory() as tmp:
            plan = build_setup_plan(SimpleNamespace(statuses=[item]), system='Linux', machine='x86_64',
                                    backend_paths={'model_assets_dir': tmp})
        self.assertFalse(plan['surface_prebuilt_required'])
        self.assertTrue(plan['surface_bundle_unavailable'])
        self.assertEqual(plan['manual_items'][0]['key'], 'surface_postprocess_runtime')

    def missing(self, key, group, package=None):
        return SimpleNamespace(key=key, label=key, install_group=group, kind='python' if package else 'setting',
                               required=True, status='missing', targets=('bundles',), install_hint='',
                               details={'package': package} if package else {})

    def test_bundle_sources_are_prepared_without_vcs_package_installs(self):
        report = SimpleNamespace(statuses=(self.missing('python:tract_querier', 'python', 'tract_querier'),
            self.missing('wma_root', 'wma'), self.missing('python:tractcloud', 'python', 'tractcloud')))
        plan = build_setup_plan(report)
        self.assertEqual(plan['backend_profiles'], ['wmql', 'wma', 'tractcloud'])
        self.assertFalse(any('git+' in package for package in plan['python_packages']))
        self.assertIn('torch>=2.3.1', plan['python_packages'])
        self.assertIn('PyTorch', plan['extensions'])
        self.assertEqual(plan['manual_items'], [])

    def test_explicit_custom_backend_path_is_not_replaced(self):
        report = SimpleNamespace(statuses=(self.missing('wma_root', 'wma'),))
        plan = build_setup_plan(report, backend_paths={'wma_root': Path('/custom/research-version')})
        self.assertEqual(plan['backend_profiles'], [])
        self.assertEqual(plan['manual_items'][0]['key'], 'wma_root')

    def test_same_package_in_different_interpreters_has_two_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            owner = Mock()
            owner._latestEnvironmentReportDir = tmp
            a, b = Path(tmp) / 'a/python', Path(tmp) / 'b/python'
            items = [SimpleNamespace(kind='python', details={'package': 'nibabel', 'python_executable': str(path)})
                     for path in (a, b)]
            owner._lastEnvironmentResult = SimpleNamespace(statuses=items)
            owner._setup_python_command.side_effect = lambda python: (python, [])
            commands, receipts = python_setup_commands(owner, ('nibabel',))
            self.assertEqual({command[0] for command in commands}, {str(a), str(b)})
            self.assertEqual(len(receipts), 2)
            self.assertTrue(all(command[2]['PIP_ONLY_BINARY'] == ':all:' for command in commands))

    def test_shared_torch_keeps_the_strongest_requested_minimum(self):
        with tempfile.TemporaryDirectory() as tmp:
            owner = Mock()
            owner._latestEnvironmentReportDir = tmp
            owner._lastEnvironmentResult = SimpleNamespace(statuses=())
            owner._setup_python_command.return_value = ('Slicer', ['--launch', 'PythonSlicer'])
            commands, _ = python_setup_commands(owner, ('torch>=2.0.0', 'torch>=2.3.1'))
            request = Path(commands[0][1][commands[0][1].index('--request') + 1])
            payload = json.loads(request.read_text())
            self.assertEqual(payload['torch_minimum'], '2.3.1')
            self.assertEqual(payload['requirements'], [])

    def test_macos_mrtrix_is_separate_from_unqualified_pnl_fsl_profiles(self):
        report = SimpleNamespace(statuses=(self.missing('command:mrconvert', 'mrtrix'),
                                          self.missing('brain_mask_backend', 'brain_mask')))
        plan = build_setup_plan(report, system='Darwin', machine='x86_64')
        self.assertEqual(plan['managed_profiles'], ['mrtrix'])
        self.assertEqual(plan['unsupported_profiles'], ['pnl_cnn'])
        from shared.managed_runtimes import bootstrap_artifact, profile_spec
        with patch('platform.system', return_value='Darwin'), patch('platform.mac_ver', return_value=('14.5', (), '')), \
                patch('shared.managed_runtimes.host_machine', return_value='x86_64'):
            self.assertIn('MacOSX-x86_64', bootstrap_artifact()[0])
            self.assertIn('mrtrix3=3.0.8=hee5d243_0', profile_spec('mrtrix')['packages'])


class PrebuiltSurfaceTests(unittest.TestCase):
    def setUp(self):
        from shared.prebuilt_surface import bundle_for_platform
        selected = bundle_for_platform('Linux', 'x86_64')
        stub = patch('shared.prebuilt_surface.bundle_for_platform', return_value=selected)
        stub.start()
        self.addCleanup(stub.stop)

    def test_plan_and_permission_gate_need_no_compiler_or_directory(self):
        from shared.prebuilt_surface import prepare
        with tempfile.TemporaryDirectory() as tmp:
            prefix, assets = Path(tmp) / 'runtime', Path(tmp) / 'assets'
            self.assertEqual(prepare(prefix=prefix, asset_root=assets)['method'], 'prebuilt')
            with self.assertRaises(PermissionError):
                prepare(prefix=prefix, asset_root=assets, execute=True)
            self.assertFalse(prefix.exists())
            self.assertFalse(assets.exists())

    def test_unpublished_archive_never_downloads_or_builds_sources(self):
        from shared.prebuilt_surface import prepare
        with tempfile.TemporaryDirectory() as tmp, patch('urllib.request.urlopen') as download:
            with self.assertRaisesRegex(FileNotFoundError, 'offline archive'):
                prepare(prefix=Path(tmp) / 'runtime', asset_root=Path(tmp) / 'assets', execute=True, accept_terms=True)
            self.assertFalse((Path(tmp) / 'runtime').exists())
            download.assert_not_called()

    def test_catalog_must_match_current_source_recipe(self):
        from shared.prebuilt_surface import prepare, bundle_for_platform
        record = dict(bundle_for_platform('Linux', 'x86_64'), recipe_sha256='0' * 64)
        with patch('shared.prebuilt_surface.bundle_for_platform', return_value=record):
            with self.assertRaisesRegex(ValueError, 'source recipe'):
                prepare()


class FunctionalSetupTests(unittest.TestCase):
    def test_cpu_choice_does_not_become_an_automatic_cuda_test(self):
        status = SimpleNamespace(kind='python', required=True, status='available', targets=('segmentation',),
                                 details={'import': 'torch', 'python_executable': 'python'})
        request = requests_for_report(SimpleNamespace(statuses=[status], targets=('segmentation',)),
                                      device_choices={'ddseg_device': 'cpu'})[0]
        self.assertEqual(request['devices'], ['cpu'])
    def test_wma_functional_probe_imports_author_package_in_its_selected_python(self):
        status = SimpleNamespace(key='python:wma:joblib', kind='python', required=True, status='available',
                                 details={'import': 'joblib', 'python_executable': 'slicer-python'})
        report = SimpleNamespace(statuses=[status], targets=('wma',))
        request = requests_for_report(report, backend_paths={'wma_root': '/prepared/wma'})[0]
        self.assertIn('whitematteranalysis', request['imports'])
        self.assertEqual(request['pythonpath'], ['/prepared/wma'])
        self.assertEqual(request['python'], 'slicer-python')

    def test_selected_device_is_preserved_in_functional_request(self):
        statuses = [SimpleNamespace(required=True, status='available', kind='python',
                                   details={'import': 'torch', 'python_executable': 'python'}),
                    SimpleNamespace(required=True, status='available', kind='device',
                                   details={'requested_device': 'auto', 'resolved_device': 'cuda:1',
                                            'python_executable': 'python'})]
        request = requests_for_report(SimpleNamespace(statuses=statuses))[0]
        self.assertEqual(request['devices'], ['cuda:1'])

    def test_synthetic_volume_and_streamline_outputs_can_be_read_back(self):
        try:
            import nibabel
            import vtk
        except ImportError:
            self.skipTest('Optional scientific packages are unavailable')
        result = run_checks({'imports': ['numpy', 'nibabel', 'vtk']})
        self.assertEqual(result['status'], 'passed')
        self.assertIn('nifti:write_read', result['checks'])
        self.assertIn('vtk:streamline_write_read', result['checks'])

    def test_probes_use_only_required_available_packages_and_their_own_python(self):
        statuses = [SimpleNamespace(required=True, status='available', kind='python',
                                   details={'import': 'numpy', 'python_executable': name, 'pythonpath': [name + '-src']})
                    for name in ('a', 'b')]
        report = SimpleNamespace(statuses=statuses, targets=('validation',))
        requests = requests_for_report(report)
        self.assertEqual({r['python'] for r in requests}, {'a', 'b'})
        self.assertTrue(all(r['pythonpath'] == [r['python'] + '-src'] for r in requests))

    def test_invalid_model_hash_is_rejected_before_loading(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'model.pt'
            checkpoint.write_bytes(b'not a model')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                run_checks({'imports': ['torch'], 'models': [{'path': str(checkpoint), 'sha256': '0' * 64}]})


if __name__ == '__main__':
    unittest.main()
