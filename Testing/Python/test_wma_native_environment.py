"""Native ORG setup uses the selected Slicer Python and the verified source cache."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from shared.environment import DependencyStatus, EnvironmentDoctorConfig, build_dependency_specs
from shared.selected_setup import apply_setup_receipt, build_setup_plan


class NativeWmaEnvironmentTests(unittest.TestCase):
    def test_windows_auto_probes_the_configured_slicer_python_for_a_native_source(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            slicer = root / 'Slicer.exe'
            slicer_python = root / 'bin' / 'PythonSlicer.exe'
            slicer_python.parent.mkdir()
            slicer.touch()
            slicer_python.touch()
            config = EnvironmentDoctorConfig(
                targets=('wma',), repo_root=root, platform_name='Windows',
                slicer_executable=slicer, python_executable=root / 'external-python',
                wma_root=root / 'wma-source',
            )
            with patch.dict(os.environ, {'DSP_WMA_EXECUTION_BACKEND': 'auto'}):
                specs = {spec.key: spec for spec in build_dependency_specs(config)}

        self.assertNotIn('command:wm_apply_ORG_atlas_to_subject.sh', specs)
        self.assertNotIn('path:wma_atlas_python_bridge', specs)
        for name in ('scipy', 'numpy', 'nibabel', 'vtk'):
            with self.subTest(name=name):
                self.assertIn(f'python:wma:{name}', specs)
                self.assertEqual(specs[f'python:wma:{name}'].python_executable, slicer_python)
        self.assertEqual(specs['python:vtk'].python_executable, root / 'external-python')
        self.assertEqual(specs['path:wma_root_path'].path_kind, 'wma_native_source')

    def test_windows_auto_can_prepare_without_a_configured_checkout(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = EnvironmentDoctorConfig(
                targets=('wma',), repo_root=root, platform_name='Windows',
            )
            with (
                patch.dict(os.environ, {'DSP_WMA_EXECUTION_BACKEND': 'auto', 'WMA_ROOT': '',
                                        'SLICERDSPACE_WMA_ROOT': '', 'SLICERDSPACE_SLICER_EXECUTABLE': ''}),
                patch('shared.backend_sources.source_root', return_value=root),
                patch('shared.environment._common_slicer_executables', return_value=()),
                patch('tractography.wma._wma_root_on_path', return_value=None),
            ):
                specs = {spec.key: spec for spec in build_dependency_specs(config)}

        self.assertNotIn('command:wm_apply_ORG_atlas_to_subject.sh', specs)
        self.assertEqual(specs['path:wma_root_path'].path_kind, 'wma_native_source')
        self.assertEqual(specs['path:wma_root_path'].path.parent, root)
        self.assertIn('setting:wma_pythonslicer', specs)
        self.assertNotIn('python:wma:scipy', specs)

    def test_windows_diffusion_cli_keeps_the_legacy_route(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = EnvironmentDoctorConfig(
                targets=('wma',), repo_root=root, platform_name='Windows',
                wma_root=root / 'legacy', wma_export_diffusion_measures=True,
            )
            with patch.dict(os.environ, {'DSP_WMA_EXECUTION_BACKEND': 'auto'}):
                specs = {spec.key: spec for spec in build_dependency_specs(config)}

        self.assertIn('command:wm_apply_ORG_atlas_to_subject.sh', specs)
        self.assertEqual(specs['path:wma_root_path'].path_kind, 'dir')

    @staticmethod
    def source_status(status):
        return DependencyStatus(
            key='path:wma_root_path', kind='path', label='Native WMA source', status=status,
            required=True, targets=('wma',), install_group='wma',
            details={'execution_backend': 'native'},
        )

    def test_windows_prepares_native_wma_without_wsl(self):
        result = SimpleNamespace(statuses=(self.source_status('missing'),), targets=('wma',))
        plan = build_setup_plan(
            result, system='Windows', machine='AMD64', closure_targets=('wma',), wsl_distributions=[],
        )

        self.assertEqual(plan['backend_profiles'], ['wma'])
        self.assertEqual(plan['unsupported_backend_profiles'], [])
        self.assertEqual(plan['python_requests'], [])
        self.assertFalse(plan['windows_runtime']['needed'])
        self.assertNotIn('path:wma_root_path', [item['key'] for item in plan['manual_items']])

    def test_native_source_receipt_persists_wma_root(self):
        from shared.backend_sources import SOURCES, file_hash, profile_path
        from shared.user_config import load_user_config

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = profile_path('wma', root)
            spec = SOURCES['wma']
            entry = source / spec['entry']
            entry.parent.mkdir(parents=True)
            entry.touch()
            (source / '.slicerdspace-source.json').write_text(
                json.dumps({'profile': 'wma', 'revision': spec['revision'],
                            'archive_sha256': spec['sha256'], 'files': {spec['entry']: file_hash(entry)}}),
                encoding='utf-8',
            )
            step = {'id': 'backend_sources', 'profiles': ['wma'], 'root': str(root)}
            receipt = {'status': 'completed', 'profiles': ['wma'],
                       'external_dependencies': {'wma_root': str(source)}}
            with patch('shared.user_config.default_user_config_path', return_value=root / 'config.json'):
                apply_setup_receipt(step, receipt)
                self.assertEqual(load_user_config()['external_dependencies']['wma_root'], str(source))

    def test_windows_wsl_setup_preserves_a_separate_native_wma(self):
        fsl_status = DependencyStatus(
            key='command:eddy', kind='command', label='eddy', status='missing',
            required=True, targets=('preprocessing',), install_group='fsl',
        )
        result = SimpleNamespace(
            statuses=(self.source_status('available'), fsl_status), targets=('preprocessing', 'wma'),
        )
        plan = build_setup_plan(
            result, system='Windows', machine='AMD64', closure_targets=result.targets,
            wsl_distributions=[{'name': 'Ubuntu', 'version': 2, 'default': True}],
        )

        self.assertIn('-PreserveWmaSettings', plan['windows_runtime']['command'])
        self.assertIn('-SkipLargeDownloads', plan['windows_runtime']['command'])


if __name__ == '__main__':
    unittest.main()
