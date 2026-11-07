"""Local surface components do not require the full suite or implicit setup."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from shared.surface_runtime import resolve_surface_runtime, inspect_surface_runtime, default_surface_runtime
from shared.environment import EnvironmentDoctorConfig, build_dependency_specs, run_environment_doctor
from shared.selected_setup import build_setup_plan
from shared.environment_check import config_from_dict
from shared.dependency_config import add_dependency_config_arguments

ROOT = Path(__file__).resolve().parents[2]


class SurfaceRuntimeTests(unittest.TestCase):
    def test_raw_slicer_python_is_launched_through_its_runtime_wrapper(self):
        from anatomy.surface.backend_bridge import python_executable
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'PythonSlicer').touch()
            self.assertEqual(python_executable(folder / 'python-real'), str(folder / 'PythonSlicer'))
            self.assertEqual(python_executable(folder / 'custom-python'), str(folder / 'custom-python'))

    def test_default_runtime_matches_upstream_recipe_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ', {'XDG_CACHE_HOME': tmp}):
            runtime = default_surface_runtime()
            self.assertTrue(str(runtime).startswith(tmp))
            self.assertIn('native', runtime.parts)
            self.assertFalse(runtime.exists())

    def test_explicit_runtime_wins_without_discovering_full_install(self):
        with tempfile.TemporaryDirectory() as tmp, patch('shared.environment.resolve_freesurfer_home') as discover:
            selected = resolve_surface_runtime(Path(tmp), Path('/old/full-install'))
            self.assertEqual(selected, ('native', Path(tmp).resolve()))   # macOS: /var is /private/var
            discover.assert_not_called()

    def test_auto_retains_an_existing_install_or_uses_user_cache(self):
        with patch('shared.environment.resolve_freesurfer_home', return_value=Path('/full/install')):
            self.assertEqual(resolve_surface_runtime(), ('freesurfer', Path('/full/install')))
        with patch('shared.environment.resolve_freesurfer_home', return_value=None), \
                patch('shared.surface_runtime.default_surface_runtime', return_value=Path('/local/components')):
            self.assertEqual(resolve_surface_runtime(), ('native', Path('/local/components')))

    def test_missing_native_components_do_not_require_freesurfer_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            specs = build_dependency_specs(EnvironmentDoctorConfig(targets=('surface_postprocess',),
                repo_root=ROOT, postprocess_runtime=Path(tmp) / 'not-built'))
            self.assertFalse(any(spec.install_group == 'freesurfer' for spec in specs))
            self.assertFalse(any(spec.key == 'env:freesurfer_home' for spec in specs))

    def test_fresh_probe_does_not_download_compile_or_create_a_missing_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            activation = root / 'activation.txt'
            activation.write_text('synthetic nonempty activation fixture; not a valid key')
            missing = root / 'not-built'
            record = inspect_surface_runtime(missing, license_file=activation)
            self.assertFalse(record['components_available'])
            self.assertTrue(record['license_available'])
            self.assertTrue(record['build_required'])
            self.assertFalse(missing.exists())

    def test_read_only_doctor_separates_components_and_personal_activation(self):
        fixture = dict(backend='native', components_available=True, license_available=False, build_required=False)
        with patch('shared.environment.build_dependency_specs', return_value=()), \
                patch('shared.surface_runtime.inspect_surface_runtime', return_value=fixture):
            result = run_environment_doctor(EnvironmentDoctorConfig(targets=('surface_postprocess',), repo_root=ROOT))
        statuses = {item.key: item for item in result.statuses}
        self.assertEqual(statuses['surface_postprocess_runtime'].status, 'available')
        self.assertEqual(statuses['surface_postprocess_license'].status, 'missing')
        self.assertEqual(result.blocker_count, 1)
        plan = build_setup_plan(result, system='Linux', machine='x86_64')
        self.assertFalse(plan['surface_build_required'])
        self.assertIn('surface_postprocess_license', [item['key'] for item in plan['manual_items']])

    def test_prebuilt_install_is_selected_without_implicit_compilation(self):
        item = SimpleNamespace(key='surface_postprocess_runtime', required=True, status='missing', kind='setting',
            targets=('surface_postprocess',), details={'build_required': True}, install_group='surface-runtime',
            label='components', install_hint='prepare')
        result = SimpleNamespace(statuses=(item,))
        from shared.prebuilt_surface import bundle_for_platform
        bundle = dict(bundle_for_platform('Linux', 'x86_64'), published=True)
        with patch('shared.prebuilt_surface.bundle_for_platform', return_value=bundle):
            plan = build_setup_plan(result, system='Linux', machine='x86_64')
        self.assertFalse(plan['surface_build_required'])
        self.assertTrue(plan['surface_prebuilt_required'])
        for system, machine in (('Darwin', 'arm64'), ('Windows', 'AMD64')):
            plan = build_setup_plan(result, system=system, machine=machine)
            self.assertFalse(plan['surface_prebuilt_required'])
            self.assertTrue(plan['manual_items'])
        item.details['build_required'] = False
        self.assertFalse(build_setup_plan(result, system='Linux', machine='x86_64')['surface_build_required'])

    def test_cli_and_worker_keep_new_paths(self):
        parser = argparse.ArgumentParser()
        add_dependency_config_arguments(parser)
        args = parser.parse_args(['--postprocess-runtime', '/components', '--postprocess-license-file', '/activation'])
        self.assertEqual(args.postprocess_runtime, Path('/components'))
        worker = config_from_dict(dict(targets=['surface_postprocess'], repo_root=str(ROOT),
            postprocess_runtime='/components', postprocess_license_file='/activation'))
        self.assertEqual(worker.postprocess_license_file, Path('/activation'))

    def test_gui_tooltips_describe_the_optional_full_installation(self):
        from shared.ui_onboarding import TRANSLATIONS
        self.assertIn('verified native', TRANSLATIONS['en']['surface_postprocess_tooltip'])
        self.assertIn('full FreeSurfer installation is optional', TRANSLATIONS['en']['surface_postprocess_tooltip'])
        self.assertIn('经校验的原生', TRANSLATIONS['zh']['surface_postprocess_tooltip'])
        self.assertIn('不强制安装完整 FreeSurfer', TRANSLATIONS['zh']['surface_postprocess_tooltip'])

    def test_builder_dry_run_and_unapproved_execute_never_prepare_resources(self):
        from scripts.setup_surface_runtime import prepare
        with tempfile.TemporaryDirectory() as tmp, patch('scripts.setup_optional_assets.prepare') as resources, \
                patch('platform.system', return_value='Linux'), patch('platform.machine', return_value='x86_64'):
            # B11: the catalog holds one prebuilt bundle, Linux x86_64. Patching only the system made the test fail on
            # Linux arm64 and on a Mac (Apple Silicon); it now fixes both parts of the platform.
            prefix = Path(tmp) / 'components'
            self.assertEqual(prepare(prefix=prefix)['status'], 'planned')
            with self.assertRaises(PermissionError):
                prepare(prefix=prefix, execute=True)
            self.assertFalse(prefix.exists())
            resources.assert_not_called()

    def test_builder_refuses_automatic_preparation_outside_linux(self):
        from scripts.setup_surface_runtime import prepare
        for system in ('Darwin', 'Windows'):
            with tempfile.TemporaryDirectory() as tmp, patch('platform.system', return_value=system), \
                    patch('scripts.setup_optional_assets.prepare') as resources:
                prefix = Path(tmp) / 'components'
                with self.assertRaisesRegex(ValueError, 'No qualified prebuilt surface bundle'):
                    prepare(prefix=prefix)
                self.assertFalse(prefix.exists())
                resources.assert_not_called()

    def test_source_recipe_and_data_manifest_are_in_install_contract_not_large_archive(self):
        from scripts.extension_runtime.runtime_resources import runtime_files
        files = set(runtime_files(ROOT))
        base = 'anatomy/surface/ddsurfer_backend/postprocessing/'
        for name in ('core/build.py', 'core/cmake/CMakeLists.txt', 'assets/manifest.json', 'assets/LICENSE.txt'):
            self.assertIn(base + name, files)
        self.assertNotIn(base + 'assets/atlas.tar.gz', files)
        self.assertFalse((ROOT / base / 'assets/atlas.tar.gz').exists())


if __name__ == '__main__':
    unittest.main()
