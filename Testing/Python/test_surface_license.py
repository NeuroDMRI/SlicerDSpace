import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared.surface_license import (
    import_surface_license,
    managed_surface_license_path,
    resolve_surface_license,
    usable_license,
)
from shared.ui_surface_license import prepare_surface_license


class SurfaceLicenseTests(unittest.TestCase):
    def test_explicit_and_environment_choices_stay_authoritative(self):
        with TemporaryDirectory() as tmp:
            home = Path(tmp).resolve()  # F4: macOS /var is a symlink to /private/var
            activation = home / '.freesurfer/license.txt'
            activation.parent.mkdir()
            activation.write_text('synthetic activation fixture')
            self.assertEqual(resolve_surface_license(home=home, environ={}), activation)
            wrong = home / 'missing-license.txt'
            self.assertEqual(resolve_surface_license(wrong, home=home, environ={}), wrong)
            self.assertEqual(resolve_surface_license(home=home, environ={'FS_LICENSE': str(wrong)}), wrong)
            activation.unlink()
            self.assertIsNone(resolve_surface_license(home=home, environ={}))

    def test_existing_install_hidden_license_and_alias(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()  # F4: macOS /var is a symlink to /private/var
            activation = root / '.license'
            activation.write_text('synthetic activation fixture')
            self.assertEqual(resolve_surface_license(freesurfer_home=root, home=root, environ={}), activation)
            self.assertEqual(resolve_surface_license(home=root, environ={'FREESURFER_LICENSE': str(activation)}), activation)

    def test_reject_unreadable_or_invalid_file_without_claiming_activation_validity(self):
        self.assertFalse(usable_license(None))
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'license.txt'
            for data in (b'', b' \n', b'\0binary', b'\xff', b'x' * 65537):
                path.write_bytes(data)
                self.assertFalse(usable_license(path))
            path.write_text('synthetic nonempty text; not an activation key')
            self.assertTrue(usable_license(path))
            self.assertFalse(usable_license(Path(tmp)))

    def test_import_surface_license_copies_into_private_user_config_and_survives_source_deletion(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            downloads = root / 'Downloads'
            downloads.mkdir()
            src = downloads / 'license.txt'
            src.write_text('user@example.edu\n12345\n*Cexample\n')
            cfg_env = {'SLICERDSPACE_USER_CONFIG': str(root / 'cfg/user_config.json')}
            copied = import_surface_license(src, home=root, environ=cfg_env)
            expected_dest = managed_surface_license_path(home=root, environ=cfg_env).resolve()
            self.assertEqual(copied, expected_dest)
            self.assertTrue(copied.is_file())
            self.assertEqual(copied.read_text(), 'user@example.edu\n12345\n*Cexample\n')
            # Removing original file from Downloads does not break resolution.
            src.unlink()
            self.assertTrue(usable_license(copied))
            self.assertEqual(resolve_surface_license(home=root, environ=cfg_env), copied)

    def owner_and_report(self, targets=('surface_postprocess',), explicit=None, available=False):
        owner = Mock()
        owner._environment_dependency_path_kwargs.return_value = {'postprocess_license_file': explicit}
        owner._environmentPathLineEdits = {'postprocess_license_file': Mock()}
        report = SimpleNamespace(targets=targets, statuses=[SimpleNamespace(
            key='surface_postprocess_license', status='available' if available else 'missing')])
        return owner, report

    def test_unselected_route_never_prompts_or_scans(self):
        owner, report = self.owner_and_report(('microstructure',))
        with patch('shared.ui_surface_license.choose_license') as ask:
            self.assertIs(prepare_surface_license(owner, report), report)
        ask.assert_not_called()
        owner._environment_dependency_path_kwargs.assert_not_called()

    def test_import_copies_license_sets_fs_license_and_refreshes_report(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            src = root / 'Downloads/license.txt'
            src.parent.mkdir()
            src.write_text('synthetic private content')
            cfg_file = root / 'cfg/user_config.json'
            expected_dest = (cfg_file.parent / 'freesurfer/license.txt').resolve()
            owner, report = self.owner_and_report()
            env_patch = {'SLICERDSPACE_USER_CONFIG': str(cfg_file)}
            with patch.dict(os.environ, env_patch, clear=False):
                os.environ.pop('FS_LICENSE', None)
                with patch('shared.ui_surface_license.resolve_surface_license', return_value=None), \
                     patch('shared.ui_surface_license.choose_license', return_value=str(src)):
                    self.assertIs(prepare_surface_license(owner, report), owner._run_environment_dependency_check.return_value)
                owner._save_environment_path_setting.assert_called_once_with('postprocess_license_file', str(expected_dest))
                self.assertEqual(os.environ.get('FS_LICENSE'), str(expected_dest))
                owner._environmentPathLineEdits['postprocess_license_file'].setText.assert_called_once_with(str(expected_dest))
                owner._run_environment_dependency_check.assert_called_once_with()
                src.unlink()
                self.assertTrue(usable_license(expected_dest))
                self.assertEqual(expected_dest.read_text(), 'synthetic private content')
                # Next run when stale explicit path is passed automatically recovers the private copy without prompting.
                owner2, report2 = self.owner_and_report(explicit=str(src), available=False)
                with patch('shared.ui_surface_license.choose_license') as ask2:
                    prepare_surface_license(owner2, report2)
                ask2.assert_not_called()
                owner2._save_environment_path_setting.assert_called_once_with('postprocess_license_file', str(expected_dest))

    def test_cancel_does_not_change_settings_or_start_installation(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            cfg_file = root / 'cfg/user_config.json'
            owner, report = self.owner_and_report()
            with patch.dict(os.environ, {'SLICERDSPACE_USER_CONFIG': str(cfg_file)}, clear=False), \
                 patch('shared.ui_surface_license.resolve_surface_license', return_value=None), \
                 patch('shared.ui_surface_license.choose_license', return_value=None):
                self.assertIsNone(prepare_surface_license(owner, report))
            owner._save_environment_path_setting.assert_not_called()
            owner._run_environment_dependency_check.assert_not_called()

    def test_auto_discovery_is_saved_and_existing_configuration_is_reused(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp).resolve() / 'license.txt'
            path.write_text('synthetic activation fixture')
            owner, report = self.owner_and_report(available=True)
            with patch.dict(os.environ, {'FS_LICENSE': str(path)}, clear=False), \
                 patch('shared.ui_surface_license.resolve_surface_license', return_value=path), \
                 patch('shared.ui_surface_license.choose_license') as ask:
                prepare_surface_license(owner, report)
            ask.assert_not_called()
            owner._save_environment_path_setting.assert_called_once()
            owner, report = self.owner_and_report(explicit=path, available=True)
            with patch.dict(os.environ, {'FS_LICENSE': str(path)}, clear=False), \
                 patch('shared.ui_surface_license.choose_license') as ask:
                self.assertIs(prepare_surface_license(owner, report), report)
            ask.assert_not_called()
            owner._save_environment_path_setting.assert_not_called()


if __name__ == '__main__':
    unittest.main()
