"""Private activation discovery and selected-route GUI preparation."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from shared.surface_license import resolve_surface_license, usable_license
from shared.ui_surface_license import prepare_surface_license


class SurfaceLicenseTests(unittest.TestCase):
    def test_known_locations_and_explicit_precedence(self):
        with TemporaryDirectory() as tmp:
            home = Path(tmp).resolve()  # F4: macOS /var is a symlink to /private/var
            activation = home / '.licenses/freesurfer/license.txt'
            activation.parent.mkdir(parents=True)
            activation.write_text('synthetic activation fixture')
            self.assertEqual(resolve_surface_license(home=home, environ={}), activation)
            wrong = home / 'missing.txt'
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

    def test_import_saves_only_path_and_refreshes_report(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'license.txt'
            path.write_text('synthetic private content')
            owner, report = self.owner_and_report()
            with patch('shared.ui_surface_license.resolve_surface_license', return_value=None), \
                 patch('shared.ui_surface_license.choose_license', return_value=str(path)):
                self.assertIs(prepare_surface_license(owner, report), owner._run_environment_dependency_check.return_value)
            owner._save_environment_path_setting.assert_called_once_with('postprocess_license_file', str(path))
            self.assertEqual(path.read_text(), 'synthetic private content')
            owner._run_environment_dependency_check.assert_called_once_with()

    def test_cancel_does_not_change_settings_or_start_installation(self):
        owner, report = self.owner_and_report()
        with patch('shared.ui_surface_license.resolve_surface_license', return_value=None), \
             patch('shared.ui_surface_license.choose_license', return_value=None):
            self.assertIsNone(prepare_surface_license(owner, report))
        owner._save_environment_path_setting.assert_not_called()
        owner._run_environment_dependency_check.assert_not_called()

    def test_auto_discovery_is_saved_and_existing_configuration_is_reused(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'license.txt'
            path.write_text('synthetic activation fixture')
            owner, report = self.owner_and_report(available=True)
            with patch('shared.ui_surface_license.resolve_surface_license', return_value=path), \
                 patch('shared.ui_surface_license.choose_license') as ask:
                prepare_surface_license(owner, report)
            ask.assert_not_called()
            owner._save_environment_path_setting.assert_called_once()
            owner, report = self.owner_and_report(explicit=path, available=True)
            with patch('shared.ui_surface_license.choose_license') as ask:
                self.assertIs(prepare_surface_license(owner, report), report)
            ask.assert_not_called()
            owner._save_environment_path_setting.assert_not_called()


if __name__ == '__main__':
    unittest.main()
