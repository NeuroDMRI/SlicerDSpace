"""Renaming does not discard user settings, caches or established launchers."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from shared.branding import LEGACY_NAME, LEGACY_ENV_PREFIX, mirror_environment, migrate_qsettings, reuse_legacy_path
from shared.model_assets import default_user_model_assets_root
from shared.user_config import default_user_config_path


class Settings:
    def __init__(self, values):
        self.values = dict(values)

    def allKeys(self):
        return list(self.values)

    def contains(self, key):
        return key in self.values

    def value(self, key):
        return self.values[key]

    def setValue(self, key, value):
        self.values[key] = value


class BrandingTests(unittest.TestCase):
    def test_environment_preserves_old_launchers_and_prefers_new_options(self):
        env = {LEGACY_ENV_PREFIX + 'DEVICE': 'cpu', 'SLICERDSPACE_DEVICE': 'cuda',
               LEGACY_ENV_PREFIX + 'USER_CONFIG': '/existing/config.json', 'OTHER': 'unchanged'}
        mirror_environment(env)
        self.assertEqual(env[LEGACY_ENV_PREFIX + 'DEVICE'], 'cuda')
        self.assertEqual(env['SLICERDSPACE_USER_CONFIG'], '/existing/config.json')
        self.assertEqual(env['OTHER'], 'unchanged')
        previous = dict(env)
        mirror_environment(env)
        self.assertEqual(previous, env)

    def test_preferences_are_copied_without_overwriting_new_or_unrelated_settings(self):
        settings = Settings({LEGACY_NAME + '/UserSettings/language': 'zh',
                             LEGACY_NAME + '/ExternalDependencies/python': '/existing/python',
                             'SlicerDSpace/UserSettings/language': 'en', 'Other/Setting': 'kept'})
        migrate_qsettings(settings)
        self.assertEqual(settings.value('SlicerDSpace/UserSettings/language'), 'en')
        self.assertEqual(settings.value('SlicerDSpace/ExternalDependencies/python'), '/existing/python')
        self.assertEqual(settings.value('Other/Setting'), 'kept')
        self.assertEqual(settings.value(LEGACY_NAME + '/UserSettings/language'), 'zh')

    def test_existing_config_and_assets_are_reused_without_moving_them(self):
        with tempfile.TemporaryDirectory() as folder:
            home = Path(folder)
            previous = home / '.config' / LEGACY_NAME / 'user_config.json'
            previous.parent.mkdir(parents=True)
            previous.write_text('{}')
            self.assertEqual(default_user_config_path(system='Linux', home=home, environ={}), previous)
            assets = home / '.local/share' / LEGACY_NAME / 'model-assets'
            assets.mkdir(parents=True)
            with patch('shared.model_assets.platform.system', return_value='Linux'), \
                 patch('shared.model_assets.Path.home', return_value=home), \
                 patch.dict(os.environ, {'XDG_DATA_HOME': ''}, clear=True):  # B2: the isolation keeps XDG_DATA_HOME; empty = unset
                self.assertEqual(default_user_model_assets_root(), assets)
            canonical = home / '.config/SlicerDSpace/user_config.json'
            canonical.parent.mkdir()
            canonical.write_text('{}')
            self.assertEqual(reuse_legacy_path(canonical), canonical)
            self.assertTrue(previous.is_file())

    def test_explicit_configuration_takes_priority_over_legacy_options(self):
        env = {'SLICERDSPACE_USER_CONFIG': '/new/config.json',
               LEGACY_ENV_PREFIX + 'USER_CONFIG': '/old/config.json'}
        self.assertEqual(default_user_config_path(environ=env), Path('/new/config.json'))
        self.assertEqual(default_user_config_path(environ={LEGACY_ENV_PREFIX + 'USER_CONFIG': '/old/config.json'}),
                         Path('/old/config.json'))


if __name__ == '__main__':
    unittest.main()
