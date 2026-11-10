import unittest
from unittest.mock import patch
from shared.prebuilt_surface import bundle_for_platform


class SurfaceMacCatalogTests(unittest.TestCase):
    def test_darwin_candidate_is_offline_only_and_requires_macos_15(self):
        with patch('platform.system', return_value='Darwin'):
            for version in ('14.5', '13.0', ''):
                with patch('platform.mac_ver', return_value=(version, (), '')):
                    self.assertIsNone(bundle_for_platform('Darwin', 'x86_64'))
            with patch('platform.mac_ver', return_value=('15.6.1', (), '')):
                record = bundle_for_platform('Darwin', 'x86_64')
                self.assertFalse(record['published'])
                self.assertEqual(record['min_macos'], '15.0')
                self.assertFalse(record['activation_key_included'])

    def test_apple_silicon_selects_its_own_published_candidate(self):
        with patch('platform.system', return_value='Darwin'):
            with patch('platform.mac_ver', return_value=('14.7', (), '')):
                self.assertIsNone(bundle_for_platform('Darwin', 'arm64'))
            with patch('platform.mac_ver', return_value=('15.8.1', (), '')):
                record = bundle_for_platform('Darwin', 'arm64')
                self.assertIsNotNone(record)
                self.assertEqual(record['machine'], 'arm64')
                self.assertEqual(record['asset'], 'runtime/surface/surface-runtime-darwin-arm64.tar.gz')
                self.assertTrue(record['published'])
                self.assertFalse(record['activation_key_included'])
