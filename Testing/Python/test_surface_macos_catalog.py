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
        # b2 runtime lane: the arm64 runtime is rebuilt with a macOS 13.0 deployment target.
        with patch('platform.system', return_value='Darwin'):
            for version in ('12.7.6', ''):
                with patch('platform.mac_ver', return_value=(version, (), '')):
                    self.assertIsNone(bundle_for_platform('Darwin', 'arm64'))
            for version in ('13.0', '13.4', '14.7', '15.8.1'):
                with patch('platform.mac_ver', return_value=(version, (), '')):
                    record = bundle_for_platform('Darwin', 'arm64')
                    self.assertIsNotNone(record, version)
                    self.assertEqual(record['machine'], 'arm64')
                    self.assertEqual(record['min_macos'], '13.0')
                    self.assertEqual(record['asset'], 'runtime/surface/surface-runtime-darwin-arm64-macos13.tar.gz')
                    self.assertTrue(record['url'].endswith('/assets/surface-runtime-darwin-arm64-macos13.tar.gz'))
                    self.assertTrue(record['published'])
                    self.assertFalse(record['activation_key_included'])
