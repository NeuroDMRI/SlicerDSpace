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
                self.assertIsNone(bundle_for_platform('Darwin', 'arm64'))
