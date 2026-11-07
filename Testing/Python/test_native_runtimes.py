"""No-network tests for verified, fail-closed native runtime provisioning."""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
import zipfile
import tarfile
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared.native_runtimes import NativeRuntimeUnavailable, select_bundle, install_bundle


class NativeRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root / 'sample.zip'
        with zipfile.ZipFile(self.archive, 'w') as z:
            z.writestr('bin/tckgen.exe', b'binary fixture')
        self.record = {
            'name': 'mrtrix', 'version': '3.0.8', 'system': 'Windows',
            'machine': 'x86_64', 'format': 'zip', 'published': True,
            'sha256': hashlib.sha256(self.archive.read_bytes()).hexdigest(),
            'license_url': 'https://example.org/licence', 'commands': ['bin/tckgen.exe'],
        }

    def test_native_windows_bundle_without_wsl_conda_or_msys2(self):
        record = select_bundle({'bundles': [self.record]}, 'mrtrix', system='Windows', machine='AMD64')
        dest = install_bundle(record, archive=self.archive, root=self.root/'runtime', accept_terms=True)
        self.assertEqual((dest/'bin/tckgen.exe').read_bytes(), b'binary fixture')
        self.assertIn('Windows', (dest/'native-runtime.json').read_text())

    def test_missing_platform_never_falls_back_to_linux(self):
        with self.assertRaises(NativeRuntimeUnavailable):
            select_bundle({'bundles': [self.record]}, 'mrtrix', system='Darwin', machine='arm64')

    def test_unpublished_build_cannot_be_silently_installed(self):
        with self.assertRaises(NativeRuntimeUnavailable):
            select_bundle({'bundles': [{**self.record, 'published': False}]}, 'mrtrix', system='Windows', machine='x86_64')

    def test_licence_required_and_checksum_checked(self):
        with self.assertRaises(PermissionError):
            install_bundle(self.record, archive=self.archive, root=self.root/'runtime')
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            install_bundle({**self.record, 'sha256': '0'*64}, archive=self.archive, root=self.root/'runtime', accept_terms=True)

    def test_rejects_path_traversal_and_links(self):
        bad = self.root / 'bad.zip'
        with zipfile.ZipFile(bad, 'w') as z:
            z.writestr('../oops', b'invalid')
        record = {**self.record, 'sha256': hashlib.sha256(bad.read_bytes()).hexdigest()}
        with self.assertRaisesRegex(ValueError, 'Unsafe archive'):
            install_bundle(record, archive=bad, root=self.root/'runtime', accept_terms=True)

    def test_missing_declared_executable_fails(self):
        with self.assertRaisesRegex(ValueError, 'Missing native executable'):
            install_bundle({**self.record, 'commands': ['bin/eddy.exe']}, archive=self.archive,
                           root=self.root/'runtime', accept_terms=True)

    def test_rejects_unsafe_bundle_name(self):
        with self.assertRaisesRegex(ValueError, 'Invalid native runtime name'):
            install_bundle({**self.record, 'name': '../escape'}, archive=self.archive,
                           root=self.root/'runtime', accept_terms=True)

    def test_existing_install_is_not_overwritten(self):
        dest = install_bundle(self.record, archive=self.archive, root=self.root/'runtime', accept_terms=True)
        with self.assertRaises(FileExistsError):
            install_bundle(self.record, archive=self.archive, root=self.root/'runtime', accept_terms=True)
        self.assertTrue((dest/'bin/tckgen.exe').is_file())


if __name__ == '__main__':
    unittest.main()
