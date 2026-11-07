"""Keep public LICENSE.txt distinct from private activation on macOS volumes."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class SurfaceNoticeCaseTests(unittest.TestCase):
    def test_notice_survives_case_insensitive_exists_but_private_key_is_rejected(self):
        backend = Path(__file__).resolve().parents[2] / 'anatomy/surface/ddsurfer_backend'
        with patch.dict(sys.modules), patch.object(sys, 'path', [str(backend), *sys.path]):
            # Use the worker's isolated namespace, as probe_surface_runtime does.
            from postprocessing.core import runtime
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                notice = root / 'LICENSE.txt'
                notice.write_text('public software notice\n')
                manifest = dict(schema=runtime.SCHEMA, source_revision=runtime.FS_REVISION,
                    activation_key_included=False, system='Darwin', machine='x86_64',
                    files={'LICENSE.txt': dict(bytes=notice.stat().st_size,
                        sha256=hashlib.sha256(notice.read_bytes()).hexdigest())})
                (root / 'runtime.json').write_text(json.dumps(manifest))
                original_exists = Path.exists
                def insensitive_exists(path):
                    if path == root / 'license.txt':
                        return True  # APFS normally aliases the uppercase notice.
                    return original_exists(path)
                with patch.object(runtime, 'host', return_value=('Darwin', 'x86_64')), \
                        patch.object(runtime, 'required_files', return_value=['LICENSE.txt']), \
                        patch.object(runtime, 'COMMANDS', ()), patch.object(Path, 'exists', insensitive_exists):
                    self.assertEqual(runtime.validate_runtime(root), manifest)
                    (root / '.license').write_text('synthetic private activation fixture')
                    with self.assertRaisesRegex(ValueError, 'private activation key'):
                        runtime.validate_runtime(root)

    def test_native_probe_keeps_components_available_without_private_activation(self):
        from scripts.probe_surface_runtime import probe
        from shared.surface_license import resolve_surface_license

        backend = Path(__file__).resolve().parents[2] / 'anatomy/surface/ddsurfer_backend'
        with tempfile.TemporaryDirectory() as folder:
            home = Path(folder).resolve()
            runtime = home / 'native'
            runtime.mkdir()
            notice = runtime / 'LICENSE.txt'
            notice.write_text('public software notice\n')
            alias = runtime / 'license.txt'
            original_is_file = Path.is_file
            original_stat = Path.stat
            original_read_bytes = Path.read_bytes

            def notice_on_apfs(path):
                return notice if path == alias else path

            args = SimpleNamespace(home=runtime, backend='native', license_file=None,
                                   hemi='both', atlases='aparc,aparc.a2009s')
            environment = {'FS_LICENSE': '', 'FREESURFER_LICENSE': '', 'FREESURFER_HOME': ''}
            with patch.dict(sys.modules), patch.object(sys, 'path', [str(backend), *sys.path]):
                from postprocessing.core import runtime as core
                with (
                    patch.object(core, 'validate_runtime', return_value={'source_revision': 'fixture'}),
                    patch.dict(os.environ, environment),
                    patch.object(Path, 'home', return_value=home),
                    patch.object(Path, 'is_file', lambda path: original_is_file(notice_on_apfs(path))),
                    patch.object(Path, 'stat', lambda path, **kwargs: original_stat(notice_on_apfs(path), **kwargs)),
                    patch.object(Path, 'read_bytes', lambda path: original_read_bytes(notice_on_apfs(path))),
                ):
                    record = probe(args)
                    self.assertTrue(record['components_available'])
                    self.assertFalse(record['license_available'])
                    self.assertFalse(record['available'])
                    self.assertIsNone(resolve_surface_license(freesurfer_home=runtime))
