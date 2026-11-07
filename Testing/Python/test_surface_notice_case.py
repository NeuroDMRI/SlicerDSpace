"""Keep public LICENSE.txt distinct from private activation on macOS volumes."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
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
