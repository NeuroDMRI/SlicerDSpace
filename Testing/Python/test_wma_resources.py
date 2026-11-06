import hashlib
import io
import json
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

from scripts.import_external_resources import extract_indexed_archive
from scripts.package_wma_atlases import select_member
from shared import optional_assets
from shared.model_asset_integrity import ModelAssetIntegrityError
from tractography.wma_atlases import atlas_definition


class WMAResourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, members=None):
        members = members or {'NABA-800FC/atlas.p': b'test atlas, not a pickle'}
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as archive:
            for name, content in members.items():
                archive.writestr(name, content)
        data = stream.getvalue()
        directory = 'bundles/wma/naba'
        destination = self.root / directory
        (destination / 'archives').mkdir(parents=True)
        cut = len(data) // 2
        parts = [directory + '/archives/runtime.zip.part-000', directory + '/archives/runtime.zip.part-001']
        for part, chunk in zip(parts, (data[:cut], data[cut:])):
            (self.root / part).write_bytes(chunk)
        index_relative = directory + '/extracted_resource_index.json'
        files = [{'path': name, 'sha256': hashlib.sha256(content).hexdigest(), 'size_bytes': len(content)}
                 for name, content in members.items()]
        (self.root / index_relative).write_text(json.dumps({'files': files}))
        definition = {'directory': directory, 'extracted_index': index_relative, 'archive_parts': parts,
                      'archive_sha256': hashlib.sha256(data).hexdigest(), 'required_paths': [*parts, index_relative]}
        records = {name: {'sha256': hashlib.sha256((self.root / name).read_bytes()).hexdigest(),
                          'size_bytes': (self.root / name).stat().st_size} for name in definition['required_paths']}
        catalog = self.root / 'catalog.json'
        catalog.write_text(json.dumps({'profiles': {'wma-naba': definition}, 'asset_records': records}))
        return definition, catalog

    def test_two_archive_parts_restore_exact_file_and_reuse_without_overwrite(self):
        definition, _ = self.fixture()
        self.assertEqual(extract_indexed_archive(self.root, definition), 1)
        path = self.root / definition['directory'] / 'NABA-800FC/atlas.p'
        self.assertEqual(path.read_bytes(), b'test atlas, not a pickle')
        stamp = path.stat().st_mtime_ns
        self.assertEqual(extract_indexed_archive(self.root, definition), 1)
        self.assertEqual(path.stat().st_mtime_ns, stamp)

    def test_changed_part_is_rejected_before_extraction(self):
        definition, _ = self.fixture()
        (self.root / definition['archive_parts'][0]).write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            extract_indexed_archive(self.root, definition)
        self.assertFalse((self.root / definition['directory'] / 'NABA-800FC/atlas.p').exists())

    def test_changed_existing_member_is_not_overwritten(self):
        definition, _ = self.fixture()
        extract_indexed_archive(self.root, definition)
        path = self.root / definition['directory'] / 'NABA-800FC/atlas.p'
        path.write_bytes(b'changed')
        with self.assertRaises(FileExistsError):
            extract_indexed_archive(self.root, definition)
        self.assertEqual(path.read_bytes(), b'changed')

    def test_unsafe_index_and_symlinks_are_rejected(self):
        definition, _ = self.fixture({'../escape': b'bad'})
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            extract_indexed_archive(self.root, definition)
        self.assertFalse((self.root.parent / 'escape').exists())

    def test_f4_symlinked_ancestor_of_the_asset_root_is_allowed(self):
        # F4: a symlink above the asset root (as /var is on macOS) is the user's folder layout.
        definition, _ = self.fixture()
        with tempfile.TemporaryDirectory() as other:
            link = Path(other) / 'link'
            link.symlink_to(self.root, target_is_directory=True)
            self.assertEqual(extract_indexed_archive(link, definition), 1)
        self.assertEqual((self.root / definition['directory'] / 'NABA-800FC/atlas.p').read_bytes(),
                         b'test atlas, not a pickle')

    def test_f4_symlink_inside_the_destination_tree_is_still_refused(self):
        definition, _ = self.fixture()
        with tempfile.TemporaryDirectory() as outside:
            (self.root / definition['directory'] / 'NABA-800FC').symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'Refusing symlink'):
                extract_indexed_archive(self.root, definition)
            self.assertEqual(list(Path(outside).iterdir()), [])

    def test_f4_symlinked_destination_folder_is_still_refused(self):
        definition, _ = self.fixture()
        destination = self.root / definition['directory']
        with tempfile.TemporaryDirectory() as outside:
            real = Path(outside) / 'naba'
            shutil.move(str(destination), str(real))
            destination.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'Refusing symlink'):
                extract_indexed_archive(self.root, definition)

    def test_optional_readiness_requires_current_extracted_hashes(self):
        definition, catalog = self.fixture()
        with mock.patch.object(optional_assets, 'MANIFEST', catalog):
            self.assertIsNone(optional_assets.verified_resource_directory('wma-naba', asset_root=self.root))
            extract_indexed_archive(self.root, definition)
            self.assertEqual(optional_assets.verified_resource_directory('wma-naba', asset_root=self.root),
                             self.root / definition['directory'])
            (self.root / definition['directory'] / 'NABA-800FC/atlas.p').write_bytes(b'bad')
            with self.assertRaises(ModelAssetIntegrityError):
                optional_assets.verified_resource_directory('wma-naba', asset_root=self.root)

    def test_runtime_subset_keeps_all_cluster_inputs_but_not_participant_data(self):
        definition = atlas_definition('east-west')
        self.assertEqual(select_member('atlas/East-West_WM_Atlas_800FC/cluster_00800.vtp', definition),
                         'East-West_WM_Atlas_800FC/cluster_00800.vtp')
        self.assertIsNone(select_member('atlas/sub-123/DWI.nii.gz', definition))
        self.assertIsNone(select_member('atlas/East-West_WM_Atlas_Reg/subject01.vtk', definition))
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            select_member('../East-West_WM_Atlas_800FC/atlas.p', definition)

    def test_setup_plan_downloads_only_the_selected_atlas(self):
        from types import SimpleNamespace
        from shared.selected_setup import build_setup_plan
        item = SimpleNamespace(key='wma_org_atlas', kind='path', status='missing', required=True,
                               targets=('wma',), label='Atlas', details={}, install_group='wma', install_hint='prepare')
        with mock.patch('scripts.setup_optional_assets.prepare', return_value={'size_bytes': 42}) as prepare:
            plan = build_setup_plan(SimpleNamespace(statuses=[item]), wma_atlas='east-west')
        self.assertEqual(plan['optional_asset_profiles'], ['wma-east-west'])
        self.assertEqual(plan['manual_items'], [])
        self.assertEqual(prepare.call_args.args[0], ['wma-east-west'])

    def test_new_atlas_profiles_use_their_published_commit_not_the_legacy_mirror(self):
        from scripts.download_model_assets import load_manifest, resolve_asset_plan
        manifest = load_manifest(optional_assets.MANIFEST)
        source = manifest['asset_sources']['wma_atlases_mirror']
        self.assertRegex(source['url_template'], r'/raw/[0-9a-f]{40}/\{path_urlencoded\}$')
        self.assertEqual(source['access'], 'public')
        for family in ('east-west', 'naba'):
            plan = resolve_asset_plan(manifest, atlas_definition(family)['profile'])
            self.assertEqual(plan.source_ids, ('wma_atlases_mirror',))
            for path, record in plan.asset_records.items():
                self.assertTrue(path.startswith('bundles/wma/' + family + '/'))
                self.assertEqual(record['source_ids'], ['wma_atlases_mirror'])
                self.assertEqual(record['license'], 'CC-BY-4.0')


if __name__ == '__main__':
    unittest.main()
