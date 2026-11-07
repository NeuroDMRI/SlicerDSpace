import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.run_wma_atlas import append_with_atlas_groups
from tractography.wma_atlases import (ATLAS_CHOICES, ATLAS_DEFINITIONS, atlas_output_contract,
                                      detect_atlas_family, parameterized_author_shell, atlas_hemisphere_source)


class WMAAtlasTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def atlas(self, family):
        definition = ATLAS_DEFINITIONS[family]
        for directory in definition['directories']:
            (self.root / directory).mkdir(exist_ok=True)
        cluster = self.root / definition['directories'][1]
        for name in (*definition.get('hemispheric', ()), *definition.get('commissural', ())):
            (cluster / (name + '.mrml')).write_text('<MRML/>')
        return cluster

    def test_three_families_and_population_labels(self):
        self.assertEqual(ATLAS_CHOICES, ('org', 'east-west', 'naba'))
        self.assertIn('neonates', ATLAS_DEFINITIONS['naba']['label'])
        self.assertIn('adults', ATLAS_DEFINITIONS['east-west']['label'])

    def test_atlas_detection_does_not_silently_choose_between_two_atlases(self):
        self.atlas('org')
        self.assertEqual(detect_atlas_family(self.root), 'org')
        self.atlas('naba')
        with self.assertRaisesRegex(ValueError, 'Multiple'):
            detect_atlas_family(self.root)
        self.assertEqual(detect_atlas_family(self.root, 'naba'), 'naba')

    def test_auto_detection_expands_the_home_directory(self):
        self.atlas('naba')
        with mock.patch.dict(os.environ, {'HOME': str(self.root.parent), 'USERPROFILE': str(self.root.parent)}):
            self.assertEqual(detect_atlas_family(Path('~') / self.root.name), 'naba')

    def test_published_grouping_differs_between_atlases(self):
        east = atlas_output_contract('east-west', self.atlas('east-west'))
        naba = atlas_output_contract('naba', self.atlas('naba'))
        self.assertEqual(len(east['expected_tract_files']), 76)
        self.assertEqual(len(naba['expected_tract_files']), 78)
        self.assertIn('T_CPC_left.vtp', east['expected_tract_files'])
        self.assertIn('T_CPC_left.vtp', naba['expected_tract_files'])
        self.assertNotIn('T_CPC.vtp', naba['expected_tract_files'])
        self.assertIn('T_CB_D_left.vtp', naba['expected_tract_files'])
        self.assertIn('T_CBLM-I&P_left.vtp', east['expected_tract_files'])
        self.assertIn('T_Intra-CBLM-I&P_left.vtp', naba['expected_tract_files'])
        self.assertNotIn('T_Intra-CBLM-I&P_left.vtp', east['expected_tract_files'])

    def test_missing_and_case_ambiguous_definitions_fail(self):
        cluster = self.atlas('naba')
        (cluster / 'T_AF.mrml').unlink()
        with self.assertRaisesRegex(ValueError, 'T_AF'):
            atlas_output_contract('naba', cluster)
        (cluster / 'T_AF.mrml').touch()
        probe = self.root / 'CaseProbe'
        probe.touch()
        case_sensitive = not (self.root / 'caseprobe').exists()  # F4: macOS folders are often case-insensitive
        probe.unlink()
        if case_sensitive:  # a case-insensitive folder cannot hold both names; only the missing part applies
            (cluster / 'T_af.mrml').touch()
            with self.assertRaisesRegex(ValueError, 'Ambiguous'):
                atlas_output_contract('naba', cluster)

    def test_other_author_driver_commands_are_preserved(self):
        source = ('#!/bin/bash\nRegAtlasFolder=$AtlasBaseFolder/ORG-RegAtlas-100HCP\n'
                  'FCAtlasFolder=$AtlasBaseFolder/ORG-800FC-100HCP\nwm_cluster_from_atlas.py "$@"\n')
        result = parameterized_author_shell(source, 'NABA-RegAtlas', 'NABA-800FC')
        self.assertIn('RegAtlasFolder="$AtlasBaseFolder/NABA-RegAtlas"', result)
        self.assertTrue(result.endswith('wm_cluster_from_atlas.py "$@"\n'))
        with self.assertRaisesRegex(ValueError, 'Unrecognized'):
            parameterized_author_shell('#!/bin/bash\nexit 0', 'x', 'y')

    def test_non_org_command_uses_the_execution_python_launcher_shim(self):
        from tractography.wma import WMAOrgAtlasConfig, build_wma_org_atlas_command
        self.atlas('naba')
        (self.root / 'bin').mkdir()
        (self.root / 'bin/wm_apply_ORG_atlas_to_subject.sh').touch()
        config = WMAOrgAtlasConfig(subject_id='test', input_tractography=self.root / 'input.vtp',
                                  atlas_dir=self.root, output_dir=self.root / 'output',
                                  slicer_executable=self.root / 'Slicer', wma_root=self.root)
        with mock.patch('tractography.wma.sys.platform', 'linux'), mock.patch(
                'tractography.wma.sys.executable', '/not-launchable/bin/python-real'):
            command = build_wma_org_atlas_command(config)
        self.assertEqual(command[0], 'python')
        self.assertEqual(command[command.index('--family') + 1], 'naba')
        self.assertTrue(command[1].endswith('scripts/run_wma_atlas.py'))

    def test_append_adapter_changes_only_group_selection(self):
        author = self.root / 'append.py'
        output = self.root / 'groups.json'
        author.write_text('''import json, sys
def get_hemispheric_mono_bundle_names(df): return ['ORG_hemi']
def get_commissural_bundle_names(df): return ['ORG_comm']
def main():
    with open(sys.argv[1], 'w') as output:
        json.dump([get_hemispheric_mono_bundle_names(None), get_commissural_bundle_names(None)], output)
if __name__ == '__main__': main()
''')
        before = author.read_bytes()
        append_with_atlas_groups(author, {'hemispheric': ['T_AF'], 'commissural': ['T_CC1']}, [str(output)])
        self.assertEqual(json.loads(output.read_text()), [['T_AF'], ['T_CC1']])
        self.assertEqual(author.read_bytes(), before)

    def test_org_specific_cpc_overrides_do_not_modify_other_atlas_labels(self):
        source = "def assess():\n    location_data = load()\n" + ''.join(
            "    location_data[" + str(index) + "][1] = b'h'\n" for index in (145, 159, 557, 677, 770))
        source += "    return location_data\n"
        candidate = atlas_hemisphere_source(source)
        self.assertNotIn("= b'h'", candidate)
        self.assertIn('return location_data', candidate)
        compile(candidate, '<atlas-assessment>', 'exec')
        self.assertEqual(atlas_hemisphere_source('location_data = load()\n'), 'location_data = load()\n')
        with self.assertRaisesRegex(ValueError, 'Unrecognized'):
            atlas_hemisphere_source("location_data[100][1] = b'h'\n")

    def test_qc_rejects_org_file_inventory_for_a_different_atlas(self):
        from tractography.wma import _wma_output_contract_error, REQUIRED_MAJOR_TRACTS
        contract = atlas_output_contract('naba', self.atlas('naba'))
        counts = {name: 100 for name in contract['expected_tract_files']}
        counts.pop('T_SCP_left.vtp')
        counts['T_unrelated.vtp'] = 100
        summary = dict(atlas_contract=contract, anatomical_tract_count=78,
                       anatomical_tract_readable_count=78, anatomical_tract_nonempty_count=78,
                       anatomical_tract_streamline_counts=counts)
        self.assertIn('T_SCP_left.vtp', _wma_output_contract_error(summary))

    def test_atlas_resolver_does_not_apply_legacy_org_environment_to_naba(self):
        from shared.optional_assets import resolve_wma_atlas_directory
        with mock.patch.dict(os.environ, {'WMA_ATLAS_DIR': '/legacy/org'}, clear=True), mock.patch(
                'shared.optional_assets.verified_resource_directory', return_value=None) as verify:
            self.assertIsNone(resolve_wma_atlas_directory(atlas_family='naba', asset_root=self.root))
            self.assertEqual(verify.call_args.args, ('wma-naba',))

    def test_cli_accepts_only_supported_atlases_and_defaults_to_org(self):
        from shared.run_subject_pipeline_cli import build_parser
        parser = build_parser()
        self.assertEqual(parser.parse_args(['--subject-root', str(self.root)]).wma_atlas, 'org')
        self.assertEqual(parser.parse_args(['--subject-root', str(self.root), '--wma-atlas', 'naba']).wma_atlas, 'naba')

    def test_inherited_resource_paths_follow_atlas_selection(self):
        from shared.optional_assets import inherited_wma_atlas_directory
        managed = self.root / 'models'
        self.assertIsNone(inherited_wma_atlas_directory(managed / 'bundles/wma', 'naba', managed))
        self.assertIsNone(inherited_wma_atlas_directory(managed / 'bundles/wma/naba', 'naba', managed))
        self.atlas('org')
        self.assertIsNone(inherited_wma_atlas_directory(self.root, 'naba'))
        self.atlas('naba')
        self.assertEqual(inherited_wma_atlas_directory(self.root, 'naba'), self.root)
        custom = self.root / 'custom'
        self.assertEqual(inherited_wma_atlas_directory(custom, 'naba'), custom)


if __name__ == '__main__':
    unittest.main()
