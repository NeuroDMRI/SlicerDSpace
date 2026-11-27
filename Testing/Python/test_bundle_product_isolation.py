"""Regression checks for decoded geometry and fresh-run product isolation."""

import json
from pathlib import Path
import tempfile
import unittest
from win_test_helpers import skip_unless_symlinks
from unittest import mock

from shared.output_contracts import polydata_streamline_count
from shared.product_staging import new_product_staging, publish_products, retain_failed_products
from vtp_fixture import vtp_lines


VTK = '# vtk DataFile Version 3.0\nfixture\nASCII\nDATASET POLYDATA\nPOINTS 2 float\n0 0 0 1 0 0\nLINES 1 3\n2 0 1\n'


class BundleIsolationTest(unittest.TestCase):
    def test_revalidation_preserves_raw_metadata_and_never_promotes_failed_sampling(self):
        from scripts.audit_bundle_products import update_qc_records
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / 'wma_org_atlas_manifest.json'
            qc = root / 'wma_org_atlas_qc.json'
            report = root / 'audit.json'
            manifest.write_text(json.dumps({'kind': 'wma_org_atlas', 'executed': True, 'status': 'completed'}))
            qc.write_text(json.dumps({'status': 'completed', 'elapsed_seconds': 123}))
            report.write_text('{}')
            result = dict(checks={'current_output_contract': False, 'geometry': True},
                          current_contract_error='CST insufficient', source_sha256='fixture', output_summary={})
            update_qc_records(manifest, result, report)
            current = json.loads(qc.read_text())
            self.assertEqual(current['status'], 'partial_failed')
            self.assertFalse(current['automatic_qc_passed'])
            self.assertFalse(current['quantification_eligible'])
            self.assertEqual(current['elapsed_seconds'], 123)
            archive = Path(current['revalidation']['previous_records'])
            self.assertEqual(json.loads((archive / qc.name).read_text())['status'], 'completed')

    def test_update_qc_refuses_running_manifest_opus_p2_wma_audit(self):
        """Backlog row 'Opus P2 (WMA audit)': --update-qc must not turn a running manifest into completed."""
        import contextlib
        import io
        from scripts.audit_bundle_products import update_qc_records
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / 'wma_org_atlas_manifest.json'
            qc = root / 'wma_org_atlas_qc.json'
            report = root / 'audit.json'
            manifest.write_text(json.dumps({'kind': 'wma_org_atlas', 'executed': True, 'status': 'running'}))
            qc.write_text(json.dumps({'status': 'running'}))
            report.write_text('{}')
            before = (manifest.read_text(), qc.read_text())
            result = dict(checks={'current_output_contract': True, 'geometry': True},
                          current_contract_error=None, source_sha256='fixture', output_summary={})
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                returned = update_qc_records(manifest, result, report)
            self.assertIsNone(returned)
            self.assertIn('Refused', err.getvalue())
            self.assertEqual((manifest.read_text(), qc.read_text()), before)
            self.assertEqual(json.loads(manifest.read_text())['status'], 'running')
            self.assertEqual(sorted(p.name for p in root.iterdir()), sorted([manifest.name, qc.name, report.name]))

    def test_update_qc_refusal_gives_a_non_zero_exit_opus_p2_wma_audit_exit(self):
        """Opus P2 (fix round): main() must not exit 0 when --update-qc refused a running manifest."""
        import contextlib
        import io
        import sys
        from scripts import audit_bundle_products
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / 'wma_org_atlas_manifest.json'
            qc = root / 'wma_org_atlas_qc.json'
            report = root / 'out' / 'audit.json'
            manifest.write_text(json.dumps({'kind': 'wma_org_atlas', 'executed': True, 'status': 'running'}))
            qc.write_text(json.dumps({'status': 'running'}))
            result = dict(status='ok', input_streamlines=1, current_contract_error=None, audit_seconds=0.,
                          checks={'current_output_contract': True, 'geometry': True},
                          source_sha256='fixture', output_summary={})
            argv = ['audit_bundle_products.py', '--manifest', str(manifest), '--report', str(report)]
            with mock.patch.object(audit_bundle_products, 'audit', return_value=result), \
                    contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                with mock.patch.object(sys, 'argv', argv):
                    self.assertEqual(audit_bundle_products.main(), 0)  # without --update-qc, passing checks exit 0
                with mock.patch.object(sys, 'argv', argv + ['--update-qc']):
                    self.assertEqual(audit_bundle_products.main(), 2)
            self.assertEqual(json.loads(manifest.read_text())['status'], 'running')

    def test_legacy_vtk_geometry_not_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bundle.vtk'
            path.write_text(VTK)
            self.assertEqual(polydata_streamline_count(path), 1)
            for data in (VTK[:80], VTK.replace('0 0 0 1', 'nan 0 0 1'), VTK.replace('2 0 1', '2 0 99'), VTK.replace('0 0 0 1 0 0', '0 0 0 0 0 0')):
                path.write_text(data)
                self.assertIsNone(polydata_streamline_count(path))

    def test_requested_wmql_identity_and_subject(self):
        from tractography.wmql import summarize_wmql_outputs, _wmql_output_contract_error
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'other_desc-wmql_AF.left.vtk').write_text(VTK)
            (root / 'subject_desc-wmql_Unexpected.vtk').write_text(VTK)
            summary = summarize_wmql_outputs(root, 'subject')
            self.assertEqual(summary['bundle_count'], 1)
            self.assertIsNotNone(_wmql_output_contract_error(summary, 1, ('AF.left',)))

    def test_published_old_classes_are_archived_not_recounted(self):
        from tractography.tractcloud import summarize_tractcloud_outputs
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'products'
            (root / 'Association').mkdir(parents=True)
            (root / 'Association' / 'old.vtp').write_text(vtp_lines(2))
            stage = new_product_staging(root, 'test')
            (stage / 'Association').mkdir()
            (stage / 'Association' / 'new.vtp').write_text(vtp_lines(3))
            archive = Path(publish_products(stage, root, ('Association',)))
            self.assertTrue((archive / 'Association' / 'old.vtp').is_file())
            self.assertFalse((root / 'Association' / 'old.vtp').exists())
            self.assertEqual(summarize_tractcloud_outputs(root)['assigned_streamline_count'], 3)

    def test_failed_products_are_not_current_results(self):
        from tractography.tractcloud import summarize_tractcloud_outputs
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'products'
            root.mkdir()
            stage = new_product_staging(root, 'test')
            (stage / 'partial.vtp').write_text(vtp_lines(2))
            retained = retain_failed_products(stage, root)
            self.assertTrue((retained / 'partial.vtp').exists())
            self.assertEqual(summarize_tractcloud_outputs(root)['bundle_count'], 0)

    @skip_unless_symlinks
    def test_virtualenv_symlink_is_preserved(self):
        from tractography.wmql import _resolve_python
        from tractography.tractcloud import resolve_tractcloud_python
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            actual = root / 'actual'
            actual.touch()
            alias = root / 'venv' / 'python'
            alias.parent.mkdir()
            alias.symlink_to(actual)
            self.assertEqual(_resolve_python(alias), alias)
            self.assertEqual(resolve_tractcloud_python(alias), alias)

    def test_external_python_environment_strips_slicer_home(self):
        from shared.external_process import python_tool_environment
        with mock.patch('shared.external_process.startup_environment', return_value={
            'PYTHONHOME': '/opt/Slicer/python', 'PATH': '/opt/Slicer-5.10/bin:/usr/bin',
        }):
            env = python_tool_environment('/opt/venv/bin/python')
            self.assertNotIn('PYTHONHOME', env)
            self.assertNotIn('Slicer-', env['PATH'])
            import os
            self.assertEqual(Path(env['PATH'].split(os.pathsep)[0]).resolve(), Path('/opt/venv/bin').resolve())

    def test_tractcloud_model_signature_tracks_changed_weights(self):
        from tractography.tractcloud import model_input_signature
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weight = root / 'TrainedModel' / 'best_tract_f1_model.pth'
            weight.parent.mkdir()
            weight.write_bytes(b'first')
            before = model_input_signature(root)
            weight.write_bytes(b'other')
            after = model_input_signature(root)
            self.assertNotEqual(before['TrainedModel/best_tract_f1_model.pth']['sha256'],
                                after['TrainedModel/best_tract_f1_model.pth']['sha256'])

    def test_unseeded_tractcloud_cache_is_not_reused(self):
        from shared.orchestrator import _tractcloud_complete
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'tractcloud_qc.json').write_text(json.dumps({'status': 'completed'}))
            (root / 'tractcloud_manifest.json').write_text(json.dumps({'status': 'completed'}))
            (root / 'tract.vtp').write_text(vtp_lines(3))
            self.assertFalse(_tractcloud_complete(root))

    def test_runtime_selection_does_not_promote_failed_or_missing_time(self):
        from scripts.collect_module_runtimes import select
        rows = select([dict(module='bundles.wma', status='scientific_contract_failed', runtime_seconds=10, priority=99),
                       dict(module='bundles.wma', status='completed', runtime_seconds=None, priority=99)])
        row = next(r for r in rows if r['module'] == 'bundles.wma')
        self.assertEqual(row['measurement_status'], 'MISSING')
        self.assertIsNone(row['seconds'])


if __name__ == '__main__':
    unittest.main()
