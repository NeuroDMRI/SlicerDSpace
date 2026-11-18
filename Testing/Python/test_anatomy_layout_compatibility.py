"""New anatomy writes and read-only compatibility with historical products."""

from pathlib import Path
import json
import tempfile
import unittest

from shared.io_layout import anatomy_input_dir, archive_development_outputs, materialize_canonical_output_tree, subject_layout


class AnatomyLayoutCompatibilityTests(unittest.TestCase):
    def test_new_writers_and_python_alias_use_anatomy(self):
        with tempfile.TemporaryDirectory() as tmp:
            layout = subject_layout(Path(tmp))
            self.assertEqual(layout.anatomy, layout.outputs / '01_anatomical_priors')
            self.assertEqual(layout.structural, layout.anatomy)
            self.assertEqual(layout.ddseg, layout.anatomy / 'tissue_segmentation')
            self.assertEqual(layout.ddparcel, layout.anatomy / 'brain_parcellation')
            self.assertEqual(layout.ddsurfer, layout.anatomy / 'cortical_surface')
            self.assertEqual(layout.registered_to_dwi, layout.stage_cache('01_anatomical_priors/registered_to_dwi'))

    def test_legacy_reads_do_not_move_or_copy_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            layout = subject_layout(Path(tmp), Path(tmp) / 'custom-output')
            legacy = layout.outputs / '03_structural/ddseg'
            legacy.mkdir(parents=True)
            saved = legacy / 'SegmentationMap_GMWMCSF.nii.gz'
            saved.write_bytes(b'legacy')
            from shared.orchestrator import _resolve_segmentation, _segmentation_outputs
            self.assertEqual(layout.anatomy_input_dir('ddseg'), legacy)
            self.assertEqual(_resolve_segmentation(layout.subject_root, layout.outputs), saved)
            self.assertTrue(all(path.parent == legacy for path in
                                _segmentation_outputs(layout.subject_root, layout.outputs).values()))
            archive_development_outputs(layout.subject_root, output_root=layout.outputs)
            self.assertEqual(saved.read_bytes(), b'legacy')
            self.assertFalse(layout.anatomy.exists())

    def test_partial_new_product_never_uses_old_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            layout = subject_layout(Path(tmp))
            old = layout.outputs / '03_structural/ddparcel'
            old.mkdir(parents=True)
            (old / 's-DDSurfer-wmparc.nii.gz').write_bytes(b'old')
            layout.ddparcel.mkdir(parents=True)
            (layout.ddparcel / 'partial.log').write_text('new run')
            from shared.orchestrator import _resolve_parcellation, _parcellation_outputs
            self.assertIsNone(_resolve_parcellation(layout.subject_root, 's'))
            expected_stage_dir = layout.stage_cache('01_anatomical_priors/brain_parcellation')
            self.assertTrue(all(path.parent == expected_stage_dir for path in
                                _parcellation_outputs(layout.subject_root, 's').values()))
            self.assertEqual((old / 's-DDSurfer-wmparc.nii.gz').read_bytes(), b'old')

    def test_flat_legacy_view_cannot_shadow_numbered_legacy_products(self):
        with tempfile.TemporaryDirectory() as tmp:
            layout = subject_layout(Path(tmp))
            old = layout.outputs / '03_structural/ddseg'
            flat = layout.outputs / 'ddseg'
            old.mkdir(parents=True)
            flat.mkdir()
            name = 'SegmentationMap_GMWMCSF.nii.gz'
            (old / name).write_bytes(b'numbered result')
            (flat / name).write_bytes(b'older flat result')
            for mode in ('copy', 'symlink'):
                created = materialize_canonical_output_tree(layout.subject_root, mode=mode)
                self.assertIn('tissue_segmentation:SegmentationMap_GMWMCSF.nii.gz', created)
                self.assertFalse((layout.ddseg / name).is_symlink())
                self.assertEqual((layout.ddseg / name).read_bytes(), b'numbered result')
                self.assertEqual((old / name).read_bytes(), b'numbered result')
                self.assertEqual((flat / name).read_bytes(), b'older flat result')

    def test_roi_reader_accepts_legacy_then_prefers_new_product(self):
        from types import SimpleNamespace
        from microstructure.pipeline import _resolve_roi_label_map
        with tempfile.TemporaryDirectory() as tmp:
            layout = subject_layout(Path(tmp))
            config = SimpleNamespace(roi_label_map=None, subject_id='s',
                                     native_bundle=SimpleNamespace(subject_root=layout.subject_root))
            old = layout.outputs / '03_structural/ddparcel'
            old.mkdir(parents=True)
            old_map = old / 's-DDSurfer-wmparc.nii.gz'
            old_map.write_bytes(b'old')
            self.assertEqual(_resolve_roi_label_map(config), old_map)
            layout.ddparcel.mkdir(parents=True)
            self.assertIsNone(_resolve_roi_label_map(config))
            new_map = layout.ddparcel / old_map.name
            new_map.write_bytes(b'new')
            self.assertEqual(_resolve_roi_label_map(config), new_map)

    def test_reports_support_both_names_without_rewriting_old_records(self):
        from shared.output_tree_report import _describe_dir, _postprocess_summary
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            old = output / '03_structural/ddsurfer/postprocess'
            old.mkdir(parents=True)
            record = old / 'qc.json'
            record.write_text(json.dumps({'status': 'warning'}))
            before = record.read_bytes()
            self.assertEqual(_postprocess_summary(output)['status'], 'warning')
            self.assertEqual(_describe_dir('03_anatomy/ddseg'), _describe_dir('03_structural/ddseg'))
            new = output / '03_anatomy/ddsurfer/postprocess'
            new.mkdir(parents=True)
            self.assertEqual(_postprocess_summary(output)['status'], 'not available')
            self.assertEqual(record.read_bytes(), before)

    def test_product_paths_cannot_escape_output_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            for product in ('../outside', str(Path(tmp).resolve())):
                with self.assertRaises(ValueError):
                    anatomy_input_dir(Path(tmp), product)

    def test_broken_new_product_link_does_not_fall_back_to_older_results(self):
        from shared.orchestrator import _resolve_segmentation
        with tempfile.TemporaryDirectory() as tmp:
            layout = subject_layout(Path(tmp))
            flat = layout.subject_root / 'segmentation'
            flat.mkdir()
            (flat / 'SegmentationMap_GMWMCSF.nii.gz').write_bytes(b'old')
            layout.anatomy.mkdir(parents=True)
            try:
                layout.ddseg.symlink_to(layout.subject_root / 'missing-run', target_is_directory=True)
            except OSError:
                self.skipTest('Directory symlinks unavailable on this host')
            self.assertEqual(layout.anatomy_input_dir('ddseg'), layout.ddseg)
            self.assertIsNone(_resolve_segmentation(layout.subject_root))

    def test_old_surface_and_postprocess_paths_still_resolve(self):
        from shared.orchestrator import _surface_outputs, _surface_postprocess_outputs
        with tempfile.TemporaryDirectory() as tmp:
            layout = subject_layout(Path(tmp))
            old = layout.outputs / '03_structural/ddsurfer'
            (old / 'pred_results_SlicerDSpace/native/s').mkdir(parents=True)
            (old / 'postprocess').mkdir()
            self.assertTrue(all(path.is_relative_to(old) for path in _surface_outputs(layout.subject_root, 's').values()))
            post = _surface_postprocess_outputs(layout.subject_root, 's')
            self.assertEqual(post['manifest'], old / 'postprocess/manifest.json')

    def test_output_tree_audit_accepts_legacy_and_reports_current_canonical_name(self):
        from shared.orchestrator import _audit_output_tree, _write_output_tree_semantics
        with tempfile.TemporaryDirectory() as tmp:
            layout = subject_layout(Path(tmp))
            (layout.outputs / '03_structural').mkdir(parents=True)
            layout.run_metadata.mkdir()
            report = _audit_output_tree(layout, subject_id='s', selected_targets=('segmentation',))
            self.assertIn('01_anatomical_priors', report['expected_top_dirs'])
            self.assertFalse(any("missing canonical output directory" in w for w in report['warnings']))
            self.assertEqual(report['status'], 'warning')
            semantics = _write_output_tree_semantics(layout=layout, subject_id='s',
                requested_targets=('segmentation',), run_targets=('segmentation',), tasks=[])
            paths = {item['path'] for item in semantics['entries']}
            self.assertIn('01_anatomical_priors', paths)
