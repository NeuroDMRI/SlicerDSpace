"""Current-run discovery and representative selection without Slicer."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shared.auto_results import auto_manifest_paths, automatic_result_candidates, uniform_streamline_indices
from shared.result_scene import result_file_suffix
from tractography.wma import REQUIRED_MAJOR_TRACTS


class AutoResultsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def file(self, name, content="x"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return str(path)

    def task(self, name, keys, status="completed"):
        return {"name": name, "status": status, "outputs": {
            key: self.file(f"{name}.{key}.{extension}") for key, extension in keys.items()}}

    def test_compound_suffix_does_not_absorb_product_dots(self):
        for filename, suffix in (("a.b.nii.gz", ".nii.gz"), ("surface.ddsurfer.white.stl", ".stl"),
                                 ("connectome.streamline_count_matrix.csv", ".csv"), ("unknown.json", "")):
            self.assertEqual(result_file_suffix(filename), suffix)

    def test_all_product_families_survive_independent_caps(self):
        tasks = [
            self.task("microstructure.dti", {"fa": "nii.gz", "md": "nii.gz", "ad": "nii.gz", "roi_stats": "csv"}),
            self.task("microstructure.freewater", {"fwf": "nii.gz", "fw_fa": "nii.gz"}),
            self.task("microstructure.noddi", {"ndi": "nii.gz", "odi": "nii.gz"}),
            self.task("segmentation.ddseg", {"segmentation": "nii.gz", "wm_probability": "nii.gz"}),
            self.task("parcellation.ddparcel", {"wmparc": "nii.gz", "fa_reg_norm": "nii.gz"}),
            self.task("surface.ddsurfer", {key: "stl" for key in ("white_lh", "white_rh", "pial_lh", "pial_rh")}),
            self.task("tractography.csd_act", {"tractogram": "tck", "tractogram_vtk": "vtk"}),
            self.task("saf.native", {"tractogram": "tck"}),
            self.task("connectome.streamline_count", {"matrix_path": "csv", "nodes_path": "tsv", "stats": "csv"}),
        ]
        bundles = self.task("bundles.wmql", {name: "trk" for name in ("AF_left", "AF_right", "CST_left", "CST_right", "CC", "other")})
        from tractography.wmql import WMQL_QC_VERSION
        bundles['qc_path'] = self.file('wmql_qc.json', json.dumps({'status': 'completed', 'version': WMQL_QC_VERSION,
            'automatic_qc_passed': True, 'output_summary': {'bundle_count': 6,
                'bundle_readable_count': 6, 'bundle_nonempty_count': 6,
                'bundles': [{'query_name': name, 'streamline_count': 20} for name in bundles['outputs']]}}))
        tasks.append(bundles)
        selected, skipped, deferred = automatic_result_candidates({"status": "running", "tasks": tasks})
        self.assertGreater(len(selected), 12)
        self.assertEqual({item["product"] for item in selected}, {task["name"] for task in tasks})
        self.assertEqual(len([item for item in selected if item["product"] == "surface.ddsurfer"]), 4)
        self.assertEqual(next(item for item in selected if item["product"] == "saf.native")["display_role"], "tractography")
        self.assertEqual(next(item for item in selected if item["product"] == "tractography.csd_act")["key"], "tractogram")
        self.assertFalse(skipped)
        self.assertTrue(deferred)

    def test_only_recorded_completed_or_explicit_reused_tasks(self):
        tasks = [self.task("microstructure.dti", {"fa": "nii.gz"}, "failed"),
                 self.task("microstructure.noddi", {"ndi": "nii.gz"}, "running"),
                 self.task("surface.ddsurfer", {"white_lh": "stl"}, "planned"),
                 self.task("microstructure.freewater", {"fwf": "nii.gz"}, "skipped")]
        self.assertEqual(automatic_result_candidates({"tasks": tasks})[0], [])
        tasks[-1]["reason"] = "completed Free Water outputs found"
        selected = automatic_result_candidates({"tasks": tasks})[0]
        self.assertEqual(len(selected), 1)

    def test_current_qc_records_only_no_legacy_bundle_folder_scan(self):
        path = self.file("bundles/AF_left.vtp")
        self.file("bundles/stale.vtp")
        counts = {name: 20 for name in REQUIRED_MAJOR_TRACTS}
        counts.update({f'other_{index}.vtp': 20 for index in range(69)})
        qc = self.file("qc.json", json.dumps({"status": "completed", "version": 2, "automatic_qc_passed": True, "output_summary": {"bundle_files": [path],
            'anatomical_tract_count': 74, 'anatomical_tract_readable_count': 74, 'anatomical_tract_nonempty_count': 74,
            'anatomical_tract_streamline_counts': counts}}))
        task = {"name": "bundles.wma", "status": "completed", "qc_path": qc, "outputs": {"output_dir": str(self.root)}}
        with patch.object(Path, "glob", side_effect=AssertionError("No automatic folder scans")), \
             patch.object(Path, "rglob", side_effect=AssertionError("No automatic folder scans")):
            selected = automatic_result_candidates({"tasks": [task]})[0]
        self.assertEqual([item["path"] for item in selected], [path])
        Path(qc).write_text(json.dumps({"status": "completed", "output_summary": {
            "anatomical_tracts_dir": str(self.root / "bundles"), "anatomical_tract_count": 2}}))
        self.assertEqual(automatic_result_candidates({"tasks": [task]})[0], [])

    def test_numeric_bundles_use_generic_first_three_nonempty_recorded_files(self):
        files = [self.file(f"cluster_{index:03}.vtp") for index in range(5)]
        from tractography.tractcloud import ADAPTER_REVISION, TRACTCLOUD_QC_VERSION
        qc = self.file("qc.json", json.dumps({"status": "completed", "version": TRACTCLOUD_QC_VERSION,
            "executed": True, "automatic_qc_passed": True, "adapter_revision": ADAPTER_REVISION,
            "output_summary": {"bundle_files": files[1:], "bundle_count": 4,
                "bundle_readable_count": 4, "bundle_nonempty_count": 4,
                "assigned_streamline_count": 80}}))
        task = {"name": "bundles.tractcloud", "status": "completed", "qc_path": qc}
        selected, _, deferred = automatic_result_candidates({"tasks": [task]})
        self.assertEqual([item["path"] for item in selected], files[1:4])
        self.assertEqual({item["display_role"] for item in selected}, {"bundle:representative"})
        self.assertEqual(len(deferred), 1)

    def test_qc_failure_is_excluded_not_relabelled_as_passed(self):
        task = self.task("microstructure.dti", {"fa": "nii.gz"})
        task["qc_path"] = self.file("qc.json", '{"status":"failed"}')
        selected, skipped, _ = automatic_result_candidates({"tasks": [task]})
        self.assertFalse(selected)
        self.assertEqual(skipped[0]["qc_state"], "failed")
        self.assertEqual(skipped[0]["reason_category"], "qc_failed")

    def test_failed_status_wins_over_contradictory_automatic_pass_flag(self):
        task = self.task('microstructure.dti', {'fa': 'nii.gz'})
        qc_path = Path(self.file('qc.json'))
        task['qc_path'] = str(qc_path)
        for status in ('failed', 'partial_failed', 'error', 'FAIL'):
            with self.subTest(status=status):
                qc_path.write_text(json.dumps({'status': status, 'automatic_qc_passed': True}))
                selected, skipped, _ = automatic_result_candidates({'tasks': [task]})
                self.assertEqual(selected, [])
                self.assertEqual(skipped[0]['reason_category'], 'qc_failed')

    def test_explicit_failed_check_or_error_cannot_be_hidden_by_pass_flag(self):
        task = self.task('microstructure.dti', {'fa': 'nii.gz'})
        qc = Path(self.file('qc.json'))
        task['qc_path'] = str(qc)
        for detail in ({'checks': {'native_grid': False}},
                       {'checks': [{'name': 'native_grid', 'passed': False}]},
                       {'error': 'Native grid mismatch'}):
            qc.write_text(json.dumps({'status': 'completed', 'automatic_qc_passed': True, **detail}))
            selected, skipped, _ = automatic_result_candidates({'tasks': [task]})
            self.assertEqual(selected, [])
            self.assertEqual(skipped[0]['qc_state'], 'failed')

    def test_dki_all_six_maps_and_distinct_corrected_maps(self):
        maps = ("mk", "ak", "rk", "kfa", "mkt", "rtk")
        tasks = [self.task("microstructure.dki", {"dki_" + key: "nii.gz" for key in maps}),
                 self.task("microstructure.mkcurve", {"corrected_" + key: "nii.gz" for key in maps})]
        selected = automatic_result_candidates({"tasks": tasks})[0]
        self.assertEqual(len(selected), 12)
        self.assertEqual(len({entry["display_role"] for entry in selected}), 12)

    def test_preprocessing_only_declared_mask_and_mean_b0_not_full_4d_dwi(self):
        task = self.task("preprocessing.dwi", {"final_mask": "nii.gz", "mean_b0": "nii.gz", "final_dwi": "nii.gz"})
        selected, _, deferred = automatic_result_candidates({"tasks": [task]})
        self.assertEqual({entry["display_role"] for entry in selected}, {"scalar:b0", "categorical:mask"})
        self.assertEqual(deferred[0]["key"], "final_dwi")

    def test_large_vtk_is_deferred_without_loading(self):
        task = self.task("tractography.csd_act", {"tractogram": "vtk"})
        with patch("shared.auto_results.AUTOMATIC_VTK_MAX_BYTES", 0):
            selected, _, deferred = automatic_result_candidates({"tasks": [task]})
        self.assertFalse(selected)
        self.assertIn("Large VTK", deferred[0]["reason"])

    def test_ukf_auto_results_selects_vtk_and_never_visualizes_trk(self):
        task = self.task("tractography.ukf", {"tractogram": "vtk", "tractogram_trk": "trk"})
        selected, skipped, deferred = automatic_result_candidates({"tasks": [task]})
        self.assertEqual([item["key"] for item in selected], ["tractogram"])
        self.assertTrue(selected[0]["path"].endswith(".vtk"))
        self.assertFalse(skipped)
        self.assertFalse(deferred)
        # Even when UKF .vtk exceeds AUTOMATIC_VTK_MAX_BYTES, .trk must not be auto-loaded instead.
        with patch("shared.auto_results.AUTOMATIC_VTK_MAX_BYTES", 0):
            large_selected, large_skipped, large_deferred = automatic_result_candidates({"tasks": [task]})
        self.assertEqual(large_selected, [])
        self.assertEqual(large_skipped, [])
        self.assertEqual([item["key"] for item in large_deferred], ["tractogram"])

    def test_csd_act_and_dti_det_with_vtk_exclude_trk_from_auto_results(self):
        tasks = [
            self.task("tractography.csd_act", {"tractogram": "tck", "tractogram_trk": "trk", "tractogram_vtk": "vtk"}),
            self.task("tractography.dti_det", {"tractogram_trk": "trk", "tractogram_vtk": "vtk"}),
            self.task("tractography.saf", {"tractogram": "tck", "tractogram_vtk": "vtk"}),
        ]
        selected, skipped, deferred = automatic_result_candidates({"tasks": tasks})
        all_paths = [item["path"] for item in selected + skipped + deferred]
        self.assertFalse(any(path.endswith(".trk") for path in all_paths), all_paths)
        dti = next(item for item in selected if item["product"] == "tractography.dti_det")
        self.assertTrue(dti["path"].endswith(".vtk"))

    def test_uniform_sample_includes_endpoints_and_respects_limit(self):
        indices = uniform_streamline_indices(1_000_000)
        self.assertEqual(len(indices), 30_000)
        self.assertEqual((indices[0], indices[-1]), (0, 999_999))
        self.assertEqual(len(set(indices)), len(indices))
        self.assertEqual(uniform_streamline_indices(3, 10), (0, 1, 2))
        with self.assertRaises(ValueError):
            uniform_streamline_indices(10, 0)

    def test_batch_discovers_only_actual_results_and_excludes_touched_old_records(self):
        current = Path(self.file("new/00_run_metadata/Test1_orchestration_manifest.json", json.dumps({
            "subject_id": "new", "execute": True, "status": "running", "updated_at": "2026-10-12T12:00:00Z", "tasks": []})))
        old = Path(self.file("old/00_run_metadata/Test1_orchestration_manifest.json", json.dumps({
            "subject_id": "old", "execute": True, "updated_at": "2020-01-01T00:00:00Z", "tasks": []})))
        os.utime(old, (2_000_000_000, 2_000_000_000))
        batch = Path(self.file("batch.json", json.dumps({"kind": "slicerdsurfer_batch_run", "execute": True,
            "subjects": [{"subject_id": "unrun", "run_manifest_path": str(current)}],
            "planned_commands": [["--output-root", str(current.parent.parent)]],
            "results": [{"subject_id": "new", "command": ["python", "cli", "--output-root", str(current.parent.parent)]},
                        {"subject_id": "old", "run_manifest_path": str(old)}]})))
        os.utime(batch, (2_000_000_000, 2_000_000_000))
        self.assertEqual(auto_manifest_paths(batch, 1_790_000_000), (current,))
        self.assertEqual(auto_manifest_paths(current, 1_790_000_000), (current,))
        self.assertEqual(auto_manifest_paths(old, 1_790_000_000), ())

    def test_actual_child_stdout_log_tail_can_resolve_manifest(self):
        child = self.file("child/Test1_orchestration_manifest.json", json.dumps({"subject_id": "s", "execute": True, "tasks": []}))
        log = self.file("child.stdout.log", "noise\n" + json.dumps({"run_manifest_path": child}) + "\n")
        batch = Path(self.file("batch.json", json.dumps({"kind": "slicerdsurfer_batch_run", "execute": True,
            "results": [{"subject_id": "s", "stdout_log": log}]})))
        self.assertEqual(auto_manifest_paths(batch), (Path(child),))
        batch.write_text(json.dumps({"kind": "slicerdsurfer_batch_run", "execute": False,
                                    "results": [{"subject_id": "s", "stdout_log": log}]}))
        self.assertEqual(auto_manifest_paths(batch), ())

    def test_multi_tracking_loads_only_started_fresh_matching_children(self):
        def child(name, timestamp, **options):
            return self.file(name + '.json', json.dumps({'execute': True, 'tasks': [],
                'subject_id': 's', 'started_at': timestamp, **options}))
        timestamp = '2026-10-21T12:00:00+00:00'
        current = child('current', timestamp)
        old = child('old', '2020-01-01T00:00:00+00:00')
        planned = child('planned', timestamp)
        other = child('other', timestamp, subject_id='different')
        os.utime(old, (2_000_000_000, 2_000_000_000))
        parent = Path(self.file('multi.json', json.dumps({'subject_id': 's', 'execute': True,
            'tractography_backends': ['csd_act', 'ukf'], 'tasks': [], 'child_runs': [
                {'manifest_path': path, 'started_at': timestamp, 'status': status}
                for path, status in ((current, 'running'), (old, 'running'), (planned, 'planned'), (other, 'running'))]})))
        self.assertEqual(auto_manifest_paths(parent), (Path(current),))
        batch = Path(self.file('multi_batch.json', json.dumps({'kind': 'slicerdsurfer_batch_run',
            'execute': True, 'results': [{'subject_id': 's', 'manifest_path': str(parent)}]})))
        self.assertEqual(auto_manifest_paths(batch), (Path(current),))


if __name__ == "__main__":
    unittest.main()
