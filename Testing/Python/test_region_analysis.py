"""Selected-region postprocessing, independent of Slicer and model fitting."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nibabel as nib
import numpy as np

from microstructure.pipeline import roi_statistics_current
from shared.auto_results import automatic_result_candidates
from shared.region_analysis import (
    atlas_data,
    extract_connectome,
    inspect_run,
    native_streamlines,
    read_json,
    run_analysis,
    select_streamlines,
    sha256,
    signatures_current,
    write_json,
)


class RegionAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.affine = np.diag([2.0, 3.0, 4.0, 1.0])
        self.affine[:3, 3] = [-10, -20, 5]
        self.labels = np.zeros((7, 5, 5), dtype=np.int32)
        self.labels[1:3, 1:4, 1:4] = 1001
        self.labels[4:6, 1:4, 1:4] = 2001
        self.atlas = self.volume("atlas.nii.gz", self.labels)
        self.fa = self.volume("fa.nii.gz", np.full(self.labels.shape, 0.6, dtype=np.float32))
        self.mask = self.volume("input/nodif_brain_mask.nii.gz", np.ones(self.labels.shape, dtype=np.uint8))
        self.valid = self.volume("valid.nii.gz", (self.labels == 1001).astype(np.uint8))
        self.lut = self.root / "labels.txt"
        self.lut.write_text("0 Background 0 0 0 0\n1001 Left-Region 80 100 200 0\n2001 Right-Region 200 100 80 0\n")
        voxel_lines = [
            [[0.0, 2.0, 2.0], [6.0, 2.0, 2.0]],  # two-point line crosses both ROIs
            [[1.0, 2.0, 2.0], [2.0, 2.0, 2.0]],  # both endpoints in left ROI
            [[4.0, 2.0, 2.0], [6.0, 2.0, 2.0]],  # one endpoint in right ROI
            [[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]],  # no ROI
            [[-2.0, 2.0, 2.0], [8.0, 2.0, 2.0]],  # out-of-grid endpoints, crosses both
        ]
        self.lines = [np.array(line) @ self.affine[:3, :3].T + self.affine[:3, 3] for line in voxel_lines]
        self.tracts = self.root / "tracts.trk"
        nib.streamlines.save(nib.streamlines.Tractogram(self.lines, affine_to_rasmm=np.eye(4)), str(self.tracts))
        self.weights = self.root / "weights.txt"
        np.savetxt(self.weights, [0.1, 0.2, 0.3, 0.4, 0.5])
        self.nodes = self.root / "nodes.tsv"
        # node_id is deliberately different from the atlas label.
        self.nodes.write_text("node_id\tlabel\tname\n1\t1001\tLeft\n2\t2001\tRight\n")
        self.matrix = self.root / "edges.csv"
        self.matrix.write_text("source_node,target_node,streamline_count\n1001,2001,7\n")
        self.graph_stage = self.root / "connectome_manifest.json"
        write_json(
            self.graph_stage,
            dict(
                status="completed",
                symmetric=True,
                dependencies={"ddparcel_parcellation": {"path": str(self.atlas), "sha256": sha256(self.atlas)}},
            ),
        )
        self.tasks = [
            self.task("parcellation.ddparcel", {"wmparc": str(self.atlas)}),
            self.task("microstructure.dti", {"fa": str(self.fa)}),
            self.task(
                "microstructure.noddi",
                {"noddi_ndi": str(self.fa), "noddi_odi": str(self.fa), "noddi_tissue_valid_mask": str(self.valid)},
            ),
            self.task("tractography.csd_act", {"tractogram": str(self.tracts), "sift2_weights": str(self.weights)}),
            self.task(
                "connectome.streamline_count",
                {"matrix_path": str(self.matrix), "nodes_path": str(self.nodes)},
                manifest_path=str(self.graph_stage),
            ),
        ]
        self.manifest = self.root / "parent.json"
        self.save_parent()

    def volume(self, name, values, affine=None):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(values, self.affine if affine is None else affine), path)
        return path

    def task(self, name, outputs, **fields):
        qc = self.root / (name + "_qc.json")
        write_json(
            qc,
            dict(
                status="completed",
                automatic_qc_passed=True,
                source_signatures={key: {"path": path, "sha256": sha256(path)} for key, path in outputs.items()},
            ),
        )
        return dict(name=name, status="completed", outputs=outputs, qc_path=str(qc), **fields)

    def save_parent(self):
        write_json(
            self.manifest,
            dict(
                kind="slicerddsurfer_product_orchestration",
                status="completed",
                subject_id="synthetic",
                subject_root=str(self.root),
                output_root=str(self.root / "outputs"),
                run_name="test",
                tasks=self.tasks,
            ),
        )

    def catalog(self):
        return inspect_run(self.manifest, lut_path=self.lut)

    def request(self, name="analysis", **kwargs):
        catalog = self.catalog()
        return dict(
            source_manifest=str(self.manifest),
            lut_path=str(self.lut),
            manifest_sha256=catalog["manifest_sha256"],
            atlas_sha256=catalog["atlas_sha256"],
            lut_sha256=catalog["lut_sha256"],
            selected_labels=[1001, 2001],
            metrics=["fa", "noddi_ndi"],
            output_dir=str(self.root / name),
            **kwargs,
        )

    def test_catalog_uses_labels_names_and_physical_volume(self):
        catalog = self.catalog()
        self.assertEqual([row["id"] for row in catalog["regions"]], [1001, 2001])
        self.assertEqual(catalog["regions"][0]["name"], "Left-Region")
        self.assertAlmostEqual(catalog["regions"][0]["volume_mm3"], 18 * 24)
        self.assertEqual(catalog["connectomes"]["connectome.streamline_count"]["atlas"], str(self.atlas))

    def test_direct_cli_and_json_produce_identical_regional_statistics(self):
        import contextlib
        import io
        from scripts import analyze_regions

        run_analysis(self.request('from_json'))
        with patch.object(analyze_regions.signal, 'signal'), contextlib.redirect_stdout(io.StringIO()):
            analyze_regions.main(['--source-manifest', str(self.manifest), '--regions', 'Left-Region', '2001',
                                  '--metrics', 'fa,noddi_ndi', '--lut', str(self.lut),
                                  '--output-dir', str(self.root / 'from_cli')])
        for name in ('selected_regions.csv', 'selected_microstructure.csv'):
            self.assertEqual((self.root / 'from_json' / name).read_bytes(), (self.root / 'from_cli' / name).read_bytes())
        results = [read_json(self.root / folder / 'selected_microstructure.json') for folder in ('from_json', 'from_cli')]
        for result in results:
            result.pop('runtime_seconds')
        self.assertEqual(*results)

    def test_direct_cli_exports_two_subjects_without_a_batch_json(self):
        import contextlib
        import io
        from scripts import analyze_regions

        parent = read_json(self.manifest)
        parent['subject_id'] = 'synthetic2'
        second = self.root / 'parent2.json'
        write_json(second, parent)
        output = self.root / 'cli_cohort'
        with patch.object(analyze_regions.signal, 'signal'), contextlib.redirect_stdout(io.StringIO()):
            analyze_regions.main(['--source-manifest', str(self.manifest), str(second),
                                  '--regions', '1001', '2001', '--metrics', 'fa', '--lut', str(self.lut),
                                  '--output-dir', str(output)])
        summary = read_json(output / 'regional_batch_manifest.json')
        self.assertEqual(summary['status'], 'completed')
        self.assertEqual(summary['completed_subjects'], 2)
        with (output / 'cohort_selected_microstructure.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 4)
        self.assertEqual({row['subject_id'] for row in rows}, {'synthetic', 'synthetic2'})

    def test_failed_stale_and_missing_qc_excluded(self):
        self.tasks[1]["status"] = "failed"
        Path(self.tasks[2]["qc_path"]).write_text('{"status":"completed","checks":{"fit":false}}')
        self.tasks[3]["qc_path"] = str(self.root / "missing.json")
        self.save_parent()
        result = self.catalog()
        self.assertFalse(result["metrics"])
        self.assertFalse(result["tractograms"])
        self.assertEqual(len(result["warnings"]), 3)

    def test_failed_retry_does_not_reuse_earlier_success(self):
        self.tasks.extend(
            [
                dict(name="microstructure.dti", status="failed", outputs={}),
                dict(name="tractography.csd_act", status="failed", outputs={}),
            ]
        )
        self.save_parent()
        result = self.catalog()
        self.assertNotIn("fa", result["metrics"])
        self.assertFalse(result["tractograms"])

    def test_changed_quantitative_file_and_failed_stage_excluded(self):
        with self.fa.open("ab") as stream:
            stream.write(b"changed")
        write_json(self.graph_stage, dict(status="failed"))
        result = self.catalog()
        self.assertFalse(result["metrics"])
        self.assertFalse(result["connectomes"])

    def test_noddi_and_fw_require_explicit_tissue_masks(self):
        self.tasks[2]["outputs"].pop("noddi_tissue_valid_mask")
        self.tasks.append(
            self.task("microstructure.freewater", dict(fwf=str(self.fa), fw_fa=str(self.fa), fw_md=str(self.fa)))
        )
        self.save_parent()
        result = self.catalog()
        self.assertNotIn("noddi_ndi", result["metrics"])
        self.assertNotIn("fw_tissue_fa", result["metrics"])
        self.assertIn("fwf", result["metrics"])

    def test_fw_resumed_output_key_aliases(self):
        self.tasks.append(
            self.task(
                "microstructure.freewater",
                dict(fwf=str(self.fa), fw_fa=str(self.fa), fw_md=str(self.fa), valid_mask=str(self.valid)),
            )
        )
        self.save_parent()
        self.assertIn("fw_tissue_fa", self.catalog()["metrics"])

    def test_noddi_resumed_and_parallel_output_key_aliases(self):
        self.tasks[2] = self.task(
            "microstructure.noddi",
            dict(ndi=str(self.fa), odi=str(self.fa), isovf=str(self.fa), valid_mask=str(self.valid)),
        )
        self.save_parent()
        self.assertTrue({"noddi_ndi", "noddi_odi", "noddi_fwf"}.issubset(self.catalog()["metrics"]))

    def test_raw_dki_and_corrected_mkcurve_masks(self):
        self.tasks.append(
            self.task(
                "microstructure.dki",
                dict(
                    mk_raw=str(self.fa),
                    mk=str(self.fa),
                    valid_mask=str(self.valid),
                    mkcurve_corrected_mk=str(self.fa),
                    mkcurve_corrected_mk_valid_mask=str(self.valid),
                ),
            )
        )
        self.save_parent()
        metrics = self.catalog()["metrics"]
        self.assertEqual(metrics["dki_mk"]["valid_mask"], str(self.valid))
        self.assertEqual(metrics["mkcurve_corrected_mk"]["valid_mask"], str(self.valid))

    def test_ukf_named_artifact_digest_is_checked(self):
        stage = self.root / "ukf_stage.json"
        write_json(stage, dict(status="completed", artifacts={"tractogram": "0" * 64}))
        self.tasks[3]["manifest_path"] = str(stage)
        self.save_parent()
        self.assertFalse(self.catalog()["tractograms"])

    def test_recorded_fitting_brain_mask_overrides_current_folder(self):
        original = self.volume("original_brain.nii.gz", (self.labels > 0).astype(np.uint8))
        qc = read_json(self.tasks[1]["qc_path"])
        qc["source_signature"] = {"mask": {"path": str(original), "sha256": sha256(original)}}
        write_json(self.tasks[1]["qc_path"], qc)
        self.assertEqual(self.catalog()["brain_mask"], str(original))

    def test_custom_atlas_excludes_incompatible_connectomes(self):
        alternate = self.volume("other_atlas.nii.gz", np.ones_like(self.labels))
        result = inspect_run(self.manifest, atlas_override=alternate)
        self.assertFalse(result["connectomes"])
        self.assertEqual([row["id"] for row in result["regions"]], [1])

    def test_preview_and_running_run_are_rejected(self):
        preview = self.root / "display_preview.trk"
        preview.write_bytes(self.tracts.read_bytes())
        self.tasks[3] = self.task("tractography.csd_act", {"tractogram": str(preview)})
        self.save_parent()
        self.assertFalse(self.catalog()["tractograms"])
        record = read_json(self.manifest)
        record["status"] = "running"
        write_json(self.manifest, record)
        with self.assertRaisesRegex(ValueError, "finish"):
            self.catalog()

    def test_streamline_rules_preserve_coordinates_indices_and_weights(self):
        image, labels = atlas_data(self.atlas)
        expected = {"through_any": [0, 1, 2, 4], "through_all": [0, 4], "endpoint_any": [1, 2], "endpoint_both": [1]}
        original = list(native_streamlines(self.tracts))
        for mode, indices in expected.items():
            with self.subTest(mode=mode):
                directory = self.root / mode
                directory.mkdir()
                result = select_streamlines(
                    self.tracts, image, labels, [1001, 2001], mode, directory, weights_path=self.weights
                )
                self.assertEqual(result["selected_streamlines"], len(indices))
                with Path(result["source_indices"]).open() as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual([int(row["source_index"]) for row in rows], indices)
                np.testing.assert_allclose(
                    [float(row["sift2_weight"]) for row in rows], np.loadtxt(self.weights)[indices]
                )
                actual = list(native_streamlines(result["output"]))
                for line, index in zip(actual, indices):
                    np.testing.assert_array_equal(line, original[index])
                self.assertLess(Path(result["preview"]).stat().st_size, 32 * 1024**2)

    def test_no_matching_lines_are_not_fabricated(self):
        directory = self.root / "empty"
        directory.mkdir()
        image, labels = atlas_data(self.atlas)
        result = select_streamlines(self.tracts, image, labels, [999], "through_any", directory)
        self.assertEqual(result["selected_streamlines"], 0)
        self.assertIsNone(result["output"])
        self.assertIsNone(result["preview"])

    def test_vtk_header_lps_is_converted_to_ras(self):
        from shared.region_analysis import VTKSpool

        spool = VTKSpool(self.root, "lps")
        original = np.array(self.lines[0], dtype=np.float64)
        lps = original.copy()
        lps[:, :2] *= -1
        spool.append(lps)
        path = self.root / "lps.vtk"
        spool.finish(path)
        path.write_bytes(path.read_bytes().replace(b"SPACE=RAS", b"SPACE=LPS"))
        np.testing.assert_array_equal(list(native_streamlines(path))[0], original)

    def test_weight_count_mismatch_fails(self):
        self.weights.write_text("1\n")
        image, labels = atlas_data(self.atlas)
        folder = self.root / "bad_weights"
        folder.mkdir()
        with self.assertRaisesRegex(ValueError, "counts disagree"):
            select_streamlines(self.tracts, image, labels, [1001], "through_any", folder, weights_path=self.weights)

    def test_sparse_graph_uses_atlas_labels_not_node_ids(self):
        directory = self.root / "graph"
        directory.mkdir()
        source = self.catalog()["connectomes"]["connectome.streamline_count"]
        info = extract_connectome(source, self.atlas, [1001, 2001], directory)
        with Path(info["matrix"]).open() as stream:
            rows = list(csv.reader(stream))
        self.assertEqual(rows[0], ["label_id", "1001", "2001"])
        self.assertEqual(float(rows[1][2]), 7.0)
        self.assertEqual(float(rows[2][1]), 7.0)
        self.assertEqual(float(rows[1][1]), 0.0)

    def test_graph_atlas_mismatch_and_conflicting_edges_fail(self):
        source = self.catalog()["connectomes"]["connectome.streamline_count"]
        directory = self.root / "bad_graph"
        directory.mkdir()
        alternate = self.volume("alternate.nii.gz", np.ones_like(self.labels))
        with self.assertRaisesRegex(ValueError, "differs"):
            extract_connectome(source, alternate, [1001], directory)
        with self.matrix.open("a") as stream:
            stream.write("2001,1001,8\n")
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            extract_connectome(source, self.atlas, [1001], directory)

    def test_dense_graph_preserves_weights_and_refuses_asymmetry(self):
        source = self.catalog()["connectomes"]["connectome.streamline_count"]
        source["backend"] = "mrtrix3.tck2connectome"
        self.matrix.write_text("0,3.25\n3.25,0\n")
        directory = self.root / "dense"
        directory.mkdir()
        info = extract_connectome(source, self.atlas, [1001, 2001], directory)
        self.assertIn("3.25", Path(info["matrix"]).read_text())
        self.matrix.write_text("0,1\n2,0\n")
        with self.assertRaisesRegex(ValueError, "Asymmetric"):
            extract_connectome(source, self.atlas, [1001], directory)

    def test_end_to_end_stats_track_graph_and_scene_candidates(self):
        request = self.request(tractogram="tractography.csd_act", connectome="connectome.streamline_count")
        record = read_json(run_analysis(request))
        self.assertEqual(record["status"], "completed")
        self.assertEqual(record["human_review"], "pending")
        folder = Path(request["output_dir"])
        stats = read_json(folder / "selected_microstructure.json")
        self.assertAlmostEqual(stats["rows"][0]["fa_mean"], 0.6, places=6)
        self.assertIsNone(stats["rows"][1]["noddi_ndi_mean"])
        self.assertEqual(stats["rows"][1]["noddi_ndi_valid_voxel_count"], 0)
        self.assertTrue(record["warnings"])
        for path, digest in record["source_sha256"].items():
            self.assertEqual(sha256(path), digest)
        self.assertTrue(
            roi_statistics_current(
                folder / "selected_microstructure.json",
                folder / "selected_microstructure.csv",
                selected_labels=(1001, 2001),
            )
        )
        self.assertFalse(
            roi_statistics_current(folder / "selected_microstructure.json", folder / "selected_microstructure.csv")
        )
        selected, skipped, deferred = automatic_result_candidates(record)
        tracks = [item for item in selected if item["product"] == "tractography.regions"]
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]["key"], "display_preview")
        self.assertEqual(tracks[0]["quantitative_source_path"], str(folder / "selected_streamlines.vtk"))
        self.assertTrue(any(item["display_role"] == "categorical:region_selection" for item in selected))
        self.assertEqual(len([item for item in selected if item["product"] == "statistics.regions"]), 3)
        self.assertFalse(skipped)
        self.assertTrue(deferred)
        from scripts.validate_region_analysis import validate

        self.assertEqual(validate(folder / "region_analysis_manifest.json")["status"], "PASS")

    def test_source_manifest_changed_since_selection_rejected(self):
        request = self.request()
        self.tasks[1]["status"] = "failed"
        self.save_parent()
        with self.assertRaisesRegex(ValueError, "changed"):
            run_analysis(request)

    def test_lut_changed_since_selection_rejected(self):
        request = self.request()
        self.lut.write_text("1001 Right-Region 80 100 200 0\n2001 Left-Region 200 100 80 0\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            run_analysis(request)

    def test_grid_mismatch_fails_without_publishing_pass(self):
        self.volume("fa.nii.gz", np.ones(self.labels.shape, dtype=np.float32), affine=np.eye(4))
        self.tasks[1] = self.task("microstructure.dti", {"fa": str(self.fa)})
        self.save_parent()
        request = self.request()
        request["metrics"] = ["fa"]
        with self.assertRaisesRegex(ValueError, "affine"):
            run_analysis(request)
        record = read_json(Path(request["output_dir"]) / "region_analysis_manifest.json")
        self.assertEqual(record["status"], "failed")
        self.assertFalse(read_json(Path(request["output_dir"]) / "region_analysis_qc.json")["automatic_qc_passed"])
        self.assertFalse(automatic_result_candidates(record)[0])

    def test_existing_outputs_are_never_overwritten(self):
        request = self.request()
        folder = Path(request["output_dir"])
        folder.mkdir()
        existing = folder / "selected_microstructure.csv"
        existing.write_text("previous result")
        with self.assertRaisesRegex(ValueError, "fresh"):
            run_analysis(request)
        self.assertEqual(existing.read_text(), "previous result")

    def test_source_changes_during_processing_fail_final_qc(self):
        request = self.request(tractogram="tractography.csd_act")
        from shared.region_analysis import select_streamlines as original

        def changed(*args, **kwargs):
            info = original(*args, **kwargs)
            self.lut.write_text("changed")
            return info

        with patch("shared.region_analysis.select_streamlines", side_effect=changed):
            with self.assertRaisesRegex(RuntimeError, "changed during"):
                run_analysis(request)
        record = read_json(Path(request["output_dir"]) / "region_analysis_manifest.json")
        self.assertEqual(record["status"], "failed")
        self.assertTrue(all(task["status"] == "failed" for task in record["tasks"]))

    def test_cancelled_job_is_never_completed_or_loadable(self):
        request = self.request(tractogram="tractography.csd_act")
        with patch("shared.region_analysis.select_streamlines", side_effect=KeyboardInterrupt("cancelled")):
            with self.assertRaises(KeyboardInterrupt):
                run_analysis(request)
        record = read_json(Path(request["output_dir"]) / "region_analysis_manifest.json")
        self.assertEqual(record["status"], "cancelled")
        self.assertTrue(all(task["status"] == "cancelled" for task in record["tasks"]))
        self.assertFalse(automatic_result_candidates(record)[0])

    def test_invalid_selection_rejected(self):
        for labels in ([], [0], [True], ["1001"], [999]):
            with self.subTest(labels=labels):
                request = self.request()
                request["selected_labels"] = labels
                with self.assertRaises(ValueError):
                    run_analysis(request)

    def test_signature_size_and_digest_dictionaries(self):
        signature = dict(path=str(self.fa), size_bytes=self.fa.stat().st_size, sha256=sha256(self.fa))
        self.assertTrue(signatures_current(signature))
        signature["size_bytes"] += 1
        self.assertFalse(signatures_current(signature))
        self.assertFalse(signatures_current({"artifact_sha256": {str(self.fa): "wrong"}}))

    def test_name_alias_search_is_atlas_scoped_and_ids_are_exact(self):
        from shared.region_selection import matches_region, resolve_regions

        regions = [dict(id=17, name="Left-Hippocampus"), dict(id=53, name="Right-Hippocampus"),
                   dict(id=1024, name="ctx-lh-precentral"), dict(id=2024, name="ctx-rh-precentral")]
        self.assertTrue(matches_region(regions[0], "左侧海马"))
        self.assertFalse(matches_region(regions[1], "左海马"))
        self.assertTrue(matches_region(regions[2], "左中央前回"))
        self.assertFalse(matches_region(regions[2], "102"))
        self.assertFalse(matches_region(regions[2], "???"))
        self.assertTrue(matches_region(regions[2], "17, 1024"))
        self.assertEqual(resolve_regions(regions, "左海马,2024"), [17, 2024])
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            resolve_regions(regions, ["海马"])
        with self.assertRaisesRegex(ValueError, "Unknown"):
            resolve_regions(regions, ["not-a-region"])
        with self.assertRaises(ValueError):
            resolve_regions(regions, [True])

    def add_endpoint_pair_lines(self):
        self.lines.extend([self.lines[1][[0, -1]].copy(), self.lines[1][[0, -1]].copy()])
        self.lines[-2] = np.array([[1, 2, 2], [4, 2, 2]]) @ self.affine[:3, :3].T + self.affine[:3, 3]
        self.lines[-1] = self.lines[-2][::-1].copy()
        nib.streamlines.save(nib.streamlines.Tractogram(self.lines, affine_to_rasmm=np.eye(4)), str(self.tracts))
        np.savetxt(self.weights, np.arange(1, 8) / 10)
        self.tasks[3] = self.task("tractography.csd_act", {"tractogram": str(self.tracts), "sift2_weights": str(self.weights)})
        self.save_parent()

    def test_pair_rule_requires_opposite_endpoints_and_is_orientation_independent(self):
        self.add_endpoint_pair_lines()
        request = self.request(tractogram="tractography.csd_act", tract_rule="endpoint_pair", target_regions=["Right-Region"])
        request["selected_regions"] = ["Left-Region"]
        path = run_analysis(request)
        info = read_json(path.parent / "selected_tractography.json")
        with Path(info["source_indices"]).open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([int(row["source_index"]) for row in rows], [5, 6])
        self.assertEqual([float(row["sift2_weight"]) for row in rows], [0.6, 0.7])
        self.assertAlmostEqual(info["length_mean_mm"], 6)
        candidates = automatic_result_candidates(read_json(path))[0]
        region_map = next(item for item in candidates if item["display_role"] == "categorical:region_selection")
        self.assertEqual(region_map["region_label_ids"], [1001, 2001])
        from scripts.validate_region_analysis import validate

        self.assertTrue(validate(path)["checks"]["endpoint_selection_membership"])

    def test_waypoints_and_exclusion_preserve_full_original_lines(self):
        image, labels = atlas_data(self.atlas)
        folder = self.root / "excluded"
        folder.mkdir()
        info = select_streamlines(self.tracts, image, labels, [1001], "endpoint_any", folder,
                                  exclude_labels=[2001])
        self.assertEqual(info["selected_streamlines"], 1)
        np.testing.assert_array_equal(list(native_streamlines(info["output"]))[0], self.lines[1])
        included = self.root / "included"
        included.mkdir()
        info = select_streamlines(self.tracts, image, labels, [1001], "through_any", included,
                                  include_labels=[2001])
        self.assertEqual(info["selected_streamlines"], 2)

    def test_pair_target_must_be_nonempty_disjoint_and_requires_tracts(self):
        image, labels = atlas_data(self.atlas)
        for target in ([], [1001]):
            with self.assertRaisesRegex(ValueError, "disjoint"):
                select_streamlines(self.tracts, image, labels, [1001], "endpoint_pair", self.root, target_labels=target)
        with self.assertRaisesRegex(ValueError, "require a tractogram"):
            run_analysis(self.request(target_labels=[2001]))

    def test_quantiles_bilateral_comparison_and_local_report(self):
        request = self.request()
        path = run_analysis(request)
        stats = read_json(path.parent / "selected_microstructure.json")["rows"]
        self.assertAlmostEqual(stats[0]["fa_median"], 0.6, places=6)
        self.assertEqual(stats[0]["fa_iqr"], 0)
        self.assertIsNone(stats[1]["noddi_ndi_median"])
        with (path.parent / "bilateral_statistics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(next(row for row in rows if row["metric"] == "fa")["asymmetry_index"], "0.0")
        self.assertEqual(next(row for row in rows if row["metric"] == "noddi_ndi")["asymmetry_index"], "")
        report = Path(read_json(path)["report_path"])
        self.assertIn("Left–right comparison", report.read_text(encoding="utf-8"))
        self.assertNotIn("<script", report.read_text(encoding="utf-8"))
        self.assertIn(str(report), read_json(path)["artifact_sha256"])

    def test_tissue_composition_includes_unclassified_and_checks_affine(self):
        data = np.zeros(self.labels.shape, np.int16)
        data[self.labels == 1001] = 1
        data[self.labels == 2001] = 2
        tissue = self.volume("segmentation.nii.gz", data)
        self.tasks.append(self.task("segmentation.ddseg", dict(segmentation=str(tissue))))
        self.save_parent()
        request = self.request(tissue=True)
        request["metrics"] = []
        path = run_analysis(request)
        with (path.parent / "selected_tissue_statistics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(float(rows[0]["wm_fraction"]), 1.0)
        self.assertEqual(float(rows[1]["gm_fraction"]), 1.0)
        from scripts.validate_region_analysis import validate

        self.assertTrue(validate(path)["checks"]["independent_tissue_counts"])
        tissue = self.volume("segmentation.nii.gz", data, np.eye(4))
        self.tasks[-1] = self.task("segmentation.ddseg", dict(segmentation=str(tissue)))
        self.save_parent()
        with self.assertRaisesRegex(ValueError, "affine"):
            run_analysis(self.request("bad_tissue_grid", tissue=True))

    def add_surface_fixture(self):
        self.lut.write_text("1001 ctx-lh-precentral 80 100 200 0\n2001 ctx-rh-precentral 200 100 80 0\n")
        surf, label = self.root / "surf", self.root / "label"
        surf.mkdir()
        label.mkdir()
        for hemi in ("lh", "rh"):
            nib.freesurfer.write_annot(str(label / f"{hemi}.aparc.annot"), np.array([1, 1, 0, 0]),
                np.array([[0, 0, 0, 0], [80, 120, 200, 0]]), ["unknown", "precentral"])
            for key, values in (("thickness", [2, 4, 0, 0]), ("area", [3, 5, 0, 0]), ("curv", [-0.2, 0.4, 0, 0])):
                nib.freesurfer.write_morph_data(str(surf / f"{hemi}.{key}"), np.array(values, np.float32))
        task = self.task("surface.ddsurfer_postprocess", {})
        task["outputs"] = dict(surf=str(surf), label=str(label))
        self.tasks.append(task)
        self.save_parent()

    def test_cortex_matches_hemisphere_names_to_real_annotations(self):
        self.add_surface_fixture()
        catalog = self.catalog()
        self.assertEqual(set(catalog["surfaces"]), {"1001", "2001"})
        request = self.request(surface=True)
        path = run_analysis(request)
        with (path.parent / "selected_surface_statistics.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 2)
        self.assertEqual(float(rows[0]["area_mm2"]), 8.0)
        self.assertEqual(float(rows[0]["thickness_mean_mm"]), 3.0)
        from scripts.validate_region_analysis import validate

        self.assertTrue(validate(path)["checks"]["independent_surface_arrays"])
        self.lut.write_text("1001 Left-Hippocampus 80 100 200 0\n2001 Right-Hippocampus 200 100 80 0\n")
        self.assertFalse(self.catalog()["surfaces"])

    def test_incompatible_cortical_names_are_never_matched_by_label_id(self):
        self.add_surface_fixture()
        self.lut.write_text("1001 ctx-lh-notprecentral 80 100 200 0\n2001 wm-rh-precentral 200 100 80 0\n")
        self.assertFalse(self.catalog()["surfaces"])

    def test_failed_tissue_and_surface_tasks_are_not_discovered(self):
        self.add_surface_fixture()
        self.tasks[-1]["status"] = "failed"
        self.tasks.append(self.task("segmentation.ddseg", dict(segmentation=str(self.atlas))))
        self.tasks[-1]["status"] = "failed"
        self.save_parent()
        self.assertFalse(self.catalog()["surfaces"])
        self.assertIsNone(self.catalog()["tissue"])

    def test_saf_full_quantitative_output_can_be_used(self):
        self.tasks.append(self.task("tractography.saf", dict(tractogram=str(self.tracts))))
        self.save_parent()
        self.assertIn("tractography.saf", self.catalog()["tractograms"])

    def test_batch_aggregates_completed_subjects_and_retains_failure(self):
        from shared.region_batch import run_region_batch

        path = run_region_batch(dict(output_dir=str(self.root / "cohort"),
            request=dict(selected_regions=["Left-Region"], metrics=["fa"]),
            subjects=[dict(source_manifest=str(self.manifest), lut_path=str(self.lut)),
                      dict(source_manifest=str(self.root / "missing.json"))]))
        self.assertEqual(read_json(path)["completed_subjects"], 1)
        self.assertEqual(read_json(path)["status"], "partial_failed")
        with (path.parent / "cohort_selected_microstructure.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label_name"], "Left-Region")
        self.assertEqual(rows[0]["subject_id"], "synthetic")

    def test_batch_compatible_subjects_and_mismatched_label_semantics(self):
        from shared.region_batch import run_region_batch

        parent = read_json(self.manifest)
        parent["subject_id"] = "synthetic2"
        second = self.root / "parent2.json"
        write_json(second, parent)
        subjects = [dict(source_manifest=str(path), lut_path=str(self.lut)) for path in (self.manifest, second)]
        config = dict(output_dir=str(self.root / "two_subjects"), request=dict(selected_labels=[1001], metrics=["fa"]), subjects=subjects)
        manifest = run_region_batch(config)
        self.assertEqual(read_json(manifest)["status"], "completed")
        with (manifest.parent / "cohort_selected_microstructure.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row["subject_id"] for row in rows], ["synthetic", "synthetic2"])
        alternate = self.root / "different_lut.txt"
        alternate.write_text("1001 Left-Another-Region 80 100 200 0\n2001 Right-Region 200 100 80 0\n")
        config["output_dir"] = str(self.root / "mismatched_cohort")
        config["subjects"][1]["lut_path"] = str(alternate)
        manifest = run_region_batch(config)
        self.assertEqual(read_json(manifest)["completed_subjects"], 1)
        self.assertIn("IDs/names differ", read_json(manifest)["subjects"][1]["error"])

    def test_batch_cancellation_is_recorded_without_false_completion(self):
        from shared.region_batch import run_region_batch

        folder = self.root / "cancelled_cohort"
        config = dict(output_dir=str(folder), request=dict(selected_labels=[1001], metrics=["fa"]),
                      subjects=[dict(source_manifest=str(self.manifest), lut_path=str(self.lut))])
        with patch("shared.region_batch.run_analysis", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                run_region_batch(config)
        record = read_json(folder / "regional_batch_manifest.json")
        self.assertEqual(record["status"], "cancelled")
        self.assertEqual(record["subjects"][0]["status"], "cancelled")

    def test_ifw_metric_catalog_and_cli_export(self):
        from microstructure.ifw import IfwConfig, run_ifw
        from scripts import analyze_regions

        fw_path = self.volume("fw_fraction.nii.gz", np.full_like(self.labels, 0.25, dtype=np.float32))
        csf_path = self.volume("csf_prob.nii.gz", np.full_like(self.labels, 0.05, dtype=np.float32))
        mask_path = self.mask
        bval_path = self.root / "ifw.bval"
        bvec_path = self.root / "ifw.bvec"
        bval_path.write_text("0 1000 2000 2000\n", encoding="utf-8")
        bvec_path.write_text("0 1 0 0\n0 0 1 0\n0 0 0 1\n", encoding="utf-8")
        ifw_dir = self.root / "ifw_stage"
        res = run_ifw(IfwConfig(
            subject_id="synthetic",
            fw_fraction_path=fw_path,
            csf_probability_path=csf_path,
            mask_path=mask_path,
            bval_path=bval_path,
            bvec_path=bvec_path,
            output_dir=ifw_dir,
            echo_time_ms=89.5,
            roi_label_map=self.atlas,
            lut_path=self.lut,
        ))
        self.tasks.append(dict(
            name="microstructure.ifw",
            status="completed",
            manifest_path=str(res.record_path),
            outputs=dict(
                ifw=str(res.ifw_path),
                fw_volume_fraction=str(res.fw_volume_fraction_path),
                roi_stats_csv=str(res.roi_stats_csv_path),
            ),
        ))
        self.save_parent()
        cat = self.catalog()
        self.assertIn("ifw", cat["metrics"])
        self.assertIn("fw_volume_fraction", cat["metrics"])
        out_dir = self.root / "ifw_region_cli"
        analyze_regions.main([
            "--source-manifest", str(self.manifest),
            "--regions", "1001", "2001",
            "--metrics", "ifw",
            "--lut", str(self.lut),
            "--output-dir", str(out_dir),
        ])
        with (out_dir / "selected_microstructure.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([int(r["label_id"]) for r in rows], [1001, 2001])
        self.assertTrue(all(float(r["ifw_mean"]) > 0 for r in rows))
        # Corrupting output_sha256 in ifw_record.json must reject stale iFW map
        rec = read_json(res.record_path)
        rec["output_sha256"]["ifw"] = "0" * 64
        write_json(res.record_path, rec)
        self.assertNotIn("ifw", self.catalog()["metrics"])


if __name__ == "__main__":
    unittest.main()
