import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from shared.result_scene import (
    clear_run, owned_nodes, recorded_qc_summary, result_candidates, result_nodes, review_state_text,
)
from tractography.wma import REQUIRED_MAJOR_TRACTS


class ResultSelectionTest(unittest.TestCase):
    def test_tables_are_owned_reused_and_cleared_as_result_nodes(self):
        class Node:
            def __init__(self, identity, run):
                self.identity, self.run = identity, run

            def GetID(self):
                return self.identity

            def GetAttribute(self, key):
                return self.run if key == "SlicerDDSurfer.Run" else None

        volume, table, other = Node("volume", "run"), Node("table", "run"), Node("other", "other-run")
        removed = []
        slicer = SimpleNamespace(
            util=SimpleNamespace(getNodesByClass=lambda name: [volume, other] if name == "vtkMRMLDisplayableNode" else [table]),
            mrmlScene=SimpleNamespace(RemoveNode=removed.append))
        self.assertEqual(result_nodes(slicer), [volume, other, table])
        self.assertEqual(owned_nodes(slicer, "run"), [volume, table])
        self.assertEqual(clear_run(slicer, "run"), 2)
        self.assertEqual(removed, [volume, table])

    def test_bundle_qc_files_are_loaded_without_scanning_stale_siblings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            good, stale = root / "af.trk", root / "stale.trk"
            good.touch()
            stale.touch()
            qc = root / "qc.json"
            from tractography.wmql import WMQL_QC_VERSION
            qc.write_text(json.dumps({"status": "completed", "version": WMQL_QC_VERSION, 'automatic_qc_passed': True,
                "output_summary": {"bundle_files": [str(good)], 'bundle_count': 1,
                    'bundle_readable_count': 1, 'bundle_nonempty_count': 1,
                    'bundles': [{'query_name': 'af.left', 'streamline_count': 20}]}}))
            for name in ("bundles.wmql", "bundles.tractcloud"):
                if name == 'bundles.tractcloud':
                    from tractography.tractcloud import ADAPTER_REVISION, TRACTCLOUD_QC_VERSION
                    qc.write_text(json.dumps({'status': 'completed', 'version': TRACTCLOUD_QC_VERSION,
                        'automatic_qc_passed': True, 'executed': True, 'adapter_revision': ADAPTER_REVISION,
                        'output_summary': {'bundle_files': [str(good)], 'bundle_count': 1,
                            'bundle_readable_count': 1, 'bundle_nonempty_count': 1,
                            'assigned_streamline_count': 20}}))
                task = {"name": name, "status": "completed", "qc_path": str(qc), "outputs": {"output_dir": str(root)}}
                candidates, _ = result_candidates({"tasks": [task]})
                self.assertEqual([c["path"] for c in candidates], [str(good)])
                task["status"] = "failed"
                self.assertEqual(result_candidates({"tasks": [task]})[0], [])

    def test_tractcloud_qc_failure_cannot_leak_explicit_task_outputs(self):
        from tractography.tractcloud import ADAPTER_REVISION, TRACTCLOUD_QC_VERSION
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            bundle = root / 'AF.vtp'
            bundle.touch()
            qc_path = root / 'tractcloud_qc.json'
            record = dict(status='completed', version=TRACTCLOUD_QC_VERSION, executed=True,
                          adapter_revision=ADAPTER_REVISION, automatic_qc_passed=True,
                          output_summary=dict(bundle_count=1, bundle_readable_count=1,
                              bundle_nonempty_count=1, assigned_streamline_count=20, bundle_files=[str(bundle)]))
            task = dict(name='bundles.tractcloud', status='completed', qc_path=str(qc_path),
                        outputs={'bundle': str(bundle)})
            for override in ({'automatic_qc_passed': False}, {'error': 'failed geometry'},
                             {'checks': {'geometry': False}}, {'version': 3}, {'executed': False}):
                with self.subTest(override=override):
                    qc_path.write_text(json.dumps({**record, **override}))
                    self.assertEqual(result_candidates({'tasks': [task]})[0], [])
                    self.assertEqual(recorded_qc_summary({'tasks': [task]})['issues'], 1)

    def test_current_wma_only_uses_qc_declared_directory_and_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            folder = root / "anatomical"
            folder.mkdir()
            names = list(REQUIRED_MAJOR_TRACTS) + [f'other_{index}.vtp' for index in range(68)]
            for name in names:
                (folder / name).touch()
            qc = root / "qc.json"
            qc.write_text(json.dumps({"status": "completed", "version": 2, "automatic_qc_passed": True, "output_summary": {
                "anatomical_tracts_dir": str(folder), "anatomical_tract_count": 73,
                'anatomical_tract_readable_count': 73, 'anatomical_tract_nonempty_count': 73,
                'anatomical_tract_streamline_counts': {name: 20 for name in names}}}))
            task = {"name": "bundles.wma", "status": "skipped", "reason": "completed outputs found", "qc_path": str(qc)}
            self.assertEqual(len(result_candidates({"tasks": [task]})[0]), 73)
            (folder / "unexpected.vtp").touch()
            self.assertEqual(result_candidates({"tasks": [task]})[0], [])

    def test_old_green_bundle_record_is_not_accepted_without_current_scientific_qc(self):
        with tempfile.TemporaryDirectory() as tmp:
            qc = Path(tmp) / 'qc.json'
            for name, summary in [('bundles.wmql', {'bundle_files': []}),
                                  ('bundles.wma', {'anatomical_tract_count': 74,
                                   'anatomical_tract_readable_count': 74, 'anatomical_tract_nonempty_count': 45})]:
                qc.write_text(json.dumps({'status': 'completed', 'output_summary': summary}))
                task = {'name': name, 'status': 'completed', 'qc_path': str(qc)}
                self.assertEqual(recorded_qc_summary({'tasks': [task]})['issues'], 1)
                self.assertEqual(result_candidates({'tasks': [task]})[0], [])

    def test_failed_bundle_qc_cannot_enable_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            qc = Path(tmp) / "qc.json"
            qc.write_text(json.dumps({"status": "failed", "output_summary": {"bundle_files": ["/unrelated/a.trk"]}}))
            self.assertEqual(result_candidates({"tasks": [{"name": "bundles.wmql", "status": "completed", "qc_path": str(qc)}]})[0], [])

    def test_completed_execution_does_not_imply_passed_qc(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qc.json"
            path.write_text('{"status": "failed"}')
            summary = recorded_qc_summary({"tasks": [
                {"status": "completed", "qc_path": str(path)}, {"status": "completed"}]})
            self.assertEqual(summary, {"passed": 0, "issues": 1, "unavailable": 1})

    def test_sparse_sampling_is_reported_as_warning_not_an_all_clear(self):
        from shared.result_scene import record_qc_state
        self.assertEqual(record_qc_state(dict(status='completed', automatic_qc_passed=True,
            sampling_warnings=['EC: 1 streamline'])), 'warning')

    def test_failed_execution_cannot_borrow_historical_passed_qc(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'qc.json'
            path.write_text(json.dumps({'status': 'completed', 'automatic_qc_passed': True}))
            summary = recorded_qc_summary({'tasks': [
                {'name': 'microstructure.dti', 'status': 'failed', 'qc_path': str(path)}]})
            self.assertEqual(summary, {'passed': 0, 'issues': 1, 'unavailable': 0})
    def test_only_successful_manifest_outputs_are_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            # result_candidates() reports str(path.resolve()); resolve here
            # too so the comparison holds under macOS's /tmp -> /private/tmp.
            tmp = str(Path(tmp).resolve())
            good = Path(tmp) / "fa.nii.gz"
            good.touch()
            old = Path(tmp) / "old.nii.gz"
            old.touch()
            manifest = {"tasks": [
                {"name": "microstructure.dti", "status": "completed", "outputs": {"fa": str(good)}},
                {"name": "microstructure.noddi", "status": "failed", "outputs": {"ndi": str(old)}},
                {"name": "microstructure.other", "status": "skipped", "outputs": {"fa": str(old)}},
            ]}
            candidates, skipped = result_candidates(manifest)
            self.assertEqual([item["path"] for item in candidates], [str(good)])
            self.assertEqual(len(skipped), 2)

    def test_no_directory_fallback_and_missing_paths_are_explained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "stale.nii.gz").touch()
            manifest = {"output_root": tmp, "tasks": [{"name": "dti", "status": "completed",
                        "outputs": {"missing": str(root / "missing.nii.gz"), "relative": "fa.nii.gz"}}]}
            candidates, skipped = result_candidates(manifest)
            self.assertEqual(candidates, [])
            self.assertEqual(len(skipped), 2)

    def test_unique_files_prioritize_scalars_over_tractograms(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = str(Path(tmp).resolve())
            paths = [Path(tmp) / name for name in ("all.tck", "fa.nii.gz")]
            for path in paths:
                path.touch()
            candidates, _ = result_candidates({"tasks": [{"name": "result", "status": "reused",
                "outputs": {"tractogram": str(paths[0]), "fa": str(paths[1]), "duplicate": str(paths[1])}}]})
            self.assertEqual([item["path"] for item in candidates], [str(paths[1]), str(paths[0])])

    def test_pipeline_resume_is_accepted_but_unselected_skip_is_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fa.nii.gz"
            path.touch()
            for reason, expected in (("completed DTI microstructure outputs found", 1), ("not selected", 0)):
                candidates, _ = result_candidates({"tasks": [{"name": "microstructure.dti", "status": "skipped",
                    "reason": reason, "outputs": {"fa": str(path)}}]})
                self.assertEqual(len(candidates), expected)

    def test_include_brain_mask_surfaces_preprocessed_mask_only_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            subj = Path(tmp).resolve() / "sub-01"
            (subj / "input").mkdir(parents=True)
            (subj / "preprocessed").mkdir(parents=True)
            input_mask = subj / "input" / "nodif_brain_mask.nii.gz"
            preproc_mask = subj / "preprocessed" / "nodif_brain_mask.nii.gz"
            final_mask = subj / "preprocessed" / "sub-01_brain_mask_final.nii.gz"
            fa_path = subj / "outputs" / "02_microstructure" / "diffusion_tensor" / "sub-01_FA.nii.gz"
            fa_path.parent.mkdir(parents=True)
            for p in (input_mask, preproc_mask, final_mask, fa_path):
                p.touch()

            manifest = {
                "subject_root": str(subj),
                "output_root": str(subj / "outputs"),
                "brain_mask_path": str(preproc_mask),
                "tasks": [{"name": "microstructure.dti", "status": "completed", "outputs": {"fa": str(fa_path)}}],
            }
            # Default: excluded
            default_cands, _ = result_candidates(manifest)
            self.assertEqual([c["path"] for c in default_cands], [str(fa_path)])
            # include_brain_mask=True: includes preprocessed mask
            mask_cands, _ = result_candidates(manifest, include_brain_mask=True)
            self.assertEqual(
                {(c["product"], c["path"]) for c in mask_cands},
                {("microstructure.dti", str(fa_path)), ("preprocessing.brain_mask", str(preproc_mask))},
            )
            # input/ mask is never surfaced as preprocessing output
            input_manifest = dict(manifest, brain_mask_path=str(input_mask))
            self.assertEqual([c["path"] for c in result_candidates(input_manifest, include_brain_mask=True)[0]], [str(fa_path)])
            # If preprocessing.dwi already emitted final_mask, no duplicate nodif_brain_mask
            dwi_manifest = dict(manifest, tasks=[
                {"name": "preprocessing.dwi", "status": "completed", "outputs": {"mask": str(final_mask)}},
                {"name": "microstructure.dti", "status": "completed", "outputs": {"fa": str(fa_path)}},
            ])
            with_dwi, _ = result_candidates(dwi_manifest, include_preprocessing=True, include_brain_mask=True)
            self.assertEqual(
                {(c["product"], c["path"]) for c in with_dwi},
                {("preprocessing.dwi", str(final_mask)), ("microstructure.dti", str(fa_path))},
            )


class ReviewStateTextTest(unittest.TestCase):
    def test_review_state_text_has_no_pipes_and_one_format(self):
        qc = {"passed": 4, "issues": 0, "unavailable": 1}
        self.assertEqual(
            review_state_text(qc, issues_found=False),
            "Recorded QC: 4 passed · 0 issues · 1 unavailable\nVisual review: pending",
        )

    def test_review_state_text_prefixes_execution_issues(self):
        qc = {"passed": 1, "issues": 2, "unavailable": 0}
        text = review_state_text(qc, issues_found=True)
        self.assertEqual(
            text,
            "Execution: issues found\nRecorded QC: 1 passed · 2 issues · 0 unavailable\nVisual review: pending",
        )
        self.assertNotIn("|", text)


class _FakeTable:
    def __init__(self, values):
        self.values = values

    def GetTableValue(self, label):
        return self.values[label]


class _FakeColor:
    """A colour table as Slicer reads FreeSurferColorLUT.txt: its unused alpha column (0) becomes alpha 0."""

    def __init__(self):
        self.names = {0: "Unknown", 2: "Left-Cerebral-White-Matter", 1001: "ctx-lh-bankssts"}
        self.values = {label: (0.1 * (i + 1), 0.2, 0.3, 0.0) for i, label in enumerate(self.names)}
        self.attributes = {}

    def GetID(self):
        return "vtkMRMLColorTableNode7"

    def GetNumberOfColors(self):
        return 1002

    def GetColorName(self, label):
        return self.names.get(label, "(none)")

    def GetLookupTable(self):
        return _FakeTable(self.values)

    def SetColor(self, label, name, r, g, b, a):
        self.names[label], self.values[label] = name, (r, g, b, a)

    def SetAttribute(self, key, value):
        self.attributes[key] = value

    def GetAttribute(self, key):
        return self.attributes.get(key)


class FreeSurferColorNodeTest(unittest.TestCase):
    """Slicer 5.x has no FreeSurfer colour table without the SlicerFreeSurfer extension (5.12.4, 2026-10-24):
    DDParcel labels fell back to the generic Labels table."""

    def _slicer(self, scene_ids=()):
        loaded, nodes = [], []

        def load(path):
            loaded.append(path)
            color = _FakeColor()
            nodes.append(color)
            return color

        scene = {identity: SimpleNamespace(GetID=lambda identity=identity: identity) for identity in scene_ids}
        return SimpleNamespace(
            mrmlScene=SimpleNamespace(GetNodeByID=scene.get),
            util=SimpleNamespace(loadColorTable=load, getNodesByClass=lambda _name: list(nodes))), loaded

    def test_bundled_lut_is_loaded_once_and_opaque(self):
        from shared.result_scene import FREESURFER_LUT, freesurfer_color_node

        self.assertTrue(FREESURFER_LUT.is_file())
        slicer, loaded = self._slicer()
        color = freesurfer_color_node(slicer)
        self.assertEqual(loaded, [str(FREESURFER_LUT)])
        self.assertEqual(color.GetLookupTable().GetTableValue(1001)[3], 1.0)
        self.assertEqual(color.GetLookupTable().GetTableValue(2)[3], 1.0)
        self.assertEqual(color.GetLookupTable().GetTableValue(0)[3], 0.0)    # background stays clear
        self.assertIs(freesurfer_color_node(slicer), color)
        self.assertEqual(len(loaded), 1)

    def test_slicers_own_freesurfer_table_is_used_when_present(self):
        from shared.result_scene import freesurfer_color_node

        slicer, loaded = self._slicer({"vtkMRMLColorTableNodeFileFreeSurferColorLUT.txt"})
        self.assertEqual(freesurfer_color_node(slicer).GetID(), "vtkMRMLColorTableNodeFileFreeSurferColorLUT.txt")
        self.assertEqual(loaded, [])


class TrkVisualizationSuppressionTest(unittest.TestCase):
    """Native .vtk/.vtp is visualized in Results; redundant .trk stays on disk only."""

    def test_ukf_suppresses_trk_when_vtk_exists_regardless_of_key_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            vtk_path = root / "100307_ukf.vtk"
            trk_path = root / "100307_ukf.trk"
            vtk_path.write_bytes(b"vtk")
            trk_path.write_bytes(b"trk")
            for outputs in (
                {"tractogram": str(vtk_path), "tractogram_trk": str(trk_path)},
                {"tractogram_trk": str(trk_path), "tractogram": str(vtk_path)},
            ):
                with self.subTest(order=list(outputs)):
                    candidates, skipped = result_candidates({"tasks": [
                        {"name": "tractography.ukf", "status": "completed", "outputs": outputs},
                    ]})
                    self.assertEqual([item["path"] for item in candidates], [str(vtk_path)])
                    self.assertEqual(skipped, [])
                    self.assertTrue(trk_path.is_file())

    def test_csd_act_suppresses_trk_and_keeps_vtk_and_tck(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            tck_path = root / "100307_act.tck"
            trk_path = root / "100307_act.trk"
            vtk_path = root / "100307_act.vtk"
            for path in (tck_path, trk_path, vtk_path):
                path.write_bytes(b"data")
            candidates, skipped = result_candidates({"tasks": [{
                "name": "tractography.csd_act",
                "status": "completed",
                "outputs": {
                    "tractogram": str(tck_path),
                    "tractogram_trk": str(trk_path),
                    "tractogram_vtk": str(vtk_path),
                },
            }]})
            self.assertEqual({item["path"] for item in candidates}, {str(tck_path), str(vtk_path)})
            self.assertEqual(skipped, [])

    def test_dti_det_prefers_vtk_when_present_and_keeps_trk_fallback_when_standalone(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            trk_path = root / "100307_dti.trk"
            vtk_path = root / "100307_dti.vtk"
            seeds_vtk = root / "seeds.vtk"
            trk_path.write_bytes(b"trk")
            seeds_vtk.write_bytes(b"seeds")
            # Standalone DTI-det (even with a non-streamline seed mesh): .trk stays viewable.
            standalone, _ = result_candidates({"tasks": [{
                "name": "tractography.dti_det",
                "status": "completed",
                "outputs": {"tractogram_trk": str(trk_path), "seed_mask": str(seeds_vtk)},
            }]})
            self.assertEqual({item["path"] for item in standalone}, {str(trk_path), str(seeds_vtk)})
            # Once native DTI VTK exists: .vtk is listed and .trk is omitted.
            vtk_path.write_bytes(b"vtk")
            with_vtk, skipped = result_candidates({"tasks": [{
                "name": "tractography.dti_det",
                "status": "completed",
                "outputs": {"tractogram_trk": str(trk_path), "tractogram_vtk": str(vtk_path)},
            }]})
            self.assertEqual([item["path"] for item in with_vtk], [str(vtk_path)])
            self.assertEqual(skipped, [])

    def test_wmql_keeps_trk_only_bundles_and_suppresses_trk_only_when_same_stem_vtk_exists(self):
        from tractography.wmql import WMQL_QC_VERSION

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            af_trk = root / "sub_desc-wmql_af.left.trk"
            af_vtk = root / "sub_desc-wmql_af.left.vtk"
            cst_trk = root / "sub_desc-wmql_cst.left.trk"
            af_trk.write_bytes(b"trk")
            cst_trk.write_bytes(b"trk")
            qc = root / "wmql_qc.json"
            qc.write_text(json.dumps({
                "status": "completed",
                "version": WMQL_QC_VERSION,
                "automatic_qc_passed": True,
                "output_summary": {
                    "bundle_files": [str(af_trk), str(cst_trk)],
                    "bundle_count": 2,
                    "bundle_readable_count": 2,
                    "bundle_nonempty_count": 2,
                    "bundles": [
                        {"query_name": "af.left", "streamline_count": 20},
                        {"query_name": "cst.left", "streamline_count": 20},
                    ],
                },
            }))
            task = {"name": "bundles.wmql", "status": "completed", "qc_path": str(qc), "outputs": {"output_dir": str(root)}}
            only_trk, _ = result_candidates({"tasks": [task]})
            self.assertEqual({item["path"] for item in only_trk}, {str(af_trk), str(cst_trk)})
            # If a matching same-stem .vtk exists for one bundle, only that bundle's .trk is suppressed.
            af_vtk.write_bytes(b"vtk")
            qc_data = json.loads(qc.read_text())
            qc_data["output_summary"]["bundle_files"] = [str(af_trk), str(af_vtk), str(cst_trk)]
            qc.write_text(json.dumps(qc_data))
            mixed, _ = result_candidates({"tasks": [task]})
            self.assertEqual({item["path"] for item in mixed}, {str(af_vtk), str(cst_trk)})

    def test_ukf_falls_back_to_trk_if_vtk_file_is_missing_on_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            missing_vtk = root / "100307_ukf.vtk"
            trk_path = root / "100307_ukf.trk"
            trk_path.write_bytes(b"trk")
            candidates, skipped = result_candidates({"tasks": [{
                "name": "tractography.ukf",
                "status": "completed",
                "outputs": {"tractogram": str(missing_vtk), "tractogram_trk": str(trk_path)},
            }]})
            self.assertEqual([item["path"] for item in candidates], [str(trk_path)])
            self.assertEqual([item["path"] for item in skipped], [str(missing_vtk)])
