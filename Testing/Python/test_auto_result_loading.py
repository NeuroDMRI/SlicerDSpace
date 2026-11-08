"""Ownership, display, preview provenance and world-geometry loader contracts."""

import hashlib
import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from shared.result_scene import source_stamp, tract_file_coordinate_system

with patch.dict(sys.modules, {"slicer": types.ModuleType("slicer")}):
    module = importlib.import_module("shared.module_logic")


class Node:
    def __init__(self, identity, kind="vtkMRMLScalarVolumeNode"):
        self.identity, self.kind, self.attributes = identity, kind, {}
        self.display = Mock()
        self.polydata = None

    def GetID(self):
        return self.identity

    def GetClassName(self):
        return self.kind

    def GetAttribute(self, key):
        return self.attributes.get(key)

    def SetAttribute(self, key, value):
        self.attributes[key] = value

    def SetName(self, name):
        self.name = name

    def IsA(self, name):
        return self.kind == name or name == "vtkMRMLScalarVolumeNode" and self.kind == "vtkMRMLLabelMapVolumeNode"

    def GetDisplayNode(self):
        return self.display

    def CreateDefaultDisplayNodes(self):
        pass

    def SetAndObservePolyData(self, data):
        self.polydata = data


class Scene:
    def __init__(self):
        self.nodes = []
        self.next_id = 0

    def AddNewNodeByClass(self, kind, name):
        self.next_id += 1
        node = Node("node" + str(self.next_id), kind)
        self.nodes.append(node)
        return node

    def IsNodeClassRegistered(self, kind):
        return kind == "vtkMRMLFiberBundleNode"

    def RemoveNode(self, node):
        self.nodes.remove(node)

    def GetNodeByID(self, identity):
        if identity.startswith("vtkMRMLColorTableNode"):
            return object()
        return next((node for node in self.nodes if node.GetID() == identity), None)


class Logic(module.SharedLogicMixin):
    def log(self, message):
        pass


class AutoResultLoadingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.scene = Scene()

        def load_volume(path, properties=None, **kwargs):
            return True, self.scene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "volume")

        self.fake = types.SimpleNamespace(
            mrmlScene=self.scene,
            vtkMRMLStorageNode=types.SimpleNamespace(CoordinateSystemRAS=0, CoordinateSystemLPS=1),
            util=types.SimpleNamespace(
                getNodesByClass=lambda kind: [node for node in self.scene.nodes if
                    (node.kind == "vtkMRMLTableNode") == (kind == "vtkMRMLTableNode")],
                loadVolume=load_volume,
                loadLabelVolume=lambda *a, **kw: self.scene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", "label"),
                setSliceViewerLayers=Mock(), resetSliceViews=Mock(), resetThreeDViews=Mock(),
            ),
        )
        self.addCleanup(patch.stopall)
        patch.object(module, "slicer", self.fake).start()

        def register(slicer, node, run, subject, run_name, product, source):
            node.SetAttribute("SlicerDDSurfer.Run", run)
            node.SetAttribute("SlicerDDSurfer.SourcePath", source)
            node.SetAttribute("SlicerDDSurfer.SourceStamp", source_stamp(source))

        patch("shared.result_scene.register_node", side_effect=register).start()
        self.logic = Logic()

    def manifest(self):
        self.fa = self.root / "microstructure.dti_fa.nii.gz"
        self.fa.write_bytes(b"fixture")
        md = self.root / "microstructure.dti_md.nii.gz"
        md.write_bytes(b"fixture")
        segmentation = self.root / "segmentation.ddseg.nii.gz"
        segmentation.write_bytes(b"fixture")
        path = self.root / "00_run_metadata" / "run_orchestration_manifest.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"subject_id": "s", "run_name": "run", "output_root": str(self.root), "tasks": [
            {"name": "microstructure.dti", "status": "completed", "outputs": {"fa": str(self.fa), "md": str(md)}},
            {"name": "segmentation.ddseg", "status": "completed", "outputs": {"segmentation": str(segmentation)}}]}))
        return path

    def test_automatic_load_reuse_refresh_only_changed_owned_source_and_show_once(self):
        manifest = self.manifest()
        unrelated = self.scene.AddNewNodeByClass("vtkMRMLModelNode", "user-owned")
        first = self.logic.loadKeyResultsFromManifest(manifest, automatic=True, max_items=1)
        self.assertEqual(len(first["loaded"]), 3)
        fa = next(item for item in first["loaded"] if item["display_role"] == "scalar:fa")
        node = self.scene.GetNodeByID(fa["node_ids"][0])
        self.assertEqual(first["view_node_ids"]["background"], node.GetID())
        node.display.SetAndObserveColorNodeID.assert_called_with("vtkMRMLColorTableNodeGrey")
        self.assertEqual(node.GetAttribute("SlicerDDSurfer.HumanReviewState"), "pending")
        self.assertEqual(node.GetAttribute("SlicerDDSurfer.QCState"), "unavailable")
        second = self.logic.loadKeyResultsFromManifest(manifest, automatic=True)
        self.assertFalse(second["loaded"])
        self.assertEqual(len(second["reused"]), 3)
        self.fake.util.setSliceViewerLayers.assert_called_once()
        stable_ids = {entry["path"]: entry["node_ids"] for entry in second["reused"] if entry["path"] != str(self.fa)}
        self.fa.write_bytes(b"new-scientific-fixture")
        third = self.logic.loadKeyResultsFromManifest(manifest, automatic=True)
        self.assertEqual(len(third["loaded"]), 1)
        self.assertIsNone(self.scene.GetNodeByID(node.GetID()))
        self.assertIn(unrelated, self.scene.nodes)
        self.assertEqual(stable_ids, {entry["path"]: entry["node_ids"] for entry in third["reused"]})

    def test_manual_reuse_does_not_reload_changed_source(self):
        path = self.manifest()
        first = self.logic.loadKeyResultsFromManifest(path)
        self.fa.write_bytes(b"changed-source-fixture")
        second = self.logic.loadKeyResultsFromManifest(path)
        self.assertFalse(second["loaded"])
        self.assertIn("Source changed", second["skipped"][-1]["reason"])
        self.assertEqual(len(self.scene.nodes), len(first["loaded"]))

    def test_failed_bundle_retracts_only_automatic_nodes_without_file_path(self):
        path = self.root / 'run.json'
        qc = self.root / 'failed_qc.json'
        qc.write_text(json.dumps({'status': 'failed', 'automatic_qc_passed': False}))
        for execution in ('completed', 'failed', 'cancelled'):
            with self.subTest(execution=execution):
                self.scene.nodes.clear()
                automatic, manual, other = [self.scene.AddNewNodeByClass('vtkMRMLModelNode', name)
                                            for name in ('automatic', 'manual', 'other')]
                for node in (automatic, manual, other):
                    node.SetAttribute('SlicerDDSurfer.Run', str(path))
                    node.SetAttribute('SlicerDDSurfer.SourcePath', str(self.root / node.GetID()))
                    node.SetAttribute('SlicerDDSurfer.Product', 'bundles.wmql' if node is not other else 'surface.ddsurfer')
                automatic.SetAttribute('SlicerDDSurfer.AutomaticLoaded', 'true')
                other.SetAttribute('SlicerDDSurfer.AutomaticLoaded', 'true')
                path.write_text(json.dumps({'tasks': [{'name': 'bundles.wmql', 'status': execution, 'qc_path': str(qc)}]}))
                result = self.logic.loadKeyResultsFromManifest(path, automatic=True)
                self.assertFalse(result['loaded'])
                self.assertEqual(self.scene.nodes, [manual, other])

    def test_failed_scalar_qc_retracts_previous_automatic_node(self):
        path = self.manifest()
        result = self.logic.loadKeyResultsFromManifest(path, automatic=True)
        identity = next(item['node_ids'][0] for item in result['loaded'] if item['path'] == str(self.fa))
        qc = self.root / 'failed_qc.json'
        qc.write_text(json.dumps({'status': 'failed', 'automatic_qc_passed': True}))
        manifest = json.loads(path.read_text())
        manifest['tasks'][0]['qc_path'] = str(qc)
        path.write_text(json.dumps(manifest))
        self.logic.loadKeyResultsFromManifest(path, automatic=True)
        self.assertIsNone(self.scene.GetNodeByID(identity))

    def test_automatic_reuse_does_not_take_ownership_of_manual_nodes(self):
        path = self.manifest()
        manual = self.logic.loadKeyResultsFromManifest(path)
        identities = [identity for entry in manual['loaded'] for identity in entry['node_ids']]
        reused = self.logic.loadKeyResultsFromManifest(path, automatic=True)
        self.assertEqual(len(reused['reused']), 3)
        self.assertFalse(reused['loaded'])
        for identity in identities:
            self.assertIsNone(self.scene.GetNodeByID(identity).GetAttribute('SlicerDDSurfer.AutomaticLoaded'))
        qc = self.root / 'failed_qc.json'
        qc.write_text(json.dumps({'status': 'failed'}))
        manifest = json.loads(path.read_text())
        for task in manifest['tasks']:
            task['qc_path'] = str(qc)
        path.write_text(json.dumps(manifest))
        self.logic.loadKeyResultsFromManifest(path, automatic=True)
        self.assertTrue(all(self.scene.GetNodeByID(identity) is not None for identity in identities))

    def test_changed_source_refresh_preserves_manual_review_nodes(self):
        path = self.manifest()
        result = self.logic.loadKeyResultsFromManifest(path)
        manual_identity = next(entry['node_ids'][0] for entry in result['loaded'] if entry['path'] == str(self.fa))
        self.fa.write_bytes(b'new result; manual scene remains an older reviewed version')
        automatic = self.logic.loadKeyResultsFromManifest(path, automatic=True)
        self.assertEqual(len(automatic['loaded']), 1)
        self.assertIsNotNone(self.scene.GetNodeByID(manual_identity))
        current_identity = automatic['loaded'][0]['node_ids'][0]
        again = self.logic.loadKeyResultsFromManifest(path, automatic=True)
        self.assertFalse(again['loaded'])
        self.assertEqual(next(entry['node_ids'] for entry in again['reused'] if entry['path'] == str(self.fa)), [current_identity])

    def test_automatic_loading_rejects_worker_thread_before_scene_mutation(self):
        self.fake.app = types.SimpleNamespace(thread=lambda: "gui")
        qt = types.SimpleNamespace(QThread=types.SimpleNamespace(currentThread=lambda: "worker"))
        with patch.dict(sys.modules, {"qt": qt}), self.assertRaisesRegex(RuntimeError, "GUI thread"):
            self.logic.loadKeyResultsFromManifest(self.root / "unread.json", automatic=True)
        self.assertFalse(self.scene.nodes)

    def test_persisted_tck_preview_is_capped_ras_and_original_file_unchanged(self):
        try:
            import nibabel as nib
            import numpy as np
            import vtk
        except ImportError:
            self.skipTest("nibabel/numpy/VTK required for preview geometry")
        source = self.root / "full.csd_act.tck"
        streamlines = [np.array([[index, 1, 2], [index, 3, 4]], dtype=np.float32) for index in range(130)]
        nib.streamlines.save(nib.streamlines.Tractogram(streamlines, affine_to_rasmm=np.eye(4)), str(source))
        original = hashlib.sha256(source.read_bytes()).hexdigest()
        seen_properties = []

        def load_model(path, kind, properties, **kwargs):
            seen_properties.append(properties)
            reader = vtk.vtkPolyDataReader()
            reader.SetFileName(path)
            reader.Update()
            node = self.scene.AddNewNodeByClass("vtkMRMLModelNode", "preview")
            node.SetAndObservePolyData(reader.GetOutput())
            return True, node

        self.fake.util.loadNodeFromFile = load_model
        with patch.object(nib.streamlines, "load", wraps=nib.streamlines.load) as load:
            self.assertTrue(self.logic._load_streamline_file_as_fiber_bundle(
                source, "preview", max_streamlines=30, preview_dir=self.root / "display_previews"))
        load.assert_called_once_with(source.as_posix(), lazy_load=True)
        node = self.scene.nodes[0]
        self.assertEqual(node.polydata.GetNumberOfLines(), 30)
        self.assertEqual(node.polydata.GetPoint(0), (0.0, 1.0, 2.0))
        self.assertEqual(node.polydata.GetPoint(59), (129.0, 3.0, 4.0))
        self.assertEqual(seen_properties, [{"coordinateSystem": 0}])
        preview = Path(node.GetAttribute("SlicerDDSurfer.DisplayPreviewPath"))
        self.assertTrue(preview.is_file())
        self.assertEqual(tract_file_coordinate_system(preview), "RAS")
        provenance = json.loads(Path(node.GetAttribute("SlicerDDSurfer.DisplayPreviewProvenancePath")).read_text())
        self.assertEqual(provenance["display_streamline_count"], 30)
        self.assertEqual(provenance["source_streamline_count"], 130)
        self.assertTrue(provenance["display_only"])
        self.assertEqual(node.GetAttribute("SlicerDDSurfer.SourcePath"), source.as_posix())
        self.assertEqual(node.GetAttribute("SlicerDDSurfer.SourceStreamlineCount"), "130")
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), original)

    def test_generic_model_respects_explicit_lps_metadata(self):
        try:
            import vtk
        except ImportError:
            self.skipTest("VTK required for coordinate metadata")
        data = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        points.InsertNextPoint(1, 2, 3)
        data.SetPoints(points)
        space = vtk.vtkStringArray()
        space.SetName("SPACE")
        space.InsertNextValue("LPS")
        data.GetFieldData().AddArray(space)
        source = self.root / "explicit_lps.vtp"
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(str(source))
        writer.SetInputData(data)
        writer.Write()
        node = Node("model", "vtkMRMLModelNode")
        loader = Mock(return_value=(True, node))
        self.fake.util.loadNodeFromFile = loader
        self.assertTrue(self.logic._load_fiber_bundle_node(source, "display"))
        self.assertEqual(loader.call_args.kwargs["properties"], {"coordinateSystem": 1})
        self.assertEqual(node.GetAttribute("SlicerDDSurfer.SourceCoordinateSystem"), "LPS")

    def test_nonidentity_tractogram_affine_is_applied_lazily_in_world_space(self):
        try:
            import nibabel as nib
            import numpy as np
            import vtk
        except ImportError:
            self.skipTest("nibabel/numpy/VTK required for world geometry")
        source = self.root / "world.trk"
        source.write_bytes(b"unit-fixture")
        affine = np.diag([2.0, 3.0, 4.0, 1.0])
        affine[:3, 3] = [10, 20, 30]
        tractogram = nib.streamlines.LazyTractogram(
            lambda: iter([np.array([[1, 2, 3], [2, 3, 4]], dtype=np.float32)]), affine_to_rasmm=affine)
        record = types.SimpleNamespace(tractogram=tractogram, header={"nb_streamlines": 1})
        with patch.object(nib.streamlines, "load", return_value=record):
            self.assertTrue(self.logic._load_streamline_file_as_fiber_bundle(source, "world"))
        self.assertEqual(self.scene.nodes[0].polydata.GetPoint(0), (12.0, 26.0, 42.0))

    def test_surface_and_color_map_presets_do_not_modify_review_claims(self):
        from shared.result_scene import automatic_display_preset
        surface = Node("pial", "vtkMRMLModelNode")
        automatic_display_preset(self.fake, surface, {
            "product": "surface.ddsurfer", "display_role": "surface:pial:lh",
            "path": "pial.stl", "qc_state": "failed"})
        surface.display.SetVisibility.assert_called_with(False)
        surface.display.SetOpacity.assert_called_with(0.3)
        self.assertEqual(surface.GetAttribute("SlicerDDSurfer.QCState"), "failed")
        ndi = Node("ndi")
        ndi.SetAttribute("SlicerDDSurfer.HumanReviewState", "reviewed")
        automatic_display_preset(self.fake, ndi, {
            "product": "microstructure.noddi", "display_role": "scalar:ndi", "path": "ndi.nii.gz"})
        ndi.display.SetAndObserveColorNodeID.assert_called_with("vtkMRMLColorTableNodeRainbow")
        self.assertEqual(ndi.GetAttribute("SlicerDDSurfer.HumanReviewState"), "reviewed")


if __name__ == "__main__":
    unittest.main()
