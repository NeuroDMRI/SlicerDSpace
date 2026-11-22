"""Batch 2, display lane, part 1: tractography and bundles load as native fiber bundles.

Request item: "Tractography and bundles should both load as fiber bundle; do not depend on other plugins;
adjust the small eye icon so the user knows the row is viewed as a fiber bundle."
Each test imports the code under test inside the test, so a missing name fails per test.
"""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

try:
    import numpy as np
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
except ImportError:  # pragma: no cover - plain python3 has no numpy/vtk
    np = vtk = None

from shared.result_scene import source_stamp

with patch.dict(sys.modules, {"slicer": types.ModuleType("slicer")}):
    module_logic = importlib.import_module("shared.module_logic")

DEC = "SlicerDSpace_DEC"


def _polydata(lines, *, permute=True, extra_cells=None):
    """Polylines given as lists of xyz points; point ids are stored in a permuted order on purpose."""
    coordinates = [point for line in lines for point in line]
    count = len(coordinates)
    order = np.arange(count)[::-1] if permute else np.arange(count)  # storage slot of logical point i
    stored = np.zeros((count, 3), dtype=np.float32)
    stored[order] = np.asarray(coordinates, dtype=np.float32).reshape(count, 3)
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(stored, deep=True))
    cells = vtk.vtkCellArray()
    start = 0
    for line in lines:
        cells.InsertNextCell(len(line))
        for index in range(start, start + len(line)):
            cells.InsertCellPoint(int(order[index]))
        start += len(line)
    data = vtk.vtkPolyData()
    data.SetPoints(points)
    data.SetLines(cells)
    for kind, ids in (extra_cells or {}).items():
        array = vtk.vtkCellArray()
        array.InsertNextCell(len(ids))
        for point_id in ids:
            array.InsertCellPoint(point_id)
        getattr(data, "Set" + kind)(array)
    return data, order


@unittest.skipIf(np is None, "numpy/VTK required (run with PythonSlicer)")
class DirectionColorTest(unittest.TestCase):
    def test_b2_fiber_dec_exact_values_for_axis_and_diagonal_lines(self) -> None:
        from shared.fiber_display import direction_colors

        lines = [
            [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)],  # left-right -> red
            [(0, 5, 0), (0, 3, 0), (0, 1, 0)],  # anterior-posterior, reversed -> green
            [(0, 0, 0), (0, 0, 2)],  # superior-inferior -> blue
            [(0, 0, 0), (1, 1, 1), (2, 2, 2)],  # diagonal -> round(255 / sqrt(3)) = 147
        ]
        data, order = _polydata(lines)
        colors = direction_colors(data)
        self.assertEqual(colors.dtype, np.uint8)
        self.assertEqual(colors.shape, (9 + 3, 3))
        expected = [(255, 0, 0)] * 4 + [(0, 255, 0)] * 3 + [(0, 0, 255)] * 2 + [(147, 147, 147)] * 3
        self.assertEqual([tuple(int(v) for v in colors[order[i]]) for i in range(12)], expected)

    def test_b2_fiber_dec_degenerate_cells_do_not_raise(self) -> None:
        from shared.fiber_display import direction_colors

        lines = [
            [(5, 5, 5)],  # one point
            [(1, 1, 1), (1, 1, 1), (1, 1, 1)],  # all repeated: no direction
            [(0, 0, 0), (0, 0, 0), (0, 4, 0)],  # repeated start point: falls back to the line chord
        ]
        data, order = _polydata(lines, extra_cells={"Verts": [0], "Polys": [1, 2, 3]})
        extra = data.GetPoints().InsertNextPoint(9, 9, 9)  # a point in no cell
        colors = direction_colors(data)
        self.assertEqual(colors.shape, (8, 3))
        self.assertFalse(np.isnan(colors.astype(float)).any())
        self.assertEqual(tuple(colors[order[0]]), (0, 0, 0))
        self.assertEqual({tuple(colors[order[i]]) for i in (1, 2, 3)}, {(0, 0, 0)})
        self.assertEqual(tuple(colors[order[4]]), (0, 255, 0))
        self.assertEqual(tuple(colors[order[6]]), (0, 255, 0))
        self.assertEqual(tuple(colors[extra]), (0, 0, 0))
        empty = vtk.vtkPolyData()
        self.assertEqual(direction_colors(empty).shape, (0, 3))

    def test_b2_fiber_dec_cost_for_two_million_points(self) -> None:
        import time

        from shared.fiber_display import direction_colors

        streamlines, points_per_line = 30_000, 66
        rng = np.random.default_rng(0)
        coordinates = np.cumsum(rng.normal(0, 1, (streamlines * points_per_line, 3)), axis=0).astype(np.float32)
        id_dtype = np.int64 if vtk.vtkIdTypeArray().GetDataTypeSize() == 8 else np.int32
        offsets = np.arange(0, streamlines * points_per_line + 1, points_per_line, dtype=id_dtype)
        cells = vtk.vtkCellArray()
        cells.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True),
                      numpy_to_vtkIdTypeArray(np.arange(len(coordinates), dtype=id_dtype), deep=True))
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(coordinates, deep=True))
        data = vtk.vtkPolyData()
        data.SetPoints(points)
        data.SetLines(cells)
        started = time.perf_counter()
        colors = direction_colors(data)
        elapsed = time.perf_counter() - started
        print(f"\n[b2 fiber DEC] {len(coordinates)} points: {elapsed:.3f} s")
        self.assertEqual(colors.shape, (len(coordinates), 3))
        self.assertLess(elapsed, 2.0)


class StreamlineKindTest(unittest.TestCase):
    def test_b2_fiber_classifier_by_product_and_key(self) -> None:
        from shared.fiber_display import streamline_kind

        cases = [
            ("tractography.csd_act", "tractogram", "/r/s_act.tck", "tractography"),
            ("tractography.csd_act", "tractogram_vtk", "/r/s_act.vtk", "tractography"),
            ("tractography.dti_det", "tractogram_trk", "/r/s_dti.trk", "tractography"),
            ("tractography.saf", "saf_vtk", "/r/s_saf.vtk", "tractography"),
            ("tractography.regions", "display_preview", "/r/s_regions.vtk", "tractography"),
            ("tractography.csd_act", "seed_mask", "/r/seeds.vtk", None),
            ("tractography.csd_act", "fod", "/r/fod.vtk", None),
            ("tractography.csd_act", "intermediate_tck", "/r/x.tck", None),
            ("tractography.csd_act", "tractogram_qc", "/r/x.nii.gz", None),
            ("bundles.wmql", "bundle_AF_left.trk", "/r/AF_left.trk", "bundle"),
            ("bundles.wma", "bundle_T_CST.vtp", "/r/T_CST.vtp", "bundle"),
            ("surface.ddsurfer", "white_lh", "/r/lh.white.vtk", None),
            ("connectome.csd_act", "matrix", "/r/m.csv", None),
        ]
        for product, key, path, expected in cases:
            with self.subTest(product=product, key=key):
                self.assertEqual(streamline_kind(product, key, path), expected)
                self.assertEqual(
                    streamline_kind({"product": product, "key": key, "path": path}), expected)
        self.assertIsNone(streamline_kind({"product": "bundles.wmql", "path": "/r/x.nii.gz"}))
        self.assertEqual(streamline_kind({"product": "bundles.wmql", "path": "/r/x.tck"}), "bundle")

    def test_b2_fiber_auto_role_uses_the_shared_classifier(self) -> None:
        from shared import auto_results

        bundle = {"product": "bundles.wmql", "key": "bundle_AF_left.trk", "path": "/r/AF_left.trk"}
        tract = {"product": "tractography.csd_act", "key": "tractogram", "path": "/r/s.tck"}
        self.assertEqual(auto_results._role(bundle), "bundle:af")
        self.assertEqual(auto_results._role(tract), "tractography")
        with patch("shared.auto_results.streamline_kind", return_value=None) as classifier:
            self.assertIsNone(auto_results._role(bundle))
            self.assertIsNone(auto_results._role(tract))
        self.assertTrue(classifier.called)


class _DisplayState:
    def __init__(self, name="display", visible=True):
        self.name = name
        self.visible = visible
        self.mock = Mock(name=name)
        self.mock.SetVisibility.side_effect = self._set_visibility
        self.mock.GetVisibility.side_effect = lambda: int(self.visible)

    def _set_visibility(self, value):
        self.visible = bool(value)

    def __getattr__(self, item):
        return getattr(self.mock, item)


class _Node:
    def __init__(self, identity, kind):
        self.identity, self.kind, self.attributes = identity, kind, {}
        self.display = _DisplayState("display-" + identity, visible=True)
        self.line_display = _DisplayState("line-" + identity, visible=True) if kind == "vtkMRMLFiberBundleNode" else None
        self.tube_display = _DisplayState("tube-" + identity, visible=False) if kind == "vtkMRMLFiberBundleNode" else None
        self.glyph_display = _DisplayState("glyph-" + identity, visible=False) if kind == "vtkMRMLFiberBundleNode" else None
        self.polydata = None
        self.default_display_calls = 0
        self.set_display_visibility_calls = []
        self.storage = []  # storage nodes (b2fix D1): the file a Save would write

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
        if self.kind == "vtkMRMLFiberBundleNode" and name in {"vtkMRMLFiberBundleNode", "vtkMRMLModelNode", "vtkMRMLDisplayableNode"}:
            return True
        return self.kind == name or (name == "vtkMRMLDisplayableNode" and self.kind != "vtkMRMLTableNode")

    def GetDisplayNode(self):
        return self.line_display if self.line_display is not None else self.display

    def GetLineDisplayNode(self):
        return self.line_display

    def GetTubeDisplayNode(self):
        return self.tube_display

    def GetGlyphDisplayNode(self):
        return self.glyph_display

    def GetNumberOfDisplayNodes(self):
        if self.kind == "vtkMRMLFiberBundleNode":
            return 3
        return 1

    def GetNthDisplayNode(self, index):
        if self.kind == "vtkMRMLFiberBundleNode":
            return (self.line_display, self.tube_display, self.glyph_display)[index]
        return self.display

    def GetDisplayVisibility(self):
        return int(self.GetDisplayNode().GetVisibility())

    def SetDisplayVisibility(self, visible):
        self.set_display_visibility_calls.append(bool(visible))
        for index in range(self.GetNumberOfDisplayNodes()):
            self.GetNthDisplayNode(index).SetVisibility(visible)

    def CreateDefaultDisplayNodes(self):
        self.default_display_calls += 1

    def SetAndObservePolyData(self, data):
        self.polydata = data

    def GetPolyData(self):
        return self.polydata

    def GetNumberOfStorageNodes(self):
        return len(self.storage)

    def GetNthStorageNode(self, index):
        return self.storage[index]


class _Storage:
    def __init__(self, path):
        self.path = path

    def GetFileName(self):
        return self.path


class _Scene:
    def __init__(self, *, fiber_bundle_registered=False):
        self.nodes, self.storage_nodes, self.next_id = [], [], 0
        self.fiber_bundle_registered = fiber_bundle_registered

    def AddNewNodeByClass(self, kind, name=""):
        if kind == "vtkMRMLFiberBundleNode" and not self.fiber_bundle_registered:
            return None
        self.next_id += 1
        node = _Node("node" + str(self.next_id), kind)
        self.nodes.append(node)
        return node

    def IsNodeClassRegistered(self, kind):
        return self.fiber_bundle_registered and kind == "vtkMRMLFiberBundleNode"

    def RemoveNode(self, node):
        if isinstance(node, _Storage):  # removing a storage node drops every node reference to it
            self.storage_nodes.remove(node)
            for owner in self.nodes:
                owner.storage = [item for item in owner.storage if item is not node]
            return
        self.nodes.remove(node)

    def GetNodeByID(self, identity):
        if str(identity).startswith("vtkMRMLColorTableNode"):
            return object()
        return next((node for node in self.nodes if node.GetID() == identity), None)


class _Logic(module_logic.SharedLogicMixin):
    def log(self, message):
        pass


@unittest.skipIf(np is None, "numpy/VTK/nibabel required (run with PythonSlicer)")
class NativeFiberLoaderTest(unittest.TestCase):
    """Fallback path when SlicerDMRI is NOT installed: core vtkMRMLModelNode + DEC RGB direction colouring."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.scene = _Scene(fiber_bundle_registered=False)
        self.load_fiber_bundle = Mock(side_effect=AssertionError("SlicerDMRI loadFiberBundle must not be called without SlicerDMRI"))
        self.factory = Mock(name="factoryManager")
        self.factory.registeredModuleNames.return_value = ["Models"]

        def load_model(path, kind, properties=None, **kwargs):
            if kind == "FiberBundleFile":
                raise AssertionError("FiberBundleFile loader must not be called without SlicerDMRI")
            reader = vtk.vtkXMLPolyDataReader() if path.endswith(".vtp") else vtk.vtkPolyDataReader()
            reader.SetFileName(path)
            reader.Update()
            node = self.scene.AddNewNodeByClass("vtkMRMLModelNode", "model")
            node.SetAndObservePolyData(reader.GetOutput())
            storage = _Storage(path)  # the reader gives the node a storage node that points at the file it read
            self.scene.storage_nodes.append(storage)
            node.storage.append(storage)
            return True, node

        self.fake = SimpleNamespace(
            mrmlScene=self.scene,
            app=SimpleNamespace(moduleManager=lambda: SimpleNamespace(factoryManager=lambda: self.factory)),
            vtkMRMLStorageNode=SimpleNamespace(CoordinateSystemRAS=0, CoordinateSystemLPS=1),
            vtkMRMLDisplayNode=SimpleNamespace(UseDirectMapping=4),
            util=SimpleNamespace(
                getNodesByClass=lambda kind: [node for node in self.scene.nodes
                                              if (node.kind == "vtkMRMLTableNode") == (kind == "vtkMRMLTableNode")],
                loadNodeFromFile=load_model,
                loadFiberBundle=self.load_fiber_bundle,
                setSliceViewerLayers=Mock(), resetSliceViews=Mock(), resetThreeDViews=Mock(),
            ),
        )
        self.addCleanup(patch.stopall)
        patch.object(module_logic, "slicer", self.fake).start()

        def register(slicer, node, run, subject, run_name, product, source):
            node.SetAttribute("SlicerDDSurfer.Run", run)
            node.SetAttribute("SlicerDDSurfer.Product", product)
            node.SetAttribute("SlicerDDSurfer.SourcePath", source)
            node.SetAttribute("SlicerDDSurfer.SourceStamp", source_stamp(source))

        patch("shared.result_scene.register_node", side_effect=register).start()
        self.logic = _Logic()

    def _write_vtp(self, path, direction):
        data, _ = _polydata([[tuple(np.asarray(direction) * step) for step in range(4)]], permute=False)
        fa = numpy_to_vtk(np.linspace(0, 1, 4).astype(np.float32), deep=True)
        fa.SetName("FA")
        data.GetPointData().SetScalars(fa)  # a file scalar must not take over the display
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(str(path))
        writer.SetInputData(data)
        writer.Write()

    def _manifest(self):
        import nibabel as nib

        tract = self.root / "outputs" / "s_act.tck"
        bundle = self.root / "outputs" / "AF_left.vtp"
        tract.parent.mkdir(parents=True)
        streamlines = [np.array([[0, index, 0], [10, index, 0], [20, index, 0]], dtype=np.float32) for index in range(5)]
        nib.streamlines.save(nib.streamlines.Tractogram(streamlines, affine_to_rasmm=np.eye(4)), str(tract))
        self._write_vtp(bundle, (0, 0, 1))
        path = self.root / "outputs" / "00_run_metadata" / "run_orchestration_manifest.json"
        path.parent.mkdir()
        path.write_text(json.dumps({
            "subject_id": "s", "run_name": "run", "output_root": str(self.root / "outputs"), "tasks": [
                {"name": "tractography.csd_act", "status": "completed", "outputs": {"tractogram": str(tract)}},
                {"name": "bundles.unit", "status": "completed", "outputs": {"bundle_AF_left.vtp": str(bundle)}},
            ]}), encoding="utf-8")
        return path, tract, bundle

    def _assert_native_fiber(self, node, *, width):
        self.assertEqual(node.GetClassName(), "vtkMRMLModelNode")
        self.assertEqual(node.GetAttribute("SlicerDDSurfer.DisplayKind"), "fiber_bundle")
        array = node.GetPolyData().GetPointData().GetArray(DEC)
        self.assertIsNotNone(array)
        self.assertEqual(array.GetNumberOfComponents(), 3)
        self.assertEqual(array.GetDataType(), vtk.VTK_UNSIGNED_CHAR)
        display = node.display
        display.SetActiveScalar.assert_called_with(DEC, vtk.vtkAssignAttribute.POINT_DATA)
        display.SetScalarRangeFlag.assert_called_with(4)
        display.SetScalarVisibility.assert_called_with(True)
        display.SetOpacity.assert_called_with(1.0)
        self.assertNotIn(0.5, [call.args[0] for call in display.SetOpacity.call_args_list])
        display.SetLineWidth.assert_called_with(width)
        display.SetVisibility2D.assert_called_with(True)
        display.SetSliceDisplayModeToIntersection.assert_called()
        display.SetColor.assert_called()

    def _nodes_for(self, path):
        return [node for node in self.scene.nodes if node.GetAttribute("SlicerDDSurfer.SourcePath") == str(path)]

    def _assert_no_slicerdmri(self):
        self.load_fiber_bundle.assert_not_called()
        self.factory.loadModules.assert_not_called()
        self.assertFalse([node for node in self.scene.nodes if node.kind == "vtkMRMLFiberBundleNode"])

    def test_b2_fiber_every_load_path_gives_the_same_native_display(self) -> None:
        manifest, tract, bundle = self._manifest()
        paths = {
            "automatic": lambda: self.logic.loadKeyResultsFromManifest(manifest, automatic=True),
            "eye": lambda: [self.logic.loadKeyResultsFromManifest(manifest, source_path=item) for item in (tract, bundle)],
            "load": lambda: self.logic.loadKeyResultsFromManifest(manifest, mode="reuse"),
            "replace": lambda: self.logic.loadKeyResultsFromManifest(manifest, mode="replace"),
            "append": lambda: self.logic.loadKeyResultsFromManifest(manifest, mode="append"),
        }
        for name, load in paths.items():
            with self.subTest(path=name):
                self.scene.nodes.clear()
                load()
                tract_nodes, bundle_nodes = self._nodes_for(tract), self._nodes_for(bundle)
                self.assertEqual(len(tract_nodes), 1)
                self.assertEqual(len(bundle_nodes), 1)
                self._assert_native_fiber(tract_nodes[0], width=1.0)
                self._assert_native_fiber(bundle_nodes[0], width=2.0)
                self.assertEqual(tract_nodes[0].GetAttribute("SlicerDDSurfer.StreamlineKind"), "tractography")
                self.assertEqual(bundle_nodes[0].GetAttribute("SlicerDDSurfer.StreamlineKind"), "bundle")
                colors = vtk_to_numpy(bundle_nodes[0].GetPolyData().GetPointData().GetArray(DEC))
                self.assertEqual({tuple(int(v) for v in row) for row in colors}, {(0, 0, 255)})
                self._assert_no_slicerdmri()

    def test_b2fix_display_D1_fiber_nodes_are_not_tied_to_the_result_file(self) -> None:
        """D1: the DEC array makes the poly data differ from the file, so a scene Save must never target the file."""
        manifest, tract, bundle = self._manifest()
        for automatic in (False, True):
            with self.subTest(automatic=automatic):
                self.scene.nodes.clear()
                self.scene.storage_nodes.clear()
                before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (tract, bundle)}
                self.logic.loadKeyResultsFromManifest(manifest, automatic=automatic, mode="replace")
                for source in (tract, bundle):
                    nodes = self._nodes_for(source)
                    self.assertEqual(len(nodes), 1)
                    self.assertIsNotNone(nodes[0].GetPolyData().GetPointData().GetArray(DEC))
                    self.assertEqual(nodes[0].GetNumberOfStorageNodes(), 0, "a Save would offer the result file")
                self.assertEqual(self.scene.storage_nodes, [], "an orphan storage node stays in the scene")
                self.assertEqual({path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (tract, bundle)}, before)

    def test_b2fix_display_D1_direct_loader_and_saf_path_also_drop_the_storage_node(self) -> None:
        source = self.root / "s_saf.vtk"
        data, _ = _polydata([[(0, 0, 0), (3, 0, 0)]], permute=False)
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(str(source))
        writer.SetInputData(data)
        writer.Write()
        self.assertTrue(self.logic._load_fiber_bundle_node(source, "s_SAF"))
        self.assertEqual(self.scene.nodes[-1].GetNumberOfStorageNodes(), 0)

    def test_b2fix_display_D1_a_plain_mesh_keeps_its_storage_node(self) -> None:
        mesh = self.root / "outputs" / "lh.white.vtk"
        mesh.parent.mkdir(parents=True)
        data, _ = _polydata([[(0, 0, 0), (3, 0, 0)]], permute=False)
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(str(mesh))
        writer.SetInputData(data)
        writer.Write()
        manifest = self.root / "outputs" / "manifest.json"
        manifest.write_text(json.dumps({"subject_id": "s", "tasks": [
            {"name": "surface.unit", "status": "completed", "outputs": {"white_lh_vtk": str(mesh)}}]}), encoding="utf-8")
        self.logic.loadKeyResultsFromManifest(manifest, source_path=mesh)
        self.assertEqual(self._nodes_for(mesh)[0].GetNumberOfStorageNodes(), 1)  # unchanged data: Save may keep the link

    def test_b2_fiber_saf_and_direct_loader_use_the_same_style(self) -> None:
        source = self.root / "s_saf.vtk"
        data, _ = _polydata([[(0, 0, 0), (3, 0, 0)]], permute=False)
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(str(source))
        writer.SetInputData(data)
        writer.Write()
        self.assertTrue(self.logic._load_fiber_bundle_node(source, "s_SAF"))
        node = self.scene.nodes[-1]
        self._assert_native_fiber(node, width=1.0)
        self.assertIsNone(node.GetAttribute("SlicerDDSurfer.DisplayFallback"))
        self._assert_no_slicerdmri()

    def test_b2_fiber_vtk_mesh_outside_tract_products_is_a_plain_model(self) -> None:
        mesh = self.root / "outputs" / "lh.white.vtk"
        mesh.parent.mkdir(parents=True)
        data, _ = _polydata([[(0, 0, 0), (3, 0, 0)]], permute=False)
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(str(mesh))
        writer.SetInputData(data)
        writer.Write()
        manifest = self.root / "outputs" / "manifest.json"
        manifest.write_text(json.dumps({"subject_id": "s", "tasks": [
            {"name": "surface.unit", "status": "completed", "outputs": {"white_lh_vtk": str(mesh)}}]}), encoding="utf-8")
        self.logic.loadKeyResultsFromManifest(manifest, source_path=mesh)
        node = self._nodes_for(mesh)[0]
        self.assertEqual(node.GetClassName(), "vtkMRMLModelNode")
        self.assertIsNone(node.GetAttribute("SlicerDDSurfer.DisplayKind"))
        self.assertIsNone(node.GetPolyData().GetPointData().GetArray(DEC))
        node.display.SetActiveScalar.assert_not_called()

    def test_b2_fiber_automatic_preset_does_not_override_the_loader_style(self) -> None:
        from shared.result_scene import automatic_display_preset

        for role in ("tractography", "bundle:af", "bundle:representative"):
            with self.subTest(role=role):
                node = _Node("n", "vtkMRMLModelNode")
                automatic_display_preset(self.fake, node, {"product": "tractography.csd_act", "display_role": role,
                                                           "path": "/r/x.tck"})
                node.display.SetOpacity.assert_not_called()
                node.display.SetColor.assert_not_called()
                self.assertEqual(node.GetAttribute("SlicerDDSurfer.AutomaticLoaded"), "true")


@unittest.skipIf(np is None, "numpy/VTK/nibabel required (run with PythonSlicer)")
class SlicerDMRIFiberLoaderTest(unittest.TestCase):
    """Dual-mode path when SlicerDMRI IS installed: real vtkMRMLFiberBundleNode styled as fiber lines."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.scene = _Scene(fiber_bundle_registered=True)
        self.factory = Mock(name="factoryManager")
        self.factory.registeredModuleNames.return_value = ["TractographyDisplay", "Models"]
        self.fail_fiber_bundle_io = False

        def load_node_from_file(path, kind, properties=None, **kwargs):
            if kind == "FiberBundleFile" and self.fail_fiber_bundle_io:
                return False, None
            reader = vtk.vtkXMLPolyDataReader() if path.endswith(".vtp") else vtk.vtkPolyDataReader()
            reader.SetFileName(path)
            reader.Update()
            node_class = "vtkMRMLFiberBundleNode" if kind == "FiberBundleFile" else "vtkMRMLModelNode"
            node = self.scene.AddNewNodeByClass(node_class, "loaded")
            node.SetAndObservePolyData(reader.GetOutput())
            storage = _Storage(path)
            self.scene.storage_nodes.append(storage)
            node.storage.append(storage)
            return True, node

        self.fake = SimpleNamespace(
            mrmlScene=self.scene,
            app=SimpleNamespace(moduleManager=lambda: SimpleNamespace(factoryManager=lambda: self.factory)),
            vtkMRMLStorageNode=SimpleNamespace(CoordinateSystemRAS=0, CoordinateSystemLPS=1),
            vtkMRMLDisplayNode=SimpleNamespace(UseDirectMapping=4),
            util=SimpleNamespace(
                getNodesByClass=lambda kind: [node for node in self.scene.nodes
                                              if (node.kind == "vtkMRMLTableNode") == (kind == "vtkMRMLTableNode")],
                loadNodeFromFile=load_node_from_file,
                setSliceViewerLayers=Mock(), resetSliceViews=Mock(), resetThreeDViews=Mock(),
                errorDisplay=Mock(),
            ),
        )
        self.addCleanup(patch.stopall)
        patch.object(module_logic, "slicer", self.fake).start()

        def register(slicer, node, run, subject, run_name, product, source):
            node.SetAttribute("SlicerDDSurfer.Run", run)
            node.SetAttribute("SlicerDDSurfer.Product", product)
            node.SetAttribute("SlicerDDSurfer.SourcePath", source)
            node.SetAttribute("SlicerDDSurfer.SourceStamp", source_stamp(source))

        patch("shared.result_scene.register_node", side_effect=register).start()
        self.logic = _Logic()

    def _write_vtp(self, path, direction):
        data, _ = _polydata([[tuple(np.asarray(direction) * step) for step in range(4)]], permute=False)
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(str(path))
        writer.SetInputData(data)
        writer.Write()

    def _manifest(self):
        import nibabel as nib

        tract = self.root / "outputs" / "s_act.tck"
        bundle = self.root / "outputs" / "AF_left.vtp"
        tract.parent.mkdir(parents=True)
        streamlines = [np.array([[0, index, 0], [10, index, 0], [20, index, 0]], dtype=np.float32) for index in range(5)]
        nib.streamlines.save(nib.streamlines.Tractogram(streamlines, affine_to_rasmm=np.eye(4)), str(tract))
        self._write_vtp(bundle, (0, 0, 1))
        path = self.root / "outputs" / "00_run_metadata" / "run_orchestration_manifest.json"
        path.parent.mkdir()
        path.write_text(json.dumps({
            "subject_id": "s", "run_name": "run", "output_root": str(self.root / "outputs"), "tasks": [
                {"name": "tractography.csd_act", "status": "completed", "outputs": {"tractogram": str(tract)}},
                {"name": "bundles.unit", "status": "completed", "outputs": {"bundle_AF_left.vtp": str(bundle)}},
            ]}), encoding="utf-8")
        return path, tract, bundle

    def _nodes_for(self, path):
        return [node for node in self.scene.nodes if node.GetAttribute("SlicerDDSurfer.SourcePath") == str(path)]

    def _assert_slicerdmri_fiber(self, node, *, width):
        self.assertEqual(node.GetClassName(), "vtkMRMLFiberBundleNode")
        self.assertEqual(node.GetAttribute("SlicerDDSurfer.DisplayKind"), "fiber_bundle")
        self.assertEqual(node.GetAttribute("SlicerDDSurfer.FiberColorMode"), "direction")
        # SlicerDMRI computes orientation colours on LineDisplayNode without mutating PolyData.
        self.assertIsNone(node.GetPolyData().GetPointData().GetArray(DEC))
        self.assertTrue(node.line_display.visible)
        self.assertFalse(node.tube_display.visible)
        self.assertFalse(node.glyph_display.visible)
        node.line_display.SetColorModeToPointFiberOrientation.assert_called()
        node.line_display.SetScalarVisibility.assert_called_with(True)
        node.line_display.SetOpacity.assert_called_with(1.0)
        node.line_display.SetLineWidth.assert_called_with(width)
        node.line_display.SetVisibility2D.assert_called_with(True)
        node.tube_display.SetVisibility.assert_called_with(False)
        node.glyph_display.SetVisibility.assert_called_with(False)

    def test_slicerdmri_present_every_load_path_creates_styled_fiber_bundle_nodes(self) -> None:
        manifest, tract, bundle = self._manifest()
        paths = {
            "automatic": lambda: self.logic.loadKeyResultsFromManifest(manifest, automatic=True),
            "eye": lambda: [self.logic.loadKeyResultsFromManifest(manifest, source_path=item) for item in (tract, bundle)],
            "load": lambda: self.logic.loadKeyResultsFromManifest(manifest, mode="reuse"),
            "replace": lambda: self.logic.loadKeyResultsFromManifest(manifest, mode="replace"),
            "append": lambda: self.logic.loadKeyResultsFromManifest(manifest, mode="append"),
        }
        for name, load in paths.items():
            with self.subTest(path=name):
                self.scene.nodes.clear()
                load()
                tract_nodes, bundle_nodes = self._nodes_for(tract), self._nodes_for(bundle)
                self.assertEqual(len(tract_nodes), 1)
                self.assertEqual(len(bundle_nodes), 1)
                self._assert_slicerdmri_fiber(tract_nodes[0], width=1.0)
                self._assert_slicerdmri_fiber(bundle_nodes[0], width=2.0)
                self.assertEqual(tract_nodes[0].GetAttribute("SlicerDDSurfer.StreamlineKind"), "tractography")
                self.assertEqual(bundle_nodes[0].GetAttribute("SlicerDDSurfer.StreamlineKind"), "bundle")

    def test_slicerdmri_eye_toggle_keeps_fiber_bundle_line_style_without_enabling_tubes_or_glyphs(self) -> None:
        manifest, tract, bundle = self._manifest()
        qt = SimpleNamespace(Qt=SimpleNamespace(UserRole=256))
        owner = SimpleNamespace(logic=self.logic, parent=None, _runOutputFiles=None, _latestResultManifestPath=manifest)

        class _Item:
            def __init__(self, path):
                self._data = {256: str(path), 257: "fiber_bundle", 258: str(manifest)}

            def data(self, _col, role):
                return self._data.get(role)

            def text(self, _col):
                return ""

        with patch.dict(sys.modules, {"qt": qt, "slicer": self.fake}):
            import importlib
            ui_run_history = importlib.import_module("shared.ui_run_history")
            for source, width in ((tract, 1.0), (bundle, 2.0)):
                with self.subTest(source=source.name):
                    item = _Item(source)
                    # First click: load and show in FiberBundle line style (never calling SetDisplayVisibility(True)).
                    ui_run_history._toggle_output(owner, item)
                    node = self._nodes_for(source)[0]
                    self._assert_slicerdmri_fiber(node, width=width)
                    self.assertEqual(node.set_display_visibility_calls, [])
                    self.assertTrue(ui_run_history._output_visible(self.fake, [node]))

                    # Even if user enabled TubeDisplayNode manually, hiding turns ALL displays off.
                    node.tube_display.SetVisibility(True)
                    ui_run_history._toggle_output(owner, item)
                    self.assertFalse(node.line_display.visible)
                    self.assertFalse(node.tube_display.visible)
                    self.assertFalse(node.glyph_display.visible)
                    self.assertFalse(ui_run_history._output_visible(self.fake, [node]))

                    # Showing again enables LineDisplayNode only; Tube and Glyph stay off.
                    ui_run_history._toggle_output(owner, item)
                    self.assertTrue(node.line_display.visible)
                    self.assertFalse(node.tube_display.visible)
                    self.assertFalse(node.glyph_display.visible)
                    self.assertTrue(ui_run_history._output_visible(self.fake, [node]))
        self.fake.util.errorDisplay.assert_not_called()

    def test_slicerdmri_plain_vtk_surface_still_loads_as_plain_model(self) -> None:
        mesh = self.root / "outputs" / "lh.white.vtk"
        mesh.parent.mkdir(parents=True)
        data, _ = _polydata([[(0, 0, 0), (3, 0, 0)]], permute=False)
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(str(mesh))
        writer.SetInputData(data)
        writer.Write()
        manifest = self.root / "outputs" / "manifest.json"
        manifest.write_text(json.dumps({"subject_id": "s", "tasks": [
            {"name": "surface.unit", "status": "completed", "outputs": {"white_lh_vtk": str(mesh)}}]}), encoding="utf-8")
        self.logic.loadKeyResultsFromManifest(manifest, source_path=mesh)
        node = self._nodes_for(mesh)[0]
        self.assertEqual(node.GetClassName(), "vtkMRMLModelNode")
        self.assertEqual(node.GetNumberOfStorageNodes(), 1)
        self.assertIsNone(node.GetAttribute("SlicerDDSurfer.DisplayKind"))

    def test_slicerdmri_loader_failure_falls_back_to_native_dec_fiber_bundle(self) -> None:
        manifest, _tract, bundle = self._manifest()
        self.fail_fiber_bundle_io = True
        self.logic.loadKeyResultsFromManifest(manifest, source_path=bundle)
        node = self._nodes_for(bundle)[0]
        self.assertEqual(node.GetClassName(), "vtkMRMLModelNode")
        self.assertEqual(node.GetAttribute("SlicerDDSurfer.DisplayKind"), "fiber_bundle")
        self.assertIsNotNone(node.GetPolyData().GetPointData().GetArray(DEC))
        self.assertEqual(node.GetNumberOfStorageNodes(), 0)

    def test_slicerdmri_lazy_module_registration_enables_fiber_bundle_node(self) -> None:
        manifest, tract, _bundle = self._manifest()
        self.scene.fiber_bundle_registered = False

        def register_on_load(modules):
            if "TractographyDisplay" in modules:
                self.scene.fiber_bundle_registered = True

        self.factory.loadModules.side_effect = register_on_load
        self.logic.loadKeyResultsFromManifest(manifest, source_path=tract)
        self.factory.loadModules.assert_called_with(["TractographyDisplay"])
        node = self._nodes_for(tract)[0]
        self._assert_slicerdmri_fiber(node, width=1.0)


class _Pixmap:
    def __init__(self, *args):
        self.svg = b""

    def loadFromData(self, data, *_args):
        self.svg = bytes(data.data) if hasattr(data, "data") else bytes(data)
        return True


class _Icon:
    def __init__(self, pixmap):
        self.svg = pixmap.svg.decode("utf-8")


class _Item:
    def __init__(self, parent, labels):
        self.labels, self.children, self._data, self.tooltips, self.icon = list(labels), [], {}, {}, None
        if parent is not None:
            parent.children.append(self)

    def text(self, col):
        return self.labels[col] if col < len(self.labels) else ""

    def setText(self, col, text):
        while len(self.labels) <= col:
            self.labels.append("")
        self.labels[col] = text

    def setData(self, col, role, value):
        self._data[(col, role)] = value

    def data(self, col, role):
        return self._data.get((col, role))

    def setToolTip(self, col, text):
        self.tooltips[col] = text

    def setIcon(self, col, icon):
        self.icon = icon

    def childCount(self):
        return len(self.children)

    def child(self, index):
        return self.children[index]


class _Tree:
    def __init__(self, items):
        self.items = items

    def topLevelItemCount(self):
        return len(self.items)

    def topLevelItem(self, index):
        return self.items[index]


def _qt():
    return SimpleNamespace(Qt=SimpleNamespace(UserRole=256), QPixmap=_Pixmap, QIcon=_Icon,
                           QByteArray=lambda data: SimpleNamespace(data=data), QTreeWidgetItem=_Item)


class FiberEyeIconTest(unittest.TestCase):
    def setUp(self):
        from shared import ui_run_history

        self.ui = ui_run_history
        patcher = patch.dict(ui_run_history._EYE_ICONS, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_b2_fiber_eye_icon_cache_key_is_kind_visible_dark(self) -> None:
        qt = _qt()
        shown = self.ui._eye_icon(qt, True, kind="fiber_bundle", dark=False)
        self.assertIs(self.ui._eye_icon(qt, True, kind="fiber_bundle", dark=False), shown)
        for kind in ("fiber_bundle", None):
            for visible in (True, False):
                for dark in (True, False):
                    self.ui._eye_icon(qt, visible, kind=kind, dark=dark)
        self.assertEqual(set(self.ui._EYE_ICONS), {(kind, visible, dark) for kind in ("fiber_bundle", "plain")
                                                    for visible in (True, False) for dark in (True, False)})
        hidden = self.ui._eye_icon(qt, False, kind="fiber_bundle", dark=False)
        plain = self.ui._eye_icon(qt, True, kind=None, dark=False)
        dark = self.ui._eye_icon(qt, True, kind="fiber_bundle", dark=True)
        # Three coloured streamlines in place of the pupil; hidden = grey curves and the slash.
        self.assertEqual(shown.svg.count("<path"), 4)
        self.assertNotIn("<circle", shown.svg)
        self.assertEqual(len({color for color in self.ui._FIBER_EYE_COLORS[False]["curves"]
                              if color in shown.svg}), 3)
        self.assertNotIn("M3 3l18 18", shown.svg)
        self.assertIn("M3 3l18 18", hidden.svg)
        self.assertFalse(any(color in hidden.svg for color in self.ui._FIBER_EYE_COLORS[False]["curves"]))
        self.assertIn("<circle", plain.svg)
        self.assertNotEqual(shown.svg, dark.svg)
        self.assertNotEqual(plain.svg, self.ui._eye_icon(qt, True, kind=None, dark=True).svg)

    def _tree(self):
        qt = _qt()
        group = _Item(None, ["Tractography (2)"])
        fiber = _Item(group, ["s_act.tck", "○"])
        fiber.setData(0, 256, "/r/s_act.tck")
        fiber.setData(0, 257, "fiber_bundle")
        fiber.setData(0, 258, "/r/manifest.json")
        surface = _Item(group, ["lh.white.stl", "○"])
        surface.setData(0, 256, "/r/lh.white.stl")
        surface.setData(0, 258, "/r/manifest.json")
        return qt, _Tree([group]), fiber, surface

    def _refresh(self, owner, qt, *, visible, dark=None):
        slicer = types.ModuleType("slicer")
        with patch.dict(sys.modules, {"qt": qt, "slicer": slicer}), \
                patch("shared.result_scene.result_nodes", return_value=[]), \
                patch.object(self.ui, "_output_visible", return_value=visible):
            if dark is None:
                self.ui._refresh_output_visibility(owner)
            else:
                self.ui.refresh_results_window(owner, dark=dark)

    def test_b2_fiber_eye_tooltips_en_zh(self) -> None:
        qt, tree, fiber, surface = self._tree()
        owner = SimpleNamespace(_runOutputFiles=tree, _onboardingLanguage="en", _lastAppliedDarkMode=False)
        expected = {("en", True): ("Hide fiber bundle", "Hide"), ("en", False): ("Show as fiber bundle", "Show"),
                    ("zh", True): ("隐藏纤维束", "隐藏"), ("zh", False): ("显示为纤维束", "显示")}
        for (language, visible), (fiber_tip, plain_tip) in expected.items():
            with self.subTest(language=language, visible=visible):
                owner._onboardingLanguage = language
                self._refresh(owner, qt, visible=visible)
                self.assertEqual(fiber.tooltips[1], fiber_tip)
                self.assertEqual(surface.tooltips[1], plain_tip)
                self.assertNotIn("<circle", fiber.icon.svg)
                self.assertIn("<circle", surface.icon.svg)

    def test_b2_fiber_eye_redraws_when_the_theme_changes(self) -> None:
        qt, tree, fiber, surface = self._tree()
        owner = SimpleNamespace(_runOutputFiles=tree, _onboardingLanguage="en", _lastAppliedDarkMode=False)
        self._refresh(owner, qt, visible=True)
        light_fiber, light_plain = fiber.icon, surface.icon
        owner._lastAppliedDarkMode = True
        self._refresh(owner, qt, visible=True, dark=True)  # theme hook, no pop-out button in this owner
        self.assertIsNot(fiber.icon, light_fiber)
        self.assertIsNot(surface.icon, light_plain)
        self.assertIs(fiber.icon, self.ui._EYE_ICONS[("fiber_bundle", True, True)])

    def test_b2_fiber_tree_row_kind(self) -> None:
        qt = SimpleNamespace(Qt=SimpleNamespace(UserRole=256), QTreeWidgetItem=_Item)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            files = {name: root / name for name in ("s_act.tck", "AF.trk", "lh.white.stl", "fa.nii.gz", "seeds.vtk")}
            for path in files.values():
                path.write_text("x", encoding="utf-8")
            manifest = {"subject_id": "s", "output_root": str(root), "tasks": [
                {"name": "tractography.csd_act", "status": "completed",
                 "outputs": {"tractogram": str(files["s_act.tck"]), "seed_mask": str(files["seeds.vtk"])}},
                {"name": "bundles.unit", "status": "completed", "outputs": {"bundle_AF.trk": str(files["AF.trk"])}},
                {"name": "surface.ddsurfer", "status": "completed", "outputs": {"white_lh": str(files["lh.white.stl"])}},
                {"name": "microstructure.dti", "status": "completed", "outputs": {"fa": str(files["fa.nii.gz"])}},
            ]}
            parent = _Item(None, ["run"])
            with patch.dict(sys.modules, {"qt": qt}):
                self.ui._populate_manifest_output_items(SimpleNamespace(_onboardingLanguage="en"), parent, manifest)
        # Results subgroups: every family of this manifest has subgroups (family > subgroup > file).
        kinds = {leaf.text(0): leaf.data(0, 256 + self.ui.ROLE_KIND)
                 for family in parent.children for subgroup in family.children for leaf in subgroup.children}
        self.assertEqual(kinds, {"s_act.tck": "fiber_bundle", "AF.trk": "fiber_bundle", "lh.white.stl": None,
                                 "fa.nii.gz": None, "seeds.vtk": None})


if __name__ == "__main__":
    unittest.main()
