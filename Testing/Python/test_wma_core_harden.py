"""Headless tests for the Slicer-core WMA transform adapter.

These tests exercise real VTK VTP serialization and core-like MRML interfaces.
They are not a substitute for running the adapter in a real 3D Slicer build.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

try:
    import vtk
except ImportError:
    vtk = None

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('wma_harden', ROOT / 'scripts/run_wma_harden_transform.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@unittest.skipUnless(vtk is not None, 'VTK is required for geometry checks')
class HardenTransformTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.source_dir = root / 'source'
        self.source_dir.mkdir()
        self.output = root / 'output'
        self.transform_path = root / 'sample.tfm'
        self.transform_path.write_text('mock transform')
        self.removed = []
        self.current_offset = 2.0
        owner = self

        class Node:
            def __init__(self, poly):
                self.poly = poly
                self.transform_id = None

            def GetPolyData(self):
                return self.poly

            def SetAndObserveTransformNodeID(self, value):
                self.transform_id = value

        class Transform:
            def __init__(self):
                self.inverted = False

            def GetID(self):
                return 'mock-transform'

            def Inverse(self):
                self.inverted = True

        class Storage:
            def SetFileName(self, name):
                self.name = name

            def WriteData(self, node):
                writer = vtk.vtkXMLPolyDataWriter() if self.name.endswith('.vtp') else vtk.vtkPolyDataWriter()
                writer.SetFileName(self.name)
                writer.SetInputData(node.GetPolyData())
                return writer.Write()

        class Scene:
            def AddNewNodeByClass(self, class_name):
                if class_name != 'vtkMRMLModelStorageNode':
                    raise AssertionError(f'Unexpected SlicerDMRI-dependent node: {class_name}')
                return Storage()

            def RemoveNode(self, node):
                owner.removed.append(node)

        class Logic:
            def hardenTransform(self, model):
                assert model.transform_id == 'mock-transform'
                offset = -owner.current_offset if owner.transform.inverted else owner.current_offset
                t = vtk.vtkTransform()
                t.Translate(offset, 0, 0)
                f = vtk.vtkTransformPolyDataFilter()
                f.SetInputData(model.GetPolyData())
                f.SetTransform(t)
                f.Update()
                model.poly = f.GetOutput()

        def load_model(path):
            reader = vtk.vtkXMLPolyDataReader() if path.endswith('.vtp') else vtk.vtkPolyDataReader()
            reader.SetFileName(path)
            reader.Update()
            return Node(reader.GetOutput())

        def load_transform(_path):
            self.transform = Transform()
            return self.transform

        self.slicer = SimpleNamespace(
            util=SimpleNamespace(loadModel=load_model, loadTransform=load_transform),
            mrmlScene=Scene(), vtkSlicerTransformLogic=Logic,
        )

    def add_poly(self, name='cluster.vtp', *, empty=False):
        path = self.source_dir / name
        poly = vtk.vtkPolyData()
        if not empty:
            pts = vtk.vtkPoints()
            pts.InsertNextPoint(1, 3, 5)
            pts.InsertNextPoint(2, 4, 6)
            poly.SetPoints(pts)
            lines = vtk.vtkCellArray()
            lines.InsertNextCell(2)
            lines.InsertCellPoint(0)
            lines.InsertCellPoint(1)
            poly.SetLines(lines)
        writer = vtk.vtkXMLPolyDataWriter() if name.endswith('.vtp') else vtk.vtkPolyDataWriter()
        writer.SetInputData(poly)
        writer.SetFileName(str(path))
        self.assertTrue(writer.Write())
        return path

    def test_forward_and_inverse_do_not_use_fiberbundle_reader(self):
        src = self.add_poly()
        self.assertEqual(module.main([str(src), str(self.transform_path), '0', str(self.output)],
                                     slicer_module=self.slicer), 0)
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(str(self.output / 'cluster.vtp'))
        reader.Update()
        self.assertEqual(tuple(reader.GetOutput().GetPoint(0)), (3.0, 3.0, 5.0))
        self.assertEqual(reader.GetOutput().GetNumberOfLines(), 1)
        self.assertEqual(len(self.removed), 3)
        self.output.joinpath('cluster.vtp').unlink()
        self.assertEqual(module.main([str(src), str(self.transform_path), '1', str(self.output)],
                                     slicer_module=self.slicer), 0)
        reader.Modified()
        reader.Update()
        self.assertEqual(tuple(reader.GetOutput().GetPoint(0)), (-1.0, 3.0, 5.0))

    def test_vtk_and_vtp_directory_and_empty_clusters(self):
        src = self.add_poly('first.vtk')
        empty = self.add_poly('empty.vtp', empty=True)
        module.main([str(self.source_dir), str(self.transform_path), '0', str(self.output)],
                    slicer_module=self.slicer)
        self.assertTrue((self.output / 'first.vtk').is_file())
        self.assertEqual((self.output / empty.name).read_bytes(), empty.read_bytes())
        self.assertEqual(len(self.removed), 4)  # empty model + nonempty storage/model/transform

    def test_missing_input_fails_explicitly(self):
        with self.assertRaises(FileNotFoundError):
            module.main([str(self.source_dir / 'not-here'), str(self.transform_path), '0', str(self.output)],
                        slicer_module=self.slicer)


if __name__ == '__main__':
    unittest.main()
