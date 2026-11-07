"""VTK reference-fixture tests for the non-equivalent native WMA statistics."""
from __future__ import annotations

import math
import sys
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import vtk
except ImportError:
    vtk = None


@unittest.skipUnless(vtk, 'VTK needed')
class WMAStatisticsTest(unittest.TestCase):
    def sample(self, *, tensor=False):
        poly = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        for p in ((0.,0.,0.), (3.,4.,0.), (3.,4.,12.)):
            points.InsertNextPoint(*p)
        poly.SetPoints(points)
        lines = vtk.vtkCellArray()
        lines.InsertNextCell(3)
        for idx in range(3):
            lines.InsertCellPoint(idx)
        poly.SetLines(lines)
        arr = vtk.vtkDoubleArray()
        arr.SetName('FA'); arr.SetNumberOfComponents(1)
        for value in (0.2, 0.4, 1.5):
            arr.InsertNextValue(value)
        poly.GetPointData().AddArray(arr)
        if tensor:
            tens = vtk.vtkDoubleArray()
            tens.SetName('Tensor'); tens.SetNumberOfComponents(9)
            tens.InsertNextTuple((1.,0.,0.,0.,1.,0.,0.,0.,1.))
            tens.InsertNextTuple((1.,0.,0.,0.,1.,0.,0.,0.,1.))
            tens.InsertNextTuple((1.,0.,0.,0.,1.,0.,0.,0.,1.))
            poly.GetPointData().AddArray(tens)
        return poly

    def test_tract_length_and_fa_clamping(self):
        from tractography.wma_native_measures import measure_polydata
        row = measure_polydata(self.sample())
        self.assertEqual(row['Num_Fibers'], 1)
        self.assertEqual(row['Num_Points'], 3)
        self.assertAlmostEqual(row['Mean_Length'], 17.)
        self.assertAlmostEqual(row['FA'], 0.3)
        self.assertEqual(row['Num_Clamp_Excluded'], 1)

    def test_tensor_data_must_not_be_silently_ignored(self):
        from tractography.wma_native_measures import measure_polydata, WMAUnsupportedMeasurement
        with self.assertRaisesRegex(WMAUnsupportedMeasurement, 'Tensor array'):
            measure_polydata(self.sample(tensor=True))

    def test_csv_written_and_existing_file_not_overwritten(self):
        from tractography.wma_native_measures import measure_directory
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); directory=root/'tracts'; directory.mkdir()
            writer=vtk.vtkXMLPolyDataWriter()
            writer.SetFileName(str(directory/'AF_left.vtp'))
            writer.SetInputData(self.sample())
            self.assertTrue(writer.Write())
            csv=root/'measurements.csv'
            record=measure_directory(directory,csv)
            self.assertEqual(record['tracts'],1)
            self.assertIn('AF_left,3,1,17.0,1,0.30000000000000004', csv.read_text())
            with self.assertRaises(FileExistsError):
                measure_directory(directory,csv)

if __name__ == '__main__':
    unittest.main()
