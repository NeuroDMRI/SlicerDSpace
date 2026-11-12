"""Exercise GUI automatic loading using actual, previously generated run records."""

import json
import os
from pathlib import Path
import time
import traceback

import qt
import slicer

ROOT = Path(os.environ["SLICERDSPACE_SMOKE_ROOT"])
SOURCE = Path(os.environ["SLICERDSPACE_AUTO_RESULT_MANIFESTS"])


def start():
    records = []
    try:
        from shared.auto_results import automatic_result_candidates
        from shared.result_scene import owned_nodes
        slicer.util.selectModule("SlicerDSpace")
        widget = slicer.modules.slicerdspace.widgetRepresentation().self()
        widget._runProgressStartedAt = None
        for value in json.loads(SOURCE.read_text(encoding="utf-8")):
            path = Path(value).resolve()
            manifest = json.loads(path.read_text(encoding="utf-8"))
            expected, skipped, deferred = automatic_result_candidates(manifest)
            for task in manifest.get("tasks", []):
                if task.get("status") not in {"completed", "reused"}:
                    continue
                outputs = task.get("outputs") or {}
                if task.get("name", "").startswith("connectome.") and any(
                    key.startswith("matrix") and Path(str(value)).is_file() for key, value in outputs.items()):
                    assert any(candidate["product"] == task["name"] for candidate in expected), "Missing automatic connectome table"
            if not expected:
                records.append({"manifest": str(path), "status": "NO_REPRESENTATIVE_OUTPUT",
                                "skipped": skipped, "deferred": deferred})
                continue
            slicer.mrmlScene.Clear(0)
            slicer.app.processEvents()
            started = time.monotonic()
            loaded = widget.logic.loadKeyResultsFromManifest(path, automatic=True)
            nodes = owned_nodes(slicer, str(path))
            actual = {node.GetAttribute("SlicerDDSurfer.SourcePath") for node in nodes}
            required = {candidate["path"] for candidate in expected}
            if actual != required:
                raise AssertionError({"manifest": str(path), "missing": sorted(required - actual),
                                      "unexpected": sorted(actual - required), "loaded": loaded})
            before = {node.GetID() for node in nodes}
            widget.logic.loadKeyResultsFromManifest(path, automatic=True)
            assert before == {node.GetID() for node in owned_nodes(slicer, str(path))}, "Duplicate nodes on repeat poll"
            slicer.app.processEvents()
            slicer.app.processEvents()
            scalar_ids = {node.GetID() for node in nodes if node.IsA("vtkMRMLScalarVolumeNode")
                          and not node.IsA("vtkMRMLLabelMapVolumeNode")}
            if scalar_ids:
                composites = slicer.util.getNodesByClass("vtkMRMLSliceCompositeNode")
                assert all(node.GetBackgroundVolumeID() in scalar_ids for node in composites), "Loaded scalar not selected in all slice viewers"
            for node in nodes:
                if node.IsA("vtkMRMLModelNode"):
                    data = node.GetPolyData()
                    assert data is not None and data.GetNumberOfPoints() > 0
                    source = Path(node.GetAttribute("SlicerDDSurfer.SourcePath"))
                    if source.suffix.lower() in {".vtk", ".vtp"} and node.GetAttribute("SlicerDDSurfer.DisplayFallback"):
                        import numpy as np
                        import vtk
                        from vtk.util.numpy_support import vtk_to_numpy
                        reader = vtk.vtkXMLPolyDataReader() if source.suffix.lower() == ".vtp" else vtk.vtkPolyDataReader()
                        reader.SetFileName(str(source))
                        reader.Update()
                        assert np.allclose(vtk_to_numpy(data.GetPoints().GetData()),
                                           vtk_to_numpy(reader.GetOutput().GetPoints().GetData())), "RAS geometry changed in fiber model fallback"
                if node.IsA("vtkMRMLScalarVolumeNode"):
                    assert node.GetImageData() is not None
                preview = node.GetAttribute("SlicerDDSurfer.DisplayPreviewPath")
                if preview:
                    assert Path(preview).is_file(), "Missing persisted VTK display preview"
                    with Path(preview).open("rb") as stream:
                        stream.readline()
                        assert b"SPACE=RAS" in stream.readline(), "Legacy VTK preview must declare RAS in its header"
                    assert Path(node.GetAttribute("SlicerDDSurfer.DisplayPreviewProvenancePath")).is_file()
                    if os.environ.get("SLICERDSPACE_REQUIRE_NATIVE_DMRI") == "1":
                        assert node.IsA("vtkMRMLFiberBundleNode"), "SlicerDMRI preview fell back to generic Model"
                    import numpy as np
                    import vtk
                    from vtk.util.numpy_support import vtk_to_numpy
                    reader = vtk.vtkPolyDataReader()
                    reader.SetFileName(preview)
                    reader.Update()
                    assert np.allclose(vtk_to_numpy(node.GetPolyData().GetPoints().GetData()),
                                       vtk_to_numpy(reader.GetOutput().GetPoints().GetData())), "Preview RAS coordinates changed in native/fallback load"
            record = {"manifest": str(path), "status": "PASS", "node_count": len(nodes),
                      "products": sorted({candidate["product"] for candidate in expected}),
                      "seconds": time.monotonic() - started,
                      "nodes": [{"name": node.GetName(), "class": node.GetClassName(),
                                 "path": node.GetAttribute("SlicerDDSurfer.SourcePath"),
                                 "display_count": node.GetAttribute("SlicerDDSurfer.DisplayStreamlineCount")}
                                for node in nodes], "deferred": deferred}
            records.append(record)
            (ROOT / "automatic_results.json").write_text(json.dumps(records, indent=2) + "\n")
            slicer.util.mainWindow().grab().save(str(ROOT / f"automatic_results_{len(records)}.png"))
        assert any(record["status"] == "PASS" for record in records), "No real outputs loaded"
        print("AUTOMATIC_RESULTS=PASS", flush=True)
        slicer.app.exit(0)
    except Exception as exc:
        traceback.print_exc()
        records.append({"status": "FAIL", "error": str(exc)})
        (ROOT / "automatic_results.json").write_text(json.dumps(records, indent=2) + "\n")
        slicer.app.exit(2)


qt.QTimer.singleShot(5000, start)
