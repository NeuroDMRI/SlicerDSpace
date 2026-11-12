"""Exercise real scene ownership and asynchronous cancellation without downloads."""

import json
import os
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import qt
import slicer

TEMP = tempfile.TemporaryDirectory()
ROOT = Path(TEMP.name)
STATE = {}


def fail():
    traceback.print_exc()
    slicer.app.exit(2)


def run():
    try:
        from shared.module_logic import SharedLogicMixin
        from shared.result_scene import owned_nodes
        from shared.ui_setup_jobs import start_setup_job

        slicer.util.selectModule("SlicerDSpace")
        widget = slicer.modules.slicerdspace.widgetRepresentation().self()
        for section in (widget.ui.configurationCollapsibleButton,
                        widget.ui.targetsCollapsibleButton,
                        widget.ui.resultsCollapsibleButton):
            assert isinstance(section, qt.QGroupBox)
        widget._latestResultManifestPath = ROOT / "previous-run.json"
        widget._reset_result_scene_actions()
        assert widget._latestResultManifestPath is None
        assert widget._runOutputFiles.topLevelItemCount == 0
        print("GUI_RESULT_ACTION_RESET=True", flush=True)
        STATE["widget"] = widget
        with slicer.util.NodeModify(widget._parameterNode):
            widget._parameterNode.targetSurface = False
            widget._parameterNode.targetSurfacePostprocess = True
        assert widget._product_target_flags().surface_postprocess
        assert "surface_postprocess" in widget._selected_product_targets()
        widget._save_common_user_settings()
        from shared.user_config import load_user_config
        assert "surface_postprocess" in load_user_config()["run_defaults"]["targets"]
        widget._parameterNode.targetSurfacePostprocess = False
        print("GUI_POSTPROCESS_SELECTION_AND_PERSISTENCE=True", flush=True)
        node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "synthetic")
        slicer.util.updateVolumeFromArray(node, np.ones((4, 5, 6), dtype=np.float32))
        image = ROOT / "fa.nii.gz"
        assert slicer.util.saveNode(node, str(image))
        slicer.mrmlScene.RemoveNode(node)
        table = ROOT / "metrics.csv"
        table.write_text("region,mean_fa\nleft,0.5\nright,0.6\n")
        manifest = ROOT / "run.json"
        manifest.write_text(json.dumps({"subject_id": "gui-test", "run_name": "Test1", "tasks": [
            {"name": "microstructure.dti", "status": "completed", "outputs": {"fa": str(image), "regional_statistics": str(table)}},
            {"name": "microstructure.noddi", "status": "failed", "outputs": {"ndi": str(image)}}]}))
        from shared.ui_run_history import _show_output_files, _toggle_output

        widget._latestResultManifestPath = manifest
        _show_output_files(widget, json.loads(manifest.read_text()))
        assert not owned_nodes(slicer, str(manifest))
        group = widget._runOutputFiles.topLevelItem(0)
        item = next(group.child(i) for i in range(group.childCount()) if group.child(i).text(0) == image.name)
        _toggle_output(widget, item)
        nodes = owned_nodes(slicer, str(manifest))
        assert len(nodes) == 1
        identity = nodes[0].GetID()
        red = slicer.app.layoutManager().sliceWidget("Red").mrmlSliceCompositeNode()
        assert red.GetBackgroundVolumeID() == identity
        _toggle_output(widget, item)
        assert red.GetBackgroundVolumeID() is None
        _toggle_output(widget, item)
        assert red.GetBackgroundVolumeID() == identity
        assert widget.logic.clearResultsFromManifest(manifest) == 1
        print("GUI_LAZY_RESULT_VISIBILITY=True", flush=True)
        logic = SharedLogicMixin()
        logic.log = print
        first = logic.loadKeyResultsFromManifest(manifest)
        assert len(first["loaded"]) == 2 and len(first["skipped"]) == 1
        assert len(logic.loadKeyResultsFromManifest(manifest)["reused"]) == 2
        assert len(owned_nodes(slicer, str(manifest))) == 2
        assert sum(node.IsA("vtkMRMLTableNode") for node in owned_nodes(slicer, str(manifest))) == 1
        logic.loadKeyResultsFromManifest(manifest, mode="append")
        assert len(owned_nodes(slicer, str(manifest))) == 4
        logic.loadKeyResultsFromManifest(manifest, mode="replace")
        assert len(owned_nodes(slicer, str(manifest))) == 2
        # Ownership persists through scene serialization.
        scene_file = ROOT / "scene.mrb"
        assert slicer.util.saveScene(str(scene_file))
        slicer.mrmlScene.Clear(0)
        assert slicer.util.loadScene(str(scene_file))
        assert len(logic.loadKeyResultsFromManifest(manifest)["reused"]) == 2
        relocated = ROOT / "relocated"
        relocated.mkdir()
        import shutil
        shutil.copy2(scene_file, relocated / "portable.mrb")
        image.rename(ROOT / "fa.original-moved")
        table.rename(ROOT / "metrics.original-moved")
        slicer.mrmlScene.Clear(0)
        assert slicer.util.loadScene(str(relocated / "portable.mrb"))
        restored = owned_nodes(slicer, str(manifest))
        assert len(restored) == 2
        volume = next(node for node in restored if node.IsA("vtkMRMLScalarVolumeNode"))
        np.testing.assert_array_equal(slicer.util.arrayFromVolume(volume), np.ones((4, 5, 6)))
        assert next(node for node in restored if node.IsA("vtkMRMLTableNode")).GetTable().GetNumberOfRows() == 2
        assert logic.clearResultsFromManifest(manifest) == 2
        print("GUI_SCENE_OWNERSHIP=True", flush=True)
        STATE["beats"] = 0
        timer = qt.QTimer()
        timer.connect("timeout()", lambda: STATE.update(beats=STATE["beats"] + 1))
        timer.start(50)
        STATE["timer"] = timer
        owner = SimpleNamespace(parent=slicer.util.mainWindow(), _latestEnvironmentReportDir=ROOT,
                                _pipelineActive=False, appendLog=print)
        owner._setPipelineActive = lambda active: setattr(owner, "_pipelineActive", active)
        STATE["owner"] = owner
        program = widget._slicer_launcher_program()
        command = (program, ["--launch", "PythonSlicer", "-u", "-c", "import time; print('working',flush=True); time.sleep(30)"], {})

        def done(status, details, log_path):
            try:
                assert status == "cancelled", status
                assert STATE["beats"] >= 5, STATE["beats"]
                assert "cancelled" in log_path.read_text()
                assert owner._setupJob is None and not owner._pipelineActive
                STATE["timer"].stop()
                print("GUI_SETUP_CANCEL_AND_RESPONSIVENESS=True", flush=True)
                assert start_setup_job(owner, "Synthetic teardown", [command],
                                       lambda *args: (_ for _ in ()).throw(AssertionError("Teardown called UI callback")))
                loop = qt.QEventLoop()
                qt.QTimer.singleShot(500, loop.quit)
                loop.exec_()
                job = owner._setupJob
                job.shutdown()
                assert job.process.state() == qt.QProcess.NotRunning
                assert owner._setupJob is None and not owner._pipelineActive
                print("GUI_SETUP_TEARDOWN=True", flush=True)
                slicer.util.mainWindow().resize(1500, 1050)
                widget.ui.environmentCollapsibleButton.collapsed = True
                widget.ui.configurationCollapsibleButton.collapsed = False
                widget.ui.advancedRunCollapsibleButton.collapsed = True
                slicer.app.processEvents()
                capture_root = Path(os.environ.get("SLICERDSPACE_SMOKE_ROOT", "/tmp"))
                slicer.util.mainWindow().grab().save(str(capture_root / "gui.png"))
                widget._runStatusPanel.grab().save(str(capture_root / "result-panel.png"))
                slicer.app.exit(0)
            except Exception:
                fail()
        def failed_done(status, details, log_path):
            try:
                assert status == "failed" and "intentional failure" in details
                print("GUI_SETUP_FAILURE=True", flush=True)
                assert start_setup_job(owner, "Synthetic cancellation test", [command], done)
                qt.QTimer.singleShot(1500, owner._setupJob.cancel)
            except Exception:
                fail()

        def success_done(status, details, log_path):
            try:
                assert status == "completed" and "step two" in details
                print("GUI_SETUP_SEQUENCE=True", flush=True)
                failure = (program, ["--launch", "PythonSlicer", "-u", "-c", "print('intentional failure'); raise SystemExit(7)"], {})
                assert start_setup_job(owner, "Synthetic failure test", [failure], failed_done)
            except Exception:
                fail()
        success_commands = [(program, ["--launch", "PythonSlicer", "-u", "-c", f"print('{message}')"], {})
                            for message in ("step one", "step two")]
        assert start_setup_job(owner, "Synthetic success test", success_commands, success_done)
    except Exception:
        fail()


qt.QTimer.singleShot(5000, run)
qt.QTimer.singleShot(90000, lambda: slicer.app.exit(3))
