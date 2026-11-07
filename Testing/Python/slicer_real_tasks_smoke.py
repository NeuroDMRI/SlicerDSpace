"""Execute configured scientific cases through the real Slicer widget/run button.

No mocked model, QC, pipeline or output is used. Route-specific extra CLI flags
are recorded and appended to the widget's normal background launch command.
"""

import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import qt
import slicer

ROOT = Path(os.environ["SLICERDSPACE_SMOKE_ROOT"])
CONFIG = json.loads(Path(os.environ["SLICERDSPACE_TASKS_CONFIG"]).read_text())
RESULT = {"kind": "real_slicer_tasks", "cases": [], "human_anatomical_review": "pending"}
STATE = {"index": 0}
TARGETS = {
    "preprocessing": "targetPreprocessing", "segmentation": "targetSegmentation",
    "parcellation": "targetParcellation", "surface": "targetSurface",
    "surface_postprocess": "targetSurfacePostprocess", "microstructure": "targetMicrostructure",
    "tractography": "targetTractography", "bundles": "targetBundles", "connectome": "targetConnectome",
}


def persist():
    (ROOT / "real_tasks.json").write_text(json.dumps(RESULT, indent=2) + "\n")


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def finish_case(error=None):
    record = STATE["record"]
    record["elapsed_seconds"] = time.monotonic() - STATE["started"]
    record["errors"] = STATE["errors"]
    if error:
        record["error"] = str(error)
    record["status"] = "FAIL" if error or STATE["errors"] else "PASS"
    RESULT["cases"].append(record)
    persist()
    print("REAL_TASK_CASE=" + json.dumps(record), flush=True)
    STATE["index"] += 1
    qt.QTimer.singleShot(1000, next_case)


def next_case():
    if STATE["index"] == len(CONFIG["cases"]):
        slicer.app.exit(0 if all(c["status"] == "PASS" for c in RESULT["cases"]) else 2)
        return
    case = CONFIG["cases"][STATE["index"]]
    STATE.update(case=case, started=time.monotonic(), errors=[], record={"name": case["name"], "config": case})
    widget = STATE["widget"]
    try:
        STATE["preserved"] = {path: file_digest(path) for path in case.get("preserve_outputs", [])}
        slicer.mrmlScene.Clear(0)
        widget.initializeParameterNode()
        for key, value in CONFIG.get("external_dependencies", {}).items():
            widget._save_environment_path_setting(key, value)
            edit = widget._environmentPathLineEdits.get(key)
            if edit is not None:
                edit.setText(value)
        node = widget._parameterNode
        with slicer.util.NodeModify(node):
            node.subjectId = case["subject_id"]
            node.inputDirectory = case["subject_root"]
            node.outputDirectory = case["output_root"]
            node.orchestrationRunName = case.get("run_name", case["name"])
            node.autoInstallPythonDependencies = False
            node.batchMode = False
            for target, field in TARGETS.items():
                setattr(node, field, target in case["targets"])
            node.orchestrationTractographyBackend = case.get("tractography_backend", "csd_act")
            node.orchestrationBundleBackend = case.get("bundle_backend", "wmql")
            node.orchestrationMicrostructureStages = case.get("microstructure_stages", "dti")
            node.orchestrationSelectStreamlines = case.get("streamlines", 50000)
            node.device = node.ddsegDevice = "cuda:0"
            node.connectomeBackend = case.get("connectome_backend", "python")
            node.preprocessingPhaseEncodingDirection = case.get("pe", "")
            node.preprocessingReadoutTime = case.get("readout", 0.)
            node.preprocessingReversePhaseB0 = case.get("reverse_b0", "")
            node.wmaInputTractography = case.get("wma_input", "")
        STATE["manifest"] = Path(case["output_root"]) / "00_run_metadata" / (node.orchestrationRunName + "_orchestration_manifest.json")
        STATE["previous_mtime"] = STATE["manifest"].stat().st_mtime_ns if STATE["manifest"].exists() else 0
        widget._checkCanApply()
        assert widget.ui.runSelectedButton.enabled, "Run button unexpectedly disabled"
        widget.ui.runSelectedButton.click()
        qt.QTimer.singleShot(1000, poll)
    except Exception as exc:
        traceback.print_exc()
        finish_case(exc)


def poll():
    widget = STATE["widget"]
    process = getattr(widget, "_backgroundProcess", None)
    active = bool(widget._pipelineActive) or (process is not None and process.state() != qt.QProcess.NotRunning)
    elapsed = time.monotonic() - STATE["started"]
    if elapsed > CONFIG.get("case_timeout_seconds", 3600):
        widget.cleanup()
        finish_case("Task timeout")
        return
    new_manifest = STATE["manifest"].is_file() and STATE["manifest"].stat().st_mtime_ns > STATE["previous_mtime"]
    if active or (not new_manifest and elapsed < 90 and not STATE["errors"]):
        qt.QTimer.singleShot(2000, poll)
        return
    try:
        payload = json.loads(STATE["manifest"].read_text())
        snapshot = ROOT / (STATE["case"]["name"] + "_manifest.json")
        snapshot.write_text(json.dumps(payload, indent=2) + "\n")
        STATE["record"]["manifest_snapshot"] = str(snapshot)
        STATE["record"]["manifest"] = str(STATE["manifest"])
        STATE["record"]["task_statuses"] = {task["name"]: task["status"] for task in payload["tasks"]}
        assert payload["status"] == "completed", payload.get("status")
        assert all(task["status"] in {"passed", "completed", "skipped"} for task in payload["tasks"])
        requested = set(STATE["case"].get("expected_tasks", []))
        assert requested <= set(STATE["record"]["task_statuses"]), "Requested task was not dispatched"
        for name, expected_status in STATE["case"].get("expected_statuses", {}).items():
            assert STATE["record"]["task_statuses"].get(name) == expected_status, (name, expected_status)
        preserved = {path: file_digest(path) for path in STATE["preserved"]}
        assert preserved == STATE["preserved"], "Another run's output was moved or changed"
        STATE["record"]["preserved_output_sha256"] = preserved
        loaded = widget.logic.loadKeyResultsFromManifest(STATE["manifest"], mode="reuse", max_items=40)
        STATE["record"]["scene_loading"] = loaded
        from shared.result_scene import result_candidates
        candidates, _ = result_candidates(payload)
        eligible = {c["path"] for c in candidates[:40]}
        displayed = {c["path"] for c in loaded["loaded"] + loaded["reused"]}
        assert eligible == displayed, {"not_loaded": sorted(eligible - displayed), "skipped": loaded["skipped"]}
        slicer.util.mainWindow().grab().save(str(ROOT / (STATE["case"]["name"] + ".png")))
        finish_case()
    except Exception as exc:
        traceback.print_exc()
        finish_case(exc)


def start():
    try:
        slicer.util.selectModule("SlicerDSpace")
        widget = slicer.modules.slicerdspace.widgetRepresentation().self()
        STATE["widget"] = widget
        original_log = widget.appendLog
        def log(message):
            print(str(message), flush=True)
            original_log(message)
        widget.appendLog = log
        slicer.app.temporaryPath = str(ROOT / "temp")
        slicer.util.infoDisplay = lambda message, *args, **kwargs: print("TASK_INFO=" + str(message), flush=True)
        slicer.util.errorDisplay = lambda message, *args, **kwargs: STATE["errors"].append(str(message))
        widget._show_actionable_error = lambda **kwargs: STATE["errors"].append(str(kwargs))
        original_finished_loading = widget._load_background_manifest_result

        def finished_loading(path):
            assert widget._pipelineActive, "Run was enabled during the completion callback"
            original_finished_loading(path)
            assert widget._pipelineActive, "Result loading released the next run too early"

        widget._load_background_manifest_result = finished_loading
        original_build = widget._build_background_cli_args

        def build(common, external):
            args = original_build(common, external)
            args += STATE["case"].get("extra_cli_args", [])
            STATE["record"]["launch_arguments"] = args
            return args

        widget._build_background_cli_args = build
        next_case()
    except Exception:
        traceback.print_exc()
        slicer.app.exit(2)


qt.QTimer.singleShot(5000, start)
