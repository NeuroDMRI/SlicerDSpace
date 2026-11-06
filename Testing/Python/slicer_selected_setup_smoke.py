"""Real GUI/worker wiring without network installs or patient processing."""

import json
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

import qt
import slicer


def run():
    root = Path(os.environ["SLICERDSPACE_SMOKE_ROOT"])
    records = []
    originals = []
    try:
        slicer.util.selectModule("SlicerDSpace")
        owner = slicer.modules.slicerdspace.widgetRepresentation().self()
        import shared.ui_selected_setup as setup
        installed, loaded = setup.extension_states(slicer.app.extensionsManagerModel(), slicer.modules)
        assert set(loaded) <= set(installed)
        class ExposedManagerOnly:
            def isExtensionInstalled(self, name):
                return True
        _, fallback_loaded = setup.extension_states(ExposedManagerOnly(), slicer.modules)
        assert ('SlicerDMRI' in fallback_loaded) == hasattr(slicer.modules, 'dwitodtiestimation')
        records.append('actual_slicer_extension_api_and_strict_510_fallback')
        from shared.selected_setup import build_setup_plan
        language = owner._onboardingWidgets["language_combo"]
        language.setCurrentIndex(language.findData("en"))
        assert owner._managedRuntimeSetupButton.text == "Prepare selected analysis environment"
        assert owner._offlineModelResourcesButton.text == "Use offline model resources"
        records.append("unified_and_offline_buttons")
        from shared.setup_resume import remember_setup, complete_setup, refresh_setup_button, pending_setup
        remember_setup('synthetic-restart-request')
        refresh_setup_button(owner)
        assert pending_setup()['request'] == 'synthetic-restart-request'
        assert 'Continue' in owner._managedRuntimeSetupButton.text
        language.setCurrentIndex(language.findData('zh'))
        assert owner._managedRuntimeSetupButton.text == '继续准备所选分析环境'
        complete_setup('synthetic-restart-request')
        refresh_setup_button(owner)
        language.setCurrentIndex(language.findData('en'))
        assert owner._managedRuntimeSetupButton.text == 'Prepare selected analysis environment'
        records.append('persistent_pending_setup_caption_without_automatic_execution')
        with slicer.util.NodeModify(owner._parameterNode):
            for field in ("targetPreprocessing", "targetSurface", "targetSurfacePostprocess", "targetSegmentation",
                          "targetParcellation", "targetMicrostructure", "targetTractography", "targetBundles", "targetConnectome"):
                setattr(owner._parameterNode, field, False)
        missing = SimpleNamespace(key="python:numpy", kind="python", status="missing", required=True,
                                  label="NumPy", targets=("validation",), install_group="python", install_hint="Install numpy",
                                  details={"package": "numpy>=1"})
        pending = SimpleNamespace(statuses=(missing,), targets=("validation",), blocker_count=1)
        scientific_statuses = tuple(SimpleNamespace(key='python:' + name, label=name, targets=('validation',),
            install_group='python', install_hint='', required=True, status='available', kind='python',
            details={'import': name, 'python_executable': str(Path(slicer.app.slicerHome) / 'bin/PythonSlicer')})
            for name in ('numpy', 'nibabel', 'vtk', 'torch'))
        ready = SimpleNamespace(statuses=scientific_statuses, targets=("validation",), blocker_count=0)
        plan = build_setup_plan(pending)
        def accept_confirmation():
            for widget in qt.QApplication.topLevelWidgets():
                if widget.objectName == "selectedSetupConfirmationDialog":
                    widget.accept()
        qt.QTimer.singleShot(200, accept_confirmation)
        assert setup._confirm(owner, plan)
        records.append("real_confirmation_dialog")
        terms_checks = []
        def check_terms_gate():
            for dialog in qt.QApplication.topLevelWidgets():
                if dialog.objectName != "selectedSetupConfirmationDialog" or not dialog.visible:
                    continue
                try:
                    # B7: no checkboxes; one "Accept licences and install" action lists FSL and PNL.
                    boxes = slicer.util.findChildren(dialog, className="QCheckBox")
                    buttons = slicer.util.findChildren(dialog, className="QDialogButtonBox")[0]
                    terms_checks.append(len(boxes) == 0)
                    terms_checks.append(buttons.button(qt.QDialogButtonBox.Ok).text == "Accept licences and install")
                finally:
                    dialog.reject()
        terms_plan = dict(plan, fsl_license_required=True, pnl_terms_required=True, managed_profiles=["fsl_cpu", "pnl_cnn"])
        qt.QTimer.singleShot(200, check_terms_gate)
        assert not setup._confirm(owner, terms_plan)
        assert terms_checks == [True, True], terms_checks
        records.append("one_accept_licences_and_install_action")
        for name in ("_run_environment_dependency_check", "_update_environment_result_label", "_update_environment_path_status_labels"):
            originals.append((owner, name, getattr(owner, name)))
        owner._run_environment_dependency_check = lambda: ready
        owner._update_environment_result_label = lambda result: None
        owner._update_environment_path_status_labels = lambda result: None
        owner._lastEnvironmentResult = pending
        originals.append((setup, "_confirm", setup._confirm))
        setup._confirm = lambda *args: True
        assert setup.prepare_selected_environment(owner, pending)

        def wait():
            deadline = time.monotonic() + 90
            while (getattr(owner, "_setupJob", None) is not None or getattr(owner, "_selectedSetupActive", False)):
                if time.monotonic() > deadline:
                    raise RuntimeError("Selected setup worker did not finish")
                loop = qt.QEventLoop()
                qt.QTimer.singleShot(100, loop.quit)
                loop.exec_()
        wait()
        results = list(Path(owner._latestEnvironmentReportDir).glob("python-*-result.json"))
        assert results and json.loads(results[-1].read_text())["status"] == "completed"
        assert json.loads((Path(owner._latestEnvironmentReportDir) / "result.json").read_text())["status"] == "ready"
        records.append("real_cancellable_python_worker_and_final_recheck")
        assert setup.prepare_selected_environment(owner, ready)
        wait()
        functional = json.loads((Path(owner._latestEnvironmentReportDir) / 'functional-0-result.json').read_text())
        assert functional['status'] == 'passed', functional
        assert 'vtk:streamline_write_read' in functional['checks'], functional
        records.append('real_functional_worker_writes_and_reads_volume_and_streamline')
        from unittest.mock import patch
        with patch('shared.setup_selftest.requests_for_report', return_value=[dict(python=None, imports=['torch'],
                models=[dict(path='/does/not/exist.pt', sha256='0' * 64, torchscript=True)])]):
            assert setup.prepare_selected_environment(owner, ready)
            wait()
        assert owner._setupFunctionalFailure
        assert json.loads((Path(owner._latestEnvironmentReportDir) / 'result.json').read_text())['status'] == 'failed'
        with patch('shared.ui_environment_check.request_prepare_environment') as retry:
            owner.ui.runSelectedButton.click()
            retry.assert_called_once_with(owner)
        assert not owner._pipelineActive
        records.append('failed_functional_test_prevents_ready_and_run')
        assert setup.prepare_selected_environment(owner, ready)
        wait()
        assert not owner._setupFunctionalFailure
        records.append('explicit_retest_recovers_without_reinstalling')
        optional_missing = SimpleNamespace(key="tractcloud_data_dir", kind="setting", status="missing", required=True,
            label="TractCloud inference resources", targets=("validation",), install_group="tractcloud", install_hint="Prepare", details={})
        optional_report = SimpleNamespace(statuses=(optional_missing,), targets=("validation",), blocker_count=1)
        assert setup.prepare_selected_environment(owner, optional_report)
        wait()
        resource_receipt = json.loads((Path(owner._latestEnvironmentReportDir) / "optional-resources-result.json").read_text())
        assert resource_receipt["status"] == "completed" and resource_receipt["profiles"] == ["tractcloud"], resource_receipt
        assert Path(owner._read_environment_path_setting("tractcloud_data_dir")).is_dir()
        assert json.loads((Path(owner._latestEnvironmentReportDir) / "result.json").read_text())["status"] == "ready"
        records.append("real_optional_resource_worker_reuses_verified_files_and_saves_paths_without_network")
        before = owner._read_environment_path_setting("model_assets_dir")
        owner._parameterNode.targetSegmentation = True
        empty = root / "empty-offline-directory"
        empty.mkdir(exist_ok=True)
        originals.append((qt.QFileDialog, "getExistingDirectory", qt.QFileDialog.getExistingDirectory))
        qt.QFileDialog.getExistingDirectory = lambda *args: str(empty)
        assert setup.use_offline_assets(owner)
        wait()
        assert owner._read_environment_path_setting("model_assets_dir") == before
        records.append("failed_offline_check_preserves_previous_path")
        owner._setupRestartRequired = True
        owner.ui.runSelectedButton.click()
        assert "Restart" in owner.ui.statusLabel.text
        owner._setupRestartRequired = False
        records.append("restart_guard_blocks_analysis")
        language.setCurrentIndex(language.findData("zh"))
        assert owner._managedRuntimeSetupButton.text == "准备所选分析环境"
        assert owner._offlineModelResourcesButton.text == "使用离线模型资源"
        records.append("chinese_translation")
        import runpy
        setup_helper = runpy.run_path(str(Path(os.environ['SLICERDSPACE_AGENT_SETUP_HELPER'])))
        request = dict(mode='plan', targets=['validation'], model_assets_dir=os.environ['SLICERDSPACE_MODEL_ASSETS_DIR'],
                       parameters={'brainMaskBackend': 'provided'})
        before_nodes = slicer.mrmlScene.GetNumberOfNodes()
        for mode in ('plan', 'prepare'):
            path = root / ('agent-setup-' + mode + '.json')
            path.unlink(missing_ok=True)
            setup_helper['start'](dict(request, mode=mode), path)
            deadline = time.monotonic() + 90
            while not path.is_file() or json.loads(path.read_text()).get('status') == 'checking':
                if time.monotonic() > deadline:
                    raise RuntimeError('Agent setup helper did not produce a receipt')
                loop = qt.QEventLoop()
                qt.QTimer.singleShot(100, loop.quit)
                loop.exec_()
            receipt = json.loads(path.read_text())
            assert receipt['status'] == ('planned' if mode == 'plan' else 'ready'), receipt
            assert not receipt['scientific_analysis_started']
        assert slicer.mrmlScene.GetNumberOfNodes() == before_nodes
        records.append('real_agent_helper_plan_and_prepare_without_patient_analysis_or_scene_reset')
        owner._advancedSettingsPanel.collapsed = False
        owner.ui.environmentCollapsibleButton.collapsed = False
        slicer.app.processEvents()
        owner._uiWidget.grab().save(str(root / "environment_setup.png"))
        (root / "selected_setup.json").write_text(json.dumps({"status": "passed", "checks": records,
            "scope": "Real Qt, MRML and background workers; no network installation or scientific QC"}, indent=2))
        print("SELECTED_SETUP_GUI_PASS", flush=True)
        code = 0
    except Exception:
        (root / "selected_setup.json").write_text(json.dumps({"status": "failed", "checks": records,
                                                               "error": traceback.format_exc()}, indent=2))
        traceback.print_exc()
        code = 1
    finally:
        for obj, name, original in reversed(originals):
            setattr(obj, name, original)
    slicer.app.exit(code)


qt.QTimer.singleShot(4000, run)
