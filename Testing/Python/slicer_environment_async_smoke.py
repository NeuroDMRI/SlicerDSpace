"""Exercise real check cancellation, stale reports and no-op preparation in Qt."""

import json
import os
from pathlib import Path
import time
import traceback

import qt
import slicer


def run():
    root = Path(os.environ['SLICERDSPACE_SMOKE_ROOT'])
    checks = []
    try:
        slicer.util.selectModule('SlicerDSpace')
        owner = slicer.modules.slicerdspace.widgetRepresentation().self()
        owner._onboardingWidgets['language_combo'].setCurrentIndex(0)
        node = owner._parameterNode
        with slicer.util.NodeModify(node):
            for field in ('targetPreprocessing', 'targetSurface', 'targetSurfacePostprocess', 'targetSegmentation',
                          'targetParcellation', 'targetMicrostructure', 'targetTractography', 'targetBundles', 'targetConnectome'):
                setattr(node, field, False)
            node.subjectId = ''
            node.inputDirectory = ''
        subject = root / 'sub-gui'
        subject.mkdir(exist_ok=True)
        owner.ui.inputDirectoryPath.currentPath = str(subject)
        assert node.subjectId == 'sub-gui'
        node.subjectId = 'manual-id'
        owner.ui.inputDirectoryPath.currentPath = str(root)
        assert node.subjectId == 'manual-id'
        checks.append('real_folder_picker_fills_empty_id_and_preserves_manual_id')
        owner.ui.environmentCollapsibleButton.collapsed = False
        assert owner._environmentMoreOptions.collapsed
        assert not owner._offlineModelResourcesButton.isVisible()
        assert not owner._repairManagedToolsButton.isVisible()
        checks.append('secondary_environment_controls_hidden_by_default')

        def wait():
            deadline = time.monotonic() + 100
            while getattr(owner, '_setupJob', None) or getattr(owner, '_selectedSetupActive', False):
                assert time.monotonic() < deadline, 'Worker timeout'
                loop = qt.QEventLoop()
                qt.QTimer.singleShot(50, loop.quit)
                loop.exec_()

        owner.ui.validateButton.click()
        job = owner._setupJob
        assert job is not None and owner._pipelineActive
        cancel = slicer.util.findChildren(job.dialog, className='QPushButton')[0]
        cancel.click()
        wait()
        assert job.finished and job.cancelled
        assert 'cancelled' in job.log_path.read_text().lower()
        assert owner._lastEnvironmentResult is None
        assert not owner._pipelineActive and owner.ui.runSelectedButton.enabled
        checks.append('actual_check_cancel_stops_process_and_unlocks_ui_without_ready_status')

        owner.ui.validateButton.click()
        node.brainMaskBackend = 'provided'
        wait()
        assert owner._lastEnvironmentResult is None
        assert 'changed' in owner.ui.statusLabel.text.lower(), owner.ui.statusLabel.text
        checks.append('configuration_change_rejects_stale_worker_report')

        owner.ui.validateButton.click()
        wait()
        assert owner._lastEnvironmentResult.blocker_count == 0, owner._lastEnvironmentResult.to_dict()
        assert not getattr(owner, '_environmentDialog', None)
        checks.append('actual_check_updates_inline_without_modal_dialog')
        from dataclasses import replace
        backend_report = replace(owner._lastEnvironmentResult, tracking_backend_blockers={
            'csd_act': {'tractography': 'preflight missing dependencies: tckgen'}, 'ukf': {}})
        owner._show_environment_results_dialog(backend_report)
        details = slicer.util.findChildren(owner._environmentDialog, className='QPlainTextEdit')
        assert len(details) == 1
        assert 'Tractography csd_act: preflight missing dependencies: tckgen' in details[0].plainText
        assert 'Tractography ukf: dependencies available' in details[0].plainText
        owner._environmentDialog.close()
        checks.append('backend_specific_dependencies_visible_in_real_qt_dialog')
        owner._managedRuntimeSetupButton.click()
        wait()
        folder = Path(owner._latestEnvironmentReportDir)
        receipt = json.loads((folder / 'result.json').read_text())
        plan = json.loads((folder / 'plan.json').read_text())
        assert receipt['status'] == 'ready' and receipt['blockers'] == 0, receipt
        assert not any(plan[key] for key in ('extensions', 'restart_extensions', 'python_packages', 'asset_profiles', 'managed_profiles'))
        assert not any(item.visible for item in qt.QApplication.topLevelWidgets()
                       if item.objectName == 'selectedSetupConfirmationDialog')
        checks.append('prepare_ready_environment_needs_no_confirm_download_or_restart')
        from shared.demo_data import create_synthetic_demo_subject
        demo = create_synthetic_demo_subject(root / 'blocked-run-input')
        with slicer.util.NodeModify(node):
            node.inputDirectory = str(demo)
            node.subjectId = demo.name
            # Q1 (D6/D8): the native tensor fit needs no SlicerDMRI, so only the WMA bundle target is blocked by its absence.
            node.targetMicrostructure = False
            node.targetBundles = True
            node.orchestrationBundleBackend = 'wma'
            node.autoInstallPythonDependencies = False
        assert not hasattr(slicer.modules, 'dwitodtiestimation'), 'This isolation case requires SlicerDMRI to be absent'
        owner.ui.runSelectedButton.click()
        wait()
        assert owner._lastEnvironmentResult.blocker_count > 0
        assert getattr(owner, '_backgroundProcess', None) is None
        assert owner._environmentDialog.visible and not owner.ui.environmentCollapsibleButton.collapsed
        owner._environmentDialog.close()
        checks.append('real_run_preflight_stops_on_missing_native_dependency_without_partial_scientific_run')
        owner._onboardingWidgets['language_combo'].setCurrentIndex(1)
        assert '更多选项' in owner._environmentMoreOptions.text
        assert owner._environmentDetailsButton.text == '查看依赖详情'
        checks.append('chinese_secondary_controls')
        owner._uiWidget.grab().save(str(root / 'environment_async.png'))
        payload = {'status': 'passed', 'checks': checks,
                   'scope': 'Real Slicer Qt, directory picker and check processes; no network installs or scientific QC'}
        code = 0
    except Exception:
        payload = {'status': 'failed', 'checks': checks, 'error': traceback.format_exc()}
        traceback.print_exc()
        code = 2
    (root / 'environment_async.json').write_text(json.dumps(payload, indent=2))
    print('ENVIRONMENT_ASYNC=' + json.dumps(payload), flush=True)
    slicer.app.exit(code)


qt.QTimer.singleShot(5000, run)
