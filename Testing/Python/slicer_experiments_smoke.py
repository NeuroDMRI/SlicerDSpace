"""Exercise real advanced widgets, preview, QProcess execution and cancellation."""

import json
import os
from pathlib import Path
import traceback

import qt
import slicer

ROOT = Path(os.environ['SLICERDSPACE_SMOKE_ROOT'])


def wait_until(predicate, seconds=40):
    import time
    deadline = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError('Timed out waiting for experiment process')
        slicer.app.processEvents()
        time.sleep(0.05)


def run():
    from unittest.mock import patch
    checks = []
    try:
        slicer.util.selectModule('SlicerDSpace')
        owner = slicer.modules.slicerdspace.widgetRepresentation().self()
        node = owner._parameterNode
        from shared.ui_experiments import preview, request_and_plan, apply_native_route
        owner._advancedSettingsPanel.collapsed = False
        owner._advancedCategorySelector.setCurrentIndex(4)
        controls = owner._experimentWidgets
        assert not node.experimentsEnabled
        assert not owner.ui.environmentCollapsibleButton.isAncestorOf(controls['enabled'])
        owner._onboardingWidgets['language_combo'].setCurrentIndex(1)
        assert '实验' in owner._advancedCategorySelector.itemText(4)
        owner._onboardingWidgets['language_combo'].setCurrentIndex(0)
        assert 'Experiments' in owner._advancedCategorySelector.itemText(4)
        checks.append('optional_experiment_category_and_bilingual_controls')
        with patch.object(slicer.util, 'confirmYesNoDisplay', return_value=True):
            apply_native_route(owner)
        assert node.brainMaskBackend == 'pnl_cnn' and node.orchestrationTractographyBackend == 'dti_det'
        assert node.device == node.ddsegDevice == 'auto'
        checks.append('explicit_native_route_preserves_device_preference')
        cohort = ROOT / 'cohort'
        for name in ('sub-01','sub-02'):
            folder = cohort / name / 'input'
            folder.mkdir(parents=True)
            # Discovery uses file layout; these fixtures are never model inputs.
            for file in ('data.nii.gz','bvals','bvecs'):
                (folder / file).write_text('scheduler fixture')
        with slicer.util.NodeModify(node):
            node.subjectId = ''
            node.inputDirectory = ''
            node.outputDirectory = str(ROOT / 'results')
            node.experimentScope = 'cohort'
            node.experimentRoot = str(cohort)
            node.experimentJobs = 2
            node.experimentsEnabled = True
            node.experimentCliOptions = '--mrtrix-nthreads 4'
            node.experimentVariants = '[{"name":"baseline","args":[]},{"name":"ukf","args":["--tractography-backend","ukf","--bundle-tractography-backend","ukf"]}]'
        slicer.app.processEvents()
        assert controls['enabled'].checked and controls['jobs'].value == 2
        assert controls['root'].visible
        assert preview(owner) is not None, controls['receipt'].toPlainText()
        request, plan, common, external = request_and_plan(owner)
        assert len(plan['subjects']) == 4
        assert not Path(plan['output_root']).exists()
        checks.append('cohort_preview_without_subject_id_or_output_mutation')
        controls['options'].setPlainText('--skip-environment-check')
        assert preview(owner) is None
        assert 'not an experiment parameter' in controls['receipt'].toPlainText()
        controls['options'].setPlainText('')
        checks.append('invalid_or_unsafe_parameters_rejected_in_gui')
        # Run the real worker in Slicer's Python, with the actual CLI doctor.
        # An intentionally absent PNL runtime gives an honest failed job, and
        # tests the GUI's records rather than presenting fixtures as QC passes.
        with slicer.util.NodeModify(node):
            for attr in ('targetPreprocessing','targetSurface','targetSurfacePostprocess','targetParcellation',
                         'targetSegmentation','targetMicrostructure','targetTractography','targetBundles','targetConnectome'):
                setattr(node, attr, False)
            node.targetMicrostructure = True
            node.brainMaskBackend = 'pnl_cnn'
            node.experimentVariants = '[{"name":"missing_runtime","args":[]}]'
        with patch.object(slicer.util, 'confirmYesNoDisplay', return_value=True):
            owner._onRunSelectedClicked()
        assert owner._pipelineActive
        wait_until(lambda: owner._backgroundProcess is None, 90)
        path = Path(owner._backgroundRunContext['experiment_manifest_path'])
        record = json.loads(path.read_text())
        assert record['experiment'] and record['status'] == 'failed', record
        assert len(record['results']) == 2 and all(item['returncode'] for item in record['results'])
        assert owner._runStatusTable.rowCount >= 2
        assert 'failed' in owner._statusMessageSource.lower(), owner._statusMessageSource
        checks.append('real_background_worker_retains_dependency_failures_per_subject')
        # Cancellation must address the owned process, not another Slicer session.
        with slicer.util.NodeModify(node):
            node.experimentVariants = '[{"name":"cancel","args":[]}]'
        with patch.object(slicer.util, 'confirmYesNoDisplay', return_value=True):
            owner._onRunSelectedClicked()
        assert owner._pipelineActive
        owner._onCancelRunClicked()
        wait_until(lambda: owner._backgroundProcess is None, 35)
        assert not owner._pipelineActive
        checks.append('gui_cancel_stops_owned_experiment_process')
        owner._advancedSettingsPanel.grab().save(str(ROOT / 'experiments.png'))
        payload, code = dict(status='passed', checks=checks), 0
    except Exception:
        payload, code = dict(status='failed', checks=checks, error=traceback.format_exc()), 1
        traceback.print_exc()
    (ROOT / 'experiments.json').write_text(json.dumps(payload, indent=2))
    slicer.app.exit(code)


qt.QTimer.singleShot(2500, run)
