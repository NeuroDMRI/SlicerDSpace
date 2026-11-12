"""Actual multi-select dialogs, MRML persistence and independent setup panel."""

import json
import os
from pathlib import Path
import traceback

import qt
import slicer

ROOT = Path(os.environ['SLICERDSPACE_SMOKE_ROOT'])


def run():
    checks = []
    try:
        slicer.util.selectModule('SlicerDSpace')
        owner = slicer.modules.slicerdspace.widgetRepresentation().self()
        from shared.ui_algorithm_picker import choose_algorithms, selected_algorithms
        combo = owner._onboardingWidgets['language_combo']
        combo.setCurrentIndex(combo.findData('en'))
        assert len(owner._algorithmButtons) == 3
        assert not owner._advancedSettingsPanel.isAncestorOf(owner.ui.environmentCollapsibleButton)
        assert not owner._advancedSettingsPanel.isAncestorOf(owner.ui.resultsCollapsibleButton)
        # Results and QC sat in the collapsed Advanced panel (hidden by it) before it moved to the main column.
        assert not owner.ui.resultsCollapsibleButton.isHidden() and owner.ui.resultsCollapsibleButton.visible
        from shared.ui_advanced_layout import CATEGORIES
        assert owner._advancedCategorySelector.count == len(CATEGORIES)
        checks.append('three_pickers_and_independent_environment')
        for button in owner._algorithmButtons.values():
            assert button.text == '...' and button.width == 28 and button.height == 22
            assert 'Select' in button.accessibleName and 'multiple allowed' in button.toolTip
        checks.append('compact_ellipsis_buttons_with_accessible_actions')
        def choose(family, values=None, cancel=False):
            def interact():
                for dialog in qt.QApplication.topLevelWidgets():
                    if dialog.objectName != 'algorithmPicker_' + family or not dialog.visible:
                        continue
                    if cancel:
                        dialog.reject()
                        return
                    boxes = slicer.util.findChildren(dialog, className='QCheckBox')
                    buttons = slicer.util.findChildren(dialog, className='QDialogButtonBox')[0]
                    for box in boxes:
                        if box.enabled:
                            box.checked = False
                    if family != 'microstructure':
                        assert not buttons.button(qt.QDialogButtonBox.Apply).enabled
                    for box in boxes:
                        box.checked = box.objectName.removeprefix('algorithm_') in values
                    assert buttons.button(qt.QDialogButtonBox.Apply).enabled
                    buttons.button(qt.QDialogButtonBox.Apply).click()
                    return
            qt.QTimer.singleShot(200, interact)
            return choose_algorithms(owner, family)
        assert choose('microstructure', ['dti', 'freewater', 'noddi', 'dki'])
        assert selected_algorithms(owner._parameterNode, 'microstructure') == ('dti', 'freewater', 'noddi', 'dki')
        assert choose('tractography', ['csd_act', 'ukf'])
        assert owner._parameterNode.orchestrationTractographyBackend == 'csd_act,ukf'
        assert choose('bundles', ['wmql', 'tractcloud'])
        assert owner._parameterNode.orchestrationBundleBackend == 'wmql,tractcloud'
        checks.append('all_three_real_multiselect_dialogs')
        before = owner._parameterNode.orchestrationBundleBackend
        assert not choose('bundles', cancel=True)
        assert owner._parameterNode.orchestrationBundleBackend == before
        checks.append('empty_selection_disabled_and_cancel_preserved')
        owner._save_common_user_settings()
        from shared.user_config import load_user_config
        defaults = load_user_config()['run_defaults']
        assert defaults['tractography_backend'] == 'csd_act,ukf'
        assert defaults['bundle_backend'] == 'wmql,tractcloud'
        checks.append('selections_persist_in_shared_config')
        owner._advancedSettingsPanel.collapsed = True
        slicer.util.mainWindow().resize(1280, 900)
        dock = slicer.util.findChild(slicer.util.mainWindow(), 'PanelDockWidget')
        dock.setFixedWidth(480)
        slicer.util.findChild(slicer.util.mainWindow(), 'DataProbeCollapsibleWidget').collapsed = True
        slicer.app.processEvents()
        geometry = {}
        for family, name in (('microstructure', 'targetMicrostructureCheckBox'),
                             ('tractography', 'targetTractographyCheckBox'), ('bundles', 'targetBundlesCheckBox')):
            checkbox = getattr(owner.ui, name)
            label = owner._algorithmSummaryLabels[name]
            geometry[family] = {key: [widget.x, widget.y, widget.width, widget.height, widget.text]
                                for key, widget in (('check', checkbox), ('label', label), ('button', owner._algorithmButtons[family]))}
            assert label.x >= checkbox.x + checkbox.width + 4, geometry[family]
        (ROOT / 'selection_geometry.json').write_text(json.dumps(geometry, indent=2))
        # Wrapped summaries must get the height of all their lines (macOS GUI capture, 2026-10-23: at a narrow dock
        # "DTI + FW + NODDI + DKI (WLS)" wrapped to two lines and the row kept one line, clipping the second).
        # The row that holds a wrapped summary must be as tall as its text, or the row clips it (the label
        # alone can be tall enough while its row is not). 260 px is Slicer's default dock width.
        for language in ('zh', 'en'):
            combo.setCurrentIndex(combo.findData(language))
            for width in (260, 400, 360):
                dock.setFixedWidth(width)
                for _ in range(5):
                    slicer.app.processEvents()
                wrapped = 0
                for name, label in owner._algorithmSummaryLabels.items():
                    metrics = label.fontMetrics()
                    need = metrics.boundingRect(0, 0, label.contentsRect().width(), 100000,
                                                int(qt.Qt.TextWordWrap), label.text).height()
                    wrapped += need > metrics.height()
                    row = label.parentWidget()
                    assert label.height >= need and row.height >= label.y + need, (
                        language, width, name, label.width, label.height, row.height, need, label.text)
                assert width > 260 or wrapped, (language, width, 'no summary wrapped at the narrowest dock')
        checks.append('wrapped_algorithm_summaries_not_clipped')
        dock.setFixedWidth(480)
        slicer.app.processEvents()
        slicer.util.mainWindow().grab().save(str(ROOT / 'main_multiselect.png'))
        owner._workflowToolsDialog.show()
        owner._advancedSettingsPanel.collapsed = False
        for index in range(len(CATEGORIES)):
            owner._advancedCategorySelector.setCurrentIndex(index)
            slicer.app.processEvents()
            owner._fitAdvancedPage()
            owner._advancedSettingsPanel.grab().save(str(ROOT / f'advanced_{index}.png'))
            assert owner._uiWidget.width <= 480, (index, owner._uiWidget.width)
        checks.append('compact_parameter_categories_and_opt_in_experiments')
        combo.setCurrentIndex(combo.findData('zh'))
        assert all(button.text == '...' and button.width == 28 and '选择' in button.accessibleName
                   and '可多选' in button.toolTip for button in owner._algorithmButtons.values())
        assert owner._dependencyDialogButton.text == '依赖与诊断'
        checks.append('chinese_picker_and_setup_labels')
        record = {'status': 'passed', 'checks': checks}
        code = 0
    except Exception:
        record = {'status': 'failed', 'checks': checks, 'error': traceback.format_exc()}
        traceback.print_exc()
        code = 1
    (ROOT / 'algorithm_picker.json').write_text(json.dumps(record, indent=2))
    slicer.app.exit(code)


qt.QTimer.singleShot(4000, run)
