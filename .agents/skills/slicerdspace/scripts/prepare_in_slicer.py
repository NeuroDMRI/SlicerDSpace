"""Preview or prepare selected dependencies in Slicer, retaining its consent dialog."""

import json
import os
from pathlib import Path
from uuid import uuid4

TARGET_FIELDS = {
    'preprocessing': 'targetPreprocessing', 'segmentation': 'targetSegmentation',
    'parcellation': 'targetParcellation', 'surface': 'targetSurface',
    'surface_postprocess': 'targetSurfacePostprocess', 'microstructure': 'targetMicrostructure',
    'tractography': 'targetTractography', 'connectome': 'targetConnectome',
}
PARAMETERS = {
    'brainMaskBackend', 'brainMaskDevice', 'ddsegDevice', 'device',
    'orchestrationMicrostructureStages', 'orchestrationDkiFitMethod',
    'orchestrationTractographyBackend', 'orchestrationBundleBackend', 'wmaAtlas', 'connectomeBackend',
}


def validate_request(record):
    if not isinstance(record, dict) or set(record) - {'mode', 'targets', 'model_assets_dir', 'parameters', 'prepare_mask'}:
        raise ValueError('Use mode, targets, model_assets_dir, parameters and optional prepare_mask')
    if record.get('mode', 'plan') not in {'plan', 'prepare'}:
        raise ValueError('mode must be plan or prepare')
    targets = record.get('targets')
    if not isinstance(targets, list) or not targets or any(not isinstance(value, str) for value in targets):
        raise ValueError('targets must be a nonempty list of analysis names')
    if set(targets) - set(TARGET_FIELDS) - {'validation', 'bundles', 'wma', 'tractcloud'}:
        raise ValueError('Unknown setup target')
    parameters = record.get('parameters', {})
    if not isinstance(parameters, dict) or set(parameters) - PARAMETERS or any(not isinstance(value, str) for value in parameters.values()):
        raise ValueError('parameters must contain only supported scientific string settings')
    root = record.get('model_assets_dir')
    if not isinstance(root, str) or not Path(root).expanduser().is_absolute():
        raise ValueError('model_assets_dir must be the absolute storage path chosen by the user')
    if not isinstance(record.get('prepare_mask', False), bool):
        raise ValueError('prepare_mask must be a boolean')
    return record


def extension_root(module_directory):
    module = Path(module_directory).resolve()
    if (module.name == 'qt-scripted-modules' and module.parent.name.startswith('Slicer-')
            and module.parent.parent.name == 'lib'):
        return module.parents[2]
    return module


def start(record, receipt, *, exit_on_finish=False):
    import qt
    import slicer
    from shared.ui_environment_check import start_environment_check
    from shared.ui_selected_setup import prepare_selected_environment, selected_setup_plan

    record = validate_request(record)
    slicer.util.selectModule('SlicerDSpace')
    owner = slicer.modules.slicerdspace.widgetRepresentation().self()
    if owner._pipelineActive or getattr(owner, '_setupJob', None) or getattr(owner, '_selectedSetupActive', False):
        raise RuntimeError('An existing job is active; do not replace its selections')
    asset_root = Path(record['model_assets_dir']).expanduser().resolve()
    if asset_root.is_relative_to(extension_root(owner.logic.module_dir)):
        raise ValueError('Choose resource storage outside the extension installation')
    node = owner._parameterNode
    # Validate on a temporary parameter node before changing the active selection.
    from shared.algorithm_selection import BUNDLES, choice_value
    parameters = dict(record.get('parameters', {}))
    if 'orchestrationTractographyBackend' in parameters:
        parameters['orchestrationTractographyBackend'] = choice_value(parameters['orchestrationTractographyBackend'])
    if 'orchestrationBundleBackend' in parameters:
        parameters['orchestrationBundleBackend'] = choice_value(parameters['orchestrationBundleBackend'], BUNDLES, 'wmql')
    bundles = [name for target, name in (('bundles', 'wmql'), ('wma', 'wma'), ('tractcloud', 'tractcloud'))
               if target in record['targets']]
    if bundles and 'orchestrationBundleBackend' not in parameters:
        parameters['orchestrationBundleBackend'] = choice_value(','.join(bundles), BUNDLES, 'wmql')
    # Parameter wrapper construction validates Choice fields without touching the scene.
    scratch = slicer.vtkMRMLScriptedModuleNode()
    candidate = type(node)(scratch)
    for field, value in parameters.items():
        setattr(candidate, field, value)
    with slicer.util.NodeModify(node):
        for target, field in TARGET_FIELDS.items():
            setattr(node, field, target in record['targets'])
        node.targetBundles = bool(bundles)
        for field, value in parameters.items():
            setattr(node, field, value)
    edit = owner._environmentPathLineEdits['model_assets_dir']
    edit.setText(str(asset_root))
    original_check = owner._environment_check_kwargs
    if record.get('prepare_mask'):
        def options(**kwargs):
            return dict(original_check(**kwargs), brain_mask_required=True)
        owner._environment_check_kwargs = options

    request_id = uuid4().hex
    def write(result):
        path = Path(receipt).expanduser().absolute()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(dict(result, request_id=request_id), indent=2) + '\n', encoding='utf-8')
        temporary.replace(path)

    write(dict(status='checking', scientific_analysis_started=False))

    def save(result):
        owner._environment_check_kwargs = original_check
        write(result)
        print('[Agent setup] ' + json.dumps(result), flush=True)
        if exit_on_finish:
            slicer.app.exit(0 if result['status'] in {'planned', 'ready', 'restart_required'} else 1)

    def poll():
        if getattr(owner, '_cleanedUp', False):
            save(dict(status='cancelled', error='Module was closed during setup'))
        elif getattr(owner, '_setupJob', None) or getattr(owner, '_selectedSetupActive', False):
            qt.QTimer.singleShot(250, poll)
        else:
            try:
                folder = Path(owner._latestEnvironmentReportDir)
                result = json.loads((folder / 'result.json').read_text())
                save(dict(result, setup_directory=str(folder), scientific_analysis_started=False))
            except Exception as exc:
                save(dict(status='failed', error=str(exc)))

    def checked(report):
        try:
            plan = selected_setup_plan(owner, report)
            if record.get('mode', 'plan') == 'plan':
                save(dict(status='planned', plan=plan, environment=report.to_dict(), scientific_analysis_started=False))
            else:
                before = owner._latestEnvironmentReportDir
                if prepare_selected_environment(owner, report):
                    qt.QTimer.singleShot(250, poll)
                elif owner._latestEnvironmentReportDir != before:
                    poll()
                else:
                    save(dict(status='cancelled', error='Preparation was not confirmed or could not start'))
        except Exception as exc:
            save(dict(status='failed', error=str(exc)))

    if not start_environment_check(owner, checked, on_failure=lambda status, message: save(dict(status=status, error=message))):
        save(dict(status='failed', error='Environment check could not start'))


def main():
    request = os.environ.get('SLICERDSPACE_SETUP_REQUEST')
    receipt = os.environ.get('SLICERDSPACE_SETUP_RECEIPT')
    if not request or not receipt:
        raise ValueError('Set SLICERDSPACE_SETUP_REQUEST and SLICERDSPACE_SETUP_RECEIPT')
    import qt
    def begin():
        try:
            start(json.loads(Path(request).read_text(encoding='utf-8')), receipt,
                  exit_on_finish=os.environ.get('SLICERDSPACE_SETUP_EXIT') == '1')
        except Exception as exc:
            destination = Path(receipt).expanduser().absolute()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(dict(status='failed', error=str(exc))) + '\n', encoding='utf-8')
            if os.environ.get('SLICERDSPACE_SETUP_EXIT') == '1':
                import slicer
                slicer.app.exit(1)
    qt.QTimer.singleShot(0, begin)


if __name__ == '__main__':
    main()
