"""Opt-in advanced experiments; the normal Run button remains single-subject."""

from __future__ import annotations

import json
from pathlib import Path

from shared.experiments import compile_plan, make_request


TEXT = {
    'en': {
        'enabled': 'Enable experiments', 'scope': 'Input scope',
        'subject': 'Current subject', 'cohort': 'Prepared subject folders', 'root': 'Cohort root',
        'jobs': 'Concurrent subjects', 'options': 'CLI parameter overrides', 'variants': 'Named variants (JSON)',
        'hint': 'When enabled, Run SlicerDSpace submits these experiments. Each subject runs its variants sequentially in separate folders; concurrency is across subjects and shares GPU memory. Preview checks syntax/paths; each job checks dependencies before execution.',
        'preview': 'Preview experiments', 'export': 'Export request...',
        'lite': 'Use Slicer-native route...',
        'lite_title': 'Use a lighter dependency route?',
        'lite_message': 'Keep PNL CNN masking. Preprocessing is not part of this route (it needs FSL). Use DTI deterministic tracking, WMQL bundles and Python connectomes. Anatomical models still use PyTorch and downloaded weights. Optional surface postprocessing keeps its native components. PNL and CSD defaults are unchanged unless you accept this preset. Inspect the mask and tractography: these are different methods, not equivalent substitutes.',
    },
    'zh': {
        'enabled': '启用实验运行器', 'scope': '输入范围',
        'subject': '当前被试', 'cohort': '已准备好的被试文件夹', 'root': '批处理根目录',
        'jobs': '并发被试数', 'options': 'CLI 参数覆盖', 'variants': '命名实验组合（JSON）',
        'hint': '启用后，Run SlicerDSpace 提交这里的实验。同一被试的实验依次执行，输出目录隔离；不同被试可并发，共享 GPU 显存。预览检查参数和路径，每个任务执行前独立检查依赖。',
        'preview': '预览实验计划', 'export': '导出配置...',
        'lite': '使用 Slicer 原生轻量路线...',
        'lite_title': '切换到较少外部依赖的路线？',
        'lite_message': '保持 PNL CNN 掩膜；预处理（需要 FSL）不属于这条路线。使用 DTI 确定性追踪、WMQL 束分析和 Python 连接组。解剖模型仍使用 PyTorch 和下载的权重；可选表面后处理保留原生组件。只有确认后才修改当前 PNL/CSD 等选择。这些算法不保证等价，请检查脑掩膜和追踪结果。',
    },
}


def texts(owner):
    return TEXT['zh' if getattr(owner, '_onboardingLanguage', 'en').startswith('zh') else 'en']


def native_route_values():
    return dict(brainMaskBackend='pnl_cnn',
                orchestrationTractographyBackend='dti_det', orchestrationBundleBackend='wmql',
                connectomeBackend='python')


def install(owner, page):
    import qt
    import ctk
    layout = page.layout()
    enabled = qt.QCheckBox()
    enabled.objectName = 'experimentsEnabledCheckBox'
    layout.addWidget(enabled)
    hint = qt.QLabel()
    hint.setWordWrap(True)
    layout.addWidget(hint)
    form = qt.QFormLayout()
    form.setRowWrapPolicy(qt.QFormLayout.WrapLongRows)
    form.setFieldGrowthPolicy(qt.QFormLayout.AllNonFixedFieldsGrow)
    layout.addLayout(form)
    controls = dict(enabled=enabled, hint=hint)
    for key, field in (('scope', qt.QComboBox()), ('root', ctk.ctkPathLineEdit()),
                       ('jobs', qt.QSpinBox()), ('options', qt.QPlainTextEdit()), ('variants', qt.QPlainTextEdit())):
        label = qt.QLabel()
        form.addRow(label, field)
        controls[key], controls[key + '_label'] = field, label
    controls['scope'].addItem('Current subject', 'subject')
    controls['scope'].addItem('Prepared subject folders', 'cohort')
    controls['root'].filters = ctk.ctkPathLineEdit.Dirs
    controls['jobs'].setRange(1, 8)
    for key in ('options', 'variants'):
        controls[key].setMinimumWidth(0)
        controls[key].setFixedHeight(75 if key == 'options' else 110)
    controls['options'].setPlaceholderText('--bundle-tractography-backend ukf --mrtrix-nthreads 8')
    controls['variants'].setPlaceholderText('[{"name":"baseline","args":[]}, {"name":"ukf","args":["--tractography-backend","ukf"]}]')
    row = qt.QHBoxLayout()
    for key in ('preview', 'export'):
        controls[key] = qt.QPushButton()
        row.addWidget(controls[key])
    layout.addLayout(row)
    controls['receipt'] = qt.QPlainTextEdit()
    controls['receipt'].setReadOnly(True)
    controls['receipt'].setFixedHeight(145)
    layout.addWidget(controls['receipt'])
    owner._experimentWidgets = controls
    for key, signal, attribute in (
        ('enabled', 'toggled(bool)', 'experimentsEnabled'),
        ('root', 'currentPathChanged(QString)', 'experimentRoot'),
        ('jobs', 'valueChanged(int)', 'experimentJobs'),
    ):
        controls[key].connect(signal, lambda value, name=attribute: owner._update_parameter_node(name, value))
    controls['scope'].currentIndexChanged.connect(lambda index: owner._update_parameter_node('experimentScope', str(controls['scope'].itemData(index))))
    for key, attribute in (('options', 'experimentCliOptions'), ('variants', 'experimentVariants')):
        controls[key].textChanged.connect(lambda name=attribute, field=controls[key]: owner._update_parameter_node(name, field.toPlainText()))
    controls['preview'].clicked.connect(lambda: preview(owner))
    controls['export'].clicked.connect(lambda: export_request(owner))
    lite = qt.QPushButton()
    lite.objectName = 'nativeDependencyRouteButton'
    owner.ui.environmentFormLayout.addRow(lite)
    owner._nativeDependencyRouteButton = lite
    lite.clicked.connect(lambda: apply_native_route(owner))
    sync(owner)
    apply_language(owner)


def apply_language(owner):
    widgets = getattr(owner, '_experimentWidgets', None)
    if not widgets:
        return
    text = texts(owner)
    for key in ('enabled', 'preview', 'export', 'hint'):
        widgets[key].setText(text[key])
    for key in ('scope', 'root', 'jobs', 'options', 'variants'):
        widgets[key + '_label'].setText(text[key])
    widgets['scope'].setItemText(0, text['subject'])
    widgets['scope'].setItemText(1, text['cohort'])
    owner._nativeDependencyRouteButton.setText(text['lite'])


def sync(owner):
    widgets = getattr(owner, '_experimentWidgets', None)
    node = getattr(owner, '_parameterNode', None)
    if not widgets or node is None:
        return
    for key, attribute in (('enabled', 'experimentsEnabled'), ('root', 'experimentRoot'),
                           ('jobs', 'experimentJobs'), ('options', 'experimentCliOptions'), ('variants', 'experimentVariants')):
        field, value = widgets[key], getattr(node, attribute)
        previous = field.blockSignals(True)
        try:
            if key == 'enabled':
                field.setChecked(value)
            elif key == 'root':
                if field.currentPath != value:
                    field.currentPath = value
            elif key == 'jobs':
                field.setValue(value)
            elif field.toPlainText() != value:
                field.setPlainText(value)
        finally:
            field.blockSignals(previous)
    widgets['scope'].blockSignals(True)
    widgets['scope'].setCurrentIndex(1 if node.experimentScope == 'cohort' else 0)
    widgets['scope'].blockSignals(False)
    for key in ('root', 'root_label', 'jobs', 'jobs_label'):
        widgets[key].setVisible(node.experimentScope == 'cohort')
    if hasattr(owner, '_fitAdvancedPage'):
        owner._fitAdvancedPage()


def snapshot_selection(owner):
    from shared.ui_run_controller import RunSelection, compress_mrtrix_dense_value
    node = owner._parameterNode
    freewater, noddi = owner._microstructure_stage_flags()
    topup = owner._topup_validation(batch_mode=node.experimentScope == 'cohort')
    if topup.errors:
        raise ValueError('; '.join(topup.errors))
    if node.experimentScope == 'cohort' and node.preprocessingReversePhaseB0.strip():
        raise ValueError('Cohort experiments cannot reuse a single reverse-PE image across subjects. Prepare subject-specific inputs first.')
    return RunSelection(subject_id=owner._effective_subject_id(), subject_root_text=node.inputDirectory.strip(),
        output_root_text=node.outputDirectory.strip(), selected_targets=owner._selected_product_targets(),
        run_name=node.orchestrationRunName, tractography_backend=owner._tractography_backend_text(),
        closure_tractography_backend=owner._closure_tractography_backend_text(),
        requested_targets=owner._requested_product_targets(),
        bundle_backend=owner._bundle_backend_text(), microstructure_freewater=freewater, microstructure_noddi=noddi,
        select_streamlines=node.orchestrationSelectStreamlines, ddseg_device=node.ddsegDevice, surface_device=node.device,
        connectome_backend=node.connectomeBackend, connectome_compress_mrtrix_dense=compress_mrtrix_dense_value(node.connectomeCompressMrtrixDenseState),
        brain_mask_backend=node.brainMaskBackend,
        brain_mask_device=node.brainMaskDevice, pnl_mask_route=getattr(node, "pnlMaskRoute", "auto") or "auto",
        microstructure_dki=owner._microstructure_dki_enabled(),
        microstructure_ifw=owner._microstructure_ifw_enabled(), ifw_echo_time_ms=owner._ifw_echo_time_ms(),
        dki_fit_method=node.orchestrationDkiFitMethod, wmql_query_preset=node.orchestrationWmqlPreset,
        wma_atlas=getattr(node, "wmaAtlas", "org"),
        wmql_query_selection=node.orchestrationWmqlSelection, max_parallel_models=node.maxParallelModels,
        ukf_parameters_text=node.ukfParameters, wma_input_tractography_text=node.wmaInputTractography.strip(), topup=topup.parameters)


def request_and_plan(owner):
    import sys
    from shared.ui_run_controller import build_single_subject_run_kwargs, build_background_cli_args
    node = owner._parameterNode
    if not owner._selected_product_targets():
        raise ValueError('Select at least one analysis product')
    selection = snapshot_selection(owner)
    external = owner._environment_dependency_path_kwargs()
    common = build_single_subject_run_kwargs(selection, external, preflight_blocked_targets={})
    root = Path(__file__).resolve().parents[1]
    args = build_background_cli_args(common, external, repo_root=root)[3:]
    managed = {'--subject-root': 1, '--subject-id': 1, '--output-root': 1, '--run-name': 1,
               '--execute': 0, '--print-manifest-only': 0}
    base, index = [], 0
    while index < len(args):
        flag = args[index]
        if flag in managed:
            index += managed[flag] + 1
        else:
            base.append(flag)
            index += 1
    output = node.outputDirectory.strip()
    if not output:
        source = node.experimentRoot.strip() if node.experimentScope == 'cohort' else node.inputDirectory.strip()
        if not source:
            raise ValueError('Set a subject/cohort directory and an output directory')
        output = str(Path(source) / 'outputs')
    request = make_request(base_args=base, subject_root=node.inputDirectory.strip(), subject_id=owner._effective_subject_id(),
        output_root=output, run_name=node.orchestrationRunName, scope=node.experimentScope,
        cohort_root=node.experimentRoot.strip(), jobs=node.experimentJobs,
        options=node.experimentCliOptions, variants=node.experimentVariants)
    plan = compile_plan(request, python_executable=sys.executable, pipeline_script=root / 'shared/run_subject_pipeline_cli.py')
    return request, plan, common, external


def preview(owner):
    try:
        request, plan, *_ = request_and_plan(owner)
        lines = [f'{len(plan["subjects"])} jobs | up to {plan["jobs"]} concurrent subjects',
                 plan['scheduling'], '', 'Output: ' + plan['output_root'], '']
        for job in plan['subjects'][:50]:
            config = job['effective_config']
            lines.extend([job['subject_id'] + ' / ' + job['variant'],
                          '  ' + ', '.join(config['targets']),
                          f'  Models: {",".join(config["microstructure"])} ({config["fit_method"]}); mask: {config["mask"]}',
                          f'  Tracks: {config["tractography"]}; bundles: {config["bundles"]}' +
                          (f' on {config["bundle_source"]}' if config['bundle_source'] else '')])
        if len(plan['subjects']) > 50:
            lines.append(f'... {len(plan["subjects"]) - 50} further jobs; export the request for the full plan.')
        if plan['skipped']:
            lines.extend(['', 'Skipped inputs:', json.dumps(plan['skipped'], ensure_ascii=False)])
        owner._experimentWidgets['receipt'].setPlainText('\n'.join(lines))
        return request, plan
    except Exception as exc:
        owner._experimentWidgets['receipt'].setPlainText(str(exc))
        owner._setStatus('error', str(exc))
        return None


def export_request(owner):
    import qt
    result = preview(owner)
    if result is None:
        return
    path = qt.QFileDialog.getSaveFileName(owner.parent, texts(owner)['export'], '', 'JSON (*.json)')
    if path:
        Path(path).write_text(json.dumps(result[0], indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def submit(owner):
    import slicer
    try:
        request, plan, common, external = request_and_plan(owner)
        question = f'Run {len(plan["subjects"])} experiment jobs with up to {plan["jobs"]} concurrent subjects?\n\n{plan["output_root"]}'
        if not slicer.util.confirmYesNoDisplay(question, parent=owner.parent):
            return
        # Keep the request beside the campaign; the worker exclusively creates
        # the output root and refuses reuse of an existing campaign directory.
        request_dir = Path(plan['output_root']).parent
        request_dir.mkdir(parents=True, exist_ok=True)
        path = request_dir / (Path(plan['output_root']).name + '.request.json')
        with path.open('x', encoding='utf-8') as stream:
            json.dump(request, stream, indent=2, ensure_ascii=False)
        script = Path(__file__).resolve().parents[1] / 'scripts/run_experiments.py'
        owner._pendingRunManifestPath = Path(plan['manifest_path'])
        from shared.ui_run_history import save_history_directory
        save_history_directory(plan['output_root'])
        owner._save_common_user_settings()
        owner._startBackgroundPipeline(common, external, owner._selected_workflows(),
            launch_args=['--launch', 'PythonSlicer', str(script), '--request', str(path), '--execute'],
            run_context=dict(experiment_manifest_path=plan['manifest_path'], output_root=Path(plan['output_root'])))
    except Exception as exc:
        owner._advancedSettingsPanel.collapsed = False
        owner._advancedCategorySelector.setCurrentIndex(4)
        owner._setStatus('error', str(exc))
        owner._experimentWidgets['receipt'].setPlainText(str(exc))


def apply_native_route(owner):
    import slicer
    if getattr(owner, '_pipelineActive', False) or getattr(owner, '_selectedSetupActive', False):
        return
    text = texts(owner)
    if not slicer.util.confirmYesNoDisplay(text['lite_message'], windowTitle=text['lite_title'], parent=owner.parent):
        return
    node = owner._parameterNode
    with slicer.util.NodeModify(node):
        for key, value in native_route_values().items():
            setattr(node, key, value)
        if node.orchestrationDkiFitMethod == 'MKCURVE_MATLAB':
            node.orchestrationDkiFitMethod = 'MKCURVE'
    owner._lastEnvironmentResult = None
    owner._lastEnvironmentSignature = None
    owner.appendLog('[Environment] Slicer-native route selected explicitly. Review the mask and tracking outputs.')
