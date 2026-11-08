"""Surface-only activation assistance; never download or copy private keys."""

from shared.surface_license import REGISTRATION_URL, resolve_surface_license, usable_license


def choose_license(owner):
    import qt
    import slicer
    from shared.ui_selected_setup import _text

    dialog = qt.QDialog(slicer.util.mainWindow())
    dialog.setObjectName('surfaceLicenseDialog')
    dialog.setWindowTitle(_text(owner, 'Surface postprocessing license', '表面后处理许可证'))
    layout = qt.QVBoxLayout(dialog)
    label = qt.QLabel(_text(owner,
        'Import your FreeSurfer license.txt, or register for a free license. The path is saved for later runs. '
        'Other analyses do not require this file.',
        '请选择您自己的 FreeSurfer license.txt，或免费申请。路径保存后可重复使用；其他分析不需要此文件。'))
    label.wordWrap = True
    layout.addWidget(label)
    selected = []
    register = qt.QPushButton(_text(owner, 'Get a free license', '免费申请'))
    register.setObjectName('surfaceLicenseRegisterButton')
    register.connect('clicked()', lambda: qt.QDesktopServices.openUrl(qt.QUrl(REGISTRATION_URL)))
    layout.addWidget(register)
    button = qt.QPushButton(_text(owner, 'Import license', '导入许可证'))
    button.setObjectName('surfaceLicenseImportButton')

    def browse():
        path = qt.QFileDialog.getOpenFileName(dialog, _text(owner, 'Select your license', '选择个人许可证'),
                                             '', 'License files (*.txt .license);;All files (*)')
        if not path:
            return
        if not usable_license(path):
            slicer.util.warningDisplay(_text(owner, 'Select a readable, nonempty text license file.',
                                             '请选择可读取的非空文本许可证文件。'))
            return
        selected.append(path)
        dialog.accept()

    button.connect('clicked()', browse)
    layout.addWidget(button)
    cancel = qt.QPushButton(_text(owner, 'Cancel', '取消'))
    cancel.connect('clicked()', dialog.reject)
    layout.addWidget(cancel)
    dialog.exec_()
    dialog.deleteLater()
    return selected[0] if selected else None


def prepare_surface_license(owner, result):
    """Discover or ask only for selected surface postprocessing; refresh its report."""
    if 'surface_postprocess' not in result.targets:
        return result
    config = owner._environment_dependency_path_kwargs()
    explicit = config.get('postprocess_license_file')
    from shared.surface_runtime import resolve_surface_runtime
    try:
        _, runtime_home = resolve_surface_runtime(config.get('postprocess_runtime'),
                                                   config.get('postprocess_freesurfer_home'))
    except ValueError:
        runtime_home = None
    path = resolve_surface_license(explicit, freesurfer_home=runtime_home)
    missing = any(item.key == 'surface_postprocess_license' and item.status != 'available'
                  for item in result.statuses)
    if usable_license(path) and not missing and explicit:
        return result
    if not usable_license(path):
        path = choose_license(owner)
        if path is None:
            return None
    # Recheck even when an explicit file has been fixed since the previous report.
    edit = (getattr(owner, '_environmentPathLineEdits', {}) or {}).get('postprocess_license_file')
    if edit is not None:
        edit.setText(str(path))
    owner._save_environment_path_setting('postprocess_license_file', str(path))
    return owner._run_environment_dependency_check()
