"""Surface-only activation assistance; copies imported user license to private user config."""

import os

from shared.surface_license import (
    REGISTRATION_URL,
    import_surface_license,
    managed_surface_license_path,
    resolve_surface_license,
    usable_license,
)


def choose_license(owner):
    import qt
    import slicer
    from shared.ui_selected_setup import _text

    dialog = qt.QDialog(slicer.util.mainWindow())
    dialog.setObjectName('surfaceLicenseDialog')
    dialog.setWindowTitle(_text(owner, 'Surface postprocessing license', '表面后处理许可证'))
    layout = qt.QVBoxLayout(dialog)
    label = qt.QLabel(_text(owner,
        'Import your FreeSurfer license.txt once, or register for a free license. SlicerDSpace copies it to '
        'your private user configuration and sets FS_LICENSE automatically for future runs and upgrades. '
        'Other analyses do not require this file.',
        '首次导入您自己的 FreeSurfer license.txt，或免费注册获取。SlicerDSpace 会将其保存至用户私有配置目录'
        '并自动配置 FS_LICENSE，后续启动与软件升级均可自动识别；其他分析不需要此文件。'))
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
    """Discover or import once for selected surface postprocessing; configure FS_LICENSE and refresh."""
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
        os.environ.setdefault('FS_LICENSE', str(path))
        return result
    if not usable_license(path):
        managed = managed_surface_license_path()
        if usable_license(managed):
            path = managed.resolve()
        else:
            chosen = choose_license(owner)
            if chosen is None:
                return None
            path = import_surface_license(chosen)
    os.environ['FS_LICENSE'] = str(path)
    # Recheck even when an explicit file has been fixed since the previous report.
    edit = (getattr(owner, '_environmentPathLineEdits', {}) or {}).get('postprocess_license_file')
    if edit is not None:
        edit.setText(str(path))
    owner._save_environment_path_setting('postprocess_license_file', str(path))
    return owner._run_environment_dependency_check()
