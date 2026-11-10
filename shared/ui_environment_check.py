"""One responsive check entry for the Check and Prepare GUI actions."""

from datetime import datetime
import json
from pathlib import Path

from shared.ui_selected_setup import _text


def start_environment_check(owner, on_complete=None, *, on_failure=None):
    import slicer
    from shared.environment_check import check_config, result_from_dict
    from shared.ui_setup_jobs import start_setup_job

    if owner._pipelineActive or getattr(owner, '_setupJob', None) or getattr(owner, '_selectedSetupActive', False):
        owner._setStatus('warning', _text(owner, 'Wait for the active job or cancel it first.', '请先等待当前任务结束，或取消任务。'))
        return False
    owner._lastEnvironmentResult = owner._lastEnvironmentSignature = None
    owner._update_environment_result_label(None)
    options = owner._environment_check_kwargs()
    signature = owner._environment_check_signature(options)
    folder = owner._environment_report_dir_from_kwargs(options)
    if folder is None:
        message = _text(owner, 'Cannot write the check report; choose a writable output directory.',
                        '无法写入检查报告，请选择可写的输出目录。')
        owner._setStatus('error', message)
        owner.appendLog('[Check] ' + message)
        if on_failure:
            on_failure('failed', message)
        return False
    stem = 'check-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    request, result_path = folder / (stem + '-request.json'), folder / (stem + '-result.json')
    modules = tuple(name for name in ('dwiconvert', 'ukftractography') if hasattr(slicer.modules, name))
    config = check_config(options, repo_root=owner.logic.module_dir, output_dir=folder, loaded_modules=modules)
    try:
        folder.mkdir(parents=True, exist_ok=True)
        request.write_text(json.dumps(config, default=str))
    except (OSError, ValueError) as exc:
        owner._setStatus('error', _text(owner, 'Cannot write the check report; choose a writable output directory.',
                                      '无法写入检查报告，请选择可写的输出目录。'))
        owner.appendLog('[Check] ' + str(exc))
        return False
    owner._latestEnvironmentReportDir = folder
    script = Path(__file__).resolve().parents[1] / 'scripts/check_environment.py'

    def done(status, details, log_path):
        if status != 'completed':
            owner._update_environment_result_label(None, error=None if status == 'cancelled' else 'Check failed')
            owner._setStatus('warning', _text(owner, 'Check cancelled.' if status == 'cancelled' else 'Check failed; open the install log for details.',
                                            '已取消检查。' if status == 'cancelled' else '检查失败，请查看日志。'))
            if on_failure:
                on_failure(status, details)
            return
        if signature != owner._environment_check_signature(owner._environment_check_kwargs()):
            owner._lastEnvironmentResult = owner._lastEnvironmentSignature = None
            owner._setStatus('warning', _text(owner, 'Selection changed during the check. Check again.', '检查期间配置已改变，请重新检查。'))
            if on_failure:
                on_failure('stale', 'Configuration changed during the final check')
            return
        try:
            report = result_from_dict(json.loads(result_path.read_text()))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            owner._setStatus('error', _text(owner, 'Could not read the environment report.', '无法读取环境检查报告。'))
            owner.appendLog('[Check] ' + str(exc))
            if on_failure:
                on_failure('failed', str(exc))
            return
        owner._cache_environment_result(report, signature)
        owner._update_environment_result_label(report)
        owner._update_environment_path_status_labels(report)
        owner._update_environment_status(report)
        if on_complete:
            on_complete(report)

    args = ['--launch', 'PythonSlicer', '-u', str(script), '--request', str(request), '--result', str(result_path)]
    return start_setup_job(owner, _text(owner, 'Checking environment', '检查环境'),
                           [(owner._slicer_launcher_program(), args, {})], done)


def request_prepare_environment(owner, *, repair=False):
    from shared.ui_selected_setup import prepare_selected_environment
    return start_environment_check(owner, lambda report: prepare_selected_environment(owner, report, repair=repair))
