"""Explicit, background provisioning from the compact Slicer environment UI."""

from datetime import datetime
import json
from pathlib import Path

from shared.managed_runtimes import FSL_TERMS_URL, cuda_candidate, install_plan, runtime_root, selected_profiles, supported_platform


def asset_profiles_for_items(items):
    targets = {target for item in items for target in getattr(item, "targets", ())}
    return tuple(name for name in ("surface", "segmentation", "parcellation") if name in targets) or ("core",)


def install_selected_runtimes(owner, profiles=None):
    import qt
    import slicer
    from shared.ui_setup_jobs import start_setup_job

    if getattr(owner, "_setupJob", None) or getattr(owner, "_pipelineActive", False):
        slicer.util.infoDisplay("Wait for the current job or cancel it before changing setup.", parent=owner.parent)
        return False
    language = str(getattr(owner, "_onboardingLanguage", "en")).startswith("zh")
    gpu = cuda_candidate()
    profiles = tuple(profiles or selected_profiles(owner._parameterNode, gpu))
    if not profiles:
        slicer.util.infoDisplay("当前所选路线不需要 MRtrix/FSL。" if language else
                               "The selected routes do not need MRtrix/FSL.", parent=owner.parent)
        return False
    if not supported_platform():
        slicer.util.infoDisplay("自动配置暂支持 Linux x86_64；其他平台保留高级路径配置。" if language else
                               "Managed setup currently supports Linux x86_64; use Advanced paths on other platforms.", parent=owner.parent)
        return False
    plan = install_plan(profiles)
    dialog = qt.QDialog(slicer.util.mainWindow())
    dialog.setObjectName("managedRuntimeConfirmationDialog")
    dialog.setWindowTitle("配置所选计算工具" if language else "Set up selected computation tools")
    layout = qt.QVBoxLayout(dialog)
    label = qt.QLabel(("将下载到用户目录，成功后自动填写路径。不会修改 Slicer Python、驱动或系统配置。\n" if language else
                      "Downloads go to your user directory; verified paths are saved automatically.\n"
                      "Slicer Python, GPU drivers and shell configuration are not changed.\n") +
                     "\n".join(plan["profiles"]) + "\n" + plan["root"])
    label.wordWrap = True
    layout.addWidget(label)
    gpu_box = None
    if plan["fsl_license_required"] and gpu:
        gpu_box = qt.QCheckBox("安装 GPU eddy（另保留 CPU 可执行文件）" if language else
                              "Install GPU eddy (also retains the CPU executable)")
        gpu_box.checked = "fsl_cuda" in profiles
        layout.addWidget(gpu_box)
    accepted = None
    if plan["fsl_license_required"]:
        terms = qt.QLabel(("FSL 的多数组件仅限非商业用途：" if language else
                          "Most FSL components are licensed for non-commercial use: ") +
                         f'<a href="{FSL_TERMS_URL}">FSL license</a>')
        terms.wordWrap = True
        terms.openExternalLinks = True
        layout.addWidget(terms)
        accepted = qt.QCheckBox("我已阅读并接受 FSL 许可，且本次使用符合许可。" if language else
                               "I have read and accept the FSL license; my use complies with its terms.")
        accepted.setObjectName("managedRuntimeFslAcceptance")
        layout.addWidget(accepted)
    buttons = qt.QDialogButtonBox(dialog)
    buttons.setStandardButtons(qt.QDialogButtonBox.Ok | qt.QDialogButtonBox.Cancel)
    buttons.setObjectName("managedRuntimeDialogButtons")
    buttons.button(qt.QDialogButtonBox.Ok).setText("安装并自动配置" if language else "Install and configure")
    buttons.button(qt.QDialogButtonBox.Ok).enabled = accepted is None
    if accepted is not None:
        accepted.connect("toggled(bool)", lambda value: setattr(buttons.button(qt.QDialogButtonBox.Ok), "enabled", value))
    buttons.connect("accepted()", dialog.accept)
    buttons.connect("rejected()", dialog.reject)
    layout.addWidget(buttons)
    if dialog.exec_() != qt.QDialog.Accepted:
        dialog.deleteLater()
        return False
    if gpu_box is not None:
        profiles = tuple(("fsl_cuda" if gpu_box.checked else "fsl_cpu") if name.startswith("fsl_") else name
                         for name in plan["profiles"])
    result_path = runtime_root() / ("setup-result-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".json")
    script = Path(__file__).resolve().parents[1] / "scripts" / "setup_managed_runtime.py"
    arguments = ["--launch", "PythonSlicer", "-u", str(script), "--profiles", *profiles,
                 "--execute", "--result", str(result_path)]
    conda = owner._environment_dependency_path_kwargs().get("conda_executable")
    if conda:
        arguments.extend(["--conda", str(conda)])
    if accepted is not None and accepted.checked:
        arguments.append("--accept-fsl-license")
    dialog.deleteLater()

    def done(status, details, log_path):
        if status == "completed":
            try:
                apply_runtime_result(owner, json.loads(result_path.read_text()))
                owner._setStatus("ready", "工具已配置；运行前重新检查依赖。" if language else
                                 "Tools configured; check dependencies before running.")
            except Exception as exc:
                owner._setStatus("warning", str(exc))
                owner.appendLog("[Setup] Runtime result could not be applied: " + str(exc))
        elif status == "failed":
            owner._show_actionable_error(title="External tool setup failed",
                                         what="Managed runtime setup did not finish.",
                                         why="The download, environment solve or runtime verification failed.",
                                         next_step="Inspect the log and retry setup; existing configured paths remain unchanged.",
                                         auto_fix="No system or Slicer Python packages were modified.", details=details + "\nLog: " + str(log_path))
        else:
            owner.appendLog("[Setup] Cancelled; configured paths were not changed. Retry the same setup to complete it.")

    return start_setup_job(owner, "配置计算工具" if language else "Set up computation tools",
                           [(owner._slicer_launcher_program(), arguments, {})], done)


def apply_runtime_result(owner, result, step="managed_runtime"):
    """Save only settings from a successful, verified setup receipt (validation: selected_setup.apply_setup_receipt)."""
    from shared.selected_setup import apply_setup_receipt

    apply_setup_settings(owner, apply_setup_receipt({"id": step}, result))


def apply_setup_settings(owner, settings):
    """GUI side of a saved receipt: Slicer settings, line edits, log and a fresh environment check."""
    for key, value in settings.items():
        owner._save_environment_path_setting(key, str(value))
        edit = (getattr(owner, "_environmentPathLineEdits", {}) or {}).get(key)
        if edit is not None:
            edit.setText(str(value))
        owner.appendLog(f"[Setup] Automatically configured {key}: {value}")
    owner._lastEnvironmentResult = None
    owner._lastEnvironmentSignature = None
