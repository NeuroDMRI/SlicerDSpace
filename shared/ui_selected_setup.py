"""One explicit preparation action, with resumable phases and truthful status."""

from datetime import datetime
import json
from pathlib import Path


def python_setup_commands(owner, packages, *, requests=None):
    """GUI wrapper of the pure selected_setup.python_package_commands (B7)."""
    from shared.selected_setup import python_package_commands
    from shared.user_config import default_user_config_path

    folder = Path(getattr(owner, "_latestEnvironmentReportDir", None) or default_user_config_path().parent / "setup-logs")
    result = getattr(owner, "_lastEnvironmentResult", None)
    commands, receipts = python_package_commands(
        packages, requests=requests, python_command=owner._setup_python_command, folder=folder,
        statuses=getattr(result, "statuses", ()),
        tract_querier_root=lambda: owner._environment_dependency_path_kwargs().get("tract_querier_root"))
    from shared.backend_sources import default_darwin_arm64_tractcloud_python, ensure_darwin_arm64_tractcloud_python, needs_darwin_arm64_tractcloud_python
    if needs_darwin_arm64_tractcloud_python():
        for program, _args, _env in commands:
            if Path(str(program)).expanduser().absolute() == default_darwin_arm64_tractcloud_python():
                ensure_darwin_arm64_tractcloud_python()
    return commands, receipts


def _text(owner, english, chinese):
    return chinese if str(getattr(owner, "_onboardingLanguage", "en")).startswith("zh") else english


def component_rows(owner, components, pending):
    """HTML rows of the licence dialog: size, licence with link and local file, and each Python package with its file."""
    from scripts.download_model_assets import _format_bytes
    import html

    rows = []
    for item in components:
        size = _format_bytes(item["size_bytes"]) if item["size_bytes"] else _text(owner, "size not known", "大小未知")
        row = "<b>%s</b> (%s)" % (html.escape(item["title"]), size)
        licence = item["licence"]
        if licence:
            local = ('<a href="%s">%s</a>' % (Path(licence["file"]).as_uri(), html.escape(Path(licence["file"]).name))
                     if licence["file"] else _text(owner, "none", "无"))
            row += "<br>%s: <a href=\"%s\">%s</a> | %s: %s%s" % (
                _text(owner, "Licence", "许可"), html.escape(licence["url"]), html.escape(licence["name"]),
                _text(owner, "local file", "本地文件"), local,
                "" if licence["id"] in pending else _text(owner, " (already accepted)", "（已接受）"))
        for package in (licence or {}).get("packages", ()):  # S3-2: each package with its licence, link and local file (A3 NB1)
            package_file = ('<a href="%s">%s</a>' % (Path(package["file"]).as_uri(), html.escape(Path(package["file"]).name))
                            if package.get("file") else _text(owner, "no local file", "无本地文件"))
            row += '<br>&nbsp;&nbsp;<a href="%s">%s</a>: %s | %s' % (
                html.escape(package["url"]), html.escape(package["name"]), html.escape(package["license"]), package_file)
        rows.append(row)
    return rows


def _compact_component_rows(owner, components, pending):
    """Concise, non-repetitive HTML bullet lines for the GUI confirmation dialog."""
    from scripts.download_model_assets import _format_bytes
    import html

    rows = []
    for item in components:
        size_suffix = f" ({_format_bytes(item['size_bytes'])})" if item.get("size_bytes") else ""
        row = f"• <b>{html.escape(item['title'])}</b>{size_suffix}"
        licence = item.get("licence")
        if licence:
            status_tag = "" if licence["id"] in pending else _text(owner, " (accepted)", "（已接受）")
            row += (
                f'<br>&nbsp;&nbsp;&nbsp;<span style="font-size:12px">'
                f'{_text(owner, "License", "许可")}: '
                f'<a href="{html.escape(licence["url"])}">{html.escape(licence["name"])}</a>{status_tag}'
                f"</span>"
            )
        rows.append(row)
    return rows


def extension_states(manager, modules):
    """Use exposed APIs and actual capabilities, not enabled == loaded.

    Slicer 5.10's C++ isExtensionLoaded method is not exposed to PythonQt.
    An enabled extension may still need restart before its modules exist.
    """
    import importlib.util
    installed, loaded = [], []
    for name, module in (("PyTorch", "pytorchutils"),
                         ("UKFTractography", "ukftractography")):
        available = hasattr(modules, module)
        if name == 'PyTorch' and not available:
            try:
                available = importlib.util.find_spec('PyTorchUtils') is not None
            except (ImportError, ValueError):
                available = False
        if available:
            loaded.append(name)
        if available or (manager is not None and manager.isExtensionInstalled(name)):
            installed.append(name)
    return installed, loaded


def _confirm(owner, plan):
    import html
    import qt
    import slicer
    from scripts.download_model_assets import _format_bytes
    from shared.managed_pnl import PNL_MODEL_TERMS_URL, PNL_TEMPLATE_TERMS_URL
    from shared.setup_components import pending_licences, plan_components, required_licences
    from shared.setup_resume import accepted_licences

    dialog = qt.QDialog(slicer.util.mainWindow())
    dialog.setObjectName("selectedSetupConfirmationDialog")
    dialog.setProperty("popupCard", True)
    dialog.setWindowTitle(_text(owner, "Prepare selected analysis", "准备所选分析环境"))
    try:
        from shared.ui_styles import ui_stylesheet

        dialog.setStyleSheet(ui_stylesheet(widget=getattr(owner, "_uiWidget", None)))
    except Exception:
        pass
    layout = qt.QVBoxLayout(dialog)
    layout.setContentsMargins(14, 12, 14, 12)
    layout.setSpacing(10)

    components = plan_components(plan)
    required = required_licences(components)
    pending = pending_licences(required, accepted_licences())

    header = _text(
        owner,
        "Missing items to prepare (existing installations are reused):",
        "将为当前分析准备以下缺失项（已有环境直接复用）：",
    )
    html_blocks = [f"<b>{html.escape(header)}</b>"]
    if plan.get("repair"):
        html_blocks.append(
            html.escape(
                _text(
                    owner,
                    "Reinstalling selected plugin-managed tools.",
                    "重新安装所选的插件受管工具。",
                )
            )
        )
    if plan.get("also_runs"):
        html_blocks.append(html.escape(str(plan["also_runs"])))

    bullet_rows = []
    if plan.get("extensions"):
        bullet_rows.append(
            f"• <b>{_text(owner, 'Slicer extensions', 'Slicer 扩展')}:</b> {html.escape(', '.join(plan['extensions']))}"
        )
    if plan.get("restart_extensions"):
        bullet_rows.append(
            f"• <b>{_text(owner, 'Enable & restart extensions', '启用扩展（需重启）')}:</b> "
            f"{html.escape(', '.join(plan['restart_extensions']))}"
        )
    bullet_rows.extend(_compact_component_rows(owner, components, pending))
    if plan.get("unsupported_profiles"):
        bullet_rows.append(
            f"• <b>{_text(owner, 'Unsupported profiles', '当前平台不支持')}:</b> "
            f"{html.escape(', '.join(plan['unsupported_profiles']))}"
        )
    if bullet_rows:
        html_blocks.append("<br>".join(bullet_rows))

    notes = []
    for note in plan.get("unsupported_notes", {}).values():
        notes.append(html.escape(str(note)))
    if plan.get("asset_profiles") or plan.get("optional_asset_profiles"):
        notes.append(
            html.escape(_text(owner, "Model directory: ", "模型目录：") + str(plan.get("asset_root", "")))
        )
    if plan.get("optional_asset_profiles") and plan.get("optional_asset_peak_disk_bytes", 0) > plan.get("optional_asset_size_bytes", 0):
        notes.append(
            html.escape(
                _text(owner, "Peak disk space needed: ", "预计峰值磁盘空间：")
                + _format_bytes(plan["optional_asset_peak_disk_bytes"])
            )
        )
    if plan.get("manual_items"):
        notes.append(
            html.escape(
                _text(owner, "Needs attention: ", "仍需手动处理：")
                + ", ".join(item["label"] for item in plan["manual_items"])
            )
        )
    if plan.get("surface_build_required") or plan.get("surface_prebuilt_required"):
        notes.append(
            html.escape(
                _text(
                    owner,
                    "Surface postprocessing: install the qualified precompiled components in your user cache. No Git, compiler or full FreeSurfer installation is needed. Your own activation file is still required for analysis.",
                    "表面后处理：安装经校验的预编译组件，无需编译器或完整 FreeSurfer；运行分析仍需个人激活文件。",
                )
            )
        )
    if plan.get("surface_bundle_unavailable"):
        notes.append(
            html.escape(
                _text(
                    owner,
                    "Surface components bundle is absent locally; supply the offline bundle or configure an existing runtime.",
                    "本地缺少表面处理组件包；请导入离线包或配置已有运行目录。",
                )
            )
        )
    if plan.get("pnl_terms_required"):
        notes.append(
            f'{html.escape(_text(owner, "Terms:", "许可条款："))} '
            f'<a href="{PNL_MODEL_TERMS_URL}">PNL models</a> | '
            f'<a href="{PNL_TEMPLATE_TERMS_URL}">IIT template</a>'
        )
    if notes:
        html_blocks.append("<br>".join(notes))

    body_label = qt.QLabel("<br><br>".join(html_blocks))
    body_label.wordWrap = True
    body_label.openExternalLinks = True
    if len(bullet_rows) + len(notes) > 6:
        scroll = qt.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body_label)
        scroll.setMinimumHeight(140)
        scroll.setMaximumHeight(320)
        layout.addWidget(scroll)
    else:
        layout.addWidget(body_label)
    dialog.setMinimumWidth(440)
    dialog.setMaximumWidth(560)
    buttons = qt.QDialogButtonBox(dialog)
    buttons.setStandardButtons(qt.QDialogButtonBox.Ok | qt.QDialogButtonBox.Cancel)
    buttons.button(qt.QDialogButtonBox.Ok).setText(
        _text(owner, "Accept licences and install", "接受许可并安装") if pending else _text(owner, "Prepare", "开始准备"))
    buttons.connect("accepted()", dialog.accept)
    buttons.connect("rejected()", dialog.reject)
    layout.addWidget(buttons)
    accepted = dialog.exec_() == qt.QDialog.Accepted
    dialog.deleteLater()
    if accepted and pending:
        from shared.setup_components import accept_licences
        # id, version and time, in the user config. Version: SHA-256 of the local licence file;
        # for a Python package group: SHA-256 of its sorted (package, licence name) list (A3 NB2).
        accept_licences(pending)
    return accepted


def _running_slicer_executable(owner=None):
    """Launcher of the running Slicer (for the Windows installer's -SlicerExecutable), or None.

    S-B9: the launcher first (on Windows applicationFilePath is bin\\SlicerApp-real.exe, which the ps1 cannot use to
    find PythonSlicer), then the running application path.
    """
    launcher = getattr(owner, "_slicer_launcher_program", None)
    try:
        value = launcher() if callable(launcher) else None
        if isinstance(value, str) and value.strip():
            return value.strip()
    except Exception:
        pass
    try:
        import slicer
        value = slicer.app.applicationFilePath
        value = value() if callable(value) else value
        return str(value) if value else None
    except Exception:
        return None


def selected_setup_plan(owner, result, *, repair=False):
    """Same selected-route plan for the confirmation dialog and Agent previews."""
    import slicer
    from shared.managed_runtimes import configured_profiles, cuda_candidate, selected_profiles
    from shared.selected_setup import build_setup_plan
    manager = slicer.app.extensionsManagerModel()
    installed, loaded = extension_states(manager, slicer.modules)
    # C2/K1: everything below follows the one closure the doctor and the run use (no second expansion).
    from shared.target_closure import Closure, closure_notice
    closure = owner._selected_closure()
    fallback_ukf = False
    if closure is None:
        # N3: no parameter node. The doctor result's targets are already closed, so take them
        # as they are (no second expansion); the default algorithm set applies.
        closure = Closure(targets=tuple(result.targets), tractography_algorithms=("csd_act",),
                          requested=tuple(result.targets))
        # N3-UKF: the algorithms are unknown here, so the doctor result decides. A missing required
        # UKF dependency item means the doctor checked the UKF route; keep it in the plan.
        from shared.ui_environment_panel import environment_missing_items
        fallback_ukf = any(item.required and item.install_group == "ukf" for item in environment_missing_items(result))
    targets = set(closure.targets)
    gpu = cuda_candidate()
    candidates = list(selected_profiles(owner._parameterNode, gpu))
    from shared.brain_mask import resolve_pnl_mask_route
    pnl_route = getattr(owner._parameterNode, "pnlMaskRoute", "auto") or "auto"
    if (getattr(owner._parameterNode, "brainMaskBackend", "pnl_cnn") == "pnl_cnn" and
            resolve_pnl_mask_route(pnl_route) == "managed" and  # SC-1c: the managed profile is for the managed route
            any(item.key == "brain_mask_backend" and item.required for item in result.statuses)):
        candidates.append("pnl_cnn")
    config = owner._environment_dependency_path_kwargs()
    existing = configured_profiles(config, candidates) if candidates else ()
    from shared.selected_setup import need_flags
    from scripts.setup_python_packages import dcm2niix_pending
    use_native = not any(
        getattr(item, "key", None) == "mrtrix" and getattr(item, "kind", None) == "path"
        for item in getattr(result, "statuses", ())
    )
    plan = build_setup_plan(result, installed_extensions=installed, loaded_extensions=loaded,
        **need_flags(targets, closure.tractography_algorithms, fallback_ukf=fallback_ukf),
        brain_mask_backend=getattr(owner._parameterNode, "brainMaskBackend", "pnl_cnn"), gpu_available=gpu,
        pnl_mask_route=pnl_route,
        wma_atlas=getattr(owner._parameterNode, "wmaAtlas", "org"),
        existing_managed_profiles=existing, repair=repair, backend_paths=config, closure_targets=tuple(closure.targets),
        slicer_executable=_running_slicer_executable(owner) or config.get("slicer_executable"),
        dcm2niix_missing=dcm2niix_pending(),  # S-B3: checked in the running Slicer's own Python
        use_native_runtime=use_native)
    # C4: the result's targets are already closed, so the owner (which knows what was ticked) words the notice.
    notice = getattr(owner, "_closure_notice_text", None)
    notice = notice() if callable(notice) else ""
    plan["also_runs"] = notice if isinstance(notice, str) else closure_notice(closure)
    plan["closure_targets"] = list(closure.targets)
    return plan


def prepare_selected_environment(owner, result=None, *, repair=False):
    import slicer
    from shared.selected_setup import complete_step
    from shared.ui_managed_runtimes import apply_setup_settings
    from shared.managed_runtimes import write_json
    from shared.user_config import default_user_config_path
    from shared.setup_resume import pending_setup

    pending_before = pending_setup()
    if (getattr(owner, "_setupJob", None) or getattr(owner, "_pipelineActive", False)
            or getattr(owner, "_selectedSetupActive", False)):
        slicer.util.infoDisplay(_text(owner, "Wait for the active job or cancel it first.", "请先等待当前任务结束，或取消当前任务。"))
        return False
    result = result or owner._run_environment_dependency_check()
    if result is None:
        return False
    from shared.ui_surface_license import prepare_surface_license
    result = prepare_surface_license(owner, result)
    if result is None:
        return False
    manager = slicer.app.extensionsManagerModel()
    config = owner._environment_dependency_path_kwargs()
    plan = selected_setup_plan(owner, result, repair=repair)
    # S-B1: the Windows ps1 step is pending work too; without it the dialog (and the licence acceptance) was skipped.
    pending_actions = bool((plan.get('windows_runtime') or {}).get('command')) or plan.get('surface_build_required', False) or plan.get('surface_prebuilt_required', False) or any(plan.get(key) for key in (
        "extensions", "restart_extensions", "python_packages", "asset_profiles", "optional_asset_profiles",
        "managed_profiles", "native_bundles", "unsupported_profiles", "manual_items", "backend_profiles"))
    if pending_actions and not _confirm(owner, plan):
        return False
    owner._selectedSetupActive = True
    folder = default_user_config_path().parent / "setup-history" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    try:
        folder.mkdir(parents=True, exist_ok=True)
        owner._latestEnvironmentReportDir = folder
        write_json(folder / "plan.json", plan)
        save_choices = getattr(owner, '_save_common_user_settings', None)
        if callable(save_choices):
            save_choices()
        if pending_actions:
            from shared.setup_resume import remember_setup, refresh_setup_button
            remember_setup(folder.name)
            refresh_setup_button(owner)
    except OSError as exc:
        owner._selectedSetupActive = False
        owner._setStatus("warning", str(exc))
        return False
    # Let Slicer's own manager select compatible extension builds. Installing
    # an extension does not make its modules available in the running session.
    installed_any_extension = False
    for name in plan["extensions"]:
        try:
            success = manager is not None and manager.installExtensionFromServer(name, False)
        except Exception as exc:
            owner.appendLog("[Setup] Extension install failed: " + str(exc))
            success = False
        if not success:
            if name == "PyTorch" and any(str(pkg).startswith("torch") for pkg in plan.get("python_packages", ())):
                owner.appendLog(
                    "[Setup] Slicer PyTorch extension could not be downloaded; continuing with direct Python torch installation."
                )
                continue
            owner._selectedSetupActive = False
            owner._setStatus("warning", _text(owner, "Extension installation did not finish: ", "扩展安装未完成：") + name)
            write_json(folder / "result.json", {"status": "blocked", "extension": name})
            return False
        installed_any_extension = True
    for name in plan["restart_extensions"]:
        try:
            manager.setExtensionEnabled(name, True)
        except Exception as exc:
            owner._selectedSetupActive = False
            write_json(folder / "result.json", {"status": "blocked", "extension": name, "error": str(exc)})
            owner._setStatus("warning", str(exc))
            return False
    if installed_any_extension or plan["restart_extensions"]:
        owner._selectedSetupActive = False
        owner._setupRestartRequired = True
        owner._setStatus("warning", _text(owner, "Restart Slicer; preparation will be offered when you reopen SlicerDSpace.", "请重启 Slicer；重新进入 SlicerDSpace 后将提示继续准备。"))
        write_json(folder / "result.json", {"status": "restart_required", "extensions": plan["extensions"] + plan["restart_extensions"]})
        slicer.util.infoDisplay(_text(owner, "Required extensions were installed. Save your work and restart Slicer. Reopen SlicerDSpace to continue preparation with the installed extensions.",
                                      "所需扩展已安装。请保存当前工作并重启 Slicer，重新进入 SlicerDSpace 后继续准备；已安装扩展会被复用。"))
        return True

    def finish(status="completed", details=""):
        if status != "completed":
            owner._selectedSetupActive = False
            owner._latestEnvironmentReportDir = folder
            write_json(folder / "result.json", {"status": status, "details": details})
            owner._setStatus("warning", _text(owner, "Preparation stopped; inspect setup logs.", "环境准备已停止，请查看安装日志。"))
            return
        owner._selectedSetupActive = False
        if not pending_actions:
            functional_check(result)
            return
        from shared.ui_environment_check import start_environment_check
        def check_failed(state, details):
            write_json(folder / "result.json", {"status": state, "details": details})
        if not start_environment_check(owner, functional_check, on_failure=check_failed):
            write_json(folder / "result.json", {"status": "blocked", "details": "Final environment check could not start"})

    def functional_check(current):
        if current is None or current.blocker_count or getattr(owner, '_setupRestartRequired', False):
            record_readiness(current)
            return
        from shared.setup_selftest import requests_for_report
        from shared.ui_setup_jobs import start_setup_job
        def failed(details):
            from shared.setup_resume import remember_functional_failure, refresh_setup_button
            owner._setupFunctionalFailure = True
            signature = owner._environment_check_signature(owner._environment_check_kwargs())
            remember_functional_failure(folder.name, signature)
            refresh_setup_button(owner)
            owner._lastEnvironmentResult = owner._lastEnvironmentSignature = None
            finish('failed', details)
        try:
            current_paths = owner._environment_dependency_path_kwargs()
            requests = requests_for_report(current, asset_root=current_paths.get('model_assets_dir'),
                                           backend_paths=current_paths,
                                           device_choices=owner._environment_check_kwargs(),
                                           repo_root=Path(__file__).resolve().parents[1])
        except Exception as exc:
            failed(str(exc))
            return
        if not requests:
            owner._setupFunctionalFailure = False
            record_readiness(current)
            return
        script = Path(__file__).resolve().parents[1] / 'scripts/check_setup_functionality.py'
        commands, receipts = [], []
        for index, request in enumerate(requests):
            path, receipt = folder / f'functional-{index}-request.json', folder / f'functional-{index}-result.json'
            write_json(path, request)
            program, prefix = owner._setup_python_command(request['python'])
            import os
            import sys
            environment = {'MPLBACKEND': 'Agg'}
            if not prefix and Path(program).absolute() != Path(sys.executable).absolute():
                from shared.external_process import external_python_environment
                clean = external_python_environment(path_prepend=[Path(program).parent])
                environment.update({key: None for key in os.environ if key not in clean})
                environment.update(clean)
            commands.append((program, [*prefix, '-u', str(script), '--request', str(path), '--result', str(receipt)],
                             environment))
            receipts.append(receipt)
        owner._selectedSetupActive = True

        def done(status, details, _log):
            owner._selectedSetupActive = False
            try:
                if status != 'completed' or any(json.loads(path.read_text()).get('status') != 'passed' for path in receipts):
                    raise RuntimeError(details or 'Installation functionality check failed')
            except Exception as exc:
                failed(str(exc))
                return
            owner._setupFunctionalFailure = False
            from shared.setup_selftest import receipt_warnings
            record_readiness(current, receipt_warnings(receipts))  # A2: warnings do not block "ready"

        start_setup_job(owner, _text(owner, 'Testing selected installation', '测试所选分析的安装'), commands, done)

    def record_readiness(current, warnings=()):
        owner._latestEnvironmentReportDir = folder
        pending = bool(getattr(owner, "_setupRestartRequired", False))
        state = "restart_required" if pending else "ready" if current and current.blocker_count == 0 else "blocked"
        if state == "ready":
            from shared.setup_resume import complete_setup, refresh_setup_button
            complete_setup(folder.name)
            if pending_before:
                complete_setup(pending_before['request'])
            refresh_setup_button(owner)
        record = {"status": state, "blockers": current.blocker_count if current else None,
                  "scope": "Environment readiness only; not scientific QC"}
        if warnings:
            record["warnings"] = list(warnings)
        write_json(folder / "result.json", record)
        if current:
            owner._update_environment_result_label(current)
            owner._update_environment_path_status_labels(current)
        note = " " + " ".join(warnings) if warnings and state == "ready" else ""  # A2: a warning does not block "ready"
        owner._setStatus("warning" if warnings or state != "ready" else "ready", _text(owner,
            "Selected environment is ready." + note if state == "ready" else
            "Restart Slicer before analysis." if pending else "Preparation finished; some selected dependencies still need attention.",
            "所选分析环境已就绪。" + note if state == "ready" else
            "请重启 Slicer 后再运行分析。" if pending else "自动准备已结束，仍有依赖需要处理。"))
        if current and state == "blocked":
            owner._show_environment_results_dialog(current)

    ZH_TITLES = {"backend_sources": "准备算法源码", "python_packages": "安装 Python 依赖", "model_assets": "下载并校验模型资源",
                 "optional_resources": "准备所选资源", "managed_runtime": "准备外部计算工具",
                 "windows_runtime": "安装 WSL 运行环境", "surface_runtime": "安装表面处理组件"}

    def report_step(step, status, details, log_path):
        """S-N1: the status writes and error dialogs that the owner's install methods made before B7."""
        if step["id"] == "python_packages":
            owner._save_python_install_status(tuple(step.get("packages", ())), status, str(log_path))
            owner._lastEnvironmentResult = owner._lastEnvironmentSignature = None
            if status == "failed":
                owner._show_actionable_error(
                    title="Python dependency installation failed", what="The background installer stopped.",
                    why="See the recorded pip output and exact command.",
                    next_step="Open install logs, correct the reported problem, then check dependencies again.",
                    auto_fix="No external software was installed.", details=str(details) + "\nLog: " + str(log_path))
            elif status == "cancelled":
                slicer.util.infoDisplay("Installation stopped. Packages already installed were not rolled back. "
                                        "Check dependencies before retrying.", parent=owner.parent)
        elif step["id"] == "model_assets":
            message = ("Model assets downloaded and checksum-verified." if status == "completed" else
                       "Download stopped; partial files are retained for retry." if status == "cancelled" else
                       "Download or checksum verification failed.")
            owner._set_model_assets_status_text(message + " " + str(step["asset_root"]))
            owner._lastEnvironmentResult = owner._lastEnvironmentSignature = None
            if status == "failed":
                owner._show_actionable_error(
                    title="Model asset setup failed", what=message,
                    why="Network, file permissions or checksum validation may have failed.",
                    next_step="Inspect the install log and retry using the same asset directory.",
                    auto_fix="Partial downloads are retained; no scientific outputs were modified.",
                    details=str(details) + "\nLog: " + str(log_path))

    def run_step(index):
        """B7: one async job per step; the steps come from setup_steps (shared with scripts/setup_selected.py)."""
        if index >= len(steps) or steps[index]["id"] == "self_check":
            finish()  # the final environment check and the functional self-check follow in finish()
            return
        step = steps[index]
        from shared.ui_setup_jobs import start_setup_job

        def done(status, details, log_path):
            if status == "completed":
                try:
                    settings = complete_step(step)
                    if settings:
                        apply_setup_settings(owner, settings)
                    if step["id"] == "python_packages" and step.get("restart_required"):
                        owner._setupRestartRequired = True
                except Exception as exc:
                    status, details = "failed", str(exc)
            report_step(step, status, details, log_path)
            if status != "completed":
                finish(status, details)
                return
            run_step(index + 1)

        start_setup_job(owner, _text(owner, step["title"], ZH_TITLES.get(step["id"], step["title"])), step["commands"], done)

    from shared.setup_components import LicenceNotAccepted
    try:
        from shared.selected_setup import setup_steps
        from shared.setup_resume import accepted_licences
        launcher = (owner._slicer_launcher_program(), ["--launch", "PythonSlicer"])
        steps = setup_steps(plan, accepted_licences(), folder=folder, launcher=launcher, config=config,
                            python_command=owner._setup_python_command,
                            statuses=getattr(getattr(owner, "_lastEnvironmentResult", None), "statuses", ()))
        run_step(0)
    except LicenceNotAccepted as exc:
        # A4 (2), OB3: a licence that is not accepted or not listed (only this error gets the licence title). Show its names, not the generic "stopped" text.
        finish("failed", str(exc))
        owner._setStatus("warning", str(exc))
        owner._show_actionable_error(
            title="Licence not accepted", what="Preparation did not start because of a licence.", why=str(exc),
            next_step="Start the preparation again and accept the licences in the dialog.",
            auto_fix="Nothing was installed.", details=str(exc))
        return False
    except Exception as exc:
        finish("failed", str(exc))
        return False
    return True


def use_offline_assets(owner):
    import qt
    import slicer
    from shared.user_config import default_user_config_path
    from shared.ui_setup_jobs import start_setup_job
    if (getattr(owner, "_setupJob", None) or getattr(owner, "_pipelineActive", False)
            or getattr(owner, "_selectedSetupActive", False)):
        return False
    selected = qt.QFileDialog.getExistingDirectory(slicer.util.mainWindow(),
        _text(owner, "Select an offline model resource directory", "选择离线模型资源目录"), str(Path.home()))
    if not selected:
        return False
    node = owner._parameterNode
    profiles = [name for name, target in (("surface", "targetSurface"), ("segmentation", "targetSegmentation"),
                                         ("parcellation", "targetParcellation")) if getattr(node, target, False)]
    report = getattr(owner, "_lastEnvironmentResult", None)
    from shared.optional_assets import missing_optional_profiles
    # F4: the offline path uses the route of the node's setting, as the plan does.
    optional = missing_optional_profiles(report, getattr(node, "brainMaskBackend", "pnl_cnn"),
                                         getattr(node, "wmaAtlas", "org"),
                                         pnl_route=getattr(node, "pnlMaskRoute", "auto") or "auto") if report else []
    if not profiles and not optional:
        slicer.util.infoDisplay(_text(owner, "Check the selected environment first to identify required resources.",
                                     "请先检查所选环境，确认需要的离线资源。"))
        return False
    receipt = default_user_config_path().parent / ("offline-assets-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".json")
    receipt.parent.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parents[1] / "scripts/verify_offline_assets.py"
    args = ["--launch", "PythonSlicer", "-u", str(script), "--root", str(selected), "--profiles", *profiles,
            "--result", str(receipt)]
    if optional:
        args.extend(("--optional-profiles", *optional))

    def done(status, details, log):
        try:
            record = json.loads(receipt.read_text()) if receipt.is_file() else {}
        except (OSError, ValueError):
            record = {}
        if status == "completed" and record.get("status") == "completed":
            owner._save_environment_path_setting("model_assets_dir", record["root"])
            owner._environmentPathLineEdits["model_assets_dir"].setText(record["root"])
            owner._lastEnvironmentResult = owner._lastEnvironmentSignature = None
            owner._setStatus("ready", _text(owner, "Offline models verified; check remaining dependencies.", "离线模型已校验，请检查其余依赖。"))
        else:
            owner._setStatus("warning", _text(owner, "Offline verification failed; previous model path was preserved.", "离线校验未通过，原模型路径保持不变。"))
        owner.appendLog("[Setup] Offline asset verification: " + str(receipt))
    return start_setup_job(owner, _text(owner, "Verify offline models", "校验离线模型"),
                           [(owner._slicer_launcher_program(), args, {})], done)
