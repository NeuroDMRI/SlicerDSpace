"""One explicit preparation action, with resumable phases and truthful status."""

from datetime import datetime
import json
from pathlib import Path


def python_setup_commands(owner, packages, *, requests=None):
    """Group requests by their selected Python, never mix runtime targets."""
    import importlib.metadata
    import os
    import sys
    from shared.managed_runtimes import write_json
    from shared.user_config import default_user_config_path

    folder = Path(getattr(owner, "_latestEnvironmentReportDir", None) or default_user_config_path().parent / "setup-logs")
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    script = Path(__file__).resolve().parents[1] / "scripts/setup_python_packages.py"
    groups = {}
    result = getattr(owner, "_lastEnvironmentResult", None)
    from shared.python_setup import _packaging_api
    InvalidRequirement, Requirement, canonicalize = _packaging_api()

    def package_name(value):
        try:
            return canonicalize(Requirement(value).name)
        except InvalidRequirement:
            return value

    selections = [(item['requirement'], item) for item in requests] if requests is not None else [(package, None) for package in packages]
    for package, explicit in selections:
        matches = [explicit] if explicit is not None else [(item.details or {}) for item in getattr(result, "statuses", ())
                        if item.kind == "python" and any(value and package_name(package) == package_name(value)
                            for value in ((item.details or {}).get("package"), (item.details or {}).get("install_requirement")))]
        for details in matches or [{}]:
            program, prefix = owner._setup_python_command(details.get("python_executable"))
            group = groups.setdefault((str(program), tuple(prefix)), {"requirements": [], "environment": {}})
            requirement = details.get("install_requirement") or package
            try:
                if Requirement(package).specifier:
                    requirement = package
            except InvalidRequirement:
                pass
            if requirement == "git+https://github.com/demianw/tract_querier.git":
                root = owner._environment_dependency_path_kwargs().get("tract_querier_root")
                if root and (Path(root) / "pyproject.toml").is_file():
                    requirement = str(root)
                    group["environment"]["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_TRACT_QUERIER"] = "0+local"
            if requirement not in group["requirements"]:
                group["requirements"].append(requirement)
    loaded = {name for module, names in importlib.metadata.packages_distributions().items()
              if module in sys.modules for name in names}
    commands, receipts = [], []
    for index, ((program, prefix), data) in enumerate(groups.items()):
        slicer_python = bool(prefix) or Path(program).absolute() == Path(sys.executable).absolute()
        requirements = data["requirements"]
        try:
            if all(not Requirement(value).url for value in requirements):
                data['environment']['PIP_ONLY_BINARY'] = ':all:'
        except InvalidRequirement:
            pass
        if not slicer_python:
            from shared.external_process import external_python_environment
            clean = external_python_environment(path_prepend=[Path(program).parent])
            overrides = {key: None for key in os.environ if key not in clean}
            overrides.update(clean)
            overrides.update(data["environment"])
            data["environment"] = overrides
        torch_minimum = None
        torchvision_required = any(item == "torchvision" or item.startswith(("torchvision>", "torchvision=", "torchvision<"))
                                   for item in requirements)
        if slicer_python and (torchvision_required or any(item == "torch" or item.startswith(("torch>", "torch=", "torch<")) for item in requirements)):
            from packaging.requirements import Requirement
            from packaging.version import Version
            minimums = ["2.0.0"]
            for value in requirements:
                if value.startswith('torch>='):
                    minimums.extend(spec.version for spec in Requirement(value).specifier if spec.operator == '>=')
            torch_minimum = str(max(map(Version, minimums)))
            requirements = [item for item in requirements if not (item == "torch" or item.startswith(("torch>", "torch=", "torch<")))]
        request, receipt = (folder / f"python-{stamp}-{index}-{suffix}.json" for suffix in ("request", "result"))
        write_json(request, {"requirements": requirements, "torch_minimum": torch_minimum,
                            "torchvision_required": torchvision_required and slicer_python,
                            "protect_bundled": slicer_python, "loaded_packages": sorted(loaded) if slicer_python else [],
                            "python_path_entries": [path for path in sys.path if path] if slicer_python else []})
        commands.append((program, [*prefix, "-u", str(script), "--request", str(request), "--result", str(receipt)], data["environment"]))
        receipts.append(receipt)
    return commands, receipts


def _text(owner, english, chinese):
    return chinese if str(getattr(owner, "_onboardingLanguage", "en")).startswith("zh") else english


def extension_states(manager, modules):
    """Use exposed APIs and actual capabilities, not enabled == loaded.

    Slicer 5.10's C++ isExtensionLoaded method is not exposed to PythonQt.
    An enabled extension may still need restart before its modules exist.
    """
    import importlib.util
    installed, loaded = [], []
    for name, module in (("SlicerDMRI", "dwitodtiestimation"), ("PyTorch", "pytorchutils"),
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
    import qt
    import slicer
    from shared.managed_runtimes import FSL_TERMS_URL
    from shared.managed_pnl import PNL_TERMS_URL, PNL_MODEL_TERMS_URL, PNL_TEMPLATE_TERMS_URL

    dialog = qt.QDialog(slicer.util.mainWindow())
    dialog.setObjectName("selectedSetupConfirmationDialog")
    dialog.setWindowTitle(_text(owner, "Prepare selected analysis", "准备所选分析环境"))
    layout = qt.QVBoxLayout(dialog)
    parts = [_text(owner, "Only missing items for the selected analysis will be prepared. Existing installations are reused. No driver, shell or algorithm selection is changed.",
                   "仅准备当前分析缺少的依赖，复用已有安装；不修改驱动、shell 或算法选择。")]
    parts.append(_text(owner, "First-use downloads may require several GB and network access. Cancellation keeps completed files for retry.",
                       "首次下载可能需要数 GB 空间及网络连接；取消后保留已完成文件供重试。"))
    if plan.get("also_runs"):
        parts.append(plan["also_runs"])
    if plan.get("repair"):
        parts.append(_text(owner, "Repair will reinstall selected plugin-managed tools. Existing administrator environments are not changed.",
                           "修复将重新安装所选的插件受管工具，不修改管理员或自行配置的环境。"))
    for key in ("extensions", "restart_extensions", "python_packages", "asset_profiles", "optional_asset_profiles", "managed_profiles", "unsupported_profiles"):
        if plan[key]:
            parts.append(key.replace("_", " ") + ": " + ", ".join(plan[key]))
    parts.extend(plan.get("unsupported_notes", {}).values())
    from scripts.download_model_assets import _format_bytes
    parts.append(_text(owner, 'Models and resources: ', '模型和资源目录：') + plan['asset_root'])
    if plan.get('managed_profiles'):
        parts.append(_text(owner, 'Managed tools: ', '受管工具目录：') + plan['runtime_root'])
    if plan.get('core_model_profile_bytes'):
        parts.append(_text(owner, 'Selected core model profiles: ', '所选核心模型分组大小：') +
                     _format_bytes(plan['core_model_profile_bytes']))
    if plan["manual_items"]:
        parts.append(_text(owner, "Some items still need attention: ", "以下项目仍需处理：") +
                     ", ".join(item["label"] for item in plan["manual_items"]))
    if plan.get('backend_profiles'):
        from scripts.download_model_assets import _format_bytes
        parts.append(_text(owner, 'Algorithm sources: ', '算法源码：') + ', '.join(plan['backend_profiles']) + '\n' +
                     str(plan['backend_root']) + '\n' +
                     _text(owner, 'Download: ', '下载量：') + _format_bytes(plan['backend_download_bytes']))
        parts.extend(plan.get('backend_terms', ()))
    if plan.get('surface_build_required') or plan.get('surface_prebuilt_required'):
        parts.append(_text(owner,
            'Surface postprocessing: install the qualified precompiled components in your user cache. No Git, compiler or full FreeSurfer installation is needed. Your own activation file is still required for analysis. Unpublished bundles must first be supplied as offline resources.',
            '表面后处理：在用户缓存中安装经校验的预编译组件，无需 Git、编译器或完整版 FreeSurfer。分析仍需您自己的激活文件；尚未发布的运行包须先通过离线资源提供。'))
    if plan.get('surface_bundle_unavailable'):
        parts.append(_text(owner,
            'Surface components: this bundle has not been published and is absent from your resource directory. Supply the qualified offline bundle or configure an existing runtime. Other selected resources can still be prepared.',
            '表面处理组件：运行包尚未发布，资源目录中也没有该包。请导入合格的离线包或配置已有运行目录；其他所选资源仍可准备。'))
    if plan["pnl_terms_required"]:
        parts.append(_text(owner, "PNL source, model and template permissions are separate; download/hash verification is not a license grant.",
                           "PNL 源码、模型与模板的权限须分别核对；下载及哈希验证不代表获得授权。"))
    if plan["optional_asset_profiles"]:
        from scripts.download_model_assets import _format_bytes
        parts.append(_text(owner, "Selected optional resources (full profile size): ", "所选可选资源（完整分组大小）：") +
                     _format_bytes(plan["optional_asset_size_bytes"]))
        if plan.get("optional_asset_peak_disk_bytes", 0) > plan["optional_asset_size_bytes"]:
            parts.append(_text(owner, "Allow disk space for download, extracted data and temporary assembly: ",
                               "下载、解压与临时拼接所需磁盘空间上限估计：") +
                         _format_bytes(plan["optional_asset_peak_disk_bytes"]))
        parts.extend(plan["optional_asset_terms"])
    label = qt.QLabel("\n\n".join(parts))
    label.wordWrap = True
    scroll = qt.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(label)
    scroll.setMinimumHeight(160)
    scroll.setMaximumHeight(320)
    layout.addWidget(scroll)
    dialog.setMinimumWidth(500)
    approvals = []
    if plan.get('surface_build_required') or plan.get('surface_prebuilt_required'):
        terms = qt.QLabel('<a href="https://surfer.nmr.mgh.harvard.edu/fswiki/FreeSurferSoftwareLicense">FreeSurfer source/data terms</a> | ITK / zlib notices')
        terms.openExternalLinks = True
        layout.addWidget(terms)
        check = qt.QCheckBox(_text(owner, 'I have reviewed the component/data terms and approve installation.',
                                  '我已阅读组件与数据许可，同意安装。'))
        approvals.append(check)
        layout.addWidget(check)
    if plan.get('backend_profiles'):
        check = qt.QCheckBox(_text(owner, 'I have reviewed the algorithm source licences above.',
                                  '我已阅读以上算法源码许可。'))
        approvals.append(check)
        layout.addWidget(check)
    if plan["optional_asset_profiles"]:
        check = qt.QCheckBox(_text(owner, "I have reviewed the resource terms above; my use complies.",
                                  "我已阅读以上资源许可，且本次使用符合许可。"))
        approvals.append(check)
        layout.addWidget(check)
    for required, url, caption in (
        (plan["fsl_license_required"], FSL_TERMS_URL, "FSL"),
        (plan["pnl_terms_required"], PNL_TERMS_URL, "PNL source, models and IIT template"),
    ):
        if required:
            link = qt.QLabel(f'<a href="{url}">{caption}</a>')
            link.openExternalLinks = True
            layout.addWidget(link)
            if caption.startswith("PNL"):
                assets = qt.QLabel(f'<a href="{PNL_MODEL_TERMS_URL}">PNL models</a> | '
                                  f'<a href="{PNL_TEMPLATE_TERMS_URL}">IIT template</a>')
                assets.openExternalLinks = True
                layout.addWidget(assets)
            check = qt.QCheckBox(_text(owner, "I have reviewed the terms and my use complies.", "我已阅读相关条款，且本次使用符合许可。"))
            approvals.append(check)
            layout.addWidget(check)
    buttons = qt.QDialogButtonBox(dialog)
    buttons.setStandardButtons(qt.QDialogButtonBox.Ok | qt.QDialogButtonBox.Cancel)
    buttons.button(qt.QDialogButtonBox.Ok).setText(_text(owner, "Prepare", "开始准备"))
    buttons.button(qt.QDialogButtonBox.Ok).enabled = not approvals
    for check in approvals:
        check.connect("toggled(bool)", lambda *_: setattr(buttons.button(qt.QDialogButtonBox.Ok),
                                                          "enabled", all(item.checked for item in approvals)))
    buttons.connect("accepted()", dialog.accept)
    buttons.connect("rejected()", dialog.reject)
    layout.addWidget(buttons)
    accepted = dialog.exec_() == qt.QDialog.Accepted
    dialog.deleteLater()
    return accepted


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
    if closure is None:
        # N3: no parameter node. The doctor result's targets are already closed, so take them
        # as they are (no second expansion); the default algorithm set applies.
        closure = Closure(targets=tuple(result.targets), tractography_algorithms=("csd_act",),
                          requested=tuple(result.targets))
    targets = set(closure.targets)
    gpu = cuda_candidate()
    candidates = list(selected_profiles(owner._parameterNode, gpu))
    if (getattr(owner._parameterNode, "brainMaskBackend", "pnl_cnn") == "pnl_cnn" and
            any(item.key == "brain_mask_backend" and item.required for item in result.statuses)):
        candidates.append("pnl_cnn")
    config = owner._environment_dependency_path_kwargs()
    existing = configured_profiles(config, candidates) if candidates else ()
    plan = build_setup_plan(result, installed_extensions=installed, loaded_extensions=loaded,
        need_dti=bool(targets & {"segmentation", "parcellation", "surface", "microstructure", "tractography", "wma"}),
        need_slicerdmri="wma" in targets,
        need_torch=bool(targets & {"segmentation", "parcellation", "surface", "tractcloud"}),
        need_ukf="tractography" in targets and "ukf" in closure.tractography_algorithms,
        brain_mask_backend=getattr(owner._parameterNode, "brainMaskBackend", "pnl_cnn"), gpu_available=gpu,
        wma_atlas=getattr(owner._parameterNode, "wmaAtlas", "org"),
        existing_managed_profiles=existing, repair=repair, backend_paths=config)
    # C4: the result's targets are already closed, so the owner (which knows what was ticked) words the notice.
    notice = getattr(owner, "_closure_notice_text", None)
    notice = notice() if callable(notice) else ""
    plan["also_runs"] = notice if isinstance(notice, str) else closure_notice(closure)
    plan["closure_targets"] = list(closure.targets)
    return plan


def prepare_selected_environment(owner, result=None, *, repair=False):
    import slicer
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
    pending_actions = plan.get('surface_build_required', False) or plan.get('surface_prebuilt_required', False) or any(plan[key] for key in (
        "extensions", "restart_extensions", "python_packages", "asset_profiles", "optional_asset_profiles",
        "managed_profiles", "unsupported_profiles", "manual_items", "backend_profiles"))
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
    for name in plan["extensions"]:
        try:
            success = manager is not None and manager.installExtensionFromServer(name, False)
        except Exception as exc:
            owner.appendLog("[Setup] Extension install failed: " + str(exc))
            success = False
        if not success:
            owner._selectedSetupActive = False
            owner._setStatus("warning", _text(owner, "Extension installation did not finish: ", "扩展安装未完成：") + name)
            write_json(folder / "result.json", {"status": "blocked", "extension": name})
            return False
    for name in plan["restart_extensions"]:
        try:
            manager.setExtensionEnabled(name, True)
        except Exception as exc:
            owner._selectedSetupActive = False
            write_json(folder / "result.json", {"status": "blocked", "extension": name, "error": str(exc)})
            owner._setStatus("warning", str(exc))
            return False
    if plan["extensions"] or plan["restart_extensions"]:
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
            record_readiness(current)

        start_setup_job(owner, _text(owner, 'Testing selected installation', '测试所选分析的安装'), commands, done)

    def record_readiness(current):
        owner._latestEnvironmentReportDir = folder
        pending = bool(getattr(owner, "_setupRestartRequired", False))
        state = "restart_required" if pending else "ready" if current and current.blocker_count == 0 else "blocked"
        if state == "ready":
            from shared.setup_resume import complete_setup, refresh_setup_button
            complete_setup(folder.name)
            if pending_before:
                complete_setup(pending_before['request'])
            refresh_setup_button(owner)
        write_json(folder / "result.json", {"status": state, "blockers": current.blocker_count if current else None,
                                            "scope": "Environment readiness only; not scientific QC"})
        if current:
            owner._update_environment_result_label(current)
            owner._update_environment_path_status_labels(current)
        owner._setStatus("ready" if state == "ready" else "warning", _text(owner,
            "Selected environment is ready." if state == "ready" else
            "Restart Slicer before analysis." if pending else "Preparation finished; some selected dependencies still need attention.",
            "所选分析环境已就绪。" if state == "ready" else
            "请重启 Slicer 后再运行分析。" if pending else "自动准备已结束，仍有依赖需要处理。"))
        if current and state == "blocked":
            owner._show_environment_results_dialog(current)

    def managed_setup():
        if not plan["managed_profiles"]:
            finish()
            return
        from shared.ui_setup_jobs import start_setup_job
        from shared.ui_managed_runtimes import apply_runtime_result
        script = Path(__file__).resolve().parents[1] / "scripts/setup_managed_runtime.py"
        receipt = folder / "runtime-result.json"
        args = ["--launch", "PythonSlicer", "-u", str(script), "--profiles", *plan["managed_profiles"],
                "--execute", "--result", str(receipt)]
        if plan["repair"]:
            args.append("--repair")
        if plan["fsl_license_required"]:
            args.append("--accept-fsl-license")
        if plan["pnl_terms_required"]:
            args.append("--accept-pnl-terms")
        asset_root = config.get("model_assets_dir")
        if asset_root:
            args.extend(("--asset-root", str(asset_root)))
        conda = owner._environment_dependency_path_kwargs().get("conda_executable")
        if conda:
            args.extend(("--conda", str(conda)))

        def done(status, details, _log):
            if status == "completed":
                try:
                    apply_runtime_result(owner, json.loads(receipt.read_text()))
                except Exception as exc:
                    finish("failed", str(exc))
                    return
            finish(status, details)
        start_setup_job(owner, _text(owner, "Prepare external tools", "准备外部计算工具"),
                        [(owner._slicer_launcher_program(), args, {})], done)

    def surface_setup():
        if not plan.get('surface_build_required') and not plan.get('surface_prebuilt_required'):
            managed_setup()
            return
        from shared.ui_setup_jobs import start_setup_job
        from shared.ui_managed_runtimes import apply_runtime_result
        receipt = folder / 'surface-runtime-result.json'
        script = Path(__file__).resolve().parents[1] / 'scripts/setup_surface_runtime.py'
        args = ['--launch', 'PythonSlicer', '-u', str(script), '--execute', '--accept-terms',
                '--jobs', '4', '--result', str(receipt)]
        for key, flag in (('model_assets_dir', '--asset-root'), ('postprocess_runtime', '--prefix')):
            if config.get(key):
                args += [flag, str(config[key])]

        def done(status, details, _log):
            if status != 'completed':
                finish(status, details)
                return
            try:
                apply_runtime_result(owner, json.loads(receipt.read_text()))
            except Exception as exc:
                finish('failed', str(exc))
                return
            managed_setup()

        start_setup_job(owner, _text(owner, 'Install surface components', '安装表面处理组件'),
                        [(owner._slicer_launcher_program(), args, {})], done)

    def optional_setup():
        if not plan["optional_asset_profiles"]:
            surface_setup()
            return
        from shared.ui_setup_jobs import start_setup_job
        from scripts.setup_optional_assets import PATH_SETTINGS
        from shared.model_assets import configured_model_assets_root
        root = configured_model_assets_root(config.get("model_assets_dir"))
        receipt = folder / "optional-resources-result.json"
        script = Path(__file__).resolve().parents[1] / "scripts/setup_optional_assets.py"
        args = ["--launch", "PythonSlicer", "-u", str(script), "--profiles", *plan["optional_asset_profiles"],
                "--asset-root", str(root), "--execute", "--accept-terms", "--result", str(receipt)]

        def done(status, details, _log):
            if status != "completed":
                finish(status, details)
                return
            try:
                record = json.loads(receipt.read_text())
                if (record.get("kind") != "slicerdsurfer_optional_asset_setup" or record.get("status") != "completed"
                        or record.get("profiles") != plan["optional_asset_profiles"] or record.get("root") != str(root)):
                    raise ValueError("Optional resource preparation receipt is incomplete")
                settings = record["external_dependencies"]
                allowed = {"model_assets_dir", *(PATH_SETTINGS[name] for name in plan["optional_asset_profiles"] if name in PATH_SETTINGS)}
                if set(settings) != allowed or any(not Path(value).is_dir() for value in settings.values()):
                    raise ValueError("Prepared resource paths are unavailable")
                for key, value in settings.items():
                    owner._save_environment_path_setting(key, value)
                    edit = (getattr(owner, "_environmentPathLineEdits", {}) or {}).get(key)
                    if edit is not None:
                        edit.setText(value)
            except Exception as exc:
                finish("failed", str(exc))
                return
            surface_setup()

        start_setup_job(owner, _text(owner, "Prepare selected resources", "准备所选资源"),
                        [(owner._slicer_launcher_program(), args, {})], done)

    def model_setup(ok=True):
        if not ok:
            finish("failed", "Python setup did not complete")
        elif plan["asset_profiles"]:
            owner._install_model_assets_from_environment_dialog(profile=tuple(plan["asset_profiles"]),
                on_complete=lambda success, message: optional_setup() if success else finish("failed", message))
        else:
            optional_setup()

    def python_setup():
        if plan["python_packages"]:
            owner._install_python_packages_from_environment_dialog(tuple(plan["python_packages"]), on_complete=model_setup,
                requests=plan['python_requests'])
        else:
            model_setup()

    def backend_setup():
        if not plan['backend_profiles']:
            python_setup()
            return
        from shared.ui_setup_jobs import start_setup_job
        from shared.backend_sources import SOURCES, verify_source, profile_path
        script = Path(__file__).resolve().parents[1] / 'scripts/setup_backend_sources.py'
        receipt = folder / 'backend-sources-result.json'
        args = ['--launch', 'PythonSlicer', '-u', str(script), '--profiles', *plan['backend_profiles'],
                '--root', str(plan['backend_root']), '--execute', '--accept-terms', '--result', str(receipt)]

        def done(status, details, _log):
            if status != 'completed':
                finish(status, details)
                return
            try:
                record = json.loads(receipt.read_text())
                expected = {SOURCES[name]['setting']: str(profile_path(name, plan['backend_root']))
                            for name in plan['backend_profiles']}
                if record.get('status') != 'completed' or record.get('profiles') != plan['backend_profiles'] or record.get('external_dependencies') != expected:
                    raise ValueError('Invalid prepared algorithm-source receipt')
                for name in plan['backend_profiles']:
                    verify_source(name, Path(expected[SOURCES[name]['setting']]))
                for key, value in expected.items():
                    owner._save_environment_path_setting(key, value)
                    edit = (getattr(owner, '_environmentPathLineEdits', {}) or {}).get(key)
                    if edit is not None:
                        edit.setText(value)
            except Exception as exc:
                finish('failed', str(exc))
                return
            python_setup()

        start_setup_job(owner, _text(owner, 'Prepare algorithm sources', '准备算法源码'),
                        [(owner._slicer_launcher_program(), args, {})], done)

    try:
        backend_setup()
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
    optional = missing_optional_profiles(report, getattr(node, "brainMaskBackend", "pnl_cnn"),
                                         getattr(node, "wmaAtlas", "org")) if report else []
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
