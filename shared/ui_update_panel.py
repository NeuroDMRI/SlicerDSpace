"""Compact in-app code-overlay update bar mounted beneath the Log section."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from shared.code_updater import (
    EXTENSION_VERSION,
    check_for_code_update,
    download_and_apply_code_overlay,
)

_POLL_INTERVAL_MS = 120


def _is_zh(owner: Any, language: str | None = None) -> bool:
    raw = language if language is not None else getattr(owner, "_onboardingLanguage", "en")
    return isinstance(raw, str) and raw.startswith("zh")


def _t(zh: bool, en_text: str, zh_text: str) -> str:
    return zh_text if zh else en_text


def _log(owner: Any, message: str) -> None:
    fn = getattr(owner, "appendLog", None)
    if callable(fn):
        try:
            fn(message)
        except Exception:
            pass


def _format_mb(size_bytes: int) -> str:
    mb = max(0.01, float(size_bytes or 0) / (1024 * 1024))
    return f"{mb:.1f} MB" if mb >= 1.0 else f"{max(1, int(round(size_bytes / 1024)))} KB"


def _parent_window(owner: Any) -> Any:
    return getattr(owner, "_uiWidget", None)


def _restart_slicer() -> bool:
    try:
        import slicer

        util = getattr(slicer, "util", None)
        restart = getattr(util, "restart", None)
        if callable(restart):
            restart()
            return True
    except Exception:
        pass
    return False


def _stop_timer(owner: Any) -> None:
    timer = getattr(owner, "_codeUpdatePollTimer", None)
    if timer is not None and hasattr(timer, "stop"):
        try:
            timer.stop()
        except Exception:
            pass


def _sync_idle_texts(owner: Any, *, zh: bool | None = None) -> None:
    label = getattr(owner, "_codeUpdateStatusLabel", None)
    button = getattr(owner, "_checkUpdatesButton", None)
    if label is None or button is None:
        return
    use_zh = _is_zh(owner) if zh is None else bool(zh)
    state = getattr(owner, "_codeUpdateState", None) or {}
    phase = str(state.get("phase") or "idle")
    version = str(state.get("version") or EXTENSION_VERSION)

    if phase == "restart_required":
        label.setText(_t(use_zh, f"Update ready (v{version}) — restart Slicer", f"更新已就绪 (v{version}) · 请重启 Slicer"))
        button.setText(_t(use_zh, "Restart Slicer", "重启 Slicer"))
        button.setEnabled(True)
    elif phase == "checking":
        label.setText(_t(use_zh, "Checking for code updates...", "正在检查代码更新..."))
        button.setText(_t(use_zh, "Checking...", "正在检查..."))
        button.setEnabled(False)
    elif phase == "downloading":
        pct = int(state.get("percent") or 0)
        label.setText(_t(use_zh, f"Downloading code update... {pct}%", f"正在下载增量更新... {pct}%"))
        button.setText(_t(use_zh, f"Updating... {pct}%", f"正在更新... {pct}%"))
        button.setEnabled(False)
    elif phase == "up_to_date":
        label.setText(_t(use_zh, f"Up to date · v{version}", f"已是最新版本 · v{version}"))
        button.setText(_t(use_zh, "Check for updates", "检查更新"))
        button.setEnabled(True)
    elif phase == "error":
        label.setText(_t(use_zh, "Update check failed (see Log)", "检查失败（详情见日志）"))
        button.setText(_t(use_zh, "Check for updates", "检查更新"))
        button.setEnabled(True)
    elif phase == "runtime_upgrade":
        label.setText(_t(use_zh, f"Full runtime update available (v{version})", f"发现完整运行包升级 (v{version})"))
        button.setText(_t(use_zh, "Check for updates", "检查更新"))
        button.setEnabled(True)
    elif phase == "update_available":
        size_text = _format_mb(int(state.get("size_bytes") or 0))
        label.setText(_t(use_zh, f"Update available ({size_text})", f"发现代码更新 ({size_text})"))
        button.setText(_t(use_zh, "Check for updates", "检查更新"))
        button.setEnabled(True)
    else:
        label.setText(_t(use_zh, f"Extension · v{version}", f"插件版本 · v{version}"))
        button.setText(_t(use_zh, "Check for updates", "检查更新"))
        button.setEnabled(True)

    button.setToolTip(
        _t(
            use_zh,
            "Check NeuroDMRI/SlicerDSpace Releases for small (~2-3 MB) incremental code updates.",
            "检查 NeuroDMRI/SlicerDSpace Release 中的轻量级（约 2-3 MB）增量代码更新。",
        )
    )


def _handle_check_result(owner: Any, result: dict[str, Any]) -> None:
    import qt

    zh = _is_zh(owner)
    state = owner._codeUpdateState
    status = str(result.get("status") or "")
    remote = dict(result.get("remote") or {})
    version = str(remote.get("version") or EXTENSION_VERSION)
    short_sha = str(remote.get("git_short_commit") or "")
    size_bytes = int(result.get("overlay_size_bytes") or 0)
    summary = str(result.get("summary") or "").strip()
    state["version"] = version
    state["size_bytes"] = size_bytes

    if status == "up_to_date":
        state["phase"] = "up_to_date"
        _sync_idle_texts(owner, zh=zh)
        _log(owner, f"[Updater] Already up to date (v{version}{', ' + short_sha if short_sha else ''}).")
        return

    if status == "runtime_upgrade_required":
        state["phase"] = "runtime_upgrade"
        _sync_idle_texts(owner, zh=zh)
        release_url = str(result.get("release_url") or "")
        _log(owner, f"[Updater] Full base package upgrade required for v{version}: {release_url}")
        box = qt.QMessageBox(_parent_window(owner))
        box.setIcon(qt.QMessageBox.Information)
        box.setWindowTitle(_t(zh, "Full Upgrade Required", "需要更新完整基础包"))
        box.setText(
            _t(
                zh,
                f"SlicerDSpace v{version} updates native or PyTorch runtimes and cannot be applied as a small code overlay.\n\n"
                "Please install the updated base package from the Release page (your user settings, results, and downloaded atlases will be kept).",
                f"SlicerDSpace v{version} 包含底层 PyTorch 或原生运行时升级，无法仅通过代码增量包更新。\n\n"
                "请从 Release 页面获取最新安装包升级（已有用户设置、分析结果与已下载图谱会自动保留）。",
            )
        )
        open_btn = box.addButton(_t(zh, "Open Release Page", "打开 Release 页面"), qt.QMessageBox.AcceptRole)
        box.addButton(qt.QMessageBox.Close)
        box.exec_()
        if box.clickedButton() == open_btn and release_url:
            qt.QDesktopServices.openUrl(qt.QUrl(release_url))
        return

    if bool(result.get("is_git_checkout")):
        state["phase"] = "update_available"
        _sync_idle_texts(owner, zh=zh)
        qt.QMessageBox.information(
            _parent_window(owner),
            _t(zh, "Git Checkout Detected", "检测到 Git 开发源码仓库"),
            _t(
                zh,
                "A newer release is available on GitHub, but this installation is a Git checkout.\n\nPlease run `git pull` to update safely.",
                "检测到新版本代码更新，但当前是从 Git 源码仓库直接加载。\n\n请直接在仓库目录执行 `git pull` 更新，以免覆盖本地源码。",
            ),
        )
        return

    if not bool(result.get("writable")):
        state["phase"] = "update_available"
        _sync_idle_texts(owner, zh=zh)
        mod_root = str((result.get("local") or {}).get("module_root") or "")
        qt.QMessageBox.warning(
            _parent_window(owner),
            _t(zh, "Directory Not Writable", "插件目录不可写"),
            _t(
                zh,
                f"A code update (~{_format_mb(size_bytes)}) is available, but Slicer lacks write permission to:\n{mod_root}\n\nRestart Slicer with write access or reinstall the latest package.",
                f"发现约 {_format_mb(size_bytes)} 的增量代码更新，但当前无权写入插件目录：\n{mod_root}\n\n请以具备写权限的用户运行或重新安装最新版。",
            ),
        )
        return

    state["phase"] = "update_available"
    _sync_idle_texts(owner, zh=zh)
    ver_line = f"v{version}" + (f" ({short_sha})" if short_sha else "")
    notes_line = f"\n{summary}\n" if summary else "\n"
    prompt = _t(
        zh,
        f"A lightweight incremental code update is available for SlicerDSpace.\n\n"
        f"Target: {ver_line}\n"
        f"Download size: {_format_mb(size_bytes)}"
        f"{notes_line}\n"
        "Your models, WMA atlases, and settings will NOT be re-downloaded.\n"
        "Download and apply now?",
        f"发现 SlicerDSpace 轻量级增量代码更新。\n\n"
        f"目标版本：{ver_line}\n"
        f"下载体积：{_format_mb(size_bytes)}"
        f"{notes_line}\n"
        "内置模型权重、WMA 图谱与用户设置均会保留，无需重下。\n"
        "是否立即下载并应用？",
    )
    reply = qt.QMessageBox.question(
        _parent_window(owner),
        _t(zh, "Code Update Available", "发现增量更新"),
        prompt,
        qt.QMessageBox.Yes | qt.QMessageBox.No,
        qt.QMessageBox.Yes,
    )
    if reply == qt.QMessageBox.Yes:
        _start_apply_worker(owner, remote)


def _handle_apply_result(owner: Any, result: dict[str, Any]) -> None:
    import qt

    zh = _is_zh(owner)
    state = owner._codeUpdateState
    version = str(result.get("version") or EXTENSION_VERSION)
    short_sha = str(result.get("git_short_commit") or "")
    state["version"] = version
    if str(result.get("status") or "") == "up_to_date":
        state["phase"] = "up_to_date"
        _sync_idle_texts(owner, zh=zh)
        return

    state["phase"] = "restart_required"
    _sync_idle_texts(owner, zh=zh)
    updated_count = int(result.get("updated_files") or 0)
    _log(
        owner,
        f"[Updater] Applied incremental update v{version} ({short_sha}, {updated_count} files). Restart Slicer to load new code.",
    )
    box = qt.QMessageBox(_parent_window(owner))
    box.setIcon(qt.QMessageBox.Information)
    box.setWindowTitle(_t(zh, "Update Applied", "增量更新完成"))
    box.setText(
        _t(
            zh,
            f"Code update v{version} ({updated_count} files) has been verified and installed.\n\nRestart 3D Slicer now to load the updated extension?",
            f"增量代码更新 v{version}（{updated_count} 个文件）已校验并安装完成。\n\n是否立即重启 3D Slicer 以启用最新版本？",
        )
    )
    restart_btn = box.addButton(_t(zh, "Restart Slicer Now", "立即重启 Slicer"), qt.QMessageBox.AcceptRole)
    box.addButton(_t(zh, "Later", "稍后重启"), qt.QMessageBox.RejectRole)
    box.exec_()
    if box.clickedButton() == restart_btn:
        _restart_slicer()


def _poll_worker(owner: Any) -> None:
    state = getattr(owner, "_codeUpdateState", None)
    if not isinstance(state, dict):
        _stop_timer(owner)
        return
    if not state.get("done"):
        if state.get("phase") == "downloading":
            _sync_idle_texts(owner)
        return

    _stop_timer(owner)
    state["busy"] = False
    kind = state.get("kind")
    error = state.get("error")
    result = state.get("result")
    state["done"] = False
    state["result"] = None
    state["error"] = None

    if error:
        zh = _is_zh(owner)
        state["phase"] = "error"
        _sync_idle_texts(owner, zh=zh)
        _log(owner, f"[Updater] {error}")
        try:
            import qt

            qt.QMessageBox.warning(
                _parent_window(owner),
                _t(zh, "Update Failed", "更新失败"),
                _t(
                    zh,
                    f"Could not complete update check:\n{error}\n\nYou can adjust mirror or proxy settings under Advanced settings.",
                    f"更新操作未能完成：\n{error}\n\n如有网络访问困难，可在高级设置中调整镜像或代理地址后重试。",
                ),
            )
        except Exception:
            pass
        return

    if kind == "check" and isinstance(result, dict):
        _handle_check_result(owner, result)
    elif kind == "apply" and isinstance(result, dict):
        _handle_apply_result(owner, result)
    else:
        _sync_idle_texts(owner)


def _ensure_timer(owner: Any) -> Any:
    import qt

    timer = getattr(owner, "_codeUpdatePollTimer", None)
    if timer is None:
        timer = qt.QTimer(_parent_window(owner))
        timer.setInterval(_POLL_INTERVAL_MS)
        timer.timeout.connect(lambda: _poll_worker(owner))
        owner._codeUpdatePollTimer = timer
    return timer


def _start_check_worker(owner: Any) -> None:
    state = owner._codeUpdateState
    if state.get("busy"):
        return
    state["busy"] = True
    state["done"] = False
    state["kind"] = "check"
    state["phase"] = "checking"
    state["result"] = None
    state["error"] = None
    _sync_idle_texts(owner)
    _ensure_timer(owner).start()

    def _run() -> None:
        try:
            state["result"] = check_for_code_update()
        except Exception as exc:
            state["error"] = str(exc)
        finally:
            state["done"] = True

    threading.Thread(target=_run, daemon=True, name="SlicerDSpaceUpdateCheck").start()


def _start_apply_worker(owner: Any, remote_manifest: dict[str, Any]) -> None:
    state = owner._codeUpdateState
    if state.get("busy"):
        return
    state["busy"] = True
    state["done"] = False
    state["kind"] = "apply"
    state["phase"] = "downloading"
    state["percent"] = 0
    state["result"] = None
    state["error"] = None
    _sync_idle_texts(owner)
    _ensure_timer(owner).start()

    def _on_progress(downloaded: int, total: int) -> None:
        if total > 0:
            state["percent"] = max(0, min(100, int(downloaded * 100 / total)))

    def _run() -> None:
        try:
            state["result"] = download_and_apply_code_overlay(
                manifest=remote_manifest,
                progress_callback=_on_progress,
            )
        except Exception as exc:
            state["error"] = str(exc)
        finally:
            state["done"] = True

    threading.Thread(target=_run, daemon=True, name="SlicerDSpaceUpdateApply").start()


def _on_check_updates_clicked(owner: Any) -> None:
    state = getattr(owner, "_codeUpdateState", None) or {}
    if state.get("busy"):
        return
    if state.get("phase") == "restart_required":
        _restart_slicer()
        return
    _start_check_worker(owner)


def install_update_panel(owner: Any) -> Any:
    """Mount the incremental code update bar directly beneath ui.logCollapsibleButton."""

    existing = getattr(owner, "_codeUpdateBar", None)
    if existing is not None:
        return existing

    import qt

    root = getattr(owner, "_uiWidget", None)
    ui = getattr(owner, "ui", None)
    if root is None or ui is None or not hasattr(root, "layout"):
        return None
    layout = root.layout()
    log_box = getattr(ui, "logCollapsibleButton", None)
    if layout is None or log_box is None:
        return None

    bar = qt.QFrame(root)
    bar.setObjectName("codeUpdateBarFrame")
    bar.setFrameShape(qt.QFrame.NoFrame)
    bar.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
    bar.setMinimumWidth(0)

    bar_layout = qt.QHBoxLayout(bar)
    bar_layout.setContentsMargins(2, 2, 2, 0)
    bar_layout.setSpacing(6)

    status_label = qt.QLabel(bar)
    status_label.setObjectName("codeUpdateStatusLabel")
    status_label.setSizePolicy(qt.QSizePolicy.Expanding, qt.QSizePolicy.Fixed)
    status_label.setMinimumWidth(0)

    check_btn = qt.QPushButton(bar)
    check_btn.setObjectName("checkUpdatesButton")
    check_btn.setSizePolicy(qt.QSizePolicy.Maximum, qt.QSizePolicy.Fixed)
    pointing = getattr(qt.Qt, "PointingHandCursor", None)
    if pointing is not None and hasattr(check_btn, "setCursor"):
        check_btn.setCursor(pointing)
    check_btn.clicked.connect(lambda: _on_check_updates_clicked(owner))

    bar_layout.addWidget(status_label)
    bar_layout.addWidget(check_btn)

    idx = layout.indexOf(log_box)
    if idx >= 0:
        layout.insertWidget(idx + 1, bar)
    else:
        layout.addWidget(bar)

    owner._codeUpdateBar = bar
    owner._codeUpdateStatusLabel = status_label
    owner._checkUpdatesButton = check_btn
    owner._codeUpdateState = {
        "phase": "idle",
        "version": EXTENSION_VERSION,
        "size_bytes": 0,
        "percent": 0,
        "busy": False,
        "done": False,
        "kind": None,
        "result": None,
        "error": None,
    }
    _sync_idle_texts(owner)
    return bar


def refresh_update_panel_language(owner: Any, language: str | None = None) -> None:
    """Update the code update bar text when the UI language switches."""

    if getattr(owner, "_codeUpdateBar", None) is None:
        return
    _sync_idle_texts(owner, zh=_is_zh(owner, language))


def cleanup_update_panel(owner: Any) -> None:
    """Stop any active update polling timer when closing the module."""

    _stop_timer(owner)
