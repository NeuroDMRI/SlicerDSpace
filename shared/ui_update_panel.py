"""Compact in-app code-overlay update bar mounted beneath the Log section."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from shared.code_updater import EXTENSION_VERSION
from shared.external_process import find_pythonslicer, normalize_slicer_python_launch
from shared.network_config import network_env_overrides
from shared.process_group import (
    capture_process_group_id,
    grouped_launch_command,
    kill_process_tree,
)
from shared.ui_output_loading import qbytearray_to_text

_POLL_INTERVAL_MS = 500
_CHECK_WATCHDOG_SECONDS = 60
_APPLY_WATCHDOG_SECONDS = 240
_MAX_OUTPUT_TAIL_CHARS = 4000


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
        elapsed = int(state.get("elapsed_sec") or 0)
        suffix = f" ({elapsed}s)" if elapsed > 0 else ""
        label.setText(_t(use_zh, f"Checking for code updates...{suffix}", f"正在检查代码更新...{suffix}"))
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
        _log(owner, "[Updater] Update available, skipped because installation is a Git checkout.")
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
        _log(owner, f"[Updater] Update available ({_format_mb(size_bytes)}), but module directory is read-only: {mod_root}")
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
    _log(owner, f"[Updater] Update available: {ver_line}, size {_format_mb(size_bytes)}.")
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
        _log(owner, f"[Updater] Already up to date (v{version}{', ' + short_sha if short_sha else ''}).")
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


def _show_error(owner: Any, error_msg: str) -> None:
    zh = _is_zh(owner)
    state = owner._codeUpdateState
    state["phase"] = "error"
    _sync_idle_texts(owner, zh=zh)
    _log(owner, f"[Updater] {error_msg}")
    try:
        import qt

        qt.QMessageBox.warning(
            _parent_window(owner),
            _t(zh, "Update Failed", "更新失败"),
            _t(
                zh,
                f"Could not complete update check:\n{error_msg}\n\nYou can adjust mirror or proxy settings under Advanced settings.",
                f"更新操作未能完成：\n{error_msg}\n\n如有网络访问困难，可在高级设置中调整镜像或代理地址后重试。",
            ),
        )
    except Exception:
        pass


def _resolve_worker_command(owner: Any, worker_args: list[str]) -> tuple[str, list[str]]:
    updater_script = Path(__file__).resolve().with_name("code_updater.py")
    launcher_fn = getattr(owner, "_slicer_launcher_program", None)
    if callable(launcher_fn):
        raw_program = str(launcher_fn() or "").strip()
        if raw_program:
            program, args = normalize_slicer_python_launch(
                raw_program,
                ["--launch", "PythonSlicer", "-u", str(updater_script), *worker_args],
            )
            return program, [arg for arg in args if arg != ""]
    candidate = find_pythonslicer(Path(sys.executable).resolve().parent.parent)
    program = str(candidate) if candidate is not None else sys.executable
    return program, ["-u", str(updater_script), *worker_args]


def _ensure_timer(owner: Any) -> Any:
    import qt

    timer = getattr(owner, "_codeUpdatePollTimer", None)
    if timer is None:
        timer = qt.QTimer(_parent_window(owner))
        timer.setInterval(_POLL_INTERVAL_MS)
        cb = lambda: _on_watchdog_tick(owner)
        timeout_sig = getattr(timer, "timeout", None)
        if timeout_sig is not None and hasattr(timeout_sig, "connect"):
            timeout_sig.connect(cb)
        else:
            timer.connect("timeout()", cb)
        owner._codeUpdatePollTimer = timer
        owner._codeUpdateTimerCallback = cb
    return timer


def _kill_job_process(job: dict[str, Any], *, force: bool = True) -> None:
    process = job.get("process")
    pid = 0
    if process is not None:
        try:
            pid = int(process.processId())
        except Exception:
            pid = 0
    if pid > 0 or job.get("pgid"):
        try:
            kill_process_tree(pid, grouped=bool(job.get("grouped")), force=force, pgid=job.get("pgid"))
        except Exception:
            pass
    if process is not None:
        try:
            process.kill()
        except Exception:
            pass


def _drain_worker_output(owner: Any, job: dict[str, Any]) -> None:
    process = job.get("process")
    if process is None:
        return
    try:
        chunk = qbytearray_to_text(process.readAllStandardOutput())
    except Exception:
        return
    if not chunk:
        return
    job["output_tail"] = (str(job.get("output_tail") or "") + chunk)[-_MAX_OUTPUT_TAIL_CHARS:]
    buf = str(job.get("line_buffer") or "") + chunk
    lines = buf.splitlines(keepends=True)
    if lines and not lines[-1].endswith(("\n", "\r")):
        job["line_buffer"] = lines.pop()
    else:
        job["line_buffer"] = ""

    progress_updated = False
    state = getattr(owner, "_codeUpdateState", None)
    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith("SLICERDSPACE_UPDATE_PROGRESS:") and isinstance(state, dict):
            try:
                pct = max(0, min(100, int(line.split(":", 1)[1].strip())))
                if pct != int(state.get("percent") or 0):
                    state["percent"] = pct
                    progress_updated = True
            except ValueError:
                pass
    if progress_updated and isinstance(state, dict) and state.get("phase") == "downloading":
        _sync_idle_texts(owner)


def _on_worker_started(owner: Any, job: dict[str, Any]) -> None:
    if job.get("completed"):
        return
    process = job.get("process")
    if process is not None and job.get("grouped"):
        try:
            job["pgid"] = capture_process_group_id(int(process.processId()))
        except Exception:
            job["pgid"] = None


def _on_watchdog_tick(owner: Any) -> None:
    job = getattr(owner, "_codeUpdateJob", None)
    if not isinstance(job, dict) or job.get("completed"):
        _stop_timer(owner)
        return
    elapsed_exact = time.monotonic() - float(job["started_monotonic"])
    elapsed_sec = max(0, int(elapsed_exact))
    state = getattr(owner, "_codeUpdateState", None)
    if isinstance(state, dict) and elapsed_sec != int(state.get("elapsed_sec") or 0):
        state["elapsed_sec"] = elapsed_sec
        if state.get("phase") == "checking":
            _sync_idle_texts(owner)
    watchdog = float(job.get("watchdog_seconds") or _CHECK_WATCHDOG_SECONDS)
    if elapsed_exact >= watchdog:
        job["timed_out"] = True
        _kill_job_process(job, force=True)
        _finalize_worker(
            owner,
            job,
            exit_code=-1,
            reason=f"Update worker timed out after {int(watchdog)}s.",
        )


def _read_result_payload(result_file: Path) -> dict[str, Any] | None:
    if not result_file.is_file():
        return None
    try:
        payload = json.loads(result_file.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _summarize_failure(job: dict[str, Any], exit_code: int, reason: str | None) -> str:
    if reason:
        base = reason
    else:
        base = f"Update worker exited with status {exit_code}."
    tail_lines = [
        ln.strip()
        for ln in str(job.get("output_tail") or "").splitlines()
        if ln.strip() and not ln.strip().startswith("SLICERDSPACE_UPDATE_")
    ]
    if tail_lines:
        return base + "\n" + "\n".join(tail_lines[-4:])
    return base


def _finalize_worker(
    owner: Any,
    job: dict[str, Any],
    *,
    exit_code: int,
    reason: str | None = None,
) -> None:
    if job.get("completed"):
        return
    job["completed"] = True
    _stop_timer(owner)
    _drain_worker_output(owner, job)

    result_file = Path(job["result_file"])
    payload = _read_result_payload(result_file)

    tmp_dir = job.get("tmp_dir")
    if tmp_dir is not None:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    process = job.get("process")
    if process is not None and hasattr(process, "deleteLater"):
        try:
            process.deleteLater()
        except Exception:
            pass
    if getattr(owner, "_codeUpdateJob", None) is job:
        owner._codeUpdateJob = None

    state = getattr(owner, "_codeUpdateState", None)
    if not isinstance(state, dict):
        return
    state["busy"] = False
    kind = job.get("kind")

    if isinstance(payload, dict) and payload.get("ok") is True and isinstance(payload.get("result"), dict):
        result = payload["result"]
        if kind == "check":
            _handle_check_result(owner, result)
        elif kind == "apply":
            _handle_apply_result(owner, result)
        else:
            _sync_idle_texts(owner)
        return

    if isinstance(payload, dict) and payload.get("ok") is False:
        _show_error(owner, str(payload.get("error") or "Update operation failed."))
        return

    _show_error(owner, _summarize_failure(job, exit_code, reason))


def _on_worker_error(owner: Any, job: dict[str, Any], proc_err: Any) -> None:
    if job.get("completed"):
        return
    try:
        err_code = int(proc_err)
    except Exception:
        err_code = -1
    # FailedToStart (0) does not emit QProcess::finished.
    if err_code == 0:
        _finalize_worker(
            owner,
            job,
            exit_code=-1,
            reason="Failed to start external PythonSlicer update worker.",
        )
        return
    try:
        import qt

        qt.QTimer.singleShot(
            50,
            lambda: _finalize_worker(
                owner,
                job,
                exit_code=-1,
                reason=f"Update worker process error ({err_code}).",
            ),
        )
    except Exception:
        _finalize_worker(owner, job, exit_code=-1, reason=f"Update worker process error ({err_code}).")


def _start_worker_process(
    owner: Any,
    *,
    kind: str,
    worker_args: list[str],
    watchdog_seconds: int,
    tmp_dir: Path,
    result_file: Path,
) -> None:
    import qt

    program, args = _resolve_worker_command(owner, worker_args)
    launch_program, launch_args, grouped = grouped_launch_command(program, args)

    process = qt.QProcess(_parent_window(owner))
    process.setProcessChannelMode(qt.QProcess.MergedChannels)
    try:
        env = qt.QProcessEnvironment.systemEnvironment()
        try:
            for key, value in network_env_overrides().items():
                env.insert(key, str(value))
        except Exception:
            pass
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        process.setProcessEnvironment(env)
    except Exception:
        pass

    job: dict[str, Any] = {
        "kind": kind,
        "process": process,
        "tmp_dir": tmp_dir,
        "result_file": result_file,
        "output_tail": "",
        "line_buffer": "",
        "pgid": None,
        "grouped": grouped,
        "watchdog_seconds": watchdog_seconds,
        "started_monotonic": time.monotonic(),
        "timed_out": False,
        "completed": False,
    }
    cb_started = lambda: _on_worker_started(owner, job)
    cb_output = lambda: _drain_worker_output(owner, job)
    cb_finished = lambda code, _exit_status=0: _finalize_worker(owner, job, exit_code=int(code))
    cb_error = lambda proc_err: _on_worker_error(owner, job, proc_err)
    job["callbacks"] = (cb_started, cb_output, cb_finished, cb_error)

    owner._codeUpdateJob = job
    process.connect("started()", cb_started)
    process.connect("readyReadStandardOutput()", cb_output)
    process.connect("finished(int,QProcess::ExitStatus)", cb_finished)
    process.connect("errorOccurred(QProcess::ProcessError)", cb_error)

    _ensure_timer(owner).start()
    process.start(launch_program, launch_args)


def _start_check_worker(owner: Any) -> None:
    state = owner._codeUpdateState
    if state.get("busy"):
        return
    state["busy"] = True
    state["phase"] = "checking"
    state["elapsed_sec"] = 0
    _sync_idle_texts(owner)
    _log(owner, "[Updater] Checking for code updates...")

    tmp_dir = Path(tempfile.mkdtemp(prefix="slicerdspace_upd_chk_"))
    result_file = tmp_dir / "result.json"
    module_root = Path(__file__).resolve().parents[1]
    _start_worker_process(
        owner,
        kind="check",
        worker_args=[
            "check",
            "--module-root",
            str(module_root),
            "--result-file",
            str(result_file),
        ],
        watchdog_seconds=_CHECK_WATCHDOG_SECONDS,
        tmp_dir=tmp_dir,
        result_file=result_file,
    )


def _start_apply_worker(owner: Any, remote_manifest: dict[str, Any]) -> None:
    state = owner._codeUpdateState
    if state.get("busy"):
        return
    state["busy"] = True
    state["phase"] = "downloading"
    state["percent"] = 0
    _sync_idle_texts(owner)
    version = str(remote_manifest.get("version") or EXTENSION_VERSION)
    _log(owner, f"[Updater] Downloading and applying code update v{version}...")

    tmp_dir = Path(tempfile.mkdtemp(prefix="slicerdspace_upd_app_"))
    manifest_file = tmp_dir / "manifest.json"
    result_file = tmp_dir / "result.json"
    manifest_file.write_text(json.dumps(remote_manifest, ensure_ascii=False), encoding="utf-8")
    module_root = Path(__file__).resolve().parents[1]
    _start_worker_process(
        owner,
        kind="apply",
        worker_args=[
            "apply",
            "--module-root",
            str(module_root),
            "--manifest-file",
            str(manifest_file),
            "--result-file",
            str(result_file),
        ],
        watchdog_seconds=_APPLY_WATCHDOG_SECONDS,
        tmp_dir=tmp_dir,
        result_file=result_file,
    )


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
    owner._codeUpdateJob = None
    owner._codeUpdateState = {
        "phase": "idle",
        "version": EXTENSION_VERSION,
        "size_bytes": 0,
        "percent": 0,
        "elapsed_sec": 0,
        "busy": False,
    }
    _sync_idle_texts(owner)
    return bar


def refresh_update_panel_language(owner: Any, language: str | None = None) -> None:
    """Update the code update bar text when the UI language switches."""

    if getattr(owner, "_codeUpdateBar", None) is None:
        return
    _sync_idle_texts(owner, zh=_is_zh(owner, language))


def cleanup_update_panel(owner: Any) -> None:
    """Stop active update timer/worker process when closing the module."""

    _stop_timer(owner)
    job = getattr(owner, "_codeUpdateJob", None)
    if isinstance(job, dict) and not job.get("completed"):
        job["completed"] = True
        _kill_job_process(job, force=True)
        tmp_dir = job.get("tmp_dir")
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        owner._codeUpdateJob = None
    state = getattr(owner, "_codeUpdateState", None)
    if isinstance(state, dict):
        state["busy"] = False
