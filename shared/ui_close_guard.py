"""Ask before the Slicer main window closes while a DSpace job runs.

``qt`` and ``slicer`` are imported lazily so unit tests run without Slicer.
The guard fails open: an error inside it never traps the user in Slicer.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Mapping

from shared.ui_background_process import qprocess_is_running

ENV_VAR = "SLICERDSPACE_CLOSE_GUARD"
_OFF_VALUES = {"0", "false", "off", "no"}
_ON_VALUES = {"1", "true", "on", "yes"}
_READ_ONLY_UPDATE_KINDS = frozenset({"check", "check_local"})


def running_jobs(owner: Any) -> list[dict[str, Any]]:
    """Return one dict per running DSpace job; each has a ``kind`` key."""

    jobs: list[dict[str, Any]] = []
    process = getattr(owner, "_backgroundProcess", None)
    if process is not None and qprocess_is_running(process):
        job: dict[str, Any] = {"kind": "run"}
        context = getattr(owner, "_backgroundRunContext", None)
        run_name = context.get("run_name") if isinstance(context, dict) else None
        if run_name:
            job["run_name"] = run_name
        jobs.append(job)
    if getattr(owner, "_setupJob", None):
        jobs.append({"kind": "setup"})
    if getattr(owner, "_inputImportPending", False) or getattr(owner, "_importProcess", None) is not None:
        jobs.append({"kind": "import"})
    update_job = getattr(owner, "_codeUpdateJob", None)
    # A "check" or "check_local" worker only reads and writes nothing to the install, so a close need not ask.
    # "apply" downloads and writes; a missing or unknown kind keeps counting (safe default).
    if (
        isinstance(update_job, dict)
        and not update_job.get("completed")
        and update_job.get("kind") not in _READ_ONLY_UPDATE_KINDS
    ):
        jobs.append({"kind": "update"})
    return jobs


def close_guard_enabled(environ: Mapping[str, str] | None = None, app: Any = None) -> bool:
    """Return True when the close guard is active (off in Slicer testing mode by default)."""

    env = os.environ if environ is None else environ
    value = str(env.get(ENV_VAR, "")).strip().lower()
    if value in _OFF_VALUES:
        return False
    if value in _ON_VALUES:
        return True
    try:
        if app is None:
            import slicer

            app = slicer.app
        return not (app.testingEnabled() is True)
    except Exception:
        return True


def may_close(owner: Any, *, action: str = "quit", ask: Callable[[Any, list[dict[str, Any]], str], bool] | None = None) -> bool:
    """Return True when closing may proceed; False when the user chose to keep running."""

    try:
        if getattr(owner, "_closeGuardAsking", False) is True:
            return False
        jobs = running_jobs(owner)
        if not jobs or not close_guard_enabled():
            return True
        question = ask if ask is not None else _ask_with_message_box
        owner._closeGuardAsking = True
        try:
            keep_running = question(owner, jobs, action)
        finally:
            owner._closeGuardAsking = False
        return not keep_running
    except Exception:
        return True


def handle_close_event(owner: Any, event: Any, *, ask: Callable[[Any, list[dict[str, Any]], str], bool] | None = None) -> bool:
    """Return True when the close event was consumed (the user chose to keep running)."""

    try:
        if may_close(owner, action="quit", ask=ask):
            return False
        event.ignore()
        return True
    except Exception:
        return False


def install_close_guard(owner: Any) -> bool:
    """Install a close-event filter on the main window; idempotent."""

    try:
        if getattr(owner, "_closeGuardFilter", None) is not None:
            return True
        import qt
        import slicer

        window = slicer.util.mainWindow()
        if window is None:
            return False

        class CloseGuardEventFilter(qt.QWidget):
            def eventFilter(self, obj, event):
                if event.type() == qt.QEvent.Close:
                    return handle_close_event(owner, event)
                return False

        event_filter = CloseGuardEventFilter()
        window.installEventFilter(event_filter)
        # Keep strong references: an unreferenced filter is garbage-collected and stops working.
        owner._closeGuardFilter = event_filter
        owner._closeGuardWindow = window
        return True
    except Exception:
        return False


def uninstall_close_guard(owner: Any) -> None:
    """Remove the close-event filter; never raises."""

    event_filter = getattr(owner, "_closeGuardFilter", None)
    window = getattr(owner, "_closeGuardWindow", None)
    try:
        if event_filter is not None and window is not None:
            window.removeEventFilter(event_filter)
    except Exception:
        pass
    try:
        delete_later = getattr(event_filter, "deleteLater", None)
        if callable(delete_later):
            delete_later()
    except Exception:
        pass
    try:
        owner._closeGuardFilter = None
        owner._closeGuardWindow = None
    except Exception:
        pass


def _job_label(job: dict[str, Any], zh: bool) -> str:
    kind = job.get("kind")
    run_name = job.get("run_name")
    if kind == "run":
        if zh:
            return f"运行“{run_name}”" if run_name else "流程运行"
        return f"run '{run_name}'" if run_name else "pipeline run"
    labels = {
        "setup": ("environment setup", "环境准备"),
        "import": ("input import", "输入导入"),
        "update": ("code update", "代码更新"),
    }
    en_label, zh_label = labels.get(str(kind), (str(kind), str(kind)))
    return zh_label if zh else en_label


def guard_message(jobs: list[dict[str, Any]], action: str, zh: bool) -> dict[str, str]:
    """Build the title, text and button labels of the question (pure, no Qt)."""

    restart = action == "restart"
    labels = ("、" if zh else ", ").join(_job_label(job, zh) for job in jobs)
    has_run = any(job.get("kind") == "run" for job in jobs)
    if zh:
        verb = "重启" if restart else "关闭"
        text = f"DSpace 任务正在运行：{labels}。\n\n现在{verb} Slicer 会停止该任务。"
        if has_run:
            text += "已完成的阶段会保留，正在进行的阶段下次要从头重跑。"
        return {
            "title": "SlicerDSpace 仍在运行",
            "text": text,
            "keep": "继续运行",
            "leave": "停止并重启" if restart else "停止并退出",
        }
    verb = "restarts" if restart else "closes"
    text = f"A DSpace job is running: {labels}.\n\nIf Slicer {verb} now, the job stops."
    if has_run:
        text += " Finished stages are kept. The stage in progress starts again from its beginning next time."
    return {
        "title": "SlicerDSpace is still working",
        "text": text,
        "keep": "Keep Running",
        "leave": "Stop and Restart" if restart else "Stop and Quit",
    }


def _ask_with_message_box(owner: Any, jobs: list[dict[str, Any]], action: str) -> bool:
    """Ask the user; return True to keep running (the default and escape answer)."""

    import qt
    import slicer

    language = getattr(owner, "_onboardingLanguage", "en")
    zh = isinstance(language, str) and language.startswith("zh")
    message = guard_message(jobs, action, zh)
    box = qt.QMessageBox(slicer.util.mainWindow())
    box.setIcon(qt.QMessageBox.Warning)
    box.setWindowTitle(message["title"])
    box.setText(message["text"])
    keep_button = box.addButton(message["keep"], qt.QMessageBox.RejectRole)
    leave_button = box.addButton(message["leave"], qt.QMessageBox.DestructiveRole)
    box.setDefaultButton(keep_button)
    box.setEscapeButton(keep_button)
    box.exec_()
    return box.clickedButton() != leave_button
