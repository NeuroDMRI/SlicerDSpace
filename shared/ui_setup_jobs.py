"""Non-blocking, cancellable setup jobs. Qt callbacks always run on the UI thread."""

from pathlib import Path
from datetime import datetime

from shared.ui_output_loading import qbytearray_to_text
from shared.process_group import capture_process_group_id, grouped_launch_command, kill_process_tree


class SetupJob:
    def __init__(self, owner, title, commands, on_done, *, show_dialog: bool = True):
        import qt
        import slicer

        self.owner, self.commands, self.on_done = owner, commands, on_done
        self.title = title
        self.show_dialog = bool(show_dialog)
        self.index = 0
        self.cancelled = self.finished = False
        self.tail = ""
        self.grouped = False
        self.pgid = None
        report_dir = getattr(owner, "_latestEnvironmentReportDir", None)
        if not report_dir:
            from shared.user_config import default_user_config_path
            report_dir = default_user_config_path().parent / "setup-logs"
        folder = Path(report_dir)
        folder.mkdir(parents=True, exist_ok=True)
        owner._latestEnvironmentReportDir = folder
        self.log_path = folder / ("setup-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f") + ".log")
        self.log = self.log_path.open("w", encoding="utf-8")
        self.dialog = None
        if self.show_dialog:
            self.dialog = qt.QProgressDialog(title, "Cancel", 0, 0, slicer.util.mainWindow())
            self.dialog.setWindowTitle(title)
            self.dialog.setWindowModality(qt.Qt.NonModal)
            self.dialog.setAutoClose(False)
            self.dialog.setMinimumDuration(0)
            self.dialog.connect("canceled()", self.cancel)
        self.process = qt.QProcess(owner.parent)
        self.process.setProcessChannelMode(qt.QProcess.MergedChannels)
        self.process.connect("readyReadStandardOutput()", self.read)
        self.process.connect("finished(int,QProcess::ExitStatus)", self.step_finished)
        self.process.connect("errorOccurred(QProcess::ProcessError)", self.error)
        self.process.connect("started()", self.started)
        if self.dialog is not None:
            self.dialog.show()

    def _set_dialog_text(self, text: str) -> None:
        if self.dialog is not None:
            self.dialog.setLabelText(text)

    def start_next(self):
        import qt
        if self.cancelled:
            self.finish("cancelled")
            return
        if self.index >= len(self.commands):
            self.finish("completed")
            return
        program, args, environment = self.commands[self.index]
        env = qt.QProcessEnvironment.systemEnvironment()
        try:
            from shared.network_config import network_env_overrides

            for key, value in network_env_overrides().items():
                env.insert(key, str(value))
        except Exception:
            pass
        for key, value in environment.items():
            if value is None:
                env.remove(key)
            else:
                env.insert(key, str(value))
        env.insert("PYTHONUNBUFFERED", "1")
        self.process.setProcessEnvironment(env)
        program, args, self.grouped = grouped_launch_command(program, args)
        self.pgid = None
        command_text = repr([program, *args])
        self.log.write(command_text + "\n")
        self.log.flush()
        self.owner.appendLog("[Setup] " + command_text)
        self._set_dialog_text(f"Step {self.index + 1}/{len(self.commands)}\nStarting...\nLog: {self.log_path}")
        self.process.start(program, args)

    def started(self):
        self.pgid = capture_process_group_id(int(self.process.processId())) if self.grouped else None
        if self.cancelled:
            self.signal_process(False)
            self.schedule_force_stop()

    def schedule_force_stop(self):
        import qt
        pid, grouped, pgid = int(self.process.processId()), self.grouped, self.pgid
        if pid <= 0:
            return
        # Do not re-read QProcess or the job state after its leader exits.
        def stop():
            self.stop_tree(pid, grouped, pgid, True)
            if self.cancelled and not self.finished:
                self.finish("cancelled")
        qt.QTimer.singleShot(2000, stop)

    def stop_tree(self, pid, grouped, pgid, force):
        try:
            kill_process_tree(pid, grouped=grouped, pgid=pgid, force=force)
        except OSError:
            pass

    def read(self):
        if self.finished:
            return
        text = qbytearray_to_text(self.process.readAllStandardOutput())
        if not text:
            return
        self.log.write(text)
        self.log.flush()
        self.tail = (self.tail + text)[-6000:]
        lines = text.strip().splitlines()
        if lines and not self.cancelled:
            self._set_dialog_text(f"Step {self.index + 1}/{len(self.commands)}\n{lines[-1][-220:]}\nLog: {self.log_path}")

    def cancel(self):
        if self.finished or self.cancelled:
            return
        self.cancelled = True
        self._set_dialog_text("Cancelling... waiting for the process to stop.")
        self.signal_process(False)
        self.schedule_force_stop()

    def shutdown(self):
        """Stop owned processes during widget teardown without UI callbacks."""
        if self.finished:
            return
        self.read()
        self.cancelled = self.finished = True
        pid, grouped, pgid = int(self.process.processId()), self.grouped, self.pgid
        self.signal_process(False)
        if not self.process.waitForFinished(2000):
            self.signal_process(True)
            self.process.waitForFinished(3000)
        self.stop_tree(pid, grouped, pgid, True)
        self.log.write("\nSTATUS: cancelled (widget teardown)\n")
        self.log.close()
        self.owner._setupJob = None
        self.owner._pipelineActive = False
        if self.dialog is not None:
            self.dialog.close()
        self.process.deleteLater()
        if self.dialog is not None:
            self.dialog.deleteLater()

    def signal_process(self, force):
        pid = int(self.process.processId())
        self.stop_tree(pid, self.grouped, self.pgid, force)

    def step_finished(self, code, exit_status):
        if self.finished:
            return
        self.read()
        if self.cancelled:
            # The launcher's exit does not prove its descendants have stopped.
            # The recorded force-stop timer completes cancellation after cleanup.
            return
        elif code != 0 or int(exit_status) != 0:
            self.finish("failed")
        else:
            self.index += 1
            self.start_next()

    def error(self, error):
        import qt
        if error == qt.QProcess.FailedToStart:
            self.tail += "\n" + self.process.errorString()
            self.finish("cancelled" if self.cancelled else "failed")

    def finish(self, status):
        if self.finished:
            return
        self.finished = True
        self.log.write(f"\nSTATUS: {status}\n")
        self.log.close()
        if self.dialog is not None:
            self.dialog.close()
        self.owner._setPipelineActive(False)
        self.owner.appendLog(f"[Setup] {status}. Log: {self.log_path}")
        self.owner._setupJob = None
        try:
            self.on_done(status, self.tail, self.log_path)
        finally:
            if (
                getattr(self.owner, "_setupJob", None) is None
                and not getattr(self.owner, "_pipelineActive", False)
                and not getattr(self.owner, "_selectedSetupActive", False)
                and not getattr(self.owner, "_inputImportPending", False)
            ):
                setter = getattr(self.owner, "_set_selection_controls_enabled", None)
                if callable(setter):
                    try:
                        setter(True)
                    except Exception:
                        pass
                refresher = getattr(self.owner, "_refresh_run_button_state", None)
                if callable(refresher):
                    try:
                        refresher()
                    except Exception:
                        pass
                if getattr(self.owner, "_runState", None) in ("preflight", "setup"):
                    try:
                        from shared.ui_run_history import set_run_state

                        set_run_state(self.owner, "idle")
                    except Exception:
                        try:
                            self.owner._runState = "idle"
                        except Exception:
                            pass
            self.process.deleteLater()
            if self.dialog is not None:
                self.dialog.deleteLater()


def start_setup_job(owner, title, commands, on_done, *, show_dialog: bool = True):
    import slicer
    if getattr(owner, "_setupJob", None) or getattr(owner, "_pipelineActive", False):
        slicer.util.infoDisplay("Wait for the current job to finish or cancel it first.", parent=owner.parent)
        return False
    owner._setPipelineActive(True)
    if hasattr(owner, "_setStatus"):
        owner._setStatus("running", title + " (background)")
    try:
        owner._setupJob = SetupJob(owner, title, commands, on_done, show_dialog=show_dialog)
        owner._setupJob.start_next()
    except Exception:
        owner._setupJob = None
        owner._setPipelineActive(False)
        raise
    return True
