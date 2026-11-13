"""Surface workflow widget glue for SlicerDSpace."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import slicer

from shared.ui_input_inspection import inspect_input_directory

LOGGER = logging.getLogger(__name__)


class SurfaceWidgetMixin:
    """Widget handlers for the surface workflow and its post-processing tools."""

    def _connect_surface_ui(self) -> None:
        if hasattr(self.ui, "applyButton"):
            self.ui.applyButton.connect("clicked()", self.onApplyButton)
        if hasattr(self.ui, "inputDirectoryPath"):
            self.ui.inputDirectoryPath.currentPathChanged.connect(self._onInputDirectoryChanged)
        if hasattr(self.ui, "outputDirectoryPath"):
            self.ui.outputDirectoryPath.currentPathChanged.connect(self._onOutputDirectoryChanged)
        if hasattr(self.ui, "predictModeComboBox"):
            self.ui.predictModeComboBox.currentTextChanged.connect(self._onPredictModeChanged)
        if hasattr(self.ui, "deviceComboBox"):
            self.ui.deviceComboBox.currentTextChanged.connect(self._onDeviceChanged)

        if hasattr(self.ui, "postprocessSubjectLineEdit"):
            self.ui.postprocessSubjectLineEdit.textChanged.connect(self._onPostprocessSubjectChanged)
        if hasattr(self.ui, "postprocessNativePath"):
            self.ui.postprocessNativePath.currentPathChanged.connect(self._onPostprocessNativeDirChanged)
        if hasattr(self.ui, "postprocessMriPath"):
            self.ui.postprocessMriPath.currentPathChanged.connect(self._onPostprocessMriDirChanged)
        if hasattr(self.ui, "postprocessRepoPath"):
            self.ui.postprocessRepoPath.currentPathChanged.connect(self._onPostprocessRepoChanged)
        if hasattr(self.ui, "postprocessLicensePath"):
            self.ui.postprocessLicensePath.currentPathChanged.connect(self._onPostprocessLicenseChanged)
        if hasattr(self.ui, "postprocessModeComboBox"):
            self.ui.postprocessModeComboBox.currentTextChanged.connect(self._onPostprocessModeChanged)
        if hasattr(self.ui, "postprocessImageLineEdit"):
            self.ui.postprocessImageLineEdit.textChanged.connect(self._onPostprocessImageChanged)
        if hasattr(self.ui, "postprocessForceCheckBox"):
            self.ui.postprocessForceCheckBox.toggled.connect(self._onPostprocessForceToggled)
        if hasattr(self.ui, "runPostprocessButton"):
            self.ui.runPostprocessButton.connect("clicked()", self._onRunLocalPostprocess)

    def _onOutputDirectoryChanged(self, newPath: str) -> None:
        if not self._update_parameter_node("outputDirectory", newPath or ""):
            return
        self._checkCanApply()

    def _onInputDirectoryChanged(self, newPath: str) -> None:
        if not self._parameterNode:
            return
        from shared.ui_input_defaults import subject_folder_defaults
        node = self._parameterNode
        root, subject = subject_folder_defaults(newPath, node.subjectId, batch=bool(getattr(node, "batchMode", False)))
        with slicer.util.NodeModify(node):
            self._update_parameter_node("inputDirectory", root)
            self._update_parameter_node("subjectId", subject)
        self._checkCanApply()

    def _onPredictModeChanged(self, newText: str) -> None:
        if not self._update_parameter_node("predictMode", newText or "all"):
            return
        self._checkCanApply()

    def _onDeviceChanged(self, newText: str) -> None:
        if not self._update_parameter_node("device", newText or "auto"):
            return
        self._checkCanApply()

    def _onPostprocessSubjectChanged(self, newText: str) -> None:
        self._update_parameter_node("postprocessSubjectId", newText or "")

    def _onPostprocessNativeDirChanged(self, path: str) -> None:
        self._update_parameter_node("postprocessNativeDir", path or "")

    def _onPostprocessMriDirChanged(self, path: str) -> None:
        self._update_parameter_node("postprocessMriDir", path or "")

    def _onPostprocessRepoChanged(self, path: str) -> None:
        self._update_parameter_node("postprocessRepoDir", path or "")

    def _onPostprocessLicenseChanged(self, path: str) -> None:
        self._update_parameter_node("postprocessLicensePath", path or "")

    def _onPostprocessModeChanged(self, text: str) -> None:
        self._update_parameter_node("postprocessMode", text or "whole")

    def _onPostprocessImageChanged(self, text: str) -> None:
        self._update_parameter_node("postprocessDockerImage", text or "")

    def _onPostprocessForceToggled(self, state: bool) -> None:
        self._update_parameter_node("postprocessForce", bool(state))

    def _onRunLocalPostprocess(self) -> None:
        if not self._parameterNode or not self.logic:
            return

        subject = (self._parameterNode.postprocessSubjectId or self._parameterNode.subjectId or "").strip()
        native_dir = (self._parameterNode.postprocessNativeDir or "").strip()
        mri_dir = (self._parameterNode.postprocessMriDir or "").strip()
        repo_dir = (self._parameterNode.postprocessRepoDir or "").strip()
        mode = (self._parameterNode.postprocessMode or "whole").strip() or "whole"
        force = bool(self._parameterNode.postprocessForce)

        missing: List[str] = []
        if not subject:
            missing.append("Subject ID")
        paths_to_check: List[tuple[str, str, bool]] = [
            (native_dir, "Native output directory", False),
            (mri_dir, "MRI directory", False),
        ]
        if repo_dir:
            paths_to_check.append((repo_dir, "DDSurfer postprocess backend directory", False))
        resolved_paths: dict[str, Path] = {}
        for path_str, label, require_file in paths_to_check:
            if not path_str:
                continue
            path = Path(path_str).expanduser()
            if require_file and not path.is_file():
                missing.append(label)
                continue
            if not require_file and not path.is_dir():
                missing.append(label)
                continue
            resolved_paths[label] = path

        if missing:
            slicer.util.errorDisplay(
                "Cannot run DDSurfer postprocess. Please provide the following:\n- " + "\n- ".join(missing),
                parent=self.parent,
            )
            return

        def local_log(message: str) -> None:
            self.appendLog(f"[DDSurfer Postprocess] {message}")

        slicer.util.showStatusMessage("Running DDSurfer postprocess...", 0)
        self._setStatus("running", "DDSurfer postprocess in progress")
        with slicer.util.tryWithErrorDisplay("DDSurfer postprocess failed.", waitCursor=True):
            self.appendLog("Starting DDSurfer postprocess...")
            self.logic.runLocalPostprocess(
                subject_id=subject,
                native_dir=resolved_paths["Native output directory"],
                mri_dir=resolved_paths["MRI directory"],
                repo_path=resolved_paths.get("DDSurfer postprocess backend directory"),
                mode=mode,
                force=force,
                log_callback=local_log,
            )
            self.appendLog("DDSurfer postprocess completed.")
            slicer.util.showStatusMessage("DDSurfer postprocess completed.", 5000)
        self._setStatus("ready", "Pipeline ready")

    def onApplyButton(self) -> None:
        if not self._parameterNode or not self.logic:
            return

        if not self.validateInputs(manual=True):
            slicer.util.showStatusMessage("Input validation failed. See log for details.", 5000)
            return

        slicer.util.showStatusMessage("SlicerDSpace pipeline running...", 0)
        with slicer.util.tryWithErrorDisplay("SlicerDSpace processing failed.", waitCursor=True):
            self._setPipelineActive(True)
            try:
                self.appendLog("Starting SlicerDSpace pipeline...")
                self.logic.process(self._parameterNode, logCallback=self.appendLog)
                self.appendLog("SlicerDSpace pipeline completed.")
                slicer.util.showStatusMessage("SlicerDSpace pipeline completed.", 5000)
            finally:
                self._setPipelineActive(False)

    def _evaluateInputs(self, *, refresh: bool = False) -> tuple[bool, List[str]]:
        node = self._parameterNode
        if not node:
            return False, ["Parameter node unavailable"]

        if getattr(node, 'experimentsEnabled', False) and getattr(node, 'experimentScope', 'subject') == 'cohort':
            root = str(getattr(node, 'experimentRoot', '')).strip()
            return (True, []) if root and Path(root).expanduser().is_dir() else (False, ['Cohort root'])

        if getattr(node, "batchMode", False):
            root = (getattr(node, "batchRoot", "") or "").strip()
            return (True, []) if root and Path(root).expanduser().is_dir() else (False, ["Batch folder"])

        missing: List[str] = []
        subject_id = node.subjectId.strip()
        input_dir_str = node.inputDirectory.strip()
        output_dir_str = node.outputDirectory.strip()

        if not subject_id:
            missing.append("Subject ID")

        if not input_dir_str:
            self._lastAutoDetectedInputSignature = None
            missing.append("Subject directory")
            return False, missing

        candidate = Path(input_dir_str).expanduser()
        if not (candidate.is_dir() or candidate.is_file()):
            missing.append("Subject directory")
            return False, missing

        inspected = inspect_input_directory(self, candidate, subject_id, refresh=refresh)
        mode = inspected.get("mode")
        if mode == "missing":
            missing.append("Subject input layout")
        else:
            from shared.ui_compact import sync_detected_input_format

            sync_detected_input_format(self, mode, signature=(input_dir_str, subject_id))
            if mode == "legacy_registered":
                if not output_dir_str:
                    missing.append("Output directory")
                expected_files = [
                    f"{subject_id}-dti-FractionalAnisotropy-Reg.nii.gz",
                    f"{subject_id}-dti-MinEigenvalue-Reg.nii.gz",
                    f"{subject_id}-dti-MidEigenvalue-Reg.nii.gz",
                    f"{subject_id}-dti-MaxEigenvalue-Reg.nii.gz",
                    f"{subject_id}-dti-Trace-Reg.nii.gz",
                    f"{subject_id}-dti-MeanDiffusivity-Reg.nii.gz",
                    f"{subject_id}-mask-Reg.nii.gz",
                    f"{subject_id}-b0ToAtlasT2.tfm",
                ]
                subject_dir = Path(inspected["subject_dir"])
                for name in expected_files:
                    if not (subject_dir / name).exists():
                        missing.append(name)

        return (len(missing) == 0), missing

    def validateInputs(self, manual: bool = False, *, update_status: bool = True) -> bool:
        ok, missing = self._evaluateInputs(refresh=manual)
        status_message = "Input layout found. Dependencies are checked when you run." if ok else self._summarise_missing(missing)
        if manual:
            self._inputValidationAttempted = True
        if ok:
            self._inputValidationAttempted = False
        show_errors = not ok and bool(getattr(self, "_inputValidationAttempted", False))

        if not self._pipelineActive:
            if update_status:
                self._setStatus("error" if show_errors else "waiting", status_message)
            if hasattr(self.ui, "applyButton"):
                tooltip = "Run the SlicerDSpace pipeline" if ok else "Waiting: " + ", ".join(missing)
                self.ui.applyButton.setToolTip(tooltip)

        if hasattr(self.ui, "applyButton"):
            self.ui.applyButton.enabled = ok and not self._pipelineActive
        if hasattr(self.ui, "validateButton"):
            self.ui.validateButton.enabled = not self._pipelineActive

        # Empty, untouched fields are pending input, not a validation failure.
        self._setWidgetError(getattr(self.ui, "subjectIdLineEdit", None), show_errors and "Subject ID" in missing)
        subject_token = ""
        node = self._parameterNode
        if node:
            subject_token = f"{node.subjectId.strip()} directory"
        file_errors = False
        if node and node.subjectId.strip():
            prefix = f"{node.subjectId.strip()}-"
            file_errors = any(
                entry.startswith(prefix)
                for entry in missing
                if entry.endswith(".nii.gz") or entry.endswith(".tfm")
            )
        self._setWidgetError(
            getattr(self.ui, "inputDirectoryPath", None),
            (show_errors or bool(getattr(node, "inputDirectory", ""))) and (
                any(token in missing for token in ("Subject directory", "Subject input layout", subject_token))
                or file_errors
            ),
        )
        self._setWidgetError(getattr(self.ui, "outputDirectoryPath", None), "Output directory" in missing)

        if update_status and status_message != self._lastStatusMessage and not self._pipelineActive:
            self._lastStatusMessage = status_message
            self.appendLog(status_message)

        if manual:
            if ok:
                self.appendLog("Validation succeeded: all required inputs are available.")
            else:
                self.appendLog("Validation failed. Missing entries: " + ", ".join(missing))

        return ok
