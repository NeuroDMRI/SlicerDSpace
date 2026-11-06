"""Segmentation workflow widget glue for SlicerDSpace."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import slicer


class SegmentationWidgetMixin:
    """Widget handlers for the DDSeg workflow."""

    def _connect_segmentation_ui(self) -> None:
        if hasattr(self.ui, "ddsegMaskPath"):
            self.ui.ddsegMaskPath.currentPathChanged.connect(self._onDDSegMaskChanged)
        if hasattr(self.ui, "ddsegDwiPath"):
            self.ui.ddsegDwiPath.currentPathChanged.connect(self._onDDSegDwiChanged)
        if hasattr(self.ui, "ddsegBvalPath"):
            self.ui.ddsegBvalPath.currentPathChanged.connect(self._onDDSegBvalChanged)
        if hasattr(self.ui, "ddsegBvecPath"):
            self.ui.ddsegBvecPath.currentPathChanged.connect(self._onDDSegBvecChanged)
        if hasattr(self.ui, "ddsegOutputPath"):
            self.ui.ddsegOutputPath.currentPathChanged.connect(self._onDDSegOutputChanged)
        if hasattr(self.ui, "ddsegWeightsPath"):
            self.ui.ddsegWeightsPath.currentPathChanged.connect(self._onDDSegWeightsChanged)
        if hasattr(self.ui, "ddsegSlicerBasePath"):
            self.ui.ddsegSlicerBasePath.currentPathChanged.connect(self._onDDSegSlicerBaseChanged)
        if hasattr(self.ui, "ddsegSlicerExtPath"):
            self.ui.ddsegSlicerExtPath.currentPathChanged.connect(self._onDDSegSlicerExtChanged)
        if hasattr(self.ui, "ddsegDeviceComboBox"):
            self.ui.ddsegDeviceComboBox.currentTextChanged.connect(self._onDDSegDeviceChanged)
        if hasattr(self.ui, "ddsegSoftmaxCheckBox"):
            self.ui.ddsegSoftmaxCheckBox.toggled.connect(self._onDDSegSoftmaxChanged)
        if hasattr(self.ui, "runDDSegButton"):
            self.ui.runDDSegButton.connect("clicked()", self._onRunDDSegPipeline)

    def _onDDSegMaskChanged(self, path: str) -> None:
        self._update_parameter_node("ddsegMaskPath", path or "")

    def _onDDSegDwiChanged(self, path: str) -> None:
        self._update_parameter_node("ddsegDwiPath", path or "")

    def _onDDSegBvalChanged(self, path: str) -> None:
        self._update_parameter_node("ddsegBvalPath", path or "")

    def _onDDSegBvecChanged(self, path: str) -> None:
        self._update_parameter_node("ddsegBvecPath", path or "")

    def _onDDSegOutputChanged(self, path: str) -> None:
        self._update_parameter_node("ddsegOutputDir", path or "")

    def _onDDSegWeightsChanged(self, path: str) -> None:
        self._update_parameter_node("ddsegWeightsDir", path or "")

    def _onDDSegSlicerBaseChanged(self, path: str) -> None:
        self._update_parameter_node("ddsegSlicerBase", path or "")

    def _onDDSegSlicerExtChanged(self, path: str) -> None:
        self._update_parameter_node("ddsegSlicerExt", path or "")

    def _onDDSegDeviceChanged(self, value: str) -> None:
        self._update_parameter_node("ddsegDevice", value or "auto")

    def _onDDSegSoftmaxChanged(self, value: bool) -> None:
        self._update_parameter_node("ddsegApplySoftmax", bool(value))

    def _onRunDDSegPipeline(self) -> None:
        if not self._parameterNode or not self.logic:
            return

        mask_path = (self._parameterNode.ddsegMaskPath or "").strip()
        dwi_path = (self._parameterNode.ddsegDwiPath or "").strip()
        bval_path = (self._parameterNode.ddsegBvalPath or "").strip()
        bvec_path = (self._parameterNode.ddsegBvecPath or "").strip()
        output_dir = (self._parameterNode.ddsegOutputDir or "").strip()
        weights_dir = (self._parameterNode.ddsegWeightsDir or "").strip()
        slicer_base = (self._parameterNode.ddsegSlicerBase or "").strip()
        slicer_ext = (self._parameterNode.ddsegSlicerExt or "").strip()
        device = (self._parameterNode.ddsegDevice or "auto").strip()
        apply_softmax = bool(self._parameterNode.ddsegApplySoftmax)
        subject = (self._parameterNode.subjectId or "").strip()

        subject_root_mode = False
        subject_root_path: Optional[Path] = None
        if dwi_path:
            dwi_candidate = Path(dwi_path).expanduser()
            if dwi_candidate.is_dir() and not any([mask_path, bval_path, bvec_path]):
                subject_root_mode = True
                subject_root_path = dwi_candidate

        missing: List[str] = []
        if subject_root_mode:
            if not subject:
                missing.append("Subject ID")
            if subject_root_path is None:
                missing.append("DDSeg subject directory")
            for label, value in (("DDSeg weights", weights_dir),):
                if not value:
                    missing.append(label)
        else:
            for label, value in (
                ("Mask NIfTI", mask_path),
                ("DWI NIfTI", dwi_path),
                ("bval", bval_path),
                ("bvec", bvec_path),
                ("DDSeg output", output_dir),
                ("DDSeg weights", weights_dir),
                ("Slicer base", slicer_base),
            ):
                if not value:
                    missing.append(label)

        if missing:
            slicer.util.errorDisplay(
                "Cannot run DDSeg. Please provide the following:\n- " + "\n- ".join(sorted(set(missing))),
                parent=self.parent,
            )
            return

        self.appendLog("Starting DDSeg pipeline...")
        self._setStatus("running", "DDSeg pipeline in progress")
        slicer.util.showStatusMessage("DDSeg pipeline running...", 0)

        with slicer.util.tryWithErrorDisplay("DDSeg pipeline failed.", waitCursor=True):
            self.logic.setupDDSegRequirements()
            if subject_root_mode and subject_root_path is not None:
                self.logic.runDDSegPipeline(
                    subject_root=subject_root_path,
                    subject_id=subject,
                    output_dir=Path(output_dir).expanduser() if output_dir else None,
                    weights_dir=Path(weights_dir).expanduser(),
                    device=device,
                    apply_softmax=apply_softmax,
                )
            else:
                self.logic.runDDSegPipeline(
                    mask_path=Path(mask_path).expanduser(),
                    dwi_path=Path(dwi_path).expanduser(),
                    bval_path=Path(bval_path).expanduser(),
                    bvec_path=Path(bvec_path).expanduser(),
                    output_dir=Path(output_dir).expanduser(),
                    weights_dir=Path(weights_dir).expanduser(),
                    slicer_base=Path(slicer_base).expanduser(),
                    slicer_ext=Path(slicer_ext).expanduser() if slicer_ext else None,
                    device=device,
                    apply_softmax=apply_softmax,
                )

        self.appendLog("DDSeg pipeline completed.")
        self._setStatus("ready", "Pipeline ready")
        slicer.util.showStatusMessage("DDSeg pipeline completed.", 5000)
