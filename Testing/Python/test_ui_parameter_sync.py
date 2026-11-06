#!/usr/bin/env python3
"""Tests for parameter-node to UI synchronization helpers."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from shared.ui_parameter_sync import sync_parameter_node_to_ui


class UIParameterSyncTest(unittest.TestCase):
    def test_sync_parameter_node_to_ui_calls_expected_widget_sync_methods(self) -> None:
        calls: list[tuple[str, str, object, object]] = []
        ui = SimpleNamespace(
            inputDirectoryPath=object(),
            outputDirectoryPath=object(),
            orchestrationRunNameLineEdit=object(),
            targetSurfaceCheckBox=object(),
            targetSurfacePostprocessCheckBox=object(),
            orchestrationTractographyBackendComboBox=object(),
            orchestrationMicrostructureStagesComboBox=object(),
        )
        node = SimpleNamespace(
            inputDirectory="/data/sub-01",
            outputDirectory="/data/out",
            predictMode="all",
            device="cpu",
            postprocessSubjectId="",
            postprocessNativeDir="",
            postprocessMriDir="",
            postprocessRepoDir="",
            postprocessLicensePath="",
            postprocessMode="whole",
            postprocessDockerImage="",
            postprocessForce=True,
            ddparcelSubjectId="",
            ddparcelInputRoot="",
            ddparcelOutputRoot="",
            ddparcelAtlasPath="",
            ddparcelFlip=1,
            microstructureSubjectId="",
            microstructureInputRoot="",
            microstructureOutputDir="",
            microstructureRoiPath="",
            microstructureFitMethod="wls",
            microstructureRunFreeWater=False,
            microstructureRunNoddi=False,
            microstructureForceRecompute=False,
            ddsegMaskPath="",
            ddsegDwiPath="",
            ddsegBvalPath="",
            ddsegBvecPath="",
            ddsegOutputDir="",
            ddsegWeightsDir="",
            ddsegSlicerBase="",
            ddsegSlicerExt="",
            ddsegDevice="cpu",
            ddsegApplySoftmax=False,
            targetPreprocessing=True,
            preprocessingBackend="dipy",
            orchestrationRunName="Test1",
            orchestrationTractographyBackend="dti_det",
            orchestrationBundleBackend="wmql",
            orchestrationSelectStreamlines=100,
            batchMode=False,
            batchRoot="",
            batchSize=1,
            batchArrangeInputs="none",
            connectomeBackend="python",
            connectomeCompressMrtrixDenseState="unset",
            autoInstallPythonDependencies=True,
            targetSurface=True,
            targetSurfacePostprocess=False,
            targetParcellation=False,
            targetSegmentation=True,
            targetMicrostructure=False,
            targetTractography=True,
            targetBundles=False,
            targetConnectome=True,
        )

        class Widget:
            def _microstructure_stages_text(self):
                return "dti"

            def _sync_path_widget(self, target, value):
                calls.append(("path", "", target, value))

            def _sync_line_edit(self, target, value):
                calls.append(("line", "", target, value))

            def _sync_combo_text(self, target, value, *, via_index=True):
                calls.append(("combo", str(via_index), target, value))

            def _sync_checkbox(self, target, value):
                calls.append(("checkbox", "", target, value))

            def _sync_spinbox(self, target, value):
                calls.append(("spinbox", "", target, value))

        widget = Widget()
        widget._parameterNode = node
        widget.ui = ui
        with patch("shared.ui_compact.refresh_compact_algorithms") as refresh_algorithms:
            sync_parameter_node_to_ui(widget)
        refresh_algorithms.assert_called_once_with(widget)

        self.assertIn(("path", "", ui.inputDirectoryPath, "/data/sub-01"), calls)
        self.assertIn(("path", "", ui.outputDirectoryPath, "/data/out"), calls)
        self.assertIn(("line", "", ui.orchestrationRunNameLineEdit, "Test1"), calls)
        self.assertIn(("checkbox", "", ui.targetSurfaceCheckBox, True), calls)
        self.assertIn(("checkbox", "", ui.targetSurfacePostprocessCheckBox, False), calls)
        self.assertIn(("combo", "False", ui.orchestrationTractographyBackendComboBox, "dti_det"), calls)
        self.assertIn(("combo", "False", ui.orchestrationMicrostructureStagesComboBox, "dti"), calls)


if __name__ == "__main__":
    unittest.main()
