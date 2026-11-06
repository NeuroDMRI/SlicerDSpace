"""Parameter-node to Qt-widget synchronization helpers."""

from __future__ import annotations

from shared.user_config import canonical_microstructure_stages


def sync_parameter_node_to_ui(widget) -> None:
    """Synchronize common SlicerDSpace parameter-node values into Qt widgets."""

    node = widget._parameterNode
    if not node:
        return

    ui = widget.ui
    widget._sync_path_widget(getattr(ui, "inputDirectoryPath", None), node.inputDirectory or "")
    widget._sync_path_widget(getattr(ui, "outputDirectoryPath", None), node.outputDirectory or "")
    widget._sync_combo_text(getattr(ui, "predictModeComboBox", None), node.predictMode or "all")
    widget._sync_combo_text(getattr(ui, "deviceComboBox", None), node.device or "auto")

    widget._sync_line_edit(getattr(ui, "postprocessSubjectLineEdit", None), node.postprocessSubjectId or "")
    widget._sync_path_widget(getattr(ui, "postprocessNativePath", None), node.postprocessNativeDir or "")
    widget._sync_path_widget(getattr(ui, "postprocessMriPath", None), node.postprocessMriDir or "")
    widget._sync_path_widget(getattr(ui, "postprocessRepoPath", None), node.postprocessRepoDir or "")
    widget._sync_path_widget(getattr(ui, "postprocessLicensePath", None), node.postprocessLicensePath or "")
    widget._sync_combo_text(getattr(ui, "postprocessModeComboBox", None), node.postprocessMode or "whole")
    widget._sync_line_edit(getattr(ui, "postprocessImageLineEdit", None), node.postprocessDockerImage or "")
    widget._sync_checkbox(getattr(ui, "postprocessForceCheckBox", None), bool(node.postprocessForce))

    widget._sync_line_edit(getattr(ui, "ddparcelSubjectLineEdit", None), node.ddparcelSubjectId or "")
    widget._sync_path_widget(getattr(ui, "ddparcelInputRootPath", None), node.ddparcelInputRoot or "")
    widget._sync_path_widget(getattr(ui, "ddparcelOutputRootPath", None), node.ddparcelOutputRoot or "")
    widget._sync_path_widget(getattr(ui, "ddparcelAtlasPath", None), node.ddparcelAtlasPath or "")
    widget._sync_spinbox(getattr(ui, "ddparcelFlipSpinBox", None), int(node.ddparcelFlip))

    widget._sync_line_edit(getattr(ui, "microstructureSubjectLineEdit", None), node.microstructureSubjectId or "")
    widget._sync_path_widget(getattr(ui, "microstructureInputRootPath", None), node.microstructureInputRoot or "")
    widget._sync_path_widget(getattr(ui, "microstructureOutputPath", None), node.microstructureOutputDir or "")
    widget._sync_path_widget(getattr(ui, "microstructureRoiPath", None), node.microstructureRoiPath or "")
    widget._sync_combo_text(getattr(ui, "microstructureFitComboBox", None), (node.microstructureFitMethod or "wls").lower())
    widget._sync_checkbox(getattr(ui, "microstructureRunFreeWaterCheckBox", None), bool(node.microstructureRunFreeWater))
    widget._sync_checkbox(getattr(ui, "microstructureRunNoddiCheckBox", None), bool(node.microstructureRunNoddi))
    widget._sync_checkbox(getattr(ui, "microstructureForceRecomputeCheckBox", None), bool(node.microstructureForceRecompute))

    widget._sync_path_widget(getattr(ui, "ddsegMaskPath", None), node.ddsegMaskPath or "")
    widget._sync_path_widget(getattr(ui, "ddsegDwiPath", None), node.ddsegDwiPath or "")
    widget._sync_path_widget(getattr(ui, "ddsegBvalPath", None), node.ddsegBvalPath or "")
    widget._sync_path_widget(getattr(ui, "ddsegBvecPath", None), node.ddsegBvecPath or "")
    widget._sync_path_widget(getattr(ui, "ddsegOutputPath", None), node.ddsegOutputDir or "")
    widget._sync_path_widget(getattr(ui, "ddsegWeightsPath", None), node.ddsegWeightsDir or "")
    widget._sync_path_widget(getattr(ui, "ddsegSlicerBasePath", None), node.ddsegSlicerBase or "")
    widget._sync_path_widget(getattr(ui, "ddsegSlicerExtPath", None), node.ddsegSlicerExt or "")
    widget._sync_combo_text(getattr(ui, "ddsegDeviceComboBox", None), node.ddsegDevice or "auto", via_index=False)
    widget._sync_checkbox(getattr(ui, "ddsegSoftmaxCheckBox", None), bool(node.ddsegApplySoftmax))

    widget._sync_checkbox(getattr(ui, "targetPreprocessingCheckBox", None), bool(node.targetPreprocessing))
    sync_combo_data = getattr(widget, "_sync_combo_data", None)
    if sync_combo_data is not None:
        sync_combo_data(
            getattr(ui, "brainMaskBackendComboBox", None),
            getattr(node, "brainMaskBackend", "pnl_cnn") or "pnl_cnn",
        )
        widget._sync_combo_text(getattr(ui, "brainMaskDeviceComboBox", None),
                                getattr(node, "brainMaskDevice", "auto") or "auto", via_index=False)
        sync_combo_data(
            getattr(ui, "preprocessingPhaseEncodingDirectionComboBox", None),
            getattr(node, "preprocessingPhaseEncodingDirection", "") or "",
        )
        widget._sync_double_spinbox(
            getattr(ui, "preprocessingReadoutTimeSpinBox", None),
            float(getattr(node, "preprocessingReadoutTime", 0.0) or 0.0),
        )
        widget._sync_path_widget(
            getattr(ui, "preprocessingReversePhaseB0Path", None),
            getattr(node, "preprocessingReversePhaseB0", "") or "",
        )
    widget._sync_combo_text(getattr(ui, "preprocessingBackendComboBox", None), node.preprocessingBackend or "dipy", via_index=False)
    widget._sync_line_edit(getattr(ui, "orchestrationRunNameLineEdit", None), node.orchestrationRunName or "")
    widget._sync_combo_text(
        getattr(ui, "orchestrationTractographyBackendComboBox", None),
        node.orchestrationTractographyBackend or "csd_act",
        via_index=False,
    )
    if hasattr(widget, '_ukfControls'):
        from shared.ui_ukf import refresh_ukf_controls
        refresh_ukf_controls(widget, node.orchestrationTractographyBackend or 'csd_act')
    widget._sync_combo_text(
        getattr(ui, "orchestrationBundleBackendComboBox", None),
        getattr(node, "orchestrationBundleBackend", "wmql") or "wmql",
        via_index=False,
    )
    widget._sync_path_widget(
        getattr(ui, "wmaInputTractographyPath", None),
        getattr(node, "wmaInputTractography", "") or "",
    )
    widget._sync_combo_text(getattr(ui, "wmaAtlasComboBox", None), getattr(node, "wmaAtlas", "org"))
    widget._sync_combo_text(getattr(ui, "wmqlQueryPresetComboBox", None),
                            getattr(node, "orchestrationWmqlPreset", "official") or "official", via_index=False)
    widget._sync_line_edit(getattr(ui, 'wmqlQuerySelectionLineEdit', None),
                           getattr(node, 'orchestrationWmqlSelection', '') or '')
    widget._sync_spinbox(getattr(ui, 'parallelModelsSpinBox', None), int(getattr(node, 'maxParallelModels', 1)))
    widget._sync_spinbox(getattr(ui, "orchestrationSelectStreamlinesSpinBox", None), int(node.orchestrationSelectStreamlines))
    widget._sync_combo_text(
        getattr(ui, "orchestrationMicrostructureStagesComboBox", None),
        widget._microstructure_stages_text(),
        via_index=False,
    )
    stages = set(canonical_microstructure_stages(widget._microstructure_stages_text()))
    for name, stage in (
        ("orchestrationFreeWaterCheckBox", "freewater"),
        ("orchestrationNoddiCheckBox", "noddi"),
        ("orchestrationDkiCheckBox", "dki"),
    ):
        widget._sync_checkbox(getattr(ui, name, None), stage in stages)
    widget._sync_combo_text(
        getattr(ui, "orchestrationDkiFitMethodComboBox", None),
        getattr(node, "orchestrationDkiFitMethod", "WLS") or "WLS",
        via_index=False,
    )
    widget._sync_checkbox(getattr(ui, "batchModeCheckBox", None), bool(getattr(node, "batchMode", False)))
    widget._sync_path_widget(getattr(ui, "batchRootPath", None), getattr(node, "batchRoot", "") or "")
    widget._sync_spinbox(getattr(ui, "batchSizeSpinBox", None), int(getattr(node, "batchSize", 1) or 1))
    widget._sync_combo_text(
        getattr(ui, "batchArrangeInputsComboBox", None),
        getattr(node, "batchArrangeInputs", "none") or "none",
        via_index=False,
    )
    widget._sync_combo_text(getattr(ui, "connectomeBackendComboBox", None), node.connectomeBackend or "auto", via_index=False)
    widget._sync_checkbox(getattr(ui, "connectomeCompressMrtrixDenseCheckBox", None), bool(node.connectomeCompressMrtrixDense))
    widget._sync_combo_text(
        getattr(ui, "unifiedTorchDeviceComboBox", None),
        node.device or node.ddsegDevice or "auto",
        via_index=False,
    )
    widget._sync_checkbox(getattr(ui, "autoInstallPythonDependenciesCheckBox", None), bool(node.autoInstallPythonDependencies))
    widget._sync_checkbox(getattr(ui, "targetSurfaceCheckBox", None), bool(node.targetSurface))
    widget._sync_checkbox(
        getattr(ui, "targetSurfacePostprocessCheckBox", None),
        bool(getattr(node, "targetSurfacePostprocess", False)),
    )
    widget._sync_checkbox(getattr(ui, "targetParcellationCheckBox", None), bool(node.targetParcellation))
    widget._sync_checkbox(getattr(ui, "targetSegmentationCheckBox", None), bool(node.targetSegmentation))
    widget._sync_checkbox(getattr(ui, "targetMicrostructureCheckBox", None), bool(node.targetMicrostructure))
    widget._sync_checkbox(getattr(ui, "targetTractographyCheckBox", None), bool(node.targetTractography))
    widget._sync_checkbox(getattr(ui, "targetBundlesCheckBox", None), bool(node.targetBundles))
    widget._sync_checkbox(getattr(ui, "targetConnectomeCheckBox", None), bool(node.targetConnectome))
    from shared.ui_compact import refresh_compact_algorithms

    refresh_compact_algorithms(widget)
