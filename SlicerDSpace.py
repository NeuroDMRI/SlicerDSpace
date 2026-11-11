"""SlicerDSpace scripted module for multi-workflow diffusion MRI analysis."""

from __future__ import annotations

from typing import Annotated, Optional

import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)
from slicer.parameterNodeWrapper import Choice, WithinRange, parameterNodeWrapper
from slicer.util import VTKObservationMixin
from shared.algorithm_selection import (
    BUNDLES,
    CONNECTOME,
    MICROSTRUCTURE,
    PREPROCESSING,
    TRACKING,
    selection_choices,
)

TORCH_DEVICE_CHOICES = ["auto", "cpu", *(f"cuda:{i}" for i in range(8))]

from connectome.module_logic import ConnectomeLogicMixin
from microstructure.module_logic import MicrostructureLogicMixin
from microstructure.module_widget import MicrostructureWidgetMixin
from anatomy.parcellation.module_logic import ParcellationLogicMixin
from anatomy.parcellation.module_widget import ParcellationWidgetMixin
from preprocessing.module_logic import PreprocessingLogicMixin
from anatomy.segmentation.module_logic import SegmentationLogicMixin
from anatomy.segmentation.module_widget import SegmentationWidgetMixin
from shared.module_logic import SharedLogicMixin
from shared.module_widget import SharedWidgetMixin
from anatomy.surface.module_logic import PipelineResult, SurfaceLogicMixin
from anatomy.surface.module_widget import SurfaceWidgetMixin
from tractography.module_logic import TractographyLogicMixin


class SlicerDSpace(ScriptedLoadableModule):
    """Main module metadata presented inside Slicer."""

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "SlicerDSpace"
        self.parent.categories = ["Diffusion"]
        # Do not set Slicer module-level dependencies here. Missing optional
        # extensions must not prevent the UI from loading; the dependency doctor
        # reports runnable and blocked targets after the module opens. Release
        # packaging declares recommended extension dependencies in CMake/.s4ext.
        self.parent.dependencies = []
        self.parent.contributors = [
            "Chengjin Li (Fan's Lab, UESTC)",
            "Fan Zhang (Fan's Lab, UESTC)",
        ]
        self.parent.helpText = (
            "Run multiple diffusion-MRI workflows inside one Slicer extension, including cortical "
            "surface reconstruction, parcellation, segmentation, microstructure analysis, "
            "tractography, bundle analysis, and connectome generation. "
            "The module reuses shared subject-root diffusion caches so related workflows can be "
            "combined without repeating the same DTI and registration preparation steps."
        )
        self.parent.acknowledgementText = (
            "Developed by Chengjin Li and Fan's Lab at the University of Electronic "
            "Science and Technology of China (UESTC)."
        )


@parameterNodeWrapper
class SlicerDSpaceParameterNode:
    """Parameter node definition stored in the MRML scene."""

    subjectId: str = ""
    inputFormat: Annotated[str, Choice(["nifti", "dicom", "nrrd"])] = "nifti"
    outputDirectory: str = ""
    inputDirectory: str = ""
    predictMode: Annotated[str, Choice(["all", "wm"])] = "all"
    device: Annotated[str, Choice(TORCH_DEVICE_CHOICES)] = "auto"
    cleanupIntermediates: bool = False
    postprocessSubjectId: str = ""
    postprocessNativeDir: str = ""
    postprocessMriDir: str = ""
    postprocessRepoDir: str = ""
    postprocessLicensePath: str = ""
    postprocessMode: Annotated[str, Choice(["whole", "left", "right"])] = "whole"
    postprocessDockerImage: str = ""
    postprocessForce: bool = True
    ddparcelSubjectId: str = ""
    ddparcelInputRoot: str = ""
    ddparcelOutputRoot: str = ""
    ddparcelAtlasPath: str = ""
    ddparcelFlip: int = 1
    ddsegMaskPath: str = ""
    ddsegDwiPath: str = ""
    ddsegBvalPath: str = ""
    ddsegBvecPath: str = ""
    ddsegOutputDir: str = ""
    ddsegWeightsDir: str = ""
    ddsegSlicerBase: str = ""
    ddsegSlicerExt: str = ""
    ddsegDevice: Annotated[str, Choice(TORCH_DEVICE_CHOICES)] = "auto"
    ddsegApplySoftmax: bool = False
    microstructureSubjectId: str = ""
    microstructureInputRoot: str = ""
    microstructureOutputDir: str = ""
    microstructureRoiPath: str = ""
    microstructureFitMethod: Annotated[str, Choice(["wls", "ls"])] = "wls"
    microstructureRunFreeWater: bool = False
    microstructureRunNoddi: bool = False
    microstructureForceRecompute: bool = False
    targetPreprocessing: bool = True
    orchestrationPreprocessingStages: Annotated[str, Choice(selection_choices(PREPROCESSING))] = "denoise,eddy,topup"
    brainMaskBackend: Annotated[str, Choice(["pnl_cnn", "provided"])] = "pnl_cnn"
    brainMaskDevice: Annotated[str, Choice(["auto", "cpu", "cuda"])] = "auto"
    pnlMaskRoute: Annotated[str, Choice(["auto", "managed", "native"])] = "auto"
    # Topup / distortion correction for the MRtrix3/FSL route. Empty/0 means
    # "not set", so nothing is forwarded and dwifslpreproc defaults apply.
    preprocessingPhaseEncodingDirection: str = ""
    preprocessingReadoutTime: float = 0.0
    preprocessingReversePhaseB0: str = ""
    orchestrationRunName: str = "Test1"
    orchestrationTractographyBackend: Annotated[str, Choice(selection_choices(TRACKING))] = "csd_act"
    ukfParameters: str = "{}"
    orchestrationBundleBackend: Annotated[str, Choice([*selection_choices(BUNDLES), "all"]) ] = "wmql"
    orchestrationWmqlPreset: Annotated[str, Choice(["official", "official_cerebellar", "cerebellar"])] = "official"
    orchestrationWmqlSelection: str = ""
    maxParallelModels: Annotated[int, WithinRange(1, 3)] = 1
    wmaInputTractography: str = ""
    wmaAtlas: Annotated[str, Choice(["org", "east-west", "naba"])] = "org"
    orchestrationSelectStreamlines: int = 1000000
    orchestrationMicrostructureStages: Annotated[str, Choice(selection_choices(MICROSTRUCTURE))] = "dti"
    # Echo time (ms) for iFW as text; empty = not set (the stage then reads EchoTime from the DWI sidecar).
    orchestrationIfwEchoTimeMs: str = ""
    orchestrationDkiFitMethod: Annotated[str, Choice(["WLS", "OLS", "MKCURVE", "MKCURVE_MATLAB"])] = "MKCURVE"
    orchestrationRunFreeWater: bool = False
    orchestrationRunNoddi: bool = False
    orchestrationRunDki: bool = False
    orchestrationRunIfw: bool = False
    batchMode: bool = False
    batchRoot: str = ""
    batchSize: int = 1
    batchArrangeInputs: Annotated[str, Choice(["none", "symlink", "copy"])] = "none"
    experimentsEnabled: bool = False
    experimentScope: Annotated[str, Choice(["subject", "cohort"])] = "subject"
    experimentRoot: str = ""
    experimentJobs: Annotated[int, WithinRange(1, 8)] = 1
    experimentCliOptions: str = ""
    experimentVariants: str = '[{"name": "baseline", "args": []}]'
    connectomeBackend: Annotated[str, Choice(["auto", "python", "mrtrix"])] = "auto"
    orchestrationConnectomeMetrics: Annotated[str, Choice(selection_choices(CONNECTOME))] = "sift2"
    # Three states (T-N1): unset = not forwarded (the auto backend then compresses on the MRtrix path),
    # on = --compress-mrtrix-dense, off = --no-compress-mrtrix-dense.
    connectomeCompressMrtrixDenseState: Annotated[str, Choice(["unset", "on", "off"])] = "unset"
    autoInstallPythonDependencies: bool = True
    targetSurface: bool = True
    targetSurfacePostprocess: bool = False
    targetParcellation: bool = True
    targetSegmentation: bool = True
    targetMicrostructure: bool = True
    targetTractography: bool = True
    targetBundles: bool = True
    targetConnectome: bool = True


class SlicerDSpaceWidget(
    SharedWidgetMixin,
    SurfaceWidgetMixin,
    ParcellationWidgetMixin,
    MicrostructureWidgetMixin,
    SegmentationWidgetMixin,
    ScriptedLoadableModuleWidget,
    VTKObservationMixin,
):
    """Slicer widget glue code."""

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic: SlicerDSpaceLogic | None = None
        self._parameterNode: SlicerDSpaceParameterNode | None = None
        self._parameterNodeGuiTag = None
        self._lastStatusMessage: Optional[str] = None
        self._pipelineActive: bool = False

    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)
        for index in range(self.layout.count()):
            item = self.layout.itemAt(index)
            existing = item.widget() if item is not None else None
            if existing is not None:
                existing.hide()
        for attr in ("reloadCollapsibleButton", "helpCollapsibleButton"):
            section = getattr(self, attr, None)
            if section is not None:
                section.hide()
        self._set_module_help_section_hidden(True)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(2)
        from shared.branding import migrate_qsettings
        migrate_qsettings(slicer.app.userSettings())
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SlicerDSpace.ui"))
        self.layout.addWidget(uiWidget)
        self._uiWidget = uiWidget
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = SlicerDSpaceLogic()
        self.logic.logCallback = self.appendLog

        self._connect_surface_ui()
        self._connect_parcellation_ui()
        self._connect_microstructure_ui()
        self._connect_segmentation_ui()
        self._connect_common_ui()
        self._initialize_common_ui_state()

class SlicerDSpaceLogic(
    SharedLogicMixin,
    PreprocessingLogicMixin,
    SurfaceLogicMixin,
    ParcellationLogicMixin,
    MicrostructureLogicMixin,
    SegmentationLogicMixin,
    TractographyLogicMixin,
    ConnectomeLogicMixin,
    ScriptedLoadableModuleLogic,
):
    """Drive the product-specific workflows exposed by the module."""

    def __init__(self) -> None:
        ScriptedLoadableModuleLogic.__init__(self)
        self._initialize_shared_logic_state()

    def getParameterNode(self) -> SlicerDSpaceParameterNode:
        from shared.branding import LEGACY_NAME
        from shared.ui_settings import mark_new_parameter_node
        raw = slicer.mrmlScene.GetSingletonNode(self.moduleName, 'vtkMRMLScriptedModuleNode')
        previous = slicer.mrmlScene.GetSingletonNode(LEGACY_NAME, 'vtkMRMLScriptedModuleNode')
        if raw is None and previous is not None:
            previous.SetSingletonTag(self.moduleName)
            previous.SetAttribute('ModuleName', self.moduleName)
            raw = previous
        if raw is None:
            raw = super().getParameterNode()
        mark_new_parameter_node(raw)
        previous = raw.GetParameter('orchestrationWmqlPreset')
        if previous in {'ddparcel_act', 'author', 'published_endpoints'}:
            with slicer.util.NodeModify(raw):
                raw.SetAttribute('SlicerDDSurfer.LegacyWMQLPreset', previous)
                raw.SetParameter('orchestrationWmqlPreset', 'official')
            self.log('[WMQL] Migrated GUI configuration to official multi-bundle rules; previous preset '
                     + previous + ' retained as a scene attribute. Historical run files are unchanged.')
        return SlicerDSpaceParameterNode(raw)


class SlicerDSpaceTest(ScriptedLoadableModuleTest):
    """Native Slicer contracts without downloading models or clearing user data."""

    def runTest(self):
        self.test_native_parameter_defaults()
        self.test_native_parameter_validation()
        self.test_native_scene_round_trip()
        self.test_native_user_defaults_do_not_replace_scene_values()

    def test_native_user_defaults_do_not_replace_scene_values(self):
        from shared.ui_settings import mark_new_parameter_node, consume_user_defaults_request
        scene = slicer.vtkMRMLScene()
        raw = scene.AddNewNodeByClass('vtkMRMLScriptedModuleNode')
        mark_new_parameter_node(raw)
        wrapped = SlicerDSpaceParameterNode(raw)
        self.assertTrue(consume_user_defaults_request(wrapped))
        wrapped.subjectId = 'preserve-scene'
        mark_new_parameter_node(raw)
        self.assertFalse(consume_user_defaults_request(wrapped))
        legacy = scene.AddNewNodeByClass('vtkMRMLScriptedModuleNode')
        legacy.SetParameter('subjectId', 'legacy-scene')
        mark_new_parameter_node(legacy)
        self.assertFalse(consume_user_defaults_request(SlicerDSpaceParameterNode(legacy)))
        self.assertEqual(wrapped.subjectId, 'preserve-scene')
        pending = scene.AddNewNodeByClass('vtkMRMLScriptedModuleNode')
        mark_new_parameter_node(pending)
        configured = SlicerDSpaceParameterNode(pending)
        configured.subjectId = 'configured-before-gui'
        self.assertFalse(consume_user_defaults_request(configured))
        self.assertEqual(configured.subjectId, 'configured-before-gui')

    def test_native_parameter_defaults(self):
        raw = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScriptedModuleNode')
        try:
            parameters = SlicerDSpaceParameterNode(raw)
            self.assertEqual(parameters.brainMaskBackend, 'pnl_cnn')
            self.assertEqual(parameters.brainMaskDevice, 'auto')
            self.assertEqual(parameters.pnlMaskRoute, 'auto')
            self.assertEqual(parameters.orchestrationMicrostructureStages, 'dti')
            self.assertFalse(parameters.orchestrationRunFreeWater)
            self.assertFalse(parameters.orchestrationRunNoddi)
            self.assertFalse(parameters.orchestrationRunDki)
            parameters.subjectId = 'native-smoke'
            self.assertEqual(SlicerDSpaceParameterNode(raw).subjectId, 'native-smoke')
            self.assertIsInstance(SlicerDSpaceLogic(), ScriptedLoadableModuleLogic)
        finally:
            slicer.mrmlScene.RemoveNode(raw)

    def test_native_parameter_validation(self):
        scene = slicer.vtkMRMLScene()
        raw = scene.AddNewNodeByClass('vtkMRMLScriptedModuleNode')
        parameters = SlicerDSpaceParameterNode(raw)
        for name, invalid in (
            ('brainMaskBackend', 'unsupported'),
            ('orchestrationTractographyBackend', 'unsupported'),
            ('orchestrationBundleBackend', 'unsupported'),
            ('orchestrationMicrostructureStages', 'unsupported'),
            ('maxParallelModels', 0),
            ('maxParallelModels', 4),
        ):
            with self.subTest(parameter=name, value=invalid):
                before = getattr(parameters, name)
                with self.assertRaises(ValueError):
                    setattr(parameters, name, invalid)
                self.assertEqual(getattr(parameters, name), before)
        for boundary in (1, 3):
            parameters.maxParallelModels = boundary
            self.assertEqual(parameters.maxParallelModels, boundary)

    def test_native_scene_round_trip(self):
        # Use private scenes so Self Tests never clear the user's loaded results.
        scene = slicer.vtkMRMLScene()
        raw = scene.AddNewNodeByClass('vtkMRMLScriptedModuleNode', 'DDSurferRoundTrip')
        parameters = SlicerDSpaceParameterNode(raw)
        expected = {
            'subjectId': 'native-round-trip',
            'inputDirectory': '/example/subject with spaces',
            'brainMaskBackend': 'provided',
            'brainMaskDevice': 'auto',
            'pnlMaskRoute': 'native',
            'orchestrationMicrostructureStages': 'dti,freewater,noddi,dki',
            'orchestrationRunFreeWater': True,
            'orchestrationRunNoddi': True,
            'orchestrationRunDki': True,
            'maxParallelModels': 3,
            'preprocessingReadoutTime': 0.032,
            'ukfParameters': '{"numTensor": 2}',
        }
        for name, value in expected.items():
            setattr(parameters, name, value)
        scene.SetSaveToXMLString(1)
        self.assertTrue(scene.Commit())
        restored_scene = slicer.vtkMRMLScene()
        restored_scene.SetLoadFromXMLString(1)
        restored_scene.SetSceneXMLString(scene.GetSceneXMLString())
        self.assertTrue(restored_scene.Import())
        restored_raw = restored_scene.GetFirstNodeByName('DDSurferRoundTrip')
        self.assertIsNotNone(restored_raw)
        restored = SlicerDSpaceParameterNode(restored_raw)
        for name, value in expected.items():
            with self.subTest(parameter=name):
                self.assertEqual(getattr(restored, name), value)
