"""English/Chinese UI text for the workflow sections, topup fields and Results.

Merged into ``shared.ui_onboarding.TRANSLATIONS`` so the language switch in
the header covers the new controls too.
"""

from __future__ import annotations


BRAIN_MASK_BACKEND_TEXT_KEYS = {
    "pnl_cnn": "brain_mask_pnl_cnn",
    "synthstrip": "brain_mask_synthstrip",
    "mrtrix": "brain_mask_mrtrix",
    "provided": "brain_mask_provided",
    "dipy": "brain_mask_dipy",
}


# SC-1b: route value -> text key. "native (experimental)" is the label of the native route.
PNL_MASK_ROUTE_TEXT_KEYS = {
    "auto": "pnl_route_auto",
    "managed": "pnl_route_managed",
    "native": "pnl_route_native",
}


WORKFLOW_TEXT = {
    "en": {
        "section_getting_started": "Getting started",
        "section_inputs": "1. Inputs",
        "section_targets": "2. Products",
        "compact_advanced": "Advanced settings",
        "compact_advanced_hint": "Choose algorithms with the ... button beside each product. Adjust only the parameters you need below.",
        "input_layout_found": "Input layout found. Dependencies are checked when you run.",
        "compact_algorithm_selection": "Selected: {algorithms}",
        "compact_algorithm_tooltip": "Selected algorithms: {algorithms}. Use the ... button beside this product to choose one or more. This is the configuration, not completed results or QC status.",
        "compact_import": "Input import / conversion",
        "compact_parameters": "Analysis parameters",
        "compact_environment": "Environment and installation",
        "compact_results": "Results and QC",
        "compact_run": "Run SlicerDSpace",
        "compact_log": "Log",
        "compact_preprocessing": "Denoise / Gibbs / motion correction",
        "compact_surface": "DDSurfer white / pial surfaces",
        "compact_postprocess": "Cortical metrics / atlas mapping",
        "compact_parcellation": "DDParcel regional labels",
        "compact_segmentation": "DDSeg WM / GM / CSF",
        "compact_microstructure": "Selected: DTI",
        "compact_tractography": "Selected: MRtrix3 CSD / ACT",
        "compact_bundles": "Selected: WMQL",
        "compact_connectome": "Structural connectivity matrices",
        "section_parameters": "3. Route and parameters",
        "section_run": "4. Run",
        "section_results": "5. Results",
        "topup_group": "Distortion correction (topup)",
        "phase_encoding_direction": "Phase-encoding direction",
        "readout_time": "Total readout time (s)",
        "reverse_phase_b0": "Reverse-PE b=0 image",
        "brain_mask_backend": "Brain masking",
        "brain_mask_device": "Mask device",
        "brain_mask_device_tooltip": "PNL CNN and SynthStrip: auto prefers the first visible CUDA GPU in the selected external runtime. CUDA requires a GPU; CPU explicitly disables it. Runtime GPU failures are not silently retried. MRtrix dwi2mask and DIPY median-Otsu are CPU algorithms.",
        "pnl_mask_route": "PNL route",
        "pnl_mask_route_tooltip": (
            "PNL CNN route. auto: managed (conda runner with ANTs) on Linux, native on macOS and Windows. "
            "native (experimental): did not meet the M-A parity rule on 2 of 24 cases (T2 design §3)."
        ),
        "pnl_route_auto": "auto",
        "pnl_route_managed": "managed",
        "pnl_route_native": "native (experimental)",
        "brain_mask_pnl_cnn": "PNL CNN (dMRI)",
        "brain_mask_synthstrip": "SynthStrip",
        "brain_mask_mrtrix": "MRtrix dwi2mask",
        "brain_mask_dipy": "DIPY median-Otsu (lightweight)",
        "brain_mask_provided": "Provided mask only",
        "microstructure_dti_always": "DTI (always enabled)",
        "microstructure_fw_option": "FW",
        "microstructure_noddi_option": "NODDI",
        "microstructure_dki_option": "DKI",
        "microstructure_ifw_option": "iFW",
        "microstructure_fw_tooltip": "Optional Free Water model. Independent of NODDI and DKI; off by default.",
        "microstructure_noddi_tooltip": "Optional NODDI model. Independent of Free Water and DKI; off by default.",
        "microstructure_dki_tooltip": "Optional diffusion kurtosis analysis. Standard DKI is not MK-Curve; off by default.",
        "microstructure_ifw_tooltip": "Optional iFW: free-water fraction corrected for T2 weighting, minus the CSF fraction. Also runs Free Water and DDSeg; needs the echo time; off by default.",
        "ifw_echo_time": "Echo time (ms)",
        "ifw_echo_time_tooltip": "Echo time of the DWI in ms. Leave empty to read EchoTime from the DWI JSON sidecar; with neither, the iFW stage stops and the other stages continue. There is no default.",
        "ifw_t2_tissue": "iFW tissue T2 (ms)",
        "ifw_t2_csf": "iFW CSF T2 (ms)",
        "ifw_t2_tooltip": "Fixed in code: T2 of tissue 95 ms and of CSF 1500 ms (Cho et al. preprint, Supplementary Methods 1). Read-only here; researchers can use --ifw-t2-gm-ms and --ifw-t2-csf-ms on the command line.",
        "dki_fit_method": "DKI fit method",
        "dki_fit_method_tooltip": (
            "MKCURVE (default) uses the bundled Python port without MATLAB; if it stops, re-run with WLS. WLS/OLS are standard DKI fits. "
            "MKCURVE_MATLAB is an optional author reference; configure its paths under Environment and installation > More options. "
            "All methods require eligible multi-shell data; no silent fallback."
        ),
        "brain_mask_backend_tooltip": (
            "Existing supplied masks are respected. This choice only controls generation of a missing mask. "
            "PNL CNN (dMRI) is the default. Prepare selected analysis environment manages its Python, official dwi_masking.py, "
            "models and ANTs. Manual path overrides are under Environment and installation > More options. "
            "SynthStrip uses FreeSurfer home; MRtrix uses MRtrix3 bin dir. "
            "DIPY median-Otsu uses the existing Python runtime without model weights or external tools; review brain boundaries. "
            "Provided mask only requires an existing mask. Missing dependencies or masks cause an error; "
            "there is no automatic fallback to another masking backend."
        ),
        "run_selector_label": "Run",
        "refresh": "Refresh",
        "browse_runs": "Browse…",
        "browse_runs_title": "Choose a run or output folder",
        "add_run_folder": "Add a run folder…",
        "no_runs_found": "No runs found",
        "no_run_selected": "No run selected.",
        "no_run_in_progress": "No run in progress",
        "current_stage": "Current stage",
        "run_state_idle": "Idle",
        "run_state_preflight": "Checking environment",
        "run_state_running": "Running",
        "run_state_completed": "Completed",
        "run_state_completed_with_blocks": "Completed with blocked stages",
        "run_state_failed": "Completed with errors",
        "run_state_cancelled": "Cancelled",
        "preprocessing_tooltip": (
            "Optional DWI denoising, Gibbs-ringing removal and motion/eddy correction. "
            "Leave unchecked to run downstream workflows from the original subject input folder."
        ),
        "parcellation_tooltip": "DDParcel aparc and wmparc label maps in native dMRI space.",
        "segmentation_tooltip": "DDSeg white matter, gray matter and CSF segmentation and probability maps.",
        "microstructure_tooltip": "DTI maps and ROI statistics by default. Independently enable Free Water, NODDI or DKI under Advanced > Analysis parameters.",
        "tractography_tooltip": "MRtrix3 CSD / ACT by default. Choose UKF, DTI deterministic or CSD + SAF under Advanced > Analysis parameters. Missing backends are reported, not silently substituted.",
        "bundles_tooltip": "White-matter bundle extraction with WMQL presets, WMA or TractCloud.",
        "connectome_tooltip": "Streamline-count and scalar-weighted structural connectivity matrices.",
    },
    "zh": {
        "section_getting_started": "快速开始",
        "section_inputs": "1. 输入",
        "section_targets": "2. 分析任务",
        "compact_advanced": "高级设置",
        "compact_advanced_hint": "通过任务旁的 ... 按钮切换算法；下面仅调整需要修改的参数。",
        "input_layout_found": "已找到输入文件结构；点击运行时将检查依赖。",
        "compact_algorithm_selection": "当前：{algorithms}",
        "compact_algorithm_tooltip": "当前算法：{algorithms}。点击任务旁的 ... 按钮可单选或多选；这是运行配置，不代表结果已生成或 QC 已通过。",
        "compact_import": "输入导入 / 格式转换",
        "compact_parameters": "分析参数",
        "compact_environment": "环境与安装",
        "compact_results": "结果与质控",
        "compact_run": "运行 SlicerDSpace",
        "compact_log": "日志",
        "compact_preprocessing": "降噪 / Gibbs / 运动校正",
        "compact_surface": "DDSurfer 白质 / 软脑膜表面",
        "compact_postprocess": "皮层指标 / 图谱映射",
        "compact_parcellation": "DDParcel 脑区标签",
        "compact_segmentation": "DDSeg 白质 / 灰质 / 脑脊液",
        "compact_microstructure": "当前：DTI",
        "compact_tractography": "当前：MRtrix3 CSD / ACT",
        "compact_bundles": "当前：WMQL",
        "compact_connectome": "结构连接矩阵",
        "section_parameters": "3. 路线与参数",
        "section_run": "4. 运行",
        "section_results": "5. 结果",
        "topup_group": "畸变校正（topup）",
        "phase_encoding_direction": "相位编码方向",
        "readout_time": "总读出时间（秒）",
        "reverse_phase_b0": "反向相位编码 b=0 图像",
        "brain_mask_backend": "脑掩膜方法",
        "brain_mask_device": "掩膜计算设备",
        "brain_mask_device_tooltip": "PNL CNN 和 SynthStrip：Auto 优先使用对应外部运行环境中可见的第一个 CUDA GPU；CUDA 要求 GPU 可用，CPU 显式禁用 GPU。GPU 运行错误不会静默重试。MRtrix dwi2mask 与 DIPY median-Otsu 是 CPU 算法。",
        "pnl_mask_route": "PNL 路线",
        "pnl_mask_route_tooltip": (
            "PNL CNN 路线。auto：Linux 用 managed（conda 运行环境，含 ANTs），macOS 和 Windows 用 native。"
            "native（实验性）：在 M-A 一致性规则的 24 例中有 2 例未达标（T2 设计 §3）。"
        ),
        "pnl_route_auto": "auto",
        "pnl_route_managed": "managed",
        "pnl_route_native": "native（实验性）",
        "brain_mask_pnl_cnn": "PNL CNN (dMRI)",
        "brain_mask_synthstrip": "SynthStrip",
        "brain_mask_mrtrix": "MRtrix dwi2mask",
        "brain_mask_dipy": "DIPY median-Otsu（轻量方案）",
        "brain_mask_provided": "仅使用提供的掩膜",
        "microstructure_dti_always": "DTI（始终启用）",
        "microstructure_fw_option": "FW",
        "microstructure_noddi_option": "NODDI",
        "microstructure_dki_option": "DKI",
        "microstructure_ifw_option": "iFW",
        "microstructure_fw_tooltip": "可选自由水模型，独立于 NODDI 和 DKI，默认关闭。",
        "microstructure_noddi_tooltip": "可选 NODDI 模型，独立于自由水和 DKI，默认关闭。",
        "microstructure_dki_tooltip": "可选扩散峰度分析，标准 DKI 不等于 MK-Curve，默认关闭。",
        "microstructure_ifw_tooltip": "可选 iFW：按 T2 加权校正的自由水分数，再减去 CSF 分数。会同时运行自由水和 DDSeg；需要回波时间；默认关闭。",
        "ifw_echo_time": "回波时间 (ms)",
        "ifw_echo_time_tooltip": "DWI 的回波时间，单位 ms。留空则读取 DWI JSON 附属文件中的 EchoTime；两者都没有时 iFW 步骤停止，其他步骤继续。没有默认值。",
        "ifw_t2_tissue": "iFW 组织 T2 (ms)",
        "ifw_t2_csf": "iFW CSF T2 (ms)",
        "ifw_t2_tooltip": "固定在代码中：组织 T2 为 95 ms，CSF T2 为 1500 ms（Cho 等预印本，补充方法 1）。此处只读；研究者可在命令行使用 --ifw-t2-gm-ms 和 --ifw-t2-csf-ms。",
        "dki_fit_method": "DKI 拟合方法",
        "dki_fit_method_tooltip": (
            "MKCURVE（默认）使用内置 Python 移植版，不需要 MATLAB；若它停止，请改用 WLS 重跑。WLS/OLS 是标准 DKI 拟合。"
            "MKCURVE_MATLAB 是可选的作者代码对照后端，可在高级设置中配置外部路径。"
            "所有方法都检查多 shell 数据条件，不会静默回退。"
        ),
        "brain_mask_backend_tooltip": (
            "尊重已有的用户提供掩膜；此选项仅控制缺失掩膜的生成。DIPY median-Otsu 不需要外部程序或权重，需检查脑边界。默认 PNL CNN (dMRI)，"
            "需在“高级设置 > 环境与模型 > 外部运行路径”配置外部 Python、官方 dwi_masking.py、"
            "模型目录和 ANTs 路径。SynthStrip 共用 FreeSurfer 主目录，MRtrix 共用 MRtrix3 bin 目录。"
            "“仅使用提供的掩膜”要求掩膜已存在。依赖或掩膜缺失时明确报错，不会自动切换其他掩膜后端。"
        ),
        "run_selector_label": "运行",
        "refresh": "刷新",
        "browse_runs": "浏览…",
        "browse_runs_title": "选择运行目录或输出目录",
        "add_run_folder": "添加运行目录…",
        "no_runs_found": "未找到运行记录",
        "no_run_selected": "未选择运行。",
        "no_run_in_progress": "当前没有运行",
        "current_stage": "当前阶段",
        "run_state_idle": "空闲",
        "run_state_preflight": "正在检查环境",
        "run_state_running": "运行中",
        "run_state_completed": "已完成",
        "run_state_completed_with_blocks": "已完成（部分阶段被阻止）",
        "run_state_failed": "完成但有错误",
        "run_state_cancelled": "已取消",
        "preprocessing_tooltip": "可选的 DWI 降噪、Gibbs 伪影去除及运动/涡流校正。不勾选时，下游流程直接使用原始受试者输入目录。",
        "parcellation_tooltip": "原生 dMRI 空间中的 DDParcel aparc 与 wmparc 标签图。",
        "segmentation_tooltip": "DDSeg 白质、灰质、脑脊液分割及概率图。",
        "microstructure_tooltip": "默认只生成 DTI 标量图和 ROI 统计。在“高级设置 > 分析参数”中独立启用自由水、NODDI 或 DKI。",
        "tractography_tooltip": "默认 MRtrix3 CSD / ACT。可在“高级设置 > 分析参数”中选择 UKF、DTI 确定性追踪或 CSD + SAF。依赖缺失时明确报告，不会静默替换算法。",
        "bundles_tooltip": "使用 WMQL 预设、WMA 或 TractCloud 提取白质纤维束。",
        "connectome_tooltip": "流线计数及标量加权的结构连接矩阵。",
    },
}


WORKFLOW_TEXT_BINDINGS = {
    "gettingStartedCollapsibleButton": "section_getting_started",
    "configurationCollapsibleButton": "section_inputs",
    "targetsCollapsibleButton": "section_targets",
    "parametersCollapsibleButton": "section_parameters",
    "resultsCollapsibleButton": "section_results",
    "preprocessingPhaseEncodingDirectionLabel": "phase_encoding_direction",
    "preprocessingReadoutTimeLabel": "readout_time",
    "preprocessingReversePhaseB0Label": "reverse_phase_b0",
    "brainMaskBackendLabel": "brain_mask_backend",
    "brainMaskDeviceLabel": "brain_mask_device",
    "pnlMaskRouteLabel": "pnl_mask_route",
    "orchestrationDtiStageLabel": "microstructure_dti_always",
    "orchestrationFreeWaterCheckBox": "microstructure_fw_option",
    "orchestrationNoddiCheckBox": "microstructure_noddi_option",
    "orchestrationDkiCheckBox": "microstructure_dki_option",
    "orchestrationIfwCheckBox": "microstructure_ifw_option",
    "orchestrationIfwEchoTimeLabel": "ifw_echo_time",
    "orchestrationIfwT2TissueLabel": "ifw_t2_tissue",
    "orchestrationIfwT2CsfLabel": "ifw_t2_csf",
    "orchestrationDkiFitMethodLabel": "dki_fit_method",
}

WORKFLOW_TOOLTIP_BINDINGS = {
    "brainMaskBackendComboBox": "brain_mask_backend_tooltip",
    "brainMaskDeviceComboBox": "brain_mask_device_tooltip",
    "pnlMaskRouteComboBox": "pnl_mask_route_tooltip",
    "orchestrationFreeWaterCheckBox": "microstructure_fw_tooltip",
    "orchestrationNoddiCheckBox": "microstructure_noddi_tooltip",
    "orchestrationDkiCheckBox": "microstructure_dki_tooltip",
    "orchestrationIfwCheckBox": "microstructure_ifw_tooltip",
    "orchestrationIfwEchoTimeLineEdit": "ifw_echo_time_tooltip",
    "orchestrationIfwT2TissueLineEdit": "ifw_t2_tooltip",
    "orchestrationIfwT2CsfLineEdit": "ifw_t2_tooltip",
    "orchestrationDkiFitMethodComboBox": "dki_fit_method_tooltip",
    "targetPreprocessingCheckBox": "preprocessing_tooltip",
    "targetParcellationCheckBox": "parcellation_tooltip",
    "targetSegmentationCheckBox": "segmentation_tooltip",
    "targetMicrostructureCheckBox": "microstructure_tooltip",
    "targetTractographyCheckBox": "tractography_tooltip",
    "targetBundlesCheckBox": "bundles_tooltip",
    "targetConnectomeCheckBox": "connectome_tooltip",
}
