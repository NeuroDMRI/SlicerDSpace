"""English/Chinese UI text for the workflow sections, topup fields and Results.

Merged into ``shared.ui_onboarding.TRANSLATIONS`` so the language switch in
the header covers the new controls too.
"""

from __future__ import annotations


BRAIN_MASK_BACKEND_TEXT_KEYS = {
    "pnl_cnn": "brain_mask_pnl_cnn",
    "provided": "brain_mask_provided",
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
        "compact_advanced_hint": "Select algorithms via ...; optional parameters below.",
        "input_layout_found": "Input layout found. Dependencies are checked when you run.",
        "compact_algorithm_selection": "Selected: {algorithms}",
        "compact_algorithm_tooltip": "Selected: {algorithms}. Use the ... button to configure.",
        "compact_import": "Input import / conversion",
        "compact_parameters": "Analysis parameters",
        "compact_environment": "Environment and installation",
        "compact_results": "Results and QC",
        "compact_run": "Run SlicerDSpace",
        "compact_log": "Log",
        "input_format_label": "Input format",
        "input_format_nifti": "NIfTI",
        "input_format_dicom": "DICOM",
        "input_format_nrrd": "NRRD",
        "input_format_placeholder_nifti": "Folder with input/data.nii.gz, bvals, bvecs (or .nii.gz + .bval/.bvec)",
        "input_format_placeholder_dicom": "Folder containing a single diffusion DICOM series",
        "input_format_placeholder_nrrd": "Diffusion .nrrd / .nhdr file or folder",
        "input_format_help_title": "Input format & paths",
        "input_format_help_close": "Close",
        "input_format_help": (
            "Input formats:\n"
            "• NIfTI: Folder with input/data.nii.gz, bvals, bvecs (optional nodif_brain_mask.nii.gz, dwi.json), "
            "or a folder with .nii(.gz) + .bval/.bvec.\n"
            "• DICOM: Folder containing one diffusion DICOM series (.dcm/.ima); converted automatically on run.\n"
            "• NRRD: Diffusion .nrrd/.nhdr volume or folder; converted automatically on run.\n\n"
            "Path fields below:\n"
            "• Subject ID: Prefix for outputs (auto-filled from folder/file name if left blank).\n"
            "• Subject root: Select the input folder (or NRRD/NIfTI file).\n"
            "• Output directory: Optional; defaults to <Subject root>/outputs."
        ),
        "compact_preprocessing": "Selected: DWI denoising + Eddy + Topup",
        "compact_surface": "DDSurfer white / pial surfaces",
        "compact_postprocess": "Cortical morphometry / parcellation",
        "compact_parcellation": "DDParcel regional labels",
        "compact_segmentation": "DDSeg WM / GM / CSF",
        "compact_microstructure": "Selected: DTI",
        "compact_tractography": "Selected: MRtrix3 CSD / ACT",
        "compact_bundles": "Selected: WMQL",
        "compact_connectome": "Selected: SIFT2",
        "section_parameters": "3. Route and parameters",
        "section_run": "4. Run",
        "section_results": "5. Results",
        "topup_group": "Distortion correction (topup)",
        "phase_encoding_direction": "Phase-encoding direction",
        "readout_time": "Total readout time (s)",
        "reverse_phase_b0": "Reverse-PE b=0 image",
        "brain_mask_backend": "Brain masking",
        "brain_mask_device": "Mask device",
        "brain_mask_device_tooltip": "PNL CNN: auto prefers the first visible CUDA GPU in the selected runtime. CUDA requires a GPU; CPU explicitly disables it. Runtime GPU failures are not silently retried.",
        "pnl_mask_route": "PNL route",
        "pnl_mask_route_tooltip": (
            "PNL CNN route. auto: managed (conda runner with ANTs) on Linux, native on macOS and Windows. "
            "native (experimental): did not meet the M-A parity rule on 2 of 24 cases (T2 design §3)."
        ),
        "pnl_route_auto": "auto",
        "pnl_route_managed": "managed",
        "pnl_route_native": "native (experimental)",
        "brain_mask_pnl_cnn": "PNL CNN (dMRI)",
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
            "Brain masking and optional DWI denoising, motion/eddy and distortion (topup) correction; "
            "uncheck for externally preprocessed data."
        ),
        "parcellation_tooltip": (
            "Whole-brain anatomical parcellation (101 cortical and subcortical regions) in native dMRI space via DDParcel."
        ),
        "segmentation_tooltip": (
            "White matter, gray matter and CSF segmentation and tissue probability maps in native dMRI space via DDSeg."
        ),
        "microstructure_tooltip": (
            "Diffusion microstructure maps (DTI, FW, NODDI, DKI / MK-Curve, iFW) and regional statistics."
        ),
        "tractography_tooltip": (
            "Whole-brain tractography (CSD / ACT, UKF, DTI) and surface-based short association fiber (SAF) tracking."
        ),
        "bundles_tooltip": (
            "White-matter tract identification via rule-based WMQL, atlas-guided WMA or deep-learning TractCloud."
        ),
        "connectome_tooltip": (
            "Structural connectivity matrices across DDParcel regions weighted by SIFT2 or fiber count."
        ),
        "compute_device_auto": "Auto (CUDA / CPU)",
        "compute_device_tooltip": (
            "Select CPU or a specific CUDA GPU for neural inference (DDSurfer, DDParcel, DDSeg, TractCloud, PNL mask). "
            "Auto uses CUDA:0 when available."
        ),
    },
    "zh": {
        "section_getting_started": "快速开始",
        "section_inputs": "1. 输入",
        "section_targets": "2. 分析任务",
        "compact_advanced": "高级设置",
        "compact_advanced_hint": "点击 ... 选择算法；下方按需调整参数。",
        "input_layout_found": "已找到输入文件结构；点击运行时将检查依赖。",
        "compact_algorithm_selection": "当前：{algorithms}",
        "compact_algorithm_tooltip": "当前已选：{algorithms}（点击 ... 按钮配置）。",
        "compact_import": "输入导入 / 格式转换",
        "compact_parameters": "分析参数",
        "compact_environment": "环境与安装",
        "compact_results": "结果与质控",
        "compact_run": "运行 SlicerDSpace",
        "compact_log": "日志",
        "input_format_label": "输入格式",
        "input_format_nifti": "NIfTI",
        "input_format_dicom": "DICOM",
        "input_format_nrrd": "NRRD",
        "input_format_placeholder_nifti": "含 input/data.nii.gz、bvals、bvecs（或 .nii.gz + .bval/.bvec）的目录",
        "input_format_placeholder_dicom": "含单个扩散加权 DICOM 序列的文件夹",
        "input_format_placeholder_nrrd": "扩散加权 .nrrd / .nhdr 文件或所在目录",
        "input_format_help_title": "输入格式与路径说明",
        "input_format_help_close": "关闭",
        "input_format_help": (
            "输入格式要求：\n"
            "• NIfTI：含 input/data.nii.gz、bvals、bvecs 的目录（可选 nodif_brain_mask.nii.gz、dwi.json），"
            "或含同名 .nii(.gz) + .bval/.bvec 的目录。\n"
            "• DICOM：含单个扩散加权 DICOM 序列（.dcm/.ima）的文件夹，运行时自动转换。\n"
            "• NRRD：扩散加权 .nrrd/.nhdr 文件或所在目录，运行时自动转换。\n\n"
            "下方路径填写说明：\n"
            "• 受试者 ID：输出文件名前缀（留空则根据所选路径自动填入）。\n"
            "• 受试者根目录：选择上述输入目录（或 NRRD/NIfTI 文件）。\n"
            "• 输出目录：选填，留空默认输出至 <受试者根目录>/outputs。"
        ),
        "compact_preprocessing": "当前：DWI denoising + Eddy + Topup",
        "compact_surface": "DDSurfer 白质 / 软脑膜表面",
        "compact_postprocess": "皮层形态学 / 分区",
        "compact_parcellation": "DDParcel 脑区标签",
        "compact_segmentation": "DDSeg 白质 / 灰质 / 脑脊液",
        "compact_microstructure": "当前：DTI",
        "compact_tractography": "当前：MRtrix3 CSD / ACT",
        "compact_bundles": "当前：WMQL",
        "compact_connectome": "当前：SIFT2",
        "section_parameters": "3. 路线与参数",
        "section_run": "4. 运行",
        "section_results": "5. 结果",
        "topup_group": "畸变校正（topup）",
        "phase_encoding_direction": "相位编码方向",
        "readout_time": "总读出时间（秒）",
        "reverse_phase_b0": "反向相位编码 b=0 图像",
        "brain_mask_backend": "脑掩膜方法",
        "brain_mask_device": "掩膜计算设备",
        "brain_mask_device_tooltip": "PNL CNN：Auto 优先使用所选运行环境中可见的第一个 CUDA GPU；CUDA 要求 GPU 可用，CPU 显式禁用 GPU。GPU 运行错误不会静默重试。",
        "pnl_mask_route": "PNL 路线",
        "pnl_mask_route_tooltip": (
            "PNL CNN 路线。auto：Linux 用 managed（conda 运行环境，含 ANTs），macOS 和 Windows 用 native。"
            "native（实验性）：在 M-A 一致性规则的 24 例中有 2 例未达标（T2 设计 §3）。"
        ),
        "pnl_route_auto": "auto",
        "pnl_route_managed": "managed",
        "pnl_route_native": "native（实验性）",
        "brain_mask_pnl_cnn": "PNL CNN (dMRI)",
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
            "尊重已有的用户提供掩膜；此选项仅控制缺失掩膜的生成。默认 PNL CNN (dMRI)，"
            "需在“高级设置 > 环境与模型 > 外部运行路径”配置外部 Python、官方 dwi_masking.py、"
            "模型目录和 ANTs 路径。"
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
        "preprocessing_tooltip": "脑掩膜生成及可选的 DWI 降噪、运动/涡流（eddy）与畸变（topup）校正；外部已预处理数据可取消勾选。",
        "parcellation_tooltip": "基于 DDParcel 在原生 dMRI 空间生成全脑 101 区皮层及皮层下解剖分区标签。",
        "segmentation_tooltip": "基于 DDSeg 在原生 dMRI 空间生成白质、灰质、脑脊液分割标签及组织概率图。",
        "microstructure_tooltip": "扩散微结构参数图（DTI、FW、NODDI、DKI / MK-Curve、iFW）及脑区统计指标。",
        "tractography_tooltip": "全脑纤维追踪（CSD / ACT、UKF、DTI）及基于皮层表面的短联络纤维（SAF）追踪。",
        "bundles_tooltip": "白质纤维束识别与提取（规则查询 WMQL、图谱引导 WMA 或深度学习 TractCloud）。",
        "connectome_tooltip": "基于 DDParcel 分区的 SIFT2 加权或纤维计数（Fiber count）结构连接矩阵。",
        "compute_device_auto": "自动 (CUDA / CPU)",
        "compute_device_tooltip": "选择用于神经网络推理（DDSurfer、DDParcel、DDSeg、TractCloud、PNL 掩膜）的计算设备；自动模式优先使用 CUDA:0。",
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
    "targetPreprocessingLabel": "preprocessing_tooltip",
    "targetPreprocessingCheckBox": "preprocessing_tooltip",
    "targetParcellationLabel": "parcellation_tooltip",
    "targetParcellationCheckBox": "parcellation_tooltip",
    "targetSegmentationLabel": "segmentation_tooltip",
    "targetSegmentationCheckBox": "segmentation_tooltip",
    "targetMicrostructureLabel": "microstructure_tooltip",
    "targetMicrostructureCheckBox": "microstructure_tooltip",
    "targetTractographyLabel": "tractography_tooltip",
    "targetTractographyCheckBox": "tractography_tooltip",
    "targetBundlesLabel": "bundles_tooltip",
    "targetBundlesCheckBox": "bundles_tooltip",
    "targetConnectomeLabel": "connectome_tooltip",
    "targetConnectomeCheckBox": "connectome_tooltip",
}
