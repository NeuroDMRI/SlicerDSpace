"""Beginner-facing onboarding helpers for the SlicerDSpace UI."""

from __future__ import annotations

import json
import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shared.demo_data import DEMO_SUBJECT_ID, create_synthetic_demo_subject
from shared.user_config import default_user_config_path, load_user_config, update_user_config
from shared.ui_workflow_text import (
    BRAIN_MASK_BACKEND_TEXT_KEYS,
    PNL_MASK_ROUTE_TEXT_KEYS,
    WORKFLOW_TEXT,
    WORKFLOW_TEXT_BINDINGS,
    WORKFLOW_TOOLTIP_BINDINGS,
)


LANGUAGE_SETTING = "SlicerDSpace/UserSettings/language"
FIRST_RUN_SETTING = "SlicerDSpace/UserSettings/firstRunWizardShown"

TRANSLATIONS = {
    "en": {
        "beginner_title": "Beginner setup",
        "beginner_body": (
            "Start here if you are new: run the setup wizard, try a synthetic demo subject, "
            "then check what this computer can run."
        ),
        "wizard": "First-run setup wizard",
        "demo": "Demo / Try with sample data",
        "copy_diag": "Copy diagnostic report",
        "open_config": "Open settings JSON",
        "language": "Language",
        "dashboard_waiting": "Run Check environment dependencies to see what is runnable on this computer.",
        "path_examples": (
            "Examples: Subject root /Users/name/data/sub-01 | Output /Users/name/data/sub-01/outputs | "
            "Model assets /Users/name/SlicerDSpace-model-assets | MRtrix bin /opt/mrtrix3/bin"
        ),
        "headline": "SlicerDSpace",
        "subtitle": (
            "Prepare inputs. Choose analyses. Review results."
        ),
        "check_env": "Check environment dependencies",
        "preflight_run": "Run SlicerDSpace",
        "status_waiting": "Waiting for input",
        "status_waiting_ellipsis": "Waiting for input...",
        "environment_section": "Advanced: Dependency profile",
        "environment_not_checked": "No environment check has been run yet.",
        "torch_device": "Compute device",
        "missing_python_dependencies": "First-use setup",
        "install_python_dependencies": "Offer to prepare missing dependencies before running",
        "subject_id": "Subject ID",
        "subject_id_placeholder": "e.g. 100307",
        "subject_root": "Subject root",
        "subject_root_placeholder": "Folder with input/data.nii.gz, bvals, bvecs",
        "output_directory": "Output directory",
        "output_directory_placeholder": "Optional; defaults to <subject root>/outputs",
        "preprocessing": "Preprocessing",
        "preprocessing_option": "DWI denoising + Eddy + Topup",
        "surface": "Cortical surfaces",
        "surface_option": "DDSurfer white / pial surfaces",
        "surface_tooltip": (
            "Bilateral white and pial cortical surfaces reconstructed in native dMRI space via DDSurfer."
        ),
        "surface_postprocess": "Surface postprocess",
        "surface_postprocess_option": "FreeSurfer-style postprocess",
        "surface_postprocess_tooltip": (
            "Cortical morphometry and parcellation from DDSurfer meshes "
            "(uses verified native components; a full FreeSurfer installation is optional, license required)."
        ),
        "parcellation": "Brain parcellation",
        "parcellation_option": "DDParcel regional labels",
        "segmentation": "Tissue segmentation",
        "segmentation_option": "DDSeg WM / GM / CSF",
        "microstructure": "Microstructure",
        "microstructure_option": "DTI maps; optional FW / NODDI / DKI in Advanced",
        "tractography": "Tractography",
        "tractography_option": "DTI or CSD / ACT tractogram",
        "bundles": "Fiber bundles",
        "bundles_option": "WMQL / WMA / TractCloud",
        "connectome": "Connectome",
        "connectome_option": "Connectivity matrices",
        "advanced_settings": "Analysis settings",
        "advanced_hint": "Change these only for non-default backends, batch runs, or optional microstructure stages.",
        "run_name": "Run name",
        "tractography_backend": "Tractography backend",
        "bundle_backend": "Bundle backend",
        "wmql_definition": "WMQL definition",
        "wmql_bundles": "WMQL bundles (empty = all)",
        "parallel_models": "Concurrent microstructure models",
        "wma_input_tractography": "WMA tractography input",
        "wma_input_placeholder": "Automatic (UKF tractogram)",
        "act_saf_streamlines": "ACT/SAF streamlines",
        "microstructure_stages": "Microstructure stages",
        "batch_mode": "Batch mode",
        "process_all_subjects": "Process all subjects under a study root",
        "batch_root": "Batch root",
        "batch_size": "Batch size",
        "arrange_inputs": "Arrange inputs",
        "run_plan_pending": "Run plan has not been generated yet.",
        "run_summary_pending": "Run summary will appear after a plan or run completes.",
        "open_latest_report": "Open report",
        "open_output_tree": "Open output tree",
        "target": "Target",
        "state": "State",
        "reason": "Reason",
        "manifest": "Manifest",
        "qc": "QC",
        "logs": "Logs",
        "cancel_job": "Cancel running job",
        "prepare_inputs": "Prepare Inputs",
        "prepare_inputs_tooltip": "Normalize DICOM, NRRD/NHDR, or NIfTI+bval/bvec inputs into a SlicerDSpace subject root.",
        "prepare_summary": "Organize external DICOM, NRRD, or NIfTI + bval/bvec files into <subject root>/input/.",
        "source_type": "Source type",
        "source_type_tooltip": "Use auto unless you need to force a specific source type.",
        "source": "Source",
        "source_placeholder": "DICOM folder, DWI NIfTI, NRRD/NHDR, or canonical subject root",
        "prepare_subject_root_placeholder": "Destination subject root, e.g. /data/study/sub-01",
        "prepare_subject_id_tooltip": "Subject identifier recorded in input/input_import_manifest.json.",
        "sidecar_summary": "Optional sidecars for NIfTI inputs or externally generated masks.",
        "bval": "bval",
        "bval_placeholder": "Optional .bval sidecar",
        "bvec": "bvec",
        "bvec_placeholder": "Optional .bvec sidecar",
        "metadata_json": "Metadata JSON",
        "metadata_placeholder": "Optional BIDS-style .json sidecar",
        "brain_mask": "Brain mask",
        "brain_mask_placeholder": "Optional nodif_brain_mask.nii.gz",
        "backend": "Backend",
        "backend_tooltip": "Use auto unless you need to force dcm2niix, MRtrix3 mrconvert, SlicerDMRI DWIConvert, or copy.",
        "output": "Output",
        "output_tooltip": "canonical-nifti is preferred. compat-nrrd is available for legacy DWIConvert routes.",
        "options": "Options",
        "preview_plan": "Preview plan",
        "preview_plan_tooltip": "Build and show the input import plan without writing files.",
        "prepare_inputs_action": "Prepare inputs",
        "prepare_inputs_action_tooltip": "Create the canonical subject-root input folder and write input_import_manifest.json.",
        "open_manifest": "Open manifest",
        "open_prepare_manifest_tooltip": "Open input/input_import_manifest.json for the selected subject root.",
        "model_manifest_tooltip": "Open models/assets_manifest.json.",
        "dry_run": "Dry run",
        "dry_run_tooltip": "Plan the import without writing files.",
        "overwrite_files": "Overwrite existing files",
        "overwrite_files_tooltip": "Allow replacing existing files in the destination input folder.",
        "actions": "Actions",
        "browse": "Browse...",
        "external_runtime_paths": "External Runtime Paths",
        "external_runtime_paths_tooltip": (
            "Discovery order: saved path, environment variables, PATH, then common install paths. "
            "Leave fields empty to rely on automatic discovery."
        ),
        "env_auto_detect": "Auto-detect if empty",
        "env_not_checked": "not checked",
        "env_check_pending": "check pending",
        "env_not_found": "not found",
        "env_not_used": "not used by current selection",
        "env_using_saved_path": "using saved path",
        "config": "Config",
        "import_dependency_config": "Import dependency config",
        "import_dependency_config_tooltip": "Load a SlicerDSpace dependency config JSON and save the external paths into Slicer settings.",
        "export_dependency_config": "Export dependency config",
        "export_dependency_config_tooltip": "Save the currently configured external paths as a CLI-compatible JSON config.",
        "open_environment_report": "Open environment report",
        "open_environment_report_tooltip": "Open the latest environment_doctor.json directory after a dependency check.",
        "open_install_logs": "Open install logs",
        "managed_runtime_setup": "Prepare selected analysis environment",
        "managed_runtime_resume": "Continue preparing selected analysis environment",
        "managed_runtime_setup_tooltip": "Prepare only selected extensions, Python packages, models and supported isolated tools. Review the plan and licenses; verified paths are saved automatically. Some changes require a Slicer restart.",
        "offline_model_resources": "Use offline model resources",
        "environment_more_options": "More options (offline, repair, paths)",
        "environment_details": "View dependency details",
        "repair_managed_tools": "Repair selected managed tools",
        "repair_managed_tools_tooltip": "Review and reinstall selected plugin-managed external tools. Does not repair arbitrary existing environments or change algorithm selections.",
        "setup_restart_required": "Restart Slicer to finish dependency setup before running analysis.",
        "open_install_logs_tooltip": "Open the latest Python dependency install logs/results directory.",
        "model_assets_bundle": "Model assets bundle",
        "check_status": "Check status",
        "check_status_tooltip": "Check whether core and all model asset profiles are present and not Git LFS pointer files.",
        "verify_core": "Verify core",
        "verify_core_tooltip": "Verify the core model asset profile required by DDSurfer, DDSeg, and DDParcel.",
        "write_lockfile": "Write lockfile",
        "write_lockfile_tooltip": "Write a SHA256 lockfile for the core model asset profile.",
        "open_guide": "Open guide",
        "open_guide_tooltip": "Open docs/assets/MODEL_ASSETS.md for unified model weights/template download and verification.",
        "terms": "Terms",
        "terms_tooltip": "Open docs/assets/MODEL_ASSET_TERMS.md.",
        "provenance": "Provenance",
        "provenance_tooltip": "Show model asset profile, verification, lockfile, and release-boundary notes.",
        "release_notes": "Release notes",
        "release_notes_tooltip": "Open docs/release/RELEASE_NOTES_v0.1.0.md.",
        "select_path": "Select",
        "log_title": "Pipeline Log",
        "log_hint": "Live background log. Long runs keep the latest messages for UI responsiveness.",
        "clear": "Clear",
        "environment_ready": "Environment ready",
        "environment_check_failed": "Environment check failed",
        "installing_dependencies": "Installing Python dependencies...",
        "dependency_install_failed": "Dependency installation failed",
        "run_planning_failed": "Run planning failed",
        "running_pipeline": "Running pipeline...",
        "pipeline_completed": "Pipeline completed",
        "pipeline_cancelled": "Pipeline cancelled",
        "background_submitted": "Background execution submitted.",
        "failed_start_background": "Failed to start background job",
        "failed_start_batch": "Failed to start batch job",
        "pipeline_no_manifest": "Pipeline failed: no manifest",
        "pipeline_invalid_manifest": "Pipeline failed: invalid manifest",
        "pipeline_no_manifest_detail": "Pipeline failed: no manifest path was reported.",
        "batch_completed": "Batch completed",
        "cancelling_background": "Cancelling background job...",
        "cancelled_summary": "Status: cancelled | User cancelled the background job.",
    },
    "zh": {
        "beginner_title": "新手设置",
        "beginner_body": "如果你不熟悉环境配置，从这里开始：运行向导、尝试合成样例，然后检查这台电脑能跑哪些功能。",
        "wizard": "首次运行设置向导",
        "demo": "Demo / 我只是想试一下",
        "copy_diag": "复制诊断报告",
        "open_config": "打开设置 JSON",
        "language": "语言",
        "dashboard_waiting": "点击“检查环境依赖”后，这里会显示这台电脑现在能跑什么、还缺什么。",
        "path_examples": (
            "路径示例：Subject root /Users/name/data/sub-01 | Output /Users/name/data/sub-01/outputs | "
            "模型权重 /Users/name/SlicerDSpace-model-assets | MRtrix bin /opt/mrtrix3/bin"
        ),
        "headline": "SlicerDSpace 运行工作台",
        "subtitle": "准备一个 subject root，检查依赖环境，并从同一个监控流程运行可复用的 dMRI 分析产品。",
        "check_env": "检查环境依赖",
        "preflight_run": "运行 SlicerDSpace",
        "status_waiting": "等待输入",
        "status_waiting_ellipsis": "等待输入…",
        "environment_section": "高级：依赖配置",
        "environment_not_checked": "尚未检查环境依赖。",
        "torch_device": "计算设备",
        "missing_python_dependencies": "首次使用准备",
        "install_python_dependencies": "运行前提示一键准备缺少的依赖",
        "subject_id": "受试者 ID",
        "subject_id_placeholder": "例如 100307",
        "subject_root": "受试者根目录",
        "subject_root_placeholder": "包含 input/data.nii.gz、bvals、bvecs 的文件夹",
        "output_directory": "输出目录",
        "output_directory_placeholder": "可选；默认为 <subject root>/outputs",
        "preprocessing": "预处理",
        "preprocessing_option": "DWI denoising + Eddy + Topup",
        "surface": "皮层表面",
        "surface_option": "DDSurfer 白质 / 软脑膜表面",
        "surface_tooltip": "基于 DDSurfer 在原生 dMRI 空间直接重建双侧白质与软脑膜皮层表面。",
        "surface_postprocess": "表面后处理",
        "surface_postprocess_option": "FreeSurfer 风格后处理",
        "surface_postprocess_tooltip": (
            "基于 DDSurfer 网格计算皮层形态学与表面分区（使用经校验的原生组件，不强制安装完整 FreeSurfer，需许可证）。"
        ),
        "parcellation": "脑区分割",
        "parcellation_option": "DDParcel aparc / wmparc 标签",
        "segmentation": "组织分割",
        "segmentation_option": "DDSeg 白质 / 灰质 / 脑脊液",
        "microstructure": "微结构",
        "microstructure_option": "DTI 参数图；高级设置可选 FW / NODDI / DKI",
        "tractography": "纤维追踪",
        "tractography_option": "DTI 或 CSD / ACT 纤维追踪",
        "bundles": "纤维束",
        "bundles_option": "WMQL / WMA / TractCloud",
        "connectome": "连接组",
        "connectome_option": "结构连接矩阵",
        "advanced_settings": "高级设置：后端、批处理、流线数量",
        "advanced_hint": "仅在使用非默认后端、批处理或可选微结构阶段时修改这些设置。",
        "run_name": "运行名称",
        "tractography_backend": "纤维追踪后端",
        "bundle_backend": "纤维束后端",
        "wmql_definition": "WMQL 解剖查询定义",
        "wmql_bundles": "WMQL 束子集（留空为全部）",
        "parallel_models": "微结构模型并发数",
        "wma_input_tractography": "WMA 纤维追踪输入",
        "wma_input_placeholder": "自动（UKF 流线）",
        "act_saf_streamlines": "ACT/SAF 流线数量",
        "microstructure_stages": "微结构阶段",
        "batch_mode": "批处理模式",
        "process_all_subjects": "处理研究根目录下的所有受试者",
        "batch_root": "批处理根目录",
        "batch_size": "并行批大小",
        "arrange_inputs": "整理输入",
        "run_plan_pending": "尚未生成运行计划。",
        "run_summary_pending": "生成计划或运行完成后将在此显示摘要。",
        "open_latest_report": "打开报告",
        "open_output_tree": "打开输出目录报告",
        "target": "目标",
        "state": "状态",
        "reason": "原因",
        "manifest": "清单",
        "qc": "质控",
        "logs": "日志",
        "cancel_job": "取消正在运行的任务",
        "prepare_inputs": "准备输入",
        "prepare_inputs_tooltip": "将 DICOM、NRRD/NHDR 或 NIfTI+bval/bvec 输入规范化为 SlicerDSpace 受试者根目录。",
        "prepare_summary": "将外部 DICOM、NRRD 或 NIfTI + bval/bvec 文件整理为标准 <subject root>/input/ 目录。",
        "source_type": "来源类型",
        "source_type_tooltip": "通常使用 auto；仅在需要强制指定来源类型时修改。",
        "source": "来源",
        "source_placeholder": "DICOM 文件夹、DWI NIfTI、NRRD/NHDR 或规范受试者根目录",
        "prepare_subject_root_placeholder": "目标受试者根目录，例如 /data/study/sub-01",
        "prepare_subject_id_tooltip": "记录到 input/input_import_manifest.json 的受试者标识。",
        "sidecar_summary": "NIfTI 输入或外部生成掩膜的可选辅助文件。",
        "bval": "bval",
        "bval_placeholder": "可选 .bval 辅助文件",
        "bvec": "bvec",
        "bvec_placeholder": "可选 .bvec 辅助文件",
        "metadata_json": "元数据 JSON",
        "metadata_placeholder": "可选 BIDS 风格 .json 辅助文件",
        "brain_mask": "脑掩膜",
        "brain_mask_placeholder": "可选 nodif_brain_mask.nii.gz",
        "backend": "后端",
        "backend_tooltip": "通常使用 auto；仅在需要强制使用 dcm2niix、MRtrix3 mrconvert、SlicerDMRI DWIConvert 或 copy 时修改。",
        "output": "输出",
        "output_tooltip": "推荐 canonical-nifti；compat-nrrd 仅用于兼容旧版 DWIConvert 流程。",
        "options": "选项",
        "preview_plan": "预览计划",
        "preview_plan_tooltip": "构建并显示输入导入计划，但不写入文件。",
        "prepare_inputs_action": "准备输入",
        "prepare_inputs_action_tooltip": "创建规范受试者根目录的 input 文件夹，并写入 input_import_manifest.json。",
        "open_manifest": "打开清单",
        "open_prepare_manifest_tooltip": "打开所选受试者根目录中的 input/input_import_manifest.json。",
        "model_manifest_tooltip": "打开 models/assets_manifest.json。",
        "dry_run": "仅预演",
        "dry_run_tooltip": "仅规划导入，不写入文件。",
        "overwrite_files": "覆盖现有文件",
        "overwrite_files_tooltip": "允许替换目标 input 文件夹中的现有文件。",
        "actions": "操作",
        "browse": "浏览…",
        "external_runtime_paths": "外部运行环境路径",
        "external_runtime_paths_tooltip": "按已保存路径、环境变量、PATH、常见安装路径的顺序查找。留空即可使用自动检测。",
        "env_auto_detect": "留空时自动检测",
        "env_not_checked": "未检查",
        "env_check_pending": "等待检查",
        "env_not_found": "未找到",
        "env_not_used": "当前选择未使用",
        "env_using_saved_path": "使用已保存路径",
        "config": "配置",
        "import_dependency_config": "导入依赖配置",
        "import_dependency_config_tooltip": "加载 SlicerDSpace 依赖配置 JSON，并将外部路径保存到 Slicer 设置。",
        "export_dependency_config": "导出依赖配置",
        "export_dependency_config_tooltip": "将当前外部路径保存为兼容 CLI 的 JSON 配置。",
        "open_environment_report": "打开环境报告",
        "open_environment_report_tooltip": "依赖检查后打开最新 environment_doctor.json 所在目录。",
        "open_install_logs": "打开安装日志",
        "managed_runtime_setup": "准备所选分析环境",
        "managed_runtime_resume": "继续准备所选分析环境",
        "managed_runtime_setup_tooltip": "按所选任务准备扩展、Python 包、模型及受支持的隔离工具；确认计划和许可后，自动保存通过校验的路径。部分安装需要重启 Slicer。",
        "offline_model_resources": "使用离线模型资源",
        "environment_more_options": "更多选项（离线、修复、路径）",
        "environment_details": "查看依赖详情",
        "repair_managed_tools": "修复所选受管工具",
        "repair_managed_tools_tooltip": "确认后重新安装所选的插件受管外部工具；不修改自行配置的环境或算法选择。",
        "setup_restart_required": "请重启 Slicer 完成依赖准备，再运行分析。",
        "open_install_logs_tooltip": "打开最新 Python 依赖安装日志和结果目录。",
        "model_assets_bundle": "模型资源包",
        "check_status": "检查状态",
        "check_status_tooltip": "检查核心及全部模型资源配置是否存在，并确认它们不是 Git LFS 指针文件。",
        "verify_core": "验证核心资源",
        "verify_core_tooltip": "验证 DDSurfer、DDSeg 和 DDParcel 所需的核心模型资源。",
        "write_lockfile": "写入锁定文件",
        "write_lockfile_tooltip": "为核心模型资源写入 SHA256 锁定文件。",
        "open_guide": "打开指南",
        "open_guide_tooltip": "打开 docs/assets/MODEL_ASSETS.md，查看统一模型权重/模板的下载与验证说明。",
        "terms": "条款",
        "terms_tooltip": "打开 docs/assets/MODEL_ASSET_TERMS.md。",
        "provenance": "来源信息",
        "provenance_tooltip": "显示模型资源配置、验证、锁定文件及发布边界说明。",
        "release_notes": "发行说明",
        "release_notes_tooltip": "打开 docs/release/RELEASE_NOTES_v0.1.0.md。",
        "select_path": "选择",
        "log_title": "Pipeline 日志",
        "log_hint": "实时后台日志。长时间运行时仅保留最新消息，以确保界面响应。",
        "clear": "清空",
        "environment_ready": "环境已就绪",
        "environment_check_failed": "环境检查失败",
        "installing_dependencies": "正在安装 Python 依赖…",
        "dependency_install_failed": "依赖安装失败",
        "run_planning_failed": "运行规划失败",
        "running_pipeline": "正在运行 Pipeline…",
        "pipeline_completed": "Pipeline 已完成",
        "pipeline_cancelled": "Pipeline 已取消",
        "background_submitted": "后台任务已提交。",
        "failed_start_background": "后台任务启动失败",
        "failed_start_batch": "批处理任务启动失败",
        "pipeline_no_manifest": "Pipeline 失败：没有清单",
        "pipeline_invalid_manifest": "Pipeline 失败：清单无效",
        "pipeline_no_manifest_detail": "Pipeline 失败：后台进程未报告清单路径。",
        "batch_completed": "批处理已完成",
        "cancelling_background": "正在取消后台任务…",
        "cancelled_summary": "状态：已取消 | 用户取消了后台任务。",
    },
}


ENVIRONMENT_SECTION_TEXT = {
    "en": {
        "model-assets": (
            "Model Asset Paths",
            "User-data folder for DDSurfer, DDSeg, and DDParcel weights/templates.",
        ),
        "core-runtime": (
            "Core Python & Runtime Paths",
            "Shared Python and external tools; leave empty to auto-detect from saved settings, environment variables, or PATH.",
        ),
        "bundle-backends": (
            "Optional Backend Paths",
            "Optional bundle-extraction backends (WMA and TractCloud).",
        ),
        "wma": ("WMA Paths", "WhiteMatterAnalysis checkout, ORG atlas, and helper Slicer/CLI paths."),
        "tractcloud": ("TractCloud Paths", "TractCloud checkout, Python runtime, and model/data directory."),
    },
    "zh": {
        "model-assets": (
            "模型资源路径",
            "存放 DDSurfer、DDSeg 和 DDParcel 权重与模板的用户目录。",
        ),
        "core-runtime": (
            "核心 Python 与运行环境路径",
            "共用的 Python 与外部工具链；留空时自动从已保存设置、环境变量或 PATH 检测。",
        ),
        "bundle-backends": (
            "可选后端路径",
            "可选纤维束提取后端（WMA 与 TractCloud）。",
        ),
        "wma": ("WMA 路径", "WhiteMatterAnalysis 代码、ORG atlas 以及辅助 Slicer/CLI 路径。"),
        "tractcloud": ("TractCloud 路径", "TractCloud 代码、Python 运行环境及模型/数据目录。"),
    },
}


ENVIRONMENT_DEPENDENCY_TEXT = {
    "en": {
        "python_executable": (
            "Python executable",
            "Optional. Python executable for dependency checks and Python package installation. Defaults to current Slicer Python.",
        ),
        "python_environment": (
            "Python/conda environment",
            "Optional. Conda environment name or Python executable path. Explicit Python executable takes precedence.",
        ),
        "conda_executable": (
            "conda/mamba executable",
            "Optional. conda or mamba executable used when Python/conda environment is an environment name.",
        ),
        "model_assets_dir": (
            "Model assets root",
            "Optional. Root directory for downloaded SlicerDSpace model assets. Defaults to SLICERDSPACE_MODEL_ASSETS_DIR or a per-user data directory.",
        ),
        "mrtrix_bin_dir": (
            "MRtrix3 bin dir",
            "Optional. Directory containing mrconvert, dwi2response, dwi2fod, tckgen, tcksift2.",
        ),
        "preprocessing_fsl_bin_dir": (
            "FSL bin dir",
            "Optional. Directory containing eddy and topup for MRtrix/FSL preprocessing.",
        ),
        "postprocess_freesurfer_home": (
            "FreeSurfer home (optional)",
            "Optional existing installation. Surface processing can instead use verified native components. FREESURFER_HOME is detected.",
        ),
        "postprocess_runtime": ("Surface components (optional)", "Verified runtime containing runtime.json; otherwise the matching user cache is detected. Supports qualified prebuilt bundles."),
        "postprocess_license_file": ("FreeSurfer activation file", "Your own activation file outside the runtime; FS_LICENSE is also detected. Never downloaded or bundled."),
        "pnl_mask_python": (
            "PNL mask Python",
            "Python executable in the external PNL CNN dMRI masking environment; not the general dependency-install Python.",
        ),
        "pnl_mask_script": (
            "PNL dwi_masking.py",
            "Path to the official external PNL CNN dwi_masking.py script. No alternative masking backend is selected automatically.",
        ),
        "pnl_mask_model_dir": (
            "PNL mask model dir",
            "External PNL CNN model directory. Configure it when generating a missing mask with PNL CNN; supplied masks are reused.",
        ),
        "pnl_mask_ants_bin_dir": (
            "PNL mask ANTs bin dir",
            "Directory containing ANTs executables required by the external PNL CNN masking script.",
        ),
        "dki_matlab_executable": (
            "MK-Curve MATLAB executable",
            "External MATLAB executable for MKCURVE_MATLAB only. Required when MKCURVE_MATLAB is selected; missing MATLAB blocks execution without fallback.",
        ),
        "dki_mkcurve_root": (
            "Original MK-Curve checkout",
            "Directory containing the author's original external MATLAB MK-Curve code. Standard DKI WLS/OLS does not use this path.",
        ),
        "tract_querier_root": (
            "TractQuerier root",
            "Optional. Checkout root used by WMQL when tract_querier is not installed in Slicer Python.",
        ),
        "wmql_python_executable": (
            "WMQL Python",
            "Optional. Use a dedicated Python executable with a native TractQuerier checkout.",
        ),
        "wma_root": ("WMA root", "Optional. External WhiteMatterAnalysis checkout root."),
        "wma_atlas_dir": (
            "WMA atlas directory",
            "Optional. Registration and 800-cluster root for the atlas chosen in Advanced settings.",
        ),
        "slicer_executable": (
            "Slicer executable for WMA",
            "Optional. Slicer executable used by WMA/SlicerDMRI helper calls.",
        ),
        "fiber_tract_measurements_cli": (
            "FiberTractMeasurements CLI",
            "Optional. SlicerDMRI FiberTractMeasurements CLI path.",
        ),
        "tractcloud_root": ("TractCloud root", "Optional. TractCloud package/check-out root."),
        "tractcloud_python_executable": (
            "TractCloud Python",
            "Optional. Dedicated Python executable for TractCloud if not using Slicer Python.",
        ),
        "tractcloud_data_dir": ("TractCloud data/model dir", "Optional. TractCloud model/data directory."),
    },
    "zh": {
        "python_executable": ("Python 可执行文件", "可选。用于依赖检查和 Python 包安装；默认使用当前 Slicer Python。"),
        "python_environment": ("Python/conda 环境", "可选。Conda 环境名称或 Python 可执行文件路径；显式 Python 路径优先。"),
        "conda_executable": ("conda/mamba 可执行文件", "可选。当 Python/conda 环境填写环境名称时，用它解析该环境。"),
        "model_assets_dir": ("模型资源根目录", "可选。下载的 SlicerDSpace 模型资源根目录；默认使用 SLICERDSPACE_MODEL_ASSETS_DIR 或用户数据目录。"),
        "mrtrix_bin_dir": ("MRtrix3 bin 目录", "可选。包含 mrconvert、dwi2response、dwi2fod、tckgen、tcksift2 的目录。"),
        "preprocessing_fsl_bin_dir": ("FSL bin 目录", "可选。包含 MRtrix/FSL 预处理所需 eddy 和 topup 的目录。"),
        "postprocess_freesurfer_home": ("FreeSurfer 主目录（可选）", "可选已有 FreeSurfer 安装根目录；表面后处理可改用经校验的原生组件；检测 FREESURFER_HOME。"),
        "postprocess_runtime": ("表面处理组件（可选）", "包含 runtime.json 的经校验运行目录，支持合格的预编译运行包；留空检测匹配的用户缓存。"),
        "postprocess_license_file": ("FreeSurfer 激活文件", "您自己的激活文件，放在组件目录外；也检测 FS_LICENSE。插件不会下载或打包私人激活文件。"),
        "pnl_mask_python": ("PNL 掩膜 Python", "外部 PNL CNN dMRI 脑掩膜环境的 Python 可执行文件，不是用于通用依赖安装的 Python。"),
        "pnl_mask_script": ("PNL dwi_masking.py", "外部官方 PNL CNN dwi_masking.py 脚本路径；不会自动切换其他脑掩膜后端。"),
        "pnl_mask_model_dir": ("PNL 掩膜模型目录", "外部 PNL CNN 模型目录。使用 PNL CNN 生成缺失掩膜时需配置；已有的用户掩膜会被复用。"),
        "pnl_mask_ants_bin_dir": ("PNL 掩膜 ANTs bin 目录", "包含外部 PNL CNN 脑掩膜脚本所需 ANTs 可执行文件的目录。"),
        "dki_matlab_executable": ("MK-Curve MATLAB 可执行文件", "仅用于 MKCURVE_MATLAB 的外部 MATLAB 可执行文件。选择 MKCURVE_MATLAB 时必须配置；缺少 MATLAB 时明确阻止执行，不会回退。"),
        "dki_mkcurve_root": ("作者原始 MK-Curve 代码目录", "包含作者原始外部 MATLAB MK-Curve 代码的目录；标准 DKI WLS/OLS 不使用此路径。"),
        "tract_querier_root": ("TractQuerier 根目录", "可选。当 Slicer Python 未安装 tract_querier 时，WMQL 使用的代码根目录。"),
        "wmql_python_executable": ("WMQL Python", "可选。与原生 TractQuerier 源码目录配合使用的 Python 可执行文件。"),
        "wma_root": ("WMA 根目录", "可选。外部 WhiteMatterAnalysis 代码根目录。"),
        "wma_atlas_dir": ("WMA 图谱目录", "可选。高级设置中所选图谱的配准图谱和 800 簇目录所在位置。"),
        "slicer_executable": ("WMA 使用的 Slicer", "可选。WMA/SlicerDMRI 辅助调用使用的 Slicer 可执行文件。"),
        "fiber_tract_measurements_cli": ("FiberTractMeasurements CLI", "可选。SlicerDMRI FiberTractMeasurements CLI 路径。"),
        "tractcloud_root": ("TractCloud 根目录", "可选。TractCloud 软件包/代码根目录。"),
        "tractcloud_python_executable": ("TractCloud Python", "可选。不使用 Slicer Python 时，TractCloud 专用的 Python 可执行文件。"),
        "tractcloud_data_dir": ("TractCloud 数据/模型目录", "可选。TractCloud 模型和数据目录。"),
    },
}


MAIN_UI_TEXT_BINDINGS = {
    "headlineLabel": "headline",
    "subtitleLabel": "subtitle",
    "environmentCollapsibleButton": "environment_section",
    "unifiedTorchDeviceLabel": "torch_device",
    "autoInstallPythonDependenciesLabel": "missing_python_dependencies",
    "autoInstallPythonDependenciesCheckBox": "install_python_dependencies",
    "subjectIdLabel": "subject_id",
    "inputDirectoryLabel": "subject_root",
    "outputDirectoryLabel": "output_directory",
    "targetPreprocessingLabel": "preprocessing",
    "targetPreprocessingCheckBox": "preprocessing_option",
    "targetSurfaceLabel": "surface",
    "targetSurfaceCheckBox": "surface_option",
    "targetSurfacePostprocessLabel": "surface_postprocess",
    "targetSurfacePostprocessCheckBox": "surface_postprocess_option",
    "targetParcellationLabel": "parcellation",
    "targetParcellationCheckBox": "parcellation_option",
    "targetSegmentationLabel": "segmentation",
    "targetSegmentationCheckBox": "segmentation_option",
    "targetMicrostructureLabel": "microstructure",
    "targetMicrostructureCheckBox": "microstructure_option",
    "targetTractographyLabel": "tractography",
    "targetTractographyCheckBox": "tractography_option",
    "targetBundlesLabel": "bundles",
    "targetBundlesCheckBox": "bundles_option",
    "targetConnectomeLabel": "connectome",
    "targetConnectomeCheckBox": "connectome_option",
    "advancedRunCollapsibleButton": "advanced_settings",
    "advancedRunSectionLabel": "advanced_hint",
    "orchestrationRunNameLabel": "run_name",
    "orchestrationTractographyBackendLabel": "tractography_backend",
    "orchestrationBundleBackendLabel": "bundle_backend",
    "wmqlQueryPresetLabel": "wmql_definition",
    "wmqlQuerySelectionLabel": "wmql_bundles",
    "parallelModelsLabel": "parallel_models",
    "wmaInputTractographyLabel": "wma_input_tractography",
    "orchestrationSelectStreamlinesLabel": "act_saf_streamlines",
    "orchestrationMicrostructureStagesLabel": "microstructure_stages",
    "batchModeLabel": "batch_mode",
    "batchModeCheckBox": "process_all_subjects",
    "batchRootLabel": "batch_root",
    "batchSizeLabel": "batch_size",
    "batchArrangeInputsLabel": "arrange_inputs",
    "validateButton": "check_env",
    "runSelectedButton": "preflight_run",
    "logHintLabel": "log_hint",
    "clearLogButton": "clear",
}


MAIN_UI_TOOLTIP_BINDINGS = {
    "targetSurfaceLabel": "surface_tooltip",
    "targetSurfaceCheckBox": "surface_tooltip",
    "targetSurfacePostprocessLabel": "surface_postprocess_tooltip",
    "targetSurfacePostprocessCheckBox": "surface_postprocess_tooltip",
}


DYNAMIC_UI_TEXT_BINDINGS = {
    "_cancelRunButton": "cancel_job",
    "_runReportButton": "open_latest_report",
    "_runOutputTreeButton": "open_output_tree",
    "_environmentAdvancedPathsWidget": "external_runtime_paths",
    "_environmentImportConfigButton": "import_dependency_config",
    "_environmentExportConfigButton": "export_dependency_config",
}

TRANSLATIONS = {
    language: {**TRANSLATIONS[language], **WORKFLOW_TEXT.get(language, {})} for language in TRANSLATIONS
}
MAIN_UI_TEXT_BINDINGS = {**MAIN_UI_TEXT_BINDINGS, **WORKFLOW_TEXT_BINDINGS}
MAIN_UI_TOOLTIP_BINDINGS = {**MAIN_UI_TOOLTIP_BINDINGS, **WORKFLOW_TOOLTIP_BINDINGS}
MAIN_UI_TITLE_BINDINGS = {"topupCollapsibleGroupBox": "topup_group"}


def normalized_language(value: str | None) -> str:
    """Return a supported UI language code."""

    return "zh" if str(value or "").lower().startswith("zh") else "en"


def onboarding_text(language: str, key: str) -> str:
    """Return translated onboarding text with English fallback."""

    lang = normalized_language(language)
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, TRANSLATIONS["en"].get(key, key))


def environment_section_text(language: str, section: str) -> tuple[str, str]:
    """Return the localized title and description for an environment section."""

    lang = normalized_language(language)
    fallback = ENVIRONMENT_SECTION_TEXT["en"].get(section, (section, ""))
    return ENVIRONMENT_SECTION_TEXT.get(lang, ENVIRONMENT_SECTION_TEXT["en"]).get(section, fallback)


def dependency_ui_text(language: str, dependency_id: str, field: str) -> str:
    """Return a localized dependency-path row label or tooltip."""

    lang = normalized_language(language)
    fallback = ENVIRONMENT_DEPENDENCY_TEXT["en"].get(dependency_id, (dependency_id, ""))
    values = ENVIRONMENT_DEPENDENCY_TEXT.get(lang, ENVIRONMENT_DEPENDENCY_TEXT["en"]).get(
        dependency_id,
        fallback,
    )
    return values[0 if field == "label" else 1]


UI_MESSAGE_TEMPLATES = (
    (
        r"Partial results: (?P<count>\d+) blocked task\(s\)",
        r"部分结果：(?P<count>\d+) 个任务被阻塞",
        "Partial results: {count} blocked task(s)",
        "部分结果：{count} 个任务被阻塞",
    ),
    (
        r"Pipeline failed: exit code (?P<code>-?\d+)",
        r"Pipeline 失败：退出码 (?P<code>-?\d+)",
        "Pipeline failed: exit code {code}",
        "Pipeline 失败：退出码 {code}",
    ),
    (
        r"Completed with (?P<count>\d+) failed task\(s\)",
        r"完成，但有 (?P<count>\d+) 个任务失败",
        "Completed with {count} failed task(s)",
        "完成，但有 {count} 个任务失败",
    ),
    (
        r"Completed with (?P<count>\d+) blocked task\(s\)",
        r"完成，但有 (?P<count>\d+) 个任务被阻塞",
        "Completed with {count} blocked task(s)",
        "完成，但有 {count} 个任务被阻塞",
    ),
    (
        r"Batch failed: (?P<count>\d+) subject\(s\)",
        r"批处理失败：(?P<count>\d+) 个受试者失败",
        "Batch failed: {count} subject(s)",
        "批处理失败：{count} 个受试者失败",
    ),
    (
        r"Environment: (?P<count>\d+) missing required item\(s\)",
        r"环境：缺少 (?P<count>\d+) 个必需项",
        "Environment: {count} missing required item(s)",
        "环境：缺少 {count} 个必需项",
    ),
    (
        r"Pipeline cancelled\. Manifest: (?P<path>.+)",
        r"Pipeline 已取消。清单：(?P<path>.+)",
        "Pipeline cancelled. Manifest: {path}",
        "Pipeline 已取消。清单：{path}",
    ),
    (
        r"Run finished\. Manifest: (?P<path>.+)",
        r"运行完成。清单：(?P<path>.+)",
        "Run finished. Manifest: {path}",
        "运行完成。清单：{path}",
    ),
    (
        r"Batch finished\. Manifest: (?P<path>.+)",
        r"批处理完成。清单：(?P<path>.+)",
        "Batch finished. Manifest: {path}",
        "批处理完成。清单：{path}",
    ),
    (
        r"Run finished\. Loading up to (?P<count>\d+) key result node\(s\)\.\.\.",
        r"运行完成。正在加载最多 (?P<count>\d+) 个关键结果节点…",
        "Run finished. Loading up to {count} key result node(s)...",
        "运行完成。正在加载最多 {count} 个关键结果节点…",
    ),
    (
        r"Batch execution submitted for (?P<path>.+)\.",
        r"已为 (?P<path>.+) 提交批处理任务。",
        "Batch execution submitted for {path}.",
        "已为 {path} 提交批处理任务。",
    ),
    (
        r"Status: failed \| Background process exited with code (?P<code>-?\d+)\.",
        r"状态：失败 \| 后台进程退出码为 (?P<code>-?\d+)。",
        "Status: failed | Background process exited with code {code}.",
        "状态：失败 | 后台进程退出码为 {code}。",
    ),
    (
        r"detected from (?P<source>.+)",
        r"检测来源：(?P<source>.+)",
        "detected from {source}",
        "检测来源：{source}",
    ),
    (
        r"Pipeline failed: (?P<detail>.+)",
        r"Pipeline 失败：(?P<detail>.+)",
        "Pipeline failed: {detail}",
        "Pipeline 失败：{detail}",
    ),
)


def translated_ui_message(language: str, message: str) -> str:
    """Translate a known UI status message while preserving unknown runtime details."""

    source = str(message or "")
    pending = re.fullmatch(r"(?:Waiting: |待填写：)(?P<fields>.+)", source)
    if pending:
        fields = pending.group("fields")
        names = {"Subject ID": "受试者 ID", "Subject directory": "受试者目录",
                 "Subject input layout": "输入文件结构", "Output directory": "输出目录",
                 "Batch root": "批处理根目录", "Target products": "分析任务"}
        chinese = normalized_language(language) == "zh"
        for english, translated in names.items():
            fields = fields.replace(english if chinese else translated, translated if chinese else english)
        return ("待填写：" if chinese else "Waiting: ") + fields
    for key, english in TRANSLATIONS["en"].items():
        if not isinstance(english, str):
            continue
        if source in {english, TRANSLATIONS["zh"].get(key)}:
            return onboarding_text(language, key)
    for english_pattern, chinese_pattern, english_template, chinese_template in UI_MESSAGE_TEMPLATES:
        match = re.fullmatch(english_pattern, source) or re.fullmatch(chinese_pattern, source)
        if match:
            template = chinese_template if normalized_language(language) == "zh" else english_template
            return template.format(**match.groupdict())
    return source


def register_ui_language_binding(owner: Any, widget: Any, property_name: str, translation_key: str) -> None:
    """Register one dynamically created widget property for language switching."""

    if widget is None:
        return
    bindings = getattr(owner, "_uiLanguageBindings", None)
    if bindings is None:
        bindings = []
        owner._uiLanguageBindings = bindings
    bindings.append((widget, property_name, translation_key))


def _set_widget_text(widget: Any, text: str) -> None:
    """Set text on Qt/PythonQt widgets without depending on one binding style."""

    if widget is None:
        return
    try:
        widget.setText(text)
    except Exception:
        pass
    try:
        widget.text = text
    except Exception:
        pass


def _set_widget_tooltip(widget: Any, text: str) -> None:
    """Set a translated tooltip on Qt/PythonQt widgets."""

    if widget is None:
        return
    try:
        widget.setToolTip(text)
    except Exception:
        pass
    try:
        widget.toolTip = text
    except Exception:
        pass


def _set_path_edit_placeholder(widget: Any, text: str) -> None:
    """Show placeholder text in a ctkPathLineEdit, whose editable combo box holds the real QLineEdit."""

    if widget is None:
        return
    try:
        import qt

        edits = widget.findChildren(qt.QLineEdit)
    except Exception:
        edits = []
    for edit in edits[:1]:
        _set_widget_placeholder(edit, text)
        return
    _set_widget_placeholder(widget, text)


def _set_widget_placeholder(widget: Any, text: str) -> None:
    """Set translated placeholder text on Qt/PythonQt input widgets."""

    if widget is None:
        return
    try:
        widget.setPlaceholderText(text)
    except Exception:
        pass
    try:
        widget.placeholderText = text
    except Exception:
        pass


def _widget_text(widget: Any) -> str:
    if widget is None:
        return ""
    try:
        value = widget.text()
    except Exception:
        value = getattr(widget, "text", "")
    return str(value or "")


def platform_path_examples(language: str, *, system: str | None = None) -> str:
    """Return path examples that match the user's operating system."""

    system_name = (system or platform.system()).lower()
    if system_name.startswith("win"):
        subject = r"C:\Users\name\data\sub-01"
        output = r"C:\Users\name\data\sub-01\outputs"
        models = r"C:\Users\name\SlicerDSpace-model-assets"
        mrtrix = r"C:\Tools\MRtrix3\bin"
    elif system_name == "darwin":
        subject = "/Users/name/data/sub-01"
        output = "/Users/name/data/sub-01/outputs"
        models = "/Users/name/Library/Application Support/SlicerDSpace/model-assets"
        mrtrix = "/opt/homebrew/bin"
    else:
        subject = "/home/name/data/sub-01"
        output = "/home/name/data/sub-01/outputs"
        models = "/home/name/.local/share/SlicerDSpace/model-assets"
        mrtrix = "/opt/mrtrix3/bin"
    if normalized_language(language) == "zh":
        return (
            f"本机路径示例：Subject root {subject} | 输出目录 {output} | "
            f"模型权重 {models} | MRtrix bin {mrtrix}"
        )
    return (
        f"Examples for this computer: Subject root {subject} | Output {output} | "
        f"Model assets {models} | MRtrix bin {mrtrix}"
    )


def capability_dashboard_text(result: Any, *, language: str = "en") -> str:
    """Summarize environment capability for non-technical users."""

    if result is None:
        return onboarding_text(language, "dashboard_waiting")
    runnable = tuple(getattr(result, "runnable_targets", ()) or ())
    blocked = tuple(getattr(result, "blocked_targets", ()) or ())
    statuses = tuple(getattr(result, "statuses", ()) or ())
    missing_by_target: dict[str, list[str]] = {}
    for item in statuses:
        if getattr(item, "status", "") != "missing":
            continue
        label = str(getattr(item, "label", "") or getattr(item, "key", "") or "dependency")
        for target in tuple(getattr(item, "targets", ()) or ()):
            missing_by_target.setdefault(str(target), []).append(label)
    if normalized_language(language) == "zh":
        lines = [
            "你现在可以运行：" + (", ".join(runnable) if runnable else "暂无，需要先补齐输入或依赖"),
        ]
        if blocked:
            lines.append("你还不能运行：")
            for target in blocked:
                reasons = ", ".join(dict.fromkeys(missing_by_target.get(target, [])) or ["缺少依赖或输入"])
                lines.append(f"- {target}: {reasons}")
        else:
            lines.append("没有检测到被阻塞的已选目标。")
        return "\n".join(lines)
    lines = [
        "Runnable now: " + (", ".join(runnable) if runnable else "none yet; add inputs or dependencies first"),
    ]
    if blocked:
        lines.append("Needs setup:")
        for target in blocked:
            reasons = ", ".join(dict.fromkeys(missing_by_target.get(target, [])) or ["missing dependency or input"])
            lines.append(f"- {target}: {reasons}")
    else:
        lines.append("No selected targets are blocked.")
    return "\n".join(lines)


def diagnostic_report_text(owner: Any) -> str:
    """Build a compact report users can paste into an issue or support message."""

    node = getattr(owner, "_parameterNode", None)
    settings_payload = {}
    for key in (
        "model_assets_dir",
        "mrtrix_bin_dir",
        "preprocessing_fsl_bin_dir",
        "postprocess_freesurfer_home",
        "pnl_mask_python",
        "pnl_mask_script",
        "pnl_mask_model_dir",
        "pnl_mask_ants_bin_dir",
        "tract_querier_root",
        "wmql_python_executable",
        "wma_root",
        "tractcloud_root",
    ):
        try:
            settings_payload[key] = owner._read_environment_path_setting(key)
        except Exception:
            settings_payload[key] = ""
    result = getattr(owner, "_lastEnvironmentResult", None)
    missing = []
    if result is not None:
        for item in tuple(getattr(result, "statuses", ()) or ()):
            if getattr(item, "status", "") == "missing":
                missing.append(
                    {
                        "key": getattr(item, "key", ""),
                        "label": getattr(item, "label", ""),
                        "targets": list(getattr(item, "targets", ()) or ()),
                    }
                )
    log_tail = ""
    log_widget = getattr(getattr(owner, "ui", None), "logPlainTextEdit", None)
    if log_widget is not None:
        try:
            log_tail = "\n".join(str(log_widget.toPlainText()).splitlines()[-80:])
        except Exception:
            log_tail = ""
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "platform": platform.platform(),
        "python": sys.executable,
        "user_config_path": default_user_config_path().expanduser().as_posix(),
        "slicer_version": getattr(getattr(sys.modules.get("slicer"), "app", None), "applicationVersion", "unknown"),
        "subject_id": getattr(node, "subjectId", "") if node else "",
        "subject_root": getattr(node, "inputDirectory", "") if node else "",
        "output_directory": getattr(node, "outputDirectory", "") if node else "",
        "external_paths": settings_payload,
        "runnable_targets": list(getattr(result, "runnable_targets", ()) or ()) if result is not None else [],
        "blocked_targets": list(getattr(result, "blocked_targets", ()) or ()) if result is not None else [],
        "missing_dependencies": missing,
        "log_tail": log_tail,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)


def install_onboarding_panel(owner: Any, parent_layout: Any) -> None:
    """Install beginner-first controls into the existing Slicer module UI."""

    import qt

    panel = qt.QFrame()
    panel.setObjectName("onboardingPanel")
    # The "Getting started" collapsible already frames this panel.
    panel.setStyleSheet("QFrame#onboardingPanel { border: none; }")
    panel.setContentsMargins(0, 0, 0, 0)
    layout = qt.QVBoxLayout(panel)

    from shared.ui_styles import is_dark_mode

    dark = is_dark_mode(getattr(owner, "_uiWidget", None))
    top = qt.QHBoxLayout()
    title = qt.QLabel()
    title.setObjectName("onboardingTitleLabel")
    title.setStyleSheet(
        "QLabel { color: #9BE0B8; font-size: 15px; font-weight: 800; }"
        if dark
        else "QLabel { color: #173D2A; font-size: 15px; font-weight: 800; }"
    )
    top.addWidget(title, 1)
    language_label = qt.QLabel()
    language_label.setObjectName("onboardingLanguageLabel")
    language_combo = qt.QComboBox()
    language_combo.addItem("English", "en")
    language_combo.addItem("中文", "zh")
    top.addWidget(language_label)
    top.addWidget(language_combo)
    layout.addLayout(top)

    body = qt.QLabel()
    body.setObjectName("onboardingBodyLabel")
    body.wordWrap = True
    body.setStyleSheet("QLabel { color: #B8C5D0; }" if dark else "QLabel { color: #426154; }")
    layout.addWidget(body)

    # 2 x 2 grid so the panel stays usable in a narrow module panel.
    buttons = qt.QGridLayout()
    wizard_button = qt.QPushButton()
    demo_button = qt.QPushButton()
    diagnostic_button = qt.QPushButton()
    config_button = qt.QPushButton()
    buttons.addWidget(wizard_button, 0, 0)
    buttons.addWidget(demo_button, 0, 1)
    buttons.addWidget(diagnostic_button, 1, 0)
    buttons.addWidget(config_button, 1, 1)
    layout.addLayout(buttons)

    dashboard = qt.QLabel()
    dashboard.setObjectName("capabilityDashboardLabel")
    dashboard.wordWrap = True
    dashboard.setStyleSheet(
        "QLabel { background: #1E2A24; border: 1px solid #354C40; border-radius: 8px; "
        "color: #C2E0D0; padding: 8px; }"
        if dark
        else "QLabel { background: #FFFFFF; border: 1px solid #DCE8E0; border-radius: 8px; "
        "color: #244034; padding: 8px; }"
    )
    layout.addWidget(dashboard)

    examples = qt.QLabel()
    examples.setObjectName("pathExamplesLabel")
    examples.wordWrap = True
    examples.setStyleSheet(
        "QLabel { color: #9BA7B4; font-size: 11px; }"
        if dark
        else "QLabel { color: #6B7D74; font-size: 11px; }"
    )
    layout.addWidget(examples)

    owner._onboardingWidgets = {
        "panel": panel,
        "title": title,
        "language_label": language_label,
        "language_combo": language_combo,
        "body": body,
        "wizard_button": wizard_button,
        "demo_button": demo_button,
        "diagnostic_button": diagnostic_button,
        "config_button": config_button,
        "dashboard": dashboard,
        "examples": examples,
    }

    language_combo.currentIndexChanged.connect(owner._on_onboarding_language_changed)
    wizard_button.connect("clicked()", owner._on_show_first_run_wizard_clicked)
    demo_button.connect("clicked()", owner._on_try_demo_clicked)
    diagnostic_button.connect("clicked()", owner._on_copy_diagnostic_report_clicked)
    config_button.connect("clicked()", owner._on_open_user_config_clicked)

    try:
        parent_layout.insertWidget(0, panel)
    except Exception:
        parent_layout.addWidget(panel)
    refresh_onboarding_styles(owner, dark=dark)


def refresh_onboarding_styles(owner: Any, *, dark: bool | None = None) -> None:
    """Refresh inline stylesheets on onboarding widgets for light and dark themes."""

    widgets = getattr(owner, "_onboardingWidgets", None) or {}
    if not widgets:
        return
    from shared.ui_styles import is_dark_mode

    use_dark = is_dark_mode(getattr(owner, "_uiWidget", None)) if dark is None else bool(dark)
    title = widgets.get("title")
    if title is not None:
        title.setStyleSheet(
            "QLabel { color: #9BE0B8; font-size: 15px; font-weight: 800; }"
            if use_dark
            else "QLabel { color: #173D2A; font-size: 15px; font-weight: 800; }"
        )
    body = widgets.get("body")
    if body is not None:
        body.setStyleSheet("QLabel { color: #B8C5D0; }" if use_dark else "QLabel { color: #426154; }")
    dashboard = widgets.get("dashboard")
    if dashboard is not None:
        dashboard.setStyleSheet(
            "QLabel { background: #1E2A24; border: 1px solid #354C40; border-radius: 8px; "
            "color: #C2E0D0; padding: 8px; }"
            if use_dark
            else "QLabel { background: #FFFFFF; border: 1px solid #DCE8E0; border-radius: 8px; "
            "color: #244034; padding: 8px; }"
        )
    examples = widgets.get("examples")
    if examples is not None:
        examples.setStyleSheet(
            "QLabel { color: #9BA7B4; font-size: 11px; }"
            if use_dark
            else "QLabel { color: #6B7D74; font-size: 11px; }"
        )


def apply_onboarding_language(owner: Any, language: str) -> None:
    """Apply translated labels to onboarding, workflow, and run-status controls."""

    lang = normalized_language(language)
    owner._onboardingLanguage = lang
    scene_labels = (("加载", "替换", "追加", "清除") if lang == "zh" else
                    ("Load", "Replace", "Append", "Clear"))
    for button, text in zip(getattr(owner, "_resultSceneButtons", []), scene_labels):
        _set_widget_text(button, text)
    widgets = getattr(owner, "_onboardingWidgets", {}) or {}
    for name, key in (
        ("title", "beginner_title"),
        ("language_label", "language"),
        ("body", "beginner_body"),
        ("wizard_button", "wizard"),
        ("demo_button", "demo"),
        ("diagnostic_button", "copy_diag"),
        ("config_button", "open_config"),
    ):
        widget = widgets.get(name)
        if widget is not None:
            _set_widget_text(widget, onboarding_text(lang, key))
    examples = widgets.get("examples")
    if examples is not None:
        _set_widget_text(examples, platform_path_examples(lang))
    combo = widgets.get("language_combo")
    if combo is not None:
        index = combo.findData(lang)
        if index >= 0 and combo.currentIndex != index:
            combo.blockSignals(True)
            combo.setCurrentIndex(index)
            combo.blockSignals(False)
    ui = getattr(owner, "ui", None)
    for widget_name, key in MAIN_UI_TEXT_BINDINGS.items():
        widget = getattr(ui, widget_name, None)
        if widget is not None:
            _set_widget_text(widget, onboarding_text(lang, key))

    for widget_name, key in MAIN_UI_TOOLTIP_BINDINGS.items():
        _set_widget_tooltip(getattr(ui, widget_name, None), onboarding_text(lang, key))

    _set_path_edit_placeholder(getattr(ui, "inputDirectoryPath", None), onboarding_text(lang, "subject_root_placeholder"))
    _set_path_edit_placeholder(getattr(ui, "outputDirectoryPath", None), onboarding_text(lang, "output_directory_placeholder"))
    _set_path_edit_placeholder(getattr(ui, "wmaInputTractographyPath", None), onboarding_text(lang, "wma_input_placeholder"))
    if hasattr(owner, "_refresh_subject_id_placeholder"):
        try:
            owner._refresh_subject_id_placeholder()
        except Exception:
            pass
    root_widget = getattr(owner, "_uiWidget", None)
    if root_widget is not None:
        try:
            from shared.ui_styles import apply_placeholder_palette

            apply_placeholder_palette(root_widget)
        except Exception:
            pass

    mask_combo = getattr(ui, "brainMaskBackendComboBox", None)
    if mask_combo is not None:
        previous = mask_combo.blockSignals(True)
        try:
            for backend, key in BRAIN_MASK_BACKEND_TEXT_KEYS.items():
                index = mask_combo.findData(backend)
                if index >= 0:
                    mask_combo.setItemText(index, onboarding_text(lang, key))
        finally:
            mask_combo.blockSignals(previous)

    route_combo = getattr(ui, "pnlMaskRouteComboBox", None)
    if route_combo is not None:
        previous = route_combo.blockSignals(True)
        try:
            for route, key in PNL_MASK_ROUTE_TEXT_KEYS.items():
                index = route_combo.findData(route)
                if index >= 0:
                    route_combo.setItemText(index, onboarding_text(lang, key))
        finally:
            route_combo.blockSignals(previous)

    for widget_name, key in MAIN_UI_TITLE_BINDINGS.items():
        widget = getattr(ui, widget_name, None)
        if widget is not None:
            try:
                widget.setTitle(onboarding_text(lang, key))
            except Exception:
                pass

    for attribute, key in DYNAMIC_UI_TEXT_BINDINGS.items():
        _set_widget_text(getattr(owner, attribute, None), onboarding_text(lang, key))

    for widget, property_name, key in tuple(getattr(owner, "_uiLanguageBindings", ()) or ()):
        value = onboarding_text(lang, key)
        if property_name == "tooltip":
            _set_widget_tooltip(widget, value)
        elif property_name == "placeholder":
            _set_widget_placeholder(widget, value)
        else:
            _set_widget_text(widget, value)

    for section, title_widget, description_widget in tuple(
        getattr(owner, "_environmentSectionLanguageWidgets", ()) or ()
    ):
        title, description = environment_section_text(lang, section)
        _set_widget_text(title_widget, title)
        _set_widget_text(description_widget, description)

    dependency_rows = getattr(owner, "_environmentPathLanguageWidgets", {}) or {}
    for dependency_id, row_widgets in dependency_rows.items():
        label = dependency_ui_text(lang, dependency_id, "label")
        tooltip = dependency_ui_text(lang, dependency_id, "tooltip")
        _set_widget_text(row_widgets.get("label"), label)
        _set_widget_placeholder(row_widgets.get("line_edit"), onboarding_text(lang, "env_auto_detect"))
        _set_widget_tooltip(row_widgets.get("line_edit"), tooltip)
        _set_widget_text(row_widgets.get("browse"), onboarding_text(lang, "browse"))
        _set_widget_tooltip(row_widgets.get("browse"), tooltip)
        _set_widget_text(row_widgets.get("clear"), onboarding_text(lang, "clear"))

    status_sources = getattr(owner, "_environmentPathStatusSources", {}) or {}
    for dependency_id, label in (getattr(owner, "_environmentPathStatusLabels", {}) or {}).items():
        source = str(status_sources.get(dependency_id) or "not checked")
        _set_widget_text(label, translated_ui_message(lang, source))

    prepare_widgets = getattr(owner, "_prepareInputWidgets", {}) or {}
    for name, key in (
        ("panel", "prepare_inputs"),
        ("preview_button", "preview_plan"),
        ("prepare_button", "prepare_inputs_action"),
        ("manifest_button", "open_manifest"),
        ("dry_run", "dry_run"),
        ("force", "overwrite_files"),
    ):
        _set_widget_text(prepare_widgets.get(name), onboarding_text(lang, key))

    status = getattr(ui, "statusLabel", None)
    status_source = getattr(owner, "_statusMessageSource", None)
    if status_source is not None:
        _set_widget_text(status, translated_ui_message(lang, status_source))
    elif status is not None:
        translated = translated_ui_message(lang, _widget_text(status))
        if translated != _widget_text(status):
            _set_widget_text(status, translated)

    environment_result = getattr(ui, "environmentResultLabel", None)
    if environment_result is not None:
        current = _widget_text(environment_result)
        known_defaults = {
            onboarding_text("en", "environment_not_checked"),
            onboarding_text("zh", "environment_not_checked"),
        }
        if not current or current in known_defaults:
            _set_widget_text(environment_result, onboarding_text(lang, "environment_not_checked"))

    run_hint = getattr(owner, "_runStatusHintLabel", None)
    run_hint_source = getattr(owner, "_runProgressMessageSource", None)
    if run_hint_source is not None:
        _set_widget_text(run_hint, translated_ui_message(lang, run_hint_source))
    elif run_hint is not None and _widget_text(run_hint) in {
        onboarding_text("en", "run_plan_pending"),
        onboarding_text("zh", "run_plan_pending"),
    }:
        _set_widget_text(run_hint, onboarding_text(lang, "run_plan_pending"))

    run_summary = getattr(owner, "_runSummaryLabel", None)
    run_summary_source = getattr(owner, "_runSummaryMessageSource", None)
    if run_summary_source is not None:
        _set_widget_text(run_summary, translated_ui_message(lang, run_summary_source))
    elif run_summary is not None and _widget_text(run_summary) in {
        onboarding_text("en", "run_summary_pending"),
        onboarding_text("zh", "run_summary_pending"),
    }:
        _set_widget_text(run_summary, onboarding_text(lang, "run_summary_pending"))

    table = getattr(owner, "_runStatusTable", None)
    if table is not None:
        table.setHorizontalHeaderLabels(
            [
                onboarding_text(lang, "target"),
                onboarding_text(lang, "state"),
                onboarding_text(lang, "reason"),
                onboarding_text(lang, "manifest"),
                onboarding_text(lang, "qc"),
                onboarding_text(lang, "logs"),
            ]
        )
    dashboard = widgets.get("dashboard")
    if dashboard is not None:
        _set_widget_text(
            dashboard,
            capability_dashboard_text(getattr(owner, "_lastEnvironmentResult", None), language=lang),
        )
    from shared.ui_compact import apply_compact_language

    apply_compact_language(owner, lang)
    from shared.ui_regions import apply_region_language
    apply_region_language(owner, lang)


def maybe_show_first_run_wizard(owner: Any) -> None:
    """Show the first-run wizard once per user settings profile."""

    import qt
    import slicer

    if os.environ.get("SLICERDSPACE_DISABLE_FIRST_RUN_WIZARD") == "1":
        return
    settings = slicer.app.userSettings()
    try:
        config = load_user_config()
        shown_in_json = bool(config.get("ui", {}).get("first_run_wizard_shown"))
    except Exception:
        shown_in_json = False
    if shown_in_json or str(settings.value(FIRST_RUN_SETTING, "") or "") == "1":
        return
    settings.setValue(FIRST_RUN_SETTING, "1")
    try:
        update_user_config({"ui": {"first_run_wizard_shown": True}})
    except Exception:
        pass
    qt.QTimer.singleShot(1200, owner._on_show_first_run_wizard_clicked)


def show_first_run_wizard(owner: Any) -> None:
    """Show a stateful setup checklist for the beginner path."""

    import qt
    import slicer

    lang = getattr(owner, "_onboardingLanguage", "en")
    dialog = qt.QDialog(slicer.util.mainWindow())
    dialog.setObjectName("firstRunSetupWizardDialog")
    dialog.setWindowTitle(onboarding_text(lang, "wizard"))
    dialog.setMinimumWidth(760)
    layout = qt.QVBoxLayout(dialog)
    is_zh = normalized_language(lang) == "zh"
    copy = {
        "intro": (
            "按顺序执行并查看每一步的状态。外部软件只在相应分析中需要，扫描后即使有可选项缺失，也可以先运行可用功能。"
            if is_zh
            else "Work through the steps and review each status. External tools are needed only for their related analyses; you can run available features even when optional items are missing."
        ),
        "progress": "设置进度" if is_zh else "Setup progress",
        "pending": "尚未执行" if is_zh else "Not run yet",
        "working": "正在执行..." if is_zh else "Working...",
        "ready": "已完成" if is_zh else "Ready",
        "failed": "需要处理" if is_zh else "Needs attention",
        "python_title": "1. 确认 Slicer Python" if is_zh else "1. Confirm Slicer Python",
        "python_body": "Python 包会安装到当前 Slicer 自带的 Python，不会改动系统 Python。" if is_zh else "Python packages are installed into this Slicer Python, not the system Python.",
        "python_action": "显示 Python 位置" if is_zh else "Show Python location",
        "root_title": "2. 选择模型权重目录" if is_zh else "2. Choose model asset folder",
        "root_body": "权重属于用户数据，应放在插件目录之外；选择一次后会自动记住。" if is_zh else "Weights are user data and should live outside the extension folder. This choice is remembered.",
        "root_action": "选择目录" if is_zh else "Choose folder",
        "models_title": "3. 下载并校验核心模型" if is_zh else "3. Download and verify core models",
        "models_body": "显示当前文件和进度；取消后保留部分文件，下次可继续下载。" if is_zh else "Shows the current file and progress. Cancelling keeps partial files so the next attempt can resume.",
        "models_action": "下载 / 继续下载" if is_zh else "Download / resume",
        "scan_title": "4. 扫描本机依赖" if is_zh else "4. Scan this computer",
        "scan_body": "自动搜索常见的 Windows、macOS 或 Linux 路径，并说明哪些功能已经可运行。" if is_zh else "Searches common Windows, macOS or Linux locations and reports which features can run now.",
        "scan_action": "开始扫描" if is_zh else "Scan now",
        "report_title": "5. 查看可运行能力报告" if is_zh else "5. Review capability report",
        "report_body": "报告会区分现在可运行的功能与仍需配置的可选功能。" if is_zh else "The report separates features that run now from optional features that still need setup.",
        "report_action": "显示报告" if is_zh else "Show report",
        "demo": "直接运行轻量示例" if is_zh else "Run lightweight demo",
        "guide": "打开本系统安装指南" if is_zh else "Open setup guide for this OS",
        "close": "关闭" if is_zh else "Close",
        "cancelled": "未选择目录；原设置没有改变。" if is_zh else "No folder selected; existing settings were unchanged.",
        "download_failed": "下载未完成。可以再次点击以继续。" if is_zh else "Download is incomplete. Click again to resume.",
        "scan_failed": "扫描没有完成；请复制诊断报告后重试。" if is_zh else "The scan did not complete. Copy the diagnostic report and retry.",
        "report_missing": "请先执行依赖扫描。" if is_zh else "Run the dependency scan first.",
    }
    from shared.ui_styles import is_dark_mode, ui_stylesheet

    dark = is_dark_mode(getattr(owner, "_uiWidget", None) or dialog)
    dialog.setStyleSheet(ui_stylesheet(dark=dark, widget=dialog))
    intro = qt.QLabel(copy["intro"])
    intro.wordWrap = True
    intro.setStyleSheet(
        "QLabel { color: #B8C5D0; padding: 4px 2px 8px 2px; }"
        if dark
        else "QLabel { color: #3E5B4A; padding: 4px 2px 8px 2px; }"
    )
    layout.addWidget(intro)

    progress_label = qt.QLabel(f"{copy['progress']}: 0 / 5")
    progress_label.setObjectName("firstRunWizardProgressLabel")
    progress_label.setStyleSheet(
        "QLabel { color: #9BE0B8; font-weight: 700; padding-bottom: 4px; }"
        if dark
        else "QLabel { color: #173D2A; font-weight: 700; padding-bottom: 4px; }"
    )
    layout.addWidget(progress_label)

    scroll = qt.QScrollArea()
    scroll.setObjectName("firstRunWizardScrollArea")
    scroll.setWidgetResizable(True)
    scroll.setMinimumHeight(430)
    steps_container = qt.QWidget()
    steps_layout = qt.QVBoxLayout(steps_container)
    steps_layout.setContentsMargins(2, 2, 2, 2)
    scroll.setWidget(steps_container)
    layout.addWidget(scroll, 1)

    def python_action() -> tuple[bool, str]:
        target = str(owner._python_install_target_text())
        slicer.util.infoDisplay(target, windowTitle=copy["python_title"], parent=dialog)
        return True, target

    def root_action() -> tuple[bool, str]:
        root = owner._choose_model_assets_root()
        return (root is not None), (str(root) if root is not None else copy["cancelled"])

    def models_action() -> tuple[bool | None, str]:
        def completed(ok, detail):
            try:
                set_step_status(2, "ready" if ok else "failed", detail)
            except RuntimeError:
                pass  # The wizard may have been closed while setup ran.
        started = owner._install_model_assets_from_environment_dialog(
            profile="core", prompt_for_root=False, on_complete=completed)
        return (None, "Downloading and verifying in background...") if started else (False, copy["download_failed"])

    def scan_action() -> tuple[bool, str]:
        result = owner._onValidateClicked()
        if result is None:
            return False, copy["scan_failed"]
        runnable = len(tuple(getattr(result, "runnable_targets", ()) or ()))
        blocked = len(tuple(getattr(result, "blocked_targets", ()) or ()))
        detail = f"{runnable} runnable; {blocked} need setup"
        if is_zh:
            detail = f"{runnable} 项可运行；{blocked} 项仍需配置"
        return True, detail

    def report_action() -> tuple[bool, str]:
        result = getattr(owner, "_lastEnvironmentResult", None)
        if result is None:
            return False, copy["report_missing"]
        report = capability_dashboard_text(result, language=lang)
        owner._update_onboarding_dashboard(result)
        slicer.util.infoDisplay(report, windowTitle=copy["report_title"], parent=dialog)
        return True, ("报告已显示。" if is_zh else "Capability report displayed.")

    steps = [
        (copy["python_title"], copy["python_body"], copy["python_action"], python_action),
        (copy["root_title"], copy["root_body"], copy["root_action"], root_action),
        (copy["models_title"], copy["models_body"], copy["models_action"], models_action),
        (copy["scan_title"], copy["scan_body"], copy["scan_action"], scan_action),
        (copy["report_title"], copy["report_body"], copy["report_action"], report_action),
    ]
    status_labels = []
    action_buttons = []
    step_frames = []
    completed_steps: set[int] = set()

    def set_step_status(index: int, state: str, detail: str) -> None:
        label = status_labels[index]
        prefix = copy["ready"] if state == "ready" else copy["failed"] if state == "failed" else copy["working"] if state == "working" else copy["pending"]
        label.setText(f"{prefix}: {detail}" if detail else prefix)
        colors = (
            {
                "ready": ("#1D3326", "#8CE99A"),
                "failed": ("#382C1B", "#FFD8A8"),
                "working": ("#1C2F42", "#8CD2F5"),
                "pending": ("#252B32", "#9BA7B4"),
            }
            if dark
            else {
                "ready": ("#EAF6EE", "#26643D"),
                "failed": ("#FFF3E8", "#9A4B13"),
                "working": ("#EAF2F8", "#285A78"),
                "pending": ("#F3F5F4", "#64736B"),
            }
        )
        background, foreground = colors.get(state, colors["pending"])
        label.setStyleSheet(
            f"QLabel {{ background: {background}; color: {foreground}; border-radius: 5px; padding: 4px 7px; }}"
        )
        if state == "ready":
            completed_steps.add(index)
        elif index in completed_steps:
            completed_steps.remove(index)
        progress_label.setText(f"{copy['progress']}: {len(completed_steps)} / {len(steps)}")

    def run_step(index: int, callback) -> None:
        set_step_status(index, "working", "")
        slicer.app.processEvents()
        try:
            ok, detail = callback()
        except Exception as exc:
            ok, detail = False, str(exc)
        set_step_status(index, "working" if ok is None else "ready" if ok else "failed", detail)

    for index, (title_text, body_text, action_text, callback) in enumerate(steps):
        frame = qt.QFrame()
        frame.setObjectName(f"firstRunWizardStep{index + 1}")
        frame.setStyleSheet(
            "QFrame { background: #22272E; border: 1px solid #3B434D; border-radius: 8px; }"
            if dark
            else "QFrame { background: #FFFFFF; border: 1px solid #DCE8E0; border-radius: 8px; }"
        )
        row = qt.QGridLayout(frame)
        title = qt.QLabel(f"<b>{title_text}</b>")
        body = qt.QLabel(body_text)
        body.wordWrap = True
        body.setStyleSheet("QLabel { color: #B8C5D0; border: none; }" if dark else "QLabel { color: #51665A; border: none; }")
        status = qt.QLabel(copy["pending"])
        status.setObjectName(f"firstRunWizardStatus{index + 1}")
        status.setWordWrap(True)
        button = qt.QPushButton(action_text)
        button.setObjectName(f"firstRunWizardAction{index + 1}")
        button.setMinimumWidth(150)
        button.connect("clicked()", lambda step=index, action=callback: run_step(step, action))
        row.addWidget(title, 0, 0)
        row.addWidget(button, 0, 1, 2, 1)
        row.addWidget(body, 1, 0)
        row.addWidget(status, 2, 0, 1, 2)
        steps_layout.addWidget(frame)
        step_frames.append(frame)
        action_buttons.append(button)
        status_labels.append(status)
        set_step_status(index, "pending", "")
    steps_layout.addStretch(1)

    set_step_status(0, "ready", str(owner._python_install_target_text()))
    configured_root = str(owner._read_environment_path_setting("model_assets_dir") or "").strip()
    if configured_root:
        set_step_status(1, "ready", configured_root)
    try:
        _, _, existing = owner._model_asset_verification("core", check_hashes=False)
        if existing.get("ok"):
            set_step_status(2, "ready", owner._format_model_asset_status_line("core", existing))
    except Exception:
        pass
    existing_result = getattr(owner, "_lastEnvironmentResult", None)
    if existing_result is not None:
        runnable = len(tuple(getattr(existing_result, "runnable_targets", ()) or ()))
        blocked = len(tuple(getattr(existing_result, "blocked_targets", ()) or ()))
        existing_detail = f"{runnable} runnable; {blocked} need setup"
        if is_zh:
            existing_detail = f"{runnable} 项可运行；{blocked} 项仍需配置"
        set_step_status(3, "ready", existing_detail)

    footer = qt.QHBoxLayout()
    guide_button = qt.QPushButton(copy["guide"])
    system_name = platform.system()
    guide_name = "WINDOWS.md" if system_name == "Windows" else "MACOS.md" if system_name == "Darwin" else "LINUX_HPC.md"
    guide_button.connect(
        "clicked()",
        lambda: owner._open_local_path(
            Path(__file__).resolve().parent.parent / "docs" / "setup" / guide_name,
            missing_message=f"docs/setup/{guide_name} is unavailable.",
        ),
    )
    demo_button = qt.QPushButton(copy["demo"])
    demo_button.connect("clicked()", lambda: (dialog.accept(), owner._on_try_demo_clicked()))
    close_button = qt.QPushButton(copy["close"])
    close_button.connect("clicked()", dialog.accept)
    footer.addWidget(guide_button)
    footer.addWidget(demo_button)
    footer.addStretch(1)
    footer.addWidget(close_button)
    layout.addLayout(footer)
    owner._firstRunWizardWidgets = {
        "dialog": dialog,
        "progress": progress_label,
        "scroll": scroll,
        "step_frames": tuple(step_frames),
        "status_labels": tuple(status_labels),
        "action_buttons": tuple(action_buttons),
    }
    dialog.exec_()


def prepare_demo_for_owner(owner: Any) -> Path:
    """Create a synthetic demo subject and configure the UI for light testing."""

    import slicer

    demo_root = Path(str(slicer.app.temporaryPath)) / "SlicerDSpace" / "demo"
    subject_root = create_synthetic_demo_subject(demo_root, subject_id=DEMO_SUBJECT_ID, overwrite=True)
    node = getattr(owner, "_parameterNode", None)
    if node:
        with slicer.util.NodeModify(node):
            node.subjectId = DEMO_SUBJECT_ID
            node.inputDirectory = subject_root.as_posix()
            node.outputDirectory = (subject_root / "outputs").as_posix()
            node.targetPreprocessing = False
            node.targetSurface = False
            node.targetSurfacePostprocess = False
            node.targetParcellation = False
            node.targetSegmentation = False
            node.targetMicrostructure = True
            node.targetTractography = True
            node.targetBundles = False
            node.targetConnectome = False
            node.orchestrationTractographyBackend = "dti_det"
            node.orchestrationMicrostructureStages = "dti"
            node.orchestrationRunName = "Demo"
            node.orchestrationSelectStreamlines = 512
        owner._checkCanApply()
    owner.appendLog(f"[Demo] Synthetic demo subject prepared: {subject_root}")
    return subject_root
