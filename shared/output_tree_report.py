"""Annotated HTML output-tree report generation."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .io_layout import ANATOMY_OUTPUT_NAME, LEGACY_ANATOMY_OUTPUT_NAME, anatomy_input_dir, normalize_run_name


@dataclass(frozen=True)
class OutputTreeReportPaths:
    report_html: Path
    report_html_latest: Optional[Path]
    regular_file_count: int
    symlink_count: int

    @property
    def total_nodes(self) -> int:
        return self.regular_file_count + self.symlink_count

    def to_dict(self) -> dict[str, object]:
        return {
            "report_html": self.report_html.as_posix(),
            "report_html_latest": self.report_html_latest.as_posix() if self.report_html_latest else None,
            "regular_file_count": self.regular_file_count,
            "symlink_count": self.symlink_count,
            "total_nodes": self.total_nodes,
        }


DIR_NOTES = {
    "": "统一 outputs 根目录",
    "00_run_metadata": "运行元数据、环境扫描、任务日志、run plan 与审计文件",
    "00_run_metadata/environment": "依赖 doctor 结果和自动安装/外部工具安装提示",
    "00_run_metadata/tasks": "按 run profile 和 task 归档的 manifest、qc、log",
    "00_run_metadata/validation": "统一输入契约校验结果",
    "01_anatomical_priors": "解剖先验：组织分割、脑区分割、皮层表面与表面形态学分析",
    "01_anatomical_priors/tissue_segmentation": "GM/WM/CSF 组织概率图与硬分割结果",
    "01_anatomical_priors/brain_parcellation": "受试者空间脑区分割图",
    "01_anatomical_priors/cortical_surface": "受试者原生空间皮层表面网格 (.obj)",
    "01_anatomical_priors/surface_morphometry": "FreeSurfer 风格表面形态学统计、厚度/曲率与注释",
    "02_microstructure": "微结构标量图与 ROI 汇总统计",
    "02_microstructure/diffusion_tensor": "DTI 扩散张量标量图 (FA/MD/AD/RD/DEC)",
    "02_microstructure/free_water": "Free-water 校正标量图与游离水分数",
    "02_microstructure/interstitial_free_water": "Interstitial free-water 结果与 ROI 统计",
    "02_microstructure/neurite_microstructure": "NODDI 神经突微结构标量图 (NDI/ODI/IsoVF)",
    "02_microstructure/diffusion_kurtosis": "DKI 扩散峰度标量图 (MK/AK/RK/KFA/MKT/RTK)",
    "03_tractography": "纤维束追踪主结果",
    "03_tractography/single_tensor_tractography": "单张量确定性 DTI 纤维束追踪",
    "03_tractography/multi_tensor_tractography": "多张量 UKF 纤维束追踪",
    "03_tractography/anatomically_constrained_tractography": "解剖约束 CSD-ACT 纤维束追踪",
    "03_tractography/short_association_fibers": "浅层联合纤维 (SAF) 追踪",
    "04_bundles": "白质纤维束提取、聚类与识别结果",
    "04_bundles/query_bundle_extraction": "WMQL 查询提取的纤维束 (.trk)",
    "04_bundles/fiber_clustering": "WMA 图谱聚类白质束与测量表 (.vtp/.csv)",
    "04_bundles/bundle_identification": "TractCloud 纤维束识别结果 (.vtp)",
    "05_connectome": "结构连接组矩阵、节点表与分区参考图",
    "06_qc": "各阶段各算法的简洁 TXT 质量控制摘要",
    "06_qc/01_anatomical_priors": "解剖先验阶段 QC TXT 摘要",
    "06_qc/02_microstructure": "微结构建模阶段 QC TXT 摘要",
    "06_qc/03_tractography": "纤维束追踪阶段 QC TXT 摘要",
    "06_qc/04_bundles": "纤维束分束阶段 QC TXT 摘要",
    "06_qc/05_connectome": "连接组构建阶段 QC TXT 摘要",
}


TASK_LABELS = {
    "bundles_tractcloud": "TractCloud bundle 分割",
    "bundles_wma": "WMA bundle 分割",
    "bundles_wmql": "WMQL bundle 查询",
    "connectome_sift2_weighted": "SIFT2 加权 connectome",
    "connectome_streamline_count": "streamline count connectome",
    "microstructure_dti": "DTI 微结构",
    "microstructure_freewater": "Free-water 微结构",
    "microstructure_noddi": "NODDI 微结构",
    "parcellation_ddparcel": "DDParcel 分区",
    "preprocessing_dwi": "DWI 预处理",
    "segmentation_ddseg": "DDSeg 分割",
    "surface_ddsurfer": "DDSurfer 表面重建",
    "surface_ddsurfer_postprocess": "DDSurfer 表面后处理",
    "tractography_csd_act": "CSD + ACT 追踪",
    "tractography_saf": "SAF 追踪/表面感知输出",
    "validation": "输入校验",
}


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _anatomy_description_path(path: str) -> str:
    if path == LEGACY_ANATOMY_OUTPUT_NAME or path.startswith(LEGACY_ANATOMY_OUTPUT_NAME + "/"):
        return ANATOMY_OUTPUT_NAME + path[len(LEGACY_ANATOMY_OUTPUT_NAME):]
    return path


def _describe_dir(path: str) -> str:
    path = _anatomy_description_path(path)
    if path in DIR_NOTES:
        return DIR_NOTES[path]
    if path.endswith("/logs"):
        return "日志目录"
    if "/regions_" in path:
        return "从 annotation 拆分出的 atlas region label 目录"
    if path.endswith("/100307-release-candidate"):
        return "受试者级输出子目录"
    return "输出子目录"


def _task_note(path: str, name: str) -> str:
    parts = path.split("/")
    profile = parts[2] if len(parts) > 2 else "profile"
    task = parts[3] if len(parts) > 3 else "task"
    label = TASK_LABELS.get(task, task.replace("_", " "))
    if name == "task.log":
        return f"{profile} / {label} 的终端运行日志"
    if name == "manifest.json":
        return f"{profile} / {label} 的任务 manifest，记录输入、输出、状态和参数"
    if name == "qc.json":
        return f"{profile} / {label} 的任务级 QC 摘要"
    return f"{profile} / {label} 的任务元数据"


def _describe_postprocess_file(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    if "/label/regions_" in path and name.endswith(".label"):
        atlas = path.split("/label/regions_", 1)[1].split("/", 1)[0]
        return f"DDSurfer post-process 从 {atlas} annotation 拆出的单 ROI label"
    if name.endswith(".annot"):
        return "DDSurfer post-process 投射到 subject surface 的 atlas annotation"
    if name.endswith(".ctab"):
        return "annotation 对应 color table"
    if name.endswith(".label"):
        return "FreeSurfer label 文件"
    if name.endswith(".stats"):
        return "按 atlas/半球汇总的 FreeSurfer 风格区域统计"
    if name in {"brain.mgz", "DDSurfer-wmparc.nii.gz"}:
        return "post-process FreeSurfer SUBJECTS_DIR 内的 MRI/staging 输入"
    if name.endswith(".fsaverage.mgh"):
        return "resample 到 fsaverage 的 morphometry overlay"
    if name in {"lh.white", "rh.white", "lh.pial", "rh.pial"}:
        return "FreeSurfer white/pial surface"
    if name.endswith(".inflated"):
        return "FreeSurfer inflated surface"
    if name.endswith(".sphere"):
        return "FreeSurfer spherical surface"
    if name.endswith(".sphere.reg"):
        return "FreeSurfer registered sphere surface"
    if name.endswith(".thickness"):
        return "cortical thickness overlay"
    if name.endswith(".area") or name.endswith(".area.pial"):
        return "per-vertex surface area overlay"
    if name.endswith(".curv") or name.endswith(".sulc"):
        return "FreeSurfer curvature/sulcal-depth overlay"
    if name.endswith(".H") or name.endswith(".K"):
        return "mean/Gaussian curvature diagnostic overlay"
    if name.endswith(".stl"):
        return "staged DDSurfer native STL 输入"
    return "DDSurfer post-process 输出文件"


def _describe_file(path: str, *, symlink: bool) -> str:
    path = _anatomy_description_path(path)
    name = path.rsplit("/", 1)[-1]
    prefix = "symlink：" if symlink else ""
    if "/tasks/" in path:
        return prefix + _task_note(path, name)
    if path.startswith("01_preproc/00_import/"):
        return prefix + "预处理导入阶段复用的原始输入文件"
    if path.endswith("_gradients_checked.bval") or path.endswith("_gradients_checked.bvec"):
        return prefix + "预处理阶段检查后的梯度文件"
    if path.endswith("_dwi_denoised.nii.gz"):
        return prefix + "DWI 去噪结果"
    if path.endswith("_dwi_degibbs.nii.gz"):
        return prefix + "DWI Gibbs ringing 修正结果"
    if path.endswith("_dwi_preproc.nii.gz"):
        return prefix + "DWI 畸变/运动校正后结果"
    if path.endswith("_dwi_final.nii.gz"):
        return prefix + "统一最终 DWI 输出"
    if path.endswith("_dwi_final.bval") or path.endswith("_dwi_final.bvec"):
        return prefix + "统一最终 DWI 对应梯度文件"
    if path.endswith("_brain_mask_final.nii.gz"):
        return prefix + "统一最终 brain mask"
    if name == "nodif_brain_mask.nii.gz":
        return prefix + "原生空间自动生成 brain mask"
    if name == "environment_doctor.json":
        return prefix + "依赖自动发现/环境 doctor JSON 结果"
    if name.startswith("install_missing_python"):
        return prefix + "缺失 Python 依赖安装 helper"
    if name == "install_external_tools.md":
        return prefix + "外部工具安装说明"
    if name.endswith("_orchestration_manifest.json"):
        return prefix + "run-level orchestration manifest"
    if name.endswith("_run_plan.json"):
        return prefix + "target run plan 和环境 blocker 记录"
    if name == "output_tree_audit.json":
        return prefix + "输出树结构审计"
    if name.endswith("_input_validation.json"):
        return prefix + "统一输入契约校验结果"
    if path.startswith("02_microstructure/diffusion_tensor/") or ("_dti_" in name and name.endswith(".nii.gz")):
        return prefix + "DTI scalar map"
    if path.startswith("02_microstructure/free_water/") or ("_fw_" in name and name.endswith(".nii.gz")):
        return prefix + "Free-water 微结构图"
    if path.startswith("02_microstructure/interstitial_free_water/"):
        return prefix + "Interstitial free-water 结果图或统计表"
    if path.startswith("02_microstructure/neurite_microstructure/") or ("_noddi_" in name and name.endswith(".nii.gz")):
        return prefix + "NODDI 微结构图"
    if path.startswith("02_microstructure/diffusion_kurtosis/"):
        return prefix + "DKI 扩散峰度微结构图"
    if name.endswith("_parcellation.nii.gz") or name.endswith("DDSurfer-wmparc.nii.gz"):
        return prefix + "受试者空间脑区分割标签图"
    if "ProbabilisticMap_" in name:
        return prefix + "组织概率图 (GM/WM/CSF)"
    if name == "SegmentationMap_GMWMCSF.nii.gz":
        return prefix + "GM/WM/CSF 硬分割标签图"
    if path.startswith("01_anatomical_priors/cortical_surface/") or "predicted_" in name:
        return prefix + "受试者原生空间皮层表面网格"
    if path.startswith("01_anatomical_priors/surface_morphometry/"):
        return prefix + _describe_postprocess_file(path)
    if path.startswith("03_tractography/"):
        return prefix + "纤维束追踪主结果文件"
    if path.startswith("04_bundles/"):
        return prefix + "解剖纤维束分割/聚类结果文件"
    if path.startswith("05_connectome/"):
        return prefix + "结构连接组矩阵、节点表或分区图"
    if path.startswith("06_qc/") and name.endswith(".txt"):
        return prefix + "算法质量控制 (QC) TXT 摘要"
    if name.endswith("_ROIWise-Microstructure.csv") or name.endswith("_ROIWise-Microstructure.json"):
        return prefix + "ROI 级微结构统计表"
    if name.endswith(".json"):
        return prefix + "JSON metadata / manifest / QC"
    if name.endswith((".nii.gz", ".mgz", ".mgh")):
        return prefix + "医学影像或 FreeSurfer morphometry 文件"
    if name.endswith((".tck", ".trk", ".vtk", ".vtp", ".stl", ".obj")):
        return prefix + "tractogram 或可视化网格文件"
    if name.endswith(".log"):
        return prefix + "运行日志"
    return prefix + "输出文件"


def _css_class(kind: str, path: str) -> str:
    path = _anatomy_description_path(path)
    if kind == "symlink":
        return "symlink"
    name = path.rsplit("/", 1)[-1]
    if "/cache/" in path or "_amico_noddi" in path:
        return "optional"
    if path.startswith("00_run_metadata/") or name.endswith((".json", ".log", ".md", ".txt", ".tsv")):
        return "metadata"
    if path.startswith("01_anatomical_priors/surface_morphometry/") or path.startswith("03_anatomy/ddsurfer/postprocess/"):
        return "generated"
    if name.endswith((".nii.gz", ".mgz", ".mgh", ".tck", ".trk", ".vtk", ".vtp", ".stl", ".obj", ".mif", ".csv", ".annot", ".label", ".stats")):
        return "generated"
    return "file"


def _collect_entries(output_root: Path) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    regular = sorted(
        ("file", path.relative_to(output_root).as_posix())
        for path in output_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    symlinks = sorted(
        ("symlink", path.relative_to(output_root).as_posix())
        for path in output_root.rglob("*")
        if path.is_symlink()
    )
    return regular, symlinks


def _collect_dirs(output_root: Path) -> list[tuple[str, str]]:
    return sorted(
        ("dir", path.relative_to(output_root).as_posix())
        for path in output_root.rglob("*")
        if path.is_dir() and not path.is_symlink()
    )


def _build_tree(entries: list[tuple[str, str]]) -> dict[str, Any]:
    root: dict[str, Any] = {"kind": "dir", "children": {}}
    for kind, rel in entries:
        node = root
        parts = rel.split("/")
        if kind == "dir":
            cur: list[str] = []
            for part in parts:
                cur.append(part)
                node = node["children"].setdefault(part, {"kind": "dir", "path": "/".join(cur), "children": {}})
            continue
        cur: list[str] = []
        for part in parts[:-1]:
            cur.append(part)
            node = node["children"].setdefault(part, {"kind": "dir", "path": "/".join(cur), "children": {}})
        node["children"][parts[-1]] = {"kind": kind, "path": rel, "children": {}}
    return root


def _render_node(name: str, node: dict[str, Any]) -> str:
    kind = node["kind"]
    path = str(node.get("path", ""))
    if kind == "dir":
        display = f"{name}/"
        cls = "folder"
        comment = _describe_dir(path)
    else:
        display = name
        cls = _css_class(kind, path)
        comment = _describe_file(path, symlink=kind == "symlink")
    child_html = ""
    if node["children"]:
        children = sorted(node["children"].items(), key=lambda item: (item[1]["kind"] != "dir", item[0].lower()))
        child_html = "<ul>\n" + "\n".join(_render_node(child_name, child) for child_name, child in children) + "\n</ul>"
    return f'<li><span class="{cls}">{_esc(display)}</span> <span class="comment"># {_esc(comment)}</span>{child_html}</li>'


def _render_tree(tree: dict[str, Any]) -> str:
    root_node = {"kind": "dir", "path": "", "children": tree["children"]}
    return '<ul class="tree">\n' + _render_node("outputs", root_node) + "\n</ul>"


def _postprocess_summary(output_root: Path) -> dict[str, Any]:
    cache_struct = (
        output_root.parent / "cache" / "stages" / "01_anatomical_priors" / "surface_morphometry" / "structural"
    )
    if (cache_struct / "qc.json").is_file() or (cache_struct / "manifest.json").is_file():
        directory = cache_struct
    else:
        directory = anatomy_input_dir(output_root, "ddsurfer/postprocess")
    manifest = _read_json(directory / "manifest.json")
    qc = _read_json(directory / "qc.json")
    core_outputs = qc.get("core_outputs") if isinstance(qc.get("core_outputs"), dict) else {}
    stage_status = manifest.get("stage_status") if isinstance(manifest.get("stage_status"), dict) else {}
    status = qc.get("status") or ("completed" if stage_status and all(value == "completed" for value in stage_status.values()) else "not available")
    return {
        "status": status,
        "stage_status": stage_status,
        "annot_count": core_outputs.get("annot_count", qc.get("annot_count")),
        "fsaverage_overlay_count": core_outputs.get("fsaverage_overlay_count", qc.get("fsaverage_overlay_count")),
        "stats_count": core_outputs.get("stats_count", qc.get("stats_count")),
    }


def _status_rows(summary: dict[str, Any]) -> str:
    stage_status = summary.get("stage_status")
    rows = [
        (
            "post-process status",
            summary.get("status") or "not available",
            "来自 postprocess/qc.json；若没有运行则显示 not available。",
        ),
        (
            "core output counts",
            f"annot={summary.get('annot_count') or 0}, fsaverage overlays={summary.get('fsaverage_overlay_count') or 0}, stats={summary.get('stats_count') or 0}",
            "来自 postprocess/qc.json 的 core_outputs。",
        ),
    ]
    if isinstance(stage_status, dict) and stage_status:
        rows.append(("stage status", ", ".join(f"{key}={value}" for key, value in sorted(stage_status.items())), "来自 postprocess/manifest.json。"))
    return "\n".join(
        f'<li><span class="metadata">{_esc(label)}</span> <span class="comment"># {_esc(value)}; {_esc(note)}</span></li>'
        for label, value, note in rows
    )


def _build_html(
    *,
    output_root: Path,
    subject_id: str,
    run_name: str,
    tree_html: str,
    regular_count: int,
    symlink_count: int,
    postprocess_summary: dict[str, Any],
) -> str:
    title = f"SlicerDSpace {subject_id or 'subject'} 输出文件树"
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{_esc(title)}</title>
  <style>
    :root {{
      --bg: #f7f5f0;
      --card: #fffdf9;
      --line: #ddd6c9;
      --text: #24303a;
      --muted: #6b7680;
      --blue: #2f5f8b;
      --green: #3f7a52;
      --amber: #94622b;
      --rose: #88445a;
      --cyan: #2d728f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", "Microsoft YaHei", Arial, sans-serif;
      background: radial-gradient(circle at 12% 0%, rgba(47, 95, 139, 0.08), transparent 34%), linear-gradient(180deg, #fbfaf6 0%, var(--bg) 42%, #efe9df 100%);
      color: var(--text);
    }}
    .page {{ max-width: 1380px; margin: 0 auto; padding: 28px 20px 44px; }}
    h1 {{ margin: 0 0 10px; font-size: 28px; letter-spacing: -0.02em; }}
    .subtitle {{ margin: 0 0 18px; color: var(--muted); line-height: 1.65; }}
    .grid {{ display: grid; gap: 16px; }}
    .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 16px; overflow: hidden; box-shadow: 0 18px 42px rgba(64, 51, 33, 0.07); }}
    .head {{ padding: 14px 16px; border-bottom: 1px solid var(--line); display: flex; justify-content: space-between; align-items: center; gap: 10px; }}
    .head b {{ font-size: 18px; }}
    .badges {{ display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 8px; }}
    .badge {{ font-size: 12px; color: var(--muted); border: 1px solid var(--line); background: #f5f2eb; padding: 4px 8px; border-radius: 999px; white-space: nowrap; }}
    .body {{ padding: 14px 16px 16px; overflow-x: auto; }}
    .summary {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }}
    .metric {{ border: 1px solid var(--line); background: #fbf8f0; border-radius: 12px; padding: 12px; }}
    .metric span {{ display: block; color: var(--muted); font-size: 12px; margin-bottom: 4px; }}
    .metric strong {{ font-size: 16px; word-break: break-all; }}
    .tree {{ margin: 0; padding: 0; list-style: none; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; font-size: 13px; line-height: 1.65; white-space: nowrap; }}
    .tree li {{ position: relative; padding-left: 20px; }}
    .tree li::before {{ content: ""; position: absolute; left: 5px; top: 0; bottom: 0; border-left: 1px solid #e1dacc; }}
    .tree li::after {{ content: ""; position: absolute; left: 5px; top: 13px; width: 10px; border-top: 1px solid #e1dacc; }}
    .tree > li::before, .tree > li::after {{ display: none; }}
    .tree ul {{ margin: 0; padding-left: 16px; list-style: none; }}
    .folder {{ color: var(--blue); font-weight: 800; }}
    .file {{ color: #41505d; }}
    .generated {{ color: var(--green); }}
    .optional {{ color: var(--amber); }}
    .metadata {{ color: var(--rose); }}
    .symlink {{ color: var(--cyan); font-weight: 700; }}
    .comment {{ color: #87919b; font-style: italic; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 12px; color: var(--muted); font-size: 13px; }}
    .legend code {{ border: 1px solid var(--line); background: #f5f2eb; border-radius: 999px; padding: 3px 7px; }}
    @media (max-width: 820px) {{
      .summary {{ grid-template-columns: 1fr; }}
      .head {{ align-items: flex-start; flex-direction: column; }}
      .badges {{ justify-content: flex-start; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <h1>{_esc(title)}</h1>
    <p class="subtitle">
      文件树数据源是当前 subject 的真实 outputs 目录。树节点包含 regular files 和 symlink entries；每个文件后面的注释说明其用途或所属流程。
    </p>
    <section class="grid">
      <article class="card">
        <div class="head">
          <b>输出树摘要</b>
          <div class="badges">
            <span class="badge">subject: {_esc(subject_id or 'unknown')}</span>
            <span class="badge">run: {_esc(run_name or 'unknown')}</span>
          </div>
        </div>
        <div class="body summary">
          <div class="metric"><span>输出根目录</span><strong>{_esc(output_root.as_posix())}</strong></div>
          <div class="metric"><span>regular files</span><strong>{regular_count}</strong></div>
          <div class="metric"><span>symlink entries</span><strong>{symlink_count}</strong></div>
          <div class="metric"><span>post-process 状态</span><strong>{_esc(postprocess_summary.get('status') or 'not available')}</strong></div>
        </div>
      </article>
      <article class="card">
        <div class="head">
          <b>图例</b>
          <span class="badge">颜色只表示文件类别</span>
        </div>
        <div class="body legend">
          <code><span class="folder">folder/</span> 目录</code>
          <code><span class="generated">generated</span> 主要生成产物</code>
          <code><span class="metadata">metadata</span> manifest / qc / report / log</code>
          <code><span class="optional">cache</span> 中间缓存或后端内部文件</code>
          <code><span class="symlink">symlink</span> 输出树链接条目</code>
        </div>
      </article>
      <article class="card">
        <div class="head">
          <b>DDSurfer post-process 实测状态</b>
          <span class="badge">从 manifest/qc 自动读取</span>
        </div>
        <div class="body">
          <ul class="tree">
            {_status_rows(postprocess_summary)}
          </ul>
        </div>
      </article>
      <article class="card">
        <div class="head">
          <b>输出树（真实节点逐项注释）</b>
          <span class="badge">{regular_count + symlink_count} file/symlink nodes</span>
        </div>
        <div class="body">
          {tree_html}
        </div>
      </article>
    </section>
  </main>
</body>
</html>
"""


def write_output_tree_report(
    *,
    output_root: Path,
    destination: Path,
    latest_destination: Optional[Path] = None,
    subject_id: str = "",
    run_name: str = "",
) -> OutputTreeReportPaths:
    """Write an annotated output-tree HTML report for one outputs directory."""

    output_root = output_root.expanduser().resolve()
    destination = destination.expanduser().resolve()
    latest_destination = latest_destination.expanduser().resolve() if latest_destination is not None else None
    if not output_root.is_dir():
        raise FileNotFoundError(f"Output root not found: {output_root}")
    normalized_run = normalize_run_name(run_name or "run")
    regular, symlinks = _collect_entries(output_root)
    dirs = _collect_dirs(output_root)
    tree = _build_tree(dirs + regular + symlinks)
    html_text = _build_html(
        output_root=output_root,
        subject_id=subject_id,
        run_name=normalized_run,
        tree_html=_render_tree(tree),
        regular_count=len(regular),
        symlink_count=len(symlinks),
        postprocess_summary=_postprocess_summary(output_root),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(html_text, encoding="utf-8")
    if latest_destination is not None:
        latest_destination.parent.mkdir(parents=True, exist_ok=True)
        latest_destination.write_text(html_text, encoding="utf-8")
    return OutputTreeReportPaths(
        report_html=destination,
        report_html_latest=latest_destination,
        regular_file_count=len(regular),
        symlink_count=len(symlinks),
    )


__all__ = ["OutputTreeReportPaths", "write_output_tree_report"]
