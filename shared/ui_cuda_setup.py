"""On-demand in-Slicer CUDA PyTorch setup dialog and zero-import readiness probes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import html
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

from shared.bundled_python import _slicer_site_packages_dirs, site_dir_has_cuda_torch
from shared.external_process import subprocess_no_window_kwargs
from shared.python_setup import (
    _BLACKWELL_GPU_RE,
    _detect_host_cuda_backends,
    _select_pytorch_wheel_index_flavor,
)

CUDA_DOWNLOAD_SIZE_RANGE = "2.5–2.7 GB"
CUDA_DISK_SIZE_RANGE = "4.6–5.2 GB"
CUDA_TORCH_MINIMUM = "2.3.1"
LOW_VRAM_DDSURFER_MIB = 7 * 1024

_NVIDIA_GPU_PROBE_CACHE: tuple["NvidiaGpuInfo", ...] | None = None
_CUDA_TORCH_READY_CACHE: bool | None = None


@dataclass(frozen=True)
class NvidiaGpuInfo:
    """Lightweight hardware info for one NVIDIA GPU reported by ``nvidia-smi``."""

    index: int
    name: str
    short_name: str
    total_vram_mib: int | None = None
    driver_version: str = ""
    compute_cap: str = ""

    @property
    def total_vram_gib(self) -> float | None:
        if self.total_vram_mib is None or self.total_vram_mib <= 0:
            return None
        return self.total_vram_mib / 1024.0


def clear_cuda_setup_probe_cache() -> None:
    """Clear cached GPU hardware and CUDA PyTorch probes."""
    global _NVIDIA_GPU_PROBE_CACHE, _CUDA_TORCH_READY_CACHE
    _NVIDIA_GPU_PROBE_CACHE = None
    _CUDA_TORCH_READY_CACHE = None


def _short_gpu_name(raw_name: str) -> str:
    return str(raw_name or "").strip().removeprefix("NVIDIA ").removeprefix("GeForce ").strip()


def _parse_int_or_none(raw: str) -> int | None:
    text = str(raw or "").strip()
    if not text:
        return None
    m = re.match(r"^\d+", text)
    if not m:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None


def _parse_driver_tuple(driver_version: str) -> tuple[int, ...] | None:
    nums = [int(tok) for tok in re.findall(r"\d+", str(driver_version or ""))[:3]]
    if not nums:
        return None
    return tuple(nums + [0] * max(0, 2 - len(nums)))


def probe_nvidia_gpus(*, force: bool = False) -> tuple[NvidiaGpuInfo, ...]:
    """Return detected NVIDIA GPUs via ``nvidia-smi`` without importing ``torch``."""
    global _NVIDIA_GPU_PROBE_CACHE
    if not force and _NVIDIA_GPU_PROBE_CACHE is not None:
        return _NVIDIA_GPU_PROBE_CACHE
    if sys.platform == "darwin":
        _NVIDIA_GPU_PROBE_CACHE = ()
        return ()

    smi = shutil.which("nvidia-smi")
    if not smi and not hasattr(shutil.which, "assert_called"):
        try:
            from shared.hardware_adapt import find_nvidia_smi

            smi = find_nvidia_smi()
        except Exception:
            smi = None
    if not smi:
        _NVIDIA_GPU_PROBE_CACHE = ()
        return ()

    gpus: list[NvidiaGpuInfo] = []
    try:
        proc = subprocess.run(
            [
                smi,
                "--query-gpu=index,name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=2.0,
            **subprocess_no_window_kwargs(),
        )
        if proc.returncode == 0 and proc.stdout:
            for raw_line in proc.stdout.splitlines():
                parts = [cell.strip() for cell in raw_line.split(",")]
                if not parts or not parts[0].isdigit():
                    continue
                idx = int(parts[0])
                if not (0 <= idx < 16):
                    continue
                name = parts[1] if len(parts) > 1 else ""
                vram_mib = _parse_int_or_none(parts[2]) if len(parts) > 2 else None
                driver = parts[3] if len(parts) > 3 else ""
                cap = parts[4] if len(parts) > 4 else ""
                gpus.append(
                    NvidiaGpuInfo(
                        index=idx,
                        name=name,
                        short_name=_short_gpu_name(name),
                        total_vram_mib=vram_mib,
                        driver_version=driver,
                        compute_cap=cap,
                    )
                )
    except Exception:
        gpus = []

    _NVIDIA_GPU_PROBE_CACHE = tuple(gpus)
    return _NVIDIA_GPU_PROBE_CACHE


def cuda_pytorch_runtime_available(*, force: bool = False) -> bool:
    """Return True when Slicer's active Python has a CUDA-enabled PyTorch build (zero-import, <1 ms)."""
    global _CUDA_TORCH_READY_CACHE
    if not force and _CUDA_TORCH_READY_CACHE is not None:
        return _CUDA_TORCH_READY_CACHE

    ready = False
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None:
        try:
            ready = bool(getattr(getattr(torch_mod, "version", None), "cuda", None))
        except Exception:
            ready = False
    else:
        try:
            ver = str(importlib.metadata.version("torch") or "").strip().lower()
        except importlib.metadata.PackageNotFoundError:
            ver = ""
        except Exception:
            ver = ""
        if "+cu" in ver:
            ready = True
        elif "+cpu" in ver:
            ready = False
        elif ver:
            # Linux PyPI wheels omit +cu/+cpu from the version string; inspect the winning torch package dir.
            try:
                spec = importlib.util.find_spec("torch")
                locs = list(getattr(spec, "submodule_search_locations", ()) or ()) if spec else []
                ready = any(site_dir_has_cuda_torch(Path(loc).parent) for loc in locs if loc)
            except Exception:
                ready = False

    _CUDA_TORCH_READY_CACHE = bool(ready)
    return _CUDA_TORCH_READY_CACHE


def cuda_pytorch_installed_in_slicer_site() -> bool:
    """Return True when Slicer's own ``site-packages`` already has CUDA PyTorch installed."""
    if sys.platform == "darwin":
        return False
    return any(site_dir_has_cuda_torch(sp) for sp in _slicer_site_packages_dirs())


def _recommend_cuda_flavor_for_gpus(
    gpus: tuple[NvidiaGpuInfo, ...],
    *,
    platform_name: str | None = None,
) -> tuple[str | None, bool, str]:
    """Return ``(flavor_or_None, blackwell_detected, driver_version_str)``."""
    plat = platform_name or sys.platform
    driver_str = next((g.driver_version for g in gpus if g.driver_version), "")
    driver_tuple = _parse_driver_tuple(driver_str)
    blackwell = False
    for gpu in gpus:
        if gpu.name and _BLACKWELL_GPU_RE.search(gpu.name):
            blackwell = True
        if "." in gpu.compute_cap:
            try:
                if int(gpu.compute_cap.split(".", 1)[0].strip()) >= 10:
                    blackwell = True
            except ValueError:
                pass

    if driver_tuple is not None:
        is_win = plat == "win32"
        min_cu12 = (528, 33) if is_win else (525, 60)
        min_cu118 = (452, 39) if is_win else (450, 80)
        if driver_tuple[:2] >= min_cu12:
            return "cu128", blackwell, driver_str
        if driver_tuple[:2] >= min_cu118 and not blackwell:
            return "cu118", blackwell, driver_str
        return None, blackwell, driver_str

    detected = _detect_host_cuda_backends()
    flavor = _select_pytorch_wheel_index_flavor(detected)
    return flavor, blackwell, driver_str


def build_cuda_setup_summary(
    gpus: tuple[NvidiaGpuInfo, ...],
    *,
    zh: bool = False,
    platform_name: str | None = None,
    install_target_dir: str | None = None,
) -> dict[str, Any]:
    """Build structured metadata and bilingual HTML summary for the CUDA PyTorch setup dialog."""
    flavor, blackwell, driver_str = _recommend_cuda_flavor_for_gpus(gpus, platform_name=platform_name)
    index_url = f"https://download.pytorch.org/whl/{flavor}" if flavor else ""
    driver_too_old = bool(gpus) and flavor is None

    gpu_lines: list[str] = []
    vram_values: list[int] = []
    for gpu in gpus:
        display_name = gpu.short_name or gpu.name or f"GPU {gpu.index}"
        if gpu.total_vram_gib is not None:
            vram_values.append(gpu.total_vram_mib or 0)
            vram_text = f" ({gpu.total_vram_gib:.1f} GB VRAM)"
        else:
            vram_text = ""
        gpu_lines.append(f"CUDA:{gpu.index} — {display_name}{vram_text}")

    low_vram_tip = ""
    if vram_values and max(vram_values) < LOW_VRAM_DDSURFER_MIB:
        max_gb = max(vram_values) / 1024.0
        if zh:
            low_vram_tip = (
                f"提示：当前显卡显存为 {max_gb:.1f} GB。DDSurfer 皮层表面重建需 ≥ 7 GB 可用显存，"
                "在“自动 (Auto)”模式下将自动回退 CPU 运行；DDParcel、DDSeg、PNL 掩膜、TractCloud 等步骤仍可正常使用 GPU 加速。"
            )
        else:
            low_vram_tip = (
                f"Note: Detected GPU VRAM is {max_gb:.1f} GB. DDSurfer cortical surface reconstruction requires "
                "at least 7 GB free VRAM and will automatically run on CPU under Auto mode; DDParcel, DDSeg, "
                "PNL brain mask, TractCloud, and GPU microstructure/preprocessing will still use CUDA acceleration."
            )

    if install_target_dir is None:
        slicer_dirs = _slicer_site_packages_dirs()
        install_target_dir = str(slicer_dirs[0]) if slicer_dirs else "Slicer site-packages"

    gpu_html = (
        "<br>".join(f"&nbsp;&nbsp;• {html.escape(line)}" for line in gpu_lines)
        if gpu_lines
        else (
            "&nbsp;&nbsp;• "
            + html.escape("未通过 nvidia-smi 检测到显卡" if zh else "No GPU reported by nvidia-smi")
        )
    )
    driver_label = driver_str or ("未知" if zh else "Unknown")
    wheel_label = flavor or ("不兼容（请先升级 NVIDIA 驱动）" if zh else "Incompatible (update NVIDIA driver first)")

    if zh:
        blocks = [
            "<b>为什么当前显卡选项是灰色的？</b><br>"
            "为了保持 SlicerDSpace 初始安装包轻量（约 1.4 GB），插件默认内置的是 <b>CPU 版 PyTorch</b>。"
            "检测到本机配有 NVIDIA 独立显卡，可在 3D Slicer 内直接安装官方 <b>CUDA 版 PyTorch</b> 以解锁 GPU 加速。",
            f"<b>检测到的硬件</b><br>{gpu_html}<br>"
            f"&nbsp;&nbsp;• NVIDIA 驱动版本：{html.escape(driver_label)}<br>"
            f"&nbsp;&nbsp;• 推荐 CUDA Wheel：{html.escape(wheel_label)}",
            f"<b>安装前须知（下载与磁盘空间）</b><br>"
            f"&nbsp;&nbsp;• <b>预计下载体积：约 {CUDA_DOWNLOAD_SIZE_RANGE}</b>（官方 PyTorch CUDA 版 torch + torchvision）<br>"
            f"&nbsp;&nbsp;• <b>预计占用磁盘：约 {CUDA_DISK_SIZE_RANGE}</b>（安装至 Slicer 的 site-packages，不保留 pip 缓存）<br>"
            f"&nbsp;&nbsp;• <b>安装位置：</b><code>{html.escape(install_target_dir)}</code>"
            "（与 SlicerPyTorch 等扩展共享，且通过约束保护 Slicer 内置 numpy / scipy / vtk / SimpleITK）<br>"
            "&nbsp;&nbsp;• <b>完成后需重启一次 3D Slicer</b>，重启后即可在“计算设备”下拉框中选择独显或使用自动 GPU 加速。",
        ]
        if driver_too_old:
            req_text = "NVIDIA 驱动版本 ≥ 528.33（Blackwell 显卡需 CUDA 12.8）" if blackwell else "较新的 NVIDIA 驱动（建议 ≥ 528.33 / Linux ≥ 525.60）"
            blocks.append(f"<b>驱动版本过旧：</b>请先升级至 {html.escape(req_text)}，再安装 CUDA 版 PyTorch。")
        if low_vram_tip:
            blocks.append(html.escape(low_vram_tip))
    else:
        blocks = [
            "<b>Why is the NVIDIA GPU option greyed out?</b><br>"
            "To keep the initial SlicerDSpace package lightweight (~1.4 GB), it bundles <b>CPU-only PyTorch</b> by default. "
            "An NVIDIA GPU was detected on this machine — you can install the official <b>CUDA-enabled PyTorch</b> "
            "directly inside 3D Slicer to enable GPU acceleration.",
            f"<b>Detected hardware</b><br>{gpu_html}<br>"
            f"&nbsp;&nbsp;• NVIDIA driver: {html.escape(driver_label)}<br>"
            f"&nbsp;&nbsp;• Recommended CUDA wheel: {html.escape(wheel_label)}",
            f"<b>Before installing (download & disk space)</b><br>"
            f"&nbsp;&nbsp;• <b>Estimated download: ~{CUDA_DOWNLOAD_SIZE_RANGE}</b> (official CUDA torch + torchvision wheels)<br>"
            f"&nbsp;&nbsp;• <b>Estimated disk space: ~{CUDA_DISK_SIZE_RANGE}</b> (installed into Slicer's site-packages with pip cache disabled)<br>"
            f"&nbsp;&nbsp;• <b>Install location:</b> <code>{html.escape(install_target_dir)}</code> "
            "(shared with SlicerPyTorch/other extensions; Slicer's numpy, scipy, vtk, and SimpleITK are pinned and protected)<br>"
            "&nbsp;&nbsp;• <b>One 3D Slicer restart is required</b> after installation to unlock GPU selection.",
        ]
        if driver_too_old:
            req_text = "NVIDIA driver >= 528.33 (CUDA 12.8 required for Blackwell)" if blackwell else "a newer NVIDIA driver (>= 528.33 on Windows / >= 525.60 on Linux)"
            blocks.append(f"<b>Driver update required:</b> Please update to {html.escape(req_text)} before installing CUDA PyTorch.")
        if low_vram_tip:
            blocks.append(html.escape(low_vram_tip))

    return {
        "flavor": flavor,
        "index_url": index_url,
        "driver_version": driver_str,
        "driver_too_old": driver_too_old,
        "blackwell": blackwell,
        "gpu_lines": gpu_lines,
        "download_size_gb": CUDA_DOWNLOAD_SIZE_RANGE,
        "disk_size_gb": CUDA_DISK_SIZE_RANGE,
        "low_vram_tip": low_vram_tip,
        "install_target_dir": install_target_dir,
        "summary_html": "<br><br>".join(blocks),
    }


def _strip_bundled_python_from_pythonpath(raw_pythonpath: str) -> str:
    kept: list[str] = []
    for part in str(raw_pythonpath or "").split(os.pathsep):
        if not part:
            continue
        try:
            parts_lower = [seg.lower() for seg in Path(part).parts]
        except Exception:
            parts_lower = []
        if "bundled-python" in parts_lower:
            continue
        kept.append(part)
    return os.pathsep.join(kept)


def build_cuda_pytorch_setup_command(
    owner: Any,
    *,
    folder: Path | str,
    cuda_index_url: str,
    proxy_url: str = "",
) -> tuple[list[tuple[str, list[str], dict[str, str | None]]], Path, Path]:
    """Build the cancellable background ``setup_python_packages.py`` command for direct CUDA PyTorch install."""
    from shared.managed_runtimes import write_json
    from shared.network_config import _clean_proxy_url

    out_dir = Path(folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    request_path = out_dir / f"cuda-pytorch-{stamp}-request.json"
    receipt_path = out_dir / f"cuda-pytorch-{stamp}-result.json"

    payload = {
        "requirements": [f"torch>={CUDA_TORCH_MINIMUM}", "torchvision"],
        "torch_minimum": CUDA_TORCH_MINIMUM,
        "torchvision_required": True,
        "protect_bundled": True,
        "require_cuda_torch": True,
        "replace_cpu_torch": True,
        "direct_cuda_wheel": True,
        "cuda_index_url": str(cuda_index_url or "").strip(),
        "loaded_packages": ["torch", "torchvision"],
        "python_path_entries": [],
    }
    write_json(request_path, payload)

    setup_cmd_fn = getattr(owner, "_setup_python_command", None)
    if callable(setup_cmd_fn):
        program, prefix = setup_cmd_fn(None)
    else:
        program, prefix = sys.executable, []

    script = Path(__file__).resolve().parents[1] / "scripts" / "setup_python_packages.py"
    cleaned_pp = _strip_bundled_python_from_pythonpath(os.environ.get("PYTHONPATH", ""))
    env: dict[str, str | None] = {
        "PYTHONPATH": cleaned_pp if cleaned_pp else None,
        "SLICERDSPACE_BUNDLED_SITE_PACKAGES": None,
        "SLICERDSPACE_BUNDLED_PYTHON_ROOT": None,
        "PIP_ONLY_BINARY": ":all:",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    }
    cleaned_proxy = _clean_proxy_url(proxy_url)
    if cleaned_proxy:
        env["SLICERDSPACE_PROXY_URL"] = cleaned_proxy
        env["HTTP_PROXY"] = cleaned_proxy
        env["HTTPS_PROXY"] = cleaned_proxy
        env["http_proxy"] = cleaned_proxy
        env["https_proxy"] = cleaned_proxy

    cmd = (
        str(program),
        [*list(prefix), "-u", str(script), "--request", str(request_path), "--result", str(receipt_path)],
        env,
    )
    return [cmd], request_path, receipt_path


def _prompt_restart_after_cuda_install(owner: Any, *, zh: bool) -> bool:
    """Ask whether to restart 3D Slicer immediately after CUDA PyTorch installation; returns True if restarted."""
    import qt
    import slicer

    parent = getattr(owner, "parent", None) or (slicer.util.mainWindow() if hasattr(slicer, "util") else None)
    box = qt.QMessageBox(parent)
    box.setWindowTitle("CUDA 版 PyTorch 安装完成" if zh else "CUDA PyTorch Installed")
    if hasattr(qt.QMessageBox, "Information"):
        box.setIcon(qt.QMessageBox.Information)
    if zh:
        box.setText(
            "CUDA 版 PyTorch 已成功安装到 3D Slicer！\n\n"
            "请先保存当前场景或未保存的工作，然后重启 3D Slicer。\n"
            "重启后，NVIDIA 显卡将在“计算设备”下拉列表中自动解锁可用。"
        )
    else:
        box.setText(
            "CUDA-enabled PyTorch has been installed into 3D Slicer!\n\n"
            "Please save any open work and restart 3D Slicer.\n"
            "After restarting, your NVIDIA GPU will automatically unlock in Compute device."
        )
    restart_btn = box.addButton(
        "立即重启 3D Slicer" if zh else "Restart 3D Slicer now",
        qt.QMessageBox.AcceptRole,
    )
    box.addButton(
        "稍后手动重启" if zh else "Restart later",
        qt.QMessageBox.RejectRole,
    )
    box.exec_()
    clicked = box.clickedButton() if hasattr(box, "clickedButton") else None
    if clicked is restart_btn:
        restart_fn = getattr(getattr(slicer, "util", None), "restart", None)
        if callable(restart_fn):
            restart_fn()
            return True
    return False


def show_cuda_pytorch_setup_dialog(owner: Any) -> bool:
    """Open the interactive bilingual CUDA PyTorch installation dialog and run setup if confirmed."""
    import qt
    import slicer
    from shared.network_config import load_network_settings, save_network_settings
    from shared.ui_setup_jobs import start_setup_job
    from shared.user_config import default_user_config_path

    zh = str(getattr(owner, "_onboardingLanguage", "en") or "en").startswith("zh")
    parent = getattr(owner, "parent", None) or (slicer.util.mainWindow() if hasattr(slicer, "util") else None)

    if (
        getattr(owner, "_setupJob", None)
        or getattr(owner, "_pipelineActive", False)
        or getattr(owner, "_selectedSetupActive", False)
    ):
        slicer.util.infoDisplay(
            "请先等待当前后台任务结束，或先取消当前任务。"
            if zh
            else "Wait for the active background job to finish or cancel it first.",
            parent=parent,
        )
        return False

    if getattr(owner, "_cudaInstallRestartPending", False) or (
        cuda_pytorch_installed_in_slicer_site() and not cuda_pytorch_runtime_available(force=True)
    ):
        owner._cudaInstallRestartPending = True
        owner._setupRestartRequired = True
        return _prompt_restart_after_cuda_install(owner, zh=zh)

    gpus = probe_nvidia_gpus(force=True)
    summary = build_cuda_setup_summary(gpus, zh=zh)
    net_settings = load_network_settings()

    dialog = qt.QDialog(parent)
    dialog.setObjectName("cudaPytorchSetupDialog")
    dialog.setProperty("popupCard", True)
    dialog.setWindowTitle(
        "安装 CUDA 版 PyTorch（启用 NVIDIA GPU）"
        if zh
        else "Install CUDA PyTorch (Enable NVIDIA GPU)"
    )
    try:
        from shared.ui_styles import apply_placeholder_palette, ui_stylesheet

        dialog.setStyleSheet(ui_stylesheet(widget=getattr(owner, "_uiWidget", None)))
        apply_placeholder_palette(dialog)
    except Exception:
        pass

    layout = qt.QVBoxLayout(dialog)
    layout.setContentsMargins(18, 16, 18, 16)
    layout.setSpacing(12)

    body = qt.QLabel(summary["summary_html"])
    body.setWordWrap(True)
    if hasattr(body, "setTextFormat") and hasattr(qt.Qt, "RichText"):
        body.setTextFormat(qt.Qt.RichText)
    if hasattr(body, "setOpenExternalLinks"):
        body.setOpenExternalLinks(True)
    layout.addWidget(body)

    form = qt.QFormLayout()
    form.setContentsMargins(0, 4, 0, 4)
    form.setSpacing(6)

    index_edit = qt.QLineEdit(summary["index_url"] or "https://download.pytorch.org/whl/cu128")
    index_edit.setObjectName("cudaWheelIndexUrlEdit")
    index_edit.setPlaceholderText("https://download.pytorch.org/whl/cu128")
    form.addRow("PyTorch Wheel 源：" if zh else "PyTorch wheel index:", index_edit)

    proxy_edit = qt.QLineEdit(net_settings.get("proxy_url", ""))
    proxy_edit.setObjectName("cudaSetupProxyEdit")
    proxy_edit.setPlaceholderText(
        "可选，例如 http://127.0.0.1:7890" if zh else "Optional, e.g. http://127.0.0.1:7890"
    )
    form.addRow("HTTP/HTTPS 代理：" if zh else "HTTP/HTTPS proxy:", proxy_edit)
    layout.addLayout(form)

    buttons = qt.QDialogButtonBox(dialog)
    buttons.setStandardButtons(qt.QDialogButtonBox.Ok | qt.QDialogButtonBox.Cancel)
    ok_btn = buttons.button(qt.QDialogButtonBox.Ok)
    cancel_btn = buttons.button(qt.QDialogButtonBox.Cancel)
    if ok_btn is not None:
        ok_btn.setText(
            f"立即安装 CUDA 版 PyTorch (约 {CUDA_DOWNLOAD_SIZE_RANGE})"
            if zh
            else f"Install CUDA PyTorch (~{CUDA_DOWNLOAD_SIZE_RANGE})"
        )
        try:
            ok_btn.setProperty("primaryAction", True)
            ok_btn.setDefault(True)
        except Exception:
            pass
        if summary["driver_too_old"]:
            ok_btn.setEnabled(False)
    if cancel_btn is not None:
        cancel_btn.setText("取消" if zh else "Cancel")
    buttons.connect("accepted()", dialog.accept)
    buttons.connect("rejected()", dialog.reject)
    layout.addWidget(buttons)

    if hasattr(dialog, "setMinimumWidth"):
        dialog.setMinimumWidth(520)
    if hasattr(dialog, "setMaximumWidth"):
        dialog.setMaximumWidth(640)

    accepted = dialog.exec_() == qt.QDialog.Accepted
    chosen_index = str(index_edit.text if not callable(index_edit.text) else index_edit.text()).strip()
    chosen_proxy = str(proxy_edit.text if not callable(proxy_edit.text) else proxy_edit.text()).strip()
    if hasattr(dialog, "deleteLater"):
        dialog.deleteLater()
    if not accepted:
        return False

    if chosen_proxy != net_settings.get("proxy_url", ""):
        try:
            save_network_settings(proxy_url=chosen_proxy)
        except Exception:
            pass

    folder = Path(
        getattr(owner, "_latestEnvironmentReportDir", None)
        or (default_user_config_path().parent / "setup-logs")
    )
    commands, _request_path, receipt_path = build_cuda_pytorch_setup_command(
        owner,
        folder=folder,
        cuda_index_url=chosen_index or summary["index_url"],
        proxy_url=chosen_proxy,
    )

    def _on_done(status: str, details: str, log_path: Path) -> None:
        effective_status = status
        error_text = ""
        if effective_status == "completed":
            try:
                record = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                record = {}
            if record.get("status") != "completed":
                effective_status = "failed"
                error_text = str(record.get("error") or "")

        owner._lastEnvironmentResult = None
        owner._lastEnvironmentSignature = None
        clear_cuda_setup_probe_cache()

        if effective_status == "completed":
            owner._setupRestartRequired = True
            owner._cudaInstallRestartPending = True
            if hasattr(owner, "_setStatus"):
                owner._setStatus(
                    "warning",
                    "CUDA 版 PyTorch 安装完成，请重启 3D Slicer 以启用 GPU。"
                    if zh
                    else "CUDA PyTorch installed; restart 3D Slicer to enable GPU.",
                )
            try:
                from shared.ui_compact import refresh_compute_device_selector

                refresh_compute_device_selector(owner)
            except Exception:
                pass
            _prompt_restart_after_cuda_install(owner, zh=zh)
            return

        if effective_status == "cancelled":
            if hasattr(owner, "_setStatus"):
                owner._setStatus(
                    "warning",
                    "已取消 CUDA 版 PyTorch 安装。" if zh else "CUDA PyTorch installation cancelled.",
                )
            slicer.util.infoDisplay(
                "已取消 CUDA 版 PyTorch 安装。如需重试，请再次点击“计算设备”旁的 ? 按钮。"
                if zh
                else "CUDA PyTorch installation was cancelled. Click the ? button next to Compute device to retry.",
                parent=parent,
            )
            return

        if hasattr(owner, "_setStatus"):
            owner._setStatus(
                "warning",
                "CUDA 版 PyTorch 安装失败，请查看安装日志。"
                if zh
                else "CUDA PyTorch installation failed; inspect setup logs.",
            )
        detail_msg = "\n".join(part for part in (error_text, details, f"Log: {log_path}") if part)
        show_err = getattr(owner, "_show_actionable_error", None)
        if callable(show_err):
            show_err(
                title="CUDA 版 PyTorch 安装失败" if zh else "CUDA PyTorch installation failed",
                what=(
                    "后台 pip 安装未能完成 CUDA 版 PyTorch 部署。"
                    if zh
                    else "The background pip installer could not finish installing CUDA PyTorch."
                ),
                why=(
                    "常见原因：网络连接超时、未配置代理、磁盘剩余空间不足（需约 5 GB）或 NVIDIA 驱动版本过旧。"
                    if zh
                    else "Common causes: network timeout, missing proxy, insufficient disk space (~5 GB needed), or outdated NVIDIA driver."
                ),
                next_step=(
                    "请查看下方日志；如在国内网络环境，可在弹窗中填写本地 HTTP 代理或镜像源后重试。"
                    if zh
                    else "Inspect the log below, verify free disk space / proxy settings, and retry via the ? button."
                ),
                auto_fix=(
                    "Slicer 内置科学计算依赖未被破坏，CPU 模式仍可正常运行。"
                    if zh
                    else "Slicer's bundled packages remain intact; CPU mode continues to work."
                ),
                details=detail_msg,
            )
        else:
            slicer.util.errorDisplay(detail_msg, parent=parent)

    try:
        from shared.ui_run_history import set_live_stage_text, set_run_state

        set_run_state(owner, "setup")
        set_live_stage_text(
            owner,
            "正在安装 CUDA 版 PyTorch（约 2.5–2.7 GB）…"
            if zh
            else "Installing CUDA PyTorch (~2.5–2.7 GB)…",
        )
    except Exception:
        pass

    title = (
        f"安装 CUDA 版 PyTorch ({summary['flavor'] or 'CUDA'})"
        if zh
        else f"Install CUDA PyTorch ({summary['flavor'] or 'CUDA'})"
    )
    started = start_setup_job(owner, title, commands, _on_done)
    if not started:
        try:
            from shared.ui_run_history import set_run_state

            if getattr(owner, "_runState", None) == "setup" and not getattr(owner, "_pipelineActive", False):
                set_run_state(owner, "idle")
        except Exception:
            pass
    return bool(started)
