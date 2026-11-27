"""Interactive DDParcel-guided ROI tractography UI (eyedropper region picker + side-task launcher)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from shared.roi_tracking import (
    ROI_ALGORITHM_LABELS,
    ROI_TRACKING_MANIFEST_KIND,
    load_label_names,
    resolve_ddparcel_outputs,
)


def _is_zh(owner: Any) -> bool:
    return str(getattr(owner, "_onboardingLanguage", "en") or "en").strip().lower().startswith("zh")


def resolve_owner_ddparcel(owner: Any) -> Optional[dict[str, Any]]:
    """Return the active subject's completed DDParcel parcellation descriptor, or None."""
    node = getattr(owner, "_parameterNode", None)
    if node is None:
        return None
    if bool(getattr(node, "batchMode", False)):
        return None
    subject_id = ""
    eff_fn = getattr(owner, "_effective_subject_id", None)
    if callable(eff_fn):
        try:
            subject_id = str(eff_fn() or "").strip()
        except Exception:
            subject_id = ""
    if not subject_id:
        subject_id = str(getattr(node, "subjectId", "") or "").strip()

    input_text = str(getattr(node, "inputDirectory", "") or "").strip()
    output_text = str(getattr(node, "outputDirectory", "") or "").strip()
    subject_root = Path(input_text).expanduser() if input_text else None
    output_root = Path(output_text).expanduser() if output_text else None

    # 1. Check run history records matching the active subject first (most up-to-date manifest).
    records = getattr(owner, "_runHistoryRecords", None) or ()
    for record in records:
        if getattr(record, "is_batch", False):
            continue
        rec_sid = str(getattr(record, "subject_id", "") or "").strip()
        if subject_id and rec_sid and rec_sid != subject_id:
            continue
        mpath = getattr(record, "manifest_path", None)
        if not mpath or not Path(mpath).is_file():
            continue
        try:
            manifest = json.loads(Path(mpath).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for task in manifest.get("tasks") or []:
            if not isinstance(task, dict) or str(task.get("name") or "") != "parcellation.ddparcel":
                continue
            if task.get("status") not in {"completed", "skipped"}:
                continue
            outputs = task.get("outputs") or {}
            wmparc = outputs.get("wmparc")
            if isinstance(wmparc, str) and Path(wmparc).is_file():
                ctbl = outputs.get("color_table")
                rec_sub_root = Path(str(manifest.get("subject_root") or "")).expanduser() if manifest.get("subject_root") else subject_root
                rec_out_root = Path(str(manifest.get("output_root") or "")).expanduser() if manifest.get("output_root") else output_root
                return {
                    "parcellation_path": Path(wmparc).resolve(),
                    "color_table_path": Path(ctbl).resolve() if isinstance(ctbl, str) and Path(ctbl).is_file() else None,
                    "subject_id": rec_sid or subject_id,
                    "subject_root": rec_sub_root,
                    "output_root": rec_out_root,
                    "manifest_path": Path(mpath).resolve(),
                }

    # 2. Direct filesystem check in configured subject_root / output_root.
    found = resolve_ddparcel_outputs(subject_root, subject_id, output_root=output_root)
    if found is not None:
        found["subject_root"] = subject_root
        if found.get("output_root") is None:
            found["output_root"] = output_root
        found["manifest_path"] = getattr(owner, "_latestResultManifestPath", None)
        return found
    return None


def _eyedropper_svg(color: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24">'
        f'<path fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
        f'd="M2 22l1-1h3l9-9M13 7l4 4M15.5 4.5l4 4a2.12 2.12 0 0 1 0 3l-1.5 1.5-7-7 1.5-1.5a2.12 2.12 0 0 1 3 0z"/>'
        f"</svg>"
    )


def _eyedropper_icon(qt: Any, *, dark: bool = False, active: bool = False) -> Any:
    color = ("#7CC4EA" if dark else "#1B5E8A") if active else ("#D2DEE8" if dark else "#334155")
    pix = qt.QPixmap()
    pix.loadFromData(qt.QByteArray(_eyedropper_svg(color).encode("utf-8")))
    return qt.QIcon(pix)


def _eyedropper_cursor(qt: Any) -> Any:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">'
        '<path fill="#FFFFFF" stroke="#0F172A" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" '
        'd="M2 22l1.5-1.5h3l9-9-3-3-9 9v3L2 22z"/>'
        '<path fill="#38BDF8" stroke="#0F172A" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" '
        'd="M12.5 8.5l3 3 5-5a2.12 2.12 0 0 0-3-3l-5 5z"/>'
        '<circle cx="2" cy="22" r="1.4" fill="#EF4444"/>'
        "</svg>"
    )
    pix = qt.QPixmap()
    pix.loadFromData(qt.QByteArray(svg.encode("utf-8")))
    if hasattr(qt, "QCursor"):
        return qt.QCursor(pix, 2, 22)
    return None


def _algorithm_label(algorithm: str, zh: bool) -> str:
    pair = ROI_ALGORITHM_LABELS.get(str(algorithm or "").strip().lower())
    if pair:
        return pair[1 if zh else 0]
    return str(algorithm or "").upper()


def install_roi_tracking_bar(owner: Any) -> None:
    """Install the compact inline ROI Tracking bar inside 2. Analysis (hidden until activated)."""
    if getattr(owner, "_roiTrackingWidgets", None) is not None:
        return
    ui = getattr(owner, "ui", None)
    targets_layout = getattr(ui, "targetsLayout", None) if ui is not None else None
    if targets_layout is None:
        return
    try:
        import qt
    except Exception:
        return

    bar = qt.QFrame()
    bar.setObjectName("dspaceRoiTrackingBar")
    styled_bg = getattr(qt.Qt, "WA_StyledBackground", None)
    if styled_bg is not None and hasattr(bar, "setAttribute"):
        bar.setAttribute(styled_bg, True)
    bar.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Maximum)
    bar.setMinimumWidth(0)

    outer = qt.QVBoxLayout(bar)
    outer.setContentsMargins(8, 6, 8, 6)
    outer.setSpacing(4)

    header = qt.QHBoxLayout()
    header.setContentsMargins(0, 0, 0, 0)
    header.setSpacing(6)
    badge = qt.QLabel("ROI")
    badge.setObjectName("dspaceRoiTrackingBadge")
    header.addWidget(badge)
    header.addStretch(1)

    pointing = getattr(qt.Qt, "PointingHandCursor", None)
    picker_btn = qt.QToolButton()
    picker_btn.setObjectName("dspaceRoiEyedropperButton")
    picker_btn.setCheckable(True)
    picker_btn.setAutoRaise(True)
    picker_btn.setFixedSize(24, 22)
    if pointing is not None and hasattr(picker_btn, "setCursor"):
        picker_btn.setCursor(pointing)
    header.addWidget(picker_btn)

    close_btn = qt.QToolButton()
    close_btn.setObjectName("dspaceRoiCloseButton")
    close_btn.setText("×")
    close_btn.setAutoRaise(True)
    close_btn.setFixedSize(20, 20)
    if pointing is not None and hasattr(close_btn, "setCursor"):
        close_btn.setCursor(pointing)
    header.addWidget(close_btn)
    outer.addLayout(header)

    row = qt.QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(5)

    region_edit = qt.QLineEdit()
    region_edit.setObjectName("dspaceRoiRegionEdit")
    region_edit.setReadOnly(True)
    region_edit.setMinimumWidth(0)
    row.addWidget(region_edit, 1)

    clear_btn = qt.QToolButton()
    clear_btn.setObjectName("dspaceRoiClearButton")
    clear_btn.setText("×")
    clear_btn.setAutoRaise(True)
    clear_btn.setFixedSize(20, 22)
    clear_btn.hide()
    if pointing is not None and hasattr(clear_btn, "setCursor"):
        clear_btn.setCursor(pointing)
    row.addWidget(clear_btn)

    launch_btn = qt.QPushButton()
    launch_btn.setObjectName("dspaceRoiLaunchButton")
    launch_btn.setEnabled(False)
    launch_btn.setSizePolicy(qt.QSizePolicy.Maximum, qt.QSizePolicy.Fixed)
    if pointing is not None and hasattr(launch_btn, "setCursor"):
        launch_btn.setCursor(pointing)
    row.addWidget(launch_btn)
    outer.addLayout(row)

    hint = qt.QLabel("")
    hint.setObjectName("dspaceRoiHintLabel")
    hint.setWordWrap(True)
    hint.setMinimumWidth(0)
    outer.addWidget(hint)

    bar.hide()
    # Insert directly above Run SlicerDSpace so it sits cleanly inside 2. Analysis.
    run_btn = getattr(ui, "runCollapsibleButton", None)
    idx = targets_layout.indexOf(run_btn) if run_btn is not None else -1
    if idx >= 0:
        targets_layout.insertWidget(idx, bar)
    else:
        targets_layout.addWidget(bar)

    owner._roiTrackingWidgets = {
        "bar": bar,
        "badge": badge,
        "picker_btn": picker_btn,
        "close_btn": close_btn,
        "region_edit": region_edit,
        "clear_btn": clear_btn,
        "launch_btn": launch_btn,
        "hint": hint,
    }
    owner._roiTrackingState = {
        "algorithm": "ukf",
        "ddparcel": None,
        "selected": [],  # list of (label_id, region_name)
        "picking": False,
        "volume_array": None,
        "ras_to_ijk": None,
        "label_names": {},
        "event_filters": [],
        "process": None,
        "output_buf": "",
    }

    picker_btn.toggled.connect(lambda checked: _set_eyedropper_active(owner, bool(checked)))
    close_btn.clicked.connect(lambda *_args: close_roi_tracking_bar(owner))
    clear_btn.clicked.connect(lambda *_args: _clear_selected_regions(owner))
    launch_btn.clicked.connect(lambda *_args: _launch_roi_tracking(owner))

    refresh_roi_tracking_language(owner)
    refresh_roi_tracking_styles(owner)


def refresh_roi_tracking_styles(owner: Any, *, dark: Optional[bool] = None) -> None:
    widgets = getattr(owner, "_roiTrackingWidgets", None)
    if not widgets:
        return
    try:
        import qt
        from shared.ui_styles import is_dark_mode
    except Exception:
        return
    use_dark = is_dark_mode(getattr(owner, "_uiWidget", None)) if dark is None else bool(dark)
    state = getattr(owner, "_roiTrackingState", None) or {}
    bg = "#16222E" if use_dark else "#F0F7FB"
    border = "#2B5673" if use_dark else "#B6D4E4"
    badge_bg = "#1D4E70" if use_dark else "#1B5E8A"
    ink = "#E6EDF3" if use_dark else "#0F2744"
    muted = "#9FB2C3" if use_dark else "#4A6073"
    edit_bg = "#0D1620" if use_dark else "#FFFFFF"
    edit_border = "#355068" if use_dark else "#C5D8E4"
    btn_bg = "#2A7298" if use_dark else "#1B5E8A"
    btn_hover = "#3386B2" if use_dark else "#144A6E"
    btn_disabled_bg = "#2A3642" if use_dark else "#CBD5E1"
    btn_disabled_fg = "#7E8E9E" if use_dark else "#64748B"

    widgets["bar"].setStyleSheet(
        f"QFrame#dspaceRoiTrackingBar {{ background: {bg}; border: 1px solid {border}; border-radius: 7px; }}"
        f"QLabel#dspaceRoiTrackingBadge {{ background: {badge_bg}; color: #FFFFFF; border-radius: 4px; "
        f"padding: 1px 7px; font-size: 11px; font-weight: 600; }}"
        f"QLineEdit#dspaceRoiRegionEdit {{ background: {edit_bg}; color: {ink}; border: 1px solid {edit_border}; "
        f"border-radius: 4px; padding: 3px 6px; font-size: 11px; }}"
        f"QLabel#dspaceRoiHintLabel {{ color: {muted}; font-size: 11px; }}"
        f"QPushButton#dspaceRoiLaunchButton {{ background: {btn_bg}; color: #FFFFFF; border: 0; border-radius: 4px; "
        f"padding: 4px 10px; font-size: 11px; font-weight: 600; }}"
        f"QPushButton#dspaceRoiLaunchButton:hover {{ background: {btn_hover}; }}"
        f"QPushButton#dspaceRoiLaunchButton:disabled {{ background: {btn_disabled_bg}; color: {btn_disabled_fg}; }}"
    )
    widgets["picker_btn"].setIcon(_eyedropper_icon(qt, dark=use_dark, active=bool(state.get("picking"))))


def refresh_roi_tracking_language(owner: Any, language: Optional[str] = None) -> None:
    widgets = getattr(owner, "_roiTrackingWidgets", None)
    if not widgets:
        return
    zh = str(language or getattr(owner, "_onboardingLanguage", "en")).startswith("zh")
    state = getattr(owner, "_roiTrackingState", None) or {}
    alg = state.get("algorithm", "ukf")
    widgets["badge"].setText(f"ROI · {_algorithm_label(alg, zh)}")
    widgets["picker_btn"].setToolTip(
        "开启/关闭脑区取色笔（在 2D 切片上点击 DDParcel 脑区）" if zh
        else "Toggle region eyedropper (click a DDParcel region in any 2D slice view)"
    )
    widgets["close_btn"].setToolTip("关闭 ROI 追踪面板" if zh else "Close ROI tracking bar")
    widgets["clear_btn"].setToolTip("清空已选脑区" if zh else "Clear selected region(s)")
    widgets["region_edit"].setPlaceholderText(
        "点击 2D 切片中的脑区进行选择…" if zh else "Click a brain region in 2D slice view…"
    )
    widgets["launch_btn"].setText("▶ 开始追踪" if zh else "▶ Track")
    widgets["launch_btn"].setToolTip(
        "启动针对所选脑区的局部 ROI 纤维追踪（独立支线结果，不影响全脑分束）" if zh
        else "Start ROI tractography for the selected region(s) (standalone side result)"
    )
    if not state.get("selected") and not widgets["hint"].text:
        widgets["hint"].setText(
            "已在场景加载 DDParcel 分区，点击任意 2D 切片脑区即可拾取（右键或 Esc 退出取色笔）。" if zh
            else "DDParcel loaded in scene. Click any brain region in 2D slice views to pick (Right-click / Esc to stop)."
        )


def _sync_region_display(owner: Any) -> None:
    widgets = getattr(owner, "_roiTrackingWidgets", None)
    state = getattr(owner, "_roiTrackingState", None)
    if not widgets or not state:
        return
    selected = state.get("selected") or []
    running = state.get("process") is not None
    if selected:
        text = ", ".join(f"{name} ({lid})" for lid, name in selected)
        widgets["region_edit"].setText(text)
        widgets["region_edit"].setToolTip(text)
    else:
        widgets["region_edit"].setText("")
        widgets["region_edit"].setToolTip("")
    widgets["clear_btn"].setVisible(bool(selected) and not running)
    widgets["launch_btn"].setEnabled(bool(selected) and not running)


def _clear_selected_regions(owner: Any) -> None:
    state = getattr(owner, "_roiTrackingState", None)
    if not state:
        return
    state["selected"] = []
    _sync_region_display(owner)


def _load_ddparcel_into_scene(owner: Any, ddparcel_info: dict[str, Any]) -> None:
    """Auto-load DDParcel parcellation (+ clean FA/b0 background if available) into the Slicer scene."""
    import slicer
    from shared.result_scene import load_categorical_result, register_node, result_nodes, source_stamp

    parc_path = Path(ddparcel_info["parcellation_path"]).resolve()
    subject_id = str(ddparcel_info.get("subject_id") or "subject")
    manifest_path = ddparcel_info.get("manifest_path")
    run_id = str(manifest_path) if manifest_path else str(parc_path.parent)

    # Check if already loaded in scene.
    existing = None
    for node in result_nodes(slicer):
        src = node.GetAttribute("SlicerDDSurfer.SourcePath")
        if src:
            try:
                if Path(src).resolve() == parc_path:
                    existing = node
                    break
            except OSError:
                pass

    if existing is None:
        candidate = {
            "path": str(parc_path),
            "product": "parcellation.ddparcel",
            "key": "wmparc",
            "label": "parcellation_ddparcel",
            "color_table_path": str(ddparcel_info["color_table_path"]) if ddparcel_info.get("color_table_path") else None,
        }
        existing = load_categorical_result(slicer, parc_path, f"{subject_id}_DDParcel", candidate)
        if existing is not None:
            register_node(slicer, existing, run_id, subject_id, "ddparcel", "parcellation.ddparcel", str(parc_path))

    if existing is not None:
        if hasattr(existing, "SetDisplayVisibility"):
            existing.SetDisplayVisibility(True)
        display = existing.GetDisplayNode() if hasattr(existing, "GetDisplayNode") else None
        if display is not None:
            if hasattr(display, "SetVisibility2DFill"):
                display.SetVisibility2DFill(True)
            if hasattr(display, "SetVisibility2DOutline"):
                display.SetVisibility2DOutline(True)
            if hasattr(display, "SetOpacity2DFill"):
                display.SetOpacity2DFill(0.40)

    # Ensure a crisp anatomical background slice (prefer existing background, else FA, else mean b0).
    has_bg = False
    try:
        for comp in slicer.util.getNodesByClass("vtkMRMLSliceCompositeNode"):
            if comp.GetBackgroundVolumeID():
                has_bg = True
                break
    except Exception:
        has_bg = False
    if not has_bg:
        out_root = ddparcel_info.get("output_root")
        if out_root is not None:
            fa_candidates = list((Path(out_root) / "02_microstructure" / "diffusion_tensor").glob("*_FA.nii.gz"))
            if fa_candidates:
                try:
                    bg = slicer.util.loadVolume(str(fa_candidates[0]), properties={"name": f"{subject_id}_FA", "show": True})
                    if bg is not None:
                        slicer.util.setSliceViewerLayers(background=bg, fit=True)
                        has_bg = True
                except Exception:
                    pass
    try:
        if existing is not None and hasattr(existing, "GetRASBounds"):
            bounds = [0.0] * 6
            existing.GetRASBounds(bounds)
            if bounds[1] > bounds[0]:
                center = [(bounds[i] + bounds[i + 1]) / 2.0 for i in (0, 2, 4)]
                slicer.vtkMRMLSliceNode.JumpAllSlices(
                    slicer.mrmlScene, *center, slicer.vtkMRMLSliceNode.CenteredJumpSlice
                )
        if hasattr(slicer.util, "resetSliceViews"):
            slicer.util.resetSliceViews()
    except Exception:
        pass


def _prepare_volume_lookup(state: dict[str, Any], ddparcel_info: dict[str, Any]) -> None:
    import nibabel as nib
    import numpy as np

    parc_path = Path(ddparcel_info["parcellation_path"]).resolve()
    img = nib.load(str(parc_path))
    arr = np.rint(np.asarray(img.get_fdata())).astype(np.int32)
    inv_aff = np.linalg.inv(np.asarray(img.affine, dtype=float))
    state["volume_array"] = arr
    state["ras_to_ijk"] = inv_aff
    state["label_names"] = load_label_names(parc_path, ddparcel_info.get("color_table_path"))


def _sample_region_at_slice_xy(state: dict[str, Any], slice_view: Any, pos_x: float, pos_y: float) -> Optional[tuple[int, str]]:
    """Convert Qt slice-view coordinates to DDParcel (label_id, region_name), or None if background."""
    import numpy as np

    arr = state.get("volume_array")
    ras_to_ijk = state.get("ras_to_ijk")
    if arr is None or ras_to_ijk is None:
        return None
    try:
        slice_node = slice_view.mrmlSliceNode()
        if slice_node is None:
            return None
        width = slice_view.width if not callable(getattr(slice_view, "width", None)) else slice_view.width()
        height = slice_view.height if not callable(getattr(slice_view, "height", None)) else slice_view.height()
        # Map Qt logical coordinates into VTK slice_node XY dimensions (exact on 100%-200% Windows DPI & Retina macOS).
        dims = slice_node.GetDimensions() if hasattr(slice_node, "GetDimensions") else None
        sx = (float(dims[0]) / float(width)) if (dims and len(dims) >= 2 and dims[0] > 0 and width > 0) else 1.0
        sy = (float(dims[1]) / float(height)) if (dims and len(dims) >= 2 and dims[1] > 0 and height > 0) else 1.0
        # VTK display Y is measured from the bottom of the widget.
        vtk_x = float(pos_x) * sx
        vtk_y = (float(height) - 1.0 - float(pos_y)) * sy
        xy_to_ras = slice_node.GetXYToRAS()
        ras = [0.0, 0.0, 0.0, 1.0]
        xy_to_ras.MultiplyPoint([vtk_x, vtk_y, 0.0, 1.0], ras)
        ijk = ras_to_ijk @ np.array([ras[0], ras[1], ras[2], 1.0], dtype=float)
        i, j, k = int(round(ijk[0])), int(round(ijk[1])), int(round(ijk[2]))
        if not (0 <= i < arr.shape[0] and 0 <= j < arr.shape[1] and 0 <= k < arr.shape[2]):
            return None
        label_id = int(arr[i, j, k])
        if label_id <= 0:
            return None
        name = (state.get("label_names") or {}).get(label_id, f"Label_{label_id}")
        return label_id, name
    except Exception:
        return None


_FILTER_CLASS_CACHE: dict[type, type] = {}


def _make_slice_picker_filter(qt: Any, owner: Any, slice_view: Any):
    qobj_cls = getattr(qt, "QObject", None)
    if not isinstance(qobj_cls, type):
        return None
    filter_cls = _FILTER_CLASS_CACHE.get(qobj_cls)
    if filter_cls is None:
        class _RoiSlicePickerFilter(qobj_cls):
            def eventFilter(self, obj, event):
                ctx = getattr(self, "_roiCtx", None)
                if not ctx:
                    return False
                own, view, qt_mod = ctx
                st = getattr(own, "_roiTrackingState", None)
                if not st or not st.get("picking"):
                    return False
                etype = event.type()
                if etype == qt_mod.QEvent.MouseMove:
                    pos = event.pos()
                    px = pos.x() if callable(getattr(pos, "x", None)) else pos.x
                    py = pos.y() if callable(getattr(pos, "y", None)) else pos.y
                    hit = _sample_region_at_slice_xy(st, view, px, py)
                    zh = _is_zh(own)
                    hint = (getattr(own, "_roiTrackingWidgets", None) or {}).get("hint")
                    if hint is not None:
                        if hit is not None:
                            hint.setText(
                                f"当前指向脑区：{hit[1]} (#{hit[0]}) — 左键点击选中" if zh
                                else f"Hover region: {hit[1]} (#{hit[0]}) — Left-click to select"
                            )
                        else:
                            hint.setText(
                                "将取色笔移至 2D 切片内的脑区并左键点击（右键/Esc 退出取色笔）。" if zh
                                else "Move eyedropper onto a brain region and Left-click (Right-click / Esc to stop)."
                            )
                    return False
                if etype == qt_mod.QEvent.MouseButtonPress:
                    btn = event.button()
                    if btn == qt_mod.Qt.RightButton:
                        _set_eyedropper_active(own, False)
                        return True
                    if btn == qt_mod.Qt.LeftButton:
                        pos = event.pos()
                        px = pos.x() if callable(getattr(pos, "x", None)) else pos.x
                        py = pos.y() if callable(getattr(pos, "y", None)) else pos.y
                        hit = _sample_region_at_slice_xy(st, view, px, py)
                        if hit is not None:
                            _record_picked_region(own, hit[0], hit[1])
                            return True
                        return False
                if etype == qt_mod.QEvent.KeyPress:
                    if event.key() == qt_mod.Qt.Key_Escape:
                        _set_eyedropper_active(own, False)
                        return True
                return False

        filter_cls = _RoiSlicePickerFilter
        _FILTER_CLASS_CACHE[qobj_cls] = filter_cls
    filt = filter_cls(slice_view)
    filt._roiCtx = (owner, slice_view, qt)
    return filt


def _record_picked_region(owner: Any, label_id: int, region_name: str) -> None:
    state = getattr(owner, "_roiTrackingState", None)
    widgets = getattr(owner, "_roiTrackingWidgets", None)
    if not state or not widgets:
        return
    selected: list[tuple[int, str]] = list(state.get("selected") or [])
    existing_ids = [lid for lid, _ in selected]
    if label_id not in existing_ids:
        selected.append((label_id, region_name))
    state["selected"] = selected
    _sync_region_display(owner)
    zh = _is_zh(owner)
    widgets["hint"].setText(
        f"已选中：{region_name} (#{label_id})。可继续点击叠加其他脑区，或点击右侧「开始追踪」。" if zh
        else f"Selected: {region_name} (#{label_id}). Click more regions to combine, or click Track on the right."
    )
    if hasattr(owner, "appendLog"):
        owner.appendLog(f"[ROI Tracking] Selected DDParcel region: {region_name} (#{label_id})")


def _set_eyedropper_active(owner: Any, active: bool) -> None:
    state = getattr(owner, "_roiTrackingState", None)
    widgets = getattr(owner, "_roiTrackingWidgets", None)
    if not state:
        return
    try:
        import qt
        import slicer
    except Exception:
        state["picking"] = False
        return

    if not active:
        state["picking"] = False
        for view, filt in list(state.get("event_filters") or ()):
            try:
                view.removeEventFilter(filt)
            except Exception:
                pass
            try:
                view.unsetCursor()
            except Exception:
                pass
        state["event_filters"] = []
        if widgets and "picker_btn" in widgets:
            btn = widgets["picker_btn"]
            blocked = btn.blockSignals(True)
            try:
                btn.setChecked(False)
            finally:
                btn.blockSignals(blocked)
        refresh_roi_tracking_styles(owner)
        return

    # Activate eyedropper across all 2D slice views.
    _set_eyedropper_active(owner, False)
    state["picking"] = True
    cursor = _eyedropper_cursor(qt)
    lm = slicer.app.layoutManager() if hasattr(slicer, "app") and hasattr(slicer.app, "layoutManager") else None
    filters = []
    if lm is not None:
        names = list(lm.sliceViewNames()) if hasattr(lm, "sliceViewNames") else ["Red", "Yellow", "Green"]
        for name in names:
            sw = lm.sliceWidget(name)
            if sw is None:
                continue
            sv = sw.sliceView()
            if sv is None:
                continue
            filt = _make_slice_picker_filter(qt, owner, sv)
            if filt is not None:
                sv.installEventFilter(filt)
                if cursor is not None:
                    sv.setCursor(cursor)
                if hasattr(sv, "setMouseTracking"):
                    sv.setMouseTracking(True)
                filters.append((sv, filt))
    state["event_filters"] = filters
    if widgets and "picker_btn" in widgets:
        btn = widgets["picker_btn"]
        blocked = btn.blockSignals(True)
        try:
            btn.setChecked(True)
        finally:
            btn.blockSignals(blocked)
    refresh_roi_tracking_styles(owner)


def deactivate_roi_eyedropper(owner: Any) -> None:
    """Safely remove slice event filters and restore default cursors (e.g. on widget cleanup)."""
    _set_eyedropper_active(owner, False)


def close_roi_tracking_bar(owner: Any) -> None:
    deactivate_roi_eyedropper(owner)
    widgets = getattr(owner, "_roiTrackingWidgets", None)
    if widgets and "bar" in widgets:
        widgets["bar"].hide()
        fit = getattr(owner, "_queueCompactFit", None)
        if callable(fit):
            fit()


def open_roi_tracking_session(owner: Any, algorithm: str, ddparcel_info: Optional[dict[str, Any]] = None) -> bool:
    """Load DDParcel into Slicer, reveal the inline ROI bar, and activate the eyedropper cursor."""
    install_roi_tracking_bar(owner)
    widgets = getattr(owner, "_roiTrackingWidgets", None)
    state = getattr(owner, "_roiTrackingState", None)
    if not widgets or not state:
        return False
    info = ddparcel_info or resolve_owner_ddparcel(owner)
    zh = _is_zh(owner)
    if info is None:
        if hasattr(owner, "_setStatus"):
            owner._setStatus(
                "warning",
                "请先运行脑区分区（DDParcel）后再使用 ROI 纤维追踪。" if zh
                else "Run Brain parcellation (DDParcel) first before using ROI tracking.",
            )
        return False

    state["algorithm"] = str(algorithm or "ukf").strip().lower()
    state["ddparcel"] = info
    state["selected"] = []
    refresh_roi_tracking_language(owner)
    _sync_region_display(owner)

    try:
        _load_ddparcel_into_scene(owner, info)
        _prepare_volume_lookup(state, info)
    except Exception as exc:
        if hasattr(owner, "appendLog"):
            owner.appendLog(f"[ROI Tracking] Failed to prepare DDParcel scene: {exc}")
        widgets["hint"].setText(f"加载 DDParcel 失败：{exc}" if zh else f"Failed to load DDParcel: {exc}")
        widgets["bar"].show()
        return False

    widgets["hint"].setText(
        "已自动加载 DDParcel 分区并切换为取色笔，请在 2D 切片中点击目标脑区（右键/Esc 退出取色笔）。" if zh
        else "DDParcel loaded and eyedropper active. Click any target brain region in 2D slice views (Right-click / Esc to stop)."
    )
    widgets["bar"].show()
    _set_eyedropper_active(owner, True)
    fit = getattr(owner, "_queueCompactFit", None)
    if callable(fit):
        fit()
    return True


def cancel_roi_tracking_job(owner: Any) -> bool:
    state = getattr(owner, "_roiTrackingState", None) or {}
    proc = state.get("process")
    if proc is None:
        return False
    try:
        from shared.ui_background_process import qprocess_is_running

        if qprocess_is_running(proc):
            proc.kill()
            return True
    except Exception:
        pass
    return False


def _launch_roi_tracking(owner: Any) -> None:
    import qt
    import slicer
    from shared.external_process import normalize_slicer_python_launch
    from shared.ui_background_process import background_process_env_overrides, qprocess_is_running
    from shared.ui_run_history import set_live_stage_text, set_run_state, stop_run_animation

    state = getattr(owner, "_roiTrackingState", None) or {}
    widgets = getattr(owner, "_roiTrackingWidgets", None) or {}
    zh = _is_zh(owner)
    selected = state.get("selected") or []
    info = state.get("ddparcel")
    if not selected or not info:
        return
    if getattr(owner, "_pipelineActive", False) or qprocess_is_running(state.get("process")):
        slicer.util.infoDisplay(
            "请等待当前任务完成或取消后再启动 ROI 追踪。" if zh else "Wait for the active job to finish or cancel it first.",
            parent=getattr(owner, "parent", None),
        )
        return

    deactivate_roi_eyedropper(owner)
    alg = str(state.get("algorithm") or "ukf")
    subject_id = str(info.get("subject_id") or owner._effective_subject_id() or "subject")
    subject_root = info.get("subject_root") or Path(str(getattr(owner._parameterNode, "inputDirectory", "") or "")).expanduser()
    output_root = info.get("output_root")
    parc_path = Path(info["parcellation_path"])
    label_ids_csv = ",".join(str(lid) for lid, _ in selected)
    region_names_csv = ",".join(name for _, name in selected)

    node = getattr(owner, "_parameterNode", None)
    ext_paths = owner._environment_dependency_path_kwargs() if hasattr(owner, "_environment_dependency_path_kwargs") else {}
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_roi_tractography.py"
    args = [
        "--launch", "PythonSlicer", "-u", str(script),
        "--subject-id", subject_id,
        "--subject-root", str(subject_root),
        "--parcellation", str(parc_path),
        "--algorithm", alg,
        "--label-ids", label_ids_csv,
        "--region-names", region_names_csv,
        "--device", str(getattr(node, "ddsegDevice", "auto") or "auto"),
    ]
    if output_root:
        args.extend(["--output-root", str(output_root)])
    if node is not None:
        args.extend(["--select-streamlines", str(min(20000, max(1000, int(getattr(node, "orchestrationSelectStreamlines", 10000) or 10000))))])
        if getattr(node, "ukfParameters", ""):
            args.extend(["--ukf-parameters", str(node.ukfParameters)])
        if getattr(node, "csdActParameters", ""):
            args.extend(["--csd-act-parameters", str(node.csdActParameters)])
        if getattr(node, "dtiParameters", ""):
            args.extend(["--dti-parameters", str(node.dtiParameters)])
        if getattr(node, "safParameters", ""):
            args.extend(["--saf-parameters", str(node.safParameters)])
    if ext_paths.get("mrtrix_bin_dir"):
        args.extend(["--mrtrix-bin-dir", str(ext_paths["mrtrix_bin_dir"])])
    if alg == "ukf":
        try:
            from tractography.ukf import resolve_executable

            ukf_bin = resolve_executable()
            args.extend(["--ukf-executable", str(ukf_bin)])
            dwc_bin = resolve_executable(name="DWIConvert")
            args.extend(["--dwiconvert-executable", str(dwc_bin)])
        except Exception:
            pass

    launcher = owner._slicer_launcher_program()
    program, launch_args = normalize_slicer_python_launch(launcher, args)

    process = qt.QProcess(getattr(owner, "parent", None))
    process.setWorkingDirectory(Path(__file__).resolve().parents[1].as_posix())
    process.setProcessChannelMode(qt.QProcess.MergedChannels)
    env = qt.QProcessEnvironment.systemEnvironment()
    for k, v in background_process_env_overrides().items():
        env.insert(k, v)
    process.setProcessEnvironment(env)

    state["process"] = process
    state["output_buf"] = ""
    _sync_region_display(owner)
    owner._setPipelineActive(True)
    set_run_state(owner, "running")
    stage_title = (
        f"03 · ROI 纤维追踪 ({_algorithm_label(alg, True)})" if zh
        else f"03 · ROI Tracking ({_algorithm_label(alg, False)})"
    )
    set_live_stage_text(owner, stage_title, region_names_csv)
    widgets["hint"].setText(
        f"正在运行 {_algorithm_label(alg, True)} ROI 纤维追踪：{region_names_csv}…" if zh
        else f"Running {_algorithm_label(alg, False)} ROI tractography for {region_names_csv}…"
    )
    owner.appendLog(f"[ROI Tracking] Starting {alg} ROI tracking for {region_names_csv}")

    def _on_ready_read() -> None:
        chunk = owner._qbytearray_to_text(process.readAllStandardOutput())
        if not chunk:
            return
        state["output_buf"] += chunk
        for line in chunk.splitlines():
            clean = line.strip()
            if clean:
                owner.appendLog(clean)
                if not clean.startswith("ROI_MANIFEST:"):
                    set_live_stage_text(owner, stage_title, clean.replace("[ROI Tracking] ", "", 1)[:96])

    def _on_finished(exit_code: int, _exit_status=None) -> None:
        _on_ready_read()
        state["process"] = None
        stop_run_animation(owner)
        owner._setPipelineActive(False, preserve_status=True)
        _sync_region_display(owner)

        manifest_file = None
        for line in reversed(state.get("output_buf", "").splitlines()):
            s = line.strip()
            if s.startswith("ROI_MANIFEST:"):
                cand = Path(s[len("ROI_MANIFEST:"):].strip())
                if cand.is_file():
                    manifest_file = cand
                    break

        if exit_code == 0 and manifest_file is not None:
            try:
                payload = json.loads(manifest_file.read_text(encoding="utf-8"))
                vtk_path = Path(str((payload.get("outputs") or {}).get("tractogram_vtk") or ""))
                count = int(payload.get("streamline_count") or 0)
                set_run_state(owner, "completed")
                set_live_stage_text(
                    owner,
                    stage_title,
                    (f"完成（{count} 条纤维）" if zh else f"Completed ({count} streamlines)"),
                )
                widgets["hint"].setText(
                    f"ROI 追踪完成：{region_names_csv}（{count} 条纤维），已加载至视图并归档于 Results → Tractography → ROI 纤维追踪。" if zh
                    else f"ROI tracking done: {region_names_csv} ({count} streamlines), loaded in scene under Results → Tractography → ROI tracking."
                )
                if vtk_path.is_file() and hasattr(owner, "logic") and owner.logic is not None:
                    owner.logic.loadKeyResultsFromManifest(manifest_file, source_path=vtk_path, mode="replace")
                    if hasattr(slicer.util, "resetThreeDViews"):
                        slicer.util.resetThreeDViews()
                if hasattr(owner, "_refresh_run_history"):
                    owner._refresh_run_history()
            except Exception as exc:
                set_run_state(owner, "failed")
                widgets["hint"].setText(f"加载 ROI 结果失败：{exc}" if zh else f"Failed to load ROI result: {exc}")
        else:
            set_run_state(owner, "failed")
            tail = [ln.strip() for ln in state.get("output_buf", "").splitlines() if ln.strip()][-3:]
            err_msg = " | ".join(tail) or f"exit code {exit_code}"
            widgets["hint"].setText(
                f"ROI 追踪失败：{err_msg}" if zh else f"ROI tracking failed: {err_msg}"
            )
        process.deleteLater()

    process.connect("readyReadStandardOutput()", _on_ready_read)
    process.connect("finished(int,QProcess::ExitStatus)", _on_finished)
    process.start(program, launch_args)
    if not process.waitForStarted(5000):
        state["process"] = None
        owner._setPipelineActive(False)
        set_run_state(owner, "failed")
        _sync_region_display(owner)
        widgets["hint"].setText("无法启动 ROI 纤维追踪后台进程。" if zh else "Could not start ROI tracking worker.")
