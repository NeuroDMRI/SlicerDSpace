"""Advanced-only regional analysis using completed, explicitly selected run products."""

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import tempfile
import uuid


TEXT = {
    "en": dict(
        title="Regional analysis (optional)",
        source="Source run manifest",
        latest="Use latest run",
        atlas="Atlas override (optional)",
        lut="Custom atlas LUT (optional)",
        inspect="Load regions and available results",
        search="Search name, hemisphere or label ID",
        select="Select visible",
        clear="Clear selection",
        metrics="Microstructure statistics",
        tissue="Tissue composition",
        surface="Cortical morphology",
        tracts="ROI streamline selection",
        graph="Regional connectivity",
        output="Output folder (optional)",
        run="Analyze selected regions",
        empty="Select a completed subject run. No models are refitted.",
        options="Options and input files",
        tract_source="Tractogram",
        rule="Selection rule",
        graph_source="Connectome",
        target="Target regions (A ↔ B)",
        include="Required waypoints (optional)",
        exclude="Excluded regions (optional)",
        selectors="Names or IDs, separated by commas",
        report="Open regional report",
        rules=[
            "Pass through any selected region",
            "Pass through all selected regions",
            "At least one endpoint in selection",
            "Both endpoints in selection",
            "Connect selected regions A ↔ target regions B",
        ],
        note="Computes regional statistics from full quantitative maps; ROI tracks are geometric subsets.",
    ),
    "zh": dict(
        title="脑区分析（可选）",
        source="来源运行记录",
        latest="使用最近一次运行",
        atlas="替代分区图（可选）",
        lut="自定义分区 LUT（可选）",
        inspect="读取脑区与可用结果",
        search="搜索脑区名称、左右半球或标签 ID",
        select="勾选可见脑区",
        clear="清空选择",
        metrics="微结构统计",
        tissue="组织构成",
        surface="皮层形态",
        tracts="选区相关流线",
        graph="脑区连接分析",
        output="输出文件夹（可选）",
        run="分析所选脑区",
        empty="选择已完成的单被试运行记录；不会重新拟合模型。",
        options="更多选项与输入文件",
        tract_source="流线来源",
        rule="筛选规则",
        graph_source="连接组来源",
        target="目标脑区（A ↔ B）",
        include="必须途经的脑区（可选）",
        exclude="排除的脑区（可选）",
        selectors="输入名称或标签号，多个用逗号分隔",
        report="打开脑区分析报告",
        rules=["经过任一所选脑区", "经过全部所选脑区", "至少一个端点在选区内", "两个端点都在选区内", "连接所选脑区 A ↔ 目标脑区 B"],
        note="基于完整定量结果计算脑区统计；选区流线为几何筛选子集。",
    ),
}


def install_region_panel(owner):
    import ctk
    import qt

    panel = ctk.ctkCollapsibleButton(owner._advancedSettingsPanel)
    panel.objectName = "regionalAnalysisPanel"
    panel.collapsed = True
    panel.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Maximum)
    panel.setMinimumWidth(0)
    layout = qt.QVBoxLayout(panel)
    controls = dict(panel=panel, labels={})
    button = qt.QPushButton()
    controls["inspect"] = button
    layout.addWidget(button)
    search = qt.QLineEdit()
    controls["search"] = search
    layout.addWidget(search)
    regions = qt.QListWidget()
    regions.objectName = "regionalLabelsList"
    regions.setMaximumHeight(140)
    regions.setMinimumHeight(80)
    controls["regions"] = regions
    layout.addWidget(regions)
    row = qt.QHBoxLayout()
    for key in ("select", "clear"):
        button = qt.QPushButton()
        controls[key] = button
        row.addWidget(button)
    layout.addLayout(row)
    selected_summary = qt.QLabel()
    selected_summary.wordWrap = True
    controls["selection_summary"] = selected_summary
    layout.addWidget(selected_summary)
    for key in ("metrics", "tissue", "surface", "tracts", "graph"):
        box = qt.QCheckBox()
        box.enabled = False
        box.visible = False
        controls[key] = box
        layout.addWidget(box)
    options = ctk.ctkCollapsibleButton()
    options.objectName = "regionalAnalysisOptions"
    options.collapsed = True
    options.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Maximum)
    options.setMinimumWidth(0)
    controls["options"] = options
    layout.addWidget(options)
    details = qt.QVBoxLayout(options)
    form = qt.QFormLayout()
    form.setFieldGrowthPolicy(qt.QFormLayout.AllNonFixedFieldsGrow)
    details.addLayout(form)
    for key in ("source", "atlas", "lut", "output"):
        path = ctk.ctkPathLineEdit()
        path.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
        path.filters = ctk.ctkPathLineEdit.Dirs if key == "output" else ctk.ctkPathLineEdit.Files
        label = qt.QLabel()
        controls[key] = path
        controls["labels"][key] = label
        form.addRow(label, path)
    controls["latest"] = qt.QPushButton()
    details.addWidget(controls["latest"])
    metric_list = qt.QListWidget()
    metric_list.objectName = "regionalMetricsList"
    metric_list.setMaximumHeight(100)
    metric_list.enabled = False
    controls["metric_list"] = metric_list
    details.addWidget(metric_list)
    sources_form = qt.QFormLayout()
    sources_form.setFieldGrowthPolicy(qt.QFormLayout.AllNonFixedFieldsGrow)
    details.addLayout(sources_form)
    for key in ("tract_source", "rule", "graph_source"):
        combo = qt.QComboBox()
        controls[key] = combo
        combo.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
        label = qt.QLabel()
        controls["labels"][key] = label
        sources_form.addRow(label, combo)
    for key in ("through_any", "through_all", "endpoint_any", "endpoint_both", "endpoint_pair"):
        controls["rule"].addItem(key, key)
    region_form = qt.QFormLayout()
    region_form.setFieldGrowthPolicy(qt.QFormLayout.AllNonFixedFieldsGrow)
    details.addLayout(region_form)
    for key in ("target", "include", "exclude"):
        edit, label = qt.QLineEdit(), qt.QLabel()
        edit.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Fixed)
        controls[key], controls["labels"][key] = edit, label
        region_form.addRow(label, edit)
    note = qt.QLabel()
    note.wordWrap = True
    controls["note"] = note
    details.addWidget(note)
    button = qt.QPushButton()
    button.objectName = "runRegionalAnalysisButton"
    button.enabled = False
    controls["run"] = button
    layout.addWidget(button)
    status = qt.QLabel()
    status.wordWrap = True
    controls["status"] = status
    layout.addWidget(status)
    controls["report"] = qt.QPushButton()
    controls["report"].enabled = False
    controls["report"].visible = False
    layout.addWidget(controls["report"])
    owner._regionWidgets = controls
    owner._regionCatalog = None
    owner._advancedSettingsPanel.layout().insertWidget(3, panel)
    controls["latest"].clicked.connect(lambda: use_latest_run(owner))
    controls["inspect"].clicked.connect(lambda: inspect_regions(owner))
    controls["run"].clicked.connect(lambda: analyze_regions(owner))
    controls["select"].clicked.connect(lambda: set_checks(owner, True))
    controls["clear"].clicked.connect(lambda: set_checks(owner, False))
    controls["search"].textChanged.connect(lambda value: filter_regions(owner, value))
    controls["search"].returnPressed.connect(lambda: set_checks(owner, True))
    controls["regions"].itemChanged.connect(lambda *_: _selection_changed(owner))
    controls["metric_list"].itemChanged.connect(lambda *_: _selection_changed(owner))
    for key in ("source", "atlas", "lut"):
        controls[key].currentPathChanged.connect(lambda *_: invalidate_catalog(owner))
    controls["metrics"].toggled.connect(lambda enabled: metric_list.setEnabled(enabled))
    controls["tracts"].toggled.connect(lambda enabled: _tract_enabled(owner, enabled))
    controls["graph"].toggled.connect(lambda enabled: controls["graph_source"].setEnabled(enabled))
    controls["rule"].currentIndexChanged.connect(lambda *_: _tract_enabled(owner, controls["tracts"].checked))
    for key in ("metrics", "tissue", "surface", "tracts", "graph"):
        controls[key].toggled.connect(lambda *_: _selection_changed(owner))
    controls["report"].clicked.connect(lambda: open_report(owner))
    panel.connect("contentsCollapsed(bool)", lambda collapsed: _expanded(owner, collapsed))
    apply_region_language(owner, getattr(owner, "_onboardingLanguage", "en"))
    _tract_enabled(owner, False)
    controls["graph_source"].enabled = False
    _selection_changed(owner)


def _count(widget):
    value = widget.count
    return value() if callable(value) else int(value)


def _paths(owner):
    return tuple(str(owner._regionWidgets[key].currentPath or "").strip() for key in ("source", "atlas", "lut"))


def _status(owner, text, error=False):
    from shared.ui_styles import is_dark_mode

    owner._regionStatusError = bool(error)
    label = owner._regionWidgets["status"]
    label.text = text
    dark = is_dark_mode(getattr(owner, "_uiWidget", None) or label)
    if dark:
        label.setStyleSheet("color: #FF7B72;" if error else "color: #B8C5D0;")
    else:
        label.setStyleSheet("color: #B3261E;" if error else "color: #52616B;")


def refresh_region_styles(owner, *, dark=None):
    widgets = getattr(owner, "_regionWidgets", None) or {}
    label = widgets.get("status")
    if label is None:
        return
    from shared.ui_styles import is_dark_mode

    use_dark = is_dark_mode(getattr(owner, "_uiWidget", None) or label) if dark is None else bool(dark)
    error = bool(getattr(owner, "_regionStatusError", False))
    if use_dark:
        label.setStyleSheet("color: #FF7B72;" if error else "color: #B8C5D0;")
    else:
        label.setStyleSheet("color: #B3261E;" if error else "color: #52616B;")


def _expanded(owner, collapsed):
    import qt

    if not collapsed and not _paths(owner)[0]:
        use_latest_run(owner)
    if not collapsed and _paths(owner)[0] and not owner._regionCatalog and not getattr(owner, "_pipelineActive", False):
        qt.QTimer.singleShot(0, lambda: inspect_regions(owner))
    fit = getattr(owner, "_fitAdvancedPage", None) or getattr(owner, "_fitCompactHeight", None)
    if callable(fit):
        fit()


def _tract_enabled(owner, enabled):
    widgets = owner._regionWidgets
    for key in ("tract_source", "rule", "include", "exclude"):
        widgets[key].enabled = enabled
    pair = str(widgets["rule"].itemData(widgets["rule"].currentIndex)) == "endpoint_pair"
    widgets["target"].enabled = enabled and pair
    vis_changed = widgets["target"].visible != pair
    widgets["target"].setVisible(pair)
    widgets["labels"]["target"].setVisible(pair)
    if vis_changed and hasattr(owner, "_fitAdvancedPage"):
        owner._fitAdvancedPage()


def _selection_changed(owner):
    widgets = owner._regionWidgets
    count = len(_checked(widgets["regions"]))
    chinese = str(getattr(owner, "_onboardingLanguage", "en")).startswith("zh")
    widgets["selection_summary"].text = f"已选择 {count} 个脑区" if chinese else f"{count} regions selected"
    available = any(widgets[key].enabled and widgets[key].checked for key in ("metrics", "tissue", "surface", "tracts", "graph"))
    widgets["run"].enabled = bool(owner._regionCatalog and count and available)


def open_report(owner):
    import qt

    path = getattr(owner, "_regionalReportPath", None)
    if path and Path(path).is_file():
        qt.QDesktopServices.openUrl(qt.QUrl.fromLocalFile(str(path)))


def invalidate_catalog(owner):
    owner._regionCatalog = None
    owner._regionWidgets["run"].enabled = False
    owner._regionalReportPath = None
    owner._regionWidgets["report"].enabled = False
    owner._regionWidgets["report"].visible = False
    for key in ("metrics", "tissue", "surface", "tracts", "graph"):
        owner._regionWidgets[key].checked = False
        owner._regionWidgets[key].enabled = False
    _status(owner, "Inputs changed; reload regions / 输入已改变，请重新读取脑区")


def use_latest_run(owner):
    path = getattr(owner, "_latestResultManifestPath", None)
    if path:
        owner._regionWidgets["source"].setCurrentPath(str(path))
    else:
        _status(owner, "Choose a completed single-subject manifest / 请选择已完成的单被试运行记录")


def apply_region_language(owner, language):
    if not hasattr(owner, "_regionWidgets"):
        return
    widgets = owner._regionWidgets
    strings = TEXT["zh" if str(language).startswith("zh") else "en"]
    widgets["panel"].text = strings["title"]
    for key, label in widgets["labels"].items():
        label.text = strings[key]
    widgets["options"].text = strings["options"]
    for key in ("latest", "inspect", "select", "clear", "metrics", "tissue", "surface", "tracts", "graph", "run", "note", "report"):
        widgets[key].text = strings[key]
    widgets["search"].placeholderText = strings["search"]
    for key in ("target", "include", "exclude"):
        widgets[key].placeholderText = strings["selectors"]
    for index, name in enumerate(strings["rules"]):
        widgets["rule"].setItemText(index, name)
    if not widgets["status"].text:
        widgets["status"].text = strings["empty"]


def filter_regions(owner, query):
    import qt
    from shared.region_selection import matches_region

    widgets = owner._regionWidgets
    regions = {row["id"]: row for row in (owner._regionCatalog or {}).get("regions", [])}
    for index in range(_count(widgets["regions"])):
        item = widgets["regions"].item(index)
        item.setHidden(not matches_region(regions.get(int(item.data(qt.Qt.UserRole)), {"id": -1, "name": item.text()}), query))


def set_checks(owner, checked):
    import qt

    widget = owner._regionWidgets["regions"]
    blocked = widget.blockSignals(True)
    try:
        for index in range(_count(widget)):
            item = widget.item(index)
            if not checked or not item.isHidden():
                item.setCheckState(qt.Qt.Checked if checked else qt.Qt.Unchecked)
    finally:
        widget.blockSignals(blocked)
    _selection_changed(owner)


def _command(owner, args):
    script = Path(__file__).resolve().parents[1] / "scripts/analyze_regions.py"
    return [(owner._slicer_launcher_program(), ["--launch", "PythonSlicer", "-u", str(script), *args], {})]


def inspect_regions(owner):
    from shared.ui_setup_jobs import start_setup_job
    import qt
    import slicer

    widgets = owner._regionWidgets
    if getattr(owner, "_pipelineActive", False):
        _status(owner, "Wait for the current job / 请等待当前任务完成", True)
        return
    source, atlas, lut = _paths(owner)
    if not source:
        use_latest_run(owner)
        source, atlas, lut = _paths(owner)
    if not source:
        return
    catalog = Path(tempfile.mkdtemp(prefix="regions-", dir=str(slicer.app.temporaryPath))) / "catalog.json"
    args = ["--inspect", source, "--catalog", str(catalog)]
    if atlas:
        args.extend(("--atlas", atlas))
    if lut:
        args.extend(("--lut", lut))
    requested = source, atlas, lut

    def done(status, details, log_path):
        widgets["panel"].enabled = True
        if status != "completed":
            _status(owner, f"{status}: {details[-800:]}\nLog: {log_path}", True)
            return
        if requested != _paths(owner):
            invalidate_catalog(owner)
            return
        try:
            data = json.loads(catalog.read_text())
            widgets["regions"].clear()
            widgets["metric_list"].clear()
            for region in data["regions"]:
                item = qt.QListWidgetItem(f'{region["id"]}  {region["name"]}  ({region["voxels"]} voxels)')
                item.setFlags(qt.Qt.ItemIsEnabled | qt.Qt.ItemIsSelectable | qt.Qt.ItemIsUserCheckable)
                item.setData(qt.Qt.UserRole, region["id"])
                item.setCheckState(qt.Qt.Unchecked)
                widgets["regions"].addItem(item)
            for key in data["metrics"]:
                item = qt.QListWidgetItem(key)
                item.setFlags(qt.Qt.ItemIsEnabled | qt.Qt.ItemIsUserCheckable)
                item.setData(qt.Qt.UserRole, key)
                item.setCheckState(qt.Qt.Checked)
                widgets["metric_list"].addItem(item)
            for kind, combo in (("tractograms", "tract_source"), ("connectomes", "graph_source")):
                widgets[combo].clear()
                for key in data[kind]:
                    widgets[combo].addItem(key, key)
            for key, available in (
                ("metrics", bool(data["metrics"])),
                ("tissue", bool(data.get("tissue"))),
                ("surface", bool(data.get("surfaces"))),
                ("tracts", bool(data["tractograms"])),
                ("graph", bool(data["connectomes"])),
            ):
                widgets[key].enabled = available
                widgets[key].visible = available
                widgets[key].checked = available if key == "metrics" else False
                widgets[key].toolTip = "" if available else (
                    "Complete the corresponding analysis with valid QC first / 请先完成相应算法并取得可用结果"
                )
            owner._regionCatalog = data
            _selection_changed(owner)
            widgets["atlas"].toolTip = data["atlas"]
            chinese = str(getattr(owner, "_onboardingLanguage", "en")).startswith("zh")
            _status(owner, "结果已读取，勾选脑区后即可分析。" if chinese else "Results loaded. Select regions to analyze.")
            widgets["status"].toolTip = (
                f'{len(data["regions"])} regions; {len(data["metrics"])} metrics; '
                f'{len(data["tractograms"])} tractograms; {len(data["connectomes"])} connectomes; '
                f'{len(data["warnings"])} input/QC notices (see log).'
            )
            for warning in data["warnings"]:
                owner.appendLog("[Regions] Excluded/not available: " + warning)
            filter_regions(owner, widgets["search"].text)
            if hasattr(owner, "_fitAdvancedPage"):
                owner._fitAdvancedPage()
        except Exception as exc:
            invalidate_catalog(owner)
            _status(owner, str(exc), True)

    try:
        if start_setup_job(owner, "Inspect regional inputs / 读取脑区", _command(owner, args), done):
            widgets["panel"].enabled = False
    except Exception as exc:
        widgets["panel"].enabled = True
        _status(owner, str(exc), True)


def _checked(widget):
    import qt

    return [
        widget.item(index).data(qt.Qt.UserRole)
        for index in range(_count(widget))
        if widget.item(index).checkState() == qt.Qt.Checked
    ]


def analyze_regions(owner):
    from shared.ui_setup_jobs import start_setup_job
    from shared.region_analysis import write_json

    widgets, catalog = owner._regionWidgets, owner._regionCatalog
    if getattr(owner, "_pipelineActive", False):
        _status(owner, "Wait for the current job / 请等待当前任务完成", True)
        return
    if not catalog:
        _status(owner, "Reload regions first / 请先读取脑区", True)
        return
    selected = _checked(widgets["regions"])
    if not selected:
        _status(owner, "Select at least one region / 请至少选择一个脑区", True)
        return
    source, atlas, lut = _paths(owner)
    root = str(widgets["output"].currentPath or "").strip()
    root = Path(root) if root else Path(catalog["output_root"]) / "05_statistics/selected_regions"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_name = re.sub(r"[^A-Za-z0-9_.-]", "_", catalog["run_name"])[:80].strip(".") or "run"
    directory = root.expanduser().resolve() / run_name / stamp
    metrics = [str(key) for key in _checked(widgets["metric_list"])] if widgets["metrics"].checked else []
    request = dict(
        source_manifest=source,
        atlas_override=atlas or None,
        lut_path=lut or None,
        manifest_sha256=catalog["manifest_sha256"],
        atlas_sha256=catalog["atlas_sha256"],
        lut_sha256=catalog.get("lut_sha256"),
        selected_labels=[int(key) for key in selected],
        metrics=metrics,
        tissue=bool(widgets["tissue"].checked),
        surface=bool(widgets["surface"].checked),
        tractogram=(
            str(widgets["tract_source"].itemData(widgets["tract_source"].currentIndex))
            if widgets["tracts"].checked
            else None
        ),
        connectome=(
            str(widgets["graph_source"].itemData(widgets["graph_source"].currentIndex))
            if widgets["graph"].checked
            else None
        ),
        tract_rule=str(widgets["rule"].itemData(widgets["rule"].currentIndex)),
        output_dir=str(directory),
    )
    if request["tractogram"]:
        from shared.region_selection import resolve_regions

        try:
            for field in ("target", "include", "exclude"):
                text = str(widgets[field].text or "") if field != "target" or request["tract_rule"] == "endpoint_pair" else ""
                request[field + "_labels"] = resolve_regions(catalog["regions"], text, allow_empty=True)
            if request["tract_rule"] == "endpoint_pair" and (
                not request["target_labels"] or set(selected) & set(request["target_labels"])
            ):
                raise ValueError("Choose target regions B different from selected regions A / 请选择与 A 不重叠的目标脑区 B")
        except ValueError as exc:
            _status(owner, str(exc), True)
            return
    if not metrics and not request["tractogram"] and not request["connectome"] and not request["tissue"] and not request["surface"]:
        _status(owner, "Select an available analysis / 请启用至少一种可用分析", True)
        return
    try:
        directory.mkdir(parents=True)
        request_path = directory / "request.json"
        write_json(request_path, request)

        def done(status, details, log_path):
            widgets["panel"].enabled = True
            if status != "completed":
                _status(owner, f"{status}: {details[-800:]}\nLog: {log_path}", True)
                return
            try:
                manifest = directory / "region_analysis_manifest.json"
                record = json.loads(manifest.read_text())
                if record.get("status") != "completed":
                    raise RuntimeError("Regional job did not publish completed outputs")
                loaded = owner.logic.loadKeyResultsFromManifest(manifest, automatic=True)
                owner._latestRegionalManifestPath = manifest
                owner._regionalReportPath = record.get("report_path")
                widgets["report"].enabled = bool(owner._regionalReportPath and Path(owner._regionalReportPath).is_file())
                widgets["report"].visible = widgets["report"].enabled
                message = f'Regional analysis completed. Loaded {len(loaded["loaded"])}; warnings: {len(record["warnings"])}. '
                _status(owner, message + "Visual review pending.\n" + str(directory))
                owner.appendLog("[Regions] " + message + str(manifest))
                for warning in record["warnings"]:
                    owner.appendLog("[Regions] " + warning)
                for skipped in loaded["skipped"]:
                    owner.appendLog(
                        "[Regions] Not loaded: " + str(skipped.get("path", "")) + ": " + str(skipped.get("reason", ""))
                    )
                if loaded["skipped"]:
                    _status(
                        owner,
                        message + f'{len(loaded["skipped"])} display items unavailable; see log.\n' + str(directory),
                        True,
                    )
                owner._setStatus("warning", "Regional results loaded; review coverage and anatomy")
            except Exception as exc:
                _status(owner, str(exc), True)

        if start_setup_job(
            owner, "Selected-region analysis / 分析所选脑区", _command(owner, ["--request", str(request_path)]), done
        ):
            widgets["panel"].enabled = False
    except Exception as exc:
        widgets["panel"].enabled = True
        _status(owner, str(exc), True)
