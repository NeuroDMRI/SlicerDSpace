"""Optional UKF controls; scientific defaults are shared with the CLI backend."""

import json
from dataclasses import asdict

from tractography.ukf import parameters


CONTROLS = (
    ("numTensor", "Tensor model", "张量数", 1, 2, 1),
    ("seedsPerVoxel", "Seeds per voxel", "每体素种子数", 1, 50, 1),
    ("seedingThreshold", "Minimum seed FA", "播种 FA 下限", 0.0, 1.0, 0.01),
    ("stoppingFA", "Stopping FA", "追踪终止 FA", 0.0, 1.0, 0.01),
    ("stoppingThreshold", "Stopping mean signal", "终止平均信号", 0.0, 1.0, 0.01),
    ("stepLength", "Tracking step (mm)", "追踪步长 (mm)", 0.1, 1.0, 0.1),
    ("recordLength", "Output spacing (mm)", "输出点间距 (mm)", 0.1, 4.0, 0.1),
    ("maxHalfFiberLength", "Maximum HALF length (mm)", "最大半束长度 (mm)", 1.0, 500.0, 10.0),
    ("numThreads", "CPU threads (0 = automatic)", "CPU 线程 (0 = 自动)", 0, 4096, 1),
    ("freeWater", "Estimate free-water term", "自由水项", None, None, None),
    ("Qm", "Direction noise (0 = author auto)", "方向噪声 (0 = 作者自动值)", 0.0, 0.25, 0.001),
    ("Ql", "Eigenvalue noise (0 = author auto)", "特征值噪声 (0 = 作者自动值)", 0.0, 1000.0, 1.0),
    ("Qw", "Free-water noise (0 = author auto)", "自由水噪声 (0 = 作者自动值)", 0.0, 0.25, 0.001),
    ("Rs", "Measurement noise (0 = author auto)", "测量噪声 (0 = 作者自动值)", 0.0, 0.25, 0.001),
)


def refresh_ukf_controls(owner, backend):
    import ctk
    import qt

    node = owner._parameterNode
    if not hasattr(owner, "_ukfControls"):
        group = ctk.ctkCollapsibleGroupBox()
        group.objectName = "ukfParametersGroup"
        group.collapsed = True
        group.setSizePolicy(qt.QSizePolicy.Ignored, qt.QSizePolicy.Preferred)
        group.setMinimumWidth(0)
        layout = qt.QFormLayout(group)
        layout.setRowWrapPolicy(qt.QFormLayout.WrapLongRows)
        layout.setFieldGrowthPolicy(qt.QFormLayout.AllNonFixedFieldsGrow)
        controls, labels = {}, {}
        owner._ukfUpdating = True

        def changed(*_):
            if owner._ukfUpdating or not owner._parameterNode:
                return
            values = {
                key: (bool(control.checked) if key == "freeWater" else control.value)
                for key, control in controls.items()
            }
            if values["numThreads"] == 0:
                values["numThreads"] = -1
            parameters(values)
            owner._update_parameter_node("ukfParameters", json.dumps(values, sort_keys=True))

        for key, en, zh, minimum, maximum, step in CONTROLS:
            if key == "freeWater":
                control = qt.QCheckBox()
                control.toggled.connect(changed)
            else:
                control = qt.QSpinBox() if isinstance(minimum, int) else qt.QDoubleSpinBox()
                control.setRange(minimum, maximum)
                control.setSingleStep(step)
                if not isinstance(minimum, int):
                    control.decimals = 5 if key in {"Qm", "Qw", "Rs"} else 3
                control.valueChanged.connect(changed)
            control.objectName = "ukf_" + key
            label = qt.QLabel(en)
            layout.addRow(label, control)
            controls[key], labels[key] = control, label
        note = qt.QLabel(
            "Author-module defaults, not a validated WMA sampling protocol. "
            "No target streamline count; all supplied shells. Anatomical review remains required."
        )
        note.wordWrap = True
        layout.addRow(note)
        reset = qt.QPushButton("Restore author defaults")
        reset.clicked.connect(lambda: owner._update_parameter_node("ukfParameters", "{}"))
        layout.addRow(reset)
        owner.ui.parametersCollapsibleButton.layout().insertWidget(1, group)
        owner._ukfControls, owner._ukfLabels = controls, labels
        owner._ukfGroup, owner._ukfNote, owner._ukfReset = group, note, reset
        if hasattr(owner, '_fitAdvancedPage'):
            group.toggled.connect(owner._fitAdvancedPage)

    from shared.algorithm_selection import contains_tracking
    target_visible = contains_tracking(backend, "ukf") and bool(
        node and (node.targetTractography or node.targetBundles or node.targetConnectome)
    )
    if owner._ukfGroup.visible != target_visible:
        owner._ukfGroup.visible = target_visible
    if hasattr(owner, '_advancedCategoryStack') and owner._ukfGroup.parent() != owner._advancedCategoryStack.widget(1):
        owner.ui.parametersLayout.removeWidget(owner._ukfGroup)
        owner._advancedCategoryStack.widget(1).layout().addWidget(owner._ukfGroup)
    zh = getattr(owner, "_onboardingLanguage", "en").startswith("zh")
    target_title = "UKF 参数 · 作者默认值" if zh else "UKF parameters · author defaults"
    if owner._ukfGroup.title != target_title:
        owner._ukfGroup.title = target_title
    target_reset = "恢复作者默认值" if zh else "Restore author defaults"
    if owner._ukfReset.text != target_reset:
        owner._ukfReset.text = target_reset
    for key, en, chinese, *_ in CONTROLS:
        target_label = chinese if zh else en
        if owner._ukfLabels[key].text != target_label:
            owner._ukfLabels[key].text = target_label
    owner._ukfUpdating = True
    try:
        values = asdict(parameters(json.loads(getattr(node, "ukfParameters", "{}") or "{}")))
        for key, control in owner._ukfControls.items():
            if key == "freeWater":
                if control.checked != values[key]:
                    control.checked = values[key]
            else:
                target_val = 0 if key == "numThreads" and values[key] == -1 else values[key]
                if abs(float(control.value) - float(target_val)) > 1e-9:
                    control.value = target_val
    finally:
        owner._ukfUpdating = False
