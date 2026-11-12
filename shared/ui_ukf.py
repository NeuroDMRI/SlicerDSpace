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


def add_staged_ukf_fields(layout, node, zh):
    import qt

    controls = {}
    for key, en, chinese, minimum, maximum, step in CONTROLS:
        if key == "freeWater":
            control = qt.QCheckBox()
        else:
            control = qt.QSpinBox() if isinstance(minimum, int) else qt.QDoubleSpinBox()
            control.setRange(minimum, maximum)
            control.setSingleStep(step)
            if not isinstance(minimum, int):
                control.setDecimals(5 if key in {"Qm", "Qw", "Rs"} else 3)
        control.setObjectName("ukf_" + key)
        layout.addRow(chinese if zh else en, control)
        controls[key] = control

    saved = {}

    def show_values(values):
        validated = asdict(parameters(values))
        saved.clear()
        saved.update({key: value for key, value in values.items() if key not in controls})
        for key, control in controls.items():
            value = validated[key]
            if key == "freeWater":
                control.setChecked(value)
            else:
                control.setValue(0 if key == "numThreads" and value == -1 else value)

    def read_values():
        values = {key: control.checked if key == "freeWater" else control.value
                  for key, control in controls.items()}
        if values["numThreads"] == 0:
            values["numThreads"] = -1
        values.update(saved)
        parameters(values)
        return json.dumps(values, sort_keys=True)

    original = node.ukfParameters or "{}"
    show_values(json.loads(original))
    initial = read_values()

    def staged_value():
        value = read_values()
        return original if value == initial else value
    reset = qt.QPushButton("恢复作者默认值" if zh else "Restore author defaults")
    reset.clicked.connect(lambda: show_values({}))
    layout.addRow(reset)
    return staged_value
