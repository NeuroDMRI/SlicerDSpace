#!/usr/bin/env python3
"""Tests for live run progress, rotating spinner, out-of-process input import, and QC surfacing."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.run_history import ProgressSnapshot
from shared.ui_run_state import (
    RUN_STATES,
    format_elapsed_mmss,
    parse_live_stage_from_log_line,
    run_state_from_manifest_status,
)
from shared.ui_styles import (
    DARK_RUN_STAGE_INDICATOR_PALETTE,
    RUN_STAGE_INDICATOR_PALETTE,
    run_stage_indicator_palette,
)


class _Signal:
    def __init__(self) -> None:
        self._callbacks = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)

    def disconnect(self, callback) -> None:
        self._callbacks = [cb for cb in self._callbacks if cb is not callback]

    def emit(self, *args, **kwargs) -> None:
        for callback in list(self._callbacks):
            callback(*args, **kwargs)


class _FakeWidget:
    def __init__(self, *args, **_kwargs) -> None:
        self._text = str(args[0]) if args else ""
        self._object_name = ""
        self._tooltip = ""
        self._stylesheet = ""
        self._visible = True
        self._enabled = True
        self._layout = None
        self._pixmap = None
        self._range = (0, 1)
        self._value = 0
        self._format = ""
        self._interval = 0
        self._active = False
        self.wordWrap = False
        self.timeout = _Signal()
        self._signals: dict[str, _Signal] = {}
        self.topLevelItems: list[Any] = []
        self.clicked = _Signal()
        self.itemClicked = _Signal()
        self.itemDoubleClicked = _Signal()
        self.customContextMenuRequested = _Signal()

    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        self._visible = bool(value)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = bool(value)

    def setVisible(self, value: bool) -> None:
        self._visible = bool(value)

    def isVisible(self) -> bool:
        return self._visible

    def isHidden(self) -> bool:
        return not self._visible

    def show(self) -> None:
        self._visible = True

    def hide(self) -> None:
        self._visible = False

    def setEnabled(self, value: bool) -> None:
        self._enabled = bool(value)

    def setObjectName(self, name: str) -> None:
        self._object_name = name

    def objectName(self) -> str:
        return self._object_name

    def setToolTip(self, text: str) -> None:
        self._tooltip = text

    def setText(self, text: str) -> None:
        self._text = text

    def text(self) -> str:
        return self._text

    def setStyleSheet(self, style: str) -> None:
        self._stylesheet = style

    def styleSheet(self) -> str:
        return self._stylesheet

    def setFixedSize(self, *_args) -> None:
        pass

    def setFixedHeight(self, *_args) -> None:
        pass

    def setMinimumWidth(self, *_args) -> None:
        pass

    def setSizePolicy(self, *_args) -> None:
        pass

    def setTextInteractionFlags(self, *_args) -> None:
        pass

    def setContentsMargins(self, *_args) -> None:
        pass

    def setSpacing(self, *_args) -> None:
        pass

    def setPixmap(self, pixmap) -> None:
        self._pixmap = pixmap

    def setRange(self, minimum: int, maximum: int) -> None:
        self._range = (int(minimum), int(maximum))

    def minimum(self) -> int:
        return self._range[0]

    def maximum(self) -> int:
        return self._range[1]

    def setValue(self, value: int) -> None:
        self._value = int(value)

    def value(self) -> int:
        return self._value

    def setTextVisible(self, _flag: bool) -> None:
        pass

    def setFormat(self, fmt: str) -> None:
        self._format = fmt

    def format(self) -> str:
        return self._format

    def setInterval(self, ms: int) -> None:
        self._interval = int(ms)

    def interval(self) -> int:
        return self._interval

    def start(self) -> None:
        self._active = True

    def stop(self) -> None:
        self._active = False

    def isActive(self) -> bool:
        return self._active

    def connect(self, signal_name: str, callback) -> None:
        if signal_name == "timeout()":
            self.timeout.connect(callback)
        else:
            self._signals.setdefault(signal_name, _Signal()).connect(callback)

    def layout(self):
        return self._layout

    def setLayout(self, layout) -> None:
        self._layout = layout

    def clear(self) -> None:
        self.topLevelItems.clear()

    @property
    def topLevelItemCount(self) -> int:
        return len(self.topLevelItems)

    def topLevelItem(self, idx: int):
        return self.topLevelItems[idx]


class _FakeLayout:
    def __init__(self, parent=None) -> None:
        self.items: list[Any] = []
        if parent is not None and hasattr(parent, "setLayout"):
            parent.setLayout(self)

    def addWidget(self, widget, *_args) -> None:
        if widget not in self.items:
            self.items.append(widget)

    def removeWidget(self, widget) -> None:
        if widget in self.items:
            self.items.remove(widget)

    def indexOf(self, widget) -> int:
        try:
            return self.items.index(widget)
        except ValueError:
            return -1

    def addLayout(self, sublayout, *_args) -> None:
        self.items.append(sublayout)

    def addStretch(self, *_args) -> None:
        self.items.append(("stretch",))

    def setContentsMargins(self, *_args) -> None:
        pass

    def setSpacing(self, *_args) -> None:
        pass

    def setVerticalSpacing(self, *_args) -> None:
        pass

    def setHorizontalSpacing(self, *_args) -> None:
        pass

    def invalidate(self) -> None:
        pass

    def activate(self) -> None:
        pass


class _FakeTreeWidgetItem:
    def __init__(self, parent=None, labels=None) -> None:
        self._labels = list(labels or [])
        self._children: list[_FakeTreeWidgetItem] = []
        self._data: dict[tuple[int, int], Any] = {}
        self._tooltips: dict[int, str] = {}
        if isinstance(parent, _FakeWidget):
            parent.topLevelItems.append(self)
        elif isinstance(parent, _FakeTreeWidgetItem):
            parent._children.append(self)

    def text(self, col: int) -> str:
        return self._labels[col] if col < len(self._labels) else ""

    def setText(self, col: int, text: str) -> None:
        while len(self._labels) <= col:
            self._labels.append("")
        self._labels[col] = text

    def setData(self, col: int, role: int, value: Any) -> None:
        self._data[(col, role)] = value

    def data(self, col: int, role: int) -> Any:
        return self._data.get((col, role))

    def setToolTip(self, col: int, text: str) -> None:
        self._tooltips[col] = text

    def setIcon(self, *_args) -> None:
        pass

    @property
    def childCount(self):
        return lambda: len(self._children)

    def child(self, idx: int) -> _FakeTreeWidgetItem:
        return self._children[idx]


class _FakePixmap:
    def __init__(self, *args) -> None:
        self.loaded_data = None
        self.dpr = 1.0

    def loadFromData(self, data, _fmt=None) -> bool:
        self.loaded_data = bytes(data.data) if hasattr(data, "data") else bytes(data)
        return True

    def scaled(self, *_args) -> _FakePixmap:
        return self

    def setDevicePixelRatio(self, ratio: float) -> None:
        self.dpr = ratio


def _build_fake_qt_module(*, process_factory=None) -> types.ModuleType:
    qt_mod = types.ModuleType("qt")
    qt_mod.Qt = SimpleNamespace(
        AlignVCenter=0x80,
        TextSelectableByMouse=1,
        CustomContextMenu=3,
        UserRole=256,
        KeepAspectRatio=1,
        SmoothTransformation=1,
        ScrollBarAlwaysOn=2,
        ScrollBarAlwaysOff=1,
    )
    qt_mod.QSizePolicy = SimpleNamespace(Expanding=7, Fixed=0, Maximum=4, Preferred=5, Ignored=13)
    qt_mod.QFrame = SimpleNamespace(NoFrame=0)
    qt_mod.QWidget = _FakeWidget
    qt_mod.QLabel = _FakeWidget
    qt_mod.QProgressBar = _FakeWidget
    qt_mod.QPushButton = _FakeWidget
    qt_mod.QTimer = _FakeWidget
    qt_mod.QTimer.singleShot = staticmethod(lambda _ms, fn: None)
    qt_mod.QHBoxLayout = _FakeLayout
    qt_mod.QVBoxLayout = _FakeLayout
    qt_mod.QTreeWidgetItem = _FakeTreeWidgetItem
    qt_mod.QPixmap = _FakePixmap
    qt_mod.QSignalBlocker = lambda *_args: SimpleNamespace(__enter__=lambda s: s, __exit__=lambda *a: False)

    class _FakeByteArray:
        def __init__(self, data: bytes) -> None:
            self.data = data

    qt_mod.QByteArray = _FakeByteArray

    class _FakeEnv(dict):
        @classmethod
        def systemEnvironment(cls):
            return cls()

        def insert(self, key: str, value: str) -> None:
            self[key] = value

    qt_mod.QProcessEnvironment = _FakeEnv

    if process_factory is None:
        class _DefaultQProcess(_FakeWidget):
            NotRunning = 0
            Running = 2
            MergedChannels = 1

            def __init__(self, parent=None) -> None:
                super().__init__()
                self._state = 0
                self.started_calls: list[tuple[str, list[str]]] = []

            def setWorkingDirectory(self, _d: str) -> None:
                pass

            def setProcessChannelMode(self, _m: int) -> None:
                pass

            def setProcessEnvironment(self, _e) -> None:
                pass

            def start(self, program: str, args: list[str]) -> None:
                self._state = 2
                self.started_calls.append((program, list(args)))

            def state(self) -> int:
                return self._state

            def processId(self) -> int:
                return 0

            def terminate(self) -> None:
                self._state = 0

            def kill(self) -> None:
                self._state = 0

            def readAllStandardOutput(self):
                return b""

        qt_mod.QProcess = _DefaultQProcess
    else:
        qt_mod.QProcess = process_factory
    return qt_mod


def _make_run_progress_owner() -> SimpleNamespace:
    run_layout = _FakeLayout()
    status_frame = _FakeWidget()
    run_layout.addWidget(status_frame)
    ui = SimpleNamespace(runLayout=run_layout, statusFrame=status_frame)
    return SimpleNamespace(
        ui=ui,
        _uiWidget=None,
        _onboardingLanguage="en",
        _poll_run_progress=lambda: None,
        _update_run_status_table_item=lambda *_args: None,
        _set_run_status_table_rows=lambda *_args: None,
        _open_local_path=lambda *_args, **_kwargs: None,
    )


class RunStateAndParsingTests(unittest.TestCase):
    def test_run_states_include_importing_and_setup_busy(self) -> None:
        self.assertIn("importing", RUN_STATES)
        self.assertTrue(RUN_STATES["importing"].busy)
        self.assertEqual(RUN_STATES["importing"].label, "Importing inputs")
        self.assertEqual(RUN_STATES["importing"].color, "#1565C0")

        self.assertIn("setup", RUN_STATES)
        self.assertTrue(RUN_STATES["setup"].busy)
        self.assertEqual(RUN_STATES["setup"].label, "Preparing environment")
        self.assertEqual(RUN_STATES["setup"].color, "#1565C0")

        self.assertEqual(run_state_from_manifest_status("importing"), "idle")
        self.assertEqual(run_state_from_manifest_status("setup"), "idle")
        self.assertEqual(run_state_from_manifest_status("preflight"), "idle")

    def test_palettes_include_importing_and_setup(self) -> None:
        for state in ("importing", "setup"):
            self.assertIn(state, RUN_STAGE_INDICATOR_PALETTE)
            self.assertIn(state, DARK_RUN_STAGE_INDICATOR_PALETTE)
            self.assertEqual(run_stage_indicator_palette(state, dark=False)["glyph"], "run")
            self.assertEqual(run_stage_indicator_palette(state, dark=True)["glyph"], "run")

    def test_format_elapsed_mmss(self) -> None:
        self.assertEqual(format_elapsed_mmss(0), "00:00")
        self.assertEqual(format_elapsed_mmss(-5), "00:00")
        self.assertEqual(format_elapsed_mmss(8), "00:08")
        self.assertEqual(format_elapsed_mmss(84), "01:24")
        self.assertEqual(format_elapsed_mmss(3665), "01:01:05")

    def test_parse_live_stage_from_log_line_en_and_zh(self) -> None:
        cases = [
            ("[Orchestrator] Running pipeline step", "Pipeline", "流水线执行"),
            ("[Preprocessing] Eddy current correction", "00 · Preprocessing", "00 · 预处理"),
            ("[BrainMask] Running SynthStrip", "00 · Preprocessing (Brain mask)", "00 · 预处理（脑掩膜）"),
            ("[DDSeg] Segmenting diffusion tissue", "01 · Tissue segmentation (DDSeg)", "01 · 组织分割 (DDSeg)"),
            ("[DDParcel] Cortical parcellation", "01 · Brain parcellation (DDParcel)", "01 · 脑区分割 (DDParcel)"),
            ("[DDSurfer] Surface reconstruction", "01 · Cortical surface (DDSurfer)", "01 · 皮层表面 (DDSurfer)"),
            (
                "Running surface_morphometry after recon-all FreeSurfer postprocess",
                "01 · Surface morphometry",
                "01 · 表面形态学后处理",
            ),
            (
                "[Microstructure] Fitting DTI Free Water NODDI DKI iFW",
                "02 · Microstructure",
                "02 · 微结构分析",
            ),
            ("Starting UKF CSD ACT SAF tckgen tractography", "03 · Tractography", "03 · 纤维追踪"),
            ("Clustering Fiber bundles with WMQL WMA TractCloud", "04 · Fiber bundles", "04 · 纤维束提取"),
            ("[Connectome] Building tck2connectome matrix", "05 · Connectome", "05 · 连接组"),
            ("[QC] Generating summary report and manifest", "06 · QC & reports", "06 · 质控与报告"),
        ]
        for line, expected_en, expected_zh in cases:
            en = parse_live_stage_from_log_line(line, "en")
            zh = parse_live_stage_from_log_line(line, "zh")
            self.assertIsNotNone(en, msg=f"Expected match in en for: {line}")
            self.assertIsNotNone(zh, msg=f"Expected match in zh for: {line}")
            assert en is not None and zh is not None
            self.assertEqual(en[0], expected_en)
            self.assertEqual(zh[0], expected_zh)
            self.assertTrue(en[1])

        self.assertIsNone(parse_live_stage_from_log_line("", "en"))
        self.assertIsNone(parse_live_stage_from_log_line("[Background] Starting: cmd", "en"))
        self.assertIsNone(parse_live_stage_from_log_line("[Orchestrator] Plan summary: 6 tasks", "en"))


class PrepareInputsCliEndToEndTests(unittest.TestCase):
    def test_cli_prepares_nifti_bval_bvec_and_writes_manifest(self) -> None:
        import nibabel as nib
        import numpy as np

        cli_path = REPO_ROOT / "shared" / "prepare_inputs_cli.py"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            src_nii = tmp_root / "dwi_raw.nii.gz"
            src_bval = tmp_root / "dwi_raw.bval"
            src_bvec = tmp_root / "dwi_raw.bvec"
            data = np.arange(2 * 2 * 2 * 2, dtype=np.float32).reshape((2, 2, 2, 2))
            nib.save(nib.Nifti1Image(data, np.eye(4)), str(src_nii))
            src_bval.write_text("0 1000\n", encoding="utf-8")
            src_bvec.write_text("0 1\n0 0\n0 0\n", encoding="utf-8")

            subject_root = tmp_root / "subjects" / "sub-01"
            req_path = tmp_root / "req.json"
            res_path = tmp_root / "res.json"
            req_path.write_text(
                json.dumps(
                    {
                        "mode": "prepare",
                        "config": {
                            "source": str(src_nii),
                            "subject_root": str(subject_root),
                            "subject_id": "sub-01",
                            "source_kind": "nifti",
                            "bval_path": str(src_bval),
                            "bvec_path": str(src_bvec),
                        },
                    }
                ),
                encoding="utf-8",
            )

            proc = subprocess.run(
                [
                    str((Path(sys.executable).with_name("PythonSlicer.exe") if Path(sys.executable).with_name("PythonSlicer.exe").is_file() else Path(sys.executable))),
                    "-u",
                    str(cli_path),
                    "--request",
                    str(req_path),
                    "--result",
                    str(res_path),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stdout + proc.stderr)
            manifest = json.loads(res_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertTrue((subject_root / "input" / "data.nii.gz").is_file())
            self.assertTrue((subject_root / "input" / "bvals").is_file())
            self.assertTrue((subject_root / "input" / "bvecs").is_file())
            self.assertTrue((subject_root / "input" / "input_import_manifest.json").is_file())

    def test_cli_failure_writes_failed_result_and_exits_nonzero(self) -> None:
        cli_path = REPO_ROOT / "shared" / "prepare_inputs_cli.py"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            req_path = tmp_root / "req.json"
            res_path = tmp_root / "res.json"
            req_path.write_text(
                json.dumps(
                    {
                        "mode": "prepare",
                        "config": {
                            "source": str(tmp_root / "does_not_exist.nii.gz"),
                            "subject_root": str(tmp_root / "sub"),
                            "subject_id": "sub-err",
                            "source_kind": "nifti",
                        },
                    }
                ),
                encoding="utf-8",
            )
            proc = subprocess.run(
                [
                    str((Path(sys.executable).with_name("PythonSlicer.exe") if Path(sys.executable).with_name("PythonSlicer.exe").is_file() else Path(sys.executable))),
                    "-u",
                    str(cli_path),
                    "--request",
                    str(req_path),
                    "--result",
                    str(res_path),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertNotEqual(proc.returncode, 0)
            payload = json.loads(res_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "failed")
            self.assertTrue(payload["error"])


class LiveRunProgressPanelTests(unittest.TestCase):
    def test_spinner_svg_frames_and_pixmap_cache(self) -> None:
        fake_qt = _build_fake_qt_module()
        with patch.dict(sys.modules, {"qt": fake_qt}):
            from shared import ui_run_history

            frames = ui_run_history._spinner_svg_frames("#1565C0", dark=False)
            self.assertEqual(len(frames), 12)
            for idx, payload in enumerate(frames):
                self.assertIn(f'rotate({idx * 30} 8 8)'.encode("utf-8"), payload)
            pix0 = ui_run_history._stage_icon_pixmap("running", dark=False, frame=0)
            pix3 = ui_run_history._stage_icon_pixmap("running", dark=False, frame=3)
            self.assertIsNotNone(pix0)
            self.assertIsNotNone(pix3)
            self.assertIsNot(pix0, pix3)

    def test_busy_indeterminate_until_done_positive_and_timer_mmss(self) -> None:
        fake_qt = _build_fake_qt_module()
        with patch.dict(sys.modules, {"qt": fake_qt}):
            from shared import ui_run_history

            owner = _make_run_progress_owner()
            ui_run_history.install_run_progress_panel(owner)
            # statusFrame must sit after _runProgressRowWidget and _runStageLabel
            items = owner.ui.runLayout.items
            idx_strip = next(
                i for i, item in enumerate(items)
                if isinstance(item, _FakeLayout) and owner._runProgressRowWidget in item.items
            )
            idx_stage = items.index(owner._runStageLabel)
            idx_status = items.index(owner.ui.statusFrame)
            self.assertGreater(idx_stage, idx_strip)
            self.assertGreater(idx_status, idx_stage)
            self.assertTrue(owner.ui.statusFrame.visible)

            fake_now = {"t": 100.0}
            with patch("shared.ui_run_history.time.time", side_effect=lambda: fake_now["t"]):
                ui_run_history.set_run_state(owner, "importing")
                self.assertEqual((owner._runProgressBar.minimum(), owner._runProgressBar.maximum()), (0, 0))
                self.assertFalse(owner.ui.statusFrame.visible)
                self.assertTrue(owner._runStateLabel.visible)
                self.assertTrue(owner._runAnimationTimer.isActive())
                self.assertIn("Importing…", owner._runStageCountLabel.text())

                fake_now["t"] = 108.0
                ui_run_history._on_run_animation_tick(owner)
                self.assertEqual(owner._runSpinnerFrame, 1)
                self.assertEqual(owner._runStageCountLabel.text(), "Importing… · 00:08")

                ui_run_history.set_run_state(owner, "running")
                ui_run_history.apply_progress_snapshot(
                    owner,
                    ProgressSnapshot(done=0, total=6, current_stage="00_preprocessing", status="running"),
                )
                # Still indeterminate (0, 0) while done == 0 even when total == 6
                self.assertEqual((owner._runProgressBar.minimum(), owner._runProgressBar.maximum()), (0, 0))
                self.assertIn("Preparing…", owner._runStageCountLabel.text())

                ui_run_history.set_live_stage_text(owner, "Microstructure", "Fitting DTI")
                self.assertEqual(owner._runStageLabel.text(), "Microstructure — Fitting DTI")
                # Subsequent checkpoint on same stage preserves live stdout detail
                ui_run_history.apply_progress_snapshot(
                    owner,
                    ProgressSnapshot(done=1, total=6, current_stage="02_microstructure", status="running"),
                )
                self.assertEqual((owner._runProgressBar.minimum(), owner._runProgressBar.maximum()), (0, 6))
                self.assertEqual(owner._runProgressBar.value(), 1)
                self.assertEqual(owner._runStageLabel.text(), "Microstructure — Fitting DTI")

                fake_now["t"] = 192.0  # +84s since entering "running" at t=108.0
                ui_run_history._on_run_animation_tick(owner)
                self.assertEqual(owner._runStageCountLabel.text(), "Stage 1 / 6 · 01:24")

                owner._onboardingLanguage = "zh"
                fake_now["t"] = 3773.0  # +3665s -> 01:01:05
                ui_run_history._on_run_animation_tick(owner)
                self.assertEqual(owner._runStageCountLabel.text(), "第 1 / 6 阶段 · 01:01:05")

                ui_run_history.set_run_state(owner, "completed")
                self.assertFalse(owner._runAnimationTimer.isActive())
                self.assertTrue(owner.ui.statusFrame.visible)
                self.assertFalse(owner._runStageLabel.visible)

    def test_compact_mode_keeps_legacy_state_badge_hidden_and_refits_on_cancel(self) -> None:
        fake_qt = _build_fake_qt_module()
        with patch.dict(sys.modules, {"qt": fake_qt}):
            from shared import ui_run_history

            owner = _make_run_progress_owner()
            owner._advancedSettingsPanel = object()
            fit_calls = []
            owner._queueCompactFit = lambda: fit_calls.append(owner._runState)
            ui_run_history.install_run_progress_panel(owner)

            ui_run_history.set_run_state(owner, "running")
            self.assertFalse(owner._runStateLabel.visible)
            self.assertTrue(owner._runProgressRowWidget.visible)
            self.assertFalse(owner.ui.statusFrame.visible)
            ui_run_history.apply_progress_snapshot(
                owner,
                ProgressSnapshot(done=2, total=6, current_stage="02_microstructure", status="running"),
            )
            fit_calls.clear()

            ui_run_history.set_run_state(owner, "cancelled")
            self.assertFalse(owner._runStateLabel.visible)
            self.assertTrue(owner._runProgressRowWidget.visible)
            self.assertTrue(owner.ui.statusFrame.visible)
            self.assertFalse(owner._runStageLabel.visible)
            self.assertEqual(owner._runStageCountLabel.text(), "2 / 6 stages")
            self.assertIn("cancelled", fit_calls)

    def test_show_output_files_surfaces_06_qc_txt_and_filters_cache(self) -> None:
        fake_qt = _build_fake_qt_module()
        slicer_mod = types.ModuleType("slicer")
        slicer_mod.util = SimpleNamespace(getNodesByClass=lambda _cls: [])
        slicer_mod.mrmlScene = SimpleNamespace()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            sys.modules, {"qt": fake_qt, "slicer": slicer_mod}
        ):
            subject_root = Path(tmp) / "sub-01"
            out_root = subject_root / "outputs"
            cache_file = subject_root / "cache" / "stages" / "cached_fa.nii.gz"
            cache_json = subject_root / "cache" / "reports" / "internal.json"
            qc_top = out_root / "06_qc" / "summary_qc.txt"
            qc_nested = out_root / "06_qc" / "sub" / "tract_qc.txt"
            fa_file = out_root / "02_microstructure" / "fa.nii.gz"
            for path in (cache_file, cache_json, qc_top, qc_nested, fa_file):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("ok", encoding="utf-8")

            manifest = {
                "subject_root": str(subject_root),
                "output_root": str(out_root),
                "tasks": [
                    {"name": "microstructure.dti", "outputs": {"fa": str(fa_file), "tmp": str(cache_json)}},
                ],
            }
            fake_candidates = [
                {"product": "microstructure.fa", "path": str(fa_file.resolve())},
                {"product": "microstructure.cached", "path": str(cache_file.resolve())},
            ]

            from shared import ui_run_history

            opened: list[Path] = []
            owner = _make_run_progress_owner()
            owner._latestResultManifestPath = out_root / "00_run_metadata" / "run_manifest.json"
            owner._latestResultManifestPath.parent.mkdir(parents=True, exist_ok=True)
            owner._open_local_path = lambda p, **_kw: opened.append(Path(p))
            owner._runOutputFiles = fake_qt.QWidget()

            with patch("shared.result_scene.result_candidates", return_value=(fake_candidates, {})), \
                 patch("shared.ui_run_history._refresh_output_visibility"):
                ui_run_history._show_output_files(owner, manifest)

            groups = {
                owner._runOutputFiles.topLevelItem(i).text(0): owner._runOutputFiles.topLevelItem(i)
                for i in range(owner._runOutputFiles.topLevelItemCount)
            }
            self.assertIn("Quality control (2)", groups)
            qc_group = groups["Quality control (2)"]
            qc_items = [qc_group.child(i) for i in range(qc_group.childCount())]
            self.assertEqual([item.text(0) for item in qc_items], ["tract_qc.txt", "summary_qc.txt"])
            self.assertTrue(all(item.text(1) == "↗" for item in qc_items))

            def _paths_under(item):  # Results subgroups: a family holds subgroups, a subgroup holds files
                rows = [item.child(j) for j in range(item.childCount())]
                return [p for row in rows for p in ([row.data(0, fake_qt.Qt.UserRole)] if row.data(0, fake_qt.Qt.UserRole)
                                                     else _paths_under(row))]

            all_paths = [
                path
                for i in range(owner._runOutputFiles.topLevelItemCount)
                for path in _paths_under(owner._runOutputFiles.topLevelItem(i))
            ]
            self.assertIn(str(fa_file.resolve()), all_paths)
            self.assertNotIn(str(cache_file.resolve()), all_paths)
            self.assertNotIn(str(cache_json.resolve()), all_paths)

            ui_run_history._toggle_output(owner, qc_items[0])
            self.assertEqual(opened, [qc_nested.resolve()])

            # Verify fallback to manifest_path.parent.parent / "06_qc" + Chinese caption when output_root omitted
            owner._onboardingLanguage = "zh"
            with patch("shared.result_scene.result_candidates", return_value=([], {})), \
                 patch("shared.ui_run_history._refresh_output_visibility"):
                ui_run_history._show_output_files(owner, {"subject_root": str(subject_root), "tasks": []})
            zh_groups = [
                owner._runOutputFiles.topLevelItem(i).text(0)
                for i in range(owner._runOutputFiles.topLevelItemCount)
            ]
            self.assertIn("质量控制 (2)", zh_groups)


class EnvironmentAndWidgetFlowTests(unittest.TestCase):
    def test_start_environment_check_sets_preflight_and_resets_idle(self) -> None:
        from shared import ui_environment_check

        captured = {}

        def fake_start_setup_job(owner, title, commands, done):
            captured["done"] = done
            captured["state_at_start"] = owner._runState
            return True

        with tempfile.TemporaryDirectory() as tmp:
            owner = _make_run_progress_owner()
            owner._pipelineActive = False
            owner._setupJob = None
            owner._selectedSetupActive = False
            owner.logic = SimpleNamespace(module_dir=REPO_ROOT)
            owner._slicer_launcher_program = lambda: "/Applications/Slicer.app/Contents/Slicer"
            owner._update_environment_result_label = lambda *_a, **_k: None
            owner._update_environment_path_status_labels = lambda *_a: None
            owner._update_environment_status = lambda *_a: None
            owner._cache_environment_result = lambda *_a: None
            owner._environment_check_kwargs = lambda: {}
            owner._environment_check_signature = lambda _opts: ("sig",)
            owner._environment_report_dir_from_kwargs = lambda _opts: Path(tmp)
            owner._setStatus = lambda *_a: None
            owner.appendLog = lambda *_a: None

            fake_slicer = types.ModuleType("slicer")
            fake_slicer.modules = SimpleNamespace()
            fake_qt = _build_fake_qt_module()
            with patch.dict(sys.modules, {"slicer": fake_slicer, "qt": fake_qt}), \
                 patch("shared.ui_setup_jobs.start_setup_job", side_effect=fake_start_setup_job), \
                 patch("shared.environment_check.check_config", return_value={}), \
                 patch("shared.environment_check.result_from_dict", return_value=SimpleNamespace(ready=True)):
                from shared import ui_run_history

                ui_run_history.install_run_progress_panel(owner)
                started = ui_environment_check.start_environment_check(owner)
                self.assertTrue(started)
                self.assertEqual(captured["state_at_start"], "preflight")
                self.assertFalse(owner.ui.statusFrame.visible)
                self.assertIn("Checking", owner._runStageCountLabel.text())

                # Simulate result written by check_environment.py
                result_file = next(Path(tmp).glob("check-*-request.json")).with_name(
                    next(Path(tmp).glob("check-*-request.json")).name.replace("-request.json", "-result.json")
                )
                result_file.write_text("{}", encoding="utf-8")
                captured["done"]("completed", "", Path(tmp) / "log.txt")
                self.assertEqual(owner._runState, "idle")
                self.assertTrue(owner.ui.statusFrame.visible)

    def test_widget_import_qprocess_cancel_failure_live_log_and_cleanup(self) -> None:
        created_processes = []

        class _StubQProcess(_FakeWidget):
            NotRunning = 0
            Running = 2
            MergedChannels = 1

            def __init__(self, parent=None) -> None:
                super().__init__()
                self._state = 0
                self.started_with = None
                self.terminated = False
                self.killed = False
                created_processes.append(self)

            def setWorkingDirectory(self, _d: str) -> None:
                pass

            def setProcessChannelMode(self, _m: int) -> None:
                pass

            def setProcessEnvironment(self, _e) -> None:
                pass

            def start(self, program: str, args: list[str]) -> None:
                self._state = 2
                self.started_with = (program, list(args))

            def state(self) -> int:
                return self._state

            def processId(self) -> int:
                return 0

            def terminate(self) -> None:
                self.terminated = True
                self._state = 0

            def kill(self) -> None:
                self.killed = True
                self._state = 0

            def readAllStandardOutput(self):
                return b""

        fake_qt = _build_fake_qt_module(process_factory=_StubQProcess)
        errors: list[str] = []
        infos: list[str] = []
        fake_slicer = types.ModuleType("slicer")
        fake_slicer.util = SimpleNamespace(
            errorDisplay=lambda msg, **_kw: errors.append(str(msg)),
            infoDisplay=lambda msg, **_kw: infos.append(str(msg)),
            NodeModify=lambda _node: SimpleNamespace(__enter__=lambda s: s, __exit__=lambda *a: False),
        )
        fake_slicer.app = SimpleNamespace(disconnect=lambda *_a: None)
        fake_vtk = types.ModuleType("vtk")
        fake_slicer_mod = types.ModuleType("slicer.ScriptedLoadableModule")

        class _BaseWidget:
            def __init__(self, parent=None) -> None:
                self.parent = parent

        fake_slicer_mod.ScriptedLoadableModuleWidget = _BaseWidget

        prior_mw = sys.modules.pop("shared.module_widget", None)
        try:
            with patch.dict(
                sys.modules,
                {
                    "qt": fake_qt,
                    "slicer": fake_slicer,
                    "slicer.ScriptedLoadableModule": fake_slicer_mod,
                    "vtk": fake_vtk,
                },
            ):
                import shared.module_widget as mw

                widget = mw.SharedWidgetMixin()
                widget.parent = None
                widget.logic = SimpleNamespace(module_dir=REPO_ROOT)
                widget._pipelineActive = False
                widget._cleanedUp = False
                widget._onboardingLanguage = "en"
                widget._parameterNode = SimpleNamespace(brainMaskBackend="auto", inputDirectory="", subjectId="sub-test")
                widget._prepareInputWidgets = {
                    "source_kind": "nifti",
                    "status": _FakeWidget(),
                    "panel": _FakeWidget(),
                }
                widget._cancelRunButton = _FakeWidget("Cancel")
                widget._cancelRunButton.visible = False
                widget._cancelRunButton.enabled = False
                widget._pendingCancelledProcessGroups = set()
                widget._runHistoryRefreshTimer = None
                widget._commonUserSettingsSaveTimer = None
                widget.removeObservers = lambda: None
                widget._set_module_help_section_hidden = lambda *_a, **_k: None
                widget._refresh_run_button_state = lambda: None
                widget._setStatus = lambda *_a: None
                widget.appendLog = lambda *_a: None
                widget._slicer_launcher_program = lambda: "/Applications/Slicer.app/Contents/Slicer"
                widget._prepare_inputs_widget_text = lambda key: "/tmp/dwi.nii.gz" if key == "source" else ""
                widget._shared_subject_id_text = lambda: "sub-test"
                widget._environment_dependency_path_kwargs = lambda: {}
                widget._apply_shared_inputs_to_workflow_fields = lambda: None
                widget._checkCanApply = lambda: None
                widget._update_run_status_table_item = lambda *_a: None
                widget._setupJob = None

                owner = _make_run_progress_owner()
                widget.ui = owner.ui
                widget._uiWidget = None
                widget._poll_run_progress = lambda: None
                mw.set_run_state(widget, "idle")
                from shared import ui_run_history
                ui_run_history.install_run_progress_panel(widget)

                with tempfile.TemporaryDirectory() as tmp:
                    cfg = mw.InputImportConfig(
                        source=Path(tmp) / "missing.nii.gz",
                        subject_root=Path(tmp) / "sub",
                        subject_id="sub-test",
                        source_kind="nifti",
                    )
                    widget._prepare_inputs_config = lambda: cfg

                    # 1. Start import -> uses QProcess, enters "importing", enables cancel button
                    widget._on_prepare_inputs_clicked(run_after=True)
                    self.assertEqual(widget._runState, "importing")
                    self.assertTrue(widget._cancelRunButton.visible)
                    self.assertTrue(widget._cancelRunButton.enabled)
                    self.assertEqual(len(created_processes), 1)
                    proc = created_processes[0]

                    def _unwrap_grouped(started_with):
                        prog, argv = started_with
                        if Path(prog).name == "perl" and "--" in argv:
                            idx = argv.index("--")
                            return argv[idx + 1], argv[idx + 2 :]
                        if Path(prog).name == "setsid" and argv:
                            return argv[0], argv[1:]
                        return prog, argv

                    unwrapped_prog, unwrapped_args = _unwrap_grouped(proc.started_with)
                    self.assertEqual(unwrapped_prog, "/Applications/Slicer.app/Contents/Slicer")
                    self.assertEqual(unwrapped_args[:3], ["--launch", "PythonSlicer", "-u"])
                    self.assertIn("prepare_inputs_cli.py", unwrapped_args[3])

                    # 2. Cancel import -> terminates QProcess cleanly and resets to "cancelled"
                    widget._onCancelRunClicked()
                    self.assertTrue(proc.terminated)
                    self.assertEqual(widget._runState, "cancelled")
                    self.assertFalse(widget._inputImportPending)
                    self.assertFalse(widget._cancelRunButton.visible)

                    # 3. Failure with run_after=True -> sets "failed" and shows errorDisplay
                    widget._on_prepare_inputs_clicked(run_after=True)
                    proc2 = created_processes[1]
                    proc2._signals["finished(int,QProcess::ExitStatus)"].emit(1, 0)
                    self.assertEqual(widget._runState, "failed")
                    self.assertEqual(len(errors), 1)

                    # 3b. Real macOS .app bundle layout uses Contents/bin/PythonSlicer directly (never Contents/MacOS/Slicer --launch)
                    bundle_root = Path(tmp) / "Slicer.app" / "Contents"
                    (bundle_root / "MacOS").mkdir(parents=True, exist_ok=True)
                    (bundle_root / "bin").mkdir(parents=True, exist_ok=True)
                    macos_launcher = bundle_root / "MacOS" / "Slicer"
                    python_slicer = bundle_root / "bin" / "PythonSlicer"
                    macos_launcher.write_text("", encoding="utf-8")
                    python_slicer.write_text("", encoding="utf-8")
                    widget._slicer_launcher_program = lambda: str(macos_launcher)
                    with patch("sys.platform", "darwin"), patch("shared.external_process.sys.platform", "darwin"):
                        widget._on_prepare_inputs_clicked(run_after=False)
                    proc3 = created_processes[2]
                    mac_prog, mac_args = _unwrap_grouped(proc3.started_with)
                    self.assertEqual(mac_prog, str(python_slicer))
                    self.assertEqual(mac_args[0], "-u")
                    self.assertIn("prepare_inputs_cli.py", mac_args[1])
                    self.assertNotIn("--launch", mac_args)
                    widget._onCancelRunClicked()

                    # 3c. _prepare_inputs_config colocates under <Output>/<Subject ID>, or <config_dir>/subjects/<Subject ID> when Output blank.
                    src_dwi = Path(tmp) / "src" / "dwi.nii.gz"
                    src_dwi.parent.mkdir(parents=True, exist_ok=True)
                    src_dwi.write_text("x", encoding="utf-8")
                    out_dir = Path(tmp).resolve() / "user_out"
                    cfg_widget = mw.SharedWidgetMixin()
                    cfg_widget._parameterNode = SimpleNamespace(outputDirectory=str(out_dir), brainMaskBackend="pnl_cnn")
                    cfg_widget._prepareInputWidgets = {"source_kind": "nifti"}
                    cfg_widget._prepare_inputs_widget_text = lambda key: str(src_dwi) if key == "source" else ""
                    cfg_widget._shared_subject_id_text = lambda: "sub-gui"
                    cfg_widget._environment_dependency_path_kwargs = lambda: {}
                    cfg_with_out = cfg_widget._prepare_inputs_config()
                    self.assertEqual(cfg_with_out.subject_root, out_dir / "sub-gui")
                    self.assertTrue(cfg_with_out.force)
                    cfg_widget._parameterNode.outputDirectory = ""
                    with patch("shared.user_config.default_user_config_path", return_value=Path(tmp).resolve() / "cfg" / "user_config.json"):
                        cfg_blank = cfg_widget._prepare_inputs_config()
                    self.assertEqual(cfg_blank.subject_root, (Path(tmp).resolve() / "cfg" / "subjects" / "sub-gui"))

                # 4. Install Python dependencies enters "setup" and resets to "idle" on success
                setup_captures = {}

                def _fake_setup_job(target_owner, title, cmds, done):
                    setup_captures["state"] = target_owner._runState
                    setup_captures["done"] = done
                    return True

                widget._save_python_install_status = lambda *_a: None
                widget._show_actionable_error = lambda **_k: None
                with patch("shared.ui_selected_setup.python_setup_commands", return_value=([], [])), \
                     patch("shared.ui_setup_jobs.start_setup_job", side_effect=_fake_setup_job):
                    widget._install_python_packages_from_environment_dialog(("nibabel",))
                self.assertEqual(setup_captures["state"], "setup")
                setup_captures["done"]("completed", "", Path("/tmp/install.log"))
                self.assertEqual(widget._runState, "idle")

                # 5. Live log line updates _runStageLabel immediately and cleanup stops animation
                mw.set_run_state(widget, "running")
                self.assertTrue(widget._runAnimationTimer.isActive())
                widget._handleBackgroundOutputLine("[Connectome] Building tck2connectome matrix")
                self.assertIn("Connectome", widget._runStageLabel.text())
                self.assertTrue(widget._runStageLabel.visible)
                widget.cleanup()
                self.assertFalse(widget._runAnimationTimer.isActive())
        finally:
            sys.modules.pop("shared.module_widget", None)
            if prior_mw is not None:
                sys.modules["shared.module_widget"] = prior_mw


if __name__ == "__main__":
    unittest.main()
