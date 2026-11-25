"""Regression tests for on-demand in-Slicer CUDA PyTorch setup UX and Linux/Windows precedence."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class _FakeModelItem:
    def __init__(self) -> None:
        self.enabled = True

    def setEnabled(self, flag: bool) -> None:
        self.enabled = bool(flag)


class _FakeComboModel:
    def __init__(self, items: list[_FakeModelItem]) -> None:
        self._items = items

    def item(self, idx: int) -> _FakeModelItem | None:
        if 0 <= idx < len(self._items):
            return self._items[idx]
        return None


class _ModelAwareFakeComboBox:
    def __init__(self) -> None:
        self.items: list[tuple[str, str]] = []
        self._model_items: list[_FakeModelItem] = []
        self._model = _FakeComboModel(self._model_items)
        self.currentIndex = 0
        self.tooltip = ""
        self._blocked = False

    def clear(self) -> None:
        self.items.clear()
        self._model_items.clear()

    def addItem(self, text: str, data: str = "") -> None:
        self.items.append((text, data))
        self._model_items.append(_FakeModelItem())

    def model(self) -> _FakeComboModel:
        return self._model

    def findData(self, value: str) -> int:
        for idx, (_text, data) in enumerate(self.items):
            if data == value:
                return idx
        return -1

    def setCurrentIndex(self, idx: int) -> None:
        self.currentIndex = int(idx)

    def itemData(self, idx: int) -> str:
        return self.items[idx][1] if 0 <= idx < len(self.items) else ""

    def blockSignals(self, flag: bool) -> bool:
        prev = self._blocked
        self._blocked = bool(flag)
        return prev

    def setToolTip(self, text: str) -> None:
        self.tooltip = text

    def setSizePolicy(self, *_a) -> None:
        pass

    def setSizeAdjustPolicy(self, *_a) -> None:
        pass

    def setMinimumContentsLength(self, *_a) -> None:
        pass

    def setCursor(self, *_a) -> None:
        pass

    def show(self) -> None:
        pass


class _FakeSignal:
    def __init__(self) -> None:
        self.callbacks: list = []

    def connect(self, cb) -> None:
        self.callbacks.append(cb)

    def emit(self) -> None:
        for cb in list(self.callbacks):
            cb()


class _FakeWidget:
    def __init__(self, _parent=None) -> None:
        self.children: list = []
        self.text = ""
        self.tooltip = ""
        self.accessible_name = ""
        self.stylesheet = ""
        self.object_name = ""
        self.visible = True
        self.clicked = _FakeSignal()

    def setObjectName(self, name: str) -> None:
        self.object_name = name

    def setText(self, text: str) -> None:
        self.text = text

    def setToolTip(self, text: str) -> None:
        self.tooltip = text

    def setAccessibleName(self, name: str) -> None:
        self.accessible_name = name

    def setStyleSheet(self, css: str) -> None:
        self.stylesheet = css

    def setAutoRaise(self, *_a) -> None:
        pass

    def setFixedSize(self, *_a) -> None:
        pass

    def setCursor(self, *_a) -> None:
        pass

    def setFocusPolicy(self, *_a) -> None:
        pass

    def setContentsMargins(self, *_a) -> None:
        pass

    def setSpacing(self, *_a) -> None:
        pass

    def addWidget(self, w, *_a) -> None:
        self.children.append(w)

    def show(self) -> None:
        self.visible = True

    def hide(self) -> None:
        self.visible = False


def _make_fake_qt():
    return SimpleNamespace(
        QWidget=_FakeWidget,
        QHBoxLayout=_FakeWidget,
        QToolButton=_FakeWidget,
        QComboBox=SimpleNamespace(AdjustToMinimumContentsLengthWithIcon=0),
        QSizePolicy=SimpleNamespace(Expanding=1, Fixed=0),
        Qt=SimpleNamespace(AlignVCenter=4, PointingHandCursor=13, StrongFocus=11),
    )


class ComputeDeviceCudaUiTests(unittest.TestCase):
    def test_gpu_detected_with_cpu_torch_disables_cuda_items_and_shows_help_button(self) -> None:
        from shared import ui_compact

        combo = _ModelAwareFakeComboBox()
        label = _FakeWidget()
        targets_layout = _FakeWidget()
        owner = SimpleNamespace(
            ui=SimpleNamespace(
                unifiedTorchDeviceLabel=label,
                unifiedTorchDeviceComboBox=combo,
                targetsLayout=targets_layout,
                environmentFormLayout=None,
            ),
            _parameterNode=SimpleNamespace(device="cuda:0", ddsegDevice="cuda:0"),
            _onboardingLanguage="zh",
        )
        fake_qt = _make_fake_qt()

        with (
            mock.patch.dict(sys.modules, {"qt": fake_qt}),
            mock.patch.object(ui_compact, "_detect_mps_devices", return_value=()),
            mock.patch.object(ui_compact, "_detect_cuda_devices", return_value=(("cuda:0", "CUDA:0 (RTX 3060)"),)),
            mock.patch.object(ui_compact, "_cuda_pytorch_runtime_available", return_value=False),
        ):
            ui_compact.install_compute_device_selector(owner)
            self.assertEqual([v for _, v in combo.items], ["auto", "cpu", "cuda:0"])
            self.assertIn("需安装 CUDA 版 PyTorch", combo.items[2][0])
            self.assertTrue(combo.model().item(0).enabled)
            self.assertTrue(combo.model().item(1).enabled)
            self.assertFalse(combo.model().item(2).enabled)
            # Stale cuda:0 selection snaps safely to Auto (index 0)
            self.assertEqual(combo.currentIndex, 0)
            self.assertEqual(owner._parameterNode.device, "auto")
            self.assertEqual(owner._parameterNode.ddsegDevice, "auto")
            # "?" button is visible with size warning tooltip
            help_btn = owner._computeDeviceCudaHelpButton
            self.assertTrue(help_btn.visible)
            self.assertEqual(help_btn.text, "?")
            self.assertIn("2.5", help_btn.tooltip)
            self.assertIn("4.6", help_btn.tooltip)

            # Clicking "?" invokes show_cuda_pytorch_setup_dialog
            with mock.patch("shared.ui_cuda_setup.show_cuda_pytorch_setup_dialog") as mock_open:
                help_btn.clicked.emit()
            mock_open.assert_called_once_with(owner)

            # Switch to English -> suffix & tooltip localize cleanly
            ui_compact.refresh_compute_device_selector(owner, "en")
            self.assertIn("CUDA PyTorch not installed", combo.items[2][0])
            self.assertIn("2.5", help_btn.tooltip)

    def test_gpu_detected_with_cuda_torch_ready_enables_cuda_and_hides_help_button(self) -> None:
        from shared import ui_compact

        combo = _ModelAwareFakeComboBox()
        label = _FakeWidget()
        targets_layout = _FakeWidget()
        owner = SimpleNamespace(
            ui=SimpleNamespace(
                unifiedTorchDeviceLabel=label,
                unifiedTorchDeviceComboBox=combo,
                targetsLayout=targets_layout,
                environmentFormLayout=None,
            ),
            _parameterNode=SimpleNamespace(device="cuda:0", ddsegDevice="cuda:0"),
            _onboardingLanguage="en",
        )
        fake_qt = _make_fake_qt()

        with (
            mock.patch.dict(sys.modules, {"qt": fake_qt}),
            mock.patch.object(ui_compact, "_detect_mps_devices", return_value=()),
            mock.patch.object(ui_compact, "_detect_cuda_devices", return_value=(("cuda:0", "CUDA:0 (RTX 3060)"),)),
            mock.patch.object(ui_compact, "_cuda_pytorch_runtime_available", return_value=True),
        ):
            ui_compact.install_compute_device_selector(owner)
            self.assertEqual(combo.items[2], ("CUDA:0 (RTX 3060)", "cuda:0"))
            self.assertTrue(combo.model().item(2).enabled)
            self.assertEqual(combo.currentIndex, 2)
            self.assertFalse(owner._computeDeviceCudaHelpButton.visible)

    def test_no_nvidia_gpu_omits_fake_cuda_fallback_and_hides_help_button(self) -> None:
        from shared import ui_compact

        orig_cache = ui_compact._DETECTED_CUDA_DEVICES_CACHE
        try:
            ui_compact._DETECTED_CUDA_DEVICES_CACHE = None
            with (
                mock.patch.dict(sys.modules, {"torch": None}),
                mock.patch("shutil.which", return_value=None),
                mock.patch("shared.hardware_adapt.find_nvidia_smi", return_value=None),
            ):
                detected = ui_compact._detect_cuda_devices()
            self.assertEqual(detected, ())
        finally:
            ui_compact._DETECTED_CUDA_DEVICES_CACHE = orig_cache

        combo = _ModelAwareFakeComboBox()
        owner = SimpleNamespace(
            ui=SimpleNamespace(
                unifiedTorchDeviceLabel=_FakeWidget(),
                unifiedTorchDeviceComboBox=combo,
                targetsLayout=_FakeWidget(),
                environmentFormLayout=None,
            ),
            _parameterNode=SimpleNamespace(device="auto", ddsegDevice="auto"),
            _onboardingLanguage="en",
        )
        with (
            mock.patch.dict(sys.modules, {"qt": _make_fake_qt()}),
            mock.patch.object(ui_compact, "_detect_mps_devices", return_value=()),
            mock.patch.object(ui_compact, "_detect_cuda_devices", return_value=()),
        ):
            ui_compact.install_compute_device_selector(owner)
            self.assertEqual([v for _, v in combo.items], ["auto", "cpu"])
            self.assertFalse(owner._computeDeviceCudaHelpButton.visible)


class CudaSetupSummaryAndCommandTests(unittest.TestCase):
    def test_summary_includes_size_warnings_and_low_vram_ddsurfer_tip(self) -> None:
        from shared.ui_cuda_setup import NvidiaGpuInfo, build_cuda_setup_summary

        rtx3060_laptop = (
            NvidiaGpuInfo(
                index=0,
                name="NVIDIA GeForce RTX 3060 Laptop GPU",
                short_name="RTX 3060 Laptop GPU",
                total_vram_mib=6144,
                driver_version="560.94",
                compute_cap="8.6",
            ),
        )
        zh = build_cuda_setup_summary(rtx3060_laptop, zh=True, platform_name="win32", install_target_dir="C:/Slicer/site-packages")
        self.assertEqual(zh["flavor"], "cu128")
        self.assertEqual(zh["index_url"], "https://download.pytorch.org/whl/cu128")
        self.assertFalse(zh["driver_too_old"])
        self.assertIn("2.5–2.7 GB", zh["summary_html"])
        self.assertIn("4.6–5.2 GB", zh["summary_html"])
        self.assertIn("DDSurfer", zh["low_vram_tip"])
        self.assertIn("重启", zh["summary_html"])

        rtx3060_12g = (
            NvidiaGpuInfo(
                index=0,
                name="NVIDIA GeForce RTX 3060",
                short_name="RTX 3060",
                total_vram_mib=12288,
                driver_version="560.94",
                compute_cap="8.6",
            ),
        )
        en = build_cuda_setup_summary(rtx3060_12g, zh=False, platform_name="win32", install_target_dir="C:/Slicer/site-packages")
        self.assertEqual(en["low_vram_tip"], "")
        self.assertIn("2.5–2.7 GB", en["summary_html"])
        self.assertIn("4.6–5.2 GB", en["summary_html"])
        self.assertIn("restart", en["summary_html"].lower())

        old_driver = (
            NvidiaGpuInfo(
                index=0,
                name="NVIDIA GeForce RTX 3060",
                short_name="RTX 3060",
                total_vram_mib=12288,
                driver_version="440.10",
                compute_cap="8.6",
            ),
        )
        old = build_cuda_setup_summary(old_driver, zh=False, platform_name="win32")
        self.assertTrue(old["driver_too_old"])
        self.assertIsNone(old["flavor"])

    def test_build_cuda_setup_command_strips_bundled_python_and_forwards_proxy(self) -> None:
        from shared.ui_cuda_setup import build_cuda_pytorch_setup_command

        owner = SimpleNamespace(_setup_python_command=lambda _ov=None: ("Slicer.exe", ["--launch", "PythonSlicer"]))
        with tempfile.TemporaryDirectory() as tmp:
            fake_pp = os.pathsep.join(
                [
                    "C:/Users/demo/AppData/Slicer/Extensions/SlicerDSpace/bundled-python/site-packages",
                    "C:/Users/demo/custom_tools",
                ]
            )
            with mock.patch.dict(os.environ, {"PYTHONPATH": fake_pp}, clear=False):
                cmds, req_path, receipt_path = build_cuda_pytorch_setup_command(
                    owner,
                    folder=Path(tmp),
                    cuda_index_url="https://download.pytorch.org/whl/cu128",
                    proxy_url="http://127.0.0.1:7890",
                )
            self.assertEqual(len(cmds), 1)
            program, args, env = cmds[0]
            self.assertEqual(program, "Slicer.exe")
            self.assertIn("setup_python_packages.py", " ".join(args))
            self.assertIn(str(req_path), args)
            self.assertIn(str(receipt_path), args)
            self.assertEqual(env["PYTHONPATH"], "C:/Users/demo/custom_tools")
            self.assertIsNone(env["SLICERDSPACE_BUNDLED_SITE_PACKAGES"])
            self.assertIsNone(env["SLICERDSPACE_BUNDLED_PYTHON_ROOT"])
            self.assertEqual(env["HTTPS_PROXY"], "http://127.0.0.1:7890")
            payload = json.loads(req_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["direct_cuda_wheel"])
            self.assertTrue(payload["require_cuda_torch"])
            self.assertTrue(payload["replace_cpu_torch"])
            self.assertTrue(payload["torchvision_required"])
            self.assertEqual(payload["cuda_index_url"], "https://download.pytorch.org/whl/cu128")
            self.assertEqual(payload["python_path_entries"], [])

    def test_restart_prompt_calls_slicer_restart_only_when_confirmed(self) -> None:
        from shared.ui_cuda_setup import _prompt_restart_after_cuda_install

        restarts: list[bool] = []
        fake_slicer = SimpleNamespace(
            util=SimpleNamespace(mainWindow=lambda: None, restart=lambda: restarts.append(True))
        )
        created: list = []

        class _FakeMsgBox:
            Information = 1
            AcceptRole = 0
            RejectRole = 1

            def __init__(self, _parent=None) -> None:
                self.buttons: list[tuple[str, int]] = []
                self._clicked_idx = 0
                created.append(self)

            def setWindowTitle(self, _t: str) -> None:
                pass

            def setIcon(self, _i) -> None:
                pass

            def setText(self, _t: str) -> None:
                pass

            def addButton(self, text: str, role: int):
                tok = (text, role)
                self.buttons.append(tok)
                return tok

            def exec_(self) -> None:
                pass

            def clickedButton(self):
                return self.buttons[self._clicked_idx]

        fake_qt = SimpleNamespace(QMessageBox=_FakeMsgBox)
        owner = SimpleNamespace(parent=None)
        with mock.patch.dict(sys.modules, {"qt": fake_qt, "slicer": fake_slicer}):
            # Click "Restart now" (button 0)
            self.assertTrue(_prompt_restart_after_cuda_install(owner, zh=True))
            self.assertEqual(len(restarts), 1)
            # Click "Restart later" (button 1)
            _FakeMsgBox.clickedButton = lambda self: self.buttons[1]
            self.assertFalse(_prompt_restart_after_cuda_install(owner, zh=False))
            self.assertEqual(len(restarts), 1)


class DirectCudaWheelBackendTests(unittest.TestCase):
    def test_direct_cuda_wheel_in_slicer_protects_packages_and_sets_windows_max_path_guards(self) -> None:
        from shared import python_setup

        state = {"torch": "2.15.0+cpu", "torchvision": "0.21.0+cpu", "numpy": "1.26.4"}
        pip_calls: list[list[str]] = []
        constraints_text: list[str] = []

        def _fake_run(cmd, **kwargs):
            cmd_list = list(cmd)
            pip_calls.append(cmd_list)
            if "uninstall" in cmd_list:
                state.pop("torch", None)
                state.pop("torchvision", None)
            elif "install" in cmd_list:
                c_idx = cmd_list.index("--constraint")
                constraints_text.append(Path(cmd_list[c_idx + 1]).read_text(encoding="utf-8"))
                state["torch"] = "2.9.1+cu128"
                state["torchvision"] = "0.24.1+cu128"
            return subprocess.CompletedProcess(cmd_list, 0)

        def _fake_meta_ver(name: str) -> str:
            key = name.lower()
            if key not in state:
                raise python_setup.importlib.metadata.PackageNotFoundError(name)
            return state[key]

        with (
            mock.patch.object(python_setup.sys, "platform", "win32"),
            mock.patch.object(python_setup, "installed_versions", side_effect=lambda: dict(state)),
            mock.patch.object(python_setup.importlib.metadata, "version", side_effect=_fake_meta_ver),
            mock.patch.object(python_setup, "requirement_satisfied", side_effect=lambda r: "torch" not in r or "torch" in state),
            mock.patch.object(python_setup, "bundled_shortfalls", return_value=[]),
            mock.patch.object(python_setup.subprocess, "run", side_effect=_fake_run),
        ):
            rec = python_setup.install_requirements(
                ("torch>=2.3.1", "torchvision"),
                protected=True,
                loaded=(),
                torch_minimum="2.3.1",
                torchvision_required=True,
                require_cuda_torch=True,
                replace_cpu_torch=True,
                direct_cuda_wheel=True,
                cuda_index_url="https://download.pytorch.org/whl/cu128",
            )

        self.assertEqual(rec["status"], "completed")
        self.assertEqual(rec["backend"], "python_subprocess_cuda_wheel")
        self.assertTrue(rec["restart_required"])
        self.assertIn("torch", rec["restart_packages"])
        self.assertEqual(len(pip_calls), 2)
        self.assertEqual(pip_calls[0][1:4], ["-m", "pip", "uninstall"])
        install_cmd = pip_calls[1]
        self.assertIn("--no-cache-dir", install_cmd)
        self.assertIn("--upgrade-strategy", install_cmd)
        self.assertIn("only-if-needed", install_cmd)
        self.assertNotIn("--force-reinstall", install_cmd)
        self.assertNotIn("--extra-index-url", install_cmd)
        self.assertIn("https://download.pytorch.org/whl/cu128", install_cmd)
        self.assertIn("torch>=2.3.1,!=2.13.*", install_cmd)
        self.assertIn("torchvision!=0.28.*", install_cmd)
        # Slicer's numpy is pinned in constraints, while torch/torchvision are NOT pinned
        self.assertIn("numpy==1.26.4", constraints_text[0])
        self.assertNotIn("torch==", constraints_text[0])


class LinuxBundledPythonCudaPrecedenceTests(unittest.TestCase):
    def test_linux_slicer_cuda_torch_takes_precedence_over_bundled_python_with_numpy(self) -> None:
        import platform
        from shared import bundled_python

        # Warm CPython's platform.uname() cache before temporarily mocking sys.platform="linux"
        # so subsequent tests in the same Windows process still see platform.system() == "Windows".
        _ = platform.uname()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            py_ver = f"{sys.version_info[0]}.{sys.version_info[1]}"
            slicer_root = root / "Slicer-5.10"
            slicer_py = slicer_root / "bin" / "PythonSlicer"
            slicer_py.parent.mkdir(parents=True)
            slicer_py.write_bytes(b"")
            slicer_sp = slicer_root / "lib" / "Python" / "lib" / f"python{py_ver}" / "site-packages"
            (slicer_sp / "torch-2.6.0+cu128.dist-info").mkdir(parents=True)
            (slicer_sp / "torch" / "lib").mkdir(parents=True)
            (slicer_sp / "torch" / "lib" / "libtorch_cuda.so").write_bytes(b"")

            ext_root = root / "SlicerDSpace"
            bundled_sp = ext_root / "bundled-python" / "site-packages"
            (bundled_sp / "numpy").mkdir(parents=True)
            (bundled_sp / "torch-2.15.0+cpu.dist-info").mkdir(parents=True)
            (ext_root / "bundled-python" / "python-runtime.json").write_text(
                json.dumps({"python_version": py_ver}), encoding="utf-8"
            )

            self.assertTrue(bundled_python.site_dir_has_cuda_torch(slicer_sp))
            self.assertFalse(bundled_python.site_dir_has_cuda_torch(bundled_sp))

            # 1. inject_bundled_python_env places Linux Slicer site-packages ahead of bundled-python
            env: dict[str, str] = {"SLICERDSPACE_BUNDLED_SITE_PACKAGES": str(bundled_sp)}
            with (
                mock.patch.object(bundled_python.sys, "platform", "linux"),
                mock.patch.object(bundled_python, "_slicer_site_packages_dirs", return_value=(slicer_sp,)),
                mock.patch.dict(os.environ, {"SLICERDSPACE_BUNDLED_PYTHON_ROOT": str(ext_root / "bundled-python")}),
            ):
                bundled_python.inject_bundled_python_env(env, python_executable=slicer_py)

            pp = env["PYTHONPATH"].split(os.pathsep)
            self.assertLess(pp.index(str(slicer_sp)), pp.index(str(bundled_sp)))
            self.assertNotIn("SLICERDSPACE_BUNDLED_SITE_PACKAGES", env)

            # 2. activate_bundled_python demotes bundled-python behind Slicer site-packages on Linux
            orig_path = list(sys.path)
            orig_numpy = sys.modules.pop("numpy", None)
            orig_torch = sys.modules.pop("torch", None)
            try:
                sys.path[:] = [str(bundled_sp), str(slicer_sp)]
                with (
                    mock.patch.object(bundled_python.sys, "platform", "linux"),
                    mock.patch.object(bundled_python, "_slicer_site_packages_dirs", return_value=(slicer_sp,)),
                    mock.patch.dict(os.environ, {"SLICERDSPACE_BUNDLED_PYTHON_ROOT": str(ext_root / "bundled-python")}, clear=False),
                ):
                    bundled_python.activate_bundled_python(ext_root)
                    self.assertLess(sys.path.index(str(slicer_sp)), sys.path.index(str(bundled_sp)))
            finally:
                sys.path[:] = orig_path
                if orig_numpy is not None:
                    sys.modules["numpy"] = orig_numpy
                if orig_torch is not None:
                    sys.modules["torch"] = orig_torch


class EnvironmentDialogCudaWiringTests(unittest.TestCase):
    def test_cuda_device_only_missing_enables_prepare_and_opens_cuda_setup_dialog(self) -> None:
        from shared.environment import DependencyStatus
        from shared.ui_environment_panel import (
            environment_dialog_state,
            environment_item_button_text,
            environment_item_button_tooltip,
            should_route_environment_item_to_cuda_setup,
            should_route_prepare_all_to_cuda_setup,
        )

        cuda_item = DependencyStatus(
            key="device:segmentation:cuda:0",
            kind="device",
            label="segmentation requested device cuda:0",
            status="missing",
            required=True,
            targets=("segmentation",),
            details={"requested_device": "cuda:0", "torch_version": "2.15.0+cpu", "torch_cuda_build": None},
            install_hint="On Windows, GPU use needs the CUDA build of torch.",
            install_group="device",
        )
        other_item = DependencyStatus(
            key="path:ddseg_weights",
            kind="path",
            label="DDSeg weights directory",
            status="missing",
            required=True,
            targets=("segmentation",),
            details={"path": "/tmp/models", "path_kind": "directory"},
            install_hint="Download models.",
            install_group="model-resources",
        )
        cuda_only_result = SimpleNamespace(
            targets=("segmentation",),
            statuses=(cuda_item,),
            missing_python_packages=(),
            missing_external_groups=("device",),
            runnable_targets=(),
            blocked_targets=("segmentation",),
            blocker_count=1,
            optional_missing_count=0,
        )
        mixed_result = SimpleNamespace(
            targets=("segmentation",),
            statuses=(cuda_item, other_item),
            missing_python_packages=(),
            missing_external_groups=("device", "model-resources"),
            runnable_targets=(),
            blocked_targets=("segmentation",),
            blocker_count=2,
            optional_missing_count=0,
        )

        state = environment_dialog_state(cuda_only_result, python_install_target="Slicer Python.")
        self.assertTrue(state.can_install_python)
        self.assertEqual(environment_item_button_text(cuda_item, zh=False), "Install CUDA PyTorch")
        self.assertEqual(environment_item_button_text(cuda_item, zh=True), "安装 CUDA 版 PyTorch")
        self.assertIn("2.5", environment_item_button_tooltip(cuda_item, zh=False))
        self.assertTrue(should_route_environment_item_to_cuda_setup(cuda_item))
        self.assertFalse(should_route_environment_item_to_cuda_setup(other_item))
        self.assertTrue(should_route_prepare_all_to_cuda_setup(cuda_only_result))
        self.assertFalse(should_route_prepare_all_to_cuda_setup(mixed_result))

        # Verify shared/module_widget.py dispatches both actions to show_cuda_pytorch_setup_dialog
        mw_src = (REPO_ROOT / "shared" / "module_widget.py").read_text(encoding="utf-8")
        self.assertIn("should_route_prepare_all_to_cuda_setup(result)", mw_src)
        self.assertIn("should_route_environment_item_to_cuda_setup(item)", mw_src)
        self.assertIn("show_cuda_pytorch_setup_dialog(self)", mw_src)


if __name__ == "__main__":
    unittest.main()
