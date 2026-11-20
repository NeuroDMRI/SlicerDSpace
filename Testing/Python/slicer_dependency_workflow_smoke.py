"""Exercise dependency path/install workflow in a real Slicer UI session.

The smoke avoids mutating the host or downloading large assets. It monkeypatches
Qt path dialogs and model-download helpers so the real widget handlers, progress
dialogs and settings persistence paths are exercised with deterministic inputs.
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import qt
import slicer


TEMP_ROOT = Path(os.environ.get("SLICERDSPACE_SMOKE_ROOT", "/tmp/slicerdsurfer-dependency-workflow-smoke")) / "dependencies"
MODEL_ROOT = TEMP_ROOT / "model-assets"
MRTRIX_ROOT = TEMP_ROOT / "mrtrix" / "bin"


@dataclass
class FakeDependencyItem:
    key: str
    kind: str
    label: str
    status: str = "missing"
    required: bool = True
    targets: tuple[str, ...] = ("tractography",)
    details: dict[str, object] = field(default_factory=dict)
    install_hint: str = ""
    install_group: str = ""


class FakePlan:
    profile = "core"
    required_paths = ("segmentation/weights/dti_axial.pt",)
    asset_records = {}
    source_ids = ("slicerdsurfer_models_release",)
    external_overrides = ()
    asset_groups = ()
    terms_notices = ()


@dataclass
class FakeEnvironmentResult:
    statuses: tuple[FakeDependencyItem, ...]
    missing_python_packages: tuple[str, ...] = ()
    runnable_targets: tuple[str, ...] = ("validation",)
    blocked_targets: tuple[str, ...] = ("tractography",)
    blocker_count: int = 2
    targets: tuple[str, ...] = ("validation", "tractography")
    missing_external_groups: tuple[str, ...] = ("mrtrix",)


def _module_widget():
    try:
        return slicer.util.getModuleWidget("SlicerDSpace")
    except Exception:
        module = slicer.util.getModule("SlicerDSpace")
        return module.widgetRepresentation().self()


def _patch_file_dialogs():
    original_existing = qt.QFileDialog.getExistingDirectory

    def fake_existing(parent, title, directory=""):
        if "model" in str(title).lower():
            return MODEL_ROOT.as_posix()
        return MRTRIX_ROOT.as_posix()

    qt.QFileDialog.getExistingDirectory = fake_existing
    return original_existing


def _patch_model_downloader(widget):
    original_loader = widget._load_model_assets_module

    class FakeDownloaderModule:
        @staticmethod
        def configured_model_assets_root():
            return MODEL_ROOT

        @staticmethod
        def load_manifest(path):
            return {"manifest": Path(path).as_posix()}

        @staticmethod
        def resolve_asset_plan(manifest, profile):
            return FakePlan()

        @staticmethod
        def download_asset_plan(asset_root, plan, manifest, **kwargs):
            target = Path(asset_root) / "segmentation" / "weights"
            target.mkdir(parents=True, exist_ok=True)
            (target / "dti_axial.pt").write_text("fake smoke asset\n", encoding="utf-8")

        @staticmethod
        def verify_asset_plan(asset_root, plan, check_hashes=True):
            return {
                "ok": True,
                "checked_count": 1,
                "hashes_checked": int(bool(check_hashes)),
                "missing": [],
                "lfs_pointers": [],
                "size_mismatches": [],
                "sha256_mismatches": [],
            }

    widget._load_model_assets_module = lambda: FakeDownloaderModule
    return original_loader


def _run() -> None:
    original_existing = None
    original_loader = None
    original_info_display = None
    original_model_path = ""
    original_mrtrix_path = ""
    widget = None
    original_start_job = None
    try:
        TEMP_ROOT.mkdir(parents=True, exist_ok=True)
        MRTRIX_ROOT.mkdir(parents=True, exist_ok=True)
        (MRTRIX_ROOT / "dwi2fod").write_text("#!/bin/sh\n", encoding="utf-8")

        slicer.util.selectModule("SlicerDSpace")
        widget = _module_widget()
        original_model_path = widget._read_environment_path_setting("model_assets_dir")
        original_mrtrix_path = widget._read_environment_path_setting("mrtrix_bin_dir")
        widget.ui.validateButton.click()
        slicer.app.processEvents()

        original_existing = _patch_file_dialogs()
        original_loader = _patch_model_downloader(widget)
        import shared.ui_setup_jobs as setup_jobs
        original_start_job = setup_jobs.start_setup_job

        def controlled_start_job(owner, title, commands, on_done):
            # Exercise the real async job and widget callbacks without network
            # downloads. This tiny fake asset is only a test fixture.
            fixture = MODEL_ROOT / "segmentation/weights/dti_axial.pt"
            code = ("from pathlib import Path; import time; time.sleep(0.3); "
                    f"p=Path({str(fixture)!r}); p.parent.mkdir(parents=True,exist_ok=True); "
                    "p.write_text('fake smoke asset'); print('fixture ready')")
            command = (widget._slicer_launcher_program(), ["--launch", "PythonSlicer", "-u", "-c", code], {})
            return original_start_job(owner, title, [command], on_done)

        setup_jobs.start_setup_job = controlled_start_job

        def wait_for_job():
            deadline = time.monotonic() + 60
            while getattr(widget, "_setupJob", None) is not None and time.monotonic() < deadline:
                loop = qt.QEventLoop()
                qt.QTimer.singleShot(50, loop.quit)
                loop.exec_()
            assert getattr(widget, "_setupJob", None) is None, "Async setup timed out"
        original_info_display = slicer.util.infoDisplay
        info_messages = []

        def fake_info_display(message, *args, **kwargs):
            info_messages.append(str(message))

        slicer.util.infoDisplay = fake_info_display

        model_item = FakeDependencyItem(
            key="path:ddseg_weights",
            kind="path",
            label="DDSeg weights directory",
            details={"path": str(MODEL_ROOT / "segmentation" / "weights"), "path_kind": "nonempty_dir"},
            install_hint="Choose a model asset root.",
            install_group="model-resources",
        )
        external_item = FakeDependencyItem(
            key="command:dwi2fod",
            kind="command",
            label="dwi2fod",
            details={"command": "dwi2fod"},
            install_hint="Install MRtrix3.",
            install_group="mrtrix",
        )

        widget._lastEnvironmentResult = FakeEnvironmentResult(statuses=(model_item, external_item))
        # The combined entry is covered end-to-end in selected_setup. This
        # fixture checks dispatch without authorizing a real MRtrix install.
        with patch("shared.ui_selected_setup.prepare_selected_environment") as prepare:
            widget._onInstallAllEnvironmentItemsClicked()
            bulk_selected_dispatch = prepare.call_args.args == (widget, widget._lastEnvironmentResult)

        widget._onInstallEnvironmentItemClicked(model_item)
        wait_for_job()
        model_saved = str(
            slicer.app.userSettings().value(widget._environment_settings_key("model_assets_dir"), "")
        )
        model_line_edit = (getattr(widget, "_environmentPathLineEdits", {}) or {}).get("model_assets_dir")
        model_line_text = model_line_edit.text if model_line_edit else ""
        model_asset_file = MODEL_ROOT / "segmentation" / "weights" / "dti_axial.pt"

        with patch("shared.managed_runtimes.supported_platform", return_value=False):
            widget._onInstallEnvironmentItemClicked(external_item)
        slicer.app.processEvents()
        mrtrix_saved = str(
            slicer.app.userSettings().value(widget._environment_settings_key("mrtrix_bin_dir"), "")
        )
        mrtrix_line_edit = (getattr(widget, "_environmentPathLineEdits", {}) or {}).get("mrtrix_bin_dir")
        mrtrix_line_text = mrtrix_line_edit.text if mrtrix_line_edit else ""

        widget._initialize_environment_dependency_path_controls()
        restored_model = widget._read_environment_path_setting("model_assets_dir")
        restored_mrtrix = widget._read_environment_path_setting("mrtrix_bin_dir")
        from shared.user_config import load_user_config

        user_config = load_user_config()
        json_dependencies = user_config.get("external_dependencies", {})
        json_model = str(json_dependencies.get("model_assets_dir") or "")
        json_mrtrix = str(json_dependencies.get("mrtrix_bin_dir") or "")
        widget._save_environment_path_setting("mrtrix_bin_dir", "")
        json_clear_ready = "mrtrix_bin_dir" not in load_user_config().get("external_dependencies", {})
        widget._save_environment_path_setting("mrtrix_bin_dir", MRTRIX_ROOT.as_posix())

        print(f"SLICERDSPACE_DEP_MODEL_SAVED={model_saved}", flush=True)
        print(f"SLICERDSPACE_DEP_MODEL_LINE_EDIT={model_line_text}", flush=True)
        print(f"SLICERDSPACE_DEP_MODEL_ASSET_FILE={model_asset_file.is_file()}", flush=True)
        print(f"SLICERDSPACE_DEP_MRTRIX_SAVED={mrtrix_saved}", flush=True)
        print(f"SLICERDSPACE_DEP_MRTRIX_LINE_EDIT={mrtrix_line_text}", flush=True)
        print(f"SLICERDSPACE_DEP_RESTORED_MODEL={restored_model}", flush=True)
        print(f"SLICERDSPACE_DEP_RESTORED_MRTRIX={restored_mrtrix}", flush=True)
        print(f"SLICERDSPACE_DEP_SELECTED_SETUP_DISPATCH={bulk_selected_dispatch}", flush=True)
        print(f"SLICERDSPACE_DEP_JSON_MODEL={json_model}", flush=True)
        print(f"SLICERDSPACE_DEP_JSON_MRTRIX={json_mrtrix}", flush=True)
        print(f"SLICERDSPACE_DEP_JSON_CLEAR_READY={json_clear_ready}", flush=True)

        ok = (
            bulk_selected_dispatch
            and model_saved == MODEL_ROOT.as_posix()
            and model_line_text == MODEL_ROOT.as_posix()
            and model_asset_file.is_file()
            and mrtrix_saved == MRTRIX_ROOT.as_posix()
            and mrtrix_line_text == MRTRIX_ROOT.as_posix()
            and restored_model == MODEL_ROOT.as_posix()
            and restored_mrtrix == MRTRIX_ROOT.as_posix()
            and json_model == MODEL_ROOT.as_posix()
            and json_mrtrix == MRTRIX_ROOT.as_posix()
            and json_clear_ready
        )
        slicer.app.exit(0 if ok else 2)
    except Exception as exc:
        print(f"SLICERDSPACE_DEP_WORKFLOW_ERROR={exc}", flush=True)
        slicer.app.exit(2)
    finally:
        if original_start_job is not None:
            setup_jobs.start_setup_job = original_start_job
        if original_existing is not None:
            qt.QFileDialog.getExistingDirectory = original_existing
        if original_loader is not None:
            try:
                _module_widget()._load_model_assets_module = original_loader
            except Exception:
                pass
        if original_info_display is not None:
            slicer.util.infoDisplay = original_info_display
        if widget is not None:
            try:
                widget._save_environment_path_setting("model_assets_dir", original_model_path)
                widget._save_environment_path_setting("mrtrix_bin_dir", original_mrtrix_path)
            except Exception:
                pass
        shutil.rmtree(TEMP_ROOT, ignore_errors=True)


qt.QTimer.singleShot(6000, _run)
