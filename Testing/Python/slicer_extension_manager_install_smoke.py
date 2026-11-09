"""Install a local package through Slicer's real Extension Manager model."""

from __future__ import annotations

import os
import json
import runpy
import shutil
from pathlib import Path

import qt
import slicer


TEMP_ROOT = Path(os.environ.get("SLICERDSPACE_SMOKE_ROOT", "/tmp/slicerdsurfer-extension-manager-install-smoke")) / "extension-manager"


def _run() -> None:
    import json
    import os
    import runpy
    import shutil
    from pathlib import Path
    import qt
    import slicer

    temp_root = Path(os.environ.get("SLICERDSPACE_SMOKE_ROOT", "/tmp/slicerdsurfer-extension-manager-install-smoke")) / "extension-manager"
    try:
        archive_path = Path(os.environ["SLICERDSPACE_EXTENSION_ARCHIVE"]).expanduser().resolve()
        shutil.rmtree(temp_root, ignore_errors=True)
        temp_root.mkdir(parents=True)
        # Load only this stdlib inspector, not scientific modules from the checkout.
        script_file = globals().get("__file__")
        default_inspector = (
            Path(script_file).resolve().parents[2] / "scripts/extension_runtime/archive_layout.py"
            if script_file
            else Path.cwd() / "scripts/extension_runtime/archive_layout.py"
        )
        inspector_path = os.environ.get("SLICERDSPACE_ARCHIVE_INSPECTOR") or str(default_inspector)
        inspector = runpy.run_path(inspector_path)
        layout = inspector["inspect_archive_layout"](archive_path)
        archive_rooted = len(layout["module_roots"]) == 1 and not layout["unsafe"]
        archive_clean = not layout["development_artifacts"]

        settings_path = temp_root / "settings.ini"
        install_root = temp_root / "extensions"
        settings = qt.QSettings(settings_path.as_posix(), qt.QSettings.IniFormat)
        settings.setValue("Extensions/InstallPath", install_root.as_posix())
        settings.setValue("Extensions/ManagerEnabled", True)
        settings.sync()

        manager = slicer.app.extensionsManagerModel()
        manager.extensionsSettingsFilePath = settings_path.as_posix()
        manager.setInteractive(False)
        # Switching settings does not invalidate the manager's installed-item cache.
        manager.updateModel()
        assert not manager.isExtensionInstalled("SlicerDSpace"), "Private install root is not empty"
        metadata = {
            "extensionname": "SlicerDSpace",
            "depends": "PyTorch",
            "slicer_revision": str(manager.slicerRevision),
            "os": str(manager.slicerOs),
            "arch": str(manager.slicerArch),
        }
        installed = bool(
            manager.installExtension(
                "SlicerDSpace",
                metadata,
                archive_path.as_posix(),
                False,
                True,
            )
        )
        manager_reports_installed = bool(manager.isExtensionInstalled("SlicerDSpace"))
        module_paths = tuple(str(path) for path in manager.extensionModulePaths("SlicerDSpace"))
        module_files = [Path(path) / "SlicerDSpace.py" for path in module_paths]
        installed_module = next((path for path in module_files if path.is_file()), None)

        factory = slicer.app.moduleManager().factoryManager()
        if installed_module:
            factory.registerModule(qt.QFileInfo(installed_module.as_posix()))
        registered = bool(factory.isRegistered("SlicerDSpace"))
        loaded = bool(registered and factory.loadModules(["SlicerDSpace"]))
        module_available = hasattr(slicer.modules, "slicerdspace") or hasattr(slicer.modules, "slicerddsurfer")
        top_level_python = []
        runtime_resources_ready = False
        installed_imports_ready = False
        bundled_runtimes_ready = True
        self_tests_passed = False
        if installed_module:
            top_level_python = sorted(path.name for path in installed_module.parent.glob("*.py"))
            root = installed_module.parent.resolve()
            runtime_resources_ready = all((root / name).is_file() for name in (
                "tractography/wmql_queries/AF_CST.qry",
                "scripts/wsl/WslCommandBridge.cs", "scripts/wsl/requirements-runtime.txt",
                "LICENSE.txt", "THIRD_PARTY_NOTICES.md",
                "anatomy/parcellation/conversion/LICENSE.txt",
                "anatomy/surface/ddsurfer_backend/LICENSE.txt",
                "licenses/FreeSurfer.txt", "licenses/PyTorch3D.txt", "licenses/provenance.json",
                "anatomy/parcellation/conversion/data/FreeSurferColorLUT.txt"))
            import tractography.wmql as wmql
            import anatomy.segmentation.pipeline as ddseg
            import scripts.audit_third_party as license_audit
            import shared.native_runtimes as native_runtimes
            import shared.surface_runtime as surface_runtime
            installed_imports_ready = (
                Path(wmql.__file__).resolve().is_relative_to(root)
                and Path(ddseg.__file__).resolve().is_relative_to(root)
                and wmql.bundled_wmql_query_path().resolve().is_relative_to(root)
                and wmql.bundled_wmql_query_path().is_file()
                and Path(license_audit.__file__).resolve().is_relative_to(root)
                and license_audit.audit(root)["status"] == "pass"
            )
            if layout.get("bundled_runtimes"):
                os.environ["SLICERDSPACE_RUNTIME_ROOT"] = str(temp_root / "empty-runtime-cache")
                mrtrix_bins = native_runtimes.installed_bundle_bin_dirs("mrtrix")
                fsl_bins = native_runtimes.installed_bundle_bin_dirs("fsl")
                surf_dir = surface_runtime.bundled_surface_runtime()
                bundled_runtimes_ready = bool(
                    mrtrix_bins
                    and mrtrix_bins[0].resolve().is_relative_to(install_root.resolve())
                    and fsl_bins
                    and fsl_bins[0].resolve().is_relative_to(install_root.resolve())
                    and surf_dir is not None
                    and surf_dir.resolve().is_relative_to(install_root.resolve())
                )
            if loaded and installed_imports_ready:
                # Exercise the official Self Tests registration on the installed package.
                nodes_before = slicer.mrmlScene.GetNumberOfNodes()
                slicer.selfTests['SlicerDSpace'](msec=0)
                self_tests_passed = slicer.mrmlScene.GetNumberOfNodes() == nodes_before

        print(f"SLICERDSPACE_EXTMGR_ARCHIVE_ROOTED={archive_rooted}", flush=True)
        print(f"SLICERDSPACE_EXTMGR_ARCHIVE_CLEAN={archive_clean}", flush=True)
        print(f"SLICERDSPACE_EXTMGR_INSTALL_RETURNED={installed}", flush=True)
        print(f"SLICERDSPACE_EXTMGR_REPORTS_INSTALLED={manager_reports_installed}", flush=True)
        print(f"SLICERDSPACE_EXTMGR_MODULE_PATHS={module_paths}", flush=True)
        print(f"SLICERDSPACE_EXTMGR_MODULE_REGISTERED={registered}", flush=True)
        print(f"SLICERDSPACE_EXTMGR_MODULE_LOADED={loaded}", flush=True)
        print(f"SLICERDSPACE_EXTMGR_MODULE_AVAILABLE={module_available}", flush=True)
        print(f"SLICERDSPACE_EXTMGR_TOP_LEVEL_PY={top_level_python}", flush=True)
        print(f"SLICERDSPACE_EXTMGR_RUNTIME_RESOURCES_READY={runtime_resources_ready}", flush=True)
        print(f"SLICERDSPACE_EXTMGR_INSTALLED_IMPORTS_READY={installed_imports_ready}", flush=True)
        print(f"SLICERDSPACE_EXTMGR_BUNDLED_RUNTIMES_READY={bundled_runtimes_ready}", flush=True)
        print(f"SLICERDSPACE_EXTMGR_SELF_TESTS_PASSED={self_tests_passed}", flush=True)
        ready = (
            archive_rooted
            and archive_clean
            and installed
            and manager_reports_installed
            and installed_module is not None
            and registered
            and loaded
            and module_available
            and top_level_python == ["SlicerDSpace.py"]
            and runtime_resources_ready
            and installed_imports_ready
            and bundled_runtimes_ready
            and self_tests_passed
        )
        (temp_root / 'installation.json').write_text(json.dumps({
            'passed': ready, 'archive': str(archive_path),
            'archive_rooted': archive_rooted, 'archive_clean': archive_clean,
            'installed': installed, 'manager_reports_installed': manager_reports_installed,
            'registered': registered, 'module_available': module_available,
            'top_level_python': top_level_python,
            'installed_module': str(installed_module), 'module_loaded': loaded,
            'installed_imports_ready': installed_imports_ready,
            'runtime_resources_ready': runtime_resources_ready,
            'bundled_runtimes_ready': bundled_runtimes_ready,
            'bundled_runtimes': layout.get("bundled_runtimes", []),
            'self_tests_passed': self_tests_passed,
            'slicer_version': str(slicer.app.applicationVersion),
            'slicer_revision': str(slicer.app.repositoryRevision),
        }, indent=2) + '\n')
        slicer.app.exit(0 if ready else 2)
    except Exception as exc:
        import traceback
        temp_root.mkdir(parents=True, exist_ok=True)
        (temp_root / 'installation.json').write_text(json.dumps({
            'passed': False,
            'error': str(exc),
            'traceback': traceback.format_exc(),
        }, indent=2) + '\n')
        print(f"SLICERDSPACE_EXTMGR_ERROR={exc}", flush=True)
        slicer.app.exit(2)


qt.QTimer.singleShot(3000, _run)
