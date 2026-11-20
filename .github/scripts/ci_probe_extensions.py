"""Slicer script (run through Slicer, not Python): print what the extension manager believes. Read-only.

Usage: Slicer --no-splash --no-main-window --python-script ci_probe_extensions.py
Writes one JSON line to the file named by $CI_PROBE_OUT (a GUI-subsystem Slicer.exe has no console).
"""
import json
import os

import qt
import slicer


def _value(thing):
    return thing() if callable(thing) else thing


def _probe():
    result = {"cwd": os.getcwd()}
    try:
        em = slicer.app.extensionsManagerModel()
        result["install_path"] = str(_value(em.extensionsInstallPath))
        result["installed_extensions"] = [str(n) for n in _value(em.installedExtensions)]
        result["scheduled_for_uninstall"] = [str(n) for n in _value(em.scheduledForUninstallExtensions)]
        result["is_installed_SlicerDSpace"] = bool(em.isExtensionInstalled("SlicerDSpace"))
        result["is_loaded_SlicerDSpace"] = bool(em.isExtensionLoaded("SlicerDSpace"))
        result["settings_file"] = str(_value(em.extensionsSettingsFilePath))
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        result["error"] = "%s: %s" % (type(exc).__name__, exc)
    line = json.dumps(result, sort_keys=True)
    print("[ci-probe] " + line, flush=True)
    target = os.environ.get("CI_PROBE_OUT")
    if target:
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(line + "\n")
    slicer.app.exit(0)


qt.QTimer.singleShot(500, _probe)
