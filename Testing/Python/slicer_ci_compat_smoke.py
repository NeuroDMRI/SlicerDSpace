"""Headless compatibility smoke for the CI matrix (real, stock 3D Slicer).

Run through Slicer, for example:

    Slicer --no-splash --no-main-window --python-script Testing/Python/slicer_ci_compat_smoke.py

Settings come from environment variables (set by .github/workflows/slicer-compat-matrix.yml):

    SLICERDSPACE_CI_MODULE_DIR  directory that holds SlicerDSpace.py (the code under test)
    SLICERDSPACE_CI_REPORT      path of the JSON report (written after every check)
    SLICERDSPACE_CI_WORK        scratch directory for the synthetic subject and logs
    SLICERDSPACE_CI_MODE        "repo-only" or "full-install" (recorded in the report)
    SLICERDSPACE_CI_PIPELINE    "required" | "optional" | "skip"
                                required = the FA map must exist and have the expected values.
                                optional = (repo-only layout, no bundled-python) the product's
                                documented partial execution is accepted: exit 0 with the target
                                "blocked" (preflight missing dependencies) in the orchestration
                                manifest, when every named package is really absent in this Slicer.
                                A worker crash caused by a missing package is "skipped"; every
                                other failure still fails.

The process exits through slicer.util.exit(0|1). The report, not the exit code alone, is the
result: the workflow fails when the report is missing or not "pass".
A silent worker only produces a warning; nothing here kills a process.
"""
from __future__ import annotations

import gzip
import importlib
import json
import os
import platform
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPORT_SCHEMA = "slicerdspace_ci_compat_smoke_v1"
TOP_PACKAGES = ("SlicerDSpace", "shared", "preprocessing", "anatomy", "microstructure", "tractography", "connectome")
SILENT_WARN_SECONDS = 120.0
EXPECTED_FA = 0.763  # tensor diag(1.7, 0.4, 0.3) e-3 mm2/s -> FA = 0.7634
FA_TOLERANCE = 0.05
SUBJECT_ID = "ci"
TASK_NAME = "microstructure.dti"
VERSION_PACKAGES = ("numpy", "torch", "dipy", "nibabel", "nrrd", "vtk")

REPORT = {
    "schema": REPORT_SCHEMA,
    "status": "running",
    "mode": os.environ.get("SLICERDSPACE_CI_MODE", ""),
    "host": {"system": platform.system(), "machine": platform.machine(), "python": sys.version.split()[0]},
    "checks": [],
}


def _paths():
    module_dir = Path(os.environ.get("SLICERDSPACE_CI_MODULE_DIR", "")).expanduser()
    work = Path(os.environ.get("SLICERDSPACE_CI_WORK", "")).expanduser() if os.environ.get("SLICERDSPACE_CI_WORK") else None
    report = Path(os.environ.get("SLICERDSPACE_CI_REPORT", "")).expanduser() if os.environ.get("SLICERDSPACE_CI_REPORT") else None
    return module_dir, work, report



def say(message, flush=True):
    """Print and append to <work>/ci_smoke.log (Slicer.exe on Windows has no usable console)."""
    print(message, flush=flush)
    _module_dir, work, _report = _paths()
    if work is not None:
        try:
            work.mkdir(parents=True, exist_ok=True)
            with open(str(work / "ci_smoke.log"), "a", encoding="utf-8") as handle:
                handle.write(message + "\n")
        except OSError:
            pass


def write_report():
    _module_dir, _work, report = _paths()
    if report is None:
        return
    report.parent.mkdir(parents=True, exist_ok=True)
    tmp = report.with_name(report.name + ".tmp")
    tmp.write_text(json.dumps(REPORT, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(str(tmp), str(report))


def record(name, status, detail="", **extra):
    entry = {"name": name, "status": status, "detail": str(detail)}
    entry.update(extra)
    REPORT["checks"].append(entry)
    say("[ci-smoke] %-5s %s %s" % (status.upper(), name, detail), flush=True)
    write_report()
    return entry


def run_check(name, function):
    """Run one check; a raised exception becomes a failed entry with the traceback."""
    say("[ci-smoke] START %s" % name, flush=True)
    try:
        result = function()
    except Exception as exc:  # noqa: BLE001 - every failure must reach the report
        record(name, "fail", "%s: %s" % (type(exc).__name__, exc), traceback=traceback.format_exc())
        return None
    return result


# --- pure helpers (unit-tested without Slicer) -------------------------------------------------

def is_shadow_module_path(path, module_dir):
    """True when sys.path entry is an installed qt-scripted-modules dir other than module_dir."""
    if not path:
        return False
    try:
        candidate = Path(path).resolve()
        target = Path(module_dir).resolve()
    except OSError:
        return False
    if candidate == target:
        return False
    parts = candidate.parts
    # An installed SlicerDSpace copy: its folder is named SlicerDSpace, or it holds SlicerDSpace.py.
    # Other extensions (for example SlicerDMRI) are left alone.
    return "qt-scripted-modules" in parts and ("SlicerDSpace" in parts or (candidate / "SlicerDSpace.py").is_file())


def purge_shadow_paths(sys_path, module_dir):
    """Return (new sys.path with module_dir first, removed shadow entries)."""
    removed = [p for p in sys_path if is_shadow_module_path(p, module_dir)]
    target = str(module_dir)
    kept = [p for p in sys_path if p not in removed and p != target]
    return [target] + kept, removed


def _under(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except (ValueError, OSError):
        return False


def write_nifti_gz(path, data, voxel=2.0):
    """Write a float32 or uint8 NIfTI-1 file with numpy and struct only (no nibabel needed)."""
    import numpy as np

    data = np.asarray(data)
    codes = {"float32": (16, 32), "uint8": (2, 8)}
    if data.dtype.name not in codes:
        raise ValueError("unsupported dtype " + data.dtype.name)
    datatype, bitpix = codes[data.dtype.name]
    header = bytearray(348)
    struct.pack_into("<i", header, 0, 348)
    struct.pack_into("<8h", header, 40, *([data.ndim] + list(data.shape) + [1] * (7 - data.ndim)))
    struct.pack_into("<h", header, 70, datatype)
    struct.pack_into("<h", header, 72, bitpix)
    struct.pack_into("<8f", header, 76, *([1.0] + [voxel] * 3 + [1.0] * 4))
    struct.pack_into("<f", header, 108, 352.0)
    struct.pack_into("<h", header, 254, 1)  # sform_code
    struct.pack_into("<4f", header, 280, voxel, 0.0, 0.0, 0.0)
    struct.pack_into("<4f", header, 296, 0.0, voxel, 0.0, 0.0)
    struct.pack_into("<4f", header, 312, 0.0, 0.0, voxel, 0.0)
    header[344:348] = b"n+1\0"
    with gzip.open(str(path), "wb") as handle:
        handle.write(bytes(header) + b"\0\0\0\0" + np.asfortranarray(data).tobytes(order="F"))


def read_nifti_gz(path):
    """Read a NIfTI-1 .nii.gz written by little-endian tools (float32/float64/uint8/int16)."""
    import numpy as np

    raw = gzip.open(str(path), "rb").read()
    if struct.unpack_from("<i", raw, 0)[0] != 348:
        raise ValueError("not a little-endian NIfTI-1 file: " + str(path))
    dim = struct.unpack_from("<8h", raw, 40)
    datatype = struct.unpack_from("<h", raw, 70)[0]
    offset = int(struct.unpack_from("<f", raw, 108)[0])
    slope, inter = struct.unpack_from("<2f", raw, 112)
    dtypes = {2: "u1", 4: "<i2", 8: "<i4", 16: "<f4", 64: "<f8"}
    if datatype not in dtypes:
        raise ValueError("unsupported NIfTI datatype %d" % datatype)
    shape = tuple(dim[1:1 + dim[0]])
    count = 1
    for n in shape:
        count *= n
    array = np.frombuffer(raw, dtype=dtypes[datatype], count=count, offset=offset).reshape(shape, order="F")
    array = array.astype("float64")
    if slope and slope != 1.0:
        array = array * slope + inter
    return array


def make_synthetic_subject(root):
    """Write input/{data.nii.gz,bvals,bvecs,nodif_brain_mask.nii.gz}: 16x16x8, 1 b0 + 6 directions."""
    import numpy as np

    inp = Path(root) / "input"
    inp.mkdir(parents=True, exist_ok=True)
    directions = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], float)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    bvecs = np.vstack([[0.0, 0.0, 0.0], directions])
    bvals = np.array([0] + [1000] * 6, float)
    tensor = np.diag([1.7e-3, 0.4e-3, 0.3e-3])
    shape = (16, 16, 8)
    volumes = [np.full(shape, 1000.0) * np.exp(-b * float(g @ tensor @ g)) for b, g in zip(bvals, bvecs)]
    write_nifti_gz(inp / "data.nii.gz", np.stack(volumes, axis=-1).astype("float32"))
    mask = np.zeros(shape, "uint8")
    mask[2:14, 2:14, 1:7] = 1
    write_nifti_gz(inp / "nodif_brain_mask.nii.gz", mask)
    (inp / "bvals").write_text(" ".join(str(int(b)) for b in bvals) + "\n", encoding="utf-8")
    (inp / "bvecs").write_text("\n".join(" ".join("%.6f" % v[i] for v in bvecs) for i in range(3)) + "\n", encoding="utf-8")
    return inp


def classify_worker_failure(output_text):
    """Return the missing third-party package name when the worker died on an import, else ''."""
    for line in reversed(output_text.splitlines()):
        line = line.strip()
        if line.startswith(("ModuleNotFoundError", "ImportError")) and "'" in line:
            return line.split("'")[1].split(".")[0]
    return ""


def read_task_outcome(subject_root, task_name=TASK_NAME):
    """Return (status, reason) of one task from the newest *_orchestration_manifest.json under subject_root.

    The CLI exits 0 for a "blocked" target (documented partial execution: --strict-environment is the
    switch that aborts instead), so the manifest, not the exit code, tells whether the target ran.
    Returns ("", "no orchestration manifest ...") when there is no readable manifest or task."""
    manifests = sorted(Path(subject_root).rglob("*_orchestration_manifest.json"), key=lambda p: p.stat().st_mtime)
    if not manifests:
        return "", "no *_orchestration_manifest.json under %s" % subject_root
    try:
        manifest = json.loads(manifests[-1].read_text(encoding="utf-8"))
    except ValueError as exc:
        return "", "manifest %s is not valid JSON: %s" % (manifests[-1], exc)
    for task in manifest.get("tasks", []):
        if task.get("name") == task_name:
            return str(task.get("status", "")), str(task.get("reason") or "")
    return "", "manifest %s has no task %s (run status %s)" % (manifests[-1].name, task_name, manifest.get("status"))


def missing_packages_from_reason(reason):
    """'preflight missing dependencies: dipy (not found), nibabel (not found)' -> ['dipy', 'nibabel'], else []."""
    marker = "preflight missing dependencies:"
    if marker not in reason:
        return []
    names = []
    for part in reason.split(marker, 1)[1].split(","):
        name = part.strip().split(" ")[0].strip()
        if name:
            names.append(name)
    return names


def judge_blocked_task(status, reason, versions):
    """Return (accepted, detail) for a target that produced no output while the CLI exited 0.

    Accepted only when the task is "blocked" for missing dependencies and each named package is one the
    versions check probed and found absent in this Slicer (so the block is real, not a hidden fault)."""
    if status != "blocked":
        return False, "task status is %r (reason: %s)" % (status, reason)
    missing = missing_packages_from_reason(reason)
    if not missing:
        return False, "blocked, but not for missing dependencies: %s" % reason
    unproven = [m for m in missing if m not in versions or versions[m].get("present")]
    if unproven:
        return False, "blocked for %s, but the versions check does not show %s absent (probed: %s)" % (
            missing, unproven, sorted(versions))
    return True, "blocked as documented (exit 0, partial execution): %s" % reason


# --- checks (need Slicer) ----------------------------------------------------------------------

def check_settings_isolation():
    import slicer

    settings = [slicer.app.userSettings().fileName(), slicer.app.revisionUserSettings().fileName()]
    REPORT["slicer_settings_files"] = settings
    record("settings_files", "info", "; ".join(settings))


def check_slicer_identity():
    import slicer

    em = slicer.app.extensionsManagerModel()
    REPORT["slicer"] = {
        "version": slicer.app.applicationVersion,
        "revision": str(em.slicerRevision),
        "os": str(em.slicerOs),
        "arch": str(em.slicerArch),
        "home": str(slicer.app.slicerHome),
        "python": sys.version.split()[0],
        "executable": str(slicer.app.applicationFilePath()),
    }
    record("slicer_identity", "info", json.dumps(REPORT["slicer"], sort_keys=True))


def check_import_module(module_dir):
    """Purge installed copies from sys.path, import from the code under test, prove where it came from."""
    new_path, removed = purge_shadow_paths(list(sys.path), module_dir)
    sys.path[:] = new_path
    dropped = []
    for package in TOP_PACKAGES:
        loaded = sys.modules.get(package)
        location = getattr(loaded, "__file__", None) if loaded is not None else None
        if loaded is not None and location and not _under(location, module_dir):
            for key in [k for k in sys.modules if k == package or k.startswith(package + ".")]:
                dropped.append(key)
                del sys.modules[key]
    importlib.invalidate_caches()
    module = importlib.import_module("SlicerDSpace")
    import shared  # noqa: F401  (the runtime tree must come from the same root)
    from shared import code_updater

    locations = {"SlicerDSpace": str(Path(module.__file__).resolve()), "shared": str(Path(shared.__file__).resolve())}
    outside = {k: v for k, v in locations.items() if not _under(v, module_dir)}
    REPORT["module"] = {
        "module_dir": str(module_dir),
        "locations": locations,
        "removed_sys_path_entries": removed,
        "dropped_sys_modules": sorted(dropped),
        "extension_version": getattr(code_updater, "EXTENSION_VERSION", ""),
        "runtime_abi_version": getattr(code_updater, "RUNTIME_ABI_VERSION", None),
    }
    if outside:
        record("import_module", "fail", "code under test is not what Python imported: %s (expected under %s); "
               "remove the shadowing install or fix sys.path" % (outside, module_dir))
        return False
    record("import_module", "pass", "SlicerDSpace and shared import from %s (purged %d shadow path(s))" % (module_dir, len(removed)))
    return True


def check_registered_module(module_dir):
    """The module that Slicer registered must also be the code under test (otherwise the widget tests the wrong code)."""
    import slicer

    registered = getattr(slicer.modules, "slicerdspace", None)
    if registered is None:
        record("registered_module", "fail", "Slicer did not register SlicerDSpace; start Slicer with "
               "--additional-module-paths <module dir> (repo-only) or install the extension (full-install)")
        return False
    path = str(getattr(registered, "path", "") or "")
    REPORT.setdefault("module", {})["registered_path"] = path
    if path and not _under(path, module_dir):
        record("registered_module", "fail", "Slicer registered %s, not the code under test %s" % (path, module_dir))
        return False
    record("registered_module", "pass", path)
    return True


def check_build_widget():
    import slicer

    try:
        from shared.user_config import update_user_config

        update_user_config({"ui": {"first_run_wizard_shown": True}})
    except Exception as exc:  # noqa: BLE001 - best effort: only suppresses a dialog
        say("[ci-smoke] note: could not pre-set the first-run flag: %s" % exc, flush=True)
    module = slicer.util.getModule("SlicerDSpace")
    try:
        widget = module.widgetRepresentation().self()
    except ImportError as exc:
        missing = getattr(exc, "name", "") or ""
        if os.environ.get("SLICERDSPACE_CI_MODE") == "repo-only" and missing.split(".")[0] not in TOP_PACKAGES:
            record("build_widget", "skipped", "the widget imports third-party package '%s', which the repo-only layout does "
                   "not bundle (full-install must build it)" % missing)
            return None
        raise
    slicer.app.processEvents()
    ready = {
        "logic": bool(getattr(widget, "logic", None)),
        "parameter_node": bool(getattr(widget, "_parameterNode", None)),
        "ui": bool(getattr(widget, "ui", None)),
    }
    if not all(ready.values()):
        record("build_widget", "fail", "widget is incomplete: %s" % ready)
        return None
    record("build_widget", "pass", "widget built: %s" % ready)
    return widget


def _version_of(name):
    try:
        mod = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - absent or broken both mean "not usable"
        return {"present": False, "error": "%s: %s" % (type(exc).__name__, exc)}
    return {"present": True, "version": str(getattr(mod, "__version__", "")), "file": str(getattr(mod, "__file__", ""))}


def check_versions():
    versions = {name: _version_of(name) for name in VERSION_PACKAGES}
    try:
        import slicer

        versions["slicer_vtk"] = {"present": True, "version": str(slicer.app.applicationVersion)}
    except Exception:  # noqa: BLE001
        pass
    REPORT["versions"] = versions
    for name in VERSION_PACKAGES:
        info = versions[name]
        say("[ci-smoke] version %-6s %s" % (name, ("%s  %s" % (info.get("version"), info.get("file"))) if info["present"] else "absent (%s)" % info.get("error")), flush=True)
    record("versions", "info", "numpy=%s numpy.__file__=%s" % (versions["numpy"].get("version"), versions["numpy"].get("file")))


def check_environment_doctor(module_dir):
    import slicer
    from shared.environment import EnvironmentDoctorConfig, run_environment_doctor

    python = Path(str(slicer.app.slicerHome)) / "bin" / ("PythonSlicer.exe" if os.name == "nt" else "PythonSlicer")
    config = EnvironmentDoctorConfig(
        repo_root=Path(module_dir),
        targets=("microstructure", "tractography", "segmentation"),
        python_executable=python,
        slicer_executable=Path(str(slicer.app.applicationFilePath())),
    )
    doctor = run_environment_doctor(config)
    rows = [{"key": s.key, "label": s.label, "status": s.status, "required": s.required, "targets": list(s.targets)}
            for s in doctor.statuses]
    for row in rows:
        say("[ci-smoke][doctor] %-10s required=%-5s %s (%s)" % (row["status"], row["required"], row["label"], row["key"]), flush=True)
    REPORT["doctor"] = {"blocker_count": doctor.blocker_count, "runnable_targets": list(doctor.runnable_targets),
                        "blocked_targets": list(doctor.blocked_targets), "rows": rows,
                        "missing_python_packages": list(doctor.missing_python_packages)}
    record("environment_doctor", "pass", "%d rows, blockers=%d, runnable=%s" % (len(rows), doctor.blocker_count, list(doctor.runnable_targets)))


def _run_worker(command, log_path, env):
    """Run one worker without a kill: warn when silent, report a non-zero exit at once."""
    with open(str(log_path), "wb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        started = last_growth = time.monotonic()
        last_size = 0
        warned = 0
        while True:
            code = process.poll()
            size = os.path.getsize(str(log_path))
            now = time.monotonic()
            if size != last_size:
                last_size, last_growth, warned = size, now, 0
            elif code is None and now - last_growth > SILENT_WARN_SECONDS * (warned + 1):
                warned += 1
                say("[ci-smoke] WARNING: worker silent for %.0f s but alive (pid %s); not killing it; "
                      "check %s" % (now - last_growth, process.pid, log_path), flush=True)
            if code is not None:
                return code, now - started
            time.sleep(0.5)


def check_pipeline(module_dir, work):
    policy = os.environ.get("SLICERDSPACE_CI_PIPELINE", "required").strip().lower()
    if policy == "skip":
        record("pipeline_dti", "skipped", "SLICERDSPACE_CI_PIPELINE=skip")
        return
    import slicer

    subject = work / "subject"
    make_synthetic_subject(subject)
    cli = Path(module_dir) / "shared" / "run_subject_pipeline_cli.py"
    # bin/PythonSlicer is what the installer and the environment doctor use. "Slicer --launch" is not
    # used: on macOS Contents/MacOS/Slicer is the real application, which would load the arguments as files.
    python = Path(str(slicer.app.slicerHome)) / "bin" / ("PythonSlicer.exe" if os.name == "nt" else "PythonSlicer")
    if not python.is_file():
        record("pipeline_dti", "fail", "PythonSlicer not found at %s; the Slicer install is incomplete" % python)
        return
    command = [str(python), str(cli),
               "--subject-root", str(subject), "--subject-id", SUBJECT_ID, "--targets", "microstructure",
               "--microstructure-stages", "dti", "--no-user-config", "--execute", "--print-manifest-only"]
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["SLICERDSPACE_USER_CONFIG"] = str(work / "worker_user_config.json")
    log_path = work / "pipeline_dti.log"
    say("[ci-smoke] worker command: %s" % " ".join(command), flush=True)
    code, seconds = _run_worker(command, log_path, env)
    output = log_path.read_text(encoding="utf-8", errors="replace")
    tail = "\n".join(output.splitlines()[-25:])
    REPORT["pipeline"] = {"exit_code": code, "seconds": round(seconds, 1), "log": str(log_path), "policy": policy}
    if code != 0:
        missing = classify_worker_failure(output)
        cause = "worker exited with code %d" % code
        if missing:
            cause += "; the worker could not import '%s'" % missing
        if missing and policy == "optional":
            record("pipeline_dti", "skipped", cause + " (repo-only layout has no bundled-python; install the base asset to run this)", tail=tail)
        else:
            record("pipeline_dti", "fail", cause + "; see the worker log tail", tail=tail)
        return
    maps = sorted(subject.rglob("%s_FA.nii.gz" % SUBJECT_ID))
    if not maps:
        status, reason = read_task_outcome(subject)
        REPORT["pipeline"].update(task_status=status, task_reason=reason)
        if policy == "optional":
            accepted, detail = judge_blocked_task(status, reason, REPORT.get("versions", {}))
            if accepted:
                record("pipeline_dti", "pass", detail, tail=tail)
            else:
                record("pipeline_dti", "fail", "worker exited 0 without %s_FA.nii.gz and the block is not the documented one: %s" % (SUBJECT_ID, detail), tail=tail)
            return
        record("pipeline_dti", "fail", "worker exited 0 but no %s_FA.nii.gz exists under %s; task %s is %r: %s" % (
            SUBJECT_ID, subject, TASK_NAME, status, reason), tail=tail)
        return
    import numpy as np

    fa = read_nifti_gz(maps[0])
    inside = fa[fa > 0]
    mean_fa = float(inside.mean()) if inside.size else float("nan")
    REPORT["pipeline"].update(fa_path=str(maps[0]), fa_mean=mean_fa, fa_voxels=int(inside.size))
    if not (inside.size and np.isfinite(mean_fa) and abs(mean_fa - EXPECTED_FA) <= FA_TOLERANCE):
        record("pipeline_dti", "fail", "FA map exists but mean FA %.3f is not %.3f +/- %.2f (wrong numbers)" % (mean_fa, EXPECTED_FA, FA_TOLERANCE))
        return
    record("pipeline_dti", "pass", "FA map %s, mean FA %.3f, %.1f s" % (maps[0].name, mean_fa, seconds))


# --- driver ------------------------------------------------------------------------------------

def finish():
    failed = [c["name"] for c in REPORT["checks"] if c["status"] == "fail"]
    REPORT["status"] = "fail" if failed else "pass"
    REPORT["failed_checks"] = failed
    write_report()
    say("[ci-smoke] RESULT %s%s" % (REPORT["status"].upper(), (" failed=" + ",".join(failed)) if failed else ""), flush=True)
    return 1 if failed else 0


def main():
    module_dir, work, report = _paths()
    if not str(module_dir) or not (module_dir / "SlicerDSpace.py").is_file():
        record("environment", "fail", "SLICERDSPACE_CI_MODULE_DIR must point at a directory with SlicerDSpace.py (got %r)" % str(module_dir))
        return finish()
    if work is None or report is None:
        record("environment", "fail", "SLICERDSPACE_CI_WORK and SLICERDSPACE_CI_REPORT are required")
        return finish()
    work.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SLICERDSPACE_USER_CONFIG", str(work / "user_config.json"))
    run_check("settings_files", check_settings_isolation)
    run_check("slicer_identity", check_slicer_identity)
    imported = run_check("import_module", lambda: check_import_module(module_dir))
    run_check("versions", check_versions)
    if imported:
        registered = run_check("registered_module", lambda: check_registered_module(module_dir))
        if registered:
            run_check("build_widget", check_build_widget)
        run_check("environment_doctor", lambda: check_environment_doctor(module_dir))
        run_check("pipeline_dti", lambda: check_pipeline(module_dir, work))
    return finish()


def _exit(code):
    import slicer

    try:
        slicer.util.exit(code)
    except Exception:  # noqa: BLE001 - older API
        slicer.app.exit(code)


def _run_and_exit():
    try:
        code = main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        REPORT["status"] = "fail"
        REPORT["fatal"] = traceback.format_exc()
        try:
            write_report()
        except Exception:  # noqa: BLE001
            pass
        code = 1
    _exit(code)


if __name__ == "__main__":
    import qt

    # Let Slicer finish startup (module registration) before the checks run.
    qt.QTimer.singleShot(1500, _run_and_exit)
