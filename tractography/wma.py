"""WhiteMatterAnalysis integration for tractography post-processing.

This module intentionally keeps WMA execution behind an explicit ``execute``
flag so the extension can construct reproducible commands and manifests before
running long tractography clustering jobs.
"""

from __future__ import annotations

from shared.cohort_schedule import scheduled_stage

import json
import os
import shutil
import subprocess
import tempfile
import time
import re
import traceback
import shlex
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Optional

from shared.external_process import (
    external_python_environment,
    find_pythonslicer,
    text_subprocess_kwargs,
    slicer_runtime_roots,
)
from shared.output_contracts import vtp_streamline_count
from shared.product_staging import archive_run_records, new_product_staging, publish_products, retain_failed_products
from tractography.wma_atlases import (atlas_definition, atlas_output_contract, detect_atlas_family,
                                      validate_group_hemispheres)


WMA_APPLY_ORG_SCRIPT = Path("bin") / "wm_apply_ORG_atlas_to_subject.sh"
WMA_APPLY_ORG_SCRIPT_NAME = "wm_apply_ORG_atlas_to_subject.sh"
WMA_REPOSITORY = "https://github.com/SlicerDMRI/whitematteranalysis.git"
EXPECTED_ORG_ATLAS_DIRS = ("ORG-RegAtlas-100HCP", "ORG-800FC-100HCP")
MIN_ANATOMICAL_TRACT_COUNT = 73
# A successful process can still emit every atlas filename with no fibers.
# Requiring three quarters of the atlas tracts to be non-empty is deliberately
# conservative while catching under-sampled or spatially invalid tractograms.
MIN_NONEMPTY_ANATOMICAL_TRACT_COUNT = 55
REQUIRED_MAJOR_TRACTS = ('T_AF_left.vtp', 'T_AF_right.vtp', 'T_CST_left.vtp',
                         'T_CST_right.vtp', 'T_CC4.vtp')
MIN_MAJOR_TRACT_STREAMLINES = 20
WMA_QC_VERSION = 2


def tract_sampling_eligibility(counts: dict[str, Optional[int]]) -> dict[str, Any]:
    """Never infer suitability of all atlas tracts from five major tracts."""
    records = {name: {"streamlines": count,
                      "sampling_passed": count is not None and count >= MIN_MAJOR_TRACT_STREAMLINES,
                      "quantification_eligible": False,
                      "reason": ("unreadable" if count is None else "insufficient_sampling"
                                 if count < MIN_MAJOR_TRACT_STREAMLINES else "anatomical_review_pending")}
               for name, count in sorted(counts.items())}
    return {"minimum_streamlines": MIN_MAJOR_TRACT_STREAMLINES,
            "scope": "Sampling sanity only, not statistical power or anatomical accuracy. No automatic upsampling or duplication.",
            "tracts": records,
            "all_tract_sampling_passed": bool(records) and all(item["sampling_passed"] for item in records.values()),
            "excluded_from_quantification": [name for name, item in records.items() if not item["sampling_passed"]],
            "human_review": "pending"}
REQUIRED_WMA_COMMANDS = (
    "wm_append_clusters_to_anatomical_tracts.py",
    "wm_assess_cluster_location_by_hemisphere.py",
    "wm_cluster_from_atlas.py",
    "wm_cluster_remove_outliers.py",
    "wm_harden_transform.py",
    "wm_register_to_atlas_new.py",
    "wm_separate_clusters_by_hemisphere.py",
)
NATIVE_WMA_COMMANDS = tuple(name for name in REQUIRED_WMA_COMMANDS if name != "wm_harden_transform.py")
OPTIONAL_WMA_COMMANDS = ("wm_diffusion_measurements.py", "xvfb-run")
EXTRA_TOOL_DIRS = (
    Path("/data/chengjin/software/xvfb/usr/bin"),
)
EXTRA_LIBRARY_DIRS = (
    Path("/data/chengjin/software/libpcre2-16-0/usr/lib/x86_64-linux-gnu"),
    Path("/data/chengjin/software/libxcb-xinerama0/usr/lib/x86_64-linux-gnu"),
)


def _file_info(path: Path) -> dict[str, object]:
    expanded = path.expanduser()
    if not expanded.exists():
        return {"path": expanded.as_posix(), "exists": False}
    stat = expanded.stat()
    return {
        "path": expanded.resolve().as_posix(),
        "exists": True,
        "is_file": expanded.is_file(),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


@dataclass(frozen=True)
class WMAOrgAtlasConfig:
    """Configuration for applying a supported WMA atlas to one tractogram."""

    subject_id: str
    input_tractography: Path
    atlas_dir: Path
    output_dir: Path
    slicer_executable: Path
    wma_root: Optional[Path] = None
    mrtrix_bin_dir: Optional[Path] = None
    transform_path: Optional[Path] = None
    registration_mode: str = "rig"
    num_threads: Optional[int] = None
    use_virtual_x: bool = False
    export_diffusion_measures: bool = False
    fiber_tract_measurements_cli: Optional[Path] = None
    clean_files: int = 0
    execute: bool = False
    registration_preprocess_cache: bool = True
    atlas_family: str = "auto"
    device: str = "auto"
    # Keep the original optional export until the new method is explicitly selected.
    measurement_backend: str = "fibertractmeasurements"


def _wma_supports_gpu_flag(root: Path) -> bool:
    return (root / "whitematteranalysis" / "gpu" / "__init__.py").is_file()


def _normalize_wma_gpu_mode(device: str) -> str:
    normalized = (device or "auto").strip().lower()
    if normalized == "auto":
        return "auto"
    if normalized in {"cpu", "off", "false", "0"}:
        return "off"
    if normalized in {"cuda", "gpu", "on", "true", "1"}:
        return "on"
    raise ValueError("WMA device must be 'auto', 'cpu', or 'cuda'.")


@dataclass(frozen=True)
class WMAOrgAtlasResult:
    """Prepared or completed WMA atlas run."""

    subject_id: str
    status: str
    output_dir: Path
    manifest_path: Path
    qc_path: Path
    command: tuple[str, ...]
    executed: bool
    backend_status: dict[str, Any]
    atlas_status: dict[str, Any]
    return_code: Optional[int] = None


def _wma_script_path(root: Path) -> Path:
    script = root / WMA_APPLY_ORG_SCRIPT
    windows_binary = script.with_name(script.name + '.exe')
    return windows_binary if sys.platform == 'win32' and windows_binary.is_file() else script


def _configured_wma_root() -> Optional[Path]:
    for env_name in ("SLICERDSPACE_WMA_ROOT", "WMA_ROOT"):
        env_value = os.environ.get(env_name)
        if env_value:
            return Path(env_value).expanduser().resolve()
    return None


def _wma_root_on_path() -> Optional[Path]:
    script = shutil.which(WMA_APPLY_ORG_SCRIPT_NAME)
    if script:
        parent = Path(script).resolve().parent
        return parent.parent if parent.name == "bin" else parent
    return None


def _has_wma_bridge(root: Path) -> bool:
    shell = root / WMA_APPLY_ORG_SCRIPT
    bridge = shell.with_name(shell.name + ".exe")
    return bridge.is_file() and Path(str(bridge) + ".slicerdsurfer-owner").is_file()


def resolve_wma_execution_backend(
    wma_root: Optional[Path],
    *,
    atlas_family: str = "org",
    export_diffusion_measures: bool = False,
    measurement_backend: str = "fibertractmeasurements",
    platform: Optional[str] = None,
) -> str:
    """Select an ORG native source or the existing author/WSL route."""
    choice = os.environ.get("DSP_WMA_EXECUTION_BACKEND", "auto").strip().lower()
    if choice not in {"auto", "native", "legacy"}:
        raise ValueError("DSP_WMA_EXECUTION_BACKEND must be auto, native or legacy.")
    needs_legacy_measurements = export_diffusion_measures and measurement_backend == "fibertractmeasurements"
    if choice == "native":
        if atlas_family != "org" or needs_legacy_measurements:
            raise ValueError("Native WMA supports ORG without the legacy FiberTractMeasurements export.")
        return choice
    if (choice == "legacy" or (platform or sys.platform).lower() not in {"win32", "windows"}
            or atlas_family != "org" or needs_legacy_measurements):
        return "legacy"
    root = wma_root or _configured_wma_root() or _wma_root_on_path()
    return "legacy" if root is not None and _has_wma_bridge(root.expanduser()) else "native"


def resolve_configured_wma_root(
    *, execution_backend: str = "legacy", platform: Optional[str] = None,
) -> Path:
    """Resolve a configured root, PATH tool, or verified source cache."""
    configured = _configured_wma_root()
    if configured is not None:
        return configured

    native = execution_backend == "native"
    path_root = _wma_root_on_path()
    if not native and path_root is not None:
        return path_root
    if native or (platform or sys.platform).lower() not in {"win32", "windows"}:
        from shared.backend_sources import available_source
        prepared = available_source("wma")
        if prepared is not None:
            return prepared.expanduser().resolve()
    if native and path_root is not None and not _has_wma_bridge(path_root):
        return path_root

    raise FileNotFoundError(
        "WMA source was not found. Prepare WMA or configure wma_root, "
        "SLICERDSPACE_WMA_ROOT or WMA_ROOT. WMA is an optional external backend."
    )


def resolve_wma_root(wma_root: Optional[Path] = None, module_dir: Optional[Path] = None) -> Path:
    del module_dir  # Backward-compatible signature; WMA discovery is no longer repo-relative.
    candidate = (wma_root.expanduser().resolve() if wma_root is not None else resolve_configured_wma_root())
    script = _wma_script_path(candidate)
    if not script.is_file():
        raise FileNotFoundError(
            f"WMA ORG atlas script not found: {script}. "
            "Configure an external WhiteMatterAnalysis checkout with wma_root, "
            "SLICERDSPACE_WMA_ROOT, WMA_ROOT, or PATH."
        )
    return candidate


def wma_source_info(wma_root: Path) -> dict[str, str]:
    git_dir = wma_root / ".git"
    info = {
        "root": wma_root.as_posix(),
        "repository": WMA_REPOSITORY,
    }
    if git_dir.exists():
        try:
            commit = subprocess.check_output(
                ["git", "-C", wma_root.as_posix(), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                **text_subprocess_kwargs(),
            ).strip()
            info["commit"] = commit
        except Exception:
            pass
    source_record = wma_root / "VENDORED_SOURCE.md"
    if source_record.is_file():
        for line in source_record.read_text(encoding="utf-8", errors="replace").splitlines():
            key, separator, value = line.removeprefix("- ").partition(":")
            if not separator or not value.strip():
                continue
            normalized_key = key.strip().lower().replace(" ", "_")
            if normalized_key in {"commit", "date_added"}:
                info[normalized_key] = value.strip()
        info["source_record"] = source_record.as_posix()
    receipt = wma_root / ".slicerdspace-source.json"
    if "commit" not in info and receipt.is_file():
        try:
            revision = json.loads(receipt.read_text(encoding="utf-8"))["revision"]
        except (OSError, ValueError, TypeError, KeyError):
            pass
        else:
            if isinstance(revision, str):
                info.update(commit=revision, source_record=receipt.as_posix())
    return info


def _command_path(command_name: str, wma_root: Path) -> Optional[Path]:
    local_candidates = [
        wma_root / "bin" / command_name,
        wma_root / "utilities" / command_name,
    ]
    if sys.platform == 'win32':
        local_candidates = [path for candidate in local_candidates
                            for path in (candidate.with_name(candidate.name + '.exe'), candidate)]
    local_candidates.extend(tool_dir / command_name for tool_dir in EXTRA_TOOL_DIRS)
    for candidate in local_candidates:
        if candidate.is_file():
            return candidate.resolve()
    found = shutil.which(command_name)
    return Path(found).resolve() if found else None


def _slicerdmri_module_paths(slicer_executable: Path) -> tuple[Path, ...]:
    roots = slicer_runtime_roots(slicer_executable.expanduser().resolve().parent)
    candidates = sorted({path for root in roots for pattern in (
        'slicer.org/Extensions-*/SlicerDMRI', 'Extensions-*/SlicerDMRI')
        for path in root.glob(pattern)})
    versions = {path.name for root in roots for path in (root / 'lib').glob('Slicer-*') if path.is_dir()}
    matches = [(candidate, [path for path in sorted((candidate / 'lib').glob('Slicer-*'))
                            if not versions or path.name in versions]) for candidate in candidates]
    matching = [paths for _, paths in matches if paths]
    # Do not mix cached extensions built against different Slicer versions.
    module_roots = matching[-1] if matching else []
    paths: list[Path] = []
    for module_root in module_roots:
        paths.extend(
            (
                module_root / "qt-loadable-modules",
                module_root / "qt-scripted-modules",
                module_root / "cli-modules",
            )
        )
    return tuple(path for path in paths if path.is_dir())


def _slicer_runtime_library_paths(slicer_executable: Path) -> tuple[Path, ...]:
    """Return library directories needed by Slicer/PythonSlicer shims."""

    roots = slicer_runtime_roots(slicer_executable.expanduser().resolve().parent)
    candidates = [path for root in roots for path in sorted((root / 'lib').glob('Slicer-*'))]
    candidates.extend(root / 'lib' for root in roots)
    # Slicer-core Model/Transform hardening needs no SlicerDMRI extension paths.
    candidates.extend(path for path in EXTRA_LIBRARY_DIRS)
    unique: list[Path] = []
    for path in candidates:
        if path.is_dir():
            resolved = path.resolve()
            if resolved not in unique:
                unique.append(resolved)
    return tuple(unique)


def _resolve_slicer_launcher(slicer_executable: Path) -> Path:
    slicer_executable = slicer_executable.expanduser().resolve()
    launcher = slicer_executable.parent.parent / "run-slicer.sh"
    return launcher if launcher.is_file() else slicer_executable


def _resolve_slicer_runtime(slicer_executable: Path) -> Path:
    """Resolve a wrapper's advertised runtime without mixing Slicer versions."""
    executable = slicer_executable.expanduser().resolve()
    if sys.platform == 'win32' or any(
        (root / relative).is_file()
        for root in slicer_runtime_roots(executable.parent)
        for relative in ('PythonSlicer', 'bin/PythonSlicer')
    ):
        return executable
    completed = subprocess.run(
        [str(executable), '--launcher-dump-environment'],
        env=external_python_environment(), timeout=30, capture_output=True,
        **text_subprocess_kwargs(),
    )
    homes = {line.partition('=')[2].strip() for line in (completed.stdout or '').splitlines()
             if line.startswith('SLICER_HOME=')}
    if not homes:
        # External CLI bridges need not advertise a native Slicer runtime.
        return executable
    if completed.returncode != 0 or len(homes) != 1:
        raise RuntimeError('Slicer wrapper did not identify one usable SLICER_HOME.')
    home = Path(homes.pop()).expanduser().resolve()
    launchers = (home / 'Slicer', home / 'MacOS' / 'Slicer')
    native = next((path for path in launchers if path.is_file()), None)
    if native is None or not (home / 'bin' / 'PythonSlicer').is_file():
        raise RuntimeError(f'Slicer wrapper advertised an incomplete runtime: {home}')
    return native.resolve()


def _resolve_runtime_display() -> Optional[str]:
    for env_name in ("SLICERDSPACE_DISPLAY", "DISPLAY"):
        value = os.environ.get(env_name)
        if value:
            return value
    return None


def _use_virtual_x(requested: bool = False) -> bool:
    """Linux's no-main-window Slicer still needs an X display."""
    return requested or (sys.platform.startswith('linux') and _resolve_runtime_display() is None)


def inspect_wma_backend(
    wma_root: Optional[Path] = None,
    *,
    require_virtual_x: bool = False,
    require_diffusion_measurements: bool = False,
    execution_backend: str = "legacy",
    platform: Optional[str] = None,
) -> dict[str, Any]:
    """Inspect the external WhiteMatterAnalysis backend without running WMA."""

    status: dict[str, Any] = {
        "available": False,
        "root": None,
        "script": None,
        "missing_required": [],
        "missing_optional": [],
        "commands": {},
        "source": None,
    }
    native = execution_backend == "native"
    if native:
        from shared.backend_sources import profile_path, verify_source

        root = (wma_root.expanduser().resolve() if wma_root is not None else
                resolve_configured_wma_root(execution_backend=execution_backend, platform=platform))
        receipt = root / ".slicerdspace-source.json"
        if root == profile_path("wma").resolve() or receipt.exists() or receipt.is_symlink():
            verify_source("wma", root)
    else:
        root = resolve_wma_root(wma_root)
    status["root"] = root.as_posix()
    status["execution_backend"] = execution_backend
    script = _wma_script_path(root)
    status["script"] = script.as_posix()
    status["source"] = wma_source_info(root)

    if native:
        for name in NATIVE_WMA_COMMANDS:
            source = root / "bin" / name
            status["commands"][name] = source.as_posix() if source.is_file() else None
            if not source.is_file():
                status["missing_required"].append(name)
        if not (root / "whitematteranalysis" / "__init__.py").is_file():
            status["missing_required"].append("whitematteranalysis package")
    required = [] if native else list(REQUIRED_WMA_COMMANDS)
    optional = list(OPTIONAL_WMA_COMMANDS)
    if require_diffusion_measurements and "wm_diffusion_measurements.py" not in required:
        required.append("wm_diffusion_measurements.py")
    runtime_platform = (platform or sys.platform).lower()
    if require_virtual_x or (runtime_platform.startswith("linux") and _resolve_runtime_display() is None):
        for command in ("xvfb-run", "Xvfb", "xauth"):
            if command not in required:
                required.append(command)
    for command_name in required:
        command_path = _command_path(command_name, root)
        status["commands"][command_name] = command_path.as_posix() if command_path else None
        if command_path is None:
            status["missing_required"].append(command_name)
    for command_name in optional:
        if command_name in required:
            continue
        command_path = _command_path(command_name, root)
        status["commands"][command_name] = command_path.as_posix() if command_path else None
        if command_path is None:
            status["missing_optional"].append(command_name)

    status["script_executable"] = os.access(script, os.X_OK)
    status["available"] = not status["missing_required"] and (native or bool(status["script_executable"]))
    return status


def inspect_wma_org_atlas(atlas_dir: Path, atlas_family: str = "auto") -> dict[str, Any]:
    """Inspect the selected atlas layout and its tract/hemisphere definitions."""

    root = atlas_dir.expanduser().resolve()
    family = detect_atlas_family(root, atlas_family)
    definition = atlas_definition(family)
    required_dirs = definition["directories"]
    status: dict[str, Any] = {
        "available": False,
        "root": root.as_posix(),
        "exists": root.is_dir(),
        "missing_required_dirs": [],
        "required_dirs": {},
        "atlas_family": family,
        "label": definition["label"],
        "source": definition["source"],
        "paper": definition["paper"],
    }
    if not root.is_dir():
        status["missing_required_dirs"] = list(required_dirs)
        return status

    for dirname in required_dirs:
        path = root / dirname
        file_count = sum(1 for item in path.rglob("*") if item.is_file()) if path.is_dir() else 0
        status["required_dirs"][dirname] = {
            "path": path.as_posix(),
            "exists": path.is_dir(),
            "file_count": file_count,
        }
        if not path.is_dir():
            status["missing_required_dirs"].append(dirname)
    status["available"] = not status["missing_required_dirs"]
    if family != "org" and status["available"]:
        try:
            status["output_contract"] = atlas_output_contract(family, root / required_dirs[1])
            for relative in (required_dirs[0] + "/registration_atlas.vtk",
                             required_dirs[1] + "/atlas.p", required_dirs[1] + "/atlas.vtp",
                             required_dirs[1] + "/cluster_hemisphere_location.txt"):
                if not (root / relative).is_file():
                    raise ValueError("Missing atlas resource: " + relative)
            if len(list((root / required_dirs[1]).glob("cluster_*.vtp"))) != 800:
                raise ValueError("The atlas requires all 800 reference clusters for WMA outlier removal")
            status["tract_hemisphere_check"] = validate_group_hemispheres(
                root / required_dirs[1], status["output_contract"]["groups"])
        except ValueError as exc:
            status.update(available=False, error=str(exc))
    return status


def summarize_wma_outputs(
    output_dir: Path, input_tractography: Path, *, registration_mode: str = "rig",
) -> dict[str, Any]:
    """Summarize expected WMA output products for manifest/QC."""

    case_id = input_tractography.stem
    case_dir = output_dir / case_id
    anatomical_dir = case_dir / "AnatomicalTracts"
    cluster_dir = case_dir / "FiberClustering"
    registered_case = f"{case_id}_reg_reg" if registration_mode == "nonrig" else f"{case_id}_reg"
    initial_dir = cluster_dir / "InitialClusters" / registered_case
    outlier_dir = cluster_dir / "OutlierRemovedClusters" / f"{registered_case}_outlier_removed"
    transformed_dir = cluster_dir / "TransformedClusters" / case_id
    separated_dir = cluster_dir / "SeparatedClusters"
    tracts = sorted(anatomical_dir.glob("*.vtp")) if anatomical_dir.is_dir() else []
    registration_dir = case_dir / "TractRegistration" / case_id
    progress_file = registration_dir / "progress.txt"
    iteration_dirs = sorted(path.name for path in registration_dir.glob("iteration_*") if path.is_dir())

    def _count_vtp(path: Path) -> int:
        return len(list(path.glob("*.vtp"))) if path.is_dir() else 0

    anatomical_line_counts = {tract.name: vtp_streamline_count(tract) for tract in tracts}
    readable_anatomical_line_counts = {
        name: count for name, count in anatomical_line_counts.items() if count is not None
    }

    progress_summary = _parse_wma_progress(progress_file)

    commissural_dir = separated_dir / "tracts_commissural"
    left_hemi_dir = separated_dir / "tracts_left_hemisphere"
    right_hemi_dir = separated_dir / "tracts_right_hemisphere"
    return {
        "case_id": case_id,
        "case_dir": case_dir.as_posix(),
        "case_dir_exists": case_dir.is_dir(),
        "anatomical_tracts_dir": anatomical_dir.as_posix(),
        "anatomical_tracts_dir_exists": anatomical_dir.is_dir(),
        "anatomical_tract_count": len(tracts),
        "anatomical_tract_files": [tract.as_posix() for tract in tracts],
        "anatomical_tract_readable_count": len(readable_anatomical_line_counts),
        "anatomical_tract_nonempty_count": sum(
            count > 0 for count in readable_anatomical_line_counts.values()
        ),
        "anatomical_streamline_count": sum(readable_anatomical_line_counts.values()),
        "anatomical_tract_streamline_counts": readable_anatomical_line_counts,
        "tract_sampling": tract_sampling_eligibility(anatomical_line_counts),
        "anatomical_empty_tracts": sorted(
            name for name, count in readable_anatomical_line_counts.items() if count == 0
        ),
        "fiber_clustering_dir": cluster_dir.as_posix(),
        "fiber_clustering_dir_exists": cluster_dir.is_dir(),
        "initial_clusters_dir": initial_dir.as_posix(),
        "initial_clusters_dir_exists": initial_dir.is_dir(),
        "initial_cluster_count": _count_vtp(initial_dir),
        "outlier_removed_dir": outlier_dir.as_posix(),
        "outlier_removed_dir_exists": outlier_dir.is_dir(),
        "outlier_removed_cluster_count": _count_vtp(outlier_dir),
        "transformed_clusters_dir": transformed_dir.as_posix(),
        "transformed_clusters_dir_exists": transformed_dir.is_dir(),
        "transformed_cluster_count": _count_vtp(transformed_dir),
        "separated_clusters_dir": separated_dir.as_posix(),
        "separated_clusters_dir_exists": separated_dir.is_dir(),
        "separated_commissural_dir": commissural_dir.as_posix(),
        "separated_commissural_count": _count_vtp(commissural_dir),
        "separated_left_hemisphere_dir": left_hemi_dir.as_posix(),
        "separated_left_hemisphere_count": _count_vtp(left_hemi_dir),
        "separated_right_hemisphere_dir": right_hemi_dir.as_posix(),
        "separated_right_hemisphere_count": _count_vtp(right_hemi_dir),
        "diffusion_measurements_csv": (anatomical_dir / "diffusion_measurements_anatomical_tracts.csv").as_posix(),
        "diffusion_measurements_csv_exists": (anatomical_dir / "diffusion_measurements_anatomical_tracts.csv").is_file(),
        "registration_dir": registration_dir.as_posix(),
        "registration_dir_exists": registration_dir.is_dir(),
        "registration_iteration_dir_count": len(iteration_dirs),
        "registration_iteration_dirs": iteration_dirs,
        "progress_file": progress_file.as_posix(),
        "progress_file_exists": progress_file.is_file(),
        "progress": progress_summary,
    }


def _parse_wma_progress(progress_file: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "exists": progress_file.is_file(),
        "last_nonempty_line": None,
        "completed_iterations": 0,
        "total_iterations": None,
        "latest_percent_finished": None,
        "latest_elapsed_minutes": None,
    }
    if not progress_file.is_file():
        return summary
    text = progress_file.read_text(encoding="utf-8", errors="replace")
    last_line = _last_nonempty_line(text)
    summary["last_nonempty_line"] = last_line
    matches = list(
        re.finditer(
            r"Done iteration\s+(\d+)\s*/\s*(\d+)\.\s*Percent finished approx:\s*([0-9.]+)\.\s*Time:.*Minutes Elapsed:\s*([0-9.]+)",
            text,
        )
    )
    if not matches:
        return summary
    latest = matches[-1]
    summary["completed_iterations"] = int(latest.group(1))
    summary["total_iterations"] = int(latest.group(2))
    summary["latest_percent_finished"] = float(latest.group(3))
    summary["latest_elapsed_minutes"] = float(latest.group(4))
    return summary


def _last_nonempty_line(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def _derive_wma_status(output_summary: dict[str, Any], execution_error: Optional[str], executed: bool) -> str:
    if not executed:
        return "planned"
    anatomical_tract_count = int(output_summary.get("anatomical_tract_count", 0) or 0)
    readable_tract_count = int(output_summary.get("anatomical_tract_readable_count", 0) or 0)
    nonempty_tract_count = int(output_summary.get("anatomical_tract_nonempty_count", 0) or 0)
    if execution_error and (
        anatomical_tract_count > 0
        or output_summary.get("transformed_cluster_count", 0) > 0
        or output_summary.get("outlier_removed_cluster_count", 0) > 0
        or output_summary.get("initial_cluster_count", 0) > 0
    ):
        return "partial_failed"
    if execution_error:
        return "failed"
    if _wma_output_contract_error(output_summary) is None:
        return "completed"
    if anatomical_tract_count > 0:
        return "partial_failed"
    if output_summary.get("transformed_cluster_count", 0) > 0:
        return "partial_failed"
    if output_summary.get("outlier_removed_cluster_count", 0) > 0:
        return "partial_failed"
    if output_summary.get("initial_cluster_count", 0) > 0:
        return "partial_failed"
    return "failed"


def _wma_output_contract_error(output_summary: dict[str, Any]) -> Optional[str]:
    contract = output_summary.get("atlas_contract", {})
    minimum = contract.get("minimum_anatomical_tract_count", MIN_ANATOMICAL_TRACT_COUNT)
    nonempty_minimum = contract.get("minimum_nonempty_anatomical_tract_count", MIN_NONEMPTY_ANATOMICAL_TRACT_COUNT)
    tract_count = int(output_summary.get("anatomical_tract_count", 0) or 0)
    readable_count = int(output_summary.get("anatomical_tract_readable_count", 0) or 0)
    nonempty_count = int(output_summary.get("anatomical_tract_nonempty_count", 0) or 0)
    if tract_count < minimum:
        return (
            "WMA command returned without a complete AnatomicalTracts output "
            f"(expected at least {minimum} files, found {tract_count})."
        )
    if readable_count != tract_count:
        return (
            "WMA AnatomicalTracts output contains unreadable VTP files "
            f"(expected all {tract_count} files to decode, found {readable_count})."
        )
    if nonempty_count < nonempty_minimum:
        return (
            "WMA AnatomicalTracts output is under-populated "
            f"(expected at least {nonempty_minimum} non-empty tracts, found {nonempty_count}). "
            "Review whole-brain tractography coverage, streamline lengths and atlas registration."
        )
    counts = output_summary.get('anatomical_tract_streamline_counts') or {}
    missing = sorted(set(contract.get("expected_tract_files", ())) - counts.keys())
    if missing:
        return "WMA atlas-specific tracts are missing: " + ", ".join(missing)
    sparse = [f'{name}={counts.get(name, "missing")}' for name in REQUIRED_MAJOR_TRACTS
              if int(counts.get(name, 0)) < MIN_MAJOR_TRACT_STREAMLINES]
    if sparse:
        return ('WMA major-tract sampling/identity check failed (local sampling minimum '
                f'{MIN_MAJOR_TRACT_STREAMLINES}, not anatomical accuracy): ' + ', '.join(sparse)
                + '. Review input coverage and atlas registration; do not use for bundle quantification.')
    return None


def build_wma_org_atlas_command(config: WMAOrgAtlasConfig) -> tuple[str, ...]:
    if config.num_threads is None:
        from dataclasses import replace
        config = replace(config, num_threads=1)
    wma_root = resolve_wma_root(config.wma_root)
    script = _wma_script_path(wma_root)
    if config.registration_mode not in {"rig", "nonrig"}:
        raise ValueError("registration_mode must be 'rig' or 'nonrig'.")
    if type(config.num_threads) is not int or config.num_threads < 1:
        raise ValueError("num_threads must be >= 1.")
    if config.clean_files not in {0, 1, 2}:
        raise ValueError("clean_files must be 0, 1, or 2.")

    cmd: list[str] = [
        script.as_posix(),
        "-i",
        config.input_tractography.expanduser().resolve().as_posix(),
        "-o",
        config.output_dir.expanduser().resolve().as_posix(),
        "-a",
        config.atlas_dir.expanduser().resolve().as_posix(),
        "-s",
        config.slicer_executable.expanduser().resolve().as_posix(),
        "-r",
        config.registration_mode,
        "-n",
        str(config.num_threads),
        "-x",
        "1" if _use_virtual_x(config.use_virtual_x) else "0",
        "-c",
        str(config.clean_files),
    ]
    if _wma_supports_gpu_flag(wma_root):
        cmd.extend(["-g", _normalize_wma_gpu_mode(config.device)])
    if config.transform_path is not None:
        cmd.extend(["-t", config.transform_path.expanduser().resolve().as_posix()])
    if config.export_diffusion_measures and config.measurement_backend == "fibertractmeasurements":
        if config.fiber_tract_measurements_cli is None:
            raise ValueError("fiber_tract_measurements_cli is required when export_diffusion_measures=True.")
        cmd.extend(["-d", "1", "-m", config.fiber_tract_measurements_cli.expanduser().resolve().as_posix()])
    # The separately selected native method runs after author clustering.
    family = detect_atlas_family(config.atlas_dir, config.atlas_family)
    if family != "org":
        runner = Path(__file__).resolve().parents[1] / "scripts/run_wma_atlas.py"
        author_shell = str(script)
        # Execution prepares the same PythonSlicer PATH shim used by the
        # author's env-python tools; python-real cannot run in a cleaned env.
        python = "python"
        if sys.platform == "win32":
            from shared.wsl_bridge import bridge_config
            bridge = bridge_config(script)
            python = str(wma_root / "bin/python.exe")
            python_bridge = bridge_config(python) if Path(python).is_file() else None
            if (not bridge or len(bridge["arguments"]) != 1 or not Path(python).is_file()
                    or not python_bridge or len(python_bridge["arguments"]) != 1
                    or python_bridge["distribution"] != bridge["distribution"]
                    or python_bridge["target"] != bridge["target"]
                    or Path(python_bridge["arguments"][0]).name != "python"):
                raise RuntimeError("Repair the Windows WMA runtime to enable its atlas Python bridge")
            author_shell = bridge["arguments"][0]
        cmd = [python, runner.as_posix(), "--family", family, "--author-shell", author_shell, "--", *cmd[1:]]
    return tuple(cmd)


def _resolve_tckconvert(config: WMAOrgAtlasConfig) -> Optional[Path]:
    configured_dirs: list[Path] = []
    if config.mrtrix_bin_dir is not None:
        configured_dirs.append(config.mrtrix_bin_dir.expanduser())
    for env_name in ("SLICERDSPACE_MRTRIX_BIN_DIR", "MRTRIX_BIN_DIR"):
        value = os.environ.get(env_name)
        if value:
            configured_dirs.append(Path(value).expanduser())
    executable_names = ("tckconvert.exe", "tckconvert") if os.name == "nt" else ("tckconvert",)
    for bin_dir in configured_dirs:
        for executable_name in executable_names:
            candidate = bin_dir / executable_name
            if candidate.is_file():
                return candidate.resolve()
    resolved = shutil.which("tckconvert")
    return Path(resolved).resolve() if resolved else None


def validate_wma_org_inputs(config: WMAOrgAtlasConfig) -> None:
    _inspect_wma_run(config)


def _inspect_wma_run(config: WMAOrgAtlasConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    if config.measurement_backend not in {"fibertractmeasurements", "native_vtk_point_scalar_summary"}:
        raise ValueError("Unknown WMA measurement backend: " + str(config.measurement_backend))
    if not config.subject_id.strip():
        raise ValueError("subject_id is required.")
    input_suffix = config.input_tractography.suffix.lower()
    if input_suffix not in {".vtk", ".vtp", ".tck"}:
        raise ValueError("WMA atlas input must be a whole-brain VTK/VTP/TCK tractography file.")
    if not config.input_tractography.expanduser().is_file():
        raise FileNotFoundError(f"Input tractography not found: {config.input_tractography}")
    if input_suffix == ".tck" and _resolve_tckconvert(config) is None:
        raise RuntimeError(
            "TCK input requires MRtrix3 tckconvert. Configure mrtrix_bin_dir, "
            "SLICERDSPACE_MRTRIX_BIN_DIR, MRTRIX_BIN_DIR, or PATH."
        )
    if not config.atlas_dir.expanduser().is_dir():
        raise FileNotFoundError(f"WMA atlas directory not found: {config.atlas_dir}")
    atlas_status = inspect_wma_org_atlas(config.atlas_dir, config.atlas_family)
    if not atlas_status["available"]:
        missing = ", ".join(atlas_status["missing_required_dirs"])
        label = "ORG" if atlas_status["atlas_family"] == "org" else atlas_status["atlas_family"]
        raise FileNotFoundError(f"WMA {label} atlas directory is incomplete. Missing: {missing}. " + atlas_status.get("error", ""))
    if not config.slicer_executable.expanduser().is_file():
        raise FileNotFoundError(f"Slicer executable not found: {config.slicer_executable}")
    if config.transform_path is not None and not config.transform_path.expanduser().is_file():
        raise FileNotFoundError(f"Transform file not found: {config.transform_path}")
    if (
        config.fiber_tract_measurements_cli is not None
        and not config.fiber_tract_measurements_cli.expanduser().is_file()
    ):
        raise FileNotFoundError(f"FiberTractMeasurements CLI not found: {config.fiber_tract_measurements_cli}")
    execution_backend = resolve_wma_execution_backend(
        config.wma_root, atlas_family=atlas_status["atlas_family"],
        export_diffusion_measures=config.export_diffusion_measures,
        measurement_backend=config.measurement_backend,
    )
    if execution_backend == "native":
        if config.registration_mode not in {"rig", "nonrig"}:
            raise ValueError("registration_mode must be 'rig' or 'nonrig'.")
        if config.clean_files not in {0, 1, 2}:
            raise ValueError("clean_files must be 0, 1, or 2.")
    backend_status = inspect_wma_backend(
        config.wma_root,
        execution_backend=execution_backend,
        require_virtual_x=config.use_virtual_x,
        require_diffusion_measurements=(config.export_diffusion_measures
                                        and config.measurement_backend == "fibertractmeasurements"),
    )
    if not backend_status["available"]:
        missing = ", ".join(backend_status["missing_required"])
        raise RuntimeError(f"WMA backend is incomplete. Missing required command(s): {missing}")
    if config.execute and sys.platform == 'win32' and execution_backend == 'legacy':
        from shared.wsl_bridge import bridge_config
        if bridge_config(Path(backend_status['script'])) is None:
            raise RuntimeError('Windows WMA requires the managed WSL command bridge; a raw Linux .sh checkout '
                               'cannot run natively. Run Advanced settings > Windows runtime setup.')
    return backend_status, atlas_status


def _run_native_org(config, staging, wma_root, slicer, python, env, manifest, stdout_log, stderr_log, log):
    from tractography.wma_native import run_org_atlas_native

    xvfb = _command_path("xvfb-run", wma_root) if config.use_virtual_x else None
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    env = dict(env, PYTHONUTF8="1")
    with tempfile.TemporaryDirectory(prefix="wma-worker-", dir=staging) as temp_dir, \
            stdout_log.open("w", encoding="utf-8") as stdout, \
            stderr_log.open("w", encoding="utf-8") as stderr:
        if sys.platform == "win32" and config.num_threads > 1:
            # Joblib's process tracker uses an ASCII wire format for temp paths.
            joblib_temp = env.get("JOBLIB_TEMP_FOLDER") or tempfile.gettempdir()
            if not joblib_temp.isascii():
                if not temp_dir.isascii():
                    raise RuntimeError(
                        "WMA with multiple workers needs a writable temporary path with ASCII characters. "
                        "Set JOBLIB_TEMP_FOLDER to that directory or run with one WMA worker."
                    )
                env["JOBLIB_TEMP_FOLDER"] = temp_dir
        worker = Path(temp_dir) / "harden.py"
        worker.write_text(
            "import sys, traceback, slicer\n"
            f"sys.path.insert(0, {str(scripts)!r})\n"
            "from run_wma_harden_transform import main\n"
            "try:\n"
            "    result = main(sys.argv[1:])\n"
            "except BaseException:\n"
            "    traceback.print_exc()\n"
            "    result = 1\n"
            "slicer.util.exit(result)\n",
            encoding="utf-8",
        )

        def run_stage(stage, argv):
            script, *args = argv
            if script == "run_wma_harden_transform.py":
                command = [str(slicer), "--no-main-window", "--python-script", str(worker), *args]
                if xvfb is not None:
                    command[:0] = [str(xvfb), "-a"]
            elif stage == "registration-nonrigid":
                command = [
                    str(python), "-u", str(scripts / "run_wma_registration.py"),
                    "--author-script", str(wma_root / "bin" / script),
                    "--native-nonrigid", "--", *args,
                ]
            elif stage in ("registration-rigid_affine_fast", "registration-affine") and config.registration_preprocess_cache:
                manifest["registration_preprocess_cache"]["installed"] = True
                command = [
                    str(python), "-u", str(scripts / "run_wma_registration.py"),
                    "--author-script", str(wma_root / "bin" / script),
                    "--report-dir", str(stdout_log.parent), "--", *args,
                ]
            else:
                command = [str(python), "-u", str(wma_root / "bin" / script), *args]
            record = {"stage": stage, "command": command}
            manifest["stage_commands"].append(record)
            if log:
                log(f"[WMA] {stage}")
            stdout.write(f"\n[WMA] {stage}\n")
            stdout.flush()
            stderr.write(f"\n[WMA] {stage}\n")
            stderr.flush()
            stage_output_start = stdout.tell()
            completed = subprocess.run(command, cwd=wma_root, env=env, stdout=stdout, stderr=stderr)
            record["return_code"] = completed.returncode
            completed.check_returncode()
            if stage == "registration-nonrigid":
                with stdout_log.open("rb") as stream:
                    stream.seek(stage_output_start)
                    if any(b"EXCEPTION WAS CAUGHT" in line for line in stream):
                        raise RuntimeError("WMA nonrigid registration reported an optimizer exception. See the WMA stdout log.")

        run_org_atlas_native(
            input_tractography=config.input_tractography.resolve(),
            output_dir=staging,
            atlas_dir=config.atlas_dir.expanduser().resolve(),
            registration_mode=config.registration_mode,
            num_threads=config.num_threads,
            clean_files=config.clean_files,
            transform_path=config.transform_path.expanduser().resolve() if config.transform_path else None,
            run_stage=run_stage,
            gpu_mode=_normalize_wma_gpu_mode(config.device),
            supports_gpu_cli=_wma_supports_gpu_flag(wma_root),
        )


@scheduled_stage('wma')
def run_wma_org_atlas(
    config: WMAOrgAtlasConfig,
    *,
    log: Optional[Callable[[str], None]] = None,
) -> WMAOrgAtlasResult:
    """Prepare or execute clustering with the selected WMA atlas."""

    from dataclasses import replace
    import os
    threads = config.num_threads or int(os.environ.get('SLICERDSPACE_STAGE_CPUS', '1'))
    if config.num_threads is not None and (type(config.num_threads) is not int or config.num_threads < 1):
        raise ValueError('WMA threads must be a positive integer')
    reservation = os.environ.get('SLICERDSPACE_STAGE_CPUS')
    if reservation and threads > int(reservation):
        raise ValueError('WMA worker count exceeds the admitted CPU reservation')
    virtual_x_requested = config.use_virtual_x
    config = replace(config, num_threads=threads, use_virtual_x=_use_virtual_x(virtual_x_requested))
    backend_status, atlas_status = _inspect_wma_run(config)
    wma_root = Path(backend_status["root"])
    execution_backend = backend_status["execution_backend"]
    native = execution_backend == "native"
    atlas_contract = atlas_status.get("output_contract", {})
    def summarize(directory, tractogram):
        summary = summarize_wma_outputs(directory, tractogram, registration_mode=config.registration_mode)
        if atlas_contract:
            summary["atlas_contract"] = atlas_contract
        return summary
    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_tractography = config.input_tractography.expanduser().resolve()
    prepared_tractography = source_tractography
    conversion_command: tuple[str, ...] = ()
    if source_tractography.suffix.lower() == ".tck":
        prepared_tractography = output_dir / "input" / f"{source_tractography.stem}.vtk"
        tckconvert = _resolve_tckconvert(config)
        if tckconvert is None:  # Defensive: validate_wma_org_inputs already checks this.
            raise RuntimeError("MRtrix3 tckconvert was not found.")
        conversion_command = (
            tckconvert.as_posix(),
            source_tractography.as_posix(),
            prepared_tractography.as_posix(),
            "-force",
        )
    runtime_config = replace(config, input_tractography=prepared_tractography)
    command = () if native else build_wma_org_atlas_command(runtime_config)
    manifest_path = output_dir / "wma_org_atlas_manifest.json"
    qc_path = output_dir / "wma_org_atlas_qc.json"
    previous_records = archive_run_records(output_dir, ('wma_org_atlas_manifest.json', 'wma_org_atlas_qc.json')) if config.execute else None
    log_dir = output_dir / "logs"
    stdout_log = log_dir / "wma_stdout.log"
    stderr_log = log_dir / "wma_stderr.log"
    conversion_stdout_log = log_dir / "tckconvert_stdout.log"
    conversion_stderr_log = log_dir / "tckconvert_stderr.log"
    previous_cache_reports = set(log_dir.glob("registration_cache_*.json"))
    output_summary = summarize(output_dir, prepared_tractography)

    manifest = {
        "kind": "wma_org_atlas",
        "version": 4,
        "name": "WMA " + atlas_status["atlas_family"] + " atlas clustering",
        "atlas_family": atlas_status["atlas_family"],
        "subject_id": config.subject_id,
        "input_tractography": source_tractography.as_posix(),
        "prepared_input_tractography": prepared_tractography.as_posix(),
        "input_conversion": {
            "required": bool(conversion_command),
            "command": list(conversion_command),
            "stdout_log": conversion_stdout_log.as_posix(),
            "stderr_log": conversion_stderr_log.as_posix(),
        },
        "atlas_dir": config.atlas_dir.expanduser().resolve().as_posix(),
        "output_dir": output_dir.as_posix(),
        "registration_mode": config.registration_mode,
        "device": config.device,
        "gpu_mode": _normalize_wma_gpu_mode(config.device),
        "registration_preprocess_cache": {"requested": config.registration_preprocess_cache,
                                           "installed": False, "reports": []},
        "num_threads": config.num_threads,
        "use_virtual_x": config.use_virtual_x,
        "virtual_x_requested": virtual_x_requested,
        "export_diffusion_measures": config.export_diffusion_measures,
        "requested_measurement_backend": (config.measurement_backend
                                          if config.export_diffusion_measures else None),
        "clean_files": config.clean_files,
        "executed": bool(config.execute),
        "status": "planned" if not config.execute else "running",
        "command": list(command),
        "execution_backend": execution_backend,
        "backend": backend_status,
        "previous_records": previous_records,
        "atlas": atlas_status,
        "dependencies": {
            "input_tractography": _file_info(source_tractography),
            "atlas_dir": _file_info(config.atlas_dir),
            "slicer_executable": _file_info(config.slicer_executable),
            "wma_root": _file_info(wma_root),
            "transform": _file_info(config.transform_path) if config.transform_path else None,
        },
        "outputs": {
            "qc": qc_path.as_posix(),
            "stdout_log": stdout_log.as_posix(),
            "stderr_log": stderr_log.as_posix(),
            "summary": output_summary,
        },
    }
    if native:
        manifest["stage_commands"] = []
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if log:
        mode = "Executing" if config.execute else "Prepared dry-run"
        route = "native Python stages" if native else " ".join(command)
        log(f"[WMA] {mode} {atlas_status['atlas_family']} atlas: {route}")
        log(f"[WMA] Manifest: {manifest_path}")

    start_time = time.monotonic()
    return_code: Optional[int] = None
    execution_error: Optional[str] = None
    staging = None
    published_archive = None
    if config.execute:
        log_dir.mkdir(parents=True, exist_ok=True)
        # PythonSlicer's launcher will restore its own runtime. Never pass a
        # caller's Python stdlib overrides to an env-python author script.
        env = external_python_environment()
        gpu_mode = _normalize_wma_gpu_mode(config.device)
        if _wma_supports_gpu_flag(wma_root):
            env["PYTHONNOUSERSITE"] = "1"
            env["WMA_GPU"] = gpu_mode
            env["WMA_CPU_FAST"] = "1"
            env["WMA_CPU_THREADS"] = str(config.num_threads)
            if gpu_mode == "off":
                env["CUDA_VISIBLE_DEVICES"] = "-1"
        python_path = wma_root.as_posix()
        env["PYTHONPATH"] = (
            python_path if not env.get("PYTHONPATH") else python_path + os.pathsep + env["PYTHONPATH"]
        )
        slicer_executable = config.slicer_executable.expanduser().resolve()
        native_posix = sys.platform != 'win32'
        slicer_python = None
        path_entries = [
            (wma_root / "bin").as_posix(),
            (wma_root / "utilities").as_posix(),
        ]
        path_entries.extend(tool_dir.as_posix() for tool_dir in EXTRA_TOOL_DIRS if tool_dir.is_dir())
        library_entries = []
        runtime_display = _resolve_runtime_display()
        temp_python_dir: Optional[tempfile.TemporaryDirectory[str]] = None
        try:
            runtime_executable = _resolve_slicer_runtime(slicer_executable) if native_posix else slicer_executable
            slicer_python_candidates = tuple(path for root in slicer_runtime_roots(runtime_executable.parent)
                                            for path in (root / 'PythonSlicer', root / 'bin' / 'PythonSlicer'))
            slicer_python = next((path for path in slicer_python_candidates if path.is_file()), None) if native_posix else None
            if native:
                slicer_python = find_pythonslicer(runtime_executable.parent)
                if slicer_python is None:
                    raise FileNotFoundError(f"Native WMA needs PythonSlicer from {runtime_executable}.")
            library_entries = [str(path) for path in _slicer_runtime_library_paths(runtime_executable)] if native_posix else []
            if native_posix and (
                (config.export_diffusion_measures and config.measurement_backend == "fibertractmeasurements")
                or _wma_supports_gpu_flag(wma_root)
            ):
                for extension in _slicerdmri_module_paths(runtime_executable):
                    library_entries.extend((str(extension), str(extension.parent)))
            manifest['runtime'] = {'configured_launcher': str(slicer_executable),
                                   'resolved_executable': str(runtime_executable),
                                   'python': str(slicer_python) if slicer_python else None}
            if conversion_command:
                prepared_tractography.parent.mkdir(parents=True, exist_ok=True)
                if log:
                    log("[WMA] Converting TCK input: " + " ".join(conversion_command))
                conversion_env = external_python_environment(
                    path_prepend=(config.mrtrix_bin_dir,) if config.mrtrix_bin_dir is not None else (),
                )
                if sys.platform == "win32" and not conversion_env.get("HOME"):
                    conversion_env["HOME"] = conversion_env.get("USERPROFILE") or str(Path.home())
                converted = subprocess.run(
                    conversion_command,
                    check=True,
                    env=conversion_env,
                    capture_output=True,
                    **text_subprocess_kwargs(),
                )
                conversion_stdout_log.write_text(converted.stdout or "", encoding="utf-8")
                conversion_stderr_log.write_text(converted.stderr or "", encoding="utf-8")
            # POSIX author scripts still use PythonSlicer/virtual-display shims;
            # no SlicerDMRI module injection is needed for ModelFile transforms.
            if not native and native_posix and (slicer_python is not None or config.use_virtual_x):
                temp_python_dir = tempfile.TemporaryDirectory(prefix="slicerdds_wma_python_")
                if slicer_python is not None:
                    for py_name in ("python", "python3"):
                        shim = Path(temp_python_dir.name) / py_name
                        if not shim.exists():
                            shim.symlink_to(slicer_python)
                    if config.registration_preprocess_cache:
                        runner = Path(__file__).resolve().parents[1] / "scripts" / "run_wma_registration.py"
                        author_registration = wma_root / "bin" / "wm_register_to_atlas_new.py"
                        if runner.is_file() and author_registration.is_file():
                            registration_shim = Path(temp_python_dir.name) / author_registration.name
                            registration_shim.write_text(
                                f"#!/bin/sh\nexec {shlex.quote(str(slicer_python))} {shlex.quote(str(runner))} "
                                f"--author-script {shlex.quote(str(author_registration))} "
                                f'--report-dir {shlex.quote(str(log_dir))} -- "$@"\n', encoding="utf-8")
                            registration_shim.chmod(0o755)
                            manifest["registration_preprocess_cache"]["installed"] = True
                author_harden = wma_root / "bin" / "harden_transform_with_slicer.py"
                harden_runner = Path(__file__).resolve().parents[1] / "scripts" / "run_wma_harden_transform.py"
                if not _wma_supports_gpu_flag(wma_root) and author_harden.is_file() and harden_runner.is_file():
                    # wm_harden_transform.py runs `Slicer --python-script $(which harden_transform_with_slicer.py)`;
                    # this shim comes first on PATH and saves fiber bundles without Slicer's default writer choice.
                    harden_shim = Path(temp_python_dir.name) / author_harden.name
                    harden_shim.write_text(
                        "import sys\n"
                        f"sys.path.insert(0, {str(harden_runner.parent)!r})\n"
                        "import run_wma_harden_transform\n"
                        f"run_wma_harden_transform.run({str(author_harden)!r})\n", encoding="utf-8")
                    harden_shim.chmod(0o755)
                    manifest["harden_transform_shim"] = harden_shim.name
                if config.use_virtual_x:
                    real_xvfb = _command_path("xvfb-run", wma_root)
                    if real_xvfb is not None:
                        xvfb_shim = Path(temp_python_dir.name) / "xvfb-run"
                        xvfb_shim.write_text(
                            f"#!/bin/sh\nexec {shlex.quote(real_xvfb.as_posix())} -a \"$@\"\n",
                            encoding="utf-8",
                        )
                        xvfb_shim.chmod(0o755)
                command_to_run = command
                path_entries.insert(0, temp_python_dir.name)
            else:
                command_to_run = command
            env["PATH"] = os.pathsep.join(path_entries + ([env["PATH"]] if env.get("PATH") else []))
            if library_entries:
                library_variable = 'DYLD_LIBRARY_PATH' if sys.platform == 'darwin' else 'LD_LIBRARY_PATH'
                env[library_variable] = os.pathsep.join(
                    library_entries + ([env[library_variable]] if env.get(library_variable) else [])
                )
            if runtime_display:
                env["DISPLAY"] = runtime_display
            staging = new_product_staging(output_dir, 'wma')
            if native:
                _run_native_org(
                    runtime_config, staging, wma_root, runtime_executable, slicer_python,
                    env, manifest, stdout_log, stderr_log, log,
                )
                return_code = 0
            else:
                runtime_command = list(command_to_run)
                runtime_command[runtime_command.index('-o') + 1] = str(staging)
                manifest['executed_command'] = runtime_command
                completed = subprocess.run(
                    runtime_command,
                    check=True,
                    cwd=wma_root.as_posix(),
                    env=env,
                    capture_output=True,
                    **text_subprocess_kwargs(),
                )
                stdout_log.write_text(completed.stdout or "", encoding="utf-8")
                stderr_log.write_text(completed.stderr or "", encoding="utf-8")
                return_code = completed.returncode
            if config.export_diffusion_measures and config.measurement_backend == "native_vtk_point_scalar_summary":
                # This is not an unqualified replacement of FiberTractMeasurements:
                # tensor-only inputs fail and the method is recorded explicitly.
                from tractography.wma_native_measures import export_wma_case_measurements
                manifest["measurement_provenance"] = export_wma_case_measurements(
                    staging, prepared_tractography.stem)
        except subprocess.CalledProcessError as exc:
            return_code = exc.returncode
            failed_stdout = conversion_stdout_log if conversion_command and not prepared_tractography.is_file() else stdout_log
            failed_stderr = conversion_stderr_log if conversion_command and not prepared_tractography.is_file() else stderr_log
            if not native or failed_stdout == conversion_stdout_log:
                failed_stdout.write_text(exc.stdout or "", encoding="utf-8")
                failed_stderr.write_text(exc.stderr or "", encoding="utf-8")
            execution_error = str(exc)
        except KeyboardInterrupt:
            return_code = 130
            execution_error = "Interrupted by user."
            with stderr_log.open("a" if native else "w", encoding="utf-8") as stream:
                stream.write(execution_error + "\n")
        except Exception as exc:
            return_code = None
            execution_error = f"{type(exc).__name__}: {exc}"
            with stderr_log.open("a" if native else "w", encoding="utf-8") as stream:
                stream.write(traceback.format_exc())
        finally:
            if temp_python_dir is not None:
                temp_python_dir.cleanup()
            output_summary = summarize(staging, prepared_tractography) if staging else {}
            contract_error = _wma_output_contract_error(output_summary) if config.execute else None
            if execution_error is None and contract_error is not None:
                log_hint = _last_nonempty_line(stderr_log.read_text(encoding="utf-8", errors="replace") if stderr_log.is_file() else "")
                if log_hint is None:
                    log_hint = _last_nonempty_line(stdout_log.read_text(encoding="utf-8", errors="replace") if stdout_log.is_file() else "")
                execution_error = contract_error
                if log_hint:
                    execution_error += f" Last log line: {log_hint}"
            if staging:
                try:
                    if execution_error is None:
                        published_archive = publish_products(staging, output_dir, (prepared_tractography.stem,))
                        output_summary = summarize(output_dir, prepared_tractography)
                    else:
                        staging = retain_failed_products(staging, output_dir)
                        output_summary = summarize(staging, prepared_tractography)
                except OSError as exc:
                    execution_error = f'Failed to publish WMA products: {exc}'

    elapsed_seconds = time.monotonic() - start_time
    if manifest["registration_preprocess_cache"]["installed"]:
        manifest["registration_preprocess_cache"]["reports"] = [
            str(path) for path in sorted(log_dir.glob("registration_cache_*.json"))
            if path not in previous_cache_reports
        ]
    status = _derive_wma_status(output_summary, execution_error, bool(config.execute))
    qc = {
        "kind": "wma_org_atlas_qc",
        "atlas_family": atlas_status["atlas_family"],
        "version": WMA_QC_VERSION,
        "subject_id": config.subject_id,
        "status": status,
        "executed": bool(config.execute),
        "return_code": return_code,
        "elapsed_seconds": elapsed_seconds,
        "human_review": "pending",
        "execution_completed": bool(config.execute and status == 'completed'),
        "automatic_qc_passed": bool(config.execute and status == 'completed'
                                    and output_summary.get('tract_sampling', {}).get('all_tract_sampling_passed')),
        "major_tract_contract_passed": bool(config.execute and status == 'completed'),
        "quantification_eligible": False,
        "tract_sampling": output_summary.get("tract_sampling", {}),
        "warnings": (["Some atlas tracts have insufficient sampling and must not be used for quantification: "
                      + ", ".join(output_summary["tract_sampling"]["excluded_from_quantification"])]
                     if output_summary.get("tract_sampling", {}).get("excluded_from_quantification") else []),
        "scientific_scope": "Fresh decoded atlas products and atlas-specific coverage and sampling checks.",
        "previous_products": published_archive,
        "backend_available": bool(backend_status["available"]),
        "atlas_available": bool(atlas_status["available"]),
        "output_summary": output_summary,
        "acceptance_criteria": {
            "minimum_anatomical_tract_count": atlas_contract.get("minimum_anatomical_tract_count", MIN_ANATOMICAL_TRACT_COUNT),
            "minimum_readable_anatomical_tract_count": atlas_contract.get("minimum_anatomical_tract_count", MIN_ANATOMICAL_TRACT_COUNT),
            "minimum_nonempty_anatomical_tract_count": atlas_contract.get("minimum_nonempty_anatomical_tract_count", MIN_NONEMPTY_ANATOMICAL_TRACT_COUNT),
            "expected_tract_files": atlas_contract.get("expected_tract_files", []),
            "required_major_tracts": list(REQUIRED_MAJOR_TRACTS),
            "minimum_streamlines_per_major_tract": MIN_MAJOR_TRACT_STREAMLINES,
            "scope": "Atlas-specific whole-brain sampling/identity checks; anatomical review is recorded separately.",
        },
        "logs": {
            "stdout": stdout_log.as_posix(),
            "stderr": stderr_log.as_posix(),
            "stdout_exists": stdout_log.is_file(),
            "stderr_exists": stderr_log.is_file(),
            "tckconvert_stdout": conversion_stdout_log.as_posix(),
            "tckconvert_stderr": conversion_stderr_log.as_posix(),
        },
        "error": execution_error,
    }
    if config.export_diffusion_measures and config.measurement_backend == "native_vtk_point_scalar_summary":
        qc["warnings"].append(
            "Native VTK point-scalar summary was used. FiberTractMeasurements tensor/statistical "
            "parity is not yet validated; review numerical methods before quantification.")
    qc_path.write_text(json.dumps(qc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest["status"] = status
    manifest["executed"] = bool(config.execute)
    manifest["return_code"] = return_code
    manifest["runtime_seconds"] = elapsed_seconds
    manifest["previous_products"] = published_archive
    manifest["error"] = execution_error
    manifest["environment"] = {
        "use_virtual_x": config.use_virtual_x,
        "runtime_display": _resolve_runtime_display(),
        "resolved_wma_root": wma_root.as_posix(),
        "resolved_tckconvert": _resolve_tckconvert(config).as_posix() if conversion_command else None,
        "resolved_slicer_executable": config.slicer_executable.expanduser().resolve().as_posix(),
        "resolved_slicer_launcher": _resolve_slicer_launcher(config.slicer_executable).as_posix(),
    }
    manifest["outputs"]["summary"] = output_summary
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return WMAOrgAtlasResult(
        subject_id=config.subject_id,
        status=status,
        output_dir=output_dir,
        manifest_path=manifest_path,
        qc_path=qc_path,
        command=command,
        executed=bool(config.execute),
        backend_status=backend_status,
        atlas_status=atlas_status,
        return_code=return_code,
    )
