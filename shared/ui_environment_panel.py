"""Pure helpers for the Slicer environment/dependency panel."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from shared.dependency_config import DEPENDENCY_PROFILE_KIND
from shared.docs import doc_path
from shared.model_assets import MODEL_ASSETS_ENV, configured_model_assets_root


DEPENDENCY_PROFILE_SCHEMA = "slicerddsurfer.dependency_profile.v2"
DEPENDENCY_PROFILE_VERSION = 2

ENVIRONMENT_SECTION_METADATA = {
    "model-assets": {
        "title": "Model Asset Paths",
        "description": (
            "Configure where DDSurfer, DDSeg and DDParcel weights/templates are stored. "
            "Use a user data directory or shared lab cache rather than the Slicer extension code directory."
        ),
    },
    "core-runtime": {
        "title": "Core Python & Runtime Paths",
        "description": (
            "Configure the shared Python runtime and external toolchain used by preprocessing, post-process, and WMQL. "
            "Leave empty to use saved settings, environment variables, PATH, and common install paths."
        ),
    },
    "bundle-backends": {
        "title": "Optional Backend Paths",
        "description": (
            "WMA and TractCloud are optional bundle-extraction backends. Keep them grouped here instead of mixing "
            "them with the core runtime toolchain."
        ),
    },
}

ENVIRONMENT_SUBSECTION_METADATA = {
    "wma": {
        "title": "WMA Paths",
        "description": "WhiteMatterAnalysis, the selected ORG / East-West / NABA atlas, and Slicer/CLI paths.",
    },
    "tractcloud": {
        "title": "TractCloud Paths",
        "description": "TractCloud checkout, Python runtime, and model/data directory.",
    },
}

DEPENDENCY_BACKEND_GROUPS = {
    "wma_root": "wma",
    "wma_atlas_dir": "wma",
    "slicer_executable": "wma",
    "fiber_tract_measurements_cli": "wma",
    "tractcloud_root": "tractcloud",
    "tractcloud_python_executable": "tractcloud",
    "tractcloud_data_dir": "tractcloud",
}

DEPENDENCY_SUBSECTION_STARTS = {
    "wma_root": "wma",
    "tractcloud_root": "tractcloud",
}


@dataclass(frozen=True)
class EnvironmentPathEntry:
    """One schema-driven environment-path UI entry."""

    kind: str
    title: str = ""
    description: str = ""
    item: dict[str, str] | None = None
    subsection: str = ""


@dataclass(frozen=True)
class EnvironmentDialogState:
    """Pure state used to render the environment-results dialog."""

    summary_text: str
    detail_text: str
    missing_items: tuple[Any, ...]
    can_install_python: bool
    has_manual_setup_notes: bool
    python_install_tooltip: str
    ready_text: str = "All selected targets are runnable with the currently detected environment."


PATH_STATUS_DEPENDENCY_KEYS = {
    "postprocess_runtime": {"surface_postprocess_runtime"},
    "postprocess_license_file": {"surface_postprocess_license"},
    "mrtrix_bin_dir": {
        "command:mrconvert",
        "command:dwi2response",
        "command:dwi2fod",
        "command:mtnormalise",
        "command:tckgen",
        "command:tcksift2",
        "command:tckstats",
        "command:tckconvert",
        "command:tck2connectome",
        "command:dwifslpreproc",
        "command:dwi2mask",
    },
    "preprocessing_fsl_bin_dir": {"command:eddy", "command:topup"},
    "postprocess_freesurfer_home": {
        "env:freesurfer_home",
        "command:recon-all",
        "command:mris_sphere",
        "command:mris_curvature",
        "command:mris_inflate",
        "command:mris_register",
        "command:mri_label2label",
        "command:mri_surf2surf",
        "command:mri_annotation2label",
        "command:mri_segstats",
    },
    "tract_querier_root": {"path:tract_querier_root", "path:tract_querier_root_script", "python:tract_querier"},
    "wmql_python_executable": {"path:wmql_python"},
    "wma_root": {"env:wma_root", "path:wma_root_path", "command:wm_apply_ORG_atlas_to_subject.sh",
                 "path:wma_atlas_python_bridge"},
    "wma_atlas_dir": {
        "setting:wma_org_atlas",
        "path:wma_org_atlas",
        "path:wma_org_atlas_ORG-RegAtlas-100HCP",
        "path:wma_org_atlas_ORG-800FC-100HCP",
        "path:wma_org_atlas_East-West_WM_Atlas_Reg",
        "path:wma_org_atlas_East-West_WM_Atlas_800FC",
        "path:wma_org_atlas_NABA-RegAtlas",
        "path:wma_org_atlas_NABA-800FC",
    },
    "slicer_executable": {"env:slicer_executable", "path:slicer_executable_path"},
    "fiber_tract_measurements_cli": {"path:fiber_tract_measurements_cli"},
    "tractcloud_root": {"env:tractcloud_root", "path:tractcloud_root_path", "python:tractcloud"},
    "tractcloud_python_executable": {"path:tractcloud_python"},
    "tractcloud_data_dir": {"path:tractcloud_data_dir"},
    "python_executable": {"python:nibabel", "python:SimpleITK", "python:torch", "device:auto", "device:cpu"},
    "python_environment": {"python:nibabel", "python:SimpleITK", "python:torch", "device:auto", "device:cpu"},
    "conda_executable": set(),
}

PATH_STATUS_COLORS = {
    "available": "#2E7D32",
    "missing": "#C62828",
    "pending": "#EF6C00",
    "unchecked": "#78909C",
    "inactive": "#90A4AE",
}

DARK_PATH_STATUS_COLORS = {
    "available": "#66BB6A",
    "missing": "#FF7B72",
    "pending": "#FFB74D",
    "unchecked": "#9BA7B4",
    "inactive": "#8B949E",
}

OPTIONAL_BACKEND_TITLE_STYLES = {
    True: (
        "QLabel { background: #E4F4EA; border: 1px solid #AFCDBA; "
        "border-radius: 8px; color: #173D2A; font-weight: 800; padding: 7px 10px; }"
    ),
    False: (
        "QLabel { background: #F4F8F5; border: 1px solid #DDEAE2; border-radius: 7px; "
        "color: #315843; font-weight: 700; padding: 7px 9px; }"
    ),
}

DARK_OPTIONAL_BACKEND_TITLE_STYLES = {
    True: (
        "QLabel { background: #1F3528; border: 1px solid #3B6E52; "
        "border-radius: 8px; color: #A8E6CF; font-weight: 800; padding: 7px 10px; }"
    ),
    False: (
        "QLabel { background: #252B32; border: 1px solid #3D4750; border-radius: 7px; "
        "color: #D2DEE8; font-weight: 700; padding: 7px 9px; }"
    ),
}

OPTIONAL_BACKEND_DESCRIPTION_STYLES = {
    True: "QLabel { background: #F7FCF8; border-radius: 7px; color: #3F6754; padding: 4px 9px 7px 10px; }",
    False: "QLabel { color: #6A7D74; padding: 1px 8px 5px 9px; }",
}

DARK_OPTIONAL_BACKEND_DESCRIPTION_STYLES = {
    True: "QLabel { background: #1C2D24; border-radius: 7px; color: #9BD1B8; padding: 4px 9px 7px 10px; }",
    False: "QLabel { color: #9BA7B4; padding: 1px 8px 5px 9px; }",
}

OPTIONAL_BACKEND_ROW_STYLES = {
    ("QLineEdit", True): (
        "QLineEdit { background: #FCFEFC; color: #1E293B; border: 1px solid #7EB892; "
        "border-radius: 8px; padding: 7px 10px; }"
    ),
    ("ctkPathLineEdit", True): (
        "QLineEdit { background: #FCFEFC; color: #1E293B; border: 1px solid #7EB892; "
        "border-radius: 8px; padding: 7px 10px; }"
    ),
    ("QPushButton", True): (
        "QPushButton { background: #F7FCF8; border: 1px solid #A9CBB4; "
        "color: #1E5A3F; border-radius: 8px; padding: 6px 12px; }"
    ),
    ("QLineEdit", False): "",
    ("ctkPathLineEdit", False): "",
    ("QPushButton", False): "",
}

DARK_OPTIONAL_BACKEND_ROW_STYLES = {
    ("QLineEdit", True): (
        "QLineEdit { background: #1E2923; color: #F0F6FC; border: 1px solid #4A8B65; "
        "border-radius: 8px; padding: 7px 10px; }"
    ),
    ("ctkPathLineEdit", True): (
        "QLineEdit { background: #1E2923; color: #F0F6FC; border: 1px solid #4A8B65; "
        "border-radius: 8px; padding: 7px 10px; }"
    ),
    ("QPushButton", True): (
        "QPushButton { background: #23362B; border: 1px solid #4A8B65; "
        "color: #A8E6CF; border-radius: 8px; padding: 6px 12px; }"
    ),
    ("QLineEdit", False): "",
    ("ctkPathLineEdit", False): "",
    ("QPushButton", False): "",
}


def dependency_items_for_section(items: tuple[dict[str, str], ...], section: str) -> list[dict[str, str]]:
    """Return dependency schema rows for one UI section in declaration order."""

    return [item for item in items if item.get("section") == section]


def environment_path_entries(items: tuple[dict[str, str], ...]) -> list[EnvironmentPathEntry]:
    """Return section, subsection and row entries for the environment-path panel."""

    entries: list[EnvironmentPathEntry] = []
    for section in ("model-assets", "core-runtime", "bundle-backends"):
        section_meta = dependency_section_metadata(section)
        entries.append(
            EnvironmentPathEntry(
                kind="section",
                title=section_meta["title"],
                description=section_meta["description"],
            )
        )
        for item in dependency_items_for_section(items, section):
            subsection = dependency_subsection_key(item["id"])
            if subsection:
                subsection_meta = dependency_subsection_metadata(subsection)
                entries.append(
                    EnvironmentPathEntry(
                        kind="subsection",
                        title=subsection_meta["title"],
                        description=subsection_meta["description"],
                        subsection=subsection,
                    )
                )
            entries.append(EnvironmentPathEntry(kind="row", item=item))
        if section == "model-assets":
            entries.append(EnvironmentPathEntry(kind="model-assets"))
    return entries


def dependency_backend_group(dependency_id: str) -> str | None:
    """Return optional backend highlight group for a dependency row."""

    return DEPENDENCY_BACKEND_GROUPS.get(dependency_id)


def dependency_subsection_key(dependency_id: str) -> str | None:
    """Return subsection key that should be inserted before this dependency row."""

    return DEPENDENCY_SUBSECTION_STARTS.get(dependency_id)


def dependency_section_metadata(section: str) -> dict[str, str]:
    """Return title/description metadata for a dependency UI section."""

    return ENVIRONMENT_SECTION_METADATA[section]


def dependency_subsection_metadata(subsection: str) -> dict[str, str]:
    """Return title/description metadata for an optional backend subsection."""

    return ENVIRONMENT_SUBSECTION_METADATA[subsection]


def environment_path_status_style(state: str, *, dark: bool | None = None) -> str:
    """Return the stylesheet for one environment path status label."""

    from shared.ui_styles import is_dark_mode

    use_dark = is_dark_mode() if dark is None else bool(dark)
    palette = DARK_PATH_STATUS_COLORS if use_dark else PATH_STATUS_COLORS
    color = palette.get(state, palette["unchecked"])
    light_color = PATH_STATUS_COLORS.get(state, PATH_STATUS_COLORS["unchecked"])
    comment = f" /* {light_color} */" if use_dark and color != light_color else ""
    return f"QLabel {{ color: {color}; font-weight: 600;{comment} }}"


def path_status_dependencies(result: Any, dependency_id: str) -> list[Any]:
    """Return dependency statuses that explain one environment path row."""

    keys = PATH_STATUS_DEPENDENCY_KEYS.get(dependency_id, set())
    return [item for item in list(getattr(result, "statuses", ()) or ()) if item.key in keys]


def format_detected_source(source: str) -> str:
    """Return compact user-facing text for a detected dependency source."""

    if not source or source in {"not found", "not configured"}:
        return "not found"
    if source == "user setting":
        return "using saved path"
    if source.startswith("env:"):
        return "detected from " + source
    if source == "PATH":
        return "detected from PATH"
    if source == "common path":
        return "detected from common path"
    return source


def environment_path_status_from_dependencies(dependencies: list[Any]) -> tuple[str, str]:
    """Return status-label text and state from dependency scan results."""

    if not dependencies:
        return "not used by current selection", "inactive"
    available = [dependency for dependency in dependencies if dependency.status == "available"]
    if available:
        source = available[0].details.get("detected_source", "")
        return format_detected_source(source), "available"
    return "not found", "missing"


def optional_backend_label_styles(active: bool, *, dark: bool | None = None) -> tuple[str, str]:
    """Return title and description styles for an optional backend subsection."""

    from shared.ui_styles import is_dark_mode

    use_dark = is_dark_mode() if dark is None else bool(dark)
    titles = DARK_OPTIONAL_BACKEND_TITLE_STYLES if use_dark else OPTIONAL_BACKEND_TITLE_STYLES
    descriptions = DARK_OPTIONAL_BACKEND_DESCRIPTION_STYLES if use_dark else OPTIONAL_BACKEND_DESCRIPTION_STYLES
    return titles[bool(active)], descriptions[bool(active)]


def optional_backend_widget_style(class_name: str, active: bool, *, dark: bool | None = None) -> str | None:
    """Return style override for a widget in an optional backend row."""

    from shared.ui_styles import is_dark_mode

    use_dark = is_dark_mode() if dark is None else bool(dark)
    styles = DARK_OPTIONAL_BACKEND_ROW_STYLES if use_dark else OPTIONAL_BACKEND_ROW_STYLES
    return styles.get((class_name, bool(active)))


def model_assets_summary_text(repo_root: Path) -> str:
    """Return the model-assets summary shown in the environment panel."""

    manifest = repo_root / "models" / "assets_manifest.json"
    default_root = configured_model_assets_root()
    command = f"PythonSlicer scripts/download_model_assets.py --profile core --asset-root {default_root.as_posix()}"
    if manifest.is_file():
        status = _model_asset_profiles_status_text(repo_root)
        status_text = f" Current status: {status}." if status else " Click Check status to verify core/all locally."
        return (
            "Unified source: models/assets_manifest.json. "
            f"Default root: {default_root}. Override with the Model assets root field or ${MODEL_ASSETS_ENV}. "
            f"Refresh or verify with `{command}`."
            f"{status_text}"
        )
    return f"Unified source expected at {manifest.as_posix()}. Refresh or verify with `{command}`."


def _model_asset_profiles_status_text(repo_root: Path) -> str:
    """Return compact core/all profile verification text, or empty on failure."""

    try:
        module_name = "slicerdsurfer_download_model_assets_summary"
        module = sys.modules.get(module_name)
        if module is None:
            script_path = repo_root / "scripts" / "download_model_assets.py"
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            if spec is None or spec.loader is None:
                return ""
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        manifest = module.load_manifest(repo_root / "models" / "assets_manifest.json")
        parts = []
        for profile in ("core", "all"):
            plan = module.resolve_asset_plan(manifest, profile)
            verification = module.verify_asset_plan(repo_root, plan, check_hashes=False)
            checked = int(verification.get("checked_count") or 0)
            if verification.get("ok"):
                parts.append(f"{profile}=present ({checked})")
            else:
                missing = len(verification.get("missing") or [])
                pointers = len(verification.get("lfs_pointers") or [])
                parts.append(f"{profile}=needs download ({checked} checked, {missing} missing, {pointers} pointer)")
        return "; ".join(parts)
    except Exception:
        return ""


def model_assets_provenance_text(repo_root: Path) -> str:
    """Return compact model-asset provenance and release-boundary text."""

    manifest = (repo_root / "models" / "assets_manifest.json").as_posix()
    terms = doc_path(repo_root, "model_asset_terms").as_posix()
    lock_command = (
        "PythonSlicer scripts/download_model_assets.py --profile core --verify-only "
        "--asset-root <asset-root> --write-lockfile outputs/00_run_metadata/model_assets_core.lock.json"
    )
    return "\n".join(
        [
            "Model asset provenance",
            f"Manifest: {manifest}",
            f"Terms decision record: {terms}",
            "Default profile: core",
            f"Default user asset root: {configured_model_assets_root()}",
            f"Environment override: {MODEL_ASSETS_ENV}",
            "Verify command: PythonSlicer scripts/download_model_assets.py --profile core --asset-root <asset-root> --verify-only",
            f"Lockfile command: {lock_command}",
            "Release boundary: local verification does not grant public redistribution or DOI-archive rights.",
        ]
    )


def is_other_slicer_installation_path(raw_value: str, current_slicer_home: str | Path | None) -> bool:
    """Return True when ``raw_value`` points inside a different 3D Slicer installation than ``current_slicer_home``."""

    text = str(raw_value or "").strip()
    home_text = str(current_slicer_home or "").strip()
    if not text or not home_text:
        return False
    try:
        current_home = Path(home_text).expanduser().resolve()
        candidate = Path(text).expanduser().resolve()
    except OSError:
        return False
    if candidate == current_home or current_home in candidate.parents:
        return False
    for root in (candidate, *candidate.parents):
        name_lower = root.name.lower()
        has_slicer_marker = (
            name_lower.startswith("slicer ")
            or name_lower.startswith("slicer-")
            or name_lower == "slicer.app"
            or (root / "bin" / "PythonSlicer.exe").is_file()
            or (root / "bin" / "PythonSlicer").is_file()
        )
        if has_slicer_marker and root != current_home:
            return True
    return False


def python_install_target_text(python_executable: str) -> str:
    """Return the user-facing target text for Python package installs."""

    executable = python_executable or "unknown"
    return (
        "Python install target: current Slicer Python environment "
        f"({executable}). External tools are never installed silently."
    )


def dependency_profile_payload(
    external_dependencies: Mapping[str, str],
    *,
    python_executable: str,
    platform_text: str,
    created_at: str | None = None,
) -> dict[str, object]:
    """Build a UI/CLI-compatible dependency profile payload."""

    return {
        "kind": DEPENDENCY_PROFILE_KIND,
        "version": DEPENDENCY_PROFILE_VERSION,
        "schema": DEPENDENCY_PROFILE_SCHEMA,
        "created_at": created_at or datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "platform": platform_text,
        "python_install_target": python_executable or "",
        "revalidate_after_import": True,
        "external_dependencies": {
            key: value
            for key, value in sorted(external_dependencies.items())
            if str(value or "").strip()
        },
    }


def external_dependencies_from_profile(payload: Mapping[str, object]) -> dict[str, str]:
    """Extract the external dependency mapping from a profile-like payload."""

    data = payload.get("external_dependencies", payload)
    if not isinstance(data, Mapping):
        raise ValueError("Dependency config must contain an external_dependencies object.")
    return {str(key): str(value).strip() for key, value in data.items() if str(value or "").strip()}


def environment_missing_items(result: Any) -> list[Any]:
    """Return missing dependency statuses sorted for display."""

    missing = [item for item in result.statuses if item.status == "missing"]
    return sorted(missing, key=lambda item: (not item.required, item.kind, item.label.lower()))


def environment_summary_text(result: Any) -> str:
    """Build the compact dependency-check summary shown in the UI."""

    missing_items = environment_missing_items(result)
    optional_count = sum(1 for item in missing_items if not item.required)
    if not missing_items:
        return "Environment ready. All selected targets are runnable with the detected setup."
    status_text = (
        f"{len(result.runnable_targets)} runnable, "
        f"{len(result.blocked_targets)} need setup, "
        f"{result.blocker_count} required missing"
    )
    if optional_count:
        status_text += f", {optional_count} optional missing"
    if result.missing_python_packages:
        status_text += f". Python packages available for auto-install: {len(result.missing_python_packages)}"
    preview = ", ".join(item.label for item in missing_items[:3])
    if len(missing_items) > 3:
        preview += f", +{len(missing_items) - 3} more"
    return f"{status_text}.\nNeeds attention: {preview}"


def environment_detail_lines(result: Any) -> list[str]:
    """Build detail lines for the expandable environment summary."""

    missing_items = environment_missing_items(result)
    optional_count = sum(1 for item in missing_items if not item.required)
    lines = [
        f"Checked targets: {', '.join(result.targets) if result.targets else 'none'}",
        (
            f"Runnable targets: {len(result.runnable_targets)}; "
            f"blocked targets: {len(result.blocked_targets)}; "
            f"missing required items: {result.blocker_count}; "
            f"missing optional items: {optional_count}"
        ),
    ]
    if result.blocked_targets:
        lines.append("Blocked targets: " + ", ".join(result.blocked_targets))
    for backend, blockers in getattr(result, 'tracking_backend_blockers', {}).items():
        lines.append(f"Tractography {backend}: " + (blockers.get('tractography') or "dependencies available"))
    if result.missing_python_packages:
        lines.append("Auto-installable Python packages: " + ", ".join(result.missing_python_packages))
    if result.missing_external_groups:
        lines.append("Manual setup groups: " + ", ".join(result.missing_external_groups))
    if not missing_items:
        lines.append("No missing dependencies detected for the selected targets.")
    else:
        preview = ", ".join(item.label for item in missing_items[:6])
        if len(missing_items) > 6:
            preview += f", ... ({len(missing_items)} total)"
        lines.append("Missing items: " + preview)
    return lines


def environment_item_text(item: Any) -> str:
    """Build the multi-line text for one missing dependency item."""

    required = "required" if item.required else "optional"
    targets = ", ".join(item.targets) if item.targets else "none"
    lines = [
        f"{item.label} ({required})",
        f"Key: {item.key}",
        f"Targets: {targets}",
    ]
    details = item.details or {}
    if details.get("detected_source"):
        lines.append(f"Detected source: {details.get('detected_source')}")
    if item.kind == "python":
        if details.get("package"):
            lines.append(f"Package: {details.get('package')}")
        else:
            lines.append(f"Runtime module: {details.get('import') or item.label}")
        lines.append(f"Python: {details.get('python_executable') or 'current Slicer Python'}")
        install_requirement = details.get("install_requirement")
        if install_requirement and install_requirement != details.get("package"):
            lines.append(f"Install source: {install_requirement}")
    elif item.kind == "command":
        lines.append(f"Command: {details.get('command') or item.label}")
        if details.get("bin_dir"):
            lines.append(f"Configured bin dir: {details.get('bin_dir')}")
    elif item.kind == "path":
        lines.append(f"Path: {details.get('path') or ''}")
        lines.append(f"Expected type: {details.get('path_kind') or 'file'}")
    elif item.kind == "env":
        env_names = ", ".join((details.get("env") or {}).keys())
        lines.append(f"Environment variable(s): {env_names}")
    elif item.kind == "device":
        lines.append(f"Requested device: {details.get('requested_device') or ''}")
    elif item.kind == "setting":
        lines.append(details.get("notes") or "Provide the required setting before running.")
    if item.install_hint:
        lines.append(f"Suggested fix: {item.install_hint}")
    checked_locations = details.get("checked_locations") or []
    if checked_locations:
        preview = ", ".join(checked_locations[:3])
        if len(checked_locations) > 3:
            preview += f", ... ({len(checked_locations)} checked)"
        lines.append(f"Checked locations: {preview}")
    return "\n".join(line for line in lines if line)


def environment_item_button_text(item: Any) -> str:
    """Return the action button text for a missing dependency row."""

    if item.kind == "python":
        return "Install" if (item.details or {}).get("package") else "Setup"
    if getattr(item, "install_group", "") == "model-resources":
        return "Download / choose path"
    if getattr(item, "install_group", "") in {"mrtrix", "fsl"}:
        from shared.managed_runtimes import supported_platform
        if supported_platform():
            return "Install and configure"
    if item.kind in {"command", "path", "env"}:
        return "Choose path / setup"
    return "Setup"


def environment_item_button_tooltip(item: Any) -> str:
    """Return the action button tooltip for a missing dependency row."""

    if item.kind == "python":
        if (item.details or {}).get("package"):
            return item.install_hint or "Install this package into the selected Slicer Python environment."
        return item.install_hint or "Use the required application runtime for this target."
    if getattr(item, "install_group", "") == "model-resources":
        return item.install_hint or "Choose a model asset root and download/verify required model assets."
    if item.kind in {"command", "path", "env"}:
        return item.install_hint or "Choose an existing local path or show manual setup instructions."
    return item.install_hint or "Show setup instructions."


def environment_dialog_state(result: Any, *, python_install_target: str) -> EnvironmentDialogState:
    """Build render state for the environment-results dialog."""

    missing_items = tuple(environment_missing_items(result))
    has_model_resources = any(getattr(item, "install_group", "") == "model-resources" for item in missing_items)
    return EnvironmentDialogState(
        summary_text=environment_summary_text(result),
        detail_text="\n".join(environment_detail_lines(result)),
        missing_items=missing_items,
        can_install_python=bool(result.missing_python_packages) or has_model_resources or any(
            item.required and getattr(item, "install_group", "") in {"mrtrix", "fsl", "brain_mask", "slicer-runtime", "slicer-dmri", "ukf"}
            for item in missing_items),
        has_manual_setup_notes=any(
            item.status == "missing"
            and (item.kind != "python" or not (item.details or {}).get("package"))
            for item in result.statuses
        ),
        python_install_tooltip=(
            python_install_target
            + " Prepare only required selected items after confirmation; Python packages use their selected interpreter. "
            "Supported external tools use isolated runtimes; licenses require approval and some extensions require a restart."
        ),
    )
