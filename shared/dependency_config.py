"""Shared dependency-config schema and helpers.

The same external dependency profile is consumed by the Slicer UI, the subject
CLI, and the install planner. Keeping the key list here prevents drift between
entry points when a new runtime path is added.
"""

from __future__ import annotations

import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any


DEPENDENCY_CONFIG_KIND = "slicerdsurfer_dependency_config"
DEPENDENCY_PROFILE_KIND = "slicerdsurfer_dependency_profile"
DEPENDENCY_CONFIG_VERSION = 1

DEPENDENCY_CONFIG_KEYS = (
    "python_executable",
    "python_environment",
    "conda_executable",
    "model_assets_dir",
    "mrtrix_bin_dir",
    "preprocessing_mrtrix_bin_dir",
    "preprocessing_fsl_bin_dir",
    "postprocess_freesurfer_home",
    "postprocess_runtime",
    "postprocess_license_file",
    "pnl_mask_python",
    "pnl_mask_script",
    "pnl_mask_model_dir",
    "pnl_mask_ants_bin_dir",
    "dki_matlab_executable",
    "dki_mkcurve_root",
    "wma_root",
    "wma_atlas_dir",
    "slicer_executable",
    "fiber_tract_measurements_cli",
    "tract_querier_root",
    "wmql_python_executable",
    "tractcloud_root",
    "tractcloud_python_executable",
    "tractcloud_data_dir",
)

DEPENDENCY_CONFIG_HELP = {
    "python_executable": "Python executable used for Python dependency checks and pip installs.",
    "python_environment": "Conda environment name or Python executable path used for Python dependency checks and pip installs.",
    "conda_executable": "conda or mamba executable used to resolve --python-environment names.",
    "model_assets_dir": "Directory containing downloaded SlicerDSpace model weights and templates.",
    "mrtrix_bin_dir": "Directory containing MRtrix3 commands for tractography/connectome routes.",
    "preprocessing_mrtrix_bin_dir": "Directory containing MRtrix3 commands for MRtrix/FSL preprocessing.",
    "preprocessing_fsl_bin_dir": "Directory containing FSL commands for MRtrix/FSL preprocessing.",
    "postprocess_freesurfer_home": "FreeSurfer installation root for DDSurfer post-processing.",
    "postprocess_runtime": "Optional locally compiled surface component directory; takes precedence over a full FreeSurfer installation.",
    "postprocess_license_file": "Personal FreeSurfer activation file outside the runtime; otherwise FS_LICENSE is used.",
    "pnl_mask_python": "Python executable for the externally configured PNL CNN dMRI masking environment.",
    "pnl_mask_script": "Official PNL CNN dwi_masking.py script path.",
    "pnl_mask_model_dir": "Directory containing the externally configured PNL CNN masking models.",
    "pnl_mask_ants_bin_dir": "Directory containing ANTs commands used by PNL CNN masking.",
    "dki_matlab_executable": "External MATLAB executable for the author's original MK-Curve method.",
    "dki_mkcurve_root": "Checkout directory containing the author's original MATLAB MK-Curve code.",
    "wma_root": "WhiteMatterAnalysis checkout or installation root.",
    "wma_atlas_dir": "Directory for the selected ORG, East-West or NABA atlas used by WhiteMatterAnalysis.",
    "slicer_executable": "Slicer executable used by external bundle-analysis helpers.",
    "fiber_tract_measurements_cli": "FiberTractMeasurements CLI executable.",
    "tract_querier_root": "TractQuerier checkout root when not installed in the active Python.",
    "wmql_python_executable": "Optional dedicated Python executable for the configured TractQuerier checkout.",
    "tractcloud_root": "TractCloud checkout or installation root.",
    "tractcloud_python_executable": "Python executable for TractCloud.",
    "tractcloud_data_dir": "TractCloud model/data directory.",
}

UI_DEPENDENCY_PATHS: tuple[dict[str, str], ...] = (
    {
        "id": "python_executable",
        "label": "Python executable",
        "section": "core-runtime",
        "mode": "file",
        "tooltip": "Optional. Python executable for dependency checks and Python package installation. Defaults to current Slicer Python.",
    },
    {
        "id": "python_environment",
        "label": "Python/conda environment",
        "section": "core-runtime",
        "mode": "file",
        "value_type": "text",
        "tooltip": "Optional. Conda environment name or Python executable path. Explicit Python executable takes precedence.",
    },
    {
        "id": "conda_executable",
        "label": "conda/mamba executable",
        "section": "core-runtime",
        "mode": "file",
        "tooltip": "Optional. conda or mamba executable used when Python/conda environment is an environment name.",
    },
    {
        "id": "model_assets_dir",
        "label": "Model assets root",
        "section": "model-assets",
        "mode": "dir",
        "tooltip": "Optional. Root directory for downloaded SlicerDSpace model assets. Defaults to SLICERDSPACE_MODEL_ASSETS_DIR or a per-user data directory.",
    },
    {
        "id": "mrtrix_bin_dir",
        "label": "MRtrix3 bin dir",
        "section": "core-runtime",
        "mode": "dir",
        "tooltip": "Optional. Directory containing mrconvert, dwi2response, dwi2fod, tckgen, tcksift2.",
    },
    {
        "id": "preprocessing_fsl_bin_dir",
        "label": "FSL bin dir",
        "section": "core-runtime",
        "mode": "dir",
        "tooltip": "Optional. Directory containing eddy and topup for MRtrix/FSL preprocessing.",
    },
    {
        "id": "postprocess_freesurfer_home",
        "label": "FreeSurfer home (optional)",
        "section": "core-runtime",
        "mode": "dir",
        "tooltip": "Optional full installation for surface processing. Locally compiled surface components do not require the full suite; FREESURFER_HOME is also detected.",
    },
    {
        "id": "postprocess_runtime",
        "label": "Surface components (optional)",
        "section": "core-runtime",
        "mode": "dir",
        "tooltip": "Local source-built runtime containing runtime.json. Leave empty to reuse the recipe-matched user cache or an existing FreeSurfer installation.",
    },
    {
        "id": "postprocess_license_file",
        "label": "FreeSurfer activation file",
        "section": "core-runtime",
        "mode": "file",
        "tooltip": "Your own activation file, kept outside the runtime. Required by both local components and full FreeSurfer; FS_LICENSE is detected. Never downloaded or bundled by the plugin.",
    },
    {
        "id": "pnl_mask_python",
        "label": "PNL mask Python",
        "section": "core-runtime",
        "mode": "file",
        "tooltip": "Python executable in the external PNL CNN dMRI masking environment; not the general dependency-install Python.",
    },
    {
        "id": "pnl_mask_script",
        "label": "PNL dwi_masking.py",
        "section": "core-runtime",
        "mode": "file",
        "tooltip": "Path to the official external PNL CNN dwi_masking.py script. No alternative masking backend is selected automatically.",
    },
    {
        "id": "pnl_mask_model_dir",
        "label": "PNL mask model dir",
        "section": "core-runtime",
        "mode": "dir",
        "tooltip": "External PNL CNN model directory. Configure it when generating a missing mask with PNL CNN; supplied masks are reused.",
    },
    {
        "id": "pnl_mask_ants_bin_dir",
        "label": "PNL mask ANTs bin dir",
        "section": "core-runtime",
        "mode": "dir",
        "tooltip": "Directory containing ANTs executables required by the external PNL CNN masking script.",
    },
    {
        "id": "dki_matlab_executable",
        "label": "MK-Curve MATLAB executable",
        "section": "core-runtime",
        "mode": "file",
        "tooltip": "External MATLAB executable for MKCURVE_MATLAB only. Required when MKCURVE_MATLAB is selected; missing MATLAB blocks execution without fallback.",
    },
    {
        "id": "dki_mkcurve_root",
        "label": "Original MK-Curve checkout",
        "section": "core-runtime",
        "mode": "dir",
        "tooltip": "Directory containing the author's original external MATLAB MK-Curve code. Standard DKI WLS/OLS does not use this path.",
    },
    {
        "id": "tract_querier_root",
        "label": "TractQuerier root",
        "section": "core-runtime",
        "mode": "dir",
        "tooltip": "Optional. Checkout root used by WMQL when tract_querier is not installed in Slicer Python.",
    },
    {
        "id": "wmql_python_executable",
        "label": "WMQL Python",
        "section": "core-runtime",
        "mode": "file",
        "tooltip": "Optional. Use a dedicated Python executable with a native TractQuerier checkout.",
    },
    {
        "id": "wma_root",
        "label": "WMA root",
        "section": "bundle-backends",
        "mode": "dir",
        "tooltip": "Optional. External WhiteMatterAnalysis checkout root.",
    },
    {
        "id": "wma_atlas_dir",
        "label": "WMA atlas directory",
        "section": "bundle-backends",
        "mode": "dir",
        "tooltip": "Optional. Root containing the registration and 800-cluster directories of the atlas selected in Advanced settings.",
    },
    {
        "id": "slicer_executable",
        "label": "Slicer executable for WMA",
        "section": "bundle-backends",
        "mode": "file",
        "tooltip": "Optional. Slicer executable used by WMA/SlicerDMRI helper calls.",
    },
    {
        "id": "fiber_tract_measurements_cli",
        "label": "FiberTractMeasurements CLI",
        "section": "bundle-backends",
        "mode": "file",
        "tooltip": "Optional. SlicerDMRI FiberTractMeasurements CLI path.",
    },
    {
        "id": "tractcloud_root",
        "label": "TractCloud root",
        "section": "bundle-backends",
        "mode": "dir",
        "tooltip": "Optional. TractCloud package/check-out root.",
    },
    {
        "id": "tractcloud_python_executable",
        "label": "TractCloud Python",
        "section": "bundle-backends",
        "mode": "file",
        "tooltip": "Optional. Dedicated Python executable for TractCloud if not using Slicer Python.",
    },
    {
        "id": "tractcloud_data_dir",
        "label": "TractCloud data/model dir",
        "section": "bundle-backends",
        "mode": "dir",
        "tooltip": "Optional. TractCloud model/data directory.",
    },
)


def cli_flag_for_key(key: str) -> str:
    return "--" + key.replace("_", "-")


TEXT_CONFIG_KEYS = {"python_environment"}


def add_dependency_config_arguments(parser: ArgumentParser, *, exclude: tuple[str, ...] = ()) -> None:
    """Register external dependency path options on an argparse parser."""

    excluded = set(exclude)
    for key in DEPENDENCY_CONFIG_KEYS:
        if key in excluded:
            continue
        value_type = str if key in TEXT_CONFIG_KEYS else Path
        parser.add_argument(cli_flag_for_key(key), dest=key, type=value_type, help=DEPENDENCY_CONFIG_HELP.get(key))


def load_dependency_config(path: Path) -> dict[str, Any]:
    """Load a CLI/UI dependency config and return path-valued settings."""

    data = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if "external_dependencies" in data and isinstance(data["external_dependencies"], dict):
        data = data["external_dependencies"]
    result: dict[str, Any] = {}
    for key in DEPENDENCY_CONFIG_KEYS:
        value = data.get(key)
        if value:
            result[key] = str(value).strip() if key in TEXT_CONFIG_KEYS else Path(str(value)).expanduser()
    return result


def apply_dependency_config(args: Namespace, path: Path | None) -> None:
    """Apply dependency config values to argparse args if not already set."""

    if not path:
        return
    config_path = path.expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Dependency config not found: {config_path}")
    for key, value in load_dependency_config(config_path).items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)


def dependency_config_payload(args: Namespace) -> dict[str, object]:
    """Build the portable dependency config payload from argparse args."""

    return {
        "kind": DEPENDENCY_CONFIG_KIND,
        "version": DEPENDENCY_CONFIG_VERSION,
        "external_dependencies": {
            key: (
                str(getattr(args, key)).strip()
                if key in TEXT_CONFIG_KEYS
                else getattr(args, key).expanduser().as_posix()
            )
            for key in DEPENDENCY_CONFIG_KEYS
            if getattr(args, key, None) is not None
        },
    }


def write_dependency_config(args: Namespace, path: Path) -> Path:
    """Write a dependency config JSON for reuse by CLI, installer, or UI."""

    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(dependency_config_payload(args), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return resolved
