"""Pure, inspectable preparation plan for only the selected workflow."""

from pathlib import Path

from shared.managed_runtimes import supported_platform, runtime_root
from shared.ui_environment_panel import environment_missing_items


PNL_UNSUPPORTED_TEXT = "Automatic brain mask is not available on this system yet. Provide a brain mask file (NIfTI)."


def build_setup_plan(result, *, installed_extensions=(), loaded_extensions=None, need_dti=False, need_torch=False, need_ukf=False,
                     need_slicerdmri=False, brain_mask_backend="pnl_cnn", wma_atlas="org", gpu_available=False, system=None, machine=None,
                     existing_managed_profiles=(), repair=False, backend_paths=None):
    installed = set(installed_extensions)
    # DTI-based targets use the native Teem tensor fit (Slicer core), so need_dti adds no extension.
    # SlicerDMRI is needed only where a step still loads its modules (the WMA harden-transform shim).
    required_extensions = (["SlicerDMRI"] if need_slicerdmri else []) + (["PyTorch"] if need_torch else [])
    if need_ukf:
        required_extensions.append("UKFTractography")
    missing = [item for item in environment_missing_items(result) if item.required]
    from shared.backend_sources import SOURCES, source_root, prepare_sources
    from pathlib import Path
    from shared.model_assets import configured_model_assets_root
    import platform
    native_windows = (system or platform.system()) == 'Windows'
    backend_paths = backend_paths or {}
    asset_value = backend_paths.get('model_assets_dir')
    asset_root = configured_model_assets_root(Path(asset_value) if asset_value else None)
    source_profiles = []
    unsupported_sources = []
    for name, source in SOURCES.items():
        root = backend_paths.get(source['setting'])
        if root and not Path(str(root)).expanduser().resolve().is_relative_to(source_root().resolve()):
            continue
        keys = {item.key for item in missing}
        packages_missing = {str((item.details or {}).get('package', '')).split('>=')[0]
                            for item in missing if item.kind == 'python'}
        if ((name == 'wmql' and ('tract_querier' in packages_missing or keys & {'tract_querier_root', 'tract_querier_root_script'}))
                or (name == 'wma' and keys & {'wma_root', 'wma_root_path', 'command:wm_apply_ORG_atlas_to_subject.sh'})
                or (name == 'tractcloud' and ('tractcloud' in packages_missing or 'tractcloud_root_path' in keys))):
            if name == 'wma' and native_windows:
                unsupported_sources.append(name)
            else:
                source_profiles.append(name)
    source_plan = prepare_sources(source_profiles) if source_profiles else {}
    if 'tractcloud' in source_profiles and 'PyTorch' not in required_extensions:
        required_extensions.append('PyTorch')
    groups = {item.install_group for item in missing}
    managed = list(existing_managed_profiles)
    if "mrtrix" in groups:
        managed.append("mrtrix")
    if "fsl" in groups:
        managed.append("fsl_cuda" if gpu_available else "fsl_cpu")
    if brain_mask_backend == "pnl_cnn" and any(item.key == "brain_mask_backend" for item in missing):
        managed.append("pnl_cnn")
    managed = list(dict.fromkeys(managed))
    supported_managed = [name for name in managed if supported_platform(system, machine, profiles=(name,))]
    unsupported_managed = [name for name in managed if name not in supported_managed]
    platforms_ok = supported_platform(system, machine)
    unsupported_notes = {"pnl_cnn": PNL_UNSUPPORTED_TEXT} if "pnl_cnn" in unsupported_managed else {}
    covered = {"model-resources", "python", "slicer-runtime", "slicer-dmri", "ukf"}
    from shared.prebuilt_surface import bundle_for_platform
    surface_prebuilt = bundle_for_platform(system, machine)
    surface_needed = any(item.key == 'surface_postprocess_runtime' and
                         (item.details or {}).get('build_required') for item in missing)
    surface_bundle_available = surface_prebuilt is not None and (surface_prebuilt.get('published', False) or
        (asset_root / surface_prebuilt['asset']).is_file())
    surface_install = surface_needed and surface_bundle_available
    if surface_install:
        covered.add('surface-runtime')
    if platforms_ok:
        covered.update(("mrtrix", "fsl"))
        if "pnl_cnn" in managed:
            covered.add("brain_mask")
    if 'mrtrix' in supported_managed:
        covered.add('mrtrix')
    assets = tuple(name for name in ("surface", "segmentation", "parcellation")
                   if any(name in item.targets and item.install_group == "model-resources" for item in missing))
    import json
    model_manifest = json.loads((Path(__file__).resolve().parents[1] / 'models/assets_manifest.json').read_text())
    model_names = {relative for name in assets for relative in model_manifest['profiles'][name]['required_paths']}
    core_model_bytes = sum(model_manifest['asset_records'].get(name, {}).get('size_bytes', 0) for name in model_names)
    packages = tuple(dict.fromkeys((item.details or {}).get("install_requirement") or (item.details or {}).get("package")
                                  for item in missing if item.kind == "python" and
                                  ((item.details or {}).get("install_requirement") or (item.details or {}).get("package"))))
    source_packages = {'tract_querier' if name == 'wmql' else name for name in source_profiles}
    packages = [package for package in packages if not any(
        package.lower().replace('-', '_') == name or
        ('/' + name in package.lower().replace('-', '_') and 'github.com' in package.lower())
        for name in source_packages)]
    backend_requirements = [requirement for name in source_profiles for requirement in SOURCES[name]['requirements']]
    packages = list(dict.fromkeys([*packages, *backend_requirements]))
    python_requests = []
    for item in missing:
        details = item.details or {}
        requirement = details.get('install_requirement') or details.get('package')
        if item.kind == 'python' and requirement in packages:
            python_requests.append(dict(requirement=requirement, python_executable=details.get('python_executable')))
    for name in source_profiles:
        if name == 'tractcloud':
            python = backend_paths.get('tractcloud_python_executable')
            candidates = [item for item in result.statuses if 'tractcloud' in item.targets and item.kind == 'python']
        elif name == 'wma':
            python = None
            candidates = [item for item in result.statuses if item.key.startswith('python:wma:')]
        else:
            python = backend_paths.get('python_executable')
            candidates = [item for item in result.statuses if item.key == 'python:tract_querier']
        if not python:
            python = next(((item.details or {}).get('python_executable') for item in candidates
                           if (item.details or {}).get('python_executable')), None)
        python_requests.extend(dict(requirement=requirement, python_executable=str(python) if python else None)
                               for requirement in SOURCES[name]['requirements'])
    from shared.optional_assets import missing_optional_profiles
    from scripts.setup_optional_assets import prepare
    optional = missing_optional_profiles(result, brain_mask_backend, wma_atlas)
    optional_plan = prepare(optional) if optional else {}
    handled_keys = {item.key for item in missing if
                    (any(name.startswith("wma") for name in optional) and item.key.startswith("wma_org_atlas")) or
                    ("tractcloud" in optional and (item.key == "tractcloud_data_dir" or item.key.startswith("tractcloud_data:")))}
    for item in missing:
        if ('wma' in source_profiles and item.install_group == 'wma' and 'atlas' not in item.key):
            handled_keys.add(item.key)
        if ('wmql' in source_profiles and item.install_group == 'wmql'):
            handled_keys.add(item.key)
        if ('tractcloud' in source_profiles and item.key in {'tractcloud_root', 'tractcloud_root_path'}):
            handled_keys.add(item.key)
    return {"kind": "slicerdsurfer_selected_setup_plan", "extensions": list(dict.fromkeys(
                name for name in required_extensions if name not in installed)),
            "restart_extensions": list(dict.fromkeys(name for name in required_extensions
                if loaded_extensions is not None and name in installed and name not in loaded_extensions)),
            "python_packages": list(packages), "asset_profiles": list(assets),
            "asset_root": str(asset_root), "core_model_profile_bytes": core_model_bytes,
            "backend_profiles": source_profiles, "backend_root": source_plan.get('root'),
            "unsupported_backend_profiles": unsupported_sources, "python_requests": python_requests,
            "backend_download_bytes": source_plan.get('download_bytes', 0),
            "backend_terms": source_plan.get('terms', []),
            "optional_asset_profiles": optional,
            "optional_asset_size_bytes": optional_plan.get("size_bytes", 0),
            "optional_asset_peak_disk_bytes": optional_plan.get("estimated_peak_disk_bytes", 0),
            "optional_asset_terms": optional_plan.get("terms", []),
            "managed_profiles": supported_managed,
            "runtime_root": str(runtime_root()),
            "surface_build_required": False, "surface_prebuilt_required": surface_install,
            "surface_download_bytes": surface_prebuilt['bytes'] if surface_install else 0,
            "surface_bundle_unavailable": bool(surface_needed and surface_prebuilt and not surface_bundle_available),
            "unsupported_profiles": unsupported_managed,
            "unsupported_notes": unsupported_notes,
            "manual_items": [{"key": item.key, "label": item.label, "hint":
                              'Windows WMA needs a configured WSL executable bridge; an author Bash checkout alone is not runnable. See docs/setup/WINDOWS.md.'
                              if native_windows and item.install_group == 'wma' and 'atlas' not in item.key
                              else PNL_UNSUPPORTED_TEXT if item.key == 'brain_mask_backend' and unsupported_notes
                              else item.install_hint}
                             for item in missing if item.install_group not in covered and item.key not in handled_keys],
            "fsl_license_required": any(name.startswith("fsl_") for name in managed) and platforms_ok,
            "pnl_terms_required": "pnl_cnn" in managed and platforms_ok,
            "repair": bool(repair),
            "scope": "Dependency preparation only; installation is not scientific QC"}


# Receipt kinds a setup step can produce. Each is validated the way the GUI handlers did before B8.
RUNTIME_FILE_SETTINGS = {"pnl_mask_python", "pnl_mask_script"}
RUNTIME_SETTINGS = {"mrtrix_bin_dir", "preprocessing_mrtrix_bin_dir", "preprocessing_fsl_bin_dir", "postprocess_runtime",
                    "pnl_mask_python", "pnl_mask_script", "pnl_mask_model_dir", "pnl_mask_ants_bin_dir"}


def validate_setup_receipt(step, receipt):
    """Return the settings a verified receipt may save; raise ValueError for anything else (B8)."""
    kind = step.get("id")
    if kind in {"managed_runtime", "surface_runtime"}:
        # Same allow-list and path checks as shared/ui_managed_runtimes.py before B8.
        if (not isinstance(receipt, dict) or receipt.get("kind") != "slicerdsurfer_runtime_setup"
                or receipt.get("status") != "completed"):
            raise ValueError("Only a completed managed runtime setup can update paths")
        settings = receipt.get("external_dependencies", {})
        if not settings or set(settings) - RUNTIME_SETTINGS:
            raise ValueError("Unexpected managed runtime settings")
        if any(not (Path(value).is_file() if key in RUNTIME_FILE_SETTINGS else Path(value).is_dir())
               for key, value in settings.items()):
            raise ValueError("Managed runtime paths are unavailable")
        return {key: str(value) for key, value in settings.items()}
    if kind == "optional_resources":
        from scripts.setup_optional_assets import PATH_SETTINGS
        record = receipt or {}
        if (record.get("kind") != "slicerdsurfer_optional_asset_setup" or record.get("status") != "completed"
                or record.get("profiles") != list(step["profiles"]) or record.get("root") != str(step["root"])):
            raise ValueError("Optional resource preparation receipt is incomplete")
        settings = record["external_dependencies"]
        allowed = {"model_assets_dir", *(PATH_SETTINGS[name] for name in step["profiles"] if name in PATH_SETTINGS)}
        if set(settings) != allowed or any(not Path(value).is_dir() for value in settings.values()):
            raise ValueError("Prepared resource paths are unavailable")
        return {key: str(value) for key, value in settings.items()}
    if kind == "backend_sources":
        from shared.backend_sources import SOURCES, verify_source, profile_path
        record = receipt or {}
        expected = {SOURCES[name]["setting"]: str(profile_path(name, step["root"])) for name in step["profiles"]}
        if (record.get("status") != "completed" or record.get("profiles") != list(step["profiles"])
                or record.get("external_dependencies") != expected):
            raise ValueError("Invalid prepared algorithm-source receipt")
        for name in step["profiles"]:
            verify_source(name, Path(expected[SOURCES[name]["setting"]]))
        return expected
    if kind == "model_assets":
        # SPEC-GAP: download_model_assets.py writes no receipt; a finished step saves the plan's asset root.
        root = Path(str(step["asset_root"]))
        if not root.is_dir():
            raise ValueError("Model asset folder is unavailable")
        return {"model_assets_dir": str(root)}
    if kind in {"python_packages", "self_check"}:
        return {}
    raise ValueError("Unknown setup step: " + str(kind))


def apply_setup_receipt(step, receipt):
    """Validate a step receipt, save its settings in the user config and return them (B8). No GUI owner."""
    from shared.user_config import update_user_config

    settings = validate_setup_receipt(step, receipt)
    if settings:
        update_user_config({"external_dependencies": settings})
    return settings
