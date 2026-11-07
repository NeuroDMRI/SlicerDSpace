"""Pure, inspectable preparation plan for only the selected workflow."""

from datetime import datetime
from pathlib import Path

from shared.managed_runtimes import supported_platform, runtime_root
from shared.ui_environment_panel import environment_missing_items


from shared.wsl_bridge import WSL_INSTALL_COMMAND, WSL_MISSING_LABEL, list_distributions, select_distribution

PNL_UNSUPPORTED_TEXT = "Automatic brain mask is not available on this system yet. Provide a brain mask file (NIfTI)."


def build_setup_plan(result, *, installed_extensions=(), loaded_extensions=None, need_dti=False, need_torch=False, need_ukf=False,
                     need_slicerdmri=False, brain_mask_backend="pnl_cnn", wma_atlas="org", gpu_available=False, system=None, machine=None,
                     existing_managed_profiles=(), repair=False, backend_paths=None, closure_targets=(),
                     wsl_distributions=None, slicer_executable=None, dcm2niix_missing=False, pnl_mask_route="auto"):
    # SC-1c: the managed pnl_cnn profile (and its licence) is for the managed route only. The native route
    # needs torch and the brain_mask assets (the pnl-cnn optional profile), not the conda runtime.
    from shared.brain_mask import resolve_pnl_mask_route
    pnl_managed = resolve_pnl_mask_route(pnl_mask_route, system) == "managed"
    installed = set(installed_extensions)
    # DTI-based targets use the native Teem tensor fit (Slicer core), so need_dti adds no extension.
    # SlicerDMRI is needed only where a step still loads its modules (the WMA harden-transform shim).
    # F1 (SC-1 fix round): the native mask runs torch in Slicer's interpreter, so a required pnl_cnn mask on the
    # native route needs PyTorch even when no segmentation, parcellation, surface or TractCloud target is selected.
    native_mask_torch = (brain_mask_backend == "pnl_cnn" and not pnl_managed and
                         any(item.key == "brain_mask_backend" and item.required for item in result.statuses))
    required_extensions = (["SlicerDMRI"] if need_slicerdmri else []) + (["PyTorch"] if need_torch or native_mask_torch else [])
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
    if brain_mask_backend == "pnl_cnn" and pnl_managed and any(item.key == "brain_mask_backend" for item in missing):
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
    windows_runtime = None
    if native_windows:
        windows_runtime = windows_runtime_plan(
            groups, closure_targets, wsl_distributions=wsl_distributions, slicer_executable=slicer_executable,
            license_file=backend_paths.get("postprocess_license_file"))
        if windows_runtime["command"]:
            covered.update(windows_runtime["covers"])
            for name in ("mrtrix", "fsl_cpu", "fsl_cuda"):
                if name in managed and name != "pnl_cnn":
                    managed.remove(name)
            supported_managed = [name for name in supported_managed if name not in {"mrtrix", "fsl_cpu", "fsl_cuda"}]
            unsupported_managed = [name for name in unsupported_managed if name not in {"mrtrix", "fsl_cpu", "fsl_cuda"}]
            unsupported_notes = {key: value for key, value in unsupported_notes.items() if key in unsupported_managed}
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
    if dcm2niix_missing:
        # S-B3: the pinned dcm2niix wheel always goes into Slicer's own Python, never into an external
        # interpreter chosen for TractCloud; it is pending even when nothing else is missing.
        from scripts.setup_python_packages import DCM2NIIX_REQUIREMENT
        if DCM2NIIX_REQUIREMENT not in packages:
            packages.append(DCM2NIIX_REQUIREMENT)
        python_requests.append(dict(requirement=DCM2NIIX_REQUIREMENT, python_executable=None, slicer_python=True))
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
    optional = missing_optional_profiles(result, brain_mask_backend, wma_atlas, pnl_route="managed" if pnl_managed else "native")
    optional_plan = prepare(optional) if optional else {}
    handled_keys = {item.key for item in missing if
                    (any(name.startswith("wma") for name in optional) and item.key.startswith("wma_org_atlas")) or
                    ("tractcloud" in optional and (item.key == "tractcloud_data_dir" or item.key.startswith("tractcloud_data:"))) or
                    # SC-1c: a native-route mask item whose only gaps are the model files is fixed by the pnl-cnn resources.
                    ("pnl-cnn" in optional and item.key == "brain_mask_backend")}
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
                             for item in missing if item.install_group not in covered and item.key not in handled_keys]
                            + ([{"key": "wsl2_ubuntu", "label": WSL_MISSING_LABEL, "hint": WSL_INSTALL_COMMAND}]
                               if windows_runtime and windows_runtime["needed"] and not windows_runtime["distribution"] else []),
            "windows_runtime": windows_runtime,
            "fsl_license_required": (any(name.startswith("fsl_") for name in managed) and platforms_ok)
                                    or bool(windows_runtime and windows_runtime["command"]),
            "pnl_terms_required": "pnl_cnn" in managed and platforms_ok,
            "repair": bool(repair),
            "scope": "Dependency preparation only; installation is not scientific QC"}


def windows_install_command(distribution, *, slicer_executable=None, install_freesurfer=False, license_file=None,
                            skip_large_downloads=False, script=None):
    """Argument list of the Windows ps1 call (B5). The caller adds the program, `powershell`."""
    script = Path(script) if script else Path(__file__).resolve().parents[1] / "scripts" / "install_deps_windows.ps1"
    command = ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script), "-Distro", distribution]
    if slicer_executable:  # SPEC-GAP: -SlicerExecutable is left out when the running Slicer path is unknown
        command += ["-SlicerExecutable", str(slicer_executable)]
    command += ["-AcceptFslLicense", "-SkipModelAssets", "-SkipSlicerPythonInstall"]
    if install_freesurfer:
        command += ["-FreeSurferLicensePath", str(license_file)]
    else:
        command.append("-SkipFreeSurfer")
    if skip_large_downloads:
        command.append("-SkipLargeDownloads")
    return ["powershell", *command]


def windows_runtime_plan(groups, closure_targets, *, wsl_distributions=None, slicer_executable=None, license_file=None):
    """What the Windows setup does for the missing install groups: WSL distribution, ps1 switches (B5)."""
    from shared.surface_license import resolve_surface_license, usable_license

    needed = bool(set(groups) & {"mrtrix", "fsl", "freesurfer"})
    distributions = list_distributions() if wsl_distributions is None and needed else (wsl_distributions or [])
    distribution = select_distribution(distributions)
    freesurfer_required = "freesurfer" in groups
    found = None
    if freesurfer_required:
        candidate = resolve_surface_license(license_file)
        found = candidate if usable_license(candidate) else None
    install_freesurfer = bool(freesurfer_required and found)
    wma = "wma" in tuple(closure_targets)
    command = None
    if needed and distribution:
        command = windows_install_command(
            distribution, slicer_executable=slicer_executable, install_freesurfer=install_freesurfer,
            license_file=found, skip_large_downloads=not install_freesurfer and not wma)
    covers = {"mrtrix", "fsl"} | ({"freesurfer"} if install_freesurfer else set())
    return {"needed": needed, "distribution": distribution, "freesurfer_required": freesurfer_required,
            "freesurfer_license": str(found) if found else None, "installs_freesurfer": install_freesurfer,
            "wma_in_closure": wma, "command": command, "covers": sorted(covers)}


def python_package_commands(packages, *, requests=None, python_command, folder, statuses=(), tract_querier_root=None,
                            root_expected=False, slicer_command=None):
    """Group requests by their selected Python, never mix runtime targets. Pure, without a GUI object (B7).

    A request with `slicer_python` set goes to the `slicer_command` (program, prefix) pair when one is given (S-B3).
    """
    import importlib.metadata
    import os
    import sys
    from shared.managed_runtimes import write_json
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    script = Path(__file__).resolve().parents[1] / "scripts/setup_python_packages.py"
    groups = {}
    from shared.python_setup import _packaging_api
    InvalidRequirement, Requirement, canonicalize = _packaging_api()

    def package_name(value):
        try:
            return canonicalize(Requirement(value).name)
        except InvalidRequirement:
            return value

    selections = [(item['requirement'], item) for item in requests] if requests is not None else [(package, None) for package in packages]
    for package, explicit in selections:
        matches = [explicit] if explicit is not None else [(item.details or {}) for item in statuses
                        if item.kind == "python" and any(value and package_name(package) == package_name(value)
                            for value in ((item.details or {}).get("package"), (item.details or {}).get("install_requirement")))]
        for details in matches or [{}]:
            if details.get("slicer_python") and slicer_command:
                program, prefix = slicer_command
            else:
                program, prefix = python_command(details.get("python_executable"))
            group = groups.setdefault((str(program), tuple(prefix)), {"requirements": [], "environment": {}})
            requirement = details.get("install_requirement") or package
            try:
                if Requirement(package).specifier:
                    requirement = package
            except InvalidRequirement:
                pass
            if requirement == "git+https://github.com/demianw/tract_querier.git":
                root = tract_querier_root() if callable(tract_querier_root) else tract_querier_root
                if root and (root_expected or (Path(root) / "pyproject.toml").is_file()):
                    requirement = str(root)
                    group["environment"]["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_TRACT_QUERIER"] = "0+local"
            if requirement not in group["requirements"]:
                group["requirements"].append(requirement)
    loaded = {name for module, names in importlib.metadata.packages_distributions().items()
              if module in sys.modules for name in names}
    commands, receipts = [], []
    for index, ((program, prefix), data) in enumerate(groups.items()):
        slicer_python = bool(prefix) or Path(program).absolute() == Path(sys.executable).absolute()
        requirements = data["requirements"]
        try:
            if all(not Requirement(value).url for value in requirements):
                data['environment']['PIP_ONLY_BINARY'] = ':all:'
        except InvalidRequirement:
            pass
        if not slicer_python:
            from shared.external_process import external_python_environment
            clean = external_python_environment(path_prepend=[Path(program).parent])
            overrides = {key: None for key in os.environ if key not in clean}
            overrides.update(clean)
            overrides.update(data["environment"])
            data["environment"] = overrides
        torch_minimum = None
        torchvision_required = any(item == "torchvision" or item.startswith(("torchvision>", "torchvision=", "torchvision<"))
                                   for item in requirements)
        if slicer_python and (torchvision_required or any(item == "torch" or item.startswith(("torch>", "torch=", "torch<")) for item in requirements)):
            from packaging.requirements import Requirement
            from packaging.version import Version
            minimums = ["2.0.0"]
            for value in requirements:
                if value.startswith('torch>='):
                    minimums.extend(spec.version for spec in Requirement(value).specifier if spec.operator == '>=')
            torch_minimum = str(max(map(Version, minimums)))
            requirements = [item for item in requirements if not (item == "torch" or item.startswith(("torch>", "torch=", "torch<")))]
        request, receipt = (folder / f"python-{stamp}-{index}-{suffix}.json" for suffix in ("request", "result"))
        write_json(request, {"requirements": requirements, "torch_minimum": torch_minimum,
                            "torchvision_required": torchvision_required and slicer_python,
                            "protect_bundled": slicer_python, "loaded_packages": sorted(loaded) if slicer_python else [],
                            "python_path_entries": [path for path in sys.path if path] if slicer_python else []})
        commands.append((program, [*prefix, "-u", str(script), "--request", str(request), "--result", str(receipt)], data["environment"]))
        receipts.append(receipt)
    return commands, receipts


# Receipt kinds a setup step can produce. Each is validated the way the GUI handlers did before B8.
RUNTIME_FILE_SETTINGS = {"pnl_mask_python", "pnl_mask_script"}
RUNTIME_SETTINGS = {"mrtrix_bin_dir", "preprocessing_mrtrix_bin_dir", "preprocessing_fsl_bin_dir", "postprocess_runtime",
                    "pnl_mask_python", "pnl_mask_script", "pnl_mask_model_dir", "pnl_mask_ants_bin_dir"}


# Settings that scripts/install_deps_windows.ps1 writes into the user config (its external_dependencies block).
WINDOWS_WRITTEN_SETTINGS = ("tract_querier_root", "wma_root", "wma_atlas_dir", "tractcloud_root", "tractcloud_python_executable",
                            "tractcloud_data_dir", "slicer_executable", "python_executable", "postprocess_freesurfer_home",
                            "model_assets_dir")


def validate_setup_receipt(step, receipt):
    """Return the settings a verified receipt may save; raise ValueError for anything else (B8)."""
    kind = step.get("id")
    if kind in {"managed_runtime", "surface_runtime"}:
        # Same allow-list and path checks as shared/ui_managed_runtimes.py before B8.
        if (not isinstance(receipt, dict) or receipt.get("kind") != "slicerdsurfer_runtime_setup"
                or receipt.get("status") != "completed"):
            raise ValueError("Only a completed managed runtime setup can update paths")
        settings = receipt.get("external_dependencies", {})
        # A5 N4: a migration that keeps every user-set path leaves no settings to save. The install itself succeeded.
        if not settings and isinstance(receipt.get("migration"), dict):
            return {}
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
    if kind == "windows_runtime":
        # The ps1 receipt is the user config it wrote (B5): the bridge folders must exist and belong to the chosen distribution.
        from shared.wsl_bridge import check_bridge_distribution
        wanted = {"mrtrix_bin_dir", "preprocessing_mrtrix_bin_dir", "preprocessing_fsl_bin_dir"}
        written = (receipt or {}).get("external_dependencies") or {}
        settings = {key: str(written[key]) for key in wanted if written.get(key)}
        distribution = str(step.get("distribution") or "")
        if set(settings) != wanted or not distribution or any(not Path(value).is_dir() for value in settings.values()):
            raise ValueError("Windows runtime setup did not write the WSL bridge folders")
        configs = [path for value in settings.values() for path in sorted(Path(value).glob("*.conf"))]
        if not configs or any(check_bridge_distribution(Path(str(path)[:-len(".conf")]), distribution) is None for path in configs):
            raise ValueError("WSL bridges are missing or were not written for distribution " + distribution)
        return settings
    if kind in {"python_packages", "self_check"}:
        return {}
    raise ValueError("Unknown setup step: " + str(kind))


def apply_setup_receipt(step, receipt):
    """Validate a step receipt, save its settings in the user config and return them (B8). No GUI owner."""
    from shared.user_config import update_user_config

    settings = validate_setup_receipt(step, receipt)
    if settings:
        update_user_config({"external_dependencies": settings})
    if step.get("id") == "windows_runtime":
        update_user_config({"wsl_distribution": str(step["distribution"])})  # SPEC-GAP: a top-level key, not an external dependency
    return settings


def need_flags(closure_targets, algorithms=("csd_act",), *, fallback_ukf=False):
    """The build_setup_plan need_* switches for one target closure (B7; one place for the GUI and the CLI)."""
    targets = set(closure_targets)
    return dict(
        need_dti=bool(targets & {"segmentation", "parcellation", "surface", "microstructure", "tractography", "wma"}),
        need_slicerdmri="wma" in targets,
        need_torch=bool(targets & {"segmentation", "parcellation", "surface", "tractcloud"}),
        need_ukf="tractography" in targets and ("ukf" in tuple(algorithms) or bool(fallback_ukf)))


def setup_steps(plan, accepted_licences, *, folder, launcher=None, python_command=None, config=None, statuses=()):
    """Ordered setup steps for one plan (B7). Each step names its commands and its receipt file.

    `launcher` is (program, prefix) that runs a repo script in Slicer's Python; the default is this interpreter.
    `python_command(override)` returns (program, prefix) for the Python step. `accepted_licences` is the user-config
    record {id: {"version", ...}}. A required licence that is not accepted raises PermissionError.
    The last step is the self-check; its commands are built at run time from a fresh doctor report (SPEC-GAP).
    """
    import sys
    from shared.setup_components import pending_licences, plan_components, required_licences

    components = plan_components(plan)
    pending = pending_licences(required_licences(components), accepted_licences)
    if pending:
        raise PermissionError("Accept these licences before installing: " + ", ".join(sorted(pending)))
    folder, config = Path(folder), dict(config or {})
    repo = Path(__file__).resolve().parents[1]
    program, prefix = launcher or (sys.executable, [])
    python_command = python_command or (lambda override: (override or sys.executable, []))

    def run(script, *args):
        return (program, [*prefix, "-u", str(repo / "scripts" / script), *args], {})

    steps = []
    if plan.get("backend_profiles"):
        receipt = folder / "backend-sources-result.json"
        steps.append(dict(id="backend_sources", title="Prepare algorithm sources", receipt=receipt,
                          profiles=list(plan["backend_profiles"]), root=str(plan["backend_root"]),
                          commands=[run("setup_backend_sources.py", "--profiles", *plan["backend_profiles"], "--root",
                                        str(plan["backend_root"]), "--execute", "--accept-terms", "--result", str(receipt))]))
    packages, requests = list(plan.get("python_packages", ())), list(plan.get("python_requests", ()))
    tract_root, expected_root = config.get("tract_querier_root"), False
    if "wmql" in plan.get("backend_profiles", ()):  # the backend step above puts the checkout here before this step runs
        from shared.backend_sources import profile_path
        tract_root, expected_root = profile_path("wmql", plan["backend_root"]), True
    if packages:
        commands, receipts = python_package_commands(
            tuple(packages), requests=requests, python_command=python_command, folder=folder, statuses=statuses,
            tract_querier_root=tract_root, root_expected=bool(expected_root), slicer_command=(program, list(prefix)))
        steps.append(dict(id="python_packages", title="Install Python packages", commands=commands,
                          receipt=receipts[0] if receipts else None, receipts=receipts, packages=packages))
    if plan.get("asset_profiles"):
        steps.append(dict(id="model_assets", title="Download and verify model assets", receipt=None,
                          asset_root=str(plan["asset_root"]), profiles=list(plan["asset_profiles"]),
                          commands=[run("download_model_assets.py", "--profile", name, "--asset-root", str(plan["asset_root"]),
                                        "--source", "url", "--source-id", "slicerdsurfer_models_git")
                                    for name in plan["asset_profiles"]]))
    if plan.get("optional_asset_profiles"):
        receipt = folder / "optional-resources-result.json"
        steps.append(dict(id="optional_resources", title="Prepare selected resources", receipt=receipt,
                          profiles=list(plan["optional_asset_profiles"]), root=str(plan["asset_root"]),
                          commands=[run("setup_optional_assets.py", "--profiles", *plan["optional_asset_profiles"],
                                        "--asset-root", str(plan["asset_root"]), "--execute", "--accept-terms",
                                        "--result", str(receipt))]))
    windows = plan.get("windows_runtime") or {}
    if windows.get("command"):
        from shared.user_config import default_user_config_path
        steps.append(dict(id="windows_runtime", title="Install the WSL runtime", receipt=default_user_config_path(),
                          distribution=windows["distribution"], commands=[(windows["command"][0], windows["command"][1:], {})]))
    elif plan.get("managed_profiles"):
        receipt = folder / "runtime-result.json"
        args = ["--profiles", *plan["managed_profiles"], "--execute", "--result", str(receipt)]
        args += ["--repair"] if plan.get("repair") else []
        args += ["--accept-fsl-license"] if plan.get("fsl_license_required") else []
        args += ["--accept-pnl-terms"] if plan.get("pnl_terms_required") else []
        args += ["--asset-root", str(config["model_assets_dir"])] if config.get("model_assets_dir") else []
        args += ["--conda", str(config["conda_executable"])] if config.get("conda_executable") else []
        steps.append(dict(id="managed_runtime", title="Prepare external tools", receipt=receipt,
                          commands=[run("setup_managed_runtime.py", *args)]))
    if plan.get("surface_build_required") or plan.get("surface_prebuilt_required"):
        receipt = folder / "surface-runtime-result.json"
        args = ["--execute", "--accept-terms", "--jobs", "4", "--result", str(receipt)]
        for key, flag in (("model_assets_dir", "--asset-root"), ("postprocess_runtime", "--prefix")):
            if config.get(key):
                args += [flag, str(config[key])]
        steps.append(dict(id="surface_runtime", title="Install surface components", receipt=receipt,
                          commands=[run("setup_surface_runtime.py", *args)]))
    steps.append(dict(id="self_check", title="Test the installation", receipt=None, commands=[],
                      script=str(repo / "scripts" / "check_setup_functionality.py")))
    from shared.setup_components import unlisted_licences
    unlisted = unlisted_licences(steps, components)
    if unlisted:  # S-B8: an install without a listed licence does not run
        raise PermissionError("No licence is listed for: " + ", ".join(unlisted))
    return steps


def complete_step(step):
    """After a step's commands succeeded: read its receipt(s), validate them, save the settings (B7/B8)."""
    import json

    if step["id"] == "python_packages":
        records = []
        for path in step.get("receipts", ()):
            try:
                records.append(json.loads(Path(path).read_text()))
            except (OSError, ValueError):
                records.append({})
        if any(record.get("status") != "completed" for record in records):
            raise ValueError("Python package installation did not complete")
        step["restart_required"] = any(record.get("restart_required") for record in records)
        return {}
    path = step.get("receipt")
    if step["id"] == "windows_runtime":
        # S-N3: the ps1 writes %APPDATA%\SlicerDSpace\user_config.json. Resolve the path now, after it ran: while only the
        # legacy SlicerDDSurfer folder existed, the path chosen when the step was built pointed there.
        from shared.user_config import default_user_config_path
        path = default_user_config_path()
    receipt = json.loads(Path(path).read_text()) if path else None
    settings = apply_setup_receipt(step, receipt)
    if step["id"] == "windows_runtime":
        # S-B2: the ps1 wrote more paths than the receipt checks. Return them all, so the GUI loads them into
        # its settings and line edits and a later save of the common settings cannot erase them.
        written = (receipt or {}).get("external_dependencies") or {}
        settings = {**{key: str(written[key]) for key in WINDOWS_WRITTEN_SETTINGS if written.get(key)}, **settings}
    return settings


def run_setup_steps(steps, *, runner=None, self_check=None, log=print, after_step=None):
    """Run the steps one after the other (CLI). Stops at the first failure. Returns one record per step.

    `self_check(step)` returns (commands, receipts) for the last step; a receipt must report status "passed".
    `after_step(step, settings)` runs after a step was validated and its settings were saved.
    """
    import json
    import os
    import subprocess

    runner = runner or subprocess.run
    records = []
    for step in steps:
        log("[Setup] " + step["title"])
        commands, receipts = step["commands"], []
        if step["id"] == "self_check":
            commands, receipts = self_check(step) if self_check else ([], [])
        status, details = "completed", ""
        for program, args, extra in commands:
            environment = dict(os.environ)
            for key, value in (extra or {}).items():
                if value is None:
                    environment.pop(key, None)
                else:
                    environment[key] = value
            if runner([program, *args], env=environment, check=False).returncode != 0:
                status, details = "failed", "Command failed: " + str(program)
                break
        settings = {}
        if status == "completed":
            try:
                if step["id"] == "self_check":
                    if any(json.loads(Path(path).read_text()).get("status") != "passed" for path in receipts):
                        raise ValueError("Installation functionality check failed")
                    from shared.setup_selftest import receipt_warnings
                    for warning in receipt_warnings(receipts):  # A2: a warning does not block "ready"
                        log("[Setup] Warning: " + warning)
                else:
                    settings = complete_step(step)
                    if after_step:
                        after_step(step, settings)
            except Exception as exc:
                status, details = "failed", str(exc)
        records.append(dict(id=step["id"], status=status, details=details))
        if status != "completed":
            break
    return records
