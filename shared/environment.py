"""Environment doctor and install-plan helpers for SlicerDSpace."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Optional

from shared.external_process import SLICER_PYTHON_NAMES, find_pythonslicer, interpreter_path, slicer_runtime_roots, text_subprocess_kwargs, python_tool_environment
from shared.model_assets import model_asset_candidates, model_asset_path_or_none


# Dependency probes launch a short-lived Python subprocess (e.g. to check
# whether a package imports, or whether CUDA is available) and must never
# block the caller indefinitely: environment discovery can run on Slicer's
# main thread, and a hung probe (e.g. a broken CUDA driver making `import
# torch` never return) would freeze the whole application.
DEPENDENCY_PROBE_TIMEOUT_SECONDS = 30


def _probe_timeout_error(exc: subprocess.TimeoutExpired) -> str:
    return f"Dependency probe timed out after {exc.timeout:g}s"


@dataclass(frozen=True)
class DependencySpec:
    """One Python package, external command, environment variable, or resource."""

    key: str
    kind: str
    label: str
    required: bool
    targets: tuple[str, ...]
    python_import: Optional[str] = None
    python_package: Optional[str] = None
    python_install_requirement: Optional[str] = None
    python_executable: Optional[Path] = None
    python_path_entries: tuple[Path, ...] = ()
    device: Optional[str] = None
    command: Optional[str] = None
    bin_dir: Optional[Path] = None
    env_vars: tuple[str, ...] = ()
    path: Optional[Path] = None
    path_kind: str = "file"
    discovery_source: str = ""
    common_paths: tuple[Path, ...] = ()
    install_group: str = "python"
    notes: str = ""


@dataclass(frozen=True)
class DependencyStatus:
    key: str
    kind: str
    label: str
    status: str
    required: bool
    targets: tuple[str, ...]
    details: dict[str, Any] = field(default_factory=dict)
    install_hint: str = ""
    install_group: str = ""


@dataclass(frozen=True)
class TargetEnvironmentStatus:
    target: str
    status: str
    missing_required: tuple[str, ...] = ()
    missing_optional: tuple[str, ...] = ()


@dataclass(frozen=True)
class EnvironmentDoctorConfig:
    """Options needed to evaluate selected products and backend choices."""

    targets: tuple[str, ...]
    repo_root: Path
    output_dir: Optional[Path] = None
    python_executable: Optional[Path] = None
    python_path_entries: tuple[Path, ...] = ()
    loaded_slicer_modules: Optional[tuple[str, ...]] = None
    check_slicer_cli_modules: bool = False
    python_environment: Optional[str] = None
    conda_executable: Optional[Path] = None
    platform_name: Optional[str] = None
    brain_mask_backend: str = "pnl_cnn"
    brain_mask_device: str = "auto"
    brain_mask_required: bool = False
    pnl_mask_python: Optional[Path] = None
    pnl_mask_script: Optional[Path] = None
    pnl_mask_model_dir: Optional[Path] = None
    pnl_mask_ants_bin_dir: Optional[Path] = None
    pnl_mask_route: str = "auto"
    tractography_backend: str = "csd_act"
    ukf_executable: Optional[Path] = None
    dwiconvert_executable: Optional[Path] = None
    connectome_backend: str = "auto"
    ddseg_device: str = "auto"
    surface_device: str = "auto"
    tractcloud_device: str = "auto"
    run_free_water: bool = False
    run_noddi: bool = False
    run_dki: bool = False
    dki_fit_method: str = "MKCURVE"
    dki_matlab_executable: Optional[Path] = None
    dki_mkcurve_root: Optional[Path] = None
    mrtrix_bin_dir: Optional[Path] = None
    preprocessing_mrtrix_bin_dir: Optional[Path] = None
    preprocessing_fsl_bin_dir: Optional[Path] = None
    postprocess_freesurfer_home: Optional[Path] = None
    postprocess_runtime: Optional[Path] = None
    postprocess_license_file: Optional[Path] = None
    model_assets_dir: Optional[Path] = None
    wma_root: Optional[Path] = None
    wma_atlas_dir: Optional[Path] = None
    wma_atlas: str = "org"
    slicer_executable: Optional[Path] = None
    fiber_tract_measurements_cli: Optional[Path] = None
    tract_querier_root: Optional[Path] = None
    tractcloud_root: Optional[Path] = None
    tractcloud_python_executable: Optional[Path] = None
    tractcloud_data_dir: Optional[Path] = None


def run_needs_brain_mask(subject_root: Path, *, targets, subject_id: str, output_root=None,
                         brain_mask_backend='pnl_cnn', force=False, resume=True) -> bool:
    """Common CLI/API mask prerequisite policy, including stale preprocessing outputs."""
    from shared.io_layout import subject_layout
    subject_root = subject_root.expanduser().resolve()
    input_dir = subject_root / 'input'
    raw = all((input_dir / name).is_file() for name in ('data.nii.gz', 'bvals', 'bvecs'))
    if not raw:
        from shared.dicom_input import is_dicom_root
        raw = is_dicom_root(subject_root)
    from shared.target_closure import MASK_TARGETS
    mask_targets = set(MASK_TARGETS)
    required = raw and bool(mask_targets.intersection(targets)) and not (input_dir / 'nodif_brain_mask.nii.gz').is_file()
    # W2a (D1): preprocessing always runs FSL eddy, so it always writes a new corrected DWI and needs a new mask.
    if raw and 'preprocessing' in targets and brain_mask_backend != 'provided':
        layout = subject_layout(subject_root, output_root=output_root)
        stem = subject_id or subject_root.name
        dwi, mask = layout.preproc / f'{stem}_dwi_final.nii.gz', layout.preproc / f'{stem}_brain_mask_final.nii.gz'
        files = (dwi, mask, layout.preproc / f'{stem}_dwi_final.bval', layout.preproc / f'{stem}_dwi_final.bvec')
        stale = force or not resume or not all(path.is_file() for path in files)
        if not stale:
            stale = mask.stat().st_mtime_ns < dwi.stat().st_mtime_ns
        required = required or stale
    return bool(required)


@dataclass(frozen=True)
class EnvironmentDoctorResult:
    platform: str
    host_platform: str
    python_executable: str
    targets: tuple[str, ...]
    statuses: tuple[DependencyStatus, ...]
    target_statuses: tuple[TargetEnvironmentStatus, ...]
    blocker_count: int
    missing_python_packages: tuple[str, ...]
    missing_external_groups: tuple[str, ...]
    tracking_backend_blockers: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def runnable_targets(self) -> tuple[str, ...]:
        return tuple(item.target for item in self.target_statuses if item.status == "runnable")

    @property
    def blocked_targets(self) -> tuple[str, ...]:
        return tuple(item.target for item in self.target_statuses if item.status == "blocked")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "environment_doctor",
            "version": 1,
            "platform": self.platform,
            "host_platform": self.host_platform,
            "python_executable": self.python_executable,
            "targets": list(self.targets),
            "summary": {
                "blocker_count": self.blocker_count,
                "runnable_targets": list(self.runnable_targets),
                "blocked_targets": list(self.blocked_targets),
                "missing_python_packages": list(self.missing_python_packages),
                "missing_external_groups": list(self.missing_external_groups),
            },
            "target_statuses": [
                {
                    "target": item.target,
                    "status": item.status,
                    "missing_required": list(item.missing_required),
                    "missing_optional": list(item.missing_optional),
                }
                for item in self.target_statuses
            ],
            "dependencies": [
                {
                    "key": item.key,
                    "kind": item.kind,
                    "label": item.label,
                    "status": item.status,
                    "required": item.required,
                    "targets": list(item.targets),
                    "details": item.details,
                    "install_hint": item.install_hint,
                    "install_group": item.install_group,
                }
                for item in self.statuses
            ],
            "tracking_backend_blockers": self.tracking_backend_blockers,
        }


def _clean_targets(targets: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(target.strip().lower() for target in targets if target and target.strip()))


def _resolve_path(path: Optional[Path]) -> Optional[Path]:
    return path.expanduser().resolve() if path is not None else None


def _resolve_interpreter(path: Optional[Path]) -> Optional[Path]:
    return interpreter_path(path) if path is not None else None


def _merge_spec(existing: DependencySpec, incoming: DependencySpec) -> DependencySpec:
    targets = tuple(dict.fromkeys((*existing.targets, *incoming.targets)))
    return DependencySpec(
        key=existing.key,
        kind=existing.kind,
        label=existing.label,
        required=existing.required or incoming.required,
        targets=targets,
        python_import=existing.python_import or incoming.python_import,
        python_package=existing.python_package or incoming.python_package,
        python_install_requirement=existing.python_install_requirement or incoming.python_install_requirement,
        python_executable=existing.python_executable or incoming.python_executable,
        python_path_entries=tuple(dict.fromkeys((*existing.python_path_entries, *incoming.python_path_entries))),
        device=existing.device or incoming.device,
        command=existing.command or incoming.command,
        bin_dir=existing.bin_dir or incoming.bin_dir,
        env_vars=tuple(dict.fromkeys((*existing.env_vars, *incoming.env_vars))),
        path=existing.path or incoming.path,
        path_kind=existing.path_kind,
        discovery_source=existing.discovery_source or incoming.discovery_source,
        common_paths=tuple(dict.fromkeys((*existing.common_paths, *incoming.common_paths))),
        install_group=existing.install_group or incoming.install_group,
        notes=existing.notes or incoming.notes,
    )


def _put(specs: dict[str, DependencySpec], spec: DependencySpec) -> None:
    existing = specs.get(spec.key)
    specs[spec.key] = _merge_spec(existing, spec) if existing else spec


def _pythonslicer_next_to(slicer_executable: Optional[Path]) -> Optional[Path]:
    """PythonSlicer of a Slicer launcher, resolved as tractography/wma.py does; None if absent."""
    if slicer_executable is None or sys.platform == "win32":
        return None
    try:
        launcher_dir = slicer_executable.expanduser().resolve().parent
    except OSError:
        return None
    for root in slicer_runtime_roots(launcher_dir):
        for candidate in (root / "PythonSlicer", root / "bin" / "PythonSlicer"):
            if candidate.is_file():
                return candidate
    return None


def _python(
    specs: dict[str, DependencySpec],
    *,
    target: str,
    import_name: str,
    package: str,
    install_requirement: Optional[str] = None,
    required: bool = True,
    python_executable: Optional[Path] = None,
    python_path_entries: tuple[Path, ...] = (),
    install_group: str = "python",
    notes: str = "",
    key: Optional[str] = None,
    label: Optional[str] = None,
) -> None:
    _put(
        specs,
        DependencySpec(
            key=key or f"python:{import_name}",
            kind="python",
            label=label or import_name,
            required=required,
            targets=(target,),
            python_import=import_name,
            python_package=package,
            python_install_requirement=install_requirement or package,
            python_executable=_resolve_interpreter(python_executable),
            python_path_entries=tuple(path.expanduser().resolve() for path in python_path_entries),
            discovery_source="python",
            install_group=install_group,
            notes=notes,
        ),
    )


def _command(
    specs: dict[str, DependencySpec],
    *,
    target: str,
    command: str,
    group: str,
    bin_dir: Optional[Path] = None,
    required: bool = True,
    notes: str = "",
) -> None:
    _put(
        specs,
        DependencySpec(
            key=f"command:{command}",
            kind="command",
            label=command,
            required=required,
            targets=(target,),
            command=command,
            bin_dir=_resolve_path(bin_dir),
            discovery_source="user setting" if bin_dir is not None else "",
            install_group=group,
            notes=notes,
        ),
    )


def _device(
    specs: dict[str, DependencySpec],
    *,
    target: str,
    device: str,
    python_executable: Optional[Path] = None,
    required: bool = True,
) -> None:
    normalized = (device or "cpu").strip().lower()
    if normalized in {"", "cpu"}:
        return
    _put(
        specs,
        DependencySpec(
            key=f"device:{target}:{normalized}",
            kind="device",
            label=f"{target} requested device {normalized}",
            required=required,
            targets=(target,),
            python_executable=_resolve_interpreter(python_executable),
            device=normalized,
            discovery_source="device",
            install_group="device",
            notes="Requested CUDA device must be visible to the selected Python torch runtime.",
        ),
    )


def _env(
    specs: dict[str, DependencySpec],
    *,
    target: str,
    key: str,
    label: str,
    env_vars: tuple[str, ...],
    group: str,
    required: bool = True,
    notes: str = "",
    resolved_path: Optional[Path] = None,
    path_kind: str = "dir",
    discovery_source: str = "",
) -> None:
    _put(
        specs,
        DependencySpec(
            key=f"env:{key}",
            kind="env",
            label=label,
            required=required,
            targets=(target,),
            env_vars=env_vars,
            path=resolved_path,
            path_kind=path_kind,
            discovery_source=discovery_source or "environment",
            install_group=group,
            notes=notes,
        ),
    )


def _setting(
    specs: dict[str, DependencySpec],
    *,
    target: str,
    key: str,
    label: str,
    group: str,
    required: bool = True,
    notes: str = "",
) -> None:
    _put(
        specs,
        DependencySpec(
            key=f"setting:{key}",
            kind="setting",
            label=label,
            required=required,
            targets=(target,),
            discovery_source="setting",
            install_group=group,
            notes=notes,
        ),
    )


def _path(
    specs: dict[str, DependencySpec],
    *,
    target: str,
    key: str,
    label: str,
    path: Path,
    path_kind: str,
    group: str,
    required: bool = True,
    notes: str = "",
    discovery_source: str = "user setting",
    common_paths: tuple[Path, ...] = (),
) -> None:
    _put(
        specs,
        DependencySpec(
            key=f"path:{key}",
            kind="path",
            label=label,
            required=required,
            targets=(target,),
            path=path.expanduser().resolve(),
            path_kind=path_kind,
            discovery_source=discovery_source,
            common_paths=tuple(candidate.expanduser() for candidate in common_paths),
            install_group=group,
            notes=notes,
        ),
    )


def _add_core_python(specs: dict[str, DependencySpec], target: str) -> None:
    for import_name, package in (
        ("numpy", "numpy"),
        ("nibabel", "nibabel>=5.0"),
    ):
        _python(specs, target=target, import_name=import_name, package=package)


def _add_dipy_stack(specs: dict[str, DependencySpec], target: str) -> None:
    _add_core_python(specs, target)
    # Slicer 5.6 embeds Python 3.9.  DIPY 1.11+ requires Python 3.10,
    # while the APIs used by this extension are available in DIPY 1.10.
    _python(specs, target=target, import_name="dipy", package="dipy>=1.10,<1.13")


def _add_slicer_python_runtime(specs: dict[str, DependencySpec], target: str) -> None:
    _python(
        specs,
        target=target,
        import_name="slicer.util",
        package="",
        install_requirement="",
        install_group="slicer-runtime",
        notes=(
            "Run this target from 3D Slicer or with Slicer's bin/PythonSlicer executable. "
            "Do not install the unrelated PyPI package named 'slicer'."
        ),
    )


def _add_deep_learning_stack(specs: dict[str, DependencySpec], target: str) -> None:
    for import_name, package in (
        ("torch", "torch>=2.0.0"),
        ("torchvision", "torchvision"),
    ):
        _python(
            specs,
            target=target,
            import_name=import_name,
            package=package,
            notes="Inside Slicer, prefer installing torch via the Slicer PyTorch extension.",
        )


def _add_mrtrix_commands(
    specs: dict[str, DependencySpec],
    target: str,
    commands: tuple[str, ...],
    bin_dir: Optional[Path],
) -> None:
    for name in commands:
        _command(specs, target=target, command=name, group="mrtrix", bin_dir=bin_dir)


def _env_path(*names: str) -> Optional[Path]:
    for name in names:
        value = os.environ.get(name)
        if value:
            return Path(value).expanduser().resolve()
    return None


def _env_path_with_source(*names: str, resolve: bool = True) -> tuple[Optional[Path], str]:
    for name in names:
        value = os.environ.get(name)
        if value:
            return (Path(value).expanduser().resolve() if resolve else interpreter_path(value)), f"env:{name}"
    return None, ""


def _env_command_dirs(group: str) -> tuple[tuple[Path, str], ...]:
    candidates: list[tuple[Path, str]] = []

    def add_dir(env_name: str, *, child: Optional[str] = None) -> None:
        value = os.environ.get(env_name)
        if not value:
            return
        path = Path(value).expanduser().resolve()
        candidates.append(((path / child).resolve() if child else path, f"env:{env_name}"))

    if group == "mrtrix":
        for env_name in ("SLICERDSPACE_MRTRIX_BIN_DIR", "MRTRIX_BIN_DIR"):
            add_dir(env_name)
        for env_name in ("SLICERDSPACE_MRTRIX_HOME", "MRTRIX_HOME", "MRTRIX3_HOME"):
            add_dir(env_name, child="bin")
    elif group == "fsl":
        for env_name in ("SLICERDSPACE_FSL_BIN_DIR", "FSL_BIN_DIR"):
            add_dir(env_name)
        add_dir("FSLDIR", child="bin")
    elif group == "freesurfer":
        for env_name in ("SLICERDSPACE_FREESURFER_HOME", "FREESURFER_HOME"):
            add_dir(env_name, child="bin")
    elif group == "wma":
        for env_name in ("SLICERDSPACE_WMA_ROOT", "WMA_ROOT"):
            add_dir(env_name, child="bin")
            add_dir(env_name, child="utilities")
    return tuple(candidates)


def _current_user_names() -> tuple[str, ...]:
    names: list[str] = []
    for value in (os.environ.get("USER"), os.environ.get("USERNAME"), Path.home().name):
        if value:
            names.append(value)
    return tuple(dict.fromkeys(names))


def _local_software_roots() -> tuple[Path, ...]:
    roots: list[Path] = [
        Path.home(),
        Path.home() / "software",
        Path.home() / "Software",
    ]
    for user_name in _current_user_names():
        roots.extend(
            [
                Path("/data") / user_name,
                Path("/data") / user_name / "software",
                Path("/data") / user_name / "Software",
            ]
        )
    for shared_root in (Path("/data"), Path("/Users"), Path("C:/Users")):
        try:
            if not shared_root.exists() or not shared_root.is_dir():
                continue
            for child in shared_root.iterdir():
                roots.extend(
                    [
                        child,
                        child / "software",
                        child / "Software",
                    ]
                )
        except Exception:
            continue
    return tuple(dict.fromkeys(root.expanduser() for root in roots))


def _common_freesurfer_homes(platform_name: Optional[str] = None) -> tuple[Path, ...]:
    candidates: list[Path] = [
        Path("/usr/local/freesurfer"),
        Path("/opt/freesurfer"),
        Path("/Applications/freesurfer"),
    ]
    for root in _local_software_roots():
        candidates.extend(
            [
                root / "freesurfer",
                root / "FreeSurfer",
            ]
        )
    if platform_name == "Darwin":
        candidates.extend(
            [
                Path("/Applications/freesurfer"),
                Path("/Applications/FreeSurfer"),
                Path.home() / "Applications" / "freesurfer",
                Path.home() / "Applications" / "FreeSurfer",
            ]
        )
    if platform_name == "Windows":
        candidates.extend(
            [
                Path("C:/freesurfer"),
                Path("C:/FreeSurfer"),
                Path("C:/Program Files/FreeSurfer"),
                Path("C:/Program Files (x86)/FreeSurfer"),
            ]
        )
    # Deduplicate by the rendered candidate, not host ``Path`` equality.
    # WindowsPath compares case-insensitively, which otherwise drops the
    # distinct macOS ``freesurfer``/``FreeSurfer`` locations when a Darwin
    # environment is inspected or tested from Windows.
    unique: dict[str, Path] = {}
    for candidate in candidates:
        expanded = candidate.expanduser()
        unique.setdefault(expanded.as_posix(), expanded)
    return tuple(unique.values())


def _is_freesurfer_home(path: Path) -> bool:
    return (
        path.is_dir()
        and any((path / "bin" / name).is_file() for name in ("recon-all", "mri_surf2surf"))
        and (path / "subjects" / "fsaverage").is_dir()
    )


def _resolve_freesurfer_home_with_source(platform_name: Optional[str] = None) -> tuple[Optional[Path], str, list[str]]:
    checked: list[str] = []
    for env_name in ("SLICERDSPACE_FREESURFER_HOME", "FREESURFER_HOME"):
        value = os.environ.get(env_name)
        if not value:
            continue
        candidate = Path(value).expanduser().resolve()
        checked.append(candidate.as_posix())
        if _is_freesurfer_home(candidate):
            return candidate, f"env:{env_name}", checked

    for executable in ("recon-all", "mri_surf2surf"):
        found = shutil.which(executable)
        if found:
            candidate = Path(found).expanduser().resolve().parent.parent
            checked.append(candidate.as_posix())
            if _is_freesurfer_home(candidate):
                return candidate, "PATH", checked

    for candidate in _common_freesurfer_homes(platform_name):
        checked.append(candidate.as_posix())
        if _is_freesurfer_home(candidate):
            return candidate.resolve(), "common path", checked

    return None, "not found", checked


def resolve_freesurfer_home() -> Optional[Path]:
    """Return an auto-discovered FreeSurfer home, if one is available."""

    resolved, _, _ = _resolve_freesurfer_home_with_source(platform.system() or "unknown")
    return resolved


def _common_command_dirs(group: str, platform_name: str) -> tuple[Path, ...]:
    candidates: list[Path] = []
    if group == "mrtrix":
        candidates.extend(
            [
                Path("/usr/local/mrtrix3/bin"),
                Path("/usr/local/MRtrix3/bin"),
                Path("/opt/mrtrix3/bin"),
                Path("/opt/MRtrix3/bin"),
                Path("/usr/lib/mrtrix3/bin"),
                Path("/usr/libexec/mrtrix3/bin"),
                Path("/usr/local/bin"),
                Path("/opt/homebrew/bin"),
            ]
        )
        for root in _local_software_roots():
            candidates.extend(
                [
                    root / "mrtrix3" / "bin",
                    root / "MRtrix3" / "bin",
                    root / "mrtrix" / "bin",
                    root / "mrtrix3_env" / "bin",
                ]
            )
        if platform_name == "Darwin":
            candidates.extend(
                [
                    Path("/Applications/MRtrix3.app/Contents/bin"),
                    Path.home() / "Applications" / "MRtrix3.app" / "Contents" / "bin",
                ]
            )
        if platform_name == "Windows":
            candidates.extend(
                [
                    Path("C:/Program Files/MRtrix3/bin"),
                    Path("C:/Program Files (x86)/MRtrix3/bin"),
                ]
            )
    elif group == "fsl":
        candidates.extend(
            [
                Path("/usr/local/fsl/bin"),
                Path("/opt/fsl/bin"),
                Path("/usr/local/fsl/share/fsl/bin"),
                Path("/opt/fsl/share/fsl/bin"),
                Path("/usr/share/fsl/6.0/bin"),
            ]
        )
        for root in _local_software_roots():
            candidates.extend(
                [
                    root / "fsl" / "bin",
                    root / "FSL" / "bin",
                    root / "fsl" / "share" / "fsl" / "bin",
                ]
            )
        if platform_name == "Windows":
            candidates.extend(
                [
                    Path("C:/fsl/bin"),
                    Path("C:/Program Files/FSL/bin"),
                    Path("C:/Program Files (x86)/FSL/bin"),
                ]
            )
        if platform_name == "Darwin":
            candidates.extend(
                [
                    Path("/Applications/fsl/bin"),
                    Path.home() / "Applications" / "fsl" / "bin",
                ]
            )
    elif group == "freesurfer":
        candidates.extend(home / "bin" for home in _common_freesurfer_homes(platform_name))
    elif group == "wma":
        candidates.extend(
            [
                Path("/usr/local/whitematteranalysis/bin"),
                Path("/opt/whitematteranalysis/bin"),
                Path("/usr/local/WhiteMatterAnalysis/bin"),
                Path("/opt/WhiteMatterAnalysis/bin"),
            ]
        )
        for root in _local_software_roots():
            candidates.extend(
                [
                    root / "whitematteranalysis" / "bin",
                    root / "WhiteMatterAnalysis" / "bin",
                    root / "whitematteranalysis" / "utilities",
                    root / "WhiteMatterAnalysis" / "utilities",
                ]
            )
    if platform_name == "Darwin":
        candidates.extend([Path("/opt/homebrew/bin"), Path("/usr/local/bin")])
    return tuple(dict.fromkeys(path for path in candidates if path))


def _common_checkout_roots(name: str, platform_name: str) -> tuple[Path, ...]:
    candidates: list[Path] = []
    work_roots = [
        Path.home(),
        Path.home() / "src",
        Path.home() / "workspace",
        Path.home() / "code",
        Path.home() / "projects",
    ]
    for root in _local_software_roots():
        work_roots.extend([root, root / "src", root / "workspace", root / "code", root / "projects"])
    if platform_name == "Darwin":
        work_roots.extend([Path("/Applications"), Path.home() / "Applications"])
    if platform_name == "Windows":
        work_roots.extend(
            [
                Path("C:/Program Files"),
                Path("C:/Program Files (x86)"),
                Path.home() / "AppData" / "Local",
            ]
        )
    unique_roots = tuple(dict.fromkeys(path.expanduser() for path in work_roots))

    if name == "tract_querier":
        suffixes = ("tract_querier", "TractQuerier")
    elif name == "wma":
        suffixes = ("whitematteranalysis", "WhiteMatterAnalysis")
    elif name == "tractcloud":
        suffixes = ("TractCloud", "tractcloud")
    else:
        suffixes = ()
    for root in unique_roots:
        for suffix in suffixes:
            candidates.append(root / suffix)
    return tuple(dict.fromkeys(candidates))


def _repo_checkout_roots(repo_root: Path, name: str) -> tuple[Path, ...]:
    resolved_root = repo_root.expanduser().resolve()
    if name == "tract_querier":
        suffixes = (
            ("third_party", "tract_querier"),
            ("third_party", "TractQuerier"),
        )
    elif name == "wma":
        suffixes = (
            ("third_party", "whitematteranalysis"),
            ("third_party", "WhiteMatterAnalysis"),
        )
    elif name == "tractcloud":
        suffixes = (
            ("third_party", "TractCloud"),
            ("third_party", "tractcloud"),
        )
    else:
        suffixes = ()

    candidates: list[Path] = []
    from shared.backend_sources import available_source
    profile = {'tract_querier': 'wmql', 'wma': 'wma', 'tractcloud': 'tractcloud'}.get(name)
    owned = available_source(profile) if profile and not (name == 'wma' and platform.system() == 'Windows') else None
    if owned is not None:
        candidates.append(owned)
    # ``Path.parents`` only gained slice support in Python 3.10.  Slicer 5.6
    # embeds Python 3.9, so materialize the sequence before taking the nearby
    # checkout roots.
    search_roots = [resolved_root, *list(resolved_root.parents)[:4]]
    for base in search_roots:
        for suffix in suffixes:
            candidates.append(base.joinpath(*suffix))
    return tuple(dict.fromkeys(candidates))


def _common_wma_atlas_dirs(platform_name: str) -> tuple[Path, ...]:
    candidates: list[Path] = []
    for root in _common_checkout_roots("wma", platform_name):
        candidates.extend(
            [
                root / "ORG-Atlases",
                root / "ORG_atlas",
                root / "atlas",
                root / "data",
            ]
        )
    for root in _local_software_roots():
        candidates.extend(
            [
                root / "ORG-Atlases",
                root / "ORG_atlas",
                root / "wma_atlas",
                root / "WMA_atlas",
            ]
        )
    return tuple(dict.fromkeys(candidates))


def _common_slicer_executables(platform_name: str) -> tuple[Path, ...]:
    candidates: list[Path] = [
        Path("/data/chengjin/software/Slicer-5.10.0-linux-amd64/Slicer"),
        Path("/usr/local/Slicer/Slicer"),
        Path("/opt/Slicer/Slicer"),
    ]
    for root in _local_software_roots():
        candidates.extend(
            [
                root / "Slicer" / "Slicer",
                root / "Slicer-5.10.0-linux-amd64" / "Slicer",
                root / "Slicer-5.8.1-linux-amd64" / "Slicer",
            ]
        )
    if platform_name == "Darwin":
        candidates.extend(
            [
                Path("/Applications/Slicer.app/Contents/MacOS/Slicer"),
                Path.home() / "Applications" / "Slicer.app" / "Contents" / "MacOS" / "Slicer",
            ]
        )
    if platform_name == "Windows":
        candidates.extend(
            [
                Path("C:/Program Files/Slicer 5.10.0/Slicer.exe"),
                Path("C:/Program Files/Slicer/Slicer.exe"),
                Path("C:/Program Files (x86)/Slicer/Slicer.exe"),
            ]
        )
    return tuple(dict.fromkeys(candidates))


def _first_existing_path(candidates: Iterable[Path], *, kind: str) -> tuple[Optional[Path], tuple[str, ...]]:
    checked: list[str] = []
    for candidate in candidates:
        resolved = candidate.expanduser()
        checked.append(resolved.as_posix())
        try:
            if kind == "file" and resolved.is_file():
                return resolved.resolve(), tuple(checked)
            if kind == "dir" and resolved.is_dir():
                return resolved.resolve(), tuple(checked)
            if kind == "nonempty_dir" and resolved.is_dir() and any(resolved.iterdir()):
                return resolved.resolve(), tuple(checked)
        except OSError:
            continue
    return None, tuple(checked)


def _command_path_candidates(directory: Path, command: str, platform_name: str) -> tuple[Path, ...]:
    candidates = [directory / command]
    if platform_name == "Windows" and not Path(command).suffix:
        candidates.extend(directory / f"{command}{suffix}" for suffix in (".exe", ".cmd", ".bat"))
    return tuple(dict.fromkeys(candidates))


def _resolve_command_with_source(spec: DependencySpec, platform_name: str) -> tuple[Optional[str], str, list[str]]:
    assert spec.command
    checked: list[str] = []

    if spec.bin_dir is not None:
        for user_candidate in _command_path_candidates(spec.bin_dir, spec.command, platform_name):
            checked.append(user_candidate.as_posix())
            try:
                if user_candidate.is_file():
                    return user_candidate.as_posix(), "user setting", checked
            except OSError:
                pass

    for directory, source in _env_command_dirs(spec.install_group):
        for candidate in _command_path_candidates(directory, spec.command, platform_name):
            checked.append(candidate.as_posix())
            try:
                if candidate.is_file():
                    return candidate.as_posix(), source, checked
            except OSError:
                continue

    found = shutil.which(spec.command)
    if found is not None:
        return found, "PATH", checked

    for directory in (*spec.common_paths, *_common_command_dirs(spec.install_group, platform_name)):
        for candidate in _command_path_candidates(directory, spec.command, platform_name):
            checked.append(candidate.as_posix())
            try:
                if candidate.is_file():
                    return candidate.as_posix(), "common path", checked
            except OSError:
                continue

    return None, "not found", checked


def _tractcloud_pythonpath(root: Path) -> Path:
    src_dir = root / "src"
    return src_dir if (src_dir / "tractcloud").is_dir() else root


def _python_install_requirement_for_import(import_name: str) -> str:
    if import_name == "tract_querier":
        return "git+https://github.com/demianw/tract_querier.git"
    return import_name


def _resource_root(config: EnvironmentDoctorConfig) -> Path:
    return config.repo_root.expanduser().resolve()


def _model_asset_spec_path(config: EnvironmentDoctorConfig, relative: str) -> tuple[Path, tuple[Path, ...], str]:
    """Resolve one model asset for a dependency spec, without letting a
    tampered/corrupt file abort the whole environment-doctor scan.

    A ``ModelAssetIntegrityError`` here must become one "missing" dependency
    item with an actionable note, not an unhandled exception that prevents
    every other target's dependency check from running too. The returned
    placeholder path is guaranteed not to exist, so the resulting
    DependencySpec correctly reports "not found" rather than accidentally
    resolving to an unrelated real path. On failure, ``common_paths`` is
    also returned empty: ``_check_spec()``'s own common-path fallback
    accepts the first *existing* candidate with no sha256 check of its own,
    so leaving the (still unverified) candidate list in place would let
    that fallback silently re-accept the very file that just failed
    verification.
    """

    repo_root = _resource_root(config)
    explicit_root = _resolve_path(config.model_assets_dir)
    candidates = model_asset_candidates(relative, repo_root=repo_root, explicit_root=explicit_root)
    resolved, error = model_asset_path_or_none(relative, repo_root=repo_root, explicit_root=explicit_root)
    if error is not None:
        placeholder = repo_root / (relative + ".integrity-check-failed")
        notes = (
            f"Integrity check failed: {error} Delete the file and re-download it with "
            "scripts/download_model_assets.py (or the Slicer UI's model-asset download button)."
        )
        return placeholder, (), notes
    return resolved, candidates, ""


def _effective_platform_name(config: EnvironmentDoctorConfig) -> str:
    return (config.platform_name or platform.system() or "unknown").strip() or "unknown"


def _apply_default_python(
    specs: Iterable[DependencySpec],
    python_executable: Optional[Path],
    python_path_entries: Iterable[Path] = (),
) -> tuple[DependencySpec, ...]:
    default_python = _resolve_interpreter(python_executable)
    default_python_paths = tuple(
        dict.fromkeys(path.expanduser().resolve() for path in python_path_entries if path)
    )
    if default_python is None and not default_python_paths:
        return tuple(specs)
    updated: list[DependencySpec] = []
    for spec in specs:
        if spec.kind in {"python", "device"}:
            spec_python = _resolve_interpreter(spec.python_executable)
            uses_default_python = spec_python is None or spec_python == default_python
            inherited_python_paths = default_python_paths if uses_default_python else ()
            updated.append(
                DependencySpec(
                    key=spec.key,
                    kind=spec.kind,
                    label=spec.label,
                    required=spec.required,
                    targets=spec.targets,
                    python_import=spec.python_import,
                    python_package=spec.python_package,
                    python_install_requirement=spec.python_install_requirement,
                    python_executable=spec_python or default_python,
                    python_path_entries=tuple(
                        dict.fromkeys((*inherited_python_paths, *spec.python_path_entries))
                    ),
                    device=spec.device,
                    command=spec.command,
                    bin_dir=spec.bin_dir,
                    env_vars=spec.env_vars,
                    path=spec.path,
                    path_kind=spec.path_kind,
                    discovery_source=spec.discovery_source,
                    common_paths=spec.common_paths,
                    install_group=spec.install_group,
                    notes=spec.notes,
                )
            )
        else:
            updated.append(spec)
    return tuple(updated)


def resolve_config_python_executable(config: EnvironmentDoctorConfig) -> Path:
    from shared.python_environment import resolve_python_executable

    return resolve_python_executable(
        python_executable=config.python_executable,
        python_environment=config.python_environment,
        conda_executable=config.conda_executable,
    )


def build_dependency_specs(config: EnvironmentDoctorConfig) -> tuple[DependencySpec, ...]:
    from shared.algorithm_selection import algorithms
    tracking = algorithms(config.tractography_backend)
    if len(tracking) > 1:
        combined = {}
        for backend in tracking:
            for spec in build_dependency_specs(replace(config, tractography_backend=backend)):
                _put(combined, spec)
        return tuple(combined.values())
    targets = _clean_targets(config.targets)
    specs: dict[str, DependencySpec] = {}
    repo_root = _resource_root(config)
    platform_name = _effective_platform_name(config)

    for target in targets:
        if target == "validation":
            _add_core_python(specs, target)
            _python(specs, target=target, import_name="SimpleITK", package="SimpleITK")

        if target == "preprocessing":
            # W2a (D1): one route. DIPY MP-PCA and Gibbs removal, then FSL eddy/topup through dwifslpreproc.
            # So the target needs MRtrix3 and FSL on every OS (the setup plan reads these install groups).
            _add_dipy_stack(specs, target)
            _add_mrtrix_commands(
                specs,
                target,
                ("dwifslpreproc", "dwi2mask"),
                config.preprocessing_mrtrix_bin_dir or config.mrtrix_bin_dir,
            )
            for command in ("eddy", "topup"):
                _command(
                    specs,
                    target=target,
                    command=command,
                    group="fsl",
                    bin_dir=config.preprocessing_fsl_bin_dir,
                )

        if target in {"microstructure", "microstructure_freewater", "microstructure_dki", "tractography"} and (
            target in {"microstructure", "microstructure_freewater", "microstructure_dki"} or config.tractography_backend == "dti_det"
        ):
            _add_dipy_stack(specs, target)
            if target != "microstructure_dki":
                _add_slicer_python_runtime(specs, target)

        # Native DTI scalar export converts Teem NRRD outputs to NIfTI with pynrrd.
        # Include it in preparation, even when no anatomical model was selected.
        if target in {"microstructure", "microstructure_freewater", "microstructure_dki", "microstructure_noddi"} or (
                target == "tractography" and config.tractography_backend == "dti_det"):
            _python(specs, target=target, import_name="nrrd", package="pynrrd")

        if target == "microstructure_noddi":
            _add_core_python(specs, target)
            _python(specs, target=target, import_name="amico", package="dmri-amico")
            _add_slicer_python_runtime(specs, target)

        if target == "segmentation":
            _add_core_python(specs, target)
            _add_slicer_python_runtime(specs, target)
            for import_name, package in (("scipy", "scipy"), ("nrrd", "pynrrd")):
                _python(specs, target=target, import_name=import_name, package=package)
            _add_deep_learning_stack(specs, target)
            _device(specs, target=target, device=config.ddseg_device, python_executable=config.python_executable)
            ddseg_weights, ddseg_candidates, ddseg_notes = _model_asset_spec_path(config, "segmentation/weights")
            _path(
                specs,
                target=target,
                key="ddseg_weights",
                label="DDSeg weights directory",
                path=ddseg_weights,
                path_kind="nonempty_dir",
                group="model-resources",
                notes=ddseg_notes,
                discovery_source="model asset root",
                common_paths=ddseg_candidates,
            )

        if target == "parcellation":
            _add_core_python(specs, target)
            _add_slicer_python_runtime(specs, target)
            for import_name, package in (
                ("SimpleITK", "SimpleITK"),
                ("scipy", "scipy"),
                ("nrrd", "pynrrd"),
                ("skimage", "scikit-image"),
                ("h5py", "h5py"),
                ("plumbum", "plumbum"),
            ):
                _python(specs, target=target, import_name=import_name, package=package)
            _add_deep_learning_stack(specs, target)
            ddparcel_weights, ddparcel_candidates, ddparcel_notes = _model_asset_spec_path(config, "parcellation/weights")
            _path(
                specs,
                target=target,
                key="ddparcel_weights",
                label="DDParcel weights directory",
                path=ddparcel_weights,
                path_kind="nonempty_dir",
                group="model-resources",
                notes=ddparcel_notes,
                discovery_source="model asset root",
                common_paths=ddparcel_candidates,
            )

        if target == "surface":
            _add_core_python(specs, target)
            _add_slicer_python_runtime(specs, target)
            for import_name, package in (
                ("SimpleITK", "SimpleITK"),
                ("scipy", "scipy"),
                ("trimesh", "trimesh"),
                ("rtree", "rtree"),
            ):
                _python(specs, target=target, import_name=import_name, package=package)
            _add_deep_learning_stack(specs, target)
            _device(specs, target=target, device=config.surface_device, python_executable=config.python_executable)
            template_path, template_candidates, template_notes = _model_asset_spec_path(
                config, "surface/template/ddsurfer_lh_template.obj"
            )
            atlas_path, atlas_candidates, atlas_notes = _model_asset_spec_path(
                config, "surface/template/100HCP-population-mean-T2-1mm.nii.gz"
            )
            checkpoint_path, checkpoint_candidates, checkpoint_notes = _model_asset_spec_path(
                config, "surface/weights/manifest.json"
            )
            _path(
                specs,
                target=target,
                key="ddsurfer_template",
                label="DDSurfer transformed left template",
                path=template_path,
                path_kind="file",
                group="model-resources",
                notes=template_notes,
                discovery_source="model asset root",
                common_paths=template_candidates,
            )
            _path(
                specs,
                target=target,
                key="ddsurfer_atlas_t2",
                label="DDSurfer registration atlas",
                path=atlas_path,
                path_kind="file",
                group="model-resources",
                notes=atlas_notes,
                discovery_source="model asset root",
                common_paths=atlas_candidates,
            )
            _path(
                specs,
                target=target,
                key="ddsurfer_checkpoints",
                label="DDSurfer model release manifest",
                path=checkpoint_path,
                path_kind="file",
                group="model-resources",
                notes=checkpoint_notes,
                discovery_source="model asset root",
                common_paths=checkpoint_candidates,
            )
            for relative in (
                'surface/template/ddsurfer_rh_template.obj', 'surface/weights/SHA256SUMS',
                'surface/weights/ddsurfer_lh_wm.pt', 'surface/weights/ddsurfer_rh_wm.pt',
                'surface/weights/ddsurfer_lh_pial.pt', 'surface/weights/ddsurfer_rh_pial.pt',
            ):
                path, candidates, notes = _model_asset_spec_path(config, relative)
                _path(specs, target=target, key='ddsurfer_' + Path(relative).name.replace('.', '_'),
                      label='DDSurfer ' + Path(relative).name, path=path, path_kind='file',
                      group='model-resources', notes=notes, discovery_source='model asset root',
                      common_paths=candidates)

        if target == "surface_postprocess":
            _add_core_python(specs, target)
            _python(specs, target=target, import_name="SimpleITK", package="SimpleITK")
            _python(specs, target=target, import_name="trimesh", package="trimesh")
            _python(specs, target=target, import_name="rtree", package="rtree")
            from shared.surface_runtime import resolve_surface_runtime
            try:
                surface_backend, freesurfer_home = resolve_surface_runtime(config.postprocess_runtime, config.postprocess_freesurfer_home)
            except (OSError, ValueError):
                surface_backend, freesurfer_home = 'native', None
            if surface_backend == 'freesurfer':
                _env(
                    specs,
                    target=target,
                    key="freesurfer_home",
                    label="FREESURFER_HOME",
                    env_vars=("SLICERDSPACE_FREESURFER_HOME", "FREESURFER_HOME"),
                    group="freesurfer",
                    required=config.postprocess_freesurfer_home is None,
                    notes="Can also be supplied with --postprocess-freesurfer-home.",
                    resolved_path=freesurfer_home,
                    discovery_source=("user setting" if config.postprocess_freesurfer_home is not None
                                      else _resolve_freesurfer_home_with_source(platform_name)[1]),
                )
                for command in (
                    "mris_sphere",
                    "mris_curvature",
                    "mris_inflate",
                    "mris_register",
                    "mri_label2label",
                    "mri_surf2surf",
                ):
                    _command(
                        specs,
                        target=target,
                        command=command,
                        group="freesurfer",
                        bin_dir=freesurfer_home / "bin" if config.postprocess_freesurfer_home is not None else None,
                    )
                metric_home = freesurfer_home or _resolve_freesurfer_home_with_source(platform_name)[0]
                modern = ((metric_home / "bin" / "mris_place_surface") if metric_home
                          else Path(shutil.which("mris_place_surface") or "/nonexistent/mris_place_surface"))
                _command(specs, target=target,
                         command="mris_place_surface" if modern.is_file() and os.access(modern, os.X_OK) else "mris_thickness",
                         group="freesurfer", bin_dir=freesurfer_home / "bin" if config.postprocess_freesurfer_home is not None else None,
                         notes="Current surface-only backend: modern metrics or the explicit older-FreeSurfer thickness fallback.")

        if target == "tractography" and config.tractography_backend in {"csd_act", "saf"}:
            _add_core_python(specs, target)
            _add_mrtrix_commands(
                specs,
                target,
                ("mrconvert", "dwi2response", "dwi2fod", "mtnormalise", "tckgen", "tcksift2"),
                config.mrtrix_bin_dir,
            )

        if target == 'tractography' and config.tractography_backend == 'ukf':
            from tractography.ukf import resolve_executable
            _add_core_python(specs, target)
            for name, package in (('nrrd', 'pynrrd'), ('vtk', 'vtk')):
                _python(specs, target=target, import_name=name, package=package)
            for name, group in (('UKFTractography', 'ukf'), ('DWIConvert', 'slicer-dmri')):
                try:
                    binary = resolve_executable(
                        config.ukf_executable if name == 'UKFTractography' else config.dwiconvert_executable,
                        name=name)
                except (OSError, RuntimeError):
                    binary = None
                _command(specs, target=target, command=name, group=group,
                         bin_dir=binary.parent if binary else None,
                         notes='Optional native Slicer extension; no vendored UKF algorithm or MRtrix dependency.')

        if target == "bundles":
            _add_core_python(specs, target)
            tract_querier_root_source = "user setting"
            tract_querier_root = _resolve_path(config.tract_querier_root)
            if tract_querier_root is None:
                tract_querier_root, tract_querier_root_source = _env_path_with_source(
                    "SLICERDSPACE_TRACT_QUERIER_ROOT",
                    "TRACT_QUERIER_ROOT",
                )
            if tract_querier_root is None:
                tract_querier_root, _ = _first_existing_path(
                    (*_repo_checkout_roots(config.repo_root, "tract_querier"), *_common_checkout_roots("tract_querier", platform_name)),
                    kind="dir",
                )
                if tract_querier_root is not None:
                    tract_querier_root_source = "common path"
            if tract_querier_root is not None:
                _path(
                    specs,
                    target=target,
                    key="tract_querier_root",
                    label="Tract Querier checkout",
                    path=tract_querier_root,
                    path_kind="dir",
                    group="wmql",
                    discovery_source=tract_querier_root_source,
                )
                _path(
                    specs,
                    target=target,
                    key="tract_querier_root_script",
                    label="Tract Querier root CLI script",
                    path=tract_querier_root / "scripts" / "cli_tract_querier.py",
                    path_kind="file",
                    group="wmql",
                    discovery_source=tract_querier_root_source,
                )
                _python(
                    specs,
                    target=target,
                    import_name="tract_querier",
                    package="tract_querier",
                    install_requirement=_python_install_requirement_for_import("tract_querier"),
                    python_path_entries=(tract_querier_root,),
                    notes="Checked by adding the configured Tract Querier checkout to PYTHONPATH.",
                )
            else:
                _python(
                    specs,
                    target=target,
                    import_name="tract_querier",
                    package="tract_querier",
                    install_requirement=_python_install_requirement_for_import("tract_querier"),
                )

        if target == "connectome":
            _add_core_python(specs, target)
            # auto needs MRtrix3 for a .tck source (csd_act, saf); python never does.
            if config.connectome_backend == "mrtrix" or (
                    config.connectome_backend == "auto" and config.tractography_backend in ("csd_act", "saf")):
                _add_mrtrix_commands(specs, target, ("tck2connectome",), config.mrtrix_bin_dir)

        if target == "saf":
            _add_core_python(specs, target)
            for import_name, package in (("scipy", "scipy"), ("sklearn", "scikit-learn")):
                _python(specs, target=target, import_name=import_name, package=package)
            _add_mrtrix_commands(specs, target, ("tckgen", "tckstats", "tckconvert"), config.mrtrix_bin_dir)

        if target == "wma":
            _add_core_python(specs, target)
            if platform_name == "Linux" and not any(os.environ.get(name) for name in (
                    "DISPLAY", "SLICERDSPACE_DISPLAY")):
                from tractography.wma import EXTRA_TOOL_DIRS
                for command in ("xvfb-run", "Xvfb", "xauth"):
                    _put(specs, DependencySpec(
                        key=f"command:{command}", kind="command", label=command,
                        required=True, targets=(target,), command=command, install_group="wma",
                        common_paths=EXTRA_TOOL_DIRS,
                        notes="WMA automatically uses a virtual X display for Slicer on headless Linux."))
            _python(specs, target=target, import_name="vtk", package="vtk",
                    notes="Required by the pipeline host to decode bundle geometry; bundled with Slicer.")
            # The author scripts run in Slicer's Python and `import whitematteranalysis` loads these at import time
            # (whitematteranalysis/__init__.py, tract_measurement.py; its requirements.txt).
            slicer_executable_source = "user setting"
            slicer_executable = _resolve_path(config.slicer_executable)
            if slicer_executable is None:
                slicer_executable, slicer_executable_source = _env_path_with_source(
                    "SLICERDSPACE_SLICER_EXECUTABLE",
                )
            if slicer_executable is None:
                slicer_executable, _ = _first_existing_path(_common_slicer_executables(platform_name), kind="file")
                if slicer_executable is not None:
                    slicer_executable_source = "common path"
            # C3: the author scripts run with the PythonSlicer of the configured Slicer (tractography/wma.py),
            # so probe that interpreter; fall back to the doctor's Python when it cannot be resolved.
            wma_python = _pythonslicer_next_to(slicer_executable)
            for import_name, package in (("joblib", "joblib"), ("scipy", "scipy"), ("xlrd", "xlrd"),
                                         ("matplotlib", "matplotlib"), ("pandas", "pandas"),
                                         ("PIL", "Pillow"), ("statsmodels", "statsmodels")):
                # N1: own key, so the host interpreter's check of the same module is not merged into this one.
                _python(specs, target=target, import_name=import_name, package=package,
                        python_executable=wma_python, key=f"python:wma:{import_name}",
                        label=f"{import_name} (WMA)",
                        notes="Imported by the WhiteMatterAnalysis package in Slicer's Python.")
            _command(
                specs,
                target=target,
                command="tckconvert",
                group="mrtrix",
                bin_dir=config.mrtrix_bin_dir,
                required=False,
                notes="Required only when the WMA input tractogram is a TCK file.",
            )
            wma_root_source = "user setting"
            wma_root = _resolve_path(config.wma_root)
            if wma_root is None:
                wma_root, wma_root_source = _env_path_with_source("SLICERDSPACE_WMA_ROOT", "WMA_ROOT")
            if wma_root is None:
                wma_root, _ = _first_existing_path(
                    (*_repo_checkout_roots(config.repo_root, "wma"), *_common_checkout_roots("wma", platform_name)),
                    kind="dir",
                )
                if wma_root is not None:
                    wma_root_source = "common path"
            _env(
                specs,
                target=target,
                key="wma_root",
                label="WhiteMatterAnalysis checkout",
                env_vars=("SLICERDSPACE_WMA_ROOT", "WMA_ROOT"),
                group="wma",
                required=False,
                notes="Can also be supplied with --wma-root or by putting wm_apply_ORG_atlas_to_subject.sh on PATH.",
                resolved_path=wma_root,
                discovery_source=wma_root_source if wma_root is not None else "",
            )
            if wma_root is not None:
                _path(
                    specs,
                    target=target,
                    key="wma_root_path",
                    label="WhiteMatterAnalysis checkout path",
                    path=wma_root,
                    path_kind="dir",
                    group="wma",
                    discovery_source=wma_root_source,
                )
            _command(
                specs,
                target=target,
                command="wm_apply_ORG_atlas_to_subject.sh",
                group="wma",
                bin_dir=wma_root / "bin" if config.wma_root is not None and wma_root is not None else None,
            )
            if platform_name == "Windows" and config.wma_atlas != "org" and wma_root is not None:
                _path(specs, target=target, key="wma_atlas_python_bridge",
                      label="WMA atlas Python bridge (repair Windows runtime if missing)",
                      path=wma_root / "bin/python.exe", path_kind="file", group="wma",
                      discovery_source="managed Windows runtime")
            wma_atlas_dir_source = "user setting"
            wma_atlas_dir = _resolve_path(config.wma_atlas_dir)
            if wma_atlas_dir is None:
                atlas_env = ("SLICERDSPACE_WMA_ATLAS_" + config.wma_atlas.upper().replace("-", "_") + "_DIR",)
                if config.wma_atlas == "org":
                    atlas_env += ("SLICERDSPACE_WMA_ATLAS_DIR", "WMA_ATLAS_DIR")
                wma_atlas_dir, wma_atlas_dir_source = _env_path_with_source(
                    *atlas_env,
                )
            if wma_atlas_dir is None:
                from shared.optional_assets import resolve_wma_atlas_directory
                wma_atlas_dir = resolve_wma_atlas_directory(asset_root=config.model_assets_dir, atlas_family=config.wma_atlas)
                wma_atlas_dir_source = "verified shared resource directory"
            if wma_atlas_dir is None and config.wma_atlas == "org":
                wma_atlas_dir, _ = _first_existing_path(_common_wma_atlas_dirs(platform_name), kind="dir")
                if wma_atlas_dir is not None:
                    wma_atlas_dir_source = "common path"
            if wma_atlas_dir is not None:
                _path(
                    specs,
                    target=target,
                    key="wma_org_atlas",
                    label="WMA " + config.wma_atlas + " atlas directory",
                    path=wma_atlas_dir,
                    path_kind="dir",
                    group="wma",
                    discovery_source=wma_atlas_dir_source,
                )
                from tractography.wma_atlases import atlas_definition
                for child in atlas_definition(config.wma_atlas)["directories"]:
                    _path(
                        specs,
                        target=target,
                        key=f"wma_org_atlas_{child}",
                        label=f"WMA atlas subdirectory {child}",
                        path=wma_atlas_dir / child,
                        path_kind="dir",
                        group="wma",
                        discovery_source=wma_atlas_dir_source,
                    )
            else:
                _setting(
                    specs,
                    target=target,
                    key="wma_org_atlas",
                    label="WMA " + config.wma_atlas + " atlas directory",
                    group="wma",
                    notes="Selected atlas: " + config.wma_atlas + ". Prepare its resource profile or pass --wma-atlas-dir.",
                )
            _env(
                specs,
                target=target,
                key="slicer_executable",
                label="Slicer executable for WMA",
                env_vars=("SLICERDSPACE_SLICER_EXECUTABLE",),
                group="slicer",
                required=config.slicer_executable is None,
                notes="Can also be supplied with --slicer-executable.",
                resolved_path=slicer_executable,
                path_kind="file",
                discovery_source=slicer_executable_source if slicer_executable is not None else "",
            )
            if slicer_executable is not None:
                _path(
                    specs,
                    target=target,
                    key="slicer_executable_path",
                    label="Slicer executable path",
                    path=slicer_executable,
                    path_kind="file",
                    group="slicer",
                    discovery_source=slicer_executable_source,
                )
            fiber_cli_source = "user setting"
            fiber_cli = _resolve_path(config.fiber_tract_measurements_cli)
            if fiber_cli is None:
                fiber_cli, fiber_cli_source = _env_path_with_source(
                    "SLICERDSPACE_FIBER_TRACT_MEASUREMENTS_CLI",
                )
            if fiber_cli is not None:
                _path(
                    specs,
                    target=target,
                    key="fiber_tract_measurements_cli",
                    label="SlicerDMRI FiberTractMeasurements CLI",
                    path=fiber_cli,
                    path_kind="file",
                    group="slicer-dmri",
                    required=False,
                    discovery_source=fiber_cli_source,
                )

        if target == "tractcloud":
            _add_core_python(specs, target)
            _python(specs, target=target, import_name="vtk", package="vtk",
                    notes="Required by the pipeline host to decode bundle geometry; bundled with Slicer.")
            tractcloud_root_source = "user setting"
            tractcloud_root = _resolve_path(config.tractcloud_root)
            if tractcloud_root is None:
                tractcloud_root, tractcloud_root_source = _env_path_with_source(
                    "SLICERDSPACE_TRACTCLOUD_ROOT",
                    "TRACTCLOUD_ROOT",
                )
            if tractcloud_root is None:
                tractcloud_root, _ = _first_existing_path(
                    (*_repo_checkout_roots(config.repo_root, "tractcloud"), *_common_checkout_roots("tractcloud", platform_name)),
                    kind="dir",
                )
                if tractcloud_root is not None:
                    tractcloud_root_source = "common path"
            tractcloud_python_source = "user setting"
            tractcloud_python = _resolve_interpreter(config.tractcloud_python_executable)
            if tractcloud_python is None:
                tractcloud_python, tractcloud_python_source = _env_path_with_source(
                    "SLICERDSPACE_TRACTCLOUD_PYTHON_EXECUTABLE",
                    "TRACTCLOUD_PYTHON_EXECUTABLE",
                    resolve=False,
                )
            if tractcloud_python is not None:
                _path(
                    specs,
                    target=target,
                    key="tractcloud_python",
                    label="TractCloud Python executable",
                    path=tractcloud_python,
                    path_kind="file",
                    group="tractcloud",
                    discovery_source=tractcloud_python_source,
                )
            _env(
                specs,
                target=target,
                key="tractcloud_root",
                label="TractCloud checkout/package",
                env_vars=("SLICERDSPACE_TRACTCLOUD_ROOT", "TRACTCLOUD_ROOT"),
                group="tractcloud",
                required=False,
                notes="Can also be supplied with --tractcloud-root.",
                resolved_path=tractcloud_root,
                discovery_source=tractcloud_root_source if tractcloud_root is not None else "",
            )
            if tractcloud_root is not None:
                _path(
                    specs,
                    target=target,
                    key="tractcloud_root_path",
                    label="TractCloud checkout path",
                    path=tractcloud_root,
                    path_kind="dir",
                    group="tractcloud",
                    discovery_source=tractcloud_root_source,
                )
                python_path_entries = (_tractcloud_pythonpath(tractcloud_root),)
            else:
                python_path_entries = ()
            for import_name, package in (('numpy', 'numpy>=1.21'), ('vtk', 'vtk>=9')):
                _python(specs, target=target, import_name=import_name, package=package,
                        python_executable=tractcloud_python, key='python:tractcloud:' + import_name,
                        label=import_name + ' (TractCloud)')
            _python(
                specs,
                target=target,
                import_name="tractcloud",
                package="tractcloud",
                python_executable=tractcloud_python,
                python_path_entries=python_path_entries,
            )
            _python(
                specs,
                target=target,
                import_name="torch",
                package="torch>=2.3.1",
                python_executable=tractcloud_python,
                key='python:tractcloud:torch',
                notes="Inside Slicer, prefer installing torch via the Slicer PyTorch extension.",
            )
            _device(
                specs,
                target=target,
                device=config.tractcloud_device,
                python_executable=tractcloud_python or config.python_executable,
            )
            tractcloud_data_dir_source = "user setting"
            tractcloud_data_dir = _resolve_path(config.tractcloud_data_dir)
            if tractcloud_data_dir is None:
                tractcloud_data_dir, tractcloud_data_dir_source = _env_path_with_source(
                    "SLICERDSPACE_TRACTCLOUD_DATA_DIR",
                    "TRACTCLOUD_DATA_DIR",
                )
            if tractcloud_data_dir is None:
                from shared.optional_assets import verified_resource_directory
                tractcloud_data_dir = verified_resource_directory("tractcloud", asset_root=config.model_assets_dir)
                tractcloud_data_dir_source = "verified shared resource directory"
            if tractcloud_data_dir is not None:
                _path(
                    specs,
                    target=target,
                    key="tractcloud_data_dir",
                    label="TractCloud model/data directory",
                    path=tractcloud_data_dir,
                    path_kind="dir",
                    group="tractcloud",
                    discovery_source=tractcloud_data_dir_source,
                )
                for relative in ("TrainedModel/best_tract_f1_model.pth", "TrainedModel/cli_args.txt",
                                 "TrainData_800clu800ol/HCP_mass_center.npy"):
                    _path(specs, target=target, key="tractcloud_data:" + relative,
                          label="TractCloud " + relative, path=tractcloud_data_dir / relative,
                          path_kind="file", group="tractcloud", discovery_source=tractcloud_data_dir_source)
            else:
                _setting(specs, target=target, key="tractcloud_data_dir", label="TractCloud inference resources",
                         group="tractcloud", notes="Prepare the selected environment to download verified inference resources, or configure an existing TractCloud data directory.")

    return tuple(
        sorted(
            _apply_default_python(
                specs.values(),
                resolve_config_python_executable(config),
                config.python_path_entries,
            ),
            key=lambda item: item.key,
        )
    )


def _install_hint(spec: DependencySpec, platform_name: str) -> str:
    if spec.kind == "python" and spec.python_package:
        python_bin = spec.python_executable.as_posix() if spec.python_executable else "python"
        requirement = spec.python_install_requirement or spec.python_package
        return f"Install in the checked Python/Slicer Python: {python_bin} -m pip install '{requirement}'"
    if spec.kind == "setting":
        return spec.notes or "Provide the required CLI/GUI setting before running this target."
    if spec.install_group == "mrtrix":
        if platform_name == "Darwin":
            return "Install MRtrix3, for example: brew install mrtrix3, or use a conda environment."
        if platform_name == "Windows":
            return "Use WSL2/Linux or a conda/container backend for MRtrix3; native Windows support is not recommended."
        return "Install MRtrix3 and add its bin directory to PATH, or pass --mrtrix-bin-dir."
    if spec.install_group == "fsl":
        return "Install FSL/eddy/topup via the official FSL installer, then add $FSLDIR/bin to PATH or pass --preprocessing-fsl-bin-dir."
    if spec.install_group == "freesurfer":
        return "Install FreeSurfer, set FREESURFER_HOME, source SetUpFreeSurfer.sh, and provide a FreeSurfer license."
    if spec.install_group == "wma":
        return "Install/configure external WhiteMatterAnalysis and ORG atlas; pass --wma-root/--wma-atlas-dir or set SLICERDSPACE_WMA_ROOT."
    if spec.install_group == "tractcloud":
        return "Install/configure external TractCloud package/check-out and model data; pass --tractcloud-root/--tractcloud-data-dir."
    if spec.install_group == "model-resources":
        return "Choose a model asset root in the Slicer UI, or run scripts/download_model_assets.py --profile core --asset-root <path>."
    if spec.install_group == "slicer":
        return "Pass --slicer-executable or set SLICERDSPACE_SLICER_EXECUTABLE to the local Slicer executable."
    if spec.install_group == "slicer-dmri":
        return "Install SlicerDMRI in Slicer and point to the required CLI if this optional export is enabled."
    if spec.install_group == 'ukf':
        return 'Install UKFTractography through Slicer Extension Manager and restart Slicer. UKF is optional; no automatic algorithm fallback.'
    if spec.install_group == "device":
        return "Use CPU, or install a CUDA-capable PyTorch runtime and NVIDIA driver visible to the selected Python/Slicer environment."
    return spec.notes


def _parse_json_probe_output(stdout: str) -> dict[str, Any]:
    """Parse a JSON probe result even when a launcher prints startup notices."""

    stripped = stdout.strip()
    if not stripped:
        return {}
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    for line in reversed(stripped.splitlines()):
        try:
            payload = json.loads(line.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError(f"Python dependency probe did not emit a JSON object: {stripped!r}")


def _python_requirement_details(package: str, installed_version: Optional[str]) -> tuple[Optional[str], Optional[bool]]:
    """Return a package specifier and whether the installed release satisfies it."""

    if not package:
        return None, None
    try:
        from packaging.requirements import Requirement

        requirement = Requirement(package)
    except (ModuleNotFoundError, ValueError):
        return None, None
    specifier = str(requirement.specifier) or None
    if specifier is None or installed_version is None:
        return specifier, None
    return specifier, requirement.specifier.contains(installed_version, prereleases=True)


def _python_distribution_name(package: str) -> Optional[str]:
    if not package:
        return None
    try:
        from packaging.requirements import Requirement

        return Requirement(package).name
    except (ModuleNotFoundError, ValueError):
        return None


def _check_cuda_device(
    device: str,
    python_executable: Optional[Path],
    python_path_entries: Iterable[Path] = (),
) -> tuple[bool, dict[str, Any]]:
    python_path = python_executable or Path(sys.executable)
    resolved_python_paths = tuple(path.expanduser().resolve() for path in python_path_entries if path)
    probe = (
        "import json, sys; "
        f"sys.path[:0] = {[path.as_posix() for path in resolved_python_paths]!r}; "
        "payload = {'torch_importable': False, 'cuda_available': False, 'device_count': 0, 'torch_version': None, 'error': None}; "
        "\ntry:\n"
        "    import torch\n"
        "    payload['torch_importable'] = True\n"
        "    payload['torch_version'] = getattr(torch, '__version__', None)\n"
        "    payload['cuda_available'] = bool(torch.cuda.is_available())\n"
        "    payload['device_count'] = int(torch.cuda.device_count()) if payload['cuda_available'] else 0\n"
        "except Exception as exc:\n"
        "    payload['error'] = str(exc)\n"
        "print(json.dumps(payload))"
    )
    try:
        completed = subprocess.run(
            [python_path.as_posix(), "-c", probe],
            check=True,
            capture_output=True,
            timeout=DEPENDENCY_PROBE_TIMEOUT_SECONDS,
            env=python_tool_environment(python_path),
            **text_subprocess_kwargs(),
        )
        details = _parse_json_probe_output(completed.stdout)
    except subprocess.TimeoutExpired as exc:
        details = {
            "torch_importable": False,
            "cuda_available": False,
            "device_count": 0,
            "torch_version": None,
            "error": _probe_timeout_error(exc),
        }
    except Exception as exc:
        details = {
            "torch_importable": False,
            "cuda_available": False,
            "device_count": 0,
            "torch_version": None,
            "error": str(exc),
        }

    requested_index: Optional[int] = None
    if ":" in device:
        try:
            requested_index = int(device.split(":", 1)[1])
        except ValueError:
            requested_index = None
    available = bool(details.get("cuda_available"))
    if requested_index is not None:
        available = available and int(details.get("device_count") or 0) > requested_index
    details["requested_device"] = device
    details["requested_index"] = requested_index
    details["python_executable"] = python_path.as_posix()
    details["pythonpath"] = [path.as_posix() for path in resolved_python_paths]
    return available, details


def resolve_auto_torch_device(
    device: str = "auto",
    python_executable: Optional[Path] = None,
    python_path_entries: Iterable[Path] = (),
) -> str:
    """Resolve ``auto`` to one concrete torch device without running both CPU and GPU paths."""

    normalized = (device or "auto").strip().lower()
    if normalized in {"", "auto"}:
        available, details = _check_cuda_device("cuda:0", python_executable, python_path_entries)
        if details.get("error"):
            raise RuntimeError("Cannot determine PyTorch GPU availability in the selected runtime: " + str(details["error"]) +
                               ". Fix that runtime or explicitly choose CPU; no silent CPU fallback.")
        return "cuda:0" if available and int(details.get("device_count") or 0) > 0 else "cpu"
    if normalized == "cuda":
        return "cuda:0"
    return normalized


def _check_spec(spec: DependencySpec, platform_name: str) -> DependencyStatus:
    details: dict[str, Any] = {}
    available = False

    if spec.kind == "python":
        assert spec.python_import
        python_executable = spec.python_executable
        python_path_entries = tuple(path for path in spec.python_path_entries if path)
        distribution_name = _python_distribution_name(spec.python_package or "")
        installed_version: Optional[str] = None
        requirement_specifier, _ = _python_requirement_details(spec.python_package or "", None)
        requirement_satisfied: Optional[bool] = None
        if python_executable is None and not python_path_entries:
            try:
                available = importlib.util.find_spec(spec.python_import) is not None
                error = None
                if available and distribution_name and requirement_specifier:
                    installed_version = importlib.metadata.version(distribution_name)
            except importlib.metadata.PackageNotFoundError as exc:
                available = False
                error = str(exc)
            except (ImportError, AttributeError, ValueError) as exc:
                available = False
                error = str(exc)
            requirement_specifier, requirement_satisfied = _python_requirement_details(
                spec.python_package or "",
                installed_version,
            )
            if available and requirement_satisfied is False:
                available = False
                error = (
                    f"installed {distribution_name} {installed_version} does not satisfy "
                    f"{requirement_specifier}"
                )
            details = {
                "import": spec.python_import,
                "package": spec.python_package,
                "install_requirement": spec.python_install_requirement or spec.python_package,
                "python_executable": sys.executable,
                "pythonpath": [],
                "detected_source": "current Python" if available else "not found",
                "installed_version": installed_version,
                "version_specifier": requirement_specifier,
                "version_satisfied": requirement_satisfied,
                "error": error,
            }
        else:
            python_path = python_executable or Path(sys.executable)
            env = python_tool_environment(python_path)
            if python_path_entries:
                env["PYTHONPATH"] = os.pathsep.join(path.as_posix() for path in python_path_entries) + os.pathsep + env.get("PYTHONPATH", "")
            probe = (
                "import importlib.metadata, importlib.util, json, sys\n"
                f"sys.path[:0] = {[path.as_posix() for path in python_path_entries]!r}\n"
                f"available = importlib.util.find_spec({spec.python_import!r}) is not None\n"
                f"distribution = {(distribution_name if requirement_specifier else None)!r}\n"
                "version = importlib.metadata.version(distribution) if available and distribution else None\n"
                "print(json.dumps({'available': available, 'version': version}))\n"
            )
            try:
                completed = subprocess.run(
                    [python_path.as_posix(), "-c", probe],
                    check=True,
                    capture_output=True,
                    env=env,
                    timeout=DEPENDENCY_PROBE_TIMEOUT_SECONDS,
                    **text_subprocess_kwargs(),
                )
                probe_details = _parse_json_probe_output(completed.stdout)
                available = bool(probe_details.get("available"))
                installed_version = probe_details.get("version")
                error = None
            except subprocess.TimeoutExpired as exc:
                available = False
                error = _probe_timeout_error(exc)
            except Exception as exc:
                available = False
                error = str(exc)
            requirement_specifier, requirement_satisfied = _python_requirement_details(
                spec.python_package or "",
                installed_version,
            )
            if available and requirement_satisfied is False:
                available = False
                error = (
                    f"installed {distribution_name} {installed_version} does not satisfy "
                    f"{requirement_specifier}"
                )
            details = {
                "import": spec.python_import,
                "package": spec.python_package,
                "install_requirement": spec.python_install_requirement or spec.python_package,
                "python_executable": python_path.as_posix(),
                "pythonpath": [path.as_posix() for path in python_path_entries],
                "detected_source": "configured Python/PYTHONPATH" if available else "not found",
                "installed_version": installed_version,
                "version_specifier": requirement_specifier,
                "version_satisfied": requirement_satisfied,
                "error": error,
            }
    elif spec.kind == "command":
        assert spec.command
        found, detected_source, checked_locations = _resolve_command_with_source(spec, platform_name)
        available = found is not None
        details = {
            "command": spec.command,
            "resolved": found,
            "bin_dir": spec.bin_dir.as_posix() if spec.bin_dir else None,
            "detected_source": detected_source,
            "checked_locations": checked_locations,
        }
    elif spec.kind == "env":
        values = {name: os.environ.get(name) for name in spec.env_vars}
        if spec.path is not None:
            if spec.path_kind == "file":
                available = spec.path.is_file()
            elif spec.path_kind == "nonempty_dir":
                available = spec.path.is_dir() and any(spec.path.iterdir())
            else:
                available = spec.path.is_dir()
            details = {
                "env": values,
                "resolved": spec.path.as_posix(),
                "notes": spec.notes,
                "detected_source": spec.discovery_source if available else "not found",
                "checked_locations": [spec.path.as_posix()],
            }
        elif spec.key == "env:freesurfer_home":
            resolved, detected_source, checked_locations = _resolve_freesurfer_home_with_source(platform_name)
            available = resolved is not None
            details = {
                "env": values,
                "resolved": resolved.as_posix() if resolved else None,
                "notes": spec.notes,
                "detected_source": detected_source,
                "checked_locations": checked_locations,
            }
        else:
            available = any(value for value in values.values())
            detected_source = next((f"env:{name}" for name, value in values.items() if value), "not found")
            details = {"env": values, "notes": spec.notes, "detected_source": detected_source}
    elif spec.kind == "device":
        requested_device = spec.device or "cpu"
        if requested_device in {"cpu", ""}:
            available = True
            details = {
                "requested_device": requested_device,
                "python_executable": (spec.python_executable or Path(sys.executable)).as_posix(),
                "detected_source": "configured device",
            }
        elif requested_device == "auto":
            try:
                resolved = resolve_auto_torch_device(requested_device, spec.python_executable, spec.python_path_entries)
                available = True
                details = {"requested_device": requested_device, "resolved_device": resolved,
                           "python_executable": (spec.python_executable or Path(sys.executable)).as_posix(),
                           "detected_source": "auto"}
            except RuntimeError as exc:
                available = False
                details = {"requested_device": requested_device, "resolved_device": "unavailable",
                           "python_executable": (spec.python_executable or Path(sys.executable)).as_posix(),
                           "detected_source": "probe failed", "error": str(exc)}
        elif requested_device.startswith("cuda"):
            available, details = _check_cuda_device(
                requested_device,
                spec.python_executable,
                spec.python_path_entries,
            )
            details["detected_source"] = "CUDA runtime" if available else "not found"
        else:
            available = False
            details = {
                "requested_device": requested_device,
                "python_executable": (spec.python_executable or Path(sys.executable)).as_posix(),
                "detected_source": "not found",
                "error": "Only cpu, auto, cuda, and cuda:N are supported by the environment doctor.",
            }
    elif spec.kind == "path":
        assert spec.path
        checked_locations = [spec.path.as_posix()]
        resolved_path = spec.path
        if spec.path_kind == "dir":
            available = spec.path.is_dir()
        elif spec.path_kind == "nonempty_dir":
            available = spec.path.is_dir() and any(spec.path.iterdir())
        else:
            available = spec.path.is_file()
        detected_source = spec.discovery_source if available else "not found"
        if not available and spec.common_paths:
            fallback, extra_checked = _first_existing_path(spec.common_paths, kind=spec.path_kind)
            checked_locations.extend(location for location in extra_checked if location not in checked_locations)
            if fallback is not None:
                resolved_path = fallback
                available = True
                detected_source = "common path"
        details = {
            "path": resolved_path.as_posix(),
            "path_kind": spec.path_kind,
            "notes": spec.notes,
            "detected_source": detected_source,
            "checked_locations": checked_locations,
        }
    elif spec.kind == "setting":
        available = False
        details = {"notes": spec.notes, "detected_source": "not configured"}
    else:
        details = {"notes": spec.notes, "detected_source": spec.discovery_source or "unknown"}

    return DependencyStatus(
        key=spec.key,
        kind=spec.kind,
        label=spec.label,
        status="available" if available else "missing",
        required=spec.required,
        targets=spec.targets,
        details=details,
        install_hint="" if available else _install_hint(spec, platform_name),
        install_group=spec.install_group,
    )


def _teem_info_in_process() -> dict[str, str]:
    """Load Slicer's Teem library in this interpreter; return version, library path and error text."""
    try:
        from shared import teem_dti

        library = teem_dti.load_teem()
        return {"teem_version": teem_dti.teem_version(library),
                "library": str(getattr(library, "_slicerdspace_path", "")), "error": ""}
    except Exception as exc:  # TeemError, ImportError (no numpy), OSError: all mean "not available here"
        return {"teem_version": "", "library": "", "error": str(exc)}


def _teem_info_in_interpreter(python_path: Path) -> dict[str, str]:
    """Run the Teem check in the selected interpreter (C2), so it answers for that runtime."""
    probe = (
        "import json, sys\n"
        f"sys.path.insert(0, {Path(__file__).resolve().parents[1].as_posix()!r})\n"
        "try:\n"
        "    from shared import teem_dti\n"
        "    lib = teem_dti.load_teem()\n"
        "    result = dict(teem_version=teem_dti.teem_version(lib), library=str(getattr(lib, '_slicerdspace_path', '')), error='')\n"
        "except Exception as exc:\n"
        "    result = dict(teem_version='', library='', error=str(exc))\n"
        "print(json.dumps(result))\n"
    )
    try:
        completed = subprocess.run(
            [python_path.as_posix(), "-c", probe],
            check=True, capture_output=True, env=python_tool_environment(python_path),
            timeout=DEPENDENCY_PROBE_TIMEOUT_SECONDS, **text_subprocess_kwargs(),
        )
        payload = _parse_json_probe_output(completed.stdout)
    except subprocess.TimeoutExpired as exc:
        return {"teem_version": "", "library": "", "error": _probe_timeout_error(exc)}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"teem_version": "", "library": "", "error": str(exc)}
    if not isinstance(payload, dict):
        return {"teem_version": "", "library": "", "error": "probe returned no result"}
    return {name: str(payload.get(name, "")) for name in ("teem_version", "library", "error")}


def _doctor_pythonslicer(config: "EnvironmentDoctorConfig") -> Optional[Path]:
    """PythonSlicer that the native tensor fit will run in (D1), found from the settings, not from the doctor's Python.

    Order: the call-site resolver ``resolve_pythonslicer_executable()`` that the fit uses, the configured Python when
    it is itself a PythonSlicer, the configured Slicer launcher, SLICERDSPACE_SLICER_EXECUTABLE,
    SLICER_HOME / SLICERDSPACE_SLICER_HOME, then the common install paths.
    """
    # T2/R3: ask the resolver that the fit uses first. Outside Slicer it also reads SLICER_HOME, SLICERDSPACE_SLICER_HOME,
    # SLICERDSPACE_SLICER_EXECUTABLE and the configured Slicer path; it raises when none gives a PythonSlicer, and the
    # search below adds the doctor's own settings (configured Python and launcher, common install paths).
    try:
        from shared.pipeline import resolve_pythonslicer_executable
        return interpreter_path(resolve_pythonslicer_executable())
    except Exception:
        pass
    configured = config.python_executable
    if configured is not None and Path(configured).name in {"PythonSlicer", "PythonSlicer.exe"}:
        return interpreter_path(configured)
    launcher = _resolve_path(config.slicer_executable)
    if launcher is None:
        launcher, _ = _env_path_with_source("SLICERDSPACE_SLICER_EXECUTABLE")
    homes: list[Path] = [launcher.parent] if launcher is not None else []
    homes += [Path(os.environ[key]) for key in ("SLICER_HOME", "SLICERDSPACE_SLICER_HOME") if os.environ.get(key)]
    if launcher is None:
        common, _ = _first_existing_path(_common_slicer_executables(_effective_platform_name(config)), kind="file")
        if common is not None:
            homes.append(common.parent)
    for home in homes:
        found = find_pythonslicer(home)
        if found is not None:
            return interpreter_path(found)
    return None


def _native_tensor_status(targets: tuple[str, ...], config: Optional["EnvironmentDoctorConfig"] = None, *,
                          in_process: bool = False) -> DependencyStatus:
    """The one requirement of the native tensor fit: Teem loads in the PythonSlicer that runs the fit (D1, D6).

    A Slicer process probes itself. Any other process probes the PythonSlicer found from the settings and never
    loads Teem in its own interpreter, so a plain Python with SLICER_HOME set cannot report the fit as available.
    """
    # T1: only Slicer's own interpreter name takes the in-process branch, never a path marker such as envs/slicer-x/.
    if in_process or Path(sys.executable).name.lower() in SLICER_PYTHON_NAMES:
        probed: Optional[Path] = interpreter_path(sys.executable)
        info = _teem_info_in_process()
    else:
        probed = _doctor_pythonslicer(config) if config is not None else None
        if probed is None:
            info = {"teem_version": "", "library": "", "error":
                    "PythonSlicer was not found. The native tensor fit runs in Slicer's PythonSlicer; "
                    "set the Slicer executable (or SLICER_HOME) so the check can find it."}
        else:
            info = _teem_info_in_interpreter(probed)
    found = not info["error"] and bool(info["library"])
    slicer_probed = probed is not None
    return DependencyStatus(
        key="native_tensor_fit", kind="setting", label="Native tensor fit (Slicer core Teem)",
        status="available" if found else "missing", required=True, targets=targets,
        details={**info, "check": "Teem library load in Slicer's PythonSlicer", "notes": info["error"],
                 "python_executable": probed.as_posix() if probed is not None else ""},
        install_group="slicer-runtime",
        # N1: the reason (searched roots, file names) reaches the panel and the install notes.
        install_hint="" if found else (info["error"] if slicer_probed else "Run with Slicer's PythonSlicer"),
    )


def run_environment_doctor(config: EnvironmentDoctorConfig, *, _probe_cache=None) -> EnvironmentDoctorResult:
    platform_name = _effective_platform_name(config)
    host_platform = platform.system() or "unknown"
    targets = _clean_targets(config.targets)
    checked_python_path = resolve_config_python_executable(config)
    checked_python = checked_python_path.as_posix()
    specs = build_dependency_specs(config)
    if _probe_cache is None:
        _probe_cache = {}

    def check(spec):
        # Targets and requiredness affect reporting, not the actual probe.
        key = replace(spec, targets=(), required=False)
        if key not in _probe_cache:
            _probe_cache[key] = _check_spec(spec, platform_name)
        checked = _probe_cache[key]
        if checked.targets == spec.targets and checked.required == spec.required:
            return checked
        return replace(checked, targets=spec.targets, required=spec.required)

    statuses = tuple(check(spec) for spec in specs)
    if 'surface_postprocess' in targets:
        from shared.surface_runtime import inspect_surface_runtime
        surface = inspect_surface_runtime(config.postprocess_runtime, config.postprocess_freesurfer_home,
                                          config.postprocess_license_file, python_bin=checked_python_path)
        statuses += (
            DependencyStatus(key='surface_postprocess_runtime', kind='setting', label='Surface postprocessing components',
                status='available' if surface['components_available'] else 'missing', required=True,
                targets=('surface_postprocess',), details=surface, install_group='surface-runtime',
                install_hint='Use Environment > Prepare for qualified precompiled components, or configure an existing runtime. If its archive is not yet published, import the verified offline bundle. Source compilation is an expert CLI opt-in; Check never installs.'),
            DependencyStatus(key='surface_postprocess_license', kind='setting', label='Personal FreeSurfer activation',
                status='available' if surface['license_available'] else 'missing', required=True,
                targets=('surface_postprocess',), details={'notes': 'Set the activation file or FS_LICENSE; it is not downloaded or included in the runtime.'},
                install_group='freesurfer-license', install_hint='Register for your own FreeSurfer activation file and select it in Environment paths, or set FS_LICENSE.'),
        )
    if config.loaded_slicer_modules is not None or config.check_slicer_cli_modules:
        native_targets = tuple(dict.fromkeys(target for spec in specs if spec.key == 'python:slicer.util'
                                             for target in spec.targets))
        if native_targets:
            # GUI session: the interpreter is Slicer's own, so the check runs in process.
            statuses += (_native_tensor_status(
                native_targets, config, in_process=config.loaded_slicer_modules is not None),)
    if "microstructure_dki" in targets and config.dki_fit_method.upper().startswith("MKCURVE"):
        matlab_reference = config.dki_fit_method.upper() == "MKCURVE_MATLAB"
        if matlab_reference:
            from microstructure.mkcurve_matlab import inspect_mkcurve_backend
        else:
            from microstructure.mkcurve import inspect_mkcurve_backend
        backend = inspect_mkcurve_backend(config.dki_matlab_executable, config.dki_mkcurve_root)
        statuses += (DependencyStatus(
            key="dki_mkcurve_backend", kind="setting", label="MK-Curve MATLAB reference" if matlab_reference else "MK-Curve Python port",
            status="available" if backend["available"] else "missing", required=True,
            targets=("microstructure_dki",), details=backend, install_group="mkcurve",
            install_hint=("Configure Advanced MATLAB and author MK-Curve source paths; no fallback is used."
                          if matlab_reference else "Bundled Python port uses NumPy/SciPy/nibabel; verify runtime resource installation."),
        ),)
    if config.brain_mask_required:
        from shared.brain_mask import brain_mask_config_from_options, inspect_brain_mask_backend
        masking = inspect_brain_mask_backend(brain_mask_config_from_options(config), system=platform_name)
        mask_targets = tuple(target for target in targets if target in {
            "validation", "preprocessing", "segmentation", "parcellation", "surface",
            "microstructure", "microstructure_freewater", "microstructure_noddi", "microstructure_dki", "tractography"})
        if mask_targets:
            statuses += (DependencyStatus(
                key="brain_mask_backend", kind="setting", label="Selected brain-mask backend",
                status="available" if masking["available"] and config.brain_mask_backend != "provided" else "missing",
                required=True, targets=mask_targets, details=masking,
                install_hint="Use Environment and installation to prepare PNL CNN, or provide a native mask (select Provided mask only in Advanced settings > Preprocessing & mask).",
                install_group="brain_mask"),)
    status_by_key = {item.key: item for item in statuses}

    target_statuses: list[TargetEnvironmentStatus] = []
    for target in targets:
        related = [item for item in statuses if target in item.targets]
        missing_required = tuple(item.key for item in related if item.required and item.status == "missing")
        missing_optional = tuple(item.key for item in related if not item.required and item.status == "missing")
        target_statuses.append(
            TargetEnvironmentStatus(
                target=target,
                status="blocked" if missing_required else "runnable",
                missing_required=missing_required,
                missing_optional=missing_optional,
            )
        )

    # B10: a passing Tract Querier checkout is installed from that checkout, so its pip line is not listed.
    wmql_root = status_by_key.get("path:tract_querier_root")
    wmql_checkout_passes = wmql_root is not None and wmql_root.status == "available"
    missing_python_packages = tuple(
        dict.fromkeys(
            item.details.get("install_requirement") or item.details.get("package")
            for item in statuses
            if item.kind == "python"
            and item.status == "missing"
            and not (wmql_checkout_passes and item.key == "python:tract_querier")
            and (item.details.get("install_requirement") or item.details.get("package"))
        )
    )
    missing_external_groups = tuple(
        sorted(
            {
                item.install_group
                for item in statuses
                if item.status == "missing" and item.install_group
                and (item.kind != "python" or not item.details.get("package"))
            }
        )
    )
    blocker_count = sum(1 for item in statuses if item.required and item.status == "missing")

    from shared.algorithm_selection import algorithms
    tracking = algorithms(config.tractography_backend)
    backend_blockers = {}
    if len(tracking) > 1 and 'tractography' in targets:
        for backend in tracking:
            variant = run_environment_doctor(replace(config, tractography_backend=backend), _probe_cache=_probe_cache)
            backend_blockers[backend] = blocked_target_reasons(variant)

    return EnvironmentDoctorResult(
        platform=platform_name,
        host_platform=host_platform,
        python_executable=checked_python,
        targets=targets,
        statuses=statuses,
        target_statuses=tuple(target_statuses),
        blocker_count=blocker_count,
        missing_python_packages=missing_python_packages,
        missing_external_groups=missing_external_groups,
        tracking_backend_blockers=backend_blockers,
    )


def summarize_environment_doctor(result: EnvironmentDoctorResult) -> str:
    lines = [
        f"Environment doctor: {len(result.runnable_targets)} runnable target(s), "
        f"{len(result.blocked_targets)} blocked target(s), {result.blocker_count} missing required item(s).",
    ]
    if result.runnable_targets:
        lines.append("Runnable: " + ", ".join(result.runnable_targets))
    if result.blocked_targets:
        lines.append("Blocked: " + ", ".join(result.blocked_targets))
    for target in result.target_statuses:
        if target.missing_required:
            lines.append(f"- {target.target}: missing {', '.join(target.missing_required)}")
    for backend, blockers in result.tracking_backend_blockers.items():
        reason = blockers.get('tractography')
        lines.append(f"- tractography.{backend}: {reason or 'dependencies available'}")
    return "\n".join(lines)


def blocked_target_reasons(result: EnvironmentDoctorResult) -> dict[str, str]:
    """Return target -> human-readable preflight blocker reason."""

    status_by_key = {item.key: item for item in result.statuses}
    reasons: dict[str, str] = {}
    for target in result.target_statuses:
        if target.status != "blocked":
            continue
        missing = []
        for key in target.missing_required:
            dependency = status_by_key.get(key)
            if dependency is None:
                missing.append(key)
                continue
            source = dependency.details.get("detected_source")
            source_suffix = f" ({source})" if source else ""
            missing.append(f"{dependency.label}{source_suffix}")
        reasons[target.target] = "preflight missing dependencies: " + ", ".join(missing)
    return reasons


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def write_environment_report(result: EnvironmentDoctorResult, output_dir: Path) -> dict[str, Path]:
    """Write JSON report plus user-editable install helper scripts."""

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "environment_doctor.json"
    report_path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    profile_path = output_dir / "dependency_profile.json"

    packages = result.missing_python_packages
    sh_path = output_dir / "install_missing_python.sh"
    ps1_path = output_dir / "install_missing_python.ps1"
    notes_path = output_dir / "install_external_tools.md"
    default_python = result.python_executable or "python3"

    if packages:
        sh_lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f'PYTHON_BIN="${{PYTHON_BIN:-{default_python}}}"',
            '"$PYTHON_BIN" -m pip install ' + " ".join(_shell_quote(pkg) for pkg in packages),
            "",
        ]
        ps1_lines = [
            '$ErrorActionPreference = "Stop"',
            f'if (-not $env:PYTHON_BIN) {{ $env:PYTHON_BIN = "{default_python}" }}',
            "& $env:PYTHON_BIN -m pip install " + " ".join(f'"{pkg}"' for pkg in packages),
            "",
        ]
    else:
        sh_lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f'PYTHON_BIN="${{PYTHON_BIN:-{default_python}}}"',
            'echo "No missing Python packages were detected for the selected targets."',
            "",
        ]
        ps1_lines = [
            '$ErrorActionPreference = "Stop"',
            f'if (-not $env:PYTHON_BIN) {{ $env:PYTHON_BIN = "{default_python}" }}',
            'Write-Host "No missing Python packages were detected for the selected targets."',
            "",
        ]
    sh_path.write_text("\n".join(sh_lines), encoding="utf-8")
    ps1_path.write_text("\n".join(ps1_lines), encoding="utf-8")

    missing_external = [
        item
        for item in result.statuses
        if item.status == "missing" and item.kind != "python"
    ]
    note_lines = [
        "# SlicerDSpace External Tool Install Notes",
        "",
        "These items cannot be installed safely without user/platform-specific choices such as admin access, license files, model data, or conda/container policy.",
        "",
    ]
    if not missing_external:
        note_lines.append("No missing external tools/resources were detected for the selected targets.")
    for item in missing_external:
        note_lines.extend(
            [
                f"## {item.label}",
                "",
                f"- Key: `{item.key}`",
                f"- Required: `{item.required}`",
                f"- Targets: `{', '.join(item.targets)}`",
                f"- Detected source: `{item.details.get('detected_source', 'not found')}`",
                # B10: a multi-line hint (for example a Teem error) would end the Markdown bullet; keep it on one line.
                f"- Suggested fix: {' '.join((item.install_hint or '').split()) or 'See docs/setup/EXTERNAL_DEPENDENCIES.md.'}",
                "",
            ]
        )
        checked_locations = item.details.get("checked_locations") or ()
        if checked_locations:
            note_lines.extend(["Checked locations:", ""])
            note_lines.extend(f"- `{location}`" for location in checked_locations)
            note_lines.append("")
    notes_path.write_text("\n".join(note_lines) + "\n", encoding="utf-8")
    profile_payload = {
        "kind": "slicerdsurfer_dependency_profile",
        "version": 1,
        "schema": "slicerddsurfer.dependency_profile.v1",
        "platform": result.platform,
        "host_platform": result.host_platform,
        "python_executable": result.python_executable,
        "current_scan": result.to_dict(),
        "install_helpers": {
            "python_sh": sh_path.as_posix(),
            "python_ps1": ps1_path.as_posix(),
            "external_notes": notes_path.as_posix(),
        },
        "portability_policy": {
            "paths_are_hints": True,
            "revalidate_after_import": True,
            "reason": (
                "External tool paths may not exist or may point to incompatible installations on another machine. "
                "Importing a profile should be followed by a fresh environment doctor scan."
            ),
        },
        "platform_notes": {
            "Linux": "Prefer distribution packages, conda modules, or documented local installs; never auto-install licensed tools silently.",
            "Darwin": "Use Homebrew/conda where supported and verify app-bundle paths such as Slicer and MRtrix3.",
            "Windows": "Prefer Slicer Python for Python packages and WSL2/conda/container backends for MRtrix/FSL-style tools.",
        },
    }
    profile_path.write_text(json.dumps(profile_payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    try:
        sh_path.chmod(0o755)
    except OSError:
        pass

    return {"json": report_path, "profile": profile_path, "python_sh": sh_path, "python_ps1": ps1_path, "external_notes": notes_path}


__all__ = [
    "DependencySpec",
    "DependencyStatus",
    "EnvironmentDoctorConfig",
    "EnvironmentDoctorResult",
    "TargetEnvironmentStatus",
    "build_dependency_specs",
    "blocked_target_reasons",
    "run_environment_doctor",
    "resolve_auto_torch_device",
    "resolve_freesurfer_home",
    "summarize_environment_doctor",
    "write_environment_report",
]
