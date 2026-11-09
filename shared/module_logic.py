"""Common scripted-module logic glue for SlicerDSpace."""

from __future__ import annotations

import logging
import os
import re
import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import slicer

from shared import (
    EnvironmentDoctorConfig,
    EnvironmentDoctorResult,
    InputValidationConfig,
    InputValidationResult,
    ProductOrchestratorConfig,
    ProductOrchestratorResult,
    RunPlanResult,
    default_task_output_dir,
    inspect_subject_directory,
    plan_subject_products,
    run_environment_doctor,
    run_subject_products,
    subject_layout,
    summarize_environment_doctor,
    validate_subject_inputs,
    write_environment_report,
)
from shared.external_process import text_subprocess_kwargs
from shared.model_assets import model_asset_candidates

LOGGER = logging.getLogger(__name__)

COMMON_PYTHON_REQUIREMENTS: tuple[tuple[str, str], ...] = (
    ("nibabel>=5.0", "nibabel"),
    ("SimpleITK", "SimpleITK"),
    ("trimesh", "trimesh"),
    ("rtree", "rtree"),
    ("tqdm", "tqdm"),
    ("networkx", "networkx"),
)


def _release_version(version: str) -> tuple[int, ...]:
    """Return a comparable numeric release tuple for torch-style versions."""

    core = str(version).split("+", 1)[0]
    numbers = [int(value) for value in re.findall(r"\d+", core)]
    return tuple((numbers + [0, 0, 0])[:3])


def _ensure_torch_via_extension(
    min_version: str = "2.0.0",
    log: Optional[Callable[[str], None]] = None,
):
    """Ensure that PyTorch is available via the official Slicer extension."""

    def _log(message: str) -> None:
        if log:
            try:
                log(message)
            except Exception:  # pragma: no cover
                LOGGER.exception("Failed to forward log message: %s", message)
        else:
            LOGGER.info(message)

    try:
        import PyTorchUtils
    except ModuleNotFoundError as exc:  # pragma: no cover - surfaced at runtime
        raise RuntimeError(
            "PyTorch extension not detected. Please install the 'PyTorch' extension via Extension Manager and restart Slicer."
        ) from exc

    torch_logic = PyTorchUtils.PyTorchUtilsLogic()

    if torch_logic.torchInstalled():
        import torch  # type: ignore[import-not-found]

        if _release_version(torch.__version__) < _release_version(min_version):
            raise RuntimeError(
                f"PyTorch {torch.__version__} is installed, but SlicerDSpace requires torch>={min_version}. "
                "Upgrade torch with the Slicer PyTorch extension and restart Slicer before running inference."
            )
        _log(f"PyTorch detected (version {torch.__version__})")
        return torch

    _log("PyTorch not found, installing via extension (may take several minutes)...")
    torch = torch_logic.installTorch(
        askConfirmation=False,
        torchVersionRequirement=f">={min_version}",
    )
    if torch is None:  # pragma: no cover - error surfaced at runtime
        raise RuntimeError(
            "PyTorch extension present, but torch installation failed. Please check network and retry in 'PyTorch Util'."
        )
    _log(f"PyTorch installed (version {torch.__version__})")
    return torch


class SharedLogicMixin:
    """Shared non-product-specific helpers for the unified module logic."""

    def _initialize_shared_logic_state(self) -> None:
        self.module_dir = Path(__file__).resolve().parent.parent
        self.segmentation_dir = self.module_dir / "anatomy" / "segmentation"
        self._pipeline_modules: Optional[Dict[str, Any]] = None
        # Missing model assets must not prevent the module UI from opening:
        # the first-run wizard is responsible for configuring/downloading them.
        self.template_dir = self._resolve_template_dir(required=False)
        self.checkpoint_root = self._resolve_checkpoint_root(required=False)
        self.template_volume = self.template_dir / "100HCP-population-mean-T2-1mm.nii.gz"
        self.registration_atlas = self._resolve_registration_atlas(required=False)
        self.logCallback = None

    def log(self, message: str) -> None:
        LOGGER.info(message)
        if self.logCallback:
            try:
                self.logCallback(message)
            except Exception:  # pragma: no cover
                LOGGER.exception("Failed to write log message via callback")

    def _ensure_python_packages(self, requirements: tuple[tuple[str, str], ...]) -> None:
        import importlib

        def ensure_package(requirement: str, import_name: Optional[str] = None) -> None:
            module_name = import_name or requirement
            try:
                importlib.import_module(module_name)
            except ModuleNotFoundError:
                self.log(f"Installing Python package: {requirement}")
                self._install_python_requirement(requirement)

        for requirement, import_name in requirements:
            ensure_package(requirement, import_name)

    def _install_python_requirement(self, requirement: str) -> None:
        """Install one Python requirement in the active Slicer Python environment."""

        env = os.environ.copy()
        resolved_requirement = requirement
        if requirement == "git+https://github.com/demianw/tract_querier.git":
            for env_name in ("SLICERDSPACE_TRACT_QUERIER_ROOT", "TRACT_QUERIER_ROOT"):
                value = os.environ.get(env_name)
                if not value:
                    continue
                candidate = Path(value).expanduser()
                if candidate.is_dir() and (candidate / "pyproject.toml").is_file():
                    resolved_requirement = candidate.as_posix()
                    env["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_TRACT_QUERIER"] = "0+local"
                    self.log(f"[Environment] Using local TractQuerier checkout from {candidate}")
                    break

        if resolved_requirement.startswith("git+") and shutil.which("git") is None:
            raise RuntimeError(
                "Git is required to install this dependency automatically. "
                "Install Git first, or configure a local checkout path instead."
            )

        try:
            import slicer.packaging as slicer_packaging  # type: ignore[import-not-found]

            if hasattr(slicer_packaging, "pip_ensure"):
                self.log(f"[Environment] Ensuring Python requirement with slicer.packaging: {resolved_requirement}")
                slicer_packaging.pip_ensure(resolved_requirement, requester="SlicerDSpace")
                return
            if hasattr(slicer_packaging, "pip_install"):
                self.log(f"[Environment] Installing Python requirement with slicer.packaging: {resolved_requirement}")
                try:
                    slicer_packaging.pip_install(resolved_requirement, show_progress=True)
                except TypeError:
                    slicer_packaging.pip_install(resolved_requirement)
                return
        except Exception as exc:
            self.log(f"[Environment] slicer.packaging unavailable or failed for {resolved_requirement}: {exc}")

        python_bin = sys.executable or "python"
        self.log(f"[Environment] Installing Python requirement with pip: {resolved_requirement}")
        try:
            completed = subprocess.run(
                [python_bin, "-m", "pip", "install", resolved_requirement],
                check=True,
                capture_output=True,
                env=env,
                **text_subprocess_kwargs(),
            )
            if completed.stdout.strip():
                self.log("[Environment] pip stdout:\n" + completed.stdout.strip())
            if completed.stderr.strip():
                self.log("[Environment] pip stderr:\n" + completed.stderr.strip())
            return
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            details = stderr or stdout or str(exc)
            raise RuntimeError(
                f"pip install failed for {resolved_requirement} using {python_bin}.\n{details}"
            ) from exc

    def _ensure_torch_requirements(self, requirements: tuple[tuple[str, str], ...]) -> None:
        _ensure_torch_via_extension(min_version="2.0.0", log=self.log)
        self._ensure_python_packages(requirements)

    def setupPythonRequirements(self) -> None:
        self._ensure_torch_requirements(COMMON_PYTHON_REQUIREMENTS)

    def _load_pipeline_modules(self) -> Dict[str, Any]:
        if self._pipeline_modules is not None:
            return self._pipeline_modules

        if str(self.module_dir) not in sys.path:
            sys.path.insert(0, str(self.module_dir))

        import importlib

        try:
            preprocessing = importlib.import_module("anatomy.surface.preprocessing")
            prediction = importlib.import_module("anatomy.surface.prediction")
            postprocessing = importlib.import_module("anatomy.surface.postprocessing")
        except ImportError as exc:  # pragma: no cover - surfaced at runtime
            raise RuntimeError(
                "Failed to import SlicerDSpace surface workflow code. Ensure surface preprocessing, prediction, and postprocessing modules are available."
            ) from exc

        self._pipeline_modules = {
            "preprocessing": preprocessing,
            "prediction": prediction,
            "postprocessing": postprocessing,
        }
        return self._pipeline_modules

    def _ensure_ddseg_on_path(self) -> None:
        if str(self.module_dir) not in sys.path:
            sys.path.insert(0, str(self.module_dir))

    def _resolve_subject_input(self, base_path: Path, subject_id: str) -> dict[str, Any]:
        inspected = inspect_subject_directory(base_path, subject_id)
        if inspected["mode"] == "missing":
            raise FileNotFoundError(
                f"Unable to resolve subject '{subject_id}' from {base_path}. "
                "Expected either a new-style subject root with input/data.nii.gz or a legacy registered directory."
            )
        return inspected

    def runInputValidation(
        self,
        *,
        subject_id: str,
        input_root: Path,
        output_dir: Optional[Path] = None,
        t1w_path: Optional[Path] = None,
        bids_metadata_path: Optional[Path] = None,
        dependency_groups: tuple[str, ...] = ("core",),
        require_phase_encoding: bool = False,
        brain_mask_backend: str = "pnl_cnn",
        brain_mask_device: str = "auto",
        pnl_mask_python: Optional[Path] = None,
        pnl_mask_script: Optional[Path] = None,
        pnl_mask_model_dir: Optional[Path] = None,
        pnl_mask_ants_bin_dir: Optional[Path] = None,
        pnl_mask_route: str = "auto",
    ) -> InputValidationResult:
        config = InputValidationConfig(
            subject_id=subject_id,
            input_root=input_root,
            output_dir=output_dir,
            t1w_path=t1w_path,
            bids_metadata_path=bids_metadata_path,
            dependency_groups=dependency_groups,
            require_phase_encoding=require_phase_encoding,
            brain_mask_backend=brain_mask_backend,
            brain_mask_device=brain_mask_device,
            pnl_mask_python=pnl_mask_python,
            pnl_mask_script=pnl_mask_script,
            pnl_mask_model_dir=pnl_mask_model_dir,
            pnl_mask_ants_bin_dir=pnl_mask_ants_bin_dir,
            pnl_mask_route=pnl_mask_route,
        )
        result = validate_subject_inputs(config)
        report = result.report_path.as_posix() if result.report_path else "not written"
        self.log(
            f"[Validation] {result.status}: {result.error_count} error(s), "
            f"{result.warning_count} warning(s). Report: {report}"
        )
        return result

    def runEnvironmentDoctor(
        self,
        *,
        targets: tuple[str, ...],
        subject_root: Path,
        output_root: Optional[Path] = None,
        brain_mask_backend: str = "pnl_cnn",
        brain_mask_device: str = "auto",
        brain_mask_required: bool = False,
        pnl_mask_python: Optional[Path] = None,
        pnl_mask_script: Optional[Path] = None,
        pnl_mask_model_dir: Optional[Path] = None,
        pnl_mask_ants_bin_dir: Optional[Path] = None,
        pnl_mask_route: str = "auto",
        tractography_backend: str = "csd_act",
        connectome_backend: str = "auto",
        ddseg_device: str = "auto",
        surface_device: str = "auto",
        tractcloud_device: str = "auto",
        microstructure_run_free_water: bool = False,
        microstructure_run_noddi: bool = False,
        microstructure_run_dki: bool = False,
        dki_fit_method: str = "MKCURVE",
        dki_matlab_executable: Optional[Path] = None,
        dki_mkcurve_root: Optional[Path] = None,
        python_executable: Optional[Path] = None,
        python_path_entries: tuple[Path, ...] = (),
        python_environment: Optional[str] = None,
        conda_executable: Optional[Path] = None,
        model_assets_dir: Optional[Path] = None,
        mrtrix_bin_dir: Optional[Path] = None,
        preprocessing_mrtrix_bin_dir: Optional[Path] = None,
        preprocessing_fsl_bin_dir: Optional[Path] = None,
        preprocessing_execution_backend: Optional[str] = None,
        postprocess_freesurfer_home: Optional[Path] = None,
        postprocess_runtime: Optional[Path] = None,
        postprocess_license_file: Optional[Path] = None,
        wma_root: Optional[Path] = None,
        wma_atlas_dir: Optional[Path] = None,
        wma_atlas: str = "org",
        slicer_executable: Optional[Path] = None,
        fiber_tract_measurements_cli: Optional[Path] = None,
        tract_querier_root: Optional[Path] = None,
        wmql_python_executable: Optional[Path] = None,
        tractcloud_root: Optional[Path] = None,
        tractcloud_python_executable: Optional[Path] = None,
        tractcloud_data_dir: Optional[Path] = None,
    ) -> EnvironmentDoctorResult:
        layout = subject_layout(subject_root.expanduser().resolve(), output_root=output_root)
        result = run_environment_doctor(
            EnvironmentDoctorConfig(
                targets=targets,
                repo_root=self.module_dir,
                output_dir=layout.run_metadata / "environment",
                python_executable=python_executable,
                python_path_entries=python_path_entries,
                loaded_slicer_modules=(tuple(name for name in ('dwiconvert', 'ukftractography')
                                             if hasattr(slicer.modules, name)) if getattr(slicer, 'app', None) else None),
                python_environment=python_environment,
                conda_executable=conda_executable,
                model_assets_dir=model_assets_dir,
                brain_mask_backend=brain_mask_backend,
                brain_mask_device=brain_mask_device,
                brain_mask_required=brain_mask_required,
                pnl_mask_python=pnl_mask_python,
                pnl_mask_script=pnl_mask_script,
                pnl_mask_model_dir=pnl_mask_model_dir,
                pnl_mask_ants_bin_dir=pnl_mask_ants_bin_dir,
                pnl_mask_route=pnl_mask_route,
                tractography_backend=tractography_backend,
                connectome_backend=connectome_backend,
                ddseg_device=ddseg_device,
                surface_device=surface_device,
                tractcloud_device=tractcloud_device,
                run_free_water=bool(microstructure_run_free_water),
                run_noddi=bool(microstructure_run_noddi),
                run_dki=bool(microstructure_run_dki),
                dki_fit_method=dki_fit_method,
                dki_matlab_executable=dki_matlab_executable,
                dki_mkcurve_root=dki_mkcurve_root,
                mrtrix_bin_dir=mrtrix_bin_dir,
                preprocessing_mrtrix_bin_dir=preprocessing_mrtrix_bin_dir,
                preprocessing_fsl_bin_dir=preprocessing_fsl_bin_dir,
                preprocessing_execution_backend=preprocessing_execution_backend,
                postprocess_freesurfer_home=postprocess_freesurfer_home,
                postprocess_runtime=postprocess_runtime,
                postprocess_license_file=postprocess_license_file,
                wma_root=wma_root,
                wma_atlas_dir=wma_atlas_dir,
                wma_atlas=wma_atlas,
                slicer_executable=slicer_executable,
                fiber_tract_measurements_cli=fiber_tract_measurements_cli,
                tract_querier_root=tract_querier_root,
                wmql_python_executable=wmql_python_executable,
                tractcloud_root=tractcloud_root,
                tractcloud_python_executable=tractcloud_python_executable,
                tractcloud_data_dir=tractcloud_data_dir,
            )
        )
        written = write_environment_report(result, layout.run_metadata / "environment")
        self.log("[Environment] " + summarize_environment_doctor(result).replace("\n", "\n[Environment] "))
        self.log(f"[Environment] Report: {written['json']}")
        self.log(f"[Environment] Python install helper: {written['python_sh']}")
        self.log(f"[Environment] External tool notes: {written['external_notes']}")
        return result

    def runEnvironmentDependencyCheck(
        self,
        *,
        targets: tuple[str, ...],
        subject_root: Optional[Path] = None,
        output_root: Optional[Path] = None,
        brain_mask_backend: str = "pnl_cnn",
        brain_mask_device: str = "auto",
        brain_mask_required: bool = False,
        pnl_mask_python: Optional[Path] = None,
        pnl_mask_script: Optional[Path] = None,
        pnl_mask_model_dir: Optional[Path] = None,
        pnl_mask_ants_bin_dir: Optional[Path] = None,
        pnl_mask_route: str = "auto",
        tractography_backend: str = "csd_act",
        connectome_backend: str = "auto",
        ddseg_device: str = "auto",
        surface_device: str = "auto",
        tractcloud_device: str = "auto",
        microstructure_run_free_water: bool = False,
        microstructure_run_noddi: bool = False,
        microstructure_run_dki: bool = False,
        dki_fit_method: str = "MKCURVE",
        dki_matlab_executable: Optional[Path] = None,
        dki_mkcurve_root: Optional[Path] = None,
        python_executable: Optional[Path] = None,
        python_path_entries: tuple[Path, ...] = (),
        python_environment: Optional[str] = None,
        conda_executable: Optional[Path] = None,
        model_assets_dir: Optional[Path] = None,
        mrtrix_bin_dir: Optional[Path] = None,
        preprocessing_mrtrix_bin_dir: Optional[Path] = None,
        preprocessing_fsl_bin_dir: Optional[Path] = None,
        preprocessing_execution_backend: Optional[str] = None,
        postprocess_freesurfer_home: Optional[Path] = None,
        postprocess_runtime: Optional[Path] = None,
        postprocess_license_file: Optional[Path] = None,
        wma_root: Optional[Path] = None,
        wma_atlas_dir: Optional[Path] = None,
        wma_atlas: str = "org",
        slicer_executable: Optional[Path] = None,
        fiber_tract_measurements_cli: Optional[Path] = None,
        tract_querier_root: Optional[Path] = None,
        wmql_python_executable: Optional[Path] = None,
        tractcloud_root: Optional[Path] = None,
        tractcloud_python_executable: Optional[Path] = None,
        tractcloud_data_dir: Optional[Path] = None,
    ) -> EnvironmentDoctorResult:
        """Check environment dependencies without requiring subject inputs."""

        if subject_root is not None and subject_root.expanduser().exists():
            report_dir = subject_layout(subject_root.expanduser().resolve(), output_root=output_root).run_metadata / "environment"
        elif output_root is not None:
            report_dir = output_root.expanduser().resolve() / "00_run_metadata" / "environment"
        else:
            report_dir = Path(str(slicer.app.temporaryPath)) / "SlicerDSpace" / "00_run_metadata" / "environment"

        result = run_environment_doctor(
            EnvironmentDoctorConfig(
                targets=targets,
                repo_root=self.module_dir,
                output_dir=report_dir,
                python_executable=python_executable,
                python_path_entries=python_path_entries,
                loaded_slicer_modules=(tuple(name for name in ('dwiconvert', 'ukftractography')
                                             if hasattr(slicer.modules, name)) if getattr(slicer, 'app', None) else None),
                python_environment=python_environment,
                conda_executable=conda_executable,
                model_assets_dir=model_assets_dir,
                brain_mask_backend=brain_mask_backend,
                brain_mask_device=brain_mask_device,
                brain_mask_required=brain_mask_required,
                pnl_mask_python=pnl_mask_python,
                pnl_mask_script=pnl_mask_script,
                pnl_mask_model_dir=pnl_mask_model_dir,
                pnl_mask_ants_bin_dir=pnl_mask_ants_bin_dir,
                pnl_mask_route=pnl_mask_route,
                tractography_backend=tractography_backend,
                connectome_backend=connectome_backend,
                ddseg_device=ddseg_device,
                surface_device=surface_device,
                tractcloud_device=tractcloud_device,
                run_free_water=bool(microstructure_run_free_water),
                run_noddi=bool(microstructure_run_noddi),
                run_dki=bool(microstructure_run_dki),
                dki_fit_method=dki_fit_method,
                dki_matlab_executable=dki_matlab_executable,
                dki_mkcurve_root=dki_mkcurve_root,
                mrtrix_bin_dir=mrtrix_bin_dir,
                preprocessing_mrtrix_bin_dir=preprocessing_mrtrix_bin_dir,
                preprocessing_fsl_bin_dir=preprocessing_fsl_bin_dir,
                preprocessing_execution_backend=preprocessing_execution_backend,
                postprocess_freesurfer_home=postprocess_freesurfer_home,
                postprocess_runtime=postprocess_runtime,
                postprocess_license_file=postprocess_license_file,
                wma_root=wma_root,
                wma_atlas_dir=wma_atlas_dir,
                wma_atlas=wma_atlas,
                slicer_executable=slicer_executable,
                fiber_tract_measurements_cli=fiber_tract_measurements_cli,
                tract_querier_root=tract_querier_root,
                wmql_python_executable=wmql_python_executable,
                tractcloud_root=tractcloud_root,
                tractcloud_python_executable=tractcloud_python_executable,
                tractcloud_data_dir=tractcloud_data_dir,
            )
        )
        written = write_environment_report(result, report_dir)
        self.log("[Environment] " + summarize_environment_doctor(result).replace("\n", "\n[Environment] "))
        self.log(f"[Environment] Report: {written['json']}")
        self.log(f"[Environment] Python install helper: {written['python_sh']}")
        self.log(f"[Environment] External tool notes: {written['external_notes']}")
        return result

    def installMissingPythonDependencies(self, packages: tuple[str, ...], progress_callback=None) -> None:
        """Install missing Python packages in the active Slicer Python environment."""

        unique_packages = tuple(dict.fromkeys(pkg for pkg in packages if pkg and pkg.strip()))
        if not unique_packages:
            self.log("[Environment] No missing Python packages to install.")
            return

        def report(index: int, total: int, message: str) -> None:
            if progress_callback:
                progress_callback(index, total, message)

        torch_requested = any(pkg == "torch" or pkg.startswith("torch>") or pkg.startswith("torch=") for pkg in unique_packages)
        if torch_requested:
            self.log("[Environment] Installing/checking torch through the Slicer PyTorch extension...")
            report(0, len(unique_packages), "Checking Slicer PyTorch extension")
            _ensure_torch_via_extension(min_version="2.0.0", log=self.log)

        for index, package in enumerate(unique_packages, start=1):
            report(index - 1, len(unique_packages), f"Installing {package}")
            if package == "torch" or package.startswith("torch>") or package.startswith("torch="):
                continue
            self.log(f"[Environment] Installing Python package: {package}")
            self._install_python_requirement(package)
            report(index, len(unique_packages), f"Installed {package}")

    def _default_task_output_dir(self, subject_root: Path, task_name: str) -> Path:
        return default_task_output_dir(subject_root, task_name)

    def runSelectedProductOrchestration(
        self,
        *,
        subject_id: str,
        subject_root: Path,
        targets: tuple[str, ...],
        output_root: Optional[Path] = None,
        run_name: str = "csd_act",
        tractography_backend: str = "csd_act",
        bundle_backend: str = "wmql",
        wmql_query_preset: str = "official",
        wmql_query_selection: tuple[str, ...] = (),
        ukf_parameters: Optional[dict] = None,
        ukf_executable: Optional[Path] = None,
        dwiconvert_executable: Optional[Path] = None,
        max_parallel_models: int = 1,
        brain_mask_backend: str = "pnl_cnn",
        brain_mask_device: str = "auto",
        pnl_mask_python: Optional[Path] = None,
        pnl_mask_script: Optional[Path] = None,
        pnl_mask_model_dir: Optional[Path] = None,
        pnl_mask_ants_bin_dir: Optional[Path] = None,
        pnl_mask_route: str = "auto",
        execute: bool = False,
        resume: bool = True,
        microstructure_run_free_water: bool = False,
        microstructure_run_noddi: bool = False,
        microstructure_run_dki: bool = False,
        microstructure_run_ifw: bool = False,
        microstructure_ifw_echo_time_ms: Optional[float] = None,
        microstructure_ifw_t2_gm_ms: float = 95.0,
        microstructure_ifw_t2_csf_ms: float = 1500.0,
        microstructure_dki_fit_method: str = "MKCURVE",
        microstructure_dki_matlab_executable: Optional[Path] = None,
        microstructure_dki_mkcurve_root: Optional[Path] = None,
        select_streamlines: int = 1_000_000,
        dti_min_length_mm: float = 10.0,
        dti_max_length_mm: float = 250.0,
        atlas_t2: Optional[Path] = None,
        model_assets_dir: Optional[Path] = None,
        ddseg_device: str = "auto",
        surface_device: str = "auto",
        tractcloud_device: str = "auto",
        mrtrix_bin_dir: Optional[Path] = None,
        preprocessing_mrtrix_bin_dir: Optional[Path] = None,
        preprocessing_fsl_bin_dir: Optional[Path] = None,
        preprocessing_execution_backend: Optional[str] = None,
        preprocessing_reverse_phase_b0: Optional[Path] = None,
        preprocessing_phase_encoding_direction: Optional[str] = None,
        preprocessing_readout_time: Optional[float] = None,
        postprocess_freesurfer_home: Optional[Path] = None,
        postprocess_runtime: Optional[Path] = None,
        postprocess_license_file: Optional[Path] = None,
        tract_querier_root: Optional[Path] = None,
        wmql_python_executable: Optional[Path] = None,
        wma_input_tractography: Optional[Path] = None,
        wma_root: Optional[Path] = None,
        wma_atlas_dir: Optional[Path] = None,
        wma_atlas: str = "org",
        slicer_executable: Optional[Path] = None,
        fiber_tract_measurements_cli: Optional[Path] = None,
        tractcloud_root: Optional[Path] = None,
        tractcloud_python_executable: Optional[Path] = None,
        tractcloud_data_dir: Optional[Path] = None,
        connectome_backend: str = "auto",
        connectome_compress_mrtrix_dense: Optional[bool] = None,
        preflight_blocked_targets: Optional[dict[str, str]] = None,
        requested_targets: tuple[str, ...] = (),
    ) -> ProductOrchestratorResult:
        result = run_subject_products(
            ProductOrchestratorConfig(
                subject_id=subject_id,
                subject_root=subject_root,
                targets=tuple(requested_targets) or targets,
                output_root=output_root,
                run_name=run_name,
                tractography_backend=tractography_backend,
                bundle_backend=bundle_backend,
                wmql_query_preset=wmql_query_preset,
                wmql_query_selection=wmql_query_selection,
                ukf_parameters=ukf_parameters or {},
                ukf_executable=ukf_executable,
                dwiconvert_executable=dwiconvert_executable,
                max_parallel_models=max_parallel_models,
                brain_mask_backend=brain_mask_backend,
                brain_mask_device=brain_mask_device,
                pnl_mask_python=pnl_mask_python,
                pnl_mask_script=pnl_mask_script,
                pnl_mask_model_dir=pnl_mask_model_dir,
                pnl_mask_ants_bin_dir=pnl_mask_ants_bin_dir,
                pnl_mask_route=pnl_mask_route,
                execute=execute,
                resume=resume,
                microstructure_run_free_water=microstructure_run_free_water,
                microstructure_run_noddi=microstructure_run_noddi,
                microstructure_run_dki=microstructure_run_dki,
                microstructure_run_ifw=microstructure_run_ifw,
                microstructure_ifw_echo_time_ms=microstructure_ifw_echo_time_ms,
                microstructure_ifw_t2_gm_ms=microstructure_ifw_t2_gm_ms,
                microstructure_ifw_t2_csf_ms=microstructure_ifw_t2_csf_ms,
                microstructure_dki_fit_method=microstructure_dki_fit_method,
                microstructure_dki_matlab_executable=microstructure_dki_matlab_executable,
                microstructure_dki_mkcurve_root=microstructure_dki_mkcurve_root,
                preprocessing_mrtrix_bin_dir=preprocessing_mrtrix_bin_dir,
                preprocessing_fsl_bin_dir=preprocessing_fsl_bin_dir,
                preprocessing_execution_backend=preprocessing_execution_backend,
                preprocessing_reverse_phase_b0=preprocessing_reverse_phase_b0,
                preprocessing_phase_encoding_direction=preprocessing_phase_encoding_direction,
                preprocessing_readout_time=preprocessing_readout_time,
                postprocess_freesurfer_home=postprocess_freesurfer_home,
                postprocess_runtime=postprocess_runtime,
                postprocess_license_file=postprocess_license_file,
                atlas_t2=atlas_t2,
                model_assets_dir=model_assets_dir,
                ddseg_device=ddseg_device,
                surface_device=surface_device,
                tractcloud_device=tractcloud_device,
                select_streamlines=select_streamlines,
                dti_min_length_mm=dti_min_length_mm,
                dti_max_length_mm=dti_max_length_mm,
                mrtrix_bin_dir=mrtrix_bin_dir,
                tract_querier_root=tract_querier_root,
                wmql_python_executable=wmql_python_executable,
                wma_input_tractography=wma_input_tractography,
                wma_root=wma_root,
                wma_atlas_dir=wma_atlas_dir,
                wma_atlas=wma_atlas,
                slicer_executable=slicer_executable,
                fiber_tract_measurements_cli=fiber_tract_measurements_cli,
                tractcloud_root=tractcloud_root,
                tractcloud_python_executable=tractcloud_python_executable,
                tractcloud_data_dir=tractcloud_data_dir,
                connectome_backend=connectome_backend,
                connectome_compress_mrtrix_dense=connectome_compress_mrtrix_dense,
                preflight_blocked_targets=preflight_blocked_targets or {},
            ),
            log=self.log,
        )
        self.log(f"[Orchestrator] Manifest: {result.run_manifest_path}")
        for task in result.tasks:
            detail = f" ({task.reason})" if task.reason else ""
            self.log(f"[Orchestrator] {task.name}: {task.status}{detail}")
        return result

    def planSelectedProductOrchestration(
        self,
        *,
        subject_id: str,
        subject_root: Path,
        targets: tuple[str, ...],
        output_root: Optional[Path] = None,
        run_name: str = "csd_act",
        tractography_backend: str = "csd_act",
        bundle_backend: str = "wmql",
        wmql_query_preset: str = "official",
        wmql_query_selection: tuple[str, ...] = (),
        ukf_parameters: Optional[dict] = None,
        ukf_executable: Optional[Path] = None,
        dwiconvert_executable: Optional[Path] = None,
        max_parallel_models: int = 1,
        brain_mask_backend: str = "pnl_cnn",
        brain_mask_device: str = "auto",
        pnl_mask_python: Optional[Path] = None,
        pnl_mask_script: Optional[Path] = None,
        pnl_mask_model_dir: Optional[Path] = None,
        pnl_mask_ants_bin_dir: Optional[Path] = None,
        pnl_mask_route: str = "auto",
        resume: bool = True,
        microstructure_run_free_water: bool = False,
        microstructure_run_noddi: bool = False,
        microstructure_run_dki: bool = False,
        microstructure_run_ifw: bool = False,
        microstructure_ifw_echo_time_ms: Optional[float] = None,
        microstructure_ifw_t2_gm_ms: float = 95.0,
        microstructure_ifw_t2_csf_ms: float = 1500.0,
        microstructure_dki_fit_method: str = "MKCURVE",
        microstructure_dki_matlab_executable: Optional[Path] = None,
        microstructure_dki_mkcurve_root: Optional[Path] = None,
        select_streamlines: int = 1_000_000,
        dti_min_length_mm: float = 10.0,
        dti_max_length_mm: float = 250.0,
        atlas_t2: Optional[Path] = None,
        model_assets_dir: Optional[Path] = None,
        ddseg_device: str = "auto",
        surface_device: str = "auto",
        tractcloud_device: str = "auto",
        mrtrix_bin_dir: Optional[Path] = None,
        preprocessing_mrtrix_bin_dir: Optional[Path] = None,
        preprocessing_fsl_bin_dir: Optional[Path] = None,
        preprocessing_execution_backend: Optional[str] = None,
        preprocessing_reverse_phase_b0: Optional[Path] = None,
        preprocessing_phase_encoding_direction: Optional[str] = None,
        preprocessing_readout_time: Optional[float] = None,
        postprocess_freesurfer_home: Optional[Path] = None,
        postprocess_runtime: Optional[Path] = None,
        postprocess_license_file: Optional[Path] = None,
        tract_querier_root: Optional[Path] = None,
        wmql_python_executable: Optional[Path] = None,
        wma_input_tractography: Optional[Path] = None,
        wma_root: Optional[Path] = None,
        wma_atlas_dir: Optional[Path] = None,
        wma_atlas: str = "org",
        slicer_executable: Optional[Path] = None,
        fiber_tract_measurements_cli: Optional[Path] = None,
        tractcloud_root: Optional[Path] = None,
        tractcloud_python_executable: Optional[Path] = None,
        tractcloud_data_dir: Optional[Path] = None,
        connectome_backend: str = "auto",
        connectome_compress_mrtrix_dense: Optional[bool] = None,
        preflight_blocked_targets: Optional[dict[str, str]] = None,
        requested_targets: tuple[str, ...] = (),
    ) -> RunPlanResult:
        plan = plan_subject_products(
            ProductOrchestratorConfig(
                subject_id=subject_id,
                subject_root=subject_root,
                targets=tuple(requested_targets) or targets,
                output_root=output_root,
                run_name=run_name,
                tractography_backend=tractography_backend,
                bundle_backend=bundle_backend,
                wmql_query_preset=wmql_query_preset,
                wmql_query_selection=wmql_query_selection,
                ukf_parameters=ukf_parameters or {},
                ukf_executable=ukf_executable,
                dwiconvert_executable=dwiconvert_executable,
                max_parallel_models=max_parallel_models,
                brain_mask_backend=brain_mask_backend,
                brain_mask_device=brain_mask_device,
                pnl_mask_python=pnl_mask_python,
                pnl_mask_script=pnl_mask_script,
                pnl_mask_model_dir=pnl_mask_model_dir,
                pnl_mask_ants_bin_dir=pnl_mask_ants_bin_dir,
                pnl_mask_route=pnl_mask_route,
                execute=True,
                resume=resume,
                microstructure_run_free_water=microstructure_run_free_water,
                microstructure_run_noddi=microstructure_run_noddi,
                microstructure_run_dki=microstructure_run_dki,
                microstructure_run_ifw=microstructure_run_ifw,
                microstructure_ifw_echo_time_ms=microstructure_ifw_echo_time_ms,
                microstructure_ifw_t2_gm_ms=microstructure_ifw_t2_gm_ms,
                microstructure_ifw_t2_csf_ms=microstructure_ifw_t2_csf_ms,
                microstructure_dki_fit_method=microstructure_dki_fit_method,
                microstructure_dki_matlab_executable=microstructure_dki_matlab_executable,
                microstructure_dki_mkcurve_root=microstructure_dki_mkcurve_root,
                preprocessing_mrtrix_bin_dir=preprocessing_mrtrix_bin_dir,
                preprocessing_fsl_bin_dir=preprocessing_fsl_bin_dir,
                preprocessing_execution_backend=preprocessing_execution_backend,
                preprocessing_reverse_phase_b0=preprocessing_reverse_phase_b0,
                preprocessing_phase_encoding_direction=preprocessing_phase_encoding_direction,
                preprocessing_readout_time=preprocessing_readout_time,
                postprocess_freesurfer_home=postprocess_freesurfer_home,
                postprocess_runtime=postprocess_runtime,
                postprocess_license_file=postprocess_license_file,
                atlas_t2=atlas_t2,
                model_assets_dir=model_assets_dir,
                ddseg_device=ddseg_device,
                surface_device=surface_device,
                tractcloud_device=tractcloud_device,
                select_streamlines=select_streamlines,
                dti_min_length_mm=dti_min_length_mm,
                dti_max_length_mm=dti_max_length_mm,
                mrtrix_bin_dir=mrtrix_bin_dir,
                tract_querier_root=tract_querier_root,
                wmql_python_executable=wmql_python_executable,
                wma_input_tractography=wma_input_tractography,
                wma_root=wma_root,
                wma_atlas_dir=wma_atlas_dir,
                wma_atlas=wma_atlas,
                slicer_executable=slicer_executable,
                fiber_tract_measurements_cli=fiber_tract_measurements_cli,
                tractcloud_root=tractcloud_root,
                tractcloud_python_executable=tractcloud_python_executable,
                tractcloud_data_dir=tractcloud_data_dir,
                connectome_backend=connectome_backend,
                connectome_compress_mrtrix_dense=connectome_compress_mrtrix_dense,
                preflight_blocked_targets=preflight_blocked_targets or {},
            ),
            write=True,
        )
        self.log(f"[Orchestrator] Run plan: {plan.run_plan_path}")
        for item in plan.items:
            detail = f" ({item.reason})" if item.reason else ""
            self.log(f"[Orchestrator] plan {item.target}: {item.status}{detail}")
        return plan

    def _resolve_optional_output_dir(
        self,
        output_dir: Optional[Path],
        *,
        subject_root: Optional[Path],
        task_name: str,
    ) -> Path:
        if output_dir is not None:
            resolved = output_dir.expanduser().resolve()
            resolved.mkdir(parents=True, exist_ok=True)
            return resolved
        if subject_root is None:
            raise ValueError(f"Output directory is required for {task_name} when no subject root is available.")
        resolved = self._default_task_output_dir(subject_root, task_name)
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    def _load_model_node(self, stl_path: Path, surface: str, subject_id: str) -> bool:
        is_ras_companion = "_predicted_" in stl_path.name and stl_path.name.endswith("_orig.stl")
        stem = stl_path.stem.removesuffix("_orig") if is_ras_companion else stl_path.stem
        name = f"{subject_id}_{surface}_{stem.split('_')[-1]}"
        coordinates = (slicer.vtkMRMLStorageNode.CoordinateSystemRAS if is_ras_companion
                       else slicer.vtkMRMLStorageNode.CoordinateSystemLPS)
        success, model_node = slicer.util.loadNodeFromFile(
            stl_path.as_posix(), "ModelFile", properties={"coordinateSystem": coordinates}, returnNode=True,
        )
        if success and model_node:
            model_node.SetName(name)
            model_node.SetAttribute("SlicerDDSurfer.SourceCoordinateSystem", "RAS" if is_ras_companion else "LPS")
            return True
        return False

    def _load_scalar_volume_node(self, volume_path: Path, node_name: str) -> bool:
        success, volume_node = slicer.util.loadVolume(
            volume_path.as_posix(),
            properties={"name": node_name, "show": False},
            returnNode=True,
        )
        if success and volume_node:
            volume_node.SetName(node_name)
            return True
        return False

    def _load_table_node(self, table_path: Path, node_name: str) -> bool:
        table_node = slicer.util.loadTable(table_path.as_posix())
        if table_node:
            table_node.SetName(node_name)
            return True
        return False

    def _load_fiber_bundle_node(
        self, fiber_path: Path, node_name: str, *, preview_dir: Optional[Path] = None,
    ) -> bool:
        if not fiber_path.is_file():
            return False
        loader = getattr(slicer.util, "loadFiberBundle", None)
        if fiber_path.suffix.lower() in {".trk", ".tck"}:
            return self._load_streamline_file_as_fiber_bundle(fiber_path, node_name, preview_dir=preview_dir)
        app = getattr(slicer, "app", None)
        manager = app.moduleManager().factoryManager() if hasattr(app, "moduleManager") else None
        scene = getattr(slicer, "mrmlScene", None)
        native_available = (
            scene.IsNodeClassRegistered("vtkMRMLFiberBundleNode")
            or manager is not None and "TractographyDisplay" in set(manager.registeredModuleNames())
        ) if hasattr(scene, "IsNodeClassRegistered") else True
        if loader is not None and native_available:
            try:
                fiber_node = loader(fiber_path.as_posix())
            except Exception as exc:  # pragma: no cover - depends on SlicerDMRI extension availability
                self.log(f"[Slicer] Native FiberBundle loader could not read {fiber_path}: {exc}")
                fiber_node = None
            if fiber_node:
                fiber_node.SetName(node_name)
                fiber_node.SetAttribute("SlicerDDSurfer.SourcePath", fiber_path.as_posix())
                return True
        if fiber_path.suffix.lower() in {".vtk", ".vtp"}:
            try:
                from shared.result_scene import tract_file_coordinate_system
                coordinates = tract_file_coordinate_system(fiber_path)
                storage = getattr(slicer, "vtkMRMLStorageNode", None)
                coordinate_system = getattr(storage, "CoordinateSystem" + coordinates, 0 if coordinates == "RAS" else 1)
                loader = getattr(slicer.util, "loadNodeFromFile", None)
                properties = {"coordinateSystem": coordinate_system}
                if loader is not None:
                    success, model_node = loader(fiber_path.as_posix(), "ModelFile", properties=properties, returnNode=True)
                else:
                    success, model_node = slicer.util.loadModel(fiber_path.as_posix(), properties=properties, returnNode=True)
            except Exception as exc:  # pragma: no cover - depends on Slicer model IO availability
                self.log(f"[Slicer] Generic model loader could not read {fiber_path}: {exc}")
                success, model_node = False, None
            if success and model_node:
                model_node.SetName(node_name)
                model_node.SetAttribute("SlicerDDSurfer.SourcePath", fiber_path.as_posix())
                model_node.SetAttribute("SlicerDDSurfer.DisplayFallback", "vtkMRMLModelNode")
                model_node.SetAttribute("SlicerDDSurfer.SourceCoordinateSystem", coordinates)
                self.log(f"[Slicer] Loaded {fiber_path} as a model because SlicerDMRI fiber IO is unavailable")
                return True
        self.log(f"[Slicer] No compatible fiber or model loader could read {fiber_path}")
        return False

    def _load_streamline_file_as_fiber_bundle(
        self, fiber_path: Path, node_name: str, *, max_streamlines: int = 30_000,
        preview_dir: Optional[Path] = None,
    ) -> bool:
        """Load a lazy, uniformly bounded TRK/TCK display, never modify the source."""

        try:
            import nibabel as nib
            import numpy as np
            import vtk
            from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
            from shared.auto_results import uniform_streamline_indices
            from shared.result_scene import result_nodes, source_stamp

            if max_streamlines < 1:
                raise ValueError("Display streamline limit must be positive")
            original_stamp = source_stamp(fiber_path)
            source = nib.streamlines.load(fiber_path.as_posix(), lazy_load=True)
            tractogram = source.tractogram
            if tractogram.affine_to_rasmm is None:
                raise ValueError("Tractogram does not declare its world coordinate transform")
            if not np.allclose(tractogram.affine_to_rasmm, np.eye(4)):
                tractogram = tractogram.to_world(lazy=True)
            total = int(source.header.get("nb_streamlines", 0))
            # An unknown header count needs a streaming count pass, not an eager load.
            if total <= 0:
                total = sum(1 for _ in tractogram.streamlines)
            indices = iter(uniform_streamline_indices(total, max_streamlines))
            next_index = next(indices, None)
            chunks = []
            for index, streamline in enumerate(tractogram.streamlines):
                if next_index is None:
                    break
                if index != next_index:
                    continue
                next_index = next(indices, None)
                coordinates = np.array(streamline, dtype=np.float32, copy=True)
                if coordinates.ndim == 2 and coordinates.shape[0] >= 2 and coordinates.shape[1] == 3:
                    if np.all(np.isfinite(coordinates)):
                        chunks.append(coordinates)
            streamline_count = len(chunks)
            if not streamline_count:
                self.log(f"[Slicer] No valid display streamlines were found in {fiber_path}")
                return False
            coordinates = np.concatenate(chunks)
            points = vtk.vtkPoints()
            points.SetData(numpy_to_vtk(coordinates, deep=True))
            lines = vtk.vtkCellArray()
            id_dtype = np.int64 if vtk.vtkIdTypeArray().GetDataTypeSize() == 8 else np.int32
            offsets = np.cumsum([0, *(len(chunk) for chunk in chunks)], dtype=id_dtype)
            lines.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True),
                          numpy_to_vtkIdTypeArray(np.arange(len(coordinates), dtype=id_dtype), deep=True))
            poly_data = vtk.vtkPolyData()
            poly_data.SetPoints(points)
            poly_data.SetLines(lines)
            space = vtk.vtkStringArray()
            space.SetName("SPACE")
            space.InsertNextValue("RAS")
            poly_data.GetFieldData().AddArray(space)
            scene = slicer.mrmlScene
            fiber_class = "vtkMRMLFiberBundleNode"
            if preview_dir is not None:
                import hashlib
                import tempfile
                preview_dir = preview_dir.expanduser().resolve()
                preview_dir.mkdir(parents=True, exist_ok=True)
                token = hashlib.sha256(f"{fiber_path.resolve()}:{original_stamp}:{max_streamlines}:ras-preview-v2".encode()).hexdigest()[:16]
                preview_path = preview_dir / f"{fiber_path.stem}_{token}_display.vtk"
                provenance_path = preview_path.with_suffix(".json")
                if source_stamp(fiber_path) != original_stamp:
                    raise RuntimeError("Source tractogram changed while generating its display preview")
                provenance = {
                    "kind": "slicerddsurfer_display_preview", "version": 1,
                    "source_path": str(fiber_path.resolve()), "source_stamp": original_stamp,
                    "preview_path": str(preview_path), "coordinate_system": "RAS",
                    "source_streamline_count": total, "display_streamline_count": streamline_count,
                    "display_streamline_limit": max_streamlines, "sampling": "uniform",
                    "display_only": True, "quantitative_source": str(fiber_path.resolve()),
                }
                with tempfile.TemporaryDirectory(dir=preview_dir, prefix=".display-preview-") as temporary:
                    staged = Path(temporary) / preview_path.name
                    writer = vtk.vtkPolyDataWriter()
                    writer.SetFileName(str(staged))
                    writer.SetHeader("SlicerDSpace display preview. SPACE=RAS")
                    writer.SetInputData(poly_data)
                    writer.SetFileTypeToBinary()
                    if writer.Write() != 1:
                        raise RuntimeError("Failed to persist the display VTK preview")
                    staged_json = Path(temporary) / provenance_path.name
                    staged_json.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
                    os.replace(staged, preview_path)
                    os.replace(staged_json, provenance_path)
                before = {node.GetID() for node in result_nodes(slicer)}
                if not self._load_fiber_bundle_node(preview_path, node_name):
                    return False
                new_nodes = [node for node in result_nodes(slicer) if node.GetID() not in before]
                if not new_nodes:
                    return False
                fiber_node = new_nodes[0]
                node_class = fiber_node.GetClassName()
                fiber_node.SetAttribute("SlicerDDSurfer.DisplayPreviewPath", str(preview_path))
                fiber_node.SetAttribute("SlicerDDSurfer.DisplayPreviewProvenancePath", str(provenance_path))
                fiber_node.SetAttribute("SlicerDDSurfer.PreviewSourceStamp", original_stamp)
            else:
                fiber_class_available = bool(
                    hasattr(scene, "IsNodeClassRegistered") and scene.IsNodeClassRegistered(fiber_class)
                )
                if not fiber_class_available:
                    factory_manager = slicer.app.moduleManager().factoryManager()
                    registered_modules = set(factory_manager.registeredModuleNames())
                    if "TractographyDisplay" in registered_modules:
                        factory_manager.loadModules(["TractographyDisplay"])
                        fiber_class_available = bool(scene.IsNodeClassRegistered(fiber_class))
                node_class = fiber_class if fiber_class_available else "vtkMRMLModelNode"
                fiber_node = scene.AddNewNodeByClass(node_class, node_name)
                if fiber_node is None:
                    self.log(f"[Slicer] Neither {fiber_class} nor vtkMRMLModelNode is available for {fiber_path}")
                    return False
                fiber_node.SetAndObservePolyData(poly_data)
            fiber_node.SetAttribute("SlicerDDSurfer.SourcePath", fiber_path.as_posix())
            fiber_node.SetAttribute("SlicerDDSurfer.SourceCoordinateSystem", "RAS")
            fiber_node.SetAttribute("SlicerDDSurfer.StreamlineCount", str(streamline_count))
            fiber_node.SetAttribute("SlicerDDSurfer.SourceStreamlineCount", str(total))
            fiber_node.SetAttribute("SlicerDDSurfer.DisplayStreamlineLimit", str(max_streamlines))
            fiber_node.SetAttribute("SlicerDDSurfer.DisplaySubsampled", "true" if total > max_streamlines else "false")
            fiber_node.SetAttribute("SlicerDDSurfer.DisplaySampling", "uniform; display only; original file unchanged")
            fiber_node.SetAttribute("SlicerDDSurfer.DisplayPreview", "true")
            fiber_node.SetAttribute("SlicerDDSurfer.QuantitativeSourcePath", fiber_path.as_posix())
            fiber_node.SetAttribute("SlicerDDSurfer.MeasurementWarning", "Display preview only; quantitative measurements must use the original tractogram")
            if node_class != fiber_class:
                fiber_node.SetAttribute("SlicerDDSurfer.DisplayFallback", node_class)
            fiber_node.CreateDefaultDisplayNodes()
            self.log(
                f"[Slicer] Loaded {streamline_count} streamlines from {fiber_path} as {node_class}"
            )
            return True
        except Exception as exc:  # pragma: no cover - requires a real Slicer/SlicerDMRI runtime
            self.log(f"[Slicer] Failed to convert streamline file {fiber_path}: {exc}")
            return False

    def clearResultsFromManifest(self, manifest_path: Path) -> int:
        from shared.result_scene import clear_run
        return clear_run(slicer, str(manifest_path.expanduser().resolve()))

    def loadKeyResultsFromManifest(
        self, manifest_path: Path, *, max_items: int = 40, mode: str = "reuse", automatic: bool = False,
    ) -> dict[str, Any]:
        """Load only explicitly successful manifest outputs, grouped by subject/run."""
        from shared.result_scene import result_candidates, result_nodes, owned_nodes, clear_run, register_node, scalar_preset, source_stamp
        from shared.result_scene import automatic_display_preset, annotate_automatic_result, result_file_suffix, show_automatic_results
        if automatic and hasattr(getattr(slicer, "app", None), "thread"):
            import qt
            if qt.QThread.currentThread() != slicer.app.thread():
                raise RuntimeError("Automatic result loading must run on the Slicer GUI thread")
        if mode not in {"reuse", "replace", "append"}:
            raise ValueError("Result mode must be reuse, replace or append")
        manifest_path = manifest_path.expanduser().resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_id = str(manifest_path)
        subject = str(manifest.get("subject_id") or "subject")
        run_name = str(manifest.get("run_name") or "Test1")
        if mode == "replace":
            clear_run(slicer, run_id)
        existing = {}
        for node in owned_nodes(slicer, run_id):
            existing.setdefault(node.GetAttribute("SlicerDDSurfer.SourcePath"), []).append(node)
        if automatic:
            from shared.auto_results import automatic_result_candidates
            candidates, skipped, deferred = automatic_result_candidates(manifest)
            for entry in skipped:
                if entry.get("reason_category") in {"qc_failed", "task_failed"}:
                    failed_path = entry.get('path')
                    failed_product = entry.get('product') or entry.get('label')
                    for source, nodes in existing.items():
                        retained = []
                        for node in nodes:
                            affected = (source == failed_path if failed_path else
                                        node.GetAttribute('SlicerDDSurfer.Product') == failed_product)
                            if affected and node.GetAttribute('SlicerDDSurfer.AutomaticLoaded') == 'true':
                                slicer.mrmlScene.RemoveNode(node)
                                self.log(f'[Slicer] Removed automatic preview for failed result: {source}')
                            else:
                                retained.append(node)
                        existing[source] = retained
        else:
            candidates, skipped = result_candidates(manifest)
            deferred = candidates[max(0, max_items):]
            candidates = candidates[:max(0, max_items)]
        loaded, reused = [], []
        for candidate in candidates:
            path, label = Path(candidate["path"]), candidate["label"]
            previous_nodes = existing.get(candidate['path'], [])
            if automatic:
                automatic_nodes = [node for node in previous_nodes
                                   if node.GetAttribute('SlicerDDSurfer.AutomaticLoaded') == 'true']
                # Reuse a matching manual node without claiming ownership; an
                # older manual review is retained alongside the new preview.
                previous_nodes = automatic_nodes or [node for node in previous_nodes
                    if node.GetAttribute('SlicerDDSurfer.SourceStamp') == source_stamp(path)]
            if mode == "reuse" and previous_nodes:
                if all(node.GetAttribute("SlicerDDSurfer.SourceStamp") == source_stamp(path) for node in previous_nodes):
                    if automatic:
                        for node in previous_nodes:
                            if node.GetAttribute('SlicerDDSurfer.AutomaticLoaded') == 'true':
                                annotate_automatic_result(node, candidate)
                    reused.append({**candidate, "node_ids": [node.GetID() for node in previous_nodes]})
                    continue
                elif automatic:
                    for node in previous_nodes:
                        if node.GetAttribute('SlicerDDSurfer.AutomaticLoaded') == 'true':
                            slicer.mrmlScene.RemoveNode(node)
                else:
                    skipped.append({"path": str(path), "reason": "Source changed since loading; use Replace to reload this run"})
                    continue
            before = {node.GetID() for node in result_nodes(slicer)}
            try:
                suffix = result_file_suffix(path)
                name = f"{subject}_{label}"
                if suffix in {".nii", ".nii.gz", ".mgz"}:
                    categorical = candidate.get("display_role", "").startswith("categorical") or candidate["product"].startswith(("segmentation.", "parcellation.")) and any(
                        word in candidate["key"].lower() for word in ("label", "segmentation", "wmparc", "aparc"))
                    if categorical:
                        success = bool(slicer.util.loadLabelVolume(str(path), properties={"name": name, "show": False}))
                    else:
                        success = self._load_scalar_volume_node(path, name)
                elif suffix == ".nrrd":
                    success = self._load_scalar_volume_node(path, name)  # D6: RGB DEC map loads as a vector volume
                elif suffix in {".csv", ".tsv"}:
                    success = self._load_table_node(path, name)
                elif suffix == ".stl":
                    success = self._load_model_node(path, candidate["product"], subject)
                else:
                    if automatic:
                        output_root = Path(manifest.get("output_root") or manifest_path.parent.parent)
                        success = self._load_fiber_bundle_node(
                            path, name, preview_dir=output_root / "06_slicer_scene_report" / "display_previews",
                        )
                    else:
                        success = self._load_fiber_bundle_node(path, name)
                nodes = [node for node in result_nodes(slicer) if node.GetID() not in before]
                if not success or not nodes:
                    raise RuntimeError("Slicer loader returned no result node")
                for node in nodes:
                    register_node(slicer, node, run_id, subject, run_name, candidate["product"], str(path))
                    if automatic:
                        automatic_display_preset(slicer, node, candidate)
                    elif node.IsA("vtkMRMLScalarVolumeNode") and not node.IsA("vtkMRMLLabelMapVolumeNode"):
                        scalar_preset(node, label)
                loaded.append({**candidate, "node_ids": [node.GetID() for node in nodes]})
                existing[candidate["path"]] = nodes
            except Exception as exc:
                if automatic:
                    for node in result_nodes(slicer):
                        if node.GetID() not in before:
                            slicer.mrmlScene.RemoveNode(node)
                skipped.append({"path": str(path), "label": label, "reason": str(exc)})
        view = show_automatic_results(slicer, loaded + reused, newly_loaded=loaded) if automatic and loaded else {}
        self.log(f"[Slicer] Loaded {len(loaded)}, reused {len(reused)}, deferred {len(deferred)} result(s).")
        return {"loaded": loaded, "reused": reused, "skipped": skipped, "deferred": deferred,
                "manifest_path": run_id, "view_node_ids": view}

    def _resolve_template_dir(self, *, required: bool = True) -> Path:
        from anatomy.surface.pipeline import resolve_surface_template_dir
        resolved = resolve_surface_template_dir()
        if resolved is not None:
            return resolved
        if not required:
            return self.module_dir / 'surface/template'
        raise FileNotFoundError('Download the current DDSurfer surface asset profile first.')

    def _resolve_checkpoint_root(self, *, required: bool = True) -> Path:
        from anatomy.surface.pipeline import resolve_surface_checkpoint_root
        resolved = resolve_surface_checkpoint_root()
        if resolved is not None:
            return resolved
        if not required:
            return self.module_dir / 'surface/weights'
        raise FileNotFoundError('Download the current DDSurfer surface asset profile first.')

    def _resolve_registration_atlas(self, *, required: bool = True) -> Path:
        from anatomy.surface.pipeline import resolve_surface_registration_atlas
        resolved = resolve_surface_registration_atlas()
        if resolved is not None:
            return resolved
        if not required:
            return self.module_dir / 'surface/template/100HCP-population-mean-T2-1mm.nii.gz'
        raise FileNotFoundError('Download the current DDSurfer surface asset profile first.')
