"""Shared subject-root pipeline helpers for SlicerDSpace."""

from .branding import mirror_environment
mirror_environment()

from .pipeline import (
    RawInputBundle,
    NativeDTIBundle,
    RegisteredDTIBundle,
    ensure_native_dti_bundle,
    ensure_registered_dti_bundle,
    find_subject_root,
    inspect_subject_directory,
    is_raw_subject_root,
    resolve_preprocessed_input_bundle,
    resolve_cli_executable,
    resolve_launcher_executable,
    resolve_pythonslicer_executable,
)
from .validation import InputValidationConfig, InputValidationResult, ValidationIssue, validate_subject_inputs
from .environment import (
    EnvironmentDoctorConfig,
    EnvironmentDoctorResult,
    blocked_target_reasons,
    run_environment_doctor,
    summarize_environment_doctor,
    write_environment_report,
)
from .io_layout import (
    SubjectIOLayout,
    archive_development_outputs,
    canonical_connectome_output_dir,
    default_task_output_dir,
    default_task_output_dir_for_dependency,
    materialize_canonical_output_tree,
    normalize_run_name,
    subject_layout,
)
from .reporting import SubjectReportPaths, write_subject_reports
from .output_tree_report import OutputTreeReportPaths, write_output_tree_report
from .input_import import InputImportConfig, build_import_plan, detect_import_source_kind, prepare_inputs
from .sample_data import SampleDataEntry, load_sample_data_registry, sample_data_registry_payload
from .orchestrator import (
    ProductOrchestratorConfig,
    ProductOrchestratorResult,
    ProductTaskRecord,
    RunPlanItem,
    RunPlanResult,
    normalize_orchestrator_targets,
    plan_subject_products,
    run_subject_products,
)
from .api import run_subject_workflow

__all__ = [
    "InputValidationConfig",
    "InputValidationResult",
    "InputImportConfig",
    "EnvironmentDoctorConfig",
    "EnvironmentDoctorResult",
    "RawInputBundle",
    "NativeDTIBundle",
    "RegisteredDTIBundle",
    "SubjectIOLayout",
    "SubjectReportPaths",
    "OutputTreeReportPaths",
    "SampleDataEntry",
    "archive_development_outputs",
    "blocked_target_reasons",
    "canonical_connectome_output_dir",
    "ProductOrchestratorConfig",
    "ProductOrchestratorResult",
    "ProductTaskRecord",
    "RunPlanItem",
    "RunPlanResult",
    "ValidationIssue",
    "default_task_output_dir",
    "default_task_output_dir_for_dependency",
    "ensure_native_dti_bundle",
    "ensure_registered_dti_bundle",
    "find_subject_root",
    "build_import_plan",
    "detect_import_source_kind",
    "inspect_subject_directory",
    "is_raw_subject_root",
    "materialize_canonical_output_tree",
    "normalize_run_name",
    "normalize_orchestrator_targets",
    "resolve_preprocessed_input_bundle",
    "resolve_cli_executable",
    "resolve_launcher_executable",
    "resolve_pythonslicer_executable",
    "subject_layout",
    "plan_subject_products",
    "prepare_inputs",
    "run_environment_doctor",
    "run_subject_products",
    "run_subject_workflow",
    "summarize_environment_doctor",
    "validate_subject_inputs",
    "load_sample_data_registry",
    "sample_data_registry_payload",
    "write_environment_report",
    "write_output_tree_report",
    "write_subject_reports",
]
