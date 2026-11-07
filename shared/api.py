"""Checked Python workflow entry point; low-level orchestration stays compatible."""

from dataclasses import fields, replace
from pathlib import Path
import sys

from shared.environment import (EnvironmentDoctorConfig, blocked_target_reasons, run_environment_doctor,
                                run_needs_brain_mask, summarize_environment_doctor, write_environment_report)
from shared.io_layout import subject_layout
from shared.orchestrator import ProductOrchestratorConfig, _normalize_targets, _with_closure, run_subject_products


def environment_for_products(config: ProductOrchestratorConfig) -> EnvironmentDoctorConfig:
    """Check the selected algorithm settings and the current Python execution context."""
    config = _with_closure(config)
    targets = _normalize_targets(config)
    if not targets:
        raise ValueError('Select at least one workflow target')
    options = {field.name: getattr(config, field.name) for field in fields(EnvironmentDoctorConfig)
               if hasattr(config, field.name)}
    for name in ('free_water', 'noddi', 'dki'):
        options['run_' + name] = getattr(config, 'microstructure_run_' + name)
    for name in ('fit_method', 'matlab_executable', 'mkcurve_root'):
        options['dki_' + name] = getattr(config, 'microstructure_dki_' + name)
    options.update(targets=targets, repo_root=Path(__file__).resolve().parents[1],
        output_dir=subject_layout(config.subject_root, output_root=config.output_root).run_metadata / 'environment',
        python_executable=Path(sys.executable),
        python_path_entries=tuple(Path(path).expanduser().resolve() for path in sys.path if path and Path(path).exists()),
        check_slicer_cli_modules=True,
        brain_mask_required=run_needs_brain_mask(config.subject_root, targets=targets,
            subject_id=config.subject_id, output_root=config.output_root,
            brain_mask_backend=config.brain_mask_backend,
            force=config.preprocessing_force, resume=config.resume))
    return EnvironmentDoctorConfig(**options)


def run_subject_workflow(config: ProductOrchestratorConfig, *, log=None,
                         check_environment=True, strict_environment=False):
    """Preflight once, write its report, then invoke shared workflow execution.

    Like the CLI, partial execution blocks unavailable targets. Strict execution
    instead fails before launching products. Known prechecked callers may
    explicitly disable this check; no dependencies are installed here.
    """
    if check_environment:
        environment = environment_for_products(config)
        result = run_environment_doctor(environment)
        reports = write_environment_report(result, environment.output_dir)
        if log:
            log(summarize_environment_doctor(result))
            log('Environment report: ' + str(reports['json']))
        blocked = {key: value for key, value in blocked_target_reasons(result).items()
                   if key in environment.targets}
        if blocked and strict_environment and config.execute:
            raise RuntimeError('Required dependencies unavailable: ' + '; '.join(f'{key}: {value}' for key, value in blocked.items()))
        by_backend = {key: dict(value) for key, value in config.preflight_blocked_by_backend.items()}
        for backend, reasons in result.tracking_backend_blockers.items():
            by_backend.setdefault(backend, {}).update(reasons)
        config = replace(config, preflight_blocked_targets={**config.preflight_blocked_targets, **blocked},
                         preflight_blocked_by_backend=by_backend)
    return run_subject_products(config, log=log)
