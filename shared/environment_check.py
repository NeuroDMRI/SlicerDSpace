"""Serializable environment-check contract shared by GUI and its worker."""

from dataclasses import fields
from pathlib import Path

from shared.environment import DependencyStatus, EnvironmentDoctorConfig, EnvironmentDoctorResult, TargetEnvironmentStatus

PATH_FIELDS = {'repo_root', 'output_dir', 'python_executable', 'conda_executable', 'pnl_mask_python',
               'pnl_mask_script', 'pnl_mask_model_dir', 'pnl_mask_ants_bin_dir',
               'ukf_executable', 'dwiconvert_executable', 'dki_matlab_executable', 'dki_mkcurve_root',
               'mrtrix_bin_dir', 'preprocessing_mrtrix_bin_dir', 'preprocessing_fsl_bin_dir',
               'postprocess_freesurfer_home', 'postprocess_runtime', 'postprocess_license_file',
               'model_assets_dir', 'wma_root', 'wma_atlas_dir',
               'slicer_executable', 'fiber_tract_measurements_cli', 'tract_querier_root', 'tractcloud_root',
               'tractcloud_python_executable', 'tractcloud_data_dir'}


def check_config(options, *, repo_root, output_dir, loaded_modules):
    known = {field.name for field in fields(EnvironmentDoctorConfig)}
    values = {key: value for key, value in options.items() if key in known}
    for name in ('free_water', 'noddi', 'dki'):
        values['run_' + name] = bool(options.get('microstructure_run_' + name, False))
    values.update(repo_root=str(repo_root), output_dir=str(output_dir), loaded_slicer_modules=list(loaded_modules))
    return values


def config_from_dict(values):
    values = dict(values)
    for key in PATH_FIELDS:
        if values.get(key) is not None:
            values[key] = Path(values[key])
    for key in ('targets', 'loaded_slicer_modules'):
        if values.get(key) is not None:
            values[key] = tuple(values[key])
    values['python_path_entries'] = tuple(Path(value) for value in values.get('python_path_entries', ()))
    return EnvironmentDoctorConfig(**values)


def result_from_dict(record):
    if record.get('kind') != 'environment_doctor' or record.get('version') != 1:
        raise ValueError('Unsupported environment check result')
    statuses = tuple(DependencyStatus(**{**item, 'targets': tuple(item['targets'])}) for item in record['dependencies'])
    targets = tuple(TargetEnvironmentStatus(**{**item, 'missing_required': tuple(item['missing_required']),
                                               'missing_optional': tuple(item['missing_optional'])})
                    for item in record['target_statuses'])
    summary = record['summary']
    return EnvironmentDoctorResult(platform=record['platform'], host_platform=record['host_platform'],
        python_executable=record['python_executable'], targets=tuple(record['targets']), statuses=statuses,
        target_statuses=targets, blocker_count=int(summary['blocker_count']),
        missing_python_packages=tuple(summary['missing_python_packages']),
        missing_external_groups=tuple(summary['missing_external_groups']),
        tracking_backend_blockers={str(backend): dict(blockers) for backend, blockers in
                                   record.get('tracking_backend_blockers', {}).items()})
