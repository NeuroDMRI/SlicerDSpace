#!/usr/bin/env python3
"""Generate or execute SlicerDSpace dependency install plans.

This script intentionally installs Python packages only when --execute is
provided. External tools such as MRtrix3, FSL, FreeSurfer, WMA, and
TractCloud require platform-specific choices, licenses, or model data, so the
script writes install notes instead of silently modifying the system.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.environment import EnvironmentDoctorConfig, run_environment_doctor, write_environment_report
from shared.external_process import text_subprocess_kwargs
from shared.dependency_config import add_dependency_config_arguments, apply_dependency_config, write_dependency_config
from shared.python_environment import resolve_python_executable
from shared.algorithm_selection import choice_value


DEFAULT_TARGETS = (
    "validation",
    "preprocessing",
    "segmentation",
    "parcellation",
    "surface",
    "microstructure",
    "tractography",
    "bundles",
    "connectome",
)
DEPENDENCY_PROFILES = {
    "core": (
        "validation",
        "preprocessing",
        "segmentation",
        "parcellation",
        "surface",
        "microstructure",
    ),
    "tractography": (
        "validation",
        "preprocessing",
        "microstructure",
        "tractography",
        "connectome",
    ),
    "bundles": (
        "validation",
        "tractography",
        "bundles",
    ),
    "slicer-ui": (
        "validation",
        "preprocessing",
        "segmentation",
        "parcellation",
        "surface",
        "microstructure",
        "tractography",
        "bundles",
        "connectome",
    ),
    "full": DEFAULT_TARGETS,
}
@dataclass(frozen=True)
class PipInstallResult:
    requirement: str
    command: tuple[str, ...]
    returncode: int
    stdout_log: str
    stderr_log: str
    status: str
    failure_category: str = ""


def _split_csv(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    result: list[str] = []
    for value in values:
        for token in value.split(","):
            token = token.strip().lower()
            if token:
                result.append(token)
    return tuple(dict.fromkeys(result))


def _apply_dependency_config(args: argparse.Namespace) -> None:
    apply_dependency_config(args, args.dependency_config)


def _write_dependency_config(args: argparse.Namespace, path: Path) -> Path:
    return write_dependency_config(args, path)


def _safe_log_name(requirement: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in requirement)[:120] or "package"


def _failure_category(returncode: int, stderr: str) -> str:
    lower = stderr.lower()
    if returncode == 0:
        return ""
    if "no matching distribution found" in lower or "could not find a version" in lower:
        return "package_unavailable"
    if "permission denied" in lower or "not permitted" in lower or "externally-managed-environment" in lower:
        return "python_environment_not_writable"
    if "temporary failure in name resolution" in lower or "connection" in lower or "timed out" in lower:
        return "network"
    return "pip_failed"


def _install_python_packages(
    python_executable: str,
    requirements: tuple[str, ...],
    output_dir: Path,
    *,
    pip_find_links: Path | None = None,
    pip_no_index: bool = False,
) -> list[PipInstallResult]:
    log_dir = output_dir.expanduser().resolve() / "python_install_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    results: list[PipInstallResult] = []
    for requirement in requirements:
        command_parts = [python_executable, "-m", "pip", "install"]
        if pip_no_index:
            command_parts.append("--no-index")
        if pip_find_links is not None:
            command_parts.extend(["--find-links", pip_find_links.expanduser().resolve().as_posix()])
        command_parts.append(requirement)
        command = tuple(command_parts)
        completed = subprocess.run(
            list(command),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **text_subprocess_kwargs(),
        )
        safe_name = _safe_log_name(requirement)
        stdout_path = log_dir / f"{safe_name}.stdout.log"
        stderr_path = log_dir / f"{safe_name}.stderr.log"
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        results.append(
            PipInstallResult(
                requirement=requirement,
                command=command,
                returncode=completed.returncode,
                stdout_log=stdout_path.as_posix(),
                stderr_log=stderr_path.as_posix(),
                status="passed" if completed.returncode == 0 else "failed",
                failure_category=_failure_category(completed.returncode, completed.stderr),
            )
        )
    summary_path = log_dir / "python_install_results.json"
    summary_path.write_text(
        json.dumps([result.__dict__ for result in results], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check and optionally install Python dependencies for selected SlicerDSpace targets. "
            "External software is reported in install_external_tools.md."
        )
    )
    parser.add_argument("--all", action="store_true", help="Use the default product target set.")
    parser.add_argument(
        "--profile",
        choices=tuple(sorted(DEPENDENCY_PROFILES)),
        help="Named dependency profile. Use core for anatomical reconstruction, bundles for WMA/TractCloud/WMQL, or full for every target.",
    )
    parser.add_argument(
        "--targets",
        action="append",
        help="Targets to check/install. May be repeated or comma-separated.",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "install_plan", help="Where to write doctor/install reports.")
    parser.add_argument("--dependency-config", type=Path, help="Optional JSON file containing saved external dependency paths.")
    parser.add_argument("--save-dependency-config", type=Path, help="Write resolved external dependency paths to this JSON file.")
    parser.add_argument("--python-executable", help="Python executable to use for pip installation. Defaults to the current Python.")
    parser.add_argument("--python-environment", help="Conda environment name or Python executable path to use for dependency checks and pip installation.")
    parser.add_argument("--conda-executable", type=Path, help="conda or mamba executable used to resolve --python-environment names.")
    parser.add_argument(
        "--platform",
        dest="platform_name",
        choices=("Linux", "Darwin", "Windows"),
        help="Use platform-specific discovery rules for a dry-run/profile. Defaults to the current host platform.",
    )
    parser.add_argument("--execute", action="store_true", help="Actually run pip install for missing Python packages.")
    parser.add_argument(
        "--pip-find-links",
        type=Path,
        help="Optional local wheelhouse passed to pip with --find-links.",
    )
    parser.add_argument(
        "--pip-no-index",
        action="store_true",
        help="Install Python packages only from --pip-find-links without contacting a package index.",
    )
    parser.add_argument(
        "--strict-exit-code",
        action="store_true",
        help="Return exit code 2 when required dependencies are still missing after planning/installation.",
    )
    parser.add_argument("--preprocessing-backend", default="dipy", choices=("dipy", "mrtrix_fsl"))
    parser.add_argument("--tractography-backend", default="csd_act", type=choice_value,
                        help="One or comma-separated tracking methods: csd_act, ukf, dti_det, saf.")
    parser.add_argument("--connectome-backend", default="python", choices=("auto", "python", "mrtrix"))
    parser.add_argument("--ddseg-device", default="auto", help="Requested DDSeg torch device. auto uses cuda:0 when available, otherwise cpu.")
    parser.add_argument("--surface-device", default="auto", help="Requested DDSurfer torch device. auto uses cuda:0 when available, otherwise cpu.")
    parser.add_argument("--tractcloud-device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--run-free-water", action="store_true")
    parser.add_argument("--run-noddi", action="store_true")
    add_dependency_config_arguments(parser, exclude=("python_executable", "python_environment", "conda_executable"))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _apply_dependency_config(args)
        resolved_python = resolve_python_executable(
            python_executable=Path(args.python_executable).expanduser() if args.python_executable else None,
            python_environment=args.python_environment,
            conda_executable=args.conda_executable,
        )
        args.python_executable = resolved_python.as_posix()
    except Exception as exc:
        parser.error(str(exc))
    if args.save_dependency_config:
        written_config = _write_dependency_config(args, args.save_dependency_config)
        print(f"Dependency config written: {written_config}", file=sys.stderr)
    if args.all and args.profile:
        parser.error("Use either --all or --profile, not both.")
    if args.pip_no_index and args.pip_find_links is None:
        parser.error("--pip-no-index requires --pip-find-links.")
    if args.profile:
        targets = DEPENDENCY_PROFILES[args.profile]
    elif args.all:
        targets = DEFAULT_TARGETS
    else:
        targets = _split_csv(args.targets)
    if not targets:
        parser.error("No targets selected. Use --all or --targets.")

    config = EnvironmentDoctorConfig(
        targets=targets,
        repo_root=REPO_ROOT,
        output_dir=args.output_dir,
        python_executable=Path(args.python_executable).expanduser().resolve(),
        python_environment=args.python_environment,
        conda_executable=args.conda_executable,
        platform_name=args.platform_name,
        preprocessing_backend=args.preprocessing_backend,
        tractography_backend=args.tractography_backend,
        connectome_backend=args.connectome_backend,
        ddseg_device=args.ddseg_device,
        surface_device=args.surface_device,
        tractcloud_device=args.tractcloud_device,
        run_free_water=bool(args.run_free_water),
        run_noddi=bool(args.run_noddi),
        mrtrix_bin_dir=args.mrtrix_bin_dir,
        preprocessing_mrtrix_bin_dir=args.preprocessing_mrtrix_bin_dir,
        preprocessing_fsl_bin_dir=args.preprocessing_fsl_bin_dir,
        postprocess_freesurfer_home=args.postprocess_freesurfer_home,
        postprocess_runtime=args.postprocess_runtime,
        postprocess_license_file=args.postprocess_license_file,
        model_assets_dir=args.model_assets_dir,
        wma_root=args.wma_root,
        wma_atlas_dir=args.wma_atlas_dir,
        slicer_executable=args.slicer_executable,
        fiber_tract_measurements_cli=args.fiber_tract_measurements_cli,
        tract_querier_root=args.tract_querier_root,
        tractcloud_root=args.tractcloud_root,
        tractcloud_python_executable=args.tractcloud_python_executable,
        tractcloud_data_dir=args.tractcloud_data_dir,
    )
    result = run_environment_doctor(config)
    written = write_environment_report(result, args.output_dir)

    python_install_results: list[PipInstallResult] = []
    if args.execute and result.missing_python_packages:
        python_install_results = _install_python_packages(
            args.python_executable,
            tuple(result.missing_python_packages),
            args.output_dir,
            pip_find_links=args.pip_find_links,
            pip_no_index=bool(args.pip_no_index),
        )
        result = run_environment_doctor(config)
        written = write_environment_report(result, args.output_dir)

    payload = {
        "environment": result.to_dict(),
        "report_paths": {key: path.as_posix() for key, path in written.items()},
        "executed_python_install": bool(args.execute),
        "python_install_results": [install_result.__dict__ for install_result in python_install_results],
        "failed_python_installs": [
            install_result.requirement for install_result in python_install_results if install_result.status == "failed"
        ],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 2 if args.strict_exit_code and result.blocker_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
