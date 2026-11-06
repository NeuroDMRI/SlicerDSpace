#!/usr/bin/env python3
"""Run the SlicerDSpace release-validation evidence chain.

The default mode is conservative: it plans the scientific workflows and runs
the inexpensive repository/report checks, then writes one JSON report. Use
``--execute`` only when the selected subject and optional batch root are ready
for long-running product execution.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ValidationStep:
    name: str
    command: tuple[str, ...]
    required: bool = True
    skipped_reason: str = ""

    @property
    def skipped(self) -> bool:
        return bool(self.skipped_reason)


def _path(value: Path | None) -> str | None:
    return value.expanduser().resolve().as_posix() if value is not None else None


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else None


def _subject_id(subject_root: Path, explicit: str | None) -> str:
    return (explicit or subject_root.expanduser().resolve().name).strip()


def _append_dependency_config(command: list[str], dependency_config: Path | None) -> None:
    if dependency_config is not None:
        command.extend(["--dependency-config", dependency_config.expanduser().resolve().as_posix()])


def _slicer_smoke_command(
    *,
    args: argparse.Namespace,
    module_dir: Path,
    script_name: str,
) -> tuple[str, ...]:
    command = [
        "env",
        f"SLICERDSPACE_EXTENSION_MODULE_DIR={module_dir.as_posix()}",
        f"SLICERDSPACE_EXTENSION_ARCHIVE={(args.report_dir / 'packages' / 'SlicerDSpace.tar.gz').as_posix()}",
        f"SLICERDSPACE_SMOKE_SUBJECT_ROOT={args.subject_root.expanduser().resolve().as_posix()}",
        f"SLICERDSPACE_SMOKE_SUBJECT_ID={_subject_id(args.subject_root, args.subject_id)}",
        f"SLICERDSPACE_SMOKE_RUN_NAME={args.run_name}",
        "SLICERDSPACE_SMOKE_MICROSTRUCTURE_STAGES=dti",
        "SLICERDSPACE_SMOKE_PLAN_ONLY=1",
        "SLICERDSPACE_DEMO_EXPECT_NODE_TYPE=model",
        f"PATH={args.xvfb_run.parent.as_posix()}:{os.environ.get('PATH', '')}",
        args.xvfb_run.as_posix(),
        "-a",
        args.slicer_launcher.as_posix(),
        "--no-splash",
        "--disable-cli-modules",
    ]
    if script_name != "slicer_extension_manager_install_smoke.py":
        command.extend(["--additional-module-paths", module_dir.as_posix()])
    command.extend(["--python-script", (REPO_ROOT / "Testing" / "Python" / script_name).as_posix()])
    return tuple(command)


def _subject_command(
    *,
    args: argparse.Namespace,
    subject_root: Path,
    output_root: Path,
    subject_id: str,
    targets: str,
    bundle_backend: str | None = None,
    run_name_suffix: str = "",
    forwarded_args: list[str],
) -> list[str]:
    run_name = args.run_name + run_name_suffix
    command = [
        sys.executable,
        (REPO_ROOT / "run_subject_pipeline.py").as_posix(),
        "--subject-root",
        subject_root.as_posix(),
        "--subject-id",
        subject_id,
        "--output-root",
        output_root.as_posix(),
        "--run-name",
        run_name,
        "--with-validation",
        "--targets",
        targets,
        "--microstructure-stages",
        "dti",
        "--tractography-backend",
        args.tractography_backend,
        "--connectome-backend",
        args.connectome_backend,
    ]
    if bundle_backend is not None:
        command.extend(["--bundle-backend", bundle_backend])
    if args.execute:
        command.append("--execute")
    else:
        command.append("--plan")
    _append_dependency_config(command, args.dependency_config)
    command.extend(forwarded_args)
    return command


def build_validation_steps(args: argparse.Namespace, forwarded_args: list[str]) -> list[ValidationStep]:
    subject_root = args.subject_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve() if args.output_root else subject_root / "outputs"
    report_dir = args.report_dir.expanduser().resolve()
    subject_id = _subject_id(subject_root, args.subject_id)
    steps: list[ValidationStep] = []

    steps.append(
        ValidationStep(
            "single_subject_all_backends",
            tuple(
                _subject_command(
                    args=args,
                    subject_root=subject_root,
                    output_root=output_root,
                    subject_id=subject_id,
                    targets="preprocessing,segmentation,parcellation,surface,microstructure,tractography,bundles,connectome",
                    bundle_backend="all",
                    forwarded_args=forwarded_args,
                )
            ),
        )
    )
    for backend in ("wmql", "wma", "tractcloud"):
        steps.append(
            ValidationStep(
                f"bundle_backend_{backend}",
                tuple(
                    _subject_command(
                        args=args,
                        subject_root=subject_root,
                        output_root=output_root,
                        subject_id=subject_id,
                        targets="bundles",
                        bundle_backend=backend,
                        run_name_suffix=f"_{backend}",
                        forwarded_args=forwarded_args,
                    )
                ),
                required=False,
            )
        )

    if args.batch_root:
        batch_command = [
            sys.executable,
            (REPO_ROOT / "scripts" / "run_batch_pipeline.py").as_posix(),
            "--batch-root",
            args.batch_root.expanduser().resolve().as_posix(),
            "--output-root",
            (report_dir / "batch_outputs").as_posix(),
            "--run-name",
            args.run_name,
            "--jobs",
            str(args.jobs),
            "--arrange-inputs",
            args.arrange_inputs,
            "--manifest",
            (report_dir / "batch_manifest.json").as_posix(),
            "--log-dir",
            (report_dir / "batch_logs").as_posix(),
            "--with-validation",
            "--targets",
            "validation",
        ]
        if args.execute:
            batch_command.append("--execute")
        _append_dependency_config(batch_command, args.dependency_config)
        batch_command.extend(forwarded_args)
        steps.append(ValidationStep("batch_validation", tuple(batch_command)))
    else:
        steps.append(ValidationStep("batch_validation", (), required=False, skipped_reason="--batch-root was not provided"))

    steps.append(
        ValidationStep(
            "output_tree_html",
            (
                sys.executable,
                (REPO_ROOT / "Testing" / "Python" / "generate_output_tree_report.py").as_posix(),
                "--output-root",
                output_root.as_posix(),
                "--dest",
                (report_dir / "output_tree.html").as_posix(),
                "--subject-id",
                subject_id,
                "--run-name",
                args.run_name,
            ),
            required=False,
        )
    )
    steps.append(
        ValidationStep(
            "representative_outputs",
            (
                sys.executable,
                (REPO_ROOT / "scripts" / "collect_representative_outputs.py").as_posix(),
                "--output-root",
                output_root.as_posix(),
                "--dest-dir",
                (report_dir / "representative_outputs").as_posix(),
            ),
            required=False,
        )
    )
    steps.append(
        ValidationStep(
            "repository_release_audit",
            (
                sys.executable,
                (REPO_ROOT / "scripts" / "release_audit.py").as_posix(),
                "--json",
                "--warn-only",
            ),
            required=False,
        )
    )

    if args.skip_asset_checks:
        steps.append(
            ValidationStep(
                "model_assets_core_verify",
                (),
                required=False,
                skipped_reason="--skip-asset-checks was provided",
            )
        )
        steps.append(
            ValidationStep(
                "model_assets_all_verify",
                (),
                required=False,
                skipped_reason="--skip-asset-checks was provided",
            )
        )
    else:
        steps.append(
            ValidationStep(
                "model_assets_core_verify",
                (
                    sys.executable,
                    (REPO_ROOT / "scripts" / "download_model_assets.py").as_posix(),
                    "--profile",
                    "core",
                    "--verify-only",
                ),
                required=False,
            )
        )
        steps.append(
            ValidationStep(
                "model_assets_all_verify",
                (
                    sys.executable,
                    (REPO_ROOT / "scripts" / "download_model_assets.py").as_posix(),
                    "--profile",
                    "all",
                    "--verify-only",
                ),
                required=False,
            )
        )

    for name, mirror_dir, profile, flag_name in (
        ("asset_mirror_core_verify", args.asset_mirror_dir, "core", "--asset-mirror-dir"),
        ("asset_mirror_all_verify", args.all_asset_mirror_dir, "all", "--all-asset-mirror-dir"),
    ):
        if mirror_dir is None:
            steps.append(
                ValidationStep(
                    name,
                    (),
                    required=False,
                    skipped_reason=f"{flag_name} was not provided",
                )
            )
        else:
            steps.append(
                ValidationStep(
                    name,
                    (
                        sys.executable,
                        (REPO_ROOT / "scripts" / "download_model_assets.py").as_posix(),
                        "--profile",
                        profile,
                        "--manifest",
                        (REPO_ROOT / "models" / "assets_manifest.json").as_posix(),
                        "--repo-root",
                        REPO_ROOT.as_posix(),
                        "--asset-root",
                        mirror_dir.expanduser().resolve().as_posix(),
                        "--verify-only",
                    ),
                    required=False,
                )
            )

    if args.run_fresh_clone_smoke:
        steps.append(
            ValidationStep(
                "fresh_clone_smoke",
                (
                    "env",
                    f"SMOKE_ASSET_PROFILE={args.fresh_clone_asset_profile}",
                    "SMOKE_ASSET_MODE=download",
                    f"SMOKE_ROOT={(report_dir / 'fresh_clone_smoke').as_posix()}",
                    "bash",
                    (REPO_ROOT / "scripts" / "smoke_fresh_clone.sh").as_posix(),
                ),
                required=False,
            )
        )
    else:
        steps.append(
            ValidationStep(
                "fresh_clone_smoke",
                (),
                required=False,
                skipped_reason="--run-fresh-clone-smoke was not provided",
            )
        )

    if args.skip_slicer_smoke:
        for name in (
            "slicer_extension_install_smoke",
            "slicer_extension_manager_install_smoke",
            "slicer_dependency_workflow_smoke",
            "slicer_demo_workflow_smoke",
            "slicer_environment_button_smoke",
            "slicer_run_plan_panel_smoke",
            "slicer_output_tree_button_smoke",
            "slicer_cancel_background_smoke",
        ):
            steps.append(
                ValidationStep(
                    name,
                    (),
                    required=False,
                    skipped_reason="--skip-slicer-smoke was provided",
                )
            )
    else:
        install_root = report_dir / "slicer_extension_install"
        module_dir = install_root / "SlicerDSpace" / "lib" / "Slicer-5.10" / "qt-scripted-modules"
        cmake_executable = os.environ.get("CMAKE_EXECUTABLE", "").strip() or shutil.which("cmake")
        if cmake_executable and Path(cmake_executable).is_file():
            steps.append(
                ValidationStep(
                    "cmake_extension_config_contract",
                    (
                        "env",
                        f"CMAKE_EXECUTABLE={Path(cmake_executable).expanduser().resolve().as_posix()}",
                        sys.executable,
                        "-m",
                        "unittest",
                        "Testing.Python.test_cmake_extension_config",
                    ),
                    required=False,
                )
            )
        else:
            steps.append(
                ValidationStep(
                    "cmake_extension_config_contract",
                    (),
                    required=False,
                    skipped_reason="CMake was not found; set CMAKE_EXECUTABLE to enable the configure contract",
                )
            )
        steps.append(
            ValidationStep(
                "stage_slicer_extension",
                (
                    sys.executable,
                    (REPO_ROOT / "scripts" / "stage_slicer_extension.py").as_posix(),
                    "--install-root",
                    install_root.as_posix(),
                    "--clean",
                    "--link-mode",
                    "copy",
                ),
                required=False,
            )
        )
        extension_root = install_root / "SlicerDSpace"
        extension_archive = report_dir / "packages" / "SlicerDSpace.tar.gz"
        steps.append(
            ValidationStep(
                "package_slicer_extension",
                (
                    sys.executable,
                    (REPO_ROOT / "scripts" / "package_slicer_extension.py").as_posix(),
                    "--extension-root",
                    extension_root.as_posix(),
                    "--output",
                    extension_archive.as_posix(),
                ),
                required=False,
            )
        )
        if args.slicer_launcher and args.xvfb_run and args.slicer_launcher.is_file() and args.xvfb_run.is_file():
            for name, script_name in (
                ("slicer_extension_install_smoke", "slicer_extension_install_smoke.py"),
                ("slicer_extension_manager_install_smoke", "slicer_extension_manager_install_smoke.py"),
                ("slicer_dependency_workflow_smoke", "slicer_dependency_workflow_smoke.py"),
                ("slicer_demo_workflow_smoke", "slicer_demo_workflow_smoke.py"),
                ("slicer_environment_button_smoke", "slicer_environment_button_smoke.py"),
                ("slicer_run_plan_panel_smoke", "slicer_run_plan_panel_smoke.py"),
                ("slicer_output_tree_button_smoke", "slicer_output_tree_button_smoke.py"),
                ("slicer_cancel_background_smoke", "slicer_cancel_background_smoke.py"),
            ):
                steps.append(
                    ValidationStep(
                        name,
                        _slicer_smoke_command(args=args, module_dir=module_dir, script_name=script_name),
                        required=False,
                    )
                )
        else:
            for name in (
                "slicer_extension_install_smoke",
                "slicer_extension_manager_install_smoke",
                "slicer_dependency_workflow_smoke",
                "slicer_demo_workflow_smoke",
                "slicer_environment_button_smoke",
                "slicer_run_plan_panel_smoke",
                "slicer_output_tree_button_smoke",
                "slicer_cancel_background_smoke",
            ):
                steps.append(
                    ValidationStep(
                        name,
                        (),
                        required=False,
                        skipped_reason="Slicer launcher or xvfb-run was not found",
                    )
                )
    return steps


def _run_step(step: ValidationStep, report_dir: Path, *, timeout_seconds: int) -> dict[str, Any]:
    started = time.time()
    if step.skipped:
        return {
            "name": step.name,
            "status": "skipped",
            "required": step.required,
            "reason": step.skipped_reason,
            "command": list(step.command),
            "elapsed_seconds": 0.0,
        }
    stdout_path = report_dir / "logs" / f"{step.name}.stdout.log"
    stderr_path = report_dir / "logs" / f"{step.name}.stderr.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(
            list(step.command),
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
        stdout_path.write_text(_text_output(completed.stdout), encoding="utf-8")
        stderr_path.write_text(_text_output(completed.stderr), encoding="utf-8")
        status = "passed" if completed.returncode == 0 else "failed"
        returncode = completed.returncode
        error = ""
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(_text_output(exc.stdout), encoding="utf-8")
        stderr_path.write_text(_text_output(exc.stderr), encoding="utf-8")
        status = "failed"
        returncode = None
        error = f"timed out after {timeout_seconds} seconds"
    elapsed = round(time.time() - started, 3)
    return {
        "name": step.name,
        "status": status,
        "required": step.required,
        "command": list(step.command),
        "returncode": returncode,
        "stdout_log": stdout_path.as_posix(),
        "stderr_log": stderr_path.as_posix(),
        "elapsed_seconds": elapsed,
        "error": error,
    }


def _text_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _result_map(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item.get("name")): item for item in results}


def _coverage_status(names: list[str], results_by_name: dict[str, dict[str, Any]]) -> str:
    statuses = [str(results_by_name.get(name, {}).get("status") or "missing") for name in names]
    if all(status == "passed" for status in statuses):
        return "passed"
    if any(status == "failed" for status in statuses):
        return "failed"
    if any(status == "skipped" for status in statuses) and all(status in {"skipped", "missing"} for status in statuses):
        return "skipped"
    if any(status == "missing" for status in statuses):
        return "missing"
    return "skipped"


def _coverage_item(
    *,
    requirement: str,
    step_names: list[str],
    results_by_name: dict[str, dict[str, Any]],
    notes: str = "",
) -> dict[str, Any]:
    return {
        "requirement": requirement,
        "status": _coverage_status(step_names, results_by_name),
        "steps": [
            {
                "name": name,
                "status": str(results_by_name.get(name, {}).get("status") or "missing"),
                "required": bool(results_by_name.get(name, {}).get("required", False)),
                "reason": str(results_by_name.get(name, {}).get("reason") or ""),
                "stdout_log": results_by_name.get(name, {}).get("stdout_log"),
                "stderr_log": results_by_name.get(name, {}).get("stderr_log"),
            }
            for name in step_names
        ],
        "notes": notes,
    }


def build_completion_audit(args: argparse.Namespace, results: list[dict[str, Any]]) -> dict[str, Any]:
    results_by_name = _result_map(results)
    coverage = [
        _coverage_item(
            requirement="local_release_regression",
            step_names=[
                "repository_release_audit",
                "model_assets_core_verify",
                "model_assets_all_verify",
                "asset_mirror_core_verify",
                "asset_mirror_all_verify",
            ],
            results_by_name=results_by_name,
            notes="Covers release audit, core/all model checksums and core/all asset mirror verification.",
        ),
        _coverage_item(
            requirement="slicer_gui_smoke",
            step_names=[
                "stage_slicer_extension",
                "cmake_extension_config_contract",
                "package_slicer_extension",
                "slicer_extension_install_smoke",
                "slicer_extension_manager_install_smoke",
                "slicer_dependency_workflow_smoke",
                "slicer_demo_workflow_smoke",
                "slicer_environment_button_smoke",
                "slicer_run_plan_panel_smoke",
                "slicer_output_tree_button_smoke",
                "slicer_cancel_background_smoke",
            ],
            results_by_name=results_by_name,
            notes=(
                "Covers staged discovery, rooted package creation, real local Extension Manager installation, "
                "dependency setup persistence, the one-click demo, "
                "environment dependency dialog, run-plan panel, output-tree button and background cancellation "
                "when Slicer and xvfb-run paths are provided."
            ),
        ),
        _coverage_item(
            requirement="scientific_workflow_smoke",
            step_names=[
                "single_subject_all_backends",
                "bundle_backend_wmql",
                "bundle_backend_wma",
                "bundle_backend_tractcloud",
                "output_tree_html",
                "representative_outputs",
            ],
            results_by_name=results_by_name,
            notes=(
                "Plans all main targets by default; use --execute for real product execution. "
                "Dedicated DTI, CSD/ACT, WMA, connectome, microstructure and surface smoke scripts remain separate."
            ),
        ),
        _coverage_item(
            requirement="fresh_clone_release_smoke",
            step_names=["fresh_clone_smoke"],
            results_by_name=results_by_name,
            notes="Simulates GitHub Release checkout with LFS smudge disabled and explicit model asset download.",
        ),
    ]
    return {
        "kind": "slicerdsurfer_release_completion_audit",
        "version": 1,
        "execute": bool(args.execute),
        "overall_status": (
            "failed"
            if any(item["status"] == "failed" for item in coverage)
            else "incomplete"
            if any(item["status"] in {"skipped", "missing"} for item in coverage)
            else "passed"
        ),
        "coverage": coverage,
        "code_and_model_asset_boundary": {
            "status": "documented",
            "code_checkout": "Git checkout is validated separately from model assets; release smoke disables LFS smudge.",
            "model_asset_download": "Model files are retrieved by scripts/download_model_assets.py using explicit profiles and checksums.",
            "fresh_clone_asset_profile": str(getattr(args, "fresh_clone_asset_profile", "core")),
        },
    }


def _write_report(args: argparse.Namespace, steps: list[ValidationStep], results: list[dict[str, Any]], report_dir: Path) -> Path:
    failed_required = [item["name"] for item in results if item["required"] and item["status"] == "failed"]
    failed_optional = [item["name"] for item in results if not item["required"] and item["status"] == "failed"]
    payload = {
        "kind": "slicerdsurfer_release_validation",
        "version": 1,
        "schema": "slicerdsurfer.release_validation.v1",
        "created_at_unix": int(time.time()),
        "execute": bool(args.execute),
        "subject_root": _path(args.subject_root),
        "subject_id": _subject_id(args.subject_root, args.subject_id),
        "output_root": _path(args.output_root) or (args.subject_root.expanduser().resolve() / "outputs").as_posix(),
        "batch_root": _path(args.batch_root),
        "run_name": args.run_name,
        "dependency_config": _path(args.dependency_config),
        "code_and_model_asset_boundary": {
            "code_checkout": "Use normal Git clone with LFS smudge disabled for release smoke.",
            "model_asset_download": (
                "Run scripts/download_model_assets.py after code checkout using "
                "an explicit profile and checksum verification."
            ),
            "fresh_clone_smoke": (
                "scripts/smoke_fresh_clone.sh records git clone/checkout "
                "separately from model asset download/verify steps."
            ),
        },
        "asset_mirrors": {
            "core": _path(args.asset_mirror_dir),
            "all": _path(args.all_asset_mirror_dir),
        },
        "summary": {
            "status": "failed" if failed_required else "passed_with_optional_failures" if failed_optional else "passed",
            "step_count": len(results),
            "failed_required": failed_required,
            "failed_optional": failed_optional,
            "skipped": [item["name"] for item in results if item["status"] == "skipped"],
        },
        "completion_audit": build_completion_audit(args, results),
        "planned_steps": [
            {
                "name": step.name,
                "required": step.required,
                "command": list(step.command),
                "skipped_reason": step.skipped_reason,
            }
            for step in steps
        ],
        "results": results,
    }
    report_path = report_dir / "release_validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subject-root",
        type=Path,
        default=_env_path("SLICERDSPACE_RELEASE_SUBJECT_ROOT"),
        help="Subject-root fixture for release validation. Can also be set with SLICERDSPACE_RELEASE_SUBJECT_ROOT.",
    )
    parser.add_argument("--subject-id")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--batch-root", type=Path)
    parser.add_argument("--report-dir", type=Path, help="Defaults to <output-root>/00_run_metadata/release_validation.")
    parser.add_argument("--run-name", default="release_validation")
    parser.add_argument("--execute", action="store_true", help="Execute scientific workflows instead of planning them.")
    parser.add_argument("--dependency-config", type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--arrange-inputs", choices=("none", "symlink", "copy"), default="none")
    parser.add_argument("--tractography-backend", default="csd_act", choices=("csd_act", "ukf", "dti_det", "saf"))
    parser.add_argument("--connectome-backend", default="python", choices=("auto", "python", "mrtrix"))
    parser.add_argument("--skip-slicer-smoke", action="store_true")
    parser.add_argument("--skip-asset-checks", action="store_true")
    parser.add_argument("--asset-mirror-dir", type=Path, help="Optional core-profile asset mirror directory to verify.")
    parser.add_argument(
        "--all-asset-mirror-dir",
        type=Path,
        help="Optional all-profile asset mirror directory to verify.",
    )
    parser.add_argument(
        "--run-fresh-clone-smoke",
        action="store_true",
        help="Run scripts/smoke_fresh_clone.sh as an optional release-validation step.",
    )
    parser.add_argument("--fresh-clone-asset-profile", default="core", help="Asset profile used by --run-fresh-clone-smoke.")
    parser.add_argument(
        "--slicer-launcher",
        type=Path,
        default=_env_path("SLICERDSPACE_SLICER_LAUNCHER"),
        help="Optional Slicer launcher for GUI smoke. Can also be set with SLICERDSPACE_SLICER_LAUNCHER.",
    )
    parser.add_argument(
        "--xvfb-run",
        type=Path,
        default=_env_path("SLICERDSPACE_XVFB_RUN"),
        help="Optional xvfb-run for headless GUI smoke. Can also be set with SLICERDSPACE_XVFB_RUN.",
    )
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--warn-only", action="store_true", help="Always exit 0 after writing the report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, forwarded_args = parser.parse_known_args(argv)
    if args.subject_root is None:
        parser.error("Release validation requires --subject-root or SLICERDSPACE_RELEASE_SUBJECT_ROOT.")
    subject_root = args.subject_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve() if args.output_root else subject_root / "outputs"
    if args.report_dir is None:
        args.report_dir = output_root / "00_run_metadata" / "release_validation"
    report_dir = args.report_dir.expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    steps = build_validation_steps(args, list(forwarded_args))
    results = [_run_step(step, report_dir, timeout_seconds=args.timeout_seconds) for step in steps]
    report_path = _write_report(args, steps, results, report_dir)
    print(report_path.as_posix())

    failed_required = any(item["required"] and item["status"] == "failed" for item in results)
    return 0 if args.warn_only or not failed_required else 2


if __name__ == "__main__":
    raise SystemExit(main())
