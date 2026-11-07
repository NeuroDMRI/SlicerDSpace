#!/usr/bin/env python3
"""Write Linux/macOS/Windows dependency profile dry-run reports."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.environment import EnvironmentDoctorConfig, run_environment_doctor, write_environment_report
from shared.preprocessing_route import ROUTE_ID, add_removed_flags


DEFAULT_PLATFORMS = ("Linux", "Darwin", "Windows")
PLATFORM_ALIASES = {
    "linux": "Linux",
    "darwin": "Darwin",
    "mac": "Darwin",
    "macos": "Darwin",
    "osx": "Darwin",
    "windows": "Windows",
    "win": "Windows",
}
DEFAULT_PROFILE_TARGETS = (
    "validation",
    "preprocessing",
    "segmentation",
    "parcellation",
    "surface",
    "surface_postprocess",
    "microstructure",
    "microstructure_freewater",
    "microstructure_noddi",
    "tractography",
    "bundles",
    "connectome",
    "saf",
    "wma",
    "tractcloud",
)


def _split_csv(values: Iterable[str] | None, *, lower: bool = False) -> tuple[str, ...]:
    if not values:
        return ()
    tokens: list[str] = []
    for value in values:
        for token in str(value).split(","):
            stripped = token.strip()
            if stripped:
                tokens.append(stripped.lower() if lower else stripped)
    return tuple(dict.fromkeys(tokens))


def _parse_platforms(values: Iterable[str] | None) -> tuple[str, ...]:
    requested = _split_csv(values) or DEFAULT_PLATFORMS
    platforms: list[str] = []
    for value in requested:
        normalized = PLATFORM_ALIASES.get(value.strip().lower())
        if normalized is None:
            raise ValueError(f"Unsupported platform {value!r}; use Linux, Darwin/macOS, or Windows.")
        platforms.append(normalized)
    return tuple(dict.fromkeys(platforms))


def _report_row(platform_name: str, result, written: dict[str, Path]) -> dict[str, object]:
    summary = result.to_dict()["summary"]
    missing_required = [
        {
            "key": item.key,
            "label": item.label,
            "kind": item.kind,
            "targets": list(item.targets),
            "detected_source": item.details.get("detected_source", "not found"),
            "install_hint": item.install_hint,
        }
        for item in result.statuses
        if item.required and item.status == "missing"
    ]
    host_platform = result.host_platform
    return {
        "platform": platform_name,
        "host_platform": host_platform,
        "native_host": platform_name == host_platform,
        "targets": list(result.targets),
        "summary": summary,
        "missing_required_preview": missing_required[:20],
        "missing_required_count": len(missing_required),
        "report_paths": {key: path.as_posix() for key, path in written.items()},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate dependency_profile.json and install notes for Linux, macOS, "
            "and Windows discovery rules. Non-native rows are dry-run profiles, "
            "not proof of a native OS installation."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "install_plan" / "dependency_profile_matrix",
        help="Directory where the matrix and per-platform reports are written.",
    )
    parser.add_argument(
        "--platforms",
        action="append",
        help="Platforms to profile. May be repeated or comma-separated. Defaults to Linux,Darwin,Windows.",
    )
    parser.add_argument(
        "--targets",
        action="append",
        help="Targets to profile. May be repeated or comma-separated. Defaults to the full dependency surface.",
    )
    parser.add_argument("--python-executable", default=sys.executable, help="Python executable to check for Python packages.")
    add_removed_flags(parser, ("--preprocessing-backend",))  # W2a (D1): preprocessing has one route
    parser.add_argument("--tractography-backend", default="csd_act", choices=("csd_act", "ukf", "dti_det", "saf"))
    parser.add_argument("--connectome-backend", default="mrtrix", choices=("auto", "python", "mrtrix"))
    parser.add_argument("--ddseg-device", default="auto")
    parser.add_argument("--surface-device", default="auto")
    parser.add_argument("--tractcloud-device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--strict-exit-code",
        action="store_true",
        help="Return exit code 2 if any profiled platform has missing required dependencies.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        platforms = _parse_platforms(args.platforms)
    except ValueError as exc:
        parser.error(str(exc))
    targets = _split_csv(args.targets, lower=True) or DEFAULT_PROFILE_TARGETS

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    python_executable = Path(args.python_executable).expanduser().resolve()
    rows: list[dict[str, object]] = []

    for platform_name in platforms:
        platform_dir = output_dir / platform_name.lower()
        config = EnvironmentDoctorConfig(
            targets=targets,
            repo_root=REPO_ROOT,
            output_dir=platform_dir,
            python_executable=python_executable,
            platform_name=platform_name,
            tractography_backend=args.tractography_backend,
            connectome_backend=args.connectome_backend,
            ddseg_device=args.ddseg_device,
            surface_device=args.surface_device,
            tractcloud_device=args.tractcloud_device,
        )
        result = run_environment_doctor(config)
        written = write_environment_report(result, platform_dir)
        rows.append(_report_row(platform_name, result, written))

    payload = {
        "kind": "slicerdsurfer_dependency_profile_matrix",
        "version": 1,
        "schema": "slicerddsurfer.dependency_profile_matrix.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host_platform": platform.system() or "unknown",
        "python_executable": python_executable.as_posix(),
        "targets": list(targets),
        "backend_profile": {
            "preprocessing": ROUTE_ID,
            "tractography": args.tractography_backend,
            "connectome": args.connectome_backend,
            "ddseg_device": args.ddseg_device,
            "surface_device": args.surface_device,
            "tractcloud_device": args.tractcloud_device,
        },
        "scan_policy": {
            "non_native_rows_are_dry_run": True,
            "paths_are_revalidated": True,
            "reason": (
                "The platform override switches discovery rules and install hints, "
                "but filesystem availability is still checked on the current host. "
                "Native Windows/macOS execution must be run on those hosts before "
                "claiming native installation success."
            ),
        },
        "rows": rows,
        "aggregate": {
            "platform_count": len(rows),
            "native_platform_count": sum(1 for row in rows if row["native_host"]),
            "platforms_with_blockers": [
                row["platform"]
                for row in rows
                if int(row["summary"]["blocker_count"]) > 0  # type: ignore[index]
            ],
        },
    }
    matrix_path = output_dir / "dependency_profile_matrix.json"
    matrix_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(matrix_path.as_posix())
    has_blockers = any(int(row["summary"]["blocker_count"]) > 0 for row in rows)  # type: ignore[index]
    return 2 if args.strict_exit_code and has_blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
