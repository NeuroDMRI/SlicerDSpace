#!/usr/bin/env python3
"""Repository hygiene/packaging audit; not a certification of public release readiness."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from scripts.extension_runtime.runtime_resources import runtime_files
except ModuleNotFoundError:
    from extension_runtime.runtime_resources import runtime_files


REPO_ROOT = Path(__file__).resolve().parents[1]
# The public Pages film is intentionally ordinary Git content, not a model asset
# or an LFS pointer that a static Pages server could not play. No blanket media exemption.
PUBLIC_FILM = "docs/site/media/SlicerDSpace_promo_1080p.mp4"
PUBLIC_FILM_BYTES = 23356794
PUBLIC_FILM_SHA256 = "0531f24d91f8abedc278899b43649eee3469fcdd71b762e6d6cdb5afe204abb7"
LFS_REQUIRED_SUFFIXES = (
    ".pt",
    ".pkl",
    ".nii.gz",
    ".mgz",
    ".tck",
    ".trk",
    ".vtk",
    ".vtp",
    ".obj",
    ".stl",
)
FORBIDDEN_TRACKED_PREFIXES = (
    "third_party/",
    "SlicerDSpace-test/",
    "outputs/",
)
FORBIDDEN_STATUS_PARTS = (
    "/outputs/",
    "/gui_smoke/",
    "/Testing/Temporary/",
)


@dataclass(frozen=True)
class AuditIssue:
    severity: str
    category: str
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "category": self.category,
            "path": self.path,
            "message": self.message,
        }


def _run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", REPO_ROOT.as_posix(), *args],
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _git_lines(args: list[str]) -> list[str]:
    completed = _run_git(args)
    if completed.returncode != 0:
        return []
    return [line for line in completed.stdout.splitlines() if line]


def _git_stdout(args: list[str]) -> str:
    completed = _run_git(args)
    return completed.stdout if completed.returncode == 0 else ""


def _status_entries() -> list[dict[str, str]]:
    entries = []
    for line in _git_lines(["status", "--porcelain=v1"]):
        status = line[:2]
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        entries.append({"status": status, "path": path})
    return entries


def _tracked_files() -> list[str]:
    return _git_lines(["ls-files"])


def _untracked_files() -> list[str]:
    return [entry["path"] for entry in _status_entries() if entry["status"] == "??"]


def _file_size(path: str) -> int:
    try:
        return (REPO_ROOT / path).stat().st_size
    except OSError:
        return 0


def _verified_public_film(path: str) -> bool:
    if path != PUBLIC_FILM or _file_size(path) != PUBLIC_FILM_BYTES:
        return False
    digest = hashlib.sha256()
    with (REPO_ROOT / path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest() == PUBLIC_FILM_SHA256


def _index_blob_size(path: str) -> int | None:
    completed = _run_git(["cat-file", "-s", f":{path}"])
    if completed.returncode != 0:
        return None
    try:
        return int(completed.stdout.strip())
    except ValueError:
        return None


def _index_blob_is_lfs_pointer(path: str) -> bool:
    size = _index_blob_size(path)
    if size is None or size > 1024:
        return False
    return _git_stdout(["show", f":{path}"]).startswith("version https://git-lfs.github.com/spec/v1\n")


def _path_uses_lfs_filter(path: str) -> bool:
    return ": filter: lfs" in _git_stdout(["check-attr", "filter", "--", path])


def _matches_suffix(path: str, suffixes: Iterable[str]) -> bool:
    return any(path.endswith(suffix) for suffix in suffixes)


def _read_lfs_patterns() -> set[str]:
    attributes = REPO_ROOT / ".gitattributes"
    patterns: set[str] = set()
    if not attributes.is_file():
        return patterns
    for line in attributes.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        token = line.split()[0]
        if "filter=lfs" in line:
            patterns.add(token)
    return patterns


def _read_text_if_exists(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _extract_cmake_set_items(text: str, variable: str) -> list[str]:
    # This inspects literal set lists, not arbitrary CMake expressions.
    text = re.sub(r"#[^\n]*", "", text)
    match = re.search(r"\bset\s*\(\s*" + re.escape(variable) + r"\s+([^)]*)\)", text)
    if not match:
        return []
    return [item.strip('"') for item in re.findall(r'"[^"\n]*"|\S+', match.group(1))]


def _runtime_packaging_issues(root: Path, cmake_text: str) -> list[AuditIssue]:
    issues = []
    required_cmake_entries = (
        'include("${CMAKE_CURRENT_SOURCE_DIR}/scripts/extension_runtime/RuntimeResources.cmake")',
        'slicerdsurfer_runtime_resources("${CMAKE_CURRENT_SOURCE_DIR}" MODULE_PYTHON_RESOURCES)',
        'list(REMOVE_ITEM MODULE_PYTHON_RESOURCES "${MODULE_NAME}.py")',
        'RESOURCES ${MODULE_PYTHON_RESOURCES}',
    )
    for entry in required_cmake_entries:
        if entry not in cmake_text:
            issues.append(AuditIssue("blocker", "cmake_packaging", "CMakeLists.txt",
                                     f"CMake must consume the shared runtime selection: {entry}"))
    try:
        selected = runtime_files(root)
    except (OSError, ValueError, KeyError, TypeError) as error:
        issues.append(AuditIssue("blocker", "cmake_packaging", "scripts/extension_runtime/runtime_resources.json",
                                 f"Runtime resource selection failed: {error}"))
        return issues
    for relative in ("tractography/wmql_queries/AF_CST.qry", "scripts/wsl/WslCommandBridge.cs",
                     "scripts/wsl/requirements-runtime.txt", "LICENSE.txt",
                     "anatomy/parcellation/conversion/data/FreeSurferColorLUT.txt"):
        if relative not in selected:
            issues.append(AuditIssue("blocker", "cmake_packaging", relative,
                                     "Required non-Python runtime resource is absent from the shared selection."))
    for relative in selected:
        parts = Path(relative).parts
        if "slicer_dmri_reference" in parts or "third_party" in parts:
            issues.append(AuditIssue("blocker", "cmake_packaging", relative,
                                     "Reference-only or third-party source must not be packaged."))
        if len(parts) == 1 and relative.endswith(".py") and relative != "SlicerDSpace.py":
            issues.append(AuditIssue("blocker", "cmake_packaging", relative,
                                     "Top-level Python helpers can be scanned as Slicer modules."))
    return issues


def audit_repository(*, large_file_mb: float = 20.0) -> dict[str, object]:
    issues: list[AuditIssue] = []
    status = _status_entries()
    tracked = _tracked_files()
    untracked = _untracked_files()
    large_threshold = int(large_file_mb * 1024 * 1024)

    if not (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines().__contains__("third_party/"):
        issues.append(
            AuditIssue(
                severity="blocker",
                category="gitignore",
                path=".gitignore",
                message="third_party/ is not ignored.",
            )
        )

    lfs_patterns = _read_lfs_patterns()
    for suffix in LFS_REQUIRED_SUFFIXES:
        if f"*{suffix}" not in lfs_patterns:
            issues.append(
                AuditIssue(
                    severity="blocker",
                    category="lfs",
                    path=".gitattributes",
                    message=f"Missing Git LFS rule for *{suffix}.",
                )
            )

    cmake_text = _read_text_if_exists(REPO_ROOT / "CMakeLists.txt")
    if not cmake_text:
        issues.append(
            AuditIssue(
                severity="blocker",
                category="cmake_packaging",
                path="CMakeLists.txt",
                message="Top-level CMakeLists.txt is missing or unreadable.",
            )
        )
    else:
        scripted_entries = _extract_cmake_set_items(cmake_text, "MODULE_PYTHON_SCRIPTS")
        if scripted_entries != ["${MODULE_NAME}.py"]:
            issues.append(
                AuditIssue(
                    severity="blocker",
                    category="cmake_packaging",
                    path="CMakeLists.txt",
                    message=(
                        "MODULE_PYTHON_SCRIPTS must contain only ${MODULE_NAME}.py. "
                        f"Found: {scripted_entries or 'none'}"
                    ),
                )
            )
        resource_entries = set(_extract_cmake_set_items(cmake_text, "MODULE_PYTHON_RESOURCES"))
        forbidden_top_level_python_resources = {"run_subject_pipeline.py", "workflow_catalog.py"}
        leaked_resources = sorted(forbidden_top_level_python_resources & resource_entries)
        if leaked_resources:
            issues.append(
                AuditIssue(
                    severity="blocker",
                    category="cmake_packaging",
                    path="CMakeLists.txt",
                    message=(
                        "Top-level Python helper resources can be scanned as Slicer modules. "
                        f"Move them under a package directory before packaging: {', '.join(leaked_resources)}."
                    ),
                )
            )
        if "third_party" in cmake_text:
            issues.append(
                AuditIssue(
                    severity="blocker",
                    category="cmake_packaging",
                    path="CMakeLists.txt",
                    message="CMakeLists.txt must not package or reference third_party content.",
                )
            )
        issues.extend(_runtime_packaging_issues(REPO_ROOT, cmake_text))
        if "SCRIPTS ${MODULE_PYTHON_SCRIPTS}" not in cmake_text:
            issues.append(
                AuditIssue(
                    severity="blocker",
                    category="cmake_packaging",
                    path="CMakeLists.txt",
                    message="slicerMacroBuildScriptedModule must use the audited MODULE_PYTHON_SCRIPTS list.",
                )
            )

    s4ext_text = _read_text_if_exists(REPO_ROOT / "SlicerDSpace.s4ext")
    if not s4ext_text:
        issues.append(
            AuditIssue(
                severity="blocker",
                category="extension_descriptor",
                path="SlicerDSpace.s4ext",
                message="Slicer extension descriptor is missing or unreadable.",
            )
        )
    else:
        required_s4ext_tokens = (
            "scm git",
            "scmurl ",
            "scmrevision ",
            "depends SlicerDMRI PyTorch",
            "build_subdirectory .",
            "category Diffusion",
            "enabled 1",
        )
        for token in required_s4ext_tokens:
            if token not in s4ext_text:
                issues.append(
                    AuditIssue(
                        severity="blocker",
                        category="extension_descriptor",
                        path="SlicerDSpace.s4ext",
                        message=f"Missing required .s4ext entry containing: {token}",
                    )
                )

    for path in tracked:
        if any(path.startswith(prefix) for prefix in FORBIDDEN_TRACKED_PREFIXES):
            issues.append(
                AuditIssue(
                    severity="blocker",
                    category="tracked_forbidden_path",
                    path=path,
                    message="Generated outputs or third_party content should not be tracked.",
                )
            )
        if _matches_suffix(path, LFS_REQUIRED_SUFFIXES) and _path_uses_lfs_filter(path) and not _index_blob_is_lfs_pointer(path):
            issues.append(
                AuditIssue(
                    severity="blocker",
                    category="lfs_pointer",
                    path=path,
                    message="Tracked file matches a Git LFS rule but the index stores a normal Git blob instead of an LFS pointer.",
                )
            )
        size = _file_size(path)
        if size >= large_threshold and not _matches_suffix(path, LFS_REQUIRED_SUFFIXES) and not _verified_public_film(path):
            issues.append(
                AuditIssue(
                    severity="blocker",
                    category="tracked_large_file",
                    path=path,
                    message=f"Tracked file is {size} bytes and is not covered by the expected LFS suffix list.",
                )
            )

    for path in untracked:
        if any(part in f"/{path}" for part in FORBIDDEN_STATUS_PARTS):
            issues.append(
                AuditIssue(
                    severity="blocker",
                    category="untracked_output",
                    path=path,
                    message="Untracked runtime output should not be committed.",
                )
            )
        size = _file_size(path)
        if size >= large_threshold:
            issues.append(
                AuditIssue(
                    severity="blocker",
                    category="untracked_large_file",
                    path=path,
                    message=f"Untracked file is {size} bytes; confirm it is ignored or stored with Git LFS before release.",
                )
            )

    dirty_source_changes = [
        entry
        for entry in status
        if entry["status"] != "??" and not any(part in f"/{entry['path']}" for part in FORBIDDEN_STATUS_PARTS)
    ]
    if dirty_source_changes:
        issues.append(
            AuditIssue(
                severity="warning",
                category="dirty_worktree",
                path=".",
                message=f"{len(dirty_source_changes)} tracked source file(s) have uncommitted changes.",
            )
        )

    blockers = [issue for issue in issues if issue.severity == "blocker"]
    warnings = [issue for issue in issues if issue.severity == "warning"]
    return {
        "kind": "slicerddsurfer_release_audit",
        "repo_root": REPO_ROOT.as_posix(),
        "status_entries": status,
        "tracked_file_count": len(tracked),
        "untracked_file_count": len(untracked),
        "large_file_threshold_mb": large_file_mb,
        "blocker_count": len(blockers),
        "warning_count": len(warnings),
        "issues": [issue.to_dict() for issue in issues],
        "scope": "repository hygiene and packaging declarations only",
        "repository_checks_passed": not blockers,
        "public_release_readiness": "NOT_ASSESSED",
        # Retained for existing CI consumers, with the limited scope above.
        "release_ready": not blockers,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--large-file-mb", type=float, default=20.0)
    parser.add_argument("--json", action="store_true", help="Print full JSON report.")
    parser.add_argument("--warn-only", action="store_true", help="Always exit 0 after printing the audit.")
    args = parser.parse_args(argv)

    report = audit_repository(large_file_mb=args.large_file_mb)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"repository_checks_passed={report['repository_checks_passed']}")
        print("public_release_readiness=NOT_ASSESSED (SDK, clean installation and asset permissions require separate evidence)")
        print(f"blockers={report['blocker_count']} warnings={report['warning_count']}")
        for issue in report["issues"]:
            print(f"- {issue['severity']} {issue['category']} {issue['path']}: {issue['message']}")
    if args.warn_only:
        return 0
    return 0 if report["release_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
