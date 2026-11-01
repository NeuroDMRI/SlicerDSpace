#!/usr/bin/env python3
"""Audit Slicer extension metadata that tends to drift before release."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_EXTENSION_DEPENDS = ("SlicerDMRI", "PyTorch")
REQUIRED_TOP_LEVEL_FILES = (
    "LICENSE.txt",
    "CITATION.cff",
    "codemeta.json",
    ".zenodo.json",
    "docs/assets/MODEL_ASSETS.md",
    "docs/release/RELEASE.md",
    "docs/release/extension-index/SlicerDSpace.json",
)
REQUIRED_HELP_TERMS = (
    "diffusion",
    "surface",
    "parcellation",
    "segmentation",
    "microstructure",
    "tractography",
    "connectome",
)
REQUIRED_README_TERMS = (
    "docs/assets/MODEL_ASSETS.md",
    "scripts/download_model_assets.sh --profile core",
    "SlicerDMRI",
    "PyTorch",
)


@dataclass(frozen=True)
class AuditFinding:
    key: str
    severity: str
    message: str


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def parse_s4ext(path: Path) -> dict[str, list[str]]:
    fields: dict[str, list[str]] = {}
    for raw_line in _read(path).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 1:
            fields.setdefault(parts[0], []).append("")
        else:
            fields.setdefault(parts[0], []).append(parts[1].strip())
    return fields


def parse_cmake_extension_depends(path: Path) -> list[str]:
    text = _read(path)
    match = re.search(r"set\s*\(\s*EXTENSION_DEPENDS\s+([^)]+)\)", text, re.IGNORECASE | re.MULTILINE)
    if not match:
        return []
    return [part for part in re.split(r"\s+", match.group(1).strip()) if part]


def parse_module_metadata(path: Path) -> dict[str, Any]:
    text = _read(path)
    metadata: dict[str, Any] = {}
    simple_fields = {
        "title": r"self\.parent\.title\s*=\s*['\"]([^'\"]+)['\"]",
        "categories": r"self\.parent\.categories\s*=\s*\[([^\]]*)\]",
        "dependencies": r"self\.parent\.dependencies\s*=\s*\[([^\]]*)\]",
        "contributors": r"self\.parent\.contributors\s*=\s*\[([^\]]*)\]",
    }
    for key, pattern in simple_fields.items():
        match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
        metadata[key] = match.group(1).strip() if match else ""

    for key in ("helpText", "acknowledgementText"):
        match = re.search(
            rf"self\.parent\.{key}\s*=\s*\((.*?)\n\s*\)",
            text,
            re.MULTILINE | re.DOTALL,
        )
        if not match:
            match = re.search(rf"self\.parent\.{key}\s*=\s*['\"]([^'\"]+)['\"]", text)
        metadata[key] = match.group(1).strip() if match else ""
    return metadata


def audit(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    findings: list[AuditFinding] = []
    s4ext_path = repo_root / "SlicerDSpace.s4ext"
    cmake_path = repo_root / "CMakeLists.txt"
    module_path = repo_root / "SlicerDSpace.py"
    readme_path = repo_root / "README.md"

    for relative in REQUIRED_TOP_LEVEL_FILES:
        if not (repo_root / relative).exists():
            findings.append(AuditFinding(relative, "error", f"Missing release metadata file: {relative}"))

    s4ext = parse_s4ext(s4ext_path)
    if (repo_root / "LICENSE").exists():
        findings.append(AuditFinding("license_duplicate", "error", "Use LICENSE.txt as the single root project license"))
    s4ext_depends = tuple(s4ext.get("depends", [""])[0].split())
    cmake_depends = tuple(parse_cmake_extension_depends(cmake_path))
    if s4ext_depends != cmake_depends:
        findings.append(
            AuditFinding(
                "extension_dependencies",
                "error",
                f"SlicerDSpace.s4ext depends {s4ext_depends!r} does not match CMake EXTENSION_DEPENDS {cmake_depends!r}",
            )
        )
    for dependency in REQUIRED_EXTENSION_DEPENDS:
        if dependency not in s4ext_depends:
            findings.append(AuditFinding("extension_dependencies", "error", f"Missing extension dependency: {dependency}"))

    for key in ("scm", "scmurl", "scmrevision", "homepage", "category", "contributors", "description", "enabled", "iconurl", "screenshoturls"):
        if not s4ext.get(key, [""])[0]:
            findings.append(AuditFinding(f"s4ext:{key}", "error", f"Missing .s4ext field: {key}"))

    catalog_path = repo_root / "docs/release/extension-index/SlicerDSpace.json"
    if catalog_path.is_file():
        try:
            catalog = json.loads(_read(catalog_path))
            allowed = {"$schema", "category", "scm_url", "scm_revision", "scm_type",
                       "build_dependencies", "build_subdirectory", "enabled", "tier", "dicom_support_rule"}
            if not isinstance(catalog, dict) or set(catalog) - allowed:
                raise ValueError("Unexpected catalog fields")
            if catalog.get("$schema") != "https://raw.githubusercontent.com/Slicer/Slicer/main/Schemas/slicer-extension-catalog-entry-schema-v1.0.2.json#":
                raise ValueError("Unexpected catalog schema")
            for key, legacy in (("scm_url", "scmurl"), ("scm_revision", "scmrevision"),
                                ("category", "category"), ("scm_type", "scm"),
                                ("build_subdirectory", "build_subdirectory")):
                if not isinstance(catalog.get(key), str) or catalog[key] != s4ext.get(legacy, [""])[0]:
                    raise ValueError(f"Catalog/s4ext mismatch: {key}")
            if catalog.get("build_dependencies") != list(cmake_depends):
                raise ValueError("Catalog/CMake dependency mismatch")
            if catalog.get("tier") != 1 or type(catalog.get("tier")) is not int:
                raise ValueError("Draft should propose Tier 1 only")
            if catalog.get("enabled") is not True:
                raise ValueError("Catalog enabled must be boolean true")
            if "dicom_support_rule" in catalog and not isinstance(catalog["dicom_support_rule"], str):
                raise ValueError("DICOM rule must be a string")
        except (ValueError, TypeError) as exc:
            findings.append(AuditFinding("catalog", "error", str(exc)))

    cmake_text = _read(cmake_path)
    for key, legacy in (("HOMEPAGE", "homepage"), ("CATEGORY", "category"),
                        ("CONTRIBUTORS", "contributors"), ("DESCRIPTION", "description"),
                        ("ICONURL", "iconurl"), ("SCREENSHOTURLS", "screenshoturls")):
        match = re.search(rf'set\(EXTENSION_{key}\s+"([^"]*)"\)', cmake_text)
        if not match or match.group(1) != s4ext.get(legacy, [""])[0]:
            findings.append(AuditFinding("cmake:" + key, "error", f"CMake/s4ext mismatch: {key}"))
    for key in ("iconurl", "screenshoturls"):
        for url in s4ext.get(key, [""])[0].split():
            prefix = "https://raw.githubusercontent.com/NeuroDMRI/SlicerDSpace/main/"
            if not url.startswith(prefix) or not (repo_root / url[len(prefix):]).is_file():
                findings.append(AuditFinding(key, "error", f"Raw URL has no matching source image: {url}"))

    metadata = parse_module_metadata(module_path)
    if metadata.get("title") != "SlicerDSpace":
        findings.append(AuditFinding("module:title", "error", "Module title should be SlicerDSpace"))
    if "Diffusion" not in metadata.get("categories", ""):
        findings.append(AuditFinding("module:categories", "error", "Module should appear under the Diffusion category"))
    if metadata.get("dependencies", "").strip():
        findings.append(
            AuditFinding(
                "module:dependencies",
                "error",
                "Runtime module dependencies should stay empty; dependency doctor reports optional blockers after UI load",
            )
        )
    help_text = metadata.get("helpText", "").lower()
    missing_help_terms = [term for term in REQUIRED_HELP_TERMS if term not in help_text]
    if missing_help_terms:
        findings.append(
            AuditFinding(
                "module:helpText",
                "error",
                f"Module help text is missing workflow terms: {', '.join(missing_help_terms)}",
            )
        )
    if not metadata.get("contributors"):
        findings.append(AuditFinding("module:contributors", "error", "Module contributors are missing"))
    if not metadata.get("acknowledgementText"):
        findings.append(AuditFinding("module:acknowledgementText", "error", "Module acknowledgement text is missing"))

    readme = _read(readme_path)
    missing_readme_terms = [term for term in REQUIRED_README_TERMS if term not in readme]
    if missing_readme_terms:
        findings.append(
            AuditFinding(
                "readme:release_paths",
                "error",
                f"README is missing release/setup terms: {', '.join(missing_readme_terms)}",
            )
        )

    return {
        "status": "pass" if not findings else "fail",
        "error_count": sum(1 for finding in findings if finding.severity == "error"),
        "warning_count": sum(1 for finding in findings if finding.severity == "warning"),
        "s4ext_depends": list(s4ext_depends),
        "cmake_depends": list(cmake_depends),
        "findings": [finding.__dict__ for finding in findings],
        "scope": "Local metadata consistency; not public URL reachability or catalog acceptance",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Write machine-readable audit output.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="Repository root to audit.")
    args = parser.parse_args(argv)

    result = audit(args.repo_root.resolve())
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"Slicer extension metadata audit: {result['status']}")
        for finding in result["findings"]:
            print(f"[{finding['severity']}] {finding['key']}: {finding['message']}")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
