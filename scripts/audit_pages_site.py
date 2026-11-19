#!/usr/bin/env python3
"""Verify that the public Pages artifact cannot include private project documents."""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

try:
    from .prepare_website_results import validate_static_svg
except ImportError:
    from prepare_website_results import validate_static_svg

ROOT = Path(__file__).resolve().parents[1]
WORDMARKS = {"assets/img/slicerdspace-wordmark.svg", "assets/img/slicerdspace-wordmark-light.svg",
             "assets/img/fans-lab-wordmark.svg", "assets/img/fans-lab-wordmark-light.svg"}
RESULTS = "assets/img/results-overview.svg"
WORKFLOW = "assets/img/workflow-overview.svg"


def audit(root=ROOT):
    site = root / "docs/site"
    required = {".nojekyll", "index.html", "404.html"}
    allowed_top = required | {"zh.html", "assets/data/performance.json", RESULTS, WORKFLOW} | WORDMARKS
    allowed_extensions = {
        "assets/css": {".css"},
        "assets/js": {".js"},
        "assets/img": {".png", ".jpg", ".webp"},
        "assets/fonts": {".woff2", ".txt"},
        "media": {".mp4", ".srt"},
    }
    problems = []
    files, total = [], 0
    if not site.is_dir() or site.is_symlink():
        problems.append("docs/site must be a real, dedicated directory")
    for path in sorted(site.rglob("*")):
        relative = path.relative_to(site).as_posix()
        if path.is_symlink():
            problems.append("Symlinks are not public website artifacts: " + relative)
            continue
        if path.is_dir():
            continue
        parent = path.parent.relative_to(site).as_posix()
        if relative not in allowed_top and path.suffix.lower() not in allowed_extensions.get(parent, set()):
            problems.append("Unexpected public file: " + relative)
        if relative in (RESULTS, WORKFLOW):
            try:
                validate_static_svg(ET.fromstring(path.read_bytes()))
                receipt = json.loads((root / "docs/website/published_assets.json").read_text())
                expected = receipt["files"]["docs/site/" + relative]["sha256"]
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise ValueError("Results artwork differs from the reviewed receipt")
            except (ValueError, KeyError, OSError, ET.ParseError) as exc:
                problems.append("Unapproved results SVG: " + str(exc))
        files.append(relative)
        total += path.stat().st_size
    for name in required:
        if name not in files:
            problems.append("Missing public entry: " + name)
    return dict(
        status="pass" if not problems else "fail",
        artifact_path="docs/site",
        file_count=len(files),
        total_bytes=total,
        findings=problems,
        scope="Allowlisted public assets only; no private documentation, experiments or source tree",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = audit(args.root.resolve())
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "pass" else 1)
