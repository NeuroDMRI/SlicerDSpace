#!/usr/bin/env python3
"""Verify recorded third-party license notices and their package inclusion offline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

try:
    from scripts.extension_runtime.runtime_resources import runtime_files
except ModuleNotFoundError:
    from extension_runtime.runtime_resources import runtime_files


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_RECORDS = {
    "conversion": "anatomy/parcellation/conversion/LICENSE.txt",
    "freesurfer_lut": "licenses/FreeSurfer.txt",
    "ddsurfer": "anatomy/surface/ddsurfer_backend/LICENSE.txt",
    "pytorch3d": "licenses/PyTorch3D.txt",
    "mkcurve": "licenses/MK-Curve-LICENSE.txt",
    "mkcurve_dki": "licenses/MK-Curve-DKI-MPL-2.0.txt",
    "wma_preprocessing": "licenses/WMA.txt",
    "freewater_wls": "licenses/DIPY.txt",
    "slicerdmri_tensor_fit": "licenses/SlicerDMRI.txt",
    "pnl_cnn": "shared/pnl_cnn/LICENSE.txt",
}


def audit(root=ROOT):
    root = root.resolve()
    findings = []
    checked = []
    try:
        spec = json.loads((root / "licenses/provenance.json").read_text(encoding="utf-8"))
        if spec.get("version") != 1 or not isinstance(spec.get("records"), list):
            raise ValueError("Unsupported license provenance schema")
        selected = set(runtime_files(root))
        seen = set()
        for record in spec["records"]:
            if not isinstance(record, dict) or not all(isinstance(record.get(key), str) for key in
                                                      ("id", "license", "path", "sha256", "source", "conditions")):
                raise ValueError("Malformed license record")
            if not isinstance(record.get("covers"), list) or not record["covers"] or not all(
                    isinstance(covered, str) for covered in record["covers"]):
                raise ValueError("Malformed attributed paths")
            component = record["id"]
            if component in seen:
                raise ValueError(f"Duplicate license record: {component}")
            seen.add(component)
            relative = record["path"]
            path = PurePosixPath(relative)
            if path.is_absolute() or ".." in path.parts or "\\" in relative:
                raise ValueError(f"Unsafe license path: {relative}")
            if REQUIRED_RECORDS.get(component) != relative:
                raise ValueError(f"Unexpected license record: {component}/{relative}")
            target = root / relative
            if not target.is_file() or not target.resolve().is_relative_to(root):
                raise ValueError(f"Missing/outside license file: {relative}")
            if hashlib.sha256(target.read_bytes()).hexdigest() != record["sha256"]:
                findings.append(f"Changed license text: {relative}")
            if not re.fullmatch(r"https://github\.com/[^/]+/[^/]+/blob/[0-9a-f]{40}/.+", record["source"]):
                findings.append(f"Unpinned license source: {component}")
            if relative not in selected:
                findings.append(f"License omitted from runtime package: {relative}")
            for covered in record["covers"]:
                location = root / covered
                if not location.exists() or not location.resolve().is_relative_to(root):
                    findings.append(f"Missing/outside attributed component: {covered}")
            checked.append(component)
        for missing in sorted(REQUIRED_RECORDS.keys() - seen):
            findings.append(f"Missing license record: {missing}")
        if "licenses/provenance.json" not in selected:
            findings.append("License provenance omitted from runtime package")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        findings.append(str(exc))
    return {"status": "pass" if not findings else "fail", "checked": checked,
            "findings": findings,
            "scope": "Local notice integrity/package inclusion, not legal certification or model redistribution permission",
            "remaining_review": ["model weights", "templates", "subject-derived fixtures", "patent/access restrictions"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    result = audit(args.root)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
