#!/usr/bin/env python3
"""Install a selected Python package request outside the GUI process."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# B6: the DICOM converter is a pip wheel installed into Slicer's Python (wheel tags cp38-abi3 on every OS).
DCM2NIIX_REQUIREMENT = "dcm2niix==1.0.20261029"


def dcm2niix_pending(version=None):
    """True when the pinned dcm2niix wheel is not installed in this Python."""
    if version is None:
        import importlib.metadata
        try:
            version = importlib.metadata.version("dcm2niix")
        except importlib.metadata.PackageNotFoundError:
            return True
    return str(version) != DCM2NIIX_REQUIREMENT.split("==", 1)[1]


def main():
    from shared.managed_runtimes import write_json
    from shared.python_setup import install_requirements
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    for path in request.get("python_path_entries", ()):
        if Path(path).is_dir():
            sys.path.append(path)
    try:
        result = install_requirements(request["requirements"], protected=request.get("protect_bundled", True),
                                      loaded=request.get("loaded_packages", ()),
                                      torch_minimum=request.get("torch_minimum"),
                                      torchvision_required=request.get("torchvision_required", False))
    except Exception as exc:
        result = {"kind": "slicerdsurfer_python_setup", "status": "failed",
                  "error": f"{type(exc).__name__}: {exc}"}
    write_json(args.result, result)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
