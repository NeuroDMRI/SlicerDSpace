#!/usr/bin/env python3
"""Plan or install verified native runtime bundles (mrtrix, fsl) without WSL or Conda."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.native_runtimes import provision_bundles
from shared.user_config import update_user_config


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles", nargs="+", default=["mrtrix", "fsl"], choices=("mrtrix", "fsl"))
    parser.add_argument("--root", type=Path, help="Runtime installation root directory.")
    parser.add_argument("--archive-dir", type=Path, help="Optional offline archive directory.")
    parser.add_argument("--execute", action="store_true", help="Install bundles instead of printing the plan.")
    parser.add_argument("--accept-terms", action="store_true", help="Confirm review of MRtrix3/FSL licences.")
    parser.add_argument("--save-config", action="store_true", help="Save installed bin directories to user_config.json.")
    parser.add_argument("--result", type=Path, help="Optional JSON receipt output file.")
    args = parser.parse_args(argv)
    try:
        payload = provision_bundles(
            args.bundles,
            root=args.root,
            archive_dir=args.archive_dir,
            accept_terms=args.accept_terms,
            execute=args.execute,
        )
        if args.execute and args.save_config and payload.get("external_dependencies"):
            update_user_config({"external_dependencies": payload["external_dependencies"]})
        code = 0
    except Exception as exc:
        payload = {"kind": "slicerdsurfer_runtime_setup", "status": "failed", "error": str(exc)}
        code = 1
    if args.result is not None:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
