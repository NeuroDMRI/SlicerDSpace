#!/usr/bin/env python3
"""Plan or install selected optional runtimes in a plugin-owned user directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared.managed_runtimes import PROFILES, install_plan, provision, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", choices=tuple(PROFILES), required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--conda", type=Path)
    parser.add_argument("--asset-root", type=Path, help="Shared model/resource root; legacy prefix-local assets remain supported")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--repair", action="store_true", help="Reinstall selected plugin-owned runtimes; never modify an unmanaged prefix")
    parser.add_argument("--accept-fsl-license", action="store_true")
    parser.add_argument("--accept-pnl-terms", action="store_true")
    parser.add_argument("--user-config", type=Path, help="User config JSON read to keep user-set tool paths during a migration")
    parser.add_argument("--result", type=Path)
    args = parser.parse_args(argv)
    try:
        user_config = None
        # R-N3: the config only serves the MRtrix migration, which exists on macOS (the tarball profile).
        if args.execute and platform.system() == "Darwin" and "mrtrix" in args.profiles:
            from shared.user_config import load_user_config
            user_config = load_user_config(args.user_config)
        result = provision(args.profiles, root=args.root, conda=args.conda,
                           accept_fsl_license=args.accept_fsl_license,
                           accept_pnl_terms=args.accept_pnl_terms, repair=args.repair,
                           asset_root=args.asset_root, user_config=user_config) if args.execute else install_plan(args.profiles, args.root)
    except Exception as exc:
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    if args.result:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.result, result)
    print(json.dumps(result, indent=2), flush=True)
    return 1 if result.get("status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
