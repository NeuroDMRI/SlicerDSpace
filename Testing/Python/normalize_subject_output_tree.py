#!/usr/bin/env python3
"""Normalize one subject output tree into the canonical SlicerDSpace layout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-root", required=True)
    parser.add_argument("--mode", choices=("symlink", "copy"), default="copy")
    parser.add_argument("--archive-legacy", action="store_true")
    parser.add_argument("--keep-legacy", action="store_true")
    parser.add_argument("--current-run-name", default="csd_act")
    parser.add_argument("--current-tractography-backend", default="csd_act", choices=("csd_act", "dti_det", "saf"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from shared import archive_development_outputs, migrate_subject_output_tree, subject_layout

    subject_root = Path(args.subject_root).expanduser().resolve()
    layout = subject_layout(subject_root)
    clean_legacy = bool(args.archive_legacy or not args.keep_legacy)
    migrated = migrate_subject_output_tree(
        subject_root,
        clean_legacy=clean_legacy,
    )
    created = migrated.get("published") or {}
    moved = archive_development_outputs(
        subject_root,
        current_run_name=args.current_run_name,
        current_tractography_backend=args.current_tractography_backend,
    )
    summary = {
        "subject_root": subject_root.as_posix(),
        "outputs_root": layout.outputs.as_posix(),
        "created_count": len(created),
        "archived_debug_count": len(moved),
        "created": {key: value.as_posix() for key, value in created.items()},
        "archived_debug": {key: value.as_posix() for key, value in moved.items()},
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
