#!/usr/bin/env python3
"""Print the installed-extension state of a CI Slicer (diagnostic only, never fails the job).

Shows the Qt settings files that mention extensions, a shallow tree of the extension root, and whether
SlicerDSpace.py is where the workflow expects it. Output is also copied to <artifacts>/state_<label>.txt.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ci_install_dspace import settings_files_near  # noqa: E402


def tree(root, depth, lines):
    root = Path(root)
    if not root.exists():
        lines.append("  (missing) %s" % root)
        return
    base = len(root.parts)
    for current, dirs, files in os.walk(str(root)):
        level = len(Path(current).parts) - base
        dirs.sort()
        if level >= depth:
            lines.append("  %s%s/ ..." % ("  " * level, Path(current).name))
            dirs[:] = []
            continue
        lines.append("  %s%s/  (%d files)" % ("  " * level, Path(current).name, len(files)))
        for name in sorted(files)[:6]:
            lines.append("  %s  %s" % ("  " * level, name))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--slicer-exe", required=True, type=Path)
    parser.add_argument("--ext-root", type=Path)
    args = parser.parse_args(argv)
    lines = ["== state: %s" % args.label]
    for ini in settings_files_near(args.slicer_exe):
        lines.append("settings file: %s" % ini)
        for line in ini.read_text(encoding="utf-8", errors="replace").splitlines():
            if any(word in line for word in ("Extension", "SlicerDSpace", "Scheduled", "Bookmarked")):
                lines.append("    " + line[:300])
    slicer_dir = args.slicer_exe.resolve().parent
    for parent in [slicer_dir] + list(slicer_dir.parents)[:2]:
        for candidate in sorted(parent.glob("slicer.org/Extensions-*")):
            lines.append("extensions dir: %s" % candidate)
            tree(candidate, 2, lines)
    if args.ext_root:
        lines.append("ext_root: %s (exists=%s)" % (args.ext_root, args.ext_root.exists()))
        tree(args.ext_root, 3, lines)
        for module in sorted(args.ext_root.glob("lib/Slicer-*/qt-scripted-modules/SlicerDSpace.py")):
            lines.append("SlicerDSpace.py: %s" % module)
        for name in ("bundled-python", "bundled-runtimes"):
            lines.append("%s exists=%s" % (name, (args.ext_root / name).is_dir()))
    text = "\n".join(lines)
    print(text, flush=True)
    temp = Path(os.environ.get("RUNNER_TEMP") or os.environ.get("TMPDIR") or "/tmp")
    out = temp / "ci-artifacts"
    out.mkdir(parents=True, exist_ok=True)
    (out / ("state_%s.txt" % args.label)).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
