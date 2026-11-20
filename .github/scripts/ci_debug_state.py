#!/usr/bin/env python3
"""Print the installed-extension state of a CI Slicer (diagnostic only, never fails the job).

Shows the Qt settings files that mention extensions, a shallow tree of the extension root, and whether
SlicerDSpace.py is where the workflow expects it. Output is also copied to <artifacts>/state_<label>.txt.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
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


def probe_extension_manager(slicer_exe, lines):
    """Start Slicer twice (cwd = the checkout, cwd = an empty folder) and record what its extension manager believes.

    Run 44819974644 (Linux, fresh Slicer): the website installer logged 'Failed to uninstall SlicerDSpace
    extension: extension is already loaded' before anything was installed, then left SlicerDSpace scheduled
    for uninstall. This shows whether a fresh Slicer already reports the extension as installed."""
    script = Path(__file__).resolve().with_name("ci_probe_extensions.py")
    for label, cwd in (("cwd=checkout", Path.cwd()), ("cwd=empty-dir", Path(tempfile.mkdtemp(prefix="probe-cwd-")))):
        out = Path(tempfile.mkdtemp(prefix="probe-out-")) / "probe.json"
        command = [str(slicer_exe), "--no-splash", "--no-main-window", "--python-script", str(script)]
        if sys.platform.startswith("linux") and shutil.which("xvfb-run"):
            command = ["xvfb-run", "-a"] + command
        env = dict(os.environ, CI_PROBE_OUT=str(out))
        try:
            subprocess.run(command, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
        except (OSError, subprocess.SubprocessError) as exc:
            lines.append("probe %s: failed to run: %s" % (label, exc))
            continue
        lines.append("probe %s: %s" % (label, out.read_text(encoding="utf-8").strip() if out.is_file() else "(no output)"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--slicer-exe", required=True, type=Path)
    parser.add_argument("--ext-root", type=Path)
    parser.add_argument("--probe", action="store_true", help="also start Slicer and ask its extension manager")
    args = parser.parse_args(argv)
    lines = ["== state: %s" % args.label]
    for ini in settings_files_near(args.slicer_exe):
        lines.append("settings file: %s" % ini)
        for line in ini.read_text(encoding="utf-8", errors="replace").splitlines():
            if any(word in line for word in ("Extension", "SlicerDSpace", "Scheduled", "Bookmarked")):
                lines.append("    " + line[:300])
    if args.probe:
        probe_extension_manager(args.slicer_exe, lines)
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
