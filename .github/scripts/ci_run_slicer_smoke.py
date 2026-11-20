#!/usr/bin/env python3
"""Run Testing/Python/slicer_ci_compat_smoke.py through real Slicer and judge the JSON report (CI helper).

Linux uses xvfb-run -a; Windows and macOS run without a display. Every OS uses --no-main-window.
Slicer's console output goes to a log file (a GUI-subsystem Slicer.exe on Windows has no console).
The report decides the result: a missing report, a non-"pass" status, or a non-zero exit all fail.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE = REPO_ROOT / "Testing" / "Python" / "slicer_ci_compat_smoke.py"


def log(message):
    print("[ci-run-smoke] " + message, flush=True)


def build_command(slicer_exe, platform, module_dir, mode, xvfb=None):
    command = [str(slicer_exe), "--no-splash", "--no-main-window"]
    if mode == "repo-only":
        command += ["--additional-module-paths", str(module_dir)]
    command += ["--python-script", str(SMOKE)]
    if platform.startswith("linux"):
        command = [xvfb or "xvfb-run", "-a"] + command
    return command


def judge(report_path):
    """Return (ok, text). A missing or unreadable report is a failure."""
    if not report_path.is_file():
        return False, "no report was written at %s (Slicer crashed or the smoke never ran)" % report_path
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        return False, "report is not valid JSON: %s" % exc
    lines = ["status: %s" % report.get("status")]
    for check in report.get("checks", []):
        lines.append("  %-5s %s  %s" % (check["status"].upper(), check["name"], check.get("detail", "")[:200]))
    return report.get("status") == "pass", "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slicer-exe", required=True, type=Path)
    parser.add_argument("--module-dir", required=True, type=Path, help="Directory with SlicerDSpace.py (code under test)")
    parser.add_argument("--mode", required=True, choices=("repo-only", "full-install"))
    parser.add_argument("--pipeline", default="required", choices=("required", "optional", "skip"))
    temp = Path(os.environ.get("RUNNER_TEMP") or os.environ.get("TMPDIR") or "/tmp")
    parser.add_argument("--artifacts-dir", type=Path, default=temp / "ci-artifacts")
    args = parser.parse_args(argv)

    artifacts = args.artifacts_dir
    work = artifacts / "smoke-work"
    report = artifacts / "smoke_report.json"
    artifacts.mkdir(parents=True, exist_ok=True)
    if report.exists():
        report.unlink()
    env = dict(os.environ)
    env.update(SLICERDSPACE_CI_MODULE_DIR=str(args.module_dir), SLICERDSPACE_CI_REPORT=str(report),
               SLICERDSPACE_CI_WORK=str(work), SLICERDSPACE_CI_MODE=args.mode, SLICERDSPACE_CI_PIPELINE=args.pipeline,
               PYTHONNOUSERSITE="1")
    command = build_command(args.slicer_exe, sys.platform, args.module_dir, args.mode, shutil.which("xvfb-run"))
    log("module dir: %s" % args.module_dir)
    log("run: " + " ".join(command))
    slicer_log = artifacts / "slicer_stdout.log"
    with open(str(slicer_log), "wb") as handle:
        code = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
    log("Slicer exit code %d; its output follows (full copy: %s)" % (code, slicer_log))
    text = slicer_log.read_text(encoding="utf-8", errors="replace")
    print(text[-20000:], flush=True)
    if 'uninstalled extension "SlicerDSpace"' in text:
        log("ERROR: Slicer deleted the SlicerDSpace extension at startup (it was scheduled for uninstall); "
            "the code under test is gone, so the checks below cannot be trusted")
    smoke_log = work / "ci_smoke.log"
    if smoke_log.is_file():
        log("smoke log:\n" + smoke_log.read_text(encoding="utf-8", errors="replace")[-20000:])
    ok, summary = judge(report)
    log("report summary\n" + summary)
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a", encoding="utf-8") as handle:
            handle.write("### Slicer smoke (%s)\n\n```\n%s\n```\n" % (args.mode, summary))
    if code != 0:
        log("Slicer exited with code %d" % code)
    return 0 if (ok and code == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
