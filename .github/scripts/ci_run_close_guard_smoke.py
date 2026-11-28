#!/usr/bin/env python3
"""Run Testing/Python/slicer_close_guard_smoke.py through real Slicer and judge its JSON report (CI helper).

The smoke needs a visible main window, so there is no --no-main-window. Linux uses xvfb-run -a.
Slicer's console output goes to a log file (a GUI-subsystem Slicer.exe on Windows has no console).
The JSON `passed` field is the result of record: a posted Quit can change Slicer's exit code, so a
non-zero exit with passed=true still passes (the exit code is printed). A missing or unreadable
report, or passed not true, fails.
On a timeout the helper kills the whole process tree (the child's process group on POSIX, so
SlicerApp-real and Xvfb die with the launcher; `taskkill /T /F` on Windows), then reads the report.
The smoke writes its report before it asks Slicer to exit, so a report with passed=true after a
timeout is a shutdown stall and still passes; no report after a timeout fails.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE = REPO_ROOT / "Testing" / "Python" / "slicer_close_guard_smoke.py"
REPORT_NAME = "slicer_close_guard_smoke.json"
LOG_NAME = "slicer_close_guard_smoke.log"
DEFAULT_TIMEOUT = 240


def log(message):
    print("[ci-close-guard-smoke] " + message, flush=True)


def build_command(slicer_exe, platform, xvfb=None):
    command = [str(slicer_exe), "--no-splash", "--disable-settings", "--disable-cli-modules",
               "--python-script", str(SMOKE)]
    if platform.startswith("linux"):
        command = [xvfb or "xvfb-run", "-a"] + command
    return command


def judge(report_path):
    """Return (ok, text). The report's `passed` field decides; a missing or unreadable report fails."""
    if not report_path.is_file():
        return False, "no report was written at %s (Slicer crashed or the smoke never ran)" % report_path
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        return False, "report is not valid JSON: %s" % exc
    if not isinstance(report, dict):
        return False, "report is not a JSON object"
    passed = report.get("passed") is True
    lines = ["passed: %s" % report.get("passed")]
    if "error" in report:
        lines.append("error: %s" % str(report["error"])[:500])
    checks = report.get("checks")
    if isinstance(checks, dict) and not passed:
        for name, value in checks.items():
            if value is not True:
                lines.append("  not True: %s = %r" % (name, value))
    return passed, "\n".join(lines)


def kill_tree(process, platform):
    """Kill the child and everything it started, then wait a bounded time. Falls back to process.kill()."""
    try:
        if platform.startswith("win"):
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60, check=True)
        else:
            os.killpg(os.getpgid(process.pid), getattr(signal, "SIGKILL", 9))
    except Exception:
        try:
            process.kill()
        except OSError:
            pass
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass


def run_slicer(command, env, cwd, log_path, timeout, platform=None):
    """Run Slicer with its output in log_path. Return (exit_code, timed_out); the tree is dead on return."""
    platform = sys.platform if platform is None else platform
    extra = {} if platform.startswith("win") else {"start_new_session": True}
    with open(str(log_path), "wb") as handle:
        process = subprocess.Popen(command, env=env, cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT, **extra)
        try:
            return process.wait(timeout=timeout), False
        except subprocess.TimeoutExpired:
            kill_tree(process, platform)
            return None, True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slicer-exe", required=True, type=Path)
    temp = Path(os.environ.get("RUNNER_TEMP") or os.environ.get("TMPDIR") or "/tmp")
    parser.add_argument("--artifacts-dir", type=Path, default=temp / "ci-artifacts")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    args = parser.parse_args(argv)

    artifacts = args.artifacts_dir
    artifacts.mkdir(parents=True, exist_ok=True)
    report = artifacts / REPORT_NAME
    slicer_log = artifacts / LOG_NAME
    if report.exists():
        report.unlink()
    env = dict(os.environ)
    env["SLICERDSPACE_SMOKE_RESULT"] = str(report)
    command = build_command(args.slicer_exe, sys.platform, shutil.which("xvfb-run"))
    log("run: " + " ".join(command))
    code, timed_out = run_slicer(command, env, artifacts, slicer_log, args.timeout)
    if timed_out:
        log("ERROR: Slicer timed out after %d s; the process tree was killed. The report below decides "
            "(a report with passed=true means the stall was during shutdown)" % args.timeout)
    else:
        log("Slicer exit code %d (informational: the report decides)" % code)
    text = slicer_log.read_text(encoding="utf-8", errors="replace")
    print(text[-20000:], flush=True)
    ok, summary = judge(report)
    log("report summary\n" + summary)
    log("VERDICT: " + ("PASS" if ok else "FAIL"))
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a", encoding="utf-8") as handle:
            handle.write("### Slicer close guard smoke\n\n```\n%s\n```\n" % summary)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
