#!/usr/bin/env python3
"""Snapshot and independently audit completed real Slicer acceptance cases."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

from audit_scientific_outputs import audit_output_root


def snapshot_case(case, output):
    source = Path(case["manifest_snapshot"])
    payload = json.loads(source.read_text(encoding="utf-8"))
    attempt = hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:8]
    dest = output / (case["name"] + "_" + attempt)
    metadata = dest / "00_run_metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    copies, files = {}, []
    for task in payload["tasks"]:
        values = dict(task.get("outputs") or {})
        values.update({key: task.get(key) for key in ("manifest_path", "qc_path")})
        for key, value in values.items():
            if not isinstance(value, str) or not Path(value).is_file() or value in copies:
                continue
            path = Path(value)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            entry = {"source": value, "sha256": digest, "bytes": path.stat().st_size}
            # Preserve numerical tables and QC before a later route can replace them.
            if path.suffix.lower() in {".json", ".csv", ".tsv"}:
                target = metadata / (task["name"] + "_" + key + "_" + path.name)
                shutil.copy2(path, target)
                copies[value] = str(target)
                entry["snapshot"] = str(target)
            files.append(entry)

    def replace(value):
        if isinstance(value, dict):
            return {key: replace(item) for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item) for item in value]
        return copies.get(value, value) if isinstance(value, str) else value

    manifest = metadata / "snapshot_orchestration_manifest.json"
    manifest.write_text(json.dumps(replace(payload), indent=2) + "\n", encoding="utf-8")
    (dest / "source_files.json").write_text(json.dumps(files, indent=2) + "\n", encoding="utf-8")
    audit = audit_output_root(dest, run_name="snapshot")
    (dest / "scientific_qc.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    checks = [check for subject in audit["subjects"] for check in subject["checks"]]
    return {"name": case["name"], "execution": case["status"], "qc": audit["status"],
            "check_count": len(checks), "failed_checks": [check for check in checks if not check["passed"]],
            "source_manifest": str(source), "snapshot": str(manifest),
            "human_anatomical_review": "pending"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-report", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, required=True)
    parser.add_argument("--wait-for-passing", action="store_true", help="Keep watching for corrected attempts after an initial failure.")
    parser.add_argument("--timeout", type=float, default=21600)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    results, attempts, done = {}, [], {}
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        cases_by_name = {}
        for path in args.case_report:
            try:
                for case in json.loads(path.read_text(encoding="utf-8"))["cases"]:
                    cases_by_name[case["name"]] = case
            except (OSError, ValueError):
                continue
        cases = list(cases_by_name.values())
        (args.output / "merged_cases.json").write_text(json.dumps({"cases": cases}, indent=2) + "\n", encoding="utf-8")
        for case in cases:
            stamp = (case.get("manifest_snapshot"), case["status"])
            if done.get(case["name"]) == stamp:
                continue
            try:
                result = snapshot_case(case, args.output)
            except Exception as exc:
                result = {"name": case["name"], "execution": case["status"], "qc": "failed", "error": str(exc)}
            results[case["name"]] = result
            attempts.append(result)
            done[case["name"]] = stamp
            print(json.dumps(result), flush=True)
            (args.output / "scientific_qc_summary.json").write_text(json.dumps(list(results.values()), indent=2) + "\n", encoding="utf-8")
            (args.output / "scientific_qc_attempts.json").write_text(json.dumps(attempts, indent=2) + "\n", encoding="utf-8")
        if len(results) >= args.expected_cases:
            passed = all(r["execution"] == "PASS" and r["qc"] == "passed" for r in results.values())
            if passed or not args.wait_for_passing:
                return 0 if passed else 1
        time.sleep(0.5)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
