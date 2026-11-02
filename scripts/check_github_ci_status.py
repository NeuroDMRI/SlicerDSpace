#!/usr/bin/env python3
"""Check GitHub Actions and branch-protection status for a commit."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPOSITORY = "NeuroDMRI/SlicerDSpace"
EXPECTED_REQUIRED_CHECK = "CI / fast-checks"
STALE_REQUIRED_CHECK_TOKENS = (
    "fresh-clone-smoke",
    "create-github-release",
    "create-github-release-with-smoke",
    "release-dispatch",
)


@dataclass(frozen=True)
class ApiFailure:
    endpoint: str
    status: int | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"endpoint": self.endpoint, "status": self.status, "message": self.message}


def _git_stdout(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", "-C", REPO_ROOT.as_posix(), *args],
        check=False,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def current_head() -> str:
    return _git_stdout(["rev-parse", "HEAD"]) or "HEAD"


def current_branch() -> str:
    return _git_stdout(["branch", "--show-current"]) or "main"


def _github_token(explicit_token: str | None = None) -> str | None:
    return explicit_token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")


def _request_json(url: str, *, token: str | None) -> dict[str, Any]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "SlicerDSpace-ci-status-check",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload) if payload else {}


def _api_url(repository: str, endpoint: str, query: dict[str, str] | None = None) -> str:
    clean_endpoint = endpoint.lstrip("/")
    url = f"https://api.github.com/repos/{repository}/{clean_endpoint}"
    if query:
        url += "?" + urllib.parse.urlencode(query)
    return url


def _safe_get(
    repository: str,
    endpoint: str,
    *,
    token: str | None,
    query: dict[str, str] | None = None,
) -> tuple[dict[str, Any] | None, ApiFailure | None]:
    url = _api_url(repository, endpoint, query)
    try:
        return _request_json(url, token=token), None
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        message = details or exc.reason
        return None, ApiFailure(endpoint=url, status=exc.code, message=message)
    except Exception as exc:
        return None, ApiFailure(endpoint=url, status=None, message=str(exc))


def summarize_check_runs(check_runs: list[dict[str, Any]]) -> dict[str, Any]:
    stale_runs = []
    unsuccessful = []
    expected_names = {"fast-checks", EXPECTED_REQUIRED_CHECK}
    expected_present = False
    for run in check_runs:
        name = str(run.get("name") or "")
        conclusion = str(run.get("conclusion") or "")
        status = str(run.get("status") or "")
        if name in expected_names:
            expected_present = True
        if any(token in name for token in STALE_REQUIRED_CHECK_TOKENS):
            stale_runs.append(name)
        if status != "completed" or conclusion not in {"success", "neutral", "skipped"}:
            unsuccessful.append(
                {
                    "name": name,
                    "status": status,
                    "conclusion": conclusion,
                    "html_url": run.get("html_url"),
                }
            )
    return {
        "total": len(check_runs),
        "expected_present": expected_present,
        "stale_named_runs": sorted(set(stale_runs)),
        "unsuccessful_runs": unsuccessful,
        "ok": expected_present and not unsuccessful,
    }


def stale_required_checks(contexts: list[str]) -> list[str]:
    return sorted(
        context
        for context in contexts
        if any(token in context for token in STALE_REQUIRED_CHECK_TOKENS)
    )


def required_check_assessment(contexts: list[str]) -> dict[str, Any]:
    expected_names = {"fast-checks", EXPECTED_REQUIRED_CHECK}
    stale = stale_required_checks(contexts)
    expected_present = any(context in expected_names for context in contexts)
    unexpected = sorted(context for context in contexts if context not in expected_names and context not in stale)
    return {
        "contexts": contexts,
        "expected_present": expected_present,
        "stale_required_checks": stale,
        "unexpected_required_checks": unexpected,
        "ok": expected_present and not stale,
    }


def branch_protection_remediation(repository: str, branch: str, contexts: list[str]) -> dict[str, Any]:
    assessment = required_check_assessment(contexts)
    stale = assessment["stale_required_checks"]
    intentional_existing = [
        context
        for context in contexts
        if context not in stale and context not in {"fast-checks", EXPECTED_REQUIRED_CHECK}
    ]
    recommended_contexts = [EXPECTED_REQUIRED_CHECK, *intentional_existing]
    recommended_contexts = list(dict.fromkeys(recommended_contexts))
    return {
        "ui_url": f"https://github.com/{repository}/settings/branches",
        "api_endpoint": f"/repos/{repository}/branches/{branch}/protection/required_status_checks",
        "expected_required_check": EXPECTED_REQUIRED_CHECK,
        "stale_required_checks_to_remove": stale,
        "recommended_required_checks": recommended_contexts,
        "manual_steps": [
            "Open repository Settings > Branches.",
            f"Edit the branch protection rule that targets `{branch}`.",
            "In required status checks, remove stale release/fresh-clone checks.",
            f"Keep or add `{EXPECTED_REQUIRED_CHECK}` as the automatic CI gate.",
            "Keep any other project-specific required checks only if they are still intentional.",
            "Save the rule, then rerun this script with a GitHub token to verify.",
        ],
    }


def format_branch_protection_guide(report: dict[str, Any]) -> str:
    remediation = report["branch_protection_remediation"]
    lines = [
        "# Branch Protection Required Checks Guide",
        "",
        f"Repository: `{report['repository']}`",
        f"Branch: `{report['branch']}`",
        f"Settings URL: {remediation['ui_url']}",
        "",
        "## Recommended State",
        "",
        "Required checks should include:",
    ]
    lines.extend(f"- `{context}`" for context in remediation["recommended_required_checks"])
    stale = remediation["stale_required_checks_to_remove"]
    if stale:
        lines.extend(["", "Remove stale checks:"])
        lines.extend(f"- `{context}`" for context in stale)
    if report["required_status_checks"]["unexpected_required_checks"]:
        lines.extend(["", "Review manually before keeping or removing:"])
        lines.extend(f"- `{context}`" for context in report["required_status_checks"]["unexpected_required_checks"])
    lines.extend(["", "## UI Steps", ""])
    lines.extend(f"{index}. {step}" for index, step in enumerate(remediation["manual_steps"], start=1))
    lines.extend(
        [
            "",
            "## API Reference",
            "",
            "The script intentionally does not patch branch protection automatically. "
            "If you choose to update by API, preserve any intentional checks and patch only the required-status-checks section.",
            "",
            f"Endpoint: `{remediation['api_endpoint']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def build_report(
    *,
    repository: str,
    ref: str,
    branch: str,
    token: str | None,
) -> dict[str, Any]:
    failures: list[ApiFailure] = []

    repo, failure = _safe_get(repository, "", token=token)
    if failure:
        failures.append(failure)

    checks, failure = _safe_get(
        repository,
        f"commits/{ref}/check-runs",
        token=token,
        query={"per_page": "100"},
    )
    if failure:
        failures.append(failure)
    check_runs = checks.get("check_runs", []) if checks else []

    combined_status, failure = _safe_get(repository, f"commits/{ref}/status", token=token)
    if failure:
        failures.append(failure)

    runs, failure = _safe_get(
        repository,
        "actions/runs",
        token=token,
        query={"head_sha": ref, "per_page": "20"},
    )
    if failure:
        failures.append(failure)

    required_status_checks, failure = _safe_get(
        repository,
        f"branches/{branch}/protection/required_status_checks",
        token=token,
    )
    if failure:
        failures.append(failure)
    contexts = []
    if required_status_checks:
        contexts = [str(context) for context in required_status_checks.get("contexts", [])]
        checks_payload = required_status_checks.get("checks", [])
        if isinstance(checks_payload, list):
            contexts.extend(str(item.get("context")) for item in checks_payload if item.get("context"))
        contexts = list(dict.fromkeys(contexts))

    required_status_checks_report = required_check_assessment(contexts)
    return {
        "kind": "slicerddsurfer_github_ci_status",
        "repository": repository,
        "ref": ref,
        "branch": branch,
        "token_present": bool(token),
        "repository_visible": bool(repo),
        "check_runs": summarize_check_runs(check_runs),
        "combined_status": {
            "state": combined_status.get("state") if combined_status else None,
            "total_count": combined_status.get("total_count") if combined_status else None,
        },
        "workflow_runs": [
            {
                "name": run.get("name"),
                "head_sha": run.get("head_sha"),
                "status": run.get("status"),
                "conclusion": run.get("conclusion"),
                "html_url": run.get("html_url"),
            }
            for run in (runs.get("workflow_runs", []) if runs else [])
        ],
        "required_status_checks": required_status_checks_report,
        "branch_protection_remediation": branch_protection_remediation(repository, branch, contexts),
        "api_failures": [failure.to_dict() for failure in failures],
    }


def report_ok(report: dict[str, Any]) -> bool:
    if report["api_failures"]:
        return False
    if not report["repository_visible"]:
        return False
    if not report["check_runs"]["ok"]:
        return False
    if report["combined_status"]["state"] not in {"success", None}:
        return False
    if not report["required_status_checks"]["ok"]:
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY, help="GitHub repository in owner/name form.")
    parser.add_argument("--ref", default=current_head(), help="Commit SHA or ref to inspect. Defaults to local HEAD.")
    parser.add_argument("--branch", default=current_branch(), help="Branch protection target. Defaults to current branch.")
    parser.add_argument("--token", help="GitHub token. Defaults to GITHUB_TOKEN or GH_TOKEN.")
    parser.add_argument("--json", action="store_true", help="Print full JSON report.")
    parser.add_argument(
        "--branch-protection-guide",
        action="store_true",
        help="Print a Markdown guide for removing stale required checks from GitHub branch protection.",
    )
    parser.add_argument("--warn-only", action="store_true", help="Always exit 0 after printing the report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_report(
        repository=args.repository,
        ref=args.ref,
        branch=args.branch,
        token=_github_token(args.token),
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"repository_visible={report['repository_visible']} token_present={report['token_present']}")
        print(f"check_runs_ok={report['check_runs']['ok']} total={report['check_runs']['total']}")
        print(f"expected_check_present={report['check_runs']['expected_present']}")
        print(f"combined_status={report['combined_status']['state']}")
        print(f"required_checks_ok={report['required_status_checks']['ok']}")
        if report["required_status_checks"]["contexts"]:
            print("required_checks=" + ", ".join(report["required_status_checks"]["contexts"]))
        if report["required_status_checks"]["stale_required_checks"]:
            print("stale_required_checks=" + ", ".join(report["required_status_checks"]["stale_required_checks"]))
        for run in report["check_runs"]["unsuccessful_runs"]:
            print(f"unsuccessful_check={run['name']} status={run['status']} conclusion={run['conclusion']}")
        for failure in report["api_failures"]:
            print(f"api_failure={failure['status']} {failure['endpoint']}: {failure['message']}")
    if args.branch_protection_guide:
        print(format_branch_protection_guide(report), end="")
    if args.warn_only:
        return 0
    return 0 if report_ok(report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
