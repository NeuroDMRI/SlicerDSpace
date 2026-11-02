#!/usr/bin/env bash
# Fresh-clone release smoke test for SlicerDSpace.
#
# This intentionally clones into /tmp by default and downloads only the selected
# model-asset profile, so it can be used as a manual release gate without making
# normal pull-request CI download large external model payloads.

set -u

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_repo_root="$(cd "${script_dir}/.." && pwd)"

default_repo_url="$(git -C "${source_repo_root}" config --get remote.origin.url 2>/dev/null || true)"
default_ref="$(git -C "${source_repo_root}" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
if [[ -z "${default_ref}" || "${default_ref}" == "HEAD" ]]; then
  default_ref="$(git -C "${source_repo_root}" rev-parse HEAD 2>/dev/null || true)"
fi

SMOKE_REPO_URL="${SMOKE_REPO_URL:-${default_repo_url:-https://github.com/NeuroDMRI/SlicerDSpace.git}}"
SMOKE_REF="${SMOKE_REF:-${default_ref:-main}}"
SMOKE_ROOT="${SMOKE_ROOT:-/tmp/slicerdsurfer-fresh-clone-smoke}"
SMOKE_ASSET_PROFILE="${SMOKE_ASSET_PROFILE:-core}"
SMOKE_ASSET_MODE="${SMOKE_ASSET_MODE:-plan}"
SMOKE_REPORT="${SMOKE_REPORT:-${SMOKE_ROOT}/fresh_clone_smoke_report.json}"

clone_dir="${SMOKE_ROOT}/checkout"
log_dir="${SMOKE_ROOT}/logs"
steps_jsonl="${SMOKE_ROOT}/steps.jsonl"
started_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

rm -rf "${SMOKE_ROOT}"
mkdir -p "${clone_dir}" "${log_dir}" "$(dirname "${SMOKE_REPORT}")"
: > "${steps_jsonl}"

json_append_step() {
  python3 - "$steps_jsonl" "$@" <<'PY'
import json
import sys
from pathlib import Path

(
    steps_jsonl,
    name,
    status,
    exit_code,
    started_at,
    ended_at,
    duration_seconds,
    cwd,
    stdout_path,
    stderr_path,
    command,
) = sys.argv[1:]

record = {
    "name": name,
    "status": status,
    "exit_code": None if exit_code == "" else int(exit_code),
    "started_at": started_at,
    "ended_at": ended_at,
    "duration_seconds": float(duration_seconds),
    "cwd": cwd,
    "stdout": stdout_path,
    "stderr": stderr_path,
    "command": command,
}
with Path(steps_jsonl).open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(record, sort_keys=True) + "\n")
PY
}

run_step() {
  local name="$1"
  local cwd="$2"
  shift 2
  local stdout_path="${log_dir}/${name}.stdout.log"
  local stderr_path="${log_dir}/${name}.stderr.log"
  local command_display
  printf -v command_display '%q ' "$@"
  local start_iso end_iso start_epoch end_epoch duration rc
  start_iso="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  start_epoch="$(date +%s)"
  (
    cd "${cwd}" && "$@"
  ) >"${stdout_path}" 2>"${stderr_path}"
  rc=$?
  end_iso="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  end_epoch="$(date +%s)"
  duration=$((end_epoch - start_epoch))
  if [[ "${rc}" -eq 0 ]]; then
    json_append_step "${name}" "passed" "${rc}" "${start_iso}" "${end_iso}" "${duration}" "${cwd}" "${stdout_path}" "${stderr_path}" "${command_display% }"
  else
    json_append_step "${name}" "failed" "${rc}" "${start_iso}" "${end_iso}" "${duration}" "${cwd}" "${stdout_path}" "${stderr_path}" "${command_display% }"
    printf 'Fresh-clone step failed: %s (exit %s)\n' "${name}" "${rc}" >&2
    printf '%s\n' "--- stdout: ${stdout_path} (last 80 lines) ---" >&2
    tail -n 80 "${stdout_path}" >&2
    printf '%s\n' "--- stderr: ${stderr_path} (last 80 lines) ---" >&2
    tail -n 80 "${stderr_path}" >&2
  fi
  return "${rc}"
}

write_report() {
  local status="$1"
  local ended_at
  ended_at="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  python3 - "$SMOKE_REPORT" "$steps_jsonl" "$status" "$started_at" "$ended_at" "$SMOKE_REPO_URL" "$SMOKE_REF" "$SMOKE_ASSET_PROFILE" "$SMOKE_ASSET_MODE" "$SMOKE_ROOT" "$clone_dir" <<'PY'
import json
import platform
import subprocess
import sys
from pathlib import Path

(
    report_path,
    steps_jsonl,
    status,
    started_at,
    ended_at,
    repo_url,
    ref,
    asset_profile,
    asset_mode,
    smoke_root,
    clone_dir,
) = sys.argv[1:]

steps = []
steps_path = Path(steps_jsonl)
if steps_path.is_file():
    with steps_path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                steps.append(json.loads(line))

revision = subprocess.run(
    ["git", "-C", clone_dir, "rev-parse", "HEAD"],
    capture_output=True, text=True, check=False,
)
report = {
    "kind": "slicerdsurfer_fresh_clone_smoke_report",
    "version": 1,
    "status": status,
    "started_at": started_at,
    "ended_at": ended_at,
    "repo_url": repo_url,
    "ref": ref,
    "resolved_commit": revision.stdout.strip() if revision.returncode == 0 else None,
    "asset_profile": asset_profile,
    "asset_mode": asset_mode,
    "code_checkout": {
        "strategy": "git clone with GIT_LFS_SKIP_SMUDGE=1",
        "step_names": ["git_clone", "git_checkout"],
        "separate_from_model_assets": True,
    },
    "model_asset_check": {
        "strategy": "manifest-plan in CI-safe mode; manifest-driven download in release-gate mode",
        "profile": asset_profile,
        "mode": asset_mode,
        "step_names": [f"model_assets_{asset_mode}_{asset_profile}"],
        "separate_from_code_checkout": True,
    },
    "smoke_root": smoke_root,
    "clone_dir": clone_dir,
    "platform": {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
    },
    "steps": steps,
}
target = Path(report_path)
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

status="passed"

if ! run_step "git_clone" "${SMOKE_ROOT}" env GIT_LFS_SKIP_SMUDGE=1 git clone "${SMOKE_REPO_URL}" "${clone_dir}"; then
  status="failed"
fi

if [[ "${status}" == "passed" && -n "${SMOKE_REF}" ]]; then
  if ! run_step "git_checkout" "${clone_dir}" git checkout "${SMOKE_REF}"; then
    status="failed"
  fi
fi

case "${SMOKE_ASSET_MODE}" in
  plan)
    if [[ "${status}" == "passed" ]]; then
      if ! run_step "model_assets_plan_${SMOKE_ASSET_PROFILE}" "${clone_dir}" python3 scripts/download_model_assets.py --profile "${SMOKE_ASSET_PROFILE}" --print-plan; then
        status="failed"
      fi
    fi
    ;;
  download)
    if [[ "${status}" == "passed" ]]; then
      if ! run_step "model_assets_download_${SMOKE_ASSET_PROFILE}" "${clone_dir}" bash scripts/download_model_assets.sh --profile "${SMOKE_ASSET_PROFILE}"; then
        status="failed"
      fi
    fi
    ;;
  *)
    echo "Unsupported SMOKE_ASSET_MODE=${SMOKE_ASSET_MODE}. Use plan or download." >&2
    status="failed"
    ;;
esac

if [[ "${status}" == "passed" ]]; then
  if ! run_step "make_test_fast" "${clone_dir}" make test-fast; then
    status="failed"
  fi
fi

if [[ "${status}" == "passed" && "${SMOKE_ASSET_MODE}" == "download" ]]; then
  if ! run_step "verify_model_assets_${SMOKE_ASSET_PROFILE}" "${clone_dir}" python3 scripts/download_model_assets.py --profile "${SMOKE_ASSET_PROFILE}" --verify-only; then
    status="failed"
  fi
fi

write_report "${status}"

echo "Fresh-clone smoke ${status}. Report: ${SMOKE_REPORT}"
if [[ "${status}" == "passed" ]]; then
  exit 0
fi
exit 1
