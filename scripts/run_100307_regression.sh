#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
read -r -a PYTHON_CMD <<< "$PYTHON_BIN"
SUBJECT_ROOT="${SUBJECT_ROOT:-/data/chengjin/SlicerDDSurfer-test/100307}"
SUBJECT_ID="${SUBJECT_ID:-100307}"
RUN_NAME="${RUN_NAME:-csd_act}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
STRICT="${STRICT:-0}"
DRY_RUN="${DRY_RUN:-0}"

TARGETS="${TARGETS:-validation,preprocessing,segmentation,parcellation,microstructure,tractography,bundles,connectome}"

cd "$REPO_ROOT"

echo "[100307 regression] repo: $REPO_ROOT"
echo "[100307 regression] subject: $SUBJECT_ROOT"
echo "[100307 regression] python: $PYTHON_BIN"
echo "[100307 regression] targets: $TARGETS"

if [[ ! -d "$SUBJECT_ROOT/input" ]]; then
  echo "[100307 regression] missing subject input folder: $SUBJECT_ROOT/input" >&2
  exit 2
fi

for required in data.nii.gz bvals bvecs nodif_brain_mask.nii.gz; do
  if [[ ! -f "$SUBJECT_ROOT/input/$required" ]]; then
    echo "[100307 regression] missing required input: $SUBJECT_ROOT/input/$required" >&2
    exit 2
  fi
done

COMMON_ARGS=(
  --subject-root "$SUBJECT_ROOT"
  --subject-id "$SUBJECT_ID"
  --run-name "$RUN_NAME"
  --targets "$TARGETS"
  --tractography-backend csd_act
  --connectome-backend python
)

if [[ -n "$OUTPUT_ROOT" ]]; then
  COMMON_ARGS+=(--output-root "$OUTPUT_ROOT")
fi

if [[ "$STRICT" == "1" ]]; then
  COMMON_ARGS+=(--strict-environment)
fi

PLAN_JSON="$(mktemp /tmp/slicerdsurfer_100307_plan_XXXXXX.json)"
RUN_JSON="$(mktemp /tmp/slicerdsurfer_100307_run_XXXXXX.json)"
RESUME_JSON="$(mktemp /tmp/slicerdsurfer_100307_resume_XXXXXX.json)"

echo "[100307 regression] generating run plan"
"${PYTHON_CMD[@]}" run_subject_pipeline.py "${COMMON_ARGS[@]}" --plan > "$PLAN_JSON"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[100307 regression] DRY_RUN=1, skipping execution"
  "${PYTHON_CMD[@]}" - "$PLAN_JSON" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
plan = payload["run_plan"]
summary = plan["summary"]
print(f"[100307 regression] plan summary: will_run={summary['will_run']} skipped={summary['skipped']} blocked={summary['blocked']}")
print(f"[100307 regression] run_plan_path={payload['run_plan_path']}")
if summary["blocked"]:
    raise SystemExit("[100307 regression] run plan contains blocked targets")
PY
  exit 0
fi

echo "[100307 regression] executing unified pipeline with resume"
"${PYTHON_CMD[@]}" run_subject_pipeline.py "${COMMON_ARGS[@]}" --execute > "$RUN_JSON"

echo "[100307 regression] re-running once to verify resume/skip behavior"
"${PYTHON_CMD[@]}" run_subject_pipeline.py "${COMMON_ARGS[@]}" --execute > "$RESUME_JSON"

echo "[100307 regression] validating manifests, QC schema, and output tree"
"${PYTHON_CMD[@]}" - "$PLAN_JSON" "$RUN_JSON" "$RESUME_JSON" <<'PY'
import json
import sys
from pathlib import Path

plan_payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
run_payload = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
resume_payload = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))

def fail(message: str) -> None:
    raise SystemExit(f"[100307 regression] {message}")

plan_path = Path(plan_payload["run_plan_path"])
if not plan_path.is_file():
    fail(f"missing run plan: {plan_path}")

manifest_path = Path(run_payload["run_manifest_path"])
if not manifest_path.is_file():
    fail(f"missing run manifest: {manifest_path}")

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
required_top_dirs = ["00_run_metadata", "01_preproc", "02_microstructure", "03_anatomy", "04_tractography", "05_statistics"]
output_root = Path(manifest["output_root"])
for dirname in required_top_dirs:
    if not (output_root / dirname).exists() and not (
        dirname == "03_anatomy" and (output_root / "03_structural").is_dir()
    ):
        fail(f"missing canonical output directory: {output_root / dirname}")

nested = output_root / Path(manifest["subject_id"]) / "outputs"
if nested.exists():
    fail(f"unexpected nested subject outputs: {nested}")

tasks = manifest.get("tasks", [])
if not tasks:
    fail("orchestration manifest contains no tasks")

failed = [task for task in tasks if task.get("status") == "failed"]
if failed:
    fail("failed task(s): " + ", ".join(task.get("name", "?") for task in failed))

for task in tasks:
    for key in ("normalized_manifest_path", "normalized_qc_path", "normalized_logs_dir"):
        value = task.get(key)
        if not value:
            fail(f"{task.get('name')} missing {key}")
    normalized_manifest = Path(task["normalized_manifest_path"])
    normalized_qc = Path(task["normalized_qc_path"])
    normalized_logs = Path(task["normalized_logs_dir"])
    if not normalized_manifest.is_file():
        fail(f"missing normalized manifest: {normalized_manifest}")
    if not normalized_qc.is_file():
        fail(f"missing normalized qc: {normalized_qc}")
    if not normalized_logs.is_dir():
        fail(f"missing normalized logs dir: {normalized_logs}")
    normalized_data = json.loads(normalized_manifest.read_text(encoding="utf-8"))
    for field in ("inputs", "outputs", "dependencies", "status", "runtime_seconds", "reused"):
        if field not in normalized_data:
            fail(f"{normalized_manifest} missing schema field: {field}")

resume_manifest_path = Path(resume_payload["run_manifest_path"])
resume_manifest = json.loads(resume_manifest_path.read_text(encoding="utf-8"))
resume_tasks = resume_manifest.get("tasks", [])
resume_bad = [task for task in resume_tasks if task.get("status") == "failed"]
if resume_bad:
    fail("resume run failed task(s): " + ", ".join(task.get("name", "?") for task in resume_bad))

blocked = [task for task in tasks if task.get("status") == "blocked"]
skipped = [task for task in resume_tasks if task.get("status") == "skipped"]
print(f"[100307 regression] manifest={manifest_path}")
print(f"[100307 regression] plan={plan_path}")
print(f"[100307 regression] first run tasks={len(tasks)} blocked={len(blocked)}")
print(f"[100307 regression] resume tasks={len(resume_tasks)} skipped={len(skipped)}")
if blocked:
    fail("blocked target(s): " + "; ".join(f"{task.get('name')}: {task.get('reason')}" for task in blocked))
PY

echo "[100307 regression] completed"
