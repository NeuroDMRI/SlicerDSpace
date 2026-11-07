#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

IMAGE_NAME="${IMAGE_NAME:-slicerddsurfer-rc-smoke:ubuntu22.04}"
DEFAULT_REPO_URL="$(git -C "$REPO_ROOT" remote get-url origin 2>/dev/null || true)"
REPO_URL="${REPO_URL:-${DEFAULT_REPO_URL:-https://github.com/NeuroDMRI/SlicerDSpace.git}}"
REPO_REF="${REPO_REF:-main}"
HOST_SOFTWARE_ROOT="${HOST_SOFTWARE_ROOT:-/data/chengjin/software}"
HOST_SLICER_LAUNCHER="${HOST_SLICER_LAUNCHER:-$HOST_SOFTWARE_ROOT/run-slicer.sh}"
HOST_MRTRIX_BIN="${HOST_MRTRIX_BIN:-$HOST_SOFTWARE_ROOT/mrtrix3_env/bin}"
HOST_WORK_ROOT="${HOST_WORK_ROOT:-/tmp/slicerddsurfer-rc-smoke}"
DOCKER_NETWORK="${DOCKER_NETWORK:-bridge}"
DOCKER_BUILD_NO_CACHE="${DOCKER_BUILD_NO_CACHE:-0}"
KEEP_WORK="${KEEP_WORK:-0}"
HOST_SSH_DIR="${HOST_SSH_DIR:-$HOME/.ssh}"
MOUNT_SSH="${MOUNT_SSH:-auto}"
HOST_SSH_IDENTITY_FILE="${HOST_SSH_IDENTITY_FILE:-}"
HOST_ASSET_SEED_REPO="${HOST_ASSET_SEED_REPO:-$REPO_ROOT}"
SMOKE_SEED_ASSETS_FROM_HOST="${SMOKE_SEED_ASSETS_FROM_HOST:-auto}"
SMOKE_MODEL_ASSET_ROOT="${SMOKE_MODEL_ASSET_ROOT:-/work/model-assets}"
SMOKE_STAGE="${SMOKE_STAGE:-install}"
if [[ -z "${SMOKE_LFS_MODE:-}" ]]; then
  if [[ "$SMOKE_STAGE" == "quick-gate" ]]; then
    SMOKE_LFS_MODE="profile"
  else
    SMOKE_LFS_MODE="quick"
  fi
fi
SMOKE_ASSET_PROFILE="${SMOKE_ASSET_PROFILE:-quick-gate}"
SMOKE_SUBJECT_ID="${SMOKE_SUBJECT_ID:-100307-release-candidate}"
SMOKE_RUN_NAME="${SMOKE_RUN_NAME:-rc_csd_act}"
# W2a (D1): preprocessing has one route (FSL eddy). It has no fast path, and the quick gate has no FSL, so it is not a default target.
SMOKE_TARGETS="${SMOKE_TARGETS:-validation,segmentation,parcellation,surface,microstructure,tractography,bundles,connectome}"
SMOKE_BUNDLE_BACKEND="${SMOKE_BUNDLE_BACKEND:-all}"
SMOKE_SELECT_STREAMLINES="${SMOKE_SELECT_STREAMLINES:-100000}"
HOST_SUBJECT_INPUT_ROOT="${HOST_SUBJECT_INPUT_ROOT:-/data/chengjin/SlicerDDSurfer-test/100307-release-candidate/input}"
HOST_SUBJECT_CACHE_ROOT="${HOST_SUBJECT_CACHE_ROOT:-$(dirname "$HOST_SUBJECT_INPUT_ROOT")/cache}"
HOST_TRACT_QUERIER_ROOT="${HOST_TRACT_QUERIER_ROOT:-}"

if [[ ! -f "$SCRIPT_DIR/rc-smoke.Dockerfile" ]]; then
  echo "Missing Dockerfile: $SCRIPT_DIR/rc-smoke.Dockerfile" >&2
  exit 2
fi
if [[ ! -x "$HOST_SLICER_LAUNCHER" ]]; then
  echo "HOST_SLICER_LAUNCHER is not executable: $HOST_SLICER_LAUNCHER" >&2
  exit 2
fi
if [[ ! -d "$HOST_MRTRIX_BIN" ]]; then
  echo "HOST_MRTRIX_BIN is not a directory: $HOST_MRTRIX_BIN" >&2
  exit 2
fi
case "$SMOKE_STAGE" in
  install|quick-gate)
    ;;
  *)
    echo "SMOKE_STAGE must be install or quick-gate, got: $SMOKE_STAGE" >&2
    exit 2
    ;;
esac
case "$SMOKE_LFS_MODE" in
  quick|profile|full)
    ;;
  *)
    echo "SMOKE_LFS_MODE must be quick, profile, or full, got: $SMOKE_LFS_MODE" >&2
    exit 2
    ;;
esac
if [[ "$SMOKE_STAGE" == "quick-gate" ]]; then
  if [[ "$SMOKE_LFS_MODE" == "quick" ]]; then
    echo "SMOKE_STAGE=quick-gate requires SMOKE_LFS_MODE=profile or full so model/data assets are available." >&2
    exit 2
  fi
  if [[ ! -d "$HOST_SUBJECT_INPUT_ROOT" ]]; then
    echo "HOST_SUBJECT_INPUT_ROOT is not a directory: $HOST_SUBJECT_INPUT_ROOT" >&2
    exit 2
  fi
  for required in data.nii.gz bvals bvecs nodif_brain_mask.nii.gz; do
    if [[ ! -f "$HOST_SUBJECT_INPUT_ROOT/$required" ]]; then
      echo "Missing quick-gate input: $HOST_SUBJECT_INPUT_ROOT/$required" >&2
      exit 2
    fi
  done
fi
if [[ -n "$HOST_SUBJECT_CACHE_ROOT" && ! -d "$HOST_SUBJECT_CACHE_ROOT" ]]; then
  echo "HOST_SUBJECT_CACHE_ROOT is not a directory: $HOST_SUBJECT_CACHE_ROOT" >&2
  exit 2
fi
if [[ -n "$HOST_TRACT_QUERIER_ROOT" && ! -d "$HOST_TRACT_QUERIER_ROOT" ]]; then
  echo "HOST_TRACT_QUERIER_ROOT is not a directory: $HOST_TRACT_QUERIER_ROOT" >&2
  exit 2
fi
if [[ "$SMOKE_SEED_ASSETS_FROM_HOST" != "0" && "$SMOKE_SEED_ASSETS_FROM_HOST" != "1" && "$SMOKE_SEED_ASSETS_FROM_HOST" != "auto" ]]; then
  echo "SMOKE_SEED_ASSETS_FROM_HOST must be 0, 1, or auto, got: $SMOKE_SEED_ASSETS_FROM_HOST" >&2
  exit 2
fi
if [[ "$DOCKER_BUILD_NO_CACHE" != "0" && "$DOCKER_BUILD_NO_CACHE" != "1" ]]; then
  echo "DOCKER_BUILD_NO_CACHE must be 0 or 1, got: $DOCKER_BUILD_NO_CACHE" >&2
  exit 2
fi
case "$SMOKE_MODEL_ASSET_ROOT" in
  /*)
    ;;
  *)
    echo "SMOKE_MODEL_ASSET_ROOT must be an absolute container path: $SMOKE_MODEL_ASSET_ROOT" >&2
    exit 2
    ;;
esac
if [[ "$SMOKE_SEED_ASSETS_FROM_HOST" == "1" && ! -d "$HOST_ASSET_SEED_REPO" ]]; then
  echo "HOST_ASSET_SEED_REPO is not a directory: $HOST_ASSET_SEED_REPO" >&2
  exit 2
fi

case "$HOST_WORK_ROOT" in
  /tmp/*|/var/tmp/*|"$REPO_ROOT"/.docker-smoke-work*)
    ;;
  *)
    echo "Refusing to clean HOST_WORK_ROOT outside /tmp, /var/tmp, or repo .docker-smoke-work: $HOST_WORK_ROOT" >&2
    exit 2
    ;;
esac

if [[ "$KEEP_WORK" != "1" ]]; then
  if [[ -e "$HOST_WORK_ROOT" ]] && ! rm -rf "$HOST_WORK_ROOT" 2>/dev/null; then
    echo "[rc-smoke] Cleaning root-owned workdir remnants via a disposable ubuntu:22.04 container"
    docker run --rm --network none \
      -e TARGET_BASENAME="$(basename "$HOST_WORK_ROOT")" \
      -v "$(dirname "$HOST_WORK_ROOT"):/host_parent" \
      ubuntu:22.04 \
      sh -c 'rm -rf -- "/host_parent/$TARGET_BASENAME"'
  fi
fi
mkdir -p "$HOST_WORK_ROOT"

echo "[rc-smoke] Building $IMAGE_NAME"
DOCKER_BUILD_ARGS=(
  -f "$SCRIPT_DIR/rc-smoke.Dockerfile"
  -t "$IMAGE_NAME"
)
if [[ "$DOCKER_BUILD_NO_CACHE" == "1" ]]; then
  DOCKER_BUILD_ARGS+=(--no-cache)
fi
docker build "${DOCKER_BUILD_ARGS[@]}" "$SCRIPT_DIR"

echo "[rc-smoke] Running fresh-clone install smoke"
DOCKER_RUN_ARGS=(
  --rm
  --network "$DOCKER_NETWORK" \
  -e REPO_URL="$REPO_URL" \
  -e REPO_REF="$REPO_REF" \
  -e SMOKE_STAGE="$SMOKE_STAGE" \
  -e HOST_SLICER_LAUNCHER="$HOST_SLICER_LAUNCHER" \
  -e HOST_MRTRIX_BIN="$HOST_MRTRIX_BIN" \
  -e SMOKE_LFS_MODE="$SMOKE_LFS_MODE" \
  -e SMOKE_ASSET_PROFILE="$SMOKE_ASSET_PROFILE" \
  -e SMOKE_MODEL_ASSET_ROOT="$SMOKE_MODEL_ASSET_ROOT" \
  -e SMOKE_SUBJECT_ID="$SMOKE_SUBJECT_ID" \
  -e SMOKE_RUN_NAME="$SMOKE_RUN_NAME" \
  -e SMOKE_TARGETS="$SMOKE_TARGETS" \
  -e SMOKE_BUNDLE_BACKEND="$SMOKE_BUNDLE_BACKEND" \
  -e SMOKE_SELECT_STREAMLINES="$SMOKE_SELECT_STREAMLINES" \
  -e GIT_TERMINAL_PROMPT=0 \
  -e HOME=/work/home \
  -e XDG_CACHE_HOME=/work/home/.cache \
  -e XDG_CONFIG_HOME=/work/home/.config \
  -v "$HOST_SOFTWARE_ROOT:$HOST_SOFTWARE_ROOT:ro" \
  -v "$HOST_WORK_ROOT:/work" \
)
if [[ "$SMOKE_STAGE" == "quick-gate" ]]; then
  DOCKER_RUN_ARGS+=(
    -v "$HOST_SUBJECT_INPUT_ROOT:/inputs/$SMOKE_SUBJECT_ID/input:ro"
  )
fi
if [[ "$SMOKE_STAGE" == "quick-gate" && -d "$HOST_SUBJECT_CACHE_ROOT" ]]; then
  echo "[rc-smoke] Mounting host subject cache read-only: $HOST_SUBJECT_CACHE_ROOT"
  DOCKER_RUN_ARGS+=(
    -v "$HOST_SUBJECT_CACHE_ROOT:/seed_subject_cache:ro"
    -e CONTAINER_SUBJECT_CACHE_SEED_ROOT=/seed_subject_cache
  )
fi
if [[ "$SMOKE_SEED_ASSETS_FROM_HOST" == "1" || ( "$SMOKE_SEED_ASSETS_FROM_HOST" == "auto" && -d "$HOST_ASSET_SEED_REPO" ) ]]; then
  echo "[rc-smoke] Mounting host asset seed repo read-only: $HOST_ASSET_SEED_REPO"
  DOCKER_RUN_ARGS+=(
    -v "$HOST_ASSET_SEED_REPO:/seed_repo:ro"
    -e CONTAINER_ASSET_SEED_REPO=/seed_repo
  )
fi
if [[ -n "$HOST_TRACT_QUERIER_ROOT" ]]; then
  DOCKER_RUN_ARGS+=(
    -v "$HOST_TRACT_QUERIER_ROOT:/external/tract_querier:ro"
    -e CONTAINER_TRACT_QUERIER_ROOT=/external/tract_querier
  )
fi
if [[ "$MOUNT_SSH" == "1" || ( "$MOUNT_SSH" == "auto" && "$REPO_URL" == git@* && -d "$HOST_SSH_DIR" ) ]]; then
  if [[ -z "$HOST_SSH_IDENTITY_FILE" ]]; then
    for candidate in "$HOST_SSH_DIR/id_ed25519" "$HOST_SSH_DIR/id_rsa" "$HOST_SSH_DIR/id_ecdsa"; do
      if [[ -f "$candidate" ]]; then
        HOST_SSH_IDENTITY_FILE="$candidate"
        break
      fi
    done
  fi
  if [[ ! -f "$HOST_SSH_IDENTITY_FILE" ]]; then
    echo "Could not find SSH identity for private Git clone. Set HOST_SSH_IDENTITY_FILE." >&2
    exit 2
  fi
  SSH_IDENTITY_BASENAME="$(basename "$HOST_SSH_IDENTITY_FILE")"
  echo "[rc-smoke] Mounting SSH credentials read-only for private Git clone: $HOST_SSH_DIR"
  DOCKER_RUN_ARGS+=(
    -v "$HOST_SSH_DIR:/ssh:ro"
    -e GIT_SSH_COMMAND="ssh -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/work/known_hosts -i /ssh/$SSH_IDENTITY_BASENAME"
  )
fi

docker run "${DOCKER_RUN_ARGS[@]}" \
  "$IMAGE_NAME" \
  bash -lc '
set -euo pipefail
mkdir -p /work/home /work/src /work/install /work/logs "$SMOKE_MODEL_ASSET_ROOT"
echo "[container] repo=$REPO_URL ref=$REPO_REF"
echo "[container] stage=$SMOKE_STAGE"
echo "[container] slicer=$HOST_SLICER_LAUNCHER"
echo "[container] mrtrix=$HOST_MRTRIX_BIN"
echo "[container] lfs_mode=$SMOKE_LFS_MODE"
echo "[container] asset_profile=$SMOKE_ASSET_PROFILE"
echo "[container] model_asset_root=$SMOKE_MODEL_ASSET_ROOT"
test -x "$HOST_SLICER_LAUNCHER"
test -x "$HOST_MRTRIX_BIN/mrconvert"
if [[ -d /work/src/SlicerDSpace/.git ]]; then
  echo "[container] Reusing existing checkout: /work/src/SlicerDSpace"
else
  GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none "$REPO_URL" /work/src/SlicerDSpace
fi
cd /work/src/SlicerDSpace
if [[ -f .git/index.lock ]]; then
  echo "[container] Removing stale git lock: .git/index.lock"
  rm -f .git/index.lock
fi
FETCHED_REF=0
if [[ -d .git ]]; then
  git remote set-url origin "$REPO_URL"
  if git fetch --depth=1 origin "$REPO_REF" || git fetch origin "$REPO_REF"; then
    FETCHED_REF=1
  fi
fi
if [[ "$FETCHED_REF" == "1" ]]; then
  git checkout --detach FETCH_HEAD
else
  git checkout "$REPO_REF"
fi
git lfs install --local
case "$SMOKE_LFS_MODE" in
  quick)
    echo "[container] Quick LFS mode: auditing pointer metadata without downloading large objects"
    git lfs ls-files -s | tee /work/logs/git_lfs_ls_files.txt
    python3 scripts/release_audit.py | tee /work/logs/release_audit.txt
    ;;
  profile)
    echo "[container] Profile asset mode: downloading only assets required by profile=$SMOKE_ASSET_PROFILE"
    python3 -u scripts/download_model_assets.py --profile "$SMOKE_ASSET_PROFILE" \
      --asset-root "$SMOKE_MODEL_ASSET_ROOT" --print-plan \
      2>&1 | tee /work/logs/model_assets_plan.json
    {
      if [[ -n "${CONTAINER_ASSET_SEED_REPO:-}" && -d "${CONTAINER_ASSET_SEED_REPO}" ]]; then
        python3 - "$SMOKE_ASSET_PROFILE" "$CONTAINER_ASSET_SEED_REPO" "$SMOKE_MODEL_ASSET_ROOT" "/work/src/SlicerDSpace" <<'PY'
import json
import shutil
import sys
from pathlib import Path

profile, seed_root_arg, asset_root_arg, repo_root_arg = sys.argv[1:5]
repo_root = Path(repo_root_arg)
seed_root = Path(seed_root_arg)
asset_root = Path(asset_root_arg)
manifest = json.loads((repo_root / "models" / "assets_manifest.json").read_text(encoding="utf-8"))
profiles = manifest["profiles"]
seen = set()
required_paths = []

def visit(name: str) -> None:
    if name in seen:
        return
    seen.add(name)
    profile_data = profiles[name]
    alias = profile_data.get("alias_for")
    if alias:
        visit(str(alias))
        return
    for child in profile_data.get("include_profiles", []):
        visit(str(child))
    required_paths.extend(str(value) for value in profile_data.get("required_paths", []))

visit(profile)
required_paths = list(dict.fromkeys(required_paths))
pointer_header = b"version https://git-lfs.github.com/spec/v1"
seeded = 0
skipped = 0
for relative in required_paths:
    source = seed_root / relative
    if not source.is_file():
        skipped += 1
        continue
    with source.open("rb") as stream:
        if stream.read(256).startswith(pointer_header):
            skipped += 1
            continue
    target = asset_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    seeded += 1

print(f"[model-assets] Seeded {seeded} required file(s) from {seed_root}")
print(f"[model-assets] Seed skipped {skipped} required file(s)")
PY
      fi
      if python3 scripts/download_model_assets.py --profile "$SMOKE_ASSET_PROFILE" \
        --asset-root "$SMOKE_MODEL_ASSET_ROOT" --verify-only; then
        echo "[model-assets] Seeded assets already satisfied profile=$SMOKE_ASSET_PROFILE."
      else
        GIT_LFS_PROGRESS=/work/logs/git-lfs-progress.log \
          python3 -u scripts/download_model_assets.py --profile "$SMOKE_ASSET_PROFILE" \
            --asset-root "$SMOKE_MODEL_ASSET_ROOT"
      fi
    } 2>&1 | tee /work/logs/model_assets_download.log
    ;;
  full)
    echo "[container] Full asset mode: downloading all manifest assets outside the code checkout"
    python3 -u scripts/download_model_assets.py --profile all \
      --asset-root "$SMOKE_MODEL_ASSET_ROOT" \
      2>&1 | tee /work/logs/model_assets_download.log
    ;;
esac
case "$SMOKE_STAGE" in
  install)
    python3 scripts/stage_slicer_extension.py \
      --install-root /work/install \
      --clean \
      --link-mode copy | tee /work/logs/stage_slicer_extension.json
    MODULE_DIR=/work/install/SlicerDSpace/lib/Slicer-5.10/qt-scripted-modules
    test -f "$MODULE_DIR/SlicerDSpace.py"
    env \
      SLICERDSPACE_EXTENSION_MODULE_DIR="$MODULE_DIR" \
      PYTHONUNBUFFERED=1 \
      PATH=/usr/bin:/bin \
      stdbuf -oL -eL xvfb-run -a -s "-screen 0 1280x1024x24" \
      "$HOST_SLICER_LAUNCHER" --no-splash \
      --additional-module-paths "$MODULE_DIR" \
      --python-script /work/src/SlicerDSpace/Testing/Python/slicer_extension_install_smoke.py \
      2>&1 | tee /work/logs/slicer_extension_install_smoke.log
    echo "[container] RC_SMOKE_OK"
    ;;
  quick-gate)
    SUBJECT_ROOT="/work/subject/$SMOKE_SUBJECT_ID"
    OUTPUT_ROOT="/work/outputs/$SMOKE_SUBJECT_ID"
    mkdir -p "$SUBJECT_ROOT" "$OUTPUT_ROOT"
    rm -f "$SUBJECT_ROOT/input"
    ln -s "/inputs/$SMOKE_SUBJECT_ID/input" "$SUBJECT_ROOT/input"
    if [[ -n "${CONTAINER_SUBJECT_CACHE_SEED_ROOT:-}" && -d "${CONTAINER_SUBJECT_CACHE_SEED_ROOT}" ]]; then
      mkdir -p "$SUBJECT_ROOT/cache"
      python3 - "$CONTAINER_SUBJECT_CACHE_SEED_ROOT" "$SUBJECT_ROOT/cache" <<'PY'
import shutil
import sys
from pathlib import Path

seed_root = Path(sys.argv[1])
target_root = Path(sys.argv[2])
seeded = 0
for relative in (
    Path("native_dti/wls"),
    Path("registered_dti/100HCP-population-mean-T2__wls"),
):
    source = seed_root / relative
    if not source.exists():
        continue
    destination = target_root / relative
    if destination.exists():
        continue
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    seeded += 1
print(f"[quick-gate] Seeded {seeded} cache subtree(s) from {seed_root}")
PY
    fi
    QUICK_GATE_ARGS=(
      /work/src/SlicerDSpace/run_subject_pipeline.py
      --subject-root "$SUBJECT_ROOT"
      --subject-id "$SMOKE_SUBJECT_ID"
      --output-root "$OUTPUT_ROOT"
      --targets "$SMOKE_TARGETS"
      --tractography-backend csd_act
      --bundle-backend "$SMOKE_BUNDLE_BACKEND"
      --microstructure-stages dti
      --run-name "$SMOKE_RUN_NAME"
      --model-assets-dir "$SMOKE_MODEL_ASSET_ROOT"
      --mrtrix-bin-dir "$HOST_MRTRIX_BIN"
      --select-streamlines "$SMOKE_SELECT_STREAMLINES"
      --execute
      --print-manifest-only
    )
    if [[ -n "${CONTAINER_TRACT_QUERIER_ROOT:-}" ]]; then
      QUICK_GATE_ARGS+=(--tract-querier-root "$CONTAINER_TRACT_QUERIER_ROOT")
    fi
    env PYTHONUNBUFFERED=1 stdbuf -oL -eL \
      "$HOST_SLICER_LAUNCHER" --launch PythonSlicer "${QUICK_GATE_ARGS[@]}" \
      2>&1 | tee /work/logs/100307_quick_gate.log
    MANIFEST_PATH="$OUTPUT_ROOT/00_run_metadata/${SMOKE_RUN_NAME}_orchestration_manifest.json"
    python3 - "$MANIFEST_PATH" <<PY
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
if not manifest_path.is_file():
    raise SystemExit(f"missing quick-gate manifest: {manifest_path}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
tasks = manifest.get("tasks", [])
by_name = {task.get("name"): task for task in tasks}
failed = [task for task in tasks if task.get("status") == "failed"]
required_statuses = {
    "validation": {"completed", "passed"},
    "segmentation.ddseg": {"completed", "skipped"},
    "parcellation.ddparcel": {"completed", "skipped"},
    "surface.ddsurfer": {"completed", "skipped"},
    "microstructure.dti": {"completed", "skipped"},
    "tractography.csd_act": {"completed", "skipped"},
    "bundles.wmql": {"completed", "skipped"},
    "connectome.streamline_count": {"completed", "skipped"},
    "connectome.sift2_weighted": {"completed", "skipped"},
}
missing = [name for name in required_statuses if name not in by_name]
not_completed = [
    "{}:{}:{}".format(name, by_name[name].get("status"), by_name[name].get("reason", ""))
    for name, allowed_statuses in required_statuses.items()
    if name in by_name and by_name[name].get("status") not in allowed_statuses
]
print(f"[quick-gate] manifest={manifest_path}")
print(f"[quick-gate] tasks={len(tasks)} failed={len(failed)} required_not_completed={len(not_completed)}")
if failed:
    raise SystemExit("[quick-gate] failed task(s): " + "; ".join("{}:{}".format(task.get("name"), task.get("reason", "")) for task in failed))
if missing:
    raise SystemExit("[quick-gate] missing required task(s): " + ", ".join(missing))
if not_completed:
    raise SystemExit("[quick-gate] required task(s) not completed: " + "; ".join(not_completed))
PY
    echo "[container] RC_QUICK_GATE_OK"
    ;;
esac
'

echo "[rc-smoke] Logs: $HOST_WORK_ROOT/logs"
