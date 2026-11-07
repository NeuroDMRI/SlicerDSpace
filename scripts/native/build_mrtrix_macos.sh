#!/usr/bin/env bash
# Maintainer/CI ONLY. Uses a published, pinned macOS universal MRtrix 3.0.8 release.
set -euo pipefail
out="${1:-dist/native-candidates}"
mkdir -p "$out"
root="$(mktemp -d)"
trap 'rm -rf "$root"' EXIT
asset="$root/mrtrix3-macos-3.0.8.tar.gz"
curl --fail --location --retry 3 --output "$asset" \
  'https://github.com/MRtrix3/mrtrix3/releases/download/3.0.8/mrtrix3-macos-3.0.8.tar.gz'
# The pinned upstream digest is already part of shared/managed_runtimes.py.
printf '%s  %s\n' '91fb80ffc9f26136d094e4a5310d051f52db9cb5faa35ad3f3e6156598090c39' "$asset" | shasum -a 256 -c -
# Inspect members before extraction; scripts/native/prepare_mrtrix_candidate.py
# further checks every required executable is present.
python - "$asset" "$root" <<'PY'
from pathlib import Path, PurePosixPath
import sys, tarfile
archive, dest = Path(sys.argv[1]), Path(sys.argv[2])
with tarfile.open(archive, 'r:gz') as t:
    for m in t:
        p = PurePosixPath(m.name)
        if p.is_absolute() or '..' in p.parts or not (m.isdir() or m.isfile()):
            raise ValueError(f'Unsafe upstream tar member: {m.name}')
    t.extractall(dest, filter='data')
PY
if [[ ! -d "$root/mrtrix3" ]]; then
  echo 'Upstream archive layout unexpected (missing mrtrix3/).' >&2
  exit 2
fi
# Candidate packaging is architecture-labelled; do not claim the other
# architecture until it has been run on that host as well.
arch="$(uname -m)"
if [[ "$arch" == "aarch64" ]]; then arch=arm64; fi
python scripts/native/prepare_mrtrix_candidate.py \
  --source "$root/mrtrix3" --system Darwin --machine "$arch" --output "$out"
