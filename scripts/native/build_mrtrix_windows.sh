#!/usr/bin/env bash
# Maintainer/CI ONLY. Run in a real MSYS2 MINGW64 shell, not WSL.
# setup-msys2 action installs all build tools on the CI runner, not user PCs.
set -euo pipefail
if [[ "${MSYSTEM:-}" != "MINGW64" ]]; then
  echo 'Expected MSYSTEM=MINGW64. Do not run this in WSL/MSYS or plain PowerShell.' >&2
  exit 2
fi
source_root="${1:-mrtrix3}"
output_root="${2:-dist/native-candidates}"
if [[ ! -f "$source_root/configure" || ! -f "$source_root/build" ]]; then
  echo 'Expected checked out, pinned MRtrix3 source; refusing unpinned download.' >&2
  exit 2
fi
# Pin the commit behind the upstream 3.0.8 tag.
if [[ "$(git -C "$source_root" rev-parse HEAD 2>/dev/null)" != "464173c5294e021b2d508e557085d90c09d642fd" ]]; then
  echo 'Expected the upstream MRtrix3 3.0.8 checkout.' >&2
  exit 2
fi
pushd "$source_root" >/dev/null
./configure -nogui
./build
popd >/dev/null
# Stage the native CSD/ACT commands, Dhollander script and their runtime dependencies.
# MinGW builds place mrtrix.dll beside the upstream executables.
dll_args=(--dll-search "$source_root/bin" --dll-search /mingw64/bin --dll-search /usr/bin)
if [[ -d "$source_root/lib" ]]; then
  dll_args+=(--dll-search "$source_root/lib")
fi
python scripts/native/prepare_mrtrix_candidate.py \
  --source "$source_root" --system Windows --machine x86_64 \
  "${dll_args[@]}" \
  --objdump /mingw64/bin/objdump.exe --output "$output_root"
