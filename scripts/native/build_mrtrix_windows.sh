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
pushd "$source_root" >/dev/null
./configure -nogui
./build
popd >/dev/null
# Include only the C++ executable closure. Python scripts and FSL are not certified.
dll_args=(--dll-search /mingw64/bin --dll-search /usr/bin)
if [[ -d "$source_root/lib" ]]; then
  dll_args+=(--dll-search "$source_root/lib")
fi
python scripts/native/prepare_mrtrix_candidate.py \
  --source "$source_root" --system Windows --machine x86_64 \
  "${dll_args[@]}" \
  --objdump /mingw64/bin/objdump.exe --output "$output_root"
