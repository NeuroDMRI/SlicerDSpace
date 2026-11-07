#!/bin/bash
# Voxel-by-voxel parity: SlicerDMRI CLIs (reference) vs shared/teem_dti.py (Slicer core Teem).
# Required env: SLICER_HOME (Slicer install root), DMRI_CLI (SlicerDMRI cli-modules folder),
#               DWI, BVAL, BVEC (subject inputs), WORK (output folder).
# Optional env: CORES (taskset list, default: no pinning), THREADS (teem_dti threads, default: auto),
#               METHODS (default "LS WLS").
# Writes WORK/compare_<method>.json and WORK/logs/*.log (with /usr/bin/time -v when available).
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=$SLICER_HOME/bin/PythonSlicer
PIN=(); [ -n "${CORES:-}" ] && PIN=(taskset -c "$CORES")
TIME=(); [ -x /usr/bin/time ] && TIME=(/usr/bin/time -v)
THREAD_ARGS=(); [ -n "${THREADS:-}" ] && THREAD_ARGS=(--threads "$THREADS")
OPS="FractionalAnisotropy Trace MeanDiffusivity MinEigenvalue MidEigenvalue MaxEigenvalue"
mkdir -p "$WORK"/in "$WORK"/ref "$WORK"/new "$WORK"/logs
cd "$WORK"
[ -s in/DWI.nhdr ] || "$PY" "$REPO/scripts/teem_dti_parity/make_dwi_nhdr.py" "$DWI" "$BVAL" "$BVEC" in > logs/make_nhdr.log 2>&1
for M in ${METHODS:-LS WLS}; do
  m=$(echo "$M" | tr A-Z a-z)
  if [ ! -s "ref/DTI_$m.nrrd" ]; then
    ${TIME[@]+"${TIME[@]}"} ${PIN[@]+"${PIN[@]}"} "$SLICER_HOME/Slicer" --launch "$DMRI_CLI/DWIToDTIEstimation" --enumeration "$M" \
      in/DWI.nhdr "ref/DTI_$m.nrrd" "ref/b0_$m.nrrd" > "logs/ref_$m.log" 2>&1
    for OP in $OPS; do
      ${TIME[@]+"${TIME[@]}"} ${PIN[@]+"${PIN[@]}"} "$SLICER_HOME/Slicer" --launch "$DMRI_CLI/DiffusionTensorScalarMeasurements" \
        --enumeration "$OP" "ref/DTI_$m.nrrd" "ref/${OP}_$m.nrrd" >> "logs/ref_${m}_scalars.log" 2>&1
    done
  fi
  ARGS=(); for OP in $OPS; do ARGS+=(--scalar "$OP=new/${OP}_$m.nrrd"); done
  ${TIME[@]+"${TIME[@]}"} ${PIN[@]+"${PIN[@]}"} "$PY" "$REPO/shared/teem_dti.py" estimate --enumeration "$M" ${THREAD_ARGS[@]+"${THREAD_ARGS[@]}"} \
    --receipt "new/receipt_$m.json" "${ARGS[@]}" in/DWI.nhdr "new/DTI_$m.nrrd" "new/b0_$m.nrrd" > "logs/new_$m.log" 2>&1
  PAIRS=("ref/DTI_$m.nrrd::new/DTI_$m.nrrd" "ref/b0_$m.nrrd::new/b0_$m.nrrd")
  for OP in $OPS; do PAIRS+=("ref/${OP}_$m.nrrd::new/${OP}_$m.nrrd"); done
  "$PY" "$REPO/scripts/teem_dti_parity/compare_nrrd.py" "compare_$m.json" "${PAIRS[@]}" > "logs/compare_$m.log" 2>&1
done
echo DONE > logs/DONE
