# Native tensor fit parity: `shared/teem_dti.py` vs SlicerDMRI

Date: 2026-11-07. Host: fan93 (Linux x86_64), 3D Slicer 5.10.0 r34045, Teem 1.12.0 from Slicer core.
Reference: SlicerDMRI a40200b (Extension Manager build for r34045) `DWIToDTIEstimation` and `DiffusionTensorScalarMeasurements`.

## Data and commands

- Subject: HCP 100307 (open access), DWI 145 x 174 x 145 x 288, float32, max b = 3010 s/mm^2. `DWI.nhdr` written by the plugin's `nhdr_write`.
- Runner: `scripts/teem_dti_parity/run_parity.sh` (cores 96-111 pinned with `taskset`, `teem_dti.py --threads 16`).
- Reference: `Slicer --launch DWIToDTIEstimation --enumeration LS|WLS`, then six `DiffusionTensorScalarMeasurements` runs.
- Candidate: one `teem_dti.py estimate --enumeration LS|WLS --scalar ...` run.
- Comparison: `scripts/teem_dti_parity/compare_nrrd.py` (file and payload SHA-256, header text, RAS-to-IJK and measurement frame, value equality, NaN masks, ULP).

## Result

| Method | Files compared | Byte-identical files | Identical voxels per file | Max ULP | Failed voxels |
|---|---|---|---|---|---|
| LS | 8 (tensor, baseline, FA, Trace, MD, min/mid/max eigenvalue) | 8 | 3,658,350 / 3,658,350 | 0 | 0 |
| WLS | 8 | 8 | 3,658,350 / 3,658,350 | 0 | 0 |

Records: `compare_ls.json`, `compare_wls.json` (per-file comparison), `receipt_ls.json`, `receipt_wls.json` (candidate receipts).

## Timing (same 16 pinned cores)

| Method | Reference tensor CLI wall | Candidate wall (tensor + 6 scalars) | Candidate read / fit / scalars |
|---|---|---|---|
| LS | 40.7 s (208 % CPU) | 45.7 s | 37.5 s / 4.0 s / 1.4 s |
| WLS | 7 min 42.8 s (1352 % CPU) | 7 min 13.2 s (1446 % CPU) | 37.0 s / 391.7 s / 1.4 s |

The reference times exclude its six scalar CLI runs (not timed); each of them re-reads the tensor file. Peak RSS was about 8.0 GiB (reference) and 8.1 GiB (candidate).

## Self-check on other platforms

- macOS, x86_64 Slicer 5.12.4 under Rosetta 2 on Apple Silicon (`libteem.1.dylib`, Teem 1.12.0): `teem_dti.py selfcheck` status ok (LS error 3.4e-11, WLS error 1.4e-11 on a synthetic 12-direction, two-shell signal plus b0).
- Windows: pending (the Teem library file name has not been confirmed on a Windows Slicer).
