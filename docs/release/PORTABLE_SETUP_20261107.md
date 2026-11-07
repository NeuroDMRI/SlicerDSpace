<div align="center">

# Portable Setup and Default Methods, 7 November 2026

[Project](../../README.md) | [Setup](../setup/MANAGED_SETUP.md) | [Platforms](../setup/SUPPORT_MATRIX.md) | [中文](PORTABLE_SETUP_20261107.zh-CN.md)

</div>

---

This note says what changed for users. It describes the code of the `r5-native-tensor` branch.

## Deploy Note

After this update, every Linux deployment regenerates the brain masks and the
preprocessing (eddy included) once. The reason: the mask signature now holds the
mask route, so the saved results of the older version no longer match.
Plan the compute time for the first run after the update.

## What Changed: Default Methods

- The DTI fit runs inside Slicer with the Teem library. SlicerDMRI is no longer
  needed for the tensor fit. Setup asks for SlicerDMRI only when you tick WMA.
- The DTI products now include a colour FA map (`<id>-dti-DEC.nrrd`).
- DKI uses the Python MK-Curve by default.
- The connectome backend is `auto`: MRtrix `tck2connectome` for a `.tck` source
  with a streamline-count metric, Python for any other source or metric.
- An optional iFW stage is available (it needs the echo time). It also runs Free Water and DDSeg.
- Each run writes `00_run_metadata/run_record.json`. It lists the methods, the
  brain-mask provenance and the accepted licences. A re-run keeps the previous
  record as `run_record-<started_at>.json`.

## What Changed: Setup on Linux, macOS and Windows

- Tick the products you need. The plugin adds the prerequisites by itself.
- One dialog lists every download with its size and licence. Press
  **Accept licences and install** once. The command line does the same with
  `PythonSlicer scripts/setup_selected.py --targets <targets> --execute --accept-licences`.
- After the install, a self-check runs for each feature. A failed self-check
  blocks "ready".
- dcm2niix: the plugin installs a pinned dcm2niix wheel into Slicer's Python.
  If the wheel, or a dcm2niix path that you set yourself, fails its check, the
  setup is not ready. If a dcm2niix found only on `PATH` fails, you get a
  warning. NIfTI input does not need dcm2niix.
- macOS installs MRtrix3 3.0.8 and FSL as managed runtimes (FSL needs macOS 14.5 or newer).
- Windows runs MRtrix3 and FSL in WSL 2 Ubuntu. The brain mask and the Python
  stages run in Windows Slicer's Python.
- Setup commands use `PythonSlicer` (Slicer's own Python), not `python3`.

## What Changed: PNL Brain Mask

- New setting `pnl_mask_route`: `auto`, `managed` or `native`
  (GUI Advanced settings, or `--pnl-mask-route`).
- On Linux, `auto` is `managed` (conda runtime with ANTs 2.3.0).
- On macOS and Windows, `auto` is `native`: SimpleITK registration and PyTorch
  in Slicer's Python. The GUI labels it "native (experimental)".
- The native route is experimental. On 2 of 24 test cases its masks did not meet
  the end-to-end parity rule against the managed route.
- The mask receipt and the run record name the route that made the mask.
- The PNL licence is in the licence dialog and in
  [docs/licenses/pnl_brain_mask.md](../licenses/pnl_brain_mask.md).

## Old Runtime Folders

When setup replaces an existing macOS MRtrix3 runtime, it renames the old folder
to `mrtrix-3.0.8-macos.replaced-<ns>` in the runtimes folder (`<model-assets parent>/runtimes`;
on macOS `~/Library/Application Support/SlicerDSpace/runtimes`). Each such folder
is about 169 MB. The plugin never deletes it. After a successful run, you can
delete these folders by hand.
