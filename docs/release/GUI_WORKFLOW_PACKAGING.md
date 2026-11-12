# GUI workflow packaging follow-up

The Windows x64 v0.1.1 archive inspected locally contains native MRtrix3
3.0.8, FSL 2602.0 preprocessing, and the surface postprocessing runtime.
It contains no model weight files with the checked extensions (.pt, .pth,
.ckpt, .onnx) and no UKFTractography.exe or DWIConvert.exe. Model assets
are prepared on demand. Algorithms that require SlicerDMRI or those
executables still depend on the selected environment.

- Inventory every algorithm's required assets and runtimes from a fresh
  installation with networking disabled.
- Confirm redistribution terms for each model, atlas, and external tool before
  embedding it. Record license notices, checksums, and installed size by platform.
- Verify Windows x64, Linux x64, and Apple Silicon macOS release archives
  independently. Record remaining setup steps per algorithm.
