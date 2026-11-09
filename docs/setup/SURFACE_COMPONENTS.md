<div align="center">

# Surface Postprocessing Components

**Optional native tools, without a full FreeSurfer installation**

[Project](../../README.md) | [Setup](QUICKSTART.md) | [Platforms](SUPPORT_MATRIX.md) | [中文](SURFACE_COMPONENTS.zh-CN.md)

</div>

DDSurfer snapshot `8a8699c0fda32be7e0a3cbca2903680b9f0c950b` supports surface
postprocessing using seven locally compiled original FreeSurfer commands.
This is native-code reuse, not a Python rewrite, a new numerical method or a
replacement for the complete FreeSurfer suite. Postprocessing remains optional.

---

## Choose A Route

| Configuration | Behavior |
| --- | --- |
| Surface component path supplied | Use that verified runtime, without discovering the full suite |
| No component path; existing FreeSurfer detected | Reuse the existing installation |
| Neither supplied | Look for the upstream recipe-matched user cache |

Use **Environment and installation > More options > advanced paths** to set
the optional component directory and your personal activation file. Keep the
component path empty to use an existing FreeSurfer installation instead.
`SLICERDSPACE_SURFACE_RUNTIME` also selects a local runtime; `FS_LICENSE`
selects a personal activation file. An explicitly selected incomplete/corrupt
runtime is an error, not a reason to fall back silently.

The current indexed native `lh/rh.white.obj` and `lh/rh.pial.obj` route is
supported. The historical STL/external-shell route is not converted to a
seven-command backend: it has additional dependencies and a different contract.

---

## First Preparation

Select surface postprocessing and click **Prepare selected analysis environment**.
The plugin looks for a configured file, `FS_LICENSE`/`FREESURFER_LICENSE`,
`.licenses/freesurfer/license.txt`, `.freesurfer/license.txt` or `license.txt` in your
home directory, and a license in the selected FreeSurfer installation. An invalid
explicit path is not silently replaced. If missing, **Get a free license** opens
official registration; **Import license** selects the file you receive. The plugin
saves its path without copying, uploading or bundling the private file. Keep it at
that location for later runs. Cancel does not start installation. This prompt is
limited to surface postprocessing. Readable text is only a preliminary check;
the original commands validate activation.

**Check never installs. The GUI preparation route never compiles sources.**
In **`v0.1.1`**, the verified 7-command native Surface Runtime (`0.1.0`) and atlas templates are **bundled directly** inside the unified platform installers:
- **Windows x64**: `SlicerDSpace-v0.1.1-Windows-x64.zip` (`bundled-runtimes/surface-runtime-windows-x86_64`)
- **Linux x86_64**: `SlicerDSpace-v0.1.1-Linux-x64.tar.gz` (`bundled-runtimes/surface-runtime-linux-x86_64`)

They are automatically discovered on first launch and verified via `tree_sha256` (with automatic repair from the release archive if deleted or corrupted). Neither bundle includes a personal FreeSurfer activation key (`activation_key_included: false`); your own free FreeSurfer license (`FS_LICENSE`) remains required at runtime.

For standalone/offline cache import (`<asset-root>/runtime/surface/`), the archive names by platform are:

| Platform | Bundled / Offline archive | Status |
| --- | --- | --- |
| Windows x86_64 | Bundled in `SlicerDSpace-v0.1.1-Windows-x64.zip` (`surface-runtime-windows-x86_64`) | **Verified native in 3D Slicer 5.10.0** |
| Linux x86_64 | Bundled in `SlicerDSpace-v0.1.1-Linux-x64.tar.gz` (`surface-runtime-linux-x86_64`) | **Verified native in 3D Slicer 5.10.0** |
| macOS 15 or later, Intel | `surface-runtime-darwin-x86_64.tar.gz` | Offline candidate; scientific comparison pending |
| macOS 15 or later, Apple Silicon | `surface-runtime-darwin-arm64.tar.gz` | Offline candidate; scientific comparison pending |

If the archive is missing, the preparation dialog lists the runtime as a
manual item.

Install an offline bundle after reviewing its component/data terms:

```bash
PythonSlicer scripts/setup_surface_runtime.py --execute --accept-terms \
  --asset-root /path/to/assets --offline
```

Without `--execute`, the command only prints its plan. Existing corrupt or
differently built prefixes are not overwritten. The default destination is the
upstream recipe-addressed user cache; `--prefix` selects another empty location.

On Linux, maintainers can use `--build-from-source --jobs 4`. This expert route
requires Git, C/C++ development tools and CMake >=3.24,<4. The upstream builder
and numerical source remain unchanged. `scripts/package_surface_runtime.py`
packages a verified matching build without an activation key. Platform, source
revision, build recipe, file hashes and archive size are recorded in
`models/runtime_bundles.json`. A bundle is enabled for online download only after
it is published and its distribution permissions are cleared.

---

## Run And Verify

```bash
PythonSlicer run_subject_pipeline.py --subject-root /data/sub-01 --subject-id sub-01 \
  --targets surface_postprocess --postprocess-runtime /data/local-components \
  --postprocess-license-file /data/private/freesurfer-license.txt --execute
```

Use the same subject root containing the reconstructed native OBJ masters, or
provide `--postprocess-native-dir`. `--postprocess-freesurfer-home` remains
available for the full-installation route. The seven surface commands do not supply masking.

The component manifest, source revision, library availability and checksums are
verified in an isolated worker. The activation precheck tests file presence,
not a license grant; the original tools still perform their activation checks.
Output geometry, morphometry and annotation integrity are checked separately.
QC success does not replace anatomical review. Runtime identity is recorded in
the manifest and cache signature; switching an explicit runtime prevents reuse
of the previous runtime's completion state.

The Apple Silicon candidate was built on macOS 15.8.1 from the pinned
FreeSurfer 7.4.1 source. The build passed the upstream Apple Silicon flags
`-DARM64 -DPNG_ARM_NEON_OPT=0` in both `CFLAGS` and `CXXFLAGS` and set
`MACOSX_DEPLOYMENT_TARGET=15.0`. All seven binaries were arm64 and passed
`--version` in a clean process environment. A fresh offline installation
repeated all seven checks. This candidate was built without OpenMP. Scientific
comparison on macOS remains pending.

On Windows x86_64 (`v0.1.1`), the 7 native FreeSurfer surface postprocessing executables (`mris_convert.exe`, `mris_inflate.exe`, `mris_curv.exe`, `mris_smooth.exe`, `mris_sphere.exe`, `mris_register.exe`, `mris_ca_label.exe`, `mris_anatomical_stats.exe`) have been ported, built, bundled in `surface-runtime-windows-x86_64`, and verified on real data in 3D Slicer 5.10.0 against the Linux reference (`Activation Key Included: false`; personal FreeSurfer license required at runtime).

---

## Attribution

The FreeSurfer commands and atlases retain the
[FreeSurfer agreement](../../licenses/FreeSurfer.txt) and third-party terms.
The pinned backend also includes its [data notices](../../anatomy/surface/ddsurfer_backend/postprocessing/assets/LICENSE.txt).
Personal activation files are never included in packages, model mirrors or
automatic downloads. See [implementation provenance](../architecture/IMPLEMENTATION_PROVENANCE.md).
