<div align="center">

# Download Scope

**Separate plugin installation, lightweight source and research records**

[Project](../../README.md) | [Quick Start](QUICKSTART.md) | [Documentation](../README.md) | [中文](DOWNLOADS.zh-CN.md)

</div>

Running the plugin does not require website source, historical performance
reports or tests. Keep those in the complete repository for development,
auditing and reproducibility, not as installation prerequisites.
Release `v0.1.1` provides unified platform archives on GitHub Releases (`SlicerDSpace-v0.1.1-Windows-x64.zip`, `SlicerDSpace-v0.1.1-Linux-x64.tar.gz`, and `SlicerDSpace-v0.1.1-macOS-arm64.tar.gz`) for local installation in **3D Slicer 5.10.0** via **Extensions Manager > Install Extensions > Install Extension from File**; online Slicer Extensions Catalog listing remains a separate proposal, and physical Intel (`x86_64`) Mac hardware remains unverified.

---

## Choose A Download

| Purpose | Recommended route | Contents |
| --- | --- | --- |
| Use SlicerDSpace on Windows, Linux, or Apple Silicon macOS | `SlicerDSpace-v0.1.1-Windows-x64.zip`, `SlicerDSpace-v0.1.1-Linux-x64.tar.gz`, or `SlicerDSpace-v0.1.1-macOS-arm64.tar.gz` from GitHub Releases (`Install Extension from File` in Slicer 5.10.0) | Extension runtime, UI, queries, licenses, user guides, and modular `bundled-runtimes/` (`MRtrix3 3.0.8`, `FSL 2602.0`, `Surface Runtime 0.1.0`) with `tree_sha256` auto-repair |
| Install source with repository access | Lightweight checkout below, then staging | Updatable runtime source without website, tests or historical optimization reports |
| Develop or reproduce research | Complete source checkout | Tests, performance evidence, release tools, website source and documentation |
| Inspect performance evidence | Browse the authenticated GitHub repository | No need to copy reports into the plugin installation |

The runtime package excludes `docs/site/`, `docs/website/`, `website/`,
`Testing/`, historical optimization/QC reports, raw benchmark JSON and timing
scripts. A few release documents and runtime CSV references are explicitly
retained. Third-party notices, WMQL queries, WSL helpers and the model manifest
remain included. While the unified `v0.1.1` release archive bundles the native C/C++ runtimes (`MRtrix3`, `FSL` diffusion preprocessing, and `FreeSurfer Surface Runtime`), **deep-learning model weights** (`DDSeg`, `DDParcel`, `DDSurfer`, `TractCloud`, `PNL`) and **large atlases** (`ORG`) remain on-demand downloads fetched on first use, and **FreeSurfer cortical surface postprocessing** requires a user-supplied FreeSurfer license file (`FS_LICENSE`; no personal activation key is bundled).

---

## Lightweight Source Installation

Use a **new checkout**, authorized access to the private repository, and Git
with partial-clone/sparse-checkout support. Bash, Git Bash and PowerShell:

```bash
git clone --filter=blob:none --no-checkout --depth 1 https://github.com/NeuroDMRI/SlicerDSpace.git
cd SlicerDSpace
git show HEAD:scripts/extension_runtime/runtime_checkout.txt | git sparse-checkout set --no-cone --stdin
git checkout
PythonSlicer scripts/stage_slicer_extension.py --install-root ~/SlicerDSpace-extension --link-mode copy --clean
```

`PythonSlicer` is Slicer's own Python, in the Slicer `bin` folder; no system Python is needed. Set `--slicer-version`
for your Slicer and follow the [module-path instructions](../../README.md#installation).
Staging is a source installation, not official SDK/CPack release certification.

`--depth 1` limits history, `--filter=blob:none` fetches file contents on demand,
and sparse rules select the working tree. **A shallow clone alone still
downloads the current website and reports.** Filtering requires server support;
if Git reports `filtering not recognized by server`, sparse checkout can still
hide files without preventing their transfer. Do not expand this route with
submodule filtering or automatic model downloads. See the official
[git clone](https://git-scm.com/docs/git-clone) and
[sparse checkout](https://git-scm.com/docs/git-sparse-checkout) documentation.

Update a lightweight checkout:

```bash
git pull --ff-only
git show HEAD:scripts/extension_runtime/runtime_checkout.txt | git sparse-checkout set --no-cone --stdin
```

Then restage and restart Slicer. This checkout does not contain complete tests
or research evidence and is not an independent reproduction. Save local changes
before `git sparse-checkout disable` to fetch the excluded source on demand.
Use `git fetch --unshallow` only if full commit history is also needed, not as a
normal installation step. Some guide links target reports available only in
the full GitHub repository.

---

## Packages Are Not Source Archives

GitHub's automatic **Source code (zip/tar.gz)** is not an Extension Manager
package. Prefer a CPack extension with corresponding validation. Maintainers
can also generate a clearly labeled local pre-release staging package:

```bash
PythonSlicer scripts/stage_slicer_extension.py --install-root /path/to/staging --link-mode copy --clean
PythonSlicer scripts/package_slicer_extension.py --extension-root /path/to/staging/SlicerDSpace --output /path/to/packages/SlicerDSpace.tar.gz
PythonSlicer scripts/audit_extension_package.py /path/to/packages/SlicerDSpace.tar.gz
```

CMake and staging share `scripts/extension_runtime/runtime_resources.json`;
they no longer recursively collect all of `docs/`. Check the lightweight
selection and package tests when adding runtime resources. Local packaging is
not a substitute for an SDK build, clean installation or scientific acceptance.

The complete-model benchmark freezes its developer harness into the campaign's
`harness/` directory, separately from the runtime snapshot. It must not add
benchmark scripts back into the installable extension to run a comparison.

Do not use `.gitignore` to hide committed reports, delete historical evidence,
rewrite Git history or remove website contributor attribution. Website
deployment stays separate and its build tools are not plugin prerequisites.
