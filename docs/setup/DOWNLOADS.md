<div align="center">

# Download Scope

**Separate plugin installation, lightweight source and research records**

[Project](../../README.md) | [Quick Start](QUICKSTART.md) | [Documentation](../README.md) | [中文](DOWNLOADS.zh-CN.md)

</div>

Running the plugin does not require website source, historical performance
reports or tests. Keep those in the complete repository for development,
auditing and reproducibility, not as installation prerequisites.
Release `v0.1.1` separates the massive white-matter atlases from each platform installer so clinicians never have to manually merge 8–10 archives:
1. **Website Custom Installer (`Install-SlicerDSpace.bat` / `.command` / `.sh`)** (<https://neurodmri.github.io/SlicerDSpace/#start>): generates a double-click setup script that installs the unified OS Base package plus any chosen WMA atlases (**`ORG`** checked by default; **`NABA`** and **`East-West`** unchecked by default; unchecking **WMA** skips all three) with built-in mirror (`ghfast.top` / `gh-proxy.com`), HTTP/HTTPS proxy, and offline folder support—plus in-Slicer hot on-demand supplement if you add another WMA atlas later.
2. **Compact OS Base Extension Packages** (`SlicerDSpace-v0.1.1-Windows-x64.zip`, `SlicerDSpace-v0.1.1-Linux-x64.tar.gz`, `SlicerDSpace-v0.1.1-macOS-arm64.tar.gz`): each bundles the module (`depends NA`), Python 3.12 packages (`bundled-python/`), native runtimes (`MRtrix3 3.0.8`, `FSL 2602.0`, `FreeSurfer Surface Runtime` with `activation_key_included: false`), `UKFTractography` CLI, `bundled-sources/`, and all non-WMA model weights (`DDSeg`, `DDParcel`, `DDSurfer` + `atlas.tar.gz`, `PNL` weights [excluding `upstream-only` NITRC `IITmean_b0_256.nii.gz`], and `TractCloud`). Physical Intel (`x86_64`) Mac hardware remains unverified.
3. **Standalone WMA White-Matter Atlas Archives**: `SlicerDSpace-v0.1.1-WMA-ORG.tar.gz` (single `< 2.0 GiB` file), `SlicerDSpace-v0.1.1-WMA-NABA.tar.gz.001..` and `SlicerDSpace-v0.1.1-WMA-EastWest.tar.gz.001..` (split only when needed under GitHub's `2.0 GiB` ceiling).

---

## Choose A Download

| Purpose | Recommended route | Contents |
| --- | --- | --- |
| Recommended everyday setup (base + chosen WMA atlases) | Use the [Website Installer Customizer](https://neurodmri.github.io/SlicerDSpace/#start) | Unified OS Base package + optional WMA atlas archives (`ORG` default; `NABA` & `East-West` optional) + NITRC `IIT` fetch + hot in-Slicer atlas supplement |
| Manual base installation on Windows, Linux, or Apple Silicon macOS | Download `SlicerDSpace-v0.1.1-Windows-x64.zip`, `SlicerDSpace-v0.1.1-Linux-x64.tar.gz`, or `SlicerDSpace-v0.1.1-macOS-arm64.tar.gz` (`Install Extension from File` in Slicer 5.10.0) | Extension runtime (`depends NA`), UI, queries, licenses, `bundled-python/`, `bundled-model-assets/` (all non-WMA weights), `bundled-sources/`, `UKFTractography` CLI, and `bundled-runtimes/` (`MRtrix3 3.0.8`, `FSL 2602.0`, `Surface Runtime`) |
| Install source with repository access | Lightweight checkout below, then staging | Updatable runtime source without website, tests or historical optimization reports |
| Develop or reproduce research | Complete source checkout | Tests, performance evidence, release tools, website source and documentation |
| Inspect performance evidence | Browse the authenticated GitHub repository | No need to copy reports into the plugin installation |

The runtime package excludes `docs/site/`, `docs/website/`, `website/`,
`Testing/`, historical optimization/QC reports, raw benchmark JSON and timing
scripts. A few release documents and runtime CSV references are explicitly
retained. Third-party notices, WMQL queries, WSL helpers and the model manifest
remain included. While the unified `v0.1.1` OS Base release archives bundle Python 3.12 dependencies, non-WMA **deep-learning model weights** (`DDSeg`, `DDParcel`, `DDSurfer`, `TractCloud`, `PNL`), algorithm sources, `UKFTractography` CLI, and native C/C++ runtimes (`MRtrix3`, `FSL` diffusion preprocessing, and `FreeSurfer Surface Runtime`), **FreeSurfer cortical surface postprocessing** still requires a user-supplied FreeSurfer license file (`FS_LICENSE`; no personal activation key is bundled), and the three WMA atlases (`ORG`, `NABA`, `East-West`) are distributed as separate optional packs. Source-staged checkouts without `bundled-*` directories download model weights and Python packages on demand on first use.

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
