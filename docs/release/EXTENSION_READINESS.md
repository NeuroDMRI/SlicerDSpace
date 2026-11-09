<div align="center">

# Extension Readiness

**Evidence before distribution**

[Project](../../README.md) | [Documentation](../README.md) | [Platforms](../setup/SUPPORT_MATRIX.md) | [Release procedure](RELEASE.md) | [中文](EXTENSION_READINESS.zh-CN.md)

</div>

Target: a Tier 1 submission after release gates pass. This is a preparation
record, not Slicer approval. The repository stays private until the maintainer
explicitly chooses otherwise. No Index PR, visibility change or package upload
is performed by the build tools.

Executed local checks and their limits are recorded in the
[Windows 5.10.0 native acceptance report](WINDOWS_NATIVE_ACCEPTANCE_5.10.0.md),
[24 October local surface components validation](SURFACE_COMPONENTS_20261024.md),
[23 October GUI follow-up](GUI_RESPONSIVENESS_20261023.md),
[22 October official integration validation](OFFICIAL_SLICER_VALIDATION_20261022.md),
[16 October follow-up](QC_FOLLOWUP_20261016.md),
[12 October follow-up](DISTRIBUTION_VALIDATION_20261012.md) and
[11 October validation record](DISTRIBUTION_VALIDATION_20261011.md).

---

## Packaging Contract

The runtime resource specification is
<code>scripts/extension_runtime/runtime_resources.json</code>.
Both CMake and staging read it. It includes WMQL query files, the WSL C# bridge
and runtime requirements, LUTs, shell/config files and license notices.
Development tests/reference sources and local model directories are excluded.
Staging is no longer an unrestricted copy of the source directories.

Validate metadata and package contents:

~~~bash
python3 scripts/audit_slicer_extension_metadata.py --json
python3 scripts/stage_slicer_extension.py --install-root ignore/release-test/install --clean --link-mode copy
python3 scripts/package_slicer_extension.py --extension-root ignore/release-test/install/SlicerDSpace --output ignore/release-test/SlicerDSpace.tar.gz
python3 scripts/audit_extension_package.py ignore/release-test/SlicerDSpace.tar.gz --report ignore/release-test/package_audit.json
~~~

The package audit compares declared files and source hashes, including missing,
unexpected and duplicate resources. Passing it is **not** a scientific test or
evidence that a real SDK built the archive.

---

## Real SDK Build

Provide a genuine Slicer build tree containing SlicerConfig.cmake; the binary
application download is not a substitute. Use a new build directory for each
platform/SDK revision:

~~~bash
python3 scripts/build_slicer_sdk.py --slicer-dir /path/to/Slicer-build \
  --build-dir ignore/sdk-release-linux --package-generator TGZ \
  --unit-test-python /path/to/test-env/bin/python
~~~

Use ZIP on Windows. The helper configures, builds, runs CTest, runs CPack and
audits produced packages. Commands, exit codes and log paths are retained in
sdk_build_report.json. Missing SDK configuration is reported as BLOCKED, not
PASS. No external model download is implicit.

Run from a terminal with backend test dependencies installed in that independent
Python. Backend unit fixtures are not executed inside an embedded Slicer app;
generic/native module tests still use the real SDK application. Linux headless
hosts need a working display, for example `xvfb-run -a` around the build command.

The manual **Slicer SDK** Actions workflow requires trusted self-hosted Linux,
Windows and macOS runners labelled <code>slicer-sdk</code>. Set the repository
variable SLICER_SDK_DIR to each runner's common SDK path, or supply the dispatch
input. Different paths can instead be handled by separate manual runs.
Do not run untrusted pull requests on these machines. Three-platform lightweight
resource checks run separately in normal CI.

**16 October evidence:** a genuine Linux Slicer v5.10.0 superbuild and extension
configure/build, 128 CTests, CPack and 460-resource hash audit passed. That actual
CPack archive was installed through Extension Manager with private settings and
no developer module path; cached DTI/ROI and real bundle loading/rejection checks
passed. Scientific dependencies/SlicerDMRI were preconfigured: this is not a
blank-machine full workflow, native Windows/macOS acceptance, or anatomical
certification. Minimal SDK stubs remain contract-test fixtures only.

---

## Independent Clean Installation

1. Install a fresh Slicer with separate settings and no developer module paths.
2. Install the CPack-produced archive from Extension Manager's local-file action.
3. Confirm module imports resolve only inside the installed extension; no development checkout or pre-existing model cache is on the path.
4. Configure only minimal DTI prerequisites and run the documented synthetic installation test.
5. Obtain authorized, de-identified real data independently, then verify DTI images, mask/gradient QC and run records.
6. For WMQL, install its declared backend, run the packaged query and record bundle existence/count/alignment. Resource presence alone is not extraction success.
7. Enable anatomical models only after explicit asset installation and matching SHA256 verification.
8. Repeat separately for each platform/backend claimed as supported.

Record OS, SDK/Slicer revision, package SHA256, installation paths, runtime
versions, input hashes and human-review status. Local staging, archived
source tests and already configured workstations cannot replace this test.

---

## Release Gates

| Gate | State |
| --- | --- |
| Unified CMake/staging resource selection and package regression checks | Implemented; run evidence retained locally |
| LICENSE.txt and Index metadata draft | Prepared; Tier 1 is a proposal |
| Compact-interface documentation, EN/ZH and screenshot sources | Prepared |
| Public repository, resolvable icon/screenshot URLs, GitHub topic | Deferred by maintainer; required before public Index submission |
| Linux real SDK/CPack and private installation | Passed for the documented selected routes; not a blank-machine full workflow |
| Windows native 5.10.0 runtime bundles & full-workflow real-data acceptance | Passed on 3D Slicer 5.10.0 (`HCP 100307`) with `SLICERDSPACE_DISABLE_WSL=1`; see [Windows 5.10.0 Acceptance Report](WINDOWS_NATIVE_ACCEPTANCE_5.10.0.md) |
| Recorded source licenses and LUT notices | Conversion/BWH, FreeSurfer/MGH, DDSurfer/MIT and PyTorch3D/BSD notices restored and checked; not a blanket legal clearance |
| Model/template and fixture redistribution | Unconfirmed groups remain excluded from public asset archives |
| Real-data tutorial and Sample Data registration | [Upstream Slicer DWIVolume walkthrough](../setup/SLICER_SAMPLE_DTI.md) tested for DTI; custom redistributable dataset registration remains pending |
| Related patent disclosure | Maintainer review required; no absence claim is made |

The descriptor under <code>docs/release/extension-index/SlicerDSpace.json</code>
is a schema-compatible draft, not a submitted catalog entry. Its source,
category, dependencies and screenshot metadata must remain aligned with
CMake and the legacy s4ext. URLs will not be publicly usable while the repo is private.

---

## References

Follow the current [Slicer extension guide](https://slicer.readthedocs.io/en/latest/developer_guide/extensions.html),
[submission checklist](https://github.com/Slicer/ExtensionsIndex/blob/main/.github/PULL_REQUEST_TEMPLATE.md)
and [catalog schema](https://github.com/Slicer/Slicer/blob/main/Schemas/slicer-extension-catalog-entry-schema-v1.0.2.json).
The user-guide structure takes cues from [SlicerTotalSegmentator](https://github.com/lassoan/SlicerTotalSegmentator)
and [SlicerDMRI](https://dmri.slicer.org/); this does not claim endorsement or
equivalent support maturity.
