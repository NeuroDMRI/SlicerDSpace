<div align="center">

# SlicerDSpace Documentation

**Install, run, inspect and reproduce**

[Project](../README.md) | [Quick Start](setup/QUICKSTART.md) | [Platforms](setup/SUPPORT_MATRIX.md) | [Release](release/EXTENSION_READINESS.md) | [中文](README.zh-CN.md)

</div>

Start with the compact-interface tutorial. Technical guides describe individual
routes; dated reports preserve historical evidence, not universal guarantees.

---

## User Guides

Use the [download guide](setup/DOWNLOADS.md) to choose a runtime package,
lightweight source checkout or complete development/research repository.

| Guide | Purpose |
| --- | --- |
| [Quick Start](setup/QUICKSTART.md) | Minimal DTI workflow and synthetic installation check |
| [CLI templates](setup/CLI_TEMPLATES.md) | Copyable routes, explicit tracking/bundle sources, PowerShell, regional analysis and reports |
| [Python API](setup/PYTHON_API.md) | Checked workflow entry, planning and partial or strict execution |
| [Cohort resource scheduling](setup/COHORT_RESOURCES.md) | Optional cross-subject compute-stage admission and resource reservations |
| [Experiments](setup/EXPERIMENTS.md) | Advanced parameter overrides, named variants and cohort concurrency |
| [Agent skill](../.agents/skills/slicerdspace/SKILL.md) | Installation, input preparation, requested analyses and result review |
| [Output layout](setup/OUTPUT_LAYOUT.md) | New `03_anatomy` products and read-only compatibility with historical `03_structural` results |
| [Capabilities and platforms](setup/SUPPORT_MATRIX.md) | Configure only the selected backend |
| [Result review](setup/GUI_RESULT_REVIEW.md) | Run-scoped loading, cancellation and QC |
| [Brain masking](setup/brain_mask_gui_config.md) | PNL CNN default (managed on Linux, native and experimental on macOS and Windows), explicit alternatives |
| [GPU execution](setup/GPU_EXECUTION.md) | GPU-preferred neural inference, explicit CPU choices and CPU-only route boundaries |
| [Managed setup](setup/MANAGED_SETUP.md) | One confirmation with all licences, self-checks, Python/models, MRtrix/FSL runtimes, repair and offline resources |
| [Microstructure](setup/MICROSTRUCTURE.md) | DTI default with colour FA map, independent FW/iFW/NODDI/DKI and original MK-Curve boundary |
| [UKF tractography](setup/UKF.md) | Optional author extension, defaults, lossless input and automatic WMA handoff |
| [WMA atlases](setup/WMA_ATLASES.md) | ORG, East-West and NABA selection, on-demand resources and atlas-specific QC |
| [WMQL cerebellar selections](setup/WMQL_CEREBELLAR.md) | Optional regional rules, CLI/GUI presets and checks |
| [Regional analysis](setup/REGIONAL_ANALYSIS.md) | Optional Advanced region selection, scalar statistics, ROI streamlines and unchanged-weight subgraphs |
| [Automatic results](setup/AUTOMATIC_RESULTS.md) | Current-run loading and RAS VTK display previews |
| [Settings](setup/USER_CONFIG.md) | Persistent GUI/CLI configuration |
| [Linux](setup/LINUX_HPC.md), [Windows](setup/WINDOWS.md), [macOS](setup/MACOS.md) | Platform-specific setup and limitations |
| [Model assets](assets/MODEL_ASSETS.md) | Profiles, download and checksum verification |
| [Asset terms](assets/MODEL_ASSET_TERMS.md) | Permission review, independent of file verification |
| [Third-party notices](../THIRD_PARTY_NOTICES.md) | Bundled source licenses, provenance and distribution conditions |
| [Implementation origins](architecture/IMPLEMENTATION_PROVENANCE.md) | DDSeg/SAF/MK-Curve ports and the modified PNL inference code versus unmodified upstream |
| [Dependencies](setup/EXTERNAL_DEPENDENCIES.md) | External software and route prerequisites |

---

## Maintainer Guides

- [Extension readiness](release/EXTENSION_READINESS.md): packaging, SDK commands and release gates.
- [Release procedure](release/RELEASE.md): code-only archives, assets, tags and evidence.
- [Repository layout](architecture/REPOSITORY_LAYOUT.md): anatomical, microstructural and connectivity packages.
- [Product architecture](architecture/PRODUCT_ARCHITECTURE.md): native client, optional dependencies, isolated backends and release boundaries.

---

## Validation Records

[Portable setup and default methods](release/PORTABLE_SETUP_20261107.md) lists
what changed for users: native tensor fit, one setup confirmation, the PNL mask
route per platform, the run record, and the one-time regeneration after the update.

[Selected dependency preparation](release/DEPENDENCY_SETUP_20261021.md) records
native setup/cancellation checks, an isolated PNL GPU installation and mask run,
offline verification, and remaining platform/clean-machine boundaries.

[21 October bundle workflow review](release/BUNDLE_WORKFLOW_20261021.md)
records automatic input handoff, three-subject WMQL re-auditing, saved WMA
re-auditing, actual TractCloud runs and sampling warnings, and isolated scene tests.

[16 October follow-up](release/QC_FOLLOWUP_20261016.md) records verified MK-Curve
exclusion, per-metric regional coverage, completed ACT-v3/WMA and Linux SDK/CPack
checks, plus the unresolved endpoint-WMQL sampling and anatomical-review gates.

[CSD/ACT investigation](release/CSD_ACT_DIAGNOSIS_20261015.md) supersedes the
earlier CSD acceptance: VentralDC tissue mapping was wrong, the controlled saved-FOD
experiment isolates its effect, and old ACT-dependent outputs require regeneration.

[Current platform audit](release/PLATFORM_AUDIT_20261015.md) records Windows/WSL
device and bridge fixes, macOS path support, timing boundaries and unverified routes.

[Current bundle QC follow-up](release/BUNDLE_QC_20261015.md) explains legacy green states,
sampling gates, real input diagnostics and platform limitations.
[WMQL CST diagnosis](release/WMQL_CST_20261015.md) distinguishes the legacy
crossing query from the published endpoint rule, with paired inputs and frozen imports.

[Bundle backend verification](release/BUNDLE_BACKENDS_20261013.md) separates
real WMA failures, sparse WMQL candidates and seed-verified TractCloud products.
[Module runtimes](release/MODULE_RUNTIMES_20261017.md) inventories historical and
new measurements without treating cache reuse or ROI timings as full computations.
[Performance optimizations](release/PERFORMANCE_OPTIMIZATIONS.md) distinguishes
original-author WMA comparisons with byte-identical geometry from older-plugin
iterations, inherited improvements, device effects and cache reuse.

[SAF Python verification](release/SAF_PYTHON_20261013.md) records independent
author-kernel comparisons, intentional corrections and actual tractogram geometry checks.

[Current DDSurfer synchronization](release/DDSURFER_SYNC_20261013.md) records
RAS-marked OBJ export, Python labels and batched fsaverage mapping with real-output comparisons.
[Pipeline QC and performance](release/PIPELINE_QC_20261015.md) records current
failure gates, full-brain NODDI repeatability, completed DKI/postprocessing checks
and the remaining anatomical and platform acceptance boundaries.
[Scientific follow-up](release/SCIENTIFIC_CLOSURE_20261015.md) adds full-brain
MK-Curve/reference checks, unchanged author WMQL on new UKF tracks, and the
status of the new WMA/joint GUI runs. Human review remains explicit and pending.
[Python MK-Curve verification](release/MKCURVE_PYTHON_20261012.md) records the
native Python correction route and independent author-code comparisons.

[Masking/microstructure/scene follow-up](release/PIPELINE_MASK_MICROSTRUCTURE_20261012.md)
records the PNL default, DKI eligibility, independent Advanced switches and
current-run automatic result loading, including the unverified MATLAB boundary.

[12 October follow-up](release/DISTRIBUTION_VALIDATION_20261012.md) records
license notices, fresh-clone tests, local installation and the outstanding SDK gate.
[11 October validation](release/DISTRIBUTION_VALIDATION_20261011.md) preserves
the preceding working-tree campaign; its test count is not the scoped Git snapshot's count.

[Scientific/GUI campaign](release/SLICER_ACCEPTANCE_20261007.md),
[compact UI](release/UI_SIMPLIFICATION_20261008.md), and
[DDSurfer synchronization](release/DDSURFER_SYNC_20261005.md) are dated records.
Older Beginner/wizard descriptions are not the current interaction guide.

Documentation follows DDSurfer's centered masthead and section navigation.
Pinned upstream Markdown and third-party license texts are left unchanged
to preserve their recorded hashes and attribution.
