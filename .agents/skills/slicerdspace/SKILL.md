---
name: slicerdspace
description: Install, enable and use SlicerDSpace through a local Agent. Prepare diffusion inputs, select analysis routes, validate CLI plans, run requested analyses or experiments, and inspect recorded results and QC. Installation alone does not authorize scientific runs or certify anatomical correctness.
---

# SlicerDSpace Workflow

Use the project's installer and Slicer's own APIs. Keep installation, dependency
readiness and scientific QC as three distinct outcomes. The verified reference
version is Slicer 5.10 on Linux; consult the support matrix for other platforms.

## Choose The Requested Mode

- **Install/enable/prepare/repair:** read [setup](references/setup.md) for source
  staging, activation, the unified GUI preparation entry and Agent automation.
  Stop after the requested setup unless analysis is also authorized.
- **Use/analyze:** if the module already works, do not reinstall it. Read
  [the usage workflow](references/usage.md), then select the relevant
  [algorithm Shell template](references/shell_templates.md).
- **Compare/batch:** use the same usage workflow and
  `docs/setup/EXPERIMENTS.md`; freeze per-variant settings before submitting.

Project-relative paths below refer to the authorized checkout or installed
runtime root containing `shared/` and `scripts/`, not this skill's directory.
Use the skill name `slicerdspace`.
Read this `SKILL.md` as the entry point. When installing or copying the Skill,
preserve its complete directory, including `references/`, `scripts/`, `assets/`
and `agents/`; the entry Markdown file alone does not include those resources.

## Setup Boundaries

- Before resource downloads, follow the [storage-choice step](references/setup.md#confirm-storage):
  validate an already supplied path, or present usable candidates and **ask and
  wait** before creating directories, moving caches or downloading resources.
- Use **Environment and installation > Prepare selected analysis environment**
  as the normal preparation entry. Prefer compatible shared Slicer Python/PyTorch
  and pinned no-Git sources; keep managed PNL isolated. Prepare only requested routes.
- Require explicit user approval of downloads and applicable terms. Never
  autoclick approval boxes, bypass the real confirmation dialog or accept terms
  on the user's behalf. Installation authorization never authorizes a scientific run.
- Preserve scenes and other sessions; ask before closing or restarting them.
  Resume preparation after extension restarts, not analysis. Platform adaptation
  and offline tests are not all-platform, zero-configuration acceptance; see
  [setup limits](references/setup.md#platform-and-delivery-limits).

## Use After Installation

Follow [references/usage.md](references/usage.md) for input staging, route
selection, device/resource planning, GUI/CLI execution, cancellation and review.
The project's templates default to `--plan`. Retain identical scientific
arguments for `--doctor`, `--plan` and the authorized `--execute` run. Preserve
the selected method when dependencies or acquisition checks fail; explain the
specific blocker instead of substituting a method or relaxing QC thresholds.

## Algorithm Code Templates

Use [references/shell_templates.md](references/shell_templates.md) for the full
code catalogue and `assets/shell/*.sh` for standalone Bash templates. Read only
the requested route and its prerequisites, not every algorithm at once.

| Section | Algorithms and variants |
| --- | --- |
| [Masks](references/shell_templates.md#brain-masks) | PNL CNN, DIPY, MRtrix, SynthStrip |
| [Preprocessing](references/shell_templates.md#preprocessing) | DIPY MP-PCA/Local PCA/Patch2Self; MRtrix/FSL |
| [Anatomical prior](references/shell_templates.md#anatomical-prior) | DDSeg, DDParcel, DDSurfer, native cortical postprocessing |
| [Microstructure](references/shell_templates.md#microstructure) | DTI WLS/OLS, FW, NODDI, DKI WLS/OLS, Python MK-Curve, optional MATLAB reference |
| [Tractography](references/shell_templates.md#tractography) | DTI deterministic, UKF, CSD/ACT with SIFT2, SAF |
| [Fiber bundles](references/shell_templates.md#fiber-bundles) | WMQL, WMA, TractCloud; source selection, ORG / East-West / NABA atlases and combinations |
| [Connectome](references/shell_templates.md#connectome-and-combined-routes) | Python count/scalar-weighted or MRtrix/SIFT2; combined routes |

Set the actual Slicer, runtime, subject, output and confirmed resource paths as
shown in the catalogue. The Skill directory and `REPO` are separate locations:
`REPO` can be an installed extension runtime even when this Skill is elsewhere.

```bash
export SKILL="/absolute/path/to/slicerdspace"
bash "$SKILL/assets/shell/ddparcel.sh" doctor
bash "$SKILL/assets/shell/ddparcel.sh" plan
# Only after the user confirms the plan:
bash "$SKILL/assets/shell/ddparcel.sh" execute
```

Omitting the mode means `plan`. Use the same script/settings for all three
modes. `--help` needs no Slicer or data. Additional scientific options belong in
a separate edited script copy, not positional arguments to the template.
Use `DEPENDENCY_CONFIG` for saved GUI dependency paths. The catalogue also
links input import, cohort, parameter-experiment, regional and report commands;
those have their own preview/authorization semantics.

Routine regional analyses use direct `analyze_regions.py` arguments (source run,
regions, metrics and output directory); JSON requests are optional. See the usage
guide for the single-subject and cohort forms.

The standalone scripts are generated from `docs/setup/CLI_TEMPLATES.md`, not
independent implementations of the algorithms. Update that guide and its
Chinese counterpart first, then run `python scripts/build_skill_shell_templates.py`.
`--check` detects missing, edited or stale generated files. Do not regenerate or
overwrite a user's customized copy during an installation or analysis.

## Finish Or Stop

For installation, report the checkout revision, Slicer version, install path,
confirmed model/resource root, prepared resource profiles, activation receipt,
persistence and remaining dependency/restart actions.
For analyses, report methods, output/run/report paths, QC and review items as
described in the usage workflow. If a helper fails, preserve its report and fix the specific
cause; do not repeatedly reinstall, delete unrelated environments or substitute
algorithms. If runtime readiness or real-data QC was not checked, say so plainly.
