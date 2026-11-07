<div align="center">

# Setup and result review in Slicer

[Project](../../README.md) | [Documentation](../README.md) | [Quick Start](../setup/QUICKSTART.md) | [Release](../release/EXTENSION_READINESS.md) | [中文索引](../README.zh-CN.md)

</div>

## Background setup

Python package installation and model download/checksum verification run in
separate processes. Their progress windows show current output; full commands
and output are saved in the environment report directory, or the user settings
directory's `setup-logs` folder if no report directory is available.

Cancel stops the running process. On Linux, subprocess groups are terminated
and force-stopped after a grace period. Already installed packages are not
rolled back. Model partial files are retained for downloader resume. After
installation, check dependencies again; restart Slicer if an imported package
was updated. Preflight-triggered installation does not automatically start
scientific analysis: review setup and click **Run SlicerDSpace** again.

PyTorch installation remains an explicit Slicer PyTorch extension operation;
the setup UI does not silently replace its runtime with a generic pip build.
External software installation policies are unchanged.

---

## Run-scoped scene results

During a GUI run, completed stages trigger automatic loading of their key
declared products. Selection is per product, not the first files found in an
output directory. Already loaded, unchanged nodes are reused. A current run may
explicitly reuse cached successful products; arbitrary older runs are never
scanned for automatic display. Actual batch children are taken from execution
records; a planned subject list is not evidence that those subjects ran.

Anatomical labels, bilateral white/pial surfaces, representative parameter
maps, a whole-brain tractogram/SAF preview, representative fiber bundles and numerical
connectome tables are selected where declared. TCK/TRK streamlines are converted
to a reproducible **RAS-mm VTK display preview** with at most 30,000 streamlines.
The full tractogram is untouched and remains the input to quantitative analysis.
Without SlicerDMRI, a fiber preview can load as a line Model; it does not expose
all SlicerDMRI tractography controls. Display coordinate handling is explicit.

Results are selected only from successful or explicitly reused tasks in the
chosen run manifest. Missing files, failed tasks and unselected tasks are not
loaded. There is no fallback scan of shared output folders. Products that are
not explicitly listed in the manifest require manual loading; this avoids
presenting unrelated historical results as current results.

### Choosing a run

Open the separate **Results and QC** section. The **Run** list shows every run found under the output
directory (or `<subject root>/outputs`), the subject root, the batch root and
any folder added earlier with **Add a run folder…**, newest first, as
`time · subject · run name · status`. GUI, command-line and batch runs all
appear, including runs from earlier Slicer sessions. The last list item, **Add a run folder…**, adds any run
or output folder (searched three levels deep for
`00_run_metadata/<run>_orchestration_manifest.json` or
`00_batch_metadata/batch_*.json`) and remembers it. The list rescans each time the section is opened.
Manifests with an unknown `kind` or an unsupported `version` are skipped and
reported. The selected manifest is validated again when a scene action is
clicked. Batch entries show per-subject status; pick a subject's own run to
load images.

Nodes are grouped by subject, run and product in Subject Hierarchy. Source and
run attributes persist when the scene is saved and reopened.

- **Load** reuses unchanged owned nodes instead of duplicating them.
- **Replace** removes nodes owned by this run and reloads its declared results.
- **Append** explicitly adds another copy.
- **Clear** removes only this run's owned nodes, never source files.

In automatic mode, a changed owned source is replaced. In manual reuse mode,
changed source file size or modification time triggers a reload warning rather
than silently reusing an old scene node. This is a scene freshness check, not a
cryptographic integrity check. Scalar and anatomical products are prioritised
ahead of potentially large tractograms. Deferred/skipped results are reported.

Categorical segmentation/parcellation outputs load as label maps; probability
maps remain scalar volumes. MD uses grayscale; FA/NDI/ODI/fraction display
ranges use 0-1 where identified. DKI display settings are recorded separately
from unbounded estimates and validity masks. These settings do not modify voxel data.

---

## Status and limitations

Execution completion, QC reporting and pending visual review are shown
separately. A finished process is not evidence that anatomical review passed.
Inspect the QC report and overlay results on subject images before acceptance.

This update does not make every UI operation asynchronous. Environment scans,
large MRML volume/tractogram loading and some legacy module-specific actions
can still take time on the main thread. The generic job cancellation tests use
synthetic commands without changing a user's Python installation or downloading
large models; real network and platform-specific cancellation need separate
acceptance testing.

---

## Editable Tissue Review

After DDSeg labels have loaded, right-click its stage row in **Results** and
choose **Edit tissue copy in Segment Editor**. The native Segmentation node
contains separate white-matter, gray-matter and CSF segments on the original
grid. Reopening preserves edits; replacing or clearing automatic results does
not remove this manual copy. Save the revision separately with Slicer's normal
Save dialog. It does not overwrite model output, change QC to passed or silently
replace the inputs to regional statistics or downstream analysis.

Scene fiber bundles (`.mrb`) can retain loaded volumes, tables and editable copies even
after the original files move. Source paths are provenance, not a promise that
the original processing files or environment are included in the bundle.

---

## Distortion correction (topup)

With **Preprocessing** checked under **2. Products**, the
**Distortion correction (topup)** group under **Advanced settings > Preprocessing & mask**
shows three fields. Preprocessing has one route (W2a), so there is no route choice. They set the same
options as the command line:

| GUI field | CLI flag | dwifslpreproc |
| --- | --- | --- |
| Phase-encoding direction | `--preprocessing-phase-encoding-direction` | `-pe_dir` |
| Total readout time (s) | `--preprocessing-readout-time` | `-readout_time` |
| Reverse-PE b=0 image | `--preprocessing-reverse-phase-b0` | `-rpe_pair -se_epi` |

Accepted directions are `i`, `i-`, `j`, `j-`, `k`, `k-`, `AP`, `PA`, `LR`,
`RL`, `IS`, `SI`. The readout time must lie in 0.001–1.0 s; values that look
like milliseconds are rejected. The reverse-PE image must be an existing
NIfTI/MRtrix file and cannot be combined with batch mode. Unset fields are not
forwarded, so dwifslpreproc defaults apply.
