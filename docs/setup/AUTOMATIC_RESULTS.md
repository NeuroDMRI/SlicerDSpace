<div align="center">

# Automatic Key Result Display

[Project](../../README.md) | [Documentation](../README.md) | [Result Review](GUI_RESULT_REVIEW.md)

</div>

---

GUI checkpoint/final hooks call `auto_manifest_paths(path, not_before=launch_time)`
from `shared.auto_results`, then call
`logic.loadKeyResultsFromManifest(path, automatic=True, mode="reuse")` on the
Slicer GUI thread. `automatic_manifest_paths` is an equivalent alias.
The result contains `loaded`, `reused`, `skipped`, `deferred`, `manifest_path`,
and `view_node_ids`. Loaded/reused entries contain original `path`, `product`,
`display_role`, and `node_ids` for view helpers.

Selection uses only completed or explicitly reused tasks in the current
manifest, including finished tasks in a running checkpoint. Batch discovery
uses actual child `results`, exact commands or recorded stdout/log tails.
It ignores planned subjects, shared output folders, and old cache timestamps.
The launch-time guard prevents older child manifests from being displayed.

Independent product limits select FA/MD/AD/RD, FWF/free-water FA/MD, NDI/ODI/IsoVF, categorical
DDSeg/DDParcel labels, all four DDSurfer STL surfaces, one representative per
tractography/SAF backend, AF/CST/CC fiber bundles, connectivity matrices/node tables,
and declared statistics tables. DKI selects six MK/AK/RK/KFA/MKT/RTK maps and,
when present, three distinct MK-Curve-corrected maps. Preprocessing can show a declared
brain mask and mean b0, never the full 4D DWI as a 3D scalar. There is no global automatic 12-item or
40-item limit that can starve later products. Numerical WMA/TractCloud bundle
names fall back to the first three recorded nonempty files, labeled generic
representatives rather than assigned anatomical identities. Legacy bundle
folder discovery remains manual-only.

TRK/TCK loading uses lazy RAS-mm coordinates and at most 30,000 uniformly
distributed streamlines. Automatic loading persists a binary RAS VTK preview
and JSON provenance under
`<subject_root>/cache/display_previews`, then loads that VTK.
The original tractogram is unchanged and remains the ownership/deduplication
source. Node attributes identify preview paths, original source/stamp,
original/display counts, and display-only sampling. Measurements must use the
original scientific file, not the preview. VTK/VTP files above 32 MiB are
deferred for manual loading rather than eagerly blocking automatic display.
Generic tract models default to RAS; explicit RAS/LPS `SPACE` metadata is
respected. Streamline/world geometry is not inferred from finite values alone.

Automatic display sets FA/MD grayscale, bounded color scales for fractional
maps, existing Slicer categorical colors, and surface/bundle colors. Pial
counterparts are loaded but initially hidden to avoid overlapping white
surfaces. New scalars initialize a FA-preferred slice background and labels
initialize an overlay; new models/fibers initialize the 3D camera. Reuse-only
polls do not reset views or user display choices.

Explicitly failed QC is excluded from automatic loading. Recorded `QCState`
is separate from `HumanReviewState=pending`. Loading is not a claim that a person
reviewed the output. A changed source
automatically replaces only its own previously loaded nodes. Manual loading
remains the default and manual reuse still asks for replacement of changed
sources. Other runs and user-owned scene nodes are never cleared.
