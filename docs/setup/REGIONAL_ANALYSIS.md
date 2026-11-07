<div align="center">

# Regional Analysis

**Optional postprocessing of completed, QC-eligible results**

[Project](../../README.md) | [Documentation](../README.md) | [Results](AUTOMATIC_RESULTS.md) | [中文](REGIONAL_ANALYSIS.zh-CN.md)

</div>

---

## Select Regions

Open **Advanced settings > Regions & input import > Regional analysis (optional)**. This section is
collapsed by default and does not change the products or methods in the main run.

1. Expand the panel to load the latest run, or use **Load regions and available results** to refresh.
2. Search by name, hemisphere or exact numeric label and check regions. Common Chinese anatomical aliases are supported. Enter selects the visible search results.
3. Check the desired analyses and click **Analyze selected regions**. Only analyses with available results appear.
4. Inspect the automatically loaded results and select **Open regional report** for the summary.

**Options and input files** holds the source manifest, atlas/LUT overrides, metric selection,
tractogram and connectivity sources, selection rules, target/waypoint/exclusion regions and output folder.
Only explicitly recorded completed/reused products with acceptable recorded QC and current signatures are eligible.

DDParcel supplies the default atlas and FreeSurfer label names. A custom integer
NIfTI atlas and optional FreeSurfer-format LUT can be selected. Statistical maps
and the brain mask must share its shape and affine: the tool does not silently
resample or register images. A connectome needs the exact atlas recorded by its
parent run, not merely overlapping label numbers.

---

## Available Analyses

| Analysis | Inputs | Result |
| --- | --- | --- |
| Microstructure | Existing DTI, FW, NODDI or DKI/MK-Curve scalar maps | Mean, SD, median, quartiles/IQR, range, valid/excluded counts and coverage |
| Tissue composition | Existing DDSeg hard tissue labels | WM/GM/CSF and unclassified counts, volumes and fractions of each entire ROI |
| Cortical morphology | Completed postprocessing aparc annotation and thickness/area/curvature overlays | Vertex count, area, mean/median thickness, thickness SD and mean curvature |
| ROI streamline selection | A completed full tractogram; matching SIFT2 weights when explicitly recorded | Complete selected lines, source indices, associated weights and a separate display preview |
| Regional connectivity | Existing symmetric connectivity matrix and node table | Induced selected-region matrix plus all nonzero connections from selected nodes to other parent nodes |

Available measures follow the source run; enabling this panel does **not** fit
additional models. DTI includes FA/MD/AD/RD and recorded eigenvalue/trace maps.
FW tissue maps and NDI/ODI require explicit tissue-validity masks. DKI statistics
use raw scalar maps with their validity mask, not display-clipped images;
MK-Curve corrected maps use recorded per-metric masks when available.
Finite zero values remain legitimate observations. Excluded fits do not enter
the denominator; a region with no valid estimates is blank/null, never zero-filled.

Five streamline rules are available:

- **Pass through any** selected region: union of region intersections.
- **Pass through all** selected regions: intersection of region matches, limited to 16 regions per request.
- **At least one endpoint** in the selection: endpoint membership in the selected-region union.
- **Both endpoints** in the selection: both endpoints in the union, not necessarily in different regions.
- **A ↔ B endpoints**: one endpoint in selected group A and the other in target group B, independent of stored direction. Both groups must be nonempty and disjoint.

Additional waypoint regions must all be traversed (up to 16), while entering any exclusion region rejects a line.
Enter names or IDs separated by commas in Options. Ambiguous names return candidate IDs for confirmation.
Statistics/connectivity cover the union of A and B. Full DTI, CSD/ACT, UKF and SAF sources are available when recorded by the parent run.

Cortical linkage requires exact hemisphere and structural-name matches between `ctx-lh-*`/`ctx-rh-*`
atlas entries and actual `aparc.annot` annotations. Thickness/curvature are vertex means and area is
the sum of vertex areas. Unmatched regions are reported; arbitrary atlas IDs are never treated as surface labels.
Matched left/right names and compartments also produce differences and `2(L-R)/(|L|+|R|)` asymmetry
indices. Missing values and zero denominators remain blank.

Through-region rules use DIPY's [line-based ROI targeting](https://docs.dipy.org/stable/reference/dipy.tracking.html#dipy.tracking.utils.target_line_based),
including segment crossings between stored vertices. The output is a geometric
subset, **not new WMQL/WMA/TractCloud anatomical bundle identification**, and
streamline count is not an axon count. Endpoint filtering is an explicit voxel
lookup, not the parent connectome's radial-assignment algorithm.

Connectome weights are not normalized, thresholded, smoothed or recalculated.
Node-table atlas labels, not sequential node indices, identify regions. Missing
edges retain the parent matrix's zero-edge convention. Directed or inconsistent
matrices are rejected instead of being silently symmetrized.

---

## Outputs And Review

Jobs run in a cancellable background process. Key outputs automatically load into
the current scene, grouped by subject, regional run and product. Repeated loads
reuse existing result nodes. The main run's latest-result reference is retained.

By default, new uniquely named output folders are created under
`05_statistics/selected_regions/<parent-run>/<timestamp-id>/`:

```text
request.json
region_analysis_manifest.json
region_analysis_qc.json
selected_regions.nii.gz
selected_regions.csv
selected_microstructure.csv / .json       # if requested
selected_tissue_statistics.csv            # if requested
selected_surface_statistics.csv           # if matching cortical regions are available
bilateral_statistics.csv                  # when selected left/right names pair exactly
regional_report.html                      # offline summary
selected_streamlines.vtk                  # complete matching geometry, if nonempty
selected_streamlines_preview.vtk          # display only
selected_streamline_ids.csv               # optional matching SIFT2 weight column
selected_tractography.json
selected_connectome_matrix.csv
selected_region_connections.csv
selected_connectome.json
```

VTK exports contain RAS-mm geometry, not copied tensor/point arrays. Source
indices retain the link to the original file. Full output coordinates are
written as doubles without clipping or resampling. The reproducible preview is
bounded to 30,000 lines and 1,000,000 points and is **not** a measurement input.
The scene records its full quantitative source separately.

FreeSurfer LUT foreground colors are made opaque in the scene's display table;
the original LUT and labels are not edited. Without a LUT, selected labels use
a deterministic display palette. Dense Slicer color tables support label IDs
up to 1,000,000 here; larger quantitative IDs remain valid exports but are not
silently relabelled for display.

Zero matching lines produce an explicit zero count and no fabricated tractogram.
Source/QC files are fingerprinted before and after processing; changes invalidate
the result. Existing output folders are not overwritten. A cancelled/incomplete
job is not accepted as completed. **Postprocessing consistency is not new upstream
fit validation or human anatomical approval.** Coverage and anatomical priors still require review.

---

## CLI And Verification

Routine analysis uses direct arguments, without a request file:

```bash
PythonSlicer scripts/analyze_regions.py --source-manifest /absolute/run_orchestration_manifest.json \
  --regions 1024 2024 --metrics fa md --output-dir /absolute/new_region_results
PythonSlicer scripts/validate_region_analysis.py /absolute/region_analysis_manifest.json \
  --report /absolute/independent_check.json
```

The source manifest is an existing workflow output, not a new configuration file.
`--regions` accepts IDs or names, including quoted multiword names, comma-separated
values and repeated flags. The default metric is FA when no analysis type is
specified; selecting only tissue, surface, tractography or connectivity does not
implicitly add scalar metrics. Metrics must be present and QC-eligible in the run.

| Option | Purpose |
| --- | --- |
| `--source-manifest RUN [RUN ...]` | One run, or several runs for cohort export; may be repeated |
| `--regions NAME_OR_ID [...]` | Selected anatomical regions |
| `--metrics fa md [...]` | Available scalar metric keys, e.g. `fwf`, `noddi_ndi`, `noddi_odi`, `dki_mk` |
| `--tissue`, `--surface` | Include tissue composition and matching cortical morphology |
| `--tractogram TASK` | Source task key, e.g. `tractography.ukf` |
| `--tract-rule RULE` | `through_any` (default), `through_all`, `endpoint_any`, `endpoint_both`, `endpoint_pair` |
| `--target-regions [...]` | Target group for A-B endpoint selection |
| `--include-regions [...]`, `--exclude-regions [...]` | Required waypoints and excluded regions |
| `--connectome TASK` | Source task key, e.g. `connectome.streamline_count` |
| `--atlas FILE`, `--lut FILE` | Optional atlas and label-name overrides |
| `--output-dir DIR` | Fresh result directory, or cohort root with several sources |

For example, select UKF tracks connecting the precentral regions:

```bash
PythonSlicer scripts/analyze_regions.py --source-manifest /absolute/run_orchestration_manifest.json \
  --regions 1024 --target-regions 2024 --tractogram tractography.ukf \
  --tract-rule endpoint_pair --output-dir /absolute/new_region_tracks
```

Use `--inspect /absolute/run_orchestration_manifest.json` to print available labels,
metrics and task keys when needed. `--catalog /absolute/catalog.json` optionally
saves that inventory. Names resolve against the current LUT; ambiguous names require IDs.

Apply the same options to multiple completed subject runs:

```bash
PythonSlicer scripts/analyze_regions.py --source-manifest /absolute/subject1/run.json /absolute/subject2/run.json \
  --regions 1024 2024 --metrics fa md --output-dir /absolute/new_region_cohort
```

For advanced saved configurations, `--request /absolute/request.json` and
`--batch /absolute/region_batch.json` remain supported; do not combine these JSON modes
with direct analysis flags. The GUI creates its request automatically.
The [batch template](examples/region_batch.json) contains a shared `request`, subject manifests
(with optional atlas/LUT overrides) and a fresh output directory. Subjects run sequentially without
refitting. `cohort_*.csv` long tables retain subject IDs, sources and atlas hashes. Selected atlas
IDs/names must have matching semantics. Failed subjects retain their errors while other subjects
continue; a partial batch returns a nonzero exit code. No group tests or normative ranges are inferred.

Native Slicer tests cover background execution, EN/ZH labels, 440-pixel layout,
label IDs above 1000, automatic result loading and duplicate reuse. Independent
checks compare regional statistics with direct Boolean-mask calculations,
every exported streamline with its recorded original, and exported connectivity
weights with the parent matrix. These checks do not establish clinical validity.

The [20 October validation record](../release/benchmarks/regional_analysis_20261020.json)
includes a real million-streamline HCP-YA case and separates postprocessing time
from upstream model fitting; it is not an algorithm-speedup benchmark.

The [29 October linkage validation](../release/benchmarks/regional_analysis_20261029.json)
covers the simplified panel, tissue/cortical integration, A-B selection with exact source
geometry/weights, and independently checked regional summaries for three real subjects.
