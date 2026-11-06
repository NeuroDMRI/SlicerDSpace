<div align="center">

# WMA Atlases

[Setup](../README.md) | [CLI Templates](CLI_TEMPLATES.md) | [Resources](https://github.com/ChengjinLii/SlicerDSpace-models) | [中文](WMA_ATLASES.zh-CN.md)

</div>

---

## Choose An Atlas

| CLI value | Atlas | Population / source |
| --- | --- | --- |
| `org` (default) | ORG fiber-clustering atlas | Adult reference atlas, applied across the lifespan; [paper](https://doi.org/10.1016/j.neuroimage.2018.06.027), [dataset](https://doi.org/10.5281/zenodo.2648292) |
| `east-west` | East-West WM Atlas | Joint Eastern / Western adult atlas; [paper](https://doi.org/10.1038/s41597-024-03624-2), [dataset](https://doi.org/10.57760/sciencedb.17354) |
| `naba` | Neonatal/adult brain atlas | Joint neonatal / adult atlas; [paper](https://doi.org/10.1016/j.fmre.2025.12.030), [dataset](https://doi.org/10.5281/zenodo.18227691), [author implementation](https://github.com/azurezhangwei/NABA-Atlas) |

NABA specifically covers neonates and adults; it is not a generic child atlas.
Match the atlas and tracking protocol to the study population, and inspect
registration and individual fibre bundles. Atlas choice does not change the
tractography source: UKF remains the reference WMA route.

## Prepare Resources

Resources are opt-in in `SlicerDSpace-models`, outside the extension code.
Only the selected atlas is downloaded. Each part and extracted runtime file is
checked against its pinned SHA256; the original data bytes and attribution are
preserved. The runtime includes all 800 reference clusters required for outlier
removal, not just the atlas's display meshes.

The new archive downloads are approximately 11.1 GB (East-West) and 3.3 GB
(NABA). A clean installation temporarily needs about 39.4 GB and 10.8 GB,
respectively, including downloaded parts, the assembled archive and extracted
files. The setup plan calculates these requirements for the selected atlas.

[Published mirror checks](../release/benchmarks/wma_public_mirror_20261030.json)
cover every resource URL and size, plus complete SHA256-checked downloads of
both extraction indices and one archive part per atlas. Full archive extraction
is also checked from the byte-identical offline mirror.

Preview one resource plan, then prepare it after accepting its terms:

```bash
PythonSlicer scripts/setup_optional_assets.py --profiles wma-east-west --asset-root /path/to/models
PythonSlicer scripts/setup_optional_assets.py --profiles wma-east-west --asset-root /path/to/models \
  --execute --accept-terms
```

Use `wma-naba` for NABA or `wma` for ORG. The same profiles accept
`--from-directory /path/to/offline-mirror`. GUI: select **WMA atlas** under
**Advanced settings > Model parameters**, then use **Environment and installation >
Prepare selected analysis environment**. Switching an atlas does not require
reinstalling WMA or downloading the other atlases.

## Run

```bash
"$SLICER" --launch PythonSlicer run_subject_pipeline.py --no-user-config \
  --subject-root /path/to/subject --subject-id SUBJECT \
  --output-root /path/to/results/east-west --run-name east_west \
  --model-assets-dir /path/to/models --targets validation,tractography,bundles \
  --tractography-backend ukf --bundle-tractography-backend ukf \
  --bundle-backend wma --wma-atlas east-west --slicer-executable "$SLICER" --plan
```

Review the plan before replacing `--plan` with `--execute`. Select `naba` or
`org` with the same flag. `--wma-atlas-dir` overrides the managed path for a
local copy of that selected atlas. Use a separate output/run for each atlas.

On Linux without `DISPLAY` or `SLICERDSPACE_DISPLAY`, WMA automatically uses
`xvfb-run -a` for the Slicer-based back-transformation. Preflight checks require
`xvfb-run`, `Xvfb` and `xauth` before analysis starts. An existing display is
reused; macOS and native Windows do not enable Xvfb automatically. The run
manifest records the selected mode. `--wma-use-virtual-x` remains an explicit
request, but is not needed for a headless Linux batch.

## Implementation And QC

WMA performs registration, spectral clustering, outlier removal, native-space
back-transformation and tract appending. The new atlas adapter selects the
published directories, hemisphere assignments and tract groups. It removes
the author's five **ORG-specific** CPC cluster-index overrides for non-ORG
atlases; all other hemisphere-assessment operations are retained. ORG uses
the existing author driver unchanged.

NABA's released helper lists CPC as commissural, but all seven CPC reference
clusters are labelled hemispheric (`h`) in its atlas. The adapter exports left
and right CPC bundles, consistently with those labels, giving 78 anatomical
outputs. The original cluster data are unchanged. This avoids creating an empty
commissural CPC result from hemispherically separated input clusters.

The runtime records atlas identity, source, selected tract groups and the
atlas-specific output inventory. Resuming a different atlas cannot reuse a
previous atlas's QC. Coverage and per-bundle sampling checks remain active.
Installing or decoding the resources does not replace anatomical review.

From a source checkout, the opt-in resource smoke test decodes the registration
geometry and checks finite coordinates, each verified spectral atlas and the real
WMA appender. It checks 32 reference fibres and every published tract definition,
including exact points, attributes and streamline order in synthetic fixtures:

```bash
"$SLICER" --launch PythonSlicer Testing/Python/run_wma_atlas_resource_smoke.py \
  --family east-west --asset-root /path/to/models \
  --wma-root /path/to/whitematteranalysis --report /path/to/atlas-check.json
```

Use `--family naba` for NABA. Recorded checks: [East-West](../release/benchmarks/wma_east-west_resources_20261030.json)
and [NABA](../release/benchmarks/wma_naba_resources_20261030.json).

### WMQL And Cerebellar Coverage

The pinned official WMQL preset contains 48 output queries, with no dedicated
cerebellar-peduncle or intracerebellar bundle queries. `CB` means **cingulum
bundle**, not cerebellum. Cerebellar labels can be present in DDParcel and the
source tractogram without producing a cerebellar WMQL output. For cerebellar
tracts, choose a suitable WMA atlas and inspect its relevant bundles. WMQL also
offers an opt-in [regional cerebellar extension](WMQL_CEREBELLAR.md), distinct
from these unchanged default rules and from peduncle-specific atlas labels.

The [coverage audit](../release/benchmarks/wmql_coverage_20261030.json) confirms
that all four cerebellar label IDs exist and the source tracks visit these
labels. The retained WMQL example has 41 nonempty outputs and 31 passing its
sampling check; empty or sparse queries remain recorded. The absence of a
dedicated cerebellar query is not an input-coverage failure. The updated Results
artwork explicitly includes the optional regional extension.

---

The source data retain CC BY 4.0 attribution; WMA retains its own software
licence. No NABA algorithm source is vendored into the extension.
