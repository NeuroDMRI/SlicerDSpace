<div align="center">

# Performance Optimizations

**Reviewed 10 November 2026 · Complete-model pairs and numerically equivalent optimizations**

[Project](../../README.md) | [Documentation](../README.md) | [Runtimes](MODULE_RUNTIMES_20261017.md) | [Author-reference CSV](benchmarks/author_wma_optimizations_20261020.csv) | [JSON](benchmarks/author_wma_optimizations_20261020.json) | [中文](PERFORMANCE_OPTIMIZATIONS.zh-CN.md)

[Next bottlenecks and measured follow-up (中文)](PERFORMANCE_BOTTLENECKS_AND_PLAN.zh-CN.md)

</div>

**Original-author comparisons execute author code; Python-port, threading and
scheduling comparisons are labeled separately.** WMA bulk copying preserves
the author's filtering/classification definition. New Free-water NLS dispatch
executes the identical DIPY author bodies without changing initialization,
solver or convergence. These optimize computation, not the scientific estimator,
and do not claim greater anatomical accuracy.

The earlier 10-11x regional-statistics and 1.42x MK-Curve observations compare
against older plugin code, **not original author implementations**. They remain
in an engineering appendix, not evidence of outperforming the original algorithm.
Device changes and inherited upstream improvements are also separate categories.

This review recomputes historical JSON and adds paired implementation experiments.
Historical fits remain untouched; new long-model fits use separate directories.
The author-WMA section and engineering appendix follow commit
`a48fc95e0a52a88a1ae99dd683a547520acc0c5d`; the long-model/scheduling supplement
uses the working tree after `65fee3e411da34f0d0a220f3384c81f604f60079`.
The tree is dirty, so a commit alone does not identify measured implementations.
JSON evidence records implementation hashes and immutable snapshots.

---

## 10 November: Five-Module Sub-Algorithm Computation Optimizations

Across the five canonical pipeline modules (`Anatomical priors`, `Microstructure`, `Tractography`, `Fibre bundles`, `Connectome`), additional computation-preserving optimizations eliminate redundant background-slice CNN evaluation, 4D thick-slice array duplication, full-volume connected-component relabeling, repeated unweighted OLS QR factorizations, per-streamline Python/VTK geometry loops, and per-vertex scalar connectome lookups ([source records: 9 November](benchmarks/runtime_optimizations_20261109.json), [source records: 10 November](benchmarks/five_module_subalgorithm_optimizations_20261110.json)). Every pair executes 1 warmup plus 3 alternating repetitions on real HCP Young Adult data and passes exact output equivalence checks:

| Module | Sub-algorithm | Optimization & scope | Reference | Optimized | Speed ratio | Time saved |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Anatomical priors | DDSeg | Uniform background slice cache + `inference_mode` (full 3-view CUDA, `100408`) | 6.20 s | 5.38 s | 1.15× | 13.12% |
| Anatomical priors | DDSeg | Uniform background slice cache + `inference_mode` (strided full-view CPU, `100408`) | 26.31 s | 12.94 s | 2.03× | 50.81% |
| Anatomical priors | DDParcel | Zero-copy `sliding_window_view` thick-slices + uniform batch cache + sagittal LUT (full 3-view CUDA, `100408`) | 69.02 s | 42.84 s | 1.61× | 37.93% |
| Anatomical priors | DDParcel | Bounding-box `region.slice` 3D connected-component cleanup ($256^3$ volume, `100408`) | 64.06 s | 35.87 s | 1.79× | 44.00% |
| Microstructure | DTI (Teem WLS) | Single zero-signal template + nonzero-row `tenEstimateUpdate` (full brain, `100206`) | 74.09 s | 31.31 s | 2.37× | 57.74% |
| Microstructure | DTI (DEC-FA) | Active-FA ($>0$) $3\times 3$ frame rotation and `eigh` (full brain, `100206`) | 4.51 s | 2.63 s | 1.71× | 41.58% |
| Microstructure | MK-Curve | Cached `dgeqp3` OLS QR (`dormqr`/`dtrsm`) + batched `eigh` + vectorized tensor basis (64 voxels, `100206` / `149741` / `877168`) | 1.35 / 1.61 / 1.27 s | 1.23 / 1.18 / 1.14 s | 1.10–1.37× | 9.45–26.95% |
| Tractography | CSD / ACT / SIFT2 | Vectorized flat `ArraySequence` tractogram QC validation (full-brain tractogram, `100408`) | 10.12 s | 5.76 s | 1.76× | 43.07% |
| Tractography | Regional tractography | Batched 4,096-streamline endpoint ROI selection (1M streamlines, `100206`) | 33.58 s | 1.75 s | 19.16× | 94.78% |
| Fibre bundles | WMA | Vectorized VTK `GetOffsetsArray`/`GetConnectivityArray` tract summary (50k streamlines, `100408`) | 8.94 s | 0.55 s | 16.32× | 93.87% |
| Connectome | Connectome (`streamline_count`) | Batched 4,096-streamline voxel + 2 mm `cKDTree` radial assignment (1M streamlines, `100206`) | 120.61 s | 4.31 s | 27.97× | 96.42% |
| Connectome | Connectome (`mean_fa`) | Batched endpoint assignment + vectorized `reduceat` along-tract FA sampling (50k streamlines, `100408`) | 84.25 s | 1.66 s | 50.61× | 98.02% |

In `tractography/csd.py`, `run_csd_act` also records a content-verified `csd_fod_preparation.json` receipt binding input DWI/mask/5TT/GMWMI hashes and spherical-harmonic parameters so repeated tracking/SIFT2 runs on unchanged inputs can reuse verified FOD and response products when `reuse_prepared_fod=True`.

---

## 11 November: Multi-Subject Cohort Stage-Level Scheduling (Post-Optimization Re-Benchmark)

The batch runner (scripts/run_batch_pipeline.py --scheduler adaptive) coordinates individual pipeline stages across subjects within a shared CPU, memory, GPU and I/O budget ([source records](benchmarks/cohort_multi_pipeline_20261109.json)). Under a 32-core CPU budget per arm on a 16-subject HCP cohort using the fused CPU and CUDA algorithm implementations, Subject-level fixed concurrency (--scheduler subjects) is compared against Adaptive stage-level scheduling (--scheduler adaptive):

| Cohort pipeline (N = 16) | Subject-level makespan | Adaptive makespan | Throughput ratio | Time saved |
| --- | ---: | ---: | ---: | ---: |
| Microstructure (DTI + FW + NODDI + DKI + MK-Curve) | 136.22 min | 100.85 min | 1.35× | 25.96% |
| Anatomical priors (DDSeg + DDParcel + DDSurfer) | 71.69 min | 64.23 min | 1.12× | 10.41% |
| Tractography, Bundles & Connectome (DDSeg + DDParcel + DTI + DTI-Det + WMQL + TractCloud + Connectome) | 71.22 min | 62.46 min | 1.14× | 12.29% |

Decomposing the downstream connectivity run by stage family gives **1.22× (18.30% saved)** on Tractography and **1.14× (12.36% saved)** on Bundles. All 16 subjects complete and pass equivalence checks in every arm. In the Microstructure cohort, stage-level scheduling overlaps lower-thread Free-water/DKI stages with multithreaded NODDI/MK-Curve stages across subjects, raising mean CPU core utilization from 12.9 to 19.9 cores without altering individual scientific estimators. On modest personal machines ($\le 16$ GiB RAM or 4–8 GiB consumer GPUs), passing `--low-memory` caps concurrency, I/O tokens and per-stage reservations conservatively.

---

## 30 October: Evidence Refresh

The supplement and homepage now include the 29 October incremental MK-Curve
experiment, with [source records](benchmarks/performance_next_20261029.json).
Chunk-local signal reuse keeps non-b0 log signals across the full synthetic-b0
scan; each candidate retains its own weights and weighted solve. All paths use
SciPy 1.18.1 and one numerical thread, with one warm-up and three alternating
repetitions. Median numeric-chunk times are:

| Acquisition chunk | Previous QR | Signal reuse | Time saved |
| --- | ---: | ---: | ---: |
| 1 | 1.193 s | 1.059 s | 11.26% |
| 2 | 1.126 s | 1.000 s | 11.24% |
| 3 | 1.102 s | 0.989 s | 10.24% |

Each chunk has 64 voxels, 181 measurements and 200–201 synthetic-b0 samples.
Coefficients, complete curves, maps, residuals and validity masks match exactly
for both solvers. The batched candidate ranges from 1.50% slower to 3.07% faster;
default reuse remains reference-solver only. These are increments over the
existing Python QR path; the full-brain record remains separate.

Supplementary Table 4 includes all six comparisons, and Supplementary Figure 1
now contains eleven operations. The same source records the trained DDParcel
fusion candidate at 59.814 versus 59.851 s, with identical logits and no
measurable gain; it remains a screening result rather than a speedup row.
The homepage retains fourteen scoped records with inline, initially collapsed
details. Shared DTI/input preparation is described with its cache boundaries;
no new full-workflow or anatomical-accuracy claim is added. This refresh reuses
recorded experiments and does not perform new scientific fits.

---

## 29 October: Free-Water Protocol-Array Reuse

The [implementation](../../microstructure/freewater_protocol.py) reuses water attenuation,
repeated spectra and fraction-grid arrays across voxels of the same acquisition design.
An instance-private read-only grid cache uses exact refinement bounds and a 2 MiB budget.
Changed design values, memory layouts or diffusivity settings retain uncached evaluation.
Observation weights, pseudoinverses, parameter searches and tensor decompositions remain voxel-specific.
Reuse is enabled for source-qualified serial WLS without adding threads or dependencies.

Three native-grid 12³ ROIs each retain 1,728 brain-mask voxels and 288 measurements.
The comparison executes original **DIPY 1.12.1**, the previous column-scaling/tensor-helper path,
and protocol reuse. Each variant has one warm-up and three rotated-order repetitions using
one numerical thread. Median complete-ROI times include reading, fitting, QC and fresh exports:

| Subject | Author | Previous | Protocol reuse | Author/reuse |
| --- | ---: | ---: | ---: | ---: |
| 100307 | 2.183 s | 1.895 s | 1.731 s | 1.26× |
| 100206 | 2.200 s | 1.884 s | 1.738 s | 1.27× |
| 149741 | 2.199 s | 1.846 s | 1.686 s | 1.30× |

All 13 raw parameters, exported arrays, affines, dtypes and QC decisions match exactly.
The incremental ratio over the previous implementation is 1.08–1.09×; the original
441/596/743 excluded tissue voxels remain excluded.

The completed full-brain pair retains 740,916 mask voxels and 288 measurements, one numerical
thread per variant, fresh outputs and immutable implementation/harness snapshots.
Original author WLS takes **938.584 s (15.64 min)** and the updated implementation takes
**713.395 s (11.89 min)**: **225.190 s (3.75 min)** saved, **23.99%** less time, **1.316x**.
All 13 raw parameters in 54 blocks, four maps/masks, affines, dtypes and QC match exactly.
The same 623,981 tissue-valid and 116,935 excluded voxels are retained; inputs and snapshots
are unchanged. The full pair includes existing column/helper optimizations plus protocol reuse;
the three-path ROI comparison isolates the incremental reuse benefit. This is one shared-host
pair, optimized first, with identical raw-parameter export in both calls.
[Full-brain JSON](benchmarks/freewater_protocol_full_20261029.json) ·
[CSV](benchmarks/freewater_protocol_full_20261029.csv).

The 29 October source audit finds PyPI stable 1.12.1 and author head `57ae6daf9824`.
The four fitting/residual definitions in both sources match the qualified SHA256 values.
[Upstream audit](benchmarks/freewater_protocol_upstream_20261029.json) ·
[JSON](benchmarks/freewater_protocol_20261029.json) ·
[CSV](benchmarks/freewater_protocol_20261029.csv) ·
[Reproduction script](../../scripts/benchmark_freewater_protocol.py).
Supplementary Table 3 in the separate manuscript supplement contains the three-path comparison.

---

## 25 October: Shared Preparation Across Products

This is **workflow-level reuse**, not a faster tensor estimator. The scientific
models and their trained preprocessing contracts remain separate.

| Preparation | Reuse policy |
| --- | --- |
| DWI/mask detached headers | Prepared once, including scaled NIfTI payloads, for both LS and WLS consumers |
| Native tensor and six scalars | Once per input bundle and fitting method; DDSeg retains LS, DDParcel/DDSurfer and matching microstructure routes share WLS |
| Atlas registration/resampling | Cached independently by atlas, fit and source format; DDSurfer retains its trained 1-mm atlas and NRRD resampling sources |
| Trained normalization and inference | Not merged: DDSeg, DDParcel and DDSurfer consume different channels, grids and models |
| DTI scalar export and QC | Existing source/QC-verified maps survive forced optional-model runs; new or changed sources still invalidate them |
| Free-water, NODDI and DKI | May share inputs and ordinary DTI prerequisites, not fitted parameters or model-specific initialization |
| Tractography and bundle consumers | Existing full-source tractograms can feed several consumers; UKF conversion, WMA registration, WMQL rules and TractCloud inference remain independent |

The shared layer now has OS-owned writer locks, atomic receipts and artifact
SHA256 checks; input changes, missing/modified products and interrupted builds
cannot qualify as cache hits. Native receipts are versioned with conversion and
preparation code. A new DDParcel task receipt also binds normalized inputs,
weights, settings and exports; a partial or changed task is rebuilt, rather
than skipped because a single final file exists. Existing legacy products without
new receipts retain their historical status, not a newly certified provenance.

The common converter was checked against the pinned DDSurfer converter. Zero-b
placeholder directions now encode zero gradients; NIfTI slope/intercept is
materialized when needed. These are **correctness fixes for those input cases**,
not bitwise-equivalent optimizations of previously incorrect conversion.

On three real 12-cubed ROIs (100206, 149741, 877168; all 288 measurements), the
tensor, b0 and six scalar arrays and geometries match both independent author-CLI
controls exactly. Two WLS consumers need **one tensor call and six scalar calls**
instead of two and twelve. The tested header/scaling, corruption, settings,
writer-lock and surface-adapter contracts are reproducible with
`scripts/benchmark_shared_preparation.py` and `Testing/Python/test_shared_preparation.py`.
The fresh full-surface experiment retains a **failed independent-registration
comparison**: both geometric QC reports pass, but independently estimated affine
transforms produce maximum vertex differences of 1.56-3.91 mm. Repeating the
unchanged author BRAINSFit command also changes its transform; the installed CLI
uses random metric sampling. This is not accepted equivalence or speed evidence.

A separate controlled test fixes the same reference transform, **newly resamples
the shared native maps**, and runs actual normalization and both surface models.
All six model-input images/affines and four white/pial surfaces/topologies match
exactly, with **zero maximum vertex difference**. No normalized image or mesh is
copied from the reference. The control does not make independent registrations
deterministic, alter the author's sampling defaults or certify anatomical accuracy.
See [frozen evidence](benchmarks/shared_preparation_20261025.json),
`scripts/benchmark_shared_surface.py` and `scripts/validate_shared_surface_transform.py`.
ROI times are not scaled to full brain; no whole-workflow speed ratio is claimed.

---

## 25 October: Optimization Figure Coverage

The previous figure selected only seven website-summary entries, including
threading and scheduling; it did not cover this report. The replacement shows
**ten recorded timing pairs above a matching-ID methods table**. The table is
one line per method; detailed equivalence checks remain in this report/data.
No new scientific benchmark was run for this editorial expansion.

| Scope | Included changes | Boundary |
| --- | --- | --- |
| Author full-model controls | Free-water WLS; complete WMA | Same estimators, one shared-host pair each |
| Author local controls | Free-water NLS dispatch; WMA filtering, point transforms and fiber-array conversion | Explicit ROI/fiber scope; local ratios are not full-module ratios |
| Repeated author preparation | WMA registration-input reuse | Includes first cache fill; overlaps filtering and complete WMA |
| Python-port computation | MK-Curve protocol/design and LAPACK workspace reuse | Same Python port, not a MATLAB speed comparison |
| Inherited upstream reuse | Four DDSurfer surface overlays in one mapping call | Credit remains with DDSurfer upstream; command scope only |
| Generic data I/O | Decode compressed DWI once, then memory-map slabs | Needs temporary disk; not model fitting |

Thread counts, device switching, scheduling, existing DIPY array-API integration
and older-plugin-only ROI aggregation are excluded from the figure, not deleted
from history. Ten operations do not mean ten scientific algorithms or ten
original contributions. The public catalogue retains 13 classified records with
source hashes; source freshness remains dated 22 October, not a new audit.

[Runtime comparison](../site/assets/img/fig-runtime-optimizations.png).

---

## Upstream Freshness Audit

Checked against official GitHub/PyPI metadata on **22 October 2026**, not just
the locally installed packages. "Latest stable release" and "current default
branch" are different references. The [machine-readable audit](benchmarks/upstream_freshness_verified_20261022.json)
records both, retrieval time, immutable revisions and relevant source hashes.

| Component | Current official reference | Verification and permitted claim |
| --- | --- | --- |
| WMA | `master@4532b911386d` (newer than release v0.4.2) | Measured author commit is still current HEAD; independently fetched filter, registration and license bytes match. The measured bulk/cache gains are not against an obsolete checkout. |
| DIPY Free-water | Stable 1.12.1; `master@afa75ecd30df` | WLS/NLS and tensor-helper definitions match the qualified hashes in stable and HEAD. The earlier 1.11.0 headline is historical; the current full-brain 1.12.1 pair independently verifies a 15.87% reduction. |
| DIPY DKI | Stable 1.12.1 | Upstream already supports array-level metrics. Our per-voxel-to-array comparison is **plugin integration of existing upstream functionality**, not an improvement over DIPY's best native usage. |
| AMICO | Stable 2.1.1 | Both original timed runs' saved configurations record 2.1.1, inspected without unpickling code. The 1/8-thread benefit is already an upstream feature, not our algorithm. |
| MK-Curve | `master@9a213e1a5850` | MATLAB author reference is still current HEAD. All Python-port comparisons remain port-only, never Python versus current MATLAB runtime. |
| SciPy / MK-Curve QR | Stable 1.18.1 | Upstream rewrote `lstsq` with native batching. We do not claim to invent that functionality. Fresh paired tests against its current public `gelsy` path still show benefit from protocol/design/workspace reuse in the Python port. |

Latest-stack checks use an **isolated Python 3.13 / DIPY 1.12.1 / SciPy 1.18.1**
environment; the user's Slicer and other shared environments are not upgraded in
place. Three alternating complete 12-cubed FW ROI pairs give **1.188-1.195x**,
with all 13 parameters, exported maps and QC exact within each pair. Three real
64-voxel MK-Curve chunks give **1.492-1.682x** for default QR and **1.036-1.068x**
for the existing batched solver; every coefficient, curve, map and decision is
exact. These are not full-brain ratios. Records:
[FW](benchmarks/freewater_latest_stable_roi_20261022.json),
[MK-Curve](benchmarks/mkcurve_qr_scipy1181_20261022.json).

The three current-stack 24-cubed DKI repeats also retain exact arrays/QC counts,
with **3.213-3.235x** relative to the prior per-voxel integration. This remains an
integration benefit from DIPY's existing array API, not a new DIPY speed record:
[DKI integration evidence](benchmarks/dki_latest_stable_integration_20261022.json).

The new **complete full-brain stable-release pair** retains all 740916 mask
voxels and 288 measurements with one thread in each path:

| DIPY 1.12.1 / SciPy 1.18.1 | Original author WLS | Equivalent optimized WLS | Saved | Reduction |
| --- | ---: | ---: | ---: | ---: |
| Subject 100307, complete model | 914.162 s / 15.24 min | 769.129 s / 12.82 min | 145.032 s / 2.42 min | 15.87% / 1.189x |

All 13 raw parameters in 54 chunks, four output maps/masks and automatic QC
match. The same 623981 tissue voxels remain valid and 116935 remain excluded.
Both automatic QC reports pass; manual anatomical review is still separate.
This replaces the website's old-DIPY headline, **not historical measurements**.
The environment, immutable snapshot/harness and source-input hashes are saved in
[current full-brain evidence](benchmarks/freewater_latest_stable_full_20261022.json).
One shared-host pair, optimized first, caches not flushed, raw-parameter export
included in both calls; no confidence interval or whole-workflow claim.

Numerical parity is **within a fixed environment**, not between library releases.
Do not interpret differences caused by upgrading dependencies as an optimization
benefit. SciPy 1.18.1 is explicitly qualified for the direct QR path; unqualified
versions keep public dispatch. A separate current-numerics CI job tests the latest
stable backend pair. Historical Python 3.10 WSL pins remain a compatibility stack,
not a latest-version claim; AMICO is updated to 2.1.1 there. Blindly upgrading that
shared WMA/pandas stack or Slicer's core SciPy is not part of this validation.

Recheck with `scripts/audit_performance_upstreams.py --output <new-report.json>`;
add both timed AMICO `--amico-config` files to verify their historical versions.
The script does not install or run downloaded upstream code. HEAD inspection
does not certify a development-branch installation or future upstream versions.

Final software verification: **1316 tests, 1313 passed and three skipped** in
each of the original and current-stable isolated environments. A newly created
CI-only Python 3.13 environment with DIPY 1.12.1, SciPy 1.18.1, NumPy 2.4.6 and
nibabel 5.4.2 passed all **25** dedicated numerical tests. The website passed
**95** desktop/mobile/interaction checks. Logs stay under `/tmp` with the
`slicerdds-upstream-` / `slicerdds-current-` prefixes and raw numerical evidence
under `ignore/validation/upstream_current_20261022/`. These are software and
paired-numerical checks, not complete new cross-platform scientific acceptance.

---

## Long Models And Independent Scheduling

The following claims are distinct, not one new scientific algorithm:

| Category | Implementation | Permitted claim |
| --- | --- | --- |
| Computational implementation | WMA bulk copying | Same estimator, less computational overhead, not extra threads or greater accuracy |
| Upstream capability integration | DIPY's existing array-level derived metrics | Less overhead than the plugin's previous per-voxel integration, not faster than native DIPY array usage |
| Further computational reduction | Default WLS without a dense diagonal; same WMA VTK transform through a bulk call | Same results, independent of threads; ROI/optimizer scopes are stated separately |
| Parallel configuration | Original AMICO NODDI at 1/8 threads | Upstream multicore capability, not a novel plugin estimator |
| Workflow scheduling | Concurrent FW/NODDI/DKI; no duplicate fresh FW fit when entering NODDI | Less route wall time, not faster single-model mathematics |

Profiling located substantial per-voxel Python-wrapper overhead in the plugin's
previous derived-DKI integration. The [implementation](../../microstructure/dki.py) retains DIPY WLS/OLS,
SVD, parameters, S0 and validity rules; existing array-capable author metrics
operate on `DiffusionKurtosisFit` rather than thousands of single-voxel wrappers.
This follows [DIPY source](https://docs.dipy.org/stable/_modules/dipy/reconst/dki.html),
not normal-equation substitution, smoothing or downsampling. This array capability
already exists upstream; the timing reference is the plugin's prior integration,
not a deliberately slower older DIPY release. Versions 1.11/1.12
are regression-tested; other versions retain per-voxel evaluation.

Three HCP subjects each had three alternating fresh complete-stage repeats on
real native-grid 24-cubed ROIs:

| Subject | Per-voxel metrics, median | Array-level metrics, median | Ratio | Agreement |
| --- | ---: | ---: | ---: | --- |
| 100307 | 54.509 s | 16.948 s | 3.22x | All exported voxel arrays and affines exact |
| 100206 | 54.506 s | 17.038 s | 3.20x | Same |
| 149741 | 54.330 s | 16.794 s | 3.24x | Same |

Timings include I/O, fitting, six metrics, residuals, raw parameters, masks, QC and
export. Fitted/valid counts also match, without removing difficult voxels.
Full measurements are updated in the
[CSV](benchmarks/complete_models_20261020.csv) and
[incremental JSON](benchmarks/complete_models_20261020.json). **ROI measurements
are not full-brain timings and cannot be applied to the historical 91.6 min.**
A DKI diagonal-weight WLS candidate was rejected because raw parameter agreement
failed on extreme real voxels; it is not used by the pipeline and is distinct
from the validated Free-water WLS adaptation below. Instrumented
14/4-second observations locate hotspots, not formal timing claims.

**Advanced -> Concurrent microstructure models** accepts 1/2/3; the default is
sequential. CLI: `--max-parallel-models 2`. Independent FW, NODDI and DKI processes
run only after input/native-DTI preparation; shared ROI tables/manifests publish
once afterward. This is not a complete parallel anatomical/CSD/WMA DAG.
CPU/memory reservations bound overlap, unknown memory serializes, and oversized
jobs fail explicitly. GPU selection is inherited unchanged, numeric CPU limits
are child-only, and failure/cancellation stops owned work rather than publishing
success. Budgets are per subject, not cluster-wide: avoid high batch and model
concurrency together. Individual logs/times live under `compute_schedules/`.

A complete FW+DKI scheduling pair on the real 24-cubed ROI measured
**43.953 -> 24.631 s, 1.78x, 19.322 s saved**, with unchanged source/code hashes
and identical output maps/masks. An earlier 55.662/25.832 pair is retained,
not selected as the headline. Neither pair proves a full-brain improvement.
An isolated native Slicer session passed Advanced-control visibility, bounds
and bidirectional binding. `WithinRange(1,3)` fixes the numeric range overwritten
by Slicer parameter binding in the first GUI test.

Fresh complete-model pairs have also started on an immutable code snapshot:
NODDI 1/8 threads, original versus optimized complete WMA, full MK-Curve before/
after Python-port caching, and DKI original/dense metrics. No original outputs
are overwritten and no full-brain masks or scientific settings are reduced.
The complete NODDI 1/8-thread pair has finished: **1748.935 s / 29.15 min ->
311.525 s / 5.19 min, 5.61x, 23.96 min saved**. Both used the same frozen
implementation, complete 740916-voxel brain mask and scientific settings, with
independent cold outputs. All NDI, ODI, IsoVF, RMSE, NRMSE and validity-mask voxel
arrays and affines match exactly, and both automatic QC reports pass. This is
native multicore benefit for this complete NODDI route, not a new algorithm or
a whole-workflow/cross-hardware guarantee. One shared-host pair has no confidence
interval. Two additional complete-model pairs have now finished:

| Complete model | Reference | Optimized | Saved | Ratio | Comparison |
| --- | ---: | ---: | ---: | ---: | --- |
| WMA | 102.10 min | 74.22 min | 27.89 min | 1.38x | Original route versus bulk geometry/cache; all 74 bundle files byte-identical |
| MK-Curve | 165.90 min | 153.53 min | 12.37 min | 1.08x | Python-port basis caching; every map, curve, coefficient and mask exact |

Each is one pair on a shared Linux host, including preparation, computation,
QC and export; neither has a confidence interval. UKF input, full atlas and
author WMA sources have separate before/after hash auditing. The MK-Curve pair
uses `batched`, one worker, chunks of 256 and a 2500 s/mm2 upper shell. It is
**not a MATLAB runtime comparison and predates the new QR-dispatch optimization
below**. Do not multiply ratios or add savings from different comparisons as
whole-workflow evidence.

The full-brain DKI pair remains incomplete. Saved progress stops at 710656 of
740916 voxels with a stale `running` status; no corresponding benchmark process
was present during the 21 October review. This is not a continuing run or a pass.
Evidence: `ignore/validation/long_tasks_full_20261020_v2/` and the incremental
[CSV](benchmarks/complete_models_20261020.csv)/[JSON](benchmarks/complete_models_20261020.json).
Historical interrupted/failed states are retained, not overwritten as success.
The summarizer does not push to GitHub.

No matched complete SlicerDDSurfer end-to-end speedup percentage is available.
New DKI phase times and workflow task intervals/fractions support further
profiling; overlapping task seconds must not be summed as wall time.
Anatomical and full cross-platform scientific acceptance remain separate.

Current backend regression: **1125 tests, 1120 passed, five skipped, no failures**;
log: `ignore/validation/long_tasks_20261020/regression_release_verified.log`.
The 1110-test count in the author-WMA section below belongs to the earlier round.
Native GUI checks used an isolated installation; neither the current user Slicer
nor MRI Agent was restarted. The active user session needs redeployment before
showing the new controls.
Two DKI agreement tests and ten scheduler tests also passed under Slicer's own
Python with DIPY 1.12.1.
Before pushing, an independent export of the staged changes excluded unrelated
uncommitted work: **1085 tests, 1077 passed, eight skipped, no failures**. The
native Slicer concurrency GUI check passed again (19.0 s). Logs are retained under
`ignore/validation/push_performance_20261020/`.

---

## 21 October: Original Free-water NLS Dispatch

This section records the first, dispatch-only implementation and measurements.
The later default-WLS diagonal-free adaptation is reported separately below.
Original records are retained, not overwritten or multiplied together. The
first implementation snapshot is under
`ignore/validation/numerical_performance_20261021/before_wls/`.

Profiling real 64-voxel NLS fitting found about 3700 calls through DIPY's
legacy positional-argument adapters. Repeated version/signature parsing cost
more than evaluating the residual formula itself. The
[implementation](../../microstructure/freewater_dispatch.py) uses DIPY's existing
callable-fit interface to execute the **identical author function code objects**.
A private namespace connects WLS initialization, NLS residuals and Jacobian
without those compatibility adapters. No DIPY globals or other model instances
are patched. Optimizer, objective, initialization, weights, defaults, termination,
precision and QC remain unchanged.

Only source-hash-matched DIPY **1.11.0/1.12.1 serial NLS** qualifies. Missing or
changed author sources, other releases, custom models, WLS and joblib/dask/ray
retain original dispatch. Progress, logs and `freewater_qc.json` retain the
implementation, DIPY release and author-source hashes. No user setting,
dependency, thread or GPU-policy change is introduced. **Default Free-water
remains WLS and does not obtain the speedup below.**

Three real native-grid 12-cubed ROIs retain every one of 1728 masked voxels
and 288 measurements. Both routes use one BLAS/OMP thread, one warmup and three
alternating repeats. Complete-stage timings include validation, DWI reading,
author NLS, four public map/mask files, QC, source checks and export:

| Subject | Author dispatch median | Optimized median | Ratio | Time reduction |
| --- | ---: | ---: | ---: | ---: |
| 100307 | 8.402 s | 4.056 s | 2.07x | 51.7% |
| 100206 | 8.538 s | 4.026 s | 2.12x | 52.9% |
| 149741 | 8.405 s | 4.033 s | 2.08x | 52.0% |

Environment: Python 3.13/DIPY 1.11.0. All **13 raw fit parameters** match
exactly, as do FWF, tissue FA/MD, validity masks, affines, dtypes, QC checks and
statistics. All three automatic QC records pass; original invalid-tissue
exclusions remain 238/681/313 voxels, not filled or relabeled valid.
These are complete **ROI** stages, not full-brain/workflow timings. Do not apply
the ratio to the historical 21.48 minutes or combine it with AMICO threading as
a new-algorithm claim. The shared-host observations are not hardware guarantees.

Independent pairs on the same three complete ROIs with Slicer's Python
3.12.10/DIPY 1.12.1:

| Subject | Author dispatch median | Optimized median | Ratio |
| --- | ---: | ---: | ---: |
| 100307 | 12.470 s | 5.175 s | 2.41x |
| 100206 | 12.511 s | 5.412 s | 2.31x |
| 149741 | 12.669 s | 5.328 s | 2.38x |

Every within-environment raw parameter and public array remains exact, with
identical QC and exclusions. Absolute differences between Python environments
are not attributed to this optimization. All repeats are retained in the
[Slicer JSON](benchmarks/freewater_dispatch_slicer_20261021.json)/
[CSV](benchmarks/freewater_dispatch_slicer_20261021.csv).

[JSON](benchmarks/freewater_dispatch_20261021.json)/
[CSV](benchmarks/freewater_dispatch_20261021.csv) retain repeated times and input/
source hashes. The [reusable benchmark](../../scripts/benchmark_freewater_dispatch.py)
accepts canonical real input directories, crops without resampling and retains
difficult voxels. Tests cover abnormal/low signals, optional losses, isolation
and fallback; they also pass with Slicer's DIPY 1.12.1.

---

## Further Optimization: Default Free-water WLS

The [adaptation](../../microstructure/freewater_wls.py) retains the author's
initialization, three-step fraction-grid refinement, signal floor, pseudoinverse,
tensor decomposition and return ordering. Only `W.T @ diag(sig**2)` becomes
equivalent C-order column scaling, avoiding a 288-by-288 weight matrix per voxel.
**No fit method or thread count is changed**: default WLS remains WLS. Serial
NLS also uses this initialization.

Only the previously qualified DIPY versions and author-function hashes enable
the path. Unqualified dtypes, zero/nonfinite weights, nonfinite protocols and
other releases retain original behavior; parallel engines are unchanged.
The [DIPY notice](../../licenses/DIPY.txt) and
[provenance](../../licenses/provenance.json) ship with the code and match the
independently retrieved pinned upstream license bytes.

Three native 12-cubed ROIs retain all 1728 voxels/288 measurements. Both routes
use one thread, one warmup and three alternating repeats. Complete stages
include checking, reading, fitting, QC, source checks and fresh exports:

| Subject | Python 3.13 / DIPY 1.11.0, author -> optimized | Slicer Python / DIPY 1.12.1, author -> optimized |
| --- | ---: | ---: |
| 100307 | 2.140 -> 1.874 s, 1.14x | 2.802 -> 2.502 s, 1.12x |
| 100206 | 2.142 -> 1.878 s, 1.14x | 2.810 -> 2.526 s, 1.11x |
| 149741 | 2.154 -> 1.879 s, 1.15x | 2.828 -> 2.512 s, 1.13x |

All 13 raw parameters, four public maps/masks, affines, dtypes and QC match
exactly. All automatic QC records pass, retaining 234/672/303 tissue exclusions.
Comparisons stay **within one method and environment**, not WLS versus NLS.
Exploratory full 24-cubed raw-fit checks also matched across 41472 voxels, but
do not replace the formal complete-stage timings.

Records: [Python JSON](benchmarks/freewater_wls_diagonal_20261021.json)/
[CSV](benchmarks/freewater_wls_diagonal_20261021.csv) and
[Slicer JSON](benchmarks/freewater_wls_diagonal_slicer_20261021.json)/
[CSV](benchmarks/freewater_wls_diagonal_slicer_20261021.csv).
These complete ROI measurements are not full-brain/workflow timing and cannot
be applied to the historical full Free-water duration.

### Full-brain Default WLS Pair

After the 05:10 resumption, a separate immutable campaign measured subject
100307's **complete native grid, 740916 mask voxels and 288 measurements**,
without cropping/downsampling or reducing iterations. Both variants are serial
with one numerical thread:

| Original DIPY WLS | Diagonal-free adapter | Saved | Time reduction | Ratio |
| ---: | ---: | ---: | ---: | ---: |
| 1219.249 s (20.32 min) | 1101.401 s (18.36 min) | 117.848 s (1.96 min) | 9.67% | 1.107x |

Timings cover complete preparation, fitting, QC/export and identical
raw-parameter sidecar recording in both variants. All 13 parameters at every
voxel across 54 chunks, four maps/masks, affines and dtypes match exactly.
QC decisions match and both automatic QC records pass: 623981 tissue voxels
retained, 116935 excluded. Uninterpretable tissue is not relabeled valid.
Input, snapshot and harness hashes remain unchanged.
[Full-brain JSON](benchmarks/freewater_full_stage_20261021.json)/
[CSV](benchmarks/freewater_full_stage_20261021.csv) retain raw times/products/QC.

This is **one subject, one pair, optimized-first order on a shared Linux host**,
not a 9.67% whole-workflow improvement. Do not multiply it by ROI/thread/other
stage ratios. The tensor-helper candidate below was not enabled in this frozen
version; there is no statistical-significance, platform or accuracy claim.

---

## Further Optimization: WMA Registration Points

The [adapter](../../tractography/wma_transform.py) uses the same author-created
`vtkTransform`, replacing point-by-point calls with `TransformPoints`. Float64,
point/fiber order and output layout remain unchanged; objectives, sampling,
COBYLA, convergence and classification are untouched. Only pinned-source hashes
and tested VTK 9.5.2 enable it. Unsupported/nonfinite data and inputs beyond the
64 MiB temporary budget retain original dispatch. Per-registration logs record
calls, points, bypasses and times; normal/error exits restore the original hook.

Author bytes match an independent upstream download. Both routes use one
VTK/BLAS thread:

| Subject | 1000-by-15-point transform, author -> optimized | Complete small optimizer median, author -> optimized | Optimizer ratio |
| --- | ---: | ---: | ---: |
| 100206 | 24.586 -> 0.322 ms | 0.387 -> 0.302 s | 1.28x |
| 149741 | 24.831 -> 0.317 ms | 0.325 -> 0.256 s | 1.27x |
| 877168 | 24.624 -> 0.296 ms | 0.405 -> 0.319 s | 1.27x |

One warmup; five alternating transform repeats and three optimizer repeats.
Fixed/moving fixtures contain 128/64 fibers of 15 points, sigma=20, Affine,
identical steps and a `maxfun=300` **budget**; actual evaluations are 60/51/64.
Real fixtures receive an explicit synthetic 20 mm R shift. Every objective
trajectory, change and final transform matches; all three objectives improve,
and sources/inputs remain hash-identical. No budget is reduced for one method.

**This is a complete small optimizer, not full-brain registration, 74-tract WMA
or anatomical acceptance.** The transform-only 76-83x must not replace the
optimizer's 1.27-1.28x or multiply the earlier complete-WMA 1.38x. Historical
CSD inputs do not gain a coverage-QC pass. Full repeats and trajectories are in
[JSON](benchmarks/wma_transform_20261021.json)/[CSV](benchmarks/wma_transform_20261021.csv);
the [benchmark](../../scripts/benchmark_wma_transform.py) executes author code.
A separate synthetic real-CLI integration test on the copied installation
enabled 30 bulk calls without bypass, not a real full-brain task. Exploratory
distance batching was about 5% slower; buffer reuse gained only about 3% with
variability. Neither is enabled; the objective/distance formula is unchanged.
[Combined evidence](benchmarks/further_optimizations_20261021.json) retains author
verification and candidate decisions.

Combined NLS regression using the new WLS initialization also passed on three
complete Slicer Python 6-cubed ROIs, 216 voxels each: all raw parameters, maps and
QC match author NLS. See [record](benchmarks/freewater_nls_wlsinit_20261021.json).
Do not compare their absolute timings to the earlier 12-cubed fixtures.

---

## 05:10 Follow-up: Parameterization And Candidates

[Bulk FiberArray conversion](../../tractography/wma_fibers.py) replaces point-call
loops with bounded array copying, retaining the author's index-based nearest
neighbors, ties-to-even rounding, hemisphere calculation and VTK traversal
state. It does not introduce arc-length interpolation or reduce samples.
The owned registration process installs/restores the hook only for the pinned
`fibers.py` and VTK 9.5.2. Verbose/custom/oversized cases retain the author path.
**This is not installed in the 74-tract classification subprocess.**

Each real subject contributes 20,000 fixed-seed lines from its million-line
input, one warmup and five alternating repeats, one VTK/BLAS thread. Timings
include conversion and hemisphere calculation, not source reading/sampling:

| Subject | Author | Bulk | Saved | Conversion ratio |
| --- | ---: | ---: | ---: | ---: |
| 100206 | 0.587 s | 0.0128 s | 0.574 s | 45.84x |
| 149741 | 0.568 s | 0.0115 s | 0.556 s | 49.55x |
| 877168 | 0.587 s | 0.0135 s | 0.574 s | 43.54x |

All coordinates and hemisphere state match at 2/3/10/15/30 points per line;
source/code hashes are unchanged. Complete small author optimizer fixtures
with explicit synthetic offsets retain their entire trajectories and final
transforms. After replacing full-length temporary boolean index checks with
min/max reductions, the entire table was remeasured: this small optimizer
improves only **1.9%-4.4%** in time, versus 0.8%-1.9% in the retained first run.
Shared-host variation is not hidden. The 43-50x conversion ratio is not
whole-WMA acceleration and must not multiply prior full-model/transform ratios.
[Original-author verification and repeats](benchmarks/wma_fibers_bounded_20261021.json),
[CSV](benchmarks/wma_fibers_bounded_20261021.csv),
[benchmark](../../scripts/benchmark_wma_fibers.py). Historical CSD sampling
failures remain failed; this is not anatomical acceptance. The first
[measurement](benchmarks/wma_fibers_20261021.json) is retained separately.

Other tested/inspected candidates are not deployed benefits:

| Candidate | Current evidence | Decision |
| --- | --- | --- |
| Free-water broadcast grid views | Exact raw parameters, 512 voxels per subject, only 3.3%-3.6% faster than the prior local adapter | Not enabled |
| Direct original tensor-helper body | Same three subsets, exact raw parameters, 19.6%-20.7% faster than the prior local adapter | Promising candidate only; source guards, edge cases and complete stages still needed |
| Binary WMA output | Pinned author `io.py` already selects binary VTK/VTP | Not a new plugin optimization |

The Free-water candidate ratios are raw-fit subset comparisons against the
prior local implementation, not original-author complete-model measurements.
They were not enabled in the completed frozen full-brain comparison.
[Candidate evidence](benchmarks/performance_candidates_20261021.json) includes
58 passing copied-install native Slicer Python tests and the synthetic real-CLI
fixture's one conversion/30 transform calls. Complete backend/distribution
regression at this stage: **1226 tests, 1223 passed, three skipped, no failures**;
latest log: `ignore/validation/runtime_distribution_20261021/full-tests-bounded-final.log`.

---

## 21 October: Repeated MK-Curve QR Calls

This targets the default `reference` solver without adding threads. Profiling
found about 13,000 least-squares wrapper calls per real 64-voxel, 200-b0 chunk.
The [implementation](../../microstructure/mkcurve_tensor.py) retains the same
LAPACK `DGELSY` pivoted QR, epsilon threshold, initial OLS, weights, rank decisions
and floating-point precision. It caches protocol-only column scaling/rank checks
and LAPACK workspace sizes, avoiding redundant checks and Python wrappers.
No voxel fits are cached; no grid samples are removed; normal equations do not
replace QR. Read-only design caches are content-keyed, limited to four entries,
each with at most 1 MiB of input. SciPy 1.13/1.17 families have paired regression;
other releases retain the public solver.

Both paths use one BLAS/OMP thread, one warmup and three alternating repeats:

| Scope | Environment | Time reduction across three subjects | Agreement |
| --- | --- | ---: | --- |
| Complete 64-voxel numeric chunk, default QR | Python 3.13 / SciPy 1.17.0 | 26.0%-28.7% | Raw coefficients, full curves, maps and decisions exact |
| Same chunk and solver | Slicer Python 3.12 / SciPy 1.13.1 | 20.7%-21.0% | Same |
| Additional gain for existing `batched` solver | Both environments | 4.8%-5.9% | Same; do not apply the default-QR ratio |

A separate complete-stage test on three real native 24-cubed grids used
**preselected sparse 64-voxel masks**, including preparation, all 200-203 b0
samples, 51 maps, curves, coefficients, masks, QC and export:

| Subject | Reference median | Optimized median | Ratio | Unchanged valid voxels |
| --- | ---: | ---: | ---: | ---: |
| 100307 | 1.869 s | 1.402 s | 1.33x | 62/64 |
| 100206 | 1.982 s | 1.388 s | 1.43x | 61/64 |
| 149741 | 1.778 s | 1.396 s | 1.27x | 64/64 |

These are not whole-ROI/full-brain timings. Joint quantitative QC still excludes
invalid voxels in the first two fixtures; exact performance parity does not turn
them valid or replace anatomical review. Inputs/source hashes remain unchanged.
Tests cover rank deficiency, extreme/nonfinite signals, array layout, cache
invalidation and fallback. No new dependency or GPU-policy change is required.

Records: [Python](benchmarks/mkcurve_qr_20261021.json),
[Slicer Python](benchmarks/mkcurve_qr_slicer_20261021.json), and
[complete-stage JSON](benchmarks/mkcurve_qr_stage_20261021.json)/
[CSV](benchmarks/mkcurve_qr_stage_20261021.csv).
The [reusable benchmark](../../scripts/benchmark_mkcurve_qr.py) accepts a frozen
previous tensor source and real-signal NPZs. The frozen file hash matches
`microstructure/mkcurve_tensor.py` at `main@6d753de`. Do not extrapolate these
subset ratios to the previous 153.53-minute run or the entire workflow.

### Regression And Copied Installation

After integrating WLS and point transforms, the 21 October backend suite ran
**1214 tests: 1209 passed, five explicitly
skipped, no failures**. Log:
`ignore/validation/numerical_performance_20261021/full_tests_wls_transform.log`.
A separate native Slicer Python process from `/tmp` imported the isolated
extension copy, verified source hashes/import locations and passed **53 numeric,
parallel-chunk and wrapper tests**. Evidence: `staged_wls_transform.json`/`.log` in that
directory. New modules/benchmarks are included by the runtime resource manifest;
earlier 1203/28-test logs remain intact, not relabeled as this result.
62 active Markdown documents passed local-link/style auditing.
These are backend/installation-import regressions, not complete SDK, GUI,
anatomical or three-platform scientific acceptance. The active user Slicer and
MRI Agent sessions were not restarted or modified.

---

## 22 October: Original Tensor-Helper Dispatch

The earlier undeployed tensor-helper candidate now has source guards, isolated
instances, edge-case regression and complete-stage comparisons. The
[dispatcher](../../microstructure/freewater_dispatch.py) calls the exact original
`decompose_tensor` and `from_lower_triangular` bodies through a private proxy.
It removes only repeated keyword/signature compatibility checks in WLS tensor
decomposition (also used to initialize NLS). LAPACK eigendecomposition, eigenvalue
ordering/clipping, solver, initialization, grid and QC are unchanged. Installed
DIPY globals are not patched. Versions 1.11.0/1.12.1 and helper-source hashes must
match; otherwise the public helpers remain in use.

The installed 1.11.0 `dti.py` matches an independently downloaded upstream file
byte for byte. Both paths below use one numerical thread and the original
author WLS as reference, not an older plugin. Three alternating repeats and one
warmup retain all 1728 mask voxels and 288 measurements per native 12-cubed ROI:

| Subject | Python 3.13/DIPY 1.11, author -> optimized | Slicer Python/DIPY 1.12, author -> optimized |
| --- | ---: | ---: |
| 100307 | 2.146 -> 1.812 s, 1.185x | 2.809 -> 2.351 s, 1.195x |
| 100206 | 2.129 -> 1.770 s, 1.203x | 2.821 -> 2.372 s, 1.189x |
| 149741 | 2.109 -> 1.788 s, 1.179x | 2.805 -> 2.348 s, 1.195x |

All 13 raw parameters, four exported maps/masks, affines, dtypes and QC match
exactly. Automatic QC passes while retaining the same 234/672/303 excluded
tissue voxels. These **15.2%-16.9% ROI time reductions** include checking, reading,
fitting and export, not just helper calls. They combine diagonal-free WLS and
helper dispatch; they are not an additional multiplier on earlier reports.

Three complete 6-cubed NLS stages also retain exact raw parameters/maps/QC:
1.170 -> 0.501 s, 1.139 -> 0.513 s and 1.131 -> 0.523 s. These ratios compare
against original author NLS, not the previous optimized NLS. Invalid-tissue
counts remain 16/132/69. Do not compare these smaller ROIs' absolute times with
the historical 12-cubed NLS tests or switch default WLS to NLS to claim a benefit.

Evidence: [WLS JSON](benchmarks/freewater_tensor_helpers_20261022.json)/
[CSV](benchmarks/freewater_tensor_helpers_20261022.csv),
[Slicer WLS JSON](benchmarks/freewater_tensor_helpers_slicer_20261022.json)/
[CSV](benchmarks/freewater_tensor_helpers_slicer_20261022.csv), and
[NLS JSON](benchmarks/freewater_tensor_helpers_nls_20261022.json).
These are shared-host complete ROI stages, not full-brain or whole-workflow ratios.

### Complete Full-Brain Pair for the Current Implementation

A separate full-brain pair has now completed with immutable code/harness
snapshots and fresh outputs. Subject 100307 retains the original **740916 mask
voxels and 288 measurements**, with one numerical thread in both paths:

| Original author WLS | Current equivalent implementation | Saved | Time reduction | Speed ratio |
| ---: | ---: | ---: | ---: | ---: |
| 918.158 s (15.30 min) | 785.648 s (13.09 min) | 132.510 s (2.21 min) | 14.43% | 1.169x |

Independent verification covers all 13 raw fit parameters in 54 computational
chunks, plus FWF, tissue FA, tissue MD and the valid-tissue mask. Arrays, dtypes,
affines and QC match exactly. Both automatic QC checks pass, preserving
**623981 valid-tissue and 116935 excluded voxels**; invalid fits are not relabeled
as valid. Input SHA256 values, frozen runtime and benchmark-harness hashes are
unchanged. Local evidence remains in `ignore/validation/full_freewater_tensor_20261022/`.
Published evidence: [full-brain JSON](benchmarks/freewater_tensor_helpers_full_20261022.json)/
[CSV](benchmarks/freewater_tensor_helpers_full_20261022.csv).

This is **one subject and one pair on a shared Linux host**, optimized first
and reference second, without flushing OS caches or estimating a confidence
interval. Both timed calls include the same raw-parameter evidence export.
Do not compare absolute times across historical runs, multiply earlier ratios,
or call this a whole-workflow improvement. No estimator, thread count, GPU policy,
sampling, precision or QC threshold changes; anatomical review remains pending.

The current regression ran **1298 tests: 1295 passed, 3 skipped, no failures**.
All nine dedicated Free-water tests also passed in Slicer's own Python.

---

## Direct Original-Author WMA Optimization

The reference is unmodified SlicerDMRI/whitematteranalysis
[`filter.py`](https://github.com/SlicerDMRI/whitematteranalysis/blob/4532b911386d2d1bc7ebba63fdd13e65b1660623/whitematteranalysis/filter.py),
pinned at `4532b911386d2d1bc7ebba63fdd13e65b1660623`. Independent downloads of
the pinned filtering, registration-dispatch and license files match local SHA256
values exactly. An older plugin is not substituted for the author reference.
Both implementations run under the same Slicer Python 3.12.10, NumPy 2.2.6,
VTK 9.5.2 and Linux host.

| Real input | Measured streamlines | Author | Optimized | Ratio | Protocol |
| --- | ---: | ---: | ---: | ---: | --- |
| HCP 100206 CSD subset | 50,000 | 6.323 s | 0.110 s | **57.27x** | One warmup, three alternating repeats, medians |
| HCP 149741 CSD subset | 50,000 | 5.899 s | 0.085 s | **69.26x** | Same |
| HCP 877168 CSD subset | 50,000 | 5.841 s | 0.108 s | **53.92x** | Same |
| Complete ORG registration atlas | 1,989,106 | 134.289 s | 2.484 s | **54.06x** | One fresh full-input pair, no warmup |
| Complete 100307 UKF | 277,875 | 37.415 s | 0.690 s | **54.19x** | One fresh full-input pair, no warmup |

**These ratios apply only to attribute-free `preprocess()` filtering and geometry
copying, never complete WMA, bundle quantification or the plugin.** Timings exclude
file loading, comparison hashes, later sampling/registration/classification and
startup. Full-input observations are single pairs, not confidence intervals;
hardware is shared. The three subsets are selected from million-line real inputs
using the author's `downsample`, with predefined seed 2026. Both methods receive
identical inputs; the optimized method does not process fewer streamlines.
Every timed call recomputes the filter, without a result cache.

The [bulk implementation](../../tractography/wma_preprocessing.py) replaces
per-point Python/VTK calls with offsets/connectivity and bounded NumPy block
copies into fresh VTK buffers. A block contains at most **262,144 points**.
Full output and line-count-proportional indices still require memory; this is
not a whole-process constant-memory claim.

The author's first-eligible-line step-size estimator, accumulation order,
point-count thresholds, Python rounding, 40/260 mm limits, point/line order,
shared-point duplication and float32 `vtkPoints` output are preserved. It does
not substitute true per-line arc lengths or introduce a new precision reduction.
Only originally attribute-free calls are accelerated; requested attributes,
indices, other filters or unsupported layouts run the author implementation.
Sampling, optimization, transforms, outlier rejection and classification are unchanged.

All output coordinate and topology bytes match across the five cases. Original
seeded sampling at 124/125/126 also matches indices, geometry, R/A/S registration
arrays and MT19937 state words. A regression test separately verifies that the
complete random state is not consumed. Source hashes remain unchanged. Edge
tests include float64-to-author-float32 copying, shared/reversed points, empty
selection, 32-bit connectivity, block boundaries and original-code fallback.

An additional real-AF-derived numeric fixture explicitly introduces a synthetic
**+20 mm R displacement**, with 128/64 fixed/moving fibers and a 300-evaluation
budget. The unchanged author affine optimizer produces identical complete
objective histories and nonidentity transforms; both objective changes are
**-58.57054461025365**. This is controlled numeric regression, not acquired data,
whole-brain registration or anatomical accuracy. The unshifted control did not
improve the objective in either route; its unchanged trajectory and unsuccessful
improvement record remain, not relabeled successful registration.

The production [registration entry](../../scripts/run_wma_registration.py)
enables bulk filtering only when author-source hashes match, with the existing
bounded cache available afterward. No user setting, new dependency or GPU-policy
change is required. Logs record real bulk calls, bypasses and implementation
hashes. The verified POSIX hook is integrated; Windows/WSL keeps its original
route and native macOS performance is unmeasured. The exact upstream license
and provenance ship with the extension. Old failed bundles and QC gates are unchanged.

Use the [author-reference benchmark](../../scripts/benchmark_wma_author_preprocessing.py)
to repeat the comparison. The [JSON](benchmarks/author_wma_optimizations_20261020.json)
retains raw times, downloaded-source hashes, full geometry/sampling signatures
and optimizer evidence; the [CSV](benchmarks/author_wma_optimizations_20261020.csv)
summarizes five cases. Original local evidence is under
`ignore/validation/wma_author_bulk_20261020/`. Those substep experiments exclude
complete 74-tract classification; the separate complete-model pair is reported
above. The substep ratios must not be extrapolated to the whole model.

This round's full backend regression ran **1110 tests: 1105 passed, five explicitly
skipped, no failures**. Its final local log is `full_tests_final.log` in that
directory. Packaged documentation links and resource integrity also pass.
This is not full-brain scientific, anatomical or cross-platform
performance acceptance. Current Slicer sessions were not restarted; older
installed copies require redeployment to use the new implementation.

---

## Engineering Appendix: Older Plugin Reference Only

These changes were implemented on **20 October 2026**, rather than merely
documented. They apply automatically, add no dependency/user setting and do not
change device selection or reduce scientific computation.

| Operation | Subject | Before, median | After, median | Ratio | Consistency |
| --- | --- | ---: | ---: | ---: | --- |
| Full-volume six-map ROI export | 100206 | 5.550 s | 0.552 s | **10.05x** | Byte-identical CSV |
| Full-volume six-map ROI export | 149741 | 5.562 s | 0.521 s | **10.68x** | Byte-identical CSV |
| Full-volume six-map ROI export | 877168 | 5.523 s | 0.483 s | **11.44x** | Byte-identical CSV |
| MK-Curve parameter extraction, 1024 voxels | 100206 | 0.528 s | 0.370 s | **1.42x** | Ten exactly equal raw parameter arrays |
| MK-Curve parameter extraction, 1024 voxels | 149741 | 0.524 s | 0.369 s | **1.42x** | Same |
| MK-Curve parameter extraction, 1024 voxels | 877168 | 0.527 s | 0.371 s | **1.42x** | Same |
| Complete MK-Curve 64-voxel numeric chunk | 100206 | 0.940 s | 0.849 s | **1.11x** | Exact coefficients, curves, parameters, masks and residuals |
| Complete MK-Curve 64-voxel numeric chunk | 149741 | 0.951 s | 0.861 s | **1.10x** | Same |
| Complete MK-Curve 64-voxel numeric chunk | 877168 | 0.974 s | 0.877 s | **1.11x** | Same |

The [ROI implementation](../../microstructure/pipeline.py) replaces repeated
whole-volume region/finite masks with one stable grouping of positive labels.
It preserves the original C-order sample sequence and reduction operations,
including per-metric validity, brain masks, nonfinite values, empty estimates and
coverage counts. Grouping indices require memory proportional to labeled voxels;
no peak-RSS reduction is claimed.

The [MK-Curve tensor routines](../../microstructure/mkcurve_tensor.py) cache
constant monomial metadata and two read-only quadrature bases for the author's
256 directions, totaling **42 KiB per process**. They do not cache voxel signals
or predictions. WLLS, QR fallback, directions, curve samples, thresholds and
validity conditions remain unchanged. Parameter extraction saves about 29.7%;
the complete numeric chunk saves **9.46%-9.97%**, not the same larger percentage.

Protocol: three independent real HCP subjects, an immutable pre-edit snapshot
versus current code, one warmup per method, three alternating-order repetitions
and median wall times in the same process. OMP/OpenBLAS/MKL were each set to one
only for these CPU benchmarks, not plugin defaults. ROI timings include reading,
validation, signatures, statistics and CSV/JSON writing for the full
`145 x 174 x 145` grid, FA/MD/AD/RD/L1/Trace and the brain mask. JSON also matches
except its runtime field.

Parameter tests uniformly select 1024 existing real coefficient rows per subject.
Full numeric chunks use 64 finite positive-signal voxels from predetermined
uniform raw-DWI candidates, retaining **200/201/200 synthetic-b0 samples** and
181 measurements. Chunk work includes WLLS, curves, detection/correction,
parameters, validity and residuals, but excludes input decompression/preparation
and full-brain file export. **The reference is the preceding Python implementation,
not author MATLAB; these ratios are not full-brain or workflow speedups.**

Input hashes remain unchanged; no old fit/failure is overwritten. The host is
shared, not exclusive, and native Windows/macOS performance and improved accuracy
are unmeasured. See [new CSV](benchmarks/compute_optimizations_20261020.csv) and
[complete JSON](benchmarks/compute_optimizations_20261020.json). The immutable
snapshot, real signal subsets, script and logs remain in local
`ignore/validation/compute_optimization_20261020/`. New
development-checkout tests in `Testing/Python/test_compute_optimizations.py` cover
read-only caches, exact sample order, sparse/empty labels and nonfinite inputs.
Acceleration does not turn previous invalid estimates into valid ones.

That earlier round's full backend regression: **1103 tests, 1098 passed, five explicitly skipped,
no failures**; the local log is `full_tests.log` in that directory. This does not
replace GUI, anatomical or cross-platform scientific acceptance. Running Slicer
sessions were not restarted; an older installed copy must be redeployed to use
these changes.

---

## Historical Stage And Device Comparisons

Speed ratio = `reference_seconds / optimized_seconds`.
Time reduction = `100 * (1 - optimized_seconds / reference_seconds)`.

| Operation | Reference | Optimized | Ratio | Reduction | Output check / scope |
| --- | ---: | ---: | ---: | ---: | --- |
| WMA registration-input preparation | 533.775 s | 180.550 s | **2.96x** | 66.17% | Three fixed-seed calls, including initial cache fill; bitwise-equal registration inputs |
| Shared mapping of four surface overlays | 13.379 s | 3.519 s | **3.80x** | 73.70% | One real left hemisphere; four commands versus one multi-frame command; four byte-identical MGH outputs |
| Three-slab DWI reads | 37.252 s | 17.079 s | **2.18x** | 54.15% | Includes sequential decompression; exact slab arrays and affine; unchanged source |
| SynthStrip CPU to GPU | 14.268 s | 9.494 s | **1.50x** | 33.46% | Same input/script/weights/runtime; Dice 0.999993918, two differing voxels |

These are **not whole-pipeline speedups**. Surface mapping measures command
execution, excluding extra Python packing, validation and publication overhead.
DWI timings do not measure FW fitting. SynthStrip is a single device pair using
the upstream GPU implementation, not a new estimator. The first three rows each
use one subject; three WMA seeds are sampling iterations, not independent
end-to-end trials. No confidence intervals are available, and local savings
cannot be added together to predict workflow acceleration.

### WMA Cache

The [bounded cache](../../tractography/wma_registration.py) and
[author-entry adapter](../../scripts/run_wma_registration.py) retain deterministic
length-filtered atlas/subject polydata. The full inputs contain 1,989,106 and
277,875 streamlines. Original random sampling, random state, objectives,
transforms and classification remain unchanged. Only verified author-source
hashes enable interception; unknown versions run unmodified. VTK modification
times and filter settings invalidate entries, and attribute/index requests bypass
the cache. `WMAOrgAtlasConfig.registration_preprocess_cache` defaults to true.

The current memory-bounded revision separately took **181.542 s**, or **2.94x**
against the preserved author reference. This is not another independent paired
control. Peak retained cache memory was **3,374,317,568 bytes / 3.14 GiB**, not
whole-process peak RSS. Its budget is one quarter of available RAM, capped at
4 GiB, or a conservative 256 MiB without memory discovery. Lower memory can
reduce hits, not scientific requirements. The POSIX adapter optimization is not
enabled on the current Windows/WSL route; native macOS performance is unmeasured.

A fresh full WMA run took **4664.826 s**, with a separate unchanged 95,755-track
assignment audit. It lacks a matched complete no-cache run, so **2.96x must not
be attributed to complete WMA**. Anatomical review remains separate. See
[workflow evidence](WORKFLOW_GENERALIZATION_20261017.md).

### Surface Postprocessing

This optimization is **inherited from pinned DDSurfer upstream `c3b2d40`**, not
a novel plugin reconstruction algorithm. The [overlay writer](../../anatomy/surface/ddsurfer_backend/postprocessing/overlays.py)
maps thickness, curvature, sulcal depth and pial area together with
`mri_surf2surf --split`. Each result has 163,842 target vertices; values,
geometry, headers and additional MGH bytes match separate commands exactly.
The displayed ratio is recomputed from the plugin's own preserved command record.

Python regional labels, in-process statistics, shared inputs and bilateral
thread budgeting are also inherited. Plugin verification found 109 byte-identical
regional labels and matching statistics, but has no complete paired replacement
timings for those operations. Additional upstream warm-file observations are
documented in its [optimization notes](../../anatomy/surface/ddsurfer_backend/postprocessing/OPTIMIZATIONS.md),
not presented as plugin end-to-end benchmarks. Scientific registration/thickness
commands are retained.

### DWI Access

The [reader](../../shared/diffusion_io.py) decompresses `.nii.gz` once to
temporary disk, reads it through memory mapping and cleans up on exit. The
unchanged real input is `145 x 174 x 145 x 288`; z=0/72/144 were compared without
`indexed_gzip`. This is one timing pair with uncontrolled OS page cache, not a
storage-independent guarantee or a measured percentage reduction in peak RSS.
Sufficient temporary disk is required. A separate exact 64-voxel FW fit does
not establish full-brain fitting speed. See the [I/O benchmark](../../scripts/benchmark_diffusion_chunk_io.py).

---

## Observations And Unmeasured Benefits

| Route | Evidence | Permitted interpretation / limit |
| --- | --- | --- |
| NODDI CPU parallelism | Historical one-thread 1784.280 s; later eight-thread 343.098 s; observed ratio 5.20 | Six saved arrays match exactly. Different-time observations are not a controlled 5.20x speedup or an improvement over AMICO itself |
| DTI export reuse | Fresh eight-map export/QC 1.102781 s; reuse 0.001251 s; output timestamps unchanged | Avoids a completed stage; not approximately 881x faster tensor estimation |
| MK-Curve Python port | Chunking, disk backing, optional processes and condition-checked batched WLLS with pivoted-QR fallback | Numeric reference/full-brain records exist, but no matched full author MATLAB/Octave-versus-Python timing |
| DDSeg Python/PyTorch port | Converted network weights, three-view inference and automatic device selection | Reduces MATLAB deployment requirements; no complete original-runtime per-view accuracy/speed comparison |
| Native DDSurfer, DDParcel, TractCloud | Upstream networks and backend-aware GPU selection; scoped TractCloud CUDA run 18.50 s | Device use and a runtime record do not demonstrate acceleration over the author algorithm |
| SAF Python port | Partial author-kernel comparisons, explicit geometry fixes and real candidate-track checks | No full original-runtime timing or clustering-output equivalence claim |

NODDI uses upstream `nthreads`, automatically bounded by available CPUs and eight;
explicit positive counts remain supported. AMICO BLAS threads are set to one to
avoid nested oversubscription; kernel reuse is protocol/version keyed. Compared
arrays are NDI, ODI, IsoVF, RMSE, NRMSE and the validity mask. Compressed files
can differ while decoded voxels agree. Historical load/cache/environment were
not controlled for causal speed attribution.

MK-Curve also retains extreme synthetic-curve comparison failures. Parameter
agreement must not become an unconditional all-curve equivalence or valid-fit
claim. See [implementation provenance](../architecture/IMPLEMENTATION_PROVENANCE.md)
and [numerical limits](WORKFLOW_GENERALIZATION_20261017.md).

---

## GPU Is Not A Speed Guarantee

PNL calls the official Python/TensorFlow CNN and ANTs; it is not a rewrite of the
whole PNLpipeline. Same-input CPU took **78.961 s**, default Auto GPU **84.437 s**
and explicit CUDA repeat **79.775 s**. Auto was about **6.94% slower** in this
wall-time pair. Startup/JIT, registration, hashing and b0 preparation can dominate.
The historical **346.70 s** CPU record has incomplete environment metadata and
cannot support a fourfold GPU acceleration claim.

SynthStrip's 1.50x ratio is a single same-input device measurement; Dice is
CPU/GPU agreement, not accuracy against expert masks. Real CUDA eddy execution
is recorded, but no matched CPU timing exists; roughly 719 s measures complete
preprocessing, not eddy alone. Failed/stopped runs are not successful baselines.
See [GPU evidence](GPU_POLICY_20261013.md).

---

## Metrics And Reproduction Limits

Recorded metrics include scoped wall time, ratios, time reduction, array/byte/
geometry agreement, device-mask Dice and differing voxels, and WMA retained-cache
bytes/hits/misses. Whole-process RSS, peak VRAM, energy and throughput do not have
a uniform paired original-versus-optimized measurement. Integrity agreement is
not anatomical or clinical accuracy.

The initial evidence review reran the WMA-cache, diffusion-I/O, scalar-cache and
MK-Curve Python suites: **36 tests passed**. The subsequent implementation changes
also passed the full regression reported above. Unit testing is implementation
regression, not a scientific timing/accuracy trial. The CSV/JSON retain unrounded times, comparison
categories, source report paths/hashes and limitations, without imaging data.
Original evidence lives under ignored local `ignore/validation/`; a GitHub reader
can inspect the saved numerical summary, not independently reproduce the local
benchmark from the summary alone.

Primary local records are `wma_cache_benchmark_20261017/acceptance.json`,
`wma_cache_memory_20261017/acceptance.json`,
`ddsurfer_sync_20261013/real-postprocess-100307/parity.json`, the I/O/refit/cache
records in `pipeline_qc_20261015/`, and the device comparison records in
`pnl_gpu_20261013/`. Exact paths and SHA256 values are in the JSON.
The new original-author comparisons are `three_subjects.json`, `full_inputs.json`,
`upstream_verification.json` and `optimizer_shifted_300.json` under
`wma_author_bulk_20261020/`; the nonimproving control remains
`optimizer_parity_300.json`.

Next benchmarks should fix inputs, parameters, source versions, thread budgets,
devices and cache state; repeat both implementations on multiple independent
subjects while recording wall time, RSS/VRAM, output error and failures. Highest
priorities are multisubject repetitions of the completed WMA/NODDI/default-WLS pairs,
finishing the interrupted complete DKI pair, and full-brain pairs for the new
Free-water NLS and default-QR MK-Curve changes. Default WLS helper dispatch now
has one qualified full-brain pair, but multisubject, repeated measurements
remain needed. Complete 74-tract WMA pairs for the new changes are
still missing; small fixtures cannot predict minutes saved. Original MATLAB MK-Curve
runtime is also unmeasured.
Linux evidence does not certify native Windows/macOS scientific performance.

**Original-author manuscript claim:** with identical inputs and parameters,
equivalent bulk geometry copying accelerated the unmodified author's
attribute-free WMA length-filter stage by 53.92-69.26x on three real-subject
50,000-streamline subsets. Single paired full-atlas and UKF measurements were
54.06x and 54.19x. All output coordinate/topology bytes, fixed-seed samples and
the controlled author-optimizer comparison matched. This optimizes computation,
not the scientific estimator; it is not a complete-WMA or accuracy improvement.

**Historical-stage manuscript claim:** selected real-input stage measurements showed about
2.96x for registration-input caching, 3.80x for shared surface mapping and 2.18x
for sequential DWI decompression, with corresponding output-consistency checks.
These are not universal end-to-end, accuracy or cross-platform improvements.
Shared surface mapping must be attributed to DDSurfer upstream.

The new work can separately be reported as a three-subject, paired **10.05-11.44x**
reduction in full six-map regional-export wall time with byte-identical CSV, and
**9.46%-9.97%** lower complete 64-voxel MK-Curve chunk time with exactly equal
compared coefficient/curve/parameter arrays and masks. The latter is not measured
whole-brain MK-Curve or complete-plugin acceleration.

---

## Full-Pipeline CPU And CUDA Accelerations (2026-11-11)

`docs/release/benchmarks/full_pipeline_accelerations_20261111.json` records 24 end-to-end and stage-complete optimization comparisons against initial unoptimized baselines on `Intel Xeon Platinum 8360Y` + `NVIDIA H100 PCIe 80 GB`:

- **Preprocessing (`MP-PCA + Gibbs`)**: Sliding-window patch vectorization, z-slab RAM capping, and batched PyTorch/cuSOLVER `eigh` + 2D `rFFT/irFFT` CUDA kernels (`136.02 s -> 114.23 s` CPU; `136.02 s -> 7.64 s` CUDA on `sub-dicom`; `4.80 min` CUDA on 288-volume `HCP 100307`).
- **Anatomical priors (`DDSeg`, `DDParcel`)**: VRAM-scaled multi-slice TorchScript/FastSurferCNN batching (`DDSeg` CUDA `8.72 s -> 3.45 s`; `DDParcel` 3-view CUDA `69.02 s -> 23.43 s`), uniform background slice caching, and padded-ROI 3D connected-component/filter post-processing (`97.96 s -> 15.17 s`, bitwise exact).
- **Microstructure (`DTI`, `Free-water`, `DKI`, `MK-Curve`)**: Chunked column-scaled/equilibrated WLS, analytical 3x3 Cardano eigensolvers, vectorized analytical kurtosis invariants, coarse-to-fine synthetic-`b0` search, and FP64 PyTorch/cuSOLVER Cholesky + Gauss-Newton CUDA solvers (`Free-water` `938.28 s -> 312.66 s` CPU, `37.65 s` CUDA; `DKI` `91.60 min -> 1.94 min` CPU, `1.47 min` CUDA; `MK-Curve` `165.90 min -> 22.86 min` 1-worker CPU, `3.30 min` 8-worker CPU, `1.32 min` CUDA).
- **Tractography, Fibre Bundles & Connectome (`DTI-Det`, `SAF`, `WMA`, `Connectome`)**: Active-mask analytical/cuSOLVER peak extraction (`DTI-Det` `68.0 s -> 21.85 s` CPU, `16.41 s` CUDA), spatial-hash BVH and batched CUDA Möller-Trumbore ray-triangle filtering (`SAF`), fused VTK bulk-geometry/input reuse with vectorized CPU and Triton/PyTorch CUDA `whitematteranalysis_gpu` kernels (`WMA` `102.10 min -> 2.75 min` CPU, `2.31 min` CUDA), and batched `cKDTree` / tiled GPU `torch.cdist` radial assignment (`Connectome` `138.0 s -> 5.75–7.57 s`).
- **Hardware Adaptability**: `shared/hardware_adapt.py` scales CUDA batch/chunk sizes to free VRAM, caps numerical threads and z-slab RAM footprints on modest 8–16 GiB / 4–8 GiB GPU machines (`all_peak_vram_under_cap: true` verified under a `3.5 GiB` VRAM cap), halves chunks on `OutOfMemoryError`, and falls back cleanly to CPU when needed.
