# Cohort validation repair and hybrid workload revision

Later on 5 November, live CPU-placement inspection identified cross-campaign
overlap. The CPU v4 reference and MK-Curve Adaptive timings are not accepted
performance comparisons, and the still-waiting Hybrid v8 was superseded by v9
before scientific execution. See
[the subsequent isolation review](MKCURVE_ISOLATION_REVIEW_20261105.md).
The account below records the earlier output-reader repair, not final timing
acceptance.

## Diagnosis

The CPU v4 Adaptive arm completed all 12 subjects in 35,044.3 seconds
(584.1 minutes), and its independent automatic QC passed 12/12. Its subsequent
fingerprint reader stopped on a valid, empty WMA intermediate cluster. Empty
atlas slots can occur after classification, outlier removal and hemisphere
separation; they are not equivalent to a truncated file or an empty final tract.

Hybrid v7 is a separate, genuine sampling failure. All four launched subjects
completed anatomical inference, microstructure, CSD/ACT/SIFT2 and connectomes.
WMA produced 74 anatomical tract files per subject, but 6, 6, 7 and 3 tracts,
respectively, were below the existing 20-streamline sampling threshold. One
tract was empty. These outputs and the failed timing record remain unchanged;
this experiment is not an accepted performance comparison.

## Repairs

- The fingerprint reader hashes valid empty WMA intermediate cluster slots.
  It still rejects malformed XML, decode errors and empty final tract/surface
  products. Nonempty-product hashing is unchanged.
- A separate verification directory can reconstruct a missing inventory only
  for the documented intermediate-cluster reader error, with passed original
  independent QC and completed computation. It checks retained file timestamps
  against trial completion and file metadata before/after reading. The first
  successful content inventory is explicitly recorded as post-run, not as a
  pre-existing content hash. Original timing/QC records are not overwritten.
- Post-validation comparisons use their own inventory files and retain all
  input-integrity, exact-product and independent-QC requirements.
- Optional batch stage selection now uses the subject CLI's existing option
  resolver. Selecting a UKF bundle source no longer erases an explicitly
  requested CSD stage from the resource plan.

No streamline-sampling thresholds, model settings or scientific QC criteria
are relaxed. Neither active CPU v4 nor MK-Curve execution is modified.

## New Hybrid Comparison

The new workload is frozen separately at:

`/data/chengjin/SlicerDDSurfer-test/cohort-heterogeneous-20261105-hybrid-v8/`

It compares Subject-level and Adaptive using the same 12 acquisitions, fresh
outputs, 16 CPU threads on CPUs 56–71, 128 GiB RAM, four-thread task settings,
and one admitted GPU stage. Its selected analyses are DDSeg, DDParcel,
DDSurfer, DTI, Free-water WLS, NODDI, CSD/ACT with one million streamlines,
SIFT2 and MRtrix connectomes. WMA is evaluated by the separate CPU UKF/WMA
pair; it is not silently counted as successful in the new hybrid experiment.
The new comparison has a different workload scope from v7 and cannot be
combined with v7 timings.

Dependency and task-plan preflight passed with no blocked stages. At launch
both GPUs were busy, so the controller waits for the existing consecutive-idle
device checks; it does not terminate or pre-empt other GPU jobs. Once admitted,
the two timed arms run automatically. An independent frozen-code verification
watcher is attached to both this campaign and CPU v4 (`post_validation_v2`).

## Verification

- Full software suite: 1,885 tests, nine skipped, no failures (156.500 seconds).
- Official Slicer Python: 36 targeted tests passed.
- The actual CPU v4 cluster that triggered the failure can now be hashed in
  intermediate-cluster mode, while strict final-product mode still rejects it.
- Regression coverage includes malformed empty files, missing inventory
  provenance, products changed after timing, immutable timing records and
  mixed tracking/bundle-source resource planning.

Logs are retained in the new campaign's `repair_validation/` directory. No
cohort acceleration is accepted until both timed arms and scientific
comparisons finish successfully.
