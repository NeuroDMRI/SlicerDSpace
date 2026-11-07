<div align="center">

# UKF Tractography

**Optional official Slicer extension · explicit, traceable defaults**

[Project](../../README.md) | [Setup](../README.md) | [Author module](https://github.com/pnlbwh/ukftractography) | [中文](UKF.zh-CN.md)

</div>

---

## Select A Route

The default tractography route remains **MRtrix CSD/ACT/SIFT2**. To use UKF,
install **UKFTractography** through Extension Manager (the UKF error message
also names SlicerDMRI, and WMA needs it), restart Slicer, and click **...** beside Tractography to choose **UKF**.
Click **...** beside Fiber bundles to choose **WMA** for automatic
UKF → WMA processing. The ORG atlas and WMA runtime remain separate dependencies.
UKF parameters are under **Advanced settings > Model parameters**. If other
tracking methods are selected too, each produces a separate run variant.

SlicerDSpace invokes the installed author CLI; it does not vendor or rewrite
UKF. Missing dependencies block that route, without silently changing algorithms.
An existing tractogram can still be selected as an optional Advanced WMA input.
UKF itself does not require DDSeg, DDParcel or a cortical surface; WMQL and
connectome analysis still require an appropriate parcellation.

---

## Author Defaults

Defaults follow the [author's module XML at the reviewed source revision](https://github.com/pnlbwh/ukftractography/blob/2d2b661539700aa825c1949823119601e11e1823/UKFTractography/UKFTractography.xml).
These are **author-module defaults**, not a universally optimal acquisition
protocol, an automatically validated WMA sampling protocol, or subject-tuned settings.

| Parameter | Default | Meaning |
| --- | ---: | --- |
| `numTensor` | 2 | Two-tensor UKF |
| `seedsPerVoxel` | 1 | Seeds per eligible voxel, not a target streamline count |
| `seedingThreshold` | 0.18 | Minimum seed FA |
| `stoppingFA` | 0.15 | Terminating FA |
| `stoppingThreshold` | 0.10 | Terminating mean normalized signal, not the deprecated GA threshold |
| `stepLength` | 0.3 mm | Integration step |
| `recordLength` | 0.9 mm | Output point spacing |
| `maxHalfFiberLength` | 250 mm | Maximum **half-fiber** length, not a 250 mm full-fiber cap |
| `freeWater` | false | Optional UKF free-water compartment, distinct from voxelwise FW fitting |
| `numThreads` | −1 | Author CPU automatic selection; GUI 0 means automatic |
| `Qm / Ql / Qw / Rs` | 0 | Author model-specific automatic noise, not zero-noise fitting |
| `recordFA / recordTensors` | true | Retain author-default tensor/FA arrays in VTK |

Only UKF settings appear when that backend is selected; the ACT streamline budget
is hidden and ignored. Restore author defaults using the button in the UKF group.
UKF's upstream implementation is CPU-threaded; this is not a plugin policy that
disables GPUs for other algorithms. All supplied diffusion volumes are used;
no implicit shell selection or density/threshold escalation occurs after QC failure.

---

## Input And Output Integrity

Inputs need finite DWI values, valid b-values/b-vectors, a rank-six tensor
direction design, and a finite binary mask on the native grid. The adapter
intersects that mask with voxels having finite, positive mean b0 signal and
saves `<subject>_ukf_valid_mask.nii.gz` as a separate tracking input. It records
excluded counts/fractions and warnings in the manifest and QC, without changing
the source DWI or supplied mask. An empty effective mask or exclusion exceeding
1% stops execution for input review; this is a conservative input-integrity
guard, not an anatomical-accuracy threshold. The derived mask is included in
cache integrity checks and is used for conversion, tracking and overlap QC.
These checks are necessary, not sufficient evidence of a scientifically suitable
two-tensor acquisition. Review acquisition quality and model suitability separately.

The adapter inspects the **installed** CLI schema before tracking. Some upstream
builds emit a complete `--xml` description and subsequently return an error when
attempting empty input; only the complete parseable XML is accepted in this case.
The probe return code is recorded. Conversion and tracking must still exit successfully.
Boolean output switches are passed explicitly: SEM-generated CLIs can initialize
them to false even when the GUI XML says true. Missing requested tensor/FA arrays
therefore fail output QC rather than being accepted as a complete result.

The tested DWIConvert only writes short images. Enabling `allowLossyConversion`
would truncate floating-point preprocessed signals, so the adapter **does not do
that**. DWIConvert instead converts a tiny integer-coded geometry/gradient probe.
Its output must be an exactly verified axis permutation/flip with unchanged
volume order. The original DWI values are copied losslessly into that official
header, and the mask receives the identical mapping. Any required resampling or
unrecognized coordinate convention is rejected. Gradient metadata comes from
the official converter, not an independently guessed FSL-to-world rule.

VTK remains the primary output with the author's tensor arrays. A geometry-only
TRK copy supports WMQL; every streamline and point is checked against VTK within
0.0001 mm. Native-mask geometry checks, input/binary/converter SHA256, exact
commands, CLI XML and runtime are saved under the run's `tractograms/ukf` directory.
Only matching completed, hash-valid outputs are reused. Failed QC cannot become a
valid cache, and downstream consumers cannot use stale UKF results after failure.
UKF connectomes use the Python backend; CSD SIFT2 weights are never reused for UKF.

**Geometry QC is not anatomical validation or WMA bundle-sampling QC.** WMA,
WMQL and TractCloud retain their own checks and review requirements. Windows and
macOS need compatible official extension builds; a Linux test is not evidence of
complete cross-platform scientific acceptance.

---

## Command Line

```bash
PythonSlicer run_subject_pipeline.py \
  --subject-root /path/to/subject --subject-id subject \
  --targets tractography,bundles --tractography-backend ukf \
  --bundle-backend wma --execute
```

Usual WMA atlas/runtime arguments still apply. Headless environments can provide
`--ukf-executable /path/to/UKFTractography` or `SLICERDSPACE_UKF_EXECUTABLE`.
The GUI resolves the installed extension automatically and forwards its path.
`DWIConvert` is discovered from Slicer; fully standalone environments can set
`SLICERDSPACE_DWICONVERT_EXECUTABLE`. Optional overrides use, for example,
`--ukf-parameters '{"seedsPerVoxel":2,"numThreads":8}'` and are recorded verbatim
alongside resolved defaults. No extra training or copying UKF source is required.

To explicitly classify the selected CSD/ACT output instead, use
`--tractography-backend csd_act --bundle-tractography-backend csd_act --bundle-backend wma`.
Run `--plan` before a large batch and resolve any blocked tasks; `--doctor`
checks dependencies but does not establish that every data route is connected.

---

## Verification

The [21 October workflow review](../release/BUNDLE_WORKFLOW_20261021.md) adds
explicit GUI-to-worker converter handoff, TractCloud QC/sampling warnings,
automatic DTI-to-VTK export and decoded real-bundle checks. Historical failures,
current automatic checks and human anatomical review remain separate; saved UKF
results do not validate the latest default protocol end to end.

The [2026-10-20 evidence summary](../release/benchmarks/ukf_integration_20261020.json)
records 1,143 passed tests and five skips, including 23 UKF-specific contracts.
Isolated Slicer 5.10 GUI and compact-layout checks passed. An actual author-source
UKF binary, supplied explicitly, processed a synthetic floating-point tensor
field with otherwise-default settings and two CPU threads: 1,728 streamlines,
100% mask overlap, requested FA/tensor arrays, complete VTK/TRK coordinate
verification and native SlicerDMRI loading. This is not a human-anatomical priors/WMA
acceptance test, a new marketplace installation test, or a cross-platform claim.
