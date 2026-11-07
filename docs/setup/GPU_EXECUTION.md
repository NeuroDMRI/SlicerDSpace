<div align="center">

# GPU Execution Policy

[Project](../../README.md) | [Documentation](../README.md) | [Brain Masking](brain_mask_gui_config.md) | [中文](GPU_EXECUTION.zh-CN.md)

</div>

---

## Defaults

Supported neural inference defaults to `auto`, preferring the first GPU visible
to the **actual inference Python**. CPU is used when that runtime has no usable
CUDA device or the user explicitly selects CPU. A failed probe or CUDA execution
is an error, not permission to silently retry the same model on CPU.

GPU visibility imposed by a scheduler, container or caller is preserved. The
plugin does not unmask GPUs, choose every GPU, or download a new CUDA stack while
running an analysis. Installing a CPU-only framework on a GPU-equipped machine
does not make CUDA available: fix that backend's environment first.

---

## Route Inventory

| Route | Policy / implementation |
| --- | --- |
| PNL CNN masking | `auto` in external PNL PyTorch; first caller-visible CUDA GPU or CPU; explicit CPU/CUDA options |
| SynthStrip masking | `auto` in FreeSurfer fspython; GPU flag supplied when CUDA is usable; preserves caller visibility |
| DDSeg | `auto` at GUI, CLI and direct API; tensor inference resolves in its executing Python |
| DDParcel | `auto`; follows the GUI PyTorch choice, including standalone legacy entrypoints; explicit CUDA cannot silently become CPU |
| DDSurfer surfaces | `auto`; resolves in its selected external Python, not the calling Slicer Python; preserves model memory/precision checks |
| TractCloud | `auto` in its selected Python; GUI PyTorch choice forwarded; CPU retry after a CUDA inference error is refused |
| MRtrix/FSL eddy | GPU-first when its CUDA binary and GPU access are available; launcher refuses upstream CUDA-error CPU retries |
| MRtrix dwi2mask | CPU implementation; no CUDA option in this route |
| DIPY denoising/Gibbs/motion | Current CPU implementations; not made into a different algorithm to claim GPU acceleration |
| Slicer DTI, DIPY free-water/DKI, Python MK-Curve | Current CPU implementations |
| AMICO NODDI | Current CPU fitting backend; threading is separate from GPU availability |
| ANTs/BRAINSFit, FreeSurfer surface refinement | Current CPU registration / geometry backends |
| MRtrix CSD/ACT/SIFT2, WMQL/WMA, SAF filtering | Current CPU implementations / external tools |
| Connectomes, regional statistics, file conversion | Current CPU numerical / I/O operations |
| Slicer rendering | Slicer's graphics context, independent of inference device |

CPU checkpoint validation and tensor-to-NumPy/file transfers are not CPU model
inference. Do not remove these necessary operations when searching for `.cpu()`.

---

## Controls And Provenance

**Advanced settings > Preprocessing & mask** has the persisted `auto/cuda/cpu` mask
selector. **Advanced settings > Run & performance** has the unified PyTorch choice
for DDSeg, DDParcel, DDSurfer and TractCloud. Existing explicit choices are
respected; the defaults do not overwrite a user's saved CPU preference.

```bash
PythonSlicer shared/run_subject_pipeline_cli.py \
  --subject-root /data/sub-01 --subject-id sub-01 \
  --targets microstructure --execute \
  --brain-mask-device auto --ddseg-device auto \
  --surface-device auto --tractcloud-device auto
```

Generated mask manifests contain `device_execution`; surface manifests record
`resolved_device`; DDSeg QC records the executing device; TractCloud writes a
device receipt and includes it in its manifest. Source/model hashes remain in
the corresponding scientific provenance.

PNL's known undefined `DEVICE_ID_LIST` startup branch is replaced **in memory**
by the launcher. SynthStrip's old fixed physical-GPU-0 assignment is likewise
suppressed in memory. Neither adjustment changes weights, CNN operations,
registration or cleanup. The pinned DDSurfer snapshot remains unmodified;
the facade supplies its generated native mask explicitly so its standalone
CPU-only SynthStrip helper is not used by SlicerDSpace.

---

## Compatibility Boundaries

[PNLBrainMask-Torch](https://github.com/ChengjinLii/PNLBrainMask-Torch)
uses converted PNL weights and PyTorch inference. Managed installation defaults
to the paired-tested CPU build; an explicit CUDA route needs a compatible PyTorch/CUDA runtime.
[SynthStrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/) uses FreeSurfer's
own interpreter by default. For newer GPUs, set optional `synthstrip_python`
under **Environment and installation > External Runtime Paths** to a
compatible independent Python with PyTorch, surfa and NumPy. The same official
FreeSurfer script/weights are used. Known unvalidated old-CUDA/new-GPU combinations
are blocked before model processing; they do not silently select CPU.

MRtrix `dwifslpreproc` itself normally retries CPU after a CUDA error; the
SlicerDSpace launcher prevents that retry. CPU-only discovery is allowed when
there is no CUDA eddy backend. Official CUDA eddy 2602.0 was installed in a separate
local FSL environment and passed a real 31-volume preprocessing acceptance.
The preceding CPU-only FSL environment is unchanged. Managed Linux setup can now
install that optional backend and save its path automatically; see
[managed setup](MANAGED_SETUP.md). The launcher sets CUDA eddy's CPU helper-thread
count to one, as required by the executable, while preserving the user's count
for CPU eddy. Requested and effective arguments are both recorded.
This route is independent of PyTorch and mask selectors. To deliberately suppress
eddy GPU access, set `CUDA_VISIBLE_DEVICES=-1` in its launch environment. See the dated
[verification record](../release/GPU_POLICY_20261013.md) for actual tests and
remaining limitations.
