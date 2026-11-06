# SlicerDSpace Hardware Requirements

Assessment date: 2026-11-05. This lightweight assessment reuses existing real-data runs and reads their monitoring logs and retained file sizes. No algorithms were rerun, data downloaded, environments installed, or ongoing experiments changed.

## Environment And Input

- Host: two Intel Xeon Platinum 8360Y CPUs, 72 physical cores / 144 logical CPUs, 503.5 GiB RAM; two NVIDIA H100 PCIe GPUs, approximately 80 GiB each. This is the observation host, not a user hardware requirement.
- Environment: Ubuntu 22.04.5 LTS, 3D Slicer 5.10.0 / Python 3.12.10. The archived environment records PyTorch 2.11.0+cu129, DIPY 1.12.1 and AMICO 2.1.1.
- Input: HCP Young Adult participant `100206`, existing corrected dMRI and matching gradients/mask; no acquisition preprocessing is repeated. Shape: `145 x 174 x 145 x 288`, 1.25-mm isotropic, float32; 18 b0 volumes and 90 directions at each of approximately 1000, 2000 and 3000 s/mm2.
- The compressed input bundle occupies 1.39 GiB; the DWI array alone is 3.93 GiB when uncompressed. Input source: `/dataset/nfs_share01/HCP/HCP-YA/3T/Diffusion_Preprocessed/100206/T1w/Diffusion`.
- Representative computations: fresh native WLS DTI and scalar export; DDSeg, DDParcel and bilateral DDSurfer; MSMT-CSD, ACT/iFOD2 with 1,000,000 streamlines, SIFT2 and TCK/TRK/VTK export. Surface postprocessing is not included.

## Measured Resource Observations

**S** is a single participant's sampled process tree, including its subprocesses. **B** is the entire four-participant batch tree during the indicated participant's task window. B values can include other participants and other tasks; they are not isolated module peaks. GPU slots are serialised, but other processes may retain CUDA contexts.

CPU use is CPU-seconds divided by wall-seconds: 1.0 core equivalent means approximately 100% of one logical CPU. RSS is summed across the owner and descendants; shared pages can be counted more than once. Existing sampling intervals are 0.2 s for S and approximately 1.2 s in the B trace (1 s configured, plus monitoring overhead). Between-sample peaks and short-lived children can be missed. GPU figures use `nvidia-smi` memory for owned process IDs, not total device memory occupied by other users.

| Computation / observed window | Wall time | Mean CPU core equivalents | Sampled peak RAM (GiB) | Sampled peak owned VRAM (GiB) | Scope |
| --- | ---: | ---: | ---: | ---: | --- |
| Fresh native WLS DTI preparation and fitting | 24.20 min | 3.79 | 8.43 | CPU only | S; four requested threads |
| DTI scalar export from that prepared tensor | 0.85 s | 1.01 | 0.59 | CPU only | S; only five samples; excludes tensor fitting |
| DDSeg admitted GPU window | 17.90 s | 1.80 | 15.37 | 1.50 | B |
| DDParcel admitted GPU window | 2.73 min | 6.95 | 30.23 | 5.84 | B |
| DDSurfer admitted GPU window | 2.22 min | 6.54 | 23.00 | 11.10 | B |
| CSD/ACT, SIFT2, conversions and output checks | 402.20 min | 3.01 | 92.06 | CPU computation | B; RAM/CPU also include concurrent tasks |

The anatomy windows include loading, inference and associated I/O, not just neural-network prediction. The enclosing DDSeg/DDParcel/DDSurfer task durations were 106.15 s, 2405.55 s and 133.79 s; upstream preparation, registration and waiting are included where applicable. They should not be substituted for the GPU-window timings above.

The hybrid batch requested 16 CPU threads overall, four per task, on a shared host. CSD commands used four threads except `tckgen -nthreads 0`, which disables multithreading; tracking alone took 340.88 min. This deliberately serial tracking setting and host contention make the recorded CSD time unsuitable as an ordinary-workstation speed prediction. See the [MRtrix3 tckgen documentation](https://userdocs.mrtrix.org/en/latest/reference/commands/tckgen.html#standard-options).

Across the complete archived hybrid batch, the sampled tree peaks were **92.06 GiB RAM** and **11.10 GiB owned VRAM**. Neither number is a single-subject minimum. The selected modules completed; subsequent WMA sampling QC failed, so this record is not a successful whole-workflow performance comparison. Isolated anatomy/CSD CPU and RAM peaks are unavailable; the batch-context numbers above are retained rather than assigning them to individual modules.

## Measured Disk Use

These are retained files measured on the assessment date, not historical peak scratch usage. Logical sizes sum file lengths; allocated sizes use `st_blocks * 512`. Regular files are counted once per inode, and input symlinks are not counted as duplicate copies. GiB means 2^30 bytes.

| Retained component | Logical size (GiB) | Allocated size (GiB) |
| --- | ---: | ---: |
| DWI, b-values, b-vectors and supplied mask | 1.393 | 1.393 |
| Native DTI caches, LS/WLS | 0.240 | 0.240 |
| Registered DTI cache | 0.194 | 0.194 |
| DTI scalar outputs | 0.045 | 0.045 |
| DDSeg outputs | 0.012 | 0.012 |
| DDParcel outputs | 0.075 | 0.075 |
| DDSurfer outputs and retained preparation files | 0.352 | 0.353 |
| ACT preparation / 5TT products | 0.015 | 0.015 |
| CSD working products, including DWI/FOD MIF files | 5.220 | 4.801 |
| Tractograms in TCK/TRK/VTK, SIFT2 weights and mu | 5.620 | 5.620 |
| Shared segmentation/parcellation/surface/Resources model directories | 1.078 | 1.078 |
| **Combined selected data, products, caches and model directories** | **14.24** | **13.83** |

The selected output directories alone contain 101 files, 11.34 GiB logical / 10.92 GiB allocated. The persistent `cache/diffusion_inputs` directory contains headers and receipts only, approximately 28 KiB logical. Completed command scratch and temporary decompressed arrays are no longer present; **their historical peak disk use was not recorded**. Allow an additional **20-30 GB of temporary workspace per active participant as an estimate**, not a measurement. One uncompressed DWI copy would already require approximately 3.93 GiB.

The total excludes Slicer/Python/external-tool installations, GUI scene caches, other algorithms' results, WMA atlases, optional surface postprocessing, backups and future participants. Most retained output bytes here are tractograms and CSD working files. Model-directory totals describe the existing selected directories, not the download size of all optional resources.

## Recommended Configuration: Estimates

These are conservative planning estimates for ordinary single-participant use, not experimentally established minimum requirements. No new run on a 32/64-GB desktop or 16-GB GPU was performed. Recommended GB values follow common hardware specifications; measured tables above use GiB.

| Intended use | CPU | RAM | GPU | Free SSD storage |
| --- | --- | --- | --- | --- |
| Limited DTI / CPU-only exploration | 4-8 physical cores | 32 GB | CUDA not needed for DTI or CSD/ACT; Slicer-compatible graphics for review | Approximately 50 GB for a limited installation and one participant |
| Preferred single-participant anatomical, microstructural and tracking workflow | 8-16 physical cores | 64 GB | CUDA-capable NVIDIA GPU with 16 GB VRAM suggested; 24 GB provides more headroom for inference and visualisation | Approximately 100 GB, including software/model installation allowance and 20-30 GB temporary workspace |

The 64-GB RAM recommendation allows room beyond the measured 8.43-GiB DTI tree and the anatomy-window batch observations for preparation, retained arrays, Slicer and the operating system. The 16/24-GB GPU suggestions add headroom beyond the observed 11.10-GiB owned batch allocation; that observation is not proof of a minimum supported card size. CPU inference is available for the anatomical models, but archived timings indicate substantially longer execution, especially for DDParcel.

For cohorts, increase retained-result storage with participant count and temporary space with active concurrency. Increase RAM or reduce concurrency for larger images, more volumes, additional models or more streamlines. The 92.06-GiB four-participant observation supports caution about batch RAM use, not a promise that all four-participant workloads fit a particular configuration. Slicer's own graphics/display guidance is documented in its [official hardware recommendations](https://slicer.readthedocs.io/en/latest/user_guide/getting_started.html#recommended-hardware-configuration).

## Manuscript-Ready Computing Hardware

> **Computing hardware.** The reference environment is an x86-64 Linux workstation running Ubuntu 22.04 and 3D Slicer 5.10.0. For single-participant processing, we recommend an 8-16-core CPU, 64 GB RAM and approximately 100 GB of free SSD storage for software, model weights, inputs, outputs and temporary files. A CUDA-capable NVIDIA GPU with 16 GB VRAM is suggested for DDSeg, DDParcel and DDSurfer inference; 24 GB provides additional headroom for inference and interactive visualisation. DTI and CSD/ACT run on the CPU. Larger datasets and concurrent participants require additional memory and storage. These specifications are conservative planning recommendations informed by existing resource observations, rather than validated minimum requirements.

## Evidence And Limits

- Single-participant DTI measurements: `/data/chengjin/SlicerDDSurfer-test/cohort-heterogeneous-20261104-cpu-v4/trials/ukf-wma-adaptive/schedule_dc72ae5e16fb429d980a1c9175477b8d/events.jsonl`, participant `100206`, completed `native_dti` and `microstructure_dti` events. Execution times, not queue-inclusive event elapsed times, are used.
- Anatomy/CSD resource trace: `/data/chengjin/SlicerDDSurfer-test/cohort-heterogeneous-20261104-hybrid-v7/trials/hybrid-subject/resources.jsonl`; GPU windows: `gpu_guard/gpu_events.jsonl`; participant task intervals: `outputs/100206/00_run_metadata/benchmark_orchestration_manifest.json`. CSD settings/timings: `outputs/100206/04_tractography/csd/benchmark/csd_act_manifest.json`, relative to the same trial.
- Environment/input provenance: the corresponding CPU v4 and hybrid v7 `campaign.json` records and current `lscpu`, `free -b`, `/etc/os-release` and `nvidia-smi` inventory. Disk inventory uses that hybrid trial's participant input/cache and selected output directories, plus the four model directories under `/data/chengjin/SlicerDDSurfer-models`.
- Each table row reuses one existing observation. Timings are neither repeated benchmarks nor additive workflow estimates. No isolated anatomy/CSD process-tree peaks, peak scratch-disk samples, GUI-inclusive VRAM peaks or cross-platform hardware minima are established by this assessment.

Only this Markdown report is produced; manuscript sources, algorithm code and Git history are unchanged by this assessment.
