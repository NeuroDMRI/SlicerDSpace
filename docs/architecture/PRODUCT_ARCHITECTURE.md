<div align="center">

# SlicerDSpace Product Architecture

[Project](../../README.md) | [Documentation](../README.md) | [Dependencies](../setup/EXTERNAL_DEPENDENCIES.md) | [Release](../release/EXTENSION_READINESS.md) | [中文](PRODUCT_ARCHITECTURE.zh-CN.md)

</div>

---

## Slicer Client, Verified Backends

SlicerDSpace is a Slicer-native workflow client, not a replacement implementation
of every external scientific tool. The scripted module owns Qt controls, MRML
parameters, task execution, cancellation, logs, run-scoped result loading and
scene review. Support packages own the scientific routes and their provenance.

The [official extension guide](https://slicer.readthedocs.io/en/latest/developer_guide/extensions.html)
allows dependencies on other extensions and third-party libraries. A Slicer-style
implementation means managed execution and scene integration; it does **not**
require rewriting FSL eddy, MRtrix or neural models. Replacing a backend requires
independent scientific validation, not merely matching its output filenames.

---

## Dependency Layers

| Layer | Components | Deployment policy |
| --- | --- | --- |
| Slicer integration | Qt, MRML, imaging I/O, display, the Teem tensor fit in Slicer core; the PyTorch extension (declared), UKFTractography and SlicerDMRI (requested at setup only when a ticked product needs them) | Declare the one extension dependency; use Slicer APIs rather than a separate desktop UI |
| Python scientific routes | DDSeg/DDParcel/DDSurfer, DIPY, AMICO, numerical statistics | Install only selected requirements; protect Slicer's existing Python environment; reuse compatible PyTorch |
| Isolated Python workers | Managed PNL runtime (Python 3.11, PyTorch, ANTs 2.3.0; Linux) independently configured model runtimes | Probe and run in the actual selected interpreter; do not force incompatible stacks into Slicer. On macOS and Windows the PNL mask runs in Slicer's own Python instead (native route) |
| Optional external tools | MRtrix/FSL, FreeSurfer refinement, WMA/WMQL/TractCloud resources | Versioned, explicit paths; dependency checks before execution; official sources and license review |
| Model assets | Weights, templates and atlases | Download selected profiles, verify hashes, retain provenance; licensing is a separate gate |

PNL CNN is the current default mask method. If no input mask is supplied, that
selection requires its PyTorch, model resources and, on Linux, the managed ANTs
runtime; it is **not** a dependency-free default. On macOS and Windows the
native route (SimpleITK registration and PyTorch in Slicer's Python, labelled
"native (experimental)") replaces the managed runtime. A supplied mask or an explicitly selected alternative
can avoid that worker. PNL does not require the large FreeSurfer Docker image used
in some local acceptance tests: that image is a test environment, not a mandatory
plugin dependency. Backend changes must remain explicit, never silent fallbacks.

The basic DTI route must not require installing every advanced backend. CUDA eddy
is an optional **Linux NVIDIA** component of MRtrix/FSL preprocessing; it is
independent of the plugin's PyTorch and brain-mask device selectors.

---

## Packages And Results

| Package | Responsibility |
| --- | --- |
| `shared/` | Input/layout contracts, brain-mask adapters, dependencies, orchestration, QC, settings and GUI result reintegration |
| `preprocessing/` | Optional DIPY or official MRtrix/FSL preprocessing, corrected gradients and diagnostic artifacts |
| `anatomy/segmentation/` | DDSeg tissue segmentation |
| `anatomy/parcellation/` | DDParcel anatomical labels and conversion |
| `anatomy/surface/` | Pinned DDSurfer inference, native surfaces and optional refinement |
| `microstructure/` | DTI, optional FW/NODDI/DKI/MK-Curve, eligibility checks and regional statistics |
| `tractography/` | DTI or MRtrix CSD/ACT, SAF and WMA/WMQL/TractCloud adapters |
| `connectome/` | Python/MRtrix matrices, node tables and quantitative/rendering products |

Key products are registered against their run record and loaded into the active
Slicer scene. TCK display uses a derived RAS VTK preview; source tractograms remain
unchanged. Execution completion, automatic QC and pending human review are
different statuses, not interchangeable assurances of scientific correctness.

---

## Existing Management And Release Gaps

The current extension declares the `PyTorch` dependency (SlicerDMRI is no longer
declared; the tensor fit uses Teem in Slicer core), uses the
PyTorch extension for framework installation, and feature-detects
`slicer.packaging` for Python requirements. Dependency profiles, external worker
paths and model hashes are implemented. [Managed setup](../setup/MANAGED_SETUP.md)
now provisions selected MRtrix/FSL environments on Linux x86_64 and, with macOS 14.5 or later,
MRtrix3 and FSL CPU tools on macOS; on Windows it lists the WSL 2 runtime in the same single
confirmation. It saves their paths automatically. Other external tools still use explicit configuration; this
is **not** a certified one-click installation of every backend on every platform.

Before public distribution, optional runtimes still need platform-specific clean
installation tests, trusted/versioned downloads, an explicit license-acceptance
path where required, and verified cancellation/progress behavior. Do not bundle
FSL binaries under the extension's MIT license: the
[FSL license](https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html) limits most components
to non-commercial use. The
[Extension Index checklist](https://github.com/Slicer/ExtensionsIndex/blob/main/.github/PULL_REQUEST_TEMPLATE.md)
requires trusted binary sources and, at Tier 3, explicit approval of optional
non-permissive components.

Full SDK/CPack and supported-platform acceptance remain separate release gates.
Local GPU execution or a passing resource audit does not close those gates.
Repository visibility is unchanged; this document does not announce a public
release or official catalog acceptance.

---

## Reference Pattern

[SlicerTotalSegmentator](https://github.com/lassoan/SlicerTotalSegmentator) is an
officially distributed Slicer interface that installs AI packages and downloads
weights. It illustrates the appropriate direction: keep proven algorithms,
minimize mandatory dependencies, and make their setup and execution understandable
inside Slicer. It is not evidence that our remaining installation tests passed.
