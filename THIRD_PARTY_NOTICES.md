<div align="center">

# Third-Party Notices

[Project](README.md) | [Documentation](docs/README.md) | [Quick Start](docs/setup/QUICKSTART.md) | [Release](docs/release/EXTENSION_READINESS.md) | [中文](docs/assets/CODE_LICENSES.zh-CN.md)

</div>

SlicerDSpace's own source code is licensed under the BSD 3-Clause license
in [LICENSE.txt](LICENSE.txt). This file documents bundled third-party code, data,
and reference material that carries different terms, and states honestly
where asset permissions still need separate review. License texts and their
pinned sources are recorded in [licenses/provenance.json](licenses/provenance.json).
The integrity check is not a legal opinion or a grant of model/template rights.

---

## WMQL Author Query Resources

`tractography/wmql_queries/official/` distributes unmodified `FreeSurfer.qry`
and `freesurfer_queries.qry` from [demianw/tract_querier](https://github.com/demianw/tract_querier)
at commit `5c69c594d489859e08485110678dca7f6f4cba40`.
The upstream [license text](tractography/wmql_queries/official/LICENSE.txt) ships
beside these files. The Tract Querier execution engine remains an optional
dependency. Historical project-specific query wrappers are distinct from the
official definitions and are not current GUI defaults.

---

## DIPY Free-water WLS Adaptation

`microstructure/freewater_wls.py` is adapted from `dipy/reconst/fwdti.py::wls_iter`
at [DIPY 1.11.0](https://github.com/dipy/dipy/blob/1b0b3e2a0b0c8611287f4677f21356b704495665/dipy/reconst/fwdti.py).
It replaces the materialized diagonal-weight product with C-order column
scaling while preserving the remaining fit operations. Only verified source
versions use this path; unsupported inputs or releases retain author behavior.
The instance-private dispatcher also invokes source-verified tensor helper
bodies directly, bypassing only redundant keyword-compatibility adapters.
It does not modify installed DIPY modules or replace tensor eigendecomposition.

Copyright (c) 2008-2025, dipy developers. All rights reserved.
The complete [DIPY BSD-3-Clause notice](licenses/DIPY.txt) ships in the runtime
package. This is a modified integration, not an official DIPY release or an
accuracy/clinical claim. Installed DIPY retains its own copyright and license.

---

## WMA Registration Preprocessing

`tractography/wma_preprocessing.py` is a modified bulk implementation of the
attribute-free length filter in SlicerDMRI/whitematteranalysis, pinned at
`4532b911386d2d1bc7ebba63fdd13e65b1660623`. The author length-estimation,
threshold and float32 copy semantics are retained; this is not an upstream release.
`tractography/wma_transform.py` adapts the same pinned author's registration
point transformation into `vtkTransform.TransformPoints`, retaining double
precision and point order. Source/VTK guards retain the author path otherwise;
no registration objective or optimizer is replaced.
`tractography/wma_fibers.py` adapts the same author's nearest-neighbor
`FiberArray` conversion to bounded bulk copying, preserving ties-to-even point
selection, hemisphere calculation and traversal state. The owned registration
process alone installs this hook; unsupported sources/VTK retain original code.

All or portions of this licensed product (such portions are the "Software")
have been obtained under license from The Brigham and Women's Hospital, Inc.
and are subject to the terms and conditions in [the full WMA/Slicer agreement](licenses/WMA.txt).
Optional upstream WMA code and atlas assets retain their own distribution terms.

## Optional WMA Atlas Families

The ORG, East-West WM and NABA atlases are separately selected data resources
in `SlicerDSpace-models`. East-West is published by Li and colleagues at
[ScienceDB](https://doi.org/10.57760/sciencedb.17354); NABA is published by
Zhang and colleagues at [Zenodo](https://doi.org/10.5281/zenodo.18227691).
Both records specify CC BY 4.0. The mirror retains original data bytes in a
runtime subset, attribution, upstream archive checksums and extracted-file
SHA256 inventories. The corresponding papers are
[Li et al. (2024)](https://doi.org/10.1038/s41597-024-03624-2) and
[Zhang et al. (2026)](https://doi.org/10.1016/j.fmre.2025.12.030).
WMA's original software licence continues to apply to its execution components.
No NABA algorithm source is redistributed with this extension; the adapter
reads published tract definitions and calls the installed WMA implementation.

---

## FastSurfer-derived code (Apache License 2.0)

- `anatomy/parcellation/models/networks.py`
- `anatomy/parcellation/models/sub_module.py`
- `anatomy/parcellation/data_loader/augmentation.py`
- `anatomy/parcellation/data_loader/conform.py`
- `anatomy/parcellation/data_loader/load_neuroimaging_data.py`

These files carry an in-file copyright header ("Copyright 2019 Image
Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn")
and are derived from [FastSurfer](https://github.com/Deep-MI/FastSurfer),
licensed under the Apache License, Version 2.0. The full license text is in
[`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt); the notice text
reproduced from these files is in
[`licenses/NOTICE-FastSurfer.txt`](licenses/NOTICE-FastSurfer.txt).

---

## SAF (short association fibre tractography) backend (MIT License)

- `tractography/saf_backend/` (all files; bundled and packaged as runtime
  code -- see `docs/architecture/REPOSITORY_LAYOUT.md`)

Translated/bundled from [SAF](https://github.com/dmitrishastin/SAF)
(Dmitri Shastin), licensed under the MIT License. The full license text is
in [`tractography/saf_backend/LICENSE.md`](tractography/saf_backend/LICENSE.md),
which CMake packaging now includes alongside the module's Python sources
(see `CMakeLists.txt`). Method paper: D. Shastin, S. Genc, G.D. Parker,
K. Koller, C.M.W. Tax, J. Evans, K. Hamandi, W.P. Gray, D.K. Jones,
M. Chamberland, "Surface-based tracking for short association fibre
tractography", NeuroImage 2022,
[doi:10.1016/j.neuroimage.2022.119423](https://doi.org/10.1016/j.neuroimage.2022.119423).

The reference is pinned to `7e2a9d152caf1cf3e6bface64e7e176faac0c14f`.
This is an **unofficial MATLAB-to-Python port**, limited to native-diffusion
surfaces. Local endpoint indexing, complete-intersection and bookkeeping
corrections are documented in [implementation provenance](docs/architecture/IMPLEMENTATION_PROVENANCE.md)
and the [SAF verification record](docs/release/SAF_PYTHON_20261013.md);
author endorsement and whole-pipeline bitwise equivalence are not implied.

---

## `anatomy/parcellation/conversion/` (NRRD/NHDR conversion utilities)

- `anatomy/parcellation/conversion/` (all files under this directory)

This code is attributed in-repo to the
[pnlbwh/conversion](https://github.com/pnlbwh/conversion) project (Tashrif
Billah): `anatomy/parcellation/conversion/nhdr_write.py` writes
`"See https://github.com/pnlbwh/conversion for more info"` directly into its
output file header, and `anatomy/parcellation/conversion/_version.py` records
`__version__ = '2.4'`, consistent with that project's own versioning.

**Recorded terms:** the BWH Conversion Contribution and Software License
Agreement, Version 1.0, verified against the
[pinned upstream license](https://github.com/pnlbwh/conversion/blob/cdd433adff3eebc486f2b5648714cbfe93b9422e/LICENSE).
The complete agreement is bundled in
[conversion/LICENSE.txt](anatomy/parcellation/conversion/LICENSE.txt).
It is a BSD-style agreement, not the project's BSD-3-Clause license.

All or portions of this licensed product (such portions are the "Software")
have been obtained under license from The Brigham and Women's Hospital, Inc.
and are subject to the following terms and conditions:
[full Conversion agreement](anatomy/parcellation/conversion/LICENSE.txt).

This is a **modified integration**, not an unmodified upstream distribution:
imports, lazy exports, input/gradient validation and file handling have been
adapted for SlicerDSpace. DDSurfer's derived helpers in
`anatomy/surface/ddsurfer_backend/preprocessing/conversion/` retain these terms
as well. Preserve the agreement and notices when distributing source/binaries
and documentation; fixture/data permissions are assessed separately.

---

## `anatomy/parcellation/conversion/data/FreeSurferColorLUT.txt`

This is FreeSurfer's own color lookup table (RCS header:
`$Id: FreeSurferColorLUT.txt,v 1.105 2015/07/29 18:23:03 greve Exp $`),
copied verbatim from a FreeSurfer distribution. It is packaged as a Slicer
extension resource by `CMakeLists.txt`.

**Recorded terms:** FreeSurfer Software License Agreement, Version 1.0
(February 2011), from the
[official pinned source](https://github.com/freesurfer/freesurfer/blob/4f8728e5132e8d39a8f1bf3d24d15fed62f4fc9d/LICENSE.txt).
The full agreement is included as [licenses/FreeSurfer.txt](licenses/FreeSurfer.txt).
Part B permits distribution subject to its notice, attribution and other
conditions; this file is not relicensed under BSD-3-Clause. Its original LUT
header is retained, with no changes to label IDs or colors.

All or portions of this licensed product (such portions are the "Software")
have been obtained under license from The General Hospital Corporation "MGH"
and are subject to the following terms and conditions:
[full FreeSurfer agreement](licenses/FreeSurfer.txt).

This statement covers the LUT, not anatomical templates, subject data,
FreeSurfer executable installation or any third-party assets.

---

## DDSurfer source and PyTorch3D-derived helpers

The source snapshot is under `anatomy/surface/ddsurfer_backend/`; code revisions
and per-file hashes remain in its `UPSTREAM.json`.
The current snapshot includes the original-algorithm source-build recipe under
`postprocessing/core/` and data metadata/notices under `postprocessing/assets/`.
The source build reuses FreeSurfer C++ at its pinned revision, with ITK/zlib;
it is not a Python rewrite or removal of their licenses. No compiled FreeSurfer
executables or personal activation files are shipped in this extension.
The optional atlas archive is fetched separately; retain its complete
[upstream notices](anatomy/surface/ddsurfer_backend/postprocessing/assets/LICENSE.txt).
The upstream project added an explicit MIT license in
[commit 0c399ef](https://github.com/NeuroDMRI/DDSurfer/blob/0c399ef8ff8c9931730cab6beaf5796ee61243ca/LICENSE).
An exact copy is bundled as
[ddsurfer_backend/LICENSE.txt](anatomy/surface/ddsurfer_backend/LICENSE.txt).
The previous "upstream license missing" description is superseded for the
project-authored source, not for weights or templates.

The attributed PyTorch3D-derived normal/Laplacian helpers in `utils/mesh.py`
and optional training-only boundary-loss helper in `utils/loss.py`
retain Meta's BSD-3-Clause terms, copied from the
[upstream notice](https://github.com/NeuroDMRI/DDSurfer/blob/c3b2d40ead82b316f0adec900a23979f629da70b/utils/LICENSE.pytorch3d)
into [licenses/PyTorch3D.txt](licenses/PyTorch3D.txt).
Derived conversion helpers retain the BWH terms above. These exceptions must
accompany the source and binary distribution; MIT does not erase them.

---

## Verification and Limits

~~~bash
python3 scripts/audit_third_party.py
~~~

CI checks complete notice hashes, pinned source URLs, attributed paths and
inclusion in the runtime resource manifest. It does not claim that every
third-party right, patent or asset redistribution condition has been cleared.
Weights, initialization meshes, registration atlases and subject-derived test
fixtures remain subject to [asset review](docs/assets/MODEL_ASSET_TERMS.md).

---

## Method papers

`docs/assets/MODEL_ASSET_TERMS.md` states that the DDSeg and DDParcel model
weights each "require ... citation of the peer-reviewed [DDSeg/DDParcel]
method paper." The DDSurfer, DDSeg, and DDParcel method papers are recorded
as `references:` entries in [`CITATION.cff`](CITATION.cff), each verified
against its DOI's own Crossref record
(`https://api.crossref.org/works/<doi>`):

- DDSurfer: *Advanced Science*, e76596 (2026), doi:10.1002/advs.76596
- DDSeg: *NeuroImage* 233:117934 (2021), doi:10.1016/j.neuroimage.2021.117934
- DDParcel: *IEEE Transactions on Medical Imaging* 43(3):1191-1202 (2024),
  doi:10.1109/TMI.2023.3331691

## MK-Curve Python Port

`microstructure/mkcurve.py` translates the MK-Curve algorithm by Fan Zhang
and collaborators at Brigham and Women's Hospital. Source:
https://github.com/zhangfanmark/MK-Curve, commit
`9a213e1a5850fe50c998793f31ed852df00242f8`.
The complete upstream license is preserved in `licenses/MK-Curve-LICENSE.txt`.

`microstructure/mkcurve_tensor.py` and `mkcurve_directions.json` derive from
the upstream DKI routines, Copyright (c) 2017 New York University and
University of Antwerp, under MPL-2.0. The full license is preserved in
`licenses/MK-Curve-DKI-MPL-2.0.txt`. These modified files remain MPL-2.0
source, shipped with the extension; the rest of the extension is not
relicensed. The translation is maintained by SlicerDSpace and is not
an official author-endorsed Python implementation. MATLAB I/O utilities
and the proprietary MathWorks helper are not redistributed.
