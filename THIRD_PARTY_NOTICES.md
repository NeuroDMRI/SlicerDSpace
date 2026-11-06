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

## Native Tensor Fit (SlicerDMRI-derived)

Native tensor fit (`shared/teem_dti.py`) is a modified Python reimplementation of SlicerDMRI
`DWIToDTIEstimation`, `DiffusionTensorScalarMeasurements` and `vtkTeemEstimateDiffusionTensor`
(SlicerDMRI commit a40200b: the revision of the Extension Manager build for Slicer 5.10.0 r34045,
per its installed `.s4ext`; the sources were read from GitHub at that commit), under the
3D Slicer Contribution and Software License Agreement ([licenses/SlicerDMRI.txt](licenses/SlicerDMRI.txt)).
It calls the Teem library that 3D Slicer ships (Teem 1.12, LGPL-2.1 with a static-linking exception);
the plugin does not redistribute Teem.

---

## PNL CNN Brain Mask (`shared/pnl_cnn/`)

- `shared/pnl_cnn/` (all files under this directory)

This is a **modified integration**, not an unmodified upstream distribution. Source: [PNLBrainMask-Torch](https://github.com/ChengjinLii/PNLBrainMask-Torch) at commit
`8d201a50447ebea8b10a4f252bfd262931603970`, itself a modified PyTorch implementation of
[pnlbwh/CNN-Diffusion-MRIBrain-Segmentation](https://github.com/pnlbwh/CNN-Diffusion-MRIBrain-Segmentation).
Vendored files: `__init__.py`, `inference.py` and `data/CompNetBasicModel.json` (unchanged);
`model.py`, `weights.py` and `pipeline.py` (modified). Local patches: `model.architecture()` reads
`data/CompNetBasicModel.json` by a path next to the file instead of `importlib.resources`;
`weights.py` keeps only the checked weight loader; `pipeline.py` keeps only the 3-D input branch of
`mean_b0`. No numeric change. Each modified file starts with a modification comment; the file
hashes and the patch list are in `shared/pnl_cnn/UPSTREAM.json`. The licence file is
[shared/pnl_cnn/LICENSE.txt](shared/pnl_cnn/LICENSE.txt); the same text is in
[docs/licenses/pnl_brain_mask.md](docs/licenses/pnl_brain_mask.md). Model weights and the IIT
template are separate assets with their own terms.

All or portions of this licensed product (such portions are the "Software")
have been obtained under license from The Brigham and Women's Hospital, Inc.
and are subject to the following terms and conditions:

~~~text
PNLBrainMask-Torch is a modified implementation, not the original PNL software.
Modifications Copyright (c) 2026 Chengjin Li.

All or portions of this licensed product (such portions are the "Software") have
been obtained under license from The Brigham and Women's Hospital, Inc. and are
subject to the following terms and conditions:

The CNN-Diffusion-MRIBrain-Segmentation license below is a BSD style license, with extensions
to cover contributions and other issues specific to CNN-Diffusion-MRIBrain-Segmentation. It is
inpsired by [3D Slicer License](https://github.com/Slicer/Slicer/blob/master/License.txt).


CNN-Diffusion-MRIBrain-Segmentation Contribution and Software License Agreement ("Agreement")
Version 1.0 (December 20, 2005)

This Agreement covers contributions to and downloads from the
CNN-Diffusion-MRIBrain-Segmentation project ("CNN-Diffusion-MRIBrain-Segmentation") maintained by The Brigham and Women's
Hospital, Inc. ("Brigham"). Part A of this Agreement applies to
contributions of software and/or data to CNN-Diffusion-MRIBrain-Segmentation (including making
revisions of or additions to code and/or data already in CNN-Diffusion-MRIBrain-Segmentation). Part
B of this Agreement applies to downloads of software and/or data from
CNN-Diffusion-MRIBrain-Segmentation. Part C of this Agreement applies to all transactions with
CNN-Diffusion-MRIBrain-Segmentation. If you distribute Software (as defined below) downloaded from
CNN-Diffusion-MRIBrain-Segmentation, all of the paragraphs of Part B of this Agreement must be
included with and apply to such Software.

Your contribution of software and/or data to CNN-Diffusion-MRIBrain-Segmentation (including prior
to the date of the first publication of this Agreement, each a
"Contribution") and/or downloading, copying, modifying, displaying,
distributing or use of any software and/or data from CNN-Diffusion-MRIBrain-Segmentation
(collectively, the "Software") constitutes acceptance of all of the
terms and conditions of this Agreement. If you do not agree to such
terms and conditions, you have no right to contribute your
Contribution, or to download, copy, modify, display, distribute or use
the Software.

PART A. CONTRIBUTION AGREEMENT - License to Brigham with Right to
Sublicense ("Contribution Agreement").

1. As used in this Contribution Agreement, "you" means the individual
   contributing the Contribution to CNN-Diffusion-MRIBrain-Segmentation and the institution or
   entity which employs or is otherwise affiliated with such
   individual in connection with such Contribution.

2. This Contribution Agreement applies to all Contributions made to
   CNN-Diffusion-MRIBrain-Segmentation, including without limitation Contributions made prior to
   the date of first publication of this Agreement. If at any time you
   make a Contribution to CNN-Diffusion-MRIBrain-Segmentation, you represent that (i) you are
   legally authorized and entitled to make such Contribution and to
   grant all licenses granted in this Contribution Agreement with
   respect to such Contribution; (ii) if your Contribution includes
   any patient data, all such data is de-identified in accordance with
   U.S. confidentiality and security laws and requirements, including
   but not limited to the Health Insurance Portability and
   Accountability Act (HIPAA) and its regulations, and your disclosure
   of such data for the purposes contemplated by this Agreement is
   properly authorized and in compliance with all applicable laws and
   regulations; and (iii) you have preserved in the Contribution all
   applicable attributions, copyright notices and licenses for any
   third party software or data included in the Contribution.

3. Except for the licenses granted in this Agreement, you reserve all
   right, title and interest in your Contribution.

4. You hereby grant to Brigham, with the right to sublicense, a
   perpetual, worldwide, non-exclusive, no charge, royalty-free,
   irrevocable license to use, reproduce, make derivative works of,
   display and distribute the Contribution. If your Contribution is
   protected by patent, you hereby grant to Brigham, with the right to
   sublicense, a perpetual, worldwide, non-exclusive, no-charge,
   royalty-free, irrevocable license under your interest in patent
   rights covering the Contribution, to make, have made, use, sell and
   otherwise transfer your Contribution, alone or in combination with
   any other code.

5. You acknowledge and agree that Brigham may incorporate your
   Contribution into CNN-Diffusion-MRIBrain-Segmentation and may make CNN-Diffusion-MRIBrain-Segmentation available to members
   of the public on an open source basis under terms substantially in
   accordance with the Software License set forth in Part B of this
   Agreement. You further acknowledge and agree that Brigham shall
   have no liability arising in connection with claims resulting from
   your breach of any of the terms of this Agreement.

6. YOU WARRANT THAT TO THE BEST OF YOUR KNOWLEDGE YOUR CONTRIBUTION
   DOES NOT CONTAIN ANY CODE THAT REQURES OR PRESCRIBES AN "OPEN
   SOURCE LICENSE" FOR DERIVATIVE WORKS (by way of non-limiting
   example, the GNU General Public License or other so-called
   "reciprocal" license that requires any derived work to be licensed
   under the GNU General Public License or other "open source
   license").

PART B. DOWNLOADING AGREEMENT - License from Brigham with Right to
Sublicense ("Software License").

1. As used in this Software License, "you" means the individual
   downloading and/or using, reproducing, modifying, displaying and/or
   distributing the Software and the institution or entity which
   employs or is otherwise affiliated with such individual in
   connection therewith. The Brigham and Women's Hospital,
   Inc. ("Brigham") hereby grants you, with right to sublicense, with
   respect to Brigham's rights in the software, and data, if any,
   which is the subject of this Software License (collectively, the
   "Software"), a royalty-free, non-exclusive license to use,
   reproduce, make derivative works of, display and distribute the
   Software, provided that:

(a) you accept and adhere to all of the terms and conditions of this
Software License;

(b) in connection with any copy of or sublicense of all or any portion
of the Software, all of the terms and conditions in this Software
License shall appear in and shall apply to such copy and such
sublicense, including without limitation all source and executable
forms and on any user documentation, prefaced with the following
words: "All or portions of this licensed product (such portions are
the "Software") have been obtained under license from The Brigham and
Women's Hospital, Inc. and are subject to the following terms and
conditions:"

(c) you preserve and maintain all applicable attributions, copyright
notices and licenses included in or applicable to the Software;

(d) modified versions of the Software must be clearly identified and
marked as such, and must not be misrepresented as being the original
Software; and

(e) you consider making, but are under no obligation to make, the
source code of any of your modifications to the Software freely
available to others on an open source basis.

2. The license granted in this Software License includes without
   limitation the right to (i) incorporate the Software into
   proprietary programs (subject to any restrictions applicable to
   such programs), (ii) add your own copyright statement to your
   modifications of the Software, and (iii) provide additional or
   different license terms and conditions in your sublicenses of
   modifications of the Software; provided that in each case your use,
   reproduction or distribution of such modifications otherwise
   complies with the conditions stated in this Software License.

3. This Software License does not grant any rights with respect to
   third party software, except those rights that Brigham has been
   authorized by a third party to grant to you, and accordingly you
   are solely responsible for (i) obtaining any permissions from third
   parties that you need to use, reproduce, make derivative works of,
   display and distribute the Software, and (ii) informing your
   sublicensees, including without limitation your end-users, of their
   obligations to secure any such required permissions.

4. The Software has been designed for research purposes only and has
   not been reviewed or approved by the Food and Drug Administration
   or by any other agency. YOU ACKNOWLEDGE AND AGREE THAT CLINICAL
   APPLICATIONS ARE NEITHER RECOMMENDED NOR ADVISED. Any
   commercialization of the Software is at the sole risk of the party
   or parties engaged in such commercialization. You further agree to
   use, reproduce, make derivative works of, display and distribute
   the Software in compliance with all applicable governmental laws,
   regulations and orders, including without limitation those relating
   to export and import control.

5. The Software is provided "AS IS" and neither Brigham nor any
   contributor to the software (each a "Contributor") shall have any
   obligation to provide maintenance, support, updates, enhancements
   or modifications thereto. BRIGHAM AND ALL CONTRIBUTORS SPECIFICALLY
   DISCLAIM ALL EXPRESS AND IMPLIED WARRANTIES OF ANY KIND INCLUDING,
   BUT NOT LIMITED TO, ANY WARRANTIES OF MERCHANTABILITY, FITNESS FOR
   A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
   BRIGHAM OR ANY CONTRIBUTOR BE LIABLE TO ANY PARTY FOR DIRECT,
   INDIRECT, SPECIAL, INCIDENTAL, EXEMPLARY OR CONSEQUENTIAL DAMAGES
   HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY ARISING IN ANY WAY
   RELATED TO THE SOFTWARE, EVEN IF BRIGHAM OR ANY CONTRIBUTOR HAS
   BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. TO THE MAXIMUM
   EXTENT NOT PROHIBITED BY LAW OR REGULATION, YOU FURTHER ASSUME ALL
   LIABILITY FOR YOUR USE, REPRODUCTION, MAKING OF DERIVATIVE WORKS,
   DISPLAY, LICENSE OR DISTRIBUTION OF THE SOFTWARE AND AGREE TO
   INDEMNIFY AND HOLD HARMLESS BRIGHAM AND ALL CONTRIBUTORS FROM AND
   AGAINST ANY AND ALL CLAIMS, SUITS, ACTIONS, DEMANDS AND JUDGMENTS
   ARISING THEREFROM.

6. None of the names, logos or trademarks of Brigham or any of
   Brigham's affiliates or any of the Contributors, or any funding
   agency, may be used to endorse or promote products produced in
   whole or in part by operation of the Software or derived from or
   based on the Software without specific prior written permission
   from the applicable party.

7. Any use, reproduction or distribution of the Software which is not
   in accordance with this Software License shall automatically revoke
   all rights granted to you under this Software License and render
   Paragraphs 1 and 2 of this Software License null and void.

8. This Software License does not grant any rights in or to any
   intellectual property owned by Brigham or any Contributor except
   those rights expressly granted hereunder.

PART C. MISCELLANEOUS

This Agreement shall be governed by and construed in accordance with
the laws of The Commonwealth of Massachusetts without regard to
principles of conflicts of law. This Agreement shall supercede and
replace any license terms that you may have agreed to previously with
respect to CNN-Diffusion-MRIBrain-Segmentation.
~~~

---

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
