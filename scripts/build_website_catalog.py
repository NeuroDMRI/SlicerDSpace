#!/usr/bin/env python3
"""Render the method inventory with direct links to primary publications."""

import argparse
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = ROOT / "docs/website/references.json"
PAGE = ROOT / "docs/site/index.html"

# Keep implementation scope explicit: an API-only measure is not a GUI default.
# Each entry identifies a real entry point and its primary method/software sources.
METHODS = (
    ("input", "Input", "DICOM / NIfTI / NRRD", "Conversion, gradient validation and spatial checks.", "dcm2niix or Slicer DWIConvert; supplied masks accepted", [36, 27, 28], "scripts/prepare_inputs.py"),
    ("pnl_cnn", "Brain mask", "PNL CNN (default)", "Author diffusion-CNN brain masking and registration.", "Managed PNL runtime, weights and ANTs", [37, 38], "shared/brain_mask.py"),
    ("mppca", "Preprocessing", "MP-PCA", "Random-matrix denoising and noise estimation.", "DIPY", [39, 52, 53, 29, 30], "preprocessing/pipeline.py"),
    ("gibbs", "Preprocessing", "Gibbs removal", "Local subvoxel-shift ringing correction.", "DIPY", [42, 29, 30], "preprocessing/pipeline.py"),
    ("eddy", "Preprocessing", "FSL eddy", "Eddy-current and motion correction.", "MRtrix3 dwifslpreproc; FSL", [43, 29], "preprocessing/pipeline.py"),
    ("topup", "Preprocessing", "FSL topup", "Susceptibility-distortion estimation and correction.", "FSL; reverse-phase data and acquisition metadata", [44, 29], "preprocessing/pipeline.py"),
    ("dti", "Microstructure", "DTI (default)", "LS/WLS tensor fitting; FA, MD, AD, RD, eigenvalues and trace.", "SlicerDMRI", [45, 27, 28], "shared/pipeline.py"),
    ("freewater", "Microstructure", "Free water", "Free-water fraction and tissue-corrected FA/MD; WLS or NLS fitting.", "DIPY; suitable multi-shell sampling", [4, 5, 6, 30], "microstructure/pipeline.py"),
    ("noddi", "Microstructure", "NODDI / AMICO", "NDI, ODI, IsoVF and fit residuals.", "AMICO; suitable multi-shell sampling", [7, 31], "microstructure/pipeline.py"),
    ("dki", "Microstructure", "DKI", "WLS/OLS fitting; MK, AK, RK, KFA, MKT, RTK and validity masks.", "DIPY; shell, direction and design-rank checks", [8, 30], "microstructure/dki.py"),
    ("mkcurve", "Microstructure", "MK-Curve", "Correction curves, corrected kurtosis maps and metric-specific validity.", "Bundled Python backend; DKI eligibility checks", [9], "microstructure/mkcurve.py"),
    ("ddseg", "Anatomical prior", "DDSeg", "Tissue labels and probabilities from converted PyTorch models.", "PyTorch; DDSeg weights", [2], "anatomy/segmentation/pipeline.py"),
    ("ddparcel", "Anatomical prior", "DDParcel", "Anatomical region inference with author-derived models.", "PyTorch; weights and registration template", [3], "anatomy/parcellation/pipeline.py"),
    ("ddsurfer", "Anatomical prior", "DDSurfer", "Native white and pial mesh reconstruction.", "PyTorch; DDSurfer weights", [1], "anatomy/surface/pipeline.py"),
    ("brainsfit", "Anatomical prior", "BRAINSFit registration", "Template registration for eligible anatomical routes.", "Slicer BRAINSFit / BRAINSResample", [51, 27], "shared/pipeline.py"),
    ("surface_postprocess", "Anatomical prior", "Surface post-processing", "Thickness, curvature, spherical registration and DK/Destrieux atlas mapping.", "Licensed native component runtime; full suite optional", [33, 47, 48, 49], "anatomy/surface/pipeline.py"),
    ("response", "Tractography", "Response estimation", "Dhollander three-tissue response estimation.", "MRtrix3", [12, 13, 29], "tractography/pipeline.py"),
    ("csd", "Tractography", "CSD / MSMT-CSD", "Single- or multi-tissue fibre-orientation distributions.", "MRtrix3; selected acquisition route", [11, 46, 10, 29], "tractography/pipeline.py"),
    ("mtnormalise", "Tractography", "Multi-tissue normalization", "Log-domain intensity and inhomogeneity normalization.", "MRtrix3 mtnormalise", [17, 18, 29], "tractography/pipeline.py"),
    ("csd_act", "Tractography", "iFOD2 / ACT (default)", "Probabilistic tracking with DDSeg-derived tissue constraints.", "MRtrix3; reviewed anatomical priors", [14, 15, 29], "tractography/pipeline.py"),
    ("dti_tracking", "Tractography", "DTI deterministic", "Tensor-based direction tracking with stopping criteria.", "SlicerDMRI; DIPY", [45, 28, 30], "tractography/dti.py"),
    ("ukf", "Tractography", "UKF two-tensor", "Automatic invocation of the official tracking CLI.", "UKFTractography extension; SlicerDMRI", [19, 28], "tractography/ukf.py"),
    ("saf", "Tractography", "SAF", "Surface-constrained short-association tracking and geometric filtering.", "MRtrix3; post-processed native surfaces", [20], "tractography/saf.py"),
    ("wmql", "Fiber bundles", "WMQL (default)", "Official bundle queries; optional subsets and regional cerebellar selections.", "Author tract_querier; DDParcel labels", [21], "tractography/wmql.py"),
    ("wma", "Fiber bundles", "WMA", "ORG, East-West or NABA atlas registration and clustering; automatic UKF handoff.", "WMA; selected atlas; UKF reference route", [22, 23, 24, 26, 54, 55], "tractography/wma.py"),
    ("tractcloud", "Fiber bundles", "TractCloud", "Author learned tract parcellation and bundle export.", "PyTorch backend; model and centroids", [25, 22], "tractography/tractcloud.py"),
    ("count", "Connectome", "Streamline count", "Endpoint assignment to DDParcel nodes.", "Python or eligible MRtrix3 backend", [50, 29], "connectome/pipeline.py"),
    ("sift2", "Connectome", "SIFT2-weighted count", "FOD-informed weighting and matrix construction with matched tracks.", "MRtrix3; corresponding tractogram and weights", [16, 29, 50], "connectome/pipeline.py"),
    ("scalar_length", "Connectome", "Scalar / length summaries", "Mean FA, MD, ODI or another scalar; mean and total streamline length.", "Python API; matched maps, nodes and tracks", [50, 29, 30], "connectome/pipeline.py"),
    ("regional", "Regional analysis", "Selected-region analysis", "Reuse maps for ROI statistics, filter tracks and extract node submatrices.", "Advanced settings; workflow utility using completed outputs", [3, 30, 50], "shared/region_analysis.py"),
)


def render_methods(references):
    by_id = {row["id"]: row for row in references}
    lines, previous = [], None
    for identifier, stage, method, summary, runtime, citations, source in METHODS:
        if not (ROOT / source).is_file():
            raise ValueError("Method entry point is missing: " + source)
        if not citations or not set(citations) <= by_id.keys():
            raise ValueError("Method has incomplete references: " + identifier)
        links = " ".join(
            f'<a class="method-reference" data-reference="{n}" href="{escape(by_id[n]["url"], quote=True)}" '
            f'title="{escape(by_id[n]["citation"], quote=True)}" aria-label="Reference {n}" '
            f'target="_blank" rel="noopener">[{n}]</a>'
            for n in citations)
        group = ' class="stage-start"' if stage != previous else ""
        lines.append(f'            <tr data-method="{identifier}"{group}><th scope="row">{escape(stage)}</th><td>{escape(method)}</td><td>{escape(summary)}</td><td class="st-cell">{escape(runtime)}</td><td class="ref-cell">{links}</td></tr>')
        previous = stage
    return "\n".join(lines)


def render(page, references):
    if len({row['id'] for row in references}) != len(references) or any(not row['url'].startswith('https://') for row in references):
        raise ValueError('References need unique IDs and primary HTTPS links')
    for key, value in (("methods-catalog", render_methods(references)),):
        start, end = f"<!-- {key}:start -->", f"<!-- {key}:end -->"
        if page.count(start) != 1 or page.count(end) != 1:
            raise ValueError("Expected one generated catalog region: " + key)
        page = page.split(start)[0] + start + "\n" + value + "\n" + end + page.split(end)[1]
    return page


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    references = json.loads(REFERENCES.read_text(encoding="utf-8"))["references"]
    original = PAGE.read_text(encoding="utf-8")
    expected = render(original, references)
    if args.check:
        if original != expected:
            raise SystemExit("Homepage catalogs are stale; run scripts/build_website_catalog.py")
    else:
        PAGE.write_text(expected, encoding="utf-8")
    print(f"PASS {len(METHODS)} method entries with direct primary-source links")


if __name__ == "__main__":
    main()
