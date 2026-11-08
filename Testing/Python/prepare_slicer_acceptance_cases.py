"""Prepare isolated acceptance subjects from existing real HCP/DICOM inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--hcp", type=Path, required=True)
    parser.add_argument("--dicom-fixture", type=Path, required=True)
    parser.add_argument("--software", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--reuse-inputs", action="store_true", help="Reuse and hash already staged real inputs.")
    parser.add_argument("--cases", nargs="+", help="Generate only the named scientific cases.")
    parser.add_argument("--config-name", default="cases.json")
    parser.add_argument("--check-run-retention", action="store_true", help="Hash prior CSD outputs across a new DTI run.")
    args = parser.parse_args()
    root = args.workspace.resolve()
    provenance = []
    for subject in ("100307", "149741", "sub-dicom"):
        source = (args.dicom_fixture / "subject/input" if subject == "sub-dicom" else
                  args.hcp / subject / "T1w/Diffusion")
        dest = root / "subjects" / subject / "input"
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
            original = source / name
            target = dest / name
            if not original.is_file():
                raise FileNotFoundError(original)
            if target.exists():
                if not args.reuse_inputs:
                    raise FileExistsError(f"Use a fresh workspace: {target}")
            else:
                shutil.copy2(original, target)
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            if args.reuse_inputs:
                if hashlib.sha256(original.read_bytes()).hexdigest() != digest:
                    raise ValueError(f"Staged input differs from original: {target}")
            provenance.append({"subject": subject, "source": str(original), "copy": str(target), "sha256": digest})
    software = args.software
    slicer = software / "Slicer-5.10.0-linux-amd64"
    external = dict(model_assets_dir=str(args.models), mrtrix_bin_dir=str(software / "mrtrix3_env/bin"),
                    preprocessing_fsl_bin_dir=str(software / "fsl_env/bin"),
                    postprocess_freesurfer_home=str(software / "freesurfer"),
                    tract_querier_root=str(args.repo / "third_party/tract_querier"),
                    tractcloud_root=str(args.repo / "third_party/TractCloud"),
                    tractcloud_python_executable=str(slicer / "bin/PythonSlicer"),
                    tractcloud_data_dir="/data/chengjin/SlicerDDSurfer-test/resources/tractcloud_data",
                    wma_root=str(args.repo / "third_party/whitematteranalysis"),
                    wma_atlas_dir="/data/chengjin/SlicerDDSurfer-test/resources/ORG-Atlases",
                    slicer_executable=str(slicer / "Slicer"))
    cases = []

    def add(name, subject, targets, **kwargs):
        subject_root = root / "subjects" / subject
        cases.append(dict(name=name, subject_id=subject, subject_root=str(subject_root),
                          output_root=str(subject_root / "outputs"), run_name="acceptance_science", targets=targets, **kwargs))

    add("anatomy_dti_csd_wmql_connectome", "100307",
        ["segmentation", "parcellation", "surface", "surface_postprocess", "microstructure", "tractography", "bundles", "connectome"],
        expected_tasks=["segmentation.ddseg", "parcellation.ddparcel", "surface.ddsurfer", "surface.ddsurfer_postprocess", "microstructure.dti", "tractography.csd_act", "bundles.wmql"],
        extra_cli_args=["--mrtrix-nthreads", "4", "--mrtrix-seed", "42"])
    add("freewater_noddi", "100307", ["microstructure"], microstructure_stages="dti,freewater,noddi",
        expected_tasks=["microstructure.freewater", "microstructure.noddi"],
        extra_cli_args=["--microstructure-free-water-engine", "joblib", "--microstructure-free-water-n-jobs", "4",
                        "--microstructure-free-water-vox-per-chunk", "10000"])
    add("tractcloud", "100307", ["bundles"], bundle_backend="tractcloud", expected_tasks=["bundles.tractcloud"])
    add("saf", "100307", ["tractography"], tractography_backend="saf", expected_tasks=["tractography.saf"],
        extra_cli_args=["--saf-tckgen-options",
                        f"-seed_image {root / 'subjects/100307/input/nodif_brain_mask.nii.gz'} -cutoff 0.05 -select 50000 -maxlength 40 -nthreads 4"])
    add("connectome_mrtrix", "100307", ["connectome"], connectome_backend="mrtrix",
        extra_cli_args=["--no-resume", "--mrtrix-nthreads", "4"],
        expected_tasks=["connectome.streamline_count", "connectome.sift2_weighted"])
    add("dti_tractography_connectome", "100307", ["tractography", "connectome"], tractography_backend="dti_det",
        streamlines=5000, expected_tasks=["tractography.dti_det", "connectome.streamline_count"])
    cases[-1]["run_name"] = "acceptance_dti_det"
    if args.check_run_retention:
        previous = root / "subjects/100307/outputs/04_tractography/tractograms/acceptance_science"
        files = sorted(previous.glob("100307_csd_act_ifod2*"))
        if len(files) != 5 or not all(path.is_file() for path in files):
            raise FileNotFoundError(f"Expected the completed CSD run's five outputs in {previous}")
        cases[-1]["preserve_outputs"] = [str(path) for path in files]
        cases[-1]["extra_cli_args"] = ["--no-resume"]
        cases[-1]["expected_statuses"] = {"tractography.dti_det": "completed"}
    add("connectome_resume", "100307", ["connectome"], connectome_backend="mrtrix",
        expected_tasks=["connectome.streamline_count", "connectome.sift2_weighted"],
        expected_statuses={"connectome.streamline_count": "skipped", "connectome.sift2_weighted": "skipped"})
    add("fsl_preprocessing", "sub-dicom", ["preprocessing"], pe="AP",
        readout=0.039655, reverse_b0=str(args.dicom_fixture / "reverse_phase/reverse_b0.nii.gz"),
        extra_cli_args=["--no-resume", "--preprocessing-eddy-threads", "4", "--preprocessing-eddy-slm", "linear"])
    cases[-1]["output_root"] = str(root / "subjects/sub-dicom/outputs_fsl")
    add("wma_ukf", "149741", ["bundles"], bundle_backend="wma",
        wma_input="/data/chengjin/SlicerDDSurfer-test/149741_wma_act/tractography/149741_ukf_pp_downsampled.vtk",
        expected_tasks=["bundles.wma"])
    root.mkdir(parents=True, exist_ok=True)
    if args.cases:
        unknown = set(args.cases) - {case["name"] for case in cases}
        if unknown:
            raise ValueError(f"Unknown cases: {sorted(unknown)}")
        cases = [case for case in cases if case["name"] in args.cases]
    config_path = root / args.config_name
    config_path.with_suffix(".inputs.json").write_text(json.dumps(provenance, indent=2) + "\n")
    config_path.write_text(json.dumps(dict(external_dependencies=external, cases=cases,
                                                    case_timeout_seconds=10800), indent=2) + "\n")
    print(config_path)


if __name__ == "__main__":
    main()
