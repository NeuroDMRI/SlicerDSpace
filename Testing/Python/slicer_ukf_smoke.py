"""Native UKF GUI handoff; optionally execute the actual CLI on synthetic DWI."""

import json
import os
from pathlib import Path
import traceback

import qt
import slicer


def run():
    try:
        slicer.util.selectModule("SlicerDSpace")
        owner = slicer.modules.slicerdspace.widgetRepresentation().self()
        node, ui = owner._parameterNode, owner.ui
        assert node.orchestrationTractographyBackend == "csd_act"
        with slicer.util.NodeModify(node):
            node.targetTractography = True
            node.orchestrationTractographyBackend = "ukf"
        from shared.ui_ukf import add_staged_ukf_fields

        draft = qt.QWidget()
        read_settings = add_staged_ukf_fields(qt.QFormLayout(draft), node, False)
        controls = slicer.util.childWidgetVariables(draft)
        assert controls.ukf_numTensor.value == 2
        assert controls.ukf_seedsPerVoxel.value == 1
        assert controls.ukf_numThreads.value == 0
        controls.ukf_seedsPerVoxel.setValue(2)
        assert node.ukfParameters == "{}"
        assert json.loads(read_settings())["seedsPerVoxel"] == 2
        controls.ukf_seedsPerVoxel.setValue(1)
        assert read_settings() == "{}"
        from shared.ui_run_controller import _ukf_run_options, RunSelection, _append_ukf_args

        selection = RunSelection(
            subject_id="synthetic",
            subject_root_text="/tmp/s",
            output_root_text="",
            selected_targets=("tractography",),
            run_name="ukf",
            tractography_backend="ukf",
            bundle_backend="wma",
            microstructure_freewater=False,
            microstructure_noddi=False,
            select_streamlines=1000000,
            ddseg_device="auto",
            surface_device="auto",
            connectome_backend="python",
            connectome_compress_mrtrix_dense=False,
            ukf_parameters_text=node.ukfParameters,
        )
        handoff = _ukf_run_options(selection)
        assert handoff["ukf_parameters"]["numThreads"] == -1
        assert handoff["ukf_parameters"]["seedsPerVoxel"] == 1
        args = []
        _append_ukf_args(args, {"tractography_backend": "ukf", **handoff})
        assert "--ukf-parameters" in args
        if handoff['dwiconvert_executable'] is not None:
            assert args[args.index('--dwiconvert-executable') + 1] == str(handoff['dwiconvert_executable'])
        destination = Path(os.environ["SLICERDSPACE_SMOKE_ROOT"])
        owner._uiWidget.grab().save(str(destination / "ukf_advanced.png"))
        if handoff["ukf_executable"] is not None:
            actual_cli_smoke(destination, handoff["ukf_executable"], handoff['dwiconvert_executable'])
        else:
            print("UKF_REAL_CLI_SKIPPED: optional extension not installed", flush=True)
        print("UKF_GUI_PASS", flush=True)
        slicer.app.exit(0)
    except Exception:
        traceback.print_exc()
        slicer.app.exit(1)


def actual_cli_smoke(directory, binary, converter=None):
    import nibabel as nib
    import numpy as np
    from shared.pipeline import RawInputBundle
    from tractography.ukf import run_ukf, paths, current

    directory = directory / "synthetic"
    directory.mkdir()
    rng = np.random.default_rng(42)
    gradients = rng.normal(size=(30, 3))
    gradients /= np.linalg.norm(gradients, axis=1)[:, None]
    gradients = np.vstack((np.zeros((1, 3)), gradients))
    b = np.array([0] + [1000] * 30)
    tensor = np.diag([0.0017, 0.0003, 0.0003])
    signal = 1000 * np.exp(-b * np.einsum("ni,ij,nj->n", gradients, tensor, gradients))
    # Synthetic tensor field: tests the real pipeline, not anatomical accuracy.
    dwi = np.broadcast_to(signal, (14, 14, 14, 31)).astype(np.float32).copy()
    mask = np.zeros((14, 14, 14), np.uint8)
    mask[1:-1, 1:-1, 1:-1] = 1
    affine = np.array([[0, -2, 0, 14], [2, 0, 0, -14], [0, 0, 2, -14], [0, 0, 0, 1.0]])
    nib.save(nib.Nifti1Image(dwi, affine), directory / "data.nii.gz")
    nib.save(nib.Nifti1Image(mask, affine), directory / "mask.nii.gz")
    np.savetxt(directory / "bvals", b[None])
    np.savetxt(directory / "bvecs", gradients.T)
    raw = RawInputBundle(
        subject_root=directory,
        subject_id="synthetic",
        input_dir=directory,
        dwi_path=directory / "data.nii.gz",
        bval_path=directory / "bvals",
        bvec_path=directory / "bvecs",
        mask_path=directory / "mask.nii.gz",
    )
    options = {"numThreads": 2}  # Bound test resources; scientific defaults otherwise unchanged.
    result = run_ukf(raw, directory / "outputs", options=options, executable=binary, converter=converter, log=print)
    assert result["status"] == "completed"
    assert current(raw, directory / "outputs", options, binary, converter)
    qc = json.loads(paths("synthetic", directory / "outputs")["qc"].read_text())
    assert qc["automatic_qc_passed"] and qc["streamline_count"] > 0
    loaded = slicer.util.loadFiberBundle(str(paths("synthetic", directory / "outputs")["tractogram"]))
    assert loaded is not None
    print("UKF_REAL_SYNTHETIC_PASS", qc["streamline_count"], flush=True)


qt.QTimer.singleShot(4000, run)
