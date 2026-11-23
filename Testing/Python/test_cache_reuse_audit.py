#!/usr/bin/env python3
"""Comprehensive regression tests for stage result reuse and scientific cache integrity.

Verifies:
1. All true reuse bugs fixed across the pipeline (DDSurfer shared_raw, surface_postprocess
   relocation/runtime move, parallel NODDI/Free Water thread drift, CSD nthreads, SAF fs_dir mtime).
2. Overly brittle cache keys relaxed (.py source-byte hashing, execution/performance knobs,
   harmless mtime_ns drift with sha256 verification, subject/tool/runtime directory relocation).
3. Strict scientific invalidation preserved (modified DWI/bval/bvec/mask, scientific algorithm
   parameters, precision/flip/mode/atlases, explicit gradient-frame checks, and file tampering).
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import nibabel as nib
import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bump_mtime(path: Path, delta_ns: int = 5_000_000_000) -> None:
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + delta_ns))


def _save_nifti(path: Path, data: np.ndarray, affine: np.ndarray | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(np.asarray(data), np.diag([-2.0, 2.0, 2.0, 1.0]) if affine is None else affine),
        str(path),
    )
    return path


class DDSurferReuseAuditTests(unittest.TestCase):
    def _build_shared_raw_surface(self, root: Path, subject_id: str = "sub01"):
        from anatomy.surface.backend_bridge import native_products, upstream_record
        from shared.io_layout import subject_layout

        subject = root / subject_id
        input_dir = subject / "input"
        input_dir.mkdir(parents=True)
        dwi = _save_nifti(input_dir / "data.nii.gz", np.ones((2, 2, 2, 3), dtype=np.float32))
        mask = _save_nifti(input_dir / "nodif_brain_mask.nii.gz", np.ones((2, 2, 2), dtype=np.uint8))
        bval = input_dir / "bvals"
        bvec = input_dir / "bvecs"
        bval.write_text("0 1000 1000\n", encoding="utf-8")
        bvec.write_text("0 1 0\n0 0 1\n0 0 0\n", encoding="utf-8")

        layout = subject_layout(subject)
        surface_root = layout.ddsurfer
        native = surface_root / "pred_results_SlicerDSpace" / "native" / subject_id
        native.mkdir(parents=True)
        for hemi in ("lh", "rh"):
            for kind in ("white", "pial"):
                trimesh.creation.icosphere(subdivisions=1).export(str(native / f"{hemi}.{kind}.obj"))
        products, checks = native_products(native, subject_id, "all")

        reg_dir = layout.cache / "registered_dti" / "atlas__wls__nrrd"
        reg_dir.mkdir(parents=True)
        extra_inputs = [dwi, bval, bvec, mask]
        for idx in range(11):
            p = reg_dir / f"reg_item_{idx}.nii.gz"
            p.write_bytes(f"reg-{idx}".encode("utf-8"))
            extra_inputs.append(p)
        atlas = root / "surface_atlas_1mm.nii.gz"
        atlas.write_bytes(b"trained-surface-atlas-v1")
        extra_inputs.append(atlas)

        input_hashes = {str(p.resolve()): _sha256(p) for p in extra_inputs}
        # Simulate a legacy v3 manifest that also hashed 4 `.py` source files (which CodeOverlay edits).
        input_hashes["/old/install/anatomy/surface/pipeline.py"] = "0" * 64
        input_hashes["/old/install/anatomy/surface/preprocessing.py"] = "1" * 64
        input_hashes["/old/install/anatomy/surface/prediction.py"] = "2" * 64
        input_hashes["/old/install/anatomy/surface/backend_bridge.py"] = "3" * 64

        outputs = [*products, *(Path(c["indexed_master"]) for c in checks.values())]
        manifest = {
            "kind": "ddsurfer_surface",
            "version": 3,
            "status": "completed",
            "input_mode": "shared_raw",
            "subject_id": subject_id,
            "subject_root": str(subject.resolve()),
            "predict_mode": "all",
            "precision": "fp32",
            "device": "cpu",
            "resolved_device": "cpu",
            "cpu_threads": 4,
            "max_gpu_gib": 24.0,
            "runtime": {"executable": "/old/slicer/bin/PythonSlicer"},
            "resources": {"registration_atlas": str(atlas.resolve())},
            "upstream_commit": upstream_record()["commit"],
            "input_sha256": input_hashes,
            "output_sha256": {str(p.resolve()): _sha256(p) for p in outputs},
        }
        (surface_root / "surface_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (surface_root / "surface_qc.json").write_text(json.dumps({"automatic_qc_passed": True}), encoding="utf-8")
        return subject, surface_root, native, atlas, outputs

    def test_shared_raw_surface_complete_reuses_and_raw_still_rejected(self) -> None:
        from shared.orchestrator import ProductOrchestratorConfig, _surface_complete

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            subject, surface_root, native, atlas, _outputs = self._build_shared_raw_surface(root)

            # Bug #1 fixed: shared_raw is now reused by _surface_complete (even with stale .py hashes).
            self.assertTrue(_surface_complete(subject, "sub01"))

            # Non-scientific knobs (cpu_threads, max_gpu_gib, python_bin) must NOT invalidate.
            varied = ProductOrchestratorConfig(
                subject_id="sub01",
                subject_root=subject,
                targets=("surface",),
                surface_cpu_threads=12,
                surface_max_gpu_gib=8.0,
                surface_python_bin=root / "new_python",
                surface_atlas_t2=atlas,
            )
            self.assertTrue(_surface_complete(subject, "sub01", config=varied))

            # Moving the subject directory preserves reuse.
            moved = root / "relocated_tree" / "sub01"
            moved.parent.mkdir()
            shutil.move(str(subject), str(moved))
            self.assertTrue(_surface_complete(moved, "sub01"))

            # Scientific precision change MUST invalidate.
            self.assertFalse(
                _surface_complete(
                    moved,
                    "sub01",
                    config=ProductOrchestratorConfig(
                        subject_id="sub01", subject_root=moved, targets=("surface",), surface_precision="bf16"
                    ),
                )
            )

            # Different atlas content MUST invalidate.
            other_atlas = root / "other_atlas.nii.gz"
            other_atlas.write_bytes(b"different-atlas-content")
            self.assertFalse(
                _surface_complete(
                    moved,
                    "sub01",
                    config=ProductOrchestratorConfig(
                        subject_id="sub01", subject_root=moved, targets=("surface",), surface_atlas_t2=other_atlas
                    ),
                )
            )

            # Modifying raw bvals MUST invalidate shared_raw.
            moved_bvals = moved / "input" / "bvals"
            old_bvals = moved_bvals.read_text(encoding="utf-8")
            moved_bvals.write_text("0 2000 2000\n", encoding="utf-8")
            self.assertFalse(_surface_complete(moved, "sub01"))
            moved_bvals.write_text(old_bvals, encoding="utf-8")
            self.assertTrue(_surface_complete(moved, "sub01"))

            # Tampering an output surface MUST invalidate.
            from shared.io_layout import subject_layout
            moved_surface_root = subject_layout(moved).ddsurfer
            moved_obj = moved_surface_root / "pred_results_SlicerDSpace" / "native" / "sub01" / "lh.white.obj"
            orig_bytes = moved_obj.read_bytes()
            moved_obj.write_bytes(b"tampered-mesh")
            self.assertFalse(_surface_complete(moved, "sub01"))
            moved_obj.write_bytes(orig_bytes)
            self.assertTrue(_surface_complete(moved, "sub01"))

            # input_mode == 'raw' MUST still be rejected by _surface_complete.
            mpath = moved_surface_root / "surface_manifest.json"
            raw_manifest = json.loads(mpath.read_text(encoding="utf-8"))
            raw_manifest["input_mode"] = "raw"
            mpath.write_text(json.dumps(raw_manifest), encoding="utf-8")
            self.assertFalse(_surface_complete(moved, "sub01"))

    def test_run_shared_raw_reuses_across_py_and_exec_knobs_and_skips_stl_rewrite(self) -> None:
        from anatomy.surface.backend_bridge import native_products
        from anatomy.surface.pipeline import SurfaceConfig, _run_shared_raw
        from shared import pipeline as shared_pipeline
        from test_shared_preparation import inputs as make_raw_inputs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            raw = make_raw_inputs(root)
            atlas = root / "surface-atlas.nii.gz"
            atlas.write_bytes(b"atlas")
            bundle = shared_pipeline._build_registered_bundle(raw, "wls", atlas, "nrrd")
            bundle.bundle_dir.mkdir(parents=True)
            for name in bundle.__dataclass_fields__:
                if name.endswith("_path"):
                    getattr(bundle, name).write_bytes(b"reg-fixture")
            bundle.manifest_path.write_text(json.dumps({"native_bundle": {"dir": str(root / "native")}}))

            weights, templates = root / "weights", root / "templates"
            weights.mkdir()
            templates.mkdir()
            (weights / "manifest.json").write_text("{}")
            checkpoints, shapes = {}, {}
            for hemi in ("left", "right"):
                checkpoints[hemi] = {}
                for kind in ("white", "pial"):
                    fn = f"{hemi}.{kind}.pt"
                    checkpoints[hemi][kind] = fn
                    (weights / fn).write_bytes(b"ckpt")
                shapes[hemi] = {"file": hemi + ".obj"}
                (templates / (hemi + ".obj")).write_bytes(b"tpl")
            resources = {
                "registration_atlas": str(atlas),
                "checkpoint_root": str(weights),
                "template_dir": str(templates),
                "model_config": {"checkpoint_files": checkpoints, "templates": shapes},
            }

            out = root / "out"
            native = out / "pred_results_SlicerDSpace" / "native" / "x"
            native.mkdir(parents=True)
            for hemi in ("lh", "rh"):
                for kind in ("white", "pial"):
                    trimesh.creation.icosphere(subdivisions=1).export(str(native / f"{hemi}.{kind}.obj"))
            products, checks = native_products(native, "x", "all")
            stl_mtime_before = {p: p.stat().st_mtime_ns for p in products}

            # First run populates manifest (no .py files inside input_sha256!).
            cfg1 = SurfaceConfig("x", root, out, input_bundle=raw, resume=True, device="cpu", cpu_threads=4, max_gpu_gib=16.0)
            m1 = {"status": "running", "subject_id": "x", "predict_mode": "all", "precision": "fp32", "upstream_commit": "abc"}
            with mock.patch.object(shared_pipeline, "ensure_registered_dti_bundle", return_value=bundle), \
                    mock.patch("anatomy.surface.pipeline._run_registered") as pred:
                reused1 = _run_shared_raw(cfg1, resources, out, m1, {}, None)
            self.assertFalse(reused1)
            self.assertEqual(pred.call_count, 1)
            self.assertFalse(any(str(k).endswith(".py") for k in m1["input_sha256"]))

            # Second run with legacy `.py` entries injected into previous + changed device/threads/gpu/runtime -> REUSED!
            prev = {
                **m1,
                "status": "completed",
                "device": "cuda:0",
                "resolved_device": "cuda:0",
                "cpu_threads": 16,
                "max_gpu_gib": 8.0,
                "runtime": {"executable": "/other/python"},
                "input_sha256": {**m1["input_sha256"], "/old/pipeline.py": "deadbeef"},
                "output_sha256": {
                    str(p.resolve()): _sha256(p)
                    for p in [*products, *(Path(c["indexed_master"]) for c in checks.values())]
                },
            }
            cfg2 = SurfaceConfig("x", root, out, input_bundle=raw, resume=True, device="mps", cpu_threads=2, max_gpu_gib=4.0)
            m2 = {"status": "running", "subject_id": "x", "predict_mode": "all", "precision": "fp32", "upstream_commit": "abc"}
            with mock.patch.object(shared_pipeline, "ensure_registered_dti_bundle", return_value=bundle), \
                    mock.patch("anatomy.surface.pipeline._run_registered") as pred:
                reused2 = _run_shared_raw(cfg2, resources, out, m2, prev, None)
            self.assertTrue(reused2)
            pred.assert_not_called()

            # Reused run calling native_products(..., export_stl=False) must NOT rewrite STLs.
            native_products(native, "x", "all", export_stl=False)
            self.assertEqual({p: p.stat().st_mtime_ns for p in products}, stl_mtime_before)

            # Precision change MUST rerun.
            m3 = {"status": "running", "subject_id": "x", "predict_mode": "all", "precision": "bf16", "upstream_commit": "abc"}
            with mock.patch.object(shared_pipeline, "ensure_registered_dti_bundle", return_value=bundle), \
                    mock.patch("anatomy.surface.pipeline._run_registered") as pred:
                reused3 = _run_shared_raw(replace(cfg2, precision="bf16"), resources, out, m3, prev, None)
            self.assertFalse(reused3)
            self.assertEqual(pred.call_count, 1)


class SurfacePostprocessReuseAuditTests(unittest.TestCase):
    def test_postprocess_reuses_across_move_mtime_and_runtime_relocation(self) -> None:
        from anatomy.surface.postprocess_release import DDSurferPostprocessConfig, run_ddsurfer_postprocess
        from shared.io_layout import subject_layout
        from shared.orchestrator import _surface_postprocess_complete
        from test_surface_postprocess_qc import write_postprocess_fixture

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            root = base / "old_dir" / "sub01"
            layout = subject_layout(root)
            native = layout.ddsurfer / "pred_results_SlicerDSpace" / "native" / "sub01"
            native.mkdir(parents=True)
            for hemi in ("lh", "rh"):
                for kind in ("white", "pial"):
                    trimesh.creation.icosphere(subdivisions=1).export(str(native / f"{hemi}.{kind}.obj"))

            ref_mri = root / "reference"
            ref_mri.mkdir(parents=True)
            nib.save(nib.MGHImage(np.zeros((3, 3, 3), dtype=np.float32), np.eye(4)), ref_mri / "brain.mgz")

            rt_v1 = base / "runtime_v1"
            rt_v1.mkdir()
            (rt_v1 / "runtime.json").write_text('{"version": "1.0.0"}\n', encoding="utf-8")

            def fake_backend(command, **kwargs):
                output = Path(command[command.index("--output-root") + 1]) / "sub01"
                write_postprocess_fixture(output)

            cfg = DDSurferPostprocessConfig("sub01", layout.outputs, subject_root=root, native_dir=native, mri_dir=ref_mri)
            with mock.patch("anatomy.surface.backend_bridge.run_backend", side_effect=fake_backend), \
                    mock.patch("shared.surface_runtime.resolve_surface_runtime", return_value=("native", rt_v1)):
                result = run_ddsurfer_postprocess(cfg)
                self.assertTrue(_surface_postprocess_complete(root, "sub01"))

                # POSIX canonical output links must be relative so folder moves keep them intact.
                if os.name == "posix" and (result.structural_root / "surf").is_symlink():
                    self.assertFalse(os.path.isabs(os.readlink(result.structural_root / "surf")))

                # 1. mtime_ns bump on outputs must NOT invalidate (sha256 matches!).
                _bump_mtime(result.structural_root / "surf" / "lh.thickness")
                _bump_mtime(result.structural_root / "label" / "lh.aparc.annot")
                self.assertTrue(_surface_postprocess_complete(root, "sub01"))

            # 2. Moving surface_runtime to a new install path with identical runtime.json MUST reuse!
            rt_v2 = base / "runtime_v2"
            shutil.move(str(rt_v1), str(rt_v2))
            with mock.patch("shared.surface_runtime.resolve_surface_runtime", return_value=("native", rt_v2)):
                self.assertTrue(_surface_postprocess_complete(root, "sub01"))
                self.assertTrue(_surface_postprocess_complete(root, "sub01", runtime=rt_v2))

                # 3. Moving the entire subject folder MUST reuse!
                moved = base / "new_location" / "sub01"
                moved.parent.mkdir()
                shutil.move(str(root), str(moved))
                self.assertTrue(_surface_postprocess_complete(moved, "sub01"))

                # 4. Legacy pre-sha256 output_artifacts with mtime drift MUST still reuse if decoded QC passes.
                moved_qc = moved / "cache" / "stages" / "01_anatomical_priors" / "surface_morphometry" / "structural" / "qc.json"
                qc_data = json.loads(moved_qc.read_text(encoding="utf-8"))
                legacy_qc = json.loads(json.dumps(qc_data))
                for entry in legacy_qc["integrity"]["output_artifacts"].values():
                    entry.pop("sha256", None)
                moved_qc.write_text(json.dumps(legacy_qc), encoding="utf-8")
                _bump_mtime(moved_qc.parent / "surf" / "rh.thickness")
                self.assertTrue(_surface_postprocess_complete(moved, "sub01"))
                moved_qc.write_text(json.dumps(qc_data), encoding="utf-8")

                # 5. Tampering thickness to same-size NaN values MUST invalidate!
                thick = moved_qc.parent / "surf" / "lh.thickness"
                orig_thick = thick.read_bytes()
                vals = nib.freesurfer.read_morph_data(thick)
                vals[0] = np.nan
                nib.freesurfer.write_morph_data(thick, vals)
                self.assertFalse(_surface_postprocess_complete(moved, "sub01"))
                thick.write_bytes(orig_thick)
                self.assertTrue(_surface_postprocess_complete(moved, "sub01"))

                # 6. Changing runtime.json (real runtime upgrade) MUST invalidate!
                (rt_v2 / "runtime.json").write_text('{"version": "2.0.0"}\n', encoding="utf-8")
                self.assertFalse(_surface_postprocess_complete(moved, "sub01"))


class DDParcelAndDDSegReuseAuditTests(unittest.TestCase):
    def test_ddparcel_reuses_across_py_edits_device_mtime_and_move(self) -> None:
        from anatomy.parcellation.pipeline import DDParcelRegisteredConfig, run_ddparcel_from_registered_bundle
        from anatomy.parcellation.qc import validate_outputs
        from shared import pipeline as shared_pipeline
        from shared.io_layout import subject_layout
        from shared.orchestrator import _parcellation_complete, _parcellation_outputs
        from test_shared_preparation import inputs as make_raw_inputs

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            subject = base / "orig" / "x"
            subject.mkdir(parents=True)
            raw = make_raw_inputs(subject)
            layout = subject_layout(subject)
            atlas = base / "atlas.nii.gz"
            atlas.write_bytes(b"atlas")
            bundle = shared_pipeline._build_registered_bundle(raw, "wls", atlas)
            bundle.bundle_dir.mkdir(parents=True)
            for name in bundle.__dataclass_fields__:
                if name.endswith("_path"):
                    getattr(bundle, name).write_bytes(b"reg-data-" + name.encode())
            bundle.manifest_path.write_text(
                json.dumps(
                    {
                        "kind": "registered_dti_bundle",
                        "native_bundle": {
                            "manifest": {
                                "source_files": {
                                    key: shared_pipeline._cache_file_signature(getattr(raw, key + "_path"))
                                    for key in ("dwi", "bval", "bvec", "mask")
                                }
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            weights = base / "weights_v1"
            weights.mkdir()
            (weights / "model.pkl").write_bytes(b"ddparcel-weights")
            out = _parcellation_outputs(subject, "x")["wmparc"].parent
            cfg = DDParcelRegisteredConfig("x", bundle, out, raw.mask_path, weights, device="cpu")

            label_data = np.ones((4, 5, 6), dtype=np.int16) * 2
            label_data[1, 1, 1] = 1001
            affine = nib.load(str(raw.mask_path)).affine

            def normalize(src, msk, dst, flip):
                _save_nifti(dst, np.ones((4, 5, 6), dtype=np.float32), affine)

            def predict(folder, sid, wdir, log, **kw):
                nib.save(nib.MGHImage(label_data.astype(np.int32), affine), str(folder / f"{sid}-DDSurfer-wmparc-Reg.mgz"))

            def resample(cmd, log=None):
                _save_nifti(Path(cmd[-1]), label_data, affine)

            with mock.patch("anatomy.parcellation.pipeline._run_normalize", side_effect=normalize), \
                    mock.patch("anatomy.parcellation.pipeline._run_ddsurfer_pred", side_effect=predict) as pred, \
                    mock.patch("anatomy.parcellation.pipeline.resolve_cli_executable", return_value="resample"), \
                    mock.patch("anatomy.parcellation.pipeline._run_cli", side_effect=resample):
                run_ddparcel_from_registered_bundle(cfg)
                self.assertEqual(pred.call_count, 1)

                # Inject stale `.py` hashes into ddparcel_preparation.json (simulating CodeOverlay update)
                # and change device -> MUST reuse!
                receipt_path = out / "ddparcel_preparation.json"
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                self.assertFalse(any(str(k).endswith(".py") for k in receipt["inputs"]))
                receipt["inputs"]["/old/anatomy/parcellation/pipeline.py"] = "0" * 64
                receipt["inputs"]["/old/anatomy/parcellation/DDSurfer_Pred.py"] = "1" * 64
                receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

                run_ddparcel_from_registered_bundle(replace(cfg, device="mps"))
                self.assertEqual(pred.call_count, 1)

                # Touch mtime on outputs -> _parcellation_complete STILL True!
                _bump_mtime(out / "x-DDSurfer-wmparc.nii.gz")
                with mock.patch("shared.orchestrator._resolve_analysis_inputs", return_value={
                    "dwi": raw.dwi_path, "bval": raw.bval_path, "bvec": raw.bvec_path, "mask": raw.mask_path
                }):
                    self.assertTrue(_parcellation_complete(subject, "x"))

                # Move subject folder -> _parcellation_complete STILL True!
                moved = base / "moved" / "x"
                moved.parent.mkdir()
                shutil.move(str(subject), str(moved))
                moved_inputs = {
                    "dwi": moved / "input" / "data.nii.gz",
                    "bval": moved / "input" / "bvals",
                    "bvec": moved / "input" / "bvecs",
                    "mask": moved / "input" / "mask.nii.gz",
                }
                with mock.patch("shared.orchestrator._resolve_analysis_inputs", return_value=moved_inputs):
                    self.assertTrue(_parcellation_complete(moved, "x"))
                    # Tampering wmparc MUST invalidate!
                    moved_wmparc = _parcellation_outputs(moved, "x")["wmparc"]
                    _save_nifti(moved_wmparc, label_data + 1, affine)
                    self.assertFalse(_parcellation_complete(moved, "x"))

    def test_ddseg_reuses_across_move_and_mtime_and_rejects_tamper(self) -> None:
        from anatomy.segmentation.config import DDSegConfig
        from anatomy.segmentation.qc import write_ddseg_qc_record
        from shared.orchestrator import _segmentation_complete, _segmentation_outputs, _write_anatomy_source
        from test_shared_preparation import inputs as make_raw_inputs

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            subject = base / "orig" / "sub01"
            subject.mkdir(parents=True)
            raw = make_raw_inputs(subject)
            outs = _segmentation_outputs(subject)
            seg_dir = outs["qc"].parent
            seg_dir.mkdir(parents=True)
            labels = np.ones((4, 5, 6), dtype=np.int16)
            probs = np.zeros((4, 5, 6, 4), dtype=np.float32)
            probs[..., 1] = 1.0
            for key in ("segmentation", "gm_probability", "wm_probability", "csf_probability"):
                _save_nifti(outs[key], labels if key == "segmentation" else probs[..., 1])
            write_ddseg_qc_record(
                DDSegConfig(
                    parameter_type="WLS",
                    input_mask_nii=raw.mask_path,
                    output_folder=seg_dir,
                    weights_dir=subject / "weights",
                    source_dwi=raw.dwi_path,
                ),
                labels=labels,
                probability_maps=probs,
                outputs={k: v for k, v in outs.items() if k != "qc"},
            )
            _write_anatomy_source(seg_dir / "ddseg_source.json", raw)

            with mock.patch("shared.orchestrator._resolve_analysis_inputs", return_value={
                "dwi": raw.dwi_path, "bval": raw.bval_path, "bvec": raw.bvec_path, "mask": raw.mask_path
            }):
                self.assertTrue(_segmentation_complete(subject))
                _bump_mtime(raw.dwi_path)
                _bump_mtime(outs["gm_probability"])
                self.assertTrue(_segmentation_complete(subject))

            moved = base / "moved" / "sub01"
            moved.parent.mkdir()
            shutil.move(str(subject), str(moved))
            moved_inputs = {
                "dwi": moved / "input" / "data.nii.gz",
                "bval": moved / "input" / "bvals",
                "bvec": moved / "input" / "bvecs",
                "mask": moved / "input" / "mask.nii.gz",
            }
            with mock.patch("shared.orchestrator._resolve_analysis_inputs", return_value=moved_inputs):
                self.assertTrue(_segmentation_complete(moved))
                # Tamper probability map -> MUST invalidate!
                _save_nifti(_segmentation_outputs(moved)["gm_probability"], probs[..., 1] * 0.5)
                self.assertFalse(_segmentation_complete(moved))


class MicrostructureReuseAuditTests(unittest.TestCase):
    def test_freewater_and_noddi_reuse_after_parallel_thread_adjustment_and_move(self) -> None:
        from microstructure.pipeline import (
            FREE_WATER_KEYS,
            MicrostructureConfig,
            NODDI_KEYS,
            _expected_free_water_manifest,
            _expected_noddi_manifest,
            _finish_scalar_qc,
            _output_paths,
            _write_json,
        )
        from shared.orchestrator import (
            ProductOrchestratorConfig,
            _microstructure_freewater_complete,
            _microstructure_noddi_complete,
            _microstructure_stage_dir,
        )
        from shared.io_layout import subject_layout
        from shared.pipeline import RawInputBundle

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            subject = base / "orig" / "sub01"
            inp = subject / "input"
            inp.mkdir(parents=True)
            dwi = _save_nifti(inp / "data.nii.gz", np.ones((2, 2, 2, 4), dtype=np.float32))
            mask = _save_nifti(inp / "nodif_brain_mask.nii.gz", np.ones((2, 2, 2), dtype=np.uint8))
            bval = inp / "bvals"
            bvec = inp / "bvecs"
            bval.write_text("0 1000 2000 2000\n", encoding="utf-8")
            bvec.write_text("0 1 0 0\n0 0 1 0\n0 0 0 1\n", encoding="utf-8")
            raw = RawInputBundle(subject, "sub01", inp, dwi, bval, bvec, mask)

            stage_dir = _microstructure_stage_dir(subject_layout(subject))
            paths = _output_paths("sub01", stage_dir)

            # Write Free Water & NODDI as produced under parallel mode (reduced threads/jobs, joblib engine).
            parallel_cfg = MicrostructureConfig(
                "sub01",
                raw,
                None,
                stage_dir,
                free_water_chunk_size=4096,
                free_water_engine="joblib",
                free_water_n_jobs=2,
                free_water_vox_per_chunk=512,
                noddi_nthreads=3,
            )
            for key in (*FREE_WATER_KEYS, *NODDI_KEYS):
                _save_nifti(paths[key], np.ones((2, 2, 2), dtype=np.float32) * 0.2)
            _write_json(paths["free_water_manifest"], _expected_free_water_manifest(parallel_cfg))
            _write_json(paths["noddi_manifest"], _expected_noddi_manifest(parallel_cfg))
            _finish_scalar_qc(paths, "free_water_qc", FREE_WATER_KEYS, {"checks": {"ok": True}})
            _finish_scalar_qc(paths, "noddi_qc", NODDI_KEYS, {"checks": {"ok": True}})

            # Orchestrator resumes with default serial / auto-thread settings -> MUST reuse both!
            orch_cfg = ProductOrchestratorConfig(
                subject_id="sub01",
                subject_root=subject,
                targets=("microstructure_freewater", "microstructure_noddi"),
                microstructure_free_water_engine="serial",
                microstructure_free_water_chunk_size=1024,
            )
            self.assertTrue(_microstructure_freewater_complete(subject, "sub01", config=orch_cfg))
            self.assertTrue(_microstructure_noddi_complete(subject, "sub01", config=orch_cfg))

            # mtime drift + folder move -> STILL reusable!
            _bump_mtime(dwi)
            _bump_mtime(paths["noddi_ndi"])
            moved = base / "moved" / "sub01"
            moved.parent.mkdir()
            shutil.move(str(subject), str(moved))
            moved_cfg = replace(orch_cfg, subject_root=moved)
            self.assertTrue(_microstructure_freewater_complete(moved, "sub01", config=moved_cfg))
            self.assertTrue(_microstructure_noddi_complete(moved, "sub01", config=moved_cfg))

            # Scientific parameter change MUST invalidate!
            self.assertFalse(
                _microstructure_freewater_complete(
                    moved, "sub01", config=replace(moved_cfg, microstructure_free_water_max_fraction_for_tissue=0.5)
                )
            )
            # Tampering an output map MUST invalidate!
            moved_paths = _output_paths("sub01", _microstructure_stage_dir(subject_layout(moved)))
            _save_nifti(moved_paths["noddi_ndi"], np.ones((2, 2, 2), dtype=np.float32) * 0.9)
            self.assertFalse(_microstructure_noddi_complete(moved, "sub01", config=moved_cfg))

    def test_roi_stats_and_ifw_reuse_across_lut_move_and_reject_lut_edit(self) -> None:
        from microstructure.ifw import IfwConfig, _roi_table_current, ifw_output_paths
        from microstructure.pipeline import _compute_roi_stats, roi_statistics_current

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            roi = _save_nifti(root / "roi.nii.gz", np.array([[[1, 1], [2, 2]]], dtype=np.int16))
            mask = _save_nifti(root / "mask.nii.gz", np.ones((1, 2, 2), dtype=np.uint8))
            ifw_dir = root / "ifw"
            paths = ifw_output_paths("sub01", ifw_dir)
            _save_nifti(paths["ifw"], np.full((1, 2, 2), 0.1, dtype=np.float32))
            _save_nifti(paths["fw_volume_fraction"], np.full((1, 2, 2), 0.2, dtype=np.float32))
            lut1 = root / "install_v1" / "FreeSurferColorLUT.txt"
            lut1.parent.mkdir()
            lut1.write_text("1 RegionOne 1 2 3 0\n2 RegionTwo 4 5 6 0\n", encoding="utf-8")

            _compute_roi_stats(
                metric_paths={"ifw": paths["ifw"], "fw_volume_fraction": paths["fw_volume_fraction"]},
                roi_label_map=roi,
                lut_path=lut1,
                csv_path=paths["roi_stats_csv"],
                json_path=paths["roi_stats_json"],
                brain_mask_path=mask,
            )

            # Move LUT to a new install path with identical bytes -> MUST reuse!
            lut2 = root / "install_v2" / "FreeSurferColorLUT.txt"
            lut2.parent.mkdir()
            shutil.move(str(lut1), str(lut2))
            _bump_mtime(lut2)
            _bump_mtime(paths["ifw"])
            self.assertTrue(
                roi_statistics_current(
                    paths["roi_stats_json"],
                    paths["roi_stats_csv"],
                    roi_label_map=roi,
                    required_metrics=("ifw", "fw_volume_fraction"),
                    brain_mask_path=mask,
                    lut_path=lut2,
                )
            )
            cfg = IfwConfig(
                subject_id="sub01",
                fw_fraction_path=paths["fw_volume_fraction"],
                csf_probability_path=paths["ifw"],
                mask_path=mask,
                bval_path=root / "bvals",
                bvec_path=root / "bvecs",
                output_dir=ifw_dir,
                roi_label_map=roi,
                lut_path=lut2,
            )
            self.assertTrue(_roi_table_current(cfg, paths))

            # Editing LUT content MUST invalidate!
            lut2.write_text("1 RegionRenamed 1 2 3 0\n2 RegionTwo 4 5 6 0\n", encoding="utf-8")
            self.assertFalse(_roi_table_current(cfg, paths))

    def test_dki_and_mkcurve_reuse_across_py_edits_workers_mtime_and_move(self) -> None:
        from microstructure.dki import DkiConfig, dki_outputs_current, run_dki
        from shared.pipeline import RawInputBundle
        from test_microstructure_dki import synthetic_case, write_case

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            subject = base / "orig" / "sub01"
            paths = write_case(subject, synthetic_case(noisy=True))
            raw = RawInputBundle(
                subject, "sub01", subject / "input", paths["dwi"], paths["bval"], paths["bvec"], paths["mask"]
            )
            cfg = DkiConfig("sub01", raw, subject / "outputs" / "dki", fit_method="MKCURVE", voxels_per_chunk=4, num_threads=1)
            with mock.patch("microstructure.mkcurve.synthetic_b0_grid", return_value=np.arange(100, 2200, 100.0)):
                result = run_dki(cfg)
                self.assertTrue(dki_outputs_current(cfg))

                # Simulate CodeOverlay edit to `.py` files inside mkcurve_backend.source_sha256.
                manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
                manifest["signature"]["mkcurve_backend"]["source_sha256"]["mkcurve.py"] = "a" * 64
                manifest["signature"]["mkcurve_backend"]["source_sha256"]["chunk_dispatch.py"] = "b" * 64
                result.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

                # Changing execution knobs + mtime drift -> MUST reuse!
                _bump_mtime(result.mk_path)
                _bump_mtime(result.mkcurve_outputs["corrected_mk"])
                varied = replace(
                    cfg,
                    voxels_per_chunk=32,
                    num_threads=4,
                    mkcurve_workers=2,
                    mkcurve_voxels_per_chunk=16,
                    mkcurve_chunk_schedule="dynamic",
                )
                self.assertTrue(dki_outputs_current(varied))

                # Moving the subject directory -> MUST reuse!
                moved = base / "moved" / "sub01"
                moved.parent.mkdir()
                shutil.move(str(subject), str(moved))
                moved_raw = RawInputBundle(
                    moved,
                    "sub01",
                    moved / "input",
                    moved / "input" / "data.nii.gz",
                    moved / "input" / "bvals",
                    moved / "input" / "bvecs",
                    moved / "input" / "nodif_brain_mask.nii.gz",
                )
                moved_cfg = replace(varied, raw_input_bundle=moved_raw, output_dir=moved / "outputs" / "dki")
                self.assertTrue(dki_outputs_current(moved_cfg))

                # Changing scientific parameter MUST invalidate!
                self.assertFalse(dki_outputs_current(replace(moved_cfg, b0_threshold=100.0)))
                self.assertFalse(dki_outputs_current(replace(moved_cfg, fit_method="OLS")))

                # Changing mkcurve_directions.json digest MUST invalidate!
                moved_mpath = moved / "outputs" / "dki" / "dki_manifest.json"
                bad_dir_manifest = json.loads(moved_mpath.read_text(encoding="utf-8"))
                bad_dir_manifest["signature"]["mkcurve_backend"]["source_sha256"]["mkcurve_directions.json"] = "0" * 64
                moved_mpath.write_text(json.dumps(bad_dir_manifest), encoding="utf-8")
                self.assertFalse(dki_outputs_current(moved_cfg))


class PreprocessingAndBrainMaskReuseAuditTests(unittest.TestCase):
    def test_preprocessing_reuses_across_py_edits_threads_env_and_tool_moves(self) -> None:
        from preprocessing import DwiPreprocessingConfig, run_dwi_preprocessing
        from shared.orchestrator import ProductOrchestratorConfig, _preprocessing_complete
        from test_preprocessing_pipeline import _fake_pnl_mask, _write_fake_backend, _write_nifti_subject

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            subject = _write_nifti_subject(root)
            bin_v1 = root / "bin_v1"
            _write_fake_backend(bin_v1)

            cfg = DwiPreprocessingConfig(
                subject_id="100307",
                input_root=subject,
                execute=True,
                mrtrix_bin_dir=bin_v1,
                fsl_bin_dir=bin_v1,
                eddy_threads=2,
            )
            with mock.patch("preprocessing.pipeline.generate_brain_mask", side_effect=_fake_pnl_mask):
                first = run_dwi_preprocessing(cfg)
                self.assertEqual(first.status, "completed")

                # Simulate CodeOverlay touching `.py` records + moving tool bin dir + changing eddy_threads & CUDA env.
                manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
                manifest["cache_key"]["eddy_device_policy"]["mtime_ns"] = 1
                manifest["cache_key"]["eddy_device_policy"]["sha256"] = "0" * 64
                first.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

                bin_v2 = root / "bin_v2"
                shutil.move(str(bin_v1), str(bin_v2))
                _bump_mtime(subject / "input" / "bvals")

                cfg2 = replace(cfg, mrtrix_bin_dir=bin_v2, fsl_bin_dir=bin_v2, eddy_threads=8, brain_mask_device="cpu")
                with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
                    cached = run_dwi_preprocessing(cfg2)
                    self.assertEqual(cached.status, "cached")
                    orch_cfg = ProductOrchestratorConfig(
                        subject_id="100307",
                        subject_root=subject,
                        targets=("preprocessing",),
                        preprocessing_mrtrix_bin_dir=bin_v2,
                        preprocessing_fsl_bin_dir=bin_v2,
                        preprocessing_eddy_threads=8,
                    )
                    self.assertTrue(_preprocessing_complete(subject, "100307", config=orch_cfg))

                # Same-size bval content change MUST STILL invalidate!
                (subject / "input" / "bvals").write_text("0 2000 2000\n", encoding="utf-8")
                self.assertFalse(_preprocessing_complete(subject, "100307", config=orch_cfg))

    def test_brain_mask_reuses_across_subject_and_tool_relocation_and_rejects_edit(self) -> None:
        from shared import brain_mask
        from shared.brain_mask import generate_brain_mask
        from test_brain_mask_backends import (
            ISOLATED_ENV,
            fake_pnl_config,
            fake_pythonslicer,
            make_subject,
            mask_data,
            save_nifti,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with mock.patch.dict(os.environ, ISOLATED_ENV), \
                    mock.patch.object(brain_mask.platform, "system", return_value="Darwin"):
                config = fake_pnl_config(root)
                tool_v1 = fake_pythonslicer(root)
                subject_v1 = make_subject(root / "loc1")
                out_v1 = subject_v1 / "preprocessed" / "nodif_brain_mask.nii.gz"

                with mock.patch.object(brain_mask, "_pnl_native_python", return_value=tool_v1):
                    first = generate_brain_mask(
                        subject_v1 / "input" / "data.nii.gz",
                        subject_v1 / "input" / "bvals",
                        subject_v1 / "input" / "bvecs",
                        out_v1,
                        config=config,
                    )
                    self.assertEqual(first["status"], "completed")

                # Move BOTH subject folder AND PythonSlicer install directory -> MUST reuse!
                subject_v2 = root / "loc2" / "sub-01"
                subject_v2.parent.mkdir()
                shutil.move(str(subject_v1), str(subject_v2))
                tool_v2 = root / "SlicerMoved" / "bin" / "PythonSlicer"
                tool_v2.parent.mkdir(parents=True)
                shutil.move(str(tool_v1), str(tool_v2))
                out_v2 = subject_v2 / "preprocessed" / "nodif_brain_mask.nii.gz"

                with mock.patch.object(brain_mask, "_pnl_native_python", return_value=tool_v2), \
                        mock.patch.object(brain_mask, "_run") as run_spy:
                    reused = generate_brain_mask(
                        subject_v2 / "input" / "data.nii.gz",
                        subject_v2 / "input" / "bvals",
                        subject_v2 / "input" / "bvecs",
                        out_v2,
                        config=config,
                    )
                    run_spy.assert_not_called()
                    self.assertEqual(reused["status"], "reused")

                    # Editing the generated mask MUST STILL raise ValueError!
                    edited = mask_data()
                    edited[0, 0, 0] = 1
                    save_nifti(out_v2, edited)
                    with self.assertRaisesRegex(ValueError, "Generated brain mask was edited"):
                        generate_brain_mask(
                            subject_v2 / "input" / "data.nii.gz",
                            subject_v2 / "input" / "bvals",
                            subject_v2 / "input" / "bvecs",
                            out_v2,
                            config=config,
                        )


class TractographyBundlesReuseAuditTests(unittest.TestCase):
    def test_csd_parameters_and_fod_reuse_ignore_nthreads_and_keep_scientific_params(self) -> None:
        from shared.orchestrator import ProductOrchestratorConfig, _csd_bundle_parameters_current
        from tractography.csd import (
            CSDActConfig,
            _fod_preparation_paths,
            _prepared_fod_reusable,
            _write_fod_preparation_receipt,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            manifest_path = root / "csd_act_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "parameters": {
                            "select_streamlines": 1000,
                            "min_length_mm": 10.0,
                            "max_length_mm": 250.0,
                            "random_seed": 12345,
                            "seeding": "dynamic",
                            "nthreads": 2,
                        }
                    }
                ),
                encoding="utf-8",
            )
            cfg = ProductOrchestratorConfig(
                subject_id="sub01",
                subject_root=root,
                targets=("tractography",),
                select_streamlines=1000,
                mrtrix_random_seed=12345,
                mrtrix_nthreads=8,
            )
            self.assertTrue(_csd_bundle_parameters_current(cfg, manifest_path))
            self.assertFalse(_csd_bundle_parameters_current(replace(cfg, select_streamlines=5000), manifest_path))
            self.assertFalse(_csd_bundle_parameters_current(replace(cfg, mrtrix_random_seed=99), manifest_path))

            # FOD preparation reuse ignores pretracking_nthreads and mtime drift.
            dwi, bval, bvec, mask = (root / n for n in ("data.nii.gz", "bvals", "bvecs", "mask.nii.gz"))
            for p in (dwi, bval, bvec, mask):
                p.write_text("stub-" + p.name, encoding="utf-8")
            out_dir = root / "csd"
            out_dir.mkdir()
            for p in _fod_preparation_paths("sub01", out_dir).values():
                p.write_bytes(b"fod-artifact")
            fod_cfg = CSDActConfig("sub01", dwi, bval, bvec, mask, root / "5tt", root / "gmwmi", out_dir, nthreads=2)
            self.assertTrue(_write_fod_preparation_receipt(fod_cfg, out_dir))
            _bump_mtime(bval)
            self.assertTrue(_prepared_fod_reusable(replace(fod_cfg, nthreads=12), out_dir))
            self.assertFalse(_prepared_fod_reusable(replace(fod_cfg, random_seed=7), out_dir))
            bval.write_text("changed-bvals", encoding="utf-8")
            self.assertFalse(_prepared_fod_reusable(fod_cfg, out_dir))

    def test_ukf_reuses_across_threads_subject_move_and_install_dir_move(self) -> None:
        from shared.pipeline import RawInputBundle
        from tractography import ukf

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            inst_v1 = base / "slicer_v1" / "cli-modules"
            inst_v1.mkdir(parents=True)
            exe_v1 = inst_v1 / "UKFTractography"
            conv_v1 = inst_v1 / "DWIConvert"
            lib_v1 = inst_v1 / "libUKFBase.dylib"
            exe_v1.write_bytes(b"ukf-bin-v1")
            conv_v1.write_bytes(b"conv-bin-v1")
            lib_v1.write_bytes(b"ukf-lib-v1")

            sub_v1 = base / "loc1" / "sub01"
            inp = sub_v1 / "input"
            inp.mkdir(parents=True)
            dwi = _save_nifti(inp / "data.nii.gz", np.ones((2, 2, 2, 3), dtype=np.float32))
            mask = _save_nifti(inp / "mask.nii.gz", np.ones((2, 2, 2), dtype=np.uint8))
            bval = inp / "bvals"
            bvec = inp / "bvecs"
            bval.write_text("0 1000 1000\n", encoding="utf-8")
            bvec.write_text("0 1 0\n0 0 1\n0 0 0\n", encoding="utf-8")
            raw_v1 = RawInputBundle(sub_v1, "sub01", inp, dwi, bval, bvec, mask)

            out_v1 = sub_v1 / "ukf_out"
            out_v1.mkdir()
            prods = ukf.paths("sub01", out_v1)
            for k, p in prods.items():
                if k == "manifest":
                    continue
                if k == "qc":
                    p.write_text(json.dumps({"automatic_qc_passed": True}), encoding="utf-8")
                else:
                    p.write_bytes(b"ukf-out-" + k.encode())
            sig = ukf.signature(raw_v1, {"numThreads": 2}, exe_v1, conv_v1)
            prods["manifest"].write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "signature": sig,
                        "artifacts": {k: ukf.digest(p) for k, p in prods.items() if k != "manifest"},
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(ukf.current(raw_v1, out_v1, {"numThreads": 8}, exe_v1, conv_v1))

            # Relocate BOTH subject folder AND Slicer install directory -> MUST reuse!
            sub_v2 = base / "loc2" / "sub01"
            sub_v2.parent.mkdir()
            shutil.move(str(sub_v1), str(sub_v2))
            inst_v2 = base / "slicer_v2" / "cli-modules"
            inst_v2.parent.mkdir()
            shutil.move(str(inst_v1), str(inst_v2))
            raw_v2 = RawInputBundle(
                sub_v2,
                "sub01",
                sub_v2 / "input",
                sub_v2 / "input" / "data.nii.gz",
                sub_v2 / "input" / "bvals",
                sub_v2 / "input" / "bvecs",
                sub_v2 / "input" / "mask.nii.gz",
            )
            self.assertTrue(
                ukf.current(raw_v2, sub_v2 / "ukf_out", {"numThreads": 16}, inst_v2 / "UKFTractography", inst_v2 / "DWIConvert")
            )

            # Changing scientific tracking parameter MUST invalidate!
            self.assertFalse(
                ukf.current(
                    raw_v2,
                    sub_v2 / "ukf_out",
                    {"seedingThreshold": 0.3},
                    inst_v2 / "UKFTractography",
                    inst_v2 / "DWIConvert",
                )
            )
            # Changing companion shared library bytes MUST invalidate!
            (inst_v2 / "libUKFBase.dylib").write_bytes(b"ukf-lib-v2-changed")
            self.assertFalse(
                ukf.current(raw_v2, sub_v2 / "ukf_out", {"numThreads": 8}, inst_v2 / "UKFTractography", inst_v2 / "DWIConvert")
            )

    def test_saf_reuses_when_fs_dir_mtime_changes_and_rejects_surface_edit(self) -> None:
        from shared.orchestrator import _saf_complete
        from tractography.saf import _file_info, _fs_surface_inputs
        from tractography.saf_backend.provenance import implementation_record

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            fs_dir = root / "fs"
            (fs_dir / "surf").mkdir(parents=True)
            (fs_dir / "mri").mkdir()
            for hemi in ("lh", "rh"):
                for kind in ("white", "pial", "inflated", "sphere", "curv"):
                    (fs_dir / "surf" / f"{hemi}.{kind}").write_text(f"{hemi}.{kind}", encoding="utf-8")
            (fs_dir / "mri" / "brain.mgz").write_text("brain", encoding="utf-8")
            fod = root / "wmfod.mif"
            fod.write_bytes(b"fod-bytes")

            out = root / "saf_out"
            out.mkdir()
            (out / "SAF.tck").write_bytes(b"tck")
            (out / "SAF.vtk").write_bytes(b"vtk")
            (out / "saf_qc.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "implementation": implementation_record(),
                        "streamline_count": 10,
                        "scientific_output_summary": {
                            "streamline_count": 10,
                            "all_points_finite": True,
                            "length_mm": {"max": 35.0},
                        },
                    }
                ),
                encoding="utf-8",
            )
            (out / "saf_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "implementation": implementation_record(),
                        "parameters": {"maxlen": 40.0},
                        "inputs": {
                            "fod": _file_info(fod),
                            "fs_dir": _file_info(fs_dir),
                            "fs_surface_inputs": _fs_surface_inputs(fs_dir),
                        },
                    }
                ),
                encoding="utf-8",
            )

            with mock.patch("shared.orchestrator.tractography_readable", return_value=True):
                self.assertTrue(_saf_complete(out, fod_path=fod, fs_dir=fs_dir))
                # Creating a log file inside fs_dir mutates fs_dir's directory mtime_ns -> MUST still reuse!
                (fs_dir / "backend.log").write_text("log line\n", encoding="utf-8")
                _bump_mtime(fs_dir)
                self.assertTrue(_saf_complete(out, fod_path=fod, fs_dir=fs_dir))

                # Modifying a core surface file inside fs_dir MUST invalidate!
                (fs_dir / "surf" / "lh.white").write_text("modified-surface", encoding="utf-8")
                self.assertFalse(_saf_complete(out, fod_path=fod, fs_dir=fs_dir))

    def test_wmql_and_atlas_reuse_across_query_relocation_and_code_edits(self) -> None:
        from shared.orchestrator import _wmql_complete
        from tractography.wmql import (
            WMQL_QC_VERSION,
            bundled_wmql_query_path,
            wmql_query_dependencies,
        )
        from tractography.wmql_atlas import (
            IMPLEMENTATION_VERSION,
            prepare_ddparcel_atlas,
            prepared_atlas_current,
        )
        from test_wmql_atlas import image as make_wmql_atlas_image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            # 1. wmql_atlas uses semantic IMPLEMENTATION_VERSION (independent of `.py` bytes).
            src_atlas = root / "wmparc.nii.gz"
            nib.save(make_wmql_atlas_image(), src_atlas)
            out_atlas, rec = prepare_ddparcel_atlas(src_atlas, root / "atlas_cache")
            self.assertEqual(rec["implementation_sha256"], IMPLEMENTATION_VERSION)
            self.assertTrue(prepared_atlas_current(rec, src_atlas))
            self.assertFalse(prepared_atlas_current(dict(rec, implementation_sha256="old"), src_atlas))

            # 2. _wmql_complete reuses bundled preset even if recorded query_path was in an old extension dir.
            wmql_dir = root / "wmql_out"
            bundled = bundled_wmql_query_path("official")
            deps = wmql_query_dependencies(bundled, [bundled.parent])
            snap_dir = wmql_dir / "queries" / "snap1"
            snap_dir.mkdir(parents=True)
            for src_path in deps:
                shutil.copy2(src_path, snap_dir / Path(src_path).name)
            staged = snap_dir / bundled.name
            snap_deps = wmql_query_dependencies(staged, [snap_dir])

            # Rewrite recorded source query paths to a nonexistent old extension install directory.
            old_deps = {f"/old/ext/wmql_queries/{Path(p).name}": d for p, d in deps.items()}
            (wmql_dir / "wmql_manifest.json").write_text(
                json.dumps(
                    {
                        "subject_id": "sub01",
                        "atlas_mode": "native",
                        "query_preset": "official",
                        "query_selection": [],
                        "query_path": f"/old/ext/wmql_queries/{bundled.name}",
                        "query_sha256": _sha256(bundled),
                        "query_include_dirs": ["/old/ext/wmql_queries"],
                        "query_dependencies": old_deps,
                        "staged_query": f"/old/subject/wmql_out/queries/snap1/{bundled.name}",
                        "query_snapshot_dependencies": {
                            f"/old/subject/wmql_out/queries/snap1/{Path(p).name}": d for p, d in snap_deps.items()
                        },
                    }
                ),
                encoding="utf-8",
            )
            bundle_vtk = wmql_dir / "sub01_desc-wmql_AF.left.vtk"
            bundle_vtk.write_text(
                "# vtk DataFile Version 3.0\nfixture\nASCII\nDATASET POLYDATA\n"
                "POINTS 2 float\n0 0 0 1 0 0\nLINES 1 3\n2 0 1\n",
                encoding="utf-8",
            )
            (wmql_dir / "wmql_qc.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "version": WMQL_QC_VERSION,
                        "official_multibundle": True,
                        "execution_passed": True,
                        "output_integrity_passed": True,
                        "automatic_qc_passed": True,
                        "error": None,
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(_wmql_complete(wmql_dir / "wmql_qc.json"))

            # Tampering staged FreeSurfer.qry MUST invalidate!
            (snap_dir / "FreeSurfer.qry").write_text("Brain_Stem |= 99999\n", encoding="utf-8")
            self.assertFalse(_wmql_complete(wmql_dir / "wmql_qc.json"))

    def test_wma_and_tractcloud_reuse_across_exec_knobs_and_dir_moves(self) -> None:
        from shared.orchestrator import ProductOrchestratorConfig, _tractcloud_complete, _wma_complete
        from tractography.tractcloud import (
            ADAPTER_REVISION,
            TRACTCLOUD_QC_VERSION,
            _file_info as tc_file_info,
            model_input_signature,
        )
        from tractography.wma import _file_info as wma_file_info
        from vtp_fixture import vtp_lines

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            # 1. TractCloud: ignores batch_size, device, tractcloud_root path, and data_dir relocation.
            data_v1 = root / "tc_data_v1"
            (data_v1 / "TrainedModel").mkdir(parents=True)
            (data_v1 / "TrainData_800clu800ol").mkdir(parents=True)
            (data_v1 / "TrainedModel" / "best_tract_f1_model.pth").write_bytes(b"weights-v1")
            (data_v1 / "TrainedModel" / "cli_args.txt").write_text("--seed 0\n", encoding="utf-8")
            (data_v1 / "TrainData_800clu800ol" / "HCP_mass_center.npy").write_bytes(b"centers-v1")

            tract = root / "wholebrain.vtp"
            tract.write_text(vtp_lines(5), encoding="utf-8")
            tc_out = root / "tc_out"
            (tc_out / "Association").mkdir(parents=True)
            (tc_out / "Association" / "AF.vtp").write_text(vtp_lines(5), encoding="utf-8")
            (tc_out / "tractcloud_qc.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "version": TRACTCLOUD_QC_VERSION,
                        "adapter_revision": ADAPTER_REVISION,
                        "executed": True,
                        "automatic_qc_passed": True,
                        "output_summary": {"bundle_count": 1, "bundle_readable_count": 1, "bundle_nonempty_count": 1, "assigned_streamline_count": 5},
                    }
                ),
                encoding="utf-8",
            )
            (tc_out / "tractcloud_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "executed": True,
                        "random_seed": 0,
                        "adapter_revision": ADAPTER_REVISION,
                        "batch_size": 1024,
                        "device": "cuda:0",
                        "tractcloud_root": "/old/install/TractCloud",
                        "dependencies": {
                            "input_tractography": tc_file_info(tract),
                            "model_inputs": model_input_signature(data_v1),
                        },
                    }
                ),
                encoding="utf-8",
            )

            data_v2 = root / "tc_data_v2"
            shutil.move(str(data_v1), str(data_v2))
            _bump_mtime(data_v2 / "TrainedModel" / "best_tract_f1_model.pth")
            cfg = ProductOrchestratorConfig(
                subject_id="sub01",
                subject_root=root,
                targets=("tractcloud",),
                tractcloud_batch_size=4096,
                tractcloud_device="cpu",
                tractcloud_root=root / "new_TractCloud",
                tractcloud_data_dir=data_v2,
            )
            self.assertTrue(_tractcloud_complete(tc_out, input_tractography=tract, random_seed=0, config=cfg))
            # Different random seed MUST invalidate!
            self.assertFalse(_tractcloud_complete(tc_out, input_tractography=tract, random_seed=42, config=cfg))
            # Modified model weight bytes MUST invalidate!
            (data_v2 / "TrainedModel" / "best_tract_f1_model.pth").write_bytes(b"weights-v2-changed")
            self.assertFalse(_tractcloud_complete(tc_out, input_tractography=tract, random_seed=0, config=cfg))

            # 2. WMA: survives atlas_dir mtime change and relocated prepared_input_tractography.
            from tractography.wma import WMA_QC_VERSION, summarize_wma_outputs

            atlas_dir = root / "ORG-Atlases"
            (atlas_dir / "ORG-RegAtlas-100HCP").mkdir(parents=True)
            (atlas_dir / "ORG-800FC-100HCP").mkdir(parents=True)
            wma_out = root / "wma_out"
            tracts_dir = wma_out / "wholebrain" / "AnatomicalTracts"
            tracts_dir.mkdir(parents=True)
            for idx in range(1, 74):
                name = {1: "T_AF_left", 2: "T_AF_right", 3: "T_CST_left", 4: "T_CST_right", 5: "T_CC4"}.get(
                    idx, f"T_{idx:03d}"
                )
                (tracts_dir / f"{name}.vtp").write_text(vtp_lines(20), encoding="utf-8")
            wma_summary = summarize_wma_outputs(wma_out, tract, registration_mode="rig")
            (wma_out / "wma_org_atlas_qc.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "version": WMA_QC_VERSION,
                        "automatic_qc_passed": True,
                        "output_summary": wma_summary,
                    }
                ),
                encoding="utf-8",
            )
            (wma_out / "wma_org_atlas_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "executed": True,
                        "atlas_family": "org",
                        "execution_backend": "native",
                        "registration_mode": "rig",
                        "prepared_input_tractography": "/old/moved/path/wholebrain.vtp",
                        "dependencies": {
                            "input_tractography": wma_file_info(tract),
                            "atlas_dir": wma_file_info(atlas_dir),
                        },
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.dict(os.environ, {"DSP_WMA_EXECUTION_BACKEND": "native"}):
                (atlas_dir / "touch.txt").write_text("x", encoding="utf-8")
                _bump_mtime(atlas_dir)
                self.assertTrue(_wma_complete(wma_out, input_tractography=tract, atlas_dir=atlas_dir))
                # Modifying input tractogram MUST invalidate!
                tract.write_text(vtp_lines(9), encoding="utf-8")
                self.assertFalse(_wma_complete(wma_out, input_tractography=tract, atlas_dir=atlas_dir))


if __name__ == "__main__":
    unittest.main()
