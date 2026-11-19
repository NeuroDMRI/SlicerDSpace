#!/usr/bin/env python3
"""Tests covering the canonical subject output tree redesign."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import nibabel as nib
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class CanonicalOutputRedesignTest(unittest.TestCase):
    def test_obj_surface_loads_in_ras_and_stl_in_lps_and_auto_results_match(self) -> None:
        from shared.auto_results import _role
        from shared.result_scene import result_file_suffix

        self.assertEqual(_role({"product": "surface.ddsurfer", "key": "white_lh", "path": "/x/lh.white.obj"}), "surface:white:lh")
        self.assertEqual(_role({"product": "surface.ddsurfer", "key": "pial_rh", "path": "/x/rh.pial.obj"}), "surface:pial:rh")
        self.assertEqual(result_file_suffix("/x/lh.white.obj"), ".obj")

        calls = []

        class _FakeModelNode:
            def __init__(self) -> None:
                self.name = None
                self.attrs = {}

            def SetName(self, name: str) -> None:
                self.name = name

            def SetAttribute(self, key: str, value: str) -> None:
                self.attrs[key] = value

        def _fake_load_node(path: str, filetype: str, properties=None, returnNode=False):
            node = _FakeModelNode()
            calls.append((Path(path), filetype, dict(properties or {}), node))
            return (True, node) if returnNode else True

        fake_slicer = SimpleNamespace(
            vtkMRMLStorageNode=SimpleNamespace(CoordinateSystemRAS=1, CoordinateSystemLPS=2),
            util=SimpleNamespace(loadNodeFromFile=_fake_load_node, VTK_NODE=None),
            ScriptedLoadableModule=SimpleNamespace(ScriptedLoadableModuleLogic=object),
        )
        with mock.patch.dict("sys.modules", {"slicer": fake_slicer, "slicer.ScriptedLoadableModule": fake_slicer.ScriptedLoadableModule, "qt": SimpleNamespace(), "ctk": SimpleNamespace(), "vtk": SimpleNamespace()}):
            import importlib
            import shared.module_logic as module_logic

            importlib.reload(module_logic)
            logic = module_logic.SharedLogicMixin.__new__(module_logic.SharedLogicMixin)
            with tempfile.TemporaryDirectory() as tmp:
                obj_path = Path(tmp) / "lh.white.obj"
                stl_path = Path(tmp) / "100307_predicted_wm_lh.stl"
                obj_path.write_text("v 0 0 0\n", encoding="utf-8")
                stl_path.write_text("solid s\nendsolid s\n", encoding="utf-8")

                self.assertTrue(logic._load_model_node(obj_path, "surface.ddsurfer", "100307"))
                self.assertTrue(logic._load_model_node(stl_path, "surface.ddsurfer", "100307"))

        self.assertEqual(calls[0][2]["coordinateSystem"], 1)
        self.assertEqual(calls[0][3].attrs["SlicerDDSurfer.SourceCoordinateSystem"], "RAS")
        self.assertEqual(calls[1][2]["coordinateSystem"], 2)
        self.assertEqual(calls[1][3].attrs["SlicerDDSurfer.SourceCoordinateSystem"], "LPS")

    def test_discover_surfaces_reads_flat_surface_morphometry_outputs(self) -> None:
        from shared.region_reports import discover_surfaces

        with tempfile.TemporaryDirectory() as tmp:
            morph_dir = Path(tmp) / "outputs" / "01_anatomical_priors" / "surface_morphometry"
            morph_dir.mkdir(parents=True)
            annot_path = morph_dir / "lh.aparc.annot"
            nib.freesurfer.write_annot(
                str(annot_path),
                np.array([0, 1, 1, 0], dtype=np.int32),
                np.array([[255, 0, 0, 0, 1], [0, 255, 0, 0, 2]], dtype=np.int32),
                ["unknown", "precentral"],
            )
            for name in ("lh.thickness", "lh.area", "lh.curv"):
                (morph_dir / name).write_bytes(b"morph")

            task = {
                "outputs": {
                    "lh.aparc.annot": str(annot_path),
                    "lh.thickness": str(morph_dir / "lh.thickness"),
                    "lh.area": str(morph_dir / "lh.area"),
                    "lh.curv": str(morph_dir / "lh.curv"),
                }
            }
            discovered = discover_surfaces(task, [{"id": 1024, "name": "ctx-lh-precentral"}])
            self.assertIn("1024", discovered)
            self.assertEqual(discovered["1024"]["hemisphere"], "lh")
            self.assertEqual(discovered["1024"]["annotation"], str(annot_path.resolve()))
            self.assertEqual(discovered["1024"]["thickness"], str((morph_dir / "lh.thickness").resolve()))

    def test_write_qc_text_summaries_writes_only_ran_or_reused_tasks(self) -> None:
        from shared.qc_text_summary import write_qc_text_summaries

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "100307"
            qc_root = root / "outputs" / "06_qc"
            dti_qc = root / "cache" / "stages" / "02_microstructure" / "dti_qc.json"
            dti_qc.parent.mkdir(parents=True)
            dti_qc.write_text(json.dumps({"status": "completed", "fa_mean": 0.42}), encoding="utf-8")

            tasks = [
                SimpleNamespace(
                    name="segmentation.ddseg",
                    status="completed",
                    reason="",
                    qc_path=None,
                    outputs={"segmentation": str(root / "outputs/01_anatomical_priors/tissue_segmentation/SegmentationMap_GMWMCSF.nii.gz")},
                ),
                SimpleNamespace(
                    name="microstructure.dti",
                    status="skipped",
                    reason="already completed",
                    qc_path=dti_qc,
                    outputs={"fa": str(root / "outputs/02_microstructure/diffusion_tensor/100307_FA.nii.gz")},
                ),
                SimpleNamespace(name="tractography.csd_act", status="planned", reason="execute=False", qc_path=None, outputs={}),
                SimpleNamespace(name="bundles.wmql", status="blocked", reason="missing dep", qc_path=None, outputs={}),
                SimpleNamespace(name="connectome.streamline_count", status="will_run", reason="", qc_path=None, outputs={}),
            ]
            written = write_qc_text_summaries(qc_root, tasks, subject_id="100307", run_name="test_run", subject_root=root)

            self.assertEqual(set(written), {"segmentation.ddseg", "microstructure.dti"})
            self.assertTrue((qc_root / "01_anatomical_priors" / "tissue_segmentation.txt").is_file())
            self.assertTrue((qc_root / "02_microstructure" / "diffusion_tensor.txt").is_file())
            self.assertIn("fa_mean", (qc_root / "02_microstructure" / "diffusion_tensor.txt").read_text(encoding="utf-8"))
            self.assertFalse((qc_root / "03_tractography").exists())
            self.assertFalse((qc_root / "04_bundles").exists())
            self.assertFalse((qc_root / "05_connectome").exists())

    def test_orchestrate_emits_flat_symlink_free_deliverables_and_isolates_cache(self) -> None:
        from shared import ProductOrchestratorConfig, run_subject_products, subject_layout
        from test_product_orchestrator import _write_valid_raw_subject

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "100307"
            _write_valid_raw_subject(root)
            layout = subject_layout(root)
            stage_dti = layout.stage_cache("02_microstructure/dti")
            stage_dti.mkdir(parents=True)
            raw = SimpleNamespace(subject_id="100307", subject_root=root)
            names = ("fa", "md", "ad", "rd", "l1", "l2", "l3", "trace")
            made = {}
            for name in names:
                p = stage_dti / f"{name}.nii.gz"
                p.write_bytes(name.encode("utf-8"))
                made[name] = p
            dec_p = stage_dti / "100307-dti-DEC.nrrd"
            dec_p.write_bytes(b"dec")
            result_obj = SimpleNamespace(
                **{f"{name}_path": made[name] for name in names},
                dec_path=dec_p,
                fwf_path=None, fw_tissue_fa_path=None, fw_tissue_md_path=None, fw_tissue_valid_mask_path=None, free_water_qc_path=None,
                noddi_ndi_path=None, noddi_odi_path=None, noddi_fwf_path=None, noddi_tissue_valid_mask_path=None,
                noddi_rmse_path=None, noddi_nrmse_path=None, noddi_qc_path=None,
                roi_stats_csv_path=None, roi_stats_json_path=None,
            )

            with mock.patch("shared.pipeline.find_subject_root", return_value=raw), \
                    mock.patch("shared.pipeline.resolve_preprocessed_input_bundle", return_value=raw), \
                    mock.patch("shared.pipeline.ensure_native_dti_bundle", return_value=SimpleNamespace()), \
                    mock.patch("microstructure.run_microstructure", return_value=result_obj):
                run_subject_products(ProductOrchestratorConfig("100307", root, ("microstructure",), execute=True, resume=False))

            fa_out = layout.diffusion_tensor / "100307_FA.nii.gz"
            dec_out = layout.diffusion_tensor / "100307_DEC.nrrd"
            qc_txt = layout.qc / "02_microstructure" / "diffusion_tensor.txt"
            self.assertTrue(fa_out.is_file())
            self.assertTrue(dec_out.is_file())
            self.assertTrue(qc_txt.is_file())
            self.assertFalse(any(p.is_symlink() for p in layout.outputs.rglob("*")))
            self.assertFalse(layout.cache_reports.exists())
            self.assertFalse((layout.outputs / "06_slicer_scene_report").exists())

    def test_migrate_subject_output_tree_converts_legacy_tree_without_symlinks(self) -> None:
        from shared import migrate_subject_output_tree, subject_layout

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "100307"
            out = root / "outputs"
            preproc = out / "01_preproc"
            ddseg = out / "03_structural" / "ddseg"
            ddsurfer = out / "03_structural" / "ddsurfer" / "pred_results_SlicerDSpace" / "native" / "100307"
            dti = out / "02_microstructure" / "dti"
            act = out / "04_tractography" / "tractograms" / "csd_act"
            conn = out / "04_tractography" / "connectome" / "csd_act"
            report = out / "06_slicer_scene_report"
            for d in (preproc, ddseg, ddsurfer, dti, act, conn, report):
                d.mkdir(parents=True)
            (preproc / "100307_dwi_final.nii.gz").write_bytes(b"dwi")
            (preproc / "100307_dwi_final.bval").write_text("0\n", encoding="utf-8")
            (preproc / "100307_dwi_final.bvec").write_text("0\n0\n0\n", encoding="utf-8")
            (preproc / "100307_brain_mask_final.nii.gz").write_bytes(b"mask")
            (ddseg / "SegmentationMap_GMWMCSF.nii.gz").write_bytes(b"seg")
            (ddsurfer / "lh.white.obj").write_text("v 0 0 0\n", encoding="utf-8")
            (dti / "100307_dti_FA.nii.gz").write_bytes(b"fa")
            (act / "100307_csd_act_ifod2.tck").write_bytes(b"tck")
            (conn / "100307-connectome-streamline_count.csv").write_text("0,1\n1,0\n", encoding="utf-8")
            (conn / "100307-connectome-nodes.tsv").write_text("1\tLeft\n", encoding="utf-8")
            (report / "report.html").write_text("<html></html>", encoding="utf-8")

            migrate_subject_output_tree(root, clean_legacy=True)
            layout = subject_layout(root)

            self.assertTrue((layout.preproc / "100307_dwi_final.nii.gz").is_file())
            self.assertTrue((layout.tissue_segmentation / "SegmentationMap_GMWMCSF.nii.gz").is_file())
            self.assertTrue((layout.cortical_surface / "lh.white.obj").is_file())
            self.assertTrue((layout.diffusion_tensor / "100307_FA.nii.gz").is_file())
            self.assertTrue((layout.anatomically_constrained_tractography / "100307_act.tck").is_file())
            self.assertTrue((layout.connectome / "100307_streamline_count.csv").is_file())
            self.assertTrue((layout.connectome / "100307_nodes.tsv").is_file())
            self.assertTrue((layout.cache_reports / "report.html").is_file())
            self.assertFalse((out / "01_preproc").exists())
            self.assertFalse((out / "03_structural").exists())
            self.assertFalse((out / "04_tractography").exists())
            self.assertFalse((out / "06_slicer_scene_report").exists())
            self.assertFalse(any(p.is_symlink() for p in layout.outputs.rglob("*")))


if __name__ == "__main__":
    unittest.main()
