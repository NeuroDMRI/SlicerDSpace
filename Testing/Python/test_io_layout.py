#!/usr/bin/env python3
"""Tests for the canonical subject I/O layout."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class IOLayoutTest(unittest.TestCase):
    def test_default_task_dirs_match_product_tree(self) -> None:
        from shared import default_task_output_dir, subject_layout

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            layout = subject_layout(subject_root)

            self.assertEqual(default_task_output_dir(subject_root, "preproc"), layout.subject_root / "preprocessed")
            self.assertEqual(default_task_output_dir(subject_root, "microstructure"), layout.outputs / "02_microstructure")
            self.assertEqual(
                default_task_output_dir(subject_root, "ddseg"),
                layout.outputs / "01_anatomical_priors" / "tissue_segmentation",
            )
            self.assertEqual(
                default_task_output_dir(subject_root, "ddsurfer"),
                layout.outputs / "01_anatomical_priors" / "cortical_surface",
            )
            self.assertEqual(
                default_task_output_dir(subject_root, "ddparcel"),
                layout.outputs / "01_anatomical_priors" / "brain_parcellation",
            )
            self.assertEqual(
                default_task_output_dir(subject_root, "surface_morphometry"),
                layout.outputs / "01_anatomical_priors" / "surface_morphometry",
            )
            self.assertEqual(
                default_task_output_dir(subject_root, "tractography_dti"),
                layout.outputs / "03_tractography" / "single_tensor_tractography",
            )
            self.assertEqual(
                default_task_output_dir(subject_root, "saf"),
                layout.outputs / "03_tractography" / "short_association_fibers",
            )
            self.assertEqual(default_task_output_dir(subject_root, "connectome"), layout.outputs / "05_connectome")
            self.assertEqual(default_task_output_dir(subject_root, "qc"), layout.outputs / "06_qc")
            self.assertEqual(default_task_output_dir(subject_root, "regional_microstructure"), layout.outputs / "02_microstructure")

    def test_dependency_path_can_infer_same_subject_output_tree(self) -> None:
        from shared import default_task_output_dir_for_dependency

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            tractogram = (
                subject_root
                / "outputs"
                / "03_tractography"
                / "single_tensor_tractography"
                / "100307_dti.trk"
            )

            self.assertEqual(
                default_task_output_dir_for_dependency(tractogram, "connectome"),
                subject_root.resolve() / "outputs" / "05_connectome",
            )
            self.assertEqual(
                default_task_output_dir_for_dependency(tractogram, "tractcloud"),
                subject_root.resolve() / "outputs" / "04_bundles" / "bundle_identification",
            )
            self.assertEqual(
                default_task_output_dir_for_dependency(tractogram, "wmql"),
                subject_root.resolve() / "outputs" / "04_bundles" / "query_bundle_extraction",
            )

    def test_legacy_outputs_can_be_materialized_under_canonical_tree(self) -> None:
        from shared import materialize_canonical_output_tree, subject_layout

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            legacy_preproc = subject_root / "outputs" / "01_preproc"
            legacy_dti = subject_root / "outputs" / "02_microstructure" / "dti"
            legacy_ddparcel = subject_root / "outputs" / "03_structural" / "ddparcel"
            legacy_tract = subject_root / "outputs" / "04_tractography" / "tractograms" / "dti_det"
            legacy_tractcloud = subject_root / "outputs" / "04_tractography" / "bundles" / "tract_cloud_dl" / "AnatomicalTracts"
            legacy_report = subject_root / "outputs" / "06_slicer_scene_report"
            for folder in (legacy_preproc, legacy_dti, legacy_ddparcel, legacy_tract, legacy_tractcloud, legacy_report):
                folder.mkdir(parents=True)
            (legacy_preproc / "100307_dwi_final.nii.gz").write_text("dwi", encoding="utf-8")
            (legacy_dti / "100307_dti_FA.nii.gz").write_text("fa", encoding="utf-8")
            (legacy_ddparcel / "100307-DDSurfer-wmparc.nii.gz").write_text("wmparc", encoding="utf-8")
            (legacy_tract / "100307-dti-deterministic.trk").write_text("trk", encoding="utf-8")
            (legacy_tractcloud / "AF_left.vtp").write_text("tract", encoding="utf-8")
            (legacy_report / "report.html").write_text("<html></html>", encoding="utf-8")

            created = materialize_canonical_output_tree(subject_root, mode="copy", archive_legacy=True)
            layout = subject_layout(subject_root)

            self.assertTrue((layout.preproc / "100307_dwi_final.nii.gz").is_file())
            self.assertTrue((layout.diffusion_tensor / "100307_FA.nii.gz").is_file())
            self.assertTrue((layout.brain_parcellation / "100307_parcellation.nii.gz").is_file())
            self.assertTrue((layout.single_tensor_tractography / "100307_dti.trk").is_file())
            self.assertTrue((layout.bundle_identification / "AF_left.vtp").is_file())
            self.assertTrue((layout.cache_reports / "report.html").is_file())
            self.assertFalse((subject_root / "outputs" / "03_structural").exists())
            self.assertFalse((subject_root / "outputs" / "04_tractography").exists())
            self.assertFalse((subject_root / "outputs" / "06_slicer_scene_report").exists())
            self.assertFalse(any(path.is_symlink() for path in layout.outputs.rglob("*")))
            self.assertTrue(created)

    def test_legacy_freewater_noddi_and_stats_migrate_without_symlinks(self) -> None:
        from shared import materialize_canonical_output_tree, subject_layout

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            layout = subject_layout(subject_root)
            legacy_fw = layout.outputs / "02_microstructure" / "freewater"
            legacy_noddi = layout.outputs / "02_microstructure" / "noddi"
            legacy_stats = layout.outputs / "05_statistics" / "regional_microstructure"
            for folder in (legacy_fw, legacy_noddi, legacy_stats, legacy_noddi / "_amico_noddi"):
                folder.mkdir(parents=True)
            (legacy_fw / "100307_fw_fraction.nii.gz").write_text("fwf", encoding="utf-8")
            (legacy_noddi / "100307_noddi_NDI.nii.gz").write_text("ndi", encoding="utf-8")
            (legacy_noddi / "_amico_noddi" / "fit_NDI.nii.gz").write_text("fit", encoding="utf-8")
            (legacy_stats / "100307_ROIWise-Microstructure.csv").write_text("roi,val\n1,0.5\n", encoding="utf-8")

            materialize_canonical_output_tree(subject_root, mode="copy", archive_legacy=True)

            self.assertTrue((layout.free_water / "100307_FW.nii.gz").is_file())
            self.assertTrue((layout.neurite_microstructure / "100307_NDI.nii.gz").is_file())
            self.assertTrue((layout.microstructure / "100307_ROIWise-Microstructure.csv").is_file())
            self.assertTrue(
                (layout.stage_cache("02_microstructure/noddi/_amico_noddi") / "fit_NDI.nii.gz").is_file()
            )
            self.assertFalse((layout.outputs / "05_statistics").exists())
            self.assertFalse((layout.outputs / "02_microstructure" / "freewater").exists())
            self.assertFalse((layout.outputs / "02_microstructure" / "noddi").exists())

    def test_development_outputs_can_be_archived_outside_canonical_tree(self) -> None:
        from shared import archive_development_outputs, subject_layout

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            layout = subject_layout(subject_root)
            probe_dir = layout.bundles / "wmql_probe_af_cst"
            check_dir = layout.connectome / "csd_act_mean_fa_check"
            saf_dir = layout.tractography / "saf_python_execute_fastsurfer"
            smoke_dir = layout.outputs / "connectome_execute_smoke"
            for path in (probe_dir, check_dir, saf_dir, smoke_dir):
                path.mkdir(parents=True)
                (path / "marker.txt").write_text("x", encoding="utf-8")

            moved = archive_development_outputs(subject_root)

            self.assertFalse(probe_dir.exists())
            self.assertFalse(check_dir.exists())
            self.assertFalse(saf_dir.exists())
            self.assertFalse(smoke_dir.exists())
            self.assertTrue((layout.cache / "legacy_outputs" / "04_bundles" / "wmql_probe_af_cst").is_dir())
            self.assertTrue((layout.cache / "legacy_outputs" / "05_connectome" / "csd_act_mean_fa_check").is_dir())
            self.assertTrue((layout.cache / "legacy_outputs" / "03_tractography" / "saf_python_execute_fastsurfer").is_dir())
            self.assertTrue((layout.cache / "legacy_outputs" / "connectome_execute_smoke").is_dir())
            self.assertGreaterEqual(len(moved), 4)

    def test_alternate_tractogram_backend_runs_move_to_cache(self) -> None:
        from shared import archive_development_outputs, subject_layout

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            layout = subject_layout(subject_root)
            main_dir = layout.tractograms / "csd_act"
            alternate_dir = layout.tractograms / "dti_det"
            main_dir.mkdir(parents=True)
            alternate_dir.mkdir(parents=True)
            (main_dir / "main.tck").write_text("main", encoding="utf-8")
            (alternate_dir / "alt.trk").write_text("alt", encoding="utf-8")

            moved = archive_development_outputs(
                subject_root,
                current_run_name="csd_act",
                current_tractography_backend="csd_act",
            )

            cached = layout.cache / "alternate_runs" / "tractograms" / "dti_det"
            self.assertTrue(main_dir.is_dir())
            self.assertFalse(alternate_dir.exists())
            self.assertTrue((cached / "alt.trk").is_file())
            self.assertIn(alternate_dir.as_posix(), moved)

    def test_partial_run_cleanup_preserves_existing_tractogram_runs(self) -> None:
        from shared import archive_development_outputs, subject_layout

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            layout = subject_layout(subject_root)
            existing_run = layout.tractograms / "windows_full"
            existing_run.mkdir(parents=True)
            tractogram = existing_run / "100307-dti-deterministic.trk"
            tractogram.write_text("tractogram", encoding="utf-8")

            moved = archive_development_outputs(
                subject_root,
                current_run_name="segmentation_only",
                current_tractography_backend="dti_det",
                archive_alternate_tractograms=False,
            )

            self.assertTrue(tractogram.is_file())
            self.assertNotIn(existing_run.as_posix(), moved)

    def test_legacy_wma_runs_can_be_materialized_under_canonical_bundles_tree(self) -> None:
        from shared import materialize_canonical_output_tree, subject_layout

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "149741"
            legacy_wma = (
                subject_root
                / "outputs"
                / "04_tractography"
                / "bundles"
                / "wma_org_atlas"
                / "wma_org_execute"
                / "AnatomicalTracts"
            )
            legacy_wma.mkdir(parents=True)
            (legacy_wma / "AF_left.vtp").write_text("vtp", encoding="utf-8")
            (legacy_wma.parent / "wma_org_atlas_qc.json").write_text("{}", encoding="utf-8")

            created = materialize_canonical_output_tree(subject_root, mode="copy", archive_legacy=True)
            layout = subject_layout(subject_root)

            self.assertTrue((layout.fiber_clustering / "AF_left.vtp").is_file())
            self.assertTrue(
                (layout.stage_cache("04_bundles/wma_org_atlas") / "wma_org_execute" / "wma_org_atlas_qc.json").is_file()
            )
            self.assertIn("wma:AF_left.vtp", created)

    def test_legacy_connectome_metric_dirs_can_be_migrated_under_canonical_connectome_and_cache(self) -> None:
        from shared import migrate_subject_output_tree, subject_layout

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp).resolve() / "100307"
            legacy_metric_dir = subject_root / "outputs" / "04_tractography" / "connectome" / "csd_act_mean_fa"
            legacy_metric_dir.mkdir(parents=True)
            (legacy_metric_dir / "100307-connectome-mean_fa.csv").write_text("a,b\n1,2\n", encoding="utf-8")
            (legacy_metric_dir / "100307-connectome-mean_fa_qc.json").write_text("{}", encoding="utf-8")

            migrate_subject_output_tree(subject_root, clean_legacy=True)
            layout = subject_layout(subject_root)

            self.assertTrue((layout.connectome / "100307_mean_fa.csv").is_file())
            self.assertTrue(
                (layout.stage_cache("05_connectome") / "csd_act" / "mean_fa" / "100307-connectome-mean_fa_qc.json").is_file()
            )
            self.assertFalse(legacy_metric_dir.exists())

    def test_legacy_connectome_root_qc_files_can_be_archived(self) -> None:
        from shared import archive_development_outputs, subject_layout

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            layout = subject_layout(subject_root)
            root_file = layout.connectome / "100307-connectome-streamline_count_qc.json"
            root_file.parent.mkdir(parents=True)
            root_file.write_text("{}", encoding="utf-8")

            moved = archive_development_outputs(subject_root)

            self.assertFalse(root_file.exists())
            self.assertTrue(
                (layout.cache / "legacy_outputs" / "05_connectome" / "100307-connectome-streamline_count_qc.json").is_file()
            )
            self.assertIn(root_file.as_posix(), moved)

    def test_scheme_a_external_output_colocates_workspace_and_preserves_cache(self) -> None:
        from shared import archive_development_outputs, default_task_output_dir, subject_layout

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            readonly_subj = root / "readonly_disk" / "sub-01"
            (readonly_subj / "input").mkdir(parents=True)
            out_dir = root / "user_out" / "sub-01" / "outputs"

            layout = subject_layout(readonly_subj, output_root=out_dir)
            self.assertEqual(layout.workspace_root, root / "user_out" / "sub-01")
            self.assertEqual(layout.outputs, out_dir)
            self.assertEqual(layout.cache, root / "user_out" / "sub-01" / "cache")
            self.assertEqual(layout.preproc, root / "user_out" / "sub-01" / "preprocessed")
            self.assertEqual(
                default_task_output_dir(readonly_subj, "preproc", output_root=out_dir),
                root / "user_out" / "sub-01" / "preprocessed",
            )
            self.assertEqual(
                default_task_output_dir(readonly_subj, "connectome", output_root=out_dir),
                out_dir / "05_connectome",
            )

            # Custom non-"outputs"-named output_root keeps cache/preproc alongside it and never archives them.
            custom_out = root / "custom_out"
            custom_layout = subject_layout(readonly_subj, output_root=custom_out)
            self.assertEqual(custom_layout.workspace_root, custom_out)
            self.assertEqual(custom_layout.cache, custom_out / "cache")
            self.assertEqual(custom_layout.preproc, custom_out / "preprocessed")
            (custom_layout.cache / "marker.txt").parent.mkdir(parents=True, exist_ok=True)
            (custom_layout.cache / "marker.txt").write_text("cached", encoding="utf-8")
            (custom_layout.preproc / "preproc.txt").parent.mkdir(parents=True, exist_ok=True)
            (custom_layout.preproc / "preproc.txt").write_text("preproc", encoding="utf-8")
            (custom_out / "input").mkdir(parents=True, exist_ok=True)
            (custom_out / "input" / "in.txt").write_text("in", encoding="utf-8")

            archived = archive_development_outputs(readonly_subj, output_root=custom_out)
            self.assertEqual(archived, {})
            self.assertTrue((custom_layout.cache / "marker.txt").is_file())
            self.assertTrue((custom_layout.preproc / "preproc.txt").is_file())
            self.assertTrue((custom_out / "input" / "in.txt").is_file())

    def test_external_output_never_writes_to_readonly_source_disk(self) -> None:
        import os
        from unittest.mock import patch
        import nibabel as nib
        import numpy as np
        from shared.convertible_input import prepare_convertible_root
        from shared.diffusion_io import ensure_cached_uncompressed_nifti, subject_cache_root
        from shared.io_layout import subject_layout
        from shared.pipeline import find_subject_root

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            src_canon = root / "src_canonical" / "sub-01"
            (src_canon / "input").mkdir(parents=True)
            dwi_path = src_canon / "input" / "data.nii.gz"
            nib.save(nib.Nifti1Image(np.ones((4, 4, 4, 2), dtype=np.float32), np.eye(4)), str(dwi_path))
            (src_canon / "input" / "bvals").write_text("0 1000\n", encoding="utf-8")
            (src_canon / "input" / "bvecs").write_text("0 1\n0 0\n0 0\n", encoding="utf-8")
            before_files = sorted(p.relative_to(src_canon).as_posix() for p in src_canon.rglob("*"))

            out_root = root / "user_out" / "sub-01" / "outputs"
            layout = subject_layout(src_canon, output_root=out_root)

            def fake_mask(dwi, bval, bvec, mask_out, **_kwargs):
                Path(mask_out).parent.mkdir(parents=True, exist_ok=True)
                nib.save(nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.uint8), np.eye(4)), str(mask_out))
                Path(mask_out).with_name("nodif_brain_mask.manifest.json").write_text('{"version": 2}\n', encoding="utf-8")
                return {"status": "completed"}

            with patch("shared.pipeline.generate_brain_mask", side_effect=fake_mask):
                bundle = find_subject_root(src_canon, "sub-01", output_root=out_root)
            self.assertEqual(bundle.mask_path, layout.preproc / "nodif_brain_mask.nii.gz")
            self.assertFalse((layout.cache / "brain_mask").exists())
            self.assertEqual(bundle.effective_cache_root, layout.cache)
            self.assertTrue(bundle.mask_path.is_file())

            with patch.dict(os.environ, {"SLICERDSPACE_UNCOMPRESSED_CACHE_MIN_BYTES": "1"}):
                with subject_cache_root(layout.cache):
                    cached_nii = ensure_cached_uncompressed_nifti(bundle.dwi_path)
            self.assertIsNotNone(cached_nii)
            self.assertTrue(cached_nii.is_relative_to(layout.cache / "uncompressed_nifti"))
            self.assertFalse((src_canon / "cache").exists())
            self.assertFalse((src_canon / "preprocessed").exists())
            self.assertFalse((src_canon / "outputs").exists())
            after_files = sorted(p.relative_to(src_canon).as_posix() for p in src_canon.rglob("*"))
            self.assertEqual(before_files, after_files)

            # Convertible raw NIfTI + bval/bvec also stages into target_root without touching raw dir.
            raw_dir = root / "src_raw" / "sub-02"
            raw_dir.mkdir(parents=True)
            nib.save(nib.Nifti1Image(np.ones((2, 2, 2, 2), dtype=np.float32), np.eye(4)), str(raw_dir / "dwi.nii.gz"))
            (raw_dir / "dwi.bval").write_text("0 1000\n", encoding="utf-8")
            (raw_dir / "dwi.bvec").write_text("0 1\n0 0\n0 0\n", encoding="utf-8")
            raw_before = sorted(p.relative_to(raw_dir).as_posix() for p in raw_dir.rglob("*"))
            dest_ws = root / "user_out" / "sub-02"
            prepared = prepare_convertible_root(raw_dir, "sub-02", target_root=dest_ws)
            self.assertEqual(prepared, dest_ws)
            self.assertTrue((dest_ws / "input" / "data.nii.gz").is_file())
            self.assertEqual(raw_before, sorted(p.relative_to(raw_dir).as_posix() for p in raw_dir.rglob("*")))


if __name__ == "__main__":
    unittest.main()
