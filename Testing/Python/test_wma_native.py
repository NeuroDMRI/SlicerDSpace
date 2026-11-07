"""Exercise native ORG stage ordering and checks without a full tractogram."""

from pathlib import Path
import tempfile
import unittest

from tractography.wma_native import run_org_atlas_native


class WMANativeTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="WMA test ")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / "input" / "case.v1.vtk"
        self.atlas = self.root / "atlas"
        self.reg_atlas = self.atlas / "ORG-RegAtlas-100HCP" / "registration_atlas.vtk"
        self.cluster_atlas = self.atlas / "ORG-800FC-100HCP"
        self.output = self.root / "output"
        self.case = self.output / "case.v1"
        self.registration = self.case / "TractRegistration"
        self.clustering = self.case / "FiberClustering"
        self.separated = self.clustering / "SeparatedClusters"
        self.transformed = self.clustering / "TransformedClusters" / "case.v1"
        self.optional_transform = self.root / "custom scale.tfm"
        self.calls = []
        for path in (self.source, self.reg_atlas, self.optional_transform, self.cluster_atlas / "atlas.p",
                     self.cluster_atlas / "atlas.vtp", self.cluster_atlas / "cluster_hemisphere_location.txt"):
            self.write(path)

    def write(self, path, content="fixture"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def clusters(self, folder):
        for index in range(1, 801):
            self.write(folder / f"cluster_{index:05d}.vtp")

    def simulate_stage(self, stage, argv):
        """Stand in for the WMA/Slicer processes and write their stage products."""
        self.calls.append(tuple(argv))
        script, *args = argv
        if script == "wm_register_to_atlas_new.py":
            _, _, source, _, registration = args
            source = Path(source)
            target = Path(registration) / source.stem
            self.write(target / "output_tractography" / f"{source.stem}_reg.vtk")
            self.write(target / "output_tractography" / f"itk_txform_{source.stem}.tfm")
            self.write(target / "iteration_00001" / "objective.txt")
            self.write(target / "progress.txt")
        elif script == "wm_cluster_from_atlas.py":
            self.clusters(Path(args[4]) / Path(args[2]).stem)
        elif script == "wm_cluster_remove_outliers.py":
            self.clusters(Path(args[4]) / (Path(args[2]).name + "_outlier_removed"))
        elif script == "wm_assess_cluster_location_by_hemisphere.py":
            self.write(Path(args[2]) / "cluster_location_by_hemisphere.log",
                       "<wm_assess_cluster_location_by_hemisphere.py> Done!!!")
        elif script == "run_wma_harden_transform.py":
            source, _, _, target = args
            source = Path(source)
            target = Path(target)
            if source.is_file():
                self.write(target / source.name)
            else:
                for path in source.glob("cluster_*.vtp"):
                    self.write(target / path.name)
        elif script == "wm_separate_clusters_by_hemisphere.py":
            for group in ("tracts_commissural", "tracts_left_hemisphere", "tracts_right_hemisphere"):
                self.clusters(Path(args[1]) / group)
        elif script == "wm_append_clusters_to_anatomical_tracts.py":
            for index in range(72):
                self.write(Path(args[2]) / f"T_named{index}.vtp")
            self.write(Path(args[2]) / "T_UF_right.vtp")
            self.write(Path(args[2]) / "T_MCP.vtp")
        else:
            self.fail(f"Unexpected script for stage {stage}: {script}")

    def run_native(self, *, mode="rig", transform=None, clean=0, run_stage=None):
        run_org_atlas_native(
            input_tractography=self.source,
            output_dir=self.output,
            atlas_dir=self.atlas,
            registration_mode=mode,
            num_threads=3,
            clean_files=clean,
            transform_path=transform,
            run_stage=run_stage or self.simulate_stage,
        )

    def test_rig_calls_fast_rigid_affine_and_keeps_author_stage_order(self):
        self.run_native()
        registered = self.registration / "case.v1" / "output_tractography" / "case.v1_reg.vtk"
        tfm = registered.with_name("itk_txform_case.v1.tfm")
        initial = self.clustering / "InitialClusters"
        outliers = self.clustering / "OutlierRemovedClusters"
        labeled = outliers / "case.v1_reg_outlier_removed"
        self.assertEqual(self.calls, [
            ("wm_register_to_atlas_new.py", "-mode", "rigid_affine_fast",
             str(self.source), str(self.reg_atlas), str(self.registration)),
            ("wm_cluster_from_atlas.py", "-j", "3", str(registered), str(self.cluster_atlas),
             str(initial), "-norender"),
            ("wm_cluster_remove_outliers.py", "-j", "3", str(initial / "case.v1_reg"),
             str(self.cluster_atlas), str(outliers)),
            ("wm_assess_cluster_location_by_hemisphere.py", "-clusterLocationFile",
             str(self.cluster_atlas / "cluster_hemisphere_location.txt"), str(labeled)),
            ("run_wma_harden_transform.py", str(labeled), str(tfm), "1", str(self.transformed)),
            ("wm_separate_clusters_by_hemisphere.py", str(self.transformed), str(self.separated)),
            ("wm_append_clusters_to_anatomical_tracts.py", str(self.separated),
             str(self.cluster_atlas), str(self.case / "AnatomicalTracts")),
        ])
        self.assertTrue((initial / "case.v1_reg" / "cluster_00800.vtp").is_file())

    def test_nonrig_inverts_nonrigid_then_affine_and_undoes_custom_scale_after_splitting(self):
        self.run_native(mode="nonrig", transform=self.optional_transform)
        pretransformed = self.case / "TransformedTracts"
        affine = self.registration / "case.v1" / "output_tractography"
        nonrigid = self.registration / "case.v1_reg" / "output_tractography"
        inverses = self.case / "InvTransformedTracts"
        labeled = self.clustering / "OutlierRemovedClusters" / "case.v1_reg_reg_outlier_removed"
        self.assertEqual(self.calls[:3], [
            ("run_wma_harden_transform.py", str(self.source), str(self.optional_transform),
             "0", str(pretransformed)),
            ("wm_register_to_atlas_new.py", "-mode", "affine", str(pretransformed / self.source.name),
             str(self.reg_atlas), str(self.registration)),
            ("wm_register_to_atlas_new.py", "-mode", "nonrigid", str(affine / "case.v1_reg.vtk"),
             str(self.reg_atlas), str(self.registration)),
        ])
        self.assertEqual(self.calls[3][3], str(nonrigid / "case.v1_reg_reg.vtk"))
        self.assertEqual(self.calls[6:], [
            ("run_wma_harden_transform.py", str(labeled), str(nonrigid / "itk_txform_case.v1_reg.tfm"),
             "1", str(self.transformed / "tmp")),
            ("run_wma_harden_transform.py", str(self.transformed / "tmp"), str(affine / "itk_txform_case.v1.tfm"),
             "1", str(self.transformed)),
            ("wm_separate_clusters_by_hemisphere.py", str(self.transformed), str(self.separated)),
            *[
                ("run_wma_harden_transform.py", str(self.separated / group), str(self.optional_transform),
                 "1", str(inverses / group))
                for group in ("tracts_commissural", "tracts_left_hemisphere", "tracts_right_hemisphere")
            ],
            ("wm_append_clusters_to_anatomical_tracts.py", str(inverses),
             str(self.cluster_atlas), str(self.case / "AnatomicalTracts")),
        ])

    def test_incomplete_split_fails_before_append_or_cleanup_even_if_commissural_is_complete(self):
        def incomplete_split(stage, argv):
            self.simulate_stage(stage, argv)
            if argv[0] == "wm_separate_clusters_by_hemisphere.py":
                right = self.separated / "tracts_right_hemisphere"
                (right / "cluster_00014.vtp").unlink()
                self.write(right / "unrelated.vtp")

        with self.assertRaisesRegex(RuntimeError, "cluster_00014.vtp"):
            self.run_native(clean=1, run_stage=incomplete_split)
        self.assertEqual(self.calls[-1][0], "wm_separate_clusters_by_hemisphere.py")
        self.assertTrue((self.clustering / "InitialClusters").is_dir())

    def test_missing_registration_transform_stops_before_clustering(self):
        def incomplete_registration(stage, argv):
            self.simulate_stage(stage, argv)
            (self.registration / "case.v1" / "output_tractography" / "itk_txform_case.v1.tfm").unlink()

        with self.assertRaisesRegex(RuntimeError, "itk_txform_case.v1.tfm"):
            self.run_native(run_stage=incomplete_registration)
        self.assertEqual(len(self.calls), 1)

    def test_missing_last_anatomical_tract_is_rejected_at_old_count_threshold(self):
        def incomplete_append(stage, argv):
            self.simulate_stage(stage, argv)
            if argv[0] == "wm_append_clusters_to_anatomical_tracts.py":
                (self.case / "AnatomicalTracts" / "T_MCP.vtp").unlink()

        with self.assertRaisesRegex(RuntimeError, "T_MCP.vtp"):
            self.run_native(clean=1, run_stage=incomplete_append)
        self.assertTrue((self.clustering / "InitialClusters").is_dir())

    def test_clean_one_removes_initial_and_transformed_directories(self):
        self.run_native(clean=1)
        self.assertFalse((self.clustering / "InitialClusters").exists())
        self.assertFalse((self.clustering / "TransformedClusters").exists())
        self.assertTrue((self.clustering / "OutlierRemovedClusters").is_dir())
        self.assertTrue((self.separated / "tracts_left_hemisphere" / "cluster_00800.vtp").is_file())

    def test_clean_two_keeps_progress_and_transforms_for_both_registrations(self):
        self.run_native(mode="nonrig", clean=2)
        for name in ("case.v1", "case.v1_reg"):
            with self.subTest(registration=name):
                registration = self.registration / name
                self.assertTrue((registration / "progress.txt").is_file())
                self.assertTrue((registration / "output_tractography" / f"itk_txform_{name}.tfm").is_file())
                self.assertEqual(list(registration.glob("iteration*")), [])
                self.assertEqual(list((registration / "output_tractography").glob("*.vtk")), [])
        for dirname in ("InitialClusters", "OutlierRemovedClusters", "TransformedClusters"):
            self.assertEqual(list((self.clustering / dirname).iterdir()), [])
        self.assertTrue((self.case / "AnatomicalTracts" / "T_UF_right.vtp").is_file())


if __name__ == "__main__":
    unittest.main()
