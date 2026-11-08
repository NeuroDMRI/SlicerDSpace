#!/usr/bin/env python3
"""Lightweight tests for connectome planning."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from connectome import ConnectomeConfig, run_connectome


class ConnectomePipelineTest(unittest.TestCase):
    def test_writes_manifest_matrix_header_and_nodes_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tractography = root / "streamlines.vtk"
            parcellation = root / "wmparc.nii.gz"
            tractography.write_text("# vtk DataFile Version 3.0\n", encoding="utf-8")
            parcellation.write_text("parcel", encoding="utf-8")

            result = run_connectome(
                ConnectomeConfig(
                    subject_id="149741",
                    tractography_path=tractography,
                    parcellation_path=parcellation,
                    output_dir=root / "connectome",
                )
            )

            self.assertFalse(result.executed)
            self.assertEqual(result.matrix_path.read_text(encoding="utf-8").strip(), "source_node,target_node,streamline_count")
            self.assertEqual(result.nodes_path.read_text(encoding="utf-8").strip(), "node_id\tlabel\tname")
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            qc = json.loads(result.qc_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], qc["status"])
            self.assertEqual(manifest["executed"], qc["executed"])
            self.assertEqual(manifest["dependencies"]["tractography"]["path"], tractography.resolve().as_posix())
            self.assertEqual(manifest["dependencies"]["ddparcel_parcellation"]["path"], parcellation.resolve().as_posix())
            self.assertEqual(qc["metric"], "streamline_count")

    def test_manifest_lifecycle_matches_qc_after_success_and_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tractography = root / "streamlines.vtk"
            parcellation = root / "wmparc.nii.gz"
            tractography.write_text("# vtk DataFile Version 3.0\n", encoding="utf-8")
            parcellation.write_text("parcel", encoding="utf-8")

            def _write_mock_outputs(config, matrix_path, nodes_path, *, lut_path):
                matrix_path.write_text("source_node,target_node,streamline_count\n10,20,1\n", encoding="utf-8")
                nodes_path.write_text("node_id\tlabel\tname\n10\t10\tLeft\n20\t20\tRight\n", encoding="utf-8")
                return {"node_count": 2, "edge_count": 1}

            with mock.patch("connectome.pipeline._write_python_connectome", side_effect=_write_mock_outputs):
                completed = run_connectome(
                    ConnectomeConfig(
                        subject_id="success",
                        tractography_path=tractography,
                        parcellation_path=parcellation,
                        output_dir=root / "completed",
                        backend="python",
                        execute=True,
                    )
                )
            completed_manifest = json.loads(completed.manifest_path.read_text(encoding="utf-8"))
            completed_qc = json.loads(completed.qc_path.read_text(encoding="utf-8"))
            self.assertEqual(completed_manifest["status"], completed_qc["status"])
            self.assertEqual(completed_manifest["executed"], completed_qc["executed"])
            self.assertEqual(completed_manifest["status"], "completed")
            self.assertTrue(completed_manifest["executed"])

            missing_dir = root / "missing_outputs"
            with mock.patch("connectome.pipeline._write_python_connectome", return_value={}):
                with self.assertRaisesRegex(RuntimeError, "incomplete outputs"):
                    run_connectome(
                        ConnectomeConfig(
                            subject_id="missing",
                            tractography_path=tractography,
                            parcellation_path=parcellation,
                            output_dir=missing_dir,
                            backend="python",
                            execute=True,
                        )
                    )
            missing_manifest = json.loads(
                (missing_dir / "missing-connectome-streamline_count_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(missing_manifest["status"], "failed")

            header_only_dir = root / "header_only"

            def _write_headers_only(config, matrix_path, nodes_path, *, lut_path):
                matrix_path.write_text("source_node,target_node,streamline_count\n", encoding="utf-8")
                nodes_path.write_text("node_id\tlabel\tname\n", encoding="utf-8")
                return {"node_count": 0, "edge_count": 0}

            with mock.patch("connectome.pipeline._write_python_connectome", side_effect=_write_headers_only):
                with self.assertRaisesRegex(RuntimeError, "incomplete outputs"):
                    run_connectome(
                        ConnectomeConfig(
                            subject_id="headers",
                            tractography_path=tractography,
                            parcellation_path=parcellation,
                            output_dir=header_only_dir,
                            backend="python",
                            execute=True,
                        )
                    )
            header_qc = json.loads(
                (header_only_dir / "headers-connectome-streamline_count_qc.json").read_text(encoding="utf-8")
            )
            self.assertEqual(header_qc["status"], "failed")
            self.assertEqual(header_qc["node_count"], 0)
            self.assertEqual(header_qc["edge_count"], 0)

            failed_dir = root / "failed"
            with mock.patch(
                "connectome.pipeline._write_python_connectome",
                side_effect=RuntimeError("synthetic connectome failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic connectome failure"):
                    run_connectome(
                        ConnectomeConfig(
                            subject_id="failure",
                            tractography_path=tractography,
                            parcellation_path=parcellation,
                            output_dir=failed_dir,
                            backend="python",
                            execute=True,
                        )
                    )
            failed_manifest = json.loads(
                (failed_dir / "failure-connectome-streamline_count_manifest.json").read_text(encoding="utf-8")
            )
            failed_qc = json.loads(
                (failed_dir / "failure-connectome-streamline_count_qc.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failed_manifest["status"], failed_qc["status"])
            self.assertEqual(failed_manifest["executed"], failed_qc["executed"])
            self.assertEqual(failed_manifest["status"], "failed")
            self.assertTrue(failed_manifest["executed"])

    def test_rejects_missing_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                run_connectome(
                    ConnectomeConfig(
                        subject_id="149741",
                        tractography_path=root / "missing.vtk",
                        parcellation_path=root / "missing.nii.gz",
                        output_dir=root / "connectome",
                    )
                )

    def test_execute_true_runs_fake_tck2connectome_for_tck_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            fake = bin_dir / "tck2connectome"
            fake.write_text(
                "#!/bin/sh\n"
                "printf '1,2,3\\n' > \"$3\"\n"
                "exit 0\n",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            if os.name == "nt":
                (bin_dir / "tck2connectome.cmd").write_text(
                    "@echo off\r\n"
                    "(echo 1,2,3)> \"%~3\"\r\n"
                    "exit /b 0\r\n",
                    encoding="utf-8",
                )
            tractography = root / "streamlines.tck"
            parcellation = root / "wmparc.nii.gz"
            tractography.write_text("tck", encoding="utf-8")
            parcellation.write_text("parcel", encoding="utf-8")

            class _FakeLabels:
                @staticmethod
                def ravel():
                    return [1, 2]

            with mock.patch.dict(os.environ, {"PATH": bin_dir.as_posix() + os.pathsep + os.environ.get("PATH", "")}), mock.patch(
                "connectome.pipeline._load_label_volume", return_value=(_FakeLabels(), None)
            ):
                result = run_connectome(
                    ConnectomeConfig(
                        subject_id="149741",
                        tractography_path=tractography,
                        parcellation_path=parcellation,
                        output_dir=root / "connectome",
                        mrtrix_bin_dir=bin_dir,
                        backend="mrtrix",
                        execute=True,
                    )
                )

            self.assertTrue(result.executed)
            self.assertEqual(result.return_code, 0)
            self.assertEqual(result.matrix_path.read_text(encoding="utf-8").strip(), "1,2,3")
            qc = json.loads(result.qc_path.read_text(encoding="utf-8"))
            self.assertEqual(qc["status"], "completed")
            self.assertEqual(qc["backend"], "mrtrix3.tck2connectome")
            self.assertEqual(qc["node_count"], 2)
            self.assertEqual(qc["dense_node_count"], 1)
            self.assertEqual(qc["edge_count"], 3)
            self.assertEqual(qc["streamline_count_assigned"], 6)
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["backend"]["mrtrix_bin_dir"], bin_dir.resolve().as_posix())
            self.assertIn("-assignment_radial_search", manifest["command"])
            self.assertIn("4.0", manifest["command"])
            self.assertIn("-zero_diagonal", manifest["command"])

    def test_execute_true_runs_fake_tck2connectome_with_sift2_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            fake = bin_dir / "tck2connectome"
            fake.write_text(
                "#!/bin/sh\n"
                "printf '1.5,0\\n0,2.5\\n' > \"$3\"\n"
                "case \"$*\" in *-tck_weights_in*) exit 0;; *) exit 2;; esac\n",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            if os.name == "nt":
                (bin_dir / "tck2connectome.cmd").write_text(
                    "@echo off\r\n"
                    "(echo 1.5,0& echo 0,2.5)> \"%~3\"\r\n"
                    "echo %* | findstr /C:\"-tck_weights_in\" >nul && exit /b 0 || exit /b 2\r\n",
                    encoding="utf-8",
                )
            tractography = root / "streamlines.tck"
            parcellation = root / "wmparc.nii.gz"
            weights = root / "sift2.txt"
            tractography.write_text("tck", encoding="utf-8")
            parcellation.write_text("parcel", encoding="utf-8")
            weights.write_text("1.0\n2.0\n", encoding="utf-8")

            class _FakeLabels:
                @staticmethod
                def ravel():
                    return [1, 2]

            with mock.patch("connectome.pipeline._load_label_volume", return_value=(_FakeLabels(), None)):
                result = run_connectome(
                    ConnectomeConfig(
                        subject_id="149741",
                        tractography_path=tractography,
                        parcellation_path=parcellation,
                        output_dir=root / "connectome",
                        metric="sift2_weighted_streamline_count",
                        sift2_weights_path=weights,
                        mrtrix_bin_dir=bin_dir,
                        backend="mrtrix",
                        execute=True,
                    )
                )

            qc = json.loads(result.qc_path.read_text(encoding="utf-8"))
            self.assertTrue(result.executed)
            self.assertEqual(qc["status"], "completed")
            self.assertEqual(qc["backend"], "mrtrix3.tck2connectome")
            self.assertEqual(qc["edge_count"], 2)
            self.assertEqual(qc["streamline_count_assigned"], 4.0)
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertIn("-tck_weights_in", manifest["command"])
            self.assertIn("-force", manifest["command"])

    def test_mrtrix_dense_matrix_can_be_compressed_to_sparse_labels(self) -> None:
        try:
            import nibabel as nib
            import numpy as np
        except ImportError:
            self.skipTest("nibabel/numpy are required for dense remap connectome test")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            fake = bin_dir / "tck2connectome"
            fake.write_text(
                "#!/bin/sh\n"
                "printf '0,0,0,0,0\\n0,2,0,0,7\\n0,0,0,0,0\\n0,0,0,0,0\\n0,0,0,0,3\\n' > \"$3\"\n"
                "exit 0\n",
                encoding="utf-8",
            )
            fake.chmod(0o755)
            if os.name == "nt":
                (bin_dir / "tck2connectome.cmd").write_text(
                    "@echo off\r\n"
                    "(echo 0,0,0,0,0& echo 0,2,0,0,7& echo 0,0,0,0,0& echo 0,0,0,0,0& echo 0,0,0,0,3)> \"%~3\"\r\n"
                    "exit /b 0\r\n",
                    encoding="utf-8",
                )
            tractography = root / "streamlines.tck"
            tractography.write_text("tck", encoding="utf-8")
            labels = np.zeros((6, 6, 6), dtype=np.int16)
            labels[1, 1, 1] = 2
            labels[5, 5, 5] = 5
            parcellation = root / "wmparc.nii.gz"
            nib.save(nib.Nifti1Image(labels, np.eye(4)), parcellation.as_posix())

            result = run_connectome(
                ConnectomeConfig(
                    subject_id="dense",
                    tractography_path=tractography,
                    parcellation_path=parcellation,
                    output_dir=root / "connectome",
                    mrtrix_bin_dir=bin_dir,
                    backend="mrtrix",
                    compress_mrtrix_dense=True,
                    execute=True,
                )
            )

            matrix = result.matrix_path.read_text(encoding="utf-8")
            self.assertIn("source_node,target_node,streamline_count", matrix)
            self.assertIn("2,2,2", matrix)
            self.assertIn("2,5,7", matrix)
            self.assertIn("5,5,3", matrix)
            qc = json.loads(result.qc_path.read_text(encoding="utf-8"))
            self.assertEqual(qc["node_count"], 2)
            self.assertEqual(qc["edge_count"], 3)
            self.assertEqual(qc["dense_node_count"], 5)
            self.assertEqual(qc["dense_remap_mode"], "one_based_label_indices")
            self.assertEqual(qc["streamline_count_assigned"], 12)
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertTrue(manifest["compress_mrtrix_dense"])
            self.assertTrue(Path(manifest["outputs"]["dense_matrix"]).is_file())

    def test_execute_true_computes_endpoint_matrix_for_ascii_vtk(self) -> None:
        try:
            import nibabel as nib
            import numpy as np
        except ImportError:
            self.skipTest("nibabel/numpy are required for endpoint-label connectome test")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tractography = root / "streamlines.vtk"
            tractography.write_text(
                "# vtk DataFile Version 3.0\n"
                "test\n"
                "ASCII\n"
                "DATASET POLYDATA\n"
                "POINTS 4 float\n"
                "1 1 1 3 3 3 1 1 1 3 3 3\n"
                "LINES 2 6\n"
                "2 0 1\n"
                "2 2 3\n",
                encoding="utf-8",
            )
            labels = np.zeros((5, 5, 5), dtype=np.int16)
            labels[1, 1, 1] = 10
            labels[3, 3, 3] = 20
            parcellation = root / "wmparc.nii.gz"
            nib.save(nib.Nifti1Image(labels, np.eye(4)), parcellation.as_posix())

            result = run_connectome(
                ConnectomeConfig(
                    subject_id="vtk",
                    tractography_path=tractography,
                    parcellation_path=parcellation,
                    output_dir=root / "connectome",
                    assignment_radius_mm=0,
                    execute=True,
                )
            )

            self.assertTrue(result.executed)
            self.assertIn("10,20,2", result.matrix_path.read_text(encoding="utf-8"))
            qc = json.loads(result.qc_path.read_text(encoding="utf-8"))
            self.assertEqual(qc["streamline_count_total"], 2)
            self.assertEqual(qc["streamline_count_assigned"], 2)
            self.assertEqual(qc["edge_count"], 1)

    def test_ddparcel_connectome_excludes_non_gray_matter_nodes(self) -> None:
        try:
            import nibabel as nib
            import numpy as np
        except ImportError:
            self.skipTest("nibabel/numpy are required for DDParcel node-filter test")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tractography = root / "streamlines.vtk"
            tractography.write_text(
                "# vtk DataFile Version 3.0\n"
                "test\n"
                "ASCII\n"
                "DATASET POLYDATA\n"
                "POINTS 4 float\n"
                "1 1 1 3 3 3 2 2 2 3 3 3\n"
                "LINES 2 6\n"
                "2 0 1\n"
                "2 2 3\n",
                encoding="utf-8",
            )
            labels = np.zeros((5, 5, 5), dtype=np.int16)
            labels[1, 1, 1] = 2
            labels[2, 2, 2] = 10
            labels[3, 3, 3] = 1001
            labels[4, 4, 4] = 5
            parcellation = root / "wmparc.nii.gz"
            nib.save(nib.Nifti1Image(labels, np.eye(4)), parcellation.as_posix())

            result = run_connectome(
                ConnectomeConfig(
                    subject_id="ddparcel",
                    tractography_path=tractography,
                    parcellation_path=parcellation,
                    output_dir=root / "connectome",
                    assignment_radius_mm=0,
                    execute=True,
                )
            )

            nodes = result.nodes_path.read_text(encoding="utf-8")
            self.assertNotIn("\t2\tLeft-Cerebral-White-Matter", nodes)
            self.assertNotIn("\t5\tLeft-Inf-Lat-Vent", nodes)
            self.assertIn("\t10\tLeft-Thalamus-Proper", nodes)
            self.assertIn("\t1001\tctx-lh-bankssts", nodes)
            self.assertIn("10,1001,1", result.matrix_path.read_text(encoding="utf-8"))
            qc = json.loads(result.qc_path.read_text(encoding="utf-8"))
            self.assertEqual(qc["node_count"], 2)
            self.assertEqual(qc["input_label_count"], 4)
            self.assertEqual(qc["excluded_node_label_count"], 2)
            self.assertEqual(qc["streamline_count_assigned"], 1)
            self.assertEqual(qc["streamline_count_skipped_non_node"], 1)

    def test_python_connectome_uses_configured_radial_assignment(self) -> None:
        try:
            import nibabel as nib
            import numpy as np
        except ImportError:
            self.skipTest("nibabel/numpy are required for radial-assignment test")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tractography = root / "streamlines.vtk"
            tractography.write_text(
                "# vtk DataFile Version 3.0\n"
                "test\n"
                "ASCII\n"
                "DATASET POLYDATA\n"
                "POINTS 4 float\n"
                "0 0 0 4 0 0 2 0 0 2 0 0\n"
                "LINES 2 6\n"
                "2 0 1\n"
                "2 2 3\n",
                encoding="utf-8",
            )
            labels = np.zeros((6, 3, 3), dtype=np.int16)
            labels[2, 0, 0] = 10
            labels[4, 0, 0] = 20
            parcellation = root / "parcellation.nii.gz"
            nib.save(nib.Nifti1Image(labels, np.eye(4)), parcellation.as_posix())

            result = run_connectome(
                ConnectomeConfig(
                    subject_id="radial",
                    tractography_path=tractography,
                    parcellation_path=parcellation,
                    output_dir=root / "connectome",
                    assignment_radius_mm=2.1,
                    execute=True,
                )
            )

            self.assertIn("10,20,1", result.matrix_path.read_text(encoding="utf-8"))
            self.assertNotIn("10,10", result.matrix_path.read_text(encoding="utf-8"))
            qc = json.loads(result.qc_path.read_text(encoding="utf-8"))
            self.assertEqual(qc["assignment_method"], "radial_search")
            self.assertEqual(qc["assignment_radius_mm"], 2.1)
            self.assertEqual(qc["endpoint_count_radially_assigned"], 1)
            self.assertEqual(qc["streamline_count_endpoint_assigned"], 2)
            self.assertEqual(qc["streamline_count_matrix_included"], 1)
            self.assertEqual(qc["streamline_count_self_connection"], 1)

    def test_nodes_table_uses_explicit_lut_names(self) -> None:
        try:
            import nibabel as nib
            import numpy as np
        except ImportError:
            self.skipTest("nibabel/numpy are required for LUT connectome test")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tractography = root / "streamlines.vtk"
            tractography.write_text(
                "# vtk DataFile Version 3.0\n"
                "test\n"
                "ASCII\n"
                "DATASET POLYDATA\n"
                "POINTS 2 float\n"
                "1 1 1 3 3 3\n"
                "LINES 1 3\n"
                "2 0 1\n",
                encoding="utf-8",
            )
            labels = np.zeros((5, 5, 5), dtype=np.int16)
            labels[1, 1, 1] = 10
            labels[3, 3, 3] = 20
            parcellation = root / "wmparc.nii.gz"
            lut = root / "labels.tsv"
            nib.save(nib.Nifti1Image(labels, np.eye(4)), parcellation.as_posix())
            lut.write_text("label\tname\n10\tLeft-Test\n20\tRight-Test\n", encoding="utf-8")

            result = run_connectome(
                ConnectomeConfig(
                    subject_id="lut",
                    tractography_path=tractography,
                    parcellation_path=parcellation,
                    output_dir=root / "connectome",
                    lut_path=lut,
                    execute=True,
                )
            )

            nodes = result.nodes_path.read_text(encoding="utf-8")
            self.assertIn("10\tLeft-Test", nodes)
            self.assertIn("20\tRight-Test", nodes)
            qc = json.loads(result.qc_path.read_text(encoding="utf-8"))
            self.assertEqual(qc["named_node_count"], 2)
            self.assertEqual(qc["unnamed_node_count"], 0)

    def test_execute_true_computes_scalar_weighted_endpoint_matrix_for_ascii_vtk(self) -> None:
        try:
            import nibabel as nib
            import numpy as np
        except ImportError:
            self.skipTest("nibabel/numpy are required for scalar-weighted connectome test")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tractography = root / "streamlines.vtk"
            tractography.write_text(
                "# vtk DataFile Version 3.0\n"
                "test\n"
                "ASCII\n"
                "DATASET POLYDATA\n"
                "POINTS 4 float\n"
                "1 1 1 3 3 3 1 1 1 3 3 3\n"
                "LINES 2 6\n"
                "2 0 1\n"
                "2 2 3\n",
                encoding="utf-8",
            )
            labels = np.zeros((5, 5, 5), dtype=np.int16)
            labels[1, 1, 1] = 10
            labels[3, 3, 3] = 20
            scalar = np.zeros((5, 5, 5), dtype=np.float32)
            scalar[1, 1, 1] = 0.2
            scalar[3, 3, 3] = 0.6
            parcellation = root / "wmparc.nii.gz"
            scalar_path = root / "fa.nii.gz"
            nib.save(nib.Nifti1Image(labels, np.eye(4)), parcellation.as_posix())
            nib.save(nib.Nifti1Image(scalar, np.eye(4)), scalar_path.as_posix())

            result = run_connectome(
                ConnectomeConfig(
                    subject_id="vtk",
                    tractography_path=tractography,
                    parcellation_path=parcellation,
                    output_dir=root / "connectome",
                    metric="mean_fa",
                    scalar_map_path=scalar_path,
                    execute=True,
                )
            )

            self.assertTrue(result.executed)
            matrix = result.matrix_path.read_text(encoding="utf-8")
            self.assertIn("source_node,target_node,mean_fa", matrix)
            self.assertIn("10,20,0.400000", matrix[:80])
            qc = json.loads(result.qc_path.read_text(encoding="utf-8"))
            self.assertEqual(qc["streamline_count_scalar_sampled"], 2)

    def test_execute_true_computes_length_endpoint_matrix_for_ascii_vtk(self) -> None:
        try:
            import nibabel as nib
            import numpy as np
        except ImportError:
            self.skipTest("nibabel/numpy are required for length-weighted connectome test")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tractography = root / "streamlines.vtk"
            tractography.write_text(
                "# vtk DataFile Version 3.0\n"
                "test\n"
                "ASCII\n"
                "DATASET POLYDATA\n"
                "POINTS 4 float\n"
                "1 1 1 4 1 1 1 1 1 1 5 1\n"
                "LINES 2 6\n"
                "2 0 1\n"
                "2 2 3\n",
                encoding="utf-8",
            )
            labels = np.zeros((6, 6, 6), dtype=np.int16)
            labels[1, 1, 1] = 10
            labels[4, 1, 1] = 20
            labels[1, 5, 1] = 20
            parcellation = root / "wmparc.nii.gz"
            nib.save(nib.Nifti1Image(labels, np.eye(4)), parcellation.as_posix())

            mean_result = run_connectome(
                ConnectomeConfig(
                    subject_id="length",
                    tractography_path=tractography,
                    parcellation_path=parcellation,
                    output_dir=root / "connectome_mean",
                    metric="mean_length_mm",
                    execute=True,
                )
            )
            total_result = run_connectome(
                ConnectomeConfig(
                    subject_id="length",
                    tractography_path=tractography,
                    parcellation_path=parcellation,
                    output_dir=root / "connectome_total",
                    metric="total_length_mm",
                    execute=True,
                )
            )

            self.assertIn("10,20,3.5", mean_result.matrix_path.read_text(encoding="utf-8"))
            self.assertIn("10,20,7.0", total_result.matrix_path.read_text(encoding="utf-8"))
            qc = json.loads(mean_result.qc_path.read_text(encoding="utf-8"))
            self.assertEqual(qc["streamline_count_length_sampled"], 2)

    def test_execute_true_computes_python_sift2_weighted_matrix_for_tck(self) -> None:
        try:
            import nibabel as nib
            import numpy as np
            from nibabel.streamlines import ArraySequence, TckFile, Tractogram
        except ImportError:
            self.skipTest("nibabel/numpy are required for SIFT2-weighted connectome test")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            streamlines = ArraySequence()
            streamlines.append(np.asarray([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]], dtype=np.float32))
            streamlines.append(np.asarray([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]], dtype=np.float32))
            tractography = root / "streamlines.tck"
            nib.streamlines.save(TckFile(Tractogram(streamlines, affine_to_rasmm=np.eye(4))), tractography.as_posix())
            labels = np.zeros((5, 5, 5), dtype=np.int16)
            labels[1, 1, 1] = 10
            labels[3, 3, 3] = 20
            parcellation = root / "wmparc.nii.gz"
            weights = root / "sift2.txt"
            nib.save(nib.Nifti1Image(labels, np.eye(4)), parcellation.as_posix())
            weights.write_text("# comment\n2.0 3.0\n", encoding="utf-8")

            result = run_connectome(
                ConnectomeConfig(
                    subject_id="tck",
                    tractography_path=tractography,
                    parcellation_path=parcellation,
                    output_dir=root / "connectome",
                    metric="sift2_weighted_streamline_count",
                    sift2_weights_path=weights,
                    backend="python",
                    execute=True,
                )
            )

            self.assertIn("10,20,5.0", result.matrix_path.read_text(encoding="utf-8"))
            qc = json.loads(result.qc_path.read_text(encoding="utf-8"))
            self.assertEqual(qc["backend"], "python.radial_search")
            self.assertEqual(qc["assignment_radius_mm"], 4.0)
            self.assertEqual(qc["sift2_weight_count"], 2)
            self.assertEqual(qc["streamline_count_assigned"], 5.0)

    def test_tck_header_streamline_count_is_read_lazily(self) -> None:
        try:
            import nibabel as nib
            import numpy as np
            from nibabel.streamlines import ArraySequence, TckFile, Tractogram
        except ImportError:
            self.skipTest("nibabel/numpy are required for TCK header count test")

        from connectome.pipeline import _tck_header_streamline_count

        with tempfile.TemporaryDirectory() as tmp:
            streamlines = ArraySequence()
            streamlines.append(np.asarray([[0, 0, 0], [1, 1, 1]], dtype=np.float32))
            streamlines.append(np.asarray([[2, 2, 2], [3, 3, 3]], dtype=np.float32))
            path = Path(tmp) / "tracks.tck"
            nib.streamlines.save(TckFile(Tractogram(streamlines, affine_to_rasmm=np.eye(4))), path.as_posix())

            self.assertEqual(_tck_header_streamline_count(path), 2)


if __name__ == "__main__":
    unittest.main()
