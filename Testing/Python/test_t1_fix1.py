"""T1 fix round 1 (T-C1, T-B1, ...): regression tests, each named after its finding."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from unittest.mock import patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "Testing" / "Python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "Testing" / "Python"))

import user_data_isolation  # noqa: E402  B2: SHA-256 cache, user config and model root of these tests are temporary

user_data_isolation.install()

from shared import teem_dti  # noqa: E402


class GetElementOnlyMatrix:
    """A VTK-like 4 x 4 matrix: ``GetElement`` only, no array protocol (as ``vtkMatrix4x4``)."""

    def __init__(self, values=None):
        self.values = np.eye(4) if values is None else np.array(values, dtype=float)

    def GetElement(self, row, column):
        return float(self.values[row, column])

    def SetElement(self, row, column, value):
        self.values[row, column] = value

    def DeepCopy(self, other):
        self.values = np.array([[other.GetElement(r, c) for c in range(4)] for r in range(4)])

    def Invert(self):
        self.values = np.linalg.inv(self.values)


class TC1DecFileWithVtkMatrixTest(unittest.TestCase):
    def test_t_c1_dec_file_converts_vtk_matrices_with_getelement(self):
        # ijk_to_ras = inverse of ras_to_ijk; use a non-trivial one so a wrong conversion shows up.
        ras_to_ijk = np.linalg.inv(np.array([[0.0, -2.0, 0.0, 10.0], [3.0, 0.0, 0.0, -20.0],
                                            [0.0, 0.0, 4.0, 30.0], [0, 0, 0, 1.0]]))
        tensor_values = np.array([[1.0, 0, 0, 0, 0, 0, 0, 0, 0]] * 24, dtype=np.float32)
        fa_values = np.ones(24, dtype=np.float32)

        class Scalars:
            def __init__(self, data):
                self.data = data

        class Reader:
            def __init__(self):
                self.name = None
                self.data = None

            def SetFileName(self, name):
                self.name = name

            def Update(self):
                pass

            def GetReadStatus(self):
                return 0

            def GetRasToIjkMatrix(self):
                return GetElementOnlyMatrix(ras_to_ijk)

            def GetMeasurementFrameMatrix(self):
                return GetElementOnlyMatrix(np.eye(4))

            def GetOutput(self):
                reader = self

                class Output:
                    def GetDimensions(self_inner):
                        return (2, 3, 4)

                    def GetPointData(self_inner):
                        is_tensor = "tensor" in reader.name

                        class Point:
                            def GetTensors(self_p):
                                return Scalars(tensor_values) if is_tensor else None

                            def GetScalars(self_p):
                                return None if is_tensor else Scalars(fa_values)

                        return Point()

                return Output()

        vtk = SimpleNamespace(vtkMatrix4x4=GetElementOnlyMatrix)
        vtk_teem = SimpleNamespace(vtkTeemNRRDReader=Reader)
        numpy_support = SimpleNamespace(vtk_to_numpy=lambda scalars: scalars.data)
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(teem_dti, "_vtk_modules", return_value=(vtk, vtk_teem, numpy_support, None)):
            dec_path = Path(tmp) / "dec.nrrd"
            receipt = teem_dti.dec_file("tensor.nrrd", "fa.nrrd", dec_path, log=None)
            header = dec_path.read_bytes().split(b"\n\n", 1)[0].decode("ascii")
        self.assertEqual(receipt["voxels"], 24)
        self.assertIn("space directions: none (0.0,-3.0,0.0) (2.0,0.0,0.0) (0.0,0.0,4.0)", header)
        self.assertIn("space origin: (-10.0,20.0,30.0)", header)


class TB1RunRecordRootTest(unittest.TestCase):
    """T-B1: only the parts below the run output root mark a work folder."""

    def record_for(self, base):
        from shared.run_record import write_run_record

        meta = base / "outputs" / "00_run_metadata"
        meta.mkdir(parents=True)
        final = base / "outputs" / "02_microstructure" / "dti" / "fa.nii.gz"
        work = base / "outputs" / "work" / "scratch.nii.gz"
        for path, data in ((final, b"fa"), (work, b"scratch")):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        outside = base.parent / "elsewhere" / "cache" / "kept.nii.gz"
        outside.parent.mkdir(parents=True, exist_ok=True)
        outside.write_bytes(b"outside")
        manifest = {"kind": "slicerddsurfer_product_orchestration", "status": "completed",
                    "output_root": (base / "outputs").as_posix(),
                    "tasks": [{"name": "t", "status": "completed",
                               "outputs": {"fa": str(final), "work": str(work), "outside": str(outside)}}]}
        path = meta / "R_orchestration_manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        record = json.loads(write_run_record(path).read_text(encoding="utf-8"))
        return {Path(item["path"]).name for item in record["products"]}

    def test_t_b1_a_root_under_a_folder_named_tmp_keeps_its_products(self):
        with tempfile.TemporaryDirectory() as tmp:
            names = self.record_for(Path(tmp) / "tmp" / "subject")
        self.assertIn("fa.nii.gz", names)
        self.assertNotIn("scratch.nii.gz", names)

    def test_t_b1_a_root_under_a_folder_named_cache_keeps_its_products(self):
        with tempfile.TemporaryDirectory() as tmp:
            names = self.record_for(Path(tmp) / "cache" / "work" / "subject")
        self.assertIn("fa.nii.gz", names)
        self.assertNotIn("scratch.nii.gz", names)

    def test_t_b1_a_product_outside_the_root_is_kept_as_is(self):
        with tempfile.TemporaryDirectory() as tmp:
            names = self.record_for(Path(tmp) / "tmp" / "subject")
        self.assertIn("kept.nii.gz", names)


class TB2ConnectomeReuseTest(unittest.TestCase):
    """T-B2 = T-C3: connectome reuse compares the backend and the compression."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def build(self, suffix, manifest_extra):
        tract = self.root / f"tracts{suffix}"
        parc = self.root / "parc.nii.gz"
        for path in (tract, parc):
            path.write_bytes(b"x" * 10)
        out = self.root / "connectome"
        out.mkdir(exist_ok=True)
        qc = out / "s-connectome-streamline_count_qc.json"
        qc.write_text(json.dumps({"status": "completed", "node_count": 3, "edge_count": 2}), encoding="utf-8")
        matrix = out / "s-connectome-streamline_count.csv"
        matrix.write_text("source_node,target_node,value\n1,2,3\n", encoding="utf-8")
        nodes = out / "s-connectome-nodes.tsv"
        nodes.write_text("node_id\tname\n1\ta\n", encoding="utf-8")

        def record(path):
            return {"path": path.resolve().as_posix(), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}

        manifest = out / "s-connectome-streamline_count_manifest.json"
        body = {"dependencies": {"tractography": record(tract), "ddparcel_parcellation": record(parc)}}
        body.update(manifest_extra)
        manifest.write_text(json.dumps(body), encoding="utf-8")
        return qc, matrix, nodes, manifest, tract, parc

    def complete(self, suffix, manifest_extra, backend, compress=None):
        from shared.orchestrator import _connectome_complete

        qc, matrix, nodes, manifest, tract, parc = self.build(suffix, manifest_extra)
        return _connectome_complete(qc, matrix, nodes, manifest_path=manifest, tractography=tract,
                                    parcellation=parc, backend=backend, compress_mrtrix_dense=compress)

    def test_t_b2_old_python_connectome_with_new_default_auto_and_tck_source_reruns(self):
        # Fails at 5ef6025: the backend is not compared, so the old python result was reused.
        self.assertFalse(self.complete(".tck", {"requested_backend": "auto", "compress_mrtrix_dense": None},
                                       backend="auto"))

    def test_t_b2_python_connectome_asked_for_mrtrix_reruns(self):
        self.assertFalse(self.complete(".tck", {"backend_used": "python", "compress_mrtrix_dense": False},
                                       backend="mrtrix"))

    def test_t_b2_unchanged_choices_reuse(self):
        mrtrix = {"backend_used": "mrtrix", "compress_mrtrix_dense": True}
        self.assertTrue(self.complete(".tck", mrtrix, backend="auto"))
        self.assertTrue(self.complete(".tck", {"backend_used": "python", "compress_mrtrix_dense": False},
                                      backend="python"))
        # A manifest without backend_used is current when python is what a run now resolves.
        self.assertTrue(self.complete(".trk", {"requested_backend": "auto"}, backend="auto"))
        self.assertTrue(self.complete(".tck", {}, backend="python"))

    def test_t_b2_compression_change_reruns_the_mrtrix_connectome(self):
        mrtrix_plain = {"backend_used": "mrtrix", "compress_mrtrix_dense": False}
        self.assertFalse(self.complete(".tck", mrtrix_plain, backend="auto"))  # auto now compresses
        self.assertTrue(self.complete(".tck", mrtrix_plain, backend="auto", compress=False))
        self.assertFalse(self.complete(".tck", mrtrix_plain, backend="mrtrix", compress=True))
        self.assertTrue(self.complete(".tck", mrtrix_plain, backend="mrtrix"))  # explicit mrtrix: off unless set

    def test_t_b2_guard_every_orchestrator_reuse_check_passes_the_backend(self):
        source = (REPO_ROOT / "shared" / "orchestrator.py").read_text(encoding="utf-8")
        calls = source.split("_connectome_complete(")[1:]
        self.assertEqual(len(calls), 5)  # one definition and four reuse checks
        for body in calls[1:]:
            self.assertIn("backend=config.connectome_backend", body.split("):")[0])
            self.assertIn("compress_mrtrix_dense=config.connectome_compress_mrtrix_dense", body.split("):")[0])


class TB3IfwRegionalTableReuseTest(unittest.TestCase):
    """T-B3 = T-C4: cached iFW maps keep their reuse, the regional table follows the label map and the LUT."""

    SHAPE = (4, 5, 6)

    def setUp(self):
        import nibabel as nib

        self.nib = nib
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        rng = np.random.default_rng(11)
        mask = np.zeros(self.SHAPE, dtype=np.uint8)
        mask[1:3, 1:4, 1:5] = 1
        (self.root / "bvals").write_text("0 0 1000 1000 2000 2000\n")
        (self.root / "bvecs").write_text("0 0 1 0 0 1\n0 0 0 1 0 0\n0 0 0 0 1 0\n")
        self.write("fw.nii.gz", rng.random(self.SHAPE).astype(np.float32))
        self.write("csf.nii.gz", (rng.random(self.SHAPE) * 0.5).astype(np.float32))
        self.write("mask.nii.gz", mask)
        self.labels("labels.nii.gz", 7, 9)

    def write(self, name, data):
        path = self.root / name
        self.nib.save(self.nib.Nifti1Image(np.asarray(data), np.eye(4)), str(path))
        return path

    def labels(self, name, first, second):
        data = np.zeros(self.SHAPE, dtype=np.int32)
        data[1:3, 1:4, 1:3] = first
        data[1:3, 1:4, 3:5] = second
        return self.write(name, data)

    def config(self, **override):
        from microstructure.ifw import IfwConfig

        values = dict(subject_id="S1", fw_fraction_path=self.root / "fw.nii.gz",
                      csf_probability_path=self.root / "csf.nii.gz", mask_path=self.root / "mask.nii.gz",
                      bval_path=self.root / "bvals", bvec_path=self.root / "bvecs", output_dir=self.root / "out",
                      echo_time_ms=89.5)
        values.update(override)
        return IfwConfig(**values)

    def table_ids(self, result):
        import csv

        with result.roi_stats_csv_path.open(newline="", encoding="utf-8") as stream:
            return [row["label_id"] for row in csv.DictReader(stream)]

    def test_t_b3_maps_first_without_parcellation_then_with_writes_the_table_from_cached_maps(self):
        from microstructure.ifw import run_ifw

        first = run_ifw(self.config())
        self.assertIsNone(first.roi_stats_csv_path)
        from microstructure import ifw as ifw_module

        with mock.patch.object(ifw_module, "_write_roi_table", wraps=ifw_module._write_roi_table) as writer:
            second = run_ifw(self.config(roi_label_map=self.root / "labels.nii.gz"))
        self.assertEqual(second.status, "cached")  # the maps were not refitted
        self.assertEqual(writer.call_count, 1)
        self.assertEqual(self.table_ids(second), ["7", "9"])
        self.assertTrue((self.root / "out" / "S1_ROIWise-iFW.json").is_file())

    def test_t_b3_changed_label_map_regenerates_the_table(self):
        from microstructure.ifw import run_ifw

        run_ifw(self.config(roi_label_map=self.root / "labels.nii.gz"))
        self.labels("labels2.nii.gz", 3, 4)
        second = run_ifw(self.config(roi_label_map=self.root / "labels2.nii.gz"))
        self.assertEqual(second.status, "cached")
        self.assertEqual(self.table_ids(second), ["3", "4"])
        # The same path with new content also counts as changed.
        self.labels("labels2.nii.gz", 5, 6)
        self.assertEqual(self.table_ids(run_ifw(self.config(roi_label_map=self.root / "labels2.nii.gz"))), ["5", "6"])

    def test_t_b3_changed_lut_regenerates_the_table(self):
        import csv
        from microstructure.ifw import run_ifw

        lut = self.root / "lut.txt"
        lut.write_text("7 Alpha 0 0 0 0\n9 Beta 0 0 0 0\n", encoding="utf-8")
        labels = self.root / "labels.nii.gz"
        run_ifw(self.config(roi_label_map=labels))
        with_lut = run_ifw(self.config(roi_label_map=labels, lut_path=lut))
        with with_lut.roi_stats_csv_path.open(newline="", encoding="utf-8") as stream:
            self.assertEqual([row["label_name"] for row in csv.DictReader(stream)], ["Alpha", "Beta"])
        lut.write_text("7 Gamma 0 0 0 0\n9 Beta 0 0 0 0\n", encoding="utf-8")
        changed = run_ifw(self.config(roi_label_map=labels, lut_path=lut))
        with changed.roi_stats_csv_path.open(newline="", encoding="utf-8") as stream:
            self.assertEqual([row["label_name"] for row in csv.DictReader(stream)], ["Gamma", "Beta"])

    def test_t_b3_a_missing_table_is_regenerated(self):
        from microstructure.ifw import run_ifw

        first = run_ifw(self.config(roi_label_map=self.root / "labels.nii.gz"))
        first.roi_stats_csv_path.unlink()
        again = run_ifw(self.config(roi_label_map=self.root / "labels.nii.gz"))
        self.assertTrue(again.roi_stats_csv_path.is_file())

    def test_t_b3_unchanged_inputs_run_nothing(self):
        from microstructure.ifw import run_ifw

        labels = self.root / "labels.nii.gz"
        run_ifw(self.config(roi_label_map=labels))
        with mock.patch("microstructure.ifw._write_roi_table") as writer, \
                mock.patch("microstructure.ifw.derive_ifw") as derive:
            again = run_ifw(self.config(roi_label_map=labels))
        self.assertEqual(again.status, "cached")
        writer.assert_not_called()
        derive.assert_not_called()
        self.assertEqual(self.table_ids(again), ["7", "9"])


class TC5SingleRunRecordTest(unittest.TestCase):
    """T-C5: several tracking backends write the run record once, from the aggregate manifest."""

    def test_t_c5_variant_children_do_not_write_a_run_record(self):
        from shared.multi_tracking import variants
        from shared.orchestrator import ProductOrchestratorConfig, _mark_orchestration_failed

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = ProductOrchestratorConfig("S1", root, ("tractography",), output_root=root / "out", run_name="cmp",
                                               tractography_backend="csd_act,ukf", execute=True)
            children = list(variants(config))
            self.assertEqual(len(children), 2)
            self.assertTrue(all(child.write_run_record is False for child in children))
            with patch("shared.run_record.write_run_record") as writer:
                _mark_orchestration_failed(children[0], RuntimeError("boom"))
                writer.assert_not_called()
                _mark_orchestration_failed(config, RuntimeError("boom"))
                writer.assert_called_once()

    def test_t_c5_the_aggregate_manifest_gets_exactly_one_record_after_the_last_publish(self):
        from shared.multi_tracking import run
        from shared.orchestrator import (ProductOrchestratorConfig, ProductOrchestratorResult, RunPlanItem,
                                         RunPlanResult)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = ProductOrchestratorConfig("S1", root, ("tractography",), output_root=root / "out", run_name="cmp",
                                               tractography_backend="csd_act,ukf", execute=True)

            def fake_plan(child, write=True):
                return RunPlanResult("S1", root, root / "out", child.run_name, child.targets,
                                     root / "out" / "00_run_metadata" / (child.run_name + "_run_plan.json"),
                                     (RunPlanItem("tractography", "tractography." + child.tractography_backend,
                                                  "will_run", will_run=True),))

            def fake_run(child, log=None):
                path = root / (child.run_name + ".json")
                path.write_text(json.dumps({"tasks": [{"name": "tractography." + child.tractography_backend,
                                                       "status": "completed", "outputs": {}}]}))
                return ProductOrchestratorResult("S1", root, path, ())

            seen = []

            def record(manifest_path, log):
                seen.append((Path(manifest_path), json.loads(Path(manifest_path).read_text())["status"]))

            with patch("shared.orchestrator.plan_subject_products", side_effect=fake_plan), \
                    patch("shared.orchestrator.run_subject_products", side_effect=fake_run), \
                    patch("shared.orchestrator._write_run_record_best_effort", side_effect=record):
                result = run(config)
            self.assertEqual(len(seen), 1)
            self.assertEqual(seen[0][0], result.run_manifest_path)
            self.assertEqual(seen[0][1], "completed")  # the manifest is final when the record is written

    def test_t_c5_a_plan_only_aggregate_writes_no_record(self):
        from shared.multi_tracking import run
        from shared.orchestrator import ProductOrchestratorConfig, RunPlanResult

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = ProductOrchestratorConfig("S1", root, ("tractography",), output_root=root / "out", run_name="cmp",
                                               tractography_backend="csd_act,ukf", execute=False)
            plan_result = RunPlanResult("S1", root, root / "out", "cmp", config.targets, root / "plan.json", ())
            with patch("shared.orchestrator.plan_subject_products", return_value=plan_result), \
                    patch("shared.orchestrator.run_subject_products", side_effect=RuntimeError("no run")), \
                    patch("shared.orchestrator._write_run_record_best_effort") as writer:
                run(config)
            writer.assert_not_called()

    def test_t_c5_ifw_is_a_shared_product_so_variants_after_the_first_skip_it(self):
        # Related guard: iFW runs once like the other microstructure stages.
        from shared.multi_tracking import SHARED, variants
        from shared.orchestrator import ProductOrchestratorConfig

        self.assertIn("microstructure_ifw", SHARED)
        config = ProductOrchestratorConfig("S1", Path("/x"), ("tractography", "microstructure_ifw"),
                                           tractography_backend="csd_act,ukf", microstructure_run_ifw=True)
        targets = [child.targets for child in variants(config)]
        self.assertIn("microstructure_ifw", targets[0])
        self.assertNotIn("microstructure_ifw", targets[1])


class TC7AssetVersionsTest(unittest.TestCase):
    """T-C7: the run record lists version and SHA-256 values of the model profiles that the run used."""

    def manifest(self, **extra):
        data = {"kind": "slicerddsurfer_product_orchestration", "status": "completed", "brain_mask_backend": "pnl_cnn",
                "wma_atlas": "naba", "model_assets_dir": "/models",
                "tasks": [{"name": "segmentation.ddseg", "status": "completed", "outputs": {}},
                          {"name": "bundles.tractcloud@ukf", "status": "skipped", "outputs": {}},
                          {"name": "bundles.wma", "status": "completed", "outputs": {}},
                          {"name": "surface.ddsurfer", "status": "blocked", "outputs": {}}]}
        data.update(extra)
        return data

    def test_t_c7_used_profiles_follow_the_finished_tasks_and_options(self):
        from shared.run_record import used_asset_profiles

        self.assertEqual(used_asset_profiles(self.manifest()),
                         ["pnl-cnn-weights", "segmentation", "tractcloud", "wma-naba"])
        self.assertEqual(used_asset_profiles({"tasks": [], "brain_mask_backend": "synthstrip"}), [])

    def test_t_c7_versions_and_hashes_come_from_the_manifests_without_hashing_files(self):
        from shared import run_record

        manifests = {name: json.loads((REPO_ROOT / name).read_text()) for name in run_record.ASSET_MANIFESTS}
        # B2: the managed root means no model-folder variable; do not depend on the caller's environment.
        with patch.dict("os.environ"), patch.object(run_record, "sha256_file", side_effect=AssertionError("must not hash weights")):
            os.environ.pop("SLICERDSPACE_MODEL_ASSETS_DIR", None)
            profiles = run_record.asset_versions(self.manifest(model_assets_dir=None))  # managed root
        segmentation = profiles["segmentation"]
        self.assertEqual(segmentation["manifest"], "models/assets_manifest.json")
        self.assertEqual(segmentation["version"], manifests["models/assets_manifest.json"]["version"])
        records = manifests["models/assets_manifest.json"]["asset_records"]
        self.assertTrue(segmentation["files"])
        for path, digest in segmentation["files"].items():
            self.assertEqual(digest, records[path]["sha256"])
        pnl = profiles["pnl-cnn-weights"]
        self.assertEqual(pnl["manifest"], "models/external_assets_manifest.json")
        self.assertIn("brain_mask/pnl_cnn_torch/axial.pt", pnl["files"])
        self.assertTrue(all(len(value) == 64 for value in pnl["files"].values()))

    def test_t_c7_a_profile_without_a_receipt_records_the_folder_and_unknown(self):
        from shared import run_record

        with tempfile.TemporaryDirectory() as tmp:
            profiles = run_record.asset_versions(self.manifest(), repo_root=Path(tmp))
        self.assertEqual(profiles["segmentation"],
                         {"manifest": None, "version": "unknown", "folder": "/models", "files": {}})

    def test_t_c7_the_written_record_carries_the_profiles(self):
        from shared.run_record import write_run_record

        with tempfile.TemporaryDirectory() as tmp:
            meta = Path(tmp) / "out" / "00_run_metadata"
            meta.mkdir(parents=True)
            path = meta / "R_orchestration_manifest.json"
            path.write_text(json.dumps(self.manifest(output_root=str(Path(tmp) / "out"))), encoding="utf-8")
            record = json.loads(write_run_record(path).read_text(encoding="utf-8"))
        self.assertIn("segmentation", record["assets"]["model_profiles"])
        self.assertEqual(record["assets"]["model_assets_dir"], "/models")


class TN2WlsHintTest(unittest.TestCase):
    """T-N2: the WLS hint follows only the two MK-Curve stops that come from the signal data."""

    def run_with_error(self, message):
        from microstructure.dki import DkiConfig, run_dki
        from shared.pipeline import RawInputBundle
        from test_microstructure_dki import synthetic_case, write_case

        case = synthetic_case()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = write_case(root, case)
            raw = RawInputBundle(root, "synthetic", root / "input", paths["dwi"], paths["bval"], paths["bvec"],
                                 paths["mask"])
            config = DkiConfig("synthetic", raw, root / "outputs/dki")
            with patch("microstructure.mkcurve.run_mkcurve", side_effect=ValueError(message)):
                with self.assertRaises(ValueError) as caught:
                    run_dki(config)
        return str(caught.exception)

    def test_t_n2_the_two_signal_stops_get_the_hint(self):
        from microstructure.mkcurve import NO_VALID_VOXELS_STOP, SIGNAL_SCALE_STOP

        for stop in (SIGNAL_SCALE_STOP, NO_VALID_VOXELS_STOP):
            with self.subTest(stop=stop):
                self.assertIn("re-run with DKI fit method WLS (--dki-fit-method WLS)", self.run_with_error(stop))

    def test_t_n2_an_option_error_keeps_its_own_message(self):
        message = self.run_with_error("MK-Curve requires a positive chunk size and finite threshold in [0,1]")
        self.assertNotIn("WLS", message)
        self.assertEqual(message, "MK-Curve requires a positive chunk size and finite threshold in [0,1]")

    def test_t_n2_the_stop_texts_are_the_ones_mkcurve_raises(self):
        # Guard (label): the constants are the raise texts, so the hint cannot drift from the source.
        from microstructure import mkcurve

        source = (REPO_ROOT / "microstructure" / "mkcurve.py").read_text(encoding="utf-8")
        self.assertIn("raise ValueError(SIGNAL_SCALE_STOP)", source)
        self.assertIn("raise ValueError(NO_VALID_VOXELS_STOP)", source)
        self.assertTrue(mkcurve.SIGNAL_SCALE_STOP.startswith("Signal scale cannot support"))


class TN3TN4TN5IfwPlanTest(unittest.TestCase):
    """T-N3 (one b-value read), T-N4 (plan text when Free Water re-runs), T-N5 (cohort scheduling)."""

    def setUp(self):
        from test_product_orchestrator import _write_valid_raw_subject

        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.subject = Path(self.temporary.name) / "100307"
        _write_valid_raw_subject(self.subject)
        (self.subject / "input" / "bvals").write_text("0 1000 2000 2000")

    def plan(self, **override):
        from shared.orchestrator import ProductOrchestratorConfig, plan_subject_products

        values = dict(subject_id="100307", subject_root=self.subject, targets=("microstructure", "microstructure_ifw"),
                      microstructure_run_ifw=True, microstructure_ifw_echo_time_ms=89.5, execute=False)
        values.update(override)
        return {item.target: item for item in plan_subject_products(ProductOrchestratorConfig(**values), write=False).items}

    def test_t_n3_the_plan_reads_the_b_values_once_for_a_blocked_free_water_stage(self):
        from shared import orchestrator

        (self.subject / "input" / "bvals").write_text("0 1000 1000 1000")
        with patch.object(orchestrator, "_free_water_ineligible", wraps=orchestrator._free_water_ineligible) as spy:
            items = self.plan(targets=("microstructure", "microstructure_freewater"), microstructure_run_ifw=False,
                              microstructure_run_free_water=True)
        self.assertEqual(items["microstructure_freewater"].status, "blocked")
        self.assertEqual(spy.call_count, 1)

    def test_t_n3_guard_the_run_path_calls_the_check_once_per_branch(self):
        # Guard (label): the run branch needs a full execution; check the source has no repeated call.
        source = (REPO_ROOT / "shared" / "orchestrator.py").read_text(encoding="utf-8")
        self.assertEqual(source.count("_free_water_ineligible(_resolve_analysis_inputs("), 1)
        self.assertEqual(source.count("_free_water_ineligible("), 4)  # definition + FW plan + FW run + iFW plan

    def test_t_n4_ifw_will_run_when_free_water_will_re_run(self):
        from shared import orchestrator

        with patch.object(orchestrator, "_microstructure_ifw_complete", return_value=True):
            items = self.plan()
        self.assertEqual(items["microstructure_freewater"].status, "will_run")
        self.assertEqual(items["microstructure_ifw"].status, "will_run")
        self.assertIn("Free Water will re-run", items["microstructure_ifw"].reason)

    def test_t_n4_ifw_is_reused_when_free_water_is_reused(self):
        from shared import orchestrator

        with patch.object(orchestrator, "_microstructure_ifw_complete", return_value=True), \
                patch.object(orchestrator, "_microstructure_freewater_complete", return_value=True):
            items = self.plan()
        self.assertEqual(items["microstructure_freewater"].status, "skipped")
        self.assertEqual(items["microstructure_ifw"].status, "skipped")

    def test_t_n5_run_ifw_is_a_scheduled_stage_named_microstructure_ifw(self):
        import contextlib

        from microstructure import ifw

        self.assertTrue(hasattr(ifw.run_ifw, "__wrapped__"))
        stages = []

        @contextlib.contextmanager
        def admission(name, gpu=False):
            stages.append((name, gpu))
            yield None

        root = Path(self.temporary.name)
        (root / "bvals").write_text("0 1000 1000 1000\n")
        (root / "bvecs").write_text("0 1 0 0\n0 0 1 0\n0 0 0 1\n")
        config = ifw.IfwConfig("S1", root / "fw.nii.gz", root / "csf.nii.gz", root / "mask.nii.gz", root / "bvals",
                               root / "bvecs", root / "out")
        with patch("shared.cohort_schedule.stage_admission", admission):
            with self.assertRaises(ifw.FreeWaterIneligible):
                ifw.run_ifw(config)
        self.assertEqual(stages, [("microstructure_ifw", False)])

    def test_t_n5_cohort_planning_has_a_memory_profile_for_the_ifw_stage(self):
        source = (REPO_ROOT / "shared" / "cohort_planning.py").read_text(encoding="utf-8")
        self.assertIn("microstructure_ifw=", source)


class TN6UnusedImportTest(unittest.TestCase):
    def test_t_n6_to_orientation_of_has_no_unused_nibabel_import(self):
        # Guard (label): static check of the function body.
        import ast

        tree = ast.parse((REPO_ROOT / "microstructure" / "ifw.py").read_text(encoding="utf-8"))
        function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_to_orientation_of")
        imports = [n for n in ast.walk(function) if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertFalse([n for n in imports if any(a.name == "nibabel" for a in n.names)])


if __name__ == "__main__":
    unittest.main()
