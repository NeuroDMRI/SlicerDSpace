"""T1 D6: direction-encoded colour FA map (C49) and run_record.json (C40)."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "Testing" / "Python") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "Testing" / "Python"))

from shared import teem_dti  # noqa: E402


def stick_tensor(direction, fa_like=1.0):
    """A rank-one tensor along ``direction`` (3,) as a flat 9-vector."""
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    return (fa_like * np.outer(direction, direction)).reshape(9)


def voxel_frame_tensor(ras_direction, frame):
    """Components in the voxel frame that give a stick along ``ras_direction`` after D_RAS = M D M^T."""
    inverse = np.linalg.inv(frame)
    return (inverse @ stick_tensor(ras_direction).reshape(3, 3) @ inverse.T).reshape(9)


# A permuted and flipped voxel frame: voxel x -> RAS y, voxel y -> -RAS x, voxel z -> RAS z.
PERMUTED_FLIPPED = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


class DecColourTest(unittest.TestCase):
    def colour(self, ras_direction, fa=1.0, frame=PERMUTED_FLIPPED):
        tensor = voxel_frame_tensor(ras_direction, frame)[None, :]
        return teem_dti.directional_colour(tensor, frame, np.array([fa]))[0].tolist()

    def test_d6_tensor_along_x_in_a_permuted_flipped_voxel_frame_is_pure_red(self):
        self.assertEqual(self.colour([1, 0, 0]), [255, 0, 0])

    def test_d6_tensor_along_y_is_pure_green_and_along_z_pure_blue(self):
        self.assertEqual(self.colour([0, 1, 0]), [0, 255, 0])
        self.assertEqual(self.colour([0, 0, 1]), [0, 0, 255])

    def test_d6_the_voxel_frame_alone_would_give_the_wrong_colour(self):
        # Without the frame, a stick along voxel x would be red; in RAS it is along y, so it must be green.
        tensor = stick_tensor([1, 0, 0])[None, :]
        self.assertEqual(teem_dti.directional_colour(tensor, PERMUTED_FLIPPED, np.array([1.0]))[0].tolist(), [0, 255, 0])
        self.assertEqual(teem_dti.directional_colour(tensor, np.eye(3), np.array([1.0]))[0].tolist(), [255, 0, 0])

    def test_d6_fa_scales_the_colour(self):
        self.assertEqual(self.colour([1, 0, 0], fa=0.5), [128, 0, 0])  # 127.5 rounds to even: 128

    def test_d6_fa_is_clipped_to_the_unit_interval_and_nan_is_black(self):
        self.assertEqual(self.colour([1, 0, 0], fa=1.7), [255, 0, 0])
        self.assertEqual(self.colour([1, 0, 0], fa=-0.2), [0, 0, 0])
        self.assertEqual(self.colour([1, 0, 0], fa=float("nan")), [0, 0, 0])

    def test_d6_a_non_orthogonal_frame_rotates_the_tensor_before_eigh(self):
        # R3-n7: oblique data with anisotropic voxels. D_RAS = M D M^T must hold also when M is not orthogonal.
        frame = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
        tensor = voxel_frame_tensor([0, 1, 0], frame)[None, :]
        self.assertEqual(teem_dti.directional_colour(tensor, frame, np.array([1.0]))[0].tolist(), [0, 255, 0])
        # The eigenvectors of D (before M) would point elsewhere, so this fails if M is applied after eigh.
        values, vectors = np.linalg.eigh(tensor.reshape(3, 3))
        self.assertFalse(np.allclose(np.abs(vectors[:, 2]), [0, 1, 0]))

    def test_d6_oblique_direction_gives_all_three_channels_and_the_sign_does_not_matter(self):
        direction = np.array([1.0, 2.0, 2.0]) / 3.0
        expected = np.round(255 * np.abs(direction)).astype(int).tolist()
        self.assertEqual(self.colour(direction), expected)
        self.assertEqual(self.colour(-direction), expected)

    def test_d6_voxel_count_mismatch_is_an_error(self):
        with self.assertRaises(teem_dti.TeemError):
            teem_dti.directional_colour(np.zeros((2, 9)), np.eye(3), np.zeros(3))

    def test_d6_output_is_uint8_n_by_3(self):
        rgb = teem_dti.directional_colour(np.tile(stick_tensor([1, 0, 0]), (4, 1)), np.eye(3), np.ones(4))
        self.assertEqual((rgb.dtype, rgb.shape), (np.dtype(np.uint8), (4, 3)))


class DecNrrdWriterTest(unittest.TestCase):
    def write(self, tmp, rgb, dims, ijk_to_ras):
        path = Path(tmp) / "dec.nrrd"
        teem_dti.write_dec_nrrd(path, rgb, dims, ijk_to_ras)
        return path

    def test_d6_header_declares_a_three_component_uchar_rgb_volume_in_lps(self):
        rgb = np.arange(2 * 3 * 4 * 3, dtype=np.uint8).reshape(-1, 3)
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write(tmp, rgb, (2, 3, 4), np.diag([2.0, 3.0, 4.0, 1.0]))
            header = path.read_bytes().split(b"\n\n", 1)[0].decode("ascii")
        for line in ("type: uchar", "dimension: 4", "space: left-posterior-superior", "sizes: 3 2 3 4",
                     "kinds: RGB-color domain domain domain", "encoding: gzip"):
            self.assertIn(line, header.splitlines())
        self.assertIn("space directions: none (-2.0,0.0,0.0) (0.0,-3.0,0.0) (0.0,0.0,4.0)", header)

    def test_d6_simpleitk_reads_back_values_size_and_space(self):
        sitk = __import__("SimpleITK")
        rng = np.random.default_rng(3)
        rgb = rng.integers(0, 256, size=(2 * 3 * 4, 3), dtype=np.uint8)
        ijk_to_ras = np.array([[0.0, -2.0, 0.0, 10.0], [3.0, 0.0, 0.0, -20.0], [0.0, 0.0, 4.0, 30.0], [0, 0, 0, 1.0]])
        with tempfile.TemporaryDirectory() as tmp:
            image = sitk.ReadImage(str(self.write(tmp, rgb, (2, 3, 4), ijk_to_ras)))
            self.assertEqual(image.GetNumberOfComponentsPerPixel(), 3)
            self.assertEqual(image.GetPixelID(), sitk.sitkVectorUInt8)
            self.assertEqual(image.GetSize(), (2, 3, 4))
            array = sitk.GetArrayFromImage(image)  # (z, y, x, 3)
            np.testing.assert_array_equal(array.reshape(-1, 3), rgb)
            lps = np.diag([-1.0, -1.0, 1.0, 1.0]) @ ijk_to_ras
            np.testing.assert_allclose(image.GetOrigin(), lps[:3, 3])
            np.testing.assert_allclose(np.array(image.GetDirection()).reshape(3, 3) * np.array(image.GetSpacing()),
                                       lps[:3, :3])


class DecCommandTest(unittest.TestCase):
    def test_d6_estimate_command_carries_dec_and_the_parser_accepts_it(self):
        command = teem_dti.estimate_command("py", "dwi.nhdr", "dti.nrrd", "b0.nrrd", "WLS",
                                            [("FractionalAnisotropy", "fa.nrrd")], receipt="r.json", dec="dec.nrrd")
        self.assertEqual(command[command.index("--dec") + 1], "dec.nrrd")
        with tempfile.TemporaryDirectory() as tmp:
            command[command.index("--receipt") + 1] = str(Path(tmp) / "r.json")
            with patch.object(teem_dti, "estimate_file", return_value={}), \
                    patch.object(teem_dti, "scalar_files", return_value={}), \
                    patch.object(teem_dti, "dec_file", return_value={"kind": "teem_dec"}) as dec:
                self.assertEqual(teem_dti.main(command[2:]), 0)
            dec.assert_called_once_with("dti.nrrd", "fa.nrrd", "dec.nrrd")
            self.assertEqual(json.loads((Path(tmp) / "r.json").read_text())["dec"], {"kind": "teem_dec"})

    def test_d6_dec_without_the_fa_scalar_is_a_clear_error(self):
        with patch.object(teem_dti, "estimate_file", return_value={}), patch.object(teem_dti, "scalar_files"):
            self.assertEqual(teem_dti.main(["estimate", "--dec", "d.nrrd", "--scalar", "Trace=t.nrrd",
                                            "dwi.nhdr", "dti.nrrd", "b0.nrrd"]), 2)


class DecBundleTest(unittest.TestCase):
    def build(self):
        from shared import pipeline
        from Testing.Python.test_native_tensor_integration import fake_teem_run
        from Testing.Python.test_shared_preparation import inputs

        commands = []
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        raw = inputs(Path(tmp.name))
        with patch.object(pipeline, "_require_slicer"), \
                patch.object(pipeline, "resolve_pythonslicer_executable", return_value="/slicer/bin/PythonSlicer"), \
                patch.object(pipeline, "_run_cli", side_effect=fake_teem_run(commands)), \
                patch.object(pipeline, "_convert_nrrd_to_nifti",
                             side_effect=lambda source, target: target.write_bytes(source.read_bytes())):
            bundle = pipeline.ensure_native_dti_bundle(raw, "wls")
        return pipeline, raw, bundle, commands

    def test_d6_the_bundle_names_and_requests_the_dec_map(self):
        _pipeline, raw, bundle, commands = self.build()
        self.assertEqual(bundle.dec_nrrd_path, bundle.bundle_dir / f"{raw.subject_id}-dti-DEC.nrrd")
        command = commands[0]
        self.assertEqual(command[command.index("--dec") + 1], str(bundle.dec_nrrd_path))
        self.assertTrue(bundle.dec_nrrd_path.is_file())

    def test_d6_a_bundle_without_the_dec_map_is_incomplete_and_refits_once(self):
        pipeline, raw, bundle, commands = self.build()
        self.assertTrue(pipeline._native_outputs_complete(bundle))
        bundle.dec_nrrd_path.unlink()
        self.assertFalse(pipeline._native_outputs_complete(bundle))

    def test_d6_the_expected_cache_bundle_of_the_tractography_helper_has_the_field(self):
        from tractography.dti import _native_bundle_from_expected_cache

        bundle = _native_bundle_from_expected_cache(Path("/subject"), "s1", "wls")
        self.assertEqual(bundle.dec_nrrd_path.name, "s1-dti-DEC.nrrd")


class DecResultPanelTest(unittest.TestCase):
    def manifest(self, tmp):
        dec = Path(tmp) / "100307-dti-DEC.nrrd"
        fa = Path(tmp) / "100307_dti_FA.nii.gz"
        other = Path(tmp) / "DTI.nrrd"
        for path in (dec, fa, other):
            path.write_bytes(b"x")
        return {"tasks": [{"name": "microstructure.dti", "status": "completed",
                           "outputs": {"fa": str(fa), "dec": str(dec), "tensor": str(other)}}]}

    def test_d6_the_dec_map_is_a_result_candidate_and_other_nrrd_files_are_not(self):
        from shared.auto_results import automatic_result_candidates

        with tempfile.TemporaryDirectory() as tmp:
            selected, _skipped, deferred = automatic_result_candidates(self.manifest(tmp))
            roles = {Path(item["path"]).name: item["display_role"] for item in selected}
            self.assertEqual(roles["100307-dti-DEC.nrrd"], "vector:dec")
            self.assertEqual(roles["100307_dti_FA.nii.gz"], "scalar:fa")
            self.assertNotIn("DTI.nrrd", roles)
            self.assertNotIn("DTI.nrrd", {Path(item["path"]).name for item in deferred})

    def test_d6_the_dec_map_never_displaces_the_four_scalar_maps(self):
        from shared.auto_results import automatic_result_candidates

        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.manifest(tmp)
            outputs = manifest["tasks"][0]["outputs"]
            for key in ("md", "ad", "rd"):
                path = Path(tmp) / f"100307_dti_{key.upper()}.nii.gz"
                path.write_bytes(b"x")
                outputs[key] = str(path)
            selected, _skipped, _deferred = automatic_result_candidates(manifest)
            self.assertEqual(sorted(item["display_role"] for item in selected),
                             ["scalar:ad", "scalar:fa", "scalar:md", "scalar:rd", "vector:dec"])

    def test_d6_the_slicer_loader_branch_handles_nrrd(self):
        source = (REPO_ROOT / "shared/module_logic.py").read_text(encoding="utf-8")
        self.assertIn('elif suffix == ".nrrd":', source)


class DecOutputSitesTest(unittest.TestCase):
    def test_d6_microstructure_output_paths_and_orchestrator_task_carry_the_dec_map(self):
        from microstructure.pipeline import _output_paths
        from shared.orchestrator import _microstructure_dti_outputs

        paths = _output_paths("s1", Path("/out/02_microstructure"))
        self.assertEqual(paths["dec"], Path("/out/02_microstructure/dti/s1-dti-DEC.nrrd"))
        outputs = _microstructure_dti_outputs(Path("/subject"), "s1")
        self.assertEqual(outputs["dec"].name, "s1-dti-DEC.nrrd")


# --------------------------------------------------------------------------------------------- run record
class RunRecordTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.meta = self.root / "outputs" / "00_run_metadata"
        self.meta.mkdir(parents=True)

    def product(self, relative, content=b"data"):
        path = self.root / "outputs" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def manifest(self, extra_outputs=None):
        fa = self.product("02_microstructure/dti/s1_dti_FA.nii.gz", b"fa-bytes")
        dec = self.product("02_microstructure/dti/s1-dti-DEC.nrrd", b"dec-bytes")
        work = self.product("work/scratch.nii.gz", b"scratch")
        cache = self.product("cache/native_dti/DTI.nrrd", b"cache")
        outputs = {"fa": str(fa), "dec": str(dec), "work_file": str(work), "cache_file": str(cache),
                   "missing": str(self.root / "outputs" / "02_microstructure" / "gone.nii.gz"),
                   "nested": {"list": [str(fa)]}, "flag": True, "relative": "relative/file.txt", "none": None}
        outputs.update(extra_outputs or {})
        manifest = {"kind": "slicerddsurfer_product_orchestration", "subject_id": "s1", "run_name": "Test1",
                    "status": "completed", "execute": True, "targets": ["microstructure"], "atlas_t2": None,
                    "tractography_seeding": "mask", "accepted_licences": ["freesurfer"],
                    "elapsed_seconds": 12.5, "performance_profile": {"big": "telemetry"},
                    "tasks": [{"name": "microstructure.dti", "status": "completed", "reason": "", "outputs": outputs}]}
        path = self.meta / "Test1_orchestration_manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path, manifest

    def record(self, **kwargs):
        from shared.run_record import write_run_record

        path, _manifest = self.manifest(**kwargs)
        return json.loads(write_run_record(path).read_text(encoding="utf-8")), path

    def test_d6_the_record_is_written_beside_the_run_manifest(self):
        from shared.run_record import write_run_record

        path, _manifest = self.manifest()
        self.assertEqual(write_run_record(path), self.meta / "run_record.json")

    def test_d6_the_record_lists_manifest_parameters_software_licences_and_qc(self):
        record, manifest_path = self.record()
        self.assertEqual(record["kind"], "slicerddsurfer_run_record")
        self.assertEqual(record["run_manifest"], {"path": manifest_path.as_posix(),
                                                  "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()})
        self.assertEqual(record["subject_id"], "s1")
        self.assertEqual(record["parameters"]["tractography_seeding"], "mask")
        self.assertNotIn("tasks", record["parameters"])
        self.assertNotIn("performance_profile", record["parameters"])
        self.assertEqual(record["accepted_licences"], ["freesurfer"])
        self.assertIn("python", record["software"])
        self.assertIn("numpy", record["software"]["packages"])
        self.assertEqual(record["qc_summary"]["task_status_counts"], {"completed": 1})
        self.assertEqual(record["qc_summary"]["manifest_status"], "completed")

    def test_d6_the_inventory_has_one_sha256_per_final_product_file(self):
        record, _path = self.record()
        by_name = {Path(item["path"]).name: item for item in record["products"]}
        self.assertEqual(by_name["s1_dti_FA.nii.gz"]["sha256"], hashlib.sha256(b"fa-bytes").hexdigest())
        self.assertEqual(by_name["s1-dti-DEC.nrrd"]["sha256"], hashlib.sha256(b"dec-bytes").hexdigest())
        self.assertEqual(by_name["s1-dti-DEC.nrrd"]["size"], len(b"dec-bytes"))
        paths = [item["path"] for item in record["products"]]
        self.assertEqual(len(paths), len(set(paths)), "a file named twice in the outputs is listed once")

    def test_d6_work_folders_intermediates_relative_paths_and_the_record_itself_are_excluded(self):
        record, _path = self.record()
        names = {Path(item["path"]).name for item in record["products"]}
        self.assertNotIn("scratch.nii.gz", names)
        self.assertNotIn("DTI.nrrd", names)
        self.assertNotIn("file.txt", names)
        self.assertNotIn("run_record.json", names)
        run_record = self.meta / "run_record.json"
        self.assertTrue(run_record.is_file())
        self.assertNotIn(run_record.as_posix(), [item["path"] for item in record["products"]])

    def test_d6_a_second_write_does_not_list_the_first_record(self):
        from shared.run_record import write_run_record

        path, _manifest = self.manifest({"record": str(self.meta / "run_record.json")})
        write_run_record(path)
        second = json.loads(write_run_record(path).read_text(encoding="utf-8"))
        self.assertNotIn("run_record.json", {Path(item["path"]).name for item in second["products"]})

    def test_d6_a_named_but_missing_product_is_listed_without_a_hash(self):
        record, _path = self.record()
        missing = next(item for item in record["products"] if item["path"].endswith("gone.nii.gz"))
        self.assertIsNone(missing["sha256"])

    def test_d6_no_output_file_is_copied(self):
        from shared.run_record import write_run_record

        path, _manifest = self.manifest()
        before = {p for p in self.root.rglob("*") if p.is_file()}
        write_run_record(path)
        after = {p for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(after - before, {self.meta / "run_record.json"})


class RunRecordWiringTest(unittest.TestCase):
    def test_d6_a_failed_run_writes_the_record_from_the_failed_manifest(self):
        from shared import ProductOrchestratorConfig, run_subject_products

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "100307"
            config = ProductOrchestratorConfig(subject_id="100307", subject_root=root, targets=("surface",),
                                               run_name="rr-fail", execute=True)
            with patch("shared.orchestrator._run_subject_products_impl", side_effect=RuntimeError("boom")):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    run_subject_products(config)
            meta = root / "outputs" / "00_run_metadata"
            record = json.loads((meta / "run_record.json").read_text(encoding="utf-8"))
            self.assertEqual(record["qc_summary"]["manifest_status"], "failed")
            manifest = meta / "rr-fail_orchestration_manifest.json"
            self.assertEqual(record["run_manifest"]["sha256"], hashlib.sha256(manifest.read_bytes()).hexdigest())

    def test_d6_a_plan_only_run_writes_no_record(self):
        from shared import ProductOrchestratorConfig, run_subject_products
        from test_product_orchestrator import _write_valid_raw_subject

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "100307"
            _write_valid_raw_subject(root)
            run_subject_products(ProductOrchestratorConfig(subject_id="100307", subject_root=root,
                                                           targets=("microstructure",), execute=False))
            self.assertFalse((root / "outputs" / "00_run_metadata" / "run_record.json").exists())

    def test_d6_a_completed_run_records_the_final_manifest_and_its_products(self):
        from shared import ProductOrchestratorConfig, run_subject_products
        from test_product_orchestrator import _write_valid_raw_subject

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "100307"
            _write_valid_raw_subject(root)
            raw = SimpleNamespace(subject_id="100307", subject_root=root)
            names = ("fa", "md", "ad", "rd", "l1", "l2", "l3", "trace", "fwf", "fw_tissue_fa", "fw_tissue_md",
                     "fw_tissue_valid_mask", "free_water_qc", "noddi_ndi", "noddi_odi", "noddi_fwf",
                     "noddi_tissue_valid_mask", "noddi_rmse", "noddi_nrmse", "noddi_qc")
            made = {}
            for name in names + ("dec",):
                path = root / "outputs" / "02_microstructure" / "dti" / (
                    "100307-dti-DEC.nrrd" if name == "dec" else name + ".nii.gz")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode())
                made[name] = path
            result = SimpleNamespace(**{name + "_path": made[name] for name in names}, dec_path=made["dec"],
                                     roi_stats_csv_path=None, roi_stats_json_path=None)
            with patch("shared.pipeline.find_subject_root", return_value=raw), \
                    patch("shared.pipeline.resolve_preprocessed_input_bundle", return_value=raw), \
                    patch("shared.pipeline.ensure_native_dti_bundle", return_value=SimpleNamespace()), \
                    patch("microstructure.run_microstructure", return_value=result):
                outcome = run_subject_products(ProductOrchestratorConfig(
                    "100307", root, ("microstructure",), execute=True, resume=False))
            meta = root / "outputs" / "00_run_metadata"
            record = json.loads((meta / "run_record.json").read_text(encoding="utf-8"))
            manifest_path = outcome.run_manifest_path
            self.assertEqual(manifest_path.parent.resolve(), meta.resolve())
            # The record hashes the manifest as it is on disk after the last write.
            self.assertEqual(record["run_manifest"]["sha256"], hashlib.sha256(manifest_path.read_bytes()).hexdigest())
            by_name = {Path(item["path"]).name: item for item in record["products"]}
            self.assertEqual(by_name["100307-dti-DEC.nrrd"]["sha256"], hashlib.sha256(b"dec").hexdigest())
            self.assertEqual(by_name["fa.nii.gz"]["sha256"], hashlib.sha256(b"fa").hexdigest())
            self.assertNotIn("run_record.json", by_name)


if __name__ == "__main__":
    unittest.main()
