"""UKF protocol/IO/orchestration contracts, not a substitute for scientific fitting."""

from dataclasses import asdict, replace
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tractography import ukf


def schema():
    root = ET.Element("executable")
    group = ET.SubElement(root, "parameters")
    for key, value in {**asdict(ukf.UKFParameters()), "dwiFile": "", "maskFile": "", "tracts": ""}.items():
        node = ET.SubElement(group, "boolean" if isinstance(value, bool) else "double")
        ET.SubElement(node, "name").text = key
        ET.SubElement(node, "default").text = str(value).lower()
    return root


class ProtocolTests(unittest.TestCase):
    def test_author_defaults(self):
        selected = ukf.parameters()
        self.assertEqual((selected.numTensor, selected.seedsPerVoxel), (2, 1))
        self.assertEqual(
            (selected.seedingThreshold, selected.stoppingFA, selected.stoppingThreshold), (0.18, 0.15, 0.1)
        )
        self.assertEqual((selected.stepLength, selected.recordLength, selected.maxHalfFiberLength), (0.3, 0.9, 250))
        self.assertFalse(selected.freeWater)
        self.assertEqual(selected.numThreads, -1)
        self.assertEqual((selected.Qm, selected.Ql, selected.Qw, selected.Rs), (0, 0, 0, 0))

    def test_reject_invalid_overrides(self):
        for value in (
            [],
            False,
            {"Qm": float("nan")},
            {"Rs": 0.3},
            {"numThreads": 0},
            {"numTensor": 2.0},
            {"seedsPerVoxel": 0},
            {"select_streamlines": 100000},
            {"stepLength": 0.01},
            {"freeWater": "false"},
            {"recordTensors": False},
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ukf.parameters(value)

    def test_command_respects_switch_defaults(self):
        cmd = ukf.build_command("ukf", "dwi", "mask", "out", ukf.parameters(), schema())
        self.assertIn("--recordFA", cmd)
        self.assertIn("--recordTensors", cmd)
        self.assertNotIn("--freeWater", cmd)
        self.assertNotIn("--select-streamlines", cmd)
        self.assertEqual(cmd[cmd.index("--numTensor") + 1], "2")
        cmd = ukf.build_command("ukf", "dwi", "mask", "out", ukf.parameters({"freeWater": True}), schema())
        self.assertIn("--freeWater", cmd)

    def test_schema_incompatibility_is_error(self):
        root = schema()
        root.find("./parameters/boolean[name='freeWater']/default").text = "true"
        with self.assertRaisesRegex(RuntimeError, "boolean default"):
            ukf.build_command("ukf", "dwi", "mask", "out", ukf.parameters(), root)
        with self.assertRaisesRegex(RuntimeError, "lacks"):
            ukf.build_command("ukf", "dwi", "mask", "out", ukf.parameters(), ET.Element("executable"))

    def test_missing_extension_never_falls_back(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch("shutil.which", return_value=None),
            mock.patch("shared.pipeline.resolve_cli_executable", side_effect=RuntimeError("missing")),
            mock.patch.object(Path, "glob", return_value=[]),
        ):
            with self.assertRaisesRegex(RuntimeError, "Extension Manager"):
                ukf.resolve_executable()

    def test_upstream_schema_probe_can_print_xml_then_attempt_empty_input(self):
        result = mock.Mock(
            returncode=1,
            stdout=ET.tostring(schema(), encoding="unicode") + "\nFound cores\n",
            stderr="LoadFiles failed after printing XML",
        )
        with mock.patch.object(ukf.subprocess, "run", return_value=result):
            parsed, xml, code = ukf.inspect_schema("ukf", {})
        self.assertEqual(code, 1)
        self.assertEqual(parsed.tag, "executable")
        self.assertTrue(xml.endswith("</executable>"))
        result.stdout = ""
        with mock.patch.object(ukf.subprocess, "run", return_value=result), self.assertRaises(RuntimeError):
            ukf.inspect_schema("ukf", {})

    def test_headless_converter_discovery(self):
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = str(Path(tmp_name).resolve())
            binary = Path(tmp) / "lib/Slicer-5.10/cli-modules/DWIConvert"
            binary.parent.mkdir(parents=True)
            binary.write_text("fixture")
            with (
                mock.patch.dict(os.environ, {"SLICER_HOME": tmp}, clear=True),
                mock.patch("shutil.which", return_value=None),
                mock.patch("shared.pipeline.resolve_cli_executable", side_effect=RuntimeError("no GUI")),
            ):
                self.assertEqual(ukf.resolve_executable(name="DWIConvert"), binary)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        import nibabel as nib
        import numpy as np
        from shared.pipeline import RawInputBundle

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Nontrivial orientation/origin detects accidental LPS or voxel-space exports.
        self.affine = np.array([[0, -2, 0, 20], [1.5, 0, 0, -10], [0, 0, 2.5, 3], [0, 0, 0, 1.0]])
        self.dwi = self.root / "data.nii.gz"
        self.mask = self.root / "mask.nii.gz"
        self.bval, self.bvec = self.root / "bvals", self.root / "bvecs"
        g = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], float)
        g[4:] /= np.sqrt(2)
        np.savetxt(self.bvec, g.T)
        np.savetxt(self.bval, [[0, 1000, 1000, 1000, 1000, 1000, 1000]])
        nib.save(nib.Nifti1Image(np.ones((6, 6, 6, 7), np.float32), self.affine), self.dwi)
        nib.save(nib.Nifti1Image(np.ones((6, 6, 6), np.uint8), self.affine), self.mask)
        self.raw = RawInputBundle(
            subject_root=self.root,
            subject_id="test",
            input_dir=self.root,
            dwi_path=self.dwi,
            bval_path=self.bval,
            bvec_path=self.bvec,
            mask_path=self.mask,
        )
        self.binary, self.converter = self.root / "UKFTractography", self.root / "DWIConvert"
        self.binary.write_text("mock CLI identity, not the author algorithm")
        self.converter.write_text("mock converter identity")

    def vtk_fixture(self):
        import vtk
        import numpy as np

        target = self.root / "fixture.vtk"
        points, cells = vtk.vtkPoints(), vtk.vtkCellArray()
        points.SetDataTypeToDouble()
        for offset in (0, 1):
            line = vtk.vtkPolyLine()
            line.GetPointIds().SetNumberOfIds(3)
            for index in range(3):
                p = self.affine @ np.array([index + 1, offset + 1, 2, 1.0])
                line.GetPointIds().SetId(index, points.InsertNextPoint(*p[:3]))
            cells.InsertNextCell(line)
        data = vtk.vtkPolyData()
        data.SetPoints(points)
        data.SetLines(cells)
        for index in (1, 2):
            fa = vtk.vtkFloatArray()
            fa.SetName(f"FA{index}")
            fa.SetNumberOfTuples(points.GetNumberOfPoints())
            fa.Fill(0.5)
            data.GetPointData().AddArray(fa)
            tensor = vtk.vtkFloatArray()
            tensor.SetName(f"tensor{index}")
            tensor.SetNumberOfComponents(9)
            tensor.SetNumberOfTuples(points.GetNumberOfPoints())
            tensor.Fill(0.001)
            data.GetPointData().AddArray(tensor)
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(str(target))
        writer.SetInputData(data)
        writer.Write()
        return target

    def test_input_grid_and_rank(self):
        import nibabel as nib
        import numpy as np

        self.assertEqual(ukf.validate_inputs(self.raw)["volumes"], 7)
        np.savetxt(self.bvec, np.array([[0, 1, 1, 1, 1, 1, 1], [0] * 7, [0] * 7]))
        with self.assertRaisesRegex(ValueError, "rank"):
            ukf.validate_inputs(self.raw)

    def test_input_invalid_signal(self):
        import nibabel as nib
        import numpy as np

        data = np.ones((6, 6, 6, 7), np.float32)
        data[0, 0, 0, 2] = np.nan
        nib.save(nib.Nifti1Image(data, self.affine), self.dwi)
        with self.assertRaisesRegex(ValueError, "NaN"):
            ukf.validate_inputs(self.raw)

    def test_isolated_nonpositive_b0_uses_separate_native_mask(self):
        import nibabel as nib
        import numpy as np

        data = np.ones((6, 6, 6, 7), np.float32)
        data[0, 0, 0, 0] = 0
        data[0, 0, 1, 0] = -1
        nib.save(nib.Nifti1Image(data, self.affine), self.dwi)
        before = {p: p.read_bytes() for p in (self.dwi, self.mask)}
        report, effective = ukf._validated_inputs(self.raw)
        self.assertEqual(report['excluded_nonpositive_b0_voxels'], 2)
        self.assertEqual(report['effective_brain_voxels'], 214)
        self.assertTrue(report['warnings'])
        output = self.root/'effective.nii.gz'
        tracking_raw = ukf.write_effective_mask(self.raw, effective, output)
        image = nib.load(output)
        self.assertTrue(np.array_equal(np.asanyarray(image.dataobj), effective))
        self.assertTrue(np.array_equal(image.affine, self.affine))
        self.assertEqual(image.get_data_dtype(), np.dtype('uint8'))
        self.assertEqual(tracking_raw.mask_path, output)
        self.assertEqual(self.raw.mask_path, self.mask)
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        with self.assertRaisesRegex(ValueError, 'must not overwrite'):
            ukf.write_effective_mask(self.raw, effective, self.mask)

    def test_widespread_or_empty_effective_mask_is_rejected(self):
        import nibabel as nib
        import numpy as np

        for excluded in (3, 216):
            with self.subTest(excluded=excluded):
                data = np.ones((6, 6, 6, 7), np.float32)
                indices = np.unravel_index(np.arange(excluded), (6, 6, 6))
                data[(*indices, np.zeros(excluded, dtype=int))] = 0
                nib.save(nib.Nifti1Image(data, self.affine), self.dwi)
                with self.assertRaisesRegex(ValueError, 'input-integrity limit'):
                    ukf.validate_inputs(self.raw)

    def test_zero_signal_outside_source_mask_does_not_change_valid_mask(self):
        import nibabel as nib
        import numpy as np

        data = np.ones((6, 6, 6, 7), np.float32)
        data[0, 0, 0, 0] = 0
        mask = np.ones((6, 6, 6), np.uint8)
        mask[0, 0, 0] = 0
        nib.save(nib.Nifti1Image(data, self.affine), self.dwi)
        nib.save(nib.Nifti1Image(mask, self.affine), self.mask)
        report, effective = ukf._validated_inputs(self.raw)
        self.assertEqual(report['excluded_nonpositive_b0_voxels'], 0)
        self.assertFalse(report['warnings'])
        self.assertTrue(np.array_equal(effective, mask))

    def test_input_mask_mismatch(self):
        import nibabel as nib
        import numpy as np

        nib.save(nib.Nifti1Image(np.ones((6, 6, 6), np.uint8), np.eye(4)), self.mask)
        with self.assertRaisesRegex(ValueError, "native grid"):
            ukf.validate_inputs(self.raw)

    def test_geometry_and_trk_preserve_ras(self):
        source = self.vtk_fixture()
        report = ukf.output_qc(source, self.raw)
        self.assertTrue(report["automatic_qc_passed"])
        self.assertEqual(report["points_inside_brain_fraction"], 1.0)
        target = self.root / "test.trk"
        ukf.export_trk(source, self.dwi, target)
        self.assertEqual(ukf.verify_trk(source, target), 2)

    def test_lossless_float_payload_and_mask_use_identical_mapping(self):
        import nibabel as nib
        import nrrd
        import numpy as np

        data = np.random.default_rng(42).uniform(0.001, 1000, (6, 6, 6, 7)).astype(np.float32)
        nib.save(nib.Nifti1Image(data, self.affine), self.dwi)
        mask_data = np.zeros((6, 6, 6), np.uint8)
        mask_data[1:3, 2:5, 3:5] = 1
        nib.save(nib.Nifti1Image(mask_data, self.affine), self.mask)
        probe = ukf.write_conversion_probe(self.raw, self.root / "probe.nii.gz")
        signed = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
        shift = np.array([0, 1, 1])
        to_lps = np.diag([-1, -1, 1])
        affine = self.affine.copy()
        affine[:3, :3] = self.affine[:3, :3] @ signed
        affine[:3, 3] = self.affine[:3, :3] @ shift + self.affine[:3, 3]
        directions = np.vstack(([np.nan] * 3, (to_lps @ affine[:3, :3]).T))
        header = {
            "space": "left-posterior-superior",
            "space directions": directions,
            "space origin": to_lps @ affine[:3, 3],
            "kinds": ["list", "domain", "domain", "domain"],
            "DWMRI_b-value": "1000",
            "measurement frame": np.eye(3),
        }
        header.update({f"DWMRI_gradient_{index:04d}": "0 0 0" for index in range(7)})

        def reorder(array):
            return np.flip(np.flip(array.transpose((2, 0, 1) + tuple(range(3, array.ndim))), 0), 2)

        converted = self.root / "probe.nrrd"
        nrrd.write(str(converted), np.moveaxis(reorder(probe), -1, 0), header)
        dwi, mask = self.root / "out.nrrd", self.root / "mask.nrrd"
        report = ukf.restore_lossless_payload(self.raw, probe, converted, dwi, mask)
        restored, output_header = nrrd.read(str(dwi))
        restored_mask, mask_header = nrrd.read(str(mask))
        self.assertTrue(np.array_equal(np.moveaxis(restored, 0, -1), reorder(data)))
        self.assertTrue(np.array_equal(restored_mask, reorder(mask_data)))
        self.assertTrue(np.array_equal(output_header["space origin"], mask_header["space origin"]))
        self.assertEqual(report["axis_order"], [2, 0, 1])
        self.assertFalse(report["allow_lossy_conversion"])
        self.assertFalse(report["resampled"])

    def test_shifted_geometry_fails_mask_qc(self):
        import nibabel as nib
        import numpy as np

        source = self.vtk_fixture()
        affine = self.affine.copy()
        affine[:3, 3] += 1000
        nib.save(nib.Nifti1Image(np.ones((6, 6, 6), np.uint8), affine), self.mask)
        self.assertFalse(ukf.output_qc(source, self.raw)["automatic_qc_passed"])

    def test_missing_requested_tensor_array_fails_qc(self):
        import vtk

        source = self.vtk_fixture()
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(str(source))
        reader.Update()
        data = reader.GetOutput()
        data.GetPointData().RemoveArray("tensor2")
        target = self.root / "missing.vtk"
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(str(target))
        writer.SetInputData(data)
        writer.Write()
        self.assertFalse(ukf.output_qc(target, self.raw)["automatic_qc_passed"])

    def mocked_run(self, directory, fail_qc=False, force=False):
        fixture = self.vtk_fixture()

        def stage(cmd, **kwargs):
            if "--tracts" in cmd:
                shutil.copyfile(fixture, cmd[cmd.index("--tracts") + 1])
            else:
                Path(cmd[cmd.index("--outputVolume") + 1]).write_text("mock conversion")
            return mock.Mock(wait=mock.Mock(return_value=0), poll=mock.Mock(return_value=0))

        def resolve(explicit=None, name="UKFTractography"):
            return self.binary if name == "UKFTractography" else self.converter

        with (
            mock.patch.object(ukf, "resolve_executable", side_effect=resolve),
            mock.patch.object(
                ukf.subprocess,
                "run",
                return_value=mock.Mock(
                    returncode=0, stdout='<?xml version="1.0"?>' + ET.tostring(schema(), encoding="unicode")
                ),
            ),
            mock.patch.object(ukf.subprocess, "Popen", side_effect=stage),
            mock.patch.object(ukf, "restore_lossless_payload", return_value={"policy": "mock conversion"}) as conversion,
            mock.patch("shared.pipeline.build_cli_environment", return_value={}),
        ):
            if fail_qc:
                with mock.patch.object(ukf, "output_qc", return_value={"automatic_qc_passed": False}):
                    with self.assertRaisesRegex(RuntimeError, "QC failed"):
                        ukf.run_ukf(self.raw, directory, force=force)
            else:
                result = ukf.run_ukf(self.raw, directory, force=force)
                if conversion.called:
                    self.assertEqual(conversion.call_args.args[0].mask_path,
                                     ukf.paths(self.raw.subject_id, directory)['effective_mask'])
                return result

    def test_mocked_process_lifecycle_cache_and_failure(self):
        directory = self.root / "outputs"
        result = self.mocked_run(directory)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(self.mocked_run(directory)["status"], "cached")
        self.mocked_run(directory, fail_qc=True, force=True)
        record = json.loads(ukf.paths("test", directory)["manifest"].read_text())
        self.assertEqual(record["status"], "failed")
        self.assertTrue(Path(record["failed_tractogram"]).is_file())
        with mock.patch.object(
            ukf,
            "resolve_executable",
            side_effect=lambda explicit=None, name="UKFTractography": (
                self.binary if name == "UKFTractography" else self.converter
            ),
        ):
            self.assertFalse(ukf.current(self.raw, directory))

    def test_cache_invalidated_by_parameters_inputs_and_artifacts(self):
        directory = self.root / "outputs"
        self.mocked_run(directory)
        with mock.patch.object(
            ukf,
            "resolve_executable",
            side_effect=lambda explicit=None, name="UKFTractography": (
                self.binary if name == "UKFTractography" else self.converter
            ),
        ):
            self.assertTrue(ukf.current(self.raw, directory))
            self.assertFalse(ukf.current(self.raw, directory, {"seedsPerVoxel": 2}))
            self.converter.write_text("different converter version")
            self.assertFalse(ukf.current(self.raw, directory))

    def test_algorithm_shared_library_is_part_of_cache_identity(self):
        library = self.root / "libUKFBase.so"
        library.write_text("first upstream build")
        first = ukf.signature(self.raw, {}, self.binary, self.converter)
        library.write_text("changed upstream implementation")
        self.assertNotEqual(first, ukf.signature(self.raw, {}, self.binary, self.converter))

    def test_effective_mask_is_recorded_and_protected_by_cache_receipt(self):
        import nibabel as nib
        import numpy as np

        data = np.ones((6, 6, 6, 7), np.float32)
        data[0, 0, 0, 0] = 0
        nib.save(nib.Nifti1Image(data, self.affine), self.dwi)
        directory = self.root/'outputs'
        self.mocked_run(directory)
        outputs = ukf.paths('test', directory)
        manifest = json.loads(outputs['manifest'].read_text())
        qc = json.loads(outputs['qc'].read_text())
        self.assertEqual(manifest['signature']['mask_policy'], ukf.MASK_POLICY)
        self.assertIn('effective_mask', manifest['artifacts'])
        self.assertEqual(qc['input_validation']['excluded_nonpositive_b0_voxels'], 1)
        self.assertEqual(qc['effective_mask']['sha256'], ukf.digest(outputs['effective_mask']))
        outputs['effective_mask'].write_bytes(b'truncated')
        with mock.patch.object(ukf, 'resolve_executable', side_effect=lambda explicit=None, name='UKFTractography':
                               self.binary if name == 'UKFTractography' else self.converter):
            self.assertFalse(ukf.current(self.raw, directory))

    def test_independent_audit_recognises_ukf_and_rejects_changed_mask(self):
        from scripts.audit_scientific_outputs import _audit_tractography

        directory = self.root/'outputs'
        self.mocked_run(directory)
        outputs = ukf.paths('test', directory)
        record = dict(subject_id='test', subject_root=str(self.root),
            analysis_inputs={key: str(path) for key, path in
                             dict(dwi=self.dwi, bval=self.bval, bvec=self.bvec, mask=self.mask).items()},
            tasks=[dict(name='tractography.ukf', status='completed',
                        outputs={key: str(path) for key, path in outputs.items()},
                        manifest_path=str(outputs['manifest']), qc_path=str(outputs['qc']))])
        checks = []
        self.assertTrue(_audit_tractography(record, checks))
        self.assertTrue(all(check['passed'] for check in checks), checks)
        outputs['effective_mask'].write_bytes(b'truncated')
        checks = []
        self.assertTrue(_audit_tractography(record, checks))
        self.assertIn('tractography.ukf.current_artifacts',
                      {check['name'] for check in checks if not check['passed']})


class IntegrationTests(unittest.TestCase):
    def test_final_output_archiving_preserves_current_ukf(self):
        from shared.io_layout import archive_development_outputs, subject_layout

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            layout = subject_layout(root)
            current = layout.tractograms / "ukf" / "current" / "s_ukf.vtk"
            old = layout.tractograms / "ukf" / "old" / "s_ukf.vtk"
            for path in (current, old):
                path.parent.mkdir(parents=True)
                path.write_text("fixture")
            archive_development_outputs(
                root, current_run_name="current", current_tractography_backend="ukf", archive_alternate_tractograms=True
            )
            self.assertTrue(current.is_file())
            self.assertFalse(old.is_file())

    def test_optional_target_expansion(self):
        from shared.orchestrator import normalize_orchestrator_targets

        targets = normalize_orchestrator_targets(("bundles",), tractography_backend="ukf", bundle_backend="wma")
        self.assertEqual(targets, ("tractography", "wma"))
        self.assertNotIn("segmentation", targets)
        self.assertEqual(normalize_orchestrator_targets(("bundles",), bundle_backend="wma"), ("wma",))

    def test_no_mrtrix_dependency_for_ukf_only(self):
        from shared.environment import EnvironmentDoctorConfig, build_dependency_specs

        with mock.patch.object(ukf, "resolve_executable", side_effect=RuntimeError("missing")):
            specs = build_dependency_specs(
                EnvironmentDoctorConfig(
                    targets=("tractography",), repo_root=ROOT, output_dir=ROOT / "ignore", tractography_backend="ukf"
                )
            )
        keys = {spec.key for spec in specs}
        self.assertIn("command:UKFTractography", keys)
        self.assertIn("command:DWIConvert", keys)
        self.assertNotIn("command:tckgen", keys)
        self.assertNotIn("python:dipy", keys)

    def test_current_ukf_planning_and_missing_backend(self):
        from shared.orchestrator import ProductOrchestratorConfig, plan_subject_products

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "input").mkdir()
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (root / "input" / name).write_text("fixture for planning only")
            config = ProductOrchestratorConfig(
                subject_id="s",
                subject_root=root,
                targets=("tractography",),
                tractography_backend="ukf",
                brain_mask_backend="provided",
            )
            with mock.patch.object(ukf, "resolve_executable", return_value=root / "UKFTractography"):
                plan = plan_subject_products(config, write=False)
            item = next(item for item in plan.items if item.target == "tractography")
            self.assertEqual(item.status, "will_run")
            self.assertTrue(str(item.outputs["tractogram"]).endswith("s_ukf.vtk"))
            with mock.patch.object(ukf, "resolve_executable", side_effect=RuntimeError("install UKF")):
                plan = plan_subject_products(config, write=False)
            self.assertEqual(next(item for item in plan.items if item.target == "tractography").status, "blocked")

    def test_cli_accepts_backend_and_overrides(self):
        from shared.run_subject_pipeline_cli import build_parser

        args = build_parser().parse_args(
            ["--subject-root", "/tmp/s", "--tractography-backend", "ukf", "--ukf-parameters", '{"seedsPerVoxel":2}']
        )
        self.assertEqual(args.tractography_backend, "ukf")
        self.assertEqual(json.loads(args.ukf_parameters), {"seedsPerVoxel": 2})

    def test_failure_blocks_all_current_run_consumers(self):
        from shared.orchestrator import ProductOrchestratorConfig, run_subject_products
        from shared.io_layout import subject_layout

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "input").mkdir()
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (root / "input" / name).write_text("fixture for mocked execution only")
            from test_product_orchestrator import _write_valid_raw_subject
            _write_valid_raw_subject(root)
            layout = subject_layout(root)
            old = ukf.paths("s", layout.tractograms / "ukf" / "run")
            old["tractogram"].parent.mkdir(parents=True)
            old["tractogram"].write_text("stale output must not be consumed")
            old["tractogram_trk"].write_text("stale output must not be consumed")
            config = ProductOrchestratorConfig(
                subject_id="s",
                subject_root=root,
                targets=("bundles", "connectome"),
                bundle_backend="all",
                run_name="run",
                tractography_backend="ukf",
                brain_mask_backend="provided",
                execute=True,
            )
            with (
                mock.patch.object(ukf, "resolve_executable", return_value=root / "UKFTractography"),
                mock.patch.object(ukf, "run_ukf", side_effect=RuntimeError("synthetic failure")),
            ):
                result = run_subject_products(config)
            statuses = {task.name: task.status for task in result.tasks}
            self.assertEqual(statuses["tractography.ukf"], "failed")
            for name in ("bundles.wmql", "bundles.wma", "bundles.tractcloud", "connectome.streamline_count"):
                self.assertEqual(statuses[name], "blocked", name)


if __name__ == "__main__":
    unittest.main()
