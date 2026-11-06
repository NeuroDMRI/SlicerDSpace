"""Native Teem tensor fit replaces the SlicerDMRI tensor CLIs at every call site (work items W1-W8, D4-D11)."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared import pipeline
from Testing.Python.test_shared_preparation import inputs

TEEM_SCRIPT = REPO_ROOT / "shared/teem_dti.py"
SIX_SCALARS = ("FractionalAnisotropy", "Trace", "MeanDiffusivity", "MinEigenvalue", "MidEigenvalue", "MaxEigenvalue")
FAKE_RECEIPT = {"kind": "teem_tensor_estimation", "method": "WLS", "threads": 3, "failed_voxels": 0,
                "teem_version": "1.12.0"}


def fake_teem_run(commands):
    """Stand-in for ``_run_cli``: record the command and write what teem_dti.py would write."""

    def execute(command, log=None):
        commands.append(list(command))
        arguments = command[2:]
        if "--receipt" in arguments:
            Path(arguments[arguments.index("--receipt") + 1]).write_text(json.dumps(FAKE_RECEIPT), encoding="utf-8")
        targets = [item.partition("=")[2] for index, item in enumerate(arguments)
                   if index and arguments[index - 1] == "--scalar"]
        for target in [*arguments[-2:], *targets]:
            Path(target).write_bytes(b"fixture output")

    return execute


class NativeTensorBundleTests(unittest.TestCase):
    def build(self, fit="wls"):
        commands = []
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        raw = inputs(Path(tmp.name))
        with patch.object(pipeline, "_require_slicer"), \
                patch.object(pipeline, "resolve_pythonslicer_executable", return_value="/slicer/bin/PythonSlicer"), \
                patch.object(pipeline, "_run_cli", side_effect=fake_teem_run(commands)), \
                patch.object(pipeline, "_convert_nrrd_to_nifti",
                             side_effect=lambda source, target: target.write_bytes(source.read_bytes())):
            bundle = pipeline.ensure_native_dti_bundle(raw, fit)
        return raw, bundle, commands

    def test_w1_one_teem_process_writes_tensor_baseline_and_six_scalar_maps(self) -> None:
        _raw, bundle, commands = self.build("wls")
        self.assertEqual(len(commands), 1)
        command = commands[0]
        self.assertEqual(command[0], "/slicer/bin/PythonSlicer")
        self.assertEqual(Path(command[1]), TEEM_SCRIPT)
        self.assertEqual(command[2:5], ["estimate", "--enumeration", "WLS"])
        self.assertEqual(command[5:7], ["--receipt", str(bundle.bundle_dir / "native_dti_receipt.json")])
        self.assertNotIn("--threads", command)
        scalars = dict(item.split("=", 1) for index, item in enumerate(command) if command[index - 1] == "--scalar")
        self.assertEqual(scalars, {
            "FractionalAnisotropy": str(bundle.fa_nrrd_path), "Trace": str(bundle.trace_nrrd_path),
            "MeanDiffusivity": str(bundle.mean_diffusivity_nrrd_path),
            "MinEigenvalue": str(bundle.min_eig_nrrd_path), "MidEigenvalue": str(bundle.mid_eig_nrrd_path),
            "MaxEigenvalue": str(bundle.max_eig_nrrd_path)})
        self.assertEqual(command[-3:], [str(bundle.dwi_nhdr_path), str(bundle.dti_nrrd_path), str(bundle.b0_nrrd_path)])

    def test_n5_the_bundle_command_takes_its_scalar_names_from_the_shared_constant(self) -> None:
        reversed_names = tuple(reversed(SIX_SCALARS))
        with patch.object(pipeline, "NATIVE_TENSOR_SCALARS", reversed_names):
            _raw, _bundle, commands = self.build("wls")
        names = [item.split("=", 1)[0] for index, item in enumerate(commands[0]) if commands[0][index - 1] == "--scalar"]
        self.assertEqual(tuple(names), reversed_names)

    def test_w1_ls_fit_uses_the_ls_enumeration(self) -> None:
        _raw, _bundle, commands = self.build("ls")
        self.assertEqual(commands[0][4], "LS")

    def test_w1_compatibility_copies_and_artifacts_are_still_written(self) -> None:
        _raw, bundle, _commands = self.build()
        for copy, source in ((bundle.ddseg_md_nrrd_path, bundle.trace_nrrd_path),
                             (bundle.ddseg_e1_nrrd_path, bundle.max_eig_nrrd_path),
                             (bundle.ddseg_e2_nrrd_path, bundle.mid_eig_nrrd_path),
                             (bundle.ddseg_e3_nrrd_path, bundle.min_eig_nrrd_path)):
            self.assertEqual(copy.read_bytes(), source.read_bytes())
        self.assertTrue(pipeline._native_outputs_complete(bundle))

    def test_d5_timing_record_carries_the_receipt_and_drops_the_itk_keys(self) -> None:
        _raw, bundle, _commands = self.build()
        receipt = json.loads((bundle.bundle_dir / "native_dti_receipt.json").read_text(encoding="utf-8"))
        timing = json.loads((bundle.bundle_dir / "native_dti_timing.json").read_text(encoding="utf-8"))
        self.assertEqual(timing["tensor_fit"], receipt)
        self.assertEqual(timing["tensor_fit"], FAKE_RECEIPT)
        self.assertNotIn("itk_default_threads", timing)
        self.assertNotIn("thread_control_scope", timing)
        self.assertIn("native Teem tensor fit and six scalar maps in one process", timing["timing_scope"])

    def test_d4_manifest_version_4_names_the_tensor_implementation(self) -> None:
        raw, bundle, _commands = self.build()
        manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["version"], 4)
        self.assertEqual(manifest["tensor_implementation"], hashlib.sha256(TEEM_SCRIPT.read_bytes()).hexdigest())
        self.assertEqual(pipeline._native_manifest(raw, bundle)["tensor_implementation"],
                         manifest["tensor_implementation"])

    def test_d4_a_version_3_cache_is_rebuilt(self) -> None:
        raw, bundle, commands = self.build()
        manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
        manifest.update(version=3)
        manifest.pop("tensor_implementation")
        bundle.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with patch.object(pipeline, "_require_slicer"), \
                patch.object(pipeline, "resolve_pythonslicer_executable", return_value="/slicer/bin/PythonSlicer"), \
                patch.object(pipeline, "_run_cli", side_effect=fake_teem_run(commands)), \
                patch.object(pipeline, "_convert_nrrd_to_nifti",
                             side_effect=lambda source, target: target.write_bytes(source.read_bytes())):
            pipeline.ensure_native_dti_bundle(raw, "wls")
        self.assertEqual(len(commands), 2)


class _StopAfterTensorStep(Exception):
    pass


class DDParcelTensorStepTests(unittest.TestCase):
    def test_w2_ddparcel_runs_one_teem_process_with_nhdr_outputs(self) -> None:
        from anatomy.parcellation import pipeline as parcel

        commands = []

        def stop_after_first(command, log=None):
            commands.append(list(command))
            raise _StopAfterTensorStep

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            diffusion = root / "in" / "sub" / "T1w" / "Diffusion"
            diffusion.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                (diffusion / name).write_bytes(b"x")
            atlas = root / "atlas.nii.gz"
            atlas.write_bytes(b"x")
            config = parcel.DDParcelConfig("sub", root / "in", root / "out", atlas)
            with patch("anatomy.parcellation.conversion.nhdr_write.nhdr_write"), \
                    patch.object(parcel, "resolve_pythonslicer_executable", return_value="/slicer/bin/PythonSlicer",
                                 create=True), \
                    patch.object(parcel, "_run_cli", side_effect=stop_after_first):
                with self.assertRaises(_StopAfterTensorStep):
                    parcel.run_ddparcel(config)
        out = root / "out" / "sub"
        command = commands[0]
        self.assertEqual(command[0], "/slicer/bin/PythonSlicer")
        self.assertEqual(Path(command[1]), TEEM_SCRIPT)
        self.assertEqual(command[2:5], ["estimate", "--enumeration", "WLS"])
        scalars = dict(item.split("=", 1) for index, item in enumerate(command) if command[index - 1] == "--scalar")
        self.assertEqual(set(scalars), set(SIX_SCALARS))
        self.assertEqual(scalars["MeanDiffusivity"], str(out / "sub-dti-MeanDiffusivity.nhdr"))
        self.assertEqual(command[-3:], [str(root / "in/sub/T1w/Diffusion/sub.nhdr"),
                                        str(out / "sub-dti.nhdr"), str(out / "sub-b0.nhdr")])


class RegisteredStageAdmissionTests(unittest.TestCase):
    """S1.2: native_dti is admitted as its own stage before registered_dti, which covers registration only."""

    def run_registered(self, *, native_ready=False, cohort=True):
        from contextlib import contextmanager

        events = []
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        raw = inputs(Path(tmp.name))
        atlas = Path(tmp.name) / "atlas.nii.gz"
        atlas.write_bytes(b"atlas")

        @contextmanager
        def admission(name, *, gpu=False):
            events.append(("enter", name))
            try:
                yield None
            finally:
                events.append(("exit", name))

        teem = fake_teem_run([])

        def run_cli(command, log=None):
            if str(command[0]).endswith("PythonSlicer"):
                events.append(("cli", "teem"))
                teem(command, log)
            elif command[0] == "brainsfit":
                events.append(("cli", "brainsfit"))
                Path(command[command.index("--linearTransform") + 1]).write_bytes(b"tfm")
            else:
                events.append(("cli", "resample"))
                Path(command[-1]).write_bytes(b"registered")

        patches = [
            patch.object(pipeline, "_require_slicer"),
            patch.object(pipeline, "resolve_pythonslicer_executable", return_value="/slicer/bin/PythonSlicer"),
            patch.object(pipeline, "_get_cli_executable", side_effect=lambda name: name),
            patch.object(pipeline, "_run_cli", side_effect=run_cli),
            patch.object(pipeline, "_write_inverse_transform",
                         side_effect=lambda forward, inverse: inverse.write_bytes(b"inverse")),
            patch.object(pipeline, "_convert_nrrd_to_nifti",
                         side_effect=lambda source, target: target.write_bytes(source.read_bytes())),
            patch("shared.cohort_schedule.mark_reused", side_effect=lambda stage: events.append(("reused", stage))),
        ]
        if cohort:
            patches.append(patch("shared.cohort_schedule.stage_admission", admission))
        from contextlib import ExitStack
        with ExitStack() as stack:
            for item in patches:
                stack.enter_context(item)
            if native_ready:
                pipeline.ensure_native_dti_bundle(raw, "wls")
                events.clear()
            bundle = pipeline.ensure_registered_dti_bundle(raw, atlas, "wls")
            first = list(events)
            events.clear()
            again = pipeline.ensure_registered_dti_bundle(raw, atlas, "wls")
        self.assertEqual(again, bundle)
        return bundle, first, list(events)

    def test_s1_2_fresh_native_fit_is_admitted_and_released_before_registered_dti(self) -> None:
        bundle, events, _ = self.run_registered()
        self.assertEqual(events[0], ("enter", "native_dti"))
        self.assertLess(events.index(("exit", "native_dti")), events.index(("enter", "registered_dti")))
        self.assertLess(events.index(("cli", "teem")), events.index(("exit", "native_dti")))
        self.assertLess(events.index(("enter", "registered_dti")), events.index(("cli", "brainsfit")))
        self.assertEqual(events[-1], ("exit", "registered_dti"))
        self.assertEqual([item for item in events if item[0] == "enter"],
                         [("enter", "native_dti"), ("enter", "registered_dti")])
        self.assertTrue(bundle.fa_reg_path.is_file())

    def test_s1_2_a_reused_native_bundle_admits_only_registered_dti_and_marks_the_reuse(self) -> None:
        _, events, _ = self.run_registered(native_ready=True)
        self.assertEqual([item for item in events if item[0] == "enter"],
                         [("enter", "native_dti"), ("enter", "registered_dti")])
        self.assertNotIn(("cli", "teem"), events)
        self.assertIn(("reused", "native_dti"), events)
        self.assertLess(events.index(("exit", "native_dti")), events.index(("enter", "registered_dti")))

    def test_s1_2_a_reused_registered_bundle_is_marked_inside_its_own_admission(self) -> None:
        _, _, second = self.run_registered()
        self.assertEqual(second, [("enter", "native_dti"), ("reused", "native_dti"), ("exit", "native_dti"),
                                  ("enter", "registered_dti"), ("reused", "registered_dti"), ("exit", "registered_dti")])

    def test_s1_2_outside_cohort_mode_the_calls_and_outputs_are_unchanged(self) -> None:
        bundle, events, second = self.run_registered(cohort=False)
        self.assertEqual([item for item in events if item[0] == "cli"],
                         [("cli", "teem"), ("cli", "brainsfit")] + [("cli", "resample")] * 7)
        self.assertFalse([item for item in events if item[0] in ("enter", "exit")])
        self.assertEqual([item for item in second if item[0] == "cli"], [])
        self.assertTrue(bundle.inverse_transform_path.is_file())

    def test_s1_2_a_missing_atlas_fails_before_any_native_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            raw = inputs(Path(tmp))
            with patch.object(pipeline, "ensure_native_dti_bundle",
                              side_effect=AssertionError("native fit must not start")):
                with self.assertRaisesRegex(FileNotFoundError, "Atlas T2 not found"):
                    pipeline.ensure_registered_dti_bundle(raw, Path(tmp) / "absent.nii.gz")


class DDParcelCompletenessTests(unittest.TestCase):
    """N3: the DDParcel tensor step reruns when any scalar file it needs is missing."""

    def first_command(self, remove=None):
        from anatomy.parcellation import pipeline as parcel

        commands = []

        def stop_after_first(command, log=None):
            commands.append(list(command))
            raise _StopAfterTensorStep

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            diffusion = root / "in" / "sub" / "T1w" / "Diffusion"
            diffusion.mkdir(parents=True)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz", "sub-mask.nhdr"):
                (diffusion / name).write_bytes(b"x")
            out = root / "out" / "sub"
            out.mkdir(parents=True)
            (out / "sub-b0.nhdr").write_bytes(b"x")
            for name in SIX_SCALARS:
                for suffix in (".nhdr", ".nii.gz"):
                    (out / f"sub-dti-{name}{suffix}").write_bytes(b"x")
            if remove:
                (out / remove).unlink()
            atlas = root / "atlas.nii.gz"
            atlas.write_bytes(b"x")
            config = parcel.DDParcelConfig("sub", root / "in", root / "out", atlas)
            with patch.object(parcel, "resolve_pythonslicer_executable", return_value="/slicer/bin/PythonSlicer",
                              create=True), \
                    patch.object(parcel, "resolve_cli_executable", return_value="/cli/BRAINSFit"), \
                    patch.object(parcel, "_run_cli", side_effect=stop_after_first):
                with self.assertRaises(_StopAfterTensorStep):
                    parcel.run_ddparcel(config)
        return commands[0]

    def test_n3_complete_outputs_skip_the_tensor_fit(self) -> None:
        self.assertNotEqual(Path(self.first_command()[1]), TEEM_SCRIPT)

    def test_n3_a_missing_scalar_header_refits(self) -> None:
        self.assertEqual(Path(self.first_command(remove="sub-dti-Trace.nhdr")[1]), TEEM_SCRIPT)

    def test_n3_a_missing_scalar_nifti_refits(self) -> None:
        self.assertEqual(Path(self.first_command(remove="sub-dti-MaxEigenvalue.nii.gz")[1]), TEEM_SCRIPT)


class DetachedHeaderTests(unittest.TestCase):
    """W2: vtkTeemNRRDWriter writes a detached header plus data file for a .nhdr name; nifti_write reads it."""

    slicer_python = Path("/Applications/Slicer.app/Contents/bin/PythonSlicer")

    @unittest.skipUnless(slicer_python.is_file(), "no local Slicer 5.12 at /Applications/Slicer.app")
    def test_w2_nhdr_outputs_are_detached_and_readable_by_nifti_write(self) -> None:
        import nibabel as nib
        import nrrd
        from anatomy.parcellation.conversion.nifti_write import nifti_write

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = inputs(root)
            header = pipeline.ensure_diffusion_headers(raw)[0]
            out = root / "out"
            out.mkdir()
            tensor, baseline = out / "x-dti.nhdr", out / "x-b0.nhdr"
            scalars = [(name, out / f"x-dti-{name}.nhdr") for name in SIX_SCALARS]
            command = pipeline.native_tensor_command(self.slicer_python, header, tensor, baseline, "WLS",
                                                     scalars, out / "receipt.json")
            completed = subprocess.run(command, capture_output=True, text=True, timeout=300)
            if completed.returncode == 2 and "Teem" in completed.stderr:
                self.skipTest("Teem library is not usable in this Slicer: " + completed.stderr.strip())
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertEqual(json.loads((out / "receipt.json").read_text())["failed_voxels"], 0)
            for path in (tensor, baseline, *(path for _, path in scalars)):
                data_file = nrrd.read_header(str(path)).get("data file")
                self.assertTrue(data_file, f"{path.name} has no detached data file")
                self.assertTrue((path.parent / data_file).is_file(), data_file)
            fa = scalars[0][1]
            nifti_write(str(fa), str(fa).replace(".nhdr", ""))
            self.assertEqual(nib.load(str(fa).replace(".nhdr", ".nii.gz")).shape, (4, 5, 6))


class LoadedModuleTupleTests(unittest.TestCase):
    """D7: the Slicer module probe no longer asks for dwitodtiestimation; dwiconvert and ukftractography stay."""

    modules = types.SimpleNamespace(dwitodtiestimation=object(), dwiconvert=object(), ukftractography=object())

    def test_d7_gui_check_request_lists_only_dwiconvert_and_ukftractography(self) -> None:
        from shared import ui_environment_check

        slicer = types.SimpleNamespace(modules=self.modules)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            owner = types.SimpleNamespace(
                _pipelineActive=False, _update_environment_result_label=lambda *args, **kwargs: None,
                _environment_check_kwargs=lambda: {}, _environment_check_signature=lambda options: (),
                _environment_report_dir_from_kwargs=lambda options: folder,
                _setStatus=lambda *args: None, appendLog=lambda message: None,
                logic=types.SimpleNamespace(module_dir=REPO_ROOT), _slicer_launcher_program=lambda: "Slicer")
            with patch.dict(sys.modules, {"slicer": slicer}), \
                    patch("shared.ui_setup_jobs.start_setup_job") as start, \
                    patch("shared.environment_check.check_config", side_effect=lambda *a, **k: captured.update(k) or {}):
                captured = {}
                ui_environment_check.start_environment_check(owner)
            start.assert_called_once()
        self.assertEqual(captured["loaded_modules"], ("dwiconvert", "ukftractography"))

    def test_d7_logic_doctors_pass_only_dwiconvert_and_ukftractography(self) -> None:
        stub = types.ModuleType("slicer")
        with patch.dict(sys.modules, {"slicer": stub}):
            import shared.module_logic as logic_module
        fake = types.SimpleNamespace(app=object(), modules=self.modules)
        logic = type("Logic", (logic_module.SharedLogicMixin,), {"log": lambda self, message: None})()
        logic.module_dir = REPO_ROOT
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("runEnvironmentDoctor", "runEnvironmentDependencyCheck"):
                with self.subTest(method=name), patch.object(logic_module, "slicer", fake), \
                        patch.object(logic_module, "run_environment_doctor") as check, \
                        patch.object(logic_module, "summarize_environment_doctor", return_value="Checked"), \
                        patch.object(logic_module, "write_environment_report", return_value={
                            key: root / key for key in ("json", "python_sh", "external_notes")}):
                    getattr(logic, name)(targets=("microstructure",), subject_root=root, output_root=root)
                    self.assertEqual(check.call_args.args[0].loaded_slicer_modules, ("dwiconvert", "ukftractography"))


class PackagingTests(unittest.TestCase):
    """D9: the extension depends on PyTorch only; the package audits agree."""

    def test_d9_cmake_and_s4ext_depend_on_pytorch_only(self) -> None:
        cmake = (REPO_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        s4ext = (REPO_ROOT / "SlicerDSpace.s4ext").read_text(encoding="utf-8")
        self.assertIn("set(EXTENSION_DEPENDS PyTorch)", cmake)
        self.assertEqual([line for line in s4ext.splitlines() if line.startswith("depends ")], ["depends PyTorch"])

    def test_d9_audit_scripts_expect_the_same_dependency_list(self) -> None:
        from scripts import audit_slicer_extension_metadata
        self.assertEqual(audit_slicer_extension_metadata.REQUIRED_EXTENSION_DEPENDS, ("PyTorch",))
        audit_source = (REPO_ROOT / "scripts/release_audit.py").read_text(encoding="utf-8")
        self.assertIn('"depends PyTorch",', audit_source)
        self.assertNotIn("depends SlicerDMRI", audit_source)
        result = audit_slicer_extension_metadata.audit(REPO_ROOT)
        self.assertEqual((result["s4ext_depends"], result["cmake_depends"]), (["PyTorch"], ["PyTorch"]))
        self.assertEqual(result["status"], "pass", result["findings"])


class SelfCheckRequestTests(unittest.TestCase):
    """D10: the setup functional check runs the Teem self-check in the Slicer interpreter only."""

    @staticmethod
    def report(statuses):
        return types.SimpleNamespace(statuses=statuses, targets=("microstructure",))

    @staticmethod
    def teem(python, status="available"):
        return types.SimpleNamespace(key="native_tensor_fit", kind="setting", required=True, status=status,
                                     targets=("microstructure",), details={"python_executable": python})

    @staticmethod
    def numpy(python):
        return types.SimpleNamespace(key="python:numpy", kind="python", required=True, status="available",
                                     targets=("microstructure",), details={"import": "numpy", "python_executable": python})

    def test_d10_only_the_slicer_interpreter_gets_the_teem_self_check(self) -> None:
        from shared.setup_selftest import requests_for_report

        requests = requests_for_report(self.report([self.teem("/slicer/PythonSlicer"), self.numpy("/slicer/PythonSlicer"),
                                                    self.numpy("/venv/bin/python")]))
        flags = {request["python"]: request.get("teem_self_check", False) for request in requests}
        self.assertEqual(flags, {"/slicer/PythonSlicer": True, "/venv/bin/python": False})

    def test_d10_the_check_runs_with_no_other_package_group(self) -> None:
        from shared.setup_selftest import requests_for_report

        requests = requests_for_report(self.report([self.teem("/slicer/PythonSlicer")]))
        self.assertEqual([(request["python"], request["teem_self_check"]) for request in requests],
                         [("/slicer/PythonSlicer", True)])

    def test_d10_a_missing_teem_adds_no_check(self) -> None:
        from shared.setup_selftest import requests_for_report

        self.assertEqual(requests_for_report(self.report([self.teem("/plain/python", status="missing")])), [])

    def test_d10_run_checks_reports_the_self_check_json(self) -> None:
        from shared import setup_selftest, teem_dti

        record = {"status": "ok", "teem_version": "1.12.0", "WLS": {"max_tensor_error": 0.0}}
        with patch.object(teem_dti, "self_check", return_value=record):
            result = setup_selftest.run_checks({"teem_self_check": True})
        self.assertEqual(result["status"], "passed")
        self.assertIn("teem:self_check", result["checks"])
        self.assertEqual(result["teem_self_check"], record)

    def test_d10_a_failed_self_check_fails_the_functional_check(self) -> None:
        from shared import setup_selftest, teem_dti

        with patch.object(teem_dti, "self_check", side_effect=teem_dti.TeemError("Teem WLS self-check failed")):
            with self.assertRaisesRegex(teem_dti.TeemError, "self-check failed"):
                setup_selftest.run_checks({"teem_self_check": True})
        with patch.object(teem_dti, "self_check") as check:
            setup_selftest.run_checks({"imports": []})
        check.assert_not_called()


class LicenceTests(unittest.TestCase):
    """D11: the SlicerDMRI licence ships with the plugin and the notice states the derivation."""

    LICENCE_SHA256 = "03fc5cd907e5c2004ce199d66318d32a08fa2de2b5b28bee184edda6307d0b86"

    def test_d11_licence_text_provenance_notice_and_package_inclusion(self) -> None:
        from scripts import audit_third_party
        from scripts.extension_runtime.runtime_resources import runtime_files

        licence = REPO_ROOT / "licenses/SlicerDMRI.txt"
        self.assertEqual(hashlib.sha256(licence.read_bytes()).hexdigest(), self.LICENCE_SHA256)
        records = {record["id"]: record for record in
                   json.loads((REPO_ROOT / "licenses/provenance.json").read_text(encoding="utf-8"))["records"]}
        record = records["slicerdmri_tensor_fit"]
        self.assertEqual((record["path"], record["sha256"], record["covers"]),
                         ("licenses/SlicerDMRI.txt", self.LICENCE_SHA256, ["shared/teem_dti.py"]))
        self.assertRegex(record["source"], r"https://github\.com/SlicerDMRI/SlicerDMRI/blob/a40200b[0-9a-f]{33}/License\.txt")
        notice = (REPO_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
        for phrase in ("shared/teem_dti.py", "modified Python reimplementation", "vtkTeemEstimateDiffusionTensor",
                       "a40200b", "licenses/SlicerDMRI.txt", "LGPL-2.1 with a static-linking exception",
                       "does not redistribute Teem"):
            self.assertIn(phrase, notice)
        self.assertIn("licenses/SlicerDMRI.txt", runtime_files(REPO_ROOT))
        result = audit_third_party.audit(REPO_ROOT)
        self.assertEqual(result["status"], "pass", result["findings"])
        self.assertIn("slicerdmri_tensor_fit", result["checked"])


if __name__ == "__main__":
    unittest.main()
