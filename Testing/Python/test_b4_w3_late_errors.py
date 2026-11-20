#!/usr/bin/env python3
"""B4 W3: errors that surfaced hours into a run are now checked before it (LE-1 .. LE-10, X7, X10)."""

from __future__ import annotations

import errno
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _plan(subject: Path, **override):
    from shared.orchestrator import ProductOrchestratorConfig, plan_subject_products

    values = dict(subject_id="100307", subject_root=subject, targets=("microstructure", "microstructure_noddi"), execute=False)
    values.update(override)
    return plan_subject_products(ProductOrchestratorConfig(**values), write=False)


class _Subject(unittest.TestCase):
    def setUp(self):
        from test_product_orchestrator import _write_valid_raw_subject

        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.subject = Path(self.temporary.name) / "100307"
        _write_valid_raw_subject(self.subject)


class Le1NoddiShellsTest(_Subject):
    def test_le1_single_shell_blocks_noddi_with_the_shell_message(self):
        (self.subject / "input" / "bvals").write_text("0 1000 1000 1000")
        items = {item.target: item for item in _plan(self.subject).items}
        self.assertEqual(items["microstructure_noddi"].status, "blocked")
        self.assertEqual(
            items["microstructure_noddi"].reason,
            "NODDI needs at least 2 non-zero b-shells; this DWI has 1 (b=1000). Deselect NODDI or use multi-shell data.",
        )
        self.assertEqual(items["microstructure"].status, "will_run")  # independent targets keep running

    def test_le1_two_shells_keep_noddi_runnable(self):
        (self.subject / "input" / "bvals").write_text("0 1000 2000 2000")
        items = {item.target: item for item in _plan(self.subject).items}
        self.assertEqual(items["microstructure_noddi"].status, "will_run")

    def test_le1_shell_count_uses_the_pipeline_rounding(self):
        from microstructure.pipeline import noddi_nonzero_shells

        # 1000 and 1049 round to the same 100 s/mm^2 shell, as in the NODDI stage.
        self.assertEqual(noddi_nonzero_shells([0, 1000, 1049, 2000], 50), [1000, 2000])
        self.assertEqual(noddi_nonzero_shells([0, 1000, 1049], 50), [1000])
        (self.subject / "input" / "bvals").write_text("0 1000 1049 1049")
        items = {item.target: item for item in _plan(self.subject).items}
        self.assertEqual(items["microstructure_noddi"].status, "blocked")


class Le2IncompleteWeightsTest(unittest.TestCase):
    NAMES = ("dti_axial.pt", "dti_coronal.pt", "dti_sagittal.pt")

    def setUp(self):
        from shared import model_assets

        self.model_assets = model_assets
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.tmp = Path(self.temporary.name)
        self.repo = self.tmp / "repo"
        self.assets = self.tmp / "assets"
        (self.repo / "models").mkdir(parents=True)
        records = {f"segmentation/weights/{name}": {"sha256": "0" * 64, "size_bytes": 8} for name in self.NAMES}
        (self.repo / "models" / "assets_manifest.json").write_text(json.dumps({"asset_records": records}), encoding="utf-8")
        self.weights = self.assets / "segmentation" / "weights"
        self.weights.mkdir(parents=True)
        self.records = records
        patcher = mock.patch.object(model_assets, "verify_against_manifest", return_value=(True, "ok"))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_le2_folder_with_one_of_three_files_is_not_accepted(self):
        (self.weights / "dti_axial.pt").write_bytes(b"12345678")
        self.assertFalse(self.model_assets._has_verified_required_asset(
            self.assets, ("segmentation/weights",), manifest_records=self.records))
        problems = self.model_assets.incomplete_manifest_files(self.assets, "segmentation/weights", self.records)
        self.assertEqual(problems, ["dti_coronal.pt (missing)", "dti_sagittal.pt (missing)"])

    def test_le2_complete_folder_is_accepted_and_a_wrong_size_is_not(self):
        for name in self.NAMES:
            (self.weights / name).write_bytes(b"12345678")
        self.assertTrue(self.model_assets._has_verified_required_asset(
            self.assets, ("segmentation/weights",), manifest_records=self.records))
        (self.weights / "dti_sagittal.pt").write_bytes(b"123")  # interrupted download: short file
        self.assertFalse(self.model_assets._has_verified_required_asset(
            self.assets, ("segmentation/weights",), manifest_records=self.records))

    def test_le2_model_asset_path_names_missing_files_and_install_action(self):
        (self.weights / "dti_axial.pt").write_bytes(b"12345678")
        with mock.patch.object(self.model_assets, "default_user_model_assets_root", return_value=self.assets):
            with self.assertRaises(self.model_assets.ModelAssetIncompleteError) as caught:
                self.model_assets.model_asset_path("segmentation/weights", repo_root=self.repo)
        message = str(caught.exception)
        self.assertIn("dti_coronal.pt", message)
        self.assertIn("dti_sagittal.pt", message)
        self.assertIn("Install models", message)

    def test_le2_doctor_rule_reports_incomplete_folder_for_an_explicit_root(self):
        from shared.environment import _model_asset_spec_path

        (self.weights / "dti_axial.pt").write_bytes(b"12345678")
        config = SimpleNamespace(repo_root=self.repo, model_assets_dir=self.assets)
        path, common, notes = _model_asset_spec_path(config, "segmentation/weights")
        self.assertFalse(path.exists())
        self.assertEqual(common, ())
        self.assertIn("dti_coronal.pt", notes)
        self.assertIn("Install models", notes)

    def test_le2_doctor_rule_accepts_the_complete_folder(self):
        from shared.environment import _model_asset_spec_path

        for name in self.NAMES:
            (self.weights / name).write_bytes(b"12345678")
        config = SimpleNamespace(repo_root=self.repo, model_assets_dir=self.assets)
        path, _common, notes = _model_asset_spec_path(config, "segmentation/weights")
        self.assertEqual(path, self.weights.resolve())
        self.assertEqual(notes, "")

    def test_le2_doctor_status_message_reaches_the_user(self):
        from shared.environment import _check_spec, _model_asset_spec_path, _path

        (self.weights / "dti_axial.pt").write_bytes(b"12345678")
        config = SimpleNamespace(repo_root=self.repo, model_assets_dir=self.assets)
        path, common, notes = _model_asset_spec_path(config, "segmentation/weights")
        specs = {}
        _path(specs, target="segmentation", key="ddseg_weights", label="DDSeg weights directory", path=path,
              path_kind="nonempty_dir", group="model-resources", notes=notes,
              discovery_source="model asset root", common_paths=common)
        status = _check_spec(next(iter(specs.values())), "Linux")
        self.assertEqual(status.status, "missing")
        for text in (status.install_hint, status.details["notes"]):
            self.assertIn("dti_coronal.pt", text)
            self.assertIn("Install models", text)

    def test_le2_plan_blocks_segmentation_for_a_partial_explicit_root(self):
        from test_product_orchestrator import _write_valid_raw_subject

        subject = self.tmp / "100307"
        _write_valid_raw_subject(subject)
        real = json.loads((REPO_ROOT / "models" / "assets_manifest.json").read_text())["asset_records"]
        (self.assets / "segmentation" / "weights" / "dti_axial.pt").write_bytes(b"x" * real["segmentation/weights/dti_axial.pt"]["size_bytes"])
        items = {item.target: item for item in _plan(subject, targets=("segmentation", "microstructure"), model_assets_dir=self.assets).items}
        self.assertEqual(items["segmentation"].status, "blocked")
        self.assertIn("dti_coronal.pt", items["segmentation"].reason)
        self.assertIn("Install models", items["segmentation"].reason)
        self.assertEqual(items["microstructure"].status, "will_run")

    def test_le2_real_manifest_lists_three_views_for_ddseg_and_ddparcel(self):
        records = self.model_assets._load_asset_manifest_records(REPO_ROOT)
        for prefix in ("segmentation/weights", "parcellation/weights"):
            names = " ".join(key for key in records if key.startswith(prefix + "/"))
            for view in ("axial", "sagittal", "coronal"):
                self.assertIn(view, names, prefix)


class Le3SurfaceGpuTest(_Subject):
    def test_le3_decision_cases(self):
        from shared import environment

        with mock.patch.object(environment, "_is_macos", return_value=False), \
                mock.patch.object(environment, "cuda_free_gib", return_value=6.5):
            action, note = environment.surface_gpu_memory_decision("auto")
            self.assertEqual(action, "cpu")
            self.assertIn("6.5 GiB", note)
            action, reason = environment.surface_gpu_memory_decision("cuda:0")
            self.assertEqual(action, "block")
            self.assertEqual(
                reason,
                "DDSurfer needs at least 12 GiB free GPU memory (found 6.5 GiB). "
                "Set the surface device to CPU or free GPU memory.",
            )
            self.assertEqual(environment.surface_gpu_memory_decision("cpu"), ("ok", ""))
        with mock.patch.object(environment, "_is_macos", return_value=False), \
                mock.patch.object(environment, "cuda_free_gib", return_value=20.0):
            self.assertEqual(environment.surface_gpu_memory_decision("auto"), ("ok", ""))
        with mock.patch.object(environment, "_is_macos", return_value=False), \
                mock.patch.object(environment, "cuda_free_gib", return_value=None):
            self.assertEqual(environment.surface_gpu_memory_decision("cuda:0"), ("ok", ""))  # unknown never blocks

    def test_le3_free_memory_probe_is_cached(self):
        from shared import environment

        environment._CUDA_FREE_CACHE.clear()
        completed = SimpleNamespace(stdout='{"free_gib": 9.25}')
        with mock.patch.object(environment.subprocess, "run", return_value=completed) as run:
            self.assertEqual(environment.cuda_free_gib("cuda:0", Path("/x/python")), 9.25)
            self.assertEqual(environment.cuda_free_gib("cuda:0", Path("/x/python")), 9.25)
        self.assertEqual(run.call_count, 1)
        environment._CUDA_FREE_CACHE.clear()

    def test_le3_plans_spawn_no_probe_and_none_on_macos(self):
        # U5 (batch-4 fix round): a plan runs on the GUI thread and never starts the torch probe; it reads only a
        # cached result. This test pinned "two plans spawn one probe" before that change.
        from shared import environment

        environment._CUDA_FREE_CACHE.clear()
        self.addCleanup(environment._CUDA_FREE_CACHE.clear)
        completed = SimpleNamespace(stdout='{"free_gib": 40.0}')
        with mock.patch.object(environment, "_is_macos", return_value=False), \
                mock.patch.object(environment.subprocess, "run", return_value=completed) as run:
            first = _plan(self.subject, targets=("surface",))
            second = _plan(self.subject, targets=("surface",))
        self.assertEqual(run.call_count, 0)
        self.assertEqual(first.items[0].status, "will_run")
        self.assertEqual(second.items[0].status, "will_run")
        with mock.patch.object(environment, "_is_macos", return_value=True), \
                mock.patch.object(environment.subprocess, "run") as run:
            environment._CUDA_FREE_CACHE.clear()
            _plan(self.subject, targets=("surface",))
        run.assert_not_called()

    def test_le3_plan_blocks_surface_for_an_explicit_cuda_device(self):
        with mock.patch("shared.orchestrator._surface_gpu_decision", return_value=("block", "DDSurfer needs at least 12 GiB free GPU memory (found 3.0 GiB). Set the surface device to CPU or free GPU memory.")):
            items = {item.target: item for item in _plan(self.subject, targets=("surface", "microstructure")).items}
        self.assertEqual(items["surface"].status, "blocked")
        self.assertIn("12 GiB", items["surface"].reason)
        self.assertEqual(items["microstructure"].status, "will_run")

    def test_le3_plan_notes_cpu_for_auto_device(self):
        with mock.patch("shared.orchestrator._surface_gpu_decision", return_value=("cpu", "only 6.0 GiB is free; the surface stage runs on CPU.")):
            items = {item.target: item for item in _plan(self.subject, targets=("surface",)).items}
        self.assertEqual(items["surface"].status, "will_run")
        self.assertIn("on CPU", items["surface"].reason)

    def test_le3_run_passes_cpu_to_the_surface_config_for_this_run_only(self):
        from shared.orchestrator import ProductOrchestratorConfig, run_subject_products

        captured = {}

        def fake_config(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(**kwargs)

        logs = []
        config = ProductOrchestratorConfig(subject_id="100307", subject_root=self.subject, targets=("surface",),
                                           execute=True, surface_device="auto")
        with mock.patch("shared.orchestrator._surface_gpu_decision", return_value=("cpu", "only 6.0 GiB is free; the surface stage runs on CPU.")), \
                mock.patch("anatomy.surface.SurfaceConfig", side_effect=fake_config), \
                mock.patch("anatomy.surface.inspect_surface_resources", return_value={"available": False, "missing": ["x"]}):
            result = run_subject_products(config, log=logs.append)
        self.assertEqual(captured["device"], "cpu")
        self.assertEqual(config.surface_device, "auto")
        self.assertTrue(any("runs on CPU" in line for line in logs))
        self.assertEqual(result.tasks[-1].status, "blocked")  # the stub resources are missing; only the device matters here


class Le4DiskTest(_Subject):
    def test_le4_estimate_adds_streamline_bytes_for_tractography(self):
        from shared.orchestrator import estimate_run_disk_bytes

        base = estimate_run_disk_bytes(1000, ["microstructure"], 100000)
        with_tracts = estimate_run_disk_bytes(1000, ["microstructure", "tractography"], 100000)
        self.assertEqual(base, 1000)
        self.assertEqual(with_tracts, 1000 + 6000 + 100000 * 2048 * 3)

    def test_le4_low_free_space_warns_and_never_blocks(self):
        usage = SimpleNamespace(total=10**12, used=10**12 - 10, free=10)
        with mock.patch("shared.orchestrator.shutil.disk_usage", return_value=usage):
            plan = _plan(self.subject)
        self.assertEqual(len(plan.warnings), 1)
        self.assertIn("GB (estimate)", plan.warnings[0])
        self.assertIn("GB free", plan.warnings[0])
        self.assertIn("not blocked", plan.warnings[0])
        self.assertEqual(plan.blocked_items, tuple(item for item in plan.items if item.status == "blocked"))
        self.assertTrue(any(item.will_run for item in plan.items))  # the estimate does not stop a target

    def test_le4_enough_free_space_gives_no_warning(self):
        usage = SimpleNamespace(total=10**15, used=0, free=10**15)
        with mock.patch("shared.orchestrator.shutil.disk_usage", return_value=usage):
            self.assertEqual(_plan(self.subject).warnings, ())

    def test_le4_enospc_maps_to_the_disk_full_sentence(self):
        from shared.orchestrator import explain_stage_error

        error = OSError(errno.ENOSPC, "No space left on device")
        self.assertEqual(
            explain_stage_error(error, Path("/data/out")),
            "Output disk is full: free space on /data/out and run again.",
        )


class Le6StageFailureTest(unittest.TestCase):
    def test_le6_known_errors_get_a_cause_and_fix(self):
        from shared.orchestrator import explain_stage_error

        self.assertIn("Install models", explain_stage_error(FileNotFoundError("/m/weights/dti_axial.pt")))
        self.assertIn("GPU ran out of memory", explain_stage_error(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")))
        missing_command = explain_stage_error(FileNotFoundError(2, "No such file or directory", "dwi2fod"))
        self.assertIn("'dwi2fod' was not found", missing_command)
        self.assertIn("Environment > Prepare", missing_command)
        self.assertEqual(explain_stage_error(ValueError("anything else")), "")
        # the real raise sites of tractography/act.py and shared/native_runtimes.py
        self.assertIn("'tckgen'", explain_stage_error(RuntimeError("MRtrix3 backend is incomplete. Missing command(s): ['tckgen', 'tcksift2']")))
        self.assertIn("'eddy'", explain_stage_error(ValueError("Missing native executable: eddy")))
        self.assertIn("'eddy'", explain_stage_error(FileNotFoundError(2, "No such file or directory", "C:\\fsl\\bin\\eddy.exe")))

    def test_le6_paths_that_merely_contain_a_command_name_do_not_match(self):
        from shared.orchestrator import explain_stage_error

        for error in (
            FileNotFoundError(2, "No such file or directory", "/out/eddy/corrected.nii.gz"),
            FileNotFoundError("Input not found: /data/eddy.nii.gz"),
            FileNotFoundError(2, "No such file or directory", "/data/bet/mask.nii.gz"),
            RuntimeError("topup output not found in /tmp/topup/"),
        ):
            self.assertEqual(explain_stage_error(error), "", repr(error))

    def test_le6_a_completed_stage_is_not_blamed_for_a_later_error(self):
        from shared import orchestrator

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            config = orchestrator.ProductOrchestratorConfig(
                subject_id="100307", subject_root=subject_root, targets=("microstructure",), run_name="le6b", execute=True)

            def impl(cfg, log=None):
                orchestrator._emit("[Orchestrator] Running Microstructure DTI", log)
                tasks = orchestrator._TimedTaskList(0.0)
                tasks.append(orchestrator.ProductTaskRecord(name="microstructure.dti", status="completed"))
                raise RuntimeError("publish step failed")

            logs = []
            with mock.patch("shared.orchestrator._run_subject_products_impl", side_effect=impl):
                with self.assertRaises(RuntimeError):
                    orchestrator.run_subject_products(config, log=logs.append)
            manifest = json.loads((subject_root / "outputs" / "00_run_metadata" / "le6b_orchestration_manifest.json").read_text())
            self.assertEqual(manifest["failure"]["message"], "publish step failed")
            self.assertNotIn("stage", manifest["failure"])
            self.assertFalse(any("Stage" in line for line in logs))

    def test_le6_failure_text_names_the_stage(self):
        from shared.orchestrator import stage_failure_text

        text = stage_failure_text("microstructure_noddi", RuntimeError("CUDA out of memory"))
        self.assertTrue(text.startswith("Stage microstructure_noddi failed: "))
        self.assertIn("CUDA out of memory", text)
        self.assertEqual(stage_failure_text("X", ValueError("boom")), "Stage X failed: boom")

    def test_le6_failed_run_record_and_log_name_the_stage(self):
        from shared import orchestrator

        with tempfile.TemporaryDirectory() as tmp:
            subject_root = Path(tmp) / "100307"
            config = orchestrator.ProductOrchestratorConfig(
                subject_id="100307", subject_root=subject_root, targets=("microstructure_noddi",),
                run_name="le6", execute=True)

            def impl(cfg, log=None):
                orchestrator._emit("[Orchestrator] Running Microstructure NODDI", log)
                raise OSError(errno.ENOSPC, "No space left on device")

            logs = []
            with mock.patch("shared.orchestrator._run_subject_products_impl", side_effect=impl):
                with self.assertRaises(OSError):
                    orchestrator.run_subject_products(config, log=logs.append)
            manifest = json.loads((subject_root / "outputs" / "00_run_metadata" / "le6_orchestration_manifest.json").read_text())
            self.assertTrue(manifest["failure"]["message"].startswith("Stage microstructure_noddi failed: Output disk is full"))
            self.assertEqual(manifest["failure"]["stage"], "microstructure_noddi")
            self.assertEqual(manifest["failure"]["type"], "OSError")
            self.assertTrue(any(line.startswith("Stage microstructure_noddi failed: ") for line in logs))


class Le9MrtrixCommandsTest(unittest.TestCase):
    def test_le9_csd_act_doctor_lists_5ttcheck_and_5tt2gmwmi(self):
        from shared.environment import EnvironmentDoctorConfig, build_dependency_specs

        specs = {spec.key: spec for spec in build_dependency_specs(
            EnvironmentDoctorConfig(repo_root=REPO_ROOT, targets=("tractography",), tractography_backend="csd_act"))}
        for command in ("5ttcheck", "5tt2gmwmi"):
            self.assertIn(f"command:{command}", specs)

    def test_le9_windows_bridge_list_has_5ttcheck(self):
        text = (REPO_ROOT / "scripts" / "install_deps_windows.ps1").read_text(encoding="utf-8")
        self.assertIn('"5ttcheck"', text)
        self.assertIn('"5tt2gmwmi"', text)


class Le10ValidationTest(unittest.TestCase):
    def _validate(self, root, targets):
        from shared import InputValidationConfig, validate_subject_inputs

        return validate_subject_inputs(InputValidationConfig(
            subject_id="100307", input_root=root, write_report=False, selected_targets=targets))

    def _subject(self, tmp, bvals):
        from test_input_validation import _make_subject_root

        root = _make_subject_root(Path(tmp))
        (root / "input" / "bvals").write_text(bvals + "\n")
        return root

    def test_le10_no_b0_is_an_error_when_microstructure_is_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._subject(tmp, "1000 1000 1000 2000")
            result = self._validate(root, ("microstructure",))
            issues = [i for i in result.messages if "No b0 volume" in i.message]
            self.assertEqual([i.level for i in issues], ["error"])
            self.assertIn("bvals", issues[0].message)
            self.assertIn("1000, 2000", issues[0].message)
            self.assertFalse(result.passed)

    def test_le10_no_b0_stays_a_warning_without_a_diffusion_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._subject(tmp, "1000 1000 1000 2000")
            result = self._validate(root, ("segmentation",))
            self.assertEqual([i.level for i in result.messages if "No b0 volume" in i.message], ["warning"])

    def test_le10_no_shell_is_an_error_for_tractography(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._subject(tmp, "0 0 0 0")
            result = self._validate(root, ("tractography",))
            issues = [i for i in result.messages if "No diffusion-weighted shell" in i.message]
            self.assertEqual([i.level for i in issues], ["error"])
            self.assertIn("found b-values: 0", issues[0].message)


class Le8LicenseFormatTest(unittest.TestCase):
    def test_le8_plausible_layouts_are_accepted(self):
        from shared.surface_license import license_problem, usable_license

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "license.txt"
            for text in ("user@example.edu\n12345\n*Cabc\n FSxyz\n", "\ufeffa.b@c.d\r\n1\r\n*C\r\n", "user@example.edu\n\n12345\n*Cabc"):
                path.write_text(text, encoding="utf-8")
                self.assertEqual(license_problem(path), "", text)
                self.assertTrue(usable_license(path))

    def test_le8_wrong_layouts_get_a_reason(self):
        from shared.surface_license import license_problem, usable_license

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "license.txt"
            for text in ("just one line of text", "no email here\n12345\n*Cabc\n", "user@example.edu\n12345\n"):
                path.write_text(text, encoding="utf-8")
                self.assertFalse(usable_license(path), text)
                self.assertIn("does not look like a FreeSurfer license", license_problem(path))
            self.assertIn("not found", license_problem(Path(tmp) / "missing.txt"))


class X7WindowsPathTest(unittest.TestCase):
    def test_x7_long_and_non_ansi_paths_warn(self):
        from shared import validation

        long_path = "C:/" + "a" * 250
        with mock.patch.object(validation, "_ansi_encodable", side_effect=lambda text: "\u4e2d" not in text):
            warnings = validation._windows_path_warnings([long_path, "C:/data/\u4e2d\u6587", "C:/data/ok", None])
        self.assertEqual(warnings, [
            f"Native Windows tools (FSL/MRtrix) may not open this path: {long_path}. Use a short ASCII-only folder.",
            "Native Windows tools (FSL/MRtrix) may not open this path: C:/data/\u4e2d\u6587. Use a short ASCII-only folder.",
        ])

    def test_x7_validation_warns_only_on_windows(self):
        from shared import InputValidationConfig, validate_subject_inputs, validation
        from test_input_validation import _make_subject_root

        with tempfile.TemporaryDirectory() as tmp:
            root = _make_subject_root(Path(tmp))
            config = InputValidationConfig(subject_id="100307", input_root=root, write_report=False)
            with mock.patch.object(validation, "_is_windows", return_value=False), \
                    mock.patch.object(validation, "_ansi_encodable", return_value=False):
                self.assertFalse([i for i in validate_subject_inputs(config).messages if i.category == "path"])
            with mock.patch.object(validation, "_is_windows", return_value=True), \
                    mock.patch.object(validation, "_ansi_encodable", return_value=False):
                issues = [i for i in validate_subject_inputs(config).messages if i.category == "path"]
            self.assertTrue(issues)
            self.assertEqual({i.level for i in issues}, {"warning"})


class X10DevPathTest(unittest.TestCase):
    def test_x10_dev_dirs_come_from_the_environment_only(self):
        from tractography import wma

        with mock.patch.dict(os.environ, {wma.DEV_XVFB_DIR_ENV: "/dev/xvfb"}):
            self.assertEqual(wma.dev_xvfb_tool_dirs(), (Path("/dev/xvfb/usr/bin"),))
            self.assertIn(Path("/dev/libpcre2-16-0/usr/lib/x86_64-linux-gnu"), wma.dev_xvfb_library_dirs())
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(wma.DEV_XVFB_DIR_ENV, None)
            self.assertEqual(wma.dev_xvfb_tool_dirs(), ())
            self.assertEqual(wma.dev_xvfb_library_dirs(), ())

    def test_x10_no_hard_coded_developer_path_remains(self):
        for relative in ("tractography/wma.py", "shared/environment.py"):
            self.assertNotIn("/data/chengjin", (REPO_ROOT / relative).read_text(encoding="utf-8"), relative)

    def test_x10_missing_xvfb_status_names_the_packages(self):
        from shared.environment import EnvironmentDoctorConfig, _install_hint, build_dependency_specs

        with mock.patch.dict(os.environ, {"DISPLAY": "", "SLICERDSPACE_DISPLAY": ""}), \
                mock.patch("shared.environment._effective_platform_name", return_value="Linux"):
            specs = {spec.key: spec for spec in build_dependency_specs(
                EnvironmentDoctorConfig(repo_root=REPO_ROOT, targets=("wma",)))}
        spec = specs["command:xvfb-run"]
        hint = _install_hint(spec, "Linux")  # what the panel shows for a missing command
        for package in ("xvfb", "libxcb-xinerama0", "libgl1", "libpcre2-16-0"):
            self.assertIn(package, hint)

    def test_x10_headless_hint_names_the_packages(self):
        from tractography.wma import headless_linux_hint

        hint = headless_linux_hint()
        for package in ("xvfb", "libxcb-xinerama0", "libgl1", "libpcre2-16-0"):
            self.assertIn(package, hint)


if __name__ == "__main__":
    unittest.main()
