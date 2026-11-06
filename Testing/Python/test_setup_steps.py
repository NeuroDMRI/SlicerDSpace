"""W-B7: setup_steps, licence acceptance in the user config, and scripts/setup_selected.py (all mocked)."""

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import setup_selected
from shared.selected_setup import build_setup_plan, python_package_commands, run_setup_steps, setup_steps
from shared.setup_components import (accept_licences, licence_record, pending_licences, plan_components,
                                     required_licences)
from shared.setup_resume import accepted_licences
from shared.user_config import empty_user_config, load_user_config, normalize_user_config

REPO = Path(__file__).resolve().parents[2]


def full_plan(folder):
    plan = build_setup_plan(SimpleNamespace(statuses=[]), system="Linux", machine="x86_64")
    plan.update(backend_profiles=["wmql"], backend_root=str(Path(folder) / "sources"),
                python_packages=["numpy>=1"], python_requests=[dict(requirement="numpy>=1", python_executable=None)],
                asset_profiles=["surface"], asset_root=str(Path(folder) / "models"), core_model_profile_bytes=1234,
                optional_asset_profiles=["tractcloud"], optional_asset_size_bytes=5678,
                managed_profiles=["fsl_cpu"], fsl_license_required=True, surface_prebuilt_required=True,
                surface_download_bytes=99)
    return plan


def accepted_for(plan):
    return {key: dict(id=key, version=record["version"], time="t")
            for key, record in required_licences(plan_components(plan)).items()}


class SetupStepsTests(unittest.TestCase):
    def test_steps_follow_the_b7_order_and_name_command_and_receipt(self):
        with tempfile.TemporaryDirectory() as folder, patch("scripts.setup_python_packages.dcm2niix_pending", return_value=False):
            plan = full_plan(folder)
            steps = setup_steps(plan, accepted_for(plan), folder=folder, config={"model_assets_dir": folder})
            self.assertEqual([step["id"] for step in steps],
                             ["backend_sources", "python_packages", "model_assets", "optional_resources",
                              "managed_runtime", "surface_runtime", "self_check"])
            for step in steps[:-1]:
                self.assertTrue(step["commands"], step["id"])
            for step in steps:
                if step["id"] not in {"model_assets", "self_check"}:
                    self.assertTrue(step["receipt"], step["id"])
            managed = steps[4]["commands"][0][1]
            self.assertIn("--accept-fsl-license", managed)
            self.assertIn("setup_managed_runtime.py", " ".join(managed))
            self.assertTrue(steps[-1]["script"].endswith("check_setup_functionality.py"))

    def requirements_by_program(self, step):
        found = {}
        for program, args, _env in step["commands"]:
            request = json.loads(Path(args[args.index("--request") + 1]).read_text())
            found.setdefault(program, []).extend(request["requirements"])
        return found

    def test_s_b3_dcm2niix_is_pending_alone_and_goes_into_slicers_python(self):
        pin = "dcm2niix==1.0.20261029"
        with tempfile.TemporaryDirectory() as folder:
            # Everything else installed, the wheel missing: the plan lists exactly the pin as pending.
            plan = build_setup_plan(SimpleNamespace(statuses=[]), system="Linux", machine="x86_64", dcm2niix_missing=True)
            self.assertEqual(plan["python_packages"], [pin])
            self.assertEqual(build_setup_plan(SimpleNamespace(statuses=[]), system="Linux", machine="x86_64")["python_packages"], [])
            # An external interpreter is selected for TractCloud: python_command returns it for every request.
            plan["python_packages"].append("tractcloud-dep>=1")
            plan["python_requests"].append(dict(requirement="tractcloud-dep>=1", python_executable="/ext/python"))
            steps = setup_steps(plan, accepted_for(plan), folder=folder,
                                launcher=("/slicer/PythonSlicer", ["--launch", "PythonSlicer"]),
                                python_command=lambda override: (override or "/ext/python", []))
            found = self.requirements_by_program(next(step for step in steps if step["id"] == "python_packages"))
            self.assertEqual(found["/slicer/PythonSlicer"], [pin])  # fails before: the pin went to the selected external Python
            self.assertNotIn(pin, found["/ext/python"])
            # Without a launcher the pin goes to the interpreter that runs the setup (the CLI under PythonSlicer).
            steps = setup_steps(plan, accepted_for(plan), folder=folder, python_command=lambda override: (override or "/ext/python", []))
            import sys
            self.assertEqual(self.requirements_by_program(steps[0])[sys.executable], [pin])

    def test_missing_licence_stops_setup_steps(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = full_plan(folder)
            accepted = accepted_for(plan)
            accepted.pop("fsl")
            with self.assertRaisesRegex(PermissionError, "fsl"):
                setup_steps(plan, accepted, folder=folder)

    def test_windows_ps1_step_replaces_the_managed_runtime_step(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = full_plan(folder)
            plan["windows_runtime"] = dict(distribution="Ubuntu", installs_freesurfer=False,
                                           command=["powershell", "-NoProfile", "-File", "install_deps_windows.ps1"])
            steps = setup_steps(plan, accepted_for(plan), folder=folder)
            ids = [step["id"] for step in steps]
            self.assertIn("windows_runtime", ids)
            self.assertNotIn("managed_runtime", ids)
            ps1 = steps[ids.index("windows_runtime")]
            self.assertEqual((ps1["commands"][0][0], ps1["distribution"]), ("powershell", "Ubuntu"))

    def test_wmql_requirement_uses_the_checkout_the_backend_step_creates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "sources" / "wmql-5c69c594d489"
            url = "git+https://github.com/demianw/tract_querier.git"
            commands, _receipts = python_package_commands(
                (url,), requests=[dict(requirement=url, python_executable=None)], python_command=lambda _x: ("py", []),
                folder=folder, tract_querier_root=root, root_expected=True)
            request = json.loads(Path(commands[0][1][commands[0][1].index("--request") + 1]).read_text())
            self.assertEqual(request["requirements"], [str(root)])


class LicenceAcceptanceTests(unittest.TestCase):
    def test_accepted_licences_are_in_the_user_config_and_normalized(self):
        self.assertEqual(empty_user_config()["accepted_licences"], {})
        kept = normalize_user_config({"accepted_licences": {"fsl": {"version": "v", "time": "t"}, "bad": {"time": "t"}, "x": 3}})
        self.assertEqual(kept["accepted_licences"], {"fsl": {"id": "fsl", "version": "v", "time": "t"}})

    def test_version_is_the_sha256_of_the_local_licence_file_and_acceptance_is_remembered(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "user_config.json"
            with patch.dict("os.environ", {"SLICERDSPACE_USER_CONFIG": str(config)}):
                record = licence_record("freesurfer")
                self.assertEqual(record["version"], hashlib.sha256((REPO / "licenses/FreeSurfer.txt").read_bytes()).hexdigest())
                self.assertEqual(record["file"], str(REPO / "licenses/FreeSurfer.txt"))
                required = {"freesurfer": record}
                self.assertEqual(set(pending_licences(required, accepted_licences())), {"freesurfer"})
                accept_licences(required)
                stored = load_user_config()["accepted_licences"]["freesurfer"]
                self.assertEqual((stored["id"], stored["version"]), ("freesurfer", record["version"]))
                self.assertTrue(stored["time"].endswith("Z"))
                self.assertEqual(pending_licences(required, accepted_licences()), {})  # asks nothing again
                changed = {"freesurfer": dict(record, version="new-version")}
                self.assertEqual(set(pending_licences(changed, accepted_licences())), {"freesurfer"})


class SetupSelectedCliTests(unittest.TestCase):
    def run_cli(self, argv, plan, folder, *, runner=None):
        closure = SimpleNamespace(targets=("validation",), tractography_algorithms=("csd_act",))
        out = io.StringIO()
        with patch.dict("os.environ", {"SLICERDSPACE_USER_CONFIG": str(Path(folder) / "user_config.json")}), \
                patch.object(setup_selected, "build_cli_plan", return_value=(closure, SimpleNamespace(), plan)), \
                patch.object(setup_selected, "self_check_commands", return_value=([], [])), \
                patch("scripts.setup_python_packages.dcm2niix_pending", return_value=False), \
                redirect_stdout(out):
            code = setup_selected.main(argv, runner=runner, log=lambda text: out.write(str(text) + "\n"))
        return code, out.getvalue()

    def small_plan(self, folder):
        plan = build_setup_plan(SimpleNamespace(statuses=[]), system="Linux", machine="x86_64")
        (Path(folder) / "models").mkdir()
        plan.update(asset_profiles=["surface"], asset_root=str(Path(folder) / "models"), core_model_profile_bytes=2048,
                    managed_profiles=["fsl_cpu"], fsl_license_required=True)
        return plan

    def test_plan_lists_size_licence_name_link_and_local_file_without_installing(self):
        with tempfile.TemporaryDirectory() as folder:
            calls = []
            code, text = self.run_cli(["--targets", "validation"], self.small_plan(folder), folder,
                                      runner=lambda *a, **k: calls.append(a))
            self.assertEqual((code, calls), (0, []))
            self.assertIn("Model assets: surface (2.0 KiB)" if "KiB" in text else "Model assets: surface", text)
            self.assertIn("licence: FSL licence", text)
            self.assertIn("link: https://fsl.fmrib.ox.ac.uk/fsl/docs/license.html", text)
            self.assertIn("local file: none", text)

    def test_execute_needs_accept_licences_then_runs_every_step_and_stores_the_acceptance(self):
        with tempfile.TemporaryDirectory() as folder:
            calls = []

            def runner(command, env=None, check=False):
                calls.append(command)
                if "--result" in command and "setup_managed_runtime.py" in " ".join(command):  # the script writes its receipt
                    Path(command[command.index("--result") + 1]).write_text(json.dumps(
                        {"kind": "slicerdsurfer_runtime_setup", "status": "completed",
                         "external_dependencies": {"mrtrix_bin_dir": folder}}))
                return SimpleNamespace(returncode=0)

            plan = self.small_plan(folder)
            code, text = self.run_cli(["--targets", "validation", "--execute"], plan, folder, runner=runner)
            self.assertEqual((code, calls), (2, []))
            self.assertIn("--accept-licences", text)
            code, text = self.run_cli(["--targets", "validation", "--execute", "--accept-licences"], plan, folder, runner=runner)
            self.assertEqual(code, 0, text)
            joined = [" ".join(command) for command in calls]
            self.assertTrue(any("download_model_assets.py" in line for line in joined))
            self.assertTrue(any("setup_managed_runtime.py" in line and "--accept-fsl-license" in line for line in joined))
            with patch.dict("os.environ", {"SLICERDSPACE_USER_CONFIG": str(Path(folder) / "user_config.json")}):
                self.assertIn("fsl", load_user_config()["accepted_licences"])

    def test_a_failed_step_stops_the_run(self):
        steps = [dict(id="model_assets", title="one", commands=[("p", ["a"], {})], receipt=None, asset_root="/nonexistent"),
                 dict(id="self_check", title="two", commands=[], receipt=None)]
        records = run_setup_steps(steps, runner=lambda *a, **k: SimpleNamespace(returncode=1), log=lambda _t: None)
        self.assertEqual([(r["id"], r["status"]) for r in records], [("model_assets", "failed")])


class SetupSelectedPlanInputsTests(unittest.TestCase):
    """Fix round 1: how scripts/setup_selected.py builds the doctor config from the saved user config."""

    def plan_for(self, targets, defaults=None, deps=None):
        seen = []

        def doctor(config):
            seen.append(config)
            return SimpleNamespace(statuses=[], targets=config.targets, blocker_count=0,
                                   python_executable=str(config.python_executable) if config.python_executable else None)

        config = empty_user_config()
        config["run_defaults"].update(defaults or {})
        config["external_dependencies"].update(deps or {})
        with patch("scripts.setup_python_packages.dcm2niix_pending", return_value=False):
            closure, report, plan = setup_selected.build_cli_plan(targets, config, doctor=doctor, system="Linux", machine="x86_64")
        return seen[0], closure, report, plan

    def test_s_b4_a_saved_interpreter_is_kept_and_the_current_one_is_only_the_default(self):
        import sys
        config, *_ = self.plan_for(["validation"], deps={"python_executable": "/ext/py/bin/python"})
        self.assertEqual(config.python_executable, Path("/ext/py/bin/python"))  # fails before: replaced by sys.executable
        config, *_ = self.plan_for(["validation"], deps={"python_environment": "tractcloud-env"})
        self.assertIsNone(config.python_executable)
        self.assertEqual(config.python_environment, "tractcloud-env")
        config, *_ = self.plan_for(["validation"])
        self.assertEqual(config.python_executable, Path(sys.executable))

    def test_s_b5_saved_multi_backend_selections_are_parsed_before_the_closure(self):
        _config, closure, *_ = self.plan_for(["bundles"], defaults={"bundle_backend": "wmql,wma", "tractography_backend": "csd_act,ukf"})
        self.assertIn("wma", closure.targets)  # fails before: "wmql,wma" was passed as one unknown backend
        self.assertIn("ukf", closure.tractography_algorithms)
        self.assertIn("csd_act", closure.tractography_algorithms)
        _config, closure, *_ = self.plan_for(["bundles"], defaults={"bundle_backend": "all", "tractography_backend": "all"})
        self.assertTrue({"wma", "tractcloud"} <= set(closure.targets))
        self.assertEqual(set(closure.tractography_algorithms), {"csd_act", "ukf", "dti_det", "saf"})

    def test_s_b6_the_doctor_gets_the_closed_algorithm_set(self):
        config, closure, *_ = self.plan_for(["wma"])  # defaults: csd_act; the WMA closure adds UKF
        self.assertIn("ukf", closure.tractography_algorithms)
        self.assertIn("ukf", config.tractography_backend.split(","))  # fails before: the saved csd_act only

    def test_s_b7_the_cli_self_check_cleans_the_environment_for_an_external_interpreter(self):
        request = {"python": "/ext/py/bin/python"}
        with tempfile.TemporaryDirectory() as folder, \
                patch("shared.setup_selftest.requests_for_report", return_value=[request]), \
                patch.dict("os.environ", {"PYTHONHOME": "/slicer/lib/python"}):
            doctor = lambda config: SimpleNamespace(statuses=[], targets=config.targets, blocker_count=0, python_executable=None)
            commands, _receipts = setup_selected.self_check_commands(
                folder, empty_user_config(), ["validation"], {"script": "check_setup_functionality.py"}, doctor=doctor)
        program, _args, environment = commands[0]
        self.assertEqual(program, "/ext/py/bin/python")
        self.assertIsNone(environment.get("PYTHONHOME", "absent"))  # fails before: only MPLBACKEND was set
        self.assertEqual(environment["MPLBACKEND"], "Agg")

    def test_s_b7_the_cli_python_step_uses_the_configured_external_interpreter_with_a_clean_environment(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = build_setup_plan(SimpleNamespace(statuses=[]), system="Linux", machine="x86_64")
            plan.update(python_packages=["numpy>=1"], python_requests=[dict(requirement="numpy>=1", python_executable=None)])
            closure = SimpleNamespace(targets=("validation",), tractography_algorithms=("csd_act",))
            report = SimpleNamespace(statuses=[], python_executable="/ext/py/bin/python")
            calls, out = [], io.StringIO()

            def runner(command, env=None, check=False):
                calls.append((command, env))
                return SimpleNamespace(returncode=1)  # stop after the first command

            with patch.dict("os.environ", {"SLICERDSPACE_USER_CONFIG": str(Path(folder) / "user_config.json"),
                                           "PYTHONHOME": "/slicer/lib/python"}), \
                    patch.object(setup_selected, "build_cli_plan", return_value=(closure, report, plan)), \
                    patch("scripts.setup_python_packages.dcm2niix_pending", return_value=False), redirect_stdout(out):
                setup_selected.main(["--targets", "validation", "--execute", "--accept-licences"], runner=runner,
                                    log=lambda text: out.write(str(text) + "\n"))
            command, env = calls[0]
            self.assertEqual(command[0], "/ext/py/bin/python")  # fails before: sys.executable
            self.assertNotIn("PYTHONHOME", env)


if __name__ == "__main__":
    unittest.main()
