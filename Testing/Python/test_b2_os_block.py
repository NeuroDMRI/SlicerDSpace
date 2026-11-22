"""Batch 2, lane os: one structured state for an OS-blocked surface runtime, the dialogs it drives, OS floors.

User report (macOS 13.4, Apple Silicon): the bundled surface runtime needs macOS 15.0, but the on-demand dialog offered
"Download and Install", the environment row repeated the reason as the "Suggested fix", and its "Setup" button only
showed the same text again. Each test names the request item (A, B, C, D) or the lane decision (1-8) it covers.
The Mac is faked (``platform`` and the host architecture), so the tests run on every OS.
"""

from __future__ import annotations

import contextlib
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

from Testing.Python.test_preproc_fsl_gate import make_surface_bundle  # noqa: E402

CATALOG = REPO_ROOT / "models" / "runtime_bundles.json"


def _catalog_record(system="Darwin", machine="arm64"):
    payload = json.loads(CATALOG.read_text(encoding="utf-8"))
    return next(r for r in payload["surface"] if r["system"] == system and r["machine"] == machine)


class MacCase(unittest.TestCase):
    """A temp bundled-runtimes root with a surface runtime whose binaries need macOS 15.0.

    ``setUp`` also installs a FIXTURE catalog whose darwin-arm64 record needs macOS 15.0, so the scenarios below do not
    depend on the ``min_macos`` of the shipped ``models/runtime_bundles.json``. A test that needs another floor calls
    ``fake_catalog`` again (the later patch wins).
    """

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.bundled = self.root / "bundled-runtimes"
        self.bundled.mkdir()
        (self.root / "runtimes").mkdir()
        (self.root / "assets").mkdir()
        env = {"SLICERDSPACE_BUNDLED_RUNTIMES_ROOT": str(self.bundled),
               "SLICERDSPACE_RUNTIME_ROOT": str(self.root / "runtimes"),
               "SLICERDSPACE_MODEL_ASSETS_DIR": str(self.root / "assets")}
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in ("FS_LICENSE", "FREESURFER_LICENSE", "SLICERDSPACE_SURFACE_RUNTIME", "FREESURFER_HOME"):
            os.environ.pop(name, None)
        self.bundle = make_surface_bundle(self.bundled, minos=(15, 0))
        self.prefix = self.root / "user-cache" / "surface-runtime"  # the default prefix; not installed yet
        self._mac_stack = None
        self.fake_catalog("15.0")  # the catalog scenario is "the runtime needs macOS 15.0", whatever the shipped value

    def mac(self, version="13.4"):
        if self._mac_stack is not None:
            self._mac_stack.close()
        stack = contextlib.ExitStack()
        for patcher in (
            mock.patch("platform.system", return_value="Darwin"),
            mock.patch("platform.machine", return_value="arm64"),
            mock.patch("platform.mac_ver", return_value=(version, ("", "", ""), "")),
            mock.patch("shared.platform_info.host_machine", return_value="arm64"),
            mock.patch("shared.environment.resolve_freesurfer_home", return_value=None),
            mock.patch("shared.surface_runtime.default_surface_runtime", return_value=self.prefix),
        ):
            stack.enter_context(patcher)
        self._mac_stack = stack
        self.addCleanup(stack.close)

    def fake_catalog(self, min_macos="13.0", published=True):
        """A catalog whose darwin-arm64 record needs ``min_macos`` (the runtime lane will publish a 13.0 build)."""
        payload = json.loads(CATALOG.read_text(encoding="utf-8"))
        for record in payload["surface"]:
            if record["system"] == "Darwin" and record["machine"] == "arm64":
                record["min_macos"] = min_macos
                record["published"] = published
        path = self.root / "runtime_bundles.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        patcher = mock.patch("shared.prebuilt_surface.CATALOG", path)
        patcher.start()
        self.addCleanup(patcher.stop)

    def atlas_present(self, present=True):
        patcher = mock.patch("shared.ui_on_demand_resources._all_files_nonempty", return_value=present)
        patcher.start()
        self.addCleanup(patcher.stop)

    def doctor(self):
        from shared.environment import EnvironmentDoctorConfig, run_environment_doctor

        licence = self.root / "license.txt"
        licence.write_text("name\nkey\n")
        with mock.patch("shared.environment.build_dependency_specs", return_value=()), \
                mock.patch("shared.surface_license.resolve_surface_license", return_value=licence), \
                mock.patch("subprocess.run", side_effect=AssertionError("probe must not start")):
            result = run_environment_doctor(EnvironmentDoctorConfig(targets=("surface_postprocess",),
                                                                    repo_root=REPO_ROOT))
        return result, {item.key: item for item in result.statuses}["surface_postprocess_runtime"]

    def surface_items(self, backend_paths=None):
        from shared import ui_on_demand_resources as od

        items = od.detect_missing_task_resources(targets=("surface_postprocess",), bundle_backends=("wmql",),
                                                 backend_paths=backend_paths or {})
        return [item for item in items if item["id"] == "surface_postprocess"]


# --------------------------------------------------------------------------- A: on-demand and the runtime state


class OnDemandStateTest(MacCase):
    def test_b2_os_a_on_demand_offers_no_download_for_an_os_blocked_surface_runtime(self):
        """A, decision 2: catalog min_macos 15.0 on 13.4 is remedy none, so no item and no size (was 40,000,000)."""
        self.mac("13.4")
        self.atlas_present(True)
        self.assertEqual(self.surface_items(), [])
        self.atlas_present(False)  # the atlas alone cannot make the step run
        self.assertEqual(self.surface_items(), [])

    def test_b2_os_1_state_is_ready_missing_or_os_blocked(self):
        """Decision 1: three states, and the versions of an OS block (a bool could not tell them apart)."""
        from shared.surface_runtime import surface_runtime_state

        self.mac("13.4")
        state = surface_runtime_state()
        self.assertEqual(state["state"], "os_blocked")
        self.assertEqual((state["macos_required"], state["macos_current"]), ("15.0", "13.4"))
        self.assertEqual(state["remedy"], "none")
        self.assertEqual(state["download_bytes"], 0)
        self.mac("15.1")
        self.assertEqual(surface_runtime_state()["state"], "ready")
        import shutil

        shutil.rmtree(self.bundle)
        self.assertEqual(surface_runtime_state()["state"], "missing")
        self.assertEqual(surface_runtime_state()["remedy"], "download")  # 15.1 meets the catalog's 15.0
        self.mac("13.4")
        missing = surface_runtime_state()
        # No runtime here, and the fixture catalog's needs 15.0: the true cause is the OS, not "missing" (audit cell A-3).
        self.assertEqual((missing["state"], missing["remedy"]), ("os_blocked", "none"))
        self.assertEqual((missing["macos_required"], missing["macos_current"]), ("15.0", "13.4"))

    def test_b2_os_e_no_runtime_and_a_catalog_above_this_mac_is_an_os_block_everywhere(self):
        """Audit cell 'missing + not downloadable' (macOS below the catalog min_macos, no runtime installed): the
        inspector says os_blocked without a probe (the probe said build_required), the hint names real actions,
        Prepare gets no manual item, and the Run skips the step. Before the fix the state was 'missing'."""
        import shutil
        from shared.selected_setup import build_setup_plan
        from shared.surface_runtime import inspect_surface_runtime, surface_postprocess_os_skip

        shutil.rmtree(self.bundle)
        self.mac("13.4")
        self.atlas_present(True)
        with mock.patch("subprocess.run", side_effect=AssertionError("probe must not start")):
            record = inspect_surface_runtime()
        self.assertEqual((record["state"], record["remedy"]), ("os_blocked", "none"))
        self.assertFalse(record["build_required"])
        self.assertIn("15.0", record["notes"])
        self.assertIn("13.4", record["notes"])
        self.assertIn("upgrade macOS to 15.0", record["install_hint"])
        self.assertIsNotNone(surface_postprocess_os_skip({}))
        self.assertEqual(self.surface_items(), [])
        result, row = self.doctor()
        self.assertNotEqual(row.install_hint, row.details["notes"])
        plan = build_setup_plan(result, system="Darwin", machine="arm64",
                                backend_paths={"model_assets_dir": str(self.root / "assets")})
        self.assertNotIn("surface_postprocess_runtime", [item["key"] for item in plan["manual_items"]])
        self.assertEqual([item["key"] for item in plan["os_blocked_items"]][:1], ["surface_postprocess_runtime"])
        self.assertFalse(plan["surface_prebuilt_required"])

    def test_b2_os_11_on_demand_runtime_size_uses_the_catalog_bytes(self):
        """C: a really missing runtime on a supported Mac is offered with the catalog size, not 40,000,000."""
        import shutil

        shutil.rmtree(self.bundle)
        self.mac("15.1")
        self.atlas_present(True)
        items = self.surface_items()
        self.assertEqual(len(items), 1, items)
        self.assertEqual(items[0]["size_bytes"], _catalog_record()["bytes"])

    def test_b2_os_7_explicit_empty_folder_is_not_a_ready_surface_runtime(self):
        """Decision 7: an empty or unrelated explicit folder is not ready (was True for any existing folder)."""
        from shared import ui_on_demand_resources as od

        empty = self.root / "empty"
        empty.mkdir()
        self.mac("15.1")
        self.assertFalse(od._surface_runtime_ready(str(empty)))
        stamp = self.root / "fs"
        stamp.mkdir()
        (stamp / "build-stamp.txt").write_text("freesurfer\n")
        self.assertTrue(od._surface_runtime_ready(str(stamp)))

    def test_b2_os_c_ticking_surface_postprocess_on_a_blocked_mac_raises_no_prompt(self):
        """Test (c), decision 6: the click handler calls maybe_prompt_on_demand_install; it builds no dialog.

        b2fix F6: this test used a closure without ``surface`` and marked every resource present, which hid the
        223 MB DDSurfer weights item. It now uses the real widget closure and no resource is present.
        """
        from shared import ui_on_demand_resources as od
        from Testing.Python import test_ui_algorithm_availability as taa

        self.mac("13.4")
        self.atlas_present(False)
        subject = taa.write_subject(Path(tempfile.mkdtemp(dir=self.root)) / "s1", taa.THREE_GROUPS)
        owner = taa.RunRequestTest.build_widget(self, subject, "dti")
        owner._parameterNode.targetSurfacePostprocess = True
        owner._parameterNode.targetMicrostructure = False
        owner._parameterNode.wmaAtlas = "org"
        owner._declinedOnDemandItems = set()
        self.assertIn("surface", owner._selected_closure().targets)  # the stored closure still holds DDSurfer
        qt_mock, slicer_mock = mock.MagicMock(), mock.MagicMock()
        with mock.patch.dict(sys.modules, {"qt": qt_mock, "slicer": slicer_mock}):
            self.assertFalse(od.maybe_prompt_on_demand_install(owner, force=True))
        qt_mock.QDialog.assert_not_called()

    def test_b2_os_resolver_a_freesurfer_folder_makes_a_blocked_mac_ready(self):
        """Decision 3: the hint's 'set a FreeSurfer folder' works: ready everywhere, no on-demand item, no Run skip."""
        from shared.surface_runtime import surface_postprocess_os_skip, surface_runtime_state

        home = self.root / "freesurfer"
        home.mkdir()
        (home / "build-stamp.txt").write_text("freesurfer\n")
        self.mac("13.4")
        self.atlas_present(True)
        self.assertEqual(surface_runtime_state(freesurfer_home=home)["state"], "ready")
        self.assertEqual(self.surface_items({"postprocess_freesurfer_home": str(home)}), [])
        self.assertIsNone(surface_postprocess_os_skip({"postprocess_freesurfer_home": str(home)}))
        self.assertIsNotNone(surface_postprocess_os_skip({}))


class CatalogDownloadTest(MacCase):
    def test_b2_os_2a_catalog_for_13_0_gives_download_and_the_installed_copy_is_used(self):
        """Test (a), decision 2: a fake catalog with min_macos 13.0 on a 13.4 Mac switches to remedy download."""
        from shared.selected_setup import build_setup_plan
        from shared.environment import EnvironmentDoctorConfig, run_environment_doctor
        from shared.surface_runtime import inspect_surface_runtime, runtime_dir_macos_block, surface_runtime_state

        self.fake_catalog("13.0")
        self.mac("13.4")
        self.atlas_present(True)
        bytes_ = _catalog_record()["bytes"]
        state = surface_runtime_state()
        self.assertEqual((state["state"], state["remedy"]), ("os_blocked", "download"))
        self.assertEqual(state["download_bytes"], bytes_)
        items = self.surface_items()
        self.assertEqual([item["size_bytes"] for item in items], [bytes_])
        # The doctor row asks for the prebuilt install, so the Prepare plan schedules prepare().
        with mock.patch("subprocess.run", side_effect=AssertionError("no probe for a runtime that is not there")):
            record = inspect_surface_runtime()
        self.assertTrue(record["build_required"])
        with mock.patch("shared.environment.build_dependency_specs", return_value=()), \
                mock.patch("shared.surface_runtime.inspect_surface_runtime", return_value=record):
            result = run_environment_doctor(EnvironmentDoctorConfig(targets=("surface_postprocess",),
                                                                    repo_root=REPO_ROOT))
        plan = build_setup_plan(result, system="Darwin", machine="arm64",
                                backend_paths={"model_assets_dir": str(self.root / "assets")})
        self.assertTrue(plan["surface_prebuilt_required"])
        self.assertEqual(plan["surface_download_bytes"], bytes_)
        self.assertNotIn("surface_postprocess_runtime", [item["key"] for item in plan["manual_items"]])
        self.assertEqual(plan.get("os_blocked_items", []), [])
        # Simulated install: prepare() writes the runtime to the default prefix.
        installed = make_surface_bundle(self.root / "user-cache", minos=(13, 0))
        installed.rename(self.prefix)
        after = surface_runtime_state()
        self.assertEqual(after["state"], "ready")
        self.assertEqual(Path(after["home"]).resolve(), self.prefix.resolve())
        self.assertTrue(runtime_dir_macos_block(self.bundle))  # the bundled copy stays blocked
        probe = SimpleNamespace(returncode=0, stdout=json.dumps({"components_available": True}) + "\n", stderr="")
        with mock.patch("subprocess.run", return_value=probe) as run:
            checked = inspect_surface_runtime()
        self.assertTrue(checked["components_available"])
        self.assertEqual(checked["state"], "ready")  # decision 1: a probe record names its state too
        command = run.call_args[0][0]
        self.assertEqual(Path(command[command.index("--home") + 1]).resolve(), self.prefix.resolve())

    def test_b2_os_2_unpublished_record_is_not_a_download(self):
        """Decision 2 (prepare() refuses an unpublished record): such a record is remedy none."""
        from shared.surface_runtime import surface_runtime_state

        self.fake_catalog("13.0", published=False)
        self.mac("13.4")
        self.assertEqual(surface_runtime_state()["remedy"], "none")


# --------------------------------------------------------------------------- B: doctor row, item text, button


class DoctorRowTest(MacCase):
    def test_b2_os_3_inspect_record_carries_the_versions_and_a_hint_that_differs(self):
        """Decision 1, B: the record has the structured fields, and its hint is not the reason."""
        from shared.surface_runtime import inspect_surface_runtime

        self.mac("13.4")
        with mock.patch("subprocess.run", side_effect=AssertionError("probe must not start")):
            record = inspect_surface_runtime()
        self.assertEqual(record["state"], "os_blocked")
        self.assertEqual(record["remedy"], "none")
        self.assertEqual((record["macos_required"], record["macos_current"]), ("15.0", "13.4"))
        self.assertTrue(record["macos_blocked"])
        self.assertFalse(record["build_required"])
        self.assertTrue(record["install_hint"])
        self.assertNotEqual(record["install_hint"], record["notes"])

    def test_b2_os_4_doctor_hint_names_real_actions_in_english_and_chinese(self):
        """B, decision 3: reason names both versions; hint: upgrade, FreeSurfer home setting, untick; zh too."""
        import re

        self.mac("13.4")
        _, status = self.doctor()
        reason, hint = status.details["notes"], status.install_hint
        self.assertNotEqual(reason, hint)
        for text in ("15.0", "13.4"):
            self.assertIn(text, reason)
        self.assertIn("upgrade macOS to 15.0", hint)
        self.assertIn("FreeSurfer home (optional)", hint)  # the setting as the UI names it
        self.assertIn("untick Surface postprocess", hint)
        self.assertIsNone(re.search(r"FreeSurfer \d", hint))  # no FreeSurfer version is named
        reason_zh, hint_zh = status.details["notes_zh"], status.details["install_hint_zh"]
        self.assertNotEqual(reason_zh, hint_zh)
        self.assertIn("15.0", reason_zh)
        self.assertIn("13.4", reason_zh)
        self.assertIn("FreeSurfer 主目录（可选）", hint_zh)
        self.assertIn("取消勾选", hint_zh)
        self.assertNotEqual(status.status, "available")
        # The setting path in both hints is the exact text of the settings UI (ui_onboarding.py).
        ui_source = (REPO_ROOT / "shared" / "ui_onboarding.py").read_text(encoding="utf-8")
        for name in ("More options (offline, repair, paths)", "External Runtime Paths", "FreeSurfer home (optional)"):
            self.assertIn(name, hint)
            self.assertIn(name, ui_source)
        for name in ("更多选项（离线、修复、路径）", "外部运行环境路径", "FreeSurfer 主目录（可选）", "表面后处理"):
            self.assertIn(name, hint_zh)
            self.assertIn(name, ui_source)

    def test_b2_os_5_item_text_omits_a_suggested_fix_equal_to_the_reason(self):
        """Decision 4: 'Suggested fix' only when the hint is not empty and differs from the reason."""
        from shared.ui_environment_panel import environment_item_text

        same = SimpleNamespace(key="k", kind="setting", label="Row", required=True, targets=("t",),
                               details={"notes": "The reason."}, install_hint="The reason.")
        self.assertNotIn("Suggested fix", environment_item_text(same))
        empty = SimpleNamespace(**dict(vars(same), install_hint=""))
        self.assertNotIn("Suggested fix", environment_item_text(empty))
        other = SimpleNamespace(**dict(vars(same), install_hint="Do this."))
        self.assertIn("Suggested fix: Do this.", environment_item_text(other))
        self.mac("13.4")
        _, status = self.doctor()
        text = environment_item_text(status)
        self.assertEqual(text.count("15.0 or later"), 2, text)  # once in the reason, once in the upgrade action
        self.assertIn("Suggested fix", text)
        zh = environment_item_text(status, zh=True)
        self.assertIn(status.details["install_hint_zh"], zh)
        self.assertIn(status.details["notes_zh"], zh)

    def test_b2_os_4b_four_more_rows_get_a_real_hint_or_none(self):
        """Decision 4: bundled_python_activation, torch_numpy_abi, slicer_numpy_floor, linux_glibc_floor."""
        from shared import environment as env
        from shared.ui_environment_panel import environment_item_text

        spec = env.DependencySpec(key="python:torch", kind="python", label="torch", required=True,
                                  targets=("segmentation",), python_import="torch", python_package="torch>=2.0.0")
        status = env.DependencyStatus(key="python:torch", kind="python", label="torch", status="available",
                                      required=True, targets=("segmentation",))
        problem = ("bundled Python stack is for Python 3.99 x86_64; this Slicer runs Python 3.12 on arm64; "
                   "install the SlicerDSpace package built for this Slicer")
        probe = {"numpy_version": "1.26.4", "torch_version": "2.2.2", "abi_mismatch": True, "error": "x"}
        with mock.patch.object(env, "torch_numpy_abi_probe", return_value=probe), \
                mock.patch("shared.bundled_python.activation_problem", return_value=problem), \
                mock.patch.object(env, "_bundle_runtime_numpy_floor", return_value="2.0.2"):
            rows = list(env._runtime_compatibility_statuses((spec,), (status,)))
        glibc_spec = env.DependencySpec(key="command:mrconvert", kind="command", label="mrconvert", required=True,
                                        targets=("preprocessing",), install_group="mrtrix")
        rows += list(env._linux_glibc_floor_statuses((glibc_spec,), (), glibc=(2, 28), host_platform="Linux"))
        keys = {row.key: row for row in rows}
        self.assertEqual(set(keys), {"bundled_python_activation", "torch_numpy_abi", "slicer_numpy_floor",
                                     "linux_glibc_floor"})
        for key, row in keys.items():
            self.assertNotEqual(row.install_hint, row.details["notes"], key)
            self.assertNotIn(row.install_hint or "\0", row.details["notes"], key)
            text = environment_item_text(row)
            self.assertEqual(text.count(row.details["notes"]), 1, key)
        self.assertIn("1.26.4", keys["slicer_numpy_floor"].details["notes"])
        self.assertIn("2.0.2", keys["slicer_numpy_floor"].install_hint)
        self.assertIn("Reinstall the SlicerDSpace package", keys["torch_numpy_abi"].install_hint)
        self.assertIn("Install the SlicerDSpace package built for this Slicer",
                      keys["bundled_python_activation"].install_hint)
        self.assertIn("Ubuntu 22.04+", keys["linux_glibc_floor"].install_hint)

    def test_b2_os_6_row_button_opens_the_freesurfer_home_chooser(self):
        """Decision 5: not a plain 'Setup' that repeats the text; English and Chinese; mapped to the setting."""
        from shared.ui_environment_panel import environment_item_button_text, environment_item_button_tooltip

        self.mac("13.4")
        _, status = self.doctor()
        self.assertEqual(environment_item_button_text(status), "Choose FreeSurfer folder")
        self.assertEqual(environment_item_button_text(status, zh=True), "选择 FreeSurfer 文件夹")
        self.assertNotEqual(environment_item_button_tooltip(status), status.details["notes"])
        mw = _import_module_widget(self)
        widget = mw.SharedWidgetMixin()
        widget._is_model_resource_item = lambda item: False
        self.assertEqual(mw.SharedWidgetMixin._environment_setting_for_item(widget, status),
                         ("postprocess_freesurfer_home", "dir"))


def _import_module_widget(case):
    mods = {name: mock.MagicMock() for name in ("qt", "slicer", "vtk", "slicer.ScriptedLoadableModule", "ctk")}
    patcher = mock.patch.dict(sys.modules, mods)
    patcher.start()
    case.addCleanup(patcher.stop)
    sys.modules.pop("shared.module_widget", None)
    case.addCleanup(sys.modules.pop, "shared.module_widget", None)
    import shared.module_widget as mw

    return mw


# --------------------------------------------------------------------------- C: Prepare plan, licence, readiness


class PrepareTest(MacCase):
    def test_b2_os_7_setup_plan_keeps_os_blocked_items_out_of_manual_items(self):
        """C: no manual item, so no forced confirm dialog and no pending.json ('Continue preparing')."""
        from shared.selected_setup import build_setup_plan
        from shared.ui_selected_setup import setup_plan_pending_actions

        self.mac("13.4")
        result, _ = self.doctor()
        plan = build_setup_plan(result, system="Darwin", machine="arm64",
                                backend_paths={"model_assets_dir": str(self.root / "assets")})
        self.assertNotIn("surface_postprocess_runtime", [item["key"] for item in plan["manual_items"]])
        self.assertEqual([item["key"] for item in plan["os_blocked_items"]], ["surface_postprocess_runtime"])
        self.assertFalse(plan["surface_prebuilt_required"])
        self.assertFalse(setup_plan_pending_actions(plan))
        self.assertTrue(setup_plan_pending_actions(dict(plan, manual_items=[{"key": "x"}])))

    def test_b2_os_8_prepare_surface_license_does_not_ask_when_the_runtime_is_os_blocked(self):
        """C: the licence chooser does not open for a runtime that cannot run on this Mac."""
        from shared import ui_surface_license

        self.mac("13.4")
        result, _ = self.doctor()
        missing_licence = [item if item.key != "surface_postprocess_license" else
                           SimpleNamespace(**dict(vars(item), status="missing")) for item in result.statuses]
        result = SimpleNamespace(targets=("surface_postprocess",), statuses=tuple(missing_licence))
        owner = SimpleNamespace(_environment_dependency_path_kwargs=lambda: {},
                                _run_environment_dependency_check=lambda: self.fail("no recheck"))
        with mock.patch.object(ui_surface_license, "choose_license", side_effect=AssertionError("no chooser")), \
                mock.patch.object(ui_surface_license, "resolve_surface_license", return_value=None), \
                mock.patch.object(ui_surface_license, "managed_surface_license_path",
                                  return_value=self.root / "none.txt"):
            self.assertIs(ui_surface_license.prepare_surface_license(owner, result), result)

    def test_b2_os_c_readiness_ignores_rows_without_a_remedy(self):
        """C: an OS-blocked row (remedy none) does not keep the setup 'blocked' (stale pending.json, repeat dialog)."""
        from shared.ui_environment_panel import readiness_blocker_count

        self.mac("13.4")
        result, status = self.doctor()
        self.assertGreater(result.blocker_count, 0)
        self.assertEqual(readiness_blocker_count(result), 0)


# --------------------------------------------------------------------------- C: Run click


class RunSkipTest(MacCase):
    """Decision 6: a stored Surface postprocess that this OS cannot run is skipped with a log line."""

    def setUp(self):
        super().setUp()
        from Testing.Python import test_ui_algorithm_availability as taa

        self.taa = taa
        availability_root = Path(tempfile.mkdtemp(dir=self.root))
        self.subject = taa.write_subject(availability_root / "s1", taa.THREE_GROUPS)
        self.blocked = {"state": "os_blocked", "remedy": "none", "macos_required": "15.0", "macos_current": "13.4",
                        "reason": "The bundled surface runtime needs macOS 15.0 or later (this Mac: 13.4)"}

    build_widget = None  # set below from RunRequestTest (the same harness as the data skip)
    click = None

    def test_b2_os_9_run_request_skips_os_blocked_surface_postprocess_and_logs_it(self):
        widget = self.build_widget(self.subject, "dti")
        widget._parameterNode.targetSurfacePostprocess = True
        before = dict(vars(widget._parameterNode))
        with mock.patch("shared.surface_runtime.surface_postprocess_os_skip", return_value=self.blocked):
            self.click(widget)
        self.assertEqual(len(self.started), 1, self.status)
        kwargs, selected = self.started[0]
        self.assertNotIn("surface_postprocess", kwargs["targets"])
        self.assertNotIn("Surface postprocess", selected)
        self.assertIn("microstructure", kwargs["targets"])
        self.assertEqual(vars(widget._parameterNode), before)  # the stored selection stays
        skipped = [line for line in self.logged if line.startswith("Skipped Surface postprocess")]
        self.assertEqual(len(skipped), 1, self.logged)
        self.assertIn("15.0", skipped[0])
        self.assertIn("13.4", skipped[0])

    def test_b2_os_9_batch_requests_skip_it_too_and_download_remedy_is_not_skipped(self):
        widget = self.build_widget(self.subject, "dti")
        widget._parameterNode.targetSurfacePostprocess = True
        with mock.patch("shared.surface_runtime.surface_postprocess_os_skip", return_value=self.blocked):
            skips = widget._data_filtered_run(True)
        self.assertIsNotNone(skips)
        self.assertFalse(skips[0].flags.surface_postprocess)
        self.assertTrue(skips[0].flags.microstructure)
        # The request built for an OS block alone equals the request of the same widget with the step unticked.
        closure = widget._data_filtered_closure(skips[0])
        widget._parameterNode.targetSurfacePostprocess = False
        unticked = widget._selected_closure()
        self.assertEqual((closure.targets, closure.requested, closure.algorithms_text),
                         (unticked.targets, unticked.requested, unticked.algorithms_text))
        freewater, noddi = widget._microstructure_stage_flags()
        self.assertEqual((skips[0].dti, skips[0].freewater, skips[0].noddi, skips[0].dki, skips[0].ifw),
                         (widget._microstructure_dti_enabled(), freewater, noddi,
                          widget._microstructure_dki_enabled(), widget._microstructure_ifw_enabled()))
        widget._parameterNode.targetSurfacePostprocess = True
        with mock.patch("shared.surface_runtime.surface_postprocess_os_skip", return_value=None):
            self.assertIsNone(widget._data_filtered_run(True))

    def test_b2_os_9_the_skip_helper_skips_only_remedy_none(self):
        from shared.surface_runtime import surface_postprocess_os_skip

        self.mac("13.4")
        self.assertEqual(surface_postprocess_os_skip({})["macos_required"], "15.0")
        self.fake_catalog("13.0")
        self.assertIsNone(surface_postprocess_os_skip({}))  # remedy download: the normal preflight path


def _borrow_run_harness():
    from Testing.Python.test_ui_algorithm_availability import RunRequestTest

    RunSkipTest.build_widget = RunRequestTest.build_widget
    RunSkipTest.click = RunRequestTest.click


_borrow_run_harness()


# --------------------------------------------------------------------------- D: managed OS floors


class ManagedFloorTest(unittest.TestCase):
    def test_b2_os_d_managed_mrtrix_floor_is_13_0_and_fsl_keeps_14_5(self):
        """Decision 8 (D). Evidence:

        * MRtrix3: the 14.5 rule came in with commit 469af300 (2026-10-31) for the Intel-Mac conda package
          ``mrtrix3=3.0.8=hee5d243_0``, which depends on ``__osx >=14.5`` (osx-64 repodata). The managed macOS route
          no longer installs it: ``profile_spec('mrtrix')`` on Darwin is the official universal tarball
          ``mrtrix3-macos-3.0.8.tar.gz`` (sha256 91fb80ff..., ``MRTRIX_TARBALL``). The bundled
          ``mrtrix-3.0.8-darwin-arm64`` repackages the same sha-pinned tarball (scripts/native/build_mrtrix_macos.sh),
          and its Mach-O minimum is 11.0 (arm64) / 10.13 (x86_64), measured with otool and
          ``shared.macho_min_os.min_macos_for_dir`` (b2 investigation report os-floor-gates). Floor:
          max(13.0, 11.0) = 13.0 (13.0 is the floor of Slicer 5.10).
        * FSL (fsl_cpu): fsl-eddy/fsl-topup need 11.0 / 10.13, but the conda-forge closure of the managed route
          (python 3.12, libcxx) and the Miniforge 26.7.2 installer were not measured. No evidence: 14.5 stays.
        """
        from shared import managed_runtimes as runtime

        for machine in ("x86_64", "arm64"):
            for version, mrtrix, fsl in (("12.7", False, False), ("13.0", True, False), ("13.4", True, False),
                                         ("14.5", True, True)):
                with mock.patch("platform.mac_ver", return_value=(version, ("", "", ""), "")):
                    self.assertEqual(runtime.supported_platform("Darwin", machine, profiles=("mrtrix",)), mrtrix,
                                     (machine, version))
                    self.assertEqual(runtime.supported_platform("Darwin", machine, profiles=("fsl_cpu",)), fsl,
                                     (machine, version))
                    self.assertEqual(runtime.supported_platform("Darwin", machine, profiles=("mrtrix", "fsl_cpu")),
                                     fsl, (machine, version))

    def test_b2_os_d_provision_names_only_the_profile_below_its_floor(self):
        """Decision 8: the provision message names a version floor only for the component it applies to."""
        from shared import managed_runtimes as runtime

        plan = {"platform_supported": True, "root": "/nowhere", "prefixes": {}, "profiles": [],
                "fsl_license_required": False}
        with mock.patch.object(runtime, "install_plan", return_value=plan), \
                mock.patch.object(runtime.platform, "system", return_value="Darwin"), \
                mock.patch.object(runtime.platform, "mac_ver", return_value=("13.4", ("", "", ""), "")):
            with self.assertRaises(RuntimeError) as caught:
                runtime.provision(("mrtrix", "fsl_cpu"), root=Path("/nowhere"))
            self.assertIn("FSL", str(caught.exception))
            self.assertIn("14.5", str(caught.exception))
            self.assertNotIn("MRtrix", str(caught.exception))
            with mock.patch.object(runtime, "setup_lock", side_effect=LookupError("floor passed")):
                with self.assertRaises(LookupError):
                    runtime.provision(("mrtrix",), root=Path("/nowhere"))

    def test_b2_os_d_fsl_text_names_the_true_cause(self):
        """Decision 8: without a bundle the text says the bundle is missing, then the managed floor."""
        from shared.selected_setup import fsl_unavailable_text

        with mock.patch("shared.native_runtimes.installed_bundle_bin_dirs", return_value=()), \
                mock.patch("platform.mac_ver", return_value=("13.4", ("", "", ""), "")):
            text = fsl_unavailable_text("Darwin", "x86_64")
            zh = fsl_unavailable_text("Darwin", "x86_64", zh=True)
        self.assertIn("no bundled FSL", text)
        self.assertIn("macOS 14.5 or later", text)
        self.assertIn("13.4", text)
        self.assertIn("未安装随包的 FSL", zh)


# --------------------------------------------------------------------------- docs and picker


class DocsAndPickerTest(MacCase):
    def test_b2_os_d_docs_state_the_managed_floors_of_the_code(self):
        """Decision 9: docs that state the managed macOS floor match the code (MRtrix 13.0, FSL 14.5)."""
        for name in ("docs/setup/MANAGED_SETUP.md", "docs/setup/MANAGED_SETUP.zh-CN.md", "docs/website/FACTS.md",
                     "docs/architecture/PRODUCT_ARCHITECTURE.md", "docs/architecture/PRODUCT_ARCHITECTURE.zh-CN.md",
                     ".agents/skills/slicerdspace/references/setup.md"):
            text = (REPO_ROOT / name).read_text(encoding="utf-8")
            self.assertNotIn("MRtrix3 and FSL require macOS 14.5", text, name)
            self.assertIn("13.0", text, name)

    def test_b2_os_6_compact_tooltip_of_surface_postprocess_names_the_block(self):
        """Decision 6: the stored tick stays and the row tooltip gives the reason and the hint (en and zh)."""
        from shared.ui_compact import surface_postprocess_block_tooltip

        self.mac("13.4")
        owner = SimpleNamespace(_environment_dependency_path_kwargs=lambda: {})
        en = surface_postprocess_block_tooltip(owner, zh=False)
        zh = surface_postprocess_block_tooltip(owner, zh=True)
        self.assertIn("15.0", en)
        self.assertIn("untick Surface postprocess", en)
        self.assertIn("15.0", zh)
        self.assertIn("取消勾选", zh)
        self.mac("15.1")
        self.assertEqual(surface_postprocess_block_tooltip(owner, zh=False), "")

    def test_b2_os_6_the_row_label_is_greyed_and_the_checkbox_stays_enabled(self):
        """Decision 6: the "Surface postprocess" label is greyed while the OS blocks the step; the checkbox is never
        disabled (the user can still untick it), and the label comes back when the block is gone."""
        from shared.ui_compact import _refresh_surface_postprocess_tooltip

        class Widget:
            def __init__(self):
                self.enabled, self.toolTip = True, "base"

            def setEnabled(self, value):
                self.enabled = bool(value)

            def setToolTip(self, value):
                self.toolTip = value

        checkbox, label = Widget(), Widget()
        owner = SimpleNamespace(_environment_dependency_path_kwargs=lambda: {}, _surfacePostprocessLabel=label)
        ui = SimpleNamespace(targetSurfacePostprocessCheckBox=checkbox)
        self.mac("13.4")
        _refresh_surface_postprocess_tooltip(owner, ui, False)
        self.assertFalse(label.enabled)
        self.assertTrue(checkbox.enabled)
        self.assertIn("15.0", checkbox.toolTip)
        self.mac("15.1")
        _refresh_surface_postprocess_tooltip(owner, ui, False)
        self.assertTrue(label.enabled)
        from shared.ui_onboarding import onboarding_text

        # b2fix F3: the onboarding tooltip of the language comes back (a stored first tooltip lost the translation).
        self.assertEqual(checkbox.toolTip, onboarding_text("en", "surface_postprocess_tooltip"))


if __name__ == "__main__":
    unittest.main()
