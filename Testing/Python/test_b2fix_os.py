"""Batch 2 fix round, lane os: the review findings F1-F8 (one test or more per finding, each red before the fix).

Lead decision for F1: when Surface postprocess is OS-blocked (remedy 'none'), it is skipped, and every ticked target
whose closure requires it (SAF) is skipped with it. The rest of the selection runs. The skip is named in the readiness
warning, in the run log and in batch mode, in English and Chinese. If nothing runnable remains, Run does not start and
the status says why. An OS block alone never starts the Prepare/Run auto-run loop.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Testing.Python.test_b2_os_block import MacCase, _catalog_record, _import_module_widget  # noqa: E402
from Testing.Python.test_preproc_fsl_gate import make_surface_bundle  # noqa: E402

BLOCKED = {"state": "os_blocked", "remedy": "none", "macos_required": "15.0", "macos_current": "13.4",
           "reason": "The bundled surface runtime needs macOS 15.0 or later (this Mac: 13.4)",
           "reason_zh": "随包的表面运行环境需要 macOS 15.0 或更高版本（本机：13.4）"}


def _runner_closure(kwargs):
    """The closure the runner recomputes from the request (the CLI gets requested_targets + tractography_backend)."""
    from shared.orchestrator import orchestrator_closure

    return orchestrator_closure(
        tuple(kwargs.get("requested_targets") or kwargs["targets"]),
        tractography_backend=kwargs["tractography_backend"], bundle_backend=kwargs["bundle_backend"],
        microstructure_run_free_water=kwargs["microstructure_run_free_water"],
        microstructure_run_noddi=kwargs["microstructure_run_noddi"],
        microstructure_run_dki=kwargs["microstructure_run_dki"],
        microstructure_run_ifw=kwargs["microstructure_run_ifw"])


class RunClickCase(MacCase):
    """The Run click harness of the data skip (test_ui_algorithm_availability.RunRequestTest)."""

    def setUp(self):
        super().setUp()
        from Testing.Python import test_ui_algorithm_availability as taa

        self.taa = taa
        self.subject = taa.write_subject(Path(tempfile.mkdtemp(dir=self.root)) / "s1", taa.THREE_GROUPS)

    def widget(self, batch=False, **node):
        from Testing.Python.test_ui_algorithm_availability import RunRequestTest

        widget = RunRequestTest.build_widget(self, self.subject, "dti", batch=batch)
        for key, value in node.items():
            setattr(widget._parameterNode, key, value)
        return widget

    def click(self, widget, blocked=BLOCKED, auto_install=True):
        """Click Run on a Mac where the OS blocks Surface postprocess. The preflight blocks the surface step only."""
        widget._parameterNode.autoInstallPythonDependencies = auto_install
        self.prepare_calls, self.preflight_targets = [], []

        def blockers(_result, targets):
            self.preflight_targets.append(tuple(targets))
            return {"surface_postprocess": "missing"} if "surface_postprocess" in targets else {}

        mw = widget._mw
        with mock.patch("shared.surface_runtime.surface_postprocess_os_skip", return_value=blocked), \
                mock.patch.object(mw, "preflight_blockers_for_targets", side_effect=blockers), \
                mock.patch("shared.ui_selected_setup.prepare_selected_environment",
                           side_effect=lambda *a, **k: self.prepare_calls.append(k)), \
                mock.patch("shared.setup_resume.pending_setup", return_value=None), \
                mock.patch("shared.ui_input_inspection.inspect_input_directory", return_value={"mode": "subject_root"}), \
                mock.patch("shared.ui_compact.sync_detected_input_format"):
            widget._onRunSelectedClicked()


class F1RunSkipTest(RunClickCase):
    def test_b2fix_os_F1_saf_is_skipped_with_the_os_blocked_surface_postprocess_and_no_prepare_starts(self):
        """F1 (Opus P1-1, Codex P1-a): SAF's closure re-added the blocked step, so the click went to Prepare and looped.
        The SAF tracking algorithm is skipped; CSD/ACT tractography runs; the runner's own closure has no surface step."""
        widget = self.widget(targetTractography=True, orchestrationTractographyBackend="csd_act,saf",
                             autoInstallPythonDependencies=True)
        before = dict(vars(widget._parameterNode))
        self.click(widget)
        self.assertEqual(self.prepare_calls, [])
        self.assertEqual(len(self.started), 1, self.status)
        kwargs, _ = self.started[0]
        for name in ("surface_postprocess", "saf"):
            self.assertNotIn(name, kwargs["targets"])
            self.assertNotIn(name, _runner_closure(kwargs).targets)
        self.assertIn("tractography", kwargs["targets"])
        self.assertEqual(kwargs["tractography_backend"], "csd_act")
        self.assertEqual(vars(widget._parameterNode), before)  # the stored selection stays
        skipped = [line for line in self.logged if line.startswith("Skipped CSD + SAF")]
        self.assertEqual(len(skipped), 1, self.logged)
        self.assertIn("Surface postprocess", skipped[0])
        self.assertIn("15.0", skipped[0])

    def test_b2fix_os_F1_saf_only_skips_every_target_that_needs_it_and_never_runs_csd_instead(self):
        """F1: with SAF as the only tracking algorithm, Tractography and Bundles need the blocked step: both skipped.
        No CSD/ACT is put in their place; Microstructure runs."""
        widget = self.widget(targetTractography=True, targetBundles=True, targetSurfacePostprocess=True,
                             orchestrationTractographyBackend="saf")
        self.click(widget)
        self.assertEqual(self.prepare_calls, [])
        self.assertEqual(len(self.started), 1, self.status)
        kwargs, selected = self.started[0]
        runner = _runner_closure(kwargs)
        for name in ("surface_postprocess", "saf", "tractography", "bundles", "surface"):
            self.assertNotIn(name, runner.targets)
        self.assertIn("microstructure", runner.targets)
        self.assertEqual(selected, ["Microstructure"])
        names = [line.split(":")[0] for line in self.logged if line.startswith("Skipped")]
        self.assertEqual(names, ["Skipped Surface postprocess", "Skipped CSD + SAF", "Skipped Tractography",
                                 "Skipped Fiber bundles"])

    def test_b2fix_os_F1_nothing_runnable_does_not_start_and_the_status_says_why(self):
        """F1: only the blocked step is ticked: no run, no Prepare, and the status names the OS reason (en and zh)."""
        for lang, needle in (("en", "macOS 15.0"), ("zh", "本机")):
            with self.subTest(lang=lang):
                widget = self.widget(targetSurfacePostprocess=True, targetMicrostructure=False)
                widget._onboardingLanguage = lang
                self.click(widget)
                self.assertEqual((self.started, self.prepare_calls), ([], []))
                text = " ".join(str(part) for call in self.status for part in call)
                self.assertIn(needle, text)
                self.assertIn("15.0", text)

    def test_b2fix_os_F1_a_download_remedy_is_not_skipped(self):
        """F1 scope: remedy 'download' is no OS skip; the step stays in the request and Prepare installs it."""
        widget = self.widget(targetTractography=True, orchestrationTractographyBackend="csd_act,saf")
        self.click(widget, blocked=None)
        self.assertEqual(self.prepare_calls, [{"auto_run_after": True}])


class F8BatchLogTest(RunClickCase):
    def test_b2fix_os_F8_batch_run_logs_the_skip_before_submitting(self):
        """F8 (Opus P2, Codex P2): the batch branch returned before _log_data_skips."""
        batch_root = self.root / "batch"
        batch_root.mkdir()
        widget = self.widget(batch=True, targetSurfacePostprocess=True, batchRoot=str(batch_root))
        self.click(widget)
        self.assertEqual(len(self.started), 1, (self.status, self.logged))
        self.assertEqual(self.started[0][0], "batch")
        skipped = [line for line in self.logged if line.startswith("Skipped Surface postprocess")]
        self.assertEqual(len(skipped), 1, self.logged)
        submit = next(i for i, line in enumerate(self.logged) if line.startswith("Submitting batch"))
        self.assertLess(self.logged.index(skipped[0]), submit)


# --------------------------------------------------------------------------- F1 readiness, F2, F8 warning


def _os_row(zh_notes=BLOCKED["reason_zh"]):
    return SimpleNamespace(key="surface_postprocess_runtime", label="Surface postprocessing components",
                           install_group="surface-runtime", kind="setting", targets=("surface_postprocess",),
                           required=True, status="missing", install_hint="Upgrade macOS.",
                           details={"state": "os_blocked", "remedy": "none", "notes": BLOCKED["reason"],
                                    "notes_zh": zh_notes})


class PrepareReadinessTest(unittest.TestCase):
    def setUp(self):
        from Testing.Python.test_selected_setup import SelectedSetupTests

        self.harness = SelectedSetupTests("test_ready_environment_skips_confirmation_installation_and_redundant_check")
        self.harness.setUp()
        self.addCleanup(self.harness.doCleanups)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.temp = Path(tmp.name)
        for patcher in (mock.patch("shared.user_config.default_user_config_path",
                                   return_value=self.temp / "config.json"),
                        mock.patch.dict("os.environ", {"SLICERDSPACE_USER_CONFIG": str(self.temp / "config.json")})):
            patcher.start()
            self.addCleanup(patcher.stop)

    def prepare(self, lang="en", requests=(), job=None, **kwargs):
        from Testing.Python.test_selected_setup import result
        from shared.ui_selected_setup import prepare_selected_environment

        # The only open rows are the OS-blocked ones (the closure is kept to "validation", so no asset step is planned).
        report = result(_os_row(), targets=("validation",), blockers=1)
        owner = self.harness.owner(report)
        owner._onboardingLanguage = lang
        context, _ = self.harness.run_preparation(owner, report)
        self.requests = mock.Mock(return_value=list(requests))
        with context, mock.patch("shared.setup_selftest.requests_for_report", self.requests), \
                mock.patch("shared.ui_setup_jobs.start_setup_job",
                           side_effect=job or (lambda *a, **k: self.fail("no job"))), \
                mock.patch("shared.ui_selected_setup._schedule_auto_run") as self.auto_run:
            prepare_selected_environment(owner, report, **kwargs)
        self.record = json.loads(next(self.temp.glob("setup-history/*/result.json")).read_text())
        return owner

    def test_b2fix_os_F1_an_os_block_alone_never_starts_the_auto_run(self):
        """F1: 'ready' came from the discounted count, and auto-run clicked Run again (the loop). The auto-run
        condition uses the raw count, which the Run filter did not change; the 'ready' label keeps the discount."""
        self.prepare(auto_run_after=True)
        self.assertEqual(self.record["status"], "ready")
        self.auto_run.assert_not_called()

    def test_b2fix_os_F2_ready_needs_the_functional_self_test_also_with_os_blocked_rows(self):
        """F2 (Opus P1-2): functional_check used the raw count and skipped the self-test; a failed test is not ready."""
        self.prepare(requests=[{"python": "python"}],
                     job=lambda widget, title, commands, done: done("failed", "Synthetic self-test failed", None))
        self.requests.assert_called_once()
        self.assertEqual(self.record["status"], "failed")

    def test_b2fix_os_F8_readiness_warning_names_the_skip_in_chinese(self):
        """F8 (Codex P2): the warning appended to the Chinese status was English (label and reason)."""
        owner = self.prepare(lang="zh")
        message = owner._setStatus.call_args.args[1]
        self.assertIn("表面后处理", message)
        self.assertIn(BLOCKED["reason_zh"], message)
        self.assertNotIn("Surface postprocessing components", message)
        self.assertNotIn(BLOCKED["reason"], message)
        english = self.prepare(lang="en")._setStatus.call_args.args[1]
        self.assertIn("Surface postprocess", english)
        self.assertIn(BLOCKED["reason"], english)


# --------------------------------------------------------------------------- F3 tooltip


class F3TooltipTest(MacCase):
    def test_b2fix_os_F3_surface_postprocess_tooltip_keeps_its_translation(self):
        """F3 (Opus P1-3): the first tooltip was stored and written back on every refresh, so a language switch
        lost the translated tooltip on every platform. Without a block the onboarding text of the language is shown."""
        from shared.ui_compact import _refresh_surface_postprocess_tooltip
        from shared.ui_onboarding import onboarding_text

        class Widget:
            def __init__(self, tooltip):
                self.enabled, self.toolTip = True, tooltip

            def setEnabled(self, value):
                self.enabled = bool(value)

            def setToolTip(self, value):
                self.toolTip = value

        checkbox = Widget("UI-file English tooltip")
        owner = SimpleNamespace(_environment_dependency_path_kwargs=lambda: {}, _surfacePostprocessLabel=Widget(""))
        ui = SimpleNamespace(targetSurfacePostprocessCheckBox=checkbox)
        self.mac("15.1")
        _refresh_surface_postprocess_tooltip(owner, ui, False)
        checkbox.setToolTip(onboarding_text("zh", "surface_postprocess_tooltip"))  # apply_onboarding_language
        _refresh_surface_postprocess_tooltip(owner, ui, True)  # then apply_compact_language refreshes the rows
        self.assertEqual(checkbox.toolTip, onboarding_text("zh", "surface_postprocess_tooltip"))
        self.mac("13.4")
        _refresh_surface_postprocess_tooltip(owner, ui, True)
        self.assertIn("15.0", checkbox.toolTip)
        self.mac("15.1")
        _refresh_surface_postprocess_tooltip(owner, ui, False)
        self.assertEqual(checkbox.toolTip, onboarding_text("en", "surface_postprocess_tooltip"))


# --------------------------------------------------------------------------- F4, F5 runtime state


class F4F5RuntimeStateTest(MacCase):
    def test_b2fix_os_F4_the_doctor_uses_the_configured_asset_root(self):
        """F4 (Codex P1-b): an unpublished 13.0 record whose archive is in the configured model_assets_dir is remedy
        'download' (on-demand and Run said so) but the inspector looked in the default asset folder: remedy 'none'."""
        from shared.environment import EnvironmentDoctorConfig, run_environment_doctor

        self.fake_catalog("13.0", published=False)
        self.mac("13.4")
        custom = self.root / "custom-assets"
        archive = custom / _catalog_record()["asset"]
        archive.parent.mkdir(parents=True)
        archive.write_bytes(b"archive")
        licence = self.root / "license.txt"
        licence.write_text("name\nkey\n")
        with mock.patch("shared.environment.build_dependency_specs", return_value=()), \
                mock.patch("shared.surface_license.resolve_surface_license", return_value=licence), \
                mock.patch("subprocess.run", side_effect=AssertionError("probe must not start")):
            result = run_environment_doctor(EnvironmentDoctorConfig(
                targets=("surface_postprocess",), repo_root=REPO_ROOT, model_assets_dir=custom))
        row = {item.key: item for item in result.statuses}["surface_postprocess_runtime"]
        self.assertEqual(row.details["remedy"], "download")
        self.assertTrue(row.details["build_required"])

    def test_b2fix_os_F5_os_blocked_default_prefix_runtime_is_replaceable_by_a_catalog_download(self):
        """F5 (Codex P1-c): an installed macOS-15 runtime in the default prefix forced remedy 'none' although the
        catalog offers a 13.0 build. prebuilt_surface.prepare() validates and replaces that prefix, so the remedy is
        'download': the on-demand item has the catalog size and the Run does not skip the step."""
        from shared.surface_runtime import surface_postprocess_os_skip, surface_runtime_state

        installed = make_surface_bundle(self.root / "user-cache", minos=(15, 0))
        installed.rename(self.prefix)
        self.fake_catalog("13.0")
        self.mac("13.4")
        self.atlas_present(True)
        state = surface_runtime_state()
        self.assertEqual((state["state"], state["remedy"]), ("os_blocked", "download"))
        self.assertEqual(state["download_bytes"], _catalog_record()["bytes"])
        self.assertIsNone(surface_postprocess_os_skip({}))
        self.assertEqual([item["size_bytes"] for item in self.surface_items()], [_catalog_record()["bytes"]])

    def test_b2fix_os_F5_an_explicit_blocked_runtime_stays_remedy_none(self):
        """F5 scope: a path the user set is kept by the resolver, so a download cannot replace it."""
        from shared.surface_runtime import surface_runtime_state

        explicit = make_surface_bundle(self.root / "mine", minos=(15, 0))
        self.fake_catalog("13.0")
        self.mac("13.4")
        state = surface_runtime_state(runtime=explicit)
        self.assertEqual((state["state"], state["remedy"]), ("os_blocked", "none"))


# --------------------------------------------------------------------------- F6 on-demand


class F6OnDemandTest(RunClickCase):
    def items(self, widget):
        from shared import ui_on_demand_resources as od

        return {item["id"]: item for item in od.detect_missing_task_resources(widget)}

    def test_b2fix_os_F6_blocked_surface_postprocess_adds_no_surface_weights_item(self):
        """F6 (Codex P1-d): the closure of the blocked step still added the 223 MB DDSurfer weights item."""
        self.mac("13.4")
        self.atlas_present(False)
        widget = self.widget(targetSurfacePostprocess=True, targetMicrostructure=False)
        items = self.items(widget)
        self.assertNotIn("surface", items)
        self.assertNotIn("surface_postprocess", items)

    def test_b2fix_os_F6_ticked_ddsurfer_keeps_its_weights_item(self):
        """F6 scope: a prerequisite that a runnable target selects by itself stays."""
        self.mac("13.4")
        self.atlas_present(False)
        widget = self.widget(targetSurfacePostprocess=True, targetSurface=True, targetMicrostructure=False)
        self.assertIn("surface", self.items(widget))


# --------------------------------------------------------------------------- F7 chooser


class F7ChooserTest(MacCase):
    def test_b2fix_os_F7_choose_freesurfer_folder_clears_the_blocked_runtime_override(self):
        """F7 (Codex P1-e): the resolver prefers postprocess_runtime, so the chosen FreeSurfer folder had no effect."""
        mw = _import_module_widget(self)
        widget = mw.SharedWidgetMixin()
        saved, logged = [], []
        runtime_edit, home_edit = mock.Mock(), mock.Mock()
        widget._environmentPathLineEdits = {"postprocess_runtime": runtime_edit,
                                            "postprocess_freesurfer_home": home_edit}
        widget._is_model_resource_item = lambda item: False
        widget._read_environment_path_setting = lambda key: {"postprocess_runtime": "/blocked/runtime"}.get(key, "")
        widget._line_edit_text = lambda edit: "/blocked/runtime" if edit is runtime_edit else ""
        widget._save_environment_path_setting = lambda key, value: saved.append((key, value))
        widget.appendLog = logged.append
        seen = {}
        widget._run_environment_dependency_check = lambda: seen.setdefault("runtime_edit", runtime_edit.setText.call_args)
        widget._show_environment_results_dialog = lambda result: None
        mw.qt.QFileDialog.getExistingDirectory.return_value = "/Applications/freesurfer"
        row = _os_row()
        self.assertTrue(widget._configure_path_for_environment_item(row))
        self.assertIn(("postprocess_freesurfer_home", "/Applications/freesurfer"), saved)
        self.assertIn(("postprocess_runtime", ""), saved)
        self.assertEqual(seen["runtime_edit"], mock.call(""))  # cleared before the re-check reads the widgets
        # Another row keeps the override.
        saved.clear()
        other = SimpleNamespace(**dict(vars(row), key="env:freesurfer_home", install_group="freesurfer", details={}))
        widget._configure_path_for_environment_item(other)
        self.assertNotIn(("postprocess_runtime", ""), saved)


if __name__ == "__main__":
    unittest.main()
