"""b2-setup: the setup package installs the core only; WMA atlases are offered inside the plugin (user, 2026-11-22).

Request: "By default the setup package holds only the core algorithms. The atlases must not take much space in the
setup; if the user selects WMA inside the plugin, remind them there to install the atlases."

- Default: no atlas pack on the website setup zip, the Python script builder and the CI check of the release zips.
- Wizard: no atlas group; one short line under the core item.
- C1 reinstall: a re-run of setup with no pack keeps an installed atlas and an existing WMA choice in the user config.
- In-plugin: the Run-time dialog names the atlas with its friendly title; the atlas tooltip says where the download is
  offered.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TESTS_ROOT = Path(__file__).resolve().parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

import test_installer_prune_fallback as prune  # module import: its TestCase classes must not be collected here twice

_Isolated, _Project, _on_ci = prune._Isolated, prune._Project, prune._on_ci

JS_PATH = REPO_ROOT / "docs" / "site" / "assets" / "js" / "installer-customizer.js"
REPACKAGE_PATH = REPO_ROOT / ".github" / "scripts" / "repackage_release_assets.py"
ATLAS_LINE = "WMA white-matter atlases are optional. SlicerDSpace offers the download when you select WMA."


def _node() -> str:
    node = shutil.which("node")
    if node is None:
        if _on_ci():
            raise AssertionError("node is required on CI for the JS installer copies (node is not in PATH)")
        raise unittest.SkipTest("node is not installed in PATH")
    return node


def _setup_py(state_js: str) -> str:
    code = ("const ic = require(process.argv[1]);"
            f"const st = {state_js};"
            "process.stdout.write(ic.buildSetupExtensionFiles(st).find((f) => f.name.endsWith('SlicerDSpaceSetup.py')).data);")
    return subprocess.run([_node(), "-e", code, str(JS_PATH)], capture_output=True, text=True, encoding="utf-8",
                          errors="strict", check=True).stdout


def _embedded_config(py_src: str) -> dict:
    line = next(l for l in py_src.splitlines() if l.startswith("CONFIG = _json.loads('"))
    return json.loads(line[len("CONFIG = _json.loads('"):-2])


def _repackage_module():
    spec = importlib.util.spec_from_file_location("repackage_release_assets_b2_setup", REPACKAGE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class B2SetupDefaultTests(unittest.TestCase):
    def test_b2_setup_page_default_selects_no_atlas_pack(self) -> None:
        for os_key in ("linux", "windows", "macos"):
            with self.subTest(os=os_key):
                cfg = _embedded_config(_setup_py(f"ic.defaultSetupState('{os_key}')"))
                self.assertEqual(cfg["selected_packs"], [])
                self.assertIs(cfg["auto_start"], False)

    def test_b2_setup_python_builder_default_selects_no_atlas_pack(self) -> None:
        from shared.installer_script_builder import generate_installer_script

        script = generate_installer_script(target_os="linux", artifacts={"packs": {}, "bundles": {}})
        block = script.split("# __SLICERDSPACE_PY_BEGIN__", 1)[1].split("# __SLICERDSPACE_PY_END__", 1)[0]
        self.assertIn('"selected_packs": []', block)

    def test_b2_setup_release_zip_check_expects_no_atlas_pack(self) -> None:
        mod = _repackage_module()
        source = REPACKAGE_PATH.read_text(encoding="utf-8")
        build = source[source.index("def build_setup_zips("):source.index("def build_code_overlay(")]
        self.assertIn("wmaEnabled: false", build)
        self.assertNotIn("wma_org", build)
        py_default = _setup_py("{ os: 'linux', wmaEnabled: false, wmaAtlases: [], autoStart: false }")
        mod._verify_setup_py(py_default, Path("SlicerDSpaceSetup-Linux.zip"), "linux", "linux_x64_base", {})
        py_org = _setup_py("{ os: 'linux', wmaEnabled: true, wmaAtlases: ['wma_org'], autoStart: false }")
        with self.assertRaises(RuntimeError):
            mod._verify_setup_py(py_org, Path("SlicerDSpaceSetup-Linux.zip"), "linux", "linux_x64_base", {})


class B2SetupWizardTextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.py_src = _setup_py("ic.defaultSetupState('linux')")
        start = cls.py_src.index("def show_setup_dialog(")
        cls.dialog = cls.py_src[start:cls.py_src.index("if _IS_WORKER:", start)]

    def test_b2_setup_wizard_has_no_atlas_group(self) -> None:
        self.assertNotIn("Optional WMA White-Matter Atlases", self.py_src)
        self.assertNotIn("atlas_checks", self.py_src)
        self.assertNotIn("selected white-matter atlases", self.py_src)
        for label in ("ORG Standard 74-Tract Atlas", "NABA Lifespan Atlas", "East-West Cross-Population Atlas"):
            self.assertNotIn(label, self.dialog)

    def test_b2_setup_wizard_core_first_then_one_atlas_line(self) -> None:
        self.assertEqual(self.dialog.count(ATLAS_LINE), 1)
        self.assertLess(self.dialog.index('qt.QGroupBox("Core Extension")'), self.dialog.index(ATLAS_LINE))
        self.assertLess(self.dialog.index(ATLAS_LINE), self.dialog.index("Network Mirror && Offline Directory"))
        self.assertTrue(all(ord(ch) < 128 for ch in self.dialog))
        ast.parse(self.py_src)

    def test_b2_setup_wizard_takes_packs_from_the_embedded_config(self) -> None:
        # No check box any more: an explicit state with packs (JS API, headless parity) still reaches phase A.
        body = self.dialog[self.dialog.index("    def _on_start():"):self.dialog.index("    def _on_cancel():")]
        self.assertIn('CONFIG.get("selected_packs")', body)


def _write_run_defaults(project: _Project, run_defaults: dict | None) -> None:
    payload = {"external_dependencies": {}}
    if run_defaults is not None:
        payload["run_defaults"] = run_defaults
    project.config_path.write_text(json.dumps(payload), encoding="utf-8")


class B2SetupReinstallKeepsAtlasTests(unittest.TestCase):
    """C1: a re-run of setup with the new empty default over an install that has WMA."""

    @classmethod
    def setUpClass(cls) -> None:
        prune.PruneFallbackTests.setUpClass.__func__(cls)

    _runners = prune.PruneFallbackTests._runners

    def _each_copy(self, run_defaults, check) -> None:
        for release_package in (True, False):
            for label, run in self._runners():
                with self.subTest(copy=label, release_package=release_package):
                    if run is None:
                        if _on_ci():
                            self.fail("node is required on CI for the JS installer copies (node is not in PATH)")
                        self.skipTest("node is not installed in PATH: the JS installer copy is not tested")
                    with tempfile.TemporaryDirectory() as td:
                        project = _Project(Path(td), release_package=release_package)
                        atlas_file = project.assets / "bundles" / "wma" / "naba" / "atlas.txt"
                        atlas_file.parent.mkdir(parents=True)
                        atlas_file.write_text("naba", encoding="utf-8")
                        _write_run_defaults(project, run_defaults)
                        with _Isolated(project, release_package=release_package):
                            run(project)
                        cfg = json.loads(project.config_path.read_text(encoding="utf-8"))
                        self.assertEqual(atlas_file.read_text(encoding="utf-8"), "naba", "setup must not touch an installed atlas")
                        check(cfg.get("run_defaults") or {})

    def test_b2_setup_reinstall_keeps_an_existing_wma_choice(self) -> None:
        before = {"bundle_backend": "wma,tractcloud,wmql", "wma_atlas": "naba",
                  "targets": ["microstructure", "bundles", "tractcloud", "wma"]}

        def check(defaults):
            self.assertEqual(defaults.get("bundle_backend"), "wma,tractcloud,wmql")
            self.assertEqual(defaults.get("wma_atlas"), "naba")
            self.assertIn("wma", defaults.get("targets") or [])

        self._each_copy(before, check)

    def test_b2_setup_first_install_has_no_wma(self) -> None:
        def check(defaults):
            self.assertEqual(defaults.get("bundle_backend"), "tractcloud,wmql")
            self.assertNotIn("wma", defaults.get("targets") or [])

        self._each_copy(None, check)

    def test_b2_setup_reinstall_without_wma_stays_without_wma(self) -> None:
        before = {"bundle_backend": "tractcloud,wmql", "targets": ["microstructure", "bundles"]}

        def check(defaults):
            self.assertEqual(defaults.get("bundle_backend"), "tractcloud,wmql")
            self.assertNotIn("wma", defaults.get("targets") or [])

        self._each_copy(before, check)

    def test_b2_setup_w2_reinstall_keeps_every_saved_wma_form(self) -> None:
        # Fix round W2 (Codex P1): shared.algorithm_selection.algorithms() also accepts 'all', '+' and ';' separators
        # and mixed case. A saved selection that includes WMA in any of these forms is kept as the user wrote it.
        # 'wma,future' (an unknown name) is kept too: the installer cannot judge a name, and keeping is not destructive.
        for value in ("all", "WMA+WMQL", "wma;tractcloud", "Wma, WMQL", "wma,future"):
            with self.subTest(bundle_backend=value):
                def check(defaults, value=value):
                    self.assertEqual(defaults.get("bundle_backend"), value)

                self._each_copy({"bundle_backend": value, "targets": ["microstructure", "bundles"]}, check)

    def test_b2_setup_w2_reinstall_without_wma_in_any_form_gets_the_default(self) -> None:
        # An empty value means the default 'wmql' in algorithms(): no WMA, so the no-pack default is written.
        for value in ("", "WMQL+TractCloud", "tractcloud;wmql", "wmql"):
            with self.subTest(bundle_backend=value):
                def check(defaults):
                    self.assertEqual(defaults.get("bundle_backend"), "tractcloud,wmql")

                self._each_copy({"bundle_backend": value, "targets": ["microstructure", "bundles"]}, check)

    def test_b2_setup_w2_inline_check_agrees_with_algorithms(self) -> None:
        # The inline check in each copy has the meaning of algorithms() for every value that algorithms() accepts.
        from shared.algorithm_selection import BUNDLES, algorithms

        values = ("all", "ALL", "wma", "WMA+WMQL", "wma;tractcloud", "Wma, WMQL", " wmql ; Tractcloud ", "wmql",
                  "tractcloud,wmql", "all,wma", "", "wmql+wma+tractcloud")
        for label, ns in prune_namespaces(self):
            for value in values:
                with self.subTest(copy=label, value=value):
                    self.assertEqual(ns["_backend_keeps_wma"](value), "wma" in algorithms(value, BUNDLES, "wmql"))


def prune_namespaces(case: unittest.TestCase):
    """The loaded Python text of the three installer copies (the JS ones need node)."""
    from test_b4_w4_platform import _js_payload, _load, _python_payload
    from test_installer_gui_reinstall_gap import load_setup_module

    yield "python_builder", _load(_python_payload("linux"))
    if shutil.which("node") is None:
        if _on_ci():
            case.fail("node is required on CI for the JS installer copies (node is not in PATH)")
        return
    yield "js_script", _load(_js_payload("linux"))
    yield "js_setup_module", load_setup_module()


class B2SetupW3WmaOnlyOfferTests(unittest.TestCase):
    """Fix round W3 (Codex P1): a WMA-only (or TractCloud-only) selection closes to the target 'wma' ('tractcloud'), not
    'bundles'; the atlas (data) check ran only under 'bundles', so Apply offered nothing."""

    def _missing(self, closure_targets, assets: Path, *, family="org", bundle_backends=()):
        from shared.ui_on_demand_resources import detect_missing_task_resources

        return [item["id"] for item in detect_missing_task_resources(
            targets=closure_targets, bundle_backends=bundle_backends, wma_atlas=family, brain_mask_backend="provided",
            backend_paths={"model_assets_dir": str(assets)})]

    def test_b2_setup_w3_wma_only_closure_offers_the_atlas(self) -> None:
        from shared.target_closure import expand_closure

        closure = expand_closure(("bundles",), bundle_backends=("wma",))
        self.assertIn("wma", closure.targets)
        self.assertNotIn("bundles", closure.targets)
        for family, expected in (("org", "wma_org"), ("naba", "wma_naba"), ("east-west", "wma_east_west")):
            for backends in ((), ("wma",)):  # the check must not depend on the bundle tuple
                with self.subTest(family=family, bundle_backends=backends), tempfile.TemporaryDirectory() as td:
                    self.assertEqual(self._missing(closure.targets, Path(td), family=family, bundle_backends=backends),
                                     [expected])

    def test_b2_setup_w3_wma_only_closure_with_the_atlas_offers_nothing(self) -> None:
        from shared.target_closure import expand_closure

        closure = expand_closure(("bundles",), bundle_backends=("wma",))
        with tempfile.TemporaryDirectory() as td:
            atlas = Path(td) / "bundles" / "wma"
            for child in ("ORG-800FC-100HCP", "ORG-RegAtlas-100HCP"):
                (atlas / child).mkdir(parents=True)
            (atlas / "extracted_resource_index.json").write_text("{}", encoding="utf-8")
            self.assertEqual(self._missing(closure.targets, Path(td)), [])

    def test_b2_setup_w3_old_bundles_form_still_offers_the_atlas(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(self._missing(("bundles", "tractography"), Path(td), bundle_backends=("wma",)), ["wma_org"])

    def test_b2_setup_w3_tractcloud_only_closure_offers_the_data(self) -> None:
        from shared.target_closure import expand_closure

        closure = expand_closure(("bundles",), bundle_backends=("tractcloud",))
        self.assertNotIn("bundles", closure.targets)
        with tempfile.TemporaryDirectory() as td:
            # The CSD/ACT tracking of this closure also needs the anatomy weights; only the bundle items matter here.
            found = self._missing(closure.targets, Path(td))
            self.assertEqual([i for i in found if i == "tractcloud" or i.startswith("wma")], ["tractcloud"])

    def test_b2_setup_w3_selection_without_wma_or_tractcloud_gets_no_bundle_offer(self) -> None:
        from shared.target_closure import expand_closure

        for targets, backends in ((("bundles",), ("wmql",)), (("microstructure",), ("wmql",))):
            closure = expand_closure(targets, bundle_backends=backends)
            with self.subTest(targets=targets), tempfile.TemporaryDirectory() as td:
                found = self._missing(closure.targets, Path(td), bundle_backends=backends)
                self.assertFalse([i for i in found if i.startswith("wma") or i == "tractcloud"], found)


class B2SetupC2AtlasPlanTests(unittest.TestCase):
    """C2 (real Slicer, 2026-11-22): the doctor reports the missing atlas as 'setting:wma_org_atlas' (and TractCloud
    data as 'setting:tractcloud_data_dir'), but the plan matched only the bare key, so "Download and Install" in the
    on-demand dialog and "Prepare selected analysis" never planned the atlas."""

    @staticmethod
    def _report(*keys):
        def item(key, group):
            kind = key.split(":", 1)[0] if ":" in key else "path"
            return SimpleNamespace(key=key, kind=kind, status="missing", required=True, targets=(group,), label=key,
                                   details={}, install_group=group, install_hint="configure")
        return SimpleNamespace(statuses=[item(k, "tractcloud" if "tractcloud" in k else "wma") for k in keys])

    def test_b2_setup_c2_prefixed_doctor_keys_plan_the_atlas(self) -> None:
        from shared.optional_assets import missing_optional_profiles

        report = self._report("setting:wma_org_atlas", "setting:tractcloud_data_dir")
        self.assertEqual(missing_optional_profiles(report, "provided", "org"), ["wma", "tractcloud"])
        self.assertEqual(missing_optional_profiles(report, "provided", "naba"), ["wma-naba", "tractcloud"])
        sub = self._report("path:wma_org_atlas_ORG-800FC-100HCP")
        self.assertEqual(missing_optional_profiles(sub, "provided", "org"), ["wma"])

    def test_b2_setup_c2_plan_handles_the_prefixed_atlas_rows(self) -> None:
        from shared.selected_setup import build_setup_plan

        report = self._report("setting:wma_org_atlas", "setting:tractcloud_data_dir")
        plan = build_setup_plan(report, brain_mask_backend="provided", system="Linux", machine="x86_64")
        self.assertEqual(plan["optional_asset_profiles"], ["wma", "tractcloud"])
        self.assertEqual(plan["manual_items"], [])


class B2SetupInPluginWordingTests(unittest.TestCase):
    def test_b2_setup_run_dialog_names_the_atlas(self) -> None:
        from shared.setup_components import plan_components
        from shared.ui_selected_setup import _compact_component_rows

        plan = {"optional_asset_profiles": ["wma-naba"], "optional_asset_size_bytes": 3365003883}
        components = plan_components(plan)
        for lang, title in (("en", "WMA NABA White-Matter Atlas"), ("zh", "WMA NABA 全龄白质图谱")):
            with self.subTest(lang=lang):
                rows = "\n".join(_compact_component_rows(SimpleNamespace(_onboardingLanguage=lang), components, set()))
                self.assertIn(title, rows)
                self.assertNotIn("Resources: wma", rows)

    def test_b2_setup_run_dialog_keeps_an_unknown_profile_id(self) -> None:
        from shared.setup_components import plan_components
        from shared.ui_selected_setup import _compact_component_rows

        components = plan_components({"optional_asset_profiles": ["wma", "future-pack"]})
        rows = "\n".join(_compact_component_rows(SimpleNamespace(_onboardingLanguage="en"), components, set()))
        self.assertIn("WMA ORG White-Matter Atlas", rows)
        self.assertIn("future-pack", rows)

    def test_b2_setup_atlas_tooltip_says_where_the_download_is_offered(self) -> None:
        from shared.ui_advanced_layout import apply_advanced_language

        for lang, expected, banned in (
            ("en", "offers the download when WMA is selected", "in the Environment panel"),
            ("zh", "勾选 WMA 时", "在环境面板按需下载"),
        ):
            with self.subTest(lang=lang):
                combo = MagicMock()
                owner = SimpleNamespace(_advancedCategorySelector=MagicMock(currentIndex=0), _onboardingLanguage=lang,
                                        ui=SimpleNamespace(wmaAtlasLabel=MagicMock(), wmaAtlasComboBox=combo))
                with patch("shared.ui_experiments.apply_language"):
                    apply_advanced_language(owner)
                tip = combo.setToolTip.call_args[0][0]
                self.assertIn(expected, tip)
                self.assertNotIn(banned, tip)


class B2SetupW1OnlySlicerDmriTests(unittest.TestCase):
    """Fix round W1 (user decision 2026-11-22): the website and the documents recommend only SlicerDMRI."""

    EN = ("To view tractography results, also install the SlicerDMRI extension from Slicer’s Extensions Manager. "
          "It is recommended for viewing only; the default analysis runs without it.")
    ZH = "查看纤维束结果时，建议另外通过 Slicer 的 Extensions Manager 安装 SlicerDMRI 扩展。它仅用于查看结果，默认分析流程不依赖它。"

    def test_b2_setup_w1_website_recommends_only_slicerdmri(self) -> None:
        site = REPO_ROOT / "docs" / "site"
        for page, sentence in (("index.html", self.EN), ("zh.html", self.ZH)):
            with self.subTest(page=page):
                text = (site / page).read_text(encoding="utf-8")
                self.assertEqual(text.count(f'<p class="start-note-inline">{sentence}</p>'), 1)
                self.assertNotIn("SlicerFreeSurfer", text)

    def test_b2_setup_w1_documents_do_not_recommend_slicerfreesurfer(self) -> None:
        paths = [REPO_ROOT / "README.md", REPO_ROOT / "README.zh-CN.md", REPO_ROOT / "docs" / "website" / "zh-CN.json",
                 REPO_ROOT / ".agents" / "skills" / "slicerdspace" / "references" / "setup.md",
                 *sorted((REPO_ROOT / "docs" / "setup").glob("*.md"))]
        for path in paths:
            with self.subTest(path=path.relative_to(REPO_ROOT).as_posix()):
                self.assertNotIn("SlicerFreeSurfer", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
