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


if __name__ == "__main__":
    unittest.main()
