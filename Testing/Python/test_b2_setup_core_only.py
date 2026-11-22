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

    def test_b2_setup_reinstall_preserves_existing_custom_model_and_wma_atlas_dirs(self) -> None:
        for label, run in self._runners():
            with self.subTest(copy=label):
                if run is None:
                    if _on_ci():
                        self.fail("node is required on CI for the JS installer copies (node is not in PATH)")
                    self.skipTest("node is not installed in PATH: the JS installer copy is not tested")
                with tempfile.TemporaryDirectory() as td:
                    project = _Project(Path(td), release_package=True)
                    default_assets = Path(td) / "default-assets"
                    (project.module_root / "shared" / "model_assets.py").write_text(
                        "from pathlib import Path\n"
                        f"def configured_model_assets_root(explicit=None):\n"
                        f"    return Path(explicit) if explicit else Path({str(default_assets)!r})\n",
                        encoding="utf-8",
                    )
                    custom_models = Path(td) / "user-models"
                    custom_wma = Path(td) / "user-wma-atlas"
                    custom_models.mkdir()
                    custom_wma.mkdir()
                    project.write_config({
                        "model_assets_dir": str(custom_models),
                        "wma_atlas_dir": str(custom_wma),
                    })
                    # Simulate wizard run with NO custom model folder selected (model_assets_dir="")
                    with _Isolated(project, release_package=True):
                        if label == "js_setup_module":
                            ns = prune.load_setup_module()
                            ns["CONFIG"] = dict(ns["CONFIG"], model_assets_dir="", selected_packs=[])
                            events = []
                            info = {
                                "mod_root": str(project.module_root), "ext_temps": [],
                                "mirror": "", "proxy": "", "offdir": "", "iit_file": "",
                                "pack_archives": [], "asset_temps": [], "sel_packs": [],
                            }
                            ns["_run_phase_c_core"](info, lambda kind, payload: events.append((kind, payload)))
                            errors = [p for k, p in events if k == "error"]
                            self.assertEqual(errors, [])
                        else:
                            payload = self.python_payload if label == "python_builder" else self.js_payload
                            ns = prune._load(payload)
                            prune._run_script_main(ns, project, network={"model_assets_dir": ""})
                    deps = project.read_deps()
                    self.assertEqual(deps.get("model_assets_dir"), str(custom_models))
                    self.assertEqual(deps.get("wma_atlas_dir"), str(custom_wma))


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


# ---------------------------------------------------------------- W4: the wizard applies the code update last
W4_HINT_EN = 'The latest code update was not applied. After the restart, click "Check for updates" in SlicerDSpace.'
W4_HINT_ZH = "最新的代码更新未能应用。重启后请在 SlicerDSpace 中点击“检查更新”。"
W4_STEP = "_apply_code_update"

_W4_UPDATE_USER_CONFIG = '''


def update_user_config(updates, path=None):
    """Test addition: the published save_network_settings() needs it (the published user_config has it)."""
    cfg = load_user_config()
    for key, value in dict(updates or {}).items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    write_user_config(cfg)
    return cfg, _path()
'''

_W4_RUNTIME_SPEC = {
    "version": 1,
    "files": ["SlicerDSpace.py", "scripts/extension_runtime/runtime_resources.py",
              "scripts/extension_runtime/runtime_resources.json"],
    "trees": [{"root": "shared", "patterns": ["*.py"]}],
    "excluded_components": ["__pycache__"],
    "excluded_prefixes": [],
    "excluded_globs": [],
}

# A stand-in for the installed shared.code_updater with the two published signatures. It records each call in the file
# that W4_STUB_CALLS names and answers as W4_STUB_MODE says.
_W4_STUB_UPDATER = '''import json, os


def _record(name, kwargs):
    path = os.environ["W4_STUB_CALLS"]
    calls = []
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            calls = json.load(fh)
    calls.append([name, {k: (str(v) if v is not None and k in ("module_root", "offline_dir") else v)
                         for k, v in kwargs.items() if k != "manifest"}])
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(calls, fh)


def check_for_code_update(*, module_root=None, manifest_url=None, manifest=None, timeout=12, offline_dir=None,
                          mirror_prefix=None, proxy_url=None, urlopen_fn=None):
    _record("check", dict(module_root=module_root, timeout=timeout, offline_dir=offline_dir,
                          mirror_prefix=mirror_prefix, proxy_url=proxy_url))
    mode = os.environ["W4_STUB_MODE"]
    if mode == "fail":
        raise RuntimeError("Failed to fetch update manifest: network is down")
    status = {"up_to_date": "up_to_date", "abi": "runtime_upgrade_required"}.get(mode, "update_available")
    return {"status": status, "remote": {"version": "0.1.2", "release_tag": "v0.1.2", "git_short_commit": "abc1234"}}


def download_and_apply_code_overlay(*, module_root=None, manifest_url=None, manifest=None, timeout=60, offline_dir=None,
                                    mirror_prefix=None, proxy_url=None, urlopen_fn=None, progress_callback=None,
                                    allow_git_checkout=False, _apply_hook=None):
    _record("apply", dict(module_root=module_root, timeout=timeout, offline_dir=offline_dir,
                          mirror_prefix=mirror_prefix, proxy_url=proxy_url, has_manifest=manifest is not None))
    if os.environ["W4_STUB_MODE"] == "apply_fails":
        raise PermissionError("Extension directory is not writable")
    return {"status": "applied", "version": "0.1.2", "release_tag": "v0.1.2", "git_short_commit": "abc1234",
            "updated_files": 3, "removed_files": 0, "restart_required": True}
'''


def _published_text(case: unittest.TestCase, rel: str) -> str:
    """A file of the commit that the published base packages are built from (skip off CI, fail on CI when missing)."""
    sha = prune.RELEASE_PACKAGE_SOURCE_COMMIT
    git = shutil.which("git")
    out = None
    if git is not None:
        out = subprocess.run([git, "show", f"{sha}:{rel}"], cwd=REPO_ROOT, capture_output=True, text=True,
                             encoding="utf-8")
    if out is None or out.returncode != 0:
        why = "git is not installed" if out is None else f"commit {sha} is not in this checkout"
        if _on_ci():
            case.fail(f"{why}; the workflow step '{prune.FETCH_STEP_NAME}' must fetch it")
        case.skipTest(f"{why}; run: git fetch --no-tags --depth=1 origin {sha}")
    return out.stdout


def _w4_tree_sha(root: Path) -> tuple[str, list[str]]:
    """The code tree hash of the published compute_code_tree_sha256() (same algorithm, read in 2e8f7315)."""
    import hashlib

    from scripts.extension_runtime.runtime_resources import runtime_files

    digest = hashlib.sha256()
    files = runtime_files(root)
    for rel in files:
        digest.update(f"{rel}\0{hashlib.sha256((root / rel).read_bytes()).hexdigest()}\n".encode("utf-8"))
    return digest.hexdigest(), files


class _W4Modules:
    """Inside _Isolated: also drop the repo's 'scripts' package, so the installed tree's own copy is imported."""

    def __enter__(self):
        for name in [n for n in sys.modules if n == "scripts" or n.startswith("scripts.")]:
            del sys.modules[name]
        return self

    def __exit__(self, *exc):
        return False


class B2SetupW4CodeUpdateTests(unittest.TestCase):
    """Fix round W4 (Opus, solved in code; user scope change): the wizard applies the published code update as its last
    step, best effort, with the installed package's own shared.code_updater and the wizard's network choices."""

    @classmethod
    def setUpClass(cls) -> None:
        prune.PruneFallbackTests.setUpClass.__func__(cls)

    def _copies(self, network=None):
        yield "python_builder", lambda project: prune._run_script_main(prune._load(self.python_payload), project,
                                                                      network=network)
        if self.js_payload is None:
            yield "js_script", None
            yield "js_setup_module", None
            return
        yield "js_script", lambda project: prune._run_script_main(prune._load(self.js_payload), project, network=network)
        yield "js_setup_module", lambda project: prune._run_setup_module(project, network=network)

    def _skip_without_node(self, run) -> None:
        if run is None:
            if _on_ci():
                self.fail("node is required on CI for the JS installer copies (node is not in PATH)")
            self.skipTest("node is not installed in PATH: the JS installer copy is not tested")

    # (a) + (c): the published updater, a local overlay in the wizard's offline folder
    def _published_project(self, td: Path):
        project = _Project(td, release_package=True)
        m = project.module_root
        (m / "shared" / "code_updater.py").write_text(_published_text(self, "shared/code_updater.py"), encoding="utf-8")
        (m / "shared" / "network_config.py").write_text(_published_text(self, "shared/network_config.py"),
                                                       encoding="utf-8")
        with (m / "shared" / "user_config.py").open("a", encoding="utf-8") as fh:
            fh.write(_W4_UPDATE_USER_CONFIG)
        rr = m / "scripts" / "extension_runtime"
        rr.mkdir(parents=True)
        (rr / "runtime_resources.py").write_text(
            _published_text(self, "scripts/extension_runtime/runtime_resources.py"), encoding="utf-8")
        (rr / "runtime_resources.json").write_text(json.dumps(_W4_RUNTIME_SPEC), encoding="utf-8")
        (m / "SlicerDSpace.py").write_text("VALUE = 1\n", encoding="utf-8")
        project.config_path.write_text(json.dumps({"external_dependencies": {}}), encoding="utf-8")

        # The overlay: the same tree with a new entry file and one new module (the marker).
        v2 = td / "v2"
        _, files = _w4_tree_sha(m)
        for rel in files:
            (v2 / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(m / rel, v2 / rel)
        (v2 / "SlicerDSpace.py").write_text("VALUE = 2\n", encoding="utf-8")
        (v2 / "shared" / "w4_overlay_marker.py").write_text("MARKER = 'w4'\n", encoding="utf-8")
        v2_sha, v2_files = _w4_tree_sha(v2)
        offline = td / "offline"
        offline.mkdir()
        from test_code_updater import _build_tar_gz

        archive = offline / "SlicerDSpace-v0.1.2-code-overlay.tar.gz"
        size, sha = _build_tar_gz(archive, {rel: (v2 / rel).read_bytes() for rel in v2_files})
        manifest = {
            "schema_version": 1, "version": "0.1.2", "runtime_abi_version": 1, "code_sha256": v2_sha,
            "git_commit": "abc1234def", "summary": "W4 test overlay",
            "overlay": {"filename": archive.name, "size_bytes": size, "sha256": sha,
                        "url": f"https://github.com/NeuroDMRI/SlicerDSpace/releases/download/v0.1.2/{archive.name}"},
        }
        (offline / "update_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return project, offline, v2_sha

    def test_b2_setup_w4_published_updater_applies_the_offline_overlay(self) -> None:
        for label, run in self._copies():
            with self.subTest(copy=label):
                self._skip_without_node(run)
                with tempfile.TemporaryDirectory() as td:
                    project, offline, v2_sha = self._published_project(Path(td))
                    network = {"offline_dir": str(offline)}
                    run_with = dict(self._copies(network))[label]
                    with _Isolated(project, release_package=True), _W4Modules():
                        logs = run_with(project)
                    m = project.module_root
                    self.assertTrue((m / "shared" / "w4_overlay_marker.py").is_file(), logs)
                    self.assertEqual((m / "SlicerDSpace.py").read_text(encoding="utf-8"), "VALUE = 2\n")
                    self.assertEqual(_w4_tree_sha(m)[0], v2_sha)
                    applied = [line for line in logs if line.startswith("Applied the latest code update")]
                    self.assertEqual(len(applied), 1, logs)
                    self.assertIn("v0.1.2", applied[0])
                    self.assertNotIn(W4_HINT_EN, logs)
                    if label != "js_setup_module":
                        self.assertEqual(logs[-1], "Installation completed successfully.")

                    # (c) A second run over the updated tree: up to date, no hint, no second apply.
                    with _Isolated(project, release_package=True), _W4Modules():
                        logs2 = run_with(project)
                    self.assertFalse([line for line in logs2 if line.startswith("Applied the latest code update")])
                    self.assertNotIn(W4_HINT_EN, logs2)

    # (b) + (c) with a stand-in updater: arguments, failures and the hint
    def _stub_case(self, mode: str, *, stub: bool = True, network=None):
        results = []
        for label, run in self._copies(network):
            with self.subTest(copy=label, mode=mode):
                self._skip_without_node(run)
                with tempfile.TemporaryDirectory() as td:
                    project = _Project(Path(td), release_package=True)
                    calls_file = Path(td) / "calls.json"
                    if stub:
                        (project.module_root / "shared" / "code_updater.py").write_text(_W4_STUB_UPDATER,
                                                                                         encoding="utf-8")
                    project.config_path.write_text(json.dumps({"external_dependencies": {}}), encoding="utf-8")
                    env = {"W4_STUB_MODE": mode, "W4_STUB_CALLS": str(calls_file)}
                    with patch.dict("os.environ", env), _Isolated(project, release_package=True), _W4Modules():
                        logs = run(project)
                    calls = json.loads(calls_file.read_text(encoding="utf-8")) if calls_file.is_file() else []
                    results.append((label, project, logs, calls))
        return results

    def test_b2_setup_w4_failed_fetch_keeps_the_install_and_prints_the_hint(self) -> None:
        for label, project, logs, calls in self._stub_case("fail"):
            self.assertEqual(logs.count(W4_HINT_EN), 1, logs)  # the runners already assert exit code 0 / all_done
            self.assertEqual([c[0] for c in calls], ["check"])

    def test_b2_setup_w4_no_updater_in_the_installed_package_prints_the_hint(self) -> None:
        for label, project, logs, calls in self._stub_case("fail", stub=False):
            self.assertEqual(logs.count(W4_HINT_EN), 1, logs)

    def test_b2_setup_w4_runtime_upgrade_or_apply_error_prints_the_hint(self) -> None:
        for mode, expected in (("abi", ["check"]), ("apply_fails", ["check", "apply"])):
            for label, project, logs, calls in self._stub_case(mode):
                self.assertEqual(logs.count(W4_HINT_EN), 1, (mode, logs))
                self.assertEqual([c[0] for c in calls], expected)

    def test_b2_setup_w4_up_to_date_prints_nothing(self) -> None:
        for label, project, logs, calls in self._stub_case("up_to_date"):
            self.assertNotIn(W4_HINT_EN, logs)
            self.assertFalse([line for line in logs if "code update" in line], logs)
            self.assertEqual([c[0] for c in calls], ["check"])

    def test_b2_setup_w4_passes_the_installed_root_and_the_wizard_network_choices(self) -> None:
        network = {"mirror_prefix": "https://ghfast.top/", "proxy_url": "http://127.0.0.1:9",
                   "offline_dir": "/tmp/w4-offline"}
        for label, project, logs, calls in self._stub_case("update", network=network):
            self.assertEqual([c[0] for c in calls], ["check", "apply"])
            for name, kwargs in calls:
                with self.subTest(copy=label, call=name):
                    self.assertEqual(Path(kwargs["module_root"]), project.module_root)
                    self.assertEqual(kwargs["mirror_prefix"], "https://ghfast.top/")
                    self.assertEqual(kwargs["proxy_url"], "http://127.0.0.1:9")
                    self.assertEqual(kwargs["offline_dir"], "/tmp/w4-offline")
            self.assertTrue(calls[1][1]["has_manifest"])
            self.assertEqual(len([line for line in logs if line.startswith("Applied the latest code update")]), 1)
            self.assertNotIn(W4_HINT_EN, logs)
        # Empty choices go as None: the saved settings (and the system proxy) apply, as for the wizard's own downloads.
        for label, project, logs, calls in self._stub_case("up_to_date"):
            kwargs = calls[0][1]
            self.assertEqual((kwargs["mirror_prefix"], kwargs["proxy_url"], kwargs["offline_dir"]), (None, None, None))

    def test_b2_setup_w4_chinese_hint(self) -> None:
        ns = prune._load(self.python_payload)
        stub_updater = SimpleNamespace(
            check_for_code_update=lambda **_kw: {"status": "error", "error": "stubbed-offline"},
            download_and_apply_code_overlay=lambda **_kw: {"status": "error", "error": "stubbed-offline"},
        )
        lines = []
        with patch.dict(sys.modules, {"shared.code_updater": stub_updater}):
            ns["_apply_code_update"]("/nonexistent/w4", "", "", "", lines.append, "zh")
            self.assertEqual(lines, [W4_HINT_ZH])

            def ascii_only(msg):
                msg.encode("cp1252")  # a console that cannot print Chinese
                lines.append(msg)

            lines.clear()
            ns["_apply_code_update"]("/nonexistent/w4", "", "", "", ascii_only, "zh")
            self.assertEqual(lines, [W4_HINT_EN])

    # (d) text / AST check of the three copies
    def _sources(self):
        from test_installer_gui_reinstall_gap import setup_module_source

        yield "python_builder", self.python_payload
        if self.js_payload is None:
            if _on_ci():
                self.fail("node is required on CI for the JS installer copies (node is not in PATH)")
            return
        yield "js_script", self.js_payload
        yield "js_setup_module", setup_module_source()

    def test_b2_setup_w4_three_copies_hold_one_identical_step_with_published_keywords(self) -> None:
        published = ast.parse(_published_text(self, "shared/code_updater.py"))
        params = {}
        for node in published.body:
            if isinstance(node, ast.FunctionDef) and node.name in ("check_for_code_update",
                                                                    "download_and_apply_code_overlay"):
                params[node.name] = {a.arg for a in node.args.args + node.args.kwonlyargs}
        self.assertEqual(set(params), {"check_for_code_update", "download_and_apply_code_overlay"})
        dumps = {}
        for label, source in self._sources():
            with self.subTest(copy=label):
                tree = ast.parse(source)
                steps = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == W4_STEP]
                self.assertEqual(len(steps), 1)
                step = steps[0]
                self.assertTrue(all(ord(ch) < 128 for ch in ast.get_source_segment(source, step)))
                dumps[label] = ast.dump(step)
                calls = [n for n in ast.walk(step) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                         and n.func.id in params]
                self.assertEqual(sorted({c.func.id for c in calls}), sorted(params))
                for call in calls:
                    names = [k.arg for k in call.keywords]
                    self.assertNotIn(None, names, "no **kwargs: every keyword must be checked")
                    self.assertFalse(call.args, "keywords only")
                    self.assertLessEqual(set(names), params[call.func.id], call.func.id)
                imports = [n for n in ast.walk(step) if isinstance(n, ast.ImportFrom)]
                self.assertEqual([n.module for n in imports], ["shared.code_updater"])
                # Order: the step is the last action before success.
                host_name = "_run_phase_c_core" if label == "js_setup_module" else "main"
                host = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == host_name)
                text = ast.get_source_segment(source, host)
                at = text.index(W4_STEP + "(")
                if label == "js_setup_module":
                    self.assertLess(text.index("write_user_config(cfg)"), at)
                    self.assertLess(at, text.index('emit("all_done"'))
                else:
                    self.assertLess(text.index("run_environment_doctor("), at)
                    self.assertLess(at, text.index('_log("Installation completed successfully.")'))
        self.assertEqual(len(set(dumps.values())), 1, "the three copies of the step differ")


if __name__ == "__main__":
    unittest.main()
