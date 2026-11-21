"""Finding PRUNE: the website installer must work with a published release package that has no prune step.

After the extension is installed, the installer imports ``shared.user_config`` from the installed package. Commit 801b4ba7
added ``prune_stale_bundled_dependencies`` to that import. The published v0.1.1 package has no such function, so every
full install failed with "cannot import name 'prune_stale_bundled_dependencies'" (CI run 44885046381).

The fix: when the installed package has no prune function, the installer keeps ``external_dependencies`` as they are (the
behaviour before 801b4ba7) and logs nothing about pruning. When it has the function, the installer prunes as before.

Each test runs the real generated post-install config code of three copies: the Python builder script, the website JS
installer script and the graphical Setup module (SETUP_MODULE_PY). The installed package is a temp tree.
"""

from __future__ import annotations

import json
import os
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
TESTS_ROOT = Path(__file__).resolve().parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from test_b4_w4_platform import _js_payload, _load, _python_payload
from test_installer_gui_reinstall_gap import load_setup_module

# The shape of shared/user_config.py in the v0.1.1 release package: load and write only, no prune function.
_V011_USER_CONFIG = '''"""Stand-in for shared/user_config.py of the v0.1.1 release package (no prune_stale_bundled_dependencies)."""
import json
import os
from pathlib import Path


def _path():
    return Path(os.environ["SLICERDSPACE_USER_CONFIG"])


def load_user_config(path=None, *, missing_ok=True):
    p = _path()
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def write_user_config(payload, path=None):
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p
'''

_STUB_NETWORK_CONFIG = '''def save_network_settings(**kwargs):
    return {}


def extract_modular_pack(*args, **kwargs):
    return []
'''

_STUB_MODEL_ASSETS = '''from pathlib import Path


def configured_model_assets_root(explicit=None):
    return Path(explicit)
'''


def _stub_environment_module() -> types.ModuleType:
    mod = types.ModuleType("shared.environment")
    mod.EnvironmentDoctorConfig = lambda **kwargs: kwargs

    def _doctor(cfg):
        raise RuntimeError("environment doctor is not part of this test")

    mod.run_environment_doctor = _doctor
    return mod


class _Project:
    """A temp project: an installed module root, a user config file and a model assets folder."""

    def __init__(self, root: Path, *, release_package: bool) -> None:
        self.root = root
        # module_root.parents[2] is the extension root, as in a Slicer extension install.
        self.ext_root = root / "Extensions-34045" / "SlicerDSpace"
        self.module_root = self.ext_root / "lib" / "Slicer-5.10" / "qt-scripted-modules"
        self.module_root.mkdir(parents=True)
        assert self.module_root.parents[2] == self.ext_root
        self.config_path = root / "user_config.json"
        self.assets = root / "assets"
        if release_package:
            pkg = self.module_root / "shared"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("", encoding="utf-8")
            (pkg / "user_config.py").write_text(_V011_USER_CONFIG, encoding="utf-8")
            (pkg / "network_config.py").write_text(_STUB_NETWORK_CONFIG, encoding="utf-8")
            (pkg / "model_assets.py").write_text(_STUB_MODEL_ASSETS, encoding="utf-8")

    def stale_dependency(self) -> Path:
        """A bundled path of another Slicer extension folder that does not exist any more."""
        return self.root / "Extensions-32448" / "SlicerDDSurfer" / "bundled-runtimes" / "missing"

    def write_config(self, deps: dict) -> None:
        self.config_path.write_text(json.dumps({"external_dependencies": deps}), encoding="utf-8")

    def read_deps(self) -> dict:
        return json.loads(self.config_path.read_text(encoding="utf-8"))["external_dependencies"]


class _Isolated:
    """Process state that the installer code changes: sys.path, sys.modules and the environment."""

    def __init__(self, project: _Project, *, release_package: bool) -> None:
        self.project = project
        self.release_package = release_package

    def __enter__(self):
        self._stack = []
        env = {"SLICERDSPACE_USER_CONFIG": str(self.project.config_path)}
        patches = [
            patch.dict(os.environ, env),
            patch.object(sys, "path", list(sys.path)),
            patch.dict(sys.modules),
        ]
        for p in patches:
            p.start()
            self._stack.append(p)
        if self.release_package:
            for name in [n for n in sys.modules if n == "shared" or n.startswith("shared.")]:
                del sys.modules[name]
        sys.modules["shared.environment"] = _stub_environment_module()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._stack):
            p.stop()
        return False


def _run_script_main(ns: dict, project: _Project) -> list:
    """Run main() of a generated installer script with the Slicer, download and install steps replaced."""
    logs: list = []
    slicer_exe = project.root / "Slicer"
    slicer_exe.write_text("", encoding="utf-8")
    archive = project.root / "ext.zip"
    archive.write_bytes(b"PK")
    config = dict(ns["CONFIG"])
    config.update(model_assets_dir=str(project.assets), selected_packs=[], mirror_prefix="", proxy_url="", offline_dir="")
    ns["CONFIG"] = config
    ns["_log"] = logs.append
    ns["_pick_supported_slicer"] = lambda exe, explicit: exe
    ns["_warn_linux_glibc_floor"] = lambda *a, **k: None
    ns["_is_intel_mac"] = lambda: False
    ns["_pick_cache_dir"] = lambda need=0: project.root
    ns["_acquire_spec"] = lambda spec, cache_dir, *a, **k: (archive, [])
    ns["_install_extension_in_slicer" if "_install_extension_in_slicer" in ns else "_install_extension"] = lambda *a, **k: {}
    ns["_cleanup_temp_files"] = lambda paths: None
    ns["_find_module_root"] = lambda exe, receipt: project.module_root
    ns["_sync_iit_template"] = lambda *a, **k: None
    with patch.dict(os.environ, {"SLICERDSPACE_RESOLVED_SLICER": str(slicer_exe)}):
        rc = ns["main"]()
    assert rc == 0, rc
    return logs


def _run_setup_module(project: _Project) -> list:
    """Run the graphical Setup module's phase C (the post-download install and config step)."""
    ns = load_setup_module()
    ns["CONFIG"] = dict(ns["CONFIG"], model_assets_dir=str(project.assets), selected_packs=[])
    events: list = []
    info = {
        "mod_root": str(project.module_root),
        "ext_temps": [],
        "mirror": "",
        "proxy": "",
        "offdir": "",
        "iit_file": "",
        "pack_archives": [],
        "asset_temps": [],
        "sel_packs": [],
    }
    ns["_run_phase_c_core"](info, lambda kind, payload: events.append((kind, payload)))
    errors = [payload for kind, payload in events if kind == "error"]
    if errors:
        raise AssertionError(errors[0])
    assert events and events[-1][0] == "all_done", events[-3:]
    return [str(payload) for kind, payload in events if kind == "log"]


class PruneFallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.python_payload = _python_payload("linux")
        try:
            cls.js_payload = _js_payload("linux")
        except unittest.SkipTest:
            cls.js_payload = None

    def _runners(self):
        yield "python_builder", lambda project: _run_script_main(_load(self.python_payload), project)
        if self.js_payload is not None:
            yield "js_script", lambda project: _run_script_main(_load(self.js_payload), project)
            yield "js_setup_module", _run_setup_module

    def _run_case(self, *, release_package: bool, make_deps, check) -> None:
        """Run every copy in its own temp project; check(project, deps, logs) runs before the temp folder is removed."""
        for label, run in self._runners():
            with self.subTest(copy=label, release_package=release_package):
                with tempfile.TemporaryDirectory() as td:
                    project = _Project(Path(td), release_package=release_package)
                    deps = make_deps(project)
                    project.write_config(deps)
                    with _Isolated(project, release_package=release_package):
                        logs = run(project)
                    check(project, deps, logs)

    def test_prune_release_package_without_the_prune_function_keeps_dependencies(self) -> None:
        # The release package has no prune function: dependencies stay as they are, and nothing is logged about pruning.
        def make(project):
            return {"mrtrix_bin_dir": str(project.root / "mrtrix" / "bin"), "postprocess_runtime": str(project.stale_dependency())}

        def check(project, deps, logs):
            self.assertEqual(project.read_deps(), {**deps, "model_assets_dir": str(project.assets)})
            self.assertFalse([line for line in logs if "stale bundled dependency" in line], logs)

        self._run_case(release_package=True, make_deps=make, check=check)

    def test_prune_release_package_still_sets_the_model_assets_dir(self) -> None:
        def check(project, deps, logs):
            self.assertEqual(project.read_deps(), {"model_assets_dir": str(project.assets)})

        self._run_case(release_package=True, make_deps=lambda project: {"wma_atlas_dir": "x"}, check=check)

    def test_prune_current_package_still_prunes_a_stale_bundled_path(self) -> None:
        def make(project):
            keep = project.root / "standalone" / "bin"
            keep.mkdir(parents=True)
            return {"dki_matlab_executable": str(keep / "matlab"), "postprocess_runtime": str(project.stale_dependency())}

        def check(project, deps, logs):
            self.assertNotIn("postprocess_runtime", project.read_deps())
            self.assertEqual(project.read_deps().get("dki_matlab_executable"), deps["dki_matlab_executable"])
            self.assertTrue(
                [line for line in logs if "Cleared stale bundled dependency override(s): postprocess_runtime" in line], logs
            )

        self._run_case(release_package=False, make_deps=make, check=check)

    def test_prune_stub_matches_the_release_package_shape(self) -> None:
        # Evidence that the stand-in is faithful: the v0.1.1 tag (when this checkout has it) has no prune function.
        git = shutil.which("git")
        if git is None:
            self.skipTest("git is not installed")
        out = subprocess.run([git, "show", "v0.1.1:shared/user_config.py"], cwd=REPO_ROOT, capture_output=True, text=True,
                             encoding="utf-8")
        if out.returncode != 0:
            self.skipTest("tag v0.1.1 is not in this checkout")
        self.assertNotIn("prune_stale_bundled_dependencies", out.stdout)
        for name in ("load_user_config", "write_user_config"):
            self.assertIn(f"def {name}(", out.stdout)


if __name__ == "__main__":
    unittest.main()
