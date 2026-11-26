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

import errno
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
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
from test_installer_gui_reinstall_gap import load_setup_module, setup_module_source

# The commit that the published base packages (repack-os-base) are built from: the parent of 5070ca29 "sync repacked
# v0.1.1 checksums and setup zips to website". The tag v0.1.1 is not that code: the packages hold fresh HEAD code.
# The matrix workflow fetches this commit (see test_workflow_fetches_the_release_package_source_commit).
RELEASE_PACKAGE_SOURCE_COMMIT = "2e8f73152e2f218ac87adcad55f1731967a162e1"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "slicer-compat-matrix.yml"
FETCH_STEP_NAME = "Fetch the release-package source commit for the installer tests"


def _on_ci() -> bool:
    return os.environ.get("CI", "").strip().lower() not in ("", "0", "false", "no")


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
    """Process state that the installer code changes: sys.path, sys.modules and the environment.

    Finding PRUNE-ORDER (CI run 45531342001): a daemon thread that an earlier test module leaves behind (the
    StageMonitor sampler of test_cohort_granularity; every 0.2 s it runs ``from shared.hardware_adapt import ...``,
    shared/cohort_profile.py:308) imports the REPO's ``shared`` package whenever ``sys.modules`` has no ``shared`` entry.
    The old harness deleted ``shared`` and then waited (node build, payload exec) until the installer imported it, and
    ``patch.dict(sys.modules)`` clears the whole dict on exit. Both gaps let the thread re-import the repo package, so a
    release_package=True case ran on the repo code (or died with KeyError: 'shared'). Now ``sys.modules['shared']`` is
    never absent: for release_package=True one assignment swaps in the stub package of module_root (a thread that asks
    the stub for a repo-only module gets ModuleNotFoundError), and the exit restores entries one by one."""

    def __init__(self, project: _Project, *, release_package: bool) -> None:
        self.project = project
        self.release_package = release_package

    def __enter__(self):
        self._stack = []
        # b2-setup W4: the installer's last step checks for a code update. A test run never reaches the network: the
        # manifest URL is a closed local port (a test that needs the step gives an offline folder instead).
        env = {"SLICERDSPACE_USER_CONFIG": str(self.project.config_path),
               "SLICERDSPACE_UPDATE_MANIFEST_URL": "http://127.0.0.1:1/update_manifest.json"}
        patches = [
            patch.dict(os.environ, env),
            patch.object(sys, "path", list(sys.path)),
        ]
        for p in patches:
            p.start()
            self._stack.append(p)
        self._modules = dict(sys.modules)
        if self.release_package:
            pkg_dir = self.project.module_root / "shared"
            spec = importlib.util.spec_from_file_location(
                "shared", pkg_dir / "__init__.py", submodule_search_locations=[str(pkg_dir)])
            stub_pkg = importlib.util.module_from_spec(spec)
            sys.modules["shared"] = stub_pkg  # one assignment: 'shared' is never absent, and it is not the repo package
            for name in [n for n in list(sys.modules) if n.startswith("shared.")]:
                sys.modules.pop(name, None)
            sys.modules["shared.environment"] = _stub_environment_module()
            spec.loader.exec_module(stub_pkg)
        else:
            sys.modules["shared.environment"] = _stub_environment_module()
        return self

    def _between_exit_phases(self) -> None:
        """Hook point of __exit__: runs after the saved ``shared.*`` children are back and before the parent entry is set.
        It does nothing. A test replaces it with the sampler's lazy import (finding B1)."""

    def __exit__(self, *exc):
        saved = self._modules
        # Finding B1 (Opus review): children first, parent last. While the stub package is still in sys.modules['shared'],
        # a leaked sampler that runs ``from shared.hardware_adapt import ...`` finds the saved child in sys.modules and
        # imports nothing. With the parent first, the thread saw the repo package without its child, imported a second
        # copy and set it as an attribute of the repo package: getattr(shared, 'hardware_adapt') was no longer
        # sys.modules['shared.hardware_adapt'], so patch('shared.hardware_adapt.X') missed the module that others use.
        # Restore entry by entry (no clear()): a foreign thread never sees sys.modules without a 'shared'.
        for name, module in saved.items():
            if name.startswith("shared.") and sys.modules.get(name) is not module:
                sys.modules[name] = module
        for name in [n for n in list(sys.modules) if n.startswith("shared.") and n not in saved]:
            sys.modules.pop(name, None)
        self._between_exit_phases()
        if "shared" in saved:
            sys.modules["shared"] = saved["shared"]
        for name, module in saved.items():
            if sys.modules.get(name) is not module:
                sys.modules[name] = module
        for name in [n for n in list(sys.modules) if n not in saved]:
            sys.modules.pop(name, None)
        for p in reversed(self._stack):
            p.stop()
        return False


def _run_script_main(ns: dict, project: _Project, *, network: dict | None = None) -> list:
    """Run main() of a generated installer script with the Slicer, download and install steps replaced.

    ``network`` (mirror_prefix, proxy_url, offline_dir) overrides the empty network choices of the wizard."""
    logs: list = []
    slicer_exe = project.root / "Slicer"
    slicer_exe.write_text("", encoding="utf-8")
    archive = project.root / "ext.zip"
    archive.write_bytes(b"PK")
    config = dict(ns["CONFIG"])
    config.update(
        model_assets_dir=str(project.assets),
        selected_packs=[],
        selected_model_packs=[],
        mirror_prefix="",
        proxy_url="",
        offline_dir="",
    )
    config.update(network or {})
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


def _run_setup_module(project: _Project, *, network: dict | None = None, ns: dict | None = None) -> list:
    """Run the graphical Setup module's phase C (the post-download install and config step).

    ``ns`` is a namespace that the caller loaded before it entered _Isolated: load_setup_module() builds the source
    with node and patches sys.modules, a long gap in which a foreign thread can import the repo package (_Isolated)."""
    ns = ns if ns is not None else load_setup_module()
    ns["CONFIG"] = dict(ns["CONFIG"], model_assets_dir=str(project.assets), selected_packs=[])
    events: list = []
    info = {
        "mod_root": str(project.module_root),
        "ext_temps": [],
        "mirror": str((network or {}).get("mirror_prefix") or ""),
        "proxy": str((network or {}).get("proxy_url") or ""),
        "offdir": str((network or {}).get("offline_dir") or ""),
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
        # Without node the JS copies cannot be built. They must not vanish: _run_case fails them on CI and skips them
        # with a message elsewhere (finding P2-B).
        cls.js_payload = _js_payload("linux") if shutil.which("node") is not None else None

    def _runners(self):
        """Yield (label, run(project)); other test modules (test_b2_setup_core_only) use this form."""
        yield "python_builder", lambda project: _run_script_main(_load(self.python_payload), project)
        if self.js_payload is not None:
            yield "js_script", lambda project: _run_script_main(_load(self.js_payload), project)
            yield "js_setup_module", _run_setup_module
        else:
            yield "js_script", None
            yield "js_setup_module", None

    def _factories(self):
        """Yield (label, factory). factory() runs BEFORE _Isolated is entered and returns run(project): the payload exec
        and the node build of the Setup module stay out of the isolated window (finding PRUNE-ORDER)."""
        def script(payload):
            def factory():
                ns = _load(payload)
                return lambda project: _run_script_main(ns, project)
            return factory

        def setup_module():
            ns = load_setup_module()
            return lambda project: _run_setup_module(project, ns=ns)

        yield "python_builder", script(self.python_payload)
        if self.js_payload is not None:
            yield "js_script", script(self.js_payload)
            yield "js_setup_module", setup_module
        else:
            yield "js_script", None
            yield "js_setup_module", None

    def _run_case(self, *, release_package: bool, make_deps, check) -> None:
        """Run every copy in its own temp project; check(project, deps, logs) runs before the temp folder is removed."""
        for label, factory in self._factories():
            with self.subTest(copy=label, release_package=release_package):
                if factory is None:
                    if _on_ci():
                        self.fail("node is required on CI for the JS installer copies (node is not in PATH)")
                    self.skipTest("node is not installed in PATH: the JS installer copy is not tested")
                with tempfile.TemporaryDirectory() as td:
                    project = _Project(Path(td), release_package=release_package)
                    deps = make_deps(project)
                    project.write_config(deps)
                    run = factory()
                    with _Isolated(project, release_package=release_package):
                        logs = run(project)
                        # Finding Opus P2 (prune harness): _Isolated puts the stub package of module_root into sys.modules
                        # itself, so a copy that lost its own sys.path.insert(0, mod_root) would still pass. Check that the
                        # installer put mod_root on sys.path (sys.path is patched here and restored on exit).
                        self.assertIn(
                            str(project.module_root), sys.path,
                            f"the {label} installer did not put mod_root on sys.path (its sys.path.insert is missing)")
                    check(project, deps, logs)

    def _assert_deps(self, project: _Project, expected: dict) -> None:
        """Compare the written dependencies. model_assets_dir is compared as a real path: the installer writes what the
        installed package's configured_model_assets_root() returns. The current package resolves it (macOS
        /var -> /private/var, Windows 8.3 names such as RUNNER~1), the v0.1.1 stand-in does not. The test must not
        depend on which package answers. Every other value is compared as written."""
        def norm(deps: dict) -> dict:
            out = dict(deps)
            if "model_assets_dir" in out:
                out["model_assets_dir"] = os.path.realpath(out["model_assets_dir"])
            return out

        self.assertEqual(norm(project.read_deps()), norm(expected))

    def test_prune_release_package_without_the_prune_function_keeps_dependencies(self) -> None:
        # The release package has no prune function: dependencies stay as they are, and nothing is logged about pruning.
        def make(project):
            return {"mrtrix_bin_dir": str(project.root / "mrtrix" / "bin"), "postprocess_runtime": str(project.stale_dependency())}

        def check(project, deps, logs):
            self._assert_deps(project, {**deps, "model_assets_dir": str(project.assets)})
            self.assertFalse([line for line in logs if "stale bundled dependency" in line], logs)

        self._run_case(release_package=True, make_deps=make, check=check)

    def test_prune_release_package_still_sets_the_model_assets_dir(self) -> None:
        def check(project, deps, logs):
            self._assert_deps(project, {"model_assets_dir": str(project.assets)})

        self._run_case(release_package=True, make_deps=lambda project: {"wma_atlas_dir": "x"}, check=check)

    def test_prune_current_package_still_sets_the_model_assets_dir(self) -> None:
        # The current package answers (it resolves the path): the same write, compared as a real path.
        def check(project, deps, logs):
            self._assert_deps(project, {"model_assets_dir": str(project.assets)})

        self._run_case(release_package=False, make_deps=lambda project: {"wma_atlas_dir": "x"}, check=check)

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
        # Evidence that the stand-in is faithful: the commit that the published base packages are built from has no
        # prune function. On CI a missing git or commit is a failure (the workflow fetches the commit); elsewhere a skip.
        sha = RELEASE_PACKAGE_SOURCE_COMMIT
        git = shutil.which("git")
        out = None
        if git is not None:
            out = subprocess.run([git, "show", f"{sha}:shared/user_config.py"], cwd=REPO_ROOT, capture_output=True,
                                 text=True, encoding="utf-8")
        if out is None or out.returncode != 0:
            why = "git is not installed" if out is None else f"commit {sha} is not in this checkout"
            if _on_ci():
                self.fail(f"{why}; the workflow step '{FETCH_STEP_NAME}' in slicer-compat-matrix.yml must fetch it")
            self.skipTest(f"{why}; run: git fetch --no-tags --depth=1 origin {sha}")
        self.assertNotIn("prune_stale_bundled_dependencies", out.stdout)
        for name in ("load_user_config", "write_user_config"):
            self.assertIn(f"def {name}(", out.stdout)


def _run_inner(test_name: str, *, ci: bool, which_without_node: bool = False, fake_git_missing_commit: bool = False):
    """Run one PruneFallbackTests test with unittest in a simulated environment; return its TestResult.

    The inner run never includes PruneGuardTests, so it cannot recurse into the caller.
    """
    real_which = shutil.which
    real_run = subprocess.run

    def fake_which(cmd, *args, **kwargs):
        return None if cmd == "node" else real_which(cmd, *args, **kwargs)

    def fake_run(cmd, *args, **kwargs):
        if fake_git_missing_commit and isinstance(cmd, (list, tuple)) and len(cmd) > 1 and cmd[1] == "show":
            return subprocess.CompletedProcess(cmd, 128, "", "fatal: invalid object name")
        return real_run(cmd, *args, **kwargs)

    with patch.dict(os.environ):
        if ci:
            os.environ["CI"] = "true"
        else:
            os.environ.pop("CI", None)
        with patch.object(shutil, "which", fake_which if which_without_node else real_which), \
                patch.object(subprocess, "run", fake_run):
            suite = unittest.TestSuite([PruneFallbackTests(test_name)])
            return unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)


class PruneGuardTests(unittest.TestCase):
    """Fail-before checks for the two findings: the guarded tests must fail on CI, not skip or pass silently."""

    def test_prune_p2a_missing_release_commit_fails_on_ci(self) -> None:
        # Finding P2-A: the commit check was skipped on CI (a depth-1 checkout has no tags). It must fail there.
        result = _run_inner("test_prune_stub_matches_the_release_package_shape", ci=True, fake_git_missing_commit=True)
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(len(result.failures) + len(result.errors), 1, (result.skipped, result.failures))
        self.assertEqual(result.skipped, [])
        self.assertIn(FETCH_STEP_NAME, result.failures[0][1])

    def test_prune_p2a_missing_release_commit_skips_off_ci(self) -> None:
        result = _run_inner("test_prune_stub_matches_the_release_package_shape", ci=False, fake_git_missing_commit=True)
        self.assertEqual(len(result.skipped), 1, result.failures)
        self.assertEqual(result.failures + result.errors, [])

    def test_prune_p2b_missing_node_fails_the_js_copies_on_ci(self) -> None:
        # Finding P2-B: without node the two JS copies were dropped silently. On CI both must fail with a message.
        name = "test_prune_release_package_still_sets_the_model_assets_dir"
        result = _run_inner(name, ci=True, which_without_node=True)
        failed = [text for _, text in result.failures + result.errors]
        self.assertEqual(len(failed), 2, (result.skipped, failed))
        for text in failed:
            self.assertIn("node is required on CI for the JS installer copies", text)

    def test_prune_p2b_missing_node_skips_the_js_copies_off_ci_and_runs_python(self) -> None:
        name = "test_prune_release_package_still_sets_the_model_assets_dir"
        result = _run_inner(name, ci=False, which_without_node=True)
        self.assertEqual(result.failures + result.errors, [])
        self.assertEqual(len(result.skipped), 2, "both JS copies must be reported as skipped, not dropped")
        for _, reason in result.skipped:
            self.assertIn("node is not installed", reason)


class PruneSysPathTests(unittest.TestCase):
    """Finding Opus P2 (prune harness): _run_case must fail a copy whose own sys.path.insert(0, mod_root) is gone.

    Each copy gets a variant with that insert replaced by ``pass``. A PruneFallbackTests subclass that yields only the
    variant must fail on sys.path, not on another error. Before the assertion in _run_case, the variants passed."""

    PY_INSERT = "sys.path.insert(0, str(module_root))"
    JS_INSERT = "sys.path.insert(0, str(mod_root))"
    CASES = ("test_prune_release_package_still_sets_the_model_assets_dir",
             "test_prune_current_package_still_sets_the_model_assets_dir")

    @classmethod
    def setUpClass(cls) -> None:
        cls.python_payload = _python_payload("linux")
        node = shutil.which("node") is not None
        cls.js_payload = _js_payload("linux") if node else None
        cls.setup_source = setup_module_source() if node else None

    @staticmethod
    def _without(text: str, insert: str, what: str) -> str:
        # Not vacuous: the insert text must exist, or the variant would be the unchanged copy.
        assert text.count(insert) >= 1, f"{what} no longer holds {insert!r}: update PruneSysPathTests"
        return text.replace(insert, "pass")

    def _variants(self):
        """Yield (label, factory) for each copy without its sys.path insert; factory is None when node is missing."""
        def script(payload):
            def factory():
                ns = _load(payload)
                return lambda project: _run_script_main(ns, project)
            return factory

        def setup_module(source):
            def factory():
                ns = load_setup_module(source=source)
                return lambda project: _run_setup_module(project, ns=ns)
            return factory

        yield "python_builder", script(self._without(self.python_payload, self.PY_INSERT, "the Python builder payload"))
        if self.js_payload is None:
            yield "js_script", None
            yield "js_setup_module", None
            return
        yield "js_script", script(self._without(self.js_payload, self.JS_INSERT, "the JS script payload"))
        yield "js_setup_module", setup_module(self._without(self.setup_source, self.JS_INSERT, "the JS Setup module"))

    def _run_variant(self, label: str, factory, test_name: str) -> unittest.TestResult:
        class OneCopy(PruneFallbackTests):
            @classmethod
            def setUpClass(cls) -> None:  # the payloads are not used: _factories yields the variant
                pass

            def _factories(self):
                yield label, factory

        result = unittest.TestResult()
        OneCopy(test_name).run(result)
        return result

    def test_prune_p2_installer_without_the_sys_path_insert_fails_the_case(self) -> None:
        for label, factory in self._variants():
            with self.subTest(copy=label):
                if factory is None:
                    if _on_ci():
                        self.fail("node is required on CI for the JS installer copies (node is not in PATH)")
                    self.skipTest("node is not installed in PATH: the JS installer copy is not tested")
                for test_name in self.CASES:
                    result = self._run_variant(label, factory, test_name)
                    self.assertEqual(result.testsRun, 1, test_name)
                    self.assertEqual(result.errors, [], f"{label}/{test_name}: an error, not a failed sys.path check")
                    self.assertEqual(len(result.failures), 1, f"{label}/{test_name}: the variant passed: {result.skipped}")
                    self.assertIn("did not put mod_root on sys.path", result.failures[0][1], f"{label}/{test_name}")


def _symlink_unsupported_reason() -> str | None:
    """None when this process can create a directory symlink; else why not (Windows without the privilege).

    Only the os.symlink call is inside the try: any other error is a real error, not a skip.
    """
    with tempfile.TemporaryDirectory() as root:
        real_dir = Path(root) / "real"
        real_dir.mkdir()
        try:
            os.symlink(real_dir, Path(root) / "link", target_is_directory=True)
        except (OSError, NotImplementedError, AttributeError) as exc:
            return f"{type(exc).__name__}: {exc}"
    return None


def _run_prune_cases_under_ci_conditions(*, leaked_sampler: bool, symlinked_tmp: bool, exit_tick: bool = False,
                                         inside_run=None, leak_join_seconds: float = 3.0):
    """Run the behaviour tests of PruneFallbackTests (not PruneGuardTests or this class: no recursion) with unittest.

    leaked_sampler: another thread runs the lazy import of the StageMonitor sampler that test_cohort_granularity leaves
    behind (shared/cohort_profile.py:308, ``from shared.hardware_adapt import ...``) at two points that the old harness
    left open: right after the harness removed ``shared`` from sys.modules, and during the node build of the Setup
    module. The real thread does this every 0.2 s at a random time; here it happens at fixed points, in the same order.
    symlinked_tmp: temp folders are reached through a symlink (macOS /var -> /private/var is the same shape).
    exit_tick: finding B1. The sampler also ticks inside _Isolated.__exit__, between the restore of the saved shared.*
    children and the restore of the parent (_between_exit_phases; this point exists in both orders). shared.hardware_adapt is
    imported before the first state(), so the saved sys.modules holds it.
    inside_run: a callable that runs on the calling thread at the same point as exit_tick (every _Isolated.__exit__).
    Finding Opus P2 (prune harness) thread count: the state holds ``leaked_threads``, the names of the threads that this run
    started from the calling thread and that are still alive after a join of at most leak_join_seconds in total. It is
    [] before the run. threading.active_count() is not compared: a foreign thread (a sampler of an earlier test module)
    that ends or starts during the run changed the count and failed the check by chance.
    Returns (TestResult, process state before, process state after)."""
    from shared import cohort_profile  # the repo module; its function object keeps working after the harness drops it

    sampler = cohort_profile._default_system_memory_snapshot
    real_run = subprocess.run
    real_stub = globals()["_stub_environment_module"]

    def tick() -> None:
        thread = threading.Thread(target=sampler, name="leaked-stage-monitor-sampler")
        thread.start()
        thread.join()

    def run_with_tick(*args, **kwargs):
        # Only the node build (the sampler itself runs subprocesses: ticking on those would recurse).
        cmd = args[0] if args else kwargs.get("args")
        if (threading.current_thread() is threading.main_thread() and isinstance(cmd, (list, tuple)) and cmd
                and Path(str(cmd[0])).stem.lower() == "node"):
            tick()
        return real_run(*args, **kwargs)

    def stub_with_tick():
        tick()
        return real_stub()

    tracked: set = set()
    caller = threading.current_thread()
    started: list = []  # Thread objects that start() was called on from the calling thread during the run
    real_start = threading.Thread.start

    def recording_start(self, *args, **kwargs):
        result = real_start(self, *args, **kwargs)
        if threading.current_thread() is caller:  # a thread started by a foreign thread is not ours
            started.append(self)
        return result

    def leaked_threads():
        deadline = time.monotonic() + leak_join_seconds
        for thread in list(started):
            thread.join(max(0.0, deadline - time.monotonic()))
        return sorted(t.name for t in started if t.is_alive())

    def state():
        # Finding B1: the identity of every shared* entry, and whether each child is still the attribute of its parent.
        # Only the keys of the first call: a module that a test imports for the first time (shared.installer_script_builder
        # in setUpClass) is new, not a leak.
        shared = sys.modules.get("shared")
        keys = [k for k in list(sys.modules) if k == "shared" or k.startswith("shared.")]
        if tracked:
            keys = [k for k in keys if k in tracked]
        ids = {k: id(sys.modules[k]) for k in keys}
        attrs = {k: getattr(shared, k.split(".", 1)[1], None) is sys.modules[k]
                 for k in keys if k.count(".") == 1 and k.split(".", 1)[1] in vars(shared)}
        if not tracked:
            tracked.update(keys)
        return {"shared": shared, "path": list(sys.path), "environ": dict(os.environ),
                "tempdir": tempfile.tempdir, "leaked_threads": leaked_threads(), "ids": ids, "attrs": attrs}

    names = [n for n in unittest.defaultTestLoader.getTestCaseNames(PruneFallbackTests)
             if n != "test_prune_stub_matches_the_release_package_shape"]
    tempfile.gettempdir()  # fill the cache first: the first call sets tempfile.tempdir, which is not a leak
    if exit_tick:
        import shared.hardware_adapt  # noqa: F401  (a saved child of the shared package that the sampler imports lazily)
    before = state()
    with tempfile.TemporaryDirectory() as real_root:
        patches = []
        if symlinked_tmp:
            link = Path(real_root) / "tmp-link"
            real_dir = Path(real_root) / "real"
            real_dir.mkdir()
            os.symlink(real_dir, link, target_is_directory=True)
            patches.append(patch.object(tempfile, "tempdir", str(link)))
        if leaked_sampler:
            patches.append(patch.object(subprocess, "run", run_with_tick))
            patches.append(patch.dict(globals(), {"_stub_environment_module": stub_with_tick}))
        if exit_tick or inside_run is not None:
            def exit_hook(self) -> None:
                if exit_tick:
                    tick()
                if inside_run is not None:
                    inside_run()
            patches.append(patch.object(_Isolated, "_between_exit_phases", exit_hook))
        patches.append(patch.object(threading.Thread, "start", recording_start))
        for p in patches:
            p.start()
        try:
            suite = unittest.TestSuite([PruneFallbackTests(n) for n in names])
            result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
        finally:
            for p in reversed(patches):
                p.stop()
    return result, before, state()


class PruneOrderIndependenceTests(unittest.TestCase):
    """Finding PRUNE-ORDER: the prune tests failed on CI only (KeyError: 'shared', a repo package in a release_package
    case, a /private/var path) and passed alone. Cause: a sampler thread left by test_cohort_granularity imports the repo's
    ``shared`` whenever sys.modules has no such entry; see _Isolated."""

    def _assert_clean(self, result) -> None:
        problems = [text for _, text in result.failures + result.errors]
        self.assertEqual(problems, [], "\n".join(problems)[:3000])
        self.assertGreater(result.testsRun, 0)

    def test_prune_order_leaked_sampler_cannot_reach_the_release_package_cases(self) -> None:
        result, before, after = _run_prune_cases_under_ci_conditions(leaked_sampler=True, symlinked_tmp=False)
        self._assert_clean(result)
        self.assertEqual(before, after)

    def test_prune_order_symlinked_temp_root_with_a_leaked_sampler(self) -> None:
        # Both CI conditions at once. The path compare works on real paths (macOS /private/var, Windows RUNNER~1).
        reason = _symlink_unsupported_reason()  # Windows without the symlink privilege
        if reason:
            self.skipTest(f"cannot create a symlink here: {reason}")
        result, before, after = _run_prune_cases_under_ci_conditions(leaked_sampler=True, symlinked_tmp=True)
        self._assert_clean(result)
        self.assertEqual(before, after)

    def test_prune_order_current_package_path_check_survives_a_symlinked_temp_root(self) -> None:
        # No sampler: only the path case. The current package resolves model_assets_dir; the test compares real paths.
        reason = _symlink_unsupported_reason()
        if reason:
            self.skipTest(f"cannot create a symlink here: {reason}")
        result, before, after = _run_prune_cases_under_ci_conditions(leaked_sampler=False, symlinked_tmp=True)
        self._assert_clean(result)
        self.assertEqual(before, after)

    def test_prune_order_harness_leaves_the_process_state_as_it_found_it(self) -> None:
        # No leaked patches: sys.modules['shared'], sys.path, os.environ, tempfile.tempdir and no thread of this run left alive.
        result, before, after = _run_prune_cases_under_ci_conditions(leaked_sampler=False, symlinked_tmp=False)
        self._assert_clean(result)
        self.assertIs(before["shared"], after["shared"])
        self.assertEqual(before, after)
        self.assertEqual(after["leaked_threads"], [])

    def test_prune_order_b1_sampler_between_the_restores_leaves_no_second_hardware_adapt(self) -> None:
        # Finding B1 (Opus review): a sampler tick between the child and parent restores of _Isolated.__exit__.
        import shared.hardware_adapt
        saved_module = sys.modules["shared.hardware_adapt"]
        result, before, after = _run_prune_cases_under_ci_conditions(
            leaked_sampler=False, symlinked_tmp=False, exit_tick=True)
        self._assert_clean(result)
        self.assertEqual(before["ids"], after["ids"], "a shared* sys.modules entry is a different object after the run")
        self.assertEqual(before, after)
        self.assertIs(sys.modules["shared.hardware_adapt"], saved_module)
        self.assertIs(getattr(sys.modules["shared"], "hardware_adapt"), sys.modules["shared.hardware_adapt"])
        self.assertTrue(before["attrs"]["shared.hardware_adapt"], "the pre-import must make the child an attribute")
        for key, was_attr in before["attrs"].items():
            if was_attr:
                self.assertTrue(after["attrs"].get(key), f"{key} is not the attribute of the shared package any more")

    def _daemon(self, target, name: str) -> threading.Thread:
        thread = threading.Thread(target=target, name=name, daemon=True)
        thread.start()
        return thread

    def _stoppable(self, name: str) -> tuple:
        """A daemon thread that waits on an Event. addCleanup sets the Event and joins the thread."""
        stop = threading.Event()
        thread = self._daemon(stop.wait, name)

        def cleanup() -> None:
            stop.set()
            thread.join(5)
        self.addCleanup(cleanup)
        return stop, thread

    def test_prune_order_thread_check_ignores_a_foreign_thread_that_ends_during_the_run(self) -> None:
        # Opus P2 (prune harness) thread count: an earlier module's sampler can end while the inner run goes on.
        # threading.active_count() dropped by one and before != after.
        release, foreign = self._stoppable("foreign-ends-during-run")

        def end_foreign_thread() -> None:
            release.set()
            foreign.join(5)

        result, before, after = _run_prune_cases_under_ci_conditions(
            leaked_sampler=False, symlinked_tmp=False, inside_run=end_foreign_thread)
        self.assertFalse(foreign.is_alive(), "the hook did not end the foreign thread")
        self._assert_clean(result)
        self.assertEqual(before, after)
        self.assertEqual(after["leaked_threads"], [])

    def test_prune_order_thread_check_ignores_a_thread_that_a_foreign_thread_starts_during_the_run(self) -> None:
        # Opus P2 (prune harness) thread count: a background thread starts another thread during the run (not the calling
        # thread) and it stays alive until the cleanup. The old count rose by one.
        go = threading.Event()
        child_started = threading.Event()
        stop = threading.Event()
        children: list = []

        def foreign_parent() -> None:
            go.wait()
            child = threading.Thread(target=stop.wait, name="child-of-foreign-thread", daemon=True)
            child.start()
            children.append(child)
            child_started.set()
            stop.wait()  # the parent stays alive too: only the child is new, so the old count rose by one

        parent = self._daemon(foreign_parent, "foreign-parent")

        def cleanup() -> None:
            go.set()
            stop.set()
            parent.join(5)
            for child in children:
                child.join(5)
        self.addCleanup(cleanup)

        def start_child_in_foreign_thread() -> None:
            go.set()
            self.assertTrue(child_started.wait(5), "the foreign thread did not start its child")

        result, before, after = _run_prune_cases_under_ci_conditions(
            leaked_sampler=False, symlinked_tmp=False, inside_run=start_child_in_foreign_thread)
        self.assertTrue(children and children[0].is_alive(), "the child must stay alive during the check")
        self._assert_clean(result)
        self.assertEqual(before, after)
        self.assertEqual(after["leaked_threads"], [])

    def test_prune_order_thread_check_still_reports_a_thread_that_the_run_started_and_left_alive(self) -> None:
        # Opus P2 (prune harness) thread count: the new check must catch what the count caught. A thread that the calling
        # thread starts inside the run and leaves alive is a leak.
        stop = threading.Event()
        threads: list = []

        def start_and_leave_a_thread() -> None:
            if not threads:
                thread = threading.Thread(target=stop.wait, name="leaked-by-the-run", daemon=True)
                thread.start()
                threads.append(thread)

        def cleanup() -> None:
            stop.set()
            for thread in threads:
                thread.join(5)
        self.addCleanup(cleanup)

        result, before, after = _run_prune_cases_under_ci_conditions(
            leaked_sampler=False, symlinked_tmp=False, inside_run=start_and_leave_a_thread, leak_join_seconds=0.2)
        self._assert_clean(result)
        self.assertEqual(before["leaked_threads"], [])
        self.assertEqual(after["leaked_threads"], ["leaked-by-the-run"])
        self.assertNotEqual(before, after)


class PruneSymlinkPrecheckTests(unittest.TestCase):
    """Finding "Opus P2 (prune harness) symlink": the two symlink tests wrapped the whole run in ``except OSError`` and
    skipped, so a Windows cleanup PermissionError looked like "cannot create a symlink". Only os.symlink is pre-checked
    now. These tests fake both the pre-check and the run, so they run (not skip) on every platform."""

    _SYMLINK_TESTS = (
        "test_prune_order_symlinked_temp_root_with_a_leaked_sampler",
        "test_prune_order_current_package_path_check_survives_a_symlinked_temp_root",
    )

    @staticmethod
    def _run_inner(name: str, reason, fake_run):
        module = sys.modules[PruneOrderIndependenceTests.__module__]
        result = unittest.TestResult()
        with patch.object(module, "_symlink_unsupported_reason", lambda: reason, create=True), \
                patch.object(module, "_run_prune_cases_under_ci_conditions", fake_run):
            PruneOrderIndependenceTests(name).run(result)
        return result

    def test_opus_p2_prune_harness_symlink_a_run_oserror_is_an_error_not_a_skip(self) -> None:
        def cleanup_fails(**_kwargs):
            raise PermissionError(13, "cleanup failed", "x")

        for name in self._SYMLINK_TESTS:
            with self.subTest(name):
                result = self._run_inner(name, None, cleanup_fails)
                self.assertEqual(result.skipped, [], f"{name}: skipped: {result.skipped}")
                self.assertEqual(len(result.errors), 1, f"{name}: errors={result.errors} failures={result.failures}")
                self.assertIn("PermissionError", result.errors[0][1])

    def test_opus_p2_prune_harness_symlink_an_unsupported_symlink_skips_without_running(self) -> None:
        calls = []

        def must_not_run(**kwargs):
            calls.append(kwargs)
            raise AssertionError("the run must not start when symlinks are unsupported")

        for name in self._SYMLINK_TESTS:
            with self.subTest(name):
                result = self._run_inner(name, "no privilege", must_not_run)
                self.assertEqual(calls, [])
                self.assertEqual(result.errors + result.failures, [])
                self.assertEqual(len(result.skipped), 1, f"{name}: {result.skipped}")
                self.assertIn("no privilege", result.skipped[0][1])

    def test_opus_p2_prune_harness_symlink_precheck_reports_a_missing_privilege(self) -> None:
        err = OSError(1314, "A required privilege is not held by the client")
        with patch.object(os, "symlink", side_effect=err):
            reason = _symlink_unsupported_reason()
        self.assertIsNotNone(reason)
        self.assertIn("privilege", reason)

    def test_opus_p2_prune_harness_symlink_precheck_reports_an_unsupported_filesystem(self) -> None:
        # Codex P1: an exFAT-like temp folder (EOPNOTSUPP) is a skip reason on any platform, not only on Windows.
        err = OSError(errno.EOPNOTSUPP, "Operation not supported")
        with patch.object(os, "symlink", side_effect=err):
            reason = _symlink_unsupported_reason()
        self.assertIsNotNone(reason)
        self.assertIn("Operation not supported", reason)

    def test_opus_p2_prune_harness_symlink_precheck_is_none_where_symlinks_work(self) -> None:
        # Codex P1: success path must not need host symlink support, so os.symlink is faked to succeed.
        calls: list[tuple[tuple, dict]] = []

        def fake_symlink(*args, **kwargs) -> None:
            calls.append((args, kwargs))

        with patch.object(os, "symlink", fake_symlink):
            reason = _symlink_unsupported_reason()
        self.assertIsNone(reason)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][1].get("target_is_directory"), True)


class PruneWorkflowSyncTests(unittest.TestCase):
    def test_workflow_fetches_the_release_package_source_commit(self) -> None:
        # Every job that checks out the repo must fetch exactly RELEASE_PACKAGE_SOURCE_COMMIT, as the test constant says.
        text = WORKFLOW_PATH.read_text(encoding="utf-8")
        jobs = len(re.findall(r"(?m)^\s*- uses: actions/checkout@", text))
        self.assertGreaterEqual(jobs, 2)
        shas = re.findall(r"(?m)^\s*run: git fetch --no-tags --depth=1 origin (\S+)\s*$", text)
        self.assertEqual(shas, [RELEASE_PACKAGE_SOURCE_COMMIT] * jobs)
        self.assertEqual(text.count(f"- name: {FETCH_STEP_NAME}"), jobs)


if __name__ == "__main__":
    unittest.main()
