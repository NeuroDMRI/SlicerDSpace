#!/usr/bin/env python3
"""Repository structure smoke tests."""

from __future__ import annotations

import unittest
import subprocess
import ast
import tokenize
import sys
from importlib.util import resolve_name
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_PACKAGE_NAMES = (
    "sharedpipeline",
    "ddsurferpipeline",
    "ddparcpipeline",
    "ddsegpipeline",
    "microstructurepipeline",
)
SCAN_ROOTS = (
    "SlicerDSpace.py",
    "Slicer" + "DDSurfer.py",  # Source-only legacy import compatibility.
    "shared",
    "anatomy",
    "microstructure",
    "tractography",
    "connectome",
    "workflow_catalog.py",
    "shared/workflow_catalog.py",
)
ALLOWED_ROOT_FILES = {
    ".gitattributes",
    ".gitignore",
    ".zenodo.json",
    "CITATION.cff",
    "CMakeLists.txt",
    "LICENSE",
    "LICENSE.txt",
    "Makefile",
    "README.md",
    "README.zh-CN.md",
    "SlicerDSpace.py",
    "Slicer" + "DDSurfer.py",  # Source-only legacy import compatibility.
    "SlicerDSpace.s4ext",
    "codemeta.json",
    "pyproject.toml",
    "run_postprocess_docker.py",
    "run_subject_pipeline.py",
    "THIRD_PARTY_NOTICES.md",
    "workflow_catalog.py",
}
FORBIDDEN_TRACKED_PATH_PARTS = (
    "manuscript_nature_protocols",
    "local_reports",
    "release_artifacts",
)
FORBIDDEN_TRACKED_ASSET_PREFIXES = (
    "paper-latex/",
    "docs/paper/",
    "docs/figures/",
    "scripts/figures/",
    "Resources/Data/",
    "surface/template/",
    "segmentation/weights/",
    "parcellation/weights/",
    "anatomy/surface/template/",
    "anatomy/surface/weights/",
    "anatomy/segmentation/weights/",
    "anatomy/parcellation/weights/",
)
FORBIDDEN_TRACKED_ROOT_PATTERNS = (
    "GOAL_PROGRESS_",
    "LOCAL_RELEASE_VALIDATION_",
)
RETIRED_SOURCE_PATHS = (
    "docs/archive/legacy-copy-20260523",
    "surface/models",
    "surface/networks",
    "parcellation/models/losses.py",
    "parcellation/models/solver.py",
    "parcellation/models/organize_weights.sh",
    "anatomy/surface/models",
    "anatomy/surface/networks",
    "anatomy/parcellation/models/losses.py",
    "anatomy/parcellation/models/solver.py",
    "anatomy/parcellation/models/organize_weights.sh",
)
RETIRED_MODULES = (
    "anatomy.surface.models",
    "anatomy.surface.networks",
    "anatomy.parcellation.models.losses",
    "anatomy.parcellation.models.solver",
)
RETIRED_IMPORT_ROOTS = ("surface", "parcellation", "segmentation")


class RepositoryStructureTest(unittest.TestCase):
    def test_anatomical_sources_share_one_namespace(self) -> None:
        self.assertTrue((REPO_ROOT / "anatomy/__init__.py").is_file())
        for domain in ("segmentation", "parcellation", "surface"):
            self.assertTrue((REPO_ROOT / "anatomy" / domain / "__init__.py").is_file())
            self.assertFalse((REPO_ROOT / domain).exists(), domain)
        cmake = (REPO_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("slicerdsurfer_runtime_resources", cmake)
        from scripts.extension_runtime.runtime_resources import runtime_files
        files = runtime_files(REPO_ROOT)
        self.assertIn("anatomy/parcellation/conversion/data/FreeSurferColorLUT.txt", files)
        self.assertIn("anatomy/surface/pipeline.py", files)

    def test_microstructure_lut_resolves_after_anatomy_migration(self) -> None:
        from microstructure.pipeline import _default_lut_path
        from connectome.pipeline import _default_lut_path as connectome_lut

        lut = _default_lut_path()
        self.assertTrue(lut.is_file())
        self.assertEqual(lut, REPO_ROOT / "anatomy/parcellation/conversion/data/FreeSurferColorLUT.txt")
        self.assertEqual(connectome_lut(), lut)

    def test_top_level_anatomy_import_is_lightweight(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", "import anatomy, sys; assert not any(name in sys.modules for name in ('torch', 'slicer', 'vtk'))"],
            cwd=REPO_ROOT, check=False, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_upstream_sync_keeps_the_external_asset_namespace(self) -> None:
        source = (REPO_ROOT / "scripts/sync_ddsurfer_upstream.py").read_text(encoding="utf-8")
        self.assertIn("ROOT / 'anatomy/surface/ddsurfer_backend'", source)
        self.assertIn("relative = f'surface/{folder}/{name}'", source)
        self.assertNotIn("relative = f'anatomy/surface/{folder}/{name}'", source)

    def test_retired_sources_are_not_part_of_the_current_checkout(self) -> None:
        for relative in RETIRED_SOURCE_PATHS:
            self.assertFalse((REPO_ROOT / relative).exists(), relative)

    def test_runtime_does_not_import_retired_implementations(self) -> None:
        offenders = []
        for relative in (*SCAN_ROOTS, "preprocessing", "scripts"):
            path = REPO_ROOT / relative
            files = [path] if path.is_file() else sorted(path.rglob("*.py"))
            for source in files:
                package = ".".join(source.relative_to(REPO_ROOT).parent.parts)
                with tokenize.open(source) as stream:
                    tree = ast.parse(stream.read(), filename=str(source))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imported = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        if node.level:
                            module = resolve_name("." * node.level + module, package)
                        imported = [module, *(f"{module}.{alias.name}" for alias in node.names)]
                    else:
                        continue
                    for name in imported:
                        if any(name == retired or name.startswith(retired + ".")
                               for retired in (*RETIRED_MODULES, *RETIRED_IMPORT_ROOTS)):
                            offenders.append(f"{source.relative_to(REPO_ROOT)}:{node.lineno}: {name}")
        self.assertEqual(offenders, [])

    def test_tracked_root_files_stay_small_and_intentional(self) -> None:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        root_files = {
            relative
            for relative in result.stdout.splitlines()
            if relative and "/" not in relative and (REPO_ROOT / relative).is_file()
        }
        self.assertEqual(sorted(root_files - ALLOWED_ROOT_FILES), [])

    def test_root_helpers_remain_thin_compatibility_entry_points(self) -> None:
        for name in ("run_subject_pipeline.py", "run_postprocess_docker.py", "workflow_catalog.py"):
            tree = ast.parse((REPO_ROOT / name).read_text(encoding="utf-8"))
            implementations = [node for node in ast.walk(tree) if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
            self.assertEqual(implementations, [], name)

    def test_ci_dependencies_and_local_outputs_have_dedicated_locations(self) -> None:
        self.assertTrue((REPO_ROOT / "scripts/dev/requirements-ci.txt").is_file())
        self.assertFalse((REPO_ROOT / "requirements-ci.txt").exists())
        self.assertFalse((REPO_ROOT / "release_artifacts").exists())
        for name in ("ci.yml", "release.yml"):
            text = (REPO_ROOT / ".github/workflows" / name).read_text(encoding="utf-8")
            self.assertIn("-r scripts/dev/requirements-ci.txt", text)

    def test_actions_workflows_are_manual_only(self) -> None:
        import re
        for workflow in (REPO_ROOT / ".github/workflows").glob("*.yml"):
            text = workflow.read_text(encoding="utf-8")
            events = re.search(r"(?ms)^on:\n(.*?)(?=^\S|\Z)", text)
            self.assertIsNotNone(events, workflow.name)
            self.assertIn("workflow_dispatch:", events.group(1), workflow.name)
            forbidden = r"(?m)^  (pull_request|schedule|workflow_run|release):" if workflow.name == "pages.yml" else r"(?m)^  (push|pull_request|schedule|workflow_run|release):"
            self.assertNotRegex(events.group(1), forbidden)

    def test_cache_settings_preserve_scientific_evidence(self) -> None:
        project = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
        self.assertIn('cache_dir = "ignore/cache/pytest"', project)
        self.assertIn("PYTHONPYCACHEPREFIX ?= $(CURDIR)/ignore/cache/pycache", makefile)
        cleanup = makefile.split("\nclean-local:\n", 1)[1].split("\nlint:\n", 1)[0]
        self.assertNotIn("ignore/local_reports", cleanup)
        self.assertNotIn("SMOKE_ROOT", cleanup)
        self.assertNotIn("ignore/releases", cleanup)

    def test_non_project_artifacts_are_not_tracked(self) -> None:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        offenders: list[str] = []
        for relative in result.stdout.splitlines():
            parts = relative.split("/")
            if any(part in FORBIDDEN_TRACKED_PATH_PARTS for part in parts):
                offenders.append(relative)
                continue
            name = Path(relative).name
            public_pages = {
                'docs/site/index.html', 'docs/site/zh.html', 'docs/site/404.html', 'docs/site/performance.html',
            }
            if (relative.endswith(".html") and relative not in public_pages) or name in {"goal.md", "goal_completion_audit.md", "AGENTS.md"}:
                offenders.append(relative)
                continue
            if "/" not in relative and any(name.startswith(prefix) for prefix in FORBIDDEN_TRACKED_ROOT_PATTERNS):
                offenders.append(relative)
        self.assertEqual(offenders, [])

    def test_product_code_does_not_reference_legacy_root_packages(self) -> None:
        offenders: list[str] = []
        for relative in SCAN_ROOTS:
            path = REPO_ROOT / relative
            files = [path] if path.is_file() else sorted(path.rglob("*.py"))
            for file_path in files:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
                for legacy_name in LEGACY_PACKAGE_NAMES:
                    if legacy_name in text:
                        offenders.append(f"{file_path.relative_to(REPO_ROOT)}: {legacy_name}")
        self.assertEqual(offenders, [])

    def test_third_party_reference_tree_is_gitignored(self) -> None:
        result = subprocess.run(
            ["git", "check-ignore", "-q", "third_party/whitematteranalysis"],
            cwd=REPO_ROOT,
            check=False,
        )
        self.assertEqual(result.returncode, 0)

    def test_model_assets_are_not_tracked_in_code_repository(self) -> None:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        offenders = [
            relative
            for relative in result.stdout.splitlines()
            if relative.startswith(FORBIDDEN_TRACKED_ASSET_PREFIXES)
        ]
        self.assertEqual(offenders, [])

    def test_reference_sources_are_explicitly_excluded_from_cmake_packaging(self) -> None:
        cmake_text = (REPO_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("scripts/extension_runtime/RuntimeResources.cmake", cmake_text)
        import json
        manifest = json.loads((REPO_ROOT / "scripts/extension_runtime/runtime_resources.json").read_text())
        self.assertIn("tractography/slicer_dmri_reference", manifest["excluded_prefixes"])

    def test_untracked_files_are_not_large_runtime_artifacts(self) -> None:
        result = subprocess.run(
            ["git", "ls-files", "-o", "--exclude-standard"],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        offenders: list[str] = []
        for relative in result.stdout.splitlines():
            path = REPO_ROOT / relative
            if path.is_file() and path.stat().st_size > 5 * 1024 * 1024:
                offenders.append(f"{relative}: {path.stat().st_size} bytes")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
