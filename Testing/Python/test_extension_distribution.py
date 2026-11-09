"""Distribution checks need only Python stdlib and CMake, not model assets."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import shutil
import stat
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts.audit_extension_package import audit_package
from scripts.build_slicer_sdk import build
from scripts.extension_runtime.runtime_resources import runtime_files
from scripts.package_slicer_extension import package_extension
from scripts.stage_slicer_extension import stage_extension
from scripts.audit_documentation import audit as audit_documentation
from scripts.release_audit import _extract_cmake_set_items, _runtime_packaging_issues

ROOT = Path(__file__).resolve().parents[2]


class ExtensionDistributionTest(unittest.TestCase):
    def test_release_audit_accepts_inline_and_multiline_cmake_lists(self):
        for text in ('set(MODULE_PYTHON_SCRIPTS ${MODULE_NAME}.py)\nnext_command(wrong)',
                     'set(MODULE_PYTHON_SCRIPTS\n  # entry point only\n  "${MODULE_NAME}.py"\n)'):
            self.assertEqual(_extract_cmake_set_items(text, "MODULE_PYTHON_SCRIPTS"), ["${MODULE_NAME}.py"])
        self.assertEqual(_extract_cmake_set_items("set(OTHER x)", "MODULE_PYTHON_SCRIPTS"), [])
        self.assertEqual(_runtime_packaging_issues(ROOT, (ROOT / "CMakeLists.txt").read_text()), [])
        self.assertTrue(_runtime_packaging_issues(ROOT, "# shared runtime selection disconnected"))

    def test_release_audit_rejects_removed_and_reference_only_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            install = stage_extension(root / "install", slicer_version="Slicer-5.10", clean=True, link_mode="copy")
            module = Path(install["module_dir"])
            manifest = module / "scripts/extension_runtime/runtime_resources.json"
            spec = json.loads(manifest.read_text())
            spec["files"].remove("tractography/wmql_queries/AF_CST.qry")
            spec["trees"] = [tree for tree in spec["trees"] if tree["root"] != "tractography"]
            spec["excluded_prefixes"].remove("tractography/slicer_dmri_reference")
            relative = "tractography/slicer_dmri_reference/unsafe.py"
            target = module / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("# reference, not a runtime module")
            spec["files"].append(relative)
            manifest.write_text(json.dumps(spec))
            issues = _runtime_packaging_issues(module, (ROOT / "CMakeLists.txt").read_text())
            paths = {issue.path for issue in issues}
            self.assertIn("tractography/wmql_queries/AF_CST.qry", paths)
            self.assertIn(relative, paths)

    def test_cmake_and_python_select_identical_resources(self):
        cmake = shutil.which("cmake")
        if not cmake:
            self.skipTest("CMake unavailable; real SDK validation remains required")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "files.txt"
            subprocess.run([cmake, f"-DSLICERDSPACE_RESOURCE_ROOT={ROOT}",
                            f"-DSLICERDSPACE_RESOURCE_OUTPUT={output}",
                            "-P", str(ROOT / "scripts/extension_runtime/RuntimeResources.cmake")], check=True)
            self.assertEqual(output.read_text().splitlines(), runtime_files(ROOT))

    def test_declared_resources_include_non_python_prerequisites(self):
        selected = runtime_files(ROOT)
        for name in ("tractography/wmql_queries/AF_CST.qry",
                     "scripts/wsl/WslCommandBridge.cs", "scripts/wsl/requirements-runtime.txt",
                     "tractography/saf_backend/LICENSE.md", "LICENSE.txt",
                     "anatomy/parcellation/conversion/LICENSE.txt",
                     "anatomy/surface/ddsurfer_backend/LICENSE.txt",
                     "licenses/FreeSurfer.txt", "licenses/PyTorch3D.txt", "licenses/provenance.json",
                     "docs/licenses/pnl_brain_mask.md",  # A-O2: linked from docs/setup/brain_mask_gui_config.md
                     "anatomy/parcellation/conversion/data/FreeSurferColorLUT.txt",
                     "shared/pnl_mask_runner.py",
                     "shared/mrtrix_preproc_runner.py", "shared/managed_runtimes.py",
                     "shared/ui_managed_runtimes.py", "scripts/setup_managed_runtime.py",
                     "shared/wsl_bridge.py", "scripts/wsl/configure_device.py",
                     "docs/release/benchmarks/module_runtimes_20261013.csv",
                     "docs/release/benchmarks/module_runtimes_20261015.csv"):
            self.assertIn(name, selected)
        self.assertFalse(any("/tests/" in name or "/slicer_dmri_reference/" in name for name in selected))
        # A-O2: every relative .md link of a packaged setup page that stays inside docs/ points to a packaged file.
        link = re.compile(r"\]\((\.\./licenses/[^)#]+\.md)")
        for page in (name for name in selected if name.startswith("docs/setup/") and name.endswith(".md")):
            for target in link.findall((ROOT / page).read_text(encoding="utf-8")):
                resolved = os.path.normpath(os.path.join(os.path.dirname(page), target)).replace(os.sep, "/")
                self.assertIn(resolved, selected, f"{page} links {target}")
        self.assertFalse(any(name.startswith('scripts/figures/') for name in selected))
        self.assertNotIn('scripts/sync_website_logo.py', selected)
        self.assertEqual([name for name in selected if "/" not in name and name.endswith(".py")], ["SlicerDSpace.py"])

    def test_gpu_and_setup_launchers_are_required_even_with_globs(self):
        with tempfile.TemporaryDirectory() as directory:
            install = stage_extension(Path(directory) / "install", slicer_version="Slicer-5.10", clean=True, link_mode="copy")
            module = Path(install["module_dir"])
            for relative in ("shared/pnl_mask_runner.py",
                             "shared/mrtrix_preproc_runner.py", "scripts/setup_managed_runtime.py"):
                with self.subTest(resource=relative):
                    target = module / relative
                    original = target.read_bytes()
                    target.unlink()
                    with self.assertRaisesRegex(FileNotFoundError, "Required runtime resource"):
                        runtime_files(module)
                    target.write_bytes(original)

    def test_staging_rejects_source_overwrite(self):
        # The checkout's actual directory name must not determine test behavior.
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "SlicerDSpace"
            source.mkdir()
            sentinel = source / "sentinel.txt"
            sentinel.write_text("source must remain intact", encoding="utf-8")
            with patch("scripts.stage_slicer_extension.REPO_ROOT", source), \
                    patch("scripts.stage_slicer_extension.runtime_files", return_value=[]):
                for clean in (True, False):
                    with self.subTest(clean=clean):
                        with self.assertRaisesRegex(ValueError, "source checkout"):
                            stage_extension(source.parent, slicer_version="Slicer-5.10", clean=clean, link_mode="copy")
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "source must remain intact")

    def test_staged_archive_contains_exact_runtime_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            install = stage_extension(root / "install", slicer_version="Slicer-5.10", clean=True, link_mode="copy")
            archive = root / "SlicerDSpace.tar.gz"
            package_extension(Path(install["extension_root"]), archive)
            self.assertEqual(audit_package(archive, ROOT)["status"], "pass")
            module = Path(install["module_dir"])
            self.assertEqual(audit_documentation(module, reference_root=ROOT)["status"], "pass")
            actual = sorted(path.relative_to(module).as_posix() for path in module.rglob("*") if path.is_file())
            self.assertEqual(actual, runtime_files(ROOT))
            (module / "tractography/wmql_queries/AF_CST.qry").unlink()
            package_extension(Path(install["extension_root"]), archive)
            audit = audit_package(archive, ROOT)
            self.assertEqual(audit["status"], "fail")
            self.assertIn("tractography/wmql_queries/AF_CST.qry", audit["missing"])

    def test_runtime_document_audit_does_not_hide_missing_user_guides(self):
        with tempfile.TemporaryDirectory() as directory:
            install = stage_extension(Path(directory), slicer_version="Slicer-5.10", clean=True, link_mode="copy")
            module = Path(install["module_dir"])
            (module / "docs/setup/QUICKSTART.md").unlink()
            audit = audit_documentation(module, reference_root=ROOT)
            self.assertEqual(audit["status"], "fail")
            self.assertTrue(any(item["file"] == "docs/setup/QUICKSTART.md" for item in audit["findings"]))

    def test_tampered_and_unexpected_resources_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            install = stage_extension(root / "install", slicer_version="Slicer-5.10", clean=True, link_mode="copy")
            module = Path(install["module_dir"])
            (module / "scripts/wsl/requirements-runtime.txt").write_text("not the declared requirements")
            (module / "stray_module.py").write_text("# must not be registered by Slicer")
            archive = root / "tampered.tar.gz"
            package_extension(Path(install["extension_root"]), archive)
            audit = audit_package(archive, ROOT)
            self.assertEqual(audit["status"], "fail")
            self.assertIn("scripts/wsl/requirements-runtime.txt", audit["changed"])
            self.assertIn("stray_module.py", audit["unexpected"])

    def test_duplicate_archive_roots_and_links_fail(self):
        import io
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "unsafe.tar.gz"
            with tarfile.open(archive, "w:gz") as stream:
                for name in ("one/lib/Slicer-5.10/qt-scripted-modules/SlicerDSpace.py",
                             "two/lib/Slicer-5.10/qt-scripted-modules/SlicerDSpace.py",
                             "../escape.txt"):
                    item = tarfile.TarInfo(name)
                    item.size = 1
                    stream.addfile(item, io.BytesIO(b"x"))
                link = tarfile.TarInfo("link")
                link.type = tarfile.SYMTYPE
                link.linkname = "/outside"
                stream.addfile(link)
            audit = audit_package(archive, ROOT)
            self.assertEqual(audit["status"], "fail")
            self.assertEqual(len(audit["module_roots"]), 2)
            self.assertTrue(audit["unsafe"])
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                link = zipfile.ZipInfo("module-link")
                link.create_system = 3
                link.external_attr = (stat.S_IFLNK | 0o777) << 16
                stream.writestr(link, "/outside")
            self.assertIn("Archive link: module-link", audit_package(archive, ROOT)["unsafe"])

    def test_sdk_missing_is_blocked_not_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = build(argparse.Namespace(source_root=ROOT, build_dir=root / "build", slicer_dir=root / "missing"))
            self.assertEqual(result["status"], "blocked")
            self.assertEqual(result["steps"], [])
            self.assertEqual(result["clean_install_test"], "NOT_RUN")
            self.assertEqual(json.loads((root / "build/sdk_build_report.json").read_text())["status"], "blocked")

    def test_unified_zip_package_bundles_modular_runtimes_without_activation_key(self):
        from scripts.extension_runtime.archive_layout import inspect_archive_layout
        from shared.native_runtimes import installed_bundle_bin_dirs
        from shared.surface_runtime import bundled_surface_runtime

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            install = stage_extension(root / "install", slicer_version="Slicer-5.10", clean=True, link_mode="copy")
            ext_root = Path(install["extension_root"])
            bundled_dir = ext_root / "bundled-runtimes"
            mrtrix_dir = bundled_dir / "mrtrix-3.0.8-windows-x86_64"
            (mrtrix_dir / "bin").mkdir(parents=True)
            (mrtrix_dir / "bin" / "tckgen.exe").write_bytes(b"MZfixture")
            (mrtrix_dir / "native-runtime.json").write_text('{"name":"mrtrix","version":"3.0.8"}', encoding="utf-8")
            surf_dir = bundled_dir / "surface-runtime-windows-x86_64"
            (surf_dir / "bin").mkdir(parents=True)
            (surf_dir / "bin" / "mris_inflate.exe").write_bytes(b"MZfixture")
            (surf_dir / "runtime.json").write_text('{"activation_key_included": false}', encoding="utf-8")

            archive = root / "SlicerDSpace-v0.1.1-Windows-x64.zip"
            package_extension(ext_root, archive)
            audit = audit_package(archive, ROOT)
            self.assertEqual(audit["status"], "pass", audit)
            self.assertEqual(
                audit["bundled_runtimes"],
                ["mrtrix-3.0.8-windows-x86_64", "surface-runtime-windows-x86_64"],
            )
            layout = inspect_archive_layout(archive)
            self.assertEqual(layout["status"], "pass", layout)
            with patch.dict(
                os.environ,
                {
                    "SLICERDSPACE_BUNDLED_RUNTIMES_ROOT": str(bundled_dir),
                    "SLICERDSPACE_RUNTIME_ROOT": str(root / "empty-runtimes"),
                },
            ):
                self.assertEqual(
                    installed_bundle_bin_dirs("mrtrix", system="Windows", machine="x86_64"),
                    ((mrtrix_dir / "bin").resolve(),),
                )
                self.assertEqual(
                    bundled_surface_runtime(system="Windows", machine="x86_64"),
                    surf_dir.resolve(),
                )

            # Verify that a bundled surface runtime with activation_key_included=true fails audit.
            (surf_dir / "runtime.json").write_text('{"activation_key_included": true}', encoding="utf-8")
            bad_archive = root / "bad.zip"
            package_extension(ext_root, bad_archive)
            bad_audit = audit_package(bad_archive, ROOT)
            self.assertEqual(bad_audit["status"], "fail")
            self.assertTrue(any("activation_key_included" in item for item in bad_audit["unsafe"]))


if __name__ == "__main__":
    unittest.main()
