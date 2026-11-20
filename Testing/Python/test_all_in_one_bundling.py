"""Unit tests for self-contained All-in-One SlicerDSpace extension bundling."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.audit_extension_package import audit_package
from scripts.extension_runtime.archive_layout import inspect_archive_layout
from scripts.package_slicer_extension import package_extension
from scripts.stage_slicer_extension import stage_extension
from shared import backend_sources, brain_mask, bundled_python, model_assets, optional_assets
from shared.external_process import python_tool_environment
from shared.model_asset_integrity import ModelAssetIntegrityError
from shared.selected_setup import build_setup_plan
from tractography import ukf

ROOT = Path(__file__).resolve().parents[2]


class AllInOneBundlingTests(unittest.TestCase):
    def test_bundled_python_activates_matching_version_and_isolates_external_python(self):
        with tempfile.TemporaryDirectory() as tmp:
            ext_root = Path(tmp) / "SlicerDSpace"
            module_root = ext_root / "lib" / "Slicer-5.10" / "qt-scripted-modules"
            sp_dir = ext_root / "bundled-python" / "site-packages"
            bin_dir = sp_dir / "bin"
            module_root.mkdir(parents=True)
            bin_dir.mkdir(parents=True)
            cur_ver = f"{sys.version_info[0]}.{sys.version_info[1]}"
            (ext_root / "bundled-python" / "python-runtime.json").write_text(
                json.dumps({"python_version": cur_ver}), encoding="utf-8"
            )
            self.assertEqual(bundled_python.bundled_site_packages_dirs(module_root), (sp_dir.resolve(),))
            env: dict[str, str] = {"PYTHONPATH": "/existing/path", "PATH": "/usr/bin"}
            bundled_python.inject_bundled_python_env(env, module_root=module_root)
            self.assertIn(str(sp_dir.resolve()), env["PYTHONPATH"].split(os.pathsep))
            self.assertEqual(env["PYTHONPATH"].split(os.pathsep)[0], "/existing/path")
            self.assertIn(str(bin_dir.resolve()), env["PATH"].split(os.pathsep))

            # Mismatched Python version must not activate binary wheels
            (ext_root / "bundled-python" / "python-runtime.json").write_text(
                json.dumps({"python_version": "2.7"}), encoding="utf-8"
            )
            self.assertEqual(bundled_python.bundled_site_packages_dirs(module_root), ())

            # Restore matching version and check python_tool_environment Slicer vs external venv isolation
            (ext_root / "bundled-python" / "python-runtime.json").write_text(
                json.dumps({"python_version": cur_ver}), encoding="utf-8"
            )
            venv_root = Path(tmp) / "external-venv"
            (venv_root / "bin").mkdir(parents=True)
            (venv_root / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
            venv_py = venv_root / "bin" / "python"
            venv_py.write_text("", encoding="utf-8")
            slicer_py = Path(tmp) / "Slicer-5.10.0" / "bin" / "PythonSlicer"
            slicer_py.parent.mkdir(parents=True)
            slicer_py.write_text("", encoding="utf-8")
            with patch.dict(os.environ, {"SLICERDSPACE_BUNDLED_PYTHON_ROOT": str(ext_root / "bundled-python")}):
                slicer_env = python_tool_environment(slicer_py)
                ext_env = python_tool_environment(venv_py)
            self.assertIn(str(sp_dir.resolve()), slicer_env.get("PYTHONPATH", "").split(os.pathsep))
            self.assertNotIn(str(sp_dir.resolve()), ext_env.get("PYTHONPATH", "").split(os.pathsep))

    def test_bundled_optional_wma_verifies_extracted_files_without_duplicate_zip_archives(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            empty_user_assets = base / "empty-user-assets"
            bundled_assets = base / "bundled-model-assets"
            wma_dir = bundled_assets / "bundles" / "wma"
            cluster_rel = "ORG-800FC-100HCP/cluster_00001.vtp"
            cluster_bytes = b"verified org cluster payload"
            (wma_dir / "ORG-800FC-100HCP").mkdir(parents=True)
            (wma_dir / cluster_rel).write_bytes(cluster_bytes)
            license_bytes = b"ORG CC-BY-4.0 notice"
            (wma_dir / "LICENSE.txt").write_bytes(license_bytes)
            index_payload = json.dumps({
                "files": [{
                    "path": cluster_rel,
                    "sha256": hashlib.sha256(cluster_bytes).hexdigest(),
                    "size_bytes": len(cluster_bytes),
                }]
            }).encode("utf-8")
            (wma_dir / "extracted_resource_index.json").write_bytes(index_payload)

            catalog = base / "catalog.json"
            catalog.write_text(json.dumps({
                "profiles": {
                    "wma": {
                        "directory": "bundles/wma",
                        "required_paths": [
                            "bundles/wma/archives/ORG-800FC-100HCP.zip",
                            "bundles/wma/LICENSE.txt",
                            "bundles/wma/extracted_resource_index.json",
                        ],
                    }
                },
                "asset_records": {
                    "bundles/wma/archives/ORG-800FC-100HCP.zip": {"sha256": "0" * 64, "size_bytes": 999},
                    "bundles/wma/LICENSE.txt": {
                        "sha256": hashlib.sha256(license_bytes).hexdigest(),
                        "size_bytes": len(license_bytes),
                    },
                    "bundles/wma/extracted_resource_index.json": {
                        "sha256": hashlib.sha256(index_payload).hexdigest(),
                        "size_bytes": len(index_payload),
                    },
                },
            }), encoding="utf-8")

            env_patch = {"SLICERDSPACE_BUNDLED_MODEL_ASSETS_ROOT": str(bundled_assets)}
            with patch.object(optional_assets, "MANIFEST", catalog), \
                    patch.object(optional_assets, "configured_model_assets_root", return_value=empty_user_assets), \
                    patch.object(optional_assets, "_persist_sha256_cache"), \
                    patch.dict(os.environ, env_patch, clear=False):
                os.environ.pop("SLICERDSPACE_MODEL_ASSETS_DIR", None)
                self.assertEqual(optional_assets.verified_resource_directory("wma"), wma_dir)
                # Tampering with an extracted file still raises ModelAssetIntegrityError
                (wma_dir / cluster_rel).write_bytes(b"corrupted")
                with self.assertRaises(ModelAssetIntegrityError):
                    optional_assets.verified_resource_directory("wma")

    def test_bundled_sources_and_linux_auto_pnl_route_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            bundled_src = base / "bundled-sources"
            tc_dir = backend_sources.profile_path("tractcloud", bundled_src)
            (tc_dir / "src" / "tractcloud").mkdir(parents=True)
            entry_bytes = b"# tractcloud cli\n"
            (tc_dir / "src" / "tractcloud" / "cli.py").write_bytes(entry_bytes)
            (tc_dir / ".slicerdspace-source.json").write_text(json.dumps({
                "profile": "tractcloud",
                "archive_sha256": backend_sources.SOURCES["tractcloud"]["sha256"],
                "files": {"src/tractcloud/cli.py": hashlib.sha256(entry_bytes).hexdigest()},
            }), encoding="utf-8")
            with patch.object(backend_sources, "source_root", return_value=base / "empty-user-sources"), \
                    patch.dict(os.environ, {"SLICERDSPACE_BUNDLED_SOURCES_ROOT": str(bundled_src)}):
                self.assertEqual(backend_sources.available_source("tractcloud"), tc_dir)

            # Without bundled PNL weights, Linux auto stays managed
            with patch.dict(os.environ, {"SLICERDSPACE_BUNDLED_MODEL_ASSETS_ROOT": str(base / "missing-assets")}):
                self.assertEqual(brain_mask.resolve_pnl_mask_route("auto", "Linux"), "managed")

            # With bundled PNL weights and no managed conda pnl_cnn runtime, Linux auto resolves to native
            pnl_dir = base / "bundled-assets" / "brain_mask" / "pnl_cnn_torch"
            pnl_dir.mkdir(parents=True)
            for fname in ("weights.json", "IITmean_b0_256.nii.gz", "axial.pt", "coronal.pt", "sagittal.pt"):
                (pnl_dir / fname).write_bytes(b"stub")
            with patch.dict(
                os.environ,
                {
                    "SLICERDSPACE_BUNDLED_MODEL_ASSETS_ROOT": str(base / "bundled-assets"),
                    "SLICERDSPACE_RUNTIME_ROOT": str(base / "empty-runtimes"),
                },
            ):
                os.environ.pop("SLICERDSPACE_PNL_MASK_PYTHON", None)
                self.assertEqual(brain_mask.resolve_pnl_mask_route("auto", "Linux"), "native")
                # If a user explicitly installed the managed Conda pnl_cnn runtime, keep managed
                managed_bin = base / "empty-runtimes" / "pnl-cnn-torch-v0.1.0" / "bin"
                managed_bin.mkdir(parents=True)
                (managed_bin / "pnl-brain-mask").write_text("#!/bin/sh\n", encoding="utf-8")
                self.assertEqual(brain_mask.resolve_pnl_mask_route("auto", "Linux"), "managed")

    def test_selected_setup_skips_pytorch_and_ukf_extensions_when_already_available(self):
        ready_mask = SimpleNamespace(
            key="brain_mask_backend", kind="setting", status="available", required=True,
            targets=("microstructure",), label="mask", details={"backend": "pnl_cnn"},
            install_group="brain_mask", install_hint="",
        )
        ready_ukf = SimpleNamespace(
            key="command:UKFTractography", kind="command", status="available", required=True,
            targets=("tractography",), label="UKFTractography", details={},
            install_group="ukf", install_hint="",
        )
        res = SimpleNamespace(statuses=[ready_mask, ready_ukf], targets=("microstructure", "tractography"))
        plan = build_setup_plan(
            res,
            need_torch=False,
            need_ukf=True,
            brain_mask_backend="pnl_cnn",
            pnl_mask_route="native",
            closure_targets=("microstructure", "tractography"),
            system="Windows",
            machine="x86_64",
        )
        self.assertEqual(plan["extensions"], [])

    def test_package_extension_all_in_one_rewrites_archive_depends_and_passes_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            install = stage_extension(root / "install", slicer_version="Slicer-5.10", clean=True, link_mode="copy")
            ext_root = Path(install["extension_root"])

            fake_sp = root / "site-packages"
            # Include a 'tests' subfolder inside a 3rd-party wheel to verify archive_layout allows it
            (fake_sp / "nibabel" / "tests").mkdir(parents=True)
            (fake_sp / "nibabel" / "__init__.py").write_text("__version__ = '5.3.0'\n", encoding="utf-8")
            (fake_sp / "nibabel" / "tests" / "test_pkg.py").write_text("x = 1\n", encoding="utf-8")
            # Ensure unused long-path C++ headers under torch/include are pruned
            (fake_sp / "torch" / "include" / "ATen").mkdir(parents=True)
            (fake_sp / "torch" / "include" / "ATen" / "long_header.h").write_text("// hdr\n", encoding="utf-8")
            (fake_sp / "bin").mkdir(parents=True)
            (fake_sp / "bin" / "helper_cli.py").write_text("#!/usr/bin/env python\n", encoding="utf-8")

            fake_cli = root / "cli-modules"
            fake_cli.mkdir()
            ukf_bin = fake_cli / ("UKFTractography.exe" if os.name == "nt" else "UKFTractography")
            ukf_bin.write_bytes(b"UKFBIN")
            (fake_cli / "UKFTractography.xml").write_text("<executable/>", encoding="utf-8")

            wma_base = ext_root / "bundled-model-assets" / "bundles" / "wma"
            for atlas_marker in (
                wma_base / "ORG-RegAtlas-100HCP" / "registration_atlas.vtk",
                wma_base / "east-west" / "East-West_WM_Atlas_Reg" / "registration_atlas.vtk",
                wma_base / "naba" / "NABA-RegAtlas" / "registration_atlas.vtk",
            ):
                atlas_marker.parent.mkdir(parents=True, exist_ok=True)
                atlas_marker.write_bytes(b"VTK")

            archive = root / "SlicerDSpace-allinone.zip"
            pkg = package_extension(
                ext_root,
                archive,
                cli_modules_dir=fake_cli,
                python_site_packages=fake_sp,
                python_version="3.12",
            )
            self.assertEqual(pkg["bundled_python"]["python_version"], "3.12")
            self.assertIn(ukf_bin.name, pkg["bundled_cli_modules"])
            # Repo SlicerDSpace.s4ext stays 'depends PyTorch', while staged archive s4ext is 'depends NA'
            self.assertIn("depends PyTorch", (ROOT / "SlicerDSpace.s4ext").read_text(encoding="utf-8"))
            self.assertIn("depends NA", (ext_root / "SlicerDSpace.s4ext").read_text(encoding="utf-8"))

            layout = inspect_archive_layout(archive)
            self.assertEqual(layout["status"], "pass", layout)
            self.assertTrue(layout["bundled_python"])
            self.assertIn(ukf_bin.name, layout["bundled_cli_modules"])
            self.assertEqual(layout["bundled_wma_atlases"], ["east-west", "naba", "org"])

            audit = audit_package(archive, ROOT)
            self.assertEqual(audit["status"], "pass", audit)
            self.assertGreaterEqual(audit["bundled_python_file_count"], 2)
            self.assertEqual(audit["bundled_wma_atlases"], ["east-west", "naba", "org"])

            tar_archive = root / "SlicerDSpace-allinone.tar.gz"
            package_extension(
                ext_root,
                tar_archive,
                cli_modules_dir=fake_cli,
                python_site_packages=fake_sp,
                python_version="3.12",
            )
            import tarfile
            with tarfile.open(tar_archive, "r:gz") as tf:
                names = set(tf.getnames())
                self.assertFalse(any("torch/include" in n for n in names))
                cli_py_info = tf.getmember("SlicerDSpace/bundled-python/site-packages/bin/helper_cli.py")
                self.assertEqual(cli_py_info.mode & 0o777, 0o755)

    def test_setup_selftest_native_runtime_script_prefers_pythonslicer_launcher(self):
        from shared import setup_selftest as selftest

        with tempfile.TemporaryDirectory() as tmp:
            rt = Path(tmp).resolve() / "mrtrix-3.0.8-linux-x86_64"
            (rt / "bin").mkdir(parents=True)
            (rt / "lib").mkdir(parents=True)
            (rt / "native-runtime.json").write_text("{}", encoding="utf-8")
            script = rt / "bin" / "dwi2response"
            script.write_text("#!/usr/bin/env python\n", encoding="utf-8")
            fake_launcher = Path(tmp).resolve() / "PythonSlicer"
            captured = []

            def fake_run(argv, **kwargs):
                captured.append((list(argv), dict(kwargs.get("env") or {})))
                return SimpleNamespace(returncode=0, stdout="ok", stderr="")

            with patch("shared.external_process._default_slicer_python", return_value=fake_launcher), \
                    patch.object(selftest.subprocess, "run", side_effect=fake_run), \
                    patch.dict(os.environ, {"PYTHONHOME": "/bad/pythonhome"}, clear=False):
                selftest._run_tool([script, "-help"], {"PATH": "/usr/bin"})

            self.assertEqual(captured[0][0], [str(fake_launcher), str(script), "-help"])
            self.assertNotIn("PYTHONHOME", captured[0][1])
            self.assertIn(str(rt / "lib"), captured[0][1].get("PYTHONPATH", ""))

    def test_stage_cli_modules_bundles_companion_ukf_library(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            install = stage_extension(root / "install", slicer_version="Slicer-5.10", clean=True, link_mode="copy")
            ext_root = Path(install["extension_root"])

            fake_slicer_lib = root / "UKFExtension" / "lib" / "Slicer-5.10"
            fake_cli = fake_slicer_lib / "cli-modules"
            fake_cli.mkdir(parents=True)
            ukf_bin = fake_cli / ("UKFTractography.exe" if os.name == "nt" else "UKFTractography")
            ukf_bin.write_bytes(b"UKFBIN")
            (fake_cli / "UKFTractography.xml").write_text("<executable/>", encoding="utf-8")
            (fake_slicer_lib / "libUKFBase.dylib").write_bytes(b"DYLIB")

            archive = root / "SlicerDSpace-macOS-test.tar.gz"
            pkg = package_extension(ext_root, archive, cli_modules_dir=fake_cli)
            self.assertIn(ukf_bin.name, pkg["bundled_cli_modules"])
            self.assertTrue((ext_root / "lib" / "Slicer-5.10" / "libUKFBase.dylib").is_file())
            self.assertFalse((ext_root / "lib" / "Slicer-5.10" / "cli-modules" / "libUKFBase.dylib").exists())

            layout = inspect_archive_layout(archive)
            self.assertEqual(layout["status"], "pass", layout)
            audit = audit_package(archive, ROOT)
            self.assertEqual(audit["status"], "pass", audit)

    def test_needs_darwin_arm64_tractcloud_python_skips_venv_when_bundled_python_active(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_sp = Path(tmp).resolve() / "site-packages"
            fake_sp.mkdir()
            with patch("shared.platform_info.host_machine", return_value="arm64"), \
                    patch("shared.platform_info.slicer_machine", return_value="x86_64"), \
                    patch("shared.external_process.is_slicer_python", return_value=True), \
                    patch.object(backend_sources, "available_darwin_arm64_tractcloud_python", return_value=None):
                with patch.object(bundled_python, "bundled_site_packages_dirs", return_value=(fake_sp,)):
                    self.assertFalse(backend_sources.needs_darwin_arm64_tractcloud_python(system="Darwin", machine="x86_64"))
                with patch.object(bundled_python, "bundled_site_packages_dirs", return_value=()):
                    self.assertTrue(backend_sources.needs_darwin_arm64_tractcloud_python(system="Darwin", machine="x86_64"))


    def test_activate_bundled_python_prepends_when_bundling_numpy_and_unloaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            ext_no_np = base / "ext_no_np"
            sp_no_np = ext_no_np / "bundled-python" / "site-packages"
            sp_no_np.mkdir(parents=True)
            ext_np = base / "ext_np"
            sp_np = ext_np / "bundled-python" / "site-packages"
            (sp_np / "numpy").mkdir(parents=True)

            orig_path = list(sys.path)
            orig_numpy = sys.modules.pop("numpy", None)
            try:
                sys.path[:] = ["/existing/slicer/site-packages"]
                with patch.dict(os.environ, {}, clear=False):
                    os.environ.pop(bundled_python.BUNDLED_PYTHON_ENV, None)
                    bundled_python.activate_bundled_python(ext_no_np)
                    self.assertNotEqual(sys.path[0], str(sp_no_np.resolve()))
                    self.assertIn(str(sp_no_np.resolve()), sys.path)
                    self.assertNotIn("SLICERDSPACE_BUNDLED_SITE_PACKAGES", os.environ)

                    sys.path[:] = ["/existing/slicer/site-packages"]
                    bundled_python.activate_bundled_python(ext_np)
                    self.assertEqual(sys.path[0], str(sp_np.resolve()))
                    self.assertEqual(os.environ.get("SLICERDSPACE_BUNDLED_SITE_PACKAGES"), str(sp_np.resolve()))
            finally:
                sys.path[:] = orig_path
                if orig_numpy is not None:
                    sys.modules["numpy"] = orig_numpy

    def test_pnl_mask_native_subprocess_prefers_bundled_numpy_over_higher_priority_pythonpath(self):
        import runpy
        import subprocess

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            decoy_sp = base / "slicer-site-packages"
            bundled_sp = base / "bundled-site-packages"
            (decoy_sp / "numpy").mkdir(parents=True)
            (bundled_sp / "numpy").mkdir(parents=True)
            (decoy_sp / "numpy" / "__init__.py").write_text(
                "__version__ = '2.9.9'\nfloat32 = lambda x: float(x)\ndef asarray(x): return x\n",
                encoding="utf-8",
            )
            (bundled_sp / "numpy" / "__init__.py").write_text(
                "__version__ = '1.26.4'\nfloat32 = lambda x: float(x)\ndef asarray(x): return x\n",
                encoding="utf-8",
            )

            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join([str(decoy_sp), *(p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p)])
            env["SLICERDSPACE_BUNDLED_SITE_PACKAGES"] = str(bundled_sp)
            probe = (
                "import runpy, sys; "
                f"sys.path.insert(0, {str(decoy_sp)!r}); "
                f"ns = runpy.run_path({str((ROOT / 'shared' / 'pnl_mask_native.py').resolve())!r}); "
                "print(ns['np'].__version__); "
                "print(ns['np'].__file__)"
            )
            completed = subprocess.run(
                [sys.executable, "-c", probe],
                check=True,
                capture_output=True,
                env=env,
                text=True,
                encoding="utf-8",
            )
            lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
            self.assertEqual(lines[-2], "1.26.4")
            self.assertEqual(Path(lines[-1]).resolve(), (bundled_sp / "numpy" / "__init__.py").resolve())

    def test_external_python_environment_strips_bundled_site_packages(self):
        from shared.external_process import external_python_environment

        with patch.dict(os.environ, {"SLICERDSPACE_BUNDLED_SITE_PACKAGES": "/tmp/bundled-sp"}, clear=False):
            cleaned = external_python_environment()
        self.assertNotIn("SLICERDSPACE_BUNDLED_SITE_PACKAGES", cleaned)

    def test_run_ddseg_resolves_repo_root(self):
        import importlib

        run_ddseg_mod = importlib.import_module("anatomy.segmentation.run_ddseg")
        self.assertEqual(run_ddseg_mod.REPO_ROOT, ROOT.resolve())


if __name__ == "__main__":
    unittest.main()
