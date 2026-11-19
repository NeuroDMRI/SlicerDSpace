"""Tests for Apple Silicon MPS acceleration, device propagation, and cross-platform safety."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class MpsAndDeviceWiringTests(unittest.TestCase):
    def test_ddparcel_decoder_scatter_unpool_matches_maxunpool2d_bit_exact(self) -> None:
        import torch
        import torch.nn as nn
        from anatomy.parcellation.models.sub_module import CompetitiveDecoderBlock

        torch.manual_seed(7)
        params = {
            "num_channels": 8,
            "num_filters": 8,
            "kernel_h": 3,
            "kernel_w": 3,
            "kernel_c": 1,
            "stride_conv": 1,
            "pool": 2,
            "stride_pool": 2,
        }
        block = CompetitiveDecoderBlock(params).eval()
        pool = nn.MaxPool2d(2, 2, return_indices=True)
        skip = torch.randn(2, 8, 32, 32, dtype=torch.float32)
        pooled, indices = pool(skip)

        with torch.no_grad():
            ref = block(pooled, skip, indices)
            n, c = pooled.shape[0], pooled.shape[1]
            h_out, w_out = skip.shape[2], skip.shape[3]
            unpool_scatter = pooled.new_zeros((n, c, h_out * w_out))
            unpool_scatter.scatter_(2, indices.reshape(n, c, -1), pooled.reshape(n, c, -1))
            unpool_scatter = unpool_scatter.view(n, c, h_out, w_out)
            ref_unpool = block.unpool(pooled, indices)
            self.assertTrue(torch.equal(ref_unpool, unpool_scatter))
            if hasattr(getattr(torch, "backends", None), "mps") and torch.backends.mps.is_available():
                block_mps = CompetitiveDecoderBlock(params).eval()
                block_mps.load_state_dict(block.state_dict())
                block_mps = block_mps.to("mps")
                out_mps = block_mps(pooled.to("mps"), skip.to("mps"), indices.to("mps")).cpu()
                self.assertTrue(torch.allclose(ref, out_mps, atol=1e-5, rtol=1e-5))

    def test_hardware_adapt_mps_oom_and_memory_compat_without_recommended_max(self) -> None:
        from shared.hardware_adapt import is_gpu_oom, empty_gpu_cache_safely, query_mps_free_total_bytes

        self.assertTrue(is_gpu_oom(RuntimeError("MPS backend out of memory (max allowed: 8.00 GB)")))
        self.assertTrue(is_gpu_oom(RuntimeError("Invalid buffer size allocation for MTLBuffer")))
        self.assertFalse(is_gpu_oom(RuntimeError("unrelated IO failure")))

        fake_mps = SimpleNamespace(
            current_allocated_memory=lambda: 512 * 1024 * 1024,
            synchronize=lambda: None,
            empty_cache=lambda: None,
        )
        fake_torch = SimpleNamespace(
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
            cuda=SimpleNamespace(is_available=lambda: False, is_initialized=lambda: False),
            mps=fake_mps,
        )
        with patch.dict(sys.modules, {"torch": fake_torch}), patch(
            "shared.hardware_adapt.query_host_available_ram_bytes",
            return_value=16 * (1024**3),
        ):
            info = query_mps_free_total_bytes()
            self.assertIsNotNone(info)
            free_b, total_b = info  # type: ignore[misc]
            self.assertEqual(total_b, int(16 * (1024**3) * 0.60))
            self.assertGreater(free_b, 0)
            self.assertLess(free_b, total_b)
            empty_gpu_cache_safely()

    def test_non_mps_solvers_accept_mps_gracefully_without_raising(self) -> None:
        from connectome.pipeline import _resolve_connectome_device
        from microstructure.dki import _resolve_dki_device
        from microstructure.freewater_wls import _resolve_fw_device
        from microstructure.mkcurve import _resolve_mkcurve_device
        from microstructure.noddi_accel import resolve_noddi_device
        from preprocessing.dipy_accel import resolve_dipy_accel_device
        from tractography.dti import _resolve_dti_device
        from tractography.saf import _resolve_saf_device
        from tractography.wma import _normalize_wma_gpu_mode
        from tractography.wmql_accel import _resolve_device as _resolve_wmql_device

        self.assertEqual(_resolve_fw_device("mps"), "cpu")
        self.assertEqual(_resolve_dki_device("mps"), "cpu")
        self.assertEqual(_resolve_mkcurve_device("mps"), "cpu")
        self.assertEqual(resolve_noddi_device("mps"), "cpu")
        self.assertEqual(_resolve_dti_device("mps"), "cpu")
        self.assertEqual(resolve_dipy_accel_device("mps"), "cpu")
        self.assertEqual(_resolve_saf_device("mps"), "cpu")
        self.assertEqual(_resolve_wmql_device("mps"), "cpu")
        self.assertEqual(_normalize_wma_gpu_mode("mps"), "off")
        self.assertEqual(_resolve_connectome_device("mps"), "cpu")

    def test_tractcloud_bootstrap_handles_hyphen_batch_size_and_mps_cap(self) -> None:
        from tractography.tractcloud import TRACTCLOUD_COMPAT_BOOTSTRAP

        self.assertIn("'--batch-size' in sys.argv", TRACTCLOUD_COMPAT_BOOTSTRAP)
        self.assertIn("query_mps_free_total_bytes", TRACTCLOUD_COMPAT_BOOTSTRAP)

    def test_orchestrator_forwards_compute_device_to_preproc_and_dki(self) -> None:
        from shared.orchestrator import ProductOrchestratorConfig, _dki_config, _preprocessing_config

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("data.nii.gz", "bvals", "bvecs", "nodif_brain_mask.nii.gz"):
                p = root / name
                p.write_bytes(b"stub")
            cfg = ProductOrchestratorConfig(
                subject_root=root,
                subject_id="sub01",
                targets=("preprocessing", "microstructure"),
                ddseg_device="cpu",
                surface_device="cpu",
                brain_mask_device="cpu",
            )
            pre = _preprocessing_config(cfg, root, root / "out")
            self.assertEqual(pre.dipy_device, "cpu")
            dki = _dki_config(
                cfg,
                {"dwi": root / "data.nii.gz", "bval": root / "bvals", "bvec": root / "bvecs", "mask": root / "nodif_brain_mask.nii.gz"},
            )
            self.assertEqual(dki.device, "cpu")

    def test_ddsurfer_mps_probe_does_not_require_unused_conv_transpose3d(self) -> None:
        src = (REPO_ROOT / "anatomy/surface/ddsurfer_backend/model/ddsurfer.py").read_text(encoding="utf-8")
        probe_block = src.split("def _mps_supports_conv3d", 1)[1].split("class TANet", 1)[0]
        self.assertIn("F.conv3d", probe_block)
        self.assertNotIn("F.conv_transpose3d", probe_block)


class CrossPlatformReliabilityTests(unittest.TestCase):
    def test_managed_runtime_root_prefers_space_free_macos_dir_with_legacy_and_env_override(self) -> None:
        from shared import managed_runtimes as mr

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets_root = root / "Application Support" / "SlicerDSpace" / "model-assets"
            legacy = assets_root.parent / "runtimes"
            fake_home = root / "home"
            fake_home.mkdir()
            with patch.object(mr, "default_user_model_assets_root", return_value=assets_root), \
                    patch.object(mr.sys, "platform", "darwin"), \
                    patch("pathlib.Path.home", return_value=fake_home), \
                    patch.dict(os.environ, {}, clear=False):
                os.environ.pop("SLICERDSPACE_RUNTIME_ROOT", None)
                chosen = mr.runtime_root()
                self.assertEqual(chosen, (fake_home / ".local" / "share" / "SlicerDSpace" / "runtimes").resolve())
                self.assertNotIn(" ", str(chosen))

                legacy.mkdir(parents=True)
                (legacy / "mrtrix-3.0.8-macos").mkdir()
                self.assertEqual(mr.runtime_root(), legacy.resolve())

                custom = root / "custom-rt"
                with patch.dict(os.environ, {"SLICERDSPACE_RUNTIME_ROOT": str(custom)}):
                    self.assertEqual(mr.runtime_root(), custom.resolve())

    def test_windows_no_window_kwargs_and_surface_wait_command_tree_kill(self) -> None:
        from contextlib import nullcontext
        import importlib.util
        from shared.external_process import subprocess_no_window_kwargs

        with patch("shared.external_process.os.name", "posix"):
            self.assertEqual(subprocess_no_window_kwargs(), {})
        with patch("shared.external_process.os.name", "nt"), \
                patch("shared.external_process.subprocess.CREATE_NO_WINDOW", 0x08000000, create=True):
            self.assertEqual(subprocess_no_window_kwargs(), {"creationflags": 0x08000000})

        backend_root = REPO_ROOT / "anatomy" / "surface" / "ddsurfer_backend"
        saved_modules = {
            k: v for k, v in sys.modules.items()
            if k == "preprocessing" or k.startswith("preprocessing.") or k == "utils" or k.startswith("utils.")
        }
        for k in list(saved_modules):
            sys.modules.pop(k, None)
        spec = importlib.util.spec_from_file_location("run_ddsurfer_pipeline_under_test", backend_root / "run_ddsurfer_pipeline.py")
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(backend_root))
        try:
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
        finally:
            sys.path.remove(str(backend_root))
            for k in list(sys.modules):
                if k == "preprocessing" or k.startswith("preprocessing.") or k == "utils" or k.startswith("utils."):
                    sys.modules.pop(k, None)
            sys.modules.update(saved_modules)

        wait_calls = [0]

        def fake_wait(timeout=None):
            wait_calls[0] += 1
            if wait_calls[0] == 1:
                raise KeyboardInterrupt()
            return 0

        fake_proc = SimpleNamespace(pid=9876, poll=lambda: None, terminate=lambda: None, kill=lambda: None, wait=fake_wait)
        run_calls = []

        with patch.object(mod.os, "name", "nt"), \
                patch.object(mod.subprocess, "CREATE_NO_WINDOW", 0x08000000, create=True), \
                patch.object(mod.subprocess, "Popen", return_value=fake_proc) as popen_mock, \
                patch.object(mod.subprocess, "run", side_effect=lambda *a, **k: run_calls.append((a, k))), \
                patch.object(mod, "interruptible_cleanup", return_value=nullcontext()):
            with self.assertRaises(KeyboardInterrupt):
                mod.wait_command(["cmd", "/c", "echo"])
        self.assertNotIn("start_new_session", popen_mock.call_args.kwargs)
        self.assertEqual(popen_mock.call_args.kwargs["creationflags"], 0x08000000)
        self.assertEqual(run_calls[0][0][0], ["taskkill", "/F", "/T", "/PID", "9876"])
        self.assertEqual(run_calls[0][1]["creationflags"], 0x08000000)

    def test_on_demand_surface_runtime_detects_bundled_and_postprocess_paths(self) -> None:
        from shared import ui_on_demand_resources as od

        with tempfile.TemporaryDirectory() as tmp:
            fake_bundled = Path(tmp) / "bundled-surface"
            fake_bundled.mkdir()
            with patch("shared.native_runtimes.bundled_runtime_roots", return_value=()), \
                    patch("shared.surface_runtime.bundled_surface_runtime", return_value=fake_bundled):
                self.assertTrue(od._surface_runtime_ready())

            fs_home = Path(tmp) / "custom-fs"
            fs_home.mkdir()
            (fs_home / "build-stamp.txt").write_text("freesurfer-7.4.1\n", encoding="utf-8")
            with patch("shared.native_runtimes.bundled_runtime_roots", return_value=()), \
                    patch("shared.surface_runtime.bundled_surface_runtime", return_value=None), \
                    patch("shared.surface_runtime.default_surface_runtime", return_value=Path(tmp) / "missing-prefix"):
                self.assertTrue(od._surface_runtime_ready(str(fs_home)))

    def test_bundled_python_accepts_str_root_and_probes_payload_binaries(self) -> None:
        from shared import bundled_python

        with tempfile.TemporaryDirectory() as tmp:
            ext_root = Path(tmp).resolve() / "SlicerDSpace"
            lib_dir = ext_root / "bundled-python" / "site-packages" / "pkg"
            lib_dir.mkdir(parents=True)
            so_file = lib_dir / "_fast.cpython-312-darwin.so"
            so_file.write_bytes(b"stub")
            mod_root = ext_root / "lib" / "Slicer-5.10" / "qt-scripted-modules"
            mod_root.mkdir(parents=True)

            roots = bundled_python._default_extension_roots(str(mod_root))
            self.assertEqual(roots[0], ext_root.resolve())

            checked: list[list[str]] = []

            def fake_run(cmd, **_kwargs):
                checked.append(cmd)
                if cmd[1] == "-p" and str(so_file) in cmd:
                    return SimpleNamespace(returncode=0)
                if cmd[1] == "-p":
                    return SimpleNamespace(returncode=1)
                return SimpleNamespace(returncode=0)

            bundled_python._DEQUARANTINED_ROOTS.clear()
            with patch.object(bundled_python.sys, "platform", "darwin"), \
                    patch.dict(os.environ, {"SLICERDSPACE_SKIP_DEQUARANTINE": "0"}), \
                    patch("shared.bundled_python.shutil.which", return_value="/usr/bin/xattr"), \
                    patch("shared.bundled_python.subprocess.run", side_effect=fake_run):
                self.assertTrue(bundled_python._clear_macos_quarantine_once(str(mod_root)))
            self.assertTrue(any(cmd[:3] == ["/usr/bin/xattr", "-dr", "com.apple.quarantine"] for cmd in checked))

    def test_network_config_proxy_normalization_socks_and_urlopen_fn(self) -> None:
        from shared import network_config as nc

        self.assertEqual(nc._clean_proxy_url("127.0.0.1:7890"), "http://127.0.0.1:7890")
        self.assertEqual(nc._clean_proxy_url("http=127.0.0.1:7890;https=127.0.0.1:8899"), "http://127.0.0.1:8899")
        with self.assertRaisesRegex(RuntimeError, "SOCKS"):
            nc.open_url("https://example.org", proxy_url="socks5://127.0.0.1:1080")
        seen = []

        def fake_urlopen(req, timeout=60):
            seen.append((req, timeout))
            return "ok"

        with patch.object(nc, "load_network_settings", return_value={"proxy_url": "", "mirror_prefix": "", "offline_assets_dir": ""}):
            self.assertEqual(nc.open_url("https://example.org", proxy_url="", timeout=15, urlopen_fn=fake_urlopen), "ok")
        self.assertEqual(seen, [("https://example.org", 15)])

    def test_installer_customizer_helpers_intel_mac_cache_range_405_and_language(self) -> None:
        import http.server
        import json
        import shutil
        import subprocess
        import threading
        import types

        node_bin = shutil.which("node")
        if node_bin is None:
            self.skipTest("node is not installed in PATH")

        js_path = REPO_ROOT / "docs" / "site" / "assets" / "js" / "installer-customizer.js"
        node_script = (
            "const ic = require(process.argv[1]);\n"
            "const filesZh = ic.buildSetupExtensionFiles({ os: 'linux', wmaEnabled: false, language: 'zh' });\n"
            "const filesEn = ic.buildSetupExtensionFiles({ os: 'linux', wmaEnabled: false });\n"
            "const py = filesZh.find((f) => f.name.endsWith('SlicerDSpaceSetup.py')).data;\n"
            "const pyEn = filesEn.find((f) => f.name.endsWith('SlicerDSpaceSetup.py')).data;\n"
            "process.stdout.write(JSON.stringify({ py, pyEn }));\n"
        )
        raw = subprocess.run([node_bin, "-e", node_script, str(js_path)], capture_output=True, text=True, check=True).stdout
        payload_js = json.loads(raw)
        self.assertIn('"language":"zh"', payload_js["py"])
        self.assertIn('"language":"en"', payload_js["pyEn"])

        qt_stub = types.ModuleType("qt")
        slicer_stub = types.ModuleType("slicer")
        slm_stub = types.ModuleType("slicer.ScriptedLoadableModule")
        slm_stub.ScriptedLoadableModule = object
        slm_stub.ScriptedLoadableModuleWidget = object
        slicer_stub.ScriptedLoadableModule = slm_stub
        saved = {k: sys.modules.get(k) for k in ("qt", "slicer", "slicer.ScriptedLoadableModule")}
        ns: dict[str, object] = {}
        try:
            sys.modules["qt"] = qt_stub
            sys.modules["slicer"] = slicer_stub
            sys.modules["slicer.ScriptedLoadableModule"] = slm_stub
            exec(compile(payload_js["py"], "SlicerDSpaceSetup.py", "exec"), ns)
        finally:
            for k, orig in saved.items():
                if orig is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = orig

        self.assertEqual(ns["_clean_proxy"]("127.0.0.1:7890"), "http://127.0.0.1:7890")
        self.assertEqual(ns["_clean_proxy"]("http=127.0.0.1:1;https=127.0.0.1:2"), "http://127.0.0.1:2")
        with patch.object(ns["sys"], "platform", "darwin"), patch.object(ns["platform"], "machine", return_value="x86_64"):
            with patch("subprocess.check_output", return_value="0\n"):
                self.assertTrue(ns["_is_intel_mac"]())
            with patch("subprocess.check_output", return_value="1\n"):
                self.assertFalse(ns["_is_intel_mac"]())
        with patch("shutil.disk_usage", return_value=SimpleNamespace(free=100)):
            with self.assertRaisesRegex(RuntimeError, "Insufficient disk space"):
                ns["_pick_cache_dir"](need_bytes=10 * 1024 * 1024)

        data = b"RANGE-405-FALLBACK-OK" * 1024
        sha = __import__("hashlib").sha256(data).hexdigest()

        class NoRangeHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: object) -> None:
                pass

            def do_GET(self) -> None:
                if self.headers.get("Range"):
                    self.send_response(405)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), NoRangeHandler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as td:
                dest = Path(td) / "no_range.bin"
                got = ns["_download"](
                    [f"http://127.0.0.1:{srv.server_address[1]}/no_range.bin"],
                    dest,
                    expected_sha256=sha,
                    expected_size=len(data),
                    allow_mirror=False,
                )
                self.assertEqual(got.read_bytes(), data)
        finally:
            srv.shutdown()
            srv.server_close()

    def test_input_import_bom_transposed_bvecs_validation_and_subject_id(self) -> None:
        import nibabel as nib
        from shared.input_import import InputImportConfig, prepare_inputs
        from shared.subject_id import derive_subject_id
        from shared.validation import InputValidationConfig, validate_subject_inputs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dwi = root / "raw.nii.gz"
            nib.save(nib.Nifti1Image(np.ones((3, 3, 3, 4), dtype=np.float32), np.eye(4)), str(dwi))
            mask = root / "mask.nii.gz"
            nib.save(nib.Nifti1Image(np.ones((3, 3, 3), dtype=np.uint8), np.eye(4)), str(mask))
            (root / "raw.bval").write_bytes(b"\xef\xbb\xbf0 1000 1000 2000\n")
            (root / "raw.bvec").write_bytes(b"\xef\xbb\xbf0 0 0\n1 0 0\n0 1 0\n0 0 1\n")

            subj = root / "sub_bom"
            prepare_inputs(InputImportConfig(source=dwi, subject_root=subj, subject_id="sub_bom", source_kind="nifti"))
            (subj / "input" / "nodif_brain_mask.nii.gz").write_bytes(mask.read_bytes())
            self.assertFalse((subj / "input" / "bvals").read_bytes().startswith(b"\xef\xbb\xbf"))
            self.assertFalse((subj / "input" / "bvecs").read_bytes().startswith(b"\xef\xbb\xbf"))
            bvecs_arr = np.loadtxt(subj / "input" / "bvecs")
            self.assertEqual(bvecs_arr.shape, (3, 4))

            val = validate_subject_inputs(InputValidationConfig(subject_id="sub_bom", input_root=subj, dependency_groups=(), brain_mask_backend="provided"))
            self.assertTrue(val.passed)

        cjk_id = derive_subject_id("/tmp/张三")
        self.assertTrue(cjk_id.startswith("sub_"))
        self.assertEqual(len(cjk_id), 12)
        self.assertEqual(derive_subject_id("/tmp/Müller 01 (a)"), "M_ller_01__a_")
        self.assertEqual(derive_subject_id("/tmp/..nii.gz"), "")
        self.assertEqual(derive_subject_id(""), "")

    def test_sample_data_os_link_fallback_and_numeric_threads_vars(self) -> None:
        import errno
        import hashlib
        from shared.numeric_threads import numerical_threads
        from shared.sample_data import SampleDataEntry, fetch_sample_source

        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.bin"
            dst = Path(tmp) / "dst.bin"
            payload = b"sample-bytes"
            src.write_bytes(payload)
            entry = SampleDataEntry("demo", "Demo", str(src), "local", "CC0", sha256=hashlib.sha256(payload).hexdigest())
            with patch("shared.sample_data.os.link", side_effect=OSError(errno.ENOTSUP, "no hardlink")):
                out = fetch_sample_source(entry, dst)
            self.assertEqual(out.read_bytes(), payload)
            with self.assertRaises(FileExistsError):
                fetch_sample_source(entry, dst)

        with patch.dict(os.environ, {"VECLIB_MAXIMUM_THREADS": "9", "NUMEXPR_NUM_THREADS": "9"}, clear=False):
            with numerical_threads(1):
                self.assertEqual(os.environ.get("VECLIB_MAXIMUM_THREADS"), "1")
                self.assertEqual(os.environ.get("NUMEXPR_NUM_THREADS"), "1")
            self.assertEqual(os.environ.get("VECLIB_MAXIMUM_THREADS"), "9")
            self.assertEqual(os.environ.get("NUMEXPR_NUM_THREADS"), "9")

    def test_installer_script_builder_hardening_and_eddy_threads_fallback(self) -> None:
        import ast
        from preprocessing.pipeline import _resolved_eddy_threads
        from shared.installer_script_builder import generate_installer_script

        self.assertGreaterEqual(_resolved_eddy_threads(SimpleNamespace()), 1)

        script = generate_installer_script(
            target_os="linux",
            mode="base",
            slicer_path="/custom/Slicer",
            language="zh",
        )
        py_block = script.split("# __SLICERDSPACE_PY_BEGIN__", 1)[1].split("# __SLICERDSPACE_PY_END__", 1)[0]
        ast.parse(py_block)
        self.assertIn('ui_cfg.get("language")', py_block)
        ns: dict[str, object] = {"__file__": str(REPO_ROOT / "Install-SlicerDSpace.sh")}
        exec(compile(py_block, "Install-SlicerDSpace.sh", "exec"), ns)
        self.assertEqual(ns["_clean_proxy"]("127.0.0.1:7890"), "http://127.0.0.1:7890")
        self.assertEqual(ns["_clean_proxy"]("http=127.0.0.1:1;https=127.0.0.1:2"), "http://127.0.0.1:2")
        with patch.object(ns["sys"], "platform", "darwin"), patch.object(ns["platform"], "machine", return_value="x86_64"):
            with patch("subprocess.check_output", return_value="0\n"):
                self.assertTrue(ns["_is_intel_mac"]())
            with patch("subprocess.check_output", return_value="1\n"):
                self.assertFalse(ns["_is_intel_mac"]())
        with patch("shutil.disk_usage", return_value=SimpleNamespace(free=100)):
            with self.assertRaisesRegex(RuntimeError, "Insufficient disk space"):
                ns["_pick_cache_dir"](need_bytes=10 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
