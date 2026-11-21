"""Regression tests for Windows compatibility hardening (#1-#12)."""

from __future__ import annotations

import ctypes
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TESTS_ROOT = Path(__file__).resolve().parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from test_b4_w4_platform import _InstallerCopies
from test_installer_gui_reinstall_gap import load_setup_module


class WindowsCompatHardeningTests(unittest.TestCase):
    # ------------------------------------------------------------------ #1 & #5: MAX_PATH + WinDLL kernel32 PID check
    def test_win_fs_path_and_flat_backup_map_handle_deep_paths(self) -> None:
        from shared import update_journal as uj

        short_p = Path("a/b/c.txt")
        self.assertEqual(uj.win_fs_path(short_p), short_p)

        td = tempfile.mkdtemp()
        try:
            root = Path(td)
            journal_dir = uj.journal_dir(root)
            rel_deep = "/".join(["seg_" + ("x" * 35)] * 6) + "/target.py"
            old_content = "OLD = 'v1'\n"
            dst = uj.win_fs_path(root / rel_deep)
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(old_content, encoding="utf-8")

            backup_dir = journal_dir / "backup"
            uj.win_fs_path(backup_dir).mkdir(parents=True, exist_ok=True)
            shutil.copy2(dst, uj.win_fs_path(backup_dir / "00000.bak"))
            uj.write_update_journal(
                journal_dir,
                target_version="0.9.9",
                backed_up=[rel_deep],
                created=["created/new_file.py"],
                backup_map={rel_deep: "00000.bak"},
            )
            # Simulate partial apply overwrite + new file created before crash
            dst.write_text("BROKEN = True\n", encoding="utf-8")
            created_p = uj.win_fs_path(root / "created/new_file.py")
            created_p.parent.mkdir(parents=True, exist_ok=True)
            created_p.write_text("NEW = 1\n", encoding="utf-8")

            res = uj.recover_interrupted_update(root)
            self.assertIsNotNone(res)
            self.assertEqual(res["status"], "rolled_back")
            self.assertEqual(uj.win_fs_path(root / rel_deep).read_text(encoding="utf-8"), old_content)
            self.assertFalse(created_p.exists())
        finally:
            shutil.rmtree(uj.win_fs_path(Path(td)), ignore_errors=True)

        if os.name == "nt":
            unc = uj.win_fs_path(Path(r"\\server\share" + ("\\nested" * 40)))
            self.assertTrue(str(unc).startswith(r"\\?\UNC\server\share"))

    def test_pid_alive_without_psutil_uses_windll_c_void_p_and_get_last_error(self) -> None:
        from shared import update_journal as uj

        if os.name == "nt":
            self.assertTrue(uj._pid_alive_without_psutil(os.getpid()))
            self.assertFalse(uj._pid_alive_without_psutil(999_999_999))

        fake_kernel = MagicMock()
        fake_kernel.OpenProcess.return_value = None
        with (
            patch.object(os, "name", "nt"),
            patch.object(ctypes, "WinDLL", create=True, return_value=fake_kernel) as windll_cls,
            patch.object(ctypes, "set_last_error", create=True),
            patch.object(ctypes, "get_last_error", create=True, return_value=87),
        ):
            self.assertFalse(uj._pid_alive_without_psutil(424242))
            windll_cls.assert_called_once_with("kernel32", use_last_error=True)
            self.assertIs(fake_kernel.OpenProcess.restype, ctypes.c_void_p)

    def test_win_long_remapped_archive_extraction_both_copies(self) -> None:
        deep_rel = "SlicerDSpace/" + "/".join(["d" * 35] * 6) + "/file.bin"
        copies = _InstallerCopies().copies()
        for label, ns in copies:
            with self.subTest(copy=label), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                arc = root / "base.zip"
                with zipfile.ZipFile(arc, "w") as zf:
                    zf.writestr("SlicerDSpace/SlicerDSpace.s4ext", "slicer_revision 34045\n")
                    zf.writestr(deep_rel, b"payload-ok")
                ext_root = root / "ext"
                try:
                    ns["_extract_base_archive_remapped"](arc, ext_root, "34045", "Slicer-5.10")
                    out_file = ext_root / deep_rel.split("/", 1)[1]
                    win_out = ns["_win_long"](out_file) if os.name == "nt" else out_file
                    self.assertTrue(win_out.is_file())
                    self.assertEqual(win_out.read_bytes(), b"payload-ok")
                finally:
                    if "_win_long" in ns and os.name == "nt":
                        shutil.rmtree(ns["_win_long"](ext_root, force=True), ignore_errors=True)

    # ------------------------------------------------------------------ #2: UTF-8 & PYTHONUTF8
    def test_ddsurfer_atomic_json_and_ui_update_panel_utf8_env(self) -> None:
        backend_files = REPO_ROOT / "anatomy" / "surface" / "ddsurfer_backend" / "utils" / "files.py"
        self.assertIn('encoding="utf-8"', backend_files.read_text(encoding="utf-8"))

        sys.path.insert(0, str(REPO_ROOT / "anatomy" / "surface" / "ddsurfer_backend"))
        try:
            from utils.files import atomic_json

            with tempfile.TemporaryDirectory() as td:
                out = Path(td) / "sub.json"
                atomic_json(out, {"subject": "患者_01_αβ"})
                loaded = json.loads(out.read_text(encoding="utf-8"))
                self.assertEqual(loaded["subject"], "患者_01_αβ")
        finally:
            sys.path.pop(0)

        from shared import ui_update_panel as panel

        recorded_envs: list[dict[str, str]] = []

        class _FakeEnv(dict):
            def insert(self, k: str, v: str) -> None:
                self[k] = v

            @classmethod
            def systemEnvironment(cls):
                return cls()

        class _FakeProc:
            MergedChannels = 1

            def __init__(self, _parent=None) -> None:
                pass

            def setProcessChannelMode(self, _mode) -> None:
                pass

            def setProcessEnvironment(self, env) -> None:
                recorded_envs.append(dict(env))

            def connect(self, _sig, _cb) -> None:
                pass

            def start(self, _prog, _args) -> None:
                pass

        fake_qt = types.SimpleNamespace(
            QProcess=_FakeProc,
            QProcessEnvironment=_FakeEnv,
        )
        fake_owner = types.SimpleNamespace()
        with tempfile.TemporaryDirectory() as td, (
            patch.dict(sys.modules, {"qt": fake_qt})
        ), (
            patch.object(panel, "_resolve_worker_command", return_value=("python", ["worker.py"]))
        ), (
            patch.object(panel, "grouped_launch_command", return_value=("python", ["worker.py"], False))
        ), (
            patch.object(panel, "_parent_window", return_value=None)
        ), (
            patch.object(panel, "_ensure_timer", return_value=types.SimpleNamespace(start=lambda: None))
        ), (
            patch.object(panel, "startup_environment", return_value={"BASE_KEY": "1"})
        ), (
            patch.object(panel, "network_env_overrides", return_value={})
        ):
            panel._start_worker_process(
                fake_owner,
                kind="check",
                worker_args=["check"],
                watchdog_seconds=30,
                tmp_dir=Path(td),
                result_file=Path(td) / "res.json",
            )
        self.assertEqual(len(recorded_envs), 1)
        self.assertEqual(recorded_envs[0]["PYTHONUTF8"], "1")
        self.assertEqual(recorded_envs[0]["PYTHONIOENCODING"], "utf-8")
        self.assertEqual(recorded_envs[0]["BASE_KEY"], "1")

        js_text = (REPO_ROOT / "docs/site/assets/js/installer-customizer.js").read_text(encoding="utf-8")
        self.assertIn('qenv.insert("PYTHONUTF8", "1")', js_text)

    # ------------------------------------------------------------------ #3: Cross-Slicer stale dependency pruning
    def test_prune_stale_bundled_dependencies_drops_old_extension_keeps_valid_paths(self) -> None:
        from shared.user_config import prune_stale_bundled_dependencies

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cur_ext = root / "Slicer 5.10.0" / "Extensions-34045" / "SlicerDSpace"
            cur_bin = cur_ext / "bundled-runtimes" / "mrtrix" / "bin"
            cur_bin.mkdir(parents=True)
            other_ext_existing = root / "Slicer 5.6.2" / "Extensions-32448" / "SlicerDDSurfer" / "bundled-runtimes" / "fsl"
            other_ext_existing.mkdir(parents=True)
            standalone_fsl = root / "standalone_fsl" / "bin"
            standalone_fsl.mkdir(parents=True)

            raw = {
                "mrtrix_bin_dir": str(cur_bin),
                "preprocessing_fsl_bin_dir": str(other_ext_existing),
                "postprocess_runtime": str(
                    root / "Slicer 5.6.2" / "Extensions-32448" / "SlicerDDSurfer" / "bundled-runtimes" / "missing"
                ),
                "dki_matlab_executable": str(standalone_fsl / "matlab"),
            }
            cleaned, pruned = prune_stale_bundled_dependencies(raw, current_extension_root=cur_ext)
            self.assertEqual(set(pruned), {"preprocessing_fsl_bin_dir", "postprocess_runtime"})
            self.assertEqual(set(cleaned.keys()), {"mrtrix_bin_dir", "dki_matlab_executable"})

    # ------------------------------------------------------------------ #4: dti.py WinError 1314 symlink fallback
    def test_dti_symlink_fallback_handles_winerror_1314(self) -> None:
        src_text = (
            REPO_ROOT / "anatomy" / "surface" / "ddsurfer_backend" / "preprocessing" / "dti.py"
        ).read_text(encoding="utf-8")
        self.assertIn("except OSError:", src_text)
        self.assertIn("os.link(", src_text)
        self.assertIn("shutil.copy2(", src_text)

    # ------------------------------------------------------------------ #6: rename_with_retry
    def test_rename_with_retry_handles_transient_permission_error(self) -> None:
        from shared.external_process import rename_with_retry

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "staged"
            src.mkdir()
            (src / "a.txt").write_text("ok", encoding="utf-8")
            dst = Path(td) / "final"
            real_rename = Path.rename
            calls = {"n": 0}

            def flaky_rename(self_p: Path, target_p: Path):
                calls["n"] += 1
                if calls["n"] < 3:
                    raise PermissionError("scanner lock")
                return real_rename(self_p, target_p)

            with patch.object(Path, "rename", flaky_rename):
                out = rename_with_retry(src, dst, timeout=2.0, interval=0.01)
            self.assertEqual(out, dst)
            self.assertTrue((dst / "a.txt").is_file())
            self.assertEqual(calls["n"], 3)

            with patch.object(Path, "rename", side_effect=PermissionError("stuck")):
                with self.assertRaises(PermissionError):
                    rename_with_retry(dst, Path(td) / "never", timeout=0.02, interval=0.01)

    # ------------------------------------------------------------------ #7: fsl_native.py POSIX slashes
    def test_fsl_native_emits_posix_paths_on_windows_paths(self) -> None:
        from preprocessing import fsl_native

        captured: list[list[str]] = []

        def fake_run(cmd):
            captured.append(list(cmd))
            program, *args = cmd
            exe = Path(program).name

            def write(path_str: str | Path, content: str = "fixture") -> None:
                p = Path(path_str)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content, encoding="utf-8")

            if exe == "mrinfo" and "-json_all" in args:
                pair = Path(args[0]).name == "epi.nii.gz"
                size = [4, 4, 4, 2 if pair else 3]
                grad = [[0, 0, 0, 0], [1, 0, 0, 1000], [0, 1, 0, 1000]]
                write(
                    args[args.index("-json_all") + 1],
                    json.dumps(
                        {
                            "size": size,
                            "strides": [1, -2, 3, 4],
                            "spacing": [2, 2, 2, 1],
                            "transform": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                            "keyval": {"dw_scheme": grad},
                        }
                    ),
                )
            elif exe == "mrconvert":
                write(args[1])
                if "-export_grad_fsl" in args:
                    pos = args.index("-export_grad_fsl")
                    write(args[pos + 1], "0 1 0\n0 0 1\n0 0 0\n")
                    write(args[pos + 2], "0 1000 1000\n")
                if "-export_pe_eddy" in args:
                    pos = args.index("-export_pe_eddy")
                    write(args[pos + 1], "0 1 0 0.05\n")
                    write(args[pos + 2], "1 1 1\n")
                if "-export_pe_topup" in args:
                    write(args[args.index("-export_pe_topup") + 1], "0 1 0 0.05\n0 -1 0 0.05\n")
            elif exe == "topup":
                field = next(a.split("=", 1)[1] for a in args if a.startswith("--out="))
                write(field + "_fieldcoef.nii.gz")
                write(field + "_movpar.txt", "0\n")
                write(next(a.split("=", 1)[1] for a in args if a.startswith("--fout=")))
            elif exe == "applytopup":
                prefix = next(a.split("=", 1)[1] for a in args if a.startswith("--out="))
                write(prefix + ".nii.gz")
            elif exe == "eddy":
                prefix = next(a.split("=", 1)[1] for a in args if a.startswith("--out="))
                write(prefix + ".nii.gz")
                write(prefix + ".eddy_rotated_bvecs", "0 1 0\n0 0 1\n0 0 0\n")
            elif exe == "mrcat":
                write(args[2])
            elif exe == "mrtransform":
                write(args[1])
            elif exe in {"mrcalc", "dwi2mask", "maskfilter"}:
                write(args[-1])
            return ""

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            fsl_dir = root / "fsl"
            fsl_dir.mkdir()
            cmds = {}
            for name in ("mrconvert", "mrinfo", "dwi2mask", "maskfilter", "eddy", "mrcat", "mrtransform", "mrcalc", "topup", "applytopup"):
                p = fsl_dir / name
                p.write_text("", encoding="utf-8")
                cmds[name] = str(p.resolve())
            topup_cnf = root / "b02b0.cnf"
            topup_cnf.write_text("", encoding="utf-8")
            in_dwi = root / "dwi.nii.gz"
            in_bvec = root / "bvecs"
            in_bval = root / "bvals"
            se_epi = root / "epi.nii.gz"
            for f in (in_dwi, in_bvec, in_bval, se_epi):
                f.write_text("", encoding="utf-8")
            out_dir = root / "out"
            out_dir.mkdir()
            fsl_native.run_fsl_correction(
                input_dwi=in_dwi,
                input_bvec=in_bvec,
                input_bval=in_bval,
                output_dwi=out_dir / "out.nii.gz",
                output_bvec=out_dir / "out.bvec",
                output_bval=out_dir / "out.bval",
                scratch_dir=root / "scratch",
                phase_encoding="AP",
                readout_time=0.05,
                se_epi_pair=se_epi,
                commands=cmds,
                eddy_options=[],
                run_stage=fake_run,
                topup_config=topup_cnf,
            )

        fsl_cmds = [c for c in captured if Path(c[0]).name in ("topup", "applytopup", "eddy")]
        self.assertEqual(len(fsl_cmds), 3)
        for cmd in fsl_cmds:
            for arg in cmd[1:]:
                if any(
                    arg.startswith(pfx)
                    for pfx in ("--imain=", "--datain=", "--out=", "--fout=", "--config=", "--logout=", "--topup=", "--mask=", "--acqp=", "--index=", "--bvecs=", "--bvals=")
                ):
                    self.assertNotIn("\\", arg, f"FSL flag {arg!r} still has backslash")

    # ------------------------------------------------------------------ #8: python-runtime.json, skipped UKF notice, UKF 32GB warning
    def test_packager_runtime_json_setup_notice_and_ukf_ram_warning(self) -> None:
        from scripts import package_slicer_extension as pkg

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ext = root / "ext"
            ext.mkdir()
            (ext / "SlicerDSpace.s4ext").write_text("arch amd64\n", encoding="utf-8")
            sp = root / "sp"
            sp.mkdir()
            (sp / "mod.py").write_text("x = 1\n", encoding="utf-8")
            pkg._stage_python_site_packages(ext, sp, python_version="3.12")
            meta = json.loads((ext / "bundled-python" / "python-runtime.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["arch"], "x86_64")
            self.assertEqual(meta["slicer_numpy_minimum"], "2.0.2")

        ns = load_setup_module()
        src = (REPO_ROOT / "docs/site/assets/js/installer-customizer.js").read_text(encoding="utf-8")
        self.assertIn('"skipped_cli_subdirs": sorted(_SKIPPED_CLI_SUBDIRS)', src)
        self.assertIn("SlicerDMRI", src)
        self.assertIn("_SKIPPED_CLI_SUBDIRS", ns)

        from tractography import ukf

        inputs = {"volumes": 288, "approximate_shells": [1000.0, 2000.0, 3000.0], "effective_brain_voxels": 950_000}
        warn = ukf._ukf_cpu_memory_runtime_warning(inputs, use_gpu=False, ram_gib=31.7)
        self.assertIsNotNone(warn)
        self.assertIn("31.7 GiB RAM", warn)
        self.assertIsNone(ukf._ukf_cpu_memory_runtime_warning(inputs, use_gpu=True, ram_gib=31.7))
        self.assertIsNone(ukf._ukf_cpu_memory_runtime_warning(inputs, use_gpu=False, ram_gib=64.0))

    # ------------------------------------------------------------------ #9 & #2b: .bat Slicer priority & PYTHONUTF8
    def test_windows_bat_sets_pythonutf8_and_prioritizes_newer_slicer(self) -> None:
        from shared import installer_script_builder as sb

        bat_py = sb.generate_installer_script(target_os="windows")
        res = subprocess.run(
            ["node", "-e", "const ic=require(process.argv[1]); process.stdout.write(ic.buildScript({os:'windows'}));",
             str(REPO_ROOT / "docs/site/assets/js/installer-customizer.js")],
            capture_output=True, text=True, encoding="utf-8", check=True,
        )
        bat_js = res.stdout
        for label, bat in (("python_builder", bat_py), ("website_js", bat_js)):
            with self.subTest(generator=label):
                self.assertIn('set "PYTHONUTF8=1"', bat)
                self.assertIn('set "PYTHONIOENCODING=utf-8"', bat)
                i512 = bat.index("Slicer 5.12*")
                i511 = bat.index("Slicer 5.11*")
                i510 = bat.index("Slicer 5.10*")
                iany = bat.rindex('Slicer*"')
                self.assertLess(i512, i511)
                self.assertLess(i511, i510)
                self.assertLess(i510, iany)

    # ------------------------------------------------------------------ #10: CREATE_NO_WINDOW
    def test_subprocess_no_window_kwargs_and_callers(self) -> None:
        from shared.external_process import subprocess_no_window_kwargs

        with patch.object(os, "name", "nt"):
            kw = subprocess_no_window_kwargs()
            flag = int(getattr(subprocess, "CREATE_NO_WINDOW", 0) or 0)
            if flag:
                self.assertEqual(kw.get("creationflags"), flag)
        with patch.object(os, "name", "posix"):
            self.assertEqual(subprocess_no_window_kwargs(), {})

        for rel in (
            "tractography/ukf.py",
            "tractography/csd.py",
            "tractography/act.py",
            "tractography/wma.py",
            "tractography/wmql.py",
            "tractography/tractcloud.py",
            "connectome/pipeline.py",
            "anatomy/segmentation/dti_slicer.py",
        ):
            self.assertIn("subprocess_no_window_kwargs", (REPO_ROOT / rel).read_text(encoding="utf-8"), rel)

    # ------------------------------------------------------------------ #11: Heartbeat Windows CPU quantum
    def test_heartbeat_ignores_windows_quantum_noise_on_short_interval(self) -> None:
        from shared import run_heartbeat as hb

        now = [0.0]
        cpu = [1.0]
        lines: list[str] = []
        beat = hb.Heartbeat(interval=1, stall_minutes=0.05, out=lines.append, clock=lambda: now[0])
        beat._cpu.sample = lambda: (cpu[0], 1)
        beat.set_stage("UKF")
        for _ in range(4):
            now[0] += 1.0
            # Simulate tiny 0.0156s Windows GetProcessTimes tick per probe
            cpu[0] += 0.015625 if os.name == "nt" else 0.0
            beat.tick()
        self.assertTrue(any(line.startswith("[Stall warning] UKF:") for line in lines), lines)

    # ------------------------------------------------------------------ #12: diffusion_image Windows lock cleanup
    def test_diffusion_image_releases_temp_gz_on_windows_exit(self) -> None:
        import gzip as gzip_mod
        import nibabel as nib
        import numpy as np
        from shared.diffusion_io import diffusion_image

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            nii_tmp = root / "raw.nii"
            nii_gz = root / "data.nii.gz"
            nib.save(nib.Nifti1Image(np.ones((4, 4, 4, 2), dtype=np.float32), np.eye(4)), str(nii_tmp))
            with nii_tmp.open("rb") as src, gzip_mod.open(nii_gz, "wb") as dst:
                shutil.copyfileobj(src, dst)
            nii_tmp.unlink()

            with diffusion_image(nii_gz, root / "work") as img:
                arr = np.asanyarray(img.dataobj)
                self.assertEqual(arr.shape, (4, 4, 4, 2))
            leftovers = list((root / "work").glob(".dwi-read-*"))
            self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
