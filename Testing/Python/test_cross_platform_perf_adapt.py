"""Cross-platform performance & low-spec hardware adaptation unit tests.

Covers Linux Server, Windows PC, and macOS (Apple Silicon / Rosetta 2 / Intel)
adaptations across shared hardware helpers, UKF, TractCloud, PNL native brain
mask, DIPY MP-PCA/Gibbs, and CSD/WMA/WMQL audit guards while verifying zero
regressions on high-RAM/GPU servers.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
import xml.etree.ElementTree as ET

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

HAS_TORCH = importlib.util.find_spec("torch") is not None


def _fat_macho_header(cputypes: tuple[int, ...], magic: int = 0xCAFEBABE) -> bytes:
    record = 20 if magic == 0xCAFEBABE else 32
    out = struct.pack(">II", magic, len(cputypes))
    for cputype in cputypes:
        out += struct.pack(">II", cputype, 3).ljust(record, b"\0")
    return out + b"\0" * 64


class SharedHardwareAdaptCrossPlatformTests(unittest.TestCase):
    def test_host_total_ram_gib_macos_sysctl_and_env_cap(self) -> None:
        from shared import hardware_adapt as ha

        fake_sysctl = subprocess.CompletedProcess(
            ["/usr/sbin/sysctl", "-n", "hw.memsize"],
            0,
            stdout=str(32 * ha.GIB) + "\n",
            stderr="",
        )
        with (
            mock.patch.object(ha.sys, "platform", "darwin"),
            mock.patch("subprocess.run", return_value=fake_sysctl),
            mock.patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("SLICERDSPACE_MAX_RAM_GIB", None)
            self.assertAlmostEqual(ha.host_total_ram_gib(), 32.0, places=3)

        with (
            mock.patch.object(ha.sys, "platform", "darwin"),
            mock.patch("subprocess.run", return_value=fake_sysctl),
            mock.patch.dict(os.environ, {"SLICERDSPACE_MAX_RAM_GIB": "12"}, clear=False),
        ):
            self.assertAlmostEqual(ha.host_total_ram_gib(respect_env_cap=True), 12.0, places=3)
            self.assertAlmostEqual(ha.host_total_ram_gib(respect_env_cap=False), 32.0, places=3)

    def test_adaptive_spawn_workers_linux_server_vs_windows_and_macos(self) -> None:
        from shared.hardware_adapt import adaptive_spawn_workers

        # High-RAM Linux server (>48 GiB): preserves requested_workers unchanged (zero regression).
        self.assertEqual(
            adaptive_spawn_workers(
                16,
                max_workers=8,
                platform_name="linux",
                total_ram_gib=256.0,
                available_ram_gib=200.0,
            ),
            16,
        )
        self.assertEqual(
            adaptive_spawn_workers(
                12,
                max_workers=8,
                platform_name="linux",
                total_ram_gib=64.0,
                available_ram_gib=50.0,
            ),
            12,
        )

        # Windows PC (spawn): capped safely on 8/16/32 GiB RAM.
        self.assertLessEqual(
            adaptive_spawn_workers(
                16,
                platform_name="win32",
                total_ram_gib=8.0,
                available_ram_gib=4.0,
            ),
            2,
        )
        self.assertLessEqual(
            adaptive_spawn_workers(
                16,
                platform_name="win32",
                total_ram_gib=16.0,
                available_ram_gib=10.0,
            ),
            3,
        )
        self.assertLessEqual(
            adaptive_spawn_workers(
                16,
                platform_name="win32",
                total_ram_gib=32.0,
                available_ram_gib=22.0,
            ),
            4,
        )

        # macOS Apple Silicon (spawn + unified memory): conservative worker count.
        self.assertLessEqual(
            adaptive_spawn_workers(
                10,
                platform_name="darwin",
                total_ram_gib=16.0,
                available_ram_gib=9.0,
            ),
            3,
        )
        self.assertEqual(
            adaptive_spawn_workers(
                1,
                platform_name="darwin",
                total_ram_gib=8.0,
                available_ram_gib=3.0,
            ),
            1,
        )

    def test_optimal_cpu_torch_threads_cross_platform(self) -> None:
        from shared.hardware_adapt import optimal_cpu_torch_threads

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SLICERDSPACE_STAGE_CPUS", None)
            with mock.patch("shared.compute_schedule.available_resources", return_value=(64, 256 * 1024**3)):
                # Linux server: capped at max_conv_threads=8 for small 2D convs
                self.assertEqual(
                    optimal_cpu_torch_threads(None, max_conv_threads=8, platform_name="linux", is_apple_silicon=False),
                    8,
                )
            with mock.patch("shared.compute_schedule.available_resources", return_value=(8, 16 * 1024**3)):
                # Windows 8-core PC: uses all 8
                self.assertEqual(
                    optimal_cpu_torch_threads(None, max_conv_threads=8, platform_name="win32", is_apple_silicon=False),
                    8,
                )
                # Apple Silicon 8-core (e.g., 4P+4E): leaves 2-core E-core headroom -> 6
                self.assertEqual(
                    optimal_cpu_torch_threads(None, max_conv_threads=8, platform_name="darwin", is_apple_silicon=True),
                    6,
                )
            with mock.patch("shared.compute_schedule.available_resources", return_value=(12, 36 * 1024**3)):
                # Apple Silicon 12-core: (12*3)//4 = 9 -> capped by max_conv_threads=8
                self.assertEqual(
                    optimal_cpu_torch_threads(None, max_conv_threads=8, platform_name="darwin", is_apple_silicon=True),
                    8,
                )
                self.assertEqual(
                    optimal_cpu_torch_threads(None, max_conv_threads=12, platform_name="darwin", is_apple_silicon=True),
                    9,
                )

        # Stage CPU reservation always respected when <= max_conv_threads.
        with mock.patch.dict(os.environ, {"SLICERDSPACE_STAGE_CPUS": "4"}, clear=False):
            self.assertEqual(
                optimal_cpu_torch_threads(None, max_conv_threads=8, platform_name="darwin", is_apple_silicon=True),
                4,
            )

        # High-RAM Linux server opt-in (server_max_threads=36) scales to single-socket physical cores,
        # while Windows PC / macOS / stage-reserved runs keep their conservative caps.
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SLICERDSPACE_STAGE_CPUS", None)
            with (
                mock.patch("shared.compute_schedule.available_resources", return_value=(144, 500 * 1024**3)),
                mock.patch("shared.hardware_adapt.linux_single_socket_physical_cores", return_value=36),
            ):
                self.assertEqual(
                    optimal_cpu_torch_threads(
                        None, max_conv_threads=8, server_max_threads=36, platform_name="linux", is_apple_silicon=False
                    ),
                    36,
                )
            with mock.patch("shared.compute_schedule.available_resources", return_value=(24, 32 * 1024**3)):
                self.assertEqual(
                    optimal_cpu_torch_threads(
                        None, max_conv_threads=8, server_max_threads=36, platform_name="win32", is_apple_silicon=False
                    ),
                    8,
                )
                self.assertEqual(
                    optimal_cpu_torch_threads(
                        None, max_conv_threads=16, server_max_threads=36, platform_name="win32", is_apple_silicon=False
                    ),
                    16,
                )
        with mock.patch.dict(os.environ, {"SLICERDSPACE_STAGE_CPUS": "12"}, clear=False):
            self.assertEqual(
                optimal_cpu_torch_threads(
                    None, max_conv_threads=8, server_max_threads=36, platform_name="linux", is_apple_silicon=False
                ),
                12,
            )

    def test_dki_and_noddi_linux_fork_worker_resolution(self) -> None:
        from microstructure import dki, noddi_accel

        fake_cfg = types.SimpleNamespace(num_threads=None)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SLICERDSPACE_STAGE_CPUS", None)
            with (
                mock.patch.object(dki.sys, "platform", "linux"),
                mock.patch("microstructure.dki.os.sched_getaffinity", return_value=set(range(144)), create=True),
                mock.patch("shared.hardware_adapt.host_total_ram_gib", return_value=500.0),
            ):
                self.assertEqual(dki._resolve_dki_fork_workers(fake_cfg), 72)
            with (
                mock.patch.object(dki.sys, "platform", "win32"),
                mock.patch("shared.hardware_adapt.host_total_ram_gib", return_value=32.0),
            ):
                self.assertEqual(dki._resolve_dki_fork_workers(fake_cfg), 1)

            with (
                mock.patch.object(noddi_accel.sys, "platform", "linux"),
                mock.patch.object(noddi_accel, "_is_linux_fork_safe", return_value=True),
                mock.patch("microstructure.noddi_accel.os.sched_getaffinity", return_value=set(range(144)), create=True),
                mock.patch("shared.hardware_adapt.host_total_ram_gib", return_value=500.0),
            ):
                self.assertEqual(
                    noddi_accel._resolve_noddi_cpu_workers(8, auto_nthreads=True, n_vox=100_000, n_dirs=500),
                    (72, True),
                )
                # Explicit manual nthreads=4 stays on ThreadPoolExecutor (not ForkPool)
                self.assertEqual(
                    noddi_accel._resolve_noddi_cpu_workers(4, auto_nthreads=False, n_vox=100_000, n_dirs=500),
                    (4, False),
                )
            with mock.patch.object(noddi_accel.sys, "platform", "win32"):
                self.assertEqual(
                    noddi_accel._resolve_noddi_cpu_workers(8, auto_nthreads=True, n_vox=100_000, n_dirs=500),
                    (8, False),
                )
            self.assertEqual(noddi_accel.resolve_noddi_device("cpu"), "cpu")


class UKFAdaptivePerformanceTests(unittest.TestCase):
    def test_host_ram_gib_delegates_to_hardware_adapt(self) -> None:
        from tractography import ukf

        with mock.patch("shared.hardware_adapt.host_total_ram_gib", return_value=24.0) as mocked:
            self.assertEqual(ukf._host_ram_gib(), 24.0)
            mocked.assert_called_once()

    def test_resolve_adaptive_cpu_num_threads_matrix(self) -> None:
        from tractography import ukf

        default_params = ukf.parameters()
        explicit_params = ukf.parameters({"numThreads": 12})
        heavy = {"volumes": 288, "effective_brain_voxels": 950_000}
        medium = {"volumes": 64, "effective_brain_voxels": 220_000}

        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("SLICERDSPACE_COHORT_SCHEDULE", "SLICERDSPACE_STAGE_CPUS")
        }
        with mock.patch.dict(os.environ, clean_env, clear=True):
            # 1. High-RAM Linux/Windows servers (>48 GiB): -1 unchanged (ZERO server regression).
            for plat in ("linux", "win32"):
                for ram in (64.0, 256.0):
                    self.assertEqual(
                        ukf.resolve_adaptive_cpu_num_threads(
                            default_params, heavy, use_gpu=False, ram_gib=ram, cpu_count=64, platform_name=plat
                        ),
                        -1,
                        (plat, ram),
                    )

            # 2. Heavy multi-shell/high-res inputs on <= 48 GiB Linux/Windows.
            for plat in ("linux", "win32"):
                self.assertEqual(
                    ukf.resolve_adaptive_cpu_num_threads(
                        default_params, heavy, use_gpu=False, ram_gib=8.0, cpu_count=16, platform_name=plat
                    ),
                    2,
                )
                self.assertEqual(
                    ukf.resolve_adaptive_cpu_num_threads(
                        default_params, heavy, use_gpu=False, ram_gib=16.0, cpu_count=16, platform_name=plat
                    ),
                    4,
                )
                self.assertEqual(
                    ukf.resolve_adaptive_cpu_num_threads(
                        default_params, heavy, use_gpu=False, ram_gib=32.0, cpu_count=16, platform_name=plat
                    ),
                    6,
                )
                self.assertEqual(
                    ukf.resolve_adaptive_cpu_num_threads(
                        default_params, heavy, use_gpu=False, ram_gib=48.0, cpu_count=16, platform_name=plat
                    ),
                    8,
                )
                # Medium dataset on Windows/Linux <= 48 GiB stays unconstrained (-1).
                self.assertEqual(
                    ukf.resolve_adaptive_cpu_num_threads(
                        default_params, medium, use_gpu=False, ram_gib=16.0, cpu_count=16, platform_name=plat
                    ),
                    -1,
                )
                # Optimized standalone UKF binary unlocks full threads on >12 GiB Windows/Linux,
                # while keeping <=12 GiB guarded at 4 threads.
                standalone_exe = Path(f"/opt/SlicerDSpace/bundled-runtimes/ukf-{plat}/bin/UKFTractography")
                self.assertTrue(ukf._is_optimized_standalone_executable(standalone_exe))
                self.assertEqual(
                    ukf.resolve_adaptive_cpu_num_threads(
                        default_params, heavy, use_gpu=False, executable=standalone_exe,
                        ram_gib=32.0, cpu_count=24, platform_name=plat,
                    ),
                    -1,
                )
                self.assertEqual(
                    ukf.resolve_adaptive_cpu_num_threads(
                        default_params, heavy, use_gpu=False, executable=standalone_exe,
                        ram_gib=16.0, cpu_count=16, platform_name=plat,
                    ),
                    -1,
                )
                self.assertEqual(
                    ukf.resolve_adaptive_cpu_num_threads(
                        default_params, heavy, use_gpu=False, executable=standalone_exe,
                        ram_gib=8.0, cpu_count=16, platform_name=plat,
                    ),
                    4,
                )

            # 3. macOS unified-memory & Apple Silicon P+E core protection (even for medium datasets).
            self.assertEqual(
                ukf.resolve_adaptive_cpu_num_threads(
                    default_params, medium, use_gpu=False, ram_gib=8.0, cpu_count=8, platform_name="darwin", is_apple_silicon=True
                ),
                2,
            )
            self.assertEqual(
                ukf.resolve_adaptive_cpu_num_threads(
                    default_params, medium, use_gpu=False, ram_gib=16.0, cpu_count=10, platform_name="darwin", is_apple_silicon=True
                ),
                4,
            )
            self.assertEqual(
                ukf.resolve_adaptive_cpu_num_threads(
                    default_params, medium, use_gpu=False, ram_gib=32.0, cpu_count=12, platform_name="darwin", is_apple_silicon=True
                ),
                6,
            )
            self.assertEqual(
                ukf.resolve_adaptive_cpu_num_threads(
                    default_params, medium, use_gpu=False, ram_gib=64.0, cpu_count=14, platform_name="darwin", is_apple_silicon=True
                ),
                10,
            )
            self.assertEqual(
                ukf.resolve_adaptive_cpu_num_threads(
                    default_params, heavy, use_gpu=False, ram_gib=256.0, cpu_count=24, platform_name="darwin", is_apple_silicon=True
                ),
                10,
            )
            # Small CPU count never exceeds host CPUs.
            self.assertEqual(
                ukf.resolve_adaptive_cpu_num_threads(
                    default_params, heavy, use_gpu=False, ram_gib=32.0, cpu_count=4, platform_name="darwin", is_apple_silicon=True
                ),
                4,
            )

            # 4. Explicit override is always preserved.
            self.assertEqual(
                ukf.resolve_adaptive_cpu_num_threads(
                    explicit_params, heavy, use_gpu=False, ram_gib=8.0, cpu_count=16, platform_name="darwin", is_apple_silicon=True
                ),
                12,
            )

            # 5. GPU bypass always returns selected.numThreads unchanged.
            self.assertEqual(
                ukf.resolve_adaptive_cpu_num_threads(
                    default_params, heavy, use_gpu=True, ram_gib=8.0, cpu_count=16, platform_name="win32"
                ),
                -1,
            )

        # 6. Cohort schedule bypass preserves selected.numThreads.
        with mock.patch.dict(os.environ, {"SLICERDSPACE_COHORT_SCHEDULE": "1", "SLICERDSPACE_STAGE_CPUS": "4"}, clear=False):
            self.assertEqual(
                ukf.resolve_adaptive_cpu_num_threads(
                    default_params, heavy, use_gpu=False, ram_gib=8.0, cpu_count=16, platform_name="win32"
                ),
                -1,
            )

        # 7. SLICERDSPACE_STAGE_CPUS caps both low-RAM and server runs when smaller.
        with mock.patch.dict(os.environ, {"SLICERDSPACE_STAGE_CPUS": "3"}, clear=True):
            self.assertEqual(
                ukf.resolve_adaptive_cpu_num_threads(
                    default_params, heavy, use_gpu=False, ram_gib=16.0, cpu_count=16, platform_name="win32"
                ),
                3,
            )
            self.assertEqual(
                ukf.resolve_adaptive_cpu_num_threads(
                    default_params, heavy, use_gpu=False, ram_gib=256.0, cpu_count=64, platform_name="linux"
                ),
                3,
            )

    def test_ukf_inspect_schema_and_run_use_native_arm64_argv_and_preserve_cache_signature(self) -> None:
        from dataclasses import asdict
        import nibabel as nib
        import vtk
        from shared import external_process as ep
        from shared.pipeline import RawInputBundle
        from tractography import ukf

        def _schema_xml():
            root = ET.Element("executable")
            group = ET.SubElement(root, "parameters")
            for key, value in {**asdict(ukf.UKFParameters()), "dwiFile": "", "maskFile": "", "tracts": ""}.items():
                node = ET.SubElement(group, "boolean" if isinstance(value, bool) else "double")
                ET.SubElement(node, "name").text = key
                ET.SubElement(node, "default").text = str(value).lower()
            return '<?xml version="1.0"?>' + ET.tostring(root, encoding="unicode")

        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            arch = root / "arch"
            arch.write_text("#!/bin/sh\n", encoding="utf-8")
            fat = _fat_macho_header((0x01000007, 0x0100000C))
            ukf_bin = root / "UKFTractography"
            dwi_conv = root / "DWIConvert"
            ukf_bin.write_bytes(fat)
            dwi_conv.write_bytes(fat)

            # 1. inspect_schema wraps universal binary with /usr/bin/arch -arm64 under Rosetta.
            seen_run: list[list[str]] = []

            def fake_run(cmd, **kwargs):
                seen_run.append(list(cmd))
                return mock.Mock(returncode=0, stdout=_schema_xml(), stderr="")

            with (
                mock.patch.object(ep, "_ARCH", str(arch)),
                mock.patch.object(ep.sys, "platform", "darwin"),
                mock.patch.object(ep, "process_is_translated", return_value=True),
                mock.patch.dict(os.environ, {}, clear=False),
                mock.patch.object(ukf.subprocess, "run", side_effect=fake_run),
            ):
                os.environ.pop(ep.ARM64_EXEC_DISABLE_ENV, None)
                ukf.inspect_schema(ukf_bin, {"PATH": "/usr/bin"})
            self.assertEqual(seen_run[0], [str(arch), "-arm64", str(ukf_bin), "--xml"])

            # 2. run_ukf wraps both DWIConvert and UKFTractography with native_arm64_argv,
            #    applies adaptive --numThreads to execution without mutating signature().
            affine = np.eye(4)
            dwi = root / "data.nii.gz"
            mask = root / "mask.nii.gz"
            bval, bvec = root / "bvals", root / "bvecs"
            g = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], float)
            g[4:] /= np.sqrt(2)
            np.savetxt(bvec, g.T)
            np.savetxt(bval, [[0, 1000, 1000, 1000, 1000, 1000, 1000]])
            nib.save(nib.Nifti1Image(np.ones((6, 6, 6, 7), np.float32), affine), dwi)
            nib.save(nib.Nifti1Image(np.ones((6, 6, 6), np.uint8), affine), mask)
            raw = RawInputBundle(root, "sub01", root, dwi, bval, bvec, mask)

            fixture = root / "fixture.vtk"
            pts, cells = vtk.vtkPoints(), vtk.vtkCellArray()
            pts.SetDataTypeToDouble()
            for off in (0, 1):
                line = vtk.vtkPolyLine()
                line.GetPointIds().SetNumberOfIds(3)
                for idx in range(3):
                    line.GetPointIds().SetId(idx, pts.InsertNextPoint(idx + 1.0, off + 1.0, 2.0))
                cells.InsertNextCell(line)
            pd = vtk.vtkPolyData()
            pd.SetPoints(pts)
            pd.SetLines(cells)
            for idx in (1, 2):
                fa = vtk.vtkFloatArray()
                fa.SetName(f"FA{idx}")
                fa.SetNumberOfTuples(pts.GetNumberOfPoints())
                fa.Fill(0.5)
                pd.GetPointData().AddArray(fa)
                ten = vtk.vtkFloatArray()
                ten.SetName(f"tensor{idx}")
                ten.SetNumberOfComponents(9)
                ten.SetNumberOfTuples(pts.GetNumberOfPoints())
                ten.Fill(0.001)
                pd.GetPointData().AddArray(ten)
            w = vtk.vtkPolyDataWriter()
            w.SetFileName(str(fixture))
            w.SetInputData(pd)
            w.Write()

            popen_cmds: list[list[str]] = []
            logs: list[str] = []

            def fake_popen(cmd, **kwargs):
                popen_cmds.append(list(cmd))
                if "--tracts" in cmd:
                    shutil.copyfile(fixture, cmd[cmd.index("--tracts") + 1])
                else:
                    Path(cmd[cmd.index("--outputVolume") + 1]).write_text("probe")
                return mock.Mock(wait=mock.Mock(return_value=0), poll=mock.Mock(return_value=0))

            expected_sig = ukf.signature(raw, None, ukf_bin, dwi_conv)
            with (
                mock.patch.object(ep, "_ARCH", str(arch)),
                mock.patch.object(ep.sys, "platform", "darwin"),
                mock.patch.object(ep, "process_is_translated", return_value=True),
                mock.patch.object(ukf, "resolve_executable", side_effect=lambda explicit=None, name="UKFTractography": ukf_bin if name == "UKFTractography" else dwi_conv),
                mock.patch.object(ukf.subprocess, "run", side_effect=fake_run),
                mock.patch.object(ukf.subprocess, "Popen", side_effect=fake_popen),
                mock.patch.object(ukf, "restore_lossless_payload", return_value={"policy": "mock"}),
                mock.patch("shared.pipeline.build_cli_environment", return_value={}),
                mock.patch.object(ukf, "resolve_adaptive_cpu_num_threads", return_value=4),
                mock.patch.object(ukf, "_host_ram_gib", return_value=16.0),
            ):
                os.environ.pop(ep.ARM64_EXEC_DISABLE_ENV, None)
                out_dir = root / "ukf_out"
                res = ukf.run_ukf(raw, out_dir, device="cpu", log=logs.append)

            self.assertEqual(res["status"], "completed")
            self.assertEqual(len(popen_cmds), 2)
            # Both DWIConvert and UKFTractography were wrapped with /usr/bin/arch -arm64!
            self.assertEqual(popen_cmds[0][:3], [str(arch), "-arm64", str(dwi_conv)])
            self.assertEqual(popen_cmds[1][:3], [str(arch), "-arm64", str(ukf_bin)])
            # Execution command received adaptive --numThreads 4 while cache signature kept -1!
            self.assertEqual(popen_cmds[1][popen_cmds[1].index("--numThreads") + 1], "4")
            manifest = json.loads(ukf.paths("sub01", out_dir)["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["signature"], expected_sig)
            self.assertEqual(manifest["signature"]["parameters"]["numThreads"], -1)
            self.assertEqual(manifest["effective_cpu_num_threads"], 4)
            self.assertTrue(any("Adaptive CPU thread cap: --numThreads 4" in line for line in logs))


@unittest.skipUnless(HAS_TORCH, "Requires PyTorch")
class TractCloudAndPnlCrossPlatformTests(unittest.TestCase):
    def test_tractcloud_mps_adaptive_batching_oom_retry_and_cpu_flush_denormal(self) -> None:
        import torch
        from tractography import tractcloud_accel as ta

        # Create minimal fake tractcloud package modules so install_tractcloud_accelerations works in isolation.
        fake_pkg = types.ModuleType("tractcloud")
        fake_inf = types.ModuleType("tractcloud.inference")
        fake_vtk = types.ModuleType("tractcloud.vtk_io")
        fake_pipe = types.ModuleType("tractcloud.pipeline")
        fake_pipe.__file__ = str(REPO_ROOT / "tractography" / "tractcloud_accel.py")

        def _fake_dist(a, b, num_points=15):
            # a: (N, 3, P), b: (M, 3, P) -> (N, M)
            diff = a.unsqueeze(1) - b.unsqueeze(0)
            return (diff * diff).sum(dim=(2, 3))

        fake_inf._fiber_distance_efficient = _fake_dist
        fake_inf.run_inference = lambda *a, **kw: []
        fake_inf.extract_ras_features = lambda *a, **kw: None
        fake_inf.RealDataDataset = object
        fake_vtk.extract_fibers = lambda *a, **kw: None
        fake_vtk.write_polydata = lambda *a, **kw: None

        with mock.patch.dict(
            sys.modules,
            {
                "tractcloud": fake_pkg,
                "tractcloud.inference": fake_inf,
                "tractcloud.vtk_io": fake_vtk,
                "tractcloud.pipeline": fake_pipe,
            },
        ):
            ta.install_tractcloud_accelerations(fake_pipe, resolved_device="cpu")
            ds_cls = fake_inf.RealDataDataset
            run_inf = fake_inf.run_inference

        rng = np.random.default_rng(123)
        feat = rng.standard_normal((260, 15, 3)).astype(np.float32)
        ds = ds_cls(feat, k=4, k_global=8, k_ds_rate=0.5)
        loader = types.SimpleNamespace(dataset=ds, batch_size=2048)

        # 1. CPU path invokes enable_cpu_flush_denormal(torch) and produces deterministic predictions.
        class _CpuModel:
            def __call__(self, points, info):
                b = points.shape[0]
                logits = torch.zeros((b, 5), dtype=torch.float32)
                logits[:, 2] = 1.0
                return logits

        with mock.patch("shared.hardware_adapt.enable_cpu_flush_denormal", return_value=True) as flush_mock:
            preds_cpu = run_inf(_CpuModel(), loader, ds.global_feat, 5, torch.device("cpu"))
        flush_mock.assert_called_once_with(torch)
        self.assertEqual(len(preds_cpu), 260)
        self.assertTrue(all(p == 2 for p in preds_cpu))

        # 2. MPS path uses adaptive_gpu_scaled_batch, retries on Metal OOM by halving batch,
        #    and calls empty_gpu_cache_safely on OOM + finally.
        seen_batches: list[int] = []
        real_to = torch.Tensor.to
        real_as_tensor = torch.as_tensor
        mps_dev = torch.device("mps")

        def _fake_to(self_t, *args, **kwargs):
            if args and getattr(args[0], "type", None) == "mps":
                args = (torch.device("cpu"), *args[1:])
            elif getattr(kwargs.get("device"), "type", None) == "mps":
                kwargs["device"] = torch.device("cpu")
            return real_to(self_t, *args, **kwargs)

        def _fake_as_tensor(data, *args, **kwargs):
            if getattr(kwargs.get("device"), "type", None) == "mps":
                kwargs["device"] = torch.device("cpu")
            return real_as_tensor(data, *args, **kwargs)

        class _FlakyMpsModel:
            def __call__(self, points, info):
                bs = int(points.shape[0])
                seen_batches.append(bs)
                if bs > 128:
                    raise RuntimeError("MPS backend out of memory (MPS allocated: 7.90 GB, max allowed: 8.00 GB)")
                logits = torch.zeros((bs, 4), dtype=torch.float32)
                logits[:, 1] = 1.0
                return logits

        with (
            mock.patch.object(torch.Tensor, "to", _fake_to),
            mock.patch.object(torch, "as_tensor", _fake_as_tensor),
            mock.patch("shared.hardware_adapt.adaptive_gpu_scaled_batch", return_value=256) as adapt_mock,
            mock.patch("shared.hardware_adapt.empty_gpu_cache_safely") as cache_mock,
        ):
            preds_mps = run_inf(_FlakyMpsModel(), loader, ds.global_feat, 4, mps_dev)

        adapt_mock.assert_called_once()
        self.assertIn(256, seen_batches)
        self.assertIn(128, seen_batches)
        self.assertGreaterEqual(cache_mock.call_count, 2)  # OOM retry + finally cleanup
        self.assertEqual(len(preds_mps), 260)
        self.assertTrue(all(p == 1 for p in preds_mps))

    def test_pnl_native_mps_oom_retry_cpu_flush_denormal_and_apply_threads_cap(self) -> None:
        import torch
        from shared import pnl_mask_native

        fake_inf = types.SimpleNamespace(
            VIEWS=("sagittal", "coronal", "axial"),
            orient=lambda vol, view: np.swapaxes(vol, 0, ("sagittal", "coronal", "axial").index(view)),
            aggregate=lambda probs: (probs["axial"], (probs["axial"] > 0.45).astype(np.float32)),
        )
        vol = np.zeros((256, 256, 256), dtype=np.float32)
        vol[100:110, 100:110, 100:110] = 0.5

        # 1. CPU path calls enable_cpu_flush_denormal(torch).
        orig_zeros = torch.zeros

        class _OkModel:
            def load_state_dict(self, _st, strict=True):
                pass

            def __call__(self, t_in, segmentation_only=True):
                return orig_zeros((int(t_in.shape[0]), 1, 256, 256), dtype=torch.float32)

        with mock.patch("shared.hardware_adapt.enable_cpu_flush_denormal", return_value=True) as flush_mock:
            mask_cpu = pnl_mask_native._fast_vendored_predict(
                vol,
                "cpu",
                8,
                ({"sagittal": {}, "coronal": {}, "axial": {}}, _OkModel()),
                fake_inf,
            )
        flush_mock.assert_called_once_with(torch)
        self.assertEqual(mask_cpu.shape, (256, 256, 256))

        # 2. MPS path uses adaptive_gpu_scaled_batch, retries on MPS OOM, calls empty_gpu_cache_safely.
        call_batches: list[int] = []
        orig_to = torch.Tensor.to

        def _fake_zeros(*args, **kwargs):
            if isinstance(kwargs.get("device"), str) and kwargs["device"].startswith("mps"):
                kwargs["device"] = "cpu"
            return orig_zeros(*args, **kwargs)

        def _fake_to(self_t, *args, **kwargs):
            if args and isinstance(args[0], str) and args[0].startswith("mps"):
                args = ("cpu", *args[1:])
            elif isinstance(kwargs.get("device"), str) and kwargs["device"].startswith("mps"):
                kwargs["device"] = "cpu"
            return orig_to(self_t, *args, **kwargs)

        class _FlakyMpsPnlModel:
            def load_state_dict(self, _st, strict=True):
                pass

            def __call__(self, t_in, segmentation_only=True):
                bs = int(t_in.shape[0])
                call_batches.append(bs)
                if bs > 4:
                    raise RuntimeError("MPS backend out of memory (max allowed: 8.00 GB)")
                return orig_zeros((bs, 1, 256, 256), dtype=torch.float32)

        with (
            mock.patch.object(torch, "zeros", _fake_zeros),
            mock.patch.object(torch.Tensor, "to", _fake_to),
            mock.patch("shared.hardware_adapt.adaptive_gpu_scaled_batch", return_value=8) as mps_adapt,
            mock.patch("shared.hardware_adapt.empty_gpu_cache_safely") as mps_cache,
        ):
            mask_mps = pnl_mask_native._fast_vendored_predict(
                vol,
                "mps",
                16,
                ({"sagittal": {}, "coronal": {}, "axial": {}}, _FlakyMpsPnlModel()),
                fake_inf,
            )
        mps_adapt.assert_called_once()
        self.assertIn(8, call_batches)
        self.assertIn(4, call_batches)
        self.assertGreaterEqual(mps_cache.call_count, 2)
        self.assertEqual(mask_mps.shape, (256, 256, 256))

        # 3. apply_threads caps torch threads (32 on >=64-CPU servers, 16 otherwise)
        #    and SimpleITK threads at 16 while keeping <=16 exact.
        fake_sitk = types.SimpleNamespace(
            ProcessObject=types.SimpleNamespace(SetGlobalDefaultNumberOfThreads=mock.Mock())
        )
        with (
            mock.patch.dict(sys.modules, {"SimpleITK": fake_sitk}),
            mock.patch.object(torch, "set_num_threads") as set_torch_threads,
        ):
            self.assertEqual(pnl_mask_native.apply_threads(4), 4)
            set_torch_threads.assert_called_with(4)
            fake_sitk.ProcessObject.SetGlobalDefaultNumberOfThreads.assert_called_with(4)

            self.assertEqual(pnl_mask_native.apply_threads(24), 24)
            set_torch_threads.assert_called_with(16)
            fake_sitk.ProcessObject.SetGlobalDefaultNumberOfThreads.assert_called_with(16)

            self.assertEqual(pnl_mask_native.apply_threads(64), 64)
            set_torch_threads.assert_called_with(32)
            fake_sitk.ProcessObject.SetGlobalDefaultNumberOfThreads.assert_called_with(16)


class MppcaAndAuditAdaptationTests(unittest.TestCase):
    def test_mppca_low_ram_scaling_vs_high_ram_unchanged(self) -> None:
        from preprocessing import dipy_accel as da
        from shared.hardware_adapt import GIB, MIB

        # Batch byte budgets: <6 GiB -> 64 MiB, <10 GiB -> 96 MiB, >=10 GiB -> full 192 MiB.
        self.assertEqual(da._mppca_cpu_batch_bytes(4 * GIB), 64 * MIB)
        self.assertEqual(da._mppca_cpu_batch_bytes(8 * GIB), 96 * MIB)
        self.assertEqual(da._mppca_cpu_batch_bytes(16 * GIB), da.MPPCA_CPU_BATCH_BYTES)
        self.assertEqual(da._mppca_cpu_batch_bytes(128 * GIB), da.MPPCA_CPU_BATCH_BYTES)

        # Patch batch counts scale down proportionally at low RAM and preserve exact default values at >=16 GiB.
        b_full = da._mppca_cpu_patch_batch(288, 125, batch_bytes=da._mppca_cpu_batch_bytes(16 * GIB))
        b_mid = da._mppca_cpu_patch_batch(288, 125, batch_bytes=da._mppca_cpu_batch_bytes(8 * GIB))
        b_low = da._mppca_cpu_patch_batch(288, 125, batch_bytes=da._mppca_cpu_batch_bytes(4 * GIB))
        self.assertEqual(b_full, 116)
        self.assertEqual(b_mid, 58)
        self.assertEqual(b_low, 38)
        self.assertLess(b_low, b_mid)
        self.assertLess(b_mid, b_full)

        # Worker counts: low-RAM capped (<=4 at <6 GiB, <=6 at <10 GiB), high-RAM unchanged, explicit override preserved.
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SLICERDSPACE_STAGE_CPUS", None)
            with (
                mock.patch.object(da, "is_apple_silicon_host", return_value=False),
                mock.patch.object(da, "effective_cpu_threads", return_value=8),
            ):
                self.assertEqual(da._resolve_dipy_cpu_workers(None, avail_ram_bytes=4 * GIB, apply_low_ram_cap=True), 4)
                self.assertEqual(da._resolve_dipy_cpu_workers(None, avail_ram_bytes=8 * GIB, apply_low_ram_cap=True), 6)
                self.assertEqual(da._resolve_dipy_cpu_workers(None, avail_ram_bytes=20 * GIB, apply_low_ram_cap=True), 8)
                # Explicit max_workers is never reduced.
                self.assertEqual(da._resolve_dipy_cpu_workers(8, avail_ram_bytes=4 * GIB, apply_low_ram_cap=True), 8)

            # Apple Silicon E-core cap when max_workers is None and SLICERDSPACE_STAGE_CPUS is unset.
            with (
                mock.patch.object(da, "is_apple_silicon_host", return_value=True),
                mock.patch.object(da, "optimal_cpu_torch_threads", return_value=6) as opt_mock,
            ):
                self.assertEqual(da._resolve_dipy_cpu_workers(None, avail_ram_bytes=20 * GIB, apply_low_ram_cap=True), 6)
                self.assertEqual(da._resolve_dipy_cpu_workers(None, avail_ram_bytes=20 * GIB, apply_low_ram_cap=False), 6)
                self.assertEqual(opt_mock.call_count, 2)

        # End-to-end _mppca_cpu exercises tight-RAM patch batch scaling without altering numerical results.
        seen_low: list[int] = []
        seen_high: list[int] = []
        orig_solver = da._solve_mppca_batch_numpy

        def _spy(bucket):
            def _wrapped(patches, **kwargs):
                bucket.append(int(patches.shape[0]))
                return orig_solver(patches, **kwargs)
            return _wrapped

        rng = np.random.default_rng(19)
        vol = (100.0 + 10.0 * rng.standard_normal((24, 30, 5, 288))).astype(np.float32)
        with (
            mock.patch.object(da, "query_host_available_ram_bytes", return_value=4 * GIB),
            mock.patch.object(da, "_solve_mppca_batch_numpy", side_effect=_spy(seen_low)),
        ):
            out_low = da.mppca_accelerated(vol, patch_radius=2, device="cpu", max_workers=1)
        with (
            mock.patch.object(da, "query_host_available_ram_bytes", return_value=32 * GIB),
            mock.patch.object(da, "_solve_mppca_batch_numpy", side_effect=_spy(seen_high)),
        ):
            out_high = da.mppca_accelerated(vol, patch_radius=2, device="cpu", max_workers=1)

        self.assertLess(max(seen_low), max(seen_high))
        np.testing.assert_allclose(out_low, out_high, rtol=1e-5, atol=1e-4)

    def test_mppca_threadpool_limits_scoped_once_outside_worker_pool_and_fsl_openblas_capped(self) -> None:
        from contextlib import contextmanager
        from preprocessing import dipy_accel as da, fsl_runtime, pipeline as preproc_pipeline
        from shared import external_process as ep

        # Verify _mppca_cpu enters threadpool_limits(limits=1) exactly ONCE for multi-worker execution,
        # never concurrently inside worker threads.
        entries: list[int] = []

        @contextmanager
        def _fake_limits(limits=None, **_kw):
            entries.append(int(limits))
            yield

        fake_tpctl = types.SimpleNamespace(threadpool_limits=_fake_limits)
        rng = np.random.default_rng(7)
        vol = (80.0 + 5.0 * rng.standard_normal((12, 12, 10, 12))).astype(np.float32)
        with mock.patch.dict(sys.modules, {"threadpoolctl": fake_tpctl}):
            out = da.mppca_accelerated(vol, patch_radius=1, device="cpu", max_workers=4)
        self.assertEqual(out.shape, vol.shape)
        self.assertEqual(entries, [1])

        # Verify server (>=32 CPUs) scales default workers/eddy threads to 16 while <=24 CPUs stays at 8.
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SLICERDSPACE_STAGE_CPUS", None)
            with (
                mock.patch.object(da, "is_apple_silicon_host", return_value=False),
                mock.patch("os.cpu_count", return_value=144),
            ):
                self.assertEqual(da._resolve_dipy_cpu_workers(None, avail_ram_bytes=128 * 1024**3), 16)
            with (
                mock.patch.object(da, "is_apple_silicon_host", return_value=False),
                mock.patch("os.cpu_count", return_value=24),
            ):
                self.assertEqual(da._resolve_dipy_cpu_workers(None, avail_ram_bytes=32 * 1024**3), 8)
            with mock.patch("os.process_cpu_count", return_value=144, create=True):
                self.assertEqual(preproc_pipeline._default_eddy_threads(), 16)
            with mock.patch("os.process_cpu_count", return_value=24, create=True):
                self.assertEqual(preproc_pipeline._default_eddy_threads(), 8)

        # Verify FSL environments default OpenBLAS/MKL/BLIS to 1 thread so topup/eddy don't oversubscribe.
        with tempfile.TemporaryDirectory() as td:
            fsl_bin = Path(td) / "fsl" / "bin"
            fsl_bin.mkdir(parents=True)
            env_managed = ep.managed_tool_environment(fsl_bin_dir=fsl_bin, system="linux")
            self.assertEqual(env_managed["OPENBLAS_NUM_THREADS"], "1")
            self.assertEqual(env_managed["MKL_NUM_THREADS"], "1")
            env_native = fsl_runtime.native_fsl_environment({}, fsl_bin.parent)
            self.assertEqual(env_native["OPENBLAS_NUM_THREADS"], "1")
            self.assertEqual(env_native["MKL_NUM_THREADS"], "1")

    def test_csd_macos_mif_transpose_skips_when_ram_is_too_tight(self) -> None:
        from tractography import csd

        with tempfile.TemporaryDirectory() as td:
            mif = Path(td) / "dwi.mif"
            # Build a tiny valid Float32LE .mif with volume-outermost layout (+0,+1,+2,+3)
            # and verify low available RAM cleanly skips in-RAM transpose without touching the file.
            hdr = (
                "mrtrix image\n"
                "dim: 2,2,2,4\n"
                "layout: +0,+1,+2,+3\n"
                "datatype: Float32LE\n"
                "file: . 128\n"
                "END\n"
            ).encode("utf-8")
            pad = b"\x00" * (128 - len(hdr))
            payload = np.arange(32, dtype="<f4").tobytes()
            original = hdr + pad + payload
            mif.write_bytes(original)

            with (
                mock.patch.object(csd.sys, "platform", "darwin"),
                mock.patch("shared.hardware_adapt.query_host_available_ram_bytes", return_value=64),
            ):
                csd._optimize_dwi_mif_volume_strides_in_ram(mif)
            self.assertEqual(mif.read_bytes(), original)

    def test_wma_tckconvert_uses_native_arm64_argv_under_rosetta(self) -> None:
        from shared import external_process as ep
        from tractography import wma

        with tempfile.TemporaryDirectory() as td:
            root = Path(td).resolve()
            arch = root / "arch"
            arch.write_text("#!/bin/sh\n", encoding="utf-8")
            mrtrix_bin = root / "mrtrix" / "bin"
            mrtrix_bin.mkdir(parents=True)
            tckconvert = mrtrix_bin / "tckconvert"
            tckconvert.write_bytes(_fat_macho_header((0x01000007, 0x0100000C)))
            tckconvert.chmod(0o755)

            with (
                mock.patch.object(ep, "_ARCH", str(arch)),
                mock.patch.object(ep.sys, "platform", "darwin"),
                mock.patch.object(ep, "process_is_translated", return_value=True),
                mock.patch.dict(os.environ, {}, clear=False),
            ):
                os.environ.pop(ep.ARM64_EXEC_DISABLE_ENV, None)
                wrapped = wma.native_arm64_argv((tckconvert.as_posix(), "in.tck", "out.vtk", "-force"), {})
            self.assertEqual(wrapped[:3], [str(arch), "-arm64", str(tckconvert)])


if __name__ == "__main__":
    unittest.main()
