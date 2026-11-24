"""Round-3 code-review regression tests for Group A:
- Item 1: UKF commit-aware memory guards & --low-memory UKF propagation.
- Item 2: Linux nested cgroup v1/v2 CPU & memory ancestor limit resolution.
- Item 3: GPU slot accounting for NODDI, MK-Curve, Free-Water (and CPU-only DKI).
- Item 4: Preprocessing Eddy/Topup disk preflight, DDParcel commit-aware RAM guard, TractCloud cuda:1 CLI/orchestrator wiring.
- Item 5: Unified timeline telemetry in StageMonitor (RSS/private, page faults, I/O R/W, system commit/swap, stage phases).
"""

from __future__ import annotations

import builtins
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

GIB = 1024 ** 3


class Item1UkfCommitGuardAndLowMemoryTests(unittest.TestCase):
    """[Item 1] Host memory snapshot (incl. Windows commit headroom), UKF guards, and --low-memory UKF wiring."""

    def test_query_host_memory_snapshot_windows_linux_and_env_cap(self) -> None:
        from shared import hardware_adapt as ha

        # 1. Real Windows host query populates physical + commit counters with zero extra deps.
        if os.name == "nt":
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("SLICERDSPACE_MAX_RAM_GIB", None)
                snap = ha.query_host_memory_snapshot()
            self.assertIsNotNone(snap["total_ram_gib"])
            self.assertGreater(snap["total_ram_gib"], 0.0)
            self.assertIsNotNone(snap["available_ram_gib"])
            self.assertIsNotNone(snap["commit_limit_gib"])
            self.assertIsNotNone(snap["commit_total_gib"])
            self.assertIsNotNone(snap["commit_available_gib"])
            self.assertAlmostEqual(
                snap["effective_available_ram_gib"],
                min(snap["available_ram_gib"], snap["commit_available_gib"]),
                places=6,
            )

        # 2. Simulated Windows commit exhaustion: 12 GiB physical free, only 2.5 GiB commit headroom.
        fake_win = {
            "total_phys_bytes": int(32 * GIB),
            "avail_phys_bytes": int(12 * GIB),
            "commit_limit_bytes": int(40 * GIB),
            "commit_total_bytes": int(37.5 * GIB),
            "commit_avail_bytes": int(2.5 * GIB),
        }
        with (
            mock.patch.object(ha.sys, "platform", "win32"),
            mock.patch.object(ha, "_read_windows_memory_counters", return_value=fake_win),
            mock.patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("SLICERDSPACE_MAX_RAM_GIB", None)
            snap = ha.query_host_memory_snapshot()
            self.assertAlmostEqual(snap["total_ram_gib"], 32.0, places=3)
            self.assertAlmostEqual(snap["available_ram_gib"], 12.0, places=3)
            self.assertAlmostEqual(snap["commit_available_gib"], 2.5, places=3)
            self.assertAlmostEqual(snap["effective_available_ram_gib"], 2.5, places=3)

        # 3. Simulated Linux /proc/meminfo + nested cgroup clamping.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            meminfo = root / "meminfo"
            meminfo.write_text(
                "MemTotal:       33554432 kB\n"
                "MemFree:         2097152 kB\n"
                "MemAvailable:   16777216 kB\n"
                "SwapTotal:       8388608 kB\n"
                "SwapFree:        6291456 kB\n",
                encoding="utf-8",
            )
            proc_cg = root / "proc_cgroup"
            proc_cg.write_text("0::/kubepods/pod1/c1\n", encoding="utf-8")
            cg_root = root / "cgroup"
            leaf = cg_root / "kubepods" / "pod1" / "c1"
            leaf.mkdir(parents=True)
            (leaf / "memory.max").write_text(str(int(4 * GIB)), encoding="utf-8")
            (leaf / "memory.current").write_text(str(int(1.5 * GIB)), encoding="utf-8")

            with (
                mock.patch.object(ha.sys, "platform", "linux"),
                mock.patch.dict(os.environ, {}, clear=False),
            ):
                os.environ.pop("SLICERDSPACE_MAX_RAM_GIB", None)
                snap = ha.query_host_memory_snapshot(
                    meminfo_path=meminfo,
                    proc_cgroup_path=proc_cg,
                    cgroup_root=cg_root,
                )
            self.assertAlmostEqual(snap["total_ram_gib"], 4.0, places=3)
            self.assertAlmostEqual(snap["available_ram_gib"], 2.5, places=3)
            self.assertAlmostEqual(snap["swap_total_gib"], 8.0, places=3)
            self.assertAlmostEqual(snap["swap_free_gib"], 6.0, places=3)
            self.assertAlmostEqual(snap["effective_available_ram_gib"], 2.5, places=3)

    def test_ukf_32gb_windows_host_with_exhausted_ram_or_commit_throttles_and_warns(self) -> None:
        from tractography import ukf

        default_params = ukf.parameters()
        heavy = {"volumes": 288, "approximate_shells": [1000.0, 2000.0, 3000.0], "effective_brain_voxels": 950_000}
        standalone_exe = Path("/opt/SlicerDSpace/bundled-runtimes/ukf-windows-x64/bin/UKFTractography.exe")

        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("SLICERDSPACE_COHORT_SCHEDULE", "SLICERDSPACE_STAGE_CPUS")
        }
        with mock.patch.dict(os.environ, clean_env, clear=True):
            # Case A: 32 GiB Windows host where other apps consumed most RAM & commit headroom (mocked snapshot).
            pressured_snap = {
                "total_ram_gib": 32.0,
                "available_ram_gib": 5.0,
                "commit_limit_gib": 40.0,
                "commit_total_gib": 36.0,
                "commit_available_gib": 4.0,
                "swap_total_gib": None,
                "swap_free_gib": None,
                "effective_available_ram_gib": 4.0,
            }
            with mock.patch("shared.hardware_adapt.query_host_memory_snapshot", return_value=pressured_snap):
                threads = ukf.resolve_adaptive_cpu_num_threads(
                    default_params,
                    heavy,
                    use_gpu=False,
                    executable=standalone_exe,
                    cpu_count=16,
                    platform_name="win32",
                )
                warn = ukf._ukf_cpu_memory_runtime_warning(
                    heavy, use_gpu=False, executable=standalone_exe, selected=default_params
                )
            self.assertEqual(threads, 2)
            self.assertIsNotNone(warn)
            self.assertIn("Windows commit headroom", warn)
            self.assertIn("4.0 GiB", warn)
            self.assertIn("5.0 GiB available RAM", warn)
            self.assertIn("VTK", warn)

            # Case B: Physical RAM looks generous (20 GiB free), ONLY Windows commit headroom is tight (3.5 GiB).
            threads_commit = ukf.resolve_adaptive_cpu_num_threads(
                default_params,
                heavy,
                use_gpu=False,
                executable=standalone_exe,
                ram_gib=32.0,
                available_ram_gib=20.0,
                commit_available_gib=3.5,
                cpu_count=16,
                platform_name="win32",
            )
            warn_commit = ukf._ukf_cpu_memory_runtime_warning(
                heavy,
                use_gpu=False,
                executable=standalone_exe,
                selected=default_params,
                ram_gib=32.0,
                available_ram_gib=20.0,
                commit_available_gib=3.5,
            )
            self.assertEqual(threads_commit, 2)
            self.assertIsNotNone(warn_commit)
            self.assertIn("3.5 GiB Windows commit headroom", warn_commit)

            # Case C: Idle 32 GiB Windows host with comfortable available RAM & commit stays fast (-1, no warning).
            threads_idle = ukf.resolve_adaptive_cpu_num_threads(
                default_params,
                heavy,
                use_gpu=False,
                executable=standalone_exe,
                ram_gib=32.0,
                available_ram_gib=24.0,
                commit_available_gib=28.0,
                cpu_count=16,
                platform_name="win32",
            )
            warn_idle = ukf._ukf_cpu_memory_runtime_warning(
                heavy,
                use_gpu=False,
                executable=standalone_exe,
                selected=default_params,
                ram_gib=32.0,
                available_ram_gib=24.0,
                commit_available_gib=28.0,
            )
            self.assertEqual(threads_idle, -1)
            self.assertIsNone(warn_idle)

            # Case D: 64 GiB server under memory pressure STILL throttles & warns; idle 64 GiB server stays -1.
            self.assertEqual(
                ukf.resolve_adaptive_cpu_num_threads(
                    default_params, heavy, use_gpu=False, ram_gib=64.0, available_ram_gib=52.0, cpu_count=64, platform_name="linux"
                ),
                -1,
            )
            self.assertEqual(
                ukf.resolve_adaptive_cpu_num_threads(
                    default_params, heavy, use_gpu=False, ram_gib=64.0, available_ram_gib=4.0, cpu_count=64, platform_name="linux"
                ),
                1,
            )
            self.assertIsNotNone(
                ukf._ukf_cpu_memory_runtime_warning(
                    heavy, use_gpu=False, ram_gib=64.0, available_ram_gib=4.0
                )
            )

    def test_low_memory_cohort_plan_propagates_ukf_caps_and_preserves_user_params(self) -> None:
        from shared.cohort_planning import plan_cohort
        from tractography import ukf

        subjects = [SimpleNamespace(subject_id="sub-0", subject_root=Path("/absent/sub-0"))]
        base_args = ["--targets", "tractography", "--tractography-backend", "ukf"]

        # Without --low-memory, 6 GiB budget rejects HCP-sized UKF (~12 GiB estimate).
        with self.assertRaisesRegex(ValueError, "estimated stage RAM"):
            plan_cohort(subjects, base_args, available=(8, 6 * GIB), allow_low_memory=False)

        # With --low-memory and an existing user UKF parameter, plan clamps UKF to <=2 threads,
        # merges --ukf-parameters cleanly, sets jobs=1, and records explicit warnings + mitigations.
        user_args = [*base_args, "--ukf-parameters", json.dumps({"seedingThreshold": 0.22})]
        plan = plan_cohort(subjects, user_args, available=(8, 6 * GIB), allow_low_memory=True)
        ukf_req = plan["subject_profiles"]["sub-0"]["ukf"]
        self.assertEqual(ukf_req["memory_gb"], 6.0)
        self.assertLessEqual(ukf_req["cpu"], 2)
        self.assertEqual(plan["jobs"], 1)

        sub_args = plan["subject_arguments"]["sub-0"]
        self.assertIn("--ukf-parameters", sub_args)
        merged = json.loads(sub_args[sub_args.index("--ukf-parameters") + 1])
        self.assertEqual(merged["seedingThreshold"], 0.22)
        self.assertEqual(merged["numThreads"], ukf_req["cpu"])
        self.assertEqual(merged["seedsPerVoxel"], 1)

        self.assertTrue(plan["warnings"])
        self.assertTrue(any("UKF estimated RAM" in w and "--ukf-parameters" in w for w in plan["warnings"]))
        self.assertIn("sub-0", plan["low_memory_mitigations"])
        ukf_mit = next(m for m in plan["low_memory_mitigations"]["sub-0"] if m["stage"] == "ukf")
        self.assertEqual(ukf_mit["clamped_cpu"], ukf_req["cpu"])
        self.assertEqual(ukf_mit["applied_ukf_parameters"], merged)

        # Propagated --ukf-parameters must be valid under cohort schedule reservation!
        with mock.patch.dict(
            os.environ,
            {"SLICERDSPACE_COHORT_SCHEDULE": "/unused", "SLICERDSPACE_STAGE_CPUS": str(ukf_req["cpu"])},
        ):
            parsed = ukf.parameters(merged)
        self.assertEqual(parsed.numThreads, ukf_req["cpu"])
        self.assertEqual(parsed.seedingThreshold, 0.22)
        self.assertEqual(parsed.seedsPerVoxel, 1)


class Item2LinuxNestedCgroupTests(unittest.TestCase):
    """[Item 2] Nested Linux cgroup v1/v2 ancestor walk for CPU quota, cpuset, and memory headroom."""

    def test_nested_cgroup_v2_takes_tightest_ancestor_cpu_and_memory(self) -> None:
        from shared.cohort_planning import available_cohort_resources
        from shared.hardware_adapt import linux_cgroup_limits

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            proc_cg = root / "proc_cgroup"
            proc_cg.write_text("0::/kubepods/podX/containerY\n", encoding="utf-8")

            cg_root = root / "cgroup"
            pod = cg_root / "kubepods" / "podX"
            leaf = pod / "containerY"
            leaf.mkdir(parents=True)

            # Root is unconstrained (like Docker/K8s host root)
            (cg_root / "cpu.max").write_text("max 100000\n", encoding="utf-8")
            (cg_root / "memory.max").write_text("max\n", encoding="utf-8")
            # Parent pod constrains memory to 2 GiB (0.5 GiB used -> 1.5 GiB free) and CPU to 4
            (pod / "cpu.max").write_text("400000 100000\n", encoding="utf-8")
            (pod / "memory.max").write_text(str(2 * GIB) + "\n", encoding="utf-8")
            (pod / "memory.current").write_text(str(int(0.5 * GIB)) + "\n", encoding="utf-8")
            # Leaf container constrains CPU to 2, memory to 4 GiB (looser than parent pod!)
            (leaf / "cpu.max").write_text("200000 100000\n", encoding="utf-8")
            (leaf / "memory.max").write_text(str(4 * GIB) + "\n", encoding="utf-8")
            (leaf / "memory.current").write_text(str(int(0.25 * GIB)) + "\n", encoding="utf-8")

            limits = linux_cgroup_limits(proc_cgroup_path=proc_cg, cgroup_root=cg_root)
            self.assertEqual(limits["cpu_limit"], 2)
            self.assertEqual(limits["memory_limit_bytes"], 2 * GIB)
            self.assertEqual(limits["memory_available_bytes"], int(1.5 * GIB))

            # Planner must see 2 CPUs and 1.5 GiB * 0.7 (NOT host 32 CPUs / 22.4 GiB!).
            with mock.patch("shared.compute_schedule.available_resources", return_value=(32, int(32 * GIB * 0.7))):
                cpus, mem = available_cohort_resources(
                    fraction=0.7, proc_cgroup_path=proc_cg, cgroup_root=cg_root
                )
            self.assertEqual(cpus, 2)
            self.assertEqual(mem, int(1.5 * GIB * 0.7))

    def test_nested_cgroup_v2_cpuset_effective_bounds_cpus(self) -> None:
        from shared.hardware_adapt import linux_cgroup_limits

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            proc_cg = root / "proc_cgroup"
            proc_cg.write_text("0::/system.slice/slurmstepd.scope/job_42\n", encoding="utf-8")
            cg_root = root / "cgroup"
            leaf = cg_root / "system.slice" / "slurmstepd.scope" / "job_42"
            leaf.mkdir(parents=True)
            (leaf / "cpu.max").write_text("800000 100000\n", encoding="utf-8")
            (leaf / "cpuset.cpus.effective").write_text("4-5\n", encoding="utf-8")

            limits = linux_cgroup_limits(proc_cgroup_path=proc_cg, cgroup_root=cg_root)
            self.assertEqual(limits["cpu_limit"], 2)

    def test_nested_cgroup_v1_combined_cpu_cpuacct_and_memory_sentinel(self) -> None:
        from shared.cohort_planning import available_cohort_resources
        from shared.hardware_adapt import linux_cgroup_limits

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            proc_cg = root / "proc_cgroup"
            proc_cg.write_text(
                "12:memory:/kubepods/podA/containerB\n"
                "6:cpu,cpuacct:/kubepods/podA/containerB\n"
                "4:cpuset:/kubepods/podA/containerB\n",
                encoding="utf-8",
            )
            cg_root = root / "cgroup"
            cpu_leaf = cg_root / "cpu,cpuacct" / "kubepods" / "podA" / "containerB"
            cpuset_leaf = cg_root / "cpuset" / "kubepods" / "podA" / "containerB"
            mem_parent = cg_root / "memory" / "kubepods" / "podA"
            mem_leaf = mem_parent / "containerB"
            for d in (cpu_leaf, cpuset_leaf, mem_leaf):
                d.mkdir(parents=True)

            # Unlimited cgroup v1 sentinel at parent must be ignored.
            (mem_parent / "memory.limit_in_bytes").write_text("9223372036854771712\n", encoding="utf-8")
            (mem_parent / "memory.usage_in_bytes").write_text("0\n", encoding="utf-8")
            # Tight 2 GiB limit at leaf container.
            (mem_leaf / "memory.limit_in_bytes").write_text(str(2 * GIB) + "\n", encoding="utf-8")
            (mem_leaf / "memory.usage_in_bytes").write_text(str(int(0.25 * GIB)) + "\n", encoding="utf-8")
            # 2 CPUs via cfs_quota_us/cfs_period_us, 3 CPUs via cpuset -> min = 2.
            (cpu_leaf / "cpu.cfs_quota_us").write_text("200000\n", encoding="utf-8")
            (cpu_leaf / "cpu.cfs_period_us").write_text("100000\n", encoding="utf-8")
            (cpuset_leaf / "cpuset.effective_cpus").write_text("0-2\n", encoding="utf-8")

            limits = linux_cgroup_limits(proc_cgroup_path=proc_cg, cgroup_root=cg_root)
            self.assertEqual(limits["cpu_limit"], 2)
            self.assertEqual(limits["memory_limit_bytes"], 2 * GIB)
            self.assertEqual(limits["memory_available_bytes"], int(1.75 * GIB))

            with mock.patch("shared.compute_schedule.available_resources", return_value=(64, int(64 * GIB * 0.7))):
                cpus, mem = available_cohort_resources(
                    fraction=0.7, proc_cgroup_path=proc_cg, cgroup_root=cg_root
                )
            self.assertEqual(cpus, 2)
            self.assertEqual(mem, int(1.75 * GIB * 0.7))


class Item3GpuSlotAccountingTests(unittest.TestCase):
    """[Item 3] Dynamic GPU slot accounting for NODDI, MK-Curve, Free-Water (and CPU-only DKI)."""

    def test_scheduled_stage_accepts_callable_gpu_predicate_safely(self) -> None:
        from shared.cohort_schedule import ENV, create_schedule, scheduled_stage

        with tempfile.TemporaryDirectory() as td:
            folder = create_schedule(Path(td) / "sched", cpu_budget=2, memory_gb=4, stage_cpus=1, stage_memory_gb=1, gpu_slots=1)

            @scheduled_stage("dyn_stage", gpu=lambda cfg: bool(cfg.want_gpu))
            def run_dyn(cfg):
                return "ok"

            @scheduled_stage("broken_pred", gpu=lambda cfg: (_ for _ in ()).throw(RuntimeError("boom")))
            def run_broken(cfg):
                return "still_ran"

            with mock.patch.dict(os.environ, {ENV: str(folder)}):
                self.assertEqual(run_dyn(SimpleNamespace(want_gpu=True, device="cuda")), "ok")
                self.assertEqual(run_dyn(SimpleNamespace(want_gpu=False, device="cuda")), "ok")
                self.assertEqual(run_dyn(SimpleNamespace(want_gpu=True, device="cpu")), "ok")
                self.assertEqual(run_broken(SimpleNamespace(device="auto")), "still_ran")

            started = [
                json.loads(line)
                for line in (folder / "events.jsonl").read_text(encoding="utf-8").splitlines()
                if json.loads(line)["event"] == "started"
            ]
            self.assertEqual([(r["stage"], r["gpu"]) for r in started], [
                ("dyn_stage", 1),
                ("dyn_stage", 0),
                ("dyn_stage", 0),
                ("broken_pred", 0),
            ])

    def test_noddi_mkcurve_freewater_reserve_gpu_slot_on_cuda_and_zero_on_cpu_and_dki(self) -> None:
        from microstructure import dki as dki_mod
        from microstructure import pipeline as mp
        from shared.cohort_schedule import ENV, create_schedule

        fake_cuda_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, is_initialized=lambda: False))
        fake_cpu_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False, is_initialized=lambda: False))

        # 1. Verify stage predicates directly under CUDA available vs unavailable.
        noddi_auto = SimpleNamespace(noddi_device="auto", noddi_directional_average=False)
        noddi_cpu = SimpleNamespace(noddi_device="cpu", noddi_directional_average=False)
        noddi_diravg = SimpleNamespace(noddi_device="cuda", noddi_directional_average=True)

        fw_auto = SimpleNamespace(free_water_fit_method="WLS", free_water_engine="serial", free_water_device="auto")
        fw_cpu = SimpleNamespace(free_water_fit_method="WLS", free_water_engine="serial", free_water_device="cpu")
        fw_nls = SimpleNamespace(free_water_fit_method="NLS", free_water_engine="serial", free_water_device="cuda")

        mkc_cuda = SimpleNamespace(fit_method="MKCURVE", device="auto")
        mkc_cpu = SimpleNamespace(fit_method="MKCURVE", device="cpu")
        mkc_matlab = SimpleNamespace(fit_method="MKCURVE_MATLAB", device="cuda")
        dki_wls = SimpleNamespace(fit_method="WLS", device="cuda")

        with mock.patch("microstructure.noddi_accel._get_torch", return_value=fake_cuda_torch), \
             mock.patch.dict(sys.modules, {"torch": fake_cuda_torch}):
            self.assertTrue(mp._noddi_uses_gpu(config=noddi_auto))
            self.assertFalse(mp._noddi_uses_gpu(config=noddi_cpu))
            self.assertFalse(mp._noddi_uses_gpu(config=noddi_diravg))

            self.assertTrue(mp._freewater_uses_gpu(config=fw_auto))
            self.assertFalse(mp._freewater_uses_gpu(config=fw_cpu))
            self.assertFalse(mp._freewater_uses_gpu(config=fw_nls))

            self.assertTrue(dki_mod._dki_uses_gpu(mkc_cuda))
            self.assertFalse(dki_mod._dki_uses_gpu(mkc_cpu))
            self.assertFalse(dki_mod._dki_uses_gpu(mkc_matlab))
            self.assertFalse(dki_mod._dki_uses_gpu(dki_wls))

        with mock.patch("microstructure.noddi_accel._get_torch", return_value=fake_cpu_torch), \
             mock.patch.dict(sys.modules, {"torch": fake_cpu_torch}):
            self.assertFalse(mp._noddi_uses_gpu(config=noddi_auto))
            self.assertFalse(mp._freewater_uses_gpu(config=fw_auto))
            self.assertFalse(dki_mod._dki_uses_gpu(mkc_cuda))

        # 2. Verify actual @scheduled_stage events.jsonl records gpu=1 for NODDI, MK-Curve, Free-Water on CUDA
        #    and gpu=0 for DKI WLS and CPU runs!
        with tempfile.TemporaryDirectory() as td:
            folder = create_schedule(Path(td) / "sched", cpu_budget=2, memory_gb=4, stage_cpus=1, stage_memory_gb=1, gpu_slots=1)

            with (
                mock.patch.dict(os.environ, {ENV: str(folder)}),
                mock.patch("microstructure.noddi_accel._get_torch", return_value=fake_cuda_torch),
                mock.patch.dict(sys.modules, {"torch": fake_cuda_torch}),
            ):
                # Trigger _compute_noddi_maps through decorator (fast error right after admission when amico=None).
                with mock.patch.object(mp, "amico", None):
                    with self.assertRaises(RuntimeError):
                        mp._compute_noddi_maps(config=noddi_auto, paths={})
                    with self.assertRaises(RuntimeError):
                        mp._compute_noddi_maps(config=noddi_cpu, paths={})

                # Trigger _compute_free_water_maps through decorator.
                with mock.patch.object(mp, "_expected_free_water_manifest", side_effect=RuntimeError("sentinel")):
                    with self.assertRaisesRegex(RuntimeError, "sentinel"):
                        mp._compute_free_water_maps(config=fw_auto, paths={})
                    with self.assertRaisesRegex(RuntimeError, "sentinel"):
                        mp._compute_free_water_maps(config=fw_cpu, paths={})

                # Trigger run_dki through decorator for MKCURVE (CUDA -> gpu=1) and WLS (CPU-only -> gpu=0).
                mkc_cfg = dki_mod.DkiConfig(
                    subject_id="sub01",
                    raw_input_bundle=SimpleNamespace(),
                    output_dir=Path(td) / "dki_mkc",
                    fit_method="MKCURVE",
                    device="auto",
                )
                wls_cfg = dki_mod.DkiConfig(
                    subject_id="sub01",
                    raw_input_bundle=SimpleNamespace(),
                    output_dir=Path(td) / "dki_wls",
                    fit_method="WLS",
                    device="cuda",
                )
                with mock.patch.object(dki_mod, "inspect_dki_inputs", return_value={"eligible": False, "reasons": ["stub"]}):
                    with self.assertRaises(ValueError):
                        dki_mod.run_dki(mkc_cfg)
                    with self.assertRaises(ValueError):
                        dki_mod.run_dki(wls_cfg)

            started = [
                (row["stage"], row["gpu"])
                for line in (folder / "events.jsonl").read_text(encoding="utf-8").splitlines()
                for row in (json.loads(line),)
                if row["event"] == "started"
            ]
            self.assertEqual(
                started,
                [
                    ("microstructure_noddi", 1),
                    ("microstructure_noddi", 0),
                    ("microstructure_freewater", 1),
                    ("microstructure_freewater", 0),
                    ("microstructure_mkcurve", 1),
                    ("microstructure_dki", 0),
                ],
            )

    def test_cohort_planner_marks_gpu_microstructure_stages_consistently(self) -> None:
        from shared.cohort_planning import plan_cohort

        subjects = [SimpleNamespace(subject_id="A", subject_root=Path("/absent/A"))]
        args_auto = [
            "--targets", "microstructure",
            "--microstructure-stages", "dti,freewater,noddi,dki",
            "--dki-fit-method", "MKCURVE",
        ]
        plan_auto = plan_cohort(subjects, args_auto, adaptive=True, available=(16, 128 * GIB))
        prof_auto = plan_auto["subject_profiles"]["A"]
        self.assertEqual(prof_auto["microstructure_noddi"]["gpu"], 1)
        self.assertEqual(prof_auto["microstructure_mkcurve"]["gpu"], 1)
        self.assertEqual(prof_auto["microstructure_freewater"]["gpu"], 1)
        self.assertEqual(prof_auto["preprocessing"]["gpu"], 1)
        self.assertEqual(prof_auto["microstructure_dki"]["gpu"], 0)
        self.assertEqual(prof_auto["microstructure_dti"]["gpu"], 0)

        plan_cpu = plan_cohort(subjects, [*args_auto, "--ddseg-device", "cpu"], adaptive=True, available=(16, 128 * GIB))
        prof_cpu = plan_cpu["subject_profiles"]["A"]
        self.assertEqual(prof_cpu["microstructure_noddi"]["gpu"], 0)
        self.assertEqual(prof_cpu["microstructure_mkcurve"]["gpu"], 0)
        self.assertEqual(prof_cpu["microstructure_freewater"]["gpu"], 0)
        self.assertNotEqual(
            prof_auto["microstructure_noddi"]["profile_key"],
            prof_cpu["microstructure_noddi"]["profile_key"],
        )


class Item4DiskPreflightDdparcelRamAndTractcloudTests(unittest.TestCase):
    """[Item 4] Preprocessing Eddy/Topup disk estimate, DDParcel commit-aware guard, TractCloud cuda:1 wiring."""

    def test_preprocessing_eddy_topup_disk_estimate_accounts_for_uncompressed_4d_scratch(self) -> None:
        from shared.orchestrator import _disk_space_warning, estimate_run_disk_bytes

        hcp_gz_bytes = int(1.3 * GIB)
        hcp_uncompressed_bytes = 145 * 174 * 145 * 288 * 4  # ~4.21 GiB float32 4D

        # Eddy-only with known uncompressed 4D footprint (~3.925 GiB) -> ~19.6 GiB (NOT ~3.9 GiB!).
        eddy_bytes = estimate_run_disk_bytes(
            hcp_gz_bytes,
            ["preprocessing"],
            0,
            uncompressed_dwi_bytes=hcp_uncompressed_bytes,
            preprocessing_eddy=True,
            preprocessing_topup=False,
        )
        self.assertGreaterEqual(eddy_bytes / GIB, 19.0)

        # Eddy + Topup on HCP (2 odd axes: 145, 174, 145) -> includes applytopup + odd-axis .mif padding (~37.9 GiB).
        topup_bytes = estimate_run_disk_bytes(
            hcp_gz_bytes,
            ["preprocessing"],
            0,
            uncompressed_dwi_bytes=hcp_uncompressed_bytes,
            preprocessing_eddy=True,
            preprocessing_topup=True,
            preprocessing_odd_axes=2,
        )
        self.assertGreater(topup_bytes, eddy_bytes)
        self.assertGreaterEqual(topup_bytes / GIB, 35.0)

        # Denoise-only (no Eddy/Topup) stays much smaller than Eddy.
        denoise_bytes = estimate_run_disk_bytes(
            hcp_gz_bytes,
            ["preprocessing"],
            0,
            uncompressed_dwi_bytes=hcp_uncompressed_bytes,
            preprocessing_eddy=False,
            preprocessing_topup=False,
        )
        self.assertLess(denoise_bytes, eddy_bytes)

        # Even without header probe, 1.3 GiB compressed .nii.gz with Eddy estimates ~19.5 GiB.
        fallback_bytes = estimate_run_disk_bytes(hcp_gz_bytes, ["preprocessing"], 0)
        self.assertGreaterEqual(fallback_bytes / GIB, 18.0)

        # _disk_space_warning warns when only 10 GiB is free on output volume for HCP preprocessing!
        with tempfile.TemporaryDirectory() as td:
            fake_dwi = Path(td) / "data.nii.gz"
            fake_dwi.write_bytes(b"0" * 16)
            fake_stat = SimpleNamespace(st_size=hcp_gz_bytes)
            fake_usage = SimpleNamespace(total=100 * GIB, used=90 * GIB, free=10 * GIB)
            with (
                mock.patch.object(Path, "stat", return_value=fake_stat),
                mock.patch("shared.orchestrator._probe_dwi_uncompressed_bytes_and_odd_axes", return_value=(hcp_uncompressed_bytes, 2)),
                mock.patch("shared.orchestrator.shutil.disk_usage", return_value=fake_usage),
            ):
                warn = _disk_space_warning(fake_dwi, ["preprocessing"], 0, Path(td))
            self.assertIn("GB (estimate)", warn)
            self.assertIn("Eddy/Topup", warn)
            self.assertIn("10.0 GB free", warn)
            self.assertIn("not blocked", warn)

    def test_ddparcel_host_memory_preflight_commit_aware_and_strict_block(self) -> None:
        from shared.orchestrator import ProductOrchestratorConfig, _ddparcel_host_memory_preflight, run_subject_products

        self.assertEqual(
            _ddparcel_host_memory_preflight({"available_ram_gib": 12.0, "commit_available_gib": 20.0, "effective_available_ram_gib": 12.0}),
            ("ok", ""),
        )
        lvl_warn, msg_warn = _ddparcel_host_memory_preflight(
            {"available_ram_gib": 4.2, "commit_available_gib": 15.0, "effective_available_ram_gib": 4.2}
        )
        self.assertEqual(lvl_warn, "warn")
        self.assertIn("4.2 GiB available RAM", msg_warn)

        # Plenty of physical RAM, critically exhausted Windows commit headroom!
        crit_snap = {"available_ram_gib": 14.0, "commit_available_gib": 1.6, "effective_available_ram_gib": 1.6}
        lvl_crit, msg_crit = _ddparcel_host_memory_preflight(crit_snap, strict=False)
        self.assertEqual(lvl_crit, "critical")
        self.assertIn("1.6 GiB Windows commit headroom", msg_crit)

        lvl_blk, msg_blk = _ddparcel_host_memory_preflight(crit_snap, strict=True)
        self.assertEqual(lvl_blk, "block")
        self.assertIn("1.6 GiB Windows commit headroom", msg_blk)

        # End-to-end strict fail-fast blocks DDParcel without calling run_ddparcel_from_registered_bundle.
        with tempfile.TemporaryDirectory() as td:
            from test_product_orchestrator import _write_valid_raw_subject

            root = Path(td) / "sub01"
            _write_valid_raw_subject(root)
            atlas = Path(td) / "atlas_t2.nii.gz"
            atlas.write_bytes(b"stub")
            cfg = ProductOrchestratorConfig(
                subject_id="sub01",
                subject_root=root,
                targets=("parcellation",),
                atlas_t2=atlas,
                closure_applied=True,
                execute=True,
                resume=False,
            )
            logs: list[str] = []
            with (
                mock.patch.dict(os.environ, {"SLICERDSPACE_STRICT_MEMORY_PREFLIGHT": "1"}),
                mock.patch("shared.hardware_adapt.query_host_memory_snapshot", return_value=crit_snap),
                mock.patch("anatomy.parcellation.run_ddparcel_from_registered_bundle") as run_parcel,
            ):
                res = run_subject_products(cfg, log=logs.append)
            run_parcel.assert_not_called()
            parcel_task = next(t for t in res.tasks if t.name == "parcellation.ddparcel")
            self.assertEqual(parcel_task.status, "blocked")
            self.assertIn("Windows commit headroom", parcel_task.reason)
            self.assertTrue(any("BLOCKED:" in line for line in logs))

    def test_tractcloud_cuda_index_preserved_in_cli_and_orchestrator(self) -> None:
        from shared.orchestrator import _resolved_tractcloud_device_for_execution
        from shared.run_subject_pipeline_cli import build_parser

        parser = build_parser()
        for val in ("auto", "cpu", "cuda", "cuda:0", "cuda:1", "mps"):
            parsed = parser.parse_args(["--tractcloud-device", val])
            self.assertEqual(parsed.tractcloud_device, val)
        with mock.patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--tractcloud-device", "tpu"])
            with self.assertRaises(SystemExit):
                parser.parse_args(["--tractcloud-device", "cuda:abc"])

        self.assertEqual(
            _resolved_tractcloud_device_for_execution("cuda:1", execute=True, target_will_run=True),
            "cuda:1",
        )


class Item5UnifiedTimelineTelemetryTests(unittest.TestCase):
    """[Item 5] Unified timeline telemetry in StageMonitor (process + system commit/swap + stage phases)."""

    def test_stage_monitor_records_unified_timeline_phases_and_pressure_summary(self) -> None:
        from shared.cohort_profile import StageMonitor

        snapshots = iter([
            {
                "available_ram_gib": 14.0,
                "commit_limit_gib": 40.0,
                "commit_total_gib": 24.0,
                "commit_available_gib": 16.0,
                "swap_total_gib": 8.0,
                "swap_free_gib": 8.0,
                "effective_available_ram_gib": 14.0,
            },
            {
                "available_ram_gib": 3.0,
                "commit_limit_gib": 40.0,
                "commit_total_gib": 38.8,
                "commit_available_gib": 1.2,
                "swap_total_gib": 8.0,
                "swap_free_gib": 7.0,
                "effective_available_ram_gib": 1.2,
            },
            {
                "available_ram_gib": 6.0,
                "commit_limit_gib": 40.0,
                "commit_total_gib": 35.0,
                "commit_available_gib": 5.0,
                "swap_total_gib": 8.0,
                "swap_free_gib": 7.2,
                "effective_available_ram_gib": 5.0,
            },
        ])
        last_snap = {}

        def next_snap():
            nonlocal last_snap
            try:
                last_snap = next(snapshots)
            except StopIteration:
                pass
            return last_snap

        with tempfile.TemporaryDirectory() as td:
            tl_path = Path(td) / "timeline.jsonl"
            mon = StageMonitor(interval=10.0, stage="load", timeline_path=tl_path, memory_snapshot_fn=next_snap)
            mon.sample()
            mon.set_stage("fit")
            mon.sample()
            summary = mon.finish("completed")

            # Legacy keys preserved verbatim
            for key in (
                "outcome", "samples", "execution_seconds", "peak_tree_rss_bytes",
                "sampled_cpu_seconds", "mean_cpu_cores", "sampled_io_bytes", "interval_seconds", "scope",
            ):
                self.assertIn(key, summary)
            self.assertEqual(summary["outcome"], "completed")
            self.assertGreaterEqual(summary["samples"], 3)
            self.assertGreater(summary["peak_tree_rss_bytes"], 0)

            # New summary metrics
            self.assertEqual(summary["stage"], "fit")
            self.assertAlmostEqual(summary["min_system_available_ram_gb"], 3.0, places=3)
            self.assertAlmostEqual(summary["min_system_commit_available_gb"], 1.2, places=3)
            self.assertAlmostEqual(summary["min_system_swap_free_gb"], 7.0, places=3)
            self.assertAlmostEqual(summary["swap_used_delta_gb"], 1.0, places=3)
            self.assertTrue(summary["swap_or_commit_pressure"])
            self.assertEqual([t["stage"] for t in summary["phase_transitions"]], ["load", "fit"])
            self.assertEqual(summary["timeline_path"], str(tl_path))

            # Unified timeline rows in memory & JSONL share the same timestamped schema
            rows = [json.loads(line) for line in tl_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), len(mon.timeline))
            self.assertGreaterEqual(len(rows), 3)
            self.assertEqual(rows[0]["stage"], "load")
            self.assertEqual(rows[1]["stage"], "fit")
            for required in (
                "time", "elapsed_seconds", "stage", "rss_bytes", "rss_gb", "private_bytes",
                "io_read_bytes", "io_write_bytes", "page_faults",
                "system_available_ram_gb", "system_commit_limit_gb",
                "system_commit_total_gb", "system_commit_available_gb",
                "system_swap_total_gb", "system_swap_free_gb",
            ):
                self.assertIn(required, rows[0])
            if os.name == "nt":
                self.assertIsNotNone(rows[0]["private_bytes"])
                self.assertIsNotNone(rows[0]["page_faults"])
                self.assertIsNotNone(rows[0]["io_read_bytes"])
                self.assertIsNotNone(rows[0]["io_write_bytes"])

    def test_linux_proc_counters_parser_and_zero_dependency_fallback(self) -> None:
        from shared.cohort_profile import StageMonitor, _linux_descendant_pids, _linux_process_counters

        # 1. Synthetic Linux /proc/<pid> with spaces/parens in comm name.
        with tempfile.TemporaryDirectory() as td:
            proc_root = Path(td)
            p100 = proc_root / "100"
            (p100 / "task" / "100").mkdir(parents=True)
            (p100 / "task" / "100" / "children").write_text("101 102\n", encoding="utf-8")
            (proc_root / "101" / "task" / "101").mkdir(parents=True)
            (proc_root / "102" / "task" / "102").mkdir(parents=True)

            # Fields after ')': state(0) ppid(1) pgrp(2) sess(3) tty(4) tpgid(5) flags(6)
            # minflt(7) cminflt(8) majflt(9) cmajflt(10) utime(11) stime(12) ... starttime(19) vsize(20) rss(21)
            after_comm = ["S"] + ["0"] * 6 + ["500", "0", "42", "0", "150", "50"] + ["0"] * 6 + ["99999", "0", "1024"]
            (p100 / "stat").write_text("100 (python (worker)) " + " ".join(after_comm) + "\n", encoding="utf-8")
            (p100 / "status").write_text("Name:\tpython\nVmRSS:\t4096 kB\nRssAnon:\t3072 kB\n", encoding="utf-8")
            (p100 / "io").write_text("rchar: 9999\nwchar: 8888\nread_bytes: 123456\nwrite_bytes: 654321\n", encoding="utf-8")

            self.assertEqual(_linux_descendant_pids(100, proc_root=proc_root), [100, 101, 102])
            parsed = _linux_process_counters(100, proc_root=proc_root, page_size=4096, clk_tck=100)
            self.assertEqual(parsed["rss_bytes"], 1024 * 4096)
            self.assertEqual(parsed["private_bytes"], 3072 * 1024)
            self.assertEqual(parsed["page_faults"], 542)
            self.assertEqual(parsed["major_page_faults"], 42)
            self.assertAlmostEqual(parsed["cpu_seconds"], 2.0, places=5)
            self.assertEqual(parsed["io_read_bytes"], 123456)
            self.assertEqual(parsed["io_write_bytes"], 654321)

        # 2. Zero third-party dependency fallback when psutil is unavailable.
        real_import = builtins.__import__

        def _no_psutil(name, *args, **kwargs):
            if name == "psutil":
                raise ImportError("simulated missing psutil")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=_no_psutil):
            mon = StageMonitor(interval=10.0, stage="no_psutil")
            mon.sample()
            summary = mon.finish("completed")
        self.assertGreaterEqual(summary["samples"], 2)
        self.assertGreater(summary["peak_tree_rss_bytes"], 0)
        if os.name == "nt":
            self.assertIsNotNone(summary["peak_tree_private_bytes"])
            self.assertIsNotNone(summary["sampled_page_faults"])
            self.assertIsNotNone(summary["sampled_io_read_bytes"])

    def test_cohort_telemetry_integrates_unified_monitor_without_key_collision(self) -> None:
        from shared.cohort_telemetry import CohortTelemetry

        with tempfile.TemporaryDirectory() as td:
            trace = Path(td) / "resources.jsonl"
            missing_gpu = dict(available=False, owned_vram_mib=None, devices=[])
            with mock.patch("shared.cohort_telemetry.gpu_sample", return_value=missing_gpu):
                telemetry = CohortTelemetry(trace, interval=0.01).start()
                time.sleep(0.04)
                summary = telemetry.finish("completed")
            self.assertEqual(telemetry.errors, [])
            rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(rows)
            for key in ("time", "elapsed_seconds", "rss_bytes", "rss_gb", "system_available_ram_gb", "gpu"):
                self.assertIn(key, rows[0])
            self.assertGreater(summary["peak_tree_rss_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
