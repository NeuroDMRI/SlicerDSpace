"""C6/D8: universal native binaries run as arm64 under Rosetta; C6/D8b: OOM retry frees the MPS cache too."""

import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared import external_process as ep  # noqa: E402

ARM64 = 0x0100000C
X86_64 = 0x01000007


def fat_header(cputypes, magic=0xCAFEBABE):
    record = 20 if magic == 0xCAFEBABE else 32
    out = struct.pack(">II", magic, len(cputypes))
    for cputype in cputypes:
        out += struct.pack(">II", cputype, 3).ljust(record, b"\0")
    return out + b"\0" * 64


class ArmExecCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="arm64exec-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        self.arch = self.tmp / "arch"
        self.arch.write_text("#!/bin/sh\n")
        for patcher in (
            patch.object(ep, "_ARCH", str(self.arch)),
            patch.object(ep.sys, "platform", "darwin"),
            patch.object(ep, "process_is_translated", return_value=True),
            patch.dict(os.environ, {}, clear=False),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        os.environ.pop(ep.ARM64_EXEC_DISABLE_ENV, None)

    def binary(self, name, data):
        path = self.tmp / name
        path.write_bytes(data)
        path.chmod(0o755)
        return str(path)


class MachOHeaderTests(ArmExecCase):
    def test_fat_with_arm64_slice(self):
        self.assertTrue(ep.macho_has_arm64_slice(self.binary("a", fat_header([X86_64, ARM64]))))
        self.assertTrue(ep.macho_has_arm64_slice(self.binary("b", fat_header([ARM64, X86_64], 0xCAFEBABF))))

    def test_not_arm64_universal(self):
        self.assertFalse(ep.macho_has_arm64_slice(self.binary("x86", fat_header([X86_64, 7]))))
        self.assertFalse(ep.macho_has_arm64_slice(self.binary("thin", struct.pack("<II", 0xFEEDFACF, ARM64) + b"\0" * 64)))
        self.assertFalse(ep.macho_has_arm64_slice(self.binary("script", b"#!/bin/sh\necho hi\n")))
        self.assertFalse(ep.macho_has_arm64_slice(self.binary("java", struct.pack(">II", 0xCAFEBABE, 52) + b"\0" * 64)))
        self.assertFalse(ep.macho_has_arm64_slice(self.binary("short", b"\xca\xfe")))
        self.assertFalse(ep.macho_has_arm64_slice(self.tmp / "missing"))


class NativeArm64ArgvTests(ArmExecCase):
    def test_universal_binary_gets_arch_prefix(self):
        tckgen = self.binary("tckgen", fat_header([X86_64, ARM64]))
        self.assertEqual(ep.native_arm64_argv([tckgen, "-h"]), [str(self.arch), "-arm64", tckgen, "-h"])

    def test_no_prefix_when_not_translated(self):
        tckgen = self.binary("tckgen", fat_header([X86_64, ARM64]))
        with patch.object(ep, "process_is_translated", return_value=False):
            self.assertEqual(ep.native_arm64_argv([tckgen]), [tckgen])

    def test_no_prefix_off_macos(self):
        tckgen = self.binary("tckgen", fat_header([X86_64, ARM64]))
        with patch.object(ep.sys, "platform", "linux"):
            self.assertEqual(ep.native_arm64_argv([tckgen]), [tckgen])

    def test_thin_script_and_slicer_python_stay(self):
        thin = self.binary("fsl_arm", struct.pack("<II", 0xFEEDFACF, ARM64) + b"\0" * 64)
        script = self.binary("dwi2response", b"#!/usr/bin/env python3\n")
        slicer_python = self.binary("PythonSlicer", fat_header([X86_64, ARM64]))
        universal2_python = self.binary("python3.12", fat_header([X86_64, ARM64]))  # an interpreter keeps its arch
        for program in (thin, script, slicer_python, universal2_python, "/nonexistent/tool"):
            self.assertEqual(ep.native_arm64_argv([program, "x"]), [program, "x"])

    def test_kill_switch_in_environment_and_in_env_argument(self):
        tckgen = self.binary("tckgen", fat_header([X86_64, ARM64]))
        with patch.dict(os.environ, {ep.ARM64_EXEC_DISABLE_ENV: "1"}):
            self.assertEqual(ep.native_arm64_argv([tckgen]), [tckgen])
        self.assertEqual(ep.native_arm64_argv([tckgen], {ep.ARM64_EXEC_DISABLE_ENV: "1"}), [tckgen])
        self.assertEqual(ep.native_arm64_argv([tckgen], {ep.ARM64_EXEC_DISABLE_ENV: "0"})[:2], [str(self.arch), "-arm64"])

    def test_already_prefixed_argv_is_not_prefixed_twice(self):
        tckgen = self.binary("tckgen", fat_header([X86_64, ARM64]))
        argv = [str(self.arch), "-arm64", tckgen]
        self.assertEqual(ep.native_arm64_argv(argv), argv)

    def test_dyld_variables_pass_through_arch_e_option(self):
        # /usr/bin/arch is SIP-protected and drops DYLD_* from its own environment.
        tckgen = self.binary("tckgen", fat_header([X86_64, ARM64]))
        argv = ep.native_arm64_argv([tckgen], {"PATH": "/usr/bin", "DYLD_FALLBACK_LIBRARY_PATH": "/opt/lib"})
        self.assertEqual(argv, [str(self.arch), "-arm64", "-e", "DYLD_FALLBACK_LIBRARY_PATH=/opt/lib", tckgen])

    def test_bare_name_resolves_through_env_path(self):
        self.binary("tckgen", fat_header([X86_64, ARM64]))
        argv = ep.native_arm64_argv(["tckgen", "-h"], {"PATH": str(self.tmp)})
        self.assertEqual(argv, [str(self.arch), "-arm64", str(self.tmp / "tckgen"), "-h"])


class CallSiteTests(ArmExecCase):
    def test_csd_invocation_prefixes_native_binaries_but_not_python_scripts(self):
        from tractography.csd import _mrtrix_csd_invocation

        tckgen = self.binary("tckgen", fat_header([X86_64, ARM64]))
        argv, env = _mrtrix_csd_invocation((tckgen, "a"), {"PATH": "/usr/bin"})
        self.assertEqual(argv, (str(self.arch), "-arm64", tckgen, "a"))
        self.assertEqual(env, {"PATH": "/usr/bin"})
        # dwi2response is a Python script of the MRtrix package: no arch prefix on it.
        bin_dir = self.tmp / "pkg" / "bin"
        bin_dir.mkdir(parents=True)
        script = bin_dir / "dwi2response"
        script.write_text("#!/usr/bin/env python3\n")
        argv, _ = _mrtrix_csd_invocation((str(script), "a"), {})
        self.assertNotEqual(argv[0], str(self.arch))

    def test_preprocessing_run_command_prefixes(self):
        from preprocessing import pipeline

        mrconvert = self.binary("mrconvert", fat_header([X86_64, ARM64]))
        seen = {}

        class Done:
            stdout = None

            def wait(self):
                return 0

        def popen(argv, **kwargs):
            seen["argv"] = argv
            return Done()

        with patch.object(pipeline.subprocess, "Popen", popen):
            pipeline._run_command([mrconvert, "in", "out"], env={"PATH": "/usr/bin"}, log=None)
        self.assertEqual(seen["argv"], [str(self.arch), "-arm64", mrconvert, "in", "out"])

    def test_act_prep_run_mrtrix_prefixes(self):
        from tractography import act_prep

        mrconvert = self.binary("mrconvert", fat_header([X86_64, ARM64]))
        seen = {}

        def run(argv, **kwargs):
            seen["argv"] = argv
            return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        with patch.object(act_prep.subprocess, "run", run):
            act_prep._run_mrtrix([mrconvert, "in"], None)
        self.assertEqual(seen["argv"][:3], [str(self.arch), "-arm64", mrconvert])

    def test_saf_run_mrtrix_command_prefixes(self):
        from tractography import saf

        tckstats = self.binary("tckstats", fat_header([X86_64, ARM64]))
        seen = {}

        def run(argv, **kwargs):
            seen["argv"] = argv
            return type("R", (), {"returncode": 0, "stdout": "1", "stderr": ""})()

        with patch.object(saf.subprocess, "run", run):
            saf._run_mrtrix_command([tckstats, "-output", "count"])
        self.assertEqual(seen["argv"][:3], [str(self.arch), "-arm64", tckstats])


class MpsOomCacheReleaseTests(unittest.TestCase):
    """C6/D8b: the chunked OOM retry must free the MPS allocator cache, not only the CUDA one."""

    def test_mps_oom_message_is_recognised(self):
        from shared.hardware_adapt import is_cuda_oom

        self.assertTrue(is_cuda_oom(RuntimeError("MPS backend out of memory (MPS allocated: 9.0 GB)")))

    @unittest.skipUnless(
        __import__("importlib.util").util.find_spec("torch") and __import__("importlib.util").util.find_spec("numpy"),
        "needs numpy and torch",
    )
    def test_gibbs_mps_oom_releases_gpu_cache_before_the_retry(self):
        import numpy as np
        import torch
        from preprocessing import dipy_accel

        events = []
        calls = {"n": 0}
        real_gibbs = dipy_accel._gibbs_1d_batch_torch
        real_device = torch.device

        def flaky_gibbs(x, axis, n_points):
            calls["n"] += 1
            if calls["n"] == 1:
                events.append("oom")
                raise RuntimeError("MPS backend out of memory (MPS allocated: 9.0 GB, max allowed: 9.0 GB)")
            events.append("chunk")
            return real_gibbs(x, axis=axis, n_points=n_points)

        volume = np.random.RandomState(0).rand(8, 8, 40).astype(np.float32)
        with patch.object(dipy_accel, "resolve_dipy_accel_device", return_value="mps"), \
                patch.object(dipy_accel, "adaptive_gpu_scaled_batch", return_value=64), \
                patch.object(dipy_accel, "_gibbs_1d_batch_torch", flaky_gibbs), \
                patch.object(dipy_accel, "empty_gpu_cache_safely", lambda: events.append("release")), \
                patch.object(torch, "device", lambda name: real_device("cpu")):
            dipy_accel.gibbs_removal_accelerated(volume, device="mps")
        # The first chunk fails with an MPS OOM; the cache is released before the next (halved) chunk runs.
        self.assertEqual(events[:3], ["oom", "release", "chunk"])


if __name__ == "__main__":
    unittest.main()
