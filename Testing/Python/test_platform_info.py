"""B1: Rosetta-aware machine names, with mocked platform and sysctl."""

import subprocess
import unittest
from unittest.mock import patch

from shared import platform_info


def sysctl(values):
    def run(argv, **kwargs):
        name = argv[-1]
        if name not in values:  # an unknown oid makes sysctl fail
            return subprocess.CompletedProcess(argv, 1, "", "unknown oid")
        return subprocess.CompletedProcess(argv, 0, values[name] + "\n", "")
    return run


class PlatformInfoTests(unittest.TestCase):
    def host(self, system, machine, values=None):
        with patch.object(platform_info.platform, "system", return_value=system), \
                patch.object(platform_info.platform, "machine", return_value=machine), \
                patch.object(platform_info.subprocess, "run", side_effect=sysctl(values or {})) as run:
            return platform_info.host_machine(), platform_info.slicer_machine(), run

    def test_rosetta_slicer_on_apple_silicon_reports_arm64_host(self):
        host, slicer, _ = self.host("Darwin", "x86_64", {"sysctl.proc_translated": "1"})
        self.assertEqual((host, slicer), ("arm64", "x86_64"))

    def test_hw_optional_arm64_alone_identifies_apple_silicon(self):
        host, slicer, _ = self.host("Darwin", "x86_64", {"sysctl.proc_translated": "0", "hw.optional.arm64": "1"})
        self.assertEqual((host, slicer), ("arm64", "x86_64"))

    def test_native_arm64_and_intel_mac(self):
        self.assertEqual(self.host("Darwin", "arm64", {"sysctl.proc_translated": "0"})[0], "arm64")
        host, slicer, _ = self.host("Darwin", "x86_64")  # Intel: both oids are unknown or 0
        self.assertEqual((host, slicer), ("x86_64", "x86_64"))

    def test_other_systems_never_call_sysctl(self):
        for system, machine in (("Linux", "x86_64"), ("Windows", "AMD64")):
            host, slicer, run = self.host(system, machine)
            self.assertEqual((host, slicer), (machine, machine))
            run.assert_not_called()

    def test_missing_sysctl_binary_is_not_fatal(self):
        with patch.object(platform_info.platform, "system", return_value="Darwin"), \
                patch.object(platform_info.platform, "machine", return_value="x86_64"), \
                patch.object(platform_info.subprocess, "run", side_effect=OSError("no sysctl")):
            self.assertEqual(platform_info.host_machine(), "x86_64")


if __name__ == "__main__":
    unittest.main()
