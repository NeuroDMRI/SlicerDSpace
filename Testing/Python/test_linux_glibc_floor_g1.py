"""Finding G1: the Linux glibc 2.34 floor is a non-blocking doctor row and an installer warning."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from shared import environment as env
from test_b4_w4_platform import _InstallerCopies

REPO_ROOT = Path(__file__).resolve().parents[2]


def _specs():
    return (env.DependencySpec(key="command:mrcrop", kind="command", label="mrcrop", required=True,
                               targets=("preprocessing",), command="mrcrop", install_group="mrtrix"),
            env.DependencySpec(key="python:numpy", kind="python", label="numpy", required=True,
                               targets=("validation",), install_group="python"))


class G1DoctorRowTests(unittest.TestCase):
    def _row(self, glibc, host="Linux"):
        rows = env._linux_glibc_floor_statuses(_specs(), (), glibc=glibc, host_platform=host)
        return rows[0] if rows else None

    def test_g1_old_glibc_is_a_non_blocking_warning_with_the_facts(self) -> None:
        row = self._row((2, 28))
        self.assertEqual(row.key, "linux_glibc_floor")
        self.assertEqual((row.status, row.required), ("missing", False))
        self.assertEqual(row.install_group, "")  # no install action; an old system cannot be fixed by the plugin
        self.assertEqual(row.targets, ("preprocessing",))
        for fact in ("2.28", "2.34", "mrcrop", "surface", "Ubuntu 22.04+", "Debian 12+", "RHEL/Rocky/Alma 9+"):
            self.assertIn(fact, row.install_hint)

    def test_g1_new_or_unknown_glibc_is_available(self) -> None:
        self.assertEqual(self._row((2, 34)).status, "available")
        self.assertEqual(self._row((2, 39)).status, "available")
        self.assertEqual(self._row(None).status, "available")  # unknown must not warn or block

    def test_g1_row_is_absent_off_linux_and_without_affected_targets(self) -> None:
        self.assertIsNone(self._row((2, 28), host="Darwin"))
        self.assertEqual(env._linux_glibc_floor_statuses((_specs()[1],), (), glibc=(2, 28), host_platform="Linux"), ())

    def test_g1_old_glibc_does_not_block_a_target(self) -> None:
        row = self._row((2, 17))
        blockers = [item for item in (row,) if item.required and item.status == "missing"]
        self.assertEqual(blockers, [])

    def test_g1_version_parse(self) -> None:
        with patch("os.confstr", return_value="glibc 2.28"):
            self.assertEqual(env.linux_glibc_version(), (2, 28))
        with patch("os.confstr", side_effect=ValueError), patch("platform.libc_ver", return_value=("glibc", "2.35")):
            self.assertEqual(env.linux_glibc_version(), (2, 35))
        with patch("os.confstr", side_effect=ValueError), patch("platform.libc_ver", return_value=("", "")):
            self.assertIsNone(env.linux_glibc_version())

    def test_g1_doctor_includes_the_row(self) -> None:
        sentinel = env.DependencyStatus(key="linux_glibc_floor", kind="setting", label="x", status="missing",
                                        required=False, targets=("preprocessing",))
        config = env.EnvironmentDoctorConfig(repo_root=REPO_ROOT, targets=("validation",))
        with patch.object(env, "_linux_glibc_floor_statuses", return_value=(sentinel,)):
            result = env.run_environment_doctor(config)
        self.assertIn("linux_glibc_floor", [item.key for item in result.statuses])


class G1InstallerWarningTests(_InstallerCopies, unittest.TestCase):
    def test_g1_installer_warns_on_old_linux_glibc_and_stays_silent_otherwise(self) -> None:
        for label, ns in self.copies():
            with self.subTest(label):
                logs: list = []
                ns["_log"] = logs.append
                warn = ns["_warn_linux_glibc_floor"]
                with patch("os.confstr", return_value="glibc 2.28"), patch("sys.platform", "linux"):
                    warn()
                text = " ".join(logs)
                for fact in ("[WARN]", "2.28", "2.34", "mrcrop", "surface", "Ubuntu 22.04+", "RHEL/Rocky/Alma 9+"):
                    self.assertIn(fact, text)
                logs.clear()
                with patch("os.confstr", return_value="glibc 2.35"), patch("sys.platform", "linux"):
                    warn()
                with patch("os.confstr", side_effect=ValueError), patch("sys.platform", "linux"):
                    warn()
                with patch("os.confstr", return_value="glibc 2.17"), patch("sys.platform", "darwin"):
                    warn()
                self.assertEqual(logs, [])


if __name__ == "__main__":
    unittest.main()
