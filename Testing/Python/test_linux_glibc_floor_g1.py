"""Finding G1: the Linux glibc 2.34 floor is a non-blocking doctor row and an installer warning."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from shared import environment as env
from test_b4_w4_platform import _InstallerCopies
from test_installer_gui_reinstall_gap import load_setup_module

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
        for fact in ("2.28", "2.34", "mrcrop", "surface"):  # b2-os, decision 4: the reason ...
            self.assertIn(fact, row.details["notes"])
        for fact in ("2.34", "Ubuntu 22.04+", "Debian 12+", "RHEL/Rocky/Alma 9+"):  # ... and the fix are two texts
            self.assertIn(fact, row.install_hint)
        self.assertNotIn(row.install_hint, row.details["notes"])

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
        with patch("os.confstr", create=True, return_value="glibc 2.28"):
            self.assertEqual(env.linux_glibc_version(), (2, 28))
        with patch("os.confstr", create=True, side_effect=ValueError), patch("platform.libc_ver", return_value=("glibc", "2.35")):
            self.assertEqual(env.linux_glibc_version(), (2, 35))
        with patch("os.confstr", create=True, side_effect=ValueError), patch("platform.libc_ver", return_value=("", "")):
            self.assertIsNone(env.linux_glibc_version())

    def test_o_p2_3_detected_source_names_the_real_source(self) -> None:
        def source(**patches):
            rows = env._linux_glibc_floor_statuses(_specs(), (), host_platform="Linux")
            return rows[0].details["detected_source"], rows[0].details["glibc_version"]

        with patch("os.confstr", create=True, return_value="glibc 2.28"):
            self.assertEqual(source(), ("os.confstr CS_GNU_LIBC_VERSION", "2.28"))
        with patch("os.confstr", create=True, side_effect=ValueError), patch("platform.libc_ver", return_value=("glibc", "2.35")):
            self.assertEqual(source(), ("platform.libc_ver", "2.35"))
        with patch("os.confstr", create=True, return_value="glibc 2.28"), patch("platform.libc_ver", return_value=("glibc", "2.35")):
            self.assertEqual(source()[0], "os.confstr CS_GNU_LIBC_VERSION")
        with patch("os.confstr", create=True, side_effect=ValueError), patch("platform.libc_ver", return_value=("", "")):
            text, version = source()
            self.assertIn("unavailable", text)
            self.assertEqual(version, "unknown")
        rows = env._linux_glibc_floor_statuses(_specs(), (), glibc=(2, 28), host_platform="Linux")
        self.assertEqual(rows[0].details["detected_source"], "value passed by the caller")

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
                with patch("os.confstr", create=True, return_value="glibc 2.28"), patch("sys.platform", "linux"):
                    warn()
                text = " ".join(logs)
                for fact in ("[WARN]", "2.28", "2.34", "mrcrop", "surface", "Ubuntu 22.04+", "RHEL/Rocky/Alma 9+"):
                    self.assertIn(fact, text)
                logs.clear()
                with patch("os.confstr", create=True, return_value="glibc 2.35"), patch("sys.platform", "linux"):
                    warn()
                with patch("os.confstr", create=True, side_effect=ValueError), patch("sys.platform", "linux"):
                    warn()
                with patch("os.confstr", create=True, return_value="glibc 2.17"), patch("sys.platform", "darwin"):
                    warn()
                self.assertEqual(logs, [])


class G1SetupExtensionWarningTests(unittest.TestCase):
    """C-P2: the website's default graphical Setup ZIP (buildSetupExtensionFiles) must emit the glibc warning."""

    def _run_download_phase(self, glibc_text: str, platform_name: str = "linux"):
        import tempfile

        ns = load_setup_module()
        self.assertNotIn("_log", ns, "the setup payload has no _log; a bare call would raise NameError")
        events: list = []

        def stop_after_the_warning(*_a, **_k):
            raise InterruptedError()

        ns["_acquire_spec"] = stop_after_the_warning
        ns["_host_bundle_key"] = lambda: "linux_x64_base"
        ns["_detect_system_proxy"] = lambda: ""
        with tempfile.TemporaryDirectory() as tmp:
            ns["_pick_cache_dir"] = lambda *_a: Path(tmp)
            with patch("os.confstr", create=True, return_value=glibc_text), patch("sys.platform", platform_name):
                ns["_run_phase_a_core"]([], "", "", "", lambda kind, payload: events.append((kind, payload)), None)
        return [str(payload) for kind, payload in events if kind == "log"]

    def test_c_p2_graphical_setup_warns_on_old_linux_glibc(self) -> None:
        logs = self._run_download_phase("glibc 2.28")
        text = " ".join(logs)
        for fact in ("[WARN]", "2.28", "2.34", "mrcrop", "Ubuntu 22.04+"):
            self.assertIn(fact, text)

    def test_c_p2_graphical_setup_is_silent_on_new_glibc_and_other_systems(self) -> None:
        for text, platform_name in (("glibc 2.35", "linux"), ("glibc 2.17", "darwin")):
            with self.subTest(text=text, platform=platform_name):
                self.assertFalse([l for l in self._run_download_phase(text, platform_name) if "[WARN]" in l])


if __name__ == "__main__":
    unittest.main()
