#!/usr/bin/env python3
"""Tests for shared/process_group.py's launch/kill helpers."""

from __future__ import annotations

import os
import signal
import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.process_group import capture_process_group_id, finish_process_group, grouped_launch_command, kill_process_tree, process_group_is_running


class TeardownProcessGroupTest(unittest.TestCase):
    def test_does_not_signal_an_exited_group(self):
        with mock.patch("shared.process_group.process_group_is_running", return_value=False), mock.patch(
                "shared.process_group.kill_process_tree") as kill:
            finish_process_group(12345, grace_seconds=0)
        kill.assert_not_called()

    def test_kills_survivors_using_captured_group_not_exited_parent(self):
        with mock.patch("shared.process_group.process_group_is_running", return_value=True), mock.patch(
                "shared.process_group.kill_process_tree") as kill:
            finish_process_group(12345, grace_seconds=0)
        kill.assert_called_once_with(0, grouped=True, force=True, pgid=12345)

    def test_rejects_invalid_group_id(self):
        self.assertFalse(process_group_is_running(0))
        self.assertFalse(process_group_is_running(-1))


class GroupedLaunchCommandTest(unittest.TestCase):
    def test_prefixes_with_setsid_when_available_on_posix(self) -> None:
        with mock.patch("shared.process_group.os.name", "posix"), mock.patch(
            "shared.process_group.shutil.which", return_value="/usr/bin/setsid"
        ):
            program, args, grouped = grouped_launch_command("/usr/bin/python3", ["-m", "pipeline"])

        self.assertEqual(program, "/usr/bin/setsid")
        self.assertEqual(args, ["/usr/bin/python3", "-m", "pipeline"])
        self.assertTrue(grouped)

    def test_falls_back_to_perl_setpgrp_when_setsid_is_unavailable(self) -> None:
        # This is the macOS case: util-linux's setsid(1) is not shipped, so
        # shutil.which("setsid") is always None there.
        with mock.patch("shared.process_group.os.name", "posix"), mock.patch(
            "shared.process_group.shutil.which", return_value=None
        ), mock.patch("shared.process_group.os.path.exists", return_value=True):
            program, args, grouped = grouped_launch_command("/usr/bin/python3", ["-m", "pipeline"])

        self.assertEqual(program, "/usr/bin/perl")
        self.assertEqual(
            args,
            [
                "-e",
                'setpgrp(0, 0); exec { $ARGV[0] } @ARGV or die "exec failed: $!\\n"',
                "--",
                "/usr/bin/python3",
                "-m",
                "pipeline",
            ],
        )
        self.assertTrue(grouped)

    def test_falls_back_to_perl_on_path_when_not_at_canonical_location(self) -> None:
        def fake_which(name: str):
            return None if name == "setsid" else "/opt/local/bin/perl"

        with mock.patch("shared.process_group.os.name", "posix"), mock.patch(
            "shared.process_group.shutil.which", side_effect=fake_which
        ), mock.patch("shared.process_group.os.path.exists", return_value=False):
            program, args, grouped = grouped_launch_command("/usr/bin/python3", ["-m", "pipeline"])

        self.assertEqual(program, "/opt/local/bin/perl")
        self.assertTrue(grouped)

    def test_leaves_command_unchanged_when_neither_setsid_nor_perl_available(self) -> None:
        with mock.patch("shared.process_group.os.name", "posix"), mock.patch(
            "shared.process_group.shutil.which", return_value=None
        ), mock.patch("shared.process_group.os.path.exists", return_value=False), mock.patch(
            "shared.process_group._python_setsid_launcher", return_value=None  # H9: third fallback also absent
        ):
            program, args, grouped = grouped_launch_command("/usr/bin/python3", ["-m", "pipeline"])

        self.assertEqual(program, "/usr/bin/python3")
        self.assertEqual(args, ["-m", "pipeline"])
        self.assertFalse(grouped)

    def test_leaves_command_unchanged_on_windows(self) -> None:
        with mock.patch("shared.process_group.os.name", "nt"):
            program, args, grouped = grouped_launch_command("C:/Python/python.exe", ["-m", "pipeline"])

        self.assertEqual(program, "C:/Python/python.exe")
        self.assertEqual(args, ["-m", "pipeline"])
        self.assertFalse(grouped)


@unittest.skipIf(os.name != "posix", "requires POSIX process-group APIs")
class CaptureProcessGroupIdTest(unittest.TestCase):
    def test_returns_the_pid_itself_without_calling_getpgid(self) -> None:
        # Must NOT call os.getpgid(): waitForStarted() returns as soon as
        # exec of setsid/perl succeeds, before that process has actually
        # called setsid()/setpgrp(0, 0) to become its own group leader.
        # Querying os.getpgid(pid) in that window can observe the
        # *inherited* pgid (e.g. Slicer's own) instead, and a later forced
        # killpg() against that stale value would signal the wrong group.
        # A mocked os.getpgid returning a different value must not affect
        # the result: it must not be consulted at all.
        with mock.patch("shared.process_group.os.getpgid", return_value=555 + 1000) as mock_getpgid:
            result = capture_process_group_id(555)
        mock_getpgid.assert_not_called()
        self.assertEqual(result, 555)

    def test_returns_none_for_a_non_positive_pid(self) -> None:
        self.assertIsNone(capture_process_group_id(0))
        self.assertIsNone(capture_process_group_id(-1))
        self.assertIsNone(capture_process_group_id(None))


class KillProcessTreeTest(unittest.TestCase):
    def test_windows_uses_taskkill_with_tree_flag(self) -> None:
        with mock.patch("shared.process_group.subprocess.run") as mock_run:
            kill_process_tree(4242, grouped=False, force=False, platform_name="Windows")
        mock_run.assert_called_once()
        command = mock_run.call_args.args[0]
        self.assertEqual(command, ["taskkill", "/PID", "4242", "/T"])

    def test_windows_force_adds_f_flag(self) -> None:
        with mock.patch("shared.process_group.subprocess.run") as mock_run:
            kill_process_tree(4242, grouped=False, force=True, platform_name="Windows")
        command = mock_run.call_args.args[0]
        self.assertEqual(command, ["taskkill", "/PID", "4242", "/T", "/F"])

    @unittest.skipIf(os.name != "posix", "requires POSIX process-group APIs")
    def test_posix_grouped_uses_killpg_against_pid_when_no_pgid_is_given(self) -> None:
        with mock.patch("shared.process_group.os.killpg") as mock_killpg:
            kill_process_tree(555, grouped=True, force=False, platform_name="Linux")
        mock_killpg.assert_called_once_with(555, signal.SIGTERM)

    @unittest.skipIf(os.name != "posix", "requires POSIX process-group APIs")
    def test_posix_grouped_force_sends_sigkill(self) -> None:
        with mock.patch("shared.process_group.os.killpg") as mock_killpg:
            kill_process_tree(555, grouped=True, force=True, platform_name="Linux")
        mock_killpg.assert_called_once_with(555, signal.SIGKILL)

    @unittest.skipIf(os.name != "posix", "requires POSIX process-group APIs")
    def test_posix_grouped_uses_explicit_pgid_over_pid(self) -> None:
        with mock.patch("shared.process_group.os.killpg") as mock_killpg:
            kill_process_tree(555, grouped=True, force=False, pgid=777, platform_name="Linux")
        mock_killpg.assert_called_once_with(777, signal.SIGTERM)

    @unittest.skipIf(os.name != "posix", "requires POSIX process-group APIs")
    def test_posix_grouped_kills_via_pgid_after_leader_pid_is_already_gone(self) -> None:
        # The launcher pid has already been reaped (QProcess now reports 0
        # for it), but the group id captured at launch is still recorded and
        # its descendants (MRtrix, FreeSurfer, ...) may still be running.
        with mock.patch("shared.process_group.os.killpg") as mock_killpg, mock.patch(
            "shared.process_group.os.getpgid"
        ) as mock_getpgid, mock.patch("shared.process_group.os.kill") as mock_kill:
            kill_process_tree(0, grouped=True, force=True, pgid=777, platform_name="Linux")
        mock_killpg.assert_called_once_with(777, signal.SIGKILL)
        mock_getpgid.assert_not_called()
        mock_kill.assert_not_called()

    @unittest.skipIf(os.name != "posix", "requires POSIX process-group APIs")
    def test_posix_grouped_ignores_esrch_from_killpg(self) -> None:
        with mock.patch("shared.process_group.os.killpg", side_effect=ProcessLookupError):
            kill_process_tree(555, grouped=True, force=True, pgid=777, platform_name="Linux")  # must not raise

    @unittest.skipIf(os.name != "posix", "requires POSIX process-group APIs")
    def test_posix_falls_back_to_os_kill_when_not_grouped(self) -> None:
        with mock.patch("shared.process_group.os.kill") as mock_kill:
            kill_process_tree(555, grouped=False, force=False, platform_name="Linux")
        mock_kill.assert_called_once_with(555, signal.SIGTERM)

    @unittest.skipIf(os.name != "posix", "requires POSIX process-group APIs")
    def test_ignores_process_lookup_error(self) -> None:
        with mock.patch("shared.process_group.os.kill", side_effect=ProcessLookupError):
            kill_process_tree(555, grouped=False, force=False, platform_name="Linux")  # must not raise

    def test_non_positive_pid_is_a_no_op(self) -> None:
        with mock.patch("shared.process_group.subprocess.run") as mock_run, mock.patch(
            "shared.process_group.os.kill"
        ) as mock_kill:
            kill_process_tree(0, grouped=False, force=False, platform_name="Windows")
            kill_process_tree(-1, grouped=False, force=False, platform_name="Linux")
        mock_run.assert_not_called()
        mock_kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
