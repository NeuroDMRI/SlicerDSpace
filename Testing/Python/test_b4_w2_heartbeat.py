"""W2 / H2: run heartbeat and stall warning (never kills)."""
from __future__ import annotations

import io
import os
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from shared import run_heartbeat as hb


class HeartbeatFormatTests(unittest.TestCase):
    def test_h2_line_round_trips_through_the_parser(self):
        line = hb.format_heartbeat("CSD/ACT run 1", 125.4, 99.6, 3, 2048, 12)
        self.assertEqual(line, "[Heartbeat] stage=CSD/ACT run 1 elapsed=125 cpu_s=100 children=3 log_bytes=2048 stalled_min=12")
        self.assertEqual(hb.parse_heartbeat_line(line), dict(stage="CSD/ACT run 1", elapsed=125, cpu_s=100,
                                                            children=3, log_bytes=2048, stalled_min=12))

    def test_h2_without_psutil_only_elapsed_is_printed(self):
        self.assertEqual(hb.format_heartbeat("WMA", 7), "[Heartbeat] stage=WMA elapsed=7")

    def test_h2_stall_warning_text_and_parser(self):
        line = hb.format_stall_warning("UKF", 10)
        self.assertEqual(line, "[Stall warning] UKF: no CPU progress and no output for 10 min. The step may be hung; "
                               "press Cancel to stop it, or wait if the tool is known to be quiet.")
        self.assertEqual(hb.parse_stall_warning(line), dict(stage="UKF", minutes=10))
        self.assertIsNone(hb.parse_heartbeat_line("something else"))

    def test_h2_env_overrides_interval_and_threshold(self):
        with patch.dict(os.environ, {hb.HEARTBEAT_SECONDS_ENV: "1", hb.STALL_WARN_MINUTES_ENV: "0.5"}):
            beat = hb.Heartbeat()
        self.assertEqual((beat.interval, beat.stall_seconds), (1.0, 30.0))
        with patch.dict(os.environ, {hb.HEARTBEAT_SECONDS_ENV: "junk"}, clear=False):
            self.assertEqual(hb.Heartbeat().interval, 60.0)
        self.assertEqual(hb.Heartbeat().stall_seconds, 600.0)


class HeartbeatStallTests(unittest.TestCase):
    def _beat(self, clock, lines, **kwargs):
        return hb.Heartbeat(interval=60, stall_minutes=10, out=lines.append, clock=clock, **kwargs)

    def test_h2_stage_follows_orchestrator_lines(self):
        beat = hb.Heartbeat(out=lambda line: None)
        beat.observe_log_line("[Orchestrator] Running WMA run1")
        self.assertEqual(beat._stage, "WMA run1")
        beat.observe_log_line("[Orchestrator] DDSeg started")
        self.assertEqual(beat._stage, "DDSeg")
        beat.observe_log_line("[Orchestrator] DDSeg completed in 3.0s")
        self.assertIsNone(beat._stage)

    def test_h2_ukf_tool_line_names_the_stage_and_preproc_commands_do_not(self):
        """UKF has no [Orchestrator] line; its wrapper logs '[UKF] Running UKFTractography'."""
        beat = hb.Heartbeat(out=lambda line: None)
        beat.observe_log_line("[Orchestrator] DDSeg completed in 3.0s")
        beat.observe_log_line("[UKF] Running UKFTractography (GPU device=cuda:0, precision=fp32)")
        self.assertEqual(beat._stage, "UKF")
        beat.observe_log_line("[Preproc] Running: dwifslpreproc a b")
        self.assertEqual(beat._stage, "UKF")

    def test_h2_stall_is_caught_even_when_no_stage_was_ever_named(self):
        now = [0.0]
        lines = []
        beat = self._beat(lambda: now[0], lines)
        beat._cpu.sample = lambda: (5.0, 1)
        for _ in range(10):
            now[0] += 60
            beat.tick()
        self.assertTrue(lines[-1].startswith("[Stall warning] pipeline (between stages): no CPU progress"), lines[-2:])
        self.assertIn("stage=- ", lines[-2])
        self.assertIn("stalled_min=10", lines[-2])

    def test_h2_stalled_stage_warns_once_per_threshold_and_never_kills(self):
        now = [1000.0]
        lines = []
        beat = self._beat(lambda: now[0], lines)
        beat._cpu.sample = lambda: (5.0, 2)  # frozen CPU, two children
        beat.set_stage("UKF")
        for _ in range(9):  # 9 minutes: still below the threshold
            now[0] += 60
            beat.tick()
        self.assertFalse([line for line in lines if line.startswith("[Stall warning]")])
        self.assertFalse([line for line in lines if "stalled_min" in line])
        now[0] += 60  # minute 10
        beat.tick()
        self.assertIn("stalled_min=10", lines[-2])
        self.assertTrue(lines[-1].startswith("[Stall warning] UKF: no CPU progress and no output for 10 min."))
        for _ in range(9):
            now[0] += 60
            beat.tick()
        self.assertEqual(len([line for line in lines if line.startswith("[Stall warning]")]), 1)
        now[0] += 60  # minute 20: the second warning
        beat.tick()
        self.assertEqual(len([line for line in lines if line.startswith("[Stall warning]")]), 2)

    def test_h2_cpu_growth_or_output_resets_the_stall_clock(self):
        now = [0.0]
        cpu = [1.0]
        lines = []
        beat = self._beat(lambda: now[0], lines)
        beat._cpu.sample = lambda: (cpu[0], 1)
        beat.set_stage("eddy")
        for minute in range(30):
            now[0] += 60
            cpu[0] += 10 if minute % 2 else 0  # CPU grows every other minute
            beat.tick()
        self.assertFalse([line for line in lines if "stalled_min" in line])
        for _ in range(12):  # frozen CPU, but a log line arrives every minute
            now[0] += 60
            beat.observe_log_line("[Preproc] eddy: iteration")
            beat.tick()
        self.assertFalse([line for line in lines if "stalled_min" in line])

    def test_h2_active_stage_log_size_is_reported_and_growth_counts_as_progress(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "ukf_1.log"
            now = [0.0]
            lines = []
            beat = self._beat(lambda: now[0], lines, log_root=tmp)
            beat._cpu.sample = lambda: (5.0, 1)
            beat.set_stage("UKF")
            log.write_text("x" * 100)
            for step in range(12):
                now[0] += 60
                with log.open("a") as stream:
                    stream.write("y" * 10)
                # FLAKY-H2: the ticks run in a few ms, and Linux stamps the append with a coarse clock that can be
                # behind the stage start (time.time()); stamp it clearly after the start, as a real tool's later
                # write is.
                stamp = time.time() + 5.0
                os.utime(log, (stamp, stamp))
                beat.tick()
            self.assertIn("log_bytes=", lines[-1])
            self.assertFalse([line for line in lines if "stalled_min" in line])

    def test_flaky_h2_file_clock_behind_the_wall_clock(self):
        """FLAKY-H2: Linux stamps a write with a coarse clock that can be a few ms behind time.time(). The ticks run
        in a few ms, so each append gets an mtime before the stage start and the log is never counted. This runs the
        H2 test with the stage-start clock 1 s ahead of the file clock (larger than any run time of the ticks)."""
        real_time = time.time
        ahead = SimpleNamespace(time=lambda: real_time() + 1.0, monotonic=time.monotonic)
        with patch.object(hb, "time", ahead):
            self.test_h2_active_stage_log_size_is_reported_and_growth_counts_as_progress()

    def test_h2_missing_psutil_prints_elapsed_only(self):
        lines = []
        beat = hb.Heartbeat(out=lines.append)
        beat._cpu.available = False
        beat.set_stage("WMQL")
        beat.tick()
        self.assertRegex(lines[0], r"^\[Heartbeat\] stage=WMQL elapsed=\d+$")

    def test_h2_real_thread_with_one_second_interval_prints_the_stall_warning(self):
        lines = []
        beat = hb.Heartbeat(interval=1, stall_minutes=0.03, out=lines.append)  # 1.8 s without progress
        beat.set_stage("fake quiet stage")
        beat.start()
        try:
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline and not any(line.startswith("[Stall warning]") for line in lines):
                time.sleep(0.1)
        finally:
            beat.stop()
        self.assertTrue(any(line.startswith("[Heartbeat] stage=fake quiet stage elapsed=") for line in lines), lines)
        self.assertTrue(any(line.startswith("[Stall warning] fake quiet stage:") for line in lines), lines)

    def test_h2_cli_runs_the_heartbeat_around_the_run_in_execute_mode(self):
        source = (REPO / "shared/run_subject_pipeline_cli.py").read_text(encoding="utf-8")
        self.assertIn("Heartbeat(", source)
        self.assertIn("heartbeat.wrap_log(_cli_log)", source)
        self.assertIn("heartbeat.stop()", source)


if __name__ == "__main__":
    unittest.main()
