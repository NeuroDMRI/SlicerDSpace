"""Batch-4 fix round, group runtime: U4 (line stream), U5 (cohort GPU check), P-f (D5 order), P-g (activation problem)."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO = Path(__file__).resolve().parents[2]
for _entry in (str(REPO), str(REPO / "Testing" / "Python")):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

from shared import line_stream
from shared.line_stream import DESCENDANTS_NOTICE, stream_process_lines

POSIX = unittest.skipUnless(os.name == "posix", "needs /bin/sh")


class _CountingQueue(line_stream.Queue):
    puts = 0

    def put(self, item, *args, **kwargs):
        type(self).puts += 1
        return super().put(item, *args, **kwargs)


class _Endless:
    """A pipe-like stream whose far writer never stops; ``closed`` records whether the reader closed it."""

    def __init__(self, delay=0.0002):
        self.stop = threading.Event()
        self.closed = False
        self.read = 0
        self.delay = delay

    def __iter__(self):
        return self

    def __next__(self):
        if self.stop.is_set():
            raise StopIteration
        self.read += 1
        if self.delay:
            time.sleep(self.delay)
        return "noise\n"

    def close(self):
        self.closed = True


class _FakeProcess:
    def __init__(self, stream, exit_after, code):
        self.stdout = stream
        self._exit_at = time.monotonic() + exit_after
        self._code = code

    def poll(self):
        return self._code if time.monotonic() >= self._exit_at else None

    def wait(self):
        while self.poll() is None:
            time.sleep(0.01)
        return self._code


class U4AbandonedReaderTests(unittest.TestCase):
    def test_u4_endless_grandchild_output_returns_within_grace_plus_two_seconds_and_keeps_the_exit_code(self):
        stream = _Endless()
        self.addCleanup(stream.stop.set)
        process = _FakeProcess(stream, exit_after=0.2, code=7)
        lines, notes = [], []
        started = time.monotonic()
        code = stream_process_lines(process, lines.append, notice=notes.append, grace_seconds=0.5)
        self.assertLess(time.monotonic() - started, 0.5 + 2.0)
        self.assertEqual(code, 7)
        self.assertEqual(notes, [DESCENDANTS_NOTICE])
        self.assertFalse(stream.closed)  # the pipe stays open: nothing is killed

    def test_u4_flooding_grandchild_faster_than_the_consumer_still_returns_in_time(self):
        stream = _Endless(delay=0)  # no pause: the reader outruns the consumer, so the queue never empties
        self.addCleanup(stream.stop.set)
        process = _FakeProcess(stream, exit_after=0.1, code=5)
        lines = []

        def on_line(line):
            lines.append(line)
            time.sleep(0.0002)

        started = time.monotonic()
        code = stream_process_lines(process, on_line, grace_seconds=0.5)
        self.assertLess(time.monotonic() - started, 0.5 + 2.0)
        self.assertEqual(code, 5)

    def test_u4_reader_stops_queueing_after_abandonment(self):
        stream = _Endless()
        self.addCleanup(stream.stop.set)
        process = _FakeProcess(stream, exit_after=0.1, code=0)
        _CountingQueue.puts = 0
        with mock.patch.object(line_stream, "Queue", _CountingQueue):
            stream_process_lines(process, lambda line: None, grace_seconds=0.3)
            puts_at_return = _CountingQueue.puts
            read_at_return = stream.read
            time.sleep(0.5)
        self.assertGreater(stream.read, read_at_return)  # the reader still drains the pipe ...
        self.assertLessEqual(_CountingQueue.puts, puts_at_return + 1)  # ... but queues nothing more

    @POSIX
    def test_u4_real_grandchild_that_writes_continuously(self):
        script = "(while :; do echo noise; done) & exit 4"
        # R3-T: a new session makes the shell the leader of its own process group; the background loop stays in
        # that group, so the cleanup kills this test's processes only (pgid == shell pid), never a system-wide match.
        process = subprocess.Popen(["/bin/sh", "-c", script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, start_new_session=True)
        pgid = process.pid
        count = [0]

        def on_line(line):
            count[0] += 1

        started = time.monotonic()
        try:
            code = stream_process_lines(process, on_line, grace_seconds=0.5)
            self.assertLess(time.monotonic() - started, 2.5)
            self.assertEqual(code, 4)
        finally:
            self.assertTrue(_kill_group(pgid), "the grandchild loop survived the process-group kill")

    def test_u4_n5_slow_consumer_of_a_finished_reader_gets_every_line_and_no_notice(self):
        class Finite:
            def __init__(self):
                self.closed = False

            def __iter__(self):
                return iter(["l%d\n" % index for index in range(30)])

            def close(self):
                self.closed = True

        process = _FakeProcess(Finite(), exit_after=0.0, code=0)
        lines, notes = [], []

        def slow(line):
            time.sleep(0.05)  # 30 lines take 1.5 s, three times the 0.5 s grace
            lines.append(line)

        code = stream_process_lines(process, slow, notice=notes.append, grace_seconds=0.5)
        self.assertEqual(code, 0)
        self.assertEqual(len(lines), 30)
        self.assertEqual(notes, [])


def _kill_group(pgid, timeout=5.0):
    """SIGKILL one process group that this test started; True once no member of the group is left.

    The loop is endless, so a group that is already gone means the loop escaped it: that returns False.
    """
    import signal

    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            pass
        time.sleep(0.05)
    return False


class R3LineStreamAndCleanupTests(unittest.TestCase):
    def test_r3_N2_an_exception_from_on_line_stops_the_reader_queueing(self):
        stream = _Endless()
        self.addCleanup(stream.stop.set)
        process = _FakeProcess(stream, exit_after=30.0, code=0)
        _CountingQueue.puts = 0

        def on_line(line):
            raise RuntimeError("callback failed")

        with mock.patch.object(line_stream, "Queue", _CountingQueue):
            with self.assertRaisesRegex(RuntimeError, "callback failed"):
                stream_process_lines(process, on_line, grace_seconds=0.3)
            puts_at_return = _CountingQueue.puts
            read_at_return = stream.read
            time.sleep(0.5)
        self.assertGreater(stream.read, read_at_return)  # the daemon reader still drains the pipe ...
        self.assertLessEqual(_CountingQueue.puts, puts_at_return + 1)  # ... but queues nothing more

    def test_r3_T_line_stream_cleanup_kills_only_its_own_process_group(self):
        import inspect

        module_source = Path(__file__).read_text(encoding="utf-8")
        for tool in ("pk" + "ill", "pg" + "rep"):  # split so this test's own text does not match
            self.assertNotIn('"%s"' % tool, module_source)
        test_source = inspect.getsource(U4AbandonedReaderTests.test_u4_real_grandchild_that_writes_continuously)
        self.assertIn("start_new_session=True", test_source)
        self.assertIn("_kill_group(pgid)", test_source)


class U5CohortGpuCheckTests(unittest.TestCase):
    def _config(self):
        from shared.orchestrator import ProductOrchestratorConfig

        return ProductOrchestratorConfig(subject_id="s1", subject_root=Path("/nonexistent"), targets=("surface",),
                                         execute=True, surface_device="cuda:0")

    def test_u5_cohort_active_helper_is_the_schedule_env_check(self):
        from shared import cohort_schedule

        with mock.patch.dict(os.environ, {cohort_schedule.ENV: "/x/schedule"}):
            self.assertTrue(cohort_schedule.cohort_active())
        with mock.patch.dict(os.environ):
            os.environ.pop(cohort_schedule.ENV, None)
            self.assertFalse(cohort_schedule.cohort_active())

    def test_u5_cohort_skips_the_gpu_check_for_plan_and_run(self):
        from shared import cohort_schedule, orchestrator

        config = self._config()
        with mock.patch.dict(os.environ, {cohort_schedule.ENV: "/x/schedule"}), \
                mock.patch("shared.environment.surface_gpu_memory_decision",
                           side_effect=AssertionError("the LE-3 check must not run")) as check:
            self.assertEqual(orchestrator._surface_gpu_decision(config), ("ok", ""))
            self.assertEqual(orchestrator._surface_gpu_decision(config, planning=True), ("ok", ""))
        check.assert_not_called()

    def test_u5_without_a_cohort_the_check_still_runs(self):
        from shared import cohort_schedule, orchestrator

        config = self._config()
        with mock.patch.dict(os.environ):
            os.environ.pop(cohort_schedule.ENV, None)
            with mock.patch("shared.environment.surface_gpu_memory_decision",
                            return_value=("block", "too little")) as check, \
                    mock.patch("anatomy.surface.backend_bridge.python_executable", return_value="/x/python"):
                self.assertEqual(orchestrator._surface_gpu_decision(config), ("block", "too little"))
                self.assertEqual(orchestrator._surface_gpu_decision(config, planning=True), ("block", "too little"))
        self.assertFalse(check.call_args_list[0].kwargs["cached_only"])  # run path: the real probe
        self.assertTrue(check.call_args_list[1].kwargs["cached_only"])  # plan path: cache only

    def test_u5_plan_and_run_call_sites_pass_the_right_mode(self):
        source = (REPO / "shared" / "orchestrator.py").read_text(encoding="utf-8")
        self.assertIn("_surface_gpu_decision(config, planning=True)", source)
        self.assertIn("(surface_gpu := _surface_gpu_decision(config))", source)

    def test_u5_plan_time_never_starts_the_torch_probe(self):
        from shared import environment

        environment._CUDA_FREE_CACHE.clear()
        self.addCleanup(environment._CUDA_FREE_CACHE.clear)
        with mock.patch.object(environment, "_is_macos", return_value=False), \
                mock.patch.object(environment.subprocess, "run", side_effect=AssertionError("probe")) as run:
            self.assertEqual(environment.surface_gpu_memory_decision("cuda:0", Path("/x/python"), cached_only=True),
                             ("ok", ""))
            self.assertEqual(environment.surface_gpu_memory_decision("auto", Path("/x/python"), cached_only=True),
                             ("ok", ""))
        run.assert_not_called()

    def test_u5_plan_time_uses_a_result_the_run_time_probe_cached(self):
        from shared import environment

        environment._CUDA_FREE_CACHE.clear()
        self.addCleanup(environment._CUDA_FREE_CACHE.clear)
        completed = SimpleNamespace(stdout='{"free_gib": 3.0}')
        with mock.patch.object(environment, "_is_macos", return_value=False), \
                mock.patch.object(environment.subprocess, "run", return_value=completed) as run:
            self.assertEqual(environment.surface_gpu_memory_decision("cuda:0", Path("/x/python"))[0], "block")
            self.assertEqual(
                environment.surface_gpu_memory_decision("cuda:0", Path("/x/python"), cached_only=True)[0], "block")
        self.assertEqual(run.call_count, 1)


class R3ParallelSubjectsGpuCheckTests(unittest.TestCase):
    ENV = "SLICERDSPACE_PARALLEL_SUBJECTS"

    def _config(self):
        return U5CohortGpuCheckTests._config(self)

    def _decide(self, value):
        from shared import cohort_schedule, orchestrator

        with mock.patch.dict(os.environ):
            os.environ.pop(cohort_schedule.ENV, None)
            os.environ.pop(self.ENV, None)
            if value is not None:
                os.environ[self.ENV] = value
            with mock.patch("shared.environment.surface_gpu_memory_decision",
                            return_value=("block", "too little")) as check, \
                    mock.patch("anatomy.surface.backend_bridge.python_executable", return_value="/x/python"):
                results = (orchestrator._surface_gpu_decision(self._config()),
                           orchestrator._surface_gpu_decision(self._config(), planning=True))
        return results, check.call_count

    def test_r3_N3_parallel_subjects_skip_the_gpu_check(self):
        self.assertEqual(self._decide("2"), ((("ok", ""), ("ok", "")), 0))

    def test_r3_N3_one_subject_unset_or_bad_value_keeps_the_gpu_check(self):
        for value in (None, "1", "0", "", "x"):
            with self.subTest(value=value):
                self.assertEqual(self._decide(value), ((("block", "too little"), ("block", "too little")), 2))

    def _batch(self, jobs, subjects):
        from scripts import run_batch_pipeline as batch_cli

        calls = []

        def run_subject(command, subject, log_dir, **kwargs):
            calls.append(kwargs)
            return batch_cli.SubjectRunResult(subject.subject_id, subject.subject_root, command, 0,
                                              log_dir / "stdout", log_dir / "stderr", 0.0,
                                              run_status="planned", status_source="child_manifest")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index in range(subjects):
                input_dir = root / ("sub-%02d" % (index + 1)) / "input"
                input_dir.mkdir(parents=True)
                for name in ("data.nii.gz", "bvals", "bvecs"):
                    (input_dir / name).touch()
            import contextlib
            import io

            with mock.patch.object(batch_cli, "_run_subject", side_effect=run_subject), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(batch_cli.main(["--batch-root", str(root), "--jobs", str(jobs),
                                                 "--manifest", str(root / "batch.json")]), 0)
        return calls

    def test_r3_N3_batch_names_the_parallel_subject_count_to_each_child(self):
        self.assertEqual([call.get("parallel_subjects") for call in self._batch(2, 3)], [2, 2, 2])
        self.assertEqual([call.get("parallel_subjects") for call in self._batch(4, 3)], [3, 3, 3])
        self.assertEqual([call.get("parallel_subjects") for call in self._batch(1, 3)], [None, None, None])
        self.assertEqual([call.get("parallel_subjects") for call in self._batch(4, 1)], [None])

    def test_r3_N3_run_subject_puts_the_count_in_the_child_environment(self):
        from scripts import run_batch_pipeline as batch_cli

        for parallel, expected in ((2, "2"), (1, None)):
            captured = {}

            def fake_run(command, **options):
                captured.update(options)
                return subprocess.CompletedProcess(command, 0)

            with self.subTest(parallel=parallel), tempfile.TemporaryDirectory() as tmp, \
                    mock.patch.dict(os.environ), mock.patch.object(batch_cli.subprocess, "run", fake_run):
                os.environ.pop(self.ENV, None)
                subject = batch_cli.BatchSubject("sub-01", Path(tmp) / "sub-01", "glob", "subject_root")
                batch_cli._run_subject(["python", "x.py"], subject, Path(tmp) / "logs", relay=None,
                                       parallel_subjects=parallel)
                self.assertEqual(captured["env"].get(self.ENV), expected)
                self.assertNotIn("SLICERDSPACE_COHORT_SUBJECT", captured["env"])


class PfBundleOrderTests(unittest.TestCase):
    def _bundle(self, root: Path) -> Path:
        site_dir = root / "bundled-python" / "site-packages"
        (site_dir / "nibabel-5.3.2.dist-info").mkdir(parents=True)
        (site_dir / "nibabel-5.3.2.dist-info" / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: nibabel\nVersion: 5.3.2\n")
        return site_dir

    def _run(self, order):
        """order: 'ahead' puts the bundle before Slicer's site-packages, 'behind' after it, 'absent' off sys.path."""
        import site
        import sysconfig
        from shared import python_setup as setup

        with tempfile.TemporaryDirectory() as td:
            bundle = self._bundle(Path(td))
            slicer = str(Path(td) / "slicer-site")
            (Path(td) / "slicer-site").mkdir()
            paths = {"purelib": slicer, "platlib": slicer}
            rest = list(sys.path)  # keep the real entries: the packaging import needs them
            entries = {"ahead": [str(bundle), slicer], "behind": [slicer, str(bundle)], "absent": [slicer]}[order] + rest
            with mock.patch.dict(os.environ, {"SLICERDSPACE_BUNDLED_PYTHON_ROOT": str(bundle.parent)}), \
                    mock.patch.object(sys, "path", entries), \
                    mock.patch.object(site, "getsitepackages", return_value=[slicer]), \
                    mock.patch.object(sysconfig, "get_paths", return_value=paths):
                return setup.bundled_shortfalls(["nibabel>=6"])

    def test_p_f_refuses_when_the_bundle_is_ahead_of_slicer_site_packages(self):
        self.assertEqual(len(self._run("ahead")), 1)

    def test_p_f_no_refusal_when_the_bundle_is_appended_behind_slicer_site_packages(self):
        self.assertEqual(self._run("behind"), [])

    def test_p_f_no_refusal_when_the_bundle_is_not_on_sys_path(self):
        self.assertEqual(self._run("absent"), [])


class PgActivationProblemTests(unittest.TestCase):
    def setUp(self):
        from shared import bundled_python

        self.bp = bundled_python
        self.bp._PROBLEMS.clear()
        self.addCleanup(self.bp._PROBLEMS.clear)
        env_patch = mock.patch.dict(os.environ, {}, clear=False)
        env_patch.start()
        self.addCleanup(env_patch.stop)
        os.environ.pop(self.bp.BUNDLED_PYTHON_ENV, None)

    def _root(self, base: Path, meta: dict) -> Path:
        import json

        root = base / "bundled-python"
        (root / "site-packages").mkdir(parents=True)
        (root / "python-runtime.json").write_text(json.dumps(meta), encoding="utf-8")
        return root

    def test_p_g_no_problem_when_another_root_is_valid_and_active(self):
        with tempfile.TemporaryDirectory() as td:
            bad = self._root(Path(td) / "old", {"python_version": "3.99", "arch": "x86_64"})
            good = self._root(Path(td) / "new", {"python_version": "%d.%d" % sys.version_info[:2]})
            with mock.patch.object(self.bp, "_default_extension_roots",
                                   return_value=(bad.parent, good.parent)):
                self.assertEqual(self.bp.bundled_python_roots(), (good.resolve(),))
                self.assertIn(str(bad), self.bp._PROBLEMS)  # the stale root was recorded ...
                self.assertIsNone(self.bp.activation_problem())  # ... but is not reported

    def test_p_g_problem_still_reported_when_no_root_is_valid(self):
        with tempfile.TemporaryDirectory() as td:
            bad = self._root(Path(td) / "old", {"python_version": "3.99", "arch": "x86_64"})
            with mock.patch.object(self.bp, "_default_extension_roots", return_value=(bad.parent,)):
                self.assertEqual(self.bp.bundled_python_roots(), ())
                self.assertIn("bundled Python stack is for Python 3.99", self.bp.activation_problem())


if __name__ == "__main__":
    unittest.main()
