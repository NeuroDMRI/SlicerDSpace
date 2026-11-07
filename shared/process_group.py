"""Launch/kill helpers for a QProcess's whole process tree, not just one PID.

``QProcess.terminate()``/``.kill()`` only signal the direct child process.
When that child is a launcher (e.g. the Slicer executable running
``run_subject_pipeline_cli.py``) that spawns further subprocesses for each
pipeline task (MRtrix, FreeSurfer, dcm2niix, ...), cancelling just the top
process leaves those descendants running as orphans. Launch the child in its
own process group (POSIX, via ``setsid`` or a ``setpgrp`` fallback) and kill
the whole group/tree instead of one PID. This mirrors the approach already
used for setup jobs in ``shared/ui_setup_jobs.py``.
"""

from __future__ import annotations

import os
import platform
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

_PERL_SETPGRP_PATH = "/usr/bin/perl"
# The indirect-object form, "exec { $ARGV[0] } @ARGV", always runs the
# program directly via execvp() -- never through a shell -- regardless of
# how many elements @ARGV has. Plain "exec @ARGV" would do the same for two
# or more elements, but degrades to shell interpretation (of the single
# string) when the launched program is invoked with zero arguments, i.e.
# @ARGV has exactly one element; if that one argument contained shell
# metacharacters (e.g. a path with a space), it would be misinterpreted.
# "or die" makes a failed exec exit non-zero; perl would otherwise exit 0.
_PERL_SETPGRP_SCRIPT = 'setpgrp(0, 0); exec { $ARGV[0] } @ARGV or die "exec failed: $!\\n"'


def _perl_setpgrp_launcher() -> Optional[str]:
    """Return a perl interpreter usable for the ``setpgrp`` launch trick, if any.

    Prefers the canonical ``/usr/bin/perl`` -- present on macOS and
    virtually every Linux distribution, including ones that ship no
    system-wide ``perl`` on PATH by default -- and falls back to whatever
    ``perl`` is on PATH for the rare system where it lives elsewhere.
    """

    if os.path.exists(_PERL_SETPGRP_PATH):
        return _PERL_SETPGRP_PATH
    return shutil.which("perl")


def grouped_launch_command(program: str, args: list[str]) -> tuple[str, list[str], bool]:
    """Return ``(program, args, grouped)`` so the target runs as its own process-group leader.

    ``grouped`` is True only when the returned command will actually run the
    target program as the leader of its own process group. Windows needs no
    launch-time change: ``taskkill /T`` can kill a process's descendant tree
    from its PID alone.

    On POSIX, ``setsid(1)`` (from util-linux) is preferred when present. It
    is absent on macOS and other BSD-derived systems -- ``shutil.which
    ("setsid")`` always returns ``None`` there -- which used to mean grouping
    silently fell back to "ungrouped" on exactly the platform where this
    extension is primarily developed and used. The fallback is a one-line
    perl script that calls the ``setpgrp(0, 0)`` syscall directly (making
    the not-yet-``exec``'d process the leader of a new process group of its
    own -- the same property ``setsid(1)`` provides for this purpose) and
    then ``exec``s the real target in its place, preserving its pid. Both
    paths leave the launched process as its own process-group leader, i.e.
    with ``pgid == pid`` -- see :func:`capture_process_group_id`.
    """

    from shared.external_process import normalize_slicer_python_launch
    program, args = normalize_slicer_python_launch(program, args)
    if os.name == "posix":
        setsid = shutil.which("setsid")
        if setsid:
            return setsid, [program, *args], True
        perl = _perl_setpgrp_launcher()
        if perl:
            return perl, ["-e", _PERL_SETPGRP_SCRIPT, "--", program, *args], True
    return program, args, False


def capture_process_group_id(pid: int) -> Optional[int]:
    """Return the process-group id ``pid`` will have as a grouped-launch leader.

    Call this once, right after a grouped launch (``grouped_launch_command``
    returned ``grouped=True``), while the launched process is known to still
    be alive (e.g. immediately after ``QProcess.waitForStarted()`` succeeds),
    and store the result for :func:`kill_process_tree` to use later.

    This deliberately does **not** call ``os.getpgid(pid)`` to look the pgid
    up. Two reasons:

    1. It would race the launched process's own ``setsid()``/``setpgrp(0,
       0)`` call: ``waitForStarted()`` returns as soon as ``exec`` of
       ``setsid``/perl succeeds, which is *before* that process has actually
       made itself its own group leader. Calling ``os.getpgid(pid)`` in that
       window can observe the *inherited* pgid instead (e.g. Slicer's own),
       and a later forced ``killpg()`` against that stale value would signal
       the wrong process group entirely.
    2. It is unnecessary: POSIX only allows a process to make itself the
       leader of a *new* group by setting its own pgid to its own pid
       (that is what both ``setsid()`` and ``setpgrp(0, 0)`` do here -- see
       :func:`grouped_launch_command`). So once that call completes, this
       process's pgid is guaranteed to equal its own pid; there is no other
       value it could be.

    ``pid`` is returned as-is (once validated), letting the eventual
    ``killpg()`` no-op with ``ESRCH`` if called before that setup completes,
    rather than risk targeting the wrong group.
    """

    if os.name != "posix" or pid is None or pid <= 0:
        return None
    return pid


def process_group_is_running(pgid: int) -> bool:
    """Check live members of an owned group, excluding Linux zombie processes."""
    if not pgid or pgid <= 0:
        return False
    if platform.system() == "Linux" and Path("/proc").is_dir():
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                fields = (entry / "stat").read_text().rsplit(") ", 1)[1].split()
                if int(fields[2]) == pgid and fields[0] not in {"Z", "X"}:
                    return True
            except (OSError, ValueError, IndexError):
                continue
        return False
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def finish_process_group(pgid: int, *, grace_seconds: float) -> None:
    """Wait for cleanup, then kill survivors even if the launcher has exited.

    Only use the group id recorded by capture_process_group_id at launch.
    This synchronous helper is for application teardown, not normal GUI slots.
    """
    deadline = time.monotonic() + grace_seconds
    while process_group_is_running(pgid):
        if time.monotonic() >= deadline:
            kill_process_tree(0, grouped=True, force=True, pgid=pgid)
            return
        time.sleep(0.1)


def kill_process_tree(
    pid: int,
    *,
    grouped: bool,
    force: bool,
    pgid: Optional[int] = None,
    platform_name: Optional[str] = None,
) -> None:
    """Signal the whole process group/tree rooted at ``pid``.

    ``force`` requests an unconditional kill (``SIGKILL`` / ``taskkill /F``);
    when False, a milder terminate signal (``SIGTERM`` / ``taskkill``
    without ``/F``) is used instead, giving children a chance to shut down
    cleanly before a later, forced follow-up call.

    ``pgid`` should be the value :func:`capture_process_group_id` returned at
    launch time. When it is available, ``killpg`` is called unconditionally
    against it, ignoring ``ProcessLookupError`` (ESRCH): by the time a forced
    follow-up call fires, the group leader may already be gone even though
    its descendants are not, and re-deriving the pgid from ``pid`` at this
    point (via ``os.getpgid``) would itself raise in exactly that case. When
    ``pgid`` is omitted, ``pid`` is used in its place: by construction,
    :func:`grouped_launch_command` only reports ``grouped=True`` when the
    launched process was made the leader of its own new process group, so
    its own pid equals its pgid at launch time.
    """

    system = platform_name or platform.system()
    if system == "Windows":
        if pid is None or pid <= 0:
            return
        command = ["taskkill", "/PID", str(pid), "/T"]
        if force:
            command.append("/F")
        subprocess.run(command, check=False, capture_output=True)
        return
    if grouped:
        target_pgid = pgid if pgid is not None else pid
        if target_pgid and target_pgid > 0:
            try:
                os.killpg(target_pgid, signal.SIGKILL if force else signal.SIGTERM)
            except ProcessLookupError:
                pass
            return
    if pid is None or pid <= 0:
        return
    try:
        os.kill(pid, signal.SIGKILL if force else signal.SIGTERM)
    except ProcessLookupError:
        pass
