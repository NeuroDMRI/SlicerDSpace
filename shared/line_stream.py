"""Read a child's merged output line by line without waiting for its descendants (H8).

A plain ``for line in process.stdout`` loop ends only when *every* writer has
closed the pipe. When the direct child exits but leaves a background
grandchild that still holds the pipe, the loop (and so the pipeline stage)
stays open until that grandchild ends. :func:`stream_process_lines` reads in a
daemon thread and hands the lines to the caller's thread, so line callbacks keep
running on the calling thread exactly as before. After the direct child exits it
gives the reader ``grace_seconds`` to deliver the output that is still in
flight, then stops reading and returns the exit code of the direct child.

It never kills anything: surviving descendants keep running. The pipe stays
open (held by the daemon reader thread) until they exit.
"""

from __future__ import annotations

from queue import Empty, Queue
import threading
import time
from typing import Callable, Optional

DESCENDANTS_NOTICE = "child exited; descendants still hold the output pipe"
DEFAULT_GRACE_SECONDS = 5.0
DRAIN_SECONDS = 1.0  # U4: the most time the final snapshot delivery may take once the reader is abandoned
_EOF = object()


def stream_process_lines(
    process,
    on_line: Callable[[str], None],
    *,
    notice: Optional[Callable[[str], None]] = None,
    grace_seconds: float = DEFAULT_GRACE_SECONDS,
) -> int:
    """Feed every output line of ``process`` to ``on_line``; return its exit code.

    ``process`` is a ``subprocess.Popen`` started with ``stdout=PIPE``.
    ``on_line`` receives the raw line (with its newline), on the calling thread.
    ``notice`` receives ``DESCENDANTS_NOTICE`` when the reader was abandoned.
    """

    stream = process.stdout
    if stream is None:
        return process.wait()
    lines: Queue = Queue()
    abandoned_event = threading.Event()  # U4: once set, the reader reads on but discards (it never queues again)
    reader_done = threading.Event()  # U4/N5: the reader reached EOF; only the consumer may still be behind

    def reader() -> None:
        try:
            for line in stream:
                if not abandoned_event.is_set():
                    lines.put(line)
        except (OSError, ValueError):
            pass
        finally:
            reader_done.set()
            try:
                stream.close()  # this thread is the only reader, so closing here cannot block
            except (OSError, ValueError):
                pass
            lines.put(_EOF)

    thread = threading.Thread(target=reader, name="line-stream-reader", daemon=True)
    thread.start()
    exited_at: Optional[float] = None
    abandoned = False
    try:
        while True:
            try:
                item = lines.get(timeout=0.1)
            except Empty:
                item = None
            if item is _EOF:
                break
            if item is not None:
                on_line(item)
            if exited_at is None and process.poll() is not None:
                exited_at = time.monotonic()
            if (
                exited_at is not None
                and time.monotonic() - exited_at >= grace_seconds
                and not reader_done.is_set()  # N5: a finished reader plus a slow consumer is not an abandoned pipe
            ):
                abandoned = True
                break
        if abandoned:
            # Stop the reader queueing, then deliver a finite snapshot of what it already queued.
            abandoned_event.set()
            drain_until = time.monotonic() + DRAIN_SECONDS
            for _ in range(lines.qsize()):
                if time.monotonic() >= drain_until:
                    break  # a descendant that floods the pipe cannot hold the stage through its backlog
                try:
                    item = lines.get_nowait()
                except Empty:
                    break
                if item is _EOF:
                    break
                on_line(item)
            if notice:
                notice(DESCENDANTS_NOTICE)
        return process.wait()
    finally:
        # R3-N2: on any exit (an on_line exception too) the daemon reader must stop queueing.
        abandoned_event.set()
        # Closing a pipe that the reader is blocked on would block here; only close a finished reader.
        thread.join(timeout=0.5 if not abandoned else 0)
        if not thread.is_alive():
            try:
                stream.close()
            except (OSError, ValueError):
                pass
