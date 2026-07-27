"""Redacting tee for the pod's stdout/stderr stream. Piped from pod_entry.sh:
    <job> 2>&1 | python3.12 -m assay.log_tee <raw_path> <redacted_path> [<spill_path>]
Child and grandchild processes (the eval subprocess, the vLLM EngineCore) inherit
fd 1/2, so their output flows through here too - one redaction point for the tree.

<raw_path>: ephemeral LOCAL container fs (for the GATE-marker grep + the watchdog's
mtime liveness signal); unredacted; written SYNCHRONOUSLY (local fs is stall-proof).
<redacted_path>: the durable network VOLUME; secrets scrubbed; written ASYNCHRONOUSLY
by a daemon flusher off a bounded queue so a slow/gone volume can NEVER backpressure
and wedge the job. On queue overflow the redacted line spills to <spill_path> (local
failover) - no loss, no wedge, no new network/auth surface. The raw stream must never
be persisted to the volume.

ASCII out, errors='replace', per-line redaction so a partial/non-utf8 line can never
crash the tee or leak a half-written secret. A value split across two lines may not
match (per-line redaction)."""
from __future__ import annotations

import os
import queue
import sys
import threading

_QUEUE_MAX = 10000  # ~lines buffered before overflow spills locally
_SENTINEL = object()


def _redact(line: str, secrets: list[str]) -> str:
    for s in secrets:
        if s:
            line = line.replace(s, "***")
    return line


def _volume_flusher(q: "queue.Queue", red_path: str, spill_path: str) -> None:
    """Drain the queue to the volume file. If a volume write raises (stall surfaced
    as an error / gone mount), fall back to appending to the local spill so nothing
    is lost. Runs until the sentinel is seen AND the queue is empty."""
    # O_NONBLOCK on the open(2) itself - not just later writes - matters because
    # opening the WRITE end of a FIFO with no reader (or certain gone/stalled mounts)
    # blocks IN THE OPEN CALL, before a single write is attempted. A plain open("w")
    # here would let the flusher wedge in open() with the queue never overflowing
    # (nothing ever reaches a write to fail), so overflow-triggered spill would never
    # fire. O_NONBLOCK makes a no-reader FIFO raise ENXIO immediately instead, which
    # routes every line straight to the local spill file via _write_durable below.
    # Regular files (the real volume sink) are unaffected by O_NONBLOCK on open(2).
    red = None
    fd = None
    try:
        fd = os.open(red_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NONBLOCK,
                     0o644)
        red = os.fdopen(fd, "w", encoding="ascii", errors="replace")
    except OSError:
        if fd is not None:
            try:
                os.close(fd)  # fdopen raised after a successful open - don't leak the fd
            except OSError:
                pass
        red = None
    while True:
        item = q.get()
        if item is _SENTINEL:
            break
        _write_durable(item, red, spill_path)
    if red is not None:
        try:
            red.flush()
            red.close()
        except OSError:
            pass


def _write_durable(line: str, red, spill_path: str) -> None:
    if red is not None:
        try:
            red.write(line)
            red.flush()
            return
        except OSError:
            pass  # volume write failed -> spill locally below
    _spill(line, spill_path)


def _spill(line: str, spill_path: str) -> None:
    try:
        with open(spill_path, "a", encoding="ascii", errors="replace") as fh:
            fh.write(line)
    except OSError:
        pass  # last resort; console passthrough already carries the line


def main(argv: list[str]) -> int:
    raw_path, red_path = argv[1], argv[2]
    # Default derives from raw_path (the local mktemp), NOT red_path (the volume):
    # the spill is the LOCAL failover for when the volume is unreachable, so it must
    # live on the same stall-proof local fs as the raw sink. A red_path-derived
    # default would put the failover ON the very mount it is meant to fail away
    # from - its own open() would then hang on a hard volume stall, wedging the
    # tee and freezing the raw mtime the watchdog treats as its PRIMARY liveness
    # signal (see build_eval_watchdog). pod_entry.sh always passes an explicit
    # argv[3]; this default only matters for direct/manual/test invocations.
    spill_path = argv[3] if len(argv) > 3 else raw_path + ".spill"
    secrets = [os.environ.get("HF_TOKEN", ""), os.environ.get("RUNPOD_API_KEY", "")]
    for p in (raw_path, red_path, spill_path):
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    # The raw sink opens "w" and the redacted sink opens O_TRUNC, but the spill was
    # opened "a" and nothing else truncated it - so a SAME-POD restart concatenated
    # the previous run's spill onto this run's, contaminating cross-run forensics
    # (same family as the confirmed heartbeat append-mode bug). Truncate exactly once
    # here, before the flusher thread starts, so _spill's appends stay within-run.
    # Best-effort with the same posture as _spill itself: a failure to truncate must
    # never stop the tee, because the tee failing wedges the job it is logging.
    try:
        with open(spill_path, "w", encoding="ascii", errors="replace"):
            pass
    except OSError:
        pass
    sys.stdin.reconfigure(errors="replace")

    q: "queue.Queue" = queue.Queue(maxsize=_QUEUE_MAX)
    flusher = threading.Thread(target=_volume_flusher,
                               args=(q, red_path, spill_path), daemon=True)
    flusher.start()

    with open(raw_path, "w", encoding="ascii", errors="replace") as raw:
        for line in sys.stdin:
            raw.write(line)          # SYNCHRONOUS local raw sink (stall-proof)
            raw.flush()
            scrubbed = _redact(line, secrets)
            sys.stdout.write(scrubbed)   # console passthrough (RunPod log tab), redacted
            sys.stdout.flush()
            try:
                q.put_nowait(scrubbed)   # never blocks the job's stdout path
            except queue.Full:
                _spill(scrubbed, spill_path)  # overflow -> local failover, no wedge

    # EOF: drain the queue best-effort, then close.
    try:
        q.put_nowait(_SENTINEL)
    except queue.Full:
        pass  # flusher is stuck on a hung volume write; the bounded join below caps exit
    flusher.join(timeout=30)
    if flusher.is_alive():
        # Flusher never woke up (hung volume write) - salvage whatever redacted
        # lines are still sitting in the (ephemeral) queue to the LOCAL spill so
        # the durable redacted record keeps its tail. Queue is thread-safe, so
        # this races the flusher harmlessly: worst case a line lands in both the
        # volume and the spill (duplicated), never neither (lost).
        while True:
            try:
                item = q.get_nowait()
            except queue.Empty:
                break
            if item is not _SENTINEL:
                _spill(item, spill_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
