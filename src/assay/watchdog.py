"""In-pod progress-stall watchdog for the eval subprocess.

The R1 avg@16 burn wedged POST-INFERENCE in lm-eval's scoring/serialization tail:
all generation completed, then ~76 min of total silence, then a blind 6h wall-clock
backstop killed a pod that had already done the expensive work. This watches
independent liveness signals and kills the eval process group ONLY when every
available signal is flat past a threshold - distinguishing "slow but working"
(CPU/GPU still advancing) from "wedged" (all flat). It replaces the blind backstop;
there is no wall-clock money guillotine anymore (RunPod credits are the external
money ceiling, and the external wks1 observer covers a whole-pod freeze).

Kill scope is the eval child's OWN process group (the child does os.setpgid(0,0)),
never group 0 - killing group 0 would take out the shell, log_tee, and the
forensics upload with it (the self-guillotine the old `kill -TERM 0` risked)."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time


def _read_mtime(path: str) -> float | None:
    """Modification time of `path` in seconds, or None if it does not exist yet /
    cannot be stat'd. This is the PRIMARY signal: tqdm writes a progress line every
    iteration through log_tee's raw local sink, so a live generating run keeps this
    advancing even though it is heartbeat-silent for hours."""
    try:
        return os.stat(path).st_mtime
    except OSError:
        return None


def _pgid_cpu_jiffies(pgid: int, proc_root: str = "/proc") -> int | None:
    """Sum utime+stime (clock ticks) across every process whose process group is
    `pgid`. None if no such process is found (nothing to watch). This is the
    slow-vs-wedged discriminator: legitimate sympy/math_verify scoring advances CPU
    (not a wedge), a true deadlock is flat."""
    total = 0
    found = False
    try:
        entries = os.listdir(proc_root)
    except OSError:
        return None
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(os.path.join(proc_root, entry, "stat"), encoding="ascii",
                      errors="replace") as fh:
                # comm (field 2) can contain spaces and parens; split on the LAST ')'
                # so the remaining fields align regardless of the process name.
                after = fh.read().rsplit(")", 1)[1].split()
            # after: state(0) ppid(1) pgrp(2) session(3) ... utime(11) stime(12)
            if int(after[2]) != pgid:
                continue
            total += int(after[11]) + int(after[12])
            found = True
        except (OSError, IndexError, ValueError):
            continue
    return total if found else None


def _read_gpu_util() -> int | None:  # pragma: no cover - exercised only with a GPU
    """Current GPU utilization percent (0-100), or None if unreadable. Prefer pynvml;
    fall back to nvidia-smi; degrade to None (drop the signal) if neither works. A
    value of 0 means the GPU is idle - during generation this is high, during the
    post-inference scoring tail it drops, which is exactly the wedge window."""
    try:
        import pynvml  # noqa: PLC0415
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return int(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False)
        if out.returncode == 0 and out.stdout.strip():
            return int(out.stdout.strip().splitlines()[0])
    except Exception:
        pass
    return None


class _Signal:
    """One liveness signal. `probe() -> value | None` (None == unavailable this
    tick, dropped from the vote). `progressed(prev, cur) -> bool` decides whether a
    new sample counts as forward motion: counters (mtime, cpu jiffies) use cur > prev;
    the GPU-util level uses cur > 0 (busy right now)."""

    def __init__(self, name, probe, progressed):
        self.name = name
        self.probe = probe
        self.progressed = progressed
        self.last_value = None
        self.last_progress_t = None


class StallWatchdog:
    """Kill the eval child's process group only when EVERY available signal has been
    flat longer than `threshold` seconds. threshold == 0 disables the watchdog
    entirely (the limitless override). Test seam: drive `_tick()` directly with an
    injected `clock`; production uses `start()`/`stop()` around a daemon thread."""

    def __init__(self, signals, on_stall, *, threshold, interval=30.0,
                 heartbeat=None, clock=time.monotonic, sleep=None,
                 beat_interval=600.0):
        self.signals = list(signals)
        self.on_stall = on_stall
        self.threshold = float(threshold)
        self.interval = interval
        self.heartbeat = heartbeat
        self.clock = clock
        # Rate limit for the "progress" heartbeat emit - default 10 min, matched
        # to the tick interval's ~30s: an emit-per-tick would hold the Heartbeat
        # lock and do a synchronous volume write on EVERY tick, so on a volume
        # stall the watchdog thread hangs in its OWN emit (blinding the stall
        # detector) and can block run_eval's post-eval emit on the same lock.
        self.beat_interval = beat_interval
        self._last_beat_t = None
        self._stop = threading.Event()
        self._sleep = sleep  # None -> use self._stop.wait (interruptible)
        self._thread = None
        self._killed = False

    def _tick(self) -> bool:
        now = self.clock()
        available = 0
        stalled = 0
        progressed_any = False
        for sig in self.signals:
            cur = sig.probe()
            if cur is None:
                continue
            available += 1
            first = sig.last_value is None
            if first or sig.progressed(sig.last_value, cur):
                sig.last_progress_t = now
                if not first:
                    progressed_any = True
            sig.last_value = cur
            if sig.last_progress_t is None:
                sig.last_progress_t = now
            if now - sig.last_progress_t > self.threshold:
                stalled += 1
        if progressed_any and self.heartbeat is not None:
            due = self._last_beat_t is None or now - self._last_beat_t >= self.beat_interval
            if due:
                # Mark due BEFORE the call (not only on success): the point of the
                # rate limit is to bound how often this thread can even ATTEMPT a
                # blocking volume write, so a hanging/erroring emit must not reopen
                # the window on the very next tick.
                self._last_beat_t = now
                try:
                    self.heartbeat.emit("progress", "eval making forward motion")
                except Exception:
                    pass  # best-effort - a raised error can never break the tick
        if self.threshold > 0 and available > 0 and stalled == available:
            self._killed = True
            self.on_stall()
            return True
        return False

    def _run(self):  # pragma: no cover - thread body, logic tested via _tick
        while not self._stop.is_set():
            try:
                if self._tick():
                    return
            except Exception:
                pass  # a watchdog must never crash the run it guards
            if self._sleep is not None:
                self._sleep(self.interval)
            else:
                self._stop.wait(self.interval)

    def start(self):  # pragma: no cover - thread lifecycle
        self._thread = threading.Thread(target=self._run, daemon=True,
                                         name="assay-stall-watchdog")
        self._thread.start()

    def stop(self):  # pragma: no cover - thread lifecycle
        self._stop.set()
        if self._thread is not None:
            try:
                self._thread.join(timeout=5)
            except RuntimeError:
                pass  # start() half-failed (thread never actually started)


def _collect_descendants(root_pid: int, proc_root: str = "/proc") -> list[int]:
    """Pure /proc ppid-walk: pids descended from root_pid, EXCLUDING root_pid
    itself. No signaling - this is a SNAPSHOT, and callers must take it BEFORE
    killing anything, because a setsid-escaped grandchild (vLLM EngineCore) is
    reparented to pid 1 the instant its in-group parent dies. A ppid walk
    performed after that reparenting no longer reaches it - the exact escapee
    this collector exists to find would already be gone from the tree."""
    if root_pid <= 1:
        return []
    children: dict[int, list[int]] = {}
    try:
        entries = os.listdir(proc_root)
    except OSError:
        entries = []
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(os.path.join(proc_root, entry, "stat"), encoding="ascii",
                      errors="replace") as fh:
                after = fh.read().rsplit(")", 1)[1].split()
            ppid = int(after[1])
        except (OSError, IndexError, ValueError):
            continue
        children.setdefault(ppid, []).append(int(entry))
    seen, stack, descendants = {root_pid}, list(children.get(root_pid, [])), []
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        descendants.append(pid)
        stack.extend(children.get(pid, []))
    return descendants


def _kill_stalled(pgid: int, *, signaler=os.killpg, killer=os.kill,
                  collector=_collect_descendants, proc_root: str = "/proc",
                  sleep=time.sleep) -> None:  # pragma: no cover - real signals
    """Kill a wedged eval subprocess's process group, plus any setsid-escaped
    grandchild that lives outside it.

    Order matters for two independent safety properties:

    1. pgid <= 1 is refused outright - 0 is the SENDER's own process group
       (os.kill/os.killpg with pid 0 targets the caller, not "no one" - the
       self-guillotine that would take the shell/log_tee/forensics-upload with
       it) and 1 is init; neither is ever a legitimate eval-child group.
    2. A liveness probe (signal 0) runs before any real signal. If the group is
       already gone (ProcessLookupError - normal exit, or the pgid got reused
       and freed again), stand down rather than kill whatever now holds that
       pgid.
    3. The descendant snapshot is collected BEFORE the group SIGKILL, not after
       - see `_collect_descendants` for why a post-kill walk misses the exact
       escapee it exists for.

    Then: SIGUSR1 (faulthandler stack dump -> stderr -> log_tee -> forensics),
    a pause, the group SIGKILL, and finally a SIGKILL to each snapshotted
    escapee pid (best-effort; a pid that's already gone is not an error)."""
    if pgid <= 1:
        return
    try:
        signaler(pgid, 0)
    except ProcessLookupError:
        return
    except OSError:
        pass
    escapees = collector(pgid, proc_root)
    # SIGUSR1 only dumps a stack for a process that registered faulthandler on that
    # signal (the direct eval child does, in evaluate.py's subprocess body). The
    # setsid-escaped vLLM EngineCore grandchild never registers it, so SIGUSR1's
    # default disposition (terminate) kills that grandchild outright here - only
    # the direct child's stack makes it into the forensics, never the EngineCore's.
    try:
        signaler(pgid, signal.SIGUSR1)
    except OSError:
        pass
    sleep(2.0)
    try:
        signaler(pgid, signal.SIGKILL)
    except OSError:
        pass
    for pid in escapees:
        try:
            killer(pid, signal.SIGKILL)
        except OSError:
            pass


def stall_drill_warning(env) -> str | None:
    """Operator warning when the Phase-B wedge rehearsal cannot possibly fire.

    ASSAY_INJECT_STALL_AFTER blocks the eval child AFTER generation to rehearse the
    kill chain on a real hang. That only exercises anything if the injected stall
    outlasts ASSAY_STALL_SECONDS; otherwise the child resumes before the watchdog
    trips and the drill is INERT while looking like it ran.

    Never raises: this runs on the paid path, so a malformed knob degrades to silence
    rather than stranding a burn (same posture as the threshold parse below). A
    disabled watchdog (threshold <= 0) is a deliberate, documented override, so an
    inert drill there is expected rather than surprising."""
    raw_inject = str(env.get("ASSAY_INJECT_STALL_AFTER", "")).strip()
    if not raw_inject:
        return None
    try:
        inject = float(raw_inject)
        threshold = float(str(env.get("ASSAY_STALL_SECONDS", "1800")).strip() or "1800")
    except ValueError:
        return None
    if inject <= 0 or threshold <= 0 or inject > threshold:
        return None
    return (f"assay.watchdog: WARNING - ASSAY_INJECT_STALL_AFTER={inject:g} is not "
            f"longer than ASSAY_STALL_SECONDS={threshold:g}, so the injected stall "
            "ends before the watchdog can trip and the kill-chain drill is INERT. "
            "Raise ASSAY_INJECT_STALL_AFTER above ASSAY_STALL_SECONDS to rehearse "
            "the kill, or lower ASSAY_STALL_SECONDS.")


def build_eval_watchdog(child_pid, raw_log_path, heartbeat, env):  # pragma: no cover
    """Production factory: after os.setpgid(0,0) in the child, its pgid == child_pid.
    ASSAY_STALL_SECONDS (default 1800; 0 == limitless) sets the threshold."""
    drill = stall_drill_warning(env)
    if drill:
        print(drill, file=sys.stderr)
    raw_threshold = env.get("ASSAY_STALL_SECONDS", "1800")
    try:
        threshold = float(raw_threshold)
    except ValueError:
        # A malformed override must never raise HERE: run_eval has already started
        # the paid child by the time this factory is called, so a raise would only
        # surface after the child is running, leaving it to the _bounded_join
        # SIGKILL backstop (~2 min) instead of a clean, immediate, named warning.
        print(f"assay.watchdog: WARNING - ASSAY_STALL_SECONDS={raw_threshold!r} is "
              f"not a valid number; falling back to 1800.0", file=sys.stderr)
        threshold = 1800.0
    if threshold <= 0:
        print(f"assay.watchdog: ASSAY_STALL_SECONDS={threshold} - progress-stall watchdog DISABLED (limitless)", file=sys.stderr)
    signals = [
        _Signal("stdout",
                (lambda: _read_mtime(raw_log_path)) if raw_log_path else (lambda: None),
                lambda prev, cur: cur > prev),
        _Signal("gpu", _read_gpu_util, lambda prev, cur: cur > 0),
        _Signal("cpu", lambda: _pgid_cpu_jiffies(child_pid),
                lambda prev, cur: cur > prev),
    ]
    return StallWatchdog(signals, lambda: _kill_stalled(child_pid),
                         threshold=threshold, heartbeat=heartbeat)
