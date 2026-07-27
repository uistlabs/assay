import os
import signal
import time

from assay import watchdog


def test_read_mtime_missing_returns_none(tmp_path):
    assert watchdog._read_mtime(str(tmp_path / "nope")) is None


def test_read_mtime_advances_on_write(tmp_path):
    p = tmp_path / "raw.log"
    p.write_text("a\n")
    t0 = watchdog._read_mtime(str(p))
    assert t0 is not None
    os.utime(str(p), (t0 + 10, t0 + 10))  # simulate a later write deterministically
    assert watchdog._read_mtime(str(p)) > t0


def test_pgid_cpu_jiffies_sums_matching_group(tmp_path):
    # Fake /proc: two pids in pgid 42 (utime+stime = 5+3 and 1+1), one in pgid 99.
    def _mkstat(pid, comm, pgrp, utime, stime):
        d = tmp_path / str(pid)
        d.mkdir()
        # field layout after ")": state ppid pgrp session ... (utime=idx11, stime=idx12)
        after = ["R", "1", str(pgrp)] + ["0"] * 8 + [str(utime), str(stime)]
        (d / "stat").write_text(f"{pid} ({comm}) " + " ".join(after) + "\n")
    _mkstat(100, "child", 42, 5, 3)
    _mkstat(101, "engine core", 42, 1, 1)  # comm with a space, exercises rsplit(')')
    _mkstat(200, "other", 99, 9, 9)
    assert watchdog._pgid_cpu_jiffies(42, proc_root=str(tmp_path)) == 10


def test_pgid_cpu_jiffies_no_match_returns_none(tmp_path):
    assert watchdog._pgid_cpu_jiffies(42, proc_root=str(tmp_path)) is None


from assay.watchdog import StallWatchdog, _Signal


def _counter_signal(name, values):
    """A monotonic-counter signal that yields successive values from `values`, then
    repeats the last (mtime/cpu-style: progress == strictly increased)."""
    state = {"i": 0}
    def probe():
        i = min(state["i"], len(values) - 1)
        state["i"] += 1
        return values[i]
    return _Signal(name, probe, lambda prev, cur: cur > prev)


def test_no_kill_while_a_signal_advances(monkeypatch):
    clock = {"t": 0.0}
    killed = {"n": 0}
    # stdout advances every tick; cpu is flat - must NOT kill (one signal moving).
    wd = StallWatchdog(
        signals=[_counter_signal("stdout", [1, 2, 3, 4, 5]),
                 _counter_signal("cpu", [7, 7, 7, 7, 7])],
        on_stall=lambda: killed.__setitem__("n", killed["n"] + 1),
        threshold=100.0, clock=lambda: clock["t"])
    for _ in range(5):
        clock["t"] += 60.0
        assert wd._tick() is False
    assert killed["n"] == 0


def test_kill_when_all_signals_flat_past_threshold():
    clock = {"t": 0.0}
    killed = {"n": 0}
    wd = StallWatchdog(
        signals=[_counter_signal("stdout", [1, 1, 1]),
                 _counter_signal("cpu", [7, 7, 7])],
        on_stall=lambda: killed.__setitem__("n", killed["n"] + 1),
        threshold=100.0, clock=lambda: clock["t"])
    clock["t"] = 10.0;  assert wd._tick() is False   # baseline, last_progress_t=10
    clock["t"] = 60.0;  assert wd._tick() is False   # flat 50s < 100
    clock["t"] = 180.0; assert wd._tick() is True     # flat 170s > 100 -> kill
    assert killed["n"] == 1


def test_threshold_zero_is_limitless():
    clock = {"t": 0.0}
    killed = {"n": 0}
    wd = StallWatchdog(
        signals=[_counter_signal("stdout", [1, 1, 1])],
        on_stall=lambda: killed.__setitem__("n", killed["n"] + 1),
        threshold=0.0, clock=lambda: clock["t"])
    for _ in range(3):
        clock["t"] += 10_000.0
        assert wd._tick() is False
    assert killed["n"] == 0


def test_never_kills_when_no_signal_is_available():
    # All probes return None (unavailable) -> never kill blind.
    killed = {"n": 0}
    wd = StallWatchdog(
        signals=[_Signal("x", lambda: None, lambda p, c: c > p)],
        on_stall=lambda: killed.__setitem__("n", killed["n"] + 1),
        threshold=1.0, clock=lambda: 10_000.0)
    assert wd._tick() is False
    assert killed["n"] == 0


def test_emits_progress_heartbeat_on_forward_motion(tmp_path):
    from assay.heartbeat import Heartbeat
    hb = Heartbeat(str(tmp_path / "hb.log"))
    clock = {"t": 0.0}
    wd = StallWatchdog(
        signals=[_counter_signal("stdout", [1, 2, 3])],
        on_stall=lambda: None, threshold=100.0, heartbeat=hb,
        clock=lambda: clock["t"])
    clock["t"] = 30.0; wd._tick()   # baseline (first obs -> no motion beat)
    clock["t"] = 60.0; wd._tick()   # 2 > 1 -> forward motion -> "progress" beat
    assert "progress" in (tmp_path / "hb.log").read_text()


def test_progress_beat_rate_limited_to_one_per_interval(tmp_path):
    """A blocking volume write on EVERY forward-motion tick (~30s) hangs the
    watchdog thread in its own emit on a volume stall, blinding the stall detector.
    Across many ticks inside one beat_interval, at most one 'progress' beat must
    reach the heartbeat log - not one per tick."""
    from assay.heartbeat import Heartbeat
    hb = Heartbeat(str(tmp_path / "hb.log"))
    clock = {"t": 0.0}
    wd = StallWatchdog(
        signals=[_counter_signal("stdout", list(range(1, 41)))],
        on_stall=lambda: None, threshold=1_000_000.0, heartbeat=hb,
        clock=lambda: clock["t"], beat_interval=600.0)
    for _ in range(20):
        clock["t"] += 30.0  # 20 ticks * 30s == 600s, all inside one beat_interval
        wd._tick()
    log_text = (tmp_path / "hb.log").read_text()
    assert log_text.count("progress") <= 1, (
        f"expected at most one rate-limited progress beat, got:\n{log_text}")


def test_progress_beat_fires_again_after_interval_elapses(tmp_path):
    """The rate limit bounds frequency, it does not silence the beat forever: once
    beat_interval has elapsed, the next forward-motion tick beats again."""
    from assay.heartbeat import Heartbeat
    hb = Heartbeat(str(tmp_path / "hb.log"))
    clock = {"t": 0.0}
    wd = StallWatchdog(
        signals=[_counter_signal("stdout", [1, 2, 3])],
        on_stall=lambda: None, threshold=1_000_000.0, heartbeat=hb,
        clock=lambda: clock["t"], beat_interval=100.0)
    clock["t"] = 10.0; wd._tick()    # baseline, no motion yet (first obs)
    clock["t"] = 20.0; wd._tick()    # 2 > 1 -> motion -> first beat (due: no prior beat)
    clock["t"] = 30.0; wd._tick()    # 3 > 2 -> motion but within 100s -> rate-limited
    clock["t"] = 200.0                # past beat_interval from the first beat at t=20
    wd.signals[0].last_value = 3     # force one more forward-motion observation
    wd.signals[0].probe = lambda: 4
    wd._tick()
    assert (tmp_path / "hb.log").read_text().count("progress") == 2


def test_progress_beat_is_best_effort_and_never_raises(tmp_path):
    """A raising heartbeat.emit (e.g. a hung/erroring volume write surfaced as an
    exception) must never propagate out of _tick and break the watchdog loop."""
    clock = {"t": 0.0}

    class _ExplodingHeartbeat:
        def emit(self, *a, **kw):
            raise OSError("volume gone")

    wd = StallWatchdog(
        signals=[_counter_signal("stdout", [1, 2])],
        on_stall=lambda: None, threshold=1_000_000.0,
        heartbeat=_ExplodingHeartbeat(), clock=lambda: clock["t"])
    clock["t"] = 10.0; wd._tick()
    clock["t"] = 40.0
    assert wd._tick() is False  # must not raise


def test_build_eval_watchdog_falls_back_on_malformed_stall_seconds(capsys):
    from assay.watchdog import build_eval_watchdog
    wd = build_eval_watchdog(12345, None, None, {"ASSAY_STALL_SECONDS": "garbage"})
    assert wd.threshold == 1800.0
    captured = capsys.readouterr()
    assert "garbage" in captured.err


def test_build_eval_watchdog_accepts_well_formed_stall_seconds():
    from assay.watchdog import build_eval_watchdog
    wd = build_eval_watchdog(12345, None, None, {"ASSAY_STALL_SECONDS": "900"})
    assert wd.threshold == 900.0


def test_build_eval_watchdog_negative_stall_seconds_surfaces_disabled_note(capsys):
    from assay.watchdog import build_eval_watchdog
    wd = build_eval_watchdog(12345, None, None, {"ASSAY_STALL_SECONDS": "-5"})
    assert wd.threshold == -5.0
    captured = capsys.readouterr()
    assert "DISABLED (limitless)" in captured.err
    assert "-5" in captured.err


def test_build_eval_watchdog_zero_stall_seconds_surfaces_disabled_note(capsys):
    from assay.watchdog import build_eval_watchdog
    wd = build_eval_watchdog(12345, None, None, {"ASSAY_STALL_SECONDS": "0"})
    assert wd.threshold == 0.0
    captured = capsys.readouterr()
    assert "DISABLED (limitless)" in captured.err


def test_kill_stalled_pgid_zero_sends_no_signals():
    # pgid 0 is the caller's OWN group - must never be touched (self-guillotine:
    # os.kill(0, SIG) targets the SENDER's own process group per POSIX). Every
    # collaborator is a fake and the collector raises if ever invoked, so this
    # stays inert no matter what path the guard does or doesn't take.
    calls = []
    escapee_kills = []
    def signaler(pgid, sig):
        calls.append((pgid, sig))
    def killer(pid, sig):
        escapee_kills.append((pid, sig))
    def collector(root_pid, proc_root):
        raise AssertionError("must not walk /proc for pgid <= 1")
    watchdog._kill_stalled(0, signaler=signaler, killer=killer,
                           collector=collector, sleep=lambda s: None)
    assert calls == []
    assert escapee_kills == []


def test_kill_stalled_pgid_one_sends_no_signals():
    # pgid 1 (init) is equally nonsensical as a kill root - guard covers <= 1.
    calls = []
    escapee_kills = []
    def signaler(pgid, sig):
        calls.append((pgid, sig))
    def killer(pid, sig):
        escapee_kills.append((pid, sig))
    def collector(root_pid, proc_root):
        raise AssertionError("must not walk /proc for pgid <= 1")
    watchdog._kill_stalled(1, signaler=signaler, killer=killer,
                           collector=collector, sleep=lambda s: None)
    assert calls == []
    assert escapee_kills == []


def test_kill_stalled_stands_down_when_group_already_gone():
    # Liveness probe (signal 0) says the group is already gone - stand down,
    # never send SIGUSR1/SIGKILL, never touch the (would-be) escapee.
    calls = []
    escapee_kills = []
    def signaler(pgid, sig):
        calls.append((pgid, sig))
        if sig == 0:
            raise ProcessLookupError
    def killer(pid, sig):
        escapee_kills.append((pid, sig))
    def collector(root_pid, proc_root):
        return [999]  # must never be reached - liveness probe short-circuits first
    watchdog._kill_stalled(42, signaler=signaler, killer=killer,
                            collector=collector, sleep=lambda s: None)
    assert calls == [(42, 0)]
    assert escapee_kills == []


def test_kill_stalled_snapshots_descendants_before_group_kill():
    # A setsid-escaped grandchild reparents to pid 1 the instant the group dies,
    # so the descendant snapshot must be taken BEFORE the group SIGKILL. Model
    # that here: the fake collector returns the escapee only while the group is
    # still alive, and returns nothing once the group SIGKILL has fired - a
    # "collect after kill" regression would see group_killed already True and
    # fail to find (and thus never signal) the escapee.
    calls = []
    escapee_kills = []
    state = {"group_killed": False}

    def signaler(pgid, sig):
        calls.append((pgid, sig))
        if sig == signal.SIGKILL:
            state["group_killed"] = True

    def collector(root_pid, proc_root):
        return [] if state["group_killed"] else [777]

    def killer(pid, sig):
        escapee_kills.append((pid, sig))

    watchdog._kill_stalled(42, signaler=signaler, killer=killer,
                            collector=collector, sleep=lambda s: None)

    assert (42, signal.SIGKILL) in calls
    assert escapee_kills == [(777, signal.SIGKILL)]


def test_stall_drill_warning_fires_when_injection_cannot_trip_the_watchdog():
    """F-019: ASSAY_INJECT_STALL_AFTER is the Phase-B wedge rehearsal. If the injected
    stall is not LONGER than the watchdog threshold, the watchdog can never fire and
    the drill is silently inert - the operator believes they tested the kill chain."""
    msg = watchdog.stall_drill_warning({"ASSAY_INJECT_STALL_AFTER": "60",
                                        "ASSAY_STALL_SECONDS": "1800"})
    assert msg is not None
    assert "ASSAY_INJECT_STALL_AFTER" in msg and "ASSAY_STALL_SECONDS" in msg
    assert msg.isascii()


def test_stall_drill_warning_silent_when_the_drill_can_actually_fire():
    assert watchdog.stall_drill_warning({"ASSAY_INJECT_STALL_AFTER": "2400",
                                         "ASSAY_STALL_SECONDS": "1800"}) is None


def test_stall_drill_warning_silent_when_no_drill_configured():
    """Production: the knob is unset, so there is nothing to warn about."""
    assert watchdog.stall_drill_warning({"ASSAY_STALL_SECONDS": "1800"}) is None
    assert watchdog.stall_drill_warning({}) is None


def test_stall_drill_warning_silent_when_watchdog_disabled():
    """threshold 0 is the documented limitless override - the watchdog is off on
    purpose, so an inert drill is not a surprise worth warning about."""
    assert watchdog.stall_drill_warning({"ASSAY_INJECT_STALL_AFTER": "60",
                                         "ASSAY_STALL_SECONDS": "0"}) is None


def test_stall_drill_warning_tolerates_malformed_values():
    """A malformed knob must never raise here - this runs on the paid path."""
    assert watchdog.stall_drill_warning({"ASSAY_INJECT_STALL_AFTER": "soon",
                                         "ASSAY_STALL_SECONDS": "1800"}) is None
    assert watchdog.stall_drill_warning({"ASSAY_INJECT_STALL_AFTER": "60",
                                         "ASSAY_STALL_SECONDS": "never"}) is None
