"""Tests for stall-injection knob and other evaluate.py helpers."""


def test_inject_stall_noop_when_unset():
    from assay.evaluate import _inject_stall_if_configured
    calls = []
    _inject_stall_if_configured(env={}, sleep=calls.append)
    assert calls == []


def test_inject_stall_sleeps_seconds_when_set():
    from assay.evaluate import _inject_stall_if_configured
    calls = []
    _inject_stall_if_configured(env={"ASSAY_INJECT_STALL_AFTER": "3"}, sleep=calls.append)
    assert calls == [3.0]


def test_inject_stall_ignores_zero_blank_and_garbage():
    from assay.evaluate import _inject_stall_if_configured
    for val in ("0", "", "   ", "abc"):
        calls = []
        _inject_stall_if_configured(env={"ASSAY_INJECT_STALL_AFTER": val}, sleep=calls.append)
        assert calls == [], val
