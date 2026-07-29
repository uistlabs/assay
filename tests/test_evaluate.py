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


def test_model_args_without_cap_is_byte_identical_to_today():
    from assay.evaluate import _model_args
    assert _model_args("/w", 0.85, None) == (
        "pretrained=/w,dtype=auto,gpu_memory_utilization=0.85")


def test_model_args_with_cap_appends_max_model_len():
    from assay.evaluate import _model_args
    assert _model_args("/w", 0.85, 36864) == (
        "pretrained=/w,dtype=auto,gpu_memory_utilization=0.85,max_model_len=36864")
