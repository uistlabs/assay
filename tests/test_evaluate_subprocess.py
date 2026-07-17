"""run_eval subprocess plumbing (torch/lm_eval-free).

run_eval spawns each lm-eval run in its own subprocess so vLLM teardown is
guaranteed by process exit (the eval->eval GPU-residue fix). These tests
exercise the parent-side plumbing -- result marshalling, child-exception
propagation, and dead-child detection -- by monkeypatching _eval_child and
using mp_context="fork": fork shares the patched module state with the child,
whereas production "spawn" would re-import the REAL module (and lm_eval, and
a GPU) in a fresh interpreter. The child bodies below therefore run in a
separate PROCESS, exactly like production, minus the GPU work.
"""
import os

import pytest

from assay import evaluate
from assay.heartbeat import Heartbeat


def _child_ok(conn, model_path, tasks, gpu_mem_util):
    # Echo the inputs back so the test proves they crossed the process
    # boundary intact (tasks list, knob value), not just that SOMETHING ran.
    conn.send(("ok", {"path": model_path, "tasks": tasks, "gmu": gpu_mem_util}))
    conn.close()


def _child_err(conn, model_path, tasks, gpu_mem_util):
    conn.send(("err", "Traceback (most recent call last):\nValueError: boom"))
    conn.close()
    raise SystemExit(1)


def _child_dies_silently(conn, model_path, tasks, gpu_mem_util):
    # Simulate a hard native crash (CUDA abort, OOM kill): no message, no
    # clean close, nonzero exit. os._exit skips interpreter cleanup entirely.
    os._exit(3)


def test_run_eval_returns_child_payload(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluate, "_eval_child", _child_ok)
    hb = Heartbeat(str(tmp_path / "hb.log"))
    raw = evaluate.run_eval("/models/m", ["gsm8k", "wikitext"], hb,
                            gpu_mem_util=0.7, mp_context="fork")
    assert raw == {"path": "/models/m", "tasks": ["gsm8k", "wikitext"], "gmu": 0.7}
    log = (tmp_path / "hb.log").read_text()
    assert "gpu_mem_util=0.7" in log and "done /models/m" in log


def test_run_eval_raises_with_child_traceback(monkeypatch):
    monkeypatch.setattr(evaluate, "_eval_child", _child_err)
    with pytest.raises(RuntimeError, match=r"(?s)failed for /models/m.*ValueError: boom"):
        evaluate.run_eval("/models/m", ["gsm8k"], mp_context="fork")


def test_run_eval_raises_on_silent_child_death(monkeypatch):
    # EOF on the pipe with no status must surface as an actionable error
    # carrying the exitcode -- NOT hang and NOT return None.
    monkeypatch.setattr(evaluate, "_eval_child", _child_dies_silently)
    with pytest.raises(RuntimeError, match=r"died without a result \(exitcode=3\)"):
        evaluate.run_eval("/models/m", ["gsm8k"], mp_context="fork")


def test_run_eval_default_gpu_mem_util_matches_config_default():
    # run_eval's fallback default must agree with config's ASSAY_GPU_MEM_UTIL
    # default so a caller that omits the arg behaves like the documented knob.
    import inspect

    from assay.config import load_config

    sig = inspect.signature(evaluate.run_eval)
    cfg = load_config({"ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16"})
    assert sig.parameters["gpu_mem_util"].default == cfg.gpu_mem_util
