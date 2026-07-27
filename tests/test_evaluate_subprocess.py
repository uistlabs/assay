"""run_eval subprocess plumbing (torch/lm_eval-free).

run_eval spawns each lm-eval run in its own subprocess so vLLM teardown is
guaranteed by process exit (the eval->eval GPU-residue fix). These tests
exercise the parent-side plumbing - result marshalling, child-exception
propagation, and dead-child detection - by monkeypatching _eval_child and
using mp_context="fork": fork shares the patched module state with the child,
whereas production "spawn" would re-import the REAL module (and lm_eval, and
a GPU) in a fresh interpreter. The child bodies below therefore run in a
separate PROCESS, exactly like production, minus the GPU work.
"""
import json
import os

import pytest

from assay import evaluate
from assay.heartbeat import Heartbeat


def _child_ok(conn, model_path, tasks, gpu_mem_util,
             apply_chat_template=False, fewshot_as_multiturn=False,
             gen_kwargs=None, system_instruction=None,
             include_path=None, repeats=None, limit=None, persist_path=None, capture_per_item=False):
    # Echo the inputs back so the test proves they crossed the process
    # boundary intact (tasks list, knob value), not just that SOMETHING ran.
    conn.send(("ok", {"path": model_path, "tasks": tasks, "gmu": gpu_mem_util}))
    conn.close()


def _child_err(conn, model_path, tasks, gpu_mem_util,
               apply_chat_template=False, fewshot_as_multiturn=False,
               gen_kwargs=None, system_instruction=None,
               include_path=None, repeats=None, limit=None, persist_path=None, capture_per_item=False):
    conn.send(("err", "Traceback (most recent call last):\nValueError: boom"))
    conn.close()
    raise SystemExit(1)


def _child_dies_silently(conn, model_path, tasks, gpu_mem_util,
                         apply_chat_template=False, fewshot_as_multiturn=False,
                         gen_kwargs=None, system_instruction=None,
                         include_path=None, repeats=None, limit=None, persist_path=None, capture_per_item=False):
    # Simulate a hard native crash (CUDA abort, OOM kill): no message, no
    # clean close, nonzero exit. os._exit skips interpreter cleanup entirely.
    os._exit(3)


def _child_echo_kwargs(conn, model_path, tasks, gpu_mem_util,
                       apply_chat_template, fewshot_as_multiturn, gen_kwargs, system_instruction,
                       include_path=None, repeats=None, limit=None, persist_path=None, capture_per_item=False):
    conn.send(("ok", {
        "apply_chat_template": apply_chat_template,
        "fewshot_as_multiturn": fewshot_as_multiturn,
        "gen_kwargs": gen_kwargs,
        "system_instruction": system_instruction,
    }))
    conn.close()


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
    # carrying the exitcode - NOT hang and NOT return None.
    monkeypatch.setattr(evaluate, "_eval_child", _child_dies_silently)
    with pytest.raises(RuntimeError, match=r"died without a result \(exitcode=3\)"):
        evaluate.run_eval("/models/m", ["gsm8k"], mp_context="fork")


def _child_echo_limit(conn, model_path, tasks, gpu_mem_util,
                      apply_chat_template=False, fewshot_as_multiturn=False,
                      gen_kwargs=None, system_instruction=None,
                      include_path=None, repeats=None, limit=None, persist_path=None, capture_per_item=False):
    conn.send(("ok", {"limit": limit}))
    conn.close()


def test_run_eval_forwards_limit(monkeypatch):
    monkeypatch.setattr(evaluate, "_eval_child", _child_echo_limit)
    raw = evaluate.run_eval("/m", ["gsm8k"], mp_context="fork", limit=1)
    assert raw == {"limit": 1}


def test_run_eval_forwards_chat_mode_kwargs(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluate, "_eval_child", _child_echo_kwargs)
    raw = evaluate.run_eval(
        "/models/m", ["aime24"], mp_context="fork",
        apply_chat_template=True, fewshot_as_multiturn=True,
        gen_kwargs={"temperature": 0.6, "top_p": 0.95}, system_instruction=None,
    )
    assert raw["apply_chat_template"] is True
    assert raw["fewshot_as_multiturn"] is True
    assert raw["gen_kwargs"] == {"temperature": 0.6, "top_p": 0.95}
    assert raw["system_instruction"] is None


def test_run_eval_forwards_include_path_and_repeats(monkeypatch):
    # NOTE: deviates from the task brief's literal snippet, which captured
    # include_path/repeats into an external `dict` mutated inside the child.
    # Under mp_context="fork" the child is a separate address space (COW): a
    # plain-dict mutation there is invisible to the parent, so that version
    # fails with KeyError regardless of what run_eval/_eval_child do - it is
    # not testing what it claims to. Following the file's own established
    # pattern (see _child_echo_kwargs below), the values must cross the
    # process boundary over conn.send/recv like every other assertion here.
    import assay.evaluate as ev

    def fake_child(conn, model_path, tasks, gpu_mem_util, apply_chat_template=False,
                   fewshot_as_multiturn=False, gen_kwargs=None, system_instruction=None,
                   include_path=None, repeats=None, limit=None, persist_path=None, capture_per_item=False):
        conn.send(("ok", {"include_path": include_path, "repeats": repeats}))
        conn.close()

    monkeypatch.setattr(ev, "_eval_child", fake_child)
    raw = ev.run_eval("model", ["aime24_avg"], None, 0.85, mp_context="fork",
                      include_path="/pkg/lm_eval_tasks", repeats={"aime24_avg": 16})
    assert raw["include_path"] == "/pkg/lm_eval_tasks"
    assert raw["repeats"] == {"aime24_avg": 16}


def _make_external_task_fn():
    """Mimic how lm-eval loads an assay-owned task util: `spec_from_file_location`
    names the module by its ABSOLUTE PATH (external modules aren't under /lm_eval/tasks/),
    so the function's __module__ is a path registered only in THIS interpreter's
    sys.modules. It pickles here (the child that ran the eval) but a fresh spawn
    interpreter (the parent) can't import it -> exactly last night's crash:
    `ModuleNotFoundError: No module named '/app/src/assay/lm_eval_tasks/utils'`."""
    import sys
    import types
    modname = "/tmp/assay_fake_task_dir/utils"  # path-as-module-name, like lm-eval
    mod = types.ModuleType(modname)

    def passthrough(resps, docs):
        return resps

    passthrough.__module__ = modname
    passthrough.__qualname__ = "passthrough"
    mod.passthrough = passthrough
    sys.modules[modname] = mod
    return passthrough


def _unpickle_in_child(conn, blob):
    import pickle
    try:
        pickle.loads(blob)
        conn.send("ok")
    except BaseException as exc:  # noqa: BLE001
        conn.send(f"{type(exc).__name__}: {exc}")
    conn.close()


def test_eval_payload_survives_spawn_unpickle_with_external_task_fn():
    # Regression for the v0.4.1 avg@K metal crash: lm-eval embeds the avg task's
    # filter_fn (our external utils.passthrough) in raw["configs"], and _eval_child
    # pickles raw back to the parent over the Pipe. The function's __module__ is an
    # unimportable path, so the PARENT's spawn-interpreter unpickle died 4.5h in.
    import multiprocessing as mp
    import pickle

    from assay import evaluate

    raw = {
        "results": {"aime24_avg": {"exact_match,avg": 0.5333, "alias": "aime24_avg"}},
        "configs": {"aime24_avg": {"filter_list": [
            {"filter": [{"filter_fn": _make_external_task_fn()}]}]}},
    }
    clean = evaluate._sanitize_raw(raw)

    ctx = mp.get_context("spawn")
    recv, send = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_unpickle_in_child, args=(send, pickle.dumps(clean)))
    proc.start()
    send.close()
    msg = recv.recv()
    recv.close()
    proc.join()

    assert msg == "ok", f"sanitized payload still not spawn-safe: {msg}"
    # the gate's numbers must survive the sanitize untouched
    assert clean["results"]["aime24_avg"]["exact_match,avg"] == 0.5333


def test_sanitize_raw_strips_callables_but_keeps_scalars():
    from assay import evaluate

    def bogus_filter(resps, docs):
        return resps

    raw = {
        "results": {"t": {"exact_match,avg": 0.42, "sample_len": 3, "alias": "t"}},
        "configs": {"t": {"filter_list": [{"filter": [{"filter_fn": bogus_filter}]}]}},
    }
    clean = evaluate._sanitize_raw(raw)

    def has_callable(obj):
        if callable(obj) and not isinstance(obj, (str, bytes)):
            return True
        if isinstance(obj, dict):
            return any(has_callable(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return any(has_callable(v) for v in obj)
        return False

    assert not has_callable(clean), "a callable survived sanitize -> pickle-boundary hazard"
    assert clean["results"]["t"]["exact_match,avg"] == 0.42
    assert clean["results"]["t"]["sample_len"] == 3


def _dummy_filter_fn(resps, docs):
    return resps


def _fake_simple_evaluate(**kwargs):
    # Mimic lm-eval returning a raw dict that embeds a live filter_fn in configs
    # (the exact shape that crashed the parent unpickle). Module-level so fork's
    # child can reference it.
    return {
        "results": {"aime24_avg": {"exact_match,avg": 0.5333}},
        "configs": {"aime24_avg": {"filter_list": [
            {"filter": [{"filter_fn": _dummy_filter_fn}]}]}},
    }


def test_real_eval_child_sanitizes_before_send(monkeypatch):
    # Pins the _sanitize_raw CALL SITE in the real _eval_child: run the ACTUAL
    # _eval_child (fork, so the stubbed lm_eval in sys.modules is visible in the
    # child) with a stubbed simple_evaluate that returns a raw carrying a live
    # filter_fn. The payload the parent receives must contain NO callable. Deleting
    # the _sanitize_raw(...) call would let the function through and fail this test
    # - which the direct-call regression tests would NOT catch.
    import sys
    import types

    fake_lm_eval = types.ModuleType("lm_eval")
    fake_lm_eval.simple_evaluate = _fake_simple_evaluate
    fake_tasks = types.ModuleType("lm_eval.tasks")
    fake_tasks.TaskManager = object
    fake_tasks.get_task_dict = lambda *a, **k: {}
    monkeypatch.setitem(sys.modules, "lm_eval", fake_lm_eval)
    monkeypatch.setitem(sys.modules, "lm_eval.tasks", fake_tasks)

    payload = evaluate.run_eval("/models/m", ["aime24_avg"], None, 0.85,
                                mp_context="fork", include_path=None, repeats=None)

    def has_callable(obj):
        if callable(obj) and not isinstance(obj, (str, bytes)):
            return True
        if isinstance(obj, dict):
            return any(has_callable(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return any(has_callable(v) for v in obj)
        return False

    assert not has_callable(payload), "real _eval_child sent a callable -> sanitize call site missing"
    assert payload["results"]["aime24_avg"]["exact_match,avg"] == 0.5333


def test_assay_task_dir_points_at_shipped_yaml():
    import os
    from assay.evaluate import assay_task_dir
    d = assay_task_dir()
    assert os.path.isfile(os.path.join(d, "aime24_avg.yaml"))
    assert os.path.isfile(os.path.join(d, "aime25_avg.yaml"))


def test_mixed_task_specs_objects_for_repeats_names_for_rest():
    # The defect-prone part of the group-task fix: a task with a repeats override
    # must go to simple_evaluate as its pre-built Task object (so set_config
    # sticks); every other task - notably a GROUP task like mmlu, whose subtasks
    # get silently dropped if handed as list(task_dict.values()) instead of a bare
    # name - must be passed through as its name string for lm-eval to resolve.
    from assay.evaluate import _mixed_task_specs
    sentinel = object()
    override = {"aime24_avg": sentinel}
    specs = _mixed_task_specs(["aime24_avg", "minerva_math500", "wikitext"], override)
    assert specs == [sentinel, "minerva_math500", "wikitext"]


def test_mixed_task_specs_empty_override_returns_all_names():
    from assay.evaluate import _mixed_task_specs
    assert _mixed_task_specs(["mmlu", "gsm8k"], {}) == ["mmlu", "gsm8k"]


def test_run_eval_default_gpu_mem_util_matches_config_default():
    # run_eval's fallback default must agree with config's ASSAY_GPU_MEM_UTIL
    # default so a caller that omits the arg behaves like the documented knob.
    import inspect

    from assay.config import load_config

    sig = inspect.signature(evaluate.run_eval)
    cfg = load_config({"ASSAY_WEIGHTS_PATH": "/vol/weights", "ASSAY_CHECKPOINT_REPO": "myorg/Model-NVFP4A16"})
    assert sig.parameters["gpu_mem_util"].default == cfg.gpu_mem_util


def test_real_eval_child_forces_log_samples_false(monkeypatch):
    # Cross the kwarg back over the Pipe so we can assert on it in the parent.
    import sys
    import types

    def _fake_simple_evaluate(**kwargs):
        return {"results": {"log_samples_seen": {"v": kwargs.get("log_samples")}}}

    fake_lm_eval = types.ModuleType("lm_eval")
    fake_lm_eval.simple_evaluate = _fake_simple_evaluate
    fake_tasks = types.ModuleType("lm_eval.tasks")
    fake_tasks.TaskManager = object
    fake_tasks.get_task_dict = lambda *a, **k: {}
    monkeypatch.setitem(sys.modules, "lm_eval", fake_lm_eval)
    monkeypatch.setitem(sys.modules, "lm_eval.tasks", fake_tasks)

    payload = evaluate.run_eval("/m", ["gsm8k"], None, 0.85, mp_context="fork")
    assert payload["results"]["log_samples_seen"]["v"] is False


def test_real_eval_child_persists_before_send(monkeypatch, tmp_path):
    # Real _eval_child under fork with stubbed lm_eval + a persist_path: the sanitized
    # result must be on disk (written in-child before the Pipe send).
    import sys
    import types
    fake_lm_eval = types.ModuleType("lm_eval")
    fake_lm_eval.simple_evaluate = lambda **k: {"results": {"gsm8k": {"exact_match,none": 0.5}}}
    fake_tasks = types.ModuleType("lm_eval.tasks")
    fake_tasks.TaskManager = object
    fake_tasks.get_task_dict = lambda *a, **k: {}
    monkeypatch.setitem(sys.modules, "lm_eval", fake_lm_eval)
    monkeypatch.setitem(sys.modules, "lm_eval.tasks", fake_tasks)
    persist = tmp_path / "child.json"
    payload = evaluate.run_eval("/m", ["gsm8k"], None, 0.85, mp_context="fork",
                                persist_path=str(persist))
    assert payload["results"]["gsm8k"]["exact_match,none"] == 0.5
    assert json.loads(persist.read_text())["results"]["gsm8k"]["exact_match,none"] == 0.5


def _child_persists_then_dies(conn, model_path, tasks, gpu_mem_util,
                              apply_chat_template=False, fewshot_as_multiturn=False,
                              gen_kwargs=None, system_instruction=None,
                              include_path=None, repeats=None, limit=None,
                              persist_path=None, capture_per_item=False):
    # Completed the eval and PERSISTED, then died before/without a clean send
    # (the wedge-then-kill signature). The parent must recover from persist_path.
    with open(persist_path, "w") as fh:
        json.dump({"results": {"gsm8k": {"exact_match,none": 0.9}}}, fh)
    os._exit(7)


def test_run_eval_recovers_persisted_result_on_child_death(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluate, "_eval_child", _child_persists_then_dies)
    persist = tmp_path / "child.json"
    raw = evaluate.run_eval("/m", ["gsm8k"], None, 0.85, mp_context="fork",
                            persist_path=str(persist))
    assert raw == {"results": {"gsm8k": {"exact_match,none": 0.9}}}


def test_run_eval_raises_when_dead_and_no_persist(monkeypatch, tmp_path):
    # Dead child, persist_path given but the child never wrote it -> hard failure,
    # NOT a silent None.
    monkeypatch.setattr(evaluate, "_eval_child", _child_dies_silently)
    with pytest.raises(RuntimeError, match=r"died without a result"):
        evaluate.run_eval("/m", ["gsm8k"], None, 0.85, mp_context="fork",
                          persist_path=str(tmp_path / "never-written.json"))


def test_run_eval_does_not_return_stale_persist_on_child_death(monkeypatch, tmp_path):
    # A prior eval/run reused persist_path and left a STALE result there. THIS
    # child dies without writing anything. run_eval must NOT hand back the
    # stale file - it must raise "died without a result", exactly as if no
    # file were there at all. Regression for the certification-pipeline
    # silent-wrong-data hazard: reading a leftover persist_path as if it were
    # THIS eval's result.
    persist = tmp_path / "child.json"
    persist.write_text(json.dumps({"results": {"stale": True}}))
    monkeypatch.setattr(evaluate, "_eval_child", _child_dies_silently)
    with pytest.raises(RuntimeError, match=r"died without a result"):
        evaluate.run_eval("/m", ["gsm8k"], None, 0.85, mp_context="fork",
                          persist_path=str(persist))


def test_run_eval_starts_and_stops_watchdog_with_child_pid(monkeypatch):
    # The factory must receive the live child pid and get started then stopped.
    events = []

    class FakeWD:
        def __init__(self, pid): self.pid = pid
        def start(self): events.append(("start", self.pid))
        def stop(self): events.append(("stop", self.pid))

    def factory(child_pid, heartbeat):
        events.append(("build", child_pid))
        return FakeWD(child_pid)

    monkeypatch.setattr(evaluate, "_eval_child", _child_ok)
    evaluate.run_eval("/m", ["gsm8k"], None, 0.85, mp_context="fork",
                      watchdog_factory=factory)
    kinds = [e[0] for e in events]
    assert kinds == ["build", "start", "stop"]
    assert events[0][1] > 0  # a real pid was passed


class _FakeWatchdog:
    """Minimal StallWatchdog stand-in: records start/stop and reports whether it
    fired the kill. `killed` is the real class's public flag (watchdog.py)."""

    def __init__(self, killed):
        self.killed = killed
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


def test_dead_child_after_watchdog_kill_names_the_watchdog_not_cuda(monkeypatch):
    """F-016: the dead-child headline must not send a 2 AM operator hunting a CUDA
    error or an OOM kill when the StallWatchdog is what killed the eval. Metal
    Phase B proved this exact misattribution on the watchdog-kill path."""
    monkeypatch.setattr(evaluate, "_eval_child", _child_dies_silently)
    wd = _FakeWatchdog(killed=True)
    with pytest.raises(RuntimeError) as excinfo:
        evaluate.run_eval("/m", ["gsm8k"], None, 0.85, mp_context="fork",
                          watchdog_factory=lambda pid, hb: wd)
    msg = str(excinfo.value)
    assert "watchdog" in msg.lower()
    assert "stall" in msg.lower()
    assert "ASSAY_STALL_SECONDS" in msg          # names the knob the operator turns
    assert "CUDA" not in msg and "OOM" not in msg  # the wrong cause must be absent
    assert msg.isascii()


def test_dead_child_without_watchdog_kill_keeps_cuda_oom_guidance(monkeypatch):
    """The CUDA/OOM guidance is correct when the watchdog did NOT fire - the fix
    must narrow the message, not replace one misattribution with another."""
    monkeypatch.setattr(evaluate, "_eval_child", _child_dies_silently)
    wd = _FakeWatchdog(killed=False)
    with pytest.raises(RuntimeError, match=r"CUDA error or OOM kill"):
        evaluate.run_eval("/m", ["gsm8k"], None, 0.85, mp_context="fork",
                          watchdog_factory=lambda pid, hb: wd)


def test_dead_child_with_no_watchdog_at_all_keeps_cuda_oom_guidance(monkeypatch):
    """No watchdog wired (unit/dev path): fall back to the original guidance rather
    than crashing on a missing attribute."""
    monkeypatch.setattr(evaluate, "_eval_child", _child_dies_silently)
    with pytest.raises(RuntimeError, match=r"CUDA error or OOM kill"):
        evaluate.run_eval("/m", ["gsm8k"], None, 0.85, mp_context="fork")


# --- BITE 2: per-item capture in the real child (D4/D5) ----------------------

def _stub_lm_eval(monkeypatch, result):
    import sys
    import types
    fake_lm_eval = types.ModuleType("lm_eval")
    fake_lm_eval.simple_evaluate = lambda **k: result(k) if callable(result) else dict(result)
    fake_tasks = types.ModuleType("lm_eval.tasks")
    fake_tasks.TaskManager = object
    fake_tasks.get_task_dict = lambda *a, **k: {}
    monkeypatch.setitem(sys.modules, "lm_eval", fake_lm_eval)
    monkeypatch.setitem(sys.modules, "lm_eval.tasks", fake_tasks)


def _samples_raw(_kwargs=None):
    return {
        "results": {"gsm8k": {"exact_match,none": 0.5}},
        "n-samples": {"gsm8k": {"original": 2, "effective": 2}},
        "samples": {"gsm8k": [
            {"doc_id": 0, "doc_hash": "h0", "filter": "none",
             "metrics": ["exact_match"], "exact_match": 1.0, "resps": ["big" * 1000]},
            {"doc_id": 1, "doc_hash": "h1", "filter": "none",
             "metrics": ["exact_match"], "exact_match": 0.0, "resps": ["big" * 1000]},
        ]},
    }


def test_real_eval_child_capture_reduces_and_deletes_samples(monkeypatch):
    # D5, non-negotiable ordering: the reduced per-item map crosses the Pipe; the
    # raw samples payload (the object that wedged a paid burn for ~6h) never does.
    _stub_lm_eval(monkeypatch, _samples_raw)
    payload = evaluate.run_eval("/m", ["gsm8k"], None, 0.85, mp_context="fork",
                                capture_per_item=True)
    assert "samples" not in payload
    assert payload["assay_per_item"]["gsm8k"]["exact_match,none"] == {
        "0": {"v": 1.0, "h": "h0"}, "1": {"v": 0.0, "h": "h1"}}


def test_real_eval_child_capture_enables_log_samples(monkeypatch):
    _stub_lm_eval(monkeypatch, lambda k: {"results": {"seen": {"v": k.get("log_samples")}}})
    payload = evaluate.run_eval("/m", ["gsm8k"], None, 0.85, mp_context="fork",
                                capture_per_item=True)
    assert payload["results"]["seen"]["v"] is True


def test_real_eval_child_reduction_failure_keeps_eval_and_marks_error(monkeypatch):
    # D4: a reduction bug must NOT crash the child after full paid generation. The
    # eval persists WITHOUT per-item data plus an explicit marker; the gate fails
    # loud later ("paired required, per-item data missing") instead of silently
    # falling back to the unpaired test.
    bad = _samples_raw()
    del bad["samples"]["gsm8k"][0]["metrics"]  # schema drift -> reduce raises
    _stub_lm_eval(monkeypatch, lambda k: bad)
    payload = evaluate.run_eval("/m", ["gsm8k"], None, 0.85, mp_context="fork",
                                capture_per_item=True)
    assert "samples" not in payload
    assert "assay_per_item" not in payload
    assert "metrics" in payload["assay_per_item_error"]
    assert payload["results"]["gsm8k"]["exact_match,none"] == 0.5  # eval survived


def test_real_eval_child_deletes_stray_samples_even_without_capture(monkeypatch):
    # Defense in depth: if the harness ever returns samples despite
    # log_samples=False, they must still never cross the Pipe.
    _stub_lm_eval(monkeypatch, _samples_raw)
    payload = evaluate.run_eval("/m", ["gsm8k"], None, 0.85, mp_context="fork")
    assert "samples" not in payload
    assert "assay_per_item" not in payload


def test_parse_results_collects_items_for_accuracy_tasks_only():
    raw = _samples_raw()
    from assay.pairing import reduce_samples
    raw["assay_per_item"] = reduce_samples(raw)
    del raw["samples"]
    raw["results"]["wikitext"] = {"word_perplexity,none": 10.0}
    out = evaluate.parse_results(raw, (("gsm8k", "exact_match,none"),),
                                 ("wikitext", "word_perplexity"), collect_items=True)
    assert set(out["gsm8k"]["items"]) == {"gsm8k:0", "gsm8k:1"}
    # Perplexity stays unpaired BY DESIGN: deterministic loglikelihood metric under
    # a ratio hard bar, aggregation is weighted perplexity - item-mean pairing
    # does not apply.
    assert "items" not in out["wikitext"]


def test_parse_results_default_shape_unchanged():
    raw = _samples_raw()
    out = evaluate.parse_results(raw, (("gsm8k", "exact_match,none"),), None)
    assert "items" not in out["gsm8k"]
