from __future__ import annotations

import multiprocessing
import traceback

from assay.config import metric_for

# lm-eval reports metrics with a ",none" (or ",<filter>") suffix on each key,
# e.g. "acc,none". We normalize to bare metric names the gate expects.


def _pick(metrics: dict, name: str) -> float:
    for key, value in metrics.items():
        if key == name or key.startswith(name + ","):
            return float(value)
    raise KeyError(f"metric {name!r} not in {sorted(metrics)}")


def parse_results(raw: dict, accuracy_tasks, perplexity_task) -> dict:
    """Normalize lm-eval nested output to task -> {"metric": name, "value": v}.

    Each task's metric is resolved via config.metric_for -- most accuracy
    tasks report "acc", but gsm8k reports exact_match,strict-match (see
    config.py for rationale). Carrying the resolved metric name alongside
    the value lets the gate label deltas with the real metric per task."""
    results = raw["results"]
    out: dict[str, dict] = {}
    for task in accuracy_tasks:
        m = metric_for(task)
        out[task] = {"metric": m, "value": _pick(results[task], m)}
    out[perplexity_task] = {
        "metric": "word_perplexity",
        "value": _pick(results[perplexity_task], "word_perplexity"),
    }
    return out


def _eval_child(conn, model_path: str, tasks: list[str], gpu_mem_util: float) -> None:
    """Subprocess body for ONE lm-eval run. Ships ("ok", results) or
    ("err", traceback-string) back over conn. Module-level so the spawn
    context can import it by reference in the child interpreter."""
    try:
        from lm_eval import simple_evaluate  # noqa: PLC0415

        raw = simple_evaluate(
            model="vllm",
            model_args=(f"pretrained={model_path},dtype=auto,"
                        f"gpu_memory_utilization={gpu_mem_util}"),
            tasks=tasks,
            batch_size="auto",
        )
        conn.send(("ok", raw))
    except BaseException:
        conn.send(("err", traceback.format_exc()))
        raise  # nonzero exitcode + traceback in the pod log
    finally:
        conn.close()


def run_eval(model_path: str, tasks: list[str], hb=None,
             gpu_mem_util: float = 0.85, mp_context: str = "spawn") -> dict:
    """Run lm-evaluation-harness on a local model path; return raw results dict.

    Each eval runs in its OWN subprocess so vLLM teardown is guaranteed by
    process exit. run_job calls this twice back-to-back; lm-eval owns the vLLM
    LLM inside simple_evaluate (we never get a handle) and its wrapper has no
    shutdown path for the non-ray case, so in-process the first engine's GPU
    memory is only released if a cycle-heavy object graph happens to be GC'd
    -- and a still-alive first engine holds ~all of the GPU, failing the second
    engine's startup free-memory check. Process exit returns every byte to the
    driver unconditionally, including the engine's own EngineCore child.

    mp_context: "spawn" in production -- this process already initialized CUDA
    (GPU preflight + quantize) and CUDA does not survive fork. It is a test
    seam: unit tests pass "fork" so a monkeypatched _eval_child is visible in
    the child without importing lm_eval or a fresh interpreter."""
    if hb:
        hb.emit("evaluate", f"lm-eval {model_path} tasks={','.join(tasks)} "
                            f"gpu_mem_util={gpu_mem_util} (subprocess)")
    ctx = multiprocessing.get_context(mp_context)
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_eval_child,
                       args=(send_conn, model_path, list(tasks), gpu_mem_util))
    proc.start()
    send_conn.close()  # parent's copy of the write end; the child holds the live one
    try:
        try:
            status, payload = recv_conn.recv()
        except EOFError:  # child died before sending anything
            status, payload = "died", None
    finally:
        recv_conn.close()
        proc.join()  # no timeout, deliberately: a slow engine shutdown is not a failure
    if status == "ok":
        if hb:
            hb.emit("evaluate", f"done {model_path}")
        return payload
    if status == "err":
        raise RuntimeError(f"eval subprocess failed for {model_path}:\n{payload}")
    raise RuntimeError(
        f"eval subprocess for {model_path} died without a result "
        f"(exitcode={proc.exitcode}); check the pod log above for a CUDA error or OOM kill"
    )
