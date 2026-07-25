from __future__ import annotations

import faulthandler
import json
import math
import multiprocessing
import os
import signal
import time
import traceback

# lm-eval reports metrics with a ",none" (or ",<filter>") suffix on each key,
# e.g. "acc,none". We normalize to bare metric names the gate expects.


def assay_task_dir() -> str:
    """Absolute path to the shipped lm-eval task dir (aime*_avg.yaml + utils.py),
    for lm-eval's TaskManager include_path."""
    return os.path.join(os.path.dirname(__file__), "lm_eval_tasks")


def _sanitize_raw(raw):
    """Make an lm-eval results dict safe to pickle back across the spawn boundary.

    lm-eval embeds live task-config objects in `raw` - notably each task's
    `filter_list[...].filter_fn`. For an assay-owned avg@K task loaded via
    include_path, that filter_fn (utils.passthrough) lives in an EXTERNAL module
    whose __module__ lm-eval sets to the module's absolute PATH - registered only
    in the child interpreter that ran the eval, never in the parent. Sending `raw`
    unmodified over the Pipe makes the parent's unpickle raise
    `ModuleNotFoundError: No module named '/.../lm_eval_tasks/utils'` - the v0.4.1
    avg@K crash, which surfaced only 4.5h into a paid pod because nothing crossed the
    real spawn boundary in tests.

    JSON-normalize with default=str so no callable/opaque reference crosses the
    boundary; the gate reads only scalar metric values, which survive intact. This
    also retires the twin 'Object of type function is not JSON serializable' failure
    the forensics eval-JSON dump hit for the same reason."""
    return json.loads(json.dumps(raw, default=str))


def _mixed_task_specs(tasks, override: dict):
    """The `tasks` list for simple_evaluate: for a task that overrides repeats, its
    pre-built (repeats-applied) Task object; otherwise the bare NAME string so lm-eval
    resolves it itself - crucially group tasks (e.g. mmlu) whose subtasks would be
    dropped if we handed simple_evaluate list(task_dict.values())."""
    return [override.get(t, t) for t in tasks]


def _pick(metrics: dict, name: str) -> float:
    for key, value in metrics.items():
        if key == name or key.startswith(name + ","):
            return float(value)
    raise KeyError(f"metric {name!r} not in {sorted(metrics)}")


def _stderr_key(metric: str) -> str:
    """lm-eval reports a metric's stderr under a sibling key with '_stderr' inserted
    before the filter suffix: 'exact_match,none' -> 'exact_match_stderr,none'; with no
    filter comma, appended: 'word_perplexity' -> 'word_perplexity_stderr'."""
    if "," in metric:
        name, filt = metric.split(",", 1)
        return f"{name}_stderr,{filt}"
    return f"{metric}_stderr"


def _pick_stderr(metrics: dict, metric: str) -> float | None:
    """The stderr for `metric`, or None if lm-eval reported none. Some metrics have no
    stderr, lm-eval emits the string 'N/A' for others, and a matched key may be
    JSON-null - all three normalize to None so the gate can decide (via k_stderr)
    whether a missing stderr actually matters.

    Uses the same comma-anchored prefix search as _pick, NOT a flat dict.get on
    _stderr_key(metric): lm-eval suffixes even a 'bare' metric like word_perplexity
    with a filter, so its stderr lands under 'word_perplexity_stderr,none' - an
    exact-key lookup would miss it and silently drop every recipe's perplexity stderr."""
    stderr_key_name = _stderr_key(metric)
    for key, value in metrics.items():
        if key == stderr_key_name or key.startswith(stderr_key_name + ","):
            if value is None or value == "N/A":
                return None
            fv = float(value)
            # A non-finite stderr (lm-eval's ddof=1 mean_stderr is NaN for n=1, e.g. a
            # limit=1 smoke eval; inf is likewise degenerate) must NOT reach the
            # significance test: combined_se = sqrt(nan) = nan and every comparison
            # against nan is False, silently turning the gate into an unconditional
            # PASS. Normalize to None - the same "missing stderr" gate._delta already
            # refuses to gate on - so a degenerate stderr fails LOUD, not silent.
            # Mirrors _require_finite on the value side.
            if not math.isfinite(fv):
                return None
            return fv
    return None


def _require_finite(value: float, task: str, metric: str) -> float:
    """Reject a non-finite metric. A NaN/inf value is never a legitimate certification
    input: it would silently PASS the gate (every comparison against NaN is False), so
    refuse it here and name the task+metric. This is the operator's 'hold and re-run'
    trigger; it lands in the heartbeat, traceback.txt, and the log-tee."""
    if not math.isfinite(value):
        raise ValueError(
            f"non-finite in {task!r} metric {metric!r}: {value} - eval produced "
            "an invalid number, refusing to gate")
    return value


def _inject_stall_if_configured(env=None, sleep=time.sleep) -> None:
    """TEST-ONLY wedge injection. If ASSAY_INJECT_STALL_AFTER=<seconds> is set (>0),
    block for that many seconds AFTER generation completes - reproducing the observed
    post-inference wedge so a metal Phase-B run can watch the StallWatchdog's kill chain
    fire on a REAL hang. Unset/0/blank/non-numeric = no-op (inert in production). Reads
    os.environ by default; env and sleep are test seams."""
    env = os.environ if env is None else env
    raw = env.get("ASSAY_INJECT_STALL_AFTER", "").strip()
    if not raw:
        return
    try:
        seconds = float(raw)
    except ValueError:
        return
    if seconds > 0:
        sleep(seconds)


def parse_results(raw: dict, accuracy_tasks, perplexity) -> dict:
    """Normalize lm-eval nested output to task -> {"metric": name, "value": v, "stderr": v|None}.

    accuracy_tasks is a tuple of (task, fully-qualified-metric-key) pairs; the
    metric key includes any lm-eval filter suffix (e.g. exact_match,flexible-extract)
    so multi-filter tasks are unambiguous. perplexity is (task, metric) or None."""
    results = raw["results"]
    out: dict[str, dict] = {}
    for task, metric in accuracy_tasks:
        value = _require_finite(_pick(results[task], metric), task, metric)
        out[task] = {"metric": metric, "value": value,
                     "stderr": _pick_stderr(results[task], metric)}
    if perplexity is not None:
        ptask, pmetric = perplexity
        value = _require_finite(_pick(results[ptask], pmetric), ptask, pmetric)
        out[ptask] = {"metric": pmetric, "value": value,
                      "stderr": _pick_stderr(results[ptask], pmetric)}
    return out


def _eval_child(conn, model_path: str, tasks: list[str], gpu_mem_util: float,
                apply_chat_template: bool = False, fewshot_as_multiturn: bool = False,
                gen_kwargs: dict | None = None, system_instruction: str | None = None,
                include_path: str | None = None, repeats: dict | None = None,
                limit: int | None = None, persist_path: str | None = None) -> None:
    """Subprocess body for ONE lm-eval run. Ships ("ok", results) or
    ("err", traceback-string) back over conn. Module-level so the spawn
    context can import it by reference in the child interpreter."""
    # Own process group so the watchdog can SIGKILL this eval tree (child + vLLM
    # EngineCore grandchild) WITHOUT touching group 0 (the shell, log_tee, upload).
    try:
        os.setpgid(0, 0)
    except OSError:
        pass  # already a group leader / not permitted; non-fatal
    # Let the watchdog dump THIS process's stacks before it SIGKILLs us, so a
    # post-inference wedge leaves a stack trace in the forensics, not just silence.
    try:
        faulthandler.register(signal.SIGUSR1)
    except (ValueError, OSError):
        pass
    try:
        from lm_eval import simple_evaluate  # noqa: PLC0415
        from lm_eval.tasks import TaskManager, get_task_dict  # noqa: PLC0415

        task_manager = TaskManager(include_path=include_path) if include_path else None
        override = {}
        if repeats:
            override = get_task_dict([t for t in tasks if t in repeats], task_manager)
            for name, k in repeats.items():
                override[name].set_config("repeats", int(k))
        raw = simple_evaluate(
            model="vllm",
            model_args=(f"pretrained={model_path},dtype=auto,"
                        f"gpu_memory_utilization={gpu_mem_util}"),
            tasks=_mixed_task_specs(tasks, override),
            batch_size="auto",
            apply_chat_template=apply_chat_template,
            fewshot_as_multiturn=fewshot_as_multiturn,
            gen_kwargs=gen_kwargs,
            system_instruction=system_instruction,
            task_manager=task_manager,
            limit=limit,
            # log_samples=False is the PRIME wedge fix: the default True builds the
            # avg@16 CoT samples into a ~16M-token payload that _sanitize_raw
            # serializes twice and pushes across the Pipe (the 4h40m post-inference
            # wedge). The gate reads only scalar metrics, never per-sample text.
            # NOTE: a `bypass` (metric-only) task REQUIRES log_samples=True and lm-eval
            # raises ValueError (evaluator.py) if one is present - none of our recipes
            # use bypass metrics; a future one that does will fail loud right there.
            log_samples=False,
        )
        # TEST-ONLY: reproduce the post-inference wedge here (generation done, result
        # in hand, not yet serialized/persisted) so a metal Phase-B run watches the
        # watchdog kill this child. No-op unless ASSAY_INJECT_STALL_AFTER is set.
        _inject_stall_if_configured()
        clean = _sanitize_raw(raw)
        # Persist the completed result in the CHILD, BEFORE the fragile Pipe send, so
        # a wedge/kill in the serialization tail leaves a recoverable result on disk
        # instead of costing another paid burn (the parent reads it on child death).
        if persist_path:
            try:
                os.makedirs(os.path.dirname(persist_path) or ".", exist_ok=True)
                with open(persist_path, "w", encoding="ascii", errors="replace") as fh:
                    json.dump(clean, fh)
            except OSError:
                pass  # best-effort; the Pipe send below is still the primary channel
        conn.send(("ok", clean))
    except BaseException:
        conn.send(("err", traceback.format_exc()))
        raise  # nonzero exitcode + traceback in the pod log
    finally:
        conn.close()


def run_eval(model_path: str, tasks: list[str], hb=None,
             gpu_mem_util: float = 0.85, mp_context: str = "spawn", *,
             apply_chat_template: bool = False, fewshot_as_multiturn: bool = False,
             gen_kwargs: dict | None = None, system_instruction: str | None = None,
             include_path: str | None = None, repeats: dict | None = None,
             limit: int | None = None, persist_path: str | None = None,
             watchdog_factory=None, join_timeout: float = 120.0) -> dict:
    """Run lm-evaluation-harness on a local model path; return raw results dict.

    Each eval runs in its OWN subprocess (own process group) so vLLM teardown is
    guaranteed by process exit. A watchdog_factory (production only) builds a
    StallWatchdog around the child pid; on a stall the watchdog SIGKILLs the child
    group, which unblocks the recv() below with EOF. persist_path lets a completed-
    but-wedged child leave its result on disk for the parent to recover instead of
    losing a paid eval. mp_context is a test seam: "fork" makes a monkeypatched
    _eval_child visible in the child; production is "spawn" (CUDA does not survive
    fork)."""
    if hb:
        hb.emit("evaluate", f"lm-eval {model_path} tasks={','.join(tasks)} "
                            f"chat={apply_chat_template} gpu_mem_util={gpu_mem_util} (subprocess)")
    if persist_path:
        # A STALE file left by a prior run/eval that reused this path must never be
        # mistaken for THIS eval's result: if this child dies before writing, the
        # dead-child path below must see NO file (-> hard failure), not a leftover
        # result silently returned as if it were fresh. Guarded: no-file-yet is the
        # common case, not an error.
        try:
            os.unlink(persist_path)
        except OSError:
            pass
    ctx = multiprocessing.get_context(mp_context)
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_eval_child,
                       args=(send_conn, model_path, list(tasks), gpu_mem_util,
                             apply_chat_template, fewshot_as_multiturn, gen_kwargs,
                             system_instruction, include_path, repeats, limit,
                             persist_path))
    proc.start()
    send_conn.close()  # parent's copy of the write end; the child holds the live one
    watchdog = None
    try:
        if watchdog_factory is not None:
            watchdog = watchdog_factory(proc.pid, hb)
            watchdog.start()
        try:
            status, payload = recv_conn.recv()
        except EOFError:  # child died before sending anything
            status, payload = "died", None
    finally:
        if watchdog is not None:
            watchdog.stop()
        recv_conn.close()
        _bounded_join(proc, join_timeout)
    if status == "ok":
        if hb:
            hb.emit("evaluate", f"done {model_path}")
        return payload
    if status == "err":
        raise RuntimeError(f"eval subprocess failed for {model_path}:\n{payload}")
    # died: recover a child-persisted result before declaring a hard failure.
    if persist_path:
        recovered = _read_persisted(persist_path)
        if recovered is not None:
            if hb:
                hb.emit("evaluate", f"recovered {model_path} from persisted child result")
            return recovered
    raise RuntimeError(
        f"eval subprocess for {model_path} died without a result "
        f"(exitcode={proc.exitcode}); check the pod log above for a CUDA error or OOM kill"
    )


def _bounded_join(proc, timeout: float) -> None:
    """Join the child, but never hang forever: a child that wedges in teardown AFTER
    delivering a valid result (the postmortem I-2 case) is SIGKILLed by process group
    once the join times out. proc.pid is the group leader (child did setpgid(0,0))."""
    proc.join(timeout=timeout)
    if proc.is_alive():
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        proc.join(timeout=10)


def _read_persisted(path: str):
    """Load a child-persisted results dict, or None if absent/unreadable."""
    try:
        with open(path, encoding="ascii", errors="replace") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None
