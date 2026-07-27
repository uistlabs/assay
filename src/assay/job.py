from __future__ import annotations

import json
import os
import shutil
import traceback
from typing import Callable, NamedTuple

from assay import __version__
from assay.config import RunConfig, TIER_PROFILES, require_secret
from assay.gate import GateResult, render_delta_table
from assay.heartbeat import Heartbeat

# The run-completion markers main() prints as its FINAL act. pod_entry.sh greps the
# ephemeral log for one of these to prove the job actually ran work (a marker-less
# exit-0 means a no-op that would otherwise self-terminate the pod and silently burn a
# rented GPU). This is a TWO-SIDED string contract: job.py prints it, pod_entry.sh
# matches it. They are single-sourced here and cross-checked by
# tests/test_pod_entry.py::test_gate_marker_contract so a reword on either side breaks
# CI instead of a metal run. Keep the literals in sync with pod_entry.sh's grep pattern.
GATE_PASSED_MARKER = "GATE PASSED"
GATE_FAILED_MARKER = "GATE FAILED"


class Deps(NamedTuple):
    quantize: Callable
    run_eval: Callable
    parse: Callable
    gate: Callable
    publish: Callable


def run_identity_lines(runcfg, env) -> list[str]:
    """Non-secret identity of THIS run, so a stale/wrong image is obvious in minute one.
    Pure (no I/O) for testability; print_run_identity does the emitting."""
    recipe = runcfg.recipe
    gate = recipe.gate_or_default
    active = [n for n, v in (("mean_retention", gate.min_mean_retention),
                             ("k_stderr", gate.k_stderr),
                             ("single_drop_pts", gate.max_single_drop_pts),
                             ("ppl_increase", gate.max_ppl_increase)) if v is not None]
    return [
        f"assay v{__version__} build={env.get('ASSAY_BUILD_SHA', 'unknown')}",
        f"recipe={recipe.slug} base={recipe.base_model} scheme={recipe.quant_scheme}",
        "accuracy_tasks=" + ", ".join(f"{t}:{m}" for t, m in recipe.eval.accuracy_tasks),
        f"perplexity={recipe.eval.perplexity} gate_active={active}",
        # Effective run TIER, derived from what the pod will actually use - so a stale
        # ASSAY_TIER (turns the blessed burn into a limit=2/max_gen_toks=8 no-op) or a
        # forgotten ASSAY_NUM_CALIB (silently under-calibrates a PUBLISHED checkpoint) is
        # obvious in the first-seconds log, not discovered after the spend. publish= is
        # derived from runcfg.pristine (the actual chokepoint), so a cert-tier run with a
        # lingering override still shows DRY-RUN here instead of only at publish time.
        f"tier={runcfg.tier} publish={'live' if runcfg.pristine else 'DRY-RUN'} "
        f"calib_samples={recipe.calib.num_samples}",
    ]


def print_run_identity(runcfg, env, hb=None) -> None:
    for line in run_identity_lines(runcfg, env):
        print(f"[identity] {line}", flush=True)
        if hb is not None:
            hb.emit("identity", line)


def default_deps(env) -> Deps:  # pragma: no cover - wires real GPU/network deps
    """Wire real implementations. Imported lazily so unit tests never touch GPU deps."""
    from assay import evaluate, publish as publish_mod, quantize

    return Deps(
        quantize=quantize.quantize_to_nvfp4,
        run_eval=evaluate.run_eval,
        parse=evaluate.parse_results,
        gate=None,  # bound in run_job (needs config tasks)
        # publish_if_passed needs the HF token as its 4th positional arg;
        # run_job calls deps.publish(runcfg, out_dir, result, hb) - 4 args, no
        # token - so bind the token from env here rather than wiring the
        # bare function directly.
        publish=lambda runcfg, out, result, hb: publish_mod.publish_if_passed(
            runcfg, out, result, require_secret(env, "HF_TOKEN"), hb,
            dry_run=not runcfg.pristine
        ),
    )


def _resource_snapshot() -> str:
    """One-line root-disk + memory snapshot for post-mortem triage: distinguishes
    ENOSPC (small container disk) from OOM from other failures without a live pod."""
    try:
        du = shutil.disk_usage("/")
        disk = f"root_disk free={du.free // 2**20}MiB/{du.total // 2**20}MiB"
    except Exception as exc:  # pragma: no cover - host-dependent
        disk = f"disk_usage err: {exc}"
    try:
        with open("/proc/meminfo") as fh:
            mem = {ln.split(":", 1)[0]: ln.split(":", 1)[1].strip() for ln in fh if ":" in ln}
        memline = f"MemAvailable={mem.get('MemAvailable', '?')} MemTotal={mem.get('MemTotal', '?')}"
    except Exception as exc:  # pragma: no cover - host-dependent
        memline = f"meminfo err: {exc}"
    return f"{disk} | {memline}"


def _write_artifact(path: str, render: Callable[[], str], hb: "Heartbeat | None") -> None:
    """Best-effort durable-artifact write (I1/I2). `render` is called lazily inside
    the try so a serialization failure is caught the same as an I/O failure - an
    artifact write must NEVER abort the run or skip teardown, just log and move on.
    Tolerates hb=None so it can run from the failure path before the heartbeat exists."""
    try:
        content = render()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="ascii", errors="replace") as fh:
            fh.write(content)
    except Exception as exc:
        if hb is not None:
            hb.emit("artifact", f"write failed: {exc}")


def run_job(runcfg: RunConfig, env, deps: Deps) -> GateResult:
    """Orchestrate quantize -> eval(baseline+nvfp4) -> gate -> publish, always tearing
    the pod down in a finally. Control-flow only; stages are injected via `deps`.

    Durable artifacts (I1/I2): the raw eval JSONs and the rendered delta table are
    written to runcfg.artifacts_dir as they become available - NOT runcfg.output_dir,
    which is the published checkpoint dir. A gate FAIL still leaves the numbers on
    disk (I1), and the heartbeat/eval/delta files never ride along into the public
    HF repo (I2) because artifacts_dir sits outside output_dir entirely."""
    from assay.gate import evaluate_gate
    from assay.watchdog import build_eval_watchdog
    raw_log_path = env.get("ASSAY_RAW_LOG")
    def _wd_factory(child_pid, heartbeat):
        return build_eval_watchdog(child_pid, raw_log_path, heartbeat, env)

    recipe = runcfg.recipe
    all_task_names = list(recipe.accuracy_task_names) + (
        [recipe.perplexity_task_name] if recipe.perplexity_task_name else [])
    ev = recipe.eval
    from assay.evaluate import assay_task_dir
    # Baseline and quantized MUST be evaluated under identical chat-mode/sampling
    # settings - eval_kwargs is built once and passed to both run_eval calls
    # below so a gate delta reflects the quantization, not a settings drift.
    # Pairing keys off the GATE, not the tier (D7/F-025): per-item capture and
    # collection run exactly when the recipe's gate consults an SE. A point-gated
    # recipe (Qwen) keeps log_samples=False and gains no new hard-fail surface for
    # a significance test its gate never runs.
    significance = recipe.gate_or_default.k_stderr is not None
    eval_kwargs = dict(
        apply_chat_template=(ev.mode == "chat"),
        fewshot_as_multiturn=(ev.mode == "chat"),
        gen_kwargs=ev.gen_kwargs,
        system_instruction=ev.system_prompt,
        include_path=assay_task_dir(),
        repeats=ev.repeats,
        capture_per_item=significance,
    )
    # Tier-driven eval-weakening (TIER_PROFILES, single source): cert leaves the
    # certified methodology untouched (every profile field None); dev/smoke cap
    # limit/repeats/max_gen_toks. limit itself is never 1 - refused at load
    # (config._resolve_eval_limit) because lm-eval computes a metric's stderr only
    # for n>1, and n=1 turns the significance gate into an unconditional PASS.
    profile = TIER_PROFILES[runcfg.tier]
    limit = runcfg.eval_limit
    if profile.repeats_cap is not None:
        # Cap avg@K per task; NEVER raise a task above its recipe K (min), and leave
        # single-sampled tasks (no repeats entry) untouched.
        eval_kwargs["repeats"] = {t: min(k, profile.repeats_cap)
                                  for t, k in ev.repeats.items()}
    if profile.max_gen_toks is not None:
        # Cap generation length identically on both eval sides; never exceed the
        # recipe's own max_gen_toks (min).
        gk = dict(ev.gen_kwargs or {})
        recipe_mgt = gk.get("max_gen_toks")
        gk["max_gen_toks"] = (min(recipe_mgt, profile.max_gen_toks)
                              if recipe_mgt else profile.max_gen_toks)
        eval_kwargs["gen_kwargs"] = gk
    hb = None
    try:
        hb = Heartbeat(runcfg.heartbeat_path,
                       secrets=[env.get("HF_TOKEN", ""), env.get("RUNPOD_API_KEY", "")])
        hb.emit("start", recipe.base_model)
        nvfp4_dir = deps.quantize(recipe, runcfg.weights_path, runcfg.output_dir, hb)

        base_raw = deps.run_eval(
            runcfg.weights_path, all_task_names, hb, runcfg.gpu_mem_util,
            limit=limit,
            persist_path=os.path.join(runcfg.artifacts_dir, "eval-baseline-child.json"),
            watchdog_factory=_wd_factory,
            **eval_kwargs)
        _write_artifact(
            os.path.join(runcfg.artifacts_dir, "eval-baseline.json"),
            # default=str: lm-eval's raw result dict carries a non-JSON-serializable
            # object (a filter callable in the per-task config) - observed on metal
            # as "Object of type function is not JSON serializable", which
            # silently dropped the I1 raw-eval forensics. Stringify the unknowns so the
            # numbers still persist; the gate reads parsed floats, not this file.
            lambda: json.dumps(base_raw, default=str), hb,
        )
        quant_raw = deps.run_eval(
            nvfp4_dir, all_task_names, hb, runcfg.gpu_mem_util,
            limit=limit,
            persist_path=os.path.join(runcfg.artifacts_dir, "eval-nvfp4-child.json"),
            watchdog_factory=_wd_factory,
            **eval_kwargs)
        _write_artifact(
            os.path.join(runcfg.artifacts_dir, "eval-nvfp4.json"),
            lambda: json.dumps(quant_raw, default=str), hb,
        )
        base = deps.parse(base_raw, ev.accuracy_tasks, ev.perplexity,
                          collect_items=significance)
        quant = deps.parse(quant_raw, ev.accuracy_tasks, ev.perplexity,
                           collect_items=significance)

        gate_fn = deps.gate or (lambda b, q, a, p, t: evaluate_gate(b, q, a, p, t))
        result = gate_fn(base, quant, recipe.accuracy_task_names,
                         recipe.perplexity_task_name, recipe.gate_or_default)
        hb.emit("gate", "PASSED" if result.passed else "FAILED")
        _write_artifact(
            os.path.join(runcfg.artifacts_dir, "delta-table.md"),
            lambda: render_delta_table(result, recipe.gate_or_default), hb,
        )

        deps.publish(runcfg, nvfp4_dir, result, hb)
        return result
    except BaseException as exc:
        # Durable forensics. A rented, self-terminating pod can fail exactly once and
        # then delete itself + its logs - so capture the failing stage's traceback and
        # a disk/mem snapshot to the volume BEFORE teardown, converting any recurrence
        # into a read instead of another debugging rental. Routed through the
        # heartbeat's redaction so a token can never leak into the artifact. hb may be
        # None if Heartbeat construction itself failed; _write_artifact tolerates that.
        snapshot = _resource_snapshot()
        if hb is not None:
            try:
                hb.emit("error", f"{type(exc).__name__}: {exc}")
                hb.emit("resource", snapshot)
            except Exception:
                pass  # forensics must never mask the original failure
        redact = hb.redact if hb is not None else (lambda s: s)
        _write_artifact(
            os.path.join(runcfg.artifacts_dir, "traceback.txt"),
            lambda: redact(traceback.format_exc()) + "\n\n" + snapshot + "\n",
            hb,
        )
        raise
    finally:
        if hb is not None:
            try:
                hb.emit("teardown", "job complete - pod_entry will self-terminate")
            except Exception:
                pass  # a heartbeat failure must never mask the result


def assert_gpu_available() -> None:
    """Refuse to run without a usable CUDA GPU. llm-compressor downgrades a dead
    GPU to a WARNING and grinds on CPU (observed on metal: the nvidia/cuda base's
    cuda-compat raised CUDA error 804 on a consumer 5090 -> 35 min of paid CPU
    calibration on a rented pod). Fail loud in the first seconds instead.

    torch.cuda.is_available()/device_count() SWALLOW the init error - that is exactly
    how the silent CPU fallback happened. torch.cuda.init() RE-RAISES the real CUDA
    error string (e.g. 'Error 804: forward compatibility...') so the log names the
    cause. The probe kernel launch + copy-back also catches an arch/wheel mismatch
    (device enumerates but the wheel ships no kernel for this SM)."""
    import torch
    try:
        torch.cuda.init()
        count = torch.cuda.device_count()
        if count == 0:
            raise RuntimeError("CUDA initialized but zero devices visible")
        probe = (torch.ones(8, device="cuda") * 2).sum().item()
        if probe != 16.0:
            raise RuntimeError(f"GPU probe kernel returned {probe}, expected 16.0")
    except Exception as exc:
        raise SystemExit(
            f"FATAL: no usable CUDA GPU ({type(exc).__name__}: {exc}). "
            "Refusing to fall back to CPU."
        ) from exc
    print(f"GPU preflight OK: {torch.cuda.get_device_name(0)} x{count}", flush=True)


def main() -> None:  # pragma: no cover - pod entrypoint
    from assay.config import load_config, resolve_mount
    from assay import smoke, verify
    # Cheap, torch-free config validation FIRST (microseconds, catches a misconfig
    # before importing torch), THEN fail loud on a dead GPU - both in the first
    # second, before any quantize/eval/network work. Never silently CPU-grind
    # (see the CUDA-804 note in the Dockerfile / runpod_ctl).
    cfg = resolve_mount(load_config(os.environ))
    print_run_identity(cfg, os.environ)
    # In-pod tier-1: catches a stale/wrong/mis-built image in the first seconds (CPU,
    # ~2s) - the runtime complement to the launch-side digest pin.
    smoke.tier1_structural()
    # Weights identity (F-015), deliberately HERE and not in run_job - run_job's
    # contract is control-flow only and its tests inject fake paths. Cheap small-file
    # pass before the GPU preflight; the minutes-long shard pass (pristine runs only)
    # after it, so a dead GPU fails first.
    verify.verify_weights_small(cfg, os.environ)
    assert_gpu_available()
    verify.verify_weights_shards(cfg, os.environ)
    deps = default_deps(os.environ)
    result = run_job(cfg, os.environ, deps)
    print(GATE_PASSED_MARKER if result.passed else GATE_FAILED_MARKER)


if __name__ == "__main__":  # pragma: no cover - exercised by pod_entry.sh / -m
    main()
