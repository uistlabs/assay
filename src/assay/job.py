from __future__ import annotations

import json
import os
import shutil
import traceback
from typing import Callable, NamedTuple

from assay.config import Config, require_secret
from assay.gate import GateResult, render_delta_table
from assay.heartbeat import Heartbeat


class Deps(NamedTuple):
    quantize: Callable
    run_eval: Callable
    parse: Callable
    gate: Callable
    publish: Callable
    terminate: Callable


def default_deps(env) -> Deps:  # pragma: no cover -- wires real GPU/network deps
    """Wire real implementations. Imported lazily so unit tests never touch GPU deps."""
    from assay import evaluate, publish as publish_mod, quantize, runpod_ctl

    return Deps(
        quantize=quantize.quantize_to_nvfp4,
        run_eval=evaluate.run_eval,
        parse=evaluate.parse_results,
        gate=None,  # bound in run_job (needs config tasks)
        # publish_if_passed needs the HF token as its 4th positional arg;
        # run_job calls deps.publish(cfg, out_dir, result, hb) -- 4 args, no
        # token -- so bind the token from env here rather than wiring the
        # bare function directly.
        publish=lambda cfg, out, result, hb: publish_mod.publish_if_passed(
            cfg, out, result, require_secret(env, "HF_TOKEN"), hb
        ),
        terminate=lambda: runpod_ctl.self_terminate(env),
    )


def _resource_snapshot() -> str:
    """One-line root-disk + memory snapshot for post-mortem triage: distinguishes
    ENOSPC (small container disk) from OOM from other failures without a live pod."""
    try:
        du = shutil.disk_usage("/")
        disk = f"root_disk free={du.free // 2**20}MiB/{du.total // 2**20}MiB"
    except Exception as exc:  # pragma: no cover -- host-dependent
        disk = f"disk_usage err: {exc}"
    try:
        with open("/proc/meminfo") as fh:
            mem = {ln.split(":", 1)[0]: ln.split(":", 1)[1].strip() for ln in fh if ":" in ln}
        memline = f"MemAvailable={mem.get('MemAvailable', '?')} MemTotal={mem.get('MemTotal', '?')}"
    except Exception as exc:  # pragma: no cover -- host-dependent
        memline = f"meminfo err: {exc}"
    return f"{disk} | {memline}"


def _write_artifact(path: str, render: Callable[[], str], hb: "Heartbeat | None") -> None:
    """Best-effort durable-artifact write (I1/I2). `render` is called lazily inside
    the try so a serialization failure is caught the same as an I/O failure -- an
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


def run_job(cfg: Config, env, deps: Deps) -> GateResult:
    """Orchestrate quantize -> eval(baseline+nvfp4) -> gate -> publish, always tearing
    the pod down in a finally. Control-flow only; stages are injected via `deps`.

    Durable artifacts (I1/I2): the raw eval JSONs and the rendered delta table are
    written to cfg.artifacts_dir as they become available -- NOT cfg.output_dir,
    which is the published checkpoint dir. A gate FAIL still leaves the numbers on
    disk (I1), and the heartbeat/eval/delta files never ride along into the public
    HF repo (I2) because artifacts_dir sits outside output_dir entirely."""
    from assay.gate import evaluate_gate

    all_tasks = list(cfg.accuracy_tasks) + [cfg.perplexity_task]
    hb = None
    try:
        hb = Heartbeat(cfg.heartbeat_path,
                       secrets=[env.get("HF_TOKEN", ""), env.get("RUNPOD_API_KEY", "")])
        hb.emit("start", cfg.base_model)
        nvfp4_dir = deps.quantize(cfg, cfg.weights_path, cfg.output_dir, hb)

        base_raw = deps.run_eval(cfg.weights_path, all_tasks, hb, cfg.gpu_mem_util)
        _write_artifact(
            os.path.join(cfg.artifacts_dir, "eval-baseline.json"),
            # default=str: lm-eval's raw result dict carries a non-JSON-serializable
            # object (a filter callable in the per-task config) -- observed on metal
            # as "Object of type function is not JSON serializable", which
            # silently dropped the I1 raw-eval forensics. Stringify the unknowns so the
            # numbers still persist; the gate reads parsed floats, not this file.
            lambda: json.dumps(base_raw, default=str), hb,
        )
        quant_raw = deps.run_eval(nvfp4_dir, all_tasks, hb, cfg.gpu_mem_util)
        _write_artifact(
            os.path.join(cfg.artifacts_dir, "eval-nvfp4.json"),
            lambda: json.dumps(quant_raw, default=str), hb,
        )
        base = deps.parse(base_raw, cfg.accuracy_tasks, cfg.perplexity_task)
        quant = deps.parse(quant_raw, cfg.accuracy_tasks, cfg.perplexity_task)

        gate_fn = deps.gate or (lambda b, q, a, p: evaluate_gate(b, q, a, p))
        result = gate_fn(base, quant, cfg.accuracy_tasks, cfg.perplexity_task)
        hb.emit("gate", "PASSED" if result.passed else "FAILED")
        _write_artifact(
            os.path.join(cfg.artifacts_dir, "delta-table.md"),
            lambda: render_delta_table(result), hb,
        )

        deps.publish(cfg, nvfp4_dir, result, hb)
        return result
    except BaseException as exc:
        # Durable forensics. A rented, self-terminating pod can fail exactly once and
        # then delete itself + its logs -- so capture the failing stage's traceback and
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
            os.path.join(cfg.artifacts_dir, "traceback.txt"),
            lambda: redact(traceback.format_exc()) + "\n\n" + snapshot + "\n",
            hb,
        )
        raise
    finally:
        if hb is not None:
            try:
                hb.emit("teardown", "self-terminating pod")
            except Exception:
                pass  # a heartbeat failure must NEVER prevent teardown
        deps.terminate()


def assert_gpu_available() -> None:
    """Refuse to run without a usable CUDA GPU. llm-compressor downgrades a dead
    GPU to a WARNING and grinds on CPU (observed on metal: the nvidia/cuda base's
    cuda-compat raised CUDA error 804 on a consumer 5090 -> 35 min of paid CPU
    calibration on a rented pod). Fail loud in the first seconds instead.

    torch.cuda.is_available()/device_count() SWALLOW the init error -- that is exactly
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


def main() -> None:  # pragma: no cover -- pod entrypoint
    from assay.config import load_config, resolve_mount
    # Cheap, torch-free config validation FIRST (microseconds, catches a misconfig
    # before importing torch), THEN fail loud on a dead GPU -- both in the first
    # second, before any quantize/eval/network work. Never silently CPU-grind
    # (see the CUDA-804 note in the Dockerfile / runpod_ctl).
    cfg = resolve_mount(load_config(os.environ))
    assert_gpu_available()
    deps = default_deps(os.environ)
    result = run_job(cfg, os.environ, deps)
    print("GATE PASSED" if result.passed else "GATE FAILED")


if __name__ == "__main__":  # pragma: no cover -- exercised by pod_entry.sh / -m
    main()
